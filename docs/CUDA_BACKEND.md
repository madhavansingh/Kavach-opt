# Kavach-Opt CUDA Execution Backend Documentation

> **Architectural Principle**: Kavach-Opt is a **CPU-first solver with one scoped, measured GPU-accelerated kernel**. Kavach-Opt is never described as a generic "GPU solver".

---

## 1. Why Batched SpMV Was Selected

In mathematical optimization, the majority of algorithmic steps are latency-sensitive, branch-heavy, and inherently sequential. However, evaluating multiple independent right-hand sides across candidate solutions, dual residual verification passes, sensitivity analysis perturbations, and scenario-based constraint evaluations naturally yield batches of independent matrix-vector products:
$$Y_k = A X_k \quad \text{for } k = 0, \dots, \text{batch\_size} - 1$$

Batched Sparse Matrix-Vector Multiplication (SpMV) possesses:
- **Massive Data Parallelism**: Independent evaluations across both batch items $k$ and matrix rows $r$.
- **High Memory Bandwidth Demand**: Streaming sparse nonzeros and dense vectors benefits directly from modern GPU global memory bandwidth ($> 500$ GB/s on desktop GPUs).
- **Throughput-Oriented Structure**: Amortizes kernel launch overhead when problem size and batch count exceed the crossover threshold.

---

## 2. Why Simplex Pivoting Is Not Moved to GPU

Simplex pivoting (both Primal and Dual Revised Simplex) is **strictly retained on the CPU**:
1. **Inherently Sequential Dependency Chain**: Each simplex iteration depends strictly on the basis inverse update and basis selection of the preceding pivot. There is zero algorithmic parallelism across successive simplex iterations.
2. **Branch-Heavy, Irregular Logic**: Simplex pivoting is dominated by sparse ratio tests, Bland's tie-breaking rule, degeneracy monitoring, and pricing loops with tiny active vectors. GPUs exhibit severe branch divergence and poor warp occupancy on such logic.
3. **PCIe Transfer Latency Penalty**: Moving small basic vectors back and forth between CPU and GPU on every pivot iteration would incur hundreds of microseconds of PCIe latency per pivot, resulting in an order-of-magnitude net slowdown.

---

## 3. Why Branch-and-Bound Traversal Is Not Moved to GPU

Branch-and-bound tree exploration is **strictly retained on the CPU**:
1. **Irregular, Asynchronous Tree Dynamics**: Node selection (best-bound, depth-first), dynamic branching variable selection, cutting plane additions, and heuristic incumbent updates form an irregular, tree-structured graph.
2. **Synchronization & Divergence**: Managing a concurrent priority queue or dynamic branch stack on the GPU introduces extreme warp divergence and lock contention.
3. **SparseLU Basis State**: Each node solve requires maintaining or updating LU basis factors, which requires complex, irregular pointer-chasing data structures ill-suited for GPU streaming architectures.

---

## 4. CUDA Backend Architecture

The CUDA backend is implemented in `src/backend/cuda/`:
- `cuda_device.hpp` & `cuda_device.cpp`: Device enumeration, property queries (SM count, VRAM, compute capability), runtime initialization, and error reporting.
- `cuda_spmv.cuh` & `cuda_spmv.cu`: Sovereign, custom double-precision batched CSR SpMV kernel with RAII buffer management and asynchronous streams.
- `cuda_backend.hpp` & `cuda_backend.cpp`: Implementation of `ExecutionBackend` providing transparent CPU fallback, `--no-gpu` support, and transfer-inclusive telemetry.

```
                   ExecutionBackend (Abstract Interface)
                                  │
       ┌──────────────────────────┼──────────────────────────┐
       ▼                          ▼                          ▼
CpuSequentialBackend     CpuMulticoreBackend            CudaBackend
(Single-thread CPU)      (Multi-thread ThreadPool)      (Scoped GPU Batched-SpMV)
                                                             │
                                                ┌────────────┴────────────┐
                                                ▼                         ▼
                                         NVIDIA GPU Kernel         CPU Fallback
                                         (cuda_spmv.cu)       (CpuMulticoreBackend)
```

---

## 5. Kernel Mapping & Work Distribution

The custom kernel `batched_csr_spmv_kernel` maps total work items $N_{\text{total}} = \text{batch\_size} \times \text{rows}$:
- **Block Dimension**: 256 threads (`blockDim.x = 256`).
- **Grid Dimension**: $\lceil N_{\text{total}} / 256 \rceil$ blocks (`gridDim.x`).
- **Index Decomposition**:
  $$\text{global\_idx} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$
  $$\text{batch\_index } k = \lfloor \text{global\_idx} / \text{rows} \rfloor, \quad \text{row\_index } r = \text{global\_idx} \pmod{\text{rows}}$$

```cuda
__global__ void batched_csr_spmv_kernel(
    std::size_t batch_size, std::size_t rows, std::size_t cols,
    const double* __restrict__ values,
    const std::size_t* __restrict__ row_ptr,
    const std::size_t* __restrict__ col_idx,
    const double* __restrict__ batch_x,
    double* __restrict__ batch_y)
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= batch_size * rows) return;

    const std::size_t k = idx / rows;
    const std::size_t r = idx % rows;

    const std::size_t start = row_ptr[r];
    const std::size_t end = row_ptr[r + 1];
    const double* x_k = batch_x + (k * cols);

    double sum = 0.0;
    for (std::size_t j = start; j < end; ++j) {
        sum += values[j] * x_k[col_idx[j]];
    }
    batch_y[k * rows + r] = sum;
}
```

### Guarantees
- **Bounds-Safe**: Strict guard `if (idx >= batch_size * rows) return;`.
- **Zero Race Conditions & Zero Atomics**: Each thread computes and writes to exactly one unique element in $Y$.
- **Deterministic Summation**: Nonzeros within each row are accumulated in strictly ascending column order ($j$ from `start` to `end - 1`), mirroring CPU execution.

---

## 6. Device Memory Layout & Host/Device Transfers

To prevent memory fragmentation and eliminate repeated allocation overhead:
- **Matrix Representation**: Stored in standard 3-array CSR format on device:
  `d_values` ($\text{nnz} \times 8$ bytes), `d_row_ptr` ($(m + 1) \times 8$ bytes), `d_col_idx` ($\text{nnz} \times 8$ bytes).
- **Batched Vectors**: Represented as contiguous, flattened host and device buffers:
  $$X = [X_0 \mid X_1 \mid \dots \mid X_{B-1}] \in \mathbb{R}^{B \cdot n}, \quad Y = [Y_0 \mid Y_1 \mid \dots \mid Y_{B-1}] \in \mathbb{R}^{B \cdot m}$$
- **RAII Buffers**: `CudaDeviceBufferRAII<T>` guarantees that device memory is allocated with `cudaMalloc` and automatically freed via `cudaFree` upon destruction, even if an exception occurs during execution.

---

## 7. Synchronization & Stream Design

- **Asynchronous Execution Pipeline**: Uses dedicated `cudaStream_t` wrapped in `CudaStreamRAII`.
- **Pipeline Stages**:
  1. Asynchronous Host-to-Device (H2D) transfer: `cudaMemcpyAsync(..., cudaMemcpyHostToDevice, stream)`.
  2. Asynchronous Kernel launch on `stream`.
  3. Kernel launch verification: `cudaGetLastError()`.
  4. Asynchronous Device-to-Host (D2H) transfer: `cudaMemcpyAsync(..., cudaMemcpyDeviceToHost, stream)`.
  5. Explicit barrier: `cudaStreamSynchronize(stream)` before accessing host output buffers.

---

## 8. Numerical Precision & Correctness Contract

- **Double-Precision Standard**: All computation is performed in IEEE 754 64-bit double precision (`double`), strictly matching Kavach-Opt's CPU numerical layer. Single-precision (`float`) is never silently substituted.
- **Reference Contract**: The CPU sequential implementation is the ground truth. Every test verifies:
  $$\max_{k, r} |Y_k^{\text{GPU}}[r] - Y_k^{\text{CPU}}[r]| \le \epsilon_{\text{abs}} = 10^{-12}$$
  $$\max_{k, r} \frac{|Y_k^{\text{GPU}}[r] - Y_k^{\text{CPU}}[r]|}{\max(1.0, |Y_k^{\text{CPU}}[r]|)} \le \epsilon_{\text{rel}} = 10^{-10}$$

---

## 9. CPU Fallback & Runtime Selection Policy

CPU fallback is **mandatory, default, and comprehensive**:
1. **Build-Time Isolation**: When built with `KAVACH_ENABLE_CUDA=OFF` (or on platforms without CUDA like macOS Apple Silicon), `CudaBackend` compiles as clean C++20 with zero CUDA header requirements.
2. **Runtime Detection**: If no CUDA-capable device is detected at runtime, `is_available()` returns `false`, and `CudaBackend` automatically routes computation to `CpuMulticoreBackend`, populating telemetry with `fallback_occurred = true` and `fallback_reason`.
3. **`--no-gpu` Support**: Programmatic flag `options.force_cpu_fallback = true` forces execution on the CPU even if a GPU is physically installed.
4. **Strict GPU Mode**: If `options.strict_gpu = true` is explicitly requested and no GPU is available, the solver throws `core::NumericalException` rather than silently degrading.

---

## 10. Transfer-Inclusive Timing & Crossover Methodology

### Hard Accounting Rule
> **Never claim GPU speedup using kernel-only timing.**

All benchmark speedups must be calculated using the **transfer-inclusive wall time**:
$$T_{\text{GPU, total}} = T_{\text{H2D}} + T_{\text{Kernel}} + T_{\text{D2H}}$$
$$\text{Speedup}_{\text{transfer-inclusive}} = \frac{T_{\text{CPU}}}{T_{\text{GPU, total}}}$$

The kernel-only speedup ($\frac{T_{\text{CPU}}}{T_{\text{Kernel}}}$) is recorded strictly as a diagnostic metric to quantify PCIe transfer bottlenecks.

### Crossover Analysis
The crossover point is defined as the tuple $(m, n, \text{nnz}, \text{batch\_size})$ below which:
$$\text{Speedup}_{\text{transfer-inclusive}} < 1.0 \quad (\text{GPU is slower than CPU due to transfer overhead})$$
and above which:
$$\text{Speedup}_{\text{transfer-inclusive}} > 1.0 \quad (\text{GPU compute throughput exceeds transfer costs})$$

- On small instances (e.g. single vectors or batch sizes $< 32$), PCIe round-trip latency ($\approx 10\text{–}50\,\mu\text{s}$) dominates compute time, and CPU execution is significantly faster.
- On large instances ($m \ge 500$, batch size $\ge 128$), GPU throughput overcomes transfer overhead.

---

## 11. Empirical Verification & Test Suite Matrix

The CUDA backend is verified by `tests/unit/test_phase11_cuda.cpp` covering 24 test cases (all passing 100%):

| Test Identifier | Category | Verified Property | Result |
| :--- | :--- | :--- | :--- |
| **Test A** | Device Detection | Clean detection of device count and status string | **PASS** |
| **Test B** | No-Device Behavior | Graceful availability report without crashes | **PASS** |
| **Test C** | Invalid Device Index | Out-of-bounds device indices throw typed `InputException` | **PASS** |
| **Test D** | CSR Representation | CSR format valid and matches matrix invariants | **PASS** |
| **Test E** | Basic SpMV | Single vector SpMV matches CPU reference | **PASS** |
| **Test F** | Batched SpMV | 32-vector batched SpMV matches CPU reference | **PASS** |
| **Test G** | Numerical Equivalence | Absolute error $< 10^{-14}$, relative error $< 10^{-14}$ | **PASS** |
| **Test H** | Irregular Sparsity | Empty rows, dense rows, and singletons handled safely | **PASS** |
| **Test I** | Empty Rows | Rows with zero nonzeros evaluate to strictly 0.0 | **PASS** |
| **Test J** | Zero Vectors | Zero input vector batch produces zero output | **PASS** |
| **Test K** | Mixed Signs | Negative and positive entries produce exact numerical match | **PASS** |
| **Test L** | Batch Size 1 | Single-element batch executes without block underflow | **PASS** |
| **Test M** | Small Batch | Batch size 2 executes cleanly | **PASS** |
| **Test N** | Large Batch | Batch size 128 executes cleanly with zero error | **PASS** |
| **Test O** | Determinism | 10 repeated executions yield bit-for-bit identical outputs | **PASS** |
| **Test P** | Input Validation | Null pointers throw `InputException` | **PASS** |
| **Test Q** | Zero-Size Batch | Batch size 0 terminates immediately without launch | **PASS** |
| **Test R** | Strict GPU Mode | Strict mode throws `NumericalException` when GPU unavailable | **PASS** |
| **Test S** | CPU Fallback | Automatic fallback to CPU multicore records diagnostic reason | **PASS** |
| **Test T** | `--no-gpu` Behavior | Forced CPU fallback matches serial reference | **PASS** |
| **Test U** | Telemetry Accuracy | Telemetry captures dimensions, timings, and status | **PASS** |
| **Test V** | Transfer Accounting | Total GPU time strictly includes H2D, Kernel, and D2H | **PASS** |
| **Test W** | Speedup Formula | Speedup calculated as $T_{\text{cpu}} / T_{\text{gpu}}$ | **PASS** |
| **Crossover** | Crossover Sweep | Sweep across tiny, small, medium, and large instances | **PASS** |
| **Full Suite** | Regression Suite | All 269 unit tests across Phases 0–11 pass cleanly | **PASS (269/269)** |

---

## 12. Dependency Sovereignty

- **Sovereign Implementation**: The CUDA kernel in `cuda_spmv.cu` is authored directly by the Kavach-Opt team.
- **Zero Third-Party Libraries**: Zero usage of external GPU solvers or libraries:
  - NO cuSPARSE (no `cusparseSpMV`)
  - NO cuBLAS (no `cublasDgemv`)
  - NO cuSolver
  - NO cuOpt
  - NO Thrust or external template libraries
- Verified by `python3 tools/check_dependencies.py`.

---

## 13. Limitations & Future Scope

1. **Scoped Kernel**: Only batched CSR SpMV is accelerated on GPU.
2. **CPU-First Fallback**: CPU remains the authoritative reference engine and default solver backend.
3. **No GPU Factorization**: SparseLU factorization remains CPU-only.
4. **No Multi-GPU / Distributed**: Single-GPU acceleration within a single node is supported; multi-GPU distributed computing is out of scope.
