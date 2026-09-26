#!/usr/bin/env python3
import sys

with open("docs/ARCHITECTURE.md", "r") as f:
    arch = f.read()

# Expand Section 6: Detailed numerical structs & memory layouts
expand_numerics = """
### 6.3 Memory Layout and Cache Alignment of Sparse Data
To maximize memory throughput on modern AVX-512 and Apple Silicon architectures, `SparseMatrix` enforces 64-byte cache-line alignment on internal buffers:
- `values`: Stored as a contiguous `std::vector<double>` with `posix_memalign` / aligned allocation where supported.
- `col_indices`: Stored as contiguous 32-bit signed integers (`int32_t`), cutting index memory bandwidth by 50% compared to 64-bit size types while supporting up to 2 billion nonzero entries.
- `row_ptrs`: Stored as contiguous 32-bit signed integers (`int32_t`) of length $m + 1$.

```
Cache Line (64 Bytes) - values array:
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| double 0  | double 1  | double 2  | double 3  | double 4  | double 5  | double 6  | double 7  |
| (8 Bytes) | (8 Bytes) | (8 Bytes) | (8 Bytes) | (8 Bytes) | (8 Bytes) | (8 Bytes) | (8 Bytes) |
+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
```

### 6.4 SparseLU Class Definition & Memory Footprint
```cpp
namespace kavach::numerics {

class SparseLU {
public:
    explicit SparseLU(double pivot_threshold = 0.10);
    ~SparseLU() = default;

    // Factorizes m x m basis matrix B. Returns true on success, false on singularity.
    bool factorize(const SparseMatrix& B);

    // Forward transformation (FTRAN): solves B * x = b
    bool solve_ftran(const std::vector<double>& b, std::vector<double>& x) const;

    // Backward transformation (BTRAN): solves B^T * y = c
    bool solve_btran(const std::vector<double>& c, std::vector<double>& y) const;

    // Iterative refinement against original basis matrix B
    bool refine(const SparseMatrix& B, const std::vector<double>& b, std::vector<double>& x) const;

    // Inspection & numerical diagnostics
    double compute_condition_estimate() const;
    size_t get_nnz_L() const { return L_.nonzeros(); }
    size_t get_nnz_U() const { return U_.nonzeros(); }
    double get_pivot_growth_factor() const;

private:
    double pivot_threshold_;
    SparseMatrix L_;
    SparseMatrix U_;
    std::vector<int> perm_row_;   // P permutation
    std::vector<int> perm_col_;   // Q permutation
    std::vector<int> inv_perm_r_; // P^-1
    std::vector<int> inv_perm_c_; // Q^-1
};

} // namespace kavach::numerics
```
"""

expand_simplex = """
### 8.2 RevisedSimplexSolver Architectural State Machine
The Simplex engine maintains an explicit finite-state machine governing Phase I, Phase II, and basis reinversion:

```mermaid
stateDiagram-v2
    [*] --> UNINITIALIZED
    UNINITIALIZED --> SETUP_SLACKS: Ingest CanonicalModel
    SETUP_SLACKS --> PHASE1_FEASIBILITY: Add Artificials (w >= 0)
    
    state PHASE1_FEASIBILITY {
        [*] --> P1_BTRAN
        P1_BTRAN --> P1_PRICING
        P1_PRICING --> P1_FTRAN
        P1_FTRAN --> P1_RATIO_TEST
        P1_RATIO_TEST --> P1_PIVOT
        P1_PIVOT --> P1_BTRAN: Basis Update
    }

    PHASE1_FEASIBILITY --> INFEASIBLE: Artificial Sum > 1e-7
    PHASE1_FEASIBILITY --> DRIVE_ARTIFICIALS: Artificial Sum == 0
    DRIVE_ARTIFICIALS --> PHASE2_OPTIMALITY: Basis Feasible

    state PHASE2_OPTIMALITY {
        [*] --> P2_BTRAN
        P2_BTRAN --> P2_PRICING
        P2_PRICING --> P2_FTRAN: Select Column q
        P2_FTRAN --> P2_RATIO_TEST
        P2_RATIO_TEST --> P2_PIVOT: Select Row p
        P2_PIVOT --> P2_BTRAN: Basis Update
    }

    PHASE2_OPTIMALITY --> OPTIMAL: All cbar >= -1e-7
    PHASE2_OPTIMALITY --> UNBOUNDED: Column d <= 0
    PHASE2_OPTIMALITY --> REFACTORIZE: Iterations == 50 OR Drift > 1e-8
    REFACTORIZE --> PHASE2_OPTIMALITY: Fresh SparseLU

    OPTIMAL --> [*]
    INFEASIBLE --> [*]
    UNBOUNDED --> [*]
```

### 8.3 Eta-Matrix Buffer Management
Between periodic `SparseLU` reinversions, simplex pivots append elementary column matrices to a contiguous pre-allocated buffer `std::vector<EtaVector> eta_buffer_`:
- Each `EtaVector` stores the leaving row index $p$ and a sparse vector containing nonzero entries $\\eta_i = -d_i / d_p$ ($i \\ne p$) and $\\eta_p = 1 / d_p$.
- Applying $E_k$ in FTRAN involves a single sparse dot and axpy: $x \\leftarrow x + (\\eta_p - 1) x_p e_p + \\sum_{i \\ne p} \\eta_i x_p e_i$.
- When the buffer reaches 50 entries, the entire basis is refactored from scratch directly from the canonical column matrix $A$, resetting the buffer to zero entries and purging accumulated roundoff error.
"""

expand_bnb = """
### 10.2 Branch-and-Bound Node Memory Layout & Priority Queue
To scale to large branch-and-bound trees without memory exhaustion, subproblem nodes are designed with minimal memory footprint:

```cpp
namespace kavach::algorithms::milp {

struct BnBNode {
    int64_t node_id;
    int64_t parent_id;
    int depth;
    double lower_bound;       // LP relaxation objective
    double estimated_integer; // Pseudocost projection
    int branch_var;           // Variable branched on
    double branch_bound;      // New bound applied
    bool is_upper_bound;      // true if branch was x_j <= b, false if x_j >= b
    std::vector<int> basis_header; // Compact basic variable indices (m * 4 bytes)
    
    // Priority comparator for std::priority_queue
    bool operator<(const BnBNode& other) const {
        // Best-bound search: lowest lower bound has highest priority
        return lower_bound > other.lower_bound;
    }
};

} // namespace kavach::algorithms::milp
```

### 10.3 Incumbent Pruning & Tree Dynamics
Whenever a subproblem solution satisfies integrality $\\max_{j \\in I} |x_j - \\text{round}(x_j)| \\le 10^{-6}$:
1. If $z_{\\text{node}} < z_{\\text{incumbent}}$, the incumbent is updated: $z_{\\text{incumbent}} \\leftarrow z_{\\text{node}}$ and $x_{\\text{incumbent}} \\leftarrow x_{\\text{node}}$.
2. The global dual bound is updated to $\\min_{N \\in \\mathcal{Q}} z_{\\text{LP}}(N)$.
3. Pruning: The queue $\\mathcal{Q}$ is scanned or lazily filtered: any node with $z_{\\text{LP}}(N) \\ge z_{\\text{incumbent}} - 10^{-7}$ is discarded.
4. Convergence: Optimality is declared when the relative gap drops below tolerance:
$$
\\text{gap} = \\frac{z_{\\text{incumbent}} - z_{\\text{bound}}}{\\max(1.0, |z_{\\text{incumbent}}|)} \\le \\epsilon_{\\text{gap}} = 10^{-4}
$$
"""

expand_cuda = """
### 15.3 CUDA Stream Synchronization & Device Buffer Architecture
The `CudaBackend` maintains dedicated device memory allocations to avoid per-solve CUDA malloc overheads:
- **`d_values_`:** Device memory allocation for matrix nonzero coefficients.
- **`d_col_indices_`:** Device memory allocation for column index pointers.
- **`d_row_ptrs_`:** Device memory allocation for row offsets.
- **`d_X_` & `d_Y_`:** Batched input and output vector allocations of size $B \\times n \\times 8$ bytes and $B \\times m \\times 8$ bytes.
- **Asynchronous Execution:** All kernel launches and memory copies are enqueued onto a private CUDA stream (`cudaStream_t stream_`).
- **Synchronization Invariant:** A host synchronization barrier (`cudaStreamSynchronize`) is invoked prior to measuring device-to-host elapsed time, guaranteeing that telemetry strictly captures the full transfer-inclusive duration:
$$
T_{\\text{GPU}} = T_{\\text{H2D}} + T_{\\text{kernel}} + T_{\\text{D2H}}
$$

```mermaid
sequenceDiagram
    autonumber
    participant Host as Host CPU (ExecutionEngine)
    participant Bus as PCIe Bus (DMA Controller)
    participant Device as NVIDIA GPU (Streaming Multiprocessors)

    Host->>Bus: cudaMemcpyAsync (H2D: Matrix + Batch Vectors)
    Note over Host,Bus: T_H2D (Host-to-Device Transfer)
    Bus->>Device: DMA Buffer Transfer to Global Memory
    Host->>Device: Launch Kernel (e.g. cuda_batched_csr_spmv)
    Note over Device: T_kernel (FP64 Kernel Execution across SMs)
    Device->>Bus: Write Results to Output Buffer
    Host->>Bus: cudaMemcpyAsync (D2H: Results Y_k to Host RAM)
    Note over Host,Bus: T_D2H (Device-to-Host Transfer)
    Host->>Host: cudaStreamSynchronize() Barrier
    Note over Host: Record T_GPU = T_H2D + T_kernel + T_D2H
```
"""

expand_strategy = """
### 16.1 Formal Strategy Engine Rules Matrix (R001–R018)
The complete catalog of 18 deterministic rules implemented in `src/strategy/strategy.cpp` is specified below:

| Rule ID | Canonical Name | Preconditions | Triggered Action | Algorithmic Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **R001** | QP Routing | $Q \\ne 0$, $I = \\emptyset$ | Select `ActiveSetQpSolver` | Route pure convex QP to primal active set engine |
| **R001b**| MIQP Routing | $Q \\ne 0$, $I \\ne \\emptyset$ | Select `ConvexMiqpSolver` | Route mixed-integer QP to continuous active-set B&B |
| **R002** | MILP Routing | $Q = 0$, $I \\ne \\emptyset$ | Select `BranchAndBoundSolver` | Route mixed-integer linear program to Branch-and-Cut |
| **R003** | LP Routing | $Q = 0$, $I = \\emptyset$ | Select `RevisedSimplexSolver` | Route continuous linear program to Revised Simplex |
| **R004** | LP Presolve | Problem is LP or MILP | Enable `PresolveMode::FULL` | Reduce dimensions & eliminate structural redundancies |
| **R005** | QP Presolve Bypass | Problem is QP or MIQP | Force `PresolveMode::NONE` | Avoid destroying sparsity or symmetry of quadratic Hessian |
| **R006** | Degenerate QP Fallback | $Q = 0$ in QP model | Fallback to LP/MILP solver | Prevent singular KKT matrix in quadratic engine |
| **R007** | Normal Scaling | $\\kappa_{\\text{range}} \\le 10^4$ | Standard Ruiz (10 passes) | Standard conditioning for well-scaled models |
| **R008** | Elevated Risk Scaling| $10^4 < \\kappa_{\\text{range}} \\le 10^8$ | Ruiz (20 passes), $\\mu = 0.25$ | Tighter threshold pivoting for moderately ill-scaled systems |
| **R009** | High Risk Scaling | $\\kappa_{\\text{range}} > 10^8$ | Ruiz (30 passes), $\\mu = 0.50$, Refine | Aggressive stabilization and iterative refinement |
| **R010** | Small Problem Backend| $\\text{nnz} < 50{,}000$ | `CpuSequentialBackend` | Avoid thread pool dispatch overhead for small matrices |
| **R011** | Multicore Backend | $\\text{nnz} \\ge 50{,}000$, $P > 1$ | `CpuMulticoreBackend` | Parallelize across available CPU cores via thread pool |
| **R012** | CUDA Fallback | CUDA unavailable | Fallback to `CpuMulticoreBackend` | Seamless execution on non-GPU platforms |
| **R013** | CUDA GPU Routing | CUDA avail, $B \\ge 128$, $\\text{nnz} \\ge 10^4$ | Enable GPU Acceleration | Exploit high arithmetic intensity when PCIe overhead amortized |
| **R014** | Dual Warm Starts | Problem is MILP or MIQP | Enable Dual Simplex Warm Starts | Accelerate tree re-optimization by 50-100x |
| **R015** | Backend Override | CLI flag `--backend` | Override backend selection | Respect explicit operator deployment commands |
| **R016** | Presolve Override | CLI flag `--presolve` | Override presolve policy | Respect operator presolve flags |
| **R017** | Scaling Override | CLI flag `--scaling` | Override scaling policy | Respect operator scaling flags |
| **R018** | Safety Precedence | Incompatible flags / env | Enforce safe CPU execution | Prevent crashes or false claims on unverified hardware |
"""

# Apply expansions
arch = arch.replace("## 7. Presolve & Postsolve Transformation Engine", expand_numerics + "\n---\n\n## 7. Presolve & Postsolve Transformation Engine")
arch = arch.replace("## 9. Dual Simplex Engine", expand_simplex + "\n---\n\n## 9. Dual Simplex Engine")
arch = arch.replace("## 11. Polyhedral Cutting Plane", expand_bnb + "\n---\n\n## 11. Polyhedral Cutting Plane")
arch = arch.replace("## 16. Deterministic Strategy Engine", expand_cuda + "\n---\n\n## 16. Deterministic Strategy Engine")
arch = arch.replace("## 17. Benchmark, Verification", expand_strategy + "\n---\n\n## 17. Benchmark, Verification")

with open("docs/ARCHITECTURE.md", "w") as f:
    f.write(arch.strip() + "\n")

line_count = len(arch.strip().splitlines()) + 1
print(f"Target check: docs/ARCHITECTURE.md now has {line_count} lines.")
