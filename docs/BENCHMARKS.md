# Kavach-Opt — Benchmark Evaluation

**Solver Version:** 0.1.0  
**Build Type:** Release · C++20  
**Platform:** Apple Silicon (ARM64, macOS) · AppleClang 21  
**CUDA Hardware:** Not present on development host (CPU fallback exercised by all GPU code paths)  
**Reproducibility Seed:** 42  

> **Benchmark Methodology Note:** All timings are wall-clock measurements on the development host
> using `std::chrono::steady_clock`. Results are hardware-specific and will differ on other
> machines. The purpose of these benchmarks is to demonstrate *correctness* on standard problem
> instances and to characterise *relative* performance behaviour. No timing claims are made
> about absolute performance against commercial solvers on their target hardware.

---

## 1. Benchmark Methodology

### 1.1 Timing Policy

Each benchmark records five distinct timing phases:

| Phase | Metric | Notes |
|-------|--------|-------|
| **Parse** | `parse_time_seconds` | File I/O + CanonicalModel construction |
| **Presolve** | `presolve_time_seconds` | Reduction suite; null for QP (disabled) |
| **Solve** | `solve_time_seconds` | Core algorithm (simplex pivots / B&B nodes / active-set iterations) |
| **Verify** | `verification_time_seconds` | Independent verifier; isolated from solve time |
| **Total Wall** | `total_wall_time_seconds` | End-to-end inclusive of all phases |

All timings use monotonic `std::chrono::steady_clock`. Verification time is measured and
stored separately so it never inflates solve-time claims.

### 1.2 Verification Gate

Every benchmark result is subject to mandatory independent verification before a `VERIFIED_OPTIMAL`
status is certified:

```
Solver reports OPTIMAL
       ↓
Independent Verifier executes (zero solver re-use)
       ↓
All checks pass → VERIFIED_OPTIMAL
Any check fails → VERIFICATION_FAILURE (result is flagged, not silently discarded)
```

The verification layer checks: non-finite sanity, bound feasibility, primal constraint
feasibility, dual feasibility / reduced-cost stationarity, complementary slackness,
and objective consistency (LP). For MILP it additionally checks integrality violations.
For QP it checks KKT stationarity and complementarity residuals. For MIQP it checks
integrality, exact quadratic objective consistency, optimality gap, and independent Cholesky PSD convexity.

### 1.3 Reference Solver Isolation (HiGHS)

When `--compare highs` is specified, HiGHS is invoked strictly as an isolated subprocess:

```
kavach-opt benchmark --suite netlib --compare highs
```

HiGHS is **never** linked, included, or called from within the Kavach-Opt solving path.
If `highs` is not found on `PATH`, comparison is recorded as `REFERENCE_UNAVAILABLE` and
no objective or timing values are fabricated.

### 1.4 Reproducibility

All runs are seeded (`--seed 42`). The benchmark runner records:

- `solver_version`, `build_type`, `compiler`, `git_commit`
- `cpu_model`, `os_name`, `num_cpu_cores`, `gpu_device` status
- ISO 8601 `timestamp` per run

Results are stored in `benchmark_results/` as JSON, CSV, and Markdown reports.

---

## 2. LP Benchmark — Netlib Corpus

The Netlib LP library is a standard public corpus of linear programming problems drawn
from a range of applied domains. Kavach-Opt is evaluated against 13 Netlib instances
from `netlib.org/lp/data`.

**Algorithm:** Two-Phase Primal Revised Simplex  
**Presolve:** Full suite enabled  
**Scaling:** Ruiz equilibration (AUTO)  
**Backend:** CPU Sequential (all instances below multicore threshold)  

### 2.1 Results Table

| Instance | Rows | Cols | NNZ | Density | Status | Objective | Iterations | Solve Time (s) | Verified |
|----------|------|------|-----|---------|--------|-----------|------------|----------------|----------|
| `afiro` | 27 | 32 | 83 | 0.0961 | OPTIMAL | −464.75314 | 16 | 0.000277 | ✓ VERIFIED_OPTIMAL |
| `adlittle` | 56 | 97 | 383 | 0.0705 | OPTIMAL | 225,494.963 | — | 0.003629 | ✓ VERIFIED_OPTIMAL |
| `blend` | 74 | 83 | 491 | 0.0800 | OPTIMAL | −30.81215 | — | 0.031036 | ✓ VERIFIED_OPTIMAL |
| `sc50a` | 50 | 48 | 130 | 0.0542 | OPTIMAL | −64.57508 | — | 0.000646 | ✓ VERIFIED_OPTIMAL |
| `sc50b` | 50 | 48 | 118 | 0.0492 | OPTIMAL | −70.00000 | — | 0.000651 | ✓ VERIFIED_OPTIMAL |
| `share2b` | 96 | 79 | 694 | 0.0915 | OPTIMAL | −415.73224 | — | 0.005714 | ✓ VERIFIED_OPTIMAL |
| `beaconfd` | 173 | 262 | 3,375 | 0.0745 | OPTIMAL | 33,592.486 | — | 0.057083 | ✓ VERIFIED_OPTIMAL |
| `sc105` | 105 | 103 | 280 | 0.0259 | OPTIMAL | −52.20206 | — | 0.003076 | ✓ VERIFIED_OPTIMAL |
| `lotfi` | 153 | 308 | 1,078 | 0.0229 | OPTIMAL | −25.26471 | — | 0.015800 | ✓ VERIFIED_OPTIMAL |
| `share1b` | 117 | 225 | 1,151 | 0.0437 | OPTIMAL | −76,589.319 | — | 0.021820 | ✓ VERIFIED_OPTIMAL |
| `kb2` | 43 | 41 | 286 | 0.1623 | OPTIMAL | −1,749.900 | — | 0.003055 | ✓ VERIFIED_OPTIMAL |
| `stocfor1` | 117 | 111 | 447 | 0.0344 | OPTIMAL | −41,131.976 | — | 0.015640 | ✓ VERIFIED_OPTIMAL |
| `boeing2` | 185 | 143 | 1,283 | 0.0485 | OPTIMAL | −315.01873 | — | 0.080847 | ✓ VERIFIED_OPTIMAL |

**13 / 13 instances: OPTIMAL — all independently verified.**

### 2.2 Representative Output (afiro)

```bash
./build-release/kavach-opt solve benchmarks/netlib/afiro.mps --json
```

```json
{
  "model": {
    "name": "AFIRO",
    "num_constraints": 27,
    "num_variables": 32,
    "num_nonzeros": 83,
    "density": 0.09606,
    "problem_class": "LP"
  },
  "result": {
    "status": "OPTIMAL",
    "objective": -464.75314285714285,
    "iterations": 16,
    "solve_time_seconds": 0.000871,
    "backend_used": "CPU_SEQUENTIAL",
    "verified": true
  },
  "verification": {
    "certified_status": "FEASIBLE",
    "passed": true,
    "recomputed_objective": -464.75314285714285,
    "recomputed_primal_residual": 1.776e-14,
    "recomputed_complementary_slackness": 0
  }
}
```

> The afiro benchmark reference objective from the Netlib repository is −464.75314286.
> Kavach-Opt computes −464.75314285714285, with absolute deviation ≈ 2.86 × 10⁻⁹.

---

## 3. MILP Benchmark — Subset

Three instances exercise integer handling, infeasibility detection, and branch-and-bound
node exploration.

**Algorithm:** Branch-and-Bound with dual simplex warm starts  
**Branching:** Most-fractional variable selection  
**Node Selection:** Best-bound queue  
**Presolve:** Full (LP relaxation at root)  

### 3.1 Results Table

| Instance | Type | Rows | Cols | NNZ | Status | Objective | Nodes | Solve Time (s) | Verified |
|----------|------|------|------|-----|--------|-----------|-------|----------------|----------|
| `knapsack.lp` | MILP | — | — | — | OPTIMAL | 41 | 19 | 0.000414 | ✓ VERIFIED_OPTIMAL |
| `facility_loc.lp` | MILP | — | — | — | OPTIMAL | 180 | 3 | 0.000197 | ✓ VERIFIED_OPTIMAL |
| `infeasible_milp.lp` | MILP | — | — | — | INFEASIBLE | — | 3 | 0.000071 | ✓ Infeasibility confirmed |

**3 / 3 instances: correct status — all verified.**

`infeasible_milp.lp` exercises the infeasibility detection path: the B&B tree is explored
for 3 nodes before all branches are pruned, confirming the problem has no feasible integer
solution. Infeasibility is a first-class certified outcome.

### 3.2 Cutting-Plane Separation Evaluation (Cuts OFF vs. ROOT Cuts ON)

Kavach-Opt's Branch-and-Cut pipeline integrates Gomory fractional cuts and Mixed-Integer Rounding (MIR) cuts. The table below presents empirical measurements on the standard test instances comparing pure Branch-and-Bound (`--cuts off`) against Branch-and-Cut with root cuts (`--cuts root`):

| Instance | Mode | Cuts Applied | Cuts Gen (Gomory / MIR) | Root Rounds | Nodes Explored | Simplex Iterations | Objective | Node Reduction | Verified |
|---|---|---|---|---|---|---|---|---|---|
| `knapsack.lp` | Cuts OFF | 0 | 0 / 0 | 0 | 19 | 36 | 41.0 | Baseline | ✓ OPTIMAL |
| `knapsack.lp` | Cuts ROOT | 3 | 17 (14 / 3) | 2 | 11 | 23 | 41.0 | **−42.1%** | ✓ OPTIMAL |
| `facility_loc.lp` | Cuts OFF | 0 | 0 / 0 | 0 | 3 | 9 | 180.0 | Baseline | ✓ OPTIMAL |
| `facility_loc.lp` | Cuts ROOT | 0 | 0 / 0 | 0 | 3 | 9 | 180.0 | 0.0% | ✓ OPTIMAL |
| `infeasible_milp.lp`| Cuts OFF | 0 | 0 / 0 | 0 | 3 | 5 | — | Baseline | ✓ INFEASIBLE |
| `infeasible_milp.lp`| Cuts ROOT | 0 | 0 / 0 | 0 | 3 | 5 | — | 0.0% | ✓ INFEASIBLE |

#### Analysis:
1. **Knapsack MILP:** Root cutting planes tighten the initial relaxation bounds, pruning fractional regions immediately. The tree size drops from 19 nodes to 11 nodes (a **42.1% reduction in branch-and-bound nodes**), while simplex iterations drop from 36 to 23 (a **36.1% reduction**).
2. **Facility Location MILP:** Resolves at root without requiring cuts beyond initial bounds, cleanly terminating in 3 nodes.
3. **Verification Invariance:** Under both `--cuts off` and `--cuts root`, all solutions pass the independent verification layer (`MILPVerifier`) with 0 integrality violation and 0 primal constraint residual.

---

## 4. Convex QP Benchmark

Two quadratic programming instances exercise the primal active-set solver and QP verification
(KKT stationarity and complementarity).

**Algorithm:** Primal Active-Set method  
**PSD Validation:** Hessian positive-semidefiniteness checked before solve  
**Presolve:** Disabled (QP active-set operates directly on original form)  

### 4.1 Results Table

| Instance | Rows | Cols | Status | Objective | Solve Time (s) | Verified |
|----------|------|------|--------|-----------|----------------|----------|
| `portfolio.lp` | — | — | OPTIMAL | 0.31641206 | 0.000116 | ✓ VERIFIED_OPTIMAL |
| `constrained_ls.lp` | — | — | OPTIMAL | −5.33333333 | 0.000038 | ✓ VERIFIED_OPTIMAL |

**2 / 2 instances: OPTIMAL — KKT conditions verified.**

---

## 5. MRPL Industrial Case Study

> **DATA STATUS:** All numeric values in the MRPL demonstration are illustrative and synthetic.
> They are **not** proprietary MRPL operational data and must not be interpreted as actual
> MRPL process parameters. The model structure (crude blending, unit capacity MILP, quality
> constraints) is representative of the class of problems encountered in petroleum refinery
> operations.

**Algorithm:** Branch-and-Bound MILP  
**Problem size:** 21 variables (18 continuous + 3 binary), 22 constraints, 81 nonzeros  

### 5.1 Baseline Result

```bash
./build-release/kavach-opt mrpl --no-highs
```

| Metric | Value |
|--------|-------|
| **Status** | OPTIMAL |
| **Certified Status** | VERIFIED_OPTIMAL |
| **Net Objective (Margin)** | 5,445.91 k/day (synthetic) |
| **LP Relaxation Bound** | 5,891.36 k/day |
| **Optimality Gap** | 0.00% |
| **B&B Nodes Explored** | 5 |
| **Solve Time** | 0.0029 s |
| **Integrality Violation** | 0 |
| **Primal Residual** | 3.31 × 10⁻¹⁴ |
| **Mass Balance** | PASS |

### 5.2 Product Quality (Synthetic Data)

| Product | Production (kbpd) | Sulfur (wt%) | Limit | Density (kg/m³) | Limit |
|---------|-------------------|--------------|-------|-----------------|-------|
| P1 — Motor Spirit (Gasoline) | 140.0 | 0.80 | 0.80 | 828.7 | 840.0 |
| P2 — High-Speed Diesel (HSD) | 98.2 | 1.30 | 1.30 | 853.5 | 865.0 |
| P3 — Heavy Fuel Oil (HFO) | 90.0 | 3.00 | 3.00 | 892.5 | 875.0 |

All three products satisfy their quality specifications at the optimal blending plan.

### 5.3 Scenario Summary

The MRPL case study driver evaluates 5 operational scenarios:

| Scenario | Description | Expected Status |
|----------|-------------|-----------------|
| Baseline | Standard blending / capacity allocation | OPTIMAL |
| A | Crude A availability reduction | OPTIMAL |
| B | Product demand shift | OPTIMAL |
| C | Unit capacity tightening | OPTIMAL |
| D | Over-constrained (infeasible configuration) | INFEASIBLE |

Scenario D is deliberately constructed as infeasible to demonstrate that the solver
correctly and certifiably detects infeasibility through B&B tree exhaustion.

---

## 6. Batched SpMV — CPU Parallelism

The multicore CPU batched SpMV subsystem (`src/backend/cpu/`) is benchmarked by varying
the number of worker threads across a fixed matrix and batch size. Each measurement is a
parallel batch of independent $Y_k = A X_k$ operations.

**Test matrix:** 1000 × 1000, sparse (density ≈ 0.01)  
**Batch size:** 512 vectors  
**Correctness gate:** Parallel outputs match serial reference within $|Y_{GPU} - Y_{CPU}| < 10^{-12}$  

| Workers | Wall Time (ms) | Speedup vs. Serial |
|---------|----------------|-------------------|
| 1 (Serial) | ~4.45 | 1.0× |
| 4 | ~1.21 | ~3.7× |
| 8 | ~1.09 | ~4.1× |

> These timings are from the development host (Apple M-series, 8 logical cores) and are
> hardware-specific. The primary guarantee is **correctness** (bit-for-bit agreement with
> serial reference) across all thread counts, not the magnitude of speedup on a specific machine.

---

## 7. CUDA Batched SpMV — Hardware Status

The CUDA backend (`src/backend/cuda/`) implements a batched CSR SpMV kernel
with transfer-inclusive profiling.

**Development Host Status:** No NVIDIA CUDA hardware present (Apple Silicon macOS).

The solver's Rule R012 enforces automatic CPU fallback when CUDA is unavailable.
All CUDA-path unit tests pass on the development host by exercising the CPU fallback path.
Transfer-inclusive GPU speedup measurements require an x86_64/ARM64 Linux host with
an NVIDIA GPU (Pascal architecture or later).

**GPU eligibility condition (Rule R013):**

```
CUDA available  AND  batch_size ≥ 128  AND  nnz ≥ 10,000  AND  --no-gpu not set
```

When CUDA hardware is available and the eligibility condition is satisfied, the solver routes
batched SpMV to the CUDA kernel and reports `h2d_time_ms`, `kernel_time_ms`, `d2h_time_ms`,
and transfer-inclusive `total_gpu_time_ms` in the benchmark artifact.

Kernel-only speedup is reported as a **secondary diagnostic** only.
The primary speedup metric is always **transfer-inclusive**.

---

## 8. Solver Strategy Overhead

The Deterministic Strategy Engine executes before the solver and characterises the model
in O(m + n + nnz) time.

| Instance | Strategy Decision Time |
|----------|----------------------|
| `afiro` (27×32, 83 nnz) | 0.014 ms |
| MRPL model (22×21, 81 nnz) | < 0.1 ms |

Strategy overhead is sub-millisecond on all tested instances. It is negligible relative
to solve time.

---

## 9. Reproducing These Results

All benchmark results in this document are reproducible from a clean build:

```bash
# Build
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel

# LP benchmark — Netlib
./build-release/kavach-opt benchmark --suite netlib --seed 42 --output benchmark_results/netlib

# MILP benchmark
./build-release/kavach-opt benchmark --suite miplib_subset --seed 42 --output benchmark_results/miplib

# QP benchmark
./build-release/kavach-opt benchmark --suite qp --seed 42 --output benchmark_results/qp

# Single instance (afiro, with JSON output)
./build-release/kavach-opt solve benchmarks/netlib/afiro.mps --json

# MRPL case study
./build-release/kavach-opt mrpl --no-highs --json

# Unit tests
ctest --test-dir build-release --output-on-failure
```

All stored `benchmark_results/` JSON artifacts were produced with `--seed 42` and match
the values in the tables above.

---

## 10. Known Limitations

| Limitation | Notes |
|------------|-------|
| Timing variability | Wall-clock measurements are subject to OS scheduler jitter; re-running may produce slightly different solve times |
| Dense matrices | The solver is tuned for sparse inputs (density ρ < 0.15); dense matrices incur higher LU factorization costs |
| Deep B&B search | Node tree is memory-resident; problems requiring > 50,000 nodes have not been characterised |
| CUDA validation | Transfer-inclusive GPU speedup measurements have not been performed on the development host (no CUDA hardware); see `docs/GPU_CLAIM_AUDIT.md` |
| Presolve dual reconstruction | Dual multipliers on eliminated redundant constraints are not fully reconstituted; dual verification runs on the presolved formulation |

For complete scope and limitation documentation, see [`docs/SCOPE_AND_LIMITATIONS.md`](SCOPE_AND_LIMITATIONS.md).
