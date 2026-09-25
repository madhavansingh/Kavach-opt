# Kavach-Opt Benchmark Methodology

This document defines the formal benchmarking methodology, result schemas, timing principles, verification gates, and reference solver isolation protocols for the **Kavach-Opt Optimization Solver Core** (SIH26119).

---

## 1. Principles and Scope

The Kavach-Opt benchmark subsystem is designed to provide **honest, reproducible, and verifiable performance evaluation** across linear programming (LP), mixed-integer linear programming (MILP), convex quadratic programming (QP), and batched sparse matrix-vector multiplication (SpMV).

### Core Invariants:
1. **First-Class Result Preservation**: Every attempted benchmark instance generates a structured result record. Instances that fail, time out, or exceed limits are preserved in all machine- and human-readable reports. No failed instance is ever silently discarded.
2. **Independent Verification Gate**: A result is certified `OPTIMAL` if and only if confirmed by the independent verification layer (`LPVerifier`, `MILPVerifier`, or `QPVerifier`). Unverified results are downgraded to `FEASIBLE` or `VERIFICATION_FAILURE`.
3. **Reference Solver Isolation**: Comparison against established solvers (HiGHS) is strictly decoupled via external process invocation. HiGHS is never linked, included, or used in the Kavach-Opt solver path.
4. **Hardware and Environment Transparency**: Platform capabilities (e.g. CPU core count, lack of CUDA hardware on macOS development systems) are recorded explicitly. CPU fallback is reported without fabricated GPU acceleration metrics.

---

## 2. Benchmark Corpus and Suites

### 2.1 LP: Netlib Benchmark Corpus
- **Source**: Authentic Netlib Linear Programming data repository (`netlib.org/lp/data`).
- **Instances**: 13 standard test problems (`afiro`, `adlittle`, `blend`, `sc50a`, `sc50b`, `share2b`, `beaconfd`, `sc105`, `lotfi`, `share1b`, `kb2`, `stocfor1`, `boeing2`).
- **Format**: Mathematical Programming System (`.mps`).
- **Manifest**: `benchmarks/manifests/netlib.json`.

### 2.2 MILP: MIPLIB Subset
- **Scope**: Size-appropriate problems stressing integer handling, weak LP relaxations, branch-and-bound node exploration, and infeasibility detection.
- **Instances**: `knapsack.lp`, `facility_loc.lp`, `infeasible_milp.lp`.
- **Format**: Native LP format (`.lp`).
- **Manifest**: `benchmarks/manifests/miplib_subset.json`.

### 2.3 Convex QP Suite
- **Scope**: Quadratic objective models with symmetric positive-semidefinite (PSD) Hessians solved via the primal active-set method.
- **Instances**: `portfolio.lp`, `constrained_ls.lp`.
- **Format**: Native LP format with bracketed quadratic objectives.
- **Manifest**: `benchmarks/manifests/qp.json`.

### 2.4 Multicore CPU & CUDA GPU Batched-SpMV
- **Scope**: Parallel throughput evaluation for batched CSR sparse matrix-vector multiplication across varying batch sizes and thread counts.
- **Telemetry**: Host-to-Device (H2D) transfer, GPU kernel execution, Device-to-Host (D2H) transfer, total transfer-inclusive time, and speedup over CPU sequential and multicore baselines.

---

## 3. Timing and Metric Accounting

All timings are recorded using monotonic wall-clock timers (`std::chrono::steady_clock`):

1. **Load & Parse Time**: Time to read file from disk and construct the validated `CanonicalModel`.
2. **Presolve Time**: Duration of presolve reductions (if enabled).
3. **Solve Time**: Core solver runtime (revised simplex pivots, B&B node tree exploration, or active-set KKT solves).
4. **Verification Time**: Independent verification execution time, strictly separated from solver runtime.
5. **Total Wall-Clock Time**: End-to-end elapsed time including parsing, solving, and verifying.

### GPU Timing Policy:
- **Transfer-Inclusive Time**: $T_{\text{GPU}} = T_{\text{H2D}} + T_{\text{Kernel}} + T_{\text{D2H}}$.
- **Reported Speedup**: $S_{\text{Transfer-Inclusive}} = T_{\text{CPU}} / T_{\text{GPU}}$.
- Kernel-only speedup ($T_{\text{CPU}} / T_{\text{Kernel}}$) is reported strictly as secondary diagnostic telemetry.

---

## 4. Verification and Certification Hierarchy

| Solver Status Claim | Verification Requirement | Certified Status |
|---|---|---|
| `OPTIMAL` (LP) | Primal feasibility, bound feasibility, objective consistency, non-finite check | `VERIFIED_OPTIMAL` |
| `OPTIMAL` (MILP) | Integrality of integer variables, constraint bounds, primal feasibility | `VERIFIED_OPTIMAL` |
| `OPTIMAL` (QP) | Stationarity, primal/dual feasibility, complementarity ($KKT \le \text{tol}$) | `VERIFIED_OPTIMAL` |
| Verification Discrepancy | Infeasible residuals or objective mismatch | `VERIFICATION_FAILURE` |
| Limit Reached | Iteration or node budget exhausted | `LIMIT_REACHED` |
| Timeout | Wall-clock time limit exceeded | `SOLVER_TIMEOUT` |

---

## 5. External Reference Solver Comparison (HiGHS)

To ensure fair and sovereign benchmarking:
1. **Decoupled Process Execution**: HiGHS is invoked as an isolated subprocess using `highs <model> --time_limit=<sec>`.
2. **Absence Handling**: If `highs` is not found on `PATH` or in `HIGHS_BIN`, the comparison is marked `REFERENCE_UNAVAILABLE`. No timings or objectives are fabricated.
3. **Objective Comparison**:
   $$\Delta_{\text{abs}} = |z_{\text{kavach}} - z_{\text{ref}}|$$
   $$\Delta_{\text{rel}} = \frac{\Delta_{\text{abs}}}{\max(1.0, |z_{\text{ref}}|)}$$
   A match is certified if $\Delta_{\text{abs}} \le 10^{-4}$ or $\Delta_{\text{rel}} \le 10^{-4}$.

---

## 6. Aggregate Statistics Traceability

All aggregate statistics (medians, min/max, geometric errors, percentiles) are **strictly derived from the per-instance records**. No arbitrary weighting or hidden instance exclusion is permitted. If a category filter is applied, the report explicitly states which instances were included.
