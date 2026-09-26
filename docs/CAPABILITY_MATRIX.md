# Kavach-Opt Capability Matrix

**Project**: Kavach-Opt Optimization Suite (SIH26119)  
**Version**: 0.1.0  
**Specification**: Release Candidate Capability Verification  

---

## 1. Comprehensive System Capability Matrix

| Capability | Status | Implementation File(s) | Verification Test(s) | Benchmark Suite | CLI / API Interface | Technical Notes |
|:---|:---:|:---|:---|:---|:---|:---|
| **Linear Programming (LP)** | **IMPLEMENTED** | `src/algorithms/lp/revised_simplex.cpp`, `src/algorithms/lp/basis.cpp` | `tests/unit/test_phase5_lp.cpp` | Netlib LP Suite (13 instances) | `kavach-opt solve <m.mps>`<br>`api::KavachSolver::solve()` | Two-Phase Revised Simplex, Bland anti-cycling, scaling, presolve. |
| **Mixed-Integer Linear (MILP)** | **IMPLEMENTED** | `src/algorithms/milp/branch_and_bound.cpp` | `tests/unit/test_phase7_milp.cpp` | MIPLIB Subset (`knapsack.lp`, `p0033`, `facility_loc.lp`) | `kavach-opt solve <m.lp>`<br>`api::KavachSolver::solve()` | Branch-and-Bound, most-fractional branching, best-bound node selection, root presolve. |
| **Dual Simplex Warm Starts** | **IMPLEMENTED** | `src/algorithms/lp/dual_simplex.cpp` | `tests/unit/test_phase8_dual_simplex.cpp` | Integrated into B&B node evaluation | Automatic in B&B; configurable via API | Accelerates child LP re-optimization after bound branches. |
| **Convex Quadratic (QP)** | **IMPLEMENTED** | `src/algorithms/qp/active_set.cpp` | `tests/unit/test_phase9_qp.cpp` | Convex QP Suite (`portfolio.lp`, `constrained_ls.lp`) | `kavach-opt solve <m.lp>`<br>`api::KavachSolver::solve()` | Primal active-set method, PSD Hessian validation, KKT residual termination. |
| **Sparse LU Factorization** | **IMPLEMENTED** | `src/numerics/lu_factorization.cpp` | `tests/unit/test_phase4_sparse_lu.cpp` | Basis factorization across Netlib instances | Internal numerical layer (`numerics::SparseLU`) | Markowitz threshold pivoting, iterative refinement for numerical stability. |
| **Presolve Pipeline** | **IMPLEMENTED** | `src/presolve/presolve.cpp` | `tests/unit/test_presolve.cpp` | Netlib & MIPLIB presolve reduction benchmarks | `kavach-opt presolve <m.mps> [--report]` | Fixed variables, empty rows/cols, singleton tightening, redundancy removal. |
| **Independent Verification** | **IMPLEMENTED** | `src/verification/lp_verifier.cpp`, `src/verification/milp_verifier.cpp`, `src/verification/qp_verifier.cpp` | `tests/unit/test_phase6_verification.cpp`, `tests/unit/test_phase10_verification.cpp` | Post-solve certification across all benchmarks | `kavach-opt verify <m.mps>`<br>`SolverResult::verification_report` | Decoupled verification of primal/dual residuals, slackness, and integrality. |
| **CPU Multicore Batched SpMV**| **IMPLEMENTED** | `src/backend/cpu/cpu_backend.cpp`, `src/backend/cpu/thread_pool.cpp` | `tests/unit/test_phase10_cpu_parallel.cpp` | Batched SpMV sweep (1 to 8 threads, batch 128 to 512) | `--backend multicore`<br>`api::SolverOptions::use_gpu` | C++20 standard library thread pool, parallel batch vector distribution. |
| **CUDA Batched SpMV** | **IMPLEMENTED** | `src/backend/cuda/cuda_backend.cpp`, `src/backend/cuda/cuda_spmv.cu` | `tests/unit/test_phase11_cuda.cpp` | GPU crossover sweep harness (batch 1 to 256) | `--backend cuda`<br>`--no-gpu` | Single scoped GPU kernel family; automatic CPU fallback when CUDA is absent. |
| **Benchmark Framework** | **IMPLEMENTED** | `src/benchmark/benchmark_runner.cpp`, `src/benchmark/benchmark_io.cpp` | `tests/unit/test_phase12_benchmark.cpp` | Netlib, MIPLIB, and QP manifests | `kavach-opt benchmark [opts]` | Deterministic seeds, JSON/CSV exports, markdown summary reports. |
| **HiGHS Reference Isolation** | **IMPLEMENTED** | `src/benchmark/reference/reference_adapter.cpp` | `tests/unit/test_phase12_benchmark.cpp` | Cross-solver verification harness | `--compare highs` | Strictly out-of-process invocation; returns `REFERENCE_UNAVAILABLE` if missing. |
| **MRPL Industrial Case Study** | **IMPLEMENTED** | `src/case_study/mrpl/mrpl_model.cpp` | `tests/unit/test_phase13_mrpl.cpp` | Baseline & 4 what-if operational scenarios | `kavach-opt mrpl [--no-highs]` | Synthetic refinery crude blending MILP with mass balance and quality limits. |
| **Rule-Based Strategy Layer** | **IMPLEMENTED** | `src/strategy/strategy.cpp`, `src/strategy/strategy.hpp` | `tests/unit/test_phase14_strategy.cpp` | 19 benchmark models characterized and mapped | `--show-strategy`<br>`--explain-strategy`<br>`api::KavachSolver::plan()` | Rules R001–R018, dynamic range detection, numerical risk profiles, audit JSON. |
| **Mixed-Integer QP (MIQP)** | **NOT IMPLEMENTED** | N/A | N/A | N/A | Rejection at parse/plan time | Explicitly rejected; quadratic objectives with discrete variables unsupported. |
| **Nonlinear Programming (NLP)**| **NOT IMPLEMENTED** | N/A | N/A | N/A | Rejection at parse/plan time | General non-linear equations (trig, exp, log) are unsupported. |
| **Mixed-Integer NLP (MINLP)** | **NOT IMPLEMENTED** | N/A | N/A | N/A | Rejection at parse/plan time | Mixed-integer general non-linear programming is unsupported. |
| **Non-Convex QP** | **NOT IMPLEMENTED** | N/A | `test_qp_t_non_psd_rejection` | N/A | Error returned upon non-PSD detection | Requires positive semi-definite (PSD) Hessian ($Q \succeq 0$). Non-PSD rejected. |
| **GPU Simplex Solver** | **NOT IMPLEMENTED** | N/A | N/A | N/A | N/A | Revised simplex basis updates run sequentially on CPU. |
| **GPU Branch-and-Bound** | **NOT IMPLEMENTED** | N/A | N/A | N/A | N/A | MILP tree search executes on CPU. |
| **GPU SparseLU Factorization** | **NOT IMPLEMENTED** | N/A | N/A | N/A | N/A | Sparse LU with Markowitz pivoting executes on CPU. |

---

## 2. Sovereignty & Dependency Governance

- **External Optimization Engines**: 0 (Zero CPLEX, Gurobi, HiGHS, GLPK, CBC, SCIP in solver path).
- **Core Standard**: ISO C++20 standard library facilities only.
- **External Linear Algebra Libraries**: 0 (Eigen, BLAS, LAPACK, SuiteSparse are NOT used).
- **Hardware Abstraction**: Dual-target CPU/CUDA with zero runtime overhead on CPU-only targets.
