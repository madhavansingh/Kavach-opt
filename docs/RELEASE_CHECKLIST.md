# Kavach-Opt Release Checklist

**Release Candidate**: v0.1.0-RC1  
**Project**: Kavach-Opt Sovereign Indigenous Optimization Solver Core (SIH26119)  
**Date**: September 2026  
**Auditor**: Release Gate & Verification Authority  

---

## 1. System Readiness Checklist

| Subsystem / Requirement | Status | Verification Evidence / Reference | Auditor Sign-off |
|:---|:---:|:---|:---:|
| **[x] Solver Core & Foundation** | **COMPLETE** | `src/core/` logging, status enums, versioning, error handling. | VERIFIED |
| **[x] Parser Pipeline** | **COMPLETE** | `src/io/` fixed/free MPS and native LP format parsers pass all syntax tests. | VERIFIED |
| **[x] Sparse Numerics Layer** | **COMPLETE** | `src/numerics/` CSR/CSC, Sparse LU, Markowitz pivoting, equilibration scaling. | VERIFIED |
| **[x] Presolve Engine** | **COMPLETE** | `src/presolve/` fixed vars, empty rows/cols, singleton tightening, redundancy removal. | VERIFIED |
| **[x] Linear Programming (LP)** | **COMPLETE** | `src/algorithms/lp/` Two-Phase Revised Simplex with Bland anti-cycling. | VERIFIED |
| **[x] Mixed-Integer LP (MILP)** | **COMPLETE** | `src/algorithms/milp/` Branch-and-Bound, most-fractional branching, root presolve. | VERIFIED |
| **[x] Dual-Simplex Warm Starts** | **COMPLETE** | `src/algorithms/lp/dual_simplex.cpp` hot basis updates during B&B tree search. | VERIFIED |
| **[x] Convex Quadratic (QP)** | **COMPLETE** | `src/algorithms/qp/` active-set method, PSD Hessian check, KKT convergence. | VERIFIED |
| **[x] Independent Verification** | **COMPLETE** | `src/verification/` independent LP/MILP/QP verifiers enforce OPTIMAL invariant. | VERIFIED |
| **[x] CPU Multicore Parallelism** | **COMPLETE** | `src/backend/cpu/` standard thread pool parallelizing batched SpMV operations. | VERIFIED |
| **[x] Scoped CUDA Backend** | **COMPLETE** | `src/backend/cuda/` CSR batched SpMV kernel, transfer-inclusive timing, CPU fallback. | VERIFIED |
| **[x] Benchmark Framework** | **COMPLETE** | `src/benchmark/` automated manifests, metrics aggregation, JSON/CSV/markdown. | VERIFIED |
| **[x] HiGHS Reference Isolation** | **COMPLETE** | `src/benchmark/reference/` out-of-process subprocess invocation; 0 build linking. | VERIFIED |
| **[x] MRPL Industrial Case Study**| **COMPLETE** | `src/case_study/mrpl/` refinery blending MILP model, 4 what-if scenarios, mass balance. | VERIFIED |
| **[x] Solver Strategy Layer** | **COMPLETE** | `src/strategy/` rules R001–R018, dynamic range profiling, risk profiles, explainability. | VERIFIED |
| **[x] Public C++20 API** | **COMPLETE** | `src/api/solver.hpp` `KavachSolver` facade, lifetime/thread-safety guarantees, example. | VERIFIED |
| **[x] Command-Line Interface (CLI)**| **COMPLETE** | `kavach-opt` supporting solve, mrpl, benchmark, verify, presolve, --show-strategy. | VERIFIED |
| **[x] Adversarial Test Coverage**| **COMPLETE** | Infeasible, unbounded, cycling, near-singular, and corrupted tests pass across suites. | VERIFIED |
| **[x] Reproducibility Guarantee** | **COMPLETE** | Fixed-seed determinism, identical objective values, `docs/REPRODUCIBILITY.md`. | VERIFIED |
| **[x] Documentation Completeness**| **COMPLETE** | 20+ comprehensive technical specifications in `docs/` matching actual code. | VERIFIED |
| **[x] Evidence & Traceability** | **COMPLETE** | `docs/EVIDENCE_MATRIX.md` maps every claim to code, test, benchmark, and artifact. | VERIFIED |
| **[x] Continuous Integration (CI)**| **COMPLETE** | Cross-platform matrix (Ubuntu Debug/Release, macOS Debug/Release) passing cleanly. | VERIFIED |
| **[x] Memory Safety & Sanitizers**| **COMPLETE** | AddressSanitizer & UndefinedBehaviorSanitizer suite clean (329/329 tests passed). | VERIFIED |
| **[x] Dependency Sovereignty** | **COMPLETE** | `tools/check_dependencies.py` confirms 0 foreign optimization dependencies. | VERIFIED |
| **[x] Code Style & Formatting** | **COMPLETE** | `clang-format --dry-run --Werror` clean across all headers and source files. | VERIFIED |
| **[x] Demo & Presentation Guide**| **COMPLETE** | `docs/FINAL_DEMO_GUIDE.md` provides 14-step judge walkthrough with fallback modes. | VERIFIED |
| **[x] Explicit Scope & Limitations**| **COMPLETE** | `docs/SCOPE_AND_LIMITATIONS.md` explicitly lists MIQP, NLP, MINLP, non-convex as NOT IMPLEMENTED. | VERIFIED |

---

## 2. Release Gate Decision

All 27 release readiness criteria have been satisfied with zero open blockers.

**Release Gate Status: APPROVED FOR DEPLOYMENT**
