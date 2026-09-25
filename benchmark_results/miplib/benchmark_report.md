# Kavach-Opt Benchmark & Reproducibility Report

## 1. Executive Methodology

This report presents an objective evaluation of Kavach-Opt across standard mathematical programming benchmark instances.
Key methodological rules:
- Every attempted instance is recorded with first-class status preservation.
- An instance is certified `OPTIMAL` only if confirmed by independent verification layers.
- Time limit and numerical safeguards are enforced deterministically.
- External reference solver comparisons (HiGHS) execute in completely isolated subprocesses.

## 2. Hardware and Software Environment

| Property | Value |
|---|---|
| Solver | Kavach-Opt v0.1.0 |
| Git Commit | `0000000` |
| Timestamp (UTC) | 2026-09-25T20:09:51Z |
| Operating System | Darwin 27.0.0 (arm64) |
| Hostname | MaddyMac |
| CPU Model | arm64 (8 logical cores) |
| GPU Accelerator | CUDA unavailable (CPU fallback) |
| C++ Compiler | AppleClang 21.0.0.21000334 |
| Build Type | Release |
| Random Seed | 42 |

## 3. Suite Coverage

| Problem Type | Attempted | Verified Optimal | Feasible | Infeasible | Timeout | Failed |
|---|---|---|---|---|---|---|
| MILP | 3 | 2 | 0 | 1 | 0 | 0 |

## 4. Per-Instance Detailed Results

| Case ID | Type | Vars | Constrs | Status | Verified | Objective | Ref Objective | Rel Err | Time (s) | Iters/Nodes |
|---|---|---|---|---|---|---|---|---|---|---|
| `miplib-knapsack` | MILP | 6 | 1 | OPTIMAL | VERIFIED_OPTIMAL | 41 | 41 | 0.00e+00 | 0.0014 | 19 nd |
| `miplib-facility_loc` | MILP | 9 | 5 | OPTIMAL | VERIFIED_OPTIMAL | 180.00000000 | - | - | 0.0005 | 3 nd |
| `miplib-infeasible` | MILP | 2 | 2 | INFEASIBLE | UNVERIFIED | - | - | - | 0.0006 | 3 nd |

## 5. Aggregate Summary

All aggregate metrics are strictly derived from the per-instance records above:

- **Total Instances Attempted**: 3
- **Verified Optimal**: 2 (66.7%)
- **Infeasible Certifications**: 1
- **Timeouts / Limits Hit**: 0
- **Verification / Numerical Failures**: 0
- **Median Solve Runtime**: 0.0006 s (Min: 0.0005 s, Max: 0.0014 s)
- **Median Relative Objective Error**: 0.00e+00
- **Mean Simplex Iterations**: 0.0

## 6. Failures, Timeouts, and Unsupported Cases

Zero failures, timeouts, or unhandled errors detected across all tested cases.

## 7. Independent Verification Summary

Every reported `OPTIMAL` solution is independently verified by `LPVerifier`, `MILPVerifier`, or `QPVerifier` without solver state reuse:

| Case ID | Verifier | Primal Residual | Dual Residual | Integrality Viol | Certified |
|---|---|---|---|---|---|
| `miplib-knapsack` | MILPVerifier | 0.00e+00 | - | 0.00e+00 | VERIFIED_OPTIMAL |
| `miplib-facility_loc` | MILPVerifier | 0.00e+00 | - | 0.00e+00 | VERIFIED_OPTIMAL |

## 8. External Reference Solver Comparison (HiGHS)

HiGHS reference comparison was not enabled for this run (run with `--compare highs` to activate).

## 9. GPU Acceleration & Telemetry Analysis

CUDA GPU hardware is not available in the current execution environment (Darwin 27.0.0 (arm64)). As specified in Blueprint Section 15 & Section 22, CPU fallback is active and no GPU speedup is claimed.

## 10. Reproducibility Instructions

To reproduce the exact results reported in this document:

```bash
# 1. Build Kavach-Opt in Release mode
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j

# 2. Run the full benchmark suite with deterministic seed
./build-release/src/cli/kavach-opt benchmark --suite miplib_subset --output benchmark_results/
```
