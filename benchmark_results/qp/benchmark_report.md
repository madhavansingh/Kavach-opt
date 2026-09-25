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
| QP | 2 | 2 | 0 | 0 | 0 | 0 |

## 4. Per-Instance Detailed Results

| Case ID | Type | Vars | Constrs | Status | Verified | Objective | Ref Objective | Rel Err | Time (s) | Iters/Nodes |
|---|---|---|---|---|---|---|---|---|---|---|
| `qp-portfolio` | QP | 3 | 1 | OPTIMAL | VERIFIED_OPTIMAL | 0.31641206 | - | - | 0.0007 | 5 it |
| `qp-constrained_ls` | QP | 2 | 1 | OPTIMAL | VERIFIED_OPTIMAL | -5.33333333 | - | - | 0.0001 | 5 it |

## 5. Aggregate Summary

All aggregate metrics are strictly derived from the per-instance records above:

- **Total Instances Attempted**: 2
- **Verified Optimal**: 2 (100.0%)
- **Infeasible Certifications**: 0
- **Timeouts / Limits Hit**: 0
- **Verification / Numerical Failures**: 0
- **Median Solve Runtime**: 0.0007 s (Min: 0.0001 s, Max: 0.0007 s)
- **Median Relative Objective Error**: 0.00e+00
- **Mean Simplex Iterations**: 5.0

## 6. Failures, Timeouts, and Unsupported Cases

Zero failures, timeouts, or unhandled errors detected across all tested cases.

## 7. Independent Verification Summary

Every reported `OPTIMAL` solution is independently verified by `LPVerifier`, `MILPVerifier`, or `QPVerifier` without solver state reuse:

| Case ID | Verifier | Primal Residual | Dual Residual | Integrality Viol | Certified |
|---|---|---|---|---|---|
| `qp-portfolio` | QPVerifier | 0.00e+00 | 0.00e+00 | 0.0 | VERIFIED_OPTIMAL |
| `qp-constrained_ls` | QPVerifier | 0.00e+00 | 0.00e+00 | 0.0 | VERIFIED_OPTIMAL |

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
./build-release/src/cli/kavach-opt benchmark --suite qp --output benchmark_results/
```
