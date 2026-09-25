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
| Timestamp (UTC) | 2026-09-25T20:05:37Z |
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
| LP | 13 | 13 | 0 | 0 | 0 | 0 |

## 4. Per-Instance Detailed Results

| Case ID | Type | Vars | Constrs | Status | Verified | Objective | Ref Objective | Rel Err | Time (s) | Iters/Nodes |
|---|---|---|---|---|---|---|---|---|---|---|
| `netlib-afiro` | LP | 32 | 27 | OPTIMAL | VERIFIED_OPTIMAL | -464.75314 | -464.75314 | 6.15e-12 | 0.0016 | 16 it |
| `netlib-adlittle` | LP | 97 | 56 | OPTIMAL | VERIFIED_OPTIMAL | 225494.96316238 | 225494.96316000 | 1.06e-11 | 0.0035 | 100 it |
| `netlib-blend` | LP | 83 | 74 | OPTIMAL | VERIFIED_OPTIMAL | -30.81214985 | -30.81214985 | 5.57e-12 | 0.0291 | 335 it |
| `netlib-sc50a` | LP | 48 | 50 | OPTIMAL | VERIFIED_OPTIMAL | -64.57507706 | -64.57507706 | 6.74e-12 | 0.0010 | 51 it |
| `netlib-sc50b` | LP | 48 | 50 | OPTIMAL | VERIFIED_OPTIMAL | -70.00000000 | -70.00000000 | 0.00e+00 | 0.0009 | 53 it |
| `netlib-share2b` | LP | 79 | 96 | OPTIMAL | VERIFIED_OPTIMAL | -415.73224074 | -415.73224074 | 3.41e-12 | 0.0062 | 138 it |
| `netlib-beaconfd` | LP | 262 | 173 | OPTIMAL | VERIFIED_OPTIMAL | 33592.48580720 | 33592.48580700 | 5.95e-12 | 0.0568 | 526 it |
| `netlib-sc105` | LP | 103 | 105 | OPTIMAL | VERIFIED_OPTIMAL | -52.20206121 | -52.20206121 | 5.61e-12 | 0.0039 | 111 it |
| `netlib-lotfi` | LP | 308 | 153 | OPTIMAL | VERIFIED_OPTIMAL | -25.26470606 | -25.26470606 | 4.75e-12 | 0.0164 | 198 it |
| `netlib-share1b` | LP | 225 | 117 | OPTIMAL | VERIFIED_OPTIMAL | -76589.31857919 | -76589.31857900 | 2.43e-12 | 0.0227 | 345 it |
| `netlib-kb2` | LP | 41 | 43 | OPTIMAL | VERIFIED_OPTIMAL | -1749.90012991 | -1749.90012990 | 3.55e-12 | 0.0031 | 124 it |
| `netlib-stocfor1` | LP | 111 | 117 | OPTIMAL | VERIFIED_OPTIMAL | -41131.97621944 | -41131.97621900 | 1.06e-11 | 0.0155 | 392 it |
| `netlib-boeing2` | LP | 143 | 185 | OPTIMAL | VERIFIED_OPTIMAL | -315.01872802 | -315.01872802 | 1.52e-11 | 0.0797 | 335 it |

## 5. Aggregate Summary

All aggregate metrics are strictly derived from the per-instance records above:

- **Total Instances Attempted**: 13
- **Verified Optimal**: 13 (100.0%)
- **Infeasible Certifications**: 0
- **Timeouts / Limits Hit**: 0
- **Verification / Numerical Failures**: 0
- **Median Solve Runtime**: 0.0062 s (Min: 0.0009 s, Max: 0.0797 s)
- **Median Relative Objective Error**: 5.61e-12
- **Mean Simplex Iterations**: 209.5

## 6. Failures, Timeouts, and Unsupported Cases

Zero failures, timeouts, or unhandled errors detected across all tested cases.

## 7. Independent Verification Summary

Every reported `OPTIMAL` solution is independently verified by `LPVerifier`, `MILPVerifier`, or `QPVerifier` without solver state reuse:

| Case ID | Verifier | Primal Residual | Dual Residual | Integrality Viol | Certified |
|---|---|---|---|---|---|
| `netlib-afiro` | LPVerifier | 5.68e-14 | 0.00e+00 | 0.0 | VERIFIED_OPTIMAL |
| `netlib-adlittle` | LPVerifier | 4.55e-13 | 0.00e+00 | 0.0 | VERIFIED_OPTIMAL |
| `netlib-blend` | LPVerifier | 1.42e-14 | 0.00e+00 | 0.0 | VERIFIED_OPTIMAL |
| `netlib-sc50a` | LPVerifier | 5.68e-14 | 0.00e+00 | 0.0 | VERIFIED_OPTIMAL |
| `netlib-sc50b` | LPVerifier | 5.68e-14 | 0.00e+00 | 0.0 | VERIFIED_OPTIMAL |
| `netlib-share2b` | LPVerifier | 3.91e-13 | 0.00e+00 | 0.0 | VERIFIED_OPTIMAL |
| `netlib-beaconfd` | LPVerifier | 4.64e-10 | 0.00e+00 | 0.0 | VERIFIED_OPTIMAL |
| `netlib-sc105` | LPVerifier | 1.14e-13 | 0.00e+00 | 0.0 | VERIFIED_OPTIMAL |
| `netlib-lotfi` | LPVerifier | 2.62e-10 | 0.00e+00 | 0.0 | VERIFIED_OPTIMAL |
| `netlib-share1b` | LPVerifier | 1.79e-10 | 0.00e+00 | 0.0 | VERIFIED_OPTIMAL |
| `netlib-kb2` | LPVerifier | 1.04e-12 | 0.00e+00 | 0.0 | VERIFIED_OPTIMAL |
| `netlib-stocfor1` | LPVerifier | 9.09e-13 | 0.00e+00 | 0.0 | VERIFIED_OPTIMAL |
| `netlib-boeing2` | LPVerifier | 5.15e-13 | 0.00e+00 | 0.0 | VERIFIED_OPTIMAL |

## 8. External Reference Solver Comparison (HiGHS)

| Case ID | Kavach Status | Kavach Obj | HiGHS Status | HiGHS Obj | Kavach Time (s) | HiGHS Time (s) |
|---|---|---|---|---|---|---|
| `netlib-afiro` | OPTIMAL | -464.753143 | UNAVAILABLE | - | 0.0016 | - |
| `netlib-adlittle` | OPTIMAL | 225494.963162 | UNAVAILABLE | - | 0.0035 | - |
| `netlib-blend` | OPTIMAL | -30.812150 | UNAVAILABLE | - | 0.0291 | - |
| `netlib-sc50a` | OPTIMAL | -64.575077 | UNAVAILABLE | - | 0.0010 | - |
| `netlib-sc50b` | OPTIMAL | -70.000000 | UNAVAILABLE | - | 0.0009 | - |
| `netlib-share2b` | OPTIMAL | -415.732241 | UNAVAILABLE | - | 0.0062 | - |
| `netlib-beaconfd` | OPTIMAL | 33592.485807 | UNAVAILABLE | - | 0.0568 | - |
| `netlib-sc105` | OPTIMAL | -52.202061 | UNAVAILABLE | - | 0.0039 | - |
| `netlib-lotfi` | OPTIMAL | -25.264706 | UNAVAILABLE | - | 0.0164 | - |
| `netlib-share1b` | OPTIMAL | -76589.318579 | UNAVAILABLE | - | 0.0227 | - |
| `netlib-kb2` | OPTIMAL | -1749.900130 | UNAVAILABLE | - | 0.0031 | - |
| `netlib-stocfor1` | OPTIMAL | -41131.976219 | UNAVAILABLE | - | 0.0155 | - |
| `netlib-boeing2` | OPTIMAL | -315.018728 | UNAVAILABLE | - | 0.0797 | - |

## 9. GPU Acceleration & Telemetry Analysis

CUDA GPU hardware is not available in the current execution environment (Darwin 27.0.0 (arm64)). As specified in Blueprint Section 15 & Section 22, CPU fallback is active and no GPU speedup is claimed.

## 10. Reproducibility Instructions

To reproduce the exact results reported in this document:

```bash
# 1. Build Kavach-Opt in Release mode
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j

# 2. Run the full benchmark suite with deterministic seed
./build-release/src/cli/kavach-opt benchmark --suite netlib --output benchmark_results/
```
