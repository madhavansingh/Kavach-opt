# Kavach-Opt Frontend Data Truth & Integrity Policy

## 1. Core Principle: Zero Fabricated Telemetry

The Kavach-Opt Demonstration Interface is built on an unwavering commitment to mathematical authenticity:
**Zero numbers, execution times, iterations, or certificates displayed in the UI may be fabricated, guessed, or simulated.**

Every metric rendered across all screens originates from:
1. Direct, live process invocation of the C++ binary (`./build-release/kavach-opt`), OR
2. Authentic repository benchmark artifacts produced by verified test runs (`benchmark_results/` and `examples/mrpl/results/`).

---

## 2. Independent Post-Solve Verification Trust Model

Commercial optimization software frequently relies on internal convergence flags (e.g. status codes reported by simplex bases) that can conceal severe numerical drift, basis degradation, or roundoff errors.

Kavach-Opt implements an air-gapped, zero-trust mathematical verifier:
- **Independent Matrix Re-multiplication**: The verifier discards all internal basis factorizations and scaling matrices, directly reading raw row vectors and decision solutions in 64-bit IEEE 754 precision.
- **Problem-Class Invariants**:
  - **Linear Programming (LP)**:
    - Primal Feasibility: \(\|Ax - b\|_\infty \le 10^{-6}\)
    - Variable Box Bounds: \(l \le x^* \le u\)
    - Dual Multiplier Feasibility: \(\|A^T y + s - c\|_\infty \le 10^{-6}\)
    - Complementary Slackness: \(|x_j \cdot s_j| \le 10^{-6}\)
    - Objective Consistency: \(|c^T x^* - z^*| \le 10^{-6}\)
  - **Mixed-Integer Linear Programming (MILP)**:
    - Integrality Violation: \(\max_{j \in I} |x_j^* - \text{round}(x_j^*)| \le 10^{-5}\)
    - Primal Row Feasibility: \(\|Ax - b\|_\infty \le 10^{-6}\)
  - **Quadratic Programming (QP)**:
    - KKT Stationarity: \(\|Q x^* + c - A^T y^*\|_\infty \le 10^{-6}\)
    - Primal Feasibility: \(\|Ax - b\|_\infty \le 10^{-6}\)

If any invariant fails, the solution status is marked `VERIFICATION_FAILED` regardless of internal solver flags.

---

## 3. Hardware Backend Transparency & Fallback Disclosure

Kavach-Opt does not claim speculative GPU acceleration:
1. **Scoped GPU Kernel**: The CUDA subsystem is strictly limited to batched sparse matrix-vector multiplication (SpMV).
2. **Honest Reporting**:
   - The UI explicitly differentiates between **Requested Backend** and **Executed Backend**.
   - If a user requests `CUDA_GPU` on a host architecture without CUDA hardware (such as Apple Silicon), the interface highlights an amber notice detailing automatic CPU fallback per Strategy Rule R012.
   - Solver execution never fails silently or claims GPU execution when running on CPU.

---

## 4. Industrial Case Study Synthetic Data Disclosure

In strict compliance with SIH26119 problem guidelines:
- All crude assays, unit operational limits, product specifications, and economic values utilized in the MRPL refinery case study are **explicitly synthetic and illustrative**.
- No proprietary operational data from Mangalore Refinery and Petrochemicals Limited is embedded in the software.
- The prominent disclosure banner on the MRPL interface is mandatory and cannot be hidden or disabled.
