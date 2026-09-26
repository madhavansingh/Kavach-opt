#!/usr/bin/env python3
import sys

with open("README.md", "r") as f:
    readme = f.read()

expand_toc = """
## Table of Contents

1. [Mission-Critical Mathematical Optimization for Sovereign Infrastructure](#1-mission-critical-mathematical-optimization-for-sovereign-infrastructure)
2. [Sovereign Architecture & Dependency Policy](#2-sovereign-architecture--dependency-policy)
3. [Core Mathematical Problem Classes](#3-core-mathematical-problem-classes)
4. [Feature Matrix & Algorithmic Capabilities](#4-feature-matrix--algorithmic-capabilities)
5. [High-Level System Architecture & Component Topology](#5-high-level-system-architecture--component-topology)
6. [Ingestion & Input Formats](#6-ingestion--input-formats)
7. [Sparse Linear Algebra & Numerical Infrastructure](#7-sparse-linear-algebra--numerical-infrastructure)
8. [SparseLU Direct Solver with Markowitz Pivoting](#8-sparselu-direct-solver-with-markowitz-pivoting)
9. [Ruiz Equilibration & Conditioning](#9-ruiz-equilibration--conditioning)
10. [Presolve Reduction Pipeline & Postsolve Stack](#10-presolve-reduction-pipeline--postsolve-stack)
11. [Two-Phase Revised Simplex Algorithm (LP)](#11-two-phase-revised-simplex-algorithm-lp)
12. [Dual Simplex & Basis Recovery Engine](#12-dual-simplex--basis-recovery-engine)
13. [Branch-and-Bound Engine (MILP)](#13-branch-and-bound-engine-milp)
14. [Polyhedral Cutting Plane Suite (GMI & MIR)](#14-polyhedral-cutting-plane-suite-gmi--mir)
15. [Primal Active-Set Solver for Convex QP](#15-primal-active-set-solver-for-convex-qp)
16. [Convex Mixed-Integer Quadratic Programming (Convex MIQP)](#16-convex-mixed-integer-quadratic-programming-convex-miqp)
17. [Air-Gapped Zero-Trust Independent Verification](#17-air-gapped-zero-trust-independent-verification)
18. [Deterministic Strategy Engine & Rule Topology (R001–R018)](#18-deterministic-strategy-engine--rule-topology-r001r018)
19. [Execution Backends & Work Partitioning](#19-execution-backends--work-partitioning)
20. [Scoped CUDA Numerical Acceleration (7 FP64 Kernels)](#20-scoped-cuda-numerical-acceleration-7-fp64-kernels)
21. [Transfer-Inclusive GPU Timing Policy & Crossover Boundary](#21-transfer-inclusive-gpu-timing-policy--crossover-boundary)
22. [Empirical Benchmark Results (Netlib LP Corpus — 13 Instances)](#22-empirical-benchmark-results-netlib-lp-corpus--13-instances)
23. [Mixed-Integer Benchmark Results (MIPLIB Subset & Ablations)](#23-mixed-integer-benchmark-results-miplib-subset--ablations)
24. [Convex QP & MIQP Benchmark Results](#24-convex-qp--miqp-benchmark-results)
25. [Industrial Operations Planning Case Study (MRPL Refinery)](#25-industrial-operations-planning-case-study-mrpl-refinery)
26. [Comparative Analysis Against Reference Solvers (HiGHS Out-of-Process)](#26-comparative-analysis-against-reference-solvers-highs-out-of-process)
27. [Comprehensive Command-Line Interface (`kavach-opt`) Reference](#27-comprehensive-command-line-interface-kavach-opt-reference)
28. [Public C++ API Reference (`include/kavach/solver.hpp`)](#28-public-c-api-reference-includekavachsolverhpp)
29. [Build Configuration, Prerequisites & Toolchains](#29-build-configuration-prerequisites--toolchains)
30. [Complete 19-Suite Test Framework & Sanitizer Verification (387 Tests)](#30-complete-19-suite-test-framework--sanitizer-verification-387-tests)
31. [Known Scope, Boundaries & Engineering Limitations](#31-known-scope-boundaries--engineering-limitations)
32. [Comprehensive Mathematical Notation Guide](#32-comprehensive-mathematical-notation-guide)
33. [Industrial Operations FAQ & Deployment Guide](#33-industrial-operations-faq--deployment-guide)
34. [Project Heritage, Architecture Team & Licensing](#34-project-heritage-architecture-team--licensing)
"""

expand_notation = """
## 32. Comprehensive Mathematical Notation Guide

| Symbol | Mathematical Domain | Operational Meaning in Kavach-Opt |
| :--- | :--- | :--- |
| $n$ | $\\mathbb{N}$ | Number of decision variables in mathematical model |
| $m$ | $\\mathbb{N}$ | Number of constraints (rows) in constraint matrix |
| $\\text{nnz}$ | $\\mathbb{N}$ | Number of non-zero coefficients in constraint matrix $A$ |
| $x$ | $\\mathbb{R}^n$ | Primal decision variable vector |
| $c$ | $\\mathbb{R}^n$ | Objective cost gradient vector |
| $c_0$ | $\\mathbb{R}$ | Constant scalar objective offset |
| $Q$ | $\\mathbb{R}^{n \\times n}$ | Symmetric positive semidefinite Hessian matrix ($Q \\succeq 0$) |
| $A$ | $\\mathbb{R}^{m \\times n}$ | Constraint matrix in CSR / CSC representation |
| $b$ | $\\mathbb{R}^m$ | Constraint right-hand side vector |
| $l, u$ | $(\\mathbb{R} \\cup \\{\\pm\\infty\\})^n$ | Lower and upper variable bound vectors |
| $I$ | $\\subseteq \\{1, \\dots, n\\}$ | Index set of integer-restricted decision variables |
| $y$ | $\\mathbb{R}^m$ | Dual multiplier vector (shadow prices on constraints) |
| $\\bar{c}$ | $\\mathbb{R}^n$ | Reduced cost vector: $\\bar{c}_j = c_j - y^T a_j$ |
| $\\lambda$ | $\\mathbb{R}^{|\\mathcal{W}|}$ | Active constraint Lagrange multipliers (QP) |
| $\\mu$ | $\\mathbb{R}^n$ | Active bound Lagrange multipliers (QP) |
| $B$ | $\\mathbb{R}^{m \\times m}$ | Simplex basis matrix |
| $\\mathcal{B}, \\mathcal{N}$ | Index sets | Indices of basic and nonbasic decision variables |
| $P, Q$ | $\\mathbb{R}^{m \\times m}$ | Row and column permutation matrices from `SparseLU` |
| $L, U$ | $\\mathbb{R}^{m \\times m}$ | Unit lower-triangular and upper-triangular factors |
| $D_r, D_c$ | Diagonal $\\mathbb{R}^{m \\times m}, \\mathbb{R}^{n \\times n}$ | Row and column scaling matrices from Ruiz equilibration |
| $\\rho$ | $[0, 1]$ | Constraint matrix density: $\\text{nnz} / (m \\cdot n)$ |
| $\\kappa_{\\text{range}}$ | $\\mathbb{R}_{>0}$ | Matrix dynamic range ratio $= a_{\\max} / a_{\\min}$ |
| $\\alpha_{\\text{int}}$ | $[0, 1]$ | Integrality ratio of model variables |
| $\\mathcal{W}_k$ | Index set | Active constraint working set at iteration $k$ (QP) |
| $p_k$ | $\\mathbb{R}^n$ | Search step direction at iteration $k$ (QP) |
| $g_k$ | $\\mathbb{R}^n$ | Quadratic objective gradient: $Q x_k + c$ |
| $\\theta^*$ | $\\mathbb{R}_+$ | Minimum ratio step length in Simplex pivot |
| $\\alpha_k$ | $[0, 1]$ | Step length to next blocking constraint (QP) |
| $z_{\\text{inc}}$ | $\\mathbb{R}$ | Best known integer objective incumbent |
| $z_{\\text{bound}}$ | $\\mathbb{R}$ | Global dual bound across active branch-and-bound queue |
| $B$ | $\\mathbb{N}$ | Batch size in batched numerical operations |
| $T_{\\text{H2D}}, T_{\\text{D2H}}$ | $\\mathbb{R}_+$ | Host-to-Device and Device-to-Host PCIe transfer times |
| $T_{\\text{kernel}}$ | $\\mathbb{R}_+$ | Active GPU kernel execution duration |
| $T_{\\text{GPU}}$ | $\\mathbb{R}_+$ | Transfer-inclusive total GPU execution time |
"""

expand_faq = """
## 33. Industrial Operations FAQ & Deployment Guide

### Q1: How does Kavach-Opt guarantee zero memory leaks in continuous 24/7 loops?
Kavach-Opt enforces strict RAII (Resource Acquisition Is Initialization) throughout its memory layer. Iterative algorithms (`RevisedSimplexSolver`, `ActiveSetQpSolver`) pre-allocate workspace buffers (`activities`, `FTRAN vectors`, `dual direction buffers`) during initialization and reuse them across iterations. The entire 387-test suite runs cleanly under AddressSanitizer and LeakSanitizer with zero memory leaks.

### Q2: How does the independent verifier handle floating-point tolerances?
The independent verifier operates with conservative tolerances: $\\epsilon_{\\text{primal}} = 10^{-7}$, $\\epsilon_{\\text{dual}} = 10^{-7}$, $\\epsilon_{\\text{int}} = 10^{-6}$, and $\\epsilon_{\\text{comp}} = 10^{-7}$. These tolerances are at least 9 orders of magnitude larger than IEEE-754 double precision machine epsilon ($\\approx 2.22 \\times 10^{-16}$), ensuring that minor floating-point cancellation is tolerated while true constraint violations are intercepted.

### Q3: What happens if a user submits an infeasible or unbounded model?
Kavach-Opt treats infeasibility and unboundedness as first-class, certifiable mathematical outcomes:
- **Primal Infeasibility:** Certified constructively via Phase I artificial variables ($\sum w_i > 10^{-7}$) or an unbounded dual ray (Farkas Lemma certificate $A^T y \ge 0, y \ge 0, b^T y < 0$).
- **Dual Unboundedness:** Certified constructively via a primal feasible ray $d$ satisfying $A d = 0, d \ge 0, c^T d < 0$.

### Q4: How is Kavach-Opt integrated into Python or Rust analytical pipelines?
Kavach-Opt provides standard C linkage bindings via `include/kavach/c_api.h` (or direct subprocess execution via JSON). Python integration is achieved in two lines using `ctypes` or `subprocess`:
```python
import subprocess, json

cmd = ["kavach-opt", "solve", "model.mps", "--json"]
result = json.loads(subprocess.check_output(cmd))
print(f"Optimal objective: {result['result']['objective']}")
```

### Q5: Why is GPU acceleration scoped to batched operations rather than the entire solver?
Simplex and Branch-and-Bound algorithms are sequential, control-flow heavy, and require irregular sparse graph traversals with frequent synchronization. Amdahl's Law and PCIe bus latencies prove that executing single-pivot Simplex steps on a GPU is significantly slower than CPU cache execution. Kavach-Opt scopes CUDA acceleration to high-throughput batched matrix-vector products ($B \ge 128$), where the GPU's memory bandwidth provides genuine, transfer-inclusive speedups.
"""

# Insert Table of Contents right after Title & Badges
readme = readme.replace("---\n\n## 1. Mission-Critical", expand_toc + "\n---\n\n## 1. Mission-Critical")
# Insert Notation and FAQ right before Project Heritage
readme = readme.replace("## 32. Project Heritage", expand_notation + "\n---\n\n" + expand_faq + "\n---\n\n## 34. Project Heritage")

with open("README.md", "w") as f:
    f.write(readme.strip() + "\n")

line_count = len(readme.strip().splitlines()) + 1
print(f"Final check: README.md now has {line_count} lines.")
