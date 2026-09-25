# Kavach-Opt Branch-and-Bound MILP Solver Documentation

## 1. Overview and Architecture

The `src/algorithms/milp` subsystem implements the sovereign, production-quality Mixed-Integer Linear Programming (MILP) Branch-and-Bound solver for Kavach-Opt, strictly fulfilling the requirements of **Section 6 [G]**, **Section 9**, **Section 11**, **Section 12**, **Section 22**, and **Section 32 (Phase 7)** from `SIH26119_Kavach-Opt_Blueprint_v2.md`.

### Core Architectural Decisions & Invariants
- **Algorithm**: Branch-and-Bound using cold-start `RevisedSimplex` LP relaxations with priority queue best-bound node selection.
- **Node Data Model**: Bound-overlay architecture (`BoundModification`). The solver modifies and restores variable bounds on a single working model in $O(1)$ memory without cloning `CanonicalModel` per node.
- **Node Selection**: Best-bound priority queue (`std::priority_queue` with `MinBestBound` / `MaxBestBound`). The node with the best relaxation bound across all unexplored nodes is processed next. Tie-breaking is deterministic via `node_id` ascending.
- **Branching Variable Selection**: Most-fractional variable selection rule ($s_j = \min(x_j - \lfloor x_j \rfloor, \lceil x_j \rceil - x_j)$). Tie-breaking is deterministic via `VarId` ascending.
- **Pruning Conditions**:
  - Bound pruning: Nodes whose relaxation bound cannot improve on the current incumbent ($z_{\text{node}} \ge z_{\text{inc}} - \text{tol}$ for minimization) are pruned immediately without branching.
  - Infeasibility pruning: Infeasible LP relaxation or contradictory child bounds ($lb > ub + \text{tol}$) are pruned immediately without queuing.
- **Safe Primal Rounding Heuristic**: Periodically rounds fractional solutions to nearest integers clamped to $[l_j, u_j]$. An independent feasibility check validates all linear constraints and bounds before updating the incumbent.
- **Honest Optimality Gap**: Computed as $\text{gap} = \frac{|z_{\text{inc}} - z_{\text{bound}}|}{\max(1.0, |z_{\text{inc}}|)}$. Never reported as zero unless the search tree is exhausted or gap $\le \text{gap\_tol}$.
- **Independent Verification Gate**: The solver **never** claims `OPTIMAL` without passing independent verification in `verification::MILPVerifier`. If integrality, feasibility, or gap conditions are violated, the reported status is honestly downgraded to `FEASIBLE` or `NUMERICAL_FAILURE`.
- **Problem Class Handling**:
  - Pure LPs (`model.is_lp()`): Bypasses B&B and routes directly to `RevisedSimplex`.
  - QP models (`model.is_qp()`): Strictly rejected with `core::SolverStatus::NUMERICAL_FAILURE`.
- **Strict Sovereignty**: **Zero external dependencies** (no SCIP, CBC, GLPK, HiGHS, CPLEX, Gurobi, or OR-Tools).

---

## 2. Solver Pipeline & Data Flow

```
                      +---------------------------------------+
                      |   CanonicalModel (const &)            |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   1. Problem Class Validation         | (Rejects QP; routes pure LP to Simplex)
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   2. Root Relaxation Solve            | (RevisedSimplex on continuous relaxation)
                      +---------------------------------------+
                                          |
                        +-----------------+-----------------+
                        | (Infeasible)                      | (Optimal)
                        v                                   v
                 [ INFEASIBLE ]               +-------------------------------+
                                              | 3. Check Root Integrality     |
                                              +-------------------------------+
                                                              |
                                            +-----------------+-----------------+
                                            | (All integer)                     | (Fractional)
                                            v                                   v
                                       [ OPTIMAL ]                +-------------------------------+
                                       (Verified)                 | 4. Priority Queue (Best Bound)|
                                                                  +-------------------------------+
                                                                                  |
                                                                                  v
                                                                  +-------------------------------+
                                                              +-->| 5. Pop Node & Prune by Bound  |
                                                              |   +-------------------------------+
                                                              |                   |
                                                              |                   v
                                                              |   +-------------------------------+
                                                              |   | 6. Apply Bound Overlay        |
                                                              |   +-------------------------------+
                                                              |                   |
                                                              |                   v
                                                              |   +-------------------------------+
                                                              |   | 7. Solve Node LP Relaxation   |
                                                              |   +-------------------------------+
                                                              |                   |
                                                              |                   v
                                                              |   +-------------------------------+
                                                              |   | 8. Restore Base Model Bounds  |
                                                              |   +-------------------------------+
                                                              |                   |
                                                              |     +-------------+-------------+
                                                              |     | (Integer)                 | (Fractional)
                                                              |     v                           v
                                                              |  Update Incumbent         Most-Fractional Branch
                                                              |  & Prune Node             Left & Right Children
                                                              |     |                           |
                                                              |     +-------------+-------------+
                                                              |                   |
                                                              +-------------------+ (Loop until queue empty or limit)
                                                                                  |
                                                                                  v
                                                                  +-------------------------------+
                                                                  | 9. Compute Honest Gap         |
                                                                  +-------------------------------+
                                                                                  |
                                                                                  v
                                                                  +-------------------------------+
                                                                  | 10. Independent Verification  | (MILPVerifier)
                                                                  +-------------------------------+
                                                                                  |
                                                                                  v
                                                                          [ Final MILPSolution ]
```

---

## 3. Mathematical Foundations

### 3.1 Node Relaxation
For a mixed-integer linear program:
$$
\begin{aligned}
\min \quad & c^T x + c_0 \\
\text{s.t.} \quad & A x \sim b \\
& l_j \le x_j \le u_j \quad \forall j \in \{1, \dots, n\} \\
& x_j \in \mathbb{Z} \quad \forall j \in I
\end{aligned}
$$
The continuous LP relaxation replaces $x_j \in \mathbb{Z}$ with $x_j \in \mathbb{R}$.

### 3.2 Most-Fractional Branching
For each variable $j \in I$ with fractional LP value $x_j^*$:
$$
\text{score}_j = \min(x_j^* - \lfloor x_j^* \rfloor, \lceil x_j^* \rceil - x_j^*)
$$
We select $j^* = \arg\max_{j \in I} \text{score}_j$. Ties are broken by choosing the smallest index $j$.

Two child subproblems are spawned:
- **Left child**: Upper bound tightened to $x_{j^*} \le \lfloor x_{j^*}^* \rfloor$.
- **Right child**: Lower bound tightened to $x_{j^*} \ge \lceil x_{j^*}^* \rceil$.

If the tightened bounds contradict ($lb > ub + \text{tol}$), the child node is pruned immediately.

### 3.3 Best-Bound Node Selection
The priority queue is keyed by:
- Minimization: $\min_{\text{node}} z_{\text{LP}}(\text{parent})$
- Maximization: $\max_{\text{node}} z_{\text{LP}}(\text{parent})$

This ensures the search expands the frontier that directly improves the global dual bound.

### 3.4 Safe Primal Rounding Heuristic
At each evaluated node (or specified frequency), candidate integer solutions are generated:
$$
x^{\text{heur}}_j = \begin{cases}
\operatorname{clamp}\left(\operatorname{round}(x^*_j), l_j, u_j\right) & \text{if } j \in I \\
x^*_j & \text{otherwise}
\end{cases}
$$
The candidate is independently validated against all constraints:
$$
\max_i \operatorname{viol}(A_i x^{\text{heur}}, b_i) \le \epsilon_{\text{primal}}
$$
If all constraints and bounds hold, and the objective improves on the incumbent, the incumbent is updated.

---

## 4. Independent Verification Layer (`MILPVerifier`)

Per **Blueprint Section 12**, the verifier **never trusts the solver's internal claims**.

`MILPVerifier` independently executes:
1. **Nonfinite scan**: Checks incumbent vector for NaNs and infinities.
2. **Bound feasibility**: Validates $l_j - \epsilon \le x_j \le u_j + \epsilon$ against original variable bounds.
3. **Primal constraint feasibility**: Recomputes $A_i x$ from scratch using original model expressions and validates against $b_i$.
4. **Integrality check**: Computes $|x_j - \operatorname{round}(x_j)|$ for all $j \in I$, ensuring violation $\le \epsilon_{\text{int}}$.
5. **Objective consistency**: Recomputes $c^T x + c_0$ directly and validates $|z_{\text{recomputed}} - z_{\text{reported}}| \le \epsilon_{\text{obj}}$.
6. **Optimality gap verification**: Recomputes $\text{gap} = \frac{|z_{\text{recomputed}} - z_{\text{bound}}|}{\max(1.0, |z_{\text{recomputed}}|)}$.
   - If solver claims `OPTIMAL` with gap $> \text{gap\_tol}$, status is downgraded to `FEASIBLE`.
   - If integrality or primal feasibility fails, status is downgraded to `NUMERICAL_FAILURE`.

---

## 5. Verification & Adversarial Results

The test suite in `tests/unit/test_phase7_milp.cpp` covers 37 comprehensive test cases:

| Test Group | Coverage | Result |
|---|---|---|
| **A. Pure LP Bypass** | Continuous models routed directly to Simplex | PASS |
| **B. Basic MILP** | Small integer-constrained models with known optima | PASS |
| **C. Binary Knapsack** | 0-1 knapsack combinatorial optimization | PASS |
| **D. Mixed-Integer** | Joint continuous + integer variable formulations | PASS |
| **E. QP Rejection** | Quadratic models rejected with `NUMERICAL_FAILURE` | PASS |
| **F. Maximization** | Correct dual bound & pruning under MAXIMIZE | PASS |
| **G. Infeasible Relaxations** | Contradictory bounds and equations detected | PASS |
| **H. Adversarial Integer Infeasibility** | Section 22 integer-infeasible models | PASS |
| **I. Node Limits** | Safe termination with `LIMIT_REACHED` / `FEASIBLE` | PASS |
| **J. Time Limits** | Safe timeout handling without false OPTIMAL | PASS |
| **K. Honest Gap** | Nonzero gap correctly reported under limits | PASS |
| **L. Determinism** | Byte-for-byte identical solutions across runs | PASS |
| **M. Verifier Fault Injection** | Injected infeasibility, non-integrality, bad objective caught | PASS |
| **N. Verifier Integration** | `KavachLPVerifier::verify_milp` integration | PASS |
| **P. End-to-End Pipeline** | Model $\to$ B&B $\to$ Incumbent $\to$ Verifier | PASS |

**Full Test Suite Summary:**
- Debug: **171/171 passed**
- Release: **171/171 passed**
- ASan + UBSan: **171/171 passed (0 leaks, 0 undefined behavior)**
- Sovereignty: **0 forbidden dependencies**
- Code Formatting: **100% clang-format compliant**
