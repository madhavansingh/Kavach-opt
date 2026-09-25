# Independent Verification Subsystem [Y] — Technical Documentation

## 1. Overview and Independence Contract

In critical defense, sovereign energy, and mission-critical optimization infrastructure (such as refinery operations at MRPL), numerical solvers must never be taken on faith. An optimization engine operating with tolerances, iterative matrix factorizations, and floating-point arithmetic can suffer from numerical drift, ill-conditioned bases, cycling, or incorrect solution reconstruction.

The **Independent Verification Layer (`[Y]`)** serves as an impartial, zero-trust mathematical judge. It enforces the foundational architecture rule of **Kavach-Opt Blueprint v2 Section 11 & 12**:

> **"The solver must never report `OPTIMAL` unless the independent verifier [Y] has confirmed feasibility and optimality conditions."**

### The Independence Boundary Contract
The verification subsystem (`src/verification/`) operates under a strict isolation contract:
1. **Zero Solver Re-use**: The verifier never calls solver-internal routines:
   - No LP standard-form transformation helpers
   - No simplex basis routines (`SimplexBasis`, `SparseLU`)
   - No solver-internal residual calculation functions
   - No presolve-internal reduction or reconstruction logic
2. **Elementary Mathematical Kernel**: Every check ($A x$, bounds, $c - A^T y$, slack $\times$ dual) is evaluated directly from scratch from the original (pre-presolve, unscaled) `CanonicalModel` using basic arithmetic.
3. **Strict Sovereignty**: Zero external dependencies (no third-party solver libraries like CPLEX, Gurobi, HiGHS, GLPK, SuiteSparse, or Eigen).

---

## 2. Mathematical Verification Criteria

Given an original optimization problem in `CanonicalModel`:

$$\begin{aligned}
\min \quad & c^T x + c_0 \\
\text{s.t.} \quad & A_i x \le b_i, \quad i \in \mathcal{I}_{\le} \\
& A_i x \ge b_i, \quad i \in \mathcal{I}_{\ge} \\
& A_i x = b_i, \quad i \in \mathcal{I}_{=} \\
& l_j \le x_j \le u_j, \quad j = 1, \dots, n
\end{aligned}$$

(or the corresponding maximization problem $\max c^T x + c_0$), the verifier performs six orthogonal checks:

### 2.1 Non-Finite Sanity Check
Before performing any algebraic calculations, all solution vectors ($x$, $y$, $rc$) are scanned. Any `NaN`, `+Inf`, or `-Inf` immediately flags a failure:
$$\forall k, \quad \text{std::isfinite}(v_k) == \text{true}$$
If violated, the certified status is downgraded directly to `core::SolverStatus::NUMERICAL_FAILURE`.

### 2.2 Bound Feasibility
For every structural variable $j$:
$$l_j - \epsilon_{\text{bound}} \le x_j \le u_j + \epsilon_{\text{bound}}$$
where $\epsilon_{\text{bound}}$ is the configured `bound_feasibility_tol`.

### 2.3 Primal Constraint Feasibility
For each constraint $i \in \{1, \dots, m\}$:
$$\text{activity}_i = A_{i, \cdot} x = \sum_{j \in \text{terms}_i} a_{ij} x_j$$
- **Less-or-Equal**: $A_{i, \cdot} x - b_i \le \epsilon_{\text{primal}}$
- **Greater-or-Equal**: $b_i - A_{i, \cdot} x \le \epsilon_{\text{primal}}$
- **Equality**: $|A_{i, \cdot} x - b_i| \le \epsilon_{\text{primal}}$

### 2.4 Dual Feasibility & Reduced-Cost Stationarity
The independent reduced cost for variable $j$ is computed directly as:
$$\hat{rc}_j = c_j - \sum_{i=1}^m A_{ij} y_i$$
At optimality, the Kuhn-Tucker stationarity conditions with respect to variable bounds demand:
- **Fixed variables** ($|u_j - l_j| \le \epsilon$): $\hat{rc}_j$ is unrestricted.
- **Strictly interior / free variables** ($l_j + \epsilon < x_j < u_j - \epsilon$):
  $$|\hat{rc}_j| \le \epsilon_{\text{dual}}$$
- **Lower-bound active only** ($x_j \le l_j + \epsilon$):
  - Minimization: $\hat{rc}_j \ge -\epsilon_{\text{dual}}$
  - Maximization: $\hat{rc}_j \le \epsilon_{\text{dual}}$
- **Upper-bound active only** ($x_j \ge u_j - \epsilon$):
  - Minimization: $\hat{rc}_j \le \epsilon_{\text{dual}}$
  - Maximization: $\hat{rc}_j \ge -\epsilon_{\text{dual}}$

Furthermore, constraint dual multipliers $y_i$ must have correct signs:
- **$\le$ constraints**:
  - Minimization: $y_i \le \epsilon_{\text{dual}}$
  - Maximization: $y_i \ge -\epsilon_{\text{dual}}$
- **$\ge$ constraints**:
  - Minimization: $y_i \ge -\epsilon_{\text{dual}}$
  - Maximization: $y_i \le \epsilon_{\text{dual}}$
- **$=$ constraints**: $y_i$ is unrestricted.

### 2.5 Complementary Slackness
For each inequality constraint $i$:
$$\text{slack}_i = \begin{cases}
b_i - A_{i, \cdot} x & \text{if } i \in \mathcal{I}_{\le} \\
A_{i, \cdot} x - b_i & \text{if } i \in \mathcal{I}_{\ge} \\
0 & \text{if } i \in \mathcal{I}_{=}
\end{cases}$$
The complementary slackness residual must satisfy:
$$|\text{slack}_i \cdot y_i| \le \epsilon_{\text{comp}}$$

### 2.6 Objective Consistency
The true objective is independently evaluated from the original model definition:
$$\text{obj}_{\text{recomputed}} = c_0 + \sum_{j \in \text{obj\_terms}} c_j x_j$$
The absolute and relative discrepancy with the solver's reported value $\text{obj}_{\text{reported}}$ must satisfy:
$$|\text{obj}_{\text{recomputed}} - \text{obj}_{\text{reported}}| \le \epsilon_{\text{obj\_abs}} \quad \lor \quad \frac{|\text{obj}_{\text{recomputed}} - \text{obj}_{\text{reported}}|}{\max(1.0, |\text{obj}_{\text{reported}}|)} \le \epsilon_{\text{obj\_rel}}$$

---

## 3. Subsystem Architecture

```
                      +----------------------------------+
                      |         CanonicalModel           |
                      |  (Original, pre-presolve, unscaled)
                      +-----------------+----------------+
                                        |
                                        | (Read-Only)
                                        v
+-----------------------+     +--------------------------+
|      LPSolution       | --> |        LPVerifier        |
| (x, y, rc, obj, stat) |     | (verification/lp_verifier)
+-----------------------+     +-------------+------------+
                                            |
                                            v
                              +--------------------------+
                              |   LPVerificationReport   |
                              |--------------------------|
                              | - primal_feasibility     |
                              | - bound_feasibility      |
                              | - dual_feasibility       |
                              | - complementary_slackness|
                              | - objective_consistency  |
                              | - nonfinite_check        |
                              | - status_consistency     |
                              | - certified_status       |
                              +--------------------------+
```

### Key Types
- **`CheckResult`**: Contains `passed` (bool), `max_violation` (double), `tolerance_used` (double), `worst_index` (size_t), and `diagnostic` (string).
- **`LPVerificationReport`**: Contains individual `CheckResult` structs, aggregate primal/dual residuals, independent recomputed objective, original and certified status, and `fully_verified()` method.
- **`VerifierTolerances`**: Dedicated verification thresholds derived from or overriding the solver tolerances.
- **`LPVerifier`**: Stateless verification engine exposing `verify(model, x, y, rc, reported_obj, reported_status)`.

---

## 4. Status Certification and Downgrading Logic

When the solver reports `OPTIMAL`:
1. If non-finite values are detected: Status is downgraded to `NUMERICAL_FAILURE`.
2. If primal feasibility or bound feasibility fails: Status is downgraded to `UNKNOWN`.
3. If objective mismatch or dual/complementarity check fails: Status is downgraded to `FEASIBLE`.
4. If and only if **all applicable checks pass**: The result is certified as `OPTIMAL`.

When presolve is enabled:
- The solver solves the compact reduced problem $M_{\text{reduced}}$ and maps the primal solution back to $M_{\text{original}}$.
- The verifier validates the exact dual certificate on $M_{\text{reduced}}$ and independently confirms primal feasibility, bound satisfaction, and objective preservation on $M_{\text{original}}$.

---

## 5. Fault Injection Matrix

The test suite in `tests/unit/test_phase6_verification.cpp` executes exhaustive fault injection tests:

| Fault Type | Injection Mechanism | Detection Asserted |
|---|---|---|
| **Fault A: Wrong Objective** | Perturbed reported objective by $+819$ | `objective_consistency.passed == false` |
| **Fault B: Primal Infeasibility** | Shifted $x_2$ by $+200$ outside feasible region | `primal_feasibility.passed == false` |
| **Fault C: Bound Violation** | Set variable $x$ to $15$ with upper bound $10$ | `bound_feasibility.passed == false` |
| **Fault D: Dual Violation** | Injected dual $y = -100$ into optimal basic solution | `dual_feasibility.passed == false` |
| **Fault E: Complementary Slackness** | Injected non-zero dual on strictly slack constraint | `complementary_slackness.passed == false` |
| **Fault F: Non-Finite (NaN / Inf)** | Injected `NaN` or `+Inf` into solution vector | `nonfinite_check.passed == false` -> `NUMERICAL_FAILURE` |
| **Fault G: Status Corruption** | Claimed `OPTIMAL` on non-optimal feasible point | `objective_consistency.passed == false` / dual failure |
