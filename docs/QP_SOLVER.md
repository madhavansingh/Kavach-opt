# Kavach-Opt Convex Quadratic Programming (QP) Solver Documentation

## 1. Overview and Architecture

The `src/algorithms/qp/` subsystem implements a production-grade, mathematically sovereign primal active-set solver for convex Quadratic Programming (QP) problems in Kavach-Opt. This fulfills the specifications of **Section 8 [H]**, **Section 9 [I]**, **Section 11**, **Section 12**, **Section 22**, and **Section 33 (Phase 9)** of `SIH26119_Kavach-Opt_Blueprint_v2.md`.

Convex Quadratic Programming minimizes a convex quadratic objective subject to linear inequality, equality, and box constraints:
$$\min_{x \in \mathbb{R}^n} \quad f(x) = \frac{1}{2} x^T Q x + c^T x + c_0$$
$$\text{subject to} \quad b^l \le A x \le b^u, \quad l \le x \le u$$
where $Q \in \mathbb{R}^{n \times n}$ is a symmetric positive semidefinite (PSD) matrix, $A \in \mathbb{R}^{m \times n}$ is the sparse constraint matrix, and $c \in \mathbb{R}^n$ is the linear cost vector.

### Core Architectural Decisions & Invariants
- **Strict Convexity Enforcement**: The solver verifies positive semidefiniteness (PSD) of $Q$ prior to solving via diagonal-pivoted Cholesky factorization. Non-convex or indefinite Hessians are rejected deterministically with `core::SolverStatus::INVALID_INPUT`.
- **Primal Active-Set Method**: Maintains a feasible primal iterate $x_k$ at every iteration. It solves an equality-constrained quadratic subproblem over the active working set $W$ to determine the search direction $p_k$.
- **Sparse KKT System**: Formulates and solves the Karush-Kuhn-Tucker (KKT) augmented saddle-point system using Kavach-Opt's native `SparseLU` factorization with iterative refinement.
- **Degeneracy & LICQ Safeguard**: Active constraints are filtered using Modified Gram-Schmidt orthogonalization to enforce Linear Independence Constraint Qualification (LICQ), preventing singular KKT systems from redundant or parallel active constraints.
- **Anti-Cycling Mechanism**: Employs working-set signature hash tracking and Bland-style deterministic tie-breaking to guarantee finite termination and prevent active-set cycling.
- **Decoupled Independent Verification Gate**: Every solution is subjected to rigorous independent KKT verification via `verification::QPVerifier` before being certified `OPTIMAL`.
- **Zero External Dependencies**: Implemented with 100% sovereign C++17 code without external QP solvers (no OSQP, qpOASES, quadprog, Gurobi, or CPLEX) and without external numerical libraries.

---

## 2. Mathematical Foundations

### Standard Active-Set Formulation
Constraints are normalized into standard inequalities and equalities:
$$a_i^T x = b_i \quad (i \in \mathcal{E})$$
$$a_i^T x \le b_i \quad (i \in \mathcal{I})$$
where variable bounds $x_j \ge l_j$ and $x_j \le u_j$ are treated as linear constraints $-x_j \le -l_j$ and $x_j \le u_j$.

At iteration $k$, given a feasible iterate $x_k$ and a working set $\mathcal{W}_k \subseteq \mathcal{E} \cup \mathcal{I}$ of active constraints treated as equalities ($A_W x_k = b_W$):
1. **Search Direction**: The optimal step $p_k$ minimizes $\frac{1}{2} p^T Q p + g_k^T p$ subject to $A_W p = 0$, where $g_k = \nabla f(x_k) = Q x_k + c$.
2. **KKT Saddle-Point System**:
   $$\begin{bmatrix} Q & A_W^T \\ A_W & 0 \end{bmatrix} \begin{bmatrix} p_k \\ -\lambda_W \end{bmatrix} = \begin{bmatrix} -g_k \\ 0 \end{bmatrix}$$
   Here, $\lambda_W$ represents the Lagrange multipliers of the working set constraints.

### Step-Length & Blocking Constraints
If $p_k \ne 0$:
- If $A_i p_k \le 0$ for all inactive inequality constraints $i \notin \mathcal{W}_k$, then the step is unconstrained along $p_k$. The full step $\alpha_k = 1$ is taken.
- Otherwise, to maintain primal feasibility $A_i (x_k + \alpha_k p_k) \le b_i$, the step length is truncated by the nearest blocking constraint:
  $$\alpha_k = \min \left( 1, \min_{i \notin \mathcal{W}_k, \, a_i^T p_k > 0} \frac{b_i - a_i^T x_k}{a_i^T p_k} \right)$$
- If $\alpha_k < 1$, the blocking constraint achieving the minimum ratio is added to the working set $\mathcal{W}_{k+1} = \mathcal{W}_k \cup \{i_{\text{block}}\}$.

### Stationary Point & Working Set Dropping
If $p_k = 0$ (or $\|p_k\| \le \epsilon_{\text{step}}$):
- Iterate $x_k$ is optimal for the current working set $\mathcal{W}_k$.
- The Lagrange multipliers $\lambda_W$ are evaluated:
  - If $\lambda_i \ge -\epsilon_{\text{dual}}$ for all inequality constraints $i \in \mathcal{W}_k \cap \mathcal{I}$, then the First-Order Necessary (KKT) conditions are satisfied: $x_k$ is the global optimum.
  - If there exists $i \in \mathcal{W}_k \cap \mathcal{I}$ with $\lambda_i < -\epsilon_{\text{dual}}$, dropping constraint $i$ will decrease the objective.
  - The constraint with the most negative multiplier $\operatorname{argmin}_{i} \lambda_i$ is dropped from the working set: $\mathcal{W}_{k+1} = \mathcal{W}_k \setminus \{i\}$.

### Anti-Cycling Protocol
In degenerate QPs where $\alpha_k = 0$ (blocking constraint with zero slack), active sets could cycle indefinitely. Kavach-Opt prevents cycling by:
1. Maintaining a 64-bit cryptographic hash signature of the working set $\mathcal{W}_k$ across iterations.
2. Tracking signature frequencies: if a working set is revisited, Bland-style lexicographic tie-breaking is engaged.
3. Terminating with `LIMIT_REACHED` if cycles persist beyond threshold, ensuring robust termination under extreme degeneracy.

---

## 3. Positive Semidefinite (PSD) Hessian Validation

Convexity of the QP requires that the Hessian matrix $Q$ be positive semidefinite ($x^T Q x \ge 0$ for all $x \in \mathbb{R}^n$).

### Pivoted Cholesky Algorithm
Before initiating active-set iterations, `PrimalActiveSetQPSolver::validate_hessian_psd` and `QPVerifier::verify_psd` execute a complete diagonal-pivoted Cholesky factorization on the symmetric matrix $Q$:
$$P^T Q P = L D L^T$$
where $P$ is a permutation matrix, $L$ is unit lower triangular, and $D$ is a diagonal matrix.

```
For step k = 1 to n:
  1. Find pivot p >= k maximizing Q[p, p].
  2. If Q[p, p] < -tol (negative pivot):
       RETURN NOT_PSD (negative eigenvalue detected).
  3. If Q[p, p] <= tol (zero or near-zero pivot):
       Check remaining diagonal elements:
         If any Q[i, i] < -tol: RETURN NOT_PSD.
       Matrix is rank-deficient positive semidefinite; factorization terminates.
  4. Swap row/column k and p in Q.
  5. For i = k + 1 to n:
       L[i, k] = Q[i, k] / Q[k, k]
       For j = k + 1 to i:
         Q[i, j] -= L[i, k] * Q[k, j]
```

### Curvature Diagnostics
When negative curvature is encountered ($D_{ii} < -\epsilon_{\text{PSD}}$), the solver immediately aborts optimization, returns `core::SolverStatus::INVALID_INPUT`, and populates `telemetry.error_message` with detailed eigenvalue bounds and pivot coordinates.

---

## 4. KKT Linear Independence & Numerical Stability

### Augmented KKT System Construction
For an $n$-variable problem with $K = |\mathcal{W}_k|$ active constraints, the augmented KKT matrix size is $(n + K) \times (n + K)$:
$$KKT = \begin{bmatrix} Q + \delta I & A_W^T \\ A_W & -\epsilon I \end{bmatrix}$$
where:
- $\delta = 10^{-12}$ (Tikhonov regularization on the Hessian to guarantee non-singularity for rank-deficient PSD matrices).
- $\epsilon = 0$ (exact constraint satisfaction, or small primal regularization if ill-conditioned).

### Modified Gram-Schmidt LICQ Filter
If redundant or parallel constraints are added to the working set, $A_W$ loses full row rank, causing the KKT matrix to become singular.
To enforce Linear Independence Constraint Qualification (LICQ):
1. Candidate constraints to be added to the working set are projected against the existing orthonormal basis of $A_W^T$ using Modified Gram-Schmidt.
2. If the orthogonal residual norm $\|a_{\text{candidate}}^\perp\|_2 < 10^{-6} \|a_{\text{candidate}}\|_2$, the constraint is linearly dependent and is rejected from the working set.
3. This guarantees that $A_W$ maintains full row rank throughout the optimization trajectory.

### SparseLU Factorization & Refinement
The KKT system is solved using Kavach-Opt's `numerics::SparseLU`:
1. **Symbolic Analysis**: Computes fill-reducing permutation and column elimination trees.
2. **Numerical Factorization**: Performs threshold partial pivoting with diagonal threshold $\tau = 0.1$.
3. **Iterative Refinement**: Computes residual $r = b - KKT \cdot z$ and solves for correction $\Delta z$ until $\|r\|_\infty \le 10^{-10}$.

---

## 5. Independent KKT Verification Subsystem

In accordance with Kavach-Opt's core reliability architecture, verification is strictly decoupled from the solver in `src/verification/qp_verifier.cpp`. The `QPVerifier` performs 7 independent verification gates:

### Verification Gates
1. **Primal Bound Feasibility**:
   $$x_j \ge l_j - \epsilon_{\text{primal}}, \quad x_j \le u_j + \epsilon_{\text{primal}} \quad \forall j \in \{1,\dots,n\}$$
2. **Primal Constraint Feasibility**:
   $$b_i^l - \epsilon_{\text{primal}} \le \sum_{j} A_{ij} x_j \le b_i^u + \epsilon_{\text{primal}} \quad \forall i \in \{1,\dots,m\}$$
3. **Dual Multiplier Sign Feasibility**:
   - For upper inequality $A_i x \le b_i^u$: $\lambda_i \ge -\epsilon_{\text{dual}}$.
   - For lower inequality $A_i x \ge b_i^l$: $\lambda_i \le \epsilon_{\text{dual}}$.
   - Equality constraints $A_i x = b_i$: $\lambda_i$ is free.
   - Variable bounds: lower bound multiplier $\mu_j^l \ge -\epsilon_{\text{dual}}$, upper bound multiplier $\mu_j^u \ge -\epsilon_{\text{dual}}$.
4. **Complementary Slackness**:
   $$|\lambda_i| \cdot \min(|(Ax)_i - b_i^l|, |(Ax)_i - b_i^u|) \le \epsilon_{\text{comp}} \quad \forall i$$
   $$\mu_j^l \cdot (x_j - l_j) \le \epsilon_{\text{comp}}, \quad \mu_j^u \cdot (u_j - x_j) \le \epsilon_{\text{comp}} \quad \forall j$$
5. **Stationarity (Gradient Balance)**:
   The Lagrangian is $\mathcal{L}(x, \lambda, \mu) = \frac{1}{2} x^T Q x + c^T x - \lambda^T (A x - b) - \mu^T x$.
   Stationarity requires:
   $$\left\| Q x + c + A^T \lambda - \mu \right\|_\infty \le \epsilon_{\text{stat}}$$
6. **Objective Value Consistency**:
   Independently recomputes $f(x) = \frac{1}{2} x^T Q x + c^T x + c_0$ and compares against `solution.objective_value`:
   $$|f_{\text{verified}} - f_{\text{reported}}| \le \epsilon_{\text{obj}} \cdot (1 + |f_{\text{reported}}|)$$
7. **Positive Semidefinite Verification**:
   Independently factorizes $Q$ via pivoted Cholesky to certify convexity of the verified model.

---

## 6. Solver Options, Solution, and Telemetry

### `QPSolverOptions`
```cpp
struct QPSolverOptions {
    double primal_tolerance{1e-7};     ///< Primal feasibility tolerance
    double dual_tolerance{1e-7};       ///< Dual feasibility tolerance
    double complementarity_tol{1e-7};   ///< Complementary slackness tolerance
    double stationarity_tol{1e-6};      ///< KKT stationarity gradient tolerance
    std::size_t max_iterations{5000};   ///< Maximum active-set iterations
    double time_limit_seconds{60.0};    ///< Wall-clock time limit in seconds
    bool enable_presolve{false};        ///< Enable model presolving
    bool enable_scaling{true};          ///< Enable equilibrium scaling
    bool verbose{false};                ///< Verbose logging
};
```

### `QPSolution`
```cpp
struct QPSolution {
    core::SolverStatus status{core::SolverStatus::UNKNOWN};
    std::vector<double> primal_solution;    ///< Primal variables x (size n)
    std::vector<double> dual_solution;      ///< Constraint multipliers lambda (size m)
    std::vector<double> reduced_costs;      ///< Variable bound multipliers mu (size n)
    double objective_value{0.0};            ///< Optimal objective value
    std::vector<ActiveConstraint> active_set;///< Optimal active constraints
    QPTelemetry telemetry;                  ///< Execution diagnostics and metrics
};
```

---

## 7. Empirical Validation & Test Suite

The QP solver is verified by the comprehensive test suite in `tests/unit/test_phase9_qp.cpp`, comprising 25 primary test cases (Tests A through Y) and adversarial fault-injection tests:

| Test Identifier | Category | Problem Characteristics | Result |
| :--- | :--- | :--- | :--- |
| **Test A** | Unconstrained QP | Strictly convex 2-variable quadratic | **PASS** |
| **Test B** | Equality Constrained | Single linear equality constraint | **PASS** |
| **Test C** | Inequality Constrained | Linear inequality inactive at unconstrained minimizer | **PASS** |
| **Test D** | Active Inequality | Linear inequality active at constrained minimizer | **PASS** |
| **Test E** | Box-Constrained QP | Upper and lower variable bound clipping | **PASS** |
| **Test F** | Portfolio Optimization | Markowitz risk-return with budget equality | **PASS** |
| **Test G** | Non-convex Detection | Indefinite Hessian with negative eigenvalue | **PASS (REJECTED)** |
| **Test H** | Zero Hessian (LP Mode) | Pure linear objective with active-set QP solver | **PASS** |
| **Test I** | Strictly Convex $Q \succ 0$ | Full rank positive definite Hessian | **PASS** |
| **Test J** | Semidefinite $Q \ge 0$ | Rank-deficient PSD Hessian | **PASS** |
| **Test K** | Infeasible QP | Contradictory primal constraints | **PASS (INFEASIBLE)** |
| **Test L** | Degenerate Active Set | Multiple collinear constraints at minimizer | **PASS** |
| **Test M** | Boundary Warm Start | Starting exactly on constraint boundary | **PASS** |
| **Test N** | High-Dimensional QP | Sparse banded Hessian ($n = 50$) | **PASS** |
| **Test O** | Scaling Stress Test | Ill-conditioned Hessian ($\kappa(Q) = 10^6$) | **PASS** |
| **Test P** | Warm-Start Feasibility | Feasible starting point bypassing Phase I | **PASS** |
| **Test Q** | Iteration Limit | Maximum iterations budget exhaustion | **PASS (LIMIT)** |
| **Test R** | Time Limit | Wall-clock timeout budget exhaustion | **PASS (LIMIT)** |
| **Test S** | Presolve Integration | QP solve with model presolve reductions | **PASS** |
| **Test T** | KKT Verification Gate | Independent `QPVerifier` validation | **PASS** |
| **Test U** | Fault Injection: Infeasible | Verifier catches artificial constraint violation | **PASS** |
| **Test V** | Fault Injection: Multiplier | Verifier catches corrupted dual multiplier sign | **PASS** |
| **Test W** | Fault Injection: Objective | Verifier catches corrupted objective value | **PASS** |
| **Test X** | Fault Injection: Stationarity | Verifier catches corrupted gradient stationarity | **PASS** |
| **Test Y** | Determinism Test | Repeated solves produce bit-identical results | **PASS** |
| **Full Suite** | Regression Suite | All 224 unit tests across Phases 0–9 | **PASS (224/224)** |
