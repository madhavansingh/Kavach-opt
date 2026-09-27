# Kavach-Opt Revised Simplex LP Solver Documentation

## 1. Overview and Architecture

The `src/algorithms/lp` subsystem implements the sovereign, production-quality Revised Simplex Linear Programming (LP) solver for Kavach-Opt, strictly fulfilling the requirements of **Section 6**, **Section 8**, **Section 11**, **Section 22**, and **Section 32** from `SIH26119_Kavach-Opt_Blueprint_v2.md`.

### Core Architectural Decisions & Invariants
- **Algorithm**: Revised Simplex with sparse LU-factorized basis representation (`SparseLU`), avoiding dense tableau manipulation or redundant inverses.
- **Pivot Selection**: Dantzig entering rule by default with minimum-ratio leaving rule and Bland's index-ordered tie-breaking.
- **Anti-Cycling Defense**: Consecutive degeneracy watchdog triggering Bland's smallest-index rule fallback upon cycle detection ($k \ge \text{threshold}$).
- **Two-Phase Simplex**: Rigorous Phase I feasibility detection ($W^* > \epsilon_{\text{feas}} \implies \text{INFEASIBLE}$) and degenerate artificial variable purging before Phase II.
- **Numerical Robustness**: Full integration with geometric-mean row/column scaling and unscaling, condition estimation $\hat{\kappa}_\infty(B)$, and named-tolerance enforcement.
- **Presolve Integration**: Seamless integration with the Netlib-grade `src/presolve` subsystem and $O(n)$ solution reconstruction.
- **Strict Sovereignty**: **Zero external dependencies** (no Eigen, SuiteSparse, GLPK, HiGHS, or LAPACK).

```
                      +---------------------------------------+
                      |   CanonicalModel (const &)            |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   1. Problem Class Validation         | (Rejects MILP, QP)
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   2. Presolve Engine [C]              |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   3. Geometric-Mean Scaling [E]       |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   4. Standard-Form LP Transformation  | (A z = b, z >= 0, b >= 0)
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   5. SimplexBasis Initialization      | (Initial B = I)
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   6. Phase I Simplex (SparseLU)       | (Artificial minimization)
                      +---------------------------------------+
                                          |
                      +-------------------+-------------------+
                      | (W* > tol)                            | (W* <= tol)
                      v                                       v
               [ INFEASIBLE ]                  +-------------------------------+
                                               | 7. Pivot out Artificials      |
                                               +-------------------------------+
                                                              |
                                                              v
                                               +-------------------------------+
                                               | 8. Phase II Simplex (SparseLU)|
                                               +-------------------------------+
                                                              |
                      +---------------------------------------+---------------------------------------+
                      | (rc >= -tol)                          | (min_ratio = inf)                     | (iter/time limit)
                      v                                       v                                       v
                 [ OPTIMAL ]                            [ UNBOUNDED ]                          [ LIMIT_REACHED ]
                      |
                      v
       +-------------------------------+
       | 9. Standard-Form Recovery     |
       +-------------------------------+
                      |
                      v
       +-------------------------------+
       | 10. Unscaling Matrix Rescaling|
       +-------------------------------+
                      |
                      v
       +-------------------------------+
       | 11. Presolve Reconstruction   |
       +-------------------------------+
                      |
                      v
       +-------------------------------+
       | LPSolution [A]                |
       +-------------------------------+
```

---

## 2. Mathematical Formulation & Standard Form Conversion

The general canonical linear programming problem is expressed as:
$$
\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & c^T x + c_0 \\
\text{s.t.} \quad & \sum_{j} a_{ij} x_j \le b_i \quad (i \in \mathcal{L}) \\
& \sum_{j} a_{ij} x_j \ge b_i \quad (i \in \mathcal{G}) \\
& \sum_{j} a_{ij} x_j = b_i \quad (i \in \mathcal{E}) \\
& l_j \le x_j \le u_j \quad (\forall j \in \{0, \dots, n-1\})
\end{aligned}
$$
where maximization objectives are transformed via $\min (-c^T x - c_0)$ with sign reversal recorded for final objective reporting.

### 2.1 Variable Transformation Taxonomy
Every canonical variable $x_j$ is classified into exactly one transformation type based on its bounds:

1. **Fixed Variable** ($|u_j - l_j| \le \epsilon_{\text{prim}}$):
   $x_j = l_j$. Substituted directly; shifts constraint RHS by $a_{ij} l_j$ and objective offset by $c_j l_j$.
2. **Shifted Lower Bound** ($l_j > -\infty, u_j = +\infty$):
   $z_k = x_j - l_j \ge 0 \implies x_j = z_k + l_j$.
3. **Shifted Upper Bound** ($l_j = -\infty, u_j < +\infty$):
   $z_k = u_j - x_j \ge 0 \implies x_j = u_j - z_k$.
4. **Boxed Variable** ($l_j > -\infty, u_j < +\infty$):
   $z_k = x_j - l_j \ge 0 \implies x_j = z_k + l_j$, with auxiliary upper bound constraint $z_k + s_{ub} = u_j - l_j$ where $s_{ub} \ge 0$ is a slack variable.
5. **Free Variable** ($l_j = -\infty, u_j = +\infty$):
   $x_j = z_{k1} - z_{k2}$, where $z_{k1}, z_{k2} \ge 0$.

### 2.2 Standard-Form Equality Constraints
All constraints are converted to equalities with non-negative right-hand sides $b \ge 0$:
1. **$\le$ Constraints**: $a_i^T z + s_i = b_i$, $s_i \ge 0$ (slack variable). If $b_i < 0$, negated to $-a_i^T z - s_i + w_i = -b_i$ with Phase I artificial $w_i \ge 0$.
2. **$\ge$ Constraints**: $a_i^T z - e_i + w_i = b_i$, $e_i \ge 0$ (surplus), $w_i \ge 0$ (artificial).
3. **$=$ Constraints**: $a_i^T z + w_i = b_i$, $w_i \ge 0$ (artificial).
4. **Boxed Upper-Bound Rows**: $z_k + s_{ub} = u_j - l_j$, $s_{ub} \ge 0$ (slack variable).

This yields the unified standard equality system:
$$
A_{\text{std}} z = b_{\text{std}}, \quad z \ge 0, \quad b_{\text{std}} \ge 0
$$
where the initial basis $B = I_{m \times m}$ is immediately formed by slacks and artificials, guaranteeing a trivial, well-conditioned starting basis.

---

## 3. Two-Phase Revised Simplex Algorithm

### 3.1 Phase I: Artificial Minimization
The auxiliary Phase I objective minimizes the sum of artificial variables:
$$
W = \sum_{k \in \mathcal{W}} w_k
$$
At each iteration:
1. Solve $B^T y = c_{B}^{\text{phase1}}$ for dual simplex multipliers $y$.
2. Compute reduced costs for nonbasic variables: $\bar{c}_j = c_j^{\text{phase1}} - y^T A_j$.
3. Check optimality: if $\bar{c}_j \ge -\epsilon_{\text{opt}}$ for all nonbasic $j$, Phase I terminates:
   - If $W^* > \epsilon_{\text{feas}}$, the problem is mathematically **INFEASIBLE**.
   - If $W^* \le \epsilon_{\text{feas}}$, the problem is **FEASIBLE**.

### 3.2 Purging Degenerate Basic Artificial Variables
If an artificial variable remains in the basis at zero value ($x_w = 0$), a dual direction $v = B^{-T} e_r$ is computed for the artificial row $r$. The algorithm pivots the artificial out against any non-artificial column $j$ having $|\alpha_{rj}| = |v^T A_j| > \epsilon_{\text{pivot}}$. All artificial columns are then permanently purged from the nonbasic pool.

### 3.3 Phase II: Primal Optimization
Using the feasible basis from Phase I, the solver minimizes the original objective $c^T z$:
1. Solve $B^T y = c_B$ for dual multipliers $y$.
2. Calculate reduced costs $\bar{c}_j = c_j - y^T A_j$.
3. If $\bar{c}_j \ge -\epsilon_{\text{opt}}$ for all nonbasic $j$, the current basic solution is **OPTIMAL**.
4. Choose entering variable $q$ (Dantzig rule or Bland rule).
5. Solve direction $B d = A_q$.
6. If $d \le \epsilon_{\text{pivot}}$ component-wise, the problem is **UNBOUNDED**.
7. Minimum-ratio test: $\theta^* = \min_{i: d_i > \epsilon} \frac{x_{Bi}}{d_i}$, with Bland index tie-breaking.
8. Pivot leaving variable out, factorize new basis, and update basic solution.

---

## 4. Basis Representation and Incremental Basis-Update Engine

Rather than refactorizing the full basis matrix $B \in \mathbb{R}^{m \times m}$ from scratch at every simplex pivot, Kavach-Opt utilizes an **Incremental Basis-Update Factorization Engine** based on sparse Product-Form-of-Inverse (PFI) elementary Eta matrices.

### 4.1 Base Factorization & Elementary Eta Updates
1. **Base Factorization**: An initial or refreshed basis $B_0$ is factored using native `SparseLU` with Markowitz ordering and threshold partial pivoting ($\tau = 0.1$).
2. **Elementary Eta Matrix ($E_k$)**: When column $p$ (leaving row) of $B_k$ is replaced by the entering column $a_j$, the new basis is:
   $$B_{k+1} = B_k E_k$$
   where $E_k = I + (d - e_p) e_p^T$ and $d = B_k^{-1} a_j$.
3. **Sparse Representation**: Only the diagonal pivot entry $d_p$ and non-zero off-diagonal entries $|d_i| > 10^{-15}$ ($i \ne p$) are stored in `SparseEtaFactor`, achieving minimal memory overhead.

### 4.2 Linear Solves via the Eta Chain
- **Primal Solve ($B_k x_B = b$)**:
  1. Solve base system $B_0 w = b$ via base `SparseLU`.
  2. Apply forward Eta factors $k = 0, \dots, K-1$:
     $$w_p \leftarrow \frac{w_p}{d_p}, \quad w_i \leftarrow w_i - d_i w_p \quad (\forall i \ne p)$$
     yielding $x_B = w$.
- **Dual Solve ($B_k^T y = c_B$)**:
  1. Initialize $v = c_B$.
  2. Apply reverse Eta factors $k = K-1, \dots, 0$:
     $$v_p \leftarrow \frac{v_p - \sum_{i \ne p} d_i v_i}{d_p}$$
  3. Solve base transpose system $B_0^T y = v$ via base `SparseLU`.
- **Direction Solve ($B_k d = a_q$)**:
  Solved identically to the primal system with right-hand side $a_q = A[:, q]$.

### 4.3 Multi-Criteria Refactorization & Stability Policy
To control numerical error propagation and memory growth, `SimplexBasis::needs_refactorization` triggers a full SparseLU refactorization under any of the following conditions:
1. **Periodic Interval**: Update chain length reaches `refactorization_interval` (default 50 pivots).
2. **Memory Footprint Bound**: Cumulative Eta chain bytes exceed $4 \times \text{base\_bytes} + 64\text{ KB}$.
3. **Numerical Drift Safeguard**: True primal residual $\|B x_B - b\|_\infty$ exceeds $100 \times \epsilon_{\text{feas}}$.
4. **Pivot Stability Guard**: Candidate pivot $|d_p| < \max(\epsilon_{\text{pivot}}, 10^{-11})$ or relative pivot drop $|d_p| < 10^{-7} \|d\|_\infty$.
5. **Fault Isolation**: Candidate vector contains non-finite entries (NaN or $\pm\infty$), triggering safe fallback without basis memory corruption.

The reciprocal condition number is continuously monitored:
$$
\hat{\kappa}_\infty(B) \approx \frac{\|B\|_\infty}{\min_i |U_{ii}|}
$$
If $\min_i |U_{ii}| < \epsilon_{\text{pivot}}$, the basis is diagnosed as numerically singular, returning `NUMERICAL_FAILURE`.

---

## 5. Anti-Cycling Defense: Bland's Rule Watchdog Fallback

Degenerate pivots ($\theta^* \le \epsilon_{\text{feas}}$) do not decrease the objective value and can theoretically cycle indefinitely on pathological instances (e.g., Beale's problem).

Kavach-Opt employs a **Watchdog-Triggered Bland Fallback**:
- Under normal operation, the **Dantzig entering rule** ($\min_j \bar{c}_j$) maximizes per-step progress.
- Consecutive degenerate pivots are tracked: `consecutive_degenerate`.
- When `consecutive_degenerate >= bland_watchdog_threshold` (default 20), the watchdog activates:
  1. **Entering Variable**: Lowest index $j$ with $\bar{c}_j < -\epsilon_{\text{opt}}$.
  2. **Leaving Variable**: Lowest index basic variable among minimum ratio ties.
- Telemetry records `bland_fallback_activated = true` and logs total iterations executed under Bland's rule.
- Once a non-degenerate pivot occurs, Dantzig's rule is restored.

---

## 6. Presolve & Scaling Pipeline

Before simplex optimization:
1. **Presolve Reduction**: Fixed variables substituted, empty rows pruned, bounds tightened, redundant constraints removed.
2. **Equilibration Scaling**: Matrix equilibrated via Ruiz or geometric equilibration ($A' = R A C$).
3. **Simplex Optimization**: Standard-form problem solved on scaled system using incremental basis updates.
4. **Unscaling**: Primal variables unscaled via $x = C x'$, dual multipliers via $y = R y'$.
5. **Full Sovereign Reconstruction**: `ReconstructionMap` reconstructs original primal variables $x_{\text{original}}$, dual multipliers $y_{\text{original}}$, reduced costs $r_{\text{original}}$, and original objective value. Solutions are delivered directly to the decoupled independent verifier on the **original** mathematical formulation.
6. **Sub-Millisecond Time Limit Support**: The solver CLI and API support fractional second time limits down to 100 microseconds ($0.0001\text{ s}$) via `api::SolverOptions::parse_time_limit_seconds` without truncation to zero.

---

## 7. Status Taxonomy and Error Hierarchy

The LP solver returns unambiguous statuses conforming to `core::SolverStatus`:

| Status | Mathematical Condition |
| :--- | :--- |
| `OPTIMAL` | Primal feasible ($A z = b, z \ge 0$) and dual feasible ($\bar{c} \ge -\epsilon$) |
| `INFEASIBLE` | Phase I optimal value $W^* > \epsilon_{\text{feas}}$ |
| `UNBOUNDED` | Entering column $A_q$ has direction $d = B^{-1} A_q \le \epsilon_{\text{pivot}}$ |
| `LIMIT_REACHED` | Exceeded `max_iterations` or `max_time_seconds` |
| `NUMERICAL_FAILURE` | Basis singular or condition estimate exceeds safety bounds |

---

## 8. Netlib Benchmark Results

Validation on 13 real-world Netlib benchmark problems across varying dimensions, densities, and condition numbers:

| Instance | Constraints ($m$) | Variables ($n$) | NNZ | Kavach-Opt Status | Objective Found | Reference Value | Iters | Time (Release) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **afiro.mps** | 27 | 32 | 83 | `OPTIMAL` | -464.75314 | -464.75314 | 16 | 0.39 ms |
| **sc50b.mps** | 50 | 48 | 118 | `OPTIMAL` | -70.000000 | -70.000000 | 53 | 2.59 ms |
| **sc50a.mps** | 50 | 48 | 130 | `OPTIMAL` | -64.575077 | -64.575077 | 51 | 2.61 ms |
| **kb2.mps** | 43 | 41 | 286 | `OPTIMAL` | -1749.9001 | -1749.9001 | 127 | 6.31 ms |
| **blend.mps** | 74 | 83 | 491 | `OPTIMAL` | -30.812150 | -30.812150 | 272 | 51.48 ms |
| **adlittle.mps** | 56 | 97 | 383 | `OPTIMAL` | 225494.96 | 225494.96 | 92 | 7.74 ms |
| **share2b.mps** | 96 | 79 | 694 | `OPTIMAL` | -415.73224 | -415.73224 | 149 | 10.51 ms |
| **sc105.mps** | 105 | 103 | 280 | `OPTIMAL` | -52.202061 | -52.202060 | 110 | 11.16 ms |
| **stocfor1.mps** | 117 | 111 | 447 | `OPTIMAL` | -41131.976 | -41131.976 | 138 | 11.48 ms |
| **share1b.mps** | 117 | 225 | 1151 | `OPTIMAL` | -76589.319 | -76589.319 | 289 | 35.30 ms |
| **lotfi.mps** | 153 | 308 | 1078 | `OPTIMAL` | -25.264706 | -25.264707 | 319 | 60.85 ms |
| **beaconfd.mps** | 173 | 262 | 3375 | `OPTIMAL` | 33592.486 | 33592.486 | 246 | 45.15 ms |
| **boeing2.mps** | 185 | 143 | 1283 | `OPTIMAL` | -315.01873 | -315.01872 | 417 | 137.10 ms |
| **Total** | | | | **100% PASS** | | | **2279** | **382.6 ms** |
