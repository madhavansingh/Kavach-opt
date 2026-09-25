# Kavach-Opt Dual-Simplex Warm-Start Subsystem Documentation

## 1. Overview and Architecture

The `src/algorithms/lp/dual_simplex.*` subsystem implements the production-grade, mathematically sovereign Dual Simplex engine for warm-starting child-node LP re-solves in Kavach-Opt's Branch-and-Bound solver. This fulfills the specifications of **Section 8 [F]**, **Section 9 [G]**, **Section 11**, **Section 12**, **Section 22**, and **Section 32 (Phase 8)** of `SIH26119_Kavach-Opt_Blueprint_v2.md`.

In Mixed-Integer Linear Programming (MILP), exploring nodes in the Branch-and-Bound search tree involves repeatedly solving continuous LP relaxations differing from parent nodes only by tightened variable bounds $[l_j, u_j]$. Whereas primal simplex would require restarting Phase I to regain feasibility, the optimal basis of the parent LP remains **dual feasible** for the child LP. Dual simplex warm-starting exploits this property to restore primal feasibility in a small fraction of iterations (typically 1–5 pivots), yielding dramatic speedups in node processing time.

### Core Architectural Decisions & Invariants
- **Dual Feasibility Invariant**: Prior to initiating dual pivots, the solver verifies that the inherited basis satisfies dual optimality conditions ($d_j = c_j - y^T A_j \ge -\epsilon_{\text{dual}}$ for nonbasic variables). If violated, the solver triggers an immediate, safe fallback to cold-start `RevisedSimplex`.
- **Structural Invariance via `preserve_bounds_structure`**: Variable bound tightening (such as $x_j \le \lfloor x_j^* \rfloor$ or $x_j \ge \lceil x_j^* \rceil$) is preserved in standard form without altering row or column indices. Boxed variables maintain an explicit upper-bound constraint row ($z_j + s_{ub} = u_j - l_j$) even when $u_j = l_j$, guaranteeing $100\%$ structural matrix $A$ identity between parent and child nodes.
- **Pivot Selection**:
  - **Leaving Row**: Identified by maximum primal infeasibility ($\max_i (-x_{B(i)})$ with violation $> \epsilon_{\text{primal}}$). Deterministic tie-breaking selects the row with the smallest basic variable index.
  - **Entering Column**: Dual ratio test selects $k = \operatorname{argmin}_{j \in N, \alpha_{lj} < -\epsilon_{\text{pivot}}} \frac{\max(0, d_j)}{-\alpha_{lj}}$. Deterministic tie-breaking selects the smallest nonbasic column index.
- **Farkas Ray Certification**: When a leaving row has $\alpha_{lj} \ge -\epsilon_{\text{pivot}}$ across all nonbasic variables, dual simplex terminates deterministically with `core::SolverStatus::INFEASIBLE` (dual unbounded ray certifying primal infeasibility of the child node).
- **Independent Verification Gate**: Every warm-start solution is independently verified by `verification::LPVerifier` before being certified `OPTIMAL`.
- **Zero External Dependencies**: Zero third-party linear programming or sparse factorization libraries (no SCIP, CBC, GLPK, HiGHS, CPLEX, Gurobi, Eigen, or SuiteSparse).

---

## 2. Mathematical Foundations

### Standard Form and Duality
The LP relaxation is transformed into standard equality form:
$$\min c^T z \quad \text{s.t.} \quad A z = b, \quad z \ge 0$$

Partitioning columns into basic $B$ and nonbasic $N$:
$$A = [B \quad N], \quad z = \begin{bmatrix} z_B \\ z_N \end{bmatrix}, \quad c = \begin{bmatrix} c_B \\ c_N \end{bmatrix}$$

The basic solution is:
$$z_B = B^{-1} b, \quad z_N = 0$$
The dual multipliers and reduced costs are:
$$y = B^{-T} c_B, \quad d_N = c_N - N^T y$$

### Dual Feasibility Preservation
When a child node tightens a bound $[l_j, u_j]$ on a variable, only the right-hand side $b$ is altered; the constraint matrix $A$ and cost vector $c$ remain identical. Consequently:
$$y = B^{-T} c_B \quad \text{and} \quad d_N = c_N - N^T y$$
remain completely unchanged! Because the parent basis was optimal, $d_N \ge -\epsilon_{\text{dual}}$ holds immediately at the child node.

However, because $b$ changed to $b'$, the primal basic values $z_B = B^{-1} b'$ may now have negative components ($z_{B(i)} < 0$), violating primal feasibility. Dual simplex operates directly on this dual-feasible, primal-infeasible basis.

### Leaving Variable Selection Rule
The leaving row $l$ is selected to address the worst violation of primal non-negativity:
$$l = \operatorname{argmax}_{i \in \{1,\dots,m\}} \{ -z_{B(i)} \mid z_{B(i)} < -\epsilon_{\text{primal}} \}$$
Ties are broken deterministically by choosing the row with the minimal column index of the basic variable:
$$l^* = \operatorname{argmin} \{ \operatorname{basis\_vars}[l] \mid -z_{B(l)} \ge \max_i(-z_{B(i)}) - \epsilon_{\text{primal}} \}$$

### Dual Minimum-Ratio Test
To determine which nonbasic variable $z_k$ enters the basis to replace $z_{B(l)}$, row $l$ of the simplex tableau is computed:
$$v = B^{-T} e_l \implies \alpha_j = v^T A_j = (B^{-1} A_j)_l$$

For $z_k$ entering to restore primal feasibility, $\alpha_k < -\epsilon_{\text{pivot}}$ is required. To maintain dual feasibility ($d_j' \ge 0$ for all $j \in N$ after the pivot):
$$d_j' = d_j - \frac{\alpha_j}{\alpha_k} d_k \ge 0 \iff \frac{d_k}{-\alpha_k} \le \frac{d_j}{-\alpha_j} \quad \forall j \text{ with } \alpha_j < -\epsilon_{\text{pivot}}$$

Therefore, the entering column is:
$$k = \operatorname{argmin}_{j \in N, \, \alpha_j < -\epsilon_{\text{pivot}}} \frac{\max(0, d_j)}{-\alpha_j}$$
Ties are broken deterministically by selecting the smallest variable index $j$.

### Primal Infeasibility / Farkas Ray Detection
If $\alpha_j \ge -\epsilon_{\text{pivot}}$ for all nonbasic variables $j \in N$, then the $l$-th tableau row represents:
$$z_{B(l)} + \sum_{j \in N} \alpha_j z_j = z_{B(l)} < 0$$
Since $\alpha_j \ge 0$ and $z_j \ge 0$, the left-hand side is non-negative for any feasible point, while the right-hand side is strictly negative. No feasible point exists. The vector $v = B^{-T} e_l$ constitutes a certified Farkas ray ($v^T A \ge 0$ and $v^T b < 0$), and the node is immediately certified `INFEASIBLE`.

---

## 3. Data Structures & State Management

### `BasisState`
The `BasisState` structure (`src/algorithms/lp/lp_types.hpp`) encapsulates the reusable basis state across nodes:
```cpp
struct BasisState {
    std::vector<std::size_t> basis_vars;      ///< Basic variable column index for each row (size m)
    std::vector<bool> is_basic;               ///< Indicator for all variables (size n)
    std::vector<std::size_t> nonbasic_vars;   ///< Nonbasic variable indices
    std::size_t num_rows{0};                  ///< Structural row dimension m
    std::size_t num_cols{0};                  ///< Structural column dimension n
    bool is_valid{false};                     ///< Validity flag
    bool scaling_applied{false};              ///< Scaling status flag
    std::optional<numerics::ScalingFactors> scaling_factors;
};
```

### Export & Import Protocol
- `SimplexBasis::export_state()` captures the current basis vector and factorization metadata upon successful solve.
- `SimplexBasis::load_state(const BasisState& state)` imports the basis vector and validates dimension consistency before factorization.

---

## 4. Structural Invariance & Standard Form Presolve Handling

In standard simplex presolve or transformation, tightening bounds could inadvertently change matrix dimensions (e.g. collapsing fixed variables $lb == ub$, adding slack columns, or creating artificial variables for negative RHS).

To guarantee $100\%$ structural compatibility between parent and child nodes:
1. **Preserving Bounds Structure**: `StandardFormLP` accepts `bool preserve_bounds_structure`. When true:
   - Variables with $ub == lb$ are maintained as `BOXED` ($z_j + s_{ub} = 0$), preventing row/column removal.
   - `LESS_EQUAL` rows with negative RHS maintain slack representation ($a^T z + s = b$) rather than being negated with Phase 1 artificial variables.
2. **Presolve Isolation**: Child node warm starts disable independent presolve (`enable_presolve = false`), guaranteeing identical dimensions $(m \times n)$ to the parent LP.
3. **Scaling Compatibility**: When parent LP solves were scaled, scaling factors are inherited and applied directly to the child standard form before dual simplex execution, unscaling the solution upon completion.

---

## 5. Robustness & Fallback Policy

Dual simplex includes comprehensive safeguards to guarantee that numerical difficulties never compromise solver stability:
1. **Structural Mismatch Fallback**: If standard-form dimensions differ ($m_{\text{child}} \ne m_{\text{basis}}$ or $n_{\text{child}} \ne n_{\text{basis}}$), warm start is rejected with `NUMERICAL_FAILURE`, triggering automatic cold-start simplex.
2. **Dual Infeasibility Fallback**: If the inherited basis exhibits dual violations ($d_j < -\epsilon_{\text{dual}}$), dual simplex safely aborts and falls back to cold start.
3. **Singular Basis Fallback**: If `SparseLU` basis factorization fails or condition estimate exceeds threshold, the solver falls back to cold start.
4. **Degeneracy & Bland Watchdog**: If consecutive pivots produce zero dual ratio progress exceeding `bland_watchdog_threshold`, Bland's rule is engaged to prevent dual cycling.
5. **Iteration & Time Limits**: Honors `max_iterations` and `time_limit_seconds`.

---

## 6. Branch-and-Bound Integration & Telemetry

In `src/algorithms/milp/branch_and_bound.cpp`:
- Each `BranchNode` carries `std::optional<lp::BasisState> warm_start_basis`.
- Root LP relaxation solves cold, exports optimal basis, and initializes `root.warm_start_basis`.
- Child nodes execute `solve_node_lp`:
  1. If `enable_warm_starts == true` and `warm_start_basis.has_value()`, invoke `RevisedSimplex::solve_warm_start`.
  2. If warm start succeeds with `OPTIMAL` or `INFEASIBLE`, record telemetry and proceed.
  3. If warm start encounters `NUMERICAL_FAILURE` or `LIMIT_REACHED`, record fallback reason and execute cold-start `RevisedSimplex::solve`.
  4. Upon optimal child solve, transfer child basis state to both left and right branches (`left.warm_start_basis = basis`, `right.warm_start_basis = basis`).

### Telemetry Diagnostics
The solver tracks detailed metrics in `MILPSolverTelemetry`:
- `warm_start_attempts`: Total warm-start invocations across all nodes.
- `warm_start_successes`: Number of warm starts successfully reaching `OPTIMAL` or `INFEASIBLE`.
- `warm_start_fallbacks`: Number of warm starts that triggered cold-start fallback.
- `warm_start_fallback_reasons`: Diagnostic strings detailing fallback triggers.
- `dual_simplex_iterations`: Cumulative iterations executed by dual simplex.
- `cold_start_iterations`: Cumulative iterations executed by cold-start primal simplex.
- `basis_transfer_count`: Total basis transfers from parent to child nodes.

---

## 7. Verification & Empirical Validation

The dual-simplex warm-start subsystem has been thoroughly validated through a dedicated test suite (`tests/unit/test_phase8_dual_simplex.cpp`) and verified across 193 unit tests under Debug, Release, and AddressSanitizer (ASan) configurations:

| Test Group | Tests | Description | Result |
| :--- | :--- | :--- | :--- |
| **Section 16 (A–N)** | 14 | Bound tightening, branching, dual feasibility, Farkas ray, tie-breaking, singular basis | **PASS** |
| **Section 17** | 4 | Mathematical equivalence: cold-start vs warm-start status, objective, residuals | **PASS** |
| **Section 18** | 4 | MILP regression: knapsack, mixed-integer, infeasible MILP with warm starts | **PASS** |
| **Section 19** | 1 | Performance benchmarking & telemetry validation (100% warm-start success rate) | **PASS** |
| **Section 20** | 2 | Adversarial child nodes: near-degenerate bounds, extreme aspect ratio coefficients | **PASS** |
| **Section 21** | 1 | Real MIP 6-item Knapsack verified against independent `MILPVerifier` | **PASS** |
| **Full Kavach Test Suite** | 193 | Regression suite covering all components from Phase 0 to Phase 8 | **PASS (193/193)** |
