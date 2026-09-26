# Kavach-Opt — Mathematical Reference

**Project:** SIH26119 / MRPL  
**Version:** 0.1.0  
**Build Standard:** C++20  
**Compiler Tested:** AppleClang  
**Document Scope:** Mathematical foundations, algorithms, and numerical architecture of the Kavach-Opt sovereign optimization engine.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Canonical Problem Representation](#2-canonical-problem-representation)
3. [Sparse Matrix Representation](#3-sparse-matrix-representation)
4. [Sparse LU Factorization](#4-sparse-lu-factorization)
5. [Ruiz Equilibration Scaling](#5-ruiz-equilibration-scaling)
6. [Presolve Reductions](#6-presolve-reductions)
7. [Two-Phase Revised Simplex LP](#7-two-phase-revised-simplex-lp)
8. [Branch-and-Bound MILP](#8-branch-and-bound-milp)
9. [Dual Simplex Method](#9-dual-simplex-method)
10. [Convex Quadratic Programming — Active-Set](#10-convex-quadratic-programming--active-set)
11. [Independent Verification Layer](#11-independent-verification-layer)
12. [Batched Sparse Matrix-Vector Multiplication](#12-batched-sparse-matrix-vector-multiplication)
13. [Deterministic Strategy Engine](#13-deterministic-strategy-engine)
14. [Numerical Tolerances](#14-numerical-tolerances)
15. [Mathematical Notation Summary](#15-mathematical-notation-summary)

---

## 1. Introduction

Kavach-Opt is a sovereign mathematical optimization engine designed for industrial operations planning under hard correctness and reproducibility requirements. It addresses three classes of mathematical programming problems:

**Linear Programming (LP).** Minimize a linear objective over a polyhedron defined by linear inequality and equality constraints and variable bound constraints:

$$
\min_{x \in \mathbb{R}^n} \; c^T x \quad \text{s.t.} \quad Ax \;\{{\le}, {=}, {\ge}\}\; b, \quad l \le x \le u
$$

**Mixed-Integer Linear Programming (MILP).** An LP in which a designated subset $I \subseteq \{1, \dots, n\}$ of variables is additionally restricted to integer values:

$$
\min_{x \in \mathbb{R}^n} \; c^T x \quad \text{s.t.} \quad Ax \;\{{\le}, {=}, {\ge}\}\; b, \quad l \le x \le u, \quad x_j \in \mathbb{Z} \; \forall j \in I
$$

**Convex Quadratic Programming (QP).** Minimize a convex quadratic objective over a polyhedral feasible set:

$$
\min_{x \in \mathbb{R}^n} \; \frac{1}{2} x^T Q x + c^T x \quad \text{s.t.} \quad Ax \;\{{\le}, {=}, {\ge}\}\; b, \quad l \le x \le u
$$

where $Q \in \mathbb{R}^{n \times n}$ is required to be symmetric positive semidefinite ($Q \succeq 0$).

Kavach-Opt implements all three problem classes using independent, sovereign solver subsystems — Revised Simplex for LP, Branch-and-Bound for MILP, and a Primal Active-Set method for convex QP — each backed by a shared sparse numerical layer (CSR/CSC storage, `SparseLU` factorization, geometric-mean scaling, and iterative refinement). Every solution is submitted to an independent, zero-trust verification layer before a status of `OPTIMAL` is certified. No forbidden optimization-engine dependencies exist in the solving path.

---

## 2. Canonical Problem Representation

### 2.1 CanonicalModel

All three problem classes are ingested through a single in-memory contract, `kavach::model::CanonicalModel`. This structure is the exclusive interface between parsers, the strategy layer, the presolve engine, and the three solver subsystems. The original `CanonicalModel` is immutable; no solver or presolve step modifies it in place.

The general form held internally is:

$$
\begin{aligned}
\min \quad & \frac{1}{2} x^T Q x + c^T x + c_0 \\
\text{s.t.} \quad & \sum_{j} a_{ij} x_j \;\{{\le}, {=}, {\ge}\}\; b_i \quad (i = 1, \dots, m) \\
& l_j \le x_j \le u_j \quad (j = 1, \dots, n) \\
& x_j \in \mathbb{Z} \quad \forall j \in I \quad \text{(MILP only)}
\end{aligned}
$$

For pure LP, $Q = 0$. For MILP, $Q = 0$ and $I \ne \emptyset$. For convex QP, $Q \succeq 0$ and $I = \emptyset$.

### 2.2 Variable Types

Variables are identified by a zero-indexed `VarId` and carry one of three types:

| Type | Domain | Bound Semantics |
| :--- | :--- | :--- |
| `CONTINUOUS` | $x_j \in [l_j, u_j] \subset \mathbb{R}$ | Finite or $\pm\infty$ via IEEE-754 `double` |
| `INTEGER` | $x_j \in \mathbb{Z} \cap [l_j, u_j]$ | Finite or $\pm\infty$ |
| `BINARY` | $x_j \in \{0, 1\}$ | Normalized: $l_j = 0$, $u_j = 1$ |

Unbounded sides are represented using `std::numeric_limits<double>::infinity()`. A free variable satisfies $l_j = -\infty$, $u_j = +\infty$.

### 2.3 Constraint Representation

Each constraint $i$ stores a `SparseLinearExpression`: a sorted, deduplicated list of `LinearTerm { VarId, double coefficient }` pairs representing:

$$
\sum_{j \in \text{terms}_i} a_{ij} x_j \;\{{\le}, {=}, {\ge}\}\; b_i
$$

The right-hand side $b_i$ must be finite. All coefficients are finite IEEE-754 doubles. Constraint senses supported are `LESS_EQUAL` ($\le$), `EQUAL` ($=$), and `GREATER_EQUAL` ($\ge$).

### 2.4 Quadratic Objective Representation

For convex QP, the Hessian is stored as a `SparseQuadraticExpression` of `QuadraticTerm { i, j, coefficient }` entries with $i \le j$ (upper triangular canonical form). The full dense Hessian is the symmetric completion:

$$
Q_{ij} = Q_{ji} = \text{coefficient of term } (i, j)
$$

Diagonal terms correspond to $\frac{1}{2} Q_{jj} x_j^2$ in the objective.

### 2.5 Model Validation Invariants

The `validate()` method enforces the following before any solver is invoked:

| Component | Invariant |
| :--- | :--- |
| Variable bounds | $l_j \le u_j$; no NaN; $l_j \ne +\infty$; $u_j \ne -\infty$ |
| Binary variables | $0 \le l_j \le u_j \le 1$ |
| Constraint coefficients | Finite; all `VarId` in range $[0, n)$ |
| RHS | Finite; no $\pm\infty$ |
| Objective coefficients | Finite; all `VarId` in range $[0, n)$ |
| Names | Non-empty names are unique within variables and within constraints |

---

## 3. Sparse Matrix Representation

### 3.1 Compressed Sparse Row (CSR)

A matrix $A \in \mathbb{R}^{m \times n}$ with $\text{nnz}$ nonzeros is stored in three arrays:

- `row_ptr` $\in \mathbb{N}^{m+1}$: `row_ptr[i]` is the index into `col_idx` and `values` where row $i$ begins. `row_ptr[0] = 0`, `row_ptr[m] = \text{nnz}`.
- `col_idx` $\in \mathbb{N}^{\text{nnz}}$: zero-indexed column position of each stored entry.
- `values` $\in \mathbb{R}^{\text{nnz}}$: the corresponding double-precision coefficient.

Row $i$ occupies indices $[\text{row\_ptr}[i], \text{row\_ptr}[i+1])$ in `col_idx` and `values`.

### 3.2 Compressed Sparse Column (CSC)

The column-compressed dual format stores:

- `col_ptr` $\in \mathbb{N}^{n+1}$: `col_ptr[j]` is the offset where column $j$ begins.
- `row_idx` $\in \mathbb{N}^{\text{nnz}}$: zero-indexed row position of each stored entry.
- `values` $\in \mathbb{R}^{\text{nnz}}$: corresponding coefficient.

### 3.3 Structural Invariants

Both formats enforce the following invariants, verified by `validate()`:

1. Pointer arrays are monotonically non-decreasing.
2. `ptr[0] = 0`, `ptr.back() = nnz`.
3. Index arrays satisfy $0 \le \text{col\_idx}[k] < n$ and $0 \le \text{row\_idx}[k] < m$.
4. Indices within each row/column are in strictly ascending order with no duplicates.
5. All stored `values` are finite (no NaN, no $\pm\infty$).

### 3.4 Sparse Matrix-Vector Product (SpMV)

The fundamental operation is:

$$
y = Ax, \quad y_i = \sum_{k=\text{row\_ptr}[i]}^{\text{row\_ptr}[i+1]-1} \text{values}[k] \cdot x[\text{col\_idx}[k]]
$$

executed in $O(\text{nnz})$ time. The transpose product $y = A^T x$ is analogously defined over the CSC representation or computed by transposing the CSR kernel.

### 3.5 Fill-in and Density

The structural density of $A$ is defined as:

$$
\rho = \frac{\text{nnz}}{m \cdot n}
$$

During sparse LU factorization, elimination introduces new nonzeros not present in $A$; these are called **fill-in** entries. Markowitz pivoting (Section 4.2) minimizes the number of such fill-in entries.

### 3.6 Memory Complexity

The memory footprint of a CSR matrix is:

$$
\text{Memory}_{\text{CSR}} = (m + 1) \cdot \text{sizeof}(\texttt{size\_t}) + \text{nnz} \cdot (\text{sizeof}(\texttt{size\_t}) + \text{sizeof}(\texttt{double}))
$$

For sparse problem classes where $\text{nnz} = O(n)$, this is $O(m + n) \ll O(m \cdot n)$, confirmed by the unit test `test_sparse_memory_scaling_subquadratic`.

### 3.7 CSR $\leftrightarrow$ CSC Conversion

Conversion is performed in $O(\text{nnz} + m + n)$ time and $O(\text{nnz} + m + n)$ space using a three-pass counting-sort algorithm without any dense intermediate allocation.

---

## 4. Sparse LU Factorization

### 4.1 Factorization Form

`SparseLU` implements sparse Gaussian elimination with partial pivoting, producing the decomposition:

$$
PA = LU
$$

where:

- $P$ is an $n \times n$ permutation matrix encoding row swaps chosen during pivoting.
- $L$ is a unit lower triangular sparse matrix ($L_{ii} = 1$ for all $i$).
- $U$ is an upper triangular sparse matrix with nonzero diagonal.

### 4.2 Markowitz Pivoting

At elimination step $k \in \{0, \dots, n-1\}$, the **Markowitz criterion** selects the pivot row $p^*$ as the row among candidates $p \ge k$ that maximizes the absolute pivot magnitude:

$$
p^* = \arg\max_{p \ge k} |A_{pk}|
$$

This partial pivoting strategy simultaneously provides numerical stability (large pivots reduce round-off growth) and implicitly bounds fill-in by concentrating heavy rows late in the elimination order. If $|A_{p^* k}| < \tau_{\text{pivot}}$ (default $10^{-9}$), the factorization halts with `NUMERICAL_FAILURE`.

### 4.3 Row Elimination

At step $k$, row $i > k$ is updated by:

$$
\text{row}_i \leftarrow \text{row}_i - m_{ik} \cdot \text{row}_k, \quad m_{ik} = \frac{A_{ik}}{A_{kk}}
$$

using a two-pointer linear merge of sorted sparse rows in $O(\text{nnz}(\text{row}_i) + \text{nnz}(\text{row}_k))$ time. Entries with magnitude below $10^{-15}$ are dropped to suppress fill-in drift.

### 4.4 Direct Solve ($Ax = b$)

Given the factorization $PA = LU$, the system $Ax = b$ is solved in three steps:

1. **Permute RHS**: $y^{(0)} = Pb$, i.e., $y^{(0)}_i = b_{\text{perm}[i]}$.
2. **Forward solve**: Solve $Ly = y^{(0)}$ (unit lower triangular; no division needed).
3. **Back solve**: Solve $Ux = y$ (upper triangular; divide by $U_{ii}$ at each step).

### 4.5 Transpose Solve ($A^T x = b$)

Using $A^T = U^T L^T P$:

1. Solve $U^T z = b$ via row-push forward elimination.
2. Solve $L^T w = z$ via row-push backward elimination.
3. Permute: $x_{\text{perm}[i]} = w_i$.

### 4.6 Iterative Refinement

To recover precision lost to floating-point round-off, iterative refinement is applied after the initial solve:

1. Compute exact sparse residual: $r_0 = b - A x_0$.
2. Test: if $\|r_0\|_\infty \le \epsilon_{\text{residual}}$ (default $10^{-8}$), accept $x_0$.
3. Solve correction equation: $A \delta_k = r_k$ using the existing LU factors.
4. Update: $x_{k+1} = x_k + \delta_k$.
5. Recompute residual and repeat until $\|r_k\|_\infty \le \epsilon_{\text{residual}}$, or until the maximum iteration count is reached, or stagnation is detected.

The correction equation is:

$$
A \delta_k = r_k, \quad r_k = b - A x_k
$$

### 4.7 Condition Estimation

The approximate infinity-norm condition number of the basis matrix $B$ is estimated as:

$$
\hat{\kappa}_\infty(B) \approx \frac{\|B\|_\infty}{\min_i |U_{ii}|}
$$

If $\min_i |U_{ii}| < \epsilon_{\text{pivot}}$, the basis is diagnosed as numerically singular and the solver returns `NUMERICAL_FAILURE`.

---

## 5. Ruiz Equilibration Scaling

### 5.1 Purpose

Kavach-Opt applies geometric-mean equilibration (a form of Ruiz scaling) to the constraint matrix $A$ before factorization when the dynamic range ratio $\kappa_{\text{range}} = a_{\max} / a_{\min}$ satisfies $\kappa_{\text{range}} \ge 10^4$ (Rule R006). Scaling reduces the effective condition number of the basis matrix, stabilizing LU pivoting.

### 5.2 Diagonal Scaling Matrices

Two diagonal matrices $D_r = \text{diag}(r_1, \dots, r_m)$ and $D_c = \text{diag}(c_1, \dots, c_n)$ are constructed such that the scaled matrix $\tilde{A} = D_r A D_c$ has rows and columns with approximately unit norm.

### 5.3 Geometric-Mean Scaling Algorithm

**Row scalers** (first pass, applied to $A$):

$$
r_i = \frac{1}{\sqrt{\min_{j : A_{ij} \ne 0} |A_{ij}| \cdot \max_{j : A_{ij} \ne 0} |A_{ij}|}}
$$

**Column scalers** (second pass, applied to the row-scaled matrix $A' = D_r A$):

$$
c_j = \frac{1}{\sqrt{\min_{i : A'_{ij} \ne 0} |A'_{ij}| \cdot \max_{i : A'_{ij} \ne 0} |A'_{ij}|}}
$$

The fully scaled matrix is $\tilde{A} = D_r A D_c$.

### 5.4 Scaled Problem and Unscaling

The solver operates on the scaled system. After a scaled primal solution $\tilde{x}$ and scaled dual multipliers $\tilde{y}$ are obtained, the original-scale quantities are recovered by:

| Quantity | Unscaling Formula |
| :--- | :--- |
| Primal solution | $x_j = c_j \tilde{x}_j$ |
| Dual multipliers | $y_i = r_i \tilde{y}_i$ |
| Reduced costs | $d_j = \tilde{d}_j / c_j$ |
| Variable bounds | $\tilde{l}_j = l_j / c_j$, $\tilde{u}_j = u_j / c_j$ |
| RHS | $\tilde{b}_i = r_i b_i$ |

### 5.5 Conditioning Taxonomy

Based on $\kappa_{\text{range}} = a_{\max} / \max(10^{-15}, a_{\min})$:

| Category | Condition |
| :--- | :--- |
| `WELL_CONDITIONED` | $\kappa_{\text{range}} < 10^4$ |
| `MODERATE` | $10^4 \le \kappa_{\text{range}} < 10^8$ |
| `ILL_CONDITIONED` | $10^8 \le \kappa_{\text{range}} < 10^{13}$ |
| `SEVERELY_ILL_CONDITIONED` | $\kappa_{\text{range}} \ge 10^{13}$ |

---

## 6. Presolve Reductions

The presolve engine reduces the model before passing it to a solver, exploiting structural properties to eliminate variables and constraints. The original `CanonicalModel` is never mutated. A `ReconstructionMap` records every substitution so the original-space solution can be recovered in $O(n_{\text{orig}})$ time. Reductions execute in a fixed-point loop (up to 20 passes) until convergence.

### 6.1 Reduction A — Fixed-Variable Elimination

**Condition:** $|u_j - l_j| \le \epsilon_{\text{primal}}$.

Variable $x_j$ is fixed at $x_j^* = \tfrac{1}{2}(l_j + u_j)$. Effects:

- **Objective offset:** $c_0' = c_0 + c_j x_j^*$; $c_j \leftarrow 0$.
- **Quadratic diagonal:** $c_0' \mathrel{+}= Q_{jj} (x_j^*)^2$.
- **Quadratic off-diagonal:** $c_k' = c_k + Q_{jk} x_j^*$ for all $k \ne j$.
- **Constraint RHS:** $b_i' = b_i - A_{ij} x_j^*$; term $A_{ij}$ removed.
- **Reconstruction:** Variable marked `FIXED_REMOVED` with value $x_j^*$.

### 6.2 Reduction B — Empty Row Removal

**Condition:** All variable terms of constraint $i$ have been eliminated.

The row degenerates to $0 \sim b_i$. Feasibility is tested:
- If the sense is violated (e.g., $0 > b_i + \epsilon_{\text{primal}}$ for a $\le$ constraint), the presolve engine returns status **INFEASIBLE**.
- Otherwise the row is deleted.

### 6.3 Reduction C — Unconstrained Variable Analysis

**Condition:** Variable $x_j$ does not appear in any remaining constraint ($A_{*j} = 0$) and has no quadratic terms.

- If $c_j = 0$: $x_j^* = \text{clamp}(0, l_j, u_j)$; variable eliminated.
- If $c_j > 0$ (minimization): optimal at $x_j^* = l_j$ if finite; else **UNBOUNDED**.
- If $c_j < 0$ (minimization): optimal at $x_j^* = u_j$ if finite; else **UNBOUNDED**.

### 6.4 Reduction D — Singleton Row Bound Tightening

**Condition:** Constraint $i$ contains exactly one nonzero term $a_k x_k \sim b$.

Derived implied bound: $\beta = b / a_k$.

- If $a_k > 0$: $(\le) \Rightarrow u_k' = \min(u_k, \beta)$; $(\ge) \Rightarrow l_k' = \max(l_k, \beta)$; $(=) \Rightarrow$ both.
- If $a_k < 0$: directions reverse.

If $l_k' > u_k' + \epsilon_{\text{primal}}$ after tightening, the solver returns **INFEASIBLE**. The singleton row is deleted.

### 6.5 Reduction E — Activity Range Bounds and Redundancy Detection

For constraint row $i$, let $P_i = \{j \mid A_{ij} > 0\}$, $N_i = \{j \mid A_{ij} < 0\}$.

$$
L_i = \sum_{j \in P_i} A_{ij} l_j + \sum_{j \in N_i} A_{ij} u_j, \quad
U_i = \sum_{j \in P_i} A_{ij} u_j + \sum_{j \in N_i} A_{ij} l_j
$$

Infinite bounds are tracked via counters to prevent IEEE-754 NaN propagation.

- **Direct infeasibility:** Row $\le b_i$ with $L_i > b_i + \epsilon_{\text{primal}}$, or row $\ge b_i$ with $U_i < b_i - \epsilon_{\text{primal}}$, etc. → **INFEASIBLE**.
- **Redundancy:** Row $\le b_i$ with $U_i \le b_i + \epsilon_{\text{primal}}$ (no infinite contributors) → deleted.
- **Bound propagation:** For variable $k$ in a $\le b_i$ row where all other terms are fully bounded:

$$
\text{avail}_k = b_i - (L_i - \text{term\_min}(A_{ik}, x_k))
$$

  Implying $u_k' = \text{avail}_k / A_{ik}$ (when $A_{ik} > 0$) or $l_k' = \text{avail}_k / A_{ik}$ (when $A_{ik} < 0$).

### 6.6 Solution Reconstruction

Every original variable $x_j$ carries a mapping status:
- `ACTIVE`: Survived into the reduced model at reduced index $k$.
- `FIXED_REMOVED`: Restored to constant $x_j^*$.
- `UNCONSTRAINED_REMOVED`: Restored to its optimal bound value.

The `restore_solution` procedure maps reduced solution $x_{\text{red}}$ back to $x_{\text{orig}}$ in a single $O(n_{\text{orig}})$ pass. The objective preservation invariant holds within machine precision:

$$
c_{\text{red}}^T x_{\text{red}} + \tfrac{1}{2} x_{\text{red}}^T Q_{\text{red}} x_{\text{red}} + \text{offset}_{\text{red}}
\;\equiv\;
c_{\text{orig}}^T x_{\text{orig}} + \tfrac{1}{2} x_{\text{orig}}^T Q_{\text{orig}} x_{\text{orig}} + \text{offset}_{\text{orig}}
$$

---

## 7. Two-Phase Revised Simplex LP

### 7.1 Standard Form

The LP solver operates on the equality standard form obtained by transforming all variables and constraints of the canonical model:

$$
\min_{z \ge 0} \; c_{\text{std}}^T z \quad \text{s.t.} \quad A_{\text{std}} z = b_{\text{std}}, \quad z \ge 0, \quad b_{\text{std}} \ge 0
$$

Variable transformations map each canonical variable $x_j$ to one or more non-negative standard-form variables $z_k$:

| Canonical Bound Structure | Standard Substitution |
| :--- | :--- |
| Fixed ($\|u_j - l_j\| \le \epsilon$) | $x_j = l_j$ (substituted; not a variable) |
| Lower-bounded only ($l_j > -\infty$, $u_j = +\infty$) | $z_k = x_j - l_j \ge 0$ |
| Upper-bounded only ($l_j = -\infty$, $u_j < +\infty$) | $z_k = u_j - x_j \ge 0$ |
| Boxed ($l_j, u_j$ both finite) | $z_k = x_j - l_j \ge 0$; upper constraint $z_k + s = u_j - l_j$ |
| Free ($l_j = -\infty$, $u_j = +\infty$) | $x_j = z_{k_1} - z_{k_2}$, $z_{k_1}, z_{k_2} \ge 0$ |

Inequality constraints become equalities via slack ($\le$) or surplus + artificial ($\ge$) variables.

### 7.2 Basis Representation

At any simplex iteration, the variable index set $\{1, \dots, n_{\text{std}}\}$ is partitioned into a **basis** $\mathcal{B}$ of $m$ indices and a **nonbasic** set $\mathcal{N}$. The basis matrix is $B = [A_{\cdot j}]_{j \in \mathcal{B}} \in \mathbb{R}^{m \times m}$. The current basic feasible solution is:

$$
z_B = B^{-1} b, \quad z_N = 0
$$

All linear system solves against $B$ and $B^T$ are executed via `SparseLU` (Section 4).

### 7.3 Reduced Costs

The dual multipliers $y$ and reduced costs $\bar{c}_j$ are computed at every iteration:

$$
y = B^{-T} c_B
$$

$$
\bar{c}_j = c_j - c_B^T B^{-1} a_j = c_j - y^T a_j \quad \forall j \in \mathcal{N}
$$

Optimality condition (Phase II): $\bar{c}_j \ge -\epsilon_{\text{opt}}$ for all $j \in \mathcal{N}$.

### 7.4 Phase I — Feasibility

Phase I minimizes the sum of artificial variables $W = \sum_{k \in \mathcal{W}} w_k$. At each Phase I iteration:

1. Solve $B^T y = c_B^{\text{phase1}}$ for the Phase I dual multipliers.
2. Compute Phase I reduced costs $\bar{c}_j^{\text{phase1}} = c_j^{\text{phase1}} - y^T a_j$.
3. If $\bar{c}_j^{\text{phase1}} \ge -\epsilon_{\text{opt}}$ for all nonbasic $j$: terminate Phase I.
   - If $W^* > \epsilon_{\text{feas}}$: the original problem is **INFEASIBLE**.
   - If $W^* \le \epsilon_{\text{feas}}$: a basic feasible solution exists; proceed to Phase II.

Degenerate artificial variables remaining at zero value in the basis are purged: for each such artificial in row $r$, compute direction $v = B^{-T} e_r$ and pivot out using any non-artificial column $j$ with $|v^T a_j| > \epsilon_{\text{pivot}}$.

### 7.5 Phase II — Optimality

Starting from the Phase I feasible basis:

1. Solve $B^T y = c_B$ for dual multipliers.
2. Compute reduced costs $\bar{c}_j = c_j - y^T a_j$ for all $j \in \mathcal{N}$.
3. If $\bar{c}_j \ge -\epsilon_{\text{opt}}$ for all $j$: **OPTIMAL**; record $x^* = B^{-1} b$.
4. Select entering variable $q$ (see Section 7.6 for pivoting rules).
5. Solve direction $d = B^{-1} a_q$.
6. If $d_i \le \epsilon_{\text{pivot}}$ for all $i$: **UNBOUNDED**.
7. Minimum-ratio test: $\theta^* = \min_{i : d_i > \epsilon_{\text{pivot}}} \dfrac{(z_B)_i}{d_i}$.
8. Pivot: update $z_B \leftarrow z_B - \theta^* d$; $(z_B)_{p} \leftarrow \theta^*$; swap $q$ into basis at row $p$.

### 7.6 Entering Variable Rule and Anti-Cycling

**Dantzig rule (default):** Select $q = \arg\min_{j \in \mathcal{N}} \bar{c}_j$ (most negative reduced cost), maximizing per-step progress.

**Bland's anti-cycling rule (fallback):** A consecutive-degenerate-pivot counter tracks pivots with $\theta^* \le \epsilon_{\text{feas}}$. When this counter reaches `bland_watchdog_threshold` (default 20), Bland's rule is activated:

- Entering: smallest index $j$ with $\bar{c}_j < -\epsilon_{\text{opt}}$.
- Leaving: smallest index basic variable among minimum-ratio ties.

Bland's rule guarantees finite termination on all LP instances by preventing cycling. The Dantzig rule is restored upon the first non-degenerate pivot.

### 7.7 Basis Update and $\eta$-Matrix Representation

After each pivot, the updated basis matrix $B' = B + (a_q - B e_p) e_p^T$ can be factored incrementally. Rather than refactoring $B'$ from scratch at every pivot, the solver accumulates $\eta$-matrix update factors of the form:

$$
B' = B \cdot \eta, \quad \eta = I + (B^{-1} a_q - e_p) e_p^T / (B^{-1} a_q)_p
$$

After a configured number of updates (the **refactorization interval**: 50 iterations for NORMAL risk, 20 for ELEVATED, 10 for HIGH — enforced absolutely by Rule R018), the basis is completely refactored from scratch using `SparseLU` to prevent accumulated numerical error.

---

## 8. Branch-and-Bound MILP

### 8.1 LP Relaxation

Given a MILP with integer index set $I$, the **LP relaxation** at a node replaces integrality constraints $x_j \in \mathbb{Z}$ with continuous constraints $x_j \in \mathbb{R}$ (equivalently, removes all restrictions in $I$). The relaxation provides a lower bound (for minimization) on the optimal integer objective:

$$
z_{\text{LP}}^* \le z_{\text{MILP}}^*
$$

The root relaxation is solved by cold-start Revised Simplex. If the root relaxation is infeasible, the MILP is infeasible. If the root relaxation solution is integer-feasible, it is optimal.

### 8.2 Variable Branching Strategies

At each fractional node, let $I_{\text{frac}} = \{j \in I : x_j^* - \lfloor x_j^* \rfloor \in (\epsilon_{\text{int}}, 1 - \epsilon_{\text{int}})\}$ denote the candidate set of fractional integer variables. For variable $j \in I_{\text{frac}}$, let $f_j = x_j^* - \lfloor x_j^* \rfloor$.

#### 1. Most-Fractional Branching
Classical baseline selecting the variable farthest from integrality:
$$
s_j^{\text{frac}} = \min(f_j,\; 1 - f_j), \quad j^* = \arg\max_{j \in I_{\text{frac}}} s_j^{\text{frac}}
$$

#### 2. Pseudocost Branching
Tracks historical objective degradation per unit of fractional change. For child LP solves or strong branching probes, let $\Delta z_j^-$ and $\Delta z_j^+$ denote the observed objective degradation (worsening) when branching down ($x_j \le \lfloor x_j^* \rfloor$) and up ($x_j \ge \lceil x_j^* \rceil$):
$$
\psi_j^- = \frac{\Delta z_j^-}{f_j}, \qquad \psi_j^+ = \frac{\Delta z_j^+}{1 - f_j}
$$
The cumulative pseudocosts $\Psi_j^-$ and $\Psi_j^+$ represent running averages over $N_j^-$ down-observations and $N_j^+$ up-observations:
$$
\Psi_j^- = \frac{1}{N_j^-} \sum_{k=1}^{N_j^-} \psi_{j,k}^-, \qquad \Psi_j^+ = \frac{1}{N_j^+} \sum_{k=1}^{N_j^+} \psi_{j,k}^+
$$
The expected directional degradations are $q_j^- = \Psi_j^- f_j$ and $q_j^+ = \Psi_j^+ (1 - f_j)$. Candidate variables are evaluated using the Achterberg product score with numerical safeguard $\epsilon_{\text{score}} = 10^{-6}$:
$$
\text{score}_j = \max(q_j^-,\; \epsilon_{\text{score}}) \cdot \max(q_j^+,\; \epsilon_{\text{score}})
$$

#### 3. Strong Branching
For the top-$K$ candidate variables (filtered by fractionality or pseudocost scores), speculative child LP relaxations are solved directly using warm-started dual simplex:
- Down probe: temporarily impose $u_j \leftarrow \lfloor x_j^* \rfloor$, solve child LP, observe $\Delta z_j^-$.
- Up probe: temporarily impose $l_j \leftarrow \lceil x_j^* \rceil$, solve child LP, observe $\Delta z_j^+$.
- Model bounds are strictly restored immediately after probing.
- **Node Infeasibility Detection:** If both down and up probes are proven infeasible, the entire subproblem node is pruned immediately without generating tree children.
- If a single probe is infeasible, that branch is discarded and the variable domain is tightened.
- Probed degradations are incorporated into the global pseudocost tracker.

#### 4. Reliability Branching (Hybrid Policy)
A variable $j$ is considered **reliable** if it has accumulated at least $\eta_{\text{rel}}$ observations in both directions:
$$
\text{reliable}(j) \iff \min(N_j^-,\; N_j^+) \ge \eta_{\text{rel}} \quad (\text{default } \eta_{\text{rel}} = 4)
$$
Reliability branching probes only unreliable candidates using strong branching until they reach threshold $\eta_{\text{rel}}$, subsequently utilizing pure pseudocosts at zero additional LP solve cost.

#### Deterministic Tie-Breaking
When candidates have identical branching scores, ties are resolved hierarchically:
1. Branching score (descending)
2. Fractionality $\min(f_j, 1 - f_j)$ (descending)
3. Total observations $N_j^- + N_j^+$ (descending)
4. Variable index `VarId` (ascending)

### 8.3 Node Prioritization with Depth Diving

The unexplored node priority queue is managed via a hierarchical Best-Bound rule:
1. **Primary Key:** LP relaxation objective bound $z_{\text{LP}}$ (min-heap for minimization, max-heap for maximization).
2. **Secondary Key (Depth Diving):** When multiple nodes share identical LP bounds (within $|z_a - z_b| \le 10^{-9}$), the search prioritizes deeper nodes:
$$
\text{depth}(a) > \text{depth}(b) \implies \text{priority}(a) > \text{priority}(b)
$$
Diving down identical-bound plateaus accelerates discovery of integer-feasible solutions, tightening $z_{\text{inc}}$ and enabling aggressive tree pruning.
3. **Deterministic Tie-Break:** Ascending unique node ID.

### 8.4 Primal Incumbent Heuristics

To find high-quality integer solutions early in the search, Kavach-Opt executes a dual-stage primal heuristic:

#### Stage 1: Feasibility-Oriented Rounding
For each fractional variable $x_j^*$, round toward the nearest integer, clamping to variable bounds $[l_j, u_j]$.

#### Stage 2: Bounded Continuous Slack Repair
For each constraint $i \in \{0, \dots, m-1\}$ with row activity $a_i^T x^{\text{heur}}$, compute constraint deficit:
$$
d_i = \begin{cases}
a_i^T x^{\text{heur}} - b_i^{\text{ub}} & \text{if } a_i^T x^{\text{heur}} > b_i^{\text{ub}} + \epsilon_{\text{primal}} \\
b_i^{\text{lb}} - a_i^T x^{\text{heur}} & \text{if } a_i^T x^{\text{heur}} < b_i^{\text{lb}} - \epsilon_{\text{primal}} \\
0 & \text{otherwise}
\end{cases}
$$
If $d_i > 0$, identify non-basic continuous slack variables with non-zero coefficients $A_{ij}$ and shift their values within bounds $[l_j, u_j]$ to absorb the deficit:
$$
\Delta x_j = \operatorname{clamp}\!\left(\frac{-d_i}{A_{ij}},\; l_j - x_j^{\text{heur}},\; u_j - x_j^{\text{heur}}\right)
$$
The candidate is subjected to strict verification: all linear constraints and integrality requirements must be satisfied within $\epsilon_{\text{primal}}$ and $\epsilon_{\text{int}}$ before the incumbent is updated.

### 8.5 Pruning Conditions

A node is pruned (no children generated) under any of:
1. **Infeasibility:** The node LP relaxation is infeasible.
2. **Bound dominance:** $z_{\text{LP}}(\text{node}) \ge z_{\text{inc}} - \epsilon_{\text{bound}}$ (for minimization).
3. **Integer feasibility:** The LP relaxation solution is integer-feasible; it updates the incumbent if superior.
4. **Strong branching double-infeasibility:** Both probing branches prove infeasible.

### 8.6 Optimality Gap

The optimality gap is computed as:

$$
\text{gap} = \frac{|z_{\text{inc}} - z_{\text{bound}}|}{\max(1.0,\; |z_{\text{inc}}|)}
$$

A zero gap is reported only when the search tree is exhausted or $\text{gap} \le \epsilon_{\text{gap}}$. The solver never claims `OPTIMAL` unless the independent verification layer confirms integrality, feasibility, and gap conditions (Section 11).

### 8.7 Cutting-Plane Separation Engine (Branch-and-Cut)

The pure Branch-and-Bound algorithm partitions the search space when fractional variables arise. To accelerate convergence and reduce tree exploration, Kavach-Opt integrates a sovereign Cutting-Plane Separation Engine into the root relaxation and search nodes (Branch-and-Cut).

Cutting planes are valid linear inequalities $\alpha^T x \le \beta$ satisfied by all integer-feasible solutions $x \in P \cap \mathbb{Z}^n$ but violated by the continuous LP relaxation solution $x^*$:

$$
\alpha^T x \le \beta \quad \forall x \in \operatorname{conv}(P \cap \mathbb{Z}^n), \qquad \alpha^T x^* > \beta
$$

Adding these cuts strictly tightens the continuous polyhedral relaxation without eliminating any feasible mixed-integer solutions.

#### 8.7.1 Gomory Fractional and Mixed-Integer Cuts (GMI)

Let the optimal basic solution of the LP relaxation in standard form be represented as $x_B = \bar{b} - \bar{A}_N x_N$, where $\bar{b} = B^{-1} b$ and $\bar{A}_N = B^{-1} N$. For any basic integer variable $x_{B(i)}$ ($i \in \{1,\ldots,m\}$) that takes a fractional value in the relaxation:

$$
x_{B(i)} + \sum_{j \in \mathcal{N}} \bar{a}_{i,j} x_j = \bar{b}_i
$$

The tableau row coefficients $\bar{a}_{i,j}$ and RHS $\bar{b}_i$ are extracted numerically via the dual system:

$$
B^T u = e_i \implies \bar{b}_i = u^T b, \quad \bar{a}_{i,j} = u^T a_j
$$

where $e_i$ is the $i$-th elementary unit vector and $u$ is obtained from the basis factorizer.

Let $f_0 = \bar{b}_i - \lfloor \bar{b}_i \rfloor$ denote the fractional part of the RHS, with $f_0 \in (0, 1)$. For each nonbasic variable $j \in \mathcal{N}$, let $f_j = \bar{a}_{i,j} - \lfloor \bar{a}_{i,j} \rfloor$.

**Case 1: Pure Integer Variables ($j \in \mathcal{I}$).**
Gomory's fractional cutting plane takes the form:

$$
\sum_{j \in \mathcal{N} \cap \mathcal{I}} f_j x_j \ge f_0 \iff -\sum_{j \in \mathcal{N} \cap \mathcal{I}} f_j x_j \le -f_0
$$

**Case 2: Mixed-Integer Case (Gomory Mixed-Integer Cut — GMI).**
When continuous nonbasic variables $\mathcal{C}$ are present, the split disjunction yields the GMI cut:

$$
\sum_{j \in \mathcal{I}, f_j \le f_0} f_j x_j + \frac{f_0}{1 - f_0} \sum_{j \in \mathcal{I}, f_j > f_0} (1 - f_j) x_j + \sum_{j \in \mathcal{C}, \bar{a}_{i,j} \ge 0} \bar{a}_{i,j} x_j - \frac{f_0}{1 - f_0} \sum_{j \in \mathcal{C}, \bar{a}_{i,j} < 0} \bar{a}_{i,j} x_j \ge f_0
$$

**Projection to Original Variable Space:**
Because standard-form variables $z$ are related to the original canonical model variables $x$ via affine transformations $z_k = s_k x_{v(k)} + d_k$ (accounting for lower bounds and split free variables), the generated cut $\sum_k \gamma_k z_k \ge \gamma_0$ is projected directly back to original model space:

$$
a_{\text{orig}}^T x \le b_{\text{orig}}
$$

#### 8.7.2 Mixed-Integer Rounding (MIR) Cuts

Kavach-Opt derives MIR cuts directly from original mixed-integer constraints $\sum_{j \in \mathcal{I}} a_j x_j + \sum_{j \in \mathcal{C}} g_j y_j \le b$.

1. **Non-Negative Bound Shifting:**
   For bounded integer variables, let $x_j' = x_j - l_j \ge 0$. The inequality shifts to:
   $$
   \sum_{j \in \mathcal{I}} a_j x_j' + \sum_{j \in \mathcal{C}} g_j (y_j - l_{y_j}) \le b - \sum_{j \in \mathcal{I}} a_j l_j - \sum_{j \in \mathcal{C}} g_j l_{y_j} \equiv \beta_0
   $$
2. **Multi-Scaling:**
   Given a positive scaling parameter $\delta > 0$, scale the constraint by $\delta$. Let $\beta = \delta \beta_0$, and fractionality $\alpha = \beta - \lfloor \beta \rfloor \in (0, 1)$.
   Candidate scaling factors evaluated include $\delta \in \{1.0, 0.5, 2.0\}$ and $\delta = 1 / |a_k|$ for dominant integer coefficients.
3. **MIR Function Application:**
   Define the MIR scalar function $F_\alpha: \mathbb{R} \to \mathbb{R}$:
   $$
   F_\alpha(d) = \lfloor d \rfloor + \frac{\max(0,\; d - \lfloor d \rfloor - \alpha)}{1 - \alpha}
   $$
   The resulting valid MIR inequality is:
   $$
   \sum_{j \in \mathcal{I}} F_\alpha(\delta a_j) x_j' + \frac{1}{1 - \alpha} \sum_{j \in \mathcal{C}, \delta g_j < 0} \delta g_j y_j' \le \lfloor \beta \rfloor
   $$
4. **Substitution & Normalization:**
   Substituting $x_j' = x_j - l_j$ back yields a valid cut in the original variables.

#### 8.7.3 Cut Filtering, Efficacy, and Orthogonality

To prevent numerical degradation and ill-conditioning, candidate cuts are evaluated through rigorous geometric and numerical filters before entering the working model:

1. **Euclidean Efficacy:**
   The geometric distance from the fractional point $x^*$ to the cut hyper-plane $a^T x \le b$ is:
   $$
   \gamma(a, b, x^*) = \frac{a^T x^* - b}{\|a\|_2}
   $$
   A candidate cut is accepted only if $\gamma(a, b, x^*) \ge \epsilon_{\text{efficacy}} = 10^{-4}$.
2. **Cosine Parallelism Deduplication:**
   For each new candidate cut $a_{\text{new}}$ against an existing cut $a_{\text{pool}}$ in the pool:
   $$
   \cos(\theta) = \frac{|a_{\text{new}}^T a_{\text{pool}}|}{\|a_{\text{new}}\|_2 \|a_{\text{pool}}\|_2}
   $$
   If $\cos(\theta) > 0.999$, the cut is deemed parallel/redundant and discarded.
3. **Numerical Safety & Dynamic Range:**
   - Coefficients with $|a_j| < 10^{-12}$ are pruned to exact zeros.
   - Any cut with $\max_j |a_j| > 10^7$ or $\min_{j, a_j \ne 0} |a_j| < 10^{-7}$ is rejected.
   - Dynamic range ratio is strictly bounded:
     $$
     \kappa_{\text{cut}} = \frac{\max_{j, a_j \ne 0} |a_j|}{\min_{j, a_j \ne 0} |a_j|} \le 10^6
     $$
4. **Tail-Off Detection:**
   Root cut separation terminates when successive LP relaxation iterations yield negligible bound improvement:
   $$
   \frac{z_{\text{LP}}^{(k)} - z_{\text{LP}}^{(k-1)}}{\max(1.0,\; |z_{\text{LP}}^{(k-1)}|)} < 10^{-4}
   $$

---

## 9. Dual Simplex Method

### 9.1 Complementary Duality

The Revised Simplex maintains a basis $B$ such that the primal basic solution $z_B = B^{-1} b$ and the dual solution $y = B^{-T} c_B$ satisfy complementary slackness at optimality. A basis is called:

- **Primal feasible:** $z_B = B^{-1} b \ge 0$.
- **Dual feasible:** $\bar{c}_j = c_j - y^T a_j \ge 0$ for all $j \in \mathcal{N}$.
- **Optimal:** both conditions hold simultaneously.

### 9.2 Role of Dual Simplex in Warm Starts

In Branch-and-Bound, a child node's LP relaxation differs from its parent only by tightened variable bounds. This changes the right-hand side $b$ in standard form but leaves $A$ and $c$ unchanged. Consequently the parent's optimal basis satisfies:

$$
y = B^{-T} c_B \quad \text{and} \quad \bar{c}_N = c_N - N^T y \ge -\epsilon_{\text{dual}}
$$

at the child node (dual feasibility is **inherited**), while the primal basic values $z_B = B^{-1} b'$ may have negative components (primal infeasibility introduced by bound tightening). Dual simplex restores primal feasibility while maintaining dual feasibility.

### 9.3 Leaving Variable Selection

The leaving row $l$ is chosen by maximum primal violation:

$$
l = \arg\max_{i} \left\{-z_{B(i)} \;\Big|\; z_{B(i)} < -\epsilon_{\text{primal}}\right\}
$$

Ties resolved by smallest basic variable index.

### 9.4 Dual Minimum-Ratio Test

The entering column direction is $v = B^{-T} e_l$; the $j$-th tableau element is $\alpha_j = v^T a_j$. The entering column $k$ maintains dual feasibility after the pivot if:

$$
d_j' = d_j - \frac{\alpha_j}{\alpha_k} d_k \ge 0 \quad \forall j \in \mathcal{N} \text{ with } \alpha_j < -\epsilon_{\text{pivot}}
$$

This requires:

$$
k = \arg\min_{j \in \mathcal{N},\; \alpha_j < -\epsilon_{\text{pivot}}} \frac{\max(0, d_j)}{-\alpha_j}
$$

Ties are broken by smallest $j$.

### 9.5 Primal Infeasibility — Farkas Ray

If $\alpha_j \ge -\epsilon_{\text{pivot}}$ for all $j \in \mathcal{N}$ when a leaving row $l$ still has $z_{B(l)} < 0$, then the vector $v = B^{-T} e_l$ constitutes a **Farkas certificate of infeasibility**: $v^T A \ge 0$ and $v^T b < 0$. The node is immediately certified **INFEASIBLE** without further pivoting.

### 9.6 Fallback Contract

If the inherited basis fails the dual feasibility check ($\bar{c}_j < -\epsilon_{\text{dual}}$ for any $j$), if basis dimensions mismatch between parent and child, or if `SparseLU` factorization fails, the dual simplex engine aborts and the node LP is re-solved from scratch using cold-start primal Revised Simplex. This fallback is always safe and lossless.

---

## 10. Convex Quadratic Programming — Active-Set

### 10.1 Problem Statement

The convex QP solver addresses:

$$
\min_{x \in \mathbb{R}^n} \; f(x) = \frac{1}{2} x^T Q x + c^T x + c_0
$$

$$
\text{s.t.} \quad b^l \le Ax \le b^u, \quad l \le x \le u
$$

where $Q \in \mathbb{R}^{n \times n}$ is symmetric and is required to satisfy $Q \succeq 0$ (positive semidefiniteness). Non-convex ($Q \not\succeq 0$) instances are rejected with `INVALID_INPUT` before any iteration begins.

### 10.2 PSD Verification

Before any active-set iteration, `SparseLU` performs a diagonal-pivoted Cholesky factorization $P^T Q P = LDL^T$. If any diagonal element $D_{ii} < -\epsilon_{\text{PSD}}$, a negative eigenvalue is detected and the solve is aborted immediately. PSD verification is also repeated independently by `QPVerifier` (Section 11).

### 10.3 Active-Set Maintenance

Let $\mathcal{W}_k \subseteq \mathcal{E} \cup \mathcal{I}$ be the **working set** of constraints treated as equalities at iteration $k$. At each iteration, given feasible iterate $x_k$:

**Step direction.** Minimize $\frac{1}{2} p^T Q p + g_k^T p$ subject to $A_{\mathcal{W}} p = 0$, where $g_k = Qx_k + c$. This yields the KKT saddle-point system:

$$
\begin{bmatrix} Q & A_{\mathcal{W}}^T \\ A_{\mathcal{W}} & 0 \end{bmatrix}
\begin{bmatrix} p_k \\ -\lambda_{\mathcal{W}} \end{bmatrix}
=
\begin{bmatrix} -g_k \\ 0 \end{bmatrix}
$$

The KKT matrix is solved via `SparseLU` with iterative refinement (convergence tolerance $10^{-10}$). A small Tikhonov regularization $\delta = 10^{-12}$ is applied to the $(1,1)$ block for rank-deficient PSD matrices.

**Step length.** If $p_k \ne 0$ and no inactive inequality constraint blocks the step, take $\alpha_k = 1$. Otherwise:

$$
\alpha_k = \min\!\left(1,\; \min_{i \notin \mathcal{W}_k,\; a_i^T p_k > 0} \frac{b_i - a_i^T x_k}{a_i^T p_k}\right)
$$

The blocking constraint achieving this minimum is added to $\mathcal{W}_{k+1}$.

### 10.4 KKT Optimality Conditions

If $\|p_k\| \le \epsilon_{\text{step}}$ (stationary point for the current working set):

- If all inequality multipliers satisfy $\lambda_i \ge -\epsilon_{\text{dual}}$ for $i \in \mathcal{W}_k \cap \mathcal{I}$, the first-order necessary conditions are satisfied:

$$
Qx^* + c = A^T \lambda^*, \quad \lambda^* \ge 0, \quad \lambda_i^*(a_i^T x^* - b_i) = 0
$$

  The point $x^*$ is the global optimum (since $Q \succeq 0$).

- Otherwise, drop the constraint $i^* = \arg\min_i \lambda_i$ from the working set: $\mathcal{W}_{k+1} = \mathcal{W}_k \setminus \{i^*\}$, and continue.

### 10.5 LICQ Enforcement

Before a constraint is added to the working set, its normal vector is projected against the existing orthonormal basis of $A_{\mathcal{W}}^T$ via Modified Gram-Schmidt. If the orthogonal residual satisfies:

$$
\|a_{\text{cand}}^\perp\|_2 < 10^{-6} \|a_{\text{cand}}\|_2
$$

the candidate constraint is linearly dependent and rejected, ensuring $A_{\mathcal{W}}$ maintains full row rank throughout (Linear Independence Constraint Qualification).

### 10.6 Anti-Cycling

When $\alpha_k = 0$ (blocking constraint with zero slack), the active set could cycle. Kavach-Opt prevents cycling by maintaining a 64-bit hash signature of the working set $\mathcal{W}_k$ across iterations. If a signature repeats, Bland-style deterministic tie-breaking is engaged. If cycles persist beyond the iteration threshold, the solver terminates with `LIMIT_REACHED`.

### 10.7 Convex Mixed-Integer Quadratic Programming (MIQP)

Kavach-Opt extends continuous active-set quadratic optimization to discrete domains via a rigorous Branch-and-Bound framework:

$$
\min_{x \in \mathbb{R}^n} \quad f(x) = \frac{1}{2} x^T Q x + c^T x + c_0
$$

$$
\text{s.t.} \quad A x \le b, \quad l \le x \le u, \quad x_j \in \mathbb{Z} \; (j \in \mathcal{I})
$$

where $Q \in \mathbb{R}^{n \times n}$ is symmetric positive semidefinite ($Q \succeq 0$).

#### 10.7.1 Active-Set QP Node Relaxations
At each node of the search tree, the continuous relaxation is a convex QP subject to the inherited variable bounds $[l_j^{\text{node}}, u_j^{\text{node}}]$. The relaxation is solved using `PrimalActiveSetQPSolver`. In contrast to LP simplex where bound branching preserves dual feasibility, bound tightening in QP immediately shifts the unconstrained minimizer and alters the active set. Thus, each node undergoes a numerically stable active-set solve initialized via internal phase-1 feasibility projection.

#### 10.7.2 Branching Strategies and Pseudocost Scoring
Branching candidates are evaluated using:
1. **Pseudocosts:** Directional degradation tracking $\Psi_j^-, \Psi_j^+$:
   $$
   \Psi_j^- = \frac{z(\lfloor x_j^* \rfloor) - z^*}{x_j^* - \lfloor x_j^* \rfloor}, \quad \Psi_j^+ = \frac{z(\lceil x_j^* \rceil) - z^*}{\lceil x_j^* \rceil - x_j^*}
   $$
2. **Strong Branching:** Probes candidate integer variables by speculatively solving child QP relaxations with updated upper/lower bounds. Dual degradations are recorded into pseudocost history.
3. **Reliability Branching (Default):** Hybrid policy using strong branching probes while observation counts $N_j < \eta_{\text{rel}}$ (default $\eta_{\text{rel}} = 4$), switching to product score $s_j = (\max(\Psi_j^-, 10^{-6})) \cdot (\max(\Psi_j^+, 10^{-6}))$ once reliable.

#### 10.7.3 Quadratic Primal Heuristics
Integer rounding combined with bounded continuous slack repair is evaluated periodically. Every heuristic candidate is evaluated under the exact quadratic objective function $f(x) = \frac{1}{2} x^T Q x + c^T x + c_0$ and verified against all linear constraints and integrality requirements.

#### 10.7.4 Maximization Canonicalization
For concave quadratic maximization ($\max f(x)$ with $-Q \succeq 0$), the model is canonicalized to minimization:
$$
\min \quad -f(x) = \frac{1}{2} x^T (-Q) x - c^T x - c_0
$$
Since $-Q$ is positive semidefinite, the problem is solved via standard convex MIQP minimization, and optimal objectives and bounds are negated upon return. Non-convex instances ($Q$ indefinite) are detected via pivoted Cholesky factorization and rejected with `NUMERICAL_FAILURE`.

---

## 11. Independent Verification Layer

### 11.1 Architecture

The verification subsystem (`src/verification/`) operates as a mathematically isolated module. Its independence contract is:

1. **Zero solver re-use**: No call is made to any LP standard-form helper, `SimplexBasis`, `SparseLU`, or presolve routine. Every check is computed directly from the original (pre-presolve, unscaled) `CanonicalModel` using elementary arithmetic.
2. **No shared state**: The verifier receives only the solution vector $(x, y, rc, \text{obj}, \text{status})$ and the original `CanonicalModel`. It shares no memory or state with any solver.

The solver never reports `OPTIMAL` unless the verifier has confirmed all applicable conditions.

### 11.2 Check 1 — Non-Finite Sanity

All elements of $x$, $y$, and $rc$ are scanned for `NaN`, `+Inf`, or `-Inf` before any algebraic check:

$$
\forall k : \texttt{std::isfinite}(v_k) = \texttt{true}
$$

Failure immediately downgrades status to `NUMERICAL_FAILURE`.

### 11.3 Check 2 — Bound Feasibility

For every variable $j$:

$$
l_j - \epsilon_{\text{bound}} \le x_j \le u_j + \epsilon_{\text{bound}}
$$

where $\epsilon_{\text{bound}}$ equals the configured bound feasibility tolerance.

### 11.4 Check 3 — Primal Constraint Feasibility

The constraint activity is evaluated independently as:

$$
\text{activity}_i = \sum_{j \in \text{terms}_i} a_{ij} x_j
$$

Then for each constraint:

- $(\le)$: $\text{activity}_i - b_i \le \epsilon_{\text{primal}}$
- $(\ge)$: $b_i - \text{activity}_i \le \epsilon_{\text{primal}}$
- $(=)$: $|\text{activity}_i - b_i| \le \epsilon_{\text{primal}}$

### 11.5 Check 4 — Dual Feasibility and Reduced-Cost Stationarity

The verifier independently computes the reduced cost for each variable:

$$
\hat{rc}_j = c_j - \sum_{i=1}^{m} A_{ij} y_i
$$

At optimality, KKT stationarity requires:

- **Free / interior variables** ($l_j + \epsilon < x_j < u_j - \epsilon$): $|\hat{rc}_j| \le \epsilon_{\text{dual}}$.
- **Lower-bound active** ($x_j \le l_j + \epsilon$): $\hat{rc}_j \ge -\epsilon_{\text{dual}}$ (minimization).
- **Upper-bound active** ($x_j \ge u_j - \epsilon$): $\hat{rc}_j \le \epsilon_{\text{dual}}$ (minimization).
- **Fixed** ($|u_j - l_j| \le \epsilon$): $\hat{rc}_j$ unrestricted.

Constraint dual multipliers $y_i$ must satisfy sign conditions consistent with constraint sense and optimization direction.

### 11.6 Check 5 — Complementary Slackness

For each inequality constraint $i$, the slack is:

$$
\text{slack}_i = \begin{cases}
b_i - \text{activity}_i & i \in \mathcal{I}_{\le} \\
\text{activity}_i - b_i & i \in \mathcal{I}_{\ge} \\
0 & i \in \mathcal{I}_{=}
\end{cases}
$$

Complementary slackness requires:

$$
|\text{slack}_i \cdot y_i| \le \epsilon_{\text{comp}}
$$

### 11.7 Check 6 — Objective Consistency

The verifier independently evaluates:

$$
\text{obj}_{\text{verified}} = c_0 + \sum_{j \in \text{obj\_terms}} c_j x_j
$$

and enforces:

$$
|\text{obj}_{\text{verified}} - \text{obj}_{\text{reported}}| \le \epsilon_{\text{obj,abs}}
\;\;\lor\;\;
\frac{|\text{obj}_{\text{verified}} - \text{obj}_{\text{reported}}|}{\max(1.0,\; |\text{obj}_{\text{reported}}|)} \le \epsilon_{\text{obj,rel}}
$$

### 11.8 MILP-Specific Check — Integrality

For all $j \in I$:

$$
|x_j - \operatorname{round}(x_j)| \le \epsilon_{\text{int}}
$$

### 11.9 QP-Specific Check — KKT Residual

The Lagrangian is $\mathcal{L}(x, \lambda, \mu) = \frac{1}{2} x^T Q x + c^T x - \lambda^T (Ax - b) - \mu^T x$. Stationarity requires:

$$
\|Qx + c + A^T \lambda - \mu\|_\infty \le \epsilon_{\text{stat}}
$$

Complementary slackness for bound multipliers:

$$
\mu_j^l (x_j - l_j) \le \epsilon_{\text{comp}}, \quad \mu_j^u (u_j - x_j) \le \epsilon_{\text{comp}}
$$

PSD of $Q$ is independently re-certified by pivoted Cholesky.

### 11.10 MIQP-Specific Verification

The `MIQPVerifier` executes seven independent validation passes without re-using solver routines:
1. **Non-Finite Sanity:** Ensures all incumbent elements are finite ($\text{std::isfinite}(x_j) = \text{true}$).
2. **Bound Feasibility:** $l_j - \epsilon_{\text{bound}} \le x_j \le u_j + \epsilon_{\text{bound}}$ for all $j \in \{0, \dots, n-1\}$.
3. **Primal Constraint Feasibility:** Independent row activity evaluation $\text{activity}_i = \sum_j A_{ij} x_j$ checked against LHS/RHS bounds within $\epsilon_{\text{primal}}$.
4. **Integrality:** $|x_j - \operatorname{round}(x_j)| \le \epsilon_{\text{int}}$ for all discrete variables $j \in \mathcal{I}$, and $x_j \in [-\epsilon_{\text{int}}, 1 + \epsilon_{\text{int}}]$ for binary variables.
5. **Independent Quadratic Objective Recomputation:** Direct evaluation from scratch:
   $$
   f(x) = \frac{1}{2} \sum_{i,j} Q_{ij} x_i x_j + \sum_j c_j x_j + c_0
   $$
   verified against reported objective within absolute/relative tolerances ($\epsilon_{\text{obj,abs}} = 10^{-6}, \epsilon_{\text{obj,rel}} = 10^{-6}$).
6. **Optimality Gap Consistency:** Recomputed gap $\text{gap} = |z_{\text{inc}} - z_{\text{bound}}| / \max(1, |z_{\text{inc}}|) \le \epsilon_{\text{gap}}$ ($10^{-4}$).
7. **Independent Convexity / PSD Certification:** Complete diagonal-pivoted Cholesky factorization of $Q$ (or $-Q$ for maximization) verifying non-negative pivots ($D_{kk} \ge -\epsilon_{\text{psd}}$).

### 11.11 Certification Hierarchy

The verifier maps check results to a certified status:

| Conditions | Certified Status |
| :--- | :--- |
| All applicable checks pass | `VERIFIED_OPTIMAL` |
| Primal/bound feasible; dual or complementarity fails | `FEASIBLE` |
| Non-finite, bound violation, or integer violation | `VERIFICATION_FAILURE` |

---

## 12. Batched Sparse & Vector Numerical Kernel Suite

### 12.1 Mathematical Operation Definitions

Kavach-Opt provides a sovereign suite of seven high-throughput numerical operations accelerated across CPU sequential, CPU multicore, and CUDA backends:

1. **Batched CSR SpMV:**
   $$
   Y_k = A X_k \quad \text{for } k = 0, \dots, B - 1, \quad A \in \mathbb{R}^{m \times n} \text{ (CSR)}
   $$
2. **Batched CSR Transpose SpMV:**
   $$
   Y_k = A^T X_k \quad \text{for } k = 0, \dots, B - 1, \quad X_k \in \mathbb{R}^m, \; Y_k \in \mathbb{R}^n
   $$
3. **Batched CSC SpMV:**
   $$
   Y_k = A X_k \quad \text{for } k = 0, \dots, B - 1, \quad A \in \mathbb{R}^{m \times n} \text{ (CSC)}
   $$
4. **Batched Vector AXPBY:**
   $$
   Y_k \leftarrow \alpha X_k + \beta Y_k \quad \text{for } k = 0, \dots, B - 1, \quad X_k, Y_k \in \mathbb{R}^n
   $$
5. **Batched Dot Product:**
   $$
   s_k = X_k^T Y_k = \sum_{i=0}^{n-1} X_{k,i} Y_{k,i} \quad \text{for } k = 0, \dots, B - 1
   $$
6. **Batched $L_2$ Euclidean Norm:**
   $$
   s_k = \|X_k\|_2 = \sqrt{\sum_{i=0}^{n-1} X_{k,i}^2} \quad \text{for } k = 0, \dots, B - 1
   $$
7. **Batched $L_\infty$ Maximum Norm:**
   $$
   s_k = \|X_k\|_\infty = \max_{0 \le i < n} |X_{k,i}| \quad \text{for } k = 0, \dots, B - 1
   $$

### 12.2 CPU Thread-Pool Partition

For $B$ batch items and $W$ workers, the partition count is $P = \min(W, B)$. Worker $w \in \{0, \dots, P-1\}$ is assigned the contiguous batch slice:

$$
\text{start}_w = \left\lfloor \frac{w \cdot B}{P} \right\rfloor, \quad \text{end}_w = \left\lfloor \frac{(w+1) \cdot B}{P} \right\rfloor
$$

This guarantees exact coverage $\bigcup_{w} [\text{start}_w, \text{end}_w) = [0, B)$, zero false sharing on output arrays, and load imbalance of at most 1 item across workers.

### 12.3 CUDA Numerical Kernel Suite & Reduction Order Mathematics

When CUDA hardware is available and eligible (Rule R013), sovereign double-precision kernels accelerate these operations.

#### Floating-Point Reduction Order
Floating-point addition in IEEE 754 arithmetic is commutative but non-associative: $(a + b) + c \ne a + (b + c)$. Sequential evaluation on CPU sums items in index order:
$$
S_{\text{seq}} = (((x_0 + x_1) + x_2) + \dots)
$$
whereas parallel GPU reduction uses a binary tree with $O(\log n)$ depth:
$$
S_{\text{GPU}} = \sum_{\text{tree}} x_i
$$
The difference is strictly bounded by $O(\epsilon_{\text{mach}} \cdot \log n \cdot \|x\|_1)$. Consequently:
- **$L_\infty$ Norm:** The $\max$ operator on floating-point absolute values is both associative and commutative. GPU and CPU $L_\infty$ norms are bitwise identical ($\epsilon = 0$).
- **Sum-based Reductions (Dot Product, $L_2$ Norm, SpMV):** Governed by the reduction tolerance policy:
  $$
  |s_{\text{GPU}} - s_{\text{CPU}}| \le \epsilon_{\text{abs}} + \epsilon_{\text{rel}} \cdot \max(1.0, |s_{\text{CPU}}|)
  $$
  with $\epsilon_{\text{abs}} = 10^{-12}$ and $\epsilon_{\text{rel}} = 10^{-10}$.

### 12.4 Device Memory Safety Pre-Allocation Bound

Before launching any GPU kernel, Kavach-Opt estimates the required device memory $M_{\text{est}}$ and queries available VRAM:

$$
M_{\text{est}}^{\text{SpMV}} = (2 \cdot \text{nnz} + m + 1) \cdot 8 + B \cdot (m + n) \cdot 8 \quad \text{bytes}
$$
$$
M_{\text{est}}^{\text{vec}} = B \cdot n \cdot (\text{num\_vectors}) \cdot 8 + B \cdot (\text{num\_scalars}) \cdot 8 \quad \text{bytes}
$$

Execution is dispatched to GPU only if:

$$
M_{\text{est}} \le 0.85 \cdot M_{\text{free}}
$$

If $M_{\text{est}} > 0.85 \cdot M_{\text{free}}$, or if CUDA is unavailable, execution safely and deterministically routes to `CpuMulticoreBackend` without driver allocation faults.

### 12.5 Transfer-Inclusive GPU Timing Policy

All GPU performance measurements include host-to-device and device-to-host transfer time:

$$
T_{\text{GPU}} = T_{\text{H2D}} + T_{\text{kernel}} + T_{\text{D2H}}
$$

The kernel-only time $T_{\text{kernel}}$ is recorded purely as a diagnostic metric to quantify PCIe transfer overhead. The GPU execution path is selected only when Rule R013 confirms that empirical crossover data exists for $B \ge 128$ and ($\text{nnz} \ge 10{,}000$ or $N \ge 100{,}000$).

---

## 13. Deterministic Strategy Engine

### 13.1 Purpose

The strategy layer translates model structural properties into a concrete execution plan before any solver is invoked. It is deterministic: given identical model, configuration, and hardware, it produces an identical decision on every run. It uses no machine learning, no neural networks, and no non-deterministic heuristics.

### 13.2 Model Characterization Metrics

The `StrategyEngine::characterize` scan runs in $O(m + n + \text{nnz})$ time and extracts:

| Metric | Definition |
| :--- | :--- |
| $n$ | Number of structural variables |
| $m$ | Number of constraints |
| $\text{nnz}$ | Nonzeros in $A$ |
| $\rho$ | Density $= \text{nnz} / \max(1, m \cdot n)$ |
| $n_{\text{bin}}$, $n_{\text{int}}$ | Count of binary and integer variables |
| $\alpha_{\text{int}}$ | Integrality ratio $= (n_{\text{bin}} + n_{\text{int}}) / n$ |
| $a_{\min}$, $a_{\max}$ | Min/max absolute nonzero coefficient in $A$ |
| $\kappa_{\text{range}}$ | Dynamic range $= a_{\max} / \max(10^{-15}, a_{\min})$ |

### 13.3 Rule Categories and Specifications

Eighteen deterministic rules R001–R018 are evaluated in strict precedence order. Sub-millisecond overhead is measured across all benchmark instances.

**R001–R006 — Problem Classification and Presolve/Scaling Selection**

| Rule | Condition | Action |
| :--- | :--- | :--- |
| R001 | `model.is_qp() == true` | Route to Active-Set QP solver |
| R002 | `model.is_milp() == true` | Route to Branch-and-Bound MILP |
| R003 | All continuous, $Q = 0$ | Route to Revised Simplex LP |
| R004 | Problem class is LP or MILP | Enable full presolve suite |
| R005 | Problem class is CONVEX_QP | Disable presolve |
| R006 | $\kappa_{\text{range}} \ge 10^4$ | Enable geometric-mean scaling |

**R007–R009 — Numerical Risk Profiling**

| Rule | Condition | Profile | Refactor Interval | Pivot Tol |
| :--- | :--- | :--- | :--- | :--- |
| R007 | $\kappa_{\text{range}} < 10^4$, no near-zero entries | NORMAL | 50 | $10^{-9}$ |
| R008 | $10^4 \le \kappa_{\text{range}} < 10^8$, no near-zero entries | ELEVATED | 20 | $10^{-9}$ |
| R009 | $\kappa_{\text{range}} \ge 10^8$ or any entry $0 < |A_{ij}| < 10^{-12}$ | HIGH | 10 | $10^{-8}$ |

**R010–R013 — Backend Routing**

| Rule | Condition | Action |
| :--- | :--- | :--- |
| R010 | $\text{nnz} < 1000$ and $n < 100$ | `CPU_SEQUENTIAL`, 1 thread |
| R011 | `--no-gpu` or `force_no_gpu` | Disable GPU; force CPU |
| R012 | $\text{nnz} \ge 1000$ or $n \ge 100$; $P > 1$ | `CPU_MULTICORE`, all hardware threads |
| R013 | CUDA device present; $B \ge 128$; $\text{nnz} \ge 10{,}000$ | GPU eligible for batched SpMV |

**R014 — MILP Heuristics**

| Rule | Condition | Action |
| :--- | :--- | :--- |
| R014 | MILP and $\alpha_{\text{int}} \ge 0.50$ | Activate incumbent rounding heuristics at every node |

**R015–R017 — User Override Rules**

| Rule | Condition | Action |
| :--- | :--- | :--- |
| R015 | `override_presolve` set | Force presolve `FULL` or `NONE` |
| R016 | `override_scaling` set | Force scaling `ON` or `OFF` |
| R017 | `override_backend` set | Force specified backend |

**R018 — Safety Precedence**

| Rule | Condition | Action |
| :--- | :--- | :--- |
| R018 | Numerical risk profile is HIGH | Unconditionally clamp refactorization interval $\le 10$, overriding all other parameters |

R018 has the highest precedence: numerical integrity supersedes throughput.

### 13.4 Explainability

Every strategy execution produces:

- `rules_fired`: machine-readable array of triggered rule codes, e.g. `["R003", "R004", "R007", "R010"]`.
- `rule_explanations`: one plain-English string per rule fired.
- `human_readable_rationale`: synthesized executive summary.

### 13.5 JSON Output

The canonical strategy artifact is a structured JSON document containing `problem_class`, `decision` (all selected parameters), `rules_fired`, `rule_explanations`, `human_readable_rationale`, and a `solve_summary`. The strategy JSON is reproducible: identical inputs produce bit-identical JSON output.

---

## 14. Numerical Tolerances

All tolerances are defined in `configs/default_tolerances.json` and enforced by name throughout the codebase. Machine epsilon for IEEE-754 double precision is $\varepsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$.

| Tolerance | Symbol | Default Value | Role |
| :--- | :--- | :--- | :--- |
| Primal feasibility | $\epsilon_{\text{primal}}$ | $10^{-7}$ | Constraint and bound satisfaction thresholds in solvers and verification |
| Dual feasibility | $\epsilon_{\text{dual}}$ | $10^{-7}$ | Reduced-cost and multiplier-sign checks in optimality tests and verification |
| Integrality | $\epsilon_{\text{int}}$ | $10^{-6}$ | Maximum permitted deviation $|x_j - \operatorname{round}(x_j)|$ for integer variables |
| Optimality | $\epsilon_{\text{opt}}$ | $10^{-7}$ | Reduced-cost threshold for Phase II termination in Revised Simplex |
| Pivot | $\tau_{\text{pivot}}$ | $10^{-9}$ | Minimum acceptable pivot magnitude in SparseLU (NORMAL/ELEVATED risk); $10^{-8}$ under HIGH risk |
| Residual | $\epsilon_{\text{residual}}$ | $10^{-6}$ ($10^{-8}$ LU internal) | Iterative refinement convergence criterion $\|r_k\|_\infty$ |
| KKT stationarity | $\epsilon_{\text{stat}}$ | $10^{-6}$ | QP gradient stationarity residual $\|Qx + c + A^T\lambda - \mu\|_\infty$ |
| Complementary slackness | $\epsilon_{\text{comp}}$ | $10^{-7}$ | $|\text{slack}_i \cdot y_i|$ and $|\mu_j (x_j - \text{bound})|$ threshold |
| LU fill-in drop | — | $10^{-15}$ | Entries below this magnitude are dropped during elimination |
| LICQ orthogonality | — | $10^{-6}$ (relative) | Constraint rejection threshold in Modified Gram-Schmidt filter |
| PSD threshold | $\epsilon_{\text{PSD}}$ | $10^{-9}$ | Negative-eigenvalue detection in pivoted Cholesky |
| GPU numerical match | $\epsilon_{\text{GPU}}$ | $10^{-12}$ (abs), $10^{-10}$ (rel) | Maximum permitted deviation between GPU and CPU SpMV results |

> [!NOTE]
> All tolerances are substantially larger than machine epsilon $\varepsilon_{\text{mach}}$, providing adequate margin for accumulated floating-point round-off while remaining small relative to typical coefficient magnitudes in industrial optimization models.

---

## 15. Mathematical Notation Summary

| Symbol | Type | Definition |
| :--- | :--- | :--- |
| $n$ | $\mathbb{N}$ | Number of decision variables |
| $m$ | $\mathbb{N}$ | Number of constraints |
| $\text{nnz}$ | $\mathbb{N}$ | Number of nonzero entries in constraint matrix $A$ |
| $x$ | $\mathbb{R}^n$ | Primal decision variable vector |
| $c$ | $\mathbb{R}^n$ | Linear objective coefficient vector |
| $c_0$ | $\mathbb{R}$ | Constant objective offset |
| $Q$ | $\mathbb{R}^{n \times n}$ | Symmetric Hessian matrix of QP objective; $Q \succeq 0$ required |
| $A$ | $\mathbb{R}^{m \times n}$ | Constraint coefficient matrix |
| $b$ | $\mathbb{R}^m$ | Constraint right-hand side vector |
| $l, u$ | $\mathbb{R}^n$ | Variable lower and upper bound vectors |
| $I$ | $\subseteq \{1,\dots,n\}$ | Index set of integer-restricted variables (MILP) |
| $y$ | $\mathbb{R}^m$ | Dual multiplier (constraint shadow price) vector |
| $\bar{c}_j$ | $\mathbb{R}$ | Reduced cost of variable $j$: $c_j - y^T a_j$ |
| $\lambda$ | $\mathbb{R}^{|\mathcal{W}|}$ | Lagrange multipliers of active set constraints (QP) |
| $\mu$ | $\mathbb{R}^n$ | Bound multiplier vector (QP) |
| $B$ | $\mathbb{R}^{m \times m}$ | Simplex basis matrix |
| $\mathcal{B}$, $\mathcal{N}$ | Index sets | Basic and nonbasic variable index sets |
| $P$ | $\mathbb{R}^{n \times n}$ | Permutation matrix (LU factorization) |
| $L$ | $\mathbb{R}^{n \times n}$ | Unit lower triangular matrix (LU) |
| $U$ | $\mathbb{R}^{n \times n}$ | Upper triangular matrix (LU) |
| $D_r, D_c$ | Diagonal $\mathbb{R}^{m \times m}, \mathbb{R}^{n \times n}$ | Row and column scaling matrices |
| $\tilde{A}$ | $\mathbb{R}^{m \times n}$ | Scaled constraint matrix $= D_r A D_c$ |
| $\rho$ | $[0, 1]$ | Matrix density $= \text{nnz} / (m \cdot n)$ |
| $\kappa_{\text{range}}$ | $\mathbb{R}_{>0}$ | Coefficient dynamic range ratio $= a_{\max} / a_{\min}$ |
| $\alpha_{\text{int}}$ | $[0, 1]$ | Integrality ratio $= (n_{\text{bin}} + n_{\text{int}}) / n$ |
| $\mathcal{W}_k$ | Index set | Active working set of constraints at QP iteration $k$ |
| $p_k$ | $\mathbb{R}^n$ | Active-set step direction at iteration $k$ |
| $g_k$ | $\mathbb{R}^n$ | QP gradient at iterate $k$: $Qx_k + c$ |
| $\theta^*$ | $\mathbb{R}_{\ge 0}$ | Simplex step length (minimum ratio) |
| $\alpha_k$ | $[0, 1]$ | Active-set step length (blocking constraint ratio) |
| $z_{\text{inc}}$ | $\mathbb{R}$ | Current incumbent objective in Branch-and-Bound |
| $z_{\text{bound}}$ | $\mathbb{R}$ | Current global dual bound in Branch-and-Bound |
| $s_j$ | $[0, 0.5]$ | Fractionality score of integer variable $j$ |
| $B$ | $\mathbb{N}$ | SpMV batch size |
| $T_{\text{H2D}}, T_{\text{D2H}}$ | $\mathbb{R}_{\ge 0}$ | Host-to-device and device-to-host transfer times |
| $T_{\text{kernel}}$ | $\mathbb{R}_{\ge 0}$ | CUDA kernel execution time |
| $T_{\text{GPU}}$ | $\mathbb{R}_{\ge 0}$ | Transfer-inclusive GPU total time $= T_{\text{H2D}} + T_{\text{kernel}} + T_{\text{D2H}}$ |
| $\varepsilon_{\text{mach}}$ | $\approx 2.22 \times 10^{-16}$ | IEEE-754 double-precision machine epsilon |
| $\epsilon_{\text{primal}}$ | $10^{-7}$ | Primal feasibility tolerance |
| $\epsilon_{\text{dual}}$ | $10^{-7}$ | Dual feasibility tolerance |
| $\epsilon_{\text{int}}$ | $10^{-6}$ | Integrality tolerance |
| $\epsilon_{\text{opt}}$ | $10^{-7}$ | Optimality tolerance (reduced-cost threshold) |
| $\tau_{\text{pivot}}$ | $10^{-9}$ | LU pivot magnitude threshold (NORMAL/ELEVATED risk) |
| $\epsilon_{\text{residual}}$ | $10^{-8}$ | Iterative refinement convergence tolerance |
| $\epsilon_{\text{stat}}$ | $10^{-6}$ | KKT stationarity residual tolerance (QP) |
| $\epsilon_{\text{comp}}$ | $10^{-7}$ | Complementary slackness tolerance |
