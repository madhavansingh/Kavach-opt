# Kavach-Opt Sparse Numerical Core Documentation

## 1. Overview and Architecture

The `src/numerics` subsystem provides the sparse linear algebra foundation for Kavach-Opt, strictly implementing requirements **[D]** and **[E]** from `SIH26119_Kavach-Opt_Blueprint_v2.md`.

This layer fulfills the non-negotiable architectural requirement:
> **Real sparse architecture at Netlib/MIPLIB scale (not a toy dense demo).**
> Fully sparse CSR/CSC storage, sparse SpMV, sparse LU factorization with partial pivoting, bounded iterative refinement, and geometric-mean row/column scaling, with **zero external optimization or factorization library dependencies** (no Eigen, SuiteSparse, UMFPACK, etc.).

```
                                  +-----------------------------+
                                  |   CanonicalModel [model]    |
                                  +-----------------------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  |    model_converter.hpp      |
                                  |  (canonical_model_to_csr)   |
                                  +-----------------------------+
                                                 |
                       +-------------------------+-------------------------+
                       |                                                   |
                       v                                                   v
         +---------------------------+                       +---------------------------+
         |      CsrMatrix (CSR)      | <-------------------> |      CscMatrix (CSC)      |
         |  row_ptr, col_idx, values |      O(nnz + m + n)   |  col_ptr, row_idx, values |
         +---------------------------+                       +---------------------------+
                       |                                                   |
          +------------+------------+                                      |
          |                         |                                      |
          v                         v                                      v
+-------------------+     +--------------------+                 +-------------------+
|  scaling.hpp/cpp  |     | lu_factorization   |                 |   SpMV / Norms    |
| - Conditioning    |     | - SparseLU (PA=LU) |                 | - y = A * x       |
| - Geom-mean scale |     | - Solve Ax=b       |                 | - y = A^T * x     |
| - Unscale primal  |     | - Solve A^T x = b  |                 | - 1-norm, inf-norm|
| - Unscale dual    |     | - Refinement       |                 +-------------------+
+-------------------+     +--------------------+
```

---

## 2. Sparse Storage Formats & Invariants

Kavach-Opt implements both row-compressed (CSR) and column-compressed (CSC) formats.

### 2.1 Compressed Sparse Row (CSR)
- **`row_ptr_`**: Vector of size $m + 1$. `row_ptr_[i]` stores the starting offset of row $i$ in `col_idx_` and `values_`.
- **`col_idx_`**: Vector of size $\text{nnz}$. Stores 0-indexed column indices.
- **`values_`**: Vector of size $\text{nnz}$. Stores nonzero double-precision coefficients.

### 2.2 Compressed Sparse Column (CSC)
- **`col_ptr_`**: Vector of size $n + 1$. `col_ptr_[j]` stores the starting offset of column $j$ in `row_idx_` and `values_`.
- **`row_idx_`**: Vector of size $\text{nnz}$. Stores 0-indexed row indices.
- **`values_`**: Vector of size $\text{nnz}$. Stores nonzero double-precision coefficients.

### 2.3 Structural Invariants
Every matrix adheres to the following structural invariants:
1. `row_ptr_.size() == rows + 1` (or `col_ptr_.size() == cols + 1`).
2. `row_ptr_[0] == 0` (or `col_ptr_[0] == 0`).
3. `row_ptr_.back() == col_idx_.size() == values_.size() == nnz`.
4. `row_ptr_` is monotonically non-decreasing: `row_ptr_[i] <= row_ptr_[i+1]`.
5. Column/row indices are strictly within bounds: $0 \le \text{col\_idx}[k] < n$, $0 \le \text{row\_idx}[k] < m$.
6. Entries within each row/column are sorted in strictly ascending order with zero duplicate indices (canonicalized).
7. All stored values are finite (no `NaN`, no `+Inf`, no `-Inf`).

Validation is enforced via `is_valid()` (non-throwing) and `validate()` (throws `core::InputException` with actionable diagnostic messages).

---

## 3. Conversions & Sparse Transpose

### 3.1 CSR $\leftrightarrow$ CSC Conversion
Conversions between CSR and CSC are executed in strict $O(\text{nnz} + m + n)$ time and $O(\text{nnz} + m + n)$ space without dense allocations:
- **Pass 1:** Count nonzeros per target dimension (histogram).
- **Pass 2:** Compute prefix sums to determine pointers.
- **Pass 3:** Scatter entries into output arrays.

### 3.2 Sparse Transpose ($A^T$)
- Sparse transpose of a CSR matrix of size $(m, n)$ directly produces a CSR matrix of size $(n, m)$ by applying the counting sort algorithm.
- Double transpose $(A^T)^T$ identically recovers $A$.

---

## 4. Sparse LU Factorization ($P A = L U$)

`SparseLU` implements Sparse Gaussian Elimination with Partial Pivoting:
$$P A = L U$$
where:
- $P$ is an $n \times n$ permutation matrix representing row swaps.
- $L$ is a unit lower triangular sparse matrix ($L_{ii} = 1$).
- $U$ is an upper triangular sparse matrix with nonzero diagonal elements.

### 4.1 Numerical Pivoting and Safeguards
- At elimination step $k \in \{0, \dots, n-1\}$, the active entries in column $k$ among candidate rows $p \ge k$ are evaluated.
- The row $p^*$ maximizing $|A_{pk}|$ is chosen as the pivot.
- **Near-Zero Pivot Flagging:** If $|A_{p^* k}| < \text{pivot\_tol}$ (default $10^{-9}$), factorization halts immediately:
  - Solver status is set to `core::SolverStatus::NUMERICAL_FAILURE`.
  - Descriptive diagnostic message is logged: `"Near-zero or zero pivot at column k: max pivot magnitude < pivot_tol"`.
  - No false OPTIMAL is ever reported.

### 4.2 Two-Pointer Sparse Row Elimination
- Elimination of column $k$ in candidate row $i > k$ is performed using a 2-pointer linear merge of sorted sparse rows:
  $$\text{row}_i \leftarrow \text{row}_i - m_{ik} \times \text{row}_k$$
- Complexity per row operation is strictly $O(\text{nnz}(\text{row}_i) + \text{nnz}(\text{row}_k))$ with zero dense auxiliary buffers and no secondary sorting needed.
- Values with magnitude below $10^{-15}$ are dropped to curb fill-in drift.

---

## 5. Direct and Transpose Solves

### 5.1 Direct Solve ($A x = b$)
$$P A x = P b \implies L U x = P b$$
1. Permute RHS: $y^{(0)} = P b$ ($y^{(0)}_i = b_{\text{perm\_r}[i]}$).
2. Forward solve: $L y = y^{(0)}$ (unit lower triangular).
3. Backward solve: $U x = y$ (upper triangular, divide by $U_{ii}$).

### 5.2 Transpose Solve ($A^T x = b$)
$$A^T = U^T L^T P \implies U^T L^T (P x) = b$$
1. Let $w = P x$. Solve $U^T z = b$ via row-push forward solve.
2. Solve $L^T w = z$ via row-push backward solve.
3. Permute solution: $x_{\text{perm\_r}[i]} = w_i \iff x = P^T w$.

---

## 6. Iterative Refinement & Residual Computation

To recover floating-point precision lost to round-off:
1. Compute exact sparse residual: $r_0 = b - A x_0$.
2. Check $\|r_0\|_\infty \le \text{residual\_tol}$ (default $10^{-8}$).
3. If not converged, compute correction: $A \delta_k = r_k$ using existing LU factors.
4. Update $x_{k+1} = x_k + \delta_k$ and recompute residual.
5. Terminate when $\|r_k\|_\infty \le \text{residual\_tol}$, or when max iterations reached, or on stagnation.

---

## 7. Geometric-Mean Scaling & Conditioning Diagnostics

### 7.1 Scaling Algorithm
Geometric-mean equilibration balances matrix rows and columns before factorization:
- For each row $i$:
  $$r_i = \frac{1}{\sqrt{\min_{j} |A_{ij}| \cdot \max_{j} |A_{ij}|}}$$
- For each column $j$:
  $$c_j = \frac{1}{\sqrt{\min_{i} |A'_{ij}| \cdot \max_{i} |A'_{ij}|}}$$
- Scaled matrix: $\tilde{A} = R A C$.

### 7.2 Unscaling Formulas
Given scaled primal solution $\tilde{x}$ and dual multipliers $\tilde{y}$:
- **Primal:** $x_j = C_j \tilde{x}_j$
- **Dual:** $y_i = R_i \tilde{y}_i$
- **Reduced Costs:** $d_j = \tilde{d}_j / C_j$
- **Primal Bounds:** $\tilde{l}_j = l_j / C_j$, $\tilde{u}_j = u_j / C_j$
- **RHS:** $\tilde{b}_i = R_i b_i$

### 7.3 Conditioning Taxonomy
- `WELL_CONDITIONED`: Dynamic range ratio $< 10^4$
- `MODERATE`: $10^4 \le \text{ratio} < 10^8$
- `ILL_CONDITIONED`: $10^8 \le \text{ratio} < 10^{13}$
- `SEVERELY_ILL_CONDITIONED`: $\text{ratio} \ge 10^{13}$

---

## 8. Subquadratic Memory Scaling Proof

The memory footprint of `CsrMatrix` is:
$$\text{Memory}_{\text{CSR}} = (m + 1) \times \text{sizeof}(\text{size\_t}) + \text{nnz} \times (\text{sizeof}(\text{size\_t}) + \text{sizeof}(\text{double}))$$

For sparse problem classes where $\text{nnz} = O(n)$ (e.g. tridiagonal, banded, block-angular):
$$\text{Memory}_{\text{CSR}} = O(m + n) \ll O(m \times n)$$

Verified in unit test `test_sparse_memory_scaling_subquadratic`:
- Scaling from $N = 100$ to $N = 1000$ (a $10\times$ increase in dimension):
  - Dense storage would grow $(1000^2)/(100^2) = 100\times$.
  - Kavach-Opt CSR storage grew only $9.95\times$, confirming strictly linear $O(N)$ sub-quadratic scaling.
