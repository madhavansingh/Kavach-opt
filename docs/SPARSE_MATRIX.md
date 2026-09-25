# Kavach-Opt Sparse Matrix Subsystem Specification & Guide

## 1. Executive Summary

The `src/numerics` sparse matrix subsystem in Kavach-Opt provides high-performance, deterministic, memory-efficient sparse linear algebra foundation. It fulfills the core architectural mandate of Kavach-Opt:
- **Zero Third-Party Dependencies:** Native implementation of Compressed Sparse Row (CSR) and Compressed Sparse Column (CSC) formats with zero reliance on Eigen, SuiteSparse, UMFPACK, or BLAS.
- **Scale-Oriented Subquadratic Complexity:** $O(\text{nnz} + m + n)$ conversions, $O(\text{nnz})$ SpMV, and strict $O(N)$ sparse storage for sparse LP/MILP/QP constraint matrices.
- **Numerical Robustness:** Deterministic duplicate aggregation, zero elimination, finite-value validation rejecting `NaN`/`Inf`, and geometric-mean conditioning scaling.

---

## 2. Core Data Structures: CSR and CSC

Both `CsrMatrix` and `CscMatrix` are 0-indexed, contiguous-array compressed representations.

### 2.1 Compressed Sparse Row (`CsrMatrix`)
Suited for row-wise operations, constraint analysis, and standard matrix-vector multiplication $y = A x$:
- **`rows_`** ($m$): Number of rows.
- **`cols_`** ($n$): Number of columns.
- **`row_ptr_`** ($m + 1$): Pointer offsets. Row $i$ spans indices `[row_ptr_[i], row_ptr_[i+1])`.
- **`col_idx_`** ($\text{nnz}$): Column index for each non-zero entry ($0 \le \text{col\_idx}[k] < n$).
- **`values_`** ($\text{nnz}$): Double-precision coefficient value for each non-zero entry.

### 2.2 Compressed Sparse Column (`CscMatrix`)
Suited for column-wise operations, simplex basis column extraction, and transposed matrix-vector multiplication $y = A^T x$:
- **`rows_`** ($m$): Number of rows.
- **`cols_`** ($n$): Number of columns.
- **`col_ptr_`** ($n + 1$): Pointer offsets. Column $j$ spans indices `[col_ptr_[j], col_ptr_[j+1])`.
- **`row_idx_`** ($\text{nnz}$): Row index for each non-zero entry ($0 \le \text{row\_idx}[k] < m$).
- **`values_`** ($\text{nnz}$): Double-precision coefficient value for each non-zero entry.

### 2.3 Direct Slice Accessors
For zero-overhead inspection without copying:
- `csr.row_indices(r)`: Pointer to the first column index in row $r$.
- `csr.row_values(r)`: Pointer to the first value in row $r$.
- `csr.row_start(r)` / `csr.row_end(r)`: Offset boundaries for row $r$.
- `csr.row_nnz(r)`: Number of non-zero entries in row $r$.
- `csc.col_indices(c)`: Pointer to the first row index in column $c$.
- `csc.col_values(c)`: Pointer to the first value in column $c$.
- `csc.col_start(c)` / `csc.col_end(c)`: Offset boundaries for column $c$.
- `csc.col_nnz(c)`: Number of non-zero entries in column $c$.

---

## 3. Structural and Numerical Invariants

Every valid sparse matrix satisfies the following invariant guarantees:
1. **Dimension Invariant:** `row_ptr_.size() == rows_ + 1` and `col_ptr_.size() == cols_ + 1`.
2. **Start Pointer Invariant:** `row_ptr_[0] == 0` and `col_ptr_[0] == 0`.
3. **Monotonicity:** `row_ptr_[i] <= row_ptr_[i+1]` and `col_ptr_[j] <= col_ptr_[j+1]`.
4. **Size Consistency:** `row_ptr_.back() == col_idx_.size() == values_.size() == nonzeros_`.
5. **Coordinate Bounds:** Every column index in CSR is $< n$, every row index in CSC is $< m$.
6. **Canonical Ordering:** Non-zeros within each row (CSR) or column (CSC) are sorted in strictly ascending order without duplicate coordinates.
7. **Finite Value Guarantee:** Stored coefficients are strictly finite; any `NaN`, `+Inf`, or `-Inf` in triplets or arrays is rejected.

---

## 4. Construction & Canonicalization

Sparse matrices can be constructed via:
1. **Raw Arrays:** `CsrMatrix(m, n, row_ptr, col_idx, values)` (optional auto-canonicalization).
2. **Coordinate Triplets (`Triplet{row, col, value}`):**
   - Step 1: Pre-validate coordinates ($0 \le \text{row} < m$, $0 \le \text{col} < n$) and finite values. Out-of-range throws `core::InputException`, non-finite throws `core::NumericalException`.
   - Step 2: Filter exact zeros (`value != 0.0`).
   - Step 3: Prefix sums for row/column histogram counting.
   - Step 4: Deterministic bucket scatter into target format.
   - Step 5: Canonicalize: Sort entries within row/column. If `sum_duplicates = true`, sum duplicate coordinates deterministically. If duplicates cancel to exact 0.0 (or below drop tolerance), eliminate the coefficient.
3. **From `CanonicalModel`:** `canonical_model_to_csr(model)` and `canonical_model_to_csc(model)` in $O(m + n + \text{nnz})$.

---

## 5. Conversions & Sparse Transpose

### 5.1 CSR $\leftrightarrow$ CSC
- Executed via counting-sort scatter in strict $O(\text{nnz} + m + n)$ time and $O(\text{nnz} + m + n)$ memory.
- Zero dense allocations ($m \times n$) are ever performed.
- Non-zeros and exact coordinate values are strictly preserved:
  $$\text{csc\_to\_csr}(\text{csr\_to\_csc}(A)) \equiv A$$

### 5.2 Sparse Transpose
- `transpose(csr)` produces an $n \times m$ `CsrMatrix` in $O(\text{nnz} + m + n)$.
- Double transpose satisfies identity: $(A^T)^T \equiv A$.

---

## 6. Matrix-Vector Multiplication (SpMV)

The sparse core provides high-efficiency SpMV routines:
1. **$y = A x$:**
   - Vector overload: `auto y = mat.spmv(x)`
   - Raw pointer overload: `mat.spmv(x.data(), y.data())`
   - Accumulator overload: `mat.spmv(alpha, x, beta, y)` computing $y \leftarrow \alpha A x + \beta y$
2. **$y = A^T x$ (Transpose SpMV):**
   - Vector overload: `auto y = mat.spmv_transpose(x)`
   - Raw pointer overload: `mat.spmv_transpose(x.data(), y.data())`
   - Accumulator overload: `mat.spmv_transpose(alpha, x, beta, y)` computing $y \leftarrow \alpha A^T x + \beta y$

Both `CsrMatrix` and `CscMatrix` implement both primal and transpose SpMV natively.

---

## 7. Scaling Subsystem Integration

Located in `src/numerics/scaling.hpp`:
- **`analyze_conditioning(A)`:** Calculates dynamic range ratio ($\max |A_{ij}| / \min |A_{ij}|$), 1-norm, $\infty$-norm, and determines conditioning classification (`WELL_CONDITIONED`, `MODERATE`, `ILL_CONDITIONED`, `SEVERELY_ILL_CONDITIONED`).
- **`compute_geometric_mean_scaling(A, max_passes)`:** Computes symmetric row scaling factors $r_i$ and column scaling factors $c_j$ using geometric mean equilibration.
- **`apply_scaling(A, factors)`:** Scales matrix coefficients $A'_{ij} = r_i \cdot A_{ij} \cdot c_j$.
- **Vector Transformations:**
  - Scaled RHS: $b'_i = r_i \cdot b_i$
  - Scaled bounds: $l'_j = l_j / c_j$, $u'_j = u_j / c_j$
  - Primal unscaling: $x_j = x'_j \cdot c_j$
  - Dual unscaling: $\pi_i = \pi'_i \cdot r_i$

---

## 8. Memory & Complexity Characteristics

| Operation | Time Complexity | Memory Complexity | Dense Memory ($5000 \times 5000$) | Sparse Memory ($5000 \times 5000$, $0.1\%$) |
|---|---|---|---|---|
| Triplet Build | $O(\text{triplets} \log d)$ | $O(\text{nnz} + m)$ | N/A | $\approx 429\text{ KB}$ |
| CSR $\to$ CSC | $O(\text{nnz} + m + n)$ | $O(\text{nnz} + n)$ | N/A | $\approx 429\text{ KB}$ |
| SpMV | $O(\text{nnz})$ | $O(1)$ workspace | $200\text{ MB}$ ($5000^2 \times 8\text{B}$) | $429\text{ KB}$ ($0.22\%$ of dense) |
| Scaling | $O(k \cdot \text{nnz})$ | $O(m + n)$ | N/A | $O(m + n)$ factors |

Sparse representation scales strictly $O(N)$ with problem dimension for constant average row degree, ensuring Netlib and MIPLIB scale instances run within minimal memory footprint.
