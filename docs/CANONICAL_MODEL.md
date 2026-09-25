# Kavach-Opt — Canonical Model Specification [A]

**Status:** Phase 1 Complete  
**Blueprint Reference:** Section 4.2 [A], Section 5, 6, 7, 11  

---

## 1. Overview & Architectural Role

The **Canonical Model Representation** (`kavach::model::CanonicalModel`) is the single immutable in-memory contract consumed by all downstream solver subsystems:
- **Presolve [C]** (reads canonical model, generates reduced canonical model + reconstruction map)
- **LP Solver [F]** (extracts sparse constraint matrix, bounds, objective for Revised Simplex)
- **MILP Solver [G]** (accesses integrality masks and variable bounds for Branch-and-Bound tree search)
- **QP Solver [H]** (reads symmetric sparse Hessian $Q$ and linear vector $c$)
- **Independent Verification Layer [Y]** (recomputes residuals against original un-presolved canonical model)
- **Solver Strategy Layer [T]** (analyzes sparsity, variable counts, and coefficient ranges to choose configurations)

The model is strictly **parser-independent** and **solver-independent**:
- Parsers (Phase 2) populate the model via public builder methods without knowledge of solver algorithms.
- Solvers (Phase 5+) read the model through read-only accessors without knowledge of file origins or parser states.

---

## 2. Variable Representation & Bound Semantics

Variables are identified by a dense, 0-indexed `VarId` (`uint32_t`).

### Variable Types
1. **`CONTINUOUS`:** Standard real-valued decision variable ($x \in [\text{lb}, \text{ub}]$).
2. **`INTEGER`:** Integer-restricted decision variable ($x \in \mathbb{Z} \cap [\text{lb}, \text{ub}]$).
3. **`BINARY`:** 0-1 decision variable ($x \in \{0, 1\}$).
   - Normalized bounds: $[0.0, 1.0]$.
   - Validation enforces $0 \le \text{lb} \le \text{ub} \le 1$.

### Bound Representation
- Finite bounds: Represented as standard IEEE-754 `double`.
- Unbounded sides: Represented safely using `INFINITY_BOUND = std::numeric_limits<double>::infinity()`.
- Unbounded above: `upper_bound = +INFINITY_BOUND`.
- Unbounded below: `lower_bound = -INFINITY_BOUND`.
- Free variable: `lower_bound = -INFINITY_BOUND` and `upper_bound = +INFINITY_BOUND`.
- Lower-only bounded (standard non-negativity): `lower_bound = 0.0`, `upper_bound = +INFINITY_BOUND`.

---

## 3. Sparse Linear Representation

Constraints and linear objective terms use `SparseLinearExpression`.

### Representation Choice
Instead of storing a dense $M \times N$ matrix, each constraint stores a collection of `LinearTerm { VarId, double coefficient }`.

**Rationale:**
1. **Memory efficiency:** Typical industrial refinery and supply-chain matrices have density $< 1\%$. Sparse storage scales as $O(\text{nnz})$ rather than $O(M \cdot N)$.
2. **Cache locality:** Contiguous term arrays minimize cache misses during linear combination evaluation.
3. **Seamless conversion to CSR/CSC:** Canonical sorted terms map directly into CSR row pointers/column indices and CSC column structures during Phase 3.

### Normalization Rules for Linear Terms
When a linear expression is canonicalized:
1. **Ordering:** Terms are sorted strictly ascending by `VarId`.
2. **Duplicate Merging:** If a variable appears multiple times (e.g., $2x_1 + 5x_2 - x_1$), coefficients are summed algebraically ($x_1: 2 - 1 = 1$).
3. **Zero Elimination:** Any term whose net coefficient is exactly $0.0$ is removed.

---

## 4. Quadratic Objective Terms

Convex QP requires minimizing/maximizing:
$$\frac{1}{2} x^T Q x + c^T x + \text{offset}$$

Quadratic terms are represented via `SparseQuadraticExpression` containing `QuadraticTerm { var1_id, var2_id, coefficient }`.

### Normalization Rules for Quadratic Terms
1. **Symmetric Canonicalization:** The pair $(i, j)$ is always stored with $i \le j$. If a term is added with $j > i$, indices are swapped.
2. **Lexicographical Sort:** Terms are sorted by $(i, j)$ ascending.
3. **Duplicate Merging:** Multiple occurrences of $x_i x_j$ are merged by coefficient addition.
4. **Zero Elimination:** Terms summing to $0.0$ are pruned.

---

## 5. Constraint Representation

Each constraint represents:
$$\sum_{j} a_{ij} x_j \quad (\text{sense}) \quad b_i$$

- **Sense:** `LESS_EQUAL` ($\le$), `EQUAL` ($=$), `GREATER_EQUAL` ($\ge$).
- **RHS:** Must be finite (no NaN, no $\pm\infty$).
- **Nonzero count:** $O(\text{nnz}_i)$.

---

## 6. Identifier Policy & Fast Name Lookup

- Variables and constraints optionally receive human-readable string names.
- **Uniqueness Policy:** Non-empty names must be unique within variables and within constraints. Attempting to add a duplicate non-empty name immediately raises a typed `kavach::core::InputException`.
- **Fast Lookup:** `CanonicalModel::find_variable_id(name)` and `find_constraint_id(name)` provide $O(1)$ amortized lookups via hash maps.
- Internal stable index access via `VarId` ($0 \dots N-1$) and `ConstrId` ($0 \dots M-1$) is always supported.

---

## 7. Model Validation & Strong Invariants

The validation engine (`validate()` and `check_valid()`) enforces the following invariants:

| Component | Invariant Enforced |
|---|---|
| **Variables** | No NaN in bounds; $\text{lb} \le \text{ub}$; $\text{lb} \neq +\infty$; $\text{ub} \neq -\infty$. |
| **Binary Variables** | Bounds must be finite and satisfy $0 \le \text{lb} \le \text{ub} \le 1$. |
| **Constraints** | RHS must be finite; all referenced `VarId` $< N$; all coefficients finite (no NaN, no Inf). |
| **Objective** | Offset must be finite; linear/quadratic variable indices $< N$; all coefficients finite. |
| **Names** | Non-empty names are unique. |

`CanonicalModel::check_valid()` runs full validation and raises a descriptive `kavach::core::InputException` detailing all violations.
