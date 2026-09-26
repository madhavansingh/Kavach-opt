#!/usr/bin/env python3
import sys

# Read the base MATHEMATICS.md
with open("docs/MATHEMATICS.md", "r") as f:
    doc = f.read()

expansions_sec4 = """
### 4.5 Fill-In Bounds & Markowitz Graph Elimination Model
From a graph-theoretic viewpoint, the sparsity pattern of the symmetric completion $B + B^T$ corresponds to an undirected graph $G = (V, E)$, where $V = \\{1, \\dots, m\\}$ represents equations and $(i, j) \\in E \\iff a_{ij} \\ne 0$. Gaussian elimination of pivot $(i, j)$ corresponds to forming the elimination graph $G_v$: node $v$ is removed, and a clique is formed among all neighbors $\\text{Adj}(v)$ that were not previously connected.

The number of new edges added (fill-in entries) is precisely bounded by:

$$
\\text{fill}(v) \\le \\frac{(d(v) - 1)(d(v) - 2)}{2}
$$

For unsymmetric basis matrices $B$, the directed bipartite graph model yields the Markowitz count as the exact upper bound on fill-in created in the uneliminated submatrix:

$$
M_{ij} = (r_i - 1)(c_j - 1)
$$

### 4.6 Algorithmic Markowitz Search with Linked Lists
To achieve $O(1)$ updates and avoid $O(m^2)$ scanning of the active submatrix, `SparseLU` organizes active matrix entries in doubly-linked row and column lists:
1. Candidate rows and columns are searched in order of increasing nonzero count: $k = 1, 2, \\dots$.
2. For rows with count $r_i = 1$, the single entry is an immediate pivot ($M_{ij} = 0$), creating zero fill-in.
3. For columns with count $c_j = 1$, the single entry is likewise a zero fill-in pivot ($M_{ij} = 0$).
4. For general entries, the search traverses rows with minimum count $r_{\\min}$, evaluating threshold condition $|a_{ij}| \\ge \\mu \\max_k |a_{kj}|$. The search terminates as soon as an entry with $M_{ij} \\le (r_{\\min} - 1)^2$ is identified, guaranteeing minimal fill-in without exhaustive search.
"""

expansions_sec7 = """
### 7.5 Bland's Anti-Cycling Rule: Mathematical Proof Sketch
Degeneracy arises when a basic variable has value zero ($x_{B_i} = 0$). In this regime, the minimum ratio test yields $\\theta^* = 0$, causing the basis to change while the primal iterate $x$ and objective $z = c^T x$ remain strictly unchanged. Under standard Dantzig pricing, degeneracy can induce cyclic sequences of bases $B_0 \\to B_1 \\to \\dots \\to B_k = B_0$, resulting in an infinite non-terminating loop.

Kavach-Opt incorporates Bland's smallest-subscript rule (1977):
1. **Entering Variable:** Select the nonbasic variable with negative reduced cost having the smallest index:
$$
q = \\min \\{ j \\in \\mathcal{N} \\mid \\bar{c}_j < -\\epsilon_{\\text{opt}} \\}
$$
2. **Leaving Variable:** Among all basic variables achieving the minimum ratio $\\theta^*$, select the one with the smallest index:
$$
p = \\min \\left\\{ B_i \\;\\middle|\\; i \\in \\mathcal{B}, \\; d_i > 0, \\; \\frac{x_{B_i}}{d_i} = \\theta^* \\right\\}
$$

**Theorem (Bland):** *Under the smallest-subscript rule, no basis can be repeated, guaranteeing finite termination of the Simplex algorithm.*

*Proof Sketch:* Suppose a cycle of degenerate bases occurs. Let $L$ be the set of variables that enter or leave the basis during the cycle, and let $t = \\max(L)$ be the variable with the highest index in $L$. Since $t$ must enter the basis in some pivot and leave in another, we consider the two corresponding tableaus. By orthogonality of the row spaces of the two canonical tableaus and the sign consistency of the ratio tests, a contradiction arises where $\\bar{c}_t < 0$ and $\\bar{c}_t > 0$ simultaneously. Thus, no cycle can exist.
"""

expansions_sec10 = """
### 10.4 Chvátal-Gomory Rank and Finite Generation
Let $P = \\{ x \\in \\mathbb{R}^n \\mid A x \\le b \\}$ be a rational polyhedron. The first Chvátal closure $P^{(1)}$ is defined as the intersection of all valid Chvátal-Gomory inequalities for $P$:

$$
P^{(1)} = \\bigcap_{u \\in \\mathbb{R}_+^m, u A \\in \\mathbb{Z}^n} \\left\\{ x \\in \\mathbb{R}^n \\;\\middle|\\; (u A) x \\le \\lfloor u b \\rfloor \\right\\}
$$

By iterating this operation, the $k$-th Chvátal closure is defined recursively as $P^{(k)} = (P^{(k-1)})^{(1)}$.

**Theorem (Chvátal 1973, Schrijver 1980):** *For any rational polyhedron $P$, there exists a finite integer $k^* \\in \\mathbb{N}$ such that:*
$$
P^{(k^*)} = P_I = \\text{conv}(P \\cap \\mathbb{Z}^n)
$$
*The minimum such integer $k^*$ is called the Chvátal rank of $P$.*

In Kavach-Opt, Gomory mixed-integer (GMI) cuts are generated directly from the optimal simplex tableau rows at the root node. Because each tableau row represents an aggregated surrogate constraint $x_B + B^{-1} A_N x_N = B^{-1} b$, GMI cuts achieve rank-1 separation with zero matrix inversion overhead.

### 10.5 Subadditivity of the Mixed-Integer Rounding (MIR) Function
The core mathematical property enabling MIR cut validity is the subadditivity of the function $F_\\beta: \\mathbb{R} \\to \\mathbb{R}$:

$$
F_\\beta(\\alpha) = \\lfloor \\alpha \\rfloor + \\frac{\\max(0, \\alpha - \\lfloor \\alpha \\rfloor - (1 - \\beta))}{\\beta}
$$

For any $\\alpha_1, \\alpha_2 \\in \\mathbb{R}$:
$$
F_\\beta(\\alpha_1 + \\alpha_2) \\le F_\\beta(\\alpha_1) + F_\\beta(\\alpha_2)
$$
Furthermore, $F_\\beta$ satisfies $F_\\beta(k) = k$ for all $k \\in \\mathbb{Z}$. Consequently, applying $F_\\beta$ to both sides of a mixed-integer constraint preserves validity for all integer vectors $x \\in \\mathbb{Z}^n$ while strictly deepening the cut against fractional continuous relaxations.
"""

expansions_sec11 = """
### 11.6 Range-Space vs Null-Space KKT Factorization
At each active-set step, Kavach-Opt solves the augmented KKT saddle-point system:

$$
\\begin{bmatrix} Q & A_{\\mathcal{W}}^T \\\\ A_{\\mathcal{W}} & 0 \\end{bmatrix} \\begin{bmatrix} p \\\\ -\\lambda \\end{bmatrix} = - \\begin{bmatrix} g_k \\\\ 0 \\end{bmatrix}
$$

When the number of active constraints $|\\mathcal{W}|$ is small relative to $n$, the system is solved via **Null-Space Projection**:
1. Compute basis $Z \\in \\mathbb{R}^{n \\times (n - |\\mathcal{W}|)}$ for the null space of $A_{\\mathcal{W}}$ such that $A_{\\mathcal{W}} Z = 0$.
2. Any feasible direction satisfies $p = Z p_z$.
3. The reduced gradient system becomes:
$$
(Z^T Q Z) p_z = - Z^T g_k
$$
4. Since $Q \\succeq 0$ and LICQ holds, the reduced Hessian $Z^T Q Z$ is strictly symmetric positive definite ($Z^T Q Z \\succ 0$), allowing rapid solution via dense or sparse Cholesky factorization without indefinite pivoting.
"""

expansions_sec14 = """
### 14.4 Arithmetic Intensity & GPU Roofline Analysis
Sparse matrix-vector multiplication (SpMV) is an intrinsically memory-bandwidth-bound numerical operation. For a double-precision (FP64) CSR matrix $A \\in \\mathbb{R}^{m \\times n}$ with $\\text{nnz}$ nonzeros and a batch of $B$ vectors:
- **Floating-point operations (FLOPs):** $2 \\cdot B \\cdot \\text{nnz}$ FLOPs.
- **Memory Traffic (Bytes transferred):**
  - Matrix values: $8 \\cdot \\text{nnz}$ bytes.
  - Column indices: $4 \\cdot \\text{nnz}$ bytes.
  - Row pointers: $4 \\cdot (m + 1)$ bytes.
  - Input vectors $X$: $8 \\cdot B \\cdot n$ bytes.
  - Output vectors $Y$: $8 \\cdot B \\cdot m$ bytes.

The operational arithmetic intensity $I(B)$ is:

$$
I(B) = \\frac{2 \\cdot B \\cdot \\text{nnz}}{12 \\cdot \\text{nnz} + 4(m + 1) + 8 B (m + n)} \\quad \\left[ \\frac{\\text{FLOP}}{\\text{Byte}} \\right]
$$

For single-vector execution ($B = 1$), $I(1) \\approx 0.10 - 0.15$ FLOP/Byte, placing the operation deep within the memory-bandwidth-limited regime of modern GPUs. For large batches ($B = 128$), matrix reads are amortized across all vectors, elevating $I(128) \\to 0.25$ FLOP/Byte and amortizing host-device PCIe bus latency ($T_{\\text{H2D}} + T_{\\text{D2H}}$). This rigorous physics model underpins Strategy Rule R013.
"""

# Insert expansions into the document at appropriate section marks
doc = doc.replace("### 4.4 Iterative Refinement", expansions_sec4 + "\n### 4.4 Iterative Refinement")
doc = doc.replace("### 7.4 Basis Updating", expansions_sec7 + "\n### 7.4 Basis Updating")
doc = doc.replace("### 10.3 Cut Pool Management & Filtering", expansions_sec10 + "\n### 10.3 Cut Pool Management & Filtering")
doc = doc.replace("### 11.5 Cholesky Positive Semidefinite Certification", expansions_sec11 + "\n### 11.5 Cholesky Positive Semidefinite Certification")
doc = doc.replace("### 14.3 Transfer-Inclusive GPU Timing Policy", expansions_sec14 + "\n### 14.3 Transfer-Inclusive GPU Timing Policy")

with open("docs/MATHEMATICS.md", "w") as f:
    f.write(doc.strip() + "\n")

line_count = len(doc.strip().splitlines()) + 1
print(f"Updated docs/MATHEMATICS.md: now {line_count} lines.")
