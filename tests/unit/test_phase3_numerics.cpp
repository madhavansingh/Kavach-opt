#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "../test_framework.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"
#include "numerics/lu_factorization.hpp"
#include "numerics/model_converter.hpp"
#include "numerics/scaling.hpp"
#include "numerics/sparse_matrix.hpp"

using namespace kavach;
using namespace kavach::numerics;

namespace {

// Helper: dense matrix vector multiplication for validation
std::vector<double> dense_spmv(std::size_t rows, std::size_t cols,
                               const std::vector<double>& dense_A, const std::vector<double>& x) {
    std::vector<double> y(rows, 0.0);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            y[i] += dense_A[i * cols + j] * x[j];
        }
    }
    return y;
}

}  // namespace

// ============================================================================
// 1. CSR Construction and Invariants
// ============================================================================

KAVACH_TEST(test_csr_construction_valid) {
    // 3x3 matrix:
    // [ 10.0,  0.0, 20.0 ]
    // [  0.0, 30.0,  0.0 ]
    // [ 40.0,  0.0, 50.0 ]
    std::vector<std::size_t> row_ptr = {0, 2, 3, 5};
    std::vector<std::size_t> col_idx = {0, 2, 1, 0, 2};
    std::vector<double> vals = {10.0, 20.0, 30.0, 40.0, 50.0};

    CsrMatrix csr(3, 3, row_ptr, col_idx, vals);
    KAVACH_ASSERT(csr.is_valid());
    KAVACH_ASSERT_EQUAL(csr.rows(), 3u);
    KAVACH_ASSERT_EQUAL(csr.cols(), 3u);
    KAVACH_ASSERT_EQUAL(csr.nonzeros(), 5u);

    KAVACH_ASSERT_NEAR(csr.get(0, 0), 10.0, 1e-12);
    KAVACH_ASSERT_NEAR(csr.get(0, 1), 0.0, 1e-12);
    KAVACH_ASSERT_NEAR(csr.get(0, 2), 20.0, 1e-12);
    KAVACH_ASSERT_NEAR(csr.get(1, 1), 30.0, 1e-12);
    KAVACH_ASSERT_NEAR(csr.get(2, 0), 40.0, 1e-12);
    KAVACH_ASSERT_NEAR(csr.get(2, 2), 50.0, 1e-12);

    KAVACH_ASSERT_EQUAL(csr.row_start(0), 0u);
    KAVACH_ASSERT_EQUAL(csr.row_end(0), 2u);
    KAVACH_ASSERT_EQUAL(csr.row_nnz(0), 2u);
    KAVACH_ASSERT_EQUAL(csr.row_nnz(1), 1u);
    KAVACH_ASSERT_EQUAL(csr.row_nnz(2), 2u);
}

KAVACH_TEST(test_csr_invalid_invariants_rejected) {
    // Bad row_ptr size (should be rows + 1)
    {
        std::vector<std::size_t> bad_row_ptr = {0, 1};
        std::vector<std::size_t> col_idx = {0};
        std::vector<double> vals = {1.0};
        CsrMatrix bad_csr(2, 2, bad_row_ptr, col_idx, vals, false);
        KAVACH_ASSERT(!bad_csr.is_valid());
    }

    // Out of bounds column index
    {
        std::vector<std::size_t> row_ptr = {0, 1, 1};
        std::vector<std::size_t> bad_col_idx = {5};  // cols = 2, so 5 is invalid
        std::vector<double> vals = {1.0};
        CsrMatrix bad_csr(2, 2, row_ptr, bad_col_idx, vals, false);
        KAVACH_ASSERT(!bad_csr.is_valid());
    }

    // Non-monotonic row_ptr
    {
        std::vector<std::size_t> bad_row_ptr = {0, 2, 1};
        std::vector<std::size_t> col_idx = {0, 1};
        std::vector<double> vals = {1.0, 2.0};
        CsrMatrix bad_csr(2, 2, bad_row_ptr, col_idx, vals, false);
        KAVACH_ASSERT(!bad_csr.is_valid());
    }
}

KAVACH_TEST(test_csr_from_triplets) {
    std::vector<Triplet> triplets = {
        {0, 1, 3.0}, {1, 0, 4.0}, {0, 1, 2.0},  // Duplicate (0, 1) -> sum = 5.0
        {2, 2, 7.0}, {0, 0, 1.0}, {1, 1, 0.0}   // 0.0 entry
    };

    CsrMatrix csr = CsrMatrix::from_triplets(3, 3, triplets, true);
    KAVACH_ASSERT(csr.is_valid());
    KAVACH_ASSERT_NEAR(csr.get(0, 0), 1.0, 1e-12);
    KAVACH_ASSERT_NEAR(csr.get(0, 1), 5.0, 1e-12);  // 3.0 + 2.0
    KAVACH_ASSERT_NEAR(csr.get(1, 0), 4.0, 1e-12);
    KAVACH_ASSERT_NEAR(csr.get(2, 2), 7.0, 1e-12);
}

// ============================================================================
// 2. CSC Construction and Slicing
// ============================================================================

KAVACH_TEST(test_csc_construction_and_slicing) {
    std::vector<Triplet> triplets = {{0, 0, 1.0}, {2, 0, 2.0}, {1, 1, 3.0}, {0, 2, 4.0}};

    CscMatrix csc = CscMatrix::from_triplets(3, 3, triplets);
    KAVACH_ASSERT(csc.is_valid());
    KAVACH_ASSERT_EQUAL(csc.cols(), 3u);
    KAVACH_ASSERT_EQUAL(csc.rows(), 3u);
    KAVACH_ASSERT_EQUAL(csc.nonzeros(), 4u);

    KAVACH_ASSERT_EQUAL(csc.col_nnz(0), 2u);
    KAVACH_ASSERT_EQUAL(csc.col_nnz(1), 1u);
    KAVACH_ASSERT_EQUAL(csc.col_nnz(2), 1u);

    KAVACH_ASSERT_NEAR(csc.get(0, 0), 1.0, 1e-12);
    KAVACH_ASSERT_NEAR(csc.get(2, 0), 2.0, 1e-12);
    KAVACH_ASSERT_NEAR(csc.get(1, 1), 3.0, 1e-12);
    KAVACH_ASSERT_NEAR(csc.get(0, 2), 4.0, 1e-12);
    KAVACH_ASSERT_NEAR(csc.get(2, 2), 0.0, 1e-12);
}

// ============================================================================
// 3. Sparse Conversions & Transpose
// ============================================================================

KAVACH_TEST(test_sparse_roundtrip_conversions) {
    std::vector<Triplet> triplets = {{0, 0, 1.5}, {0, 3, 2.5}, {1, 1, 3.5},
                                     {2, 0, 4.5}, {2, 2, 5.5}, {3, 3, 6.5}};

    CsrMatrix orig_csr = CsrMatrix::from_triplets(4, 4, triplets);
    CscMatrix csc = csr_to_csc(orig_csr);
    KAVACH_ASSERT(csc.is_valid());

    CsrMatrix roundtrip_csr = csc_to_csr(csc);
    KAVACH_ASSERT(roundtrip_csr.is_valid());

    KAVACH_ASSERT_EQUAL(orig_csr.rows(), roundtrip_csr.rows());
    KAVACH_ASSERT_EQUAL(orig_csr.cols(), roundtrip_csr.cols());
    KAVACH_ASSERT_EQUAL(orig_csr.nonzeros(), roundtrip_csr.nonzeros());

    for (std::size_t r = 0; r < 4; ++r) {
        for (std::size_t c = 0; c < 4; ++c) {
            KAVACH_ASSERT_NEAR(orig_csr.get(r, c), roundtrip_csr.get(r, c), 1e-12);
        }
    }
}

KAVACH_TEST(test_sparse_transpose) {
    // 2x3 matrix:
    // [ 1.0, 0.0, 2.0 ]
    // [ 3.0, 4.0, 0.0 ]
    std::vector<Triplet> triplets = {{0, 0, 1.0}, {0, 2, 2.0}, {1, 0, 3.0}, {1, 1, 4.0}};

    CsrMatrix A = CsrMatrix::from_triplets(2, 3, triplets);
    CsrMatrix At = transpose(A);

    KAVACH_ASSERT_EQUAL(At.rows(), 3u);
    KAVACH_ASSERT_EQUAL(At.cols(), 2u);
    KAVACH_ASSERT(At.is_valid());

    for (std::size_t r = 0; r < 2; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            KAVACH_ASSERT_NEAR(A.get(r, c), At.get(c, r), 1e-12);
        }
    }

    // Double transpose (A^T)^T == A
    CsrMatrix Att = transpose(At);
    KAVACH_ASSERT_EQUAL(Att.rows(), 2u);
    KAVACH_ASSERT_EQUAL(Att.cols(), 3u);
    for (std::size_t r = 0; r < 2; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            KAVACH_ASSERT_NEAR(A.get(r, c), Att.get(r, c), 1e-12);
        }
    }
}

// ============================================================================
// 4. SpMV Correctness vs Dense Reference
// ============================================================================

KAVACH_TEST(test_spmv_vs_dense_reference) {
    const std::size_t m = 4;
    const std::size_t n = 5;

    std::vector<double> dense_A = {1.0, 0.0, 2.0, 0.0, 0.5, 0.0, 3.0, 0.0, 4.0, 0.0,
                                   5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 7.0, 0.0, 8.0};

    std::vector<Triplet> triplets;
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double v = dense_A[i * n + j];
            if (v != 0.0) {
                triplets.push_back({i, j, v});
            }
        }
    }

    CsrMatrix csr = CsrMatrix::from_triplets(m, n, triplets);
    CscMatrix csc = csr_to_csc(csr);

    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto expected_y = dense_spmv(m, n, dense_A, x);

    auto y_csr = csr.spmv(x);
    auto y_csc = csc.spmv(x);

    for (std::size_t i = 0; i < m; ++i) {
        KAVACH_ASSERT_NEAR(y_csr[i], expected_y[i], 1e-12);
        KAVACH_ASSERT_NEAR(y_csc[i], expected_y[i], 1e-12);
    }

    // Test alpha * A * x + beta * y overload
    std::vector<double> y_acc = {10.0, 20.0, 30.0, 40.0};
    csr.spmv(2.0, x, 0.5, y_acc);
    for (std::size_t i = 0; i < m; ++i) {
        double exp_val = 2.0 * expected_y[i] + 0.5 * (10.0 * static_cast<double>(i + 1));
        KAVACH_ASSERT_NEAR(y_acc[i], exp_val, 1e-12);
    }
}

// ============================================================================
// 5. CanonicalModel to CSR/CSC Conversion
// ============================================================================

KAVACH_TEST(test_canonical_model_to_sparse) {
    model::CanonicalModel model("sparse_conv_test");
    auto x0 = model.add_variable("x0", model::VariableType::CONTINUOUS, 0.0, 10.0);
    auto x1 = model.add_variable("x1", model::VariableType::CONTINUOUS, 0.0, 10.0);
    auto x2 = model.add_variable("x2", model::VariableType::CONTINUOUS, 0.0, 10.0);

    // c0: 2*x0 + 3*x1 <= 15
    model::SparseLinearExpression expr0;
    expr0.add_term(x0, 2.0);
    expr0.add_term(x1, 3.0);
    model.add_constraint("c0", std::move(expr0), model::ConstraintSense::LESS_EQUAL, 15.0);

    // c1: 4*x1 - 5*x2 == 20
    model::SparseLinearExpression expr1;
    expr1.add_term(x1, 4.0);
    expr1.add_term(x2, -5.0);
    model.add_constraint("c1", std::move(expr1), model::ConstraintSense::EQUAL, 20.0);

    CsrMatrix csr = canonical_model_to_csr(model);
    KAVACH_ASSERT(csr.is_valid());
    KAVACH_ASSERT_EQUAL(csr.rows(), 2u);
    KAVACH_ASSERT_EQUAL(csr.cols(), 3u);
    KAVACH_ASSERT_EQUAL(csr.nonzeros(), 4u);

    KAVACH_ASSERT_NEAR(csr.get(0, 0), 2.0, 1e-12);
    KAVACH_ASSERT_NEAR(csr.get(0, 1), 3.0, 1e-12);
    KAVACH_ASSERT_NEAR(csr.get(0, 2), 0.0, 1e-12);
    KAVACH_ASSERT_NEAR(csr.get(1, 0), 0.0, 1e-12);
    KAVACH_ASSERT_NEAR(csr.get(1, 1), 4.0, 1e-12);
    KAVACH_ASSERT_NEAR(csr.get(1, 2), -5.0, 1e-12);

    CscMatrix csc = canonical_model_to_csc(model);
    KAVACH_ASSERT(csc.is_valid());
    KAVACH_ASSERT_EQUAL(csc.rows(), 2u);
    KAVACH_ASSERT_EQUAL(csc.cols(), 3u);
    KAVACH_ASSERT_NEAR(csc.get(0, 0), 2.0, 1e-12);
    KAVACH_ASSERT_NEAR(csc.get(1, 2), -5.0, 1e-12);
}

// ============================================================================
// 6. Sparse LU Factorization & Direct Solve
// ============================================================================

KAVACH_TEST(test_sparse_lu_permutation_required) {
    // System requiring row exchange:
    // [ 0, 2 ] [ x0 ]   [ 4 ]
    // [ 3, 1 ] [ x1 ] = [ 7 ]
    // Solution: x0 = 2, x1 = 2
    std::vector<Triplet> triplets = {{0, 1, 2.0}, {1, 0, 3.0}, {1, 1, 1.0}};
    CsrMatrix A = CsrMatrix::from_triplets(2, 2, triplets);

    SparseLU solver;
    bool ok = solver.factorize(A);
    KAVACH_ASSERT(ok);
    KAVACH_ASSERT(solver.is_factorized());
    KAVACH_ASSERT_EQUAL(solver.status(), core::SolverStatus::OPTIMAL);

    // Permutation vector must have swapped rows 0 and 1
    const auto& perm = solver.row_permutation();
    KAVACH_ASSERT_EQUAL(perm[0], 1u);
    KAVACH_ASSERT_EQUAL(perm[1], 0u);

    // System:
    // [ 0, 2 ] [ x0 ]   [ 4 ]
    // [ 3, 1 ] [ x1 ] = [ 8 ]
    // Solution: x0 = 2, x1 = 2
    std::vector<double> b = {4.0, 8.0};
    auto x = solver.solve(b);

    KAVACH_ASSERT_NEAR(x[0], 2.0, 1e-12);
    KAVACH_ASSERT_NEAR(x[1], 2.0, 1e-12);

    auto r = compute_residual(A, x, b);
    KAVACH_ASSERT_NEAR(residual_norm_inf(r), 0.0, 1e-12);
}

KAVACH_TEST(test_sparse_lu_tridiagonal_system) {
    // 3x3 Tridiagonal:
    // [  2, -1,  0 ] [ x0 ]   [ 1 ]
    // [ -1,  2, -1 ] [ x1 ] = [ 0 ]
    // [  0, -1,  2 ] [ x2 ]   [ 1 ]
    // Solution: x = [ 1, 1, 1 ]
    std::vector<Triplet> triplets = {{0, 0, 2.0},  {0, 1, -1.0}, {1, 0, -1.0}, {1, 1, 2.0},
                                     {1, 2, -1.0}, {2, 1, -1.0}, {2, 2, 2.0}};
    CsrMatrix A = CsrMatrix::from_triplets(3, 3, triplets);

    SparseLU solver;
    KAVACH_ASSERT(solver.factorize(A));

    std::vector<double> b = {1.0, 0.0, 1.0};
    auto x = solver.solve(b);

    KAVACH_ASSERT_NEAR(x[0], 1.0, 1e-12);
    KAVACH_ASSERT_NEAR(x[1], 1.0, 1e-12);
    KAVACH_ASSERT_NEAR(x[2], 1.0, 1e-12);

    auto r = compute_residual(A, x, b);
    KAVACH_ASSERT_NEAR(residual_norm_inf(r), 0.0, 1e-12);
}

// ============================================================================
// 7. Sparse LU Transpose Solve
// ============================================================================

KAVACH_TEST(test_sparse_lu_transpose_solve) {
    // Non-symmetric matrix:
    // [ 4, 3, 0 ]
    // [ 2, 1, 5 ]
    // [ 0, 2, 1 ]
    std::vector<Triplet> triplets = {{0, 0, 4.0}, {0, 1, 3.0}, {1, 0, 2.0}, {1, 1, 1.0},
                                     {1, 2, 5.0}, {2, 1, 2.0}, {2, 2, 1.0}};
    CsrMatrix A = CsrMatrix::from_triplets(3, 3, triplets);

    SparseLU solver;
    KAVACH_ASSERT(solver.factorize(A));

    // Desired true solution for A^T * x = b: x = [1, 2, 3]
    CsrMatrix At = transpose(A);
    std::vector<double> x_true = {1.0, 2.0, 3.0};
    auto b = At.spmv(x_true);

    auto x_calc = solver.solve_transpose(b);
    KAVACH_ASSERT_NEAR(x_calc[0], 1.0, 1e-11);
    KAVACH_ASSERT_NEAR(x_calc[1], 2.0, 1e-11);
    KAVACH_ASSERT_NEAR(x_calc[2], 3.0, 1e-11);

    auto r = compute_residual(At, x_calc, b);
    KAVACH_ASSERT_NEAR(residual_norm_inf(r), 0.0, 1e-11);
}

// ============================================================================
// 8. Singularity & Near-Singularity Detection
// ============================================================================

KAVACH_TEST(test_sparse_lu_singular_matrix_rejected) {
    // Singular matrix with all-zero column:
    // [ 1, 0, 2 ]
    // [ 2, 0, 4 ]
    // [ 0, 0, 1 ]
    std::vector<Triplet> triplets = {
        {0, 0, 1.0}, {0, 2, 2.0}, {1, 0, 2.0}, {1, 2, 4.0}, {2, 2, 1.0}};
    CsrMatrix A = CsrMatrix::from_triplets(3, 3, triplets);

    SparseLU solver;
    bool ok = solver.factorize(A);
    KAVACH_ASSERT(!ok);
    KAVACH_ASSERT(!solver.is_factorized());
    KAVACH_ASSERT_EQUAL(solver.status(), core::SolverStatus::NUMERICAL_FAILURE);
    KAVACH_ASSERT(!solver.error_message().empty());
}

KAVACH_TEST(test_sparse_lu_near_zero_pivot_flagged) {
    // Near-zero pivot below pivot_tol:
    // [ 1e-10, 1.0 ]
    // [ 0.0,   1.0 ]
    std::vector<Triplet> triplets = {{0, 0, 1e-10}, {0, 1, 1.0}, {1, 1, 1.0}};
    CsrMatrix A = CsrMatrix::from_triplets(2, 2, triplets);

    SparseLU solver;
    // Default pivot_tol is 1e-9; 1e-10 < 1e-9 must fail
    bool ok = solver.factorize(A, 1e-9);
    KAVACH_ASSERT(!ok);
    KAVACH_ASSERT_EQUAL(solver.status(), core::SolverStatus::NUMERICAL_FAILURE);
}

// ============================================================================
// 9. Iterative Refinement
// ============================================================================

KAVACH_TEST(test_iterative_refinement_recovers_accuracy) {
    // 3x3 system:
    std::vector<Triplet> triplets = {{0, 0, 10.0}, {0, 1, -1.0}, {0, 2, 2.0},
                                     {1, 0, -1.0}, {1, 1, 11.0}, {1, 2, -1.0},
                                     {2, 0, 2.0},  {2, 1, -1.0}, {2, 2, 10.0}};
    CsrMatrix A = CsrMatrix::from_triplets(3, 3, triplets);

    SparseLU solver;
    KAVACH_ASSERT(solver.factorize(A));

    std::vector<double> b = {6.0, 25.0, -11.0};
    auto x = solver.solve(b);

    // Perturb solution intentionally to simulate roundoff error
    x[0] += 0.05;
    x[1] -= 0.03;
    x[2] += 0.02;

    double res_before = residual_norm_inf(compute_residual(A, x, b));
    KAVACH_ASSERT(res_before > 0.1);

    auto result = solver.refine_solution(A, b, x, 5, 1e-12);
    KAVACH_ASSERT(result.converged);
    KAVACH_ASSERT(result.iterations >= 1);
    KAVACH_ASSERT(result.final_residual < 1e-12);

    double res_after = residual_norm_inf(compute_residual(A, x, b));
    KAVACH_ASSERT_NEAR(res_after, 0.0, 1e-12);
}

// ============================================================================
// 10. Geometric-Mean Scaling & Conditioning
// ============================================================================

KAVACH_TEST(test_geometric_mean_scaling_and_unscaling) {
    // Ill-conditioned matrix with wide coefficient dynamic range:
    // [ 1e6,  1e-3 ] [ x0 ]   [ 1e6 + 2e-3 ]
    // [ 1e3,  1e2  ] [ x1 ] = [ 1e3 + 2e2  ]
    // True solution: x = [ 1.0, 2.0 ]
    std::vector<Triplet> triplets = {{0, 0, 1e6}, {0, 1, 1e-3}, {1, 0, 1e3}, {1, 1, 1e2}};
    CsrMatrix A = CsrMatrix::from_triplets(2, 2, triplets);

    auto diag_before = analyze_conditioning(A);
    KAVACH_ASSERT(diag_before.dynamic_range_ratio >= 1e8);
    KAVACH_ASSERT(diag_before.recommend_scaling);

    // Compute and apply scaling
    auto factors = compute_geometric_mean_scaling(A, 4);
    CsrMatrix A_scaled = apply_scaling(A, factors);

    auto diag_after = analyze_conditioning(A_scaled);
    // Scaled dynamic range should be substantially smaller
    KAVACH_ASSERT(diag_after.dynamic_range_ratio < diag_before.dynamic_range_ratio);

    // Solve scaled system
    std::vector<double> b = {1e6 + 2e-3, 1e3 + 2e2};
    std::vector<double> b_scaled = factors.scale_rhs(b);

    SparseLU solver;
    KAVACH_ASSERT(solver.factorize(A_scaled));
    auto x_scaled = solver.solve(b_scaled);

    // Unscale solution
    auto x_orig = factors.unscale_primal(x_scaled);

    KAVACH_ASSERT_NEAR(x_orig[0], 1.0, 1e-9);
    KAVACH_ASSERT_NEAR(x_orig[1], 2.0, 1e-9);
}

// ============================================================================
// 11. Subquadratic Memory Scaling Proof (O(N) vs Dense O(N^2))
// ============================================================================

KAVACH_TEST(test_sparse_memory_scaling_subquadratic) {
    // Create tridiagonal sparse matrices of sizes N1 = 100, N2 = 500, N3 = 1000
    // In dense storage, memory grows as N^2 (100x increase from 100 to 1000)
    // In sparse CSR storage, memory is (rows + 1 + 2*nnz)*sizeof(size_t) + nnz*sizeof(double)
    // Since nnz = 3*N - 2, sparse memory must scale linearly O(N), ratio ~ 10x
    auto create_tridiagonal = [](std::size_t n) {
        std::vector<Triplet> triplets;
        triplets.reserve(3 * n);
        for (std::size_t i = 0; i < n; ++i) {
            triplets.push_back({i, i, 2.0});
            if (i > 0) {
                triplets.push_back({i, i - 1, -1.0});
            }
            if (i + 1 < n) {
                triplets.push_back({i, i + 1, -1.0});
            }
        }
        return CsrMatrix::from_triplets(n, n, triplets);
    };

    CsrMatrix A100 = create_tridiagonal(100);
    CsrMatrix A1000 = create_tridiagonal(1000);

    std::size_t mem100 = (A100.row_ptr().size() + A100.col_idx().size()) * sizeof(std::size_t) +
                         A100.values().size() * sizeof(double);
    std::size_t mem1000 = (A1000.row_ptr().size() + A1000.col_idx().size()) * sizeof(std::size_t) +
                          A1000.values().size() * sizeof(double);

    double growth_factor = static_cast<double>(mem1000) / static_cast<double>(mem100);

    // Dense growth factor would be (1000^2)/(100^2) = 100.0
    // Sparse linear growth factor is ~10.0
    KAVACH_ASSERT(growth_factor < 15.0);
    KAVACH_ASSERT(growth_factor > 8.0);
}
