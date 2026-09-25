#include <chrono>
#include <cmath>
#include <iomanip>
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

// Helper: dense transpose matrix vector multiplication for validation
std::vector<double> dense_spmv_transpose(std::size_t rows, std::size_t cols,
                                         const std::vector<double>& dense_A,
                                         const std::vector<double>& x) {
    std::vector<double> y(cols, 0.0);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            y[j] += dense_A[i * cols + j] * x[i];
        }
    }
    return y;
}

struct ErrorMetrics {
    double max_abs_error{0.0};
    double max_rel_error{0.0};
};

ErrorMetrics compute_error_metrics(const std::vector<double>& actual,
                                   const std::vector<double>& expected) {
    ErrorMetrics em;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        double abs_err = std::abs(actual[i] - expected[i]);
        if (abs_err > em.max_abs_error) {
            em.max_abs_error = abs_err;
        }
        double denom = std::max(1.0, std::abs(expected[i]));
        double rel_err = abs_err / denom;
        if (rel_err > em.max_rel_error) {
            em.max_rel_error = rel_err;
        }
    }
    return em;
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

// ============================================================================
// 12. Requirements A & B: Direct CSR / CSC Construction & Slice Accessors
// ============================================================================

KAVACH_TEST(test_req_a_csr_construction) {
    // 3x4 matrix:
    // [ 1.0, 0.0, 2.0, 0.0 ]
    // [ 0.0, 3.0, 0.0, 4.0 ]
    // [ 5.0, 6.0, 0.0, 7.0 ]
    std::vector<std::size_t> row_ptr = {0, 2, 4, 7};
    std::vector<std::size_t> col_idx = {0, 2, 1, 3, 0, 1, 3};
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

    CsrMatrix csr(3, 4, row_ptr, col_idx, values);
    KAVACH_ASSERT(csr.is_valid());
    KAVACH_ASSERT_EQUAL(csr.rows(), 3u);
    KAVACH_ASSERT_EQUAL(csr.cols(), 4u);
    KAVACH_ASSERT_EQUAL(csr.nonzeros(), 7u);

    // Verify row pointer intervals and slice accessors
    KAVACH_ASSERT_EQUAL(csr.row_start(0), 0u);
    KAVACH_ASSERT_EQUAL(csr.row_end(0), 2u);
    KAVACH_ASSERT_EQUAL(csr.row_nnz(0), 2u);

    const std::size_t* r0_cols = csr.row_indices(0);
    const double* r0_vals = csr.row_values(0);
    KAVACH_ASSERT_EQUAL(r0_cols[0], 0u);
    KAVACH_ASSERT_EQUAL(r0_cols[1], 2u);
    KAVACH_ASSERT_NEAR(r0_vals[0], 1.0, 1e-15);
    KAVACH_ASSERT_NEAR(r0_vals[1], 2.0, 1e-15);

    const std::size_t* r2_cols = csr.row_indices(2);
    const double* r2_vals = csr.row_values(2);
    KAVACH_ASSERT_EQUAL(csr.row_nnz(2), 3u);
    KAVACH_ASSERT_EQUAL(r2_cols[0], 0u);
    KAVACH_ASSERT_EQUAL(r2_cols[1], 1u);
    KAVACH_ASSERT_EQUAL(r2_cols[2], 3u);
    KAVACH_ASSERT_NEAR(r2_vals[0], 5.0, 1e-15);
    KAVACH_ASSERT_NEAR(r2_vals[1], 6.0, 1e-15);
    KAVACH_ASSERT_NEAR(r2_vals[2], 7.0, 1e-15);

    // Coordinate get
    KAVACH_ASSERT_NEAR(csr.get(0, 0), 1.0, 1e-15);
    KAVACH_ASSERT_NEAR(csr.get(0, 1), 0.0, 1e-15);
    KAVACH_ASSERT_NEAR(csr.get(1, 3), 4.0, 1e-15);
    KAVACH_ASSERT_NEAR(csr.get(2, 2), 0.0, 1e-15);
}

KAVACH_TEST(test_req_b_csc_construction) {
    // 4x3 matrix:
    // [ 1.0, 0.0, 5.0 ]
    // [ 0.0, 3.0, 6.0 ]
    // [ 2.0, 0.0, 0.0 ]
    // [ 0.0, 4.0, 7.0 ]
    std::vector<std::size_t> col_ptr = {0, 2, 4, 7};
    std::vector<std::size_t> row_idx = {0, 2, 1, 3, 0, 1, 3};
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

    CscMatrix csc(4, 3, col_ptr, row_idx, values);
    KAVACH_ASSERT(csc.is_valid());
    KAVACH_ASSERT_EQUAL(csc.rows(), 4u);
    KAVACH_ASSERT_EQUAL(csc.cols(), 3u);
    KAVACH_ASSERT_EQUAL(csc.nonzeros(), 7u);

    // Verify column pointer intervals and slice accessors
    KAVACH_ASSERT_EQUAL(csc.col_start(0), 0u);
    KAVACH_ASSERT_EQUAL(csc.col_end(0), 2u);
    KAVACH_ASSERT_EQUAL(csc.col_nnz(0), 2u);

    const std::size_t* c0_rows = csc.col_indices(0);
    const double* c0_vals = csc.col_values(0);
    KAVACH_ASSERT_EQUAL(c0_rows[0], 0u);
    KAVACH_ASSERT_EQUAL(c0_rows[1], 2u);
    KAVACH_ASSERT_NEAR(c0_vals[0], 1.0, 1e-15);
    KAVACH_ASSERT_NEAR(c0_vals[1], 2.0, 1e-15);

    const std::size_t* c2_rows = csc.col_indices(2);
    const double* c2_vals = csc.col_values(2);
    KAVACH_ASSERT_EQUAL(csc.col_nnz(2), 3u);
    KAVACH_ASSERT_EQUAL(c2_rows[0], 0u);
    KAVACH_ASSERT_EQUAL(c2_rows[1], 1u);
    KAVACH_ASSERT_EQUAL(c2_rows[2], 3u);
    KAVACH_ASSERT_NEAR(c2_vals[0], 5.0, 1e-15);
    KAVACH_ASSERT_NEAR(c2_vals[1], 6.0, 1e-15);
    KAVACH_ASSERT_NEAR(c2_vals[2], 7.0, 1e-15);

    // Coordinate get
    KAVACH_ASSERT_NEAR(csc.get(0, 0), 1.0, 1e-15);
    KAVACH_ASSERT_NEAR(csc.get(0, 1), 0.0, 1e-15);
    KAVACH_ASSERT_NEAR(csc.get(3, 1), 4.0, 1e-15);
    KAVACH_ASSERT_NEAR(csc.get(2, 2), 0.0, 1e-15);
}

// ============================================================================
// 13. Requirements C & D: Conversions CSR <-> CSC with Non-Zero Preservation
// ============================================================================

KAVACH_TEST(test_req_c_and_d_csr_csc_conversions) {
    // Generate a random 6x8 sparse matrix
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> val_dist(-10.0, 10.0);
    std::uniform_int_distribution<std::size_t> r_dist(0, 5);
    std::uniform_int_distribution<std::size_t> c_dist(0, 7);

    std::vector<Triplet> triplets;
    for (int k = 0; k < 25; ++k) {
        triplets.push_back({r_dist(rng), c_dist(rng), val_dist(rng)});
    }

    CsrMatrix original_csr = CsrMatrix::from_triplets(6, 8, triplets);
    KAVACH_ASSERT(original_csr.is_valid());

    // C: CSR -> CSC
    CscMatrix csc = csr_to_csc(original_csr);
    KAVACH_ASSERT(csc.is_valid());
    KAVACH_ASSERT_EQUAL(csc.rows(), original_csr.rows());
    KAVACH_ASSERT_EQUAL(csc.cols(), original_csr.cols());
    KAVACH_ASSERT_EQUAL(csc.nonzeros(), original_csr.nonzeros());

    for (std::size_t r = 0; r < 6; ++r) {
        for (std::size_t c = 0; c < 8; ++c) {
            KAVACH_ASSERT_NEAR(original_csr.get(r, c), csc.get(r, c), 1e-14);
        }
    }

    // D: CSC -> CSR (round-trip identity)
    CsrMatrix reconstructed_csr = csc_to_csr(csc);
    KAVACH_ASSERT(reconstructed_csr.is_valid());
    KAVACH_ASSERT_EQUAL(reconstructed_csr.rows(), original_csr.rows());
    KAVACH_ASSERT_EQUAL(reconstructed_csr.cols(), original_csr.cols());
    KAVACH_ASSERT_EQUAL(reconstructed_csr.nonzeros(), original_csr.nonzeros());

    KAVACH_ASSERT_EQUAL(reconstructed_csr.row_ptr().size(), original_csr.row_ptr().size());
    KAVACH_ASSERT_EQUAL(reconstructed_csr.col_idx().size(), original_csr.col_idx().size());
    KAVACH_ASSERT_EQUAL(reconstructed_csr.values().size(), original_csr.values().size());

    for (std::size_t i = 0; i < original_csr.row_ptr().size(); ++i) {
        KAVACH_ASSERT_EQUAL(reconstructed_csr.row_ptr()[i], original_csr.row_ptr()[i]);
    }
    for (std::size_t i = 0; i < original_csr.col_idx().size(); ++i) {
        KAVACH_ASSERT_EQUAL(reconstructed_csr.col_idx()[i], original_csr.col_idx()[i]);
        KAVACH_ASSERT_NEAR(reconstructed_csr.values()[i], original_csr.values()[i], 1e-14);
    }
}

// ============================================================================
// 14. Requirements E, F, G: SpMV and SpMV Transpose vs Dense Reference
// ============================================================================

KAVACH_TEST(test_req_e_f_g_spmv_and_transpose_multiplication) {
    const std::size_t m = 5;
    const std::size_t n = 6;
    std::vector<double> dense_A = {
        2.5,  0.0, -1.2, 0.0,  0.0, 3.4, 0.0, 1.8, 0.0, 4.1, -0.5, 0.0,
        0.0,  0.0, 3.3,  0.0,  0.0, 0.0,                                // only 1 nnz
        -4.2, 0.0, 0.0,  -1.0, 2.2, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0  // completely empty row!
    };

    std::vector<Triplet> triplets;
    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            double v = dense_A[r * n + c];
            if (v != 0.0) {
                triplets.push_back({r, c, v});
            }
        }
    }

    CsrMatrix csr = CsrMatrix::from_triplets(m, n, triplets);
    CscMatrix csc = csr_to_csc(csr);

    std::vector<double> x = {1.5, -2.0, 0.5, 3.0, -1.0, 2.5};
    auto y_dense = dense_spmv(m, n, dense_A, x);

    // E: CSR SpMV
    auto y_csr = csr.spmv(x);
    auto err_csr = compute_error_metrics(y_csr, y_dense);
    KAVACH_ASSERT(err_csr.max_abs_error < 1e-14);
    KAVACH_ASSERT(err_csr.max_rel_error < 1e-14);

    // E: CSR SpMV pointer overload
    std::vector<double> y_csr_ptr(m, 0.0);
    csr.spmv(x.data(), y_csr_ptr.data());
    auto err_csr_ptr = compute_error_metrics(y_csr_ptr, y_dense);
    KAVACH_ASSERT(err_csr_ptr.max_abs_error < 1e-14);

    // E: CSR SpMV alpha * A * x + beta * y
    std::vector<double> y_csr_acc = {1.0, 2.0, 3.0, 4.0, 5.0};
    csr.spmv(2.5, x, -1.5, y_csr_acc);
    for (std::size_t i = 0; i < m; ++i) {
        double exp_val = 2.5 * y_dense[i] - 1.5 * static_cast<double>(i + 1);
        KAVACH_ASSERT_NEAR(y_csr_acc[i], exp_val, 1e-13);
    }

    // F: CSC SpMV
    auto y_csc = csc.spmv(x);
    auto err_csc = compute_error_metrics(y_csc, y_dense);
    KAVACH_ASSERT(err_csc.max_abs_error < 1e-14);
    KAVACH_ASSERT(err_csc.max_rel_error < 1e-14);

    // F: CSC SpMV pointer overload
    std::vector<double> y_csc_ptr(m, 0.0);
    csc.spmv(x.data(), y_csc_ptr.data());
    auto err_csc_ptr = compute_error_metrics(y_csc_ptr, y_dense);
    KAVACH_ASSERT(err_csc_ptr.max_abs_error < 1e-14);

    // F: CSC SpMV alpha * A * x + beta * y
    std::vector<double> y_csc_acc = {1.0, 2.0, 3.0, 4.0, 5.0};
    csc.spmv(2.5, x, -1.5, y_csc_acc);
    for (std::size_t i = 0; i < m; ++i) {
        double exp_val = 2.5 * y_dense[i] - 1.5 * static_cast<double>(i + 1);
        KAVACH_ASSERT_NEAR(y_csc_acc[i], exp_val, 1e-13);
    }

    // G: Transpose multiplication y = A^T * u
    std::vector<double> u = {2.0, -1.5, 3.0, 0.5, -4.0};
    auto yt_dense = dense_spmv_transpose(m, n, dense_A, u);

    // CSR spmv_transpose
    auto yt_csr = csr.spmv_transpose(u);
    auto err_yt_csr = compute_error_metrics(yt_csr, yt_dense);
    KAVACH_ASSERT(err_yt_csr.max_abs_error < 1e-14);
    KAVACH_ASSERT(err_yt_csr.max_rel_error < 1e-14);

    // CSR spmv_transpose pointer overload
    std::vector<double> yt_csr_ptr(n, 0.0);
    csr.spmv_transpose(u.data(), yt_csr_ptr.data());
    auto err_yt_csr_ptr = compute_error_metrics(yt_csr_ptr, yt_dense);
    KAVACH_ASSERT(err_yt_csr_ptr.max_abs_error < 1e-14);

    // CSR spmv_transpose alpha * A^T * u + beta * y
    std::vector<double> yt_csr_acc = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
    csr.spmv_transpose(3.0, u, 0.5, yt_csr_acc);
    for (std::size_t j = 0; j < n; ++j) {
        double exp_val = 3.0 * yt_dense[j] + 0.5 * (10.0 * static_cast<double>(j + 1));
        KAVACH_ASSERT_NEAR(yt_csr_acc[j], exp_val, 1e-13);
    }

    // CSC spmv_transpose
    auto yt_csc = csc.spmv_transpose(u);
    auto err_yt_csc = compute_error_metrics(yt_csc, yt_dense);
    KAVACH_ASSERT(err_yt_csc.max_abs_error < 1e-14);
    KAVACH_ASSERT(err_yt_csc.max_rel_error < 1e-14);

    // CSC spmv_transpose alpha * A^T * u + beta * y
    std::vector<double> yt_csc_acc = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
    csc.spmv_transpose(3.0, u, 0.5, yt_csc_acc);
    for (std::size_t j = 0; j < n; ++j) {
        double exp_val = 3.0 * yt_dense[j] + 0.5 * (10.0 * static_cast<double>(j + 1));
        KAVACH_ASSERT_NEAR(yt_csc_acc[j], exp_val, 1e-13);
    }
}

// ============================================================================
// 15. Requirements H & I: Duplicate Merging and Exact Zero Elimination
// ============================================================================

KAVACH_TEST(test_req_h_and_i_duplicates_and_zero_elimination) {
    std::vector<Triplet> triplets = {
        {0, 0, 2.5}, {0, 0, 1.5},                // Duplicate (0, 0) -> sum = 4.0
        {1, 1, 7.0}, {1, 1, -7.0},               // Cancellation to exact 0.0 -> must be pruned!
        {0, 2, 0.0},                             // Explicit 0.0 -> must be pruned!
        {2, 1, 3.0}, {2, 1, 2.0},  {2, 1, -1.0}  // Triplicate (2, 1) -> sum = 4.0
    };

    // CSR
    CsrMatrix csr = CsrMatrix::from_triplets(3, 3, triplets, true);
    KAVACH_ASSERT(csr.is_valid());
    KAVACH_ASSERT_EQUAL(csr.nonzeros(), 2u);  // Only (0, 0)=4.0 and (2, 1)=4.0 remain
    KAVACH_ASSERT_NEAR(csr.get(0, 0), 4.0, 1e-15);
    KAVACH_ASSERT_NEAR(csr.get(1, 1), 0.0, 1e-15);  // Cancelled to zero
    KAVACH_ASSERT_NEAR(csr.get(0, 2), 0.0, 1e-15);  // Explicit zero removed
    KAVACH_ASSERT_NEAR(csr.get(2, 1), 4.0, 1e-15);

    // CSC
    CscMatrix csc = CscMatrix::from_triplets(3, 3, triplets, true);
    KAVACH_ASSERT(csc.is_valid());
    KAVACH_ASSERT_EQUAL(csc.nonzeros(), 2u);
    KAVACH_ASSERT_NEAR(csc.get(0, 0), 4.0, 1e-15);
    KAVACH_ASSERT_NEAR(csc.get(1, 1), 0.0, 1e-15);
    KAVACH_ASSERT_NEAR(csc.get(0, 2), 0.0, 1e-15);
    KAVACH_ASSERT_NEAR(csc.get(2, 1), 4.0, 1e-15);
}

// ============================================================================
// 16. Requirements J & K: Empty Rows and Empty Columns
// ============================================================================

KAVACH_TEST(test_req_j_and_k_empty_rows_and_cols) {
    // 5x5 matrix where:
    // Rows 1 and 3 are completely empty (0 nonzeros)
    // Cols 0 and 4 are completely empty (0 nonzeros)
    // Nonzeros: (0, 1)=1.0, (0, 2)=2.0, (2, 1)=3.0, (2, 3)=4.0, (4, 2)=5.0, (4, 3)=6.0
    std::vector<Triplet> triplets = {{0, 1, 1.0}, {0, 2, 2.0}, {2, 1, 3.0},
                                     {2, 3, 4.0}, {4, 2, 5.0}, {4, 3, 6.0}};

    CsrMatrix csr = CsrMatrix::from_triplets(5, 5, triplets);
    KAVACH_ASSERT(csr.is_valid());
    KAVACH_ASSERT_EQUAL(csr.row_nnz(1), 0u);
    KAVACH_ASSERT_EQUAL(csr.row_nnz(3), 0u);
    KAVACH_ASSERT_EQUAL(csr.row_start(1), csr.row_end(1));
    KAVACH_ASSERT_EQUAL(csr.row_start(3), csr.row_end(3));

    CscMatrix csc = csr_to_csc(csr);
    KAVACH_ASSERT(csc.is_valid());
    KAVACH_ASSERT_EQUAL(csc.col_nnz(0), 0u);
    KAVACH_ASSERT_EQUAL(csc.col_nnz(4), 0u);
    KAVACH_ASSERT_EQUAL(csc.col_start(0), csc.col_end(0));
    KAVACH_ASSERT_EQUAL(csc.col_start(4), csc.col_end(4));

    // SpMV with vector of all 1s
    std::vector<double> x = {1.0, 1.0, 1.0, 1.0, 1.0};
    auto y = csr.spmv(x);
    KAVACH_ASSERT_NEAR(y[0], 3.0, 1e-15);
    KAVACH_ASSERT_NEAR(y[1], 0.0, 1e-15);  // Empty row produces 0
    KAVACH_ASSERT_NEAR(y[2], 7.0, 1e-15);
    KAVACH_ASSERT_NEAR(y[3], 0.0, 1e-15);  // Empty row produces 0
    KAVACH_ASSERT_NEAR(y[4], 11.0, 1e-15);

    // SpMV transpose: empty columns produce 0
    auto yt = csr.spmv_transpose(x);
    KAVACH_ASSERT_NEAR(yt[0], 0.0, 1e-15);  // Empty column produces 0
    KAVACH_ASSERT_NEAR(yt[1], 4.0, 1e-15);
    KAVACH_ASSERT_NEAR(yt[2], 7.0, 1e-15);
    KAVACH_ASSERT_NEAR(yt[3], 10.0, 1e-15);
    KAVACH_ASSERT_NEAR(yt[4], 0.0, 1e-15);  // Empty column produces 0

    // CSC matches
    auto y_csc = csc.spmv(x);
    for (std::size_t i = 0; i < 5; ++i) {
        KAVACH_ASSERT_NEAR(y_csc[i], y[i], 1e-15);
    }
    auto yt_csc = csc.spmv_transpose(x);
    for (std::size_t j = 0; j < 5; ++j) {
        KAVACH_ASSERT_NEAR(yt_csc[j], yt[j], 1e-15);
    }

    // Roundtrip conversion preserves empty rows/cols
    CsrMatrix roundtrip = csc_to_csr(csc);
    KAVACH_ASSERT(roundtrip.is_valid());
    KAVACH_ASSERT_EQUAL(roundtrip.row_nnz(1), 0u);
    KAVACH_ASSERT_EQUAL(roundtrip.row_nnz(3), 0u);
}

// ============================================================================
// 17. Requirement L: Invalid Indices & Bounds Rejection
// ============================================================================

KAVACH_TEST(test_req_l_invalid_indices_rejected) {
    // 1. Out-of-bounds row in triplet throws InputException
    {
        std::vector<Triplet> bad_row = {{3, 0, 1.0}};  // 3 >= 3
        bool caught = false;
        try {
            [[maybe_unused]] auto m = CsrMatrix::from_triplets(3, 3, bad_row);
        } catch (const core::InputException&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);

        caught = false;
        try {
            [[maybe_unused]] auto m = CscMatrix::from_triplets(3, 3, bad_row);
        } catch (const core::InputException&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);
    }

    // 2. Out-of-bounds col in triplet throws InputException
    {
        std::vector<Triplet> bad_col = {{0, 5, 1.0}};  // 5 >= 4
        bool caught = false;
        try {
            [[maybe_unused]] auto m = CsrMatrix::from_triplets(3, 4, bad_col);
        } catch (const core::InputException&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);

        caught = false;
        try {
            [[maybe_unused]] auto m = CscMatrix::from_triplets(3, 4, bad_col);
        } catch (const core::InputException&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);
    }

    // 3. Out-of-bounds row/col slice access throws InputException; get() returns 0.0 safely
    {
        CsrMatrix csr(2, 2);
        KAVACH_ASSERT_NEAR(csr.get(2, 0), 0.0, 1e-15);
        KAVACH_ASSERT_NEAR(csr.get(0, 2), 0.0, 1e-15);

        bool caught = false;
        try {
            [[maybe_unused]] auto ptr = csr.row_indices(2);
        } catch (const core::InputException&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);

        caught = false;
        try {
            [[maybe_unused]] auto ptr = csr.row_values(2);
        } catch (const core::InputException&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);

        CscMatrix csc(2, 2);
        KAVACH_ASSERT_NEAR(csc.get(2, 0), 0.0, 1e-15);
        KAVACH_ASSERT_NEAR(csc.get(0, 2), 0.0, 1e-15);

        caught = false;
        try {
            [[maybe_unused]] auto ptr = csc.col_indices(2);
        } catch (const core::InputException&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);

        caught = false;
        try {
            [[maybe_unused]] auto ptr = csc.col_values(2);
        } catch (const core::InputException&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);
    }

    // 4. Dimension mismatch in SpMV
    {
        CsrMatrix csr(3, 4);
        std::vector<double> bad_x = {1.0, 2.0};  // size 2 != 4
        bool caught = false;
        try {
            [[maybe_unused]] auto y = csr.spmv(bad_x);
        } catch (const core::InputException&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);

        caught = false;
        try {
            [[maybe_unused]] auto y = csr.spmv_transpose(bad_x);  // size 2 != 3
        } catch (const core::InputException&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);
    }
}

// ============================================================================
// 18. Requirement M: Malformed Pointer Arrays Validation
// ============================================================================

KAVACH_TEST(test_req_m_malformed_pointer_arrays) {
    std::string err;

    // Bad row_ptr[0] != 0
    {
        CsrMatrix bad_csr(2, 2, {1, 1, 1}, {0}, {1.0}, false);
        KAVACH_ASSERT(!bad_csr.is_valid(&err));
        KAVACH_ASSERT(!err.empty());
    }

    // Mismatched col_idx and values sizes
    {
        CsrMatrix bad_csr(2, 2, {0, 1, 2}, {0, 1, 0}, {1.0, 2.0}, false);
        KAVACH_ASSERT(!bad_csr.is_valid(&err));
        KAVACH_ASSERT(!err.empty());
    }

    // CSC Bad col_ptr[0] != 0
    {
        CscMatrix bad_csc(2, 2, {2, 2, 2}, {0}, {1.0}, false);
        KAVACH_ASSERT(!bad_csc.is_valid(&err));
        KAVACH_ASSERT(!err.empty());
    }

    // CSC non-monotonic col_ptr
    {
        CscMatrix bad_csc(2, 2, {0, 3, 1}, {0, 1, 0}, {1.0, 2.0, 3.0}, false);
        KAVACH_ASSERT(!bad_csc.is_valid(&err));
        KAVACH_ASSERT(!err.empty());
    }

    // CSC out-of-bounds row_idx
    {
        CscMatrix bad_csc(2, 2, {0, 1, 1}, {5}, {1.0}, false);
        KAVACH_ASSERT(!bad_csc.is_valid(&err));
        KAVACH_ASSERT(!err.empty());
    }
}

// ============================================================================
// 19. Requirement N: NaN and Infinity Rejection
// ============================================================================

KAVACH_TEST(test_req_n_nan_infinity_rejection) {
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    // 1. Triplets with NaN for CSR and CSC
    {
        std::vector<Triplet> nan_triplets = {{0, 0, qnan}};
        bool caught = false;
        try {
            [[maybe_unused]] auto m = CsrMatrix::from_triplets(2, 2, nan_triplets);
        } catch (const core::NumericalException&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);

        caught = false;
        try {
            [[maybe_unused]] auto m = CscMatrix::from_triplets(2, 2, nan_triplets);
        } catch (const core::NumericalException&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);
    }

    // 2. Triplets with +Inf for CSR and CSC
    {
        std::vector<Triplet> inf_triplets = {{0, 0, inf}};
        bool caught = false;
        try {
            [[maybe_unused]] auto m = CsrMatrix::from_triplets(2, 2, inf_triplets);
        } catch (const core::NumericalException&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);

        caught = false;
        try {
            [[maybe_unused]] auto m = CscMatrix::from_triplets(2, 2, inf_triplets);
        } catch (const core::NumericalException&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);
    }

    // 3. Triplets with -Inf for CSR and CSC
    {
        std::vector<Triplet> minf_triplets = {{0, 0, -inf}};
        bool caught = false;
        try {
            [[maybe_unused]] auto m = CsrMatrix::from_triplets(2, 2, minf_triplets);
        } catch (const core::NumericalException&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);

        caught = false;
        try {
            [[maybe_unused]] auto m = CscMatrix::from_triplets(2, 2, minf_triplets);
        } catch (const core::NumericalException&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);
    }

    // 4. Matrix validation checks non-finite values in arrays
    {
        CsrMatrix nan_csr(1, 1, {0, 1}, {0}, {qnan}, false);
        KAVACH_ASSERT(!nan_csr.is_valid());

        CscMatrix inf_csc(1, 1, {0, 1}, {0}, {inf}, false);
        KAVACH_ASSERT(!inf_csc.is_valid());
    }
}

// ============================================================================
// 20. Requirement O: Deterministic Construction Under Permutations
// ============================================================================

KAVACH_TEST(test_req_o_deterministic_construction) {
    // A base set of triplets including duplicate indices and cancellations
    std::vector<Triplet> base_triplets = {{0, 1, 2.0},  {2, 3, 5.0},  {1, 0, -1.0}, {0, 1, 3.0},
                                          {1, 2, 4.0},  {2, 3, -1.0}, {3, 0, 6.0},  {0, 0, 7.0},
                                          {2, 2, 8.0},  {1, 1, 9.0},  {3, 3, 10.0}, {0, 3, 1.5},
                                          {3, 1, -2.5}, {1, 0, 2.0}};

    CsrMatrix ref_csr = CsrMatrix::from_triplets(4, 4, base_triplets, true);
    CscMatrix ref_csc = CscMatrix::from_triplets(4, 4, base_triplets, true);

    std::mt19937 g(12345);
    for (int trial = 0; trial < 10; ++trial) {
        std::vector<Triplet> permuted = base_triplets;
        std::shuffle(permuted.begin(), permuted.end(), g);

        CsrMatrix cur_csr = CsrMatrix::from_triplets(4, 4, permuted, true);
        CscMatrix cur_csc = CscMatrix::from_triplets(4, 4, permuted, true);

        // Verify CSR deterministic identity
        KAVACH_ASSERT_EQUAL(cur_csr.row_ptr().size(), ref_csr.row_ptr().size());
        KAVACH_ASSERT_EQUAL(cur_csr.col_idx().size(), ref_csr.col_idx().size());
        KAVACH_ASSERT_EQUAL(cur_csr.values().size(), ref_csr.values().size());
        for (std::size_t i = 0; i < ref_csr.row_ptr().size(); ++i) {
            KAVACH_ASSERT_EQUAL(cur_csr.row_ptr()[i], ref_csr.row_ptr()[i]);
        }
        for (std::size_t i = 0; i < ref_csr.col_idx().size(); ++i) {
            KAVACH_ASSERT_EQUAL(cur_csr.col_idx()[i], ref_csr.col_idx()[i]);
            KAVACH_ASSERT_NEAR(cur_csr.values()[i], ref_csr.values()[i], 1e-15);
        }

        // Verify CSC deterministic identity
        KAVACH_ASSERT_EQUAL(cur_csc.col_ptr().size(), ref_csc.col_ptr().size());
        KAVACH_ASSERT_EQUAL(cur_csc.row_idx().size(), ref_csc.row_idx().size());
        KAVACH_ASSERT_EQUAL(cur_csc.values().size(), ref_csc.values().size());
        for (std::size_t i = 0; i < ref_csc.col_ptr().size(); ++i) {
            KAVACH_ASSERT_EQUAL(cur_csc.col_ptr()[i], ref_csc.col_ptr()[i]);
        }
        for (std::size_t i = 0; i < ref_csc.row_idx().size(); ++i) {
            KAVACH_ASSERT_EQUAL(cur_csc.row_idx()[i], ref_csc.row_idx()[i]);
            KAVACH_ASSERT_NEAR(cur_csc.values()[i], ref_csc.values()[i], 1e-15);
        }
    }
}

// ============================================================================
// 21. Requirement P: Sparse Scaling Integration & Dynamic Range Reduction
// ============================================================================

KAVACH_TEST(test_req_p_sparse_scaling_verification) {
    // 3x3 matrix with unbalanced rows/cols:
    // Row 0: values ~ 1e4
    // Row 1: values ~ 1e-2
    // Row 2: values ~ 1.0
    std::vector<Triplet> triplets = {{0, 0, 1e4},  {0, 1, 2e4}, {1, 1, 3e-2},
                                     {1, 2, 4e-2}, {2, 0, 5.0}, {2, 2, 6.0}};
    CsrMatrix A = CsrMatrix::from_triplets(3, 3, triplets);

    auto diag_orig = analyze_conditioning(A);
    KAVACH_ASSERT(diag_orig.dynamic_range_ratio > 1e5);
    KAVACH_ASSERT(diag_orig.recommend_scaling);

    auto factors = compute_geometric_mean_scaling(A, 5);
    CsrMatrix A_scaled = apply_scaling(A, factors);

    // Verify mathematical relation: A'_ij = r_i * A_ij * c_j
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            double orig_val = A.get(r, c);
            double scaled_val = A_scaled.get(r, c);
            double expected = factors.row_scale[r] * orig_val * factors.col_scale[c];
            KAVACH_ASSERT_NEAR(scaled_val, expected, 1e-12);
        }
    }

    auto diag_scaled = analyze_conditioning(A_scaled);
    // Dynamic range should be drastically reduced
    KAVACH_ASSERT(diag_scaled.dynamic_range_ratio < diag_orig.dynamic_range_ratio);
    KAVACH_ASSERT(diag_scaled.dynamic_range_ratio < 100.0);

    // Verify RHS & bound scaling consistency
    std::vector<double> rhs = {1e4, 1e-2, 5.0};
    auto rhs_scaled = factors.scale_rhs(rhs);
    for (std::size_t r = 0; r < 3; ++r) {
        KAVACH_ASSERT_NEAR(rhs_scaled[r], rhs[r] * factors.row_scale[r], 1e-12);
    }
}

// ============================================================================
// 22. Scale-Oriented Test: 5,000 x 5,000 Sparse Matrix Performance & Subquadratic Memory
// ============================================================================

KAVACH_TEST(test_scale_sparse_5000x5000) {
    const std::size_t N = 5000;
    // Pentadiagonal structure: diagonals at -2, -1, 0, +1, +2
    // Expected nonzeros: 5*N - 6 = 24,994 entries. Density = 24994 / 25000000 = ~0.1%
    std::vector<Triplet> triplets;
    triplets.reserve(5 * N);

    for (std::size_t i = 0; i < N; ++i) {
        triplets.push_back({i, i, 4.0});
        if (i >= 1) {
            triplets.push_back({i, i - 1, -1.0});
        }
        if (i + 1 < N) {
            triplets.push_back({i, i + 1, -1.0});
        }
        if (i >= 2) {
            triplets.push_back({i, i - 2, 0.5});
        }
        if (i + 2 < N) {
            triplets.push_back({i, i + 2, 0.5});
        }
    }

    auto t_start = std::chrono::steady_clock::now();
    CsrMatrix csr = CsrMatrix::from_triplets(N, N, triplets);
    auto t_built_csr = std::chrono::steady_clock::now();
    CscMatrix csc = csr_to_csc(csr);
    auto t_converted_csc = std::chrono::steady_clock::now();

    KAVACH_ASSERT(csr.is_valid());
    KAVACH_ASSERT(csc.is_valid());
    KAVACH_ASSERT_EQUAL(csr.rows(), N);
    KAVACH_ASSERT_EQUAL(csr.cols(), N);
    KAVACH_ASSERT_EQUAL(csc.rows(), N);
    KAVACH_ASSERT_EQUAL(csc.cols(), N);
    KAVACH_ASSERT_EQUAL(csr.nonzeros(), 5 * N - 6);
    KAVACH_ASSERT_EQUAL(csc.nonzeros(), 5 * N - 6);

    // SpMV with vector x = [1.0, 1.0, ..., 1.0]
    std::vector<double> x(N, 1.0);
    auto t_spmv_start = std::chrono::steady_clock::now();
    auto y_csr = csr.spmv(x);
    auto t_spmv_csr = std::chrono::steady_clock::now();
    auto y_csc = csc.spmv(x);
    auto t_spmv_csc = std::chrono::steady_clock::now();

    // Verify SpMV analytically:
    // For interior row (2 <= i <= N-3): 4.0 - 1.0 - 1.0 + 0.5 + 0.5 = 3.0
    // For i=0: 4.0 - 1.0 + 0.5 = 3.5
    // For i=1: 4.0 - 1.0 - 1.0 + 0.5 = 2.5
    // For i=N-2: 4.0 - 1.0 - 1.0 + 0.5 = 2.5
    // For i=N-1: 4.0 - 1.0 + 0.5 = 3.5
    double max_abs_err = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        double expected = 3.0;
        if (i == 0 || i == N - 1) {
            expected = 3.5;
        } else if (i == 1 || i == N - 2) {
            expected = 2.5;
        }

        double err_csr = std::abs(y_csr[i] - expected);
        double err_csc = std::abs(y_csc[i] - expected);
        if (err_csr > max_abs_err) {
            max_abs_err = err_csr;
        }
        if (err_csc > max_abs_err) {
            max_abs_err = err_csc;
        }
    }
    KAVACH_ASSERT(max_abs_err < 1e-14);

    // SpMV Transpose (symmetric pentadiagonal, so y_transpose == y)
    auto yt_csr = csr.spmv_transpose(x);
    auto yt_csc = csc.spmv_transpose(x);
    for (std::size_t i = 0; i < N; ++i) {
        KAVACH_ASSERT_NEAR(yt_csr[i], y_csr[i], 1e-14);
        KAVACH_ASSERT_NEAR(yt_csc[i], y_csc[i], 1e-14);
    }

    // Memory computation
    // Dense storage: 5000 x 5000 x 8 bytes = 200,000,000 bytes (~200 MB)
    std::size_t dense_bytes = N * N * sizeof(double);
    // Sparse CSR storage: (N + 1)*sizeof(size_t) + nnz*sizeof(size_t) + nnz*sizeof(double)
    std::size_t sparse_csr_bytes =
        (csr.row_ptr().size() + csr.col_idx().size()) * sizeof(std::size_t) +
        csr.values().size() * sizeof(double);
    double mem_ratio_percent =
        100.0 * static_cast<double>(sparse_csr_bytes) / static_cast<double>(dense_bytes);

    // Sparse representation must consume < 0.5% of dense matrix memory
    KAVACH_ASSERT(mem_ratio_percent < 0.5);

    auto ms_build =
        std::chrono::duration_cast<std::chrono::microseconds>(t_built_csr - t_start).count();
    auto ms_conv =
        std::chrono::duration_cast<std::chrono::microseconds>(t_converted_csc - t_built_csr)
            .count();
    auto us_spmv_csr =
        std::chrono::duration_cast<std::chrono::microseconds>(t_spmv_csr - t_spmv_start).count();
    auto us_spmv_csc =
        std::chrono::duration_cast<std::chrono::microseconds>(t_spmv_csc - t_spmv_csr).count();

    // Log diagnostic metrics for reporting
    std::cout << "[SCALE TEST 5000x5000] CSR Build: " << ms_build << " us, CSR->CSC: " << ms_conv
              << " us, CSR SpMV: " << us_spmv_csr << " us, CSC SpMV: " << us_spmv_csc << " us\n"
              << "Dense memory: " << (dense_bytes / 1024 / 1024)
              << " MB, Sparse memory: " << (sparse_csr_bytes / 1024)
              << " KB (Ratio: " << mem_ratio_percent << "%)\n"
              << "Max SpMV Absolute Error: " << max_abs_err << "\n";
}
