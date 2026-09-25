#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "numerics/sparse_matrix.hpp"

namespace kavach::numerics {

/**
 * @brief Base interface for Sparse LU Factorization.
 */
class SparseLUFactorization {
public:
    virtual ~SparseLUFactorization() = default;

    virtual bool factorize(const SparseMatrix& matrix, double pivot_tol) = 0;
    virtual bool solve(const std::vector<double>& rhs, std::vector<double>& solution) const = 0;
    virtual bool solve_transpose(const std::vector<double>& rhs,
                                 std::vector<double>& solution) const = 0;
    virtual void refactorize() = 0;
};

/**
 * @brief Outcome of iterative refinement.
 */
struct RefinementResult {
    bool converged{false};
    std::size_t iterations{0};
    double initial_residual{0.0};
    double final_residual{0.0};
};

/**
 * @brief Production Sparse LU Factorization with Partial Pivoting: P * A = L * U.
 *
 * Implements:
 * - Markowitz / partial pivoting with configurable pivot_tol
 * - Singularity / near-singularity detection returning SolverStatus::NUMERICAL_FAILURE
 * - Forward and backward triangular solves (direct and transpose)
 * - Residual computation and bounded iterative refinement
 * - Zero external solver dependencies (no Eigen, SuiteSparse, UMFPACK, etc.)
 * - Memory footprint O(n + nnz)
 */
class SparseLU : public SparseLUFactorization {
public:
    SparseLU() = default;

    // --- Factorization ---
    bool factorize(const CsrMatrix& matrix, double pivot_tol = core::SolverTolerances{}.pivot_tol);

    bool factorize(const SparseMatrix& matrix, double pivot_tol) override;

    void refactorize() override;

    // --- Solves ---
    bool solve(const std::vector<double>& rhs, std::vector<double>& solution) const override;
    [[nodiscard]] std::vector<double> solve(const std::vector<double>& rhs) const;

    bool solve_transpose(const std::vector<double>& rhs,
                         std::vector<double>& solution) const override;
    [[nodiscard]] std::vector<double> solve_transpose(const std::vector<double>& rhs) const;

    // --- Iterative Refinement ---
    [[nodiscard]] RefinementResult refine_solution(
        const CsrMatrix& A, const std::vector<double>& b, std::vector<double>& x,
        std::size_t max_iterations = 10,
        double residual_tol = core::SolverTolerances{}.residual_tol) const;

    // --- Accessors & Telemetry ---
    [[nodiscard]] bool is_factorized() const noexcept {
        return status_ == core::SolverStatus::OPTIMAL;
    }
    [[nodiscard]] std::size_t dimension() const noexcept { return n_; }
    [[nodiscard]] core::SolverStatus status() const noexcept { return status_; }
    [[nodiscard]] const std::string& error_message() const noexcept { return error_msg_; }

    [[nodiscard]] const CsrMatrix& L() const noexcept { return L_; }
    [[nodiscard]] const CsrMatrix& U() const noexcept { return U_; }
    [[nodiscard]] const std::vector<std::size_t>& row_permutation() const noexcept {
        return perm_r_;
    }
    [[nodiscard]] double min_pivot() const noexcept { return min_pivot_; }
    [[nodiscard]] double max_pivot() const noexcept { return max_pivot_; }
    [[nodiscard]] double condition_estimate() const noexcept;

private:
    std::size_t n_{0};
    core::SolverStatus status_{core::SolverStatus::UNKNOWN};
    std::string error_msg_;
    double pivot_tol_{1e-9};

    CsrMatrix L_;
    CsrMatrix U_;
    std::vector<std::size_t> perm_r_;
    std::vector<std::size_t> inv_perm_r_;

    double min_pivot_{0.0};
    double max_pivot_{0.0};

    // Stored matrix for refactorization
    CsrMatrix original_matrix_;
};

// --- Numerical Residual Helpers ---
[[nodiscard]] std::vector<double> compute_residual(const CsrMatrix& A, const std::vector<double>& x,
                                                   const std::vector<double>& b);

[[nodiscard]] double residual_norm_inf(const std::vector<double>& r) noexcept;

[[nodiscard]] double relative_residual_norm(const CsrMatrix& A, const std::vector<double>& x,
                                            const std::vector<double>& b);

}  // namespace kavach::numerics
