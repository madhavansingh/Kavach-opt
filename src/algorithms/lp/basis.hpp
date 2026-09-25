#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "algorithms/lp/lp_types.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "numerics/lu_factorization.hpp"
#include "numerics/sparse_matrix.hpp"

namespace kavach::algorithms::lp {

/**
 * @brief Manages the revised simplex basis matrix B, its SparseLU factorization,
 * and column operations.
 */
class SimplexBasis {
public:
    SimplexBasis() = default;

    /**
     * @brief Initialize basis with a given set of m basic variable indices.
     */
    void initialize(std::vector<std::size_t> initial_basis, std::size_t total_cols);

    /**
     * @brief Extract columns from A_csc corresponding to current basis and factorize B using
     * SparseLU.
     * @return true if factorization succeeded, false on numerical failure or singularity.
     */
    bool factorize(const numerics::CscMatrix& A_csc, double pivot_tol);

    /**
     * @brief Solve B * x_B = rhs.
     */
    [[nodiscard]] std::vector<double> solve_primal(const std::vector<double>& rhs) const;

    /**
     * @brief Solve B^T * y = c_B (simplex multipliers).
     */
    [[nodiscard]] std::vector<double> solve_dual(const std::vector<double>& c_B) const;

    /**
     * @brief Solve B * d = A[:, entering_col] (entering direction).
     */
    [[nodiscard]] std::vector<double> solve_direction(const numerics::CscMatrix& A_csc,
                                                      std::size_t entering_col) const;

    /**
     * @brief Perform basis swap: variable entering_col enters at row leaving_row.
     */
    void update_pivot(std::size_t leaving_row, std::size_t entering_col);

    /**
     * @brief Check whether basis needs periodic refactorization.
     */
    [[nodiscard]] bool needs_refactorization(std::size_t interval) const noexcept {
        return updates_since_refactor_ >= interval;
    }

    // --- Accessors ---
    [[nodiscard]] std::size_t num_rows() const noexcept { return basis_vars_.size(); }
    [[nodiscard]] const std::vector<std::size_t>& basis_vars() const noexcept {
        return basis_vars_;
    }
    [[nodiscard]] const std::vector<std::size_t>& nonbasic_vars() const noexcept {
        return nonbasic_vars_;
    }
    [[nodiscard]] bool is_basic(std::size_t col_idx) const noexcept {
        return col_idx < is_basic_.size() && is_basic_[col_idx];
    }
    [[nodiscard]] std::size_t updates_since_refactor() const noexcept {
        return updates_since_refactor_;
    }
    [[nodiscard]] double condition_estimate() const noexcept { return lu_.condition_estimate(); }
    [[nodiscard]] core::SolverStatus lu_status() const noexcept { return lu_.status(); }
    [[nodiscard]] const std::string& error_message() const noexcept { return lu_.error_message(); }

    /**
     * @brief Remove column from nonbasic set (e.g. purging artificial variables).
     */
    void purge_nonbasic_variable(std::size_t var_idx);

    /**
     * @brief Export current basis state for warm starting.
     */
    [[nodiscard]] BasisState export_state() const;

    /**
     * @brief Load basis state from an existing export.
     */
    void load_state(const BasisState& state);

private:
    std::vector<std::size_t> basis_vars_;     ///< Basic variable column index for each row (size m)
    std::vector<bool> is_basic_;              ///< Indicator for all variables (size n)
    std::vector<std::size_t> nonbasic_vars_;  ///< List of nonbasic variable indices
    numerics::SparseLU lu_;                   ///< Sparse LU factorization of B
    std::size_t updates_since_refactor_{0};
};

}  // namespace kavach::algorithms::lp
