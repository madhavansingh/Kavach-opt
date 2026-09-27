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
 * @brief Sparse elementary column update (Eta matrix) representing a single simplex pivot:
 * B_{k+1} = B_k * E_k, where E_k replaces column p (pivot_row) of I with d = B_k^{-1} A[:,
 * entering_col].
 */
struct SparseEtaFactor {
    std::size_t base_generation{0};          ///< Generation ID of the base SparseLU factorization
    std::size_t pivot_row{0};                ///< Leaving row index p in [0, m-1]
    std::size_t entering_col{0};             ///< Entering column index j
    std::size_t leaving_col{0};              ///< Leaving column index
    double pivot_elem{1.0};                  ///< Diagonal entry d[p]
    std::vector<std::size_t> off_diag_rows;  ///< Row indices i != p where |d[i]| > 1e-15
    std::vector<double> off_diag_vals;       ///< Corresponding values d[i]

    [[nodiscard]] std::size_t byte_footprint() const noexcept {
        return sizeof(SparseEtaFactor) +
               off_diag_rows.size() * (sizeof(std::size_t) + sizeof(double));
    }
};

/**
 * @brief Manages the revised simplex basis matrix B, its base SparseLU factorization,
 * and incremental sparse Product-Form-of-Inverse (Eta) updates.
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
     * SparseLU, clearing any incremental Eta update chain and advancing factorization generation.
     * @return true if factorization succeeded, false on numerical failure or singularity.
     */
    bool factorize(const numerics::CscMatrix& A_csc, double pivot_tol,
                   const std::string& reason = "INITIAL_FACTORIZATION");

    /**
     * @brief Solve B_k * x_B = rhs using base SparseLU followed by forward Eta chain.
     */
    [[nodiscard]] std::vector<double> solve_primal(const std::vector<double>& rhs) const;

    /**
     * @brief Solve B_k^T * y = c_B using reverse Eta chain followed by base SparseLU transpose.
     */
    [[nodiscard]] std::vector<double> solve_dual(const std::vector<double>& c_B) const;

    /**
     * @brief Solve B_k * d = A[:, entering_col] (entering direction in current basis coordinates).
     */
    [[nodiscard]] std::vector<double> solve_direction(const numerics::CscMatrix& A_csc,
                                                      std::size_t entering_col) const;

    /**
     * @brief Perform combinatorial basis swap only (invalidates current incremental factorization
     * until factorize() is called). Prefer apply_eta_update() during simplex iterations.
     */
    void update_pivot(std::size_t leaving_row, std::size_t entering_col);

    /**
     * @brief Perform basis swap AND append a sparse Eta update factor from direction d = B_k^{-1}
     * a_j. Validates pivot magnitude and relative stability. If unstable, performs the
     * combinatorial swap, marks factorization invalid, increments update_failures, and returns
     * false so caller refactorizes.
     */
    bool apply_eta_update(std::size_t leaving_row, std::size_t entering_col,
                          const std::vector<double>& d, double pivot_tol,
                          double min_rel_pivot = 1e-7);

    /**
     * @brief Compute true primal basis residual ||B * x_B - b||_inf directly from A_csc columns.
     */
    [[nodiscard]] double compute_primal_residual(const numerics::CscMatrix& A_csc,
                                                 const std::vector<double>& x_B,
                                                 const std::vector<double>& b) const;

    /**
     * @brief Compute true dual basis residual ||B^T * y - c_B||_inf directly from A_csc columns.
     */
    [[nodiscard]] double compute_dual_residual(const numerics::CscMatrix& A_csc,
                                               const std::vector<double>& y,
                                               const std::vector<double>& c_B) const;

    /**
     * @brief Check whether basis needs full refactorization due to chain length, invalidation,
     * or memory growth.
     */
    [[nodiscard]] bool needs_refactorization(std::size_t interval) const noexcept {
        if (!factorization_valid_) {
            return true;
        }
        if (updates_since_refactor_ >= interval || eta_chain_.size() >= interval) {
            return true;
        }
        if (base_factorization_bytes_ > 0 &&
            update_chain_bytes_ > 4 * base_factorization_bytes_ + 65536) {
            return true;
        }
        return false;
    }

    /**
     * @brief Determine diagnostic reason for a pending refactorization.
     */
    [[nodiscard]] std::string pending_refactorization_reason(std::size_t interval) const {
        if (!factorization_valid_) {
            return last_refactor_reason_.empty() ? "INVALIDATED_BASIS" : last_refactor_reason_;
        }
        if (base_factorization_bytes_ > 0 &&
            update_chain_bytes_ > 4 * base_factorization_bytes_ + 65536) {
            return "UPDATE_MEMORY_BOUND_EXCEEDED";
        }
        if (updates_since_refactor_ >= interval || eta_chain_.size() >= interval) {
            return "PERIODIC_CHAIN_INTERVAL";
        }
        return "DRIFT_OR_STABILITY_TRIGGER";
    }

    /**
     * @brief Accumulate basis-update and refactorization metrics into LPSolverTelemetry.
     */
    void accumulate_telemetry(LPSolverTelemetry& tel) const;

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
    [[nodiscard]] std::size_t update_chain_length() const noexcept { return eta_chain_.size(); }
    [[nodiscard]] std::size_t factorization_generation() const noexcept {
        return factorization_generation_;
    }
    [[nodiscard]] bool is_factorization_valid() const noexcept { return factorization_valid_; }
    [[nodiscard]] const std::vector<SparseEtaFactor>& eta_chain() const noexcept {
        return eta_chain_;
    }
    [[nodiscard]] std::size_t basis_updates_count() const noexcept { return basis_updates_; }
    [[nodiscard]] std::size_t full_refactorizations_count() const noexcept {
        return full_refactorizations_;
    }
    [[nodiscard]] std::size_t update_failures_count() const noexcept { return update_failures_; }
    [[nodiscard]] std::size_t base_factorization_bytes() const noexcept {
        return base_factorization_bytes_;
    }
    [[nodiscard]] std::size_t update_chain_bytes() const noexcept { return update_chain_bytes_; }
    [[nodiscard]] std::size_t peak_memory_bytes() const noexcept { return peak_memory_bytes_; }
    [[nodiscard]] std::size_t memory_after_refactor_bytes() const noexcept {
        return memory_after_refactor_bytes_;
    }
    [[nodiscard]] double max_observed_residual() const noexcept { return max_observed_residual_; }
    [[nodiscard]] const std::string& last_refactor_reason() const noexcept {
        return last_refactor_reason_;
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
    void update_pivot_combinatorial(std::size_t leaving_row, std::size_t entering_col);

    std::vector<std::size_t> basis_vars_;     ///< Basic variable column index for each row (size m)
    std::vector<bool> is_basic_;              ///< Indicator for all variables (size n)
    std::vector<std::size_t> nonbasic_vars_;  ///< List of nonbasic variable indices
    numerics::SparseLU lu_;                   ///< Base Sparse LU factorization of B_0
    std::vector<SparseEtaFactor> eta_chain_;  ///< Incremental sparse Eta factors E_0, ..., E_{k-1}

    std::size_t updates_since_refactor_{0};
    std::size_t factorization_generation_{0};
    bool factorization_valid_{false};

    // Telemetry counters
    std::size_t basis_updates_{0};
    std::size_t full_refactorizations_{0};
    std::size_t update_failures_{0};
    std::size_t maximum_update_chain_{0};
    std::size_t cumulative_chain_lengths_{0};
    std::size_t chain_length_samples_{0};
    std::string last_refactor_reason_{"INITIAL_FACTORIZATION"};
    double update_application_time_seconds_{0.0};
    double full_refactorization_time_seconds_{0.0};
    std::size_t base_factorization_bytes_{0};
    std::size_t update_chain_bytes_{0};
    std::size_t peak_memory_bytes_{0};
    std::size_t memory_after_refactor_bytes_{0};
    mutable double max_observed_residual_{0.0};
};

}  // namespace kavach::algorithms::lp
