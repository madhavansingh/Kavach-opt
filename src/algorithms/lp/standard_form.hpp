#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "algorithms/lp/lp_types.hpp"
#include "model/model.hpp"
#include "numerics/scaling.hpp"
#include "numerics/sparse_matrix.hpp"

namespace kavach::algorithms::lp {

enum class VariableTransformType {
    FIXED,          ///< Fixed variable: x = l_j
    SHIFTED_LOWER,  ///< Lower bounded: x = l_j + z_k
    BOXED,          ///< Boxed: x = l_j + z_k, and z_k + s_ub = u_j - l_j
    SHIFTED_UPPER,  ///< Upper bounded only: x = u_j - z_k
    FREE            ///< Free variable: x = z_k1 - z_k2
};

struct VariableTransform {
    VariableTransformType type{VariableTransformType::SHIFTED_LOWER};
    double shift{0.0};
    std::size_t z_index1{0};
    std::size_t z_index2{0};        ///< For FREE variables (z_k2)
    std::size_t ub_row_index{0};    ///< For BOXED variables
    std::size_t ub_slack_index{0};  ///< For BOXED variables
};

struct RowTransform {
    std::size_t original_row_id{0};
    bool negated{false};
    std::size_t slack_or_surplus_var{0};
    bool has_slack_or_surplus{false};
    bool is_surplus{false};
    std::size_t artificial_var{0};
    bool has_artificial{false};
};

/**
 * @brief Standard-form linear program representation:
 *        minimize c^T z  subject to A z = b, z >= 0, b >= 0
 */
class StandardFormLP {
public:
    StandardFormLP(const model::CanonicalModel& model, const core::SolverTolerances& tolerances,
                   bool preserve_bounds_structure = false);

    // --- Accessors ---
    [[nodiscard]] std::size_t num_rows() const noexcept { return num_rows_; }
    [[nodiscard]] std::size_t num_cols() const noexcept { return num_cols_; }
    [[nodiscard]] std::size_t num_artificials() const noexcept {
        return artificial_indices_.size();
    }
    [[nodiscard]] bool has_artificials() const noexcept { return !artificial_indices_.empty(); }

    [[nodiscard]] const numerics::CsrMatrix& A_csr() const noexcept { return A_csr_; }
    [[nodiscard]] const numerics::CscMatrix& A_csc() const noexcept { return A_csc_; }
    [[nodiscard]] const std::vector<double>& b() const noexcept { return b_; }
    [[nodiscard]] const std::vector<double>& c() const noexcept { return c_; }
    [[nodiscard]] double objective_offset() const noexcept { return obj_offset_; }
    [[nodiscard]] model::ObjectiveSense original_sense() const noexcept { return original_sense_; }

    [[nodiscard]] const std::vector<std::size_t>& initial_basis() const noexcept {
        return initial_basis_;
    }
    [[nodiscard]] const std::vector<std::size_t>& artificial_indices() const noexcept {
        return artificial_indices_;
    }
    [[nodiscard]] bool is_artificial(std::size_t col_idx) const noexcept {
        return col_idx < is_artificial_.size() && is_artificial_[col_idx];
    }

    // --- Solution Reconstruction ---
    [[nodiscard]] std::vector<double> reconstruct_primal(const std::vector<double>& z) const;
    [[nodiscard]] std::vector<double> reconstruct_dual(const std::vector<double>& y) const;
    [[nodiscard]] std::vector<double> reconstruct_reduced_costs(const std::vector<double>& d) const;
    [[nodiscard]] double compute_original_objective(const std::vector<double>& x) const;

    // --- Transforms & Substitution ---
    [[nodiscard]] const std::vector<VariableTransform>& var_transforms() const noexcept {
        return var_transforms_;
    }
    [[nodiscard]] const std::vector<RowTransform>& row_transforms() const noexcept {
        return row_transforms_;
    }
    [[nodiscard]] std::size_t num_original_vars() const noexcept { return num_original_vars_; }
    [[nodiscard]] std::size_t num_original_constrs() const noexcept {
        return num_original_constrs_;
    }

    /**
     * @brief Translate a linear inequality in standard-form variables z:
     *        \sum_k gamma_k * z_k >= delta
     *        into an equivalent linear inequality on original model variables x:
     *        \sum_j alpha_j * x_j <= beta (or >= beta).
     */
    void substitute_standard_to_model(const model::CanonicalModel& model,
                                      const std::vector<double>& gamma, double delta,
                                      std::vector<double>& out_alpha, double& out_beta,
                                      bool as_less_equal = true) const;

    // Mutator for scaling support
    void apply_scaling(const numerics::ScalingFactors& factors);

private:
    std::size_t num_original_vars_{0};
    std::size_t num_original_constrs_{0};
    model::ObjectiveSense original_sense_{model::ObjectiveSense::MINIMIZE};
    double original_offset_{0.0};
    std::vector<double> original_obj_coeffs_;

    std::size_t num_rows_{0};
    std::size_t num_cols_{0};
    double obj_offset_{0.0};

    numerics::CsrMatrix A_csr_;
    numerics::CscMatrix A_csc_;
    std::vector<double> b_;
    std::vector<double> c_;  ///< Phase II minimization objective

    std::vector<std::size_t> initial_basis_;
    std::vector<std::size_t> artificial_indices_;
    std::vector<bool> is_artificial_;

    std::vector<VariableTransform> var_transforms_;
    std::vector<RowTransform> row_transforms_;
};

}  // namespace kavach::algorithms::lp
