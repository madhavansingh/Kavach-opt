#include "algorithms/lp/standard_form.hpp"

#include <algorithm>
#include <cmath>

#include "core/error.hpp"
#include "numerics/scaling.hpp"

namespace kavach::algorithms::lp {

namespace {
constexpr double INFINITY_THRESHOLD = 1e30;
}

StandardFormLP::StandardFormLP(const model::CanonicalModel& model,
                               const core::SolverTolerances& tolerances,
                               bool preserve_bounds_structure)
    : num_original_vars_(model.num_variables()),
      num_original_constrs_(model.num_constraints()),
      original_sense_(model.objective().sense),
      original_offset_(model.objective().offset) {
    // 1. Capture original objective coefficients
    original_obj_coeffs_.assign(num_original_vars_, 0.0);
    for (const auto& term : model.objective().linear_terms.terms()) {
        if (term.var_id < num_original_vars_) {
            original_obj_coeffs_[term.var_id] = term.coefficient;
        }
    }

    const double sigma = (original_sense_ == model::ObjectiveSense::MAXIMIZE) ? -1.0 : 1.0;
    obj_offset_ = original_offset_ * sigma;

    // 2. Classify variables and map them into structural working variables
    var_transforms_.resize(num_original_vars_);
    std::size_t next_z = 0;
    std::size_t num_boxed = 0;

    for (std::size_t j = 0; j < num_original_vars_; ++j) {
        const auto& var = model.variable(static_cast<model::VarId>(j));
        double lb = var.lower_bound;
        double ub = var.upper_bound;

        if (!preserve_bounds_structure && std::abs(ub - lb) <= tolerances.primal_feasibility_tol) {
            // Fixed variable
            var_transforms_[j].type = VariableTransformType::FIXED;
            var_transforms_[j].shift = lb;
        } else if (lb > -INFINITY_THRESHOLD && ub < INFINITY_THRESHOLD) {
            // Boxed (preserved even if ub == lb when preserve_bounds_structure is true)
            var_transforms_[j].type = VariableTransformType::BOXED;
            var_transforms_[j].shift = lb;
            var_transforms_[j].z_index1 = next_z++;
            ++num_boxed;
        } else if (lb > -INFINITY_THRESHOLD && ub >= INFINITY_THRESHOLD) {
            // Lower-bounded only
            var_transforms_[j].type = VariableTransformType::SHIFTED_LOWER;
            var_transforms_[j].shift = lb;
            var_transforms_[j].z_index1 = next_z++;
        } else if (lb <= -INFINITY_THRESHOLD && ub < INFINITY_THRESHOLD) {
            // Upper-bounded only
            var_transforms_[j].type = VariableTransformType::SHIFTED_UPPER;
            var_transforms_[j].shift = ub;
            var_transforms_[j].z_index1 = next_z++;
        } else {
            // Free variable
            var_transforms_[j].type = VariableTransformType::FREE;
            var_transforms_[j].shift = 0.0;
            var_transforms_[j].z_index1 = next_z++;
            var_transforms_[j].z_index2 = next_z++;
        }
    }

    // Initialize c with structural variable costs
    c_.assign(next_z, 0.0);
    for (std::size_t j = 0; j < num_original_vars_; ++j) {
        const auto& trans = var_transforms_[j];
        double c_orig = original_obj_coeffs_[j] * sigma;
        switch (trans.type) {
            case VariableTransformType::FIXED:
                obj_offset_ += c_orig * trans.shift;
                break;
            case VariableTransformType::SHIFTED_LOWER:
            case VariableTransformType::BOXED:
                obj_offset_ += c_orig * trans.shift;
                c_[trans.z_index1] += c_orig;
                break;
            case VariableTransformType::SHIFTED_UPPER:
                obj_offset_ += c_orig * trans.shift;
                c_[trans.z_index1] -= c_orig;
                break;
            case VariableTransformType::FREE:
                c_[trans.z_index1] += c_orig;
                c_[trans.z_index2] -= c_orig;
                break;
        }
    }

    // 3. Process original constraints and allocate slack/surplus/artificial variables
    num_rows_ = num_original_constrs_ + num_boxed;
    b_.assign(num_rows_, 0.0);
    initial_basis_.assign(num_rows_, 0);
    row_transforms_.resize(num_original_constrs_);

    std::vector<numerics::Triplet> triplets;
    std::size_t next_col = next_z;

    for (std::size_t i = 0; i < num_original_constrs_; ++i) {
        const auto& constr = model.constraint(static_cast<model::ConstrId>(i));
        double rhs_shift = 0.0;

        struct RowTerm {
            std::size_t col;
            double coeff;
        };
        std::vector<RowTerm> row_terms;

        for (const auto& term : constr.expression.terms()) {
            if (term.var_id >= num_original_vars_) {
                continue;
            }
            const auto& trans = var_transforms_[term.var_id];
            double a = term.coefficient;

            switch (trans.type) {
                case VariableTransformType::FIXED:
                    rhs_shift += a * trans.shift;
                    break;
                case VariableTransformType::SHIFTED_LOWER:
                case VariableTransformType::BOXED:
                    rhs_shift += a * trans.shift;
                    row_terms.push_back({trans.z_index1, a});
                    break;
                case VariableTransformType::SHIFTED_UPPER:
                    rhs_shift += a * trans.shift;
                    row_terms.push_back({trans.z_index1, -a});
                    break;
                case VariableTransformType::FREE:
                    row_terms.push_back({trans.z_index1, a});
                    row_terms.push_back({trans.z_index2, -a});
                    break;
            }
        }

        double eff_rhs = constr.rhs - rhs_shift;
        auto& rtrans = row_transforms_[i];
        rtrans.original_row_id = i;

        if (constr.sense == model::ConstraintSense::LESS_EQUAL) {
            if (preserve_bounds_structure || eff_rhs >= -tolerances.primal_feasibility_tol) {
                // a^T z + s = b  (s >= 0)
                b_[i] = eff_rhs;
                for (const auto& rt : row_terms) {
                    triplets.push_back({i, rt.col, rt.coeff});
                }
                std::size_t s_col = next_col++;
                triplets.push_back({i, s_col, 1.0});
                rtrans.has_slack_or_surplus = true;
                rtrans.is_surplus = false;
                rtrans.slack_or_surplus_var = s_col;
                initial_basis_[i] = s_col;
            } else {
                // Negate: -a^T z - s + w = -b
                b_[i] = -eff_rhs;
                rtrans.negated = true;
                for (const auto& rt : row_terms) {
                    triplets.push_back({i, rt.col, -rt.coeff});
                }
                std::size_t s_col = next_col++;
                triplets.push_back({i, s_col, -1.0});
                rtrans.has_slack_or_surplus = true;
                rtrans.is_surplus = true;
                rtrans.slack_or_surplus_var = s_col;

                std::size_t w_col = next_col++;
                triplets.push_back({i, w_col, 1.0});
                rtrans.has_artificial = true;
                rtrans.artificial_var = w_col;
                artificial_indices_.push_back(w_col);
                initial_basis_[i] = w_col;
            }
        } else if (constr.sense == model::ConstraintSense::GREATER_EQUAL) {
            if (eff_rhs >= -tolerances.primal_feasibility_tol) {
                // a^T z - s + w = b
                b_[i] = std::max(0.0, eff_rhs);
                for (const auto& rt : row_terms) {
                    triplets.push_back({i, rt.col, rt.coeff});
                }
                std::size_t s_col = next_col++;
                triplets.push_back({i, s_col, -1.0});
                rtrans.has_slack_or_surplus = true;
                rtrans.is_surplus = true;
                rtrans.slack_or_surplus_var = s_col;

                std::size_t w_col = next_col++;
                triplets.push_back({i, w_col, 1.0});
                rtrans.has_artificial = true;
                rtrans.artificial_var = w_col;
                artificial_indices_.push_back(w_col);
                initial_basis_[i] = w_col;
            } else {
                // Negate: -a^T z + s = -b
                b_[i] = -eff_rhs;
                rtrans.negated = true;
                for (const auto& rt : row_terms) {
                    triplets.push_back({i, rt.col, -rt.coeff});
                }
                std::size_t s_col = next_col++;
                triplets.push_back({i, s_col, 1.0});
                rtrans.has_slack_or_surplus = true;
                rtrans.is_surplus = false;
                rtrans.slack_or_surplus_var = s_col;
                initial_basis_[i] = s_col;
            }
        } else {
            // EQUAL
            if (eff_rhs >= -tolerances.primal_feasibility_tol) {
                b_[i] = std::max(0.0, eff_rhs);
                for (const auto& rt : row_terms) {
                    triplets.push_back({i, rt.col, rt.coeff});
                }
            } else {
                b_[i] = -eff_rhs;
                rtrans.negated = true;
                for (const auto& rt : row_terms) {
                    triplets.push_back({i, rt.col, -rt.coeff});
                }
            }
            std::size_t w_col = next_col++;
            triplets.push_back({i, w_col, 1.0});
            rtrans.has_artificial = true;
            rtrans.artificial_var = w_col;
            artificial_indices_.push_back(w_col);
            initial_basis_[i] = w_col;
        }
    }

    // 4. Add upper bound rows for BOXED variables: z_k + s_ub = u_j - l_j
    std::size_t current_row = num_original_constrs_;
    for (std::size_t j = 0; j < num_original_vars_; ++j) {
        auto& trans = var_transforms_[j];
        if (trans.type == VariableTransformType::BOXED) {
            const auto& var = model.variable(static_cast<model::VarId>(j));
            double diff = var.upper_bound - var.lower_bound;
            b_[current_row] = std::max(0.0, diff);

            triplets.push_back({current_row, trans.z_index1, 1.0});
            std::size_t s_ub = next_col++;
            triplets.push_back({current_row, s_ub, 1.0});

            trans.ub_row_index = current_row;
            trans.ub_slack_index = s_ub;
            initial_basis_[current_row] = s_ub;
            ++current_row;
        }
    }

    num_cols_ = next_col;
    c_.resize(num_cols_, 0.0);  // Slacks, surplus, and artificials have cost 0 in Phase II

    is_artificial_.assign(num_cols_, false);
    for (std::size_t w : artificial_indices_) {
        is_artificial_[w] = true;
    }

    // Build sparse matrices
    A_csr_ = numerics::CsrMatrix::from_triplets(num_rows_, num_cols_, triplets);
    A_csc_ = numerics::csr_to_csc(A_csr_);
}

std::vector<double> StandardFormLP::reconstruct_primal(const std::vector<double>& z) const {
    std::vector<double> x(num_original_vars_, 0.0);
    for (std::size_t j = 0; j < num_original_vars_; ++j) {
        const auto& trans = var_transforms_[j];
        switch (trans.type) {
            case VariableTransformType::FIXED:
                x[j] = trans.shift;
                break;
            case VariableTransformType::SHIFTED_LOWER:
            case VariableTransformType::BOXED:
                x[j] = trans.shift + (trans.z_index1 < z.size() ? z[trans.z_index1] : 0.0);
                break;
            case VariableTransformType::SHIFTED_UPPER:
                x[j] = trans.shift - (trans.z_index1 < z.size() ? z[trans.z_index1] : 0.0);
                break;
            case VariableTransformType::FREE:
                double z1 = trans.z_index1 < z.size() ? z[trans.z_index1] : 0.0;
                double z2 = trans.z_index2 < z.size() ? z[trans.z_index2] : 0.0;
                x[j] = z1 - z2;
                break;
        }
    }
    return x;
}

std::vector<double> StandardFormLP::reconstruct_dual(const std::vector<double>& y) const {
    std::vector<double> dual(num_original_constrs_, 0.0);
    for (std::size_t i = 0; i < num_original_constrs_; ++i) {
        if (i < y.size()) {
            double y_val = y[i];
            if (row_transforms_[i].negated) {
                y_val = -y_val;
            }
            if (original_sense_ == model::ObjectiveSense::MAXIMIZE) {
                y_val = -y_val;
            }
            dual[i] = y_val;
        }
    }
    return dual;
}

std::vector<double> StandardFormLP::reconstruct_reduced_costs(const std::vector<double>& d) const {
    std::vector<double> rc(num_original_vars_, 0.0);
    for (std::size_t j = 0; j < num_original_vars_; ++j) {
        const auto& trans = var_transforms_[j];
        switch (trans.type) {
            case VariableTransformType::FIXED:
                rc[j] = 0.0;
                break;
            case VariableTransformType::SHIFTED_LOWER:
            case VariableTransformType::BOXED:
                rc[j] = trans.z_index1 < d.size() ? d[trans.z_index1] : 0.0;
                break;
            case VariableTransformType::SHIFTED_UPPER:
                rc[j] = trans.z_index1 < d.size() ? -d[trans.z_index1] : 0.0;
                break;
            case VariableTransformType::FREE:
                rc[j] = trans.z_index1 < d.size() ? d[trans.z_index1] : 0.0;
                break;
        }
        if (original_sense_ == model::ObjectiveSense::MAXIMIZE) {
            rc[j] = -rc[j];
        }
    }
    return rc;
}

double StandardFormLP::compute_original_objective(const std::vector<double>& x) const {
    double obj = original_offset_;
    for (std::size_t j = 0; j < num_original_vars_; ++j) {
        if (j < x.size()) {
            obj += original_obj_coeffs_[j] * x[j];
        }
    }
    return obj;
}

void StandardFormLP::apply_scaling(const numerics::ScalingFactors& factors) {
    A_csr_ = numerics::apply_scaling(A_csr_, factors);
    A_csc_ = numerics::apply_scaling(A_csc_, factors);
    b_ = factors.scale_rhs(b_);
    c_ = factors.scale_objective(c_);
}

}  // namespace kavach::algorithms::lp
