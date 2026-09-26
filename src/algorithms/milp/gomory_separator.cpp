#include "algorithms/milp/gomory_separator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace kavach::algorithms::milp {

namespace {

double dot_sparse_col(const std::vector<double>& u, const numerics::CscMatrix& A_csc,
                      std::size_t col_idx) {
    const std::size_t nnz = A_csc.col_nnz(col_idx);
    const std::size_t* r_idx = A_csc.col_indices(col_idx);
    const double* vals = A_csc.col_values(col_idx);

    double sum = 0.0;
    for (std::size_t p = 0; p < nnz; ++p) {
        std::size_t r = r_idx[p];
        if (r < u.size()) {
            sum += u[r] * vals[p];
        }
    }
    return sum;
}

}  // anonymous namespace

GomorySeparator::GomorySeparator(const CutOptions& options) : options_(options) {}

std::vector<bool> GomorySeparator::classify_standard_form_integrality(
    const model::CanonicalModel& model, const lp::StandardFormLP& standard_lp) const {
    const std::size_t n_sf = standard_lp.num_cols();
    std::vector<bool> is_int(n_sf, false);

    const auto& var_transforms = standard_lp.var_transforms();
    const std::size_t n_orig = model.num_variables();

    for (std::size_t j = 0; j < n_orig; ++j) {
        const auto& var = model.variable(static_cast<model::VarId>(j));
        if (var.is_integer() || var.is_binary()) {
            if (j < var_transforms.size()) {
                const auto& trans = var_transforms[j];
                // Check if lower bound shift is integer
                double shift_err = std::abs(trans.shift - std::round(trans.shift));
                if (shift_err <= 1e-6) {
                    if (trans.z_index1 < n_sf) {
                        is_int[trans.z_index1] = true;
                    }
                    if (trans.type == lp::VariableTransformType::BOXED) {
                        double diff = var.upper_bound - var.lower_bound;
                        if (std::abs(diff - std::round(diff)) <= 1e-6) {
                            if (trans.ub_slack_index < n_sf) {
                                is_int[trans.ub_slack_index] = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // Check constraint slacks
    const auto& row_transforms = standard_lp.row_transforms();
    const std::size_t m_orig = model.num_constraints();
    for (std::size_t i = 0; i < m_orig; ++i) {
        const auto& constr = model.constraint(static_cast<model::ConstrId>(i));
        bool all_int_coeffs = true;
        for (const auto& term : constr.expression.terms()) {
            if (term.var_id < n_orig) {
                const auto& var = model.variable(term.var_id);
                if (!var.is_integer() && !var.is_binary()) {
                    all_int_coeffs = false;
                    break;
                }
                if (std::abs(term.coefficient - std::round(term.coefficient)) > 1e-6) {
                    all_int_coeffs = false;
                    break;
                }
            }
        }
        if (all_int_coeffs && std::abs(constr.rhs - std::round(constr.rhs)) <= 1e-6) {
            if (i < row_transforms.size() && row_transforms[i].has_slack_or_surplus) {
                std::size_t svar = row_transforms[i].slack_or_surplus_var;
                if (svar < n_sf) {
                    is_int[svar] = true;
                }
            }
        }
    }

    return is_int;
}

std::vector<LinearCut> GomorySeparator::separate(
    const model::CanonicalModel& model, const lp::StandardFormLP& standard_lp,
    const lp::SimplexBasis& basis, const std::vector<double>& x_lp, const std::vector<double>& z_lp,
    std::size_t originating_node, std::size_t current_round) const {
    std::vector<LinearCut> generated_cuts;

    if (!options_.enable_gomory) {
        return generated_cuts;
    }

    const std::size_t m_sf = standard_lp.num_rows();
    const std::size_t n_sf = standard_lp.num_cols();
    if (m_sf == 0 || n_sf == 0 || basis.num_rows() != m_sf) {
        return generated_cuts;
    }

    auto is_int_sf = classify_standard_form_integrality(model, standard_lp);

    // Loop through each basic variable row
    for (std::size_t i = 0; i < m_sf; ++i) {
        std::size_t basic_var = basis.basis_vars()[i];
        if (basic_var >= n_sf || !is_int_sf[basic_var]) {
            continue;  // Basic variable is not integer-constrained
        }

        // Get value of basic variable
        double x_val = (basic_var < z_lp.size()) ? z_lp[basic_var] : 0.0;
        double f0 = x_val - std::floor(x_val);

        // Numerically significant fractionality check
        if (f0 < options_.min_fractionality || f0 > 1.0 - options_.min_fractionality) {
            continue;
        }

        // Compute row of B^{-1}: solve B^T u = e_i
        std::vector<double> e_i(m_sf, 0.0);
        e_i[i] = 1.0;
        auto u = basis.solve_dual(e_i);

        // Sanity check: u must be finite
        bool u_finite = true;
        for (double val : u) {
            if (!std::isfinite(val)) {
                u_finite = false;
                break;
            }
        }
        if (!u_finite) {
            continue;
        }

        // Compute tableau row coefficients for nonbasic variables
        // gamma_k z_k >= f0
        std::vector<double> gamma(n_sf, 0.0);

        for (std::size_t k : basis.nonbasic_vars()) {
            if (k >= n_sf) {
                continue;
            }
            double a_bar = dot_sparse_col(u, standard_lp.A_csc(), k);
            if (std::abs(a_bar) < 1e-12) {
                continue;
            }

            if (is_int_sf[k]) {
                // Integer variable
                double fk = a_bar - std::floor(a_bar);
                if (fk < 1e-12) {
                    fk = 0.0;
                }
                if (fk <= f0) {
                    gamma[k] = fk;
                } else {
                    gamma[k] = (f0 * (1.0 - fk)) / (1.0 - f0);
                }
            } else {
                // Continuous variable
                if (a_bar > 0.0) {
                    gamma[k] = a_bar;
                } else {
                    gamma[k] = -(f0 / (1.0 - f0)) * a_bar;
                }
            }
        }

        // Substitute standard form variables back to original variables x:
        // sum_j alpha_j x_j <= beta
        std::vector<double> out_alpha;
        double out_beta = 0.0;
        standard_lp.substitute_standard_to_model(model, gamma, f0, out_alpha, out_beta,
                                                 /*as_less_equal=*/true);

        // Sparsity pruning & numerical validation
        std::vector<model::VarId> cut_var_ids;
        std::vector<double> cut_coeffs;
        double max_abs = 0.0;
        double min_abs = std::numeric_limits<double>::infinity();

        for (std::size_t j = 0; j < out_alpha.size(); ++j) {
            double c = out_alpha[j];
            if (std::abs(c) > 1e-12) {
                double abs_c = std::abs(c);
                max_abs = std::max(max_abs, abs_c);
                min_abs = std::min(min_abs, abs_c);
                cut_var_ids.push_back(static_cast<model::VarId>(j));
                cut_coeffs.push_back(c);
            }
        }

        if (cut_var_ids.empty() || !std::isfinite(out_beta)) {
            continue;
        }

        // Numerical safety checks
        if (max_abs > options_.max_absolute_coefficient) {
            continue;
        }
        if (min_abs > 0.0 && (max_abs / min_abs) > options_.max_coefficient_range) {
            continue;
        }

        // Check violation and efficacy at x_lp
        double act = 0.0;
        for (std::size_t k = 0; k < cut_var_ids.size(); ++k) {
            model::VarId vid = cut_var_ids[k];
            if (vid < x_lp.size()) {
                act += cut_coeffs[k] * x_lp[vid];
            }
        }
        double viol = std::max(0.0, act - out_beta);
        if (viol < options_.min_cut_violation) {
            continue;
        }

        double sum_sq = 0.0;
        for (double c : cut_coeffs) {
            sum_sq += c * c;
        }
        double norm2 = std::sqrt(sum_sq);
        if (norm2 <= 1e-12) {
            continue;
        }

        double efficacy = viol / norm2;
        if (efficacy < options_.min_cut_efficacy) {
            continue;
        }

        LinearCut cut;
        cut.type = CutType::GOMORY;
        cut.originating_node = originating_node;
        cut.creation_round = current_round;
        cut.var_ids = std::move(cut_var_ids);
        cut.coefficients = std::move(cut_coeffs);
        cut.rhs = out_beta;
        cut.sense = model::ConstraintSense::LESS_EQUAL;
        cut.violation = viol;
        cut.efficacy = efficacy;
        cut.norm2 = norm2;
        cut.is_active = true;

        generated_cuts.push_back(std::move(cut));
    }

    return generated_cuts;
}

}  // namespace kavach::algorithms::milp
