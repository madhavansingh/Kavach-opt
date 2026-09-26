#include "algorithms/milp/mir_separator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace kavach::algorithms::milp {

namespace {

/**
 * @brief Evaluates the standard MIR function F_alpha(a).
 * F_alpha(a) = floor(a) + max(0, (a - floor(a)) - alpha) / (1 - alpha)
 */
double mir_function(double a, double alpha) noexcept {
    double fl = std::floor(a);
    double frac = a - fl;
    if (frac < 1e-12) {
        frac = 0.0;
    }
    double excess = std::max(0.0, frac - alpha);
    return fl + (excess / (1.0 - alpha));
}

}  // anonymous namespace

MIRSeparator::MIRSeparator(const CutOptions& options) : options_(options) {}

std::vector<LinearCut> MIRSeparator::separate(const model::CanonicalModel& model,
                                              const std::vector<double>& x_lp,
                                              std::size_t originating_node,
                                              std::size_t current_round) const {
    std::vector<LinearCut> cuts;
    if (!options_.enable_mir) {
        return cuts;
    }

    const std::size_t m = model.num_constraints();
    for (std::size_t i = 0; i < m; ++i) {
        const auto& constr = model.constraint(static_cast<model::ConstrId>(i));
        auto row_cuts = separate_from_row(model, constr, x_lp, originating_node, current_round);
        for (auto& rc : row_cuts) {
            cuts.push_back(std::move(rc));
        }
    }
    return cuts;
}

std::vector<LinearCut> MIRSeparator::separate_from_row(const model::CanonicalModel& model,
                                                       const model::Constraint& constr,
                                                       const std::vector<double>& x_lp,
                                                       std::size_t originating_node,
                                                       std::size_t current_round) const {
    std::vector<LinearCut> result;

    // Check whether the row has any integer variables
    bool has_integer = false;
    for (const auto& term : constr.expression.terms()) {
        if (term.var_id < model.num_variables()) {
            const auto& var = model.variable(term.var_id);
            if (var.is_integer() || var.is_binary()) {
                has_integer = true;
                break;
            }
        }
    }
    if (!has_integer) {
        return result;  // Pure continuous row: MIR not applicable
    }

    // Determine orientations to test:
    // sense == LESS_EQUAL -> sign = +1
    // sense == GREATER_EQUAL -> sign = -1
    // sense == EQUAL -> test both +1 and -1
    std::vector<double> orientations;
    if (constr.sense == model::ConstraintSense::LESS_EQUAL) {
        orientations.push_back(1.0);
    } else if (constr.sense == model::ConstraintSense::GREATER_EQUAL) {
        orientations.push_back(-1.0);
    } else {
        orientations.push_back(1.0);
        orientations.push_back(-1.0);
    }

    for (double orient : orientations) {
        // Inequality is: orient * \sum a_j x_j <= orient * rhs
        double base_rhs = orient * constr.rhs;

        // Try several scaling factors delta to find best cut
        std::vector<double> scales = {1.0, 0.5, 2.0};
        for (const auto& term : constr.expression.terms()) {
            double coeff = std::abs(term.coefficient);
            if (coeff > 1e-4 && coeff < 1e4) {
                double inv = 1.0 / coeff;
                if (std::abs(inv - 1.0) > 1e-4) {
                    scales.push_back(inv);
                }
            }
        }

        for (double delta : scales) {
            if (delta <= 1e-6 || !std::isfinite(delta)) {
                continue;
            }

            // Bound shifts:
            // x_j' = x_j - l_j >= 0 for lower-bounded variables
            double shifted_rhs = base_rhs * delta;
            bool valid_bounds = true;

            for (const auto& term : constr.expression.terms()) {
                if (term.var_id >= model.num_variables()) {
                    continue;
                }
                const auto& var = model.variable(term.var_id);
                double a_j = orient * term.coefficient * delta;

                if (var.is_integer() || var.is_binary()) {
                    if (var.has_lower_bound()) {
                        double lb = var.lower_bound;
                        shifted_rhs -= a_j * lb;
                    } else if (var.has_upper_bound()) {
                        // Upper bounded only: x_j' = u_j - x_j >= 0
                        // Not supported in basic MIR without sign reversal: skip row
                        valid_bounds = false;
                        break;
                    } else {
                        // Free integer variable
                        valid_bounds = false;
                        break;
                    }
                } else {
                    // Continuous variable
                    if (var.has_lower_bound() && var.lower_bound != 0.0) {
                        double lb = var.lower_bound;
                        shifted_rhs -= a_j * lb;
                    }
                }
            }

            if (!valid_bounds) {
                continue;
            }

            double f0 = shifted_rhs - std::floor(shifted_rhs);
            if (f0 < options_.min_fractionality || f0 > 1.0 - options_.min_fractionality) {
                continue;  // Not fractional enough
            }

            double alpha = f0;
            double cut_rhs = std::floor(shifted_rhs);

            std::vector<model::VarId> cut_vids;
            std::vector<double> cut_coeffs;
            bool numerical_ok = true;

            for (const auto& term : constr.expression.terms()) {
                if (term.var_id >= model.num_variables()) {
                    continue;
                }
                const auto& var = model.variable(term.var_id);
                double a_j = orient * term.coefficient * delta;

                if (var.is_integer() || var.is_binary()) {
                    double mir_coeff = mir_function(a_j, alpha);
                    double lb = var.has_lower_bound() ? var.lower_bound : 0.0;
                    cut_rhs += mir_coeff * lb;

                    if (std::abs(mir_coeff) > 1e-12) {
                        if (std::abs(mir_coeff) > options_.max_absolute_coefficient) {
                            numerical_ok = false;
                            break;
                        }
                        cut_vids.push_back(term.var_id);
                        cut_coeffs.push_back(mir_coeff);
                    }
                } else {
                    // Continuous variable
                    double mir_coeff = 0.0;
                    if (a_j < 0.0) {
                        mir_coeff = a_j / (1.0 - alpha);
                    }
                    double lb = var.has_lower_bound() ? var.lower_bound : 0.0;
                    cut_rhs += mir_coeff * lb;

                    if (std::abs(mir_coeff) > 1e-12) {
                        if (std::abs(mir_coeff) > options_.max_absolute_coefficient) {
                            numerical_ok = false;
                            break;
                        }
                        cut_vids.push_back(term.var_id);
                        cut_coeffs.push_back(mir_coeff);
                    }
                }
            }

            if (!numerical_ok || cut_vids.empty() || !std::isfinite(cut_rhs)) {
                continue;
            }

            // Check dynamic range
            double max_abs = 0.0;
            double min_abs = std::numeric_limits<double>::infinity();
            for (double c : cut_coeffs) {
                double abs_c = std::abs(c);
                max_abs = std::max(max_abs, abs_c);
                min_abs = std::min(min_abs, abs_c);
            }
            if (min_abs > 0.0 && (max_abs / min_abs) > options_.max_coefficient_range) {
                continue;
            }

            // Check violation at x_lp
            double act = 0.0;
            for (std::size_t k = 0; k < cut_vids.size(); ++k) {
                model::VarId vid = cut_vids[k];
                if (vid < x_lp.size()) {
                    act += cut_coeffs[k] * x_lp[vid];
                }
            }
            double viol = std::max(0.0, act - cut_rhs);
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
            cut.type = CutType::MIR;
            cut.originating_node = originating_node;
            cut.creation_round = current_round;
            cut.var_ids = std::move(cut_vids);
            cut.coefficients = std::move(cut_coeffs);
            cut.rhs = cut_rhs;
            cut.sense = model::ConstraintSense::LESS_EQUAL;
            cut.violation = viol;
            cut.efficacy = efficacy;
            cut.norm2 = norm2;
            cut.is_active = true;

            result.push_back(std::move(cut));
            break;  // Found good scale for this orientation
        }
    }

    return result;
}

}  // namespace kavach::algorithms::milp
