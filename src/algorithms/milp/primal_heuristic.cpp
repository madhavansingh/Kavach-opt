#include "algorithms/milp/primal_heuristic.hpp"

#include <algorithm>
#include <cmath>

#include "numerics/model_converter.hpp"

namespace kavach::algorithms::milp {

PrimalHeuristic::PrimalHeuristic(const core::SolverTolerances& tols, std::size_t max_repair_rounds)
    : tols_(tols), max_repair_rounds_(max_repair_rounds) {}

PrimalHeuristicResult PrimalHeuristic::search(const model::CanonicalModel& model,
                                              const std::vector<double>& x_lp,
                                              backend::ExecutionBackend* backend) const {
    PrimalHeuristicResult res;
    if (x_lp.empty()) {
        return res;
    }

    const std::size_t n = std::min(x_lp.size(), model.num_variables());
    const std::size_t m = model.num_constraints();
    std::vector<double> x = x_lp;

    // Phase 1: Round all integer and binary variables deterministically
    std::vector<std::size_t> fractional_indices;
    for (std::size_t j = 0; j < n; ++j) {
        const auto& var = model.variable(static_cast<model::VarId>(j));
        if (var.is_integer() || var.is_binary()) {
            double frac = std::abs(x[j] - std::round(x[j]));
            if (frac > tols_.integrality_tol) {
                fractional_indices.push_back(j);
            }
            double rounded = std::round(x[j]);
            rounded = std::max(var.lower_bound, std::min(var.upper_bound, rounded));
            x[j] = rounded;
        }
    }

    // Batched candidate activity screening via ExecutionBackend::batched_spmv
    // Generates up to 16 candidate vectors (nearest-integer + single-variable floor/ceil flips)
    // and evaluates Y = A * X in a single batched CSR SpMV call.
    if (backend != nullptr && m > 0 && n == model.num_variables()) {
        const auto A_csr = numerics::canonical_model_to_csr(model);
        const std::size_t batch_size =
            std::min(std::size_t(16), std::size_t(1) + fractional_indices.size());
        std::vector<double> batch_x(batch_size * n, 0.0);
        for (std::size_t j = 0; j < n; ++j) {
            batch_x[j] = x[j];
        }
        for (std::size_t b = 1; b < batch_size; ++b) {
            for (std::size_t j = 0; j < n; ++j) {
                batch_x[b * n + j] = x[j];
            }
            std::size_t flip_j = fractional_indices[b - 1];
            const auto& var = model.variable(static_cast<model::VarId>(flip_j));
            double lp_val = x_lp[flip_j];
            double alt_val = (x[flip_j] >= lp_val) ? std::floor(lp_val) : std::ceil(lp_val);
            alt_val = std::max(var.lower_bound, std::min(var.upper_bound, alt_val));
            batch_x[b * n + flip_j] = alt_val;
        }
        std::vector<double> batch_activities(batch_size * m, 0.0);
        backend->batched_spmv(A_csr, batch_x.data(), batch_activities.data(), batch_size);
    }

    // Phase 2: Bounded repair iterations on continuous variables
    std::size_t rounds_used = 0;
    for (std::size_t round = 0; round < max_repair_rounds_; ++round) {
        bool all_satisfied = true;
        rounds_used = round + 1;

        for (std::size_t i = 0; i < model.num_constraints(); ++i) {
            const auto& constr = model.constraint(static_cast<model::ConstrId>(i));
            double activity = 0.0;
            for (const auto& term : constr.expression.terms()) {
                if (term.var_id < x.size()) {
                    activity += term.coefficient * x[term.var_id];
                }
            }

            double rhs = constr.rhs;
            double viol = 0.0;
            bool need_increase = false;
            bool need_decrease = false;

            switch (constr.sense) {
                case model::ConstraintSense::LESS_EQUAL:
                    if (activity > rhs + tols_.primal_feasibility_tol) {
                        viol = activity - rhs;
                        need_decrease = true;
                        all_satisfied = false;
                    }
                    break;
                case model::ConstraintSense::GREATER_EQUAL:
                    if (activity < rhs - tols_.primal_feasibility_tol) {
                        viol = rhs - activity;
                        need_increase = true;
                        all_satisfied = false;
                    }
                    break;
                case model::ConstraintSense::EQUAL:
                    if (activity > rhs + tols_.primal_feasibility_tol) {
                        viol = activity - rhs;
                        need_decrease = true;
                        all_satisfied = false;
                    } else if (activity < rhs - tols_.primal_feasibility_tol) {
                        viol = rhs - activity;
                        need_increase = true;
                        all_satisfied = false;
                    }
                    break;
            }

            if (viol <= tols_.primal_feasibility_tol) {
                continue;
            }

            // Attempt to compensate violation using continuous variables in this constraint
            double rem_viol = viol;
            for (const auto& term : constr.expression.terms()) {
                if (rem_viol <= tols_.primal_feasibility_tol) {
                    break;
                }
                if (term.var_id >= n) {
                    continue;
                }
                const auto& var = model.variable(term.var_id);
                if (var.is_integer() || var.is_binary()) {
                    continue;  // Do not alter integer variables during repair
                }
                if (std::abs(term.coefficient) < 1e-12) {
                    continue;
                }

                double coeff = term.coefficient;
                double cur_val = x[term.var_id];

                if (need_decrease) {
                    // Need to decrease activity: either decrease x_j (if coeff > 0) or increase x_j
                    // (if coeff < 0)
                    if (coeff > 0.0 && var.has_lower_bound() && cur_val > var.lower_bound) {
                        double max_step = cur_val - var.lower_bound;
                        double needed_step = rem_viol / coeff;
                        double step = std::min(max_step, needed_step);
                        x[term.var_id] -= step;
                        rem_viol -= step * coeff;
                    } else if (coeff < 0.0 && var.has_upper_bound() && cur_val < var.upper_bound) {
                        double max_step = var.upper_bound - cur_val;
                        double needed_step = rem_viol / (-coeff);
                        double step = std::min(max_step, needed_step);
                        x[term.var_id] += step;
                        rem_viol -= step * (-coeff);
                    }
                } else if (need_increase) {
                    // Need to increase activity: either increase x_j (if coeff > 0) or decrease x_j
                    // (if coeff < 0)
                    if (coeff > 0.0 && var.has_upper_bound() && cur_val < var.upper_bound) {
                        double max_step = var.upper_bound - cur_val;
                        double needed_step = rem_viol / coeff;
                        double step = std::min(max_step, needed_step);
                        x[term.var_id] += step;
                        rem_viol -= step * coeff;
                    } else if (coeff < 0.0 && var.has_lower_bound() && cur_val > var.lower_bound) {
                        double max_step = cur_val - var.lower_bound;
                        double needed_step = rem_viol / (-coeff);
                        double step = std::min(max_step, needed_step);
                        x[term.var_id] -= step;
                        rem_viol -= step * (-coeff);
                    }
                }
            }
        }

        if (all_satisfied) {
            break;
        }
    }

    // Phase 3: Strict independent feasibility verification
    // Check all constraints
    for (std::size_t i = 0; i < model.num_constraints(); ++i) {
        const auto& constr = model.constraint(static_cast<model::ConstrId>(i));
        double activity = 0.0;
        for (const auto& term : constr.expression.terms()) {
            if (term.var_id < x.size()) {
                activity += term.coefficient * x[term.var_id];
            }
        }
        double rhs = constr.rhs;
        double viol = 0.0;
        switch (constr.sense) {
            case model::ConstraintSense::LESS_EQUAL:
                viol = std::max(0.0, activity - rhs);
                break;
            case model::ConstraintSense::GREATER_EQUAL:
                viol = std::max(0.0, rhs - activity);
                break;
            case model::ConstraintSense::EQUAL:
                viol = std::abs(activity - rhs);
                break;
        }
        if (viol > tols_.primal_feasibility_tol) {
            return res;  // Infeasible
        }
    }

    // Check variable bounds
    for (std::size_t j = 0; j < n; ++j) {
        const auto& var = model.variable(static_cast<model::VarId>(j));
        if (var.has_lower_bound() && x[j] < var.lower_bound - tols_.primal_feasibility_tol) {
            return res;
        }
        if (var.has_upper_bound() && x[j] > var.upper_bound + tols_.primal_feasibility_tol) {
            return res;
        }
    }

    // Check integer integrality
    for (std::size_t j = 0; j < n; ++j) {
        const auto& var = model.variable(static_cast<model::VarId>(j));
        if (var.is_integer() || var.is_binary()) {
            double frac = std::abs(x[j] - std::round(x[j]));
            if (frac > tols_.integrality_tol) {
                return res;
            }
        }
    }

    // Phase 4: Compute objective value
    double obj = model.objective_offset();
    for (const auto& term : model.objective().linear_terms.terms()) {
        if (term.var_id < x.size()) {
            obj += term.coefficient * x[term.var_id];
        }
    }

    if (!std::isfinite(obj)) {
        return res;
    }

    res.found_feasible = true;
    res.solution = std::move(x);
    res.objective = obj;
    res.repair_rounds_used = rounds_used;
    return res;
}

}  // namespace kavach::algorithms::milp
