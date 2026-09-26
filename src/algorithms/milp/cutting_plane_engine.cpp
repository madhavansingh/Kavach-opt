#include "algorithms/milp/cutting_plane_engine.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "algorithms/lp/basis.hpp"
#include "algorithms/lp/standard_form.hpp"

namespace kavach::algorithms::milp {

namespace {

bool check_integrality(const model::CanonicalModel& model, const std::vector<double>& x,
                       double int_tol) noexcept {
    const std::size_t n = std::min(x.size(), model.num_variables());
    for (std::size_t j = 0; j < n; ++j) {
        const auto& var = model.variable(static_cast<model::VarId>(j));
        if (var.is_integer() || var.is_binary()) {
            double frac = std::abs(x[j] - std::round(x[j]));
            if (frac > int_tol) {
                return false;
            }
        }
    }
    return true;
}

std::vector<double> compute_z_from_x(const model::CanonicalModel& model,
                                     const lp::StandardFormLP& sf, const std::vector<double>& x) {
    std::vector<double> z(sf.num_cols(), 0.0);
    const auto& vtrans = sf.var_transforms();
    const std::size_t n = model.num_variables();

    for (std::size_t j = 0; j < n; ++j) {
        if (j >= vtrans.size() || j >= x.size()) {
            continue;
        }
        const auto& trans = vtrans[j];
        double xj = x[j];
        switch (trans.type) {
            case lp::VariableTransformType::FIXED:
                break;
            case lp::VariableTransformType::SHIFTED_LOWER:
            case lp::VariableTransformType::BOXED:
                if (trans.z_index1 < z.size()) {
                    z[trans.z_index1] = std::max(0.0, xj - trans.shift);
                }
                if (trans.type == lp::VariableTransformType::BOXED &&
                    trans.ub_slack_index < z.size()) {
                    const auto& var = model.variable(static_cast<model::VarId>(j));
                    z[trans.ub_slack_index] = std::max(0.0, var.upper_bound - xj);
                }
                break;
            case lp::VariableTransformType::SHIFTED_UPPER:
                if (trans.z_index1 < z.size()) {
                    z[trans.z_index1] = std::max(0.0, trans.shift - xj);
                }
                break;
            case lp::VariableTransformType::FREE:
                if (trans.z_index1 < z.size() && trans.z_index2 < z.size()) {
                    if (xj >= 0.0) {
                        z[trans.z_index1] = xj;
                        z[trans.z_index2] = 0.0;
                    } else {
                        z[trans.z_index1] = 0.0;
                        z[trans.z_index2] = -xj;
                    }
                }
                break;
        }
    }

    // Constraint slacks
    const auto& rtrans = sf.row_transforms();
    const std::size_t m = model.num_constraints();
    for (std::size_t i = 0; i < m; ++i) {
        if (i >= rtrans.size() || !rtrans[i].has_slack_or_surplus) {
            continue;
        }
        const auto& constr = model.constraint(static_cast<model::ConstrId>(i));
        double act = 0.0;
        for (const auto& term : constr.expression.terms()) {
            if (term.var_id < x.size()) {
                act += term.coefficient * x[term.var_id];
            }
        }
        std::size_t s_col = rtrans[i].slack_or_surplus_var;
        if (s_col < z.size()) {
            if (constr.sense == model::ConstraintSense::LESS_EQUAL) {
                z[s_col] = std::max(0.0, constr.rhs - act);
            } else if (constr.sense == model::ConstraintSense::GREATER_EQUAL) {
                z[s_col] = std::max(0.0, act - constr.rhs);
            }
        }
    }

    return z;
}

}  // anonymous namespace

CuttingPlaneEngine::CuttingPlaneEngine(CutOptions options)
    : options_(options), pool_(options.max_total_cuts), gomory_sep_(options), mir_sep_(options) {}

void CuttingPlaneEngine::set_options(CutOptions options) noexcept {
    options_ = options;
    pool_.set_max_capacity(options.max_total_cuts);
    gomory_sep_.set_options(options);
    mir_sep_.set_options(options);
}

CutRoundResult CuttingPlaneEngine::run_root_cuts(model::CanonicalModel& work_model,
                                                 lp::RevisedSimplex& lp_solver,
                                                 const lp::LPSolverOptions& lp_opts,
                                                 const lp::LPSolution& initial_lp_sol,
                                                 CutTelemetry& telemetry, double int_tol) {
    CutRoundResult res;
    res.objective = initial_lp_sol.objective_value;
    res.primal_solution = initial_lp_sol.primal_solution;
    res.basis = initial_lp_sol.basis;

    if (options_.strategy == CutStrategy::OFF || options_.max_cut_rounds == 0) {
        return res;
    }

    // If initial LP solution is already integer feasible, nothing to cut
    if (check_integrality(work_model, res.primal_solution, int_tol)) {
        res.integer_feasible = true;
        return res;
    }

    std::vector<model::VariableType> orig_types(work_model.num_variables());
    for (std::size_t j = 0; j < work_model.num_variables(); ++j) {
        orig_types[j] = work_model.variable(static_cast<model::VarId>(j)).type;
    }

    std::size_t total_added = 0;
    std::size_t next_cut_idx = 1;

    for (std::size_t round = 1; round <= options_.max_cut_rounds; ++round) {
        if (total_added >= options_.max_total_cuts) {
            break;
        }

        std::vector<LinearCut> round_candidates;

        // 1. Separate Gomory cuts from simplex tableau
        if (options_.enable_gomory) {
            // Solve standard form to factorize basis if needed
            lp::StandardFormLP standard_lp(work_model, lp_opts.tolerances,
                                           lp_opts.preserve_bounds_structure);
            lp::SimplexBasis basis;

            bool basis_ok = false;
            if (res.basis.has_value() && res.basis->is_valid &&
                res.basis->num_rows == standard_lp.num_rows() &&
                res.basis->num_cols == standard_lp.num_cols()) {
                basis.initialize(res.basis->basis_vars, standard_lp.num_cols());
                if (basis.factorize(standard_lp.A_csc(), lp_opts.tolerances.pivot_tol)) {
                    basis_ok = true;
                }
            }

            if (!basis_ok) {
                // Initial basis factorization
                basis.initialize(standard_lp.initial_basis(), standard_lp.num_cols());
                if (basis.factorize(standard_lp.A_csc(), lp_opts.tolerances.pivot_tol)) {
                    basis_ok = true;
                }
            }

            if (basis_ok) {
                auto z = compute_z_from_x(work_model, standard_lp, res.primal_solution);
                auto g_cuts = gomory_sep_.separate(work_model, standard_lp, basis,
                                                   res.primal_solution, z, 0, round);
                telemetry.gomory_cuts += g_cuts.size();
                telemetry.cuts_generated += g_cuts.size();
                for (auto& c : g_cuts) {
                    round_candidates.push_back(std::move(c));
                }
            }
        }

        // 2. Separate MIR cuts from mixed-integer constraints
        if (options_.enable_mir) {
            auto m_cuts = mir_sep_.separate(work_model, res.primal_solution, 0, round);
            telemetry.mir_cuts += m_cuts.size();
            telemetry.cuts_generated += m_cuts.size();
            for (auto& c : m_cuts) {
                round_candidates.push_back(std::move(c));
            }
        }

        if (round_candidates.empty()) {
            break;  // No candidate cuts separated
        }

        // 3. Filter candidates through the cut pool
        std::vector<LinearCut> accepted_for_round;
        for (auto& cand : round_candidates) {
            if (pool_.add_cut(std::move(cand))) {
                accepted_for_round.push_back(pool_.cuts().back());
            } else {
                ++telemetry.cuts_rejected;
            }
        }

        if (accepted_for_round.empty()) {
            break;  // All candidate cuts rejected
        }

        // Sort by efficacy descending, tie-break by cut_id ascending
        std::sort(accepted_for_round.begin(), accepted_for_round.end(),
                  [](const LinearCut& a, const LinearCut& b) {
                      if (a.efficacy != b.efficacy) {
                          return a.efficacy > b.efficacy;
                      }
                      return a.cut_id < b.cut_id;
                  });

        std::size_t cuts_to_add = std::min(accepted_for_round.size(), options_.max_cuts_per_round);
        if (cuts_to_add == 0) {
            break;
        }

        // 4. Add cuts to work_model
        for (std::size_t k = 0; k < cuts_to_add; ++k) {
            const auto& cut = accepted_for_round[k];
            std::string cname = "cut_" + std::string(to_string(cut.type)) + "_r" +
                                std::to_string(round) + "_" + std::to_string(next_cut_idx++);
            work_model.add_constraint(cname, cut.to_expression(), cut.sense, cut.rhs);
            ++total_added;
            ++telemetry.cuts_accepted;
            ++telemetry.root_cuts;
        }
        telemetry.active_cut_count = total_added;
        ++telemetry.cut_rounds;

        // 5. Re-solve work_model LP relaxation (disable presolve to preserve cutting planes)
        for (std::size_t j = 0; j < work_model.num_variables(); ++j) {
            work_model.variable(static_cast<model::VarId>(j)).type =
                model::VariableType::CONTINUOUS;
        }

        lp::LPSolverOptions re_opts = lp_opts;
        re_opts.enable_presolve = false;
        auto re_sol = lp_solver.solve(work_model, re_opts);

        // Restore variable types
        for (std::size_t j = 0; j < work_model.num_variables(); ++j) {
            work_model.variable(static_cast<model::VarId>(j)).type = orig_types[j];
        }

        if (re_sol.status == core::SolverStatus::INFEASIBLE) {
            res.infeasible = true;
            res.cuts_added = total_added;
            return res;
        }

        if (re_sol.status != core::SolverStatus::OPTIMAL) {
            // Re-solve failed numerically or reached limit: stop adding cuts
            break;
        }

        double old_obj = res.objective;
        res.objective = re_sol.objective_value;
        res.primal_solution = re_sol.primal_solution;
        res.basis = re_sol.basis;

        // Check if new relaxation solution is integer feasible
        if (check_integrality(work_model, res.primal_solution, int_tol)) {
            res.integer_feasible = true;
            res.cuts_added = total_added;
            return res;
        }

        // Tail-off detection: if objective improved less than 1e-4, stop
        double improvement = std::abs(res.objective - old_obj);
        if (improvement < 1e-4) {
            break;
        }
    }

    res.cuts_added = total_added;
    return res;
}

}  // namespace kavach::algorithms::milp
