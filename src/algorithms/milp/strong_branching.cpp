#include "algorithms/milp/strong_branching.hpp"

#include <algorithm>
#include <cmath>

namespace kavach::algorithms::milp {

StrongBrancher::StrongBrancher(const core::SolverTolerances& tols) : tols_(tols) {}

std::vector<StrongBranchProbeResult> StrongBrancher::evaluate_candidates(
    model::CanonicalModel& work_model, const std::vector<model::VarId>& candidates,
    const std::vector<double>& parent_primal_sol, double parent_obj, model::ObjectiveSense sense,
    const std::optional<lp::BasisState>& parent_basis, const NodeLPSolver& solve_node_lp,
    PseudocostTracker& pseudocost_tracker, std::size_t& infeasible_children_count) {
    std::vector<StrongBranchProbeResult> results;
    results.reserve(candidates.size());

    // RAII helper to temporarily relax types to continuous for LP probes
    auto relax_types = [&work_model]() {
        std::vector<model::VariableType> saved(work_model.num_variables());
        for (std::size_t j = 0; j < work_model.num_variables(); ++j) {
            auto& v = work_model.variable(static_cast<model::VarId>(j));
            saved[j] = v.type;
            v.type = model::VariableType::CONTINUOUS;
        }
        return saved;
    };

    auto restore_types = [&work_model](const std::vector<model::VariableType>& saved) {
        for (std::size_t j = 0; j < saved.size(); ++j) {
            work_model.variable(static_cast<model::VarId>(j)).type = saved[j];
        }
    };

    const double inf_penalty = 1e5;

    for (model::VarId var_id : candidates) {
        if (var_id >= parent_primal_sol.size() || var_id >= work_model.num_variables()) {
            continue;
        }

        double val = parent_primal_sol[var_id];
        double f = val - std::floor(val);
        if (f <= tols_.integrality_tol || f >= 1.0 - tols_.integrality_tol) {
            continue;  // not fractional
        }

        StrongBranchProbeResult probe;
        probe.var_id = var_id;
        auto& var = work_model.variable(var_id);
        double orig_lb = var.lower_bound;
        double orig_ub = var.upper_bound;

        // ---------------------------------------------------------------------
        // Probe DOWN branch: x_j <= floor(val)
        // ---------------------------------------------------------------------
        double floor_val = std::floor(val);
        if (orig_lb <= floor_val + tols_.primal_feasibility_tol) {
            var.upper_bound = floor_val;
            auto saved_types = relax_types();

            auto down_sol = solve_node_lp(work_model, parent_basis);

            restore_types(saved_types);
            var.upper_bound = orig_ub;  // Restore bound immediately

            probe.down_status = down_sol.status;
            if (down_sol.status == core::SolverStatus::OPTIMAL) {
                double delta = (sense == model::ObjectiveSense::MINIMIZE)
                                   ? (down_sol.objective_value - parent_obj)
                                   : (parent_obj - down_sol.objective_value);
                probe.down_degradation = std::max(0.0, delta);
                pseudocost_tracker.update_down(var_id, parent_obj, down_sol.objective_value, val,
                                               tols_.integrality_tol);
            } else if (down_sol.status == core::SolverStatus::INFEASIBLE) {
                probe.down_infeasible = true;
                probe.down_degradation = inf_penalty;
                ++infeasible_children_count;
            } else {
                // Numerical issue or failure
                probe.down_degradation = 0.0;
            }
        } else {
            // Lower bound already exceeds floor_val
            probe.down_infeasible = true;
            probe.down_degradation = inf_penalty;
            ++infeasible_children_count;
        }

        // ---------------------------------------------------------------------
        // Probe UP branch: x_j >= ceil(val)
        // ---------------------------------------------------------------------
        double ceil_val = std::ceil(val);
        if (ceil_val <= orig_ub + tols_.primal_feasibility_tol) {
            var.lower_bound = ceil_val;
            auto saved_types = relax_types();

            auto up_sol = solve_node_lp(work_model, parent_basis);

            restore_types(saved_types);
            var.lower_bound = orig_lb;  // Restore bound immediately

            probe.up_status = up_sol.status;
            if (up_sol.status == core::SolverStatus::OPTIMAL) {
                double delta = (sense == model::ObjectiveSense::MINIMIZE)
                                   ? (up_sol.objective_value - parent_obj)
                                   : (parent_obj - up_sol.objective_value);
                probe.up_degradation = std::max(0.0, delta);
                pseudocost_tracker.update_up(var_id, parent_obj, up_sol.objective_value, val,
                                             tols_.integrality_tol);
            } else if (up_sol.status == core::SolverStatus::INFEASIBLE) {
                probe.up_infeasible = true;
                probe.up_degradation = inf_penalty;
                ++infeasible_children_count;
            } else {
                probe.up_degradation = 0.0;
            }
        } else {
            // Ceil exceeds upper bound
            probe.up_infeasible = true;
            probe.up_degradation = inf_penalty;
            ++infeasible_children_count;
        }

        // Domain deductions
        if (probe.down_infeasible && probe.up_infeasible) {
            probe.node_proved_infeasible = true;
        } else if (probe.down_infeasible) {
            probe.can_fix_up = true;
        } else if (probe.up_infeasible) {
            probe.can_fix_down = true;
        }

        // Compute combined strong branching product score
        probe.score = std::max(probe.down_degradation, 1e-6) * std::max(probe.up_degradation, 1e-6);
        if (!std::isfinite(probe.score) || probe.score > 1e15) {
            probe.score = 1e15;
        }

        results.push_back(probe);

        if (probe.node_proved_infeasible) {
            // Entire node is proved infeasible, stop evaluating further candidates
            break;
        }
    }

    return results;
}

}  // namespace kavach::algorithms::milp
