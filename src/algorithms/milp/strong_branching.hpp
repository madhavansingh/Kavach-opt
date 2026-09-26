#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <vector>

#include "algorithms/lp/lp_types.hpp"
#include "algorithms/milp/pseudocost.hpp"
#include "algorithms/milp/search_types.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"

namespace kavach::algorithms::milp {

/**
 * @brief Evaluation result for a strong branching probe on a candidate variable.
 */
struct StrongBranchProbeResult {
    model::VarId var_id{model::INVALID_VAR_ID};
    double score{0.0};
    double down_degradation{0.0};
    double up_degradation{0.0};
    core::SolverStatus down_status{core::SolverStatus::UNKNOWN};
    core::SolverStatus up_status{core::SolverStatus::UNKNOWN};
    bool down_infeasible{false};
    bool up_infeasible{false};
    bool node_proved_infeasible{false};  ///< True if both branches are infeasible
    bool can_fix_down{false};            ///< True if up is infeasible -> fix x_j <= floor(x_j*)
    bool can_fix_up{false};              ///< True if down is infeasible -> fix x_j >= ceil(x_j*)
};

/**
 * @brief Strong brancher probing candidate variables with temporary bound changes.
 */
class StrongBrancher {
public:
    using NodeLPSolver = std::function<lp::LPSolution(
        model::CanonicalModel& wm, const std::optional<lp::BasisState>& inherited_basis)>;

    explicit StrongBrancher(const core::SolverTolerances& tols);

    /**
     * @brief Probe candidate variables using temporary bound modifications and dual simplex warm
     * starts.
     *
     * Invariants:
     * - Parent working model bounds are 100% restored after every probe.
     * - Real observed degradations are fed back into PseudocostTracker.
     * - Child infeasibility is detected and reported.
     */
    [[nodiscard]] std::vector<StrongBranchProbeResult> evaluate_candidates(
        model::CanonicalModel& work_model, const std::vector<model::VarId>& candidates,
        const std::vector<double>& parent_primal_sol, double parent_obj,
        model::ObjectiveSense sense, const std::optional<lp::BasisState>& parent_basis,
        const NodeLPSolver& solve_node_lp, PseudocostTracker& pseudocost_tracker,
        std::size_t& infeasible_children_count);

private:
    core::SolverTolerances tols_;
};

}  // namespace kavach::algorithms::milp
