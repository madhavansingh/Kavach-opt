#pragma once

#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"

namespace kavach::algorithms::milp {

struct MILPSolution {
    core::SolverStatus status{core::SolverStatus::UNKNOWN};
    double best_objective{0.0};
    double best_bound{0.0};
    double optimality_gap{1.0};
    std::vector<double> incumbent_solution;
    std::size_t nodes_explored{0};
};

/**
 * @brief Architecture interface scaffolding for [G] MILP Solver (Branch-and-Bound).
 * Full implementation scheduled for Phase 7 per Blueprint Section 4 & Section 32.
 */
class BranchAndBoundSolver {
public:
    virtual ~BranchAndBoundSolver() = default;

    [[nodiscard]] virtual MILPSolution solve(const model::CanonicalModel& model,
                                             const core::SolverTolerances& tolerances) = 0;
};

}  // namespace kavach::algorithms::milp
