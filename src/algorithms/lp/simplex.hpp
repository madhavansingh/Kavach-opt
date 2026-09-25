#pragma once

#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"

namespace kavach::algorithms::lp {

struct LPSolution {
    core::SolverStatus status{core::SolverStatus::UNKNOWN};
    double objective_value{0.0};
    std::vector<double> primal_solution;
    std::vector<double> dual_multipliers;
    std::vector<double> reduced_costs;
    std::size_t iterations{0};
};

/**
 * @brief Architecture interface scaffolding for [F] LP Solver (Revised Simplex).
 * Full implementation scheduled for Phase 5 per Blueprint Section 4 & Section 32.
 */
class RevisedSimplexSolver {
public:
    virtual ~RevisedSimplexSolver() = default;

    [[nodiscard]] virtual LPSolution solve(const model::CanonicalModel& model,
                                           const core::SolverTolerances& tolerances) = 0;
};

}  // namespace kavach::algorithms::lp
