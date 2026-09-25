#pragma once

#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"

namespace kavach::algorithms::qp {

struct QPSolution {
    core::SolverStatus status{core::SolverStatus::UNKNOWN};
    double objective_value{0.0};
    std::vector<double> primal_solution;
    std::vector<double> dual_solution;
    double kkt_residual{0.0};
    std::size_t iterations{0};
};

/**
 * @brief Architecture interface scaffolding for [H] QP Solver (Convex Active-Set).
 * Full implementation scheduled for Phase 9 per Blueprint Section 4 & Section 32.
 */
class ActiveSetQPSolver {
public:
    virtual ~ActiveSetQPSolver() = default;

    [[nodiscard]] virtual QPSolution solve(const model::CanonicalModel& model,
                                           const core::SolverTolerances& tolerances) = 0;
};

}  // namespace kavach::algorithms::qp
