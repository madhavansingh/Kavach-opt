#pragma once

/**
 * @file miqp_solver.hpp
 * @brief Sovereign Convex Mixed-Integer Quadratic Programming (MIQP) Solver.
 *
 * Solves:
 *   minimize / maximize   1/2 x^T Q x + c^T x + c_0
 *   subject to            A x <= b
 *                         lb <= x <= ub
 *                         x_j in Z  for j in I
 *   where Q is symmetric positive semidefinite (PSD).
 *
 * Architecture:
 *   - Sovereign C++20 zero-external-dependency implementation
 *   - Active-Set QP node relaxations
 *   - Best-Bound priority queue node selection
 *   - Pseudocost, Strong, and Reliability branching with real QP probes
 *   - Primal heuristic evaluated on true quadratic objective
 *   - Explicit non-convex rejection when Q is indefinite
 *   - Independent zero-trust MIQP verification
 */

#include "algorithms/miqp/miqp_types.hpp"
#include "algorithms/qp/active_set.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"

namespace kavach::algorithms::miqp {

/**
 * @brief Production Convex MIQP Solver.
 */
class ConvexMIQPSolver {
public:
    explicit ConvexMIQPSolver(MIQPSolverOptions default_options = {})
        : options_(std::move(default_options)) {}

    [[nodiscard]] MIQPSolution solve(const model::CanonicalModel& model);
    [[nodiscard]] MIQPSolution solve(const model::CanonicalModel& model,
                                     const core::SolverTolerances& tolerances);
    [[nodiscard]] MIQPSolution solve(const model::CanonicalModel& model,
                                     const MIQPSolverOptions& options);

    [[nodiscard]] const MIQPSolverOptions& options() const noexcept { return options_; }
    void set_options(MIQPSolverOptions opts) noexcept { options_ = std::move(opts); }

private:
    MIQPSolverOptions options_;
};

}  // namespace kavach::algorithms::miqp
