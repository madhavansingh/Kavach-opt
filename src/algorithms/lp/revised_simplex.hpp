#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

#include "algorithms/lp/basis.hpp"
#include "algorithms/lp/lp_types.hpp"
#include "algorithms/lp/standard_form.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"

namespace kavach::algorithms::lp {

struct SimplexExecutionResult {
    core::SolverStatus status{core::SolverStatus::UNKNOWN};
    std::vector<double> z;              ///< Standard-form primal solution
    std::vector<double> y;              ///< Standard-form dual multipliers (simplex multipliers)
    std::vector<double> reduced_costs;  ///< Standard-form reduced costs
    double objective_value{0.0};
    std::string message;
    std::optional<BasisState> basis{std::nullopt};  ///< Optimal basis state if solved to optimality

    SimplexExecutionResult() = default;
    SimplexExecutionResult(core::SolverStatus s, std::vector<double> z_vec,
                           std::vector<double> y_vec, std::vector<double> rc_vec, double obj,
                           std::string msg, std::optional<BasisState> b = std::nullopt)
        : status(s),
          z(std::move(z_vec)),
          y(std::move(y_vec)),
          reduced_costs(std::move(rc_vec)),
          objective_value(obj),
          message(std::move(msg)),
          basis(std::move(b)) {}
};

/**
 * @brief Core Revised Simplex Engine implementing Phase I and Phase II with
 * Dantzig entering rule, minimum-ratio leaving rule, Bland watchdog fallback,
 * and SparseLU basis solves.
 */
class RevisedSimplexCore {
public:
    RevisedSimplexCore(const StandardFormLP& lp, const LPSolverOptions& options);

    /**
     * @brief Run the complete two-phase revised simplex optimization.
     */
    [[nodiscard]] SimplexExecutionResult run(LPSolverTelemetry& telemetry);

private:
    [[nodiscard]] core::SolverStatus run_phase1(
        SimplexBasis& basis, LPSolverTelemetry& telemetry,
        const std::chrono::steady_clock::time_point& start_time);

    [[nodiscard]] core::SolverStatus run_phase2(
        SimplexBasis& basis, LPSolverTelemetry& telemetry,
        const std::chrono::steady_clock::time_point& start_time);

    [[nodiscard]] bool check_time_limit(
        const std::chrono::steady_clock::time_point& start_time) const;

    const StandardFormLP& lp_;
    const LPSolverOptions& options_;
    const core::SolverTolerances& tol_;
};

}  // namespace kavach::algorithms::lp
