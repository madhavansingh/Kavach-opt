#pragma once

#include <string_view>

namespace kavach::core {

/**
 * @brief Solver Status Taxonomy as strictly defined in Blueprint v2 Section 11.
 *
 * Rule: The solver must never report OPTIMAL unless the independent verifier [Y]
 * has confirmed feasibility and optimality conditions.
 */
enum class SolverStatus {
    OPTIMAL,     ///< Solved and independently verified against feasibility + optimality conditions
                 ///< within tolerance.
    FEASIBLE,    ///< Feasible point found, but optimality not (yet) certified (e.g., MILP with
                 ///< nonzero gap at limit).
    INFEASIBLE,  ///< Proven infeasible (LP: Farkas certificate; MILP: exhaustive search).
    UNBOUNDED,   ///< Proven unbounded (improving feasible ray exists).
    LIMIT_REACHED,      ///< Time or iteration/node limit hit before OPTIMAL or definitive proof.
    NUMERICAL_FAILURE,  ///< Numerical safeguard (near-zero pivot, residual > tol, non-PSD QP)
                        ///< prevented trustworthy result.
    UNKNOWN  ///< Undetermined outcome (logged for investigation, never silently discarded).
};

[[nodiscard]] constexpr std::string_view to_string(SolverStatus status) noexcept {
    switch (status) {
        case SolverStatus::OPTIMAL:
            return "OPTIMAL";
        case SolverStatus::FEASIBLE:
            return "FEASIBLE";
        case SolverStatus::INFEASIBLE:
            return "INFEASIBLE";
        case SolverStatus::UNBOUNDED:
            return "UNBOUNDED";
        case SolverStatus::LIMIT_REACHED:
            return "LIMIT_REACHED";
        case SolverStatus::NUMERICAL_FAILURE:
            return "NUMERICAL_FAILURE";
        case SolverStatus::UNKNOWN:
            return "UNKNOWN";
    }
    return "UNKNOWN";
}

[[nodiscard]] constexpr bool is_terminal(SolverStatus status) noexcept {
    return status == SolverStatus::OPTIMAL || status == SolverStatus::INFEASIBLE ||
           status == SolverStatus::UNBOUNDED || status == SolverStatus::LIMIT_REACHED ||
           status == SolverStatus::NUMERICAL_FAILURE;
}

}  // namespace kavach::core
