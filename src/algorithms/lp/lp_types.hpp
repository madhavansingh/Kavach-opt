#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

#include "core/status.hpp"
#include "core/tolerances.hpp"

namespace kavach::algorithms::lp {

/**
 * @brief Configuration options for the Revised Simplex LP solver.
 */
struct LPSolverOptions {
    core::SolverTolerances tolerances;
    std::size_t max_iterations{100000};
    double time_limit_seconds{0.0};  ///< 0.0 or negative means no time limit
    bool enable_presolve{true};
    bool enable_scaling{true};
    std::size_t refactorization_interval{50};
    std::size_t bland_watchdog_threshold{30};  ///< Degenerate iterations before Bland fallback
    bool force_bland_rule{false};  ///< Unconditionally use Bland's rule from iteration 1
    bool verbose{false};
};

/**
 * @brief Telemetry and execution diagnostics recorded during the LP solve.
 */
struct LPSolverTelemetry {
    std::size_t original_rows{0};
    std::size_t original_cols{0};
    std::size_t presolved_rows{0};
    std::size_t presolved_cols{0};
    std::size_t standard_form_rows{0};
    std::size_t standard_form_cols{0};
    std::size_t standard_form_nnz{0};
    std::size_t phase1_iterations{0};
    std::size_t phase2_iterations{0};
    std::size_t total_iterations{0};
    std::size_t pivot_count{0};
    std::size_t refactorization_count{0};
    bool bland_fallback_activated{false};
    std::size_t bland_fallback_iterations{0};
    double solve_time_seconds{0.0};
    double condition_estimate{1.0};
    std::vector<std::string> numerical_warnings;
};

/**
 * @brief Complete solution output from the Revised Simplex solver.
 */
struct LPSolution {
    core::SolverStatus status{core::SolverStatus::UNKNOWN};
    double objective_value{0.0};
    std::vector<double> primal_solution;   ///< Original model variable space (size n)
    std::vector<double> dual_multipliers;  ///< Original model constraint space (size m)
    std::vector<double> reduced_costs;     ///< Original model variable space (size n)
    std::size_t iterations{0};
    LPSolverTelemetry telemetry;
    std::string message;

    [[nodiscard]] bool is_optimal() const noexcept { return status == core::SolverStatus::OPTIMAL; }
    [[nodiscard]] bool is_infeasible() const noexcept {
        return status == core::SolverStatus::INFEASIBLE;
    }
    [[nodiscard]] bool is_unbounded() const noexcept {
        return status == core::SolverStatus::UNBOUNDED;
    }
};

}  // namespace kavach::algorithms::lp
