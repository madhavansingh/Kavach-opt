#pragma once

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "numerics/scaling.hpp"

namespace kavach::algorithms::lp {

/**
 * @brief Reusable simplex basis state for warm starting.
 *
 * Encapsulates the complete basis structure from an optimal LP solve:
 * - basic and nonbasic variable column indices
 * - dimension verification metadata
 * - scaling regime compatibility
 */
struct BasisState {
    std::vector<std::size_t> basis_vars;     ///< Basic variable column index for each row (size m)
    std::vector<bool> is_basic;              ///< Basic indicator (size n)
    std::vector<std::size_t> nonbasic_vars;  ///< Nonbasic variable indices
    std::size_t num_rows{0};                 ///< Structural row dimension m
    std::size_t num_cols{0};                 ///< Structural column dimension n
    bool is_valid{false};
    bool scaling_applied{false};
    std::optional<numerics::ScalingFactors> scaling_factors;
};

/**
 * @brief Configuration options for the Revised Simplex LP solver.
 */
struct LPSolverOptions {
    core::SolverTolerances tolerances;
    std::size_t max_iterations{100000};
    double time_limit_seconds{0.0};  ///< 0.0 or negative means no time limit
    bool enable_presolve{true};
    bool enable_scaling{true};
    bool enable_warm_start{false};
    bool preserve_bounds_structure{false};  ///< Keep BOXED representation even if ub == lb
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

    // Dual Simplex / Warm-Start Telemetry
    std::size_t warm_start_attempts{0};
    std::size_t warm_start_successes{0};
    std::size_t warm_start_fallbacks{0};
    std::vector<std::string> warm_start_fallback_reasons;
    std::size_t dual_simplex_iterations{0};
    std::size_t cold_start_iterations{0};
    double warm_start_time_seconds{0.0};
    double cold_start_time_seconds{0.0};
    std::size_t basis_transfer_count{0};
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
    std::optional<BasisState> basis;  ///< Reusable basis state if solved to optimality
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
