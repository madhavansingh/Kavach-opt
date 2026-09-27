#pragma once

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "backend/execution_backend.hpp"
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
    double basis_drift_tol{1e-9};  ///< Max acceptable ||B x_B - b||_inf before refactorization
    std::size_t bland_watchdog_threshold{30};  ///< Degenerate iterations before Bland fallback
    bool force_bland_rule{false};  ///< Unconditionally use Bland's rule from iteration 1
    bool verbose{false};
    backend::ExecutionBackend* execution_backend{nullptr};  ///< Optional active execution backend
    backend::BackendType requested_backend{backend::BackendType::CPU_SEQUENTIAL};
    backend::BackendType selected_backend{backend::BackendType::CPU_SEQUENTIAL};
    std::size_t num_workers{0};
    bool force_cuda_fallback{false};
    std::string initial_fallback_reason;
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

    // Basis-Update Factorization Telemetry
    std::size_t basis_updates{0};
    std::size_t full_refactorizations{0};
    std::size_t update_failures{0};
    std::string refactorization_reason{"INITIAL_FACTORIZATION"};
    std::size_t maximum_update_chain{0};
    double average_update_chain{0.0};
    double basis_residual_max{0.0};
    double update_application_time_seconds{0.0};
    double full_refactorization_time_seconds{0.0};
    std::size_t base_factorization_bytes{0};
    std::size_t update_chain_bytes{0};
    std::size_t peak_memory_bytes{0};
    std::size_t memory_after_refactor_bytes{0};

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

    // Execution Backend Telemetry
    backend::BackendExecutionTelemetry backend_telemetry;
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
