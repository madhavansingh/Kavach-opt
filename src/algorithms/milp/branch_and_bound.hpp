#pragma once

/**
 * @file branch_and_bound.hpp
 * @brief Production MILP Branch-and-Bound Engine [G] — Kavach-Opt Blueprint Section 9, 11, 32.
 *
 * Architecture:
 *   - Best-bound priority queue node selection
 *   - Most-fractional branching variable selection
 *   - Bound-overlay (no full model copy per node)
 *   - RevisedSimplex LP relaxation at each node (cold start)
 *   - Safe incumbent rounding heuristic (strict independent feasibility check)
 *   - Bound-based and infeasibility pruning
 *   - Honest optimality gap: |z_inc - z_bound| / max(1, |z_inc|)
 *   - Independent MILP verification before any OPTIMAL is reported
 *   - Node/time limits → LIMIT_REACHED with honest gap
 *   - QP models rejected with NUMERICAL_FAILURE
 *   - Pure LP models bypass B&B and are solved directly
 */

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

#include "algorithms/milp/cut_types.hpp"
#include "algorithms/milp/search_types.hpp"
#include "backend/execution_backend.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"

namespace kavach::algorithms::milp {

// ============================================================================
// Solver Options
// ============================================================================

struct MILPSolverOptions {
    core::SolverTolerances tolerances;
    std::size_t max_nodes{100000};         ///< Node exploration limit
    double time_limit_seconds{0.0};        ///< 0.0 = no limit
    bool enable_rounding_heuristic{true};  ///< Apply rounding heuristic at each node
    std::size_t rounding_interval{1};      ///< Apply rounding every N nodes
    bool enable_presolve{true};            ///< Pass through to LP sub-solver
    bool enable_scaling{true};             ///< Pass through to LP sub-solver
    bool enable_warm_starts{true};         ///< Enable dual simplex warm starts on child nodes
    CutOptions cut_options;                ///< Cutting plane engine configuration (Gomory + MIR)

    // Advanced Branch-and-Cut Search Engine Options
    BranchingStrategy branching_strategy{
        BranchingStrategy::RELIABILITY};  ///< Variable branching rule
    std::size_t strong_branching_candidates{
        5};  ///< Max candidate variables probed by strong branching
    std::size_t pseudocost_reliability_threshold{
        4};                              ///< Minimum observations per direction for reliability
    bool enable_primal_heuristic{true};  ///< Enable feasibility-oriented rounding and repair
    std::size_t heuristic_max_repair_rounds{5};  ///< Maximum constraint deficit repair passes

    bool verbose{false};
    backend::ExecutionBackend* execution_backend{nullptr};
    backend::BackendType requested_backend{backend::BackendType::CPU_SEQUENTIAL};
    backend::BackendType selected_backend{backend::BackendType::CPU_SEQUENTIAL};
    std::size_t num_workers{0};
    bool force_cuda_fallback{false};
    std::string initial_fallback_reason{};
};

// ============================================================================
// Telemetry
// ============================================================================

struct MILPSolverTelemetry {
    std::size_t nodes_explored{0};
    std::size_t nodes_pruned_bound{0};
    std::size_t nodes_pruned_infeasible{0};
    std::size_t incumbent_updates{0};
    std::size_t rounding_successes{0};
    std::size_t rounding_failures{0};
    double root_lp_bound{0.0};
    double final_best_bound{0.0};
    double solve_time_seconds{0.0};
    bool node_limit_reached{false};
    bool time_limit_reached{false};
    std::string message;

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

    // Cutting Plane Telemetry
    CutTelemetry cut_telemetry;

    // Advanced Search Telemetry
    SearchTelemetry search_telemetry;

    // Execution Backend Telemetry
    backend::BackendExecutionTelemetry backend_telemetry;
};

// ============================================================================
// Solution
// ============================================================================

struct MILPSolution {
    core::SolverStatus status{core::SolverStatus::UNKNOWN};
    double best_objective{0.0};              ///< Incumbent objective (primal bound)
    double best_bound{0.0};                  ///< Best dual bound (lower bound for MIN)
    double optimality_gap{1.0};              ///< |z_inc - z_bound| / max(1, |z_inc|)
    std::vector<double> incumbent_solution;  ///< Original variable space
    MILPSolverTelemetry telemetry;
    std::string message;

    [[nodiscard]] bool is_optimal() const noexcept { return status == core::SolverStatus::OPTIMAL; }
    [[nodiscard]] bool is_infeasible() const noexcept {
        return status == core::SolverStatus::INFEASIBLE;
    }
    [[nodiscard]] bool has_incumbent() const noexcept { return !incumbent_solution.empty(); }
};

// ============================================================================
// Abstract interface
// ============================================================================

class BranchAndBoundSolver {
public:
    virtual ~BranchAndBoundSolver() = default;

    [[nodiscard]] virtual MILPSolution solve(const model::CanonicalModel& model,
                                             const core::SolverTolerances& tolerances) = 0;
};

// ============================================================================
// Production B&B Engine
// ============================================================================

/**
 * @brief Production Branch-and-Bound MILP solver for Kavach-Opt.
 *
 * Implements (per Blueprint Section 9):
 *  - Best-bound node selection (priority queue)
 *  - Most-fractional branching variable selection
 *  - Bound-overlay node representation (no full model copy per node)
 *  - RevisedSimplex LP relaxation at each node
 *  - Safe primal rounding heuristic with strict independent feasibility check
 *  - Bound-based and infeasibility pruning
 *  - Honest optimality gap computation
 *  - Incumbent tracking
 *  - Independent MILPVerifier certification before reporting OPTIMAL
 *  - Node/time limits with honest LIMIT_REACHED + FEASIBLE status
 *  - Deterministic execution
 */
class BranchAndBound final : public BranchAndBoundSolver {
public:
    BranchAndBound() = default;
    explicit BranchAndBound(MILPSolverOptions options) : options_(std::move(options)) {}

    [[nodiscard]] MILPSolution solve(const model::CanonicalModel& model,
                                     const core::SolverTolerances& tolerances) override;

    [[nodiscard]] MILPSolution solve(const model::CanonicalModel& model);

    [[nodiscard]] MILPSolution solve(const model::CanonicalModel& model,
                                     const MILPSolverOptions& options);

    [[nodiscard]] const MILPSolverOptions& options() const noexcept { return options_; }
    void set_options(MILPSolverOptions options) noexcept { options_ = std::move(options); }

private:
    MILPSolverOptions options_;
};

}  // namespace kavach::algorithms::milp
