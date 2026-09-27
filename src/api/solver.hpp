#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "algorithms/lp/lp_types.hpp"
#include "algorithms/milp/cut_types.hpp"
#include "algorithms/milp/search_types.hpp"
#include "algorithms/miqp/miqp_types.hpp"
#include "backend/execution_backend.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"
#include "strategy/strategy.hpp"
#include "verification/verifier.hpp"

namespace kavach::api {

enum class ProblemType { AUTO, LP, MILP, QP, MIQP };

struct SolverOptions {
    ProblemType problem_type{ProblemType::AUTO};
    core::SolverTolerances tolerances;
    std::optional<std::chrono::milliseconds> time_limit{std::nullopt};
    std::size_t max_iterations{100000};
    std::size_t max_nodes{50000};
    uint64_t seed{42};
    bool use_gpu{false};
    bool deterministic{true};
    bool run_presolve{true};
    strategy::StrategyConfig strategy_config;
    algorithms::milp::CutOptions cut_options;
    algorithms::milp::BranchingStrategy branching_strategy{
        algorithms::milp::BranchingStrategy::RELIABILITY};
    std::size_t strong_branching_candidates{5};
    std::size_t pseudocost_reliability_threshold{4};
    bool enable_primal_heuristic{true};
    bool force_cuda_fallback{false};
};

struct SolverResult {
    core::SolverStatus status{core::SolverStatus::UNKNOWN};
    double objective_value{0.0};
    std::vector<double> primal_solution;
    std::vector<double> dual_solution;
    std::vector<double> reduced_costs;
    double optimality_gap{0.0};
    std::size_t iterations{0};
    std::size_t nodes{0};
    double solve_time_seconds{0.0};
    verification::VerificationReport verification_report;
    backend::BackendType backend_used{backend::BackendType::CPU_SEQUENTIAL};
    backend::BackendExecutionTelemetry backend_telemetry;
    strategy::StrategyDecision strategy_decision;
    std::optional<algorithms::milp::CutTelemetry> cut_telemetry{std::nullopt};
    std::optional<algorithms::milp::SearchTelemetry> search_telemetry{std::nullopt};
    std::optional<algorithms::miqp::MIQPTelemetry> miqp_telemetry{std::nullopt};
    std::optional<algorithms::lp::LPSolverTelemetry> lp_telemetry{std::nullopt};
};

/**
 * @brief Abstract solver interface.
 *
 * Defines the contract for mathematical programming solvers in Kavach-Opt.
 */
class Solver {
public:
    virtual ~Solver() = default;

    /**
     * @brief Solves the given canonical model with the specified options.
     *
     * @param model Input canonical model. The caller retains ownership.
     *              The model is accessed read-only and must remain valid for
     *              the duration of the solve() call.
     * @param options Solver configuration options.
     * @return SolverResult Complete result including status, primal/dual solutions,
     *                      and independent verification certification.
     */
    [[nodiscard]] virtual SolverResult solve(const model::CanonicalModel& model,
                                             const SolverOptions& options) = 0;
};

/**
 * @brief Production programmatic solver implementation for Kavach-Opt.
 *
 * Integrates rule-based strategy selection, core numerical algorithms
 * (Revised Simplex, Branch-and-Bound, Primal Active-Set QP), independent
 * verification (LPVerifier, MILPVerifier, QPVerifier), and telemetry.
 *
 * ### Architectural Invariants & Guarantees:
 * - **Memory & Lifetime**: Models are accepted by const reference and never
 *   stored internally. SolverResult instances own all allocated vectors
 *   (primal, dual, reduced costs) by value, allowing safe decoupling from the
 *   originating solver instance.
 * - **Thread Safety**: Distinct KavachSolver instances operating on separate
 *   models are concurrently thread-safe. A single KavachSolver instance is
 *   re-entrant for sequential calls on the same thread.
 * - **Exceptions**: Throws std::invalid_argument for ill-formed models or
 *   incompatible options; throws std::runtime_error for fatal numerical
 *   breakdowns. Mathematical statuses (e.g. INFEASIBLE, UNBOUNDED, OPTIMAL)
 *   are communicated via SolverResult::status without throwing exceptions.
 * - **Platform Sovereignty**: Compiles and operates cleanly on CPU-only
 *   architectures without requiring CUDA runtime libraries or GPU headers.
 */
class KavachSolver : public Solver {
public:
    KavachSolver() = default;
    explicit KavachSolver(strategy::StrategyConfig config);

    [[nodiscard]] SolverResult solve(const model::CanonicalModel& model,
                                     const SolverOptions& options) override;

    [[nodiscard]] SolverResult solve(const model::CanonicalModel& model);

    [[nodiscard]] SolverResult solve(const model::CanonicalModel& model,
                                     const strategy::StrategyDecision& decision,
                                     const SolverOptions& options = SolverOptions{});

    [[nodiscard]] strategy::StrategyDecision plan(const model::CanonicalModel& model,
                                                  const SolverOptions& options = SolverOptions{});

    [[nodiscard]] std::string explain_plan(const model::CanonicalModel& model,
                                           const SolverOptions& options = SolverOptions{});

    [[nodiscard]] const strategy::StrategyEngine& engine() const noexcept { return engine_; }
    [[nodiscard]] strategy::StrategyEngine& engine() noexcept { return engine_; }

private:
    strategy::StrategyEngine engine_;
    strategy::StrategyConfig default_config_;
};

}  // namespace kavach::api
