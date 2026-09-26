#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "backend/execution_backend.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"
#include "strategy/strategy.hpp"
#include "verification/verifier.hpp"

namespace kavach::api {

enum class ProblemType { AUTO, LP, MILP, QP };

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
    strategy::StrategyDecision strategy_decision;
};

/**
 * @brief Abstract solver interface.
 */
class Solver {
public:
    virtual ~Solver() = default;

    [[nodiscard]] virtual SolverResult solve(const model::CanonicalModel& model,
                                             const SolverOptions& options) = 0;
};

/**
 * @brief Production programmatic solver implementation for Kavach-Opt.
 * Integrates strategy selection, core numerical algorithms, independent
 * verification, and telemetry.
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
