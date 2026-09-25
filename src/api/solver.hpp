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
};

/**
 * @brief Architecture interface scaffolding for [V] Solver API.
 * Full implementation scheduled for Phase 15 per Blueprint Section 4, 17 & Section 32.
 */
class Solver {
public:
    virtual ~Solver() = default;

    [[nodiscard]] virtual SolverResult solve(const model::CanonicalModel& model,
                                             const SolverOptions& options) = 0;
};

}  // namespace kavach::api
