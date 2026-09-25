#pragma once

#include "core/tolerances.hpp"
#include "model/model.hpp"

namespace kavach::strategy {

struct ProblemCharacteristics {
    std::size_t num_variables{0};
    std::size_t num_constraints{0};
    std::size_t num_nonzeros{0};
    double density{0.0};
    double integer_variable_ratio{0.0};
    double coefficient_min_nonzero{0.0};
    double coefficient_max_nonzero{0.0};
    double estimated_condition_number{1.0};
};

struct SolverPlan {
    bool enable_presolve{true};
    bool enable_scaling{true};
    bool prefer_gpu_for_batched_spmv{false};
    core::SolverTolerances tuned_tolerances;
};

/**
 * @brief Architecture interface scaffolding for [T] Solver Strategy Layer.
 * Full implementation scheduled for Phase 14 per Blueprint Section 4, 16 & Section 32.
 * Rule-based, deterministic problem-characteristic to setting mapping.
 */
class SolverStrategy {
public:
    virtual ~SolverStrategy() = default;

    [[nodiscard]] virtual ProblemCharacteristics analyze(
        const model::CanonicalModel& model) const = 0;
    [[nodiscard]] virtual SolverPlan plan(const ProblemCharacteristics& characteristics) const = 0;
};

}  // namespace kavach::strategy
