#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "backend/execution_backend.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"

namespace kavach::algorithms::milp {

/**
 * @brief Result produced by the primal heuristic.
 */
struct PrimalHeuristicResult {
    bool found_feasible{false};
    std::vector<double> solution;
    double objective{0.0};
    std::size_t repair_rounds_used{0};
};

/**
 * @brief Feasibility-oriented rounding and bounded repair primal heuristic.
 *
 * Implements deterministic integer rounding followed by bounded linear deficit
 * repair on continuous slack variables to restore constraint feasibility.
 */
class PrimalHeuristic {
public:
    explicit PrimalHeuristic(const core::SolverTolerances& tols, std::size_t max_repair_rounds = 5);

    /**
     * @brief Attempt to construct a certified feasible integer solution from an LP point.
     *
     * @param model Base canonical model.
     * @param x_lp Fractional solution vector from LP relaxation.
     * @param backend Optional execution backend for batched candidate activity evaluation.
     * @return PrimalHeuristicResult containing candidate solution if certified feasible.
     */
    [[nodiscard]] PrimalHeuristicResult search(const model::CanonicalModel& model,
                                               const std::vector<double>& x_lp,
                                               backend::ExecutionBackend* backend = nullptr) const;

private:
    core::SolverTolerances tols_;
    std::size_t max_repair_rounds_{5};
};

}  // namespace kavach::algorithms::milp
