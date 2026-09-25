#pragma once

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "algorithms/lp/basis.hpp"
#include "algorithms/lp/lp_types.hpp"
#include "algorithms/lp/revised_simplex.hpp"
#include "algorithms/lp/standard_form.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"

namespace kavach::algorithms::lp {

/**
 * @brief Production Dual Simplex Engine for warm starting LP child-node re-solves.
 *
 * Implements:
 * - Inherited basis validation (dimension, structural consistency, factorizability)
 * - Dual feasibility verification of inherited basis
 * - Most-infeasible leaving variable selection rule with deterministic tie-breaking
 * - Dual ratio test with stable candidate selection and deterministic tie-breaking
 * - Farkas ray / primal infeasibility certification
 * - SparseLU basis factorization and periodic refactorization
 * - Iteration / time limits and numerical drift safeguards with explicit fallbacks
 * - Zero external solver dependencies
 */
class DualSimplexCore {
public:
    DualSimplexCore(const StandardFormLP& lp, const LPSolverOptions& options);

    /**
     * @brief Run dual simplex optimization starting from an inherited basis state.
     *
     * @param initial_basis Basis state captured from optimal parent LP solve.
     * @param telemetry Telemetry diagnostics to record iterations, pivots, etc.
     * @param resulting_basis [out] Final optimal basis if solved to optimality.
     * @return SimplexExecutionResult (OPTIMAL, INFEASIBLE, LIMIT_REACHED, NUMERICAL_FAILURE)
     */
    [[nodiscard]] SimplexExecutionResult run_warm_start(const BasisState& initial_basis,
                                                        LPSolverTelemetry& telemetry,
                                                        std::optional<BasisState>& resulting_basis);

private:
    [[nodiscard]] bool check_time_limit(
        const std::chrono::steady_clock::time_point& start_time) const;

    const StandardFormLP& lp_;
    const LPSolverOptions& options_;
    const core::SolverTolerances& tol_;
};

}  // namespace kavach::algorithms::lp
