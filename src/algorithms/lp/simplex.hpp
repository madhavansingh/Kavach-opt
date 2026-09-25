#pragma once

#include "algorithms/lp/lp_types.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"

namespace kavach::algorithms::lp {

/**
 * @brief Architecture interface scaffolding for [F] LP Solver (Revised Simplex).
 */
class RevisedSimplexSolver {
public:
    virtual ~RevisedSimplexSolver() = default;

    [[nodiscard]] virtual LPSolution solve(const model::CanonicalModel& model,
                                           const core::SolverTolerances& tolerances) = 0;
};

/**
 * @brief Production Revised Simplex LP Solver for Kavach-Opt.
 *
 * Implements:
 * - Two-phase revised simplex algorithm
 * - SparseLU basis factorization with partial pivoting
 * - Dantzig entering rule with Bland watchdog anti-cycling fallback
 * - Minimum-ratio test with deterministic Bland tie-breaking
 * - Bounded and free variable transformations
 * - Geometric-mean scaling integration
 * - Presolve integration with complete primal and objective reconstruction
 * - Strict solver status reporting (OPTIMAL, INFEASIBLE, UNBOUNDED, LIMIT_REACHED,
 * NUMERICAL_FAILURE)
 * - Zero third-party solver or factorization library dependencies
 */
class RevisedSimplex : public RevisedSimplexSolver {
public:
    RevisedSimplex() = default;
    explicit RevisedSimplex(LPSolverOptions options) : options_(std::move(options)) {}

    [[nodiscard]] LPSolution solve(const model::CanonicalModel& model,
                                   const core::SolverTolerances& tolerances) override;

    [[nodiscard]] LPSolution solve(const model::CanonicalModel& model,
                                   const LPSolverOptions& options);

    [[nodiscard]] LPSolution solve(const model::CanonicalModel& model);

    [[nodiscard]] const LPSolverOptions& options() const noexcept { return options_; }
    void set_options(LPSolverOptions options) noexcept { options_ = std::move(options); }

private:
    LPSolverOptions options_;
};

}  // namespace kavach::algorithms::lp
