#pragma once

/**
 * @file active_set.hpp
 * @brief Primal Active-Set Convex Quadratic Programming (QP) Solver [H].
 *
 * Architecture strictly adheres to Blueprint v2 Sections 4, 6, 11, 12, and 32.
 * Solves convex quadratic programs:
 *   minimize    1/2 x^T Q x + c^T x + c_0
 *   subject to  A_i x <= b_i, A_i x >= b_i, A_i x == b_i
 *               lb <= x <= ub
 *   where Q is positive semidefinite.
 *
 * Solved via sparse KKT systems and SparseLU with zero third-party dependencies.
 */

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"
#include "numerics/lu_factorization.hpp"
#include "numerics/sparse_matrix.hpp"
#include "verification/qp_verifier.hpp"

namespace kavach::algorithms::qp {

/**
 * @brief Telemetry for convex QP solves.
 */
struct QPTelemetry {
    std::size_t num_variables{0};
    std::size_t num_constraints{0};
    std::size_t q_nonzeros{0};
    std::size_t a_nonzeros{0};
    std::size_t iterations{0};
    std::size_t active_constraints{0};
    std::size_t constraints_added{0};
    std::size_t constraints_removed{0};
    std::size_t kkt_solves{0};
    std::size_t refactorizations{0};
    double max_stationarity_residual{0.0};
    double max_primal_residual{0.0};
    double max_dual_residual{0.0};
    double max_complementarity_residual{0.0};
    double solve_time_ms{0.0};
    std::string error_message;
};

/**
 * @brief Configuration options for the active-set QP solver.
 */
struct QPSolverOptions {
    std::size_t max_iterations{1000};
    double time_limit_seconds{0.0};  ///< <= 0 means unlimited
    bool enable_presolve{false};
    bool enable_scaling{true};
    bool verify_with_independent_verifier{true};
    core::SolverTolerances tolerances;

    static QPSolverOptions from_tolerances(const core::SolverTolerances& tols) noexcept {
        QPSolverOptions opts;
        opts.tolerances = tols;
        return opts;
    }
};

/**
 * @brief Result of a QP solve.
 */
struct QPSolution {
    core::SolverStatus status{core::SolverStatus::UNKNOWN};
    double objective_value{0.0};
    std::vector<double> primal_solution;
    std::vector<double> dual_solution;      ///< Multipliers for model constraints
    std::vector<double> bound_multipliers;  ///< Multipliers for variable bounds (size n)
    double kkt_residual{0.0};
    std::size_t iterations{0};
    QPTelemetry telemetry;
    verification::QPVerificationReport verification_report;

    [[nodiscard]] bool is_optimal() const noexcept { return status == core::SolverStatus::OPTIMAL; }
};

/**
 * @brief Abstract interface for [H] QP Solver (Convex Active-Set).
 */
class ActiveSetQPSolver {
public:
    virtual ~ActiveSetQPSolver() = default;

    [[nodiscard]] virtual QPSolution solve(const model::CanonicalModel& model,
                                           const core::SolverTolerances& tolerances) = 0;

    [[nodiscard]] virtual QPSolution solve(const model::CanonicalModel& model,
                                           const QPSolverOptions& options = {}) = 0;
};

/**
 * @brief Classification of constraint types in the active set.
 */
enum class ActiveConstraintKind : std::uint8_t { MODEL_CONSTRAINT, LOWER_BOUND, UPPER_BOUND };

/**
 * @brief Active constraint descriptor in working set W.
 */
struct ActiveConstraint {
    ActiveConstraintKind kind{ActiveConstraintKind::MODEL_CONSTRAINT};
    std::size_t index{0};     ///< Constraint index (0..m-1) or variable index (0..n-1)
    double orientation{1.0};  ///< +1.0 for <= / ub, -1.0 for >= / lb
    bool is_equality{false};  ///< True if permanent equality (cannot be removed)

    bool operator==(const ActiveConstraint& other) const noexcept {
        return kind == other.kind && index == other.index;
    }
};

/**
 * @brief Production Primal Active-Set QP Solver for Convex Quadratic Programs.
 */
class PrimalActiveSetQPSolver : public ActiveSetQPSolver {
public:
    explicit PrimalActiveSetQPSolver(QPSolverOptions default_options = {})
        : options_(std::move(default_options)) {}

    [[nodiscard]] QPSolution solve(const model::CanonicalModel& model,
                                   const core::SolverTolerances& tolerances) override {
        QPSolverOptions opts = options_;
        opts.tolerances = tolerances;
        return solve(model, opts);
    }

    [[nodiscard]] QPSolution solve(const model::CanonicalModel& model,
                                   const QPSolverOptions& options = {}) override;

    // --- Exposed Static Helpers (used in unit tests) ---

    /// Build symmetric sparse Hessian Q from CanonicalModel quadratic terms: 1/2 x^T Q x
    [[nodiscard]] static numerics::CsrMatrix build_hessian(const model::CanonicalModel& model);

    /// Verify positive semidefiniteness of Q via diagonal-pivoted Cholesky factorization
    [[nodiscard]] static bool check_psd(const numerics::CsrMatrix& Q, double tol,
                                        std::string* error_msg = nullptr);

    /// Evaluate quadratic objective f(x) = 1/2 x^T Q x + c^T x + c0
    [[nodiscard]] static double evaluate_objective(const model::CanonicalModel& model,
                                                   const std::vector<double>& x);

private:
    QPSolverOptions options_;

    // Core active-set engine on unscaled/unpresolved or reduced model
    [[nodiscard]] QPSolution solve_internal(
        const model::CanonicalModel& model, const QPSolverOptions& opts,
        const std::chrono::steady_clock::time_point& start_time) const;

    // Feasible point finder
    [[nodiscard]] bool find_initial_feasible_point(const model::CanonicalModel& model,
                                                   const core::SolverTolerances& tols,
                                                   std::vector<double>& x0) const;
};

}  // namespace kavach::algorithms::qp
