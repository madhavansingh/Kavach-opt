#pragma once

/**
 * @file qp_verifier.hpp
 * @brief Independent Convex QP Verification Layer [Y] — Kavach-Opt Blueprint Section 4, 6, 12, 32.
 *
 * INDEPENDENCE CONTRACT
 * =====================
 * This file and its implementation (qp_verifier.cpp) MUST NOT depend on any
 * solver-internal calculation functions:
 *   - No QP solver feasibility helpers
 *   - No QP active-set bookkeeping
 *   - No solver-specific residual functions
 *   - No presolve internal residual calculations
 *   - No numerics LU / SpMV library helpers
 *
 * Every mathematical check is performed afresh from the original CanonicalModel
 * using only elementary arithmetic and the named tolerances in SolverTolerances.
 */

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"
#include "verification/lp_verifier.hpp"

namespace kavach::verification {

/**
 * @brief Named verification tolerances for independent convex QP verification.
 */
struct QPVerifierTolerances {
    double primal_feasibility_tol{1e-7};
    double dual_feasibility_tol{1e-7};
    double optimality_tol{1e-7};
    double residual_tol{1e-6};
    double objective_tol{1e-6};
    double psd_tol{1e-9};

    static QPVerifierTolerances from_solver_tolerances(
        const core::SolverTolerances& tols = {}) noexcept {
        QPVerifierTolerances vt;
        vt.primal_feasibility_tol = tols.primal_feasibility_tol;
        vt.dual_feasibility_tol = tols.dual_feasibility_tol;
        vt.optimality_tol = tols.optimality_tol;
        vt.residual_tol = tols.residual_tol;
        vt.objective_tol = tols.residual_tol;
        vt.psd_tol = tols.pivot_tol;
        return vt;
    }
};

/**
 * @brief Structured verification report for a convex QP solution.
 */
struct QPVerificationReport {
    // Individual checks
    CheckResult nonfinite_check;
    CheckResult bound_feasibility;
    CheckResult primal_feasibility;
    CheckResult stationarity;
    CheckResult dual_feasibility;
    CheckResult complementary_slackness;
    CheckResult objective_consistency;
    CheckResult psd_check;
    CheckResult status_consistency;

    // Aggregated metrics
    double aggregate_primal_residual{0.0};
    double aggregate_dual_residual{0.0};
    double aggregate_stationarity_residual{0.0};
    double aggregate_complementarity_residual{0.0};
    double recomputed_objective{0.0};
    double objective_absolute_error{0.0};
    double objective_relative_error{0.0};

    // Status outcomes
    core::SolverStatus original_reported_status{core::SolverStatus::UNKNOWN};
    core::SolverStatus certified_status{core::SolverStatus::UNKNOWN};
    bool status_downgraded{false};
    std::string summary;

    [[nodiscard]] bool fully_verified() const noexcept {
        return certified_status == core::SolverStatus::OPTIMAL && !status_downgraded;
    }
};

/**
 * @brief Independent verifier for Convex Quadratic Programming solutions.
 */
class QPVerifier {
public:
    explicit QPVerifier(QPVerifierTolerances tolerances = {})
        : tolerances_(std::move(tolerances)) {}

    [[nodiscard]] QPVerificationReport verify(const model::CanonicalModel& model,
                                              const std::vector<double>& primal_solution,
                                              const std::vector<double>& dual_multipliers,
                                              double reported_objective,
                                              core::SolverStatus reported_status,
                                              const core::SolverTolerances& solver_tols = {}) const;

    [[nodiscard]] const QPVerifierTolerances& tolerances() const noexcept { return tolerances_; }
    void set_tolerances(QPVerifierTolerances tols) noexcept { tolerances_ = std::move(tols); }

private:
    QPVerifierTolerances tolerances_;

    [[nodiscard]] CheckResult check_nonfinite(const std::vector<double>& primal,
                                              const std::vector<double>& dual) const;

    [[nodiscard]] CheckResult check_bounds(const model::CanonicalModel& model,
                                           const std::vector<double>& primal) const;

    [[nodiscard]] CheckResult check_primal_feasibility(
        const model::CanonicalModel& model, const std::vector<double>& primal,
        std::vector<double>& constraint_activities) const;

    [[nodiscard]] CheckResult check_stationarity_and_duals(
        const model::CanonicalModel& model, const std::vector<double>& primal,
        const std::vector<double>& dual, const std::vector<double>& constraint_activities,
        CheckResult& dual_feas_result, CheckResult& cs_result) const;

    [[nodiscard]] CheckResult check_objective(const model::CanonicalModel& model,
                                              const std::vector<double>& primal,
                                              double reported_objective,
                                              double& recomputed_obj) const;

    [[nodiscard]] CheckResult check_psd(const model::CanonicalModel& model) const;
};

}  // namespace kavach::verification
