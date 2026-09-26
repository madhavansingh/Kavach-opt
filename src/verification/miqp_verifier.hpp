#pragma once

/**
 * @file miqp_verifier.hpp
 * @brief Independent Convex MIQP Verification Layer [Y] — Kavach-Opt Sovereign Engine.
 *
 * INDEPENDENCE CONTRACT
 * =====================
 * This file and its implementation (miqp_verifier.cpp) MUST NOT depend on any
 * solver-internal calculation functions:
 *   - No MIQP solver feasibility helpers
 *   - No B&B node data or solver tree structures
 *   - No QP active-set bookkeeping
 *   - No presolve internal residual calculations
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
 * @brief Named verification tolerances for independent convex MIQP verification.
 */
struct MIQPVerifierTolerances {
    double primal_feasibility_tol{1e-6};  ///< Max allowable Ax - b violation
    double bound_feasibility_tol{1e-6};   ///< Max allowable lb/ub violation
    double integrality_tol{1e-5};         ///< Max allowable |x_j - round(x_j)|
    double objective_abs_tol{1e-4};       ///< Max allowable |recomputed - reported| obj
    double objective_rel_tol{1e-6};       ///< Relative objective tolerance
    double gap_tolerance{1e-4};           ///< Max allowable gap to certify OPTIMAL
    double psd_tol{1e-9};                 ///< Positive semidefiniteness tolerance

    static MIQPVerifierTolerances from_solver_tolerances(
        const core::SolverTolerances& tols = {}) noexcept {
        MIQPVerifierTolerances vt;
        vt.primal_feasibility_tol = tols.primal_feasibility_tol;
        vt.bound_feasibility_tol = tols.primal_feasibility_tol;
        vt.integrality_tol = tols.integrality_tol;
        vt.objective_abs_tol = tols.residual_tol;
        vt.objective_rel_tol = 1e-6;
        vt.gap_tolerance = 1e-4;
        vt.psd_tol = tols.pivot_tol;
        return vt;
    }
};

/**
 * @brief Structured verification report for a convex MIQP solution.
 */
struct MIQPVerificationReport {
    // Individual checks
    CheckResult nonfinite_check;     ///< NaN / ±Inf scan of incumbent
    CheckResult bound_feasibility;   ///< Original model lb <= x <= ub
    CheckResult primal_feasibility;  ///< Original model Ax <=/=/ >= b
    CheckResult integrality;         ///< |x_j - round(x_j)| <= integrality_tol for int/bin
    CheckResult
        objective_consistency;       ///< Independent 1/2 x^T Q x + c^T x + c0 vs reported objective
    CheckResult gap_consistency;     ///< Independent gap recomputation vs reported status
    CheckResult status_consistency;  ///< Solver status vs verification evidence
    CheckResult psd_check;           ///< Independent convexity check

    // Aggregated metrics
    double aggregate_primal_residual{0.0};  ///< Max constraint violation
    double max_bound_violation{0.0};        ///< Max bound violation
    double max_integrality_violation{0.0};  ///< Max integrality violation
    double recomputed_objective{0.0};       ///< 1/2 x^T Q x + c^T x + c0 computed from scratch
    double objective_absolute_error{0.0};   ///< |recomputed - reported|
    double recomputed_gap{0.0};             ///< (|z_inc - z_bound|) / max(1, |z_inc|)

    // Status output
    core::SolverStatus original_reported_status{core::SolverStatus::UNKNOWN};
    core::SolverStatus certified_status{core::SolverStatus::UNKNOWN};
    bool status_downgraded{false};
    bool incumbent_available{false};

    // Diagnostic summary
    std::string summary;

    [[nodiscard]] bool fully_verified() const noexcept {
        if (!nonfinite_check.passed) {
            return false;
        }
        if (incumbent_available) {
            if (!bound_feasibility.passed || !primal_feasibility.passed || !integrality.passed ||
                !objective_consistency.passed) {
                return false;
            }
        }
        if (!psd_check.passed) {
            return false;
        }
        if (!status_consistency.passed) {
            return false;
        }
        if (original_reported_status == core::SolverStatus::OPTIMAL && !gap_consistency.passed) {
            return false;
        }
        return certified_status == original_reported_status;
    }
};

/**
 * @brief Independent verifier for Convex Mixed-Integer Quadratic Programming solutions.
 */
class MIQPVerifier {
public:
    explicit MIQPVerifier(MIQPVerifierTolerances tolerances = {})
        : tolerances_(std::move(tolerances)) {}

    [[nodiscard]] MIQPVerificationReport verify(const model::CanonicalModel& model,
                                                const std::vector<double>& incumbent_solution,
                                                double best_bound, double reported_objective,
                                                core::SolverStatus reported_status,
                                                const core::SolverTolerances& solver_tols = {},
                                                double solver_reported_gap = -1.0) const;

    [[nodiscard]] const MIQPVerifierTolerances& tolerances() const noexcept { return tolerances_; }
    void set_tolerances(MIQPVerifierTolerances tols) noexcept { tolerances_ = std::move(tols); }

private:
    MIQPVerifierTolerances tolerances_;

    [[nodiscard]] CheckResult check_nonfinite(const std::vector<double>& incumbent) const;
    [[nodiscard]] CheckResult check_bounds(const model::CanonicalModel& model,
                                           const std::vector<double>& incumbent) const;
    [[nodiscard]] CheckResult check_primal_feasibility(const model::CanonicalModel& model,
                                                       const std::vector<double>& incumbent) const;
    [[nodiscard]] CheckResult check_integrality(const model::CanonicalModel& model,
                                                const std::vector<double>& incumbent) const;
    [[nodiscard]] CheckResult check_objective(const model::CanonicalModel& model,
                                              const std::vector<double>& incumbent,
                                              double reported_objective,
                                              double& recomputed_obj_out) const;
    [[nodiscard]] CheckResult check_convexity(const model::CanonicalModel& model) const;
};

}  // namespace kavach::verification
