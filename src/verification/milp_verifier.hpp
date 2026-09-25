#pragma once

/**
 * @file milp_verifier.hpp
 * @brief Independent MILP Verification Layer [Y] — Kavach-Opt Blueprint Section 4, 6, 11, 12, 32.
 *
 * INDEPENDENCE CONTRACT
 * =====================
 * This file and its implementation (milp_verifier.cpp) MUST NOT depend on any
 * solver-internal calculation functions:
 *   - No MILP solver feasibility helpers
 *   - No B&B node data or solver tree structures
 *   - No LP solver reduced-cost or basis routines
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

// ============================================================================
// Full Structured MILP Verification Report
// ============================================================================

/**
 * @brief Complete structured verification report produced by MILPVerifier.
 */
struct MILPVerificationReport {
    // Individual checks
    CheckResult nonfinite_check;        ///< NaN / ±Inf scan of incumbent
    CheckResult bound_feasibility;      ///< Original model lb ≤ x ≤ ub
    CheckResult primal_feasibility;     ///< Original model Ax ≤/≥/= b
    CheckResult integrality;            ///< |x_j - round(x_j)| ≤ integrality_tol for int/bin
    CheckResult objective_consistency;  ///< Independent cᵀx + c₀ vs. reported objective
    CheckResult gap_consistency;        ///< Independent gap recomputation vs. reported status
    CheckResult status_consistency;     ///< Solver status vs. verification evidence

    // Aggregated metrics
    double aggregate_primal_residual{0.0};  ///< Max constraint violation
    double max_bound_violation{0.0};        ///< Max bound violation
    double max_integrality_violation{0.0};  ///< Max integrality violation
    double recomputed_objective{0.0};       ///< cᵀx + c₀ computed from scratch
    double objective_absolute_error{0.0};   ///< |recomputed - reported|
    double recomputed_gap{0.0};             ///< (|z_inc - z_bound|) / max(1, |z_inc|)

    // Status output
    core::SolverStatus original_reported_status{core::SolverStatus::UNKNOWN};
    core::SolverStatus certified_status{core::SolverStatus::UNKNOWN};
    bool status_downgraded{false};
    bool incumbent_available{false};

    // Diagnostic summary
    std::string summary;

    /**
     * @brief Returns true only if ALL applicable checks passed.
     */
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
        if (!status_consistency.passed) {
            return false;
        }
        if (original_reported_status == core::SolverStatus::OPTIMAL && !gap_consistency.passed) {
            return false;
        }
        return certified_status == original_reported_status;
    }
};

// ============================================================================
// Verification Tolerance Configuration for MILP
// ============================================================================

struct MILPVerifierTolerances {
    double primal_feasibility_tol{1e-6};  ///< Max allowable Ax - b violation
    double bound_feasibility_tol{1e-6};   ///< Max allowable lb/ub violation
    double integrality_tol{1e-5};         ///< Max allowable |x_j - round(x_j)|
    double objective_abs_tol{1e-4};       ///< Max allowable |recomputed - reported| obj
    double objective_rel_tol{1e-6};       ///< Relative objective tolerance
    double gap_tolerance{1e-4};           ///< Max allowable gap to certify OPTIMAL

    static MILPVerifierTolerances from_solver_tolerances(
        const core::SolverTolerances& tol) noexcept {
        MILPVerifierTolerances vt;
        vt.primal_feasibility_tol = tol.residual_tol;
        vt.bound_feasibility_tol = tol.residual_tol;
        vt.integrality_tol = tol.integrality_tol;
        vt.objective_abs_tol = tol.residual_tol * 1e2;
        vt.objective_rel_tol = tol.residual_tol;
        vt.gap_tolerance = 1e-4;
        return vt;
    }
};

// ============================================================================
// MILPVerifier — Main Independent MILP Verification Class
// ============================================================================

/**
 * @brief Independent MILP solution verifier.
 *
 * Recomputes primal feasibility, variable bounds, integrality, objective,
 * and optimality gap directly against the original unpresolved CanonicalModel.
 */
class MILPVerifier {
public:
    MILPVerifier() = default;
    explicit MILPVerifier(MILPVerifierTolerances tols) : tols_(tols) {}

    /**
     * @brief Verify a MILP solution.
     *
     * @param model              Original CanonicalModel (pre-presolve, unscaled).
     * @param incumbent          Primal incumbent in original variable space (size n), or empty.
     * @param best_bound         Reported best dual bound.
     * @param reported_objective Reported incumbent objective value.
     * @param reported_status    Solver reported status.
     * @param reported_gap       Solver reported optimality gap (negative = compute afresh).
     * @return Full structured MILPVerificationReport.
     */
    [[nodiscard]] MILPVerificationReport verify(const model::CanonicalModel& model,
                                                const std::vector<double>& incumbent,
                                                double best_bound, double reported_objective,
                                                core::SolverStatus reported_status,
                                                double reported_gap = -1.0) const;

    [[nodiscard]] MILPVerificationReport verify(const model::CanonicalModel& model,
                                                const std::vector<double>& incumbent,
                                                double best_bound, double reported_objective,
                                                core::SolverStatus reported_status,
                                                const core::SolverTolerances& solver_tols,
                                                double reported_gap = -1.0) const;

    [[nodiscard]] const MILPVerifierTolerances& tolerances() const noexcept { return tols_; }
    void set_tolerances(MILPVerifierTolerances tols) noexcept { tols_ = tols; }

private:
    MILPVerifierTolerances tols_;

    [[nodiscard]] CheckResult check_nonfinite(const std::vector<double>& x) const;

    [[nodiscard]] CheckResult check_bound_feasibility(const model::CanonicalModel& model,
                                                      const std::vector<double>& x) const;

    [[nodiscard]] CheckResult check_primal_feasibility(const model::CanonicalModel& model,
                                                       const std::vector<double>& x) const;

    [[nodiscard]] CheckResult check_integrality(const model::CanonicalModel& model,
                                                const std::vector<double>& x) const;

    [[nodiscard]] CheckResult check_objective(const model::CanonicalModel& model,
                                              const std::vector<double>& x, double reported_obj,
                                              double& recomputed_obj_out) const;

    [[nodiscard]] CheckResult check_gap(double recomputed_obj, double best_bound,
                                        model::ObjectiveSense sense,
                                        core::SolverStatus reported_status,
                                        double& recomputed_gap_out) const;

    [[nodiscard]] CheckResult check_status_consistency(
        core::SolverStatus reported_status, bool has_incumbent, const CheckResult& nonfinite,
        const CheckResult& bounds, const CheckResult& primal, const CheckResult& integrality,
        const CheckResult& obj, const CheckResult& gap,
        core::SolverStatus& certified_status_out) const;

    [[nodiscard]] static double compute_row_activity(const model::CanonicalModel& model,
                                                     std::size_t row_idx,
                                                     const std::vector<double>& x);
};

}  // namespace kavach::verification
