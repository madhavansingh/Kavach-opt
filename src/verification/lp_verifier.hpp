#pragma once

/**
 * @file lp_verifier.hpp
 * @brief Independent LP Verification Layer [Y] — Kavach-Opt Blueprint Section 4, 6, 12, 32.
 *
 * INDEPENDENCE CONTRACT
 * =====================
 * This file and its implementation (lp_verifier.cpp) MUST NOT depend on any
 * solver-internal calculation functions:
 *   - No LP solver feasibility helpers
 *   - No LP solver reduced-cost routines
 *   - No LP solver objective evaluators
 *   - No solver-specific residual functions
 *   - No basis bookkeeping
 *   - No simplex stopping criteria
 *   - No presolve internal residual calculations
 *
 * Every mathematical check is performed afresh from the original CanonicalModel
 * using only elementary arithmetic and the named tolerances in SolverTolerances.
 *
 * Dependencies allowed:
 *   - model/model.hpp         (read-only access to constraint/bound/objective data)
 *   - core/status.hpp         (SolverStatus taxonomy)
 *   - core/tolerances.hpp     (named tolerance constants)
 *
 * Dependencies strictly FORBIDDEN inside verification computations:
 *   - algorithms/lp/[all]     (any LP solver internals)
 *   - presolve/[all]           (any presolve residual helpers)
 *   - numerics/[all]           (SpMV, LU, scaling helpers — recompute directly)
 */

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"

namespace kavach::verification {

// ============================================================================
// Per-Check Detailed Result
// ============================================================================

/**
 * @brief Detailed result for a single verification check.
 *
 * Never exposes only a boolean — always carries the recomputed residual,
 * configured tolerance, and a human-readable diagnostic.
 */
struct CheckResult {
    bool passed{false};
    double max_violation{0.0};   ///< Maximum absolute violation found
    double tolerance_used{0.0};  ///< Tolerance that was applied
    std::string diagnostic;      ///< Human-readable description of outcome
    std::size_t worst_index{std::numeric_limits<std::size_t>::max()};  ///< Row/var of max violation
};

// ============================================================================
// Full Structured Verification Report
// ============================================================================

/**
 * @brief Complete structured verification report produced by LPVerifier.
 *
 * Each field captures a distinct mathematical verification claim with enough
 * information to explain what was checked, what was found, and whether it passed.
 */
struct LPVerificationReport {
    // Individual checks
    CheckResult primal_feasibility;       ///< Ax ≤/≥/= b and lb ≤ x ≤ ub
    CheckResult bound_feasibility;        ///< lb ≤ x ≤ ub independently
    CheckResult dual_feasibility;         ///< Reduced costs ≥ 0 (or correctly signed)
    CheckResult complementary_slackness;  ///< (b - Ax)ᵢ · yᵢ ≈ 0
    CheckResult objective_consistency;    ///< Independent cᵀx vs. reported objective
    CheckResult nonfinite_check;          ///< NaN / ±Inf scan of solution vectors
    CheckResult status_consistency;       ///< Solver status vs. verification evidence

    // Aggregated residuals
    double aggregate_primal_residual{0.0};  ///< Max over all constraint violations
    double aggregate_dual_residual{0.0};    ///< Max over all dual violations
    double recomputed_objective{0.0};       ///< cᵀx + c₀ computed from scratch
    double objective_absolute_error{0.0};   ///< |recomputed - reported|
    double objective_relative_error{0.0};   ///< Relative error (if |reported| > ε)

    // Status output
    core::SolverStatus original_reported_status{core::SolverStatus::UNKNOWN};
    core::SolverStatus certified_status{core::SolverStatus::UNKNOWN};
    bool status_downgraded{false};  ///< true if verifier overrode solver status

    // Availability flags (for non-optimal results or missing dual info)
    bool dual_info_available{false};
    bool certificate_validation_available{false};

    // Diagnostic string for summary reporting
    std::string summary;

    /**
     * @brief Returns true only if ALL applicable checks passed.
     *
     * Checks that are not applicable (e.g. dual for INFEASIBLE) are excluded.
     */
    [[nodiscard]] bool fully_verified() const noexcept;
};

// ============================================================================
// Verification Tolerance Configuration
// ============================================================================

/**
 * @brief Verifier-specific tolerance overrides.
 *
 * The verifier uses its own acceptance thresholds, which may differ from the
 * solver's internal operational tolerances. This enforces the independence
 * boundary: a slightly looser solver tolerance does not automatically widen
 * the verifier's acceptance window.
 */
struct VerifierTolerances {
    double primal_feasibility_tol{1e-6};  ///< Max allowable Ax-b violation
    double bound_feasibility_tol{1e-6};   ///< Max allowable lb/ub violation
    double dual_feasibility_tol{1e-6};    ///< Max allowable dual/reduced-cost violation
    double complementarity_tol{1e-6};     ///< Max allowable complementary slackness residual
    double objective_abs_tol{1e-4};       ///< Max allowable |recomputed - reported| objective
    double objective_rel_tol{1e-6};       ///< Relative tolerance when |reported| > 1
    double nonfinite_tol{0.0};            ///< Tolerance for NaN/Inf detection (always strict)

    /**
     * @brief Construct verifier tolerances from solver tolerances (tighter defaults).
     */
    static VerifierTolerances from_solver_tolerances(const core::SolverTolerances& tol) noexcept {
        VerifierTolerances vt;
        // Use residual_tol (defined for independent verifier) and primal_feasibility_tol
        vt.primal_feasibility_tol = tol.residual_tol;
        vt.bound_feasibility_tol = tol.residual_tol;
        vt.dual_feasibility_tol = tol.residual_tol;
        vt.complementarity_tol = tol.residual_tol;
        vt.objective_abs_tol = tol.residual_tol * 1e2;  // slightly looser for objective
        vt.objective_rel_tol = tol.residual_tol;
        return vt;
    }
};

// ============================================================================
// MILP / QP Extension Hooks (stub — implementations pending)
// ============================================================================

/** @brief Extension point for MILP integrality verification (not yet implemented). */
struct MILPVerificationHints {
    bool integrality_check_enabled{false};  ///< Will be true when MILP solver exists
    double best_bound{0.0};
    double gap_tolerance{1e-4};
};

/** @brief Extension point for QP KKT residual verification (not yet implemented). */
struct QPVerificationHints {
    bool kkt_check_enabled{false};  ///< Will be true when QP solver exists
};

// ============================================================================
// LPVerifier — Main Verification Class
// ============================================================================

/**
 * @brief Independent LP solution verifier.
 *
 * Performs all mathematical checks afresh from the original CanonicalModel
 * without reusing any solver-internal computation.
 *
 * Usage:
 * @code
 *   LPVerifier verifier;
 *   auto report = verifier.verify(original_model, sol, tols);
 *   if (!report.fully_verified()) { ... }
 * @endcode
 */
class LPVerifier {
public:
    LPVerifier() = default;
    explicit LPVerifier(VerifierTolerances tols) : tols_(tols) {}

    /**
     * @brief Verify an LP solution against the original (pre-presolve) model.
     *
     * @param model     Original CanonicalModel (pre-presolve, unscaled).
     * @param x         Primal solution in original variable space (size == num_variables).
     * @param y         Dual multipliers in original constraint space (size == num_constraints),
     *                  or empty if unavailable.
     * @param rc        Reduced costs in original variable space, or empty if unavailable.
     * @param reported_obj  Objective value as reported by the solver.
     * @param reported_status  Solver-claimed status.
     * @return LPVerificationReport with full structured diagnostics.
     */
    [[nodiscard]] LPVerificationReport verify(const model::CanonicalModel& model,
                                              const std::vector<double>& x,
                                              const std::vector<double>& y,
                                              const std::vector<double>& rc, double reported_obj,
                                              core::SolverStatus reported_status) const;

    /**
     * @brief Convenience overload that derives verifier tolerances from solver tolerances.
     */
    [[nodiscard]] LPVerificationReport verify(const model::CanonicalModel& model,
                                              const std::vector<double>& x,
                                              const std::vector<double>& y,
                                              const std::vector<double>& rc, double reported_obj,
                                              core::SolverStatus reported_status,
                                              const core::SolverTolerances& solver_tols) const;

    [[nodiscard]] const VerifierTolerances& tolerances() const noexcept { return tols_; }
    void set_tolerances(VerifierTolerances tols) { tols_ = std::move(tols); }

private:
    VerifierTolerances tols_;

    // --- Individual check implementations ---
    // All methods are const-qualified and stateless; they only read model + solution data.

    /** Scan all solution vectors for NaN / ±Inf. */
    [[nodiscard]] CheckResult check_nonfinite(const std::vector<double>& x,
                                              const std::vector<double>& y,
                                              const std::vector<double>& rc) const;

    /** Independently verify lb ≤ x ≤ ub for all original variables. */
    [[nodiscard]] CheckResult check_bound_feasibility(const model::CanonicalModel& model,
                                                      const std::vector<double>& x) const;

    /** Independently compute Ax residuals and check constraint senses. */
    [[nodiscard]] CheckResult check_primal_feasibility(const model::CanonicalModel& model,
                                                       const std::vector<double>& x) const;

    /** Independently verify dual feasibility (reduced costs for each variable). */
    [[nodiscard]] CheckResult check_dual_feasibility(const model::CanonicalModel& model,
                                                     const std::vector<double>& x,
                                                     const std::vector<double>& y,
                                                     const std::vector<double>& rc) const;

    /** Independently verify complementary slackness conditions. */
    [[nodiscard]] CheckResult check_complementary_slackness(const model::CanonicalModel& model,
                                                            const std::vector<double>& x,
                                                            const std::vector<double>& y) const;

    /** Independently recompute cᵀx + c₀ and compare with reported objective. */
    [[nodiscard]] CheckResult check_objective(const model::CanonicalModel& model,
                                              const std::vector<double>& x, double reported_obj,
                                              double& recomputed_obj_out) const;

    /** Determine whether reported_status is supported by verification evidence. */
    [[nodiscard]] CheckResult check_status_consistency(
        core::SolverStatus reported_status, const CheckResult& primal_feas,
        const CheckResult& bound_feas, const CheckResult& dual_feas, const CheckResult& cs,
        const CheckResult& obj_check, const CheckResult& nonfinite) const;

    /** Helper: compute Aᵢx (row dot-product) from CanonicalModel. */
    [[nodiscard]] static double compute_row_activity(const model::CanonicalModel& model,
                                                     std::size_t row_idx,
                                                     const std::vector<double>& x);
};

// ============================================================================
// Free helper: run verification on an LPSolution (convenience wrapper)
// ============================================================================

}  // namespace kavach::verification
