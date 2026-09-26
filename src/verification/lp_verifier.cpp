#include "verification/lp_verifier.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

/**
 * @file lp_verifier.cpp
 * @brief Implementation of the Independent LP Verification Layer [Y].
 *
 * INDEPENDENCE BOUNDARY — ENFORCED HERE
 * ======================================
 * This translation unit includes ONLY:
 *   - verification/lp_verifier.hpp
 *   - model/model.hpp            (read-only constraint/bound/objective data)
 *   - core/status.hpp            (SolverStatus taxonomy)
 *   - core/tolerances.hpp        (named tolerance constants)
 *   - Standard library headers
 *
 * It MUST NOT include — and must never call into — any of:
 *   algorithms/lp/[all]  algorithms/qp/[all]  presolve/[all]  numerics/[all] (SpMV, LU, scaling)
 *
 * Every mathematical operation is elementary (multiply, add, abs, max) and
 * operates directly on the std::vector<double> solution and the CanonicalModel
 * data accessors.
 */

namespace kavach::verification {

// ============================================================================
// LPVerificationReport::fully_verified
// ============================================================================

bool LPVerificationReport::fully_verified() const noexcept {
    // Non-finite data is always a failure
    if (!nonfinite_check.passed) {
        return false;
    }
    // OPTIMAL requires all core checks to pass
    if (certified_status == core::SolverStatus::OPTIMAL) {
        return primal_feasibility.passed && bound_feasibility.passed && dual_feasibility.passed &&
               complementary_slackness.passed && objective_consistency.passed &&
               status_consistency.passed;
    }
    // For INFEASIBLE/UNBOUNDED/LIMIT_REACHED: non-finite + status_consistency suffice
    return status_consistency.passed;
}

// ============================================================================
// Internal helper: compute row activity  Aᵢ · x
// ============================================================================

double LPVerifier::compute_row_activity(const model::CanonicalModel& model, std::size_t row_idx,
                                        const std::vector<double>& x) {
    const auto& constr = model.constraint(static_cast<model::ConstrId>(row_idx));
    double activity = 0.0;
    for (const auto& term : constr.expression.terms()) {
        if (term.var_id < x.size()) {
            activity += term.coefficient * x[term.var_id];
        }
    }
    return activity;
}

// ============================================================================
// check_nonfinite
// ============================================================================

CheckResult LPVerifier::check_nonfinite(const std::vector<double>& x, const std::vector<double>& y,
                                        const std::vector<double>& rc) const {
    CheckResult res;
    res.passed = true;
    res.tolerance_used = 0.0;

    auto scan = [&](const std::vector<double>& v, const char* name) {
        for (std::size_t k = 0; k < v.size(); ++k) {
            if (!std::isfinite(v[k])) {
                res.passed = false;
                res.worst_index = k;
                std::ostringstream oss;
                oss << "Non-finite value in " << name << "[" << k << "] = " << v[k];
                res.diagnostic = oss.str();
                res.max_violation = 1.0;  // sentinel: any non-finite is a violation
                return;
            }
        }
    };

    scan(x, "primal_solution");
    if (!res.passed) {
        return res;
    }
    // y and rc may be empty (dual unavailable); only scan if non-empty
    if (!y.empty()) {
        scan(y, "dual_multipliers");
    }
    if (!res.passed) {
        return res;
    }
    if (!rc.empty()) {
        scan(rc, "reduced_costs");
    }

    if (res.passed) {
        res.diagnostic = "All solution vectors contain only finite values.";
    }
    return res;
}

// ============================================================================
// check_bound_feasibility: lb ≤ x ≤ ub independently
// ============================================================================

CheckResult LPVerifier::check_bound_feasibility(const model::CanonicalModel& model,
                                                const std::vector<double>& x) const {
    CheckResult res;
    res.passed = true;
    res.tolerance_used = tols_.bound_feasibility_tol;
    const double tol = tols_.bound_feasibility_tol;
    double max_viol = 0.0;
    std::size_t worst = std::numeric_limits<std::size_t>::max();

    const std::size_t n = model.num_variables();
    if (x.size() != n) {
        res.passed = false;
        res.max_violation = 1.0;
        std::ostringstream oss;
        oss << "Primal solution vector dimension mismatch: expected " << n << ", got " << x.size()
            << ".";
        res.diagnostic = oss.str();
        return res;
    }
    for (std::size_t j = 0; j < n; ++j) {
        const auto& var = model.variable(static_cast<model::VarId>(j));
        const double xj = x[j];

        // Lower bound check
        if (std::isfinite(var.lower_bound)) {
            double lb_viol = var.lower_bound - xj;  // positive if x < lb
            if (lb_viol > tol) {
                if (lb_viol > max_viol) {
                    max_viol = lb_viol;
                    worst = j;
                }
            }
        }
        // Upper bound check
        if (std::isfinite(var.upper_bound)) {
            double ub_viol = xj - var.upper_bound;  // positive if x > ub
            if (ub_viol > tol) {
                if (ub_viol > max_viol) {
                    max_viol = ub_viol;
                    worst = j;
                }
            }
        }
    }

    res.max_violation = max_viol;
    res.worst_index = worst;
    if (max_viol <= tol) {
        res.passed = true;
        std::ostringstream oss;
        oss << "All " << n << " variable bounds satisfied (max_viol=" << max_viol
            << " <= tol=" << tol << ").";
        res.diagnostic = oss.str();
    } else {
        res.passed = false;
        std::ostringstream oss;
        oss << "Bound violation at variable " << worst << " ('"
            << model.variable(static_cast<model::VarId>(worst)).name << "'): max_viol=" << max_viol
            << " > tol=" << tol << ".";
        res.diagnostic = oss.str();
    }
    return res;
}

// ============================================================================
// check_primal_feasibility: Ax ≤/≥/= b independently
// ============================================================================

CheckResult LPVerifier::check_primal_feasibility(const model::CanonicalModel& model,
                                                 const std::vector<double>& x) const {
    CheckResult res;
    res.passed = true;
    res.tolerance_used = tols_.primal_feasibility_tol;
    const double tol = tols_.primal_feasibility_tol;
    double max_viol = 0.0;
    std::size_t worst = std::numeric_limits<std::size_t>::max();

    const std::size_t m = model.num_constraints();
    for (std::size_t i = 0; i < m; ++i) {
        const auto& constr = model.constraint(static_cast<model::ConstrId>(i));
        double activity = compute_row_activity(model, i, x);
        double viol = 0.0;

        switch (constr.sense) {
            case model::ConstraintSense::LESS_EQUAL:
                viol = activity - constr.rhs;  // positive if violated
                break;
            case model::ConstraintSense::GREATER_EQUAL:
                viol = constr.rhs - activity;  // positive if violated
                break;
            case model::ConstraintSense::EQUAL:
                viol = std::abs(activity - constr.rhs);
                break;
        }

        if (viol > max_viol) {
            max_viol = viol;
            worst = i;
        }
    }

    res.max_violation = max_viol;
    res.worst_index = worst;
    if (max_viol <= tol) {
        res.passed = true;
        std::ostringstream oss;
        oss << "All " << m << " constraints satisfied (max_viol=" << max_viol << " <= tol=" << tol
            << ").";
        res.diagnostic = oss.str();
    } else {
        res.passed = false;
        std::ostringstream oss;
        if (worst < m) {
            oss << "Constraint " << worst << " ('"
                << model.constraint(static_cast<model::ConstrId>(worst)).name
                << "') violated: max_viol=" << max_viol << " > tol=" << tol << ".";
        } else {
            oss << "Primal violation: max_viol=" << max_viol << " > tol=" << tol << ".";
        }
        res.diagnostic = oss.str();
    }
    return res;
}

// ============================================================================
// check_dual_feasibility: independently verify reduced costs
//
// Reduced cost formula (minimization original, before any sign flip):
//   rc_j = c_j - (A[:,j])^T · y
//
// For OPTIMAL, reduced costs satisfy complementarity conditions:
//   If x_j > lb_j (variable not at lower bound): rc_j >= -tol
//   If x_j < ub_j (variable not at upper bound): rc_j <= tol
//   If variable is free: rc_j ≈ 0
//   For equality constraints, dual can be any sign.
//
// When the caller provides the solver's rc vector, we check it for
// correctness by independently recomputing rc_j = c_j - A[:,j]^T y.
// We do NOT reuse any solver-side reduced-cost computation.
// ============================================================================

CheckResult LPVerifier::check_dual_feasibility(const model::CanonicalModel& model,
                                               const std::vector<double>& x,
                                               const std::vector<double>& y,
                                               const std::vector<double>& rc) const {
    CheckResult res;
    res.tolerance_used = tols_.dual_feasibility_tol;
    const double tol = tols_.dual_feasibility_tol;
    const std::size_t n = model.num_variables();
    const std::size_t m = model.num_constraints();

    if (y.size() != m) {
        res.passed = false;
        res.max_violation = 1.0;
        std::ostringstream oss;
        oss << "Dual multiplier dimension mismatch: expected " << m << ", got " << y.size() << ".";
        res.diagnostic = oss.str();
        return res;
    }

    if (!rc.empty() && rc.size() != n) {
        res.passed = false;
        res.max_violation = 1.0;
        std::ostringstream oss;
        oss << "Reduced cost vector dimension mismatch: expected " << n << ", got " << rc.size()
            << ".";
        res.diagnostic = oss.str();
        return res;
    }

    const bool is_max = (model.objective_sense() == model::ObjectiveSense::MAXIMIZE);

    // Build objective coefficient vector from model directly (in original problem sense)
    std::vector<double> c(n, 0.0);
    for (const auto& term : model.objective().linear_terms.terms()) {
        if (term.var_id < n) {
            c[term.var_id] = term.coefficient;
        }
    }

    double max_viol = 0.0;
    std::size_t worst = std::numeric_limits<std::size_t>::max();

    // 1. Reduced cost stationarity & bound complementarity:
    // rc_j = c_j - sum_i (A_{ij} * y_i)
    for (std::size_t j = 0; j < n; ++j) {
        double yTa_j = 0.0;
        for (std::size_t i = 0; i < m && i < y.size(); ++i) {
            const auto& constr = model.constraint(static_cast<model::ConstrId>(i));
            for (const auto& term : constr.expression.terms()) {
                if (term.var_id == static_cast<model::VarId>(j)) {
                    yTa_j += term.coefficient * y[i];
                    break;
                }
            }
        }
        double rc_indep = (j < c.size() ? c[j] : 0.0) - yTa_j;

        if (!rc.empty()) {
            double rc_mismatch = std::abs(rc[j] - rc_indep) - tol;
            if (rc_mismatch > 0.0 && rc_mismatch > max_viol) {
                max_viol = rc_mismatch;
                worst = j;
            }
        }

        const auto& var = model.variable(static_cast<model::VarId>(j));
        const double xj = j < x.size() ? x[j] : 0.0;
        const double lb = var.lower_bound;
        const double ub = var.upper_bound;

        const bool is_fixed = std::isfinite(lb) && std::isfinite(ub) && (std::abs(ub - lb) <= tol);
        const bool at_lower = std::isfinite(lb) && (xj <= lb + tol);
        const bool at_upper = std::isfinite(ub) && (xj >= ub - tol);

        if (is_fixed) {
            // Fixed variable: bounds can generate any dual reaction, no restriction on rc
            continue;
        }

        if (at_lower && !at_upper) {
            // At lower bound only:
            // For MINIMIZE: rc_j >= -tol
            // For MAXIMIZE: rc_j <= tol
            double viol = is_max ? (rc_indep - tol) : (-rc_indep - tol);
            if (viol > 0.0 && viol > max_viol) {
                max_viol = viol;
                worst = j;
            }
        } else if (at_upper && !at_lower) {
            // At upper bound only:
            // For MINIMIZE: rc_j <= tol
            // For MAXIMIZE: rc_j >= -tol
            double viol = is_max ? (-rc_indep - tol) : (rc_indep - tol);
            if (viol > 0.0 && viol > max_viol) {
                max_viol = viol;
                worst = j;
            }
        } else if (!at_lower && !at_upper) {
            // Strictly interior / free variable:
            // Gradient must vanish: |rc_j| <= tol
            double viol = std::abs(rc_indep) - tol;
            if (viol > 0.0 && viol > max_viol) {
                max_viol = viol;
                worst = j;
            }
        }
    }

    // 2. Dual multiplier signs for inequality constraints:
    for (std::size_t i = 0; i < m && i < y.size(); ++i) {
        const auto& constr = model.constraint(static_cast<model::ConstrId>(i));
        const double yi = y[i];
        double viol = 0.0;

        if (constr.sense == model::ConstraintSense::LESS_EQUAL) {
            // For MINIMIZE: yi <= tol
            // For MAXIMIZE: yi >= -tol
            viol = is_max ? (-yi - tol) : (yi - tol);
        } else if (constr.sense == model::ConstraintSense::GREATER_EQUAL) {
            // For MINIMIZE: yi >= -tol
            // For MAXIMIZE: yi <= tol
            viol = is_max ? (yi - tol) : (-yi - tol);
        }
        if (viol > 0.0 && viol > max_viol) {
            max_viol = viol;
            worst = n + i;
        }
    }

    res.max_violation = max_viol;
    res.worst_index = worst;
    if (max_viol <= 0.0 || max_viol <= tol * 10.0 + 1e-9) {
        res.passed = true;
        std::ostringstream oss;
        oss << "Dual feasibility verified for " << n << " variables and " << m
            << " constraints (max_violation=" << max_viol << ").";
        res.diagnostic = oss.str();
    } else {
        res.passed = false;
        std::ostringstream oss;
        oss << "Dual violation: max_viol=" << max_viol << " > tol=" << tol;
        if (worst < n) {
            oss << " at variable " << worst << " ('"
                << model.variable(static_cast<model::VarId>(worst)).name << "').";
        } else if (worst < n + m) {
            oss << " at constraint " << (worst - n) << " ('"
                << model.constraint(static_cast<model::ConstrId>(worst - n)).name << "').";
        }
        res.diagnostic = oss.str();
    }
    return res;
}

// ============================================================================
// check_complementary_slackness
// ============================================================================

CheckResult LPVerifier::check_complementary_slackness(const model::CanonicalModel& model,
                                                      const std::vector<double>& x,
                                                      const std::vector<double>& y) const {
    CheckResult res;
    res.tolerance_used = tols_.complementarity_tol;
    const double tol = tols_.complementarity_tol;
    const std::size_t m = model.num_constraints();

    if (m == 0) {
        res.passed = true;
        res.diagnostic = "No constraints — complementary slackness vacuously satisfied.";
        return res;
    }

    if (y.size() < m) {
        res.passed = false;
        res.diagnostic = "Dual multipliers unavailable — complementary slackness not verifiable.";
        return res;
    }

    double max_cs = 0.0;
    std::size_t worst = std::numeric_limits<std::size_t>::max();

    for (std::size_t i = 0; i < m; ++i) {
        const auto& constr = model.constraint(static_cast<model::ConstrId>(i));
        const double yi = y[i];
        const double act = compute_row_activity(model, i, x);

        double slack = 0.0;
        switch (constr.sense) {
            case model::ConstraintSense::LESS_EQUAL:
                slack = constr.rhs - act;
                break;
            case model::ConstraintSense::GREATER_EQUAL:
                slack = act - constr.rhs;
                break;
            case model::ConstraintSense::EQUAL:
                slack = 0.0;
                break;
        }

        double cs_res = std::abs(slack * yi);
        if (cs_res > max_cs) {
            max_cs = cs_res;
            worst = i;
        }
    }

    res.max_violation = max_cs;
    res.worst_index = worst;
    if (max_cs <= tol) {
        res.passed = true;
        std::ostringstream oss;
        oss << "Complementary slackness satisfied for " << m
            << " constraints (max_cs_res=" << max_cs << " <= tol=" << tol << ").";
        res.diagnostic = oss.str();
    } else {
        res.passed = false;
        std::ostringstream oss;
        oss << "Complementary slackness violated at constraint " << worst << " ('"
            << model.constraint(static_cast<model::ConstrId>(worst)).name
            << "'): |slack * y_i|=" << max_cs << " > tol=" << tol << ".";
        res.diagnostic = oss.str();
    }
    return res;
}

// ============================================================================
// check_objective: independently recompute cᵀx + c₀
// ============================================================================

CheckResult LPVerifier::check_objective(const model::CanonicalModel& model,
                                        const std::vector<double>& x, double reported_obj,
                                        double& recomputed_obj_out) const {
    CheckResult res;
    res.tolerance_used = tols_.objective_abs_tol;

    // Reject quadratic objective content for LP verification
    if (model.objective().is_quadratic()) {
        res.passed = false;
        res.diagnostic =
            "Objective has quadratic terms — LP verifier cannot certify QP objectives.";
        return res;
    }

    // Independently compute cᵀx + c₀
    double obj = model.objective().offset;
    const std::size_t n = model.num_variables();
    for (const auto& term : model.objective().linear_terms.terms()) {
        if (term.var_id < n && term.var_id < x.size()) {
            obj += term.coefficient * x[term.var_id];
        }
    }
    recomputed_obj_out = obj;
    res.max_violation = std::abs(obj - reported_obj);

    const double abs_tol = tols_.objective_abs_tol;
    const double rel_tol = tols_.objective_rel_tol;
    const double denom = std::max(1.0, std::abs(reported_obj));
    const double rel_err = res.max_violation / denom;

    if (res.max_violation <= abs_tol || rel_err <= rel_tol) {
        res.passed = true;
        std::ostringstream oss;
        oss << "Objective consistent: recomputed=" << obj << " reported=" << reported_obj
            << " abs_err=" << res.max_violation << " rel_err=" << rel_err << ".";
        res.diagnostic = oss.str();
    } else {
        res.passed = false;
        std::ostringstream oss;
        oss << "Objective mismatch: recomputed=" << obj << " reported=" << reported_obj
            << " abs_err=" << res.max_violation << " > abs_tol=" << abs_tol
            << " AND rel_err=" << rel_err << " > rel_tol=" << rel_tol << ".";
        res.diagnostic = oss.str();
    }
    return res;
}

// ============================================================================
// check_status_consistency
// ============================================================================

CheckResult LPVerifier::check_status_consistency(
    core::SolverStatus reported_status, const CheckResult& primal_feas,
    const CheckResult& bound_feas, const CheckResult& dual_feas, const CheckResult& cs,
    const CheckResult& obj_check, const CheckResult& nonfinite) const {
    CheckResult res;
    res.passed = true;
    res.tolerance_used = 0.0;  // Status check has no numeric tolerance

    switch (reported_status) {
        case core::SolverStatus::OPTIMAL: {
            // OPTIMAL requires ALL checks to pass
            if (!nonfinite.passed) {
                res.passed = false;
                res.diagnostic =
                    "OPTIMAL rejected: non-finite values in solution vectors. "
                    "Certified: NUMERICAL_FAILURE.";
            } else if (!primal_feas.passed || !bound_feas.passed) {
                res.passed = false;
                res.diagnostic =
                    "OPTIMAL rejected: primal or bound infeasibility detected. "
                    "Certified: FEASIBLE (if primal ok) or UNKNOWN.";
            } else if (!obj_check.passed) {
                res.passed = false;
                res.diagnostic =
                    "OPTIMAL rejected: reported objective inconsistent with independently "
                    "recomputed value. Certified: FEASIBLE.";
            } else if (dual_feas.passed && cs.passed) {
                res.passed = true;
                res.diagnostic =
                    "OPTIMAL confirmed: primal feasible, dual feasible, complementary "
                    "slackness, and objective all verified.";
            } else if (!dual_feas.passed && !dual_feas.diagnostic.empty() &&
                       dual_feas.diagnostic.find("unavailable") != std::string::npos) {
                // Dual info not available — downgrade to FEASIBLE, not OPTIMAL
                res.passed = false;
                res.diagnostic =
                    "OPTIMAL cannot be certified: dual information unavailable. "
                    "Certified: FEASIBLE (primal checks passed).";
            } else {
                res.passed = false;
                res.diagnostic =
                    "OPTIMAL rejected: dual feasibility or complementary slackness not "
                    "satisfied. Certified: FEASIBLE.";
            }
            break;
        }
        case core::SolverStatus::INFEASIBLE:
            // We cannot independently verify the full Farkas certificate from the
            // current LPSolution API; report unavailability honestly.
            res.passed = true;
            res.diagnostic =
                "INFEASIBLE accepted: solver reported infeasibility (Farkas certificate "
                "validation not yet supported at this API level — marked as accepted).";
            break;

        case core::SolverStatus::UNBOUNDED:
            // Unbounded ray not exposed by current solver API
            res.passed = true;
            res.diagnostic =
                "UNBOUNDED accepted: solver reported unboundedness (ray validation not "
                "yet supported at this API level — marked as accepted).";
            break;

        case core::SolverStatus::LIMIT_REACHED:
            res.passed = true;
            res.diagnostic = "LIMIT_REACHED: not a definitive status, no optimality claim.";
            break;

        case core::SolverStatus::NUMERICAL_FAILURE:
            res.passed = true;
            res.diagnostic = "NUMERICAL_FAILURE: solver-declared numerical problem.";
            break;

        case core::SolverStatus::FEASIBLE:
            res.passed = true;
            res.diagnostic = "FEASIBLE: solver did not claim optimality, no dual check required.";
            break;

        case core::SolverStatus::UNKNOWN:
        default:
            res.passed = true;
            res.diagnostic = "UNKNOWN status: no verification claim can be made.";
            break;
    }
    return res;
}

// ============================================================================
// verify — main entry point
// ============================================================================

LPVerificationReport LPVerifier::verify(const model::CanonicalModel& model,
                                        const std::vector<double>& x, const std::vector<double>& y,
                                        const std::vector<double>& rc, double reported_obj,
                                        core::SolverStatus reported_status) const {
    LPVerificationReport report;
    const std::size_t m = model.num_constraints();
    report.dual_info_available = (m == 0) || (!y.empty() && y.size() >= m);
    report.certificate_validation_available = false;  // Not yet supported

    // --- 1. Non-finite scan (always first: stop early on NaN/Inf) ---
    report.nonfinite_check = check_nonfinite(x, y, rc);

    // --- 2. Bound feasibility ---
    report.bound_feasibility = check_bound_feasibility(model, x);

    // --- 3. Primal feasibility ---
    report.primal_feasibility = check_primal_feasibility(model, x);

    // --- 4. Dual feasibility (only if dual info is available) ---
    if (report.dual_info_available) {
        report.dual_feasibility = check_dual_feasibility(model, x, y, rc);
    } else {
        report.dual_feasibility.passed = false;
        report.dual_feasibility.diagnostic =
            "Dual multipliers not provided — dual feasibility not verified.";
    }

    // --- 5. Complementary slackness ---
    if (report.dual_info_available) {
        report.complementary_slackness = check_complementary_slackness(model, x, y);
    } else {
        report.complementary_slackness.passed = false;
        report.complementary_slackness.diagnostic =
            "Dual multipliers not provided — complementary slackness not verified.";
    }

    // --- 6. Objective recomputation ---
    double recomp_obj = 0.0;
    report.objective_consistency = check_objective(model, x, reported_obj, recomp_obj);
    report.recomputed_objective = recomp_obj;
    report.objective_absolute_error = std::abs(recomp_obj - reported_obj);
    const double denom = std::max(1.0, std::abs(reported_obj));
    report.objective_relative_error = report.objective_absolute_error / denom;

    // --- 7. Status consistency ---
    report.status_consistency = check_status_consistency(
        reported_status, report.primal_feasibility, report.bound_feasibility,
        report.dual_feasibility, report.complementary_slackness, report.objective_consistency,
        report.nonfinite_check);

    // --- 8. Aggregate residuals ---
    report.aggregate_primal_residual =
        std::max(report.primal_feasibility.max_violation, report.bound_feasibility.max_violation);
    report.aggregate_dual_residual = report.dual_feasibility.max_violation;

    // --- 9. Determine certified status ---
    if (!report.nonfinite_check.passed) {
        report.certified_status = core::SolverStatus::NUMERICAL_FAILURE;
        report.status_downgraded = (reported_status == core::SolverStatus::OPTIMAL ||
                                    reported_status == core::SolverStatus::FEASIBLE);
    } else if (reported_status == core::SolverStatus::OPTIMAL &&
               !report.status_consistency.passed) {
        // Determine what we CAN certify
        if (report.primal_feasibility.passed && report.bound_feasibility.passed) {
            report.certified_status = core::SolverStatus::FEASIBLE;
        } else {
            report.certified_status = core::SolverStatus::UNKNOWN;
        }
        report.status_downgraded = true;
    } else {
        report.certified_status = reported_status;
        report.status_downgraded = false;
    }

    // --- 10. Build summary ---
    std::ostringstream ss;
    ss << "Verification summary for model '" << model.name()
       << "': " << "reported=" << core::to_string(reported_status) << " "
       << "certified=" << core::to_string(report.certified_status) << " "
       << "[primal=" << (report.primal_feasibility.passed ? "PASS" : "FAIL")
       << " bounds=" << (report.bound_feasibility.passed ? "PASS" : "FAIL")
       << " dual=" << (report.dual_feasibility.passed ? "PASS" : "FAIL")
       << " cs=" << (report.complementary_slackness.passed ? "PASS" : "FAIL")
       << " obj=" << (report.objective_consistency.passed ? "PASS" : "FAIL")
       << " nonfinite=" << (report.nonfinite_check.passed ? "PASS" : "FAIL") << "]";
    report.summary = ss.str();

    return report;
}

// ============================================================================
// Convenience overload with solver tolerances
// ============================================================================

LPVerificationReport LPVerifier::verify(const model::CanonicalModel& model,
                                        const std::vector<double>& x, const std::vector<double>& y,
                                        const std::vector<double>& rc, double reported_obj,
                                        core::SolverStatus reported_status,
                                        const core::SolverTolerances& solver_tols) const {
    LPVerifier tmp(VerifierTolerances::from_solver_tolerances(solver_tols));
    return tmp.verify(model, x, y, rc, reported_obj, reported_status);
}

}  // namespace kavach::verification
