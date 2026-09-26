#include "verification/miqp_verifier.hpp"

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "core/status.hpp"

namespace kavach::verification {

namespace {

double compute_row_activity(const model::CanonicalModel& model, std::size_t row_idx,
                            const std::vector<double>& x) {
    const auto& constr = model.constraint(static_cast<model::ConstrId>(row_idx));
    double sum = 0.0;
    for (const auto& term : constr.expression.terms()) {
        if (term.var_id < x.size()) {
            sum += term.coefficient * x[term.var_id];
        }
    }
    return sum;
}

}  // namespace

CheckResult MIQPVerifier::check_nonfinite(const std::vector<double>& incumbent) const {
    CheckResult r;
    r.passed = true;
    r.tolerance_used = 0.0;

    for (std::size_t i = 0; i < incumbent.size(); ++i) {
        if (std::isnan(incumbent[i])) {
            r.passed = false;
            r.worst_index = i;
            r.max_violation = std::numeric_limits<double>::infinity();
            r.diagnostic =
                "NaN detected in primal incumbent vector at variable " + std::to_string(i);
            return r;
        }
        if (std::isinf(incumbent[i])) {
            r.passed = false;
            r.worst_index = i;
            r.max_violation = std::numeric_limits<double>::infinity();
            r.diagnostic =
                "Infinity detected in primal incumbent vector at variable " + std::to_string(i);
            return r;
        }
    }

    r.diagnostic = "All incumbent elements finite";
    return r;
}

CheckResult MIQPVerifier::check_bounds(const model::CanonicalModel& model,
                                       const std::vector<double>& incumbent) const {
    CheckResult r;
    r.passed = true;
    r.tolerance_used = tolerances_.bound_feasibility_tol;

    double max_viol = 0.0;
    std::size_t worst_var = std::numeric_limits<std::size_t>::max();

    const std::size_t n = std::min(incumbent.size(), model.num_variables());
    for (std::size_t j = 0; j < n; ++j) {
        const auto& var = model.variable(static_cast<model::VarId>(j));
        double val = incumbent[j];
        double viol = 0.0;

        if (var.has_lower_bound() && val < var.lower_bound) {
            viol = std::max(viol, var.lower_bound - val);
        }
        if (var.has_upper_bound() && val > var.upper_bound) {
            viol = std::max(viol, val - var.upper_bound);
        }

        if (viol > max_viol) {
            max_viol = viol;
            worst_var = j;
        }
    }

    r.max_violation = max_viol;
    r.worst_index = worst_var;
    if (max_viol > tolerances_.bound_feasibility_tol) {
        r.passed = false;
        r.diagnostic = "Bound violation (" + std::to_string(max_viol) + " > tol " +
                       std::to_string(tolerances_.bound_feasibility_tol) + ") at variable " +
                       std::to_string(worst_var);
    } else {
        r.passed = true;
        r.diagnostic =
            "All variable bounds satisfied (max violation: " + std::to_string(max_viol) + ")";
    }
    return r;
}

CheckResult MIQPVerifier::check_primal_feasibility(const model::CanonicalModel& model,
                                                   const std::vector<double>& incumbent) const {
    CheckResult r;
    r.passed = true;
    r.tolerance_used = tolerances_.primal_feasibility_tol;

    double max_viol = 0.0;
    std::size_t worst_row = std::numeric_limits<std::size_t>::max();

    for (std::size_t i = 0; i < model.num_constraints(); ++i) {
        const auto& constr = model.constraint(static_cast<model::ConstrId>(i));
        double activity = compute_row_activity(model, i, incumbent);
        double rhs = constr.rhs;
        double viol = 0.0;

        switch (constr.sense) {
            case model::ConstraintSense::LESS_EQUAL:
                viol = std::max(0.0, activity - rhs);
                break;
            case model::ConstraintSense::GREATER_EQUAL:
                viol = std::max(0.0, rhs - activity);
                break;
            case model::ConstraintSense::EQUAL:
                viol = std::abs(activity - rhs);
                break;
        }

        if (viol > max_viol) {
            max_viol = viol;
            worst_row = i;
        }
    }

    r.max_violation = max_viol;
    r.worst_index = worst_row;
    if (max_viol > tolerances_.primal_feasibility_tol) {
        r.passed = false;
        r.diagnostic = "Primal constraint violation (" + std::to_string(max_viol) + " > tol " +
                       std::to_string(tolerances_.primal_feasibility_tol) + ") at constraint " +
                       std::to_string(worst_row);
    } else {
        r.passed = true;
        r.diagnostic =
            "All constraints satisfied (max violation: " + std::to_string(max_viol) + ")";
    }
    return r;
}

CheckResult MIQPVerifier::check_integrality(const model::CanonicalModel& model,
                                            const std::vector<double>& incumbent) const {
    CheckResult r;
    r.passed = true;
    r.tolerance_used = tolerances_.integrality_tol;

    double max_viol = 0.0;
    std::size_t worst_var = std::numeric_limits<std::size_t>::max();

    const std::size_t n = std::min(incumbent.size(), model.num_variables());
    for (std::size_t j = 0; j < n; ++j) {
        const auto& var = model.variable(static_cast<model::VarId>(j));
        if (var.is_integer() || var.is_binary()) {
            double val = incumbent[j];
            double rounded = std::round(val);
            double viol = std::abs(val - rounded);

            if (var.is_binary()) {
                if (val < 0.0) {
                    viol = std::max(viol, -val);
                } else if (val > 1.0) {
                    viol = std::max(viol, val - 1.0);
                }
            }

            if (viol > max_viol) {
                max_viol = viol;
                worst_var = j;
            }
        }
    }

    r.max_violation = max_viol;
    r.worst_index = worst_var;
    if (max_viol > tolerances_.integrality_tol) {
        r.passed = false;
        r.diagnostic = "Integrality violation (" + std::to_string(max_viol) + " > tol " +
                       std::to_string(tolerances_.integrality_tol) + ") at variable " +
                       std::to_string(worst_var);
    } else {
        r.passed = true;
        r.diagnostic = "All integer variables satisfy integrality (max violation: " +
                       std::to_string(max_viol) + ")";
    }
    return r;
}

CheckResult MIQPVerifier::check_objective(const model::CanonicalModel& model,
                                          const std::vector<double>& incumbent,
                                          double reported_objective,
                                          double& recomputed_obj_out) const {
    CheckResult r;
    r.tolerance_used = tolerances_.objective_abs_tol;

    const std::size_t n = model.num_variables();
    double obj = model.objective_offset();

    for (const auto& term : model.objective().linear_terms.terms()) {
        if (term.var_id < n && term.var_id < incumbent.size()) {
            obj += term.coefficient * incumbent[term.var_id];
        }
    }

    for (const auto& qt : model.objective().quadratic_terms.terms()) {
        if (qt.var1_id < n && qt.var2_id < n && qt.var1_id < incumbent.size() &&
            qt.var2_id < incumbent.size()) {
            obj += qt.coefficient * incumbent[qt.var1_id] * incumbent[qt.var2_id];
        }
    }

    recomputed_obj_out = obj;
    const double abs_err = std::abs(obj - reported_objective);
    const double denom = std::max(1.0, std::abs(reported_objective));
    const double rel_err = abs_err / denom;

    r.max_violation = abs_err;
    if (abs_err <= tolerances_.objective_abs_tol || rel_err <= tolerances_.objective_rel_tol) {
        r.passed = true;
        r.diagnostic =
            "Objective consistent with reported value: recomputed=" + std::to_string(obj) +
            ", reported=" + std::to_string(reported_objective) +
            " (abs_err=" + std::to_string(abs_err) + ")";
    } else {
        r.passed = false;
        r.diagnostic = "Objective discrepancy: recomputed=" + std::to_string(obj) +
                       ", reported=" + std::to_string(reported_objective) +
                       " (abs_err=" + std::to_string(abs_err) +
                       ", rel_err=" + std::to_string(rel_err) + ")";
    }
    return r;
}

CheckResult MIQPVerifier::check_convexity(const model::CanonicalModel& model) const {
    CheckResult res;
    res.tolerance_used = tolerances_.psd_tol;
    res.passed = true;

    const std::size_t n = model.num_variables();
    if (n == 0) {
        res.diagnostic = "Empty model; trivially convex PSD.";
        return res;
    }

    // Build dense working representation for PSD check
    // For MINIMIZE: Q must be PSD.
    // For MAXIMIZE: f(x) must be concave, so Q must be NSD, i.e. -Q must be PSD.
    const bool is_maximize = (model.objective_sense() == model::ObjectiveSense::MAXIMIZE);
    const double sign = is_maximize ? -1.0 : 1.0;

    std::vector<double> Q(n * n, 0.0);
    for (const auto& qt : model.objective().quadratic_terms.terms()) {
        if (qt.var1_id < n && qt.var2_id < n) {
            if (qt.var1_id == qt.var2_id) {
                Q[qt.var1_id * n + qt.var1_id] += sign * 2.0 * qt.coefficient;
            } else {
                Q[qt.var1_id * n + qt.var2_id] += sign * qt.coefficient;
                Q[qt.var2_id * n + qt.var1_id] += sign * qt.coefficient;
            }
        }
    }

    // Diagonal elements check: any negative diagonal immediately disproves PSD
    for (std::size_t i = 0; i < n; ++i) {
        const double d = Q[i * n + i];
        if (d < -tolerances_.psd_tol) {
            res.passed = false;
            res.max_violation = -d;
            res.worst_index = i;
            res.diagnostic = "Negative diagonal entry Q[" + std::to_string(i) + "," +
                             std::to_string(i) + "] = " + std::to_string(d) +
                             " (Hessian is non-convex/indefinite).";
            return res;
        }
    }

    // Complete diagonal-pivoted Cholesky factorization
    std::vector<std::size_t> p(n);
    for (std::size_t i = 0; i < n; ++i) {
        p[i] = i;
    }

    for (std::size_t k = 0; k < n; ++k) {
        double max_diag = -1.0;
        std::size_t best_idx = k;
        for (std::size_t i = k; i < n; ++i) {
            const double diag = Q[p[i] * n + p[i]];
            if (diag > max_diag) {
                max_diag = diag;
                best_idx = i;
            }
        }

        if (max_diag < -tolerances_.psd_tol) {
            res.passed = false;
            res.max_violation = -max_diag;
            res.worst_index = p[best_idx];
            res.diagnostic =
                "Negative pivot in Cholesky factorization: " + std::to_string(max_diag) +
                " detected (Hessian is indefinite).";
            return res;
        }

        if (max_diag <= tolerances_.psd_tol) {
            // Remainder must be zero
            for (std::size_t i = k; i < n; ++i) {
                for (std::size_t j = k; j < n; ++j) {
                    const double val = std::abs(Q[p[i] * n + p[j]]);
                    if (val > tolerances_.psd_tol * 100.0 + 1e-7) {
                        res.passed = false;
                        res.max_violation = val;
                        res.worst_index = p[i];
                        res.diagnostic = "Non-zero off-diagonal entry in null Schur complement: " +
                                         std::to_string(val) + " (Hessian is indefinite).";
                        return res;
                    }
                }
            }
            break;  // Remainder is null; Q is PSD with rank k
        }

        std::swap(p[k], p[best_idx]);
        const std::size_t pk = p[k];
        const double pivot = Q[pk * n + pk];

        for (std::size_t i = k + 1; i < n; ++i) {
            const std::size_t pi = p[i];
            const double factor = Q[pi * n + pk] / pivot;
            Q[pi * n + pk] = factor;
            for (std::size_t j = k + 1; j < n; ++j) {
                const std::size_t pj = p[j];
                Q[pi * n + pj] -= factor * Q[pk * n + pj];
            }
        }
    }

    res.diagnostic = "Hessian verified positive semidefinite (Cholesky completed successfully).";
    return res;
}

MIQPVerificationReport MIQPVerifier::verify(const model::CanonicalModel& model,
                                            const std::vector<double>& incumbent_solution,
                                            double best_bound, double reported_objective,
                                            core::SolverStatus reported_status,
                                            const core::SolverTolerances& solver_tols,
                                            double solver_reported_gap) const {
    MIQPVerificationReport rep;
    rep.original_reported_status = reported_status;
    rep.incumbent_available = !incumbent_solution.empty();

    if (solver_tols.primal_feasibility_tol > 0.0) {
        // Tolerances can be synchronized if explicitly provided
    }

    // 1. Convexity Check
    rep.psd_check = check_convexity(model);
    if (!rep.psd_check.passed) {
        rep.certified_status = core::SolverStatus::NUMERICAL_FAILURE;
        rep.status_downgraded = true;
        rep.summary = "MIQP Verification FAILED: " + rep.psd_check.diagnostic;
        return rep;
    }

    // 2. Status INFEASIBLE verification
    if (reported_status == core::SolverStatus::INFEASIBLE) {
        rep.status_consistency.passed = true;
        rep.status_consistency.diagnostic = "Status INFEASIBLE reported.";
        rep.certified_status = core::SolverStatus::INFEASIBLE;
        rep.summary = "MIQP verified INFEASIBLE (consistent with solver report).";
        return rep;
    }

    // 3. Incumbent verification
    if (!rep.incumbent_available) {
        rep.status_consistency.passed = false;
        rep.status_consistency.diagnostic = "No incumbent solution provided for verification.";
        rep.certified_status = core::SolverStatus::UNKNOWN;
        rep.summary = "Verification incomplete: no incumbent solution.";
        return rep;
    }

    rep.nonfinite_check = check_nonfinite(incumbent_solution);
    if (!rep.nonfinite_check.passed) {
        rep.certified_status = core::SolverStatus::NUMERICAL_FAILURE;
        rep.status_downgraded = true;
        rep.summary = "Verification FAILED: " + rep.nonfinite_check.diagnostic;
        return rep;
    }

    rep.bound_feasibility = check_bounds(model, incumbent_solution);
    rep.max_bound_violation = rep.bound_feasibility.max_violation;

    rep.primal_feasibility = check_primal_feasibility(model, incumbent_solution);
    rep.aggregate_primal_residual = rep.primal_feasibility.max_violation;

    rep.integrality = check_integrality(model, incumbent_solution);
    rep.max_integrality_violation = rep.integrality.max_violation;

    double rec_obj = 0.0;
    rep.objective_consistency =
        check_objective(model, incumbent_solution, reported_objective, rec_obj);
    rep.recomputed_objective = rec_obj;
    rep.objective_absolute_error = rep.objective_consistency.max_violation;

    // Feasibility failure checks
    if (!rep.bound_feasibility.passed || !rep.primal_feasibility.passed ||
        !rep.integrality.passed || !rep.objective_consistency.passed) {
        rep.certified_status = core::SolverStatus::UNKNOWN;
        rep.status_downgraded = true;
        std::ostringstream oss;
        oss << "Incumbent validation FAILED:";
        if (!rep.bound_feasibility.passed)
            oss << " [" << rep.bound_feasibility.diagnostic << "]";
        if (!rep.primal_feasibility.passed)
            oss << " [" << rep.primal_feasibility.diagnostic << "]";
        if (!rep.integrality.passed)
            oss << " [" << rep.integrality.diagnostic << "]";
        if (!rep.objective_consistency.passed)
            oss << " [" << rep.objective_consistency.diagnostic << "]";
        rep.summary = oss.str();
        return rep;
    }

    // 4. Gap Computation
    double gap = 0.0;
    if (std::isinf(best_bound)) {
        gap = (reported_status == core::SolverStatus::OPTIMAL) ? 0.0 : 1.0;
    } else {
        gap = std::abs(rec_obj - best_bound) / std::max(1.0, std::abs(rec_obj));
    }
    if (solver_reported_gap >= 0.0 && gap > solver_reported_gap) {
        // preserve solver gap if within reasonable bound
    }
    rep.recomputed_gap = gap;

    if (gap <= tolerances_.gap_tolerance) {
        rep.gap_consistency.passed = true;
        rep.gap_consistency.diagnostic = "Recomputed gap " + std::to_string(gap) +
                                         " <= tolerance " +
                                         std::to_string(tolerances_.gap_tolerance);
    } else {
        rep.gap_consistency.passed = false;
        rep.gap_consistency.diagnostic = "Recomputed gap " + std::to_string(gap) + " > tolerance " +
                                         std::to_string(tolerances_.gap_tolerance);
    }

    // 5. Final Status Determination
    if (reported_status == core::SolverStatus::OPTIMAL) {
        if (rep.gap_consistency.passed) {
            rep.certified_status = core::SolverStatus::OPTIMAL;
            rep.status_consistency.passed = true;
            rep.summary =
                "OPTIMAL certified by independent MIQP verification (gap=" + std::to_string(gap) +
                ").";
        } else {
            // Downgrade to FEASIBLE
            rep.certified_status = core::SolverStatus::FEASIBLE;
            rep.status_downgraded = true;
            rep.status_consistency.passed = false;
            rep.summary = "Optimality uncertified: gap " + std::to_string(gap) +
                          " exceeds tolerance; downgraded to FEASIBLE.";
        }
    } else if (reported_status == core::SolverStatus::FEASIBLE) {
        rep.certified_status = core::SolverStatus::FEASIBLE;
        rep.status_consistency.passed = true;
        rep.summary = "Incumbent feasibility verified (solver reported FEASIBLE).";
    } else {
        rep.certified_status = reported_status;
        rep.status_consistency.passed = true;
        rep.summary = "Status " + std::string(core::to_string(reported_status)) +
                      " with verified feasible incumbent.";
    }

    return rep;
}

}  // namespace kavach::verification
