#include "verification/qp_verifier.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace kavach::verification {

CheckResult QPVerifier::check_nonfinite(const std::vector<double>& primal,
                                        const std::vector<double>& dual) const {
    CheckResult res;
    res.tolerance_used = 0.0;
    res.passed = true;

    auto scan = [&](const std::vector<double>& v, const char* name) {
        for (std::size_t k = 0; k < v.size(); ++k) {
            if (!std::isfinite(v[k])) {
                res.passed = false;
                res.worst_index = k;
                std::ostringstream oss;
                oss << "Non-finite value in " << name << "[" << k << "] = " << v[k];
                res.diagnostic = oss.str();
                res.max_violation = 1.0;
                return;
            }
        }
    };

    scan(primal, "primal_solution");
    if (!res.passed) {
        return res;
    }

    scan(dual, "dual_multipliers");
    if (!res.passed) {
        return res;
    }

    res.diagnostic = "All solution components are finite.";
    return res;
}

CheckResult QPVerifier::check_bounds(const model::CanonicalModel& model,
                                     const std::vector<double>& primal) const {
    CheckResult res;
    const double tol = tolerances_.primal_feasibility_tol;
    res.tolerance_used = tol;
    res.passed = true;

    const std::size_t n = model.num_variables();
    if (primal.size() != n) {
        res.passed = false;
        res.diagnostic = "Dimension mismatch: primal size " + std::to_string(primal.size()) +
                         " != model variables " + std::to_string(n);
        res.max_violation = std::numeric_limits<double>::infinity();
        return res;
    }

    double max_viol = 0.0;
    std::size_t worst = std::numeric_limits<std::size_t>::max();

    for (std::size_t j = 0; j < n; ++j) {
        const auto& var = model.variable(static_cast<model::VarId>(j));
        const double x_j = primal[j];

        if (var.has_lower_bound()) {
            const double viol_lb = var.lower_bound - x_j;
            if (viol_lb > max_viol) {
                max_viol = viol_lb;
                worst = j;
            }
        }

        if (var.has_upper_bound()) {
            const double viol_ub = x_j - var.upper_bound;
            if (viol_ub > max_viol) {
                max_viol = viol_ub;
                worst = j;
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

CheckResult QPVerifier::check_primal_feasibility(const model::CanonicalModel& model,
                                                 const std::vector<double>& primal,
                                                 std::vector<double>& constraint_activities) const {
    CheckResult res;
    const double tol = tolerances_.primal_feasibility_tol;
    res.tolerance_used = tol;
    res.passed = true;

    const std::size_t m = model.num_constraints();
    constraint_activities.assign(m, 0.0);

    double max_viol = 0.0;
    std::size_t worst = std::numeric_limits<std::size_t>::max();

    for (std::size_t i = 0; i < m; ++i) {
        const auto& constr = model.constraint(static_cast<model::ConstrId>(i));
        double act = 0.0;
        for (const auto& term : constr.expression.terms()) {
            if (term.var_id < primal.size()) {
                act += term.coefficient * primal[term.var_id];
            }
        }
        constraint_activities[i] = act;

        double viol = 0.0;
        switch (constr.sense) {
            case model::ConstraintSense::LESS_EQUAL:
                viol = act - constr.rhs;
                break;
            case model::ConstraintSense::GREATER_EQUAL:
                viol = constr.rhs - act;
                break;
            case model::ConstraintSense::EQUAL:
                viol = std::abs(act - constr.rhs);
                break;
        }

        if (viol > max_viol) {
            max_viol = viol;
            worst = i;
        }
    }

    res.max_violation = std::max(0.0, max_viol);
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
        oss << "Constraint " << worst << " ('"
            << model.constraint(static_cast<model::ConstrId>(worst)).name
            << "') violated: max_viol=" << max_viol << " > tol=" << tol << ".";
        res.diagnostic = oss.str();
    }

    return res;
}

CheckResult QPVerifier::check_stationarity_and_duals(
    const model::CanonicalModel& model, const std::vector<double>& primal,
    const std::vector<double>& dual, const std::vector<double>& constraint_activities,
    CheckResult& dual_feas_result, CheckResult& cs_result) const {
    const std::size_t n = model.num_variables();
    const std::size_t m = model.num_constraints();

    dual_feas_result.passed = true;
    dual_feas_result.tolerance_used = tolerances_.dual_feasibility_tol;
    dual_feas_result.max_violation = 0.0;

    cs_result.passed = true;
    cs_result.tolerance_used = tolerances_.residual_tol;
    cs_result.max_violation = 0.0;

    CheckResult stat_res;
    stat_res.passed = true;
    stat_res.tolerance_used = tolerances_.optimality_tol;
    stat_res.max_violation = 0.0;

    if (dual.size() != m) {
        dual_feas_result.passed = false;
        dual_feas_result.diagnostic = "Dual vector size mismatch: " + std::to_string(dual.size()) +
                                      " != constraints " + std::to_string(m);
        stat_res.passed = false;
        stat_res.diagnostic = dual_feas_result.diagnostic;
        cs_result.passed = false;
        cs_result.diagnostic = dual_feas_result.diagnostic;
        return stat_res;
    }

    // 1. Recompute gradient: g = c + Q * x
    std::vector<double> grad(n, 0.0);
    for (const auto& lt : model.objective().linear_terms.terms()) {
        if (lt.var_id < n) {
            grad[lt.var_id] += lt.coefficient;
        }
    }
    for (const auto& qt : model.objective().quadratic_terms.terms()) {
        if (qt.var1_id < n && qt.var2_id < n) {
            if (qt.var1_id == qt.var2_id) {
                // Diagonal: f(x) has q_ii * x_i^2 => d/dx_i = 2 * q_ii * x_i
                grad[qt.var1_id] += 2.0 * qt.coefficient * primal[qt.var1_id];
            } else {
                // Off-diagonal: f(x) has q_ij * x_i * x_j => d/dx_i = q_ij * x_j, d/dx_j = q_ij *
                // x_i
                grad[qt.var1_id] += qt.coefficient * primal[qt.var2_id];
                grad[qt.var2_id] += qt.coefficient * primal[qt.var1_id];
            }
        }
    }

    // 2. Add constraint contributions: A^T * lambda
    // In min f(x) s.t. A_i x <= b_i (lambda_i >= 0), A_i x >= b_i (lambda_i <= 0), A_i x == b_i
    // Stationarity: grad f(x) + A^T lambda - mu = 0 => mu_j = grad_j + (A^T lambda)_j
    std::vector<double> AT_lambda(n, 0.0);
    double max_dual_viol = 0.0;
    std::size_t worst_dual = std::numeric_limits<std::size_t>::max();

    double max_cs_viol = 0.0;
    std::size_t worst_cs = std::numeric_limits<std::size_t>::max();

    for (std::size_t i = 0; i < m; ++i) {
        const auto& constr = model.constraint(static_cast<model::ConstrId>(i));
        const double y_i = dual[i];

        for (const auto& term : constr.expression.terms()) {
            if (term.var_id < n) {
                AT_lambda[term.var_id] += term.coefficient * y_i;
            }
        }

        // Dual feasibility check on constraint multipliers
        double dual_viol = 0.0;
        if (constr.sense == model::ConstraintSense::LESS_EQUAL) {
            // lambda_i >= 0
            if (y_i < 0.0) {
                dual_viol = -y_i;
            }
        } else if (constr.sense == model::ConstraintSense::GREATER_EQUAL) {
            // lambda_i <= 0
            if (y_i > 0.0) {
                dual_viol = y_i;
            }
        }

        if (dual_viol > max_dual_viol) {
            max_dual_viol = dual_viol;
            worst_dual = i;
        }

        // Complementary slackness on constraint: |slack_i * y_i|
        double slack = 0.0;
        if (i < constraint_activities.size()) {
            slack = std::abs(constr.rhs - constraint_activities[i]);
        }
        const double cs_viol = std::abs(slack * y_i);
        if (cs_viol > max_cs_viol) {
            max_cs_viol = cs_viol;
            worst_cs = i;
        }
    }

    // 3. Bound multipliers: mu_j = grad_j + (A^T lambda)_j
    // For x_j > lb_j and x_j < ub_j (strictly interior), mu_j should be 0.
    // At lower bound (x_j == lb_j): mu_j = mu_lb >= 0. If mu_j < 0, dual violation.
    // At upper bound (x_j == ub_j): mu_j = -mu_ub <= 0. If mu_j > 0, dual violation.
    // For fixed variable (lb_j == ub_j): mu_j can be any sign (equality bound).
    double max_stat_viol = 0.0;
    std::size_t worst_stat = std::numeric_limits<std::size_t>::max();

    const double prim_tol = tolerances_.primal_feasibility_tol;
    for (std::size_t j = 0; j < n; ++j) {
        const auto& var = model.variable(static_cast<model::VarId>(j));
        const double mu_j = grad[j] + AT_lambda[j];
        const double x_j = primal[j];

        const bool at_lb = var.has_lower_bound() && std::abs(x_j - var.lower_bound) <= prim_tol;
        const bool at_ub = var.has_upper_bound() && std::abs(x_j - var.upper_bound) <= prim_tol;
        const bool is_fixed = var.has_lower_bound() && var.has_upper_bound() &&
                              std::abs(var.lower_bound - var.upper_bound) <= prim_tol;

        if (is_fixed) {
            // Fixed variable: equality bound, mu_j can take any value, no violation
        } else if (at_lb && !at_ub) {
            // Lower bound active: required mu_j >= 0
            if (mu_j < 0.0) {
                const double v = -mu_j;
                if (v > max_dual_viol) {
                    max_dual_viol = v;
                    worst_dual = m + j;
                }
            }
        } else if (at_ub && !at_lb) {
            // Upper bound active: required mu_j <= 0
            if (mu_j > 0.0) {
                const double v = mu_j;
                if (v > max_dual_viol) {
                    max_dual_viol = v;
                    worst_dual = m + j;
                }
            }
        } else if (!at_lb && !at_ub) {
            // Strictly interior: stationarity requires mu_j = 0
            const double stat_viol = std::abs(mu_j);
            if (stat_viol > max_stat_viol) {
                max_stat_viol = stat_viol;
                worst_stat = j;
            }
        }

        // Bound complementary slackness
        if (var.has_lower_bound() && !is_fixed) {
            const double dist_lb = std::abs(x_j - var.lower_bound);
            if (mu_j > 0.0) {  // mu_lb > 0
                const double cs_b = dist_lb * mu_j;
                if (cs_b > max_cs_viol) {
                    max_cs_viol = cs_b;
                    worst_cs = m + j;
                }
            }
        }
        if (var.has_upper_bound() && !is_fixed) {
            const double dist_ub = std::abs(var.upper_bound - x_j);
            if (mu_j < 0.0) {  // mu_ub > 0 (since mu_j = -mu_ub)
                const double cs_b = dist_ub * (-mu_j);
                if (cs_b > max_cs_viol) {
                    max_cs_viol = cs_b;
                    worst_cs = m + n + j;
                }
            }
        }
    }

    stat_res.max_violation = max_stat_viol;
    stat_res.worst_index = worst_stat;
    stat_res.passed = (max_stat_viol <= tolerances_.optimality_tol);
    if (stat_res.passed) {
        stat_res.diagnostic = "Stationarity satisfied: max |grad + A^T lambda - mu| = " +
                              std::to_string(max_stat_viol) +
                              " <= tol=" + std::to_string(tolerances_.optimality_tol) + ".";
    } else {
        stat_res.diagnostic = "Stationarity violation at variable " + std::to_string(worst_stat) +
                              ": residual=" + std::to_string(max_stat_viol) +
                              " > tol=" + std::to_string(tolerances_.optimality_tol) + ".";
    }

    dual_feas_result.max_violation = max_dual_viol;
    dual_feas_result.worst_index = worst_dual;
    dual_feas_result.passed = (max_dual_viol <= tolerances_.dual_feasibility_tol);
    if (dual_feas_result.passed) {
        dual_feas_result.diagnostic =
            "Dual feasibility verified (max_viol=" + std::to_string(max_dual_viol) +
            " <= tol=" + std::to_string(tolerances_.dual_feasibility_tol) + ").";
    } else {
        dual_feas_result.diagnostic =
            "Dual feasibility violated at index " + std::to_string(worst_dual) +
            ": max_viol=" + std::to_string(max_dual_viol) +
            " > tol=" + std::to_string(tolerances_.dual_feasibility_tol) + ".";
    }

    cs_result.max_violation = max_cs_viol;
    cs_result.worst_index = worst_cs;
    cs_result.passed = (max_cs_viol <= tolerances_.residual_tol);
    if (cs_result.passed) {
        cs_result.diagnostic =
            "Complementary slackness verified (max_viol=" + std::to_string(max_cs_viol) +
            " <= tol=" + std::to_string(tolerances_.residual_tol) + ").";
    } else {
        cs_result.diagnostic = "Complementary slackness violated at index " +
                               std::to_string(worst_cs) +
                               ": violation=" + std::to_string(max_cs_viol) +
                               " > tol=" + std::to_string(tolerances_.residual_tol) + ".";
    }

    return stat_res;
}

CheckResult QPVerifier::check_objective(const model::CanonicalModel& model,
                                        const std::vector<double>& primal,
                                        double reported_objective, double& recomputed_obj) const {
    CheckResult res;
    const double abs_tol = tolerances_.objective_tol;
    const double rel_tol = tolerances_.objective_tol;
    res.tolerance_used = abs_tol;

    const std::size_t n = model.num_variables();
    double obj = model.objective_offset();

    for (const auto& term : model.objective().linear_terms.terms()) {
        if (term.var_id < n) {
            obj += term.coefficient * primal[term.var_id];
        }
    }

    for (const auto& qt : model.objective().quadratic_terms.terms()) {
        if (qt.var1_id < n && qt.var2_id < n) {
            obj += qt.coefficient * primal[qt.var1_id] * primal[qt.var2_id];
        }
    }

    recomputed_obj = obj;
    const double abs_err = std::abs(obj - reported_objective);
    const double denom = std::max(1.0, std::abs(reported_objective));
    const double rel_err = abs_err / denom;

    res.max_violation = abs_err;
    if (abs_err <= abs_tol || rel_err <= rel_tol) {
        res.passed = true;
        std::ostringstream oss;
        oss << "Objective consistent: recomputed=" << obj << " reported=" << reported_objective
            << " abs_err=" << abs_err << " rel_err=" << rel_err << ".";
        res.diagnostic = oss.str();
    } else {
        res.passed = false;
        std::ostringstream oss;
        oss << "Objective mismatch: recomputed=" << obj << " reported=" << reported_objective
            << " abs_err=" << abs_err << " > abs_tol=" << abs_tol << " AND rel_err=" << rel_err
            << " > rel_tol=" << rel_tol << ".";
        res.diagnostic = oss.str();
    }

    return res;
}

CheckResult QPVerifier::check_psd(const model::CanonicalModel& model) const {
    CheckResult res;
    res.tolerance_used = tolerances_.psd_tol;
    res.passed = true;

    const std::size_t n = model.num_variables();
    // Build dense working representation for PSD check
    std::vector<double> Q(n * n, 0.0);
    for (const auto& qt : model.objective().quadratic_terms.terms()) {
        if (qt.var1_id < n && qt.var2_id < n) {
            if (qt.var1_id == qt.var2_id) {
                Q[qt.var1_id * n + qt.var1_id] += 2.0 * qt.coefficient;
            } else {
                Q[qt.var1_id * n + qt.var2_id] += qt.coefficient;
                Q[qt.var2_id * n + qt.var1_id] += qt.coefficient;
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
        // Find best diagonal pivot among remaining
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
                " (Hessian is indefinite).";
            return res;
        }

        if (max_diag <= tolerances_.psd_tol) {
            // Remaining submatrix must have negligible off-diagonals
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
            break;  // Remainder is zero, matrix is PSD with rank k
        }

        // Swap pivot
        std::swap(p[k], p[best_idx]);
        const std::size_t pk = p[k];
        const double pivot = Q[pk * n + pk];

        // Eliminate
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

    res.passed = true;
    res.max_violation = 0.0;
    res.diagnostic =
        "Hessian Q confirmed positive semidefinite via pivoted Cholesky factorization.";
    return res;
}

QPVerificationReport QPVerifier::verify(const model::CanonicalModel& model,
                                        const std::vector<double>& primal_solution,
                                        const std::vector<double>& dual_multipliers,
                                        double reported_objective,
                                        core::SolverStatus reported_status,
                                        const core::SolverTolerances& solver_tols) const {
    QPVerificationReport report;
    report.original_reported_status = reported_status;

    // Override tolerances if provided
    QPVerifier local_verifier = *this;
    if (solver_tols.primal_feasibility_tol > 0.0) {
        local_verifier.tolerances_ = QPVerifierTolerances::from_solver_tolerances(solver_tols);
    }

    // 1. Non-finite check
    report.nonfinite_check = local_verifier.check_nonfinite(primal_solution, dual_multipliers);
    if (!report.nonfinite_check.passed) {
        report.certified_status = core::SolverStatus::NUMERICAL_FAILURE;
        report.status_downgraded = (reported_status != report.certified_status);
        report.summary = "QP Verification FAILED: " + report.nonfinite_check.diagnostic;
        return report;
    }

    // 2. Bound feasibility
    report.bound_feasibility = local_verifier.check_bounds(model, primal_solution);

    // 3. Primal constraint feasibility
    std::vector<double> activities;
    report.primal_feasibility =
        local_verifier.check_primal_feasibility(model, primal_solution, activities);
    report.aggregate_primal_residual =
        std::max(report.bound_feasibility.max_violation, report.primal_feasibility.max_violation);

    // 4. Hessian PSD check
    report.psd_check = local_verifier.check_psd(model);

    // 5. Objective consistency
    report.objective_consistency = local_verifier.check_objective(
        model, primal_solution, reported_objective, report.recomputed_objective);
    report.objective_absolute_error = report.objective_consistency.max_violation;
    report.objective_relative_error =
        report.objective_absolute_error / std::max(1.0, std::abs(reported_objective));

    // 6. Stationarity, dual feasibility, complementary slackness
    if (reported_status == core::SolverStatus::OPTIMAL) {
        report.stationarity = local_verifier.check_stationarity_and_duals(
            model, primal_solution, dual_multipliers, activities, report.dual_feasibility,
            report.complementary_slackness);
        report.aggregate_stationarity_residual = report.stationarity.max_violation;
        report.aggregate_dual_residual = report.dual_feasibility.max_violation;
        report.aggregate_complementarity_residual = report.complementary_slackness.max_violation;
    }

    // 7. Status certification
    bool all_passed = report.nonfinite_check.passed && report.bound_feasibility.passed &&
                      report.primal_feasibility.passed && report.psd_check.passed &&
                      report.objective_consistency.passed;

    if (reported_status == core::SolverStatus::OPTIMAL) {
        all_passed = all_passed && report.stationarity.passed && report.dual_feasibility.passed &&
                     report.complementary_slackness.passed;
    }

    if (all_passed) {
        report.certified_status = reported_status;
        report.status_downgraded = false;
    } else {
        report.certified_status = core::SolverStatus::NUMERICAL_FAILURE;
        report.status_downgraded = true;
    }

    // 8. Build summary
    std::ostringstream ss;
    ss << "QP Verification [" << core::to_string(report.certified_status) << "]: ";
    if (report.status_downgraded) {
        ss << "(DOWNGRADED from " << core::to_string(reported_status) << ") ";
    }
    ss << "primal_res=" << std::scientific << std::setprecision(2)
       << report.aggregate_primal_residual
       << ", stat_res=" << report.aggregate_stationarity_residual
       << ", dual_res=" << report.aggregate_dual_residual
       << ", cs_res=" << report.aggregate_complementarity_residual
       << ", obj_err=" << report.objective_absolute_error;
    report.summary = ss.str();

    return report;
}

}  // namespace kavach::verification
