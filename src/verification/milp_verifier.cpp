#include "verification/milp_verifier.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>

#include "core/status.hpp"

namespace kavach::verification {

double MILPVerifier::compute_row_activity(const model::CanonicalModel& model, std::size_t row_idx,
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

CheckResult MILPVerifier::check_nonfinite(const std::vector<double>& x) const {
    CheckResult r;
    r.passed = true;
    r.tolerance_used = 0.0;

    for (std::size_t i = 0; i < x.size(); ++i) {
        if (std::isnan(x[i])) {
            r.passed = false;
            r.worst_index = i;
            r.max_violation = std::numeric_limits<double>::infinity();
            r.diagnostic =
                "NaN detected in primal incumbent vector at variable " + std::to_string(i);
            return r;
        }
        if (std::isinf(x[i])) {
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

CheckResult MILPVerifier::check_bound_feasibility(const model::CanonicalModel& model,
                                                  const std::vector<double>& x) const {
    CheckResult r;
    r.passed = true;
    r.tolerance_used = tols_.bound_feasibility_tol;

    double max_viol = 0.0;
    std::size_t worst_var = std::numeric_limits<std::size_t>::max();

    const std::size_t n = std::min(x.size(), model.num_variables());
    for (std::size_t j = 0; j < n; ++j) {
        const auto& var = model.variable(static_cast<model::VarId>(j));
        double val = x[j];
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
    if (max_viol > tols_.bound_feasibility_tol) {
        r.passed = false;
        r.diagnostic = "Bound violation (" + std::to_string(max_viol) + " > tol " +
                       std::to_string(tols_.bound_feasibility_tol) + ") at variable " +
                       std::to_string(worst_var);
    } else {
        r.passed = true;
        r.diagnostic =
            "All variable bounds satisfied (max violation: " + std::to_string(max_viol) + ")";
    }
    return r;
}

CheckResult MILPVerifier::check_primal_feasibility(const model::CanonicalModel& model,
                                                   const std::vector<double>& x) const {
    CheckResult r;
    r.passed = true;
    r.tolerance_used = tols_.primal_feasibility_tol;

    double max_viol = 0.0;
    std::size_t worst_row = std::numeric_limits<std::size_t>::max();

    for (std::size_t i = 0; i < model.num_constraints(); ++i) {
        const auto& constr = model.constraint(static_cast<model::ConstrId>(i));
        double activity = compute_row_activity(model, i, x);
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
    if (max_viol > tols_.primal_feasibility_tol) {
        r.passed = false;
        r.diagnostic = "Primal constraint violation (" + std::to_string(max_viol) + " > tol " +
                       std::to_string(tols_.primal_feasibility_tol) + ") at constraint " +
                       std::to_string(worst_row);
    } else {
        r.passed = true;
        r.diagnostic =
            "All constraints satisfied (max violation: " + std::to_string(max_viol) + ")";
    }
    return r;
}

CheckResult MILPVerifier::check_integrality(const model::CanonicalModel& model,
                                            const std::vector<double>& x) const {
    CheckResult r;
    r.passed = true;
    r.tolerance_used = tols_.integrality_tol;

    double max_viol = 0.0;
    std::size_t worst_var = std::numeric_limits<std::size_t>::max();

    const std::size_t n = std::min(x.size(), model.num_variables());
    for (std::size_t j = 0; j < n; ++j) {
        const auto& var = model.variable(static_cast<model::VarId>(j));
        if (var.is_integer() || var.is_binary()) {
            double val = x[j];
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
    if (max_viol > tols_.integrality_tol) {
        r.passed = false;
        r.diagnostic = "Integrality violation (" + std::to_string(max_viol) + " > tol " +
                       std::to_string(tols_.integrality_tol) + ") at variable " +
                       std::to_string(worst_var);
    } else {
        r.passed = true;
        r.diagnostic = "All integer variables satisfy integrality (max violation: " +
                       std::to_string(max_viol) + ")";
    }
    return r;
}

CheckResult MILPVerifier::check_objective(const model::CanonicalModel& model,
                                          const std::vector<double>& x, double reported_obj,
                                          double& recomputed_obj_out) const {
    CheckResult r;
    r.tolerance_used = tols_.objective_abs_tol;

    double rec_obj = model.objective_offset();
    for (const auto& term : model.objective().linear_terms.terms()) {
        if (term.var_id < x.size()) {
            rec_obj += term.coefficient * x[term.var_id];
        }
    }
    recomputed_obj_out = rec_obj;

    double abs_err = std::abs(rec_obj - reported_obj);
    double rel_err = abs_err / std::max(1.0, std::abs(reported_obj));

    r.max_violation = abs_err;
    if (abs_err <= tols_.objective_abs_tol || rel_err <= tols_.objective_rel_tol) {
        r.passed = true;
        r.diagnostic =
            "Objective consistent with reported value (err: " + std::to_string(abs_err) + ")";
    } else {
        r.passed = false;
        r.diagnostic = "Objective discrepancy: recomputed=" + std::to_string(rec_obj) +
                       ", reported=" + std::to_string(reported_obj) +
                       " (abs_err=" + std::to_string(abs_err) +
                       ", rel_err=" + std::to_string(rel_err) + ")";
    }
    return r;
}

CheckResult MILPVerifier::check_gap(double recomputed_obj, double best_bound,
                                    model::ObjectiveSense /*sense*/,
                                    core::SolverStatus reported_status,
                                    double& recomputed_gap_out) const {
    CheckResult r;
    r.tolerance_used = tols_.gap_tolerance;

    if (std::isinf(best_bound) || std::isnan(best_bound)) {
        recomputed_gap_out = std::numeric_limits<double>::infinity();
        r.max_violation = std::numeric_limits<double>::infinity();
        if (reported_status == core::SolverStatus::OPTIMAL) {
            r.passed = false;
            r.diagnostic = "Infinite dual bound reported with OPTIMAL status";
        } else {
            r.passed = true;
            r.diagnostic = "Dual bound is infinite (non-optimal status)";
        }
        return r;
    }

    double diff = std::abs(recomputed_obj - best_bound);
    double denom = std::max(1.0, std::abs(recomputed_obj));
    double gap = diff / denom;
    recomputed_gap_out = gap;
    r.max_violation = gap;

    if (reported_status == core::SolverStatus::OPTIMAL) {
        if (gap <= tols_.gap_tolerance) {
            r.passed = true;
            r.diagnostic = "Optimality gap closed: " + std::to_string(gap) + " <= tol " +
                           std::to_string(tols_.gap_tolerance);
        } else {
            r.passed = false;
            r.diagnostic =
                "Optimality gap not closed for OPTIMAL claim: gap=" + std::to_string(gap) +
                " > tol " + std::to_string(tols_.gap_tolerance);
        }
    } else {
        r.passed = true;
        r.diagnostic = "Reported non-optimal status; gap is " + std::to_string(gap);
    }
    return r;
}

CheckResult MILPVerifier::check_status_consistency(core::SolverStatus reported_status,
                                                   bool has_incumbent, const CheckResult& nonfinite,
                                                   const CheckResult& bounds,
                                                   const CheckResult& primal,
                                                   const CheckResult& integrality,
                                                   const CheckResult& obj, const CheckResult& gap,
                                                   core::SolverStatus& certified_status_out) const {
    CheckResult r;
    r.passed = true;

    if (!nonfinite.passed) {
        r.passed = false;
        certified_status_out = core::SolverStatus::NUMERICAL_FAILURE;
        r.diagnostic = "Nonfinite values in incumbent; certified status NUMERICAL_FAILURE";
        return r;
    }

    switch (reported_status) {
        case core::SolverStatus::OPTIMAL: {
            if (!has_incumbent) {
                r.passed = false;
                certified_status_out = core::SolverStatus::NUMERICAL_FAILURE;
                r.diagnostic = "OPTIMAL reported without an incumbent solution";
                return r;
            }
            if (!bounds.passed || !primal.passed) {
                r.passed = false;
                certified_status_out = core::SolverStatus::NUMERICAL_FAILURE;
                r.diagnostic = "OPTIMAL reported but primal constraints/bounds violated";
                return r;
            }
            if (!integrality.passed) {
                r.passed = false;
                certified_status_out = core::SolverStatus::NUMERICAL_FAILURE;
                r.diagnostic = "OPTIMAL reported but integrality condition violated";
                return r;
            }
            if (!obj.passed) {
                r.passed = false;
                certified_status_out = core::SolverStatus::NUMERICAL_FAILURE;
                r.diagnostic = "OPTIMAL reported but objective value is inaccurate";
                return r;
            }
            if (!gap.passed) {
                r.passed = false;
                certified_status_out = core::SolverStatus::FEASIBLE;
                r.diagnostic =
                    "OPTIMAL reported but optimality gap is not closed; downgraded to FEASIBLE";
                return r;
            }
            certified_status_out = core::SolverStatus::OPTIMAL;
            r.diagnostic = "OPTIMAL status certified by independent verification";
            break;
        }

        case core::SolverStatus::FEASIBLE: {
            if (!has_incumbent) {
                r.passed = false;
                certified_status_out = core::SolverStatus::LIMIT_REACHED;
                r.diagnostic = "FEASIBLE reported without incumbent; downgraded to LIMIT_REACHED";
                return r;
            }
            if (!bounds.passed || !primal.passed || !integrality.passed) {
                r.passed = false;
                certified_status_out = core::SolverStatus::NUMERICAL_FAILURE;
                r.diagnostic = "FEASIBLE reported but feasibility/integrality conditions violated";
                return r;
            }
            certified_status_out = core::SolverStatus::FEASIBLE;
            r.diagnostic = "FEASIBLE status verified independently";
            break;
        }

        case core::SolverStatus::INFEASIBLE: {
            if (has_incumbent) {
                r.passed = false;
                certified_status_out = core::SolverStatus::NUMERICAL_FAILURE;
                r.diagnostic = "INFEASIBLE reported but an incumbent solution exists";
                return r;
            }
            certified_status_out = core::SolverStatus::INFEASIBLE;
            r.diagnostic = "INFEASIBLE status accepted (no incumbent)";
            break;
        }

        case core::SolverStatus::UNBOUNDED: {
            certified_status_out = core::SolverStatus::UNBOUNDED;
            r.diagnostic = "UNBOUNDED status accepted";
            break;
        }

        case core::SolverStatus::LIMIT_REACHED: {
            if (has_incumbent && bounds.passed && primal.passed && integrality.passed) {
                certified_status_out = core::SolverStatus::FEASIBLE;
                r.diagnostic = "Valid incumbent exists at LIMIT_REACHED; certified as FEASIBLE";
            } else {
                certified_status_out = core::SolverStatus::LIMIT_REACHED;
                r.diagnostic = "LIMIT_REACHED certified";
            }
            break;
        }

        default: {
            certified_status_out = reported_status;
            r.diagnostic = "Status passed through without verification";
            break;
        }
    }

    return r;
}

MILPVerificationReport MILPVerifier::verify(const model::CanonicalModel& model,
                                            const std::vector<double>& incumbent, double best_bound,
                                            double reported_objective,
                                            core::SolverStatus reported_status,
                                            double reported_gap) const {
    MILPVerificationReport rpt;
    rpt.original_reported_status = reported_status;
    rpt.incumbent_available = (!incumbent.empty() && incumbent.size() == model.num_variables());

    if (rpt.incumbent_available) {
        rpt.nonfinite_check = check_nonfinite(incumbent);
        rpt.bound_feasibility = check_bound_feasibility(model, incumbent);
        rpt.primal_feasibility = check_primal_feasibility(model, incumbent);
        rpt.integrality = check_integrality(model, incumbent);

        double recomputed_obj = 0.0;
        rpt.objective_consistency =
            check_objective(model, incumbent, reported_objective, recomputed_obj);
        rpt.recomputed_objective = recomputed_obj;
        rpt.objective_absolute_error = std::abs(recomputed_obj - reported_objective);

        rpt.aggregate_primal_residual = rpt.primal_feasibility.max_violation;
        rpt.max_bound_violation = rpt.bound_feasibility.max_violation;
        rpt.max_integrality_violation = rpt.integrality.max_violation;

        double recomputed_gap = 0.0;
        rpt.gap_consistency = check_gap(recomputed_obj, best_bound, model.objective_sense(),
                                        reported_status, recomputed_gap);
        rpt.recomputed_gap = (reported_gap >= 0.0 && reported_status != core::SolverStatus::OPTIMAL)
                                 ? reported_gap
                                 : recomputed_gap;
    } else {
        rpt.nonfinite_check.passed = true;
        rpt.bound_feasibility.passed = (reported_status != core::SolverStatus::OPTIMAL);
        rpt.primal_feasibility.passed = (reported_status != core::SolverStatus::OPTIMAL);
        rpt.integrality.passed = (reported_status != core::SolverStatus::OPTIMAL);
        rpt.objective_consistency.passed = (reported_status != core::SolverStatus::OPTIMAL);
        rpt.gap_consistency.passed = (reported_status != core::SolverStatus::OPTIMAL);
        rpt.recomputed_gap = std::numeric_limits<double>::infinity();
    }

    core::SolverStatus certified = core::SolverStatus::UNKNOWN;
    rpt.status_consistency =
        check_status_consistency(reported_status, rpt.incumbent_available, rpt.nonfinite_check,
                                 rpt.bound_feasibility, rpt.primal_feasibility, rpt.integrality,
                                 rpt.objective_consistency, rpt.gap_consistency, certified);
    rpt.certified_status = certified;
    rpt.status_downgraded = (certified != reported_status);

    std::ostringstream ss;
    ss << "MILP Verification [" << core::to_string(certified) << "]: "
       << (rpt.status_downgraded
               ? "(DOWNGRADED from " + std::string(core::to_string(reported_status)) + ") "
               : "")
       << "primal_viol=" << std::scientific << std::setprecision(2) << rpt.aggregate_primal_residual
       << ", int_viol=" << rpt.max_integrality_violation << ", gap=" << rpt.recomputed_gap;
    rpt.summary = ss.str();

    return rpt;
}

MILPVerificationReport MILPVerifier::verify(const model::CanonicalModel& model,
                                            const std::vector<double>& incumbent, double best_bound,
                                            double reported_objective,
                                            core::SolverStatus reported_status,
                                            const core::SolverTolerances& solver_tols,
                                            double reported_gap) const {
    MILPVerifier verifier(MILPVerifierTolerances::from_solver_tolerances(solver_tols));
    return verifier.verify(model, incumbent, best_bound, reported_objective, reported_status,
                           reported_gap);
}

}  // namespace kavach::verification
