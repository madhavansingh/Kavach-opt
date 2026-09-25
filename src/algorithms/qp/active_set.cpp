#include "algorithms/qp/active_set.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "algorithms/lp/simplex.hpp"
#include "core/error.hpp"
#include "numerics/scaling.hpp"
#include "presolve/presolve.hpp"

namespace kavach::algorithms::qp {

namespace {

struct NormalizedConstraint {
    ActiveConstraintKind kind{ActiveConstraintKind::MODEL_CONSTRAINT};
    std::size_t original_index{0};
    bool is_equality{false};
    double rhs{0.0};
    std::vector<std::pair<std::size_t, double>> terms;  // (var_id, coeff)
};

std::string make_working_set_signature(const std::vector<std::size_t>& W) {
    std::vector<std::size_t> sorted_W = W;
    std::sort(sorted_W.begin(), sorted_W.end());
    std::ostringstream oss;
    for (std::size_t idx : sorted_W) {
        oss << idx << ",";
    }
    return oss.str();
}

}  // namespace

numerics::CsrMatrix PrimalActiveSetQPSolver::build_hessian(const model::CanonicalModel& model) {
    const std::size_t n = model.num_variables();
    std::vector<numerics::Triplet> triplets;
    triplets.reserve(model.objective().quadratic_terms.num_terms() * 2 + n);

    // Track which diagonals have been set
    std::vector<bool> has_diag(n, false);

    for (const auto& qt : model.objective().quadratic_terms.terms()) {
        const std::size_t i = qt.var1_id;
        const std::size_t j = qt.var2_id;
        if (i >= n || j >= n) {
            continue;
        }

        if (i == j) {
            // Objective term is q_ii * x_i^2 => 1/2 * x^T Q x requires Q_ii = 2 * q_ii
            triplets.push_back({.row = i, .col = i, .value = 2.0 * qt.coefficient});
            has_diag[i] = true;
        } else {
            // Objective term is q_ij * x_i * x_j => 1/2 * (Q_ij + Q_ji) * x_i * x_j => Q_ij = Q_ji
            // = q_ij
            triplets.push_back({.row = i, .col = j, .value = qt.coefficient});
            triplets.push_back({.row = j, .col = i, .value = qt.coefficient});
        }
    }

    // Ensure every diagonal exists (insert 0.0 if not present) for consistent structure
    for (std::size_t i = 0; i < n; ++i) {
        if (!has_diag[i]) {
            triplets.push_back({.row = i, .col = i, .value = 0.0});
        }
    }

    return numerics::CsrMatrix::from_triplets(n, n, triplets, true);
}

bool PrimalActiveSetQPSolver::check_psd(const numerics::CsrMatrix& Q, double tol,
                                        std::string* error_msg) {
    const std::size_t n = Q.rows();
    if (n == 0) {
        return true;
    }

    // 1. Initial diagonal check: any negative diagonal immediately proves non-PSD
    for (std::size_t i = 0; i < n; ++i) {
        const double d = Q.get(i, i);
        if (d < -tol) {
            if (error_msg) {
                *error_msg = "Negative diagonal entry Q[" + std::to_string(i) + "," +
                             std::to_string(i) + "] = " + std::to_string(d) +
                             " detected: Hessian is indefinite (non-convex).";
            }
            return false;
        }
    }

    // 2. Dense working representation for complete diagonal-pivoted Cholesky factorization
    std::vector<double> S(n * n, 0.0);
    const auto& q_cols = Q.col_idx();
    const auto& q_vals = Q.values();
    for (std::size_t r = 0; r < n; ++r) {
        const std::size_t start = Q.row_start(r);
        const std::size_t end = Q.row_end(r);
        for (std::size_t k = start; k < end; ++k) {
            S[r * n + q_cols[k]] = q_vals[k];
        }
    }

    std::vector<std::size_t> p(n);
    for (std::size_t i = 0; i < n; ++i) {
        p[i] = i;
    }

    for (std::size_t k = 0; k < n; ++k) {
        double max_diag = -1.0;
        std::size_t best_idx = k;
        for (std::size_t i = k; i < n; ++i) {
            const double diag = S[p[i] * n + p[i]];
            if (diag > max_diag) {
                max_diag = diag;
                best_idx = i;
            }
        }

        if (max_diag < -tol) {
            if (error_msg) {
                *error_msg =
                    "Negative pivot in Cholesky factorization: " + std::to_string(max_diag) +
                    " detected (Hessian is indefinite).";
            }
            return false;
        }

        if (max_diag <= tol) {
            // Remainder must be zero
            for (std::size_t i = k; i < n; ++i) {
                for (std::size_t j = k; j < n; ++j) {
                    const double val = std::abs(S[p[i] * n + p[j]]);
                    if (val > tol * 100.0 + 1e-7) {
                        if (error_msg) {
                            *error_msg = "Non-zero off-diagonal entry in null Schur complement: " +
                                         std::to_string(val) + " (Hessian is indefinite).";
                        }
                        return false;
                    }
                }
            }
            break;  // Remainder is null; Q is PSD with rank k
        }

        std::swap(p[k], p[best_idx]);
        const std::size_t pk = p[k];
        const double pivot = S[pk * n + pk];

        for (std::size_t i = k + 1; i < n; ++i) {
            const std::size_t pi = p[i];
            const double factor = S[pi * n + pk] / pivot;
            S[pi * n + pk] = factor;
            for (std::size_t j = k + 1; j < n; ++j) {
                const std::size_t pj = p[j];
                S[pi * n + pj] -= factor * S[pk * n + pj];
            }
        }
    }

    return true;
}

double PrimalActiveSetQPSolver::evaluate_objective(const model::CanonicalModel& model,
                                                   const std::vector<double>& x) {
    const std::size_t n = model.num_variables();
    double obj = model.objective_offset();

    for (const auto& term : model.objective().linear_terms.terms()) {
        if (term.var_id < n && term.var_id < x.size()) {
            obj += term.coefficient * x[term.var_id];
        }
    }

    for (const auto& qt : model.objective().quadratic_terms.terms()) {
        if (qt.var1_id < n && qt.var2_id < n && qt.var1_id < x.size() && qt.var2_id < x.size()) {
            obj += qt.coefficient * x[qt.var1_id] * x[qt.var2_id];
        }
    }

    return obj;
}

bool PrimalActiveSetQPSolver::find_initial_feasible_point(const model::CanonicalModel& model,
                                                          const core::SolverTolerances& tols,
                                                          std::vector<double>& x0) const {
    const std::size_t n = model.num_variables();
    const std::size_t m = model.num_constraints();
    x0.assign(n, 0.0);

    auto is_feasible = [&](const std::vector<double>& pt) {
        if (pt.size() != n)
            return false;
        // Bounds
        for (std::size_t j = 0; j < n; ++j) {
            const auto& v = model.variable(static_cast<model::VarId>(j));
            if (v.has_lower_bound() && v.lower_bound - pt[j] > tols.primal_feasibility_tol) {
                return false;
            }
            if (v.has_upper_bound() && pt[j] - v.upper_bound > tols.primal_feasibility_tol) {
                return false;
            }
        }
        // Constraints
        for (std::size_t i = 0; i < m; ++i) {
            const auto& c = model.constraint(static_cast<model::ConstrId>(i));
            double act = 0.0;
            for (const auto& t : c.expression.terms()) {
                if (t.var_id < n)
                    act += t.coefficient * pt[t.var_id];
            }
            if (c.sense == model::ConstraintSense::LESS_EQUAL &&
                act - c.rhs > tols.primal_feasibility_tol) {
                return false;
            }
            if (c.sense == model::ConstraintSense::GREATER_EQUAL &&
                c.rhs - act > tols.primal_feasibility_tol) {
                return false;
            }
            if (c.sense == model::ConstraintSense::EQUAL &&
                std::abs(act - c.rhs) > tols.primal_feasibility_tol) {
                return false;
            }
        }
        return true;
    };

    // 1. Try x = 0
    std::vector<double> zero_pt(n, 0.0);
    if (is_feasible(zero_pt)) {
        x0 = zero_pt;
        return true;
    }

    // 2. Try bound midpoint
    std::vector<double> mid_pt(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        const auto& v = model.variable(static_cast<model::VarId>(j));
        if (v.has_lower_bound() && v.has_upper_bound()) {
            mid_pt[j] = 0.5 * (v.lower_bound + v.upper_bound);
        } else if (v.has_lower_bound()) {
            mid_pt[j] = std::max(0.0, v.lower_bound);
        } else if (v.has_upper_bound()) {
            mid_pt[j] = std::min(0.0, v.upper_bound);
        } else {
            mid_pt[j] = 0.0;
        }
    }
    if (is_feasible(mid_pt)) {
        x0 = mid_pt;
        return true;
    }

    // 3. Fallback: Phase I feasibility LP subproblem
    model::CanonicalModel phase1("phase1_feasibility");
    for (std::size_t j = 0; j < n; ++j) {
        const auto& v = model.variable(static_cast<model::VarId>(j));
        phase1.add_variable(v.name, model::VariableType::CONTINUOUS, v.lower_bound, v.upper_bound);
    }
    for (std::size_t i = 0; i < m; ++i) {
        const auto& c = model.constraint(static_cast<model::ConstrId>(i));
        phase1.add_constraint(c.name, c.expression, c.sense, c.rhs);
    }
    // Zero objective
    phase1.set_objective_sense(model::ObjectiveSense::MINIMIZE);

    algorithms::lp::RevisedSimplex lp_solver;
    auto lp_res = lp_solver.solve(phase1, tols);
    if (lp_res.status == core::SolverStatus::OPTIMAL && is_feasible(lp_res.primal_solution)) {
        x0 = lp_res.primal_solution;
        return true;
    }

    return false;
}

QPSolution PrimalActiveSetQPSolver::solve(const model::CanonicalModel& model,
                                          const QPSolverOptions& options) {
    const auto start_time = std::chrono::steady_clock::now();
    QPSolution sol;

    // --- 1. Problem Class Validation ---
    model.check_valid();

    if (model.num_integer_variables() > 0 || model.num_binary_variables() > 0) {
        throw core::InputException(
            "PrimalActiveSetQPSolver rejects discrete variables: model contains integer or binary "
            "variables");
    }

    if (!model.objective().is_quadratic()) {
        throw core::InputException(
            "PrimalActiveSetQPSolver requires quadratic objective terms: model has no quadratic "
            "terms");
    }

    // Check finite coefficients in quadratic objective
    for (const auto& qt : model.objective().quadratic_terms.terms()) {
        if (!std::isfinite(qt.coefficient)) {
            throw core::InputException("Non-finite quadratic coefficient detected in objective");
        }
    }

    // --- 2. Hessian Construction & PSD Check ---
    numerics::CsrMatrix Q = build_hessian(model);
    std::string psd_error;
    if (!check_psd(Q, options.tolerances.pivot_tol, &psd_error)) {
        sol.status = core::SolverStatus::NUMERICAL_FAILURE;
        sol.telemetry.error_message = psd_error;
        sol.verification_report.certified_status = core::SolverStatus::NUMERICAL_FAILURE;
        sol.verification_report.summary = psd_error;
        return sol;
    }

    // --- 3. Presolve Pipeline ---
    if (options.enable_presolve) {
        presolve::PresolveOptions p_opts;
        p_opts.tolerances = options.tolerances;
        presolve::Presolver presolver;
        auto p_res = presolver.apply(model, p_opts);

        if (p_res.status == core::SolverStatus::INFEASIBLE) {
            sol.status = core::SolverStatus::INFEASIBLE;
            sol.telemetry.error_message = "Proven infeasible during presolve reductions";
            return sol;
        }

        if (p_res.reduced_model && p_res.reduced_model->num_variables() == 0) {
            // All variables fixed by presolve!
            std::vector<double> empty_red;
            sol.primal_solution = p_res.reconstruction_map.restore_solution(empty_red);
            sol.dual_solution.assign(model.num_constraints(), 0.0);
            sol.objective_value = evaluate_objective(model, sol.primal_solution);
            sol.status = core::SolverStatus::OPTIMAL;

            // Verify against original model
            verification::QPVerifier verifier(
                verification::QPVerifierTolerances::from_solver_tolerances(options.tolerances));
            sol.verification_report =
                verifier.verify(model, sol.primal_solution, sol.dual_solution, sol.objective_value,
                                sol.status, options.tolerances);
            sol.status = sol.verification_report.certified_status;
            return sol;
        }

        const bool has_reductions =
            (p_res.stats.variables_removed > 0 || p_res.stats.rows_removed > 0 ||
             p_res.stats.fixed_variables > 0);

        if (p_res.reduced_model && has_reductions &&
            p_res.reduced_model->objective().is_quadratic()) {
            QPSolverOptions no_presolve_opts = options;
            no_presolve_opts.enable_presolve = false;
            auto red_sol = solve_internal(*p_res.reduced_model, no_presolve_opts, start_time);

            if (red_sol.status == core::SolverStatus::OPTIMAL ||
                red_sol.status == core::SolverStatus::FEASIBLE) {
                sol = red_sol;
                // Reconstruct to original variable space
                sol.primal_solution =
                    p_res.reconstruction_map.restore_solution(red_sol.primal_solution);
                // Map dual multipliers back to original constraints by name
                sol.dual_solution.assign(model.num_constraints(), 0.0);
                for (std::size_t r = 0; r < p_res.reduced_model->num_constraints(); ++r) {
                    const auto& red_c =
                        p_res.reduced_model->constraint(static_cast<model::ConstrId>(r));
                    auto orig_id = model.find_constraint_id(red_c.name);
                    if (orig_id.has_value() && *orig_id < sol.dual_solution.size()) {
                        sol.dual_solution[*orig_id] = red_sol.dual_solution[r];
                    }
                }
                sol.objective_value = evaluate_objective(model, sol.primal_solution);

                // Independent verification against ORIGINAL model
                verification::QPVerifier verifier(
                    verification::QPVerifierTolerances::from_solver_tolerances(options.tolerances));
                sol.verification_report =
                    verifier.verify(model, sol.primal_solution, sol.dual_solution,
                                    sol.objective_value, sol.status, options.tolerances);
                sol.status = sol.verification_report.certified_status;
                return sol;
            } else {
                sol = red_sol;
                return sol;
            }
        }
    }

    // Direct solve on current model
    return solve_internal(model, options, start_time);
}

QPSolution PrimalActiveSetQPSolver::solve_internal(
    const model::CanonicalModel& model, const QPSolverOptions& opts,
    const std::chrono::steady_clock::time_point& start_time) const {
    QPSolution sol;
    const std::size_t n = model.num_variables();
    const std::size_t m = model.num_constraints();

    sol.telemetry.num_variables = n;
    sol.telemetry.num_constraints = m;

    // --- 1. Build Hessian and Linear Cost ---
    numerics::CsrMatrix Q = build_hessian(model);
    sol.telemetry.q_nonzeros = Q.nonzeros();
    sol.telemetry.a_nonzeros = model.num_nonzeros();

    std::vector<double> c(n, 0.0);
    for (const auto& lt : model.objective().linear_terms.terms()) {
        if (lt.var_id < n) {
            c[lt.var_id] += lt.coefficient;
        }
    }

    // --- 2. Initial Feasible Point ---
    std::vector<double> x;
    if (!find_initial_feasible_point(model, opts.tolerances, x)) {
        sol.status = core::SolverStatus::INFEASIBLE;
        sol.telemetry.error_message =
            "No feasible starting point found: constraints are infeasible";
        return sol;
    }

    // --- 3. Build Unified Normalized Constraints: a_l^T x <= b_l or a_l^T x == b_l ---
    std::vector<NormalizedConstraint> all_constrs;
    all_constrs.reserve(m + 2 * n);

    // Model constraints
    for (std::size_t i = 0; i < m; ++i) {
        const auto& mc = model.constraint(static_cast<model::ConstrId>(i));
        NormalizedConstraint nc;
        nc.kind = ActiveConstraintKind::MODEL_CONSTRAINT;
        nc.original_index = i;

        if (mc.sense == model::ConstraintSense::LESS_EQUAL) {
            nc.is_equality = false;
            nc.rhs = mc.rhs;
            for (const auto& t : mc.expression.terms()) {
                if (t.var_id < n)
                    nc.terms.push_back({t.var_id, t.coefficient});
            }
        } else if (mc.sense == model::ConstraintSense::GREATER_EQUAL) {
            // A x >= b <=> -A x <= -b
            nc.is_equality = false;
            nc.rhs = -mc.rhs;
            for (const auto& t : mc.expression.terms()) {
                if (t.var_id < n)
                    nc.terms.push_back({t.var_id, -t.coefficient});
            }
        } else {
            // Equality
            nc.is_equality = true;
            nc.rhs = mc.rhs;
            for (const auto& t : mc.expression.terms()) {
                if (t.var_id < n)
                    nc.terms.push_back({t.var_id, t.coefficient});
            }
        }
        all_constrs.push_back(std::move(nc));
    }

    // Variable lower bounds: x_j >= l_j <=> -x_j <= -l_j
    for (std::size_t j = 0; j < n; ++j) {
        const auto& var = model.variable(static_cast<model::VarId>(j));
        if (var.has_lower_bound()) {
            NormalizedConstraint nc;
            nc.kind = ActiveConstraintKind::LOWER_BOUND;
            nc.original_index = j;
            if (var.has_upper_bound() &&
                std::abs(var.lower_bound - var.upper_bound) <= opts.tolerances.pivot_tol) {
                // Fixed variable: x_j = l_j
                nc.is_equality = true;
                nc.rhs = var.lower_bound;
                nc.terms.push_back({j, 1.0});
            } else {
                nc.is_equality = false;
                nc.rhs = -var.lower_bound;
                nc.terms.push_back({j, -1.0});
            }
            all_constrs.push_back(std::move(nc));
        }
    }

    // Variable upper bounds: x_j <= u_j
    for (std::size_t j = 0; j < n; ++j) {
        const auto& var = model.variable(static_cast<model::VarId>(j));
        if (var.has_upper_bound()) {
            if (var.has_lower_bound() &&
                std::abs(var.lower_bound - var.upper_bound) <= opts.tolerances.pivot_tol) {
                // Fixed variable handled in lower bound
                continue;
            }
            NormalizedConstraint nc;
            nc.kind = ActiveConstraintKind::UPPER_BOUND;
            nc.original_index = j;
            nc.is_equality = false;
            nc.rhs = var.upper_bound;
            nc.terms.push_back({j, 1.0});
            all_constrs.push_back(std::move(nc));
        }
    }

    const std::size_t num_total_constrs = all_constrs.size();

    // Helper: compute a_l^T v
    auto dot_constr = [&](const NormalizedConstraint& c, const std::vector<double>& v) {
        double val = 0.0;
        for (const auto& [var_id, coeff] : c.terms) {
            val += coeff * v[var_id];
        }
        return val;
    };

    // Helper: dense row extraction and Gram-Schmidt linear independence check
    auto get_dense_a = [&](const NormalizedConstraint& c) {
        std::vector<double> a(n, 0.0);
        for (const auto& [var_id, coeff] : c.terms) {
            if (var_id < n)
                a[var_id] += coeff;
        }
        return a;
    };

    auto is_linearly_independent = [&](const std::vector<double>& a,
                                       const std::vector<std::vector<double>>& b) {
        double norm_a = 0.0;
        for (double val : a)
            norm_a += val * val;
        norm_a = std::sqrt(norm_a);
        if (norm_a < 1e-12)
            return false;

        std::vector<double> v = a;
        for (const auto& q : b) {
            double dot = 0.0;
            for (std::size_t j = 0; j < n; ++j)
                dot += q[j] * v[j];
            for (std::size_t j = 0; j < n; ++j)
                v[j] -= dot * q[j];
        }
        double norm_v = 0.0;
        for (double val : v)
            norm_v += val * val;
        norm_v = std::sqrt(norm_v);
        return norm_v > 1e-4 * norm_a;
    };

    auto add_to_basis = [&](const std::vector<double>& a, std::vector<std::vector<double>>& b) {
        std::vector<double> v = a;
        for (const auto& q : b) {
            double dot = 0.0;
            for (std::size_t j = 0; j < n; ++j)
                dot += q[j] * v[j];
            for (std::size_t j = 0; j < n; ++j)
                v[j] -= dot * q[j];
        }
        double norm_v = 0.0;
        for (double val : v)
            norm_v += val * val;
        norm_v = std::sqrt(norm_v);
        if (norm_v > 1e-12) {
            for (double& val : v)
                val /= norm_v;
            b.push_back(std::move(v));
        }
    };

    auto rebuild_basis = [&](const std::vector<std::size_t>& current_W) {
        std::vector<std::vector<double>> new_basis;
        new_basis.reserve(current_W.size());
        for (std::size_t idx : current_W) {
            auto a = get_dense_a(all_constrs[idx]);
            add_to_basis(a, new_basis);
        }
        return new_basis;
    };

    // --- 4. Initialize Working Set W with LICQ Filter ---
    std::vector<std::size_t> W;
    W.reserve(n);
    std::vector<std::vector<double>> basis;
    basis.reserve(n);

    // Always include all equality constraints (if linearly independent)
    for (std::size_t l = 0; l < num_total_constrs; ++l) {
        if (all_constrs[l].is_equality) {
            auto a = get_dense_a(all_constrs[l]);
            if (is_linearly_independent(a, basis)) {
                W.push_back(l);
                add_to_basis(a, basis);
            }
        }
    }

    // Add inequality constraints active at x0 (if linearly independent)
    for (std::size_t l = 0; l < num_total_constrs; ++l) {
        if (!all_constrs[l].is_equality) {
            const double ax = dot_constr(all_constrs[l], x);
            if (std::abs(ax - all_constrs[l].rhs) <= opts.tolerances.primal_feasibility_tol) {
                if (W.size() < n) {
                    auto a = get_dense_a(all_constrs[l]);
                    if (is_linearly_independent(a, basis)) {
                        W.push_back(l);
                        add_to_basis(a, basis);
                    }
                }
            }
        }
    }

    // Anti-cycling tracking
    std::unordered_set<std::string> visited_sets;
    std::size_t cycling_hits = 0;

    sol.status = core::SolverStatus::UNKNOWN;
    std::size_t iter = 0;

    // Dual multiplier store for final output
    std::vector<double> lambda_out(m, 0.0);
    std::vector<double> mu_out(n, 0.0);

    // --- 5. Main Active-Set Iteration Loop ---
    while (iter < opts.max_iterations) {
        iter++;
        sol.telemetry.iterations = iter;
        sol.telemetry.active_constraints = W.size();

        // Check time limit
        if (opts.time_limit_seconds > 0.0) {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - start_time).count();
            if (elapsed > opts.time_limit_seconds) {
                sol.status = core::SolverStatus::LIMIT_REACHED;
                sol.telemetry.error_message = "Time limit exceeded during active-set iteration";
                break;
            }
        }

        // Gradient g = Q * x + c
        std::vector<double> Qx = Q.spmv(x);
        std::vector<double> g(n);
        for (std::size_t j = 0; j < n; ++j) {
            g[j] = Qx[j] + c[j];
        }

        // --- 5a. Formulate and Solve KKT System for Working Set W ---
        const std::size_t K = W.size();
        const std::size_t dim = n + K;

        std::vector<numerics::Triplet> kkt_triplets;
        kkt_triplets.reserve(Q.nonzeros() + 2 * K * n + n);

        // Top-left: Q + small regularization on diagonal for numerical stability on semidefinite Q
        const auto& kkt_q_cols = Q.col_idx();
        const auto& kkt_q_vals = Q.values();
        for (std::size_t r = 0; r < n; ++r) {
            const std::size_t start = Q.row_start(r);
            const std::size_t end = Q.row_end(r);
            for (std::size_t idx = start; idx < end; ++idx) {
                double val = kkt_q_vals[idx];
                if (r == kkt_q_cols[idx]) {
                    val += 1e-12;  // Tikhonov regularization
                }
                kkt_triplets.push_back({.row = r, .col = kkt_q_cols[idx], .value = val});
            }
        }

        // Top-right and Bottom-left: A_W^T and A_W
        for (std::size_t k = 0; k < K; ++k) {
            const auto& constr = all_constrs[W[k]];
            for (const auto& [var_id, coeff] : constr.terms) {
                // A_W row k: row (n + k), col var_id
                kkt_triplets.push_back({.row = n + k, .col = var_id, .value = coeff});
                // A_W^T col k: row var_id, col (n + k)
                kkt_triplets.push_back({.row = var_id, .col = n + k, .value = coeff});
            }
        }

        numerics::CsrMatrix KKT_matrix =
            numerics::CsrMatrix::from_triplets(dim, dim, kkt_triplets, true);

        // RHS: [ -g ; -(A_W x - b_W) ]
        std::vector<double> rhs(dim, 0.0);
        for (std::size_t j = 0; j < n; ++j) {
            rhs[j] = -g[j];
        }
        for (std::size_t k = 0; k < K; ++k) {
            const double ax = dot_constr(all_constrs[W[k]], x);
            rhs[n + k] = -(ax - all_constrs[W[k]].rhs);
        }

        numerics::SparseLU lu;
        bool factor_ok = lu.factorize(KKT_matrix, opts.tolerances.pivot_tol);
        sol.telemetry.kkt_solves++;

        if (!factor_ok) {
            sol.telemetry.refactorizations++;
            // Try with relaxed pivot tolerance
            factor_ok = lu.factorize(KKT_matrix, opts.tolerances.pivot_tol * 100.0);
        }

        if (!factor_ok) {
            sol.status = core::SolverStatus::NUMERICAL_FAILURE;
            sol.telemetry.error_message = "Singular KKT matrix encountered during active-set solve";
            break;
        }

        std::vector<double> kkt_sol = lu.solve(rhs);
        (void)lu.refine_solution(KKT_matrix, rhs, kkt_sol);

        std::vector<double> p(n);
        for (std::size_t j = 0; j < n; ++j) {
            p[j] = kkt_sol[j];
        }
        std::vector<double> lambda_W(K);
        for (std::size_t k = 0; k < K; ++k) {
            lambda_W[k] = kkt_sol[n + k];
        }

        // Compute ||p||_inf
        double p_norm = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            p_norm = std::max(p_norm, std::abs(p[j]));
        }

        // --- 5b. Branch on Step Norm ---
        if (p_norm <= opts.tolerances.optimality_tol) {
            // Stationary point on current working manifold!
            // Check dual feasibility: lambda_W[k] >= -dual_feasibility_tol for inequalities
            double most_negative_mult = 0.0;
            int drop_k = -1;

            for (std::size_t k = 0; k < K; ++k) {
                if (!all_constrs[W[k]].is_equality) {
                    if (lambda_W[k] < most_negative_mult) {
                        most_negative_mult = lambda_W[k];
                        drop_k = static_cast<int>(k);
                    }
                }
            }

            if (drop_k == -1 || most_negative_mult >= -opts.tolerances.dual_feasibility_tol) {
                // OPTIMAL SOLUTION FOUND!
                sol.status = core::SolverStatus::OPTIMAL;

                // Extract final dual multipliers
                for (std::size_t k = 0; k < K; ++k) {
                    const auto& constr = all_constrs[W[k]];
                    const double mult = lambda_W[k];

                    if (constr.kind == ActiveConstraintKind::MODEL_CONSTRAINT) {
                        const auto& mc =
                            model.constraint(static_cast<model::ConstrId>(constr.original_index));
                        if (mc.sense == model::ConstraintSense::GREATER_EQUAL) {
                            lambda_out[constr.original_index] = -mult;
                        } else {
                            lambda_out[constr.original_index] = mult;
                        }
                    } else if (constr.kind == ActiveConstraintKind::LOWER_BOUND) {
                        mu_out[constr.original_index] += mult;  // mu_lb >= 0
                    } else if (constr.kind == ActiveConstraintKind::UPPER_BOUND) {
                        mu_out[constr.original_index] -= mult;  // mu_ub >= 0 => net mu <= 0
                    }
                }
                break;
            } else {
                // Drop constraint with most negative multiplier
                W.erase(W.begin() + drop_k);
                basis = rebuild_basis(W);
                sol.telemetry.constraints_removed++;
                continue;
            }
        } else {
            // --- 5c. Step along direction p ---
            double alpha_max = 1.0;
            int blocking_l = -1;

            for (std::size_t l = 0; l < num_total_constrs; ++l) {
                // Skip constraints already in W
                if (std::find(W.begin(), W.end(), l) != W.end()) {
                    continue;
                }

                const double ap = dot_constr(all_constrs[l], p);
                if (ap > opts.tolerances.pivot_tol) {
                    const double ax = dot_constr(all_constrs[l], x);
                    const double dist = all_constrs[l].rhs - ax;
                    double step = 0.0;
                    if (dist > 0.0) {
                        step = dist / ap;
                    }
                    if (step < alpha_max) {
                        alpha_max = step;
                        blocking_l = static_cast<int>(l);
                    }
                }
            }

            const double alpha = std::max(0.0, std::min(1.0, alpha_max));
            for (std::size_t j = 0; j < n; ++j) {
                x[j] += alpha * p[j];
            }

            if (alpha < 1.0 - opts.tolerances.pivot_tol && blocking_l >= 0) {
                auto a = get_dense_a(all_constrs[static_cast<std::size_t>(blocking_l)]);
                if (is_linearly_independent(a, basis)) {
                    W.push_back(static_cast<std::size_t>(blocking_l));
                    add_to_basis(a, basis);
                    sol.telemetry.constraints_added++;
                }
            }

            // Anti-cycling protection
            const std::string sig = make_working_set_signature(W);
            if (alpha <= opts.tolerances.pivot_tol * 10.0) {
                if (visited_sets.contains(sig)) {
                    cycling_hits++;
                    if (cycling_hits >= 5) {
                        sol.status = core::SolverStatus::NUMERICAL_FAILURE;
                        sol.telemetry.error_message =
                            "Active-set cycling detected on degenerate working set";
                        break;
                    }
                }
            }
            visited_sets.insert(sig);
        }
    }

    if (iter >= opts.max_iterations && sol.status != core::SolverStatus::OPTIMAL) {
        sol.status = core::SolverStatus::LIMIT_REACHED;
    }

    sol.primal_solution = std::move(x);
    sol.dual_solution = std::move(lambda_out);
    sol.bound_multipliers = std::move(mu_out);
    sol.objective_value = evaluate_objective(model, sol.primal_solution);
    sol.iterations = iter;

    const auto end_time = std::chrono::steady_clock::now();
    sol.telemetry.solve_time_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // --- 6. Independent Verification ---
    verification::QPVerifier verifier(
        verification::QPVerifierTolerances::from_solver_tolerances(opts.tolerances));
    sol.verification_report = verifier.verify(model, sol.primal_solution, sol.dual_solution,
                                              sol.objective_value, sol.status, opts.tolerances);

    sol.status = sol.verification_report.certified_status;
    sol.kkt_residual = sol.verification_report.aggregate_stationarity_residual;
    sol.telemetry.max_stationarity_residual =
        sol.verification_report.aggregate_stationarity_residual;
    sol.telemetry.max_primal_residual = sol.verification_report.aggregate_primal_residual;
    sol.telemetry.max_dual_residual = sol.verification_report.aggregate_dual_residual;
    sol.telemetry.max_complementarity_residual =
        sol.verification_report.aggregate_complementarity_residual;

    return sol;
}

}  // namespace kavach::algorithms::qp
