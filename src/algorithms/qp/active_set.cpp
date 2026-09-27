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
                                        std::string* error_msg, std::size_t* peak_working_bytes) {
    if (peak_working_bytes) {
        *peak_working_bytes = 0;
    }
    const std::size_t n = Q.rows();
    if (n == 0) {
        return true;
    }

    const auto& q_cols = Q.col_idx();
    const auto& q_vals = Q.values();

    // 1. Sparse pass over CSR: check diagonals, null-diagonal off-diagonal coupling,
    //    and Gershgorin diagonal dominance in O(nnz(Q)) time and O(1) auxiliary space.
    bool is_purely_diagonal = true;
    bool is_diagonally_dominant = true;

    for (std::size_t r = 0; r < n; ++r) {
        double diag_val = 0.0;
        double off_diag_abs_sum = 0.0;
        double max_off_diag_in_row = 0.0;

        const std::size_t start = Q.row_start(r);
        const std::size_t end = Q.row_end(r);
        for (std::size_t k = start; k < end; ++k) {
            const std::size_t c = q_cols[k];
            const double val = q_vals[k];
            if (c == r) {
                diag_val += val;
            } else {
                const double abs_val = std::abs(val);
                if (abs_val > 1e-15) {
                    is_purely_diagonal = false;
                    off_diag_abs_sum += abs_val;
                    max_off_diag_in_row = std::max(max_off_diag_in_row, abs_val);
                }
            }
        }

        if (diag_val < -tol) {
            if (error_msg) {
                *error_msg = "Negative diagonal entry Q[" + std::to_string(r) + "," +
                             std::to_string(r) + "] = " + std::to_string(diag_val) +
                             " detected: Hessian is indefinite (non-convex).";
            }
            return false;
        }

        if (diag_val <= tol && max_off_diag_in_row > tol * 100.0 + 1e-7) {
            if (error_msg) {
                *error_msg = "Non-zero off-diagonal entry in null Schur complement: " +
                             std::to_string(max_off_diag_in_row) + " (Hessian is indefinite).";
            }
            return false;
        }

        if (diag_val + tol < off_diag_abs_sum) {
            is_diagonally_dominant = false;
        }
    }

    // Fast-path A: Purely diagonal matrix with all Q_ii >= -tol is provably PSD (0 working bytes)
    // Fast-path B: Symmetric diagonally dominant matrix with Q_ii >= sum_{j!=i} |Q_ij| - tol
    //              has all eigenvalues >= -tol by Gershgorin's Circle Theorem (0 working bytes)
    if (is_purely_diagonal || is_diagonally_dominant) {
        return true;
    }

    // 2. Connected-Component Decomposition of the off-diagonal sparsity graph of Q.
    //    Since Q is symmetric block-diagonal across connected components C_1, ..., C_r,
    //    Q is PSD if and only if every principal submatrix Q[C_c, C_c] is PSD.
    std::vector<std::uint8_t> visited(n, 0);
    std::vector<std::size_t> global_to_local(n, 0);
    std::vector<std::size_t> comp_nodes;
    comp_nodes.reserve(std::min<std::size_t>(n, 64));

    std::size_t max_comp_bytes = visited.capacity() * sizeof(std::uint8_t) +
                                 global_to_local.capacity() * sizeof(std::size_t);

    for (std::size_t root = 0; root < n; ++root) {
        if (visited[root]) {
            continue;
        }

        // Check if root has any non-zero off-diagonal entries
        bool has_off_diag = false;
        for (std::size_t k = Q.row_start(root); k < Q.row_end(root); ++k) {
            if (q_cols[k] != root && std::abs(q_vals[k]) > 1e-15) {
                has_off_diag = true;
                break;
            }
        }
        if (!has_off_diag) {
            visited[root] = 1;
            continue;  // Singleton component already certified in pass 1
        }

        // BFS to collect connected component
        comp_nodes.clear();
        comp_nodes.push_back(root);
        visited[root] = 1;
        for (std::size_t head = 0; head < comp_nodes.size(); ++head) {
            const std::size_t u = comp_nodes[head];
            for (std::size_t k = Q.row_start(u); k < Q.row_end(u); ++k) {
                const std::size_t v = q_cols[k];
                if (v != u && std::abs(q_vals[k]) > 1e-15 && !visited[v]) {
                    visited[v] = 1;
                    comp_nodes.push_back(v);
                }
            }
        }

        const std::size_t s = comp_nodes.size();
        for (std::size_t local_idx = 0; local_idx < s; ++local_idx) {
            global_to_local[comp_nodes[local_idx]] = local_idx;
        }

        if (s <= 128) {
            // Small coupled component: exact diagonal-pivoted Cholesky on s x s submatrix
            std::vector<double> S(s * s, 0.0);
            std::vector<std::size_t> p(s);
            for (std::size_t local_r = 0; local_r < s; ++local_r) {
                p[local_r] = local_r;
                const std::size_t r = comp_nodes[local_r];
                for (std::size_t k = Q.row_start(r); k < Q.row_end(r); ++k) {
                    const std::size_t c = q_cols[k];
                    if (c == r || std::abs(q_vals[k]) > 1e-15) {
                        S[local_r * s + global_to_local[c]] = q_vals[k];
                    }
                }
            }

            const std::size_t comp_bytes =
                S.capacity() * sizeof(double) + p.capacity() * sizeof(std::size_t);
            max_comp_bytes = std::max(max_comp_bytes, comp_bytes);

            for (std::size_t k = 0; k < s; ++k) {
                double max_diag = -1.0;
                std::size_t best_idx = k;
                for (std::size_t i = k; i < s; ++i) {
                    const double diag = S[p[i] * s + p[i]];
                    if (diag > max_diag) {
                        max_diag = diag;
                        best_idx = i;
                    }
                }

                if (max_diag < -tol) {
                    if (error_msg) {
                        *error_msg = "Negative pivot in Cholesky factorization: " +
                                     std::to_string(max_diag) +
                                     " detected (Hessian is indefinite).";
                    }
                    if (peak_working_bytes) {
                        *peak_working_bytes = max_comp_bytes;
                    }
                    return false;
                }

                if (max_diag <= tol) {
                    for (std::size_t i = k; i < s; ++i) {
                        for (std::size_t j = k; j < s; ++j) {
                            const double val = std::abs(S[p[i] * s + p[j]]);
                            if (val > tol * 100.0 + 1e-7) {
                                if (error_msg) {
                                    *error_msg =
                                        "Non-zero off-diagonal entry in null Schur complement: " +
                                        std::to_string(val) + " (Hessian is indefinite).";
                                }
                                if (peak_working_bytes) {
                                    *peak_working_bytes = max_comp_bytes;
                                }
                                return false;
                            }
                        }
                    }
                    break;
                }

                std::swap(p[k], p[best_idx]);
                const std::size_t pk = p[k];
                const double pivot = S[pk * s + pk];

                for (std::size_t i = k + 1; i < s; ++i) {
                    const std::size_t pi = p[i];
                    const double factor = S[pi * s + pk] / pivot;
                    S[pi * s + pk] = factor;
                    if (std::abs(factor) > 1e-16) {
                        for (std::size_t j = k + 1; j < s; ++j) {
                            const std::size_t pj = p[j];
                            S[pi * s + pj] -= factor * S[pk * s + pj];
                        }
                    }
                }
            }
        } else {
            // Large coupled component (s > 128): Sparse Row-Map Diagonal-Pivoted Cholesky
            std::vector<double> diag(s, 0.0);
            std::vector<std::vector<std::pair<std::size_t, double>>> rows(s);
            std::vector<std::uint8_t> eliminated(s, 0);

            std::size_t sparse_rows_bytes =
                diag.capacity() * sizeof(double) + eliminated.capacity() * sizeof(std::uint8_t);

            for (std::size_t local_r = 0; local_r < s; ++local_r) {
                const std::size_t r = comp_nodes[local_r];
                for (std::size_t k = Q.row_start(r); k < Q.row_end(r); ++k) {
                    const std::size_t c = q_cols[k];
                    const double val = q_vals[k];
                    if (c == r) {
                        diag[local_r] = val;
                    } else if (std::abs(val) > 1e-15) {
                        rows[local_r].push_back({global_to_local[c], val});
                    }
                }
                std::sort(rows[local_r].begin(), rows[local_r].end());
                sparse_rows_bytes +=
                    rows[local_r].capacity() * sizeof(std::pair<std::size_t, double>);
            }
            max_comp_bytes = std::max(max_comp_bytes, sparse_rows_bytes);

            for (std::size_t step = 0; step < s; ++step) {
                double max_diag = -1.0;
                std::size_t pk = s;
                for (std::size_t i = 0; i < s; ++i) {
                    if (!eliminated[i] && diag[i] > max_diag) {
                        max_diag = diag[i];
                        pk = i;
                    }
                }

                if (pk == s || max_diag < -tol) {
                    if (error_msg) {
                        *error_msg = "Negative pivot in Cholesky factorization: " +
                                     std::to_string(max_diag) +
                                     " detected (Hessian is indefinite).";
                    }
                    if (peak_working_bytes) {
                        *peak_working_bytes = max_comp_bytes;
                    }
                    return false;
                }

                if (max_diag <= tol) {
                    for (std::size_t i = 0; i < s; ++i) {
                        if (eliminated[i]) {
                            continue;
                        }
                        for (const auto& [col_j, val] : rows[i]) {
                            if (!eliminated[col_j] && std::abs(val) > tol * 100.0 + 1e-7) {
                                if (error_msg) {
                                    *error_msg =
                                        "Non-zero off-diagonal entry in null Schur complement: " +
                                        std::to_string(std::abs(val)) + " (Hessian is indefinite).";
                                }
                                if (peak_working_bytes) {
                                    *peak_working_bytes = max_comp_bytes;
                                }
                                return false;
                            }
                        }
                    }
                    break;
                }

                eliminated[pk] = 1;
                const double pivot = diag[pk];

                // Extract active neighbors of pk in the remaining Schur complement
                std::vector<std::pair<std::size_t, double>> pk_neighbors;
                for (const auto& [col_j, val] : rows[pk]) {
                    if (!eliminated[col_j] && std::abs(val) > 1e-15) {
                        pk_neighbors.push_back({col_j, val});
                    }
                }

                for (const auto& [pi, val_ip] : pk_neighbors) {
                    const double factor = val_ip / pivot;
                    diag[pi] -= factor * val_ip;

                    // Sparse row merge: rows[pi] -= factor * pk_neighbors
                    std::vector<std::pair<std::size_t, double>> merged;
                    merged.reserve(rows[pi].size() + pk_neighbors.size());
                    std::size_t idx_a = 0;
                    std::size_t idx_b = 0;
                    while (idx_a < rows[pi].size() || idx_b < pk_neighbors.size()) {
                        if (idx_a < rows[pi].size() && eliminated[rows[pi][idx_a].first]) {
                            ++idx_a;
                            continue;
                        }
                        if (idx_b == pk_neighbors.size() ||
                            (idx_a < rows[pi].size() &&
                             rows[pi][idx_a].first < pk_neighbors[idx_b].first)) {
                            merged.push_back(rows[pi][idx_a++]);
                        } else if (idx_a == rows[pi].size() ||
                                   pk_neighbors[idx_b].first < rows[pi][idx_a].first) {
                            const std::size_t col_j = pk_neighbors[idx_b].first;
                            const double upd = -factor * pk_neighbors[idx_b].second;
                            if (col_j != pi && std::abs(upd) > 1e-15) {
                                merged.push_back({col_j, upd});
                            }
                            ++idx_b;
                        } else {
                            const std::size_t col_j = rows[pi][idx_a].first;
                            const double upd =
                                rows[pi][idx_a].second - factor * pk_neighbors[idx_b].second;
                            if (col_j != pi && std::abs(upd) > 1e-15) {
                                merged.push_back({col_j, upd});
                            }
                            ++idx_a;
                            ++idx_b;
                        }
                    }
                    rows[pi] = std::move(merged);
                }
                rows[pk].clear();
                rows[pk].shrink_to_fit();
            }
        }
    }

    if (peak_working_bytes) {
        *peak_working_bytes = max_comp_bytes;
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
    std::size_t psd_working_bytes = 0;
    if (!check_psd(Q, options.tolerances.pivot_tol, &psd_error, &psd_working_bytes)) {
        sol.status = core::SolverStatus::NUMERICAL_FAILURE;
        sol.telemetry.dense_equivalent_hessian_bytes =
            model.num_variables() * model.num_variables() * sizeof(double);
        sol.telemetry.sparse_hessian_bytes = Q.values().capacity() * sizeof(double) +
                                             Q.col_idx().capacity() * sizeof(std::size_t) +
                                             (Q.rows() + 1) * sizeof(std::size_t);
        sol.telemetry.sparse_psd_working_bytes = psd_working_bytes;
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
    auto res = solve_internal(model, options, start_time);
    res.telemetry.sparse_psd_working_bytes = psd_working_bytes;
    res.telemetry.peak_working_memory_bytes =
        std::max(res.telemetry.peak_working_memory_bytes,
                 res.telemetry.sparse_hessian_bytes + psd_working_bytes);
    return res;
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
    sol.telemetry.dense_equivalent_hessian_bytes = n * n * sizeof(double);
    sol.telemetry.sparse_hessian_bytes = Q.values().capacity() * sizeof(double) +
                                         Q.col_idx().capacity() * sizeof(std::size_t) +
                                         (Q.rows() + 1) * sizeof(std::size_t);

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
    auto dot_constr = [&](const NormalizedConstraint& constraint, const std::vector<double>& v) {
        double val = 0.0;
        for (const auto& [var_id, coeff] : constraint.terms) {
            val += coeff * v[var_id];
        }
        return val;
    };

    // Helper: Sparse Modified Gram-Schmidt LICQ check (zero O(n) / O(K*n) dense row vectors)
    struct SparseBasisVec {
        std::vector<std::pair<std::size_t, double>> terms;  // sorted by var_id
        [[nodiscard]] std::size_t byte_footprint() const noexcept {
            return terms.capacity() * sizeof(std::pair<std::size_t, double>);
        }
    };

    auto make_sparse_vec = [&](const NormalizedConstraint& constraint) {
        SparseBasisVec sv;
        sv.terms.reserve(constraint.terms.size());
        for (const auto& [var_id, coeff] : constraint.terms) {
            if (var_id < n && std::abs(coeff) > 1e-15) {
                sv.terms.push_back({var_id, coeff});
            }
        }
        std::sort(sv.terms.begin(), sv.terms.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
        // Combine duplicate variable indices if any
        std::size_t out = 0;
        for (std::size_t idx = 0; idx < sv.terms.size(); ++idx) {
            if (out > 0 && sv.terms[out - 1].first == sv.terms[idx].first) {
                sv.terms[out - 1].second += sv.terms[idx].second;
            } else {
                sv.terms[out++] = sv.terms[idx];
            }
        }
        sv.terms.resize(out);
        return sv;
    };

    auto sparse_dot = [](const SparseBasisVec& u, const SparseBasisVec& v) {
        double sum = 0.0;
        std::size_t i = 0;
        std::size_t j = 0;
        while (i < u.terms.size() && j < v.terms.size()) {
            if (u.terms[i].first < v.terms[j].first) {
                ++i;
            } else if (v.terms[j].first < u.terms[i].first) {
                ++j;
            } else {
                sum += u.terms[i].second * v.terms[j].second;
                ++i;
                ++j;
            }
        }
        return sum;
    };

    auto sparse_axpy = [](SparseBasisVec& v, double alpha, const SparseBasisVec& q) {
        std::vector<std::pair<std::size_t, double>> merged;
        merged.reserve(v.terms.size() + q.terms.size());
        std::size_t i = 0;
        std::size_t j = 0;
        while (i < v.terms.size() || j < q.terms.size()) {
            if (j == q.terms.size() ||
                (i < v.terms.size() && v.terms[i].first < q.terms[j].first)) {
                merged.push_back(v.terms[i++]);
            } else if (i == v.terms.size() || q.terms[j].first < v.terms[i].first) {
                const double val = alpha * q.terms[j].second;
                if (std::abs(val) > 1e-15) {
                    merged.push_back({q.terms[j].first, val});
                }
                ++j;
            } else {
                const double val = v.terms[i].second + alpha * q.terms[j].second;
                if (std::abs(val) > 1e-15) {
                    merged.push_back({v.terms[i].first, val});
                }
                ++i;
                ++j;
            }
        }
        v.terms = std::move(merged);
    };

    auto project_sparse = [&](const SparseBasisVec& a, const std::vector<SparseBasisVec>& b,
                              double& out_norm_a, double& out_norm_v) {
        out_norm_a = std::sqrt(sparse_dot(a, a));
        out_norm_v = 0.0;
        SparseBasisVec v = a;
        if (out_norm_a < 1e-12) {
            return v;
        }
        for (const auto& q : b) {
            const double dot = sparse_dot(q, v);
            if (std::abs(dot) > 1e-16) {
                sparse_axpy(v, -dot, q);
            }
        }
        out_norm_v = std::sqrt(sparse_dot(v, v));
        return v;
    };

    auto update_licq_bytes = [&](const std::vector<SparseBasisVec>& b) {
        std::size_t total = b.capacity() * sizeof(SparseBasisVec);
        for (const auto& q : b) {
            total += q.byte_footprint();
        }
        sol.telemetry.sparse_licq_basis_bytes =
            std::max(sol.telemetry.sparse_licq_basis_bytes, total);
    };

    auto try_add_to_basis = [&](const NormalizedConstraint& constr,
                                std::vector<SparseBasisVec>& b) {
        SparseBasisVec a = make_sparse_vec(constr);
        double norm_a = 0.0;
        double norm_v = 0.0;
        SparseBasisVec v = project_sparse(a, b, norm_a, norm_v);
        if (norm_a >= 1e-12 && norm_v > 1e-4 * norm_a && norm_v > 1e-12) {
            const double inv_norm = 1.0 / norm_v;
            for (auto& [var_id, val] : v.terms) {
                val *= inv_norm;
            }
            b.push_back(std::move(v));
            update_licq_bytes(b);
            return true;
        }
        return false;
    };

    auto rebuild_basis = [&](const std::vector<std::size_t>& current_W) {
        std::vector<SparseBasisVec> new_basis;
        new_basis.reserve(current_W.size());
        for (std::size_t idx : current_W) {
            (void)try_add_to_basis(all_constrs[idx], new_basis);
        }
        update_licq_bytes(new_basis);
        return new_basis;
    };

    // --- 4. Initialize Working Set W with LICQ Filter ---
    std::vector<std::size_t> W;
    W.reserve(n);
    std::vector<SparseBasisVec> basis;
    basis.reserve(std::min<std::size_t>(n, 64));

    // Always include all equality constraints (if linearly independent)
    for (std::size_t l = 0; l < num_total_constrs; ++l) {
        if (all_constrs[l].is_equality) {
            if (try_add_to_basis(all_constrs[l], basis)) {
                W.push_back(l);
            }
        }
    }

    // Add inequality constraints active at x0 (if linearly independent)
    for (std::size_t l = 0; l < num_total_constrs; ++l) {
        if (!all_constrs[l].is_equality) {
            const double ax = dot_constr(all_constrs[l], x);
            if (std::abs(ax - all_constrs[l].rhs) <= opts.tolerances.primal_feasibility_tol) {
                if (W.size() < n) {
                    if (try_add_to_basis(all_constrs[l], basis)) {
                        W.push_back(l);
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

        std::size_t active_terms_nnz = 0;
        for (std::size_t k = 0; k < K; ++k) {
            active_terms_nnz += all_constrs[W[k]].terms.size();
        }

        std::vector<numerics::Triplet> kkt_triplets;
        kkt_triplets.reserve(Q.nonzeros() + 2 * active_terms_nnz + n);

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

        const std::size_t kkt_bytes = kkt_triplets.capacity() * sizeof(numerics::Triplet) +
                                      KKT_matrix.values().capacity() * sizeof(double) +
                                      KKT_matrix.col_idx().capacity() * sizeof(std::size_t) +
                                      (KKT_matrix.rows() + 1) * sizeof(std::size_t);
        sol.telemetry.sparse_kkt_peak_bytes =
            std::max(sol.telemetry.sparse_kkt_peak_bytes, kkt_bytes);

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
                if (try_add_to_basis(all_constrs[static_cast<std::size_t>(blocking_l)], basis)) {
                    W.push_back(static_cast<std::size_t>(blocking_l));
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
    sol.telemetry.peak_working_memory_bytes = sol.telemetry.sparse_hessian_bytes +
                                              sol.telemetry.sparse_licq_basis_bytes +
                                              sol.telemetry.sparse_kkt_peak_bytes;

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
