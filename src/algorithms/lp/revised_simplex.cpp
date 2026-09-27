#include "algorithms/lp/revised_simplex.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace kavach::algorithms::lp {

RevisedSimplexCore::RevisedSimplexCore(const StandardFormLP& lp, const LPSolverOptions& options)
    : lp_(lp), options_(options), tol_(options.tolerances) {}

bool RevisedSimplexCore::check_time_limit(
    const std::chrono::steady_clock::time_point& start_time) const {
    if (options_.time_limit_seconds <= 0.0) {
        return false;
    }
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time).count();
    return elapsed >= options_.time_limit_seconds;
}

SimplexExecutionResult RevisedSimplexCore::run(LPSolverTelemetry& telemetry) {
    auto start_time = std::chrono::steady_clock::now();

    telemetry.standard_form_rows = lp_.num_rows();
    telemetry.standard_form_cols = lp_.num_cols();
    telemetry.standard_form_nnz = lp_.A_csc().nonzeros();

    const std::size_t m = lp_.num_rows();
    const std::size_t n = lp_.num_cols();

    // Edge case: no constraints
    if (m == 0) {
        std::vector<double> z(n, 0.0);
        std::vector<double> d = lp_.c();
        for (std::size_t j = 0; j < n; ++j) {
            if (d[j] < -tol_.optimality_tol) {
                return {core::SolverStatus::UNBOUNDED,
                        {},
                        {},
                        {},
                        0.0,
                        "Unbounded: negative cost with no constraints"};
            }
        }
        double obj = lp_.objective_offset();
        return {core::SolverStatus::OPTIMAL, z, {}, d, obj, "Optimal (unconstrained)"};
    }

    SimplexBasis basis;
    basis.initialize(lp_.initial_basis(), n);

    if (!basis.factorize(lp_.A_csc(), tol_.pivot_tol, "INITIAL_FACTORIZATION")) {
        basis.accumulate_telemetry(telemetry);
        telemetry.numerical_warnings.push_back("Initial basis factorization failed: " +
                                               basis.error_message());
        return {core::SolverStatus::NUMERICAL_FAILURE,
                {},
                {},
                {},
                0.0,
                "Numerical failure in initial basis factorization"};
    }

    // Phase I: Artificial variable feasibility problem
    if (lp_.has_artificials()) {
        auto p1_status = run_phase1(basis, telemetry, start_time);
        if (p1_status != core::SolverStatus::OPTIMAL) {
            basis.accumulate_telemetry(telemetry);
            return {p1_status, {}, {}, {}, 0.0, "Phase I terminated with non-optimal status"};
        }
    }

    // Phase II: Optimization of true objective
    auto p2_status = run_phase2(basis, telemetry, start_time);
    if (p2_status != core::SolverStatus::OPTIMAL) {
        basis.accumulate_telemetry(telemetry);
        return {p2_status, {}, {}, {}, 0.0, "Phase II terminated with non-optimal status"};
    }

    // Construct final solution
    auto x_B = basis.solve_primal(lp_.b());
    std::vector<double> z(n, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        std::size_t var = basis.basis_vars()[i];
        if (var < n) {
            z[var] = std::max(0.0, x_B[i]);
        }
    }

    std::vector<double> c_B(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        c_B[i] = lp_.c()[basis.basis_vars()[i]];
    }
    auto y = basis.solve_dual(c_B);

    std::vector<double> at_y(n, 0.0);
    if (options_.execution_backend != nullptr) {
        options_.execution_backend->batched_spmv_transpose(lp_.A_csr(), y.data(), at_y.data(), 1);
    } else {
        lp_.A_csr().spmv_transpose(y.data(), at_y.data());
    }

    std::vector<double> reduced_costs(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        if (!basis.is_basic(j)) {
            reduced_costs[j] = lp_.c()[j] - at_y[j];
        }
    }

    double obj = lp_.objective_offset();
    for (std::size_t j = 0; j < n; ++j) {
        obj += lp_.c()[j] * z[j];
    }

    (void)basis.compute_primal_residual(lp_.A_csc(), x_B, lp_.b());
    (void)basis.compute_dual_residual(lp_.A_csc(), y, c_B);
    basis.accumulate_telemetry(telemetry);
    telemetry.condition_estimate = basis.condition_estimate();
    return {core::SolverStatus::OPTIMAL, std::move(z), std::move(y),
            std::move(reduced_costs),    obj,          "Optimal",
            basis.export_state()};
}

core::SolverStatus RevisedSimplexCore::run_phase1(
    SimplexBasis& basis, LPSolverTelemetry& telemetry,
    const std::chrono::steady_clock::time_point& start_time) {
    const std::size_t m = lp_.num_rows();
    const std::size_t n = lp_.num_cols();

    std::vector<double> c_phase1(n, 0.0);
    for (std::size_t w : lp_.artificial_indices()) {
        if (w < n) {
            c_phase1[w] = 1.0;
        }
    }

    auto x_B = basis.solve_primal(lp_.b());
    std::size_t consecutive_degenerate = 0;
    std::vector<double> at_y(n, 0.0);

    while (true) {
        if (telemetry.total_iterations >= options_.max_iterations) {
            return core::SolverStatus::LIMIT_REACHED;
        }
        if (check_time_limit(start_time)) {
            return core::SolverStatus::LIMIT_REACHED;
        }

        std::vector<double> c_B(m, 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            c_B[i] = c_phase1[basis.basis_vars()[i]];
        }
        auto y = basis.solve_dual(c_B);

        if (options_.execution_backend != nullptr) {
            options_.execution_backend->batched_spmv_transpose(lp_.A_csr(), y.data(), at_y.data(),
                                                               1);
        } else {
            lp_.A_csr().spmv_transpose(y.data(), at_y.data());
        }

        // Find improving entering variable
        std::size_t entering_var = std::numeric_limits<std::size_t>::max();
        double best_rc = 0.0;

        bool use_bland = options_.force_bland_rule ||
                         consecutive_degenerate >= options_.bland_watchdog_threshold;
        if (use_bland) {
            telemetry.bland_fallback_activated = true;
            ++telemetry.bland_fallback_iterations;
        }

        for (std::size_t j : basis.nonbasic_vars()) {
            double yTa = at_y[j];
            double rc = c_phase1[j] - yTa;

            if (rc < -tol_.optimality_tol) {
                if (use_bland) {
                    if (entering_var == std::numeric_limits<std::size_t>::max() ||
                        j < entering_var) {
                        entering_var = j;
                        best_rc = rc;
                    }
                } else {
                    if (rc < best_rc) {
                        best_rc = rc;
                        entering_var = j;
                    }
                }
            }
        }

        // Phase I optimality reached?
        if (entering_var == std::numeric_limits<std::size_t>::max()) {
            if (basis.update_chain_length() > 0) {
                const double p_res = basis.compute_primal_residual(lp_.A_csc(), x_B, lp_.b());
                const double d_res = basis.compute_dual_residual(lp_.A_csc(), y, c_B);
                if (p_res > 1e-11 || d_res > 1e-11) {
                    if (!basis.factorize(lp_.A_csc(), tol_.pivot_tol,
                                         "PHASE1_OPTIMALITY_VERIFICATION")) {
                        return core::SolverStatus::NUMERICAL_FAILURE;
                    }
                    x_B = basis.solve_primal(lp_.b());
                    continue;
                }
            }

            double phase1_obj = 0.0;
            for (std::size_t i = 0; i < m; ++i) {
                phase1_obj += c_B[i] * x_B[i];
            }

            if (phase1_obj > tol_.primal_feasibility_tol) {
                return core::SolverStatus::INFEASIBLE;
            }

            // Phase I is feasible. If multiple artificial variables remain in the basis,
            // pre-evaluate their dual row vectors V = [v_0, ..., v_{K-1}] in a single batched
            // transpose SpMV call before sequential pivot updates.
            std::vector<std::size_t> art_rows;
            for (std::size_t r = 0; r < m; ++r) {
                if (lp_.is_artificial(basis.basis_vars()[r])) {
                    art_rows.push_back(r);
                }
            }
            if (!art_rows.empty() && options_.execution_backend != nullptr) {
                const std::size_t K = art_rows.size();
                std::vector<double> batch_v(K * m, 0.0);
                for (std::size_t k = 0; k < K; ++k) {
                    std::vector<double> e_r(m, 0.0);
                    e_r[art_rows[k]] = 1.0;
                    auto vk = basis.solve_dual(e_r);
                    std::copy(vk.begin(), vk.end(),
                              batch_v.begin() + static_cast<std::ptrdiff_t>(k * m));
                }
                std::vector<double> batch_w(K * n, 0.0);
                options_.execution_backend->batched_spmv_transpose(lp_.A_csr(), batch_v.data(),
                                                                   batch_w.data(), K);
            }

            // Pivot out any basic artificial variables
            for (std::size_t r = 0; r < m; ++r) {
                std::size_t b_var = basis.basis_vars()[r];
                if (lp_.is_artificial(b_var)) {
                    std::vector<double> e_r(m, 0.0);
                    e_r[r] = 1.0;
                    auto v = basis.solve_dual(e_r);

                    if (options_.execution_backend != nullptr) {
                        options_.execution_backend->batched_spmv_transpose(lp_.A_csr(), v.data(),
                                                                           at_y.data(), 1);
                    } else {
                        lp_.A_csr().spmv_transpose(v.data(), at_y.data());
                    }

                    std::size_t best_j = std::numeric_limits<std::size_t>::max();
                    double max_alpha = tol_.pivot_tol;
                    for (std::size_t j : basis.nonbasic_vars()) {
                        if (!lp_.is_artificial(j)) {
                            double alpha = std::abs(at_y[j]);
                            if (alpha > max_alpha) {
                                max_alpha = alpha;
                                best_j = j;
                            }
                        }
                    }
                    if (best_j != std::numeric_limits<std::size_t>::max()) {
                        auto d_art = basis.solve_direction(lp_.A_csc(), best_j);
                        if (!basis.apply_eta_update(r, best_j, d_art, tol_.pivot_tol) ||
                            basis.needs_refactorization(options_.refactorization_interval)) {
                            if (!basis.factorize(lp_.A_csc(), tol_.pivot_tol,
                                                 "ARTIFICIAL_BASIS_PURGE")) {
                                basis.update_pivot(r, b_var);
                                if (!basis.factorize(lp_.A_csc(), tol_.pivot_tol,
                                                     "ARTIFICIAL_PURGE_REVERT")) {
                                    return core::SolverStatus::NUMERICAL_FAILURE;
                                }
                            }
                        }
                    }
                }
            }

            // Purge artificial variables from nonbasic list
            for (std::size_t w : lp_.artificial_indices()) {
                basis.purge_nonbasic_variable(w);
            }

            return core::SolverStatus::OPTIMAL;
        }

        // Direction vector: solve B * d = A[:, entering_var]
        auto d = basis.solve_direction(lp_.A_csc(), entering_var);

        // Minimum ratio test
        double min_ratio = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < m; ++i) {
            if (d[i] > tol_.pivot_tol) {
                double ratio = std::max(0.0, x_B[i]) / d[i];
                if (ratio < min_ratio) {
                    min_ratio = ratio;
                }
            }
        }

        if (std::isinf(min_ratio)) {
            // Direction nonpositive
            break;
        }

        // Select leaving variable using Bland's tie breaking
        std::size_t leaving_row = std::numeric_limits<std::size_t>::max();
        std::size_t min_leaving_var = std::numeric_limits<std::size_t>::max();

        for (std::size_t i = 0; i < m; ++i) {
            if (d[i] > tol_.pivot_tol) {
                double ratio = std::max(0.0, x_B[i]) / d[i];
                if (std::abs(ratio - min_ratio) <= tol_.primal_feasibility_tol) {
                    std::size_t cand_var = basis.basis_vars()[i];
                    if (cand_var < min_leaving_var) {
                        min_leaving_var = cand_var;
                        leaving_row = i;
                    }
                }
            }
        }

        if (leaving_row == std::numeric_limits<std::size_t>::max()) {
            break;
        }

        if (min_ratio <= tol_.primal_feasibility_tol) {
            ++consecutive_degenerate;
        } else {
            consecutive_degenerate = 0;
        }

        const std::size_t prev_leaving_var = basis.basis_vars()[leaving_row];
        const bool eta_ok = basis.apply_eta_update(leaving_row, entering_var, d, tol_.pivot_tol);
        ++telemetry.pivot_count;
        ++telemetry.phase1_iterations;
        ++telemetry.total_iterations;

        bool need_refactor =
            !eta_ok || basis.needs_refactorization(options_.refactorization_interval);
        std::string refactor_reason =
            need_refactor ? basis.pending_refactorization_reason(options_.refactorization_interval)
                          : "";

        if (!need_refactor) {
            x_B = basis.solve_primal(lp_.b());
            const double p_res = basis.compute_primal_residual(lp_.A_csc(), x_B, lp_.b());
            if (p_res > options_.basis_drift_tol) {
                need_refactor = true;
                refactor_reason = "PRIMAL_RESIDUAL_DRIFT";
            }
        }

        if (need_refactor) {
            if (!basis.factorize(lp_.A_csc(), tol_.pivot_tol, refactor_reason)) {
                // Revert singular pivot caused by near-zero LU roundoff in d[leaving_row]
                // and select the tied candidate row in the minimum-ratio band with largest d[i].
                basis.update_pivot(leaving_row, prev_leaving_var);
                std::size_t best_row = std::numeric_limits<std::size_t>::max();
                double max_d = tol_.pivot_tol;
                for (std::size_t i = 0; i < m; ++i) {
                    if (i != leaving_row && d[i] > max_d) {
                        double ratio = std::max(0.0, x_B[i]) / d[i];
                        if (std::abs(ratio - min_ratio) <= tol_.primal_feasibility_tol) {
                            max_d = d[i];
                            best_row = i;
                        }
                    }
                }
                if (best_row == std::numeric_limits<std::size_t>::max()) {
                    return core::SolverStatus::NUMERICAL_FAILURE;
                }
                basis.update_pivot(best_row, entering_var);
                if (!basis.factorize(lp_.A_csc(), tol_.pivot_tol, "SINGULAR_PIVOT_RECOVERY")) {
                    return core::SolverStatus::NUMERICAL_FAILURE;
                }
            }
            x_B = basis.solve_primal(lp_.b());
            (void)basis.compute_primal_residual(lp_.A_csc(), x_B, lp_.b());
        }
    }

    return core::SolverStatus::UNKNOWN;
}

core::SolverStatus RevisedSimplexCore::run_phase2(
    SimplexBasis& basis, LPSolverTelemetry& telemetry,
    const std::chrono::steady_clock::time_point& start_time) {
    const std::size_t m = lp_.num_rows();
    const std::size_t n = lp_.num_cols();

    auto x_B = basis.solve_primal(lp_.b());
    std::size_t consecutive_degenerate = 0;
    std::vector<double> at_y(n, 0.0);

    while (true) {
        if (telemetry.total_iterations >= options_.max_iterations) {
            return core::SolverStatus::LIMIT_REACHED;
        }
        if (check_time_limit(start_time)) {
            return core::SolverStatus::LIMIT_REACHED;
        }

        std::vector<double> c_B(m, 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            c_B[i] = lp_.c()[basis.basis_vars()[i]];
        }
        auto y = basis.solve_dual(c_B);

        if (options_.execution_backend != nullptr) {
            options_.execution_backend->batched_spmv_transpose(lp_.A_csr(), y.data(), at_y.data(),
                                                               1);
        } else {
            lp_.A_csr().spmv_transpose(y.data(), at_y.data());
        }

        // Find improving entering variable
        std::size_t entering_var = std::numeric_limits<std::size_t>::max();
        double best_rc = 0.0;

        bool use_bland = options_.force_bland_rule ||
                         consecutive_degenerate >= options_.bland_watchdog_threshold;
        if (use_bland) {
            telemetry.bland_fallback_activated = true;
            ++telemetry.bland_fallback_iterations;
        }

        for (std::size_t j : basis.nonbasic_vars()) {
            double yTa = at_y[j];
            double rc = lp_.c()[j] - yTa;

            if (rc < -tol_.optimality_tol) {
                if (use_bland) {
                    if (entering_var == std::numeric_limits<std::size_t>::max() ||
                        j < entering_var) {
                        entering_var = j;
                        best_rc = rc;
                    }
                } else {
                    if (rc < best_rc) {
                        best_rc = rc;
                        entering_var = j;
                    }
                }
            }
        }

        // Optimality criterion satisfied: all reduced costs >= -optimality_tol
        if (entering_var == std::numeric_limits<std::size_t>::max()) {
            if (basis.update_chain_length() > 0) {
                const double p_res = basis.compute_primal_residual(lp_.A_csc(), x_B, lp_.b());
                const double d_res = basis.compute_dual_residual(lp_.A_csc(), y, c_B);
                std::cout << "[PHASE2 OPT CHECK] p_res=" << p_res << ", d_res=" << d_res
                          << ", chain=" << basis.update_chain_length() << std::endl;
                if (p_res > 1e-11 || d_res > 1e-11) {
                    if (!basis.factorize(lp_.A_csc(), tol_.pivot_tol,
                                         "PHASE2_OPTIMALITY_VERIFICATION")) {
                        return core::SolverStatus::NUMERICAL_FAILURE;
                    }
                    x_B = basis.solve_primal(lp_.b());
                    continue;
                }
            }
            return core::SolverStatus::OPTIMAL;
        }

        // Direction vector: solve B * d = A[:, entering_var]
        auto d = basis.solve_direction(lp_.A_csc(), entering_var);

        // Minimum ratio test
        double min_ratio = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < m; ++i) {
            if (d[i] > tol_.pivot_tol) {
                double ratio = std::max(0.0, x_B[i]) / d[i];
                if (ratio < min_ratio) {
                    min_ratio = ratio;
                }
            }
        }

        // Unboundedness test: entering column has no positive components in direction
        if (std::isinf(min_ratio)) {
            return core::SolverStatus::UNBOUNDED;
        }

        // Select leaving variable using Bland's tie breaking
        std::size_t leaving_row = std::numeric_limits<std::size_t>::max();
        std::size_t min_leaving_var = std::numeric_limits<std::size_t>::max();

        for (std::size_t i = 0; i < m; ++i) {
            if (d[i] > tol_.pivot_tol) {
                double ratio = std::max(0.0, x_B[i]) / d[i];
                if (std::abs(ratio - min_ratio) <= tol_.primal_feasibility_tol) {
                    std::size_t cand_var = basis.basis_vars()[i];
                    if (cand_var < min_leaving_var) {
                        min_leaving_var = cand_var;
                        leaving_row = i;
                    }
                }
            }
        }

        if (leaving_row == std::numeric_limits<std::size_t>::max()) {
            return core::SolverStatus::UNBOUNDED;
        }

        if (min_ratio <= tol_.primal_feasibility_tol) {
            ++consecutive_degenerate;
        } else {
            consecutive_degenerate = 0;
        }

        const std::size_t prev_leaving_var = basis.basis_vars()[leaving_row];
        const bool eta_ok = basis.apply_eta_update(leaving_row, entering_var, d, tol_.pivot_tol);
        ++telemetry.pivot_count;
        ++telemetry.phase2_iterations;
        ++telemetry.total_iterations;

        bool need_refactor =
            !eta_ok || basis.needs_refactorization(options_.refactorization_interval);
        std::string refactor_reason =
            need_refactor ? basis.pending_refactorization_reason(options_.refactorization_interval)
                          : "";

        if (!need_refactor) {
            x_B = basis.solve_primal(lp_.b());
            const double p_res = basis.compute_primal_residual(lp_.A_csc(), x_B, lp_.b());
            if (p_res > options_.basis_drift_tol) {
                need_refactor = true;
                refactor_reason = "PRIMAL_RESIDUAL_DRIFT";
            }
        }

        if (need_refactor) {
            if (!basis.factorize(lp_.A_csc(), tol_.pivot_tol, refactor_reason)) {
                basis.update_pivot(leaving_row, prev_leaving_var);
                std::size_t best_row = std::numeric_limits<std::size_t>::max();
                double max_d = tol_.pivot_tol;
                for (std::size_t i = 0; i < m; ++i) {
                    if (i != leaving_row && d[i] > max_d) {
                        double ratio = std::max(0.0, x_B[i]) / d[i];
                        if (std::abs(ratio - min_ratio) <= tol_.primal_feasibility_tol) {
                            max_d = d[i];
                            best_row = i;
                        }
                    }
                }
                if (best_row == std::numeric_limits<std::size_t>::max()) {
                    return core::SolverStatus::NUMERICAL_FAILURE;
                }
                basis.update_pivot(best_row, entering_var);
                if (!basis.factorize(lp_.A_csc(), tol_.pivot_tol, "SINGULAR_PIVOT_RECOVERY")) {
                    return core::SolverStatus::NUMERICAL_FAILURE;
                }
            }
            x_B = basis.solve_primal(lp_.b());
            (void)basis.compute_primal_residual(lp_.A_csc(), x_B, lp_.b());
        }
    }
}

}  // namespace kavach::algorithms::lp
