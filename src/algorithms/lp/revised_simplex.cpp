#include "algorithms/lp/revised_simplex.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace kavach::algorithms::lp {

namespace {

double dot_sparse_col(const std::vector<double>& y, const numerics::CscMatrix& A_csc,
                      std::size_t col_idx) {
    const std::size_t nnz = A_csc.col_nnz(col_idx);
    const std::size_t* r_idx = A_csc.col_indices(col_idx);
    const double* vals = A_csc.col_values(col_idx);

    double sum = 0.0;
    for (std::size_t p = 0; p < nnz; ++p) {
        std::size_t r = r_idx[p];
        if (r < y.size()) {
            sum += y[r] * vals[p];
        }
    }
    return sum;
}

}  // namespace

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

    if (!basis.factorize(lp_.A_csc(), tol_.pivot_tol)) {
        telemetry.numerical_warnings.push_back("Initial basis factorization failed: " +
                                               basis.error_message());
        return {core::SolverStatus::NUMERICAL_FAILURE,
                {},
                {},
                {},
                0.0,
                "Numerical failure in initial basis factorization"};
    }
    ++telemetry.refactorization_count;

    // Phase I: Artificial variable feasibility problem
    if (lp_.has_artificials()) {
        auto p1_status = run_phase1(basis, telemetry, start_time);
        if (p1_status != core::SolverStatus::OPTIMAL) {
            return {p1_status, {}, {}, {}, 0.0, "Phase I terminated with non-optimal status"};
        }
    }

    // Phase II: Optimization of true objective
    auto p2_status = run_phase2(basis, telemetry, start_time);
    if (p2_status != core::SolverStatus::OPTIMAL) {
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

    std::vector<double> reduced_costs(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        if (!basis.is_basic(j)) {
            double yTa = dot_sparse_col(y, lp_.A_csc(), j);
            reduced_costs[j] = lp_.c()[j] - yTa;
        }
    }

    double obj = lp_.objective_offset();
    for (std::size_t j = 0; j < n; ++j) {
        obj += lp_.c()[j] * z[j];
    }

    telemetry.condition_estimate = basis.condition_estimate();
    return {core::SolverStatus::OPTIMAL, std::move(z), std::move(y),
            std::move(reduced_costs),    obj,          "Optimal"};
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
            double yTa = dot_sparse_col(y, lp_.A_csc(), j);
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
            double phase1_obj = 0.0;
            for (std::size_t i = 0; i < m; ++i) {
                phase1_obj += c_B[i] * x_B[i];
            }

            if (phase1_obj > tol_.primal_feasibility_tol) {
                return core::SolverStatus::INFEASIBLE;
            }

            // Phase I is feasible. Pivot out any basic artificial variables
            for (std::size_t r = 0; r < m; ++r) {
                std::size_t b_var = basis.basis_vars()[r];
                if (lp_.is_artificial(b_var)) {
                    std::vector<double> e_r(m, 0.0);
                    e_r[r] = 1.0;
                    auto v = basis.solve_dual(e_r);

                    for (std::size_t j : basis.nonbasic_vars()) {
                        if (!lp_.is_artificial(j)) {
                            double alpha = dot_sparse_col(v, lp_.A_csc(), j);
                            if (std::abs(alpha) > tol_.pivot_tol) {
                                basis.update_pivot(r, j);
                                if (!basis.factorize(lp_.A_csc(), tol_.pivot_tol)) {
                                    return core::SolverStatus::NUMERICAL_FAILURE;
                                }
                                ++telemetry.refactorization_count;
                                break;
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

        basis.update_pivot(leaving_row, entering_var);
        ++telemetry.pivot_count;
        ++telemetry.phase1_iterations;
        ++telemetry.total_iterations;

        if (!basis.factorize(lp_.A_csc(), tol_.pivot_tol)) {
            return core::SolverStatus::NUMERICAL_FAILURE;
        }
        ++telemetry.refactorization_count;
        x_B = basis.solve_primal(lp_.b());
    }

    return core::SolverStatus::UNKNOWN;
}

core::SolverStatus RevisedSimplexCore::run_phase2(
    SimplexBasis& basis, LPSolverTelemetry& telemetry,
    const std::chrono::steady_clock::time_point& start_time) {
    const std::size_t m = lp_.num_rows();

    auto x_B = basis.solve_primal(lp_.b());
    std::size_t consecutive_degenerate = 0;

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
            double yTa = dot_sparse_col(y, lp_.A_csc(), j);
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

        basis.update_pivot(leaving_row, entering_var);
        ++telemetry.pivot_count;
        ++telemetry.phase2_iterations;
        ++telemetry.total_iterations;

        if (!basis.factorize(lp_.A_csc(), tol_.pivot_tol)) {
            return core::SolverStatus::NUMERICAL_FAILURE;
        }
        ++telemetry.refactorization_count;
        x_B = basis.solve_primal(lp_.b());
    }
}

}  // namespace kavach::algorithms::lp
