/**
 * @file dual_simplex.cpp
 * @brief Production Dual Simplex Engine for warm starting LP child-node re-solves.
 */

#include "algorithms/lp/dual_simplex.hpp"

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

DualSimplexCore::DualSimplexCore(const StandardFormLP& lp, const LPSolverOptions& options)
    : lp_(lp), options_(options), tol_(options.tolerances) {}

bool DualSimplexCore::check_time_limit(
    const std::chrono::steady_clock::time_point& start_time) const {
    if (options_.time_limit_seconds <= 0.0) {
        return false;
    }
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time).count();
    return elapsed >= options_.time_limit_seconds;
}

SimplexExecutionResult DualSimplexCore::run_warm_start(const BasisState& initial_basis,
                                                       LPSolverTelemetry& telemetry,
                                                       std::optional<BasisState>& resulting_basis) {
    auto start_time = std::chrono::steady_clock::now();

    const std::size_t m = lp_.num_rows();
    const std::size_t n = lp_.num_cols();

    // 1. Verify structural compatibility
    if (initial_basis.num_rows != m || initial_basis.num_cols != n) {
        return {core::SolverStatus::NUMERICAL_FAILURE,
                {},
                {},
                {},
                0.0,
                "Structural dimension mismatch: LP has (" + std::to_string(m) + "x" +
                    std::to_string(n) + ") but basis has (" +
                    std::to_string(initial_basis.num_rows) + "x" +
                    std::to_string(initial_basis.num_cols) + ")"};
    }

    if (initial_basis.basis_vars.size() != m) {
        return {core::SolverStatus::NUMERICAL_FAILURE,
                {},
                {},
                {},
                0.0,
                "Invalid basis size: expected " + std::to_string(m) + " basic variables but got " +
                    std::to_string(initial_basis.basis_vars.size())};
    }

    // 2. Initialize and factorize basis
    SimplexBasis basis;
    basis.load_state(initial_basis);

    if (!basis.factorize(lp_.A_csc(), tol_.pivot_tol)) {
        return {core::SolverStatus::NUMERICAL_FAILURE,
                {},
                {},
                {},
                0.0,
                "Inherited basis factorization failed: " + basis.error_message()};
    }
    ++telemetry.refactorization_count;

    // 3. Compute initial simplex multipliers y = B^{-T} c_B
    std::vector<double> c_B(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        std::size_t v = basis.basis_vars()[i];
        if (v >= n) {
            return {core::SolverStatus::NUMERICAL_FAILURE,
                    {},
                    {},
                    {},
                    0.0,
                    "Basic variable index out of bounds: " + std::to_string(v)};
        }
        c_B[i] = lp_.c()[v];
    }
    auto y = basis.solve_dual(c_B);

    // 4. Verify dual feasibility of inherited basis
    for (std::size_t j : basis.nonbasic_vars()) {
        if (j < n) {
            double yTa = dot_sparse_col(y, lp_.A_csc(), j);
            double rc = lp_.c()[j] - yTa;
            if (rc < -tol_.dual_feasibility_tol) {
                return {core::SolverStatus::NUMERICAL_FAILURE,
                        {},
                        {},
                        {},
                        0.0,
                        "Inherited basis violates dual feasibility: reduced cost " +
                            std::to_string(rc) + " at col " + std::to_string(j)};
            }
        }
    }

    // 5. Compute initial basic variable values x_B = B^{-1} b
    auto x_B = basis.solve_primal(lp_.b());

    // 6. Dual Simplex iteration loop
    std::size_t consecutive_degenerate = 0;
    const std::size_t max_dual_iters =
        std::min(options_.max_iterations, std::max(std::size_t(1000), 10 * m + 100));

    while (true) {
        if (telemetry.total_iterations >= options_.max_iterations ||
            telemetry.dual_simplex_iterations >= max_dual_iters) {
            return {core::SolverStatus::LIMIT_REACHED,     {}, {}, {}, 0.0,
                    "Dual simplex iteration limit reached"};
        }
        if (check_time_limit(start_time)) {
            return {core::SolverStatus::LIMIT_REACHED, {}, {}, {}, 0.0,
                    "Dual simplex time limit reached"};
        }

        // --- Step 1: Identify leaving row (most negative basic variable) ---
        std::size_t leaving_row = std::numeric_limits<std::size_t>::max();
        double max_violation = tol_.primal_feasibility_tol;

        for (std::size_t i = 0; i < m; ++i) {
            double viol = -x_B[i];
            if (viol > max_violation) {
                max_violation = viol;
                leaving_row = i;
            } else if (leaving_row != std::numeric_limits<std::size_t>::max() &&
                       std::abs(viol - max_violation) <= tol_.primal_feasibility_tol) {
                // Deterministic tie-breaking on basic variable column index
                if (basis.basis_vars()[i] < basis.basis_vars()[leaving_row]) {
                    leaving_row = i;
                }
            }
        }

        // --- Step 2: Check primal feasibility termination ---
        if (leaving_row == std::numeric_limits<std::size_t>::max()) {
            // All basic variables are non-negative within primal feasibility tolerance!
            // Primal feasibility restored + dual feasibility maintained => OPTIMAL
            std::vector<double> z(n, 0.0);
            for (std::size_t i = 0; i < m; ++i) {
                std::size_t v = basis.basis_vars()[i];
                if (v < n) {
                    z[v] = std::max(0.0, x_B[i]);
                }
            }

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

            resulting_basis = basis.export_state();
            telemetry.condition_estimate = basis.condition_estimate();

            return {core::SolverStatus::OPTIMAL, std::move(z), std::move(y),
                    std::move(reduced_costs),    obj,          "Optimal (dual simplex warm start)"};
        }

        // --- Step 3: Compute tableau row l: B^T v = e_l ---
        std::vector<double> e_l(m, 0.0);
        e_l[leaving_row] = 1.0;
        auto v = basis.solve_dual(e_l);

        // --- Step 4: Dual Ratio Test to find entering variable ---
        std::size_t entering_col = std::numeric_limits<std::size_t>::max();
        double min_ratio = std::numeric_limits<double>::infinity();

        for (std::size_t j : basis.nonbasic_vars()) {
            double alpha = dot_sparse_col(v, lp_.A_csc(), j);

            // In dual simplex, for x_B[leaving_row] < 0, entering column must have alpha < 0
            if (alpha < -tol_.pivot_tol) {
                double yTa = dot_sparse_col(y, lp_.A_csc(), j);
                double rc = lp_.c()[j] - yTa;
                double ratio = std::max(0.0, rc) / (-alpha);

                if (ratio < min_ratio - 1e-12) {
                    min_ratio = ratio;
                    entering_col = j;
                } else if (std::abs(ratio - min_ratio) <= 1e-12) {
                    // Deterministic tie-breaking on smallest variable index
                    if (entering_col == std::numeric_limits<std::size_t>::max() ||
                        j < entering_col) {
                        entering_col = j;
                    }
                }
            }
        }

        // --- Step 5: Check primal infeasibility (dual unboundedness) ---
        if (entering_col == std::numeric_limits<std::size_t>::max()) {
            // Row has no negative entries: dual ray proven, problem is primal infeasible!
            return {core::SolverStatus::INFEASIBLE,
                    {},
                    {},
                    {},
                    0.0,
                    "Dual simplex detected primal infeasibility (dual ray in leaving row " +
                        std::to_string(leaving_row) + ")"};
        }

        // Track degeneracy for watchdog
        if (min_ratio <= tol_.dual_feasibility_tol) {
            ++consecutive_degenerate;
        } else {
            consecutive_degenerate = 0;
        }

        if (consecutive_degenerate >= options_.bland_watchdog_threshold) {
            telemetry.bland_fallback_activated = true;
            ++telemetry.bland_fallback_iterations;
        }

        // --- Step 6: Pivot ---
        basis.update_pivot(leaving_row, entering_col);
        ++telemetry.pivot_count;
        ++telemetry.dual_simplex_iterations;
        ++telemetry.total_iterations;

        // Periodic or required refactorization
        if (!basis.factorize(lp_.A_csc(), tol_.pivot_tol)) {
            return {core::SolverStatus::NUMERICAL_FAILURE,
                    {},
                    {},
                    {},
                    0.0,
                    "Basis factorization failed during dual pivot: " + basis.error_message()};
        }
        ++telemetry.refactorization_count;

        // Recompute primal and dual quantities
        x_B = basis.solve_primal(lp_.b());
        for (std::size_t i = 0; i < m; ++i) {
            c_B[i] = lp_.c()[basis.basis_vars()[i]];
        }
        y = basis.solve_dual(c_B);
    }
}

}  // namespace kavach::algorithms::lp
