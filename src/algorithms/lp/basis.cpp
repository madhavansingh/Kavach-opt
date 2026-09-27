#include "algorithms/lp/basis.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kavach::algorithms::lp {

void SimplexBasis::initialize(std::vector<std::size_t> initial_basis, std::size_t total_cols) {
    basis_vars_ = std::move(initial_basis);
    is_basic_.assign(total_cols, false);
    for (std::size_t var : basis_vars_) {
        if (var < total_cols) {
            is_basic_[var] = true;
        }
    }

    nonbasic_vars_.clear();
    if (total_cols >= basis_vars_.size()) {
        nonbasic_vars_.reserve(total_cols - basis_vars_.size());
    }
    for (std::size_t j = 0; j < total_cols; ++j) {
        if (!is_basic_[j]) {
            nonbasic_vars_.push_back(j);
        }
    }

    eta_chain_.clear();
    updates_since_refactor_ = 0;
    factorization_valid_ = false;
    update_chain_bytes_ = 0;
}

bool SimplexBasis::factorize(const numerics::CscMatrix& A_csc, double pivot_tol,
                             const std::string& reason) {
    const auto t_start = std::chrono::steady_clock::now();

    eta_chain_.clear();
    update_chain_bytes_ = 0;
    updates_since_refactor_ = 0;
    ++factorization_generation_;
    ++full_refactorizations_;
    last_refactor_reason_ = reason;

    const std::size_t m = basis_vars_.size();
    if (m == 0) {
        factorization_valid_ = true;
        base_factorization_bytes_ = 0;
        memory_after_refactor_bytes_ = 0;
        return true;
    }

    // 1. Count nonzeros per row in B
    std::vector<std::size_t> row_counts(m, 0);
    for (std::size_t k = 0; k < m; ++k) {
        std::size_t c = basis_vars_[k];
        std::size_t c_nnz = A_csc.col_nnz(c);
        const std::size_t* r_idx = A_csc.col_indices(c);
        for (std::size_t p = 0; p < c_nnz; ++p) {
            if (r_idx[p] < m) {
                ++row_counts[r_idx[p]];
            }
        }
    }

    // 2. Prefix sums into row_ptr
    std::vector<std::size_t> row_ptr(m + 1, 0);
    for (std::size_t i = 0; i < m; ++i) {
        row_ptr[i + 1] = row_ptr[i] + row_counts[i];
    }

    std::size_t total_nnz = row_ptr[m];
    std::vector<std::size_t> col_idx(total_nnz, 0);
    std::vector<double> values(total_nnz, 0.0);
    std::vector<std::size_t> current_pos = row_ptr;

    // 3. Scatter entries. Since k increases 0..m-1, columns in each row are naturally sorted!
    for (std::size_t k = 0; k < m; ++k) {
        std::size_t c = basis_vars_[k];
        std::size_t c_nnz = A_csc.col_nnz(c);
        const std::size_t* r_idx = A_csc.col_indices(c);
        const double* vals = A_csc.col_values(c);
        for (std::size_t p = 0; p < c_nnz; ++p) {
            std::size_t r = r_idx[p];
            if (r < m) {
                std::size_t dest = current_pos[r]++;
                col_idx[dest] = k;
                values[dest] = vals[p];
            }
        }
    }

    numerics::CsrMatrix B(m, m, std::move(row_ptr), std::move(col_idx), std::move(values), false);
    bool ok = lu_.factorize(B, pivot_tol);

    const auto t_end = std::chrono::steady_clock::now();
    full_refactorization_time_seconds_ += std::chrono::duration<double>(t_end - t_start).count();

    factorization_valid_ = ok;
    if (ok) {
        base_factorization_bytes_ = lu_.L().nonzeros() * (sizeof(std::size_t) + sizeof(double)) +
                                    (lu_.L().rows() + 1) * sizeof(std::size_t) +
                                    lu_.U().nonzeros() * (sizeof(std::size_t) + sizeof(double)) +
                                    (lu_.U().rows() + 1) * sizeof(std::size_t) +
                                    2 * lu_.dimension() * sizeof(std::size_t);
        memory_after_refactor_bytes_ = base_factorization_bytes_;
        peak_memory_bytes_ = std::max(peak_memory_bytes_, base_factorization_bytes_);
    }
    return ok;
}

std::vector<double> SimplexBasis::solve_primal(const std::vector<double>& rhs) const {
    const std::size_t m = basis_vars_.size();
    if (m == 0) {
        return {};
    }
    if (!factorization_valid_) {
        throw std::logic_error(
            "SimplexBasis::solve_primal invoked on invalidated basis factorization");
    }

    std::vector<double> u = lu_.solve(rhs);
    if (u.size() != m) {
        return u;
    }

    for (const auto& eta : eta_chain_) {
        if (eta.base_generation != factorization_generation_) {
            throw std::logic_error(
                "SimplexBasis::solve_primal detected stale Eta update generation");
        }
        const double up = u[eta.pivot_row] / eta.pivot_elem;
        u[eta.pivot_row] = up;
        const std::size_t nnz = eta.off_diag_rows.size();
        for (std::size_t idx = 0; idx < nnz; ++idx) {
            u[eta.off_diag_rows[idx]] -= eta.off_diag_vals[idx] * up;
        }
    }
    return u;
}

std::vector<double> SimplexBasis::solve_dual(const std::vector<double>& c_B) const {
    const std::size_t m = basis_vars_.size();
    if (m == 0) {
        return {};
    }
    if (!factorization_valid_) {
        throw std::logic_error(
            "SimplexBasis::solve_dual invoked on invalidated basis factorization");
    }

    std::vector<double> w = c_B;
    if (w.size() != m) {
        return {};
    }

    for (std::size_t rev = eta_chain_.size(); rev > 0; --rev) {
        const auto& eta = eta_chain_[rev - 1];
        if (eta.base_generation != factorization_generation_) {
            throw std::logic_error("SimplexBasis::solve_dual detected stale Eta update generation");
        }
        double dot = 0.0;
        const std::size_t nnz = eta.off_diag_rows.size();
        for (std::size_t idx = 0; idx < nnz; ++idx) {
            dot += eta.off_diag_vals[idx] * w[eta.off_diag_rows[idx]];
        }
        w[eta.pivot_row] = (w[eta.pivot_row] - dot) / eta.pivot_elem;
    }

    return lu_.solve_transpose(w);
}

std::vector<double> SimplexBasis::solve_direction(const numerics::CscMatrix& A_csc,
                                                  std::size_t entering_col) const {
    const std::size_t m = basis_vars_.size();
    std::vector<double> a(m, 0.0);
    std::size_t c_nnz = A_csc.col_nnz(entering_col);
    const std::size_t* r_idx = A_csc.col_indices(entering_col);
    const double* vals = A_csc.col_values(entering_col);
    for (std::size_t p = 0; p < c_nnz; ++p) {
        if (r_idx[p] < m) {
            a[r_idx[p]] = vals[p];
        }
    }
    return solve_primal(a);
}

void SimplexBasis::update_pivot_combinatorial(std::size_t leaving_row, std::size_t entering_col) {
    std::size_t leaving_var = basis_vars_[leaving_row];
    basis_vars_[leaving_row] = entering_col;
    is_basic_[leaving_var] = false;
    is_basic_[entering_col] = true;

    for (std::size_t& nb : nonbasic_vars_) {
        if (nb == entering_col) {
            nb = leaving_var;
            break;
        }
    }

    ++updates_since_refactor_;
}

void SimplexBasis::update_pivot(std::size_t leaving_row, std::size_t entering_col) {
    update_pivot_combinatorial(leaving_row, entering_col);
    factorization_valid_ = false;
    last_refactor_reason_ = "UNTRACKED_COMBINATORIAL_PIVOT";
}

bool SimplexBasis::apply_eta_update(std::size_t leaving_row, std::size_t entering_col,
                                    const std::vector<double>& d, double pivot_tol,
                                    double min_rel_pivot) {
    const auto t_start = std::chrono::steady_clock::now();
    const std::size_t m = basis_vars_.size();

    if (!factorization_valid_ || !lu_.is_factorized() || leaving_row >= m || d.size() != m) {
        update_pivot_combinatorial(leaving_row, entering_col);
        factorization_valid_ = false;
        ++update_failures_;
        last_refactor_reason_ = "INVALID_BASE_OR_DIMENSION";
        return false;
    }

    const double d_p = d[leaving_row];
    double d_inf = 0.0;
    bool all_finite = std::isfinite(d_p);
    for (std::size_t i = 0; i < m; ++i) {
        if (!std::isfinite(d[i])) {
            all_finite = false;
            break;
        }
        d_inf = std::max(d_inf, std::abs(d[i]));
    }

    const double abs_pivot = std::abs(d_p);
    const double min_abs_pivot = std::max(pivot_tol, 1e-11);
    if (!all_finite || abs_pivot < min_abs_pivot || abs_pivot < min_rel_pivot * d_inf) {
        update_pivot_combinatorial(leaving_row, entering_col);
        factorization_valid_ = false;
        ++update_failures_;
        last_refactor_reason_ = "UNSTABLE_ETA_PIVOT";
        const auto t_end = std::chrono::steady_clock::now();
        update_application_time_seconds_ += std::chrono::duration<double>(t_end - t_start).count();
        return false;
    }

    const std::size_t leaving_var = basis_vars_[leaving_row];
    update_pivot_combinatorial(leaving_row, entering_col);

    SparseEtaFactor eta;
    eta.base_generation = factorization_generation_;
    eta.pivot_row = leaving_row;
    eta.entering_col = entering_col;
    eta.leaving_col = leaving_var;
    eta.pivot_elem = d_p;

    for (std::size_t i = 0; i < m; ++i) {
        if (i != leaving_row && std::abs(d[i]) > 1e-15) {
            eta.off_diag_rows.push_back(i);
            eta.off_diag_vals.push_back(d[i]);
        }
    }

    update_chain_bytes_ += eta.byte_footprint();
    eta_chain_.push_back(std::move(eta));
    factorization_valid_ = true;

    ++basis_updates_;
    maximum_update_chain_ = std::max(maximum_update_chain_, eta_chain_.size());
    cumulative_chain_lengths_ += eta_chain_.size();
    ++chain_length_samples_;
    peak_memory_bytes_ =
        std::max(peak_memory_bytes_, base_factorization_bytes_ + update_chain_bytes_);

    const auto t_end = std::chrono::steady_clock::now();
    update_application_time_seconds_ += std::chrono::duration<double>(t_end - t_start).count();
    return true;
}

double SimplexBasis::compute_primal_residual(const numerics::CscMatrix& A_csc,
                                             const std::vector<double>& x_B,
                                             const std::vector<double>& b) const {
    const std::size_t m = basis_vars_.size();
    if (m == 0 || x_B.size() < m || b.size() < m) {
        return 0.0;
    }

    std::vector<double> r(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        r[i] = -b[i];
    }

    for (std::size_t k = 0; k < m; ++k) {
        const double xk = x_B[k];
        if (!std::isfinite(xk)) {
            max_observed_residual_ = std::numeric_limits<double>::infinity();
            return max_observed_residual_;
        }
        if (xk == 0.0) {
            continue;
        }
        const std::size_t c = basis_vars_[k];
        const std::size_t c_nnz = A_csc.col_nnz(c);
        const std::size_t* r_idx = A_csc.col_indices(c);
        const double* vals = A_csc.col_values(c);
        for (std::size_t p = 0; p < c_nnz; ++p) {
            const std::size_t row = r_idx[p];
            if (row < m) {
                r[row] += vals[p] * xk;
            }
        }
    }

    double max_res = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        if (!std::isfinite(r[i])) {
            max_observed_residual_ = std::numeric_limits<double>::infinity();
            return max_observed_residual_;
        }
        max_res = std::max(max_res, std::abs(r[i]));
    }

    max_observed_residual_ = std::max(max_observed_residual_, max_res);
    return max_res;
}

double SimplexBasis::compute_dual_residual(const numerics::CscMatrix& A_csc,
                                           const std::vector<double>& y,
                                           const std::vector<double>& c_B) const {
    const std::size_t m = basis_vars_.size();
    if (m == 0 || y.size() < m || c_B.size() < m) {
        return 0.0;
    }

    double max_res = 0.0;
    for (std::size_t k = 0; k < m; ++k) {
        const std::size_t c = basis_vars_[k];
        const std::size_t c_nnz = A_csc.col_nnz(c);
        const std::size_t* r_idx = A_csc.col_indices(c);
        const double* vals = A_csc.col_values(c);
        double dot = 0.0;
        for (std::size_t p = 0; p < c_nnz; ++p) {
            const std::size_t row = r_idx[p];
            if (row < m) {
                dot += vals[p] * y[row];
            }
        }
        const double diff = std::abs(dot - c_B[k]);
        if (!std::isfinite(diff)) {
            max_observed_residual_ = std::numeric_limits<double>::infinity();
            return max_observed_residual_;
        }
        max_res = std::max(max_res, diff);
    }

    max_observed_residual_ = std::max(max_observed_residual_, max_res);
    return max_res;
}

void SimplexBasis::accumulate_telemetry(LPSolverTelemetry& tel) const {
    const std::size_t prev_samples = tel.basis_updates;
    const double prev_sum = tel.average_update_chain * static_cast<double>(prev_samples);

    tel.basis_updates += basis_updates_;
    tel.full_refactorizations += full_refactorizations_;
    tel.refactorization_count = tel.full_refactorizations;
    tel.update_failures += update_failures_;
    if (!last_refactor_reason_.empty()) {
        tel.refactorization_reason = last_refactor_reason_;
    }
    tel.maximum_update_chain = std::max(tel.maximum_update_chain, maximum_update_chain_);
    if (tel.basis_updates > 0) {
        tel.average_update_chain = (prev_sum + static_cast<double>(cumulative_chain_lengths_)) /
                                   static_cast<double>(tel.basis_updates);
    }
    tel.basis_residual_max = std::max(tel.basis_residual_max, max_observed_residual_);
    tel.update_application_time_seconds += update_application_time_seconds_;
    tel.full_refactorization_time_seconds += full_refactorization_time_seconds_;
    tel.base_factorization_bytes =
        std::max(tel.base_factorization_bytes, base_factorization_bytes_);
    tel.update_chain_bytes = update_chain_bytes_;
    tel.peak_memory_bytes = std::max(tel.peak_memory_bytes, peak_memory_bytes_);
    tel.memory_after_refactor_bytes =
        std::max(tel.memory_after_refactor_bytes, memory_after_refactor_bytes_);
}

void SimplexBasis::purge_nonbasic_variable(std::size_t var_idx) {
    if (var_idx < is_basic_.size()) {
        is_basic_[var_idx] = false;
    }
    auto it = std::find(nonbasic_vars_.begin(), nonbasic_vars_.end(), var_idx);
    if (it != nonbasic_vars_.end()) {
        nonbasic_vars_.erase(it);
    }
}

BasisState SimplexBasis::export_state() const {
    BasisState state;
    state.basis_vars = basis_vars_;
    state.is_basic = is_basic_;
    state.nonbasic_vars = nonbasic_vars_;
    state.num_rows = basis_vars_.size();
    state.num_cols = is_basic_.size();
    state.is_valid = !basis_vars_.empty();
    return state;
}

void SimplexBasis::load_state(const BasisState& state) {
    basis_vars_ = state.basis_vars;
    is_basic_ = state.is_basic;
    nonbasic_vars_ = state.nonbasic_vars;
    eta_chain_.clear();
    updates_since_refactor_ = 0;
    factorization_valid_ = false;
    update_chain_bytes_ = 0;
}

}  // namespace kavach::algorithms::lp
