#include "algorithms/lp/basis.hpp"

#include <algorithm>

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
    nonbasic_vars_.reserve(total_cols - basis_vars_.size());
    for (std::size_t j = 0; j < total_cols; ++j) {
        if (!is_basic_[j]) {
            nonbasic_vars_.push_back(j);
        }
    }

    updates_since_refactor_ = 0;
}

bool SimplexBasis::factorize(const numerics::CscMatrix& A_csc, double pivot_tol) {
    const std::size_t m = basis_vars_.size();
    if (m == 0) {
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
    updates_since_refactor_ = 0;
    return ok;
}

std::vector<double> SimplexBasis::solve_primal(const std::vector<double>& rhs) const {
    return lu_.solve(rhs);
}

std::vector<double> SimplexBasis::solve_dual(const std::vector<double>& c_B) const {
    return lu_.solve_transpose(c_B);
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
    return lu_.solve(a);
}

void SimplexBasis::update_pivot(std::size_t leaving_row, std::size_t entering_col) {
    std::size_t leaving_var = basis_vars_[leaving_row];
    basis_vars_[leaving_row] = entering_col;
    is_basic_[leaving_var] = false;
    is_basic_[entering_col] = true;

    // Swap leaving and entering in nonbasic list
    for (std::size_t& nb : nonbasic_vars_) {
        if (nb == entering_col) {
            nb = leaving_var;
            break;
        }
    }

    ++updates_since_refactor_;
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
    updates_since_refactor_ = 0;
}

}  // namespace kavach::algorithms::lp
