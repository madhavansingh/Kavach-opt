#include "numerics/lu_factorization.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kavach::numerics {

namespace {

struct SparseEntry {
    std::size_t col{0};
    double val{0.0};
};

}  // namespace

bool SparseLU::factorize(const CsrMatrix& matrix, double pivot_tol) {
    pivot_tol_ = pivot_tol;
    original_matrix_ = matrix;
    error_msg_.clear();

    if (!matrix.is_valid()) {
        status_ = core::SolverStatus::NUMERICAL_FAILURE;
        error_msg_ = "Input matrix fails CSR structural invariants";
        return false;
    }

    if (matrix.rows() != matrix.cols()) {
        status_ = core::SolverStatus::NUMERICAL_FAILURE;
        error_msg_ = "SparseLU requires a square matrix: rows=" + std::to_string(matrix.rows()) +
                     " != cols=" + std::to_string(matrix.cols());
        return false;
    }

    n_ = matrix.rows();
    if (n_ == 0) {
        L_ = CsrMatrix();
        U_ = CsrMatrix();
        perm_r_.clear();
        inv_perm_r_.clear();
        min_pivot_ = 0.0;
        max_pivot_ = 0.0;
        status_ = core::SolverStatus::OPTIMAL;
        return true;
    }

    // Working rows for U and active submatrix
    std::vector<std::vector<SparseEntry>> working_rows(n_);
    const auto& row_ptr = matrix.row_ptr();
    const auto& col_idx = matrix.col_idx();
    const auto& vals = matrix.values();

    for (std::size_t i = 0; i < n_; ++i) {
        std::size_t start = row_ptr[i];
        std::size_t end = row_ptr[i + 1];
        working_rows[i].reserve(end - start);
        for (std::size_t p = start; p < end; ++p) {
            if (std::abs(vals[p]) > 1e-18) {
                working_rows[i].push_back({col_idx[p], vals[p]});
            }
        }
        // Ensure sorted by column
        std::sort(working_rows[i].begin(), working_rows[i].end(),
                  [](const SparseEntry& a, const SparseEntry& b) { return a.col < b.col; });
    }

    // Working rows for L (multipliers)
    std::vector<std::vector<SparseEntry>> L_rows(n_);
    perm_r_.resize(n_);
    inv_perm_r_.resize(n_);
    for (std::size_t i = 0; i < n_; ++i) {
        perm_r_[i] = i;
    }

    min_pivot_ = std::numeric_limits<double>::infinity();
    max_pivot_ = 0.0;

    // Elimination step k from 0 to n - 1
    for (std::size_t k = 0; k < n_; ++k) {
        // Step 1: Find row p >= k with column k maximizing |A_pk|
        std::size_t pivot_row = n_;
        double max_val = 0.0;

        for (std::size_t p = k; p < n_; ++p) {
            if (!working_rows[p].empty() && working_rows[p][0].col == k) {
                double abs_v = std::abs(working_rows[p][0].val);
                if (abs_v > max_val) {
                    max_val = abs_v;
                    pivot_row = p;
                }
            }
        }

        // Singularity / near-singularity check against pivot_tol
        if (pivot_row == n_ || max_val < pivot_tol_) {
            status_ = core::SolverStatus::NUMERICAL_FAILURE;
            error_msg_ = "Near-zero or zero pivot at column " + std::to_string(k) +
                         ": max pivot magnitude=" + std::to_string(max_val) +
                         " < pivot_tol=" + std::to_string(pivot_tol_);
            return false;
        }

        // Step 2: Swap row k and pivot_row
        if (pivot_row != k) {
            std::swap(working_rows[k], working_rows[pivot_row]);
            std::swap(L_rows[k], L_rows[pivot_row]);
            std::swap(perm_r_[k], perm_r_[pivot_row]);
        }

        min_pivot_ = std::min(min_pivot_, max_val);
        max_pivot_ = std::max(max_pivot_, max_val);

        double pivot_val = working_rows[k][0].val;

        // Step 3: Eliminate column k from all rows i > k with entry at column k
        for (std::size_t i = k + 1; i < n_; ++i) {
            if (working_rows[i].empty() || working_rows[i][0].col != k) {
                continue;
            }

            double a_ik = working_rows[i][0].val;
            double mult = a_ik / pivot_val;

            // Record multiplier in L
            L_rows[i].push_back({k, mult});

            // Perform row operation: row[i] = row[i] - mult * row[k]
            // Both row[i] and row[k] have first entry at col k.
            // Merge remaining entries (index >= 1) with 2-pointer linear merge.
            const auto& r_i = working_rows[i];
            const auto& r_k = working_rows[k];

            std::vector<SparseEntry> new_row;
            new_row.reserve(r_i.size() + r_k.size());

            std::size_t p1 = 1;
            std::size_t p2 = 1;

            while (p1 < r_i.size() && p2 < r_k.size()) {
                if (r_i[p1].col < r_k[p2].col) {
                    new_row.push_back(r_i[p1]);
                    p1++;
                } else if (r_k[p2].col < r_i[p1].col) {
                    double v = -mult * r_k[p2].val;
                    if (std::abs(v) > 1e-15) {
                        new_row.push_back({r_k[p2].col, v});
                    }
                    p2++;
                } else {
                    double v = r_i[p1].val - mult * r_k[p2].val;
                    if (std::abs(v) > 1e-15) {
                        new_row.push_back({r_i[p1].col, v});
                    }
                    p1++;
                    p2++;
                }
            }

            while (p1 < r_i.size()) {
                new_row.push_back(r_i[p1]);
                p1++;
            }
            while (p2 < r_k.size()) {
                double v = -mult * r_k[p2].val;
                if (std::abs(v) > 1e-15) {
                    new_row.push_back({r_k[p2].col, v});
                }
                p2++;
            }

            working_rows[i] = std::move(new_row);
        }
    }

    // Build inverse row permutation
    for (std::size_t i = 0; i < n_; ++i) {
        inv_perm_r_[perm_r_[i]] = i;
    }

    // Construct L as CsrMatrix
    // Row k of L has multipliers L_rows[k] for j < k, plus (k, 1.0) on diagonal
    std::vector<std::size_t> L_row_ptr(n_ + 1, 0);
    std::size_t L_nnz = 0;
    for (std::size_t i = 0; i < n_; ++i) {
        L_nnz += L_rows[i].size() + 1;  // +1 for diagonal 1.0
        L_row_ptr[i + 1] = L_nnz;
    }

    std::vector<std::size_t> L_col_idx;
    std::vector<double> L_vals;
    L_col_idx.reserve(L_nnz);
    L_vals.reserve(L_nnz);

    for (std::size_t i = 0; i < n_; ++i) {
        // Multipliers are already sorted by col < i
        for (const auto& entry : L_rows[i]) {
            L_col_idx.push_back(entry.col);
            L_vals.push_back(entry.val);
        }
        // Unit diagonal
        L_col_idx.push_back(i);
        L_vals.push_back(1.0);
    }
    L_ = CsrMatrix(n_, n_, std::move(L_row_ptr), std::move(L_col_idx), std::move(L_vals));

    // Construct U as CsrMatrix
    std::vector<std::size_t> U_row_ptr(n_ + 1, 0);
    std::size_t U_nnz = 0;
    for (std::size_t i = 0; i < n_; ++i) {
        U_nnz += working_rows[i].size();
        U_row_ptr[i + 1] = U_nnz;
    }

    std::vector<std::size_t> U_col_idx;
    std::vector<double> U_vals;
    U_col_idx.reserve(U_nnz);
    U_vals.reserve(U_nnz);

    for (std::size_t i = 0; i < n_; ++i) {
        for (const auto& entry : working_rows[i]) {
            U_col_idx.push_back(entry.col);
            U_vals.push_back(entry.val);
        }
    }
    U_ = CsrMatrix(n_, n_, std::move(U_row_ptr), std::move(U_col_idx), std::move(U_vals));

    status_ = core::SolverStatus::OPTIMAL;
    return true;
}

bool SparseLU::factorize(const SparseMatrix& matrix, double pivot_tol) {
    if (const auto* csr = dynamic_cast<const CsrMatrix*>(&matrix)) {
        return factorize(*csr, pivot_tol);
    }
    if (const auto* csc = dynamic_cast<const CscMatrix*>(&matrix)) {
        CsrMatrix csr = csc_to_csr(*csc);
        return factorize(csr, pivot_tol);
    }
    status_ = core::SolverStatus::NUMERICAL_FAILURE;
    error_msg_ = "Unsupported SparseMatrix format for factorization";
    return false;
}

void SparseLU::refactorize() {
    if (n_ > 0 || original_matrix_.rows() > 0) {
        factorize(original_matrix_, pivot_tol_);
    }
}

bool SparseLU::solve(const std::vector<double>& rhs, std::vector<double>& solution) const {
    if (!is_factorized()) {
        return false;
    }
    if (rhs.size() != n_) {
        return false;
    }

    solution.resize(n_);
    if (n_ == 0) {
        return true;
    }

    // 1. Permute rhs: y_rhs = P * b
    // Row k of P*b corresponds to b[perm_r_[k]]
    std::vector<double> y(n_);
    for (std::size_t i = 0; i < n_; ++i) {
        y[i] = rhs[perm_r_[i]];
    }

    // 2. Forward solve: L * y = P * b
    // L has unit diagonal.
    const auto& L_row_ptr = L_.row_ptr();
    const auto& L_col_idx = L_.col_idx();
    const auto& L_vals = L_.values();

    for (std::size_t i = 0; i < n_; ++i) {
        std::size_t start = L_row_ptr[i];
        std::size_t end = L_row_ptr[i + 1];
        for (std::size_t p = start; p < end; ++p) {
            std::size_t j = L_col_idx[p];
            if (j < i) {
                y[i] -= L_vals[p] * y[j];
            }
        }
    }

    // 3. Backward solve: U * x = y
    // U is upper triangular with non-zero diagonal
    const auto& U_row_ptr = U_.row_ptr();
    const auto& U_col_idx = U_.col_idx();
    const auto& U_vals = U_.values();

    for (std::size_t row_plus_one = n_; row_plus_one > 0; --row_plus_one) {
        std::size_t i = row_plus_one - 1;
        std::size_t start = U_row_ptr[i];
        std::size_t end = U_row_ptr[i + 1];

        double diag_val = 0.0;
        bool diag_found = false;
        double sum = y[i];

        for (std::size_t p = start; p < end; ++p) {
            std::size_t j = U_col_idx[p];
            if (j == i) {
                diag_val = U_vals[p];
                diag_found = true;
            } else if (j > i) {
                sum -= U_vals[p] * solution[j];
            }
        }

        if (!diag_found || std::abs(diag_val) < 1e-18) {
            return false;
        }

        solution[i] = sum / diag_val;
    }

    return true;
}

std::vector<double> SparseLU::solve(const std::vector<double>& rhs) const {
    std::vector<double> solution;
    if (!solve(rhs, solution)) {
        throw std::runtime_error("SparseLU::solve failed: " + error_msg_);
    }
    return solution;
}

bool SparseLU::solve_transpose(const std::vector<double>& rhs,
                               std::vector<double>& solution) const {
    if (!is_factorized()) {
        return false;
    }
    if (rhs.size() != n_) {
        return false;
    }

    solution.resize(n_);
    if (n_ == 0) {
        return true;
    }

    // System: A^T * x = b
    // A = P^T * L * U => A^T = U^T * L^T * P
    // U^T * L^T * (P * x) = b
    // 1. Solve U^T * z = b
    // 2. Solve L^T * w = z
    // 3. x = P^T * w, so x[perm_r_[k]] = w[k]

    // 1. Forward solve: U^T * z = b
    // U^T is lower triangular. Using push-algorithm over rows of U:
    std::vector<double> z = rhs;
    const auto& U_row_ptr = U_.row_ptr();
    const auto& U_col_idx = U_.col_idx();
    const auto& U_vals = U_.values();

    for (std::size_t i = 0; i < n_; ++i) {
        std::size_t start = U_row_ptr[i];
        std::size_t end = U_row_ptr[i + 1];

        double diag_val = 0.0;
        bool diag_found = false;

        // Find U_ii
        for (std::size_t p = start; p < end; ++p) {
            if (U_col_idx[p] == i) {
                diag_val = U_vals[p];
                diag_found = true;
                break;
            }
        }

        if (!diag_found || std::abs(diag_val) < 1e-18) {
            return false;
        }

        z[i] /= diag_val;

        // Push updates to z[j] for j > i
        for (std::size_t p = start; p < end; ++p) {
            std::size_t j = U_col_idx[p];
            if (j > i) {
                z[j] -= U_vals[p] * z[i];
            }
        }
    }

    // 2. Backward solve: L^T * w = z
    // L^T is unit upper triangular. Using push-algorithm over rows of L from n-1 down to 0:
    std::vector<double> w = z;
    const auto& L_row_ptr = L_.row_ptr();
    const auto& L_col_idx = L_.col_idx();
    const auto& L_vals = L_.values();

    for (std::size_t row_plus_one = n_; row_plus_one > 0; --row_plus_one) {
        std::size_t i = row_plus_one - 1;
        // w[i] is already finalized since L is unit lower triangular
        std::size_t start = L_row_ptr[i];
        std::size_t end = L_row_ptr[i + 1];

        for (std::size_t p = start; p < end; ++p) {
            std::size_t j = L_col_idx[p];
            if (j < i) {
                w[j] -= L_vals[p] * w[i];
            }
        }
    }

    // 3. x = P^T * w => x[perm_r_[k]] = w[k]
    for (std::size_t k = 0; k < n_; ++k) {
        solution[perm_r_[k]] = w[k];
    }

    return true;
}

std::vector<double> SparseLU::solve_transpose(const std::vector<double>& rhs) const {
    std::vector<double> solution;
    if (!solve_transpose(rhs, solution)) {
        throw std::runtime_error("SparseLU::solve_transpose failed: " + error_msg_);
    }
    return solution;
}

RefinementResult SparseLU::refine_solution(const CsrMatrix& A, const std::vector<double>& b,
                                           std::vector<double>& x, std::size_t max_iterations,
                                           double residual_tol) const {
    RefinementResult result;
    result.converged = false;
    result.iterations = 0;

    auto r = compute_residual(A, x, b);
    result.initial_residual = residual_norm_inf(r);
    result.final_residual = result.initial_residual;

    if (result.initial_residual <= residual_tol) {
        result.converged = true;
        return result;
    }

    double prev_res = result.initial_residual;
    std::vector<double> delta(n_, 0.0);

    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        result.iterations++;

        if (!solve(r, delta)) {
            break;
        }

        for (std::size_t i = 0; i < n_; ++i) {
            x[i] += delta[i];
        }

        r = compute_residual(A, x, b);
        double cur_res = residual_norm_inf(r);
        result.final_residual = cur_res;

        if (cur_res <= residual_tol) {
            result.converged = true;
            break;
        }

        // Stagnation or divergence check
        if (cur_res >= prev_res * 0.9999) {
            break;
        }
        prev_res = cur_res;
    }

    return result;
}

double SparseLU::condition_estimate() const noexcept {
    if (status_ != core::SolverStatus::OPTIMAL || min_pivot_ <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return max_pivot_ / min_pivot_;
}

std::vector<double> compute_residual(const CsrMatrix& A, const std::vector<double>& x,
                                     const std::vector<double>& b) {
    if (A.cols() != x.size() || A.rows() != b.size()) {
        throw std::invalid_argument("compute_residual: dimension mismatch between A, x, and b");
    }
    auto ax = A.spmv(x);
    std::vector<double> r(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) {
        r[i] = b[i] - ax[i];
    }
    return r;
}

double residual_norm_inf(const std::vector<double>& r) noexcept {
    double max_val = 0.0;
    for (double v : r) {
        max_val = std::max(max_val, std::abs(v));
    }
    return max_val;
}

double relative_residual_norm(const CsrMatrix& A, const std::vector<double>& x,
                              const std::vector<double>& b) {
    if (A.cols() != x.size() || A.rows() != b.size()) {
        return std::numeric_limits<double>::infinity();
    }
    auto r = compute_residual(A, x, b);
    double res_inf = residual_norm_inf(r);

    double a_inf = A.infinity_norm();
    double x_inf = 0.0;
    for (double val : x) {
        x_inf = std::max(x_inf, std::abs(val));
    }
    double b_inf = residual_norm_inf(b);

    double denom = (a_inf * x_inf) + b_inf + 1e-12;
    return res_inf / denom;
}

}  // namespace kavach::numerics
