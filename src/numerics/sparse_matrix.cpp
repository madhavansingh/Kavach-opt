#include "numerics/sparse_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <string>

#include "core/error.hpp"

namespace kavach::numerics {

// ============================================================================
// CsrMatrix Implementation
// ============================================================================

CsrMatrix::CsrMatrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), row_ptr_(rows + 1, 0) {}

CsrMatrix::CsrMatrix(std::size_t rows, std::size_t cols, std::vector<std::size_t> row_ptr,
                     std::vector<std::size_t> col_idx, std::vector<double> values,
                     bool canonicalize_entries)
    : rows_(rows),
      cols_(cols),
      row_ptr_(std::move(row_ptr)),
      col_idx_(std::move(col_idx)),
      values_(std::move(values)) {
    if (canonicalize_entries) {
        canonicalize();
    }
}

CsrMatrix CsrMatrix::from_triplets(std::size_t rows, std::size_t cols,
                                   const std::vector<Triplet>& triplets, bool sum_duplicates) {
    CsrMatrix mat(rows, cols);
    if (triplets.empty()) {
        return mat;
    }

    // 1. Count nonzeros per row
    std::vector<std::size_t> row_counts(rows, 0);
    for (const auto& t : triplets) {
        if (t.row >= rows || t.col >= cols) {
            throw core::InputException("Triplet index out of range: (" + std::to_string(t.row) +
                                       ", " + std::to_string(t.col) + ") for size (" +
                                       std::to_string(rows) + ", " + std::to_string(cols) + ")");
        }
        if (std::isnan(t.value) || std::isinf(t.value)) {
            throw core::NumericalException("Non-finite value in triplet at (" +
                                           std::to_string(t.row) + ", " + std::to_string(t.col) +
                                           "): " + std::to_string(t.value));
        }
        if (t.value != 0.0) {
            ++row_counts[t.row];
        }
    }

    // 2. Prefix sum into row_ptr
    mat.row_ptr_[0] = 0;
    for (std::size_t i = 0; i < rows; ++i) {
        mat.row_ptr_[i + 1] = mat.row_ptr_[i] + row_counts[i];
    }

    mat.col_idx_.resize(mat.row_ptr_.back());
    mat.values_.resize(mat.row_ptr_.back());

    // 3. Populate entries
    std::vector<std::size_t> current_pos = mat.row_ptr_;
    for (const auto& t : triplets) {
        if (t.value != 0.0) {
            std::size_t dest = current_pos[t.row]++;
            mat.col_idx_[dest] = t.col;
            mat.values_[dest] = t.value;
        }
    }

    // 4. Canonicalize (sort by column within each row and sum duplicates)
    mat.canonicalize(sum_duplicates);
    return mat;
}

CsrMatrix CsrMatrix::identity(std::size_t n) {
    CsrMatrix mat(n, n);
    mat.row_ptr_.resize(n + 1);
    mat.col_idx_.resize(n);
    mat.values_.resize(n, 1.0);

    for (std::size_t i = 0; i < n; ++i) {
        mat.row_ptr_[i] = i;
        mat.col_idx_[i] = i;
    }
    mat.row_ptr_[n] = n;
    return mat;
}

CsrMatrix CsrMatrix::diagonal(const std::vector<double>& diag) {
    std::size_t n = diag.size();
    CsrMatrix mat(n, n);
    mat.row_ptr_.resize(n + 1, 0);

    std::size_t nnz = 0;
    for (std::size_t i = 0; i < n; ++i) {
        mat.row_ptr_[i] = nnz;
        if (diag[i] != 0.0) {
            mat.col_idx_.push_back(i);
            mat.values_.push_back(diag[i]);
            ++nnz;
        }
    }
    mat.row_ptr_[n] = nnz;
    return mat;
}

std::size_t CsrMatrix::row_start(std::size_t r) const {
    if (r >= rows_) {
        throw core::InputException("Row index out of range: " + std::to_string(r));
    }
    return row_ptr_[r];
}

std::size_t CsrMatrix::row_end(std::size_t r) const {
    if (r >= rows_) {
        throw core::InputException("Row index out of range: " + std::to_string(r));
    }
    return row_ptr_[r + 1];
}

std::size_t CsrMatrix::row_nnz(std::size_t r) const {
    if (r >= rows_) {
        throw core::InputException("Row index out of range: " + std::to_string(r));
    }
    return row_ptr_[r + 1] - row_ptr_[r];
}

const std::size_t* CsrMatrix::row_indices(std::size_t r) const {
    if (r >= rows_) {
        throw core::InputException("Row index out of range: " + std::to_string(r));
    }
    return col_idx_.data() + row_ptr_[r];
}

const double* CsrMatrix::row_values(std::size_t r) const {
    if (r >= rows_) {
        throw core::InputException("Row index out of range: " + std::to_string(r));
    }
    return values_.data() + row_ptr_[r];
}

double CsrMatrix::get(std::size_t r, std::size_t c) const noexcept {
    if (r >= rows_ || c >= cols_) {
        return 0.0;
    }
    std::size_t start = row_ptr_[r];
    std::size_t end = row_ptr_[r + 1];

    auto it = std::lower_bound(col_idx_.begin() + static_cast<std::ptrdiff_t>(start),
                               col_idx_.begin() + static_cast<std::ptrdiff_t>(end), c);
    if (it != col_idx_.begin() + static_cast<std::ptrdiff_t>(end) && *it == c) {
        auto idx = static_cast<std::size_t>(std::distance(col_idx_.begin(), it));
        return values_[idx];
    }
    return 0.0;
}

void CsrMatrix::canonicalize(bool sum_duplicates, double drop_tol) {
    if (row_ptr_.size() != rows_ + 1) {
        return;
    }

    std::vector<std::size_t> new_row_ptr(rows_ + 1, 0);
    std::vector<std::size_t> new_col_idx;
    std::vector<double> new_values;
    new_col_idx.reserve(col_idx_.size());
    new_values.reserve(values_.size());

    struct Element {
        std::size_t col;
        double val;
    };
    std::vector<Element> row_elements;

    for (std::size_t r = 0; r < rows_; ++r) {
        new_row_ptr[r] = new_col_idx.size();
        std::size_t start = row_ptr_[r];
        std::size_t end = row_ptr_[r + 1];

        row_elements.clear();
        for (std::size_t k = start; k < end; ++k) {
            row_elements.push_back({.col = col_idx_[k], .val = values_[k]});
        }

        std::ranges::sort(row_elements, [](const Element& a, const Element& b) noexcept {
            return a.col < b.col;
        });

        // Merge duplicate columns and drop near-zeros
        for (const auto& elem : row_elements) {
            if (new_col_idx.size() > new_row_ptr[r] && new_col_idx.back() == elem.col) {
                if (sum_duplicates) {
                    new_values.back() += elem.val;
                } else {
                    new_values.back() = elem.val;
                }
            } else {
                new_col_idx.push_back(elem.col);
                new_values.push_back(elem.val);
            }
        }

        // Post-filter exact zeros or entries below drop_tol
        std::size_t write_pos = new_row_ptr[r];
        for (std::size_t read_pos = new_row_ptr[r]; read_pos < new_col_idx.size(); ++read_pos) {
            if (std::abs(new_values[read_pos]) > drop_tol) {
                new_col_idx[write_pos] = new_col_idx[read_pos];
                new_values[write_pos] = new_values[read_pos];
                ++write_pos;
            }
        }
        new_col_idx.resize(write_pos);
        new_values.resize(write_pos);
    }

    new_row_ptr[rows_] = new_col_idx.size();
    row_ptr_ = std::move(new_row_ptr);
    col_idx_ = std::move(new_col_idx);
    values_ = std::move(new_values);
}

bool CsrMatrix::is_valid(std::string* error_msg) const noexcept {
    if (row_ptr_.size() != rows_ + 1) {
        if (error_msg != nullptr) {
            *error_msg = "row_ptr size (" + std::to_string(row_ptr_.size()) + ") != rows + 1 (" +
                         std::to_string(rows_ + 1) + ")";
        }
        return false;
    }
    if (row_ptr_[0] != 0) {
        if (error_msg != nullptr) {
            *error_msg = "row_ptr[0] is not 0 (is " + std::to_string(row_ptr_[0]) + ")";
        }
        return false;
    }
    if (row_ptr_.back() != values_.size() || col_idx_.size() != values_.size()) {
        if (error_msg != nullptr) {
            *error_msg = "Mismatched sizes: row_ptr.back()=" + std::to_string(row_ptr_.back()) +
                         ", col_idx=" + std::to_string(col_idx_.size()) +
                         ", values=" + std::to_string(values_.size());
        }
        return false;
    }
    for (std::size_t r = 0; r < rows_; ++r) {
        if (row_ptr_[r] > row_ptr_[r + 1]) {
            if (error_msg != nullptr) {
                *error_msg = "Non-monotonic row_ptr at row " + std::to_string(r);
            }
            return false;
        }
        for (std::size_t k = row_ptr_[r]; k < row_ptr_[r + 1]; ++k) {
            if (col_idx_[k] >= cols_) {
                if (error_msg != nullptr) {
                    *error_msg = "col_idx (" + std::to_string(col_idx_[k]) + ") >= cols (" +
                                 std::to_string(cols_) + ") at row " + std::to_string(r);
                }
                return false;
            }
            if (k > row_ptr_[r] && col_idx_[k] <= col_idx_[k - 1]) {
                if (error_msg != nullptr) {
                    *error_msg = "Unsorted or duplicate col_idx in row " + std::to_string(r);
                }
                return false;
            }
            if (std::isnan(values_[k]) || std::isinf(values_[k])) {
                if (error_msg != nullptr) {
                    *error_msg = "Non-finite value at (" + std::to_string(r) + ", " +
                                 std::to_string(col_idx_[k]) + ")";
                }
                return false;
            }
        }
    }
    return true;
}

void CsrMatrix::validate() const {
    std::string err;
    if (!is_valid(&err)) {
        throw core::NumericalException("CSR Matrix validation failed: " + err);
    }
}

void CsrMatrix::spmv(const double* x, double* y) const {
    if (x == nullptr || y == nullptr) {
        throw core::InputException("Null pointer passed to CsrMatrix::spmv");
    }
    for (std::size_t r = 0; r < rows_; ++r) {
        double sum = 0.0;
        const std::size_t start = row_ptr_[r];
        const std::size_t end = row_ptr_[r + 1];
        for (std::size_t k = start; k < end; ++k) {
            sum += values_[k] * x[col_idx_[k]];
        }
        y[r] = sum;
    }
}

std::vector<double> CsrMatrix::spmv(const std::vector<double>& x) const {
    if (x.size() != cols_) {
        throw core::InputException("CsrMatrix::spmv dimension mismatch: x size (" +
                                   std::to_string(x.size()) + ") != cols (" +
                                   std::to_string(cols_) + ")");
    }
    std::vector<double> y(rows_, 0.0);
    spmv(x.data(), y.data());
    return y;
}

void CsrMatrix::spmv(double alpha, const std::vector<double>& x, double beta,
                     std::vector<double>& y) const {
    if (x.size() != cols_) {
        throw core::InputException("CsrMatrix::spmv dimension mismatch: x size (" +
                                   std::to_string(x.size()) + ") != cols (" +
                                   std::to_string(cols_) + ")");
    }
    if (y.size() != rows_) {
        throw core::InputException("CsrMatrix::spmv dimension mismatch: y size (" +
                                   std::to_string(y.size()) + ") != rows (" +
                                   std::to_string(rows_) + ")");
    }
    for (std::size_t r = 0; r < rows_; ++r) {
        double sum = 0.0;
        const std::size_t start = row_ptr_[r];
        const std::size_t end = row_ptr_[r + 1];
        for (std::size_t k = start; k < end; ++k) {
            sum += values_[k] * x[col_idx_[k]];
        }
        y[r] = (alpha * sum) + (beta * y[r]);
    }
}

void CsrMatrix::spmv_transpose(const double* x, double* y) const {
    if (x == nullptr || y == nullptr) {
        throw core::InputException("Null pointer passed to CsrMatrix::spmv_transpose");
    }
    std::fill_n(y, cols_, 0.0);
    for (std::size_t r = 0; r < rows_; ++r) {
        const double xr = x[r];
        if (xr == 0.0) {
            continue;
        }
        const std::size_t start = row_ptr_[r];
        const std::size_t end = row_ptr_[r + 1];
        for (std::size_t k = start; k < end; ++k) {
            y[col_idx_[k]] += values_[k] * xr;
        }
    }
}

std::vector<double> CsrMatrix::spmv_transpose(const std::vector<double>& x) const {
    if (x.size() != rows_) {
        throw core::InputException("CsrMatrix::spmv_transpose dimension mismatch: x size (" +
                                   std::to_string(x.size()) + ") != rows (" +
                                   std::to_string(rows_) + ")");
    }
    std::vector<double> y(cols_, 0.0);
    spmv_transpose(x.data(), y.data());
    return y;
}

void CsrMatrix::spmv_transpose(double alpha, const std::vector<double>& x, double beta,
                               std::vector<double>& y) const {
    if (x.size() != rows_) {
        throw core::InputException("CsrMatrix::spmv_transpose dimension mismatch: x size (" +
                                   std::to_string(x.size()) + ") != rows (" +
                                   std::to_string(rows_) + ")");
    }
    if (y.size() != cols_) {
        throw core::InputException("CsrMatrix::spmv_transpose dimension mismatch: y size (" +
                                   std::to_string(y.size()) + ") != cols (" +
                                   std::to_string(cols_) + ")");
    }
    if (beta != 1.0) {
        for (double& v : y) {
            v *= beta;
        }
    }
    for (std::size_t r = 0; r < rows_; ++r) {
        const double axr = alpha * x[r];
        if (axr == 0.0) {
            continue;
        }
        const std::size_t start = row_ptr_[r];
        const std::size_t end = row_ptr_[r + 1];
        for (std::size_t k = start; k < end; ++k) {
            y[col_idx_[k]] += values_[k] * axr;
        }
    }
}

double CsrMatrix::frobenius_norm() const noexcept {
    double sum_sq = 0.0;
    for (double v : values_) {
        sum_sq += v * v;
    }
    return std::sqrt(sum_sq);
}

double CsrMatrix::infinity_norm() const noexcept {
    double max_row_sum = 0.0;
    for (std::size_t r = 0; r < rows_; ++r) {
        double row_sum = 0.0;
        for (std::size_t k = row_ptr_[r]; k < row_ptr_[r + 1]; ++k) {
            row_sum += std::abs(values_[k]);
        }
        max_row_sum = std::max(max_row_sum, row_sum);
    }
    return max_row_sum;
}

double CsrMatrix::one_norm() const noexcept {
    if (cols_ == 0) {
        return 0.0;
    }
    std::vector<double> col_sums(cols_, 0.0);
    for (std::size_t k = 0; k < values_.size(); ++k) {
        col_sums[col_idx_[k]] += std::abs(values_[k]);
    }
    double max_col_sum = 0.0;
    for (double cs : col_sums) {
        if (cs > max_col_sum) {
            max_col_sum = cs;
        }
    }
    return max_col_sum;
}

// ============================================================================
// CscMatrix Implementation
// ============================================================================

CscMatrix::CscMatrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), col_ptr_(cols + 1, 0) {}

CscMatrix::CscMatrix(std::size_t rows, std::size_t cols, std::vector<std::size_t> col_ptr,
                     std::vector<std::size_t> row_idx, std::vector<double> values,
                     bool canonicalize_entries)
    : rows_(rows),
      cols_(cols),
      col_ptr_(std::move(col_ptr)),
      row_idx_(std::move(row_idx)),
      values_(std::move(values)) {
    if (canonicalize_entries) {
        canonicalize();
    }
}

CscMatrix CscMatrix::from_triplets(std::size_t rows, std::size_t cols,
                                   const std::vector<Triplet>& triplets, bool sum_duplicates) {
    CscMatrix mat(rows, cols);
    if (triplets.empty()) {
        return mat;
    }

    std::vector<std::size_t> col_counts(cols, 0);
    for (const auto& t : triplets) {
        if (t.row >= rows || t.col >= cols) {
            throw core::InputException("Triplet index out of range: (" + std::to_string(t.row) +
                                       ", " + std::to_string(t.col) + ") for size (" +
                                       std::to_string(rows) + ", " + std::to_string(cols) + ")");
        }
        if (std::isnan(t.value) || std::isinf(t.value)) {
            throw core::NumericalException("Non-finite value in triplet at (" +
                                           std::to_string(t.row) + ", " + std::to_string(t.col) +
                                           "): " + std::to_string(t.value));
        }
        if (t.value != 0.0) {
            ++col_counts[t.col];
        }
    }

    mat.col_ptr_[0] = 0;
    for (std::size_t j = 0; j < cols; ++j) {
        mat.col_ptr_[j + 1] = mat.col_ptr_[j] + col_counts[j];
    }

    mat.row_idx_.resize(mat.col_ptr_.back());
    mat.values_.resize(mat.col_ptr_.back());

    std::vector<std::size_t> current_pos = mat.col_ptr_;
    for (const auto& t : triplets) {
        if (t.value != 0.0) {
            std::size_t dest = current_pos[t.col]++;
            mat.row_idx_[dest] = t.row;
            mat.values_[dest] = t.value;
        }
    }

    mat.canonicalize(sum_duplicates);
    return mat;
}

CscMatrix CscMatrix::identity(std::size_t n) {
    CscMatrix mat(n, n);
    mat.col_ptr_.resize(n + 1);
    mat.row_idx_.resize(n);
    mat.values_.resize(n, 1.0);

    for (std::size_t j = 0; j < n; ++j) {
        mat.col_ptr_[j] = j;
        mat.row_idx_[j] = j;
    }
    mat.col_ptr_[n] = n;
    return mat;
}

CscMatrix CscMatrix::diagonal(const std::vector<double>& diag) {
    std::size_t n = diag.size();
    CscMatrix mat(n, n);
    mat.col_ptr_.resize(n + 1, 0);

    std::size_t nnz = 0;
    for (std::size_t j = 0; j < n; ++j) {
        mat.col_ptr_[j] = nnz;
        if (diag[j] != 0.0) {
            mat.row_idx_.push_back(j);
            mat.values_.push_back(diag[j]);
            ++nnz;
        }
    }
    mat.col_ptr_[n] = nnz;
    return mat;
}

std::size_t CscMatrix::col_start(std::size_t c) const {
    if (c >= cols_) {
        throw core::InputException("Column index out of range: " + std::to_string(c));
    }
    return col_ptr_[c];
}

std::size_t CscMatrix::col_end(std::size_t c) const {
    if (c >= cols_) {
        throw core::InputException("Column index out of range: " + std::to_string(c));
    }
    return col_ptr_[c + 1];
}

std::size_t CscMatrix::col_nnz(std::size_t c) const {
    if (c >= cols_) {
        throw core::InputException("Column index out of range: " + std::to_string(c));
    }
    return col_ptr_[c + 1] - col_ptr_[c];
}

const std::size_t* CscMatrix::col_indices(std::size_t c) const {
    if (c >= cols_) {
        throw core::InputException("Column index out of range: " + std::to_string(c));
    }
    return row_idx_.data() + col_ptr_[c];
}

const double* CscMatrix::col_values(std::size_t c) const {
    if (c >= cols_) {
        throw core::InputException("Column index out of range: " + std::to_string(c));
    }
    return values_.data() + col_ptr_[c];
}

double CscMatrix::get(std::size_t r, std::size_t c) const noexcept {
    if (r >= rows_ || c >= cols_) {
        return 0.0;
    }
    std::size_t start = col_ptr_[c];
    std::size_t end = col_ptr_[c + 1];

    auto it = std::lower_bound(row_idx_.begin() + static_cast<std::ptrdiff_t>(start),
                               row_idx_.begin() + static_cast<std::ptrdiff_t>(end), r);
    if (it != row_idx_.begin() + static_cast<std::ptrdiff_t>(end) && *it == r) {
        auto idx = static_cast<std::size_t>(std::distance(row_idx_.begin(), it));
        return values_[idx];
    }
    return 0.0;
}

void CscMatrix::canonicalize(bool sum_duplicates, double drop_tol) {
    if (col_ptr_.size() != cols_ + 1) {
        return;
    }

    std::vector<std::size_t> new_col_ptr(cols_ + 1, 0);
    std::vector<std::size_t> new_row_idx;
    std::vector<double> new_values;
    new_row_idx.reserve(row_idx_.size());
    new_values.reserve(values_.size());

    struct Element {
        std::size_t row;
        double val;
    };
    std::vector<Element> col_elements;

    for (std::size_t c = 0; c < cols_; ++c) {
        new_col_ptr[c] = new_row_idx.size();
        std::size_t start = col_ptr_[c];
        std::size_t end = col_ptr_[c + 1];

        col_elements.clear();
        for (std::size_t k = start; k < end; ++k) {
            col_elements.push_back({.row = row_idx_[k], .val = values_[k]});
        }

        std::ranges::sort(col_elements, [](const Element& a, const Element& b) noexcept {
            return a.row < b.row;
        });

        for (const auto& elem : col_elements) {
            if (new_row_idx.size() > new_col_ptr[c] && new_row_idx.back() == elem.row) {
                if (sum_duplicates) {
                    new_values.back() += elem.val;
                } else {
                    new_values.back() = elem.val;
                }
            } else {
                new_row_idx.push_back(elem.row);
                new_values.push_back(elem.val);
            }
        }

        std::size_t write_pos = new_col_ptr[c];
        for (std::size_t read_pos = new_col_ptr[c]; read_pos < new_row_idx.size(); ++read_pos) {
            if (std::abs(new_values[read_pos]) > drop_tol) {
                new_row_idx[write_pos] = new_row_idx[read_pos];
                new_values[write_pos] = new_values[read_pos];
                ++write_pos;
            }
        }
        new_row_idx.resize(write_pos);
        new_values.resize(write_pos);
    }

    new_col_ptr[cols_] = new_row_idx.size();
    col_ptr_ = std::move(new_col_ptr);
    row_idx_ = std::move(new_row_idx);
    values_ = std::move(new_values);
}

bool CscMatrix::is_valid(std::string* error_msg) const noexcept {
    if (col_ptr_.size() != cols_ + 1) {
        if (error_msg != nullptr) {
            *error_msg = "col_ptr size (" + std::to_string(col_ptr_.size()) + ") != cols + 1 (" +
                         std::to_string(cols_ + 1) + ")";
        }
        return false;
    }
    if (col_ptr_[0] != 0) {
        if (error_msg != nullptr) {
            *error_msg = "col_ptr[0] is not 0 (is " + std::to_string(col_ptr_[0]) + ")";
        }
        return false;
    }
    if (col_ptr_.back() != values_.size() || row_idx_.size() != values_.size()) {
        if (error_msg != nullptr) {
            *error_msg = "Mismatched sizes: col_ptr.back()=" + std::to_string(col_ptr_.back()) +
                         ", row_idx=" + std::to_string(row_idx_.size()) +
                         ", values=" + std::to_string(values_.size());
        }
        return false;
    }
    for (std::size_t c = 0; c < cols_; ++c) {
        if (col_ptr_[c] > col_ptr_[c + 1]) {
            if (error_msg != nullptr) {
                *error_msg = "Non-monotonic col_ptr at col " + std::to_string(c);
            }
            return false;
        }
        for (std::size_t k = col_ptr_[c]; k < col_ptr_[c + 1]; ++k) {
            if (row_idx_[k] >= rows_) {
                if (error_msg != nullptr) {
                    *error_msg = "row_idx (" + std::to_string(row_idx_[k]) + ") >= rows (" +
                                 std::to_string(rows_) + ") at col " + std::to_string(c);
                }
                return false;
            }
            if (k > col_ptr_[c] && row_idx_[k] <= row_idx_[k - 1]) {
                if (error_msg != nullptr) {
                    *error_msg = "Unsorted or duplicate row_idx in col " + std::to_string(c);
                }
                return false;
            }
            if (std::isnan(values_[k]) || std::isinf(values_[k])) {
                if (error_msg != nullptr) {
                    *error_msg = "Non-finite value at (" + std::to_string(row_idx_[k]) + ", " +
                                 std::to_string(c) + ")";
                }
                return false;
            }
        }
    }
    return true;
}

void CscMatrix::validate() const {
    std::string err;
    if (!is_valid(&err)) {
        throw core::NumericalException("CSC Matrix validation failed: " + err);
    }
}

void CscMatrix::spmv(const double* x, double* y) const {
    if (x == nullptr || y == nullptr) {
        throw core::InputException("Null pointer passed to CscMatrix::spmv");
    }
    std::fill_n(y, rows_, 0.0);
    for (std::size_t c = 0; c < cols_; ++c) {
        double xc = x[c];
        const std::size_t start = col_ptr_[c];
        const std::size_t end = col_ptr_[c + 1];
        for (std::size_t k = start; k < end; ++k) {
            y[row_idx_[k]] += values_[k] * xc;
        }
    }
}

std::vector<double> CscMatrix::spmv(const std::vector<double>& x) const {
    if (x.size() != cols_) {
        throw core::InputException("CscMatrix::spmv dimension mismatch: x size (" +
                                   std::to_string(x.size()) + ") != cols (" +
                                   std::to_string(cols_) + ")");
    }
    std::vector<double> y(rows_, 0.0);
    spmv(x.data(), y.data());
    return y;
}

void CscMatrix::spmv(double alpha, const std::vector<double>& x, double beta,
                     std::vector<double>& y) const {
    if (x.size() != cols_) {
        throw core::InputException("CscMatrix::spmv dimension mismatch: x size (" +
                                   std::to_string(x.size()) + ") != cols (" +
                                   std::to_string(cols_) + ")");
    }
    if (y.size() != rows_) {
        throw core::InputException("CscMatrix::spmv dimension mismatch: y size (" +
                                   std::to_string(y.size()) + ") != rows (" +
                                   std::to_string(rows_) + ")");
    }
    if (beta != 1.0) {
        for (double& v : y) {
            v *= beta;
        }
    }
    for (std::size_t c = 0; c < cols_; ++c) {
        double axc = alpha * x[c];
        const std::size_t start = col_ptr_[c];
        const std::size_t end = col_ptr_[c + 1];
        for (std::size_t k = start; k < end; ++k) {
            y[row_idx_[k]] += values_[k] * axc;
        }
    }
}

void CscMatrix::spmv_transpose(const double* x, double* y) const {
    if (x == nullptr || y == nullptr) {
        throw core::InputException("Null pointer passed to CscMatrix::spmv_transpose");
    }
    for (std::size_t c = 0; c < cols_; ++c) {
        double sum = 0.0;
        const std::size_t start = col_ptr_[c];
        const std::size_t end = col_ptr_[c + 1];
        for (std::size_t k = start; k < end; ++k) {
            sum += values_[k] * x[row_idx_[k]];
        }
        y[c] = sum;
    }
}

std::vector<double> CscMatrix::spmv_transpose(const std::vector<double>& x) const {
    if (x.size() != rows_) {
        throw core::InputException("CscMatrix::spmv_transpose dimension mismatch: x size (" +
                                   std::to_string(x.size()) + ") != rows (" +
                                   std::to_string(rows_) + ")");
    }
    std::vector<double> y(cols_, 0.0);
    spmv_transpose(x.data(), y.data());
    return y;
}

void CscMatrix::spmv_transpose(double alpha, const std::vector<double>& x, double beta,
                               std::vector<double>& y) const {
    if (x.size() != rows_) {
        throw core::InputException("CscMatrix::spmv_transpose dimension mismatch: x size (" +
                                   std::to_string(x.size()) + ") != rows (" +
                                   std::to_string(rows_) + ")");
    }
    if (y.size() != cols_) {
        throw core::InputException("CscMatrix::spmv_transpose dimension mismatch: y size (" +
                                   std::to_string(y.size()) + ") != cols (" +
                                   std::to_string(cols_) + ")");
    }
    for (std::size_t c = 0; c < cols_; ++c) {
        double sum = 0.0;
        const std::size_t start = col_ptr_[c];
        const std::size_t end = col_ptr_[c + 1];
        for (std::size_t k = start; k < end; ++k) {
            sum += values_[k] * x[row_idx_[k]];
        }
        y[c] = (alpha * sum) + (beta * y[c]);
    }
}

double CscMatrix::frobenius_norm() const noexcept {
    double sum_sq = 0.0;
    for (double v : values_) {
        sum_sq += v * v;
    }
    return std::sqrt(sum_sq);
}

double CscMatrix::infinity_norm() const noexcept {
    if (rows_ == 0) {
        return 0.0;
    }
    std::vector<double> row_sums(rows_, 0.0);
    for (std::size_t k = 0; k < values_.size(); ++k) {
        row_sums[row_idx_[k]] += std::abs(values_[k]);
    }
    double max_row_sum = 0.0;
    for (double rs : row_sums) {
        max_row_sum = std::max(max_row_sum, rs);
    }
    return max_row_sum;
}

double CscMatrix::one_norm() const noexcept {
    double max_col_sum = 0.0;
    for (std::size_t c = 0; c < cols_; ++c) {
        double col_sum = 0.0;
        for (std::size_t k = col_ptr_[c]; k < col_ptr_[c + 1]; ++k) {
            col_sum += std::abs(values_[k]);
        }
        max_col_sum = std::max(max_col_sum, col_sum);
    }
    return max_col_sum;
}

// ============================================================================
// Conversions and Transpose Implementation (Zero Dense Allocation)
// ============================================================================

CscMatrix csr_to_csc(const CsrMatrix& csr) {
    const std::size_t m = csr.rows();
    const std::size_t n = csr.cols();
    const std::size_t nnz = csr.nonzeros();

    CscMatrix csc(m, n);
    if (nnz == 0) {
        return csc;
    }

    // 1. Count nonzeros per column
    std::vector<std::size_t> col_counts(n, 0);
    for (std::size_t k = 0; k < nnz; ++k) {
        ++col_counts[csr.col_idx()[k]];
    }

    // 2. Prefix sum into col_ptr
    csc.col_ptr()[0] = 0;
    for (std::size_t j = 0; j < n; ++j) {
        csc.col_ptr()[j + 1] = csc.col_ptr()[j] + col_counts[j];
    }

    csc.row_idx().resize(nnz);
    csc.values().resize(nnz);

    // 3. Scatter entries into CSC
    std::vector<std::size_t> current_pos = csc.col_ptr();
    for (std::size_t r = 0; r < m; ++r) {
        const std::size_t start = csr.row_ptr()[r];
        const std::size_t end = csr.row_ptr()[r + 1];
        for (std::size_t k = start; k < end; ++k) {
            std::size_t c = csr.col_idx()[k];
            double val = csr.values()[k];
            std::size_t dest = current_pos[c]++;
            csc.row_idx()[dest] = r;
            csc.values()[dest] = val;
        }
    }

    return csc;
}

CsrMatrix csc_to_csr(const CscMatrix& csc) {
    const std::size_t m = csc.rows();
    const std::size_t n = csc.cols();
    const std::size_t nnz = csc.nonzeros();

    CsrMatrix csr(m, n);
    if (nnz == 0) {
        return csr;
    }

    // 1. Count nonzeros per row
    std::vector<std::size_t> row_counts(m, 0);
    for (std::size_t k = 0; k < nnz; ++k) {
        ++row_counts[csc.row_idx()[k]];
    }

    // 2. Prefix sum into row_ptr
    csr.row_ptr()[0] = 0;
    for (std::size_t i = 0; i < m; ++i) {
        csr.row_ptr()[i + 1] = csr.row_ptr()[i] + row_counts[i];
    }

    csr.col_idx().resize(nnz);
    csr.values().resize(nnz);

    // 3. Scatter entries into CSR
    std::vector<std::size_t> current_pos = csr.row_ptr();
    for (std::size_t c = 0; c < n; ++c) {
        const std::size_t start = csc.col_ptr()[c];
        const std::size_t end = csc.col_ptr()[c + 1];
        for (std::size_t k = start; k < end; ++k) {
            std::size_t r = csc.row_idx()[k];
            double val = csc.values()[k];
            std::size_t dest = current_pos[r]++;
            csr.col_idx()[dest] = c;
            csr.values()[dest] = val;
        }
    }

    return csr;
}

CsrMatrix transpose(const CsrMatrix& csr) {
    // Transpose of CSR (m, n) has dimensions (n, m).
    // The CSC representation of csr has col_ptr (size n+1), row_idx (size nnz), values (size nnz).
    // These match row_ptr, col_idx, values for the transposed CSR!
    CscMatrix csc = csr_to_csc(csr);
    return CsrMatrix(csr.cols(), csr.rows(), std::move(csc.col_ptr()), std::move(csc.row_idx()),
                     std::move(csc.values()), false);
}

CscMatrix transpose(const CscMatrix& csc) {
    // Transpose of CSC (m, n) has dimensions (n, m).
    // The CSR representation of csc has row_ptr (size m+1), col_idx (size nnz), values (size nnz).
    // These match col_ptr, row_idx, values for the transposed CSC!
    CsrMatrix csr = csc_to_csr(csc);
    return CscMatrix(csc.cols(), csc.rows(), std::move(csr.row_ptr()), std::move(csr.col_idx()),
                     std::move(csr.values()), false);
}

}  // namespace kavach::numerics
