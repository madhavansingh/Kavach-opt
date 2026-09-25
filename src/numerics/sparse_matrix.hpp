#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace kavach::numerics {

enum class SparseStorageFormat : std::uint8_t {
    CSR,  ///< Compressed Sparse Row (row-major)
    CSC   ///< Compressed Sparse Column (column-major)
};

[[nodiscard]] constexpr std::string_view to_string(SparseStorageFormat format) noexcept {
    switch (format) {
        case SparseStorageFormat::CSR:
            return "CSR";
        case SparseStorageFormat::CSC:
            return "CSC";
    }
    return "UNKNOWN";
}

/**
 * @brief Coordinate (triplet) entry: (row, col, value).
 */
struct Triplet {
    std::size_t row{0};
    std::size_t col{0};
    double value{0.0};

    bool operator==(const Triplet& other) const noexcept {
        return row == other.row && col == other.col && value == other.value;
    }
};

/**
 * @brief Abstract interface for sparse matrices.
 */
class SparseMatrix {
public:
    virtual ~SparseMatrix() = default;

    [[nodiscard]] virtual std::size_t rows() const noexcept = 0;
    [[nodiscard]] virtual std::size_t cols() const noexcept = 0;
    [[nodiscard]] virtual std::size_t nonzeros() const noexcept = 0;
    [[nodiscard]] virtual SparseStorageFormat format() const noexcept = 0;

    virtual void spmv(const double* x, double* y) const = 0;

    [[nodiscard]] virtual double frobenius_norm() const noexcept = 0;
    [[nodiscard]] virtual double infinity_norm() const noexcept = 0;
    [[nodiscard]] virtual double one_norm() const noexcept = 0;
};

class CscMatrix;  // Forward declaration for conversions

/**
 * @brief Compressed Sparse Row (CSR) matrix representation.
 *
 * Invariants:
 * - row_ptr.size() == rows + 1
 * - row_ptr[0] == 0
 * - row_ptr.back() == nnz == col_idx.size() == values.size()
 * - row_ptr is monotonically non-decreasing
 * - col_idx[k] < cols for all entries
 * - Columns within each row are strictly ascending (canonicalized)
 * - All values are finite
 */
class CsrMatrix : public SparseMatrix {
public:
    CsrMatrix() : CsrMatrix(0, 0) {}
    CsrMatrix(std::size_t rows, std::size_t cols);
    CsrMatrix(std::size_t rows, std::size_t cols, std::vector<std::size_t> row_ptr,
              std::vector<std::size_t> col_idx, std::vector<double> values,
              bool canonicalize_entries = true);

    [[nodiscard]] static CsrMatrix from_triplets(std::size_t rows, std::size_t cols,
                                                 const std::vector<Triplet>& triplets,
                                                 bool sum_duplicates = true);

    [[nodiscard]] static CsrMatrix identity(std::size_t n);
    [[nodiscard]] static CsrMatrix diagonal(const std::vector<double>& diag);

    // --- Dimensions & Format ---
    [[nodiscard]] std::size_t rows() const noexcept override { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept override { return cols_; }
    [[nodiscard]] std::size_t nonzeros() const noexcept override { return values_.size(); }
    [[nodiscard]] SparseStorageFormat format() const noexcept override {
        return SparseStorageFormat::CSR;
    }

    // --- Raw Storage Access ---
    [[nodiscard]] const std::vector<std::size_t>& row_ptr() const noexcept { return row_ptr_; }
    [[nodiscard]] const std::vector<std::size_t>& col_idx() const noexcept { return col_idx_; }
    [[nodiscard]] const std::vector<double>& values() const noexcept { return values_; }

    [[nodiscard]] std::vector<std::size_t>& row_ptr() noexcept { return row_ptr_; }
    [[nodiscard]] std::vector<std::size_t>& col_idx() noexcept { return col_idx_; }
    [[nodiscard]] std::vector<double>& values() noexcept { return values_; }

    // --- Row Slicing ---
    [[nodiscard]] std::size_t row_start(std::size_t r) const;
    [[nodiscard]] std::size_t row_end(std::size_t r) const;
    [[nodiscard]] std::size_t row_nnz(std::size_t r) const;

    // --- Element Access ---
    [[nodiscard]] double get(std::size_t r, std::size_t c) const noexcept;

    // --- Invariant Validation & Canonicalization ---
    void canonicalize(bool sum_duplicates = true, double drop_tol = 0.0);
    [[nodiscard]] bool is_valid(std::string* error_msg = nullptr) const noexcept;
    void validate() const;

    // --- SpMV ---
    void spmv(const double* x, double* y) const override;
    [[nodiscard]] std::vector<double> spmv(const std::vector<double>& x) const;
    void spmv(double alpha, const std::vector<double>& x, double beta,
              std::vector<double>& y) const;

    // --- Norms ---
    [[nodiscard]] double frobenius_norm() const noexcept override;
    [[nodiscard]] double infinity_norm() const noexcept override;
    [[nodiscard]] double one_norm() const noexcept override;

private:
    std::size_t rows_{0};
    std::size_t cols_{0};
    std::vector<std::size_t> row_ptr_;
    std::vector<std::size_t> col_idx_;
    std::vector<double> values_;
};

/**
 * @brief Compressed Sparse Column (CSC) matrix representation.
 *
 * Invariants:
 * - col_ptr.size() == cols + 1
 * - col_ptr[0] == 0
 * - col_ptr.back() == nnz == row_idx.size() == values.size()
 * - col_ptr is monotonically non-decreasing
 * - row_idx[k] < rows for all entries
 * - Rows within each column are strictly ascending (canonicalized)
 * - All values are finite
 */
class CscMatrix : public SparseMatrix {
public:
    CscMatrix() : CscMatrix(0, 0) {}
    CscMatrix(std::size_t rows, std::size_t cols);
    CscMatrix(std::size_t rows, std::size_t cols, std::vector<std::size_t> col_ptr,
              std::vector<std::size_t> row_idx, std::vector<double> values,
              bool canonicalize_entries = true);

    [[nodiscard]] static CscMatrix from_triplets(std::size_t rows, std::size_t cols,
                                                 const std::vector<Triplet>& triplets,
                                                 bool sum_duplicates = true);

    [[nodiscard]] static CscMatrix identity(std::size_t n);
    [[nodiscard]] static CscMatrix diagonal(const std::vector<double>& diag);

    // --- Dimensions & Format ---
    [[nodiscard]] std::size_t rows() const noexcept override { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept override { return cols_; }
    [[nodiscard]] std::size_t nonzeros() const noexcept override { return values_.size(); }
    [[nodiscard]] SparseStorageFormat format() const noexcept override {
        return SparseStorageFormat::CSC;
    }

    // --- Raw Storage Access ---
    [[nodiscard]] const std::vector<std::size_t>& col_ptr() const noexcept { return col_ptr_; }
    [[nodiscard]] const std::vector<std::size_t>& row_idx() const noexcept { return row_idx_; }
    [[nodiscard]] const std::vector<double>& values() const noexcept { return values_; }

    [[nodiscard]] std::vector<std::size_t>& col_ptr() noexcept { return col_ptr_; }
    [[nodiscard]] std::vector<std::size_t>& row_idx() noexcept { return row_idx_; }
    [[nodiscard]] std::vector<double>& values() noexcept { return values_; }

    // --- Column Slicing ---
    [[nodiscard]] std::size_t col_start(std::size_t c) const;
    [[nodiscard]] std::size_t col_end(std::size_t c) const;
    [[nodiscard]] std::size_t col_nnz(std::size_t c) const;

    // --- Element Access ---
    [[nodiscard]] double get(std::size_t r, std::size_t c) const noexcept;

    // --- Invariant Validation & Canonicalization ---
    void canonicalize(bool sum_duplicates = true, double drop_tol = 0.0);
    [[nodiscard]] bool is_valid(std::string* error_msg = nullptr) const noexcept;
    void validate() const;

    // --- SpMV ---
    void spmv(const double* x, double* y) const override;
    [[nodiscard]] std::vector<double> spmv(const std::vector<double>& x) const;
    void spmv(double alpha, const std::vector<double>& x, double beta,
              std::vector<double>& y) const;

    // --- SpMV Transpose (y = A^T x) ---
    void spmv_transpose(const double* x, double* y) const;
    [[nodiscard]] std::vector<double> spmv_transpose(const std::vector<double>& x) const;

    // --- Norms ---
    [[nodiscard]] double frobenius_norm() const noexcept override;
    [[nodiscard]] double infinity_norm() const noexcept override;
    [[nodiscard]] double one_norm() const noexcept override;

private:
    std::size_t rows_{0};
    std::size_t cols_{0};
    std::vector<std::size_t> col_ptr_;
    std::vector<std::size_t> row_idx_;
    std::vector<double> values_;
};

// --- Exact Conversion Functions (Zero Dense Allocation) ---
[[nodiscard]] CscMatrix csr_to_csc(const CsrMatrix& csr);
[[nodiscard]] CsrMatrix csc_to_csr(const CscMatrix& csc);

// --- Sparse Transpose ---
[[nodiscard]] CsrMatrix transpose(const CsrMatrix& csr);
[[nodiscard]] CscMatrix transpose(const CscMatrix& csc);

}  // namespace kavach::numerics
