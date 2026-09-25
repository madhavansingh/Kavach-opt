#pragma once

#include <cstddef>
#include <vector>

namespace kavach::numerics {

enum class SparseStorageFormat {
    CSR,  ///< Compressed Sparse Row (row-major)
    CSC   ///< Compressed Sparse Column (column-major)
};

/**
 * @brief Architecture interface scaffolding for [D] Sparse Matrix Layer.
 * Full implementation scheduled for Phase 3 per Blueprint Section 4 & Section 32.
 */
class SparseMatrix {
public:
    virtual ~SparseMatrix() = default;

    [[nodiscard]] virtual std::size_t rows() const noexcept = 0;
    [[nodiscard]] virtual std::size_t cols() const noexcept = 0;
    [[nodiscard]] virtual std::size_t nonzeros() const noexcept = 0;
    [[nodiscard]] virtual SparseStorageFormat format() const noexcept = 0;

    virtual void spmv(const double* x, double* y) const = 0;
};

}  // namespace kavach::numerics
