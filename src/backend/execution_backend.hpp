#pragma once

#include <cstddef>
#include <vector>

namespace kavach::backend {

enum class BackendType { CPU_SEQUENTIAL, CPU_MULTICORE, CUDA_GPU };

/**
 * @brief Architecture interface scaffolding for [S]/[U] Execution Backend.
 * Full implementation scheduled for Phases 10 & 11 per Blueprint Section 4, 14, 15 & Section 32.
 * Strictly scoped to batched Sparse Matrix-Vector Product (batched SpMV).
 */
class ExecutionBackend {
public:
    virtual ~ExecutionBackend() = default;

    [[nodiscard]] virtual BackendType type() const noexcept = 0;
    [[nodiscard]] virtual bool is_available() const noexcept = 0;

    /**
     * @brief Batched SpMV execution.
     * Evaluates y[i] = A * x[i] across batch of vectors.
     */
    virtual void batched_spmv(std::size_t batch_size, std::size_t rows, std::size_t cols,
                              const double* values, const int* row_ptr, const int* col_indices,
                              const double* batch_x, double* batch_y) = 0;
};

}  // namespace kavach::backend
