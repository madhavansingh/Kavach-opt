#pragma once

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

namespace kavach::numerics {
class CsrMatrix;
}

namespace kavach::backend {

enum class BackendType { CPU_SEQUENTIAL, CPU_MULTICORE, CUDA_GPU };

[[nodiscard]] constexpr const char* to_string(BackendType type) noexcept {
    switch (type) {
        case BackendType::CPU_SEQUENTIAL:
            return "CPU_SEQUENTIAL";
        case BackendType::CPU_MULTICORE:
            return "CPU_MULTICORE";
        case BackendType::CUDA_GPU:
            return "CUDA_GPU";
    }
    return "UNKNOWN";
}

inline std::ostream& operator<<(std::ostream& os, BackendType type) {
    return os << to_string(type);
}

/**
 * @brief Options configuring batched SpMV execution.
 */
struct BatchedSpmvOptions {
    std::size_t num_workers{0};  ///< Worker threads (0 = auto-detect hardware concurrency)
    bool transpose{false};       ///< If true, compute Y_k = A^T * X_k
    bool run_parallel{true};     ///< If false, forces serial execution on caller thread
};

/**
 * @brief Telemetry metrics captured during batched SpMV execution.
 */
struct BatchedSpmvTelemetry {
    bool parallel_enabled{false};
    std::size_t worker_count{1};
    std::size_t batch_size{0};
    std::size_t rows{0};
    std::size_t cols{0};
    std::size_t nonzeros{0};
    double serial_time_seconds{0.0};
    double parallel_time_seconds{0.0};
    double speedup{1.0};
    double parallel_efficiency{1.0};
};

/**
 * @brief Detailed numerical comparison result between serial reference and parallel SpMV.
 */
struct SpmvComparisonResult {
    bool passed{true};
    double max_absolute_error{0.0};
    double max_relative_error{0.0};
    std::size_t mismatch_batch_idx{0};
    std::size_t mismatch_elem_idx{0};
    double serial_val{0.0};
    double parallel_val{0.0};
    std::string error_message;
};

/**
 * @brief Architecture interface for Execution Backends (CPU Sequential, CPU Multicore, CUDA GPU).
 * Fulfills Blueprint Section 4, 14, 15, and Section 32 (Phases 10 & 11).
 * Strictly scoped to batched Sparse Matrix-Vector Product (batched SpMV).
 */
class ExecutionBackend {
public:
    virtual ~ExecutionBackend() = default;

    [[nodiscard]] virtual BackendType type() const noexcept = 0;
    [[nodiscard]] virtual bool is_available() const noexcept = 0;

    /**
     * @brief Batched SpMV execution with raw int row_ptr/col_indices.
     * Evaluates y[k] = A * x[k] across batch of vectors.
     */
    virtual void batched_spmv(std::size_t batch_size, std::size_t rows, std::size_t cols,
                              const double* values, const int* row_ptr, const int* col_indices,
                              const double* batch_x, double* batch_y) = 0;

    /**
     * @brief Batched SpMV execution with size_t row_ptr/col_indices.
     */
    virtual void batched_spmv(std::size_t batch_size, std::size_t rows, std::size_t cols,
                              const double* values, const std::size_t* row_ptr,
                              const std::size_t* col_indices, const double* batch_x,
                              double* batch_y) = 0;

    /**
     * @brief Batched SpMV execution using CsrMatrix.
     */
    virtual void batched_spmv(const numerics::CsrMatrix& A, const double* batch_x, double* batch_y,
                              std::size_t batch_size, const BatchedSpmvOptions& options = {}) = 0;
};

}  // namespace kavach::backend
