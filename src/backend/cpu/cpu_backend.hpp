#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <vector>

#include "backend/cpu/thread_pool.hpp"
#include "backend/execution_backend.hpp"
#include "numerics/sparse_matrix.hpp"

namespace kavach::backend::cpu {

/**
 * @brief Serial CPU execution backend.
 * Implements ExecutionBackend interface sequentially on the calling thread.
 */
class CpuSequentialBackend : public ExecutionBackend {
public:
    CpuSequentialBackend() = default;
    ~CpuSequentialBackend() override = default;

    [[nodiscard]] BackendType type() const noexcept override { return BackendType::CPU_SEQUENTIAL; }
    [[nodiscard]] bool is_available() const noexcept override { return true; }

    void batched_spmv(std::size_t batch_size, std::size_t rows, std::size_t cols,
                      const double* values, const int* row_ptr, const int* col_indices,
                      const double* batch_x, double* batch_y) override;

    void batched_spmv(std::size_t batch_size, std::size_t rows, std::size_t cols,
                      const double* values, const std::size_t* row_ptr,
                      const std::size_t* col_indices, const double* batch_x,
                      double* batch_y) override;

    void batched_spmv(const numerics::CsrMatrix& A, const double* batch_x, double* batch_y,
                      std::size_t batch_size, const BatchedSpmvOptions& options = {}) override;
};

/**
 * @brief Multicore CPU execution backend using an internal ThreadPool.
 * Implements ExecutionBackend interface in parallel across CPU cores.
 */
class CpuMulticoreBackend : public ExecutionBackend {
public:
    /**
     * @brief Construct with an optional worker count (0 = auto-detect).
     */
    explicit CpuMulticoreBackend(std::size_t num_workers = 0);

    /**
     * @brief Construct sharing an existing external ThreadPool.
     */
    explicit CpuMulticoreBackend(std::shared_ptr<ThreadPool> thread_pool);

    ~CpuMulticoreBackend() override = default;

    [[nodiscard]] BackendType type() const noexcept override { return BackendType::CPU_MULTICORE; }
    [[nodiscard]] bool is_available() const noexcept override { return true; }

    [[nodiscard]] std::size_t worker_count() const noexcept {
        return thread_pool_ ? thread_pool_->worker_count() : 1;
    }

    [[nodiscard]] const BatchedSpmvTelemetry& last_telemetry() const noexcept {
        return last_telemetry_;
    }

    void batched_spmv(std::size_t batch_size, std::size_t rows, std::size_t cols,
                      const double* values, const int* row_ptr, const int* col_indices,
                      const double* batch_x, double* batch_y) override;

    void batched_spmv(std::size_t batch_size, std::size_t rows, std::size_t cols,
                      const double* values, const std::size_t* row_ptr,
                      const std::size_t* col_indices, const double* batch_x,
                      double* batch_y) override;

    void batched_spmv(const numerics::CsrMatrix& A, const double* batch_x, double* batch_y,
                      std::size_t batch_size, const BatchedSpmvOptions& options = {}) override;

private:
    std::shared_ptr<ThreadPool> thread_pool_;
    BatchedSpmvTelemetry last_telemetry_;
};

// ============================================================================
// Free / Utility Functions for Batched SpMV
// ============================================================================

/**
 * @brief Reference serial batched SpMV execution.
 * Evaluates Y_k = A * X_k (or A^T * X_k) sequentially for k in [0, batch_size).
 */
void serial_batched_spmv(const numerics::CsrMatrix& A, const double* batch_x, double* batch_y,
                         std::size_t batch_size, bool transpose = false);

/**
 * @brief Multicore parallel batched SpMV execution using CsrMatrix.
 * Partitions independent matrix-vector products across CPU worker threads.
 */
void parallel_batched_spmv(const numerics::CsrMatrix& A, const double* batch_x, double* batch_y,
                           std::size_t batch_size, const BatchedSpmvOptions& options = {},
                           BatchedSpmvTelemetry* telemetry = nullptr,
                           ThreadPool* custom_pool = nullptr);

/**
 * @brief Multicore parallel batched SpMV execution using CscMatrix.
 */
void parallel_batched_spmv(const numerics::CscMatrix& A, const double* batch_x, double* batch_y,
                           std::size_t batch_size, const BatchedSpmvOptions& options = {},
                           BatchedSpmvTelemetry* telemetry = nullptr,
                           ThreadPool* custom_pool = nullptr);

/**
 * @brief Multicore parallel batched SpMV execution using STL vector of vectors.
 */
void parallel_batched_spmv(const numerics::CsrMatrix& A,
                           const std::vector<std::vector<double>>& batch_x,
                           std::vector<std::vector<double>>& batch_y,
                           const BatchedSpmvOptions& options = {},
                           BatchedSpmvTelemetry* telemetry = nullptr,
                           ThreadPool* custom_pool = nullptr);

/**
 * @brief Raw size_t pointer parallel batched SpMV execution.
 */
void parallel_batched_spmv_raw(std::size_t batch_size, std::size_t rows, std::size_t cols,
                               const double* values, const std::size_t* row_ptr,
                               const std::size_t* col_indices, const double* batch_x,
                               double* batch_y, const BatchedSpmvOptions& options = {},
                               BatchedSpmvTelemetry* telemetry = nullptr,
                               ThreadPool* custom_pool = nullptr);

/**
 * @brief Raw int pointer parallel batched SpMV execution.
 */
void parallel_batched_spmv_raw(std::size_t batch_size, std::size_t rows, std::size_t cols,
                               const double* values, const int* row_ptr, const int* col_indices,
                               const double* batch_x, double* batch_y,
                               const BatchedSpmvOptions& options = {},
                               BatchedSpmvTelemetry* telemetry = nullptr,
                               ThreadPool* custom_pool = nullptr);

/**
 * @brief Compares serial and parallel results element by element.
 * Reports max absolute error, max relative error, and details of any mismatch.
 */
[[nodiscard]] SpmvComparisonResult compare_spmv_results(std::size_t batch_size, std::size_t out_dim,
                                                        const double* serial_y,
                                                        const double* parallel_y,
                                                        double abs_tol = 1e-12,
                                                        double rel_tol = 1e-10);

}  // namespace kavach::backend::cpu
