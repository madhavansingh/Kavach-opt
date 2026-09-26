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

    void batched_spmv_transpose(const numerics::CsrMatrix& A, const double* batch_x,
                                double* batch_y, std::size_t batch_size,
                                const BatchedSpmvOptions& options = {}) override;

    void batched_csc_spmv(const numerics::CscMatrix& A, const double* batch_x, double* batch_y,
                          std::size_t batch_size, const BatchedSpmvOptions& options = {}) override;

    void batched_axpby(std::size_t batch_size, std::size_t n, double alpha, const double* batch_x,
                       double beta, double* batch_y) override;

    void batched_dot(std::size_t batch_size, std::size_t n, const double* batch_x,
                     const double* batch_y, double* results) override;

    void batched_l2_norm(std::size_t batch_size, std::size_t n, const double* batch_x,
                         double* results) override;

    void batched_linf_norm(std::size_t batch_size, std::size_t n, const double* batch_x,
                           double* results) override;
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

    void batched_spmv_transpose(const numerics::CsrMatrix& A, const double* batch_x,
                                double* batch_y, std::size_t batch_size,
                                const BatchedSpmvOptions& options = {}) override;

    void batched_csc_spmv(const numerics::CscMatrix& A, const double* batch_x, double* batch_y,
                          std::size_t batch_size, const BatchedSpmvOptions& options = {}) override;

    void batched_axpby(std::size_t batch_size, std::size_t n, double alpha, const double* batch_x,
                       double beta, double* batch_y) override;

    void batched_dot(std::size_t batch_size, std::size_t n, const double* batch_x,
                     const double* batch_y, double* results) override;

    void batched_l2_norm(std::size_t batch_size, std::size_t n, const double* batch_x,
                         double* results) override;

    void batched_linf_norm(std::size_t batch_size, std::size_t n, const double* batch_x,
                           double* results) override;

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
 * @brief Reference serial batched Transpose SpMV execution: Y_k = A^T * X_k.
 */
void serial_batched_spmv_transpose(const numerics::CsrMatrix& A, const double* batch_x,
                                   double* batch_y, std::size_t batch_size);

/**
 * @brief Multicore parallel batched Transpose SpMV execution using CsrMatrix.
 */
void parallel_batched_spmv_transpose(const numerics::CsrMatrix& A, const double* batch_x,
                                     double* batch_y, std::size_t batch_size,
                                     const BatchedSpmvOptions& options = {},
                                     BatchedSpmvTelemetry* telemetry = nullptr,
                                     ThreadPool* custom_pool = nullptr);

/**
 * @brief Reference serial batched SpMV execution using CscMatrix: Y_k = A * X_k.
 */
void serial_batched_csc_spmv(const numerics::CscMatrix& A, const double* batch_x, double* batch_y,
                             std::size_t batch_size);

/**
 * @brief Serial batched vector AXPBY: y[k] = alpha * x[k] + beta * y[k].
 */
void serial_batched_axpby(std::size_t batch_size, std::size_t n, double alpha,
                          const double* batch_x, double beta, double* batch_y);

/**
 * @brief Multicore parallel batched vector AXPBY.
 */
void parallel_batched_axpby(std::size_t batch_size, std::size_t n, double alpha,
                            const double* batch_x, double beta, double* batch_y,
                            ThreadPool* custom_pool = nullptr);

/**
 * @brief Serial batched dot product: results[k] = x[k]^T * y[k].
 */
void serial_batched_dot(std::size_t batch_size, std::size_t n, const double* batch_x,
                        const double* batch_y, double* results);

/**
 * @brief Multicore parallel batched dot product.
 */
void parallel_batched_dot(std::size_t batch_size, std::size_t n, const double* batch_x,
                          const double* batch_y, double* results,
                          ThreadPool* custom_pool = nullptr);

/**
 * @brief Serial batched L2 norm: results[k] = ||x[k]||_2.
 */
void serial_batched_l2_norm(std::size_t batch_size, std::size_t n, const double* batch_x,
                            double* results);

/**
 * @brief Multicore parallel batched L2 norm.
 */
void parallel_batched_l2_norm(std::size_t batch_size, std::size_t n, const double* batch_x,
                              double* results, ThreadPool* custom_pool = nullptr);

/**
 * @brief Serial batched Linf norm: results[k] = ||x[k]||_inf.
 */
void serial_batched_linf_norm(std::size_t batch_size, std::size_t n, const double* batch_x,
                              double* results);

/**
 * @brief Multicore parallel batched Linf norm.
 */
void parallel_batched_linf_norm(std::size_t batch_size, std::size_t n, const double* batch_x,
                                double* results, ThreadPool* custom_pool = nullptr);

/**
 * @brief Compares serial and parallel results element by element.
 * Reports max absolute error, max relative error, and details of any mismatch.
 */
[[nodiscard]] SpmvComparisonResult compare_spmv_results(std::size_t batch_size, std::size_t out_dim,
                                                        const double* serial_y,
                                                        const double* parallel_y,
                                                        double abs_tol = 1e-12,
                                                        double rel_tol = 1e-10);

/**
 * @brief Compares contiguous vector results.
 */
[[nodiscard]] SpmvComparisonResult compare_vector_results(std::size_t total_elements,
                                                          const double* ref_vec,
                                                          const double* test_vec,
                                                          double abs_tol = 1e-12,
                                                          double rel_tol = 1e-10);

/**
 * @brief Compares scalar results across batch.
 */
[[nodiscard]] SpmvComparisonResult compare_scalar_results(std::size_t count,
                                                          const double* ref_scalars,
                                                          const double* test_scalars,
                                                          double abs_tol = 1e-12,
                                                          double rel_tol = 1e-10);

}  // namespace kavach::backend::cpu
