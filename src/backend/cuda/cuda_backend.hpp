#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "backend/cpu/cpu_backend.hpp"
#include "backend/cuda/cuda_device.hpp"
#include "backend/execution_backend.hpp"
#include "numerics/sparse_matrix.hpp"

namespace kavach::backend::cuda {

/**
 * @brief Options configuring the CUDA execution backend.
 */
struct CudaBackendOptions {
    int device_id{0};        ///< CUDA GPU device index
    bool strict_gpu{false};  ///< If true and GPU is unavailable, throw instead of falling back
    bool force_cpu_fallback{false};  ///< Force CPU fallback (implements --no-gpu)
    std::size_t cpu_fallback_workers{
        0};  ///< Worker threads to use if CPU fallback occurs (0 = auto)
};

/**
 * @brief Comprehensive execution telemetry for CUDA operations.
 * Fulfills Blueprint Section 15 and Section 22.
 */
struct CudaTelemetry {
    bool gpu_available{false};              ///< Whether CUDA hardware was available
    bool fallback_occurred{false};          ///< Whether execution fell back to CPU
    std::string fallback_reason;            ///< Diagnostic explanation if fallback occurred
    int device_id{0};                       ///< Device ordinal
    std::string device_name{"None"};        ///< GPU device model name
    std::string compute_capability{"0.0"};  ///< CUDA compute capability (e.g. "8.6")
    std::string kernel_name{"None"};        ///< Active CUDA kernel name
    std::size_t rows{0};                    ///< Matrix row count
    std::size_t cols{0};                    ///< Matrix col count
    std::size_t nonzeros{0};                ///< Matrix nonzeros count
    std::size_t batch_size{0};              ///< Batch vector count
    std::size_t bytes_transferred{0};       ///< Total H2D + D2H bytes transferred
    double h2d_time_seconds{0.0};           ///< Host-to-Device memory transfer time
    double kernel_time_seconds{0.0};        ///< Pure GPU kernel execution time
    double d2h_time_seconds{0.0};           ///< Device-to-Host memory transfer time
    double total_gpu_time_seconds{0.0};  ///< Transfer-inclusive total GPU time (H2D + Kernel + D2H)
    double cpu_reference_time_seconds{0.0};  ///< CPU reference execution time
    double transfer_inclusive_speedup{1.0};  ///< CPU_time / total_gpu_time_seconds
    double kernel_only_speedup{1.0};  ///< CPU_time / kernel_time_seconds (for diagnostic only)
    BackendType backend_used{BackendType::CPU_SEQUENTIAL};
    DispatchDecision
        dispatch_decision;  ///< Explicit dispatch tracking (requested, selected, executed)
};

/**
 * @brief CUDA Execution Backend for batched sparse and vector operations.
 * Implements ExecutionBackend interface. CPU fallback is mandatory and default.
 */
class CudaBackend : public ExecutionBackend {
public:
    explicit CudaBackend(int device_id = 0);
    explicit CudaBackend(CudaBackendOptions options);
    ~CudaBackend() override = default;

    [[nodiscard]] BackendType type() const noexcept override { return BackendType::CUDA_GPU; }

    /**
     * @brief Checks if CUDA GPU execution is available and enabled.
     */
    [[nodiscard]] bool is_available() const noexcept override;

    [[nodiscard]] const CudaTelemetry& last_telemetry() const noexcept { return last_telemetry_; }
    void set_options(const CudaBackendOptions& options) { options_ = options; }
    [[nodiscard]] const CudaBackendOptions& options() const noexcept { return options_; }
    void set_device(int device_id);
    [[nodiscard]] int device_id() const noexcept { return options_.device_id; }

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
    CudaBackendOptions options_;
    mutable CudaTelemetry last_telemetry_;
    mutable cpu::CpuMulticoreBackend cpu_fallback_backend_;
};

}  // namespace kavach::backend::cuda
