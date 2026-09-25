#include "backend/cuda/cuda_backend.hpp"

#include <chrono>
#include <cmath>

#include "backend/cuda/cuda_spmv.cuh"
#include "core/error.hpp"

namespace kavach::backend::cuda {

CudaBackend::CudaBackend(int device_id)
    : options_{device_id, false, false, 0}, cpu_fallback_backend_(0) {}

CudaBackend::CudaBackend(CudaBackendOptions options)
    : options_(options), cpu_fallback_backend_(options.cpu_fallback_workers) {}

bool CudaBackend::is_available() const noexcept {
    if (options_.force_cpu_fallback) {
        return false;
    }
    return is_cuda_available();
}

void CudaBackend::set_device(int device_id) {
    if (is_cuda_available()) {
        set_cuda_device(device_id);
    }
    options_.device_id = device_id;
}

void CudaBackend::batched_spmv(std::size_t batch_size, std::size_t rows, std::size_t cols,
                               const double* values, const int* row_ptr, const int* col_indices,
                               const double* batch_x, double* batch_y) {
    if (batch_size == 0) {
        last_telemetry_ = {};
        last_telemetry_.rows = rows;
        last_telemetry_.cols = cols;
        return;
    }
    if (batch_x == nullptr || batch_y == nullptr || values == nullptr || row_ptr == nullptr ||
        col_indices == nullptr) {
        throw core::InputException("Null pointer passed to CudaBackend::batched_spmv");
    }

    const std::size_t nnz = static_cast<std::size_t>(row_ptr[rows]);
    const bool can_run_gpu = (!options_.force_cpu_fallback) && is_cuda_available();

    if (!can_run_gpu) {
        if (options_.strict_gpu) {
            std::string reason = options_.force_cpu_fallback
                                     ? "GPU disabled via force_cpu_fallback (--no-gpu)"
                                     : "CUDA unavailable or binary compiled in CPU-only mode";
            throw core::NumericalException("Strict GPU execution requested but unavailable: " +
                                           reason);
        }

        // Mandatory CPU Fallback
        const auto t0 = std::chrono::steady_clock::now();
        cpu_fallback_backend_.batched_spmv(batch_size, rows, cols, values, row_ptr, col_indices,
                                           batch_x, batch_y);
        const auto t1 = std::chrono::steady_clock::now();
        const double cpu_sec = std::chrono::duration<double>(t1 - t0).count();

        last_telemetry_ = {};
        last_telemetry_.gpu_available = false;
        last_telemetry_.fallback_occurred = true;
        last_telemetry_.fallback_reason =
            options_.force_cpu_fallback
                ? "Explicit CPU fallback requested (--no-gpu)"
                : "CUDA runtime unavailable; automatically fell back to CPU";
        last_telemetry_.rows = rows;
        last_telemetry_.cols = cols;
        last_telemetry_.nonzeros = nnz;
        last_telemetry_.batch_size = batch_size;
        last_telemetry_.cpu_reference_time_seconds = cpu_sec;
        last_telemetry_.total_gpu_time_seconds = cpu_sec;
        last_telemetry_.transfer_inclusive_speedup = 1.0;
        last_telemetry_.kernel_only_speedup = 1.0;
        last_telemetry_.backend_used = cpu_fallback_backend_.type();
        return;
    }

#ifdef KAVACH_USE_CUDA
    try {
        auto prop = get_cuda_device_properties(options_.device_id);
        last_telemetry_.device_id = options_.device_id;
        last_telemetry_.device_name = prop.name;
        last_telemetry_.compute_capability =
            std::to_string(prop.major) + "." + std::to_string(prop.minor);

        CudaSpmvExecutionTimes times;
        cuda_batched_csr_spmv_int(batch_size, rows, cols, nnz, values, row_ptr, col_indices,
                                  batch_x, batch_y, options_.device_id, &times);

        last_telemetry_.gpu_available = true;
        last_telemetry_.fallback_occurred = false;
        last_telemetry_.fallback_reason = "None";
        last_telemetry_.rows = rows;
        last_telemetry_.cols = cols;
        last_telemetry_.nonzeros = nnz;
        last_telemetry_.batch_size = batch_size;
        last_telemetry_.h2d_time_seconds = times.h2d_time_seconds;
        last_telemetry_.kernel_time_seconds = times.kernel_time_seconds;
        last_telemetry_.d2h_time_seconds = times.d2h_time_seconds;
        last_telemetry_.total_gpu_time_seconds = times.total_gpu_time_seconds;
        last_telemetry_.backend_used = BackendType::CUDA_GPU;

        // Measure CPU reference for honest speedup comparison
        std::vector<double> ref_y(batch_size * rows, 0.0);
        const auto t0 = std::chrono::steady_clock::now();
        cpu_fallback_backend_.batched_spmv(batch_size, rows, cols, values, row_ptr, col_indices,
                                           batch_x, ref_y.data());
        const auto t1 = std::chrono::steady_clock::now();
        last_telemetry_.cpu_reference_time_seconds = std::chrono::duration<double>(t1 - t0).count();

        if (last_telemetry_.total_gpu_time_seconds > 0.0) {
            last_telemetry_.transfer_inclusive_speedup =
                last_telemetry_.cpu_reference_time_seconds / last_telemetry_.total_gpu_time_seconds;
        }
        if (last_telemetry_.kernel_time_seconds > 0.0) {
            last_telemetry_.kernel_only_speedup =
                last_telemetry_.cpu_reference_time_seconds / last_telemetry_.kernel_time_seconds;
        }
    } catch (const std::exception& e) {
        if (options_.strict_gpu) {
            throw;
        }
        // Graceful fallback to CPU
        cpu_fallback_backend_.batched_spmv(batch_size, rows, cols, values, row_ptr, col_indices,
                                           batch_x, batch_y);
        last_telemetry_.fallback_occurred = true;
        last_telemetry_.fallback_reason = "CUDA runtime failure: " + std::string(e.what());
        last_telemetry_.backend_used = cpu_fallback_backend_.type();
    }
#endif
}

void CudaBackend::batched_spmv(std::size_t batch_size, std::size_t rows, std::size_t cols,
                               const double* values, const std::size_t* row_ptr,
                               const std::size_t* col_indices, const double* batch_x,
                               double* batch_y) {
    if (batch_size == 0) {
        last_telemetry_ = {};
        last_telemetry_.rows = rows;
        last_telemetry_.cols = cols;
        return;
    }
    if (batch_x == nullptr || batch_y == nullptr || values == nullptr || row_ptr == nullptr ||
        col_indices == nullptr) {
        throw core::InputException("Null pointer passed to CudaBackend::batched_spmv");
    }

    const std::size_t nnz = row_ptr[rows];
    const bool can_run_gpu = (!options_.force_cpu_fallback) && is_cuda_available();

    if (!can_run_gpu) {
        if (options_.strict_gpu) {
            std::string reason = options_.force_cpu_fallback
                                     ? "GPU disabled via force_cpu_fallback (--no-gpu)"
                                     : "CUDA unavailable or binary compiled in CPU-only mode";
            throw core::NumericalException("Strict GPU execution requested but unavailable: " +
                                           reason);
        }

        const auto t0 = std::chrono::steady_clock::now();
        cpu_fallback_backend_.batched_spmv(batch_size, rows, cols, values, row_ptr, col_indices,
                                           batch_x, batch_y);
        const auto t1 = std::chrono::steady_clock::now();
        const double cpu_sec = std::chrono::duration<double>(t1 - t0).count();

        last_telemetry_ = {};
        last_telemetry_.gpu_available = false;
        last_telemetry_.fallback_occurred = true;
        last_telemetry_.fallback_reason =
            options_.force_cpu_fallback
                ? "Explicit CPU fallback requested (--no-gpu)"
                : "CUDA runtime unavailable; automatically fell back to CPU";
        last_telemetry_.rows = rows;
        last_telemetry_.cols = cols;
        last_telemetry_.nonzeros = nnz;
        last_telemetry_.batch_size = batch_size;
        last_telemetry_.cpu_reference_time_seconds = cpu_sec;
        last_telemetry_.total_gpu_time_seconds = cpu_sec;
        last_telemetry_.transfer_inclusive_speedup = 1.0;
        last_telemetry_.kernel_only_speedup = 1.0;
        last_telemetry_.backend_used = cpu_fallback_backend_.type();
        return;
    }

#ifdef KAVACH_USE_CUDA
    try {
        auto prop = get_cuda_device_properties(options_.device_id);
        last_telemetry_.device_id = options_.device_id;
        last_telemetry_.device_name = prop.name;
        last_telemetry_.compute_capability =
            std::to_string(prop.major) + "." + std::to_string(prop.minor);

        CudaSpmvExecutionTimes times;
        cuda_batched_csr_spmv(batch_size, rows, cols, nnz, values, row_ptr, col_indices, batch_x,
                              batch_y, options_.device_id, &times);

        last_telemetry_.gpu_available = true;
        last_telemetry_.fallback_occurred = false;
        last_telemetry_.fallback_reason = "None";
        last_telemetry_.rows = rows;
        last_telemetry_.cols = cols;
        last_telemetry_.nonzeros = nnz;
        last_telemetry_.batch_size = batch_size;
        last_telemetry_.h2d_time_seconds = times.h2d_time_seconds;
        last_telemetry_.kernel_time_seconds = times.kernel_time_seconds;
        last_telemetry_.d2h_time_seconds = times.d2h_time_seconds;
        last_telemetry_.total_gpu_time_seconds = times.total_gpu_time_seconds;
        last_telemetry_.backend_used = BackendType::CUDA_GPU;

        std::vector<double> ref_y(batch_size * rows, 0.0);
        const auto t0 = std::chrono::steady_clock::now();
        cpu_fallback_backend_.batched_spmv(batch_size, rows, cols, values, row_ptr, col_indices,
                                           batch_x, ref_y.data());
        const auto t1 = std::chrono::steady_clock::now();
        last_telemetry_.cpu_reference_time_seconds = std::chrono::duration<double>(t1 - t0).count();

        if (last_telemetry_.total_gpu_time_seconds > 0.0) {
            last_telemetry_.transfer_inclusive_speedup =
                last_telemetry_.cpu_reference_time_seconds / last_telemetry_.total_gpu_time_seconds;
        }
        if (last_telemetry_.kernel_time_seconds > 0.0) {
            last_telemetry_.kernel_only_speedup =
                last_telemetry_.cpu_reference_time_seconds / last_telemetry_.kernel_time_seconds;
        }
    } catch (const std::exception& e) {
        if (options_.strict_gpu) {
            throw;
        }
        cpu_fallback_backend_.batched_spmv(batch_size, rows, cols, values, row_ptr, col_indices,
                                           batch_x, batch_y);
        last_telemetry_.fallback_occurred = true;
        last_telemetry_.fallback_reason = "CUDA runtime failure: " + std::string(e.what());
        last_telemetry_.backend_used = cpu_fallback_backend_.type();
    }
#endif
}

void CudaBackend::batched_spmv(const numerics::CsrMatrix& A, const double* batch_x, double* batch_y,
                               std::size_t batch_size, const BatchedSpmvOptions&) {
    batched_spmv(batch_size, A.rows(), A.cols(), A.values().data(), A.row_ptr().data(),
                 A.col_idx().data(), batch_x, batch_y);
}

}  // namespace kavach::backend::cuda
