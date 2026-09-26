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

void CudaBackend::batched_spmv_transpose(const numerics::CsrMatrix& A, const double* batch_x,
                                         double* batch_y, std::size_t batch_size,
                                         const BatchedSpmvOptions&) {
    if (batch_size == 0) {
        last_telemetry_ = {};
        last_telemetry_.rows = A.cols();
        last_telemetry_.cols = A.rows();
        return;
    }
    if (batch_x == nullptr || batch_y == nullptr) {
        throw core::InputException("Null pointer passed to CudaBackend::batched_spmv_transpose");
    }

    const std::size_t rows = A.rows();
    const std::size_t cols = A.cols();
    const std::size_t nnz = A.nonzeros();
    const std::size_t est_bytes = estimate_spmv_memory_bytes(batch_size, rows, cols, nnz);
    const bool mem_ok = check_cuda_memory_sufficient(est_bytes);
    const bool can_run_gpu = (!options_.force_cpu_fallback) && is_cuda_available() && mem_ok;

    if (!can_run_gpu) {
        if (options_.strict_gpu) {
            std::string reason =
                options_.force_cpu_fallback
                    ? "GPU disabled via force_cpu_fallback (--no-gpu)"
                    : (!is_cuda_available() ? "CUDA unavailable or binary compiled in CPU-only mode"
                                            : "Insufficient CUDA GPU memory for transpose SpMV");
            throw core::NumericalException("Strict GPU execution requested but unavailable: " +
                                           reason);
        }

        const auto t0 = std::chrono::steady_clock::now();
        cpu_fallback_backend_.batched_spmv_transpose(A, batch_x, batch_y, batch_size);
        const auto t1 = std::chrono::steady_clock::now();
        const double cpu_sec = std::chrono::duration<double>(t1 - t0).count();

        last_telemetry_ = {};
        last_telemetry_.gpu_available = false;
        last_telemetry_.fallback_occurred = true;
        last_telemetry_.fallback_reason =
            options_.force_cpu_fallback
                ? "Explicit CPU fallback requested (--no-gpu)"
                : (!is_cuda_available() ? "CUDA runtime unavailable; automatically fell back to CPU"
                                        : "Insufficient GPU memory; fell back to CPU");
        last_telemetry_.kernel_name = "cuda_batched_csr_transpose_spmv";
        last_telemetry_.rows = cols;
        last_telemetry_.cols = rows;
        last_telemetry_.nonzeros = nnz;
        last_telemetry_.batch_size = batch_size;
        last_telemetry_.bytes_transferred = 0;
        last_telemetry_.cpu_reference_time_seconds = cpu_sec;
        last_telemetry_.total_gpu_time_seconds = cpu_sec;
        last_telemetry_.transfer_inclusive_speedup = 1.0;
        last_telemetry_.kernel_only_speedup = 1.0;
        last_telemetry_.backend_used = cpu_fallback_backend_.type();
        last_telemetry_.dispatch_decision = {BackendType::CUDA_GPU, cpu_fallback_backend_.type(),
                                             cpu_fallback_backend_.type(),
                                             last_telemetry_.fallback_reason};
        return;
    }

#ifdef KAVACH_USE_CUDA
    try {
        auto prop = get_cuda_device_properties(options_.device_id);
        last_telemetry_.device_id = options_.device_id;
        last_telemetry_.device_name = prop.name;
        last_telemetry_.compute_capability =
            std::to_string(prop.major) + "." + std::to_string(prop.minor);
        last_telemetry_.kernel_name = "cuda_batched_csr_transpose_spmv";

        CudaSpmvExecutionTimes times;
        cuda_batched_csr_transpose_spmv(batch_size, rows, cols, nnz, A.values().data(),
                                        A.row_ptr().data(), A.col_idx().data(), batch_x, batch_y,
                                        options_.device_id, &times);

        last_telemetry_.gpu_available = true;
        last_telemetry_.fallback_occurred = false;
        last_telemetry_.fallback_reason = "None";
        last_telemetry_.rows = cols;
        last_telemetry_.cols = rows;
        last_telemetry_.nonzeros = nnz;
        last_telemetry_.batch_size = batch_size;
        last_telemetry_.bytes_transferred = est_bytes;
        last_telemetry_.h2d_time_seconds = times.h2d_time_seconds;
        last_telemetry_.kernel_time_seconds = times.kernel_time_seconds;
        last_telemetry_.d2h_time_seconds = times.d2h_time_seconds;
        last_telemetry_.total_gpu_time_seconds = times.total_gpu_time_seconds;
        last_telemetry_.backend_used = BackendType::CUDA_GPU;
        last_telemetry_.dispatch_decision = {BackendType::CUDA_GPU, BackendType::CUDA_GPU,
                                             BackendType::CUDA_GPU,
                                             "CUDA GPU transpose SpMV executed"};
    } catch (const std::exception& e) {
        if (options_.strict_gpu) {
            throw;
        }
        cpu_fallback_backend_.batched_spmv_transpose(A, batch_x, batch_y, batch_size);
        last_telemetry_.fallback_occurred = true;
        last_telemetry_.fallback_reason = "CUDA runtime failure: " + std::string(e.what());
        last_telemetry_.backend_used = cpu_fallback_backend_.type();
        last_telemetry_.dispatch_decision = {BackendType::CUDA_GPU, BackendType::CUDA_GPU,
                                             cpu_fallback_backend_.type(),
                                             last_telemetry_.fallback_reason};
    }
#endif
}

void CudaBackend::batched_csc_spmv(const numerics::CscMatrix& A, const double* batch_x,
                                   double* batch_y, std::size_t batch_size,
                                   const BatchedSpmvOptions&) {
    if (batch_size == 0) {
        last_telemetry_ = {};
        last_telemetry_.rows = A.rows();
        last_telemetry_.cols = A.cols();
        return;
    }
    if (batch_x == nullptr || batch_y == nullptr) {
        throw core::InputException("Null pointer passed to CudaBackend::batched_csc_spmv");
    }

    const std::size_t rows = A.rows();
    const std::size_t cols = A.cols();
    const std::size_t nnz = A.nonzeros();
    const std::size_t est_bytes = estimate_spmv_memory_bytes(batch_size, rows, cols, nnz);
    const bool mem_ok = check_cuda_memory_sufficient(est_bytes);
    const bool can_run_gpu = (!options_.force_cpu_fallback) && is_cuda_available() && mem_ok;

    if (!can_run_gpu) {
        if (options_.strict_gpu) {
            std::string reason =
                options_.force_cpu_fallback
                    ? "GPU disabled via force_cpu_fallback (--no-gpu)"
                    : (!is_cuda_available() ? "CUDA unavailable or binary compiled in CPU-only mode"
                                            : "Insufficient CUDA GPU memory for CSC SpMV");
            throw core::NumericalException("Strict GPU execution requested but unavailable: " +
                                           reason);
        }

        const auto t0 = std::chrono::steady_clock::now();
        cpu_fallback_backend_.batched_csc_spmv(A, batch_x, batch_y, batch_size);
        const auto t1 = std::chrono::steady_clock::now();
        const double cpu_sec = std::chrono::duration<double>(t1 - t0).count();

        last_telemetry_ = {};
        last_telemetry_.gpu_available = false;
        last_telemetry_.fallback_occurred = true;
        last_telemetry_.fallback_reason =
            options_.force_cpu_fallback
                ? "Explicit CPU fallback requested (--no-gpu)"
                : (!is_cuda_available() ? "CUDA runtime unavailable; automatically fell back to CPU"
                                        : "Insufficient GPU memory; fell back to CPU");
        last_telemetry_.kernel_name = "cuda_batched_csc_spmv";
        last_telemetry_.rows = rows;
        last_telemetry_.cols = cols;
        last_telemetry_.nonzeros = nnz;
        last_telemetry_.batch_size = batch_size;
        last_telemetry_.bytes_transferred = 0;
        last_telemetry_.cpu_reference_time_seconds = cpu_sec;
        last_telemetry_.total_gpu_time_seconds = cpu_sec;
        last_telemetry_.transfer_inclusive_speedup = 1.0;
        last_telemetry_.kernel_only_speedup = 1.0;
        last_telemetry_.backend_used = cpu_fallback_backend_.type();
        last_telemetry_.dispatch_decision = {BackendType::CUDA_GPU, cpu_fallback_backend_.type(),
                                             cpu_fallback_backend_.type(),
                                             last_telemetry_.fallback_reason};
        return;
    }

#ifdef KAVACH_USE_CUDA
    try {
        auto prop = get_cuda_device_properties(options_.device_id);
        last_telemetry_.device_id = options_.device_id;
        last_telemetry_.device_name = prop.name;
        last_telemetry_.compute_capability =
            std::to_string(prop.major) + "." + std::to_string(prop.minor);
        last_telemetry_.kernel_name = "cuda_batched_csc_spmv";

        CudaSpmvExecutionTimes times;
        cuda_batched_csc_spmv(batch_size, rows, cols, nnz, A.values().data(), A.col_ptr().data(),
                              A.row_idx().data(), batch_x, batch_y, options_.device_id, &times);

        last_telemetry_.gpu_available = true;
        last_telemetry_.fallback_occurred = false;
        last_telemetry_.fallback_reason = "None";
        last_telemetry_.rows = rows;
        last_telemetry_.cols = cols;
        last_telemetry_.nonzeros = nnz;
        last_telemetry_.batch_size = batch_size;
        last_telemetry_.bytes_transferred = est_bytes;
        last_telemetry_.h2d_time_seconds = times.h2d_time_seconds;
        last_telemetry_.kernel_time_seconds = times.kernel_time_seconds;
        last_telemetry_.d2h_time_seconds = times.d2h_time_seconds;
        last_telemetry_.total_gpu_time_seconds = times.total_gpu_time_seconds;
        last_telemetry_.backend_used = BackendType::CUDA_GPU;
        last_telemetry_.dispatch_decision = {BackendType::CUDA_GPU, BackendType::CUDA_GPU,
                                             BackendType::CUDA_GPU, "CUDA GPU CSC SpMV executed"};
    } catch (const std::exception& e) {
        if (options_.strict_gpu) {
            throw;
        }
        cpu_fallback_backend_.batched_csc_spmv(A, batch_x, batch_y, batch_size);
        last_telemetry_.fallback_occurred = true;
        last_telemetry_.fallback_reason = "CUDA runtime failure: " + std::string(e.what());
        last_telemetry_.backend_used = cpu_fallback_backend_.type();
        last_telemetry_.dispatch_decision = {BackendType::CUDA_GPU, BackendType::CUDA_GPU,
                                             cpu_fallback_backend_.type(),
                                             last_telemetry_.fallback_reason};
    }
#endif
}

void CudaBackend::batched_axpby(std::size_t batch_size, std::size_t n, double alpha,
                                const double* batch_x, double beta, double* batch_y) {
    if (batch_size == 0 || n == 0)
        return;
    if (batch_x == nullptr || batch_y == nullptr) {
        throw core::InputException("Null pointer passed to CudaBackend::batched_axpby");
    }

    const std::size_t est_bytes = estimate_vector_memory_bytes(batch_size, n, 2);
    const bool mem_ok = check_cuda_memory_sufficient(est_bytes);
    const bool can_run_gpu = (!options_.force_cpu_fallback) && is_cuda_available() && mem_ok;

    if (!can_run_gpu) {
        if (options_.strict_gpu) {
            throw core::NumericalException("Strict GPU execution requested but unavailable");
        }
        cpu_fallback_backend_.batched_axpby(batch_size, n, alpha, batch_x, beta, batch_y);
        last_telemetry_ = {};
        last_telemetry_.fallback_occurred = true;
        last_telemetry_.fallback_reason =
            options_.force_cpu_fallback ? "Explicit CPU fallback" : "CUDA unavailable";
        last_telemetry_.kernel_name = "cuda_batched_axpby";
        last_telemetry_.batch_size = batch_size;
        last_telemetry_.backend_used = cpu_fallback_backend_.type();
        last_telemetry_.dispatch_decision = {BackendType::CUDA_GPU, cpu_fallback_backend_.type(),
                                             cpu_fallback_backend_.type(),
                                             last_telemetry_.fallback_reason};
        return;
    }

#ifdef KAVACH_USE_CUDA
    try {
        CudaSpmvExecutionTimes times;
        cuda_batched_axpby(batch_size, n, alpha, batch_x, beta, batch_y, options_.device_id,
                           &times);
        last_telemetry_.gpu_available = true;
        last_telemetry_.fallback_occurred = false;
        last_telemetry_.kernel_name = "cuda_batched_axpby";
        last_telemetry_.batch_size = batch_size;
        last_telemetry_.bytes_transferred = est_bytes;
        last_telemetry_.h2d_time_seconds = times.h2d_time_seconds;
        last_telemetry_.kernel_time_seconds = times.kernel_time_seconds;
        last_telemetry_.d2h_time_seconds = times.d2h_time_seconds;
        last_telemetry_.total_gpu_time_seconds = times.total_gpu_time_seconds;
        last_telemetry_.backend_used = BackendType::CUDA_GPU;
        last_telemetry_.dispatch_decision = {BackendType::CUDA_GPU, BackendType::CUDA_GPU,
                                             BackendType::CUDA_GPU, "CUDA GPU AXPBY executed"};
    } catch (const std::exception& e) {
        if (options_.strict_gpu)
            throw;
        cpu_fallback_backend_.batched_axpby(batch_size, n, alpha, batch_x, beta, batch_y);
        last_telemetry_.fallback_occurred = true;
        last_telemetry_.fallback_reason = "CUDA runtime failure: " + std::string(e.what());
        last_telemetry_.backend_used = cpu_fallback_backend_.type();
    }
#endif
}

void CudaBackend::batched_dot(std::size_t batch_size, std::size_t n, const double* batch_x,
                              const double* batch_y, double* results) {
    if (batch_size == 0)
        return;
    if (results == nullptr) {
        throw core::InputException("Null pointer for results in CudaBackend::batched_dot");
    }

    const std::size_t est_bytes = estimate_vector_memory_bytes(batch_size, n, 2, batch_size);
    const bool mem_ok = check_cuda_memory_sufficient(est_bytes);
    const bool can_run_gpu = (!options_.force_cpu_fallback) && is_cuda_available() && mem_ok;

    if (!can_run_gpu) {
        if (options_.strict_gpu) {
            throw core::NumericalException("Strict GPU execution requested but unavailable");
        }
        cpu_fallback_backend_.batched_dot(batch_size, n, batch_x, batch_y, results);
        last_telemetry_ = {};
        last_telemetry_.fallback_occurred = true;
        last_telemetry_.fallback_reason =
            options_.force_cpu_fallback ? "Explicit CPU fallback" : "CUDA unavailable";
        last_telemetry_.kernel_name = "cuda_batched_dot";
        last_telemetry_.batch_size = batch_size;
        last_telemetry_.backend_used = cpu_fallback_backend_.type();
        last_telemetry_.dispatch_decision = {BackendType::CUDA_GPU, cpu_fallback_backend_.type(),
                                             cpu_fallback_backend_.type(),
                                             last_telemetry_.fallback_reason};
        return;
    }

#ifdef KAVACH_USE_CUDA
    try {
        CudaSpmvExecutionTimes times;
        cuda_batched_dot(batch_size, n, batch_x, batch_y, results, options_.device_id, &times);
        last_telemetry_.gpu_available = true;
        last_telemetry_.fallback_occurred = false;
        last_telemetry_.kernel_name = "cuda_batched_dot";
        last_telemetry_.batch_size = batch_size;
        last_telemetry_.bytes_transferred = est_bytes;
        last_telemetry_.h2d_time_seconds = times.h2d_time_seconds;
        last_telemetry_.kernel_time_seconds = times.kernel_time_seconds;
        last_telemetry_.d2h_time_seconds = times.d2h_time_seconds;
        last_telemetry_.total_gpu_time_seconds = times.total_gpu_time_seconds;
        last_telemetry_.backend_used = BackendType::CUDA_GPU;
        last_telemetry_.dispatch_decision = {BackendType::CUDA_GPU, BackendType::CUDA_GPU,
                                             BackendType::CUDA_GPU,
                                             "CUDA GPU dot product executed"};
    } catch (const std::exception& e) {
        if (options_.strict_gpu)
            throw;
        cpu_fallback_backend_.batched_dot(batch_size, n, batch_x, batch_y, results);
        last_telemetry_.fallback_occurred = true;
        last_telemetry_.fallback_reason = "CUDA runtime failure: " + std::string(e.what());
        last_telemetry_.backend_used = cpu_fallback_backend_.type();
    }
#endif
}

void CudaBackend::batched_l2_norm(std::size_t batch_size, std::size_t n, const double* batch_x,
                                  double* results) {
    if (batch_size == 0)
        return;
    if (results == nullptr) {
        throw core::InputException("Null pointer for results in CudaBackend::batched_l2_norm");
    }

    const std::size_t est_bytes = estimate_vector_memory_bytes(batch_size, n, 1, batch_size);
    const bool mem_ok = check_cuda_memory_sufficient(est_bytes);
    const bool can_run_gpu = (!options_.force_cpu_fallback) && is_cuda_available() && mem_ok;

    if (!can_run_gpu) {
        if (options_.strict_gpu) {
            throw core::NumericalException("Strict GPU execution requested but unavailable");
        }
        cpu_fallback_backend_.batched_l2_norm(batch_size, n, batch_x, results);
        last_telemetry_ = {};
        last_telemetry_.fallback_occurred = true;
        last_telemetry_.fallback_reason =
            options_.force_cpu_fallback ? "Explicit CPU fallback" : "CUDA unavailable";
        last_telemetry_.kernel_name = "cuda_batched_l2_norm";
        last_telemetry_.batch_size = batch_size;
        last_telemetry_.backend_used = cpu_fallback_backend_.type();
        last_telemetry_.dispatch_decision = {BackendType::CUDA_GPU, cpu_fallback_backend_.type(),
                                             cpu_fallback_backend_.type(),
                                             last_telemetry_.fallback_reason};
        return;
    }

#ifdef KAVACH_USE_CUDA
    try {
        CudaSpmvExecutionTimes times;
        cuda_batched_l2_norm(batch_size, n, batch_x, results, options_.device_id, &times);
        last_telemetry_.gpu_available = true;
        last_telemetry_.fallback_occurred = false;
        last_telemetry_.kernel_name = "cuda_batched_l2_norm";
        last_telemetry_.batch_size = batch_size;
        last_telemetry_.bytes_transferred = est_bytes;
        last_telemetry_.h2d_time_seconds = times.h2d_time_seconds;
        last_telemetry_.kernel_time_seconds = times.kernel_time_seconds;
        last_telemetry_.d2h_time_seconds = times.d2h_time_seconds;
        last_telemetry_.total_gpu_time_seconds = times.total_gpu_time_seconds;
        last_telemetry_.backend_used = BackendType::CUDA_GPU;
        last_telemetry_.dispatch_decision = {BackendType::CUDA_GPU, BackendType::CUDA_GPU,
                                             BackendType::CUDA_GPU, "CUDA GPU L2 norm executed"};
    } catch (const std::exception& e) {
        if (options_.strict_gpu)
            throw;
        cpu_fallback_backend_.batched_l2_norm(batch_size, n, batch_x, results);
        last_telemetry_.fallback_occurred = true;
        last_telemetry_.fallback_reason = "CUDA runtime failure: " + std::string(e.what());
        last_telemetry_.backend_used = cpu_fallback_backend_.type();
    }
#endif
}

void CudaBackend::batched_linf_norm(std::size_t batch_size, std::size_t n, const double* batch_x,
                                    double* results) {
    if (batch_size == 0)
        return;
    if (results == nullptr) {
        throw core::InputException("Null pointer for results in CudaBackend::batched_linf_norm");
    }

    const std::size_t est_bytes = estimate_vector_memory_bytes(batch_size, n, 1, batch_size);
    const bool mem_ok = check_cuda_memory_sufficient(est_bytes);
    const bool can_run_gpu = (!options_.force_cpu_fallback) && is_cuda_available() && mem_ok;

    if (!can_run_gpu) {
        if (options_.strict_gpu) {
            throw core::NumericalException("Strict GPU execution requested but unavailable");
        }
        cpu_fallback_backend_.batched_linf_norm(batch_size, n, batch_x, results);
        last_telemetry_ = {};
        last_telemetry_.fallback_occurred = true;
        last_telemetry_.fallback_reason =
            options_.force_cpu_fallback ? "Explicit CPU fallback" : "CUDA unavailable";
        last_telemetry_.kernel_name = "cuda_batched_linf_norm";
        last_telemetry_.batch_size = batch_size;
        last_telemetry_.backend_used = cpu_fallback_backend_.type();
        last_telemetry_.dispatch_decision = {BackendType::CUDA_GPU, cpu_fallback_backend_.type(),
                                             cpu_fallback_backend_.type(),
                                             last_telemetry_.fallback_reason};
        return;
    }

#ifdef KAVACH_USE_CUDA
    try {
        CudaSpmvExecutionTimes times;
        cuda_batched_linf_norm(batch_size, n, batch_x, results, options_.device_id, &times);
        last_telemetry_.gpu_available = true;
        last_telemetry_.fallback_occurred = false;
        last_telemetry_.kernel_name = "cuda_batched_linf_norm";
        last_telemetry_.batch_size = batch_size;
        last_telemetry_.bytes_transferred = est_bytes;
        last_telemetry_.h2d_time_seconds = times.h2d_time_seconds;
        last_telemetry_.kernel_time_seconds = times.kernel_time_seconds;
        last_telemetry_.d2h_time_seconds = times.d2h_time_seconds;
        last_telemetry_.total_gpu_time_seconds = times.total_gpu_time_seconds;
        last_telemetry_.backend_used = BackendType::CUDA_GPU;
        last_telemetry_.dispatch_decision = {BackendType::CUDA_GPU, BackendType::CUDA_GPU,
                                             BackendType::CUDA_GPU, "CUDA GPU Linf norm executed"};
    } catch (const std::exception& e) {
        if (options_.strict_gpu)
            throw;
        cpu_fallback_backend_.batched_linf_norm(batch_size, n, batch_x, results);
        last_telemetry_.fallback_occurred = true;
        last_telemetry_.fallback_reason = "CUDA runtime failure: " + std::string(e.what());
        last_telemetry_.backend_used = cpu_fallback_backend_.type();
    }
#endif
}

}  // namespace kavach::backend::cuda
