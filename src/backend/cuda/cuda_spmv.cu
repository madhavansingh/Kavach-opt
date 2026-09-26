#include "backend/cuda/cuda_spmv.cuh"
#include "core/error.hpp"

#ifdef KAVACH_USE_CUDA
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

namespace kavach::backend::cuda {

namespace {

// ============================================================================
// RAII CUDA Utilities
// ============================================================================

struct CudaStreamRAII {
    cudaStream_t stream{nullptr};
    CudaStreamRAII() {
        cudaError_t err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            throw core::NumericalException("cudaStreamCreate failed: " +
                                           std::string(cudaGetErrorString(err)));
        }
    }
    ~CudaStreamRAII() {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
    CudaStreamRAII(const CudaStreamRAII&) = delete;
    CudaStreamRAII& operator=(const CudaStreamRAII&) = delete;
};

struct CudaEventRAII {
    cudaEvent_t event{nullptr};
    CudaEventRAII() {
        cudaError_t err = cudaEventCreate(&event);
        if (err != cudaSuccess) {
            throw core::NumericalException("cudaEventCreate failed: " +
                                           std::string(cudaGetErrorString(err)));
        }
    }
    ~CudaEventRAII() {
        if (event) {
            cudaEventDestroy(event);
        }
    }
    void record(cudaStream_t s) {
        cudaError_t err = cudaEventRecord(event, s);
        if (err != cudaSuccess) {
            throw core::NumericalException("cudaEventRecord failed: " +
                                           std::string(cudaGetErrorString(err)));
        }
    }
    static double elapsed_sec(const CudaEventRAII& start, const CudaEventRAII& stop) {
        float ms = 0.0f;
        cudaError_t err = cudaEventElapsedTime(&ms, start.event, stop.event);
        if (err != cudaSuccess) {
            throw core::NumericalException("cudaEventElapsedTime failed: " +
                                           std::string(cudaGetErrorString(err)));
        }
        return static_cast<double>(ms) * 1e-3;
    }
    CudaEventRAII(const CudaEventRAII&) = delete;
    CudaEventRAII& operator=(const CudaEventRAII&) = delete;
};

template <typename T>
class CudaDeviceBufferRAII {
public:
    explicit CudaDeviceBufferRAII(std::size_t count) : count_(count) {
        if (count_ == 0)
            return;
        cudaError_t err = cudaMalloc(&ptr_, count_ * sizeof(T));
        if (err != cudaSuccess) {
            throw core::NumericalException("cudaMalloc failed for " + std::to_string(count_) +
                                           " elements: " + std::string(cudaGetErrorString(err)));
        }
    }

    ~CudaDeviceBufferRAII() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    [[nodiscard]] T* get() noexcept { return ptr_; }
    [[nodiscard]] const T* get() const noexcept { return ptr_; }

    void upload_async(const T* host_src, std::size_t count, cudaStream_t stream) {
        if (count == 0)
            return;
        cudaError_t err =
            cudaMemcpyAsync(ptr_, host_src, count * sizeof(T), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            throw core::NumericalException("cudaMemcpyAsync (H2D) failed: " +
                                           std::string(cudaGetErrorString(err)));
        }
    }

    void download_async(T* host_dst, std::size_t count, cudaStream_t stream) const {
        if (count == 0)
            return;
        cudaError_t err =
            cudaMemcpyAsync(host_dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            throw core::NumericalException("cudaMemcpyAsync (D2H) failed: " +
                                           std::string(cudaGetErrorString(err)));
        }
    }

    CudaDeviceBufferRAII(const CudaDeviceBufferRAII&) = delete;
    CudaDeviceBufferRAII& operator=(const CudaDeviceBufferRAII&) = delete;

private:
    T* ptr_{nullptr};
    std::size_t count_{0};
};

// ============================================================================
// Custom Production CUDA Kernels (Sovereign, Zero-External-Dependency)
// ============================================================================

__global__ void batched_csr_spmv_kernel_size_t(std::size_t batch_size, std::size_t rows,
                                               std::size_t cols, const double* __restrict__ values,
                                               const std::size_t* __restrict__ row_ptr,
                                               const std::size_t* __restrict__ col_idx,
                                               const double* __restrict__ batch_x,
                                               double* __restrict__ batch_y) {
    const std::size_t global_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total_items = batch_size * rows;
    if (global_idx >= total_items) {
        return;
    }

    const std::size_t k = global_idx / rows;  // Batch vector index
    const std::size_t r = global_idx % rows;  // Row index

    const std::size_t start = row_ptr[r];
    const std::size_t end = row_ptr[r + 1];

    const double* x_k = batch_x + (k * cols);
    double sum = 0.0;

    for (std::size_t j = start; j < end; ++j) {
        sum += values[j] * x_k[col_idx[j]];
    }

    batch_y[k * rows + r] = sum;
}

__global__ void batched_csr_spmv_kernel_int(std::size_t batch_size, std::size_t rows,
                                            std::size_t cols, const double* __restrict__ values,
                                            const int* __restrict__ row_ptr,
                                            const int* __restrict__ col_idx,
                                            const double* __restrict__ batch_x,
                                            double* __restrict__ batch_y) {
    const std::size_t global_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total_items = batch_size * rows;
    if (global_idx >= total_items) {
        return;
    }

    const std::size_t k = global_idx / rows;
    const std::size_t r = global_idx % rows;

    const int start = row_ptr[r];
    const int end = row_ptr[r + 1];

    const double* x_k = batch_x + (k * cols);
    double sum = 0.0;

    for (int j = start; j < end; ++j) {
        sum += values[j] * x_k[col_idx[j]];
    }

    batch_y[k * rows + r] = sum;
}

__global__ void batched_csr_transpose_spmv_kernel(
    std::size_t batch_size, std::size_t rows, std::size_t cols, const double* __restrict__ values,
    const std::size_t* __restrict__ row_ptr, const std::size_t* __restrict__ col_idx,
    const double* __restrict__ batch_x, double* __restrict__ batch_y) {
    const std::size_t global_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total_items = batch_size * rows;
    if (global_idx >= total_items) {
        return;
    }

    const std::size_t k = global_idx / rows;
    const std::size_t r = global_idx % rows;

    const std::size_t start = row_ptr[r];
    const std::size_t end = row_ptr[r + 1];
    const double x_val = batch_x[k * rows + r];
    double* y_k = batch_y + (k * cols);

    for (std::size_t j = start; j < end; ++j) {
        atomicAdd(&y_k[col_idx[j]], values[j] * x_val);
    }
}

__global__ void batched_csc_spmv_kernel(std::size_t batch_size, std::size_t rows, std::size_t cols,
                                        const double* __restrict__ values,
                                        const std::size_t* __restrict__ col_ptr,
                                        const std::size_t* __restrict__ row_idx,
                                        const double* __restrict__ batch_x,
                                        double* __restrict__ batch_y) {
    const std::size_t global_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total_items = batch_size * cols;
    if (global_idx >= total_items) {
        return;
    }

    const std::size_t k = global_idx / cols;
    const std::size_t c = global_idx % cols;

    const std::size_t start = col_ptr[c];
    const std::size_t end = col_ptr[c + 1];
    const double x_val = batch_x[k * cols + c];
    double* y_k = batch_y + (k * rows);

    for (std::size_t j = start; j < end; ++j) {
        atomicAdd(&y_k[row_idx[j]], values[j] * x_val);
    }
}

__global__ void batched_axpby_kernel(std::size_t total_elements, double alpha,
                                     const double* __restrict__ x, double beta,
                                     double* __restrict__ y) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        y[idx] = alpha * x[idx] + beta * y[idx];
    }
}

__global__ void batched_dot_kernel(std::size_t batch_size, std::size_t n,
                                   const double* __restrict__ x, const double* __restrict__ y,
                                   double* __restrict__ results) {
    const std::size_t k = blockIdx.x;
    if (k >= batch_size)
        return;

    const double* x_k = x + k * n;
    const double* y_k = y + k * n;

    double thread_sum = 0.0;
    for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) {
        thread_sum += x_k[i] * y_k[i];
    }

    __shared__ double sdata[256];
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        results[k] = sdata[0];
    }
}

__global__ void batched_l2_norm_kernel(std::size_t batch_size, std::size_t n,
                                       const double* __restrict__ x, double* __restrict__ results) {
    const std::size_t k = blockIdx.x;
    if (k >= batch_size)
        return;

    const double* x_k = x + k * n;

    double thread_sum = 0.0;
    for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) {
        const double v = x_k[i];
        thread_sum += v * v;
    }

    __shared__ double sdata[256];
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        results[k] = sqrt(sdata[0]);
    }
}

__global__ void batched_linf_norm_kernel(std::size_t batch_size, std::size_t n,
                                         const double* __restrict__ x,
                                         double* __restrict__ results) {
    const std::size_t k = blockIdx.x;
    if (k >= batch_size)
        return;

    const double* x_k = x + k * n;

    double thread_max = 0.0;
    for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) {
        const double v = fabs(x_k[i]);
        if (v > thread_max)
            thread_max = v;
    }

    __shared__ double sdata[256];
    sdata[threadIdx.x] = thread_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sdata[threadIdx.x + s] > sdata[threadIdx.x]) {
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        results[k] = sdata[0];
    }
}

}  // namespace

void cuda_batched_csr_spmv(std::size_t batch_size, std::size_t rows, std::size_t cols,
                           std::size_t nnz, const double* h_values, const std::size_t* h_row_ptr,
                           const std::size_t* h_col_idx, const double* h_batch_x, double* h_batch_y,
                           int device_id, CudaSpmvExecutionTimes* timings) {
    if (batch_size == 0)
        return;
    if (rows == 0 || cols == 0) {
        std::fill_n(h_batch_y, batch_size * rows, 0.0);
        return;
    }

    cudaError_t dev_err = cudaSetDevice(device_id);
    if (dev_err != cudaSuccess) {
        throw core::NumericalException("cudaSetDevice failed: " +
                                       std::string(cudaGetErrorString(dev_err)));
    }

    CudaStreamRAII stream;
    CudaEventRAII ev_h2d_start, ev_h2d_end;
    CudaEventRAII ev_kern_start, ev_kern_end;
    CudaEventRAII ev_d2h_start, ev_d2h_end;

    CudaDeviceBufferRAII<double> d_values(nnz);
    CudaDeviceBufferRAII<std::size_t> d_row_ptr(rows + 1);
    CudaDeviceBufferRAII<std::size_t> d_col_idx(nnz);
    CudaDeviceBufferRAII<double> d_batch_x(batch_size * cols);
    CudaDeviceBufferRAII<double> d_batch_y(batch_size * rows);

    ev_h2d_start.record(stream.stream);
    d_values.upload_async(h_values, nnz, stream.stream);
    d_row_ptr.upload_async(h_row_ptr, rows + 1, stream.stream);
    d_col_idx.upload_async(h_col_idx, nnz, stream.stream);
    d_batch_x.upload_async(h_batch_x, batch_size * cols, stream.stream);
    ev_h2d_end.record(stream.stream);

    const unsigned int block_dim = 256;
    const std::size_t total_items = batch_size * rows;
    const unsigned int grid_dim =
        static_cast<unsigned int>((total_items + block_dim - 1) / block_dim);

    ev_kern_start.record(stream.stream);
    batched_csr_spmv_kernel_size_t<<<grid_dim, block_dim, 0, stream.stream>>>(
        batch_size, rows, cols, d_values.get(), d_row_ptr.get(), d_col_idx.get(), d_batch_x.get(),
        d_batch_y.get());
    ev_kern_end.record(stream.stream);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw core::NumericalException("CUDA kernel launch failed: " +
                                       std::string(cudaGetErrorString(launch_err)));
    }

    ev_d2h_start.record(stream.stream);
    d_batch_y.download_async(h_batch_y, batch_size * rows, stream.stream);
    ev_d2h_end.record(stream.stream);

    cudaError_t sync_err = cudaStreamSynchronize(stream.stream);
    if (sync_err != cudaSuccess) {
        throw core::NumericalException("cudaStreamSynchronize failed: " +
                                       std::string(cudaGetErrorString(sync_err)));
    }

    if (timings != nullptr) {
        timings->h2d_time_seconds = CudaEventRAII::elapsed_sec(ev_h2d_start, ev_h2d_end);
        timings->kernel_time_seconds = CudaEventRAII::elapsed_sec(ev_kern_start, ev_kern_end);
        timings->d2h_time_seconds = CudaEventRAII::elapsed_sec(ev_d2h_start, ev_d2h_end);
        timings->total_gpu_time_seconds =
            timings->h2d_time_seconds + timings->kernel_time_seconds + timings->d2h_time_seconds;
    }
}

void cuda_batched_csr_spmv_int(std::size_t batch_size, std::size_t rows, std::size_t cols,
                               std::size_t nnz, const double* h_values, const int* h_row_ptr,
                               const int* h_col_idx, const double* h_batch_x, double* h_batch_y,
                               int device_id, CudaSpmvExecutionTimes* timings) {
    if (batch_size == 0)
        return;
    if (rows == 0 || cols == 0) {
        std::fill_n(h_batch_y, batch_size * rows, 0.0);
        return;
    }

    cudaError_t dev_err = cudaSetDevice(device_id);
    if (dev_err != cudaSuccess) {
        throw core::NumericalException("cudaSetDevice failed: " +
                                       std::string(cudaGetErrorString(dev_err)));
    }

    CudaStreamRAII stream;
    CudaEventRAII ev_h2d_start, ev_h2d_end;
    CudaEventRAII ev_kern_start, ev_kern_end;
    CudaEventRAII ev_d2h_start, ev_d2h_end;

    CudaDeviceBufferRAII<double> d_values(nnz);
    CudaDeviceBufferRAII<int> d_row_ptr(rows + 1);
    CudaDeviceBufferRAII<int> d_col_idx(nnz);
    CudaDeviceBufferRAII<double> d_batch_x(batch_size * cols);
    CudaDeviceBufferRAII<double> d_batch_y(batch_size * rows);

    ev_h2d_start.record(stream.stream);
    d_values.upload_async(h_values, nnz, stream.stream);
    d_row_ptr.upload_async(h_row_ptr, rows + 1, stream.stream);
    d_col_idx.upload_async(h_col_idx, nnz, stream.stream);
    d_batch_x.upload_async(h_batch_x, batch_size * cols, stream.stream);
    ev_h2d_end.record(stream.stream);

    const unsigned int block_dim = 256;
    const std::size_t total_items = batch_size * rows;
    const unsigned int grid_dim =
        static_cast<unsigned int>((total_items + block_dim - 1) / block_dim);

    ev_kern_start.record(stream.stream);
    batched_csr_spmv_kernel_int<<<grid_dim, block_dim, 0, stream.stream>>>(
        batch_size, rows, cols, d_values.get(), d_row_ptr.get(), d_col_idx.get(), d_batch_x.get(),
        d_batch_y.get());
    ev_kern_end.record(stream.stream);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw core::NumericalException("CUDA kernel launch failed: " +
                                       std::string(cudaGetErrorString(launch_err)));
    }

    ev_d2h_start.record(stream.stream);
    d_batch_y.download_async(h_batch_y, batch_size * rows, stream.stream);
    ev_d2h_end.record(stream.stream);

    cudaError_t sync_err = cudaStreamSynchronize(stream.stream);
    if (sync_err != cudaSuccess) {
        throw core::NumericalException("cudaStreamSynchronize failed: " +
                                       std::string(cudaGetErrorString(sync_err)));
    }

    if (timings != nullptr) {
        timings->h2d_time_seconds = CudaEventRAII::elapsed_sec(ev_h2d_start, ev_h2d_end);
        timings->kernel_time_seconds = CudaEventRAII::elapsed_sec(ev_kern_start, ev_kern_end);
        timings->d2h_time_seconds = CudaEventRAII::elapsed_sec(ev_d2h_start, ev_d2h_end);
        timings->total_gpu_time_seconds =
            timings->h2d_time_seconds + timings->kernel_time_seconds + timings->d2h_time_seconds;
    }
}

void cuda_batched_csr_transpose_spmv(std::size_t batch_size, std::size_t rows, std::size_t cols,
                                     std::size_t nnz, const double* h_values,
                                     const std::size_t* h_row_ptr, const std::size_t* h_col_idx,
                                     const double* h_batch_x, double* h_batch_y, int device_id,
                                     CudaSpmvExecutionTimes* timings) {
    if (batch_size == 0)
        return;
    if (rows == 0 || cols == 0) {
        std::fill_n(h_batch_y, batch_size * cols, 0.0);
        return;
    }

    cudaError_t dev_err = cudaSetDevice(device_id);
    if (dev_err != cudaSuccess) {
        throw core::NumericalException("cudaSetDevice failed: " +
                                       std::string(cudaGetErrorString(dev_err)));
    }

    CudaStreamRAII stream;
    CudaEventRAII ev_h2d_start, ev_h2d_end;
    CudaEventRAII ev_kern_start, ev_kern_end;
    CudaEventRAII ev_d2h_start, ev_d2h_end;

    CudaDeviceBufferRAII<double> d_values(nnz);
    CudaDeviceBufferRAII<std::size_t> d_row_ptr(rows + 1);
    CudaDeviceBufferRAII<std::size_t> d_col_idx(nnz);
    CudaDeviceBufferRAII<double> d_batch_x(batch_size * rows);
    CudaDeviceBufferRAII<double> d_batch_y(batch_size * cols);

    ev_h2d_start.record(stream.stream);
    d_values.upload_async(h_values, nnz, stream.stream);
    d_row_ptr.upload_async(h_row_ptr, rows + 1, stream.stream);
    d_col_idx.upload_async(h_col_idx, nnz, stream.stream);
    d_batch_x.upload_async(h_batch_x, batch_size * rows, stream.stream);
    cudaMemsetAsync(d_batch_y.get(), 0, batch_size * cols * sizeof(double), stream.stream);
    ev_h2d_end.record(stream.stream);

    const unsigned int block_dim = 256;
    const std::size_t total_items = batch_size * rows;
    const unsigned int grid_dim =
        static_cast<unsigned int>((total_items + block_dim - 1) / block_dim);

    ev_kern_start.record(stream.stream);
    batched_csr_transpose_spmv_kernel<<<grid_dim, block_dim, 0, stream.stream>>>(
        batch_size, rows, cols, d_values.get(), d_row_ptr.get(), d_col_idx.get(), d_batch_x.get(),
        d_batch_y.get());
    ev_kern_end.record(stream.stream);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw core::NumericalException("CUDA kernel launch failed: " +
                                       std::string(cudaGetErrorString(launch_err)));
    }

    ev_d2h_start.record(stream.stream);
    d_batch_y.download_async(h_batch_y, batch_size * cols, stream.stream);
    ev_d2h_end.record(stream.stream);

    cudaError_t sync_err = cudaStreamSynchronize(stream.stream);
    if (sync_err != cudaSuccess) {
        throw core::NumericalException("cudaStreamSynchronize failed: " +
                                       std::string(cudaGetErrorString(sync_err)));
    }

    if (timings != nullptr) {
        timings->h2d_time_seconds = CudaEventRAII::elapsed_sec(ev_h2d_start, ev_h2d_end);
        timings->kernel_time_seconds = CudaEventRAII::elapsed_sec(ev_kern_start, ev_kern_end);
        timings->d2h_time_seconds = CudaEventRAII::elapsed_sec(ev_d2h_start, ev_d2h_end);
        timings->total_gpu_time_seconds =
            timings->h2d_time_seconds + timings->kernel_time_seconds + timings->d2h_time_seconds;
    }
}

void cuda_batched_csc_spmv(std::size_t batch_size, std::size_t rows, std::size_t cols,
                           std::size_t nnz, const double* h_values, const std::size_t* h_col_ptr,
                           const std::size_t* h_row_idx, const double* h_batch_x, double* h_batch_y,
                           int device_id, CudaSpmvExecutionTimes* timings) {
    if (batch_size == 0)
        return;
    if (rows == 0 || cols == 0) {
        std::fill_n(h_batch_y, batch_size * rows, 0.0);
        return;
    }

    cudaError_t dev_err = cudaSetDevice(device_id);
    if (dev_err != cudaSuccess) {
        throw core::NumericalException("cudaSetDevice failed: " +
                                       std::string(cudaGetErrorString(dev_err)));
    }

    CudaStreamRAII stream;
    CudaEventRAII ev_h2d_start, ev_h2d_end;
    CudaEventRAII ev_kern_start, ev_kern_end;
    CudaEventRAII ev_d2h_start, ev_d2h_end;

    CudaDeviceBufferRAII<double> d_values(nnz);
    CudaDeviceBufferRAII<std::size_t> d_col_ptr(cols + 1);
    CudaDeviceBufferRAII<std::size_t> d_row_idx(nnz);
    CudaDeviceBufferRAII<double> d_batch_x(batch_size * cols);
    CudaDeviceBufferRAII<double> d_batch_y(batch_size * rows);

    ev_h2d_start.record(stream.stream);
    d_values.upload_async(h_values, nnz, stream.stream);
    d_col_ptr.upload_async(h_col_ptr, cols + 1, stream.stream);
    d_row_idx.upload_async(h_row_idx, nnz, stream.stream);
    d_batch_x.upload_async(h_batch_x, batch_size * cols, stream.stream);
    cudaMemsetAsync(d_batch_y.get(), 0, batch_size * rows * sizeof(double), stream.stream);
    ev_h2d_end.record(stream.stream);

    const unsigned int block_dim = 256;
    const std::size_t total_items = batch_size * cols;
    const unsigned int grid_dim =
        static_cast<unsigned int>((total_items + block_dim - 1) / block_dim);

    ev_kern_start.record(stream.stream);
    batched_csc_spmv_kernel<<<grid_dim, block_dim, 0, stream.stream>>>(
        batch_size, rows, cols, d_values.get(), d_col_ptr.get(), d_row_idx.get(), d_batch_x.get(),
        d_batch_y.get());
    ev_kern_end.record(stream.stream);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw core::NumericalException("CUDA kernel launch failed: " +
                                       std::string(cudaGetErrorString(launch_err)));
    }

    ev_d2h_start.record(stream.stream);
    d_batch_y.download_async(h_batch_y, batch_size * rows, stream.stream);
    ev_d2h_end.record(stream.stream);

    cudaError_t sync_err = cudaStreamSynchronize(stream.stream);
    if (sync_err != cudaSuccess) {
        throw core::NumericalException("cudaStreamSynchronize failed: " +
                                       std::string(cudaGetErrorString(sync_err)));
    }

    if (timings != nullptr) {
        timings->h2d_time_seconds = CudaEventRAII::elapsed_sec(ev_h2d_start, ev_h2d_end);
        timings->kernel_time_seconds = CudaEventRAII::elapsed_sec(ev_kern_start, ev_kern_end);
        timings->d2h_time_seconds = CudaEventRAII::elapsed_sec(ev_d2h_start, ev_d2h_end);
        timings->total_gpu_time_seconds =
            timings->h2d_time_seconds + timings->kernel_time_seconds + timings->d2h_time_seconds;
    }
}

void cuda_batched_axpby(std::size_t batch_size, std::size_t n, double alpha, const double* h_x,
                        double beta, double* h_y, int device_id, CudaSpmvExecutionTimes* timings) {
    const std::size_t total = batch_size * n;
    if (total == 0)
        return;

    cudaError_t dev_err = cudaSetDevice(device_id);
    if (dev_err != cudaSuccess) {
        throw core::NumericalException("cudaSetDevice failed: " +
                                       std::string(cudaGetErrorString(dev_err)));
    }

    CudaStreamRAII stream;
    CudaEventRAII ev_h2d_start, ev_h2d_end;
    CudaEventRAII ev_kern_start, ev_kern_end;
    CudaEventRAII ev_d2h_start, ev_d2h_end;

    CudaDeviceBufferRAII<double> d_x(total);
    CudaDeviceBufferRAII<double> d_y(total);

    ev_h2d_start.record(stream.stream);
    d_x.upload_async(h_x, total, stream.stream);
    d_y.upload_async(h_y, total, stream.stream);
    ev_h2d_end.record(stream.stream);

    const unsigned int block_dim = 256;
    const unsigned int grid_dim = static_cast<unsigned int>((total + block_dim - 1) / block_dim);

    ev_kern_start.record(stream.stream);
    batched_axpby_kernel<<<grid_dim, block_dim, 0, stream.stream>>>(total, alpha, d_x.get(), beta,
                                                                    d_y.get());
    ev_kern_end.record(stream.stream);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw core::NumericalException("CUDA kernel launch failed: " +
                                       std::string(cudaGetErrorString(launch_err)));
    }

    ev_d2h_start.record(stream.stream);
    d_y.download_async(h_y, total, stream.stream);
    ev_d2h_end.record(stream.stream);

    cudaError_t sync_err = cudaStreamSynchronize(stream.stream);
    if (sync_err != cudaSuccess) {
        throw core::NumericalException("cudaStreamSynchronize failed: " +
                                       std::string(cudaGetErrorString(sync_err)));
    }

    if (timings != nullptr) {
        timings->h2d_time_seconds = CudaEventRAII::elapsed_sec(ev_h2d_start, ev_h2d_end);
        timings->kernel_time_seconds = CudaEventRAII::elapsed_sec(ev_kern_start, ev_kern_end);
        timings->d2h_time_seconds = CudaEventRAII::elapsed_sec(ev_d2h_start, ev_d2h_end);
        timings->total_gpu_time_seconds =
            timings->h2d_time_seconds + timings->kernel_time_seconds + timings->d2h_time_seconds;
    }
}

void cuda_batched_dot(std::size_t batch_size, std::size_t n, const double* h_x, const double* h_y,
                      double* h_results, int device_id, CudaSpmvExecutionTimes* timings) {
    if (batch_size == 0)
        return;
    if (n == 0) {
        std::fill_n(h_results, batch_size, 0.0);
        return;
    }

    cudaError_t dev_err = cudaSetDevice(device_id);
    if (dev_err != cudaSuccess) {
        throw core::NumericalException("cudaSetDevice failed: " +
                                       std::string(cudaGetErrorString(dev_err)));
    }

    CudaStreamRAII stream;
    CudaEventRAII ev_h2d_start, ev_h2d_end;
    CudaEventRAII ev_kern_start, ev_kern_end;
    CudaEventRAII ev_d2h_start, ev_d2h_end;

    const std::size_t total = batch_size * n;
    CudaDeviceBufferRAII<double> d_x(total);
    CudaDeviceBufferRAII<double> d_y(total);
    CudaDeviceBufferRAII<double> d_results(batch_size);

    ev_h2d_start.record(stream.stream);
    d_x.upload_async(h_x, total, stream.stream);
    d_y.upload_async(h_y, total, stream.stream);
    ev_h2d_end.record(stream.stream);

    const unsigned int block_dim = 256;
    const unsigned int grid_dim = static_cast<unsigned int>(batch_size);

    ev_kern_start.record(stream.stream);
    batched_dot_kernel<<<grid_dim, block_dim, 0, stream.stream>>>(batch_size, n, d_x.get(),
                                                                  d_y.get(), d_results.get());
    ev_kern_end.record(stream.stream);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw core::NumericalException("CUDA kernel launch failed: " +
                                       std::string(cudaGetErrorString(launch_err)));
    }

    ev_d2h_start.record(stream.stream);
    d_results.download_async(h_results, batch_size, stream.stream);
    ev_d2h_end.record(stream.stream);

    cudaError_t sync_err = cudaStreamSynchronize(stream.stream);
    if (sync_err != cudaSuccess) {
        throw core::NumericalException("cudaStreamSynchronize failed: " +
                                       std::string(cudaGetErrorString(sync_err)));
    }

    if (timings != nullptr) {
        timings->h2d_time_seconds = CudaEventRAII::elapsed_sec(ev_h2d_start, ev_h2d_end);
        timings->kernel_time_seconds = CudaEventRAII::elapsed_sec(ev_kern_start, ev_kern_end);
        timings->d2h_time_seconds = CudaEventRAII::elapsed_sec(ev_d2h_start, ev_d2h_end);
        timings->total_gpu_time_seconds =
            timings->h2d_time_seconds + timings->kernel_time_seconds + timings->d2h_time_seconds;
    }
}

void cuda_batched_l2_norm(std::size_t batch_size, std::size_t n, const double* h_x,
                          double* h_results, int device_id, CudaSpmvExecutionTimes* timings) {
    if (batch_size == 0)
        return;
    if (n == 0) {
        std::fill_n(h_results, batch_size, 0.0);
        return;
    }

    cudaError_t dev_err = cudaSetDevice(device_id);
    if (dev_err != cudaSuccess) {
        throw core::NumericalException("cudaSetDevice failed: " +
                                       std::string(cudaGetErrorString(dev_err)));
    }

    CudaStreamRAII stream;
    CudaEventRAII ev_h2d_start, ev_h2d_end;
    CudaEventRAII ev_kern_start, ev_kern_end;
    CudaEventRAII ev_d2h_start, ev_d2h_end;

    const std::size_t total = batch_size * n;
    CudaDeviceBufferRAII<double> d_x(total);
    CudaDeviceBufferRAII<double> d_results(batch_size);

    ev_h2d_start.record(stream.stream);
    d_x.upload_async(h_x, total, stream.stream);
    ev_h2d_end.record(stream.stream);

    const unsigned int block_dim = 256;
    const unsigned int grid_dim = static_cast<unsigned int>(batch_size);

    ev_kern_start.record(stream.stream);
    batched_l2_norm_kernel<<<grid_dim, block_dim, 0, stream.stream>>>(batch_size, n, d_x.get(),
                                                                      d_results.get());
    ev_kern_end.record(stream.stream);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw core::NumericalException("CUDA kernel launch failed: " +
                                       std::string(cudaGetErrorString(launch_err)));
    }

    ev_d2h_start.record(stream.stream);
    d_results.download_async(h_results, batch_size, stream.stream);
    ev_d2h_end.record(stream.stream);

    cudaError_t sync_err = cudaStreamSynchronize(stream.stream);
    if (sync_err != cudaSuccess) {
        throw core::NumericalException("cudaStreamSynchronize failed: " +
                                       std::string(cudaGetErrorString(sync_err)));
    }

    if (timings != nullptr) {
        timings->h2d_time_seconds = CudaEventRAII::elapsed_sec(ev_h2d_start, ev_h2d_end);
        timings->kernel_time_seconds = CudaEventRAII::elapsed_sec(ev_kern_start, ev_kern_end);
        timings->d2h_time_seconds = CudaEventRAII::elapsed_sec(ev_d2h_start, ev_d2h_end);
        timings->total_gpu_time_seconds =
            timings->h2d_time_seconds + timings->kernel_time_seconds + timings->d2h_time_seconds;
    }
}

void cuda_batched_linf_norm(std::size_t batch_size, std::size_t n, const double* h_x,
                            double* h_results, int device_id, CudaSpmvExecutionTimes* timings) {
    if (batch_size == 0)
        return;
    if (n == 0) {
        std::fill_n(h_results, batch_size, 0.0);
        return;
    }

    cudaError_t dev_err = cudaSetDevice(device_id);
    if (dev_err != cudaSuccess) {
        throw core::NumericalException("cudaSetDevice failed: " +
                                       std::string(cudaGetErrorString(dev_err)));
    }

    CudaStreamRAII stream;
    CudaEventRAII ev_h2d_start, ev_h2d_end;
    CudaEventRAII ev_kern_start, ev_kern_end;
    CudaEventRAII ev_d2h_start, ev_d2h_end;

    const std::size_t total = batch_size * n;
    CudaDeviceBufferRAII<double> d_x(total);
    CudaDeviceBufferRAII<double> d_results(batch_size);

    ev_h2d_start.record(stream.stream);
    d_x.upload_async(h_x, total, stream.stream);
    ev_h2d_end.record(stream.stream);

    const unsigned int block_dim = 256;
    const unsigned int grid_dim = static_cast<unsigned int>(batch_size);

    ev_kern_start.record(stream.stream);
    batched_linf_norm_kernel<<<grid_dim, block_dim, 0, stream.stream>>>(batch_size, n, d_x.get(),
                                                                        d_results.get());
    ev_kern_end.record(stream.stream);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        throw core::NumericalException("CUDA kernel launch failed: " +
                                       std::string(cudaGetErrorString(launch_err)));
    }

    ev_d2h_start.record(stream.stream);
    d_results.download_async(h_results, batch_size, stream.stream);
    ev_d2h_end.record(stream.stream);

    cudaError_t sync_err = cudaStreamSynchronize(stream.stream);
    if (sync_err != cudaSuccess) {
        throw core::NumericalException("cudaStreamSynchronize failed: " +
                                       std::string(cudaGetErrorString(sync_err)));
    }

    if (timings != nullptr) {
        timings->h2d_time_seconds = CudaEventRAII::elapsed_sec(ev_h2d_start, ev_h2d_end);
        timings->kernel_time_seconds = CudaEventRAII::elapsed_sec(ev_kern_start, ev_kern_end);
        timings->d2h_time_seconds = CudaEventRAII::elapsed_sec(ev_d2h_start, ev_d2h_end);
        timings->total_gpu_time_seconds =
            timings->h2d_time_seconds + timings->kernel_time_seconds + timings->d2h_time_seconds;
    }
}

}  // namespace kavach::backend::cuda

#endif
