#pragma once

#include <cstddef>
#include <string>

namespace kavach::backend::cuda {

/**
 * @brief Detailed breakdown of measured GPU execution timings.
 * Fulfills Blueprint Section 15 and Section 22: transfer-inclusive timing requirement.
 */
struct CudaSpmvExecutionTimes {
    double h2d_time_seconds{0.0};        ///< Host-to-Device memory copy time
    double kernel_time_seconds{0.0};     ///< Pure GPU kernel execution time
    double d2h_time_seconds{0.0};        ///< Device-to-Host memory copy time
    double total_gpu_time_seconds{0.0};  ///< Transfer-inclusive total GPU wall time
};

#ifdef KAVACH_USE_CUDA

/**
 * @brief Executes batched CSR SpMV directly on CUDA device (size_t indices).
 * Evaluates Y_k = A * X_k for k in [0, batch_size).
 *
 * @param batch_size Number of independent vectors in batch
 * @param rows Number of rows in matrix A
 * @param cols Number of columns in matrix A
 * @param nnz Number of nonzeros in matrix A
 * @param h_values Host array of nonzeros (size nnz)
 * @param h_row_ptr Host array of row pointers (size rows + 1)
 * @param h_col_idx Host array of column indices (size nnz)
 * @param h_batch_x Host contiguous array of input vectors (size batch_size * cols)
 * @param h_batch_y Host contiguous array of output vectors (size batch_size * rows)
 * @param device_id CUDA device ordinal
 * @param timings Optional pointer to receive detailed H2D, Kernel, D2H, and total timings
 */
void cuda_batched_csr_spmv(std::size_t batch_size, std::size_t rows, std::size_t cols,
                           std::size_t nnz, const double* h_values, const std::size_t* h_row_ptr,
                           const std::size_t* h_col_idx, const double* h_batch_x, double* h_batch_y,
                           int device_id = 0, CudaSpmvExecutionTimes* timings = nullptr);

/**
 * @brief Executes batched CSR SpMV directly on CUDA device (int indices).
 */
void cuda_batched_csr_spmv_int(std::size_t batch_size, std::size_t rows, std::size_t cols,
                               std::size_t nnz, const double* h_values, const int* h_row_ptr,
                               const int* h_col_idx, const double* h_batch_x, double* h_batch_y,
                               int device_id = 0, CudaSpmvExecutionTimes* timings = nullptr);

#endif

}  // namespace kavach::backend::cuda
