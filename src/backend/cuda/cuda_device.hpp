#pragma once

#include <cstddef>
#include <string>

namespace kavach::backend::cuda {

/**
 * @brief Hardware and runtime properties of an NVIDIA CUDA-capable GPU.
 * Fulfills Blueprint Section 15 and Section 23 (Hardware capture).
 */
struct CudaDeviceProperties {
    int device_id{-1};
    std::string name{"None"};
    int major{0};
    int minor{0};
    std::size_t total_global_memory_bytes{0};
    int multi_processor_count{0};
    int max_threads_per_block{0};
    int max_threads_per_multiprocessor{0};
    int warp_size{32};
    bool is_valid{false};
};

/**
 * @brief Checks if CUDA runtime and at least one compatible GPU device are available.
 * Safe to call on any platform; never throws or crashes if CUDA is absent.
 */
[[nodiscard]] bool is_cuda_available() noexcept;

/**
 * @brief Query total count of available CUDA-capable devices.
 * Returns 0 if CUDA is not compiled or no compatible GPUs are found.
 */
[[nodiscard]] int get_cuda_device_count() noexcept;

/**
 * @brief Query hardware properties of a specific CUDA device.
 * @param device_id GPU device ordinal (0 to device_count - 1).
 * @throws core::InputException if device_id is invalid or CUDA is unavailable.
 */
[[nodiscard]] CudaDeviceProperties get_cuda_device_properties(int device_id);

/**
 * @brief Set the active CUDA device for the calling thread.
 * @param device_id GPU device ordinal.
 * @throws core::InputException if device_id is invalid.
 */
void set_cuda_device(int device_id);

/**
 * @brief Informative human-readable status string of the CUDA backend.
 */
[[nodiscard]] std::string get_cuda_status_string() noexcept;

/**
 * @brief Query currently free and total global memory in bytes on the active CUDA device.
 * @return Pair of {free_bytes, total_bytes}. Returns {0, 0} if CUDA is unavailable.
 */
[[nodiscard]] std::pair<std::size_t, std::size_t> get_cuda_memory_info() noexcept;

/**
 * @brief Checks if CUDA device has sufficient free memory with safety headroom.
 * @param required_bytes Total memory required by the operation
 * @param safety_margin Fractional headroom required (default 0.85)
 */
[[nodiscard]] bool check_cuda_memory_sufficient(std::size_t required_bytes,
                                                double safety_margin = 0.85) noexcept;

/**
 * @brief Estimate device memory required for batched CSR/CSC SpMV.
 */
[[nodiscard]] constexpr std::size_t estimate_spmv_memory_bytes(std::size_t batch_size,
                                                               std::size_t rows, std::size_t cols,
                                                               std::size_t nnz) noexcept {
    return (nnz * sizeof(double)) + ((rows + 1) * sizeof(std::size_t)) +
           (nnz * sizeof(std::size_t)) + (batch_size * cols * sizeof(double)) +
           (batch_size * rows * sizeof(double));
}

/**
 * @brief Estimate device memory required for batched vector operation.
 */
[[nodiscard]] constexpr std::size_t estimate_vector_memory_bytes(
    std::size_t batch_size, std::size_t n, std::size_t num_vectors,
    std::size_t num_scalars = 0) noexcept {
    return (batch_size * n * num_vectors * sizeof(double)) + (num_scalars * sizeof(double));
}

}  // namespace kavach::backend::cuda
