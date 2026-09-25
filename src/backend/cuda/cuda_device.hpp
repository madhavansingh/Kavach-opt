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

}  // namespace kavach::backend::cuda
