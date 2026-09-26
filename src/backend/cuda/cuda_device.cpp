#include "backend/cuda/cuda_device.hpp"

#include "core/error.hpp"

#ifdef KAVACH_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <sstream>

namespace kavach::backend::cuda {

#ifdef KAVACH_USE_CUDA

bool is_cuda_available() noexcept {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

int get_cuda_device_count() noexcept {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return 0;
    }
    return count;
}

CudaDeviceProperties get_cuda_device_properties(int device_id) {
    int count = get_cuda_device_count();
    if (device_id < 0 || device_id >= count) {
        throw core::InputException("Invalid CUDA device ID: " + std::to_string(device_id) +
                                   " (available devices: " + std::to_string(count) + ")");
    }

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        throw core::NumericalException("cudaGetDeviceProperties failed: " +
                                       std::string(cudaGetErrorString(err)));
    }

    CudaDeviceProperties res;
    res.device_id = device_id;
    res.name = prop.name;
    res.major = prop.major;
    res.minor = prop.minor;
    res.total_global_memory_bytes = prop.totalGlobalMem;
    res.multi_processor_count = prop.multiProcessorCount;
    res.max_threads_per_block = prop.maxThreadsPerBlock;
    res.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
    res.warp_size = prop.warpSize;
    res.is_valid = true;
    return res;
}

void set_cuda_device(int device_id) {
    int count = get_cuda_device_count();
    if (device_id < 0 || device_id >= count) {
        throw core::InputException("Invalid CUDA device ID: " + std::to_string(device_id) +
                                   " (available devices: " + std::to_string(count) + ")");
    }
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        throw core::NumericalException("cudaSetDevice failed: " +
                                       std::string(cudaGetErrorString(err)));
    }
}

std::string get_cuda_status_string() noexcept {
    int count = get_cuda_device_count();
    if (count == 0) {
        return "CUDA runtime detected, but 0 compatible GPU devices found";
    }
    std::ostringstream oss;
    oss << "CUDA active: " << count << " device(s) detected. ";
    try {
        auto prop = get_cuda_device_properties(0);
        oss << "Device 0: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ", "
            << (prop.total_global_memory_bytes / (1024 * 1024)) << " MB VRAM)";
    } catch (...) {
        oss << "(Unable to query device 0 properties)";
    }
    return oss.str();
}

std::pair<std::size_t, std::size_t> get_cuda_memory_info() noexcept {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        return {0, 0};
    }
    return {free_bytes, total_bytes};
}

bool check_cuda_memory_sufficient(std::size_t required_bytes, double safety_margin) noexcept {
    auto [free_bytes, total_bytes] = get_cuda_memory_info();
    if (free_bytes == 0) {
        return false;
    }
    const double allowed = static_cast<double>(free_bytes) * safety_margin;
    return static_cast<double>(required_bytes) <= allowed;
}

#else

bool is_cuda_available() noexcept {
    return false;
}

int get_cuda_device_count() noexcept {
    return 0;
}

CudaDeviceProperties get_cuda_device_properties(int device_id) {
    throw core::InputException("CUDA backend is not enabled in this build (device_id=" +
                               std::to_string(device_id) + "). Built with KAVACH_ENABLE_CUDA=OFF.");
}

void set_cuda_device(int device_id) {
    throw core::InputException("Cannot set CUDA device (device_id=" + std::to_string(device_id) +
                               "): CUDA is disabled or unavailable.");
}

std::string get_cuda_status_string() noexcept {
    return "CUDA unavailable (binary compiled in CPU-only mode)";
}

std::pair<std::size_t, std::size_t> get_cuda_memory_info() noexcept {
    return {0, 0};
}

bool check_cuda_memory_sufficient(std::size_t, double) noexcept {
    return false;
}

#endif

}  // namespace kavach::backend::cuda
