#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace kavach::backend::cuda {

enum class CudaKernelId {
    BATCHED_CSR_SPMV,
    BATCHED_CSR_TRANSPOSE_SPMV,
    BATCHED_CSC_SPMV,
    BATCHED_AXPBY,
    BATCHED_DOT,
    BATCHED_L2_NORM,
    BATCHED_LINF_NORM
};

struct CudaKernelDescriptor {
    CudaKernelId id;
    std::string name;
    std::string precision{"FP64 (double)"};
    std::string layout;
    unsigned int default_block_size{256};
    bool supports_variable_length{true};
    std::string cpu_fallback{"CpuMulticoreBackend"};
};

class CudaKernelRegistry {
public:
    [[nodiscard]] static const std::vector<CudaKernelDescriptor>& get_all_kernels() {
        static const std::vector<CudaKernelDescriptor> registry = {
            {CudaKernelId::BATCHED_CSR_SPMV, "cuda_batched_csr_spmv", "FP64 (double)",
             "CSR (row_ptr, col_indices)", 256, true, "CpuMulticoreBackend::batched_spmv"},
            {CudaKernelId::BATCHED_CSR_TRANSPOSE_SPMV, "cuda_batched_csr_transpose_spmv",
             "FP64 (double)", "CSR (row_ptr, col_indices)", 256, true,
             "CpuMulticoreBackend::batched_spmv_transpose"},
            {CudaKernelId::BATCHED_CSC_SPMV, "cuda_batched_csc_spmv", "FP64 (double)",
             "CSC (col_ptr, row_indices)", 256, true, "CpuMulticoreBackend::batched_csc_spmv"},
            {CudaKernelId::BATCHED_AXPBY, "cuda_batched_axpby", "FP64 (double)", "Contiguous 1D",
             256, true, "CpuMulticoreBackend::batched_axpby"},
            {CudaKernelId::BATCHED_DOT, "cuda_batched_dot", "FP64 (double)", "Contiguous 1D", 256,
             true, "CpuMulticoreBackend::batched_dot"},
            {CudaKernelId::BATCHED_L2_NORM, "cuda_batched_l2_norm", "FP64 (double)",
             "Contiguous 1D", 256, true, "CpuMulticoreBackend::batched_l2_norm"},
            {CudaKernelId::BATCHED_LINF_NORM, "cuda_batched_linf_norm", "FP64 (double)",
             "Contiguous 1D", 256, true, "CpuMulticoreBackend::batched_linf_norm"}};
        return registry;
    }

    [[nodiscard]] static const CudaKernelDescriptor* find_kernel(CudaKernelId id) {
        for (const auto& k : get_all_kernels()) {
            if (k.id == id) {
                return &k;
            }
        }
        return nullptr;
    }
};

}  // namespace kavach::backend::cuda
