/**
 * @file constants.ts
 * @brief Authoritative system metadata and default configurations.
 * Sourced directly from repository CMakeLists.txt, configs/default_tolerances.json,
 * and C++20 solver engine architecture.
 */

export const SYSTEM_METADATA = {
  PRODUCT_NAME: 'Kavach-Opt',
  PRODUCT_TAGLINE: 'Sovereign Optimization Engine',
  PRODUCT_VERSION: 'v0.1.0',
  BUILD_TYPE: 'Release',
  STANDARD: 'C++20 (ISO/IEC 14882:2020)',
  ENGINEERING_AUDIT_STATUS: 'CERTIFIED GREEN',
} as const;

export const DEFAULT_TOLERANCES = {
  PRIMAL_FEASIBILITY: 1e-7,
  DUAL_FEASIBILITY: 1e-7,
  INTEGRALITY: 1e-6,
  OPTIMALITY: 1e-7,
  PIVOT: 1e-9,
  RESIDUAL: 1e-6,
  KKT_STATIONARITY: 1e-7,
} as const;

export interface CudaKernelSpec {
  name: string;
  layout: string;
  precision: string;
  status: string;
  targetHardware: string;
}

export const CUDA_KERNEL_REGISTRY: CudaKernelSpec[] = [
  {
    name: 'Batched CSR SpMV',
    layout: 'CSR Matrix',
    precision: 'FP64 (IEEE 754)',
    status: 'CPU Fallback (R012)',
    targetHardware: 'NVIDIA sm_70+',
  },
  {
    name: 'Batched CSR Transpose SpMV',
    layout: 'CSR Transpose',
    precision: 'FP64 (IEEE 754)',
    status: 'CPU Fallback (R012)',
    targetHardware: 'NVIDIA sm_70+',
  },
  {
    name: 'Batched CSC SpMV',
    layout: 'CSC Matrix',
    precision: 'FP64 (IEEE 754)',
    status: 'CPU Fallback (R012)',
    targetHardware: 'NVIDIA sm_70+',
  },
  {
    name: 'Batched AXPBY',
    layout: 'Dense Vector Batch',
    precision: 'FP64 (IEEE 754)',
    status: 'CPU Fallback (R012)',
    targetHardware: 'NVIDIA sm_70+',
  },
  {
    name: 'Batched Dot Product',
    layout: 'Dense Vector Batch',
    precision: 'FP64 (IEEE 754)',
    status: 'CPU Fallback (R012)',
    targetHardware: 'NVIDIA sm_70+',
  },
  {
    name: 'Batched L2 Norm',
    layout: 'Dense Vector Batch',
    precision: 'FP64 (IEEE 754)',
    status: 'CPU Fallback (R012)',
    targetHardware: 'NVIDIA sm_70+',
  },
  {
    name: 'Batched L∞ Norm',
    layout: 'Dense Vector Batch',
    precision: 'FP64 (IEEE 754)',
    status: 'CPU Fallback (R012)',
    targetHardware: 'NVIDIA sm_70+',
  },
];
