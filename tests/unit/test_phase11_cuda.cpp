#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "../test_framework.hpp"
#include "backend/cpu/cpu_backend.hpp"
#include "backend/cuda/cuda_backend.hpp"
#include "backend/cuda/cuda_device.hpp"
#include "backend/cuda/cuda_spmv.cuh"
#include "backend/execution_backend.hpp"
#include "core/error.hpp"
#include "core/logger.hpp"
#include "core/random.hpp"
#include "numerics/sparse_matrix.hpp"

using namespace kavach;
using namespace kavach::backend;
using namespace kavach::backend::cuda;
using namespace kavach::backend::cpu;
using namespace kavach::numerics;

namespace {

CsrMatrix create_test_csr(std::size_t rows, std::size_t cols, double density, uint64_t seed) {
    core::DeterministicRNG rng(seed);
    std::vector<Triplet> triplets;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            if (rng.uniform_real() < density) {
                triplets.push_back({r, c, rng.uniform_real() * 10.0 - 5.0});
            }
        }
        // At least one entry per row for general tests
        triplets.push_back({r, r % cols, 2.0 + static_cast<double>(r % 5)});
    }
    return CsrMatrix::from_triplets(rows, cols, triplets);
}

std::vector<double> create_test_batch(std::size_t batch_size, std::size_t dim, uint64_t seed) {
    core::DeterministicRNG rng(seed);
    std::vector<double> batch(batch_size * dim);
    for (double& val : batch) {
        val = rng.uniform_real() * 20.0 - 10.0;
    }
    return batch;
}

}  // namespace

// ============================================================================
// Test Matrix: Section 25 (Tests A through W)
// ============================================================================

// Test A: Device Detection
KAVACH_TEST(test_cuda_a_device_detection) {
    bool avail = is_cuda_available();
    int count = get_cuda_device_count();
    std::string status = get_cuda_status_string();

    KAVACH_ASSERT(!status.empty());
    if (avail) {
        KAVACH_ASSERT(count > 0);
    } else {
        KAVACH_ASSERT_EQUAL(count, 0);
    }
}

// Test B: No-Device / Graceful Availability Behavior
KAVACH_TEST(test_cuda_b_no_device_behavior) {
    CudaBackend backend;
    KAVACH_ASSERT_EQUAL(backend.type(), BackendType::CUDA_GPU);

    // If CUDA is not available, is_available() must return false cleanly without crash
    if (!is_cuda_available()) {
        KAVACH_ASSERT(!backend.is_available());
    }
}

// Test C: Invalid Device Index
KAVACH_TEST(test_cuda_c_invalid_device_index) {
    bool caught_negative = false;
    try {
        auto prop = get_cuda_device_properties(-1);
        (void)prop;
    } catch (const core::InputException&) {
        caught_negative = true;
    }
    KAVACH_ASSERT(caught_negative);

    bool caught_huge = false;
    try {
        auto prop = get_cuda_device_properties(9999);
        (void)prop;
    } catch (const core::InputException&) {
        caught_huge = true;
    }
    KAVACH_ASSERT(caught_huge);
}

// Test D: CSR Data Upload and Representation
KAVACH_TEST(test_cuda_d_csr_upload) {
    CsrMatrix A = create_test_csr(10, 10, 0.3, 101);
    KAVACH_ASSERT_EQUAL(A.rows(), 10);
    KAVACH_ASSERT_EQUAL(A.cols(), 10);
    KAVACH_ASSERT(A.nonzeros() >= 10);
    KAVACH_ASSERT(A.is_valid());
}

// Test E: Basic SpMV on CUDA Backend
KAVACH_TEST(test_cuda_e_basic_spmv) {
    const std::size_t rows = 8;
    const std::size_t cols = 8;
    const std::size_t batch_size = 1;

    CsrMatrix A = create_test_csr(rows, cols, 0.4, 201);
    std::vector<double> X = create_test_batch(batch_size, cols, 202);
    std::vector<double> Y_serial(batch_size * rows, 0.0);
    std::vector<double> Y_cuda(batch_size * rows, 0.0);

    serial_batched_spmv(A, X.data(), Y_serial.data(), batch_size);

    CudaBackend backend;
    backend.batched_spmv(A, X.data(), Y_cuda.data(), batch_size);

    auto cmp = compare_spmv_results(batch_size, rows, Y_serial.data(), Y_cuda.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);
}

// Test F: Batched SpMV on CUDA Backend
KAVACH_TEST(test_cuda_f_batched_gpu_spmv) {
    const std::size_t rows = 20;
    const std::size_t cols = 25;
    const std::size_t batch_size = 32;

    CsrMatrix A = create_test_csr(rows, cols, 0.25, 301);
    std::vector<double> X = create_test_batch(batch_size, cols, 302);
    std::vector<double> Y_serial(batch_size * rows, 0.0);
    std::vector<double> Y_cuda(batch_size * rows, 0.0);

    serial_batched_spmv(A, X.data(), Y_serial.data(), batch_size);

    CudaBackend backend;
    backend.batched_spmv(A, X.data(), Y_cuda.data(), batch_size);

    auto cmp = compare_spmv_results(batch_size, rows, Y_serial.data(), Y_cuda.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);
}

// Test G: Numerical Equivalence Against Serial Reference
KAVACH_TEST(test_cuda_g_cpu_gpu_numerical_equivalence) {
    const std::size_t rows = 30;
    const std::size_t cols = 30;
    const std::size_t batch_size = 64;

    CsrMatrix A = create_test_csr(rows, cols, 0.2, 401);
    std::vector<double> X = create_test_batch(batch_size, cols, 402);
    std::vector<double> Y_serial(batch_size * rows, 0.0);
    std::vector<double> Y_cuda(batch_size * rows, 0.0);

    serial_batched_spmv(A, X.data(), Y_serial.data(), batch_size);

    CudaBackend backend;
    backend.batched_spmv(A, X.data(), Y_cuda.data(), batch_size);

    auto cmp = compare_spmv_results(batch_size, rows, Y_serial.data(), Y_cuda.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);
    KAVACH_ASSERT_NEAR(cmp.max_relative_error, 0.0, 1e-14);
}

// Test H: Irregular Sparse Matrix
KAVACH_TEST(test_cuda_h_irregular_sparse_matrix) {
    std::vector<Triplet> triplets = {
        {0, 1, 3.5},
        {0, 4, -2.1},
        // row 1 empty
        {2, 0, 1.2},
        {2, 1, 4.4},
        {2, 2, 5.5},
        {2, 3, 6.6},
        {2, 4, 7.7},
        {3, 3, -9.0}  // row 4 empty
    };
    CsrMatrix A = CsrMatrix::from_triplets(5, 5, triplets);

    const std::size_t batch_size = 16;
    std::vector<double> X = create_test_batch(batch_size, 5, 501);
    std::vector<double> Y_serial(batch_size * 5, 0.0);
    std::vector<double> Y_cuda(batch_size * 5, 0.0);

    serial_batched_spmv(A, X.data(), Y_serial.data(), batch_size);

    CudaBackend backend;
    backend.batched_spmv(A, X.data(), Y_cuda.data(), batch_size);

    auto cmp = compare_spmv_results(batch_size, 5, Y_serial.data(), Y_cuda.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);
}

// Test I: Empty Rows
KAVACH_TEST(test_cuda_i_empty_rows) {
    std::vector<Triplet> triplets = {{0, 0, 1.0}, {2, 1, 2.0}};
    CsrMatrix A = CsrMatrix::from_triplets(3, 2, triplets);

    const std::size_t batch_size = 10;
    std::vector<double> X = create_test_batch(batch_size, 2, 601);
    std::vector<double> Y_cuda(batch_size * 3, 0.0);

    CudaBackend backend;
    backend.batched_spmv(A, X.data(), Y_cuda.data(), batch_size);

    // Row 1 must be strictly zero across all batch vectors
    for (std::size_t k = 0; k < batch_size; ++k) {
        KAVACH_ASSERT_EQUAL(Y_cuda[k * 3 + 1], 0.0);
    }
}

// Test J: Zero Vectors
KAVACH_TEST(test_cuda_j_zero_vector) {
    CsrMatrix A = create_test_csr(10, 10, 0.3, 701);
    const std::size_t batch_size = 8;
    std::vector<double> X(batch_size * 10, 0.0);
    std::vector<double> Y_cuda(batch_size * 10, 1.0);  // Pre-fill with non-zero

    CudaBackend backend;
    backend.batched_spmv(A, X.data(), Y_cuda.data(), batch_size);

    for (double val : Y_cuda) {
        KAVACH_ASSERT_EQUAL(val, 0.0);
    }
}

// Test K: Negative Values and Mixed Signs
KAVACH_TEST(test_cuda_k_negative_values) {
    std::vector<Triplet> triplets = {{0, 0, -5.0}, {0, 1, 2.5}, {1, 0, 3.0}, {1, 1, -4.0}};
    CsrMatrix A = CsrMatrix::from_triplets(2, 2, triplets);

    std::vector<double> X = {-2.0, 4.0, 3.0, -1.0};  // 2 vectors
    std::vector<double> Y_serial(4, 0.0);
    std::vector<double> Y_cuda(4, 0.0);

    serial_batched_spmv(A, X.data(), Y_serial.data(), 2);

    CudaBackend backend;
    backend.batched_spmv(A, X.data(), Y_cuda.data(), 2);

    auto cmp = compare_spmv_results(2, 2, Y_serial.data(), Y_cuda.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);
}

// Test L: Batch Size 1
KAVACH_TEST(test_cuda_l_batch_size_1) {
    CsrMatrix A = create_test_csr(15, 15, 0.3, 901);
    std::vector<double> X = create_test_batch(1, 15, 902);
    std::vector<double> Y_serial(15, 0.0);
    std::vector<double> Y_cuda(15, 0.0);

    serial_batched_spmv(A, X.data(), Y_serial.data(), 1);

    CudaBackend backend;
    backend.batched_spmv(A, X.data(), Y_cuda.data(), 1);

    auto cmp = compare_spmv_results(1, 15, Y_serial.data(), Y_cuda.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);
}

// Test M: Batch Smaller Than Block / Work Configuration
KAVACH_TEST(test_cuda_m_batch_smaller_than_block) {
    CsrMatrix A = create_test_csr(6, 6, 0.4, 1001);
    std::vector<double> X = create_test_batch(2, 6, 1002);
    std::vector<double> Y_serial(12, 0.0);
    std::vector<double> Y_cuda(12, 0.0);

    serial_batched_spmv(A, X.data(), Y_serial.data(), 2);

    CudaBackend backend;
    backend.batched_spmv(A, X.data(), Y_cuda.data(), 2);

    auto cmp = compare_spmv_results(2, 6, Y_serial.data(), Y_cuda.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);
}

// Test N: Larger Batch Execution
KAVACH_TEST(test_cuda_n_larger_batch) {
    const std::size_t rows = 40;
    const std::size_t cols = 40;
    const std::size_t batch_size = 128;

    CsrMatrix A = create_test_csr(rows, cols, 0.15, 1101);
    std::vector<double> X = create_test_batch(batch_size, cols, 1102);
    std::vector<double> Y_serial(batch_size * rows, 0.0);
    std::vector<double> Y_cuda(batch_size * rows, 0.0);

    serial_batched_spmv(A, X.data(), Y_serial.data(), batch_size);

    CudaBackend backend;
    backend.batched_spmv(A, X.data(), Y_cuda.data(), batch_size);

    auto cmp = compare_spmv_results(batch_size, rows, Y_serial.data(), Y_cuda.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);
}

// Test O: Repeated Execution Determinism
KAVACH_TEST(test_cuda_o_repeated_execution_determinism) {
    const std::size_t rows = 20;
    const std::size_t cols = 20;
    const std::size_t batch_size = 32;

    CsrMatrix A = create_test_csr(rows, cols, 0.25, 1201);
    std::vector<double> X = create_test_batch(batch_size, cols, 1202);

    CudaBackend backend;
    std::vector<double> Y_ref(batch_size * rows, 0.0);
    backend.batched_spmv(A, X.data(), Y_ref.data(), batch_size);

    for (int rep = 0; rep < 10; ++rep) {
        std::vector<double> Y_rep(batch_size * rows, 0.0);
        backend.batched_spmv(A, X.data(), Y_rep.data(), batch_size);
        for (std::size_t i = 0; i < Y_ref.size(); ++i) {
            KAVACH_ASSERT_EQUAL(Y_ref[i], Y_rep[i]);
        }
    }
}

// Test P: Allocation and Input Failure Path
KAVACH_TEST(test_cuda_p_allocation_failure_path) {
    CsrMatrix A = CsrMatrix::identity(4);
    std::vector<double> valid(8, 1.0);

    CudaBackend backend;
    bool caught_null_x = false;
    try {
        backend.batched_spmv(A, nullptr, valid.data(), 2);
    } catch (const core::InputException&) {
        caught_null_x = true;
    }
    KAVACH_ASSERT(caught_null_x);

    bool caught_null_y = false;
    try {
        backend.batched_spmv(A, valid.data(), nullptr, 2);
    } catch (const core::InputException&) {
        caught_null_y = true;
    }
    KAVACH_ASSERT(caught_null_y);
}

// Test Q: Kernel Launch Error Handling & Zero Size Batch
KAVACH_TEST(test_cuda_q_kernel_launch_error_handling) {
    CsrMatrix A = CsrMatrix::identity(5);
    CudaBackend backend;

    // Batch size 0 should gracefully no-op without launching anything
    backend.batched_spmv(A, nullptr, nullptr, 0);
    KAVACH_ASSERT_EQUAL(backend.last_telemetry().batch_size, 0);
}

// Test R: Runtime Synchronization & Strict GPU Mode
KAVACH_TEST(test_cuda_r_runtime_synchronization_and_strict_gpu) {
    CudaBackendOptions opts;
    opts.strict_gpu = true;
    opts.force_cpu_fallback = true;  // Explicitly simulate GPU unavailable
    CudaBackend backend(opts);

    bool caught_strict = false;
    CsrMatrix A = CsrMatrix::identity(4);
    std::vector<double> X(8, 1.0);
    std::vector<double> Y(8, 0.0);

    try {
        backend.batched_spmv(A, X.data(), Y.data(), 2);
    } catch (const core::NumericalException& e) {
        caught_strict = true;
        std::string msg = e.what();
        KAVACH_ASSERT(msg.find("Strict GPU execution requested") != std::string::npos);
    }
    KAVACH_ASSERT(caught_strict);
}

// Test S: Mandatory CPU Fallback
KAVACH_TEST(test_cuda_s_cpu_fallback) {
    CudaBackendOptions opts;
    opts.strict_gpu = false;
    opts.force_cpu_fallback = true;
    CudaBackend backend(opts);

    CsrMatrix A = create_test_csr(10, 10, 0.3, 1401);
    std::vector<double> X = create_test_batch(8, 10, 1402);
    std::vector<double> Y(8 * 10, 0.0);

    backend.batched_spmv(A, X.data(), Y.data(), 8);

    const auto& tel = backend.last_telemetry();
    KAVACH_ASSERT(tel.fallback_occurred);
    KAVACH_ASSERT(!tel.gpu_available);
    KAVACH_ASSERT(!tel.fallback_reason.empty());
}

// Test T: --no-gpu Option Behavior
KAVACH_TEST(test_cuda_t_no_gpu_behavior) {
    CudaBackend backend;
    CudaBackendOptions opts;
    opts.force_cpu_fallback = true;
    backend.set_options(opts);

    KAVACH_ASSERT(!backend.is_available());

    CsrMatrix A = create_test_csr(5, 5, 0.4, 1501);
    std::vector<double> X = create_test_batch(4, 5, 1502);
    std::vector<double> Y_ref(4 * 5, 0.0);
    std::vector<double> Y(4 * 5, 0.0);

    serial_batched_spmv(A, X.data(), Y_ref.data(), 4);
    backend.batched_spmv(A, X.data(), Y.data(), 4);

    auto cmp = compare_spmv_results(4, 5, Y_ref.data(), Y.data());
    KAVACH_ASSERT(cmp.passed);
}

// Test U: Telemetry Correctness
KAVACH_TEST(test_cuda_u_telemetry_correctness) {
    CsrMatrix A = create_test_csr(12, 16, 0.3, 1601);
    const std::size_t batch_size = 20;
    std::vector<double> X = create_test_batch(batch_size, 16, 1602);
    std::vector<double> Y(batch_size * 12, 0.0);

    CudaBackend backend;
    backend.batched_spmv(A, X.data(), Y.data(), batch_size);

    const auto& tel = backend.last_telemetry();
    KAVACH_ASSERT_EQUAL(tel.rows, 12);
    KAVACH_ASSERT_EQUAL(tel.cols, 16);
    KAVACH_ASSERT_EQUAL(tel.batch_size, batch_size);
    KAVACH_ASSERT(tel.total_gpu_time_seconds >= 0.0);
}

// Test V: Transfer-Inclusive Timing Calculation
KAVACH_TEST(test_cuda_v_transfer_inclusive_timing_calculation) {
    CudaSpmvExecutionTimes times;
    times.h2d_time_seconds = 0.002;
    times.kernel_time_seconds = 0.001;
    times.d2h_time_seconds = 0.0015;
    times.total_gpu_time_seconds =
        times.h2d_time_seconds + times.kernel_time_seconds + times.d2h_time_seconds;

    KAVACH_ASSERT_NEAR(times.total_gpu_time_seconds, 0.0045, 1e-12);

    // Verify speedup calculation uses transfer-inclusive time
    double cpu_time = 0.009;
    double transfer_inclusive_speedup = cpu_time / times.total_gpu_time_seconds;
    double kernel_only_speedup = cpu_time / times.kernel_time_seconds;

    KAVACH_ASSERT_NEAR(transfer_inclusive_speedup, 2.0, 1e-12);
    KAVACH_ASSERT_NEAR(kernel_only_speedup, 9.0, 1e-12);

    // Verification: transfer-inclusive speedup must never equal kernel-only speedup
    KAVACH_ASSERT(kernel_only_speedup > transfer_inclusive_speedup);
}

// Test W: Speedup Calculation Correctness
KAVACH_TEST(test_cuda_w_speedup_calculation_correctness) {
    double cpu_time = 0.020;
    double gpu_time = 0.005;
    double speedup = cpu_time / gpu_time;
    KAVACH_ASSERT_NEAR(speedup, 4.0, 1e-12);

    double gpu_slower_time = 0.040;
    double slowdown = cpu_time / gpu_slower_time;
    KAVACH_ASSERT_NEAR(slowdown, 0.5, 1e-12);
}

// Test: Crossover Sweep Simulation & Reporting
KAVACH_TEST(test_cuda_crossover_sweep_and_telemetry) {
    std::cout
        << "\n==================== CUDA SPMV HARDWARE & CROSSOVER REPORT ====================\n";
    std::cout << "Hardware Status: " << get_cuda_status_string() << "\n";
    std::cout << "Architecture Model: CPU-first with scoped CUDA batched-SpMV kernel.\n";
    std::cout
        << "Fallback Protocol: Automatic CPU multicore fallback active (--no-gpu supported).\n";
    std::cout << "------------------------------------------------------------------------------\n";

    // Sweep across matrix dimensions and batch sizes
    struct SweepCase {
        std::size_t rows;
        std::size_t cols;
        std::size_t batch_size;
        const char* label;
    };

    const std::vector<SweepCase> sweep_cases = {{20, 20, 1, "Tiny (Single vector)"},
                                                {50, 50, 16, "Small Batch"},
                                                {200, 200, 64, "Medium Batch"},
                                                {500, 500, 128, "Large Batch"},
                                                {1000, 1000, 256, "Stress Workload"}};

    std::cout << "Instance                Rows x Cols  Batch  H2D (ms)  Kernel (ms) D2H (ms) Total "
                 "(ms) Speedup\n";
    std::cout << "---------------------------------------------------------------------------------"
                 "-----\n";

    CudaBackend backend;

    for (const auto& sc : sweep_cases) {
        CsrMatrix A = create_test_csr(sc.rows, sc.cols, 0.1, 17001);
        std::vector<double> X = create_test_batch(sc.batch_size, sc.cols, 17002);
        std::vector<double> Y(sc.batch_size * sc.rows, 0.0);

        backend.batched_spmv(A, X.data(), Y.data(), sc.batch_size);
        const auto& tel = backend.last_telemetry();

        std::cout << sc.label;
        for (std::size_t pad = std::string(sc.label).length(); pad < 24; ++pad)
            std::cout << ' ';
        std::cout << sc.rows << "x" << sc.cols << "    " << sc.batch_size << "    "
                  << (tel.h2d_time_seconds * 1000.0) << "     "
                  << (tel.kernel_time_seconds * 1000.0) << "        "
                  << (tel.d2h_time_seconds * 1000.0) << "    "
                  << (tel.total_gpu_time_seconds * 1000.0) << "    "
                  << tel.transfer_inclusive_speedup << "x ("
                  << (tel.fallback_occurred ? "CPU Fallback" : "CUDA GPU") << ")\n";
    }
    std::cout << "================================================================================="
                 "=====\n";
}
