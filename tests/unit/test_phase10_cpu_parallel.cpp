#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "../test_framework.hpp"
#include "backend/cpu/cpu_backend.hpp"
#include "backend/cpu/thread_pool.hpp"
#include "backend/execution_backend.hpp"
#include "core/error.hpp"
#include "core/logger.hpp"
#include "core/random.hpp"
#include "numerics/sparse_matrix.hpp"

using namespace kavach;
using namespace kavach::backend;
using namespace kavach::backend::cpu;
using namespace kavach::numerics;

namespace {

// Helper to generate a reproducible sparse matrix
CsrMatrix create_test_csr(std::size_t rows, std::size_t cols, double density, uint64_t seed) {
    core::DeterministicRNG rng(seed);
    std::vector<Triplet> triplets;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            if (rng.uniform_real() < density) {
                triplets.push_back({r, c, rng.uniform_real() * 10.0 - 5.0});
            }
        }
    }
    // Ensure every row has at least one entry for general test matrices
    for (std::size_t r = 0; r < rows; ++r) {
        triplets.push_back({r, r % cols, 1.5 + static_cast<double>(r % 7)});
    }
    return CsrMatrix::from_triplets(rows, cols, triplets);
}

// Helper to generate a batch of random vectors in contiguous storage
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
// Test Matrix: Tests A through P + Additional Backend Tests
// ============================================================================

// Test A: Serial Equivalence
KAVACH_TEST(test_cpu_parallel_a_serial_equivalence) {
    const std::size_t rows = 20;
    const std::size_t cols = 25;
    const std::size_t batch_size = 40;

    CsrMatrix A = create_test_csr(rows, cols, 0.3, 1001);
    std::vector<double> X = create_test_batch(batch_size, cols, 1002);
    std::vector<double> Y_serial(batch_size * rows, 0.0);
    std::vector<double> Y_parallel(batch_size * rows, 0.0);

    serial_batched_spmv(A, X.data(), Y_serial.data(), batch_size);

    BatchedSpmvOptions options;
    options.run_parallel = true;
    BatchedSpmvTelemetry tel;
    parallel_batched_spmv(A, X.data(), Y_parallel.data(), batch_size, options, &tel);

    auto cmp = compare_spmv_results(batch_size, rows, Y_serial.data(), Y_parallel.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);
    KAVACH_ASSERT(tel.parallel_enabled);
    KAVACH_ASSERT_EQUAL(tel.batch_size, batch_size);
}

// Test B: 1-Worker Equivalence
KAVACH_TEST(test_cpu_parallel_b_1_worker_equivalence) {
    const std::size_t rows = 15;
    const std::size_t cols = 15;
    const std::size_t batch_size = 20;

    CsrMatrix A = create_test_csr(rows, cols, 0.25, 2001);
    std::vector<double> X = create_test_batch(batch_size, cols, 2002);
    std::vector<double> Y_serial(batch_size * rows, 0.0);
    std::vector<double> Y_parallel(batch_size * rows, 0.0);

    serial_batched_spmv(A, X.data(), Y_serial.data(), batch_size);

    BatchedSpmvOptions options;
    options.num_workers = 1;
    parallel_batched_spmv(A, X.data(), Y_parallel.data(), batch_size, options);

    auto cmp = compare_spmv_results(batch_size, rows, Y_serial.data(), Y_parallel.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);
}

// Test C: 2-Worker Equivalence
KAVACH_TEST(test_cpu_parallel_c_2_worker_equivalence) {
    const std::size_t rows = 25;
    const std::size_t cols = 30;
    const std::size_t batch_size = 50;

    CsrMatrix A = create_test_csr(rows, cols, 0.2, 3001);
    std::vector<double> X = create_test_batch(batch_size, cols, 3002);
    std::vector<double> Y_serial(batch_size * rows, 0.0);
    std::vector<double> Y_parallel(batch_size * rows, 0.0);

    serial_batched_spmv(A, X.data(), Y_serial.data(), batch_size);

    BatchedSpmvOptions options;
    options.num_workers = 2;
    parallel_batched_spmv(A, X.data(), Y_parallel.data(), batch_size, options);

    auto cmp = compare_spmv_results(batch_size, rows, Y_serial.data(), Y_parallel.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);
}

// Test D: 4-Worker Equivalence
KAVACH_TEST(test_cpu_parallel_d_4_worker_equivalence) {
    const std::size_t rows = 35;
    const std::size_t cols = 40;
    const std::size_t batch_size = 80;

    CsrMatrix A = create_test_csr(rows, cols, 0.15, 4001);
    std::vector<double> X = create_test_batch(batch_size, cols, 4002);
    std::vector<double> Y_serial(batch_size * rows, 0.0);
    std::vector<double> Y_parallel(batch_size * rows, 0.0);

    serial_batched_spmv(A, X.data(), Y_serial.data(), batch_size);

    BatchedSpmvOptions options;
    options.num_workers = 4;
    parallel_batched_spmv(A, X.data(), Y_parallel.data(), batch_size, options);

    auto cmp = compare_spmv_results(batch_size, rows, Y_serial.data(), Y_parallel.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);
}

// Test E: Hardware-Count Equivalence
KAVACH_TEST(test_cpu_parallel_e_hardware_count_equivalence) {
    const std::size_t rows = 30;
    const std::size_t cols = 30;
    const std::size_t batch_size = 64;

    CsrMatrix A = create_test_csr(rows, cols, 0.2, 5001);
    std::vector<double> X = create_test_batch(batch_size, cols, 5002);
    std::vector<double> Y_serial(batch_size * rows, 0.0);
    std::vector<double> Y_parallel(batch_size * rows, 0.0);

    serial_batched_spmv(A, X.data(), Y_serial.data(), batch_size);

    BatchedSpmvOptions options;
    options.num_workers = 0;  // Auto-detect hardware concurrency
    BatchedSpmvTelemetry tel;
    parallel_batched_spmv(A, X.data(), Y_parallel.data(), batch_size, options, &tel);

    auto cmp = compare_spmv_results(batch_size, rows, Y_serial.data(), Y_parallel.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);
    KAVACH_ASSERT(tel.worker_count >= 1);
}

// Test F: Batch Partition Correctness
KAVACH_TEST(test_cpu_parallel_f_batch_partition_correctness) {
    // 1. Standard even partition
    auto chunks = ThreadPool::partition_batch(100, 4);
    KAVACH_ASSERT_EQUAL(chunks.size(), 4);
    KAVACH_ASSERT_EQUAL(chunks[0].start, 0);
    KAVACH_ASSERT_EQUAL(chunks[0].end, 25);
    KAVACH_ASSERT_EQUAL(chunks[1].start, 25);
    KAVACH_ASSERT_EQUAL(chunks[1].end, 50);
    KAVACH_ASSERT_EQUAL(chunks[2].start, 50);
    KAVACH_ASSERT_EQUAL(chunks[2].end, 75);
    KAVACH_ASSERT_EQUAL(chunks[3].start, 75);
    KAVACH_ASSERT_EQUAL(chunks[3].end, 100);

    // 2. Uneven partition (7 items, 3 chunks)
    chunks = ThreadPool::partition_batch(7, 3);
    KAVACH_ASSERT_EQUAL(chunks.size(), 3);
    std::size_t covered = 0;
    for (std::size_t i = 0; i < chunks.size(); ++i) {
        KAVACH_ASSERT_EQUAL(chunks[i].start, covered);
        covered = chunks[i].end;
        KAVACH_ASSERT(chunks[i].size() >= 2 && chunks[i].size() <= 3);
    }
    KAVACH_ASSERT_EQUAL(covered, 7);

    // 3. Batch smaller than requested chunks (3 items, 5 chunks)
    chunks = ThreadPool::partition_batch(3, 5);
    KAVACH_ASSERT_EQUAL(chunks.size(), 3);
    KAVACH_ASSERT_EQUAL(chunks[0].start, 0);
    KAVACH_ASSERT_EQUAL(chunks[0].end, 1);
    KAVACH_ASSERT_EQUAL(chunks[1].start, 1);
    KAVACH_ASSERT_EQUAL(chunks[1].end, 2);
    KAVACH_ASSERT_EQUAL(chunks[2].start, 2);
    KAVACH_ASSERT_EQUAL(chunks[2].end, 3);

    // 4. Empty batch
    chunks = ThreadPool::partition_batch(0, 4);
    KAVACH_ASSERT(chunks.empty());

    // 5. Zero chunks requested
    chunks = ThreadPool::partition_batch(10, 0);
    KAVACH_ASSERT(chunks.empty());

    // 6. Single item
    chunks = ThreadPool::partition_batch(1, 1);
    KAVACH_ASSERT_EQUAL(chunks.size(), 1);
    KAVACH_ASSERT_EQUAL(chunks[0].start, 0);
    KAVACH_ASSERT_EQUAL(chunks[0].end, 1);
}

// Test G: Empty Batch Handling
KAVACH_TEST(test_cpu_parallel_g_empty_batch) {
    CsrMatrix A = CsrMatrix::identity(5);
    double* null_x = nullptr;
    double* null_y = nullptr;

    BatchedSpmvTelemetry tel;
    parallel_batched_spmv(A, null_x, null_y, 0, {}, &tel);
    KAVACH_ASSERT_EQUAL(tel.batch_size, 0);
    KAVACH_ASSERT_EQUAL(tel.rows, 5);
    KAVACH_ASSERT_EQUAL(tel.cols, 5);
}

// Test H: Batch Smaller Than Workers
KAVACH_TEST(test_cpu_parallel_h_batch_smaller_than_workers) {
    const std::size_t rows = 10;
    const std::size_t cols = 10;
    const std::size_t batch_size = 2;

    CsrMatrix A = create_test_csr(rows, cols, 0.4, 8001);
    std::vector<double> X = create_test_batch(batch_size, cols, 8002);
    std::vector<double> Y_serial(batch_size * rows, 0.0);
    std::vector<double> Y_parallel(batch_size * rows, 0.0);

    serial_batched_spmv(A, X.data(), Y_serial.data(), batch_size);

    BatchedSpmvOptions options;
    options.num_workers = 8;  // 8 workers for 2 items
    parallel_batched_spmv(A, X.data(), Y_parallel.data(), batch_size, options);

    auto cmp = compare_spmv_results(batch_size, rows, Y_serial.data(), Y_parallel.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);
}

// Test I: Irregular Sparse Matrix
KAVACH_TEST(test_cpu_parallel_i_irregular_sparse_matrix) {
    // 5 rows: row 0 empty, row 1 dense, row 2 singleton, row 3 empty, row 4 partial
    std::vector<Triplet> triplets = {{1, 0, 1.0}, {1, 1, 2.0}, {1, 2, 3.0}, {1, 3, 4.0},
                                     {2, 2, 5.0}, {4, 0, 6.0}, {4, 3, 7.0}};
    CsrMatrix A = CsrMatrix::from_triplets(5, 4, triplets);

    const std::size_t batch_size = 33;
    std::vector<double> X = create_test_batch(batch_size, 4, 9001);
    std::vector<double> Y_serial(batch_size * 5, 0.0);
    std::vector<double> Y_parallel(batch_size * 5, 0.0);

    serial_batched_spmv(A, X.data(), Y_serial.data(), batch_size);

    BatchedSpmvOptions options;
    options.num_workers = 4;
    parallel_batched_spmv(A, X.data(), Y_parallel.data(), batch_size, options);

    auto cmp = compare_spmv_results(batch_size, 5, Y_serial.data(), Y_parallel.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);
}

// Test J: Transpose Batched SpMV
KAVACH_TEST(test_cpu_parallel_j_transpose_batched_spmv) {
    const std::size_t rows = 20;
    const std::size_t cols = 15;
    const std::size_t batch_size = 40;

    CsrMatrix A = create_test_csr(rows, cols, 0.3, 10001);
    // For transpose, input vectors are of size rows, output of size cols
    std::vector<double> X = create_test_batch(batch_size, rows, 10002);
    std::vector<double> Y_serial(batch_size * cols, 0.0);
    std::vector<double> Y_parallel(batch_size * cols, 0.0);

    serial_batched_spmv(A, X.data(), Y_serial.data(), batch_size, /*transpose=*/true);

    BatchedSpmvOptions options;
    options.transpose = true;
    options.num_workers = 4;
    parallel_batched_spmv(A, X.data(), Y_parallel.data(), batch_size, options);

    auto cmp = compare_spmv_results(batch_size, cols, Y_serial.data(), Y_parallel.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);
}

// Test K: Invalid Dimensions and Null Pointers
KAVACH_TEST(test_cpu_parallel_k_invalid_dimensions) {
    CsrMatrix A = CsrMatrix::identity(5);
    std::vector<double> valid(10, 1.0);

    // Null pointer for X with batch_size > 0
    bool caught_null_x = false;
    try {
        parallel_batched_spmv(A, nullptr, valid.data(), 2);
    } catch (const core::InputException&) {
        caught_null_x = true;
    }
    KAVACH_ASSERT(caught_null_x);

    // Null pointer for Y with batch_size > 0
    bool caught_null_y = false;
    try {
        parallel_batched_spmv(A, valid.data(), nullptr, 2);
    } catch (const core::InputException&) {
        caught_null_y = true;
    }
    KAVACH_ASSERT(caught_null_y);

    // Vector-of-vectors dimension mismatch
    std::vector<std::vector<double>> bad_x = {{1.0, 2.0}, {1.0}};  // Inconsistent dimensions
    std::vector<std::vector<double>> out_y;
    bool caught_vec_mismatch = false;
    try {
        parallel_batched_spmv(A, bad_x, out_y);
    } catch (const core::InputException&) {
        caught_vec_mismatch = true;
    }
    KAVACH_ASSERT(caught_vec_mismatch);
}

// Test L: Deterministic Repeated Execution
KAVACH_TEST(test_cpu_parallel_l_deterministic_repeated_execution) {
    const std::size_t rows = 30;
    const std::size_t cols = 30;
    const std::size_t batch_size = 50;

    CsrMatrix A = create_test_csr(rows, cols, 0.25, 12001);
    std::vector<double> X = create_test_batch(batch_size, cols, 12002);

    std::vector<double> Y_run1(batch_size * rows, 0.0);
    BatchedSpmvOptions options;
    options.num_workers = 4;
    parallel_batched_spmv(A, X.data(), Y_run1.data(), batch_size, options);

    // 10 repeated runs must produce identical bits
    for (int rep = 0; rep < 10; ++rep) {
        std::vector<double> Y_rep(batch_size * rows, 0.0);
        parallel_batched_spmv(A, X.data(), Y_rep.data(), batch_size, options);
        for (std::size_t i = 0; i < Y_run1.size(); ++i) {
            KAVACH_ASSERT_EQUAL(Y_run1[i], Y_rep[i]);
        }
    }
}

// Test M: Exception Propagation
KAVACH_TEST(test_cpu_parallel_m_exception_propagation) {
    ThreadPool pool(2);

    bool exception_caught = false;
    try {
        pool.parallel_for_chunks(10, 2, [](std::size_t start, std::size_t) {
            if (start == 0) {
                throw core::NumericalException("Injected worker failure for testing");
            }
        });
    } catch (const core::NumericalException& e) {
        exception_caught = true;
        std::string msg = e.what();
        KAVACH_ASSERT(msg.find("Injected worker failure") != std::string::npos);
    }
    KAVACH_ASSERT(exception_caught);

    // Verify pool remains healthy and can execute subsequent tasks normally
    std::atomic<int> counter{0};
    pool.parallel_for_chunks(10, 2, [&counter](std::size_t, std::size_t) { counter.fetch_add(1); });
    KAVACH_ASSERT_EQUAL(counter.load(), 2);
}

// Test N: Graceful Pool Shutdown
KAVACH_TEST(test_cpu_parallel_n_graceful_pool_shutdown) {
    // Construct and immediately shutdown
    {
        ThreadPool pool(4);
        pool.shutdown();
        KAVACH_ASSERT(pool.is_stopped());
    }

    // Submit asynchronous tasks and allow destructor to join
    {
        ThreadPool pool(4);
        std::vector<std::future<int>> futures;
        for (int i = 0; i < 20; ++i) {
            futures.push_back(pool.submit([i]() {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                return i * i;
            }));
        }
        for (int i = 0; i < 20; ++i) {
            KAVACH_ASSERT_EQUAL(futures[static_cast<std::size_t>(i)].get(), i * i);
        }
    }
}

// Test O: Repeated Pool Reuse
KAVACH_TEST(test_cpu_parallel_o_repeated_pool_reuse) {
    const std::size_t rows = 15;
    const std::size_t cols = 15;
    const std::size_t batch_size = 20;

    CsrMatrix A = create_test_csr(rows, cols, 0.3, 15001);
    std::vector<double> X = create_test_batch(batch_size, cols, 15002);
    std::vector<double> Y_ref(batch_size * rows, 0.0);
    serial_batched_spmv(A, X.data(), Y_ref.data(), batch_size);

    ThreadPool shared_pool(4);
    BatchedSpmvOptions options;
    options.num_workers = 4;

    for (int iter = 0; iter < 50; ++iter) {
        std::vector<double> Y(batch_size * rows, 0.0);
        parallel_batched_spmv(A, X.data(), Y.data(), batch_size, options, nullptr, &shared_pool);
        auto cmp = compare_spmv_results(batch_size, rows, Y_ref.data(), Y.data());
        KAVACH_ASSERT(cmp.passed);
    }
}

// Test P: Large Sparse Stress Workload & Benchmark Speedup
KAVACH_TEST(test_cpu_parallel_p_large_sparse_stress_workload) {
    const std::size_t rows = 500;
    const std::size_t cols = 500;
    const std::size_t batch_size = 128;
    const int repetitions = 5;

    // Banded 5-diagonal matrix (500x500, ~2500 NNZ)
    std::vector<Triplet> triplets;
    for (std::size_t i = 0; i < rows; ++i) {
        triplets.push_back({i, i, 4.0});
        if (i + 1 < cols)
            triplets.push_back({i, i + 1, -1.0});
        if (i >= 1)
            triplets.push_back({i, i - 1, -1.0});
        if (i + 5 < cols)
            triplets.push_back({i, i + 5, 0.5});
        if (i >= 5)
            triplets.push_back({i, i - 5, 0.5});
    }
    CsrMatrix A = CsrMatrix::from_triplets(rows, cols, triplets);
    std::vector<double> X = create_test_batch(batch_size, cols, 16001);

    // 1. Serial Baseline Measurement
    std::vector<double> Y_serial(batch_size * rows, 0.0);
    std::vector<double> serial_times;
    serial_times.reserve(repetitions);

    for (int r = 0; r < repetitions; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        serial_batched_spmv(A, X.data(), Y_serial.data(), batch_size);
        auto t1 = std::chrono::steady_clock::now();
        serial_times.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::sort(serial_times.begin(), serial_times.end());
    const double median_serial_sec = serial_times[repetitions / 2];

    // 2. Multicore Parallel Measurements (across 1, 2, 4, 8 workers)
    const std::vector<std::size_t> worker_configs = {1, 2, 4, 8};
    std::cout
        << "\n==================== BATCHED SPMV CPU MULTICORE BENCHMARK ====================\n";
    std::cout << "Matrix: " << rows << " x " << cols << " (" << A.nonzeros()
              << " nnz), Batch Size: " << batch_size << ", Repetitions: " << repetitions << "\n";
    std::cout << "Serial Median Time: " << (median_serial_sec * 1000.0) << " ms\n";
    std::cout << "------------------------------------------------------------------------------\n";
    std::cout << "Workers  Parallel (ms)  Min (ms)  Max (ms)  Speedup   Efficiency  Verification\n";
    std::cout << "------------------------------------------------------------------------------\n";

    for (std::size_t workers : worker_configs) {
        ThreadPool pool(workers);
        BatchedSpmvOptions opts;
        opts.num_workers = workers;
        opts.run_parallel = (workers > 1);

        std::vector<double> Y_parallel(batch_size * rows, 0.0);
        std::vector<double> par_times;
        par_times.reserve(repetitions);

        for (int r = 0; r < repetitions; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            parallel_batched_spmv(A, X.data(), Y_parallel.data(), batch_size, opts, nullptr, &pool);
            auto t1 = std::chrono::steady_clock::now();
            par_times.push_back(std::chrono::duration<double>(t1 - t0).count());
        }

        std::sort(par_times.begin(), par_times.end());
        const double median_par_sec = par_times[repetitions / 2];
        const double min_par_sec = par_times.front();
        const double max_par_sec = par_times.back();
        const double speedup = median_serial_sec / median_par_sec;
        const double efficiency = speedup / static_cast<double>(workers);

        auto cmp = compare_spmv_results(batch_size, rows, Y_serial.data(), Y_parallel.data());
        KAVACH_ASSERT(cmp.passed);
        KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);

        std::cout << workers << "        " << (median_par_sec * 1000.0) << "       "
                  << (min_par_sec * 1000.0) << "    " << (max_par_sec * 1000.0) << "   " << speedup
                  << "x     " << (efficiency * 100.0) << "%       "
                  << (cmp.passed ? "PASSED (0 err)" : "FAILED") << "\n";
    }

    // Workload 2: Stress / Large (1000x1000, batch_size=512)
    const std::size_t rows2 = 1000;
    const std::size_t cols2 = 1000;
    const std::size_t batch_size2 = 512;
    std::vector<Triplet> triplets2;
    for (std::size_t i = 0; i < rows2; ++i) {
        triplets2.push_back({i, i, 5.0});
        if (i + 1 < cols2)
            triplets2.push_back({i, i + 1, -1.0});
        if (i >= 1)
            triplets2.push_back({i, i - 1, -1.0});
        if (i + 2 < cols2)
            triplets2.push_back({i, i + 2, 0.5});
        if (i >= 2)
            triplets2.push_back({i, i - 2, 0.5});
        if (i + 10 < cols2)
            triplets2.push_back({i, i + 10, -0.2});
        if (i >= 10)
            triplets2.push_back({i, i - 10, -0.2});
    }
    CsrMatrix A2 = CsrMatrix::from_triplets(rows2, cols2, triplets2);
    std::vector<double> X2 = create_test_batch(batch_size2, cols2, 16002);

    std::vector<double> Y_serial2(batch_size2 * rows2, 0.0);
    std::vector<double> serial_times2;
    serial_times2.reserve(repetitions);
    for (int r = 0; r < repetitions; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        serial_batched_spmv(A2, X2.data(), Y_serial2.data(), batch_size2);
        auto t1 = std::chrono::steady_clock::now();
        serial_times2.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::sort(serial_times2.begin(), serial_times2.end());
    const double median_serial_sec2 = serial_times2[repetitions / 2];

    std::cout
        << "\n----------------- LARGE STRESS WORKLOAD (1000x1000, Batch 512) -----------------\n";
    std::cout << "Matrix: " << rows2 << " x " << cols2 << " (" << A2.nonzeros()
              << " nnz), Batch Size: " << batch_size2 << "\n";
    std::cout << "Serial Median Time: " << (median_serial_sec2 * 1000.0) << " ms\n";
    std::cout << "------------------------------------------------------------------------------\n";
    std::cout << "Workers  Parallel (ms)  Min (ms)  Max (ms)  Speedup   Efficiency  Verification\n";
    std::cout << "------------------------------------------------------------------------------\n";

    for (std::size_t workers : worker_configs) {
        ThreadPool pool(workers);
        BatchedSpmvOptions opts;
        opts.num_workers = workers;
        opts.run_parallel = (workers > 1);

        std::vector<double> Y_parallel2(batch_size2 * rows2, 0.0);
        std::vector<double> par_times2;
        par_times2.reserve(repetitions);

        for (int r = 0; r < repetitions; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            parallel_batched_spmv(A2, X2.data(), Y_parallel2.data(), batch_size2, opts, nullptr,
                                  &pool);
            auto t1 = std::chrono::steady_clock::now();
            par_times2.push_back(std::chrono::duration<double>(t1 - t0).count());
        }

        std::sort(par_times2.begin(), par_times2.end());
        const double median_par_sec2 = par_times2[repetitions / 2];
        const double min_par_sec2 = par_times2.front();
        const double max_par_sec2 = par_times2.back();
        const double speedup2 = median_serial_sec2 / median_par_sec2;
        const double efficiency2 = speedup2 / static_cast<double>(workers);

        auto cmp2 = compare_spmv_results(batch_size2, rows2, Y_serial2.data(), Y_parallel2.data());
        KAVACH_ASSERT(cmp2.passed);
        KAVACH_ASSERT_NEAR(cmp2.max_absolute_error, 0.0, 1e-14);

        std::cout << workers << "        " << (median_par_sec2 * 1000.0) << "       "
                  << (min_par_sec2 * 1000.0) << "    " << (max_par_sec2 * 1000.0) << "   "
                  << speedup2 << "x     " << (efficiency2 * 100.0) << "%       "
                  << (cmp2.passed ? "PASSED (0 err)" : "FAILED") << "\n";
    }
    std::cout << "==============================================================================\n";
}

// Test: CSC Matrix Parallel SpMV
KAVACH_TEST(test_cpu_parallel_csc_matrix) {
    const std::size_t rows = 20;
    const std::size_t cols = 20;
    const std::size_t batch_size = 30;

    CsrMatrix csr = create_test_csr(rows, cols, 0.3, 17001);
    // Convert to CSC triplets
    std::vector<Triplet> triplets;
    for (std::size_t r = 0; r < rows; ++r) {
        const std::size_t start = csr.row_start(r);
        const std::size_t end = csr.row_end(r);
        const auto* cols_ptr = csr.row_indices(r);
        const auto* vals_ptr = csr.row_values(r);
        for (std::size_t k = 0; k < end - start; ++k) {
            triplets.push_back({r, cols_ptr[k], vals_ptr[k]});
        }
    }
    CscMatrix csc = CscMatrix::from_triplets(rows, cols, triplets);

    std::vector<double> X = create_test_batch(batch_size, cols, 17002);
    std::vector<double> Y_ref(batch_size * rows, 0.0);
    std::vector<double> Y_par(batch_size * rows, 0.0);

    for (std::size_t k = 0; k < batch_size; ++k) {
        csc.spmv(X.data() + k * cols, Y_ref.data() + k * rows);
    }

    BatchedSpmvOptions opts;
    opts.num_workers = 4;
    parallel_batched_spmv(csc, X.data(), Y_par.data(), batch_size, opts);

    auto cmp = compare_spmv_results(batch_size, rows, Y_ref.data(), Y_par.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);
}

// Test: Vector-of-Vectors Interface
KAVACH_TEST(test_cpu_parallel_vector_of_vectors) {
    const std::size_t rows = 12;
    const std::size_t cols = 16;
    const std::size_t batch_size = 25;

    CsrMatrix A = create_test_csr(rows, cols, 0.35, 18001);
    core::DeterministicRNG rng(18002);

    std::vector<std::vector<double>> batch_x(batch_size, std::vector<double>(cols));
    for (auto& vec : batch_x) {
        for (double& val : vec) {
            val = rng.uniform_real() * 10.0;
        }
    }

    std::vector<std::vector<double>> batch_y;
    BatchedSpmvOptions opts;
    opts.num_workers = 4;
    parallel_batched_spmv(A, batch_x, batch_y, opts);

    KAVACH_ASSERT_EQUAL(batch_y.size(), batch_size);
    for (std::size_t k = 0; k < batch_size; ++k) {
        KAVACH_ASSERT_EQUAL(batch_y[k].size(), rows);
        std::vector<double> expected = A.spmv(batch_x[k]);
        for (std::size_t i = 0; i < rows; ++i) {
            KAVACH_ASSERT_NEAR(batch_y[k][i], expected[i], 1e-14);
        }
    }
}

// Test: Raw Pointer Interface
KAVACH_TEST(test_cpu_parallel_raw_pointers) {
    const std::size_t rows = 10;
    const std::size_t cols = 10;
    const std::size_t batch_size = 16;

    CsrMatrix A = create_test_csr(rows, cols, 0.4, 19001);
    std::vector<double> X = create_test_batch(batch_size, cols, 19002);
    std::vector<double> Y_ref(batch_size * rows, 0.0);
    std::vector<double> Y_raw_sz(batch_size * rows, 0.0);
    std::vector<double> Y_raw_int(batch_size * rows, 0.0);

    serial_batched_spmv(A, X.data(), Y_ref.data(), batch_size);

    // 1. size_t raw pointers
    parallel_batched_spmv_raw(batch_size, rows, cols, A.values().data(), A.row_ptr().data(),
                              A.col_idx().data(), X.data(), Y_raw_sz.data());
    auto cmp1 = compare_spmv_results(batch_size, rows, Y_ref.data(), Y_raw_sz.data());
    KAVACH_ASSERT(cmp1.passed);

    // 2. int raw pointers
    std::vector<int> row_ptr_int(A.row_ptr().begin(), A.row_ptr().end());
    std::vector<int> col_idx_int(A.col_idx().begin(), A.col_idx().end());
    parallel_batched_spmv_raw(batch_size, rows, cols, A.values().data(), row_ptr_int.data(),
                              col_idx_int.data(), X.data(), Y_raw_int.data());
    auto cmp2 = compare_spmv_results(batch_size, rows, Y_ref.data(), Y_raw_int.data());
    KAVACH_ASSERT(cmp2.passed);
}

// Test: ExecutionBackend Polymorphic Interface
KAVACH_TEST(test_cpu_parallel_execution_backend_interface) {
    const std::size_t rows = 8;
    const std::size_t cols = 8;
    const std::size_t batch_size = 12;

    CsrMatrix A = create_test_csr(rows, cols, 0.5, 20001);
    std::vector<double> X = create_test_batch(batch_size, cols, 20002);
    std::vector<double> Y_seq(batch_size * rows, 0.0);
    std::vector<double> Y_par(batch_size * rows, 0.0);

    std::unique_ptr<ExecutionBackend> seq_backend = std::make_unique<CpuSequentialBackend>();
    std::unique_ptr<ExecutionBackend> par_backend = std::make_unique<CpuMulticoreBackend>(4);

    KAVACH_ASSERT_EQUAL(seq_backend->type(), BackendType::CPU_SEQUENTIAL);
    KAVACH_ASSERT(seq_backend->is_available());
    KAVACH_ASSERT_EQUAL(par_backend->type(), BackendType::CPU_MULTICORE);
    KAVACH_ASSERT(par_backend->is_available());

    seq_backend->batched_spmv(A, X.data(), Y_seq.data(), batch_size);
    par_backend->batched_spmv(A, X.data(), Y_par.data(), batch_size);

    auto cmp = compare_spmv_results(batch_size, rows, Y_seq.data(), Y_par.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);
}

// Test: Comparison Fault Injection
KAVACH_TEST(test_cpu_parallel_comparison_fault_injection) {
    const std::size_t batch_size = 5;
    const std::size_t out_dim = 4;
    std::vector<double> Y1(batch_size * out_dim, 1.0);
    std::vector<double> Y2 = Y1;

    // Inject deliberate mismatch at batch index 2, element index 3
    Y2[2 * out_dim + 3] += 0.05;

    auto cmp = compare_spmv_results(batch_size, out_dim, Y1.data(), Y2.data(), 1e-6, 1e-6);
    KAVACH_ASSERT(!cmp.passed);
    KAVACH_ASSERT_EQUAL(cmp.mismatch_batch_idx, 2);
    KAVACH_ASSERT_EQUAL(cmp.mismatch_elem_idx, 3);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.05, 1e-12);
    KAVACH_ASSERT(!cmp.error_message.empty());
}

// Test: Parallel Batched Transpose SpMV Equivalence
KAVACH_TEST(test_cpu_parallel_transpose_spmv) {
    const std::size_t rows = 15;
    const std::size_t cols = 25;
    const std::size_t batch_size = 40;

    CsrMatrix A = create_test_csr(rows, cols, 0.35, 21001);
    std::vector<double> X = create_test_batch(batch_size, rows, 21002);
    std::vector<double> Y_ser(batch_size * cols, 0.0);
    std::vector<double> Y_par(batch_size * cols, 0.0);

    serial_batched_spmv_transpose(A, X.data(), Y_ser.data(), batch_size);

    BatchedSpmvOptions opts;
    opts.num_workers = 4;
    opts.run_parallel = true;
    parallel_batched_spmv_transpose(A, X.data(), Y_par.data(), batch_size, opts);

    auto cmp = compare_spmv_results(batch_size, cols, Y_ser.data(), Y_par.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);

    // Also test via ExecutionBackend polymorphic interface
    CpuSequentialBackend seq_backend;
    CpuMulticoreBackend par_backend(4);
    std::vector<double> Y_backend(batch_size * cols, 0.0);
    par_backend.batched_spmv_transpose(A, X.data(), Y_backend.data(), batch_size);
    auto cmp_backend = compare_spmv_results(batch_size, cols, Y_ser.data(), Y_backend.data());
    KAVACH_ASSERT(cmp_backend.passed);
}

// Test: Parallel Batched CSC SpMV Equivalence
KAVACH_TEST(test_cpu_parallel_csc_spmv_equivalence) {
    const std::size_t rows = 24;
    const std::size_t cols = 18;
    const std::size_t batch_size = 32;

    CsrMatrix csr = create_test_csr(rows, cols, 0.3, 22001);
    std::vector<Triplet> triplets;
    for (std::size_t r = 0; r < rows; ++r) {
        const std::size_t start = csr.row_start(r);
        const std::size_t end = csr.row_end(r);
        const auto* cols_ptr = csr.row_indices(r);
        const auto* vals_ptr = csr.row_values(r);
        for (std::size_t k = 0; k < end - start; ++k) {
            triplets.push_back({r, cols_ptr[k], vals_ptr[k]});
        }
    }
    CscMatrix csc = CscMatrix::from_triplets(rows, cols, triplets);

    std::vector<double> X = create_test_batch(batch_size, cols, 22002);
    std::vector<double> Y_ser(batch_size * rows, 0.0);
    std::vector<double> Y_par(batch_size * rows, 0.0);

    serial_batched_csc_spmv(csc, X.data(), Y_ser.data(), batch_size);

    CpuMulticoreBackend par_backend(4);
    par_backend.batched_csc_spmv(csc, X.data(), Y_par.data(), batch_size);

    auto cmp = compare_spmv_results(batch_size, rows, Y_ser.data(), Y_par.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);
}

// Test: Parallel Batched Vector AXPBY
KAVACH_TEST(test_cpu_parallel_batched_axpby) {
    const std::size_t batch_size = 50;
    const std::size_t n = 120;
    const std::size_t total = batch_size * n;
    const double alpha = 2.5;
    const double beta = -1.2;

    std::vector<double> X = create_test_batch(batch_size, n, 23001);
    std::vector<double> Y_ser = create_test_batch(batch_size, n, 23002);
    std::vector<double> Y_par = Y_ser;

    serial_batched_axpby(batch_size, n, alpha, X.data(), beta, Y_ser.data());

    ThreadPool pool(4);
    parallel_batched_axpby(batch_size, n, alpha, X.data(), beta, Y_par.data(), &pool);

    auto cmp = compare_vector_results(total, Y_ser.data(), Y_par.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-14);

    // Also test via CpuMulticoreBackend
    std::vector<double> Y_backend = create_test_batch(batch_size, n, 23002);
    CpuMulticoreBackend backend(4);
    backend.batched_axpby(batch_size, n, alpha, X.data(), beta, Y_backend.data());
    auto cmp_backend = compare_vector_results(total, Y_ser.data(), Y_backend.data());
    KAVACH_ASSERT(cmp_backend.passed);
}

// Test: Parallel Batched Dot Product
KAVACH_TEST(test_cpu_parallel_batched_dot) {
    const std::size_t batch_size = 64;
    const std::size_t n = 200;

    std::vector<double> X = create_test_batch(batch_size, n, 24001);
    std::vector<double> Y = create_test_batch(batch_size, n, 24002);
    std::vector<double> dots_ser(batch_size, 0.0);
    std::vector<double> dots_par(batch_size, 0.0);

    serial_batched_dot(batch_size, n, X.data(), Y.data(), dots_ser.data());

    ThreadPool pool(4);
    parallel_batched_dot(batch_size, n, X.data(), Y.data(), dots_par.data(), &pool);

    auto cmp = compare_scalar_results(batch_size, dots_ser.data(), dots_par.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-13);
}

// Test: Parallel Batched L2 Norm
KAVACH_TEST(test_cpu_parallel_batched_l2_norm) {
    const std::size_t batch_size = 48;
    const std::size_t n = 150;

    std::vector<double> X = create_test_batch(batch_size, n, 25001);
    std::vector<double> norms_ser(batch_size, 0.0);
    std::vector<double> norms_par(batch_size, 0.0);

    serial_batched_l2_norm(batch_size, n, X.data(), norms_ser.data());

    ThreadPool pool(4);
    parallel_batched_l2_norm(batch_size, n, X.data(), norms_par.data(), &pool);

    auto cmp = compare_scalar_results(batch_size, norms_ser.data(), norms_par.data());
    KAVACH_ASSERT(cmp.passed);
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-13);
}

// Test: Parallel Batched Linf Norm
KAVACH_TEST(test_cpu_parallel_batched_linf_norm) {
    const std::size_t batch_size = 50;
    const std::size_t n = 100;

    std::vector<double> X = create_test_batch(batch_size, n, 26001);
    std::vector<double> norms_ser(batch_size, 0.0);
    std::vector<double> norms_par(batch_size, 0.0);

    serial_batched_linf_norm(batch_size, n, X.data(), norms_ser.data());

    ThreadPool pool(4);
    parallel_batched_linf_norm(batch_size, n, X.data(), norms_par.data(), &pool);

    auto cmp = compare_scalar_results(batch_size, norms_ser.data(), norms_par.data());
    KAVACH_ASSERT(cmp.passed);
    // Linf norm max-abs is exact and order-independent!
    KAVACH_ASSERT_NEAR(cmp.max_absolute_error, 0.0, 1e-15);
}

// Test: Vector and Scalar Comparison Utilities
KAVACH_TEST(test_cpu_parallel_vector_and_scalar_comparison_utilities) {
    std::vector<double> vec1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> vec2 = {1.0, 2.0, 3.0, 4.00000000000001};

    auto res = compare_vector_results(4, vec1.data(), vec2.data(), 1e-12, 1e-10);
    KAVACH_ASSERT(res.passed);

    vec2[3] = 4.5;
    auto fail_res = compare_vector_results(4, vec1.data(), vec2.data(), 1e-6, 1e-6);
    KAVACH_ASSERT(!fail_res.passed);
    KAVACH_ASSERT_NEAR(fail_res.max_absolute_error, 0.5, 1e-12);
}
