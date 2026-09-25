#include "backend/cpu/cpu_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>

#include "core/error.hpp"

namespace kavach::backend::cpu {

namespace {

inline void raw_spmv_internal(std::size_t rows, const double* values, const std::size_t* row_ptr,
                              const std::size_t* col_indices, const double* x, double* y) {
    for (std::size_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        const std::size_t start = row_ptr[r];
        const std::size_t end = row_ptr[r + 1];
        for (std::size_t k = start; k < end; ++k) {
            sum += values[k] * x[col_indices[k]];
        }
        y[r] = sum;
    }
}

inline void raw_spmv_int_internal(std::size_t rows, const double* values, const int* row_ptr,
                                  const int* col_indices, const double* x, double* y) {
    for (std::size_t r = 0; r < rows; ++r) {
        double sum = 0.0;
        const int start = row_ptr[r];
        const int end = row_ptr[r + 1];
        for (int k = start; k < end; ++k) {
            sum += values[k] * x[col_indices[k]];
        }
        y[r] = sum;
    }
}

}  // namespace

// ============================================================================
// Free / Utility Functions
// ============================================================================

void serial_batched_spmv(const numerics::CsrMatrix& A, const double* batch_x, double* batch_y,
                         std::size_t batch_size, bool transpose) {
    if (batch_size == 0) {
        return;
    }
    if (batch_x == nullptr || batch_y == nullptr) {
        throw core::InputException("Null pointer passed to serial_batched_spmv");
    }

    const std::size_t in_dim = transpose ? A.rows() : A.cols();
    const std::size_t out_dim = transpose ? A.cols() : A.rows();

    for (std::size_t k = 0; k < batch_size; ++k) {
        const double* x_k = batch_x + (k * in_dim);
        double* y_k = batch_y + (k * out_dim);
        if (transpose) {
            A.spmv_transpose(x_k, y_k);
        } else {
            A.spmv(x_k, y_k);
        }
    }
}

void parallel_batched_spmv(const numerics::CsrMatrix& A, const double* batch_x, double* batch_y,
                           std::size_t batch_size, const BatchedSpmvOptions& options,
                           BatchedSpmvTelemetry* telemetry, ThreadPool* custom_pool) {
    if (batch_size == 0) {
        if (telemetry != nullptr) {
            *telemetry = {};
            telemetry->rows = A.rows();
            telemetry->cols = A.cols();
            telemetry->nonzeros = A.nonzeros();
        }
        return;
    }

    if (batch_x == nullptr || batch_y == nullptr) {
        throw core::InputException("Null pointer passed for batch vectors in batched_spmv");
    }

    const std::size_t in_dim = options.transpose ? A.rows() : A.cols();
    const std::size_t out_dim = options.transpose ? A.cols() : A.rows();

    if (A.rows() == 0 || A.cols() == 0) {
        std::fill_n(batch_y, batch_size * out_dim, 0.0);
        if (telemetry != nullptr) {
            *telemetry = {};
            telemetry->rows = A.rows();
            telemetry->cols = A.cols();
            telemetry->nonzeros = A.nonzeros();
            telemetry->batch_size = batch_size;
        }
        return;
    }

    std::size_t workers = options.num_workers;
    if (workers == 0) {
        workers = static_cast<std::size_t>(std::max(1u, std::thread::hardware_concurrency()));
    }

    const bool execute_parallel = options.run_parallel && workers > 1 && batch_size > 1;
    const auto t_start = std::chrono::steady_clock::now();

    if (!execute_parallel) {
        serial_batched_spmv(A, batch_x, batch_y, batch_size, options.transpose);
    } else {
        std::unique_ptr<ThreadPool> local_pool;
        ThreadPool* pool = custom_pool;
        if (pool == nullptr) {
            local_pool = std::make_unique<ThreadPool>(workers);
            pool = local_pool.get();
        }

        pool->parallel_for_chunks(
            batch_size, pool->worker_count(),
            [&A, batch_x, batch_y, in_dim, out_dim, transpose = options.transpose](
                std::size_t start, std::size_t end) {
                for (std::size_t k = start; k < end; ++k) {
                    const double* x_k = batch_x + (k * in_dim);
                    double* y_k = batch_y + (k * out_dim);
                    if (transpose) {
                        A.spmv_transpose(x_k, y_k);
                    } else {
                        A.spmv(x_k, y_k);
                    }
                }
            });
    }

    const auto t_end = std::chrono::steady_clock::now();
    const double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();

    if (telemetry != nullptr) {
        telemetry->parallel_enabled = execute_parallel;
        telemetry->worker_count = execute_parallel ? workers : 1;
        telemetry->batch_size = batch_size;
        telemetry->rows = A.rows();
        telemetry->cols = A.cols();
        telemetry->nonzeros = A.nonzeros();
        telemetry->parallel_time_seconds = elapsed_sec;
    }
}

void parallel_batched_spmv(const numerics::CscMatrix& A, const double* batch_x, double* batch_y,
                           std::size_t batch_size, const BatchedSpmvOptions& options,
                           BatchedSpmvTelemetry* telemetry, ThreadPool* custom_pool) {
    if (batch_size == 0) {
        if (telemetry != nullptr) {
            *telemetry = {};
            telemetry->rows = A.rows();
            telemetry->cols = A.cols();
            telemetry->nonzeros = A.nonzeros();
        }
        return;
    }

    if (batch_x == nullptr || batch_y == nullptr) {
        throw core::InputException("Null pointer passed for batch vectors in batched_spmv");
    }

    const std::size_t in_dim = A.cols();
    const std::size_t out_dim = A.rows();

    if (A.rows() == 0 || A.cols() == 0) {
        std::fill_n(batch_y, batch_size * out_dim, 0.0);
        return;
    }

    std::size_t workers = options.num_workers;
    if (workers == 0) {
        workers = static_cast<std::size_t>(std::max(1u, std::thread::hardware_concurrency()));
    }

    const bool execute_parallel = options.run_parallel && workers > 1 && batch_size > 1;
    const auto t_start = std::chrono::steady_clock::now();

    if (!execute_parallel) {
        for (std::size_t k = 0; k < batch_size; ++k) {
            A.spmv(batch_x + (k * in_dim), batch_y + (k * out_dim));
        }
    } else {
        std::unique_ptr<ThreadPool> local_pool;
        ThreadPool* pool = custom_pool;
        if (pool == nullptr) {
            local_pool = std::make_unique<ThreadPool>(workers);
            pool = local_pool.get();
        }

        pool->parallel_for_chunks(
            batch_size, pool->worker_count(),
            [&A, batch_x, batch_y, in_dim, out_dim](std::size_t start, std::size_t end) {
                for (std::size_t k = start; k < end; ++k) {
                    A.spmv(batch_x + (k * in_dim), batch_y + (k * out_dim));
                }
            });
    }

    const auto t_end = std::chrono::steady_clock::now();
    const double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();

    if (telemetry != nullptr) {
        telemetry->parallel_enabled = execute_parallel;
        telemetry->worker_count = execute_parallel ? workers : 1;
        telemetry->batch_size = batch_size;
        telemetry->rows = A.rows();
        telemetry->cols = A.cols();
        telemetry->nonzeros = A.nonzeros();
        telemetry->parallel_time_seconds = elapsed_sec;
    }
}

void parallel_batched_spmv(const numerics::CsrMatrix& A,
                           const std::vector<std::vector<double>>& batch_x,
                           std::vector<std::vector<double>>& batch_y,
                           const BatchedSpmvOptions& options, BatchedSpmvTelemetry* telemetry,
                           ThreadPool* custom_pool) {
    const std::size_t batch_size = batch_x.size();
    if (batch_size == 0) {
        batch_y.clear();
        if (telemetry != nullptr) {
            *telemetry = {};
            telemetry->rows = A.rows();
            telemetry->cols = A.cols();
            telemetry->nonzeros = A.nonzeros();
        }
        return;
    }

    const std::size_t in_dim = options.transpose ? A.rows() : A.cols();
    const std::size_t out_dim = options.transpose ? A.cols() : A.rows();

    for (std::size_t k = 0; k < batch_size; ++k) {
        if (batch_x[k].size() != in_dim) {
            throw core::InputException(
                "Dimension mismatch in batch_x at index " + std::to_string(k) + ": vector size " +
                std::to_string(batch_x[k].size()) + " != expected " + std::to_string(in_dim));
        }
    }

    batch_y.resize(batch_size);
    for (std::size_t k = 0; k < batch_size; ++k) {
        batch_y[k].resize(out_dim);
    }

    std::size_t workers = options.num_workers;
    if (workers == 0) {
        workers = static_cast<std::size_t>(std::max(1u, std::thread::hardware_concurrency()));
    }

    const bool execute_parallel = options.run_parallel && workers > 1 && batch_size > 1;
    const auto t_start = std::chrono::steady_clock::now();

    if (!execute_parallel) {
        for (std::size_t k = 0; k < batch_size; ++k) {
            if (options.transpose) {
                A.spmv_transpose(batch_x[k].data(), batch_y[k].data());
            } else {
                A.spmv(batch_x[k].data(), batch_y[k].data());
            }
        }
    } else {
        std::unique_ptr<ThreadPool> local_pool;
        ThreadPool* pool = custom_pool;
        if (pool == nullptr) {
            local_pool = std::make_unique<ThreadPool>(workers);
            pool = local_pool.get();
        }

        pool->parallel_for_chunks(batch_size, pool->worker_count(),
                                  [&A, &batch_x, &batch_y, transpose = options.transpose](
                                      std::size_t start, std::size_t end) {
                                      for (std::size_t k = start; k < end; ++k) {
                                          if (transpose) {
                                              A.spmv_transpose(batch_x[k].data(),
                                                               batch_y[k].data());
                                          } else {
                                              A.spmv(batch_x[k].data(), batch_y[k].data());
                                          }
                                      }
                                  });
    }

    const auto t_end = std::chrono::steady_clock::now();
    const double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();

    if (telemetry != nullptr) {
        telemetry->parallel_enabled = execute_parallel;
        telemetry->worker_count = execute_parallel ? workers : 1;
        telemetry->batch_size = batch_size;
        telemetry->rows = A.rows();
        telemetry->cols = A.cols();
        telemetry->nonzeros = A.nonzeros();
        telemetry->parallel_time_seconds = elapsed_sec;
    }
}

void parallel_batched_spmv_raw(std::size_t batch_size, std::size_t rows, std::size_t cols,
                               const double* values, const std::size_t* row_ptr,
                               const std::size_t* col_indices, const double* batch_x,
                               double* batch_y, const BatchedSpmvOptions& options,
                               BatchedSpmvTelemetry* telemetry, ThreadPool* custom_pool) {
    if (batch_size == 0) {
        return;
    }
    if (values == nullptr || row_ptr == nullptr || col_indices == nullptr || batch_x == nullptr ||
        batch_y == nullptr) {
        throw core::InputException("Null pointer passed to parallel_batched_spmv_raw");
    }

    std::size_t workers = options.num_workers;
    if (workers == 0) {
        workers = static_cast<std::size_t>(std::max(1u, std::thread::hardware_concurrency()));
    }

    const bool execute_parallel = options.run_parallel && workers > 1 && batch_size > 1;
    const auto t_start = std::chrono::steady_clock::now();

    if (!execute_parallel) {
        for (std::size_t k = 0; k < batch_size; ++k) {
            raw_spmv_internal(rows, values, row_ptr, col_indices, batch_x + (k * cols),
                              batch_y + (k * rows));
        }
    } else {
        std::unique_ptr<ThreadPool> local_pool;
        ThreadPool* pool = custom_pool;
        if (pool == nullptr) {
            local_pool = std::make_unique<ThreadPool>(workers);
            pool = local_pool.get();
        }

        pool->parallel_for_chunks(batch_size, pool->worker_count(),
                                  [rows, cols, values, row_ptr, col_indices, batch_x, batch_y](
                                      std::size_t start, std::size_t end) {
                                      for (std::size_t k = start; k < end; ++k) {
                                          raw_spmv_internal(rows, values, row_ptr, col_indices,
                                                            batch_x + (k * cols),
                                                            batch_y + (k * rows));
                                      }
                                  });
    }

    const auto t_end = std::chrono::steady_clock::now();
    const double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();

    if (telemetry != nullptr) {
        telemetry->parallel_enabled = execute_parallel;
        telemetry->worker_count = execute_parallel ? workers : 1;
        telemetry->batch_size = batch_size;
        telemetry->rows = rows;
        telemetry->cols = cols;
        telemetry->nonzeros = row_ptr[rows];
        telemetry->parallel_time_seconds = elapsed_sec;
    }
}

void parallel_batched_spmv_raw(std::size_t batch_size, std::size_t rows, std::size_t cols,
                               const double* values, const int* row_ptr, const int* col_indices,
                               const double* batch_x, double* batch_y,
                               const BatchedSpmvOptions& options, BatchedSpmvTelemetry* telemetry,
                               ThreadPool* custom_pool) {
    if (batch_size == 0) {
        return;
    }
    if (values == nullptr || row_ptr == nullptr || col_indices == nullptr || batch_x == nullptr ||
        batch_y == nullptr) {
        throw core::InputException("Null pointer passed to parallel_batched_spmv_raw (int)");
    }

    std::size_t workers = options.num_workers;
    if (workers == 0) {
        workers = static_cast<std::size_t>(std::max(1u, std::thread::hardware_concurrency()));
    }

    const bool execute_parallel = options.run_parallel && workers > 1 && batch_size > 1;
    const auto t_start = std::chrono::steady_clock::now();

    if (!execute_parallel) {
        for (std::size_t k = 0; k < batch_size; ++k) {
            raw_spmv_int_internal(rows, values, row_ptr, col_indices, batch_x + (k * cols),
                                  batch_y + (k * rows));
        }
    } else {
        std::unique_ptr<ThreadPool> local_pool;
        ThreadPool* pool = custom_pool;
        if (pool == nullptr) {
            local_pool = std::make_unique<ThreadPool>(workers);
            pool = local_pool.get();
        }

        pool->parallel_for_chunks(batch_size, pool->worker_count(),
                                  [rows, cols, values, row_ptr, col_indices, batch_x, batch_y](
                                      std::size_t start, std::size_t end) {
                                      for (std::size_t k = start; k < end; ++k) {
                                          raw_spmv_int_internal(rows, values, row_ptr, col_indices,
                                                                batch_x + (k * cols),
                                                                batch_y + (k * rows));
                                      }
                                  });
    }

    const auto t_end = std::chrono::steady_clock::now();
    const double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();

    if (telemetry != nullptr) {
        telemetry->parallel_enabled = execute_parallel;
        telemetry->worker_count = execute_parallel ? workers : 1;
        telemetry->batch_size = batch_size;
        telemetry->rows = rows;
        telemetry->cols = cols;
        telemetry->nonzeros = static_cast<std::size_t>(row_ptr[rows]);
        telemetry->parallel_time_seconds = elapsed_sec;
    }
}

SpmvComparisonResult compare_spmv_results(std::size_t batch_size, std::size_t out_dim,
                                          const double* serial_y, const double* parallel_y,
                                          double abs_tol, double rel_tol) {
    SpmvComparisonResult result;
    if (batch_size == 0 || out_dim == 0) {
        return result;
    }
    if (serial_y == nullptr || parallel_y == nullptr) {
        result.passed = false;
        result.error_message = "Null pointer passed to compare_spmv_results";
        return result;
    }

    for (std::size_t b = 0; b < batch_size; ++b) {
        const std::size_t offset = b * out_dim;
        for (std::size_t i = 0; i < out_dim; ++i) {
            const double s = serial_y[offset + i];
            const double p = parallel_y[offset + i];
            const double abs_err = std::abs(s - p);
            const double denom = std::max(1.0, std::max(std::abs(s), std::abs(p)));
            const double rel_err = abs_err / denom;

            if (abs_err > result.max_absolute_error) {
                result.max_absolute_error = abs_err;
            }
            if (rel_err > result.max_relative_error) {
                result.max_relative_error = rel_err;
            }

            if (abs_err > abs_tol && rel_err > rel_tol) {
                if (result.passed) {
                    result.passed = false;
                    result.mismatch_batch_idx = b;
                    result.mismatch_elem_idx = i;
                    result.serial_val = s;
                    result.parallel_val = p;
                    result.error_message = "Mismatch at batch " + std::to_string(b) + ", elem " +
                                           std::to_string(i) + ": serial=" + std::to_string(s) +
                                           ", parallel=" + std::to_string(p) +
                                           ", abs_err=" + std::to_string(abs_err);
                }
            }
        }
    }
    return result;
}

// ============================================================================
// CpuSequentialBackend Implementation
// ============================================================================

void CpuSequentialBackend::batched_spmv(std::size_t batch_size, std::size_t rows, std::size_t cols,
                                        const double* values, const int* row_ptr,
                                        const int* col_indices, const double* batch_x,
                                        double* batch_y) {
    BatchedSpmvOptions options;
    options.run_parallel = false;
    parallel_batched_spmv_raw(batch_size, rows, cols, values, row_ptr, col_indices, batch_x,
                              batch_y, options);
}

void CpuSequentialBackend::batched_spmv(std::size_t batch_size, std::size_t rows, std::size_t cols,
                                        const double* values, const std::size_t* row_ptr,
                                        const std::size_t* col_indices, const double* batch_x,
                                        double* batch_y) {
    BatchedSpmvOptions options;
    options.run_parallel = false;
    parallel_batched_spmv_raw(batch_size, rows, cols, values, row_ptr, col_indices, batch_x,
                              batch_y, options);
}

void CpuSequentialBackend::batched_spmv(const numerics::CsrMatrix& A, const double* batch_x,
                                        double* batch_y, std::size_t batch_size,
                                        const BatchedSpmvOptions& options) {
    BatchedSpmvOptions opts = options;
    opts.run_parallel = false;
    parallel_batched_spmv(A, batch_x, batch_y, batch_size, opts);
}

// ============================================================================
// CpuMulticoreBackend Implementation
// ============================================================================

CpuMulticoreBackend::CpuMulticoreBackend(std::size_t num_workers)
    : thread_pool_(std::make_shared<ThreadPool>(num_workers)) {}

CpuMulticoreBackend::CpuMulticoreBackend(std::shared_ptr<ThreadPool> thread_pool)
    : thread_pool_(std::move(thread_pool)) {
    if (!thread_pool_) {
        thread_pool_ = std::make_shared<ThreadPool>(0);
    }
}

void CpuMulticoreBackend::batched_spmv(std::size_t batch_size, std::size_t rows, std::size_t cols,
                                       const double* values, const int* row_ptr,
                                       const int* col_indices, const double* batch_x,
                                       double* batch_y) {
    BatchedSpmvOptions options;
    options.run_parallel = true;
    options.num_workers = worker_count();
    parallel_batched_spmv_raw(batch_size, rows, cols, values, row_ptr, col_indices, batch_x,
                              batch_y, options, &last_telemetry_, thread_pool_.get());
}

void CpuMulticoreBackend::batched_spmv(std::size_t batch_size, std::size_t rows, std::size_t cols,
                                       const double* values, const std::size_t* row_ptr,
                                       const std::size_t* col_indices, const double* batch_x,
                                       double* batch_y) {
    BatchedSpmvOptions options;
    options.run_parallel = true;
    options.num_workers = worker_count();
    parallel_batched_spmv_raw(batch_size, rows, cols, values, row_ptr, col_indices, batch_x,
                              batch_y, options, &last_telemetry_, thread_pool_.get());
}

void CpuMulticoreBackend::batched_spmv(const numerics::CsrMatrix& A, const double* batch_x,
                                       double* batch_y, std::size_t batch_size,
                                       const BatchedSpmvOptions& options) {
    BatchedSpmvOptions opts = options;
    if (opts.num_workers == 0) {
        opts.num_workers = worker_count();
    }
    parallel_batched_spmv(A, batch_x, batch_y, batch_size, opts, &last_telemetry_,
                          thread_pool_.get());
}

}  // namespace kavach::backend::cpu
