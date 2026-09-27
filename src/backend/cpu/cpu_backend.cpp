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

    const bool execute_parallel =
        options.run_parallel && workers > 1 && (batch_size > 1 || out_dim > 1);
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

        if (batch_size > 1) {
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
        } else if (!options.transpose) {
            // batch_size == 1, non-transpose: partition rows [0, A.rows()) deterministically
            const auto& row_ptr = A.row_ptr();
            const auto& col_idx = A.col_idx();
            const auto& values = A.values();
            pool->parallel_for_chunks(out_dim, pool->worker_count(),
                                      [&row_ptr, &col_idx, &values, batch_x, batch_y](
                                          std::size_t r_start, std::size_t r_end) {
                                          for (std::size_t r = r_start; r < r_end; ++r) {
                                              double sum = 0.0;
                                              const std::size_t p_start = row_ptr[r];
                                              const std::size_t p_end = row_ptr[r + 1];
                                              for (std::size_t p = p_start; p < p_end; ++p) {
                                                  sum += values[p] * batch_x[col_idx[p]];
                                              }
                                              batch_y[r] = sum;
                                          }
                                      });
        } else {
            // batch_size == 1, transpose: partition output columns [0, A.cols()) deterministically
            const auto& row_ptr = A.row_ptr();
            const auto& col_idx = A.col_idx();
            const auto& values = A.values();
            const std::size_t rows = A.rows();
            pool->parallel_for_chunks(out_dim, pool->worker_count(),
                                      [&row_ptr, &col_idx, &values, rows, batch_x, batch_y](
                                          std::size_t c_start, std::size_t c_end) {
                                          std::fill(batch_y + c_start, batch_y + c_end, 0.0);
                                          for (std::size_t r = 0; r < rows; ++r) {
                                              const double xr = batch_x[r];
                                              if (xr == 0.0) {
                                                  continue;
                                              }
                                              const std::size_t p_start = row_ptr[r];
                                              const std::size_t p_end = row_ptr[r + 1];
                                              for (std::size_t p = p_start; p < p_end; ++p) {
                                                  const std::size_t c = col_idx[p];
                                                  if (c >= c_start && c < c_end) {
                                                      batch_y[c] += values[p] * xr;
                                                  }
                                              }
                                          }
                                      });
        }
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

SpmvComparisonResult compare_vector_results(std::size_t total_elements, const double* ref_vec,
                                            const double* test_vec, double abs_tol,
                                            double rel_tol) {
    return compare_spmv_results(1, total_elements, ref_vec, test_vec, abs_tol, rel_tol);
}

SpmvComparisonResult compare_scalar_results(std::size_t count, const double* ref_scalars,
                                            const double* test_scalars, double abs_tol,
                                            double rel_tol) {
    return compare_spmv_results(1, count, ref_scalars, test_scalars, abs_tol, rel_tol);
}

void serial_batched_spmv_transpose(const numerics::CsrMatrix& A, const double* batch_x,
                                   double* batch_y, std::size_t batch_size) {
    if (batch_size == 0) {
        return;
    }
    if (batch_x == nullptr || batch_y == nullptr) {
        throw core::InputException("Null pointer passed to serial_batched_spmv_transpose");
    }
    const std::size_t in_dim = A.rows();
    const std::size_t out_dim = A.cols();
    for (std::size_t k = 0; k < batch_size; ++k) {
        A.spmv_transpose(batch_x + (k * in_dim), batch_y + (k * out_dim));
    }
}

void parallel_batched_spmv_transpose(const numerics::CsrMatrix& A, const double* batch_x,
                                     double* batch_y, std::size_t batch_size,
                                     const BatchedSpmvOptions& options,
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
        throw core::InputException("Null pointer passed to parallel_batched_spmv_transpose");
    }
    const std::size_t in_dim = A.rows();
    const std::size_t out_dim = A.cols();
    if (A.rows() == 0 || A.cols() == 0) {
        std::fill_n(batch_y, batch_size * out_dim, 0.0);
        return;
    }

    std::size_t workers = options.num_workers;
    if (workers == 0) {
        workers = static_cast<std::size_t>(std::max(1u, std::thread::hardware_concurrency()));
    }
    const bool execute_parallel =
        options.run_parallel && workers > 1 && (batch_size > 1 || out_dim > 1);
    const auto t_start = std::chrono::steady_clock::now();

    if (!execute_parallel) {
        serial_batched_spmv_transpose(A, batch_x, batch_y, batch_size);
    } else {
        std::unique_ptr<ThreadPool> local_pool;
        ThreadPool* pool = custom_pool;
        if (pool == nullptr) {
            local_pool = std::make_unique<ThreadPool>(workers);
            pool = local_pool.get();
        }
        if (batch_size > 1) {
            pool->parallel_for_chunks(
                batch_size, pool->worker_count(),
                [&A, batch_x, batch_y, in_dim, out_dim](std::size_t start, std::size_t end) {
                    for (std::size_t k = start; k < end; ++k) {
                        A.spmv_transpose(batch_x + (k * in_dim), batch_y + (k * out_dim));
                    }
                });
        } else {
            // batch_size == 1: partition output columns [0, A.cols()) deterministically
            const auto& row_ptr = A.row_ptr();
            const auto& col_idx = A.col_idx();
            const auto& values = A.values();
            const std::size_t rows = A.rows();
            pool->parallel_for_chunks(out_dim, pool->worker_count(),
                                      [&row_ptr, &col_idx, &values, rows, batch_x, batch_y](
                                          std::size_t c_start, std::size_t c_end) {
                                          std::fill(batch_y + c_start, batch_y + c_end, 0.0);
                                          for (std::size_t r = 0; r < rows; ++r) {
                                              const double xr = batch_x[r];
                                              if (xr == 0.0) {
                                                  continue;
                                              }
                                              const std::size_t p_start = row_ptr[r];
                                              const std::size_t p_end = row_ptr[r + 1];
                                              for (std::size_t p = p_start; p < p_end; ++p) {
                                                  const std::size_t c = col_idx[p];
                                                  if (c >= c_start && c < c_end) {
                                                      batch_y[c] += values[p] * xr;
                                                  }
                                              }
                                          }
                                      });
        }
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

void serial_batched_csc_spmv(const numerics::CscMatrix& A, const double* batch_x, double* batch_y,
                             std::size_t batch_size) {
    if (batch_size == 0) {
        return;
    }
    if (batch_x == nullptr || batch_y == nullptr) {
        throw core::InputException("Null pointer passed to serial_batched_csc_spmv");
    }
    const std::size_t in_dim = A.cols();
    const std::size_t out_dim = A.rows();
    for (std::size_t k = 0; k < batch_size; ++k) {
        A.spmv(batch_x + (k * in_dim), batch_y + (k * out_dim));
    }
}

void serial_batched_axpby(std::size_t batch_size, std::size_t n, double alpha,
                          const double* batch_x, double beta, double* batch_y) {
    if (batch_size == 0 || n == 0) {
        return;
    }
    if (batch_x == nullptr || batch_y == nullptr) {
        throw core::InputException("Null pointer passed to serial_batched_axpby");
    }
    const std::size_t total = batch_size * n;
    for (std::size_t i = 0; i < total; ++i) {
        batch_y[i] = alpha * batch_x[i] + beta * batch_y[i];
    }
}

void parallel_batched_axpby(std::size_t batch_size, std::size_t n, double alpha,
                            const double* batch_x, double beta, double* batch_y,
                            ThreadPool* custom_pool) {
    if (batch_size == 0 || n == 0) {
        return;
    }
    if (batch_x == nullptr || batch_y == nullptr) {
        throw core::InputException("Null pointer passed to parallel_batched_axpby");
    }
    if (batch_size == 1 || custom_pool == nullptr) {
        serial_batched_axpby(batch_size, n, alpha, batch_x, beta, batch_y);
        return;
    }
    custom_pool->parallel_for_chunks(
        batch_size, custom_pool->worker_count(),
        [n, alpha, batch_x, beta, batch_y](std::size_t start, std::size_t end) {
            for (std::size_t k = start; k < end; ++k) {
                const double* x_k = batch_x + (k * n);
                double* y_k = batch_y + (k * n);
                for (std::size_t i = 0; i < n; ++i) {
                    y_k[i] = alpha * x_k[i] + beta * y_k[i];
                }
            }
        });
}

void serial_batched_dot(std::size_t batch_size, std::size_t n, const double* batch_x,
                        const double* batch_y, double* results) {
    if (batch_size == 0) {
        return;
    }
    if (results == nullptr) {
        throw core::InputException("Null pointer for results in serial_batched_dot");
    }
    if (n == 0) {
        std::fill_n(results, batch_size, 0.0);
        return;
    }
    if (batch_x == nullptr || batch_y == nullptr) {
        throw core::InputException("Null pointer passed to serial_batched_dot");
    }
    for (std::size_t k = 0; k < batch_size; ++k) {
        const double* x_k = batch_x + (k * n);
        const double* y_k = batch_y + (k * n);
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            sum += x_k[i] * y_k[i];
        }
        results[k] = sum;
    }
}

void parallel_batched_dot(std::size_t batch_size, std::size_t n, const double* batch_x,
                          const double* batch_y, double* results, ThreadPool* custom_pool) {
    if (batch_size == 0) {
        return;
    }
    if (results == nullptr) {
        throw core::InputException("Null pointer for results in parallel_batched_dot");
    }
    if (batch_size == 1 || custom_pool == nullptr || n == 0) {
        serial_batched_dot(batch_size, n, batch_x, batch_y, results);
        return;
    }
    if (batch_x == nullptr || batch_y == nullptr) {
        throw core::InputException("Null pointer passed to parallel_batched_dot");
    }
    custom_pool->parallel_for_chunks(
        batch_size, custom_pool->worker_count(),
        [n, batch_x, batch_y, results](std::size_t start, std::size_t end) {
            for (std::size_t k = start; k < end; ++k) {
                const double* x_k = batch_x + (k * n);
                const double* y_k = batch_y + (k * n);
                double sum = 0.0;
                for (std::size_t i = 0; i < n; ++i) {
                    sum += x_k[i] * y_k[i];
                }
                results[k] = sum;
            }
        });
}

void serial_batched_l2_norm(std::size_t batch_size, std::size_t n, const double* batch_x,
                            double* results) {
    if (batch_size == 0) {
        return;
    }
    if (results == nullptr) {
        throw core::InputException("Null pointer for results in serial_batched_l2_norm");
    }
    if (n == 0) {
        std::fill_n(results, batch_size, 0.0);
        return;
    }
    if (batch_x == nullptr) {
        throw core::InputException("Null pointer for batch_x in serial_batched_l2_norm");
    }
    for (std::size_t k = 0; k < batch_size; ++k) {
        const double* x_k = batch_x + (k * n);
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            sum += x_k[i] * x_k[i];
        }
        results[k] = std::sqrt(sum);
    }
}

void parallel_batched_l2_norm(std::size_t batch_size, std::size_t n, const double* batch_x,
                              double* results, ThreadPool* custom_pool) {
    if (batch_size == 0) {
        return;
    }
    if (results == nullptr) {
        throw core::InputException("Null pointer for results in parallel_batched_l2_norm");
    }
    if (batch_size == 1 || custom_pool == nullptr || n == 0) {
        serial_batched_l2_norm(batch_size, n, batch_x, results);
        return;
    }
    if (batch_x == nullptr) {
        throw core::InputException("Null pointer for batch_x in parallel_batched_l2_norm");
    }
    custom_pool->parallel_for_chunks(batch_size, custom_pool->worker_count(),
                                     [n, batch_x, results](std::size_t start, std::size_t end) {
                                         for (std::size_t k = start; k < end; ++k) {
                                             const double* x_k = batch_x + (k * n);
                                             double sum = 0.0;
                                             for (std::size_t i = 0; i < n; ++i) {
                                                 sum += x_k[i] * x_k[i];
                                             }
                                             results[k] = std::sqrt(sum);
                                         }
                                     });
}

void serial_batched_linf_norm(std::size_t batch_size, std::size_t n, const double* batch_x,
                              double* results) {
    if (batch_size == 0) {
        return;
    }
    if (results == nullptr) {
        throw core::InputException("Null pointer for results in serial_batched_linf_norm");
    }
    if (n == 0) {
        std::fill_n(results, batch_size, 0.0);
        return;
    }
    if (batch_x == nullptr) {
        throw core::InputException("Null pointer for batch_x in serial_batched_linf_norm");
    }
    for (std::size_t k = 0; k < batch_size; ++k) {
        const double* x_k = batch_x + (k * n);
        double max_val = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            max_val = std::max(max_val, std::abs(x_k[i]));
        }
        results[k] = max_val;
    }
}

void parallel_batched_linf_norm(std::size_t batch_size, std::size_t n, const double* batch_x,
                                double* results, ThreadPool* custom_pool) {
    if (batch_size == 0) {
        return;
    }
    if (results == nullptr) {
        throw core::InputException("Null pointer for results in parallel_batched_linf_norm");
    }
    if (batch_size == 1 || custom_pool == nullptr || n == 0) {
        serial_batched_linf_norm(batch_size, n, batch_x, results);
        return;
    }
    if (batch_x == nullptr) {
        throw core::InputException("Null pointer for batch_x in parallel_batched_linf_norm");
    }
    custom_pool->parallel_for_chunks(batch_size, custom_pool->worker_count(),
                                     [n, batch_x, results](std::size_t start, std::size_t end) {
                                         for (std::size_t k = start; k < end; ++k) {
                                             const double* x_k = batch_x + (k * n);
                                             double max_val = 0.0;
                                             for (std::size_t i = 0; i < n; ++i) {
                                                 max_val = std::max(max_val, std::abs(x_k[i]));
                                             }
                                             results[k] = max_val;
                                         }
                                     });
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
    BatchedSpmvTelemetry tel;
    parallel_batched_spmv_raw(batch_size, rows, cols, values, row_ptr, col_indices, batch_x,
                              batch_y, options, &tel);
    record_invocation(BackendType::CPU_SEQUENTIAL, "serial_batched_csr_spmv", batch_size, 1, 0.0,
                      tel.parallel_time_seconds, 0.0, tel.parallel_time_seconds);
}

void CpuSequentialBackend::batched_spmv(std::size_t batch_size, std::size_t rows, std::size_t cols,
                                        const double* values, const std::size_t* row_ptr,
                                        const std::size_t* col_indices, const double* batch_x,
                                        double* batch_y) {
    BatchedSpmvOptions options;
    options.run_parallel = false;
    BatchedSpmvTelemetry tel;
    parallel_batched_spmv_raw(batch_size, rows, cols, values, row_ptr, col_indices, batch_x,
                              batch_y, options, &tel);
    record_invocation(BackendType::CPU_SEQUENTIAL, "serial_batched_csr_spmv", batch_size, 1, 0.0,
                      tel.parallel_time_seconds, 0.0, tel.parallel_time_seconds);
}

void CpuSequentialBackend::batched_spmv(const numerics::CsrMatrix& A, const double* batch_x,
                                        double* batch_y, std::size_t batch_size,
                                        const BatchedSpmvOptions& options) {
    BatchedSpmvOptions opts = options;
    opts.run_parallel = false;
    BatchedSpmvTelemetry tel;
    parallel_batched_spmv(A, batch_x, batch_y, batch_size, opts, &tel);
    record_invocation(
        BackendType::CPU_SEQUENTIAL,
        opts.transpose ? "serial_batched_csr_transpose_spmv" : "serial_batched_csr_spmv",
        batch_size, 1, 0.0, tel.parallel_time_seconds, 0.0, tel.parallel_time_seconds);
}

void CpuSequentialBackend::batched_spmv_transpose(const numerics::CsrMatrix& A,
                                                  const double* batch_x, double* batch_y,
                                                  std::size_t batch_size,
                                                  const BatchedSpmvOptions&) {
    const auto t0 = std::chrono::steady_clock::now();
    serial_batched_spmv_transpose(A, batch_x, batch_y, batch_size);
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    record_invocation(BackendType::CPU_SEQUENTIAL, "serial_batched_csr_transpose_spmv", batch_size,
                      1, 0.0, sec, 0.0, sec);
}

void CpuSequentialBackend::batched_csc_spmv(const numerics::CscMatrix& A, const double* batch_x,
                                            double* batch_y, std::size_t batch_size,
                                            const BatchedSpmvOptions&) {
    const auto t0 = std::chrono::steady_clock::now();
    serial_batched_csc_spmv(A, batch_x, batch_y, batch_size);
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    record_invocation(BackendType::CPU_SEQUENTIAL, "serial_batched_csc_spmv", batch_size, 1, 0.0,
                      sec, 0.0, sec);
}

void CpuSequentialBackend::batched_axpby(std::size_t batch_size, std::size_t n, double alpha,
                                         const double* batch_x, double beta, double* batch_y) {
    const auto t0 = std::chrono::steady_clock::now();
    serial_batched_axpby(batch_size, n, alpha, batch_x, beta, batch_y);
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    record_invocation(BackendType::CPU_SEQUENTIAL, "serial_batched_axpby", batch_size, 1, 0.0, sec,
                      0.0, sec);
}

void CpuSequentialBackend::batched_dot(std::size_t batch_size, std::size_t n, const double* batch_x,
                                       const double* batch_y, double* results) {
    const auto t0 = std::chrono::steady_clock::now();
    serial_batched_dot(batch_size, n, batch_x, batch_y, results);
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    record_invocation(BackendType::CPU_SEQUENTIAL, "serial_batched_dot", batch_size, 1, 0.0, sec,
                      0.0, sec);
}

void CpuSequentialBackend::batched_l2_norm(std::size_t batch_size, std::size_t n,
                                           const double* batch_x, double* results) {
    const auto t0 = std::chrono::steady_clock::now();
    serial_batched_l2_norm(batch_size, n, batch_x, results);
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    record_invocation(BackendType::CPU_SEQUENTIAL, "serial_batched_l2_norm", batch_size, 1, 0.0,
                      sec, 0.0, sec);
}

void CpuSequentialBackend::batched_linf_norm(std::size_t batch_size, std::size_t n,
                                             const double* batch_x, double* results) {
    const auto t0 = std::chrono::steady_clock::now();
    serial_batched_linf_norm(batch_size, n, batch_x, results);
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    record_invocation(BackendType::CPU_SEQUENTIAL, "serial_batched_linf_norm", batch_size, 1, 0.0,
                      sec, 0.0, sec);
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
    record_invocation(BackendType::CPU_MULTICORE, "parallel_batched_csr_spmv", batch_size,
                      worker_count(), 0.0, last_telemetry_.parallel_time_seconds, 0.0,
                      last_telemetry_.parallel_time_seconds);
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
    record_invocation(BackendType::CPU_MULTICORE, "parallel_batched_csr_spmv", batch_size,
                      worker_count(), 0.0, last_telemetry_.parallel_time_seconds, 0.0,
                      last_telemetry_.parallel_time_seconds);
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
    record_invocation(
        BackendType::CPU_MULTICORE,
        opts.transpose ? "parallel_batched_csr_transpose_spmv" : "parallel_batched_csr_spmv",
        batch_size, worker_count(), 0.0, last_telemetry_.parallel_time_seconds, 0.0,
        last_telemetry_.parallel_time_seconds);
}

void CpuMulticoreBackend::batched_spmv_transpose(const numerics::CsrMatrix& A,
                                                 const double* batch_x, double* batch_y,
                                                 std::size_t batch_size,
                                                 const BatchedSpmvOptions& options) {
    BatchedSpmvOptions opts = options;
    if (opts.num_workers == 0) {
        opts.num_workers = worker_count();
    }
    parallel_batched_spmv_transpose(A, batch_x, batch_y, batch_size, opts, &last_telemetry_,
                                    thread_pool_.get());
    record_invocation(BackendType::CPU_MULTICORE, "parallel_batched_csr_transpose_spmv", batch_size,
                      worker_count(), 0.0, last_telemetry_.parallel_time_seconds, 0.0,
                      last_telemetry_.parallel_time_seconds);
}

void CpuMulticoreBackend::batched_csc_spmv(const numerics::CscMatrix& A, const double* batch_x,
                                           double* batch_y, std::size_t batch_size,
                                           const BatchedSpmvOptions& options) {
    BatchedSpmvOptions opts = options;
    if (opts.num_workers == 0) {
        opts.num_workers = worker_count();
    }
    parallel_batched_spmv(A, batch_x, batch_y, batch_size, opts, &last_telemetry_,
                          thread_pool_.get());
    record_invocation(BackendType::CPU_MULTICORE, "parallel_batched_csc_spmv", batch_size,
                      worker_count(), 0.0, last_telemetry_.parallel_time_seconds, 0.0,
                      last_telemetry_.parallel_time_seconds);
}

void CpuMulticoreBackend::batched_axpby(std::size_t batch_size, std::size_t n, double alpha,
                                        const double* batch_x, double beta, double* batch_y) {
    const auto t0 = std::chrono::steady_clock::now();
    parallel_batched_axpby(batch_size, n, alpha, batch_x, beta, batch_y, thread_pool_.get());
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    record_invocation(BackendType::CPU_MULTICORE, "parallel_batched_axpby", batch_size,
                      worker_count(), 0.0, sec, 0.0, sec);
}

void CpuMulticoreBackend::batched_dot(std::size_t batch_size, std::size_t n, const double* batch_x,
                                      const double* batch_y, double* results) {
    const auto t0 = std::chrono::steady_clock::now();
    parallel_batched_dot(batch_size, n, batch_x, batch_y, results, thread_pool_.get());
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    record_invocation(BackendType::CPU_MULTICORE, "parallel_batched_dot", batch_size,
                      worker_count(), 0.0, sec, 0.0, sec);
}

void CpuMulticoreBackend::batched_l2_norm(std::size_t batch_size, std::size_t n,
                                          const double* batch_x, double* results) {
    const auto t0 = std::chrono::steady_clock::now();
    parallel_batched_l2_norm(batch_size, n, batch_x, results, thread_pool_.get());
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    record_invocation(BackendType::CPU_MULTICORE, "parallel_batched_l2_norm", batch_size,
                      worker_count(), 0.0, sec, 0.0, sec);
}

void CpuMulticoreBackend::batched_linf_norm(std::size_t batch_size, std::size_t n,
                                            const double* batch_x, double* results) {
    const auto t0 = std::chrono::steady_clock::now();
    parallel_batched_linf_norm(batch_size, n, batch_x, results, thread_pool_.get());
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    record_invocation(BackendType::CPU_MULTICORE, "parallel_batched_linf_norm", batch_size,
                      worker_count(), 0.0, sec, 0.0, sec);
}

}  // namespace kavach::backend::cpu
