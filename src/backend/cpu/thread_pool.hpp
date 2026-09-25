#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

#include "core/error.hpp"

namespace kavach::backend::cpu {

/**
 * @brief Contiguous slice representing a chunk of batch indices [start, end).
 */
struct BatchChunk {
    std::size_t start{0};
    std::size_t end{0};

    [[nodiscard]] std::size_t size() const noexcept { return end >= start ? end - start : 0; }
    [[nodiscard]] bool empty() const noexcept { return start >= end; }
};

/**
 * @brief Lightweight, reusable Thread Pool for multicore CPU execution.
 * Fulfills Blueprint Section 14 and Section 32 (Phase 10).
 *
 * Guarantees:
 * - RAII lifecycle: threads are joined on destruction; no detached threads.
 * - Configurable worker count (1 to N, default auto-detected hardware concurrency).
 * - Safe task submission and deterministic batch partitioning.
 * - Disjoint output regions (no shared mutable state).
 * - Exception propagation to the calling thread.
 * - Clean shutdown and reuse across multiple successive batches.
 */
class ThreadPool {
public:
    /**
     * @brief Construct a new ThreadPool with the specified worker count.
     * @param num_workers Number of worker threads (0 = auto-detect via
     * std::thread::hardware_concurrency()).
     */
    explicit ThreadPool(std::size_t num_workers = 0);

    /**
     * @brief Destructor safely joins all worker threads.
     */
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    /**
     * @brief Number of active worker threads in the pool.
     */
    [[nodiscard]] std::size_t worker_count() const noexcept { return workers_.size(); }

    /**
     * @brief Number of tasks currently queued and waiting for execution.
     */
    [[nodiscard]] std::size_t queued_tasks() const noexcept;

    /**
     * @brief Check whether the thread pool is currently stopped or shutting down.
     */
    [[nodiscard]] bool is_stopped() const noexcept { return stop_.load(std::memory_order_acquire); }

    /**
     * @brief Partitions a batch of items deterministically across a specified number of chunks.
     * Guarantees all chunks are contiguous, mutually disjoint, and cover [0, batch_size) exactly.
     */
    [[nodiscard]] static std::vector<BatchChunk> partition_batch(std::size_t batch_size,
                                                                 std::size_t num_chunks);

    /**
     * @brief Submit a callable task returning a std::future.
     */
    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using ReturnType = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            [func = std::forward<F>(f),
             ... captured_args = std::forward<Args>(args)]() mutable -> ReturnType {
                return std::invoke(func, captured_args...);
            });

        std::future<ReturnType> res = task->get_future();

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_.load(std::memory_order_relaxed)) {
                throw core::InputException("Cannot submit task to stopped ThreadPool");
            }
            tasks_.emplace([task]() { (*task)(); });
        }

        cv_task_.notify_one();
        return res;
    }

    /**
     * @brief Execute a function across partitioned batch chunks in parallel.
     * If an exception occurs in any worker, waits for all chunk tasks in the current batch to
     * finish and rethrows the first captured exception safely to the caller.
     *
     * @tparam Func Signature void(std::size_t start, std::size_t end)
     */
    template <typename Func>
    void parallel_for_chunks(std::size_t batch_size, std::size_t num_chunks, Func&& func) {
        if (batch_size == 0) {
            return;
        }

        if (num_chunks <= 1 || workers_.empty()) {
            func(0, batch_size);
            return;
        }

        auto chunks = partition_batch(batch_size, num_chunks);
        if (chunks.size() <= 1) {
            func(0, batch_size);
            return;
        }

        std::vector<std::future<void>> futures;
        futures.reserve(chunks.size());

        for (const auto& chunk : chunks) {
            futures.push_back(submit([&func, chunk]() { func(chunk.start, chunk.end); }));
        }

        std::exception_ptr first_exception = nullptr;
        for (auto& fut : futures) {
            try {
                fut.get();
            } catch (...) {
                if (!first_exception) {
                    first_exception = std::current_exception();
                }
            }
        }

        if (first_exception) {
            std::rethrow_exception(first_exception);
        }
    }

    /**
     * @brief Gracefully initiates shutdown and joins all worker threads.
     */
    void shutdown() noexcept;

private:
    void worker_loop();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    mutable std::mutex queue_mutex_;
    std::condition_variable cv_task_;
    std::condition_variable cv_done_;
    std::atomic<bool> stop_{false};
    std::atomic<std::size_t> active_tasks_{0};
};

}  // namespace kavach::backend::cpu
