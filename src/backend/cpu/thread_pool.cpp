#include "backend/cpu/thread_pool.hpp"

#include <algorithm>

namespace kavach::backend::cpu {

ThreadPool::ThreadPool(std::size_t num_workers) {
    if (num_workers == 0) {
        num_workers = static_cast<std::size_t>(std::max(1u, std::thread::hardware_concurrency()));
    }

    workers_.reserve(num_workers);
    try {
        for (std::size_t i = 0; i < num_workers; ++i) {
            workers_.emplace_back(&ThreadPool::worker_loop, this);
        }
    } catch (...) {
        shutdown();
        throw core::NumericalException("Failed to spawn worker threads in ThreadPool");
    }
}

ThreadPool::~ThreadPool() {
    shutdown();
}

void ThreadPool::shutdown() noexcept {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_.load(std::memory_order_relaxed)) {
            return;
        }
        stop_.store(true, std::memory_order_release);
    }

    cv_task_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
}

std::size_t ThreadPool::queued_tasks() const noexcept {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    return tasks_.size();
}

std::vector<BatchChunk> ThreadPool::partition_batch(std::size_t batch_size,
                                                    std::size_t num_chunks) {
    if (batch_size == 0 || num_chunks == 0) {
        return {};
    }

    std::size_t actual_chunks = std::min(num_chunks, batch_size);
    std::vector<BatchChunk> chunks;
    chunks.reserve(actual_chunks);

    for (std::size_t i = 0; i < actual_chunks; ++i) {
        std::size_t start = (i * batch_size) / actual_chunks;
        std::size_t end = ((i + 1) * batch_size) / actual_chunks;
        if (start < end) {
            chunks.push_back({start, end});
        }
    }

    return chunks;
}

void ThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_task_.wait(lock, [this]() {
                return stop_.load(std::memory_order_relaxed) || !tasks_.empty();
            });

            if (stop_.load(std::memory_order_relaxed) && tasks_.empty()) {
                return;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
            active_tasks_.fetch_add(1, std::memory_order_relaxed);
        }

        try {
            task();
        } catch (...) {
            // Task exceptions are caught and stored by std::packaged_task in std::future
        }

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            active_tasks_.fetch_sub(1, std::memory_order_relaxed);
            if (tasks_.empty() && active_tasks_.load(std::memory_order_relaxed) == 0) {
                cv_done_.notify_all();
            }
        }
    }
}

}  // namespace kavach::backend::cpu
