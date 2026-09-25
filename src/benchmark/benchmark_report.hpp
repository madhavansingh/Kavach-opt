#pragma once

#include <string>
#include <vector>

#include "benchmark/benchmark_result.hpp"

namespace kavach::benchmark {

/**
 * @brief Aggregate summary metrics strictly derived from per-instance rows (Section 23).
 */
struct AggregateStatistics {
    std::size_t total_attempted{0};
    std::size_t verified_optimal{0};
    std::size_t feasible{0};
    std::size_t infeasible{0};
    std::size_t unbounded{0};
    std::size_t limit_reached{0};
    std::size_t numerical_failure{0};
    std::size_t parse_failure{0};
    std::size_t verification_failure{0};
    std::size_t unsupported{0};
    std::size_t runtime_error{0};
    std::size_t solver_timeout{0};
    std::size_t reference_unavailable{0};

    double median_runtime_seconds{0.0};
    double min_runtime_seconds{0.0};
    double max_runtime_seconds{0.0};
    double median_objective_error_rel{0.0};
    double mean_iterations{0.0};
    double mean_nodes{0.0};
    double median_spmv_speedup{1.0};
};

[[nodiscard]] AggregateStatistics compute_aggregates(const std::vector<BenchmarkResult>& results);

/**
 * @brief Markdown report generator (Section 22).
 */
class BenchmarkReporter {
public:
    [[nodiscard]] static std::string generate_markdown_report(
        const std::vector<BenchmarkResult>& results, const ReproducibilityManifest& manifest);
};

}  // namespace kavach::benchmark
