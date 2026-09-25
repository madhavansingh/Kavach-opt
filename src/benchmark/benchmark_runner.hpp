#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "benchmark/benchmark_case.hpp"
#include "benchmark/benchmark_result.hpp"
#include "benchmark/reference/reference_adapter.hpp"

namespace kavach::benchmark {

/**
 * @brief Options configuring benchmark runner execution.
 */
struct BenchmarkRunnerOptions {
    std::string suite_name{"benchmark"};
    bool compare_reference{false};             ///< Run isolated HiGHS external comparison
    std::string reference_bin_path;            ///< Custom HiGHS executable path
    std::optional<double> time_limit_seconds;  ///< Override time limit
    std::optional<uint64_t> seed;              ///< Override random seed
    bool force_cpu{false};                     ///< Force CPU fallback (disable GPU)
    std::optional<std::string> tag_filter;     ///< Filter by tag
    std::optional<std::string> case_filter;    ///< Filter by case_id pattern
    bool verbose{false};
    std::filesystem::path output_dir{"benchmark_results"};
};

/**
 * @brief Orchestrator for loading, solving, measuring, verifying, and comparing benchmark cases.
 */
class BenchmarkRunner {
public:
    explicit BenchmarkRunner(BenchmarkRunnerOptions options = {});

    /// Run a single benchmark case
    [[nodiscard]] BenchmarkResult run_case(const BenchmarkCase& c) const;

    /// Run an entire benchmark suite
    [[nodiscard]] std::vector<BenchmarkResult> run_suite(const BenchmarkSuite& suite) const;

    [[nodiscard]] const BenchmarkRunnerOptions& options() const noexcept { return options_; }
    void set_options(BenchmarkRunnerOptions options) noexcept { options_ = std::move(options); }

private:
    BenchmarkRunnerOptions options_;
    ExternalReferenceAdapter reference_adapter_;
};

}  // namespace kavach::benchmark
