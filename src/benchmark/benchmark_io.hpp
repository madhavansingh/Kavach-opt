#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "benchmark/benchmark_result.hpp"
#include "benchmark/json.hpp"

namespace kavach::benchmark {

/**
 * @brief Exporter and loader for machine-readable benchmark outputs (Section 21).
 */
class BenchmarkIO {
public:
    /// Export benchmark run to structured JSON preserving nulls, nested telemetry, and manifests
    static void export_json(const std::vector<BenchmarkResult>& results,
                            const ReproducibilityManifest& manifest,
                            const std::filesystem::path& path);

    /// Convert a single result to JsonValue
    [[nodiscard]] static JsonValue to_json(const BenchmarkResult& r);

    /// Convert ReproducibilityManifest to JsonValue
    [[nodiscard]] static JsonValue to_json(const ReproducibilityManifest& m);

    /// Export benchmark run to CSV
    static void export_csv(const std::vector<BenchmarkResult>& results,
                           const std::filesystem::path& path);

    /// Load benchmark results from JSON file
    [[nodiscard]] static std::pair<ReproducibilityManifest, std::vector<BenchmarkResult>>
    import_json(const std::filesystem::path& path);

    /// Parse a single result from JsonValue
    [[nodiscard]] static BenchmarkResult from_json(const JsonValue& j);
};

}  // namespace kavach::benchmark
