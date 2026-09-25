#pragma once

#include <filesystem>
#include <string>
#include <string_view>

#include "benchmark/benchmark_case.hpp"

namespace kavach::benchmark {

class ManifestException : public std::runtime_error {
public:
    explicit ManifestException(const std::string& msg) : std::runtime_error(msg) {}
};

/**
 * @brief Manifest loader for benchmark suites (Section 8 & 9).
 */
class BenchmarkManifestLoader {
public:
    [[nodiscard]] static BenchmarkSuite load_from_file(const std::filesystem::path& path);
    [[nodiscard]] static BenchmarkSuite load_from_string(std::string_view content,
                                                         const std::string& base_dir = ".");
};

}  // namespace kavach::benchmark
