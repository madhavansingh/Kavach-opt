#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace kavach::benchmark {

enum class ReferenceSolverStatus {
    OPTIMAL,
    FEASIBLE,
    INFEASIBLE,
    UNBOUNDED,
    TIMEOUT,
    UNAVAILABLE,
    ERROR
};

[[nodiscard]] std::string to_string(ReferenceSolverStatus s);

inline std::ostream& operator<<(std::ostream& os, ReferenceSolverStatus s) {
    return os << to_string(s);
}

/**
 * @brief Structured result from external reference solver execution.
 */
struct ReferenceRunResult {
    ReferenceSolverStatus status{ReferenceSolverStatus::UNAVAILABLE};
    std::string solver_name{"HiGHS"};
    std::string solver_version;
    std::optional<double> objective;
    std::optional<double> runtime_seconds;
    std::optional<std::size_t> iterations;
    std::optional<std::size_t> nodes;
    std::optional<double> gap;
    std::string raw_output;
    std::string error_message;
};

/**
 * @brief Isolated external process adapter for reference solver comparison (Section 14-16).
 *
 * CRITICAL ARCHITECTURAL BOUNDARY:
 * - HiGHS is an external reference only.
 * - Kavach-Opt must NEVER link against HiGHS.
 * - Subprocess execution only; zero library includes.
 */
class ExternalReferenceAdapter {
public:
    explicit ExternalReferenceAdapter(std::string custom_bin_path = "");

    [[nodiscard]] bool is_available() const noexcept;
    [[nodiscard]] const std::string& executable_path() const noexcept { return binary_path_; }

    [[nodiscard]] ReferenceRunResult run(const std::string& model_path,
                                         double time_limit_seconds = 60.0) const;

    /// Public fixture parser to test parsing without requiring external installation
    [[nodiscard]] static ReferenceRunResult parse_output(std::string_view stdout_str,
                                                         std::string_view stderr_str, int exit_code,
                                                         bool timed_out);

    [[nodiscard]] static std::string discover_binary();

private:
    std::string binary_path_;
};

}  // namespace kavach::benchmark
