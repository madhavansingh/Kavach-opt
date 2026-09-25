#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/tolerances.hpp"

namespace kavach::benchmark {

enum class BenchmarkProblemType { LP, MILP, QP, SPMV, UNKNOWN };

[[nodiscard]] std::string to_string(BenchmarkProblemType type);
[[nodiscard]] BenchmarkProblemType parse_problem_type(std::string_view str);

/**
 * @brief Benchmark case definition (Section 4).
 */
struct BenchmarkCase {
    std::string case_id;        ///< Unique case identifier (e.g. "netlib-afiro")
    std::string instance_name;  ///< Problem name (e.g. "afiro")
    std::string instance_path;  ///< Path to problem file (.mps / .lp)
    BenchmarkProblemType problem_type{BenchmarkProblemType::LP};
    std::string source{"Netlib"};   ///< Source corpus (e.g. "Netlib", "MIPLIB")
    std::vector<std::string> tags;  ///< Categories: small, medium, large, sparse, degenerate, etc.
    std::optional<double> timeout_seconds{60.0};
    std::optional<double> reference_objective;               ///< Known optimal objective
    std::optional<std::string> reference_status{"OPTIMAL"};  ///< Known status
    std::optional<double> reference_best_bound;              ///< Known best bound (MILP)
    uint64_t seed{42};
    bool enable_presolve{true};
    bool enable_scaling{true};
    core::SolverTolerances tolerances;
    std::size_t max_iterations{100000};
    std::size_t max_nodes{50000};
    std::size_t spmv_batch_size{64};  ///< Batch size for SpMV cases

    [[nodiscard]] bool has_tag(std::string_view tag) const noexcept;
};

/**
 * @brief Benchmark suite comprising a collection of cases and metadata.
 */
struct BenchmarkSuite {
    std::string name;
    std::string description;
    std::string version{"1.0.0"};
    std::vector<BenchmarkCase> cases;

    [[nodiscard]] std::size_t size() const noexcept { return cases.size(); }
    [[nodiscard]] bool empty() const noexcept { return cases.empty(); }
    [[nodiscard]] const BenchmarkCase* find_case(std::string_view id) const noexcept;
};

}  // namespace kavach::benchmark
