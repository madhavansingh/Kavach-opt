#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "backend/cuda/cuda_backend.hpp"
#include "backend/execution_backend.hpp"
#include "benchmark/benchmark_case.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"

namespace kavach::benchmark {

/**
 * @brief Status Taxonomy for benchmark results (Section 6 & 26).
 */
enum class BenchmarkStatus {
    OPTIMAL,
    FEASIBLE,
    INFEASIBLE,
    UNBOUNDED,
    LIMIT_REACHED,
    NUMERICAL_FAILURE,
    PARSE_FAILURE,
    VERIFICATION_FAILURE,
    UNSUPPORTED,
    RUNTIME_ERROR,
    SOLVER_TIMEOUT,
    REFERENCE_UNAVAILABLE,
    REFERENCE_TIMEOUT
};

[[nodiscard]] std::string to_string(BenchmarkStatus status);
[[nodiscard]] BenchmarkStatus parse_benchmark_status(std::string_view str);
[[nodiscard]] BenchmarkStatus from_solver_status(core::SolverStatus s);

inline std::ostream& operator<<(std::ostream& os, BenchmarkStatus s) {
    return os << to_string(s);
}

/**
 * @brief Normalized comparison between solver and reference objectives (Section 17).
 */
struct ObjectiveComparison {
    double abs_error{0.0};
    double rel_error{0.0};
    bool matches{false};
};

[[nodiscard]] ObjectiveComparison compare_objectives(double solver_obj, double ref_obj,
                                                     double tol_rel = 1e-4, double tol_abs = 1e-4);

/**
 * @brief Structured result for every attempted benchmark instance (Section 5).
 */
struct BenchmarkResult {
    // Identification
    std::string timestamp;  ///< ISO 8601 UTC timestamp
    std::string suite;
    std::string case_id;
    std::string instance_name;
    std::string instance_path;
    std::string problem_type;  ///< "LP", "MILP", "QP", "SPMV"
    std::vector<std::string> tags;

    // Dimensions (null if parse failed)
    std::optional<std::size_t> variables;
    std::optional<std::size_t> constraints;
    std::optional<std::size_t> nonzeros;
    std::optional<double> density;

    // Solver metadata
    std::string solver_name{"Kavach-Opt"};
    std::string solver_version;
    BenchmarkStatus status{BenchmarkStatus::RUNTIME_ERROR};
    std::string verified_status{"UNVERIFIED"};
    bool verification_passed{false};

    // Solution & Objectives
    std::optional<double> objective_value;
    std::optional<double> reference_objective;
    std::optional<std::string> reference_status;
    std::optional<double> objective_error_abs;
    std::optional<double> objective_error_rel;
    std::optional<double> best_bound;
    std::optional<double> optimality_gap;

    // Counts
    std::optional<std::size_t> iterations;
    std::optional<std::size_t> nodes;

    // Timings (seconds)
    double total_wall_time_seconds{0.0};
    std::optional<double> parse_time_seconds;
    std::optional<double> presolve_time_seconds;
    std::optional<double> solve_time_seconds;
    std::optional<double> verification_time_seconds;

    // Residuals & Verification
    std::optional<double> primal_residual;
    std::optional<double> dual_residual;
    std::optional<double> kkt_residual;
    std::optional<double> integrality_violation;

    // Memory (null/unknown if platform does not expose reliable metric)
    std::optional<int64_t> memory_bytes;

    // Execution environment & backend
    std::string backend{"CPU"};
    backend::BackendType backend_type{backend::BackendType::CPU_SEQUENTIAL};
    std::size_t num_threads{1};
    std::optional<std::string> gpu_device;
    std::optional<backend::cuda::CudaTelemetry> gpu_telemetry;
    std::string cpu_details;
    std::string compiler;
    std::string build_type;
    std::string git_commit;
    uint64_t random_seed{42};
    core::SolverTolerances tolerances;

    // Diagnostic / Error message
    std::string error_message;

    // External Reference Comparison (HiGHS) (Section 14-16)
    bool has_reference_comparison{false};
    std::string reference_solver_name{"HiGHS"};
    std::optional<std::string> reference_solver_version;
    std::optional<std::string> reference_solver_status;
    std::optional<double> reference_objective_val;
    std::optional<double> reference_runtime_seconds;
    std::optional<std::size_t> reference_iterations;
    std::optional<double> reference_gap;
    std::string reference_error_message;
};

/**
 * @brief Environment & Reproducibility Manifest (Section 20).
 */
struct ReproducibilityManifest {
    std::string git_commit;
    std::string timestamp;
    std::string hostname;
    std::string cpu_model;
    std::size_t num_cpu_cores{1};
    std::string gpu_device{"None (CPU fallback)"};
    std::string os_name;
    std::string compiler_info;
    std::string build_type;
    std::string kavach_version;
    std::string suite_name;
    std::string manifest_version{"1.0.0"};
    uint64_t seed{42};
    std::string reference_solver_info{"None"};

    [[nodiscard]] static ReproducibilityManifest capture_current_environment(
        const std::string& suite = "benchmark");
};

}  // namespace kavach::benchmark
