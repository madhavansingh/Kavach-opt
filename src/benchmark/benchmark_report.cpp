#include "benchmark/benchmark_report.hpp"

#include <algorithm>
#include <iomanip>
#include <map>
#include <sstream>

namespace kavach::benchmark {

AggregateStatistics compute_aggregates(const std::vector<BenchmarkResult>& results) {
    AggregateStatistics agg;
    agg.total_attempted = results.size();
    if (results.empty()) {
        return agg;
    }

    std::vector<double> runtimes;
    std::vector<double> rel_errors;
    std::vector<double> speedups;
    double total_iters = 0.0;
    std::size_t count_iters = 0;
    double total_nodes = 0.0;
    std::size_t count_nodes = 0;

    for (const auto& r : results) {
        runtimes.push_back(r.total_wall_time_seconds);

        switch (r.status) {
            case BenchmarkStatus::OPTIMAL:
                if (r.verification_passed) {
                    ++agg.verified_optimal;
                }
                break;
            case BenchmarkStatus::FEASIBLE:
                ++agg.feasible;
                break;
            case BenchmarkStatus::INFEASIBLE:
                ++agg.infeasible;
                break;
            case BenchmarkStatus::UNBOUNDED:
                ++agg.unbounded;
                break;
            case BenchmarkStatus::LIMIT_REACHED:
                ++agg.limit_reached;
                break;
            case BenchmarkStatus::NUMERICAL_FAILURE:
                ++agg.numerical_failure;
                break;
            case BenchmarkStatus::PARSE_FAILURE:
                ++agg.parse_failure;
                break;
            case BenchmarkStatus::VERIFICATION_FAILURE:
                ++agg.verification_failure;
                break;
            case BenchmarkStatus::UNSUPPORTED:
                ++agg.unsupported;
                break;
            case BenchmarkStatus::RUNTIME_ERROR:
                ++agg.runtime_error;
                break;
            case BenchmarkStatus::SOLVER_TIMEOUT:
                ++agg.solver_timeout;
                break;
            case BenchmarkStatus::REFERENCE_UNAVAILABLE:
                ++agg.reference_unavailable;
                break;
            case BenchmarkStatus::REFERENCE_TIMEOUT:
                break;
        }

        if (r.has_reference_comparison && r.reference_solver_status == "REFERENCE_UNAVAILABLE") {
            ++agg.reference_unavailable;
        }

        if (r.objective_error_rel.has_value()) {
            rel_errors.push_back(*r.objective_error_rel);
        }

        if (r.iterations.has_value()) {
            total_iters += static_cast<double>(*r.iterations);
            ++count_iters;
        }

        if (r.nodes.has_value()) {
            total_nodes += static_cast<double>(*r.nodes);
            ++count_nodes;
        }

        if (r.gpu_telemetry.has_value()) {
            speedups.push_back(r.gpu_telemetry->transfer_inclusive_speedup);
        }
    }

    std::sort(runtimes.begin(), runtimes.end());
    agg.min_runtime_seconds = runtimes.front();
    agg.max_runtime_seconds = runtimes.back();
    agg.median_runtime_seconds = runtimes[runtimes.size() / 2];

    if (!rel_errors.empty()) {
        std::sort(rel_errors.begin(), rel_errors.end());
        agg.median_objective_error_rel = rel_errors[rel_errors.size() / 2];
    }

    if (count_iters > 0) {
        agg.mean_iterations = total_iters / static_cast<double>(count_iters);
    }
    if (count_nodes > 0) {
        agg.mean_nodes = total_nodes / static_cast<double>(count_nodes);
    }

    if (!speedups.empty()) {
        std::sort(speedups.begin(), speedups.end());
        agg.median_spmv_speedup = speedups[speedups.size() / 2];
    }

    return agg;
}

std::string BenchmarkReporter::generate_markdown_report(const std::vector<BenchmarkResult>& results,
                                                        const ReproducibilityManifest& manifest) {
    std::ostringstream ss;
    AggregateStatistics agg = compute_aggregates(results);

    ss << "# Kavach-Opt Benchmark & Reproducibility Report\n\n";

    // 1. Executive Methodology
    ss << "## 1. Executive Methodology\n\n"
       << "This report presents an objective evaluation of Kavach-Opt across standard mathematical "
          "programming benchmark instances.\n"
       << "Key methodological rules:\n"
       << "- Every attempted instance is recorded with first-class status preservation.\n"
       << "- An instance is certified `OPTIMAL` only if confirmed by independent verification "
          "layers.\n"
       << "- Time limit and numerical safeguards are enforced deterministically.\n"
       << "- External reference solver comparisons (HiGHS) execute in completely isolated "
          "subprocesses.\n\n";

    // 2. Hardware / Software Environment
    ss << "## 2. Hardware and Software Environment\n\n"
       << "| Property | Value |\n"
       << "|---|---|\n"
       << "| Solver | " << "Kavach-Opt v" << manifest.kavach_version << " |\n"
       << "| Git Commit | `" << manifest.git_commit << "` |\n"
       << "| Timestamp (UTC) | " << manifest.timestamp << " |\n"
       << "| Operating System | " << manifest.os_name << " |\n"
       << "| Hostname | " << manifest.hostname << " |\n"
       << "| CPU Model | " << manifest.cpu_model << " (" << manifest.num_cpu_cores
       << " logical cores) |\n"
       << "| GPU Accelerator | " << manifest.gpu_device << " |\n"
       << "| C++ Compiler | " << manifest.compiler_info << " |\n"
       << "| Build Type | " << manifest.build_type << " |\n"
       << "| Random Seed | " << manifest.seed << " |\n\n";

    // 3. Suite Coverage
    ss << "## 3. Suite Coverage\n\n"
       << "| Problem Type | Attempted | Verified Optimal | Feasible | Infeasible | Timeout | "
          "Failed "
          "|\n"
       << "|---|---|---|---|---|---|---|\n";

    std::map<std::string, std::vector<BenchmarkResult>> by_type;
    for (const auto& r : results) {
        by_type[r.problem_type].push_back(r);
    }
    for (const auto& [type_name, type_results] : by_type) {
        auto sub_agg = compute_aggregates(type_results);
        ss << "| " << type_name << " | " << sub_agg.total_attempted << " | "
           << sub_agg.verified_optimal << " | " << sub_agg.feasible << " | " << sub_agg.infeasible
           << " | " << sub_agg.solver_timeout << " | "
           << (sub_agg.numerical_failure + sub_agg.verification_failure + sub_agg.parse_failure +
               sub_agg.runtime_error)
           << " |\n";
    }
    ss << "\n";

    // 4. Per-Instance Detailed Table
    ss << "## 4. Per-Instance Detailed Results\n\n"
       << "| Case ID | Type | Vars | Constrs | Status | Verified | Objective | Ref Objective | Rel "
          "Err | Time (s) | Iters/Nodes |\n"
       << "|---|---|---|---|---|---|---|---|---|---|---|\n";

    for (const auto& r : results) {
        ss << "| `" << r.case_id << "` | " << r.problem_type << " | "
           << (r.variables ? std::to_string(*r.variables) : "-") << " | "
           << (r.constraints ? std::to_string(*r.constraints) : "-") << " | " << to_string(r.status)
           << " | " << r.verified_status << " | ";

        if (r.objective_value) {
            ss << std::setprecision(8) << *r.objective_value;
        } else {
            ss << "-";
        }
        ss << " | ";

        if (r.reference_objective) {
            ss << std::setprecision(8) << *r.reference_objective;
        } else {
            ss << "-";
        }
        ss << " | ";

        if (r.objective_error_rel) {
            ss << std::scientific << std::setprecision(2) << *r.objective_error_rel
               << std::defaultfloat;
        } else {
            ss << "-";
        }
        ss << " | " << std::fixed << std::setprecision(4) << r.total_wall_time_seconds << " | ";

        if (r.iterations) {
            ss << *r.iterations << " it";
        } else if (r.nodes) {
            ss << *r.nodes << " nd";
        } else {
            ss << "-";
        }
        ss << " |\n";
    }
    ss << "\n";

    // 5. Aggregate Summary
    ss << "## 5. Aggregate Summary\n\n"
       << "All aggregate metrics are strictly derived from the per-instance records above:\n\n"
       << "- **Total Instances Attempted**: " << agg.total_attempted << "\n"
       << "- **Verified Optimal**: " << agg.verified_optimal << " (" << std::fixed
       << std::setprecision(1)
       << (agg.total_attempted > 0 ? 100.0 * static_cast<double>(agg.verified_optimal) /
                                         static_cast<double>(agg.total_attempted)
                                   : 0.0)
       << "%)\n"
       << "- **Infeasible Certifications**: " << agg.infeasible << "\n"
       << "- **Timeouts / Limits Hit**: " << (agg.solver_timeout + agg.limit_reached) << "\n"
       << "- **Verification / Numerical Failures**: "
       << (agg.verification_failure + agg.numerical_failure) << "\n"
       << "- **Median Solve Runtime**: " << std::fixed << std::setprecision(4)
       << agg.median_runtime_seconds << " s (Min: " << agg.min_runtime_seconds
       << " s, Max: " << agg.max_runtime_seconds << " s)\n"
       << "- **Median Relative Objective Error**: " << std::scientific << std::setprecision(2)
       << agg.median_objective_error_rel << std::defaultfloat << "\n"
       << "- **Mean Simplex Iterations**: " << std::fixed << std::setprecision(1)
       << agg.mean_iterations << "\n\n";

    // 6. Failure & Timeout Table
    ss << "## 6. Failures, Timeouts, and Unsupported Cases\n\n";
    std::vector<BenchmarkResult> failures;
    for (const auto& r : results) {
        if (r.status != BenchmarkStatus::OPTIMAL && r.status != BenchmarkStatus::FEASIBLE &&
            r.status != BenchmarkStatus::INFEASIBLE) {
            failures.push_back(r);
        }
    }
    if (failures.empty()) {
        ss << "Zero failures, timeouts, or unhandled errors detected across all tested cases.\n\n";
    } else {
        ss << "| Case ID | Status | Message |\n"
           << "|---|---|---|\n";
        for (const auto& f : failures) {
            ss << "| `" << f.case_id << "` | " << to_string(f.status) << " | "
               << (f.error_message.empty() ? "-" : f.error_message) << " |\n";
        }
        ss << "\n";
    }

    // 7. Independent Verification Summary
    ss << "## 7. Independent Verification Summary\n\n"
       << "Every reported `OPTIMAL` solution is independently verified by `LPVerifier`, "
          "`MILPVerifier`, or `QPVerifier` without solver state reuse:\n\n"
       << "| Case ID | Verifier | Primal Residual | Dual Residual | Integrality Viol | Certified "
          "|\n"
       << "|---|---|---|---|---|---|\n";
    for (const auto& r : results) {
        if (r.status == BenchmarkStatus::OPTIMAL) {
            ss << "| `" << r.case_id << "` | " << r.problem_type << "Verifier | ";
            if (r.primal_residual) {
                ss << std::scientific << std::setprecision(2) << *r.primal_residual
                   << std::defaultfloat;
            } else {
                ss << "-";
            }
            ss << " | ";
            if (r.dual_residual) {
                ss << std::scientific << std::setprecision(2) << *r.dual_residual
                   << std::defaultfloat;
            } else {
                ss << "-";
            }
            ss << " | ";
            if (r.integrality_violation) {
                ss << std::scientific << std::setprecision(2) << *r.integrality_violation
                   << std::defaultfloat;
            } else {
                ss << "0.0";
            }
            ss << " | " << r.verified_status << " |\n";
        }
    }
    ss << "\n";

    // 8. Kavach-Opt vs HiGHS Comparison
    ss << "## 8. External Reference Solver Comparison (HiGHS)\n\n";
    bool any_ref = false;
    for (const auto& r : results) {
        if (r.has_reference_comparison) {
            any_ref = true;
            break;
        }
    }
    if (!any_ref) {
        ss << "HiGHS reference comparison was not enabled for this run (run with `--compare highs` "
              "to activate).\n\n";
    } else {
        ss << "| Case ID | Kavach Status | Kavach Obj | HiGHS Status | HiGHS Obj | Kavach Time (s) "
              "| HiGHS Time (s) |\n"
           << "|---|---|---|---|---|---|---|\n";
        for (const auto& r : results) {
            if (r.has_reference_comparison) {
                ss << "| `" << r.case_id << "` | " << to_string(r.status) << " | "
                   << (r.objective_value ? std::to_string(*r.objective_value) : "-") << " | "
                   << (r.reference_solver_status ? *r.reference_solver_status : "-") << " | "
                   << (r.reference_objective_val ? std::to_string(*r.reference_objective_val) : "-")
                   << " | " << std::fixed << std::setprecision(4) << r.total_wall_time_seconds
                   << " | "
                   << (r.reference_runtime_seconds ? std::to_string(*r.reference_runtime_seconds)
                                                   : "-")
                   << " |\n";
            }
        }
        ss << "\n";
    }

    // 9. GPU Telemetry & Acceleration Analysis
    ss << "## 9. GPU Acceleration & Telemetry Analysis\n\n";
    if (manifest.gpu_device.find("unavailable") != std::string::npos) {
        ss << "CUDA GPU hardware is not available in the current execution environment ("
           << manifest.os_name
           << "). As specified in Blueprint Section 15 & Section 22, CPU fallback is active and "
              "no GPU speedup is claimed.\n\n";
    } else {
        ss << "GPU Batched-SpMV execution telemetry active.\n"
           << "- Device: " << manifest.gpu_device << "\n"
           << "- Median Transfer-Inclusive Speedup: " << std::fixed << std::setprecision(2)
           << agg.median_spmv_speedup << "x\n\n";
    }

    // 10. Reproducibility Instructions
    ss << "## 10. Reproducibility Instructions\n\n"
       << "To reproduce the exact results reported in this document:\n\n"
       << "```bash\n"
       << "# 1. Build Kavach-Opt in Release mode\n"
       << "cmake -B build-release -DCMAKE_BUILD_TYPE=Release\n"
       << "cmake --build build-release -j\n\n"
       << "# 2. Run the full benchmark suite with deterministic seed\n"
       << "./build-release/src/cli/kavach-opt benchmark --suite " << manifest.suite_name
       << " --output benchmark_results/\n"
       << "```\n";

    return ss.str();
}

}  // namespace kavach::benchmark
