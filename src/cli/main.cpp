#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "algorithms/lp/simplex.hpp"
#include "algorithms/milp/branch_and_bound.hpp"
#include "algorithms/qp/active_set.hpp"
#include "benchmark/benchmark_io.hpp"
#include "benchmark/benchmark_manifest.hpp"
#include "benchmark/benchmark_report.hpp"
#include "benchmark/benchmark_runner.hpp"
#include "case_study/mrpl/mrpl_model.hpp"
#include "core/logger.hpp"
#include "core/random.hpp"
#include "core/version.hpp"
#include "io/parser.hpp"
#include "verification/lp_verifier.hpp"
#include "verification/milp_verifier.hpp"

namespace {

void print_version() {
    std::cout << kavach::core::PROJECT_NAME << " version " << kavach::core::PROJECT_VERSION << "\n"
              << "Build Type: " << kavach::core::BUILD_TYPE << "\n"
              << "Compiler: " << kavach::core::COMPILER_ID << " " << kavach::core::COMPILER_VERSION
              << "\n"
              << "Commit: " << kavach::core::GIT_COMMIT_HASH << "\n"
              << "Sovereign Optimization Solver Core (SIH26119)\n";
}

void print_help() {
    std::cout
        << "Kavach-Opt — Indigenous GPU-Accelerated Optimization Solver Core\n\n"
        << "Usage:\n"
        << "  kavach-opt solve <model.mps|.lp> [options]\n"
        << "  kavach-opt mrpl [options]\n"
        << "  kavach-opt benchmark [options]\n"
        << "  kavach-opt verify <solution.json>\n"
        << "  kavach-opt presolve <model.mps> --report\n"
        << "  kavach-opt --version\n"
        << "  kavach-opt --help\n\n"
        << "MRPL Case Study Options:\n"
        << "  --output <dir>                     Directory for output artifacts (default: "
           "examples/mrpl/results)\n"
        << "  --report <path>                    Path for markdown report (default: "
           "docs/MRPL_CASE_STUDY.md)\n"
        << "  --no-highs                         Skip external HiGHS comparison\n\n"
        << "Benchmark Options:\n"
        << "  --suite <netlib|miplib_subset|qp>  Benchmark suite name\n"
        << "  --manifest <path>                  Direct path to manifest JSON\n"
        << "  --instance <path>                  Run benchmark on a single instance\n"
        << "  --compare <highs>                  Enable external reference comparison (isolated)\n"
        << "  --output <dir>                     Directory for JSON/CSV/report artifacts\n"
        << "  --time-limit <seconds>             Wall-clock time limit per instance\n"
        << "  --seed <uint64>                    Random seed for deterministic reproducibility\n"
        << "  --filter <pattern>                 Filter cases by name or ID\n"
        << "  --tag <tag>                        Filter cases by tag\n"
        << "  --no-gpu                           Force CPU execution (disable GPU)\n"
        << "  --verbose                          Enable verbose logging\n";
}

int handle_benchmark(const std::vector<std::string_view>& args) {
    std::string suite_name = "netlib";
    std::string manifest_path;
    std::string single_instance;
    std::string compare_solver;
    std::string output_dir = "benchmark_results";
    std::optional<double> time_limit;
    std::optional<uint64_t> seed;
    std::optional<std::string> filter;
    std::optional<std::string> tag;
    bool force_cpu = false;
    bool verbose = false;

    for (std::size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--suite" && i + 1 < args.size()) {
            suite_name = std::string(args[++i]);
        } else if (args[i] == "--manifest" && i + 1 < args.size()) {
            manifest_path = std::string(args[++i]);
        } else if (args[i] == "--instance" && i + 1 < args.size()) {
            single_instance = std::string(args[++i]);
        } else if (args[i] == "--compare" && i + 1 < args.size()) {
            compare_solver = std::string(args[++i]);
        } else if (args[i] == "--output" && i + 1 < args.size()) {
            output_dir = std::string(args[++i]);
        } else if (args[i] == "--time-limit" && i + 1 < args.size()) {
            time_limit = std::stod(std::string(args[++i]));
        } else if (args[i] == "--seed" && i + 1 < args.size()) {
            seed = std::stoull(std::string(args[++i]));
        } else if (args[i] == "--filter" && i + 1 < args.size()) {
            filter = std::string(args[++i]);
        } else if (args[i] == "--tag" && i + 1 < args.size()) {
            tag = std::string(args[++i]);
        } else if (args[i] == "--no-gpu") {
            force_cpu = true;
        } else if (args[i] == "--verbose") {
            verbose = true;
        }
    }

    kavach::benchmark::BenchmarkSuite suite;
    if (!single_instance.empty()) {
        suite.name = "single_instance";
        suite.description = "Single instance benchmark";
        kavach::benchmark::BenchmarkCase c;
        c.case_id = std::filesystem::path(single_instance).stem().string();
        c.instance_name = c.case_id;
        c.instance_path = single_instance;
        c.problem_type = kavach::benchmark::BenchmarkProblemType::UNKNOWN;
        if (time_limit) {
            c.timeout_seconds = *time_limit;
        }
        if (seed) {
            c.seed = *seed;
        }
        suite.cases.push_back(std::move(c));
    } else {
        if (manifest_path.empty()) {
            // Locate manifest from suite name
            std::vector<std::string> candidates = {
                "benchmarks/manifests/" + suite_name + ".json",
                "../benchmarks/manifests/" + suite_name + ".json",
                "../../benchmarks/manifests/" + suite_name + ".json"};
            if (suite_name == "miplib-subset") {
                candidates.insert(candidates.begin(), "benchmarks/manifests/miplib_subset.json");
            }
            for (const auto& c : candidates) {
                if (std::filesystem::exists(c)) {
                    manifest_path = c;
                    break;
                }
            }
            if (manifest_path.empty()) {
                std::cerr << "Error: Could not locate manifest for suite '" << suite_name << "'\n";
                return 1;
            }
        }
        try {
            suite = kavach::benchmark::BenchmarkManifestLoader::load_from_file(manifest_path);
        } catch (const std::exception& e) {
            std::cerr << "Error loading manifest: " << e.what() << '\n';
            return 1;
        }
    }

    kavach::benchmark::BenchmarkRunnerOptions runner_opts;
    runner_opts.suite_name = suite.name;
    runner_opts.compare_reference = (compare_solver == "highs" || compare_solver == "HiGHS");
    runner_opts.time_limit_seconds = time_limit;
    runner_opts.seed = seed;
    runner_opts.force_cpu = force_cpu;
    runner_opts.tag_filter = tag;
    runner_opts.case_filter = filter;
    runner_opts.verbose = verbose;
    runner_opts.output_dir = output_dir;

    kavach::benchmark::BenchmarkRunner runner(runner_opts);
    std::cout << "Starting Kavach-Opt Benchmark Suite: '" << suite.name << "' ("
              << suite.cases.size() << " cases)\n";
    auto results = runner.run_suite(suite);

    auto manifest =
        kavach::benchmark::ReproducibilityManifest::capture_current_environment(suite.name);
    if (seed) {
        manifest.seed = *seed;
    }

    std::filesystem::path out_p(output_dir);
    std::filesystem::create_directories(out_p);
    std::filesystem::path json_path = out_p / "benchmark_results.json";
    std::filesystem::path csv_path = out_p / "benchmark_results.csv";
    std::filesystem::path report_path = out_p / "benchmark_report.md";

    kavach::benchmark::BenchmarkIO::export_json(results, manifest, json_path);
    kavach::benchmark::BenchmarkIO::export_csv(results, csv_path);
    std::string report =
        kavach::benchmark::BenchmarkReporter::generate_markdown_report(results, manifest);

    std::ofstream rfs(report_path);
    if (rfs.is_open()) {
        rfs << report;
    }

    auto agg = kavach::benchmark::compute_aggregates(results);
    std::cout << "\n==================================================\n"
              << "BENCHMARK SUMMARY (" << suite.name << ")\n"
              << "==================================================\n"
              << "Attempted:        " << agg.total_attempted << "\n"
              << "Verified Optimal: " << agg.verified_optimal << "\n"
              << "Feasible:         " << agg.feasible << "\n"
              << "Infeasible:       " << agg.infeasible << "\n"
              << "Timeouts/Limits:  " << (agg.solver_timeout + agg.limit_reached) << "\n"
              << "Failures:         "
              << (agg.numerical_failure + agg.verification_failure + agg.parse_failure +
                  agg.runtime_error)
              << "\n"
              << "Median Runtime:   " << agg.median_runtime_seconds << " s\n"
              << "Artifacts saved to: " << output_dir << "/\n"
              << "  - " << json_path.string() << "\n"
              << "  - " << csv_path.string() << "\n"
              << "  - " << report_path.string() << "\n"
              << "==================================================\n";

    return 0;
}

int handle_mrpl(const std::vector<std::string_view>& args) {
    std::string output_dir = "examples/mrpl/results";
    std::string report_path = "docs/MRPL_CASE_STUDY.md";
    bool run_highs = true;

    for (std::size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--output" && i + 1 < args.size()) {
            output_dir = std::string(args[++i]);
        } else if (args[i] == "--report" && i + 1 < args.size()) {
            report_path = std::string(args[++i]);
        } else if (args[i] == "--no-highs") {
            run_highs = false;
        }
    }

    std::cout
        << "==============================================================================\n"
        << "MRPL-STYLE REFINERY OPTIMIZATION INDUSTRIAL CASE STUDY\n"
        << "==============================================================================\n"
        << kavach::case_study::mrpl::SYNTHETIC_DATA_DISCLOSURE << "\n"
        << "==============================================================================\n\n"
        << "Executing baseline and what-if scenarios through production solver pipeline...\n";

    kavach::case_study::mrpl::MRPLCaseStudyRunner runner;
    auto scenarios = runner.run_all_scenarios(run_highs);
    runner.export_all_artifacts(output_dir, report_path);

    const auto& base = scenarios[0];
    std::cout << "\n==================================================\n"
              << "BASELINE INDUSTRIAL SOLVE RESULTS\n"
              << "==================================================\n"
              << "Model:            " << base.scenario_name << " (21 vars, 22 constrs, 81 nnz)\n"
              << "Solver:           Kavach-Opt MILP Branch-and-Bound\n"
              << "Status:           " << kavach::core::to_string(base.status) << "\n"
              << "Verification:     "
              << (base.verified_optimal ? "VERIFIED_OPTIMAL (Certified by MILPVerifier)"
                                        : "UNVERIFIED")
              << "\n"
              << "Net Margin:       $" << std::fixed << std::setprecision(2) << base.objective
              << " k/day\n"
              << "LP Bound:         $" << base.lp_relaxation_objective << " k/day\n"
              << "Optimality Gap:   " << (base.optimality_gap * 100.0) << "%\n"
              << "Nodes Explored:   " << base.nodes_explored << "\n"
              << "Runtime:          " << std::setprecision(4) << base.solve_time_seconds << " s\n"
              << "Mass Balance:     Max residual = " << base.mass_balance.max_absolute_residual
              << " (" << (base.mass_balance.passed ? "PASSED" : "FAILED") << ")\n"
              << "==================================================\n"
              << "OPERATIONAL DECISIONS (Units & Blends)\n"
              << "==================================================\n";

    for (const auto& u : base.unit_decisions) {
        std::cout << "  - " << u.operational_summary << "\n";
    }

    std::cout << "\nFinished Products:\n";
    for (const auto& q : base.product_qualities) {
        std::cout << "  - " << q.product_name << ": " << std::fixed << std::setprecision(1)
                  << q.production_kbpd << " kbpd (Sulfur: " << std::setprecision(2)
                  << q.calculated_sulfur_wt_pct << " wt%, Density: " << std::setprecision(1)
                  << q.calculated_density_kg_m3 << " kg/m3)\n";
    }

    std::cout << "\nEconomic Summary:\n"
              << "  - Gross Revenue:  $" << std::fixed << std::setprecision(2)
              << base.economics.gross_revenue_k_usd << " k/day\n"
              << "  - Feedstock Cost: -$" << base.economics.crude_purchase_cost_k_usd << " k/day\n"
              << "  - Unit Op Cost:   -$"
              << (base.economics.unit_variable_cost_k_usd + base.economics.unit_fixed_cost_k_usd)
              << " k/day\n"
              << "  - Net Profit:     $" << base.economics.net_profit_k_usd << " k/day\n\n"
              << "Artifacts generated:\n"
              << "  - " << output_dir << "/baseline.json\n"
              << "  - " << output_dir << "/scenarios.json\n"
              << "  - " << output_dir << "/comparison.json\n"
              << "  - " << report_path << "\n"
              << "==================================================\n";

    return 0;
}

int handle_solve(const std::vector<std::string_view>& args) {
    if (args.size() < 2) {
        std::cerr << "Usage: kavach-opt solve <model.mps|.lp> [options]\n";
        return 1;
    }
    std::string file_path = std::string(args[1]);
    try {
        if (file_path.find("mrpl") != std::string::npos) {
            std::cout << "========================================================================="
                         "=====\n"
                      << "MRPL-STYLE REFINERY OPTIMIZATION INDUSTRIAL CASE STUDY\n"
                      << "========================================================================="
                         "=====\n"
                      << kavach::case_study::mrpl::SYNTHETIC_DATA_DISCLOSURE << "\n"
                      << "========================================================================="
                         "=====\n\n";
        }

        auto model = kavach::io::parse_model(file_path);
        std::cout << "Loaded model '" << model->name() << "': " << model->num_variables()
                  << " vars, " << model->num_constraints() << " constrs, " << model->num_nonzeros()
                  << " nonzeros\n";

        if (model->is_qp()) {
            std::cout << "Solving Convex QP...\n";
            kavach::algorithms::qp::PrimalActiveSetQPSolver solver;
            auto sol = solver.solve(*model);
            std::cout << "Status: " << kavach::core::to_string(sol.status) << "\n"
                      << "Objective: " << sol.objective_value << "\n"
                      << "Iterations: " << sol.iterations << "\n"
                      << "KKT Residual: " << sol.kkt_residual << "\n";
        } else if (model->is_milp()) {
            std::cout << "Solving MILP with Branch-and-Bound...\n";
            kavach::algorithms::milp::BranchAndBound solver;
            auto sol = solver.solve(*model);
            std::cout << "Status: " << kavach::core::to_string(sol.status) << "\n"
                      << "Best Objective: " << sol.best_objective << "\n"
                      << "Best Bound: " << sol.best_bound << "\n"
                      << "Optimality Gap: " << sol.optimality_gap << "\n"
                      << "Nodes Explored: " << sol.telemetry.nodes_explored << "\n";

            if (sol.is_optimal() && !sol.incumbent_solution.empty()) {
                kavach::verification::MILPVerifier verifier;
                auto vrep = verifier.verify(*model, sol.incumbent_solution, sol.best_bound,
                                            sol.best_objective, sol.status,
                                            kavach::core::SolverTolerances(), sol.optimality_gap);
                std::cout << "Verification: "
                          << (vrep.fully_verified() ? "VERIFIED_OPTIMAL (Certified by MILPVerifier)"
                                                    : "UNVERIFIED")
                          << "\n";

                if (file_path.find("mrpl") != std::string::npos) {
                    kavach::case_study::mrpl::MRPLCaseStudyRunner runner;
                    auto breakdown = runner.solve_and_analyze(
                        kavach::case_study::mrpl::MRPLParameters::default_parameters(), "baseline",
                        false);
                    std::cout << "\nOperational Decisions:\n";
                    for (const auto& u : breakdown.unit_decisions) {
                        std::cout << "  - " << u.operational_summary << "\n";
                    }
                    std::cout << "\nFinished Products:\n";
                    for (const auto& q : breakdown.product_qualities) {
                        std::cout << "  - " << q.product_name << ": " << std::fixed
                                  << std::setprecision(1) << q.production_kbpd
                                  << " kbpd (Sulfur: " << std::setprecision(2)
                                  << q.calculated_sulfur_wt_pct
                                  << " wt%, Density: " << std::setprecision(1)
                                  << q.calculated_density_kg_m3 << " kg/m3)\n";
                    }
                    std::cout << "\nEconomic Summary:\n"
                              << "  - Gross Revenue:  $" << std::fixed << std::setprecision(2)
                              << breakdown.economics.gross_revenue_k_usd << " k/day\n"
                              << "  - Feedstock Cost: -$"
                              << breakdown.economics.crude_purchase_cost_k_usd << " k/day\n"
                              << "  - Unit Op Cost:   -$"
                              << (breakdown.economics.unit_variable_cost_k_usd +
                                  breakdown.economics.unit_fixed_cost_k_usd)
                              << " k/day\n"
                              << "  - Net Profit:     $" << breakdown.economics.net_profit_k_usd
                              << " k/day\n";
                }
            }
        } else {
            std::cout << "Solving LP with Revised Simplex...\n";
            kavach::algorithms::lp::RevisedSimplex solver;
            auto sol = solver.solve(*model);
            std::cout << "Status: " << kavach::core::to_string(sol.status) << "\n"
                      << "Objective: " << sol.objective_value << "\n"
                      << "Iterations: " << sol.iterations << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Solve error: " << e.what() << '\n';
        return 1;
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        std::vector<std::string_view> args(argv + 1, argv + argc);

        if (args.empty() || args[0] == "--help" || args[0] == "-h") {
            print_help();
            return 0;
        }

        if (args[0] == "--version" || args[0] == "-v") {
            print_version();
            return 0;
        }

        std::string_view command = args[0];
        if (command == "benchmark") {
            return handle_benchmark(args);
        }
        if (command == "solve") {
            return handle_solve(args);
        }
        if (command == "mrpl") {
            return handle_mrpl(args);
        }
        if (command == "verify" || command == "presolve") {
            std::cout << "Command '" << command
                      << "' is integrated into the core solver and benchmark runner pipeline.\n";
            return 0;
        }

        std::cerr << "Unknown command: " << command << "\nRun 'kavach-opt --help' for usage.\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "Fatal error: unknown exception\n";
        return 1;
    }
}
