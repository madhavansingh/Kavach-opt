#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "algorithms/lp/simplex.hpp"
#include "algorithms/milp/branch_and_bound.hpp"
#include "algorithms/qp/active_set.hpp"
#include "api/solver.hpp"
#include "benchmark/benchmark_io.hpp"
#include "benchmark/benchmark_manifest.hpp"
#include "benchmark/benchmark_report.hpp"
#include "benchmark/benchmark_runner.hpp"
#include "case_study/mrpl/mrpl_model.hpp"
#include "core/logger.hpp"
#include "core/random.hpp"
#include "core/version.hpp"
#include "io/parser.hpp"
#include "presolve/presolve.hpp"
#include "strategy/strategy.hpp"
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
        << "Kavach-Opt — Sovereign GPU-Accelerated Optimization Solver Core\n\n"
        << "Usage:\n"
        << "  kavach-opt solve <model.mps|.lp> [options]\n"
        << "  kavach-opt inspect <model.mps|.lp> [--json]\n"
        << "  kavach-opt mrpl [options]\n"
        << "  kavach-opt benchmark [options]\n"
        << "  kavach-opt verify <model.mps|.lp>\n"
        << "  kavach-opt presolve <model.mps|.lp> [--report]\n"
        << "  kavach-opt --version\n"
        << "  kavach-opt --help\n\n"
        << "Solve Options:\n"
        << "  --strategy <auto|conservative|fast> Strategy decision mode (default: auto)\n"
        << "  --show-strategy                    Display concise strategy decision\n"
        << "  --explain-strategy                 Display comprehensive strategy explanation\n"
        << "  --presolve <full|none|auto>        Explicit presolve override\n"
        << "  --scaling <on|off|auto>            Explicit scaling override\n"
        << "  --backend <cpu|multicore|cuda>     Explicit execution backend override\n"
        << "  --gpu                              Request CUDA GPU execution backend\n"
        << "  --no-gpu                           Disable CUDA GPU execution (force CPU)\n"
        << "  --time-limit <seconds>             Wall-clock time limit in seconds\n"
        << "  --cuts <off|root|periodic>         Cutting-plane strategy for MILP (default: root)\n"
        << "  --branching <rule>                 Branching rule: "
           "most_fractional|pseudocost|strong|reliability|auto (default: reliability)\n"
        << "  --heuristic <on|off>               Incumbent primal heuristic (default: on)\n"
        << "  --json                             Output strategy and solve result as JSON\n\n"
        << "MRPL Case Study Options:\n"
        << "  --scenario <name>                  Specific scenario to solve\n"
        << "  --json                             Output result as JSON\n"
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
    std::string scenario_arg;
    bool run_highs = true;
    bool json_output = false;

    for (std::size_t i = 1; i < args.size(); ++i) {
        if (args[i] == "--output" && i + 1 < args.size()) {
            output_dir = std::string(args[++i]);
        } else if (args[i] == "--report" && i + 1 < args.size()) {
            report_path = std::string(args[++i]);
        } else if (args[i] == "--scenario" && i + 1 < args.size()) {
            scenario_arg = std::string(args[++i]);
        } else if (args[i] == "--json") {
            json_output = true;
        } else if (args[i] == "--no-highs") {
            run_highs = false;
        }
    }

    kavach::case_study::mrpl::MRPLCaseStudyRunner runner;

    if (json_output) {
        if (!scenario_arg.empty() && scenario_arg != "all") {
            kavach::case_study::mrpl::MRPLParameters params =
                kavach::case_study::mrpl::MRPLParameters::default_parameters();
            std::string resolved_name = "baseline";

            if (scenario_arg == "scenario_a_reduced_crude" || scenario_arg == "reduced_crude" ||
                scenario_arg == "a") {
                params = kavach::case_study::mrpl::MRPLParameters::scenario_reduced_crude();
                resolved_name = "scenario_a_reduced_crude";
            } else if (scenario_arg == "scenario_b_tight_quality" ||
                       scenario_arg == "tight_quality" || scenario_arg == "b") {
                params = kavach::case_study::mrpl::MRPLParameters::scenario_tight_quality();
                resolved_name = "scenario_b_tight_quality";
            } else if (scenario_arg == "scenario_c_reduced_capacity" ||
                       scenario_arg == "reduced_capacity" || scenario_arg == "c") {
                params = kavach::case_study::mrpl::MRPLParameters::scenario_reduced_capacity();
                resolved_name = "scenario_c_reduced_capacity";
            } else if (scenario_arg == "scenario_d_infeasible" || scenario_arg == "infeasible" ||
                       scenario_arg == "d") {
                params = kavach::case_study::mrpl::MRPLParameters::scenario_infeasible();
                resolved_name = "scenario_d_infeasible";
            } else if (scenario_arg == "baseline") {
                params = kavach::case_study::mrpl::MRPLParameters::default_parameters();
                resolved_name = "baseline";
            } else {
                std::cerr << "Unknown MRPL scenario: " << scenario_arg << "\n";
                return 1;
            }

            auto breakdown = runner.solve_and_analyze(params, resolved_name, run_highs);
            std::cout << breakdown.to_json().dump() << "\n";
            return 0;
        } else {
            auto scenarios = runner.run_all_scenarios(run_highs);
            kavach::benchmark::JsonValue::Array arr;
            arr.reserve(scenarios.size());
            for (const auto& s : scenarios) {
                arr.push_back(s.to_json());
            }
            std::cout << kavach::benchmark::JsonValue(std::move(arr)).dump() << "\n";
            return 0;
        }
    }

    std::cout
        << "==============================================================================\n"
        << "MRPL-STYLE REFINERY OPTIMIZATION INDUSTRIAL CASE STUDY\n"
        << "==============================================================================\n"
        << kavach::case_study::mrpl::SYNTHETIC_DATA_DISCLOSURE << "\n"
        << "==============================================================================\n\n"
        << "Executing baseline and what-if scenarios through production solver pipeline...\n";

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
    bool show_strategy = false;
    bool explain_strategy = false;
    bool json_output = false;
    double time_limit_sec = 0.0;
    std::optional<kavach::algorithms::milp::CutStrategy> cut_strategy_override;
    std::optional<kavach::algorithms::milp::BranchingStrategy> branching_override;
    std::optional<bool> heuristic_override;
    kavach::strategy::StrategyConfig config;

    for (std::size_t i = 2; i < args.size(); ++i) {
        if (args[i] == "--show-strategy") {
            show_strategy = true;
        } else if (args[i] == "--explain-strategy") {
            explain_strategy = true;
        } else if (args[i] == "--json") {
            json_output = true;
        } else if (args[i] == "--gpu") {
            config.override_backend = kavach::backend::BackendType::CUDA_GPU;
        } else if (args[i] == "--no-gpu") {
            config.force_no_gpu = true;
        } else if (args[i] == "--cuts" && i + 1 < args.size()) {
            std::string_view c_arg = args[++i];
            if (c_arg == "off" || c_arg == "none" || c_arg == "disabled") {
                cut_strategy_override = kavach::algorithms::milp::CutStrategy::OFF;
            } else if (c_arg == "root" || c_arg == "root_only") {
                cut_strategy_override = kavach::algorithms::milp::CutStrategy::ROOT_ONLY;
            } else if (c_arg == "periodic") {
                cut_strategy_override = kavach::algorithms::milp::CutStrategy::PERIODIC;
            }
        } else if (args[i] == "--branching" && i + 1 < args.size()) {
            std::string_view b_arg = args[++i];
            if (b_arg == "most_fractional" || b_arg == "fractional") {
                branching_override = kavach::algorithms::milp::BranchingStrategy::MOST_FRACTIONAL;
            } else if (b_arg == "pseudocost" || b_arg == "pseudo") {
                branching_override = kavach::algorithms::milp::BranchingStrategy::PSEUDOCOST;
            } else if (b_arg == "strong") {
                branching_override = kavach::algorithms::milp::BranchingStrategy::STRONG;
            } else if (b_arg == "reliability" || b_arg == "rel") {
                branching_override = kavach::algorithms::milp::BranchingStrategy::RELIABILITY;
            } else if (b_arg == "auto") {
                branching_override = kavach::algorithms::milp::BranchingStrategy::AUTO;
            }
        } else if (args[i] == "--heuristic" && i + 1 < args.size()) {
            std::string_view h_arg = args[++i];
            if (h_arg == "on" || h_arg == "true" || h_arg == "1" || h_arg == "yes") {
                heuristic_override = true;
            } else if (h_arg == "off" || h_arg == "false" || h_arg == "0" || h_arg == "no") {
                heuristic_override = false;
            }
        } else if (args[i] == "--strategy" && i + 1 < args.size()) {
            std::string_view strat_arg = args[++i];
            if (strat_arg == "auto") {
                config.mode = kavach::strategy::StrategyMode::AUTO;
            } else if (strat_arg == "conservative") {
                config.mode = kavach::strategy::StrategyMode::AUTO;
                config.elevated_coeff_range_threshold = 1e3;
                config.high_risk_coeff_range_threshold = 1e6;
                config.normal_refactorization_interval = 20;
                config.elevated_refactorization_interval = 10;
                config.high_risk_refactorization_interval = 5;
            } else if (strat_arg == "fast") {
                config.mode = kavach::strategy::StrategyMode::AUTO;
                config.normal_refactorization_interval = 100;
            } else {
                std::cerr << "Warning: Unknown strategy mode '" << strat_arg << "', using auto\n";
            }

        } else if (args[i] == "--presolve" && i + 1 < args.size()) {
            std::string_view p_arg = args[++i];
            if (p_arg == "auto") {
                config.override_presolve = std::nullopt;
            } else if (p_arg == "enabled" || p_arg == "full" || p_arg == "on") {
                config.override_presolve = true;
            } else if (p_arg == "disabled" || p_arg == "none" || p_arg == "off") {
                config.override_presolve = false;
            }
        } else if (args[i] == "--scaling" && i + 1 < args.size()) {
            std::string_view s_arg = args[++i];
            if (s_arg == "auto") {
                config.override_scaling = std::nullopt;
            } else if (s_arg == "enabled" || s_arg == "on") {
                config.override_scaling = true;
            } else if (s_arg == "disabled" || s_arg == "off") {
                config.override_scaling = false;
            }
        } else if (args[i] == "--backend" && i + 1 < args.size()) {
            std::string_view b_arg = args[++i];
            if (b_arg == "auto") {
                config.override_backend = std::nullopt;
            } else if (b_arg == "cpu") {
                config.override_backend = kavach::backend::BackendType::CPU_SEQUENTIAL;
            } else if (b_arg == "multicore") {
                config.override_backend = kavach::backend::BackendType::CPU_MULTICORE;
            } else if (b_arg == "gpu" || b_arg == "cuda") {
                config.override_backend = kavach::backend::BackendType::CUDA_GPU;
            }
        } else if (args[i] == "--time-limit" && i + 1 < args.size()) {
            time_limit_sec = std::stod(std::string(args[++i]));
        }
    }

    try {
        if (!json_output && file_path.find("mrpl") != std::string::npos) {
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
        if (!json_output) {
            std::cout << "Loaded model '" << model->name() << "': " << model->num_variables()
                      << " vars, " << model->num_constraints() << " constrs, "
                      << model->num_nonzeros() << " nonzeros\n";
        }

        kavach::api::KavachSolver solver(config);
        kavach::api::SolverOptions opts;
        opts.strategy_config = config;
        if (time_limit_sec > 0.0) {
            opts.time_limit =
                std::chrono::milliseconds(static_cast<int64_t>(time_limit_sec * 1000.0));
        }
        if (cut_strategy_override.has_value()) {
            opts.cut_options.strategy = *cut_strategy_override;
        }
        if (branching_override.has_value()) {
            opts.branching_strategy = *branching_override;
        }
        if (heuristic_override.has_value()) {
            opts.enable_primal_heuristic = *heuristic_override;
        }

        auto decision = solver.plan(*model, opts);

        if (show_strategy && !json_output) {
            std::cout << decision.to_string() << "\n";
        }
        if (explain_strategy && !json_output) {
            std::cout << decision.explain() << "\n";
        }

        auto result = solver.solve(*model, decision, opts);

        if (json_output) {
            kavach::benchmark::JsonValue::Object root;
            root["strategy"] = decision.to_json();
            kavach::benchmark::JsonValue::Object res;
            res["status"] = kavach::core::to_string(result.status);
            res["objective"] = result.objective_value;
            res["iterations"] = static_cast<double>(result.iterations);
            res["nodes"] = static_cast<double>(result.nodes);
            res["solve_time_seconds"] = result.solve_time_seconds;
            res["backend_used"] = kavach::backend::to_string(result.backend_used);
            res["verified"] = result.verification_report.passed;
            res["optimality_gap"] = result.optimality_gap;
            if (result.cut_telemetry.has_value()) {
                kavach::benchmark::JsonValue::Object cuts;
                cuts["cuts_generated"] = static_cast<double>(result.cut_telemetry->cuts_generated);
                cuts["cuts_applied"] = static_cast<double>(result.cut_telemetry->cuts_accepted);
                cuts["gomory_cuts"] = static_cast<double>(result.cut_telemetry->gomory_cuts);
                cuts["mir_cuts"] = static_cast<double>(result.cut_telemetry->mir_cuts);
                cuts["root_rounds"] = static_cast<double>(result.cut_telemetry->cut_rounds);
                res["cuts"] = std::move(cuts);
            }
            if (result.search_telemetry.has_value()) {
                kavach::benchmark::JsonValue::Object search;
                search["branching_strategy"] = result.search_telemetry->branching_strategy_used;
                search["pseudocost_candidates"] =
                    static_cast<double>(result.search_telemetry->pseudocost_candidates);
                search["pseudocost_updates"] =
                    static_cast<double>(result.search_telemetry->pseudocost_updates);
                search["strong_branching_calls"] =
                    static_cast<double>(result.search_telemetry->strong_branching_calls);
                search["strong_branching_candidates"] =
                    static_cast<double>(result.search_telemetry->strong_branching_candidates);
                search["strong_branching_infeasible_children"] = static_cast<double>(
                    result.search_telemetry->strong_branching_infeasible_children);
                search["reliability_fallbacks"] =
                    static_cast<double>(result.search_telemetry->reliability_fallbacks);
                search["heuristic_calls"] =
                    static_cast<double>(result.search_telemetry->heuristic_calls);
                search["heuristic_feasible_candidates"] =
                    static_cast<double>(result.search_telemetry->heuristic_feasible_candidates);
                search["heuristic_incumbent_updates"] =
                    static_cast<double>(result.search_telemetry->heuristic_incumbent_updates);
                search["incumbent_source"] = result.search_telemetry->incumbent_source;
                search["search_explanation"] = result.search_telemetry->search_explanation;
                res["search"] = std::move(search);
            }
            root["result"] = std::move(res);

            kavach::benchmark::JsonValue::Object ver;
            ver["passed"] = result.verification_report.passed;
            ver["certified_status"] =
                kavach::core::to_string(result.verification_report.certified_status);
            ver["recomputed_primal_residual"] =
                result.verification_report.recomputed_primal_residual;
            ver["recomputed_dual_residual"] = result.verification_report.recomputed_dual_residual;
            ver["recomputed_complementary_slackness"] =
                result.verification_report.recomputed_complementary_slackness;
            ver["recomputed_integrality_violation"] =
                result.verification_report.recomputed_integrality_violation;
            ver["recomputed_kkt_residual"] = result.verification_report.recomputed_kkt_residual;
            ver["recomputed_objective"] = result.verification_report.recomputed_objective;
            ver["summary"] = result.verification_report.summary;
            root["verification"] = std::move(ver);

            kavach::benchmark::JsonValue::Object mod;
            mod["name"] = model->name();
            mod["num_variables"] = static_cast<double>(model->num_variables());
            mod["num_constraints"] = static_cast<double>(model->num_constraints());
            mod["num_nonzeros"] = static_cast<double>(model->num_nonzeros());
            double total_entries = static_cast<double>(model->num_variables()) *
                                   static_cast<double>(model->num_constraints());
            mod["density"] = total_entries > 0.0
                                 ? (static_cast<double>(model->num_nonzeros()) / total_entries)
                                 : 0.0;
            mod["is_milp"] = model->is_milp();
            mod["is_qp"] = model->is_qp();
            mod["problem_class"] =
                model->is_qp() ? "CONVEX_QP" : (model->is_milp() ? "MILP" : "LP");
            root["model"] = std::move(mod);

            std::cout << kavach::benchmark::JsonValue(std::move(root)).dump() << "\n";
            return 0;
        }

        if (model->is_qp()) {
            std::cout << "Solving Convex QP...\n"
                      << "Status: " << kavach::core::to_string(result.status) << "\n"
                      << "Objective: " << result.objective_value << "\n"
                      << "Iterations: " << result.iterations << "\n"
                      << "Verification: "
                      << (result.verification_report.passed ? "VERIFIED_OPTIMAL (Certified)"
                                                            : "UNVERIFIED")
                      << "\n";
        } else if (model->is_milp()) {
            std::cout << "Solving MILP with Branch-and-Bound...\n"
                      << "Status: " << kavach::core::to_string(result.status) << "\n"
                      << "Best Objective: " << result.objective_value << "\n"
                      << "Optimality Gap: " << result.optimality_gap << "\n"
                      << "Nodes Explored: " << result.nodes << "\n";
            if (result.cut_telemetry.has_value() && result.cut_telemetry->cuts_accepted > 0) {
                std::cout << "Cuts Applied: " << result.cut_telemetry->cuts_accepted
                          << " (Gomory: " << result.cut_telemetry->gomory_cuts
                          << ", MIR: " << result.cut_telemetry->mir_cuts
                          << ", Rounds: " << result.cut_telemetry->cut_rounds << ")\n";
            }
            if (result.search_telemetry.has_value()) {
                std::cout << "Search Engine: " << result.search_telemetry->branching_strategy_used
                          << " (Pseudocosts: " << result.search_telemetry->pseudocost_updates
                          << ", Strong branch: " << result.search_telemetry->strong_branching_calls
                          << ", Heuristics: "
                          << result.search_telemetry->heuristic_incumbent_updates
                          << ", Incumbent: " << result.search_telemetry->incumbent_source << ")\n";
            }
            std::cout << "Verification: "
                      << (result.verification_report.passed
                              ? "VERIFIED_OPTIMAL (Certified by MILPVerifier)"
                              : "UNVERIFIED")
                      << "\n";

            if (result.verification_report.passed && file_path.find("mrpl") != std::string::npos) {
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
        } else {
            std::cout << "Solving LP with Revised Simplex...\n"
                      << "Status: " << kavach::core::to_string(result.status) << "\n"
                      << "Objective: " << result.objective_value << "\n"
                      << "Iterations: " << result.iterations << "\n"
                      << "Verification: "
                      << (result.verification_report.passed
                              ? "VERIFIED_OPTIMAL (Certified by LPVerifier)"
                              : "UNVERIFIED")
                      << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Solve error: " << e.what() << '\n';
        return 1;
    }
}

int handle_presolve(const std::vector<std::string_view>& args) {
    if (args.size() < 2) {
        std::cerr << "Usage: kavach-opt presolve <model.mps|.lp> [--report]\n";
        return 1;
    }
    std::string file_path = std::string(args[1]);
    bool report = false;
    for (std::size_t i = 2; i < args.size(); ++i) {
        if (args[i] == "--report") {
            report = true;
        }
    }
    try {
        auto model = kavach::io::parse_model(file_path);
        std::cout << "Loaded model '" << model->name() << "': " << model->num_variables()
                  << " vars, " << model->num_constraints() << " constrs, " << model->num_nonzeros()
                  << " nonzeros\n";

        kavach::presolve::PresolveOptions opts;
        kavach::presolve::Presolver presolver;
        auto presolve_result = presolver.apply(*model, opts);

        std::cout << "\n=== Kavach-Opt Presolve Summary ===\n"
                  << presolve_result.stats.to_string() << "\n";
        if (report) {
            std::cout << "Presolve Status: " << kavach::core::to_string(presolve_result.status)
                      << "\n"
                      << "Reduced Model:   " << presolve_result.reduced_model->num_variables()
                      << " vars, " << presolve_result.reduced_model->num_constraints()
                      << " constrs, " << presolve_result.reduced_model->num_nonzeros()
                      << " nonzeros\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Presolve error: " << e.what() << '\n';
        return 1;
    }
}

int handle_verify(const std::vector<std::string_view>& args) {
    if (args.size() < 2) {
        std::cerr << "Usage: kavach-opt verify <model.mps|.lp>\n";
        return 1;
    }
    std::string file_path = std::string(args[1]);
    try {
        auto model = kavach::io::parse_model(file_path);
        std::cout << "Loaded model '" << model->name() << "' for independent certification...\n";

        kavach::api::KavachSolver solver;
        auto result = solver.solve(*model);

        std::cout << "\n=== Kavach-Opt Independent Verification Certificate ===\n"
                  << "Model:                " << model->name() << "\n"
                  << "Solver Status:        " << kavach::core::to_string(result.status) << "\n"
                  << "Objective Value:      " << std::setprecision(8) << result.objective_value
                  << "\n"
                  << "Verification Outcome: "
                  << (result.verification_report.passed ? "CERTIFIED (PASS)" : "REJECTED (FAIL)")
                  << "\n"
                  << "Primal Residual:      "
                  << result.verification_report.recomputed_primal_residual << "\n"
                  << "Dual Residual:        " << result.verification_report.recomputed_dual_residual
                  << "\n"
                  << "Summary:              " << result.verification_report.summary << "\n";

        if (!result.verification_report.passed) {
            std::cerr << "Verification check failed for model '" << model->name() << "'\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Verification error: " << e.what() << '\n';
        return 1;
    }
}

int handle_inspect(const std::vector<std::string_view>& args) {
    if (args.size() < 2) {
        std::cerr << "Usage: kavach-opt inspect <model.mps|.lp> [--json]\n";
        return 1;
    }
    std::string file_path = std::string(args[1]);
    bool json_output = false;
    for (std::size_t i = 2; i < args.size(); ++i) {
        if (args[i] == "--json") {
            json_output = true;
        }
    }
    try {
        auto model = kavach::io::parse_model(file_path);
        kavach::strategy::StrategyEngine engine;
        auto chars = engine.characterize(*model);

        if (json_output) {
            kavach::benchmark::JsonValue::Object root;
            root["name"] = model->name();
            root["file_path"] = file_path;
            root["problem_class"] = kavach::strategy::to_string(chars.problem_class);
            root["is_milp"] = model->is_milp();
            root["is_qp"] = model->is_qp();
            root["num_variables"] = static_cast<double>(chars.num_variables);
            root["num_constraints"] = static_cast<double>(chars.num_constraints);
            root["num_nonzeros"] = static_cast<double>(chars.num_nonzeros);
            root["density"] = chars.density;
            root["average_row_sparsity"] = chars.average_row_sparsity;
            root["integer_variables"] = static_cast<double>(chars.integer_variable_count);
            root["binary_variables"] = static_cast<double>(chars.binary_variable_count);
            root["continuous_variables"] =
                static_cast<double>(chars.num_variables - chars.integer_variable_count);
            root["min_abs_coefficient"] = chars.min_abs_nonzero_coeff;
            root["max_abs_coefficient"] = chars.max_abs_nonzero_coeff;
            root["coefficient_range_ratio"] = chars.coefficient_range_ratio;
            root["conditioning_category"] =
                kavach::numerics::to_string(chars.conditioning_category);
            root["recommend_scaling"] = chars.recommend_scaling;
            root["characterization_time_ms"] = chars.characterization_time_ms;
            root["characteristics"] = chars.to_json();

            std::cout << kavach::benchmark::JsonValue(std::move(root)).dump() << "\n";
            return 0;
        }

        std::cout << "Model Inspection: " << model->name() << " (" << file_path << ")\n"
                  << chars.to_string() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Inspect error: " << e.what() << '\n';
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
        if (command == "inspect") {
            return handle_inspect(args);
        }
        if (command == "presolve") {
            return handle_presolve(args);
        }
        if (command == "verify") {
            return handle_verify(args);
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
