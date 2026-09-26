/**
 * @file test_phase12_benchmark.cpp
 * @brief Comprehensive test suite for Kavach-Opt Benchmark & Reproducibility Subsystem.
 *
 * Implements Section 29 Test Matrix:
 *   Test A: Benchmark case parsing
 *   Test B: Manifest parsing
 *   Test C: Result serialization roundtrip
 *   Test D: CSV output generation
 *   Test E: JSON output generation
 *   Test F: Status taxonomy normalization
 *   Test G: Objective normalization
 *   Test H: Objective comparison logic
 *   Test I: Timing aggregation & percentiles
 *   Test J: Failure preservation as first-class results
 *   Test K: Aggregate statistics derivation
 *   Test L: Deterministic rerun reproducibility
 *   Test M: Missing instance handling
 *   Test N: Invalid manifest rejection
 *   Test O: Reference solver unavailable handling
 *   Test P: Reference solver timeout handling
 *   Test Q: CLI benchmark command invocation
 *   Test R: GPU telemetry ingestion & speedup preservation
 *   Test S: Null/unavailable field preservation
 *   Test T: Markdown report generation (all 10 sections)
 */

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "../test_framework.hpp"
#include "benchmark/benchmark_case.hpp"
#include "benchmark/benchmark_io.hpp"
#include "benchmark/benchmark_manifest.hpp"
#include "benchmark/benchmark_report.hpp"
#include "benchmark/benchmark_result.hpp"
#include "benchmark/benchmark_runner.hpp"
#include "benchmark/json.hpp"
#include "benchmark/reference/reference_adapter.hpp"
#include "core/status.hpp"

using namespace kavach;
using namespace kavach::benchmark;

// ============================================================================
// Test A: Benchmark Case Parsing
// ============================================================================

KAVACH_TEST(test_a_benchmark_case_parsing) {
    std::string json_str = R"({
        "case_id": "test-case-1",
        "instance_name": "toy_lp",
        "instance_path": "tests/data/toy.mps",
        "problem_type": "LP",
        "source": "Synthetic",
        "tags": ["small", "sparse", "convex QP"],
        "timeout_seconds": 45.0,
        "reference_objective": -123.456,
        "reference_status": "OPTIMAL",
        "seed": 999,
        "enable_presolve": false,
        "spmv_batch_size": 128
    })";

    std::string manifest_wrapper = R"({
        "suite": "test_suite",
        "description": "A test suite",
        "cases": [)" + json_str + R"(]
    })";

    auto suite = BenchmarkManifestLoader::load_from_string(manifest_wrapper);
    KAVACH_ASSERT_EQUAL(suite.cases.size(), 1);

    const auto& c = suite.cases[0];
    KAVACH_ASSERT_EQUAL(c.case_id, "test-case-1");
    KAVACH_ASSERT_EQUAL(c.instance_name, "toy_lp");
    KAVACH_ASSERT_EQUAL(c.instance_path, "tests/data/toy.mps");
    KAVACH_ASSERT(c.problem_type == BenchmarkProblemType::LP);
    KAVACH_ASSERT_EQUAL(c.source, "Synthetic");
    KAVACH_ASSERT_EQUAL(c.tags.size(), 3);
    KAVACH_ASSERT(c.has_tag("small"));
    KAVACH_ASSERT(c.has_tag("convex QP"));
    KAVACH_ASSERT(!c.has_tag("large"));
    KAVACH_ASSERT(c.timeout_seconds.has_value());
    KAVACH_ASSERT_NEAR(*c.timeout_seconds, 45.0, 1e-6);
    KAVACH_ASSERT(c.reference_objective.has_value());
    KAVACH_ASSERT_NEAR(*c.reference_objective, -123.456, 1e-6);
    KAVACH_ASSERT_EQUAL(*c.reference_status, "OPTIMAL");
    KAVACH_ASSERT_EQUAL(c.seed, 999);
    KAVACH_ASSERT(!c.enable_presolve);
    KAVACH_ASSERT_EQUAL(c.spmv_batch_size, 128);
}

// ============================================================================
// Test B: Manifest Parsing
// ============================================================================

KAVACH_TEST(test_b_manifest_parsing) {
    std::string manifest_json = R"({
        "suite": "multi_suite",
        "description": "Multi-case benchmark suite",
        "version": "2.1.0",
        "cases": [
            {
                "case_id": "c1",
                "instance_path": "p1.mps",
                "problem_type": "LP"
            },
            {
                "case_id": "c2",
                "instance_path": "p2.lp",
                "problem_type": "MILP",
                "reference_best_bound": 10.5
            },
            {
                "case_id": "c3",
                "instance_path": "p3.lp",
                "problem_type": "QP"
            }
        ]
    })";

    auto suite = BenchmarkManifestLoader::load_from_string(manifest_json);
    KAVACH_ASSERT_EQUAL(suite.name, "multi_suite");
    KAVACH_ASSERT_EQUAL(suite.description, "Multi-case benchmark suite");
    KAVACH_ASSERT_EQUAL(suite.version, "2.1.0");
    KAVACH_ASSERT_EQUAL(suite.size(), 3);

    KAVACH_ASSERT(suite.find_case("c1") != nullptr);
    KAVACH_ASSERT(suite.find_case("c2") != nullptr);
    KAVACH_ASSERT(suite.find_case("c3") != nullptr);
    KAVACH_ASSERT(suite.find_case("c4") == nullptr);

    KAVACH_ASSERT(suite.find_case("c2")->reference_best_bound.has_value());
    KAVACH_ASSERT_NEAR(*suite.find_case("c2")->reference_best_bound, 10.5, 1e-6);
}

// ============================================================================
// Test C: Result Serialization Roundtrip
// ============================================================================

KAVACH_TEST(test_c_result_serialization_roundtrip) {
    BenchmarkResult r;
    r.timestamp = "2026-09-25T12:00:00Z";
    r.suite = "test_suite";
    r.case_id = "test-c1";
    r.instance_name = "inst_c1";
    r.instance_path = "/tmp/inst_c1.mps";
    r.problem_type = "LP";
    r.tags = {"small", "sparse"};
    r.variables = 100;
    r.constraints = 50;
    r.nonzeros = 350;
    r.density = 0.07;
    r.solver_name = "Kavach-Opt";
    r.solver_version = "0.1.0";
    r.status = BenchmarkStatus::OPTIMAL;
    r.verified_status = "VERIFIED_OPTIMAL";
    r.verification_passed = true;
    r.objective_value = -987.654321;
    r.reference_objective = -987.654320;
    r.reference_status = "OPTIMAL";
    r.objective_error_abs = 0.000001;
    r.objective_error_rel = 1.01e-9;
    r.iterations = 42;
    r.total_wall_time_seconds = 0.0456;
    r.solve_time_seconds = 0.0350;
    r.primal_residual = 1.2e-12;
    r.dual_residual = 3.4e-12;

    JsonValue j = BenchmarkIO::to_json(r);
    BenchmarkResult r2 = BenchmarkIO::from_json(j);

    KAVACH_ASSERT_EQUAL(r2.case_id, r.case_id);
    KAVACH_ASSERT_EQUAL(r2.instance_name, r.instance_name);
    KAVACH_ASSERT_EQUAL(r2.problem_type, r.problem_type);
    KAVACH_ASSERT_EQUAL(r2.status, r.status);
    KAVACH_ASSERT_EQUAL(r2.verified_status, r.verified_status);
    KAVACH_ASSERT(r2.verification_passed);
    KAVACH_ASSERT(r2.variables.has_value() && *r2.variables == 100);
    KAVACH_ASSERT(r2.constraints.has_value() && *r2.constraints == 50);
    KAVACH_ASSERT(r2.objective_value.has_value());
    KAVACH_ASSERT_NEAR(*r2.objective_value, -987.654321, 1e-6);
    KAVACH_ASSERT(r2.iterations.has_value() && *r2.iterations == 42);
    KAVACH_ASSERT_NEAR(r2.total_wall_time_seconds, 0.0456, 1e-6);
}

// ============================================================================
// Test D: CSV Output Generation
// ============================================================================

KAVACH_TEST(test_d_csv_output) {
    BenchmarkResult r;
    r.case_id = "csv-test-1";
    r.instance_name = "test_inst";
    r.suite = "csv_suite";
    r.problem_type = "LP";
    r.status = BenchmarkStatus::OPTIMAL;
    r.verified_status = "VERIFIED_OPTIMAL";
    r.objective_value = 100.5;
    r.reference_objective = 100.5;
    r.objective_error_rel = 0.0;
    r.total_wall_time_seconds = 0.012;
    r.solve_time_seconds = 0.010;
    r.iterations = 25;
    r.primal_residual = 1e-14;

    std::filesystem::path csv_path = "/tmp/test_phase12_output.csv";
    BenchmarkIO::export_csv({r}, csv_path);

    std::ifstream ifs(csv_path);
    KAVACH_ASSERT(ifs.is_open());
    std::string header;
    std::getline(ifs, header);
    KAVACH_ASSERT(header.find("case_id,instance,suite") != std::string::npos);

    std::string row;
    std::getline(ifs, row);
    KAVACH_ASSERT(row.find("csv-test-1") != std::string::npos);
    KAVACH_ASSERT(row.find("OPTIMAL") != std::string::npos);
    KAVACH_ASSERT(row.find("100.5") != std::string::npos);
}

// ============================================================================
// Test E: JSON Output Generation
// ============================================================================

KAVACH_TEST(test_e_json_output) {
    BenchmarkResult r;
    r.case_id = "json-test-1";
    r.suite = "json_suite";
    r.status = BenchmarkStatus::OPTIMAL;
    r.objective_value = 50.0;

    ReproducibilityManifest m;
    m.git_commit = "abcdef123";
    m.suite_name = "json_suite";

    std::filesystem::path json_path = "/tmp/test_phase12_output.json";
    BenchmarkIO::export_json({r}, m, json_path);

    auto [loaded_m, loaded_results] = BenchmarkIO::import_json(json_path);
    KAVACH_ASSERT_EQUAL(loaded_m.git_commit, "abcdef123");
    KAVACH_ASSERT_EQUAL(loaded_m.suite_name, "json_suite");
    KAVACH_ASSERT_EQUAL(loaded_results.size(), 1);
    KAVACH_ASSERT_EQUAL(loaded_results[0].case_id, "json-test-1");
    KAVACH_ASSERT(loaded_results[0].objective_value.has_value());
    KAVACH_ASSERT_NEAR(*loaded_results[0].objective_value, 50.0, 1e-6);
}

// ============================================================================
// Test F: Status Normalization
// ============================================================================

KAVACH_TEST(test_f_status_normalization) {
    KAVACH_ASSERT_EQUAL(from_solver_status(core::SolverStatus::OPTIMAL), BenchmarkStatus::OPTIMAL);
    KAVACH_ASSERT_EQUAL(from_solver_status(core::SolverStatus::FEASIBLE),
                        BenchmarkStatus::FEASIBLE);
    KAVACH_ASSERT_EQUAL(from_solver_status(core::SolverStatus::INFEASIBLE),
                        BenchmarkStatus::INFEASIBLE);
    KAVACH_ASSERT_EQUAL(from_solver_status(core::SolverStatus::UNBOUNDED),
                        BenchmarkStatus::UNBOUNDED);
    KAVACH_ASSERT_EQUAL(from_solver_status(core::SolverStatus::LIMIT_REACHED),
                        BenchmarkStatus::LIMIT_REACHED);
    KAVACH_ASSERT_EQUAL(from_solver_status(core::SolverStatus::NUMERICAL_FAILURE),
                        BenchmarkStatus::NUMERICAL_FAILURE);

    KAVACH_ASSERT_EQUAL(to_string(BenchmarkStatus::OPTIMAL), "OPTIMAL");
    KAVACH_ASSERT_EQUAL(to_string(BenchmarkStatus::SOLVER_TIMEOUT), "SOLVER_TIMEOUT");
    KAVACH_ASSERT_EQUAL(to_string(BenchmarkStatus::PARSE_FAILURE), "PARSE_FAILURE");
    KAVACH_ASSERT_EQUAL(to_string(BenchmarkStatus::VERIFICATION_FAILURE), "VERIFICATION_FAILURE");
    KAVACH_ASSERT_EQUAL(to_string(BenchmarkStatus::UNSUPPORTED), "UNSUPPORTED");

    KAVACH_ASSERT_EQUAL(parse_benchmark_status("OPTIMAL"), BenchmarkStatus::OPTIMAL);
    KAVACH_ASSERT_EQUAL(parse_benchmark_status("SOLVER_TIMEOUT"), BenchmarkStatus::SOLVER_TIMEOUT);
}

// ============================================================================
// Test G: Objective Normalization
// ============================================================================

KAVACH_TEST(test_g_objective_normalization) {
    // Normalization handles sign / scale and absolute differences
    double solver_min_obj = -464.75314286;
    double ref_obj = -464.75314286;
    auto comp = compare_objectives(solver_min_obj, ref_obj);
    KAVACH_ASSERT(comp.matches);
    KAVACH_ASSERT_NEAR(comp.abs_error, 0.0, 1e-9);
    KAVACH_ASSERT_NEAR(comp.rel_error, 0.0, 1e-9);
}

// ============================================================================
// Test H: Objective Comparison
// ============================================================================

KAVACH_TEST(test_h_objective_comparison) {
    // Case 1: exact match
    auto cmp1 = compare_objectives(10.0, 10.0, 1e-4, 1e-4);
    KAVACH_ASSERT(cmp1.matches);

    // Case 2: match within relative tolerance
    auto cmp2 = compare_objectives(1000000.0, 1000001.0, 1e-5, 1e-6);
    KAVACH_ASSERT(cmp2.matches);

    // Case 3: difference exceeds tolerance
    auto cmp3 = compare_objectives(10.0, 20.0, 1e-4, 1e-4);
    KAVACH_ASSERT(!cmp3.matches);
    KAVACH_ASSERT_NEAR(cmp3.abs_error, 10.0, 1e-9);
}

// ============================================================================
// Test I: Timing Aggregation
// ============================================================================

KAVACH_TEST(test_i_timing_aggregation) {
    std::vector<BenchmarkResult> results;
    for (double t : {0.05, 0.01, 0.10, 0.02, 0.08}) {
        BenchmarkResult r;
        r.total_wall_time_seconds = t;
        r.status = BenchmarkStatus::OPTIMAL;
        r.verification_passed = true;
        results.push_back(r);
    }

    auto agg = compute_aggregates(results);
    KAVACH_ASSERT_EQUAL(agg.total_attempted, 5);
    KAVACH_ASSERT_NEAR(agg.min_runtime_seconds, 0.01, 1e-6);
    KAVACH_ASSERT_NEAR(agg.max_runtime_seconds, 0.10, 1e-6);
    KAVACH_ASSERT_NEAR(agg.median_runtime_seconds, 0.05, 1e-6);
}

// ============================================================================
// Test J: Failure Preservation
// ============================================================================

KAVACH_TEST(test_j_failure_preservation) {
    BenchmarkResult r_timeout;
    r_timeout.case_id = "timeout_case";
    r_timeout.status = BenchmarkStatus::SOLVER_TIMEOUT;
    r_timeout.error_message = "Time limit exceeded";

    BenchmarkResult r_parse;
    r_parse.case_id = "bad_file";
    r_parse.status = BenchmarkStatus::PARSE_FAILURE;
    r_parse.error_message = "File not found";

    BenchmarkResult r_verif;
    r_verif.case_id = "corrupted_solution";
    r_verif.status = BenchmarkStatus::VERIFICATION_FAILURE;
    r_verif.error_message = "Verifier rejected primal solution";

    auto agg = compute_aggregates({r_timeout, r_parse, r_verif});
    KAVACH_ASSERT_EQUAL(agg.total_attempted, 3);
    KAVACH_ASSERT_EQUAL(agg.verified_optimal, 0);
    KAVACH_ASSERT_EQUAL(agg.solver_timeout, 1);
    KAVACH_ASSERT_EQUAL(agg.parse_failure, 1);
    KAVACH_ASSERT_EQUAL(agg.verification_failure, 1);
}

// ============================================================================
// Test K: Aggregate Generation
// ============================================================================

KAVACH_TEST(test_k_aggregate_generation) {
    std::vector<BenchmarkResult> results;
    BenchmarkResult r1;
    r1.status = BenchmarkStatus::OPTIMAL;
    r1.verification_passed = true;
    r1.iterations = 10;
    r1.objective_error_rel = 1e-8;
    results.push_back(r1);

    BenchmarkResult r2;
    r2.status = BenchmarkStatus::OPTIMAL;
    r2.verification_passed = true;
    r2.iterations = 30;
    r2.objective_error_rel = 2e-8;
    results.push_back(r2);

    BenchmarkResult r3;
    r3.status = BenchmarkStatus::INFEASIBLE;
    results.push_back(r3);

    auto agg = compute_aggregates(results);
    KAVACH_ASSERT_EQUAL(agg.total_attempted, 3);
    KAVACH_ASSERT_EQUAL(agg.verified_optimal, 2);
    KAVACH_ASSERT_EQUAL(agg.infeasible, 1);
    KAVACH_ASSERT_NEAR(agg.mean_iterations, 20.0, 1e-6);
}

// ============================================================================
// Test L: Deterministic Rerun Test (Section 25)
// ============================================================================

KAVACH_TEST(test_l_deterministic_rerun) {
    BenchmarkCase c;
    c.case_id = "rerun_case";
    c.instance_name = "afiro";
    c.instance_path = "benchmarks/netlib/afiro.mps";
    c.problem_type = BenchmarkProblemType::LP;
    c.seed = 12345;
    c.enable_presolve = false;

    if (!std::filesystem::exists(c.instance_path)) {
        c.instance_path = "../benchmarks/netlib/afiro.mps";
    }
    if (!std::filesystem::exists(c.instance_path)) {
        return;  // Skip if netlib not available
    }

    BenchmarkRunner runner;
    auto res1 = runner.run_case(c);
    auto res2 = runner.run_case(c);

    KAVACH_ASSERT_EQUAL(res1.status, BenchmarkStatus::OPTIMAL);
    KAVACH_ASSERT_EQUAL(res2.status, BenchmarkStatus::OPTIMAL);
    KAVACH_ASSERT_EQUAL(res1.status, res2.status);
    KAVACH_ASSERT(res1.objective_value.has_value());
    KAVACH_ASSERT(res2.objective_value.has_value());
    KAVACH_ASSERT_NEAR(*res1.objective_value, *res2.objective_value, 1e-10);
    KAVACH_ASSERT_EQUAL(res1.iterations.has_value(), res2.iterations.has_value());
    if (res1.iterations.has_value() && res2.iterations.has_value()) {
        KAVACH_ASSERT_EQUAL(*res1.iterations, *res2.iterations);
    }
    KAVACH_ASSERT_EQUAL(res1.random_seed, res2.random_seed);
}

// ============================================================================
// Test M: Missing Instance Handling
// ============================================================================

KAVACH_TEST(test_m_missing_instance_handling) {
    BenchmarkCase c;
    c.case_id = "missing_case";
    c.instance_path = "non_existent_directory/phantom_model.mps";
    c.problem_type = BenchmarkProblemType::LP;

    BenchmarkRunner runner;
    auto res = runner.run_case(c);

    KAVACH_ASSERT_EQUAL(res.status, BenchmarkStatus::PARSE_FAILURE);
    KAVACH_ASSERT(!res.error_message.empty());
    KAVACH_ASSERT(res.error_message.find("File not found") != std::string::npos);
}

// ============================================================================
// Test N: Invalid Manifest Handling
// ============================================================================

KAVACH_TEST(test_n_invalid_manifest) {
    // Malformed JSON
    bool threw1 = false;
    try {
        auto ignored = BenchmarkManifestLoader::load_from_string("{ bad json");
        (void)ignored;
    } catch (const ManifestException&) {
        threw1 = true;
    }
    KAVACH_ASSERT(threw1);

    // Missing required fields
    bool threw2 = false;
    try {
        auto ignored = BenchmarkManifestLoader::load_from_string(R"({"suite": "test"})");
        (void)ignored;
    } catch (const ManifestException&) {
        threw2 = true;
    }
    KAVACH_ASSERT(threw2);
}

// ============================================================================
// Test O: Reference Solver Unavailable
// ============================================================================

KAVACH_TEST(test_o_reference_solver_unavailable) {
    ExternalReferenceAdapter adapter("/path/to/nonexistent/highs_binary");
    KAVACH_ASSERT(!adapter.is_available());

    auto run_res = adapter.run("any_model.mps");
    KAVACH_ASSERT_EQUAL(run_res.status, ReferenceSolverStatus::UNAVAILABLE);
    KAVACH_ASSERT(!run_res.error_message.empty());
}

// ============================================================================
// Test P: Reference Solver Parsing & Timeout Fixtures
// ============================================================================

KAVACH_TEST(test_p_reference_solver_parsing) {
    std::string sample_highs_output =
        "Running HiGHS 1.5.0 (git hash: e9b4)\n"
        "Model   status      : Optimal\n"
        "Objective value     : -4.6475314286e+02\n"
        "HiGHS run time      :          0.04\n"
        "Simplex   iterations: 38\n";

    auto res = ExternalReferenceAdapter::parse_output(sample_highs_output, "", 0, false);
    KAVACH_ASSERT_EQUAL(res.status, ReferenceSolverStatus::OPTIMAL);
    KAVACH_ASSERT(res.objective.has_value());
    KAVACH_ASSERT_NEAR(*res.objective, -464.75314286, 1e-6);
    KAVACH_ASSERT(res.runtime_seconds.has_value());
    KAVACH_ASSERT_NEAR(*res.runtime_seconds, 0.04, 1e-6);
    KAVACH_ASSERT(res.iterations.has_value() && *res.iterations == 38);

    // Test timeout fixture
    auto res_timeout = ExternalReferenceAdapter::parse_output("", "", -1, true);
    KAVACH_ASSERT_EQUAL(res_timeout.status, ReferenceSolverStatus::TIMEOUT);
}

// ============================================================================
// Test Q: CLI Benchmark Execution
// ============================================================================

KAVACH_TEST(test_q_cli_benchmark_execution) {
    std::string repo_root = ".";
    for (const auto& p : {".", "..", "../.."}) {
        if (std::filesystem::exists(std::filesystem::path(p) / "benchmarks" / "manifests" /
                                    "qp.json")) {
            repo_root = std::filesystem::canonical(p).string();
            break;
        }
    }

    std::string bin_path = "./kavach-opt";
    for (const auto& b : {"./kavach-opt", "./build-release/kavach-opt", "../kavach-opt",
                          "../build-release/kavach-opt"}) {
        if (std::filesystem::exists(b)) {
            bin_path = std::filesystem::canonical(b).string();
            break;
        }
    }

    std::string cmd = "cd \"" + repo_root + "\" && \"" + bin_path +
                      "\" benchmark --suite qp --output /tmp/cli_test_qp > /dev/null 2>&1";
    int rc = std::system(cmd.c_str());
    KAVACH_ASSERT_EQUAL(rc, 0);

    KAVACH_ASSERT(std::filesystem::exists("/tmp/cli_test_qp/benchmark_results.json"));
    KAVACH_ASSERT(std::filesystem::exists("/tmp/cli_test_qp/benchmark_results.csv"));
    KAVACH_ASSERT(std::filesystem::exists("/tmp/cli_test_qp/benchmark_report.md"));
}

// ============================================================================
// Test R: GPU Telemetry Ingestion
// ============================================================================

KAVACH_TEST(test_r_gpu_telemetry_ingestion) {
    BenchmarkResult r;
    r.case_id = "gpu-spmv-case";
    backend::cuda::CudaTelemetry tel;
    tel.gpu_available = true;
    tel.device_name = "NVIDIA H100 PCIe";
    tel.h2d_time_seconds = 0.001;
    tel.kernel_time_seconds = 0.0005;
    tel.d2h_time_seconds = 0.0005;
    tel.total_gpu_time_seconds = 0.002;
    tel.cpu_reference_time_seconds = 0.010;
    tel.transfer_inclusive_speedup = 5.0;
    tel.kernel_only_speedup = 20.0;
    r.gpu_telemetry = tel;

    JsonValue j = BenchmarkIO::to_json(r);
    KAVACH_ASSERT(j.contains("gpu_telemetry"));
    KAVACH_ASSERT(j["gpu_telemetry"].is_object());
    KAVACH_ASSERT_EQUAL(j["gpu_telemetry"]["device_name"].as_string(), "NVIDIA H100 PCIe");
    KAVACH_ASSERT_NEAR(j["gpu_telemetry"]["transfer_inclusive_speedup"].as_double(), 5.0, 1e-6);

    BenchmarkResult r2 = BenchmarkIO::from_json(j);
    KAVACH_ASSERT(r2.gpu_telemetry.has_value());
    KAVACH_ASSERT_EQUAL(r2.gpu_telemetry->device_name, "NVIDIA H100 PCIe");
    KAVACH_ASSERT_NEAR(r2.gpu_telemetry->transfer_inclusive_speedup, 5.0, 1e-6);
}

// ============================================================================
// Test S: Null / Unavailable Field Preservation
// ============================================================================

KAVACH_TEST(test_s_null_field_preservation) {
    BenchmarkResult r;
    r.case_id = "null_fields_case";
    // Deliberately leave variables, constraints, dual_residual, etc. unpopulated

    JsonValue j = BenchmarkIO::to_json(r);
    KAVACH_ASSERT(j["variables"].is_null());
    KAVACH_ASSERT(j["constraints"].is_null());
    KAVACH_ASSERT(j["dual_residual"].is_null());
    KAVACH_ASSERT(j["kkt_residual"].is_null());
    KAVACH_ASSERT(j["memory_bytes"].is_null());
}

// ============================================================================
// Test T: Report Generation (All 10 Sections)
// ============================================================================

KAVACH_TEST(test_t_report_generation_all_sections) {
    BenchmarkResult r;
    r.case_id = "report-c1";
    r.instance_name = "c1";
    r.problem_type = "LP";
    r.status = BenchmarkStatus::OPTIMAL;
    r.verification_passed = true;
    r.verified_status = "VERIFIED_OPTIMAL";
    r.objective_value = 12.34;
    r.total_wall_time_seconds = 0.05;

    ReproducibilityManifest m = ReproducibilityManifest::capture_current_environment("report_test");
    std::string report = BenchmarkReporter::generate_markdown_report({r}, m);

    // Verify all 10 required sections
    KAVACH_ASSERT(report.find("## 1. Executive Methodology") != std::string::npos);
    KAVACH_ASSERT(report.find("## 2. Hardware and Software Environment") != std::string::npos);
    KAVACH_ASSERT(report.find("## 3. Suite Coverage") != std::string::npos);
    KAVACH_ASSERT(report.find("## 4. Per-Instance Detailed Results") != std::string::npos);
    KAVACH_ASSERT(report.find("## 5. Aggregate Summary") != std::string::npos);
    KAVACH_ASSERT(report.find("## 6. Failures, Timeouts, and Unsupported Cases") !=
                  std::string::npos);
    KAVACH_ASSERT(report.find("## 7. Independent Verification Summary") != std::string::npos);
    KAVACH_ASSERT(report.find("## 8. External Reference Solver Comparison") != std::string::npos);
    KAVACH_ASSERT(report.find("## 9. GPU Acceleration & Telemetry Analysis") != std::string::npos);
    KAVACH_ASSERT(report.find("## 10. Reproducibility Instructions") != std::string::npos);
}
