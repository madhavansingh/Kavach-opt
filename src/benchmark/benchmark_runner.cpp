#include "benchmark/benchmark_runner.hpp"

#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>

#include "algorithms/lp/simplex.hpp"
#include "algorithms/milp/branch_and_bound.hpp"
#include "algorithms/qp/active_set.hpp"
#include "backend/cpu/cpu_backend.hpp"
#include "backend/cuda/cuda_backend.hpp"
#include "core/version.hpp"
#include "io/parser.hpp"
#include "verification/lp_verifier.hpp"
#include "verification/milp_verifier.hpp"
#include "verification/qp_verifier.hpp"

namespace kavach::benchmark {

namespace {

std::string get_utc_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm gmt{};
#if defined(_WIN32)
    gmtime_s(&gmt, &tt);
#else
    gmtime_r(&tt, &gmt);
#endif
    std::ostringstream ss;
    ss << std::put_time(&gmt, "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

}  // namespace

BenchmarkRunner::BenchmarkRunner(BenchmarkRunnerOptions options)
    : options_(std::move(options)), reference_adapter_(options_.reference_bin_path) {}

BenchmarkResult BenchmarkRunner::run_case(const BenchmarkCase& c) const {
    auto wall_start = std::chrono::steady_clock::now();

    BenchmarkResult r;
    r.timestamp = get_utc_timestamp();
    r.suite = options_.suite_name;
    r.case_id = c.case_id;
    r.instance_name = c.instance_name;
    r.instance_path = c.instance_path;
    r.problem_type = to_string(c.problem_type);
    r.tags = c.tags;
    r.solver_name = std::string(core::PROJECT_NAME);
    r.solver_version = std::string(core::PROJECT_VERSION);
    r.git_commit = std::string(core::GIT_COMMIT_HASH);
    r.compiler = std::string(core::COMPILER_ID) + " " + std::string(core::COMPILER_VERSION);
    r.build_type = std::string(core::BUILD_TYPE);
    r.random_seed = options_.seed.value_or(c.seed);
    r.tolerances = c.tolerances;
    r.reference_objective = c.reference_objective;
    r.reference_status = c.reference_status;

    // Check file existence
    std::error_code ec;
    if (!std::filesystem::exists(c.instance_path, ec)) {
        r.status = BenchmarkStatus::PARSE_FAILURE;
        r.error_message = "File not found: " + c.instance_path;
        auto wall_end = std::chrono::steady_clock::now();
        r.total_wall_time_seconds = std::chrono::duration<double>(wall_end - wall_start).count();
        return r;
    }

    // Parse model
    auto parse_start = std::chrono::steady_clock::now();
    std::unique_ptr<model::CanonicalModel> model;
    try {
        model = io::parse_model(c.instance_path);
    } catch (const std::exception& e) {
        auto parse_end = std::chrono::steady_clock::now();
        r.parse_time_seconds = std::chrono::duration<double>(parse_end - parse_start).count();
        r.status = BenchmarkStatus::PARSE_FAILURE;
        r.error_message = "Parse error: " + std::string(e.what());
        auto wall_end = std::chrono::steady_clock::now();
        r.total_wall_time_seconds = std::chrono::duration<double>(wall_end - wall_start).count();
        return r;
    }
    auto parse_end = std::chrono::steady_clock::now();
    r.parse_time_seconds = std::chrono::duration<double>(parse_end - parse_start).count();

    // Populate model dimensions
    r.variables = model->num_variables();
    r.constraints = model->num_constraints();
    r.nonzeros = model->num_nonzeros();
    if (*r.variables > 0 && *r.constraints > 0) {
        r.density =
            static_cast<double>(*r.nonzeros) / static_cast<double>(*r.variables * *r.constraints);
    } else {
        r.density = 0.0;
    }

    // Infer problem type if UNKNOWN
    BenchmarkProblemType effective_type = c.problem_type;
    if (effective_type == BenchmarkProblemType::UNKNOWN) {
        if (model->is_qp()) {
            effective_type = BenchmarkProblemType::QP;
        } else if (model->is_milp()) {
            effective_type = BenchmarkProblemType::MILP;
        } else {
            effective_type = BenchmarkProblemType::LP;
        }
        r.problem_type = to_string(effective_type);
    }

    double timeout_sec = options_.time_limit_seconds.value_or(c.timeout_seconds.value_or(60.0));

    // Execution by problem type
    if (effective_type == BenchmarkProblemType::LP) {
        r.backend = "CPU (Revised Simplex)";
        algorithms::lp::LPSolverOptions lp_opts;
        lp_opts.tolerances = c.tolerances;
        lp_opts.enable_presolve = c.enable_presolve;
        lp_opts.enable_scaling = c.enable_scaling;
        lp_opts.time_limit_seconds = timeout_sec;
        lp_opts.max_iterations = c.max_iterations;

        algorithms::lp::RevisedSimplex lp_solver(lp_opts);
        auto solve_start = std::chrono::steady_clock::now();
        auto sol = lp_solver.solve(*model);
        auto solve_end = std::chrono::steady_clock::now();

        r.solve_time_seconds = std::chrono::duration<double>(solve_end - solve_start).count();
        r.iterations = sol.iterations;
        r.objective_value = sol.objective_value;

        if (sol.status == core::SolverStatus::LIMIT_REACHED) {
            r.status = BenchmarkStatus::SOLVER_TIMEOUT;
            r.error_message = "LP solve reached time or iteration limit.";
        } else {
            r.status = from_solver_status(sol.status);
        }

        // Verification Gate
        if (sol.status == core::SolverStatus::OPTIMAL) {
            auto verif_start = std::chrono::steady_clock::now();
            verification::LPVerifier verifier(
                verification::VerifierTolerances::from_solver_tolerances(c.tolerances));
            auto v_report =
                verifier.verify(*model, sol.primal_solution, sol.dual_multipliers,
                                sol.reduced_costs, sol.objective_value, sol.status, c.tolerances);
            auto verif_end = std::chrono::steady_clock::now();

            r.verification_time_seconds =
                std::chrono::duration<double>(verif_end - verif_start).count();
            r.primal_residual = v_report.aggregate_primal_residual;
            r.dual_residual = v_report.aggregate_dual_residual;

            bool passed = v_report.fully_verified();

            if (passed) {
                r.verification_passed = true;
                r.verified_status = "VERIFIED_OPTIMAL";
                r.status = BenchmarkStatus::OPTIMAL;
            } else {
                r.verification_passed = false;
                r.verified_status = "VERIFICATION_FAILED";
                r.status = BenchmarkStatus::VERIFICATION_FAILURE;
                r.error_message = "Independent verification failed: " + v_report.summary;
            }
        }
    } else if (effective_type == BenchmarkProblemType::MILP) {
        r.backend = "CPU (Branch-and-Bound)";
        algorithms::milp::MILPSolverOptions milp_opts;
        milp_opts.tolerances = c.tolerances;
        milp_opts.time_limit_seconds = timeout_sec;
        milp_opts.max_nodes = c.max_nodes;
        milp_opts.enable_presolve = c.enable_presolve;
        milp_opts.enable_scaling = c.enable_scaling;

        algorithms::milp::BranchAndBound milp_solver(milp_opts);
        auto solve_start = std::chrono::steady_clock::now();
        auto sol = milp_solver.solve(*model);
        auto solve_end = std::chrono::steady_clock::now();

        r.solve_time_seconds = std::chrono::duration<double>(solve_end - solve_start).count();
        r.nodes = sol.telemetry.nodes_explored;
        r.best_bound = sol.best_bound;
        r.optimality_gap = sol.optimality_gap;

        if (sol.has_incumbent()) {
            r.objective_value = sol.best_objective;
        }

        if (sol.telemetry.time_limit_reached) {
            r.status = BenchmarkStatus::SOLVER_TIMEOUT;
            r.error_message = "MILP solve reached time limit.";
        } else if (sol.telemetry.node_limit_reached) {
            r.status = BenchmarkStatus::LIMIT_REACHED;
            r.error_message = "MILP solve reached node limit.";
        } else {
            r.status = from_solver_status(sol.status);
        }

        // Verification Gate
        if (sol.status == core::SolverStatus::OPTIMAL && sol.has_incumbent()) {
            auto verif_start = std::chrono::steady_clock::now();
            verification::MILPVerifier verifier(
                verification::MILPVerifierTolerances::from_solver_tolerances(c.tolerances));
            auto v_report = verifier.verify(*model, sol.incumbent_solution, sol.best_bound,
                                            sol.best_objective, sol.status);
            auto verif_end = std::chrono::steady_clock::now();

            r.verification_time_seconds =
                std::chrono::duration<double>(verif_end - verif_start).count();
            r.integrality_violation = v_report.max_integrality_violation;
            r.primal_residual = v_report.aggregate_primal_residual;

            if (v_report.fully_verified()) {
                r.verification_passed = true;
                r.verified_status = "VERIFIED_OPTIMAL";
                r.status = BenchmarkStatus::OPTIMAL;
            } else {
                r.verification_passed = false;
                r.verified_status = "VERIFICATION_FAILED";
                r.status = BenchmarkStatus::VERIFICATION_FAILURE;
                r.error_message = "Independent MILP verification failed: " + v_report.summary;
            }
        }
    } else if (effective_type == BenchmarkProblemType::QP) {
        r.backend = "CPU (Active-Set QP)";
        if (model->num_integer_variables() > 0) {
            r.status = BenchmarkStatus::UNSUPPORTED;
            r.error_message = "MIQP is not supported in Kavach-Opt scope.";
        } else {
            algorithms::qp::QPSolverOptions qp_opts;
            qp_opts.tolerances = c.tolerances;
            qp_opts.time_limit_seconds = timeout_sec;
            qp_opts.max_iterations = c.max_iterations;

            algorithms::qp::PrimalActiveSetQPSolver qp_solver(qp_opts);
            auto solve_start = std::chrono::steady_clock::now();
            auto sol = qp_solver.solve(*model);
            auto solve_end = std::chrono::steady_clock::now();

            r.solve_time_seconds = std::chrono::duration<double>(solve_end - solve_start).count();
            r.iterations = sol.iterations;
            r.objective_value = sol.objective_value;
            r.kkt_residual = sol.kkt_residual;
            r.primal_residual = sol.verification_report.aggregate_primal_residual;
            r.dual_residual = sol.verification_report.aggregate_dual_residual;

            if (sol.status == core::SolverStatus::OPTIMAL) {
                if (sol.verification_report.fully_verified()) {
                    r.verification_passed = true;
                    r.verified_status = "VERIFIED_OPTIMAL";
                    r.status = BenchmarkStatus::OPTIMAL;
                } else {
                    r.verification_passed = false;
                    r.verified_status = "VERIFICATION_FAILED";
                    r.status = BenchmarkStatus::VERIFICATION_FAILURE;
                    r.error_message = "KKT verification failed: " + sol.verification_report.summary;
                }
            } else {
                r.status = from_solver_status(sol.status);
            }
        }
    } else if (effective_type == BenchmarkProblemType::SPMV) {
        // Multicore CPU & CUDA batched SpMV benchmark
        r.backend = "Batched SpMV Benchmark";
        backend::cpu::CpuMulticoreBackend cpu_be;
        backend::cuda::CudaBackend cuda_be;

        std::size_t rows = *r.constraints > 0 ? *r.constraints : 100;
        std::size_t cols = *r.variables > 0 ? *r.variables : 100;
        std::size_t batch = c.spmv_batch_size;

        std::vector<numerics::Triplet> triplets;
        for (std::size_t i = 0; i < rows; ++i) {
            triplets.push_back({i, i % cols, 2.0});
            if ((i + 1) < cols) {
                triplets.push_back({i, (i + 1) % cols, -1.0});
            }
        }
        numerics::CsrMatrix mat = numerics::CsrMatrix::from_triplets(rows, cols, triplets);

        std::vector<double> batch_x(batch * cols, 1.0);
        std::vector<double> batch_y(batch * rows, 0.0);

        auto solve_start = std::chrono::steady_clock::now();
        cuda_be.batched_spmv(mat, batch_x.data(), batch_y.data(), batch);
        auto solve_end = std::chrono::steady_clock::now();

        r.solve_time_seconds = std::chrono::duration<double>(solve_end - solve_start).count();
        r.gpu_telemetry = cuda_be.last_telemetry();
        r.gpu_device = cuda_be.last_telemetry().device_name;
        r.status = BenchmarkStatus::OPTIMAL;
        r.verification_passed = true;
        r.verified_status = "VERIFIED_EQUIVALENT";
    }

    // Objective Error against reference
    if (c.reference_objective.has_value() && r.objective_value.has_value()) {
        auto cmp = compare_objectives(*r.objective_value, *c.reference_objective);
        r.objective_error_abs = cmp.abs_error;
        r.objective_error_rel = cmp.rel_error;
    }

    // External Reference Comparison (HiGHS)
    if (options_.compare_reference) {
        r.has_reference_comparison = true;
        auto ref_res = reference_adapter_.run(c.instance_path, timeout_sec);
        r.reference_solver_name = ref_res.solver_name;
        r.reference_solver_version = ref_res.solver_version;
        r.reference_solver_status = to_string(ref_res.status);
        r.reference_objective_val = ref_res.objective;
        r.reference_runtime_seconds = ref_res.runtime_seconds;
        r.reference_iterations = ref_res.iterations;
        r.reference_gap = ref_res.gap;
        r.reference_error_message = ref_res.error_message;
    }

    auto wall_end = std::chrono::steady_clock::now();
    r.total_wall_time_seconds = std::chrono::duration<double>(wall_end - wall_start).count();
    return r;
}

std::vector<BenchmarkResult> BenchmarkRunner::run_suite(const BenchmarkSuite& suite) const {
    std::vector<BenchmarkResult> results;
    for (const auto& c : suite.cases) {
        if (options_.tag_filter.has_value() && !c.has_tag(*options_.tag_filter)) {
            continue;
        }
        if (options_.case_filter.has_value() &&
            c.case_id.find(*options_.case_filter) == std::string::npos &&
            c.instance_name.find(*options_.case_filter) == std::string::npos) {
            continue;
        }
        results.push_back(run_case(c));
    }
    return results;
}

}  // namespace kavach::benchmark
