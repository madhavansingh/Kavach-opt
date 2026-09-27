#include "api/solver.hpp"

#include <chrono>

#include "algorithms/lp/revised_simplex.hpp"
#include "algorithms/lp/simplex.hpp"
#include "algorithms/milp/branch_and_bound.hpp"
#include "algorithms/miqp/miqp_solver.hpp"
#include "algorithms/qp/active_set.hpp"
#include "numerics/model_converter.hpp"
#include "verification/lp_verifier.hpp"
#include "verification/milp_verifier.hpp"
#include "verification/miqp_verifier.hpp"
#include "verification/qp_verifier.hpp"

namespace kavach::api {

KavachSolver::KavachSolver(strategy::StrategyConfig config) : default_config_(std::move(config)) {}

SolverResult KavachSolver::solve(const model::CanonicalModel& model) {
    return solve(model, SolverOptions{});
}

strategy::StrategyDecision KavachSolver::plan(const model::CanonicalModel& model,
                                              const SolverOptions& options) {
    auto config = options.strategy_config;
    const bool explicit_cuda_override =
        config.override_backend.has_value() &&
        *config.override_backend == backend::BackendType::CUDA_GPU;
    if (!options.use_gpu && !explicit_cuda_override) {
        config.force_no_gpu = true;
    }
    if (!options.run_presolve) {
        config.override_presolve = false;
    }

    auto characteristics = engine_.characterize(model);
    if (options.problem_type == ProblemType::MIQP) {
        characteristics.problem_class = strategy::ProblemClass::CONVEX_MIQP;
    } else if (options.problem_type == ProblemType::QP) {
        characteristics.problem_class = strategy::ProblemClass::CONVEX_QP;
    } else if (options.problem_type == ProblemType::MILP) {
        characteristics.problem_class = strategy::ProblemClass::MILP;
    } else if (options.problem_type == ProblemType::LP) {
        characteristics.problem_class = strategy::ProblemClass::LP;
    }
    return engine_.decide(characteristics, config);
}

std::string KavachSolver::explain_plan(const model::CanonicalModel& model,
                                       const SolverOptions& options) {
    auto decision = plan(model, options);
    return decision.explain();
}

SolverResult KavachSolver::solve(const model::CanonicalModel& model, const SolverOptions& options) {
    auto decision = plan(model, options);
    return solve(model, decision, options);
}

SolverResult KavachSolver::solve(const model::CanonicalModel& model,
                                 const strategy::StrategyDecision& decision,
                                 const SolverOptions& options) {
    auto start_time = std::chrono::steady_clock::now();
    SolverResult result;
    result.strategy_decision = decision;

    const backend::BackendType requested_backend =
        options.strategy_config.override_backend.value_or(decision.recommended_backend);
    backend::BackendType selected_backend = decision.recommended_backend;
    std::string initial_fallback_reason;

    if (requested_backend == backend::BackendType::CUDA_GPU &&
        options.strategy_config.force_no_gpu) {
        selected_backend = backend::BackendType::CPU_MULTICORE;
        initial_fallback_reason = "GPU execution disabled via force_no_gpu / --no-gpu";
    }

    const std::size_t num_workers =
        (selected_backend == backend::BackendType::CPU_MULTICORE ||
         requested_backend == backend::BackendType::CPU_MULTICORE ||
         requested_backend == backend::BackendType::CUDA_GPU)
            ? std::max<std::size_t>(2, decision.recommended_cpu_threads)
            : 1;

    auto exec_backend = backend::create_execution_backend(
        requested_backend, selected_backend, num_workers, options.force_cuda_fallback,
        initial_fallback_reason);

    double time_limit_sec = 0.0;
    if (options.time_limit.has_value()) {
        time_limit_sec = static_cast<double>(options.time_limit->count()) / 1000.0;
    }

    if (decision.problem_class == strategy::ProblemClass::CONVEX_QP) {
        algorithms::qp::QPSolverOptions qp_opts;
        qp_opts.enable_presolve = false;  // QP presolve unsupported
        qp_opts.enable_scaling = decision.scaling_enabled;
        qp_opts.time_limit_seconds = time_limit_sec;
        qp_opts.max_iterations = options.max_iterations;

        algorithms::qp::PrimalActiveSetQPSolver qp_solver(qp_opts);
        auto sol = qp_solver.solve(model);

        if (exec_backend != nullptr && model.num_constraints() > 0 &&
            sol.primal_solution.size() == model.num_variables()) {
            const auto orig_csr = numerics::canonical_model_to_csr(model);
            std::vector<double> activities(model.num_constraints(), 0.0);
            exec_backend->batched_spmv(orig_csr, sol.primal_solution.data(),
                                       activities.data(), 1);
        }

        auto end_time = std::chrono::steady_clock::now();
        result.solve_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
        result.status = sol.status;
        result.objective_value = sol.objective_value;
        result.primal_solution = sol.primal_solution;
        result.dual_solution = sol.dual_solution;
        result.iterations = sol.iterations;

        if (sol.is_optimal() || sol.status == core::SolverStatus::FEASIBLE) {
            verification::QPVerifier verifier;
            auto qp_rep = verifier.verify(model, sol.primal_solution, sol.dual_solution,
                                          sol.objective_value, sol.status, options.tolerances);
            result.verification_report = verification::VerificationReport::from_qp(qp_rep);
            if (result.status == core::SolverStatus::OPTIMAL && !qp_rep.fully_verified()) {
                result.status = qp_rep.certified_status;
            }
        }
    } else if (decision.problem_class == strategy::ProblemClass::MILP) {
        algorithms::milp::MILPSolverOptions milp_opts;
        milp_opts.tolerances = decision.recommended_tolerances;
        milp_opts.enable_presolve = (decision.presolve_policy == strategy::PresolvePolicy::FULL);
        milp_opts.enable_scaling = decision.scaling_enabled;
        milp_opts.enable_warm_starts = decision.prefer_dual_simplex_warm_start;
        milp_opts.cut_options = options.cut_options;
        milp_opts.branching_strategy = options.branching_strategy;
        milp_opts.strong_branching_candidates = options.strong_branching_candidates;
        milp_opts.pseudocost_reliability_threshold = options.pseudocost_reliability_threshold;
        milp_opts.enable_primal_heuristic = options.enable_primal_heuristic;
        milp_opts.time_limit_seconds = time_limit_sec;
        milp_opts.max_nodes = options.max_nodes;
        milp_opts.execution_backend = exec_backend.get();
        milp_opts.requested_backend = requested_backend;
        milp_opts.selected_backend = selected_backend;
        milp_opts.num_workers = num_workers;
        milp_opts.force_cuda_fallback = options.force_cuda_fallback;
        milp_opts.initial_fallback_reason = initial_fallback_reason;

        algorithms::milp::BranchAndBound milp_solver(milp_opts);
        auto sol = milp_solver.solve(model, milp_opts);

        auto end_time = std::chrono::steady_clock::now();
        result.solve_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
        result.status = sol.status;
        result.objective_value = sol.best_objective;
        result.primal_solution = sol.incumbent_solution;
        result.optimality_gap = sol.optimality_gap;
        result.nodes = sol.telemetry.nodes_explored;
        result.iterations =
            sol.telemetry.dual_simplex_iterations + sol.telemetry.cold_start_iterations;
        result.cut_telemetry = sol.telemetry.cut_telemetry;
        result.search_telemetry = sol.telemetry.search_telemetry;

        if (sol.is_optimal() && sol.has_incumbent()) {
            verification::MILPVerifier verifier;
            auto milp_rep =
                verifier.verify(model, sol.incumbent_solution, sol.best_bound, sol.best_objective,
                                sol.status, options.tolerances, sol.optimality_gap);
            result.verification_report = verification::VerificationReport::from_milp(milp_rep);
            if (result.status == core::SolverStatus::OPTIMAL && !milp_rep.fully_verified()) {
                result.status = milp_rep.certified_status;
            }
        }
    } else if (decision.problem_class == strategy::ProblemClass::CONVEX_MIQP) {
        algorithms::miqp::MIQPSolverOptions miqp_opts;
        miqp_opts.tolerances = decision.recommended_tolerances;
        miqp_opts.enable_presolve = (decision.presolve_policy == strategy::PresolvePolicy::FULL);
        miqp_opts.enable_scaling = decision.scaling_enabled;
        miqp_opts.cut_options = options.cut_options;
        miqp_opts.branching_strategy = options.branching_strategy;
        miqp_opts.strong_branching_candidates = options.strong_branching_candidates;
        miqp_opts.pseudocost_reliability_threshold = options.pseudocost_reliability_threshold;
        miqp_opts.enable_primal_heuristic = options.enable_primal_heuristic;
        miqp_opts.time_limit_seconds = time_limit_sec;
        miqp_opts.max_nodes = options.max_nodes;

        algorithms::miqp::ConvexMIQPSolver miqp_solver(miqp_opts);
        auto sol = miqp_solver.solve(model);

        if (exec_backend != nullptr && model.num_constraints() > 0 &&
            sol.incumbent_solution.size() == model.num_variables()) {
            const auto orig_csr = numerics::canonical_model_to_csr(model);
            std::vector<double> activities(model.num_constraints(), 0.0);
            exec_backend->batched_spmv(orig_csr, sol.incumbent_solution.data(),
                                       activities.data(), 1);
        }

        auto end_time = std::chrono::steady_clock::now();
        result.solve_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
        result.status = sol.status;
        result.objective_value = sol.best_objective;
        result.primal_solution = sol.incumbent_solution;
        result.optimality_gap = sol.optimality_gap;
        result.nodes = sol.nodes;
        result.iterations = sol.iterations;
        result.search_telemetry = sol.telemetry.search_telemetry;
        result.miqp_telemetry = sol.telemetry;

        if (sol.is_optimal() && sol.has_incumbent()) {
            verification::MIQPVerifier verifier;
            auto miqp_rep =
                verifier.verify(model, sol.incumbent_solution, sol.best_bound, sol.best_objective,
                                sol.status, options.tolerances, sol.optimality_gap);
            result.verification_report = verification::VerificationReport::from_miqp(miqp_rep);
            if (result.status == core::SolverStatus::OPTIMAL && !miqp_rep.fully_verified()) {
                result.status = miqp_rep.certified_status;
            }
        }
    } else {
        algorithms::lp::LPSolverOptions lp_opts;
        lp_opts.tolerances = decision.recommended_tolerances;
        lp_opts.enable_presolve = (decision.presolve_policy == strategy::PresolvePolicy::FULL);
        lp_opts.enable_scaling = decision.scaling_enabled;
        lp_opts.refactorization_interval = decision.lp_refactorization_interval;
        lp_opts.force_bland_rule = decision.lp_force_bland_rule;
        lp_opts.time_limit_seconds = time_limit_sec;
        lp_opts.max_iterations = options.max_iterations;
        lp_opts.execution_backend = exec_backend.get();
        lp_opts.requested_backend = requested_backend;
        lp_opts.selected_backend = selected_backend;
        lp_opts.num_workers = num_workers;
        lp_opts.force_cuda_fallback = options.force_cuda_fallback;
        lp_opts.initial_fallback_reason = initial_fallback_reason;

        algorithms::lp::RevisedSimplex lp_solver(lp_opts);
        auto sol = lp_solver.solve(model, lp_opts);

        auto end_time = std::chrono::steady_clock::now();
        result.solve_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
        result.status = sol.status;
        result.objective_value = sol.objective_value;
        result.primal_solution = sol.primal_solution;
        result.dual_solution = sol.dual_multipliers;
        result.reduced_costs = sol.reduced_costs;
        result.iterations = sol.iterations;

        if (sol.is_optimal()) {
            verification::LPVerifier verifier;
            auto lp_rep =
                verifier.verify(model, sol.primal_solution, sol.dual_multipliers, sol.reduced_costs,
                                sol.objective_value, sol.status, options.tolerances);
            result.verification_report = verification::VerificationReport::from_lp(lp_rep);
            if (result.status == core::SolverStatus::OPTIMAL &&
                !result.verification_report.passed) {
                result.status = lp_rep.certified_status;
            }
        }
    }

    if (exec_backend != nullptr) {
        result.backend_telemetry = exec_backend->execution_telemetry();
    }
    if (result.backend_telemetry.invocations > 0) {
        result.backend_used = result.backend_telemetry.executed_backend;
    } else {
        result.backend_used = backend::BackendType::CPU_SEQUENTIAL;
    }

    return result;
}

}  // namespace kavach::api
