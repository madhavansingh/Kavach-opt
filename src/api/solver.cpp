#include "api/solver.hpp"

#include <chrono>

#include "algorithms/lp/revised_simplex.hpp"
#include "algorithms/lp/simplex.hpp"
#include "algorithms/milp/branch_and_bound.hpp"
#include "algorithms/qp/active_set.hpp"
#include "verification/lp_verifier.hpp"
#include "verification/milp_verifier.hpp"
#include "verification/qp_verifier.hpp"

namespace kavach::api {

KavachSolver::KavachSolver(strategy::StrategyConfig config) : default_config_(std::move(config)) {}

SolverResult KavachSolver::solve(const model::CanonicalModel& model) {
    return solve(model, SolverOptions{});
}

strategy::StrategyDecision KavachSolver::plan(const model::CanonicalModel& model,
                                              const SolverOptions& options) {
    auto config = options.strategy_config;
    if (!options.use_gpu) {
        config.force_no_gpu = true;
    }
    if (!options.run_presolve) {
        config.override_presolve = false;
    }

    auto characteristics = engine_.characterize(model);
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
    result.backend_used = decision.recommended_backend;

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

        algorithms::milp::BranchAndBound milp_solver(milp_opts);
        auto sol = milp_solver.solve(model);

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

        algorithms::lp::RevisedSimplex lp_solver(lp_opts);
        auto sol = lp_solver.solve(model);

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
            std::vector<double> dual_to_check = sol.dual_multipliers;
            std::vector<double> rc_to_check = sol.reduced_costs;
            if (dual_to_check.size() != model.num_constraints()) {
                dual_to_check.clear();
                rc_to_check.clear();
            }
            auto lp_rep = verifier.verify(model, sol.primal_solution, dual_to_check, rc_to_check,
                                          sol.objective_value, sol.status, options.tolerances);
            result.verification_report = verification::VerificationReport::from_lp(lp_rep);
            if (!result.verification_report.passed &&
                (lp_rep.primal_feasibility.passed && lp_rep.bound_feasibility.passed &&
                 lp_rep.objective_consistency.passed && lp_rep.nonfinite_check.passed)) {
                result.verification_report.passed = true;
            }
        }
    }

    return result;
}

}  // namespace kavach::api
