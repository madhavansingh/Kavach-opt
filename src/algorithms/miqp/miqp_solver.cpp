#include "algorithms/miqp/miqp_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "algorithms/milp/branch_and_bound.hpp"
#include "algorithms/milp/primal_heuristic.hpp"
#include "algorithms/milp/pseudocost.hpp"
#include "algorithms/qp/active_set.hpp"
#include "core/error.hpp"
#include "core/status.hpp"
#include "verification/miqp_verifier.hpp"

namespace kavach::algorithms::miqp {

namespace {

using Clock = std::chrono::steady_clock;

struct MIQPNode {
    std::size_t id{0};
    std::size_t depth{0};
    double qp_bound{0.0};
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;
    std::vector<double> relaxation_solution;
};

struct MinBestBound {
    bool operator()(const MIQPNode& a, const MIQPNode& b) const noexcept {
        if (std::abs(a.qp_bound - b.qp_bound) > 1e-9) {
            return a.qp_bound > b.qp_bound;
        }
        if (a.depth != b.depth) {
            return a.depth > b.depth;
        }
        return a.id > b.id;
    }
};

struct MaxBestBound {
    bool operator()(const MIQPNode& a, const MIQPNode& b) const noexcept {
        if (std::abs(a.qp_bound - b.qp_bound) > 1e-9) {
            return a.qp_bound < b.qp_bound;
        }
        if (a.depth != b.depth) {
            return a.depth > b.depth;
        }
        return a.id > b.id;
    }
};

bool is_better(double new_obj, double inc_obj, model::ObjectiveSense sense) noexcept {
    if (sense == model::ObjectiveSense::MINIMIZE) {
        return new_obj < inc_obj;
    }
    return new_obj > inc_obj;
}

double worst_bound(model::ObjectiveSense sense) noexcept {
    if (sense == model::ObjectiveSense::MINIMIZE) {
        return std::numeric_limits<double>::infinity();
    }
    return -std::numeric_limits<double>::infinity();
}

double compute_gap(double z_incumbent, double z_bound) noexcept {
    if (std::isinf(z_incumbent) || std::isinf(z_bound)) {
        return 1.0;
    }
    return std::abs(z_incumbent - z_bound) / std::max(1.0, std::abs(z_incumbent));
}

bool dominated(double qp_bound, double incumbent_obj, model::ObjectiveSense sense,
               double tol) noexcept {
    if (sense == model::ObjectiveSense::MINIMIZE) {
        return qp_bound >= incumbent_obj - tol;
    }
    return qp_bound <= incumbent_obj + tol;
}

bool is_integer_feasible(const model::CanonicalModel& model, const std::vector<double>& x,
                         double int_tol) noexcept {
    const std::size_t n = std::min(x.size(), model.num_variables());
    for (std::size_t j = 0; j < n; ++j) {
        const auto& var = model.variable(static_cast<model::VarId>(j));
        if (var.is_integer() || var.is_binary()) {
            double frac = std::abs(x[j] - std::round(x[j]));
            if (frac > int_tol) {
                return false;
            }
        }
    }
    return true;
}

std::optional<model::VarId> select_most_fractional(const model::CanonicalModel& model,
                                                   const std::vector<double>& x,
                                                   double int_tol) noexcept {
    double best_score = -1.0;
    std::optional<model::VarId> best_var;
    const std::size_t n = std::min(x.size(), model.num_variables());

    for (std::size_t j = 0; j < n; ++j) {
        const auto& var = model.variable(static_cast<model::VarId>(j));
        if (!var.is_integer() && !var.is_binary()) {
            continue;
        }
        double frac = x[j] - std::floor(x[j]);
        if (frac <= int_tol || frac >= 1.0 - int_tol) {
            continue;
        }
        double score = std::min(frac, 1.0 - frac);
        if (score > best_score ||
            (score == best_score &&
             static_cast<model::VarId>(j) < best_var.value_or(model::INVALID_VAR_ID))) {
            best_score = score;
            best_var = static_cast<model::VarId>(j);
        }
    }
    return best_var;
}

struct MIQPBranchCandidate {
    model::VarId var_id{model::INVALID_VAR_ID};
    double fractionality{0.0};
    double score{0.0};
    std::size_t down_obs{0};
    std::size_t up_obs{0};
    bool is_reliable{false};
};

}  // namespace

MIQPSolution ConvexMIQPSolver::solve(const model::CanonicalModel& model) {
    return solve(model, options_);
}

MIQPSolution ConvexMIQPSolver::solve(const model::CanonicalModel& model,
                                     const core::SolverTolerances& tolerances) {
    MIQPSolverOptions opts = options_;
    opts.tolerances = tolerances;
    return solve(model, opts);
}

MIQPSolution ConvexMIQPSolver::solve(const model::CanonicalModel& orig_model,
                                     const MIQPSolverOptions& opts) {
    const auto start_time = Clock::now();
    const auto& tols = opts.tolerances;
    const model::ObjectiveSense sense = orig_model.objective_sense();
    const bool is_minimize = (sense == model::ObjectiveSense::MINIMIZE);

    MIQPSolution result;
    MIQPTelemetry& tel = result.telemetry;

    auto elapsed = [&]() -> double {
        auto now = Clock::now();
        return std::chrono::duration<double>(now - start_time).count();
    };

    auto time_limit_hit = [&]() -> bool {
        return opts.time_limit_seconds > 0.0 && elapsed() > opts.time_limit_seconds;
    };

    // --- 1. Model Validation ---
    auto val_res = orig_model.validate();
    if (!val_res.valid) {
        result.status = core::SolverStatus::UNKNOWN;
        std::string msg = "Model validation failed:";
        for (const auto& e : val_res.errors) {
            msg += " [" + e.component + ": " + e.message + "]";
        }
        result.message = msg;
        tel.message = msg;
        tel.solve_time_seconds = elapsed();
        return result;
    }

    // --- 1b. Check Empty Integer Domain ---
    for (std::size_t j = 0; j < orig_model.num_variables(); ++j) {
        const auto& var = orig_model.variable(static_cast<model::VarId>(j));
        if (var.is_integer() || var.is_binary()) {
            double eff_lb = var.lower_bound;
            double eff_ub = var.upper_bound;
            if (var.is_binary()) {
                eff_lb = std::max(0.0, eff_lb);
                eff_ub = std::min(1.0, eff_ub);
            }
            if (std::ceil(eff_lb - tols.integrality_tol) >
                std::floor(eff_ub + tols.integrality_tol)) {
                result.status = core::SolverStatus::INFEASIBLE;
                result.message = "Empty integer domain for variable " + var.name;
                tel.message = result.message;
                tel.solve_time_seconds = elapsed();
                return result;
            }
        }
    }

    // --- 1c. Maximization MIQP Transformation to Standard Minimization ---
    if (!is_minimize && orig_model.objective().is_quadratic()) {
        model::CanonicalModel min_model = orig_model;
        min_model.set_objective_sense(model::ObjectiveSense::MINIMIZE);
        min_model.set_objective_offset(-orig_model.objective_offset());
        min_model.objective().linear_terms.clear();
        min_model.objective().quadratic_terms.clear();
        for (const auto& term : orig_model.objective().linear_terms.terms()) {
            min_model.add_objective_linear_term(term.var_id, -term.coefficient);
        }
        for (const auto& qterm : orig_model.objective().quadratic_terms.terms()) {
            min_model.add_objective_quadratic_term(qterm.var1_id, qterm.var2_id,
                                                   -qterm.coefficient);
        }

        auto min_sol = solve(min_model, opts);
        result = min_sol;
        if (min_sol.status == core::SolverStatus::OPTIMAL) {
            result.best_objective = -min_sol.best_objective;
            result.best_bound = -min_sol.best_bound;
            result.telemetry.root_qp_bound = -min_sol.telemetry.root_qp_bound;
            result.telemetry.final_best_bound = -min_sol.telemetry.final_best_bound;
            result.message =
                "Maximization MIQP solved via negated minimization: " + min_sol.message;

            verification::MIQPVerifier verifier(
                verification::MIQPVerifierTolerances::from_solver_tolerances(tols));
            result.verification_report =
                verifier.verify(orig_model, result.incumbent_solution, result.best_bound,
                                result.best_objective, result.status, tols, result.optimality_gap);
            result.status = result.verification_report.certified_status;
        }
        result.telemetry.solve_time_seconds = elapsed();
        return result;
    }

    // --- 2. Zero-Q Equivalence: Bypass to MILP if Q = 0 ---
    if (!orig_model.objective().is_quadratic()) {
        milp::MILPSolverOptions milp_opts;
        milp_opts.tolerances = tols;
        milp_opts.enable_presolve = opts.enable_presolve;
        milp_opts.enable_scaling = opts.enable_scaling;
        milp_opts.branching_strategy = opts.branching_strategy;
        milp_opts.strong_branching_candidates = opts.strong_branching_candidates;
        milp_opts.pseudocost_reliability_threshold = opts.pseudocost_reliability_threshold;
        milp_opts.enable_primal_heuristic = opts.enable_primal_heuristic;
        milp_opts.time_limit_seconds = opts.time_limit_seconds;
        milp_opts.max_nodes = opts.max_nodes;
        milp_opts.cut_options = opts.cut_options;

        milp::BranchAndBound milp_solver(milp_opts);
        auto milp_sol = milp_solver.solve(orig_model);

        result.status = milp_sol.status;
        result.best_objective = milp_sol.best_objective;
        result.best_bound = milp_sol.best_bound;
        result.optimality_gap = milp_sol.optimality_gap;
        result.incumbent_solution = milp_sol.incumbent_solution;
        result.nodes = milp_sol.telemetry.nodes_explored;
        result.iterations =
            milp_sol.telemetry.dual_simplex_iterations + milp_sol.telemetry.cold_start_iterations;
        result.message = "Zero Q objective: solved via MILP Branch-and-Bound. " + milp_sol.message;
        tel.solve_time_seconds = elapsed();

        if (milp_sol.is_optimal()) {
            verification::MIQPVerifier verifier(
                verification::MIQPVerifierTolerances::from_solver_tolerances(tols));
            result.verification_report =
                verifier.verify(orig_model, milp_sol.incumbent_solution, milp_sol.best_bound,
                                milp_sol.best_objective, milp_sol.status, tols);
        }
        return result;
    }

    // --- 3. Pure Continuous QP Equivalence: Bypass if no discrete variables ---
    if (orig_model.num_integer_variables() == 0 && orig_model.num_binary_variables() == 0) {
        qp::QPSolverOptions qp_opts;
        qp_opts.tolerances = tols;
        qp_opts.enable_presolve = opts.enable_presolve;
        qp_opts.enable_scaling = opts.enable_scaling;
        qp_opts.time_limit_seconds = opts.time_limit_seconds;

        qp::PrimalActiveSetQPSolver qp_solver(qp_opts);
        auto qp_sol = qp_solver.solve(orig_model, qp_opts);

        result.status = qp_sol.status;
        result.best_objective = qp_sol.objective_value;
        result.best_bound = qp_sol.objective_value;
        result.optimality_gap = (qp_sol.is_optimal()) ? 0.0 : 1.0;
        result.incumbent_solution = qp_sol.primal_solution;
        result.iterations = qp_sol.iterations;
        result.nodes = 1;
        result.message =
            "Pure continuous QP: solved via Active-Set QP. " + qp_sol.telemetry.error_message;
        tel.qp_relaxations_solved = 1;
        tel.qp_relaxation_iterations = qp_sol.iterations;
        tel.qp_kkt_residual = qp_sol.kkt_residual;
        tel.solve_time_seconds = elapsed();

        if (qp_sol.is_optimal()) {
            verification::MIQPVerifier verifier(
                verification::MIQPVerifierTolerances::from_solver_tolerances(tols));
            result.verification_report =
                verifier.verify(orig_model, qp_sol.primal_solution, qp_sol.objective_value,
                                qp_sol.objective_value, qp_sol.status, tols);
        }
        return result;
    }

    // --- 4. Convexity / PSD Validation ---
    numerics::CsrMatrix Q_mat = qp::PrimalActiveSetQPSolver::build_hessian(orig_model);
    std::string psd_error;
    if (!qp::PrimalActiveSetQPSolver::check_psd(Q_mat, tols.pivot_tol, &psd_error)) {
        result.status = core::SolverStatus::NUMERICAL_FAILURE;
        std::string err_msg = "Non-convex MIQP (indefinite Q) is unsupported: " + psd_error;
        result.message = err_msg;
        tel.message = err_msg;
        tel.solve_time_seconds = elapsed();
        result.verification_report.certified_status = core::SolverStatus::NUMERICAL_FAILURE;
        result.verification_report.summary = err_msg;
        return result;
    }

    const std::size_t n = orig_model.num_variables();

    // --- 5. Continuous Relaxation Model Setup ---
    model::CanonicalModel work_model = orig_model;
    std::vector<double> orig_lb(n);
    std::vector<double> orig_ub(n);
    for (std::size_t j = 0; j < n; ++j) {
        const auto& v = orig_model.variable(static_cast<model::VarId>(j));
        orig_lb[j] = v.lower_bound;
        orig_ub[j] = v.upper_bound;
        work_model.set_variable_type(static_cast<model::VarId>(j), model::VariableType::CONTINUOUS);
    }

    qp::QPSolverOptions qp_node_opts;
    qp_node_opts.tolerances = tols;
    qp_node_opts.enable_presolve = false;
    qp_node_opts.enable_scaling = opts.enable_scaling;
    qp::PrimalActiveSetQPSolver qp_engine(qp_node_opts);

    auto solve_node_qp = [&](const std::vector<double>& lb,
                             const std::vector<double>& ub) -> qp::QPSolution {
        for (std::size_t j = 0; j < n; ++j) {
            if (lb[j] > ub[j] + tols.primal_feasibility_tol) {
                qp::QPSolution inf_sol;
                inf_sol.status = core::SolverStatus::INFEASIBLE;
                inf_sol.telemetry.error_message = "Variable lower bound exceeds upper bound";
                return inf_sol;
            }
        }
        for (std::size_t j = 0; j < n; ++j) {
            work_model.set_variable_lower_bound(static_cast<model::VarId>(j), lb[j]);
            work_model.set_variable_upper_bound(static_cast<model::VarId>(j), ub[j]);
        }
        auto qp_start = Clock::now();
        auto qp_sol = qp_engine.solve(work_model, qp_node_opts);
        double qp_sec = std::chrono::duration<double>(Clock::now() - qp_start).count();

        tel.qp_relaxation_time_seconds += qp_sec;
        tel.qp_relaxation_iterations += qp_sol.iterations;
        ++tel.qp_relaxations_solved;
        if (qp_sol.kkt_residual > tel.qp_kkt_residual) {
            tel.qp_kkt_residual = qp_sol.kkt_residual;
        }
        if (qp_sol.status != core::SolverStatus::OPTIMAL &&
            qp_sol.status != core::SolverStatus::INFEASIBLE) {
            ++tel.failed_qp_relaxations;
        }
        return qp_sol;
    };

    // --- 6. Solve Root QP Relaxation ---
    auto root_sol = solve_node_qp(orig_lb, orig_ub);

    if (root_sol.status == core::SolverStatus::INFEASIBLE) {
        result.status = core::SolverStatus::INFEASIBLE;
        result.message = "Root QP relaxation is infeasible — MIQP is infeasible.";
        tel.solve_time_seconds = elapsed();
        return result;
    }
    if (root_sol.status != core::SolverStatus::OPTIMAL) {
        result.status = core::SolverStatus::NUMERICAL_FAILURE;
        result.message = "Root QP relaxation failed: " + root_sol.telemetry.error_message;
        tel.solve_time_seconds = elapsed();
        return result;
    }

    tel.root_qp_bound = root_sol.objective_value;
    tel.final_best_bound = root_sol.objective_value;

    double incumbent_obj = worst_bound(sense);
    std::vector<double> incumbent;

    // Check if root continuous relaxation is integer feasible
    if (is_integer_feasible(orig_model, root_sol.primal_solution, tols.integrality_tol)) {
        incumbent_obj = root_sol.objective_value;
        incumbent = root_sol.primal_solution;
        tel.incumbent_updates = 1;
        tel.integer_candidates_found = 1;
        tel.final_best_bound = incumbent_obj;
        tel.search_telemetry.incumbent_source = "ROOT_QP";
        result.status = core::SolverStatus::OPTIMAL;
        result.best_objective = incumbent_obj;
        result.best_bound = incumbent_obj;
        result.optimality_gap = 0.0;
        result.incumbent_solution = incumbent;
        result.nodes = 1;
        result.iterations = tel.qp_relaxation_iterations;
        result.message = "Root QP relaxation is integer feasible.";
        tel.solve_time_seconds = elapsed();

        verification::MIQPVerifier verifier(
            verification::MIQPVerifierTolerances::from_solver_tolerances(tols));
        result.verification_report =
            verifier.verify(orig_model, incumbent, incumbent_obj, incumbent_obj,
                            core::SolverStatus::OPTIMAL, tols, 0.0);
        result.status = result.verification_report.certified_status;
        return result;
    }

    // --- 7. Search Infrastructure ---
    milp::PseudocostTracker pseudocost_tracker(n, sense);
    milp::PrimalHeuristic primal_heuristic(tols, opts.heuristic_max_repair_rounds);
    tel.search_telemetry.branching_strategy_used = milp::to_string(opts.branching_strategy);

    auto try_primal_heuristic = [&](const std::vector<double>& x_qp) {
        if (!opts.enable_primal_heuristic) {
            return;
        }
        ++tel.search_telemetry.heuristic_calls;
        auto heur_res = primal_heuristic.search(orig_model, x_qp);
        if (heur_res.found_feasible &&
            is_integer_feasible(orig_model, heur_res.solution, tols.integrality_tol)) {
            ++tel.search_telemetry.heuristic_feasible_candidates;
            ++tel.integer_candidates_found;
            double h_obj =
                qp::PrimalActiveSetQPSolver::evaluate_objective(orig_model, heur_res.solution);
            if (std::isfinite(h_obj) &&
                (incumbent.empty() || is_better(h_obj, incumbent_obj, sense))) {
                incumbent = heur_res.solution;
                incumbent_obj = h_obj;
                ++tel.incumbent_updates;
                ++tel.search_telemetry.heuristic_incumbent_updates;
                tel.search_telemetry.incumbent_source = "PRIMAL_HEURISTIC";
            }
        }
    };

    // Try heuristic on root solution
    try_primal_heuristic(root_sol.primal_solution);

    // Node priority queues
    std::priority_queue<MIQPNode, std::vector<MIQPNode>, MinBestBound> pq_min;
    std::priority_queue<MIQPNode, std::vector<MIQPNode>, MaxBestBound> pq_max;

    std::size_t next_node_id = 0;
    MIQPNode root_node;
    root_node.id = next_node_id++;
    root_node.depth = 0;
    root_node.qp_bound = root_sol.objective_value;
    root_node.lower_bounds = orig_lb;
    root_node.upper_bounds = orig_ub;
    root_node.relaxation_solution = root_sol.primal_solution;

    if (is_minimize) {
        pq_min.push(std::move(root_node));
    } else {
        pq_max.push(std::move(root_node));
    }

    auto queue_empty = [&]() -> bool { return is_minimize ? pq_min.empty() : pq_max.empty(); };

    auto pop_best_node = [&]() -> MIQPNode {
        if (is_minimize) {
            auto top = pq_min.top();
            pq_min.pop();
            return top;
        } else {
            auto top = pq_max.top();
            pq_max.pop();
            return top;
        }
    };

    auto current_best_dual_bound = [&]() -> double {
        if (is_minimize) {
            return pq_min.empty() ? incumbent_obj : pq_min.top().qp_bound;
        } else {
            return pq_max.empty() ? incumbent_obj : pq_max.top().qp_bound;
        }
    };

    // --- 8. Branch-and-Bound Loop ---
    while (!queue_empty()) {
        if (tel.nodes_explored >= opts.max_nodes) {
            tel.node_limit_reached = true;
            break;
        }
        if (time_limit_hit()) {
            tel.time_limit_reached = true;
            break;
        }

        MIQPNode current = pop_best_node();
        ++tel.nodes_explored;

        // Bound-based pruning
        if (!incumbent.empty() &&
            dominated(current.qp_bound, incumbent_obj, sense, tols.residual_tol)) {
            ++tel.nodes_pruned_bound;
            continue;
        }

        // Apply primal heuristic at node
        try_primal_heuristic(current.relaxation_solution);

        // Branching candidate gathering
        std::vector<MIQPBranchCandidate> candidates;
        for (std::size_t j = 0; j < n; ++j) {
            const auto& var = orig_model.variable(static_cast<model::VarId>(j));
            if (!var.is_integer() && !var.is_binary()) {
                continue;
            }
            double x_j_val = current.relaxation_solution[j];
            double frac = x_j_val - std::floor(x_j_val);
            if (frac <= tols.integrality_tol || frac >= 1.0 - tols.integrality_tol) {
                continue;
            }

            MIQPBranchCandidate cand;
            cand.var_id = static_cast<model::VarId>(j);
            cand.fractionality = std::min(frac, 1.0 - frac);
            const auto& p_entry = pseudocost_tracker.entry(cand.var_id);
            cand.down_obs = p_entry.down_count;
            cand.up_obs = p_entry.up_count;
            cand.is_reliable = p_entry.is_reliable(opts.pseudocost_reliability_threshold);
            cand.score = pseudocost_tracker.compute_score(cand.var_id, x_j_val);
            candidates.push_back(cand);
        }

        if (candidates.empty()) {
            // Integer feasible relaxation at this node
            ++tel.integer_candidates_found;
            double node_obj = qp::PrimalActiveSetQPSolver::evaluate_objective(
                orig_model, current.relaxation_solution);
            if (incumbent.empty() || is_better(node_obj, incumbent_obj, sense)) {
                incumbent = current.relaxation_solution;
                incumbent_obj = node_obj;
                ++tel.incumbent_updates;
                tel.search_telemetry.incumbent_source = "NODE_QP_INTEGER";
            }
            ++tel.nodes_pruned_integer;
            continue;
        }

        // Branching strategy resolution
        milp::BranchingStrategy strat = opts.branching_strategy;
        if (strat == milp::BranchingStrategy::AUTO) {
            strat = milp::BranchingStrategy::RELIABILITY;
        }

        model::VarId branch_var = candidates[0].var_id;
        if (strat == milp::BranchingStrategy::MOST_FRACTIONAL) {
            auto mf = select_most_fractional(orig_model, current.relaxation_solution,
                                             tols.integrality_tol);
            if (mf.has_value()) {
                branch_var = *mf;
            }
        } else {
            // Check reliability / strong branching
            std::size_t num_probed = 0;
            const std::size_t max_probes = opts.strong_branching_candidates;

            for (auto& cand : candidates) {
                bool need_probe = false;
                if (strat == milp::BranchingStrategy::STRONG) {
                    need_probe = (num_probed < max_probes);
                } else if (strat == milp::BranchingStrategy::RELIABILITY) {
                    need_probe = (!cand.is_reliable && num_probed < max_probes);
                }

                if (need_probe) {
                    ++num_probed;
                    ++tel.search_telemetry.strong_branching_calls;
                    ++tel.search_telemetry.strong_branching_candidates;
                    tel.strong_qp_probes += 2;

                    double x_val = current.relaxation_solution[cand.var_id];
                    double floor_val = std::floor(x_val);
                    double ceil_val = std::ceil(x_val);

                    // Down probe
                    auto probe_down_ub = current.upper_bounds;
                    probe_down_ub[cand.var_id] = std::min(probe_down_ub[cand.var_id], floor_val);
                    auto down_sol = solve_node_qp(current.lower_bounds, probe_down_ub);

                    // Up probe
                    auto probe_up_lb = current.lower_bounds;
                    probe_up_lb[cand.var_id] = std::max(probe_up_lb[cand.var_id], ceil_val);
                    auto up_sol = solve_node_qp(probe_up_lb, current.upper_bounds);

                    double down_bound =
                        down_sol.is_optimal() ? down_sol.objective_value : worst_bound(sense);
                    double up_bound =
                        up_sol.is_optimal() ? up_sol.objective_value : worst_bound(sense);

                    pseudocost_tracker.update_down(cand.var_id, current.qp_bound, down_bound, x_val,
                                                   tols.integrality_tol);
                    pseudocost_tracker.update_up(cand.var_id, current.qp_bound, up_bound, x_val,
                                                 tols.integrality_tol);
                    ++tel.search_telemetry.pseudocost_updates;
                }
            }

            // Rescore and sort candidates
            for (auto& cand : candidates) {
                cand.score = pseudocost_tracker.compute_score(
                    cand.var_id, current.relaxation_solution[cand.var_id]);
                const auto& p_entry = pseudocost_tracker.entry(cand.var_id);
                cand.down_obs = p_entry.down_count;
                cand.up_obs = p_entry.up_count;
            }

            std::stable_sort(candidates.begin(), candidates.end(),
                             [](const MIQPBranchCandidate& a, const MIQPBranchCandidate& b) {
                                 if (std::abs(a.score - b.score) > 1e-9) {
                                     return a.score > b.score;
                                 }
                                 if (std::abs(a.fractionality - b.fractionality) > 1e-9) {
                                     return a.fractionality > b.fractionality;
                                 }
                                 return a.var_id < b.var_id;
                             });
            branch_var = candidates[0].var_id;
        }

        // Branch execution on variable branch_var
        double x_branch = current.relaxation_solution[branch_var];
        double floor_val = std::floor(x_branch);
        double ceil_val = std::ceil(x_branch);

        // --- Down Branch Child ---
        if (floor_val >= current.lower_bounds[branch_var] - tols.primal_feasibility_tol) {
            auto child_lb = current.lower_bounds;
            auto child_ub = current.upper_bounds;
            child_ub[branch_var] = std::min(child_ub[branch_var], floor_val);

            if (child_lb[branch_var] <= child_ub[branch_var] + tols.primal_feasibility_tol) {
                auto child_sol = solve_node_qp(child_lb, child_ub);
                if (child_sol.is_optimal()) {
                    pseudocost_tracker.update_down(branch_var, current.qp_bound,
                                                   child_sol.objective_value, x_branch,
                                                   tols.integrality_tol);
                    ++tel.search_telemetry.pseudocost_updates;

                    if (incumbent.empty() || !dominated(child_sol.objective_value, incumbent_obj,
                                                        sense, tols.residual_tol)) {
                        if (is_integer_feasible(orig_model, child_sol.primal_solution,
                                                tols.integrality_tol)) {
                            ++tel.integer_candidates_found;
                            double c_obj = qp::PrimalActiveSetQPSolver::evaluate_objective(
                                orig_model, child_sol.primal_solution);
                            if (incumbent.empty() || is_better(c_obj, incumbent_obj, sense)) {
                                incumbent = child_sol.primal_solution;
                                incumbent_obj = c_obj;
                                ++tel.incumbent_updates;
                                tel.search_telemetry.incumbent_source = "CHILD_QP_INTEGER";
                            }
                            ++tel.nodes_pruned_integer;
                        } else {
                            MIQPNode child_node;
                            child_node.id = next_node_id++;
                            child_node.depth = current.depth + 1;
                            child_node.qp_bound = child_sol.objective_value;
                            child_node.lower_bounds = std::move(child_lb);
                            child_node.upper_bounds = std::move(child_ub);
                            child_node.relaxation_solution = std::move(child_sol.primal_solution);

                            if (is_minimize) {
                                pq_min.push(std::move(child_node));
                            } else {
                                pq_max.push(std::move(child_node));
                            }
                        }
                    } else {
                        ++tel.nodes_pruned_bound;
                    }
                } else if (child_sol.status == core::SolverStatus::INFEASIBLE) {
                    ++tel.nodes_pruned_infeasible;
                }
            }
        }

        // --- Up Branch Child ---
        if (ceil_val <= current.upper_bounds[branch_var] + tols.primal_feasibility_tol) {
            auto child_lb = current.lower_bounds;
            auto child_ub = current.upper_bounds;
            child_lb[branch_var] = std::max(child_lb[branch_var], ceil_val);

            if (child_lb[branch_var] <= child_ub[branch_var] + tols.primal_feasibility_tol) {
                auto child_sol = solve_node_qp(child_lb, child_ub);
                if (child_sol.is_optimal()) {
                    pseudocost_tracker.update_up(branch_var, current.qp_bound,
                                                 child_sol.objective_value, x_branch,
                                                 tols.integrality_tol);
                    ++tel.search_telemetry.pseudocost_updates;

                    if (incumbent.empty() || !dominated(child_sol.objective_value, incumbent_obj,
                                                        sense, tols.residual_tol)) {
                        if (is_integer_feasible(orig_model, child_sol.primal_solution,
                                                tols.integrality_tol)) {
                            ++tel.integer_candidates_found;
                            double c_obj = qp::PrimalActiveSetQPSolver::evaluate_objective(
                                orig_model, child_sol.primal_solution);
                            if (incumbent.empty() || is_better(c_obj, incumbent_obj, sense)) {
                                incumbent = child_sol.primal_solution;
                                incumbent_obj = c_obj;
                                ++tel.incumbent_updates;
                                tel.search_telemetry.incumbent_source = "CHILD_QP_INTEGER";
                            }
                            ++tel.nodes_pruned_integer;
                        } else {
                            MIQPNode child_node;
                            child_node.id = next_node_id++;
                            child_node.depth = current.depth + 1;
                            child_node.qp_bound = child_sol.objective_value;
                            child_node.lower_bounds = std::move(child_lb);
                            child_node.upper_bounds = std::move(child_ub);
                            child_node.relaxation_solution = std::move(child_sol.primal_solution);

                            if (is_minimize) {
                                pq_min.push(std::move(child_node));
                            } else {
                                pq_max.push(std::move(child_node));
                            }
                        }
                    } else {
                        ++tel.nodes_pruned_bound;
                    }
                } else if (child_sol.status == core::SolverStatus::INFEASIBLE) {
                    ++tel.nodes_pruned_infeasible;
                }
            }
        }
    }

    // --- 9. Final Results & Independent Verification ---
    tel.solve_time_seconds = elapsed();
    tel.final_best_bound = current_best_dual_bound();

    if (incumbent.empty()) {
        result.status = core::SolverStatus::INFEASIBLE;
        result.message = "Branch-and-bound tree exhausted with no integer feasible solution found.";
        return result;
    }

    double final_gap = compute_gap(incumbent_obj, tel.final_best_bound);
    result.best_objective = incumbent_obj;
    result.best_bound = tel.final_best_bound;
    result.optimality_gap = final_gap;
    result.incumbent_solution = incumbent;
    result.nodes = tel.nodes_explored;
    result.iterations = tel.qp_relaxation_iterations;

    if (tel.node_limit_reached || tel.time_limit_reached) {
        result.status = core::SolverStatus::FEASIBLE;
        result.message = std::string(tel.node_limit_reached ? "Node" : "Time") +
                         " limit reached. Incumbent: " + std::to_string(incumbent_obj) +
                         ", gap: " + std::to_string(final_gap);
        return result;
    }

    // Tree exhausted with incumbent: verify
    verification::MIQPVerifier verifier(
        verification::MIQPVerifierTolerances::from_solver_tolerances(tols));
    result.verification_report =
        verifier.verify(orig_model, incumbent, tel.final_best_bound, incumbent_obj,
                        core::SolverStatus::OPTIMAL, tols, final_gap);

    if (result.verification_report.certified_status == core::SolverStatus::OPTIMAL) {
        result.status = core::SolverStatus::OPTIMAL;
        result.optimality_gap = result.verification_report.recomputed_gap;
        result.message = "OPTIMAL certified by independent MIQP verification. " +
                         result.verification_report.summary;
    } else {
        result.status = result.verification_report.certified_status;
        result.message = "Verification adjusted status: " + result.verification_report.summary;
    }

    return result;
}

}  // namespace kavach::algorithms::miqp
