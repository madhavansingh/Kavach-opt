/**
 * @file branch_and_bound.cpp
 * @brief Production Branch-and-Bound MILP Engine [G] — Kavach-Opt Blueprint Section 9, 11, 32.
 */

#include "algorithms/milp/branch_and_bound.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <queue>
#include <sstream>
#include <vector>

#include "algorithms/lp/lp_types.hpp"
#include "algorithms/lp/simplex.hpp"
#include "algorithms/milp/cutting_plane_engine.hpp"
#include "algorithms/milp/primal_heuristic.hpp"
#include "algorithms/milp/pseudocost.hpp"
#include "algorithms/milp/search_types.hpp"
#include "algorithms/milp/strong_branching.hpp"
#include "core/error.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"
#include "model/model.hpp"
#include "verification/milp_verifier.hpp"

namespace kavach::algorithms::milp {

// ============================================================================
// Internal types
// ============================================================================

namespace {

using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

/** A single bound modification applied on top of the base model for a given node. */
struct BoundModification {
    model::VarId var_id{model::INVALID_VAR_ID};
    bool is_lower{true};  ///< true = lower bound modification, false = upper bound
    double new_bound{0.0};
    double old_bound{0.0};
};

/** Node in the B&B tree. */
struct BranchNode {
    std::uint64_t node_id{0};
    double lp_bound{0.0};  ///< LP relaxation objective at parent (used for queue priority)
    std::vector<BoundModification> mods;             ///< Bound mods relative to base model
    std::optional<lp::BasisState> warm_start_basis;  ///< Inherited basis from parent solve
    model::VarId branch_var{model::INVALID_VAR_ID};  ///< Variable branched on to create this node
    bool is_down_branch{true};  ///< True if left/down branch, false if right/up
    double parent_obj{0.0};     ///< LP relaxation objective of parent node
    double fractionality{0.0};  ///< Parent fractional value of branch_var
    std::size_t depth{0};       ///< Tree depth (root = 0)
};

struct MinBestBound {
    bool operator()(const BranchNode& a, const BranchNode& b) const noexcept {
        if (std::abs(a.lp_bound - b.lp_bound) > 1e-9) {
            return a.lp_bound > b.lp_bound;  // min-heap: smaller bound = higher priority
        }
        if (a.depth != b.depth) {
            return a.depth < b.depth;  // secondary key: deeper nodes first (diving for incumbents)
        }
        return a.node_id > b.node_id;  // deterministic tie-break
    }
};

struct MaxBestBound {
    bool operator()(const BranchNode& a, const BranchNode& b) const noexcept {
        if (std::abs(a.lp_bound - b.lp_bound) > 1e-9) {
            return a.lp_bound < b.lp_bound;  // max-heap: larger bound = higher priority
        }
        if (a.depth != b.depth) {
            return a.depth < b.depth;  // secondary key: deeper nodes first
        }
        return a.node_id > b.node_id;
    }
};

// ============================================================================
// Helpers
// ============================================================================

/** Apply a list of BoundModifications to the model in-place. */
void apply_modifications(model::CanonicalModel& m, const std::vector<BoundModification>& mods) {
    for (const auto& mod : mods) {
        auto& var = m.variable(mod.var_id);
        if (mod.is_lower) {
            var.lower_bound = mod.new_bound;
        } else {
            var.upper_bound = mod.new_bound;
        }
    }
}

/** Restore original bounds after node solve. */
void restore_modifications(model::CanonicalModel& m, const std::vector<BoundModification>& mods) {
    for (const auto& mod : mods) {
        auto& var = m.variable(mod.var_id);
        if (mod.is_lower) {
            var.lower_bound = mod.old_bound;
        } else {
            var.upper_bound = mod.old_bound;
        }
    }
}

/** Compute honest optimality gap: |z_inc - z_bound| / max(1, |z_inc|). */
double compute_gap(double z_incumbent, double z_bound) noexcept {
    return std::abs(z_incumbent - z_bound) / std::max(1.0, std::abs(z_incumbent));
}

/** Check whether a node LP bound is dominated by the incumbent (pruning). */
bool dominated(double lp_bound, double incumbent_obj, model::ObjectiveSense sense,
               double tol) noexcept {
    if (sense == model::ObjectiveSense::MINIMIZE) {
        return lp_bound >= incumbent_obj - tol;  // can't improve; prune
    }
    return lp_bound <= incumbent_obj + tol;  // can't improve; prune
}

/** Check whether all integer/binary variables in x satisfy integrality. */
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

/** Most-fractional branching variable selection.
 *  Score = min(frac, 1 - frac). Tie-break: ascending VarId (deterministic). */
std::optional<model::VarId> select_branching_variable(const model::CanonicalModel& model,
                                                      const std::vector<double>& x,
                                                      double int_tol) {
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
            continue;  // already integer
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

struct BranchSelectionResult {
    std::optional<model::VarId> var_id{std::nullopt};
    bool node_proved_infeasible{false};
};

/**
 * @brief Advanced branching variable selection.
 *
 * Implements:
 *  - Pseudocost branching with Achterberg product scoring
 *  - Strong branching for top-K candidates with child bound probes and infeasibility detection
 *  - Reliability branching hybrid policy
 *  - Deterministic tie-breaking hierarchy:
 *      1. Score (descending)
 *      2. Fractionality min(f, 1 - f) (descending)
 *      3. Total observations N^- + N^+ (descending)
 *      4. VarId (ascending)
 *  - Deterministic fallback chain: Reliability -> Pseudocost -> Strong -> Most-Fractional.
 */
BranchSelectionResult select_branching_variable_advanced(
    model::CanonicalModel& work_model, const model::CanonicalModel& orig_model,
    const std::vector<double>& x, double parent_obj, model::ObjectiveSense sense,
    const std::optional<lp::BasisState>& parent_basis,
    const std::function<lp::LPSolution(model::CanonicalModel&,
                                       const std::optional<lp::BasisState>&)>& solve_node_lp,
    PseudocostTracker& pseudocost_tracker, StrongBrancher& strong_brancher,
    const MILPSolverOptions& opts, MILPSolverTelemetry& tel) {
    const std::size_t n = std::min(x.size(), orig_model.num_variables());
    std::vector<BranchCandidate> candidates;

    for (std::size_t j = 0; j < n; ++j) {
        const auto& var = orig_model.variable(static_cast<model::VarId>(j));
        if (!var.is_integer() && !var.is_binary()) {
            continue;
        }
        double frac = x[j] - std::floor(x[j]);
        if (frac <= opts.tolerances.integrality_tol ||
            frac >= 1.0 - opts.tolerances.integrality_tol) {
            continue;
        }

        BranchCandidate cand;
        cand.var_id = static_cast<model::VarId>(j);
        cand.fractionality = std::min(frac, 1.0 - frac);
        cand.down_dist = frac;
        cand.up_dist = 1.0 - frac;

        const auto& p_entry = pseudocost_tracker.entry(cand.var_id);
        cand.down_pseudocost = p_entry.down_pseudocost;
        cand.up_pseudocost = p_entry.up_pseudocost;
        cand.down_obs = p_entry.down_count;
        cand.up_obs = p_entry.up_count;
        cand.is_reliable = p_entry.is_reliable(opts.pseudocost_reliability_threshold);
        cand.score = pseudocost_tracker.compute_score(cand.var_id, x[j]);

        candidates.push_back(cand);
    }

    if (candidates.empty()) {
        return {std::nullopt, false};
    }
    if (candidates.size() == 1) {
        return {candidates[0].var_id, false};
    }

    BranchingStrategy strategy = opts.branching_strategy;
    if (strategy == BranchingStrategy::AUTO) {
        strategy = BranchingStrategy::RELIABILITY;
    }

    if (strategy == BranchingStrategy::MOST_FRACTIONAL) {
        return {select_branching_variable(orig_model, x, opts.tolerances.integrality_tol), false};
    }

    if (strategy == BranchingStrategy::PSEUDOCOST) {
        tel.search_telemetry.pseudocost_candidates += candidates.size();
        std::stable_sort(candidates.begin(), candidates.end(),
                         [](const BranchCandidate& a, const BranchCandidate& b) {
                             if (std::abs(a.score - b.score) > 1e-9) {
                                 return a.score > b.score;
                             }
                             if (std::abs(a.fractionality - b.fractionality) > 1e-9) {
                                 return a.fractionality > b.fractionality;
                             }
                             std::size_t obs_a = a.down_obs + a.up_obs;
                             std::size_t obs_b = b.down_obs + b.up_obs;
                             if (obs_a != obs_b) {
                                 return obs_a > obs_b;
                             }
                             return a.var_id < b.var_id;
                         });
        return {candidates[0].var_id, false};
    }

    if (strategy == BranchingStrategy::STRONG) {
        std::stable_sort(candidates.begin(), candidates.end(),
                         [](const BranchCandidate& a, const BranchCandidate& b) {
                             if (std::abs(a.fractionality - b.fractionality) > 1e-9) {
                                 return a.fractionality > b.fractionality;
                             }
                             return a.var_id < b.var_id;
                         });

        std::size_t k = std::min(opts.strong_branching_candidates, candidates.size());
        std::vector<model::VarId> probe_vars;
        probe_vars.reserve(k);
        for (std::size_t i = 0; i < k; ++i) {
            probe_vars.push_back(candidates[i].var_id);
        }

        tel.search_telemetry.strong_branching_calls += 1;
        tel.search_telemetry.strong_branching_candidates += probe_vars.size();

        auto probes = strong_brancher.evaluate_candidates(
            work_model, probe_vars, x, parent_obj, sense, parent_basis, solve_node_lp,
            pseudocost_tracker, tel.search_telemetry.strong_branching_infeasible_children);

        for (const auto& pr : probes) {
            if (pr.node_proved_infeasible) {
                return {std::nullopt, true};
            }
        }

        if (probes.empty()) {
            tel.search_telemetry.reliability_fallbacks += 1;
            return {candidates[0].var_id, false};
        }

        auto best_probe_it = std::max_element(
            probes.begin(), probes.end(),
            [](const StrongBranchProbeResult& a, const StrongBranchProbeResult& b) {
                if (std::abs(a.score - b.score) > 1e-9) {
                    return a.score < b.score;
                }
                return a.var_id > b.var_id;
            });
        return {best_probe_it->var_id, false};
    }

    // RELIABILITY (Hybrid)
    tel.search_telemetry.pseudocost_candidates += candidates.size();

    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const BranchCandidate& a, const BranchCandidate& b) {
                         double eff_a = a.is_reliable ? a.score : (a.fractionality * 1000.0);
                         double eff_b = b.is_reliable ? b.score : (b.fractionality * 1000.0);
                         if (std::abs(eff_a - eff_b) > 1e-9) {
                             return eff_a > eff_b;
                         }
                         return a.var_id < b.var_id;
                     });

    std::vector<model::VarId> probe_vars;
    std::size_t k = std::min(opts.strong_branching_candidates, candidates.size());
    for (std::size_t i = 0; i < k; ++i) {
        if (!candidates[i].is_reliable) {
            probe_vars.push_back(candidates[i].var_id);
        }
    }

    if (!probe_vars.empty()) {
        tel.search_telemetry.strong_branching_calls += 1;
        tel.search_telemetry.strong_branching_candidates += probe_vars.size();

        auto probes = strong_brancher.evaluate_candidates(
            work_model, probe_vars, x, parent_obj, sense, parent_basis, solve_node_lp,
            pseudocost_tracker, tel.search_telemetry.strong_branching_infeasible_children);

        for (const auto& pr : probes) {
            if (pr.node_proved_infeasible) {
                return {std::nullopt, true};
            }
        }
    }

    // Re-score all candidates with updated pseudocosts
    for (auto& cand : candidates) {
        cand.score = pseudocost_tracker.compute_score(cand.var_id, x[cand.var_id]);
        const auto& p_entry = pseudocost_tracker.entry(cand.var_id);
        cand.down_obs = p_entry.down_count;
        cand.up_obs = p_entry.up_count;
        cand.is_reliable = p_entry.is_reliable(opts.pseudocost_reliability_threshold);
    }

    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const BranchCandidate& a, const BranchCandidate& b) {
                         if (std::abs(a.score - b.score) > 1e-9) {
                             return a.score > b.score;
                         }
                         if (std::abs(a.fractionality - b.fractionality) > 1e-9) {
                             return a.fractionality > b.fractionality;
                         }
                         std::size_t obs_a = a.down_obs + a.up_obs;
                         std::size_t obs_b = b.down_obs + b.up_obs;
                         if (obs_a != obs_b) {
                             return obs_a > obs_b;
                         }
                         return a.var_id < b.var_id;
                     });

    return {candidates[0].var_id, false};
}

/**
 * @brief Safe rounding heuristic — try rounding fractional integer vars.
 *
 * For each integer-fractional variable, round to nearest integer clamped to [lb, ub].
 * Strict independent feasibility check: reject if ANY constraint is violated.
 * Returns the rounded x if feasible, or empty vector on failure.
 */
std::vector<double> try_rounding(const model::CanonicalModel& model,
                                 const std::vector<double>& x_lp,
                                 const core::SolverTolerances& tols) {
    std::vector<double> x = x_lp;
    const std::size_t n = std::min(x.size(), model.num_variables());

    // Round all integer/binary variables
    for (std::size_t j = 0; j < n; ++j) {
        const auto& var = model.variable(static_cast<model::VarId>(j));
        if (var.is_integer() || var.is_binary()) {
            double rounded = std::round(x[j]);
            // Clamp to bounds
            rounded = std::max(var.lower_bound, std::min(var.upper_bound, rounded));
            x[j] = rounded;
        }
    }

    // Independent feasibility check — recompute Ax and check constraints
    for (std::size_t i = 0; i < model.num_constraints(); ++i) {
        const auto& constr = model.constraint(static_cast<model::ConstrId>(i));
        double activity = 0.0;
        for (const auto& term : constr.expression.terms()) {
            if (term.var_id < x.size()) {
                activity += term.coefficient * x[term.var_id];
            }
        }
        double rhs = constr.rhs;
        double viol = 0.0;
        switch (constr.sense) {
            case model::ConstraintSense::LESS_EQUAL:
                viol = std::max(0.0, activity - rhs);
                break;
            case model::ConstraintSense::GREATER_EQUAL:
                viol = std::max(0.0, rhs - activity);
                break;
            case model::ConstraintSense::EQUAL:
                viol = std::abs(activity - rhs);
                break;
        }
        if (viol > tols.primal_feasibility_tol) {
            return {};  // infeasible rounded solution
        }
    }

    // Check variable bounds
    for (std::size_t j = 0; j < n; ++j) {
        const auto& var = model.variable(static_cast<model::VarId>(j));
        if (var.has_lower_bound() && x[j] < var.lower_bound - tols.primal_feasibility_tol) {
            return {};
        }
        if (var.has_upper_bound() && x[j] > var.upper_bound + tols.primal_feasibility_tol) {
            return {};
        }
    }

    return x;
}

/** Recompute the objective value from scratch. */
double recompute_objective(const model::CanonicalModel& model,
                           const std::vector<double>& x) noexcept {
    double obj = model.objective_offset();
    for (const auto& term : model.objective().linear_terms.terms()) {
        if (term.var_id < x.size()) {
            obj += term.coefficient * x[term.var_id];
        }
    }
    return obj;
}

/** Check whether x_new is better than x_incumbent for the given sense. */
bool is_better(double new_obj, double inc_obj, model::ObjectiveSense sense) noexcept {
    if (sense == model::ObjectiveSense::MINIMIZE) {
        return new_obj < inc_obj;
    }
    return new_obj > inc_obj;
}

/** Initial incumbent bound value (worst possible). */
double worst_bound(model::ObjectiveSense sense) noexcept {
    if (sense == model::ObjectiveSense::MINIMIZE) {
        return std::numeric_limits<double>::infinity();
    }
    return -std::numeric_limits<double>::infinity();
}

/** Best bound across all open nodes (remaining dual bound). */
double best_remaining_bound(
    const std::priority_queue<BranchNode, std::vector<BranchNode>, MinBestBound>& pq,
    double incumbent_obj, model::ObjectiveSense /*sense*/) noexcept {
    if (pq.empty()) {
        return incumbent_obj;  // tree exhausted → bound = incumbent
    }
    return pq.top().lp_bound;
}

double best_remaining_bound_max(
    const std::priority_queue<BranchNode, std::vector<BranchNode>, MaxBestBound>& pq,
    double incumbent_obj, model::ObjectiveSense /*sense*/) noexcept {
    if (pq.empty()) {
        return incumbent_obj;
    }
    return pq.top().lp_bound;
}

// ============================================================================
// Core B&B solve
// ============================================================================

MILPSolution run_bb(const model::CanonicalModel& orig_model, const MILPSolverOptions& opts) {
    const auto start = Clock::now();
    const auto& tols = opts.tolerances;
    const model::ObjectiveSense sense = orig_model.objective_sense();
    const bool is_minimize = (sense == model::ObjectiveSense::MINIMIZE);

    MILPSolution result;
    MILPSolverTelemetry& tel = result.telemetry;

    // --- Working copy of the model for bound overlay ---
    model::CanonicalModel work_model = orig_model;

    // Setup LP sub-solver
    lp::LPSolverOptions lp_opts;
    lp_opts.tolerances = tols;
    lp_opts.enable_presolve = opts.enable_presolve;
    lp_opts.enable_scaling = opts.enable_scaling;
    lp_opts.preserve_bounds_structure = true;
    lp_opts.verbose = false;

    lp::RevisedSimplex lp_solver(lp_opts);

    auto solve_node_lp =
        [&](model::CanonicalModel& wm,
            const std::optional<lp::BasisState>& inherited_basis) -> lp::LPSolution {
        if (opts.enable_warm_starts && inherited_basis.has_value()) {
            ++tel.warm_start_attempts;
            auto ws_start = Clock::now();
            lp::LPSolverOptions ws_opts = lp_opts;
            ws_opts.enable_warm_start = true;
            ws_opts.preserve_bounds_structure = true;
            ws_opts.enable_presolve = false;

            auto ws_sol = lp_solver.solve_warm_start(wm, *inherited_basis, ws_opts);
            double ws_sec = std::chrono::duration<double>(Clock::now() - ws_start).count();
            tel.warm_start_time_seconds += ws_sec;
            tel.dual_simplex_iterations += ws_sol.telemetry.dual_simplex_iterations;

            if (ws_sol.is_optimal() || ws_sol.is_infeasible()) {
                ++tel.warm_start_successes;
                return ws_sol;
            }

            // Fallback to cold start
            ++tel.warm_start_fallbacks;
            tel.warm_start_fallback_reasons.push_back(ws_sol.message);
        }

        auto cs_start = Clock::now();
        lp::LPSolverOptions cs_opts = lp_opts;
        cs_opts.enable_presolve = false;
        cs_opts.preserve_bounds_structure = true;
        auto cs_sol = lp_solver.solve(wm, cs_opts);
        double cs_sec = std::chrono::duration<double>(Clock::now() - cs_start).count();
        tel.cold_start_time_seconds += cs_sec;
        tel.cold_start_iterations += cs_sol.iterations;
        return cs_sol;
    };

    // Incumbent tracking
    double incumbent_obj = worst_bound(sense);
    std::vector<double> incumbent;

    // Advanced Branch-and-Cut Search Subsystems
    PseudocostTracker pseudocost_tracker(orig_model.num_variables(), sense);
    StrongBrancher strong_brancher(tols);
    PrimalHeuristic primal_heuristic(tols, opts.heuristic_max_repair_rounds);
    tel.search_telemetry.branching_strategy_used = to_string(opts.branching_strategy);

    auto elapsed = [&]() -> double {
        auto now = Clock::now();
        return std::chrono::duration<double>(now - start).count();
    };

    auto time_limit_hit = [&]() -> bool {
        return opts.time_limit_seconds > 0.0 && elapsed() > opts.time_limit_seconds;
    };

    // Helper: Try rounding and update incumbent
    auto try_update_incumbent_via_rounding = [&](const std::vector<double>& x_lp) -> bool {
        auto rounded = try_rounding(orig_model, x_lp, tols);
        if (rounded.empty()) {
            ++tel.rounding_failures;
            return false;
        }
        // Verify integrality of rounded solution
        if (!is_integer_feasible(orig_model, rounded, tols.integrality_tol)) {
            ++tel.rounding_failures;
            return false;
        }
        double rounded_obj = recompute_objective(orig_model, rounded);
        if (!std::isfinite(rounded_obj)) {
            ++tel.rounding_failures;
            return false;
        }
        if (incumbent.empty() || is_better(rounded_obj, incumbent_obj, sense)) {
            incumbent_obj = rounded_obj;
            incumbent = rounded;
            ++tel.rounding_successes;
            ++tel.incumbent_updates;
            return true;
        }
        ++tel.rounding_failures;
        return false;
    };

    // -----------------------------------------------------------------------
    // 1. Solve root LP relaxation
    // -----------------------------------------------------------------------
    {
        // Make all variables continuous for LP relaxation
        for (std::size_t j = 0; j < work_model.num_variables(); ++j) {
            work_model.variable(static_cast<model::VarId>(j)).type =
                model::VariableType::CONTINUOUS;
        }
        auto root_sol = lp_solver.solve(work_model, lp_opts);
        // Restore types
        for (std::size_t j = 0; j < work_model.num_variables(); ++j) {
            work_model.variable(static_cast<model::VarId>(j)).type =
                orig_model.variable(static_cast<model::VarId>(j)).type;
        }

        if (root_sol.status == core::SolverStatus::INFEASIBLE) {
            result.status = core::SolverStatus::INFEASIBLE;
            result.message = "Root LP relaxation is infeasible — MILP is infeasible.";
            tel.solve_time_seconds = elapsed();
            return result;
        }
        if (root_sol.status == core::SolverStatus::UNBOUNDED) {
            result.status = core::SolverStatus::UNBOUNDED;
            result.message = "Root LP relaxation is unbounded.";
            tel.solve_time_seconds = elapsed();
            return result;
        }
        if (root_sol.status != core::SolverStatus::OPTIMAL) {
            result.status = core::SolverStatus::NUMERICAL_FAILURE;
            result.message = "Root LP relaxation failed: " + root_sol.message;
            tel.solve_time_seconds = elapsed();
            return result;
        }

        tel.root_lp_bound = root_sol.objective_value;
        tel.final_best_bound = root_sol.objective_value;

        // Check: if root LP solution is already integer-feasible
        if (is_integer_feasible(orig_model, root_sol.primal_solution, tols.integrality_tol)) {
            incumbent_obj = root_sol.objective_value;
            incumbent = root_sol.primal_solution;
            tel.incumbent_updates = 1;
            tel.final_best_bound = incumbent_obj;
            tel.search_telemetry.incumbent_source = "ROOT_LP";
            result.best_bound = incumbent_obj;
            result.best_objective = incumbent_obj;
            result.incumbent_solution = incumbent;
            result.optimality_gap = 0.0;
            tel.solve_time_seconds = elapsed();

            // Verify before reporting OPTIMAL
            verification::MILPVerifier verifier;
            auto vrpt = verifier.verify(orig_model, incumbent, incumbent_obj, incumbent_obj,
                                        core::SolverStatus::OPTIMAL, tols, 0.0);
            if (vrpt.certified_status == core::SolverStatus::OPTIMAL) {
                result.status = core::SolverStatus::OPTIMAL;
                result.message = "Root LP is integer feasible and verified.";
            } else {
                result.status = core::SolverStatus::NUMERICAL_FAILURE;
                result.message =
                    "Root LP appears integer-feasible but verification failed: " + vrpt.summary;
            }
            return result;
        }

        // Try primal heuristic on root
        if (opts.enable_primal_heuristic) {
            ++tel.search_telemetry.heuristic_calls;
            auto heur_res = primal_heuristic.search(orig_model, root_sol.primal_solution);
            if (heur_res.found_feasible) {
                ++tel.search_telemetry.heuristic_feasible_candidates;
                if (incumbent.empty() || is_better(heur_res.objective, incumbent_obj, sense)) {
                    incumbent_obj = heur_res.objective;
                    incumbent = heur_res.solution;
                    tel.search_telemetry.incumbent_source = "PRIMAL_HEURISTIC";
                    ++tel.search_telemetry.heuristic_incumbent_updates;
                    ++tel.incumbent_updates;
                }
            }
        }

        // Try fast rounding heuristic on root
        if (opts.enable_rounding_heuristic) {
            if (try_update_incumbent_via_rounding(root_sol.primal_solution)) {
                if (tel.search_telemetry.incumbent_source == "NONE") {
                    tel.search_telemetry.incumbent_source = "ROUNDING";
                }
            }
        }

        // -------------------------------------------------------------------
        // Root Cutting Plane Engine (Gomory + MIR)
        // -------------------------------------------------------------------
        if (opts.cut_options.strategy != CutStrategy::OFF) {
            CuttingPlaneEngine cut_engine(opts.cut_options);
            auto cut_res = cut_engine.run_root_cuts(work_model, lp_solver, lp_opts, root_sol,
                                                    tel.cut_telemetry, tols.integrality_tol);

            if (cut_res.infeasible) {
                result.status = core::SolverStatus::INFEASIBLE;
                result.message =
                    "Root LP became infeasible after valid cut separation — MILP is infeasible.";
                tel.solve_time_seconds = elapsed();
                return result;
            }

            if (cut_res.cuts_added > 0) {
                root_sol.objective_value = cut_res.objective;
                root_sol.primal_solution = cut_res.primal_solution;
                root_sol.basis = cut_res.basis;
                tel.root_lp_bound = root_sol.objective_value;
                tel.final_best_bound = root_sol.objective_value;

                if (cut_res.integer_feasible ||
                    is_integer_feasible(orig_model, root_sol.primal_solution,
                                        tols.integrality_tol)) {
                    incumbent_obj = root_sol.objective_value;
                    incumbent = root_sol.primal_solution;
                    tel.incumbent_updates = 1;
                    tel.final_best_bound = incumbent_obj;
                    result.best_bound = incumbent_obj;
                    result.best_objective = incumbent_obj;
                    result.incumbent_solution = incumbent;
                    result.optimality_gap = 0.0;
                    tel.solve_time_seconds = elapsed();

                    verification::MILPVerifier verifier;
                    auto vrpt = verifier.verify(orig_model, incumbent, incumbent_obj, incumbent_obj,
                                                core::SolverStatus::OPTIMAL, tols, 0.0);
                    if (vrpt.certified_status == core::SolverStatus::OPTIMAL) {
                        result.status = core::SolverStatus::OPTIMAL;
                        result.message =
                            "Root LP with cutting planes is integer feasible and verified.";
                    } else {
                        result.status = core::SolverStatus::NUMERICAL_FAILURE;
                        result.message =
                            "Root LP with cuts appears integer-feasible but verification failed: " +
                            vrpt.summary;
                    }
                    return result;
                }
            }
        }

        // Initialize B&B tree with root node
        if (is_minimize) {
            std::priority_queue<BranchNode, std::vector<BranchNode>, MinBestBound> pq;
            BranchNode root;
            root.node_id = 0;
            root.lp_bound = root_sol.objective_value;
            root.warm_start_basis = root_sol.basis;
            pq.push(std::move(root));

            std::uint64_t next_id = 1;

            while (!pq.empty()) {
                if (tel.nodes_explored >= opts.max_nodes) {
                    tel.node_limit_reached = true;
                    break;
                }
                if (time_limit_hit()) {
                    tel.time_limit_reached = true;
                    break;
                }

                BranchNode node = pq.top();
                pq.pop();

                // Bound pruning: node can't improve on incumbent
                if (!incumbent.empty() &&
                    dominated(node.lp_bound, incumbent_obj, sense, tols.primal_feasibility_tol)) {
                    ++tel.nodes_pruned_bound;
                    continue;
                }

                // Apply bound overlay
                apply_modifications(work_model, node.mods);

                // Relax integer types
                for (std::size_t j = 0; j < work_model.num_variables(); ++j) {
                    work_model.variable(static_cast<model::VarId>(j)).type =
                        model::VariableType::CONTINUOUS;
                }

                auto node_sol = solve_node_lp(work_model, node.warm_start_basis);

                // Restore types
                for (std::size_t j = 0; j < work_model.num_variables(); ++j) {
                    work_model.variable(static_cast<model::VarId>(j)).type =
                        orig_model.variable(static_cast<model::VarId>(j)).type;
                }

                // Restore bounds
                restore_modifications(work_model, node.mods);

                ++tel.nodes_explored;
                if (node_sol.status == core::SolverStatus::INFEASIBLE) {
                    ++tel.nodes_pruned_infeasible;
                    continue;
                }
                if (node_sol.status != core::SolverStatus::OPTIMAL) {
                    // Numerical issue at this node — skip
                    continue;
                }

                double node_obj = node_sol.objective_value;

                // Record observed degradation from parent branch into pseudocost tracker
                if (node.branch_var != model::INVALID_VAR_ID) {
                    if (node.is_down_branch) {
                        pseudocost_tracker.update_down(node.branch_var, node.parent_obj, node_obj,
                                                       node.fractionality, tols.integrality_tol);
                    } else {
                        pseudocost_tracker.update_up(node.branch_var, node.parent_obj, node_obj,
                                                     node.fractionality, tols.integrality_tol);
                    }
                }

                // Bound pruning after solve
                if (!incumbent.empty() &&
                    dominated(node_obj, incumbent_obj, sense, tols.primal_feasibility_tol)) {
                    ++tel.nodes_pruned_bound;
                    continue;
                }

                // Check integer feasibility
                if (is_integer_feasible(orig_model, node_sol.primal_solution,
                                        tols.integrality_tol)) {
                    if (incumbent.empty() || is_better(node_obj, incumbent_obj, sense)) {
                        incumbent_obj = node_obj;
                        incumbent = node_sol.primal_solution;
                        tel.search_telemetry.incumbent_source = "INTEGER_LP_NODE";
                        ++tel.incumbent_updates;
                    }
                    continue;  // This node is done
                }

                // Primal heuristic
                if (opts.enable_primal_heuristic &&
                    (tel.nodes_explored % opts.rounding_interval == 0)) {
                    ++tel.search_telemetry.heuristic_calls;
                    auto heur_res = primal_heuristic.search(orig_model, node_sol.primal_solution);
                    if (heur_res.found_feasible) {
                        ++tel.search_telemetry.heuristic_feasible_candidates;
                        if (incumbent.empty() ||
                            is_better(heur_res.objective, incumbent_obj, sense)) {
                            incumbent_obj = heur_res.objective;
                            incumbent = heur_res.solution;
                            tel.search_telemetry.incumbent_source = "PRIMAL_HEURISTIC";
                            ++tel.search_telemetry.heuristic_incumbent_updates;
                            ++tel.incumbent_updates;
                        }
                    }
                }

                // Rounding heuristic
                if (opts.enable_rounding_heuristic &&
                    (tel.nodes_explored % opts.rounding_interval == 0)) {
                    if (try_update_incumbent_via_rounding(node_sol.primal_solution)) {
                        if (tel.search_telemetry.incumbent_source == "NONE") {
                            tel.search_telemetry.incumbent_source = "ROUNDING";
                        }
                    }
                }

                // Branching variable selection (Advanced Search Engine)
                auto branch_decision = select_branching_variable_advanced(
                    work_model, orig_model, node_sol.primal_solution, node_obj, sense,
                    node_sol.basis, solve_node_lp, pseudocost_tracker, strong_brancher, opts, tel);

                if (branch_decision.node_proved_infeasible) {
                    ++tel.nodes_pruned_infeasible;
                    continue;
                }

                if (!branch_decision.var_id.has_value()) {
                    continue;  // No fractional variable found or integer feasible
                }

                model::VarId bv = *branch_decision.var_id;
                double bv_val = node_sol.primal_solution[bv];
                double bv_floor = std::floor(bv_val);
                double bv_ceil = std::ceil(bv_val);
                const auto& base_var = orig_model.variable(bv);

                // LEFT child: x_j <= floor(x_j)
                double left_ub = bv_floor;
                double left_lb = base_var.lower_bound;
                // Apply any existing mods from parent
                for (const auto& m : node.mods) {
                    if (m.var_id == bv) {
                        if (m.is_lower) {
                            left_lb = std::max(left_lb, m.new_bound);
                        } else {
                            left_ub = std::min(left_ub, m.new_bound);
                        }
                    }
                }

                if (left_lb <= left_ub + tols.primal_feasibility_tol) {
                    BranchNode left;
                    left.node_id = next_id++;
                    left.lp_bound = node_obj;
                    left.mods = node.mods;
                    left.branch_var = bv;
                    left.is_down_branch = true;
                    left.parent_obj = node_obj;
                    left.fractionality = bv_val;
                    left.depth = node.depth + 1;
                    // Find and update existing UB mod if any, else add new
                    bool found_ub = false;
                    for (auto& m : left.mods) {
                        if (m.var_id == bv && !m.is_lower) {
                            m.old_bound = orig_model.variable(bv).upper_bound;
                            m.new_bound = left_ub;
                            found_ub = true;
                            break;
                        }
                    }
                    if (!found_ub) {
                        left.mods.push_back(
                            {bv, false, left_ub, orig_model.variable(bv).upper_bound});
                    }
                    left.warm_start_basis = node_sol.basis;
                    pq.push(std::move(left));
                    ++tel.basis_transfer_count;
                }

                // RIGHT child: x_j >= ceil(x_j)
                double right_lb = bv_ceil;
                double right_ub = base_var.upper_bound;
                for (const auto& m : node.mods) {
                    if (m.var_id == bv) {
                        if (m.is_lower) {
                            right_lb = std::max(right_lb, m.new_bound);
                        } else {
                            right_ub = std::min(right_ub, m.new_bound);
                        }
                    }
                }

                if (right_lb <= right_ub + tols.primal_feasibility_tol) {
                    BranchNode right;
                    right.node_id = next_id++;
                    right.lp_bound = node_obj;
                    right.mods = node.mods;
                    right.branch_var = bv;
                    right.is_down_branch = false;
                    right.parent_obj = node_obj;
                    right.fractionality = bv_val;
                    right.depth = node.depth + 1;
                    bool found_lb = false;
                    for (auto& m : right.mods) {
                        if (m.var_id == bv && m.is_lower) {
                            m.old_bound = orig_model.variable(bv).lower_bound;
                            m.new_bound = right_lb;
                            found_lb = true;
                            break;
                        }
                    }
                    if (!found_lb) {
                        right.mods.push_back(
                            {bv, true, right_lb, orig_model.variable(bv).lower_bound});
                    }
                    right.warm_start_basis = node_sol.basis;
                    pq.push(std::move(right));
                    ++tel.basis_transfer_count;
                }
            }  // end while

            // Compute final best bound
            double final_bound = best_remaining_bound(pq, incumbent_obj, sense);
            tel.final_best_bound = final_bound;

        } else {
            // MAXIMIZE — same loop with MaxBestBound comparator
            std::priority_queue<BranchNode, std::vector<BranchNode>, MaxBestBound> pq;
            BranchNode root;
            root.node_id = 0;
            root.lp_bound = root_sol.objective_value;
            root.warm_start_basis = root_sol.basis;
            pq.push(std::move(root));

            std::uint64_t next_id = 1;

            while (!pq.empty()) {
                if (tel.nodes_explored >= opts.max_nodes) {
                    tel.node_limit_reached = true;
                    break;
                }
                if (time_limit_hit()) {
                    tel.time_limit_reached = true;
                    break;
                }

                BranchNode node = pq.top();
                pq.pop();

                if (!incumbent.empty() &&
                    dominated(node.lp_bound, incumbent_obj, sense, tols.primal_feasibility_tol)) {
                    ++tel.nodes_pruned_bound;
                    continue;
                }

                apply_modifications(work_model, node.mods);

                for (std::size_t j = 0; j < work_model.num_variables(); ++j) {
                    work_model.variable(static_cast<model::VarId>(j)).type =
                        model::VariableType::CONTINUOUS;
                }
                auto node_sol = solve_node_lp(work_model, node.warm_start_basis);
                for (std::size_t j = 0; j < work_model.num_variables(); ++j) {
                    work_model.variable(static_cast<model::VarId>(j)).type =
                        orig_model.variable(static_cast<model::VarId>(j)).type;
                }
                restore_modifications(work_model, node.mods);

                ++tel.nodes_explored;

                if (node_sol.status == core::SolverStatus::INFEASIBLE) {
                    ++tel.nodes_pruned_infeasible;
                    continue;
                }
                if (node_sol.status != core::SolverStatus::OPTIMAL) {
                    continue;
                }

                double node_obj = node_sol.objective_value;

                // Record observed degradation from parent branch into pseudocost tracker
                if (node.branch_var != model::INVALID_VAR_ID) {
                    if (node.is_down_branch) {
                        pseudocost_tracker.update_down(node.branch_var, node.parent_obj, node_obj,
                                                       node.fractionality, tols.integrality_tol);
                    } else {
                        pseudocost_tracker.update_up(node.branch_var, node.parent_obj, node_obj,
                                                     node.fractionality, tols.integrality_tol);
                    }
                }

                if (!incumbent.empty() &&
                    dominated(node_obj, incumbent_obj, sense, tols.primal_feasibility_tol)) {
                    ++tel.nodes_pruned_bound;
                    continue;
                }

                if (is_integer_feasible(orig_model, node_sol.primal_solution,
                                        tols.integrality_tol)) {
                    if (incumbent.empty() || is_better(node_obj, incumbent_obj, sense)) {
                        incumbent_obj = node_obj;
                        incumbent = node_sol.primal_solution;
                        tel.search_telemetry.incumbent_source = "INTEGER_LP_NODE";
                        ++tel.incumbent_updates;
                    }
                    continue;
                }

                // Primal heuristic
                if (opts.enable_primal_heuristic &&
                    (tel.nodes_explored % opts.rounding_interval == 0)) {
                    ++tel.search_telemetry.heuristic_calls;
                    auto heur_res = primal_heuristic.search(orig_model, node_sol.primal_solution);
                    if (heur_res.found_feasible) {
                        ++tel.search_telemetry.heuristic_feasible_candidates;
                        if (incumbent.empty() ||
                            is_better(heur_res.objective, incumbent_obj, sense)) {
                            incumbent_obj = heur_res.objective;
                            incumbent = heur_res.solution;
                            tel.search_telemetry.incumbent_source = "PRIMAL_HEURISTIC";
                            ++tel.search_telemetry.heuristic_incumbent_updates;
                            ++tel.incumbent_updates;
                        }
                    }
                }

                if (opts.enable_rounding_heuristic &&
                    (tel.nodes_explored % opts.rounding_interval == 0)) {
                    if (try_update_incumbent_via_rounding(node_sol.primal_solution)) {
                        if (tel.search_telemetry.incumbent_source == "NONE") {
                            tel.search_telemetry.incumbent_source = "ROUNDING";
                        }
                    }
                }

                // Branching variable selection (Advanced Search Engine)
                auto branch_decision = select_branching_variable_advanced(
                    work_model, orig_model, node_sol.primal_solution, node_obj, sense,
                    node_sol.basis, solve_node_lp, pseudocost_tracker, strong_brancher, opts, tel);

                if (branch_decision.node_proved_infeasible) {
                    ++tel.nodes_pruned_infeasible;
                    continue;
                }

                if (!branch_decision.var_id.has_value()) {
                    continue;
                }

                model::VarId bv = *branch_decision.var_id;
                double bv_val = node_sol.primal_solution[bv];
                double bv_floor = std::floor(bv_val);
                double bv_ceil = std::ceil(bv_val);
                const auto& base_var = orig_model.variable(bv);

                double left_ub = bv_floor;
                double left_lb = base_var.lower_bound;
                for (const auto& m : node.mods) {
                    if (m.var_id == bv) {
                        if (m.is_lower)
                            left_lb = std::max(left_lb, m.new_bound);
                        else
                            left_ub = std::min(left_ub, m.new_bound);
                    }
                }

                if (left_lb <= left_ub + tols.primal_feasibility_tol) {
                    BranchNode left;
                    left.node_id = next_id++;
                    left.lp_bound = node_obj;
                    left.mods = node.mods;
                    left.branch_var = bv;
                    left.is_down_branch = true;
                    left.parent_obj = node_obj;
                    left.fractionality = bv_val;
                    left.depth = node.depth + 1;
                    bool found = false;
                    for (auto& m : left.mods) {
                        if (m.var_id == bv && !m.is_lower) {
                            m.new_bound = left_ub;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        left.mods.push_back(
                            {bv, false, left_ub, orig_model.variable(bv).upper_bound});
                    }
                    left.warm_start_basis = node_sol.basis;
                    pq.push(std::move(left));
                    ++tel.basis_transfer_count;
                }

                double right_lb = bv_ceil;
                double right_ub = base_var.upper_bound;
                for (const auto& m : node.mods) {
                    if (m.var_id == bv) {
                        if (m.is_lower)
                            right_lb = std::max(right_lb, m.new_bound);
                        else
                            right_ub = std::min(right_ub, m.new_bound);
                    }
                }

                if (right_lb <= right_ub + tols.primal_feasibility_tol) {
                    BranchNode right;
                    right.node_id = next_id++;
                    right.lp_bound = node_obj;
                    right.mods = node.mods;
                    right.branch_var = bv;
                    right.is_down_branch = false;
                    right.parent_obj = node_obj;
                    right.fractionality = bv_val;
                    right.depth = node.depth + 1;
                    bool found = false;
                    for (auto& m : right.mods) {
                        if (m.var_id == bv && m.is_lower) {
                            m.new_bound = right_lb;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        right.mods.push_back(
                            {bv, true, right_lb, orig_model.variable(bv).lower_bound});
                    }
                    right.warm_start_basis = node_sol.basis;
                    pq.push(std::move(right));
                    ++tel.basis_transfer_count;
                }
            }  // end while

            double final_bound = best_remaining_bound_max(pq, incumbent_obj, sense);
            tel.final_best_bound = final_bound;
        }
    }

    tel.solve_time_seconds = elapsed();
    tel.search_telemetry.pseudocost_updates = pseudocost_tracker.total_updates();

    // Construct search telemetry explanation
    {
        std::string expl = "Branching strategy: " + tel.search_telemetry.branching_strategy_used;
        expl += "; pseudocost updates: " + std::to_string(tel.search_telemetry.pseudocost_updates);
        expl += "; strong branching calls: " +
                std::to_string(tel.search_telemetry.strong_branching_calls);
        expl += " (probed " + std::to_string(tel.search_telemetry.strong_branching_candidates) +
                " candidates, pruned " +
                std::to_string(tel.search_telemetry.strong_branching_infeasible_children) +
                " infeasible children)";
        expl += "; heuristic calls: " + std::to_string(tel.search_telemetry.heuristic_calls) +
                " (feasible found: " +
                std::to_string(tel.search_telemetry.heuristic_feasible_candidates) +
                ", incumbent updates: " +
                std::to_string(tel.search_telemetry.heuristic_incumbent_updates) + ")";
        expl += "; incumbent source: " + tel.search_telemetry.incumbent_source;
        tel.search_telemetry.search_explanation = std::move(expl);
    }

    // -----------------------------------------------------------------------
    // 2. Final status determination
    // -----------------------------------------------------------------------

    if (incumbent.empty()) {
        // No feasible integer solution found
        if (tel.node_limit_reached || tel.time_limit_reached) {
            result.status = core::SolverStatus::LIMIT_REACHED;
            result.best_bound = tel.final_best_bound;
            result.optimality_gap = 1.0;
            result.message = tel.node_limit_reached
                                 ? "Node limit reached with no integer-feasible solution found."
                                 : "Time limit reached with no integer-feasible solution found.";
        } else {
            result.status = core::SolverStatus::INFEASIBLE;
            result.message = "Branch-and-bound exhausted search tree. Problem is infeasible.";
        }
        return result;
    }

    double final_gap = compute_gap(incumbent_obj, tel.final_best_bound);
    result.best_objective = incumbent_obj;
    result.best_bound = tel.final_best_bound;
    result.optimality_gap = final_gap;
    result.incumbent_solution = incumbent;

    if (tel.node_limit_reached || tel.time_limit_reached) {
        result.status = core::SolverStatus::FEASIBLE;
        result.message =
            std::string(tel.node_limit_reached ? "Node" : "Time") +
            " limit reached. Best incumbent objective: " + std::to_string(incumbent_obj) +
            ", gap: " + std::to_string(final_gap);
        return result;
    }

    // Tree exhausted with incumbent: check gap
    const double gap_tol = 1e-4;
    if (final_gap <= gap_tol) {
        // Attempt independent verification before reporting OPTIMAL
        verification::MILPVerifier verifier;
        auto vrpt = verifier.verify(orig_model, incumbent, tel.final_best_bound, incumbent_obj,
                                    core::SolverStatus::OPTIMAL, opts.tolerances, final_gap);
        if (vrpt.certified_status == core::SolverStatus::OPTIMAL) {
            result.status = core::SolverStatus::OPTIMAL;
            result.optimality_gap = vrpt.recomputed_gap;
            result.message = "OPTIMAL certified by independent verification. " + vrpt.summary;
        } else {
            // Verification downgraded status
            result.status = vrpt.certified_status;
            result.message = "Verification downgraded status: " + vrpt.summary;
        }
    } else {
        result.status = core::SolverStatus::FEASIBLE;
        result.message = "B&B complete. Gap " + std::to_string(final_gap) +
                         " exceeds tolerance — reporting FEASIBLE.";
    }

    return result;
}

}  // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

MILPSolution BranchAndBound::solve(const model::CanonicalModel& model) {
    return solve(model, options_);
}

MILPSolution BranchAndBound::solve(const model::CanonicalModel& model,
                                   const core::SolverTolerances& tolerances) {
    MILPSolverOptions opts = options_;
    opts.tolerances = tolerances;
    return solve(model, opts);
}

MILPSolution BranchAndBound::solve(const model::CanonicalModel& model,
                                   const MILPSolverOptions& options) {
    // Reject QP models
    if (model.is_qp()) {
        MILPSolution sol;
        sol.status = core::SolverStatus::NUMERICAL_FAILURE;
        sol.message = "MILP solver received a model with quadratic objective terms. Rejected.";
        return sol;
    }

    // Validate model
    auto val = model.validate();
    if (!val.valid) {
        MILPSolution sol;
        sol.status = core::SolverStatus::UNKNOWN;
        std::string msg = "Model validation failed:";
        for (const auto& e : val.errors) {
            msg += " [" + e.component + ": " + e.message + "]";
        }
        sol.message = msg;
        return sol;
    }

    // Pure LP: no integer/binary variables — route directly to LP solver
    if (model.is_lp()) {
        lp::LPSolverOptions lp_opts;
        lp_opts.tolerances = options.tolerances;
        lp_opts.enable_presolve = options.enable_presolve;
        lp_opts.enable_scaling = options.enable_scaling;
        lp::RevisedSimplex lp_solver(lp_opts);
        auto lp_sol = lp_solver.solve(model, lp_opts);

        MILPSolution sol;
        sol.status = lp_sol.status;
        sol.best_objective = lp_sol.objective_value;
        sol.best_bound = lp_sol.objective_value;
        sol.optimality_gap = (lp_sol.status == core::SolverStatus::OPTIMAL) ? 0.0 : 1.0;
        sol.incumbent_solution = lp_sol.primal_solution;
        sol.message = "Pure LP (no integer variables) — routed to LP solver. " + lp_sol.message;
        sol.telemetry.solve_time_seconds = lp_sol.telemetry.solve_time_seconds;
        return sol;
    }

    return run_bb(model, options);
}

}  // namespace kavach::algorithms::milp
