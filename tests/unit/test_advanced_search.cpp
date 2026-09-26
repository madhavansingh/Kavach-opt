/**
 * @file test_advanced_search.cpp
 * @brief Comprehensive test suite for the Advanced MILP Search Engine:
 *        Pseudocosts, Strong Branching, Reliability Branching, Primal Heuristics,
 *        and Deterministic Node Prioritization.
 */

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "../test_framework.hpp"
#include "algorithms/lp/simplex.hpp"
#include "algorithms/milp/branch_and_bound.hpp"
#include "algorithms/milp/primal_heuristic.hpp"
#include "algorithms/milp/pseudocost.hpp"
#include "algorithms/milp/search_types.hpp"
#include "algorithms/milp/strong_branching.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"
#include "verification/milp_verifier.hpp"

using namespace kavach;
using namespace kavach::model;
using namespace kavach::algorithms::lp;
using namespace kavach::algorithms::milp;
using namespace kavach::verification;

namespace {

CanonicalModel make_knapsack_6var() {
    CanonicalModel m("knapsack_6var");
    m.set_objective_sense(ObjectiveSense::MAXIMIZE);

    const double vals[6] = {10.0, 13.0, 18.0, 31.0, 7.0, 15.0};
    const double weights[6] = {11.0, 15.0, 20.0, 35.0, 10.0, 18.0};

    std::vector<VarId> vars;
    for (int j = 0; j < 6; ++j) {
        auto v = m.add_binary_variable("x" + std::to_string(j));
        m.add_objective_linear_term(v, vals[j]);
        vars.push_back(v);
    }

    SparseLinearExpression expr;
    for (int j = 0; j < 6; ++j) {
        expr.add_term(vars[static_cast<std::size_t>(j)], weights[j]);
    }
    m.add_constraint("capacity", std::move(expr), ConstraintSense::LESS_EQUAL, 47.0);

    return m;
}

CanonicalModel make_facility_location() {
    CanonicalModel m("facility_loc");
    m.set_objective_sense(ObjectiveSense::MINIMIZE);

    // 3 facilities (binary y0, y1, y2, fixed cost 50, 60, 70)
    std::vector<VarId> y;
    for (int i = 0; i < 3; ++i) {
        auto v = m.add_binary_variable("y" + std::to_string(i));
        m.add_objective_linear_term(v, 50.0 + i * 10.0);
        y.push_back(v);
    }

    // 2 customers served by facilities (continuous x_ij, j in {0,1}, i in {0,1,2})
    double cost[3][2] = {{30.0, 40.0}, {25.0, 35.0}, {40.0, 20.0}};
    std::vector<std::vector<VarId>> x(3, std::vector<VarId>(2));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            auto v = m.add_variable("x" + std::to_string(i) + "_" + std::to_string(j),
                                    VariableType::CONTINUOUS, 0.0, 1.0);
            m.add_objective_linear_term(v, cost[i][j]);
            x[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = v;
        }
    }

    // Demand constraints: sum_i x_ij == 1
    for (int j = 0; j < 2; ++j) {
        SparseLinearExpression dem;
        for (int i = 0; i < 3; ++i) {
            dem.add_term(x[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)], 1.0);
        }
        m.add_constraint("dem_" + std::to_string(j), std::move(dem), ConstraintSense::EQUAL, 1.0);
    }

    // Linking constraints: x_ij - y_i <= 0
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            SparseLinearExpression link;
            link.add_term(x[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)], 1.0);
            link.add_term(y[static_cast<std::size_t>(i)], -1.0);
            m.add_constraint("link_" + std::to_string(i) + "_" + std::to_string(j), std::move(link),
                             ConstraintSense::LESS_EQUAL, 0.0);
        }
    }

    return m;
}

}  // anonymous namespace

// ============================================================================
// 1. Pseudocost Tracker Tests
// ============================================================================

KAVACH_TEST(pseudocost_initialization_and_priors) {
    PseudocostTracker tracker(10, ObjectiveSense::MINIMIZE, 2.5);
    KAVACH_ASSERT_EQUAL(tracker.total_updates(), 0);

    const auto& entry = tracker.entry(3);
    KAVACH_ASSERT_NEAR(entry.down_pseudocost, 2.5, 1e-9);
    KAVACH_ASSERT_NEAR(entry.up_pseudocost, 2.5, 1e-9);
    KAVACH_ASSERT_EQUAL(entry.down_count, 0);
    KAVACH_ASSERT_EQUAL(entry.up_count, 0);
    KAVACH_ASSERT(!entry.is_reliable(4));

    tracker.reset(5, ObjectiveSense::MAXIMIZE, 1.0);
    KAVACH_ASSERT_EQUAL(tracker.total_updates(), 0);
    KAVACH_ASSERT_NEAR(tracker.entry(4).down_pseudocost, 1.0, 1e-9);
}

KAVACH_TEST(pseudocost_updates_and_scoring) {
    PseudocostTracker tracker(5, ObjectiveSense::MINIMIZE, 1.0);

    // Parent obj = 100.0, child obj = 106.0, var val = 2.4 (f = 0.4)
    // Down branch delta = 6.0, unit deg = 6.0 / 0.4 = 15.0
    bool ok_down = tracker.update_down(1, 100.0, 106.0, 2.4, 1e-5);
    KAVACH_ASSERT(ok_down);
    KAVACH_ASSERT_EQUAL(tracker.entry(1).down_count, 1);
    KAVACH_ASSERT_NEAR(tracker.entry(1).down_pseudocost, 15.0, 1e-9);

    // Up branch: var val = 2.4 (1 - f = 0.6), parent = 100.0, child = 112.0
    // Up branch delta = 12.0, unit deg = 12.0 / 0.6 = 20.0
    bool ok_up = tracker.update_up(1, 100.0, 112.0, 2.4, 1e-5);
    KAVACH_ASSERT(ok_up);
    KAVACH_ASSERT_EQUAL(tracker.entry(1).up_count, 1);
    KAVACH_ASSERT_NEAR(tracker.entry(1).up_pseudocost, 20.0, 1e-9);

    // Compute Achterberg score for x = 2.4:
    // q_down = 15.0 * 0.4 = 6.0
    // q_up = 20.0 * 0.6 = 12.0
    // Achterberg product score: max(q_down, 1e-6) * max(q_up, 1e-6) = 6.0 * 12.0 = 72.0
    double score = tracker.compute_score(1, 2.4);
    KAVACH_ASSERT_NEAR(score, 72.0, 1e-3);
}

KAVACH_TEST(pseudocost_reliability_threshold) {
    PseudocostTracker tracker(3, ObjectiveSense::MINIMIZE, 1.0);
    const std::size_t threshold = 3;

    KAVACH_ASSERT(!tracker.is_reliable(0, threshold));

    // Add 3 down updates
    tracker.update_down(0, 10.0, 12.0, 0.5, 1e-5);
    tracker.update_down(0, 10.0, 13.0, 0.5, 1e-5);
    tracker.update_down(0, 10.0, 14.0, 0.5, 1e-5);
    KAVACH_ASSERT(!tracker.is_reliable(0, threshold));  // up still 0

    // Add 2 up updates
    tracker.update_up(0, 10.0, 11.0, 0.5, 1e-5);
    tracker.update_up(0, 10.0, 12.0, 0.5, 1e-5);
    KAVACH_ASSERT(!tracker.is_reliable(0, threshold));  // up is 2 < 3

    // Add 3rd up update
    tracker.update_up(0, 10.0, 13.0, 0.5, 1e-5);
    KAVACH_ASSERT(tracker.is_reliable(0, threshold));  // both down >= 3 and up >= 3
}

KAVACH_TEST(pseudocost_float_safeguards) {
    PseudocostTracker tracker(2, ObjectiveSense::MINIMIZE, 1.0);

    // Integral value (f <= 1e-5) -> rejected safely
    bool res1 = tracker.update_down(0, 10.0, 12.0, 3.0000001, 1e-5);
    KAVACH_ASSERT(!res1);
    KAVACH_ASSERT_EQUAL(tracker.entry(0).down_count, 0);

    // Negative delta (float noise making child obj slightly better than parent) -> clamped to 0
    bool res2 = tracker.update_down(0, 100.0, 99.9999, 2.5, 1e-5);
    KAVACH_ASSERT(res2);
    KAVACH_ASSERT_NEAR(tracker.entry(0).down_pseudocost, 0.0, 1e-9);
}

// ============================================================================
// 2. Strong Branching Tests
// ============================================================================

KAVACH_TEST(strong_branching_probes_and_isolation) {
    auto model = make_knapsack_6var();
    auto lp_model = model;
    for (std::size_t j = 0; j < lp_model.num_variables(); ++j) {
        lp_model.variable(static_cast<VarId>(j)).type = VariableType::CONTINUOUS;
    }

    core::SolverTolerances tols;
    StrongBrancher strong_brancher(tols);
    PseudocostTracker tracker(lp_model.num_variables(), ObjectiveSense::MAXIMIZE);

    // Solve root relaxation
    LPSolverOptions lp_opts;
    lp_opts.enable_presolve = false;
    RevisedSimplex simplex(lp_opts);
    auto lp_sol = simplex.solve(lp_model);
    KAVACH_ASSERT_EQUAL(static_cast<int>(lp_sol.status),
                        static_cast<int>(core::SolverStatus::OPTIMAL));

    // Record original model bounds
    std::vector<std::pair<double, double>> orig_bounds;
    for (std::size_t j = 0; j < lp_model.num_variables(); ++j) {
        orig_bounds.emplace_back(lp_model.variable(static_cast<VarId>(j)).lower_bound,
                                 lp_model.variable(static_cast<VarId>(j)).upper_bound);
    }

    auto solve_lp_func = [&](CanonicalModel& m,
                             const std::optional<BasisState>& basis) -> LPSolution {
        LPSolverOptions opts;
        opts.enable_presolve = false;
        opts.tolerances = tols;
        RevisedSimplex solver(opts);
        if (basis.has_value()) {
            return solver.solve_warm_start(m, *basis, opts);
        }
        return solver.solve(m, opts);
    };

    std::vector<VarId> probe_vars;
    for (std::size_t j = 0; j < lp_model.num_variables(); ++j) {
        double val = lp_sol.primal_solution[j];
        double f = val - std::floor(val);
        if (f > tols.integrality_tol && f < 1.0 - tols.integrality_tol) {
            probe_vars.push_back(static_cast<VarId>(j));
        }
    }
    std::size_t infeas_count = 0;
    auto probes = strong_brancher.evaluate_candidates(
        lp_model, probe_vars, lp_sol.primal_solution, lp_sol.objective_value,
        ObjectiveSense::MAXIMIZE, lp_sol.basis, solve_lp_func, tracker, infeas_count);

    // Verify isolation: model bounds MUST be completely unchanged after probing
    for (std::size_t j = 0; j < lp_model.num_variables(); ++j) {
        KAVACH_ASSERT_NEAR(lp_model.variable(static_cast<VarId>(j)).lower_bound,
                           orig_bounds[j].first, 1e-12);
        KAVACH_ASSERT_NEAR(lp_model.variable(static_cast<VarId>(j)).upper_bound,
                           orig_bounds[j].second, 1e-12);
    }

    KAVACH_ASSERT(!probes.empty());
    for (const auto& pr : probes) {
        KAVACH_ASSERT(!pr.node_proved_infeasible);
        KAVACH_ASSERT(pr.score >= 0.0);
    }
}

// ============================================================================
// 3. Primal Heuristic Tests
// ============================================================================

KAVACH_TEST(primal_heuristic_rounding_and_repair) {
    auto model = make_knapsack_6var();
    core::SolverTolerances tols;
    PrimalHeuristic heuristic(tols, 5);

    // Fractional LP solution that slightly exceeds knapsack capacity if rounded up
    // Weights: [11, 15, 20, 35, 10, 18], Capacity = 47
    // Fractional x = [1, 1, 1, 0.3, 0, 0] -> sum = 11 + 15 + 20 + 0.3*35 = 46 + 10.5 = 56.5
    // (violates 47)
    std::vector<double> frac_sol = {1.0, 1.0, 1.0, 0.3, 0.0, 0.0};

    auto res = heuristic.search(model, frac_sol);
    if (res.found_feasible) {
        // Solution must strictly satisfy knapsack capacity <= 47
        double total_weight = 0.0;
        const double weights[6] = {11.0, 15.0, 20.0, 35.0, 10.0, 18.0};
        for (int j = 0; j < 6; ++j) {
            double v = res.solution[static_cast<std::size_t>(j)];
            KAVACH_ASSERT(std::abs(v - std::round(v)) <= tols.integrality_tol);
            total_weight += v * weights[j];
        }
        KAVACH_ASSERT(total_weight <= 47.0 + tols.primal_feasibility_tol);

        // Verify with independent MILPVerifier
        MILPVerifier verifier;
        auto vrep = verifier.verify(model, res.solution, res.objective, res.objective,
                                    core::SolverStatus::FEASIBLE, tols, 0.0);
        KAVACH_ASSERT(vrep.certified_status == core::SolverStatus::FEASIBLE ||
                      vrep.certified_status == core::SolverStatus::OPTIMAL);
    }
}

// ============================================================================
// 4. Advanced B&B Search Engine Integration Tests
// ============================================================================

KAVACH_TEST(search_strategies_solve_knapsack) {
    auto model = make_knapsack_6var();

    std::vector<BranchingStrategy> strategies = {
        BranchingStrategy::MOST_FRACTIONAL, BranchingStrategy::PSEUDOCOST,
        BranchingStrategy::STRONG, BranchingStrategy::RELIABILITY};

    for (auto strat : strategies) {
        MILPSolverOptions opts;
        opts.branching_strategy = strat;
        opts.enable_primal_heuristic = true;
        opts.cut_options.strategy = CutStrategy::ROOT_ONLY;

        BranchAndBound solver(opts);
        auto sol = solver.solve(model);

        KAVACH_ASSERT_EQUAL(static_cast<int>(sol.status),
                            static_cast<int>(core::SolverStatus::OPTIMAL));
        KAVACH_ASSERT_NEAR(sol.best_objective, 41.0, 1e-5);
        KAVACH_ASSERT(sol.optimality_gap <= 1e-4);

        // Check telemetry
        KAVACH_ASSERT(sol.telemetry.search_telemetry.branching_strategy_used == to_string(strat));
        KAVACH_ASSERT(!sol.telemetry.search_telemetry.search_explanation.empty());

        if (strat == BranchingStrategy::STRONG) {
            KAVACH_ASSERT(sol.telemetry.search_telemetry.strong_branching_calls > 0);
        }
    }
}

KAVACH_TEST(search_strategies_solve_facility_location) {
    auto model = make_facility_location();

    MILPSolverOptions opts;
    opts.branching_strategy = BranchingStrategy::RELIABILITY;
    opts.enable_primal_heuristic = true;

    BranchAndBound solver(opts);
    auto sol = solver.solve(model);

    KAVACH_ASSERT_EQUAL(static_cast<int>(sol.status),
                        static_cast<int>(core::SolverStatus::OPTIMAL));
    KAVACH_ASSERT_NEAR(sol.best_objective, 120.0, 1e-4);

    MILPVerifier verifier;
    auto vrpt = verifier.verify(model, sol.incumbent_solution, sol.best_bound, sol.best_objective,
                                core::SolverStatus::OPTIMAL, opts.tolerances, sol.optimality_gap);
    KAVACH_ASSERT_EQUAL(static_cast<int>(vrpt.certified_status),
                        static_cast<int>(core::SolverStatus::OPTIMAL));
}

KAVACH_TEST(search_determinism_repeated_runs) {
    auto model = make_knapsack_6var();

    MILPSolverOptions opts;
    opts.branching_strategy = BranchingStrategy::RELIABILITY;
    opts.enable_primal_heuristic = true;

    BranchAndBound solver(opts);
    auto baseline_sol = solver.solve(model);

    for (int run = 0; run < 10; ++run) {
        auto sol = solver.solve(model);
        KAVACH_ASSERT_EQUAL(static_cast<int>(sol.status), static_cast<int>(baseline_sol.status));
        KAVACH_ASSERT_NEAR(sol.best_objective, baseline_sol.best_objective, 1e-12);
        KAVACH_ASSERT_EQUAL(sol.telemetry.nodes_explored, baseline_sol.telemetry.nodes_explored);
        KAVACH_ASSERT_EQUAL(sol.telemetry.search_telemetry.pseudocost_updates,
                            baseline_sol.telemetry.search_telemetry.pseudocost_updates);
        KAVACH_ASSERT_EQUAL(sol.telemetry.search_telemetry.strong_branching_calls,
                            baseline_sol.telemetry.search_telemetry.strong_branching_calls);
    }
}
