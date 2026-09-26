/**
 * @file test_cutting_planes.cpp
 * @brief Comprehensive test suite for the Cutting-Plane Subsystem (Gomory & MIR).
 *
 * Covers:
 *  1. CutPool: Insertion, cosine similarity deduplication, capacity bounding, efficacy sorting
 *  2. MIRSeparator: Mixed-integer rounding cut generation, scaling factors, continuous bounds
 *  3. CuttingPlaneEngine: Root rounds, Gomory + MIR separation, bound strengthening
 *  4. Witness Verification: Feasible integer solutions satisfy generated cuts
 *  5. Branch-and-Cut Integration: Bound tightening, node reduction, equivalence with baseline
 *  6. Determinism: Repeated execution produces identical cuts and bounds
 *  7. Robustness: Pure continuous LP bypass and numerical stability
 */

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "../test_framework.hpp"
#include "algorithms/lp/simplex.hpp"
#include "algorithms/milp/branch_and_bound.hpp"
#include "algorithms/milp/cut_pool.hpp"
#include "algorithms/milp/cut_types.hpp"
#include "algorithms/milp/cutting_plane_engine.hpp"
#include "algorithms/milp/gomory_separator.hpp"
#include "algorithms/milp/mir_separator.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"
#include "verification/milp_verifier.hpp"

using namespace kavach;
using namespace kavach::model;
using namespace kavach::algorithms::milp;
using namespace kavach::verification;

// ============================================================================
// Helper Builders
// ============================================================================

namespace {

CanonicalModel make_knapsack_milp() {
    CanonicalModel m("knapsack_milp");
    auto x1 = m.add_binary_variable("x1");
    auto x2 = m.add_binary_variable("x2");
    auto x3 = m.add_binary_variable("x3");
    m.set_objective_sense(ObjectiveSense::MAXIMIZE);
    m.add_objective_linear_term(x1, 6.0);
    m.add_objective_linear_term(x2, 4.0);
    m.add_objective_linear_term(x3, 5.0);
    SparseLinearExpression e;
    e.add_term(x1, 3.0);
    e.add_term(x2, 2.0);
    e.add_term(x3, 1.0);
    m.add_constraint("cap", std::move(e), ConstraintSense::LESS_EQUAL, 5.0);
    return m;
}

CanonicalModel make_mixed_integer_model() {
    CanonicalModel m("mixed_int_cont");
    auto x = m.add_variable("x", VariableType::CONTINUOUS, 0.0, 3.9);
    auto y = m.add_integer_variable("y", 0.0, 100.0);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x, 2.5);
    m.add_objective_linear_term(y, 3.0);
    SparseLinearExpression e;
    e.add_term(x, 1.0);
    e.add_term(y, 1.0);
    m.add_constraint("lb", std::move(e), ConstraintSense::GREATER_EQUAL, 4.7);
    return m;
}

}  // namespace

// ============================================================================
// 1. CutPool Tests
// ============================================================================

KAVACH_TEST(cut_pool_deduplication_and_sorting) {
    CutPool pool(5);

    LinearCut cut1;
    cut1.type = CutType::GOMORY;
    cut1.var_ids = {0, 1};
    cut1.coefficients = {1.0, 1.0};
    cut1.rhs = 1.0;
    cut1.sense = ConstraintSense::LESS_EQUAL;
    cut1.violation = 0.5;
    cut1.norm2 = std::sqrt(2.0);
    cut1.efficacy = 0.5 / std::sqrt(2.0);

    // Identical parallel cut (scaled by 2, identical direction)
    LinearCut cut1_dup;
    cut1_dup.type = CutType::GOMORY;
    cut1_dup.var_ids = {0, 1};
    cut1_dup.coefficients = {2.0, 2.0};
    cut1_dup.rhs = 2.0;
    cut1_dup.sense = ConstraintSense::LESS_EQUAL;
    cut1_dup.violation = 1.0;
    cut1_dup.norm2 = std::sqrt(8.0);
    cut1_dup.efficacy = 0.5 / std::sqrt(2.0);

    // Orthogonal cut
    LinearCut cut2;
    cut2.type = CutType::MIR;
    cut2.var_ids = {0, 1};
    cut2.coefficients = {1.0, -1.0};
    cut2.rhs = 0.0;
    cut2.sense = ConstraintSense::LESS_EQUAL;
    cut2.violation = 0.8;
    cut2.norm2 = std::sqrt(2.0);
    cut2.efficacy = 0.8 / std::sqrt(2.0);

    KAVACH_ASSERT(pool.add_cut(cut1));
    KAVACH_ASSERT_EQUAL(pool.size(), 1u);

    // Duplicate should be rejected due to cosine similarity > 0.999
    KAVACH_ASSERT(!pool.add_cut(cut1_dup));
    KAVACH_ASSERT_EQUAL(pool.size(), 1u);

    // Distinct cut should be accepted
    KAVACH_ASSERT(pool.add_cut(cut2));
    KAVACH_ASSERT_EQUAL(pool.size(), 2u);

    // Best cuts ordered by efficacy descending: cut2 (MIR, 0.8/sqrt(2)) then cut1 (GOMORY,
    // 0.5/sqrt(2))
    auto best = pool.select_best_cuts(2);
    KAVACH_ASSERT_EQUAL(best.size(), 2u);
    KAVACH_ASSERT(best[0].type == CutType::MIR);
    KAVACH_ASSERT(best[1].type == CutType::GOMORY);
}

KAVACH_TEST(cut_pool_capacity_pruning) {
    CutPool pool(2);

    for (std::size_t i = 0; i < 5; ++i) {
        LinearCut c;
        c.type = CutType::GOMORY;
        c.var_ids = {static_cast<VarId>(i)};
        c.coefficients = {1.0};
        c.rhs = 1.0;
        c.sense = ConstraintSense::LESS_EQUAL;
        c.violation = 0.1 * static_cast<double>(i + 1);
        c.norm2 = 1.0;
        c.efficacy = c.violation;
        pool.add_cut(c);
    }

    pool.enforce_capacity();

    // Pool capacity is 2, so only top 2 most efficacious should remain
    KAVACH_ASSERT_EQUAL(pool.size(), 2u);
    auto best = pool.select_best_cuts(2);
    KAVACH_ASSERT_NEAR(best[0].efficacy, 0.5, 1e-9);
    KAVACH_ASSERT_NEAR(best[1].efficacy, 0.4, 1e-9);
}

// ============================================================================
// 2. MIR Separator Direct Test
// ============================================================================

KAVACH_TEST(mir_separation_mixed_integer) {
    // Textbook mixed-integer instance: x1 - x2 <= 0.5 with x1 integer, x2 continuous
    CanonicalModel model("mir_test");
    auto x1 = model.add_integer_variable("x1", 0.0, 10.0);
    auto x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, 10.0);
    SparseLinearExpression e;
    e.add_term(x1, 1.0);
    e.add_term(x2, -1.0);
    model.add_constraint("c", std::move(e), ConstraintSense::LESS_EQUAL, 0.5);

    // LP relaxation point: x1 = 0.8, x2 = 0.3 (satisfies x1 - x2 = 0.5 <= 0.5)
    std::vector<double> x_lp = {0.8, 0.3};

    MIRSeparator separator;
    CutOptions opts;
    opts.min_cut_violation = 1e-4;
    separator.set_options(opts);

    auto cuts = separator.separate(model, x_lp, 0, 0);
    KAVACH_ASSERT(cuts.size() > 0);

    for (const auto& cut : cuts) {
        KAVACH_ASSERT(cut.type == CutType::MIR);
        KAVACH_ASSERT(cut.efficacy > 0.0);
        KAVACH_ASSERT(cut.violation > 0.0);
        for (double coeff : cut.coefficients) {
            KAVACH_ASSERT(!std::isnan(coeff));
            KAVACH_ASSERT(!std::isinf(coeff));
            KAVACH_ASSERT(coeff != 0.0);
        }
    }
}

// ============================================================================
// 3. CuttingPlaneEngine Root Rounds & Witness Verification
// ============================================================================

KAVACH_TEST(cutting_plane_engine_root_rounds) {
    auto model = make_knapsack_milp();
    algorithms::lp::RevisedSimplex lp_solver;
    algorithms::lp::LPSolverOptions lp_opts;
    lp_opts.enable_presolve = false;

    auto rel = model;
    for (std::size_t i = 0; i < rel.num_variables(); ++i) {
        rel.set_variable_type(static_cast<VarId>(i), VariableType::CONTINUOUS);
    }
    auto lp_sol = lp_solver.solve(rel);
    KAVACH_ASSERT(lp_sol.is_optimal());

    double continuous_obj = lp_sol.objective_value;

    CuttingPlaneEngine engine;
    CutOptions opts;
    opts.strategy = CutStrategy::ROOT_ONLY;
    opts.max_cut_rounds = 3;
    engine.set_options(opts);

    CutTelemetry telemetry;
    auto work_model = model;
    auto result = engine.run_root_cuts(work_model, lp_solver, lp_opts, lp_sol, telemetry, 1e-6);

    KAVACH_ASSERT(telemetry.cut_rounds > 0);
    KAVACH_ASSERT(telemetry.cuts_accepted > 0);
    KAVACH_ASSERT(!result.infeasible);

    // Known optimal integer solution: x1 = 1, x2 = 0, x3 = 1 -> obj = 11, cap = 4 <= 5
    std::vector<double> integer_optimal = {1.0, 0.0, 1.0};
    // Alternative feasible integer solution: x1 = 1, x2 = 1, x3 = 0 -> obj = 10, cap = 5 <= 5
    std::vector<double> integer_subopt = {1.0, 1.0, 0.0};

    // Verify all generated and accepted cuts do NOT cut off feasible integer points
    for (const auto& cut : engine.pool().cuts()) {
        KAVACH_ASSERT(cut.compute_violation(integer_optimal) <= 1e-6);
        KAVACH_ASSERT(cut.compute_violation(integer_subopt) <= 1e-6);
    }

    // Strengthened bound should not be looser than original continuous LP
    KAVACH_ASSERT(result.objective <= continuous_obj + 1e-6);
}

// ============================================================================
// 4. Branch-and-Cut Integration & Equivalence
// ============================================================================

KAVACH_TEST(branch_and_cut_knapsack_equivalence) {
    auto model = make_knapsack_milp();

    // 1. Solve with cuts OFF
    MILPSolverOptions opts_off;
    opts_off.cut_options.strategy = CutStrategy::OFF;
    BranchAndBound solver_off(opts_off);
    auto res_off = solver_off.solve(model);

    // 2. Solve with cuts ROOT_ONLY
    MILPSolverOptions opts_cuts;
    opts_cuts.cut_options.strategy = CutStrategy::ROOT_ONLY;
    BranchAndBound solver_cuts(opts_cuts);
    auto res_cuts = solver_cuts.solve(model);

    KAVACH_ASSERT(res_off.status == core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(res_cuts.status == core::SolverStatus::OPTIMAL);

    // Both must find the exact same optimal objective
    KAVACH_ASSERT_NEAR(res_off.best_objective, 11.0, 1e-6);
    KAVACH_ASSERT_NEAR(res_cuts.best_objective, 11.0, 1e-6);
    KAVACH_ASSERT_NEAR(res_off.best_objective, res_cuts.best_objective, 1e-6);

    // With cuts, root cuts should tighten relaxation and explore <= nodes
    KAVACH_ASSERT(res_cuts.telemetry.nodes_explored <= res_off.telemetry.nodes_explored);

    // Verify both solutions with independent zero-trust MILPVerifier
    MILPVerifier verifier;
    auto rep_off = verifier.verify(model, res_off.incumbent_solution, res_off.best_bound,
                                   res_off.best_objective, res_off.status);
    auto rep_cuts = verifier.verify(model, res_cuts.incumbent_solution, res_cuts.best_bound,
                                    res_cuts.best_objective, res_cuts.status);

    KAVACH_ASSERT(rep_off.fully_verified());
    KAVACH_ASSERT(rep_cuts.fully_verified());
    KAVACH_ASSERT(rep_off.certified_status == core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(rep_cuts.certified_status == core::SolverStatus::OPTIMAL);
}

KAVACH_TEST(branch_and_cut_mixed_integer_equivalence) {
    auto model = make_mixed_integer_model();

    MILPSolverOptions opts_off;
    opts_off.cut_options.strategy = CutStrategy::OFF;
    BranchAndBound solver_off(opts_off);
    auto res_off = solver_off.solve(model);

    MILPSolverOptions opts_cuts;
    opts_cuts.cut_options.strategy = CutStrategy::ROOT_ONLY;
    BranchAndBound solver_cuts(opts_cuts);
    auto res_cuts = solver_cuts.solve(model);

    KAVACH_ASSERT(res_off.status == core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(res_cuts.status == core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(res_off.best_objective, 12.25, 1e-6);
    KAVACH_ASSERT_NEAR(res_cuts.best_objective, 12.25, 1e-6);
    KAVACH_ASSERT_NEAR(res_off.best_objective, res_cuts.best_objective, 1e-6);

    MILPVerifier verifier;
    auto rep_cuts = verifier.verify(model, res_cuts.incumbent_solution, res_cuts.best_bound,
                                    res_cuts.best_objective, res_cuts.status);
    KAVACH_ASSERT(rep_cuts.fully_verified());
}

// ============================================================================
// 5. Determinism & Repeated Solves
// ============================================================================

KAVACH_TEST(branch_and_cut_determinism) {
    auto model = make_knapsack_milp();

    MILPSolverOptions opts;
    opts.cut_options.strategy = CutStrategy::ROOT_ONLY;

    BranchAndBound s1(opts);
    auto r1 = s1.solve(model);

    BranchAndBound s2(opts);
    auto r2 = s2.solve(model);

    KAVACH_ASSERT(r1.status == r2.status);
    KAVACH_ASSERT_NEAR(r1.best_objective, r2.best_objective, 1e-12);
    KAVACH_ASSERT_EQUAL(r1.telemetry.cut_telemetry.cuts_accepted,
                        r2.telemetry.cut_telemetry.cuts_accepted);
    KAVACH_ASSERT_EQUAL(r1.telemetry.cut_telemetry.gomory_cuts,
                        r2.telemetry.cut_telemetry.gomory_cuts);
    KAVACH_ASSERT_EQUAL(r1.telemetry.cut_telemetry.mir_cuts, r2.telemetry.cut_telemetry.mir_cuts);
    KAVACH_ASSERT_EQUAL(r1.telemetry.nodes_explored, r2.telemetry.nodes_explored);

    for (std::size_t i = 0; i < r1.incumbent_solution.size(); ++i) {
        KAVACH_ASSERT_NEAR(r1.incumbent_solution[i], r2.incumbent_solution[i], 1e-12);
    }
}

// ============================================================================
// 6. Robustness: Pure Continuous LP
// ============================================================================

KAVACH_TEST(cutting_planes_pure_continuous_safety) {
    // Pure continuous model: no integer variables
    CanonicalModel m("pure_continuous");
    auto x = m.add_variable("x", VariableType::CONTINUOUS, 0.0, 10.0);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x, 1.0);
    SparseLinearExpression e;
    e.add_term(x, 1.0);
    m.add_constraint("c", std::move(e), ConstraintSense::GREATER_EQUAL, 2.5);

    MILPSolverOptions opts;
    opts.cut_options.strategy = CutStrategy::ROOT_ONLY;
    BranchAndBound solver(opts);
    auto res = solver.solve(m);

    KAVACH_ASSERT(res.status == core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(res.best_objective, 2.5, 1e-6);
    // No cuts should have been generated or applied
    KAVACH_ASSERT_EQUAL(res.telemetry.cut_telemetry.cuts_accepted, 0u);
}
