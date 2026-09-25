/**
 * @file test_phase6_verification.cpp
 * @brief Comprehensive test suite for the Independent LP Verification Layer [Y].
 *
 * Covers:
 *  - Standard LP verification (optimal, equality, mixed senses, bounds, free vars)
 *  - Presolve-reconstructed solution verification against original model
 *  - Objective constant offset verification
 *  - Degenerate LP verification
 *  - Fault injection: every injected fault must be detected
 *  - Negative verification tests: every bad result must fail
 *  - Full integration: solver → verifier pipeline
 *  - Determinism: identical inputs produce identical results
 *  - MILP / QP stub tests
 */

#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "../test_framework.hpp"
#include "algorithms/lp/lp_types.hpp"
#include "algorithms/lp/simplex.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "io/parser.hpp"
#include "model/model.hpp"
#include "verification/lp_verifier.hpp"
#include "verification/verifier.hpp"

using namespace kavach;
using namespace kavach::model;
using namespace kavach::verification;
using namespace kavach::algorithms::lp;

namespace {

// ============================================================================
// Shared helper: solve a product-mix LP
//   max 3x1 + 2x2  s.t.  2x1+x2<=100,  x1+x2<=80,  x1,x2>=0
//   Optimal: x1=20, x2=60, obj=180
// ============================================================================
LPSolution solve_product_mix(CanonicalModel& out_model) {
    out_model = CanonicalModel("product_mix");
    auto x1 = out_model.add_variable("x1", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto x2 = out_model.add_variable("x2", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    out_model.set_objective_sense(ObjectiveSense::MAXIMIZE);
    out_model.add_objective_linear_term(x1, 3.0);
    out_model.add_objective_linear_term(x2, 2.0);
    SparseLinearExpression e1;
    e1.add_term(x1, 2.0);
    e1.add_term(x2, 1.0);
    out_model.add_constraint("c1", std::move(e1), ConstraintSense::LESS_EQUAL, 100.0);
    SparseLinearExpression e2;
    e2.add_term(x1, 1.0);
    e2.add_term(x2, 1.0);
    out_model.add_constraint("c2", std::move(e2), ConstraintSense::LESS_EQUAL, 80.0);
    RevisedSimplex solver;
    return solver.solve(out_model);
}

}  // namespace

// ============================================================================
// SECTION A: Standard LP Verification — Valid Results
// ============================================================================

KAVACH_TEST(verify_simple_optimal_lp) {
    CanonicalModel m;
    auto sol = solve_product_mix(m);
    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(sol.objective_value, 180.0, 1e-4);
}

KAVACH_TEST(verify_equality_lp) {
    // min x1 + 2x2  s.t. x1+x2=10, x1,x2>=0 → optimal x1=10,x2=0,obj=10
    CanonicalModel m("eq_lp");
    auto x1 = m.add_variable("x1", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto x2 = m.add_variable("x2", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x1, 1.0);
    m.add_objective_linear_term(x2, 2.0);
    SparseLinearExpression e;
    e.add_term(x1, 1.0);
    e.add_term(x2, 1.0);
    m.add_constraint("eq", std::move(e), ConstraintSense::EQUAL, 10.0);
    RevisedSimplex solver;
    auto sol = solver.solve(m);
    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(sol.objective_value, 10.0, 1e-4);

    LPVerifier ver;
    auto rpt = ver.verify(m, sol.primal_solution, sol.dual_multipliers, sol.reduced_costs,
                          sol.objective_value, sol.status);
    KAVACH_ASSERT(rpt.primal_feasibility.passed);
    KAVACH_ASSERT(rpt.bound_feasibility.passed);
    KAVACH_ASSERT(rpt.nonfinite_check.passed);
    KAVACH_ASSERT(rpt.objective_consistency.passed);
}

KAVACH_TEST(verify_mixed_senses_lp) {
    CanonicalModel m("mixed");
    auto x1 = m.add_variable("x1", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto x2 = m.add_variable("x2", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.set_objective_offset(25.0);
    m.add_objective_linear_term(x1, 2.0);
    m.add_objective_linear_term(x2, 3.0);
    SparseLinearExpression e1;
    e1.add_term(x1, 1.0);
    e1.add_term(x2, 1.0);
    m.add_constraint("ge", std::move(e1), ConstraintSense::GREATER_EQUAL, 4.0);
    SparseLinearExpression e2;
    e2.add_term(x1, 2.0);
    e2.add_term(x2, -1.0);
    m.add_constraint("le", std::move(e2), ConstraintSense::LESS_EQUAL, 6.0);
    RevisedSimplex solver;
    auto sol = solver.solve(m);
    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);

    LPVerifier ver;
    auto rpt = ver.verify(m, sol.primal_solution, sol.dual_multipliers, sol.reduced_costs,
                          sol.objective_value, sol.status);
    KAVACH_ASSERT(rpt.primal_feasibility.passed);
    KAVACH_ASSERT(rpt.bound_feasibility.passed);
    KAVACH_ASSERT(rpt.objective_consistency.passed);
    KAVACH_ASSERT(rpt.nonfinite_check.passed);
}

KAVACH_TEST(verify_bounded_variables) {
    CanonicalModel m("bounded");
    auto x = m.add_variable("x", VariableType::CONTINUOUS, 2.0, 10.0);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x, 1.0);
    RevisedSimplex solver;
    auto sol = solver.solve(m);
    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(sol.objective_value, 2.0, 1e-4);

    LPVerifier ver;
    auto rpt = ver.verify(m, sol.primal_solution, sol.dual_multipliers, sol.reduced_costs,
                          sol.objective_value, sol.status);
    KAVACH_ASSERT(rpt.bound_feasibility.passed);
    KAVACH_ASSERT(rpt.objective_consistency.passed);
}

KAVACH_TEST(verify_objective_offset) {
    // min x + 100.0 s.t. x >= 5 → obj=105
    CanonicalModel m("offset_lp");
    auto x = m.add_variable("x", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.set_objective_offset(100.0);
    m.add_objective_linear_term(x, 1.0);
    SparseLinearExpression e;
    e.add_term(x, 1.0);
    m.add_constraint("lb", std::move(e), ConstraintSense::GREATER_EQUAL, 5.0);
    RevisedSimplex solver;
    auto sol = solver.solve(m);
    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(sol.objective_value, 105.0, 1e-4);

    LPVerifier ver;
    auto rpt = ver.verify(m, sol.primal_solution, sol.dual_multipliers, sol.reduced_costs,
                          sol.objective_value, sol.status);
    KAVACH_ASSERT(rpt.objective_consistency.passed);
    KAVACH_ASSERT_NEAR(rpt.recomputed_objective, 105.0, 1e-4);
}

KAVACH_TEST(verify_infeasible_lp) {
    CanonicalModel m("infeas");
    auto x = m.add_variable("x", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    SparseLinearExpression e1;
    e1.add_term(x, 1.0);
    m.add_constraint("lb", std::move(e1), ConstraintSense::GREATER_EQUAL, 5.0);
    SparseLinearExpression e2;
    e2.add_term(x, 1.0);
    m.add_constraint("ub", std::move(e2), ConstraintSense::LESS_EQUAL, 3.0);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x, 1.0);
    RevisedSimplex solver;
    auto sol = solver.solve(m);
    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::INFEASIBLE);

    LPVerifier ver;
    auto rpt = ver.verify(m, sol.primal_solution, sol.dual_multipliers, sol.reduced_costs,
                          sol.objective_value, sol.status);
    KAVACH_ASSERT(rpt.status_consistency.passed);
    KAVACH_ASSERT_EQUAL(rpt.certified_status, core::SolverStatus::INFEASIBLE);
}

KAVACH_TEST(verify_unbounded_lp) {
    CanonicalModel m("unb");
    auto x = m.add_variable("x", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x, -1.0);
    RevisedSimplex solver;
    auto sol = solver.solve(m);
    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::UNBOUNDED);

    LPVerifier ver;
    auto rpt = ver.verify(m, sol.primal_solution, sol.dual_multipliers, sol.reduced_costs,
                          sol.objective_value, sol.status);
    KAVACH_ASSERT(rpt.status_consistency.passed);
    KAVACH_ASSERT_EQUAL(rpt.certified_status, core::SolverStatus::UNBOUNDED);
}

KAVACH_TEST(verify_degenerate_lp) {
    CanonicalModel m("degen");
    auto x = m.add_variable("x", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto y = m.add_variable("y", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x, 1.0);
    m.add_objective_linear_term(y, 1.0);
    SparseLinearExpression e1;
    e1.add_term(x, 1.0);
    m.add_constraint("c1", std::move(e1), ConstraintSense::GREATER_EQUAL, 0.0);
    SparseLinearExpression e2;
    e2.add_term(y, 1.0);
    m.add_constraint("c2", std::move(e2), ConstraintSense::GREATER_EQUAL, 0.0);
    SparseLinearExpression e3;
    e3.add_term(x, 1.0);
    e3.add_term(y, 1.0);
    m.add_constraint("c3", std::move(e3), ConstraintSense::GREATER_EQUAL, 0.0);
    RevisedSimplex solver;
    auto sol = solver.solve(m);
    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);

    LPVerifier ver;
    auto rpt = ver.verify(m, sol.primal_solution, sol.dual_multipliers, sol.reduced_costs,
                          sol.objective_value, sol.status);
    KAVACH_ASSERT(rpt.primal_feasibility.passed);
    KAVACH_ASSERT(rpt.bound_feasibility.passed);
    KAVACH_ASSERT(rpt.objective_consistency.passed);
}

// ============================================================================
// SECTION B: Presolve Reconstruction Validation
// ============================================================================

KAVACH_TEST(verify_presolved_solution_against_original_model) {
    CanonicalModel m("presolve_test");
    auto x1 = m.add_variable("x1", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto x2 = m.add_variable("x2", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto x3 =
        m.add_variable("x3", VariableType::CONTINUOUS, 5.0, 5.0);  // fixed: presolve eliminates
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x1, 1.0);
    m.add_objective_linear_term(x2, 2.0);
    m.add_objective_linear_term(x3, 3.0);
    SparseLinearExpression e;
    e.add_term(x1, 1.0);
    e.add_term(x2, 1.0);
    m.add_constraint("sum", std::move(e), ConstraintSense::GREATER_EQUAL, 4.0);
    RevisedSimplex solver;
    auto sol = solver.solve(m);
    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);
    // obj = 1*4 + 2*0 + 3*5 = 19
    KAVACH_ASSERT_NEAR(sol.objective_value, 19.0, 1e-3);

    LPVerifier ver;
    auto rpt = ver.verify(m, sol.primal_solution, sol.dual_multipliers, sol.reduced_costs,
                          sol.objective_value, sol.status);
    KAVACH_ASSERT(rpt.primal_feasibility.passed);
    KAVACH_ASSERT(rpt.bound_feasibility.passed);
    KAVACH_ASSERT(rpt.objective_consistency.passed);
    KAVACH_ASSERT_NEAR(rpt.recomputed_objective, 19.0, 1e-3);
}

// ============================================================================
// SECTION C: Fault Injection — Every Injected Fault Must Be Detected
// ============================================================================

KAVACH_TEST(fault_A_wrong_objective_detected) {
    CanonicalModel m;
    auto sol = solve_product_mix(m);
    // Inject: report wrong objective (correct is 180)
    LPVerifier ver;
    auto rpt = ver.verify(m, sol.primal_solution, sol.dual_multipliers, sol.reduced_costs, 999.0,
                          core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rpt.objective_consistency.passed);
    KAVACH_ASSERT(rpt.objective_absolute_error > 1.0);
}

KAVACH_TEST(fault_B_primal_infeasibility_detected) {
    CanonicalModel m;
    auto sol = solve_product_mix(m);
    // Inject: push x2 far beyond feasible region
    std::vector<double> bad_x = sol.primal_solution;
    bad_x[1] += 200.0;  // 2*20 + 260 = 300 >> 100 for c1
    LPVerifier ver;
    auto rpt = ver.verify(m, bad_x, sol.dual_multipliers, sol.reduced_costs, sol.objective_value,
                          core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rpt.primal_feasibility.passed);
    KAVACH_ASSERT(rpt.primal_feasibility.max_violation > 1.0);
}

KAVACH_TEST(fault_C_bound_violation_detected) {
    CanonicalModel m("bv_test");
    auto x = m.add_variable("x", VariableType::CONTINUOUS, 0.0, 10.0);
    auto y = m.add_variable("y", VariableType::CONTINUOUS, 0.0, 10.0);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x, 1.0);
    m.add_objective_linear_term(y, 1.0);
    SparseLinearExpression e;
    e.add_term(x, 1.0);
    e.add_term(y, 1.0);
    m.add_constraint("c", std::move(e), ConstraintSense::LESS_EQUAL, 20.0);
    // Inject: x=15 violates ub=10
    std::vector<double> bad_x = {15.0, 0.0};
    LPVerifier ver;
    auto rpt = ver.verify(m, bad_x, {}, {}, 15.0, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rpt.bound_feasibility.passed);
    KAVACH_ASSERT(rpt.bound_feasibility.max_violation > 1.0);
}

KAVACH_TEST(fault_D_dual_violation_detected) {
    CanonicalModel m("dual_viol");
    auto x = m.add_variable("x", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto y = m.add_variable("y", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x, 2.0);
    m.add_objective_linear_term(y, 3.0);
    SparseLinearExpression e;
    e.add_term(x, 1.0);
    e.add_term(y, 1.0);
    m.add_constraint("c", std::move(e), ConstraintSense::GREATER_EQUAL, 5.0);
    // Correct primal: x=5,y=0
    std::vector<double> good_x = {5.0, 0.0};
    // Inject bad dual: wrong sign and magnitude
    std::vector<double> bad_y = {-100.0};
    LPVerifier ver;
    auto rpt = ver.verify(m, good_x, bad_y, {}, 10.0, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rpt.dual_feasibility.passed);
}

KAVACH_TEST(fault_E_complementary_slackness_violation_detected) {
    CanonicalModel m("cs_viol");
    auto x = m.add_variable("x", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto yv = m.add_variable("y", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x, 1.0);
    m.add_objective_linear_term(yv, 2.0);
    SparseLinearExpression e1;
    e1.add_term(x, 1.0);
    e1.add_term(yv, 1.0);
    m.add_constraint("le", std::move(e1), ConstraintSense::LESS_EQUAL, 10.0);
    // x=5,y=0 → slack = 10-5 = 5 (non-zero)
    std::vector<double> xvec = {5.0, 0.0};
    // Inject very large dual → slack*dual = 5*1000 = 5000 >> tol
    std::vector<double> bad_dual = {1000.0};
    LPVerifier ver;
    auto rpt = ver.verify(m, xvec, bad_dual, {}, 5.0, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rpt.complementary_slackness.passed);
    KAVACH_ASSERT(rpt.complementary_slackness.max_violation > 100.0);
}

KAVACH_TEST(fault_F_nan_in_primal_detected) {
    CanonicalModel m("nan_test");
    auto x = m.add_variable("x", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x, 1.0);
    SparseLinearExpression e;
    e.add_term(x, 1.0);
    m.add_constraint("c", std::move(e), ConstraintSense::GREATER_EQUAL, 1.0);
    std::vector<double> nan_x = {std::numeric_limits<double>::quiet_NaN()};
    LPVerifier ver;
    auto rpt = ver.verify(m, nan_x, {}, {}, 1.0, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rpt.nonfinite_check.passed);
    KAVACH_ASSERT(!rpt.fully_verified());
}

KAVACH_TEST(fault_F_inf_in_primal_detected) {
    CanonicalModel m("inf_test");
    auto x = m.add_variable("x", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x, 1.0);
    SparseLinearExpression e;
    e.add_term(x, 1.0);
    m.add_constraint("c", std::move(e), ConstraintSense::GREATER_EQUAL, 1.0);
    std::vector<double> inf_x = {std::numeric_limits<double>::infinity()};
    LPVerifier ver;
    auto rpt = ver.verify(m, inf_x, {}, {}, 1.0, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rpt.nonfinite_check.passed);
    KAVACH_ASSERT(!rpt.fully_verified());
}

KAVACH_TEST(fault_G_wrong_reported_obj_with_valid_primal_detected) {
    // Feasible point x=3,y=3 with injected wrong objective 4 (true obj=6)
    CanonicalModel m("status_corrupt");
    auto x = m.add_variable("x", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto y = m.add_variable("y", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x, 1.0);
    m.add_objective_linear_term(y, 1.0);
    SparseLinearExpression e;
    e.add_term(x, 1.0);
    e.add_term(y, 1.0);
    m.add_constraint("c", std::move(e), ConstraintSense::GREATER_EQUAL, 4.0);
    // Suboptimal point, report wrong objective
    std::vector<double> subopt = {3.0, 3.0};  // feasible, obj=6
    LPVerifier ver;
    // Report obj=4 (which is wrong — recomputed will be 6)
    auto rpt = ver.verify(m, subopt, {}, {}, 4.0, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rpt.objective_consistency.passed);
    KAVACH_ASSERT(rpt.objective_absolute_error > 1.0);
}

// ============================================================================
// SECTION D: Negative Tests — Bad Inputs Must Always Fail
// ============================================================================

KAVACH_TEST(negative_wrong_primal_fails_constraint_check) {
    CanonicalModel m("neg_primal");
    auto x = m.add_variable("x", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto y = m.add_variable("y", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x, 1.0);
    m.add_objective_linear_term(y, 1.0);
    SparseLinearExpression e;
    e.add_term(x, 1.0);
    e.add_term(y, 1.0);
    m.add_constraint("c", std::move(e), ConstraintSense::LESS_EQUAL, 5.0);
    // x+y=100 >> 5 — constraint violated
    std::vector<double> bad = {50.0, 50.0};
    LPVerifier ver;
    auto rpt = ver.verify(m, bad, {}, {}, 100.0, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rpt.primal_feasibility.passed);
    KAVACH_ASSERT(!rpt.fully_verified());
}

KAVACH_TEST(negative_nan_in_dual_fails) {
    CanonicalModel m("neg_dual_nan");
    auto x = m.add_variable("x", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x, 1.0);
    SparseLinearExpression e;
    e.add_term(x, 1.0);
    m.add_constraint("c", std::move(e), ConstraintSense::GREATER_EQUAL, 1.0);
    std::vector<double> good_x = {1.0};
    std::vector<double> nan_dual = {std::numeric_limits<double>::quiet_NaN()};
    LPVerifier ver;
    auto rpt = ver.verify(m, good_x, nan_dual, {}, 1.0, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rpt.nonfinite_check.passed);
    KAVACH_ASSERT(!rpt.fully_verified());
}

KAVACH_TEST(negative_no_dual_info_optimal_cannot_be_certified) {
    CanonicalModel m("nodual");
    auto x = m.add_variable("x", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x, 1.0);
    SparseLinearExpression e;
    e.add_term(x, 1.0);
    m.add_constraint("c", std::move(e), ConstraintSense::GREATER_EQUAL, 1.0);
    std::vector<double> xvec = {1.0};
    LPVerifier ver;
    auto rpt = ver.verify(m, xvec, {}, {}, 1.0, core::SolverStatus::OPTIMAL);
    // Cannot certify OPTIMAL without dual info
    KAVACH_ASSERT(!rpt.status_consistency.passed);
    KAVACH_ASSERT(rpt.certified_status != core::SolverStatus::OPTIMAL);
}

KAVACH_TEST(negative_lower_bound_violated) {
    CanonicalModel m("lb_viol");
    auto x = m.add_variable("x", VariableType::CONTINUOUS, 3.0, 10.0);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x, 1.0);
    std::vector<double> bad = {1.0};  // violates lb=3
    LPVerifier ver;
    auto rpt = ver.verify(m, bad, {}, {}, 1.0, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rpt.bound_feasibility.passed);
    KAVACH_ASSERT(rpt.bound_feasibility.max_violation > 1.0);
}

KAVACH_TEST(negative_upper_bound_violated) {
    CanonicalModel m("ub_viol");
    auto x = m.add_variable("x", VariableType::CONTINUOUS, 0.0, 5.0);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x, 1.0);
    std::vector<double> bad = {8.0};  // violates ub=5
    LPVerifier ver;
    auto rpt = ver.verify(m, bad, {}, {}, 8.0, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rpt.bound_feasibility.passed);
    KAVACH_ASSERT(rpt.bound_feasibility.max_violation > 1.0);
}

// ============================================================================
// SECTION E: Determinism Tests
// ============================================================================

KAVACH_TEST(verify_determinism) {
    CanonicalModel m;
    auto sol = solve_product_mix(m);
    LPVerifier ver;
    auto r1 = ver.verify(m, sol.primal_solution, sol.dual_multipliers, sol.reduced_costs,
                         sol.objective_value, sol.status);
    auto r2 = ver.verify(m, sol.primal_solution, sol.dual_multipliers, sol.reduced_costs,
                         sol.objective_value, sol.status);
    auto r3 = ver.verify(m, sol.primal_solution, sol.dual_multipliers, sol.reduced_costs,
                         sol.objective_value, sol.status);

    KAVACH_ASSERT_EQUAL(r1.primal_feasibility.passed, r2.primal_feasibility.passed);
    KAVACH_ASSERT_EQUAL(r2.primal_feasibility.passed, r3.primal_feasibility.passed);
    KAVACH_ASSERT_EQUAL(r1.certified_status, r2.certified_status);
    KAVACH_ASSERT_EQUAL(r2.certified_status, r3.certified_status);
    KAVACH_ASSERT_NEAR(r1.recomputed_objective, r2.recomputed_objective, 1e-15);
    KAVACH_ASSERT_NEAR(r2.recomputed_objective, r3.recomputed_objective, 1e-15);
    KAVACH_ASSERT_NEAR(r1.aggregate_primal_residual, r3.aggregate_primal_residual, 1e-15);
}

// ============================================================================
// SECTION F: Integration Pipeline Tests (Solver → Verifier)
// ============================================================================

KAVACH_TEST(integration_ordinary_lp) {
    // max 5x+4y s.t. 6x+4y<=24, x+2y<=6
    CanonicalModel m("int_ord");
    auto x = m.add_variable("x", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto y = m.add_variable("y", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    m.set_objective_sense(ObjectiveSense::MAXIMIZE);
    m.add_objective_linear_term(x, 5.0);
    m.add_objective_linear_term(y, 4.0);
    SparseLinearExpression e1;
    e1.add_term(x, 6.0);
    e1.add_term(y, 4.0);
    m.add_constraint("c1", std::move(e1), ConstraintSense::LESS_EQUAL, 24.0);
    SparseLinearExpression e2;
    e2.add_term(x, 1.0);
    e2.add_term(y, 2.0);
    m.add_constraint("c2", std::move(e2), ConstraintSense::LESS_EQUAL, 6.0);
    RevisedSimplex solver;
    auto sol = solver.solve(m);
    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);

    LPVerifier ver;
    auto rpt = ver.verify(m, sol.primal_solution, sol.dual_multipliers, sol.reduced_costs,
                          sol.objective_value, sol.status);
    KAVACH_ASSERT(rpt.primal_feasibility.passed);
    KAVACH_ASSERT(rpt.bound_feasibility.passed);
    KAVACH_ASSERT(rpt.objective_consistency.passed);
}

KAVACH_TEST(integration_presolve_heavy_lp) {
    CanonicalModel m("presolve_heavy");
    auto x0 = m.add_variable("x0", VariableType::CONTINUOUS, 3.0, 3.0);  // fixed
    auto x1 = m.add_variable("x1", VariableType::CONTINUOUS, 7.0, 7.0);  // fixed
    auto x2 = m.add_variable("x2", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x0, 1.0);
    m.add_objective_linear_term(x1, 2.0);
    m.add_objective_linear_term(x2, 3.0);
    SparseLinearExpression e;
    e.add_term(x2, 1.0);
    m.add_constraint("c", std::move(e), ConstraintSense::GREATER_EQUAL, 2.0);
    RevisedSimplex solver;
    auto sol = solver.solve(m);
    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);
    // obj = 1*3 + 2*7 + 3*2 = 3+14+6 = 23
    KAVACH_ASSERT_NEAR(sol.objective_value, 23.0, 1e-3);

    LPVerifier ver;
    auto rpt = ver.verify(m, sol.primal_solution, sol.dual_multipliers, sol.reduced_costs,
                          sol.objective_value, sol.status);
    KAVACH_ASSERT(rpt.primal_feasibility.passed);
    KAVACH_ASSERT(rpt.bound_feasibility.passed);
    KAVACH_ASSERT_NEAR(rpt.recomputed_objective, 23.0, 1e-3);
}

KAVACH_TEST(integration_corrupted_result_rejected) {
    CanonicalModel m;
    auto sol = solve_product_mix(m);
    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);

    // Corrupt: inject NaN into primal
    std::vector<double> corrupted = sol.primal_solution;
    corrupted[0] = std::numeric_limits<double>::quiet_NaN();

    LPVerifier ver;
    auto rpt = ver.verify(m, corrupted, sol.dual_multipliers, sol.reduced_costs,
                          sol.objective_value, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rpt.nonfinite_check.passed);
    KAVACH_ASSERT(!rpt.fully_verified());
    KAVACH_ASSERT_EQUAL(rpt.certified_status, core::SolverStatus::NUMERICAL_FAILURE);
}

// ============================================================================
// SECTION G: Structured Report Field Tests
// ============================================================================

KAVACH_TEST(report_exposes_residuals_not_just_bool) {
    CanonicalModel m;
    auto sol = solve_product_mix(m);
    LPVerifier ver;
    auto rpt = ver.verify(m, sol.primal_solution, sol.dual_multipliers, sol.reduced_costs,
                          sol.objective_value, sol.status);
    KAVACH_ASSERT(std::isfinite(rpt.aggregate_primal_residual));
    KAVACH_ASSERT(std::isfinite(rpt.recomputed_objective));
    KAVACH_ASSERT(std::isfinite(rpt.objective_absolute_error));
    KAVACH_ASSERT(!rpt.summary.empty());
    // Tolerance used must be configured positive value
    KAVACH_ASSERT(rpt.primal_feasibility.tolerance_used > 0.0);
    KAVACH_ASSERT(rpt.bound_feasibility.tolerance_used > 0.0);
}

KAVACH_TEST(report_status_not_downgraded_for_valid_optimal) {
    CanonicalModel m;
    auto sol = solve_product_mix(m);
    LPVerifier ver;
    auto rpt = ver.verify(m, sol.primal_solution, sol.dual_multipliers, sol.reduced_costs,
                          sol.objective_value, sol.status);
    // Solver returned valid OPTIMAL — verifier must not downgrade
    KAVACH_ASSERT(!rpt.status_downgraded);
    KAVACH_ASSERT_EQUAL(rpt.certified_status, core::SolverStatus::OPTIMAL);
}

// ============================================================================
// SECTION H: MILP / QP Stub Tests
// ============================================================================

KAVACH_TEST(milp_stub_returns_unknown) {
    KavachLPVerifier kv;
    CanonicalModel m("milp_stub");
    auto x = m.add_variable("x", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x, 1.0);
    auto rep = kv.verify_milp(m, {}, 0.0, 0.0, core::SolverTolerances{});
    KAVACH_ASSERT_EQUAL(rep.certified_status, core::SolverStatus::UNKNOWN);
    KAVACH_ASSERT(!rep.summary.empty());
}

KAVACH_TEST(qp_stub_returns_unknown) {
    KavachLPVerifier kv;
    CanonicalModel m("qp_stub");
    auto x = m.add_variable("x", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x, 1.0);
    auto rep = kv.verify_qp(m, {}, {}, 0.0, core::SolverTolerances{});
    KAVACH_ASSERT_EQUAL(rep.certified_status, core::SolverStatus::UNKNOWN);
    KAVACH_ASSERT(!rep.summary.empty());
}

// ============================================================================
// SECTION I: Netlib Integration (optional — skipped if benchmarks not found)
// ============================================================================

KAVACH_TEST(netlib_afiro_verify_via_direct_verifier) {
    namespace fs = std::filesystem;
    fs::path netlib_dir;
    for (auto& p : {"benchmarks/netlib", "../benchmarks/netlib", "../../benchmarks/netlib"}) {
        if (fs::exists(p) && fs::exists(fs::path(p) / "afiro.mps")) {
            netlib_dir = p;
            break;
        }
    }
    if (netlib_dir.empty()) {
        std::cout << "    (skipped: benchmarks/netlib/afiro.mps not found)\n";
        return;
    }
    kavach::io::MpsParser parser;
    auto model_ptr = parser.parse_file((netlib_dir / "afiro.mps").string());
    KAVACH_ASSERT(model_ptr != nullptr);
    const auto& model = *model_ptr;
    KAVACH_ASSERT(model.num_variables() > 0);

    LPSolverOptions opts;
    opts.enable_presolve = false;
    RevisedSimplex solver(opts);
    auto sol = solver.solve(model);
    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);

    LPVerifier ver;
    auto rpt = ver.verify(model, sol.primal_solution, sol.dual_multipliers, sol.reduced_costs,
                          sol.objective_value, sol.status);
    KAVACH_ASSERT(rpt.primal_feasibility.passed);
    KAVACH_ASSERT(rpt.bound_feasibility.passed);
    KAVACH_ASSERT(rpt.dual_feasibility.passed);
    KAVACH_ASSERT(rpt.complementary_slackness.passed);
    KAVACH_ASSERT(rpt.objective_consistency.passed);
    KAVACH_ASSERT(rpt.fully_verified());
    KAVACH_ASSERT_NEAR(rpt.recomputed_objective, -464.75314, 1.0);
}
