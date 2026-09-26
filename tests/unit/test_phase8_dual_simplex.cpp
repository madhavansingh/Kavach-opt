/**
 * @file test_phase8_dual_simplex.cpp
 * @brief Comprehensive test suite for the Dual-Simplex Warm-Start Subsystem.
 *
 * Covers:
 *  - Section 16: Dual Simplex Unit Tests (A through N)
 *  - Section 17: Warm vs Cold Mathematical Equivalence on Child Nodes
 *  - Section 18: MILP Regression Suite (Cold vs Warm Starts in B&B)
 *  - Section 19: Performance & Telemetry Validation
 *  - Section 20: Ill-conditioned and Adversarial Child Nodes
 *  - Section 21: Real MIP Instances Verified with Independent Verifier
 */

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "../test_framework.hpp"
#include "algorithms/lp/basis.hpp"
#include "algorithms/lp/dual_simplex.hpp"
#include "algorithms/lp/lp_types.hpp"
#include "algorithms/lp/revised_simplex.hpp"
#include "algorithms/lp/simplex.hpp"
#include "algorithms/lp/standard_form.hpp"
#include "algorithms/milp/branch_and_bound.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"
#include "verification/milp_verifier.hpp"
#include "verification/verifier.hpp"

using namespace kavach;
using namespace kavach::model;
using namespace kavach::algorithms::lp;
using namespace kavach::algorithms::milp;
using namespace kavach::verification;

namespace {

/**
 * Base 2-variable LP:
 *  max 3*x1 + 2*x2
 *  s.t. 2*x1 + x2 <= 10
 *       x1 + 2*x2 <= 8
 *       x1, x2 >= 0
 * Optimal: x1 = 4, x2 = 2, obj = 16
 */
CanonicalModel build_base_2d_lp() {
    CanonicalModel m("base_2d_lp");
    auto x1 = m.add_variable("x1", VariableType::CONTINUOUS, 0.0, 10.0);
    auto x2 = m.add_variable("x2", VariableType::CONTINUOUS, 0.0, 10.0);
    m.set_objective_sense(ObjectiveSense::MAXIMIZE);
    m.add_objective_linear_term(x1, 3.0);
    m.add_objective_linear_term(x2, 2.0);

    SparseLinearExpression c1;
    c1.add_term(x1, 2.0);
    c1.add_term(x2, 1.0);
    m.add_constraint("c1", std::move(c1), ConstraintSense::LESS_EQUAL, 10.0);

    SparseLinearExpression c2;
    c2.add_term(x1, 1.0);
    c2.add_term(x2, 2.0);
    m.add_constraint("c2", std::move(c2), ConstraintSense::LESS_EQUAL, 8.0);

    return m;
}

/**
 * Base LP with truly fractional optimal solution:
 *  max x1 + x2
 *  s.t. 2*x1 + x2 <= 7
 *       x1 + 2*x2 <= 7
 *       x1, x2 in [0, 10]
 * Optimal: x1 = 7/3 (~2.333), x2 = 7/3 (~2.333), obj = 14/3 (~4.667)
 */
CanonicalModel build_fractional_3d_lp() {
    CanonicalModel m("fractional_3d_lp");
    auto x1 = m.add_variable("x1", VariableType::CONTINUOUS, 0.0, 10.0);
    auto x2 = m.add_variable("x2", VariableType::CONTINUOUS, 0.0, 10.0);
    m.set_objective_sense(ObjectiveSense::MAXIMIZE);
    m.add_objective_linear_term(x1, 1.0);
    m.add_objective_linear_term(x2, 1.0);

    SparseLinearExpression c1;
    c1.add_term(x1, 2.0);
    c1.add_term(x2, 1.0);
    m.add_constraint("c1", std::move(c1), ConstraintSense::LESS_EQUAL, 7.0);

    SparseLinearExpression c2;
    c2.add_term(x1, 1.0);
    c2.add_term(x2, 2.0);
    m.add_constraint("c2", std::move(c2), ConstraintSense::LESS_EQUAL, 7.0);

    return m;
}

/**
 * 6-item Knapsack Problem:
 *  max 10*x1 + 8*x2 + 7*x3 + 5*x4 + 3*x5 + 1*x6
 *  s.t. 7*x1 + 5*x2 + 4*x3 + 3*x4 + 2*x5 + 1*x6 <= 10
 *  all binary
 */
CanonicalModel build_knapsack_6item() {
    CanonicalModel m("knapsack_6item");
    std::vector<double> vals = {10.0, 8.0, 7.0, 5.0, 3.0, 1.0};
    std::vector<double> wts = {7.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    std::vector<VarId> vars;
    m.set_objective_sense(ObjectiveSense::MAXIMIZE);

    for (std::size_t i = 0; i < vals.size(); ++i) {
        auto v = m.add_binary_variable("x" + std::to_string(i + 1));
        vars.push_back(v);
        m.add_objective_linear_term(v, vals[i]);
    }

    SparseLinearExpression cap;
    for (std::size_t i = 0; i < wts.size(); ++i) {
        cap.add_term(vars[i], wts[i]);
    }
    m.add_constraint("capacity", std::move(cap), ConstraintSense::LESS_EQUAL, 10.0);
    return m;
}

}  // namespace

// ============================================================================
// Section 16: Dual Simplex Unit Tests (A through N)
// ============================================================================

// Test A: Single upper bound tightening (x1 <= 3 from parent x1=4, x2=2)
KAVACH_TEST(dual_simplex_test_a_upper_bound_tightening) {
    auto parent = build_base_2d_lp();
    RevisedSimplex solver;
    LPSolverOptions opts;
    opts.preserve_bounds_structure = true;
    opts.enable_presolve = false;

    auto parent_sol = solver.solve(parent, opts);
    KAVACH_ASSERT(parent_sol.status == core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(parent_sol.basis.has_value());
    KAVACH_ASSERT_NEAR(parent_sol.objective_value, 16.0, 1e-6);

    // Create child with tightened upper bound x1 <= 3
    auto child = parent;
    child.set_variable_bounds(0, 0.0, 3.0);

    auto warm_sol = solver.solve_warm_start(child, *parent_sol.basis, opts);
    KAVACH_ASSERT(warm_sol.status == core::SolverStatus::OPTIMAL);
    // At x1=3: 2(3)+x2 <= 10 => x2 <= 4; 3+2*x2 <= 8 => x2 <= 2.5.
    // Optimal: x1 = 3, x2 = 2.5, obj = 3*3 + 2*2.5 = 14.0
    KAVACH_ASSERT_NEAR(warm_sol.objective_value, 14.0, 1e-6);
    KAVACH_ASSERT_NEAR(warm_sol.primal_solution[0], 3.0, 1e-6);
    KAVACH_ASSERT_NEAR(warm_sol.primal_solution[1], 2.5, 1e-6);
}

// Test B: Single lower bound tightening (x2 >= 3 from parent x1=4, x2=2)
KAVACH_TEST(dual_simplex_test_b_lower_bound_tightening) {
    auto parent = build_base_2d_lp();
    RevisedSimplex solver;
    LPSolverOptions opts;
    opts.preserve_bounds_structure = true;
    opts.enable_presolve = false;

    auto parent_sol = solver.solve(parent, opts);
    KAVACH_ASSERT(parent_sol.basis.has_value());

    // Create child with lower bound x2 >= 3
    auto child = parent;
    child.set_variable_bounds(1, 3.0, 10.0);

    auto warm_sol = solver.solve_warm_start(child, *parent_sol.basis, opts);
    KAVACH_ASSERT(warm_sol.status == core::SolverStatus::OPTIMAL);
    // x1 + 2(3) <= 8 => x1 <= 2; 2(2) + 3 <= 10 (satisfied)
    // Optimal: x1 = 2, x2 = 3, obj = 3*2 + 2*3 = 12.0
    KAVACH_ASSERT_NEAR(warm_sol.objective_value, 12.0, 1e-6);
    KAVACH_ASSERT_NEAR(warm_sol.primal_solution[0], 2.0, 1e-6);
    KAVACH_ASSERT_NEAR(warm_sol.primal_solution[1], 3.0, 1e-6);
}

// Test C: Binary branch down (x1 <= floor(x1))
KAVACH_TEST(dual_simplex_test_c_branch_down) {
    auto parent = build_fractional_3d_lp();
    RevisedSimplex solver;
    LPSolverOptions opts;
    opts.preserve_bounds_structure = true;
    opts.enable_presolve = false;

    auto parent_sol = solver.solve(parent, opts);
    KAVACH_ASSERT(parent_sol.status == core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(parent_sol.basis.has_value());

    double x1_val = parent_sol.primal_solution[0];
    double floor_x1 = std::floor(x1_val);

    auto left_child = parent;
    left_child.set_variable_bounds(0, 0.0, floor_x1);

    auto child_sol = solver.solve_warm_start(left_child, *parent_sol.basis, opts);
    KAVACH_ASSERT(child_sol.status == core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(child_sol.primal_solution[0] <= floor_x1 + 1e-6);
    KAVACH_ASSERT(child_sol.objective_value <= parent_sol.objective_value + 1e-6);
}

// Test D: Binary branch up (x1 >= ceil(x1))
KAVACH_TEST(dual_simplex_test_d_branch_up) {
    auto parent = build_fractional_3d_lp();
    RevisedSimplex solver;
    LPSolverOptions opts;
    opts.preserve_bounds_structure = true;
    opts.enable_presolve = false;

    auto parent_sol = solver.solve(parent, opts);
    KAVACH_ASSERT(parent_sol.status == core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(parent_sol.basis.has_value());

    double x1_val = parent_sol.primal_solution[0];
    double ceil_x1 = std::ceil(x1_val);

    auto right_child = parent;
    right_child.set_variable_bounds(0, ceil_x1, 10.0);

    auto child_sol = solver.solve_warm_start(right_child, *parent_sol.basis, opts);
    KAVACH_ASSERT(child_sol.status == core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(child_sol.primal_solution[0] >= ceil_x1 - 1e-6);
    KAVACH_ASSERT(child_sol.objective_value <= parent_sol.objective_value + 1e-6);
}

// Test E: Multiple simultaneous bound changes
KAVACH_TEST(dual_simplex_test_e_multiple_bound_changes) {
    auto parent = build_base_2d_lp();
    RevisedSimplex solver;
    LPSolverOptions opts;
    opts.preserve_bounds_structure = true;
    opts.enable_presolve = false;

    auto parent_sol = solver.solve(parent, opts);
    KAVACH_ASSERT(parent_sol.basis.has_value());

    auto child = parent;
    child.set_variable_bounds(0, 1.0, 3.0);  // Tighten both lower and upper bounds of x1
    child.set_variable_bounds(1, 0.5, 2.0);  // Tighten both bounds of x2

    auto warm_sol = solver.solve_warm_start(child, *parent_sol.basis, opts);
    KAVACH_ASSERT(warm_sol.status == core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(warm_sol.primal_solution[0] >= 1.0 - 1e-6 &&
                  warm_sol.primal_solution[0] <= 3.0 + 1e-6);
    KAVACH_ASSERT(warm_sol.primal_solution[1] >= 0.5 - 1e-6 &&
                  warm_sol.primal_solution[1] <= 2.0 + 1e-6);
}

// Test F: Dual feasibility preservation
KAVACH_TEST(dual_simplex_test_f_dual_feasibility_preservation) {
    auto parent = build_base_2d_lp();
    RevisedSimplex solver;
    LPSolverOptions opts;
    opts.preserve_bounds_structure = true;
    opts.enable_presolve = false;

    auto parent_sol = solver.solve(parent, opts);
    KAVACH_ASSERT(parent_sol.basis.has_value());

    auto child = parent;
    child.set_variable_bounds(0, 0.0, 3.5);

    StandardFormLP sf(child, opts.tolerances, opts.preserve_bounds_structure);
    SimplexBasis basis;
    basis.load_state(*parent_sol.basis);
    bool fact_ok = basis.factorize(sf.A_csc(), opts.tolerances.pivot_tol);
    KAVACH_ASSERT(fact_ok);

    // Compute dual solution y = c_B * B^{-1}
    std::vector<double> c_B(sf.num_rows(), 0.0);
    for (std::size_t i = 0; i < sf.num_rows(); ++i) {
        c_B[i] = sf.c()[basis.basis_vars()[i]];
    }
    auto y = basis.solve_dual(c_B);

    // Verify dual reduced costs: d = c - A^T * y >= -tol for nonbasic vars
    auto a_t_y = sf.A_csc().spmv_transpose(y);
    for (std::size_t j = 0; j < sf.num_cols(); ++j) {
        if (basis.is_basic(j))
            continue;
        double d_j = sf.c()[j] - a_t_y[j];
        KAVACH_ASSERT(d_j >= -opts.tolerances.dual_feasibility_tol);
    }
}

// Test G: Infeasible child detection (Farkas ray)
KAVACH_TEST(dual_simplex_test_g_infeasible_child_detection) {
    auto parent = build_base_2d_lp();
    RevisedSimplex solver;
    LPSolverOptions opts;
    opts.preserve_bounds_structure = true;
    opts.enable_presolve = false;

    auto parent_sol = solver.solve(parent, opts);
    KAVACH_ASSERT(parent_sol.basis.has_value());

    // Impose contradictory bounds: x1 >= 6 (since 2*x1 + x2 <= 10 => x1 <= 5)
    auto child = parent;
    child.set_variable_bounds(0, 6.0, 10.0);

    auto warm_sol = solver.solve_warm_start(child, *parent_sol.basis, opts);
    KAVACH_ASSERT(warm_sol.status == core::SolverStatus::INFEASIBLE);
}

// Test H: Deterministic pivot selection
KAVACH_TEST(dual_simplex_test_h_deterministic_pivots) {
    auto parent = build_base_2d_lp();
    RevisedSimplex solver;
    LPSolverOptions opts;
    opts.preserve_bounds_structure = true;
    opts.enable_presolve = false;

    auto parent_sol = solver.solve(parent, opts);
    auto child = parent;
    child.set_variable_bounds(0, 0.0, 2.5);

    auto sol1 = solver.solve_warm_start(child, *parent_sol.basis, opts);
    auto sol2 = solver.solve_warm_start(child, *parent_sol.basis, opts);

    KAVACH_ASSERT(sol1.status == sol2.status);
    KAVACH_ASSERT_EQUAL(sol1.iterations, sol2.iterations);
    KAVACH_ASSERT_NEAR(sol1.objective_value, sol2.objective_value, 1e-12);
    for (std::size_t i = 0; i < sol1.primal_solution.size(); ++i) {
        KAVACH_ASSERT_NEAR(sol1.primal_solution[i], sol2.primal_solution[i], 1e-12);
    }
}

// Test I: Rejection of invalid BasisState
KAVACH_TEST(dual_simplex_test_i_invalid_basis_rejection) {
    auto model = build_base_2d_lp();
    RevisedSimplex solver;
    LPSolverOptions opts;

    BasisState invalid_basis;
    invalid_basis.is_valid = false;

    auto sol = solver.solve_warm_start(model, invalid_basis, opts);
    KAVACH_ASSERT(sol.status == core::SolverStatus::NUMERICAL_FAILURE);
}

// Test J: Structural dimension mismatch rejection
KAVACH_TEST(dual_simplex_test_j_dimension_mismatch_rejection) {
    auto model = build_base_2d_lp();
    RevisedSimplex solver;
    LPSolverOptions opts;
    opts.preserve_bounds_structure = true;

    auto parent_sol = solver.solve(model, opts);
    KAVACH_ASSERT(parent_sol.basis.has_value());

    // Corrupt dimensions in basis state
    auto bad_basis = *parent_sol.basis;
    bad_basis.num_rows += 1;

    auto sol = solver.solve_warm_start(model, bad_basis, opts);
    KAVACH_ASSERT(sol.status == core::SolverStatus::NUMERICAL_FAILURE);
}

// Test K: Rejection of QP models
KAVACH_TEST(dual_simplex_test_k_qp_model_rejection) {
    CanonicalModel qp("qp_model");
    auto x = qp.add_variable("x", VariableType::CONTINUOUS, 0.0, 10.0);
    qp.add_objective_linear_term(x, 1.0);
    qp.add_objective_quadratic_term(x, x, 2.0);  // Quadratic term makes it QP

    RevisedSimplex solver;
    BasisState dummy;
    dummy.is_valid = true;
    dummy.num_rows = 1;
    dummy.num_cols = 1;

    auto sol = solver.solve_warm_start(qp, dummy, LPSolverOptions{});
    KAVACH_ASSERT(sol.status == core::SolverStatus::NUMERICAL_FAILURE);
}

// Test L: Basis export & load state round-trip
KAVACH_TEST(dual_simplex_test_l_basis_round_trip) {
    auto model = build_base_2d_lp();
    LPSolverOptions opts;
    opts.preserve_bounds_structure = true;

    RevisedSimplex solver;
    auto sol = solver.solve(model, opts);
    KAVACH_ASSERT(sol.basis.has_value());

    BasisState state = *sol.basis;
    KAVACH_ASSERT(state.is_valid);

    SimplexBasis basis2;
    basis2.load_state(state);
    KAVACH_ASSERT_EQUAL(state.basis_vars.size(), basis2.basis_vars().size());
    for (std::size_t i = 0; i < state.basis_vars.size(); ++i) {
        KAVACH_ASSERT_EQUAL(state.basis_vars[i], basis2.basis_vars()[i]);
    }

    BasisState re_exported = basis2.export_state();
    KAVACH_ASSERT_EQUAL(state.basis_vars.size(), re_exported.basis_vars.size());
}

// Test M: Maximum iterations limit reached
KAVACH_TEST(dual_simplex_test_m_iteration_limit) {
    auto parent = build_base_2d_lp();
    RevisedSimplex solver;
    LPSolverOptions opts;
    opts.preserve_bounds_structure = true;
    opts.enable_presolve = false;

    auto parent_sol = solver.solve(parent, opts);
    auto child = parent;
    child.set_variable_bounds(0, 0.0, 1.0);

    // Set max_iterations = 0 to trigger iteration limit
    LPSolverOptions tight_opts = opts;
    tight_opts.max_iterations = 0;

    auto sol = solver.solve_warm_start(child, *parent_sol.basis, tight_opts);
    KAVACH_ASSERT(sol.status == core::SolverStatus::LIMIT_REACHED ||
                  sol.status == core::SolverStatus::NUMERICAL_FAILURE);
}

// Test N: Singular basis fallback
KAVACH_TEST(dual_simplex_test_n_singular_basis_fallback) {
    auto model = build_base_2d_lp();
    RevisedSimplex solver;
    LPSolverOptions opts;
    opts.preserve_bounds_structure = true;

    auto parent_sol = solver.solve(model, opts);
    KAVACH_ASSERT(parent_sol.basis.has_value());

    // Corrupt basis state to make basis columns linearly dependent (repeat index 0)
    auto singular_basis = *parent_sol.basis;
    for (auto& b : singular_basis.basis_vars) {
        b = 0;  // All columns identical => singular
    }

    auto sol = solver.solve_warm_start(model, singular_basis, opts);
    KAVACH_ASSERT(sol.status == core::SolverStatus::NUMERICAL_FAILURE);
}

// ============================================================================
// Section 17: Warm vs Cold Mathematical Equivalence
// ============================================================================

KAVACH_TEST(warm_vs_cold_mathematical_equivalence) {
    auto parent = build_base_2d_lp();
    RevisedSimplex solver;
    LPSolverOptions opts;
    opts.preserve_bounds_structure = true;
    opts.enable_presolve = false;

    auto parent_sol = solver.solve(parent, opts);
    KAVACH_ASSERT(parent_sol.status == core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(parent_sol.basis.has_value());

    std::vector<std::pair<double, double>> bound_variations = {
        {0.0, 3.0},  // x1 <= 3
        {0.0, 2.0},  // x1 <= 2
        {1.0, 3.5},  // 1 <= x1 <= 3.5
        {0.0, 1.0},  // x1 <= 1
    };

    for (const auto& b : bound_variations) {
        auto child = parent;
        child.set_variable_bounds(0, b.first, b.second);

        auto cold_sol = solver.solve(child, opts);
        auto warm_sol = solver.solve_warm_start(child, *parent_sol.basis, opts);

        KAVACH_ASSERT(cold_sol.status == warm_sol.status);
        if (cold_sol.status == core::SolverStatus::OPTIMAL) {
            KAVACH_ASSERT_NEAR(cold_sol.objective_value, warm_sol.objective_value, 1e-6);
            for (std::size_t i = 0; i < cold_sol.primal_solution.size(); ++i) {
                KAVACH_ASSERT_NEAR(cold_sol.primal_solution[i], warm_sol.primal_solution[i], 1e-5);
            }

            // Verify with LPVerifier
            LPVerifier verifier;
            auto vr =
                verifier.verify(child, warm_sol.primal_solution, warm_sol.dual_multipliers,
                                warm_sol.reduced_costs, warm_sol.objective_value, warm_sol.status);
            KAVACH_ASSERT(vr.primal_feasibility.passed);
            KAVACH_ASSERT(vr.dual_feasibility.passed);
            KAVACH_ASSERT(vr.complementary_slackness.passed);
        }
    }
}

// ============================================================================
// Section 18: MILP Regression Suite (Cold vs Warm Starts in B&B)
// ============================================================================

KAVACH_TEST(milp_regression_knapsack_binary) {
    CanonicalModel m("knapsack_binary");
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
    m.add_constraint("capacity", std::move(e), ConstraintSense::LESS_EQUAL, 5.0);

    // 1. Solve with warm starts disabled (pure cold start)
    MILPSolverOptions opts_cold;
    opts_cold.enable_warm_starts = false;
    opts_cold.cut_options.strategy = CutStrategy::OFF;
    BranchAndBound solver_cold(opts_cold);
    auto sol_cold = solver_cold.solve(m);

    // 2. Solve with warm starts enabled
    MILPSolverOptions opts_warm;
    opts_warm.enable_warm_starts = true;
    opts_warm.cut_options.strategy = CutStrategy::OFF;
    BranchAndBound solver_warm(opts_warm);
    auto sol_warm = solver_warm.solve(m);

    KAVACH_ASSERT(sol_cold.status == core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(sol_warm.status == core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(sol_cold.best_objective, sol_warm.best_objective, 1e-6);
    KAVACH_ASSERT_NEAR(sol_warm.best_objective, 11.0, 1e-6);

    // Verify warm start telemetry was populated
    KAVACH_ASSERT(sol_warm.telemetry.warm_start_attempts > 0);
    KAVACH_ASSERT(sol_warm.telemetry.warm_start_successes > 0);
    KAVACH_ASSERT(sol_warm.telemetry.basis_transfer_count > 0);
}

KAVACH_TEST(milp_regression_mixed_integer) {
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

    MILPSolverOptions opts_cold;
    opts_cold.enable_warm_starts = false;
    BranchAndBound solver_cold(opts_cold);
    auto sol_cold = solver_cold.solve(m);

    MILPSolverOptions opts_warm;
    opts_warm.enable_warm_starts = true;
    BranchAndBound solver_warm(opts_warm);
    auto sol_warm = solver_warm.solve(m);

    KAVACH_ASSERT(sol_cold.status == core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(sol_warm.status == core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(sol_cold.best_objective, sol_warm.best_objective, 1e-6);
    KAVACH_ASSERT_NEAR(sol_warm.best_objective, 12.25, 1e-6);
}

KAVACH_TEST(milp_regression_infeasible_model) {
    CanonicalModel m("int_infeasible");
    auto x = m.add_integer_variable("x", 0.0, 100.0);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x, 1.0);
    SparseLinearExpression e1;
    e1.add_term(x, 1.0);
    m.add_constraint("lb", std::move(e1), ConstraintSense::GREATER_EQUAL, 0.1);
    SparseLinearExpression e2;
    e2.add_term(x, 1.0);
    m.add_constraint("ub", std::move(e2), ConstraintSense::LESS_EQUAL, 0.9);

    MILPSolverOptions opts_warm;
    opts_warm.enable_warm_starts = true;
    BranchAndBound solver(opts_warm);
    auto sol = solver.solve(m);

    KAVACH_ASSERT(sol.status == core::SolverStatus::INFEASIBLE);
}

// ============================================================================
// Section 19: Performance & Telemetry Validation
// ============================================================================

KAVACH_TEST(dual_simplex_performance_knapsack_6item) {
    auto m = build_knapsack_6item();

    MILPSolverOptions opts_cold;
    opts_cold.enable_warm_starts = false;
    BranchAndBound solver_cold(opts_cold);
    auto start_cold = std::chrono::steady_clock::now();
    auto sol_cold = solver_cold.solve(m);
    auto dur_cold =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_cold)
            .count();

    MILPSolverOptions opts_warm;
    opts_warm.enable_warm_starts = true;
    BranchAndBound solver_warm(opts_warm);
    auto start_warm = std::chrono::steady_clock::now();
    auto sol_warm = solver_warm.solve(m);
    auto dur_warm =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_warm)
            .count();

    KAVACH_ASSERT(sol_cold.status == core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(sol_warm.status == core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(sol_cold.best_objective, sol_warm.best_objective, 1e-6);

    // Warm start telemetry verification
    KAVACH_ASSERT(sol_warm.telemetry.warm_start_attempts > 0);
    KAVACH_ASSERT(sol_warm.telemetry.warm_start_successes > 0);
    double rate = static_cast<double>(sol_warm.telemetry.warm_start_successes) /
                  static_cast<double>(sol_warm.telemetry.warm_start_attempts);
    KAVACH_ASSERT(rate > 0.5);

    std::cout << "[INFO] 6-Item Knapsack: Cold = " << dur_cold << " ms ("
              << sol_cold.telemetry.cold_start_iterations << " cold iters), "
              << "Warm = " << dur_warm << " ms (" << sol_warm.telemetry.dual_simplex_iterations
              << " dual iters), " << "Warm Success Rate = " << (rate * 100.0) << "%\n";
}

// ============================================================================
// Section 20: Ill-conditioned and Adversarial Child Nodes
// ============================================================================

KAVACH_TEST(adversarial_near_degenerate_child_bounds) {
    auto parent = build_base_2d_lp();
    RevisedSimplex solver;
    LPSolverOptions opts;
    opts.preserve_bounds_structure = true;
    opts.enable_presolve = false;

    auto parent_sol = solver.solve(parent, opts);
    KAVACH_ASSERT(parent_sol.status == core::SolverStatus::OPTIMAL);

    // Almost identical lower and upper bound (width 1e-5)
    auto child = parent;
    child.set_variable_bounds(0, 3.99999, 4.0);

    auto warm_sol = solver.solve_warm_start(child, *parent_sol.basis, opts);
    KAVACH_ASSERT(warm_sol.status == core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(warm_sol.primal_solution[0] >= 3.99999 - 1e-6);
    KAVACH_ASSERT(warm_sol.primal_solution[0] <= 4.0 + 1e-6);
}

KAVACH_TEST(adversarial_high_aspect_ratio_coefficients) {
    CanonicalModel m("high_aspect");
    auto x1 = m.add_variable("x1", VariableType::CONTINUOUS, 0.0, 100.0);
    auto x2 = m.add_variable("x2", VariableType::CONTINUOUS, 0.0, 100.0);
    m.set_objective_sense(ObjectiveSense::MAXIMIZE);
    m.add_objective_linear_term(x1, 1e4);
    m.add_objective_linear_term(x2, 1e-2);

    SparseLinearExpression c1;
    c1.add_term(x1, 1e4);
    c1.add_term(x2, 1e-2);
    m.add_constraint("c1", std::move(c1), ConstraintSense::LESS_EQUAL, 1e4);

    RevisedSimplex solver;
    LPSolverOptions opts;
    opts.preserve_bounds_structure = true;
    opts.enable_presolve = false;

    auto parent_sol = solver.solve(m, opts);
    KAVACH_ASSERT(parent_sol.status == core::SolverStatus::OPTIMAL);

    // Tighten upper bound on x1
    auto child = m;
    child.set_variable_bounds(0, 0.0, 0.5);

    auto warm_sol = solver.solve_warm_start(child, *parent_sol.basis, opts);
    KAVACH_ASSERT(warm_sol.status == core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(warm_sol.primal_solution[0], 0.5, 1e-5);
}

// ============================================================================
// Section 21: Real MIP Instances Verified with Independent Verifier
// ============================================================================

KAVACH_TEST(real_mip_knapsack_independent_verification) {
    auto m = build_knapsack_6item();

    MILPSolverOptions opts;
    opts.enable_warm_starts = true;
    BranchAndBound solver(opts);
    auto sol = solver.solve(m);

    KAVACH_ASSERT(sol.status == core::SolverStatus::OPTIMAL);

    // Independent MILP verification layer [Y]
    MILPVerifier verifier;
    auto rpt =
        verifier.verify(m, sol.incumbent_solution, sol.best_objective, sol.best_bound, sol.status);
    KAVACH_ASSERT(rpt.fully_verified());
    KAVACH_ASSERT(rpt.primal_feasibility.passed);
    KAVACH_ASSERT(rpt.integrality.passed);
}
