/**
 * @file test_phase9_qp.cpp
 * @brief Comprehensive unit and regression test suite for Kavach-Opt Convex QP Solver.
 *
 * Covers:
 *  - Test A: Basic unconstrained convex QP
 *  - Test B: Equality-constrained QP
 *  - Test C: <= inequality QP
 *  - Test D: >= inequality QP
 *  - Test E: Mixed constraints
 *  - Test F: Variable lower bounds
 *  - Test G: Variable upper bounds
 *  - Test H: Boxed variables
 *  - Test I: Free variables
 *  - Test J: Fixed variables
 *  - Test K: Diagonal Hessian with wide scaling
 *  - Test L: Dense-small symmetric Hessian
 *  - Test M: Off-diagonal quadratic terms
 *  - Test N: Objective constant offset
 *  - Test O: Multiple active-set changes
 *  - Test P: Constraint activation (blocking step < 1)
 *  - Test Q: Constraint removal (negative multiplier drop)
 *  - Test R: Degenerate active set (redundant constraints)
 *  - Test S: Numerically challenging QP
 *  - Test T: Non-PSD Hessian rejection
 *  - Test U: KKT residual failure injection
 *  - Test V: Objective corruption injection
 *  - Test W: Deterministic repeated solve
 *  - Test X: Iteration limit reached
 *  - Test Y: Time limit reached
 *  - Fault Injection: Primal bound violation caught by verifier
 *  - Fault Injection: Primal constraint violation caught by verifier
 *  - Fault Injection: NaN / Inf injection caught by verifier
 *  - Infeasible QP handling
 *  - Presolve integration with QP solver
 */

#include <cmath>
#include <limits>
#include <vector>

#include "../test_framework.hpp"
#include "algorithms/qp/active_set.hpp"
#include "core/status.hpp"
#include "model/model.hpp"
#include "verification/qp_verifier.hpp"
#include "verification/verifier.hpp"

using namespace kavach;
using namespace kavach::model;
using namespace kavach::algorithms::qp;
using namespace kavach::verification;

// ============================================================================
// Helper to build canonical model easily
// ============================================================================
namespace {

CanonicalModel make_simple_model(const std::string& name) {
    CanonicalModel model(name);
    model.set_objective_sense(ObjectiveSense::MINIMIZE);
    return model;
}

}  // namespace

// ============================================================================
// Test A: Basic unconstrained convex QP
// min f(x1, x2) = x1^2 + x2^2 - 2*x1 - 4*x2
// Unconstrained minimizer: x* = (1, 2), f(x*) = -5
// ============================================================================
KAVACH_TEST(test_qp_a_unconstrained) {
    auto model = make_simple_model("qp_unconstrained");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);

    model.set_objective_linear_coefficient(x1, -2.0);
    model.set_objective_linear_coefficient(x2, -4.0);
    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.add_objective_quadratic_term(x2, x2, 1.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    QPSolverOptions opts;
    opts.verify_with_independent_verifier = true;
    auto res = solver.solve(model, opts);

    KAVACH_ASSERT(res.is_optimal());
    KAVACH_ASSERT(res.verification_report.fully_verified());
    KAVACH_ASSERT_NEAR(res.primal_solution[x1], 1.0, 1e-5);
    KAVACH_ASSERT_NEAR(res.primal_solution[x2], 2.0, 1e-5);
    KAVACH_ASSERT_NEAR(res.objective_value, -5.0, 1e-5);
}

// ============================================================================
// Test B: Equality-constrained QP
// min x1^2 + x2^2  s.t.  x1 + x2 = 1
// Solution: x1 = 0.5, x2 = 0.5, obj = 0.5
// ============================================================================
KAVACH_TEST(test_qp_b_equality_constrained) {
    auto model = make_simple_model("qp_equality");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);

    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.add_objective_quadratic_term(x2, x2, 1.0);

    SparseLinearExpression expr;
    expr.add_term(x1, 1.0);
    expr.add_term(x2, 1.0);
    model.add_constraint("eq1", std::move(expr), ConstraintSense::EQUAL, 1.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);

    KAVACH_ASSERT(res.is_optimal());
    KAVACH_ASSERT(res.verification_report.fully_verified());
    KAVACH_ASSERT_NEAR(res.primal_solution[x1], 0.5, 1e-5);
    KAVACH_ASSERT_NEAR(res.primal_solution[x2], 0.5, 1e-5);
    KAVACH_ASSERT_NEAR(res.objective_value, 0.5, 1e-5);
}

// ============================================================================
// Test C: <= inequality QP (active at optimum)
// min x1^2 + x2^2  s.t.  x1 + x2 <= -1
// Solution: x1 = -0.5, x2 = -0.5, obj = 0.5
// ============================================================================
KAVACH_TEST(test_qp_c_less_equal_inequality) {
    auto model = make_simple_model("qp_le");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);

    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.add_objective_quadratic_term(x2, x2, 1.0);

    SparseLinearExpression expr;
    expr.add_term(x1, 1.0);
    expr.add_term(x2, 1.0);
    model.add_constraint("c1", std::move(expr), ConstraintSense::LESS_EQUAL, -1.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);

    KAVACH_ASSERT(res.is_optimal());
    KAVACH_ASSERT(res.verification_report.fully_verified());
    KAVACH_ASSERT_NEAR(res.primal_solution[x1], -0.5, 1e-5);
    KAVACH_ASSERT_NEAR(res.primal_solution[x2], -0.5, 1e-5);
    KAVACH_ASSERT_NEAR(res.objective_value, 0.5, 1e-5);
}

// ============================================================================
// Test D: >= inequality QP (active at optimum)
// min x1^2 + x2^2  s.t.  x1 + x2 >= 2
// Solution: x1 = 1.0, x2 = 1.0, obj = 2.0
// ============================================================================
KAVACH_TEST(test_qp_d_greater_equal_inequality) {
    auto model = make_simple_model("qp_ge");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);

    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.add_objective_quadratic_term(x2, x2, 1.0);

    SparseLinearExpression expr;
    expr.add_term(x1, 1.0);
    expr.add_term(x2, 1.0);
    model.add_constraint("c1", std::move(expr), ConstraintSense::GREATER_EQUAL, 2.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);

    KAVACH_ASSERT(res.is_optimal());
    KAVACH_ASSERT(res.verification_report.fully_verified());
    KAVACH_ASSERT_NEAR(res.primal_solution[x1], 1.0, 1e-5);
    KAVACH_ASSERT_NEAR(res.primal_solution[x2], 1.0, 1e-5);
    KAVACH_ASSERT_NEAR(res.objective_value, 2.0, 1e-5);
}

// ============================================================================
// Test E: Mixed constraints (<=, ==, >=)
// min x1^2 + x2^2 + x3^2
// s.t. x1 + x2 + x3 = 3
//      x1 >= 1.5
//      x2 <= 0.8
// ============================================================================
KAVACH_TEST(test_qp_e_mixed_constraints) {
    auto model = make_simple_model("qp_mixed");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);
    VarId x3 = model.add_variable("x3", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);

    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.add_objective_quadratic_term(x2, x2, 1.0);
    model.add_objective_quadratic_term(x3, x3, 1.0);

    // x1 + x2 + x3 = 3
    SparseLinearExpression eq;
    eq.add_term(x1, 1.0);
    eq.add_term(x2, 1.0);
    eq.add_term(x3, 1.0);
    model.add_constraint("eq", std::move(eq), ConstraintSense::EQUAL, 3.0);

    // x1 >= 1.5
    SparseLinearExpression ge;
    ge.add_term(x1, 1.0);
    model.add_constraint("ge", std::move(ge), ConstraintSense::GREATER_EQUAL, 1.5);

    // x2 <= 0.8
    SparseLinearExpression le;
    le.add_term(x2, 1.0);
    model.add_constraint("le", std::move(le), ConstraintSense::LESS_EQUAL, 0.8);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);

    KAVACH_ASSERT(res.is_optimal());
    KAVACH_ASSERT(res.verification_report.fully_verified());
    // x1 will be at least 1.5. Minimum of sum of squares under x1+x2+x3=3 with x1>=1.5, x2<=0.8
    // Unconstrained with x1=1.5: x2 + x3 = 1.5 -> x2 = 0.75, x3 = 0.75 <= 0.8.
    // So optimum is x1 = 1.5, x2 = 0.75, x3 = 0.75.
    KAVACH_ASSERT_NEAR(res.primal_solution[x1], 1.5, 1e-4);
    KAVACH_ASSERT_NEAR(res.primal_solution[x2], 0.75, 1e-4);
    KAVACH_ASSERT_NEAR(res.primal_solution[x3], 0.75, 1e-4);
    KAVACH_ASSERT_NEAR(res.objective_value, 1.5 * 1.5 + 0.75 * 0.75 + 0.75 * 0.75, 1e-4);
}

// ============================================================================
// Test F: Variable lower bounds
// min x1^2 + x2^2  s.t.  x1 >= 2, x2 >= 3
// ============================================================================
KAVACH_TEST(test_qp_f_variable_lower_bounds) {
    auto model = make_simple_model("qp_lb");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, 2.0, INFINITY_BOUND);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, 3.0, INFINITY_BOUND);

    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.add_objective_quadratic_term(x2, x2, 1.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);

    KAVACH_ASSERT(res.is_optimal());
    KAVACH_ASSERT(res.verification_report.fully_verified());
    KAVACH_ASSERT_NEAR(res.primal_solution[x1], 2.0, 1e-5);
    KAVACH_ASSERT_NEAR(res.primal_solution[x2], 3.0, 1e-5);
    KAVACH_ASSERT_NEAR(res.objective_value, 13.0, 1e-5);
}

// ============================================================================
// Test G: Variable upper bounds
// min (x1 - 5)^2 + (x2 - 5)^2 = x1^2 + x2^2 - 10*x1 - 10*x2 + 50
// s.t. x1 <= 2, x2 <= 1
// ============================================================================
KAVACH_TEST(test_qp_g_variable_upper_bounds) {
    auto model = make_simple_model("qp_ub");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, -INFINITY_BOUND, 2.0);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, -INFINITY_BOUND, 1.0);

    model.set_objective_offset(50.0);
    model.set_objective_linear_coefficient(x1, -10.0);
    model.set_objective_linear_coefficient(x2, -10.0);
    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.add_objective_quadratic_term(x2, x2, 1.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);

    KAVACH_ASSERT(res.is_optimal());
    KAVACH_ASSERT(res.verification_report.fully_verified());
    KAVACH_ASSERT_NEAR(res.primal_solution[x1], 2.0, 1e-5);
    KAVACH_ASSERT_NEAR(res.primal_solution[x2], 1.0, 1e-5);
    KAVACH_ASSERT_NEAR(res.objective_value, (2.0 - 5.0) * (2.0 - 5.0) + (1.0 - 5.0) * (1.0 - 5.0),
                       1e-5);
}

// ============================================================================
// Test H: Boxed variables
// min (x1 - 2)^2 + (x2 + 2)^2 = x1^2 + x2^2 - 4*x1 + 4*x2 + 8
// s.t. 0 <= x1 <= 1, 0 <= x2 <= 1
// Opt: x1 = 1, x2 = 0
// ============================================================================
KAVACH_TEST(test_qp_h_boxed_variables) {
    auto model = make_simple_model("qp_box");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 1.0);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, 1.0);

    model.set_objective_offset(8.0);
    model.set_objective_linear_coefficient(x1, -4.0);
    model.set_objective_linear_coefficient(x2, 4.0);
    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.add_objective_quadratic_term(x2, x2, 1.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);

    KAVACH_ASSERT(res.is_optimal());
    KAVACH_ASSERT(res.verification_report.fully_verified());
    KAVACH_ASSERT_NEAR(res.primal_solution[x1], 1.0, 1e-5);
    KAVACH_ASSERT_NEAR(res.primal_solution[x2], 0.0, 1e-5);
    KAVACH_ASSERT_NEAR(res.objective_value, 1.0 + 4.0, 1e-5);
}

// ============================================================================
// Test I: Free variables (-inf to +inf)
// min 2*x1^2 + 3*x2^2 - 4*x1 + 6*x2
// Opt: x1 = 1, x2 = -1
// ============================================================================
KAVACH_TEST(test_qp_i_free_variables) {
    auto model = make_simple_model("qp_free");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);

    model.set_objective_linear_coefficient(x1, -4.0);
    model.set_objective_linear_coefficient(x2, 6.0);
    model.add_objective_quadratic_term(x1, x1, 2.0);
    model.add_objective_quadratic_term(x2, x2, 3.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);

    KAVACH_ASSERT(res.is_optimal());
    KAVACH_ASSERT(res.verification_report.fully_verified());
    KAVACH_ASSERT_NEAR(res.primal_solution[x1], 1.0, 1e-5);
    KAVACH_ASSERT_NEAR(res.primal_solution[x2], -1.0, 1e-5);
    KAVACH_ASSERT_NEAR(res.objective_value, 2.0 * 1 - 4.0 * 1 + 3.0 * 1 - 6.0 * 1, 1e-5);
}

// ============================================================================
// Test J: Fixed variables (lb == ub)
// min x1^2 + x2^2 - 4*x2  s.t.  x1 = 3.0
// Opt: x1 = 3.0, x2 = 2.0
// ============================================================================
KAVACH_TEST(test_qp_j_fixed_variables) {
    auto model = make_simple_model("qp_fixed");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, 3.0, 3.0);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);

    model.set_objective_linear_coefficient(x2, -4.0);
    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.add_objective_quadratic_term(x2, x2, 1.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);

    KAVACH_ASSERT(res.is_optimal());
    KAVACH_ASSERT(res.verification_report.fully_verified());
    KAVACH_ASSERT_NEAR(res.primal_solution[x1], 3.0, 1e-5);
    KAVACH_ASSERT_NEAR(res.primal_solution[x2], 2.0, 1e-5);
    KAVACH_ASSERT_NEAR(res.objective_value, 9.0 + 4.0 - 8.0, 1e-5);
}

// ============================================================================
// Test K: Diagonal Hessian with wide scaling
// min 100*x1^2 + 0.01*x2^2 - 200*x1 - 0.02*x2
// Opt: x1 = 1.0, x2 = 1.0
// ============================================================================
KAVACH_TEST(test_qp_k_diagonal_hessian_scaling) {
    auto model = make_simple_model("qp_diag_scaling");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);

    model.set_objective_linear_coefficient(x1, -200.0);
    model.set_objective_linear_coefficient(x2, -0.02);
    model.add_objective_quadratic_term(x1, x1, 100.0);
    model.add_objective_quadratic_term(x2, x2, 0.01);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);

    KAVACH_ASSERT(res.is_optimal());
    KAVACH_ASSERT(res.verification_report.fully_verified());
    KAVACH_ASSERT_NEAR(res.primal_solution[x1], 1.0, 1e-4);
    KAVACH_ASSERT_NEAR(res.primal_solution[x2], 1.0, 1e-4);
}

// ============================================================================
// Test L: Dense-small symmetric Hessian
// Q = [4 1 1; 1 3 1; 1 1 2]
// min 0.5 * x^T Q x - [1, 2, 3]^T x
// ============================================================================
KAVACH_TEST(test_qp_l_dense_symmetric_hessian) {
    auto model = make_simple_model("qp_dense_hessian");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);
    VarId x3 = model.add_variable("x3", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);

    model.set_objective_linear_coefficient(x1, -1.0);
    model.set_objective_linear_coefficient(x2, -2.0);
    model.set_objective_linear_coefficient(x3, -3.0);

    // Q_11 = 4 -> 0.5*4*x1^2 = 2*x1^2
    model.add_objective_quadratic_term(x1, x1, 2.0);
    model.add_objective_quadratic_term(x2, x2, 1.5);
    model.add_objective_quadratic_term(x3, x3, 1.0);

    // Off diagonals: Q_12 = 1 -> x1*x2
    model.add_objective_quadratic_term(x1, x2, 1.0);
    model.add_objective_quadratic_term(x1, x3, 1.0);
    model.add_objective_quadratic_term(x2, x3, 1.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);

    KAVACH_ASSERT(res.is_optimal());
    KAVACH_ASSERT(res.verification_report.fully_verified());

    // Check stationarity Q*x + c == 0:
    // 4*x1 + x2 + x3 = 1
    // x1 + 3*x2 + x3 = 2
    // x1 + x2 + 2*x3 = 3
    double x1_val = res.primal_solution[x1];
    double x2_val = res.primal_solution[x2];
    double x3_val = res.primal_solution[x3];
    KAVACH_ASSERT_NEAR(4 * x1_val + x2_val + x3_val, 1.0, 1e-4);
    KAVACH_ASSERT_NEAR(x1_val + 3 * x2_val + x3_val, 2.0, 1e-4);
    KAVACH_ASSERT_NEAR(x1_val + x2_val + 2 * x3_val, 3.0, 1e-4);
}

// ============================================================================
// Test M: Off-diagonal quadratic terms
// min x1^2 + 2*x1*x2 + 2*x2^2 - 2*x1 - 2*x2
// Q = [2 2; 2 4], det = 8 - 4 = 4 > 0 (strictly positive definite)
// Grad = [2*x1 + 2*x2 - 2, 2*x1 + 4*x2 - 2] = 0 -> x2 = 0, x1 = 1
// ============================================================================
KAVACH_TEST(test_qp_m_off_diagonal_quadratic_terms) {
    auto model = make_simple_model("qp_off_diag");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);

    model.set_objective_linear_coefficient(x1, -2.0);
    model.set_objective_linear_coefficient(x2, -2.0);
    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.add_objective_quadratic_term(x1, x2, 2.0);
    model.add_objective_quadratic_term(x2, x2, 2.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);

    KAVACH_ASSERT(res.is_optimal());
    KAVACH_ASSERT(res.verification_report.fully_verified());
    KAVACH_ASSERT_NEAR(res.primal_solution[x1], 1.0, 1e-5);
    KAVACH_ASSERT_NEAR(res.primal_solution[x2], 0.0, 1e-5);
    KAVACH_ASSERT_NEAR(res.objective_value, -1.0, 1e-5);
}

// ============================================================================
// Test N: Objective constant offset
// min x1^2 + x2^2 + 42.0
// Opt: x1 = 0, x2 = 0, obj = 42.0
// ============================================================================
KAVACH_TEST(test_qp_n_constant_offset) {
    auto model = make_simple_model("qp_offset");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);

    model.set_objective_offset(42.0);
    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.add_objective_quadratic_term(x2, x2, 1.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);

    KAVACH_ASSERT(res.is_optimal());
    KAVACH_ASSERT(res.verification_report.fully_verified());
    KAVACH_ASSERT_NEAR(res.primal_solution[x1], 0.0, 1e-5);
    KAVACH_ASSERT_NEAR(res.primal_solution[x2], 0.0, 1e-5);
    KAVACH_ASSERT_NEAR(res.objective_value, 42.0, 1e-5);
}

// ============================================================================
// Test O: Multiple active-set changes
// min (x1 - 4)^2 + (x2 - 4)^2
// s.t. x1 + x2 <= 4
//      x1 <= 2
//      x2 <= 3
// ============================================================================
KAVACH_TEST(test_qp_o_multiple_active_set_changes) {
    auto model = make_simple_model("qp_multi_changes");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 5.0);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, 5.0);

    model.set_objective_offset(32.0);
    model.set_objective_linear_coefficient(x1, -8.0);
    model.set_objective_linear_coefficient(x2, -8.0);
    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.add_objective_quadratic_term(x2, x2, 1.0);

    SparseLinearExpression c1;
    c1.add_term(x1, 1.0);
    c1.add_term(x2, 1.0);
    model.add_constraint("c1", std::move(c1), ConstraintSense::LESS_EQUAL, 4.0);

    SparseLinearExpression c2;
    c2.add_term(x1, 1.0);
    model.add_constraint("c2", std::move(c2), ConstraintSense::LESS_EQUAL, 2.0);

    SparseLinearExpression c3;
    c3.add_term(x2, 1.0);
    model.add_constraint("c3", std::move(c3), ConstraintSense::LESS_EQUAL, 3.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);

    KAVACH_ASSERT(res.is_optimal());
    KAVACH_ASSERT(res.verification_report.fully_verified());
    // Closest point in x1+x2<=4, x1<=2, x2<=3 to (4,4) is (2, 2).
    KAVACH_ASSERT_NEAR(res.primal_solution[x1], 2.0, 1e-4);
    KAVACH_ASSERT_NEAR(res.primal_solution[x2], 2.0, 1e-4);
    KAVACH_ASSERT_NEAR(res.objective_value, 8.0, 1e-4);
}

// ============================================================================
// Test P: Constraint activation (blocking step alpha < 1)
// min (x1 - 3)^2 + (x2 - 3)^2  s.t.  x1 + x2 <= 2, x1, x2 >= 0
// Unconstrained step from (0,0) towards (3,3) hits x1+x2=2 at alpha = 1/3
// ============================================================================
KAVACH_TEST(test_qp_p_constraint_activation) {
    auto model = make_simple_model("qp_blocking");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 5.0);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, 5.0);

    model.set_objective_offset(18.0);
    model.set_objective_linear_coefficient(x1, -6.0);
    model.set_objective_linear_coefficient(x2, -6.0);
    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.add_objective_quadratic_term(x2, x2, 1.0);

    SparseLinearExpression c;
    c.add_term(x1, 1.0);
    c.add_term(x2, 1.0);
    model.add_constraint("sum_le_2", std::move(c), ConstraintSense::LESS_EQUAL, 2.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);

    KAVACH_ASSERT(res.is_optimal());
    KAVACH_ASSERT(res.verification_report.fully_verified());
    KAVACH_ASSERT_NEAR(res.primal_solution[x1], 1.0, 1e-4);
    KAVACH_ASSERT_NEAR(res.primal_solution[x2], 1.0, 1e-4);
    KAVACH_ASSERT_NEAR(res.objective_value, 8.0, 1e-4);
}

// ============================================================================
// Test Q: Constraint removal (negative multiplier drop)
// min x1^2 + x2^2  s.t.  x1 + x2 >= 0, x1 >= 1
// Solution: x1 = 1, x2 = 0. x1 + x2 >= 0 is inactive and its multiplier is 0
// ============================================================================
KAVACH_TEST(test_qp_q_constraint_removal) {
    auto model = make_simple_model("qp_drop");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, 1.0, INFINITY_BOUND);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);

    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.add_objective_quadratic_term(x2, x2, 1.0);

    SparseLinearExpression c;
    c.add_term(x1, 1.0);
    c.add_term(x2, 1.0);
    model.add_constraint("sum_ge_0", std::move(c), ConstraintSense::GREATER_EQUAL, 0.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);

    KAVACH_ASSERT(res.is_optimal());
    KAVACH_ASSERT(res.verification_report.fully_verified());
    KAVACH_ASSERT_NEAR(res.primal_solution[x1], 1.0, 1e-5);
    KAVACH_ASSERT_NEAR(res.primal_solution[x2], 0.0, 1e-5);
    KAVACH_ASSERT_NEAR(res.objective_value, 1.0, 1e-5);
}

// ============================================================================
// Test R: Degenerate active set (redundant constraints at optimum)
// min x1^2 + x2^2  s.t.  x1 >= 1, 2*x1 >= 2, 3*x1 >= 3
// ============================================================================
KAVACH_TEST(test_qp_r_degenerate_active_set) {
    auto model = make_simple_model("qp_degenerate");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);

    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.add_objective_quadratic_term(x2, x2, 1.0);

    SparseLinearExpression c1;
    c1.add_term(x1, 1.0);
    model.add_constraint("c1", std::move(c1), ConstraintSense::GREATER_EQUAL, 1.0);

    SparseLinearExpression c2;
    c2.add_term(x1, 2.0);
    model.add_constraint("c2", std::move(c2), ConstraintSense::GREATER_EQUAL, 2.0);

    SparseLinearExpression c3;
    c3.add_term(x1, 3.0);
    model.add_constraint("c3", std::move(c3), ConstraintSense::GREATER_EQUAL, 3.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);

    KAVACH_ASSERT(res.is_optimal());
    KAVACH_ASSERT(res.verification_report.fully_verified());
    KAVACH_ASSERT_NEAR(res.primal_solution[x1], 1.0, 1e-4);
    KAVACH_ASSERT_NEAR(res.primal_solution[x2], 0.0, 1e-4);
    KAVACH_ASSERT_NEAR(res.objective_value, 1.0, 1e-4);
}

// ============================================================================
// Test S: Numerically challenging QP (ill-conditioned Hessian)
// min 1e4*x1^2 + 1e-2*x2^2 s.t. x1 + x2 >= 1
// ============================================================================
KAVACH_TEST(test_qp_s_numerically_challenging) {
    auto model = make_simple_model("qp_ill_conditioned");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);

    model.add_objective_quadratic_term(x1, x1, 1e4);
    model.add_objective_quadratic_term(x2, x2, 1e-2);

    SparseLinearExpression c;
    c.add_term(x1, 1.0);
    c.add_term(x2, 1.0);
    model.add_constraint("sum_ge_1", std::move(c), ConstraintSense::GREATER_EQUAL, 1.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);

    KAVACH_ASSERT(res.is_optimal());
    KAVACH_ASSERT(res.verification_report.fully_verified());
    // Almost all weight on x2 because 1e4 >> 1e-2
    // Analytical min: 2*1e4*x1 = 2*1e-2*x2 -> x1 = 1e-6 * x2.
    // x1 + x2 = 1 -> x2 approx 1.0, x1 approx 1e-6.
    KAVACH_ASSERT_NEAR(res.primal_solution[x2], 1.0, 1e-3);
    KAVACH_ASSERT_NEAR(res.primal_solution[x1], 0.0, 1e-3);
}

// ============================================================================
// Test T: Non-PSD rejection
// min -x1^2 + x2^2 (indefinite / non-convex)
// ============================================================================
KAVACH_TEST(test_qp_t_non_psd_rejection) {
    auto model = make_simple_model("qp_non_psd");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, -1.0, 1.0);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, -1.0, 1.0);

    model.add_objective_quadratic_term(x1, x1, -1.0);
    model.add_objective_quadratic_term(x2, x2, 1.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);

    KAVACH_ASSERT_EQUAL(res.status, core::SolverStatus::NUMERICAL_FAILURE);
    KAVACH_ASSERT(!res.is_optimal());
}

// ============================================================================
// Test U: KKT residual failure injection (corrupted multipliers)
// ============================================================================
KAVACH_TEST(test_qp_u_kkt_residual_failure_injection) {
    auto model = make_simple_model("qp_verifier_fault_stationarity");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, 1.0, INFINITY_BOUND);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, 1.0, INFINITY_BOUND);

    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.add_objective_quadratic_term(x2, x2, 1.0);

    SparseLinearExpression c;
    c.add_term(x1, 1.0);
    c.add_term(x2, 1.0);
    model.add_constraint("c", std::move(c), ConstraintSense::LESS_EQUAL, 3.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);
    KAVACH_ASSERT(res.is_optimal());

    // Corrupt dual multiplier
    res.dual_solution[0] += 50.0;

    QPVerifier verifier;
    auto report = verifier.verify(model, res.primal_solution, res.dual_solution,
                                  res.objective_value, res.status);
    KAVACH_ASSERT(!report.fully_verified());
    KAVACH_ASSERT(report.aggregate_complementarity_residual > 1.0 ||
                  report.aggregate_stationarity_residual > 1.0);
}

// ============================================================================
// Test V: Objective corruption injection
// ============================================================================
KAVACH_TEST(test_qp_v_objective_corruption_injection) {
    auto model = make_simple_model("qp_verifier_fault_objective");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 1.0);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, 1.0);

    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.add_objective_quadratic_term(x2, x2, 1.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);
    KAVACH_ASSERT(res.is_optimal());

    // Corrupt objective
    res.objective_value += 100.0;

    QPVerifier verifier;
    auto report = verifier.verify(model, res.primal_solution, res.dual_solution,
                                  res.objective_value, res.status);
    KAVACH_ASSERT(!report.fully_verified());
    KAVACH_ASSERT(report.objective_absolute_error > 10.0);
}

// ============================================================================
// Test W: Deterministic repeated solve
// ============================================================================
KAVACH_TEST(test_qp_w_deterministic_repeated_solve) {
    auto model = make_simple_model("qp_repeat");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 5.0);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, 5.0);

    model.set_objective_linear_coefficient(x1, -2.0);
    model.set_objective_linear_coefficient(x2, -3.0);
    model.add_objective_quadratic_term(x1, x1, 2.0);
    model.add_objective_quadratic_term(x1, x2, 0.5);
    model.add_objective_quadratic_term(x2, x2, 1.5);

    SparseLinearExpression c;
    c.add_term(x1, 1.0);
    c.add_term(x2, 2.0);
    model.add_constraint("c", std::move(c), ConstraintSense::LESS_EQUAL, 2.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto ref = solver.solve(model);
    KAVACH_ASSERT(ref.is_optimal());

    for (int rep = 0; rep < 5; ++rep) {
        auto run = solver.solve(model);
        KAVACH_ASSERT_EQUAL(run.status, ref.status);
        KAVACH_ASSERT_EQUAL(run.iterations, ref.iterations);
        KAVACH_ASSERT_EQUAL(run.objective_value, ref.objective_value);
        KAVACH_ASSERT_EQUAL(run.primal_solution.size(), ref.primal_solution.size());
        for (std::size_t i = 0; i < run.primal_solution.size(); ++i) {
            KAVACH_ASSERT_EQUAL(run.primal_solution[i], ref.primal_solution[i]);
        }
    }
}

// ============================================================================
// Test X: Iteration limit
// ============================================================================
KAVACH_TEST(test_qp_x_iteration_limit) {
    auto model = make_simple_model("qp_iter_limit");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 10.0);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, 10.0);

    model.set_objective_linear_coefficient(x1, -5.0);
    model.set_objective_linear_coefficient(x2, -5.0);
    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.add_objective_quadratic_term(x2, x2, 1.0);

    SparseLinearExpression c;
    c.add_term(x1, 1.0);
    c.add_term(x2, 1.0);
    model.add_constraint("c", std::move(c), ConstraintSense::LESS_EQUAL, 3.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    QPSolverOptions opts;
    opts.max_iterations = 1;
    auto res = solver.solve(model, opts);

    KAVACH_ASSERT_EQUAL(res.status, core::SolverStatus::LIMIT_REACHED);
}

// ============================================================================
// Test Y: Time limit
// ============================================================================
KAVACH_TEST(test_qp_y_time_limit) {
    auto model = make_simple_model("qp_time_limit");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 10.0);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, 10.0);

    model.set_objective_linear_coefficient(x1, -5.0);
    model.set_objective_linear_coefficient(x2, -5.0);
    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.add_objective_quadratic_term(x2, x2, 1.0);

    SparseLinearExpression c;
    c.add_term(x1, 1.0);
    c.add_term(x2, 1.0);
    model.add_constraint("c", std::move(c), ConstraintSense::LESS_EQUAL, 3.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    QPSolverOptions opts;
    opts.time_limit_seconds = 1e-9;
    auto res = solver.solve(model, opts);

    KAVACH_ASSERT_EQUAL(res.status, core::SolverStatus::LIMIT_REACHED);
}

// ============================================================================
// Fault Injection: Primal bound violation caught by verifier
// ============================================================================
KAVACH_TEST(test_qp_fault_bound_violation) {
    auto model = make_simple_model("qp_fault_bound");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 2.0);
    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);
    KAVACH_ASSERT(res.is_optimal());

    // Corrupt bound
    res.primal_solution[x1] = 5.0;  // exceeds upper bound 2.0

    QPVerifier verifier;
    auto report = verifier.verify(model, res.primal_solution, res.dual_solution,
                                  res.objective_value, res.status);
    KAVACH_ASSERT(!report.fully_verified());
    KAVACH_ASSERT(report.bound_feasibility.max_violation > 2.0);
}

// ============================================================================
// Fault Injection: Primal constraint violation caught by verifier
// ============================================================================
KAVACH_TEST(test_qp_fault_constraint_violation) {
    auto model = make_simple_model("qp_fault_con");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 10.0);
    model.add_objective_quadratic_term(x1, x1, 1.0);

    SparseLinearExpression c;
    c.add_term(x1, 1.0);
    model.add_constraint("c", std::move(c), ConstraintSense::LESS_EQUAL, 1.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);
    KAVACH_ASSERT(res.is_optimal());

    // Corrupt primal value violating constraint
    res.primal_solution[x1] = 4.0;

    QPVerifier verifier;
    auto report = verifier.verify(model, res.primal_solution, res.dual_solution,
                                  res.objective_value, res.status);
    KAVACH_ASSERT(!report.fully_verified());
    KAVACH_ASSERT(report.primal_feasibility.max_violation > 2.0);
}

// ============================================================================
// Fault Injection: NaN / Inf caught by verifier
// ============================================================================
KAVACH_TEST(test_qp_fault_nan_inf) {
    auto model = make_simple_model("qp_fault_nan");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 1.0);
    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);
    KAVACH_ASSERT(res.is_optimal());

    res.primal_solution[x1] = std::numeric_limits<double>::quiet_NaN();

    QPVerifier verifier;
    auto report = verifier.verify(model, res.primal_solution, res.dual_solution,
                                  res.objective_value, res.status);
    KAVACH_ASSERT(!report.fully_verified());
    KAVACH_ASSERT(!report.nonfinite_check.passed);
}

// ============================================================================
// Infeasible QP
// ============================================================================
KAVACH_TEST(test_qp_infeasible_problem) {
    auto model = make_simple_model("qp_infeasible");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 10.0);
    model.add_objective_quadratic_term(x1, x1, 1.0);

    SparseLinearExpression c1;
    c1.add_term(x1, 1.0);
    model.add_constraint("c1", std::move(c1), ConstraintSense::LESS_EQUAL, 1.0);

    SparseLinearExpression c2;
    c2.add_term(x1, 1.0);
    model.add_constraint("c2", std::move(c2), ConstraintSense::GREATER_EQUAL, 5.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);

    KAVACH_ASSERT_EQUAL(res.status, core::SolverStatus::INFEASIBLE);
}

// ============================================================================
// Presolve integration with QP
// ============================================================================
KAVACH_TEST(test_qp_presolve_integration) {
    auto model = make_simple_model("qp_presolve");
    // x1 is fixed by bounds [2, 2]
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, 2.0, 2.0);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, 10.0);

    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.add_objective_quadratic_term(x2, x2, 1.0);
    model.set_objective_linear_coefficient(x2, -4.0);

    // Redundant row: x1 + x2 >= 0 (since x1=2, x2>=0)
    SparseLinearExpression c;
    c.add_term(x1, 1.0);
    c.add_term(x2, 1.0);
    model.add_constraint("red", std::move(c), ConstraintSense::GREATER_EQUAL, 0.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    QPSolverOptions opts;
    opts.enable_presolve = true;
    auto res = solver.solve(model, opts);

    KAVACH_ASSERT(res.is_optimal());
    KAVACH_ASSERT(res.verification_report.fully_verified());
    KAVACH_ASSERT_NEAR(res.primal_solution[x1], 2.0, 1e-5);
    KAVACH_ASSERT_NEAR(res.primal_solution[x2], 2.0, 1e-5);
}

// ============================================================================
// Unified verifier interface check
// ============================================================================
KAVACH_TEST(test_qp_unified_verifier) {
    auto model = make_simple_model("qp_unified_ver");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 5.0);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, 5.0);

    model.add_objective_quadratic_term(x1, x1, 1.0);
    model.add_objective_quadratic_term(x2, x2, 1.0);
    model.canonicalize();

    PrimalActiveSetQPSolver solver;
    auto res = solver.solve(model);
    KAVACH_ASSERT(res.is_optimal());

    KavachLPVerifier verifier;
    auto report = verifier.verify_qp(model, res.primal_solution, res.dual_solution,
                                     res.objective_value, core::SolverTolerances{});
    KAVACH_ASSERT(report.passed);
}
