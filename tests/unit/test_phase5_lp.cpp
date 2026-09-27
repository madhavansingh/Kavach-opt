#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../test_framework.hpp"
#include "algorithms/lp/basis.hpp"
#include "algorithms/lp/simplex.hpp"
#include "core/error.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "io/parser.hpp"
#include "model/model.hpp"

using namespace kavach;
using namespace kavach::algorithms::lp;
using namespace kavach::model;

namespace {

[[nodiscard]] std::filesystem::path find_netlib_dir() {
    std::vector<std::filesystem::path> candidates = {"benchmarks/netlib", "../benchmarks/netlib",
                                                     "../../benchmarks/netlib"};
    for (const auto& p : candidates) {
        if (std::filesystem::exists(p) && std::filesystem::exists(p / "afiro.mps")) {
            return p;
        }
    }
    throw std::runtime_error("Could not find benchmarks/netlib directory containing afiro.mps");
}

}  // namespace

// ============================================================================
// A. Basic Feasible LPs
// ============================================================================

KAVACH_TEST(test_lp_single_variable) {
    CanonicalModel model("single_var");
    auto x = model.add_variable("x", VariableType::CONTINUOUS, 5.0, INFINITY_BOUND);
    model.set_objective_sense(ObjectiveSense::MINIMIZE);
    model.add_objective_linear_term(x, 3.0);
    model.set_objective_offset(10.0);

    RevisedSimplex solver;
    auto sol = solver.solve(model);

    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(sol.primal_solution[x], 5.0, 1e-7);
    KAVACH_ASSERT_NEAR(sol.objective_value, 25.0, 1e-7);  // 3 * 5 + 10
}

KAVACH_TEST(test_lp_two_variable_product_mix) {
    // Maximize 3*x1 + 2*x2
    // Subject to:
    // 2*x1 + x2 <= 100
    // x1 + x2 <= 80
    // x1, x2 >= 0
    // Optimal: x1 = 20, x2 = 60, obj = 180
    CanonicalModel model("product_mix");
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);

    model.set_objective_sense(ObjectiveSense::MAXIMIZE);
    model.add_objective_linear_term(x1, 3.0);
    model.add_objective_linear_term(x2, 2.0);

    SparseLinearExpression e1;
    e1.add_term(x1, 2.0);
    e1.add_term(x2, 1.0);
    model.add_constraint("c1", std::move(e1), ConstraintSense::LESS_EQUAL, 100.0);

    SparseLinearExpression e2;
    e2.add_term(x1, 1.0);
    e2.add_term(x2, 1.0);
    model.add_constraint("c2", std::move(e2), ConstraintSense::LESS_EQUAL, 80.0);

    RevisedSimplex solver;
    auto sol = solver.solve(model);

    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(sol.primal_solution[x1], 20.0, 1e-6);
    KAVACH_ASSERT_NEAR(sol.primal_solution[x2], 60.0, 1e-6);
    KAVACH_ASSERT_NEAR(sol.objective_value, 180.0, 1e-6);
}

KAVACH_TEST(test_lp_equality_constraint) {
    // Minimize x1 + 2*x2
    // Subject to:
    // x1 + x2 == 10
    // x1, x2 >= 0
    // Optimal: x1 = 10, x2 = 0, obj = 10
    CanonicalModel model("equality_lp");
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);

    model.set_objective_sense(ObjectiveSense::MINIMIZE);
    model.add_objective_linear_term(x1, 1.0);
    model.add_objective_linear_term(x2, 2.0);

    SparseLinearExpression e;
    e.add_term(x1, 1.0);
    e.add_term(x2, 1.0);
    model.add_constraint("eq", std::move(e), ConstraintSense::EQUAL, 10.0);

    RevisedSimplex solver;
    auto sol = solver.solve(model);

    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(sol.primal_solution[x1], 10.0, 1e-6);
    KAVACH_ASSERT_NEAR(sol.primal_solution[x2], 0.0, 1e-6);
    KAVACH_ASSERT_NEAR(sol.objective_value, 10.0, 1e-6);
}

KAVACH_TEST(test_lp_mixed_senses_and_offset) {
    // Minimize 2*x1 + 3*x2 + 25.0
    // x1 + x2 >= 4
    // 2*x1 - x2 <= 6
    // x1, x2 >= 0
    // Optimal vertex is (10/3, 2/3) with obj = 20/3 + 6/3 + 25 = 26/3 + 25 = 33.666667
    CanonicalModel model("mixed_senses");
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);

    model.set_objective_sense(ObjectiveSense::MINIMIZE);
    model.set_objective_offset(25.0);
    model.add_objective_linear_term(x1, 2.0);
    model.add_objective_linear_term(x2, 3.0);

    SparseLinearExpression e1;
    e1.add_term(x1, 1.0);
    e1.add_term(x2, 1.0);
    model.add_constraint("ge", std::move(e1), ConstraintSense::GREATER_EQUAL, 4.0);

    SparseLinearExpression e2;
    e2.add_term(x1, 2.0);
    e2.add_term(x2, -1.0);
    model.add_constraint("le", std::move(e2), ConstraintSense::LESS_EQUAL, 6.0);

    RevisedSimplex solver;
    auto sol = solver.solve(model);

    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(sol.primal_solution[x1], 10.0 / 3.0, 1e-6);
    KAVACH_ASSERT_NEAR(sol.primal_solution[x2], 2.0 / 3.0, 1e-6);
    KAVACH_ASSERT_NEAR(sol.objective_value, 33.0 + 2.0 / 3.0, 1e-6);
}

// ============================================================================
// B. Free Variables & Negative Bounds
// ============================================================================

KAVACH_TEST(test_lp_free_and_negative_bounds) {
    // Minimize x1 - 2*x2
    // x1 + x2 <= 10
    // x1 >= -5.0
    // -10.0 <= x2 <= 8.0
    // Optimal: x1 = -5.0, x2 = 8.0, obj = -5 - 16 = -21.0
    CanonicalModel model("neg_bounds");
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, -5.0, INFINITY_BOUND);
    auto x2 = model.add_variable("x2", VariableType::CONTINUOUS, -10.0, 8.0);

    model.set_objective_sense(ObjectiveSense::MINIMIZE);
    model.add_objective_linear_term(x1, 1.0);
    model.add_objective_linear_term(x2, -2.0);

    SparseLinearExpression e;
    e.add_term(x1, 1.0);
    e.add_term(x2, 1.0);
    model.add_constraint("c", std::move(e), ConstraintSense::LESS_EQUAL, 10.0);

    RevisedSimplex solver;
    auto sol = solver.solve(model);

    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(sol.primal_solution[x1], -5.0, 1e-6);
    KAVACH_ASSERT_NEAR(sol.primal_solution[x2], 8.0, 1e-6);
    KAVACH_ASSERT_NEAR(sol.objective_value, -21.0, 1e-6);
}

// ============================================================================
// C. Phase I Feasibility & Infeasibility
// ============================================================================

KAVACH_TEST(test_lp_phase1_infeasible_equalities) {
    // x1 + x2 == 5
    // x1 + x2 == 10
    // x1, x2 >= 0
    CanonicalModel model("infeasible_eq");
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);

    SparseLinearExpression e1;
    e1.add_term(x1, 1.0);
    e1.add_term(x2, 1.0);
    model.add_constraint("c1", std::move(e1), ConstraintSense::EQUAL, 5.0);

    SparseLinearExpression e2;
    e2.add_term(x1, 1.0);
    e2.add_term(x2, 1.0);
    model.add_constraint("c2", std::move(e2), ConstraintSense::EQUAL, 10.0);

    RevisedSimplex solver;
    auto sol = solver.solve(model);

    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::INFEASIBLE);
}

KAVACH_TEST(test_lp_phase1_infeasible_contradictory_bounds) {
    // x1 + x2 <= 2
    // x1 >= 2, x2 >= 2
    CanonicalModel model("infeasible_bounds");
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 2.0, INFINITY_BOUND);
    auto x2 = model.add_variable("x2", VariableType::CONTINUOUS, 2.0, INFINITY_BOUND);

    SparseLinearExpression e;
    e.add_term(x1, 1.0);
    e.add_term(x2, 1.0);
    model.add_constraint("c", std::move(e), ConstraintSense::LESS_EQUAL, 2.0);

    RevisedSimplex solver;
    auto sol = solver.solve(model);

    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::INFEASIBLE);
}

// ============================================================================
// D. Unboundedness Detection
// ============================================================================

KAVACH_TEST(test_lp_unbounded_minimization) {
    // Minimize -2*x1 - x2
    // x1 - x2 >= 0
    // x1, x2 >= 0
    CanonicalModel model("unbounded_min");
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);

    model.set_objective_sense(ObjectiveSense::MINIMIZE);
    model.add_objective_linear_term(x1, -2.0);
    model.add_objective_linear_term(x2, -1.0);

    SparseLinearExpression e;
    e.add_term(x1, 1.0);
    e.add_term(x2, -1.0);
    model.add_constraint("c", std::move(e), ConstraintSense::GREATER_EQUAL, 0.0);

    RevisedSimplex solver;
    auto sol = solver.solve(model);

    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::UNBOUNDED);
}

KAVACH_TEST(test_lp_unbounded_maximization) {
    // Maximize x1 + x2
    // -x1 + x2 <= 5
    // x1, x2 >= 0
    CanonicalModel model("unbounded_max");
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);

    model.set_objective_sense(ObjectiveSense::MAXIMIZE);
    model.add_objective_linear_term(x1, 1.0);
    model.add_objective_linear_term(x2, 1.0);

    SparseLinearExpression e;
    e.add_term(x1, -1.0);
    e.add_term(x2, 1.0);
    model.add_constraint("c", std::move(e), ConstraintSense::LESS_EQUAL, 5.0);

    RevisedSimplex solver;
    auto sol = solver.solve(model);

    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::UNBOUNDED);
}

// ============================================================================
// E. Degeneracy
// ============================================================================

KAVACH_TEST(test_lp_degenerate_bfs) {
    // Maximize 2*x1 + 3*x2
    // x1 + x2 <= 4
    // 2*x1 + 2*x2 <= 8 (redundant, forces degeneracy at boundary)
    // x1, x2 >= 0
    CanonicalModel model("degen_lp");
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);

    model.set_objective_sense(ObjectiveSense::MAXIMIZE);
    model.add_objective_linear_term(x1, 2.0);
    model.add_objective_linear_term(x2, 3.0);

    SparseLinearExpression e1;
    e1.add_term(x1, 1.0);
    e1.add_term(x2, 1.0);
    model.add_constraint("c1", std::move(e1), ConstraintSense::LESS_EQUAL, 4.0);

    SparseLinearExpression e2;
    e2.add_term(x1, 2.0);
    e2.add_term(x2, 2.0);
    model.add_constraint("c2", std::move(e2), ConstraintSense::LESS_EQUAL, 8.0);

    RevisedSimplex solver;
    auto sol = solver.solve(model);

    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(sol.objective_value, 12.0, 1e-6);
}

// ============================================================================
// F. Cycling Protection: Beale's Cycling Instance
// ============================================================================

KAVACH_TEST(test_lp_cycling_beale_watchdog_bland) {
    // Beale's classical cycling problem (1955):
    // Minimize -0.75*x1 + 20*x2 - 0.5*x3 + 6*x4
    // Subject to:
    // 0.25*x1 - 8*x2 - x3 + 9*x4 <= 0
    // 0.5*x1 - 12*x2 - 0.5*x3 + 3*x4 <= 0
    // x3 <= 1
    // x1, x2, x3, x4 >= 0
    CanonicalModel model("beale_cycling");
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto x3 = model.add_variable("x3", VariableType::CONTINUOUS, 0.0, 1.0);
    auto x4 = model.add_variable("x4", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);

    model.set_objective_sense(ObjectiveSense::MINIMIZE);
    model.add_objective_linear_term(x1, -0.75);
    model.add_objective_linear_term(x2, 20.0);
    model.add_objective_linear_term(x3, -0.5);
    model.add_objective_linear_term(x4, 6.0);

    SparseLinearExpression e1;
    e1.add_term(x1, 0.25);
    e1.add_term(x2, -8.0);
    e1.add_term(x3, -1.0);
    e1.add_term(x4, 9.0);
    model.add_constraint("c1", std::move(e1), ConstraintSense::LESS_EQUAL, 0.0);

    SparseLinearExpression e2;
    e2.add_term(x1, 0.5);
    e2.add_term(x2, -12.0);
    e2.add_term(x3, -0.5);
    e2.add_term(x4, 3.0);
    model.add_constraint("c2", std::move(e2), ConstraintSense::LESS_EQUAL, 0.0);

    LPSolverOptions opts;
    opts.bland_watchdog_threshold = 6;  // Cycle repeats every 6 iterations under Dantzig
    RevisedSimplex solver(opts);

    auto sol = solver.solve(model);

    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);
    // Bland fallback must have activated to break the 6-iteration cycle
    KAVACH_ASSERT(sol.telemetry.bland_fallback_activated);
    KAVACH_ASSERT(sol.iterations < 50);
}

// ============================================================================
// G. Numerical Robustness & Scaling
// ============================================================================

KAVACH_TEST(test_lp_ill_conditioned_scaling) {
    // Problem with wide coefficient range: 1e5 to 1e-2
    CanonicalModel model("scaled_lp");
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 10.0);
    auto x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, 10.0);

    model.set_objective_sense(ObjectiveSense::MINIMIZE);
    model.add_objective_linear_term(x1, 1e5);
    model.add_objective_linear_term(x2, 2.0);

    SparseLinearExpression e;
    e.add_term(x1, 1e4);
    e.add_term(x2, 1e-2);
    model.add_constraint("c", std::move(e), ConstraintSense::GREATER_EQUAL, 1e4);

    LPSolverOptions opts;
    opts.enable_scaling = true;
    RevisedSimplex solver(opts);

    auto sol = solver.solve(model);

    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(sol.primal_solution[x1] >= 0.0);
}

// ============================================================================
// H. Presolve Integration
// ============================================================================

KAVACH_TEST(test_lp_presolve_integration) {
    // Model with fixed variable, redundant row, and tightened bound
    CanonicalModel model("presolve_lp");
    auto x0 = model.add_variable("x0", VariableType::CONTINUOUS, 5.0, 5.0);  // Fixed
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 10.0);
    auto x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, 10.0);

    model.set_objective_sense(ObjectiveSense::MINIMIZE);
    model.add_objective_linear_term(x0, 2.0);
    model.add_objective_linear_term(x1, 3.0);
    model.add_objective_linear_term(x2, 1.0);

    SparseLinearExpression e1;
    e1.add_term(x0, 1.0);
    e1.add_term(x1, 1.0);
    model.add_constraint("c1", std::move(e1), ConstraintSense::GREATER_EQUAL, 8.0);  // x1 >= 3

    SparseLinearExpression e2;
    e2.add_term(x2, 1.0);
    model.add_constraint("c2", std::move(e2), ConstraintSense::GREATER_EQUAL, 2.0);  // x2 >= 2

    SparseLinearExpression e3;
    e3.add_term(x1, 1.0);
    model.add_constraint("c3", std::move(e3), ConstraintSense::LESS_EQUAL, 20.0);  // redundant

    LPSolverOptions opts;
    opts.enable_presolve = true;
    RevisedSimplex solver(opts);

    auto sol = solver.solve(model);

    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(sol.primal_solution[x0], 5.0, 1e-6);
    KAVACH_ASSERT_NEAR(sol.primal_solution[x1], 3.0, 1e-6);
    KAVACH_ASSERT_NEAR(sol.primal_solution[x2], 2.0, 1e-6);
    // Objective: 2*5 + 3*3 + 1*2 = 10 + 9 + 2 = 21.0
    KAVACH_ASSERT_NEAR(sol.objective_value, 21.0, 1e-6);
}

// ============================================================================
// I. Problem Class Rejection: MILP & QP
// ============================================================================

KAVACH_TEST(test_lp_reject_integer_and_binary) {
    RevisedSimplex solver;

    // Integer variable
    {
        CanonicalModel m1("milp_int");
        m1.add_variable("x", VariableType::INTEGER, 0.0, 10.0);
        bool caught = false;
        try {
            [[maybe_unused]] auto sol1 = solver.solve(m1);
        } catch (const core::InputException&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);
    }

    // Binary variable
    {
        CanonicalModel m2("milp_bin");
        m2.add_variable("y", VariableType::BINARY, 0.0, 1.0);
        bool caught = false;
        try {
            [[maybe_unused]] auto sol2 = solver.solve(m2);
        } catch (const core::InputException&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);
    }

    // Quadratic objective
    {
        CanonicalModel m3("qp_model");
        auto x = m3.add_variable("x", VariableType::CONTINUOUS, 0.0, 10.0);
        m3.add_objective_quadratic_term(x, x, 2.0);
        bool caught = false;
        try {
            [[maybe_unused]] auto sol3 = solver.solve(m3);
        } catch (const core::InputException&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);
    }
}

// ============================================================================
// J. Limit Handling
// ============================================================================

KAVACH_TEST(test_lp_iteration_limit) {
    // Multi-iteration problem
    CanonicalModel model("product_mix");
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);

    model.set_objective_sense(ObjectiveSense::MAXIMIZE);
    model.add_objective_linear_term(x1, 3.0);
    model.add_objective_linear_term(x2, 2.0);

    SparseLinearExpression e1;
    e1.add_term(x1, 2.0);
    e1.add_term(x2, 1.0);
    model.add_constraint("c1", std::move(e1), ConstraintSense::LESS_EQUAL, 100.0);

    SparseLinearExpression e2;
    e2.add_term(x1, 1.0);
    e2.add_term(x2, 1.0);
    model.add_constraint("c2", std::move(e2), ConstraintSense::LESS_EQUAL, 80.0);

    LPSolverOptions opts;
    opts.max_iterations = 1;  // Artificially low limit
    RevisedSimplex solver(opts);

    auto sol = solver.solve(model);

    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::LIMIT_REACHED);
    KAVACH_ASSERT(sol.iterations <= 1);
}

// ============================================================================
// K. Determinism Test
// ============================================================================

KAVACH_TEST(test_lp_determinism) {
    CanonicalModel model("product_mix");
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);

    model.set_objective_sense(ObjectiveSense::MAXIMIZE);
    model.add_objective_linear_term(x1, 3.0);
    model.add_objective_linear_term(x2, 2.0);

    SparseLinearExpression e1;
    e1.add_term(x1, 2.0);
    e1.add_term(x2, 1.0);
    model.add_constraint("c1", std::move(e1), ConstraintSense::LESS_EQUAL, 100.0);

    SparseLinearExpression e2;
    e2.add_term(x1, 1.0);
    e2.add_term(x2, 1.0);
    model.add_constraint("c2", std::move(e2), ConstraintSense::LESS_EQUAL, 80.0);

    RevisedSimplex s1;
    auto sol1 = s1.solve(model);

    RevisedSimplex s2;
    auto sol2 = s2.solve(model);

    KAVACH_ASSERT_EQUAL(sol1.status, sol2.status);
    KAVACH_ASSERT_EQUAL(sol1.iterations, sol2.iterations);
    KAVACH_ASSERT_NEAR(sol1.objective_value, sol2.objective_value, 1e-12);
    KAVACH_ASSERT_NEAR(sol1.primal_solution[x1], sol2.primal_solution[x1], 1e-12);
    KAVACH_ASSERT_NEAR(sol1.primal_solution[x2], sol2.primal_solution[x2], 1e-12);
}

// ============================================================================
// L. Netlib LP Corpus Validation
// ============================================================================

struct NetlibRef {
    std::string file_name;
    double expected_objective;
    double tol;
};

KAVACH_TEST(test_lp_netlib_corpus) {
    const auto netlib_dir = find_netlib_dir();

    std::vector<NetlibRef> instances = {
        {"afiro.mps", -464.75314, 1e-4},   {"sc50b.mps", -70.000000, 1e-4},
        {"sc50a.mps", -64.575077, 1e-4},   {"kb2.mps", -1749.9001, 1e-3},
        {"blend.mps", -30.81215, 1e-4},    {"adlittle.mps", 225494.963, 1.0},
        {"share2b.mps", -415.73224, 1e-3}, {"sc105.mps", -52.20206, 1e-4},
        {"stocfor1.mps", -41131.976, 1.0}, {"share1b.mps", -76589.319, 1.0},
        {"lotfi.mps", -25.264707, 1e-3},   {"beaconfd.mps", 33592.4858, 1.0},
        {"boeing2.mps", -315.01872, 1.0},
    };

    io::MpsParser parser;
    RevisedSimplex solver;

    std::cout << "\n==================== NETLIB LP BENCHMARK RESULTS ====================\n";
    std::cout << std::left << std::setw(14) << "Instance" << std::setw(8) << "Rows" << std::setw(8)
              << "Cols" << std::setw(8) << "NNZ" << std::setw(12) << "Status" << std::setw(18)
              << "Objective" << std::setw(18) << "Reference" << std::setw(8) << "Iters"
              << std::setw(10) << "Time (s)" << "\n";
    std::cout << "--------------------------------------------------------------------------------"
                 "-----------------\n";

    for (const auto& inst : instances) {
        auto model = parser.parse_file(netlib_dir / inst.file_name);
        auto sol = solver.solve(*model);

        std::cout << std::left << std::setw(14) << inst.file_name << std::setw(8)
                  << sol.telemetry.original_rows << std::setw(8) << sol.telemetry.original_cols
                  << std::setw(8) << model->num_nonzeros() << std::setw(12)
                  << core::to_string(sol.status) << std::setw(18) << std::setprecision(8)
                  << sol.objective_value << std::setw(18) << std::setprecision(8)
                  << inst.expected_objective << std::setw(8) << sol.iterations << std::setw(10)
                  << std::setprecision(4) << sol.telemetry.solve_time_seconds << "\n";

        KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);
        KAVACH_ASSERT_NEAR(sol.objective_value, inst.expected_objective, inst.tol);
    }
    std::cout << "================================================================================"
                 "-\n\n";
}

// ============================================================================
// M. Basis-Update Factorization Engine Unit & Adversarial Tests
// ============================================================================

namespace {

numerics::CscMatrix build_well_conditioned_test_csc(std::size_t m, std::size_t n) {
    std::vector<std::size_t> col_ptr(n + 1, 0);
    std::vector<std::size_t> row_idx;
    std::vector<double> values;

    for (std::size_t j = 0; j < n; ++j) {
        col_ptr[j] = row_idx.size();
        if (j < m) {
            // Identity columns for initial basis
            row_idx.push_back(j);
            values.push_back(2.0 + 0.1 * static_cast<double>(j));
            if (j + 1 < m) {
                row_idx.push_back(j + 1);
                values.push_back(-0.25);
            }
        } else {
            // Diagonally dominant Entering columns
            const std::size_t primary_row = (j - m) % m;
            for (std::size_t i = 0; i < m; ++i) {
                if (i == primary_row) {
                    row_idx.push_back(i);
                    values.push_back(3.0 + 0.15 * static_cast<double>(j));
                } else if ((i + j) % 2 == 0) {
                    row_idx.push_back(i);
                    values.push_back(0.2 * ((i < primary_row) ? 1.0 : -1.0));
                }
            }
        }
    }
    col_ptr[n] = row_idx.size();
    return numerics::CscMatrix(m, n, std::move(col_ptr), std::move(row_idx), std::move(values));
}

}  // namespace

KAVACH_TEST(test_basis_update_single_pivot_oracle) {
    const std::size_t m = 4;
    const std::size_t n = 8;
    auto A_csc = build_well_conditioned_test_csc(m, n);

    SimplexBasis basis;
    basis.initialize({0, 1, 2, 3}, n);
    KAVACH_ASSERT(basis.factorize(A_csc, 1e-9, "INITIAL"));
    KAVACH_ASSERT_EQUAL(basis.factorization_generation(), std::size_t(1));
    KAVACH_ASSERT_EQUAL(basis.update_chain_length(), std::size_t(0));

    // Pivot column 4 into row 1
    const std::size_t leaving_row = 1;
    const std::size_t entering_col = 4;
    auto d = basis.solve_direction(A_csc, entering_col);
    KAVACH_ASSERT(basis.apply_eta_update(leaving_row, entering_col, d, 1e-9));
    KAVACH_ASSERT_EQUAL(basis.update_chain_length(), std::size_t(1));
    KAVACH_ASSERT_EQUAL(basis.basis_updates_count(), std::size_t(1));
    KAVACH_ASSERT_EQUAL(basis.full_refactorizations_count(), std::size_t(1));

    // Build fresh LU oracle on the exact same post-pivot basis {0, 4, 2, 3}
    SimplexBasis oracle;
    oracle.initialize(basis.basis_vars(), n);
    KAVACH_ASSERT(oracle.factorize(A_csc, 1e-9, "ORACLE"));

    const std::vector<double> rhs = {1.5, -2.25, 3.75, 0.5};
    const std::vector<double> c_B = {0.8, -1.2, 2.4, -0.6};

    auto x_inc = basis.solve_primal(rhs);
    auto x_orc = oracle.solve_primal(rhs);
    auto y_inc = basis.solve_dual(c_B);
    auto y_orc = oracle.solve_dual(c_B);

    for (std::size_t i = 0; i < m; ++i) {
        KAVACH_ASSERT_NEAR(x_inc[i], x_orc[i], 1e-13);
        KAVACH_ASSERT_NEAR(y_inc[i], y_orc[i], 1e-13);
    }
    KAVACH_ASSERT(basis.compute_primal_residual(A_csc, x_inc, rhs) < 1e-13);
    KAVACH_ASSERT(basis.compute_dual_residual(A_csc, y_inc, c_B) < 1e-13);
}

KAVACH_TEST(test_basis_update_multi_pivot_chain_oracle) {
    const std::size_t m = 6;
    const std::size_t n = 26;
    auto A_csc = build_well_conditioned_test_csc(m, n);

    SimplexBasis basis;
    basis.initialize({0, 1, 2, 3, 4, 5}, n);
    KAVACH_ASSERT(basis.factorize(A_csc, 1e-9, "INITIAL"));

    const std::vector<double> rhs = {2.0, -1.5, 4.2, -3.1, 0.9, 1.7};
    const std::vector<double> c_B = {-1.1, 0.4, 2.3, -0.8, 1.6, -2.5};

    // Apply 20 sequential updates without refactorization and verify at steps 2, 5, 10, 20
    for (std::size_t step = 1; step <= 20; ++step) {
        const std::size_t leaving_row = (step - 1) % m;
        const std::size_t entering_col = m + (step - 1);
        auto d = basis.solve_direction(A_csc, entering_col);
        KAVACH_ASSERT(basis.apply_eta_update(leaving_row, entering_col, d, 1e-9));
        KAVACH_ASSERT_EQUAL(basis.update_chain_length(), step);

        if (step == 2 || step == 5 || step == 10 || step == 20) {
            SimplexBasis oracle;
            oracle.initialize(basis.basis_vars(), n);
            KAVACH_ASSERT(oracle.factorize(A_csc, 1e-9, "ORACLE"));

            auto x_inc = basis.solve_primal(rhs);
            auto x_orc = oracle.solve_primal(rhs);
            auto y_inc = basis.solve_dual(c_B);
            auto y_orc = oracle.solve_dual(c_B);

            for (std::size_t i = 0; i < m; ++i) {
                KAVACH_ASSERT_NEAR(x_inc[i], x_orc[i], 1e-11);
                KAVACH_ASSERT_NEAR(y_inc[i], y_orc[i], 1e-11);
            }
            KAVACH_ASSERT(basis.compute_primal_residual(A_csc, x_inc, rhs) < 1e-11);
            KAVACH_ASSERT(basis.compute_dual_residual(A_csc, y_inc, c_B) < 1e-11);
        }
    }
    KAVACH_ASSERT_EQUAL(basis.full_refactorizations_count(), std::size_t(1));
    KAVACH_ASSERT_EQUAL(basis.basis_updates_count(), std::size_t(20));
}

KAVACH_TEST(test_basis_update_refactorization_reset_and_generation) {
    const std::size_t m = 5;
    const std::size_t n = 15;
    auto A_csc = build_well_conditioned_test_csc(m, n);

    SimplexBasis basis;
    basis.initialize({0, 1, 2, 3, 4}, n);
    KAVACH_ASSERT(basis.factorize(A_csc, 1e-9, "INITIAL"));
    KAVACH_ASSERT_EQUAL(basis.factorization_generation(), std::size_t(1));

    for (std::size_t k = 0; k < 8; ++k) {
        auto d = basis.solve_direction(A_csc, m + k);
        KAVACH_ASSERT(basis.apply_eta_update(k % m, m + k, d, 1e-9));
    }
    KAVACH_ASSERT_EQUAL(basis.update_chain_length(), std::size_t(8));
    KAVACH_ASSERT(basis.update_chain_bytes() > 0);
    KAVACH_ASSERT(basis.needs_refactorization(8));

    const std::vector<double> rhs = {1.0, 2.0, -1.0, 3.0, 0.5};
    auto x_before_ref = basis.solve_primal(rhs);

    // Full refactorization must clear chain, advance generation, and reset chain bytes
    KAVACH_ASSERT(basis.factorize(A_csc, 1e-9, "PERIODIC_CHAIN_INTERVAL"));
    KAVACH_ASSERT_EQUAL(basis.factorization_generation(), std::size_t(2));
    KAVACH_ASSERT_EQUAL(basis.update_chain_length(), std::size_t(0));
    KAVACH_ASSERT_EQUAL(basis.update_chain_bytes(), std::size_t(0));
    KAVACH_ASSERT_EQUAL(basis.updates_since_refactor(), std::size_t(0));

    auto x_after_ref = basis.solve_primal(rhs);
    for (std::size_t i = 0; i < m; ++i) {
        KAVACH_ASSERT_NEAR(x_before_ref[i], x_after_ref[i], 1e-12);
    }

    // Untracked combinatorial pivot must invalidate basis until factorize() is called
    basis.update_pivot(0, 13);
    KAVACH_ASSERT(!basis.is_factorization_valid());
    bool threw_on_stale = false;
    try {
        (void)basis.solve_primal(rhs);
    } catch (const std::logic_error&) {
        threw_on_stale = true;
    }
    KAVACH_ASSERT(threw_on_stale);
}

KAVACH_TEST(test_basis_update_numerical_drift_trigger) {
    const auto netlib_dir = find_netlib_dir();
    io::MpsParser parser;
    auto model = parser.parse_file(netlib_dir / "adlittle.mps");

    LPSolverOptions opts_tight;
    opts_tight.refactorization_interval = 100;
    opts_tight.basis_drift_tol = 1e-15;  // Force ultra-tight drift refactorization trigger
    RevisedSimplex solver_tight(opts_tight);
    auto sol_tight = solver_tight.solve(*model, opts_tight);

    LPSolverOptions opts_std;
    opts_std.refactorization_interval = 50;
    opts_std.basis_drift_tol = 1e-9;
    RevisedSimplex solver_std(opts_std);
    auto sol_std = solver_std.solve(*model, opts_std);

    KAVACH_ASSERT_EQUAL(sol_tight.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_EQUAL(sol_std.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(sol_tight.objective_value, sol_std.objective_value, 1e-6);
    KAVACH_ASSERT(sol_tight.telemetry.full_refactorizations >
                  sol_std.telemetry.full_refactorizations);
}

KAVACH_TEST(test_basis_update_small_pivot_fallback) {
    const std::size_t m = 4;
    const std::size_t n = 8;
    auto A_csc = build_well_conditioned_test_csc(m, n);

    SimplexBasis basis;
    basis.initialize({0, 1, 2, 3}, n);
    KAVACH_ASSERT(basis.factorize(A_csc, 1e-9, "INITIAL"));

    // Construct an artificially tiny pivot direction d[1] = 1e-13 while ||d||_inf = 5.0
    std::vector<double> bad_d = {5.0, 1e-13, -2.0, 1.0};
    KAVACH_ASSERT(!basis.apply_eta_update(1, 4, bad_d, 1e-9));
    KAVACH_ASSERT_EQUAL(basis.update_failures_count(), std::size_t(1));
    KAVACH_ASSERT(!basis.is_factorization_valid());
    KAVACH_ASSERT_EQUAL(basis.last_refactor_reason(), std::string("UNSTABLE_ETA_PIVOT"));

    // Recovery via full refactorization succeeds cleanly
    KAVACH_ASSERT(basis.factorize(A_csc, 1e-9, basis.last_refactor_reason()));
    KAVACH_ASSERT(basis.is_factorization_valid());
}

KAVACH_TEST(test_basis_update_degeneracy_and_dual_simplex) {
    const auto netlib_dir = find_netlib_dir();
    io::MpsParser parser;
    auto model = parser.parse_file(netlib_dir / "blend.mps");

    LPSolverOptions opts;
    opts.preserve_bounds_structure = true;
    opts.enable_presolve = false;
    RevisedSimplex solver(opts);
    auto sol = solver.solve(*model, opts);

    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(sol.telemetry.basis_updates > 0);
    // Full refactorizations must be dramatically smaller than total iterations
    KAVACH_ASSERT(sol.telemetry.full_refactorizations * 5 < sol.iterations);
    KAVACH_ASSERT(sol.telemetry.basis_residual_max < 1e-9);
    KAVACH_ASSERT(sol.basis.has_value());

    // Perturb variable bound slightly and warm-start with Dual Simplex
    auto modified_model = *model;
    modified_model.variable(0).lower_bound = 0.05;
    auto ws_sol = solver.solve_warm_start(modified_model, *sol.basis, opts);
    KAVACH_ASSERT_EQUAL(ws_sol.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(ws_sol.telemetry.full_refactorizations <=
                  1 + ws_sol.telemetry.dual_simplex_iterations);
    KAVACH_ASSERT(ws_sol.telemetry.basis_residual_max < 1e-9);
}
