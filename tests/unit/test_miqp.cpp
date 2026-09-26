/**
 * @file test_miqp.cpp
 * @brief Comprehensive test suite for Convex Mixed-Integer Quadratic Programming (MIQP):
 *        Model classification, PSD/convexity validation, small exhaustive oracle verification,
 *        binary portfolio selection, mixed-integer continuous QP, infeasibility,
 *        zero-Q MILP equivalence, pure QP equivalence, maximization, advanced branching,
 *        determinism, telemetry, and independent zero-trust verification.
 */

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "../test_framework.hpp"
#include "algorithms/milp/branch_and_bound.hpp"
#include "algorithms/miqp/miqp_solver.hpp"
#include "algorithms/miqp/miqp_types.hpp"
#include "algorithms/qp/active_set.hpp"
#include "api/solver.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"
#include "strategy/strategy.hpp"
#include "verification/miqp_verifier.hpp"

using namespace kavach;
using namespace kavach::model;
using namespace kavach::algorithms::miqp;
using namespace kavach::algorithms::qp;
using namespace kavach::algorithms::milp;
using namespace kavach::verification;

namespace {

/** Helper: build a small 3-variable mixed-integer quadratic model */
CanonicalModel make_small_miqp_model() {
    CanonicalModel m("small_miqp");
    m.set_objective_sense(ObjectiveSense::MINIMIZE);

    // x0: continuous [0, 5]
    // x1: integer [0, 3]
    // x2: binary [0, 1]
    auto x0 = m.add_variable("x0", VariableType::CONTINUOUS, 0.0, 5.0);
    auto x1 = m.add_variable("x1", VariableType::INTEGER, 0.0, 3.0);
    auto x2 = m.add_variable("x2", VariableType::BINARY, 0.0, 1.0);

    // Objective: (x0 - 2)^2 + 2*(x1 - 1.5)^2 + 3*(x2 - 1)^2
    // = x0^2 - 4*x0 + 2*x1^2 - 6*x1 + 3*x2^2 - 6*x2 + (4 + 4.5 + 3 = 11.5)
    // In canonical form:
    // 1/2 * (2*x0^2 + 4*x1^2 + 6*x2^2) - 4*x0 - 6*x1 - 6*x2 + 11.5
    // Quadratic terms in CanonicalModel:
    // sum_{i,j} q_ij * x_i * x_j
    // For x0^2: coeff = 1.0
    // For x1^2: coeff = 2.0
    // For x2^2: coeff = 3.0
    m.set_objective_offset(11.5);
    m.add_objective_linear_term(x0, -4.0);
    m.add_objective_linear_term(x1, -6.0);
    m.add_objective_linear_term(x2, -6.0);

    m.add_objective_quadratic_term(x0, x0, 1.0);
    m.add_objective_quadratic_term(x1, x1, 2.0);
    m.add_objective_quadratic_term(x2, x2, 3.0);

    // Constraint: x0 + x1 + 2*x2 <= 4.2
    SparseLinearExpression expr1;
    expr1.add_term(x0, 1.0);
    expr1.add_term(x1, 1.0);
    expr1.add_term(x2, 2.0);
    m.add_constraint("c1", expr1, ConstraintSense::LESS_EQUAL, 4.2);

    return m;
}

}  // namespace

// ============================================================================
// 1. Model Classification Tests
// ============================================================================

KAVACH_TEST(miqp_model_classification) {
    // LP model
    CanonicalModel lp_model("lp");
    auto lp_x = lp_model.add_variable("x", VariableType::CONTINUOUS, 0.0, 10.0);
    lp_model.add_objective_linear_term(lp_x, 1.0);
    KAVACH_ASSERT(lp_model.is_lp());
    KAVACH_ASSERT(!lp_model.is_milp());
    KAVACH_ASSERT(!lp_model.is_qp());
    KAVACH_ASSERT(!lp_model.is_miqp());

    // MILP model
    CanonicalModel milp_model("milp");
    auto milp_x = milp_model.add_variable("x", VariableType::INTEGER, 0.0, 10.0);
    milp_model.add_objective_linear_term(milp_x, 1.0);
    KAVACH_ASSERT(!milp_model.is_lp());
    KAVACH_ASSERT(milp_model.is_milp());
    KAVACH_ASSERT(!milp_model.is_qp());
    KAVACH_ASSERT(!milp_model.is_miqp());

    // QP model
    CanonicalModel qp_model("qp");
    auto qp_x = qp_model.add_variable("x", VariableType::CONTINUOUS, 0.0, 10.0);
    qp_model.add_objective_quadratic_term(qp_x, qp_x, 1.0);
    KAVACH_ASSERT(!qp_model.is_lp());
    KAVACH_ASSERT(!qp_model.is_milp());
    KAVACH_ASSERT(qp_model.is_qp());
    KAVACH_ASSERT(!qp_model.is_miqp());

    // MIQP model
    CanonicalModel miqp_model("miqp");
    auto miqp_x = miqp_model.add_variable("x", VariableType::INTEGER, 0.0, 10.0);
    miqp_model.add_objective_quadratic_term(miqp_x, miqp_x, 1.0);
    KAVACH_ASSERT(!miqp_model.is_lp());
    KAVACH_ASSERT(!miqp_model.is_milp());
    KAVACH_ASSERT(!miqp_model.is_qp());
    KAVACH_ASSERT(miqp_model.is_miqp());

    // Strategy routing check
    strategy::StrategyEngine engine;
    auto chars = engine.characterize(miqp_model);
    KAVACH_ASSERT_EQUAL(static_cast<int>(chars.problem_class),
                        static_cast<int>(strategy::ProblemClass::CONVEX_MIQP));
    auto dec = engine.decide(chars);
    KAVACH_ASSERT_EQUAL(static_cast<int>(dec.problem_class),
                        static_cast<int>(strategy::ProblemClass::CONVEX_MIQP));
}

// ============================================================================
// 2. Convexity / PSD Validation and Non-Convex Rejection
// ============================================================================

KAVACH_TEST(miqp_psd_validation_and_nonconvex_rejection) {
    // Case A: Valid PSD Q
    auto valid_model = make_small_miqp_model();
    ConvexMIQPSolver solver;
    auto sol_valid = solver.solve(valid_model);
    KAVACH_ASSERT_EQUAL(static_cast<int>(sol_valid.status),
                        static_cast<int>(core::SolverStatus::OPTIMAL));

    // Case B: Non-convex with negative diagonal term (q_00 = -1.0)
    CanonicalModel neg_diag("nonconvex_neg_diag");
    auto v0 = neg_diag.add_variable("y0", VariableType::INTEGER, 0.0, 5.0);
    neg_diag.add_objective_quadratic_term(v0, v0, -2.0);  // negative diagonal
    auto sol_neg = solver.solve(neg_diag);
    KAVACH_ASSERT_EQUAL(static_cast<int>(sol_neg.status),
                        static_cast<int>(core::SolverStatus::NUMERICAL_FAILURE));
    KAVACH_ASSERT(sol_neg.message.find("Non-convex MIQP") != std::string::npos);

    // Case C: Non-convex with positive diagonals but indefinite cross-term (saddle)
    // Q = [1  3; 3  1] -> det = 1 - 9 = -8 < 0
    CanonicalModel saddle("nonconvex_saddle");
    auto s0 = saddle.add_variable("y0", VariableType::INTEGER, 0.0, 5.0);
    auto s1 = saddle.add_variable("y1", VariableType::INTEGER, 0.0, 5.0);
    saddle.add_objective_quadratic_term(s0, s0, 0.5);  // Q_00 = 1.0
    saddle.add_objective_quadratic_term(s1, s1, 0.5);  // Q_11 = 1.0
    saddle.add_objective_quadratic_term(s0, s1, 3.0);  // Q_01 = 3.0
    auto sol_saddle = solver.solve(saddle);
    KAVACH_ASSERT_EQUAL(static_cast<int>(sol_saddle.status),
                        static_cast<int>(core::SolverStatus::NUMERICAL_FAILURE));
    KAVACH_ASSERT(sol_saddle.message.find("Non-convex MIQP") != std::string::npos);
}

// ============================================================================
// 3. Small Exhaustive Verification Oracle
// ============================================================================

KAVACH_TEST(miqp_small_exhaustive_oracle_verification) {
    // Model:
    // min f(x0, y0, y1) = (x0 - 1.2)^2 + (y0 - 1.7)^2 + (y1 - 0.8)^2
    //     = x0^2 - 2.4*x0 + y0^2 - 3.4*y0 + y1^2 - 1.6*y1 + (1.44 + 2.89 + 0.64 = 4.97)
    // s.t. x0 + y0 + y1 <= 3.5
    //      0 <= x0 <= 5.0 (continuous)
    //      y0 in {0, 1, 2} (integer)
    //      y1 in {0, 1, 2, 3} (integer)
    CanonicalModel model("oracle_test");
    model.set_objective_sense(ObjectiveSense::MINIMIZE);

    auto x0 = model.add_variable("x0", VariableType::CONTINUOUS, 0.0, 5.0);
    auto y0 = model.add_variable("y0", VariableType::INTEGER, 0.0, 2.0);
    auto y1 = model.add_variable("y1", VariableType::INTEGER, 0.0, 3.0);

    model.set_objective_offset(4.97);
    model.add_objective_linear_term(x0, -2.4);
    model.add_objective_linear_term(y0, -3.4);
    model.add_objective_linear_term(y1, -1.6);

    model.add_objective_quadratic_term(x0, x0, 1.0);
    model.add_objective_quadratic_term(y0, y0, 1.0);
    model.add_objective_quadratic_term(y1, y1, 1.0);

    SparseLinearExpression expr;
    expr.add_term(x0, 1.0);
    expr.add_term(y0, 1.0);
    expr.add_term(y1, 1.0);
    model.add_constraint("sum_budget", expr, ConstraintSense::LESS_EQUAL, 3.5);

    // --- Exhaustive Integer Enumeration Oracle ---
    double oracle_best_obj = std::numeric_limits<double>::infinity();
    std::vector<double> oracle_best_sol(3, 0.0);

    PrimalActiveSetQPSolver qp_engine;
    core::SolverTolerances tols;

    for (int int_y0 = 0; int_y0 <= 2; ++int_y0) {
        for (int int_y1 = 0; int_y1 <= 3; ++int_y1) {
            double dy0 = static_cast<double>(int_y0);
            double dy1 = static_cast<double>(int_y1);

            // Sub-QP on x0:
            // min (x0 - 1.2)^2 + (dy0 - 1.7)^2 + (dy1 - 0.8)^2
            // s.t. x0 <= 3.5 - dy0 - dy1, 0 <= x0 <= 5.0
            double max_x0 = 3.5 - dy0 - dy1;
            if (max_x0 < -tols.primal_feasibility_tol) {
                continue;  // Infeasible
            }

            CanonicalModel sub_qp("sub_qp");
            auto sub_x = sub_qp.add_variable("x0", VariableType::CONTINUOUS, 0.0,
                                             std::min(5.0, std::max(0.0, max_x0)));
            double const_term = 4.97 + (dy0 * dy0 - 3.4 * dy0) + (dy1 * dy1 - 1.6 * dy1);
            sub_qp.set_objective_offset(const_term);
            sub_qp.add_objective_linear_term(sub_x, -2.4);
            sub_qp.add_objective_quadratic_term(sub_x, sub_x, 1.0);

            QPSolverOptions qp_opts;
            qp_opts.tolerances = tols;
            auto sub_sol = qp_engine.solve(sub_qp, qp_opts);
            if (sub_sol.is_optimal()) {
                if (sub_sol.objective_value < oracle_best_obj) {
                    oracle_best_obj = sub_sol.objective_value;
                    oracle_best_sol[0] = sub_sol.primal_solution[0];
                    oracle_best_sol[1] = dy0;
                    oracle_best_sol[2] = dy1;
                }
            }
        }
    }

    // --- Solve using ConvexMIQPSolver ---
    ConvexMIQPSolver solver;
    auto sol = solver.solve(model);

    KAVACH_ASSERT_EQUAL(static_cast<int>(sol.status),
                        static_cast<int>(core::SolverStatus::OPTIMAL));
    KAVACH_ASSERT_NEAR(sol.best_objective, oracle_best_obj, 1e-5);
    KAVACH_ASSERT_NEAR(sol.incumbent_solution[0], oracle_best_sol[0], 1e-4);
    KAVACH_ASSERT_NEAR(sol.incumbent_solution[1], oracle_best_sol[1], 1e-4);
    KAVACH_ASSERT_NEAR(sol.incumbent_solution[2], oracle_best_sol[2], 1e-4);

    // Independent verification
    MIQPVerifier verifier;
    auto vrpt = verifier.verify(model, sol.incumbent_solution, sol.best_bound, sol.best_objective,
                                core::SolverStatus::OPTIMAL, tols, sol.optimality_gap);
    KAVACH_ASSERT(vrpt.fully_verified());
}

// ============================================================================
// 4. Binary Portfolio Optimization MIQP
// ============================================================================

KAVACH_TEST(miqp_binary_portfolio_selection) {
    // 4 assets, select at most 2:
    // x_i in {0, 1}
    // min 1/2 x^T Q x - r^T x
    // where r = [0.15, 0.12, 0.10, 0.08]
    // Q = diag([0.10, 0.08, 0.05, 0.03]) + 0.01 * 1 1^T
    CanonicalModel model("portfolio_selection");
    model.set_objective_sense(ObjectiveSense::MINIMIZE);

    const double returns[4] = {0.15, 0.12, 0.10, 0.08};
    const double sigma_diag[4] = {0.10, 0.08, 0.05, 0.03};

    std::vector<VarId> x_vars;
    SparseLinearExpression card_expr;

    for (std::size_t i = 0; i < 4; ++i) {
        auto v = model.add_variable("asset_" + std::to_string(i), VariableType::BINARY, 0.0, 1.0);
        x_vars.push_back(v);
        model.add_objective_linear_term(v, -returns[i]);
        card_expr.add_term(v, 1.0);

        for (std::size_t j = 0; j < 4; ++j) {
            double cov = (i == j ? sigma_diag[i] : 0.0) + 0.01;
            // 1/2 x^T Q x: quadratic terms in model are q_ij * x_i * x_j
            // For i == j: q_ii = cov / 2
            // For i < j: q_ij = cov
            if (i == j) {
                model.add_objective_quadratic_term(v, v, cov * 0.5);
            } else if (i < j) {
                // will be handled when j reached or add half
            }
        }
    }

    // Add off-diagonals
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = i + 1; j < 4; ++j) {
            model.add_objective_quadratic_term(x_vars[i], x_vars[j], 0.01);
        }
    }

    // Constraint: select at most 2 assets: sum x_i <= 2
    model.add_constraint("cardinality", card_expr, ConstraintSense::LESS_EQUAL, 2.0);

    ConvexMIQPSolver solver;
    auto sol = solver.solve(model);

    KAVACH_ASSERT_EQUAL(static_cast<int>(sol.status),
                        static_cast<int>(core::SolverStatus::OPTIMAL));
    KAVACH_ASSERT(sol.has_incumbent());

    // Check cardinality: sum x_i <= 2
    double sum_selected = 0.0;
    for (double val : sol.incumbent_solution) {
        KAVACH_ASSERT(std::abs(val - std::round(val)) < 1e-5);
        sum_selected += val;
    }
    KAVACH_ASSERT(sum_selected <= 2.0 + 1e-6);

    MIQPVerifier verifier;
    auto vrpt = verifier.verify(model, sol.incumbent_solution, sol.best_bound, sol.best_objective,
                                core::SolverStatus::OPTIMAL, {}, sol.optimality_gap);
    KAVACH_ASSERT(vrpt.fully_verified());
}

// ============================================================================
// 5. Infeasibility Tests
// ============================================================================

KAVACH_TEST(miqp_infeasible_root) {
    CanonicalModel model("infeasible_root");
    model.set_objective_sense(ObjectiveSense::MINIMIZE);

    auto x0 = model.add_variable("x0", VariableType::CONTINUOUS, 0.0, 5.0);
    auto x1 = model.add_variable("x1", VariableType::INTEGER, 0.0, 5.0);

    model.add_objective_quadratic_term(x0, x0, 1.0);
    model.add_objective_quadratic_term(x1, x1, 1.0);

    // Contradictory constraints: x0 + x1 <= 1 AND x0 + x1 >= 3
    SparseLinearExpression c1;
    c1.add_term(x0, 1.0);
    c1.add_term(x1, 1.0);
    model.add_constraint("c1", c1, ConstraintSense::LESS_EQUAL, 1.0);

    SparseLinearExpression c2;
    c2.add_term(x0, 1.0);
    c2.add_term(x1, 1.0);
    model.add_constraint("c2", c2, ConstraintSense::GREATER_EQUAL, 3.0);

    ConvexMIQPSolver solver;
    auto sol = solver.solve(model);

    KAVACH_ASSERT_EQUAL(static_cast<int>(sol.status),
                        static_cast<int>(core::SolverStatus::INFEASIBLE));
}

KAVACH_TEST(miqp_infeasible_integer_domain) {
    CanonicalModel model("infeasible_integer");
    model.set_objective_sense(ObjectiveSense::MINIMIZE);

    // y in [0.2, 0.8] but y is INTEGER -> empty feasible domain!
    auto y = model.add_variable("y", VariableType::INTEGER, 0.2, 0.8);
    model.add_objective_quadratic_term(y, y, 1.0);

    ConvexMIQPSolver solver;
    auto sol = solver.solve(model);

    KAVACH_ASSERT_EQUAL(static_cast<int>(sol.status),
                        static_cast<int>(core::SolverStatus::INFEASIBLE));
}

// ============================================================================
// 6. Regression & Equivalence Tests
// ============================================================================

KAVACH_TEST(miqp_zero_q_milp_equivalence) {
    // Knapsack problem with Q = 0
    CanonicalModel knapsack("knapsack_linear");
    knapsack.set_objective_sense(ObjectiveSense::MAXIMIZE);

    auto x0 = knapsack.add_variable("x0", VariableType::BINARY, 0.0, 1.0);
    auto x1 = knapsack.add_variable("x1", VariableType::BINARY, 0.0, 1.0);
    auto x2 = knapsack.add_variable("x2", VariableType::BINARY, 0.0, 1.0);

    knapsack.add_objective_linear_term(x0, 10.0);
    knapsack.add_objective_linear_term(x1, 15.0);
    knapsack.add_objective_linear_term(x2, 20.0);

    SparseLinearExpression w_expr;
    w_expr.add_term(x0, 2.0);
    w_expr.add_term(x1, 3.0);
    w_expr.add_term(x2, 4.0);
    knapsack.add_constraint("weight", w_expr, ConstraintSense::LESS_EQUAL, 5.0);

    // Solve via MILP solver directly
    BranchAndBound milp_solver;
    auto milp_sol = milp_solver.solve(knapsack);

    // Solve via MIQP solver (should detect Q=0 and yield identical solution)
    ConvexMIQPSolver miqp_solver;
    auto miqp_sol = miqp_solver.solve(knapsack);

    KAVACH_ASSERT_EQUAL(static_cast<int>(miqp_sol.status), static_cast<int>(milp_sol.status));
    KAVACH_ASSERT_NEAR(miqp_sol.best_objective, milp_sol.best_objective, 1e-9);
    for (std::size_t j = 0; j < milp_sol.incumbent_solution.size(); ++j) {
        KAVACH_ASSERT_NEAR(miqp_sol.incumbent_solution[j], milp_sol.incumbent_solution[j], 1e-9);
    }
}

KAVACH_TEST(miqp_pure_continuous_qp_equivalence) {
    // Model with continuous variables only
    CanonicalModel qp_model("pure_qp");
    qp_model.set_objective_sense(ObjectiveSense::MINIMIZE);

    auto x0 = qp_model.add_variable("x0", VariableType::CONTINUOUS, 0.0, 5.0);
    auto x1 = qp_model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 5.0);

    // min (x0 - 1)^2 + (x1 - 2)^2 = x0^2 - 2*x0 + x1^2 - 4*x1 + 5
    qp_model.set_objective_offset(5.0);
    qp_model.add_objective_linear_term(x0, -2.0);
    qp_model.add_objective_linear_term(x1, -4.0);
    qp_model.add_objective_quadratic_term(x0, x0, 1.0);
    qp_model.add_objective_quadratic_term(x1, x1, 1.0);

    SparseLinearExpression c;
    c.add_term(x0, 1.0);
    c.add_term(x1, 1.0);
    qp_model.add_constraint("c", c, ConstraintSense::LESS_EQUAL, 2.0);

    PrimalActiveSetQPSolver qp_solver;
    auto qp_sol = qp_solver.solve(qp_model);

    ConvexMIQPSolver miqp_solver;
    auto miqp_sol = miqp_solver.solve(qp_model);

    KAVACH_ASSERT_EQUAL(static_cast<int>(miqp_sol.status), static_cast<int>(qp_sol.status));
    KAVACH_ASSERT_NEAR(miqp_sol.best_objective, qp_sol.objective_value, 1e-6);
    KAVACH_ASSERT_NEAR(miqp_sol.incumbent_solution[0], qp_sol.primal_solution[0], 1e-6);
    KAVACH_ASSERT_NEAR(miqp_sol.incumbent_solution[1], qp_sol.primal_solution[1], 1e-6);
}

// ============================================================================
// 7. Maximization MIQP
// ============================================================================

KAVACH_TEST(miqp_maximization) {
    // max -x0^2 - x1^2 + 6*x0 + 4*x1
    // subject to x0, x1 in {0, 1, 2, 3, 4} (integer)
    // f(x0, x1) = -(x0 - 3)^2 - (x1 - 2)^2 + 13
    // Unconstrained optimum: x0 = 3, x1 = 2 -> f = 13.0
    CanonicalModel max_model("max_miqp");
    max_model.set_objective_sense(ObjectiveSense::MAXIMIZE);

    auto x0 = max_model.add_variable("x0", VariableType::INTEGER, 0.0, 4.0);
    auto x1 = max_model.add_variable("x1", VariableType::INTEGER, 0.0, 4.0);

    max_model.add_objective_linear_term(x0, 6.0);
    max_model.add_objective_linear_term(x1, 4.0);
    // In canonical form, max f(x) has -x0^2, -x1^2
    max_model.add_objective_quadratic_term(x0, x0, -1.0);
    max_model.add_objective_quadratic_term(x1, x1, -1.0);

    ConvexMIQPSolver solver;
    auto sol = solver.solve(max_model);

    KAVACH_ASSERT_EQUAL(static_cast<int>(sol.status),
                        static_cast<int>(core::SolverStatus::OPTIMAL));
    KAVACH_ASSERT_NEAR(sol.best_objective, 13.0, 1e-4);
    KAVACH_ASSERT_NEAR(sol.incumbent_solution[0], 3.0, 1e-4);
    KAVACH_ASSERT_NEAR(sol.incumbent_solution[1], 2.0, 1e-4);
}

// ============================================================================
// 8. Advanced Branching Strategies Comparison
// ============================================================================

KAVACH_TEST(miqp_branching_strategies) {
    auto model = make_small_miqp_model();

    BranchingStrategy strategies[] = {
        BranchingStrategy::MOST_FRACTIONAL,
        BranchingStrategy::PSEUDOCOST,
        BranchingStrategy::STRONG,
        BranchingStrategy::RELIABILITY,
    };

    double baseline_obj = 0.0;
    bool baseline_set = false;

    for (auto strat : strategies) {
        MIQPSolverOptions opts;
        opts.branching_strategy = strat;
        ConvexMIQPSolver solver(opts);
        auto sol = solver.solve(model);

        KAVACH_ASSERT_EQUAL(static_cast<int>(sol.status),
                            static_cast<int>(core::SolverStatus::OPTIMAL));
        if (!baseline_set) {
            baseline_obj = sol.best_objective;
            baseline_set = true;
        } else {
            KAVACH_ASSERT_NEAR(sol.best_objective, baseline_obj, 1e-4);
        }
    }
}

// ============================================================================
// 9. Determinism Test
// ============================================================================

KAVACH_TEST(miqp_search_determinism) {
    auto model = make_small_miqp_model();
    MIQPSolverOptions opts;
    opts.branching_strategy = BranchingStrategy::RELIABILITY;

    ConvexMIQPSolver solver(opts);
    auto baseline = solver.solve(model);

    for (int run = 0; run < 5; ++run) {
        auto sol = solver.solve(model);
        KAVACH_ASSERT_EQUAL(static_cast<int>(sol.status), static_cast<int>(baseline.status));
        KAVACH_ASSERT_NEAR(sol.best_objective, baseline.best_objective, 1e-12);
        KAVACH_ASSERT_EQUAL(sol.nodes, baseline.nodes);
        KAVACH_ASSERT_EQUAL(sol.iterations, baseline.iterations);
        for (std::size_t j = 0; j < sol.incumbent_solution.size(); ++j) {
            KAVACH_ASSERT_NEAR(sol.incumbent_solution[j], baseline.incumbent_solution[j], 1e-12);
        }
    }
}

// ============================================================================
// 10. Telemetry & Verification Fault Injection
// ============================================================================

KAVACH_TEST(miqp_telemetry_and_verifier_fault_injection) {
    auto model = make_small_miqp_model();
    ConvexMIQPSolver solver;
    auto sol = solver.solve(model);

    // Verify telemetry
    KAVACH_ASSERT(sol.telemetry.qp_relaxations_solved > 0);
    KAVACH_ASSERT(sol.telemetry.qp_relaxation_iterations > 0);
    KAVACH_ASSERT(sol.telemetry.nodes_explored > 0);

    MIQPVerifier verifier;

    // 1. Valid solution passes
    auto v_good = verifier.verify(model, sol.incumbent_solution, sol.best_bound, sol.best_objective,
                                  sol.status);
    KAVACH_ASSERT(v_good.fully_verified());

    // 2. Bound violation fault injection
    auto bad_bound = sol.incumbent_solution;
    bad_bound[0] = 100.0;  // upper bound was 5.0
    auto v_bound =
        verifier.verify(model, bad_bound, sol.best_bound, sol.best_objective, sol.status);
    KAVACH_ASSERT(!v_bound.bound_feasibility.passed);
    KAVACH_ASSERT(!v_bound.fully_verified());

    // 3. Integrality fault injection
    auto bad_int = sol.incumbent_solution;
    bad_int[1] = 1.45;  // variable x1 is integer!
    auto v_int = verifier.verify(model, bad_int, sol.best_bound, sol.best_objective, sol.status);
    KAVACH_ASSERT(!v_int.integrality.passed);
    KAVACH_ASSERT(!v_int.fully_verified());

    // 4. Altered objective fault injection
    auto v_obj = verifier.verify(model, sol.incumbent_solution, sol.best_bound,
                                 sol.best_objective + 10.0, sol.status);
    KAVACH_ASSERT(!v_obj.objective_consistency.passed);
    KAVACH_ASSERT(!v_obj.fully_verified());
}

// ============================================================================
// 11. Programmatic API Roundtrip
// ============================================================================

KAVACH_TEST(miqp_programmatic_api_roundtrip) {
    auto model = make_small_miqp_model();

    api::KavachSolver solver;
    api::SolverOptions opts;
    opts.problem_type = api::ProblemType::AUTO;

    auto res = solver.solve(model, opts);
    KAVACH_ASSERT_EQUAL(static_cast<int>(res.status),
                        static_cast<int>(core::SolverStatus::OPTIMAL));
    KAVACH_ASSERT(res.verification_report.passed);
    KAVACH_ASSERT(res.miqp_telemetry.has_value());
    KAVACH_ASSERT(res.miqp_telemetry->qp_relaxations_solved > 0);
}
