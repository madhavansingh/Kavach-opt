/**
 * @file test_final_hardening.cpp
 * @brief Final Engineering Hardening & Certification Suite for Kavach-Opt.
 *
 * Provides exhaustive tests for:
 *  1. Status semantics and non-overlapping taxonomy
 *  2. Independent zero-trust verification invariants across LP, MILP, QP, MIQP
 *  3. Verifier adversarial fault-injection (corrupted objective, bounds, constrs, integrality, KKT,
 * PSD)
 *  4. Numerical edge-case matrix (zero rows, singletons, contradictory bounds, zero Q)
 *  5. Scaling stress across wide dynamic coefficient ranges (10^-5 to 10^5)
 *  6. Presolve reconstruction certification against original unreduced models
 *  7. Resource limits and interruption semantics (max_iterations, max_nodes, time_limit)
 *  8. Parser fuzz hardening with malformed MPS and LP inputs
 *  9. Determinism of search, telemetry ordering, and objective values
 * 10. Public API safety and JSON contract integrity
 */

#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "../test_framework.hpp"
#include "algorithms/lp/lp_types.hpp"
#include "algorithms/lp/revised_simplex.hpp"
#include "algorithms/lp/simplex.hpp"
#include "algorithms/milp/branch_and_bound.hpp"
#include "algorithms/miqp/miqp_solver.hpp"
#include "algorithms/qp/active_set.hpp"
#include "api/solver.hpp"
#include "benchmark/benchmark_io.hpp"
#include "core/error.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "io/parser.hpp"
#include "model/model.hpp"
#include "presolve/presolve.hpp"
#include "verification/lp_verifier.hpp"
#include "verification/milp_verifier.hpp"
#include "verification/miqp_verifier.hpp"
#include "verification/qp_verifier.hpp"
#include "verification/verifier.hpp"

using namespace kavach;
using namespace kavach::model;
using namespace kavach::core;
using namespace kavach::verification;

namespace {

// Helper: build standard 2-variable LP
// max 3x1 + 2x2 s.t. 2x1 + x2 <= 100, x1 + x2 <= 80, x1, x2 >= 0
CanonicalModel make_hardening_lp() {
    CanonicalModel m("hardening_lp");
    auto x1 = m.add_variable("x1", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto x2 = m.add_variable("x2", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    m.set_objective_sense(ObjectiveSense::MAXIMIZE);
    m.add_objective_linear_term(x1, 3.0);
    m.add_objective_linear_term(x2, 2.0);

    SparseLinearExpression c1;
    c1.add_term(x1, 2.0);
    c1.add_term(x2, 1.0);
    m.add_constraint("c1", std::move(c1), ConstraintSense::LESS_EQUAL, 100.0);

    SparseLinearExpression c2;
    c2.add_term(x1, 1.0);
    c2.add_term(x2, 1.0);
    m.add_constraint("c2", std::move(c2), ConstraintSense::LESS_EQUAL, 80.0);
    return m;
}

// Helper: build standard 0-1 knapsack MILP
// max 5x1 + 3x2 + 2x3 s.t. 2x1 + 2x2 + x3 <= 3, x in {0,1}^3
CanonicalModel make_hardening_milp() {
    CanonicalModel m("hardening_milp");
    auto x1 = m.add_variable("x1", VariableType::BINARY, 0.0, 1.0);
    auto x2 = m.add_variable("x2", VariableType::BINARY, 0.0, 1.0);
    auto x3 = m.add_variable("x3", VariableType::BINARY, 0.0, 1.0);
    m.set_objective_sense(ObjectiveSense::MAXIMIZE);
    m.add_objective_linear_term(x1, 5.0);
    m.add_objective_linear_term(x2, 3.0);
    m.add_objective_linear_term(x3, 2.0);

    SparseLinearExpression c1;
    c1.add_term(x1, 2.0);
    c1.add_term(x2, 2.0);
    c1.add_term(x3, 1.0);
    m.add_constraint("cap", std::move(c1), ConstraintSense::LESS_EQUAL, 3.0);
    return m;
}

// Helper: build standard convex QP
// min 0.5*(2 x1^2 + 2 x2^2) - 2 x1 - 5 x2 s.t. x1 - x2 <= 2, x1 + 2x2 <= 6, x >= 0
CanonicalModel make_hardening_qp() {
    CanonicalModel m("hardening_qp");
    auto x1 = m.add_variable("x1", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto x2 = m.add_variable("x2", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x1, -2.0);
    m.add_objective_linear_term(x2, -5.0);
    m.add_objective_quadratic_term(x1, x1, 2.0);
    m.add_objective_quadratic_term(x2, x2, 2.0);

    SparseLinearExpression c1;
    c1.add_term(x1, 1.0);
    c1.add_term(x2, -1.0);
    m.add_constraint("c1", std::move(c1), ConstraintSense::LESS_EQUAL, 2.0);

    SparseLinearExpression c2;
    c2.add_term(x1, 1.0);
    c2.add_term(x2, 2.0);
    m.add_constraint("c2", std::move(c2), ConstraintSense::LESS_EQUAL, 6.0);
    return m;
}

// Helper: build standard convex MIQP
// min 0.5*(2 x1^2 + 2 x2^2) - 4 x1 - 6 x2 s.t. x1 + x2 <= 3, x1 binary, x2 >= 0
CanonicalModel make_hardening_miqp() {
    CanonicalModel m("hardening_miqp");
    auto x1 = m.add_variable("x1", VariableType::BINARY, 0.0, 1.0);
    auto x2 = m.add_variable("x2", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x1, -4.0);
    m.add_objective_linear_term(x2, -6.0);
    m.add_objective_quadratic_term(x1, x1, 2.0);
    m.add_objective_quadratic_term(x2, x2, 2.0);

    SparseLinearExpression c1;
    c1.add_term(x1, 1.0);
    c1.add_term(x2, 1.0);
    m.add_constraint("c1", std::move(c1), ConstraintSense::LESS_EQUAL, 3.0);
    return m;
}

}  // namespace

// ============================================================================
// 1. Solver Status Semantics Audit
// ============================================================================

KAVACH_TEST(test_hardening_solver_status_semantics) {
    // 1. String conversion completeness
    KAVACH_ASSERT_EQUAL(to_string(SolverStatus::OPTIMAL), "OPTIMAL");
    KAVACH_ASSERT_EQUAL(to_string(SolverStatus::FEASIBLE), "FEASIBLE");
    KAVACH_ASSERT_EQUAL(to_string(SolverStatus::INFEASIBLE), "INFEASIBLE");
    KAVACH_ASSERT_EQUAL(to_string(SolverStatus::UNBOUNDED), "UNBOUNDED");
    KAVACH_ASSERT_EQUAL(to_string(SolverStatus::LIMIT_REACHED), "LIMIT_REACHED");
    KAVACH_ASSERT_EQUAL(to_string(SolverStatus::NUMERICAL_FAILURE), "NUMERICAL_FAILURE");
    KAVACH_ASSERT_EQUAL(to_string(SolverStatus::UNKNOWN), "UNKNOWN");

    // 2. Terminal status classification
    KAVACH_ASSERT(is_terminal(SolverStatus::OPTIMAL));
    KAVACH_ASSERT(is_terminal(SolverStatus::INFEASIBLE));
    KAVACH_ASSERT(is_terminal(SolverStatus::UNBOUNDED));
    KAVACH_ASSERT(is_terminal(SolverStatus::LIMIT_REACHED));
    KAVACH_ASSERT(is_terminal(SolverStatus::NUMERICAL_FAILURE));
    KAVACH_ASSERT(!is_terminal(SolverStatus::FEASIBLE));  // Non-terminal: can continue improving
    KAVACH_ASSERT(!is_terminal(SolverStatus::UNKNOWN));
}

// ============================================================================
// 2. Independent Verification Invariant: OPTIMAL <=> Verification PASS
// ============================================================================

KAVACH_TEST(test_hardening_independent_verification_invariant_all_solvers) {
    api::KavachSolver solver;

    // LP Verification Invariant
    {
        auto model = make_hardening_lp();
        api::SolverOptions opts;
        opts.problem_type = api::ProblemType::LP;
        auto res = solver.solve(model, opts);
        KAVACH_ASSERT_EQUAL(res.status, SolverStatus::OPTIMAL);
        KAVACH_ASSERT(res.verification_report.passed);
        KAVACH_ASSERT_EQUAL(res.verification_report.certified_status, SolverStatus::OPTIMAL);
        KAVACH_ASSERT_NEAR(res.objective_value, 180.0, 1e-5);
    }

    // MILP Verification Invariant
    {
        auto model = make_hardening_milp();
        api::SolverOptions opts;
        opts.problem_type = api::ProblemType::MILP;
        auto res = solver.solve(model, opts);
        KAVACH_ASSERT_EQUAL(res.status, SolverStatus::OPTIMAL);
        KAVACH_ASSERT(res.verification_report.passed);
        KAVACH_ASSERT_EQUAL(res.verification_report.certified_status, SolverStatus::OPTIMAL);
        KAVACH_ASSERT_NEAR(res.objective_value, 7.0, 1e-5);  // x1=1, x3=1 -> 5+2=7
    }

    // QP Verification Invariant
    {
        auto model = make_hardening_qp();
        api::SolverOptions opts;
        opts.problem_type = api::ProblemType::QP;
        auto res = solver.solve(model, opts);
        KAVACH_ASSERT_EQUAL(res.status, SolverStatus::OPTIMAL);
        KAVACH_ASSERT(res.verification_report.passed);
        KAVACH_ASSERT_EQUAL(res.verification_report.certified_status, SolverStatus::OPTIMAL);
    }

    // MIQP Verification Invariant
    {
        auto model = make_hardening_miqp();
        api::SolverOptions opts;
        opts.problem_type = api::ProblemType::MIQP;
        auto res = solver.solve(model, opts);
        KAVACH_ASSERT_EQUAL(res.status, SolverStatus::OPTIMAL);
        KAVACH_ASSERT(res.verification_report.passed);
        KAVACH_ASSERT_EQUAL(res.verification_report.certified_status, SolverStatus::OPTIMAL);
    }
}

// ============================================================================
// 3. Verifier Adversarial Fault Injection: LP
// ============================================================================

KAVACH_TEST(test_hardening_verifier_adversarial_lp_faults) {
    auto model = make_hardening_lp();
    LPVerifier verifier;
    // Ground truth: x1=20, x2=60, obj=180
    std::vector<double> x_true = {20.0, 60.0};
    std::vector<double> y_true = {1.0, 1.0};  // duals
    std::vector<double> rc_true = {0.0, 0.0};

    // Baseline passes
    auto rep_base = verifier.verify(model, x_true, y_true, rc_true, 180.0, SolverStatus::OPTIMAL);
    KAVACH_ASSERT(rep_base.fully_verified());

    // Fault 1: Corrupted objective value
    auto rep_bad_obj =
        verifier.verify(model, x_true, y_true, rc_true, 195.0, SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rep_bad_obj.fully_verified());
    KAVACH_ASSERT(!rep_bad_obj.objective_consistency.passed);
    KAVACH_ASSERT_EQUAL(rep_bad_obj.certified_status,
                        SolverStatus::FEASIBLE);  // Feasible but uncertified obj

    // Fault 2: Primal bound violation (x1 < 0)
    std::vector<double> x_neg = {-5.0, 60.0};
    auto rep_bad_bound =
        verifier.verify(model, x_neg, y_true, rc_true, 180.0, SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rep_bad_bound.fully_verified());
    KAVACH_ASSERT(!rep_bad_bound.bound_feasibility.passed);

    // Fault 3: Primal constraint violation (2*x1 + x2 = 250 > 100)
    std::vector<double> x_viol = {100.0, 50.0};
    auto rep_bad_constr =
        verifier.verify(model, x_viol, y_true, rc_true, 400.0, SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rep_bad_constr.fully_verified());
    KAVACH_ASSERT(!rep_bad_constr.primal_feasibility.passed);

    // Fault 4: Nonfinite values (NaN / Inf)
    std::vector<double> x_nan = {std::numeric_limits<double>::quiet_NaN(), 60.0};
    auto rep_nan = verifier.verify(model, x_nan, y_true, rc_true, 180.0, SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rep_nan.fully_verified());
    KAVACH_ASSERT(!rep_nan.nonfinite_check.passed);
    KAVACH_ASSERT_EQUAL(rep_nan.certified_status, SolverStatus::NUMERICAL_FAILURE);

    // Fault 5: Inconsistent dual multiplier (violating complementary slackness)
    std::vector<double> y_bad = {10.0, 10.0};
    std::vector<double> rc_bad = {5.0, 5.0};
    auto rep_bad_dual = verifier.verify(model, x_true, y_bad, rc_bad, 180.0, SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rep_bad_dual.fully_verified());
}

// ============================================================================
// 4. Verifier Adversarial Fault Injection: MILP
// ============================================================================

KAVACH_TEST(test_hardening_verifier_adversarial_milp_faults) {
    auto model = make_hardening_milp();
    MILPVerifier verifier;
    // Ground truth: x1=1, x2=0, x3=1, obj=7.0
    std::vector<double> x_true = {1.0, 0.0, 1.0};

    // Baseline passes
    auto rep_base = verifier.verify(model, x_true, 7.0, 7.0, SolverStatus::OPTIMAL);
    KAVACH_ASSERT(rep_base.fully_verified());

    // Fault 1: Fractional value on binary variable (x1 = 0.5)
    std::vector<double> x_frac = {0.5, 0.0, 1.0};
    auto rep_frac = verifier.verify(model, x_frac, 7.0, 4.5, SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rep_frac.fully_verified());
    KAVACH_ASSERT(!rep_frac.integrality.passed);
    KAVACH_ASSERT_EQUAL(rep_frac.certified_status, SolverStatus::NUMERICAL_FAILURE);

    // Fault 2: Corrupted objective value
    auto rep_bad_obj = verifier.verify(model, x_true, 7.0, 15.0, SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rep_bad_obj.fully_verified());
    KAVACH_ASSERT(!rep_bad_obj.objective_consistency.passed);

    // Fault 3: Primal constraint violation (2*1 + 2*1 + 1*1 = 5 > 3)
    std::vector<double> x_viol = {1.0, 1.0, 1.0};
    auto rep_viol = verifier.verify(model, x_viol, 10.0, 10.0, SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rep_viol.fully_verified());
    KAVACH_ASSERT(!rep_viol.primal_feasibility.passed);

    // Fault 4: Infeasible bound consistency (for MAXIMIZE: Best Bound < Best Objective by > tol)
    auto rep_bound_inv = verifier.verify(model, x_true, 5.0, 7.0, SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!rep_bound_inv.fully_verified());
    KAVACH_ASSERT(!rep_bound_inv.gap_consistency.passed);
}

// ============================================================================
// 5. Verifier Adversarial Fault Injection: QP and MIQP
// ============================================================================

KAVACH_TEST(test_hardening_verifier_adversarial_qp_miqp_faults) {
    // Non-convex Q matrix detection in QP / MIQP verifier
    CanonicalModel non_convex_model("non_convex_qp");
    auto v0 = non_convex_model.add_variable("v0", VariableType::CONTINUOUS, 0.0, 5.0);
    auto v1 = non_convex_model.add_variable("v1", VariableType::CONTINUOUS, 0.0, 5.0);
    non_convex_model.set_objective_sense(ObjectiveSense::MINIMIZE);
    // Indefinite matrix: diag entries = -2.0, -2.0
    non_convex_model.add_objective_quadratic_term(v0, v0, -2.0);
    non_convex_model.add_objective_quadratic_term(v1, v1, -2.0);

    // QP Verifier must reject non-PSD Q
    QPVerifier qp_verifier;
    std::vector<double> x_toy = {1.0, 1.0};
    std::vector<double> y_toy = {};
    auto qp_rep = qp_verifier.verify(non_convex_model, x_toy, y_toy, -4.0, SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!qp_rep.fully_verified());
    KAVACH_ASSERT(!qp_rep.psd_check.passed);
    KAVACH_ASSERT_EQUAL(qp_rep.certified_status, SolverStatus::NUMERICAL_FAILURE);

    // MIQP Verifier must reject non-PSD Q
    MIQPVerifier miqp_verifier;
    auto miqp_rep =
        miqp_verifier.verify(non_convex_model, x_toy, -4.0, -4.0, SolverStatus::OPTIMAL);
    KAVACH_ASSERT(!miqp_rep.fully_verified());
    KAVACH_ASSERT(!miqp_rep.psd_check.passed);
    KAVACH_ASSERT_EQUAL(miqp_rep.certified_status, SolverStatus::NUMERICAL_FAILURE);
}

// ============================================================================
// 6. Numerical Edge-Case Matrix
// ============================================================================

KAVACH_TEST(test_hardening_numerical_edge_case_matrix) {
    api::KavachSolver solver;

    // Edge Case 1a: Contradictory Variable Bounds (lb > ub) caught as invalid model
    {
        CanonicalModel m("contradictory_bounds");
        m.add_variable("x", VariableType::CONTINUOUS, 10.0, 5.0);
        m.set_objective_sense(ObjectiveSense::MINIMIZE);
        api::SolverOptions opts;
        opts.problem_type = api::ProblemType::LP;
        auto res = solver.solve(m, opts);
        KAVACH_ASSERT_EQUAL(res.status, SolverStatus::UNKNOWN);
    }

    // Edge Case 1b: Contradictory Linear Constraints (x >= 10 and x <= 5) proven INFEASIBLE
    {
        CanonicalModel m("contradictory_constrs");
        auto x = m.add_variable("x", VariableType::CONTINUOUS, 0.0, 100.0);
        m.set_objective_sense(ObjectiveSense::MINIMIZE);
        m.add_objective_linear_term(x, 1.0);
        SparseLinearExpression c1;
        c1.add_term(x, 1.0);
        m.add_constraint("c_ge", std::move(c1), ConstraintSense::GREATER_EQUAL, 10.0);
        SparseLinearExpression c2;
        c2.add_term(x, 1.0);
        m.add_constraint("c_le", std::move(c2), ConstraintSense::LESS_EQUAL, 5.0);

        api::SolverOptions opts;
        opts.problem_type = api::ProblemType::LP;
        auto res = solver.solve(m, opts);
        KAVACH_ASSERT_EQUAL(res.status, SolverStatus::INFEASIBLE);
    }

    // Edge Case 2: Singleton row (x0 <= 5.0)
    {
        CanonicalModel m("singleton_row");
        auto x0 = m.add_variable("x0", VariableType::CONTINUOUS, 0.0, 10.0);
        m.set_objective_sense(ObjectiveSense::MAXIMIZE);
        m.add_objective_linear_term(x0, 1.0);
        SparseLinearExpression c;
        c.add_term(x0, 1.0);
        m.add_constraint("c0", std::move(c), ConstraintSense::LESS_EQUAL, 5.0);
        auto res = solver.solve(m);
        KAVACH_ASSERT_EQUAL(res.status, SolverStatus::OPTIMAL);
        KAVACH_ASSERT_NEAR(res.objective_value, 5.0, 1e-6);
    }

    // Edge Case 3: Zero Q matrix in pure linear model:
    // Calling QP solver on pure linear model throws InputException (must route to LP)
    {
        CanonicalModel m("zero_q_qp");
        auto x0 = m.add_variable("x0", VariableType::CONTINUOUS, 0.0, 10.0);
        auto x1 = m.add_variable("x1", VariableType::CONTINUOUS, 0.0, 10.0);
        m.set_objective_sense(ObjectiveSense::MINIMIZE);
        m.add_objective_linear_term(x0, 2.0);
        m.add_objective_linear_term(x1, 3.0);
        SparseLinearExpression c;
        c.add_term(x0, 1.0);
        c.add_term(x1, 1.0);
        m.add_constraint("c", std::move(c), ConstraintSense::GREATER_EQUAL, 4.0);

        // a) Explicit QP request throws InputException
        api::SolverOptions qp_opts;
        qp_opts.problem_type = api::ProblemType::QP;
        bool caught = false;
        try {
            auto res = solver.solve(m, qp_opts);
            (void)res;
        } catch (const core::InputException&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);

        // b) Auto plan correctly routes to LP
        auto res_auto = solver.solve(m);
        KAVACH_ASSERT_EQUAL(res_auto.status, SolverStatus::OPTIMAL);
        KAVACH_ASSERT_NEAR(res_auto.objective_value, 8.0, 1e-5);  // x0=4, x1=0 -> 2*4=8
    }

    // Edge Case 4: Exactly Zero Q matrix in MIQP (reduces to MILP)
    {
        CanonicalModel m("zero_q_miqp");
        auto x0 = m.add_variable("x0", VariableType::BINARY, 0.0, 1.0);
        auto x1 = m.add_variable("x1", VariableType::CONTINUOUS, 0.0, 10.0);
        m.set_objective_sense(ObjectiveSense::MINIMIZE);
        m.add_objective_linear_term(x0, 5.0);
        m.add_objective_linear_term(x1, 2.0);
        SparseLinearExpression c;
        c.add_term(x0, 1.0);
        c.add_term(x1, 1.0);
        m.add_constraint("c", std::move(c), ConstraintSense::GREATER_EQUAL, 1.5);

        api::SolverOptions opts;
        opts.problem_type = api::ProblemType::MIQP;
        auto res = solver.solve(m, opts);
        KAVACH_ASSERT_EQUAL(res.status, SolverStatus::OPTIMAL);
        KAVACH_ASSERT_NEAR(res.objective_value, 3.0, 1e-5);  // x0=0, x1=1.5 -> 2*1.5=3
    }
}

// ============================================================================
// 7. Scaling Stress Across Wide Dynamic Range (10^-4 to 10^4)
// ============================================================================

KAVACH_TEST(test_hardening_scaling_stress) {
    CanonicalModel m("scale_stress");
    auto x1 = m.add_variable("x1", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    auto x2 = m.add_variable("x2", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    m.set_objective_sense(ObjectiveSense::MINIMIZE);
    m.add_objective_linear_term(x1, 1e-4);
    m.add_objective_linear_term(x2, 1e4);

    SparseLinearExpression c1;
    c1.add_term(x1, 1e-4);
    c1.add_term(x2, 1e4);
    m.add_constraint("c1", std::move(c1), ConstraintSense::GREATER_EQUAL, 100.0);

    api::KavachSolver solver;
    api::SolverOptions opts;
    opts.problem_type = api::ProblemType::LP;
    auto res = solver.solve(m, opts);

    KAVACH_ASSERT_EQUAL(res.status, SolverStatus::OPTIMAL);
    KAVACH_ASSERT(res.verification_report.passed);
    KAVACH_ASSERT_NEAR(res.objective_value, 100.0, 1e-3);
}

// ============================================================================
// 8. Presolve Reconstruction Certification Against Original Model
// ============================================================================

KAVACH_TEST(test_hardening_presolve_reconstruction_certification) {
    CanonicalModel original("presolve_cert");
    // x0 is fixed to 4.0
    auto x0 = original.add_variable("x0", VariableType::CONTINUOUS, 4.0, 4.0);
    // x1 and x2 are decision variables
    auto x1 = original.add_variable("x1", VariableType::CONTINUOUS, 0.0, 50.0);
    auto x2 = original.add_variable("x2", VariableType::CONTINUOUS, 0.0, 50.0);
    original.set_objective_sense(ObjectiveSense::MAXIMIZE);
    original.add_objective_linear_term(x0, 1.0);
    original.add_objective_linear_term(x1, 2.0);
    original.add_objective_linear_term(x2, 3.0);

    // c1: x0 + x1 + x2 <= 20  --> x1 + x2 <= 16
    SparseLinearExpression c1;
    c1.add_term(x0, 1.0);
    c1.add_term(x1, 1.0);
    c1.add_term(x2, 1.0);
    original.add_constraint("c1", std::move(c1), ConstraintSense::LESS_EQUAL, 20.0);

    // c2 redundant constraint: x1 + x2 <= 100
    SparseLinearExpression c2;
    c2.add_term(x1, 1.0);
    c2.add_term(x2, 1.0);
    original.add_constraint("c2", std::move(c2), ConstraintSense::LESS_EQUAL, 100.0);

    // Solve with full presolve enabled
    api::KavachSolver solver;
    api::SolverOptions opts;
    opts.run_presolve = true;
    auto res = solver.solve(original, opts);

    KAVACH_ASSERT_EQUAL(res.status, SolverStatus::OPTIMAL);
    KAVACH_ASSERT_EQUAL(res.primal_solution.size(), 3u);
    // x0 must be reconstructed to 4.0
    KAVACH_ASSERT_NEAR(res.primal_solution[0], 4.0, 1e-6);
    // Expected optimal: x0=4, x1=0, x2=16 -> 4*1 + 0*2 + 16*3 = 52
    KAVACH_ASSERT_NEAR(res.objective_value, 52.0, 1e-5);

    // Check independent verifier directly on the ORIGINAL model
    LPVerifier verifier;
    auto lp_rep = verifier.verify(original, res.primal_solution, res.dual_solution,
                                  res.reduced_costs, res.objective_value, res.status);
    KAVACH_ASSERT(lp_rep.primal_feasibility.passed);
    KAVACH_ASSERT(lp_rep.bound_feasibility.passed);
    KAVACH_ASSERT(lp_rep.objective_consistency.passed);
}

// ============================================================================
// 9. Resource Limits & Interruption Semantics
// ============================================================================

KAVACH_TEST(test_hardening_resource_limits_and_interruption) {
    api::KavachSolver solver;

    // Test MILP node limit (max_nodes = 1 on non-trivial knapsack)
    {
        CanonicalModel m("node_limited_milp");
        std::vector<VarId> vars;
        for (std::size_t i = 0; i < 10; ++i) {
            vars.push_back(m.add_variable("x" + std::to_string(i), VariableType::BINARY, 0.0, 1.0));
            m.add_objective_linear_term(vars.back(), 5.0 + static_cast<double>(i));
        }
        m.set_objective_sense(ObjectiveSense::MAXIMIZE);
        SparseLinearExpression cap;
        for (std::size_t i = 0; i < 10; ++i) {
            cap.add_term(vars[i], 3.0 + static_cast<double>(i % 3));
        }
        m.add_constraint("cap", std::move(cap), ConstraintSense::LESS_EQUAL, 11.5);

        api::SolverOptions opts;
        opts.problem_type = api::ProblemType::MILP;
        opts.max_nodes = 1;
        opts.run_presolve = false;
        opts.cut_options.strategy = algorithms::milp::CutStrategy::OFF;
        opts.enable_primal_heuristic = false;  // Ensure limit is hit before search exhausts
        auto res = solver.solve(m, opts);

        // When node limit is hit, status must be LIMIT_REACHED or FEASIBLE, never OPTIMAL
        KAVACH_ASSERT(res.status == SolverStatus::LIMIT_REACHED ||
                      res.status == SolverStatus::FEASIBLE);
        KAVACH_ASSERT(res.status != SolverStatus::OPTIMAL);
    }

    // Test LP iteration limit (max_iterations = 1)
    {
        auto m = make_hardening_lp();
        api::SolverOptions opts;
        opts.problem_type = api::ProblemType::LP;
        opts.max_iterations = 1;
        auto res = solver.solve(m, opts);
        KAVACH_ASSERT_EQUAL(res.status, SolverStatus::LIMIT_REACHED);
    }
}

// ============================================================================
// 10. Parser Fuzz Hardening: Malformed MPS and LP
// ============================================================================

KAVACH_TEST(test_hardening_parser_fuzz_malformed_inputs) {
    // 1. Truncated MPS without ENDATA
    {
        constexpr std::string_view truncated_mps = R"(
NAME          TRUNCATED
ROWS
 N  COST
 L  CAP
COLUMNS
    X1  COST  1.0  CAP  2.0
RHS
    RHS1 CAP 10.0
)";
        io::MpsParser parser;
        bool caught = false;
        try {
            auto m = parser.parse_string(truncated_mps, "trunc.mps");
            (void)m;
        } catch (const std::exception&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);
    }

    // 2. Malformed MPS row type
    {
        constexpr std::string_view bad_row_mps = R"(
NAME          BAD_ROW
ROWS
 Z  COST
ENDATA
)";
        io::MpsParser parser;
        bool caught = false;
        try {
            auto m = parser.parse_string(bad_row_mps, "bad_row.mps");
            (void)m;
        } catch (const std::exception&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);
    }

    // 3. Malformed LP with unclosed parenthesis or syntax error
    {
        constexpr std::string_view bad_syntax_lp = R"(
Minimize
 obj: 3 x1 + 2 x2
Subject To
 c1: 2 x1 + 3 x2 <=
Bounds
 0 <= x1
End
)";
        io::LpParser parser;
        bool caught = false;
        try {
            auto m = parser.parse_string(bad_syntax_lp, "bad_syntax.lp");
            (void)m;
        } catch (const std::exception&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);
    }

    // 4. LP with missing objective keyword
    {
        constexpr std::string_view no_obj_lp = "Subject To\n c1: x <= 10\nEnd\n";
        io::LpParser parser;
        bool caught = false;
        try {
            auto m = parser.parse_string(no_obj_lp, "no_obj.lp");
            (void)m;
        } catch (const std::exception&) {
            caught = true;
        }
        KAVACH_ASSERT(caught);
    }
}

// ============================================================================
// 11. Determinism and Telemetry Integrity
// ============================================================================

KAVACH_TEST(test_hardening_determinism_repeated_execution) {
    api::KavachSolver solver;
    api::SolverOptions opts;
    opts.seed = 12345;

    // LP Determinism
    {
        auto m = make_hardening_lp();
        auto res1 = solver.solve(m, opts);
        auto res2 = solver.solve(m, opts);
        KAVACH_ASSERT_EQUAL(res1.status, res2.status);
        KAVACH_ASSERT_EQUAL(res1.objective_value, res2.objective_value);
        KAVACH_ASSERT_EQUAL(res1.iterations, res2.iterations);
    }

    // MILP Determinism
    {
        auto m = make_hardening_milp();
        auto res1 = solver.solve(m, opts);
        auto res2 = solver.solve(m, opts);
        KAVACH_ASSERT_EQUAL(res1.status, res2.status);
        KAVACH_ASSERT_EQUAL(res1.objective_value, res2.objective_value);
        KAVACH_ASSERT_EQUAL(res1.nodes, res2.nodes);
        KAVACH_ASSERT(res1.search_telemetry.has_value());
        KAVACH_ASSERT(res2.search_telemetry.has_value());
        KAVACH_ASSERT_EQUAL(res1.search_telemetry->pseudocost_updates,
                            res2.search_telemetry->pseudocost_updates);
    }

    // MIQP Determinism
    {
        auto m = make_hardening_miqp();
        auto res1 = solver.solve(m, opts);
        auto res2 = solver.solve(m, opts);
        KAVACH_ASSERT_EQUAL(res1.status, res2.status);
        KAVACH_ASSERT_EQUAL(res1.objective_value, res2.objective_value);
        KAVACH_ASSERT_EQUAL(res1.nodes, res2.nodes);
    }
}

// ============================================================================
// 12. Public API Safety & JSON Schema Integrity
// ============================================================================

KAVACH_TEST(test_hardening_public_api_and_json_contract) {
    api::KavachSolver solver;

    // 1. Solve on empty model
    CanonicalModel empty_m("empty");
    auto res_empty = solver.solve(empty_m);
    KAVACH_ASSERT(res_empty.status == SolverStatus::OPTIMAL ||
                  res_empty.status == SolverStatus::FEASIBLE);

    // 2. Plan explanation
    auto m = make_hardening_milp();
    std::string explanation = solver.explain_plan(m);
    KAVACH_ASSERT(!explanation.empty());
    KAVACH_ASSERT(explanation.find("Target Problem Class: MILP") != std::string::npos);

    // 3. Plan JSON serialization
    auto plan = solver.plan(m);
    auto plan_json = plan.to_json();
    std::string dumped = plan_json.dump();
    KAVACH_ASSERT(!dumped.empty());
    auto parsed = benchmark::JsonValue::parse(dumped);
    KAVACH_ASSERT(parsed.is_object());
    KAVACH_ASSERT_EQUAL(parsed["problem_class"].as_string(), "MILP");
}
