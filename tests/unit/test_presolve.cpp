#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../test_framework.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "io/parser.hpp"
#include "model/model.hpp"
#include "presolve/presolve.hpp"

using namespace kavach;
using namespace kavach::model;
using namespace kavach::presolve;

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

// Feasibility evaluator for presolve test verification
[[nodiscard]] double compute_max_constraint_violation(const CanonicalModel& model,
                                                      const std::vector<double>& solution) {
    double max_violation = 0.0;
    for (std::size_t i = 0; i < model.num_constraints(); ++i) {
        const auto& c = model.constraint(static_cast<ConstrId>(i));
        double lhs = 0.0;
        for (const auto& term : c.expression.terms()) {
            lhs += term.coefficient * solution[term.var_id];
        }
        double viol = 0.0;
        if (c.sense == ConstraintSense::LESS_EQUAL) {
            viol = std::max(0.0, lhs - c.rhs);
        } else if (c.sense == ConstraintSense::GREATER_EQUAL) {
            viol = std::max(0.0, c.rhs - lhs);
        } else {  // EQUAL
            viol = std::abs(lhs - c.rhs);
        }
        max_violation = std::max(max_violation, viol);
    }
    return max_violation;
}

// Evaluate total objective value
[[nodiscard]] double compute_objective_value(const CanonicalModel& model,
                                             const std::vector<double>& solution) {
    const auto& obj = model.objective();
    double val = obj.offset;
    for (const auto& lt : obj.linear_terms.terms()) {
        val += lt.coefficient * solution[lt.var_id];
    }
    for (const auto& qt : obj.quadratic_terms.terms()) {
        val += qt.coefficient * solution[qt.var1_id] * solution[qt.var2_id];
    }
    return val;
}

}  // namespace

// ============================================================================
// A. Fixed-Variable Substitution Tests
// ============================================================================

KAVACH_TEST(test_presolve_fixed_variable_substitution) {
    CanonicalModel model("fixed_var_test");
    auto x0 = model.add_variable("x0", VariableType::CONTINUOUS, 0.0, 10.0);
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 5.0, 5.0);  // Fixed to 5.0
    auto x2 = model.add_variable("x2", VariableType::CONTINUOUS, 2.0, 2.0);  // Fixed to 2.0

    // Objective: 3*x0 + 4*x1 - 2*x2 + 10
    model.set_objective_offset(10.0);
    model.set_objective_linear_coefficient(x0, 3.0);
    model.set_objective_linear_coefficient(x1, 4.0);
    model.set_objective_linear_coefficient(x2, -2.0);

    // c0: 2*x0 + 3*x1 <= 25 -> 2*x0 + 15 <= 25 -> 2*x0 <= 10
    SparseLinearExpression e0;
    e0.add_term(x0, 2.0);
    e0.add_term(x1, 3.0);
    model.add_constraint("c0", std::move(e0), ConstraintSense::LESS_EQUAL, 25.0);

    // c1: x0 - x2 >= 1 -> x0 - 2 >= 1 -> x0 >= 3
    SparseLinearExpression e1;
    e1.add_term(x0, 1.0);
    e1.add_term(x2, -1.0);
    model.add_constraint("c1", std::move(e1), ConstraintSense::GREATER_EQUAL, 1.0);

    PresolveOptions opt;
    opt.enable_singleton_bound_tightening = false;
    opt.enable_empty_column_removal = false;
    Presolver presolver;
    auto res = presolver.apply(model, opt);

    KAVACH_ASSERT(res.has_reduced_model());
    KAVACH_ASSERT_EQUAL(res.reduced_model->num_variables(), 1u);
    KAVACH_ASSERT_EQUAL(res.stats.fixed_variables, 2u);

    // Objective offset shift: original 10.0 + 4*5.0 - 2*2.0 = 10 + 20 - 4 = 26.0
    KAVACH_ASSERT_NEAR(res.reduced_model->objective_offset(), 26.0, 1e-9);

    // Check constraint c0 RHS in reduced model: 25 - 15 = 10.0
    auto c0_id = *res.reduced_model->find_constraint_id("c0");
    KAVACH_ASSERT_NEAR(res.reduced_model->constraint(c0_id).rhs, 10.0, 1e-9);

    // Solution reconstruction & objective preservation test
    std::vector<double> reduced_sol = {4.0};  // x0 = 4.0
    auto orig_sol = res.reconstruction_map.restore_solution(reduced_sol);
    KAVACH_ASSERT_EQUAL(orig_sol.size(), 3u);
    KAVACH_ASSERT_NEAR(orig_sol[0], 4.0, 1e-9);
    KAVACH_ASSERT_NEAR(orig_sol[1], 5.0, 1e-9);
    KAVACH_ASSERT_NEAR(orig_sol[2], 2.0, 1e-9);

    double obj_orig = compute_objective_value(model, orig_sol);
    double obj_red = compute_objective_value(*res.reduced_model, reduced_sol);
    KAVACH_ASSERT_NEAR(obj_orig, obj_red, 1e-9);
    KAVACH_ASSERT_NEAR(compute_max_constraint_violation(model, orig_sol), 0.0, 1e-9);
}

// ============================================================================
// B. Empty Row Analysis Tests
// ============================================================================

KAVACH_TEST(test_presolve_empty_rows_satisfied) {
    CanonicalModel model("empty_rows_sat");
    auto x0 = model.add_variable("x0", VariableType::CONTINUOUS, 0.0, 10.0);

    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 10.0);

    // Satisfied empty rows: 0 <= 5, 0 >= -2, 0 == 0
    model.add_constraint("empty_le", SparseLinearExpression{}, ConstraintSense::LESS_EQUAL, 5.0);
    model.add_constraint("empty_ge", SparseLinearExpression{}, ConstraintSense::GREATER_EQUAL,
                         -2.0);
    model.add_constraint("empty_eq", SparseLinearExpression{}, ConstraintSense::EQUAL, 0.0);

    // Normal multi-variable row (not singleton)
    SparseLinearExpression e;
    e.add_term(x0, 1.0);
    e.add_term(x1, 1.0);
    model.add_constraint("normal", std::move(e), ConstraintSense::LESS_EQUAL, 8.0);

    Presolver presolver;
    auto res = presolver.apply(model);

    KAVACH_ASSERT(res.has_reduced_model());
    KAVACH_ASSERT_EQUAL(res.stats.empty_rows_removed, 3u);
    KAVACH_ASSERT_EQUAL(res.reduced_model->num_constraints(), 1u);
}

KAVACH_TEST(test_presolve_empty_rows_infeasible) {
    // Impossible empty row: 0 <= -5
    {
        CanonicalModel m1("infeasible_empty_le");
        m1.add_variable("x");
        m1.add_constraint("c", SparseLinearExpression{}, ConstraintSense::LESS_EQUAL, -5.0);
        Presolver p;
        auto res = p.apply(m1);
        KAVACH_ASSERT(res.is_infeasible());
    }

    // Impossible empty row: 0 >= 5
    {
        CanonicalModel m2("infeasible_empty_ge");
        m2.add_variable("x");
        m2.add_constraint("c", SparseLinearExpression{}, ConstraintSense::GREATER_EQUAL, 5.0);
        Presolver p;
        auto res = p.apply(m2);
        KAVACH_ASSERT(res.is_infeasible());
    }

    // Impossible empty row: 0 == 5
    {
        CanonicalModel m3("infeasible_empty_eq");
        m3.add_variable("x");
        m3.add_constraint("c", SparseLinearExpression{}, ConstraintSense::EQUAL, 5.0);
        Presolver p;
        auto res = p.apply(m3);
        KAVACH_ASSERT(res.is_infeasible());
    }
}

// ============================================================================
// C. Singleton Bound Tightening Tests
// ============================================================================

KAVACH_TEST(test_presolve_singleton_bound_tightening) {
    CanonicalModel model("singleton_test");
    auto x0 = model.add_variable("x0", VariableType::CONTINUOUS, 0.0, 100.0);
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, -50.0, 50.0);

    // 2*x0 <= 10 -> x0 <= 5.0
    SparseLinearExpression e0;
    e0.add_term(x0, 2.0);
    model.add_constraint("s0", std::move(e0), ConstraintSense::LESS_EQUAL, 10.0);

    // -3*x1 <= 15 -> x1 >= -5.0
    SparseLinearExpression e1;
    e1.add_term(x1, -3.0);
    model.add_constraint("s1", std::move(e1), ConstraintSense::LESS_EQUAL, 15.0);

    // Linking constraint so variables are not eliminated as empty columns
    SparseLinearExpression e2;
    e2.add_term(x0, 1.0);
    e2.add_term(x1, 1.0);
    model.add_constraint("c_link", std::move(e2), ConstraintSense::LESS_EQUAL, 50.0);

    Presolver presolver;
    auto res = presolver.apply(model);

    KAVACH_ASSERT(res.has_reduced_model());
    KAVACH_ASSERT_EQUAL(res.stats.singleton_reductions, 2u);
    KAVACH_ASSERT_EQUAL(res.reduced_model->num_constraints(), 1u);
    KAVACH_ASSERT_EQUAL(res.reduced_model->num_variables(), 2u);

    auto red_x0 = res.reduced_model->variable(0);
    auto red_x1 = res.reduced_model->variable(1);

    KAVACH_ASSERT_NEAR(red_x0.upper_bound, 5.0, 1e-9);
    KAVACH_ASSERT_NEAR(red_x1.lower_bound, -5.0, 1e-9);
}

// ============================================================================
// D. Redundant Rows Tests
// ============================================================================

KAVACH_TEST(test_presolve_redundant_row_removal) {
    CanonicalModel model("redundant_test");
    auto x0 = model.add_variable("x0", VariableType::CONTINUOUS, 0.0, 3.0);
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 4.0);

    // x0 + x1 <= 10. Since max(x0 + x1) = 3 + 4 = 7 <= 10, this is redundant!
    SparseLinearExpression e0;
    e0.add_term(x0, 1.0);
    e0.add_term(x1, 1.0);
    model.add_constraint("red_le", std::move(e0), ConstraintSense::LESS_EQUAL, 10.0);

    // x0 + x1 >= -5. Since min(x0 + x1) = 0 >= -5, this is redundant!
    SparseLinearExpression e1;
    e1.add_term(x0, 1.0);
    e1.add_term(x1, 1.0);
    model.add_constraint("red_ge", std::move(e1), ConstraintSense::GREATER_EQUAL, -5.0);

    // Non-redundant constraint: x0 + x1 <= 5
    SparseLinearExpression e2;
    e2.add_term(x0, 1.0);
    e2.add_term(x1, 1.0);
    model.add_constraint("active", std::move(e2), ConstraintSense::LESS_EQUAL, 5.0);

    Presolver presolver;
    auto res = presolver.apply(model);

    KAVACH_ASSERT(res.has_reduced_model());
    KAVACH_ASSERT_EQUAL(res.stats.redundant_rows_removed, 2u);
    KAVACH_ASSERT_EQUAL(res.reduced_model->num_constraints(), 1u);
}

// ============================================================================
// E. Contradictory Bounds Tests
// ============================================================================

KAVACH_TEST(test_presolve_contradictory_bounds) {
    CanonicalModel model("contradiction_test");
    auto x0 = model.add_variable("x0", VariableType::CONTINUOUS, 10.0, 5.0);  // Already lb > ub
    SparseLinearExpression e;
    e.add_term(x0, 1.0);
    model.add_constraint("c", std::move(e), ConstraintSense::LESS_EQUAL, 10.0);

    Presolver presolver;
    auto res = presolver.apply(model);
    KAVACH_ASSERT(res.is_infeasible());
}

KAVACH_TEST(test_presolve_contradictory_singleton) {
    CanonicalModel model("singleton_contradiction");
    auto x0 = model.add_variable("x0", VariableType::CONTINUOUS, 0.0, 5.0);

    // 1*x0 >= 10 -> x0 >= 10. But ub is 5.0 -> Contradiction!
    SparseLinearExpression e;
    e.add_term(x0, 1.0);
    model.add_constraint("c", std::move(e), ConstraintSense::GREATER_EQUAL, 10.0);

    Presolver presolver;
    auto res = presolver.apply(model);
    KAVACH_ASSERT(res.is_infeasible());
}

// ============================================================================
// F. Unbounded / Unconstrained Variable Tests
// ============================================================================

KAVACH_TEST(test_presolve_unconstrained_unbounded_variable) {
    CanonicalModel model("unbounded_var");
    auto x0 = model.add_variable("x0", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    model.set_objective_sense(ObjectiveSense::MINIMIZE);
    model.set_objective_linear_coefficient(x0, -10.0);  // Minimize -10*x0 with x0 in [0, inf)

    Presolver presolver;
    auto res = presolver.apply(model);
    KAVACH_ASSERT(res.is_unbounded());
}

KAVACH_TEST(test_presolve_unconstrained_bounded_variable) {
    CanonicalModel model("bounded_unconstrained");
    auto x0 = model.add_variable("x0", VariableType::CONTINUOUS, 3.0, 15.0);
    model.set_objective_sense(ObjectiveSense::MINIMIZE);
    model.set_objective_linear_coefficient(x0, 5.0);  // Minimize 5*x0 -> optimal is x0 = 3.0

    Presolver presolver;
    auto res = presolver.apply(model);

    KAVACH_ASSERT(res.has_reduced_model());
    KAVACH_ASSERT_EQUAL(res.reduced_model->num_variables(), 0u);
    // Objective offset shift = 5 * 3.0 = 15.0
    KAVACH_ASSERT_NEAR(res.reduced_model->objective_offset(), 15.0, 1e-9);

    auto orig_sol = res.reconstruction_map.restore_solution({});
    KAVACH_ASSERT_EQUAL(orig_sol.size(), 1u);
    KAVACH_ASSERT_NEAR(orig_sol[0], 3.0, 1e-9);
}

// ============================================================================
// G. Quadratic Objective Tests
// ============================================================================

KAVACH_TEST(test_presolve_quadratic_fixed_variable) {
    CanonicalModel model("qp_fixed");
    auto x0 = model.add_variable("x0", VariableType::CONTINUOUS, 0.0, 10.0);
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 4.0, 4.0);  // Fixed to 4.0

    // Quadratic objective: 2*x0^2 + 3*x0*x1 + 5*x1^2
    // Substituting x1 = 4:
    // 2*x0^2 + 3*x0*(4) + 5*(16) = 2*x0^2 + 12*x0 + 80
    model.set_objective_quadratic_coefficient(x0, x0, 2.0);
    model.set_objective_quadratic_coefficient(x0, x1, 3.0);
    model.set_objective_quadratic_coefficient(x1, x1, 5.0);

    Presolver presolver;
    auto res = presolver.apply(model);

    KAVACH_ASSERT(res.has_reduced_model());
    KAVACH_ASSERT_EQUAL(res.reduced_model->num_variables(), 1u);

    // Check quadratic offset: 5 * 16 = 80
    KAVACH_ASSERT_NEAR(res.reduced_model->objective_offset(), 80.0, 1e-9);

    // Check linear term for x0: 12.0
    auto red_x0 = res.reduced_model->variable(0).id;
    KAVACH_ASSERT_NEAR(res.reduced_model->objective().linear_terms.get_coefficient(red_x0), 12.0,
                       1e-9);

    // Evaluate solution equivalence
    std::vector<double> red_sol = {2.0};  // x0 = 2.0
    auto orig_sol = res.reconstruction_map.restore_solution(red_sol);
    KAVACH_ASSERT_NEAR(orig_sol[0], 2.0, 1e-9);
    KAVACH_ASSERT_NEAR(orig_sol[1], 4.0, 1e-9);

    double obj_orig = compute_objective_value(model, orig_sol);
    double obj_red = compute_objective_value(*res.reduced_model, red_sol);
    KAVACH_ASSERT_NEAR(obj_orig, obj_red, 1e-9);
}

// ============================================================================
// H. Ranged Constraint Representation
// ============================================================================

KAVACH_TEST(test_presolve_ranged_representation) {
    CanonicalModel model("range_test");
    auto x0 = model.add_variable("x0", VariableType::CONTINUOUS, 0.0, 10.0);
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 10.0);

    // 10 <= x0 + x1 <= 15
    SparseLinearExpression e_lb;
    e_lb.add_term(x0, 1.0);
    e_lb.add_term(x1, 1.0);
    model.add_constraint("r_lb", std::move(e_lb), ConstraintSense::GREATER_EQUAL, 10.0);

    SparseLinearExpression e_ub;
    e_ub.add_term(x0, 1.0);
    e_ub.add_term(x1, 1.0);
    model.add_constraint("r_ub", std::move(e_ub), ConstraintSense::LESS_EQUAL, 15.0);

    Presolver presolver;
    auto res = presolver.apply(model);

    KAVACH_ASSERT(res.has_reduced_model());
    // Neither bound is redundant under [0, 10] box bounds: max is 20 > 15, min is 0 < 10
    KAVACH_ASSERT_EQUAL(res.reduced_model->num_constraints(), 2u);
}

// ============================================================================
// I. Multi-Pass Cascading Reductions
// ============================================================================

KAVACH_TEST(test_presolve_multi_pass_cascading) {
    CanonicalModel model("cascade_test");
    auto x0 = model.add_variable("x0", VariableType::CONTINUOUS, 0.0, 20.0);
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 20.0);
    auto x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, 20.0);

    // Pass 1: 2*x0 == 10 -> fixes x0 = 5.0
    SparseLinearExpression e0;
    e0.add_term(x0, 2.0);
    model.add_constraint("s0", std::move(e0), ConstraintSense::EQUAL, 10.0);

    // Constraint: x0 + 3*x1 <= 20
    // After pass 1 fixes x0 = 5, this becomes singleton 3*x1 <= 15
    // Pass 2: singleton row 3*x1 <= 15 tightens x1 <= 5.0 and removes the row!
    SparseLinearExpression e1;
    e1.add_term(x0, 1.0);
    e1.add_term(x1, 3.0);
    model.add_constraint("c1", std::move(e1), ConstraintSense::LESS_EQUAL, 20.0);

    // Linking constraint keeping x1 and x2 active in reduced model
    SparseLinearExpression e2;
    e2.add_term(x1, 1.0);
    e2.add_term(x2, 1.0);
    model.add_constraint("c2", std::move(e2), ConstraintSense::LESS_EQUAL, 10.0);

    Presolver presolver;
    auto res = presolver.apply(model);

    KAVACH_ASSERT(res.has_reduced_model());
    KAVACH_ASSERT(res.stats.passes_performed >= 2);
    KAVACH_ASSERT_EQUAL(res.reduced_model->num_variables(), 2u);
    KAVACH_ASSERT_EQUAL(res.reduced_model->num_constraints(), 1u);
    KAVACH_ASSERT_NEAR(res.reduced_model->variable(0).upper_bound, 5.0, 1e-9);
}

// ============================================================================
// J & K. Solution Reconstruction & Objective Preservation
// ============================================================================

KAVACH_TEST(test_presolve_reconstruction_and_objective_preservation) {
    CanonicalModel model("reconstruction_test");
    auto x0 = model.add_variable("x0", VariableType::CONTINUOUS, 1.0, 5.0);
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 3.0, 3.0);  // Fixed to 3.0
    auto x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, 10.0);

    // Objective: 2*x0 + 7*x1 + 4*x2 + 50.0
    model.set_objective_offset(50.0);
    model.set_objective_linear_coefficient(x0, 2.0);
    model.set_objective_linear_coefficient(x1, 7.0);
    model.set_objective_linear_coefficient(x2, 4.0);

    // c0: x0 + 2*x1 + x2 <= 20
    SparseLinearExpression e0;
    e0.add_term(x0, 1.0);
    e0.add_term(x1, 2.0);
    e0.add_term(x2, 1.0);
    model.add_constraint("c0", std::move(e0), ConstraintSense::LESS_EQUAL, 20.0);

    Presolver presolver;
    auto res = presolver.apply(model);

    KAVACH_ASSERT(res.has_reduced_model());
    KAVACH_ASSERT_EQUAL(res.reduced_model->num_variables(), 2u);

    // Reconstruct solution for (x0=2.0, x2=4.0)
    std::vector<double> reduced_sol = {2.0, 4.0};
    auto orig_sol = res.reconstruction_map.restore_solution(reduced_sol);

    KAVACH_ASSERT_NEAR(orig_sol[0], 2.0, 1e-9);
    KAVACH_ASSERT_NEAR(orig_sol[1], 3.0, 1e-9);  // Restored fixed variable
    KAVACH_ASSERT_NEAR(orig_sol[2], 4.0, 1e-9);

    double orig_val = compute_objective_value(model, orig_sol);
    double red_val = compute_objective_value(*res.reduced_model, reduced_sol);
    KAVACH_ASSERT_NEAR(orig_val, red_val, 1e-9);
    KAVACH_ASSERT_NEAR(compute_max_constraint_violation(model, orig_sol), 0.0, 1e-9);
}

// ============================================================================
// L. Original Model Immutability
// ============================================================================

KAVACH_TEST(test_presolve_original_model_immutability) {
    CanonicalModel model("immutability_test");
    auto x0 = model.add_variable("x0", VariableType::CONTINUOUS, 2.0, 2.0);
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 10.0);

    SparseLinearExpression e;
    e.add_term(x0, 3.0);
    e.add_term(x1, 1.0);
    model.add_constraint("c", std::move(e), ConstraintSense::LESS_EQUAL, 15.0);

    const std::size_t orig_vars = model.num_variables();
    const std::size_t orig_constrs = model.num_constraints();
    const double orig_x0_lb = model.variable(x0).lower_bound;
    const double orig_x0_ub = model.variable(x0).upper_bound;
    const double orig_rhs = model.constraint(0).rhs;

    Presolver presolver;
    auto res = presolver.apply(model);

    // Original model properties must remain identical
    KAVACH_ASSERT_EQUAL(model.num_variables(), orig_vars);
    KAVACH_ASSERT_EQUAL(model.num_constraints(), orig_constrs);
    KAVACH_ASSERT_NEAR(model.variable(x0).lower_bound, orig_x0_lb, 1e-12);
    KAVACH_ASSERT_NEAR(model.variable(x0).upper_bound, orig_x0_ub, 1e-12);
    KAVACH_ASSERT_NEAR(model.constraint(0).rhs, orig_rhs, 1e-12);
}

// ============================================================================
// M. Determinism Test
// ============================================================================

KAVACH_TEST(test_presolve_determinism) {
    CanonicalModel model("det_test");
    auto x0 = model.add_variable("x0", VariableType::CONTINUOUS, 1.0, 1.0);
    auto x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 10.0);
    auto x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, 5.0);

    SparseLinearExpression e0;
    e0.add_term(x0, 2.0);
    e0.add_term(x1, 1.0);
    model.add_constraint("c0", std::move(e0), ConstraintSense::LESS_EQUAL, 12.0);

    SparseLinearExpression e1;
    e1.add_term(x2, 1.0);
    model.add_constraint("c1", std::move(e1), ConstraintSense::LESS_EQUAL, 20.0);  // redundant

    Presolver p1;
    auto res1 = p1.apply(model);

    Presolver p2;
    auto res2 = p2.apply(model);

    KAVACH_ASSERT_EQUAL(res1.stats.reduced_variables, res2.stats.reduced_variables);
    KAVACH_ASSERT_EQUAL(res1.stats.reduced_constraints, res2.stats.reduced_constraints);
    KAVACH_ASSERT_EQUAL(res1.stats.fixed_variables, res2.stats.fixed_variables);
    KAVACH_ASSERT_EQUAL(res1.stats.redundant_rows_removed, res2.stats.redundant_rows_removed);
    KAVACH_ASSERT_EQUAL(res1.stats.passes_performed, res2.stats.passes_performed);
    KAVACH_ASSERT_NEAR(res1.stats.objective_constant_shift, res2.stats.objective_constant_shift,
                       1e-12);
}

// ============================================================================
// Netlib Corpus Validation Test
// ============================================================================

KAVACH_TEST(test_presolve_netlib_corpus) {
    const auto netlib_dir = find_netlib_dir();
    std::vector<std::string> netlib_instances = {
        "afiro.mps",   "adlittle.mps", "blend.mps",  "sc50a.mps", "sc50b.mps",
        "share2b.mps", "beaconfd.mps", "sc105.mps",  "lotfi.mps", "share1b.mps",
        "kb2.mps",     "stocfor1.mps", "boeing2.mps"};

    Presolver presolver;
    io::MpsParser parser;

    std::cout << "\n================ NETLIB PRESOLVE BENCHMARK ================\n";
    std::cout << std::left << std::setw(12) << "Instance" << std::setw(10) << "Orig Vars"
              << std::setw(10) << "Red Vars" << std::setw(10) << "Orig Rows" << std::setw(10)
              << "Red Rows" << std::setw(10) << "Fixed" << std::setw(10) << "Rows Rem"
              << std::setw(10) << "Tighten" << std::setw(12) << "Status" << "\n";
    std::cout << "--------------------------------------------------------------------------------"
                 "--------\n";

    for (const auto& file_name : netlib_instances) {
        auto model = parser.parse_file(netlib_dir / file_name);
        auto res = presolver.apply(*model);

        std::cout << std::left << std::setw(12) << file_name << std::setw(10)
                  << res.stats.original_variables << std::setw(10) << res.stats.reduced_variables
                  << std::setw(10) << res.stats.original_constraints << std::setw(10)
                  << res.stats.reduced_constraints << std::setw(10) << res.stats.fixed_variables
                  << std::setw(10) << res.stats.rows_removed << std::setw(10)
                  << res.stats.bound_tightenings << std::setw(12) << core::to_string(res.status)
                  << "\n";

        // Every model must be valid and non-infeasible
        KAVACH_ASSERT(!res.is_infeasible());
        KAVACH_ASSERT(res.has_reduced_model());
        KAVACH_ASSERT(res.reduced_model->validate().valid);
    }
    std::cout << "===========================================================\n\n";
}
