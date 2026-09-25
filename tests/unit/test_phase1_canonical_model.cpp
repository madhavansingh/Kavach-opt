#include <cmath>
#include <vector>

#include "../test_framework.hpp"
#include "core/error.hpp"
#include "model/model.hpp"

using namespace kavach;
using namespace kavach::model;

// ============================================================================
// A. Variable Creation & Bound Semantics Tests
// ============================================================================

KAVACH_TEST(test_variable_types_and_bounds) {
    CanonicalModel model("var_test_model");

    // Continuous bounded
    VarId v0 = model.add_variable("x0", VariableType::CONTINUOUS, -5.0, 10.0);
    const auto& var0 = model.variable(v0);
    KAVACH_ASSERT_EQUAL(var0.id, 0u);
    KAVACH_ASSERT_EQUAL(var0.name, "x0");
    KAVACH_ASSERT(var0.is_continuous());
    KAVACH_ASSERT_EQUAL(var0.lower_bound, -5.0);
    KAVACH_ASSERT_EQUAL(var0.upper_bound, 10.0);
    KAVACH_ASSERT(var0.has_lower_bound());
    KAVACH_ASSERT(var0.has_upper_bound());
    KAVACH_ASSERT(!var0.is_free());

    // Continuous lower-only bounded
    VarId v1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    const auto& var1 = model.variable(v1);
    KAVACH_ASSERT(var1.has_lower_bound());
    KAVACH_ASSERT(!var1.has_upper_bound());
    KAVACH_ASSERT(!var1.is_free());

    // Continuous upper-only bounded
    VarId v2 = model.add_variable("x2", VariableType::CONTINUOUS, -INFINITY_BOUND, 5.0);
    const auto& var2 = model.variable(v2);
    KAVACH_ASSERT(!var2.has_lower_bound());
    KAVACH_ASSERT(var2.has_upper_bound());
    KAVACH_ASSERT(!var2.is_free());

    // Free variable (-inf, +inf)
    VarId v3 = model.add_variable("x3", VariableType::CONTINUOUS, -INFINITY_BOUND, INFINITY_BOUND);
    const auto& var3 = model.variable(v3);
    KAVACH_ASSERT(var3.is_free());

    // Integer variable
    VarId v4 = model.add_integer_variable("int_y", 0.0, 100.0);
    const auto& var4 = model.variable(v4);
    KAVACH_ASSERT(var4.is_integer());
    KAVACH_ASSERT(!var4.is_continuous());
    KAVACH_ASSERT_EQUAL(var4.lower_bound, 0.0);
    KAVACH_ASSERT_EQUAL(var4.upper_bound, 100.0);

    // Binary variable default bounds [0, 1]
    VarId v5 = model.add_binary_variable("bin_z");
    const auto& var5 = model.variable(v5);
    KAVACH_ASSERT(var5.is_binary());
    KAVACH_ASSERT_EQUAL(var5.lower_bound, 0.0);
    KAVACH_ASSERT_EQUAL(var5.upper_bound, 1.0);

    KAVACH_ASSERT_EQUAL(model.num_variables(), 6u);
    KAVACH_ASSERT_EQUAL(model.num_continuous_variables(), 4u);
    KAVACH_ASSERT_EQUAL(model.num_integer_variables(), 1u);
    KAVACH_ASSERT_EQUAL(model.num_binary_variables(), 1u);

    // Fast lookup
    auto found = model.find_variable_id("bin_z");
    KAVACH_ASSERT(found.has_value());
    KAVACH_ASSERT_EQUAL(found.value(), v5);

    auto not_found = model.find_variable_id("non_existent");
    KAVACH_ASSERT(!not_found.has_value());
}

// ============================================================================
// B. Linear Constraint Creation & Sparse Terms Tests
// ============================================================================

KAVACH_TEST(test_constraint_creation_and_senses) {
    CanonicalModel model("constr_test_model");
    VarId x1 = model.add_variable("x1", VariableType::CONTINUOUS, 0.0, 10.0);
    VarId x2 = model.add_variable("x2", VariableType::CONTINUOUS, 0.0, 20.0);

    // Constraint 1: 2*x1 + 3*x2 <= 25
    SparseLinearExpression expr1;
    expr1.add_term(x1, 2.0);
    expr1.add_term(x2, 3.0);
    ConstrId c0 = model.add_constraint("c_le", std::move(expr1), ConstraintSense::LESS_EQUAL, 25.0);

    // Constraint 2: x1 - x2 >= 5
    SparseLinearExpression expr2;
    expr2.add_term(x1, 1.0);
    expr2.add_term(x2, -1.0);
    ConstrId c1 =
        model.add_constraint("c_ge", std::move(expr2), ConstraintSense::GREATER_EQUAL, 5.0);

    // Constraint 3: 4*x1 == 12
    SparseLinearExpression expr3;
    expr3.add_term(x1, 4.0);
    ConstrId c2 = model.add_constraint("c_eq", std::move(expr3), ConstraintSense::EQUAL, 12.0);

    KAVACH_ASSERT_EQUAL(model.num_constraints(), 3u);
    KAVACH_ASSERT_EQUAL(model.constraint(c0).sense, ConstraintSense::LESS_EQUAL);
    KAVACH_ASSERT_EQUAL(model.constraint(c1).sense, ConstraintSense::GREATER_EQUAL);
    KAVACH_ASSERT_EQUAL(model.constraint(c2).sense, ConstraintSense::EQUAL);

    KAVACH_ASSERT_EQUAL(model.constraint(c0).rhs, 25.0);
    KAVACH_ASSERT_EQUAL(model.constraint(c1).rhs, 5.0);
    KAVACH_ASSERT_EQUAL(model.constraint(c2).rhs, 12.0);

    KAVACH_ASSERT_EQUAL(model.num_nonzeros(), 5u);

    // Fast constraint lookup
    auto found_c = model.find_constraint_id("c_ge");
    KAVACH_ASSERT(found_c.has_value());
    KAVACH_ASSERT_EQUAL(found_c.value(), c1);
}

// ============================================================================
// C. Objective Function Tests (Linear & Quadratic)
// ============================================================================

KAVACH_TEST(test_objective_linear_and_quadratic) {
    CanonicalModel model("qp_model");
    VarId x = model.add_variable("x");
    VarId y = model.add_variable("y");

    model.set_objective_sense(ObjectiveSense::MAXIMIZE);
    model.set_objective_offset(10.5);

    // Linear terms: 3*x + 5*y
    model.set_objective_linear_coefficient(x, 3.0);
    model.set_objective_linear_coefficient(y, 5.0);

    KAVACH_ASSERT(!model.objective().is_quadratic());
    KAVACH_ASSERT(model.is_lp());

    // Quadratic terms: 2*x^2 + 4*x*y + 6*y^2
    model.add_objective_quadratic_term(x, x, 2.0);
    model.add_objective_quadratic_term(y, x, 4.0);  // Unordered (y, x) should normalize to (x, y)
    model.add_objective_quadratic_term(y, y, 6.0);

    model.canonicalize();

    KAVACH_ASSERT(model.objective().is_quadratic());
    KAVACH_ASSERT(model.is_qp());
    KAVACH_ASSERT(!model.is_lp());

    KAVACH_ASSERT_EQUAL(model.objective_sense(), ObjectiveSense::MAXIMIZE);
    KAVACH_ASSERT_EQUAL(model.objective_offset(), 10.5);

    const auto& q_terms = model.objective().quadratic_terms.terms();
    KAVACH_ASSERT_EQUAL(q_terms.size(), 3u);
    KAVACH_ASSERT_EQUAL(q_terms[0].var1_id, x);
    KAVACH_ASSERT_EQUAL(q_terms[0].var2_id, x);
    KAVACH_ASSERT_EQUAL(q_terms[0].coefficient, 2.0);

    KAVACH_ASSERT_EQUAL(q_terms[1].var1_id, x);
    KAVACH_ASSERT_EQUAL(q_terms[1].var2_id, y);
    KAVACH_ASSERT_EQUAL(q_terms[1].coefficient, 4.0);

    KAVACH_ASSERT_EQUAL(q_terms[2].var1_id, y);
    KAVACH_ASSERT_EQUAL(q_terms[2].var2_id, y);
    KAVACH_ASSERT_EQUAL(q_terms[2].coefficient, 6.0);
}

// ============================================================================
// D. Normalization & Canonicalization Tests
// ============================================================================

KAVACH_TEST(test_sparse_expression_normalization) {
    VarId v1 = 1;
    VarId v2 = 2;
    VarId v3 = 3;

    // Linear: 2*x1 + 5*x3 + 3*x1 - 1*x1 + 0*x2
    SparseLinearExpression expr;
    expr.add_term(v1, 2.0);
    expr.add_term(v3, 5.0);
    expr.add_term(v1, 3.0);
    expr.add_term(v1, -1.0);
    expr.add_term(v2, 0.0);  // zero term should be eliminated

    expr.canonicalize();

    // Expect: 4*x1 + 5*x3 (sorted, merged, zero-free)
    KAVACH_ASSERT_EQUAL(expr.num_terms(), 2u);
    KAVACH_ASSERT_EQUAL(expr.terms()[0].var_id, v1);
    KAVACH_ASSERT_EQUAL(expr.terms()[0].coefficient, 4.0);
    KAVACH_ASSERT_EQUAL(expr.terms()[1].var_id, v3);
    KAVACH_ASSERT_EQUAL(expr.terms()[1].coefficient, 5.0);
    KAVACH_ASSERT_EQUAL(expr.get_coefficient(v1), 4.0);
    KAVACH_ASSERT_EQUAL(expr.get_coefficient(v2), 0.0);
    KAVACH_ASSERT_EQUAL(expr.get_coefficient(v3), 5.0);

    // Terms canceling to exact zero
    SparseLinearExpression canceling_expr;
    canceling_expr.add_term(v1, 10.0);
    canceling_expr.add_term(v1, -10.0);
    canceling_expr.canonicalize();
    KAVACH_ASSERT(canceling_expr.empty());
    KAVACH_ASSERT_EQUAL(canceling_expr.num_terms(), 0u);

    // Quadratic normalization
    SparseQuadraticExpression q_expr;
    q_expr.add_term(v2, v1, 3.0);  // (2, 1) -> (1, 2)
    q_expr.add_term(v1, v2, 4.0);  // (1, 2)
    q_expr.add_term(v3, v3, 0.0);  // zero -> eliminated
    q_expr.canonicalize();

    KAVACH_ASSERT_EQUAL(q_expr.num_terms(), 1u);
    KAVACH_ASSERT_EQUAL(q_expr.terms()[0].var1_id, v1);
    KAVACH_ASSERT_EQUAL(q_expr.terms()[0].var2_id, v2);
    KAVACH_ASSERT_EQUAL(q_expr.terms()[0].coefficient, 7.0);
}

// ============================================================================
// E. Validation & Invariant Failure Tests
// ============================================================================

KAVACH_TEST(test_validation_invalid_bounds) {
    CanonicalModel model("invalid_bounds");
    // lower > upper
    model.add_variable("bad_bounds", VariableType::CONTINUOUS, 10.0, 5.0);
    auto res = model.validate();
    KAVACH_ASSERT(!res.valid);
    KAVACH_ASSERT(res.has_errors());

    bool caught = false;
    try {
        model.check_valid();
    } catch (const core::InputException&) {
        caught = true;
    }
    KAVACH_ASSERT(caught);
}

KAVACH_TEST(test_validation_nan_and_infinities) {
    CanonicalModel model("nan_inf_test");
    double nan_val = std::numeric_limits<double>::quiet_NaN();

    // NaN lower bound
    model.add_variable("nan_var", VariableType::CONTINUOUS, nan_val, 10.0);
    auto res1 = model.validate();
    KAVACH_ASSERT(!res1.valid);

    // Positive infinity lower bound
    CanonicalModel model2("pos_inf_lb");
    model2.add_variable("inf_var", VariableType::CONTINUOUS, INFINITY_BOUND, INFINITY_BOUND);
    auto res2 = model2.validate();
    KAVACH_ASSERT(!res2.valid);

    // Infinite RHS
    CanonicalModel model3("inf_rhs");
    VarId v = model3.add_variable("v");
    SparseLinearExpression expr;
    expr.add_term(v, 1.0);
    model3.add_constraint("inf_c", std::move(expr), ConstraintSense::LESS_EQUAL, INFINITY_BOUND);
    auto res3 = model3.validate();
    KAVACH_ASSERT(!res3.valid);
}

KAVACH_TEST(test_validation_binary_bounds) {
    CanonicalModel model("bad_binary");
    // Binary variable with invalid bounds [0, 2]
    model.add_variable("b1", VariableType::BINARY, 0.0, 2.0);
    auto res = model.validate();
    KAVACH_ASSERT(!res.valid);

    CanonicalModel model_neg("bad_binary_neg");
    model_neg.add_variable("b2", VariableType::BINARY, -1.0, 1.0);
    auto res_neg = model_neg.validate();
    KAVACH_ASSERT(!res_neg.valid);
}

KAVACH_TEST(test_duplicate_name_rejection) {
    CanonicalModel model("dup_names");
    model.add_variable("v1", VariableType::CONTINUOUS, 0.0, 1.0);

    bool caught_var = false;
    try {
        model.add_variable("v1", VariableType::CONTINUOUS, 0.0, 1.0);
    } catch (const core::InputException&) {
        caught_var = true;
    }
    KAVACH_ASSERT(caught_var);

    SparseLinearExpression expr;
    expr.add_term(0, 1.0);
    model.add_constraint("c1", expr, ConstraintSense::EQUAL, 1.0);

    bool caught_constr = false;
    try {
        model.add_constraint("c1", expr, ConstraintSense::EQUAL, 2.0);
    } catch (const core::InputException&) {
        caught_constr = true;
    }
    KAVACH_ASSERT(caught_constr);
}

// ============================================================================
// F. Scale-Oriented Construction Test
// ============================================================================

KAVACH_TEST(test_scale_oriented_sparse_construction) {
    CanonicalModel model("large_sparse_model");

    constexpr std::size_t NUM_VARS = 2000;
    constexpr std::size_t NUM_CONSTRS = 2000;

    for (std::size_t i = 0; i < NUM_VARS; ++i) {
        model.add_variable("x_" + std::to_string(i), VariableType::CONTINUOUS, 0.0, 100.0);
    }

    // Build sparse banded structure (3 nonzeros per constraint)
    for (std::size_t i = 0; i < NUM_CONSTRS; ++i) {
        SparseLinearExpression expr;
        auto v1 = static_cast<VarId>(i);
        auto v2 = static_cast<VarId>((i + 1) % NUM_VARS);
        auto v3 = static_cast<VarId>((i + 5) % NUM_VARS);

        expr.add_term(v1, 2.0);
        expr.add_term(v2, -1.0);
        expr.add_term(v3, 0.5);

        model.add_constraint("c_" + std::to_string(i), std::move(expr), ConstraintSense::LESS_EQUAL,
                             50.0);
    }

    KAVACH_ASSERT_EQUAL(model.num_variables(), NUM_VARS);
    KAVACH_ASSERT_EQUAL(model.num_constraints(), NUM_CONSTRS);
    KAVACH_ASSERT_EQUAL(model.num_nonzeros(), NUM_CONSTRS * 3);

    // Verify validation passes cleanly and fast
    auto val_res = model.validate();
    KAVACH_ASSERT(val_res.valid);
}

// ============================================================================
// G. Standard Model Fixtures (LP, MILP, QP)
// ============================================================================

KAVACH_TEST(fixture_simple_lp_product_mix) {
    // Problem: Maximize profit = 40*x1 + 30*x2
    // Subject to:
    //   Labor:    x1 +   x2 <= 16
    //   Material: 2*x1 + x2 <= 20
    //   x1, x2 >= 0
    CanonicalModel model("product_mix_lp");
    VarId x1 = model.add_variable("Chairs", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);
    VarId x2 = model.add_variable("Tables", VariableType::CONTINUOUS, 0.0, INFINITY_BOUND);

    model.set_objective_sense(ObjectiveSense::MAXIMIZE);
    model.set_objective_linear_coefficient(x1, 40.0);
    model.set_objective_linear_coefficient(x2, 30.0);

    SparseLinearExpression labor_expr;
    labor_expr.add_term(x1, 1.0);
    labor_expr.add_term(x2, 1.0);
    model.add_constraint("LaborLimit", std::move(labor_expr), ConstraintSense::LESS_EQUAL, 16.0);

    SparseLinearExpression mat_expr;
    mat_expr.add_term(x1, 2.0);
    mat_expr.add_term(x2, 1.0);
    model.add_constraint("MaterialLimit", std::move(mat_expr), ConstraintSense::LESS_EQUAL, 20.0);

    KAVACH_ASSERT(model.is_lp());
    KAVACH_ASSERT(!model.is_milp());
    KAVACH_ASSERT(!model.is_qp());
    model.check_valid();
}

KAVACH_TEST(fixture_small_milp_facility_blending) {
    // Blending with binary operational switch:
    // Maximize revenue = 15*blend1 + 20*blend2 - 100*facility_on
    // Subject to:
    //   blend1 + blend2 - 50*facility_on <= 0 (facility throughput limit)
    //   blend1 <= 30
    //   facility_on in {0, 1}
    CanonicalModel model("refinery_blending_milp");
    VarId b1 = model.add_variable("BlendStream1", VariableType::CONTINUOUS, 0.0, 30.0);
    VarId b2 = model.add_variable("BlendStream2", VariableType::CONTINUOUS, 0.0, 50.0);
    VarId unit = model.add_binary_variable("CrackingUnitOn");

    model.set_objective_sense(ObjectiveSense::MAXIMIZE);
    model.set_objective_linear_coefficient(b1, 15.0);
    model.set_objective_linear_coefficient(b2, 20.0);
    model.set_objective_linear_coefficient(unit, -100.0);

    SparseLinearExpression cap_expr;
    cap_expr.add_term(b1, 1.0);
    cap_expr.add_term(b2, 1.0);
    cap_expr.add_term(unit, -50.0);
    model.add_constraint("CapacitySwitch", std::move(cap_expr), ConstraintSense::LESS_EQUAL, 0.0);

    KAVACH_ASSERT(model.is_milp());
    KAVACH_ASSERT(!model.is_lp());
    KAVACH_ASSERT(!model.is_qp());
    KAVACH_ASSERT_EQUAL(model.num_binary_variables(), 1u);
    model.check_valid();
}

KAVACH_TEST(fixture_small_convex_qp_portfolio) {
    // Portfolio optimization:
    // Minimize risk = 1/2 * (2*w1^2 + 2*w2^2 + 1.2*w1*w2)
    // Subject to:
    //   Budget: w1 + w2 == 1.0
    //   Target return: 0.10*w1 + 0.15*w2 >= 0.12
    //   w1, w2 >= 0
    CanonicalModel model("markowitz_qp");
    VarId w1 = model.add_variable("WeightAsset1", VariableType::CONTINUOUS, 0.0, 1.0);
    VarId w2 = model.add_variable("WeightAsset2", VariableType::CONTINUOUS, 0.0, 1.0);

    model.set_objective_sense(ObjectiveSense::MINIMIZE);

    // Q matrix terms: 0.5 * [w1, w2] * [2, 0.6; 0.6, 2] * [w1; w2] = w1^2 + 0.6*w1*w2 + w2^2
    model.add_objective_quadratic_term(w1, w1, 1.0);
    model.add_objective_quadratic_term(w1, w2, 0.6);
    model.add_objective_quadratic_term(w2, w2, 1.0);

    SparseLinearExpression budget_expr;
    budget_expr.add_term(w1, 1.0);
    budget_expr.add_term(w2, 1.0);
    model.add_constraint("FullInvestment", std::move(budget_expr), ConstraintSense::EQUAL, 1.0);

    SparseLinearExpression return_expr;
    return_expr.add_term(w1, 0.10);
    return_expr.add_term(w2, 0.15);
    model.add_constraint("MinReturn", std::move(return_expr), ConstraintSense::GREATER_EQUAL, 0.12);

    KAVACH_ASSERT(model.is_qp());
    KAVACH_ASSERT(!model.is_lp());
    KAVACH_ASSERT(!model.is_milp());
    model.check_valid();
}
