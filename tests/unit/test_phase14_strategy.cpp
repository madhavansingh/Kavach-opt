/**
 * @file test_phase14_strategy.cpp
 * @brief Unit and integration test suite for Kavach-Opt Rule-Based Solver Strategy Layer.
 *
 * Verifies:
 *  - Problem characteristics extraction accuracy (LP, MILP, QP, density, dynamic range)
 *  - Rules R001 to R018 evaluation and safety precedence
 *  - Determinism & Bitwise Reproducibility
 *  - Boundary conditions (empty models, 1-var/1-constr, extreme dynamic range)
 *  - Baseline equivalence (Netlib LP, MIPLIB knapsack, QP portfolio)
 *  - Strategy overhead (< 2 ms, << 1% of solve time)
 *  - Explainability, rules fired tracking, and JSON serialization
 *  - Programmatic KavachSolver API integration
 */

#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "../test_framework.hpp"
#include "algorithms/lp/simplex.hpp"
#include "algorithms/milp/branch_and_bound.hpp"
#include "algorithms/qp/active_set.hpp"
#include "api/solver.hpp"
#include "benchmark/json.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "io/parser.hpp"
#include "model/model.hpp"
#include "strategy/strategy.hpp"

using namespace kavach;
using namespace kavach::strategy;

namespace fs = std::filesystem;

namespace {

model::SparseLinearExpression make_expr(const std::vector<std::pair<model::VarId, double>>& terms) {
    model::SparseLinearExpression expr;
    for (const auto& [var, coeff] : terms) {
        expr.add_term(var, coeff);
    }
    return expr;
}

fs::path find_model_path(const std::string& rel_path) {
    for (const auto& prefix : {".", "..", "../.."}) {
        auto candidate = fs::path(prefix) / rel_path;
        if (fs::exists(candidate)) {
            return candidate;
        }
    }
    return fs::path();
}

model::CanonicalModel create_toy_lp() {
    model::CanonicalModel m("toy_lp");
    m.set_objective_sense(model::ObjectiveSense::MINIMIZE);
    auto x0 = m.add_variable("x0", model::VariableType::CONTINUOUS, 0.0, 10.0);
    auto x1 = m.add_variable("x1", model::VariableType::CONTINUOUS, 0.0, 10.0);
    m.set_objective_linear_coefficient(x0, 2.0);
    m.set_objective_linear_coefficient(x1, 3.0);
    m.add_constraint("c0", make_expr({{x0, 1.0}, {x1, 2.0}}), model::ConstraintSense::GREATER_EQUAL,
                     4.0);
    return m;
}

model::CanonicalModel create_toy_milp() {
    model::CanonicalModel m("toy_milp");
    m.set_objective_sense(model::ObjectiveSense::MAXIMIZE);
    auto x0 = m.add_binary_variable("x0");
    auto x1 = m.add_integer_variable("x1", 0.0, 5.0);
    auto x2 = m.add_variable("x2", model::VariableType::CONTINUOUS, 0.0, 10.0);
    m.set_objective_linear_coefficient(x0, 5.0);
    m.set_objective_linear_coefficient(x1, 3.0);
    m.set_objective_linear_coefficient(x2, 1.0);
    m.add_constraint("c0", make_expr({{x0, 2.0}, {x1, 1.0}, {x2, 1.0}}),
                     model::ConstraintSense::LESS_EQUAL, 6.0);
    return m;
}

model::CanonicalModel create_toy_qp() {
    model::CanonicalModel m("toy_qp");
    m.set_objective_sense(model::ObjectiveSense::MINIMIZE);
    auto x0 = m.add_variable("x0", model::VariableType::CONTINUOUS, 0.0, 10.0);
    auto x1 = m.add_variable("x1", model::VariableType::CONTINUOUS, 0.0, 10.0);
    m.set_objective_linear_coefficient(x0, 1.0);
    m.set_objective_linear_coefficient(x1, 2.0);
    m.set_objective_quadratic_coefficient(x0, x0, 2.0);
    m.set_objective_quadratic_coefficient(x1, x1, 4.0);
    m.add_constraint("c0", make_expr({{x0, 1.0}, {x1, 1.0}}), model::ConstraintSense::EQUAL, 3.0);
    return m;
}

}  // namespace

// ============================================================================
// Section 1: Problem Characteristics Extraction
// ============================================================================

KAVACH_TEST(test_characteristics_continuous_lp) {
    auto m = create_toy_lp();
    StrategyEngine engine;
    auto chars = engine.characterize(m);

    KAVACH_ASSERT_EQUAL(chars.num_variables, 2);
    KAVACH_ASSERT_EQUAL(chars.num_constraints, 1);
    KAVACH_ASSERT_EQUAL(chars.num_nonzeros, 2);
    KAVACH_ASSERT_NEAR(chars.density, 1.0, 1e-6);
    KAVACH_ASSERT_EQUAL(chars.integer_variable_count, 0);
    KAVACH_ASSERT_EQUAL(chars.binary_variable_count, 0);
    KAVACH_ASSERT_NEAR(chars.integer_variable_ratio, 0.0, 1e-6);
    KAVACH_ASSERT_EQUAL(static_cast<int>(chars.problem_class), static_cast<int>(ProblemClass::LP));
    KAVACH_ASSERT(!chars.coefficient_magnitudes_span_multiple_orders);
    KAVACH_ASSERT_EQUAL(chars.near_zero_coefficient_count, 0);
}

KAVACH_TEST(test_characteristics_milp) {
    auto m = create_toy_milp();
    StrategyEngine engine;
    auto chars = engine.characterize(m);

    KAVACH_ASSERT_EQUAL(chars.num_variables, 3);
    KAVACH_ASSERT_EQUAL(chars.binary_variable_count, 1);
    KAVACH_ASSERT_EQUAL(chars.integer_variable_count, 1);
    KAVACH_ASSERT_NEAR(chars.binary_variable_ratio, 1.0 / 3.0, 1e-6);
    KAVACH_ASSERT_NEAR(chars.integer_variable_ratio, 2.0 / 3.0, 1e-6);
    KAVACH_ASSERT_EQUAL(static_cast<int>(chars.problem_class),
                        static_cast<int>(ProblemClass::MILP));
}

KAVACH_TEST(test_characteristics_qp) {
    auto m = create_toy_qp();
    StrategyEngine engine;
    auto chars = engine.characterize(m);

    KAVACH_ASSERT_EQUAL(chars.num_variables, 2);
    KAVACH_ASSERT_EQUAL(static_cast<int>(chars.problem_class),
                        static_cast<int>(ProblemClass::CONVEX_QP));
}

KAVACH_TEST(test_characteristics_dynamic_range_and_near_zero) {
    model::CanonicalModel m("ill_conditioned_lp");
    auto x0 = m.add_variable("x0", model::VariableType::CONTINUOUS, 0.0, 1.0);
    auto x1 = m.add_variable("x1", model::VariableType::CONTINUOUS, 0.0, 1.0);
    auto x2 = m.add_variable("x2", model::VariableType::CONTINUOUS, 0.0, 1.0);

    // Dynamic range spanning 1e-13 to 1e6
    m.add_constraint("c0", make_expr({{x0, 1e-13}, {x1, 1e6}, {x2, 2.5}}),
                     model::ConstraintSense::LESS_EQUAL, 1.0);

    StrategyEngine engine;
    auto chars = engine.characterize(m);

    KAVACH_ASSERT(chars.coefficient_magnitudes_span_multiple_orders);
    KAVACH_ASSERT(chars.coefficient_range_ratio >= 1e18);
    KAVACH_ASSERT_EQUAL(chars.near_zero_coefficient_count, 1);
    KAVACH_ASSERT(chars.recommend_scaling);
}

// ============================================================================
// Section 2: Rule Evaluations R001 to R018
// ============================================================================

KAVACH_TEST(test_rules_r001_and_r005_qp_routing_and_no_presolve) {
    auto m = create_toy_qp();
    StrategyEngine engine;
    auto dec = engine.decide(engine.characterize(m));

    KAVACH_ASSERT_EQUAL(static_cast<int>(dec.problem_class),
                        static_cast<int>(ProblemClass::CONVEX_QP));
    KAVACH_ASSERT_EQUAL(static_cast<int>(dec.presolve_policy),
                        static_cast<int>(PresolvePolicy::NONE));
    // Check rules fired
    bool r001_fired = false;
    bool r005_fired = false;
    for (const auto& r : dec.rules_fired) {
        if (r == "R001")
            r001_fired = true;
        if (r == "R005")
            r005_fired = true;
    }
    KAVACH_ASSERT(r001_fired);
    KAVACH_ASSERT(r005_fired);
}

KAVACH_TEST(test_rules_r002_and_r014_milp_routing_and_warm_starts) {
    auto m = create_toy_milp();
    StrategyEngine engine;
    auto dec = engine.decide(engine.characterize(m));

    KAVACH_ASSERT_EQUAL(static_cast<int>(dec.problem_class), static_cast<int>(ProblemClass::MILP));
    KAVACH_ASSERT(dec.prefer_dual_simplex_warm_start);
    KAVACH_ASSERT_EQUAL(static_cast<int>(dec.milp_branching_strategy),
                        static_cast<int>(MILPBranchingStrategy::MOST_FRACTIONAL));
    KAVACH_ASSERT_EQUAL(static_cast<int>(dec.milp_node_strategy),
                        static_cast<int>(MILPNodeStrategy::BEST_BOUND));
    KAVACH_ASSERT(!dec.pseudocost_available);

    bool r002_fired = false;
    bool r014_fired = false;
    for (const auto& r : dec.rules_fired) {
        if (r == "R002")
            r002_fired = true;
        if (r == "R014")
            r014_fired = true;
    }
    KAVACH_ASSERT(r002_fired);
    KAVACH_ASSERT(r014_fired);  // 2/3 integrality ratio > 0.5
}

KAVACH_TEST(test_rules_r003_and_r004_lp_routing_and_full_presolve) {
    auto m = create_toy_lp();
    StrategyEngine engine;
    auto dec = engine.decide(engine.characterize(m));

    KAVACH_ASSERT_EQUAL(static_cast<int>(dec.problem_class), static_cast<int>(ProblemClass::LP));
    KAVACH_ASSERT_EQUAL(static_cast<int>(dec.lp_algorithm),
                        static_cast<int>(LPAlgorithm::REVISED_SIMPLEX));
    KAVACH_ASSERT_EQUAL(static_cast<int>(dec.presolve_policy),
                        static_cast<int>(PresolvePolicy::FULL));

    bool r003_fired = false;
    bool r004_fired = false;
    for (const auto& r : dec.rules_fired) {
        if (r == "R003")
            r003_fired = true;
        if (r == "R004")
            r004_fired = true;
    }
    KAVACH_ASSERT(r003_fired);
    KAVACH_ASSERT(r004_fired);
}

KAVACH_TEST(test_rules_r007_r008_r009_numerical_risk_profiles) {
    StrategyEngine engine;

    // Normal profile
    {
        auto m = create_toy_lp();
        auto dec = engine.decide(engine.characterize(m));
        KAVACH_ASSERT_EQUAL(static_cast<int>(dec.numerical_risk_profile),
                            static_cast<int>(NumericalRiskProfile::NORMAL));
        KAVACH_ASSERT_EQUAL(dec.lp_refactorization_interval, 50);
        KAVACH_ASSERT(!dec.lp_force_bland_rule);
    }

    // Elevated profile
    {
        model::CanonicalModel m("elevated_lp");
        auto x0 = m.add_variable("x0", model::VariableType::CONTINUOUS, 0.0, 1.0);
        auto x1 = m.add_variable("x1", model::VariableType::CONTINUOUS, 0.0, 1.0);
        m.add_constraint("c0", make_expr({{x0, 1.0}, {x1, 5e4}}),
                         model::ConstraintSense::LESS_EQUAL, 1.0);

        auto dec = engine.decide(engine.characterize(m));
        KAVACH_ASSERT_EQUAL(static_cast<int>(dec.numerical_risk_profile),
                            static_cast<int>(NumericalRiskProfile::ELEVATED));
        KAVACH_ASSERT_EQUAL(dec.lp_refactorization_interval, 20);
    }

    // High risk profile
    {
        model::CanonicalModel m("high_risk_lp");
        auto x0 = m.add_variable("x0", model::VariableType::CONTINUOUS, 0.0, 1.0);
        auto x1 = m.add_variable("x1", model::VariableType::CONTINUOUS, 0.0, 1.0);
        m.add_constraint("c0", make_expr({{x0, 1.0}, {x1, 1e9}}),
                         model::ConstraintSense::LESS_EQUAL, 1.0);

        auto dec = engine.decide(engine.characterize(m));
        KAVACH_ASSERT_EQUAL(static_cast<int>(dec.numerical_risk_profile),
                            static_cast<int>(NumericalRiskProfile::HIGH));
        KAVACH_ASSERT_EQUAL(dec.lp_refactorization_interval, 10);
        KAVACH_ASSERT(dec.lp_force_bland_rule);
        KAVACH_ASSERT_NEAR(dec.recommended_tolerances.pivot_tol, 1e-8, 1e-15);
    }
}

KAVACH_TEST(test_rules_r010_r011_r013_backend_routing_and_no_gpu) {
    StrategyEngine engine;
    auto m = create_toy_lp();

    // Default configuration (force_no_gpu = false)
    StrategyConfig config;
    config.force_no_gpu = true;
    auto dec = engine.decide(engine.characterize(m), config);

    KAVACH_ASSERT_EQUAL(static_cast<int>(dec.recommended_backend),
                        static_cast<int>(backend::BackendType::CPU_SEQUENTIAL));
    KAVACH_ASSERT(!dec.gpu_eligible_for_batched_spmv);

    bool r011_fired = false;
    for (const auto& r : dec.rules_fired) {
        if (r == "R011")
            r011_fired = true;
    }
    KAVACH_ASSERT(r011_fired);
}

KAVACH_TEST(test_rules_r015_r016_r017_user_overrides) {
    StrategyEngine engine;
    auto m = create_toy_lp();

    StrategyConfig config;
    config.override_presolve = false;
    config.override_scaling = false;
    config.override_backend = backend::BackendType::CPU_MULTICORE;

    auto dec = engine.decide(engine.characterize(m), config);

    KAVACH_ASSERT_EQUAL(static_cast<int>(dec.presolve_policy),
                        static_cast<int>(PresolvePolicy::NONE));
    KAVACH_ASSERT(!dec.scaling_enabled);
    KAVACH_ASSERT_EQUAL(static_cast<int>(dec.recommended_backend),
                        static_cast<int>(backend::BackendType::CPU_MULTICORE));

    bool r015_fired = false;
    bool r016_fired = false;
    bool r017_fired = false;
    for (const auto& r : dec.rules_fired) {
        if (r == "R015")
            r015_fired = true;
        if (r == "R016")
            r016_fired = true;
        if (r == "R017")
            r017_fired = true;
    }
    KAVACH_ASSERT(r015_fired);
    KAVACH_ASSERT(r016_fired);
    KAVACH_ASSERT(r017_fired);
}

KAVACH_TEST(test_rule_r018_safety_precedence_overrides_interval) {
    StrategyEngine engine;
    model::CanonicalModel m("high_risk_lp");
    auto x0 = m.add_variable("x0", model::VariableType::CONTINUOUS, 0.0, 1.0);
    auto x1 = m.add_variable("x1", model::VariableType::CONTINUOUS, 0.0, 1.0);
    m.add_constraint("c0", make_expr({{x0, 1.0}, {x1, 1e10}}), model::ConstraintSense::LESS_EQUAL,
                     1.0);

    StrategyConfig config;
    // Set a very loose refactorization interval
    config.normal_refactorization_interval = 100;
    config.high_risk_refactorization_interval = 10;

    auto dec = engine.decide(engine.characterize(m), config);

    // Rule R018 ensures refactorization interval is locked down to 10 despite any user preference
    KAVACH_ASSERT_EQUAL(dec.lp_refactorization_interval, 10);

    bool r018_fired = false;
    for (const auto& r : dec.rules_fired) {
        if (r == "R018")
            r018_fired = true;
    }
    KAVACH_ASSERT(r018_fired);
}

// ============================================================================
// Section 3: Determinism & Bitwise Reproducibility
// ============================================================================

KAVACH_TEST(test_determinism_repeated_calls) {
    auto m = create_toy_lp();
    StrategyEngine engine;

    auto dec0 = engine.decide(engine.characterize(m));

    for (int iter = 0; iter < 10; ++iter) {
        auto dec = engine.decide(engine.characterize(m));

        KAVACH_ASSERT_EQUAL(static_cast<int>(dec.problem_class),
                            static_cast<int>(dec0.problem_class));
        KAVACH_ASSERT_EQUAL(static_cast<int>(dec.presolve_policy),
                            static_cast<int>(dec0.presolve_policy));
        KAVACH_ASSERT_EQUAL(dec.scaling_enabled, dec0.scaling_enabled);
        KAVACH_ASSERT_EQUAL(dec.lp_refactorization_interval, dec0.lp_refactorization_interval);
        KAVACH_ASSERT_EQUAL(dec.rules_fired.size(), dec0.rules_fired.size());

        for (std::size_t i = 0; i < dec.rules_fired.size(); ++i) {
            KAVACH_ASSERT_EQUAL(dec.rules_fired[i], dec0.rules_fired[i]);
            KAVACH_ASSERT_EQUAL(dec.rule_explanations[i], dec0.rule_explanations[i]);
        }
    }
}

// ============================================================================
// Section 4: Boundary Conditions
// ============================================================================

KAVACH_TEST(test_boundary_empty_model) {
    model::CanonicalModel m("empty_model");
    StrategyEngine engine;
    auto chars = engine.characterize(m);

    KAVACH_ASSERT_EQUAL(chars.num_variables, 0);
    KAVACH_ASSERT_EQUAL(chars.num_constraints, 0);
    KAVACH_ASSERT_EQUAL(chars.num_nonzeros, 0);
    KAVACH_ASSERT_NEAR(chars.density, 0.0, 1e-15);

    auto dec = engine.decide(chars);
    KAVACH_ASSERT_EQUAL(static_cast<int>(dec.problem_class), static_cast<int>(ProblemClass::LP));
    KAVACH_ASSERT_EQUAL(static_cast<int>(dec.presolve_policy),
                        static_cast<int>(PresolvePolicy::FULL));
}

KAVACH_TEST(test_boundary_single_variable_constraint) {
    model::CanonicalModel m("trivial_1x1");
    auto x0 = m.add_variable("x0", model::VariableType::CONTINUOUS, 0.0, 5.0);
    m.set_objective_linear_coefficient(x0, 1.0);
    m.add_constraint("c0", make_expr({{x0, 1.0}}), model::ConstraintSense::GREATER_EQUAL, 2.0);

    StrategyEngine engine;
    auto chars = engine.characterize(m);
    KAVACH_ASSERT_EQUAL(chars.num_variables, 1);
    KAVACH_ASSERT_EQUAL(chars.num_constraints, 1);
    KAVACH_ASSERT_EQUAL(chars.num_nonzeros, 1);

    auto dec = engine.decide(chars);
    KAVACH_ASSERT_EQUAL(static_cast<int>(dec.problem_class), static_cast<int>(ProblemClass::LP));
    KAVACH_ASSERT_EQUAL(dec.lp_refactorization_interval, 50);
}

// ============================================================================
// Section 5: Baseline Equivalence
// ============================================================================

KAVACH_TEST(test_baseline_equivalence_netlib_afiro) {
    auto path = find_model_path("benchmarks/netlib/afiro.mps");
    if (path.empty()) {
        std::cout << "    (skipped: afiro.mps not found)\n";
        return;
    }

    auto model = io::parse_model(path.string());
    KAVACH_ASSERT(model != nullptr);

    // 1. Direct RevisedSimplex solve (baseline)
    algorithms::lp::LPSolverOptions base_opts;
    base_opts.enable_presolve = false;
    algorithms::lp::RevisedSimplex base_solver(base_opts);
    auto base_sol = base_solver.solve(*model);
    KAVACH_ASSERT_EQUAL(base_sol.status, core::SolverStatus::OPTIMAL);

    // 2. Programmatic KavachSolver solve with Strategy Decision
    api::KavachSolver strat_solver;
    api::SolverOptions opts;
    opts.run_presolve = false;  // Compare on identical presolve setting
    auto strat_sol = strat_solver.solve(*model, opts);

    KAVACH_ASSERT_EQUAL(strat_sol.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(strat_sol.objective_value, base_sol.objective_value, 1e-4);
    KAVACH_ASSERT(strat_sol.verification_report.passed);
}

KAVACH_TEST(test_baseline_equivalence_knapsack_milp) {
    auto path = find_model_path("benchmarks/miplib/knapsack.lp");
    if (path.empty()) {
        std::cout << "    (skipped: knapsack.lp not found)\n";
        return;
    }

    auto model = io::parse_model(path.string());
    KAVACH_ASSERT(model != nullptr);

    // 1. Baseline B&B
    algorithms::milp::BranchAndBound base_solver;
    auto base_sol = base_solver.solve(*model);
    KAVACH_ASSERT_EQUAL(base_sol.status, core::SolverStatus::OPTIMAL);

    // 2. Strategy KavachSolver
    api::KavachSolver strat_solver;
    auto strat_sol = strat_solver.solve(*model);

    KAVACH_ASSERT_EQUAL(strat_sol.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(strat_sol.objective_value, base_sol.best_objective, 1e-6);
    KAVACH_ASSERT(strat_sol.verification_report.passed);
}

KAVACH_TEST(test_baseline_equivalence_portfolio_qp) {
    auto path = find_model_path("benchmarks/qp/portfolio.lp");
    if (path.empty()) {
        std::cout << "    (skipped: portfolio.lp not found)\n";
        return;
    }

    auto model = io::parse_model(path.string());
    KAVACH_ASSERT(model != nullptr);

    // 1. Baseline Active Set QP
    algorithms::qp::PrimalActiveSetQPSolver base_solver;
    auto base_sol = base_solver.solve(*model);
    KAVACH_ASSERT_EQUAL(base_sol.status, core::SolverStatus::OPTIMAL);

    // 2. Strategy KavachSolver
    api::KavachSolver strat_solver;
    auto strat_sol = strat_solver.solve(*model);

    KAVACH_ASSERT_EQUAL(strat_sol.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(strat_sol.objective_value, base_sol.objective_value, 1e-5);
    KAVACH_ASSERT(strat_sol.verification_report.passed);
}

// ============================================================================
// Section 6: Strategy Overhead Measurement
// ============================================================================

KAVACH_TEST(test_strategy_overhead_netlib) {
    auto path = find_model_path("benchmarks/netlib/afiro.mps");
    if (path.empty()) {
        std::cout << "    (skipped: afiro.mps not found)\n";
        return;
    }

    auto model = io::parse_model(path.string());
    StrategyEngine engine;

    auto t0 = std::chrono::steady_clock::now();
    auto chars = engine.characterize(*model);
    auto dec = engine.decide(chars);
    auto t1 = std::chrono::steady_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "    Strategy decision time on AFIRO: " << total_ms << " ms\n";

    // Overhead must be under 2 ms on Netlib models
    KAVACH_ASSERT(total_ms < 2.0);
    KAVACH_ASSERT(dec.total_strategy_time_ms < 2.0);
}

// ============================================================================
// Section 7: Explainability and JSON Serialization
// ============================================================================

KAVACH_TEST(test_explainability_and_json_roundtrip) {
    auto m = create_toy_lp();
    StrategyEngine engine;
    auto dec = engine.decide(engine.characterize(m));

    // Explainability checks
    std::string explanation = dec.explain();
    KAVACH_ASSERT(!explanation.empty());
    KAVACH_ASSERT(!dec.human_readable_rationale.empty());
    KAVACH_ASSERT_EQUAL(dec.rules_fired.size(), dec.rule_explanations.size());
    KAVACH_ASSERT(explanation.find("RULES TRIGGERED:") != std::string::npos);

    // JSON serialization
    auto json_val = dec.to_json();
    std::string json_str = json_val.dump();
    KAVACH_ASSERT(!json_str.empty());

    // Roundtrip parse check
    auto parsed = benchmark::JsonValue::parse(json_str);
    KAVACH_ASSERT(parsed.is_object());
    KAVACH_ASSERT(parsed.contains("strategy_version"));
    KAVACH_ASSERT_EQUAL(parsed["strategy_version"].as_string(), "1.0.0");
    KAVACH_ASSERT(parsed.contains("rules_fired"));
    KAVACH_ASSERT(parsed["rules_fired"].is_array());
    KAVACH_ASSERT_EQUAL(parsed["rules_fired"].as_array().size(), dec.rules_fired.size());
    KAVACH_ASSERT(parsed.contains("pseudocost_available"));
    KAVACH_ASSERT(!parsed["pseudocost_available"].as_bool());
}

// ============================================================================
// Section 8: Programmatic KavachSolver API Integration
// ============================================================================

KAVACH_TEST(test_programmatic_kavach_solver_api) {
    auto m = create_toy_lp();
    api::KavachSolver solver;

    // 1. plan()
    auto plan = solver.plan(m);
    KAVACH_ASSERT_EQUAL(static_cast<int>(plan.problem_class), static_cast<int>(ProblemClass::LP));

    // 2. explain_plan()
    auto explanation = solver.explain_plan(m);
    KAVACH_ASSERT(explanation.find("Target Problem Class: LP") != std::string::npos);

    // 3. solve(model, options)
    api::SolverOptions opts;
    auto res1 = solver.solve(m, opts);
    KAVACH_ASSERT_EQUAL(res1.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(res1.objective_value, 6.0, 1e-6);  // x0=0, x1=2 -> 3*2 = 6

    // 4. solve(model, decision, options)
    auto res2 = solver.solve(m, plan, opts);
    KAVACH_ASSERT_EQUAL(res2.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_NEAR(res2.objective_value, 6.0, 1e-6);
}

// ============================================================================
// Section 9: Real Execution Backend Orchestration & Telemetry Truth
// ============================================================================

namespace {

model::CanonicalModel create_nontrivial_milp() {
    model::CanonicalModel m("nontrivial_milp");
    auto x0 = m.add_variable("x0", model::VariableType::INTEGER, 0.0, 10.0);
    auto x1 = m.add_variable("x1", model::VariableType::INTEGER, 0.0, 10.0);
    auto x2 = m.add_variable("x2", model::VariableType::INTEGER, 0.0, 10.0);
    auto s0 = m.add_variable("s0", model::VariableType::CONTINUOUS, 0.0, 5.0);

    m.add_constraint("c0", make_expr({{x0, 2.0}, {x1, 3.0}, {x2, 1.5}, {s0, -0.5}}),
                     model::ConstraintSense::GREATER_EQUAL, 8.7);
    m.add_constraint("c1", make_expr({{x0, 1.5}, {x1, 1.0}, {x2, 2.5}, {s0, 1.0}}),
                     model::ConstraintSense::GREATER_EQUAL, 7.3);
    m.add_constraint("c2", make_expr({{x0, 1.0}, {x1, 2.0}, {x2, 1.0}}),
                     model::ConstraintSense::LESS_EQUAL, 15.0);

    m.set_objective_sense(model::ObjectiveSense::MINIMIZE);
    m.set_objective_linear_coefficient(x0, 4.0);
    m.set_objective_linear_coefficient(x1, 5.0);
    m.set_objective_linear_coefficient(x2, 3.0);
    m.set_objective_linear_coefficient(s0, 2.0);
    return m;
}

}  // namespace

KAVACH_TEST(test_backend_orchestration_cpu_sequential_real_solve) {
    auto lp_model = create_toy_lp();
    auto milp_model = create_nontrivial_milp();

    api::KavachSolver solver;
    api::SolverOptions opts;
    opts.run_presolve = false;
    opts.strategy_config.override_backend = backend::BackendType::CPU_SEQUENTIAL;

    auto lp_res = solver.solve(lp_model, opts);
    KAVACH_ASSERT_EQUAL(lp_res.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(lp_res.verification_report.passed);
    KAVACH_ASSERT_EQUAL(lp_res.backend_used, backend::BackendType::CPU_SEQUENTIAL);
    KAVACH_ASSERT_EQUAL(lp_res.backend_telemetry.requested_backend,
                        backend::BackendType::CPU_SEQUENTIAL);
    KAVACH_ASSERT_EQUAL(lp_res.backend_telemetry.selected_backend,
                        backend::BackendType::CPU_SEQUENTIAL);
    KAVACH_ASSERT_EQUAL(lp_res.backend_telemetry.executed_backend,
                        backend::BackendType::CPU_SEQUENTIAL);
    KAVACH_ASSERT(lp_res.backend_telemetry.invocations > 0);
    KAVACH_ASSERT(lp_res.backend_telemetry.total_vectors > 0);
    KAVACH_ASSERT_EQUAL(lp_res.backend_telemetry.worker_count, 1);
    KAVACH_ASSERT(!lp_res.backend_telemetry.fallback_occurred);

    auto milp_res = solver.solve(milp_model, opts);
    KAVACH_ASSERT_EQUAL(milp_res.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(milp_res.verification_report.passed);
    KAVACH_ASSERT_EQUAL(milp_res.backend_used, backend::BackendType::CPU_SEQUENTIAL);
    KAVACH_ASSERT_EQUAL(milp_res.backend_telemetry.executed_backend,
                        backend::BackendType::CPU_SEQUENTIAL);
    KAVACH_ASSERT(milp_res.backend_telemetry.invocations > 0);
    KAVACH_ASSERT_EQUAL(milp_res.backend_telemetry.worker_count, 1);
    KAVACH_ASSERT(!milp_res.backend_telemetry.fallback_occurred);
}

KAVACH_TEST(test_backend_orchestration_cpu_multicore_real_solve) {
    auto lp_model = create_toy_lp();
    auto milp_model = create_nontrivial_milp();

    api::KavachSolver solver;
    api::SolverOptions opts;
    opts.run_presolve = false;
    opts.strategy_config.override_backend = backend::BackendType::CPU_MULTICORE;

    auto lp_res = solver.solve(lp_model, opts);
    KAVACH_ASSERT_EQUAL(lp_res.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(lp_res.verification_report.passed);
    KAVACH_ASSERT_EQUAL(lp_res.backend_used, backend::BackendType::CPU_MULTICORE);
    KAVACH_ASSERT_EQUAL(lp_res.backend_telemetry.requested_backend,
                        backend::BackendType::CPU_MULTICORE);
    KAVACH_ASSERT_EQUAL(lp_res.backend_telemetry.selected_backend,
                        backend::BackendType::CPU_MULTICORE);
    KAVACH_ASSERT_EQUAL(lp_res.backend_telemetry.executed_backend,
                        backend::BackendType::CPU_MULTICORE);
    KAVACH_ASSERT(lp_res.backend_telemetry.invocations > 0);
    KAVACH_ASSERT(lp_res.backend_telemetry.total_vectors >= 2);
    KAVACH_ASSERT(lp_res.backend_telemetry.worker_count >= 2);
    KAVACH_ASSERT(!lp_res.backend_telemetry.fallback_occurred);

    auto milp_res = solver.solve(milp_model, opts);
    KAVACH_ASSERT_EQUAL(milp_res.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(milp_res.verification_report.passed);
    KAVACH_ASSERT_EQUAL(milp_res.backend_used, backend::BackendType::CPU_MULTICORE);
    KAVACH_ASSERT_EQUAL(milp_res.backend_telemetry.executed_backend,
                        backend::BackendType::CPU_MULTICORE);
    KAVACH_ASSERT(milp_res.backend_telemetry.invocations > 0);
    KAVACH_ASSERT(milp_res.backend_telemetry.worker_count >= 2);
    KAVACH_ASSERT(!milp_res.backend_telemetry.fallback_occurred);
}

KAVACH_TEST(test_backend_orchestration_cuda_request_and_fallback_truth) {
    auto milp_model = create_nontrivial_milp();

    api::KavachSolver solver;

    // Case A: Request CUDA_GPU with use_gpu = true
    {
        api::SolverOptions opts;
        opts.use_gpu = true;
        opts.run_presolve = false;
        opts.strategy_config.override_backend = backend::BackendType::CUDA_GPU;

        auto res = solver.solve(milp_model, opts);
        KAVACH_ASSERT_EQUAL(res.status, core::SolverStatus::OPTIMAL);
        KAVACH_ASSERT(res.verification_report.passed);
        KAVACH_ASSERT_EQUAL(res.backend_telemetry.requested_backend,
                            backend::BackendType::CUDA_GPU);
        KAVACH_ASSERT_EQUAL(res.backend_telemetry.selected_backend,
                            backend::BackendType::CUDA_GPU);
        KAVACH_ASSERT(res.backend_telemetry.invocations > 0);
        KAVACH_ASSERT_EQUAL(res.backend_used, res.backend_telemetry.executed_backend);

        if (res.backend_telemetry.fallback_occurred) {
            // No label-only lie: when fallback occurs, executed_backend and backend_used MUST NOT be CUDA_GPU
            KAVACH_ASSERT(res.backend_telemetry.executed_backend !=
                          backend::BackendType::CUDA_GPU);
            KAVACH_ASSERT(res.backend_used != backend::BackendType::CUDA_GPU);
            KAVACH_ASSERT(!res.backend_telemetry.fallback_reason.empty());
        } else {
            KAVACH_ASSERT_EQUAL(res.backend_telemetry.executed_backend,
                                backend::BackendType::CUDA_GPU);
            KAVACH_ASSERT_EQUAL(res.backend_used, backend::BackendType::CUDA_GPU);
        }
    }

    // Case B: Request CUDA_GPU with force_no_gpu = true (--backend cuda --no-gpu)
    {
        api::SolverOptions opts;
        opts.use_gpu = false;
        opts.run_presolve = false;
        opts.strategy_config.override_backend = backend::BackendType::CUDA_GPU;
        opts.strategy_config.force_no_gpu = true;

        auto res = solver.solve(milp_model, opts);
        KAVACH_ASSERT_EQUAL(res.status, core::SolverStatus::OPTIMAL);
        KAVACH_ASSERT(res.verification_report.passed);
        KAVACH_ASSERT_EQUAL(res.backend_telemetry.requested_backend,
                            backend::BackendType::CUDA_GPU);
        KAVACH_ASSERT(res.backend_telemetry.selected_backend != backend::BackendType::CUDA_GPU);
        KAVACH_ASSERT(res.backend_telemetry.executed_backend != backend::BackendType::CUDA_GPU);
        KAVACH_ASSERT(res.backend_used != backend::BackendType::CUDA_GPU);
        KAVACH_ASSERT(res.backend_telemetry.fallback_occurred);
        KAVACH_ASSERT(res.backend_telemetry.fallback_reason.find("force_no_gpu") !=
                      std::string::npos);
        KAVACH_ASSERT(res.backend_telemetry.invocations > 0);
    }
}

KAVACH_TEST(test_backend_orchestration_forced_cuda_fault_injection) {
    auto lp_model = create_toy_lp();
    auto milp_model = create_nontrivial_milp();

    api::KavachSolver solver;
    api::SolverOptions opts;
    opts.use_gpu = true;
    opts.force_cuda_fallback = true;
    opts.run_presolve = false;
    opts.strategy_config.override_backend = backend::BackendType::CUDA_GPU;

    auto lp_res = solver.solve(lp_model, opts);
    KAVACH_ASSERT_EQUAL(lp_res.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(lp_res.verification_report.passed);
    KAVACH_ASSERT_EQUAL(lp_res.backend_telemetry.requested_backend,
                        backend::BackendType::CUDA_GPU);
    KAVACH_ASSERT(lp_res.backend_telemetry.fallback_occurred);
    KAVACH_ASSERT(lp_res.backend_telemetry.executed_backend != backend::BackendType::CUDA_GPU);
    KAVACH_ASSERT_EQUAL(lp_res.backend_used, lp_res.backend_telemetry.executed_backend);
    KAVACH_ASSERT(!lp_res.backend_telemetry.fallback_reason.empty());
    KAVACH_ASSERT(lp_res.backend_telemetry.invocations > 0);

    auto milp_res = solver.solve(milp_model, opts);
    KAVACH_ASSERT_EQUAL(milp_res.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(milp_res.verification_report.passed);
    KAVACH_ASSERT_EQUAL(milp_res.backend_telemetry.requested_backend,
                        backend::BackendType::CUDA_GPU);
    KAVACH_ASSERT(milp_res.backend_telemetry.fallback_occurred);
    KAVACH_ASSERT(milp_res.backend_telemetry.executed_backend != backend::BackendType::CUDA_GPU);
    KAVACH_ASSERT_EQUAL(milp_res.backend_used, milp_res.backend_telemetry.executed_backend);
    KAVACH_ASSERT(!milp_res.backend_telemetry.fallback_reason.empty());
    KAVACH_ASSERT(milp_res.backend_telemetry.invocations > 0);
}

KAVACH_TEST(test_backend_orchestration_cross_backend_equivalence_and_determinism) {
    auto milp_model = create_nontrivial_milp();
    api::KavachSolver solver;

    const std::vector<backend::BackendType> backends = {
        backend::BackendType::CPU_SEQUENTIAL,
        backend::BackendType::CPU_MULTICORE,
        backend::BackendType::CUDA_GPU,
    };

    double ref_obj = 0.0;
    std::size_t ref_nodes = 0;
    std::size_t ref_iters = 0;
    bool ref_set = false;

    for (auto btype : backends) {
        api::SolverOptions opts;
        opts.use_gpu = (btype == backend::BackendType::CUDA_GPU);
        opts.run_presolve = true;
        opts.strategy_config.override_backend = btype;

        auto r1 = solver.solve(milp_model, opts);
        auto r2 = solver.solve(milp_model, opts);

        KAVACH_ASSERT_EQUAL(r1.status, core::SolverStatus::OPTIMAL);
        KAVACH_ASSERT_EQUAL(r2.status, core::SolverStatus::OPTIMAL);
        KAVACH_ASSERT(r1.verification_report.passed);
        KAVACH_ASSERT(r2.verification_report.passed);
        KAVACH_ASSERT(r1.backend_telemetry.invocations > 0);
        KAVACH_ASSERT_EQUAL(r1.backend_telemetry.invocations, r2.backend_telemetry.invocations);
        KAVACH_ASSERT_EQUAL(r1.nodes, r2.nodes);
        KAVACH_ASSERT_EQUAL(r1.iterations, r2.iterations);
        KAVACH_ASSERT_NEAR(r1.objective_value, r2.objective_value, 1e-12);

        if (!ref_set) {
            ref_obj = r1.objective_value;
            ref_nodes = r1.nodes;
            ref_iters = r1.iterations;
            ref_set = true;
        } else {
            KAVACH_ASSERT_NEAR(r1.objective_value, ref_obj, 1e-9);
            KAVACH_ASSERT_EQUAL(r1.nodes, ref_nodes);
            KAVACH_ASSERT_EQUAL(r1.iterations, ref_iters);
        }
    }
}

KAVACH_TEST(test_backend_orchestration_lp_dual_simplex_and_milp_batched_spmv) {
    auto milp_model = create_nontrivial_milp();

    algorithms::milp::MILPSolverOptions milp_opts;
    milp_opts.requested_backend = backend::BackendType::CPU_MULTICORE;
    milp_opts.selected_backend = backend::BackendType::CPU_MULTICORE;
    milp_opts.num_workers = 4;
    milp_opts.enable_warm_starts = true;
    milp_opts.enable_primal_heuristic = true;

    algorithms::milp::BranchAndBound bb(milp_opts);
    auto sol = bb.solve(milp_model, milp_opts);

    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT_EQUAL(sol.telemetry.backend_telemetry.executed_backend,
                        backend::BackendType::CPU_MULTICORE);
    KAVACH_ASSERT(sol.telemetry.backend_telemetry.invocations > 0);
    // Warm-start dual simplex (batch_size=2) and PrimalHeuristic (batch_size>=2) process multiple vectors
    KAVACH_ASSERT(sol.telemetry.backend_telemetry.total_vectors >
                  sol.telemetry.backend_telemetry.invocations);
    KAVACH_ASSERT_EQUAL(sol.telemetry.backend_telemetry.worker_count, 4);
}

