#include <iomanip>
#include <iostream>

#include "api/solver.hpp"
#include "model/model.hpp"

/**
 * @brief Public API Usage Demonstration for Kavach-Opt.
 *
 * Demonstrates:
 * 1. Programmatic formulation of an optimization problem via CanonicalModel.
 * 2. Pre-solve strategy inspection and mathematical explainability.
 * 3. Certified solving with the independent verification layer.
 */
int main() {
    std::cout << "========================================================\n"
              << "Kavach-Opt Public C++ API Usage Demonstration\n"
              << "========================================================\n\n";

    // 1. Programmatic Model Construction
    // Maximize Z = 3*x1 + 2*x2
    // Subject to:
    //   2*x1 +   x2 <= 100  (Capacity Constraint 1)
    //     x1 +   x2 <= 80   (Capacity Constraint 2)
    //     x1 + 2*x2 <= 120  (Quality Constraint)
    //   x1 >= 0, x2 >= 0
    kavach::model::CanonicalModel model("production_planning_demo");
    model.set_objective_sense(kavach::model::ObjectiveSense::MAXIMIZE);

    auto x1 = model.add_variable("x1", kavach::model::VariableType::CONTINUOUS, 0.0, 100.0);
    auto x2 = model.add_variable("x2", kavach::model::VariableType::CONTINUOUS, 0.0, 100.0);

    model.set_objective_linear_coefficient(x1, 3.0);
    model.set_objective_linear_coefficient(x2, 2.0);

    kavach::model::SparseLinearExpression c1;
    c1.add_term(x1, 2.0);
    c1.add_term(x2, 1.0);
    model.add_constraint("capacity_1", std::move(c1), kavach::model::ConstraintSense::LESS_EQUAL,
                         100.0);

    kavach::model::SparseLinearExpression c2;
    c2.add_term(x1, 1.0);
    c2.add_term(x2, 1.0);
    model.add_constraint("capacity_2", std::move(c2), kavach::model::ConstraintSense::LESS_EQUAL,
                         80.0);

    kavach::model::SparseLinearExpression c3;
    c3.add_term(x1, 1.0);
    c3.add_term(x2, 2.0);
    model.add_constraint("quality_limit", std::move(c3), kavach::model::ConstraintSense::LESS_EQUAL,
                         120.0);

    std::cout << "1. Model Formulated: " << model.name() << "\n"
              << "   Variables: " << model.num_variables()
              << ", Constraints: " << model.num_constraints()
              << ", Nonzeros: " << model.num_nonzeros() << "\n\n";

    // 2. Instantiate Solver & Inspect Strategy Plan
    kavach::api::KavachSolver solver;
    kavach::strategy::StrategyDecision decision = solver.plan(model);

    std::cout << "2. Rule-Based Strategy Plan:\n"
              << "   Class:       " << kavach::strategy::to_string(decision.problem_class) << "\n"
              << "   Presolve:    " << kavach::strategy::to_string(decision.presolve_policy) << "\n"
              << "   Scaling:     " << kavach::strategy::to_string(decision.scaling_policy) << "\n"
              << "   Fired Rules: ";
    for (std::size_t i = 0; i < decision.rules_fired.size(); ++i) {
        std::cout << decision.rules_fired[i] << (i + 1 < decision.rules_fired.size() ? ", " : "\n");
    }
    std::cout << "\n";

    // 3. Solve with Autonomous Strategy Routing & Independent Verification
    kavach::api::SolverResult result = solver.solve(model);

    std::cout << "3. Solve & Certification Result:\n"
              << "   Status:          " << kavach::core::to_string(result.status) << "\n"
              << "   Objective Value: " << std::fixed << std::setprecision(4)
              << result.objective_value << "\n"
              << "   Iterations:      " << result.iterations << "\n"
              << "   Solve Time:      " << std::setprecision(6) << result.solve_time_seconds
              << " s\n"
              << "   Verification:    "
              << (result.verification_report.passed ? "CERTIFIED (PASS)" : "REJECTED (FAIL)")
              << "\n";

    if (!result.primal_solution.empty()) {
        std::cout << "   Primal Solution:\n"
                  << "     x1 = " << result.primal_solution[x1] << "\n"
                  << "     x2 = " << result.primal_solution[x2] << "\n";
    }

    std::cout << "\n========================================================\n"
              << "API Demonstration Completed Successfully.\n"
              << "========================================================\n";

    return (result.status == kavach::core::SolverStatus::OPTIMAL &&
            result.verification_report.passed)
               ? 0
               : 1;
}
