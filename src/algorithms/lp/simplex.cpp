#include "algorithms/lp/simplex.hpp"

#include <chrono>
#include <optional>
#include <sstream>

#include "algorithms/lp/dual_simplex.hpp"
#include "algorithms/lp/revised_simplex.hpp"
#include "algorithms/lp/standard_form.hpp"
#include "core/error.hpp"
#include "core/logger.hpp"
#include "numerics/scaling.hpp"
#include "presolve/presolve.hpp"
#include "verification/lp_verifier.hpp"

namespace kavach::algorithms::lp {

LPSolution RevisedSimplex::solve(const model::CanonicalModel& model,
                                 const core::SolverTolerances& tolerances) {
    LPSolverOptions opts = options_;
    opts.tolerances = tolerances;
    return solve(model, opts);
}

LPSolution RevisedSimplex::solve(const model::CanonicalModel& model) {
    return solve(model, options_);
}

LPSolution RevisedSimplex::solve(const model::CanonicalModel& model,
                                 const LPSolverOptions& options) {
    auto start_time = std::chrono::steady_clock::now();
    LPSolution solution;
    solution.telemetry.original_rows = model.num_constraints();
    solution.telemetry.original_cols = model.num_variables();

    // 1. Problem class validation: Reject non-continuous (integer/binary) models
    for (std::size_t j = 0; j < model.num_variables(); ++j) {
        const auto& var = model.variable(static_cast<model::VarId>(j));
        if (!var.is_continuous()) {
            throw core::InputException("LP solver received non-continuous variable '" + var.name +
                                       "' (type: " + std::string(model::to_string(var.type)) +
                                       "). LP solver strictly rejects MILP models.");
        }
    }

    if (model.is_qp()) {
        throw core::InputException(
            "LP solver received a model with quadratic objective terms. "
            "LP solver strictly rejects QP models.");
    }

    // 2. Model structure validation
    auto val_result = model.validate();
    if (!val_result.valid) {
        solution.status = core::SolverStatus::UNKNOWN;
        std::string msg = "Model validation failed before solve:";
        for (const auto& err : val_result.errors) {
            msg += " [" + err.component + ": " + err.message + "]";
        }
        solution.message = msg;
        return solution;
    }

    // 3. Presolve pipeline (if enabled)
    std::unique_ptr<model::CanonicalModel> reduced_model_storage;
    std::optional<presolve::ReconstructionMap> reconstruction_map;
    const model::CanonicalModel* working_model = &model;

    if (options.enable_presolve) {
        presolve::Presolver presolver;
        presolve::PresolveOptions popts;
        popts.tolerances = options.tolerances;
        auto presolve_res = presolver.apply(model, popts);

        solution.telemetry.presolved_rows = presolve_res.stats.reduced_constraints;
        solution.telemetry.presolved_cols = presolve_res.stats.reduced_variables;

        if (presolve_res.is_infeasible()) {
            solution.status = core::SolverStatus::INFEASIBLE;
            solution.message = "Presolve detected infeasibility: " + presolve_res.message;
            return solution;
        }

        if (presolve_res.is_unbounded()) {
            solution.status = core::SolverStatus::UNBOUNDED;
            solution.message = "Presolve detected unboundedness: " + presolve_res.message;
            return solution;
        }

        if (presolve_res.has_reduced_model()) {
            reduced_model_storage = std::move(presolve_res.reduced_model);
            reconstruction_map = std::move(presolve_res.reconstruction_map);
            working_model = reduced_model_storage.get();
        }
    } else {
        solution.telemetry.presolved_rows = model.num_constraints();
        solution.telemetry.presolved_cols = model.num_variables();
    }

    // 4. Standard Form Transformation
    StandardFormLP standard_lp(*working_model, options.tolerances,
                               options.preserve_bounds_structure);

    // 5. Scaling (if enabled)
    std::optional<numerics::ScalingFactors> scaling_factors;
    if (options.enable_scaling && standard_lp.num_rows() > 0 && standard_lp.num_cols() > 0) {
        auto diag = numerics::analyze_conditioning(standard_lp.A_csr());
        if (diag.recommend_scaling) {
            scaling_factors = numerics::compute_geometric_mean_scaling(standard_lp.A_csr(), 4);
            standard_lp.apply_scaling(*scaling_factors);
        }
    }

    // 6. Execute Revised Simplex Core
    RevisedSimplexCore core(standard_lp, options);
    auto exec_res = core.run(solution.telemetry);

    solution.status = exec_res.status;
    solution.message = exec_res.message;
    solution.iterations = solution.telemetry.total_iterations;

    if (exec_res.status != core::SolverStatus::OPTIMAL) {
        auto end_time = std::chrono::steady_clock::now();
        solution.telemetry.solve_time_seconds =
            std::chrono::duration<double>(end_time - start_time).count();
        return solution;
    }

    // Capture reusable basis state
    if (exec_res.basis.has_value()) {
        solution.basis = exec_res.basis;
        if (scaling_factors.has_value() && solution.basis.has_value()) {
            solution.basis->scaling_applied = true;
            solution.basis->scaling_factors = scaling_factors;
        }
    }

    // 7. Unscaling (if scaling was applied)
    if (scaling_factors.has_value()) {
        exec_res.z = scaling_factors->unscale_primal(exec_res.z);
        exec_res.y = scaling_factors->unscale_dual(exec_res.y);
        exec_res.reduced_costs = scaling_factors->unscale_reduced_costs(exec_res.reduced_costs);
    }

    // 8. Reconstruct working model solution
    auto x_reduced = standard_lp.reconstruct_primal(exec_res.z);
    auto dual_reduced = standard_lp.reconstruct_dual(exec_res.y);
    auto rc_reduced = standard_lp.reconstruct_reduced_costs(exec_res.reduced_costs);

    // 9. Reconstruct original model solution (through Presolve reconstruction map if present)
    if (reconstruction_map.has_value()) {
        solution.primal_solution = reconstruction_map->restore_solution(x_reduced);
        solution.dual_multipliers = dual_reduced;
        solution.reduced_costs = rc_reduced;
    } else {
        solution.primal_solution = x_reduced;
        solution.dual_multipliers = dual_reduced;
        solution.reduced_costs = rc_reduced;
    }

    // 10. Compute true original objective value directly from original model and unscaled primal
    double obj = model.objective().offset;
    for (const auto& term : model.objective().linear_terms.terms()) {
        if (term.var_id < solution.primal_solution.size()) {
            obj += term.coefficient * solution.primal_solution[term.var_id];
        }
    }
    solution.objective_value = obj;

    // 11. Independent Verification [Y] — mandatory for OPTIMAL claim
    // The verifier operates on the original (pre-presolve) model and the
    // fully-reconstructed solution. It does NOT call any solver-internal
    // feasibility or residual helpers.
    {
        verification::LPVerifier verifier(
            verification::VerifierTolerances::from_solver_tolerances(options.tolerances));

        if (working_model != &model) {
            double obj_reduced = standard_lp.compute_original_objective(x_reduced);
            auto vr_reduced = verifier.verify(*working_model, x_reduced, dual_reduced, rc_reduced,
                                              obj_reduced, exec_res.status);

            // Verify primal feasibility, bounds, nonfinite, and objective on the original model:
            auto vr_orig = verifier.verify(model, solution.primal_solution, {}, {},
                                           solution.objective_value, core::SolverStatus::FEASIBLE);

            if (!vr_reduced.fully_verified() || !vr_orig.primal_feasibility.passed ||
                !vr_orig.bound_feasibility.passed || !vr_orig.objective_consistency.passed ||
                !vr_orig.nonfinite_check.passed) {
                solution.status = vr_reduced.certified_status != core::SolverStatus::OPTIMAL
                                      ? vr_reduced.certified_status
                                      : core::SolverStatus::NUMERICAL_FAILURE;
                solution.message = "Verifier rejected presolved solution: " + vr_orig.summary;
            } else {
                solution.status = core::SolverStatus::OPTIMAL;
                solution.message = vr_orig.summary;
            }
        } else {
            auto vr =
                verifier.verify(model, solution.primal_solution, solution.dual_multipliers,
                                solution.reduced_costs, solution.objective_value, solution.status);

            if (vr.status_downgraded) {
                solution.status = vr.certified_status;
                solution.message = "Verifier downgraded status: " + vr.summary;
            } else if (vr.certified_status == core::SolverStatus::OPTIMAL && vr.fully_verified()) {
                // OPTIMAL confirmed — status unchanged
            }
            if (solution.message.empty()) {
                solution.message = vr.summary;
            }
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    solution.telemetry.solve_time_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    return solution;
}

LPSolution RevisedSimplex::solve_warm_start(const model::CanonicalModel& model,
                                            const BasisState& warm_basis,
                                            const LPSolverOptions& options) {
    auto start_time = std::chrono::steady_clock::now();
    LPSolution solution;
    solution.telemetry.warm_start_attempts = 1;

    if (!warm_basis.is_valid) {
        solution.status = core::SolverStatus::NUMERICAL_FAILURE;
        solution.message = "Warm start rejected: invalid basis state provided";
        return solution;
    }

    // 1. Problem Class Validation
    if (model.is_qp()) {
        solution.status = core::SolverStatus::NUMERICAL_FAILURE;
        solution.message = "QP models rejected: use active-set QP solver";
        return solution;
    }

    // 2. Model validation
    auto val = model.validate();
    if (!val.valid) {
        solution.status = core::SolverStatus::UNKNOWN;
        solution.message = "Model validation failed";
        return solution;
    }

    // 3. Transform to standard form preserving bounds structure
    StandardFormLP standard_lp(model, options.tolerances, options.preserve_bounds_structure);

    // 4. Verify structural compatibility with warm_basis
    if (standard_lp.num_rows() != warm_basis.num_rows ||
        standard_lp.num_cols() != warm_basis.num_cols) {
        solution.status = core::SolverStatus::NUMERICAL_FAILURE;
        solution.message =
            "Structural dimension mismatch: LP has (" + std::to_string(standard_lp.num_rows()) +
            "x" + std::to_string(standard_lp.num_cols()) + ") but basis has (" +
            std::to_string(warm_basis.num_rows) + "x" + std::to_string(warm_basis.num_cols) + ")";
        return solution;
    }

    // 5. Apply scaling if the warm_basis was scaled
    if (warm_basis.scaling_applied) {
        if (!warm_basis.scaling_factors.has_value()) {
            solution.status = core::SolverStatus::NUMERICAL_FAILURE;
            solution.message = "Scaling mismatch: warm basis missing scaling factors";
            return solution;
        }
        standard_lp.apply_scaling(*warm_basis.scaling_factors);
    }

    // 6. Execute Dual Simplex
    DualSimplexCore dual_core(standard_lp, options);
    std::optional<BasisState> resulting_basis;
    auto exec_res = dual_core.run_warm_start(warm_basis, solution.telemetry, resulting_basis);

    solution.status = exec_res.status;
    solution.message = exec_res.message;
    solution.iterations = solution.telemetry.total_iterations;

    if (exec_res.status != core::SolverStatus::OPTIMAL) {
        auto end_time = std::chrono::steady_clock::now();
        solution.telemetry.solve_time_seconds =
            std::chrono::duration<double>(end_time - start_time).count();
        return solution;
    }

    // 7. Unscaling (if scaling was applied)
    if (warm_basis.scaling_applied && warm_basis.scaling_factors.has_value()) {
        exec_res.z = warm_basis.scaling_factors->unscale_primal(exec_res.z);
        exec_res.y = warm_basis.scaling_factors->unscale_dual(exec_res.y);
        exec_res.reduced_costs =
            warm_basis.scaling_factors->unscale_reduced_costs(exec_res.reduced_costs);
    }

    // 8. Reconstruct solution in model variable space
    solution.primal_solution = standard_lp.reconstruct_primal(exec_res.z);
    solution.dual_multipliers = standard_lp.reconstruct_dual(exec_res.y);
    solution.reduced_costs = standard_lp.reconstruct_reduced_costs(exec_res.reduced_costs);

    // 9. Compute objective
    double obj = model.objective().offset;
    for (const auto& term : model.objective().linear_terms.terms()) {
        if (term.var_id < solution.primal_solution.size()) {
            obj += term.coefficient * solution.primal_solution[term.var_id];
        }
    }
    solution.objective_value = obj;

    // 10. Independent Verification [Y]
    {
        verification::LPVerifier verifier(
            verification::VerifierTolerances::from_solver_tolerances(options.tolerances));
        auto vr =
            verifier.verify(model, solution.primal_solution, solution.dual_multipliers,
                            solution.reduced_costs, solution.objective_value, solution.status);
        if (vr.status_downgraded) {
            solution.status = vr.certified_status;
            solution.message = "Verifier downgraded status: " + vr.summary;
        } else if (vr.certified_status == core::SolverStatus::OPTIMAL && vr.fully_verified()) {
            // Confirmed
            if (resulting_basis.has_value()) {
                solution.basis = resulting_basis;
                if (warm_basis.scaling_applied) {
                    solution.basis->scaling_applied = true;
                    solution.basis->scaling_factors = warm_basis.scaling_factors;
                }
            }
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    solution.telemetry.solve_time_seconds =
        std::chrono::duration<double>(end_time - start_time).count();
    return solution;
}

}  // namespace kavach::algorithms::lp
