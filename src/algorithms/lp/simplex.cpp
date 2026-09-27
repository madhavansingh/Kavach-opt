#include "algorithms/lp/simplex.hpp"

#include <chrono>
#include <optional>
#include <sstream>

#include "algorithms/lp/dual_simplex.hpp"
#include "algorithms/lp/revised_simplex.hpp"
#include "algorithms/lp/standard_form.hpp"
#include "core/error.hpp"
#include "core/logger.hpp"
#include "numerics/model_converter.hpp"
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

    std::unique_ptr<backend::ExecutionBackend> local_backend;
    LPSolverOptions effective_options = options;
    if (effective_options.execution_backend == nullptr) {
        local_backend = backend::create_execution_backend(
            effective_options.requested_backend, effective_options.selected_backend,
            effective_options.num_workers, effective_options.force_cuda_fallback,
            effective_options.initial_fallback_reason);
        effective_options.execution_backend = local_backend.get();
    }
    backend::ExecutionBackend* active_backend = effective_options.execution_backend;

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
        if (active_backend != nullptr) {
            solution.telemetry.backend_telemetry = active_backend->execution_telemetry();
        }
        return solution;
    }

    // 3. Presolve pipeline (if enabled)
    std::unique_ptr<model::CanonicalModel> reduced_model_storage;
    std::optional<presolve::ReconstructionMap> reconstruction_map;
    const model::CanonicalModel* working_model = &model;

    if (effective_options.enable_presolve) {
        presolve::Presolver presolver;
        presolve::PresolveOptions popts;
        popts.tolerances = effective_options.tolerances;
        auto presolve_res = presolver.apply(model, popts);

        solution.telemetry.presolved_rows = presolve_res.stats.reduced_constraints;
        solution.telemetry.presolved_cols = presolve_res.stats.reduced_variables;

        if (presolve_res.is_infeasible()) {
            solution.status = core::SolverStatus::INFEASIBLE;
            solution.message = "Presolve detected infeasibility: " + presolve_res.message;
            if (active_backend != nullptr) {
                solution.telemetry.backend_telemetry = active_backend->execution_telemetry();
            }
            return solution;
        }

        if (presolve_res.is_unbounded()) {
            solution.status = core::SolverStatus::UNBOUNDED;
            solution.message = "Presolve detected unboundedness: " + presolve_res.message;
            if (active_backend != nullptr) {
                solution.telemetry.backend_telemetry = active_backend->execution_telemetry();
            }
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
    StandardFormLP standard_lp(*working_model, effective_options.tolerances,
                               effective_options.preserve_bounds_structure);

    // 5. Scaling (if enabled)
    std::optional<numerics::ScalingFactors> scaling_factors;
    if (effective_options.enable_scaling && standard_lp.num_rows() > 0 &&
        standard_lp.num_cols() > 0) {
        auto diag = numerics::analyze_conditioning(standard_lp.A_csr());
        if (diag.recommend_scaling) {
            scaling_factors = numerics::compute_geometric_mean_scaling(standard_lp.A_csr(), 4);
            standard_lp.apply_scaling(*scaling_factors);
        }
    }

    // 6. Execute Revised Simplex Core
    RevisedSimplexCore core(standard_lp, effective_options);
    auto exec_res = core.run(solution.telemetry);

    solution.status = exec_res.status;
    solution.message = exec_res.message;
    solution.iterations = solution.telemetry.total_iterations;

    if (exec_res.status != core::SolverStatus::OPTIMAL) {
        if (active_backend != nullptr) {
            solution.telemetry.backend_telemetry = active_backend->execution_telemetry();
        }
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

    // 9. Reconstruct original model solution and optimality certificate
    if (reconstruction_map.has_value()) {
        auto cert = reconstruction_map->restore_certificate(
            x_reduced, dual_reduced, rc_reduced, effective_options.tolerances);
        if (!cert.valid) {
            solution.status = core::SolverStatus::NUMERICAL_FAILURE;
            solution.message = "Presolve certificate reconstruction failed: " + cert.diagnostic;
            if (active_backend != nullptr) {
                solution.telemetry.backend_telemetry = active_backend->execution_telemetry();
            }
            auto end_time = std::chrono::steady_clock::now();
            solution.telemetry.solve_time_seconds =
                std::chrono::duration<double>(end_time - start_time).count();
            return solution;
        }
        solution.primal_solution = std::move(cert.primal_solution);
        solution.dual_multipliers = std::move(cert.dual_multipliers);
        solution.reduced_costs = std::move(cert.reduced_costs);
        solution.objective_value = cert.objective_value;
    } else {
        solution.primal_solution = x_reduced;
        solution.dual_multipliers = dual_reduced;
        solution.reduced_costs = rc_reduced;
        double obj = model.objective().offset;
        for (const auto& term : model.objective().linear_terms.terms()) {
            if (term.var_id < solution.primal_solution.size()) {
                obj += term.coefficient * solution.primal_solution[term.var_id];
            }
        }
        solution.objective_value = obj;
    }

    // 10. Original-model constraint activity evaluation via ExecutionBackend
    // (Also ensures backend execution occurs even if presolve reduced all rows)
    if (active_backend != nullptr && model.num_constraints() > 0 &&
        solution.primal_solution.size() == model.num_variables()) {
        const auto orig_csr = numerics::canonical_model_to_csr(model);
        std::vector<double> orig_activities(model.num_constraints(), 0.0);
        active_backend->batched_spmv(orig_csr, solution.primal_solution.data(),
                                     orig_activities.data(), 1);
    }

    // 11. Independent Verification [Y] — mandatory for OPTIMAL claim
    // The verifier operates on the original (pre-presolve) model and the
    // fully-reconstructed primal/dual/reduced-cost certificate.
    {
        verification::LPVerifier verifier(
            verification::VerifierTolerances::from_solver_tolerances(effective_options.tolerances));

        auto vr =
            verifier.verify(model, solution.primal_solution, solution.dual_multipliers,
                            solution.reduced_costs, solution.objective_value, solution.status);

        if (vr.status_downgraded) {
            solution.status = vr.certified_status;
            solution.message = "Verifier downgraded status: " + vr.summary;
        } else if (vr.certified_status == core::SolverStatus::OPTIMAL && vr.fully_verified()) {
            solution.status = core::SolverStatus::OPTIMAL;
            solution.message = vr.summary;
        }
        if (solution.message.empty()) {
            solution.message = vr.summary;
        }
    }

    if (active_backend != nullptr) {
        solution.telemetry.backend_telemetry = active_backend->execution_telemetry();
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

    std::unique_ptr<backend::ExecutionBackend> local_backend;
    LPSolverOptions effective_options = options;
    if (effective_options.execution_backend == nullptr) {
        local_backend = backend::create_execution_backend(
            effective_options.requested_backend, effective_options.selected_backend,
            effective_options.num_workers, effective_options.force_cuda_fallback,
            effective_options.initial_fallback_reason);
        effective_options.execution_backend = local_backend.get();
    }
    backend::ExecutionBackend* active_backend = effective_options.execution_backend;

    if (!warm_basis.is_valid) {
        solution.status = core::SolverStatus::NUMERICAL_FAILURE;
        solution.message = "Warm start rejected: invalid basis state provided";
        if (active_backend != nullptr) {
            solution.telemetry.backend_telemetry = active_backend->execution_telemetry();
        }
        return solution;
    }

    // 1. Problem Class Validation
    if (model.is_qp()) {
        solution.status = core::SolverStatus::NUMERICAL_FAILURE;
        solution.message = "QP models rejected: use active-set QP solver";
        if (active_backend != nullptr) {
            solution.telemetry.backend_telemetry = active_backend->execution_telemetry();
        }
        return solution;
    }

    // 2. Model validation
    auto val = model.validate();
    if (!val.valid) {
        solution.status = core::SolverStatus::UNKNOWN;
        solution.message = "Model validation failed";
        if (active_backend != nullptr) {
            solution.telemetry.backend_telemetry = active_backend->execution_telemetry();
        }
        return solution;
    }

    // 3. Transform to standard form preserving bounds structure
    StandardFormLP standard_lp(model, effective_options.tolerances,
                               effective_options.preserve_bounds_structure);

    // 4. Verify structural compatibility with warm_basis
    if (standard_lp.num_rows() != warm_basis.num_rows ||
        standard_lp.num_cols() != warm_basis.num_cols) {
        solution.status = core::SolverStatus::NUMERICAL_FAILURE;
        solution.message =
            "Structural dimension mismatch: LP has (" + std::to_string(standard_lp.num_rows()) +
            "x" + std::to_string(standard_lp.num_cols()) + ") but basis has (" +
            std::to_string(warm_basis.num_rows) + "x" + std::to_string(warm_basis.num_cols) + ")";
        if (active_backend != nullptr) {
            solution.telemetry.backend_telemetry = active_backend->execution_telemetry();
        }
        return solution;
    }

    // 5. Apply scaling if the warm_basis was scaled
    if (warm_basis.scaling_applied) {
        if (!warm_basis.scaling_factors.has_value()) {
            solution.status = core::SolverStatus::NUMERICAL_FAILURE;
            solution.message = "Scaling mismatch: warm basis missing scaling factors";
            if (active_backend != nullptr) {
                solution.telemetry.backend_telemetry = active_backend->execution_telemetry();
            }
            return solution;
        }
        standard_lp.apply_scaling(*warm_basis.scaling_factors);
    }

    // 6. Execute Dual Simplex
    DualSimplexCore dual_core(standard_lp, effective_options);
    std::optional<BasisState> resulting_basis;
    auto exec_res = dual_core.run_warm_start(warm_basis, solution.telemetry, resulting_basis);

    solution.status = exec_res.status;
    solution.message = exec_res.message;
    solution.iterations = solution.telemetry.total_iterations;

    if (exec_res.status != core::SolverStatus::OPTIMAL) {
        if (active_backend != nullptr) {
            solution.telemetry.backend_telemetry = active_backend->execution_telemetry();
        }
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
            verification::VerifierTolerances::from_solver_tolerances(effective_options.tolerances));
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

    if (active_backend != nullptr) {
        solution.telemetry.backend_telemetry = active_backend->execution_telemetry();
    }

    auto end_time = std::chrono::steady_clock::now();
    solution.telemetry.solve_time_seconds =
        std::chrono::duration<double>(end_time - start_time).count();
    return solution;
}

}  // namespace kavach::algorithms::lp
