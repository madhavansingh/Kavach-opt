#include "presolve/presolve.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/error.hpp"
#include "core/logger.hpp"

namespace kavach::presolve {

namespace {

struct WorkingTerm {
    model::VarId var_id{model::INVALID_VAR_ID};
    double coeff{0.0};
};

struct WorkingConstraint {
    model::ConstrId id{model::INVALID_CONSTR_ID};
    std::string name;
    std::vector<WorkingTerm> terms;
    model::ConstraintSense sense{model::ConstraintSense::LESS_EQUAL};
    double rhs{0.0};
    bool is_deleted{false};
};

struct WorkingVariable {
    model::VarId id{model::INVALID_VAR_ID};
    std::string name;
    model::VariableType type{model::VariableType::CONTINUOUS};
    double lb{0.0};
    double ub{model::INFINITY_BOUND};
    bool is_fixed{false};
    double fixed_val{0.0};
    bool is_deleted{false};
};

struct WorkingQuadTerm {
    model::VarId var1{model::INVALID_VAR_ID};
    model::VarId var2{model::INVALID_VAR_ID};
    double coeff{0.0};
};

struct WorkingObjective {
    model::ObjectiveSense sense{model::ObjectiveSense::MINIMIZE};
    double offset{0.0};
    std::vector<WorkingTerm> linear_terms;
    std::vector<WorkingQuadTerm> quadratic_terms;
};

}  // namespace

std::string PresolveStats::to_string() const {
    std::ostringstream oss;
    oss << "PresolveStats {\n"
        << "  passes: " << passes_performed << "\n"
        << "  variables: " << original_variables << " -> " << reduced_variables << " (-"
        << variables_removed << ", fixed: " << fixed_variables << ")\n"
        << "  constraints: " << original_constraints << " -> " << reduced_constraints << " (-"
        << rows_removed << ", empty: " << empty_rows_removed
        << ", redundant: " << redundant_rows_removed << ", singletons: " << singleton_reductions
        << ")\n"
        << "  bound tightenings: " << bound_tightenings << "\n"
        << "  coefficients eliminated: " << coefficients_eliminated << "\n"
        << std::scientific << std::setprecision(4)
        << "  objective offset shift: " << objective_constant_shift << "\n"
        << std::fixed << std::setprecision(4) << "  elapsed time: " << elapsed_time_seconds << "s\n"
        << "}";
    return oss.str();
}

ReconstructionMap::ReconstructionMap(std::size_t num_original_vars,
                                     std::size_t num_original_constrs)
    : variable_mappings_(num_original_vars), constraint_mappings_(num_original_constrs) {
    for (std::size_t i = 0; i < num_original_vars; ++i) {
        variable_mappings_[i].original_var_id = static_cast<model::VarId>(i);
        variable_mappings_[i].status = VariableMappingStatus::RETAINED;
        variable_mappings_[i].reduced_var_id = static_cast<model::VarId>(i);
        variable_mappings_[i].fixed_value = 0.0;
    }
    for (std::size_t i = 0; i < num_original_constrs; ++i) {
        constraint_mappings_[i].original_constr_id = static_cast<model::ConstrId>(i);
        constraint_mappings_[i].status = ConstraintMappingStatus::RETAINED;
        constraint_mappings_[i].reduced_constr_id = static_cast<model::ConstrId>(i);
    }
}

void ReconstructionMap::initialize_model_context(const model::CanonicalModel& original_model) {
    const std::size_t n = original_model.num_variables();
    const std::size_t m = original_model.num_constraints();

    if (variable_mappings_.size() != n) {
        variable_mappings_.resize(n);
        for (std::size_t j = 0; j < n; ++j) {
            variable_mappings_[j].original_var_id = static_cast<model::VarId>(j);
            variable_mappings_[j].status = VariableMappingStatus::RETAINED;
            variable_mappings_[j].reduced_var_id = static_cast<model::VarId>(j);
            variable_mappings_[j].fixed_value = 0.0;
        }
    }
    if (constraint_mappings_.size() != m) {
        constraint_mappings_.resize(m);
        for (std::size_t i = 0; i < m; ++i) {
            constraint_mappings_[i].original_constr_id = static_cast<model::ConstrId>(i);
            constraint_mappings_[i].status = ConstraintMappingStatus::RETAINED;
            constraint_mappings_[i].reduced_constr_id = static_cast<model::ConstrId>(i);
        }
    }

    original_sense_ = original_model.objective().sense;
    original_offset_ = original_model.objective().offset;
    original_obj_coeffs_.assign(n, 0.0);
    for (const auto& t : original_model.objective().linear_terms.terms()) {
        if (t.var_id < n) {
            original_obj_coeffs_[t.var_id] += t.coefficient;
        }
    }

    original_var_lb_.resize(n);
    original_var_ub_.resize(n);
    for (std::size_t j = 0; j < n; ++j) {
        const auto& v = original_model.variable(static_cast<model::VarId>(j));
        original_var_lb_[j] = v.lower_bound;
        original_var_ub_[j] = v.upper_bound;
    }

    original_row_sense_.resize(m);
    original_row_rhs_.resize(m);
    orig_row_ptr_.assign(m + 1, 0);
    orig_col_idx_.clear();
    orig_values_.clear();

    for (std::size_t i = 0; i < m; ++i) {
        const auto& c = original_model.constraint(static_cast<model::ConstrId>(i));
        original_row_sense_[i] = c.sense;
        original_row_rhs_[i] = c.rhs;
        for (const auto& term : c.expression.terms()) {
            if (term.var_id < n && term.coefficient != 0.0) {
                orig_col_idx_.push_back(term.var_id);
                orig_values_.push_back(term.coefficient);
            }
        }
        orig_row_ptr_[i + 1] = orig_col_idx_.size();
    }

    has_model_context_ = true;
}

void ReconstructionMap::set_mapping(model::VarId original_id, VariableMapping mapping) {
    if (original_id >= variable_mappings_.size()) {
        throw core::InputException("ReconstructionMap: original VarId out of range: " +
                                   std::to_string(original_id));
    }
    variable_mappings_[original_id] = mapping;
}

const VariableMapping& ReconstructionMap::mapping(model::VarId original_id) const {
    if (original_id >= variable_mappings_.size()) {
        throw core::InputException("ReconstructionMap: original VarId out of range: " +
                                   std::to_string(original_id));
    }
    return variable_mappings_[original_id];
}

void ReconstructionMap::set_constraint_mapping(model::ConstrId original_id,
                                               ConstraintMapping mapping) {
    if (original_id >= constraint_mappings_.size()) {
        throw core::InputException("ReconstructionMap: original ConstrId out of range: " +
                                   std::to_string(original_id));
    }
    constraint_mappings_[original_id] = mapping;
}

const ConstraintMapping& ReconstructionMap::constraint_mapping(model::ConstrId original_id) const {
    if (original_id >= constraint_mappings_.size()) {
        throw core::InputException("ReconstructionMap: original ConstrId out of range: " +
                                   std::to_string(original_id));
    }
    return constraint_mappings_[original_id];
}

void ReconstructionMap::add_ledger_entry(PresolveTransformEntry entry) {
    entry.step_id = ledger_.size();
    ledger_.push_back(std::move(entry));
}

std::vector<double> ReconstructionMap::restore_primal(
    const std::vector<double>& reduced_primal) const {
    std::vector<double> original(variable_mappings_.size(), 0.0);
    for (std::size_t i = 0; i < variable_mappings_.size(); ++i) {
        const auto& m = variable_mappings_[i];
        if (m.original_var_id != static_cast<model::VarId>(i)) {
            throw core::NumericalException(
                "ReconstructionMap: corrupted variable mapping at index " + std::to_string(i));
        }
        if (m.status == VariableMappingStatus::RETAINED) {
            if (m.reduced_var_id >= reduced_primal.size()) {
                throw core::NumericalException(
                    "ReconstructionMap: reduced VarId " + std::to_string(m.reduced_var_id) +
                    " exceeds reduced solution size " + std::to_string(reduced_primal.size()));
            }
            original[i] = reduced_primal[m.reduced_var_id];
        } else {
            original[i] = m.fixed_value;
        }
    }
    return original;
}

std::vector<double> ReconstructionMap::restore_solution(
    const std::vector<double>& reduced_solution) const {
    return restore_primal(reduced_solution);
}

std::vector<double> ReconstructionMap::restore_reduced_costs(
    const std::vector<double>& original_dual) const {
    const std::size_t n = variable_mappings_.size();
    const std::size_t m = constraint_mappings_.size();
    if (!has_model_context_ || original_obj_coeffs_.size() != n || original_dual.size() < m) {
        return {};
    }
    std::vector<double> rc = original_obj_coeffs_;
    for (std::size_t i = 0; i < m; ++i) {
        const double yi = original_dual[i];
        if (yi == 0.0) {
            continue;
        }
        for (std::size_t p = orig_row_ptr_[i]; p < orig_row_ptr_[i + 1]; ++p) {
            rc[orig_col_idx_[p]] -= orig_values_[p] * yi;
        }
    }
    return rc;
}

double ReconstructionMap::restore_objective(const std::vector<double>& original_primal) const {
    if (!has_model_context_ || original_primal.size() < original_obj_coeffs_.size()) {
        return 0.0;
    }
    double obj = original_offset_;
    for (std::size_t j = 0; j < original_obj_coeffs_.size(); ++j) {
        obj += original_obj_coeffs_[j] * original_primal[j];
    }
    return obj;
}

std::vector<double> ReconstructionMap::restore_dual(
    const std::vector<double>& reduced_primal, const std::vector<double>& reduced_dual,
    const std::vector<double>& reduced_rc, const core::SolverTolerances& tolerances) const {
    auto cert = restore_certificate(reduced_primal, reduced_dual, reduced_rc, tolerances);
    if (!cert.valid) {
        return {};
    }
    return cert.dual_multipliers;
}

ReconstructedCertificate ReconstructionMap::restore_certificate(
    const std::vector<double>& reduced_primal, const std::vector<double>& reduced_dual,
    const std::vector<double>& reduced_rc, const core::SolverTolerances& tolerances) const {
    ReconstructedCertificate cert;
    const std::size_t n = variable_mappings_.size();
    const std::size_t m = constraint_mappings_.size();

    if (!has_model_context_ || original_obj_coeffs_.size() != n || orig_row_ptr_.size() != m + 1) {
        cert.valid = false;
        cert.diagnostic = "ReconstructionMap: original model context not initialized.";
        return cert;
    }

    if (has_reduced_dimensions_) {
        if (reduced_primal.size() != num_reduced_vars_) {
            cert.valid = false;
            cert.diagnostic = "ReconstructionMap: reduced_primal size mismatch.";
            return cert;
        }
        if (reduced_dual.size() != num_reduced_constrs_) {
            cert.valid = false;
            cert.diagnostic = "ReconstructionMap: reduced_dual size mismatch.";
            return cert;
        }
        if (!reduced_rc.empty() && reduced_rc.size() != num_reduced_vars_) {
            cert.valid = false;
            cert.diagnostic = "ReconstructionMap: reduced_rc size mismatch.";
            return cert;
        }
    }

    // Check finite input values
    for (double v : reduced_primal) {
        if (!std::isfinite(v)) {
            cert.valid = false;
            cert.diagnostic = "ReconstructionMap: non-finite value in reduced_primal.";
            return cert;
        }
    }
    for (double v : reduced_dual) {
        if (!std::isfinite(v)) {
            cert.valid = false;
            cert.diagnostic = "ReconstructionMap: non-finite value in reduced_dual.";
            return cert;
        }
    }
    for (double v : reduced_rc) {
        if (!std::isfinite(v)) {
            cert.valid = false;
            cert.diagnostic = "ReconstructionMap: non-finite value in reduced_rc.";
            return cert;
        }
    }

    // 1. Validate variable mappings and initialize primal vector x
    std::vector<double> x(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        const auto& vm = variable_mappings_[j];
        if (vm.original_var_id != static_cast<model::VarId>(j)) {
            cert.valid = false;
            cert.diagnostic =
                "ReconstructionMap: corrupted variable mapping at index " + std::to_string(j);
            return cert;
        }
        if (vm.status == VariableMappingStatus::RETAINED) {
            if (vm.reduced_var_id >= reduced_primal.size()) {
                cert.valid = false;
                cert.diagnostic = "ReconstructionMap: reduced VarId out of range at variable " +
                                  std::to_string(j);
                return cert;
            }
            x[j] = reduced_primal[vm.reduced_var_id];
        } else {
            if (!std::isfinite(vm.fixed_value)) {
                cert.valid = false;
                cert.diagnostic =
                    "ReconstructionMap: non-finite fixed_value at variable " + std::to_string(j);
                return cert;
            }
            x[j] = vm.fixed_value;
        }
    }

    // 2. Validate constraint mappings and initialize dual vector y at terminal state S_T
    std::vector<double> y(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        const auto& cm = constraint_mappings_[i];
        if (cm.original_constr_id != static_cast<model::ConstrId>(i)) {
            cert.valid = false;
            cert.diagnostic =
                "ReconstructionMap: corrupted constraint mapping at index " + std::to_string(i);
            return cert;
        }
        if (cm.status == ConstraintMappingStatus::RETAINED) {
            if (cm.reduced_constr_id >= reduced_dual.size()) {
                cert.valid = false;
                cert.diagnostic =
                    "ReconstructionMap: reduced ConstrId out of range at row " + std::to_string(i);
                return cert;
            }
            y[i] = reduced_dual[cm.reduced_constr_id];
        } else {
            y[i] = 0.0;
        }
    }

    // 3. Initialize reduced costs r at terminal state S_T for retained variables
    std::vector<double> r = original_obj_coeffs_;
    for (std::size_t i = 0; i < m; ++i) {
        if (constraint_mappings_[i].status == ConstraintMappingStatus::RETAINED && y[i] != 0.0) {
            for (std::size_t p = orig_row_ptr_[i]; p < orig_row_ptr_[i + 1]; ++p) {
                r[orig_col_idx_[p]] -= orig_values_[p] * y[i];
            }
        }
    }

    // Track any discrepancy in caller-supplied reduced_rc so fault injection is preserved
    std::vector<double> rc_input_delta(n, 0.0);
    if (!reduced_rc.empty()) {
        for (std::size_t j = 0; j < n; ++j) {
            const auto& vm = variable_mappings_[j];
            if (vm.status == VariableMappingStatus::RETAINED &&
                vm.reduced_var_id < reduced_rc.size()) {
                double diff = reduced_rc[vm.reduced_var_id] - r[j];
                if (std::abs(diff) > 1e-8) {
                    rc_input_delta[j] = diff;
                }
            }
        }
    }

    // 4. Reverse Compositional Postsolve Walk (t = T - 1 down to 0)
    const double sigma = (original_sense_ == model::ObjectiveSense::MAXIMIZE) ? -1.0 : 1.0;
    const double prim_tol = tolerances.primal_feasibility_tol;
    const double dual_tol = tolerances.optimality_tol;
    const double piv_tol = tolerances.pivot_tol;

    for (std::size_t idx = ledger_.size(); idx > 0; --idx) {
        const auto& entry = ledger_[idx - 1];

        switch (entry.type) {
            case PresolveTransformType::FIXED_VARIABLE: {
                if (entry.target_var_id >= n) {
                    cert.valid = false;
                    cert.diagnostic =
                        "ReconstructionMap: ledger FIXED_VARIABLE VarId out of range.";
                    return cert;
                }
                const std::size_t j = entry.target_var_id;
                x[j] = entry.fixed_value;
                double yTa = 0.0;
                for (const auto& [cid, coeff] : entry.active_terms) {
                    if (cid >= m) {
                        cert.valid = false;
                        cert.diagnostic =
                            "ReconstructionMap: ledger FIXED_VARIABLE ConstrId out of range.";
                        return cert;
                    }
                    yTa += coeff * y[cid];
                }
                r[j] = entry.obj_coeff - yTa;
                break;
            }

            case PresolveTransformType::EMPTY_COLUMN: {
                if (entry.target_var_id >= n) {
                    cert.valid = false;
                    cert.diagnostic = "ReconstructionMap: ledger EMPTY_COLUMN VarId out of range.";
                    return cert;
                }
                const std::size_t j = entry.target_var_id;
                x[j] = entry.fixed_value;
                r[j] = entry.obj_coeff;
                break;
            }

            case PresolveTransformType::EMPTY_ROW:
            case PresolveTransformType::REDUNDANT_ROW: {
                if (entry.target_constr_id >= m) {
                    cert.valid = false;
                    cert.diagnostic = "ReconstructionMap: ledger row index out of range.";
                    return cert;
                }
                // Initialized to 0.0; remains 0.0 unless an earlier bound propagation step
                // from this same row transfers a multiplier onto y[entry.target_constr_id].
                break;
            }

            case PresolveTransformType::SINGLETON_ROW: {
                if (entry.target_constr_id >= m || entry.target_var_id >= n) {
                    cert.valid = false;
                    cert.diagnostic = "ReconstructionMap: ledger SINGLETON_ROW index out of range.";
                    return cert;
                }
                const std::size_t i = entry.target_constr_id;
                const std::size_t j = entry.target_var_id;
                const double a = entry.target_coeff;
                if (std::abs(a) <= piv_tol) {
                    break;
                }

                const bool tightened_ub = (entry.new_ub < entry.old_ub - 1e-12);
                const bool tightened_lb = (entry.new_lb > entry.old_lb + 1e-12);

                const bool needs_ub_mult = tightened_ub && (sigma * r[j] < -dual_tol) &&
                                           (x[j] >= entry.new_ub - prim_tol * 10.0);
                const bool needs_lb_mult = tightened_lb && (sigma * r[j] > dual_tol) &&
                                           (x[j] <= entry.new_lb + prim_tol * 10.0);
                const bool is_eq_singleton = (entry.row_sense == model::ConstraintSense::EQUAL) &&
                                             (std::abs(r[j]) > dual_tol);

                if (needs_ub_mult || needs_lb_mult || is_eq_singleton) {
                    const double delta_y = r[j] / a;
                    y[i] += delta_y;
                    r[j] = 0.0;
                }
                break;
            }

            case PresolveTransformType::BOUND_PROPAGATION: {
                if (entry.target_constr_id >= m || entry.target_var_id >= n) {
                    cert.valid = false;
                    cert.diagnostic =
                        "ReconstructionMap: ledger BOUND_PROPAGATION index out of range.";
                    return cert;
                }
                const std::size_t i = entry.target_constr_id;
                const std::size_t j = entry.target_var_id;
                const double a = entry.target_coeff;
                if (std::abs(a) <= piv_tol) {
                    break;
                }

                const bool tightened_ub = (entry.new_ub < entry.old_ub - 1e-12);
                const bool tightened_lb = (entry.new_lb > entry.old_lb + 1e-12);

                const bool needs_ub_mult = tightened_ub && (sigma * r[j] < -dual_tol) &&
                                           (x[j] >= entry.new_ub - prim_tol * 10.0);
                const bool needs_lb_mult = tightened_lb && (sigma * r[j] > dual_tol) &&
                                           (x[j] <= entry.new_lb + prim_tol * 10.0);

                if (needs_ub_mult || needs_lb_mult) {
                    // Verify row i is tight at x so complementary slackness holds
                    double row_act = 0.0;
                    for (const auto& [vid, coeff] : entry.active_terms) {
                        if (vid >= n) {
                            cert.valid = false;
                            cert.diagnostic =
                                "ReconstructionMap: ledger BOUND_PROPAGATION VarId out of range.";
                            return cert;
                        }
                        row_act += coeff * x[vid];
                    }
                    if (std::abs(entry.row_rhs - row_act) <= prim_tol * 100.0) {
                        const double delta_y = r[j] / a;
                        y[i] += delta_y;
                        for (const auto& [vid, coeff] : entry.active_terms) {
                            r[vid] -= coeff * delta_y;
                        }
                        r[j] = 0.0;
                    }
                }
                break;
            }
        }
    }

    // 5. Recompute canonical original-model reduced costs r = c_orig - A_orig^T * y
    std::vector<double> rc_orig = restore_reduced_costs(y);
    for (std::size_t j = 0; j < n; ++j) {
        rc_orig[j] += rc_input_delta[j];
    }

    cert.valid = true;
    cert.primal_solution = std::move(x);
    cert.dual_multipliers = std::move(y);
    cert.reduced_costs = std::move(rc_orig);
    cert.objective_value = restore_objective(cert.primal_solution);
    return cert;
}

double ReconstructionMap::restore_objective_value(double reduced_objective_value) noexcept {
    // When solution is substituted and reconstructed, the total objective value
    // (terms + offset) is mathematically preserved.
    return reduced_objective_value;
}

PresolveResult Presolver::apply(const model::CanonicalModel& original_model,
                                const PresolveOptions& options) {
    const auto start_time = std::chrono::steady_clock::now();

    PresolveResult result;
    result.stats.original_variables = original_model.num_variables();
    result.stats.original_constraints = original_model.num_constraints();
    result.reconstruction_map =
        ReconstructionMap(original_model.num_variables(), original_model.num_constraints());
    result.reconstruction_map.initialize_model_context(original_model);

    const double prim_tol = options.tolerances.primal_feasibility_tol;
    const double piv_tol = options.tolerances.pivot_tol;
    const double int_tol = options.tolerances.integrality_tol;

    // Build independent working structures to ensure original_model immutability
    std::vector<WorkingVariable> vars(original_model.num_variables());
    for (std::size_t j = 0; j < vars.size(); ++j) {
        const auto& v = original_model.variable(static_cast<model::VarId>(j));
        vars[j].id = v.id;
        vars[j].name = v.name;
        vars[j].type = v.type;
        vars[j].lb = v.lower_bound;
        vars[j].ub = v.upper_bound;
        vars[j].is_fixed = false;
        vars[j].fixed_val = 0.0;
        vars[j].is_deleted = false;
    }

    std::vector<WorkingConstraint> constrs(original_model.num_constraints());
    std::vector<ConstraintMappingStatus> row_status(original_model.num_constraints(),
                                                    ConstraintMappingStatus::RETAINED);
    for (std::size_t i = 0; i < constrs.size(); ++i) {
        const auto& c = original_model.constraint(static_cast<model::ConstrId>(i));
        constrs[i].id = c.id;
        constrs[i].name = c.name;
        constrs[i].sense = c.sense;
        constrs[i].rhs = c.rhs;
        constrs[i].is_deleted = false;
        constrs[i].terms.reserve(c.expression.terms().size());
        for (const auto& t : c.expression.terms()) {
            if (std::abs(t.coefficient) > piv_tol) {
                constrs[i].terms.push_back({.var_id = t.var_id, .coeff = t.coefficient});
            }
        }
    }

    WorkingObjective obj;
    const auto& orig_obj = original_model.objective();
    obj.sense = orig_obj.sense;
    obj.offset = orig_obj.offset;
    for (const auto& t : orig_obj.linear_terms.terms()) {
        if (std::abs(t.coefficient) > piv_tol) {
            obj.linear_terms.push_back({.var_id = t.var_id, .coeff = t.coefficient});
        }
    }
    for (const auto& q : orig_obj.quadratic_terms.terms()) {
        if (std::abs(q.coefficient) > piv_tol) {
            obj.quadratic_terms.push_back(
                {.var1 = q.var1_id, .var2 = q.var2_id, .coeff = q.coefficient});
        }
    }

    // Deterministic Multi-Pass Loop
    for (std::size_t pass = 0; pass < options.max_passes; ++pass) {
        bool changed = false;

        // --- Reduction A: Fixed-Variable Substitution ---
        if (options.enable_fixed_variable_substitution) {
            for (std::size_t j = 0; j < vars.size(); ++j) {
                auto& v = vars[j];
                if (v.is_deleted || v.is_fixed) {
                    continue;
                }

                // Check for bound equality
                if (v.lb == v.ub || (v.ub - v.lb <= prim_tol && v.lb <= v.ub)) {
                    double fix_val = (v.lb == v.ub) ? v.lb : (0.5 * (v.lb + v.ub));

                    // Integrality verification
                    if (v.type == model::VariableType::INTEGER ||
                        v.type == model::VariableType::BINARY) {
                        double rounded = std::round(fix_val);
                        if (std::abs(fix_val - rounded) > int_tol) {
                            result.status = core::SolverStatus::INFEASIBLE;
                            result.message = "Fixed variable " + v.name + " value " +
                                             std::to_string(fix_val) +
                                             " violates integrality tolerance";
                            return result;
                        }
                        fix_val = rounded;
                    }

                    v.is_fixed = true;
                    v.fixed_val = fix_val;
                    v.is_deleted = true;

                    result.reconstruction_map.set_mapping(
                        static_cast<model::VarId>(j),
                        VariableMapping{.original_var_id = static_cast<model::VarId>(j),
                                        .status = VariableMappingStatus::FIXED_VALUE,
                                        .reduced_var_id = model::INVALID_VAR_ID,
                                        .fixed_value = fix_val});

                    result.stats.fixed_variables++;
                    result.stats.variables_removed++;
                    changed = true;

                    double current_obj_coeff = 0.0;
                    // Substitute into linear objective
                    for (auto& lt : obj.linear_terms) {
                        if (lt.var_id == static_cast<model::VarId>(j)) {
                            current_obj_coeff += lt.coeff;
                            obj.offset += lt.coeff * fix_val;
                            lt.coeff = 0.0;
                            result.stats.coefficients_eliminated++;
                        }
                    }

                    // Substitute into quadratic objective
                    for (auto& qt : obj.quadratic_terms) {
                        if (qt.var1 == static_cast<model::VarId>(j) &&
                            qt.var2 == static_cast<model::VarId>(j)) {
                            obj.offset += qt.coeff * fix_val * fix_val;
                            qt.coeff = 0.0;
                            result.stats.coefficients_eliminated++;
                        } else if (qt.var1 == static_cast<model::VarId>(j)) {
                            double extra_linear = qt.coeff * fix_val;
                            bool found = false;
                            for (auto& lt : obj.linear_terms) {
                                if (lt.var_id == qt.var2) {
                                    lt.coeff += extra_linear;
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                obj.linear_terms.push_back(
                                    {.var_id = qt.var2, .coeff = extra_linear});
                            }
                            qt.coeff = 0.0;
                            result.stats.coefficients_eliminated++;
                        } else if (qt.var2 == static_cast<model::VarId>(j)) {
                            double extra_linear = qt.coeff * fix_val;
                            bool found = false;
                            for (auto& lt : obj.linear_terms) {
                                if (lt.var_id == qt.var1) {
                                    lt.coeff += extra_linear;
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                obj.linear_terms.push_back(
                                    {.var_id = qt.var1, .coeff = extra_linear});
                            }
                            qt.coeff = 0.0;
                            result.stats.coefficients_eliminated++;
                        }
                    }

                    PresolveTransformEntry ledger_entry;
                    ledger_entry.pass_index = pass;
                    ledger_entry.type = PresolveTransformType::FIXED_VARIABLE;
                    ledger_entry.target_var_id = static_cast<model::VarId>(j);
                    ledger_entry.old_lb = v.lb;
                    ledger_entry.old_ub = v.ub;
                    ledger_entry.new_lb = fix_val;
                    ledger_entry.new_ub = fix_val;
                    ledger_entry.fixed_value = fix_val;
                    ledger_entry.obj_coeff = current_obj_coeff;

                    // Substitute into constraints
                    for (auto& c : constrs) {
                        if (c.is_deleted) {
                            continue;
                        }
                        for (auto& t : c.terms) {
                            if (t.var_id == static_cast<model::VarId>(j)) {
                                ledger_entry.active_terms.push_back({c.id, t.coeff});
                                c.rhs -= t.coeff * fix_val;
                                t.coeff = 0.0;
                                result.stats.coefficients_eliminated++;
                            }
                        }
                        std::erase_if(c.terms, [piv_tol](const WorkingTerm& term) {
                            return std::abs(term.coeff) <= piv_tol;
                        });
                    }

                    result.reconstruction_map.add_ledger_entry(std::move(ledger_entry));
                }
            }
        }

        // --- Reduction B: Empty Row Analysis & Removal ---
        if (options.enable_empty_row_removal) {
            for (auto& c : constrs) {
                if (c.is_deleted) {
                    continue;
                }
                if (c.terms.empty()) {
                    bool feasible = false;
                    switch (c.sense) {
                        case model::ConstraintSense::LESS_EQUAL:
                            feasible = (c.rhs >= -prim_tol);
                            break;
                        case model::ConstraintSense::GREATER_EQUAL:
                            feasible = (c.rhs <= prim_tol);
                            break;
                        case model::ConstraintSense::EQUAL:
                            feasible = (std::abs(c.rhs) <= prim_tol);
                            break;
                    }
                    if (!feasible) {
                        result.status = core::SolverStatus::INFEASIBLE;
                        result.message =
                            "Empty constraint " + c.name + " is mathematically impossible with 0 " +
                            std::string(model::to_string(c.sense)) + " " + std::to_string(c.rhs);
                        return result;
                    }
                    c.is_deleted = true;
                    row_status[c.id] = ConstraintMappingStatus::REMOVED_EMPTY;
                    result.stats.rows_removed++;
                    result.stats.empty_rows_removed++;
                    changed = true;

                    PresolveTransformEntry ledger_entry;
                    ledger_entry.pass_index = pass;
                    ledger_entry.type = PresolveTransformType::EMPTY_ROW;
                    ledger_entry.target_constr_id = c.id;
                    ledger_entry.row_sense = c.sense;
                    ledger_entry.row_rhs = c.rhs;
                    result.reconstruction_map.add_ledger_entry(std::move(ledger_entry));
                }
            }
        }

        // --- Reduction D: Singleton Row Bound Tightening ---
        if (options.enable_singleton_bound_tightening) {
            for (auto& c : constrs) {
                if (c.is_deleted) {
                    continue;
                }
                if (c.terms.size() == 1) {
                    const auto& t = c.terms[0];
                    if (std::abs(t.coeff) < piv_tol) {
                        continue;
                    }

                    auto& v = vars[t.var_id];
                    const double old_lb = v.lb;
                    const double old_ub = v.ub;
                    double bound_val = c.rhs / t.coeff;

                    if (t.coeff > 0.0) {
                        if (c.sense == model::ConstraintSense::LESS_EQUAL) {
                            if (bound_val < v.ub - 1e-12) {
                                v.ub = bound_val;
                                result.stats.bound_tightenings++;
                            }
                        } else if (c.sense == model::ConstraintSense::GREATER_EQUAL) {
                            if (bound_val > v.lb + 1e-12) {
                                v.lb = bound_val;
                                result.stats.bound_tightenings++;
                            }
                        } else {  // EQUAL
                            v.ub = std::min(v.ub, bound_val);
                            v.lb = std::max(v.lb, bound_val);
                            result.stats.bound_tightenings++;
                        }
                    } else {  // t.coeff < 0.0
                        if (c.sense == model::ConstraintSense::LESS_EQUAL) {
                            if (bound_val > v.lb + 1e-12) {
                                v.lb = bound_val;
                                result.stats.bound_tightenings++;
                            }
                        } else if (c.sense == model::ConstraintSense::GREATER_EQUAL) {
                            if (bound_val < v.ub - 1e-12) {
                                v.ub = bound_val;
                                result.stats.bound_tightenings++;
                            }
                        } else {  // EQUAL
                            v.ub = std::min(v.ub, bound_val);
                            v.lb = std::max(v.lb, bound_val);
                            result.stats.bound_tightenings++;
                        }
                    }

                    if (v.lb > v.ub + prim_tol) {
                        result.status = core::SolverStatus::INFEASIBLE;
                        result.message = "Contradictory bounds on variable " + v.name +
                                         " after singleton tightening: lb=" + std::to_string(v.lb) +
                                         " > ub=" + std::to_string(v.ub);
                        return result;
                    }

                    c.is_deleted = true;
                    row_status[c.id] = ConstraintMappingStatus::REMOVED_SINGLETON;
                    result.stats.rows_removed++;
                    result.stats.singleton_reductions++;
                    changed = true;

                    PresolveTransformEntry ledger_entry;
                    ledger_entry.pass_index = pass;
                    ledger_entry.type = PresolveTransformType::SINGLETON_ROW;
                    ledger_entry.target_constr_id = c.id;
                    ledger_entry.target_var_id = t.var_id;
                    ledger_entry.row_sense = c.sense;
                    ledger_entry.row_rhs = c.rhs;
                    ledger_entry.target_coeff = t.coeff;
                    ledger_entry.old_lb = old_lb;
                    ledger_entry.old_ub = old_ub;
                    ledger_entry.new_lb = v.lb;
                    ledger_entry.new_ub = v.ub;
                    result.reconstruction_map.add_ledger_entry(std::move(ledger_entry));
                }
            }
        }

        // --- Reduction E & F: Activity Range, Redundancy, and Bound Propagation ---
        if (options.enable_redundant_row_removal || options.enable_bound_propagation) {
            for (auto& c : constrs) {
                if (c.is_deleted) {
                    continue;
                }

                // Compute conservative row activity bounds
                double min_act = 0.0;
                double max_act = 0.0;
                std::size_t unb_min = 0;
                std::size_t unb_max = 0;

                for (const auto& t : c.terms) {
                    const auto& v = vars[t.var_id];
                    if (t.coeff > 0.0) {
                        if (std::isinf(v.lb)) {
                            unb_min++;
                        } else {
                            min_act += t.coeff * v.lb;
                        }
                        if (std::isinf(v.ub)) {
                            unb_max++;
                        } else {
                            max_act += t.coeff * v.ub;
                        }
                    } else {
                        if (std::isinf(v.ub)) {
                            unb_min++;
                        } else {
                            min_act += t.coeff * v.ub;
                        }
                        if (std::isinf(v.lb)) {
                            unb_max++;
                        } else {
                            max_act += t.coeff * v.lb;
                        }
                    }
                }

                // Infeasibility check
                if (unb_min == 0) {
                    if ((c.sense == model::ConstraintSense::LESS_EQUAL &&
                         min_act > c.rhs + prim_tol) ||
                        (c.sense == model::ConstraintSense::EQUAL && min_act > c.rhs + prim_tol)) {
                        result.status = core::SolverStatus::INFEASIBLE;
                        result.message =
                            "Constraint " + c.name +
                            " min activity exceeds RHS: min=" + std::to_string(min_act) +
                            " > rhs=" + std::to_string(c.rhs);
                        return result;
                    }
                }
                if (unb_max == 0) {
                    if ((c.sense == model::ConstraintSense::GREATER_EQUAL &&
                         max_act < c.rhs - prim_tol) ||
                        (c.sense == model::ConstraintSense::EQUAL && max_act < c.rhs - prim_tol)) {
                        result.status = core::SolverStatus::INFEASIBLE;
                        result.message =
                            "Constraint " + c.name +
                            " max activity falls below RHS: max=" + std::to_string(max_act) +
                            " < rhs=" + std::to_string(c.rhs);
                        return result;
                    }
                }

                // Redundant row removal
                if (options.enable_redundant_row_removal) {
                    if (c.sense == model::ConstraintSense::LESS_EQUAL && unb_max == 0 &&
                        max_act <= c.rhs + prim_tol) {
                        c.is_deleted = true;
                        row_status[c.id] = ConstraintMappingStatus::REMOVED_REDUNDANT;
                        result.stats.rows_removed++;
                        result.stats.redundant_rows_removed++;
                        changed = true;

                        PresolveTransformEntry ledger_entry;
                        ledger_entry.pass_index = pass;
                        ledger_entry.type = PresolveTransformType::REDUNDANT_ROW;
                        ledger_entry.target_constr_id = c.id;
                        ledger_entry.row_sense = c.sense;
                        ledger_entry.row_rhs = c.rhs;
                        result.reconstruction_map.add_ledger_entry(std::move(ledger_entry));
                        continue;
                    }
                    if (c.sense == model::ConstraintSense::GREATER_EQUAL && unb_min == 0 &&
                        min_act >= c.rhs - prim_tol) {
                        c.is_deleted = true;
                        row_status[c.id] = ConstraintMappingStatus::REMOVED_REDUNDANT;
                        result.stats.rows_removed++;
                        result.stats.redundant_rows_removed++;
                        changed = true;

                        PresolveTransformEntry ledger_entry;
                        ledger_entry.pass_index = pass;
                        ledger_entry.type = PresolveTransformType::REDUNDANT_ROW;
                        ledger_entry.target_constr_id = c.id;
                        ledger_entry.row_sense = c.sense;
                        ledger_entry.row_rhs = c.rhs;
                        result.reconstruction_map.add_ledger_entry(std::move(ledger_entry));
                        continue;
                    }
                }

                // General bound propagation from multi-variable rows
                if (options.enable_bound_propagation && c.terms.size() > 1) {
                    auto make_prop_entry = [&](model::VarId vid, double coeff, double old_lb,
                                               double old_ub, double new_lb, double new_ub) {
                        PresolveTransformEntry entry;
                        entry.pass_index = pass;
                        entry.type = PresolveTransformType::BOUND_PROPAGATION;
                        entry.target_constr_id = c.id;
                        entry.target_var_id = vid;
                        entry.row_sense = c.sense;
                        entry.row_rhs = c.rhs;
                        entry.target_coeff = coeff;
                        entry.old_lb = old_lb;
                        entry.old_ub = old_ub;
                        entry.new_lb = new_lb;
                        entry.new_ub = new_ub;
                        entry.active_terms.reserve(c.terms.size());
                        for (const auto& rt : c.terms) {
                            entry.active_terms.push_back({rt.var_id, rt.coeff});
                        }
                        result.reconstruction_map.add_ledger_entry(std::move(entry));
                    };

                    if (c.sense == model::ConstraintSense::LESS_EQUAL ||
                        c.sense == model::ConstraintSense::EQUAL) {
                        for (const auto& t : c.terms) {
                            auto& v = vars[t.var_id];
                            bool term_unb = (t.coeff > 0.0 && std::isinf(v.lb)) ||
                                            (t.coeff < 0.0 && std::isinf(v.ub));
                            std::size_t remaining_unb = unb_min - (term_unb ? 1 : 0);
                            if (remaining_unb == 0) {
                                double term_contrib = 0.0;
                                if (!term_unb) {
                                    term_contrib =
                                        (t.coeff > 0.0) ? (t.coeff * v.lb) : (t.coeff * v.ub);
                                }
                                double other_min = min_act - term_contrib;
                                double avail = c.rhs - other_min;
                                if (t.coeff > 0.0) {
                                    double implied_ub = avail / t.coeff;
                                    if (implied_ub < v.ub - 1e-12) {
                                        double old_ub = v.ub;
                                        v.ub = implied_ub;
                                        result.stats.bound_tightenings++;
                                        changed = true;
                                        make_prop_entry(t.var_id, t.coeff, v.lb, old_ub, v.lb,
                                                        v.ub);
                                        if (v.lb > v.ub + prim_tol) {
                                            result.status = core::SolverStatus::INFEASIBLE;
                                            result.message = "Bound contradiction on " + v.name +
                                                             " during propagation";
                                            return result;
                                        }
                                    }
                                } else if (t.coeff < 0.0) {
                                    double implied_lb = avail / t.coeff;
                                    if (implied_lb > v.lb + 1e-12) {
                                        double old_lb = v.lb;
                                        v.lb = implied_lb;
                                        result.stats.bound_tightenings++;
                                        changed = true;
                                        make_prop_entry(t.var_id, t.coeff, old_lb, v.ub, v.lb,
                                                        v.ub);
                                        if (v.lb > v.ub + prim_tol) {
                                            result.status = core::SolverStatus::INFEASIBLE;
                                            result.message = "Bound contradiction on " + v.name +
                                                             " during propagation";
                                            return result;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if (c.sense == model::ConstraintSense::GREATER_EQUAL ||
                        c.sense == model::ConstraintSense::EQUAL) {
                        for (const auto& t : c.terms) {
                            auto& v = vars[t.var_id];
                            bool term_unb = (t.coeff > 0.0 && std::isinf(v.ub)) ||
                                            (t.coeff < 0.0 && std::isinf(v.lb));
                            std::size_t remaining_unb = unb_max - (term_unb ? 1 : 0);
                            if (remaining_unb == 0) {
                                double term_contrib = 0.0;
                                if (!term_unb) {
                                    term_contrib =
                                        (t.coeff > 0.0) ? (t.coeff * v.ub) : (t.coeff * v.lb);
                                }
                                double other_max = max_act - term_contrib;
                                double avail = c.rhs - other_max;
                                if (t.coeff > 0.0) {
                                    double implied_lb = avail / t.coeff;
                                    if (implied_lb > v.lb + 1e-12) {
                                        double old_lb = v.lb;
                                        v.lb = implied_lb;
                                        result.stats.bound_tightenings++;
                                        changed = true;
                                        make_prop_entry(t.var_id, t.coeff, old_lb, v.ub, v.lb,
                                                        v.ub);
                                        if (v.lb > v.ub + prim_tol) {
                                            result.status = core::SolverStatus::INFEASIBLE;
                                            result.message = "Bound contradiction on " + v.name +
                                                             " during propagation";
                                            return result;
                                        }
                                    }
                                } else if (t.coeff < 0.0) {
                                    double implied_ub = avail / t.coeff;
                                    if (implied_ub < v.ub - 1e-12) {
                                        double old_ub = v.ub;
                                        v.ub = implied_ub;
                                        result.stats.bound_tightenings++;
                                        changed = true;
                                        make_prop_entry(t.var_id, t.coeff, v.lb, old_ub, v.lb,
                                                        v.ub);
                                        if (v.lb > v.ub + prim_tol) {
                                            result.status = core::SolverStatus::INFEASIBLE;
                                            result.message = "Bound contradiction on " + v.name +
                                                             " during propagation";
                                            return result;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // --- Reduction C: Empty Column / Unconstrained Variable Analysis ---
        if (options.enable_empty_column_removal) {
            std::vector<std::size_t> var_usage(vars.size(), 0);
            for (const auto& c : constrs) {
                if (c.is_deleted) {
                    continue;
                }
                for (const auto& t : c.terms) {
                    var_usage[t.var_id]++;
                }
            }

            for (std::size_t j = 0; j < vars.size(); ++j) {
                auto& v = vars[j];
                if (v.is_deleted || v.is_fixed || var_usage[j] > 0) {
                    continue;
                }

                // Check quadratic participation
                bool has_quad = false;
                for (const auto& qt : obj.quadratic_terms) {
                    if (qt.coeff != 0.0 && (qt.var1 == j || qt.var2 == j)) {
                        has_quad = true;
                        break;
                    }
                }
                if (has_quad) {
                    continue;  // Retain unconstrained quadratic variables safely
                }

                double obj_c = 0.0;
                for (const auto& lt : obj.linear_terms) {
                    if (lt.var_id == j) {
                        obj_c += lt.coeff;
                    }
                }

                if (std::abs(obj_c) <= piv_tol) {
                    // Cost is zero, not in constraints -> fix to feasible point
                    double fix_val = 0.0;
                    if (v.lb <= 0.0 && v.ub >= 0.0) {
                        fix_val = 0.0;
                    } else if (!std::isinf(v.lb)) {
                        fix_val = v.lb;
                    } else if (!std::isinf(v.ub)) {
                        fix_val = v.ub;
                    }
                    if (v.type == model::VariableType::INTEGER ||
                        v.type == model::VariableType::BINARY) {
                        fix_val = std::round(fix_val);
                    }
                    v.is_fixed = true;
                    v.fixed_val = fix_val;
                    v.is_deleted = true;
                    obj.offset += obj_c * fix_val;

                    result.reconstruction_map.set_mapping(
                        static_cast<model::VarId>(j),
                        VariableMapping{.original_var_id = static_cast<model::VarId>(j),
                                        .status = VariableMappingStatus::UNCONSTRAINED_REMOVED,
                                        .reduced_var_id = model::INVALID_VAR_ID,
                                        .fixed_value = fix_val});

                    PresolveTransformEntry ledger_entry;
                    ledger_entry.pass_index = pass;
                    ledger_entry.type = PresolveTransformType::EMPTY_COLUMN;
                    ledger_entry.target_var_id = static_cast<model::VarId>(j);
                    ledger_entry.old_lb = v.lb;
                    ledger_entry.old_ub = v.ub;
                    ledger_entry.new_lb = fix_val;
                    ledger_entry.new_ub = fix_val;
                    ledger_entry.fixed_value = fix_val;
                    ledger_entry.obj_coeff = obj_c;
                    result.reconstruction_map.add_ledger_entry(std::move(ledger_entry));

                    result.stats.variables_removed++;
                    changed = true;
                } else {
                    // Nonzero cost, unconstrained: check unboundedness
                    bool is_min = (obj.sense == model::ObjectiveSense::MINIMIZE);
                    double eff_c = is_min ? obj_c : -obj_c;

                    if (eff_c > 0.0) {
                        if (std::isinf(v.lb)) {
                            result.status = core::SolverStatus::UNBOUNDED;
                            result.message = "Unconstrained variable " + v.name +
                                             " can improve objective towards -infinity";
                            return result;
                        }
                        double fix_val = v.lb;
                        v.is_fixed = true;
                        v.fixed_val = fix_val;
                        v.is_deleted = true;
                        obj.offset += obj_c * fix_val;
                        result.reconstruction_map.set_mapping(
                            static_cast<model::VarId>(j),
                            VariableMapping{.original_var_id = static_cast<model::VarId>(j),
                                            .status = VariableMappingStatus::UNCONSTRAINED_REMOVED,
                                            .reduced_var_id = model::INVALID_VAR_ID,
                                            .fixed_value = fix_val});

                        PresolveTransformEntry ledger_entry;
                        ledger_entry.pass_index = pass;
                        ledger_entry.type = PresolveTransformType::EMPTY_COLUMN;
                        ledger_entry.target_var_id = static_cast<model::VarId>(j);
                        ledger_entry.old_lb = v.lb;
                        ledger_entry.old_ub = v.ub;
                        ledger_entry.new_lb = fix_val;
                        ledger_entry.new_ub = fix_val;
                        ledger_entry.fixed_value = fix_val;
                        ledger_entry.obj_coeff = obj_c;
                        result.reconstruction_map.add_ledger_entry(std::move(ledger_entry));

                        result.stats.variables_removed++;
                        changed = true;
                    } else {  // eff_c < 0.0
                        if (std::isinf(v.ub)) {
                            result.status = core::SolverStatus::UNBOUNDED;
                            result.message = "Unconstrained variable " + v.name +
                                             " can improve objective towards +infinity";
                            return result;
                        }
                        double fix_val = v.ub;
                        v.is_fixed = true;
                        v.fixed_val = fix_val;
                        v.is_deleted = true;
                        obj.offset += obj_c * fix_val;
                        result.reconstruction_map.set_mapping(
                            static_cast<model::VarId>(j),
                            VariableMapping{.original_var_id = static_cast<model::VarId>(j),
                                            .status = VariableMappingStatus::UNCONSTRAINED_REMOVED,
                                            .reduced_var_id = model::INVALID_VAR_ID,
                                            .fixed_value = fix_val});

                        PresolveTransformEntry ledger_entry;
                        ledger_entry.pass_index = pass;
                        ledger_entry.type = PresolveTransformType::EMPTY_COLUMN;
                        ledger_entry.target_var_id = static_cast<model::VarId>(j);
                        ledger_entry.old_lb = v.lb;
                        ledger_entry.old_ub = v.ub;
                        ledger_entry.new_lb = fix_val;
                        ledger_entry.new_ub = fix_val;
                        ledger_entry.fixed_value = fix_val;
                        ledger_entry.obj_coeff = obj_c;
                        result.reconstruction_map.add_ledger_entry(std::move(ledger_entry));

                        result.stats.variables_removed++;
                        changed = true;
                    }
                }
            }
        }

        result.stats.passes_performed++;
        if (!changed) {
            break;
        }
    }

    // --- Build Compact Reduced CanonicalModel ---
    auto reduced = std::make_unique<model::CanonicalModel>(original_model.name() + "_presolved");

    std::vector<model::VarId> working_to_reduced(vars.size(), model::INVALID_VAR_ID);
    model::VarId next_red_var_id = 0;

    for (std::size_t j = 0; j < vars.size(); ++j) {
        const auto& v = vars[j];
        if (!v.is_deleted) {
            working_to_reduced[j] = next_red_var_id;
            reduced->add_variable(v.name, v.type, v.lb, v.ub);
            result.reconstruction_map.set_mapping(
                static_cast<model::VarId>(j),
                VariableMapping{.original_var_id = static_cast<model::VarId>(j),
                                .status = VariableMappingStatus::RETAINED,
                                .reduced_var_id = next_red_var_id,
                                .fixed_value = 0.0});
            next_red_var_id++;
        }
    }

    // Set Objective
    reduced->set_objective_sense(obj.sense);
    reduced->set_objective_offset(obj.offset);

    for (const auto& lt : obj.linear_terms) {
        if (std::abs(lt.coeff) > piv_tol && !vars[lt.var_id].is_deleted) {
            reduced->add_objective_linear_term(working_to_reduced[lt.var_id], lt.coeff);
        }
    }

    for (const auto& qt : obj.quadratic_terms) {
        if (std::abs(qt.coeff) > piv_tol && !vars[qt.var1].is_deleted &&
            !vars[qt.var2].is_deleted) {
            reduced->add_objective_quadratic_term(working_to_reduced[qt.var1],
                                                  working_to_reduced[qt.var2], qt.coeff);
        }
    }

    // Add Active Constraints and record exact row mapping
    model::ConstrId next_red_constr_id = 0;
    for (std::size_t i = 0; i < constrs.size(); ++i) {
        const auto& c = constrs[i];
        if (c.is_deleted) {
            result.reconstruction_map.set_constraint_mapping(
                static_cast<model::ConstrId>(i),
                ConstraintMapping{.original_constr_id = static_cast<model::ConstrId>(i),
                                  .status = row_status[i],
                                  .reduced_constr_id = model::INVALID_CONSTR_ID});
            continue;
        }
        model::SparseLinearExpression red_expr;
        for (const auto& t : c.terms) {
            if (std::abs(t.coeff) > piv_tol && !vars[t.var_id].is_deleted) {
                red_expr.add_term(working_to_reduced[t.var_id], t.coeff);
            }
        }
        if (!red_expr.empty()) {
            reduced->add_constraint(c.name, std::move(red_expr), c.sense, c.rhs);
            result.reconstruction_map.set_constraint_mapping(
                static_cast<model::ConstrId>(i),
                ConstraintMapping{.original_constr_id = static_cast<model::ConstrId>(i),
                                  .status = ConstraintMappingStatus::RETAINED,
                                  .reduced_constr_id = next_red_constr_id});
            next_red_constr_id++;
        } else {
            result.reconstruction_map.set_constraint_mapping(
                static_cast<model::ConstrId>(i),
                ConstraintMapping{.original_constr_id = static_cast<model::ConstrId>(i),
                                  .status = ConstraintMappingStatus::REMOVED_EMPTY,
                                  .reduced_constr_id = model::INVALID_CONSTR_ID});
        }
    }

    reduced->canonicalize();

    // Check invariants on reduced model
    try {
        reduced->check_valid();
    } catch (const std::exception& e) {
        throw core::NumericalException(std::string("Presolve produced invalid reduced model: ") +
                                       e.what());
    }

    result.reconstruction_map.set_reduced_dimensions(reduced->num_variables(),
                                                     reduced->num_constraints());
    result.reconstruction_map.set_objective_offset_shift(obj.offset -
                                                         original_model.objective_offset());

    result.stats.reduced_variables = reduced->num_variables();
    result.stats.reduced_constraints = reduced->num_constraints();
    result.stats.variables_removed =
        result.stats.original_variables - result.stats.reduced_variables;
    result.stats.rows_removed =
        result.stats.original_constraints - result.stats.reduced_constraints;
    result.stats.objective_constant_shift = obj.offset - original_model.objective_offset();

    const auto end_time = std::chrono::steady_clock::now();
    result.stats.elapsed_time_seconds =
        std::chrono::duration<double>(end_time - start_time).count();

    result.reduced_model = std::move(reduced);
    result.status = core::SolverStatus::UNKNOWN;  // Ready for downstream solver
    result.message = "Presolve completed successfully";

    return result;
}

}  // namespace kavach::presolve
