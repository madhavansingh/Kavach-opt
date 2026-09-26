#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"

namespace kavach::presolve {

/**
 * @brief Options controlling presolve passes and enabled reduction techniques.
 */
struct PresolveOptions {
    std::size_t max_passes{10};
    core::SolverTolerances tolerances;
    bool enable_fixed_variable_substitution{true};
    bool enable_empty_row_removal{true};
    bool enable_empty_column_removal{true};
    bool enable_singleton_bound_tightening{true};
    bool enable_redundant_row_removal{true};
    bool enable_bound_propagation{true};
    bool log_details{false};
};

/**
 * @brief Detailed statistics describing presolve reductions and execution.
 */
struct PresolveStats {
    std::size_t original_variables{0};
    std::size_t reduced_variables{0};
    std::size_t variables_removed{0};
    std::size_t fixed_variables{0};
    std::size_t original_constraints{0};
    std::size_t reduced_constraints{0};
    std::size_t rows_removed{0};
    std::size_t empty_rows_removed{0};
    std::size_t redundant_rows_removed{0};
    std::size_t bound_tightenings{0};
    std::size_t singleton_reductions{0};
    std::size_t passes_performed{0};
    std::size_t coefficients_eliminated{0};
    double objective_constant_shift{0.0};
    double elapsed_time_seconds{0.0};

    [[nodiscard]] std::string to_string() const;
};

/**
 * @brief Mapping status for an original variable.
 */
enum class VariableMappingStatus : std::uint8_t {
    RETAINED,              ///< Variable retained in reduced model with a new VarId
    FIXED_VALUE,           ///< Variable fixed to a known constant
    UNCONSTRAINED_REMOVED  ///< Variable unconstrained and safely fixed
};

[[nodiscard]] constexpr std::string_view to_string(VariableMappingStatus status) noexcept {
    switch (status) {
        case VariableMappingStatus::RETAINED:
            return "RETAINED";
        case VariableMappingStatus::FIXED_VALUE:
            return "FIXED_VALUE";
        case VariableMappingStatus::UNCONSTRAINED_REMOVED:
            return "UNCONSTRAINED_REMOVED";
    }
    return "UNKNOWN";
}

/**
 * @brief Mapping status for an original constraint row.
 */
enum class ConstraintMappingStatus : std::uint8_t {
    RETAINED,           ///< Constraint retained in reduced model with a new ConstrId
    REMOVED_EMPTY,      ///< Constraint removed after becoming empty
    REMOVED_SINGLETON,  ///< Constraint converted to variable bound tightening and removed
    REMOVED_REDUNDANT   ///< Constraint proven redundant by activity bounds and removed
};

[[nodiscard]] constexpr std::string_view to_string(ConstraintMappingStatus status) noexcept {
    switch (status) {
        case ConstraintMappingStatus::RETAINED:
            return "RETAINED";
        case ConstraintMappingStatus::REMOVED_EMPTY:
            return "REMOVED_EMPTY";
        case ConstraintMappingStatus::REMOVED_SINGLETON:
            return "REMOVED_SINGLETON";
        case ConstraintMappingStatus::REMOVED_REDUNDANT:
            return "REMOVED_REDUNDANT";
    }
    return "UNKNOWN";
}

/**
 * @brief Mapping entry for a single original variable.
 */
struct VariableMapping {
    model::VarId original_var_id{model::INVALID_VAR_ID};
    VariableMappingStatus status{VariableMappingStatus::RETAINED};
    model::VarId reduced_var_id{model::INVALID_VAR_ID};
    double fixed_value{0.0};
};

/**
 * @brief Mapping entry for a single original constraint row.
 */
struct ConstraintMapping {
    model::ConstrId original_constr_id{model::INVALID_CONSTR_ID};
    ConstraintMappingStatus status{ConstraintMappingStatus::RETAINED};
    model::ConstrId reduced_constr_id{model::INVALID_CONSTR_ID};
};

/**
 * @brief Atomic presolve transformation type recorded in the postsolve ledger.
 */
enum class PresolveTransformType : std::uint8_t {
    FIXED_VARIABLE,     ///< Variable fixed to constant and substituted out of rows/objective
    EMPTY_ROW,          ///< Empty row removed
    SINGLETON_ROW,      ///< Singleton row converted into variable bound tightening and removed
    REDUNDANT_ROW,      ///< Redundant row removed via activity bounds
    BOUND_PROPAGATION,  ///< Multi-variable row propagated a tighter bound onto a variable
    EMPTY_COLUMN        ///< Unconstrained column fixed to optimal bound/feasible value and removed
};

[[nodiscard]] constexpr std::string_view to_string(PresolveTransformType type) noexcept {
    switch (type) {
        case PresolveTransformType::FIXED_VARIABLE:
            return "FIXED_VARIABLE";
        case PresolveTransformType::EMPTY_ROW:
            return "EMPTY_ROW";
        case PresolveTransformType::SINGLETON_ROW:
            return "SINGLETON_ROW";
        case PresolveTransformType::REDUNDANT_ROW:
            return "REDUNDANT_ROW";
        case PresolveTransformType::BOUND_PROPAGATION:
            return "BOUND_PROPAGATION";
        case PresolveTransformType::EMPTY_COLUMN:
            return "EMPTY_COLUMN";
    }
    return "UNKNOWN";
}

/**
 * @brief Explicit transformation ledger entry supporting reverse compositional postsolve.
 */
struct PresolveTransformEntry {
    std::size_t step_id{0};
    std::size_t pass_index{0};
    PresolveTransformType type{PresolveTransformType::FIXED_VARIABLE};
    model::VarId target_var_id{model::INVALID_VAR_ID};
    model::ConstrId target_constr_id{model::INVALID_CONSTR_ID};
    model::ConstraintSense row_sense{model::ConstraintSense::LESS_EQUAL};
    double row_rhs{0.0};
    double target_coeff{0.0};
    double old_lb{0.0};
    double old_ub{0.0};
    double new_lb{0.0};
    double new_ub{0.0};
    double fixed_value{0.0};
    double obj_coeff{0.0};
    /// For FIXED_VARIABLE: (constr_id, coeff) pairs where target_var_id was substituted out.
    /// For BOUND_PROPAGATION: (var_id, coeff) pairs active in target_constr_id at tightening time.
    std::vector<std::pair<std::uint32_t, double>> active_terms;
};

/**
 * @brief Complete reconstructed LP certificate in original model coordinates.
 */
struct ReconstructedCertificate {
    bool valid{false};
    std::string diagnostic;
    std::vector<double> primal_solution;
    std::vector<double> dual_multipliers;
    std::vector<double> reduced_costs;
    double objective_value{0.0};
};

/**
 * @brief Mechanism capable of restoring the full original primal-dual certificate after presolve.
 */
class ReconstructionMap {
public:
    ReconstructionMap() = default;
    explicit ReconstructionMap(std::size_t num_original_vars, std::size_t num_original_constrs = 0);

    void initialize_model_context(const model::CanonicalModel& original_model);
    void set_reduced_dimensions(std::size_t num_reduced_vars,
                                std::size_t num_reduced_constrs) noexcept {
        num_reduced_vars_ = num_reduced_vars;
        num_reduced_constrs_ = num_reduced_constrs;
        has_reduced_dimensions_ = true;
    }

    void set_mapping(model::VarId original_id, VariableMapping mapping);
    [[nodiscard]] const VariableMapping& mapping(model::VarId original_id) const;

    void set_constraint_mapping(model::ConstrId original_id, ConstraintMapping mapping);
    [[nodiscard]] const ConstraintMapping& constraint_mapping(model::ConstrId original_id) const;

    [[nodiscard]] std::size_t num_original_variables() const noexcept {
        return variable_mappings_.size();
    }
    [[nodiscard]] std::size_t num_original_constraints() const noexcept {
        return constraint_mappings_.size();
    }
    [[nodiscard]] std::size_t num_reduced_variables() const noexcept { return num_reduced_vars_; }
    [[nodiscard]] std::size_t num_reduced_constraints() const noexcept {
        return num_reduced_constrs_;
    }

    [[nodiscard]] const std::vector<VariableMapping>& mappings() const noexcept {
        return variable_mappings_;
    }
    [[nodiscard]] const std::vector<ConstraintMapping>& constraint_mappings() const noexcept {
        return constraint_mappings_;
    }

    void add_ledger_entry(PresolveTransformEntry entry);
    [[nodiscard]] const std::vector<PresolveTransformEntry>& ledger() const noexcept {
        return ledger_;
    }
    [[nodiscard]] std::vector<PresolveTransformEntry>& mutable_ledger() noexcept { return ledger_; }

    void set_objective_offset_shift(double shift) noexcept { objective_offset_shift_ = shift; }
    [[nodiscard]] double objective_offset_shift() const noexcept { return objective_offset_shift_; }

    /**
     * @brief Restores a full primal solution in the original variable space from a reduced
     * solution.
     */
    [[nodiscard]] std::vector<double> restore_solution(
        const std::vector<double>& reduced_solution) const;

    /**
     * @brief Restores a full primal solution in the original variable space.
     */
    [[nodiscard]] std::vector<double> restore_primal(
        const std::vector<double>& reduced_primal) const;

    /**
     * @brief Restores original constraint dual multipliers from reduced certificate vectors.
     */
    [[nodiscard]] std::vector<double> restore_dual(
        const std::vector<double>& reduced_primal, const std::vector<double>& reduced_dual,
        const std::vector<double>& reduced_rc = {},
        const core::SolverTolerances& tolerances = {}) const;

    /**
     * @brief Reconstructs reduced costs in original variable coordinates from original duals.
     */
    [[nodiscard]] std::vector<double> restore_reduced_costs(
        const std::vector<double>& original_dual) const;

    /**
     * @brief Reconstructs the true objective value in the original model coordinate system.
     */
    [[nodiscard]] double restore_objective(const std::vector<double>& original_primal) const;

    /**
     * @brief Restores the original objective value from a reduced objective value.
     */
    [[nodiscard]] static double restore_objective_value(double reduced_objective_value) noexcept;

    /**
     * @brief Reconstructs the complete primal-dual optimality certificate in original coordinates.
     */
    [[nodiscard]] ReconstructedCertificate restore_certificate(
        const std::vector<double>& reduced_primal, const std::vector<double>& reduced_dual,
        const std::vector<double>& reduced_rc = {},
        const core::SolverTolerances& tolerances = {}) const;

private:
    std::vector<VariableMapping> variable_mappings_;
    std::vector<ConstraintMapping> constraint_mappings_;
    std::vector<PresolveTransformEntry> ledger_;
    double objective_offset_shift_{0.0};

    bool has_model_context_{false};
    bool has_reduced_dimensions_{false};
    std::size_t num_reduced_vars_{0};
    std::size_t num_reduced_constrs_{0};

    model::ObjectiveSense original_sense_{model::ObjectiveSense::MINIMIZE};
    double original_offset_{0.0};
    std::vector<double> original_obj_coeffs_;
    std::vector<double> original_var_lb_;
    std::vector<double> original_var_ub_;
    std::vector<model::ConstraintSense> original_row_sense_;
    std::vector<double> original_row_rhs_;
    std::vector<std::size_t> orig_row_ptr_;
    std::vector<model::VarId> orig_col_idx_;
    std::vector<double> orig_values_;
};

/**
 * @brief Structured outcome of applying presolve reductions to a CanonicalModel.
 */
struct PresolveResult {
    std::unique_ptr<model::CanonicalModel> reduced_model;
    ReconstructionMap reconstruction_map;
    PresolveStats stats;
    core::SolverStatus status{core::SolverStatus::UNKNOWN};
    std::string message;

    [[nodiscard]] bool is_infeasible() const noexcept {
        return status == core::SolverStatus::INFEASIBLE;
    }
    [[nodiscard]] bool is_unbounded() const noexcept {
        return status == core::SolverStatus::UNBOUNDED;
    }
    [[nodiscard]] bool has_reduced_model() const noexcept { return reduced_model != nullptr; }
};

/**
 * @brief Abstract interface for [C] Presolve.
 */
class PresolveEngine {
public:
    virtual ~PresolveEngine() = default;

    [[nodiscard]] virtual PresolveResult apply(
        const model::CanonicalModel& original_model,
        const PresolveOptions& options = PresolveOptions{}) = 0;
};

/**
 * @brief Production Presolver implementing mathematically safe reduction passes.
 *
 * Implements:
 * - Fixed-variable substitution (linear & quadratic)
 * - Empty row analysis (removal / infeasibility detection)
 * - Empty column / unconstrained variable analysis
 * - Singleton row bound tightening and contradiction detection
 * - Conservative activity-range reasoning and redundancy removal
 * - Bound propagation across general linear rows
 * - Zero-coefficient cleanup and expression canonicalization
 * - Deterministic multi-pass scheduling
 * - Complete solution reconstruction and objective preservation
 * - Input immutability (original model is never modified)
 */
class Presolver : public PresolveEngine {
public:
    Presolver() = default;

    [[nodiscard]] PresolveResult apply(const model::CanonicalModel& original_model,
                                       const PresolveOptions& options = PresolveOptions{}) override;

    [[nodiscard]] PresolveResult presolve(const model::CanonicalModel& original_model,
                                          const PresolveOptions& options = PresolveOptions{}) {
        return apply(original_model, options);
    }
};

}  // namespace kavach::presolve
