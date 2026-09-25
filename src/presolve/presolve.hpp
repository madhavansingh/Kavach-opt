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
 * @brief Mapping entry for a single original variable.
 */
struct VariableMapping {
    model::VarId original_var_id{model::INVALID_VAR_ID};
    VariableMappingStatus status{VariableMappingStatus::RETAINED};
    model::VarId reduced_var_id{model::INVALID_VAR_ID};
    double fixed_value{0.0};
};

/**
 * @brief Mechanism capable of restoring the original solution space after presolve reductions.
 */
class ReconstructionMap {
public:
    ReconstructionMap() = default;
    explicit ReconstructionMap(std::size_t num_original_vars);

    void set_mapping(model::VarId original_id, VariableMapping mapping);
    [[nodiscard]] const VariableMapping& mapping(model::VarId original_id) const;

    [[nodiscard]] std::size_t num_original_variables() const noexcept {
        return variable_mappings_.size();
    }
    [[nodiscard]] const std::vector<VariableMapping>& mappings() const noexcept {
        return variable_mappings_;
    }

    void set_objective_offset_shift(double shift) noexcept { objective_offset_shift_ = shift; }
    [[nodiscard]] double objective_offset_shift() const noexcept { return objective_offset_shift_; }

    /**
     * @brief Restores a full primal solution in the original variable space from a reduced
     * solution.
     */
    [[nodiscard]] std::vector<double> restore_solution(
        const std::vector<double>& reduced_solution) const;

    /**
     * @brief Restores the original objective value from a reduced objective value.
     */
    [[nodiscard]] static double restore_objective_value(double reduced_objective_value) noexcept;

private:
    std::vector<VariableMapping> variable_mappings_;
    double objective_offset_shift_{0.0};
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
