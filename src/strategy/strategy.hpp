#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "backend/execution_backend.hpp"
#include "benchmark/json.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"
#include "numerics/scaling.hpp"

namespace kavach::strategy {

// ============================================================================
// Enums & Classifications
// ============================================================================

enum class ProblemClass : std::uint8_t { LP, MILP, CONVEX_QP, CONVEX_MIQP };

[[nodiscard]] constexpr const char* to_string(ProblemClass pc) noexcept {
    switch (pc) {
        case ProblemClass::LP:
            return "LP";
        case ProblemClass::MILP:
            return "MILP";
        case ProblemClass::CONVEX_QP:
            return "CONVEX_QP";
        case ProblemClass::CONVEX_MIQP:
            return "CONVEX_MIQP";
    }
    return "UNKNOWN";
}

enum class PresolvePolicy : std::uint8_t {
    FULL,  ///< Standard comprehensive presolve (production pipeline)
    NONE   ///< Skip presolve (e.g. for QP or explicit user override)
};

[[nodiscard]] constexpr const char* to_string(PresolvePolicy pp) noexcept {
    switch (pp) {
        case PresolvePolicy::FULL:
            return "FULL";
        case PresolvePolicy::NONE:
            return "NONE";
    }
    return "UNKNOWN";
}

enum class ScalingPolicy : std::uint8_t { AUTO, FORCE_ON, FORCE_OFF };

[[nodiscard]] constexpr const char* to_string(ScalingPolicy sp) noexcept {
    switch (sp) {
        case ScalingPolicy::AUTO:
            return "AUTO";
        case ScalingPolicy::FORCE_ON:
            return "FORCE_ON";
        case ScalingPolicy::FORCE_OFF:
            return "FORCE_OFF";
    }
    return "UNKNOWN";
}

enum class NumericalRiskProfile : std::uint8_t {
    NORMAL,    ///< Standard tolerances & refactorization interval
    ELEVATED,  ///< Moderately high dynamic range; conservative refactorization
    HIGH       ///< Extreme dynamic range or near-zero coefficients; high vigilance
};

[[nodiscard]] constexpr const char* to_string(NumericalRiskProfile nr) noexcept {
    switch (nr) {
        case NumericalRiskProfile::NORMAL:
            return "NORMAL";
        case NumericalRiskProfile::ELEVATED:
            return "ELEVATED";
        case NumericalRiskProfile::HIGH:
            return "HIGH";
    }
    return "UNKNOWN";
}

enum class LPAlgorithm : std::uint8_t { REVISED_SIMPLEX };

[[nodiscard]] constexpr const char* to_string(LPAlgorithm alg) noexcept {
    switch (alg) {
        case LPAlgorithm::REVISED_SIMPLEX:
            return "REVISED_SIMPLEX";
    }
    return "UNKNOWN";
}

enum class MILPBranchingStrategy : std::uint8_t { MOST_FRACTIONAL };

[[nodiscard]] constexpr const char* to_string(MILPBranchingStrategy strat) noexcept {
    switch (strat) {
        case MILPBranchingStrategy::MOST_FRACTIONAL:
            return "MOST_FRACTIONAL";
    }
    return "UNKNOWN";
}

enum class MILPNodeStrategy : std::uint8_t { BEST_BOUND };

[[nodiscard]] constexpr const char* to_string(MILPNodeStrategy strat) noexcept {
    switch (strat) {
        case MILPNodeStrategy::BEST_BOUND:
            return "BEST_BOUND";
    }
    return "UNKNOWN";
}

enum class StrategyMode : std::uint8_t { AUTO, EXPLICIT };

[[nodiscard]] constexpr const char* to_string(StrategyMode mode) noexcept {
    switch (mode) {
        case StrategyMode::AUTO:
            return "AUTO";
        case StrategyMode::EXPLICIT:
            return "EXPLICIT";
    }
    return "UNKNOWN";
}

// ============================================================================
// Model Characterization
// ============================================================================

struct ProblemCharacteristics {
    // Structural
    std::size_t num_variables{0};
    std::size_t num_constraints{0};
    std::size_t num_nonzeros{0};
    double density{0.0};               ///< nnz / (m * n) or 0.0 if empty
    double average_row_sparsity{0.0};  ///< nnz / m or 0.0 if empty
    std::size_t max_row_nonzeros{0};
    std::size_t max_col_nonzeros{0};

    // Integrality
    std::size_t integer_variable_count{0};
    std::size_t binary_variable_count{0};
    double integer_variable_ratio{0.0};  ///< (integer + binary) / num_vars
    double binary_variable_ratio{0.0};   ///< binary / num_vars

    // Numerical
    double min_abs_nonzero_coeff{0.0};
    double max_abs_nonzero_coeff{0.0};
    double coefficient_range_ratio{1.0};                      ///< max / min
    bool coefficient_magnitudes_span_multiple_orders{false};  ///< range >= 1e4
    std::size_t near_zero_coefficient_count{0};  ///< count of coeffs with 0 < |v| < 1e-12
    numerics::ConditioningCategory conditioning_category{
        numerics::ConditioningCategory::WELL_CONDITIONED};
    bool recommend_scaling{false};

    // Problem Class
    ProblemClass problem_class{ProblemClass::LP};

    // Execution / Hardware Environment
    bool cuda_available{false};
    int cuda_device_count{0};
    std::size_t cpu_hardware_concurrency{1};
    bool measured_gpu_crossover_available{false};
    std::size_t measured_gpu_crossover_batch_size{128};
    std::size_t measured_gpu_crossover_matrix_nnz{10000};

    // Timing
    double characterization_time_ms{0.0};

    [[nodiscard]] benchmark::JsonValue to_json() const;
    [[nodiscard]] std::string to_string() const;
};

// ============================================================================
// Strategy Configuration & Centralized Thresholds
// ============================================================================

struct StrategyConfig {
    std::string config_version{"1.0.0"};

    // Execution mode
    StrategyMode mode{StrategyMode::AUTO};

    // Structural thresholds
    std::size_t small_problem_max_vars{100};
    std::size_t small_problem_max_constrs{100};
    std::size_t large_problem_min_vars{5000};

    // Dynamic range & Numerical risk thresholds
    double elevated_coeff_range_threshold{1e4};       ///< >= 1e4 -> ELEVATED
    double high_risk_coeff_range_threshold{1e8};      ///< >= 1e8 -> HIGH
    double severely_ill_coeff_range_threshold{1e13};  ///< >= 1e13 -> SEVERELY ILL

    // Integrality thresholds
    double high_integer_ratio_threshold{0.5};

    // Safeguard refactorization intervals
    std::size_t normal_refactorization_interval{50};
    std::size_t elevated_refactorization_interval{20};
    std::size_t high_risk_refactorization_interval{10};

    // Backend & GPU crossover thresholds
    std::size_t multicore_min_batch_size{16};
    std::size_t multicore_min_matrix_nnz{1000};
    std::size_t gpu_crossover_min_batch_size{128};
    std::size_t gpu_crossover_min_matrix_nnz{10000};

    // Explicit user overrides
    std::optional<bool> override_presolve;
    std::optional<bool> override_scaling;
    std::optional<backend::BackendType> override_backend;
    bool force_no_gpu{false};
    bool strict_gpu{false};

    [[nodiscard]] bool validate(std::string* error_out = nullptr) const noexcept;
    [[nodiscard]] benchmark::JsonValue to_json() const;
};

// ============================================================================
// Strategy Decision
// ============================================================================

struct StrategyDecision {
    std::string strategy_version{"1.0.0"};
    std::string config_version{"1.0.0"};

    ProblemClass problem_class{ProblemClass::LP};

    // Presolve & Scaling
    PresolvePolicy presolve_policy{PresolvePolicy::FULL};
    ScalingPolicy scaling_policy{ScalingPolicy::AUTO};
    bool scaling_enabled{true};

    // LP Strategy
    LPAlgorithm lp_algorithm{LPAlgorithm::REVISED_SIMPLEX};
    std::size_t lp_refactorization_interval{50};
    bool lp_force_bland_rule{false};

    // MILP Strategy
    MILPBranchingStrategy milp_branching_strategy{MILPBranchingStrategy::MOST_FRACTIONAL};
    bool pseudocost_available{false};  ///< Explicitly false (unsupported capability)
    MILPNodeStrategy milp_node_strategy{MILPNodeStrategy::BEST_BOUND};
    bool prefer_dual_simplex_warm_start{true};
    bool enable_rounding_heuristic{true};

    // Numerical Risk Profile
    NumericalRiskProfile numerical_risk_profile{NumericalRiskProfile::NORMAL};
    core::SolverTolerances recommended_tolerances;

    // Backend Decision
    backend::BackendType recommended_backend{backend::BackendType::CPU_SEQUENTIAL};
    std::size_t recommended_cpu_threads{1};
    bool gpu_eligible_for_batched_spmv{false};
    std::string gpu_routing_reason;

    // Rules & Explainability
    std::vector<std::string> rules_fired;
    std::vector<std::string> rule_explanations;
    std::string human_readable_rationale;

    // Timing
    double decision_time_ms{0.0};
    double total_strategy_time_ms{0.0};

    [[nodiscard]] benchmark::JsonValue to_json() const;
    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string explain() const;
};

// Compatibility struct for Phase 0 interface
struct SolverPlan {
    bool enable_presolve{true};
    bool enable_scaling{true};
    bool prefer_gpu_for_batched_spmv{false};
    core::SolverTolerances tuned_tolerances;

    static SolverPlan from_decision(const StrategyDecision& dec) {
        SolverPlan plan;
        plan.enable_presolve = (dec.presolve_policy == PresolvePolicy::FULL);
        plan.enable_scaling = dec.scaling_enabled;
        plan.prefer_gpu_for_batched_spmv = dec.gpu_eligible_for_batched_spmv;
        plan.tuned_tolerances = dec.recommended_tolerances;
        return plan;
    }
};

// ============================================================================
// Strategy Engine
// ============================================================================

class StrategyEngine {
public:
    explicit StrategyEngine(StrategyConfig config = StrategyConfig());

    [[nodiscard]] ProblemCharacteristics characterize(const model::CanonicalModel& model) const;
    [[nodiscard]] StrategyDecision decide(const ProblemCharacteristics& chars) const;
    [[nodiscard]] StrategyDecision decide(const ProblemCharacteristics& chars,
                                          const StrategyConfig& config) const;
    [[nodiscard]] StrategyDecision analyze_and_decide(const model::CanonicalModel& model) const;

    [[nodiscard]] const StrategyConfig& config() const noexcept { return config_; }
    void set_config(StrategyConfig config) noexcept { config_ = std::move(config); }

private:
    StrategyConfig config_;
};

// Abstract base class to maintain polymorphism
class SolverStrategy {
public:
    virtual ~SolverStrategy() = default;
    [[nodiscard]] virtual ProblemCharacteristics analyze(
        const model::CanonicalModel& model) const = 0;
    [[nodiscard]] virtual SolverPlan plan(const ProblemCharacteristics& characteristics) const = 0;
};

class DefaultSolverStrategy : public SolverStrategy {
public:
    explicit DefaultSolverStrategy(StrategyConfig config = StrategyConfig())
        : engine_(std::move(config)) {}

    [[nodiscard]] ProblemCharacteristics analyze(
        const model::CanonicalModel& model) const override {
        return engine_.characterize(model);
    }

    [[nodiscard]] SolverPlan plan(const ProblemCharacteristics& characteristics) const override {
        auto dec = engine_.decide(characteristics);
        return SolverPlan::from_decision(dec);
    }

private:
    StrategyEngine engine_;
};

}  // namespace kavach::strategy
