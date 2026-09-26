#include "strategy/strategy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <thread>

#include "backend/cuda/cuda_device.hpp"

namespace kavach::strategy {

// ============================================================================
// ProblemCharacteristics JSON & String
// ============================================================================

benchmark::JsonValue ProblemCharacteristics::to_json() const {
    benchmark::JsonValue::Object obj;
    obj["num_variables"] = static_cast<int64_t>(num_variables);
    obj["num_constraints"] = static_cast<int64_t>(num_constraints);
    obj["num_nonzeros"] = static_cast<int64_t>(num_nonzeros);
    obj["density"] = density;
    obj["average_row_sparsity"] = average_row_sparsity;
    obj["max_row_nonzeros"] = static_cast<int64_t>(max_row_nonzeros);
    obj["max_col_nonzeros"] = static_cast<int64_t>(max_col_nonzeros);

    obj["integer_variable_count"] = static_cast<int64_t>(integer_variable_count);
    obj["binary_variable_count"] = static_cast<int64_t>(binary_variable_count);
    obj["integer_variable_ratio"] = integer_variable_ratio;
    obj["binary_variable_ratio"] = binary_variable_ratio;

    obj["min_abs_nonzero_coeff"] = min_abs_nonzero_coeff;
    obj["max_abs_nonzero_coeff"] = max_abs_nonzero_coeff;
    obj["coefficient_range_ratio"] = coefficient_range_ratio;
    obj["coefficient_magnitudes_span_multiple_orders"] =
        coefficient_magnitudes_span_multiple_orders;
    obj["near_zero_coefficient_count"] = static_cast<int64_t>(near_zero_coefficient_count);
    obj["conditioning_category"] = numerics::to_string(conditioning_category);
    obj["recommend_scaling"] = recommend_scaling;

    obj["problem_class"] = kavach::strategy::to_string(problem_class);
    obj["cuda_available"] = cuda_available;
    obj["cuda_device_count"] = cuda_device_count;
    obj["cpu_hardware_concurrency"] = static_cast<int64_t>(cpu_hardware_concurrency);
    obj["measured_gpu_crossover_available"] = measured_gpu_crossover_available;
    obj["characterization_time_ms"] = characterization_time_ms;

    return benchmark::JsonValue(std::move(obj));
}

std::string ProblemCharacteristics::to_string() const {
    std::ostringstream ss;
    ss << "Problem Characteristics [" << kavach::strategy::to_string(problem_class) << "]:\n"
       << "  - Dimensions: " << num_variables << " vars, " << num_constraints << " constrs, "
       << num_nonzeros << " nnz (density: " << std::fixed << std::setprecision(4)
       << (density * 100.0) << "%)\n"
       << "  - Integrality: " << integer_variable_count << " integer ("
       << (integer_variable_ratio * 100.0) << "%), " << binary_variable_count << " binary\n"
       << "  - Dynamic Range: [" << std::scientific << std::setprecision(2) << min_abs_nonzero_coeff
       << ", " << max_abs_nonzero_coeff << "], ratio = " << coefficient_range_ratio << " ("
       << numerics::to_string(conditioning_category) << ")\n"
       << "  - Execution Environment: CUDA " << (cuda_available ? "YES" : "NO")
       << ", CPU threads = " << cpu_hardware_concurrency;
    return ss.str();
}

// ============================================================================
// StrategyConfig Validation & JSON
// ============================================================================

bool StrategyConfig::validate(std::string* error_out) const noexcept {
    if (small_problem_max_vars == 0) {
        if (error_out)
            *error_out = "small_problem_max_vars must be > 0";
        return false;
    }
    if (small_problem_max_constrs == 0) {
        if (error_out)
            *error_out = "small_problem_max_constrs must be > 0";
        return false;
    }
    if (large_problem_min_vars < small_problem_max_vars) {
        if (error_out)
            *error_out = "large_problem_min_vars must be >= small_problem_max_vars";
        return false;
    }
    if (elevated_coeff_range_threshold <= 1.0) {
        if (error_out)
            *error_out = "elevated_coeff_range_threshold must be > 1.0";
        return false;
    }
    if (high_risk_coeff_range_threshold <= elevated_coeff_range_threshold) {
        if (error_out)
            *error_out = "high_risk_coeff_range_threshold must be > elevated_coeff_range_threshold";
        return false;
    }
    if (severely_ill_coeff_range_threshold <= high_risk_coeff_range_threshold) {
        if (error_out)
            *error_out =
                "severely_ill_coeff_range_threshold must be > high_risk_coeff_range_threshold";
        return false;
    }
    if (high_integer_ratio_threshold < 0.0 || high_integer_ratio_threshold > 1.0) {
        if (error_out)
            *error_out = "high_integer_ratio_threshold must be in [0.0, 1.0]";
        return false;
    }
    if (normal_refactorization_interval == 0 || elevated_refactorization_interval == 0 ||
        high_risk_refactorization_interval == 0) {
        if (error_out)
            *error_out = "refactorization intervals must be > 0";
        return false;
    }
    if (elevated_refactorization_interval > normal_refactorization_interval ||
        high_risk_refactorization_interval > elevated_refactorization_interval) {
        if (error_out)
            *error_out = "refactorization intervals must satisfy high_risk <= elevated <= normal";
        return false;
    }
    return true;
}

benchmark::JsonValue StrategyConfig::to_json() const {
    benchmark::JsonValue::Object obj;
    obj["config_version"] = config_version;
    obj["mode"] = kavach::strategy::to_string(mode);
    obj["small_problem_max_vars"] = static_cast<int64_t>(small_problem_max_vars);
    obj["small_problem_max_constrs"] = static_cast<int64_t>(small_problem_max_constrs);
    obj["large_problem_min_vars"] = static_cast<int64_t>(large_problem_min_vars);
    obj["elevated_coeff_range_threshold"] = elevated_coeff_range_threshold;
    obj["high_risk_coeff_range_threshold"] = high_risk_coeff_range_threshold;
    obj["severely_ill_coeff_range_threshold"] = severely_ill_coeff_range_threshold;
    obj["high_integer_ratio_threshold"] = high_integer_ratio_threshold;
    obj["normal_refactorization_interval"] = static_cast<int64_t>(normal_refactorization_interval);
    obj["elevated_refactorization_interval"] =
        static_cast<int64_t>(elevated_refactorization_interval);
    obj["high_risk_refactorization_interval"] =
        static_cast<int64_t>(high_risk_refactorization_interval);
    obj["force_no_gpu"] = force_no_gpu;
    obj["strict_gpu"] = strict_gpu;
    return benchmark::JsonValue(std::move(obj));
}

// ============================================================================
// StrategyDecision JSON & String & Explain
// ============================================================================

benchmark::JsonValue StrategyDecision::to_json() const {
    benchmark::JsonValue::Object obj;
    obj["strategy_version"] = strategy_version;
    obj["config_version"] = config_version;
    obj["problem_class"] = kavach::strategy::to_string(problem_class);
    obj["presolve_policy"] = kavach::strategy::to_string(presolve_policy);
    obj["scaling_policy"] = kavach::strategy::to_string(scaling_policy);
    obj["scaling_enabled"] = scaling_enabled;

    obj["lp_algorithm"] = kavach::strategy::to_string(lp_algorithm);
    obj["lp_refactorization_interval"] = static_cast<int64_t>(lp_refactorization_interval);
    obj["lp_force_bland_rule"] = lp_force_bland_rule;

    obj["milp_branching_strategy"] = kavach::strategy::to_string(milp_branching_strategy);
    obj["pseudocost_available"] = pseudocost_available;
    obj["milp_node_strategy"] = kavach::strategy::to_string(milp_node_strategy);
    obj["prefer_dual_simplex_warm_start"] = prefer_dual_simplex_warm_start;
    obj["enable_rounding_heuristic"] = enable_rounding_heuristic;

    obj["numerical_risk_profile"] = kavach::strategy::to_string(numerical_risk_profile);
    obj["recommended_backend"] = backend::to_string(recommended_backend);
    obj["recommended_cpu_threads"] = static_cast<int64_t>(recommended_cpu_threads);
    obj["gpu_eligible_for_batched_spmv"] = gpu_eligible_for_batched_spmv;
    obj["gpu_routing_reason"] = gpu_routing_reason;

    benchmark::JsonValue::Array rarr;
    rarr.reserve(rules_fired.size());
    for (const auto& r : rules_fired) {
        rarr.emplace_back(r);
    }
    obj["rules_fired"] = std::move(rarr);

    benchmark::JsonValue::Array exarr;
    exarr.reserve(rule_explanations.size());
    for (const auto& ex : rule_explanations) {
        exarr.emplace_back(ex);
    }
    obj["rule_explanations"] = std::move(exarr);

    obj["human_readable_rationale"] = human_readable_rationale;
    obj["decision_time_ms"] = decision_time_ms;
    obj["total_strategy_time_ms"] = total_strategy_time_ms;

    return benchmark::JsonValue(std::move(obj));
}

std::string StrategyDecision::to_string() const {
    std::ostringstream ss;
    ss << "Strategy Decision v" << strategy_version << " ["
       << kavach::strategy::to_string(problem_class) << "]:\n"
       << "  - Presolve: " << kavach::strategy::to_string(presolve_policy) << "\n"
       << "  - Scaling: " << (scaling_enabled ? "ENABLED" : "DISABLED") << " ("
       << kavach::strategy::to_string(scaling_policy) << ")\n"
       << "  - Risk Profile: " << kavach::strategy::to_string(numerical_risk_profile)
       << " (Refactor every " << lp_refactorization_interval << " iters)\n"
       << "  - Backend: " << backend::to_string(recommended_backend) << " ("
       << recommended_cpu_threads << " threads)\n"
       << "  - GPU Batched SpMV Eligible: " << (gpu_eligible_for_batched_spmv ? "YES" : "NO")
       << " (" << gpu_routing_reason << ")\n"
       << "  - Rules Fired: " << rules_fired.size();
    return ss.str();
}

std::string StrategyDecision::explain() const {
    std::ostringstream ss;
    ss << "==============================================================================\n"
       << "KAVACH-OPT DETERMINISTIC SOLVER STRATEGY EXPLANATION (v" << strategy_version << ")\n"
       << "==============================================================================\n"
       << "Target Problem Class: " << kavach::strategy::to_string(problem_class) << "\n"
       << "Selected Presolve:    " << kavach::strategy::to_string(presolve_policy) << "\n"
       << "Selected Scaling:     " << (scaling_enabled ? "ENABLED" : "DISABLED") << " ("
       << kavach::strategy::to_string(scaling_policy) << ")\n"
       << "Numerical Risk:       " << kavach::strategy::to_string(numerical_risk_profile) << "\n"
       << "Refactor Interval:    " << lp_refactorization_interval << " iters\n"
       << "Recommended Backend:  " << backend::to_string(recommended_backend) << " ("
       << recommended_cpu_threads << " workers)\n"
       << "GPU SpMV Routing:     " << (gpu_eligible_for_batched_spmv ? "ELIGIBLE" : "CPU FALLBACK")
       << " [" << gpu_routing_reason << "]\n\n"
       << "RULES TRIGGERED:\n";
    for (std::size_t i = 0; i < rules_fired.size(); ++i) {
        ss << "  [" << rules_fired[i] << "] "
           << (i < rule_explanations.size() ? rule_explanations[i] : "") << "\n";
    }
    ss << "\nSUMMARY RATIONALE:\n"
       << human_readable_rationale << "\n"
       << "==============================================================================\n";
    return ss.str();
}

// ============================================================================
// StrategyEngine Implementation
// ============================================================================

StrategyEngine::StrategyEngine(StrategyConfig config) : config_(std::move(config)) {
    std::string err;
    if (!config_.validate(&err)) {
        // Fall back to safe default if invalid config passed
        config_ = StrategyConfig();
    }
}

ProblemCharacteristics StrategyEngine::characterize(const model::CanonicalModel& model) const {
    const auto start_time = std::chrono::steady_clock::now();

    ProblemCharacteristics chars;
    chars.num_variables = model.num_variables();
    chars.num_constraints = model.num_constraints();

    // Check problem class
    if (model.is_miqp()) {
        chars.problem_class = ProblemClass::CONVEX_MIQP;
    } else if (model.is_qp()) {
        chars.problem_class = ProblemClass::CONVEX_QP;
    } else if (model.is_milp()) {
        chars.problem_class = ProblemClass::MILP;
    } else {
        chars.problem_class = ProblemClass::LP;
    }

    // Integrality scan
    for (std::size_t j = 0; j < chars.num_variables; ++j) {
        const auto& v = model.variable(static_cast<model::VarId>(j));
        if (v.type == model::VariableType::BINARY) {
            chars.binary_variable_count++;
        } else if (v.type == model::VariableType::INTEGER) {
            chars.integer_variable_count++;
        }
    }
    if (chars.num_variables > 0) {
        chars.integer_variable_ratio =
            static_cast<double>(chars.integer_variable_count + chars.binary_variable_count) /
            static_cast<double>(chars.num_variables);
        chars.binary_variable_ratio = static_cast<double>(chars.binary_variable_count) /
                                      static_cast<double>(chars.num_variables);
    }

    // Constraint and coefficient scan
    std::vector<std::size_t> col_nnz(chars.num_variables, 0);
    double min_abs = 0.0;
    double max_abs = 0.0;
    bool first_coeff = true;
    std::size_t total_nnz = 0;

    for (std::size_t i = 0; i < chars.num_constraints; ++i) {
        const auto& c = model.constraint(static_cast<model::ConstrId>(i));
        const auto& terms = c.expression.terms();
        chars.max_row_nonzeros = std::max(chars.max_row_nonzeros, terms.size());
        total_nnz += terms.size();

        for (const auto& t : terms) {
            if (t.var_id < chars.num_variables) {
                col_nnz[t.var_id]++;
            }
            const double abs_val = std::abs(t.coefficient);
            if (abs_val > 0.0) {
                if (first_coeff) {
                    min_abs = abs_val;
                    max_abs = abs_val;
                    first_coeff = false;
                } else {
                    min_abs = std::min(min_abs, abs_val);
                    max_abs = std::max(max_abs, abs_val);
                }
                if (abs_val < 1e-12) {
                    chars.near_zero_coefficient_count++;
                }
            }
        }
    }

    // Include linear and quadratic objective terms in coefficient range check
    for (const auto& lt : model.objective().linear_terms.terms()) {
        const double abs_val = std::abs(lt.coefficient);
        if (abs_val > 0.0) {
            if (first_coeff) {
                min_abs = abs_val;
                max_abs = abs_val;
                first_coeff = false;
            } else {
                min_abs = std::min(min_abs, abs_val);
                max_abs = std::max(max_abs, abs_val);
            }
            if (abs_val < 1e-12) {
                chars.near_zero_coefficient_count++;
            }
        }
    }

    if (model.is_qp()) {
        for (const auto& qt : model.objective().quadratic_terms.terms()) {
            const double abs_val = std::abs(qt.coefficient);
            if (abs_val > 0.0) {
                if (first_coeff) {
                    min_abs = abs_val;
                    max_abs = abs_val;
                    first_coeff = false;
                } else {
                    min_abs = std::min(min_abs, abs_val);
                    max_abs = std::max(max_abs, abs_val);
                }
                if (abs_val < 1e-12) {
                    chars.near_zero_coefficient_count++;
                }
            }
        }
    }

    chars.num_nonzeros = total_nnz;
    for (std::size_t nnz : col_nnz) {
        chars.max_col_nonzeros = std::max(chars.max_col_nonzeros, nnz);
    }

    if (chars.num_variables > 0 && chars.num_constraints > 0) {
        chars.density = static_cast<double>(total_nnz) /
                        static_cast<double>(chars.num_variables * chars.num_constraints);
        chars.average_row_sparsity =
            static_cast<double>(total_nnz) / static_cast<double>(chars.num_constraints);
    }

    chars.min_abs_nonzero_coeff = min_abs;
    chars.max_abs_nonzero_coeff = max_abs;
    chars.coefficient_range_ratio = (min_abs > 0.0) ? (max_abs / min_abs) : 1.0;
    chars.coefficient_magnitudes_span_multiple_orders =
        (chars.coefficient_range_ratio >= config_.elevated_coeff_range_threshold);

    // Conditioning category
    if (chars.coefficient_range_ratio < 1e4) {
        chars.conditioning_category = numerics::ConditioningCategory::WELL_CONDITIONED;
        chars.recommend_scaling = false;
    } else if (chars.coefficient_range_ratio < 1e8) {
        chars.conditioning_category = numerics::ConditioningCategory::MODERATE;
        chars.recommend_scaling = true;
    } else if (chars.coefficient_range_ratio < 1e13) {
        chars.conditioning_category = numerics::ConditioningCategory::ILL_CONDITIONED;
        chars.recommend_scaling = true;
    } else {
        chars.conditioning_category = numerics::ConditioningCategory::SEVERELY_ILL_CONDITIONED;
        chars.recommend_scaling = true;
    }

    // Hardware probing
    chars.cuda_available = backend::cuda::is_cuda_available();
    chars.cuda_device_count = backend::cuda::get_cuda_device_count();
    chars.cpu_hardware_concurrency = std::max<std::size_t>(1, std::thread::hardware_concurrency());

    // Measured crossover empirical check
    // If CUDA is available, we benchmarked that crossover occurs at batch >= 128 and nnz >= 10000
    chars.measured_gpu_crossover_available = chars.cuda_available;
    chars.measured_gpu_crossover_batch_size = config_.gpu_crossover_min_batch_size;
    chars.measured_gpu_crossover_matrix_nnz = config_.gpu_crossover_min_matrix_nnz;

    const auto end_time = std::chrono::steady_clock::now();
    chars.characterization_time_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return chars;
}

StrategyDecision StrategyEngine::decide(const ProblemCharacteristics& chars) const {
    return decide(chars, config_);
}

StrategyDecision StrategyEngine::decide(const ProblemCharacteristics& chars,
                                        const StrategyConfig& config) const {
    const auto start_time = std::chrono::steady_clock::now();

    StrategyDecision dec;
    dec.strategy_version = "1.0.0";
    dec.config_version = config.config_version;
    dec.problem_class = chars.problem_class;

    std::ostringstream rationale;

    // --- RULE R001 / R002 / R003: Problem Classification & Target Solver ---
    if (chars.problem_class == ProblemClass::CONVEX_MIQP) {
        dec.rules_fired.emplace_back("R001b");
        dec.rule_explanations.emplace_back(
            "Convex MIQP model detected; targeted Convex MIQP Branch-and-Bound with Active-Set QP "
            "relaxations.");
        dec.milp_branching_strategy = MILPBranchingStrategy::MOST_FRACTIONAL;
        dec.milp_node_strategy = MILPNodeStrategy::BEST_BOUND;
        dec.prefer_dual_simplex_warm_start = false;
        dec.enable_rounding_heuristic = true;
        dec.pseudocost_available = false;
        rationale << "Classified as Convex MIQP (" << chars.integer_variable_count << " integer, "
                  << chars.binary_variable_count
                  << " binary, quadratic objective). Routed to Convex MIQP Branch-and-Bound with "
                     "Active-Set QP relaxations. ";
    } else if (chars.problem_class == ProblemClass::CONVEX_QP) {
        dec.rules_fired.emplace_back("R001");
        dec.rule_explanations.emplace_back(
            "Convex QP model detected; targeted Primal Active-Set QP Solver.");
        rationale << "Classified as Convex QP. Routed to Primal Active-Set Solver. ";
    } else if (chars.problem_class == ProblemClass::MILP) {
        dec.rules_fired.emplace_back("R002");
        dec.rule_explanations.emplace_back(
            "MILP model detected; targeted Branch-and-Bound with most-fractional branching, "
            "best-bound node selection, and dual-simplex warm starts.");
        dec.milp_branching_strategy = MILPBranchingStrategy::MOST_FRACTIONAL;
        dec.milp_node_strategy = MILPNodeStrategy::BEST_BOUND;
        dec.prefer_dual_simplex_warm_start = true;
        dec.enable_rounding_heuristic = true;
        dec.pseudocost_available = false;
        rationale << "Classified as MILP (" << chars.integer_variable_count << " integer, "
                  << chars.binary_variable_count
                  << " binary). Routed to Branch-and-Bound with dual warm-start. ";
    } else {
        dec.rules_fired.emplace_back("R003");
        dec.rule_explanations.emplace_back(
            "Continuous LP model detected; targeted Two-Phase Revised Simplex LP Solver.");
        dec.lp_algorithm = LPAlgorithm::REVISED_SIMPLEX;
        rationale << "Classified as continuous LP. Routed to Revised Simplex. ";
    }

    // --- RULE R004 / R005: Presolve Strategy ---
    if (chars.problem_class == ProblemClass::CONVEX_QP ||
        chars.problem_class == ProblemClass::CONVEX_MIQP) {
        dec.rules_fired.emplace_back("R005");
        dec.rule_explanations.emplace_back(
            "Presolve skipped for quadratic objective models: disabled by default to preserve "
            "quadratic objective mapping.");
        dec.presolve_policy = PresolvePolicy::NONE;
    } else {
        dec.rules_fired.emplace_back("R004");
        dec.rule_explanations.emplace_back(
            "Standard full presolve suite enabled (fixed variables, empty rows/cols, bound "
            "tightening, redundant rows).");
        dec.presolve_policy = PresolvePolicy::FULL;
    }

    // --- RULE R006: Scaling Policy ---
    dec.scaling_enabled = true;
    dec.scaling_policy = ScalingPolicy::AUTO;
    if (chars.coefficient_range_ratio >= config.elevated_coeff_range_threshold ||
        chars.recommend_scaling) {
        dec.rules_fired.emplace_back("R006");
        dec.rule_explanations.emplace_back(
            "Coefficient dynamic range ratio >= 1e4; geometric-mean scaling recommended (AUTO).");
        rationale << "Scaling enabled for dynamic range " << std::scientific << std::setprecision(1)
                  << chars.coefficient_range_ratio << ". ";
    }

    // --- RULE R007 / R008 / R009: Numerical Risk Profile ---
    if (chars.coefficient_range_ratio >= config.high_risk_coeff_range_threshold ||
        chars.near_zero_coefficient_count > 0) {
        dec.rules_fired.emplace_back("R009");
        dec.rule_explanations.emplace_back(
            "High numerical risk: dynamic range >= 1e8 or near-zero coefficients present. "
            "Defensive tolerances, Bland watchdog, and refactorization interval 10 applied.");
        dec.numerical_risk_profile = NumericalRiskProfile::HIGH;
        dec.lp_refactorization_interval = config.high_risk_refactorization_interval;
        dec.lp_force_bland_rule = true;
        dec.recommended_tolerances.pivot_tol = 1e-8;
        rationale << "Numerical risk HIGH: maximum refactorization vigilance applied. ";
    } else if (chars.coefficient_range_ratio >= config.elevated_coeff_range_threshold) {
        dec.rules_fired.emplace_back("R008");
        dec.rule_explanations.emplace_back(
            "Elevated numerical risk: dynamic range in [1e4, 1e8). Refactorization interval "
            "tightened to 20 iterations.");
        dec.numerical_risk_profile = NumericalRiskProfile::ELEVATED;
        dec.lp_refactorization_interval = config.elevated_refactorization_interval;
        rationale << "Numerical risk ELEVATED: refactorization interval set to 20. ";
    } else {
        dec.rules_fired.emplace_back("R007");
        dec.rule_explanations.emplace_back(
            "Normal numerical profile: well-conditioned matrix. Standard refactorization interval "
            "(50) applied.");
        dec.numerical_risk_profile = NumericalRiskProfile::NORMAL;
        dec.lp_refactorization_interval = config.normal_refactorization_interval;
    }

    // --- RULE R010 / R012: CPU Execution Routing ---
    if (chars.cpu_hardware_concurrency > 1 &&
        (chars.num_nonzeros >= config.multicore_min_matrix_nnz ||
         chars.num_variables >= config.small_problem_max_vars)) {
        dec.rules_fired.emplace_back("R012");
        dec.rule_explanations.emplace_back(
            "Multicore CPU selected: workload exceeds parallel threshold for batched SpMV.");
        dec.recommended_backend = backend::BackendType::CPU_MULTICORE;
        dec.recommended_cpu_threads = chars.cpu_hardware_concurrency;
        rationale << "Recommended backend: CPU_MULTICORE with " << chars.cpu_hardware_concurrency
                  << " threads. ";
    } else {
        dec.rules_fired.emplace_back("R010");
        dec.rule_explanations.emplace_back(
            "CPU sequential selected: small problem / single-batch workload.");
        dec.recommended_backend = backend::BackendType::CPU_SEQUENTIAL;
        dec.recommended_cpu_threads = 1;
        rationale << "Recommended backend: CPU_SEQUENTIAL. ";
    }

    // --- RULE R011 / R013: GPU Routing ---
    if (config.force_no_gpu) {
        dec.rules_fired.emplace_back("R011");
        dec.rule_explanations.emplace_back(
            "CPU forced when --no-gpu specified; GPU execution disabled.");
        dec.gpu_eligible_for_batched_spmv = false;
        dec.gpu_routing_reason = "--no-gpu requested by user";
    } else if (!chars.cuda_available) {
        dec.gpu_eligible_for_batched_spmv = false;
        dec.gpu_routing_reason = "No CUDA-capable GPU detected";
    } else if (chars.measured_gpu_crossover_available) {
        dec.rules_fired.emplace_back("R013");
        dec.rule_explanations.emplace_back(
            "GPU crossover condition verified with empirical benchmark data; GPU eligible for "
            "batched SpMV.");
        dec.gpu_eligible_for_batched_spmv = true;
        dec.gpu_routing_reason =
            "CUDA device detected; eligible for batched SpMV when workload >= crossover";
    } else {
        dec.rules_fired.emplace_back("R013");
        dec.rule_explanations.emplace_back(
            "No measured crossover data; CPU execution is default and optimal.");
        dec.gpu_eligible_for_batched_spmv = false;
        dec.gpu_routing_reason = "No empirical crossover benchmark; CPU default";
    }

    // --- RULE R014: High Integrality Ratio Heuristics ---
    if ((chars.problem_class == ProblemClass::MILP ||
         chars.problem_class == ProblemClass::CONVEX_MIQP) &&
        chars.integer_variable_ratio >= config.high_integer_ratio_threshold) {
        dec.rules_fired.emplace_back("R014");
        dec.rule_explanations.emplace_back(
            "High integrality ratio detected (>= 50%); aggressive rounding heuristic enabled.");
        dec.enable_rounding_heuristic = true;
    }

    // --- RULE R015 / R016 / R017: Explicit User Overrides ---
    if (config.override_presolve.has_value()) {
        dec.rules_fired.emplace_back("R015");
        dec.rule_explanations.emplace_back("Explicit user override applied to presolve policy.");
        dec.presolve_policy =
            (*config.override_presolve ? PresolvePolicy::FULL : PresolvePolicy::NONE);
        rationale << "[Override: Presolve=" << (*config.override_presolve ? "FULL" : "NONE")
                  << "] ";
    }

    if (config.override_scaling.has_value()) {
        dec.rules_fired.emplace_back("R016");
        dec.rule_explanations.emplace_back("Explicit user override applied to scaling policy.");
        dec.scaling_enabled = *config.override_scaling;
        dec.scaling_policy =
            (*config.override_scaling ? ScalingPolicy::FORCE_ON : ScalingPolicy::FORCE_OFF);
        rationale << "[Override: Scaling=" << (*config.override_scaling ? "FORCE_ON" : "FORCE_OFF")
                  << "] ";
    }

    if (config.override_backend.has_value()) {
        dec.rules_fired.emplace_back("R017");
        dec.rule_explanations.emplace_back("Explicit user override applied to execution backend.");
        dec.recommended_backend = *config.override_backend;
        rationale << "[Override: Backend=" << backend::to_string(*config.override_backend) << "] ";
    }

    // --- RULE R018: Precedence Check - Safety Overrides Optimization ---
    if (dec.numerical_risk_profile == NumericalRiskProfile::HIGH) {
        dec.rules_fired.emplace_back("R018");
        dec.rule_explanations.emplace_back(
            "Safety Precedence: High numerical risk overrides refactorization interval to "
            "10 iterations.");
        dec.lp_refactorization_interval = config.high_risk_refactorization_interval;
    }

    dec.human_readable_rationale = rationale.str();

    const auto end_time = std::chrono::steady_clock::now();
    dec.decision_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    dec.total_strategy_time_ms = chars.characterization_time_ms + dec.decision_time_ms;

    return dec;
}

StrategyDecision StrategyEngine::analyze_and_decide(const model::CanonicalModel& model) const {
    auto chars = characterize(model);
    return decide(chars, config_);
}

}  // namespace kavach::strategy
