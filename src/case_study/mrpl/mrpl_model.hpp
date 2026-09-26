#pragma once

/**
 * @file mrpl_model.hpp
 * @brief MRPL-Style Refinery Optimization Industrial Demonstration Model.
 *
 * DATA STATUS:
 * All numeric values in this demonstration are illustrative/synthetic.
 * They are not proprietary MRPL operational data and must not be
 * interpreted as actual MRPL process parameters.
 */

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "algorithms/milp/branch_and_bound.hpp"
#include "benchmark/json.hpp"
#include "benchmark/reference/reference_adapter.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"
#include "verification/milp_verifier.hpp"

namespace kavach::case_study::mrpl {

constexpr std::string_view SYNTHETIC_DATA_DISCLOSURE =
    "DATA STATUS:\n"
    "All numeric values in this demonstration are illustrative/synthetic.\n"
    "They are not proprietary MRPL operational data and must not be\n"
    "interpreted as actual MRPL process parameters.";

// ============================================================================
// Parameter Specifications
// ============================================================================

struct CrudeData {
    std::string name;
    std::string description;
    double availability{0.0};  ///< kbpd
    double cost{0.0};          ///< $/bbl
    double sulfur{0.0};        ///< wt %
    double density{0.0};       ///< kg/m³
};

struct ProductData {
    std::string name;
    std::string description;
    double max_demand{0.0};   ///< kbpd
    double min_demand{0.0};   ///< kbpd (default 0)
    double price{0.0};        ///< $/bbl
    double max_sulfur{0.0};   ///< wt %
    double max_density{0.0};  ///< kg/m³ (0 = no upper limit)
    double min_density{0.0};  ///< kg/m³ (0 = no lower limit)
};

struct UnitData {
    std::string name;
    std::string description;
    double min_capacity{0.0};   ///< kbpd (when active)
    double max_capacity{0.0};   ///< kbpd (when active)
    double variable_cost{0.0};  ///< $/bbl
    double fixed_cost{0.0};     ///< $k/day
};

struct MRPLParameters {
    std::string case_name{"mrpl_blending_capacity"};
    std::string version{"1.0.0"};
    std::vector<CrudeData> crudes;
    std::vector<ProductData> products;
    std::vector<UnitData> units;

    [[nodiscard]] static MRPLParameters default_parameters();
    [[nodiscard]] static MRPLParameters scenario_reduced_crude();
    [[nodiscard]] static MRPLParameters scenario_tight_quality();
    [[nodiscard]] static MRPLParameters scenario_reduced_capacity();
    [[nodiscard]] static MRPLParameters scenario_infeasible();
};

// ============================================================================
// Model Generator
// ============================================================================

class MRPLModelGenerator {
public:
    [[nodiscard]] static model::CanonicalModel build_canonical_model(const MRPLParameters& params);
    [[nodiscard]] static std::string generate_lp_string(const MRPLParameters& params);
    static void save_lp_file(const MRPLParameters& params, const std::string& filepath);
};

// ============================================================================
// Solution Analysis & Domain Validation
// ============================================================================

struct ProductQualityEvaluation {
    std::string product_name;
    double production_kbpd{0.0};
    double calculated_sulfur_wt_pct{0.0};
    double sulfur_limit_wt_pct{0.0};
    double sulfur_margin_wt_pct{0.0};  ///< limit - actual (positive = satisfied)
    double calculated_density_kg_m3{0.0};
    double density_limit_kg_m3{0.0};
    bool density_is_upper_limit{true};
    double density_margin_kg_m3{0.0};  ///< positive = satisfied
    bool quality_satisfied{true};
};

struct EconomicBreakdown {
    double gross_revenue_k_usd{0.0};
    double crude_purchase_cost_k_usd{0.0};
    double unit_variable_cost_k_usd{0.0};
    double unit_fixed_cost_k_usd{0.0};
    double total_cost_k_usd{0.0};
    double net_profit_k_usd{0.0};
};

struct UnitOperationalStatus {
    std::string unit_name;
    std::string description;
    bool is_active{false};
    double throughput_kbpd{0.0};
    double min_capacity_kbpd{0.0};
    double max_capacity_kbpd{0.0};
    double utilization_pct{0.0};
    std::string operational_summary;
};

struct MassBalanceValidation {
    double max_absolute_residual{0.0};
    double crude_availability_residual{0.0};
    double product_balance_residual{0.0};
    double unit_throughput_residual{0.0};
    double unit_capacity_residual{0.0};
    bool passed{true};
    std::string diagnostic;
};

struct MRPLSolutionBreakdown {
    std::string scenario_name;
    core::SolverStatus status{core::SolverStatus::UNKNOWN};
    double objective{0.0};
    double lp_relaxation_objective{0.0};
    double best_bound{0.0};
    double optimality_gap{0.0};
    std::size_t nodes_explored{0};
    double solve_time_seconds{0.0};
    double lp_relaxation_time_seconds{0.0};
    std::vector<std::string> lp_fractional_variables;

    std::map<std::pair<std::string, std::string>, double>
        blend_flows;                                   ///< (crude, product) -> flow
    std::map<std::string, double> product_production;  ///< product -> total
    std::map<std::string, double> unit_throughputs;    ///< unit -> throughput
    std::map<std::string, double> unit_binaries;       ///< unit -> binary (0 or 1)

    MassBalanceValidation mass_balance;
    std::vector<ProductQualityEvaluation> product_qualities;
    EconomicBreakdown economics;
    std::vector<UnitOperationalStatus> unit_decisions;

    verification::MILPVerificationReport verification_report;
    bool verified_optimal{false};

    benchmark::ReferenceRunResult highs_result;

    [[nodiscard]] benchmark::JsonValue to_json() const;
};

// ============================================================================
// Case Study Runner & Report Generator
// ============================================================================

class MRPLCaseStudyRunner {
public:
    explicit MRPLCaseStudyRunner(core::SolverTolerances tolerances = core::SolverTolerances());

    [[nodiscard]] MRPLSolutionBreakdown solve_and_analyze(const MRPLParameters& params,
                                                          std::string scenario_name = "baseline",
                                                          bool run_highs_comparison = true) const;

    [[nodiscard]] std::vector<MRPLSolutionBreakdown> run_all_scenarios(
        bool run_highs_comparison = true) const;

    void export_all_artifacts(const std::string& results_dir,
                              const std::string& report_filepath) const;

    [[nodiscard]] static std::string generate_markdown_report(
        const std::vector<MRPLSolutionBreakdown>& scenarios);

private:
    core::SolverTolerances tolerances_;
};

}  // namespace kavach::case_study::mrpl
