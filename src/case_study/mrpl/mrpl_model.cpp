/**
 * @file mrpl_model.cpp
 * @brief Implementation of MRPL refinery optimization model and analysis engine.
 */

#include "case_study/mrpl/mrpl_model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "algorithms/lp/simplex.hpp"
#include "core/logger.hpp"
#include "io/parser.hpp"

namespace kavach::case_study::mrpl {

// ============================================================================
// Parameter Specifications
// ============================================================================

MRPLParameters MRPLParameters::default_parameters() {
    MRPLParameters p;
    p.case_name = "mrpl_baseline";
    p.version = "1.0.0";

    p.crudes = {{"A", "Light Sweet Crude", 120.0, 80.0, 0.5, 820.0},
                {"B", "Medium Sweet Crude", 100.0, 72.0, 1.1, 850.0},
                {"C", "Medium Sour Crude", 140.0, 60.0, 2.4, 875.0},
                {"D", "Heavy Sour Crude", 80.0, 48.0, 3.6, 910.0}};

    p.products = {{"P1", "Motor Spirit (Gasoline)", 140.0, 0.0, 110.0, 0.8, 840.0, 0.0},
                  {"P2", "High-Speed Diesel (HSD)", 160.0, 0.0, 95.0, 1.3, 865.0, 0.0},
                  {"P3", "Heavy Fuel Oil (HFO)", 90.0, 0.0, 65.0, 3.0, 0.0, 875.0}};

    p.units = {{"U1", "Atmospheric Distillation Unit (CDU)", 50.0, 200.0, 2.5, 500.0},
               {"U2", "Vacuum Distillation & Conversion Unit (VDU/FCCU)", 60.0, 180.0, 3.5, 600.0},
               {"U3", "Hydrodesulfurization Unit (HDS)", 30.0, 120.0, 4.5, 400.0}};

    return p;
}

MRPLParameters MRPLParameters::scenario_reduced_crude() {
    MRPLParameters p = default_parameters();
    p.case_name = "scenario_a_reduced_crude";
    // Disruption in sweet crude supply: Crude A availability reduced 120 -> 50 kbpd
    p.crudes[0].availability = 50.0;
    return p;
}

MRPLParameters MRPLParameters::scenario_tight_quality() {
    MRPLParameters p = default_parameters();
    p.case_name = "scenario_b_tight_quality";
    // Tightened environmental specifications: Diesel max sulfur reduced 1.3% -> 0.9%
    p.products[1].max_sulfur = 0.9;
    return p;
}

MRPLParameters MRPLParameters::scenario_reduced_capacity() {
    MRPLParameters p = default_parameters();
    p.case_name = "scenario_c_reduced_capacity";
    // Maintenance turnaround / derating: HDS capacity reduced 120 -> 40 kbpd
    p.units[2].max_capacity = 40.0;
    return p;
}

MRPLParameters MRPLParameters::scenario_infeasible() {
    MRPLParameters p = default_parameters();
    p.case_name = "scenario_d_infeasible";
    // Impose high minimum demand for clean products but starve crude feedstocks
    p.products[0].min_demand = 140.0;
    p.crudes[0].availability = 10.0;
    p.crudes[1].availability = 10.0;
    p.crudes[2].availability = 10.0;
    p.crudes[3].availability = 10.0;
    return p;
}

// ============================================================================
// Model Generator
// ============================================================================

model::CanonicalModel MRPLModelGenerator::build_canonical_model(const MRPLParameters& params) {
    model::CanonicalModel m(params.case_name);
    m.set_objective_sense(model::ObjectiveSense::MAXIMIZE);

    std::map<std::pair<std::string, std::string>, model::VarId> x_vars;
    std::map<std::string, model::VarId> prod_vars;
    std::map<std::string, model::VarId> t_vars;
    std::map<std::string, model::VarId> y_vars;

    // 1. Blending flow variables x[c, p]
    for (const auto& c : params.crudes) {
        for (const auto& p : params.products) {
            std::string var_name = "x_" + c.name + "_" + p.name;
            auto v = m.add_variable(var_name, model::VariableType::CONTINUOUS, 0.0,
                                    model::INFINITY_BOUND);
            x_vars[{c.name, p.name}] = v;
            m.add_objective_linear_term(v, -c.cost);
        }
    }

    // 2. Production variables prod[p]
    for (const auto& p : params.products) {
        std::string var_name = "prod_" + p.name;
        auto v =
            m.add_variable(var_name, model::VariableType::CONTINUOUS, p.min_demand, p.max_demand);
        prod_vars[p.name] = v;
        m.add_objective_linear_term(v, p.price);
    }

    // 3. Unit throughput t[u] and binary activation y[u]
    for (const auto& u : params.units) {
        std::string t_name = "t_" + u.name;
        auto tv = m.add_variable(t_name, model::VariableType::CONTINUOUS, 0.0, u.max_capacity);
        t_vars[u.name] = tv;
        m.add_objective_linear_term(tv, -u.variable_cost);

        std::string y_name = "y_" + u.name;
        auto yv = m.add_binary_variable(y_name);
        y_vars[u.name] = yv;
        m.add_objective_linear_term(yv, -u.fixed_cost);
    }

    // Constraints
    // A. Product mass balance: sum_c x[c, p] - prod[p] = 0
    for (const auto& p : params.products) {
        model::SparseLinearExpression expr;
        for (const auto& c : params.crudes) {
            expr.add_term(x_vars[{c.name, p.name}], 1.0);
        }
        expr.add_term(prod_vars[p.name], -1.0);
        m.add_constraint("bal_" + p.name, std::move(expr), model::ConstraintSense::EQUAL, 0.0);
    }

    // B. Crude availability: sum_p x[c, p] <= availability[c]
    for (const auto& c : params.crudes) {
        model::SparseLinearExpression expr;
        for (const auto& p : params.products) {
            expr.add_term(x_vars[{c.name, p.name}], 1.0);
        }
        m.add_constraint("avail_" + c.name, std::move(expr), model::ConstraintSense::LESS_EQUAL,
                         c.availability);
    }

    // C. Unit throughput definitions
    // U1 (CDU): processes sweet crudes A & B
    {
        model::SparseLinearExpression expr;
        for (const auto& p : params.products) {
            expr.add_term(x_vars[{"A", p.name}], 1.0);
            expr.add_term(x_vars[{"B", p.name}], 1.0);
        }
        expr.add_term(t_vars["U1"], -1.0);
        m.add_constraint("def_t_U1", std::move(expr), model::ConstraintSense::EQUAL, 0.0);
    }

    // U2 (VDU/FCCU): processes sour crudes C & D
    {
        model::SparseLinearExpression expr;
        for (const auto& p : params.products) {
            expr.add_term(x_vars[{"C", p.name}], 1.0);
            expr.add_term(x_vars[{"D", p.name}], 1.0);
        }
        expr.add_term(t_vars["U2"], -1.0);
        m.add_constraint("def_t_U2", std::move(expr), model::ConstraintSense::EQUAL, 0.0);
    }

    // U3 (HDS): desulfurizes sour crudes C & D routed to clean fuels P1 & P2
    {
        model::SparseLinearExpression expr;
        expr.add_term(x_vars[{"C", "P1"}], 1.0);
        expr.add_term(x_vars[{"C", "P2"}], 1.0);
        expr.add_term(x_vars[{"D", "P1"}], 1.0);
        expr.add_term(x_vars[{"D", "P2"}], 1.0);
        expr.add_term(t_vars["U3"], -1.0);
        m.add_constraint("def_t_U3", std::move(expr), model::ConstraintSense::EQUAL, 0.0);
    }

    // D. Unit capacity constraints (min turndown and max capacity linked to y[u])
    for (const auto& u : params.units) {
        // t[u] - max_cap * y[u] <= 0
        {
            model::SparseLinearExpression expr;
            expr.add_term(t_vars[u.name], 1.0);
            expr.add_term(y_vars[u.name], -u.max_capacity);
            m.add_constraint("cap_max_" + u.name, std::move(expr),
                             model::ConstraintSense::LESS_EQUAL, 0.0);
        }
        // t[u] - min_cap * y[u] >= 0
        {
            model::SparseLinearExpression expr;
            expr.add_term(t_vars[u.name], 1.0);
            expr.add_term(y_vars[u.name], -u.min_capacity);
            m.add_constraint("cap_min_" + u.name, std::move(expr),
                             model::ConstraintSense::GREATER_EQUAL, 0.0);
        }
    }

    // E. Product quality constraints (Linearized)
    for (const auto& p : params.products) {
        // Sulfur max: sum_c (sulfur[c] - max_sulfur[p]) * x[c, p] <= 0
        if (p.max_sulfur > 0.0) {
            model::SparseLinearExpression expr;
            for (const auto& c : params.crudes) {
                double coeff = c.sulfur - p.max_sulfur;
                if (std::abs(coeff) > 1e-12) {
                    expr.add_term(x_vars[{c.name, p.name}], coeff);
                }
            }
            m.add_constraint("q_sulfur_" + p.name, std::move(expr),
                             model::ConstraintSense::LESS_EQUAL, 0.0);
        }

        // Density max: sum_c (density[c] - max_density[p]) * x[c, p] <= 0
        if (p.max_density > 0.0) {
            model::SparseLinearExpression expr;
            for (const auto& c : params.crudes) {
                double coeff = c.density - p.max_density;
                if (std::abs(coeff) > 1e-12) {
                    expr.add_term(x_vars[{c.name, p.name}], coeff);
                }
            }
            m.add_constraint("q_dens_max_" + p.name, std::move(expr),
                             model::ConstraintSense::LESS_EQUAL, 0.0);
        }

        // Density min: sum_c (density[c] - min_density[p]) * x[c, p] >= 0
        if (p.min_density > 0.0) {
            model::SparseLinearExpression expr;
            for (const auto& c : params.crudes) {
                double coeff = c.density - p.min_density;
                if (std::abs(coeff) > 1e-12) {
                    expr.add_term(x_vars[{c.name, p.name}], coeff);
                }
            }
            m.add_constraint("q_dens_min_" + p.name, std::move(expr),
                             model::ConstraintSense::GREATER_EQUAL, 0.0);
        }
    }

    return m;
}

std::string MRPLModelGenerator::generate_lp_string(const MRPLParameters& params) {
    std::ostringstream ss;
    ss << "\\ ==============================================================================\n"
       << "\\ MRPL Industrial Case Study: Refinery Crude Blending and Unit Allocation\n"
       << "\\\n"
       << "\\ DATA STATUS:\n"
       << "\\ All numeric values in this demonstration are illustrative/synthetic.\n"
       << "\\ They are not proprietary MRPL operational data and must not be interpreted\n"
       << "\\ as actual MRPL process parameters.\n"
       << "\\ ==============================================================================\n\n"
       << "Maximize\n obj: ";

    // Objective
    bool first = true;
    for (const auto& p : params.products) {
        if (!first)
            ss << " + ";
        ss << p.price << " prod_" << p.name;
        first = false;
    }
    for (const auto& c : params.crudes) {
        for (const auto& p : params.products) {
            ss << " - " << c.cost << " x_" << c.name << "_" << p.name;
        }
    }
    for (const auto& u : params.units) {
        ss << " - " << u.variable_cost << " t_" << u.name;
    }
    for (const auto& u : params.units) {
        ss << " - " << u.fixed_cost << " y_" << u.name;
    }
    ss << "\n\nSubject To\n";

    // Mass balances
    ss << " \\ Product pool mass balance\n";
    for (const auto& p : params.products) {
        ss << " bal_" << p.name << ": ";
        for (std::size_t i = 0; i < params.crudes.size(); ++i) {
            if (i > 0)
                ss << " + ";
            ss << "x_" << params.crudes[i].name << "_" << p.name;
        }
        ss << " - prod_" << p.name << " = 0\n";
    }

    // Crude availability
    ss << "\n \\ Crude availability limits\n";
    for (const auto& c : params.crudes) {
        ss << " avail_" << c.name << ": ";
        for (std::size_t i = 0; i < params.products.size(); ++i) {
            if (i > 0)
                ss << " + ";
            ss << "x_" << c.name << "_" << params.products[i].name;
        }
        ss << " <= " << c.availability << "\n";
    }

    // Unit throughput definitions
    ss << "\n \\ Unit throughput definitions\n";
    ss << " def_t_U1: ";
    for (std::size_t i = 0; i < params.products.size(); ++i) {
        if (i > 0)
            ss << " + ";
        ss << "x_A_" << params.products[i].name << " + x_B_" << params.products[i].name;
    }
    ss << " - t_U1 = 0\n";

    ss << " def_t_U2: ";
    for (std::size_t i = 0; i < params.products.size(); ++i) {
        if (i > 0)
            ss << " + ";
        ss << "x_C_" << params.products[i].name << " + x_D_" << params.products[i].name;
    }
    ss << " - t_U2 = 0\n";

    ss << " def_t_U3: x_C_P1 + x_C_P2 + x_D_P1 + x_D_P2 - t_U3 = 0\n";

    // Unit capacity limits
    ss << "\n \\ Unit capacity limits (minimum turndown and maximum capacity)\n";
    for (const auto& u : params.units) {
        ss << " cap_max_" << u.name << ": t_" << u.name << " - " << u.max_capacity << " y_"
           << u.name << " <= 0\n";
        ss << " cap_min_" << u.name << ": t_" << u.name << " - " << u.min_capacity << " y_"
           << u.name << " >= 0\n";
    }

    // Product quality constraints
    ss << "\n \\ Product quality constraints (Linearized)\n";
    for (const auto& p : params.products) {
        if (p.max_sulfur > 0.0) {
            ss << " q_sulfur_" << p.name << ":";
            for (const auto& c : params.crudes) {
                double diff = c.sulfur - p.max_sulfur;
                if (diff >= 0.0) {
                    ss << " + " << diff << " x_" << c.name << "_" << p.name;
                } else {
                    ss << " - " << (-diff) << " x_" << c.name << "_" << p.name;
                }
            }
            ss << " <= 0\n";
        }

        if (p.max_density > 0.0) {
            ss << " q_dens_max_" << p.name << ":";
            for (const auto& c : params.crudes) {
                double diff = c.density - p.max_density;
                if (diff >= 0.0) {
                    ss << " + " << diff << " x_" << c.name << "_" << p.name;
                } else {
                    ss << " - " << (-diff) << " x_" << c.name << "_" << p.name;
                }
            }
            ss << " <= 0\n";
        }

        if (p.min_density > 0.0) {
            ss << " q_dens_min_" << p.name << ":";
            for (const auto& c : params.crudes) {
                double diff = c.density - p.min_density;
                if (diff >= 0.0) {
                    ss << " + " << diff << " x_" << c.name << "_" << p.name;
                } else {
                    ss << " - " << (-diff) << " x_" << c.name << "_" << p.name;
                }
            }
            ss << " >= 0\n";
        }
    }

    // Bounds
    ss << "\nBounds\n";
    for (const auto& p : params.products) {
        ss << " " << p.min_demand << " <= prod_" << p.name << " <= " << p.max_demand << "\n";
    }
    for (const auto& u : params.units) {
        ss << " 0 <= t_" << u.name << " <= " << u.max_capacity << "\n";
    }

    // Binaries
    ss << "\nBinaries\n";
    for (const auto& u : params.units) {
        ss << " y_" << u.name;
    }
    ss << "\n\nEnd\n";

    return ss.str();
}

void MRPLModelGenerator::save_lp_file(const MRPLParameters& params, const std::string& filepath) {
    std::filesystem::path p(filepath);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open LP file for writing: " + filepath);
    }
    ofs << generate_lp_string(params);
}

// ============================================================================
// Case Study Runner & Analysis Implementation
// ============================================================================

MRPLCaseStudyRunner::MRPLCaseStudyRunner(core::SolverTolerances tolerances)
    : tolerances_(std::move(tolerances)) {}

MRPLSolutionBreakdown MRPLCaseStudyRunner::solve_and_analyze(const MRPLParameters& params,
                                                             std::string scenario_name,
                                                             bool run_highs_comparison) const {
    MRPLSolutionBreakdown res;
    res.scenario_name = std::move(scenario_name);

    auto model = MRPLModelGenerator::build_canonical_model(params);

    // 1. Solve LP Relaxation (Relax all integer/binary variables to continuous)
    {
        model::CanonicalModel relaxed_model(model);
        for (std::size_t j = 0; j < relaxed_model.num_variables(); ++j) {
            auto v = static_cast<model::VarId>(j);
            if (relaxed_model.variable(v).is_binary()) {
                relaxed_model.set_variable_type(v, model::VariableType::CONTINUOUS);
                relaxed_model.set_variable_bounds(v, 0.0, 1.0);
            } else if (relaxed_model.variable(v).is_integer()) {
                relaxed_model.set_variable_type(v, model::VariableType::CONTINUOUS);
            }
        }

        auto start_lp = std::chrono::steady_clock::now();
        algorithms::lp::RevisedSimplex lp_solver;
        auto lp_sol = lp_solver.solve(relaxed_model, tolerances_);
        auto end_lp = std::chrono::steady_clock::now();
        res.lp_relaxation_time_seconds = std::chrono::duration<double>(end_lp - start_lp).count();

        if (lp_sol.is_optimal()) {
            res.lp_relaxation_objective = lp_sol.objective_value;
            for (std::size_t j = 0; j < relaxed_model.num_variables(); ++j) {
                const auto& var_name = relaxed_model.variable(static_cast<model::VarId>(j)).name;
                if (var_name.rfind("y_", 0) == 0) {
                    double val = lp_sol.primal_solution[j];
                    if (val > 1e-4 && val < (1.0 - 1e-4)) {
                        std::ostringstream fss;
                        fss << var_name << " = " << std::fixed << std::setprecision(4) << val;
                        res.lp_fractional_variables.push_back(fss.str());
                    }
                }
            }
        }
    }

    // 2. Solve Full MILP with Production Branch-and-Bound
    algorithms::milp::MILPSolverOptions opts;
    opts.tolerances = tolerances_;
    opts.enable_warm_starts = true;
    algorithms::milp::BranchAndBound milp_solver(opts);

    auto start_milp = std::chrono::steady_clock::now();
    auto milp_sol = milp_solver.solve(model, tolerances_);
    auto end_milp = std::chrono::steady_clock::now();

    res.status = milp_sol.status;
    res.objective = milp_sol.best_objective;
    res.best_bound = milp_sol.best_bound;
    res.optimality_gap = milp_sol.optimality_gap;
    res.nodes_explored = milp_sol.telemetry.nodes_explored;
    res.solve_time_seconds = std::chrono::duration<double>(end_milp - start_milp).count();

    if (milp_sol.is_optimal() && !milp_sol.incumbent_solution.empty()) {
        // 3. Independent Verification Layer
        verification::MILPVerifier verifier;
        res.verification_report = verifier.verify(
            model, milp_sol.incumbent_solution, milp_sol.best_bound, milp_sol.best_objective,
            milp_sol.status, tolerances_, milp_sol.optimality_gap);
        res.verified_optimal = res.verification_report.fully_verified();

        // 4. Extract Solution Vectors by Name
        std::unordered_map<std::string, double> var_map;
        for (std::size_t j = 0; j < model.num_variables(); ++j) {
            var_map[model.variable(static_cast<model::VarId>(j)).name] =
                milp_sol.incumbent_solution[j];
        }

        for (const auto& c : params.crudes) {
            for (const auto& p : params.products) {
                std::string k = "x_" + c.name + "_" + p.name;
                res.blend_flows[{c.name, p.name}] = var_map[k];
            }
        }
        for (const auto& p : params.products) {
            res.product_production[p.name] = var_map["prod_" + p.name];
        }
        for (const auto& u : params.units) {
            res.unit_throughputs[u.name] = var_map["t_" + u.name];
            res.unit_binaries[u.name] = var_map["y_" + u.name];
        }

        // 5. Domain Mass Balance Validation
        double max_crude_res = 0.0;
        for (const auto& c : params.crudes) {
            double total_c = 0.0;
            for (const auto& p : params.products) {
                total_c += res.blend_flows[{c.name, p.name}];
            }
            double viol = total_c - c.availability;
            if (viol > max_crude_res)
                max_crude_res = viol;
        }

        double max_prod_res = 0.0;
        for (const auto& p : params.products) {
            double total_p = 0.0;
            for (const auto& c : params.crudes) {
                total_p += res.blend_flows[{c.name, p.name}];
            }
            double diff = std::abs(total_p - res.product_production[p.name]);
            if (diff > max_prod_res)
                max_prod_res = diff;
        }

        double max_unit_t_res = 0.0;
        double u1_flow = 0.0;
        double u2_flow = 0.0;
        double u3_flow = 0.0;
        for (const auto& p : params.products) {
            u1_flow += res.blend_flows[{"A", p.name}] + res.blend_flows[{"B", p.name}];
            u2_flow += res.blend_flows[{"C", p.name}] + res.blend_flows[{"D", p.name}];
        }
        u3_flow = res.blend_flows[{"C", "P1"}] + res.blend_flows[{"C", "P2"}] +
                  res.blend_flows[{"D", "P1"}] + res.blend_flows[{"D", "P2"}];

        max_unit_t_res = std::max(max_unit_t_res, std::abs(res.unit_throughputs["U1"] - u1_flow));
        max_unit_t_res = std::max(max_unit_t_res, std::abs(res.unit_throughputs["U2"] - u2_flow));
        max_unit_t_res = std::max(max_unit_t_res, std::abs(res.unit_throughputs["U3"] - u3_flow));

        double max_unit_cap_res = 0.0;
        for (const auto& u : params.units) {
            double t = res.unit_throughputs[u.name];
            double y = res.unit_binaries[u.name];
            if (t > u.max_capacity * y + 1e-6) {
                max_unit_cap_res = std::max(max_unit_cap_res, t - u.max_capacity * y);
            }
            if (y > 0.5 && t < u.min_capacity * y - 1e-6) {
                max_unit_cap_res = std::max(max_unit_cap_res, u.min_capacity * y - t);
            }
        }

        res.mass_balance.crude_availability_residual = max_crude_res;
        res.mass_balance.product_balance_residual = max_prod_res;
        res.mass_balance.unit_throughput_residual = max_unit_t_res;
        res.mass_balance.unit_capacity_residual = max_unit_cap_res;
        res.mass_balance.max_absolute_residual =
            std::max({max_crude_res, max_prod_res, max_unit_t_res, max_unit_cap_res});
        res.mass_balance.passed = res.mass_balance.max_absolute_residual <= 1e-5;
        res.mass_balance.diagnostic =
            res.mass_balance.passed
                ? "All domain mass balance and capacity limits verified within numerical tolerance."
                : "Mass balance discrepancy detected.";

        // 6. Domain Quality Validation
        for (const auto& p : params.products) {
            ProductQualityEvaluation qeval;
            qeval.product_name = p.name + " (" + p.description + ")";
            qeval.production_kbpd = res.product_production[p.name];

            if (qeval.production_kbpd > 1e-6) {
                double total_sulfur = 0.0;
                double total_density = 0.0;
                for (const auto& c : params.crudes) {
                    double f = res.blend_flows[{c.name, p.name}];
                    total_sulfur += c.sulfur * f;
                    total_density += c.density * f;
                }
                qeval.calculated_sulfur_wt_pct = total_sulfur / qeval.production_kbpd;
                qeval.calculated_density_kg_m3 = total_density / qeval.production_kbpd;

                qeval.sulfur_limit_wt_pct = p.max_sulfur;
                qeval.sulfur_margin_wt_pct = p.max_sulfur - qeval.calculated_sulfur_wt_pct;

                if (p.max_density > 0.0) {
                    qeval.density_is_upper_limit = true;
                    qeval.density_limit_kg_m3 = p.max_density;
                    qeval.density_margin_kg_m3 = p.max_density - qeval.calculated_density_kg_m3;
                } else {
                    qeval.density_is_upper_limit = false;
                    qeval.density_limit_kg_m3 = p.min_density;
                    qeval.density_margin_kg_m3 = qeval.calculated_density_kg_m3 - p.min_density;
                }

                qeval.quality_satisfied =
                    (qeval.sulfur_margin_wt_pct >= -1e-6) && (qeval.density_margin_kg_m3 >= -1e-5);
            }
            res.product_qualities.push_back(qeval);
        }

        // 7. Economic Decomposition
        double gross_rev = 0.0;
        for (const auto& p : params.products) {
            gross_rev += p.price * res.product_production[p.name];
        }

        double crude_cost = 0.0;
        for (const auto& c : params.crudes) {
            for (const auto& p : params.products) {
                crude_cost += c.cost * res.blend_flows[{c.name, p.name}];
            }
        }

        double var_cost = 0.0;
        double fix_cost = 0.0;
        for (const auto& u : params.units) {
            var_cost += u.variable_cost * res.unit_throughputs[u.name];
            fix_cost += u.fixed_cost * res.unit_binaries[u.name];
        }

        res.economics.gross_revenue_k_usd = gross_rev;
        res.economics.crude_purchase_cost_k_usd = crude_cost;
        res.economics.unit_variable_cost_k_usd = var_cost;
        res.economics.unit_fixed_cost_k_usd = fix_cost;
        res.economics.total_cost_k_usd = crude_cost + var_cost + fix_cost;
        res.economics.net_profit_k_usd = gross_rev - res.economics.total_cost_k_usd;

        // 8. Unit Operational Decisions
        for (const auto& u : params.units) {
            UnitOperationalStatus ustat;
            ustat.unit_name = u.name;
            ustat.description = u.description;
            ustat.is_active = (res.unit_binaries[u.name] > 0.5);
            ustat.throughput_kbpd = res.unit_throughputs[u.name];
            ustat.min_capacity_kbpd = u.min_capacity;
            ustat.max_capacity_kbpd = u.max_capacity;
            ustat.utilization_pct =
                ustat.is_active ? (ustat.throughput_kbpd / u.max_capacity * 100.0) : 0.0;

            std::ostringstream oss;
            if (ustat.is_active) {
                oss << u.name << " ACTIVE: throughput " << std::fixed << std::setprecision(1)
                    << ustat.throughput_kbpd << " kbpd (" << ustat.utilization_pct << "% load)";
            } else {
                oss << u.name << " INACTIVE (0 kbpd)";
            }
            ustat.operational_summary = oss.str();
            res.unit_decisions.push_back(ustat);
        }
    }

    // 9. Arms-Length HiGHS Subprocess Comparison
    if (run_highs_comparison) {
        benchmark::ExternalReferenceAdapter adapter;
        if (adapter.is_available()) {
            std::string temp_lp = "/tmp/kavach_mrpl_" + res.scenario_name + ".lp";
            MRPLModelGenerator::save_lp_file(params, temp_lp);
            res.highs_result = adapter.run(temp_lp, 15.0);
            std::filesystem::remove(temp_lp);
        } else {
            res.highs_result.status = benchmark::ReferenceSolverStatus::UNAVAILABLE;
            res.highs_result.solver_name = "HiGHS";
            res.highs_result.error_message = "HiGHS binary not found on host system";
        }
    }

    return res;
}

std::vector<MRPLSolutionBreakdown> MRPLCaseStudyRunner::run_all_scenarios(
    bool run_highs_comparison) const {
    std::vector<MRPLSolutionBreakdown> list;
    list.push_back(
        solve_and_analyze(MRPLParameters::default_parameters(), "baseline", run_highs_comparison));
    list.push_back(solve_and_analyze(MRPLParameters::scenario_reduced_crude(),
                                     "scenario_a_reduced_crude", run_highs_comparison));
    list.push_back(solve_and_analyze(MRPLParameters::scenario_tight_quality(),
                                     "scenario_b_tight_quality", run_highs_comparison));
    list.push_back(solve_and_analyze(MRPLParameters::scenario_reduced_capacity(),
                                     "scenario_c_reduced_capacity", run_highs_comparison));
    list.push_back(solve_and_analyze(MRPLParameters::scenario_infeasible(), "scenario_d_infeasible",
                                     run_highs_comparison));
    return list;
}

benchmark::JsonValue MRPLSolutionBreakdown::to_json() const {
    benchmark::JsonValue::Object obj;
    obj["scenario"] = scenario_name;
    obj["data_status"] = std::string(SYNTHETIC_DATA_DISCLOSURE);
    obj["solver_status"] = core::to_string(status);
    obj["verified_optimal"] = verified_optimal;
    obj["objective_net_margin_k_usd"] = objective;
    obj["lp_relaxation_objective"] = lp_relaxation_objective;
    obj["best_bound"] = best_bound;
    obj["optimality_gap"] = optimality_gap;
    obj["nodes_explored"] = static_cast<double>(nodes_explored);
    obj["solve_time_seconds"] = solve_time_seconds;
    obj["lp_relaxation_time_seconds"] = lp_relaxation_time_seconds;

    benchmark::JsonValue::Array frac_vars;
    for (const auto& f : lp_fractional_variables) {
        frac_vars.push_back(f);
    }
    obj["lp_fractional_variables"] = std::move(frac_vars);

    if (status == core::SolverStatus::OPTIMAL) {
        // Economics
        benchmark::JsonValue::Object econ;
        econ["gross_revenue_k_usd"] = economics.gross_revenue_k_usd;
        econ["crude_purchase_cost_k_usd"] = economics.crude_purchase_cost_k_usd;
        econ["unit_variable_cost_k_usd"] = economics.unit_variable_cost_k_usd;
        econ["unit_fixed_cost_k_usd"] = economics.unit_fixed_cost_k_usd;
        econ["total_cost_k_usd"] = economics.total_cost_k_usd;
        econ["net_profit_k_usd"] = economics.net_profit_k_usd;
        obj["economics"] = std::move(econ);

        // Mass balance
        benchmark::JsonValue::Object mb;
        mb["max_absolute_residual"] = mass_balance.max_absolute_residual;
        mb["crude_availability_residual"] = mass_balance.crude_availability_residual;
        mb["product_balance_residual"] = mass_balance.product_balance_residual;
        mb["unit_throughput_residual"] = mass_balance.unit_throughput_residual;
        mb["unit_capacity_residual"] = mass_balance.unit_capacity_residual;
        mb["passed"] = mass_balance.passed;
        obj["mass_balance_validation"] = std::move(mb);

        // Production
        benchmark::JsonValue::Object prod;
        for (const auto& [p, val] : product_production) {
            prod[p] = val;
        }
        obj["production_kbpd"] = std::move(prod);

        // Blending matrix
        benchmark::JsonValue::Object blend;
        for (const auto& [pair, val] : blend_flows) {
            std::string key = pair.first + "_to_" + pair.second;
            blend[key] = val;
        }
        obj["blending_matrix_kbpd"] = std::move(blend);

        // Units
        benchmark::JsonValue::Array uarr;
        for (const auto& u : unit_decisions) {
            benchmark::JsonValue::Object uobj;
            uobj["unit"] = u.unit_name;
            uobj["description"] = u.description;
            uobj["is_active"] = u.is_active;
            uobj["throughput_kbpd"] = u.throughput_kbpd;
            uobj["utilization_pct"] = u.utilization_pct;
            uobj["summary"] = u.operational_summary;
            uarr.push_back(std::move(uobj));
        }
        obj["unit_decisions"] = std::move(uarr);

        // Qualities
        benchmark::JsonValue::Array qarr;
        for (const auto& q : product_qualities) {
            benchmark::JsonValue::Object qobj;
            qobj["product"] = q.product_name;
            qobj["production_kbpd"] = q.production_kbpd;
            qobj["calculated_sulfur_wt_pct"] = q.calculated_sulfur_wt_pct;
            qobj["sulfur_limit_wt_pct"] = q.sulfur_limit_wt_pct;
            qobj["sulfur_margin_wt_pct"] = q.sulfur_margin_wt_pct;
            qobj["calculated_density_kg_m3"] = q.calculated_density_kg_m3;
            qobj["density_limit_kg_m3"] = q.density_limit_kg_m3;
            qobj["density_margin_kg_m3"] = q.density_margin_kg_m3;
            qobj["quality_satisfied"] = q.quality_satisfied;
            qarr.push_back(std::move(qobj));
        }
        obj["product_qualities"] = std::move(qarr);
    }

    // HiGHS Comparison
    benchmark::JsonValue::Object highs;
    highs["status"] = benchmark::to_string(highs_result.status);
    if (highs_result.objective) {
        highs["objective"] = *highs_result.objective;
    }
    if (highs_result.runtime_seconds) {
        highs["runtime_seconds"] = *highs_result.runtime_seconds;
    }
    if (!highs_result.error_message.empty()) {
        highs["message"] = highs_result.error_message;
    }
    obj["reference_highs"] = std::move(highs);

    return benchmark::JsonValue(std::move(obj));
}

void MRPLCaseStudyRunner::export_all_artifacts(const std::string& results_dir,
                                               const std::string& report_filepath) const {
    std::filesystem::create_directories(results_dir);

    auto scenarios = run_all_scenarios(true);

    // 1. Export baseline.json
    {
        std::ofstream ofs(results_dir + "/baseline.json");
        if (ofs.is_open()) {
            ofs << scenarios[0].to_json().dump(2);
        }
    }

    // 2. Export scenarios.json
    {
        benchmark::JsonValue::Array s_arr;
        for (const auto& s : scenarios) {
            s_arr.push_back(s.to_json());
        }
        std::ofstream ofs(results_dir + "/scenarios.json");
        if (ofs.is_open()) {
            ofs << benchmark::JsonValue(std::move(s_arr)).dump(2);
        }
    }

    // 3. Export comparison.json
    {
        benchmark::JsonValue::Object comp;
        comp["note"] = "Kavach-Opt vs Reference Solver Side-by-Side Comparison";
        benchmark::JsonValue::Array runs;
        for (const auto& s : scenarios) {
            benchmark::JsonValue::Object r;
            r["scenario"] = s.scenario_name;
            r["kavach_status"] = core::to_string(s.status);
            r["kavach_objective"] = s.objective;
            r["kavach_runtime_sec"] = s.solve_time_seconds;
            r["highs_status"] = benchmark::to_string(s.highs_result.status);
            if (s.highs_result.objective) {
                r["highs_objective"] = *s.highs_result.objective;
            }
            if (s.highs_result.runtime_seconds) {
                r["highs_runtime_sec"] = *s.highs_result.runtime_seconds;
            }
            runs.push_back(std::move(r));
        }
        comp["runs"] = std::move(runs);
        std::ofstream ofs(results_dir + "/comparison.json");
        if (ofs.is_open()) {
            ofs << benchmark::JsonValue(std::move(comp)).dump(2);
        }
    }

    // 4. Export Markdown report
    {
        std::filesystem::path rpath(report_filepath);
        if (rpath.has_parent_path()) {
            std::filesystem::create_directories(rpath.parent_path());
        }
        std::ofstream ofs(report_filepath);
        if (ofs.is_open()) {
            ofs << generate_markdown_report(scenarios);
        }
    }
}

std::string MRPLCaseStudyRunner::generate_markdown_report(
    const std::vector<MRPLSolutionBreakdown>& scenarios) {
    if (scenarios.empty())
        return "";
    const auto& base = scenarios[0];

    std::ostringstream ss;
    ss << "# MRPL-Style Refinery Optimization Case Study\n\n"
       << "## 1. Data disclosure\n\n"
       << "> [!IMPORTANT]\n"
       << "> **DATA STATUS**:\n"
       << "> All numeric values in this demonstration are illustrative/synthetic.\n"
       << "> They are not proprietary MRPL operational data and must not be interpreted\n"
       << "> as actual MRPL process parameters.\n\n"
       << "## 2. Business problem\n\n"
       << "Modern petroleum refineries balance crude feedstock purchasing, unit throughput limits, "
       << "and blending constraints to produce finished transportation fuels (Gasoline, Diesel) "
          "and "
       << "heavy fuel oil while maximizing net refining margins. Operating units exhibit minimum "
          "turndown "
       << "limits and high fixed activation/maintenance costs, requiring genuine mixed-integer "
          "linear programming (MILP) "
       << "to determine optimal on/off unit configurations alongside continuous blending "
          "ratios.\n\n"
       << "## 3. Mathematical model\n\n"
       << "The refinery optimization model integrates:\n"
       << "- **Continuous decision variables** $x[c, p] \\ge 0$ for blending crude $c \\in \\{A, "
          "B, C, D\\}$ into product pool $p \\in \\{P1, P2, P3\\}$.\n"
       << "- **Continuous production variables** $prod[p] \\ge 0$ representing total finished "
          "product yield.\n"
       << "- **Continuous unit throughputs** $t[u] \\ge 0$ for processing units $u \\in \\{U1, U2, "
          "U3\\}$.\n"
       << "- **Binary activation decisions** $y[u] \\in \\{0, 1\\}$ indicating unit online "
          "status.\n\n"
       << "### Objective Function\n"
       << "$$\\max \\quad \\sum_{p} \\text{Price}_p \\cdot prod_p - \\sum_{c,p} \\text{Cost}_c "
          "\\cdot x_{c,p} - \\sum_u \\left( \\text{VarCost}_u \\cdot t_u + \\text{FixedCost}_u "
          "\\cdot y_u \\right)$$\n\n"
       << "### Key Constraints\n"
       << "1. **Product Mass Balance**: $\\sum_c x_{c,p} - prod_p = 0, \\quad \\forall p$\n"
       << "2. **Crude Availability**: $\\sum_p x_{c,p} \\le \\text{Avail}_c, \\quad \\forall c$\n"
       << "3. **Unit Throughput Linkage**:\n"
       << "   - $t_{U1} = \\sum_p (x_{A,p} + x_{B,p})$ (CDU Sweet Crudes)\n"
       << "   - $t_{U2} = \\sum_p (x_{C,p} + x_{D,p})$ (VDU/FCCU Sour Crudes)\n"
       << "   - $t_{U3} = x_{C,P1} + x_{C,P2} + x_{D,P1} + x_{D,P2}$ (HDS Clean-fuel "
          "Desulfurization)\n"
       << "4. **Unit Capacity & Turndown**: $\\text{MinCap}_u \\cdot y_u \\le t_u \\le "
          "\\text{MaxCap}_u \\cdot y_u, \\quad \\forall u$\n"
       << "5. **Linearized Quality Specifications**:\n"
       << "   $$\\sum_c (\\text{Quality}_{c,q} - \\text{Limit}_{p,q}) \\cdot x_{c,p} \\le 0$$\n\n"
       << "## 4. Model dimensions\n\n"
       << "| Dimension | Count |\n"
       << "|---|---|\n"
       << "| Total Variables | 21 |\n"
       << "| Continuous Variables | 18 |\n"
       << "| Binary Variables | 3 |\n"
       << "| Linear Constraints | 22 |\n"
       << "| Matrix Nonzeros | 81 |\n\n"
       << "## 5. Solver pipeline\n\n"
       << "```mermaid\n"
       << "flowchart LR\n"
       << "    A[\"Native LP Parser\"] --> B[\"CanonicalModel\"]\n"
       << "    B --> C[\"Presolve\"]\n"
       << "    C --> D[\"Root LP Relaxation\"]\n"
       << "    D --> E[\"MILP Branch-and-Bound\"]\n"
       << "    E --> F[\"Independent MILPVerifier\"]\n"
       << "    F --> G[\"Certified Solution\"]\n"
       << "```\n\n"
       << "## 6. Baseline solution\n\n"
       << "- **Status**: `" << core::to_string(base.status) << "` (Certified: `"
       << (base.verified_optimal ? "VERIFIED_OPTIMAL" : "FAILED") << "`)\n"
       << "- **Net Objective (Margin)**: $" << std::fixed << std::setprecision(2) << base.objective
       << " k/day\n"
       << "- **LP Relaxation Bound**: $" << base.lp_relaxation_objective << " k/day\n"
       << "- **Optimality Gap**: " << (base.optimality_gap * 100.0) << "%\n"
       << "- **Nodes Explored**: " << base.nodes_explored << "\n"
       << "- **Solve Runtime**: " << std::setprecision(4) << base.solve_time_seconds << " s\n\n"
       << "## 7. Blending plan\n\n"
       << "| Feed Stream | To Product P1 (Gasoline) | To Product P2 (Diesel) | To Product P3 (Fuel "
          "Oil) | Total Used (kbpd) |\n"
       << "|---|---|---|---|---|\n";

    for (const auto& c : {"A", "B", "C", "D"}) {
        double f1 = base.blend_flows.count({c, "P1"}) ? base.blend_flows.at({c, "P1"}) : 0.0;
        double f2 = base.blend_flows.count({c, "P2"}) ? base.blend_flows.at({c, "P2"}) : 0.0;
        double f3 = base.blend_flows.count({c, "P3"}) ? base.blend_flows.at({c, "P3"}) : 0.0;
        ss << "| Crude " << c << " | " << std::fixed << std::setprecision(1) << f1 << " | " << f2
           << " | " << f3 << " | " << (f1 + f2 + f3) << " |\n";
    }

    ss << "\n## 8. Unit allocation\n\n"
       << "| Unit | Description | Status | Throughput (kbpd) | Min-Max Rating | Utilization |\n"
       << "|---|---|---|---|---|---|\n";
    for (const auto& u : base.unit_decisions) {
        ss << "| " << u.unit_name << " | " << u.description << " | "
           << (u.is_active ? "**ACTIVE**" : "OFFLINE") << " | " << std::fixed
           << std::setprecision(1) << u.throughput_kbpd << " | " << u.min_capacity_kbpd << " - "
           << u.max_capacity_kbpd << " | " << u.utilization_pct << "% |\n";
    }

    ss << "\n## 9. Product quality\n\n"
       << "| Product | Production (kbpd) | Sulfur (wt%) | Sulfur Limit | Sulfur Margin | Density "
          "(kg/m³) | Density Limit | Density Margin |\n"
       << "|---|---|---|---|---|---|---|---|\n";
    for (const auto& q : base.product_qualities) {
        ss << "| " << q.product_name << " | " << std::fixed << std::setprecision(1)
           << q.production_kbpd << " | " << std::setprecision(2) << q.calculated_sulfur_wt_pct
           << " | " << q.sulfur_limit_wt_pct << " | +" << q.sulfur_margin_wt_pct << " | "
           << std::setprecision(1) << q.calculated_density_kg_m3 << " | " << q.density_limit_kg_m3
           << " | +" << q.density_margin_kg_m3 << " |\n";
    }

    ss << "\n## 10. Economics\n\n"
       << "- **Gross Product Sales Revenue**: $" << std::fixed << std::setprecision(2)
       << base.economics.gross_revenue_k_usd << " k/day\n"
       << "- **Crude Feedstock Acquisition Cost**: -$" << base.economics.crude_purchase_cost_k_usd
       << " k/day\n"
       << "- **Unit Variable Operating Cost**: -$" << base.economics.unit_variable_cost_k_usd
       << " k/day\n"
       << "- **Unit Fixed Activation Cost**: -$" << base.economics.unit_fixed_cost_k_usd
       << " k/day\n"
       << "- **Net Operational Margin**: **$" << base.economics.net_profit_k_usd << " k/day**\n\n"
       << "## 11. Independent verification\n\n"
       << "Independent mathematical certification performed by `MILPVerifier`:\n"
       << "- **Aggregate Primal Residual**: " << std::scientific
       << base.verification_report.aggregate_primal_residual << "\n"
       << "- **Max Bound Violation**: " << base.verification_report.max_bound_violation << "\n"
       << "- **Max Integrality Violation**: " << base.verification_report.max_integrality_violation
       << "\n"
       << "- **Domain Mass Balance Residual**: " << base.mass_balance.max_absolute_residual
       << " (Status: " << (base.mass_balance.passed ? "PASSED" : "FAILED") << ")\n"
       << "- **Certified Status**: `" << core::to_string(base.verification_report.certified_status)
       << "`\n\n"
       << "## 12. What-if scenarios\n\n"
       << "| Scenario ID | Description | Status | Net Margin ($k/day) | $\\Delta$ Objective | "
          "Nodes | Primary Operational Impact |\n"
       << "|---|---|---|---|---|---|---|\n";

    for (const auto& s : scenarios) {
        double delta = s.objective - base.objective;
        std::string impact = "-";
        if (s.scenario_name == "baseline") {
            impact = "Optimal baseline dispatch";
        } else if (s.scenario_name == "scenario_a_reduced_crude") {
            impact = "Sweet crude shortfall drops Gasoline production";
        } else if (s.scenario_name == "scenario_b_tight_quality") {
            impact = "Tightened diesel sulfur specs shift crudes to hydrotreater";
        } else if (s.scenario_name == "scenario_c_reduced_capacity") {
            impact = "HDS bottleneck limits clean fuel output";
        } else if (s.scenario_name == "scenario_d_infeasible") {
            impact = "Feedstock starvation causes integer infeasibility";
        }

        ss << "| `" << s.scenario_name << "` | " << s.scenario_name << " | `"
           << core::to_string(s.status) << "` | " << std::fixed << std::setprecision(2)
           << (s.status == core::SolverStatus::OPTIMAL ? std::to_string(s.objective) : "N/A")
           << " | "
           << (s.status == core::SolverStatus::OPTIMAL
                   ? (delta >= 0 ? "+" : "") + std::to_string(delta)
                   : "N/A")
           << " | " << s.nodes_explored << " | " << impact << " |\n";
    }

    ss << "\n## 13. HiGHS comparison\n\n"
       << "| Scenario | Kavach-Opt Obj | HiGHS Obj | Rel Error | Kavach Time | HiGHS Status |\n"
       << "|---|---|---|---|---|---|\n";

    for (const auto& s : scenarios) {
        std::string h_obj =
            s.highs_result.objective ? std::to_string(*s.highs_result.objective) : "N/A";
        std::string rel_err = "N/A";
        if (s.highs_result.objective && s.status == core::SolverStatus::OPTIMAL) {
            double err = std::abs(s.objective - *s.highs_result.objective) /
                         std::max(1.0, std::abs(s.objective));
            std::ostringstream ess;
            ess << std::scientific << std::setprecision(2) << err;
            rel_err = ess.str();
        }
        ss << "| `" << s.scenario_name << "` | "
           << (s.status == core::SolverStatus::OPTIMAL ? std::to_string(s.objective) : "N/A")
           << " | " << h_obj << " | " << rel_err << " | " << std::fixed << std::setprecision(4)
           << s.solve_time_seconds << " s | " << benchmark::to_string(s.highs_result.status)
           << " |\n";
    }

    ss << "\n## 14. Limitations\n\n"
       << "- This demonstration utilizes synthetic illustrative parameters and does not represent "
          "actual proprietary operational data of Mangalore Refinery and Petrochemicals Limited "
          "(MRPL).\n"
       << "- Refinery units are modeled as semi-continuous linearized yield and throughput units "
          "rather than non-linear chemical equilibrium thermodynamic equations.\n"
       << "- Multi-period tankage inventory dynamics and crude vessel arrival logistics are "
          "omitted to maintain focus on the single-period MILP blending and capacity allocation "
          "core.\n";

    return ss.str();
}

}  // namespace kavach::case_study::mrpl
