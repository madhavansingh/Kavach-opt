#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "../test_framework.hpp"
#include "case_study/mrpl/mrpl_model.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "io/parser.hpp"
#include "verification/milp_verifier.hpp"

using namespace kavach;
using namespace kavach::case_study::mrpl;

// ============================================================================
// A. Deterministic Model Generation
// ============================================================================

KAVACH_TEST(test_mrpl_a_deterministic_generation) {
    auto p1 = MRPLParameters::default_parameters();
    auto p2 = MRPLParameters::default_parameters();

    auto m1 = MRPLModelGenerator::build_canonical_model(p1);
    auto m2 = MRPLModelGenerator::build_canonical_model(p2);

    KAVACH_ASSERT_EQUAL(m1.num_variables(), m2.num_variables());
    KAVACH_ASSERT_EQUAL(m1.num_constraints(), m2.num_constraints());
    KAVACH_ASSERT_EQUAL(m1.num_nonzeros(), m2.num_nonzeros());

    for (std::size_t i = 0; i < m1.num_variables(); ++i) {
        auto v = static_cast<model::VarId>(i);
        KAVACH_ASSERT_EQUAL(m1.variable(v).name, m2.variable(v).name);
        KAVACH_ASSERT_EQUAL(m1.variable(v).type, m2.variable(v).type);
        KAVACH_ASSERT_NEAR(m1.variable(v).lower_bound, m2.variable(v).lower_bound, 1e-12);
        KAVACH_ASSERT_NEAR(m1.variable(v).upper_bound, m2.variable(v).upper_bound, 1e-12);
    }
}

// ============================================================================
// B. Variable Count
// ============================================================================

KAVACH_TEST(test_mrpl_b_variable_count) {
    auto p = MRPLParameters::default_parameters();
    auto m = MRPLModelGenerator::build_canonical_model(p);

    // 12 blending + 3 prod + 3 throughput + 3 binary = 21
    KAVACH_ASSERT_EQUAL(m.num_variables(), 21);

    std::size_t n_cont = 0;
    std::size_t n_bin = 0;
    for (std::size_t i = 0; i < m.num_variables(); ++i) {
        if (m.variable(static_cast<model::VarId>(i)).is_continuous())
            ++n_cont;
        if (m.variable(static_cast<model::VarId>(i)).is_binary())
            ++n_bin;
    }
    KAVACH_ASSERT_EQUAL(n_cont, 18);
    KAVACH_ASSERT_EQUAL(n_bin, 3);
}

// ============================================================================
// C. Constraint Count
// ============================================================================

KAVACH_TEST(test_mrpl_c_constraint_count) {
    auto p = MRPLParameters::default_parameters();
    auto m = MRPLModelGenerator::build_canonical_model(p);

    // 3 bal + 4 avail + 3 def_t + 6 cap + 3 sulfur + 3 dens = 22
    KAVACH_ASSERT_EQUAL(m.num_constraints(), 22);
}

// ============================================================================
// D. Binary Variable Declaration
// ============================================================================

KAVACH_TEST(test_mrpl_d_binary_declaration) {
    auto p = MRPLParameters::default_parameters();
    auto m = MRPLModelGenerator::build_canonical_model(p);

    bool found_u1 = false, found_u2 = false, found_u3 = false;
    for (std::size_t i = 0; i < m.num_variables(); ++i) {
        const auto& v = m.variable(static_cast<model::VarId>(i));
        if (v.name == "y_U1") {
            found_u1 = true;
            KAVACH_ASSERT(v.is_binary());
            KAVACH_ASSERT_NEAR(v.lower_bound, 0.0, 1e-12);
            KAVACH_ASSERT_NEAR(v.upper_bound, 1.0, 1e-12);
        } else if (v.name == "y_U2") {
            found_u2 = true;
            KAVACH_ASSERT(v.is_binary());
        } else if (v.name == "y_U3") {
            found_u3 = true;
            KAVACH_ASSERT(v.is_binary());
        }
    }
    KAVACH_ASSERT(found_u1 && found_u2 && found_u3);
}

// ============================================================================
// E. Crude Availability Constraints
// ============================================================================

KAVACH_TEST(test_mrpl_e_crude_availability) {
    auto p = MRPLParameters::default_parameters();
    auto m = MRPLModelGenerator::build_canonical_model(p);

    bool found_a = false, found_d = false;
    for (std::size_t i = 0; i < m.num_constraints(); ++i) {
        const auto& c = m.constraint(static_cast<model::ConstrId>(i));
        if (c.name == "avail_A") {
            found_a = true;
            KAVACH_ASSERT_EQUAL(c.sense, model::ConstraintSense::LESS_EQUAL);
            KAVACH_ASSERT_NEAR(c.rhs, 120.0, 1e-12);
        } else if (c.name == "avail_D") {
            found_d = true;
            KAVACH_ASSERT_NEAR(c.rhs, 80.0, 1e-12);
        }
    }
    KAVACH_ASSERT(found_a && found_d);
}

// ============================================================================
// F. Product Mass Balance
// ============================================================================

KAVACH_TEST(test_mrpl_f_product_mass_balance) {
    auto p = MRPLParameters::default_parameters();
    auto m = MRPLModelGenerator::build_canonical_model(p);

    bool found_p1 = false;
    for (std::size_t i = 0; i < m.num_constraints(); ++i) {
        const auto& c = m.constraint(static_cast<model::ConstrId>(i));
        if (c.name == "bal_P1") {
            found_p1 = true;
            KAVACH_ASSERT_EQUAL(c.sense, model::ConstraintSense::EQUAL);
            KAVACH_ASSERT_NEAR(c.rhs, 0.0, 1e-12);
            KAVACH_ASSERT_EQUAL(c.num_nonzeros(), 5);  // 4 crudes + 1 prod
        }
    }
    KAVACH_ASSERT(found_p1);
}

// ============================================================================
// G. Quality Constraints
// ============================================================================

KAVACH_TEST(test_mrpl_g_quality_constraints) {
    auto p = MRPLParameters::default_parameters();
    auto m = MRPLModelGenerator::build_canonical_model(p);

    bool found_sulfur_p1 = false;
    bool found_dens_min_p3 = false;
    for (std::size_t i = 0; i < m.num_constraints(); ++i) {
        const auto& c = m.constraint(static_cast<model::ConstrId>(i));
        if (c.name == "q_sulfur_P1") {
            found_sulfur_p1 = true;
            KAVACH_ASSERT_EQUAL(c.sense, model::ConstraintSense::LESS_EQUAL);
            KAVACH_ASSERT_NEAR(c.rhs, 0.0, 1e-12);
        } else if (c.name == "q_dens_min_P3") {
            found_dens_min_p3 = true;
            KAVACH_ASSERT_EQUAL(c.sense, model::ConstraintSense::GREATER_EQUAL);
            KAVACH_ASSERT_NEAR(c.rhs, 0.0, 1e-12);
        }
    }
    KAVACH_ASSERT(found_sulfur_p1 && found_dens_min_p3);
}

// ============================================================================
// H. Unit Capacity Linkage
// ============================================================================

KAVACH_TEST(test_mrpl_h_unit_capacity_linkage) {
    auto p = MRPLParameters::default_parameters();
    auto m = MRPLModelGenerator::build_canonical_model(p);

    bool found_cap_max_u1 = false;
    bool found_cap_min_u1 = false;
    for (std::size_t i = 0; i < m.num_constraints(); ++i) {
        const auto& c = m.constraint(static_cast<model::ConstrId>(i));
        if (c.name == "cap_max_U1") {
            found_cap_max_u1 = true;
            KAVACH_ASSERT_EQUAL(c.sense, model::ConstraintSense::LESS_EQUAL);
        } else if (c.name == "cap_min_U1") {
            found_cap_min_u1 = true;
            KAVACH_ASSERT_EQUAL(c.sense, model::ConstraintSense::GREATER_EQUAL);
        }
    }
    KAVACH_ASSERT(found_cap_max_u1 && found_cap_min_u1);
}

// ============================================================================
// I. Objective Calculation
// ============================================================================

KAVACH_TEST(test_mrpl_i_objective_calculation) {
    auto p = MRPLParameters::default_parameters();
    auto m = MRPLModelGenerator::build_canonical_model(p);

    KAVACH_ASSERT_EQUAL(m.objective_sense(), model::ObjectiveSense::MAXIMIZE);
    KAVACH_ASSERT_EQUAL(m.objective().linear_terms.num_terms(), 21);
}

// ============================================================================
// J. Model Parse Round-Trip
// ============================================================================

KAVACH_TEST(test_mrpl_j_model_parse_roundtrip) {
    auto p = MRPLParameters::default_parameters();
    std::string temp_lp = "/tmp/kavach_test_mrpl_roundtrip.lp";
    MRPLModelGenerator::save_lp_file(p, temp_lp);

    auto parsed = io::parse_model(temp_lp);
    KAVACH_ASSERT_EQUAL(parsed->num_variables(), 21);
    KAVACH_ASSERT_EQUAL(parsed->num_constraints(), 22);

    std::filesystem::remove(temp_lp);
}

// ============================================================================
// K. Baseline Feasibility
// ============================================================================

KAVACH_TEST(test_mrpl_k_baseline_feasibility) {
    MRPLCaseStudyRunner runner;
    auto sol =
        runner.solve_and_analyze(MRPLParameters::default_parameters(), "test_baseline", false);

    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::OPTIMAL);
    KAVACH_ASSERT(sol.objective > 5000.0);
    KAVACH_ASSERT_NEAR(sol.optimality_gap, 0.0, 1e-4);
}

// ============================================================================
// L. Binary Integrality
// ============================================================================

KAVACH_TEST(test_mrpl_l_binary_integrality) {
    MRPLCaseStudyRunner runner;
    auto sol =
        runner.solve_and_analyze(MRPLParameters::default_parameters(), "test_baseline", false);

    for (const auto& [name, val] : sol.unit_binaries) {
        double dist = std::min(std::abs(val - 0.0), std::abs(val - 1.0));
        KAVACH_ASSERT_NEAR(dist, 0.0, 1e-5);
    }
}

// ============================================================================
// M. Independent Verification
// ============================================================================

KAVACH_TEST(test_mrpl_m_independent_verification) {
    MRPLCaseStudyRunner runner;
    auto sol =
        runner.solve_and_analyze(MRPLParameters::default_parameters(), "test_baseline", false);

    KAVACH_ASSERT(sol.verified_optimal);
    KAVACH_ASSERT(sol.verification_report.fully_verified());
    KAVACH_ASSERT(sol.verification_report.primal_feasibility.passed);
    KAVACH_ASSERT(sol.verification_report.bound_feasibility.passed);
    KAVACH_ASSERT(sol.verification_report.integrality.passed);
    KAVACH_ASSERT_NEAR(sol.verification_report.aggregate_primal_residual, 0.0, 1e-5);
}

// ============================================================================
// N. Quality Recomputation
// ============================================================================

KAVACH_TEST(test_mrpl_n_quality_recomputation) {
    MRPLCaseStudyRunner runner;
    auto sol =
        runner.solve_and_analyze(MRPLParameters::default_parameters(), "test_baseline", false);

    for (const auto& q : sol.product_qualities) {
        if (q.production_kbpd > 1e-4) {
            KAVACH_ASSERT(q.quality_satisfied);
            KAVACH_ASSERT(q.sulfur_margin_wt_pct >= -1e-5);
            KAVACH_ASSERT(q.density_margin_kg_m3 >= -1e-4);
        }
    }
}

// ============================================================================
// O. Mass Balance Recomputation
// ============================================================================

KAVACH_TEST(test_mrpl_o_mass_balance_recomputation) {
    MRPLCaseStudyRunner runner;
    auto sol =
        runner.solve_and_analyze(MRPLParameters::default_parameters(), "test_baseline", false);

    KAVACH_ASSERT(sol.mass_balance.passed);
    KAVACH_ASSERT_NEAR(sol.mass_balance.max_absolute_residual, 0.0, 1e-5);
}

// ============================================================================
// P. Scenario Generation
// ============================================================================

KAVACH_TEST(test_mrpl_p_scenario_generation) {
    auto base = MRPLParameters::default_parameters();
    auto sc_a = MRPLParameters::scenario_reduced_crude();
    auto sc_b = MRPLParameters::scenario_tight_quality();
    auto sc_c = MRPLParameters::scenario_reduced_capacity();
    auto sc_d = MRPLParameters::scenario_infeasible();

    KAVACH_ASSERT_NEAR(sc_a.crudes[0].availability, 50.0, 1e-12);
    KAVACH_ASSERT_NEAR(sc_b.products[1].max_sulfur, 0.9, 1e-12);
    KAVACH_ASSERT_NEAR(sc_c.units[2].max_capacity, 40.0, 1e-12);
    KAVACH_ASSERT_NEAR(sc_d.crudes[0].availability, 10.0, 1e-12);
}

// ============================================================================
// Q. Scenario Reproducibility
// ============================================================================

KAVACH_TEST(test_mrpl_q_scenario_reproducibility) {
    MRPLCaseStudyRunner runner;
    auto sol1 = runner.solve_and_analyze(MRPLParameters::default_parameters(), "run1", false);
    auto sol2 = runner.solve_and_analyze(MRPLParameters::default_parameters(), "run2", false);

    KAVACH_ASSERT_EQUAL(sol1.status, sol2.status);
    KAVACH_ASSERT_NEAR(sol1.objective, sol2.objective, 1e-6);
    KAVACH_ASSERT_EQUAL(sol1.nodes_explored, sol2.nodes_explored);

    for (const auto& [u, val] : sol1.unit_binaries) {
        KAVACH_ASSERT_NEAR(val, sol2.unit_binaries[u], 1e-12);
    }
}

// ============================================================================
// R. Adversarial Infeasible Configuration
// ============================================================================

KAVACH_TEST(test_mrpl_r_adversarial_infeasible) {
    MRPLCaseStudyRunner runner;
    auto sol =
        runner.solve_and_analyze(MRPLParameters::scenario_infeasible(), "infeasible_sc", false);

    KAVACH_ASSERT_EQUAL(sol.status, core::SolverStatus::INFEASIBLE);
    KAVACH_ASSERT(!sol.verified_optimal);
}

// ============================================================================
// S. CLI Smoke Test
// ============================================================================

KAVACH_TEST(test_mrpl_s_cli_smoke_test) {
    std::string repo_root = ".";
    for (const auto& p : {".", "..", "../.."}) {
        if (std::filesystem::exists(std::filesystem::path(p) / "examples" / "mrpl" /
                                    "mrpl_blending_capacity.lp")) {
            repo_root = std::filesystem::canonical(p).string();
            break;
        }
    }

    std::string bin_path = "./kavach-opt";
    for (const auto& b : {"./kavach-opt", "./build-release/kavach-opt", "../kavach-opt",
                          "../build-release/kavach-opt"}) {
        if (std::filesystem::exists(b)) {
            bin_path = std::filesystem::canonical(b).string();
            break;
        }
    }

    std::string cmd1 = "cd \"" + repo_root + "\" && \"" + bin_path +
                       "\" solve examples/mrpl/mrpl_blending_capacity.lp > /dev/null 2>&1";
    int rc1 = std::system(cmd1.c_str());
    KAVACH_ASSERT_EQUAL(rc1, 0);

    std::string cmd2 = "cd \"" + repo_root + "\" && \"" + bin_path +
                       "\" mrpl --no-highs --output /tmp/mrpl_cli_test > /dev/null 2>&1";
    int rc2 = std::system(cmd2.c_str());
    KAVACH_ASSERT_EQUAL(rc2, 0);
    KAVACH_ASSERT(std::filesystem::exists("/tmp/mrpl_cli_test/baseline.json"));
}

// ============================================================================
// T. Reference Comparison Adapter
// ============================================================================

KAVACH_TEST(test_mrpl_t_reference_comparison_adapter) {
    benchmark::ExternalReferenceAdapter adapter;
    // Test that invoking the adapter on the MRPL model does not throw,
    // and returns either OPTIMAL (if highs is installed) or UNAVAILABLE (if not)
    std::string temp_lp = "/tmp/kavach_mrpl_ref_test.lp";
    MRPLModelGenerator::save_lp_file(MRPLParameters::default_parameters(), temp_lp);
    auto res = adapter.run(temp_lp, 10.0);
    std::filesystem::remove(temp_lp);

    KAVACH_ASSERT(res.status == benchmark::ReferenceSolverStatus::UNAVAILABLE ||
                  res.status == benchmark::ReferenceSolverStatus::OPTIMAL);
}
