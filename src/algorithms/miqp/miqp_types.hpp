#pragma once

/**
 * @file miqp_types.hpp
 * @brief Data structures, options, and telemetry for Convex MIQP Solver.
 */

#include <chrono>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "algorithms/milp/cut_types.hpp"
#include "algorithms/milp/search_types.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "verification/miqp_verifier.hpp"

namespace kavach::algorithms::miqp {

/**
 * @brief Configuration options for Convex MIQP Solver.
 */
struct MIQPSolverOptions {
    core::SolverTolerances tolerances;
    std::size_t max_nodes{50000};    ///< Maximum B&B nodes to explore
    double time_limit_seconds{0.0};  ///< Time limit (0.0 = unlimited)
    bool enable_presolve{false};     ///< Presolve enabled (default false for QP/MIQP)
    bool enable_scaling{true};       ///< Scaling enabled
    milp::CutOptions cut_options;    ///< Cut options (Gomory cuts disabled for MIQP)

    // Advanced Branch-and-Bound Search Options
    milp::BranchingStrategy branching_strategy{milp::BranchingStrategy::RELIABILITY};
    std::size_t strong_branching_candidates{5};
    std::size_t pseudocost_reliability_threshold{4};
    bool enable_primal_heuristic{true};
    std::size_t heuristic_max_repair_rounds{5};

    bool verbose{false};
};

/**
 * @brief Runtime search and relaxation telemetry for Convex MIQP.
 */
struct MIQPTelemetry {
    std::size_t qp_relaxations_solved{0};
    std::size_t qp_relaxation_iterations{0};
    double qp_relaxation_time_seconds{0.0};
    double qp_kkt_residual{0.0};

    std::size_t nodes_explored{0};
    std::size_t nodes_pruned_bound{0};
    std::size_t nodes_pruned_infeasible{0};
    std::size_t nodes_pruned_integer{0};

    std::size_t integer_candidates_found{0};
    std::size_t incumbent_updates{0};
    std::size_t strong_qp_probes{0};
    std::size_t failed_qp_relaxations{0};

    double root_qp_bound{0.0};
    double final_best_bound{0.0};
    double solve_time_seconds{0.0};

    bool node_limit_reached{false};
    bool time_limit_reached{false};

    milp::SearchTelemetry search_telemetry;
    std::string message;
};

/**
 * @brief Complete solution outcome of a Convex MIQP solve.
 */
struct MIQPSolution {
    core::SolverStatus status{core::SolverStatus::UNKNOWN};
    double best_objective{0.0};              ///< Incumbent objective value
    double best_bound{0.0};                  ///< Dual bound from relaxation tree
    double optimality_gap{1.0};              ///< Relative gap |z_inc - z_bound| / max(1, |z_inc|)
    std::vector<double> incumbent_solution;  ///< Optimal primal integer assignment
    std::size_t iterations{0};               ///< Total QP relaxation iterations
    std::size_t nodes{0};                    ///< Total B&B nodes explored
    MIQPTelemetry telemetry;
    verification::MIQPVerificationReport verification_report;
    std::string message;

    [[nodiscard]] bool is_optimal() const noexcept { return status == core::SolverStatus::OPTIMAL; }
    [[nodiscard]] bool is_infeasible() const noexcept {
        return status == core::SolverStatus::INFEASIBLE;
    }
    [[nodiscard]] bool has_incumbent() const noexcept { return !incumbent_solution.empty(); }
};

}  // namespace kavach::algorithms::miqp
