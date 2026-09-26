#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "algorithms/lp/lp_types.hpp"
#include "algorithms/lp/simplex.hpp"
#include "algorithms/milp/cut_pool.hpp"
#include "algorithms/milp/cut_types.hpp"
#include "algorithms/milp/gomory_separator.hpp"
#include "algorithms/milp/mir_separator.hpp"
#include "model/model.hpp"

namespace kavach::algorithms::milp {

/**
 * @brief Result of executing a cutting plane separation pass at a node.
 */
struct CutRoundResult {
    std::size_t cuts_added{0};
    bool integer_feasible{false};
    bool infeasible{false};
    double objective{0.0};
    std::vector<double> primal_solution;
    std::optional<lp::BasisState> basis;
};

/**
 * @brief Orchestrator for cutting plane generation, validation, and relaxation re-solving.
 */
class CuttingPlaneEngine {
public:
    explicit CuttingPlaneEngine(CutOptions options = CutOptions{});

    /**
     * @brief Run separation rounds at the root node of the B&B tree.
     *
     * @param work_model Working copy of the model to be strengthened with valid cuts.
     * @param lp_solver LP solver instance.
     * @param lp_opts Solver options for LP relaxations.
     * @param initial_lp_sol Solution of the initial continuous relaxation.
     * @param telemetry [out] Cut metrics recorder.
     * @param int_tol Integrality tolerance.
     * @return CutRoundResult containing strengthened bound, primal solution, and status.
     */
    CutRoundResult run_root_cuts(model::CanonicalModel& work_model, lp::RevisedSimplex& lp_solver,
                                 const lp::LPSolverOptions& lp_opts,
                                 const lp::LPSolution& initial_lp_sol, CutTelemetry& telemetry,
                                 double int_tol);

    [[nodiscard]] const CutPool& pool() const noexcept { return pool_; }
    [[nodiscard]] CutPool& pool() noexcept { return pool_; }
    [[nodiscard]] const CutOptions& options() const noexcept { return options_; }
    void set_options(CutOptions options) noexcept;

private:
    CutOptions options_;
    CutPool pool_;
    GomorySeparator gomory_sep_;
    MIRSeparator mir_sep_;
};

}  // namespace kavach::algorithms::milp
