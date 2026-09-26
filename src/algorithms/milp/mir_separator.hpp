#pragma once

#include <vector>

#include "algorithms/milp/cut_types.hpp"
#include "model/model.hpp"

namespace kavach::algorithms::milp {

/**
 * @brief Mixed-Integer Rounding (MIR) Cut Separator.
 *
 * Implements mathematically sound separation of MIR inequalities from mixed-integer
 * model constraints:
 *   \sum_{j \in I} F_\alpha(\delta a_j) x_j + \frac{1}{1 - \alpha} \sum_{j \in C: \delta g_j < 0}
 * (\delta g_j) y_j \le \lfloor \delta b' \rfloor + \text{shift}
 *
 * Supports:
 * - Arbitrary mixtures of integer and continuous variables
 * - Lower and upper variable bounds (affine translation to non-negative rays)
 * - LESS_EQUAL, GREATER_EQUAL, and EQUAL constraint orientations
 * - Multi-scale candidate evaluation to maximize violation and efficacy
 * - Strict numerical stability and dynamic range validation
 */
class MIRSeparator {
public:
    explicit MIRSeparator(const CutOptions& options = CutOptions{});

    /**
     * @brief Separate MIR cuts from all applicable mixed-integer constraints in model.
     *
     * @param model CanonicalModel containing the constraints.
     * @param x_lp Primal solution vector from the current LP relaxation.
     * @param originating_node B&B node ID where cuts are separated.
     * @param current_round Current cut generation round.
     * @return List of validated, violated candidate cuts.
     */
    [[nodiscard]] std::vector<LinearCut> separate(const model::CanonicalModel& model,
                                                  const std::vector<double>& x_lp,
                                                  std::size_t originating_node = 0,
                                                  std::size_t current_round = 0) const;

    [[nodiscard]] const CutOptions& options() const noexcept { return options_; }
    void set_options(CutOptions options) noexcept { options_ = options; }

private:
    [[nodiscard]] std::vector<LinearCut> separate_from_row(const model::CanonicalModel& model,
                                                           const model::Constraint& constr,
                                                           const std::vector<double>& x_lp,
                                                           std::size_t originating_node,
                                                           std::size_t current_round) const;

    CutOptions options_;
};

}  // namespace kavach::algorithms::milp
