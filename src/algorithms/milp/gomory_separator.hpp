#pragma once

#include <vector>

#include "algorithms/lp/basis.hpp"
#include "algorithms/lp/standard_form.hpp"
#include "algorithms/milp/cut_types.hpp"
#include "model/model.hpp"

namespace kavach::algorithms::milp {

/**
 * @brief Gomory Fractional / Mixed-Integer Cut Separator.
 *
 * Extracts cutting planes from fractional integer basic variables of the LP relaxation
 * tableau:
 *   \sum_{j \in I: f_j \le f_0} f_j z_j + \sum_{j \in I: f_j > f_0} \frac{f_0(1-f_j)}{1-f_0} z_j
 *     + \sum_{j \in C: \bar{a}_j > 0} \bar{a}_j z_j - \sum_{j \in C: \bar{a}_j < 0}
 * \frac{f_0}{1-f_0} \bar{a}_j z_j \ge f_0
 *
 * and translates the inequality into original model variable space.
 */
class GomorySeparator {
public:
    explicit GomorySeparator(const CutOptions& options = CutOptions{});

    /**
     * @brief Separate Gomory cuts from all eligible fractional basic rows of the LP relaxation.
     *
     * @param model CanonicalModel (original variable space).
     * @param standard_lp Standard-form LP corresponding to model.
     * @param basis Simplex basis factorized at the optimal LP solution.
     * @param x_lp Primal solution vector in model variable space.
     * @param z_lp Primal solution vector in standard-form space.
     * @param originating_node B&B node ID where cuts are separated.
     * @param current_round Current cut generation round.
     * @return List of validated, violated candidate cuts.
     */
    [[nodiscard]] std::vector<LinearCut> separate(const model::CanonicalModel& model,
                                                  const lp::StandardFormLP& standard_lp,
                                                  const lp::SimplexBasis& basis,
                                                  const std::vector<double>& x_lp,
                                                  const std::vector<double>& z_lp,
                                                  std::size_t originating_node = 0,
                                                  std::size_t current_round = 0) const;

    [[nodiscard]] const CutOptions& options() const noexcept { return options_; }
    void set_options(CutOptions options) noexcept { options_ = options; }

private:
    [[nodiscard]] std::vector<bool> classify_standard_form_integrality(
        const model::CanonicalModel& model, const lp::StandardFormLP& standard_lp) const;

    CutOptions options_;
};

}  // namespace kavach::algorithms::milp
