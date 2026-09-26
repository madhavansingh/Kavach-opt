#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "algorithms/milp/search_types.hpp"
#include "model/model.hpp"

namespace kavach::algorithms::milp {

/**
 * @brief Directional pseudocost record for an individual integer variable.
 */
struct PseudocostEntry {
    double down_sum{0.0};         ///< Accumulated objective degradation on down-branches
    double up_sum{0.0};           ///< Accumulated objective degradation on up-branches
    std::size_t down_count{0};    ///< Number of down-branch observations (N_j^-)
    std::size_t up_count{0};      ///< Number of up-branch observations (N_j^+)
    double down_pseudocost{1.0};  ///< Unit down-pseudocost Psi_j^-
    double up_pseudocost{1.0};    ///< Unit up-pseudocost Psi_j^+

    [[nodiscard]] bool is_reliable(std::size_t threshold) const noexcept {
        return down_count >= threshold && up_count >= threshold;
    }

    [[nodiscard]] std::size_t total_observations() const noexcept { return down_count + up_count; }
};

/**
 * @brief Deterministic Pseudocost Tracker for MILP branch-and-bound.
 *
 * Maintains directional objective degradation rates per unit integer violation
 * across all integer/binary variables in the model.
 *
 * Mathematical formulas:
 *   For a fractional integer variable x_j with LP relaxation value x_j* and f_j = x_j* -
 * floor(x_j*): down distance d_j^- = f_j up distance   d_j^+ = 1 - f_j
 *
 *   Observed objective degradation:
 *     Minimization: Delta_obj = max(0.0, child_obj - parent_obj)
 *     Maximization: Delta_obj = max(0.0, parent_obj - child_obj)
 *
 *   Unit pseudocost update:
 *     psi^- = Delta_obj / d_j^-
 *     Psi_j^- = (down_sum + psi^-) / (down_count + 1)
 *
 *   Branching score calculation (Achterberg product):
 *     q_j^- = Psi_j^- * d_j^-
 *     q_j^+ = Psi_j^+ * d_j^+
 *     score_j = max(q_j^-, 1e-6) * max(q_j^+, 1e-6)
 */
class PseudocostTracker {
public:
    explicit PseudocostTracker(std::size_t num_variables,
                               model::ObjectiveSense sense = model::ObjectiveSense::MINIMIZE,
                               double default_prior = 1.0);

    void reset(std::size_t num_variables,
               model::ObjectiveSense sense = model::ObjectiveSense::MINIMIZE,
               double default_prior = 1.0);

    /**
     * @brief Record an observed down-branch result (x_j <= floor(x_j*)).
     * @return true if update was valid and applied, false if rejected due to numerical bounds.
     */
    bool update_down(model::VarId var_id, double parent_obj, double child_obj, double fractionality,
                     double int_tol = 1e-6);

    /**
     * @brief Record an observed up-branch result (x_j >= ceil(x_j*)).
     * @return true if update was valid and applied, false if rejected due to numerical bounds.
     */
    bool update_up(model::VarId var_id, double parent_obj, double child_obj, double fractionality,
                   double int_tol = 1e-6);

    /**
     * @brief Compute the deterministic pseudocost branching score for a candidate.
     */
    [[nodiscard]] double compute_score(model::VarId var_id, double fractionality) const noexcept;

    /**
     * @brief Query reliability for a given variable.
     */
    [[nodiscard]] bool is_reliable(model::VarId var_id, std::size_t threshold) const noexcept;

    /**
     * @brief Access the pseudocost entry for inspection or telemetry.
     */
    [[nodiscard]] const PseudocostEntry& entry(model::VarId var_id) const;

    [[nodiscard]] std::size_t num_variables() const noexcept { return entries_.size(); }
    [[nodiscard]] std::size_t total_updates() const noexcept { return total_updates_; }

private:
    model::ObjectiveSense sense_{model::ObjectiveSense::MINIMIZE};
    double default_prior_{1.0};
    std::vector<PseudocostEntry> entries_;
    std::size_t total_updates_{0};
};

}  // namespace kavach::algorithms::milp
