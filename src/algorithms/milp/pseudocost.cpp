#include "algorithms/milp/pseudocost.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace kavach::algorithms::milp {

PseudocostTracker::PseudocostTracker(std::size_t num_variables, model::ObjectiveSense sense,
                                     double default_prior)
    : sense_(sense),
      default_prior_(default_prior > 1e-6 ? default_prior : 1.0),
      entries_(num_variables, PseudocostEntry{0.0, 0.0, 0, 0, default_prior_, default_prior_}) {}

void PseudocostTracker::reset(std::size_t num_variables, model::ObjectiveSense sense,
                              double default_prior) {
    sense_ = sense;
    default_prior_ = (default_prior > 1e-6 ? default_prior : 1.0);
    entries_.assign(num_variables, PseudocostEntry{0.0, 0.0, 0, 0, default_prior_, default_prior_});
    total_updates_ = 0;
}

bool PseudocostTracker::update_down(model::VarId var_id, double parent_obj, double child_obj,
                                    double fractionality, double int_tol) {
    if (var_id >= entries_.size()) {
        return false;
    }
    if (!std::isfinite(parent_obj) || !std::isfinite(child_obj)) {
        return false;
    }

    double f = fractionality - std::floor(fractionality);
    if (f <= int_tol || f >= 1.0 - int_tol) {
        return false;  // essentially integral, no fractional distance
    }

    double delta = 0.0;
    if (sense_ == model::ObjectiveSense::MINIMIZE) {
        delta = child_obj - parent_obj;
    } else {
        delta = parent_obj - child_obj;
    }

    // Degradation is non-negative; if solver found better or equal child due to float noise, clamp
    // to 0
    if (delta < 0.0) {
        delta = 0.0;
    }

    // Unit pseudocost
    double unit_degradation = delta / f;
    if (!std::isfinite(unit_degradation) || unit_degradation > 1e12) {
        unit_degradation = 1e12;
    }

    auto& entry = entries_[var_id];
    entry.down_sum += unit_degradation;
    entry.down_count += 1;
    entry.down_pseudocost = entry.down_sum / static_cast<double>(entry.down_count);
    total_updates_ += 1;
    return true;
}

bool PseudocostTracker::update_up(model::VarId var_id, double parent_obj, double child_obj,
                                  double fractionality, double int_tol) {
    if (var_id >= entries_.size()) {
        return false;
    }
    if (!std::isfinite(parent_obj) || !std::isfinite(child_obj)) {
        return false;
    }

    double f = fractionality - std::floor(fractionality);
    double up_dist = 1.0 - f;
    if (up_dist <= int_tol || up_dist >= 1.0 - int_tol) {
        return false;
    }

    double delta = 0.0;
    if (sense_ == model::ObjectiveSense::MINIMIZE) {
        delta = child_obj - parent_obj;
    } else {
        delta = parent_obj - child_obj;
    }

    if (delta < 0.0) {
        delta = 0.0;
    }

    double unit_degradation = delta / up_dist;
    if (!std::isfinite(unit_degradation) || unit_degradation > 1e12) {
        unit_degradation = 1e12;
    }

    auto& entry = entries_[var_id];
    entry.up_sum += unit_degradation;
    entry.up_count += 1;
    entry.up_pseudocost = entry.up_sum / static_cast<double>(entry.up_count);
    total_updates_ += 1;
    return true;
}

double PseudocostTracker::compute_score(model::VarId var_id, double fractionality) const noexcept {
    if (var_id >= entries_.size()) {
        return 0.0;
    }

    double f = fractionality - std::floor(fractionality);
    if (f <= 0.0 || f >= 1.0) {
        return 0.0;
    }

    const auto& entry = entries_[var_id];
    double down_dist = f;
    double up_dist = 1.0 - f;

    double q_down = entry.down_pseudocost * down_dist;
    double q_up = entry.up_pseudocost * up_dist;

    // Achterberg product score: max(q_down, 1e-6) * max(q_up, 1e-6)
    double score = std::max(q_down, 1e-6) * std::max(q_up, 1e-6);
    if (!std::isfinite(score) || score > 1e15) {
        score = 1e15;
    }
    return score;
}

bool PseudocostTracker::is_reliable(model::VarId var_id, std::size_t threshold) const noexcept {
    if (var_id >= entries_.size()) {
        return false;
    }
    return entries_[var_id].is_reliable(threshold);
}

const PseudocostEntry& PseudocostTracker::entry(model::VarId var_id) const {
    if (var_id >= entries_.size()) {
        throw std::out_of_range("VarId out of range in PseudocostTracker::entry");
    }
    return entries_[var_id];
}

}  // namespace kavach::algorithms::milp
