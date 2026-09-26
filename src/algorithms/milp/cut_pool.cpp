#include "algorithms/milp/cut_pool.hpp"

#include <algorithm>
#include <cmath>

namespace kavach::algorithms::milp {

CutPool::CutPool(std::size_t max_capacity) : max_capacity_(max_capacity) {}

bool CutPool::is_parallel_to_existing(const LinearCut& cut, double max_parallelism) const {
    if (cut.norm2 <= 1e-12) {
        return true;  // Zero norm cut is invalid/parallel to everything
    }

    for (const auto& existing : cuts_) {
        if (!existing.is_active || existing.norm2 <= 1e-12) {
            continue;
        }
        if (existing.sense != cut.sense) {
            continue;
        }

        // Compute sparse dot product
        double dot = 0.0;
        std::size_t p1 = 0;
        std::size_t p2 = 0;
        const std::size_t n1 = cut.var_ids.size();
        const std::size_t n2 = existing.var_ids.size();

        while (p1 < n1 && p2 < n2) {
            if (cut.var_ids[p1] < existing.var_ids[p2]) {
                ++p1;
            } else if (existing.var_ids[p2] < cut.var_ids[p1]) {
                ++p2;
            } else {
                dot += cut.coefficients[p1] * existing.coefficients[p2];
                ++p1;
                ++p2;
            }
        }

        double cos_sim = dot / (cut.norm2 * existing.norm2);
        if (std::abs(cos_sim) >= max_parallelism) {
            // Highly parallel: check if RHS is also close
            double rhs_diff = std::abs(cut.rhs / cut.norm2 - existing.rhs / existing.norm2);
            if (rhs_diff < 0.1) {
                return true;  // Redundant / near-parallel
            }
        }
    }
    return false;
}

bool CutPool::add_cut(LinearCut cut, double max_parallelism) {
    // 1. Sanity check: finite coefficients and RHS
    if (!std::isfinite(cut.rhs)) {
        return false;
    }
    if (cut.var_ids.empty() || cut.coefficients.empty() ||
        cut.var_ids.size() != cut.coefficients.size()) {
        return false;
    }

    // Compute norm2 and check finiteness
    double sum_sq = 0.0;
    for (double c : cut.coefficients) {
        if (!std::isfinite(c)) {
            return false;
        }
        sum_sq += c * c;
    }
    cut.norm2 = std::sqrt(sum_sq);
    if (cut.norm2 <= 1e-12) {
        return false;  // Zero vector
    }

    // 2. Parallelism / redundancy check
    if (is_parallel_to_existing(cut, max_parallelism)) {
        return false;
    }

    // 3. Assign ID and append
    cut.cut_id = next_cut_id_++;
    cut.is_active = true;
    cuts_.push_back(std::move(cut));

    // 4. Enforce capacity if needed
    if (cuts_.size() > max_capacity_ * 2) {
        enforce_capacity();
    }

    return true;
}

std::vector<LinearCut> CutPool::select_best_cuts(std::size_t max_count) const {
    std::vector<LinearCut> candidates;
    candidates.reserve(cuts_.size());
    for (const auto& c : cuts_) {
        if (c.is_active) {
            candidates.push_back(c);
        }
    }

    // Deterministic sort: efficacy descending, tie-break by cut_id ascending
    std::sort(candidates.begin(), candidates.end(), [](const LinearCut& a, const LinearCut& b) {
        if (a.efficacy != b.efficacy) {
            return a.efficacy > b.efficacy;
        }
        return a.cut_id < b.cut_id;
    });

    if (candidates.size() > max_count) {
        candidates.resize(max_count);
    }
    return candidates;
}

void CutPool::enforce_capacity() {
    if (cuts_.size() <= max_capacity_) {
        return;
    }

    // Sort cuts: active before inactive, then by efficacy descending
    std::sort(cuts_.begin(), cuts_.end(), [](const LinearCut& a, const LinearCut& b) {
        if (a.is_active != b.is_active) {
            return a.is_active > b.is_active;
        }
        if (a.efficacy != b.efficacy) {
            return a.efficacy > b.efficacy;
        }
        return a.cut_id < b.cut_id;
    });

    // Deactivate excess cuts
    for (std::size_t i = max_capacity_; i < cuts_.size(); ++i) {
        cuts_[i].is_active = false;
    }
    cuts_.resize(max_capacity_);
}

void CutPool::clear() {
    cuts_.clear();
}

std::size_t CutPool::active_count() const noexcept {
    std::size_t cnt = 0;
    for (const auto& c : cuts_) {
        if (c.is_active) {
            ++cnt;
        }
    }
    return cnt;
}

bool CutPool::verify_against_witnesses(const LinearCut& cut,
                                       const std::vector<std::vector<double>>& integer_witnesses,
                                       double tol) {
    for (const auto& x : integer_witnesses) {
        double viol = cut.compute_violation(x);
        if (viol > tol) {
            return false;
        }
    }
    return true;
}

}  // namespace kavach::algorithms::milp
