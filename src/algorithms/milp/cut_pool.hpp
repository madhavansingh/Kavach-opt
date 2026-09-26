#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "algorithms/milp/cut_types.hpp"
#include "model/model.hpp"

namespace kavach::algorithms::milp {

/**
 * @brief Thread-safe and deterministic pool of valid linear cutting planes.
 *
 * Implements:
 * - Duplicate & near-parallel cut rejection
 * - Numerical coefficient and dynamic range validation
 * - Deterministic sorting by efficacy (ties broken by ascending cut_id)
 * - Bounded pool management with capacity limits
 * - Sparse representation preservation
 */
class CutPool {
public:
    explicit CutPool(std::size_t max_capacity = 100);

    /**
     * @brief Add a candidate cut to the pool if valid and non-redundant.
     *
     * @param cut Candidate cut to add.
     * @param max_parallelism Maximum cosine similarity with existing cuts (0.0 to 1.0).
     * @return true if cut was accepted, false if rejected (redundant, weak, or invalid).
     */
    bool add_cut(LinearCut cut, double max_parallelism = 0.999);

    /**
     * @brief Select up to max_count best cuts ordered by efficacy descending.
     */
    [[nodiscard]] std::vector<LinearCut> select_best_cuts(std::size_t max_count) const;

    /**
     * @brief Prune least effective cuts if pool exceeds max_capacity.
     */
    void enforce_capacity();

    /**
     * @brief Clear all cuts from the pool.
     */
    void clear();

    [[nodiscard]] std::size_t size() const noexcept { return cuts_.size(); }
    [[nodiscard]] std::size_t active_count() const noexcept;
    [[nodiscard]] const std::vector<LinearCut>& cuts() const noexcept { return cuts_; }
    [[nodiscard]] std::size_t max_capacity() const noexcept { return max_capacity_; }
    void set_max_capacity(std::size_t cap) noexcept { max_capacity_ = cap; }

    /**
     * @brief Mathematically verify that a cut is satisfied by known integer-feasible points.
     * Returns true if all witnesses satisfy the cut within tol.
     */
    static bool verify_against_witnesses(const LinearCut& cut,
                                         const std::vector<std::vector<double>>& integer_witnesses,
                                         double tol = 1e-6);

private:
    [[nodiscard]] bool is_parallel_to_existing(const LinearCut& cut, double max_parallelism) const;

    std::size_t max_capacity_{100};
    std::uint64_t next_cut_id_{1};
    std::vector<LinearCut> cuts_;
};

}  // namespace kavach::algorithms::milp
