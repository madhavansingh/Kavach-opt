#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "model/model.hpp"

namespace kavach::algorithms::milp {

/**
 * @brief Categorization of generated valid cutting planes.
 */
enum class CutType : std::uint8_t {
    GOMORY,  ///< Gomory fractional / mixed-integer cut from LP tableau row
    MIR      ///< Mixed-Integer Rounding (MIR) cut from mixed-integer row
};

[[nodiscard]] constexpr const char* to_string(CutType ct) noexcept {
    switch (ct) {
        case CutType::GOMORY:
            return "GOMORY";
        case CutType::MIR:
            return "MIR";
    }
    return "UNKNOWN";
}

/**
 * @brief High-level separation policy across the branch-and-bound tree.
 */
enum class CutStrategy : std::uint8_t {
    OFF,        ///< Cutting planes completely disabled (pure branch-and-bound baseline)
    ROOT_ONLY,  ///< Generate cuts only at the root node (conservative, robust default)
    PERIODIC    ///< Generate cuts at root and periodically at selected non-root nodes
};

[[nodiscard]] constexpr const char* to_string(CutStrategy cs) noexcept {
    switch (cs) {
        case CutStrategy::OFF:
            return "OFF";
        case CutStrategy::ROOT_ONLY:
            return "ROOT_ONLY";
        case CutStrategy::PERIODIC:
            return "PERIODIC";
    }
    return "UNKNOWN";
}

/**
 * @brief Configuration parameters controlling cut generation and limits.
 */
struct CutOptions {
    CutStrategy strategy{CutStrategy::ROOT_ONLY};
    std::size_t max_cut_rounds{5};         ///< Max separation rounds at root node
    std::size_t max_cuts_per_round{20};    ///< Max cuts added to the model per round
    std::size_t max_total_cuts{100};       ///< Maximum active cut pool capacity
    double min_cut_violation{1e-4};        ///< Minimum violation to accept a candidate cut
    double min_cut_efficacy{1e-4};         ///< Minimum Euclidean efficacy (distance to plane)
    double min_fractionality{0.01};        ///< Distance to nearest integer required for cut row
    double max_coefficient_range{1e6};     ///< Max ratio max|coeff| / min|coeff|
    double max_absolute_coefficient{1e6};  ///< Reject cuts with exploding coefficients
    bool enable_gomory{true};              ///< Enable Gomory fractional separator
    bool enable_mir{true};                 ///< Enable MIR separator
    std::size_t node_cut_interval{10};     ///< Separation frequency in non-root nodes
    std::size_t max_node_cut_rounds{2};    ///< Max rounds at a non-root node
};

/**
 * @brief Execution metrics and diagnostics for cutting plane separation.
 */
struct CutTelemetry {
    std::size_t cuts_generated{0};    ///< Total candidate cuts generated before filtering
    std::size_t gomory_cuts{0};       ///< Total Gomory candidate cuts generated
    std::size_t mir_cuts{0};          ///< Total MIR candidate cuts generated
    std::size_t cuts_accepted{0};     ///< Total cuts added to the active model
    std::size_t cuts_rejected{0};     ///< Candidates rejected by numerical or efficacy filters
    std::size_t cut_rounds{0};        ///< Total separation rounds executed
    std::size_t root_cuts{0};         ///< Total cuts added at the root node
    std::size_t active_cut_count{0};  ///< Current active cuts in model
};

/**
 * @brief Representation of a single valid linear cutting plane.
 *
 * Expressed in the canonical model's original variable space:
 *   \sum_{k} coefficients[k] * x[var_ids[k]] <= rhs   (or >= rhs)
 */
struct LinearCut {
    CutType type{CutType::GOMORY};
    std::uint64_t cut_id{0};
    std::size_t originating_node{0};
    std::size_t creation_round{0};

    // Sparse representation of cut coefficients
    std::vector<model::VarId> var_ids;
    std::vector<double> coefficients;
    double rhs{0.0};
    model::ConstraintSense sense{model::ConstraintSense::LESS_EQUAL};

    // Diagnostic metrics
    double violation{0.0};  ///< Activity at LP relaxation point minus rhs (for <=)
    double efficacy{0.0};   ///< Euclidean distance from LP point to cut hyperplane
    double norm2{0.0};      ///< L2 norm of the coefficient vector
    bool is_active{true};

    /**
     * @brief Compute the activity of this cut at a candidate solution x.
     */
    [[nodiscard]] double compute_activity(const std::vector<double>& x) const noexcept {
        double act = 0.0;
        const std::size_t nnz = std::min(var_ids.size(), coefficients.size());
        for (std::size_t k = 0; k < nnz; ++k) {
            model::VarId vid = var_ids[k];
            if (vid < x.size()) {
                act += coefficients[k] * x[vid];
            }
        }
        return act;
    }

    /**
     * @brief Compute the violation of this cut at a candidate solution x.
     * Positive value indicates violation of the cut.
     */
    [[nodiscard]] double compute_violation(const std::vector<double>& x) const noexcept {
        double act = compute_activity(x);
        if (sense == model::ConstraintSense::LESS_EQUAL) {
            return std::max(0.0, act - rhs);
        }
        if (sense == model::ConstraintSense::GREATER_EQUAL) {
            return std::max(0.0, rhs - act);
        }
        return std::abs(act - rhs);
    }

    /**
     * @brief Convert the cut into a CanonicalModel SparseLinearExpression.
     */
    [[nodiscard]] model::SparseLinearExpression to_expression() const {
        model::SparseLinearExpression expr;
        const std::size_t nnz = std::min(var_ids.size(), coefficients.size());
        for (std::size_t k = 0; k < nnz; ++k) {
            expr.add_term(var_ids[k], coefficients[k]);
        }
        return expr;
    }
};

}  // namespace kavach::algorithms::milp
