#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "model/model.hpp"

namespace kavach::algorithms::milp {

/**
 * @brief Strategy governing branching variable selection at fractional nodes.
 */
enum class BranchingStrategy : std::uint8_t {
    MOST_FRACTIONAL,  ///< Classical most-fractional branching (deterministic baseline)
    PSEUDOCOST,       ///< Pure pseudocost branching based on accumulated history
    STRONG,           ///< Pure strong branching evaluating LP child bounds
    RELIABILITY,  ///< Hybrid reliability branching (strong branch until reliable, then pseudocost)
    AUTO          ///< Automatic choice based on problem characteristics
};

[[nodiscard]] constexpr const char* to_string(BranchingStrategy bs) noexcept {
    switch (bs) {
        case BranchingStrategy::MOST_FRACTIONAL:
            return "MOST_FRACTIONAL";
        case BranchingStrategy::PSEUDOCOST:
            return "PSEUDOCOST";
        case BranchingStrategy::STRONG:
            return "STRONG";
        case BranchingStrategy::RELIABILITY:
            return "RELIABILITY";
        case BranchingStrategy::AUTO:
            return "AUTO";
    }
    return "UNKNOWN";
}

/**
 * @brief Source of the currently known best integer incumbent.
 */
enum class IncumbentSource : std::uint8_t {
    NONE,
    ROOT_LP,
    ROUNDING,
    PRIMAL_HEURISTIC,
    INTEGER_LP_NODE
};

[[nodiscard]] constexpr const char* to_string(IncumbentSource src) noexcept {
    switch (src) {
        case IncumbentSource::NONE:
            return "NONE";
        case IncumbentSource::ROOT_LP:
            return "ROOT_LP";
        case IncumbentSource::ROUNDING:
            return "ROUNDING";
        case IncumbentSource::PRIMAL_HEURISTIC:
            return "PRIMAL_HEURISTIC";
        case IncumbentSource::INTEGER_LP_NODE:
            return "INTEGER_LP_NODE";
    }
    return "UNKNOWN";
}

/**
 * @brief Candidate integer variable evaluated for branching.
 */
struct BranchCandidate {
    model::VarId var_id{model::INVALID_VAR_ID};
    double score{0.0};
    double fractionality{0.0};  ///< min(f_j, 1 - f_j)
    double down_dist{0.0};      ///< f_j = x_j* - floor(x_j*)
    double up_dist{0.0};        ///< 1 - f_j
    double down_pseudocost{0.0};
    double up_pseudocost{0.0};
    std::size_t down_obs{0};
    std::size_t up_obs{0};
    bool is_reliable{false};
};

/**
 * @brief Telemetry metrics covering the advanced branch-and-cut search engine.
 */
struct SearchTelemetry {
    std::string branching_strategy_used{"RELIABILITY"};
    std::size_t pseudocost_candidates{0};
    std::size_t pseudocost_updates{0};
    std::size_t strong_branching_calls{0};
    std::size_t strong_branching_candidates{0};
    std::size_t strong_branching_infeasible_children{0};
    std::size_t reliability_fallbacks{0};
    std::size_t heuristic_calls{0};
    std::size_t heuristic_feasible_candidates{0};
    std::size_t heuristic_incumbent_updates{0};
    std::string incumbent_source{"NONE"};
    std::string search_explanation;
};

}  // namespace kavach::algorithms::milp
