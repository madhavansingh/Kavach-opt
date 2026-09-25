#pragma once

#include <memory>

#include "model/model.hpp"

namespace kavach::presolve {

struct PresolveResult {
    std::unique_ptr<model::CanonicalModel> reduced_model;
    bool proven_infeasible{false};
    bool proven_unbounded{false};
};

/**
 * @brief Architecture interface scaffolding for [C] Presolve.
 * Full implementation scheduled for Phase 4 per Blueprint Section 4 & Section 32.
 */
class PresolveEngine {
public:
    virtual ~PresolveEngine() = default;

    [[nodiscard]] virtual PresolveResult apply(const model::CanonicalModel& original_model) = 0;
    virtual void restore_solution(const std::vector<double>& reduced_solution,
                                  std::vector<double>& original_solution) const = 0;
};

}  // namespace kavach::presolve
