#pragma once

#include <vector>

#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"

namespace kavach::verification {

struct VerificationReport {
    bool passed{false};
    double recomputed_primal_residual{0.0};
    double recomputed_dual_residual{0.0};
    double recomputed_complementary_slackness{0.0};
    double recomputed_integrality_violation{0.0};
    double recomputed_kkt_residual{0.0};
    double recomputed_objective{0.0};
    core::SolverStatus certified_status{core::SolverStatus::UNKNOWN};
};

/**
 * @brief Architecture interface scaffolding for [Y] Independent Verification Layer.
 * Full implementation scheduled for Phase 6 per Blueprint Section 4, 6, 12 & Section 32.
 *
 * Hard architectural invariant: Must share ZERO code/logic with solver core algorithms.
 */
class IndependentVerifier {
public:
    virtual ~IndependentVerifier() = default;

    [[nodiscard]] virtual VerificationReport verify_lp(
        const model::CanonicalModel& original_model, const std::vector<double>& primal_solution,
        const std::vector<double>& dual_multipliers, double reported_objective,
        const core::SolverTolerances& tolerances) const = 0;

    [[nodiscard]] virtual VerificationReport verify_milp(
        const model::CanonicalModel& original_model, const std::vector<double>& incumbent_solution,
        double best_bound, double reported_objective,
        const core::SolverTolerances& tolerances) const = 0;

    [[nodiscard]] virtual VerificationReport verify_qp(
        const model::CanonicalModel& original_model, const std::vector<double>& primal_solution,
        const std::vector<double>& dual_solution, double reported_objective,
        const core::SolverTolerances& tolerances) const = 0;
};

}  // namespace kavach::verification
