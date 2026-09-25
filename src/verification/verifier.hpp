#pragma once

/**
 * @file verifier.hpp
 * @brief Architecture interface for the Independent Verification Layer [Y].
 *
 * For LP verification, use verification::LPVerifier (lp_verifier.hpp) directly.
 * The abstract base class here is retained for future MILP / QP extension points.
 */

#include <vector>

#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "model/model.hpp"
#include "verification/lp_verifier.hpp"
#include "verification/milp_verifier.hpp"

namespace kavach::verification {

/**
 * @brief Aggregated structured verification report (legacy / future MILP-QP bridge).
 *
 * Thin wrapper around LPVerificationReport for the abstract interface.
 * Use LPVerificationReport directly for LP results.
 */
struct VerificationReport {
    bool passed{false};
    double recomputed_primal_residual{0.0};
    double recomputed_dual_residual{0.0};
    double recomputed_complementary_slackness{0.0};
    double recomputed_integrality_violation{0.0};  ///< MILP extension
    double recomputed_kkt_residual{0.0};           ///< QP extension (stub)
    double recomputed_objective{0.0};
    core::SolverStatus certified_status{core::SolverStatus::UNKNOWN};
    std::string summary;

    /** Construct from an LPVerificationReport. */
    static VerificationReport from_lp(const LPVerificationReport& lp_report) {
        VerificationReport r;
        r.passed = lp_report.fully_verified();
        r.recomputed_primal_residual = lp_report.aggregate_primal_residual;
        r.recomputed_dual_residual = lp_report.aggregate_dual_residual;
        r.recomputed_complementary_slackness = lp_report.complementary_slackness.max_violation;
        r.recomputed_integrality_violation = 0.0;
        r.recomputed_kkt_residual = 0.0;
        r.recomputed_objective = lp_report.recomputed_objective;
        r.certified_status = lp_report.certified_status;
        r.summary = lp_report.summary;
        return r;
    }

    /** Construct from a MILPVerificationReport. */
    static VerificationReport from_milp(const MILPVerificationReport& milp_report) {
        VerificationReport r;
        r.passed = milp_report.fully_verified();
        r.recomputed_primal_residual = milp_report.aggregate_primal_residual;
        r.recomputed_dual_residual = 0.0;
        r.recomputed_complementary_slackness = 0.0;
        r.recomputed_integrality_violation = milp_report.max_integrality_violation;
        r.recomputed_kkt_residual = 0.0;
        r.recomputed_objective = milp_report.recomputed_objective;
        r.certified_status = milp_report.certified_status;
        r.summary = milp_report.summary;
        return r;
    }
};

/**
 * @brief Architecture interface scaffolding for [Y] Independent Verification Layer.
 *
 * For production LP verification use LPVerifier directly.
 * For production MILP verification use MILPVerifier directly.
 * QP methods are stubs pending future solver implementation.
 */
class IndependentVerifier {
public:
    virtual ~IndependentVerifier() = default;

    [[nodiscard]] virtual VerificationReport verify_lp(
        const model::CanonicalModel& original_model, const std::vector<double>& primal_solution,
        const std::vector<double>& dual_multipliers, double reported_objective,
        const core::SolverTolerances& tolerances) const = 0;

    /**
     * @brief Independent verification of a MILP solution.
     */
    [[nodiscard]] virtual VerificationReport verify_milp(
        const model::CanonicalModel& original_model, const std::vector<double>& incumbent_solution,
        double best_bound, double reported_objective,
        const core::SolverTolerances& tolerances) const {
        if (incumbent_solution.empty()) {
            VerificationReport r;
            r.certified_status = core::SolverStatus::UNKNOWN;
            r.summary = "MILP verification unavailable: no incumbent solution provided.";
            return r;
        }
        MILPVerifier verifier(MILPVerifierTolerances::from_solver_tolerances(tolerances));
        auto rpt = verifier.verify(original_model, incumbent_solution, best_bound,
                                   reported_objective, core::SolverStatus::OPTIMAL);
        return VerificationReport::from_milp(rpt);
    }

    /**
     * @brief QP verification stub (not yet implemented — solver does not exist).
     * Returns UNKNOWN with appropriate diagnostic.
     */
    [[nodiscard]] virtual VerificationReport verify_qp(
        const model::CanonicalModel& /*original_model*/,
        const std::vector<double>& /*primal_solution*/,
        const std::vector<double>& /*dual_solution*/, double /*reported_objective*/,
        const core::SolverTolerances& /*tolerances*/) const {
        VerificationReport r;
        r.certified_status = core::SolverStatus::UNKNOWN;
        r.summary = "QP solver not yet implemented — verification unavailable.";
        return r;
    }
};

/**
 * @brief Concrete production LP verifier implementing the abstract interface.
 */
class KavachLPVerifier final : public IndependentVerifier {
public:
    explicit KavachLPVerifier(VerifierTolerances tols = {}) : lp_verifier_(std::move(tols)) {}

    [[nodiscard]] VerificationReport verify_lp(
        const model::CanonicalModel& original_model, const std::vector<double>& primal_solution,
        const std::vector<double>& dual_multipliers, double reported_objective,
        const core::SolverTolerances& tolerances) const override {
        std::vector<double> empty_rc;
        auto lp_report =
            lp_verifier_.verify(original_model, primal_solution, dual_multipliers, empty_rc,
                                reported_objective, core::SolverStatus::OPTIMAL, tolerances);
        return VerificationReport::from_lp(lp_report);
    }

private:
    LPVerifier lp_verifier_;
};

}  // namespace kavach::verification
