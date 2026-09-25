#pragma once

namespace kavach::core {

/**
 * @brief Named Numerical Tolerances Taxonomy as defined in Blueprint v2 Section 11.
 *
 * Never use magic numbers; every numerical check in the solver and verifier
 * references one of these named parameters.
 */
struct SolverTolerances {
    double primal_feasibility_tol{1e-7};  ///< Deviation allowed in Ax - b <= 0 (LP, MILP, QP)
    double dual_feasibility_tol{1e-7};  ///< Deviation allowed for reduced costs / dual multipliers
    double integrality_tol{1e-6};  ///< Deviation allowed for integer-constrained variables (MILP)
    double optimality_tol{1e-7};   ///< Reduced-cost / stationarity threshold for OPTIMAL
    double pivot_tol{1e-9};        ///< Threshold below which pivot is treated as numerically zero
    double residual_tol{1e-6};     ///< Threshold for independent verifier's recomputed residuals
};

}  // namespace kavach::core
