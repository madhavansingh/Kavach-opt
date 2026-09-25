#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "numerics/sparse_matrix.hpp"

namespace kavach::numerics {

/**
 * @brief Categorization of matrix condition / coefficient dynamic range.
 */
enum class ConditioningCategory : std::uint8_t {
    WELL_CONDITIONED,         ///< Dynamic range < 1e4
    MODERATE,                 ///< 1e4 <= Dynamic range < 1e8
    ILL_CONDITIONED,          ///< 1e8 <= Dynamic range < 1e13
    SEVERELY_ILL_CONDITIONED  ///< Dynamic range >= 1e13
};

[[nodiscard]] std::string to_string(ConditioningCategory category);

/**
 * @brief Conditioning and coefficient magnitude diagnostics for a sparse matrix.
 */
struct ConditioningDiagnostics {
    std::size_t rows{0};
    std::size_t cols{0};
    std::size_t nnz{0};
    double min_abs_val{0.0};
    double max_abs_val{0.0};
    double dynamic_range_ratio{1.0};
    double norm_1{0.0};
    double norm_inf{0.0};
    ConditioningCategory category{ConditioningCategory::WELL_CONDITIONED};
    bool recommend_scaling{false};

    [[nodiscard]] std::string to_string() const;
};

/**
 * @brief Row and column scaling factors: A_scaled = R * A * C.
 */
struct ScalingFactors {
    std::vector<double> row_scale;  ///< R vector of size m (R_ii)
    std::vector<double> col_scale;  ///< C vector of size n (C_jj)

    [[nodiscard]] std::size_t rows() const noexcept { return row_scale.size(); }
    [[nodiscard]] std::size_t cols() const noexcept { return col_scale.size(); }

    // --- Unscaling Operations ---

    /// Unscales primal solution: x = C * x_scaled (x_j = col_scale[j] * x_scaled[j])
    [[nodiscard]] std::vector<double> unscale_primal(
        const std::vector<double>& scaled_primal) const;

    /// Unscales dual multipliers: y = R * y_scaled (y_i = row_scale[i] * y_scaled[i])
    [[nodiscard]] std::vector<double> unscale_dual(const std::vector<double>& scaled_dual) const;

    /// Unscales reduced costs: d_j = d_scaled[j] / col_scale[j]
    [[nodiscard]] std::vector<double> unscale_reduced_costs(
        const std::vector<double>& scaled_reduced_costs) const;

    // --- Scaling Operations on Vectors ---

    /// Scales RHS vector: b_scaled = R * b
    [[nodiscard]] std::vector<double> scale_rhs(const std::vector<double>& rhs) const;

    /// Scales objective vector: c_scaled = C * c
    [[nodiscard]] std::vector<double> scale_objective(const std::vector<double>& obj) const;

    /// Scales primal lower/upper bounds: bounds_scaled = bounds / col_scale
    void scale_bounds(std::vector<double>& lower, std::vector<double>& upper) const;
};

/**
 * @brief Analyze matrix conditioning and coefficient dynamic range.
 */
[[nodiscard]] ConditioningDiagnostics analyze_conditioning(const SparseMatrix& matrix);

/**
 * @brief Compute geometric-mean row/column scaling factors for a CSR matrix.
 *
 * Implements standard geometric-mean equilibration:
 * For each row i: r_i = 1 / sqrt(min_nz * max_nz)
 * For each col j: c_j = 1 / sqrt(min_nz * max_nz)
 * Iterated up to max_passes or until dynamic range change is below tol.
 */
[[nodiscard]] ScalingFactors compute_geometric_mean_scaling(const CsrMatrix& matrix,
                                                            std::size_t max_passes = 4,
                                                            double tol = 1e-3);

/**
 * @brief Apply scaling factors to CSR matrix: A_scaled = R * A * C.
 */
[[nodiscard]] CsrMatrix apply_scaling(const CsrMatrix& matrix, const ScalingFactors& factors);

/**
 * @brief Apply scaling factors to CSC matrix: A_scaled = R * A * C.
 */
[[nodiscard]] CscMatrix apply_scaling(const CscMatrix& matrix, const ScalingFactors& factors);

}  // namespace kavach::numerics
