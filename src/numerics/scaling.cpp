#include "numerics/scaling.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace kavach::numerics {

std::string to_string(ConditioningCategory category) {
    switch (category) {
        case ConditioningCategory::WELL_CONDITIONED:
            return "WELL_CONDITIONED";
        case ConditioningCategory::MODERATE:
            return "MODERATE";
        case ConditioningCategory::ILL_CONDITIONED:
            return "ILL_CONDITIONED";
        case ConditioningCategory::SEVERELY_ILL_CONDITIONED:
            return "SEVERELY_ILL_CONDITIONED";
    }
    return "UNKNOWN";
}

std::string ConditioningDiagnostics::to_string() const {
    std::ostringstream oss;
    oss << "ConditioningDiagnostics {"
        << " rows=" << rows << " cols=" << cols << " nnz=" << nnz << std::scientific
        << std::setprecision(3) << " min_abs_val=" << min_abs_val << " max_abs_val=" << max_abs_val
        << " dynamic_range_ratio=" << dynamic_range_ratio << " norm_1=" << norm_1
        << " norm_inf=" << norm_inf << " category=" << numerics::to_string(category)
        << " recommend_scaling=" << (recommend_scaling ? "true" : "false") << " }";
    return oss.str();
}

std::vector<double> ScalingFactors::unscale_primal(const std::vector<double>& scaled_primal) const {
    if (scaled_primal.size() != col_scale.size()) {
        throw std::invalid_argument(
            "unscale_primal: vector size does not match column scaling count");
    }
    std::vector<double> original(scaled_primal.size());
    for (std::size_t j = 0; j < scaled_primal.size(); ++j) {
        original[j] = scaled_primal[j] * col_scale[j];
    }
    return original;
}

std::vector<double> ScalingFactors::unscale_dual(const std::vector<double>& scaled_dual) const {
    if (scaled_dual.size() != row_scale.size()) {
        throw std::invalid_argument("unscale_dual: vector size does not match row scaling count");
    }
    std::vector<double> original(scaled_dual.size());
    for (std::size_t i = 0; i < scaled_dual.size(); ++i) {
        original[i] = scaled_dual[i] * row_scale[i];
    }
    return original;
}

std::vector<double> ScalingFactors::unscale_reduced_costs(
    const std::vector<double>& scaled_reduced_costs) const {
    if (scaled_reduced_costs.size() != col_scale.size()) {
        throw std::invalid_argument(
            "unscale_reduced_costs: vector size does not match column scaling count");
    }
    std::vector<double> original(scaled_reduced_costs.size());
    for (std::size_t j = 0; j < scaled_reduced_costs.size(); ++j) {
        original[j] = scaled_reduced_costs[j] / col_scale[j];
    }
    return original;
}

std::vector<double> ScalingFactors::scale_rhs(const std::vector<double>& rhs) const {
    if (rhs.size() != row_scale.size()) {
        throw std::invalid_argument("scale_rhs: vector size does not match row scaling count");
    }
    std::vector<double> scaled(rhs.size());
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        scaled[i] = rhs[i] * row_scale[i];
    }
    return scaled;
}

std::vector<double> ScalingFactors::scale_objective(const std::vector<double>& obj) const {
    if (obj.size() != col_scale.size()) {
        throw std::invalid_argument(
            "scale_objective: vector size does not match column scaling count");
    }
    std::vector<double> scaled(obj.size());
    for (std::size_t j = 0; j < obj.size(); ++j) {
        scaled[j] = obj[j] * col_scale[j];
    }
    return scaled;
}

void ScalingFactors::scale_bounds(std::vector<double>& lower, std::vector<double>& upper) const {
    if (lower.size() != col_scale.size() || upper.size() != col_scale.size()) {
        throw std::invalid_argument("scale_bounds: bound sizes do not match column scaling count");
    }
    constexpr double INF = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < col_scale.size(); ++j) {
        double c = col_scale[j];
        if (lower[j] > -INF) {
            lower[j] /= c;
        }
        if (upper[j] < INF) {
            upper[j] /= c;
        }
    }
}

ConditioningDiagnostics analyze_conditioning(const SparseMatrix& matrix) {
    ConditioningDiagnostics diag;
    diag.rows = matrix.rows();
    diag.cols = matrix.cols();
    diag.nnz = matrix.nonzeros();
    diag.norm_1 = matrix.one_norm();
    diag.norm_inf = matrix.infinity_norm();

    if (diag.nnz == 0) {
        diag.min_abs_val = 0.0;
        diag.max_abs_val = 0.0;
        diag.dynamic_range_ratio = 1.0;
        diag.category = ConditioningCategory::WELL_CONDITIONED;
        diag.recommend_scaling = false;
        return diag;
    }

    double min_v = std::numeric_limits<double>::infinity();
    double max_v = 0.0;

    if (const auto* csr = dynamic_cast<const CsrMatrix*>(&matrix)) {
        for (double v : csr->values()) {
            double abs_v = std::abs(v);
            if (abs_v > 1e-18) {
                min_v = std::min(min_v, abs_v);
                max_v = std::max(max_v, abs_v);
            }
        }
    } else if (const auto* csc = dynamic_cast<const CscMatrix*>(&matrix)) {
        for (double v : csc->values()) {
            double abs_v = std::abs(v);
            if (abs_v > 1e-18) {
                min_v = std::min(min_v, abs_v);
                max_v = std::max(max_v, abs_v);
            }
        }
    }

    if (std::isinf(min_v)) {
        min_v = 0.0;
    }

    diag.min_abs_val = min_v;
    diag.max_abs_val = max_v;
    diag.dynamic_range_ratio = (min_v > 0.0) ? (max_v / min_v) : 1.0;

    if (diag.dynamic_range_ratio < 1e4) {
        diag.category = ConditioningCategory::WELL_CONDITIONED;
    } else if (diag.dynamic_range_ratio < 1e8) {
        diag.category = ConditioningCategory::MODERATE;
    } else if (diag.dynamic_range_ratio < 1e13) {
        diag.category = ConditioningCategory::ILL_CONDITIONED;
    } else {
        diag.category = ConditioningCategory::SEVERELY_ILL_CONDITIONED;
    }

    diag.recommend_scaling = (diag.dynamic_range_ratio > 1e3);
    return diag;
}

ScalingFactors compute_geometric_mean_scaling(const CsrMatrix& matrix, std::size_t max_passes,
                                              double tol) {
    const std::size_t m = matrix.rows();
    const std::size_t n = matrix.cols();

    ScalingFactors factors;
    factors.row_scale.assign(m, 1.0);
    factors.col_scale.assign(n, 1.0);

    if (m == 0 || n == 0 || matrix.nonzeros() == 0) {
        return factors;
    }

    // Copy working values
    std::vector<double> working_vals = matrix.values();
    const auto& row_ptr = matrix.row_ptr();
    const auto& col_idx = matrix.col_idx();

    double prev_ratio = std::numeric_limits<double>::infinity();

    for (std::size_t pass = 0; pass < max_passes; ++pass) {
        // --- Row pass ---
        for (std::size_t i = 0; i < m; ++i) {
            std::size_t start = row_ptr[i];
            std::size_t end = row_ptr[i + 1];
            double r_min = std::numeric_limits<double>::infinity();
            double r_max = 0.0;
            for (std::size_t p = start; p < end; ++p) {
                double abs_v = std::abs(working_vals[p]);
                if (abs_v > 1e-18) {
                    r_min = std::min(r_min, abs_v);
                    r_max = std::max(r_max, abs_v);
                }
            }
            if (r_max > 0.0 && !std::isinf(r_min)) {
                double geom_mean = std::sqrt(r_min * r_max);
                if (geom_mean > 1e-18) {
                    double r_factor = 1.0 / geom_mean;
                    factors.row_scale[i] *= r_factor;
                    for (std::size_t p = start; p < end; ++p) {
                        working_vals[p] *= r_factor;
                    }
                }
            }
        }

        // --- Column pass ---
        std::vector<double> c_min(n, std::numeric_limits<double>::infinity());
        std::vector<double> c_max(n, 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            std::size_t start = row_ptr[i];
            std::size_t end = row_ptr[i + 1];
            for (std::size_t p = start; p < end; ++p) {
                std::size_t j = col_idx[p];
                double abs_v = std::abs(working_vals[p]);
                if (abs_v > 1e-18) {
                    c_min[j] = std::min(c_min[j], abs_v);
                    c_max[j] = std::max(c_max[j], abs_v);
                }
            }
        }

        for (std::size_t j = 0; j < n; ++j) {
            if (c_max[j] > 0.0 && !std::isinf(c_min[j])) {
                double geom_mean = std::sqrt(c_min[j] * c_max[j]);
                if (geom_mean > 1e-18) {
                    double c_factor = 1.0 / geom_mean;
                    factors.col_scale[j] *= c_factor;
                }
            }
        }

        // Apply column factors to working_vals
        for (std::size_t p = 0; p < working_vals.size(); ++p) {
            std::size_t j = col_idx[p];
            working_vals[p] *=
                (1.0 / std::sqrt(c_min[j] * c_max[j] > 1e-18 ? c_min[j] * c_max[j] : 1.0));
        }

        // Check dynamic range progress
        double cur_min = std::numeric_limits<double>::infinity();
        double cur_max = 0.0;
        for (double v : working_vals) {
            double abs_v = std::abs(v);
            if (abs_v > 1e-18) {
                cur_min = std::min(cur_min, abs_v);
                cur_max = std::max(cur_max, abs_v);
            }
        }
        double cur_ratio = (cur_min > 0.0) ? (cur_max / cur_min) : 1.0;
        if (std::abs(prev_ratio - cur_ratio) / (cur_ratio + 1.0) < tol) {
            break;
        }
        prev_ratio = cur_ratio;
    }

    return factors;
}

CsrMatrix apply_scaling(const CsrMatrix& matrix, const ScalingFactors& factors) {
    if (matrix.rows() != factors.rows() || matrix.cols() != factors.cols()) {
        throw std::invalid_argument(
            "apply_scaling: matrix dimensions do not match scaling factors");
    }

    const std::size_t m = matrix.rows();
    const auto& row_ptr = matrix.row_ptr();
    const auto& col_idx = matrix.col_idx();
    const auto& vals = matrix.values();

    std::vector<double> scaled_vals(vals.size());
    for (std::size_t i = 0; i < m; ++i) {
        double r = factors.row_scale[i];
        std::size_t start = row_ptr[i];
        std::size_t end = row_ptr[i + 1];
        for (std::size_t p = start; p < end; ++p) {
            std::size_t j = col_idx[p];
            scaled_vals[p] = r * vals[p] * factors.col_scale[j];
        }
    }

    return {m, matrix.cols(), std::vector<std::size_t>(row_ptr), std::vector<std::size_t>(col_idx),
            std::move(scaled_vals)};
}

CscMatrix apply_scaling(const CscMatrix& matrix, const ScalingFactors& factors) {
    if (matrix.rows() != factors.rows() || matrix.cols() != factors.cols()) {
        throw std::invalid_argument(
            "apply_scaling: matrix dimensions do not match scaling factors");
    }

    const std::size_t n = matrix.cols();
    const auto& col_ptr = matrix.col_ptr();
    const auto& row_idx = matrix.row_idx();
    const auto& vals = matrix.values();

    std::vector<double> scaled_vals(vals.size());
    for (std::size_t j = 0; j < n; ++j) {
        double c = factors.col_scale[j];
        std::size_t start = col_ptr[j];
        std::size_t end = col_ptr[j + 1];
        for (std::size_t p = start; p < end; ++p) {
            std::size_t i = row_idx[p];
            scaled_vals[p] = factors.row_scale[i] * vals[p] * c;
        }
    }

    return {matrix.rows(), n, std::vector<std::size_t>(col_ptr), std::vector<std::size_t>(row_idx),
            std::move(scaled_vals)};
}

}  // namespace kavach::numerics
