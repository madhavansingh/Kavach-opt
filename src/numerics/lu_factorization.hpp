#pragma once

#include <vector>

#include "numerics/sparse_matrix.hpp"

namespace kavach::numerics {

/**
 * @brief Architecture interface scaffolding for [E] Numerical Linear Algebra (Sparse LU).
 * Full implementation scheduled for Phase 3 per Blueprint Section 4 & Section 32.
 */
class SparseLUFactorization {
public:
    virtual ~SparseLUFactorization() = default;

    virtual bool factorize(const SparseMatrix& matrix, double pivot_tol) = 0;
    virtual bool solve(const std::vector<double>& rhs, std::vector<double>& solution) const = 0;
    virtual bool solve_transpose(const std::vector<double>& rhs,
                                 std::vector<double>& solution) const = 0;
    virtual void refactorize() = 0;
};

}  // namespace kavach::numerics
