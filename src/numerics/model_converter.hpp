#pragma once

#include "model/model.hpp"
#include "numerics/sparse_matrix.hpp"

namespace kavach::numerics {

/**
 * @brief Converts the linear constraint system of a CanonicalModel into a CsrMatrix.
 *
 * Each row corresponds to a constraint index (0 to model.num_constraints() - 1).
 * Each column corresponds to a variable ID (0 to model.num_variables() - 1).
 * Invariant: Zero dense allocation. Strictly O(rows + nnz) time and space complexity.
 */
[[nodiscard]] CsrMatrix canonical_model_to_csr(const model::CanonicalModel& model);

/**
 * @brief Converts the linear constraint system of a CanonicalModel into a CscMatrix.
 */
[[nodiscard]] CscMatrix canonical_model_to_csc(const model::CanonicalModel& model);

}  // namespace kavach::numerics
