#include "numerics/model_converter.hpp"

namespace kavach::numerics {

CsrMatrix canonical_model_to_csr(const model::CanonicalModel& model) {
    const std::size_t m = model.num_constraints();
    const std::size_t n = model.num_variables();
    const std::size_t nnz = model.num_nonzeros();

    std::vector<std::size_t> row_ptr(m + 1, 0);
    std::vector<std::size_t> col_idx;
    std::vector<double> values;
    col_idx.reserve(nnz);
    values.reserve(nnz);

    row_ptr[0] = 0;
    for (std::size_t i = 0; i < m; ++i) {
        const auto& constr = model.constraint(static_cast<model::ConstrId>(i));
        const auto& terms = constr.expression.terms();
        for (const auto& term : terms) {
            col_idx.push_back(term.var_id);
            values.push_back(term.coefficient);
        }
        row_ptr[i + 1] = col_idx.size();
    }

    // Since CanonicalModel guarantees canonicalized expressions (sorted by var_id, no zeros),
    // we can pass canonicalize_entries = false.
    return {m, n, std::move(row_ptr), std::move(col_idx), std::move(values), false};
}

CscMatrix canonical_model_to_csc(const model::CanonicalModel& model) {
    return csr_to_csc(canonical_model_to_csr(model));
}

}  // namespace kavach::numerics
