#include "model/model.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "core/error.hpp"

namespace kavach::model {

// ============================================================================
// SparseLinearExpression Implementation
// ============================================================================

SparseLinearExpression::SparseLinearExpression(std::vector<LinearTerm> terms)
    : terms_(std::move(terms)) {
    canonicalize();
}

void SparseLinearExpression::add_term(VarId var_id, double coefficient) {
    if (coefficient != 0.0 && !std::isnan(coefficient)) {
        terms_.push_back({.var_id = var_id, .coefficient = coefficient});
    }
}

void SparseLinearExpression::canonicalize() {
    if (terms_.empty()) {
        return;
    }

    // Sort ascending by VarId
    std::ranges::sort(terms_, [](const LinearTerm& a, const LinearTerm& b) noexcept {
        return a.var_id < b.var_id;
    });

    std::vector<LinearTerm> merged;
    merged.reserve(terms_.size());

    for (const auto& term : terms_) {
        if (merged.empty() || merged.back().var_id != term.var_id) {
            merged.push_back(term);
        } else {
            merged.back().coefficient += term.coefficient;
        }
    }

    // Eliminate terms with exactly zero coefficient (or NaN/Inf handled in validation)
    terms_.clear();
    for (const auto& term : merged) {
        if (term.coefficient != 0.0) {
            terms_.push_back(term);
        }
    }
}

double SparseLinearExpression::get_coefficient(VarId var_id) const noexcept {
    // Binary search with C++20 ranges projection because terms_ are sorted by VarId
    auto it = std::ranges::lower_bound(terms_, var_id, {}, &LinearTerm::var_id);
    if (it != terms_.end() && it->var_id == var_id) {
        return it->coefficient;
    }
    return 0.0;
}

// ============================================================================
// SparseQuadraticExpression Implementation
// ============================================================================

SparseQuadraticExpression::SparseQuadraticExpression(std::vector<QuadraticTerm> terms)
    : terms_(std::move(terms)) {
    canonicalize();
}

void SparseQuadraticExpression::add_term(VarId var1_id, VarId var2_id, double coefficient) {
    if (coefficient != 0.0 && !std::isnan(coefficient)) {
        if (var1_id > var2_id) {
            std::swap(var1_id, var2_id);
        }
        terms_.push_back({.var1_id = var1_id, .var2_id = var2_id, .coefficient = coefficient});
    }
}

void SparseQuadraticExpression::canonicalize() {
    if (terms_.empty()) {
        return;
    }

    // Ensure all pairs have var1_id <= var2_id
    for (auto& term : terms_) {
        if (term.var1_id > term.var2_id) {
            std::swap(term.var1_id, term.var2_id);
        }
    }

    // Sort lexicographically by (var1_id, var2_id)
    std::ranges::sort(terms_, [](const QuadraticTerm& a, const QuadraticTerm& b) noexcept {
        if (a.var1_id != b.var1_id) {
            return a.var1_id < b.var1_id;
        }
        return a.var2_id < b.var2_id;
    });

    std::vector<QuadraticTerm> merged;
    merged.reserve(terms_.size());

    for (const auto& term : terms_) {
        if (merged.empty() || merged.back().var1_id != term.var1_id ||
            merged.back().var2_id != term.var2_id) {
            merged.push_back(term);
        } else {
            merged.back().coefficient += term.coefficient;
        }
    }

    // Eliminate terms with exactly zero coefficient
    terms_.clear();
    for (const auto& term : merged) {
        if (term.coefficient != 0.0) {
            terms_.push_back(term);
        }
    }
}

// ============================================================================
// CanonicalModel Implementation
// ============================================================================

CanonicalModel::CanonicalModel(std::string name) : name_(std::move(name)) {}

VarId CanonicalModel::add_variable(const std::string& name, VariableType type, double lower_bound,
                                   double upper_bound) {
    if (!name.empty() && var_name_to_id_.contains(name)) {
        throw core::InputException("Duplicate variable identifier: '" + name + "'");
    }

    if (type == VariableType::BINARY) {
        // If default unconstrained bounds were provided, normalize to standard binary bounds [0, 1]
        if (lower_bound == 0.0 && std::isinf(upper_bound)) {
            upper_bound = 1.0;
        }
    }

    const auto id = static_cast<VarId>(variables_.size());
    Variable var{.id = id,
                 .name = name,
                 .type = type,
                 .lower_bound = lower_bound,
                 .upper_bound = upper_bound};
    variables_.push_back(std::move(var));

    if (!name.empty()) {
        var_name_to_id_[name] = id;
    }

    return id;
}

VarId CanonicalModel::add_binary_variable(const std::string& name) {
    return add_variable(name, VariableType::BINARY, 0.0, 1.0);
}

VarId CanonicalModel::add_integer_variable(const std::string& name, double lower_bound,
                                           double upper_bound) {
    return add_variable(name, VariableType::INTEGER, lower_bound, upper_bound);
}

const Variable& CanonicalModel::variable(VarId var_id) const {
    if (var_id >= variables_.size()) {
        throw core::InputException("Variable ID out of range: " + std::to_string(var_id));
    }
    return variables_[var_id];
}

Variable& CanonicalModel::variable(VarId var_id) {
    if (var_id >= variables_.size()) {
        throw core::InputException("Variable ID out of range: " + std::to_string(var_id));
    }
    return variables_[var_id];
}

void CanonicalModel::set_variable_bounds(VarId var_id, double lower_bound, double upper_bound) {
    if (var_id >= variables_.size()) {
        throw core::InputException("Variable ID out of range: " + std::to_string(var_id));
    }
    variables_[var_id].lower_bound = lower_bound;
    variables_[var_id].upper_bound = upper_bound;
}

void CanonicalModel::set_variable_lower_bound(VarId var_id, double lower_bound) {
    if (var_id >= variables_.size()) {
        throw core::InputException("Variable ID out of range: " + std::to_string(var_id));
    }
    variables_[var_id].lower_bound = lower_bound;
}

void CanonicalModel::set_variable_upper_bound(VarId var_id, double upper_bound) {
    if (var_id >= variables_.size()) {
        throw core::InputException("Variable ID out of range: " + std::to_string(var_id));
    }
    variables_[var_id].upper_bound = upper_bound;
}

void CanonicalModel::set_variable_type(VarId var_id, VariableType type) {
    if (var_id >= variables_.size()) {
        throw core::InputException("Variable ID out of range: " + std::to_string(var_id));
    }
    variables_[var_id].type = type;
}

std::optional<VarId> CanonicalModel::find_variable_id(std::string_view name) const {
    auto iter = var_name_to_id_.find(std::string(name));
    if (iter != var_name_to_id_.end()) {
        return iter->second;
    }
    return std::nullopt;
}

ConstrId CanonicalModel::add_constraint(const std::string& name, SparseLinearExpression expr,
                                        ConstraintSense sense, double rhs) {
    if (!name.empty() && constr_name_to_id_.contains(name)) {
        throw core::InputException("Duplicate constraint identifier: '" + name + "'");
    }

    expr.canonicalize();

    const auto constr_id = static_cast<ConstrId>(constraints_.size());
    Constraint constr{
        .id = constr_id, .name = name, .expression = std::move(expr), .sense = sense, .rhs = rhs};
    constraints_.push_back(std::move(constr));

    if (!name.empty()) {
        constr_name_to_id_[name] = constr_id;
    }

    return constr_id;
}

const Constraint& CanonicalModel::constraint(ConstrId constr_id) const {
    if (constr_id >= constraints_.size()) {
        throw core::InputException("Constraint ID out of range: " + std::to_string(constr_id));
    }
    return constraints_[constr_id];
}

Constraint& CanonicalModel::constraint(ConstrId constr_id) {
    if (constr_id >= constraints_.size()) {
        throw core::InputException("Constraint ID out of range: " + std::to_string(constr_id));
    }
    return constraints_[constr_id];
}

std::optional<ConstrId> CanonicalModel::find_constraint_id(std::string_view name) const {
    auto iter = constr_name_to_id_.find(std::string(name));
    if (iter != constr_name_to_id_.end()) {
        return iter->second;
    }
    return std::nullopt;
}

void CanonicalModel::set_objective_linear_coefficient(VarId var_id, double coeff) {
    if (var_id >= variables_.size()) {
        throw core::InputException("Objective variable ID out of range: " + std::to_string(var_id));
    }
    objective_.linear_terms.add_term(var_id, coeff);
    objective_.linear_terms.canonicalize();
}

void CanonicalModel::add_objective_linear_term(VarId var_id, double coeff) {
    if (var_id >= variables_.size()) {
        throw core::InputException("Objective variable ID out of range: " + std::to_string(var_id));
    }
    objective_.linear_terms.add_term(var_id, coeff);
}

void CanonicalModel::set_objective_quadratic_coefficient(VarId var1_id, VarId var2_id,
                                                         double coeff) {
    if (var1_id >= variables_.size() || var2_id >= variables_.size()) {
        throw core::InputException("Quadratic objective variable ID out of range: (" +
                                   std::to_string(var1_id) + ", " + std::to_string(var2_id) + ")");
    }
    objective_.quadratic_terms.add_term(var1_id, var2_id, coeff);
    objective_.quadratic_terms.canonicalize();
}

void CanonicalModel::add_objective_quadratic_term(VarId var1_id, VarId var2_id, double coeff) {
    if (var1_id >= variables_.size() || var2_id >= variables_.size()) {
        throw core::InputException("Quadratic objective variable ID out of range: (" +
                                   std::to_string(var1_id) + ", " + std::to_string(var2_id) + ")");
    }
    objective_.quadratic_terms.add_term(var1_id, var2_id, coeff);
}

std::size_t CanonicalModel::num_nonzeros() const noexcept {
    std::size_t nnz = 0;
    for (const auto& c : constraints_) {
        nnz += c.num_nonzeros();
    }
    return nnz;
}

std::size_t CanonicalModel::num_integer_variables() const noexcept {
    return static_cast<std::size_t>(
        std::count_if(variables_.begin(), variables_.end(),
                      [](const Variable& v) noexcept { return v.type == VariableType::INTEGER; }));
}

std::size_t CanonicalModel::num_binary_variables() const noexcept {
    return static_cast<std::size_t>(
        std::count_if(variables_.begin(), variables_.end(),
                      [](const Variable& v) noexcept { return v.type == VariableType::BINARY; }));
}

std::size_t CanonicalModel::num_continuous_variables() const noexcept {
    return static_cast<std::size_t>(std::count_if(
        variables_.begin(), variables_.end(),
        [](const Variable& v) noexcept { return v.type == VariableType::CONTINUOUS; }));
}

bool CanonicalModel::is_lp() const noexcept {
    return num_integer_variables() == 0 && num_binary_variables() == 0 &&
           !objective_.is_quadratic();
}

bool CanonicalModel::is_milp() const noexcept {
    return (num_integer_variables() > 0 || num_binary_variables() > 0) &&
           !objective_.is_quadratic();
}

bool CanonicalModel::is_qp() const noexcept {
    return num_integer_variables() == 0 && num_binary_variables() == 0 && objective_.is_quadratic();
}

bool CanonicalModel::is_miqp() const noexcept {
    return (num_integer_variables() > 0 || num_binary_variables() > 0) && objective_.is_quadratic();
}

void CanonicalModel::canonicalize() {
    for (auto& c : constraints_) {
        c.expression.canonicalize();
    }
    objective_.linear_terms.canonicalize();
    objective_.quadratic_terms.canonicalize();
}

void CanonicalModel::validate_variables(ValidationResult& result) const {
    for (const auto& var : variables_) {
        if (std::isnan(var.lower_bound) || std::isnan(var.upper_bound)) {
            result.add_error("variable", var.id, var.name, "Bounds contain NaN");
        } else if (var.lower_bound > var.upper_bound) {
            result.add_error("variable", var.id, var.name,
                             "Lower bound exceeds upper bound: " + std::to_string(var.lower_bound) +
                                 " > " + std::to_string(var.upper_bound));
        } else if (std::isinf(var.lower_bound) && var.lower_bound > 0.0) {
            result.add_error("variable", var.id, var.name,
                             "Lower bound cannot be positive infinity");
        } else if (std::isinf(var.upper_bound) && var.upper_bound < 0.0) {
            result.add_error("variable", var.id, var.name,
                             "Upper bound cannot be negative infinity");
        }

        if (var.type == VariableType::BINARY) {
            if (std::isinf(var.lower_bound) || std::isinf(var.upper_bound)) {
                result.add_error("variable", var.id, var.name,
                                 "Binary variable must have finite bounds");
            } else if (var.lower_bound < 0.0 || var.upper_bound > 1.0) {
                result.add_error("variable", var.id, var.name,
                                 "Binary variable bounds must be within [0, 1]");
            }
        }
    }
}

void CanonicalModel::validate_constraints(ValidationResult& result) const {
    for (const auto& constr : constraints_) {
        if (std::isnan(constr.rhs) || std::isinf(constr.rhs)) {
            result.add_error("constraint", constr.id, constr.name, "Constraint RHS must be finite");
        }

        for (const auto& term : constr.expression.terms()) {
            if (term.var_id >= variables_.size()) {
                result.add_error("constraint", constr.id, constr.name,
                                 "References invalid variable ID: " + std::to_string(term.var_id));
            }
            if (std::isnan(term.coefficient) || std::isinf(term.coefficient)) {
                result.add_error("constraint", constr.id, constr.name,
                                 "Linear coefficient must be finite");
            }
        }
    }
}

void CanonicalModel::validate_objective(ValidationResult& result) const {
    if (std::isnan(objective_.offset) || std::isinf(objective_.offset)) {
        result.add_error("objective", 0, "offset", "Objective constant offset must be finite");
    }

    for (const auto& term : objective_.linear_terms.terms()) {
        if (term.var_id >= variables_.size()) {
            result.add_error(
                "objective", term.var_id, "linear_term",
                "Objective references invalid variable ID: " + std::to_string(term.var_id));
        }
        if (std::isnan(term.coefficient) || std::isinf(term.coefficient)) {
            result.add_error("objective", term.var_id, "linear_term",
                             "Objective coefficient must be finite");
        }
    }

    for (const auto& term : objective_.quadratic_terms.terms()) {
        if (term.var1_id >= variables_.size() || term.var2_id >= variables_.size()) {
            result.add_error("objective", 0, "quadratic_term",
                             "Quadratic objective references invalid variable ID: (" +
                                 std::to_string(term.var1_id) + ", " +
                                 std::to_string(term.var2_id) + ")");
        }
        if (std::isnan(term.coefficient) || std::isinf(term.coefficient)) {
            result.add_error("objective", 0, "quadratic_term",
                             "Quadratic coefficient must be finite");
        }
    }
}

ValidationResult CanonicalModel::validate() const {
    ValidationResult result;
    validate_variables(result);
    validate_constraints(result);
    validate_objective(result);
    return result;
}

void CanonicalModel::check_valid() const {
    auto val_report = validate();
    if (!val_report.valid) {
        std::ostringstream oss;
        oss << "Model validation failed with " << val_report.errors.size() << " error(s):\n";
        for (const auto& err : val_report.errors) {
            oss << "  [" << err.component << " " << err.index << " (" << err.name
                << ")]: " << err.message << "\n";
        }
        throw core::InputException(oss.str());
    }
}

}  // namespace kavach::model
