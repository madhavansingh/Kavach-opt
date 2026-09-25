#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/error.hpp"

namespace kavach::model {

// Types for stable internal indexing
using VarId = std::uint32_t;
using ConstrId = std::uint32_t;

inline constexpr VarId INVALID_VAR_ID = std::numeric_limits<VarId>::max();
inline constexpr ConstrId INVALID_CONSTR_ID = std::numeric_limits<ConstrId>::max();
inline constexpr double INFINITY_BOUND = std::numeric_limits<double>::infinity();

/**
 * @brief Variable classification.
 */
enum class VariableType : std::uint8_t { CONTINUOUS, INTEGER, BINARY };

[[nodiscard]] constexpr std::string_view to_string(VariableType type) noexcept {
    switch (type) {
        case VariableType::CONTINUOUS:
            return "CONTINUOUS";
        case VariableType::INTEGER:
            return "INTEGER";
        case VariableType::BINARY:
            return "BINARY";
    }
    return "UNKNOWN";
}

inline std::ostream& operator<<(std::ostream& ostream, VariableType type) {
    return ostream << to_string(type);
}

/**
 * @brief Constraint relation / sense.
 */
enum class ConstraintSense : std::uint8_t {
    LESS_EQUAL,    ///< <=
    EQUAL,         ///< ==
    GREATER_EQUAL  ///< >=
};

[[nodiscard]] constexpr std::string_view to_string(ConstraintSense sense) noexcept {
    switch (sense) {
        case ConstraintSense::LESS_EQUAL:
            return "<=";
        case ConstraintSense::EQUAL:
            return "==";
        case ConstraintSense::GREATER_EQUAL:
            return ">=";
    }
    return "??";
}

inline std::ostream& operator<<(std::ostream& ostream, ConstraintSense sense) {
    return ostream << to_string(sense);
}

/**
 * @brief Objective optimization sense.
 */
enum class ObjectiveSense : std::uint8_t { MINIMIZE, MAXIMIZE };

[[nodiscard]] constexpr std::string_view to_string(ObjectiveSense sense) noexcept {
    switch (sense) {
        case ObjectiveSense::MINIMIZE:
            return "MINIMIZE";
        case ObjectiveSense::MAXIMIZE:
            return "MAXIMIZE";
    }
    return "UNKNOWN";
}

inline std::ostream& operator<<(std::ostream& ostream, ObjectiveSense sense) {
    return ostream << to_string(sense);
}

/**
 * @brief Canonical representation of an optimization variable.
 */
struct Variable {
    VarId id{INVALID_VAR_ID};
    std::string name;
    VariableType type{VariableType::CONTINUOUS};
    double lower_bound{0.0};
    double upper_bound{INFINITY_BOUND};

    [[nodiscard]] bool is_continuous() const noexcept { return type == VariableType::CONTINUOUS; }
    [[nodiscard]] bool is_integer() const noexcept { return type == VariableType::INTEGER; }
    [[nodiscard]] bool is_binary() const noexcept { return type == VariableType::BINARY; }
    [[nodiscard]] bool is_free() const noexcept {
        return std::isinf(lower_bound) && lower_bound < 0.0 && std::isinf(upper_bound) &&
               upper_bound > 0.0;
    }
    [[nodiscard]] bool has_lower_bound() const noexcept {
        return !std::isinf(lower_bound) || lower_bound > 0.0;
    }
    [[nodiscard]] bool has_upper_bound() const noexcept {
        return !std::isinf(upper_bound) || upper_bound < 0.0;
    }
};

/**
 * @brief A single sparse linear coefficient term: coeff * x[var_id].
 */
struct LinearTerm {
    VarId var_id{INVALID_VAR_ID};
    double coefficient{0.0};

    bool operator==(const LinearTerm& other) const noexcept {
        return var_id == other.var_id && coefficient == other.coefficient;
    }
};

/**
 * @brief Sparse linear expression representation.
 * Terms are maintained in canonical sorted order (by VarId ascending) with no duplicate
 * VarIds and no zero coefficients.
 */
class SparseLinearExpression {
public:
    SparseLinearExpression() = default;
    explicit SparseLinearExpression(std::vector<LinearTerm> terms);

    void add_term(VarId var_id, double coefficient);
    void canonicalize();

    [[nodiscard]] const std::vector<LinearTerm>& terms() const noexcept { return terms_; }
    [[nodiscard]] std::size_t num_terms() const noexcept { return terms_.size(); }
    [[nodiscard]] bool empty() const noexcept { return terms_.empty(); }
    void clear() noexcept { terms_.clear(); }

    [[nodiscard]] double get_coefficient(VarId var_id) const noexcept;

private:
    std::vector<LinearTerm> terms_;
};

/**
 * @brief Quadratic term in objective: coeff * x[var1] * x[var2].
 * Normalized such that var1_id <= var2_id.
 */
struct QuadraticTerm {
    VarId var1_id{INVALID_VAR_ID};
    VarId var2_id{INVALID_VAR_ID};
    double coefficient{0.0};

    bool operator==(const QuadraticTerm& other) const noexcept {
        return var1_id == other.var1_id && var2_id == other.var2_id &&
               coefficient == other.coefficient;
    }
};

/**
 * @brief Sparse quadratic expression representation.
 * Terms are canonicalized with var1_id <= var2_id, sorted lexicographically,
 * duplicates merged by addition, and exact zero coefficients removed.
 */
class SparseQuadraticExpression {
public:
    SparseQuadraticExpression() = default;
    explicit SparseQuadraticExpression(std::vector<QuadraticTerm> terms);

    void add_term(VarId var1_id, VarId var2_id, double coefficient);
    void canonicalize();

    [[nodiscard]] const std::vector<QuadraticTerm>& terms() const noexcept { return terms_; }
    [[nodiscard]] std::size_t num_terms() const noexcept { return terms_.size(); }
    [[nodiscard]] bool empty() const noexcept { return terms_.empty(); }
    void clear() noexcept { terms_.clear(); }

private:
    std::vector<QuadraticTerm> terms_;
};

/**
 * @brief Canonical representation of a linear constraint: a^T x (sense) rhs.
 */
struct Constraint {
    ConstrId id{INVALID_CONSTR_ID};
    std::string name;
    SparseLinearExpression expression;
    ConstraintSense sense{ConstraintSense::LESS_EQUAL};
    double rhs{0.0};

    [[nodiscard]] std::size_t num_nonzeros() const noexcept { return expression.num_terms(); }
};

/**
 * @brief Complete model objective function representation:
 * Sense + constant offset + linear terms + quadratic terms.
 */
struct Objective {
    ObjectiveSense sense{ObjectiveSense::MINIMIZE};
    double offset{0.0};
    SparseLinearExpression linear_terms;
    SparseQuadraticExpression quadratic_terms;

    [[nodiscard]] bool is_quadratic() const noexcept { return !quadratic_terms.empty(); }
};

/**
 * @brief Detailed validation error report item.
 */
struct ValidationError {
    enum class Severity : std::uint8_t { ERROR, WARNING };
    Severity severity{Severity::ERROR};
    std::string component;  // "variable", "constraint", "objective", "general"
    std::size_t index{0};
    std::string name;
    std::string message;
};

/**
 * @brief Validation result holding error/warning diagnostics.
 */
struct ValidationResult {
    bool valid{true};
    std::vector<ValidationError> errors;

    void add_error(std::string component, std::size_t index, std::string name,
                   std::string message) {
        valid = false;
        errors.push_back({.severity = ValidationError::Severity::ERROR,
                          .component = std::move(component),
                          .index = index,
                          .name = std::move(name),
                          .message = std::move(message)});
    }

    [[nodiscard]] bool has_errors() const noexcept { return !valid; }
};

/**
 * @brief Production Canonical Optimization Model [A].
 *
 * Fully conforms to Blueprint Section 4.2 [A], Section 5, 6, 7, and 11.
 * Serves as the single immutable/validated contract consumed by Presolve, LP,
 * MILP, QP, Strategy, and Independent Verification layers.
 */
class CanonicalModel {
public:
    explicit CanonicalModel(std::string name = "");
    virtual ~CanonicalModel() = default;

    CanonicalModel(const CanonicalModel&) = default;
    CanonicalModel& operator=(const CanonicalModel&) = default;
    CanonicalModel(CanonicalModel&&) noexcept = default;
    CanonicalModel& operator=(CanonicalModel&&) noexcept = default;

    // --- Model Identification ---
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    void set_name(std::string name) { name_ = std::move(name); }

    // --- Variable Management ---
    VarId add_variable(const std::string& name = "", VariableType type = VariableType::CONTINUOUS,
                       double lower_bound = 0.0, double upper_bound = INFINITY_BOUND);

    VarId add_binary_variable(const std::string& name = "");
    VarId add_integer_variable(const std::string& name = "", double lower_bound = 0.0,
                               double upper_bound = INFINITY_BOUND);

    [[nodiscard]] std::size_t num_variables() const noexcept { return variables_.size(); }
    [[nodiscard]] const Variable& variable(VarId var_id) const;
    [[nodiscard]] const std::vector<Variable>& variables() const noexcept { return variables_; }
    [[nodiscard]] std::optional<VarId> find_variable_id(std::string_view name) const;

    // --- Constraint Management ---
    ConstrId add_constraint(const std::string& name, SparseLinearExpression expr,
                            ConstraintSense sense, double rhs);

    ConstrId add_constraint(SparseLinearExpression expr, ConstraintSense sense, double rhs) {
        return add_constraint("", std::move(expr), sense, rhs);
    }

    [[nodiscard]] std::size_t num_constraints() const noexcept { return constraints_.size(); }
    [[nodiscard]] const Constraint& constraint(ConstrId constr_id) const;
    [[nodiscard]] const std::vector<Constraint>& constraints() const noexcept {
        return constraints_;
    }
    [[nodiscard]] std::optional<ConstrId> find_constraint_id(std::string_view name) const;

    // --- Objective Management ---
    void set_objective_sense(ObjectiveSense sense) noexcept { objective_.sense = sense; }
    [[nodiscard]] ObjectiveSense objective_sense() const noexcept { return objective_.sense; }

    void set_objective_offset(double offset) noexcept { objective_.offset = offset; }
    [[nodiscard]] double objective_offset() const noexcept { return objective_.offset; }

    void set_objective_linear_coefficient(VarId var_id, double coeff);
    void add_objective_linear_term(VarId var_id, double coeff);

    void set_objective_quadratic_coefficient(VarId var1_id, VarId var2_id, double coeff);
    void add_objective_quadratic_term(VarId var1_id, VarId var2_id, double coeff);

    [[nodiscard]] const Objective& objective() const noexcept { return objective_; }
    [[nodiscard]] Objective& objective() noexcept { return objective_; }

    // --- Model Metrics & Invariants ---
    [[nodiscard]] std::size_t num_nonzeros() const noexcept;
    [[nodiscard]] std::size_t num_integer_variables() const noexcept;
    [[nodiscard]] std::size_t num_binary_variables() const noexcept;
    [[nodiscard]] std::size_t num_continuous_variables() const noexcept;

    [[nodiscard]] bool is_lp() const noexcept;
    [[nodiscard]] bool is_milp() const noexcept;
    [[nodiscard]] bool is_qp() const noexcept;

    // --- Canonicalization & Validation ---
    void canonicalize();
    [[nodiscard]] ValidationResult validate() const;
    void check_valid() const;  // Throws InputException or NumericalException if invalid

private:
    std::string name_;
    std::vector<Variable> variables_;
    std::vector<Constraint> constraints_;
    Objective objective_;

    // Fast O(1) lookup by non-empty human-readable identifier
    std::unordered_map<std::string, VarId> var_name_to_id_;
    std::unordered_map<std::string, ConstrId> constr_name_to_id_;

    void validate_variables(ValidationResult& result) const;
    void validate_constraints(ValidationResult& result) const;
    void validate_objective(ValidationResult& result) const;
};

}  // namespace kavach::model
