#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "io/parser.hpp"

namespace kavach::io {

namespace {

[[nodiscard]] bool iequals(std::string_view a, std::string_view b) noexcept {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i])) !=
            std::tolower(static_cast<unsigned char>(b[i]))) {
            return false;
        }
    }
    return true;
}

enum class TokenKind : std::uint8_t {
    IDENTIFIER,
    NUMBER,
    COLON,          // :
    PLUS,           // +
    MINUS,          // -
    STAR,           // *
    CARET,          // ^
    SLASH,          // /
    LBRACKET,       // [
    RBRACKET,       // ]
    LESS_EQUAL,     // <= or <
    GREATER_EQUAL,  // >= or >
    EQUAL,          // = or ==
    END_OF_FILE
};

struct Token {
    TokenKind kind{TokenKind::END_OF_FILE};
    std::string text;
    double number_value{0.0};
    std::size_t line{1};
};

class LpTokenizer {
public:
    explicit LpTokenizer(std::string_view input, std::string filename)
        : input_(input), filename_(std::move(filename)) {}

    Token next_token() {
        skip_whitespace_and_comments();
        if (pos_ >= input_.size()) {
            return {.kind = TokenKind::END_OF_FILE, .text = "", .number_value = 0.0, .line = line_};
        }

        const char c = input_[pos_];
        const std::size_t token_line = line_;

        if (c == ':') {
            ++pos_;
            return {.kind = TokenKind::COLON, .text = ":", .line = token_line};
        }
        if (c == '+') {
            ++pos_;
            return {.kind = TokenKind::PLUS, .text = "+", .line = token_line};
        }
        if (c == '-') {
            ++pos_;
            return {.kind = TokenKind::MINUS, .text = "-", .line = token_line};
        }
        if (c == '*') {
            ++pos_;
            return {.kind = TokenKind::STAR, .text = "*", .line = token_line};
        }
        if (c == '^') {
            ++pos_;
            return {.kind = TokenKind::CARET, .text = "^", .line = token_line};
        }
        if (c == '/') {
            ++pos_;
            return {.kind = TokenKind::SLASH, .text = "/", .line = token_line};
        }
        if (c == '[') {
            ++pos_;
            return {.kind = TokenKind::LBRACKET, .text = "[", .line = token_line};
        }
        if (c == ']') {
            ++pos_;
            return {.kind = TokenKind::RBRACKET, .text = "]", .line = token_line};
        }
        if (c == '<') {
            ++pos_;
            if (pos_ < input_.size() && input_[pos_] == '=') {
                ++pos_;
            }
            return {.kind = TokenKind::LESS_EQUAL, .text = "<=", .line = token_line};
        }
        if (c == '>') {
            ++pos_;
            if (pos_ < input_.size() && input_[pos_] == '=') {
                ++pos_;
            }
            return {.kind = TokenKind::GREATER_EQUAL, .text = ">=", .line = token_line};
        }
        if (c == '=') {
            ++pos_;
            if (pos_ < input_.size() && input_[pos_] == '=') {
                ++pos_;
            }
            return {.kind = TokenKind::EQUAL, .text = "=", .line = token_line};
        }

        // Numbers: digits or dot followed by digit
        if (std::isdigit(static_cast<unsigned char>(c)) != 0 ||
            (c == '.' && pos_ + 1 < input_.size() &&
             std::isdigit(static_cast<unsigned char>(input_[pos_ + 1])) != 0)) {
            std::size_t start = pos_;
            while (pos_ < input_.size() &&
                   (std::isdigit(static_cast<unsigned char>(input_[pos_])) != 0 ||
                    input_[pos_] == '.')) {
                ++pos_;
            }
            // Exponent handling (e.g. 1e-4)
            if (pos_ < input_.size() && (input_[pos_] == 'e' || input_[pos_] == 'E')) {
                ++pos_;
                if (pos_ < input_.size() && (input_[pos_] == '+' || input_[pos_] == '-')) {
                    ++pos_;
                }
                while (pos_ < input_.size() &&
                       std::isdigit(static_cast<unsigned char>(input_[pos_])) != 0) {
                    ++pos_;
                }
            }

            std::string num_str(input_.substr(start, pos_ - start));
            char* end_ptr = nullptr;
            double val = std::strtod(num_str.c_str(), &end_ptr);
            if (end_ptr == num_str.c_str()) {
                throw ParserException(filename_, token_line, "TOKENIZER",
                                      "Invalid number literal: " + num_str);
            }
            return {.kind = TokenKind::NUMBER,
                    .text = std::move(num_str),
                    .number_value = val,
                    .line = token_line};
        }

        // Identifiers: letters, digits, underscores, dots, hashes, etc.
        if (std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_' || c == '.' || c == '$') {
            std::size_t start = pos_;
            while (pos_ < input_.size() &&
                   (std::isalnum(static_cast<unsigned char>(input_[pos_])) != 0 ||
                    input_[pos_] == '_' || input_[pos_] == '.' || input_[pos_] == '$' ||
                    input_[pos_] == '#' || input_[pos_] == '@' || input_[pos_] == '[' ||
                    input_[pos_] == ']' || input_[pos_] == '(' || input_[pos_] == ')')) {
                // If it's bracket, break if used for quadratic terms
                if (input_[pos_] == '[' || input_[pos_] == ']') {
                    break;
                }
                ++pos_;
            }
            std::string id_str(input_.substr(start, pos_ - start));
            return {.kind = TokenKind::IDENTIFIER,
                    .text = std::move(id_str),
                    .number_value = 0.0,
                    .line = token_line};
        }

        throw ParserException(filename_, token_line, "TOKENIZER",
                              "Unexpected character in LP input: '" + std::string(1, c) + "'");
    }

private:
    void skip_whitespace_and_comments() {
        while (pos_ < input_.size()) {
            char c = input_[pos_];
            if (c == '\n') {
                ++line_;
                ++pos_;
            } else if (std::isspace(static_cast<unsigned char>(c)) != 0) {
                ++pos_;
            } else if (c == '\\') {
                // Comment to end of line
                while (pos_ < input_.size() && input_[pos_] != '\n') {
                    ++pos_;
                }
            } else {
                break;
            }
        }
    }

    std::string_view input_;
    std::string filename_;
    std::size_t pos_{0};
    std::size_t line_{1};
};

enum class LpSection : std::uint8_t { OBJECTIVE, SUBJECT_TO, BOUNDS, GENERALS, BINARIES, END };

}  // namespace

std::unique_ptr<model::CanonicalModel> LpParser::parse_string(std::string_view content,
                                                              std::string_view source_name,
                                                              const ParserOptions& options) {
    const std::string filename(source_name);
    LpTokenizer tokenizer(content, filename);

    auto canonical_model = std::make_unique<model::CanonicalModel>();
    std::unordered_map<std::string, model::VarId> var_map;

    auto get_or_create_var = [&](const std::string& name) -> model::VarId {
        auto it = var_map.find(name);
        if (it != var_map.end()) {
            return it->second;
        }
        auto vid = canonical_model->add_variable(name, model::VariableType::CONTINUOUS, 0.0,
                                                 model::INFINITY_BOUND);
        var_map[name] = vid;
        return vid;
    };

    std::vector<Token> tokens;
    for (Token tok = tokenizer.next_token(); tok.kind != TokenKind::END_OF_FILE;
         tok = tokenizer.next_token()) {
        tokens.push_back(std::move(tok));
    }

    std::size_t cursor = 0;
    const std::size_t num_tokens = tokens.size();

    auto peek = [&](std::size_t offset = 0) -> const Token& {
        static const Token eof_token{.kind = TokenKind::END_OF_FILE, .text = "", .line = 0};
        if (cursor + offset < num_tokens) {
            return tokens[cursor + offset];
        }
        return eof_token;
    };

    auto check_keyword = [&](std::string_view kw) -> bool {
        return peek().kind == TokenKind::IDENTIFIER && iequals(peek().text, kw);
    };

    // 1. Parse Objective Section
    if (cursor >= num_tokens) {
        throw ParserException(filename, 1, "LP", "Empty LP file");
    }

    if (check_keyword("minimize") || check_keyword("min") || check_keyword("minimum")) {
        canonical_model->set_objective_sense(model::ObjectiveSense::MINIMIZE);
        ++cursor;
    } else if (check_keyword("maximize") || check_keyword("max") || check_keyword("maximum")) {
        canonical_model->set_objective_sense(model::ObjectiveSense::MAXIMIZE);
        ++cursor;
    } else {
        throw ParserException(filename, peek().line, "OBJECTIVE",
                              "Expected Minimize or Maximize objective keyword");
    }

    // Optional objective name: e.g. "obj:"
    if (peek(0).kind == TokenKind::IDENTIFIER && peek(1).kind == TokenKind::COLON) {
        cursor += 2;
    }

    auto is_section_header = [&]() -> bool {
        if (cursor >= num_tokens) {
            return true;
        }
        if (check_keyword("subject") && peek(1).kind == TokenKind::IDENTIFIER &&
            iequals(peek(1).text, "to")) {
            return true;
        }
        if (check_keyword("such") && peek(1).kind == TokenKind::IDENTIFIER &&
            iequals(peek(1).text, "that")) {
            return true;
        }
        if (check_keyword("s.t.") || check_keyword("st.") || check_keyword("st") ||
            check_keyword("s.t")) {
            return true;
        }
        if (check_keyword("bounds") || check_keyword("bound") || check_keyword("generals") ||
            check_keyword("general") || check_keyword("gen") || check_keyword("integers") ||
            check_keyword("integer") || check_keyword("int") || check_keyword("binaries") ||
            check_keyword("binary") || check_keyword("bin") || check_keyword("end")) {
            return true;
        }
        return false;
    };

    // Parse objective expression (linear terms, constant offsets, bracketed quadratic terms)
    while (cursor < num_tokens && !is_section_header()) {
        double sign = 1.0;
        if (peek().kind == TokenKind::PLUS) {
            ++cursor;
        } else if (peek().kind == TokenKind::MINUS) {
            sign = -1.0;
            ++cursor;
        }

        if (peek().kind == TokenKind::LBRACKET) {
            // Bracketed quadratic expression: [ ... ] or [ ... ] / 2
            ++cursor;  // Consume '['
            double quad_divisor = 1.0;

            struct TempQuadTerm {
                model::VarId var1;
                model::VarId var2;
                double coeff;
            };
            std::vector<TempQuadTerm> quad_terms;

            while (cursor < num_tokens && peek().kind != TokenKind::RBRACKET) {
                double term_sign = 1.0;
                if (peek().kind == TokenKind::PLUS) {
                    ++cursor;
                } else if (peek().kind == TokenKind::MINUS) {
                    term_sign = -1.0;
                    ++cursor;
                }

                double coeff = 1.0;
                if (peek().kind == TokenKind::NUMBER) {
                    coeff = peek().number_value;
                    ++cursor;
                    if (peek().kind == TokenKind::STAR) {
                        ++cursor;
                    }
                }

                if (peek().kind != TokenKind::IDENTIFIER) {
                    throw ParserException(filename, peek().line, "OBJECTIVE_QUAD",
                                          "Expected variable identifier in quadratic expression");
                }

                std::string v1_name = peek().text;
                auto v1_id = get_or_create_var(v1_name);
                ++cursor;

                model::VarId v2_id = v1_id;
                if (peek().kind == TokenKind::CARET) {
                    ++cursor;
                    if (peek().kind != TokenKind::NUMBER || peek().number_value != 2.0) {
                        throw ParserException(filename, peek().line, "OBJECTIVE_QUAD",
                                              "Expected '^ 2' for quadratic term");
                    }
                    ++cursor;
                } else if (peek().kind == TokenKind::STAR) {
                    ++cursor;
                    if (peek().kind != TokenKind::IDENTIFIER) {
                        throw ParserException(
                            filename, peek().line, "OBJECTIVE_QUAD",
                            "Expected second variable after '*' in quadratic term");
                    }
                    v2_id = get_or_create_var(peek().text);
                    ++cursor;
                }

                quad_terms.push_back(
                    {.var1 = v1_id, .var2 = v2_id, .coeff = sign * term_sign * coeff});
            }

            if (peek().kind != TokenKind::RBRACKET) {
                throw ParserException(filename, peek().line, "OBJECTIVE_QUAD",
                                      "Unclosed quadratic term bracket ']'");
            }
            ++cursor;  // Consume ']'

            if (peek(0).kind == TokenKind::SLASH && peek(1).kind == TokenKind::NUMBER &&
                peek(1).number_value == 2.0) {
                quad_divisor = 2.0;
                cursor += 2;
            }

            for (const auto& qt : quad_terms) {
                canonical_model->add_objective_quadratic_term(qt.var1, qt.var2,
                                                              qt.coeff / quad_divisor);
            }
            continue;
        }

        // Linear term or constant offset
        double coeff = 1.0;
        bool has_number = false;
        if (peek().kind == TokenKind::NUMBER) {
            coeff = peek().number_value;
            has_number = true;
            ++cursor;
            if (peek().kind == TokenKind::STAR) {
                ++cursor;
            }
        }

        if (peek().kind == TokenKind::IDENTIFIER && !is_section_header()) {
            auto vid = get_or_create_var(peek().text);
            canonical_model->add_objective_linear_term(vid, sign * coeff);
            ++cursor;
        } else if (has_number) {
            // Constant objective offset
            canonical_model->set_objective_offset(canonical_model->objective_offset() +
                                                  (sign * coeff));
        } else {
            throw ParserException(filename, peek().line, "OBJECTIVE",
                                  "Unexpected token in objective: " + peek().text);
        }
    }

    // 2. Parse Subject To Section
    if (cursor < num_tokens) {
        if ((check_keyword("subject") && peek(1).kind == TokenKind::IDENTIFIER &&
             iequals(peek(1).text, "to")) ||
            (check_keyword("such") && peek(1).kind == TokenKind::IDENTIFIER &&
             iequals(peek(1).text, "that"))) {
            cursor += 2;
        } else if (check_keyword("s.t.") || check_keyword("st.") || check_keyword("st") ||
                   check_keyword("s.t")) {
            ++cursor;
        } else {
            throw ParserException(filename, peek().line, "CONSTRAINTS",
                                  "Expected 'Subject To' section header");
        }
    }

    std::size_t auto_constr_idx = 1;

    while (cursor < num_tokens && !is_section_header()) {
        std::string constr_name;
        // Check for constraint name: "c1:"
        if (peek(0).kind == TokenKind::IDENTIFIER && peek(1).kind == TokenKind::COLON) {
            constr_name = peek(0).text;
            cursor += 2;
        } else {
            constr_name = "c_" + std::to_string(auto_constr_idx++);
        }

        // Check if ranged constraint starting with number: "10 <= x1 + x2 <= 20"
        double left_bound = -model::INFINITY_BOUND;
        bool is_ranged_left = false;
        if (peek(0).kind == TokenKind::NUMBER &&
            (peek(1).kind == TokenKind::LESS_EQUAL || peek(1).kind == TokenKind::GREATER_EQUAL)) {
            left_bound = peek(0).number_value;
            if (peek(1).kind == TokenKind::GREATER_EQUAL) {
                throw ParserException(filename, peek().line, "CONSTRAINTS",
                                      "Invalid range constraint operator");
            }
            is_ranged_left = true;
            cursor += 2;
        }

        // Parse linear expression
        model::SparseLinearExpression expr;
        while (cursor < num_tokens && peek().kind != TokenKind::LESS_EQUAL &&
               peek().kind != TokenKind::GREATER_EQUAL && peek().kind != TokenKind::EQUAL &&
               !is_section_header()) {
            double sign = 1.0;
            if (peek().kind == TokenKind::PLUS) {
                ++cursor;
            } else if (peek().kind == TokenKind::MINUS) {
                sign = -1.0;
                ++cursor;
            }

            double coeff = 1.0;
            if (peek().kind == TokenKind::NUMBER) {
                coeff = peek().number_value;
                ++cursor;
                if (peek().kind == TokenKind::STAR) {
                    ++cursor;
                }
            }

            if (peek().kind != TokenKind::IDENTIFIER) {
                throw ParserException(filename, peek().line, "CONSTRAINTS",
                                      "Expected variable name in constraint: " + peek().text);
            }

            auto vid = get_or_create_var(peek().text);
            expr.add_term(vid, sign * coeff);
            ++cursor;
        }

        if (cursor >= num_tokens ||
            (peek().kind != TokenKind::LESS_EQUAL && peek().kind != TokenKind::GREATER_EQUAL &&
             peek().kind != TokenKind::EQUAL)) {
            throw ParserException(filename, peek().line, "CONSTRAINTS",
                                  "Constraint missing relational operator (<=, >=, =)");
        }

        model::ConstraintSense sense = model::ConstraintSense::LESS_EQUAL;
        if (peek().kind == TokenKind::LESS_EQUAL) {
            sense = model::ConstraintSense::LESS_EQUAL;
        } else if (peek().kind == TokenKind::GREATER_EQUAL) {
            sense = model::ConstraintSense::GREATER_EQUAL;
        } else if (peek().kind == TokenKind::EQUAL) {
            sense = model::ConstraintSense::EQUAL;
        }
        ++cursor;

        // Parse RHS
        double sign = 1.0;
        if (peek().kind == TokenKind::PLUS) {
            ++cursor;
        } else if (peek().kind == TokenKind::MINUS) {
            sign = -1.0;
            ++cursor;
        }

        if (peek().kind != TokenKind::NUMBER) {
            throw ParserException(filename, peek().line, "CONSTRAINTS",
                                  "Expected numerical RHS value in constraint");
        }
        double rhs_val = sign * peek().number_value;
        ++cursor;

        if (is_ranged_left) {
            // Dual constraints: expr >= left_bound and expr <= rhs_val
            canonical_model->add_constraint(constr_name + "_lb", expr,
                                            model::ConstraintSense::GREATER_EQUAL, left_bound);
            canonical_model->add_constraint(constr_name + "_ub", std::move(expr),
                                            model::ConstraintSense::LESS_EQUAL, rhs_val);
        } else if (cursor < num_tokens && peek().kind == TokenKind::LESS_EQUAL) {
            // Two-sided constraint: expr <= rhs_val <= upper_val (e.g. 10 <= expr <= 20)
            ++cursor;
            double ub_sign = 1.0;
            if (peek().kind == TokenKind::PLUS) {
                ++cursor;
            } else if (peek().kind == TokenKind::MINUS) {
                ub_sign = -1.0;
                ++cursor;
            }
            if (peek().kind != TokenKind::NUMBER) {
                throw ParserException(filename, peek().line, "CONSTRAINTS",
                                      "Expected upper bound number in ranged constraint");
            }
            double ub_val = ub_sign * peek().number_value;
            ++cursor;

            canonical_model->add_constraint(constr_name + "_lb", expr,
                                            model::ConstraintSense::GREATER_EQUAL, rhs_val);
            canonical_model->add_constraint(constr_name + "_ub", std::move(expr),
                                            model::ConstraintSense::LESS_EQUAL, ub_val);
        } else {
            canonical_model->add_constraint(constr_name, std::move(expr), sense, rhs_val);
        }
    }

    // 3. Optional Bounds Section
    if (cursor < num_tokens && (check_keyword("bounds") || check_keyword("bound"))) {
        ++cursor;
        while (cursor < num_tokens && !is_section_header()) {
            // Case A: number <= var [<= number]
            if (peek().kind == TokenKind::NUMBER || peek().kind == TokenKind::MINUS ||
                peek().kind == TokenKind::PLUS) {
                double sign = 1.0;
                if (peek().kind == TokenKind::PLUS) {
                    ++cursor;
                } else if (peek().kind == TokenKind::MINUS) {
                    sign = -1.0;
                    ++cursor;
                }

                double val1 = 0.0;
                if (check_keyword("infinity") || check_keyword("inf")) {
                    val1 = sign * model::INFINITY_BOUND;
                    ++cursor;
                } else if (peek().kind == TokenKind::NUMBER) {
                    val1 = sign * peek().number_value;
                    ++cursor;
                } else {
                    throw ParserException(filename, peek().line, "BOUNDS",
                                          "Expected number or infinity");
                }

                if (peek().kind != TokenKind::LESS_EQUAL) {
                    throw ParserException(filename, peek().line, "BOUNDS",
                                          "Expected '<=' in bound");
                }
                ++cursor;

                if (peek().kind != TokenKind::IDENTIFIER) {
                    throw ParserException(filename, peek().line, "BOUNDS",
                                          "Expected variable name in bound");
                }
                auto vid = get_or_create_var(peek().text);
                ++cursor;

                canonical_model->set_variable_lower_bound(vid, val1);

                if (peek().kind == TokenKind::LESS_EQUAL) {
                    ++cursor;
                    double ub_sign = 1.0;
                    if (peek().kind == TokenKind::PLUS) {
                        ++cursor;
                    } else if (peek().kind == TokenKind::MINUS) {
                        ub_sign = -1.0;
                        ++cursor;
                    }
                    double val2 = model::INFINITY_BOUND;
                    if (check_keyword("infinity") || check_keyword("inf")) {
                        val2 = ub_sign * model::INFINITY_BOUND;
                        ++cursor;
                    } else if (peek().kind == TokenKind::NUMBER) {
                        val2 = ub_sign * peek().number_value;
                        ++cursor;
                    }
                    canonical_model->set_variable_upper_bound(vid, val2);
                }
                continue;
            }

            // Case B: var ...
            if (peek().kind == TokenKind::IDENTIFIER) {
                std::string vname = peek().text;
                auto vid = get_or_create_var(vname);
                ++cursor;

                if (check_keyword("free")) {
                    canonical_model->set_variable_bounds(vid, -model::INFINITY_BOUND,
                                                         model::INFINITY_BOUND);
                    ++cursor;
                    continue;
                }

                if (peek().kind == TokenKind::LESS_EQUAL) {
                    ++cursor;
                    double sign = 1.0;
                    if (peek().kind == TokenKind::PLUS) {
                        ++cursor;
                    } else if (peek().kind == TokenKind::MINUS) {
                        sign = -1.0;
                        ++cursor;
                    }
                    double ub = model::INFINITY_BOUND;
                    if (check_keyword("infinity") || check_keyword("inf")) {
                        ub = sign * model::INFINITY_BOUND;
                        ++cursor;
                    } else if (peek().kind == TokenKind::NUMBER) {
                        ub = sign * peek().number_value;
                        ++cursor;
                    }
                    canonical_model->set_variable_upper_bound(vid, ub);
                } else if (peek().kind == TokenKind::GREATER_EQUAL) {
                    ++cursor;
                    double sign = 1.0;
                    if (peek().kind == TokenKind::PLUS) {
                        ++cursor;
                    } else if (peek().kind == TokenKind::MINUS) {
                        sign = -1.0;
                        ++cursor;
                    }
                    double lb = -model::INFINITY_BOUND;
                    if (check_keyword("infinity") || check_keyword("inf")) {
                        lb = sign * model::INFINITY_BOUND;
                        ++cursor;
                    } else if (peek().kind == TokenKind::NUMBER) {
                        lb = sign * peek().number_value;
                        ++cursor;
                    }
                    canonical_model->set_variable_lower_bound(vid, lb);
                } else if (peek().kind == TokenKind::EQUAL) {
                    ++cursor;
                    double sign = 1.0;
                    if (peek().kind == TokenKind::PLUS) {
                        ++cursor;
                    } else if (peek().kind == TokenKind::MINUS) {
                        sign = -1.0;
                        ++cursor;
                    }
                    if (peek().kind != TokenKind::NUMBER) {
                        throw ParserException(filename, peek().line, "BOUNDS",
                                              "Expected fixed value in '=' bound");
                    }
                    double fixed_val = sign * peek().number_value;
                    ++cursor;
                    canonical_model->set_variable_bounds(vid, fixed_val, fixed_val);
                } else {
                    throw ParserException(filename, peek().line, "BOUNDS",
                                          "Unexpected token in bound: " + peek().text);
                }
                continue;
            }

            throw ParserException(filename, peek().line, "BOUNDS",
                                  "Unexpected bound entry starting with: " + peek().text);
        }
    }

    // 4. Optional Generals / Integers Section
    if (cursor < num_tokens &&
        (check_keyword("generals") || check_keyword("general") || check_keyword("gen") ||
         check_keyword("integers") || check_keyword("integer") || check_keyword("int"))) {
        ++cursor;
        while (cursor < num_tokens && !is_section_header()) {
            if (peek().kind != TokenKind::IDENTIFIER) {
                throw ParserException(filename, peek().line, "GENERALS",
                                      "Expected variable name in Generals: " + peek().text);
            }
            auto vid = get_or_create_var(peek().text);
            canonical_model->set_variable_type(vid, model::VariableType::INTEGER);
            ++cursor;
        }
    }

    // 5. Optional Binaries Section
    if (cursor < num_tokens &&
        (check_keyword("binaries") || check_keyword("binary") || check_keyword("bin"))) {
        ++cursor;
        while (cursor < num_tokens && !is_section_header()) {
            if (peek().kind != TokenKind::IDENTIFIER) {
                throw ParserException(filename, peek().line, "BINARIES",
                                      "Expected variable name in Binaries: " + peek().text);
            }
            auto vid = get_or_create_var(peek().text);
            canonical_model->set_variable_type(vid, model::VariableType::BINARY);
            canonical_model->set_variable_bounds(vid, 0.0, 1.0);
            ++cursor;
        }
    }

    // 6. Optional End
    if (cursor < num_tokens && check_keyword("end")) {
        ++cursor;
    }

    canonical_model->canonicalize();

    if (options.validate_after_parse) {
        canonical_model->check_valid();
    }

    return canonical_model;
}

std::unique_ptr<model::CanonicalModel> LpParser::parse_file(const std::filesystem::path& file_path,
                                                            const ParserOptions& options) {
    if (!std::filesystem::exists(file_path)) {
        throw ParserException(file_path.string(), 0, "FILE",
                              "File not found: " + file_path.string());
    }

    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        throw ParserException(file_path.string(), 0, "FILE",
                              "Could not open file: " + file_path.string());
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return parse_string(content, file_path.string(), options);
}

}  // namespace kavach::io
