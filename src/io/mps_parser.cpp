#include <algorithm>
#include <cctype>
#include <charconv>
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

enum class MpsSection : std::uint8_t {
    NONE,
    NAME,
    OBJSENSE,
    OBJNAME,
    ROWS,
    COLUMNS,
    RHS,
    RANGES,
    BOUNDS,
    ENDATA
};

[[nodiscard]] std::string_view to_string(MpsSection section) noexcept {
    switch (section) {
        case MpsSection::NONE:
            return "HEADER";
        case MpsSection::NAME:
            return "NAME";
        case MpsSection::OBJSENSE:
            return "OBJSENSE";
        case MpsSection::OBJNAME:
            return "OBJNAME";
        case MpsSection::ROWS:
            return "ROWS";
        case MpsSection::COLUMNS:
            return "COLUMNS";
        case MpsSection::RHS:
            return "RHS";
        case MpsSection::RANGES:
            return "RANGES";
        case MpsSection::BOUNDS:
            return "BOUNDS";
        case MpsSection::ENDATA:
            return "ENDATA";
    }
    return "UNKNOWN";
}

[[nodiscard]] std::string trim(std::string_view str) {
    while (!str.empty() && (std::isspace(static_cast<unsigned char>(str.front())) != 0)) {
        str.remove_prefix(1);
    }
    while (!str.empty() && (std::isspace(static_cast<unsigned char>(str.back())) != 0)) {
        str.remove_suffix(1);
    }
    return std::string(str);
}

[[nodiscard]] std::vector<std::string> split_tokens(std::string_view line) {
    std::vector<std::string> tokens;
    std::size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && (std::isspace(static_cast<unsigned char>(line[i])) != 0)) {
            ++i;
        }
        if (i >= line.size()) {
            break;
        }
        std::size_t start = i;
        while (i < line.size() && (std::isspace(static_cast<unsigned char>(line[i])) == 0)) {
            ++i;
        }
        tokens.emplace_back(line.substr(start, i - start));
    }
    return tokens;
}

[[nodiscard]] double parse_number(std::string_view raw_token, const std::string& filename,
                                  std::size_t line_num, std::string_view section) {
    std::string token = trim(raw_token);
    if (token.empty()) {
        throw ParserException(filename, line_num, std::string(section),
                              "Empty numerical coefficient");
    }

    // Normalize Fortran D/d exponents to E/e: e.g. 1.23D+04 -> 1.23E+04
    for (char& c : token) {
        if (c == 'D' || c == 'd') {
            c = 'E';
        }
    }

    // Handle Fortran compact exponent without E: e.g. 1.234+04 or 1.234-04
    for (std::size_t i = 1; i + 1 < token.size(); ++i) {
        char prev = token[i - 1];
        char curr = token[i];
        if ((curr == '+' || curr == '-') && std::isdigit(static_cast<unsigned char>(prev)) != 0 &&
            prev != 'e' && prev != 'E') {
            token.insert(i, 1, 'E');
            break;
        }
    }

    char* end_ptr = nullptr;
    double val = std::strtod(token.c_str(), &end_ptr);
    if (end_ptr == token.c_str() || (end_ptr != nullptr && *end_ptr != '\0')) {
        throw ParserException(filename, line_num, std::string(section),
                              "Invalid numerical value: '" + std::string(raw_token) + "'");
    }

    if (std::isnan(val) || std::isinf(val)) {
        throw ParserException(
            filename, line_num, std::string(section),
            "Numerical coefficient cannot be NaN or infinite: '" + std::string(raw_token) + "'");
    }

    return val;
}

enum class MpsRowType : std::uint8_t { OBJECTIVE, FREE, LESS_EQUAL, GREATER_EQUAL, EQUAL };

struct ParsedRow {
    std::string name;
    MpsRowType type{MpsRowType::FREE};
    model::ConstraintSense sense{model::ConstraintSense::LESS_EQUAL};
    bool is_objective{false};
    double rhs{0.0};
    bool has_rhs{false};
    double range{0.0};
    bool has_range{false};
    std::vector<model::LinearTerm> terms;
};

struct ParsedCol {
    std::string name;
    model::VarId id{model::INVALID_VAR_ID};
    model::VariableType type{model::VariableType::CONTINUOUS};
    double lower_bound{0.0};
    double upper_bound{model::INFINITY_BOUND};
    bool lower_bound_set{false};
    bool upper_bound_set{false};
};

}  // namespace

std::unique_ptr<model::CanonicalModel> MpsParser::parse_string(std::string_view content,
                                                               std::string_view source_name,
                                                               const ParserOptions& options) {
    const std::string filename(source_name);
    std::istringstream stream{std::string(content)};
    std::string line;
    std::size_t line_num = 0;

    MpsSection current_section = MpsSection::NONE;
    std::string model_name;
    std::string chosen_objective_name = options.objective_name;
    model::ObjectiveSense obj_sense = model::ObjectiveSense::MINIMIZE;

    std::vector<ParsedRow> rows;
    std::unordered_map<std::string, std::size_t> row_name_to_idx;

    std::vector<ParsedCol> cols;
    std::unordered_map<std::string, std::size_t> col_name_to_idx;

    bool in_integer_block = false;
    std::string chosen_rhs_name;
    std::string chosen_range_name;
    std::string chosen_bound_name;
    bool found_endata = false;

    while (std::getline(stream, line)) {
        ++line_num;

        // Strip trailing \r and whitespace
        while (!line.empty() &&
               (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }

        if (line.empty() || line[0] == '*' || line[0] == '$' || line[0] == '\\') {
            continue;  // Comment or blank line
        }

        // Check if section header (first character is non-whitespace)
        if (line[0] != ' ' && line[0] != '\t') {
            auto header_tokens = split_tokens(line);
            if (header_tokens.empty()) {
                continue;
            }
            const std::string& keyword = header_tokens[0];

            if (keyword == "NAME") {
                current_section = MpsSection::NAME;
                if (header_tokens.size() > 1) {
                    model_name = header_tokens[1];
                }
                continue;
            }
            if (keyword == "OBJSENSE") {
                current_section = MpsSection::OBJSENSE;
                if (header_tokens.size() > 1) {
                    std::string sense_tok = header_tokens[1];
                    std::ranges::transform(sense_tok, sense_tok.begin(), [](unsigned char c) {
                        return static_cast<char>(std::toupper(c));
                    });
                    if (sense_tok == "MAX" || sense_tok == "MAXIMIZE") {
                        obj_sense = model::ObjectiveSense::MAXIMIZE;
                    }
                }
                continue;
            }
            if (keyword == "OBJNAME") {
                current_section = MpsSection::OBJNAME;
                if (header_tokens.size() > 1) {
                    chosen_objective_name = header_tokens[1];
                }
                continue;
            }
            if (keyword == "ROWS") {
                current_section = MpsSection::ROWS;
                continue;
            }
            if (keyword == "COLUMNS") {
                current_section = MpsSection::COLUMNS;
                continue;
            }
            if (keyword == "RHS") {
                current_section = MpsSection::RHS;
                continue;
            }
            if (keyword == "RANGES") {
                current_section = MpsSection::RANGES;
                continue;
            }
            if (keyword == "BOUNDS") {
                current_section = MpsSection::BOUNDS;
                continue;
            }
            if (keyword == "ENDATA") {
                current_section = MpsSection::ENDATA;
                found_endata = true;
                break;
            }

            throw ParserException(filename, line_num, std::string(to_string(current_section)),
                                  "Unknown MPS section header: '" + keyword + "'");
        }

        // Data lines within sections
        switch (current_section) {
            case MpsSection::OBJSENSE: {
                auto tokens = split_tokens(line);
                if (!tokens.empty()) {
                    std::string sense_tok = tokens[0];
                    std::ranges::transform(sense_tok, sense_tok.begin(), [](unsigned char c) {
                        return static_cast<char>(std::toupper(c));
                    });
                    if (sense_tok == "MAX" || sense_tok == "MAXIMIZE") {
                        obj_sense = model::ObjectiveSense::MAXIMIZE;
                    } else if (sense_tok == "MIN" || sense_tok == "MINIMIZE") {
                        obj_sense = model::ObjectiveSense::MINIMIZE;
                    } else {
                        throw ParserException(filename, line_num, "OBJSENSE",
                                              "Invalid OBJSENSE specification: " + tokens[0]);
                    }
                }
                break;
            }
            case MpsSection::OBJNAME: {
                auto tokens = split_tokens(line);
                if (!tokens.empty()) {
                    chosen_objective_name = tokens[0];
                }
                break;
            }
            case MpsSection::ROWS: {
                auto tokens = split_tokens(line);
                if (tokens.size() < 2) {
                    throw ParserException(filename, line_num, "ROWS",
                                          "Malformed ROWS line (requires type and row name)");
                }
                const std::string& type_str = tokens[0];
                const std::string& rname = tokens[1];

                if (row_name_to_idx.contains(rname)) {
                    throw ParserException(filename, line_num, "ROWS",
                                          "Duplicate row name: '" + rname + "'");
                }

                ParsedRow row;
                row.name = rname;

                if (type_str == "N") {
                    if (chosen_objective_name.empty()) {
                        chosen_objective_name = rname;
                        row.type = MpsRowType::OBJECTIVE;
                        row.is_objective = true;
                    } else if (rname == chosen_objective_name) {
                        row.type = MpsRowType::OBJECTIVE;
                        row.is_objective = true;
                    } else {
                        row.type = MpsRowType::FREE;
                        row.is_objective = false;  // Free unconstrained row
                    }
                } else if (type_str == "L") {
                    row.type = MpsRowType::LESS_EQUAL;
                    row.sense = model::ConstraintSense::LESS_EQUAL;
                } else if (type_str == "G") {
                    row.type = MpsRowType::GREATER_EQUAL;
                    row.sense = model::ConstraintSense::GREATER_EQUAL;
                } else if (type_str == "E") {
                    row.type = MpsRowType::EQUAL;
                    row.sense = model::ConstraintSense::EQUAL;
                } else {
                    throw ParserException(filename, line_num, "ROWS",
                                          "Unknown row sense indicator: '" + type_str + "'");
                }

                row_name_to_idx[rname] = rows.size();
                rows.push_back(std::move(row));
                break;
            }
            case MpsSection::COLUMNS: {
                auto tokens = split_tokens(line);
                if (tokens.empty()) {
                    break;
                }

                // Check for integer markers: e.g. MARK0001 'MARKER' 'INTORG' or 'INTEND'
                bool has_marker_keyword = false;
                for (const auto& tok : tokens) {
                    if (tok.find("MARKER") != std::string::npos) {
                        has_marker_keyword = true;
                        break;
                    }
                }
                if (has_marker_keyword) {
                    bool has_intorg = false;
                    bool has_intend = false;
                    for (const auto& tok : tokens) {
                        if (tok.find("INTORG") != std::string::npos) {
                            has_intorg = true;
                        } else if (tok.find("INTEND") != std::string::npos) {
                            has_intend = true;
                        }
                    }
                    if (has_intorg) {
                        in_integer_block = true;
                        break;
                    }
                    if (has_intend) {
                        in_integer_block = false;
                        break;
                    }
                }

                // Normal column entry: ColName RowName1 Val1 [RowName2 Val2]
                if (tokens.size() < 3) {
                    throw ParserException(filename, line_num, "COLUMNS",
                                          "Malformed COLUMNS line: expected at least 3 tokens");
                }

                const std::string& cname = tokens[0];
                auto col_it = col_name_to_idx.find(cname);
                if (col_it == col_name_to_idx.end()) {
                    ParsedCol col;
                    col.name = cname;
                    col.id = static_cast<model::VarId>(cols.size());
                    col.type = in_integer_block ? model::VariableType::INTEGER
                                                : model::VariableType::CONTINUOUS;
                    col.lower_bound = 0.0;
                    col.upper_bound = model::INFINITY_BOUND;
                    col_name_to_idx[cname] = cols.size();
                    cols.push_back(std::move(col));
                    col_it = col_name_to_idx.find(cname);
                }

                const auto var_id = cols[col_it->second].id;

                for (std::size_t idx = 1; idx + 1 < tokens.size(); idx += 2) {
                    const std::string& rname = tokens[idx];
                    double coeff = parse_number(tokens[idx + 1], filename, line_num, "COLUMNS");

                    auto row_it = row_name_to_idx.find(rname);
                    if (row_it == row_name_to_idx.end()) {
                        throw ParserException(
                            filename, line_num, "COLUMNS",
                            "Row referenced in COLUMNS not declared: '" + rname + "'");
                    }

                    rows[row_it->second].terms.push_back({.var_id = var_id, .coefficient = coeff});
                }
                break;
            }
            case MpsSection::RHS: {
                auto tokens = split_tokens(line);
                if (tokens.empty()) {
                    break;
                }

                std::size_t start_idx = 0;
                if (tokens.size() == 3 || tokens.size() == 5) {
                    // Vector name present: [rhs_name, row1, val1, ...]
                    const std::string& rhs_vec = tokens[0];
                    if (chosen_rhs_name.empty()) {
                        chosen_rhs_name = rhs_vec;
                    }
                    if (rhs_vec != chosen_rhs_name) {
                        break;  // Ignore secondary RHS vectors
                    }
                    start_idx = 1;
                } else if (tokens.size() == 2 || tokens.size() == 4) {
                    // Vector name omitted: [row1, val1, ...]
                    start_idx = 0;
                } else {
                    throw ParserException(
                        filename, line_num, "RHS",
                        "Malformed RHS line: invalid token count " + std::to_string(tokens.size()));
                }

                for (std::size_t idx = start_idx; idx + 1 < tokens.size(); idx += 2) {
                    const std::string& rname = tokens[idx];
                    double val = parse_number(tokens[idx + 1], filename, line_num, "RHS");

                    auto row_it = row_name_to_idx.find(rname);
                    if (row_it == row_name_to_idx.end()) {
                        throw ParserException(
                            filename, line_num, "RHS",
                            "Row referenced in RHS not declared: '" + rname + "'");
                    }

                    rows[row_it->second].rhs = val;
                    rows[row_it->second].has_rhs = true;
                }
                break;
            }
            case MpsSection::RANGES: {
                auto tokens = split_tokens(line);
                if (tokens.empty()) {
                    break;
                }

                std::size_t start_idx = 0;
                if (tokens.size() == 3 || tokens.size() == 5) {
                    const std::string& range_vec = tokens[0];
                    if (chosen_range_name.empty()) {
                        chosen_range_name = range_vec;
                    }
                    if (range_vec != chosen_range_name) {
                        break;  // Ignore secondary RANGES vectors
                    }
                    start_idx = 1;
                } else if (tokens.size() == 2 || tokens.size() == 4) {
                    start_idx = 0;
                } else {
                    throw ParserException(filename, line_num, "RANGES",
                                          "Malformed RANGES line: invalid token count " +
                                              std::to_string(tokens.size()));
                }

                for (std::size_t idx = start_idx; idx + 1 < tokens.size(); idx += 2) {
                    const std::string& rname = tokens[idx];
                    double val = parse_number(tokens[idx + 1], filename, line_num, "RANGES");

                    auto row_it = row_name_to_idx.find(rname);
                    if (row_it == row_name_to_idx.end()) {
                        throw ParserException(
                            filename, line_num, "RANGES",
                            "Row referenced in RANGES not declared: '" + rname + "'");
                    }

                    rows[row_it->second].range = val;
                    rows[row_it->second].has_range = true;
                }
                break;
            }
            case MpsSection::BOUNDS: {
                auto tokens = split_tokens(line);
                if (tokens.empty()) {
                    break;
                }

                const std::string& btype = tokens[0];
                std::string bname;
                std::string cname;
                std::string val_str;

                if (tokens.size() == 4) {
                    bname = tokens[1];
                    cname = tokens[2];
                    val_str = tokens[3];
                } else if (tokens.size() == 3) {
                    // Check if tokens[2] is a number
                    char* end_ptr = nullptr;
                    (void)std::strtod(tokens[2].c_str(), &end_ptr);
                    if (end_ptr != tokens[2].c_str() && *end_ptr == '\0') {
                        // tokens[2] is a number -> [type, col_name, value]
                        cname = tokens[1];
                        val_str = tokens[2];
                    } else {
                        // tokens[2] is not a number -> [type, bound_name, col_name] (for FR, BV,
                        // etc.)
                        bname = tokens[1];
                        cname = tokens[2];
                    }
                } else if (tokens.size() == 2) {
                    cname = tokens[1];
                } else {
                    throw ParserException(filename, line_num, "BOUNDS",
                                          "Malformed BOUNDS line: invalid token count " +
                                              std::to_string(tokens.size()));
                }

                if (!bname.empty()) {
                    if (chosen_bound_name.empty()) {
                        chosen_bound_name = bname;
                    }
                    if (bname != chosen_bound_name) {
                        break;  // Ignore secondary BOUNDS vectors
                    }
                }

                auto col_it = col_name_to_idx.find(cname);
                if (col_it == col_name_to_idx.end()) {
                    throw ParserException(
                        filename, line_num, "BOUNDS",
                        "Column referenced in BOUNDS not declared: '" + cname + "'");
                }
                auto& col = cols[col_it->second];

                if (btype == "LO") {
                    double val = parse_number(val_str, filename, line_num, "BOUNDS");
                    col.lower_bound = val;
                    col.lower_bound_set = true;
                } else if (btype == "UP") {
                    double val = parse_number(val_str, filename, line_num, "BOUNDS");
                    col.upper_bound = val;
                    col.upper_bound_set = true;
                    if (val <= 0.0 && !col.lower_bound_set) {
                        col.lower_bound = -model::INFINITY_BOUND;
                    }
                } else if (btype == "FX") {
                    double val = parse_number(val_str, filename, line_num, "BOUNDS");
                    col.lower_bound = val;
                    col.upper_bound = val;
                    col.lower_bound_set = true;
                    col.upper_bound_set = true;
                } else if (btype == "FR") {
                    col.lower_bound = -model::INFINITY_BOUND;
                    col.upper_bound = model::INFINITY_BOUND;
                    col.lower_bound_set = true;
                    col.upper_bound_set = true;
                } else if (btype == "MI") {
                    col.lower_bound = -model::INFINITY_BOUND;
                    col.lower_bound_set = true;
                } else if (btype == "PL") {
                    col.upper_bound = model::INFINITY_BOUND;
                    col.upper_bound_set = true;
                } else if (btype == "BV") {
                    col.type = model::VariableType::BINARY;
                    col.lower_bound = 0.0;
                    col.upper_bound = 1.0;
                    col.lower_bound_set = true;
                    col.upper_bound_set = true;
                } else if (btype == "LI") {
                    double val = parse_number(val_str, filename, line_num, "BOUNDS");
                    col.type = model::VariableType::INTEGER;
                    col.lower_bound = val;
                    col.lower_bound_set = true;
                } else if (btype == "UI") {
                    double val = parse_number(val_str, filename, line_num, "BOUNDS");
                    col.type = model::VariableType::INTEGER;
                    col.upper_bound = val;
                    col.upper_bound_set = true;
                    if (val <= 0.0 && !col.lower_bound_set) {
                        col.lower_bound = -model::INFINITY_BOUND;
                    }
                } else {
                    throw ParserException(filename, line_num, "BOUNDS",
                                          "Unknown bound type: '" + btype + "'");
                }
                break;
            }
            default:
                break;
        }
    }

    if (!found_endata && current_section == MpsSection::NONE) {
        throw ParserException(filename, line_num, "MPS",
                              "Empty or invalid MPS file (missing sections)");
    }

    auto canonical_model = std::make_unique<model::CanonicalModel>(model_name);

    // 1. Add all variables in order
    for (const auto& col : cols) {
        canonical_model->add_variable(col.name, col.type, col.lower_bound, col.upper_bound);
    }

    // 2. Locate objective row and configure objective
    canonical_model->set_objective_sense(obj_sense);

    for (const auto& row : rows) {
        if (row.is_objective) {
            for (const auto& term : row.terms) {
                canonical_model->add_objective_linear_term(term.var_id, term.coefficient);
            }
            if (row.has_rhs) {
                // In standard MPS, c^T x - b_0 minimizes c^T x with constant offset -b_0
                canonical_model->set_objective_offset(-row.rhs);
            }
            break;
        }
    }

    // 3. Add constraint rows (expanding RANGES into dual bounding constraints)
    for (auto& row : rows) {
        if (row.type == MpsRowType::OBJECTIVE || row.type == MpsRowType::FREE) {
            continue;  // Objective stored in model.objective(); Free rows have no bounds
        }

        if (!row.has_range) {
            canonical_model->add_constraint(
                row.name, model::SparseLinearExpression(std::move(row.terms)), row.sense, row.rhs);
        } else {
            // RANGES mathematically transformed into dual bounding constraints per blueprint [B]
            double r = row.range;
            if (row.sense == model::ConstraintSense::GREATER_EQUAL) {
                // b <= a^T x <= b + |r|
                double r_abs = std::abs(r);
                canonical_model->add_constraint(row.name, model::SparseLinearExpression(row.terms),
                                                model::ConstraintSense::GREATER_EQUAL, row.rhs);
                canonical_model->add_constraint(
                    row.name + "_range_ub", model::SparseLinearExpression(std::move(row.terms)),
                    model::ConstraintSense::LESS_EQUAL, row.rhs + r_abs);
            } else if (row.sense == model::ConstraintSense::LESS_EQUAL) {
                // b - |r| <= a^T x <= b
                double r_abs = std::abs(r);
                canonical_model->add_constraint(row.name, model::SparseLinearExpression(row.terms),
                                                model::ConstraintSense::LESS_EQUAL, row.rhs);
                canonical_model->add_constraint(
                    row.name + "_range_lb", model::SparseLinearExpression(std::move(row.terms)),
                    model::ConstraintSense::GREATER_EQUAL, row.rhs - r_abs);
            } else if (row.sense == model::ConstraintSense::EQUAL) {
                if (r > 0.0) {
                    // b <= a^T x <= b + r
                    canonical_model->add_constraint(row.name + "_range_lb",
                                                    model::SparseLinearExpression(row.terms),
                                                    model::ConstraintSense::GREATER_EQUAL, row.rhs);
                    canonical_model->add_constraint(
                        row.name + "_range_ub", model::SparseLinearExpression(std::move(row.terms)),
                        model::ConstraintSense::LESS_EQUAL, row.rhs + r);
                } else {
                    // b + r <= a^T x <= b (where r is negative)
                    canonical_model->add_constraint(
                        row.name + "_range_lb", model::SparseLinearExpression(row.terms),
                        model::ConstraintSense::GREATER_EQUAL, row.rhs + r);
                    canonical_model->add_constraint(
                        row.name + "_range_ub", model::SparseLinearExpression(std::move(row.terms)),
                        model::ConstraintSense::LESS_EQUAL, row.rhs);
                }
            }
        }
    }

    canonical_model->canonicalize();

    if (options.validate_after_parse) {
        canonical_model->check_valid();
    }

    return canonical_model;
}

std::unique_ptr<model::CanonicalModel> MpsParser::parse_file(const std::filesystem::path& file_path,
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
