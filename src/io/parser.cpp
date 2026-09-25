#include "io/parser.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <string>

namespace kavach::io {

namespace {

[[nodiscard]] std::string to_lower(std::string_view sv) {
    std::string result(sv);
    std::ranges::transform(result, result.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return result;
}

}  // namespace

ModelFormat detect_format(const std::filesystem::path& file_path) {
    std::string ext = to_lower(file_path.extension().string());
    if (ext == ".mps") {
        return ModelFormat::MPS;
    }
    if (ext == ".lp") {
        return ModelFormat::LP;
    }

    if (!std::filesystem::exists(file_path)) {
        return ModelFormat::UNKNOWN;
    }

    std::ifstream file(file_path);
    if (!file) {
        return ModelFormat::UNKNOWN;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Strip leading whitespace
        std::size_t start = 0;
        while (start < line.size() &&
               (std::isspace(static_cast<unsigned char>(line[start])) != 0)) {
            ++start;
        }
        if (start >= line.size()) {
            continue;
        }
        if (line[start] == '*' || line[start] == '\\' || line[start] == '$') {
            continue;  // Comment line
        }

        std::string first_line = to_lower(line.substr(start));
        if (first_line.starts_with("name") || first_line.starts_with("rows")) {
            return ModelFormat::MPS;
        }
        if (first_line.starts_with("min") || first_line.starts_with("max") ||
            first_line.starts_with("minimize") || first_line.starts_with("maximize")) {
            return ModelFormat::LP;
        }
        break;
    }

    return ModelFormat::UNKNOWN;
}

std::unique_ptr<model::CanonicalModel> parse_model(const std::filesystem::path& file_path,
                                                   const ParserOptions& options) {
    ModelFormat format = detect_format(file_path);
    switch (format) {
        case ModelFormat::MPS: {
            MpsParser parser;
            return parser.parse_file(file_path, options);
        }
        case ModelFormat::LP: {
            LpParser parser;
            return parser.parse_file(file_path, options);
        }
        case ModelFormat::UNKNOWN:
        default:
            throw ParserException(
                file_path.string(), 0, "DETECT",
                "Unable to determine model format for file: " + file_path.string());
    }
}

}  // namespace kavach::io
