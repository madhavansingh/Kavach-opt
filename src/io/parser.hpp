#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

#include "core/error.hpp"
#include "model/model.hpp"

namespace kavach::io {

/**
 * @brief Supported optimization model file formats.
 */
enum class ModelFormat : std::uint8_t {
    MPS,     ///< Mathematical Programming System (.mps)
    LP,      ///< Native LP format (.lp)
    UNKNOWN  ///< Unrecognized format
};

[[nodiscard]] constexpr std::string_view to_string(ModelFormat format) noexcept {
    switch (format) {
        case ModelFormat::MPS:
            return "MPS";
        case ModelFormat::LP:
            return "LP";
        case ModelFormat::UNKNOWN:
            return "UNKNOWN";
    }
    return "UNKNOWN";
}

/**
 * @brief Typed parser exception containing contextual source location.
 */
class ParserException : public core::InputException {
public:
    ParserException(const std::string& filename, std::size_t line, const std::string& section,
                    const std::string& message)
        : core::InputException(format_message(filename, line, section, message)),
          filename_(filename),
          line_(line),
          section_(section),
          raw_message_(message) {}

    [[nodiscard]] const std::string& filename() const noexcept { return filename_; }
    [[nodiscard]] std::size_t line() const noexcept { return line_; }
    [[nodiscard]] const std::string& section() const noexcept { return section_; }
    [[nodiscard]] const std::string& raw_message() const noexcept { return raw_message_; }

private:
    static std::string format_message(const std::string& filename, std::size_t line,
                                      const std::string& section, const std::string& message) {
        std::string result = filename.empty() ? "<input>" : filename;
        if (line > 0) {
            result += ":" + std::to_string(line);
        }
        if (!section.empty()) {
            result += " [" + section + "]";
        }
        result += ": " + message;
        return result;
    }

    std::string filename_;
    std::size_t line_{0};
    std::string section_;
    std::string raw_message_;
};

/**
 * @brief Parser configuration options.
 */
struct ParserOptions {
    bool strict_fixed_mps{false};  ///< Force strict column-position MPS checking
    std::string objective_name;  ///< Specific objective row name if model contains multiple N rows
    bool validate_after_parse{true};  ///< Automatically run CanonicalModel::check_valid()
};

/**
 * @brief Abstract base class for optimization model parsers.
 */
class ModelParser {
public:
    virtual ~ModelParser() = default;

    [[nodiscard]] virtual std::unique_ptr<model::CanonicalModel> parse_file(
        const std::filesystem::path& file_path, const ParserOptions& options = {}) = 0;

    [[nodiscard]] virtual std::unique_ptr<model::CanonicalModel> parse_string(
        std::string_view content, std::string_view source_name = "<buffer>",
        const ParserOptions& options = {}) = 0;
};

/**
 * @brief MPS (Mathematical Programming System) Parser.
 * Supports fixed-column and free-format MPS, integer markers, bounds, and ranges.
 */
class MpsParser : public ModelParser {
public:
    [[nodiscard]] std::unique_ptr<model::CanonicalModel> parse_file(
        const std::filesystem::path& file_path, const ParserOptions& options = {}) override;

    [[nodiscard]] std::unique_ptr<model::CanonicalModel> parse_string(
        std::string_view content, std::string_view source_name = "<buffer>",
        const ParserOptions& options = {}) override;
};

/**
 * @brief Native LP Format Parser.
 * Supports Minimize/Maximize, Subject To, Bounds, Generals, Binaries, and End.
 */
class LpParser : public ModelParser {
public:
    [[nodiscard]] std::unique_ptr<model::CanonicalModel> parse_file(
        const std::filesystem::path& file_path, const ParserOptions& options = {}) override;

    [[nodiscard]] std::unique_ptr<model::CanonicalModel> parse_string(
        std::string_view content, std::string_view source_name = "<buffer>",
        const ParserOptions& options = {}) override;
};

/**
 * @brief Automatically detect model format based on file extension or header content.
 */
[[nodiscard]] ModelFormat detect_format(const std::filesystem::path& file_path);

/**
 * @brief Convenience function to parse a model file with automatic format detection.
 */
[[nodiscard]] std::unique_ptr<model::CanonicalModel> parse_model(
    const std::filesystem::path& file_path, const ParserOptions& options = {});

}  // namespace kavach::io
