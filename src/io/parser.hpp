#pragma once

#include <filesystem>
#include <memory>

#include "model/model.hpp"

namespace kavach::io {

/**
 * @brief Architecture interface scaffolding for [B] Parser (MPS / native LP).
 * Full implementation scheduled for Phase 2 per Blueprint Section 4 & Section 32.
 */
class ModelParser {
public:
    virtual ~ModelParser() = default;

    [[nodiscard]] virtual std::unique_ptr<model::CanonicalModel> parse_file(
        const std::filesystem::path& file_path) = 0;
};

}  // namespace kavach::io
