#pragma once

#include <cstdint>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>

namespace kavach::core {

enum class LogLevel : std::uint8_t { DEBUG, INFO, WARNING, ERROR, TELEMETRY };

[[nodiscard]] constexpr std::string_view to_string(LogLevel level) noexcept {
    switch (level) {
        case LogLevel::DEBUG:
            return "DEBUG";
        case LogLevel::INFO:
            return "INFO";
        case LogLevel::WARNING:
            return "WARN";
        case LogLevel::ERROR:
            return "ERROR";
        case LogLevel::TELEMETRY:
            return "TELEM";
    }
    return "UNKNOWN";
}

class Logger {
public:
    static Logger& instance();

    void set_level(LogLevel level);
    [[nodiscard]] LogLevel get_level() const noexcept;

    void log(LogLevel level, const std::string& message);
    void telemetry(const std::string& key, double value);
    void telemetry(const std::string& key, const std::string& value);

private:
    Logger() = default;
    LogLevel current_level_{LogLevel::INFO};
    std::mutex mutex_;
};

#define KAVACH_LOG_DEBUG(msg)                                                                      \
    kavach::core::Logger::instance().log(kavach::core::LogLevel::DEBUG, msg)
#define KAVACH_LOG_INFO(msg) kavach::core::Logger::instance().log(kavach::core::LogLevel::INFO, msg)
#define KAVACH_LOG_WARN(msg)                                                                       \
    kavach::core::Logger::instance().log(kavach::core::LogLevel::WARNING, msg)
#define KAVACH_LOG_ERROR(msg)                                                                      \
    kavach::core::Logger::instance().log(kavach::core::LogLevel::ERROR, msg)

}  // namespace kavach::core
