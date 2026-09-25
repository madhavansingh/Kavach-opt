#include "core/logger.hpp"

#include <chrono>
#include <iomanip>

namespace kavach::core {

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

void Logger::set_level(LogLevel level) {
    std::scoped_lock lock(mutex_);
    current_level_ = level;
}

LogLevel Logger::get_level() const noexcept {
    return current_level_;
}

void Logger::log(LogLevel level, const std::string& message) {
    std::scoped_lock lock(mutex_);
    if (level < current_level_) {
        return;
    }

    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::tm time_info{};
#ifdef _WIN32
    localtime_s(&time_info, &in_time_t);
#else
    localtime_r(&in_time_t, &time_info);
#endif

    std::cout << "[" << std::put_time(&time_info, "%Y-%m-%d %H:%M:%S") << "]" << " ["
              << to_string(level) << "] " << message << '\n';
}

void Logger::telemetry(const std::string& key, double value) {
    std::scoped_lock lock(mutex_);
    std::cout << "[TELEMETRY] " << key << "=" << value << '\n';
}

void Logger::telemetry(const std::string& key, const std::string& value) {
    std::scoped_lock lock(mutex_);
    std::cout << "[TELEMETRY] " << key << "=\"" << value << "\"\n";
}

}  // namespace kavach::core
