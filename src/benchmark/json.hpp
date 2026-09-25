#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace kavach::benchmark {

enum class JsonType { Null, Boolean, Number, String, Array, Object };

class JsonValue {
public:
    using Array = std::vector<JsonValue>;
    using Object = std::map<std::string, JsonValue>;

    JsonValue() : data_(std::monostate{}) {}
    JsonValue(std::nullptr_t) : data_(std::monostate{}) {}
    JsonValue(bool b) : data_(b) {}
    JsonValue(int num) : data_(static_cast<double>(num)) {}
    JsonValue(int64_t num) : data_(static_cast<double>(num)) {}
    JsonValue(uint64_t num) : data_(static_cast<double>(num)) {}
    JsonValue(double num) : data_(num) {}
    JsonValue(const char* s) : data_(std::string(s)) {}
    JsonValue(std::string s) : data_(std::move(s)) {}
    JsonValue(std::string_view s) : data_(std::string(s)) {}
    JsonValue(Array arr) : data_(std::move(arr)) {}
    JsonValue(Object obj) : data_(std::move(obj)) {}

    [[nodiscard]] JsonType type() const noexcept;
    [[nodiscard]] bool is_null() const noexcept { return type() == JsonType::Null; }
    [[nodiscard]] bool is_bool() const noexcept { return type() == JsonType::Boolean; }
    [[nodiscard]] bool is_number() const noexcept { return type() == JsonType::Number; }
    [[nodiscard]] bool is_string() const noexcept { return type() == JsonType::String; }
    [[nodiscard]] bool is_array() const noexcept { return type() == JsonType::Array; }
    [[nodiscard]] bool is_object() const noexcept { return type() == JsonType::Object; }

    [[nodiscard]] bool as_bool(bool default_val = false) const;
    [[nodiscard]] double as_double(double default_val = 0.0) const;
    [[nodiscard]] int64_t as_int64(int64_t default_val = 0) const;
    [[nodiscard]] const std::string& as_string(const std::string& default_val = "") const;
    [[nodiscard]] const Array& as_array() const;
    [[nodiscard]] Array& as_array();
    [[nodiscard]] const Object& as_object() const;
    [[nodiscard]] Object& as_object();

    [[nodiscard]] bool contains(const std::string& key) const;
    [[nodiscard]] const JsonValue& operator[](const std::string& key) const;
    [[nodiscard]] JsonValue& operator[](const std::string& key);
    [[nodiscard]] const JsonValue& operator[](std::size_t idx) const;
    [[nodiscard]] JsonValue& operator[](std::size_t idx);

    [[nodiscard]] std::string dump(int indent = 2) const;
    static JsonValue parse(std::string_view json_str);

private:
    std::variant<std::monostate, bool, double, std::string, Array, Object> data_;
    static const JsonValue null_instance;
    static const std::string empty_string;
    static const Array empty_array;
    static const Object empty_object;
};

}  // namespace kavach::benchmark
