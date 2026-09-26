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

    JsonValue() noexcept : data_(std::in_place_type<std::monostate>) {}
    JsonValue(std::nullptr_t) noexcept : data_(std::in_place_type<std::monostate>) {}
    JsonValue(bool b) noexcept : data_(std::in_place_type<bool>, b) {}
    JsonValue(int num) noexcept : data_(std::in_place_type<double>, static_cast<double>(num)) {}
    JsonValue(int64_t num) noexcept : data_(std::in_place_type<double>, static_cast<double>(num)) {}
    JsonValue(uint64_t num) noexcept
        : data_(std::in_place_type<double>, static_cast<double>(num)) {}
    JsonValue(double num) noexcept : data_(std::in_place_type<double>, num) {}
    JsonValue(const char* s)
        : data_(std::in_place_type<std::string>, s ? std::string(s) : std::string{}) {}
    JsonValue(const std::string& s) : data_(std::in_place_type<std::string>, s) {}
    JsonValue(std::string&& s) noexcept : data_(std::in_place_type<std::string>, std::move(s)) {}
    JsonValue(std::string_view s) : data_(std::in_place_type<std::string>, std::string(s)) {}
    JsonValue(const Array& arr) : data_(std::in_place_type<Array>, arr) {}
    JsonValue(Array&& arr) noexcept : data_(std::in_place_type<Array>, std::move(arr)) {}
    JsonValue(const Object& obj) : data_(std::in_place_type<Object>, obj) {}
    JsonValue(Object&& obj) noexcept : data_(std::in_place_type<Object>, std::move(obj)) {}

    JsonValue(const JsonValue&) = default;
    JsonValue(JsonValue&&) noexcept = default;
    JsonValue& operator=(const JsonValue&) = default;
    JsonValue& operator=(JsonValue&&) noexcept = default;
    ~JsonValue() = default;

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
    std::variant<std::monostate, bool, double, std::string, Array, Object> data_{
        std::in_place_type<std::monostate>};
    static const JsonValue null_instance;
    static const std::string empty_string;
    static const Array empty_array;
    static const Object empty_object;
};

}  // namespace kavach::benchmark
