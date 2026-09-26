#include "benchmark/json.hpp"

#include <cctype>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace kavach::benchmark {

const JsonValue JsonValue::null_instance{};
const std::string JsonValue::empty_string{};
const JsonValue::Array JsonValue::empty_array{};
const JsonValue::Object JsonValue::empty_object{};

JsonType JsonValue::type() const noexcept {
    switch (data_.index()) {
        case 0:
            return JsonType::Null;
        case 1:
            return JsonType::Boolean;
        case 2:
            return JsonType::Number;
        case 3:
            return JsonType::String;
        case 4:
            return JsonType::Array;
        case 5:
            return JsonType::Object;
        default:
            return JsonType::Null;
    }
}

bool JsonValue::as_bool(bool default_val) const {
    if (auto* b = std::get_if<bool>(&data_)) {
        return *b;
    }
    return default_val;
}

double JsonValue::as_double(double default_val) const {
    if (auto* d = std::get_if<double>(&data_)) {
        return *d;
    }
    return default_val;
}

int64_t JsonValue::as_int64(int64_t default_val) const {
    if (auto* d = std::get_if<double>(&data_)) {
        return static_cast<int64_t>(*d);
    }
    return default_val;
}

const std::string& JsonValue::as_string(const std::string& default_val) const {
    if (auto* s = std::get_if<std::string>(&data_)) {
        return *s;
    }
    return default_val;
}

const JsonValue::Array& JsonValue::as_array() const {
    if (auto* arr = std::get_if<Array>(&data_)) {
        return *arr;
    }
    return empty_array;
}

JsonValue::Array& JsonValue::as_array() {
    if (!std::holds_alternative<Array>(data_)) {
        data_.emplace<Array>();
    }
    return std::get<Array>(data_);
}

const JsonValue::Object& JsonValue::as_object() const {
    if (auto* obj = std::get_if<Object>(&data_)) {
        return *obj;
    }
    return empty_object;
}

JsonValue::Object& JsonValue::as_object() {
    if (!std::holds_alternative<Object>(data_)) {
        data_.emplace<Object>();
    }
    return std::get<Object>(data_);
}

bool JsonValue::contains(const std::string& key) const {
    if (auto* obj = std::get_if<Object>(&data_)) {
        return obj->find(key) != obj->end();
    }
    return false;
}

const JsonValue& JsonValue::operator[](const std::string& key) const {
    if (auto* obj = std::get_if<Object>(&data_)) {
        auto it = obj->find(key);
        if (it != obj->end()) {
            return it->second;
        }
    }
    return null_instance;
}

JsonValue& JsonValue::operator[](const std::string& key) {
    if (!std::holds_alternative<Object>(data_)) {
        data_.emplace<Object>();
    }
    return std::get<Object>(data_)[key];
}

const JsonValue& JsonValue::operator[](std::size_t idx) const {
    if (auto* arr = std::get_if<Array>(&data_)) {
        if (idx < arr->size()) {
            return (*arr)[idx];
        }
    }
    return null_instance;
}

JsonValue& JsonValue::operator[](std::size_t idx) {
    if (!std::holds_alternative<Array>(data_)) {
        data_.emplace<Array>();
    }
    auto& arr = std::get<Array>(data_);
    if (idx >= arr.size()) {
        arr.resize(idx + 1);
    }
    return arr[idx];
}

namespace {

void dump_string(const std::string& s, std::ostream& os) {
    os << '"';
    for (char c : s) {
        switch (c) {
            case '"':
                os << "\\\"";
                break;
            case '\\':
                os << "\\\\";
                break;
            case '\b':
                os << "\\b";
                break;
            case '\f':
                os << "\\f";
                break;
            case '\n':
                os << "\\n";
                break;
            case '\r':
                os << "\\r";
                break;
            case '\t':
                os << "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    os << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<int>(static_cast<unsigned char>(c)) << std::dec;
                } else {
                    os << c;
                }
                break;
        }
    }
    os << '"';
}

void dump_value(const JsonValue& val, std::ostream& os, int indent, int current_indent) {
    auto indent_str = [indent](int level) -> std::string {
        if (indent <= 0) {
            return "";
        }
        return std::string(static_cast<std::size_t>(level * indent), ' ');
    };

    switch (val.type()) {
        case JsonType::Null:
            os << "null";
            break;
        case JsonType::Boolean:
            os << (val.as_bool() ? "true" : "false");
            break;
        case JsonType::Number: {
            double d = val.as_double();
            if (std::isnan(d) || std::isinf(d)) {
                os << "null";
            } else if (d == std::floor(d) && !std::isinf(d) && std::abs(d) < 1e15) {
                os << static_cast<int64_t>(d);
            } else {
                std::ostringstream ss;
                ss << std::setprecision(17) << d;
                os << ss.str();
            }
            break;
        }
        case JsonType::String:
            dump_string(val.as_string(), os);
            break;
        case JsonType::Array: {
            const auto& arr = val.as_array();
            if (arr.empty()) {
                os << "[]";
                break;
            }
            os << '[';
            if (indent > 0) {
                os << '\n';
            }
            for (std::size_t i = 0; i < arr.size(); ++i) {
                if (indent > 0) {
                    os << indent_str(current_indent + 1);
                }
                dump_value(arr[i], os, indent, current_indent + 1);
                if (i + 1 < arr.size()) {
                    os << ',';
                }
                if (indent > 0) {
                    os << '\n';
                }
            }
            if (indent > 0) {
                os << indent_str(current_indent);
            }
            os << ']';
            break;
        }
        case JsonType::Object: {
            const auto& obj = val.as_object();
            if (obj.empty()) {
                os << "{}";
                break;
            }
            os << '{';
            if (indent > 0) {
                os << '\n';
            }
            std::size_t i = 0;
            for (const auto& [k, v] : obj) {
                if (indent > 0) {
                    os << indent_str(current_indent + 1);
                }
                dump_string(k, os);
                os << ": ";
                dump_value(v, os, indent, current_indent + 1);
                if (i + 1 < obj.size()) {
                    os << ',';
                }
                if (indent > 0) {
                    os << '\n';
                }
                ++i;
            }
            if (indent > 0) {
                os << indent_str(current_indent);
            }
            os << '}';
            break;
        }
    }
}

class JsonParser {
public:
    explicit JsonParser(std::string_view text) : text_(text), pos_(0) {}

    JsonValue parse() {
        skip_whitespace();
        if (pos_ >= text_.size()) {
            throw std::runtime_error("Empty JSON string");
        }
        JsonValue result = parse_value();
        skip_whitespace();
        if (pos_ < text_.size()) {
            throw std::runtime_error("Unexpected trailing character at index " +
                                     std::to_string(pos_));
        }
        return result;
    }

private:
    std::string_view text_;
    std::size_t pos_{0};

    void skip_whitespace() {
        while (pos_ < text_.size()) {
            char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else if (c == '/' && pos_ + 1 < text_.size() && text_[pos_ + 1] == '/') {
                // Line comment support in JSON
                pos_ += 2;
                while (pos_ < text_.size() && text_[pos_] != '\n') {
                    ++pos_;
                }
            } else {
                break;
            }
        }
    }

    char peek() const {
        if (pos_ >= text_.size()) {
            return '\0';
        }
        return text_[pos_];
    }

    char get() {
        if (pos_ >= text_.size()) {
            throw std::runtime_error("Unexpected end of JSON input");
        }
        return text_[pos_++];
    }

    JsonValue parse_value() {
        skip_whitespace();
        char c = peek();
        if (c == 'n') {
            return parse_null();
        }
        if (c == 't' || c == 'f') {
            return parse_bool();
        }
        if (c == '"') {
            return parse_string();
        }
        if (c == '[') {
            return parse_array();
        }
        if (c == '{') {
            return parse_object();
        }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            return parse_number();
        }
        throw std::runtime_error(std::string("Unexpected character in JSON: '") + c + "' at pos " +
                                 std::to_string(pos_));
    }

    JsonValue parse_null() {
        if (text_.substr(pos_, 4) == "null") {
            pos_ += 4;
            return JsonValue();
        }
        throw std::runtime_error("Expected 'null' at pos " + std::to_string(pos_));
    }

    JsonValue parse_bool() {
        if (text_.substr(pos_, 4) == "true") {
            pos_ += 4;
            return JsonValue(true);
        }
        if (text_.substr(pos_, 5) == "false") {
            pos_ += 5;
            return JsonValue(false);
        }
        throw std::runtime_error("Expected boolean at pos " + std::to_string(pos_));
    }

    JsonValue parse_string() {
        get();  // skip opening quote
        std::string s;
        while (true) {
            char c = get();
            if (c == '"') {
                break;
            }
            if (c == '\\') {
                char esc = get();
                switch (esc) {
                    case '"':
                        s.push_back('"');
                        break;
                    case '\\':
                        s.push_back('\\');
                        break;
                    case '/':
                        s.push_back('/');
                        break;
                    case 'b':
                        s.push_back('\b');
                        break;
                    case 'f':
                        s.push_back('\f');
                        break;
                    case 'n':
                        s.push_back('\n');
                        break;
                    case 'r':
                        s.push_back('\r');
                        break;
                    case 't':
                        s.push_back('\t');
                        break;
                    case 'u': {
                        if (pos_ + 4 > text_.size()) {
                            throw std::runtime_error("Incomplete unicode escape at pos " +
                                                     std::to_string(pos_));
                        }
                        pos_ += 4;
                        s.push_back('?');  // simple fallback
                        break;
                    }
                    default:
                        s.push_back(esc);
                        break;
                }
            } else {
                s.push_back(c);
            }
        }
        return JsonValue(std::move(s));
    }

    JsonValue parse_number() {
        std::size_t start = pos_;
        if (peek() == '-') {
            ++pos_;
        }
        while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
        if (pos_ < text_.size() && text_[pos_] == '.') {
            ++pos_;
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                ++pos_;
            }
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
                ++pos_;
            }
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
                ++pos_;
            }
        }
        std::string num_str(text_.substr(start, pos_ - start));
        double val = std::stod(num_str);
        return JsonValue(val);
    }

    JsonValue parse_array() {
        get();  // skip '['
        JsonValue::Array arr;
        skip_whitespace();
        if (peek() == ']') {
            get();
            return JsonValue(std::move(arr));
        }
        while (true) {
            arr.push_back(parse_value());
            skip_whitespace();
            char c = peek();
            if (c == ']') {
                get();
                break;
            }
            if (c == ',') {
                get();
                skip_whitespace();
            } else {
                throw std::runtime_error(std::string("Expected ',' or ']' in array, got '") + c +
                                         "' at pos " + std::to_string(pos_));
            }
        }
        return JsonValue(std::move(arr));
    }

    JsonValue parse_object() {
        get();  // skip '{'
        JsonValue::Object obj;
        skip_whitespace();
        if (peek() == '}') {
            get();
            return JsonValue(std::move(obj));
        }
        while (true) {
            skip_whitespace();
            if (peek() != '"') {
                throw std::runtime_error(std::string("Expected string key in object at pos ") +
                                         std::to_string(pos_));
            }
            JsonValue key_val = parse_string();
            std::string key = key_val.as_string();
            skip_whitespace();
            if (get() != ':') {
                throw std::runtime_error("Expected ':' after key in object at pos " +
                                         std::to_string(pos_));
            }
            JsonValue val = parse_value();
            obj[std::move(key)] = std::move(val);
            skip_whitespace();
            char c = peek();
            if (c == '}') {
                get();
                break;
            }
            if (c == ',') {
                get();
                skip_whitespace();
            } else {
                throw std::runtime_error(std::string("Expected ',' or '}' in object, got '") + c +
                                         "' at pos " + std::to_string(pos_));
            }
        }
        return JsonValue(std::move(obj));
    }
};

}  // namespace

std::string JsonValue::dump(int indent) const {
    std::ostringstream ss;
    dump_value(*this, ss, indent, 0);
    return ss.str();
}

JsonValue JsonValue::parse(std::string_view json_str) {
    JsonParser parser(json_str);
    return parser.parse();
}

}  // namespace kavach::benchmark
