#pragma once

#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace kavach::test {

struct TestCase {
    std::string name;
    std::function<void()> func;
};

class TestRegistry {
public:
    static TestRegistry& instance() {
        static TestRegistry reg;
        return reg;
    }

    void add_test(const std::string& name, std::function<void()> func) {
        tests_.push_back({name, std::move(func)});
    }

    int run_all() {
        std::size_t passed = 0;
        std::size_t failed = 0;

        std::cout << "Running " << tests_.size() << " tests...\n";
        std::cout << "----------------------------------------\n";

        for (const auto& test : tests_) {
            try {
                test.func();
                std::cout << "[PASS] " << test.name << "\n";
                ++passed;
            } catch (const std::exception& e) {
                std::cout << "[FAIL] " << test.name << ": " << e.what() << "\n";
                ++failed;
            } catch (...) {
                std::cout << "[FAIL] " << test.name << ": Unknown exception\n";
                ++failed;
            }
        }

        std::cout << "----------------------------------------\n";
        std::cout << "Total: " << tests_.size() << " | Passed: " << passed
                  << " | Failed: " << failed << "\n";

        return (failed == 0) ? 0 : 1;
    }

private:
    std::vector<TestCase> tests_;
};

#define KAVACH_TEST(test_name)                                                                     \
    static void test_name();                                                                       \
    namespace {                                                                                    \
    struct Register_##test_name {                                                                  \
        Register_##test_name() {                                                                   \
            ::kavach::test::TestRegistry::instance().add_test(#test_name, test_name);              \
        }                                                                                          \
    } register_instance_##test_name;                                                               \
    }                                                                                              \
    static void test_name()

#define KAVACH_ASSERT(cond)                                                                        \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::ostringstream oss;                                                                \
            oss << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__;             \
            throw std::runtime_error(oss.str());                                                   \
        }                                                                                          \
    } while (false)

#define KAVACH_ASSERT_EQUAL(a, b)                                                                  \
    do {                                                                                           \
        if ((a) != (b)) {                                                                          \
            std::ostringstream oss;                                                                \
            oss << "Equality failed: (" #a " == " #b ") [" << (a) << " != " << (b) << "] at "      \
                << __FILE__ << ":" << __LINE__;                                                    \
            throw std::runtime_error(oss.str());                                                   \
        }                                                                                          \
    } while (false)

#define KAVACH_ASSERT_NEAR(a, b, tol)                                                              \
    do {                                                                                           \
        if (std::abs((a) - (b)) > (tol)) {                                                         \
            std::ostringstream oss;                                                                \
            oss << "Near-equality failed: |" #a " - " #b "| <= " #tol " [diff="                    \
                << std::abs((a) - (b)) << "] at " << __FILE__ << ":" << __LINE__;                  \
            throw std::runtime_error(oss.str());                                                   \
        }                                                                                          \
    } while (false)

}  // namespace kavach::test
