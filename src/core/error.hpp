#pragma once

#include <stdexcept>
#include <string>

namespace kavach::core {

class KavachException : public std::runtime_error {
public:
    explicit KavachException(const std::string& message) : std::runtime_error(message) {}
};

class InputException : public KavachException {
public:
    explicit InputException(const std::string& message)
        : KavachException("Input error: " + message) {}
};

class NumericalException : public KavachException {
public:
    explicit NumericalException(const std::string& message)
        : KavachException("Numerical error: " + message) {}
};

class VerificationException : public KavachException {
public:
    explicit VerificationException(const std::string& message)
        : KavachException("Verification failure: " + message) {}
};

}  // namespace kavach::core
