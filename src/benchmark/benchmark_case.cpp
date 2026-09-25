#include "benchmark/benchmark_case.hpp"

#include <algorithm>

namespace kavach::benchmark {

std::string to_string(BenchmarkProblemType type) {
    switch (type) {
        case BenchmarkProblemType::LP:
            return "LP";
        case BenchmarkProblemType::MILP:
            return "MILP";
        case BenchmarkProblemType::QP:
            return "QP";
        case BenchmarkProblemType::SPMV:
            return "SPMV";
        case BenchmarkProblemType::UNKNOWN:
            return "UNKNOWN";
    }
    return "UNKNOWN";
}

BenchmarkProblemType parse_problem_type(std::string_view str) {
    if (str == "LP" || str == "lp") {
        return BenchmarkProblemType::LP;
    }
    if (str == "MILP" || str == "milp") {
        return BenchmarkProblemType::MILP;
    }
    if (str == "QP" || str == "qp") {
        return BenchmarkProblemType::QP;
    }
    if (str == "SPMV" || str == "spmv") {
        return BenchmarkProblemType::SPMV;
    }
    return BenchmarkProblemType::UNKNOWN;
}

bool BenchmarkCase::has_tag(std::string_view tag) const noexcept {
    for (const auto& t : tags) {
        if (t == tag) {
            return true;
        }
    }
    return false;
}

const BenchmarkCase* BenchmarkSuite::find_case(std::string_view id) const noexcept {
    for (const auto& c : cases) {
        if (c.case_id == id) {
            return &c;
        }
    }
    return nullptr;
}

}  // namespace kavach::benchmark
