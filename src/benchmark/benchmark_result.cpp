#include "benchmark/benchmark_result.hpp"

#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <thread>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/utsname.h>
#include <unistd.h>
#endif

#include "core/version.hpp"

namespace kavach::benchmark {

std::string to_string(BenchmarkStatus status) {
    switch (status) {
        case BenchmarkStatus::OPTIMAL:
            return "OPTIMAL";
        case BenchmarkStatus::FEASIBLE:
            return "FEASIBLE";
        case BenchmarkStatus::INFEASIBLE:
            return "INFEASIBLE";
        case BenchmarkStatus::UNBOUNDED:
            return "UNBOUNDED";
        case BenchmarkStatus::LIMIT_REACHED:
            return "LIMIT_REACHED";
        case BenchmarkStatus::NUMERICAL_FAILURE:
            return "NUMERICAL_FAILURE";
        case BenchmarkStatus::PARSE_FAILURE:
            return "PARSE_FAILURE";
        case BenchmarkStatus::VERIFICATION_FAILURE:
            return "VERIFICATION_FAILURE";
        case BenchmarkStatus::UNSUPPORTED:
            return "UNSUPPORTED";
        case BenchmarkStatus::RUNTIME_ERROR:
            return "RUNTIME_ERROR";
        case BenchmarkStatus::SOLVER_TIMEOUT:
            return "SOLVER_TIMEOUT";
        case BenchmarkStatus::REFERENCE_UNAVAILABLE:
            return "REFERENCE_UNAVAILABLE";
        case BenchmarkStatus::REFERENCE_TIMEOUT:
            return "REFERENCE_TIMEOUT";
    }
    return "RUNTIME_ERROR";
}

BenchmarkStatus parse_benchmark_status(std::string_view str) {
    if (str == "OPTIMAL") {
        return BenchmarkStatus::OPTIMAL;
    }
    if (str == "FEASIBLE") {
        return BenchmarkStatus::FEASIBLE;
    }
    if (str == "INFEASIBLE") {
        return BenchmarkStatus::INFEASIBLE;
    }
    if (str == "UNBOUNDED") {
        return BenchmarkStatus::UNBOUNDED;
    }
    if (str == "LIMIT_REACHED") {
        return BenchmarkStatus::LIMIT_REACHED;
    }
    if (str == "NUMERICAL_FAILURE") {
        return BenchmarkStatus::NUMERICAL_FAILURE;
    }
    if (str == "PARSE_FAILURE") {
        return BenchmarkStatus::PARSE_FAILURE;
    }
    if (str == "VERIFICATION_FAILURE") {
        return BenchmarkStatus::VERIFICATION_FAILURE;
    }
    if (str == "UNSUPPORTED") {
        return BenchmarkStatus::UNSUPPORTED;
    }
    if (str == "SOLVER_TIMEOUT") {
        return BenchmarkStatus::SOLVER_TIMEOUT;
    }
    if (str == "REFERENCE_UNAVAILABLE") {
        return BenchmarkStatus::REFERENCE_UNAVAILABLE;
    }
    if (str == "REFERENCE_TIMEOUT") {
        return BenchmarkStatus::REFERENCE_TIMEOUT;
    }
    return BenchmarkStatus::RUNTIME_ERROR;
}

BenchmarkStatus from_solver_status(core::SolverStatus s) {
    switch (s) {
        case core::SolverStatus::OPTIMAL:
            return BenchmarkStatus::OPTIMAL;
        case core::SolverStatus::FEASIBLE:
            return BenchmarkStatus::FEASIBLE;
        case core::SolverStatus::INFEASIBLE:
            return BenchmarkStatus::INFEASIBLE;
        case core::SolverStatus::UNBOUNDED:
            return BenchmarkStatus::UNBOUNDED;
        case core::SolverStatus::LIMIT_REACHED:
            return BenchmarkStatus::LIMIT_REACHED;
        case core::SolverStatus::NUMERICAL_FAILURE:
            return BenchmarkStatus::NUMERICAL_FAILURE;
        case core::SolverStatus::UNKNOWN:
            return BenchmarkStatus::RUNTIME_ERROR;
    }
    return BenchmarkStatus::RUNTIME_ERROR;
}

ObjectiveComparison compare_objectives(double solver_obj, double ref_obj, double tol_rel,
                                       double tol_abs) {
    ObjectiveComparison comp;
    comp.abs_error = std::abs(solver_obj - ref_obj);
    double denom = std::max(1.0, std::abs(ref_obj));
    comp.rel_error = comp.abs_error / denom;

    comp.matches = (comp.abs_error <= tol_abs) || (comp.rel_error <= tol_rel);
    return comp;
}

ReproducibilityManifest ReproducibilityManifest::capture_current_environment(
    const std::string& suite) {
    ReproducibilityManifest m;
    m.suite_name = suite;
    m.git_commit = std::string(core::GIT_COMMIT_HASH);
    m.build_type = std::string(core::BUILD_TYPE);
    m.compiler_info = std::string(core::COMPILER_ID) + " " + std::string(core::COMPILER_VERSION);
    m.kavach_version = std::string(core::PROJECT_VERSION);

    // Timestamp UTC ISO 8601
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm gmt{};
#if defined(_WIN32)
    gmtime_s(&gmt, &tt);
#else
    gmtime_r(&tt, &gmt);
#endif
    std::ostringstream ss;
    ss << std::put_time(&gmt, "%Y-%m-%dT%H:%M:%SZ");
    m.timestamp = ss.str();

    // Hardware
    m.num_cpu_cores = std::thread::hardware_concurrency();

#if defined(__unix__) || defined(__APPLE__)
    char hbuf[256];
    if (gethostname(hbuf, sizeof(hbuf)) == 0) {
        m.hostname = hbuf;
    } else {
        m.hostname = "localhost";
    }

    struct utsname un;
    if (uname(&un) == 0) {
        m.os_name = std::string(un.sysname) + " " + un.release + " (" + un.machine + ")";
        m.cpu_model = un.machine;
    } else {
        m.os_name = "POSIX";
        m.cpu_model = "Unknown CPU";
    }
#else
    m.hostname = "WindowsHost";
    m.os_name = "Windows";
    m.cpu_model = "x86_64";
#endif

    // CUDA status
    backend::cuda::CudaBackend cuda_test;
    if (cuda_test.is_available()) {
        m.gpu_device = "CUDA GPU Active";
    } else {
        m.gpu_device = "CUDA unavailable (CPU fallback)";
    }

    return m;
}

}  // namespace kavach::benchmark
