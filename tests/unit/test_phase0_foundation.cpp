#include "../test_framework.hpp"
#include "core/error.hpp"
#include "core/logger.hpp"
#include "core/random.hpp"
#include "core/status.hpp"
#include "core/tolerances.hpp"
#include "core/version.hpp"

// Architecture interface headers
#include "api/solver.hpp"
#include "backend/execution_backend.hpp"
#include "model/model.hpp"
#include "strategy/strategy.hpp"
#include "verification/verifier.hpp"

using namespace kavach;

KAVACH_TEST(test_status_taxonomy) {
    using namespace core;
    KAVACH_ASSERT_EQUAL(to_string(SolverStatus::OPTIMAL), "OPTIMAL");
    KAVACH_ASSERT_EQUAL(to_string(SolverStatus::FEASIBLE), "FEASIBLE");
    KAVACH_ASSERT_EQUAL(to_string(SolverStatus::INFEASIBLE), "INFEASIBLE");
    KAVACH_ASSERT_EQUAL(to_string(SolverStatus::UNBOUNDED), "UNBOUNDED");
    KAVACH_ASSERT_EQUAL(to_string(SolverStatus::LIMIT_REACHED), "LIMIT_REACHED");
    KAVACH_ASSERT_EQUAL(to_string(SolverStatus::NUMERICAL_FAILURE), "NUMERICAL_FAILURE");
    KAVACH_ASSERT_EQUAL(to_string(SolverStatus::UNKNOWN), "UNKNOWN");

    KAVACH_ASSERT(is_terminal(SolverStatus::OPTIMAL));
    KAVACH_ASSERT(is_terminal(SolverStatus::INFEASIBLE));
    KAVACH_ASSERT(is_terminal(SolverStatus::UNBOUNDED));
    KAVACH_ASSERT(is_terminal(SolverStatus::LIMIT_REACHED));
    KAVACH_ASSERT(is_terminal(SolverStatus::NUMERICAL_FAILURE));
    KAVACH_ASSERT(!is_terminal(SolverStatus::FEASIBLE));
    KAVACH_ASSERT(!is_terminal(SolverStatus::UNKNOWN));
}

KAVACH_TEST(test_default_tolerances) {
    using namespace core;
    SolverTolerances tol;
    KAVACH_ASSERT_NEAR(tol.primal_feasibility_tol, 1e-7, 1e-15);
    KAVACH_ASSERT_NEAR(tol.dual_feasibility_tol, 1e-7, 1e-15);
    KAVACH_ASSERT_NEAR(tol.integrality_tol, 1e-6, 1e-15);
    KAVACH_ASSERT_NEAR(tol.optimality_tol, 1e-7, 1e-15);
    KAVACH_ASSERT_NEAR(tol.pivot_tol, 1e-9, 1e-15);
    KAVACH_ASSERT_NEAR(tol.residual_tol, 1e-6, 1e-15);
}

KAVACH_TEST(test_error_hierarchy) {
    using namespace core;
    try {
        throw InputException("malformed MPS line");
    } catch (const KavachException& e) {
        std::string msg = e.what();
        KAVACH_ASSERT(msg.find("Input error") != std::string::npos);
    }

    try {
        throw NumericalException("near-zero pivot detected");
    } catch (const KavachException& e) {
        std::string msg = e.what();
        KAVACH_ASSERT(msg.find("Numerical error") != std::string::npos);
    }
}

KAVACH_TEST(test_deterministic_rng) {
    using namespace core;
    DeterministicRNG rng1(12345);
    DeterministicRNG rng2(12345);

    // Same seed must produce byte-for-byte identical sequence
    for (int i = 0; i < 100; ++i) {
        double r1 = rng1.uniform_real();
        double r2 = rng2.uniform_real();
        KAVACH_ASSERT_EQUAL(r1, r2);

        int i1 = rng1.uniform_int(0, 1000);
        int i2 = rng2.uniform_int(0, 1000);
        KAVACH_ASSERT_EQUAL(i1, i2);
    }

    // Different seed must produce different sequence
    DeterministicRNG rng3(54321);
    bool any_difference = false;
    for (int i = 0; i < 10; ++i) {
        if (rng1.uniform_real() != rng3.uniform_real()) {
            any_difference = true;
            break;
        }
    }
    KAVACH_ASSERT(any_difference);
}

KAVACH_TEST(test_logger_telemetry) {
    using namespace core;
    Logger::instance().set_level(LogLevel::DEBUG);
    KAVACH_ASSERT(Logger::instance().get_level() == LogLevel::DEBUG);

    KAVACH_LOG_DEBUG("Testing debug log in Phase 0 test");
    KAVACH_LOG_INFO("Testing info log in Phase 0 test");

    Logger::instance().telemetry("test_metric", 42.0);
    Logger::instance().telemetry("test_status", "OK");

    Logger::instance().set_level(LogLevel::INFO);
}

KAVACH_TEST(test_architecture_interfaces) {
    // Verify that canonical model default constructor works
    model::CanonicalModel empty_model;
    KAVACH_ASSERT_EQUAL(empty_model.num_variables(), 0);
    KAVACH_ASSERT_EQUAL(empty_model.num_constraints(), 0);
    KAVACH_ASSERT_EQUAL(empty_model.num_nonzeros(), 0);

    // Verify version symbols
    KAVACH_ASSERT(!core::PROJECT_NAME.empty());
    KAVACH_ASSERT(!core::PROJECT_VERSION.empty());
}
