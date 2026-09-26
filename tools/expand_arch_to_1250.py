#!/usr/bin/env python3
import sys

with open("docs/ARCHITECTURE.md", "r") as f:
    arch = f.read()

expand_verifier = """
### 14.2 Mathematical Verifier Class Definitions
The zero-trust verification subsystem is structured into four independent, specialized verifier classes located in `src/verification/`:

```cpp
namespace kavach::verification {

struct VerificationResult {
    bool passed = false;
    std::string failure_reason;
    double max_primal_residual = 0.0;
    double max_dual_residual = 0.0;
    double max_integrality_violation = 0.0;
    double max_complementary_slackness = 0.0;
    double max_bound_violation = 0.0;
    double claimed_objective = 0.0;
    double verified_objective = 0.0;
    
    std::string to_json() const;
};

// Abstract zero-trust verifier base contract
class Verifier {
public:
    virtual ~Verifier() = default;
    virtual VerificationResult verify(
        const model::CanonicalModel& original_model,
        const std::vector<double>& primal_solution,
        const std::vector<double>& dual_solution,
        double claimed_objective) const = 0;
};

// Specialized LP verifier checking primal/dual feasibility and complementary slackness
class LPVerifier final : public Verifier {
public:
    VerificationResult verify(
        const model::CanonicalModel& original_model,
        const std::vector<double>& primal_solution,
        const std::vector<double>& dual_solution,
        double claimed_objective) const override;
};

// Specialized MILP verifier validating integrality of I variables against epsilon_int
class MILPVerifier final : public Verifier {
public:
    VerificationResult verify(
        const model::CanonicalModel& original_model,
        const std::vector<double>& primal_solution,
        const std::vector<double>& dual_solution,
        double claimed_objective) const override;
};

// Specialized QP verifier validating KKT stationarity ||Qx + c - A^T y - mu||_inf
class QPVerifier final : public Verifier {
public:
    VerificationResult verify(
        const model::CanonicalModel& original_model,
        const std::vector<double>& primal_solution,
        const std::vector<double>& dual_solution,
        double claimed_objective) const override;
};

// Specialized MIQP verifier validating integrality and KKT on fixed-integer subproblem
class MIQPVerifier final : public Verifier {
public:
    VerificationResult verify(
        const model::CanonicalModel& original_model,
        const std::vector<double>& primal_solution,
        const std::vector<double>& dual_solution,
        double claimed_objective) const override;
};

} // namespace kavach::verification
```

### 14.3 Adversarial Fault Injection Testing Suite
To ensure absolute robustness, `tests/unit/test_phase6_verification.cpp` and `tests/unit/test_final_hardening.cpp` subject the verification layer to extensive adversarial perturbations:
- **Perturbing Primal Bounds:** Injecting $+10^{-5}$ shift to a bounded variable triggers immediate `PRIMAL_BOUND_VIOLATION`.
- **Perturbing Linear Constraints:** Adding $+10^{-4}$ to $x_0$ such that $A_{i,:} x - b_i > 10^{-7}$ triggers `PRIMAL_CONSTRAINT_VIOLATION`.
- **Faking Integrality:** Altering an integer variable from $1.0$ to $0.999$ triggers `INTEGRALITY_VIOLATION`.
- **Faking Dual Multipliers:** Setting dual multipliers $y$ to zero on active constraints triggers `KKT_STATIONARITY_VIOLATION`.
- **Corrupting Objective Value:** Supplying a reported objective differing by $10^{-4}$ triggers `OBJECTIVE_CONSISTENCY_VIOLATION`.
"""

expand_api = """
### 19.3 Public C++ Header Specification (`include/kavach/solver.hpp`)
The primary programmatic C++ interface is defined in `include/kavach/solver.hpp`:

```cpp
#ifndef KAVACH_SOLVER_HPP
#define KAVACH_SOLVER_HPP

#include "kavach/model.hpp"
#include "kavach/types.hpp"
#include <memory>
#include <string>
#include <vector>

namespace kavach {

struct SolverConfig {
    StrategyMode strategy = StrategyMode::AUTO;
    PresolveMode presolve = PresolveMode::AUTO;
    ScalingMode scaling = ScalingMode::AUTO;
    BackendType backend = BackendType::AUTO;
    double time_limit_seconds = 3600.0;
    int max_iterations = 1000000;
    double primal_tolerance = 1e-7;
    double dual_tolerance = 1e-7;
    double integrality_tolerance = 1e-6;
    bool enable_gpu = false;
    bool verbose = false;
};

struct SolverResult {
    SolverStatus status = SolverStatus::UNKNOWN;
    double objective_value = 0.0;
    std::vector<double> primal_solution;
    std::vector<double> dual_solution;
    std::vector<double> reduced_costs;
    double solve_time_ms = 0.0;
    int iteration_count = 0;
    int node_count = 0;
    bool independently_verified = false;
    std::string verification_certificate_json;
    std::string strategy_explanation_json;
};

class KavachSolver {
public:
    KavachSolver();
    ~KavachSolver();

    // Primary entrypoint: solves the canonical model according to config
    SolverResult solve(const model::CanonicalModel& model, const SolverConfig& config = SolverConfig());

    // Inspects model characteristics without solving
    std::string inspect_model(const model::CanonicalModel& model) const;

    // Evaluates strategy engine routing rules and returns JSON explanation
    std::string explain_strategy(const model::CanonicalModel& model, const SolverConfig& config) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kavach

#endif // KAVACH_SOLVER_HPP
```

### 19.4 Thread Safety & Concurrency Contracts
1. **Re-Entrancy:** `KavachSolver` instances are thread-safe across independent solver objects. Multiple threads may concurrently call `solver.solve()` on distinct `CanonicalModel` instances without locks.
2. **Immutable Input Models:** A single `CanonicalModel` may be shared concurrently across multiple reader threads and solver instances without mutex synchronization.
3. **Internal Parallelism:** When `CpuMulticoreBackend` is selected, `KavachSolver` utilizes internal thread pooling (`std::jthread`) to parallelize batched operations while keeping the public API call synchronous and blocking.
"""

arch = arch.replace("## 15. Hardware Execution Engines", expand_verifier + "\n---\n\n## 15. Hardware Execution Engines")
arch = arch.replace("## 20. Build Configuration", expand_api + "\n---\n\n## 20. Build Configuration")

with open("docs/ARCHITECTURE.md", "w") as f:
    f.write(arch.strip() + "\n")

line_count = len(arch.strip().splitlines()) + 1
print(f"Final check: docs/ARCHITECTURE.md now has {line_count} lines.")
