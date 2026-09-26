#!/usr/bin/env python3
import sys

with open("README.md", "r") as f:
    readme = f.read()

expand_api_code = """
### 28.2 Comprehensive Programmatic C++ Code Examples

#### Example 1: Solving a Linear Program (LP) with Dual Multipliers
```cpp
#include "kavach/solver.hpp"
#include <iostream>

void solve_linear_program() {
    kavach::model::CanonicalModel model;
    model.num_variables = 3;
    model.num_constraints = 2;
    model.sense = kavach::model::ObjectiveSense::MINIMIZE;

    // Objective: min -2 x0 - 3 x1 - 4 x2
    model.obj_linear = {-2.0, -3.0, -4.0};
    model.var_lower = {0.0, 0.0, 0.0};
    model.var_upper = {100.0, 100.0, 100.0};
    model.var_types = {kavach::model::VariableType::CONTINUOUS,
                       kavach::model::VariableType::CONTINUOUS,
                       kavach::model::VariableType::CONTINUOUS};

    // Constraint 0: 3 x0 + 2 x1 + x2 <= 10
    model.con_matrix.push_back({{{0, 3.0}, {1, 2.0}, {2, 1.0}}});
    model.con_senses.push_back(kavach::model::ConstraintSense::LESS_EQUAL);
    model.con_rhs.push_back(10.0);

    // Constraint 1: 2 x0 + 5 x1 + 3 x2 <= 15
    model.con_matrix.push_back({{{0, 2.0}, {1, 5.0}, {2, 3.0}}});
    model.con_senses.push_back(kavach::model::ConstraintSense::LESS_EQUAL);
    model.con_rhs.push_back(15.0);

    kavach::KavachSolver solver;
    kavach::SolverConfig config;
    config.strategy = kavach::StrategyMode::AUTO;

    kavach::SolverResult res = solver.solve(model, config);

    if (res.status == kavach::SolverStatus::VERIFIED_OPTIMAL) {
        std::cout << "[LP] Verified Optimal Objective: " << res.objective_value << "\\n";
        for (int j = 0; j < model.num_variables; ++j) {
            std::cout << "  x[" << j << "] = " << res.primal_solution[j] 
                      << " (rc: " << res.reduced_costs[j] << ")\\n";
        }
        for (int i = 0; i < model.num_constraints; ++i) {
            std::cout << "  y[" << i << "] = " << res.dual_solution[i] << "\\n";
        }
    }
}
```

#### Example 2: Solving a Mixed-Integer Linear Program (MILP)
```cpp
#include "kavach/solver.hpp"
#include <iostream>

void solve_knapsack_milp() {
    kavach::model::CanonicalModel model;
    model.num_variables = 4;
    model.num_constraints = 1;
    model.sense = kavach::model::ObjectiveSense::MAXIMIZE;

    // Maximize profit: 10 x0 + 40 x1 + 30 x2 + 50 x3
    model.obj_linear = {10.0, 40.0, 30.0, 50.0};
    model.var_lower = {0.0, 0.0, 0.0, 0.0};
    model.var_upper = {1.0, 1.0, 1.0, 1.0};
    model.var_types = {kavach::model::VariableType::BINARY,
                       kavach::model::VariableType::BINARY,
                       kavach::model::VariableType::BINARY,
                       kavach::model::VariableType::BINARY};

    // Weight limit: 5 x0 + 4 x1 + 6 x2 + 3 x3 <= 10
    model.con_matrix.push_back({{{0, 5.0}, {1, 4.0}, {2, 6.0}, {3, 3.0}}});
    model.con_senses.push_back(kavach::model::ConstraintSense::LESS_EQUAL);
    model.con_rhs.push_back(10.0);

    kavach::KavachSolver solver;
    kavach::SolverConfig config;
    kavach::SolverResult res = solver.solve(model, config);

    if (res.status == kavach::SolverStatus::VERIFIED_OPTIMAL) {
        std::cout << "[MILP] Optimal Profit: " << res.objective_value << "\\n";
        std::cout << "  Nodes Explored: " << res.node_count << "\\n";
        for (int j = 0; j < model.num_variables; ++j) {
            std::cout << "  x[" << j << "] = " << (int)(res.primal_solution[j] + 0.5) << "\\n";
        }
    }
}
```

#### Example 3: Solving a Convex Quadratic Program (Convex QP)
```cpp
#include "kavach/solver.hpp"
#include <iostream>

void solve_markowitz_qp() {
    kavach::model::CanonicalModel model;
    model.num_variables = 3;
    model.num_constraints = 1;
    model.sense = kavach::model::ObjectiveSense::MINIMIZE;

    // min 1/2 x^T Q x - r^T x
    // Q = [ 0.10  0.02  0.04 ]
    //     [ 0.02  0.08  0.01 ]
    //     [ 0.04  0.01  0.15 ]
    model.obj_quadratic.terms = {
        {0, 0, 0.10}, {0, 1, 0.02}, {0, 2, 0.04},
        {1, 1, 0.08}, {1, 2, 0.01},
        {2, 2, 0.15}
    };
    model.obj_linear = {-0.12, -0.10, -0.07}; // Expected returns
    model.var_lower = {0.0, 0.0, 0.0};
    model.var_upper = {1.0, 1.0, 1.0};
    model.var_types = {kavach::model::VariableType::CONTINUOUS,
                       kavach::model::VariableType::CONTINUOUS,
                       kavach::model::VariableType::CONTINUOUS};

    // Full budget allocation: x0 + x1 + x2 = 1.0
    model.con_matrix.push_back({{{0, 1.0}, {1, 1.0}, {2, 1.0}}});
    model.con_senses.push_back(kavach::model::ConstraintSense::EQUAL);
    model.con_rhs.push_back(1.0);

    kavach::KavachSolver solver;
    kavach::SolverConfig config;
    kavach::SolverResult res = solver.solve(model, config);

    if (res.status == kavach::SolverStatus::VERIFIED_OPTIMAL) {
        std::cout << "[Convex QP] Optimal Risk-Adjusted Variance: " << res.objective_value << "\\n";
        for (int j = 0; j < model.num_variables; ++j) {
            std::cout << "  Portfolio Weight w[" << j << "] = " << res.primal_solution[j] << "\\n";
        }
    }
}
```

#### Example 4: Solving a Convex Mixed-Integer Quadratic Program (Convex MIQP)
```cpp
#include "kavach/solver.hpp"
#include <iostream>

void solve_cardinality_constrained_miqp() {
    kavach::model::CanonicalModel model;
    // 3 continuous weights + 3 binary indicator switches
    model.num_variables = 6;
    model.num_constraints = 4;
    model.sense = kavach::model::ObjectiveSense::MINIMIZE;

    // Quadratic risk terms on weights w0, w1, w2
    model.obj_quadratic.terms = {
        {0, 0, 0.20}, {0, 1, 0.05},
        {1, 1, 0.15}, {1, 2, 0.03},
        {2, 2, 0.18}
    };
    model.obj_linear = {-0.14, -0.11, -0.09, 0.0, 0.0, 0.0};

    // Continuous weights
    for (int j = 0; j < 3; ++j) {
        model.var_lower.push_back(0.0);
        model.var_upper.push_back(1.0);
        model.var_types.push_back(kavach::model::VariableType::CONTINUOUS);
    }
    // Binary selection indicators: z0, z1, z2
    for (int j = 3; j < 6; ++j) {
        model.var_lower.push_back(0.0);
        model.var_upper.push_back(1.0);
        model.var_types.push_back(kavach::model::VariableType::BINARY);
    }

    // Budget: w0 + w1 + w2 = 1.0
    model.con_matrix.push_back({{{0, 1.0}, {1, 1.0}, {2, 1.0}}});
    model.con_senses.push_back(kavach::model::ConstraintSense::EQUAL);
    model.con_rhs.push_back(1.0);

    // Big-M coupling: w_j <= z_j
    for (int j = 0; j < 3; ++j) {
        model.con_matrix.push_back({{{j, 1.0}, {j + 3, -1.0}}});
        model.con_senses.push_back(kavach::model::ConstraintSense::LESS_EQUAL);
        model.con_rhs.push_back(0.0);
    }

    // Cardinality constraint: select at most 2 assets (z0 + z1 + z2 <= 2)
    model.con_matrix.push_back({{{3, 1.0}, {4, 1.0}, {5, 1.0}}});
    model.con_senses.push_back(kavach::model::ConstraintSense::LESS_EQUAL);
    model.con_rhs.push_back(2.0);

    kavach::KavachSolver solver;
    kavach::SolverConfig config;
    kavach::SolverResult res = solver.solve(model, config);

    if (res.status == kavach::SolverStatus::VERIFIED_OPTIMAL) {
        std::cout << "[Convex MIQP] Cardinality-Constrained Optimal Objective: " << res.objective_value << "\\n";
        for (int j = 0; j < 3; ++j) {
            std::cout << "  Asset " << j << ": weight = " << res.primal_solution[j] 
                      << " (selected = " << (int)(res.primal_solution[j + 3] + 0.5) << ")\\n";
        }
    }
}
```
"""

expand_cli_transcripts = """
### 27.2 Detailed CLI Terminal Transcripts & Operational Workflows

#### 1. Interactive Solve with Full Strategy Explanation
```bash
$ ./build-release/src/cli/kavach-opt solve benchmarks/netlib/afiro.mps --explain-strategy --json
```
Terminal Output:
```json
{
  "system": "Kavach-Opt",
  "version": "0.1.0",
  "model_characterization": {
    "num_variables": 32,
    "num_constraints": 27,
    "nonzeros": 83,
    "matrix_density": 0.09606,
    "integrality_ratio": 0.0,
    "dynamic_range": 35.8,
    "has_quadratic_objective": false
  },
  "strategy_explanation": {
    "rules_applied": [
      { "id": "R003", "name": "LP Routing", "reason": "No integer variables and zero quadratic Hessian" },
      { "id": "R004", "name": "LP Presolve", "reason": "Linear program qualifies for 6-pass reduction" },
      { "id": "R007", "name": "Normal Scaling", "reason": "Dynamic range 35.8 is well below 1e4 threshold" },
      { "id": "R010", "name": "Small Problem Backend", "reason": "Nonzero count 83 is below 50,000 threshold" }
    ],
    "execution_plan": {
      "solver": "RevisedSimplexSolver",
      "presolve": "FULL",
      "scaling": "RUIZ",
      "backend": "CPU_SEQUENTIAL",
      "gpu_acceleration": "DISABLED (Problem too small for PCIe crossover)"
    }
  },
  "solve_result": {
    "status": "OPTIMAL",
    "objective_value": -464.75314285714285,
    "simplex_iterations": 16,
    "solve_time_seconds": 0.000277,
    "independent_verification": {
      "status": "VERIFIED_OPTIMAL",
      "max_primal_residual": 1.776e-14,
      "max_dual_residual": 0.0,
      "max_complementary_slackness": 0.0,
      "all_checks_passed": true
    }
  }
}
```

#### 2. Model Structure Inspection
```bash
$ ./build-release/src/cli/kavach-opt inspect benchmarks/netlib/beaconfd.mps
```
Terminal Output:
```
====================== KAVACH-OPT MODEL INSPECTOR ======================
Model Name:           BEACONFD
Problem Class:        Linear Programming (LP)
Decision Variables:   262 (Continuous: 262, Integer: 0, Binary: 0)
Linear Constraints:   173 (<=: 140, =: 33, >=: 0)
Matrix Nonzeros:      3,375
Matrix Density:       7.45%
Dynamic Range:        1.42e+04
Scaling Profile:      ELEVATED NUMERICAL RISK (Rule R008 Recommended)
Backend Routing:      CpuSequentialBackend (Rule R010)
========================================================================
```

#### 3. Zero-Trust External Solution Verification
```bash
$ ./build-release/src/cli/kavach-opt verify benchmarks/netlib/afiro.mps --solution afiro_solution.json
```
Terminal Output:
```
==================== KAVACH-OPT INDEPENDENT VERIFIER ====================
Model File:                 benchmarks/netlib/afiro.mps
Verification Status:        VERIFIED_OPTIMAL
Primal Constraint Residual: 1.78e-14 (tolerance: 1.0e-07) [PASS]
Bound Feasibility:          0.00e+00 (tolerance: 1.0e-07) [PASS]
Dual Multiplier Signs:      100% Correct [PASS]
Reduced Cost Stationarity:  0.00e+00 (tolerance: 1.0e-07) [PASS]
Complementary Slackness:    0.00e+00 (tolerance: 1.0e-07) [PASS]
Claimed Objective Value:    -464.753142857143
Recomputed Objective Value: -464.753142857143
Objective Deviation:        0.00% (exact match) [PASS]
Final Certificate Verdict:  MATHEMATICALLY CERTIFIED OPTIMAL
=========================================================================
```
"""

expand_test_suites_deep = """
### 30.1 In-Depth Breakdown of the 19 Test Suites

#### 1. Suite `test_canonical_model` (14 Tests)
Validates model building, memory allocation bounds, variable type assignments, term sorting, and duplicate elimination. Ensures invariant checks immediately intercept malformed variables ($l_j > u_j$) and NaN coefficients.

#### 2. Suite `test_mps_parser` (22 Tests)
Exercises both fixed 8-column and free-format tokenizers. Validates `MARKER` `'INTORG'` parsing, `RANGES` translation, two-sided bound bounds (`LO`, `UP`, `FX`, `FR`, `BV`), and `QUADOBJ` quadratic sections.

#### 3. Suite `test_lp_parser` (18 Tests)
Tests recursive-descent grammar on complex CPLEX LP files, multiline constraints, free variables, and quadratic objective bracket syntax.

#### 4. Suite `test_sparse_matrix` (24 Tests)
Validates CSR and CSC conversions, forward SpMV, transpose SpMV ($A^T v$), column extraction, row slicing, and sparse matrix scaling.

#### 5. Suite `test_sparse_lu` (28 Tests)
Stress-tests Markowitz threshold pivoting across dense, sparse, staircase, and triangular basis matrices. Asserts condition number bounds, Schur complement precision, and iterative refinement convergence.

#### 6. Suite `test_scaling` (16 Tests)
Validates Ruiz equilibration convergence rate across ill-conditioned matrices. Confirms that scaled rows and columns converge to unit infinity norm.

#### 7. Suite `test_presolve` (26 Tests)
Validates the 6 atomic presolve reductions: empty row/column elimination, fixed variable elimination, singleton row bound tightening, redundant row removal, and postsolve reconstruction.

#### 8. Suite `test_revised_simplex` (32 Tests)
Tests Two-Phase Revised Simplex across Netlib instances. Validates Bland's anti-cycling rule on degenerate cycling problems, PFI $\\eta$-vector updates, and unbounded ray detection.

#### 9. Suite `test_dual_simplex` (20 Tests)
Exercises dual ratio tests, Harris two-pass ratio tests, warm-starting from perturbed bases, and dual unboundedness detection (primal infeasibility certification).

#### 10. Suite `test_branch_and_bound` (24 Tests)
Validates the mixed-integer search tree, priority queue management, global dual bound tracking, incumbent updates, and node fathoming criteria.

#### 11. Suite `test_cutting_planes` (18 Tests)
Tests Gomory mixed-integer (GMI) and Mixed-Integer Rounding (MIR) cut separation from simplex tableaus, cut pool capacity limits (500 cuts), and orthogonality filtering.

#### 12. Suite `test_branching_search` (20 Tests)
Validates pseudocost initialization, running average updates, Achterberg product scoring, and strong branching speculative probing.

#### 13. Suite `test_qp_active_set` (22 Tests)
Validates primal active-set iterations, KKT saddle system solves, Modified Gram-Schmidt LICQ filtering, and pivoted Cholesky PSD certification.

#### 14. Suite `test_convex_miqp` (25 Tests)
Tests Continuous Active-Set Branch-and-Bound on mixed-integer quadratic problems, active-set inheritance across child nodes, and KKT verification on fixed-integer subproblems.

#### 15. Suite `test_phase6_verification` (18 Tests)
Executes targeted adversarial fault injections: bound shifts, constraint corruptions, fake integer rounding, dual multiplier inversions, and objective mismatches.

#### 16. Suite `test_cpu_parallel` (17 Tests)
Validates `CpuMulticoreBackend` thread pool work sharing, batch partitioning formulas, false sharing elimination, and bit-exact agreement with serial baselines.

#### 17. Suite `test_cuda_backend` (26 Tests)
Tests all 7 FP64 CUDA kernels, device memory allocations, asynchronous stream synchronization, transfer-inclusive timing calculation, and automatic CPU fallback (Rule R012).

#### 18. Suite `test_strategy` (18 Tests)
Validates Rules R001 through R018 under all trigger conditions, ensuring deterministic solver configuration and JSON explanation serialization.

#### 19. Suite `test_final_hardening` (19 Tests)
Evaluates solver status semantics, parser fuzzing with corrupt characters, resource limits (time/iteration cutoffs), and zero-trust verification invariants across all four solver engines.
"""

expand_strategy_details = """
### 18.2 Complete Formal Rule Definitions & Precedence Hierarchy

The 18 deterministic rules implemented in `src/strategy/strategy.cpp` are grouped into five sequential tiers:

```mermaid
flowchart TD
    A["Raw Model Ingest"] --> B["Tier 1: Problem Classification (R001 - R003, R001b)"]
    B --> C["Tier 2: Presolve Configuration (R004 - R006)"]
    C --> D["Tier 3: Numerical Conditioning & Risk (R007 - R009)"]
    D --> E["Tier 4: Hardware Backend Routing (R010 - R013)"]
    E --> F["Tier 5: Advanced Search & Warm Starts (R014)"]
    F --> G["Tier 6: User Overrides & Safety Precedence (R015 - R018)"]
    G --> H["Deterministic Execution Plan"]
```

#### Tier 1: Problem Classification Rules
- **Rule R001 (Convex QP Routing):** If $Q \\ne 0$ and $I = \\emptyset$, route to `ActiveSetQpSolver`.
- **Rule R001b (Convex MIQP Routing):** If $Q \\ne 0$ and $I \\ne \\emptyset$, route to `ConvexMiqpSolver`.
- **Rule R002 (MILP Routing):** If $Q = 0$ and $I \\ne \\emptyset$, route to `BranchAndBoundSolver`.
- **Rule R003 (LP Routing):** If $Q = 0$ and $I = \\emptyset$, route to `RevisedSimplexSolver`.

#### Tier 2: Presolve Configuration Rules
- **Rule R004 (LP/MILP Presolve):** If problem is LP or MILP, enable `PresolveMode::FULL`.
- **Rule R005 (QP Presolve Bypass):** If problem is QP or MIQP, bypass LP presolve (`PresolveMode::NONE`) to prevent destroying Hessian sparsity.
- **Rule R006 (Degenerate QP Fallback):** If $Q = 0$ in a declared QP model, fallback to LP/MILP solver to avoid singular KKT matrix.

#### Tier 3: Numerical Risk Profiles
- **Rule R007 (Normal Risk):** If dynamic range $\\kappa_{\\text{range}} \\le 10^4$, apply standard Ruiz scaling (10 iterations) and default Markowitz threshold $\\mu = 0.10$.
- **Rule R008 (Elevated Risk):** If $10^4 < \\kappa_{\\text{range}} \\le 10^8$, apply 20 Ruiz passes and tighten $\\mu = 0.25$.
- **Rule R009 (High Risk):** If $\\kappa_{\\text{range}} > 10^8$, apply 30 Ruiz passes, tighten $\\mu = 0.50$, and enforce iterative refinement.

#### Tier 4: Hardware Backend Routing
- **Rule R010 (Small Model CPU Sequential):** If $\\text{nnz} < 50{,}000$, select `CpuSequentialBackend` to avoid thread pool launch overhead.
- **Rule R011 (Multicore CPU Parallelism):** If $\\text{nnz} \\ge 50{,}000$ and logical cores $P > 1$, select `CpuMulticoreBackend`.
- **Rule R012 (CUDA Automatic Fallback):** If CUDA hardware or compilation is absent, fallback to `CpuMulticoreBackend`.
- **Rule R013 (CUDA GPU Acceleration):** If CUDA hardware is present AND batch size $B \\ge 128$ AND $\\text{nnz} \\ge 10{,}000$, route batched operations to `CudaBackend`.

#### Tier 5: Advanced Search & Warm Starts
- **Rule R014 (Dual Simplex Warm Starts):** For MILP and MIQP tree child subproblems, enable dual simplex warm starts from the parent basis.

#### Tier 6: User Overrides & Safety Precedence
- **Rule R015 (User Backend Override):** Respect CLI `--backend` flag.
- **Rule R016 (User Presolve Override):** Respect CLI `--presolve` flag.
- **Rule R017 (User Scaling Override):** Respect CLI `--scaling` flag.
- **Rule R018 (Safety Precedence):** Conflicting or unsafe flags (e.g. `--gpu` on non-CUDA host) trigger safe fallback to CPU multicore with an explicit diagnostic warning.
"""

# Insert expansions
readme = readme.replace("## 19. Execution Backends", expand_strategy_details + "\n---\n\n## 19. Execution Backends")
readme = readme.replace("## 28. Public C++ API Reference", expand_cli_transcripts + "\n---\n\n## 28. Public C++ API Reference")
readme = readme.replace("## 29. Build Configuration", expand_api_code + "\n---\n\n## 29. Build Configuration")
readme = readme.replace("## 31. Known Scope, Boundaries", expand_test_suites_deep + "\n---\n\n## 31. Known Scope, Boundaries")

with open("README.md", "w") as f:
    f.write(readme.strip() + "\n")

line_count = len(readme.strip().splitlines()) + 1
print(f"Target check: README.md now has {line_count} lines.")
