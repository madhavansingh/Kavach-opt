# Kavach-Opt Reproducibility Guide

**Project**: Kavach-Opt Optimization Suite (SIH26119)  
**Standard**: IEEE / ACM Artifact Reproducibility Specification  
**Scope**: Deterministic Mathematical Program Solving & Telemetry  

---

## 1. Reproducibility Guarantee

Kavach-Opt guarantees **bitwise mathematical determinism** under fixed solver configurations:
- **Identical Objective Value**: Given the same model, seed, and tolerances, repeated executions produce identical objective values to double-precision tolerance ($< 10^{-14}$).
- **Identical Basis & Node Trajectory**: Revised simplex pivoting sequences and Branch-and-Bound node expansions follow identical deterministic paths.
- **Identical Rule Strategy Decisions**: The rule-based strategy layer (`R001`–`R018`) evaluates identical signals to fire the exact same rules across platforms.
- **Environment Invariance**: Randomness is governed exclusively by `core::RandomGenerator` using a 64-bit PCG engine initialized with an explicit user seed (`--seed <uint64>`).

---

## 2. Reproduction Recipes

### 2.1. Linear Programming (LP) Solve
Solve the Netlib `afiro` benchmark problem using Revised Simplex with standard presolve:
```bash
# Standard solve
./build-release/kavach-opt solve benchmarks/netlib/afiro.mps

# Expected Output:
# Status: OPTIMAL
# Objective: -464.753
# Iterations: 16
# Verification: VERIFIED_OPTIMAL (Certified by LPVerifier)
```

Solve with JSON structured telemetry:
```bash
./build-release/kavach-opt solve benchmarks/netlib/afiro.mps --json
```

---

### 2.2. Mixed-Integer Linear Programming (MILP) Solve
Solve the knapsack integer problem using Branch-and-Bound with dual simplex warm starts:
```bash
./build-release/kavach-opt solve benchmarks/miplib/knapsack.lp

# Expected Output:
# Status: OPTIMAL
# Best Objective: 41
# Optimality Gap: 0
# Nodes Explored: 19
# Verification: VERIFIED_OPTIMAL (Certified by MILPVerifier)
```

---

### 2.3. Strictly Convex Quadratic Programming (QP) Solve
Solve the Markowitz portfolio optimization problem using the primal active-set solver:
```bash
./build-release/kavach-opt solve benchmarks/qp/portfolio.lp

# Expected Output:
# Status: OPTIMAL
# Objective: 0.316412
# Iterations: 5
# Verification: VERIFIED_OPTIMAL (Certified)
```

---

### 2.4. Independent Verification Certification
Run decoupled verification on an optimization model:
```bash
./build-release/kavach-opt verify benchmarks/netlib/afiro.mps

# Expected Output:
# Verification Outcome: CERTIFIED (PASS)
# Primal Residual:      < 1e-12
```

---

### 2.5. Presolve Analysis & Reduction Report
Inspect model reduction and singleton tightening before solving:
```bash
./build-release/kavach-opt presolve benchmarks/netlib/afiro.mps --report

# Expected Output:
# variables: 32 -> 32
# constraints: 27 -> 23 (-4)
# bound tightenings: 43
```

---

### 2.6. Rule-Based Strategy Planning & Explainability
Inspect rule firings and mathematical rationale:
```bash
# Concise strategy summary
./build-release/kavach-opt solve benchmarks/netlib/beaconfd.mps --show-strategy

# Comprehensive explanation of rules fired
./build-release/kavach-opt solve benchmarks/netlib/beaconfd.mps --explain-strategy
```

---

### 2.7. MRPL Industrial Case Study Demonstration
Execute the baseline refinery model and 4 what-if operational scenarios:
```bash
./build-release/kavach-opt mrpl --no-highs

# Expected Output:
# Model: baseline (21 vars, 22 constrs, 81 nnz)
# Status: OPTIMAL
# Net Profit: $5445.91 k/day
# Mass Balance: PASSED
# Artifacts generated in examples/mrpl/results/ and docs/MRPL_CASE_STUDY.md
```

---

### 2.8. Automated Benchmark Suites
Run the standardized Netlib LP benchmark suite with fixed seed:
```bash
./build-release/kavach-opt benchmark --suite netlib --seed 42 --output benchmark_results

# Check generated reports:
cat benchmark_results/benchmark_report.md
```

---

### 2.9. CPU-Only & Fallback Enforcement
Force sequential CPU execution regardless of hardware:
```bash
./build-release/kavach-opt solve benchmarks/netlib/afiro.mps --no-gpu --backend cpu
```

---

### 2.10. C++ Programmatic API Example
Compile and execute the standalone API demonstration binary:
```bash
./build-release/kavach_api_example
```

---

## 3. Environment & Seed Details

All tests and benchmarks capture execution provenance:
- **Compiler**: AppleClang (macOS) / GCC 13 (Ubuntu)
- **Standard**: C++20 (`-std=c++20`)
- **Default Seed**: `42`
- **Tolerances**: Feasibility $\epsilon_{\text{feas}} = 10^{-8}$, Optimality $\epsilon_{\text{opt}} = 10^{-8}$, Pivot $\epsilon_{\text{piv}} = 10^{-10}$
