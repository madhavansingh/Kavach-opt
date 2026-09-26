# MRPL Industrial Demonstration Case Study

## Data Status Disclosure

> [!IMPORTANT]
> **DATA STATUS:**
> All numeric values in this demonstration are illustrative/synthetic.
> They are not proprietary MRPL operational data and must not be interpreted
> as actual MRPL process parameters.

## Overview

This case study demonstrates the capability of the **Kavach-Opt** indigenous optimization solver core to formulate, solve, independently verify, and operationally interpret a real-world refinery crude blending and unit capacity allocation problem.

### Mathematical Formulation
- **Type**: Mixed-Integer Linear Program (MILP)
- **Dimensions**: 21 variables (18 continuous, 3 binary), 22 linear constraints, 81 matrix nonzeros
- **Variables**:
  - Blending flows: $x[c, p] \ge 0$ for crudes $c \in \{A, B, C, D\}$ to products $p \in \{P1, P2, P3\}$
  - Product yields: $prod[p] \ge 0$
  - Unit throughputs: $t[u] \ge 0$ for units $u \in \{U1, U2, U3\}$
  - Unit binary activation: $y[u] \in \{0, 1\}$
- **Constraints**:
  - Exact product pool mass balances
  - Crude stream availability limits
  - Unit throughput accounting & minimum turndown limits
  - Linearized product quality specifications (sulfur maximums and density limits)
- **Objective**: Maximize net refining margin ($/day)

## Running the Case Study

### 1. Solve the Model Directly
```bash
./build-release/kavach-opt solve examples/mrpl/mrpl_blending_capacity.lp
```

### 2. Run Complete Scenario Analysis & Generate Artifacts
```bash
./build-release/kavach-opt mrpl --output examples/mrpl/results --report docs/MRPL_CASE_STUDY.md
```

### 3. Run Through Benchmark Suite
```bash
./build-release/kavach-opt benchmark --manifest benchmarks/manifests/mrpl.json --output benchmark_results/mrpl
```

## Generated Artifacts
- `results/baseline.json`: Machine-readable per-variable solution, domain mass balance, and economic breakdown.
- `results/scenarios.json`: Results across all what-if scenarios (reduced crude, tight quality, reduced capacity, infeasible).
- `results/comparison.json`: Side-by-side comparison against external reference solvers (HiGHS).
- `docs/MRPL_CASE_STUDY.md`: Complete human-readable technical report.
