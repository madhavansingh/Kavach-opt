# Kavach-Opt Final Presentation & Demo Guide

**Project**: Kavach-Opt Sovereign Indigenous Optimization Solver Core (SIH26119)  
**Target Audience**: Smart India Hackathon Judges & MRPL Industrial Panel  
**Execution Environment**: macOS / Linux (Clean Checkout)  

---

## 1. Demo Overview & Ground Rules

This script outlines a 10-to-15 minute technical presentation and live terminal demonstration. Every command can be executed live from the terminal in a clean checkout.

| Demo Stage | Mode | Focus / Verification Point |
|---|:---:|---|
| **Steps 1–10** | **LIVE** | Fully executable in real-time on any standard laptop or workstation. |
| **Step 11** | **ARCHITECTURAL / PRE-RECORDED** | GPU Crossover benchmarks if host lacks NVIDIA GPU (e.g. Apple Silicon). |
| **Steps 12–14** | **LIVE** | Live fallback demonstration, benchmark runner, and limitations discussion. |

---

## 2. Step-by-Step Presentation Script

### Step 1: System Architecture Walkthrough [LIVE]
**Narrative**: *"Kavach-Opt is an indigenous, sovereign mathematical optimization engine written in C++20 from mathematical first principles. It comprises an ingest pipeline, canonical representation, sparse LU numerical core, Revised Simplex, Branch-and-Bound, Active-Set QP, and an independent verification layer."*

**Action**: Display root architecture diagram:
```bash
cat << 'EOF'
  Input File (.mps / .lp)
            ↓
  Parser & CanonicalModel
            ↓
  Characterization & Strategy Engine (Rules R001-R018)
            ↓
  Presolve Pipeline (Fixed vars, singleton tightening, redundancy removal)
            ↓
  Numerical Core (CSR/CSC, Sparse LU, Ruiz Scaling, Iterative Refinement)
            ↓
  Solvers (Revised Simplex LP | Branch-and-Bound MILP | Active-Set QP)
            ↓
  Independent Verification Layer (LPVerifier | MILPVerifier | QPVerifier)
            ↓
  Certified Optimal Solution (Status, Primal/Dual, JSON Telemetry)
EOF
```

---

### Step 2: Source & Dependency Sovereignty Audit [LIVE]
**Narrative**: *"To ensure complete sovereign independence, Kavach-Opt contains zero foreign optimization solver dependencies. We audit our entire source tree against CPLEX, Gurobi, HiGHS, GLPK, CBC, and SCIP."*

**Command**:
```bash
python3 tools/check_dependencies.py
```
**Expected Output**:
```text
SUCCESS: Zero forbidden optimization dependencies detected. Sovereignty rule satisfied.
```

---

### Step 3: Solve Standard Netlib Linear Program [LIVE]
**Narrative**: *"We solve the canonical Netlib LP benchmark instance 'afiro' using our Two-Phase Revised Simplex solver with standard presolve."*

**Command**:
```bash
./build-release/kavach-opt solve benchmarks/netlib/afiro.mps
```
**Expected Output**:
```text
Loaded model 'AFIRO': 32 vars, 27 constrs, 83 nonzeros
Solving LP with Revised Simplex...
Status: OPTIMAL
Objective: -464.753
Iterations: 16
Verification: VERIFIED_OPTIMAL (Certified by LPVerifier)
```

---

### Step 4: Independent Verification Certification [LIVE]
**Narrative**: *"In Kavach-Opt, we never trust the solver's self-reported status. Our independent verification layer independently recomputes primal feasibility, dual feasibility, and complementary slackness residuals."*

**Command**:
```bash
./build-release/kavach-opt verify benchmarks/netlib/afiro.mps
```
**Expected Output**:
```text
=== Kavach-Opt Independent Verification Certificate ===
Model:                AFIRO
Solver Status:        OPTIMAL
Objective Value:      -464.75314
Verification Outcome: CERTIFIED (PASS)
Primal Residual:      < 1e-12
```

---

### Step 5: Fault Injection & Verification Rejection [LIVE]
**Narrative**: *"To prove our verification layer is not a rubber stamp, our test suite includes deliberate fault injection tests that corrupt primal solutions, bounds, and KKT residuals. The verifier detects and rejects these corruptions 100% of the time."*

**Command**:
```bash
./build-release/tests/kavach_unit_tests --filter test_lp_fault
```
**Expected Output**:
```text
[PASS] test_lp_fault_primal_infeasibility_detected
[PASS] test_lp_fault_bound_violation_detected
[PASS] test_lp_fault_dual_infeasibility_detected
[PASS] test_lp_fault_complementary_slackness_detected
[PASS] test_lp_fault_objective_mismatch_detected
```

---

### Step 6: Mixed-Integer Linear Programming (MILP) Solve [LIVE]
**Narrative**: *"We solve discrete optimization problems using our Branch-and-Bound engine with most-fractional branching and dual simplex warm starts."*

**Command**:
```bash
./build-release/kavach-opt solve benchmarks/miplib/knapsack.lp
```
**Expected Output**:
```text
Loaded model '': 6 vars, 1 constrs, 6 nonzeros
Solving MILP with Branch-and-Bound...
Status: OPTIMAL
Best Objective: 41
Optimality Gap: 0
Nodes Explored: 19
Verification: VERIFIED_OPTIMAL (Certified by MILPVerifier)
```

---

### Step 7: Strictly Convex Quadratic Programming (QP) Solve [LIVE]
**Narrative**: *"For quadratic objectives such as portfolio risk minimization, we use our primal active-set QP solver with positive semi-definite (PSD) Hessian validation."*

**Command**:
```bash
./build-release/kavach-opt solve benchmarks/qp/portfolio.lp
```
**Expected Output**:
```text
Loaded model '': 3 vars, 1 constrs, 3 nonzeros
Solving Convex QP...
Status: OPTIMAL
Objective: 0.316412
Iterations: 5
Verification: VERIFIED_OPTIMAL (Certified)
```

---

### Step 8: MRPL Industrial Case Study Baseline [LIVE]
**Narrative**: *"We now demonstrate our industrial application: an MRPL-style refinery crude distillation, unit capacity, and product quality blending MILP model. All data is illustrative and synthetic."*

**Command**:
```bash
./build-release/kavach-opt mrpl --no-highs
```
**Expected Output**:
```text
BASELINE INDUSTRIAL SOLVE RESULTS
Model:            baseline (21 vars, 22 constrs, 81 nnz)
Status:           OPTIMAL
Net Profit:       $5445.91 k/day
LP Bound:         $5891.36 k/day
Optimality Gap:   0.00%
Mass Balance:     Max residual = 0.0000 (PASSED)
```

---

### Step 9: MRPL What-If Scenario Analysis [LIVE]
**Narrative**: *"The case study runner evaluates 4 what-if operational scenarios, including crude supply disruptions, tighter sulfur specifications, and unit outages."*

**Command**:
```bash
cat examples/mrpl/results/scenarios.json | head -n 35
```
**Expected Output**: Clean JSON scenario comparisons displaying changes in refinery throughput and gross margins.

---

### Step 10: Rule-Based Strategy Layer & Explainability [LIVE]
**Narrative**: *"Kavach-Opt contains an autonomous strategy engine. It characterizes matrix dimensions, sparsity, dynamic range, and numerical risk to fire deterministic rules."*

**Command**:
```bash
./build-release/kavach-opt solve benchmarks/netlib/beaconfd.mps --explain-strategy
```
**Expected Output**:
```text
=== Kavach-Opt Strategy Plan ===
Problem Class:         CONTINUOUS_LP
Dimensions:            173 rows, 262 cols, 3375 nonzeros (density: 0.0745)
Dynamic Range:         2.000000e+04 (min: 5.000000e-02, max: 1.000000e+03)
Numerical Risk:        Elevated
Algorithm:             REVISED_SIMPLEX
Presolve Policy:       ENABLED
Scaling Policy:        ENABLED
Backend:               CPU_SEQUENTIAL
Refactor Interval:     20
Fired Rules:           R003, R004, R006, R008, R010
Rationale:
  - [R003] Continuous linear program detected; routed to revised simplex solver.
  - [R004] Standard presolve enabled for linear problem.
  - [R006] Wide dynamic range detected; equilibration scaling enabled.
  - [R008] Elevated numerical risk: refactorization interval set to 20.
  - [R010] Small or non-batched workload; routed to sequential CPU backend.
```

---

### Step 11: GPU Acceleration & Honest Crossover [PRE-RECORDED / ARCHITECTURAL]
**Narrative**: *"GPU acceleration in Kavach-Opt is strictly confined to batched sparse matrix-vector multiplication (SpMV). We measure transfer-inclusive time (H2D + Kernel + D2H) and assert speedup only where GPU genuinely outperforms the CPU multicore pool (typically batch size $\ge 128$ and $nnz \ge 10,000$)."*

**Reference Document**: Display `docs/GPU_CLAIM_AUDIT.md`.

---

### Step 12: Robust CPU Fallback Demonstration [LIVE]
**Narrative**: *"If a system lacks a GPU or the user requests CPU execution via `--no-gpu`, Kavach-Opt gracefully and automatically falls back to CPU multicore with zero errors."*

**Command**:
```bash
./build-release/kavach-opt solve benchmarks/netlib/afiro.mps --backend cuda --no-gpu
```
**Expected Output**: Resolves seamlessly to CPU execution and returns status `OPTIMAL`.

---

### Step 13: C++ Programmatic API Demonstration [LIVE]
**Narrative**: *"Developers can integrate Kavach-Opt directly into C++ applications using our clean `api::KavachSolver` interface."*

**Command**:
```bash
./build-release/kavach_api_example
```
**Expected Output**:
```text
1. Model Formulated: production_planning_demo (2 vars, 3 constrs)
2. Rule-Based Strategy Plan: Class: LP, Presolve: FULL, Scaling: AUTO
3. Solve & Certification Result: Status: OPTIMAL, Objective: 173.3333, Verification: CERTIFIED (PASS)
```

---

### Step 14: Scope, Limitations & Future Horizons [LIVE]
**Narrative**: *"We conclude with scientific transparency. Kavach-Opt is built for Linear, Mixed-Integer Linear, and Convex Quadratic Optimization. We explicitly do NOT claim support for Non-Convex QP, MIQP, or general NLP. Our focus is rock-solid correctness, mathematical sovereignty, and verifiable industrial utility."*

**Reference Document**: Display `docs/SCOPE_AND_LIMITATIONS.md`.
