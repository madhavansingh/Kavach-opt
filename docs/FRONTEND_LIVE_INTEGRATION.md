# Kavach-Opt Live Frontend Integration Guide

## 1. Executive Summary & Architecture Overview

Kavach-Opt (कवच-Opt) is an indigenous, sovereign mathematical optimization solver core engineered in C++20 for SIH26119 and Mangalore Refinery and Petrochemicals Limited (MRPL).

The frontend layer is a demonstration and operational interface that connects directly to the authoritative C++ solver executable. It is designed under a **Zero-Fabrication Data Truth Policy**: no solve results, residuals, or iterations are mocked or simulated in production execution.

```
┌────────────────────────────────────────────────────────┐
│                   React + Vite SPA                     │
│         (Overview, Solve, Verification, System)        │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP / JSON (Fetch API)
                            ▼
┌────────────────────────────────────────────────────────┐
│             Local Node.js Adapter / Router             │
│   (server/solverAdapter.mjs & server/apiRouter.mjs)    │
│  - Path sanitization (benchmarks/ and examples/ only)   │
│  - Parameter formatting and validation                 │
└───────────────────────────┬────────────────────────────┘
                            │ execFile (Zero shell injection)
                            ▼
┌────────────────────────────────────────────────────────┐
│              ./build-release/kavach-opt                │
│   - Strategy Rules Engine (R001–R018)                  │
│   - Sparse Numerics (CSR/CSC, Markowitz SparseLU)      │
│   - Two-Phase Revised Simplex (LP)                     │
│   - Branch-and-Bound + Dual Simplex Warm Starts (MILP) │
│   - Active-Set Solver (Convex QP)                      │
│   - Air-gapped Independent Post-Solve Verifier         │
└────────────────────────────────────────────────────────┘
```

---

## 2. Security & Sovereignty Architecture

1. **No External Solver Dependencies**: The entire pipeline operates without foreign closed-source solvers (HiGHS, Gurobi, CPLEX, GLPK, SCIP).
2. **Safe Process Spawning**: Child execution utilizes `node:child_process.execFile` with an array of arguments, completely eliminating shell interpolation, command chaining, and shell injection vulnerabilities.
3. **Strict Path Sanitization**: The adapter checks all incoming file paths using path normalization and directory boundary constraints. Only files within `benchmarks/` and `examples/` are allowed; directory traversal attempts (`../`) are immediately rejected.
4. **Offline First**: All dependencies are vendored or bundled locally; no telemetry or data leaves the local machine.

---

## 3. Local REST API Specification

The local adapter exposes the following endpoints (via Vite development middleware or standalone server on port 3001):

### `GET /api/health`
Returns the operational status of the solver binary and host environment.
- **Response**:
  ```json
  {
    "status": "ok",
    "solver_binary": "/Users/.../build-release/kavach-opt",
    "binary_present": true,
    "cuda_supported_build": false,
    "cuda_device_present": false,
    "version_info": "kavach_opt version 0.1.0\nBuild Type: Release...",
    "server_time": "2026-09-26T15:20:00.000Z"
  }
  ```

### `GET /api/models`
Discovers all real benchmark models and industrial case study instances in the repository.
- **Discovered Categories**:
  - Netlib Linear Programming (`benchmarks/netlib/*.mps`)
  - MIPLIB Mixed-Integer Linear Programming (`benchmarks/miplib/*.lp`)
  - Convex Quadratic Programming (`benchmarks/qp/*.lp`)
  - MRPL Refinery Industrial Blending (`examples/mrpl/*.lp`)

### `POST /api/solve`
Executes `./build-release/kavach-opt solve <path> --json [options]`.
- **Request Body**:
  ```json
  {
    "model_path": "benchmarks/netlib/afiro.mps",
    "options": {
      "backend": "CPU_SEQUENTIAL",
      "presolve_policy": "FULL",
      "scaling_policy": "AUTO",
      "time_limit_seconds": 60
    }
  }
  ```
- **Response Structure**:
  - `result`: Status (`OPTIMAL`, `INFEASIBLE`, etc.), objective value, solve time, iterations, nodes, backend used, verification boolean.
  - `strategy`: Rules fired (`R001`, `R003`, etc.), human readable rationale, rule explanations, decision time.
  - `verification`: Recomputed primal residual, dual residual, complementary slackness, integrality violation, KKT residual, certificate summary.
  - `model`: Dimensions, density, problem class.

### `GET /api/benchmarks`
Serves authentic recorded benchmark results from `benchmark_results/` across Netlib, MIPLIB, and QP suites.

### `GET /api/mrpl`
Serves authentic baseline economics and operational scenario evaluations from `examples/mrpl/results/`.

---

## 4. Hardware Backend & Fallback Behavior

Kavach-Opt adheres strictly to its CPU-first architecture:
- When a user requests `CUDA_GPU`, the CLI evaluates availability.
- On platforms lacking CUDA hardware runtimes (e.g. macOS Apple Silicon), Strategy Rule R012 enforces deterministic CPU execution.
- The UI transparently presents both the requested backend and the executed backend with an explicit fallback explanation notice.

---

## 5. Automated Verification & Testing

The frontend integration is continuously verified through an automated integration suite (`frontend/test/integration.test.mjs`):
- Run command: `npm test` inside `frontend/`
- Validates model discovery (19 models across LP, MILP, QP, MRPL).
- Validates authentic solve runs on `afiro.mps`, `knapsack.lp`, `portfolio.lp`, and `infeasible_milp.lp`.
- Validates path traversal rejections.
- Validates benchmark artifact integrity.
- Validates MRPL refinery scenario evaluations and mandatory synthetic data disclosures.
