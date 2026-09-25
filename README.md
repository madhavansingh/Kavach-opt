# Kavach-Opt (कवच-Opt)

## Indigenous GPU-Accelerated Optimization Solver Core
### Smart India Hackathon (SIH26119) · Mangalore Refinery and Petrochemicals Limited (MRPL)

---

## Overview

**Kavach-Opt** is a from-scratch, sparse, numerically self-verifying LP/MILP/QP optimization solver core built from mathematical first principles. It provides an indigenous, sovereign alternative to foreign proprietary commercial solvers (CPLEX, Gurobi, Xpress) for industrial planning, crude blending, and refinery operations.

### Key Pillars
1. **Sovereign & From-Scratch:** Zero external optimization-engine libraries (CPLEX, Gurobi, HiGHS, GLPK, CBC, SCIP) in the solving path.
2. **Independent Verification Layer [Y]:** Independent recomputation of feasibility, optimality, and KKT residuals—never trusting the solver's self-reported status.
3. **Sparse & Scalable:** CSR/CSC sparse representation and sparse LU factorization with fill-in reduction and iterative refinement.
4. **Honest GPU Acceleration:** One scoped CUDA kernel (batched SpMV) benchmarked with transfer-inclusive timing and an honest crossover point; robust CPU fallback always available.
5. **Reproducibility:** Deterministic RNG, configuration logging, and command-line reproduction for every experiment.

---

## Current Status: Phase 1 Complete

- **Phase 0 (Repository Foundation & Engineering Infrastructure):** **COMPLETE**
- **Phase 1 (Canonical Model Representation [A]):** **COMPLETE**
- **Phase 2 (MPS / Native LP Parsing [B]):** **PENDING (Awaiting explicit instruction)**

---

## Quick Start

### Build Requirements
- C++20 compiler (`clang++` 14+ or `g++` 12+)
- CMake 3.20+ and Ninja
- Python 3.8+

### Building

```bash
# Debug build with strict warnings-as-errors
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Run tests
ctest --test-dir build --output-on-failure
```

### CLI Verification
```bash
./build/kavach-opt --version
./build/kavach-opt --help
```

For complete documentation on building, testing, linting, sanitizers, and architecture, see [Developer Guide](docs/DEVELOPER_GUIDE.md).
