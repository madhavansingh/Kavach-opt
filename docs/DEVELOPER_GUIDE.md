# Kavach-Opt — Developer Guide (Phase 0 Foundation)

Welcome to the **Kavach-Opt** optimization engine engineering repository. This guide covers how to build, test, lint, format, and audit the repository locally.

---

## 1. Prerequisites

- **C++ Compiler:** C++20 compliant compiler (Apple Clang 14+, Clang 14+, or GCC 12+)
- **Build System:** CMake 3.20+ and Ninja
- **Linters & Formatters:** `clang-format` (15+) and `clang-tidy` (15+)
- **Python:** Python 3.8+ (for dependency and benchmark tooling)

---

## 2. Building

### Debug Build (with strict warnings-as-errors)

```bash
cmake -B build-debug -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DKAVACH_ENABLE_WARNINGS_AS_ERRORS=ON \
    -DKAVACH_BUILD_TESTS=ON \
    -DKAVACH_BUILD_CLI=ON

cmake --build build-debug
```

### Release Build

```bash
cmake -B build-release -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DKAVACH_ENABLE_WARNINGS_AS_ERRORS=ON \
    -DKAVACH_BUILD_TESTS=ON \
    -DKAVACH_BUILD_CLI=ON

cmake --build build-release
```

---

## 3. Running Tests

Run the full CTest test suite from the build directory:

```bash
ctest --test-dir build-debug --output-on-failure
# Or directly run the binary:
./build-debug/tests/kavach_unit_tests
```

---

## 4. Building & Testing with Sanitizers (ASan + UBSan)

AddressSanitizer and UndefinedBehaviorSanitizer are configured to catch memory bugs and undefined behaviors early:

```bash
cmake -B build-asan -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DKAVACH_ENABLE_SANITIZERS=ON \
    -DKAVACH_BUILD_TESTS=ON

cmake --build build-asan
ctest --test-dir build-asan --output-on-failure
```

---

## 5. Code Formatting

Check formatting without modifying files:
```bash
find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format --dry-run --Werror
```

Format all files in-place:
```bash
find src tests -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i
```

---

## 6. Static Analysis (clang-tidy)

Ensure compile commands are generated (`compile_commands.json` is exported automatically):

```bash
clang-tidy -p build-debug src/core/logger.cpp src/core/random.cpp src/model/model.cpp src/cli/main.cpp
```

---

## 7. Dependency Sovereignty Audit

Per Blueprint Section 1, 2, 5, 24, and 30, no third-party optimization engine library may enter the solving path. Check this invariant:

```bash
python3 tools/check_dependencies.py
```

---

## 8. Repository Layout

```
kavach-opt/
├── src/
│   ├── core/           # Status taxonomy, tolerances, errors, deterministic RNG, logger
│   ├── model/          # [A] Canonical model representation (Phase 1)
│   ├── io/             # [B] MPS / LP parsers (Phase 2)
│   ├── numerics/       # [D][E] Sparse matrix (CSR/CSC), sparse LU factorization (Phase 3)
│   ├── presolve/       # [C] Presolve reductions & restoration (Phase 4)
│   ├── algorithms/
│   │   ├── lp/         # [F] Revised simplex LP solver (Phase 5)
│   │   ├── milp/       # [G][I]-[P] Branch-and-bound MILP solver (Phase 7)
│   │   └── qp/         # [H] Convex active-set QP solver (Phase 9)
│   ├── verification/   # [Y] Independent verification layer (Phase 6)
│   ├── strategy/       # [T] Rule-based solver strategy layer (Phase 14)
│   ├── backend/
│   │   ├── cpu/        # [S] Multicore CPU batched SpMV (Phase 10)
│   │   └── cuda/       # [U] Scoped CUDA batched SpMV (Phase 11)
│   ├── api/            # [V] C++ Solver API (Phase 15)
│   └── cli/            # [W] CLI entry point (Phase 15)
├── benchmarks/         # [X] Benchmark harness (Netlib, MIPLIB)
├── tests/              # Unit, adversarial, regression, fault-injection tests
├── examples/           # [AB] MRPL demo models
├── configs/            # Version-tracked tolerance configs
├── scripts/            # Reproducibility scripts
├── tools/              # Dependency checker & audit tools
└── docs/               # Architecture and engineering documentation
```
