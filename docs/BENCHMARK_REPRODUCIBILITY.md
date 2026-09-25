# Kavach-Opt Benchmark Reproducibility Guide

This guide provides step-by-step instructions for reproducing all benchmark results, generating machine-readable artifacts, and conducting independent external comparisons.

---

## 1. Prerequisites and Build Modes

Kavach-Opt is implemented in modern C++20 with **zero external optimization engine or numerical dependencies**. A standard C++20 compiler (`clang++` $\ge 14$ or `g++` $\ge 11$) and CMake ($\ge 3.20$) are required.

### 1.1 Release Build (Optimized Benchmarking)
```bash
# Configure Release build
cmake -B build-release -DCMAKE_BUILD_TYPE=Release

# Compile core library, CLI, and unit test suite
cmake --build build-release -j
```

### 1.2 Debug Build (Traceability & Diagnostics)
```bash
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug -j
```

### 1.3 AddressSanitizer & UndefinedBehaviorSanitizer (Memory Safety)
```bash
cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug -DKAVACH_ENABLE_SANITIZERS=ON
cmake --build build-asan -j
```

---

## 2. Running Benchmark Suites via CLI

The Kavach-Opt executable provides the `benchmark` subcommand:

### 2.1 Netlib LP Suite
```bash
./build-release/kavach-opt benchmark --suite netlib --output benchmark_results/netlib/
```

### 2.2 MIPLIB Subset
```bash
./build-release/kavach-opt benchmark --suite miplib_subset --output benchmark_results/miplib/
```

### 2.3 Convex QP Suite
```bash
./build-release/kavach-opt benchmark --suite qp --output benchmark_results/qp/
```

### 2.4 External Comparison against HiGHS
```bash
# If HiGHS is installed in PATH or specified via HIGHS_BIN:
./build-release/kavach-opt benchmark --suite netlib --compare highs --output benchmark_results/netlib_highs/
```
*Note: If HiGHS is not detected, the framework marks reference comparisons as `REFERENCE_UNAVAILABLE` without error.*

### 2.5 Single Instance Benchmarking
```bash
./build-release/kavach-opt benchmark --instance benchmarks/netlib/afiro.mps --output benchmark_results/single/
```

---

## 3. Command Line Options Reference

| Flag | Argument | Description | Default |
|---|---|---|---|
| `--suite` | `<name>` | Suite name matching `benchmarks/manifests/<name>.json` | `netlib` |
| `--manifest` | `<path>` | Explicit path to a benchmark manifest JSON | Auto-located |
| `--instance` | `<path>` | Path to a single problem file (.mps / .lp) | None |
| `--compare` | `highs` | Enable isolated external comparison | Disabled |
| `--output` | `<dir>` | Destination folder for JSON/CSV/MD outputs | `benchmark_results` |
| `--time-limit` | `<seconds>` | Per-instance wall-clock timeout override | Suite default |
| `--seed` | `<uint64>` | Random seed for deterministic reproducibility | `42` |
| `--filter` | `<pattern>` | Substring match on case ID or instance name | All cases |
| `--tag` | `<tag>` | Filter by problem category (e.g. `small`, `sparse`) | All tags |
| `--no-gpu` | None | Force CPU execution (disable GPU batched-SpMV) | CPU fallback active |
| `--verbose` | None | Enable verbose solver logging | Disabled |

---

## 4. Generated Artifact Structure

Every benchmark run produces three authoritative artifacts in the specified output directory:

```
benchmark_results/
├── benchmark_results.json    # Complete structured machine-readable records with full precision
├── benchmark_results.csv     # Tabular summary suitable for spreadsheet analysis
└── benchmark_report.md       # Full 10-section human-readable Markdown report
```

### 4.1 Reproducibility Manifest in JSON
Every `benchmark_results.json` includes an embedded hardware and git environment manifest:
```json
{
  "manifest": {
    "git_commit": "0000000",
    "timestamp": "2026-09-25T20:00:00Z",
    "hostname": "HostName",
    "cpu_model": "arm64 (8 logical cores)",
    "gpu_device": "CUDA unavailable (CPU fallback)",
    "os_name": "Darwin 27.0.0 (arm64)",
    "compiler_info": "AppleClang 21.0.0.21000334",
    "build_type": "Release",
    "kavach_version": "0.1.0",
    "suite_name": "netlib",
    "seed": 42
  },
  "results": [ ... ]
}
```

---

## 5. Deterministic Rerun Verification

To confirm deterministic reproducibility across executions:
1. Run the same case twice with identical random seed (`--seed 42`).
2. Verify that solver status, iteration count, and objective values match identically within numerical tolerances ($10^{-10}$).
3. This invariant is continuously checked by unit test `test_l_deterministic_rerun` in `tests/unit/test_phase12_benchmark.cpp`.
