#!/usr/bin/env python3
"""
tools/check_dependencies.py - Sovereign Dependency Graph and Forbidden Library Checker
Enforces Blueprint Section 1, 2, 5, 24, 30:
Guarantees zero optimization-engine libraries (CPLEX, Gurobi, HiGHS, GLPK, SCIP, CBC,
OR-Tools, cuOpt, etc.) are included, linked, or vendored in the solving path.
"""

import sys
from pathlib import Path

FORBIDDEN_IDENTIFIERS = [
    "gurobi",
    "cplex",
    "highs",
    "glpk",
    "scip",
    "clp",
    "cbc",
    "ortools",
    "cuopt",
    "xpress",
    "mosek",
    "ipopt",
]

# Files/extensions to inspect
SOURCE_EXTS = {".cpp", ".hpp", ".c", ".h", ".cmake"}

def check_file(path: Path) -> list[str]:
    violations = []
    # Ignore build directories or git folders
    for part in path.parts:
        if part in {"build", ".git", "CMakeFiles", "Testing"}:
            return []

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return [f"Could not read {path}: {e}"]

    for line_num, line in enumerate(content.splitlines(), start=1):
        line_lower = line.lower()
        # Look for #include or target_link_libraries containing forbidden libraries
        if "#include" in line_lower or "target_link_libraries" in line_lower or "find_package" in line_lower:
            for ident in FORBIDDEN_IDENTIFIERS:
                if ident in line_lower:
                    violations.append(
                        f"{path}:{line_num}: Forbidden optimization dependency '{ident}' in: {line.strip()}"
                    )
    return violations

def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    cmake_file = repo_root / "CMakeLists.txt"

    all_violations = []

    if cmake_file.exists():
        all_violations.extend(check_file(cmake_file))

    if src_dir.exists():
        for path in src_dir.rglob("*"):
            if path.is_file() and path.suffix in SOURCE_EXTS:
                all_violations.extend(check_file(path))

    if all_violations:
        print("ERROR: Forbidden optimization engine dependencies detected!")
        for v in all_violations:
            print(f"  {v}")
        return 1

    print("SUCCESS: Zero forbidden optimization dependencies detected. Sovereignty rule satisfied.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
