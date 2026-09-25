# Kavach-Opt Parser Subsystem [B] Specification

## Overview

The `src/io` parser subsystem is responsible for translating external optimization model representations (`.mps` and `.lp` formats) into the sovereign in-memory `CanonicalModel` representation (`src/model/`).

This subsystem operates with **zero external optimization engine dependencies** (e.g. no Coin-OR, HiGHS, GLPK, Gurobi, CPLEX, or SCIP), strictly adhering to the Sovereign Indigenous mandate defined in `SIH26119_Kavach-Opt_Blueprint_v2.md`.

---

## 1. Supported Formats & Syntax

### 1.1 Mathematical Programming System (MPS) Format
Kavach-Opt supports both fixed-column and free-format MPS files, handling all industry-standard sections:
- `NAME`: Model identifier.
- `OBJSENSE` (optional extension): `MIN` or `MAX` objective sense.
- `OBJNAME` (optional extension): Named objective row identifier.
- `ROWS`: Row definitions with sense indicators:
  - `N`: Non-constrained row. The primary `N` row (either matching `options.objective_name` or the first `N` row encountered) is assigned as the model objective function. Additional `N` rows are classified as free rows and are excluded from the canonical constraint system, preventing downstream corruption.
  - `L`: Less-than-or-equal inequality ($a^T x \le b$).
  - `G`: Greater-than-or-equal inequality ($a^T x \ge b$).
  - `E`: Strict equality ($a^T x = b$).
- `COLUMNS`: Variable entries and matrix nonzeros.
  - Supports standard integer markers:
    `MARK0000  'MARKER'                 'INTORG'`
    `MARK0001  'MARKER'                 'INTEND'`
    Variables declared within an integer block are tagged as `VariableType::INTEGER`.
- `RHS`: Right-hand side constants.
  - If a right-hand side constant is defined for the objective row, it is normalized into the canonical objective offset ($-b_0$).
  - Supports multiple RHS vectors, selecting the primary active vector.
- `RANGES`: Two-sided constraint intervals.
  - Mathematically converted into dual canonical constraints ($a^T x \le b_u$ and $a^T x \ge b_l$), preserving the exact feasible region without artificial slack variables:
    - Row sense `G`: $b \le a^T x \le b + |r|$ $\implies$ original $a^T x \ge b$ and dual $a^T x \le b + |r|$ (`name_range_ub`).
    - Row sense `L`: $b - |r| \le a^T x \le b$ $\implies$ original $a^T x \le b$ and dual $a^T x \ge b - |r|$ (`name_range_lb`).
    - Row sense `E` ($r > 0$): $b \le a^T x \le b + r$ $\implies$ dual $a^T x \ge b$ (`name_range_lb`) and dual $a^T x \le b + r$ (`name_range_ub`).
    - Row sense `E` ($r < 0$): $b + r \le a^T x \le b$ $\implies$ dual $a^T x \ge b - |r|$ (`name_range_lb`) and dual $a^T x \le b$ (`name_range_ub`).
- `BOUNDS`: Variable upper, lower, fixed, binary, and integer bounds:
  - `LO`: Lower bound $x_j \ge v$.
  - `UP`: Upper bound $x_j \le v$ (if $v \le 0$ and no `LO` is set, lower bound is normalized to $-\infty$).
  - `FX`: Fixed variable $x_j = v$.
  - `FR`: Free variable $-\infty < x_j < +\infty$.
  - `MI`: Negative infinite lower bound $-\infty$.
  - `PL`: Positive infinite upper bound $+\infty$.
  - `BV`: Binary variable $x_j \in \{0, 1\}$.
  - `LI`: Integer lower bound $x_j \in \mathbb{Z}, x_j \ge v$.
  - `UI`: Integer upper bound $x_j \in \mathbb{Z}, x_j \le v$.
- `ENDATA`: Terminates the model file.

### 1.2 Native LP Format
Designed for human readability and authoring (e.g. MRPL industrial demo models):
- **Objective Sense**: Case-insensitive `Minimize`, `Minimum`, `Min`, `Maximize`, `Maximum`, `Max`.
- **Quadratic Objectives**: Full support for CPLEX-style bracketed quadratic expressions:
  `[ 2 x1^2 + 4 x1 * x2 ] / 2` or `[ 2 x1 * x2 ]`.
- **Constraints**: Optional labels (e.g. `c1:`), supporting single-sided (`<=`, `>=`, `=`) and ranged constraints (`10 <= 2 x1 + 3 x2 <= 50`).
- **Bounds**: Single and two-sided bounds (`0 <= x <= 10`, `x free`, `x = 5.0`).
- **Integrality**: Explicit `Generals` / `Integers` and `Binaries` sections.
- **Comments**: Full-line or inline comments prefixed by `\`.
- **End**: Clean termination with `End`.

---

## 2. Error Handling & Adversarial Rejection

All parsing errors raise a strongly-typed `kavach::io::ParserException` inheriting from `kavach::core::InputException`.
Every exception captures:
1. `filename()`: Path or buffer name.
2. `line()`: 1-indexed source line number.
3. `section()`: Active parser section (e.g. `ROWS`, `COLUMNS`, `BOUNDS`, `CONSTRAINTS`).
4. `raw_message()`: Exact human-readable diagnostic.

The parser strictly rejects:
- Unrecognized section headers.
- Duplicate row or variable declarations.
- Columns referencing undeclared rows.
- Bounds referencing undeclared variables.
- Non-finite (NaN or infinite) coefficients in matrix entries.
- Unclosed brackets or missing operators in LP expressions.

---

## 3. Verified Netlib Benchmark Corpus

The parser subsystem has been verified against 13 authentic Netlib instances downloaded directly from `netlib.org/lp/data/`:

| Benchmark | Variables | Constraints | Nonzeros | Verified Characteristics |
|---|---|---|---|---|
| `afiro.mps` | 32 | 27 | 88 | Fixed MPS, spot-checked matrix & RHS entries |
| `adlittle.mps` | 97 | 56 | 465 | Exact dimension & coefficient integrity |
| `blend.mps` | 83 | 74 | 521 | Unlabelled RHS vector handling |
| `sc50a.mps` | 48 | 50 | 160 | Fixed MPS dimension validation |
| `sc50b.mps` | 48 | 50 | 148 | Fixed MPS dimension validation |
| `share2b.mps` | 79 | 96 | 730 | Exact dimension & coefficient integrity |
| `beaconfd.mps` | 262 | 173 | 3476 | Scaled LP problem |
| `sc105.mps` | 103 | 105 | 340 | Fixed MPS dimension validation |
| `lotfi.mps` | 308 | 153 | 1086 | Medium-scale LP problem |
| `share1b.mps` | 225 | 117 | 1182 | Medium-scale LP problem |
| `kb2.mps` | 41 | 43 | 291 | Empty RHS section & BOUNDS verification |
| `stocfor1.mps` | 111 | 117 | 474 | Stochastic LP problem |
| `boeing2.mps` | 143 | 166 (orig) + duals | 1339 | RANGES dual constraint expansion & BOUNDS |
