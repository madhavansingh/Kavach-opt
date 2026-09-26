# Kavach-Opt Frontend Design System

## 1. Executive Summary & Design Philosophy

Kavach-Opt (कवच-Opt) is an indigenous mathematical optimization solver engineered for mission-critical industrial scheduling, refinery planning, and sovereign operational self-reliance under SIH26119.

The Kavach-Opt frontend is designed as a **precision industrial optimization interface**—clean, authoritative, spacious, and scientifically restrained. It rejects the clichés of consumer software (glowing blobs, neon gradients, floating cards, neuromorphic skeuomorphism) in favor of the aesthetic principles of high-reliability control rooms, flight telemetry monitors, and aerospace instrumentation consoles.

### Core Visual Principles
1. **Calm Mathematical Confidence**: Dense information presented with generous padding, clear hierarchy, and quiet typography. No distracting animations or visual noise.
2. **Strict Typographic Separation**: Proportional UI font (Inter) for labels, headings, and explanations; fixed-width font (JetBrains Mono) strictly for numbers, variables, residuals, mathematical tolerances, and status tags.
3. **Layered Dark Palette**: Depth is achieved through restrained surface elevation (`#0b0d11` to `#272f3d`), subtle sub-pixel borders (`#1e2430`), and muted fills, never through drop shadows or bright neon accents.
4. **Honest States & Zero Fabrications**: If no optimization has run, the interface displays an unadorned, explicit empty state ("Awaiting Execution"). It never shows fabricated live charts or synthetic progress when disconnected from execution.

---

## 2. Color System & Semantic Tokens

All colors are configured in `frontend/src/styles/tokens.ts` and styled via Tailwind CSS tokens.

### 2.1 Neutral Surface Hierarchy
| Token Name | Hex Code | Usage |
| :--- | :--- | :--- |
| `surface-canvas` | `#0b0d11` | Primary application canvas background |
| `surface-base` | `#0d1016` | Main content background, top bar, sidebar background |
| `surface-panel` | `#11141a` | Standard panel cards, metric blocks, container surfaces |
| `surface-raised` | `#161a22` | Hover states, active tabs, nested sub-panels, input fields |
| `surface-overlay`| `#1b202a` | Modals, floating dropdowns, popovers |
| `border-subtle` | `#1e2430` | Default panel and card boundaries (1px) |
| `border-strong` | `#272f3d` | Emphasized dividers, active input borders, selected cards |

### 2.2 Semantic Status & Verification Palette
| Semantic Role | Accent Color | Subtle Background | Border Token | Usage |
| :--- | :--- | :--- | :--- | :--- |
| **Optimal / Verified** | `#10b981` (Emerald) | `rgba(6, 78, 59, 0.25)` | `rgba(16, 185, 129, 0.4)` | Proven optimality, verification pass, zero residuals |
| **Feasible / Active** | `#38bdf8` (Sky) | `rgba(7, 89, 133, 0.25)` | `rgba(56, 189, 248, 0.4)` | Feasible solution, running solve, CPU multicore backend |
| **Warning / Derated** | `#f59e0b` (Amber) | `rgba(120, 53, 15, 0.25)` | `rgba(245, 158, 11, 0.4)` | Ill-conditioned matrix, capacity derating, synthetic notice |
| **Infeasible / Error** | `#f43f5e` (Rose) | `rgba(136, 19, 55, 0.25)` | `rgba(244, 63, 94, 0.4)` | Proven infeasibility, verification failure, bounds violation |
| **Idle / Unverified** | `#94a3b8` (Slate) | `rgba(30, 41, 59, 0.3)` | `rgba(148, 163, 184, 0.3)` | Standby units, pending verification, default metadata |

---

## 3. Typography & Numerical Representation

### 3.1 Font Families
- **Interface Font**: `Inter, system-ui, sans-serif`
  - Used for titles, page descriptions, table headers, button labels, and explanatory prose.
- **Numerical & Code Font**: `JetBrains Mono, ui-monospace, SFMono-Regular, monospace`
  - Used for all numerical values, objectives, residuals, iterations, wall times, variable names, and matrix dimensions.

### 3.2 Numerical Precision Rules
1. **Objective Values**: Displayed to 4 or 5 decimal places (`-464.75314`), right-aligned in tables.
2. **Residuals & Violations**: Always formatted in scientific exponential notation with 3-4 significant digits (`1.776e-14`, `0.000e+00`).
3. **Execution Times**: Displayed in milliseconds (`0.35 ms`) with microsecond resolution for sub-millisecond solves (`0.000349 s`).
4. **Mass Balances**: Formatted to 6 decimal places to prove exact conservation (`0.000000`).

---

## 4. Component Library Architecture

### 4.1 Surface & Layout Containers
- **`AppShell`**: Responsive container managing the persistent sidebar (desktop fixed, mobile drawer), top telemetry bar, and scrollable content canvas.
- **`Panel`**: Primary container component with crisp 1px borders, subtle title headers, optional status badges, and optional header actions.
- **`PageHeader`**: Standardized page title header with descriptive subtitle and status badge chip.

### 4.2 Telemetry & Metrics
- **`Metric`**: Displays a single Key Performance Indicator with uppercase label, prominent numerical value, optional subtext, and semantic color highlight.
- **`MetricGroup`**: Responsive grid container (2, 3, 4, or 6 columns) managing consistent spacing across metric sets.
- **`StatusBadge`**: Pill-shaped status indicator rendering authoritative solver and verification states (`OPTIMAL`, `FEASIBLE`, `INFEASIBLE`, `VERIFIED_OPTIMAL`, `VERIFICATION_FAILED`).
- **`BackendIndicator`**: Subtle hardware indicator badge displaying `CPU_SEQUENTIAL`, `CPU_MULTICORE`, or `CUDA_GPU` with status dot.

### 4.3 Interactive Controls
- **`PrimaryButton`**: Emerald-themed primary action button for key triggers (e.g. "Run Optimization"). Supports disabled/loading state and leading icon.
- **`SecondaryButton`**: Dark slate surface button for secondary navigation or inspections.
- **`GhostButton`**: Borderless subtle button for non-intrusive actions (e.g. "Expand Full JSON").

### 4.4 Data Presentation & Certification
- **`DataTable`**: Dense technical data table with custom column renderers, right-aligned numerical data, and row click handlers.
- **`Pipeline`**: Horizontal sequence visualizer illustrating the four core stages: Model Ingest & Presolve -> Basis Factorization -> Optimization Kernel -> Independent Verification.
- **`VerificationCheck`**: Mathematical assertion card displaying check name, description, exact computed residual, and tolerance threshold.
- **`EmptyState`**: Honest empty container with dashed border and explicit call to action when a model has not yet been executed.

---

## 5. Screen Information Architecture

| Screen Tab | Purpose | Key Content Elements |
| :--- | :--- | :--- |
| **`Overview`** | System identity and executive dashboard | Hero banner, session status, pipeline visualizer, 6 engine capability cards, MRPL showcase banner |
| **`Solve`** | Optimization model configuration & execution | Model selector (AFIRO, Knapsack, Portfolio, MRPL), matrix metrics, solver config form, strategy advisor sidebar, solve results, and CLI command generator |
| **`Verification`** | Zero-trust independent mathematical audit | Four-stage verification pipeline, summary certificate, KKT/slackness assertions, expandable JSON proof artifact |
| **`Benchmarks`** | Performance evidence & multi-core scaling | KPI summary banner, 8-core CPU SpMV Amdahl scaling analysis, filterable Netlib/MIPLIB/QP data table, instance modal inspector |
| **`MRPL Case`** | Real-world refinery planning demonstration | Synthetic data disclosure, baseline economics ($5.4M/day profit), unit load factors, Euro-VI clean fuel quality compliance, what-if scenario matrix |
| **`System`** | Subsystem architecture & rule catalog | 6 subsystem breakdowns, CI/CD quality gate badges (329/329 tests green), interactive Strategy Rules Catalog (R001–R018) |
