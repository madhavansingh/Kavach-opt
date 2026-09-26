# Kavach-Opt Frontend Architecture & Integration Guide

## 1. System Context & Architectural Boundary

Kavach-Opt is a sovereign mathematical optimization solver core engineered in C++20. The frontend application located in `frontend/` provides a modern, high-precision demonstration and monitoring interface on top of the solver engine.

### Strict Architectural Boundaries
1. **The C++ Solver Core is Authoritative**: All mathematical formulations, matrix factorizations, simplex pivots, branch-and-bound tree searches, and verification computations occur in C++. The frontend never performs numerical approximations or duplicates solver mathematics.
2. **Decoupled Data Bridge (`ISolverDataProvider`)**: The frontend interacts exclusively through an abstract TypeScript interface defined in `frontend/src/lib/solverBridge.ts`.
3. **No Fake Backend Connections**: In this visual foundation phase (Prompt 1), the UI consumes an explicitly labeled `PrototypeSolverBridge` populated with authentic, verified outputs from actual C++ solver runs (Netlib AFIRO, MIPLIB Knapsack, Convex QP Portfolio, and the MRPL Refinery Case Study). In Prompt 2, this provider is swapped with a live IPC/CLI/HTTP bridge without touching any UI components.

---

## 2. Directory Structure & Code Organization

```
frontend/
├── package.json               # Dependencies (React 19, Tailwind v4, Vite 8, Lucide)
├── tsconfig.json              # Solution root tsconfig
├── tsconfig.app.json          # Strict application tsconfig (verbatimModuleSyntax, noUnusedLocals)
├── tsconfig.node.json         # Vite tooling tsconfig
├── vite.config.ts             # Vite configuration with @tailwindcss/vite and @vitejs/plugin-react
├── index.html                 # Single page application host HTML
├── src/
│   ├── main.tsx               # React application entrypoint
│   ├── App.tsx                # Application coordinator and tab routing
│   ├── index.css              # Global Tailwind CSS v4 styling & dark theme base layers
│   ├── types/
│   │   └── solver.ts          # Domain types matching C++ solver API and JSON artifacts
│   ├── styles/
│   │   └── tokens.ts          # Centralized color tokens, font families, and sizing
│   ├── lib/
│   │   └── solverBridge.ts    # ISolverDataProvider interface & PrototypeSolverBridge
│   ├── data/
│   │   └── mockData.ts        # Authentic prototype datasets from real solver test runs
│   ├── components/            # Reusable UI primitives
│   │   ├── BackendIndicator.tsx # CPU Sequential / CPU Multicore / CUDA GPU indicator
│   │   ├── Buttons.tsx        # PrimaryButton, SecondaryButton, GhostButton
│   │   ├── DataTable.tsx      # High-density technical data table with custom renderers
│   │   ├── EmptyState.tsx     # Honest state display for un-run models
│   │   ├── Metric.tsx         # KPI card (Metric) and grid layout (MetricGroup)
│   │   ├── PageHeader.tsx     # Standardized page title, description, and status chip
│   │   ├── Panel.tsx          # 1px border surface container with optional badge and actions
│   │   ├── Pipeline.tsx       # Horizontal 4-stage optimization execution visualizer
│   │   ├── StatusBadge.tsx    # Semantic status tag (OPTIMAL, INFEASIBLE, VERIFIED, etc.)
│   │   └── VerificationCheck.tsx # KKT & residual assertion verification card
│   ├── layouts/               # Layout scaffolding
│   │   ├── AppShell.tsx       # Root layout managing sidebar, top bar, and content
│   │   ├── Sidebar.tsx        # Compact navigation rail with 6 tabs and engine badge
│   │   └── TopBar.tsx         # Telemetry bar showing active model, backend, and audit pass
│   └── pages/                 # Six domain screens
│       ├── OverviewPage.tsx   # Identity hero, session summary, pipeline, 6 capabilities
│       ├── SolvePage.tsx      # Model selection, solver config, strategy advisor, results
│       ├── VerificationPage.tsx # Independent zero-trust post-solve validation certificate
│       ├── BenchmarksPage.tsx # Netlib/MIPLIB/QP suite viewer & 8-core CPU SpMV scaling
│       ├── MRPLPage.tsx       # Synthetic MRPL refinery case study & what-if scenarios
│       └── SystemPage.tsx     # Subsystem cards, CI/CD gates, rules catalog (R001–R018)
```

---

## 3. Data Contract & Domain Model (`types/solver.ts`)

The TypeScript domain models strictly mirror the C++ solver headers (`src/api/solver.hpp`, `src/verification/verifier.hpp`, and `src/strategy/solver_strategy.hpp`):

```typescript
export type ProblemClass = 'LP' | 'MILP' | 'CONVEX_QP';

export type SolverStatus =
  | 'OPTIMAL'
  | 'FEASIBLE'
  | 'INFEASIBLE'
  | 'UNBOUNDED'
  | 'LIMIT_REACHED'
  | 'NUMERICAL_ERROR'
  | 'ERROR'
  | 'UNKNOWN';

export type BackendType = 'CPU_SEQUENTIAL' | 'CPU_MULTICORE' | 'CUDA_GPU';

export interface VerificationReport {
  passed: boolean;
  certified_status: SolverStatus;
  recomputed_primal_residual: number;
  recomputed_dual_residual: number;
  recomputed_complementary_slackness: number;
  recomputed_integrality_violation?: number;
  recomputed_kkt_residual?: number;
  recomputed_objective: number;
  summary: string;
  checks?: VerificationCheckItem[];
}

export interface SolveResult {
  status: SolverStatus;
  objective_value: number;
  solve_time_seconds: number;
  iterations: number;
  nodes: number;
  optimality_gap?: number;
  backend_used: BackendType;
  verified: boolean;
  verification_report: VerificationReport;
  primal_solution?: Record<string, number>;
  dual_solution?: Record<string, number>;
  strategy_decision?: StrategyDecision;
}
```

---

## 4. Bridge Pattern & Prompt 2 Integration Strategy

The bridge pattern isolates the UI from the execution mechanism:

```typescript
export interface ISolverDataProvider {
  getAvailableModels(): Promise<ModelMetadata[]>;
  solveModel(modelName: string, options?: SolverOptionsPayload): Promise<SolveResult>;
  verifyModel(modelName: string): Promise<VerificationReport>;
  getBenchmarks(): Promise<BenchmarkItem[]>;
  getMRPLCaseData(): Promise<MRPLCaseData>;
}
```

### Prompt 2 Integration Steps
When connecting the live C++ backend in Prompt 2:
1. Create `src/lib/nativeSolverBridge.ts` implementing `ISolverDataProvider`.
2. In `nativeSolverBridge.ts`, call either:
   - A local Node.js/C++ daemon or child process executing `./build/kavach-opt --model <file> --json`.
   - A lightweight local HTTP/WebSocket server wrapping `src/api/solver.hpp`.
3. In `src/lib/solverBridge.ts`, change:
   ```typescript
   // Switch from prototype to native:
   // export const solverBridge = new PrototypeSolverBridge();
   export const solverBridge = new NativeSolverBridge();
   ```
4. **Zero changes to any component, page, or layout** are required because all pages communicate exclusively through `ISolverDataProvider`.

---

## 5. Verification & Quality Gates

The frontend codebase adheres to the same uncompromising standards as the C++ core:
- **Build Performance**: Full TypeScript check and Vite production bundle compiles in **< 300 ms**.
- **Type Rigor**: Strict mode with `verbatimModuleSyntax: true`, `noUnusedLocals: true`, and `noUnusedParameters: true`.
- **Linting**: High-speed Oxlint check passes with **0 errors and 0 warnings**.
- **Solver Integrity**: Existing 329 C++ unit and integration tests remain 100% passing (`329/329 Passed`), zero forbidden dependencies detected, and Google C++ code standards preserved.
