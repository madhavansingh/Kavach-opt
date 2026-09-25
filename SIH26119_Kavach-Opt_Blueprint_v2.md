# SIH26119 — KAVACH-OPT
## Master Engineering Blueprint v2
### Indigenous GPU-Accelerated Optimization Solver
### MRPL · Software · Smart Automation

**Blueprint status: FROZEN** — specification corrections applied 2026-09-25. No further changes to this document without explicit authorization. The four corrections applied are: (1) QP priority annotation in Section 1; (2) anti-cycling guarantee softened in Sections 6 and 8; (3) "byte-identical" / "zero divergence" / "machine precision" wording replaced with numerical-tolerance equivalents in Sections 4 and 6; (4) open pre-phase decisions annotated in Section 32.

**Status of this document:** this is the source of truth for Phase 0 onward. No
implementation code is included. Everything described below is **PLANNED**
unless explicitly marked otherwise — the attached repository at the time of
this revision contains no solver code, so nothing in this document is tagged
**IMPLEMENTED**. The tagging convention is used consistently throughout:

- **IMPLEMENTED** — exists, builds, and is tested in the repository today.
- **PLANNED** — committed engineering scope for the phases in Section 32; not yet built.
- **EXPERIMENTAL** — scoped and understood, attempted only if time/evidence permits; not a core commitment.
- **THEORETICAL** — understood conceptually, deliberately not attempted in this project's timeline; listed for architectural completeness and future-work honesty only.

No claim in this document should be read as a promise that Kavach-Opt will
outperform CPLEX, Gurobi, or HiGHS in general. It will not. The goal is a
correct, numerically defensible, from-scratch LP/MILP/QP solver core, with
every capability claim backed by a test, a benchmark, or an explicit PLANNED
label — nothing else.

---

## Section 0 — Executive Summary

**Problem Statement ID:** SIH26119
**Organization:** Mangalore Refinery and Petrochemicals Limited (MRPL)
**Title:** Indigenous GPU-Accelerated Optimization Solver (Sovereign Alternative to CPLEX/Gurobi/Xpress)

**What MRPL is asking for.** MRPL's refinery scheduling, crude blending,
production planning, and supply-chain optimization workflows currently depend
on foreign commercial LP/MILP/QP solvers (CPLEX, Gurobi, Xpress) — expensive to
license, restrictive, and opaque at the algorithm level. The PS asks for a
**sovereign optimization solver *core*** — not a modeling environment, not a
UI — built from mathematical first principles, supporting LP, MILP, and QP
initially, with a modular architecture that can later extend to MIQP, NLP, and
MINLP. It must exploit sparse matrix techniques, efficient numerical linear
algebra, multicore parallelism, and GPU acceleration where GPU acceleration is
*measurably* beneficial, with numerical stability and reliable convergence as
the primary engineering emphasis — not graphical polish.

**What Kavach-Opt is.** Kavach-Opt is a from-scratch, sparse, numerically
self-verifying LP/MILP/QP solver core with a CPU-first execution backend and
one scoped, honestly-benchmarked GPU-accelerated kernel. It exposes a solver
API and CLI (no GUI is required or planned as load-bearing). It reads MPS and
a simple native LP format, solves via revised simplex (LP), branch-and-bound
(MILP), and active-set (convex QP), and reports every result through an
**independent verification layer** that recomputes feasibility and optimality
without trusting the solving algorithm's own bookkeeping.

**What exactly we will build**, at the Strong-SIH-Prototype scope defined in
Section 5: a correct, sparse, presolved, scaled revised-simplex LP engine; a
branch-and-bound MILP engine built on it; a convex active-set QP engine;
CSR/CSC sparse numerical machinery with LU factorization; a named-tolerance
numerical robustness policy; an independent verification subsystem; one
GPU-accelerated kernel (batched sparse matrix-vector product) with an honest
CPU/GPU crossover benchmark; a benchmark harness against Netlib LP, a
size-appropriate MIPLIB subset, and (if convex QP is stable in time) QPLIB;
an external, non-embedded comparison against a reference solver (HiGHS); and
an illustrative MRPL crude-blending + unit-capacity MILP demonstration model.

**What makes it sovereign.** Every optimization algorithm — simplex pivoting,
branch-and-bound search, active-set QP — is implemented by the team from its
mathematical definition. No optimization-engine library (CPLEX, Gurobi, Xpress,
HiGHS, SCIP, GLPK, CBC, OR-Tools, cuOpt, or any other) is imported, linked, or
vendored anywhere in the solving path. Standard, well-understood numerical
building blocks (a sparse LU factorization routine the team writes itself) are
expected and normal; the sovereignty claim is about the **optimization
algorithm**, not a blanket refusal of numerical computing conventions.

**Core solver classes:** LP (Linear Programming), MILP (Mixed-Integer Linear
Programming), and convex QP (Quadratic Programming), in that priority order,
because MILP is built on top of LP (every B&B node re-solves an LP) and QP is
sequenced last so it does not draw engineering time away from the two PS-
mandated core classes while they are still stabilizing.

**Core differentiation.** Not a claim to have out-invented simplex or
branch-and-bound. The differentiation is: (1) an independent verification
layer that never trusts the solver's self-reported "optimal" status; (2)
transparent, reproducible, honestly-reported benchmarking — including losses
and timeouts, not a curated subset; (3) a real sparse architecture proven at
Netlib/MIPLIB scale, not a toy dense demo; (4) an honestly-scoped GPU claim —
one kernel, one crossover chart, never "a GPU solver."

**Primary engineering objective.** Correctness first, then numerical
robustness, then measured performance. A working, verified CPU-only LP/MILP/QP
solver is a stronger submission than a GPU-accelerated one that has not been
proven correct.

**What will actually be demonstrated on demo day** (Section 28): a live solve
of a real Netlib LP instance with presolve statistics and simplex telemetry; a
live MILP solve with branch-and-bound tree/gap telemetry; the independent
verifier running against both, including a deliberately-broken build catching
an injected wrong answer; a CPU-vs-GPU timing chart for the one accelerated
kernel, including the crossover point below which GPU is slower; the MRPL
blending/capacity model solved end-to-end; one honest benchmark table against
HiGHS, wins and losses both shown; and a closing statement of what is
currently NOT implemented.

**Reading this document.** A technical judge should be able to verify every
claim against Section 24's evidence matrix. An optimization expert should be
able to audit the algorithmic decisions in Sections 7–10. A non-specialist
judge should be able to follow Section 2 (what we are building) and Section 21
(the MRPL demo) without needing the mathematics.

---

## Section 1 — Official PS Requirements Traceability

Every row below traces directly to language or clearly stated intent in the
official SIH26119 problem statement. No requirement is invented. "Our
implementation" always cites a specific section of this document.

| # | Requirement | Source / PS Intent | Priority | Our Implementation | Verification Method | Evidence Artifact | Demo Method | Status |
|---|---|---|---|---|---|---|---|---|
| R1 | Solve Linear Programming (LP) | "should support Linear Programming (LP)... as the initial focus" | MANDATORY | Revised simplex engine, Section 8 | Adversarial suite (Sec. 22) + independent verifier (Sec. 12) | LP test report, Netlib results | Live solve of a Netlib instance | PLANNED |
| R2 | Solve Mixed-Integer LP (MILP) | "...Mixed-Integer Linear Programming (MILP)... as the initial focus" | MANDATORY | Branch-and-bound engine, Section 9 | Adversarial suite + verifier | MILP test report, MIPLIB subset results | Live solve with B&B telemetry | PLANNED |
| R3 | Solve Quadratic Programming (QP) | "...Quadratic Programming (QP) as the initial focus" | MANDATORY per PS; **implementation priority sequenced after LP and MILP** — QP is Category B (Strong SIH Target), not Category A, because LP and MILP are the PS's stated primary initial focus and QP must not draw engineering time from them while they are still stabilizing (see Section 5 and Section 10 sequencing note) | Convex active-set QP, Section 10 | KKT residual verification | QP test report | Live small convex QP solve | PLANNED |
| R4 | Modular extensibility to MIQP/NLP/MINLP | "modular architecture that can later be extended to MIQP, NLP and MINLP" | MANDATORY (architectural, not functional) | Extension points in Section 4 architecture; no functional MIQP/NLP/MINLP claimed | Architecture review against extension points | Architecture doc (Sec. 29) | Point to extension points in diagram, no functional demo claimed | PLANNED (architectural hooks only) |
| R5 | Built from scratch, not on any existing solver library | "shall not be built upon any existing open source solver library but shall be built from scratch from mathematical foundation" | MANDATORY | Zero optimization-engine dependency anywhere in the solve path, Section 2/30 | Dependency-graph grep in CI | Build manifest, CI dependency check | Live `grep`/dependency listing on demand | PLANNED (policy defined now, enforced from Phase 0) |
| R6 | Revised simplex / interior-point for continuous optimization | "Core algorithms may include revised simplex and interior-point methods" | STRONGLY DESIRABLE (revised simplex chosen; interior-point deferred) | Revised simplex, Section 8; interior-point explicitly FUTURE, Section 36 | Adversarial + benchmark suite | LP algorithm doc | Live solve | Simplex: PLANNED; interior-point: FUTURE |
| R7 | Branch-and-bound / branch-and-cut / cutting planes / presolve / heuristics / node selection for MILP | "branch-and-bound, branch-and-cut, cutting planes, presolve, heuristics and advanced node selection strategies" | MANDATORY core (B&B, presolve, basic heuristics); STRONGLY DESIRABLE (cuts, advanced node selection) | Section 9; cuts scoped as EXPERIMENTAL stretch | MILP adversarial suite | MILP algorithm doc, benchmark report | Live MILP solve with gap closing | Core: PLANNED; cuts: EXPERIMENTAL |
| R8 | Sparse matrix techniques | "exploit sparse matrix techniques" | MANDATORY | CSR/CSC layer, Section 13 | Memory-scaling benchmark vs. instance size | Sparse layer test report | Show memory scaling sub-quadratically | PLANNED |
| R9 | Efficient numerical linear algebra | "efficient numerical linear algebra" | MANDATORY | Sparse LU factorization, scaling, refinement, Section 13 | Factorization correctness tests on known systems | Numerics test report | — | PLANNED |
| R10 | Numerical robustness / stability | "The emphasis is on numerical stability... rather than on graphical interfaces" | MANDATORY | Section 11 (tolerance taxonomy, status taxonomy, verification) | Adversarial suite (ill-conditioned, degenerate) | Numerical robustness doc | Live degenerate/ill-conditioned case | PLANNED |
| R11 | Reliable convergence, scalability | "reliable convergence across large industrial optimization problems" | MANDATORY | Presolve + scaling + anti-cycling, Sections 8–11 | Benchmark on Netlib/MIPLIB with honest size ceiling reporting | Benchmark report | Live solve at largest verified size | PLANNED |
| R12 | Multicore CPU parallelization | "multi-core parallelization" | STRONGLY DESIRABLE | Section 14 | CPU parallel-vs-serial correctness + speedup benchmark | CPU parallelism report | Measured speedup chart | PLANNED |
| R13 | GPU acceleration where measurable | "GPU acceleration considered where it provides measurable benefits" | STRONGLY DESIRABLE | Section 15, one scoped kernel (batched SpMV) | CPU/GPU equivalence test + crossover benchmark | GPU benchmark report | Live crossover chart, `--no-gpu` fallback shown | PLANNED |
| R14 | Basic API or CLI (no GUI required) | "A robust optimization engine with a basic API or CLI is sufficient; a polished GUI is not required" | MANDATORY (API/CLI); explicitly NOT required (GUI) | Section 17 | CLI smoke tests | API/CLI doc | Live CLI solve | PLANNED |
| R15 | Solve standard benchmark problems (MIPLIB, Netlib, Mittelmann) | "should successfully solve standard benchmark problems from... MIPLIB, Netlib or Mittelmann" | MANDATORY | Section 18 | Full benchmark run, honestly reported | Benchmark report | Live + pre-recorded backup | PLANNED |
| R16 | QPLIB where applicable | Dataset guidance: "QPLIB (for quadratic programming where applicable)" | OPTIONAL / FUTURE (depends on QP stability and time) | Section 18 category | QP benchmark subset | QP benchmark report | If time permits | PLANNED (conditional), else FUTURE |
| R17 | Comparison against at least one established solver | "solution quality and computational performance compared against at least one established commercial or open-source solver" | MANDATORY | Section 20, HiGHS as external reference only | Fair-comparison protocol | Comparison report | Live table, wins and losses | PLANNED |
| R18 | Demonstrate numerical robustness on degeneracy/weak relaxations/ill-conditioning | "clear demonstration of numerical robustness... degeneracy, weak LP relaxations or ill-conditioned constraint matrices" | MANDATORY | Section 22 adversarial suite | Deliberate adversarial test cases | Adversarial test report | Live degenerate/ill-conditioned solve | PLANNED |
| R19 | Industrial relevance (refinery scheduling, blending, planning, logistics, power, transport, supply chain) | "scope is to solve optimization problems arising from refinery scheduling, crude blending, process optimization, production planning, logistics, power system dispatch, transportation and supply chain management" | MANDATORY (at least one worked example) | Section 21, MRPL crude-blending + capacity-allocation model | Model-specific correctness check | Case study doc | Live MRPL demo solve | PLANNED |
| R20 | Extensibility (transparent, extensible, sovereign foundation) | "transparent, extensible and sovereign foundation for future Indian optimization software" | MANDATORY (architectural) | Section 4 architecture, Section 36 future extensions | Architecture review | Architecture doc | Point to extension points | PLANNED |

No PS requirement has been invented or expanded beyond stated or clearly
implied intent. Items explicitly marked OPTIONAL/FUTURE above (interior-point
methods, full cutting-plane arsenal, QPLIB, MIQP/NLP/MINLP functional support)
are PS-permitted future extensions, not core obligations, and are treated as
such throughout this document.

---

## Section 2 — What We Are Actually Building

**Four things that are easy to conflate, and must not be:**

1. **Optimization application** — software that models a specific business
   problem (e.g., "refinery blending planner") and calls an optimization
   engine to solve it. This is *not* the PS ask; a polished application
   without a real engine underneath would fail the PS's explicit "not built
   upon any existing solver library" requirement by definition, since it would
   have to call something.
2. **Optimization engine** — the numerical core that actually computes an
   optimum: the simplex iterations, the branch-and-bound search, the active-
   set QP steps, the sparse linear algebra underneath all three. **This is
   what Kavach-Opt is.**
3. **Modeling interface** — a language or UI for expressing an optimization
   problem in domain terms (e.g., AMPL, Pyomo-style modeling). The PS
   explicitly states a modeling environment is *not* what is being asked for
   ("The objective is to develop a sovereign mathematical optimization solver
   core rather than a complete modeling environment").
4. **Numerical solver core** — synonym for (2) in this document; the
   phrase "solver core" in the PS and this term are used interchangeably.

**One-sentence definition:** Kavach-Opt is a from-scratch numerical
optimization engine — the algorithms that turn a matrix of constraints and an
objective into a verified optimal (or provably infeasible/unbounded) answer —
not an application, not a UI, not a wrapper.

**30-second explanation:** Every industrial planning decision — how much crude
to blend, how to allocate refinery unit capacity, how to route a supply chain
— can be written as a system of equations and inequalities with an objective
to maximize or minimize. Solving that system, at real industrial scale, with a
guarantee that the answer is actually correct, is the entire engineering
problem. Kavach-Opt is the code that does that solving, built from the
mathematics up, without depending on a foreign vendor's engine.

**Technical definition:** Kavach-Opt is a sparse-matrix-native LP/MILP/QP
solver library exposing a stable API and CLI, implementing revised simplex
(LP), branch-and-bound (MILP), and active-set (convex QP) from their
mathematical formulations, backed by a from-scratch sparse LU factorization
layer, a named-tolerance numerical robustness policy, and an independently-
coded verification layer, with one CPU/GPU execution backend abstraction
scoped to a single measured GPU kernel.

**System boundary — what is inside the solver:**
- Canonical model representation, parsing (MPS/native LP)
- Presolve
- Sparse numerical core (CSR/CSC, LU factorization, scaling, refinement)
- LP engine (revised simplex)
- MILP engine (branch-and-bound on top of the LP engine)
- QP engine (active-set, convex only)
- Independent verification layer
- CPU and (scoped) GPU execution backends
- Solver API and CLI
- Benchmark harness

**What is outside the solver (explicitly not part of the sovereign core):**
- Any GUI/dashboard (optional, thin, never load-bearing — Section 4)
- Any modeling language or algebraic front-end beyond MPS/native-LP parsing
- Any use of an external optimization-engine library as a dependency
- The reference solver (HiGHS) used in Section 20 — it sits entirely outside
  the solve path, invoked only for arms-length comparison, never called by
  Kavach-Opt internally

**Statement of record:** *We are building the optimization engine itself, not
an application that delegates optimization to another solver.* Every
algorithmic decision in Sections 7–10 exists to make that statement literally
true and independently checkable (Section 24).

---

## Section 3 — Final Proposed Solution

**Product name:** Kavach-Opt — "kavach" (shield/armor) signals robustness and
sovereignty; "Opt" states the domain plainly. *(To be cross-checked for naming
collisions before final submission, same diligence as any other project name.)*

**One-line problem:** MRPL, and Indian industry broadly, needs an optimization
solver core it does not have to license or trust to a foreign vendor.

**One-line solution:** Kavach-Opt is a from-scratch, sparse, numerically
self-verifying LP/MILP/QP solver core with a CPU-first, selectively-GPU-
accelerated execution backend.

**Detailed solution concept:** a model (MPS or native LP file) is parsed into
a canonical sparse representation, reduced by presolve, solved by revised
simplex (LP) or branch-and-bound-on-simplex (MILP) or active-set (QP), and
every reported result is independently re-verified against feasibility and
optimality conditions before being returned. Execution runs on CPU by default;
one specific, benchmarked operation (batched sparse matrix-vector product) may
run on GPU when the problem size justifies it, with automatic and always-
available CPU fallback.

**USP:** "We show our work, and we check our own answers." The differentiation
is rigor and honesty — an independent verifier that never trusts the solver's
self-report, and benchmark reporting that includes losses — not a claim to
beat commercial solvers outright.

**Technical differentiators:**

| Differentiator | Provability status |
|---|---|
| Independent, separately-coded verification layer that recomputes feasibility/optimality | **PROVABLE** — deliberate fault-injection test (Section 12) demonstrates it live |
| Honest, non-cherry-picked benchmark reporting (wins and losses both shown) | **PROVABLE** — full results table published, re-runnable on demand |
| Real sparse architecture at Netlib/MIPLIB scale (not a toy dense demo) | **PROVABLE** — memory-scaling measurement vs. instance size |
| Reproducible experiments (fixed seeds, logged configuration, re-runnable) | **PROVABLE** — live re-run with same seed in front of judges |
| From-scratch algorithm implementation (no optimization-engine dependency) | **PROVABLE** — dependency-graph grep, live code walkthrough |
| Measured (not claimed) CPU/GPU speedup with honest crossover point | **PARTIALLY PROVABLE** — provable once the GPU kernel and benchmark harness exist (Phase 11); until then it is a committed method, not yet a result |
| Solver strategy layer choosing settings from problem characteristics | **PARTIALLY PROVABLE** — rule-based version (Section 16) is buildable and testable; not a headline claim on its own |
| Industrial relevance to MRPL (blending/capacity structure) | **PARTIALLY PROVABLE** — structurally realistic with illustrative data; would become fully provable only with real MRPL data, which is not assumed available |
| Extensibility toward MIQP/NLP/MINLP | **FUTURE** — architectural extension points only, no functional claim |

**MRPL relevance:** refinery crude/product blending, unit-capacity allocation,
and production planning are LP/MILP problems by mathematical structure — the
same solver core that passes Netlib/MIPLIB benchmarks is the core that solves
MRPL's planning problems once the input pipeline exists (Section 21).

**Broader applicability:** any organization currently paying commercial
LP/MILP licensing costs for small-to-medium industrial planning problems
(power dispatch, transportation, general supply chain) is a potential
downstream user of the same solver core, without modification to the core
itself.

**On the word "innovation," stated directly:** Kavach-Opt does not claim to
have invented simplex, branch-and-bound, or active-set QP — these are
70-plus-year-old, well-published algorithms. The innovation being claimed is
narrower and more defensible: a sovereign, from-scratch *implementation* of
these algorithms, integrated with an independent verification discipline and
transparent benchmarking practice that (per the PS's own description of
common competing-team weaknesses) most comparable submissions skip.

---

## Section 4 — Final Architecture

### 4.1 High-level architecture

```
Model file (.lp / .mps)
        │
        ▼
   [B] Parser ──────────────► [A] Canonical Model Representation
        │                              │
        ▼                              ▼
   [C] Presolve  ◄──────────────────────
        │
        ▼
   [D] Sparse Matrix Layer (CSR/CSC)
        │
        ▼
   [E] Numerical Linear Algebra (sparse LU, scaling, refinement)
        │
   ┌────┴─────┬─────────────┐
   ▼          ▼             ▼
[F] LP     [H] QP        (shared numerics)
Solver     Solver
   │
   ▼
[G] MILP Solver
   ├─ [I] Branch-and-bound tree / node objects
   ├─ [J] Branch-and-cut orchestration (cuts EXPERIMENTAL, see Sec. 9)
   ├─ [K] Cutting planes (EXPERIMENTAL stretch)
   ├─ [L] Node selection strategy
   ├─ [M] Branching-variable strategy
   ├─ [N] Primal heuristics
   ├─ [O] Incumbent management
   └─ [P] Optimality-gap tracking
        │
        ▼
   [Q] Feasibility/status computation ──► [Y] Independent Verification Layer
        │
        ▼
   [R] Numerical stability policy (tolerances/scaling — cross-cutting D–P)
        │
   ┌────┴────┐
   ▼         ▼
[S] Multicore   [U] CUDA/GPU backend (one scoped kernel, see Sec. 15)
CPU execution   │
   └──────┬─────┘
          ▼
   [T] Solver Strategy Layer (rule-based problem-characteristic → settings)
          │
          ▼
   [V] Solver API → [W] CLI
          │
          ▼
   [X] Benchmark framework → [Z] Logging/telemetry → [AA] Reproducibility records
          │
          ▼
   [AB] Industrial application layer (MRPL demo model) → [AC] Optional dashboard (thin, last, never load-bearing)
```

### 4.2 Subsystem specification

For each component: Purpose, Inputs, Outputs, Dependencies, Critical
Algorithms, Tests, Failure Modes, Completion Criteria.

**[A] Canonical Model Representation**
- *Purpose:* in-memory model of variables, objective, constraints, bounds, integrality flags — the single representation every other component reads from and writes to.
- *Inputs:* parsed model data from [B], or programmatic API construction.
- *Outputs:* an immutable, validated model object consumed by [C].
- *Dependencies:* none (leaf data structure).
- *Critical algorithms:* none (data structure + validation logic).
- *Tests:* construction/validation unit tests, round-trip tests with [B].
- *Failure modes:* malformed bounds (lower > upper), duplicate variable/constraint IDs, non-finite coefficients.
- *Completion criteria:* rejects all malformed inputs above with a specific, typed error; accepted models round-trip through serialization with structural metadata (variable/constraint counts, names, bound types, integrality flags) preserved exactly and floating-point coefficient values preserved within a small multiple of machine epsilon (accounting for decimal-representation rounding).

**[B] Parser (MPS / native LP)**
- *Purpose:* turn a model file into [A].
- *Inputs:* `.mps` file (fixed or free format) or `.lp` native-format file.
- *Outputs:* a populated [A] instance.
- *Dependencies:* [A].
- *Critical algorithms:* tokenizer + section-state-machine parsing (MPS has distinct ROWS/COLUMNS/RHS/BOUNDS/RANGES sections).
- *Tests:* parses a corpus of real Netlib MPS files correctly; rejects malformed files with a clear error, never silently mis-parses.
- *Failure modes:* silent misparse of an edge-case MPS section (RANGES, free rows) producing a wrong-but-plausible model — this is the most dangerous failure mode for this component because it corrupts everything downstream without an obvious symptom.
- *Completion criteria:* correctly parses a minimum of 10 real Netlib instances, verified by comparing parsed variable/constraint counts and a handful of spot-checked coefficients against published instance metadata.

**[C] Presolve**
- *Purpose:* remove redundant rows/columns and tighten bounds before solving, reducing problem size without changing the optimal solution set.
- *Inputs:* validated [A] model.
- *Outputs:* a reduced model plus a reconstruction map (to translate the reduced solution back to original variable space).
- *Dependencies:* [A].
- *Critical algorithms:* fixed-variable substitution, redundant/empty-row removal, bound tightening from single-row implications.
- *Tests:* before/after size checks on real instances; reduced-then-restored solution matches directly-solved solution within tolerance.
- *Failure modes:* an incorrect reduction that silently changes the feasible region (the single most dangerous bug class in this component — every presolve reduction must have a paired reconstruction test).
- *Completion criteria:* measurable size reduction on real Netlib instances; reduced-then-restored solutions agree with direct-solve results within `primal_feasibility_tol` across the full LP test suite (finite-precision arithmetic precludes exact zero divergence; this tolerance is the acceptance gate).

**[D] Sparse Matrix Layer (CSR/CSC)**
- *Purpose:* store the constraint matrix without wasting memory/time on structural zeros.
- *Inputs:* the (possibly presolved) model's constraint matrix.
- *Outputs:* CSR and/or CSC representations as needed by [E].
- *Dependencies:* [A]/[C].
- *Critical algorithms:* CSR/CSC construction and conversion, sparse matrix-vector product (SpMV).
- *Tests:* SpMV correctness against dense reference on small matrices; memory footprint scales with nonzero count, not with `rows × cols`.
- *Failure modes:* incorrect index bookkeeping producing silently wrong SpMV results.
- *Completion criteria:* SpMV results agree with the dense reference implementation within a small multiple of machine epsilon times the matrix's Frobenius norm (accounting for floating-point summation-order differences between sparse and dense traversal); memory usage measured and shown to scale with nonzeros on real instances.

**[E] Numerical Linear Algebra (sparse LU, scaling, refinement)**
- *Purpose:* factorize and solve the linear systems that simplex and active-set QP require at every iteration.
- *Inputs:* the current basis matrix (sparse, from [D]).
- *Outputs:* factorized basis (LU factors), triangular-solve results, refined solutions.
- *Dependencies:* [D].
- *Critical algorithms:* sparse LU factorization with partial pivoting, fill-in-aware ordering, product-form basis updates between iterations, iterative refinement, geometric-mean row/column scaling.
- *Tests:* factorization matches hand-computed results on small known systems; residual after solve is below the documented tolerance (Section 11) on all test instances.
- *Failure modes:* near-zero pivot silently accepted (producing a numerically meaningless factorization); fill-in growth without bound on poorly-ordered matrices.
- *Completion criteria:* near-zero pivots are detected and flagged, never silently accepted; refinement measurably reduces residual on ill-conditioned test cases.

**[F] LP Solver (revised simplex)** — full specification in Section 8.

**[G] MILP Solver (branch-and-bound)** — full specification in Section 9.

**[H] QP Solver (active-set, convex)** — full specification in Section 10.

**[Q]/[Y] Feasibility computation and Independent Verification Layer**
- *Purpose:* recompute primal/dual feasibility, complementary slackness (LP), integrality (MILP), and KKT residuals (QP) independently of the solving algorithm, so a bug in the solver cannot silently produce a false "optimal."
- Full specification in Section 12.

**[R] Numerical stability policy**
- *Purpose:* a single, cross-cutting, named-tolerance policy (Section 11) applied consistently by [E] through [P] — not a scattered set of ad-hoc epsilons.

**[S]/[U] Execution backends**
- *Purpose:* CPU and (scoped) CUDA implementations of specific parallel-friendly operations, selected per-operation, never globally. Full specification in Sections 14–15.

**[T] Solver Strategy Layer**
- *Purpose:* rule-based selection of presolve level, scaling, branching strategy, and CPU/GPU routing from measurable problem characteristics. Full specification in Section 16.

**[V]/[W] Solver API / CLI** — full specification in Section 17.

**[X]/[Z]/[AA] Benchmark framework, telemetry, reproducibility records** — full specification in Sections 18, 23.

**[AB] Industrial application layer (MRPL demo model)** — full specification in Section 21.

**[AC] Optional dashboard** — explicitly **not** a core dependency; built last if
at all, thin, and no other component's Definition of Done depends on it
existing.


---

## Section 5 — Hard Scope Boundary

This is the single most important decision-table in this document; every
later section is calibrated against it.

| Category | Content |
|---|---|
| **A. NON-NEGOTIABLE CORE** | Canonical mathematical model [A]; MPS + native LP parsing [B]; sparse CSR/CSC representation [D]; sparse LU factorization, scaling, refinement [E]; presolve (bound tightening + redundancy removal) [C]; revised simplex LP [F] with named numerical tolerances; infeasibility and unboundedness detection with certificates; branch-and-bound MILP [G] with best-bound node selection, most-fractional branching, integer-feasibility checking, node/bound management, optimality-gap tracking; independent verification layer [Y]; solver API and CLI [V][W]. |
| **B. STRONG SIH TARGET** *(the recommended, realistic build target)* | Everything in A, plus: Bland's-rule anti-cycling fallback; dual simplex for MILP child-node warm starts; 1–2 simple primal heuristics (rounding); convex QP via active-set [H]; one scoped, measured GPU kernel (batched SpMV) with honest crossover benchmarking; multicore batched SpMV on CPU; Netlib + size-appropriate MIPLIB benchmark reporting; external HiGHS comparison; the MRPL blending/capacity demo model; the full adversarial test suite (Section 22). |
| **C. STRETCH** *(attempted only if A and B are solid and time remains)* | Gomory mixed-integer cuts; pseudocost branching; parallel branch-and-bound; feasibility-pump-grade heuristics; QPLIB benchmark coverage; a thin optional dashboard. |
| **D. FUTURE / OUT OF SCOPE** *(explicitly not attempted in this project)* | Interior-point LP; MIQP; NLP; MINLP; general non-convex QP; GPU-accelerated sparse factorization; a full cutting-plane arsenal; distributed/multi-node optimization; learned (ML-based) solver strategy. |

**GPU scoping rule (stated once here, detailed in Section 15):** GPU
acceleration is scoped to **exactly one** kernel — batched sparse
matrix-vector product — inside Category B. GPU is never promoted to Category
A; the solver must be fully correct and demonstrable with `--no-gpu` at all
times. No other component is made to depend on GPU availability.

**Why this recommendation, stated directly:** each individual Category-D item
(interior-point methods, GPU factorization, a full cutting-plane suite,
parallel tree search) is itself a substantial piece of ongoing numerical-
optimization research. Attempting several simultaneously, under a hackathon
timeline, risks exactly the failure pattern the PS itself warns competing
teams fall into — unsupported GPU claims and absent real benchmark evidence.
A Category B (Strong) target, executed completely and verified rigorously, is
more defensible in front of judges than a Category D (Advanced) target
attempted shallowly and unable to survive a follow-up question.

---

## Section 6 — Definition of Done

For every major subsystem, a concrete, checkable Definition of Done —
Implementation Requirement, Tests Required, Acceptance Criteria, Evidence
Required, Failure Condition. No threshold below is invented to look
impressive; each is either directly tied to correctness or to a measurement
process, not to a fabricated performance number.

**Model Layer [A]**
- *Implementation requirement:* validated in-memory representation of variables, objective, constraints, bounds, integrality.
- *Tests required:* construction/validation unit tests; malformed-input rejection tests.
- *Acceptance criteria:* every malformed input in the test corpus is rejected with a specific error; every valid input round-trips with structural metadata (variable/constraint counts, names, bound types, integrality flags) preserved exactly and floating-point coefficients preserved within a small multiple of machine epsilon.
- *Evidence required:* unit test report.
- *Failure condition:* any malformed model is silently accepted and passed downstream.

**Parser [B]**
- *Implementation requirement:* correct MPS and native-LP parsing.
- *Tests required:* parse ≥10 real Netlib MPS files; adversarial malformed-file tests.
- *Acceptance criteria:* parsed variable/constraint counts and spot-checked coefficients match published instance metadata exactly.
- *Evidence required:* parser test report listing each instance and pass/fail.
- *Failure condition:* any silent misparse (wrong model accepted without error) on the tested corpus.

**Sparse Matrix Layer [D]**
- *Implementation requirement:* correct CSR/CSC construction, conversion, SpMV.
- *Tests required:* SpMV vs. dense reference on the unit-test corpus; memory-scaling measurement on real instances.
- *Acceptance criteria:* SpMV results agree with the dense reference within a small multiple of machine epsilon times the matrix's Frobenius norm (accounting for summation-order differences); measured memory grows with nonzero count, not `rows × cols`.
- *Evidence required:* numerics test report + memory-scaling chart.
- *Failure condition:* any SpMV mismatch beyond floating-point rounding; memory growth that is clearly quadratic in matrix dimension rather than nonzero count.

**Factorization [E]**
- *Implementation requirement:* sparse LU with partial pivoting, fill-in-aware ordering, product-form updates, iterative refinement.
- *Tests required:* factorization vs. hand-computed small systems; residual measurement on all LP test instances.
- *Acceptance criteria:* residuals below documented tolerance (Section 11); near-zero pivots always flagged, never silently accepted.
- *Evidence required:* factorization test report.
- *Failure condition:* any accepted factorization with residual above tolerance and no flag raised.

**Presolve [C]**
- *Implementation requirement:* bound tightening, redundant-row/empty-row removal, fixed-variable substitution, with a reconstruction map.
- *Tests required:* before/after size checks; reduced-then-restored solution vs. direct-solve comparison on every LP test instance.
- *Acceptance criteria:* measurable size reduction on real instances; reduced-then-restored solutions agree with direct-solve results within `primal_feasibility_tol` on every LP test instance.
- *Evidence required:* presolve reduction report.
- *Failure condition:* any single instance where reduced-and-restored solving produces a different feasible region than direct solving.

**LP [F]**
- *Implementation requirement:* revised simplex, sparse-LU basis, Dantzig entering rule with Bland's-rule anti-cycling fallback, ratio test, scaling.
- *Tests required:* full adversarial suite item set relevant to LP (Section 22); Netlib instance corpus.
- *Acceptance criteria — "LP is complete only when":* known feasible problems solve correctly; infeasible cases are detected and reported with a certificate, never looped or crashed; unbounded cases are detected and reported; degenerate cases do not cycle in the adversarial test suite (verified via the cycling-prone case — the watchdog-triggered Bland's-rule fallback provides practical cycling resistance; note that the theoretical finite-termination guarantee of Bland's rule applies only to unconditional application, not to the watchdog-triggered fallback used here, as clarified in Section 8); objective and feasibility independently verify via [Y]; results are reproducible (same seed, same result); benchmark evidence is recorded for every attempted instance, including failures.
- *Evidence required:* LP test report + Netlib benchmark report.
- *Failure condition:* any instance where the solver reports OPTIMAL but the verifier disagrees; any observed cycling; any silent timeout misreported as optimal.

**Verification [Y]**
- *Implementation requirement:* independently-coded recomputation of primal/dual feasibility, complementary slackness, integrality, KKT residuals.
- *Tests required:* deliberate fault-injection test (a build with an intentionally wrong "optimal" claim).
- *Acceptance criteria:* the verifier catches every injected fault in the fault-injection suite; the verifier's code path shares no logic with the solving code path for the checks it performs.
- *Evidence required:* fault-injection test report.
- *Failure condition:* any injected fault the verifier fails to catch; any shared logic between solver and verifier that could let a common bug pass both.

**MILP [G]**
- *Implementation requirement:* branch-and-bound on top of [F], best-bound node selection, most-fractional branching, incumbent tracking, optimality-gap computation.
- *Tests required:* integer-infeasible instances; weak-relaxation instances; node-explosion/timeout instances; MIPLIB subset.
- *Acceptance criteria:* integer feasibility of every reported solution independently verified; correct, honestly-reported optimality gap on every instance, including ones that time out; no MILP instance ever reports OPTIMAL without a verified zero gap.
- *Evidence required:* MILP test report + MIPLIB subset benchmark report.
- *Failure condition:* any reported OPTIMAL with nonzero true gap; any timeout silently reported as a definitive result rather than "best found, gap X%."

**QP [H]**
- *Implementation requirement:* convex active-set method, PSD Hessian check before solving.
- *Tests required:* KKT-residual tests on convex QP test instances; PSD rejection test on a deliberately non-convex input.
- *Acceptance criteria:* KKT residual below documented tolerance on all convex test instances; non-convex input is rejected/flagged, never silently "solved."
- *Evidence required:* QP test report.
- *Failure condition:* any accepted non-convex input; any KKT residual above tolerance reported as OPTIMAL.

**CPU parallelism [S]**
- *Implementation requirement:* thread pool for batched SpMV across independent right-hand sides.
- *Tests required:* parallel-vs-serial correctness; measured speedup on a real workload.
- *Acceptance criteria:* parallel results match serial results within tolerance; measured (not assumed) speedup reported honestly, including cases with no speedup.
- *Evidence required:* CPU parallelism benchmark report.
- *Failure condition:* any parallel/serial result mismatch; a claimed speedup that does not reproduce on re-run.

**GPU [U]**
- *Implementation requirement:* one CUDA kernel (batched SpMV), CPU fallback always available.
- *Tests required:* GPU/CPU equivalence test; problem-size sweep benchmark including transfer time.
- *Acceptance criteria:* GPU and CPU results agree within tolerance; a documented crossover point is measured, including sizes where GPU is honestly slower.
- *Evidence required:* GPU benchmark report with crossover chart.
- *Failure condition:* any GPU/CPU result mismatch beyond tolerance; a GPU speedup number that does not reproduce live.

**Benchmarking [X]**
- *Implementation requirement:* a harness that runs Netlib/MIPLIB/(QPLIB if reached) instances and records the full field set defined in Section 18.
- *Tests required:* reproducibility re-run test (same seed, same result).
- *Acceptance criteria:* every attempted instance is reported, including failures and timeouts; no aggregate statistic hides individual failures.
- *Evidence required:* full benchmark report.
- *Failure condition:* any instance silently dropped from the report; any aggregate number that cannot be traced back to individual instance rows.

**MRPL Demo [AB]**
- *Implementation requirement:* the crude-blending + unit-capacity MILP model (Section 21), structurally realistic, illustrative data clearly labeled.
- *Tests required:* model-specific correctness check (constraint satisfaction, mass balance).
- *Acceptance criteria:* the model solves, is independently verified, and its result is operationally interpretable (Section 21).
- *Evidence required:* case-study document.
- *Failure condition:* an unverified or unlabeled-as-illustrative result presented as real MRPL data.

**CLI/API [V][W]**
- *Implementation requirement:* documented solver API, CLI commands per Section 17.
- *Tests required:* CLI smoke tests; usability check by a team member who did not write the CLI.
- *Acceptance criteria:* every documented command runs and produces the documented output fields.
- *Evidence required:* CLI/API test report.
- *Failure condition:* any documented command that does not work as documented.

**Documentation [Sec. 29 set]**
- *Implementation requirement:* the full documentation set defined in Section 29.
- *Tests required:* cross-check every claim in every document against Section 24's evidence matrix.
- *Acceptance criteria:* no claim in any document lacks a traceable evidence path.
- *Evidence required:* the documents themselves, cross-referenced.
- *Failure condition:* any unsupported claim discovered during review.

**Final validation [all]**
- *Implementation requirement:* full regression suite passes across every subsystem above.
- *Tests required:* the complete adversarial + benchmark + verification + reproducibility suite, run end-to-end.
- *Acceptance criteria:* every claim in this document is reproducible on demand, live, by someone other than the original author.
- *Evidence required:* final validation run log.
- *Failure condition:* any claim in this document that cannot be reproduced on demand at review time.

---

## Section 7 — Mathematical Foundations

Implementation-level, not survey-level: the team must be able to write the
pivoting rule, not just recite what pivoting is. For each topic, the depth of
understanding required is marked.

**LP**
| Topic | Depth |
|---|---|
| Canonical/standard form conversion, slack variables | NEEDED TO IMPLEMENT |
| Basis, basic feasible solutions (BFS) | NEEDED TO IMPLEMENT |
| Revised simplex (product-form or LU-based basis, not naive tableau) | NEEDED TO IMPLEMENT |
| Reduced costs, entering/leaving variable rules | NEEDED TO IMPLEMENT |
| Ratio test | NEEDED TO IMPLEMENT |
| Duality (primal/dual relationship) | NEEDED TO IMPLEMENT (drives verification and MILP warm-starts) |
| Infeasibility / unboundedness certificates | NEEDED TO IMPLEMENT |
| Degeneracy, anti-cycling (Bland's rule) | NEEDED TO IMPLEMENT |
| Scaling (geometric-mean row/column) | NEEDED TO IMPLEMENT |
| Presolve reductions | NEEDED TO IMPLEMENT |
| Interior-point methods | NEEDED TO UNDERSTAND ONLY (Section 36, FUTURE) |

**MILP**
| Topic | Depth |
|---|---|
| LP relaxation, integrality gap | NEEDED TO IMPLEMENT |
| Branch-and-bound tree mechanics | NEEDED TO IMPLEMENT |
| Node selection (best-bound; depth-first as fallback) | NEEDED TO IMPLEMENT |
| Branching-variable selection (most-fractional) | NEEDED TO IMPLEMENT |
| Pruning (by bound, by infeasibility) | NEEDED TO IMPLEMENT |
| Incumbent tracking, gap formula $\text{gap} = \frac{|UB-LB|}{|UB|}$ | NEEDED TO IMPLEMENT |
| Gomory mixed-integer cuts | NEEDED TO UNDERSTAND, OPTIONAL ADVANCED (EXPERIMENTAL, Sec. 9) |
| Pseudocost branching | NEEDED TO UNDERSTAND, OPTIONAL ADVANCED |
| Rounding / feasibility-pump heuristics | NEEDED TO IMPLEMENT (rounding); NEEDED TO UNDERSTAND (feasibility pump, not required) |

**QP (convex only)**
| Topic | Depth |
|---|---|
| Convexity, PSD Hessian requirement | NEEDED TO IMPLEMENT |
| KKT stationarity / complementarity / dual feasibility | NEEDED TO IMPLEMENT |
| Active-set method (working-set updates) | NEEDED TO IMPLEMENT |
| Stopping conditions (KKT residual) | NEEDED TO IMPLEMENT |
| General non-convex QP | NEEDED TO UNDERSTAND ONLY (Section 36, FUTURE) |

**Numerical linear algebra**
| Topic | Depth |
|---|---|
| CSR and CSC formats, when each is preferred | NEEDED TO IMPLEMENT |
| Sparse LU factorization with partial pivoting | NEEDED TO IMPLEMENT |
| Fill-in and ordering (e.g., minimum-degree) | NEEDED TO IMPLEMENT |
| Iterative refinement | NEEDED TO IMPLEMENT |
| Condition-number estimation (not exact computation for large sparse systems) | NEEDED TO IMPLEMENT (estimate only) |
| Residual-based verification checks | NEEDED TO IMPLEMENT |
| Floating-point behavior (rounding, cancellation) | NEEDED TO UNDERSTAND, directly informs tolerance design (Section 11) |

---

## Section 8 — LP Engine Specification

**Decision: revised simplex**, with an LU-factorized basis representation,
updated via product-form updates between iterations rather than refactorizing
from scratch every step.

**Justification.** Not interior-point for the primary path: revised simplex
is more implementable correctly within available engineering time and gives
cleaner warm-starting for MILP's repeated child-node LP solves (interior-point
methods do not warm-start as naturally, since they do not preserve a discrete
basis between solves). Not naive tableau simplex: tableau simplex densifies
the working representation and does not scale to sparse industrial instances
— using it would silently violate the sparse-architecture requirement (R8/R9
in Section 1) even if labeled "simplex."

**Basis representation:** sparse LU factors of the current basis matrix,
updated by product-form (rank-one) updates after each pivot, with periodic
full refactorization to control numerical drift (refactorization policy:
after a fixed number of updates, or when a numerical-quality check — Section
11 — indicates drift).

**Entering variable rule:** Dantzig's rule (most negative reduced cost) by
default. **Leaving variable rule:** standard minimum-ratio test, with ties
broken by Bland's rule.

**Anti-cycling policy:** Bland's rule, when applied **unconditionally from iteration one**, provides a theoretical finite-termination guarantee (no cycling ever). In this implementation, Bland's rule is used only as a **fallback**, triggered when a cycling-detection watchdog (iteration count since the last basis-size-changing pivot exceeds a threshold) fires — it is not applied unconditionally from iteration one. The theoretical finite-termination guarantee therefore does **not** apply to this implementation. The claim is instead that the watchdog-triggered fallback provides **practical cycling resistance**, verified by the cycling-prone adversarial test case in Section 22. Unconditional Bland's rule is not applied because the per-iteration performance cost on non-degenerate instances — where it is never needed — would be disproportionate.

**Scaling:** geometric-mean row/column scaling applied before factorization
to reduce the coefficient-magnitude spread that causes ill-conditioning.

**Presolve integration:** LP solves the presolved model from [C]; the
reconstruction map restores the original-variable-space solution before
reporting.

**Stopping criteria:** all reduced costs within the optimality tolerance of
zero (Section 11) → OPTIMAL; no valid leaving variable on an improving
direction → UNBOUNDED; presolve or Phase-1 infeasibility → INFEASIBLE;
iteration/time limit reached without either → LIMIT_REACHED (never silently
reported as OPTIMAL).

**Warm starts:** dual simplex is added specifically to support MILP re-solves
after a bound change, since a child node's LP is numerically "close to" its
parent's optimal basis — **PLANNED as a Strong-tier (Category B) addition
once primal simplex is solid, not required for the minimum-viable path.**

**Refactorization policy:** full refactorization on a fixed update-count
interval and whenever a numerical-quality indicator (Section 11) flags
degradation, rather than on a purely fixed schedule alone.

**Implementation order:** (1) canonical form + presolve stub → (2) dense
correctness prototype on tiny instances → (3) swap in sparse CSR/CSC + LU
basis → (4) add scaling → (5) add Bland's-rule anti-cycling fallback → (6)
add dual simplex for warm-starts → (7) wire in the independent verifier.

**Pseudocode — revised simplex core loop:**
```
function REVISED_SIMPLEX(A, b, c, basis):
    while True:
        x_B = solve(B, b)                        # B = basis columns of A, sparse LU solve
        y = solve(B^T, c_B)                       # simplex multipliers
        reduced_costs = c_N - A_N^T * y
        if all(reduced_costs >= -opt_tol):
            return OPTIMAL, x_B, basis
        entering = select_entering(reduced_costs)     # Dantzig, or Bland's on cycling watchdog
        d = solve(B, A[:, entering])               # direction
        if all(d <= pivot_tol):
            return UNBOUNDED, None, None
        leaving, ratio = MIN_RATIO_TEST(x_B, d)     # ties broken by Bland's rule
        basis = update_basis(basis, entering, leaving)
        update_LU_factorization(basis)              # product-form update, not full refactor
        if update_count_since_refactor > threshold or numerical_quality_degraded():
            refactorize(basis)                       # periodic reset of accumulated drift
```

**Data structures:** sparse CSC basis matrix; LU factors stored sparse;
dense working vectors sized to `m` (constraint count) for `x_B`, `y`;
sparse `A_N` accessed column-wise for reduced-cost computation.

**Failure modes:** ratio-test ties on degenerate problems (handled by the
Bland's-rule fallback above, not by ignoring the tie); accumulated numerical
drift from too many product-form updates without refactorization (handled by
the refactorization policy above); near-singular basis during factorization
(flagged by [E], never silently solved through).

**Tests / acceptance criteria:** as specified in Section 6's LP Definition of
Done.

**Dual simplex — when added:** as a Category B (Strong SIH Target) item,
specifically for MILP child-node re-solves after a single bound change, where
the parent's optimal basis is dual-feasible for the child and only a small
number of dual-simplex iterations are needed to restore primal feasibility —
this is the concrete mechanism behind the "measurable MILP node-solve
speedup from warm-starting" exit criterion in Section 32's Phase 7.

---

## Section 9 — MILP Engine Specification

Treated as its own major subsystem, built strictly on top of a working LP
engine — every B&B node re-solves an LP via Section 8's engine, so LP
correctness and speed directly gate MILP feasibility.

**Node data structure:** a node holds the parent's bound modifications (a
list of variable-bound tightenings relative to the root, not a full copy of
the model), a reference to the warm-startable basis inherited from the
parent, and cached LP-relaxation results once solved.

**Tree management:** a priority queue keyed by best-bound (parent LP
relaxation objective), giving better final-bound quality than pure
depth-first at some added memory cost; a node cap and fallback to depth-first
exploration is defined for memory-pressure situations, so the tree does not
grow without bound on hard instances.

**MVP branching:** most-fractional variable selection (the integer-
constrained variable whose LP-relaxation value is furthest from an integer).
**Strong-tier branching:** pseudocost branching, added only once the MVP
branching rule is verified correct and stable — pseudocost branching is
explicitly **not** the starting point, because it is an optimization of a
selection mechanism that must first be provably correct.

**MVP node selection:** best-bound, with the depth-first fallback above under
memory pressure. **Strong-tier node selection:** not planned beyond this —
more elaborate node-selection strategies (e.g., hybrid best-first/depth-first
switching heuristics) are EXPERIMENTAL, attempted only if MVP node selection
is demonstrably a bottleneck on real MIPLIB instances.

**MVP heuristics:** a simple rounding heuristic applied to the LP-relaxation
solution at select nodes, to establish or improve an incumbent early — enough
to demonstrate primal-bound improvement without building a feasibility-pump-
grade system. **Stretch heuristics:** feasibility pump, EXPERIMENTAL, not
committed.

**Stretch cuts:** Gomory mixed-integer cuts only, as an explicit
EXPERIMENTAL stretch — never a core requirement. **Why not more:** a full
cutting-plane arsenal (mixed-integer rounding cuts, cover cuts, clique cuts,
etc.) is each individually a substantial implementation effort with its own
numerical-stability considerations (cuts can themselves introduce
ill-conditioning if not managed carefully); attempting several under a
hackathon timeline risks weakening the core B&B correctness guarantee for a
feature that is not PS-mandatory beyond "may include."

**Justification for why each selected algorithm is worth its implementation
cost:** best-bound node selection is chosen over pure depth-first because it
directly improves the reported optimality gap quality, which is the
verification-facing metric judges will check (Section 12); most-fractional
branching is chosen as MVP because it is simple to implement correctly and
verify, which matters more at this stage than the marginal node-count
reduction pseudocost branching would provide; the rounding heuristic is
chosen because a single, simple heuristic materially improves demo-time
incumbent quality on MIPLIB instances at very low implementation risk,
compared to the higher-risk feasibility-pump approach.

**Pseudocode — branch-and-bound skeleton:**
```
function BRANCH_AND_BOUND(root_LP):
    queue = PriorityQueue(key = best_bound)
    queue.push(root_LP)
    incumbent = None
    while queue not empty and not time_limit_exceeded() and not node_limit_exceeded():
        node = queue.pop()
        status, x, obj = solve_LP_relaxation(node)      # warm-started via dual simplex where possible
        if status == INFEASIBLE: continue
        if incumbent and obj is worse than incumbent.obj: continue     # bound-based pruning
        if is_integer_feasible(x, integrality_tol):
            if incumbent is None or obj better than incumbent.obj:
                incumbent = (x, obj)
            continue
        branch_var = select_branching_variable(x)        # most-fractional (MVP)
        left, right = create_child_nodes(node, branch_var)
        queue.push(left); queue.push(right)
    gap = compute_optimality_gap(incumbent, queue)         # honest even on timeout
    status = OPTIMAL if gap == 0 else (LIMIT_REACHED if incumbent else INFEASIBLE_OR_UNKNOWN)
    return incumbent, gap, status
```

**Parallel node processing:** an EXPERIMENTAL, Strong-tier-adjacent stretch,
not MVP — thread-safe node-queue access and per-thread LU-factorization state
are real added complexity; sequenced strictly after single-threaded
correctness is solid (Section 14 defines the threading model if attempted).

---

## Section 10 — QP Engine

Scoped to **convex QP only** for the initial and only committed target; general
non-convex QP is explicitly FUTURE work (Section 36) unless evidence during
development changes this decision.

**Mathematical formulation:** minimize $\tfrac{1}{2}x^TQx + c^Tx$ subject to
$Ax \le b$, $x \ge 0$ (and equality constraints where present), with $Q$
required positive semidefinite for the convexity guarantee this method
depends on.

**PSD check:** the Hessian $Q$ is checked for positive-semidefiniteness
before attempting to solve (a Cholesky-attempt or eigenvalue-sign check on a
representative subspace); a non-PSD $Q$ is **rejected and flagged**, never
silently solved as if convex — a non-convex QP solved via a convex method
would produce a meaningless local result reported with unwarranted
confidence, which is exactly the class of false-"optimal" failure this
project's verification discipline exists to prevent.

**Algorithm: active-set method**, chosen because it is more implementable
correctly within available time than interior-point QP, and it reuses the
same KKT-linear-system-solving machinery ([E]'s sparse LU/refinement layer)
that the LP engine already needs — sharing infrastructure rather than
building a second, unrelated numerical stack.

**Working-set mechanics:** start from a feasible point (or Phase-1
feasibility step analogous to LP); at each iteration, solve the equality-
constrained QP subproblem over the current active (working) set via the KKT
system; check Lagrange-multiplier signs to decide whether to remove a
constraint from the working set, or take a full/partial step and add a newly-
active constraint.

**Stopping criteria:** KKT residual (stationarity, primal feasibility, dual
feasibility, complementary slackness, each checked against Section 11's named
tolerances) below the documented threshold.

**Numerical tolerances:** the same named-tolerance taxonomy as LP (Section
11), with a QP-specific KKT-residual tolerance as the primary stopping/
acceptance gate.

**Verification:** the independent verification layer ([Y], Section 12)
recomputes the full KKT residual for every reported QP solution, exactly as
it does for LP feasibility/duality and MILP integrality — QP gets no
exemption from the verification discipline.

**Failure cases:** non-PSD Hessian (rejected at input, see above); working-
set cycling on degenerate QPs (handled analogously to LP's Bland's-rule
fallback — an anti-cycling safeguard on the active-set updates); numerical
singularity in the KKT system (flagged by [E], never silently solved
through).

**Definition of Done:** as specified in Section 6's QP entry.

**Sequencing note, stated directly per the PS's own class ordering:** QP is
explicitly sequenced *after* LP and MILP are stable. The PS lists LP and
MILP as the primary initial focus; a QP layer built or stabilized before the
LP/MILP core is solid would draw engineering time away from the two
higher-priority, PS-mandatory classes.

---

## Section 11 — Numerical Robustness

A formal, named-tolerance numerical policy — never one global epsilon.

**Tolerance taxonomy:**

| Tolerance | Purpose | Applies to |
|---|---|---|
| `primal_feasibility_tol` | How far $Ax - b$ may deviate from satisfying constraints | LP, MILP relaxations, QP |
| `dual_feasibility_tol` | How far reduced costs may deviate from dual-feasibility sign conditions | LP, QP |
| `integrality_tol` | How far an integer-constrained variable's value may deviate from an integer | MILP |
| `optimality_tol` | Reduced-cost / stationarity threshold for declaring OPTIMAL | LP, QP |
| `pivot_tol` | Threshold below which a pivot element is treated as numerically zero | LP factorization/updates |
| `residual_tol` | Threshold for the independent verifier's recomputed residuals | LP, MILP, QP (verification layer) |

Each is a documented, tunable, named constant — never an unlabeled inline
magic number — and every benchmark run logs its exact tolerance
configuration alongside the result (Section 23).

**Scaling:** geometric-mean row/column scaling applied before factorization,
reducing the coefficient-magnitude spread that drives ill-conditioning; this
is a well-known, directly implementable equilibration technique, not a
novel contribution claimed by this project.

**Near-zero pivots and singularity:** detected during LU factorization
against `pivot_tol` and **flagged, never silently proceeded through** — a
factorization that would require dividing by a near-zero pivot returns a
NUMERICAL_FAILURE status rather than producing a numerically meaningless
result reported with false confidence.

**Ill-conditioning:** condition-number is *estimated* (never computed exactly
for large sparse systems, which is itself computationally as expensive as
solving) as a diagnostic signal feeding the Solver Strategy Layer's scaling/
presolve-level decisions (Section 16).

**Degeneracy:** handled at the algorithm level (Bland's-rule anti-cycling
fallback in LP, Section 8; anti-cycling safeguard in QP's active-set updates,
Section 10) rather than purely at the tolerance level — tolerances alone
cannot fix a cycling algorithm.

**Iterative refinement:** applied after each linear-system solve in [E] to
recover accuracy lost to floating-point round-off, particularly important on
ill-conditioned or poorly-scaled instances; refinement's effect is itself
measured and reported (residual before vs. after refinement) on the
adversarial ill-conditioned test cases (Section 22).

**Residual checking:** every reported solution has its constraint and
(where applicable) KKT residuals checked against `residual_tol` by the
independent verifier ([Y]) before the solver is permitted to report a
definitive status.

**Numerical warnings and failure reporting:** the solver distinguishes
between "solved, verified" and "solved, but verification flagged a
concern" — the latter is reported honestly as a warning attached to the
result, not silently upgraded to a clean OPTIMAL.

**Solver status taxonomy** (used consistently across LP, MILP, QP):

| Status | Meaning |
|---|---|
| `OPTIMAL` | Solved and independently verified against feasibility + optimality conditions within tolerance |
| `FEASIBLE` | A feasible solution exists and was found, but optimality is not (yet) certified — e.g., MILP with nonzero gap at a time/node limit |
| `INFEASIBLE` | Proven infeasible (LP: via a Farkas-type certificate; MILP: exhaustive node search with no feasible integer point found) |
| `UNBOUNDED` | Proven unbounded (an improving, feasible ray exists) |
| `LIMIT_REACHED` | Time or iteration/node limit hit before OPTIMAL or a definitive INFEASIBLE/UNBOUNDED could be established |
| `NUMERICAL_FAILURE` | A numerical safeguard (near-zero pivot, residual above tolerance, non-PSD QP input) prevented a trustworthy result |
| `UNKNOWN` | None of the above could be established — used sparingly and always logged for investigation, never silently discarded |

**Rule, stated once and enforced everywhere: the solver must never report
`OPTIMAL` unless the independent verifier ([Y]) has confirmed it.** A
solver-internal claim of optimality that fails verification is downgraded to
`FEASIBLE` (if a feasible point exists) or `NUMERICAL_FAILURE` (if even
feasibility is in doubt), and the discrepancy is logged, not hidden.

---

## Section 12 — Independent Verification Engine

**Why this subsystem exists.** LP/MILP/QP solving via floating-point
arithmetic is well known to produce confidently wrong answers on
ill-conditioned or degenerate problems if a solver's own internal bookkeeping
is trusted uncritically. A solver that reports "optimal" on a subtly
infeasible or unbounded problem is worse than one that honestly reports "I
don't know" — this is specifically the class of failure judges in this space
probe for, and it is why this document treats the verification layer as a
**first-class subsystem**, not an afterthought bolted onto the solver.

**Design principle: never trust the solver's own status claim.** Before any
result is returned to the caller, the verification layer independently
recomputes the conditions that actually define correctness for that problem
class.

**LP verification recomputes:**
- Primal feasibility — substitute the reported solution into $Ax \le b$ (and
  equality rows) directly and check the residual.
- Dual feasibility — check the reported dual values satisfy sign conditions
  on reduced costs.
- Complementary slackness — check that for every constraint, either the
  slack is zero or the corresponding dual value is zero (within tolerance).
- Objective — recompute $c^Tx$ directly from the reported solution and
  compare to the solver's reported objective value.

**MILP verification recomputes:**
- Primal feasibility of the final incumbent (same check as LP).
- Integrality — every integer-constrained variable's value checked against
  `integrality_tol`.
- Objective — recomputed directly, as in LP.
- Bounds and gap consistency — the reported upper/lower bound and gap are
  recomputed from the final incumbent and the best remaining node bound, not
  taken as-given from the solver's internal tracking.

**QP verification recomputes:**
- Primal feasibility (as in LP).
- Full KKT residual (stationarity, dual feasibility, complementary
  slackness) computed directly from the reported primal/dual solution and
  the original $Q$, $c$, $A$, $b$ — not from any intermediate quantity the
  active-set solver computed along the way.

**Independence requirement, stated precisely.** The verification code path
must be written as a **separate implementation** from the solving code path
for every check it performs — it must not call into, or share a bug-prone
common subroutine with, the solver's own residual/feasibility computations.
If both paths shared a bug, the verification would be worthless; independence
is the entire point of this subsystem, and it is treated as a hard
architectural constraint, not a style preference.

**Verifier inputs:** the original (pre-presolve) canonical model, and the
solver's reported solution/status/duals (where applicable).

**Verifier outputs:** a pass/fail per check above, the recomputed residuals
themselves (not just a boolean), and a final adjusted status per Section 11's
taxonomy (which may downgrade the solver's self-reported status).

**Tolerances used:** `residual_tol` and the problem-class-specific
tolerances from Section 11's taxonomy, applied identically regardless of
which internal tolerance the solver itself used — the verifier's tolerance
policy is not permitted to silently relax to match whatever the solver
happened to produce.

**Test strategy:**
- Standard-case tests: every LP/MILP/QP test instance's reported solution is
  run through the verifier as part of normal test execution — verification
  is not an optional post-processing step, it runs by default.
- **Deliberate fault-injection test:** a specifically-built test configuration
  that deliberately injects a wrong "optimal" claim (e.g., a mutated
  objective value, or a solution nudged outside feasibility) and asserts
  that the verifier catches it. This is the concrete, reproducible mechanism
  behind the "verification layer catching a deliberately injected wrong
  answer live" demo item in Sections 25 and 28.

**Acceptance criteria:** every fault in the fault-injection suite is caught;
every standard-case test instance's genuinely-correct solution passes
verification; zero shared logic between the solver and verifier code paths
for any check the verifier performs (checked at code-review time, not just at
test time).

---

## Section 13 — Sparse Numerical Core

**CSR (Compressed Sparse Row):** row-major access pattern — efficient for
row-oriented operations (e.g., computing $Ax$ directly). **CSC (Compressed
Sparse Column):** column-major access pattern — efficient for column-oriented
operations (e.g., accessing $A_N$ columns during reduced-cost computation,
and for LU factorization, which is naturally column-access-heavy). Both
representations are maintained where the access pattern justifies the
conversion cost; the model layer defaults to CSC for factorization-facing
code and CSR for direct constraint-residual checks.

**Vector representation:** dense vectors for the working simplex quantities
($x_B$, $y$, reduced costs) since these are sized to `m` or `n` and are not
themselves sparse in general, even when $A$ is sparse.

**Sparse matrix-vector multiplication (SpMV):** standard compressed-format
SpMV — iterate nonzero entries per row (CSR) or per column (CSC) and
accumulate; this is also the operation batched for the GPU kernel in Section
15.

**Sparse triangular solve:** forward/backward substitution restricted to the
nonzero structure of the triangular factors, exploiting sparsity in $L$ and
$U$ rather than treating them as dense.

**LU factorization:** sparse Gaussian elimination with partial pivoting,
producing $L$ and $U$ such that $PA = LU$ (with $P$ a row-permutation
matrix from pivoting).

**Pivoting:** partial pivoting for numerical stability (choosing the
pivot with largest magnitude in the eligible column), balanced against
fill-in considerations — a pivot choice that is numerically ideal but
causes severe fill-in is a real trade-off this component must manage, not
ignore.

**Fill-in and ordering:** eliminating a sparse system in an arbitrary
order can turn a sparse matrix into a dense one during factorization
("fill-in"). A fill-in-reducing ordering heuristic (e.g., minimum-degree
ordering) is applied before factorization to keep the factors sparse.

**Refactorization:** product-form (rank-one) updates are applied between
simplex iterations to avoid a full refactorization on every pivot; a full
refactorization is triggered periodically (update-count threshold) and on
numerical-quality degradation (Section 11), to bound the accumulated drift
from repeated approximate updates.

**Memory model:** nonzero-count-driven allocation (not `rows × cols`);
memory usage is measured directly, not assumed, as part of the Definition of
Done for this component (Section 6).

**Data structures and invariants:** CSR/CSC arrays (`values`,
`col_indices`/`row_indices`, `row_ptr`/`col_ptr`) with the invariant that
`row_ptr`/`col_ptr` is strictly non-decreasing and indices within each
row/column segment are maintained in a consistent (typically ascending)
order to support efficient merge-style operations during factorization.

**Performance metrics tracked:** memory footprint vs. nonzero count;
factorization time vs. instance size; fill-in ratio (nonzeros in factors
vs. nonzeros in original matrix) — tracked as diagnostic/benchmark data, not
promoted to marketing numbers without the full benchmark context of Section
18.

---

## Section 14 — CPU Parallelism

**CPU correctness vs. CPU parallelism vs. CPU optimization — kept as three
separate concerns, in that priority order.** A parallel implementation that
is faster but occasionally produces a different (or wrong) result is not
acceptable; correctness always gates parallelism, and parallelism always
gates hand-optimization (no speculative SIMD/threading without a profiler
showing a specific hot path justifies it).

**Where multicore processing is safe (identified, not assumed):**
- Batched SpMV across independent right-hand sides — the operation
  identified in Section 13 as genuinely data-parallel with no cross-batch
  dependency.
- Independent B&B child-node LP relaxation solves, *if* parallel B&B is
  attempted (EXPERIMENTAL, Section 9) — each node's LP solve is
  independent of sibling nodes given the parent's warm-start basis.

**Where multicore processing is explicitly not applied, and why:** simplex
pivoting itself within a single LP solve (inherently sequential — each
pivot depends on the previous one); branch-and-bound tree traversal in its
single-threaded form (irregular, latency-bound, not a good parallel-
decomposition target without the added complexity of a thread-safe shared
node queue). Recommending parallelization here without a specific,
profiled technical reason would violate this document's own "do not
recommend parallelizing inherently sequential operations without a
technical reason" standard.

**Thread pool:** a fixed-size thread pool used for the two data-parallel
workloads identified above; not used to speculatively parallelize
components without a demonstrated data-parallel structure.

**Batch operations:** the granularity of parallel work is a full SpMV (or a
full child-node LP solve, if parallel B&B is attempted) per task — coarse
enough to amortize thread-scheduling overhead, not so coarse that load
balancing across cores becomes uneven on realistic problem sizes.

**Synchronization / data ownership:** for batched SpMV, each thread owns a
disjoint output range — no shared mutable state, no locking required for
correctness. For parallel B&B (EXPERIMENTAL), a mutex-protected shared node
queue is the accepted starting design; a lock-free queue is a further
optimization not attempted before the mutex-protected version is proven
correct.

**Deterministic mode:** a `--seed` flag and, where parallel execution could
otherwise introduce nondeterministic tie-breaking (e.g., in node selection
order under parallel B&B), a deterministic mode that forces single-threaded
or ordered execution — reproducibility (Section 23) takes priority over
parallel throughput when the two are in tension and the mode is requested.

**Race prevention:** the two identified parallel workloads are chosen
*because* they decompose into disjoint-output tasks with no shared mutable
state in their base form — this is a design choice made specifically to
avoid needing complex synchronization primitives in the MVP, not an
afterthought.

---

## Section 15 — GPU Strategy

**Principle stated once, applied throughout this document: Kavach-Opt is
never described as "a GPU solver."** It is a CPU-first solver with one
scoped, measured GPU-accelerated kernel. Describing the whole product as
GPU-accelerated when only one kernel is would misrepresent the actual
engineering, and would not survive the kind of live scrutiny described in
Section 26.

**CPU-preferred workloads (GPU not attempted here, with reasons):**
- Simplex pivoting itself — sequential, branch-heavy, small working set per
  iteration; GPU offers no advantage here and would add transfer overhead
  that would net-negative the runtime.
- Branch-and-bound tree traversal — irregular, latency-bound, a poor fit for
  GPU's throughput-oriented, regular-structure execution model.

**GPU-candidate workloads:**
- **Batched sparse matrix-vector product (SpMV)** — the one kernel this
  project actually builds and measures. Why GPU may help: SpMV batched
  across multiple independent right-hand sides (arising naturally from
  evaluating multiple B&B child-node relaxations' matrix-vector operations,
  or from a batch of iterative-refinement passes across nearly-identical
  bases) is data-parallel and throughput-oriented — a good structural match
  for GPU execution. Parallelism type: data-parallel across batch items
  and, within an item, across nonzero entries. Data size requirement:
  benefit is expected to appear only once matrix size and/or batch count
  exceed a threshold large enough to amortize transfer cost — this
  threshold is **measured, not assumed** (Section 18's crossover-point
  methodology). Transfer cost: host-to-device and device-to-host transfer
  time is included in every reported GPU timing number, never excluded to
  make the number look better. Correctness requirement: GPU and CPU results
  must agree within `residual_tol` on every test case (the GPU/CPU
  equivalence test in Section 6). Benchmark requirement: the full
  problem-size sweep methodology below. Fallback: CPU SpMV is always
  available and is the default; `--no-gpu` always works.

**Hybrid workloads (THEORETICAL for this project):** dense linear-algebra
sub-kernels inside the sparse factorization (small dense blocks arising from
fill-in) could in principle use GPU dense kernels. This is explicitly
**THEORETICAL** — real benefit only appears at scales this solver is
unlikely to reach within this project's timeframe, and it is not built
unless the SpMV kernel above is fully done, measured, and time remains.

**GPU acceptance criterion — stated as a hard gate, not a target:** a GPU
claim is valid **only** when benchmarked under all of the following
controlled conditions simultaneously: same instance/problem across both
runs; same numerical result (within `residual_tol`) on CPU and GPU; same
hardware, disclosed explicitly in the report; CPU runtime reported;
GPU compute-only runtime reported; GPU transfer-inclusive runtime reported
separately; problem size reported; speedup ratio computed from the
transfer-inclusive GPU time (never from compute-only time alone, which would
overstate real-world benefit); and the crossover point — the problem size
below which GPU is slower than CPU — explicitly identified and reported.

**No fabricated speedup target.** This document does not state a number
(e.g., "2x," "5x") that the GPU kernel must hit. The only committed claim is
methodological: *whatever the measured speedup and crossover point turn out
to be, they will be reported exactly as measured*, including the outcome
"GPU is slower below this problem size" — which is an explicitly permitted
and expected finding, not a failure to hide.

**CUDA backend design:** a narrow `ExecutionBackend` interface with CPU and
CUDA implementations of the batched-SpMV operation specifically — not a
general "everything can run on GPU" abstraction, which would overpromise
scope beyond what is actually built. CPU is the default backend; GPU is
strictly opt-in via a flag, and the solver's correctness and every other
Definition of Done in Section 6 hold with GPU absent or disabled.

---

## Section 16 — Solver Intelligence / Strategy Layer

A rule-based (explicitly **not** ML-based) layer that inspects measurable
problem characteristics and chooses solver settings accordingly — added
because a single fixed configuration is known to be suboptimal across the
range of instance types (Netlib-style small/well-conditioned vs. MIPLIB-
style large/degenerate), and because a rule-based strategy is directly
testable and explainable to a judge, unlike a black-box ML component would
be.

**Signals inspected (all directly computable from the parsed/presolved
model, no learning required):**
- Number of variables, number of constraints
- Sparsity (nonzero density)
- Integer-variable ratio (fraction of variables that are integer-constrained)
- Coefficient range (ratio of largest to smallest nonzero magnitude — a
  direct ill-conditioning indicator)
- Conditioning indicators (the estimated condition number from Section 11)
- Problem class (LP / MILP / QP, and, within MILP, whether the instance
  looks integer-dense vs. integer-sparse)

**Decisions this layer makes:**
- Presolve aggressiveness level (light vs. full reduction, weighed against
  presolve's own runtime cost on already-small instances)
- Whether to apply scaling, and how (skipped when the coefficient range is
  already tight, to avoid unnecessary overhead)
- LP algorithm parameters (e.g., whether to prefer dual-simplex warm start
  for MILP child nodes, once dual simplex is implemented per Section 8)
- Branching strategy selection (most-fractional vs., once implemented,
  pseudocost)
- Node strategy (best-bound vs. depth-first fallback under detected memory
  pressure — Section 9)
- CPU/GPU routing for the batched-SpMV kernel (only offered to GPU when
  batch size/matrix size exceeds the *measured* crossover point from Section
  15 — this layer is the concrete mechanism that applies that measured
  threshold at solve time)
- Numerical safeguard aggressiveness (e.g., more frequent refactorization on
  instances flagged as poorly conditioned)

**Explicit boundary, stated directly:** this is **RULE-BASED ENGINEERING** —
a documented, deterministic mapping from measured problem characteristics to
solver settings, fully inspectable and testable. **OPTIONAL FUTURE ML** (a
learned model replacing or augmenting these rules) is listed only in Section
36 as a future extension and is **not** part of this project's committed
scope. ML is not introduced here for buzzword value; the rule-based version
is both sufficient for the identified decisions and far easier to verify
correct.

---

## Section 17 — Input / API / CLI

**Input formats:**
- **MPS** (fixed or free format) — the well-documented, decades-old
  industry-standard text format that Netlib and MIPLIB instances are
  distributed in; supporting it is required simply to use those benchmark
  sets at all.
- **Native LP format** — a simpler, easier-to-hand-write secondary input
  path, used primarily for the MRPL industrial demo model (Section 21) where
  hand-authoring an MPS file would be unnecessarily tedious.

**Canonical internal representation:** sparse matrix + bound vectors +
objective + integrality mask, as defined in Section 4's component [A].

**Flow:** file → parser [B] → canonical model [A] → presolve [C] → solver
([F]/[G]/[H]) → verifier [Y] → reported solution.

**Native API (illustrative shape, not final signatures):** a `Model` object
constructed either from a parsed file or programmatically; a `solve()` call
accepting solver-class-specific options (time limit, tolerance overrides,
GPU opt-in); a `Result` object exposing status (per Section 11's taxonomy),
objective value, primal/dual solution vectors, verification report, and
solve telemetry (iteration/node counts, timing).

**CLI commands (design, illustrative):**
```
kavach-opt solve model.mps
kavach-opt solve model.mps --problem-type milp
kavach-opt solve model.mps --problem-type qp
kavach-opt solve model.mps --gpu
kavach-opt solve model.mps --no-gpu
kavach-opt solve model.mps --time-limit 300 --seed 42
kavach-opt benchmark --suite netlib
kavach-opt benchmark --suite miplib-subset --compare highs
kavach-opt verify solution.json
kavach-opt presolve model.mps --report
```

**Expected output fields (solve command):** status (per Section 11's
taxonomy), objective value, solve time, iteration count (LP) or node count
and optimality gap (MILP), verification result (pass/fail + recomputed
residuals), solver configuration used (tolerances, seed, CPU/GPU backend),
and — for GPU-enabled runs — the transfer-inclusive timing breakdown from
Section 15.

---

## Section 18 — Benchmark Framework

**Sources:** Netlib LP set; a size-appropriate subset of MIPLIB (explicitly
not the largest, hardest MIPLIB instances — see Section 19's honesty
discipline on this point); Mittelmann benchmark instances as an additional,
non-required source if time permits; QPLIB where applicable, conditional on
the QP engine reaching stability with time remaining (per Section 1's R16).

**Instance categories:**
| Category | Purpose |
|---|---|
| Small | Fast correctness/regression signal |
| Medium | Representative of realistic small-industrial scale |
| Large | Scalability/size-ceiling identification |
| Sparse | Confirms the sparse architecture's actual benefit vs. a dense baseline |
| Degenerate | Stresses the anti-cycling fallback |
| Ill-conditioned | Stresses scaling + iterative refinement |
| Difficult MILP | Stresses branch-and-bound node management, honest gap reporting |
| Convex QP | Stresses the active-set engine and KKT verification |

**Fields recorded per experiment (no exceptions, every attempted instance):**
instance name; variable count; constraint count; nonzero count; density;
objective value; status (Section 11 taxonomy); runtime; iteration count (LP)
or node count (MILP); memory usage; optimality gap; primal residual (from
the independent verifier, never the solver's self-report); dual residual
(where applicable); verification result; CPU or GPU backend used; hardware
specification; compiler and build mode; exact tolerance configuration;
random seed.

**Explicit rule: no fake aggregate.** No single "average performance"
number is reported in a way that hides individual instance failures — every
aggregate statistic in any report must be traceable back to the individual
per-instance rows that produced it, and any instance excluded from an
aggregate for a stated reason (e.g., a different problem class) must be
disclosed, not silently dropped.

---

## Section 19 — Benchmark Acceptance Criteria

**Correctness-first criteria, not arbitrary performance claims:**

| Class | Acceptance criterion |
|---|---|
| LP | Objective agreement with the independently-verified recomputation within `optimality_tol`; verified primal/dual feasibility; correct status classification (OPTIMAL/INFEASIBLE/UNBOUNDED/LIMIT_REACHED) on every instance |
| MILP | Verified feasibility and integrality of the reported incumbent; honestly-reported optimality gap, including on timeout; correct status classification |
| QP | KKT residual within `residual_tol` on the independently-verified recomputation; correct rejection of any non-PSD input |
| Numerical robustness | Every adversarial-suite case (Section 22) produces its documented expected behavior, not a crash or silent wrong answer |
| GPU | The full acceptance gate defined in Section 15 (same-hardware, transfer-inclusive, honest crossover point) |
| CPU scaling | Measured (not assumed) speedup on the batched-SpMV workload, reported honestly including cases with negligible or no speedup at small sizes |

**What this document explicitly does NOT claim as an acceptance criterion:**
statements of the form "must beat HiGHS by 2x" or any other fixed
competitive-margin target. Such a target would be arbitrary, not tied to
correctness, and not achievable-with-integrity by a from-scratch student
implementation against a mature, heavily-optimized commercial-grade
open-source solver.

**Instead, the benchmark measures performance rather than assuming success:**
runtime and quality are recorded and reported for every attempted instance
regardless of outcome; problem size is increased within each category until
the actual size ceiling — the point where the solver times out, fails
numerically, or exceeds the demo-relevant time budget — is identified and
reported honestly as part of the results, not hidden or extrapolated past.

---

## Section 20 — Reference Solver Comparison

**Reference solver: HiGHS** — open-source, actively maintained, well-
regarded, straightforward to run standalone for an arms-length comparison.
Using HiGHS as an external comparison target is categorically different from
wrapping or embedding it: HiGHS is invoked as a **separate process**, purely
for evaluation, and is never imported, linked, or called from within
Kavach-Opt's solving path — this distinction is what keeps the comparison
compliant with the PS's explicit ban on building the sovereign solver upon
any existing solver library.

**Comparison protocol:**
- Same problem instances for both solvers, no exceptions.
- Same hardware where practical (both runs on the same machine in the same
  session).
- Same time limits and documented tolerances for both solvers.
- No cherry-picking — the full instance set attempted is the full instance
  set reported.
- Failures and timeouts are reported for both solvers, not just for
  Kavach-Opt.
- Objective quality compared directly (with the verified objective values,
  not either solver's raw self-report).
- Feasibility compared (both solutions independently checked against the
  original constraints).
- Runtime compared, disclosed per-instance, not only as an aggregate.
- Optimality gap compared where either solver reports a nonzero gap.

**HiGHS is architecturally optional, not mandatory, where reasonable:** the
comparison harness treats the reference solver as a configurable, swappable
component (`--compare <solver>`), so the comparison protocol itself does
not architecturally depend on HiGHS specifically being present — HiGHS is
simply the chosen default reference given its accessibility and
respectability as a comparison point, per PS's requirement of comparison
against "at least one established commercial or open-source solver."

---

## Section 21 — Industrial MRPL Demo

**Model: crude/product blending and unit-capacity allocation** — a classic,
well-understood LP/MILP structure directly recognizable to MRPL's own
refining domain, built from published, generic refinery-blending
formulations as the structural basis, **not** from proprietary MRPL data
(none is assumed available). All non-proprietary data used is clearly
labeled **illustrative**.

**Variables:**
- Continuous: blend fractions of $n$ crude/intermediate streams into $m$
  product pools.
- Binary/integer: unit on/off or capacity-tier selection decisions — this is
  what gives the model genuine MILP structure rather than a disguised pure
  LP.

**Objective:** maximize blended-product value minus feedstock and processing
cost.

**Constraints:**
- Mass balance across pools (inflow/outflow consistency for each blend
  pool).
- Product quality specification bounds — a weighted-average constraint per
  quality attribute (e.g., sulfur content, octane rating), which is
  precisely what makes blending problems classically hard and interesting
  as an LP/MILP structure, not an arbitrary complication added for demo
  effect.
- Unit capacity limits.
- Binary unit-selection logic (e.g., a unit's throughput bound is only
  active if its on/off binary variable is 1).

**Why this structure is industrially relevant to MRPL specifically:** crude
blending and unit-capacity allocation are core, recurring refinery planning
problems — this is not a generic textbook LP repurposed for the demo, it is
the same problem *class* MRPL's own planning workflows solve today with
commercial solvers, at a size Kavach-Opt can credibly and verifiably handle.

**Demonstration flow:** business problem (which crudes to blend, which
units to run, at what levels) → mathematical model (the variables/objective/
constraints above) → solver (LP relaxation, then MILP for the binary
unit-selection layer) → verified result (independent verification layer
confirms feasibility, integrality, and objective) → operational
interpretation (what the blend fractions and unit selections mean in
refinery-operations terms, stated in plain language for the non-specialist
judge audience identified in Section 0).

---

## Section 22 — Adversarial Testing

A deliberate adversarial suite, each case with a defined expected behavior,
test method, failure interpretation, and fallback.

| Adversarial case | Expected behavior | Test method | Failure interpretation | Fallback |
|---|---|---|---|---|
| Infeasible LP | Report `INFEASIBLE` with a certificate | Hand-constructed and real infeasible Netlib-derived instances | Reporting OPTIMAL or looping is a critical bug | None — this must simply be correct |
| Unbounded LP | Report `UNBOUNDED` | Hand-constructed unbounded instances | Reporting OPTIMAL or crashing is a critical bug | None |
| Degenerate LP | Solve correctly without excessive iteration blow-up | Known degenerate Netlib-style instances | Excess iterations beyond a documented bound flags a tuning issue, not silently accepted | Bland's-rule fallback engages (Section 8) |
| Cycling-prone case | Must not cycle indefinitely | A known classically cycling-prone constructed instance | Any observed cycle is a critical bug | Bland's-rule anti-cycling fallback |
| Ill-conditioned matrix | Solve with refinement, or flag instability — never silently return garbage | Instances with a wide coefficient-magnitude spread | An unflagged, out-of-tolerance result is a critical bug | Scaling + iterative refinement (Section 11); `NUMERICAL_FAILURE` status if refinement is insufficient |
| Near-singular basis | Flag rather than silently factorize through | Constructed near-singular basis matrices | An unflagged factorization through a near-zero pivot is a critical bug | `NUMERICAL_FAILURE` status |
| Tiny coefficients | Correct handling under scaling | Constructed instances with very small nonzero magnitudes | Silent underflow-driven wrong result | Scaling (Section 11) |
| Huge coefficients | Correct handling under scaling | Constructed instances with very large nonzero magnitudes | Silent overflow or precision-loss-driven wrong result | Scaling (Section 11) |
| Integer-infeasible MILP | Report `INFEASIBLE` after exhaustive node search | Hand-constructed integer-infeasible instances | Reporting a feasible/optimal result is a critical bug | None — must be correct |
| Weak LP relaxation | Correct (if slow) solving, stresses B&B node count | Known weak-relaxation MIPLIB-style instances | Node explosion without a reported honest gap is a failure | Time/node limit + honest `LIMIT_REACHED` reporting |
| Node explosion | Time/node limit reached, honest gap reported, never a false OPTIMAL | Deliberately hard MILP instances | Reporting OPTIMAL past the limit is a critical bug | `LIMIT_REACHED` with best-found incumbent and honest gap |
| Timeout | Best-found result reported honestly with gap, not disguised as final | Any instance run against a tight artificial time limit | Reporting OPTIMAL on timeout is a critical bug | `LIMIT_REACHED` status |
| GPU/CPU result mismatch | Results agree within `residual_tol` | Direct GPU-vs-CPU equivalence test on the batched-SpMV kernel | Any mismatch beyond tolerance is a critical bug | Solver defaults to CPU (`--no-gpu`) if GPU is untrusted |

---

## Section 23 — Reproducibility

A formal engineering property, not an afterthought.

- **Fixed seeds** wherever randomization appears (e.g., tie-breaking in node
  selection, if any randomized component is ever introduced).
- **Deterministic mode:** a `--seed` flag producing byte-for-byte
  reproducible runs, including under CPU parallelism (Section 14).
- **Solver configuration capture:** every run logs the exact tolerance
  values, presolve level, scaling choice, and backend (CPU/GPU) used.
- **Hardware capture:** CPU model, core count, memory, and (for GPU runs)
  GPU model, logged alongside every benchmark result.
- **Compiler/version capture:** compiler identity and version, build mode
  (debug/release), and the exact solver build hash logged per run.
- **Benchmark manifest:** the exact instance list, source, and category
  tags used for any reported benchmark run, checked into the repository.
- **Result JSON:** every solve produces a structured result record
  (status, objective, timing, residuals, configuration) suitable for
  automated re-verification, not just a human-readable log line.
- **Logs:** structured solver telemetry (Section 30), not ad-hoc `printf`
  output, retained per run.
- **Command-line reproduction:** every reported result is accompanied by
  the exact CLI command that produced it.
- **Version/hash tracking:** the exact git commit hash of the solver build
  used for any reported number, logged alongside that number.

**Standard, stated directly:** every important result presented in this
project's documentation or demo is reproducible — re-running the exact
logged command, on the same or comparably-specified hardware, with the same
seed, must reproduce the same status, objective value, and (within normal
timing variance) comparable performance numbers.

---

## Section 24 — Evidence Matrix

Every headline claim made anywhere in this document traces to an
implementation, a test, a benchmark, an artifact, and a demo method. This
table is the single place a judge can check any claim against its proof.

| Claim | Implementation | Test | Benchmark | Artifact | Demo Method |
|---|---|---|---|---|---|
| "From scratch" | No optimization-engine dependency anywhere in the solve path (Sec. 2, 30) | CI dependency-graph grep | — | Build manifest, dependency list | Live `grep`/dependency listing on demand |
| "Numerically robust" | Named-tolerance policy, scaling, refinement, near-zero-pivot flagging (Sec. 11) | Ill-conditioned/degenerate adversarial suite (Sec. 22) | Ill-conditioned instance benchmark | Numerical robustness doc | Live ill-conditioned/degenerate solve |
| "Independently verified" | Verification layer [Y] (Sec. 12) | Fault-injection test | — | Fault-injection test report | Live deliberately-broken build catching a bug |
| "GPU accelerated" | One scoped batched-SpMV CUDA kernel (Sec. 15) | GPU/CPU equivalence test | Problem-size sweep, transfer-inclusive, crossover point | GPU benchmark report | Live crossover chart, `--no-gpu` fallback shown |
| "Scalable sparse architecture" | CSR/CSC layer [D], sparse LU [E] (Sec. 13) | SpMV vs. dense correctness | Memory-vs-nonzero-count measurement | Sparse layer test + memory-scaling chart | Show memory scaling on a real large instance |
| "Correct MILP" | Branch-and-bound [G] on verified LP [F] (Sec. 9) | Integer-infeasibility, weak-relaxation, node-explosion adversarial cases | MIPLIB subset benchmark | MILP test report | Live MILP solve with gap closing telemetry |
| "Correct LP" | Revised simplex [F] (Sec. 8) | Infeasible/unbounded/degenerate/cycling adversarial cases | Netlib benchmark | LP test report | Live Netlib solve |
| "Correct convex QP" | Active-set [H] (Sec. 10) | KKT-residual tests, PSD-rejection test | QP test instances (+ QPLIB if reached) | QP test report | Live small convex QP solve |
| "MRPL relevance" | Crude-blending + capacity-allocation model (Sec. 21) | Model-specific correctness check | — | Case-study document | Live MRPL demo solve |
| "Honest benchmarking" | Full-instance, no-cherry-picking reporting policy (Sec. 18–20) | Reproducibility re-run test | Full Netlib/MIPLIB/HiGHS comparison table | Benchmark report | Live re-run with same seed |
| "Extensible toward MIQP/NLP/MINLP" | Architecture extension points only (Sec. 4, 36) | Architecture review | — | Architecture doc | Point to extension points, explicitly not a functional demo |
| "Reproducible" | Seed/config/hardware/build capture (Sec. 23) | Re-run test | — | Result JSON + manifest | Live re-run, same result |

Any claim not appearing in this table, anywhere in this project's
documentation or demo script, should be treated as unsupported and either
added here with a real evidence path or removed before presentation.

---

## Section 25 — Competitive Differentiation

Based on commonly observed weaknesses in comparable technical submissions in
this problem space — described here at the pattern level, without ranking or
naming any specific team or project.

**Common implementation weaknesses in this problem space:**
- Building a simulation or a modeling/application layer and calling it a
  "solver," rather than implementing the actual optimization algorithm.
- Dashboard-heavy architectures where UI polish substitutes for solver
  substance.
- Unsupported GPU claims — a "GPU-accelerated" label with no reproducible,
  transfer-inclusive benchmark behind it.
- Toy-only instances — a 5-to-10-variable demo presented as evidence of
  industrial scale.
- No independent verification — trusting the solver's own "optimal" claim
  with no separate check.
- No difficult adversarial cases attempted — only the easy, happy-path
  instances shown.
- No reproducible benchmark — numbers presented without the run
  configuration needed to reproduce them.
- No honest failure reporting — a results table that shows only wins.

**For each Kavach-Opt differentiator, how it is proven (cross-referenced to
Section 24's evidence matrix, not restated as an unsupported claim here):**

| Differentiator | How it is proven |
|---|---|
| Real solver core, not a wrapper | Live source walkthrough; dependency-graph grep showing zero optimization-engine imports |
| Independent verification layer | Live fault-injection demo — a deliberately-broken build caught by the verifier |
| Honest, non-cherry-picked benchmarking | Full results table shown, including failures/timeouts |
| Scalable sparse architecture | Live memory-scaling measurement on a real instance |
| Reproducible experiments | Live re-run with the same seed, same result |
| Measured (not claimed) CPU/GPU speedup | Live crossover-point chart, including sizes where GPU is slower |
| Industrial use case | Live MRPL blending/capacity model solve |
| Solver transparency | Live iteration/node/gap telemetry during solving, not just a final number |
| Modular extensibility | Architecture diagram walkthrough, explicitly labeled as extension points, not built capability |

---

## Section 26 — Judge Attack Questions

Answers below are written to be technically honest — including where the
honest answer is a limitation, not a strength.

**1. Why don't you just use HiGHS?** Because the PS explicitly requires a
from-scratch solver core, not a wrapper — using HiGHS internally would fail
the core requirement (R5, Section 1) regardless of how good the resulting
product might be. HiGHS is used only as an external, arms-length comparison
target (Section 20), never called by Kavach-Opt at solve time.

**2. How is your solver sovereign?** The simplex, branch-and-bound, and
active-set algorithms are implemented from their mathematical definitions by
this team, with zero optimization-engine dependency anywhere in the solving
path — verifiable by inspecting the dependency graph directly.

**3. Are you wrapping any existing optimizer?** No. This is checked
continuously by a CI dependency-graph grep (Section 30), not asserted once
and left unverified.

**4. Why did you choose revised simplex?** It scales to sparse industrial
instances (unlike naive tableau simplex) and warm-starts naturally for
MILP's repeated child-node re-solves (unlike interior-point methods, which
do not preserve a reusable discrete basis between solves) — see Section 8's
justification.

**5. Why is MILP difficult?** LP is polynomial-time solvable; MILP is
NP-hard. All of the difficulty is in the search (branch-and-bound tree
management, bound quality, node count) layered on top of a correct LP
solver — and since every B&B node re-solves an LP, LP correctness and speed
directly gate MILP feasibility (Section 9).

**6. Why not GPU-accelerate simplex?** Simplex pivoting is inherently
sequential and branch-heavy with a small per-iteration working set — a poor
structural fit for GPU's throughput-oriented, data-parallel execution model.
Attempting a "GPU simplex" as a headline claim is a known hard research
problem, not something achievable to a verified, defensible standard on this
timeline (Section 15).

**7. Where exactly does the GPU help?** Exactly one kernel: batched sparse
matrix-vector product, when batched across enough independent right-hand
sides that transfer overhead is amortized — measured, not assumed (Section
15).

**8. What happens when GPU is slower?** It is reported exactly as measured.
The crossover-point chart explicitly includes and labels the region where
GPU is slower than CPU — this is a permitted, expected finding, not a result
to hide.

**9. How do you know your solution is actually feasible?** The independent
verification layer (Section 12) recomputes constraint residuals directly
from the original model and the reported solution, using code that shares no
logic with the solving path — it does not trust the solver's own
bookkeeping.

**10. How do you know "OPTIMAL" is correct?** The solver is not permitted to
report `OPTIMAL` unless the independent verifier confirms the relevant
optimality conditions (reduced-cost sign conditions and complementary
slackness for LP; zero verified gap for MILP; KKT residual for QP) — Section
11's status-taxonomy rule.

**11. How do you handle numerical instability?** Named per-purpose
tolerances (Section 11), geometric-mean scaling, near-zero-pivot detection
that flags rather than silently proceeds, and iterative refinement — with a
dedicated `NUMERICAL_FAILURE` status for cases where these safeguards cannot
produce a trustworthy result.

**12. How do you handle degeneracy?** Dantzig's rule by default, with a
cycling-detection watchdog that switches to Bland's rule (correctness-
guaranteeing, though slower) as a fallback — not applying Bland's rule
unconditionally, which would be slower than necessary on the vast majority
of non-degenerate instances (Section 8).

**13. What is your biggest successfully solved instance?** To be filled in
honestly from the actual benchmark report (Section 18) once solved and
verified — this document commits to reporting that number exactly as
measured, not to a target size stated in advance.

**14. What happens when your solver times out?** It reports `LIMIT_REACHED`
with the best incumbent found and an honestly-computed optimality gap — it
never silently reports a timeout as `OPTIMAL` (Section 11).

**15. Why don't you claim to beat CPLEX?** Because that claim would not be
credible from a from-scratch, hackathon-timeline implementation against a
decades-mature commercial engine, and an unsupportable claim would undermine
the credibility of every other, genuinely supportable claim in this
submission (Section 19).

**16. How fair is your benchmark comparison?** Same instances, same
hardware where practical, same time limits and tolerances, full-results
reporting including both solvers' failures/timeouts, no cherry-picking
(Section 20's protocol).

**17. What happens with an infeasible model?** The LP or MILP engine
detects and reports `INFEASIBLE` (with a certificate for LP), verified
independently — never a loop, crash, or false feasible/optimal report
(Section 22's adversarial suite).

**18. What happens with an unbounded model?** The LP engine detects and
reports `UNBOUNDED` when no valid leaving variable exists on an improving
direction (Section 8), verified independently, never silently returning an
arbitrary large value as "optimal."

**19. Why do you need sparse matrices?** Real industrial LP/MILP instances
— including MRPL's own — typically have constraint matrices well under 1%
dense. A dense implementation would be both computationally slow and
memory-infeasible at any real industrial scale; sparsity is a correctness-
of-scale requirement here, not an optional optimization (Section 13).

**20. Why is your industrial example relevant to MRPL?** Crude blending and
unit-capacity allocation are core, recurring refinery planning problem
*classes* — not a generic textbook LP repurposed for the demo — solved at a
size this solver can credibly and verifiably handle today (Section 21).

**21. What is currently NOT implemented?** Everything listed in Section 35
(Limitations) and Category D of Section 5's scope table: interior-point LP,
MIQP, NLP, MINLP, general non-convex QP, a full cutting-plane arsenal,
GPU-accelerated factorization, and parallel branch-and-bound beyond the
EXPERIMENTAL stretch stage — stated here up front, not discovered under
questioning.

**22. What is your biggest limitation?** The benchmark scale achievable
within this project's timeline is Netlib and a size-appropriate MIPLIB
subset — not the largest, hardest MIPLIB instances, and not parity with
mature commercial solvers at industrial-maximum scale (Section 19, Section
35).

**23. What would you build next?** The Future Extensions list in Section
36, in the priority order given there — dual-simplex-driven MILP
performance work and QPLIB coverage first, since they extend already-
working subsystems; interior-point methods, MIQP, and cutting-plane breadth
later, since they are each substantial standalone research-adjacent efforts.

---

## Section 27 — Failure and Fallback Strategy

| Scenario | Fallback |
|---|---|
| GPU unavailable | Solver runs CPU-only by default; `--no-gpu` is always a valid, fully-functional path — nothing else in the solver depends on GPU presence |
| GPU benchmark negative (GPU slower at demo-relevant sizes) | Reported honestly as the measured crossover point; the demo explicitly shows this as an expected, permitted finding, not concealed |
| Large MILP timeout | `LIMIT_REACHED` status with best-found incumbent and honest gap, never a false `OPTIMAL` |
| Numerical instability | `NUMERICAL_FAILURE` status, safeguards (scaling, refinement, pivot flagging) engaged per Section 11; if a live demo instance triggers this, it is shown and explained, not hidden |
| Parser failure | A clear, typed error identifying the malformed section/line, never a silent wrong model |
| QP unstable | PSD-rejection and KKT-residual safeguards (Section 10) prevent a false "solved" result; falls back to reporting `NUMERICAL_FAILURE` |
| Benchmark mismatch (CPU vs. GPU, or Kavach-Opt vs. HiGHS disagreement beyond tolerance) | Flagged and investigated before any result is reported publicly; a live demo would show this as a bug being caught, consistent with the verification-layer differentiation story |
| CPU performance issue on demo hardware | The demo instance set includes a pre-sized "known good on typical hardware" tier, so the live demo is not dependent on the largest, slowest instances |
| Demo machine differences | Hardware is disclosed and logged per Section 23; timing numbers are understood to vary by hardware, and this is stated explicitly rather than presented as a fixed guarantee |
| Network unavailable | The live demo has no network dependency — all instance files, the solver binary, and any pre-recorded backups are local |

**Standing rule:** the live demo must not depend on internet access, and any
pre-recorded result shown as backup is explicitly labeled as pre-recorded
when shown — never disguised as a live result (Section 28).

---

## Section 28 — Demo-Day Plan

A complete 5–10 minute technical demo.

1. Explain the problem (30–45s): what MRPL needs, why a sovereign solver
   core, not a wrapper.
2. Load a real benchmark model (Netlib MPS instance) — show model
   statistics (variables, constraints, nonzeros, density).
3. Show presolve — before/after variable and constraint counts.
4. Solve LP live — show simplex iteration telemetry (iteration count,
   current objective, reduced-cost convergence).
5. Switch to a MILP instance — show branch-and-bound tree growth and the
   optimality gap closing live.
6. Run the independent verifier on both reported solutions — show it
   passing; then, on a deliberately-broken build, show it correctly
   catching an injected wrong answer.
7. Show CPU vs. GPU timing on the one batched-SpMV kernel, including the
   crossover chart (both the region where GPU wins and where it is
   honestly slower).
8. Solve the MRPL blending/unit-capacity demo model end-to-end, with
   operational interpretation of the result.
9. Show one benchmark table slide — Kavach-Opt vs. HiGHS, including any
   honest losses.
10. Close on the roadmap (Section 36's future extensions), explicitly
    labeled as future work, not present capability.
11. State current limitations directly (Section 35) before being asked.

**LIVE vs. pre-recorded backup, stated explicitly:** steps 2–8 are intended
to run live on demo hardware. If a specific benchmark instance's runtime
exceeds the live-demo time budget, a pre-recorded run of that exact instance
is the designated backup, and it is **labeled as pre-recorded on screen**
when shown — never presented as if it were live. The GPU crossover chart
(step 7) is pre-computed benchmark data by nature (a sweep across many sizes
cannot itself run live in real time) and is labeled as a benchmark result,
not a live measurement, while the specific CPU-vs-GPU timing comparison for
one chosen instance size is run live alongside it.

**Fallback design:** every step works with `--no-gpu` (falls back to CPU
silently, demo continues uninterrupted); the full instance set used is
entirely local, no network dependency during the demo.

---

## Section 29 — Documentation Set

| Document | Purpose | Owner | When created | Evidence it must contain |
|---|---|---|---|---|
| README | Orientation, build/run instructions | Whole team | Phase 0 | Build/run commands, verified working |
| Architecture doc | This document's Section 4, kept current | Architecture track lead | Phase 0, updated each phase | Current component diagram matching actual code |
| Mathematical foundations doc | Section 7, expanded with derivations as implemented | Solver-math track lead | Alongside each algorithm's implementation | Derivations matching actual implemented algorithms |
| LP algorithm doc | Section 8 as built, deviations noted and justified | LP track lead | Phase 5 | Pseudocode matching real code, deviation log |
| MILP algorithm doc | Section 9 as built | MILP track lead | Phase 7 | Pseudocode matching real code, deviation log |
| QP algorithm doc | Section 10 as built | QP track lead | Phase 9 | Pseudocode matching real code, deviation log |
| Numerical robustness doc | Section 11's tolerance table, scaling policy, verification design | Numerics track lead | Phase 3, updated through Phase 8 | Tolerance values, scaling method, safeguard behavior |
| Benchmark methodology doc | Section 18–19 | Benchmark track lead | Phase 12 | Full field list, instance sourcing, acceptance criteria |
| Benchmark results doc | Full Netlib/MIPLIB/(QPLIB) results, honest including failures | Benchmark track lead | Phase 12 onward | Full per-instance results table |
| GPU design + benchmark doc | Section 15, crossover chart, hardware spec | GPU track lead | Phase 11 | Crossover chart, transfer-inclusive timings |
| MRPL case study doc | Section 21, worked through with real solve output | Demo track lead | Phase 13 | Model, verified solve result, operational interpretation |
| API/CLI doc | Section 17 as built | API/CLI track lead | Phase 14 | Every documented command's actual output |
| Test report | Pass/fail counts, adversarial-case coverage | All tracks | Continuous, finalized Phase 15 | Full test-suite results including adversarial suite |
| Reproducibility guide | Section 23 — exact commands, seeds, expected outputs | Benchmark track lead | Phase 12 | Re-runnable commands with expected results |
| Limitations doc | Section 35 | Whole team, reviewed together | Phase 15 | Explicit list, cross-checked against Section 24 |
| Final demo guide | Section 28's script, LIVE/pre-recorded labeling | Demo track lead | Phase 15 | Full script with fallback plan |

---

## Section 30 — Engineering Standards

**Language and build:** C++20 (structured bindings, concepts where they
clarify numerical interfaces without over-engineering). CMake as the build
system.

**Static quality gates:** compiler warnings treated as errors in CI
(`-Wall -Wextra -Werror` or equivalent). Sanitizers (AddressSanitizer,
UndefinedBehaviorSanitizer) run in CI on the test suite — numerical code is
exactly where subtle memory/UB bugs hide silently. Static analysis
(clang-tidy). Consistent formatting (clang-format, enforced in CI).

**Performance discipline:** profiling before any hand-optimization — no
speculative SIMD/threading without a profiler showing the specific hot path
justifies it.

**Determinism and observability:** a `--seed` flag making every run
byte-for-byte reproducible (Section 23). Structured logging (not ad-hoc
`printf`) for solver telemetry.

**Resource management:** RAII throughout; clear ownership rules for GPU
memory specifically (RAII wrappers around CUDA allocations, no manually
scattered `cudaFree` calls).

**Additional standards for this revision:**
- **Numerical assertions:** debug-build assertions on invariants the
  numerical layer depends on (e.g., basis matrix dimensions, factorization
  pivot validity) — cheap correctness insurance that costs nothing in
  release builds.
- **Invariant checks:** CSR/CSC structural invariants (Section 13) checked
  in debug builds after every mutating operation.
- **Benchmark isolation:** benchmark runs execute in a controlled
  environment (no other heavy processes competing for CPU/GPU) and this
  environment is disclosed as part of every benchmark report (Section 23).
- **Configuration versioning:** tolerance/scaling/presolve configuration is
  itself version-tracked, so a benchmark result can be traced to the exact
  configuration that produced it, not just the exact code commit.
- **Build reproducibility:** a pinned toolchain version (compiler, CMake,
  CUDA toolkit where used) so a build from a given commit is reproducible
  by a third party.
- **Clean dependency policy:** the only third-party dependencies permitted
  are general-purpose, non-optimization-engine libraries (e.g., a testing
  framework, a CLI-argument-parsing library) — enforced by the same CI
  dependency-graph check that verifies the "from scratch" claim (Section
  24).

---

## Section 31 — Project Repository Structure

```
kavach-opt/
├── src/
│   ├── model/              # [A] canonical representation
│   ├── io/                 # [B] MPS/LP parsers
│   ├── numerics/            # [D][E] sparse layer, LU, scaling
│   ├── presolve/            # [C]
│   ├── algorithms/
│   │   ├── lp/               # [F]
│   │   ├── milp/             # [G][I]-[P]
│   │   └── qp/                # [H]
│   ├── verification/         # [Y]
│   ├── strategy/              # [T] rule-based solver strategy layer
│   ├── backend/
│   │   ├── cpu/                # [S]
│   │   └── cuda/               # [U]
│   ├── api/                    # [V]
│   └── cli/                    # [W]
├── benchmarks/                 # [X], Netlib/MIPLIB/QPLIB harness
├── tests/                      # unit + adversarial + regression + fault-injection
├── examples/                   # [AB] MRPL demo model
├── configs/                    # tolerance/scaling/presolve configuration files, version-tracked
├── scripts/                    # reproducibility helper scripts (Section 23)
├── tools/                      # dependency-graph checker, benchmark report generator
└── docs/                       # Section 29's documents
```

No unnecessary microservices — the solver core is the primary project; the
CLI and API are thin layers over the same core library, not separate
deployable services.

---

## Section 32 — Phased Implementation Roadmap

Strict gated development: later phases do not proceed while foundational
correctness in an earlier phase is failing.

**Pre-phase open decisions — must be finalized before the gating phase begins:**

| Decision | Must be finalized before | Notes |
|---|---|---|
| **Target hardware specification** — GPU model, CUDA capability level, CPU model, core count, memory configuration on which Phases 10, 11, and 12 benchmarks will run | **Phase 10** (multicore SpMV) | Without a concrete hardware target, GPU crossover benchmarking (Section 15) and CPU scaling claims (Section 14) cannot be planned or reproduced; hardware must be disclosed in every benchmark report per Section 23 |
| **Concrete MIPLIB and QPLIB instance selection** — the specific named instances constituting the "size-appropriate subset" used throughout this document | **Phase 12** (benchmark framework) | The selection must be recorded in the benchmark manifest (Section 18) and checked into the repository before Phase 12 begins, so that no cherry-picking occurs after results are seen |
| **Team size and calendar time budget** — headcount per track (Section 33) and total calendar time available | **Phase 0** (repo foundation) | Required to assess whether the Strong SIH Target (Category B, Section 5) is realistic for the actual team; if the team is smaller than the 6-track parallelization assumes, phases must be re-sequenced but not redesigned |

| Phase | Goal | Files | Dependencies | Algorithms | Tests | Benchmark | Definition of Done | Exit Criteria | Known Risks | Fallback |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Repo foundation | CMake, CI skeleton, directory tree | — | — | Build passes | — | Clean build on CI | CI green | Tooling setup delays | Simplify CI matrix if needed |
| 1 | Canonical model | `src/model` | Phase 0 | — | Construction/validation unit tests | — | Sec. 6 [A] | Rejects all malformed test inputs | Under-specified edge cases | Expand validation iteratively |
| 2 | Parser | `src/io` | Phase 1 | MPS/LP parsing | Round-trip on real Netlib files | — | Sec. 6 [B] | Parses ≥10 real Netlib files correctly | Silent misparse on edge-case MPS sections | Expand adversarial malformed-file tests |
| 3 | Sparse numerics | `src/numerics` | Phase 1 | CSR/CSC, sparse LU | Factorization vs. hand-computed systems | Memory-scaling | Sec. 6 [D][E] | Matches hand-computed results; residuals within tolerance | Fill-in growth, pivot bugs | Add refactorization safeguard early |
| 4 | Presolve | `src/presolve` | Phase 1–3 | Bound tightening, redundancy removal | Before/after size checks; reduced-restored comparison | — | Sec. 6 [C] | Measurable reduction; reduced-restored solutions agree within `primal_feasibility_tol` | Incorrect reduction silently changing feasible region | Pair every reduction with a restore test before merging |
| 5 | LP solver | `src/algorithms/lp` | Phase 2–4 | Revised simplex, Bland's rule | Infeasible/unbounded/degenerate adversarial suite | Netlib | Sec. 6 [F] | Solves all small Netlib instances correctly, verifiably | Cycling, ratio-test ties | Bland's-rule fallback |
| 6 | Verification layer | `src/verification` | Phase 5 | Independent residual/KKT checks | Fault-injection test | — | Sec. 6 [Y] | Catches every injected fault | Shared logic with solver undermining independence | Code-review gate specifically for independence |
| 7 | MILP solver | `src/algorithms/milp` | Phase 5–6 | Branch-and-bound, best-bound selection | Integer-infeasible + weak-relaxation adversarial suite | Small-medium MIPLIB | Sec. 6 [G] | Solves subset with correct, verified gap | Node explosion | Node/time cap + honest `LIMIT_REACHED` |
| 8 | Scaling + dual-simplex warm start | `src/numerics`, `src/algorithms/lp` | Phase 5, 7 | Scaling, dual simplex | Ill-conditioned adversarial suite | MILP node-solve timing | Measurable MILP node-solve speedup from warm-starting | Speedup measured and reproducible | Warm-start correctness bugs | Fall back to cold-start if warm-start diverges |
| 9 | QP solver | `src/algorithms/qp` | Phase 3, 6 | Active-set | KKT residual tests, PSD-rejection test | Small convex QP set (+ QPLIB if reached) | Sec. 6 [H] | Solves small convex QP correctly, verified | Non-convex input silently accepted | PSD check gate before solve |
| 10 | Multicore batched SpMV | `src/backend/cpu` | Phase 3 | Thread pool | Parallel-vs-serial correctness | Measured speedup | Sec. 6 [S] | Correct, honest speedup reported | Race conditions | Disjoint-output task design (Sec. 14) |
| 11 | CUDA batched SpMV | `src/backend/cuda` | Phase 10 | GPU kernel + CPU fallback | GPU/CPU equivalence test | Crossover-point sweep | Sec. 6 [U] | Honest crossover chart produced | Result mismatch, overclaiming scope | CPU fallback always default |
| 12 | Benchmark framework | `benchmarks/` | Phase 5, 7, 9, 11 | — | Reproducibility re-run test | Full Netlib/MIPLIB/(QPLIB) report | Sec. 6 [X] | Full report including failures | Cherry-picking temptation | External review before demo day (Sec. 23) |
| 13 | MRPL demo model | `examples/` | Phase 5, 7 | — | Model-specific correctness check | — | Sec. 21 | Solves and verifies, interpretable | Illustrative-data mislabeling | Explicit labeling review |
| 14 | Solver strategy layer | `src/strategy` | Phase 3–11 | Rule-based signal→setting mapping | Rule-application unit tests | — | Sec. 16 | Deterministic, documented, testable rule set | Rules conflicting or untested | Keep rule set small, test each rule individually |
| 15 | CLI/API polish | `src/api`, `src/cli` | Phase 5, 7, 9 | — | CLI smoke tests | — | Sec. 6 [V][W] | Usable by someone other than the author | Undocumented behavior | Cross-check every command against Section 17 |
| 16 | Optional dashboard | (last, non-load-bearing) | — | — | Manual smoke test | — | Never blocks any other phase | N/A | Time overrun | Cut entirely if time-constrained |
| 17 | Final validation | All | All prior phases | — | Full regression suite | — | Every claim in Section 24 reproducible on demand | Third-party reproduction succeeds | Any claim failing to reproduce | Downgrade or remove the unsupported claim before submission |

Gomory cuts, pseudocost branching, and parallel B&B are explicitly **not**
numbered phases — they are Category C stretch additions (Section 5),
insertable between Phases 7–11 *only if the phase they would enhance is
already solid and time remains.*

---

## Section 33 — Team Parallelization

Parallel workstreams designed to avoid integration chaos, with explicit
dependencies and checkpoints.

| Track | Owns | Depends on | Integration checkpoint |
|---|---|---|---|
| **Track A — Mathematical solver** | LP (Phase 5), MILP (Phase 7), warm-start (Phase 8) | Track B's numerics layer (Phase 3) | After Phase 3 lands, before Phase 5 begins in earnest |
| **Track B — Numerical core** | Sparse layer, LU factorization, scaling, refinement (Phase 3) | Track (model) Phase 1 | Before any of Tracks A/C/D can proceed |
| **Track C — MILP-specific** | Branch-and-bound orchestration, node management, heuristics (Phase 7) | Track A's LP engine (Phase 5) | After Phase 5's exit criteria are met |
| **Track D — GPU/performance** | CPU parallelism (Phase 10), CUDA kernel (Phase 11) | Track B's SpMV implementation (Phase 3) | After Phase 3; GPU work never blocks Tracks A/C |
| **Track E — Benchmarking/verification** | Verification layer (Phase 6), benchmark harness (Phase 12) | Track A's LP/MILP for anything to verify against | Verification layer (Phase 6) can start as soon as Phase 5's LP interface is stable, even before MILP lands |
| **Track F — CLI/demo/documentation** | API/CLI (Phase 15), MRPL demo model (Phase 13), documentation set (Section 29) | Tracks A/C for functioning solver calls | Ongoing, with hard integration passes at Phases 13 and 15 |

**Explanation of dependencies:** Track B is the critical-path prerequisite
for almost everything — no LP, MILP, QP, or GPU work can proceed
meaningfully until the sparse numerical core is correct, so Track B is
prioritized and adequately staffed from Phase 0. Track E's verification
layer is deliberately made to start as early as Phase 5 completes (not
waiting for the full MILP/QP stack), because the earlier a genuinely
independent verifier exists, the earlier every subsequent subsystem
benefits from having its results checked. Track D (GPU) is structured so
that no other track's Definition of Done depends on it, consistent with
Section 5's rule that GPU never enters Category A.

**Integration discipline:** each checkpoint above is a point where the
owning track's Definition of Done (Section 6) must be met before dependent
tracks are unblocked — a track is not "handed off" on a schedule alone.

---

## Section 34 — Final KPI Dashboard

Measurable categories only — no invented targets.

**Correctness**
- Verified LP instance count (passed independent verification)
- Verified MILP instance count
- Verified QP instance count
- Correct status-detection count (INFEASIBLE/UNBOUNDED cases correctly identified)

**Performance**
- Runtime per instance (recorded, not aggregated away)
- Iteration count (LP), node count (MILP)
- Memory usage per instance

**Numerical robustness**
- Verification-layer residuals (recorded per instance)
- Verification pass rate (fraction of solved instances passing independent verification)
- Numerical-failure rate (`NUMERICAL_FAILURE` status frequency, tracked honestly, not minimized)

**Scalability**
- Variable count, constraint count, nonzero count of the largest solved-and-verified instance per problem class
- Largest solved instance overall, per benchmark category (Section 18)

**Acceleration**
- CPU baseline runtime (per kernel/workload)
- GPU runtime (compute-only and transfer-inclusive, reported separately)
- Speedup ratio (from transfer-inclusive time)
- Crossover problem size (measured, not assumed)

**Engineering**
- Test count (unit + adversarial + fault-injection)
- Code coverage (where meaningful for numerical code — coverage percentage alone is not treated as a proxy for correctness)
- Reproducibility (fraction of reported benchmark results that reproduce on re-run with the logged configuration)

No target values are stated for any of the above in this document — every
number in the KPI dashboard is filled in from actual measurement as
development proceeds, per Section 19's correctness-first, no-fabricated-
number discipline.

---

## Section 35 — Limitations

Stated explicitly, to make the submission more credible, not weaker.

- **Not universal parity with commercial solvers.** Kavach-Opt does not
  match CPLEX, Gurobi, or Xpress across the board, and does not claim to.
- **No general nonlinear solver yet.** NLP/MINLP are explicitly future work
  (Section 36), not present capability.
- **No MIQP** unless separately implemented and verified — not committed in
  this scope.
- **No full cutting-plane arsenal.** At most Gomory mixed-integer cuts,
  and only as an EXPERIMENTAL stretch (Section 9).
- **GPU does not accelerate every solver operation** — exactly one kernel
  (batched SpMV) is GPU-accelerated; simplex pivoting and B&B tree traversal
  remain CPU-only by deliberate design (Section 15).
- **Benchmark ceiling depends on implementation maturity.** The largest
  verifiably-solved instance size is whatever the actual benchmark
  measurement shows at submission time — not a size chosen in advance to
  sound impressive.
- **No general non-convex QP.** Convex QP only; non-convex input is
  rejected at input, not attempted (Section 10).
- **No interior-point LP path.** Revised simplex only, by deliberate choice
  (Section 8); interior-point is understood but not implemented.
- **Illustrative, not proprietary, MRPL data.** The industrial demo model
  (Section 21) uses structurally realistic but illustrative data, clearly
  labeled as such, because proprietary MRPL data is not assumed available.
- **Parallel B&B is EXPERIMENTAL, not guaranteed.** Section 9's stretch
  item; single-threaded B&B correctness is the guaranteed baseline.

---

## Section 36 — Future Extensions

All items below are explicitly marked **FUTURE / NOT CURRENTLY
IMPLEMENTED**. They are listed for architectural completeness and to answer
Section 26's "what would you build next" question honestly, not because any
of them is committed scope for this submission.

- **MIQP** (Mixed-Integer Quadratic Programming) — FUTURE
- **NLP** (Nonlinear Programming) — FUTURE
- **MINLP** (Mixed-Integer Nonlinear Programming) — FUTURE
- **Interior-point LP** — FUTURE (understood conceptually, Section 7; not implemented)
- **Advanced presolve** (beyond bound tightening + redundancy removal) — FUTURE
- **Pseudocost branching** — Category C stretch if time permits within this project; otherwise FUTURE
- **Parallel branch-and-bound** — EXPERIMENTAL stretch if time permits; full production-grade version is FUTURE
- **Advanced cuts** (beyond Gomory mixed-integer, if even that is reached) — FUTURE
- **GPU-accelerated factorization** — THEORETICAL/FUTURE (Section 15)
- **Distributed/multi-node optimization** — FUTURE
- **Optional learned (ML-based) solver strategy**, extending the rule-based Solver Strategy Layer of Section 16 — explicitly FUTURE, not introduced now for buzzword value

---

## Section 37 — Final "Build Contract"

**WE WILL BUILD:**
- A from-scratch revised-simplex LP engine, sparse, presolved, scaled.
- A from-scratch branch-and-bound MILP engine built on that LP engine.
- A from-scratch convex active-set QP engine.
- A sparse CSR/CSC numerical core with LU factorization, refinement, and scaling.
- A named-tolerance numerical robustness policy and solver status taxonomy.
- An independently-coded verification layer for LP, MILP, and QP results.
- One scoped, measured GPU kernel (batched SpMV) with CPU fallback always available.
- A rule-based solver strategy layer.
- A solver API and CLI (no GUI required or committed as load-bearing).
- A benchmark harness against Netlib, a size-appropriate MIPLIB subset, and (conditionally) QPLIB.
- An illustrative MRPL crude-blending + unit-capacity MILP demonstration model.

**WE WILL MEASURE:**
- Correctness (verified instance counts, per problem class).
- Runtime, iteration/node counts, memory, every attempted instance.
- Numerical robustness (verification pass rate, numerical-failure rate).
- CPU and GPU performance, including the honest crossover point.
- Benchmark comparison against HiGHS, wins and losses both.

**WE WILL VERIFY:**
- Every reported LP/MILP/QP result, independently, before it is returned.
- Every GPU result against its CPU counterpart.
- Every benchmark result's reproducibility via a logged seed/configuration.

**WE WILL DEMONSTRATE:**
- A live solve of a real Netlib LP instance and a real MIPLIB-subset MILP instance.
- The verification layer catching a deliberately injected wrong answer.
- The GPU crossover chart, including the region where GPU is slower.
- The MRPL blending/capacity model, solved end-to-end and interpreted.
- One honest benchmark table against HiGHS.

**WE WILL NOT CLAIM:**
- That Kavach-Opt beats CPLEX, Gurobi, or Xpress in general.
- A fixed GPU speedup target chosen in advance.
- Support for MIQP, NLP, MINLP, or general non-convex QP as present capability.
- Solving the largest, hardest MIPLIB instances.
- That any part of the solving path wraps or embeds an existing optimization-engine library.
- An `OPTIMAL` status that the independent verifier has not confirmed.

**Definition of a successful implementation:** every Section 6 Definition of
Done is met; every Section 24 evidence-matrix claim is reproducible on
demand; the demo in Section 28 runs live, with `--no-gpu` fallback proven to
work, and every pre-recorded segment (if any) is clearly labeled as such.

**Definition of a failed subsystem:** any subsystem whose Section 6
Definition of Done is not met, or whose claims cannot be reproduced per
Section 24, is reported as such in the Limitations document (Section 35) —
it is not hidden, quietly dropped from the demo script, or silently
re-labeled as "future work" without disclosure.

**Rules, restated as the closing discipline of this document:**
- Correctness before speed.
- Evidence before claims.
- Solver before UI.
- Measured GPU before GPU claims.
- Reproducibility before presentation.
- Honest limitations, stated up front.

---

## Section 38 — Final One-Page Master Summary

**Product:** Kavach-Opt — a sovereign, from-scratch LP/MILP/QP solver core.

**Problem:** MRPL and Indian industry depend on licensed, opaque foreign
optimization engines (CPLEX/Gurobi/Xpress) for refinery scheduling, crude
blending, and production planning.

**Solution:** a sparse, numerically self-verifying LP/MILP/QP solver core,
built from mathematical foundations, CPU-first with one scoped, honestly
benchmarked GPU kernel.

**Core architecture:** Parser → Canonical Model → Presolve → Sparse
Numerical Core (CSR/CSC, LU) → LP (revised simplex) → MILP (branch-and-
bound) / QP (active-set) → Independent Verification Layer → CPU/GPU
Execution → Solver Strategy Layer → API/CLI → Benchmark Framework →
Industrial (MRPL) Application Layer → (optional, thin) Dashboard.

**Algorithms:** revised simplex with LU-factorized basis and Bland's-rule
anti-cycling fallback (LP); branch-and-bound with best-bound node selection
and most-fractional branching (MILP); active-set with PSD pre-check
(convex QP).

**Numerical robustness:** named-tolerance taxonomy, geometric-mean scaling,
near-zero-pivot flagging, iterative refinement, a solver status taxonomy
that never silently reports false `OPTIMAL`.

**GPU:** exactly one kernel — batched sparse matrix-vector product —
benchmarked with transfer-inclusive timing and an honest crossover point;
CPU is always the default and always available.

**Benchmarks:** Netlib LP, a size-appropriate MIPLIB subset, QPLIB if
reached; every attempted instance reported, including failures.

**MRPL demo:** crude-blending + unit-capacity MILP model, illustrative
data, structurally realistic, solved and independently verified live.

**Evidence:** every headline claim traces to an implementation, a test, a
benchmark, and a demo method via Section 24's evidence matrix.

**Differentiators:** independent verification layer (never trusts the
solver's own "optimal" claim); honest, non-cherry-picked benchmark
reporting; a real sparse architecture at Netlib/MIPLIB scale; an honestly-
scoped, measured GPU claim.

**Build sequence:** Phases 0–9 (foundation through QP) are the
non-negotiable core; Phases 10–11 (CPU/GPU acceleration) proceed only once
the core is verified and stable; Phases 12–17 (benchmarking, demo,
polish, final validation) close out the submission.

**Top risks:** simplex cycling on degenerate instances (mitigated: Bland's-
rule fallback); MILP node explosion (mitigated: node/time caps, honest
`LIMIT_REACHED` reporting); an overclaimed GPU story (mitigated: one
kernel, transfer-inclusive honest crossover benchmarking); numerical
instability on real Netlib/MIPLIB instances (mitigated: scaling, iterative
refinement, `NUMERICAL_FAILURE` status); the "built from scratch" claim
being challenged (mitigated: CI-enforced zero-dependency policy, live
code walkthrough available on demand).

---

*Implementation order: Phases 0–9 (Section 32) are the non-negotiable core.
Phases 10–11 (multicore/GPU) proceed only once 0–9 are verified and stable —
a working, verified CPU-only LP/MILP/QP solver is a stronger submission than
a GPU-accelerated one that has not been proven correct. Code generation
begins at Phase 0 only upon explicit go-ahead.*
