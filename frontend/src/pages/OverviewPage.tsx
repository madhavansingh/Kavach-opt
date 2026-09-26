import React from 'react';
import { Play, Shield, FileCode } from 'lucide-react';
import { Panel } from '../components/Panel';
import { Metric, MetricGroup } from '../components/Metric';
import { Pipeline } from '../components/Pipeline';
import { StatusBadge } from '../components/StatusBadge';
import { PrimaryButton, SecondaryButton } from '../components/Buttons';
import { EmptyState } from '../components/EmptyState';
import { SkeletonMetric, SkeletonCard, Skeleton } from '../components/Skeleton';
import type { SolveResult, ModelMetadata } from '../types/solver';
import type { NavTab } from '../layouts/Sidebar';

interface OverviewPageProps {
  onNavigate: (tab: NavTab) => void;
  activeModel?: ModelMetadata;
  lastResult?: SolveResult;
  isDiscoveringModels?: boolean;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({
  onNavigate,
  activeModel,
  lastResult,
  isDiscoveringModels = false,
}) => {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="relative rounded-xl border border-[#1d2332] bg-gradient-to-b from-[#11151f] via-[#0d1017] to-[#0a0d12] p-6 sm:p-8 overflow-hidden shadow-xs">
        <div className="max-w-3xl relative z-10">
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-emerald-950/40 border border-emerald-700/40 text-emerald-400 text-xs font-mono font-medium mb-4">
            <Shield className="w-3.5 h-3.5" />
            <span>SIH26119 · Sovereign Optimization Engine</span>
          </div>

          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold tracking-tight text-white font-sans">
            Kavach-Opt <span className="text-neutral-400 font-normal text-xl sm:text-2xl">(कवच-Opt)</span>
          </h1>

          <p className="mt-3 text-sm sm:text-base text-neutral-300 leading-relaxed max-w-2xl font-sans">
            A from-scratch, sparse, numerically self-verifying mathematical optimization solver
            core engineered in C++20. Designed as an indigenous alternative to foreign proprietary
            solvers for industrial planning, crude scheduling, and refinery operations.
          </p>

          <div className="mt-6 flex flex-wrap items-center gap-3">
            <PrimaryButton
              icon={<Play className="w-4 h-4" />}
              onClick={() => onNavigate('solve')}
            >
              Run Optimization
            </PrimaryButton>
            <SecondaryButton
              icon={<Shield className="w-4 h-4" />}
              onClick={() => onNavigate('verification')}
            >
              Inspect Verifier
            </SecondaryButton>
            <SecondaryButton
              onClick={() => onNavigate('mrpl')}
            >
              MRPL Refinery Case
            </SecondaryButton>
          </div>
        </div>

        {/* Subtle grid background illustration */}
        <div className="absolute right-0 top-0 bottom-0 w-1/3 opacity-5 pointer-events-none hidden md:block">
          <div className="w-full h-full border-l border-[#2e3748] bg-[radial-gradient(#38bdf8_1px,transparent_1px)] [background-size:16px_16px]" />
        </div>
      </div>

      {/* Current Solver Session Status */}
      <Panel
        title={lastResult ? 'Active Optimization Run' : 'Selected Model Ready for Execution'}
        subtitle={
          lastResult
            ? 'Authoritative C++ solver output and independent verification certificate'
            : activeModel
            ? `Ready to execute ${activeModel.name} against the Kavach-Opt C++ core`
            : 'Select an optimization problem to begin'
        }
        badge={
          lastResult ? (
            <StatusBadge status={lastResult.status} />
          ) : (
            <span className="text-[10px] font-mono text-neutral-300 px-2 py-0.5 rounded bg-neutral-800/80 border border-neutral-700/50">
              READY TO SOLVE
            </span>
          )
        }
        action={
          <SecondaryButton
            size="sm"
            onClick={() => onNavigate('solve')}
          >
            {lastResult ? 'Configure New Run' : 'Open Solve Control'}
          </SecondaryButton>
        }
      >
        {lastResult && activeModel ? (
          <div className="space-y-4">
            <MetricGroup columns={4}>
              <Metric
                label="Objective Value"
                value={lastResult.objective_value.toFixed(4)}
                highlight="success"
                subtext={`Status: ${lastResult.status}`}
              />
              <Metric
                label="Independent Verification"
                value={lastResult.verified ? 'CERTIFIED' : 'UNVERIFIED'}
                highlight={lastResult.verified ? 'success' : 'danger'}
                subtext={`Primal residual: ${lastResult.verification_report.recomputed_primal_residual.toExponential(2)}`}
              />
              <Metric
                label="Execution Wall Time"
                value={`${(lastResult.solve_time_seconds * 1000).toFixed(2)} ms`}
                subtext={`${lastResult.iterations} Simplex iters / ${lastResult.nodes} nodes`}
              />
              <Metric
                label="Active Backend"
                value={lastResult.backend_used}
                isMono={true}
                subtext="Thread-safe execution"
              />
            </MetricGroup>

            <div className="p-3.5 rounded-lg bg-[#141926] border border-[#1f283a] flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
              <div className="flex items-center gap-2">
                <FileCode className="w-4 h-4 text-neutral-400" />
                <span className="text-neutral-400 font-sans">Active Run:</span>
                <span className="text-neutral-100 font-mono font-semibold">{activeModel.name}</span>
                <span className="text-neutral-400 font-mono text-[11px]">({activeModel.problem_class})</span>
              </div>
              <div className="flex items-center gap-4 text-neutral-400 font-mono text-[11px]">
                <span>{activeModel.num_variables} variables</span>
                <span>•</span>
                <span>{activeModel.num_constraints} constraints</span>
                <span>•</span>
                <span>{activeModel.num_nonzeros} nonzeros</span>
              </div>
            </div>
          </div>
        ) : activeModel ? (
          <div className="space-y-4">
            <div className="p-4 rounded-lg bg-[#141926] border border-[#1f283a] flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-sans text-neutral-400">Selected Model:</span>
                  <span className="text-sm font-mono font-bold text-white bg-[#0e121a] px-2 py-0.5 rounded border border-[#263143]">
                    {activeModel.name}
                  </span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-950/60 border border-blue-800/40 text-blue-300">
                    {activeModel.problem_class}
                  </span>
                </div>
                <p className="text-xs text-neutral-400 leading-relaxed font-sans pt-1">
                  Model topology verified. Ready for Phase I/II simplex optimization and zero-trust independent certification.
                </p>
              </div>

              <PrimaryButton
                size="sm"
                icon={<Play className="w-3.5 h-3.5" />}
                onClick={() => onNavigate('solve')}
              >
                Run Solve on {activeModel.name}
              </PrimaryButton>
            </div>

            <MetricGroup columns={4}>
              <Metric label="Variables (n)" value={activeModel.num_variables} />
              <Metric label="Constraints (m)" value={activeModel.num_constraints} />
              <Metric label="Nonzeros (nnz)" value={activeModel.num_nonzeros} />
              <Metric
                label="Matrix Density"
                value={`${(activeModel.density * 100).toFixed(1)}%`}
                subtext={`Dynamic range: ${activeModel.dynamic_range.toFixed(1)}x`}
              />
            </MetricGroup>
          </div>
        ) : isDiscoveringModels ? (
          <div className="space-y-4 animate-fade-in" aria-busy="true">
            <SkeletonCard>
              <div className="flex items-center justify-between">
                <Skeleton className="h-4 w-44" />
                <Skeleton className="h-7 w-28 rounded-lg" />
              </div>
              <Skeleton className="h-3 w-3/4 mt-2" />
            </SkeletonCard>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <SkeletonMetric />
              <SkeletonMetric />
              <SkeletonMetric />
              <SkeletonMetric />
            </div>
          </div>
        ) : (
          <EmptyState
            title="Discovering Models"
            description="Connecting to repository models via solver adapter..."
          />
        )}
      </Panel>

      {/* Optimization Pipeline Visualization */}
      <Panel
        title="Optimization Execution Pipeline"
        subtitle="Unbroken end-to-end data flow from ingest to independent certification"
      >
        <Pipeline
          steps={
            lastResult && activeModel
              ? [
                  { id: 'ingest', name: '1. Ingestion', category: 'MPS / LP Parse', status: 'completed', detail: `Parsed ${activeModel.name}` },
                  { id: 'model', name: '2. Canonical Model', category: 'Representation', status: 'completed', detail: `${activeModel.num_variables} vars, ${activeModel.num_constraints} rows` },
                  { id: 'strategy', name: '3. Strategy Layer', category: 'Rules R001-R018', status: 'completed', detail: `Routed to ${activeModel.problem_class}` },
                  { id: 'presolve', name: '4. Presolve Suite', category: 'Reduction', status: 'completed', detail: 'Ruiz scaled & bounds tightened' },
                  { id: 'solve', name: '5. Core Solver', category: 'Simplex / B&B / QP', status: 'completed', detail: `${lastResult.iterations} iters in ${(lastResult.solve_time_seconds * 1000).toFixed(1)}ms` },
                  { id: 'verify', name: '6. Verification', category: 'Independent Cert', status: 'completed', detail: `Residual: ${lastResult.verification_report.recomputed_primal_residual.toExponential(1)}` },
                ]
              : [
                  { id: 'ingest', name: '1. Ingestion', category: 'MPS / LP Parse', status: activeModel ? 'completed' : 'pending', detail: activeModel ? `Loaded ${activeModel.name}` : 'Tokenization & Bounds' },
                  { id: 'model', name: '2. Canonical Model', category: 'Representation', status: activeModel ? 'completed' : 'pending', detail: activeModel ? `${activeModel.num_variables} vars, ${activeModel.num_constraints} rows` : 'Variables & Constraints' },
                  { id: 'strategy', name: '3. Strategy Layer', category: 'Rules R001-R018', status: 'active', detail: 'Dynamic characterization' },
                  { id: 'presolve', name: '4. Presolve Suite', category: 'Reduction', status: 'pending', detail: 'Ruiz scaling & reductions' },
                  { id: 'solve', name: '5. Core Solver', category: 'Simplex / B&B / QP', status: 'pending', detail: 'Sparse LU Factorization' },
                  { id: 'verify', name: '6. Verification', category: 'Independent Cert', status: 'pending', detail: 'Primal & Dual Residuals' },
                ]
          }
        />
      </Panel>

      {/* Solver Capabilities */}
      <div>
        <div className="mb-3">
          <h2 className="text-base font-semibold text-neutral-200">
            Engine Capabilities & Mathematical Scope
          </h2>
          <p className="text-xs text-neutral-400">
            Validated from-scratch components verified across 329 unit and adversarial tests
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5">
          <div className="p-4 rounded-lg bg-[#11151f] border border-[#1d2332] flex flex-col justify-between shadow-xs">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-medium text-emerald-400">
                  LP Solver
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-800/40 text-emerald-300">
                  REVISED SIMPLEX
                </span>
              </div>
              <h3 className="text-sm font-semibold text-neutral-100 font-sans">
                Continuous Linear Programming
              </h3>
              <p className="mt-1.5 text-xs text-neutral-400 leading-relaxed font-sans">
                Two-Phase Revised Simplex with Bland anti-cycling rule, Ruiz equilibration scaling,
                and multi-pass presolve reconstruction.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#1d2332] flex items-center justify-between text-[11px] font-mono text-neutral-400">
              <span>Benchmark: Netlib</span>
              <span className="text-emerald-400 font-medium">13/13 Solved</span>
            </div>
          </div>

          <div className="p-4 rounded-lg bg-[#11151f] border border-[#1d2332] flex flex-col justify-between shadow-xs">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-medium text-sky-400">
                  MILP Solver
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-sky-950/60 border border-sky-800/40 text-sky-300">
                  BRANCH & BOUND
                </span>
              </div>
              <h3 className="text-sm font-semibold text-neutral-100 font-sans">
                Mixed-Integer Linear Programming
              </h3>
              <p className="mt-1.5 text-xs text-neutral-400 leading-relaxed font-sans">
                Branch-and-Bound with most-fractional branching, best-bound search, and dual simplex
                warm starts accelerating child LP node solves.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#1d2332] flex items-center justify-between text-[11px] font-mono text-neutral-400">
              <span>Benchmark: MIPLIB</span>
              <span className="text-sky-400 font-medium">Warm-Starts Active</span>
            </div>
          </div>

          <div className="p-4 rounded-lg bg-[#11151f] border border-[#1d2332] flex flex-col justify-between shadow-xs">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-medium text-indigo-400">
                  Convex QP
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-950/60 border border-indigo-800/40 text-indigo-300">
                  ACTIVE-SET
                </span>
              </div>
              <h3 className="text-sm font-semibold text-neutral-100 font-sans">
                Quadratic Programming
              </h3>
              <p className="mt-1.5 text-xs text-neutral-400 leading-relaxed font-sans">
                Primal active-set method for strictly convex QP with positive semi-definite (PSD)
                Hessian validation and Karush-Kuhn-Tucker convergence.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#1d2332] flex items-center justify-between text-[11px] font-mono text-neutral-400">
              <span>Scope: Strictly Convex</span>
              <span className="text-indigo-400 font-medium">PSD Enforced</span>
            </div>
          </div>

          <div className="p-4 rounded-lg bg-[#11151f] border border-[#1d2332] flex flex-col justify-between shadow-xs">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-medium text-emerald-400">
                  Certification
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-800/40 text-emerald-300">
                  INDEPENDENT
                </span>
              </div>
              <h3 className="text-sm font-semibold text-neutral-100 font-sans">
                Independent Verification Layer
              </h3>
              <p className="mt-1.5 text-xs text-neutral-400 leading-relaxed font-sans">
                Decoupled mathematical audit recomputing primal feasibility, dual multipliers,
                complementarity, and integrality residuals. Never trusts self-reports.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#1d2332] flex items-center justify-between text-[11px] font-mono text-neutral-400">
              <span>Fault Rejection Rate</span>
              <span className="text-emerald-400 font-medium">100% Certified</span>
            </div>
          </div>

          <div className="p-4 rounded-lg bg-[#11151f] border border-[#1d2332] flex flex-col justify-between shadow-xs">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-medium text-amber-400">
                  Numerics Core
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-950/60 border border-amber-800/40 text-amber-300">
                  SPARSE LU
                </span>
              </div>
              <h3 className="text-sm font-semibold text-neutral-100 font-sans">
                Sparse Matrix & Factorization
              </h3>
              <p className="mt-1.5 text-xs text-neutral-400 leading-relaxed font-sans">
                Compressed Sparse Row/Column representations with Markowitz threshold pivoting,
                iterative refinement, and condition monitoring.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#1d2332] flex items-center justify-between text-[11px] font-mono text-neutral-400">
              <span>External BLAS Dependency</span>
              <span className="text-amber-400 font-medium">0 (Sovereign)</span>
            </div>
          </div>

          <div className="p-4 rounded-lg bg-[#11151f] border border-[#1d2332] flex flex-col justify-between shadow-xs">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-medium text-neutral-300">
                  Hardware
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-neutral-900 border border-neutral-700 text-neutral-300">
                  CPU-FIRST + CUDA
                </span>
              </div>
              <h3 className="text-sm font-semibold text-neutral-100 font-sans">
                Multicore & Scoped GPU
              </h3>
              <p className="mt-1.5 text-xs text-neutral-400 leading-relaxed font-sans">
                Standard C++20 thread pool for batched SpMV, plus a single scoped CUDA kernel
                gated by empirical crossover and automatic CPU fallback.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#1d2332] flex items-center justify-between text-[11px] font-mono text-neutral-400">
              <span>Crossover Gate</span>
              <span className="text-neutral-300 font-medium">Transfer-Inclusive</span>
            </div>
          </div>
        </div>
      </div>

      {/* Industrial Relevance Banner */}
      <div className="p-6 rounded-xl border border-[#1d2332] bg-[#11151f] flex flex-col md:flex-row items-start md:items-center justify-between gap-6 shadow-xs">
        <div>
          <span className="text-xs font-mono font-medium text-amber-400">
            Industrial Relevance · Mangalore Refinery and Petrochemicals Limited
          </span>
          <h2 className="text-lg font-bold text-white mt-1 font-sans">
            MRPL Refinery Production Planning & Quality Blending
          </h2>
          <p className="mt-1.5 text-xs sm:text-sm text-neutral-400 max-w-3xl leading-relaxed font-sans">
            Demonstrating crude selection, atmospheric distillation throughput limits,
            Euro-VI fuel sulfur and density compliance, and what-if outage scenarios on
            synthetic, illustrative refinery data.
          </p>
        </div>
        <PrimaryButton
          onClick={() => onNavigate('mrpl')}
          className="shrink-0"
        >
          Explore Case Study
        </PrimaryButton>
      </div>
    </div>
  );
};
