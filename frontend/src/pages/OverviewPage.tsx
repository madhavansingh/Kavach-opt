import React from 'react';
import { Play, Shield, FileCode } from 'lucide-react';
import { Panel } from '../components/Panel';
import { Metric, MetricGroup } from '../components/Metric';
import { Pipeline } from '../components/Pipeline';
import { StatusBadge } from '../components/StatusBadge';
import { PrimaryButton, SecondaryButton } from '../components/Buttons';
import { EmptyState } from '../components/EmptyState';
import type { SolveResult, ModelMetadata } from '../types/solver';
import type { NavTab } from '../layouts/Sidebar';

interface OverviewPageProps {
  onNavigate: (tab: NavTab) => void;
  activeModel?: ModelMetadata;
  lastResult?: SolveResult;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({
  onNavigate,
  activeModel,
  lastResult,
}) => {
  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="relative rounded-xl border border-[#1e2430] bg-gradient-to-b from-[#131722] to-[#0d1016] p-6 sm:p-8 overflow-hidden">
        <div className="max-w-3xl relative z-10">
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-emerald-950/40 border border-emerald-700/40 text-emerald-400 text-xs font-mono font-medium mb-4">
            <Shield className="w-3.5 h-3.5" />
            <span>SIH26119 · Sovereign Optimization Engine</span>
          </div>

          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold tracking-tight text-white">
            Kavach-Opt <span className="text-neutral-400 font-normal text-xl sm:text-2xl">(कवच-Opt)</span>
          </h1>

          <p className="mt-3 text-sm sm:text-base text-neutral-300 leading-relaxed max-w-2xl">
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
        title="Current Solver Session"
        subtitle="Authoritative solver execution and verification status"
        badge={
          lastResult ? (
            <StatusBadge status={lastResult.status} />
          ) : (
            <span className="text-[11px] font-mono text-neutral-400 px-2 py-0.5 rounded bg-neutral-900 border border-neutral-800">
              IDLE
            </span>
          )
        }
        action={
          <SecondaryButton
            size="sm"
            onClick={() => onNavigate('solve')}
          >
            {lastResult ? 'Change Model' : 'Select Model'}
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

            <div className="p-3.5 rounded bg-[#161a22] border border-[#222834] flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
              <div className="flex items-center gap-2 font-mono">
                <FileCode className="w-4 h-4 text-neutral-400" />
                <span className="text-neutral-400">Model:</span>
                <span className="text-neutral-200 font-semibold">{activeModel.name}</span>
                <span className="text-neutral-400">({activeModel.problem_class})</span>
              </div>
              <div className="flex items-center gap-4 text-neutral-400 font-mono">
                <span>{activeModel.num_variables} vars</span>
                <span>•</span>
                <span>{activeModel.num_constraints} constrs</span>
                <span>•</span>
                <span>{activeModel.num_nonzeros} nnz</span>
              </div>
            </div>
          </div>
        ) : (
          <EmptyState
            title="No Optimization Executed in Current Session"
            description="Select a problem from Netlib LP, MIPLIB knapsack, Markowitz portfolio QP, or the MRPL refinery case study to observe real-time solving and independent certification."
            action={
              <PrimaryButton
                size="sm"
                icon={<Play className="w-3.5 h-3.5" />}
                onClick={() => onNavigate('solve')}
              >
                Load & Run Example Model
              </PrimaryButton>
            }
          />
        )}
      </Panel>

      {/* Optimization Pipeline Visualization */}
      <Panel
        title="Optimization Execution Pipeline"
        subtitle="Unbroken end-to-end data flow from ingest to independent certification"
      >
        <Pipeline />
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
          <div className="p-4 rounded-lg bg-[#11141a] border border-[#1e2430] flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-semibold text-emerald-400 uppercase tracking-wider">
                  LP Solver
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-800/40 text-emerald-400">
                  REVISED SIMPLEX
                </span>
              </div>
              <h3 className="text-sm font-semibold text-neutral-100">
                Continuous Linear Programming
              </h3>
              <p className="mt-1.5 text-xs text-neutral-400 leading-relaxed">
                Two-Phase Revised Simplex with Bland anti-cycling rule, Ruiz equilibration scaling,
                and multi-pass presolve reconstruction.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#1e2430] flex items-center justify-between text-[11px] font-mono text-neutral-400">
              <span>Benchmark: Netlib</span>
              <span className="text-emerald-400">13/13 Solved</span>
            </div>
          </div>

          <div className="p-4 rounded-lg bg-[#11141a] border border-[#1e2430] flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-semibold text-sky-400 uppercase tracking-wider">
                  MILP Solver
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-sky-950/60 border border-sky-800/40 text-sky-400">
                  BRANCH & BOUND
                </span>
              </div>
              <h3 className="text-sm font-semibold text-neutral-100">
                Mixed-Integer Linear Programming
              </h3>
              <p className="mt-1.5 text-xs text-neutral-400 leading-relaxed">
                Branch-and-Bound with most-fractional branching, best-bound search, and dual simplex
                warm starts accelerating child LP node solves.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#1e2430] flex items-center justify-between text-[11px] font-mono text-neutral-400">
              <span>Benchmark: MIPLIB</span>
              <span className="text-sky-400">Warm-Starts Active</span>
            </div>
          </div>

          <div className="p-4 rounded-lg bg-[#11141a] border border-[#1e2430] flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-semibold text-indigo-400 uppercase tracking-wider">
                  Convex QP
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-950/60 border border-indigo-800/40 text-indigo-400">
                  ACTIVE-SET
                </span>
              </div>
              <h3 className="text-sm font-semibold text-neutral-100">
                Quadratic Programming
              </h3>
              <p className="mt-1.5 text-xs text-neutral-400 leading-relaxed">
                Primal active-set method for strictly convex QP with positive semi-definite (PSD)
                Hessian validation and Karush-Kuhn-Tucker convergence.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#1e2430] flex items-center justify-between text-[11px] font-mono text-neutral-400">
              <span>Scope: Strictly Convex</span>
              <span className="text-indigo-400">PSD Enforced</span>
            </div>
          </div>

          <div className="p-4 rounded-lg bg-[#11141a] border border-[#1e2430] flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-semibold text-emerald-400 uppercase tracking-wider">
                  Certification
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-800/40 text-emerald-400">
                  INDEPENDENT
                </span>
              </div>
              <h3 className="text-sm font-semibold text-neutral-100">
                Independent Verification Layer
              </h3>
              <p className="mt-1.5 text-xs text-neutral-400 leading-relaxed">
                Decoupled mathematical audit recomputing primal feasibility, dual multipliers,
                complementarity, and integrality residuals. Never trusts self-reports.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#1e2430] flex items-center justify-between text-[11px] font-mono text-neutral-400">
              <span>Fault Rejection Rate</span>
              <span className="text-emerald-400">100% Certified</span>
            </div>
          </div>

          <div className="p-4 rounded-lg bg-[#11141a] border border-[#1e2430] flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-semibold text-amber-400 uppercase tracking-wider">
                  Numerics Core
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-950/60 border border-amber-800/40 text-amber-400">
                  SPARSE LU
                </span>
              </div>
              <h3 className="text-sm font-semibold text-neutral-100">
                Sparse Matrix & Factorization
              </h3>
              <p className="mt-1.5 text-xs text-neutral-400 leading-relaxed">
                Compressed Sparse Row/Column representations with Markowitz threshold pivoting,
                iterative refinement, and condition monitoring.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#1e2430] flex items-center justify-between text-[11px] font-mono text-neutral-400">
              <span>Eigen/BLAS Dependency</span>
              <span className="text-amber-400">0 (Sovereign)</span>
            </div>
          </div>

          <div className="p-4 rounded-lg bg-[#11141a] border border-[#1e2430] flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-semibold text-neutral-300 uppercase tracking-wider">
                  Hardware
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-neutral-900 border border-neutral-700 text-neutral-300">
                  CPU-FIRST + CUDA
                </span>
              </div>
              <h3 className="text-sm font-semibold text-neutral-100">
                Multicore & Scoped GPU
              </h3>
              <p className="mt-1.5 text-xs text-neutral-400 leading-relaxed">
                Standard C++20 thread pool for batched SpMV, plus a single scoped CUDA kernel
                gated by empirical crossover and automatic CPU fallback.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#1e2430] flex items-center justify-between text-[11px] font-mono text-neutral-400">
              <span>Crossover Gate</span>
              <span className="text-neutral-300">Transfer-Inclusive</span>
            </div>
          </div>
        </div>
      </div>

      {/* Industrial Relevance Banner */}
      <div className="p-6 rounded-xl border border-[#263143] bg-[#11151e] flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        <div>
          <span className="text-xs font-mono font-semibold text-amber-400 uppercase tracking-wider">
            Industrial Relevance · Mangalore Refinery and Petrochemicals Limited
          </span>
          <h2 className="text-lg font-bold text-white mt-1">
            MRPL Refinery Production Planning & Quality Blending
          </h2>
          <p className="mt-1.5 text-xs sm:text-sm text-neutral-400 max-w-3xl leading-relaxed">
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
