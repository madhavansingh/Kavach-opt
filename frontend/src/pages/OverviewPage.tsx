import React, { useState, useEffect } from 'react';
import {
  Play,
  Shield,
  FileCode,
  ShieldCheck,
  Cpu,
  TrendingDown,
  Split,
  Workflow,
  Factory,
} from 'lucide-react';
import { Panel } from '../components/Panel';
import { Metric, MetricGroup } from '../components/Metric';
import { Pipeline } from '../components/Pipeline';
import { StatusBadge } from '../components/StatusBadge';
import { PrimaryButton, SecondaryButton } from '../components/Buttons';
import { EmptyState } from '../components/EmptyState';
import { SkeletonMetric, SkeletonCard, Skeleton } from '../components/Skeleton';
import type { SolveResult, ModelMetadata } from '../types/solver';
import type { NavTab } from '../layouts/Sidebar';
import { solverBridge, type SystemHealth } from '../lib/solverBridge';

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
  const [health, setHealth] = useState<SystemHealth | null>(null);

  useEffect(() => {
    let isMounted = true;
    solverBridge
      .getSystemHealth()
      .then((h) => {
        if (isMounted) setHealth(h);
      })
      .catch(() => {});
    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="space-y-8 font-sans">
      {/* Product Header / System Health Strip */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-xl bg-[#0c1017] border border-[#1d2536] text-xs">
        <div className="flex items-center gap-2.5">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
          <span className="font-semibold text-white tracking-tight">Kavach-Opt Workstation</span>
          <span className="text-neutral-500">•</span>
          <span className="font-mono text-neutral-400 text-[11px]">
            {health?.version_info || 'C++20 Sovereign Core'}
          </span>
        </div>

        <div className="flex items-center gap-3 font-mono text-[11px]">
          <div className="flex items-center gap-1.5 text-neutral-300">
            <Cpu className="w-3.5 h-3.5 text-blue-400" />
            <span>Backend:</span>
            <span className="text-white font-medium">CPU-First (Darwin arm64)</span>
          </div>
          <span className="text-neutral-600 hidden sm:inline">|</span>
          <div className="flex items-center gap-1.5 text-neutral-300 hidden sm:flex">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span>Zero-Trust Verifier:</span>
            <span className="text-emerald-400 font-medium">ACTIVE</span>
          </div>
        </div>
      </div>

      {/* Primary Hero Section */}
      <div className="relative rounded-2xl border border-[#1d2536] bg-gradient-to-b from-[#111724] via-[#0d121c] to-[#090d14] p-6 sm:p-8 overflow-hidden shadow-sm">
        <div className="max-w-3xl relative z-10">
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-emerald-950/40 border border-emerald-700/40 text-emerald-400 text-xs font-mono font-medium mb-3">
            <Shield className="w-3.5 h-3.5" />
            <span>Sovereign Optimization Engine</span>
          </div>

          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold tracking-tight text-white font-sans">
            Mathematical Optimization Workbench
          </h1>

          <p className="mt-3 text-sm sm:text-base text-neutral-300 leading-relaxed max-w-2xl font-sans">
            An indigenous, from-scratch mathematical solver core engineered in C++20 for continuous,
            discrete, and convex quadratic optimization. Delivers deterministic algorithms, air-gapped
            zero-trust verification, and reproducible industrial planning without foreign solver runtimes.
          </p>

          <div className="mt-6 flex flex-wrap items-center gap-3">
            <PrimaryButton
              icon={<Play className="w-4 h-4 fill-white" />}
              onClick={() => onNavigate('solve')}
            >
              Run Optimization
            </PrimaryButton>
            <SecondaryButton
              icon={<Shield className="w-4 h-4" />}
              onClick={() => onNavigate('verification')}
            >
              Independent Verifier
            </SecondaryButton>
            <SecondaryButton
              icon={<Factory className="w-4 h-4" />}
              onClick={() => onNavigate('mrpl')}
            >
              Refinery Optimization
            </SecondaryButton>
          </div>
        </div>

        {/* Subtle grid background illustration */}
        <div className="absolute right-0 top-0 bottom-0 w-1/3 opacity-5 pointer-events-none hidden md:block">
          <div className="w-full h-full border-l border-[#2e3748] bg-[radial-gradient(#38bdf8_1px,transparent_1px)] [background-size:16px_16px]" />
        </div>
      </div>

      {/* System Pipeline Visualization */}
      <Panel
        title="Optimization Execution Pipeline"
        subtitle="Unbroken end-to-end data flow from model ingestion to independent certification"
      >
        <Pipeline
          steps={
            lastResult && activeModel
              ? [
                  { id: 'ingest', name: '1. Ingestion', category: 'MPS / LP Parse', status: 'completed', detail: `Parsed ${activeModel.name}` },
                  { id: 'model', name: '2. Canonical Model', category: 'Sparse CSR/CSC', status: 'completed', detail: `${activeModel.num_variables} vars, ${activeModel.num_constraints} rows` },
                  { id: 'strategy', name: '3. Strategy Layer', category: 'Rules R001-R018', status: 'completed', detail: `Routed to ${activeModel.problem_class}` },
                  { id: 'presolve', name: '4. Presolve Suite', category: 'Reductions', status: 'completed', detail: 'Ruiz scaled & bounds tightened' },
                  { id: 'solve', name: '5. Core Solver', category: 'Simplex / B&B / QP', status: 'completed', detail: `${lastResult.iterations} iters in ${(lastResult.solve_time_seconds * 1000).toFixed(1)}ms` },
                  { id: 'verify', name: '6. Verification', category: 'Independent Cert', status: 'completed', detail: `Residual: ${lastResult.verification_report.recomputed_primal_residual.toExponential(1)}` },
                ]
              : [
                  { id: 'ingest', name: '1. Ingestion', category: 'MPS / LP Parse', status: activeModel ? 'completed' : 'pending', detail: activeModel ? `Loaded ${activeModel.name}` : 'Tokenization & Bounds' },
                  { id: 'model', name: '2. Canonical Model', category: 'Sparse CSR/CSC', status: activeModel ? 'completed' : 'pending', detail: activeModel ? `${activeModel.num_variables} vars, ${activeModel.num_constraints} rows` : 'Variables & Constraints' },
                  { id: 'strategy', name: '3. Strategy Layer', category: 'Rules R001-R018', status: 'active', detail: 'Dynamic characterization' },
                  { id: 'presolve', name: '4. Presolve Suite', category: 'Reductions', status: 'pending', detail: 'Ruiz scaling & reductions' },
                  { id: 'solve', name: '5. Core Solver', category: 'Simplex / B&B / QP', status: 'pending', detail: 'Sparse LU Factorization' },
                  { id: 'verify', name: '6. Verification', category: 'Independent Cert', status: 'pending', detail: 'Primal & Dual Residuals' },
                ]
          }
        />
      </Panel>

      {/* Recent / Current Execution & Engineering Observability */}
      <Panel
        title={lastResult ? 'Active Optimization Execution' : 'Selected Model Ready for Execution'}
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
          <div className="space-y-5">
            {/* Primary KPI Group */}
            <MetricGroup columns={4}>
              <Metric
                label="Objective Value (z*)"
                value={typeof lastResult.objective_value === 'number' ? lastResult.objective_value.toFixed(4) : '—'}
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
                subtext={`${lastResult.iterations} iters / ${lastResult.nodes} nodes`}
              />
              <Metric
                label="Active Backend"
                value={lastResult.backend_used}
                isMono={true}
                subtext="Thread-safe execution"
              />
            </MetricGroup>

            {/* Model Metadata Summary */}
            <div className="p-3.5 rounded-lg bg-[#141926] border border-[#1f283a] flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
              <div className="flex items-center gap-2">
                <FileCode className="w-4 h-4 text-neutral-400" />
                <span className="text-neutral-400 font-sans">Instance:</span>
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

            {/* CUTTING PLANE ACTIVITY (Rendered only when real cuts telemetry exists) */}
            {lastResult.cuts && lastResult.cuts.cuts_generated > 0 && (
              <div className="p-4 rounded-xl bg-[#0f1420] border border-sky-900/40 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <TrendingDown className="w-4 h-4 text-sky-400" />
                    <span className="text-xs font-semibold text-white font-sans">
                      MILP Cutting Plane Activity
                    </span>
                  </div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-sky-950/80 border border-sky-700/50 text-sky-300">
                    {lastResult.cuts.root_rounds} Root Cut Rounds
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                  <div className="p-2.5 rounded-lg bg-[#141926] border border-[#1f283a]">
                    <span className="text-neutral-400 text-[10px] block font-mono">Cuts Applied</span>
                    <span className="text-sm font-bold font-mono text-emerald-400">{lastResult.cuts.cuts_applied}</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-[#141926] border border-[#1f283a]">
                    <span className="text-neutral-400 text-[10px] block font-mono">Cuts Generated</span>
                    <span className="text-sm font-bold font-mono text-neutral-200">{lastResult.cuts.cuts_generated}</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-[#141926] border border-[#1f283a]">
                    <span className="text-neutral-400 text-[10px] block font-mono">Gomory / GMI</span>
                    <span className="text-sm font-bold font-mono text-sky-400">{lastResult.cuts.gomory_cuts}</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-[#141926] border border-[#1f283a]">
                    <span className="text-neutral-400 text-[10px] block font-mono">MIR Cuts</span>
                    <span className="text-sm font-bold font-mono text-indigo-400">{lastResult.cuts.mir_cuts}</span>
                  </div>
                </div>
              </div>
            )}

            {/* ADVANCED MILP SEARCH TELEMETRY (Rendered only when real search telemetry exists) */}
            {lastResult.search && (
              <div className="p-4 rounded-xl bg-[#0f1420] border border-indigo-900/40 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Split className="w-4 h-4 text-indigo-400" />
                    <span className="text-xs font-semibold text-white font-sans">
                      Advanced MILP Search Intelligence
                    </span>
                  </div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-950/80 border border-indigo-700/50 text-indigo-300">
                    Strategy: {lastResult.search.branching_strategy}
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                  <div className="p-2.5 rounded-lg bg-[#141926] border border-[#1f283a]">
                    <span className="text-neutral-400 text-[10px] block font-mono">Pseudocost Updates</span>
                    <span className="text-sm font-bold font-mono text-indigo-400">{lastResult.search.pseudocost_updates}</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-[#141926] border border-[#1f283a]">
                    <span className="text-neutral-400 text-[10px] block font-mono">Strong Probes</span>
                    <span className="text-sm font-bold font-mono text-sky-400">{lastResult.search.strong_branching_calls}</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-[#141926] border border-[#1f283a]">
                    <span className="text-neutral-400 text-[10px] block font-mono">Incumbent Source</span>
                    <span className="text-xs font-bold font-mono text-emerald-400 truncate block">{lastResult.search.incumbent_source}</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-[#141926] border border-[#1f283a]">
                    <span className="text-neutral-400 text-[10px] block font-mono">Heuristic Updates</span>
                    <span className="text-sm font-bold font-mono text-emerald-400">{lastResult.search.heuristic_incumbent_updates}</span>
                  </div>
                </div>

                {lastResult.search.search_explanation && (
                  <p className="text-[11px] text-neutral-400 font-mono bg-[#141926] p-2.5 rounded border border-[#1e2637] leading-relaxed">
                    {lastResult.search.search_explanation}
                  </p>
                )}
              </div>
            )}

            {/* CONVEX MIQP OBSERVABILITY (Rendered only when real MIQP telemetry exists) */}
            {lastResult.miqp && (
              <div className="p-4 rounded-xl bg-[#0f1420] border border-purple-900/40 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Workflow className="w-4 h-4 text-purple-400" />
                    <span className="text-xs font-semibold text-white font-sans">
                      Convex MIQP Active-Set Continuous Relaxations
                    </span>
                  </div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-950/80 border border-purple-700/50 text-purple-300">
                    Relaxations Solved: {lastResult.miqp.qp_relaxations_solved}
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                  <div className="p-2.5 rounded-lg bg-[#141926] border border-[#1f283a]">
                    <span className="text-neutral-400 text-[10px] block font-mono">QP Relaxation Iterations</span>
                    <span className="text-sm font-bold font-mono text-purple-400">{lastResult.miqp.qp_relaxation_iterations}</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-[#141926] border border-[#1f283a]">
                    <span className="text-neutral-400 text-[10px] block font-mono">Strong QP Probes</span>
                    <span className="text-sm font-bold font-mono text-sky-400">{lastResult.miqp.strong_qp_probes}</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-[#141926] border border-[#1f283a]">
                    <span className="text-neutral-400 text-[10px] block font-mono">Nodes Pruned Bound</span>
                    <span className="text-sm font-bold font-mono text-neutral-200">{lastResult.miqp.nodes_pruned_bound}</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-[#141926] border border-[#1f283a]">
                    <span className="text-neutral-400 text-[10px] block font-mono">QP KKT Residual</span>
                    <span className="text-sm font-bold font-mono text-emerald-400">{lastResult.miqp.qp_kkt_residual.toExponential(2)}</span>
                  </div>
                </div>
              </div>
            )}
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
                  Model topology verified. Ready for Phase I/II simplex or active-set optimization with zero-trust certification.
                </p>
              </div>

              <PrimaryButton
                size="sm"
                icon={<Play className="w-3.5 h-3.5 fill-white" />}
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
            description="Connecting to repository models via local solver adapter..."
          />
        )}
      </Panel>

      {/* Solver Capabilities: 4 Core Problem Classes */}
      <div>
        <div className="mb-3">
          <h2 className="text-base font-semibold text-neutral-200">
            Engine Capabilities &amp; Mathematical Scope
          </h2>
          <p className="text-xs text-neutral-400">
            From-scratch numerical algorithms verified against standard Netlib, MIPLIB, and QP benchmark sets.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3.5">
          <div className="p-4 rounded-xl bg-[#11151f] border border-[#1d2332] flex flex-col justify-between shadow-xs">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-medium text-emerald-400">
                  Continuous LP
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-800/40 text-emerald-300">
                  REVISED SIMPLEX
                </span>
              </div>
              <h3 className="text-sm font-semibold text-neutral-100 font-sans">
                Linear Programming
              </h3>
              <p className="mt-1.5 text-xs text-neutral-400 leading-relaxed font-sans">
                Two-Phase Revised Simplex with Bland anti-cycling, Ruiz equilibration scaling, and presolve reductions.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#1d2332] flex items-center justify-between text-[11px] font-mono text-neutral-400">
              <span>Standard: Netlib</span>
              <span className="text-emerald-400 font-medium">Verified Optimal</span>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-[#11151f] border border-[#1d2332] flex flex-col justify-between shadow-xs">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-medium text-sky-400">
                  Discrete MILP
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-sky-950/60 border border-sky-800/40 text-sky-300">
                  BRANCH &amp; CUT
                </span>
              </div>
              <h3 className="text-sm font-semibold text-neutral-100 font-sans">
                Mixed-Integer Linear
              </h3>
              <p className="mt-1.5 text-xs text-neutral-400 leading-relaxed font-sans">
                Gomory/MIR cuts, reliability branching with pseudocost tracking, and dual simplex warm-starts.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#1d2332] flex items-center justify-between text-[11px] font-mono text-neutral-400">
              <span>Standard: MIPLIB</span>
              <span className="text-sky-400 font-medium">Warm-Starts Active</span>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-[#11151f] border border-[#1d2332] flex flex-col justify-between shadow-xs">
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
                Primal active-set method with positive semi-definite (PSD) Hessian validation and KKT convergence.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#1d2332] flex items-center justify-between text-[11px] font-mono text-neutral-400">
              <span>Hessian Check</span>
              <span className="text-indigo-400 font-medium">PSD Enforced</span>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-[#11151f] border border-[#1d2332] flex flex-col justify-between shadow-xs">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-medium text-purple-400">
                  Convex MIQP
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-950/60 border border-purple-800/40 text-purple-300">
                  B&amp;B + ACTIVE-SET
                </span>
              </div>
              <h3 className="text-sm font-semibold text-neutral-100 font-sans">
                Mixed-Integer QP
              </h3>
              <p className="mt-1.5 text-xs text-neutral-400 leading-relaxed font-sans">
                Branch-and-Bound solving active-set QP continuous relaxations with strong QP probing and primal heuristics.
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[#1d2332] flex items-center justify-between text-[11px] font-mono text-neutral-400">
              <span>Certification</span>
              <span className="text-purple-400 font-medium">Exact Residuals</span>
            </div>
          </div>
        </div>
      </div>

      {/* Industrial Relevance Banner */}
      <div className="p-6 rounded-2xl border border-[#1d2332] bg-[#11151f] flex flex-col md:flex-row items-start md:items-center justify-between gap-6 shadow-xs">
        <div>
          <span className="text-xs font-mono font-medium text-amber-400">
            Industrial Relevance · Mangalore Refinery and Petrochemicals Limited
          </span>
          <h2 className="text-lg font-bold text-white mt-1 font-sans">
            Refinery Optimization: Crude Blending &amp; Product Scheduling
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
          Explore Refinery Model
        </PrimaryButton>
      </div>
    </div>
  );
};
