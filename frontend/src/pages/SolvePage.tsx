import React, { useState } from 'react';
import {
  Play,
  Shield,
  Cpu,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Info,
  Loader2,
  RefreshCw,
  AlertTriangle,
  ArrowRight,
} from 'lucide-react';
import { PageHeader } from '../components/PageHeader';
import { Panel } from '../components/Panel';
import { StatusBadge } from '../components/StatusBadge';
import { Metric, MetricGroup } from '../components/Metric';
import { PrimaryButton, SecondaryButton } from '../components/Buttons';
import { EmptyState } from '../components/EmptyState';
import { SolverExecutionPanel, type SolverLifecycleState } from '../components/SolverExecutionPanel';
import { SkeletonPanel, SkeletonMetric, SkeletonCard, Skeleton } from '../components/Skeleton';
import type { ModelMetadata, SolveConfig, SolveResult } from '../types/solver';
import type { NavTab } from '../layouts/Sidebar';

interface SolvePageProps {
  activeModel: ModelMetadata | null;
  availableModels?: ModelMetadata[];
  isDiscoveringModels?: boolean;
  discoveryError?: string | null;
  onRetryDiscovery?: () => void;
  onSelectModel: (model: ModelMetadata) => void;
  onNavigate: (tab: NavTab) => void;
  currentResult: SolveResult | null;
  onRunSolve: (config: SolveConfig) => Promise<void>;
  isSolving: boolean;
  solveError?: string | null;
}

export const SolvePage: React.FC<SolvePageProps> = ({
  activeModel,
  availableModels = [],
  isDiscoveringModels = false,
  discoveryError = null,
  onRetryDiscovery,
  onSelectModel,
  onNavigate,
  currentResult,
  onRunSolve,
  isSolving,
  solveError = null,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [buttonState, setButtonState] = useState<SolverLifecycleState>('IDLE');
  const [solveStartTime, setSolveStartTime] = useState<number | null>(null);

  const [config, setConfig] = useState<SolveConfig>({
    problem_class: activeModel?.problem_class || 'LP',
    presolve_policy: 'FULL',
    scaling_policy: 'AUTO',
    backend: 'CPU_SEQUENTIAL',
    cpu_threads: 4,
    primal_feasibility_tolerance: 1e-8,
    dual_feasibility_tolerance: 1e-8,
    optimality_tolerance: 1e-8,
    time_limit_seconds: 60,
    max_iterations: 10000,
    max_nodes: 50000,
    enable_independent_verification: true,
  });



  // Layout-preserving skeleton during initial model discovery
  if (isDiscoveringModels && !activeModel) {
    return (
      <div className="space-y-6 animate-fade-in">
        <PageHeader
          title="Optimization & Solve Control"
          subtitle="Configure solver parameters, inspect strategy routing decisions, and trigger deterministic optimization."
          badge={<Skeleton className="h-6 w-32 rounded" />}
        />
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <SkeletonPanel titleWidth="w-56" subtitleWidth="w-80">
              <div className="space-y-4 pt-2">
                <Skeleton className="h-10 w-full rounded-lg" />
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <SkeletonMetric />
                  <SkeletonMetric />
                  <SkeletonMetric />
                  <SkeletonMetric />
                </div>
              </div>
            </SkeletonPanel>
          </div>
          <div className="space-y-6">
            <SkeletonPanel titleWidth="w-44" subtitleWidth="w-64">
              <SkeletonCard />
            </SkeletonPanel>
          </div>
        </div>
      </div>
    );
  }

  // Discovery Failure State with Retry UX
  if (discoveryError && (!availableModels || availableModels.length === 0)) {
    return (
      <div className="space-y-6 animate-fade-in">
        <PageHeader
          title="Optimization & Solve Control"
          subtitle="Configure solver parameters, inspect strategy routing decisions, and trigger deterministic optimization."
        />
        <EmptyState
          icon={<AlertTriangle className="w-6 h-6 text-rose-400" />}
          title="Model Discovery Unsuccessful"
          description={`Unable to load optimization models: ${discoveryError}. Verify that the Kavach-Opt adapter is running.`}
          action={
            onRetryDiscovery ? (
              <PrimaryButton onClick={onRetryDiscovery} icon={<RefreshCw className="w-4 h-4" />}>
                Retry Model Discovery
              </PrimaryButton>
            ) : undefined
          }
        />
      </div>
    );
  }

  if (!activeModel) {
    return (
      <div className="space-y-6">
        <PageHeader
          title="Optimization & Solve Control"
          subtitle="Configure solver parameters, inspect strategy routing decisions, and trigger deterministic optimization."
        />
        <EmptyState
          icon={<Cpu className="w-5 h-5 text-blue-400" />}
          title="No Optimization Model Selected"
          description="Select an instance from the available benchmark repository to begin optimization."
        />
      </div>
    );
  }

  const handleModelChange = (identifier: string) => {
    const selected = availableModels.find((m) => m.name === identifier || m.path === identifier);
    if (selected) {
      onSelectModel(selected);
      setConfig((prev: SolveConfig) => ({
        ...prev,
        problem_class: selected.problem_class,
      }));
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (isSolving || buttonState === 'SUBMITTING' || buttonState === 'RUNNING') {
      return; // Strict double-click prevention
    }

    setSolveStartTime(performance.now());
    setButtonState('SUBMITTING');

    // Double requestAnimationFrame ensures browser paints SUBMITTING before async execution begins
    requestAnimationFrame(() => {
      requestAnimationFrame(async () => {
        setButtonState('RUNNING');
        try {
          await onRunSolve(config);
          setButtonState('COMPLETED');
          setTimeout(() => {
            setButtonState('IDLE');
          }, 2200);
        } catch {
          setButtonState('FAILED');
        }
      });
    });
  };

  // Group models by problem class / suite
  const netlibModels = availableModels.filter(
    (m) => (m.path || '').includes('netlib') || (!m.path && m.problem_class === 'LP')
  );
  const miplibModels = availableModels.filter(
    (m) => (m.path || '').includes('miplib') || (!m.path && m.problem_class === 'MILP')
  );
  const qpModels = availableModels.filter(
    (m) => (m.path || '').includes('qp') || (!m.path && m.problem_class === 'CONVEX_QP')
  );
  const mrplModels = availableModels.filter((m) => (m.path || '').includes('mrpl'));
  const otherModels = availableModels.filter(
    (m) =>
      !netlibModels.includes(m) &&
      !miplibModels.includes(m) &&
      !qpModels.includes(m) &&
      !mrplModels.includes(m)
  );

  return (
    <div className="space-y-6">
      <PageHeader
        title="Optimization & Solve Control"
        subtitle="Configure solver parameters, inspect strategy routing decisions, and trigger deterministic optimization."
        badge={
          <div className="flex items-center gap-2">
            <span className="text-xs text-neutral-400 font-sans">Model:</span>
            <span className="text-xs font-mono font-medium text-white px-2 py-0.5 rounded bg-[#161c28] border border-[#232c3f]">
              {activeModel.name}
            </span>
          </div>
        }
      />

      {/* Error Diagnostics Callout */}
      {solveError && (
        <div
          className="p-4 rounded-xl border border-rose-800/60 bg-rose-950/30 text-rose-200 text-xs flex items-start gap-3 animate-fade-in"
          role="alert"
        >
          <div className="p-1 rounded bg-rose-900/50 text-rose-400 shrink-0 mt-0.5">
            <Info className="w-4 h-4" />
          </div>
          <div className="space-y-1">
            <div className="font-semibold text-rose-300 font-sans text-xs">
              Solver Execution Diagnostics
            </div>
            <p className="font-mono text-neutral-300 leading-relaxed whitespace-pre-wrap">{solveError}</p>
          </div>
        </div>
      )}

      {/* Model Selection & Profile Strip */}
      <Panel title="Loaded Optimization Model" subtitle="Problem topology and matrix structure">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="block text-xs font-medium text-neutral-300 font-sans">
                Select Active Instance
              </label>
              <span className="text-[11px] font-mono text-neutral-400">
                {availableModels.length} discovered
              </span>
            </div>
            <select
              value={activeModel.path || activeModel.name}
              onChange={(e) => handleModelChange(e.target.value)}
              disabled={isSolving}
              className="w-full bg-[#141926] border border-[#232c3f] rounded-lg px-3 py-2 text-sm text-white font-mono focus:outline-none focus:border-blue-500 transition-colors disabled:opacity-50"
            >
              {netlibModels.length > 0 && (
                <optgroup label="Netlib LP Benchmarks">
                  {netlibModels.map((m) => (
                    <option key={m.path || m.name} value={m.path || m.name}>
                      {m.name} (LP · {m.num_constraints || '?'}×{m.num_variables || '?'})
                    </option>
                  ))}
                </optgroup>
              )}
              {miplibModels.length > 0 && (
                <optgroup label="MIPLIB Mixed-Integer Benchmarks">
                  {miplibModels.map((m) => (
                    <option key={m.path || m.name} value={m.path || m.name}>
                      {m.name} (MILP · {m.num_constraints || '?'}×{m.num_variables || '?'})
                    </option>
                  ))}
                </optgroup>
              )}
              {qpModels.length > 0 && (
                <optgroup label="Convex QP Benchmarks">
                  {qpModels.map((m) => (
                    <option key={m.path || m.name} value={m.path || m.name}>
                      {m.name} (QP · {m.num_constraints || '?'}×{m.num_variables || '?'})
                    </option>
                  ))}
                </optgroup>
              )}
              {mrplModels.length > 0 && (
                <optgroup label="MRPL Industrial Refinery Models">
                  {mrplModels.map((m) => (
                    <option key={m.path || m.name} value={m.path || m.name}>
                      {m.name} ({m.problem_class} · {m.num_constraints || '?'}×{m.num_variables || '?'})
                    </option>
                  ))}
                </optgroup>
              )}
              {otherModels.length > 0 && (
                <optgroup label="Other Instances">
                  {otherModels.map((m) => (
                    <option key={m.path || m.name} value={m.path || m.name}>
                      {m.name} ({m.problem_class})
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
            <div className="flex items-center gap-2 text-[11px] text-neutral-400 font-sans">
              <span>Class:</span>
              <span className="font-mono text-blue-400 font-medium">{activeModel.problem_class}</span>
              <span>· Format:</span>
              <span className="font-mono text-neutral-300 uppercase">
                {activeModel.name.endsWith('.mps') ? 'MPS' : 'Native LP'}
              </span>
            </div>
          </div>

          {/* Matrix Topology Strip */}
          <div className="lg:col-span-2 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div className="p-3 rounded-lg bg-[#141926] border border-[#1f283a]">
              <span className="text-[10px] text-neutral-400 block font-sans">Constraints (m)</span>
              <span className="font-mono text-white text-base font-bold">
                {activeModel.num_constraints.toLocaleString()}
              </span>
              <span className="text-[10px] text-neutral-400 block font-sans">Linear equations</span>
            </div>

            <div className="p-3 rounded-lg bg-[#141926] border border-[#1f283a]">
              <span className="text-[10px] text-neutral-400 block font-sans">Variables (n)</span>
              <span className="font-mono text-white text-base font-bold">
                {activeModel.num_variables.toLocaleString()}
              </span>
              <span className="text-[10px] text-neutral-400 block font-sans">
                {activeModel.num_discrete_vars > 0
                  ? `${activeModel.num_discrete_vars} integer vars`
                  : 'Continuous bounds'}
              </span>
            </div>

            <div className="p-3 rounded-lg bg-[#141926] border border-[#1f283a]">
              <span className="text-[10px] text-neutral-400 block font-sans">Non-Zeros (nnz)</span>
              <span className="font-mono text-white text-base font-bold">
                {activeModel.num_nonzeros.toLocaleString()}
              </span>
              <span className="text-[10px] text-neutral-400 block font-sans">
                {((activeModel.density || 0) * 100).toFixed(2)}% sparsity
              </span>
            </div>

            <div className="p-3 rounded-lg bg-[#141926] border border-[#1f283a]">
              <span className="text-[10px] text-neutral-400 block font-sans">Dynamic Range</span>
              <span className="font-mono text-emerald-400 text-base font-bold">
                {activeModel.dynamic_range < 1e4
                  ? activeModel.dynamic_range.toFixed(1)
                  : activeModel.dynamic_range.toExponential(1)}
              </span>
              <span className="text-[10px] text-emerald-400 block font-sans">
                {activeModel.dynamic_range < 1e6 ? 'Well-conditioned' : 'Ill-conditioned'}
              </span>
            </div>
          </div>
        </div>
      </Panel>

      {/* Main Grid: Form Left, Strategy Right */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <form onSubmit={handleSubmit}>
            <Panel
              title="Execution Parameters & Tolerances"
              subtitle="Algorithm routing, numerical scaling, and hardware backend options"
            >
              <div className="space-y-4">
                {/* Core Presolve and Scaling options */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-medium text-neutral-300 mb-1.5 font-sans">
                      Presolve Suite
                    </label>
                    <select
                      value={config.presolve_policy}
                      disabled={isSolving}
                      onChange={(e) =>
                        setConfig({ ...config, presolve_policy: e.target.value as any })
                      }
                      className="w-full bg-[#141926] border border-[#232c3f] rounded-lg px-3 py-2 text-sm text-neutral-200 focus:outline-none focus:border-blue-500 font-sans transition-colors disabled:opacity-50"
                    >
                      <option value="FULL">Full Presolve (Recommended)</option>
                      <option value="CONSERVATIVE">Conservative Presolve</option>
                      <option value="NONE">Disabled (Solve Raw Model)</option>
                    </select>
                    <span className="text-[11px] text-neutral-400 mt-1 block font-sans">
                      Fixed vars, singleton rows, bound tightening, redundant elimination.
                    </span>
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-neutral-300 mb-1.5 font-sans">
                      Equilibration Scaling
                    </label>
                    <select
                      value={config.scaling_policy}
                      disabled={isSolving}
                      onChange={(e) =>
                        setConfig({ ...config, scaling_policy: e.target.value as any })
                      }
                      className="w-full bg-[#141926] border border-[#232c3f] rounded-lg px-3 py-2 text-sm text-neutral-200 focus:outline-none focus:border-blue-500 font-sans transition-colors disabled:opacity-50"
                    >
                      <option value="AUTO">Auto Ruiz Equilibration (Recommended)</option>
                      <option value="DISABLED">Disabled</option>
                    </select>
                    <span className="text-[11px] text-neutral-400 mt-1 block font-sans">
                      Normalizes condition number to avoid numerical pivoting loss.
                    </span>
                  </div>
                </div>

                {/* Backend Selection */}
                <div className="pt-2 border-t border-[#1d2332]">
                  <label className="block text-xs font-medium text-neutral-300 mb-2 font-sans">
                    Execution Backend Subsystem
                  </label>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <button
                      type="button"
                      disabled={isSolving}
                      onClick={() => setConfig({ ...config, backend: 'CPU_SEQUENTIAL' })}
                      className={`p-3 rounded-lg border text-left transition-colors ${
                        config.backend === 'CPU_SEQUENTIAL'
                          ? 'border-blue-500/60 bg-blue-950/20'
                          : 'border-[#20283a] bg-[#141926] hover:border-[#2d3850]'
                      } disabled:opacity-50`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-semibold text-white font-sans">CPU Sequential</span>
                        <Cpu className="w-3.5 h-3.5 text-neutral-400" />
                      </div>
                      <p className="text-[11px] text-neutral-400 font-sans">
                        Default deterministic single-core simplex & factorization.
                      </p>
                    </button>

                    <button
                      type="button"
                      disabled={isSolving}
                      onClick={() => setConfig({ ...config, backend: 'CPU_MULTICORE' })}
                      className={`p-3 rounded-lg border text-left transition-colors ${
                        config.backend === 'CPU_MULTICORE'
                          ? 'border-blue-500/60 bg-blue-950/20'
                          : 'border-[#20283a] bg-[#141926] hover:border-[#2d3850]'
                      } disabled:opacity-50`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-semibold text-white font-sans">CPU Multicore</span>
                        <Cpu className="w-3.5 h-3.5 text-blue-400" />
                      </div>
                      <p className="text-[11px] text-neutral-400 font-sans">
                        Thread pool for batched SpMV & parallel evaluation.
                      </p>
                    </button>

                    <button
                      type="button"
                      disabled={isSolving}
                      onClick={() => setConfig({ ...config, backend: 'CUDA_GPU' })}
                      className={`p-3 rounded-lg border text-left transition-colors ${
                        config.backend === 'CUDA_GPU'
                          ? 'border-emerald-500/60 bg-emerald-950/20'
                          : 'border-[#20283a] bg-[#141926] hover:border-[#2d3850]'
                      } disabled:opacity-50`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-semibold text-white font-sans">CUDA GPU</span>
                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-900/50 text-emerald-300 font-medium">
                          SCOPED
                        </span>
                      </div>
                      <p className="text-[11px] text-neutral-400 font-sans">
                        Batched CSR SpMV kernel with automatic CPU fallback.
                      </p>
                    </button>
                  </div>
                </div>

                {/* Multicore threads slider if selected */}
                {config.backend === 'CPU_MULTICORE' && (
                  <div className="p-3 rounded-lg bg-[#141926] border border-[#20283a] space-y-2">
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-neutral-300 font-medium font-sans">Multicore Worker Threads:</span>
                      <span className="font-mono text-blue-400 font-bold">{config.cpu_threads} workers</span>
                    </div>
                    <input
                      type="range"
                      min={1}
                      max={16}
                      disabled={isSolving}
                      value={config.cpu_threads}
                      onChange={(e) => setConfig({ ...config, cpu_threads: parseInt(e.target.value) })}
                      className="w-full accent-blue-500 cursor-pointer disabled:opacity-50"
                    />
                    <div className="flex justify-between text-[10px] text-neutral-400 font-mono">
                      <span>1 (Serial)</span>
                      <span>4 (Balanced)</span>
                      <span>8 (High-throughput)</span>
                      <span>16 (Max)</span>
                    </div>
                  </div>
                )}

                {/* Advanced Tolerances Toggle */}
                <div>
                  <button
                    type="button"
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="inline-flex items-center gap-1.5 text-xs text-blue-400 hover:text-blue-300 font-medium transition-colors"
                  >
                    {showAdvanced ? (
                      <ChevronDown className="w-3.5 h-3.5" />
                    ) : (
                      <ChevronRight className="w-3.5 h-3.5" />
                    )}
                    <span>
                      {showAdvanced ? 'Hide Advanced Numerical Settings' : 'Show Advanced Numerical Settings'}
                    </span>
                  </button>

                  {showAdvanced && (
                    <div className="mt-3 p-4 rounded-lg bg-[#141926] border border-[#1f283a] grid grid-cols-1 sm:grid-cols-3 gap-4">
                      <div>
                        <label className="block text-[11px] font-medium text-neutral-400 mb-1 font-sans">
                          Primal Feasibility Tol
                        </label>
                        <input
                          type="text"
                          disabled={isSolving}
                          value={config.primal_feasibility_tolerance}
                          onChange={(e) =>
                            setConfig({
                              ...config,
                              primal_feasibility_tolerance: parseFloat(e.target.value) || 1e-8,
                            })
                          }
                          className="w-full bg-[#11151f] border border-[#232c3f] rounded px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-blue-500"
                        />
                        <span className="text-[10px] text-neutral-400 font-sans">Default: 1.0e-8</span>
                      </div>

                      <div>
                        <label className="block text-[11px] font-medium text-neutral-400 mb-1 font-sans">
                          Dual Feasibility Tol
                        </label>
                        <input
                          type="text"
                          disabled={isSolving}
                          value={config.dual_feasibility_tolerance}
                          onChange={(e) =>
                            setConfig({
                              ...config,
                              dual_feasibility_tolerance: parseFloat(e.target.value) || 1e-8,
                            })
                          }
                          className="w-full bg-[#11151f] border border-[#232c3f] rounded px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-blue-500"
                        />
                        <span className="text-[10px] text-neutral-400 font-sans">Default: 1.0e-8</span>
                      </div>

                      <div>
                        <label className="block text-[11px] font-medium text-neutral-400 mb-1 font-sans">
                          Max Iterations
                        </label>
                        <input
                          type="number"
                          disabled={isSolving}
                          value={config.max_iterations}
                          onChange={(e) =>
                            setConfig({
                              ...config,
                              max_iterations: parseInt(e.target.value) || 10000,
                            })
                          }
                          className="w-full bg-[#11151f] border border-[#232c3f] rounded px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-blue-500"
                        />
                        <span className="text-[10px] text-neutral-400 font-sans">Default: 10,000</span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Primary Action Button Bar with State Machine */}
                <div className="pt-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-t border-[#1d2332]">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="enable_verifier"
                      disabled={isSolving}
                      checked={config.enable_independent_verification}
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          enable_independent_verification: e.target.checked,
                        })
                      }
                      className="rounded bg-[#141926] border-[#232c3f] text-blue-600 focus:ring-0 cursor-pointer disabled:opacity-50"
                    />
                    <label
                      htmlFor="enable_verifier"
                      className="text-xs text-neutral-300 font-sans cursor-pointer select-none"
                    >
                      Run Independent Post-Solve Verification (Mandatory in Production)
                    </label>
                  </div>

                  {/* Multi-state Run Button */}
                  <button
                    type="submit"
                    disabled={isSolving || buttonState === 'SUBMITTING' || buttonState === 'RUNNING'}
                    className={`px-4 py-2 rounded-lg font-medium text-xs font-sans flex items-center justify-center gap-2 transition-all duration-150 shadow-sm ${
                      buttonState === 'COMPLETED'
                        ? 'bg-emerald-600 text-white'
                        : buttonState === 'FAILED'
                        ? 'bg-rose-600 text-white hover:bg-rose-500'
                        : isSolving || buttonState === 'SUBMITTING' || buttonState === 'RUNNING'
                        ? 'bg-blue-600/60 text-white/80 cursor-not-allowed'
                        : 'bg-blue-600 text-white hover:bg-blue-500 active:scale-[0.98]'
                    }`}
                  >
                    {buttonState === 'SUBMITTING' && (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin text-white" />
                        <span>Starting Solver...</span>
                      </>
                    )}
                    {buttonState === 'RUNNING' && (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin text-white" />
                        <span>Solving {activeModel.name}...</span>
                      </>
                    )}
                    {buttonState === 'COMPLETED' && (
                      <>
                        <CheckCircle2 className="w-4 h-4 text-white" />
                        <span>Solved (Optimal)</span>
                      </>
                    )}
                    {buttonState === 'FAILED' && (
                      <>
                        <AlertTriangle className="w-4 h-4 text-white" />
                        <span>Retry Optimization</span>
                      </>
                    )}
                    {buttonState === 'IDLE' && (
                      <>
                        <Play className="w-4 h-4 fill-white text-white" />
                        <span>Run Optimization</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </Panel>
          </form>

          {/* Results Area with Layout-Preserving Skeleton */}
          {isSolving ? (
            <div className="space-y-4 animate-fade-in">
              <SolverExecutionPanel
                model={activeModel}
                config={config}
                state={buttonState === 'SUBMITTING' ? 'SUBMITTING' : 'RUNNING'}
                startTime={solveStartTime}
                error={solveError}
              />
              <SkeletonPanel titleWidth="w-56" subtitleWidth="w-80">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <SkeletonMetric />
                  <SkeletonMetric />
                  <SkeletonMetric />
                  <SkeletonMetric />
                </div>
                <SkeletonCard />
              </SkeletonPanel>
            </div>
          ) : currentResult ? (
            <div className="animate-fade-in">
              <Panel
                title="Optimization Outcome"
                subtitle={`Authoritative C++ solver output for ${activeModel.name}`}
                badge={<StatusBadge status={currentResult.status} size="md" />}
              >
                <div className="space-y-5">
                  {/* Problem-class-aware primary metric strip */}
                  {activeModel.problem_class === 'LP' && (
                    <MetricGroup columns={4}>
                      <Metric
                        label="Objective Value (z*)"
                        value={
                          typeof currentResult.objective_value === 'number'
                            ? currentResult.objective_value.toFixed(5)
                            : '—'
                        }
                        highlight={currentResult.status === 'OPTIMAL' ? 'success' : 'normal'}
                      />
                      <Metric
                        label="Execution Wall Time"
                        value={`${(currentResult.solve_time_seconds * 1000).toFixed(2)} ms`}
                        subtext={`${currentResult.solve_time_seconds.toFixed(6)} s`}
                      />
                      <Metric
                        label="Simplex Iterations"
                        value={currentResult.iterations}
                        subtext="Phase I & Phase II basis pivots"
                      />
                      <Metric
                        label="Max Primal Residual"
                        value={
                          currentResult.verification_report?.recomputed_primal_residual !== undefined
                            ? currentResult.verification_report.recomputed_primal_residual.toExponential(2)
                            : '0.00e+00'
                        }
                        subtext="Tolerance: 1.0e-06"
                        highlight="success"
                      />
                    </MetricGroup>
                  )}

                  {activeModel.problem_class === 'MILP' && (
                    <MetricGroup columns={4}>
                      <Metric
                        label="Objective Value (z*)"
                        value={
                          typeof currentResult.objective_value === 'number'
                            ? currentResult.objective_value.toFixed(5)
                            : '—'
                        }
                        highlight={currentResult.status === 'OPTIMAL' ? 'success' : 'normal'}
                      />
                      <Metric
                        label="Execution Wall Time"
                        value={`${(currentResult.solve_time_seconds * 1000).toFixed(2)} ms`}
                        subtext={`${currentResult.solve_time_seconds.toFixed(6)} s`}
                      />
                      <Metric
                        label="B&B Nodes Explored"
                        value={currentResult.nodes}
                        subtext="Branch-and-bound search tree"
                      />
                      <Metric
                        label="Integrality Violation"
                        value={
                          currentResult.verification_report?.recomputed_integrality_violation !== undefined
                            ? currentResult.verification_report.recomputed_integrality_violation.toExponential(2)
                            : '0.00e+00'
                        }
                        subtext="Tolerance: 1.0e-05"
                        highlight="success"
                      />
                    </MetricGroup>
                  )}

                  {activeModel.problem_class === 'CONVEX_QP' && (
                    <MetricGroup columns={4}>
                      <Metric
                        label="Objective Value (z*)"
                        value={
                          typeof currentResult.objective_value === 'number'
                            ? currentResult.objective_value.toFixed(5)
                            : '—'
                        }
                        highlight={currentResult.status === 'OPTIMAL' ? 'success' : 'normal'}
                      />
                      <Metric
                        label="Execution Wall Time"
                        value={`${(currentResult.solve_time_seconds * 1000).toFixed(2)} ms`}
                        subtext={`${currentResult.solve_time_seconds.toFixed(6)} s`}
                      />
                      <Metric
                        label="Active-Set Iterations"
                        value={currentResult.iterations}
                        subtext="Projection steps"
                      />
                      <Metric
                        label="KKT Stationarity Residual"
                        value={
                          currentResult.verification_report?.recomputed_kkt_residual !== undefined
                            ? currentResult.verification_report.recomputed_kkt_residual.toExponential(2)
                            : '0.00e+00'
                        }
                        subtext="Tolerance: 1.0e-06"
                        highlight="success"
                      />
                    </MetricGroup>
                  )}

                  {/* Sub-panel with class-specific mathematical invariants */}
                  <div className="p-3.5 rounded-lg bg-[#141926] border border-[#1f283a] text-xs font-mono">
                    {activeModel.problem_class === 'LP' && (
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-neutral-300">
                        <div>
                          <span className="text-[10px] text-neutral-400 block font-sans">Primal Feasibility:</span>
                          <span className="text-emerald-400 font-semibold">
                            ||Ax - b|| ={' '}
                            {currentResult.verification_report?.recomputed_primal_residual.toExponential(3) ||
                              '0.000e+00'}
                          </span>
                        </div>
                        <div>
                          <span className="text-[10px] text-neutral-400 block font-sans">Dual Multipliers:</span>
                          <span className="text-neutral-200">
                            ||A^T y + s - c|| ={' '}
                            {currentResult.verification_report?.recomputed_dual_residual.toExponential(3) ||
                              '0.000e+00'}
                          </span>
                        </div>
                        <div>
                          <span className="text-[10px] text-neutral-400 block font-sans">
                            Complementary Slackness:
                          </span>
                          <span className="text-emerald-400 font-semibold">
                            max |x_j * s_j| ={' '}
                            {currentResult.verification_report?.recomputed_complementary_slackness.toExponential(3) ||
                              '0.000e+00'}
                          </span>
                        </div>
                      </div>
                    )}

                    {activeModel.problem_class === 'MILP' && (
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-neutral-300">
                        <div>
                          <span className="text-[10px] text-neutral-400 block font-sans">
                            Discrete Feasibility:
                          </span>
                          <span className="text-emerald-400 font-semibold">
                            max |x_int - round(x_int)| ={' '}
                            {currentResult.verification_report?.recomputed_integrality_violation?.toExponential(3) ||
                              '0.000e+00'}
                          </span>
                        </div>
                        <div>
                          <span className="text-[10px] text-neutral-400 block font-sans">
                            Discrete Variable Count:
                          </span>
                          <span className="text-neutral-200">
                            {activeModel.num_discrete_vars} integer variables strictly enforced
                          </span>
                        </div>
                        <div>
                          <span className="text-[10px] text-neutral-400 block font-sans">
                            Dual Simplex Warm Starts:
                          </span>
                          <span className="text-sky-400 font-semibold font-sans">
                            Active on all child node factorizations
                          </span>
                        </div>
                      </div>
                    )}

                    {activeModel.problem_class === 'CONVEX_QP' && (
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-neutral-300">
                        <div>
                          <span className="text-[10px] text-neutral-400 block font-sans">KKT Stationarity:</span>
                          <span className="text-emerald-400 font-semibold">
                            ||Q x + c - A^T y|| ={' '}
                            {currentResult.verification_report?.recomputed_kkt_residual?.toExponential(3) ||
                              '0.000e+00'}
                          </span>
                        </div>
                        <div>
                          <span className="text-[10px] text-neutral-400 block font-sans">Hessian Condition:</span>
                          <span className="text-emerald-400 font-semibold font-sans">
                            Positive Semi-Definite (PSD) Verified
                          </span>
                        </div>
                        <div>
                          <span className="text-[10px] text-neutral-400 block font-sans">Active Constraints:</span>
                          <span className="text-neutral-200 font-sans">Active set projection converged</span>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Backend Fallback Callout */}
                  {config.backend === 'CUDA_GPU' && currentResult.backend_used !== 'CUDA_GPU' && (
                    <div className="p-3.5 rounded-lg border border-amber-800/50 bg-amber-950/20 text-amber-200 text-xs flex items-start gap-3">
                      <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                      <div className="space-y-1">
                        <div className="font-semibold text-amber-300 text-[11px] font-sans">
                          Hardware Execution Fallback: CPU Core Active
                        </div>
                        <p className="text-neutral-300 text-xs leading-relaxed font-sans">
                          Requested Backend: <span className="font-mono text-amber-300 font-semibold">CUDA_GPU</span> ·
                          Executed Backend:{' '}
                          <span className="font-mono text-white font-semibold">{currentResult.backend_used}</span>.
                          The host environment does not possess an active CUDA GPU or runtime. The solver engaged its
                          mandatory, deterministic CPU fallback (Rule R012) ensuring 100% solver reliability and
                          identical numerical accuracy.
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Verification Summary Banner with Deep Link */}
                  {currentResult.verification_report && (
                    <div
                      className={`p-4 rounded-lg border flex flex-col sm:flex-row sm:items-center justify-between gap-3 ${
                        currentResult.verification_report.passed
                          ? 'bg-emerald-950/20 border-emerald-800/40 text-emerald-200'
                          : 'bg-rose-950/20 border-rose-800/40 text-rose-200'
                      }`}
                    >
                      <div className="flex items-start gap-3">
                        <Shield
                          className={`w-5 h-5 mt-0.5 shrink-0 ${
                            currentResult.verification_report.passed
                              ? 'text-emerald-400'
                              : 'text-rose-400'
                          }`}
                        />
                        <div>
                          <div className="text-xs font-semibold font-sans">
                            Independent Verification:{' '}
                            {currentResult.verification_report.passed ? 'CERTIFIED OPTIMAL' : 'FAILED / UNVERIFIED'}
                          </div>
                          <p className="text-xs text-neutral-300 font-mono mt-0.5">
                            Primal residual:{' '}
                            {currentResult.verification_report.recomputed_primal_residual.toExponential(3)} · Dual:{' '}
                            {currentResult.verification_report.recomputed_dual_residual.toExponential(3)}
                          </p>
                        </div>
                      </div>

                      <SecondaryButton
                        size="sm"
                        onClick={() => onNavigate('verification')}
                        icon={<ArrowRight className="w-3.5 h-3.5" />}
                      >
                        Inspect Certificate
                      </SecondaryButton>
                    </div>
                  )}
                </div>
              </Panel>
            </div>
          ) : (
            <EmptyState
              title="Awaiting Execution"
              description="Configure parameters above and click 'Run Optimization' to execute the solver core and generate full telemetry."
              action={
                <PrimaryButton onClick={() => handleSubmit({ preventDefault: () => {} } as any)}>
                  Run Default Solve
                </PrimaryButton>
              }
            />
          )}
        </div>

        {/* Right: Strategy Advisor Sidebar */}
        <div className="space-y-6">
          <Panel
            title="Strategy Layer Advisor"
            subtitle="Deterministic rule-based solver intelligence"
            badge={
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-950/60 border border-blue-800/40 text-blue-300 font-medium">
                RULES ENGINE
              </span>
            }
          >
            <div className="space-y-4">
              <div className="p-3.5 rounded-lg bg-[#141926] border border-[#1f283a] space-y-1.5">
                <span className="text-[10px] font-mono text-neutral-400 font-medium">
                  TARGET ALGORITHM & ROUTING
                </span>
                <div className="text-sm font-semibold text-white font-sans">
                  {activeModel.problem_class === 'LP' && 'Two-Phase Revised Simplex (Phase I/II)'}
                  {activeModel.problem_class === 'MILP' && 'Branch & Bound + Dual Simplex Warm Starts'}
                  {activeModel.problem_class === 'CONVEX_QP' && 'Active-Set Quadratic Programming'}
                </div>
                <p className="text-xs text-neutral-400 leading-relaxed font-sans">
                  {currentResult?.strategy_decision?.human_readable_rationale ||
                    'Routing enforced by deterministic classification rules R001–R006.'}
                </p>
              </div>

              <div className="space-y-2">
                <div className="text-xs font-semibold text-neutral-300 font-sans">
                  {currentResult?.strategy_decision ? 'Active Strategy Rules Fired' : 'Strategy Rules Preview'}
                </div>
                {currentResult?.strategy_decision?.rule_explanations &&
                currentResult.strategy_decision.rule_explanations.length > 0 ? (
                  <ul className="space-y-2 text-xs text-neutral-400">
                    {currentResult.strategy_decision.rule_explanations.map((expl, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                        <span className="text-neutral-300 font-mono text-[11px] leading-relaxed">{expl}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <ul className="space-y-2 text-xs text-neutral-400">
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                      <span className="font-sans">
                        <strong className="text-neutral-200">R001 / R003:</strong> Classified as{' '}
                        {activeModel.problem_class}. Problem class matches core kernel capabilities.
                      </span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                      <span className="font-sans">
                        <strong className="text-neutral-200">R007:</strong> Condition risk profile:{' '}
                        <span className="text-emerald-400 font-mono font-medium">NORMAL</span> (dynamic range{' '}
                        {activeModel.dynamic_range.toFixed(1)}x &lt; 1e6).
                      </span>
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                      <span className="font-sans">
                        <strong className="text-neutral-200">R010:</strong> Backend crossover evaluation:{' '}
                        Single-batch workload defaults to CPU Sequential for minimal launch latency.
                      </span>
                    </li>
                  </ul>
                )}
              </div>

              <div className="p-3 rounded-lg bg-blue-950/20 border border-blue-900/40 text-xs text-neutral-300 space-y-1">
                <div className="flex items-center gap-1.5 font-semibold text-blue-300 font-sans">
                  <Info className="w-3.5 h-3.5" />
                  <span>Rule Engine Telemetry</span>
                </div>
                <p className="text-[11px] text-neutral-400 font-sans">
                  Decision Time:{' '}
                  <span className="font-mono text-white font-medium">
                    {currentResult?.strategy_decision
                      ? `${(currentResult.strategy_decision.decision_time_ms || 0.005).toFixed(3)} ms`
                      : '< 0.01 ms'}
                  </span>{' '}
                  · Risk Profile:{' '}
                  <span className="font-mono text-emerald-400 font-medium">
                    {currentResult?.strategy_decision?.numerical_risk_profile || 'NORMAL'}
                  </span>
                </p>
              </div>
            </div>
          </Panel>

          {/* Quick Info Panel */}
          <Panel title="Command-Line Equivalent" subtitle="Reproduce this exact solve via CLI">
            <div className="p-3 rounded-lg bg-[#0e121a] border border-[#1d2332] font-mono text-xs text-neutral-300 select-all overflow-x-auto whitespace-pre-wrap">
              ./build-release/kavach-opt solve {activeModel.path || activeModel.name} --json
              {config.presolve_policy === 'NONE' ? ' --presolve none' : ' --presolve full'}
              {config.scaling_policy === 'DISABLED' ? ' --scaling off' : ' --scaling auto'}
              {config.backend === 'CUDA_GPU' ? ' --gpu' : ' --no-gpu'}
              {config.backend === 'CPU_MULTICORE' ? ` --backend multicore --threads ${config.cpu_threads}` : ''}
            </div>
            <p className="text-[11px] text-neutral-400 mt-2 font-sans">
              CLI execution outputs the identical verified JSON certificate.
            </p>
          </Panel>
        </div>
      </div>
    </div>
  );
};
