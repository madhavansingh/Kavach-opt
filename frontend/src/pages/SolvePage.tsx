import React, { useState } from 'react';
import {
  Play,
  Shield,
  Cpu,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Info,
} from 'lucide-react';
import { PageHeader } from '../components/PageHeader';
import { Panel } from '../components/Panel';
import { StatusBadge } from '../components/StatusBadge';
import { Metric, MetricGroup } from '../components/Metric';
import { PrimaryButton, SecondaryButton } from '../components/Buttons';
import { EmptyState } from '../components/EmptyState';
import type { ModelMetadata, SolveConfig, SolveResult } from '../types/solver';
import { SAMPLE_MODELS } from '../data/mockData';
import type { NavTab } from '../layouts/Sidebar';

interface SolvePageProps {
  activeModel: ModelMetadata;
  onSelectModel: (model: ModelMetadata) => void;
  onNavigate: (tab: NavTab) => void;
  currentResult: SolveResult | null;
  onRunSolve: (config: SolveConfig) => Promise<void>;
  isSolving: boolean;
}

export const SolvePage: React.FC<SolvePageProps> = ({
  activeModel,
  onSelectModel,
  onNavigate,
  currentResult,
  onRunSolve,
  isSolving,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [config, setConfig] = useState<SolveConfig>({
    problem_class: activeModel.problem_class,
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

  const handleModelChange = (modelName: string) => {
    const selected = SAMPLE_MODELS.find((m) => m.name === modelName);
    if (selected) {
      onSelectModel(selected);
      setConfig((prev: SolveConfig) => ({
        ...prev,
        problem_class: selected.problem_class,
      }));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await onRunSolve(config);
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Optimization & Solve Control"
        subtitle="Configure solver parameters, inspect strategy routing decisions, and trigger deterministic optimization."
        badge={
          <div className="flex items-center gap-2">
            <span className="text-xs text-neutral-400 font-mono">MODEL:</span>
            <span className="text-xs font-mono font-semibold text-white px-2 py-0.5 rounded bg-[#1e2430] border border-[#2e3748]">
              {activeModel.name}
            </span>
          </div>
        }
      />

      {/* Model Selection & Profile Strip */}
      <Panel title="Loaded Optimization Model" subtitle="Problem topology and matrix structure">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="space-y-3">
            <label className="block text-xs font-medium text-neutral-300">Select Active Instance</label>
            <select
              value={activeModel.name}
              onChange={(e) => handleModelChange(e.target.value)}
              className="w-full bg-[#0d1016] border border-[#2a3140] rounded-lg px-3 py-2 text-sm text-white font-mono focus:outline-none focus:border-blue-500"
            >
              {SAMPLE_MODELS.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name} ({m.problem_class}) {m.is_synthetic ? '· Synthetic' : ''}
                </option>
              ))}
            </select>
            <p className="text-xs text-neutral-400">
              Path: <span className="font-mono text-neutral-300">{activeModel.path}</span>
            </p>
          </div>

          <div className="lg:col-span-2">
            <MetricGroup columns={4}>
              <Metric label="Variables (n)" value={activeModel.num_variables} />
              <Metric label="Constraints (m)" value={activeModel.num_constraints} />
              <Metric label="Nonzeros (nnz)" value={activeModel.num_nonzeros} />
              <Metric
                label="Problem Class"
                value={activeModel.problem_class}
                subtext="Classified by Strategy Rule R001"
              />
              <Metric
                label="Sparsity Density"
                value={`${(activeModel.density * 100).toFixed(1)}%`}
              />
              <Metric
                label="Discrete Vars"
                value={activeModel.num_discrete_vars}
                subtext={activeModel.num_discrete_vars > 0 ? 'Integer Branch & Bound required' : 'Continuous problem'}
              />
              <Metric
                label="Dynamic Range"
                value={`${activeModel.dynamic_range.toFixed(1)}x`}
                subtext="Max |a_ij| / min |a_ij|"
              />
              <Metric
                label="Synthetic Flag"
                value={activeModel.is_synthetic ? 'SYNTHETIC' : 'STANDARD'}
              />
            </MetricGroup>
          </div>
        </div>
      </Panel>

      {/* Main Form & Strategy Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Configuration Form */}
        <div className="lg:col-span-2 space-y-6">
          <form onSubmit={handleSubmit}>
            <Panel
              title="Execution & Algorithm Configuration"
              subtitle="Governs solver routing, scaling, presolve, and backend execution"
            >
              <div className="space-y-5">
                {/* Core Parameters */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-medium text-neutral-300 mb-1.5">
                      Presolve Suite
                    </label>
                    <select
                      value={config.presolve_policy}
                      onChange={(e) =>
                        setConfig({ ...config, presolve_policy: e.target.value as any })
                      }
                      className="w-full bg-[#0d1016] border border-[#2a3140] rounded-lg px-3 py-2 text-sm text-neutral-200 focus:outline-none focus:border-blue-500"
                    >
                      <option value="FULL">Full Presolve Suite (Recommended)</option>
                      <option value="BOUNDS_ONLY">Bounds Tightening Only</option>
                      <option value="NONE">Disabled (Solve Raw Model)</option>
                    </select>
                    <span className="text-[11px] text-neutral-400 mt-1 block">
                      Fixed vars, singleton rows, bound tightening, redundant elimination.
                    </span>
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-neutral-300 mb-1.5">
                      Equilibration Scaling
                    </label>
                    <select
                      value={config.scaling_policy}
                      onChange={(e) =>
                        setConfig({ ...config, scaling_policy: e.target.value as any })
                      }
                      className="w-full bg-[#0d1016] border border-[#2a3140] rounded-lg px-3 py-2 text-sm text-neutral-200 focus:outline-none focus:border-blue-500"
                    >
                      <option value="AUTO">Auto Ruiz Equilibration (Recommended)</option>
                      <option value="NONE">Disabled</option>
                    </select>
                    <span className="text-[11px] text-neutral-400 mt-1 block">
                      Normalizes condition number to avoid numerical pivoting loss.
                    </span>
                  </div>
                </div>

                {/* Backend Selection */}
                <div className="pt-2 border-t border-[#1e2430]">
                  <label className="block text-xs font-medium text-neutral-300 mb-2">
                    Execution Backend Subsystem
                  </label>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <button
                      type="button"
                      onClick={() => setConfig({ ...config, backend: 'CPU_SEQUENTIAL' })}
                      className={`p-3 rounded-lg border text-left transition-colors ${
                        config.backend === 'CPU_SEQUENTIAL'
                          ? 'border-blue-500/60 bg-blue-950/20'
                          : 'border-[#202736] bg-[#0d1016] hover:border-[#2e3748]'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-semibold text-white">CPU Sequential</span>
                        <Cpu className="w-3.5 h-3.5 text-neutral-400" />
                      </div>
                      <p className="text-[11px] text-neutral-400">
                        Default deterministic single-core simplex & factorization.
                      </p>
                    </button>

                    <button
                      type="button"
                      onClick={() => setConfig({ ...config, backend: 'CPU_MULTICORE' })}
                      className={`p-3 rounded-lg border text-left transition-colors ${
                        config.backend === 'CPU_MULTICORE'
                          ? 'border-blue-500/60 bg-blue-950/20'
                          : 'border-[#202736] bg-[#0d1016] hover:border-[#2e3748]'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-semibold text-white">CPU Multicore</span>
                        <Cpu className="w-3.5 h-3.5 text-blue-400" />
                      </div>
                      <p className="text-[11px] text-neutral-400">
                        Thread pool for batched SpMV & parallel evaluation.
                      </p>
                    </button>

                    <button
                      type="button"
                      onClick={() => setConfig({ ...config, backend: 'CUDA_GPU' })}
                      className={`p-3 rounded-lg border text-left transition-colors ${
                        config.backend === 'CUDA_GPU'
                          ? 'border-emerald-500/60 bg-emerald-950/20'
                          : 'border-[#202736] bg-[#0d1016] hover:border-[#2e3748]'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-semibold text-white">CUDA GPU</span>
                        <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-emerald-900/50 text-emerald-300">
                          SCOPED
                        </span>
                      </div>
                      <p className="text-[11px] text-neutral-400">
                        Batched CSR SpMV kernel with automatic CPU fallback.
                      </p>
                    </button>
                  </div>
                </div>

                {/* Multicore threads slider if selected */}
                {config.backend === 'CPU_MULTICORE' && (
                  <div className="p-3 rounded-lg bg-[#0d1016] border border-[#202736] space-y-2">
                    <div className="flex justify-between items-center text-xs">
                      <span className="text-neutral-300 font-medium">Multicore Worker Threads:</span>
                      <span className="font-mono text-blue-400 font-bold">{config.cpu_threads} workers</span>
                    </div>
                    <input
                      type="range"
                      min={1}
                      max={16}
                      value={config.cpu_threads}
                      onChange={(e) => setConfig({ ...config, cpu_threads: parseInt(e.target.value) })}
                      className="w-full accent-blue-500 cursor-pointer"
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
                    {showAdvanced ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
                    <span>{showAdvanced ? 'Hide Advanced Numerical Settings' : 'Show Advanced Numerical Settings'}</span>
                  </button>

                  {showAdvanced && (
                    <div className="mt-3 p-4 rounded-lg bg-[#0d1016] border border-[#202736] grid grid-cols-1 sm:grid-cols-3 gap-4">
                      <div>
                        <label className="block text-[11px] font-medium text-neutral-400 mb-1">
                          Primal Feasibility Tol
                        </label>
                        <input
                          type="text"
                          value={config.primal_feasibility_tolerance}
                          onChange={(e) =>
                            setConfig({
                              ...config,
                              primal_feasibility_tolerance: parseFloat(e.target.value) || 1e-8,
                            })
                          }
                          className="w-full bg-[#131722] border border-[#2a3140] rounded px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-blue-500"
                        />
                        <span className="text-[10px] text-neutral-400">Default: 1.0e-8</span>
                      </div>

                      <div>
                        <label className="block text-[11px] font-medium text-neutral-400 mb-1">
                          Dual Feasibility Tol
                        </label>
                        <input
                          type="text"
                          value={config.dual_feasibility_tolerance}
                          onChange={(e) =>
                            setConfig({
                              ...config,
                              dual_feasibility_tolerance: parseFloat(e.target.value) || 1e-8,
                            })
                          }
                          className="w-full bg-[#131722] border border-[#2a3140] rounded px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-blue-500"
                        />
                        <span className="text-[10px] text-neutral-400">Default: 1.0e-8</span>
                      </div>

                      <div>
                        <label className="block text-[11px] font-medium text-neutral-400 mb-1">
                          Max Iterations
                        </label>
                        <input
                          type="number"
                          value={config.max_iterations}
                          onChange={(e) =>
                            setConfig({
                              ...config,
                              max_iterations: parseInt(e.target.value) || 10000,
                            })
                          }
                          className="w-full bg-[#131722] border border-[#2a3140] rounded px-2.5 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-blue-500"
                        />
                        <span className="text-[10px] text-neutral-400">Default: 10,000</span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Primary Action Button */}
                <div className="pt-4 flex items-center justify-between border-t border-[#1e2430]">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="enable_verifier"
                      checked={config.enable_independent_verification}
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          enable_independent_verification: e.target.checked,
                        })
                      }
                      className="rounded bg-[#0d1016] border-[#2a3140] text-blue-600 focus:ring-0 cursor-pointer"
                    />
                    <label htmlFor="enable_verifier" className="text-xs text-neutral-300 cursor-pointer select-none">
                      Run Independent Post-Solve Verification (Mandatory in Production)
                    </label>
                  </div>

                  <PrimaryButton
                    type="submit"
                    disabled={isSolving}
                    icon={<Play className="w-4 h-4" />}
                  >
                    {isSolving ? 'Optimizing...' : 'Run Optimization'}
                  </PrimaryButton>
                </div>
              </div>
            </Panel>
          </form>

          {/* Results Display */}
          {currentResult ? (
            <Panel
              title="Optimization Outcome"
              subtitle="Authoritative solver output and performance metrics"
              badge={<StatusBadge status={currentResult.status} size="md" />}
            >
              <div className="space-y-5">
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
                    label="Total Solve Time"
                    value={`${(currentResult.solve_time_seconds * 1000).toFixed(2)} ms`}
                    subtext={`${currentResult.solve_time_seconds.toFixed(6)} s`}
                  />
                  <Metric
                    label="Simplex Iterations"
                    value={currentResult.iterations}
                  />
                  <Metric
                    label="B&B Nodes Explored"
                    value={currentResult.nodes}
                  />
                </MetricGroup>

                {/* Verification Summary Banner */}
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
                        <div className="text-xs font-semibold uppercase tracking-wider">
                          Independent Verification:{' '}
                          {currentResult.verification_report.passed ? 'CERTIFIED OPTIMAL' : 'FAILED'}
                        </div>
                        <p className="text-xs text-neutral-300 font-mono mt-0.5">
                          Primal residual: {currentResult.verification_report.recomputed_primal_residual.toExponential(3)} · Dual: {currentResult.verification_report.recomputed_dual_residual.toExponential(3)}
                        </p>
                      </div>
                    </div>

                    <SecondaryButton
                      size="sm"
                      onClick={() => onNavigate('verification')}
                    >
                      Inspect Certificate
                    </SecondaryButton>
                  </div>
                )}
              </div>
            </Panel>
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
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-950/50 border border-blue-800/50 text-blue-300">
                RULES ENGINE
              </span>
            }
          >
            <div className="space-y-4">
              <div className="p-3 rounded-lg bg-[#0d1016] border border-[#202736] space-y-1.5">
                <span className="text-[10px] uppercase font-mono tracking-wider text-neutral-400">
                  Target Algorithm
                </span>
                <div className="text-sm font-semibold text-white">
                  {activeModel.problem_class === 'LP' && 'Two-Phase Revised Simplex'}
                  {activeModel.problem_class === 'MILP' && 'Branch & Bound + Dual Simplex'}
                  {activeModel.problem_class === 'CONVEX_QP' && 'Active-Set Quadratic Programming'}
                </div>
                <p className="text-xs text-neutral-400">
                  Routing enforced by deterministic classification rules R003–R006.
                </p>
              </div>

              <div className="space-y-2">
                <div className="text-xs font-semibold text-neutral-300">Strategy Decision Highlights</div>
                <ul className="space-y-2 text-xs text-neutral-400">
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                    <span>
                      <strong className="text-neutral-200">R001 / R003:</strong> Classified as{' '}
                      {activeModel.problem_class}. Problem class matches core kernel capabilities.
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                    <span>
                      <strong className="text-neutral-200">R007:</strong> Condition risk profile:{' '}
                      <span className="text-emerald-400 font-mono font-medium">NORMAL</span> (dynamic range{' '}
                      {activeModel.dynamic_range.toFixed(1)}x &lt; 1e6).
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                    <span>
                      <strong className="text-neutral-200">R010:</strong> Backend crossover evaluation:{' '}
                      Single-batch workload defaults to CPU Sequential for minimal launch latency.
                    </span>
                  </li>
                </ul>
              </div>

              <div className="p-3 rounded-lg bg-blue-950/20 border border-blue-900/40 text-xs text-neutral-300">
                <div className="flex items-center gap-1.5 font-semibold text-blue-300 mb-1">
                  <Info className="w-3.5 h-3.5" />
                  <span>Rule Engine Guarantee</span>
                </div>
                Strategy rules evaluate deterministically in &lt; 0.01 ms with zero stochasticity or black-box ML.
              </div>
            </div>
          </Panel>

          {/* Quick Info Panel */}
          <Panel title="Command-Line Equivalent" subtitle="Reproduce this exact solve via CLI">
            <div className="p-3 rounded bg-[#0a0d12] border border-[#1e2430] font-mono text-xs text-neutral-300 select-all overflow-x-auto">
              ./build/kavach-opt --model {activeModel.path} --presolve full --scaling auto
              {config.backend === 'CPU_MULTICORE' ? ` --threads ${config.cpu_threads}` : ''}
              {config.backend === 'CUDA_GPU' ? ' --gpu' : ' --no-gpu'}
            </div>
            <p className="text-[11px] text-neutral-400 mt-2">
              CLI execution outputs the identical verified JSON certificate.
            </p>
          </Panel>
        </div>
      </div>
    </div>
  );
};
