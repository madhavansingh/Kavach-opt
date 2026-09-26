import React, { useEffect, useState } from 'react';
import { Cpu, Terminal, Loader2 } from 'lucide-react';
import type { ModelMetadata, SolveConfig } from '../types/solver';

export type SolverLifecycleState = 'IDLE' | 'SUBMITTING' | 'RUNNING' | 'COMPLETED' | 'FAILED';

interface SolverExecutionPanelProps {
  model: ModelMetadata;
  config: SolveConfig;
  state: SolverLifecycleState;
  startTime?: number | null;
  error?: string | null;
}

export const SolverExecutionPanel: React.FC<SolverExecutionPanelProps> = ({
  model,
  config,
  state,
  startTime,
  error,
}) => {
  const [elapsedMs, setElapsedMs] = useState<number>(0);

  useEffect(() => {
    if (state !== 'RUNNING' && state !== 'SUBMITTING') {
      return;
    }

    const start = startTime || performance.now();
    const interval = setInterval(() => {
      setElapsedMs(Math.max(0, performance.now() - start));
    }, 16); // ~60fps timer update

    return () => clearInterval(interval);
  }, [state, startTime]);

  return (
    <div
      className="p-5 rounded-xl bg-[#11151f] border border-blue-500/40 shadow-lg space-y-4 animate-fade-in"
      role="status"
      aria-busy={state === 'RUNNING' || state === 'SUBMITTING'}
    >
      {/* Header with state indicator */}
      <div className="flex items-center justify-between pb-3 border-b border-[#1d2332]">
        <div className="flex items-center gap-2.5">
          <div className="w-2.5 h-2.5 rounded-full bg-blue-500 animate-ping" />
          <span className="text-xs font-semibold text-white tracking-wide font-sans uppercase">
            Active Solver Execution
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-mono text-neutral-400">STATE:</span>
          <span className="text-xs font-mono font-bold text-blue-400 px-2 py-0.5 rounded bg-blue-950/60 border border-blue-800/60 flex items-center gap-1.5">
            <Loader2 className="w-3 h-3 animate-spin text-blue-400" />
            {state === 'SUBMITTING' ? 'STARTING KERNEL' : 'SOLVING'}
          </span>
        </div>
      </div>

      {/* Target Instance Telemetry Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
        <div className="p-3 rounded-lg bg-[#141926] border border-[#1f283a] space-y-1">
          <div className="text-[10px] text-neutral-400 font-sans">TARGET MODEL</div>
          <div className="font-mono text-white font-semibold truncate" title={model.name}>
            {model.name}
          </div>
          <div className="text-[10px] text-blue-400 font-mono">
            {model.problem_class} ({model.num_constraints}×{model.num_variables})
          </div>
        </div>

        <div className="p-3 rounded-lg bg-[#141926] border border-[#1f283a] space-y-1">
          <div className="text-[10px] text-neutral-400 font-sans">BACKEND ENGINE</div>
          <div className="font-mono text-white font-semibold flex items-center gap-1">
            <Cpu className="w-3.5 h-3.5 text-neutral-400" />
            <span>{config.backend}</span>
          </div>
          <div className="text-[10px] text-neutral-400 font-sans">
            {config.backend === 'CPU_MULTICORE'
              ? `${config.cpu_threads} Worker Threads`
              : config.backend === 'CUDA_GPU'
              ? 'GPU Kernel (w/ R012 CPU fallback)'
              : 'Sequential Simplex'}
          </div>
        </div>

        <div className="p-3 rounded-lg bg-[#141926] border border-[#1f283a] space-y-1">
          <div className="text-[10px] text-neutral-400 font-sans">PRESOLVE & SCALING</div>
          <div className="font-mono text-white font-semibold truncate">
            {config.presolve_policy} Presolve
          </div>
          <div className="text-[10px] text-neutral-400 font-sans">
            {config.scaling_policy === 'AUTO' ? 'Ruiz Equilibration' : 'Raw Matrix'}
          </div>
        </div>

        <div className="p-3 rounded-lg bg-[#141926] border border-[#1f283a] space-y-1">
          <div className="text-[10px] text-neutral-400 font-sans">MEASURED ELAPSED</div>
          <div className="font-mono text-blue-400 font-bold text-sm">
            {elapsedMs < 1000
              ? `${elapsedMs.toFixed(1)} ms`
              : `${(elapsedMs / 1000).toFixed(2)} s`}
          </div>
          <div className="text-[10px] text-neutral-400 font-sans">Real-time client timer</div>
        </div>
      </div>

      {/* Activity Bar & Status Phase */}
      <div className="p-3.5 rounded-lg bg-[#0c0f17] border border-[#1d2332] space-y-2">
        <div className="flex items-center justify-between text-xs font-sans">
          <span className="text-neutral-300 font-medium flex items-center gap-1.5">
            <Terminal className="w-3.5 h-3.5 text-blue-400" />
            <span>
              {state === 'SUBMITTING'
                ? 'Validating parameters & preparing C++ solve context...'
                : config.enable_independent_verification
                ? 'Solving problem & preparing independent verification pipeline...'
                : 'Solving problem via native optimization algorithm...'}
            </span>
          </span>
          <span className="text-neutral-400 font-mono text-[11px]">
            {config.enable_independent_verification ? 'Air-Gapped Audit Enforced' : 'Fast Solve Mode'}
          </span>
        </div>

        {/* Indeterminate Activity Progress Bar */}
        <div className="h-1.5 w-full bg-[#161c28] rounded-full overflow-hidden relative">
          <div className="h-full bg-blue-500 rounded-full w-1/3 animate-indeterminate" />
        </div>
      </div>

      {error && (
        <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-xs text-rose-300 font-mono">
          {error}
        </div>
      )}
    </div>
  );
};
