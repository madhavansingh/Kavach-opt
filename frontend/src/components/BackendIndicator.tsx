import React from 'react';
import { Cpu, Zap } from 'lucide-react';
import type { BackendType } from '../types/solver';

interface BackendIndicatorProps {
  backend: BackendType;
  isSimulated?: boolean;
  className?: string;
}

export const BackendIndicator: React.FC<BackendIndicatorProps> = ({
  backend,
  isSimulated = false,
  className = '',
}) => {
  if (backend === 'CUDA_GPU') {
    return (
      <div
        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-indigo-950/40 border border-indigo-800/50 text-indigo-300 text-xs font-mono font-medium ${className}`}
        title="CUDA Acceleration Active for Batched SpMV"
      >
        <Zap className="w-3.5 h-3.5 text-indigo-400" />
        <span>CUDA_GPU</span>
        {isSimulated && (
          <span className="text-[10px] text-indigo-400/80 uppercase">
            (Scoped)
          </span>
        )}
      </div>
    );
  }

  if (backend === 'CPU_MULTICORE') {
    return (
      <div
        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-emerald-950/40 border border-emerald-800/50 text-emerald-300 text-xs font-mono font-medium ${className}`}
        title="CPU Multicore Parallelism Active"
      >
        <Cpu className="w-3.5 h-3.5 text-emerald-400" />
        <span>CPU_MULTICORE</span>
      </div>
    );
  }

  return (
    <div
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-neutral-900 border border-neutral-800 text-neutral-300 text-xs font-mono font-medium ${className}`}
      title="Sequential CPU Thread"
    >
      <Cpu className="w-3.5 h-3.5 text-neutral-400" />
      <span>CPU_SEQUENTIAL</span>
    </div>
  );
};
