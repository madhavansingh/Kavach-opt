import React from 'react';
import { CheckCircle2, ChevronRight } from 'lucide-react';

export interface PipelineStep {
  id: string;
  name: string;
  category: string;
  status: 'completed' | 'active' | 'pending';
  detail?: string;
}

interface PipelineProps {
  currentStep?: string;
  steps?: PipelineStep[];
  className?: string;
}

export const Pipeline: React.FC<PipelineProps> = ({
  steps = [
    { id: 'ingest', name: '1. Ingestion', category: 'MPS / LP Parse', status: 'completed', detail: 'Tokenization & Bounds' },
    { id: 'model', name: '2. Canonical Model', category: 'Representation', status: 'completed', detail: 'Variables & Constraints' },
    { id: 'strategy', name: '3. Strategy Layer', category: 'Rules R001-R018', status: 'completed', detail: 'Characterization' },
    { id: 'presolve', name: '4. Presolve Suite', category: 'Reduction', status: 'completed', detail: 'Fixed Vars & Singletons' },
    { id: 'solve', name: '5. Core Solver', category: 'Simplex / B&B / QP', status: 'active', detail: 'Sparse LU Factorization' },
    { id: 'verify', name: '6. Verification', category: 'Independent Cert', status: 'pending', detail: 'Primal & Dual Residuals' },
  ],
  className = '',
}) => {
  return (
    <div className={`w-full overflow-x-auto ${className}`}>
      <div className="flex flex-col lg:flex-row items-stretch gap-2 min-w-[700px]">
        {steps.map((step, idx) => {
          const isCompleted = step.status === 'completed';
          const isActive = step.status === 'active';

          let borderClass = 'border-[#1e2430] bg-[#11141a] text-neutral-400';
          if (isActive) {
            borderClass = 'border-emerald-500/60 bg-emerald-950/20 text-emerald-200 shadow-sm';
          } else if (isCompleted) {
            borderClass = 'border-[#273142] bg-[#131720] text-neutral-200';
          }

          return (
            <React.Fragment key={step.id}>
              <div
                className={`flex-1 flex flex-col p-3 rounded-lg border transition-all ${borderClass}`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[11px] font-mono tracking-wider uppercase text-neutral-400">
                    {step.category}
                  </span>
                  {isCompleted && (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  )}
                  {isActive && (
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  )}
                </div>
                <span className="text-xs font-semibold tracking-tight text-neutral-100">
                  {step.name}
                </span>
                {step.detail && (
                  <span className="text-[11px] text-neutral-400 font-mono mt-1">
                    {step.detail}
                  </span>
                )}
              </div>
              {idx < steps.length - 1 && (
                <div className="hidden lg:flex items-center justify-center text-neutral-400">
                  <ChevronRight className="w-4 h-4" />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};
