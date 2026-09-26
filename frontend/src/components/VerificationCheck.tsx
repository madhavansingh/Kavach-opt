import { CheckCircle2, XCircle } from 'lucide-react';
import type { VerificationCheckItem } from '../types/solver';

interface VerificationCheckProps {
  check: VerificationCheckItem;
  className?: string;
}

export const VerificationCheck: React.FC<VerificationCheckProps> = ({
  check,
  className = '',
}) => {
  return (
    <div
      className={`flex items-start justify-between gap-4 p-4 rounded-lg border ${
        check.passed
          ? 'bg-[#12181f]/80 border-emerald-900/40 text-neutral-200'
          : 'bg-[#1e1316]/80 border-rose-900/40 text-rose-200'
      } ${className}`}
    >
      <div className="flex items-start gap-3">
        <div className="mt-0.5">
          {check.passed ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          ) : (
            <XCircle className="w-4 h-4 text-rose-400" />
          )}
        </div>
        <div>
          <h4 className="text-xs font-semibold tracking-wide text-neutral-100">
            {check.name}
          </h4>
          <p className="mt-0.5 text-[11px] text-neutral-400 leading-relaxed">
            {check.description}
          </p>
        </div>
      </div>

      <div className="text-right font-mono shrink-0">
        <div className="text-xs font-medium">
          <span className="text-neutral-400">residual: </span>
          <span
            className={check.passed ? 'text-emerald-400' : 'text-rose-400 font-bold'}
          >
            {check.residual === 0
              ? '0.0000e+00'
              : check.residual.toExponential(4)}
          </span>
        </div>
        <div className="text-[10px] text-neutral-400 mt-0.5">
          tol: {check.tolerance.toExponential(0)}
        </div>
      </div>
    </div>
  );
};
