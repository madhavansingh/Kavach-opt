import React from 'react';
import type { SolverStatus, VerificationStatus } from '../types/solver';

type BadgeType = SolverStatus | VerificationStatus | string;

interface StatusBadgeProps {
  status: BadgeType;
  size?: 'sm' | 'md';
  className?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  size = 'md',
  className = '',
}) => {
  const norm = status.toUpperCase();

  let colorClasses = 'text-slate-400 bg-slate-900/60 border-slate-800';
  let dotColor = 'bg-slate-500';

  if (norm === 'OPTIMAL' || norm === 'CERTIFIED' || norm === 'VERIFIED') {
    colorClasses = 'text-emerald-400 bg-emerald-950/40 border-emerald-800/60';
    dotColor = 'bg-emerald-400';
  } else if (norm === 'FEASIBLE' || norm === 'RUNNING' || norm === 'SOLVING') {
    colorClasses = 'text-sky-400 bg-sky-950/40 border-sky-800/60';
    dotColor = 'bg-sky-400';
  } else if (norm === 'INFEASIBLE' || norm === 'REJECTED' || norm === 'FAILED' || norm === 'ERROR') {
    colorClasses = 'text-rose-400 bg-rose-950/40 border-rose-800/60';
    dotColor = 'bg-rose-400';
  } else if (norm === 'UNBOUNDED' || norm === 'LIMIT_REACHED' || norm === 'TIME_LIMIT' || norm === 'WARNING') {
    colorClasses = 'text-amber-400 bg-amber-950/40 border-amber-800/60';
    dotColor = 'bg-amber-400';
  } else if (norm === 'WAITING' || norm === 'IDLE' || norm === 'UNKNOWN') {
    colorClasses = 'text-neutral-400 bg-neutral-900/60 border-neutral-800';
    dotColor = 'bg-neutral-500';
  }

  const sizeClasses =
    size === 'sm'
      ? 'px-2 py-0.5 text-[11px] tracking-wider'
      : 'px-2.5 py-1 text-xs tracking-wider';

  return (
    <span
      className={`inline-flex items-center gap-1.5 font-mono font-medium rounded-md border ${colorClasses} ${sizeClasses} ${className}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${dotColor}`} />
      {norm}
    </span>
  );
};
