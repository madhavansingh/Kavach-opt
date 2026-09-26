import React from 'react';

interface MetricProps {
  label: string;
  value: string | number;
  subtext?: string;
  isMono?: boolean;
  highlight?: 'normal' | 'success' | 'warning' | 'danger' | 'info';
  className?: string;
}

export const Metric: React.FC<MetricProps> = ({
  label,
  value,
  subtext,
  isMono = true,
  highlight = 'normal',
  className = '',
}) => {
  let valColor = 'text-neutral-100';
  if (highlight === 'success') valColor = 'text-emerald-400';
  if (highlight === 'warning') valColor = 'text-amber-400';
  if (highlight === 'danger') valColor = 'text-rose-400';
  if (highlight === 'info') valColor = 'text-sky-400';

  return (
    <div className={`flex flex-col gap-1 p-4 rounded-lg bg-[#11141a] border border-[#1e2430] ${className}`}>
      <span className="text-xs font-medium uppercase tracking-wider text-neutral-400">
        {label}
      </span>
      <span className={`text-2xl font-semibold tracking-tight ${valColor} ${isMono ? 'font-mono' : 'font-sans'}`}>
        {value}
      </span>
      {subtext && (
        <span className="text-xs text-neutral-400 font-mono">
          {subtext}
        </span>
      )}
    </div>
  );
};

export const MetricGroup: React.FC<{ children: React.ReactNode; columns?: number; className?: string }> = ({
  children,
  columns = 4,
  className = '',
}) => {
  const gridCols =
    columns === 2
      ? 'grid-cols-1 md:grid-cols-2'
      : columns === 3
      ? 'grid-cols-1 md:grid-cols-3'
      : columns === 4
      ? 'grid-cols-2 lg:grid-cols-4'
      : 'grid-cols-2 md:grid-cols-3 lg:grid-cols-6';

  return (
    <div className={`grid gap-3 ${gridCols} ${className}`}>
      {children}
    </div>
  );
};
