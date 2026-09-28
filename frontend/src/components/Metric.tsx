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

  const valStr = String(value);
  const textSize =
    valStr.length >= 13
      ? 'text-base sm:text-lg lg:text-base xl:text-lg 2xl:text-xl'
      : valStr.length >= 11
      ? 'text-lg sm:text-xl xl:text-2xl'
      : 'text-xl sm:text-2xl';

  return (
    <div className={`flex flex-col gap-1 p-3.5 rounded-lg bg-[#141926] border border-[#1f283a] shadow-xs min-w-0 ${className}`}>
      <span className="text-[11px] font-sans font-medium text-neutral-400 truncate" title={label}>
        {label}
      </span>
      <span
        className={`font-semibold tracking-tight truncate ${textSize} ${valColor} ${isMono ? 'font-mono' : 'font-sans'}`}
        title={valStr}
      >
        {value}
      </span>
      {subtext && (
        <span className="text-[11px] text-neutral-400 font-mono mt-0.5 truncate" title={subtext}>
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
    columns === 1
      ? 'grid-cols-1'
      : columns === 2
      ? 'grid-cols-1 md:grid-cols-2'
      : columns === 3
      ? 'grid-cols-1 md:grid-cols-3'
      : columns === 4
      ? 'grid-cols-2 lg:grid-cols-4'
      : columns === 5
      ? 'grid-cols-2 sm:grid-cols-3 lg:grid-cols-5'
      : 'grid-cols-2 md:grid-cols-3 lg:grid-cols-6';

  return (
    <div className={`grid gap-3 ${gridCols} ${className}`}>
      {children}
    </div>
  );
};
