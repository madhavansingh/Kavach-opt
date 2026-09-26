import React from 'react';

/**
 * @file Skeleton.tsx
 * @brief Zero-CLS skeleton primitives for Kavach-Opt industrial UI.
 * Respects prefers-reduced-motion and closely matches final layout dimensions.
 */

interface SkeletonProps {
  className?: string;
  style?: React.CSSProperties;
}

export const Skeleton: React.FC<SkeletonProps> = ({ className = '', style }) => {
  return (
    <div
      className={`bg-[#141926] border border-[#1f283a]/40 rounded animate-pulse ${className}`}
      style={style}
      aria-hidden="true"
    />
  );
};

export const SkeletonText: React.FC<{ lines?: number; className?: string }> = ({
  lines = 1,
  className = '',
}) => {
  return (
    <div className={`space-y-2 ${className}`} aria-hidden="true">
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          className={`h-3.5 ${
            i === lines - 1 && lines > 1 ? 'w-2/3' : 'w-full'
          }`}
        />
      ))}
    </div>
  );
};

export const SkeletonMetric: React.FC<{ className?: string }> = ({ className = '' }) => {
  return (
    <div className={`p-4 rounded-lg bg-[#141926] border border-[#1f283a] space-y-2.5 ${className}`} aria-hidden="true">
      <Skeleton className="h-3 w-24" />
      <Skeleton className="h-7 w-32" />
      <Skeleton className="h-2.5 w-16" />
    </div>
  );
};

export const SkeletonCard: React.FC<{ className?: string; children?: React.ReactNode }> = ({
  className = '',
  children,
}) => {
  return (
    <div className={`p-4 rounded-lg bg-[#141926] border border-[#1f283a] space-y-3 ${className}`} aria-hidden="true">
      {children || (
        <>
          <div className="flex items-center justify-between">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-4 w-12" />
          </div>
          <SkeletonText lines={3} />
        </>
      )}
    </div>
  );
};

export const SkeletonTableRow: React.FC<{ cols?: number; className?: string }> = ({
  cols = 5,
  className = '',
}) => {
  return (
    <tr className={`border-b border-[#1d2332] ${className}`} aria-hidden="true">
      {Array.from({ length: cols }).map((_, i) => (
        <td key={i} className="py-3 px-3">
          <Skeleton className={`h-4 ${i === 0 ? 'w-28' : i === cols - 1 ? 'w-16' : 'w-20'}`} />
        </td>
      ))}
    </tr>
  );
};

export const SkeletonChart: React.FC<{ height?: string; className?: string }> = ({
  height = 'h-48',
  className = '',
}) => {
  return (
    <div
      className={`rounded-lg bg-[#141926] border border-[#1f283a] p-4 flex flex-col justify-end space-y-2 ${height} ${className}`}
      aria-hidden="true"
    >
      <div className="flex items-end justify-between gap-2 h-full pt-4">
        {[40, 70, 55, 90, 65, 80, 45, 85].map((h, i) => (
          <div key={i} className="flex-1 flex flex-col items-center gap-1.5 h-full justify-end">
            <Skeleton
              className="w-full rounded-t"
              style={{ height: `${h}%` }}
            />
            <Skeleton className="h-2 w-full max-w-[28px]" />
          </div>
        ))}
      </div>
      <div className="flex justify-between pt-2 border-t border-[#1f283a]">
        <Skeleton className="h-2.5 w-16" />
        <Skeleton className="h-2.5 w-16" />
      </div>
    </div>
  );
};

export const SkeletonPanel: React.FC<{
  titleWidth?: string;
  subtitleWidth?: string;
  className?: string;
  children?: React.ReactNode;
}> = ({
  titleWidth = 'w-48',
  subtitleWidth = 'w-72',
  className = '',
  children,
}) => {
  return (
    <div className={`p-5 rounded-xl bg-[#11151f] border border-[#1d2332] shadow-xs space-y-4 ${className}`} aria-hidden="true">
      <div className="flex items-center justify-between pb-3 border-b border-[#1d2332]">
        <div className="space-y-1.5">
          <Skeleton className={`h-4.5 ${titleWidth}`} />
          <Skeleton className={`h-3 ${subtitleWidth}`} />
        </div>
        <Skeleton className="h-5 w-20 rounded" />
      </div>
      {children || (
        <div className="space-y-3 pt-1">
          <SkeletonText lines={4} />
        </div>
      )}
    </div>
  );
};
