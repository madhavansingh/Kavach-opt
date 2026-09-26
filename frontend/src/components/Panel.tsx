import React from 'react';

interface PanelProps {
  title?: string;
  subtitle?: string;
  badge?: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  headerBorder?: boolean;
}

export const Panel: React.FC<PanelProps> = ({
  title,
  subtitle,
  badge,
  action,
  children,
  className = '',
  headerBorder = true,
}) => {
  return (
    <div className={`bg-[#11151f] rounded-lg border border-[#1d2332] overflow-hidden shadow-xs ${className}`}>
      {(title || action || badge) && (
        <div
          className={`px-5 py-3.5 flex items-center justify-between gap-4 ${
            headerBorder ? 'border-b border-[#1d2332]' : ''
          }`}
        >
          <div className="flex items-center gap-2.5">
            {title && (
              <h3 className="text-sm font-semibold tracking-normal text-neutral-100 font-sans">
                {title}
              </h3>
            )}
            {badge}
            {subtitle && (
              <span className="text-xs text-neutral-400 font-normal">
                {subtitle}
              </span>
            )}
          </div>
          {action && <div className="flex items-center gap-2">{action}</div>}
        </div>
      )}
      <div className="p-5">{children}</div>
    </div>
  );
};
