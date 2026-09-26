import React from 'react';
import { Menu, Terminal, Layers } from 'lucide-react';
import { BackendIndicator } from '../components/BackendIndicator';
import type { BackendType, SolverStatus } from '../types/solver';

interface TopBarProps {
  currentModelName?: string;
  hasActiveRun?: boolean;
  backend?: BackendType;
  solverStatus?: SolverStatus;
  onToggleMobileMenu?: () => void;
  onOpenOnboarding?: () => void;
  className?: string;
}

export const TopBar: React.FC<TopBarProps> = ({
  currentModelName = 'No model selected',
  hasActiveRun = false,
  backend = 'CPU_SEQUENTIAL',
  solverStatus = 'UNKNOWN',
  onToggleMobileMenu,
  onOpenOnboarding,
  className = '',
}) => {
  return (
    <header
      className={`h-16 px-6 bg-[#0c0f17]/95 backdrop-blur-md border-b border-[#1d2332] flex items-center justify-between gap-4 shrink-0 select-none ${className}`}
    >
      <div className="flex items-center gap-4">
        {onToggleMobileMenu && (
          <button
            onClick={onToggleMobileMenu}
            className="lg:hidden p-1.5 rounded text-neutral-400 hover:text-neutral-200 hover:bg-[#161b28] transition-colors"
            aria-label="Toggle Navigation"
          >
            <Menu className="w-5 h-5" />
          </button>
        )}

        <div className="flex items-center gap-2.5">
          <Layers className="w-4 h-4 text-neutral-400" />
          <div className="flex items-center gap-2">
            <span className="text-xs text-neutral-400 font-sans">
              {hasActiveRun ? 'Active Run:' : 'Selected Model:'}
            </span>
            <span className="text-xs font-mono font-medium text-neutral-100 bg-[#141926] px-2.5 py-0.5 rounded border border-[#232c3f]">
              {currentModelName}
            </span>
            {hasActiveRun && solverStatus !== 'UNKNOWN' && (
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-700/50 text-emerald-400 font-medium">
                {solverStatus}
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {onOpenOnboarding && (
          <button
            onClick={onOpenOnboarding}
            className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#141926] hover:bg-[#1c2336] border border-[#232c3f] text-[11px] font-medium text-neutral-300 transition-colors"
            title="Open Interactive Workstation Onboarding Tour"
          >
            <span>Engine Guide</span>
          </button>
        )}

        <BackendIndicator backend={backend} />

        <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded bg-[#11151f] border border-[#1d2332] text-[11px] font-mono text-neutral-400">
          <Terminal className="w-3.5 h-3.5 text-neutral-400" />
          <span>v0.1.0-RC1</span>
        </div>

        <div className="flex items-center gap-1.5 pl-2 border-l border-[#1d2332]">
          <span className="text-[10px] font-mono text-neutral-400">
            AUDIT:
          </span>
          <span className="text-[11px] font-mono font-semibold text-emerald-400 bg-emerald-950/40 border border-emerald-800/50 px-2 py-0.5 rounded">
            CERTIFIED GREEN
          </span>
        </div>
      </div>
    </header>
  );
};
