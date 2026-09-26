import React from 'react';
import { Menu, Terminal, Layers, Loader2 } from 'lucide-react';
import { BackendIndicator } from '../components/BackendIndicator';
import type { BackendType, SolverStatus } from '../types/solver';
import { SYSTEM_METADATA } from '../lib/constants';

interface TopBarProps {
  currentModelName?: string;
  isSolving?: boolean;
  hasCompletedRun?: boolean;
  backend?: BackendType;
  solverStatus?: SolverStatus;
  onToggleMobileMenu?: () => void;
  onOpenOnboarding?: () => void;
  className?: string;
}

export const TopBar: React.FC<TopBarProps> = ({
  currentModelName = 'No model selected',
  isSolving = false,
  hasCompletedRun = false,
  backend = 'CPU_SEQUENTIAL',
  solverStatus = 'UNKNOWN',
  onToggleMobileMenu,
  onOpenOnboarding,
  className = '',
}) => {
  return (
    <header
      className={`h-16 px-4 sm:px-6 bg-[#0c0f17]/95 backdrop-blur-md border-b border-[#1d2332] flex items-center justify-between gap-3 shrink-0 select-none ${className}`}
    >
      <div className="flex items-center gap-3 sm:gap-4 min-w-0">
        {onToggleMobileMenu && (
          <button
            onClick={onToggleMobileMenu}
            className="lg:hidden p-1.5 rounded text-neutral-400 hover:text-neutral-200 hover:bg-[#161b28] transition-colors shrink-0"
            aria-label="Toggle Navigation"
          >
            <Menu className="w-5 h-5" />
          </button>
        )}

        <div className="flex items-center gap-2 sm:gap-2.5 min-w-0">
          <Layers className="w-4 h-4 text-neutral-400 shrink-0" />
          <div className="flex items-center gap-2 min-w-0">
            <span className="text-xs text-neutral-400 font-sans shrink-0">
              {isSolving ? 'Solving:' : hasCompletedRun ? 'Last Run:' : 'Selected Model:'}
            </span>
            <span
              className="text-xs font-mono font-medium text-neutral-100 bg-[#141926] px-2.5 py-0.5 rounded border border-[#232c3f] max-w-[130px] sm:max-w-[200px] md:max-w-[280px] truncate"
              title={currentModelName}
            >
              {currentModelName}
            </span>
            {isSolving ? (
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-blue-950/60 border border-blue-700/50 text-blue-300 font-medium flex items-center gap-1 shrink-0 animate-pulse">
                <Loader2 className="w-2.5 h-2.5 animate-spin" />
                OPTIMIZING
              </span>
            ) : hasCompletedRun && solverStatus !== 'UNKNOWN' ? (
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-700/50 text-emerald-400 font-medium shrink-0">
                {solverStatus}
              </span>
            ) : null}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 sm:gap-3 shrink-0">
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
          <span>{SYSTEM_METADATA.PRODUCT_VERSION}</span>
        </div>

        <div className="hidden xs:flex items-center gap-1.5 pl-2 border-l border-[#1d2332]">
          <span className="text-[10px] font-mono text-neutral-400">
            AUDIT:
          </span>
          <span className="text-[11px] font-mono font-semibold text-emerald-400 bg-emerald-950/40 border border-emerald-800/50 px-2 py-0.5 rounded">
            {SYSTEM_METADATA.ENGINEERING_AUDIT_STATUS}
          </span>
        </div>
      </div>
    </header>
  );
};
