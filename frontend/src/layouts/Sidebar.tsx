import React from 'react';
import {
  Compass,
  Play,
  ShieldCheck,
  BarChart3,
  Factory,
  Cpu,
  Shield,
} from 'lucide-react';

export type NavTab = 'overview' | 'solve' | 'verification' | 'benchmarks' | 'mrpl' | 'system';

interface SidebarProps {
  currentTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  className?: string;
}

interface NavItem {
  id: NavTab;
  label: string;
  sublabel: string;
  icon: React.ReactNode;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onSelectTab,
  className = '',
}) => {
  const navItems: NavItem[] = [
    {
      id: 'overview',
      label: 'Overview',
      sublabel: 'Architecture & Status',
      icon: <Compass className="w-4 h-4" />,
    },
    {
      id: 'solve',
      label: 'Solve',
      sublabel: 'Model Run & Strategy',
      icon: <Play className="w-4 h-4" />,
    },
    {
      id: 'verification',
      label: 'Verification',
      sublabel: 'Independent Certification',
      icon: <ShieldCheck className="w-4 h-4" />,
    },
    {
      id: 'benchmarks',
      label: 'Benchmarks',
      sublabel: 'Netlib, MIPLIB & QP',
      icon: <BarChart3 className="w-4 h-4" />,
    },
    {
      id: 'mrpl',
      label: 'MRPL Case Study',
      sublabel: 'Refinery Blending MILP',
      icon: <Factory className="w-4 h-4" />,
    },
    {
      id: 'system',
      label: 'System',
      sublabel: 'Engine & Hardware',
      icon: <Cpu className="w-4 h-4" />,
    },
  ];

  return (
    <aside
      className={`w-64 flex flex-col justify-between bg-[#0e1117] border-r border-[#1e2430] shrink-0 select-none ${className}`}
    >
      <div>
        {/* Brand Header */}
        <div className="h-16 px-5 flex items-center gap-3 border-b border-[#1e2430]">
          <div className="w-8 h-8 rounded-md bg-emerald-950/80 border border-emerald-600/40 flex items-center justify-center text-emerald-400">
            <Shield className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold tracking-tight text-sm text-neutral-100 font-sans">
                KAVACH-OPT
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-neutral-800 text-neutral-300">
                RC-1
              </span>
            </div>
            <p className="text-[11px] text-neutral-400 font-mono">
              SIH26119 · MRPL
            </p>
          </div>
        </div>

        {/* Navigation Section */}
        <nav className="p-3 space-y-1">
          <div className="px-3 py-2 text-[10px] font-mono uppercase tracking-wider text-neutral-400 font-medium">
            Navigation
          </div>
          {navItems.map((item) => {
            const isActive = currentTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onSelectTab(item.id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-left transition-all ${
                  isActive
                    ? 'bg-[#181d26] text-emerald-400 border border-emerald-500/30 font-medium'
                    : 'text-neutral-400 hover:text-neutral-200 hover:bg-[#13161d] border border-transparent'
                }`}
              >
                <div
                  className={`${
                    isActive ? 'text-emerald-400' : 'text-neutral-400'
                  }`}
                >
                  {item.icon}
                </div>
                <div className="truncate">
                  <div className="text-xs tracking-wide leading-tight">
                    {item.label}
                  </div>
                  <div className="text-[10px] text-neutral-400 font-mono truncate">
                    {item.sublabel}
                  </div>
                </div>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Footer Info */}
      <div className="p-4 border-t border-[#1e2430] bg-[#0b0d12]">
        <div className="flex items-center justify-between text-[11px] text-neutral-400 font-mono">
          <span>Engine Status</span>
          <span className="flex items-center gap-1.5 text-emerald-400">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            STANDALONE
          </span>
        </div>
        <div className="mt-2 text-[10px] text-neutral-400 font-mono leading-tight">
          Sovereign C++20 Core
        </div>
      </div>
    </aside>
  );
};
