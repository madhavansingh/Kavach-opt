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
import { SYSTEM_METADATA } from '../lib/constants';

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
      label: 'Refinery Optimization',
      sublabel: 'Crude Blending & Capacity',
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
      className={`w-64 flex flex-col justify-between bg-[#0c0f17] border-r border-[#1d2332] shrink-0 select-none ${className}`}
    >
      <div>
        {/* Brand Header */}
        <div className="h-16 px-5 flex items-center gap-3 border-b border-[#1d2332]">
          <div className="w-8 h-8 rounded-md bg-emerald-950/60 border border-emerald-600/30 flex items-center justify-center text-emerald-400">
            <Shield className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold tracking-tight text-sm text-neutral-100 font-sans">
                KAVACH-OPT
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-950/60 text-emerald-400 border border-emerald-800/50 font-medium">
                {SYSTEM_METADATA.PRODUCT_VERSION}
              </span>
            </div>
            <p className="text-[11px] text-neutral-400 font-mono">
              {SYSTEM_METADATA.PRODUCT_TAGLINE}
            </p>
          </div>
        </div>

        {/* Navigation Section */}
        <nav className="p-3 space-y-1">
          <div className="px-3 py-2 text-[10px] font-mono text-neutral-400 font-medium">
            WORKSTATION VIEWS
          </div>
          {navItems.map((item) => {
            const isActive = currentTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onSelectTab(item.id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-left transition-all ${
                  isActive
                    ? 'bg-[#161c28] text-neutral-100 border border-[#273245] shadow-xs'
                    : 'text-neutral-400 hover:text-neutral-200 hover:bg-[#121622] border border-transparent'
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
                  <div className={`text-xs tracking-normal leading-tight ${isActive ? 'font-semibold text-white' : 'font-medium'}`}>
                    {item.label}
                  </div>
                  <div className="text-[10px] text-neutral-400 font-mono truncate mt-0.5">
                    {item.sublabel}
                  </div>
                </div>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Footer Info */}
      <div className="p-4 border-t border-[#1d2332] bg-[#090c12]">
        <div className="flex items-center justify-between text-[11px] text-neutral-400 font-mono">
          <span>Engine Core</span>
          <span className="flex items-center gap-1.5 text-emerald-400 font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            STANDALONE
          </span>
        </div>
        <div className="mt-1 text-[10px] text-neutral-400 font-mono leading-tight">
          Sovereign C++20 · Independent Core
        </div>
      </div>
    </aside>
  );
};
