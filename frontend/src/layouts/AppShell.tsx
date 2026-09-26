import React, { useState } from 'react';
import { Sidebar } from './Sidebar';
import type { NavTab } from './Sidebar';
import { TopBar } from './TopBar';
import type { BackendType, SolverStatus } from '../types/solver';

interface AppShellProps {
  currentTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  currentModelName?: string;
  hasActiveRun?: boolean;
  backend?: BackendType;
  solverStatus?: SolverStatus;
  onOpenOnboarding?: () => void;
  children: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({
  currentTab,
  onSelectTab,
  currentModelName,
  hasActiveRun = false,
  backend,
  solverStatus,
  onOpenOnboarding,
  children,
}) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleTabChange = (tab: NavTab) => {
    onSelectTab(tab);
    setMobileMenuOpen(false);
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0a0d12] text-neutral-200">
      {/* Desktop Sidebar */}
      <Sidebar
        currentTab={currentTab}
        onSelectTab={handleTabChange}
        className="hidden lg:flex"
      />

      {/* Mobile Sidebar Overlay */}
      {mobileMenuOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm lg:hidden"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      {/* Mobile Drawer */}
      <div
        className={`fixed inset-y-0 left-0 z-50 transform transition-transform duration-200 lg:hidden ${
          mobileMenuOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <Sidebar currentTab={currentTab} onSelectTab={handleTabChange} />
      </div>

      {/* Main Content Area */}
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <TopBar
          currentModelName={currentModelName}
          hasActiveRun={hasActiveRun}
          backend={backend}
          solverStatus={solverStatus}
          onOpenOnboarding={onOpenOnboarding}
          onToggleMobileMenu={() => setMobileMenuOpen((o) => !o)}
        />

        <main id="main-content" className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8">
          <div key={currentTab} className="max-w-7xl mx-auto animate-fade-in">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
};
