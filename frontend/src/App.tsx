import { useState, useEffect, useRef } from 'react';
import { AppShell } from './layouts/AppShell';
import type { NavTab } from './layouts/Sidebar';
import { OverviewPage } from './pages/OverviewPage';
import { SolvePage } from './pages/SolvePage';
import { VerificationPage } from './pages/VerificationPage';
import { BenchmarksPage } from './pages/BenchmarksPage';
import { MRPLPage } from './pages/MRPLPage';
import { SystemPage } from './pages/SystemPage';
import type { ModelMetadata, SolveConfig, SolveResult } from './types/solver';
import { solverBridge } from './lib/solverBridge';
import { ToastProvider } from './context/ToastContext';
import { useToast } from './context/useToast';
import { OnboardingModal } from './components/OnboardingModal';

function AppInner() {
  const [currentTab, setCurrentTab] = useState<NavTab>('overview');
  const [availableModels, setAvailableModels] = useState<ModelMetadata[]>([]);
  const [activeModel, setActiveModel] = useState<ModelMetadata | null>(null);
  const [currentResult, setCurrentResult] = useState<SolveResult | null>(null);
  const [isSolving, setIsSolving] = useState(false);
  const [isVerifying, setIsVerifying] = useState(false);
  const [solveError, setSolveError] = useState<string | null>(null);
  const [isDiscoveringModels, setIsDiscoveringModels] = useState(true);
  const [discoveryError, setDiscoveryError] = useState<string | null>(null);
  const [discoveryAttempt, setDiscoveryAttempt] = useState<number>(0);
  const [showOnboarding, setShowOnboarding] = useState<boolean>(() => {
    try {
      return localStorage.getItem('kavach_onboarding_completed') !== 'true';
    } catch {
      return false;
    }
  });

  const { showToast, dismissByLifecycleKey } = useToast();

  // Mathematical race condition & stale data protection token
  const solveRequestIdRef = useRef<number>(0);

  // Discover all authentic models from repository
  useEffect(() => {
    let isMounted = true;

    solverBridge
      .getAvailableModels()
      .then((models) => {
        if (!isMounted) return;
        if (models && models.length > 0) {
          setAvailableModels(models);
          setActiveModel((prev) => prev || models.find((m) => m.name === 'afiro.mps') || models[0]);
        } else {
          setDiscoveryError('No optimization models found in repository');
        }
      })
      .catch((err) => {
        if (!isMounted) return;
        const msg = err?.message || 'Failed to connect to solver adapter';
        setDiscoveryError(msg);
        showToast('error', 'Discovery Failed', msg);
      })
      .finally(() => {
        if (isMounted) setIsDiscoveringModels(false);
      });

    return () => {
      isMounted = false;
    };
  }, [discoveryAttempt, showToast]);

  const handleRetryDiscovery = () => {
    setIsDiscoveringModels(true);
    setDiscoveryAttempt((prev) => prev + 1);
  };

  const handleSelectTab = (tab: NavTab) => {
    setCurrentTab(tab);
    const mainElem = document.getElementById('main-content');
    if (mainElem) {
      mainElem.scrollTo({ top: 0, behavior: 'instant' });
    }
  };

  const handleSelectModel = (model: ModelMetadata) => {
    if (activeModel?.path !== model.path) {
      // Invalidate any in-flight solve on previous model
      solveRequestIdRef.current++;
      setIsSolving(false);
      setIsVerifying(false);
      dismissByLifecycleKey('solve_lifecycle');
      dismissByLifecycleKey('verify_lifecycle');
      setActiveModel(model);
      // STALE STATE PREVENTION: Immediately purge previous solve/verification results
      setCurrentResult(null);
      setSolveError(null);
      showToast('info', 'Model Selected', `Active model set to ${model.name}`);
    }
  };

  const handleRunSolve = async (config: SolveConfig) => {
    if (!activeModel) return;

    // Increment request token for race condition immunity
    const thisRequestId = ++solveRequestIdRef.current;

    setIsSolving(true);
    // STALE STATE PREVENTION: Clear previous result when new solve begins
    setCurrentResult(null);
    setSolveError(null);

    // Lifecycle toast: stays active until replaced on completion or error
    showToast('info', 'Starting Optimization', `Executing Kavach-Opt core for ${activeModel.name}...`, 0, 'solve_lifecycle');

    try {
      const targetPath = activeModel.path || activeModel.name;
      const solvePromise = solverBridge.solveModel(targetPath, {
        algorithm: config.problem_class,
        backend: config.backend,
        presolve: config.presolve_policy !== 'NONE',
        presolve_policy: config.presolve_policy,
        scaling: config.scaling_policy !== 'DISABLED',
        scaling_policy: config.scaling_policy,
        time_limit_seconds: config.time_limit_seconds,
        cpu_threads: config.cpu_threads,
      });

      // Ensure minimum 500ms smooth loading animation for visual UI feedback
      const [result] = await Promise.all([
        solvePromise,
        new Promise((resolve) => setTimeout(resolve, 500)),
      ]);

      // RACE CONDITION CHECK: Drop response if a newer solve was dispatched or model was changed
      if (thisRequestId !== solveRequestIdRef.current) {
        return;
      }

      setCurrentResult(result);

      if (result.status === 'OPTIMAL') {
        const objStr = typeof result.objective_value === 'number' ? result.objective_value.toFixed(4) : '';
        showToast(
          'success',
          'Optimal Solution Found',
          `${activeModel.name} solved in ${(result.solve_time_seconds * 1000).toFixed(2)} ms (z* = ${objStr})`,
          3500,
          'solve_lifecycle'
        );
      } else if (result.status === 'INFEASIBLE') {
        showToast(
          'warning',
          'Problem Infeasible',
          `${activeModel.name} proven mathematically infeasible.`,
          4000,
          'solve_lifecycle'
        );
      } else {
        showToast('info', 'Solve Completed', `Status: ${result.status}`, 3500, 'solve_lifecycle');
      }
    } catch (err: any) {
      if (thisRequestId !== solveRequestIdRef.current) return;
      console.error('Solve error:', err);
      const msg = err?.message || 'Solver execution failed';
      setSolveError(msg);
      showToast('error', 'Execution Failed', msg, 5000, 'solve_lifecycle');
      throw err;
    } finally {
      if (thisRequestId === solveRequestIdRef.current) {
        setIsSolving(false);
      }
    }
  };

  const handleVerifyCurrent = async () => {
    if (!activeModel) return;
    const thisRequestId = ++solveRequestIdRef.current;

    setIsVerifying(true);
    setSolveError(null);
    showToast('info', 'Verification Started', `Computing zero-trust mathematical checks for ${activeModel.name}...`, 0, 'verify_lifecycle');

    try {
      const targetPath = activeModel.path || activeModel.name;
      const verReport = await solverBridge.verifyModel(targetPath);

      if (thisRequestId !== solveRequestIdRef.current) return;

      if (currentResult) {
        setCurrentResult({
          ...currentResult,
          verification_report: verReport,
          verified: verReport.passed,
        });
      }

      if (verReport.passed) {
        showToast(
          'success',
          'Verification Certified',
          `Primal residual: ${verReport.recomputed_primal_residual.toExponential(2)} (within 1e-7 tol)`,
          3500,
          'verify_lifecycle'
        );
      } else {
        showToast('error', 'Verification Failed', 'One or more mathematical invariant checks failed.', 4500, 'verify_lifecycle');
      }
    } catch (err: any) {
      if (thisRequestId !== solveRequestIdRef.current) return;
      console.error('Verification error:', err);
      const msg = err?.message || 'Verification execution failed';
      setSolveError(msg);
      showToast('error', 'Verification Error', msg, 5000, 'verify_lifecycle');
    } finally {
      if (thisRequestId === solveRequestIdRef.current) {
        setIsVerifying(false);
      }
    }
  };

  return (
    <>
      <AppShell
        currentTab={currentTab}
        onSelectTab={handleSelectTab}
        currentModelName={activeModel?.name || (isDiscoveringModels ? 'Discovering Models...' : 'No Model')}
        isSolving={isSolving}
        hasCompletedRun={currentResult !== null}
        backend={currentResult?.backend_used || 'CPU_SEQUENTIAL'}
        solverStatus={currentResult?.status}
        onOpenOnboarding={() => setShowOnboarding(true)}
      >
        {currentTab === 'overview' && (
          <OverviewPage
            onNavigate={handleSelectTab}
            activeModel={activeModel ?? undefined}
            lastResult={currentResult ?? undefined}
            isDiscoveringModels={isDiscoveringModels}
          />
        )}

        {currentTab === 'solve' && (
          <SolvePage
            activeModel={activeModel}
            availableModels={availableModels}
            isDiscoveringModels={isDiscoveringModels}
            discoveryError={discoveryError}
            onRetryDiscovery={handleRetryDiscovery}
            onSelectModel={handleSelectModel}
            onNavigate={handleSelectTab}
            currentResult={currentResult}
            onRunSolve={handleRunSolve}
            isSolving={isSolving}
            solveError={solveError}
          />
        )}

        {currentTab === 'verification' && (
          <VerificationPage
            activeModel={activeModel}
            result={currentResult}
            onNavigate={handleSelectTab}
            onRunVerify={handleVerifyCurrent}
            isVerifying={isVerifying}
          />
        )}

        {currentTab === 'benchmarks' && <BenchmarksPage />}

        {currentTab === 'mrpl' && <MRPLPage />}

        {currentTab === 'system' && <SystemPage />}
      </AppShell>

      <OnboardingModal
        isOpen={showOnboarding}
        onClose={() => setShowOnboarding(false)}
        onNavigateToOverview={() => {
          setShowOnboarding(false);
          handleSelectTab('overview');
        }}
        onNavigateToSolve={() => {
          setShowOnboarding(false);
          handleSelectTab('solve');
        }}
      />
    </>
  );
}

export function App() {
  return (
    <ToastProvider>
      <AppInner />
    </ToastProvider>
  );
}

export default App;
