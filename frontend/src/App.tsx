import { useState, useEffect } from 'react';
import { AppShell } from './layouts/AppShell';
import type { NavTab } from './layouts/Sidebar';
import { OverviewPage } from './pages/OverviewPage';
import { SolvePage } from './pages/SolvePage';
import { VerificationPage } from './pages/VerificationPage';
import { BenchmarksPage } from './pages/BenchmarksPage';
import { MRPLPage } from './pages/MRPLPage';
import { SystemPage } from './pages/SystemPage';
import type { ModelMetadata, SolveConfig, SolveResult } from './types/solver';
import { SAMPLE_MODELS } from './data/mockData';
import { solverBridge } from './lib/solverBridge';

export function App() {
  const [currentTab, setCurrentTab] = useState<NavTab>('overview');
  const [availableModels, setAvailableModels] = useState<ModelMetadata[]>(SAMPLE_MODELS);
  const [activeModel, setActiveModel] = useState<ModelMetadata>(SAMPLE_MODELS[0]);
  const [currentResult, setCurrentResult] = useState<SolveResult | null>(null);
  const [isSolving, setIsSolving] = useState(false);
  const [solveError, setSolveError] = useState<string | null>(null);

  // Discover all authentic models from repository
  useEffect(() => {
    let isMounted = true;
    solverBridge
      .getAvailableModels()
      .then((models) => {
        if (isMounted && models && models.length > 0) {
          setAvailableModels(models);
          // Set afiro.mps as default active model if present
          const afiro = models.find((m) => m.name === 'afiro.mps') || models[0];
          setActiveModel(afiro);
        }
      })
      .catch((err) => {
        console.warn('Failed to load discovered models from adapter:', err);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const handleRunSolve = async (config: SolveConfig) => {
    setIsSolving(true);
    setSolveError(null);
    try {
      const targetPath = activeModel.path || activeModel.name;
      const result = await solverBridge.solveModel(targetPath, {
        algorithm: config.problem_class,
        backend: config.backend,
        presolve: config.presolve_policy !== 'NONE',
        presolve_policy: config.presolve_policy,
        scaling: config.scaling_policy !== 'DISABLED',
        scaling_policy: config.scaling_policy,
        time_limit_seconds: config.time_limit_seconds,
        cpu_threads: config.cpu_threads,
      });
      setCurrentResult(result);
    } catch (err: any) {
      console.error('Solve error:', err);
      setSolveError(err?.message || 'Solver execution failed');
    } finally {
      setIsSolving(false);
    }
  };

  const handleVerifyCurrent = async () => {
    if (!activeModel) return;
    setIsSolving(true);
    setSolveError(null);
    try {
      const targetPath = activeModel.path || activeModel.name;
      const verReport = await solverBridge.verifyModel(targetPath);
      if (currentResult) {
        setCurrentResult({
          ...currentResult,
          verification_report: verReport,
          verified: verReport.passed,
        });
      }
    } catch (err: any) {
      console.error('Verification error:', err);
      setSolveError(err?.message || 'Verification execution failed');
    } finally {
      setIsSolving(false);
    }
  };

  return (
    <AppShell
      currentTab={currentTab}
      onSelectTab={setCurrentTab}
      currentModelName={activeModel.name}
      backend={currentResult?.backend_used || 'CPU_SEQUENTIAL'}
      solverStatus={currentResult?.status}
    >
      {currentTab === 'overview' && (
        <OverviewPage
          onNavigate={setCurrentTab}
          activeModel={activeModel}
          lastResult={currentResult ?? undefined}
        />
      )}

      {currentTab === 'solve' && (
        <SolvePage
          activeModel={activeModel}
          availableModels={availableModels}
          onSelectModel={setActiveModel}
          onNavigate={setCurrentTab}
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
          onNavigate={setCurrentTab}
          onRunVerify={handleVerifyCurrent}
          isVerifying={isSolving}
        />
      )}

      {currentTab === 'benchmarks' && <BenchmarksPage />}

      {currentTab === 'mrpl' && <MRPLPage />}

      {currentTab === 'system' && <SystemPage />}
    </AppShell>
  );
}

export default App;

