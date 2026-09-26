import { useState } from 'react';
import { AppShell } from './layouts/AppShell';
import type { NavTab } from './layouts/Sidebar';
import { OverviewPage } from './pages/OverviewPage';
import { SolvePage } from './pages/SolvePage';
import { VerificationPage } from './pages/VerificationPage';
import { BenchmarksPage } from './pages/BenchmarksPage';
import { MRPLPage } from './pages/MRPLPage';
import { SystemPage } from './pages/SystemPage';
import type { ModelMetadata, SolveConfig, SolveResult } from './types/solver';
import { SAMPLE_MODELS, MOCK_AFIRO_RESULT } from './data/mockData';
import { solverBridge } from './lib/solverBridge';

export function App() {
  const [currentTab, setCurrentTab] = useState<NavTab>('overview');
  const [activeModel, setActiveModel] = useState<ModelMetadata>(SAMPLE_MODELS[0]);
  const [currentResult, setCurrentResult] = useState<SolveResult | null>(MOCK_AFIRO_RESULT);
  const [isSolving, setIsSolving] = useState(false);

  const handleRunSolve = async (config: SolveConfig) => {
    setIsSolving(true);
    try {
      const result = await solverBridge.solveModel(activeModel.name, {
        algorithm: config.problem_class,
        backend: config.backend,
        presolve: config.presolve_policy !== 'NONE',
        scaling: config.scaling_policy !== 'DISABLED',
        time_limit_seconds: config.time_limit_seconds,
      });
      setCurrentResult(result);
    } catch (err) {
      console.error('Solve error:', err);
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
          onSelectModel={setActiveModel}
          onNavigate={setCurrentTab}
          currentResult={currentResult}
          onRunSolve={handleRunSolve}
          isSolving={isSolving}
        />
      )}

      {currentTab === 'verification' && (
        <VerificationPage
          activeModel={activeModel}
          result={currentResult}
          onNavigate={setCurrentTab}
        />
      )}

      {currentTab === 'benchmarks' && <BenchmarksPage />}

      {currentTab === 'mrpl' && <MRPLPage />}

      {currentTab === 'system' && <SystemPage />}
    </AppShell>
  );
}

export default App;
