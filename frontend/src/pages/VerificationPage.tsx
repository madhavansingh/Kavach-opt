import React, { useState } from 'react';
import {
  Copy,
  Lock,
  RefreshCw,
  ShieldAlert,
  ArrowRight,
  Loader2,
} from 'lucide-react';
import { PageHeader } from '../components/PageHeader';
import { Panel } from '../components/Panel';
import { StatusBadge } from '../components/StatusBadge';
import { Metric, MetricGroup } from '../components/Metric';
import { VerificationCheck } from '../components/VerificationCheck';
import { PrimaryButton, GhostButton } from '../components/Buttons';
import { EmptyState } from '../components/EmptyState';
import { SkeletonPanel, SkeletonMetric, SkeletonCard } from '../components/Skeleton';
import type { SolveResult, ModelMetadata } from '../types/solver';
import type { NavTab } from '../layouts/Sidebar';
import { useToast } from '../context/useToast';

interface VerificationPageProps {
  activeModel: ModelMetadata | null;
  result: SolveResult | null;
  onNavigate: (tab: NavTab) => void;
  onRunVerify?: () => Promise<void>;
  isVerifying?: boolean;
}

export const VerificationPage: React.FC<VerificationPageProps> = ({
  activeModel,
  result,
  onNavigate,
  onRunVerify,
  isVerifying = false,
}) => {
  const [copied, setCopied] = useState(false);
  const [showJson, setShowJson] = useState(false);
  const { showToast } = useToast();

  const report = result?.verification_report;

  const handleCopyJson = () => {
    if (!report) return;
    navigator.clipboard.writeText(JSON.stringify(report, null, 2));
    setCopied(true);
    showToast('success', 'Copied Certificate', 'Verification certificate JSON copied to clipboard.');
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Independent Mathematical Verification"
        subtitle="Zero-trust post-solve validation. Recomputes all residuals from raw model input data without trusting internal solver state."
        badge={
          isVerifying ? (
            <span className="text-xs font-mono font-bold text-blue-400 px-2.5 py-1 rounded bg-blue-950/60 border border-blue-800/60 flex items-center gap-1.5 animate-pulse">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              AUDIT RUNNING
            </span>
          ) : report ? (
            <div className="flex items-center gap-2">
              <span className="text-xs text-neutral-400 font-sans">Status:</span>
              <StatusBadge status={report.passed ? 'VERIFIED_OPTIMAL' : 'VERIFICATION_FAILED'} size="md" />
            </div>
          ) : undefined
        }
      />

      {/* Architecture Rationale Banner */}
      <div className="rounded-xl border border-blue-900/40 bg-gradient-to-r from-blue-950/30 via-[#11151f] to-[#0a0d12] p-5 shadow-xs">
        <div className="flex items-start gap-4">
          <div className="p-2.5 rounded-lg bg-blue-900/30 border border-blue-700/50 text-blue-400 shrink-0">
            <Lock className="w-5 h-5" />
          </div>
          <div className="space-y-1.5">
            <h2 className="text-sm font-semibold text-white font-sans">
              The Kavach-Opt Zero-Trust Mathematical Audit
            </h2>
            <p className="text-xs text-neutral-300 leading-relaxed max-w-3xl font-sans">
              The solver is not trusted by itself. Solvers can report &ldquo;OPTIMAL&rdquo; based on internal
              convergence flags that may be masked by ill-conditioned refactorizations. Kavach-Opt enforces an
              air-gapped verification subsystem that completely discards internal solver flags and independently
              re-multiplies primal feasibility \(\|Ax - b\|_\infty\), dual multipliers, complementary slackness,
              and discrete integrality using 64-bit IEEE-754 precision directly against original raw inputs.
            </p>
          </div>
        </div>
      </div>

      {/* Verification Pipeline Flow */}
      <Panel title="Verification Pipeline Architecture" subtitle="Four-stage air-gapped validation sequence">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-lg bg-[#141926] border border-[#1f283a] relative">
            <div className="text-[10px] font-mono text-neutral-400 mb-1">STAGE 1</div>
            <div className="text-xs font-semibold text-white mb-2 font-sans">Original Raw Model</div>
            <p className="text-[11px] text-neutral-400 font-sans">
              Unaltered canonical sparse matrix A, right-hand side b, and objective vector c.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-[#141926] border border-[#1f283a] relative">
            <div className="text-[10px] font-mono text-neutral-400 mb-1">STAGE 2</div>
            <div className="text-xs font-semibold text-white mb-2 font-sans">Candidate Solution</div>
            <p className="text-[11px] text-neutral-400 font-sans">
              Raw decision vector x* and dual multipliers y* exported by the optimization kernel.
            </p>
          </div>

          <div
            className={`p-4 rounded-lg border relative transition-colors ${
              isVerifying
                ? 'bg-blue-950/40 border-blue-500/60 ring-1 ring-blue-500/30'
                : 'bg-blue-950/20 border-blue-800/40'
            }`}
          >
            <div className="text-[10px] font-mono text-blue-400 mb-1 flex items-center justify-between">
              <span>STAGE 3</span>
              {isVerifying && <Loader2 className="w-3 h-3 animate-spin text-blue-400" />}
            </div>
            <div className="text-xs font-semibold text-white mb-2 font-sans">Independent Verifier</div>
            <p className="text-[11px] text-neutral-300 font-sans">
              Independent SpMV re-multiplication without internal solver scaling factors.
            </p>
          </div>

          <div
            className={`p-4 rounded-lg border relative transition-colors ${
              report?.passed
                ? 'bg-emerald-950/20 border-emerald-800/40'
                : report && !report.passed
                ? 'bg-rose-950/20 border-rose-800/40'
                : 'bg-[#141926] border-[#1f283a]'
            }`}
          >
            <div className="text-[10px] font-mono mb-1 text-emerald-400">STAGE 4</div>
            <div className="text-xs font-semibold text-white mb-2 font-sans">Certified Certificate</div>
            <p className="text-[11px] text-neutral-300 font-sans">
              Immutable certificate confirming feasibility within strictly bounded tolerances.
            </p>
          </div>
        </div>
      </Panel>

      {/* Main Verification Content or Loading Skeleton */}
      {isVerifying ? (
        <div className="space-y-6 animate-fade-in" role="status" aria-busy="true">
          <Panel
            title="Independent Verification in Progress"
            subtitle={`Recomputing exact residuals for ${activeModel?.name || 'Current Solution'}...`}
            badge={
              <span className="text-xs font-mono text-blue-400 flex items-center gap-1.5">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Computing ||Ax - b||
              </span>
            }
          >
            <div className="space-y-4">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <SkeletonMetric />
                <SkeletonMetric />
                <SkeletonMetric />
                <SkeletonMetric />
              </div>
              <SkeletonCard />
            </div>
          </Panel>
          <SkeletonPanel titleWidth="w-64" subtitleWidth="w-96">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <SkeletonCard />
              <SkeletonCard />
            </div>
          </SkeletonPanel>
        </div>
      ) : report ? (
        <div className="space-y-6 animate-fade-in">
          {/* Certificate Overview Card */}
          <Panel
            title="Verification Certificate Summary"
            subtitle={`Target: ${activeModel?.name || 'Selected Model'} · Certified Status: ${report.certified_status}`}
            badge={
              <div className="flex items-center gap-2">
                {onRunVerify && (
                  <button
                    type="button"
                    disabled={isVerifying}
                    onClick={onRunVerify}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-xs font-mono text-white transition-colors"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${isVerifying ? 'animate-spin' : ''}`} />
                    <span>Re-run Verifier</span>
                  </button>
                )}
                <button
                  type="button"
                  onClick={handleCopyJson}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#161c28] hover:bg-[#20283a] border border-[#273245] text-xs font-mono text-neutral-200 transition-colors"
                >
                  <Copy className="w-3.5 h-3.5" />
                  <span>{copied ? 'Copied JSON' : 'Copy JSON'}</span>
                </button>
              </div>
            }
          >
            <div className="space-y-6">
              {/* Failure Alert Banner if verification failed */}
              {!report.passed && (
                <div className="p-4 rounded-xl border border-rose-800/60 bg-rose-950/30 text-rose-200 text-xs flex items-start gap-3">
                  <ShieldAlert className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
                  <div className="space-y-1">
                    <div className="font-semibold text-rose-300 font-sans text-xs">
                      Zero-Trust Verification Detected Invariant Violation
                    </div>
                    <p className="text-neutral-300 font-sans text-xs leading-relaxed">
                      The candidate solution generated by the optimization algorithm does not satisfy the independent
                      feasibility tolerances. The solver&apos;s internal status has been overridden to{' '}
                      <span className="font-mono text-rose-400 font-bold">VERIFICATION_FAILED</span>.
                    </p>
                  </div>
                </div>
              )}

              <MetricGroup columns={4}>
                <Metric
                  label="Certified Feasibility"
                  value={report.passed ? 'PASSED' : 'FAILED'}
                  highlight={report.passed ? 'success' : 'danger'}
                />
                <Metric
                  label="Max Primal Residual ||Ax - b||"
                  value={report.recomputed_primal_residual.toExponential(3)}
                  subtext="Tolerance: 1.0e-06"
                  highlight={report.recomputed_primal_residual <= 1e-6 ? 'success' : 'danger'}
                />
                {activeModel?.problem_class === 'LP' ? (
                  <>
                    <Metric
                      label="Max Dual Residual"
                      value={report.recomputed_dual_residual.toExponential(3)}
                      subtext="Tolerance: 1.0e-06"
                      highlight={report.recomputed_dual_residual <= 1e-6 ? 'success' : 'danger'}
                    />
                    <Metric
                      label="Complementary Slackness"
                      value={report.recomputed_complementary_slackness.toExponential(3)}
                      subtext="Condition: x_j * s_j == 0"
                      highlight={report.recomputed_complementary_slackness <= 1e-6 ? 'success' : 'danger'}
                    />
                  </>
                ) : activeModel?.problem_class === 'MILP' ? (
                  <>
                    <Metric
                      label="Integrality Violation"
                      value={(report.recomputed_integrality_violation ?? 0).toExponential(3)}
                      subtext="Tolerance: 1.0e-05"
                      highlight={(report.recomputed_integrality_violation ?? 0) <= 1e-5 ? 'success' : 'danger'}
                    />
                    <Metric
                      label="Recomputed Objective"
                      value={(report.recomputed_objective ?? 0).toFixed(4)}
                      subtext="Direct c^T x* evaluation"
                      highlight="normal"
                    />
                  </>
                ) : (
                  <>
                    <Metric
                      label="KKT Stationarity Residual"
                      value={(report.recomputed_kkt_residual ?? 0).toExponential(3)}
                      subtext="Tolerance: 1.0e-06"
                      highlight={(report.recomputed_kkt_residual ?? 0) <= 1e-6 ? 'success' : 'danger'}
                    />
                    <Metric
                      label="Recomputed Objective"
                      value={(report.recomputed_objective ?? 0).toFixed(4)}
                      subtext="Direct 0.5 x^T Q x + c^T x"
                      highlight="normal"
                    />
                  </>
                )}
              </MetricGroup>

              <div className="p-3.5 rounded-lg bg-[#141926] border border-[#1f283a] font-mono text-xs text-neutral-300">
                <span className="text-neutral-400 font-sans">Audit Trail: </span>
                <span>{report.summary}</span>
              </div>
            </div>
          </Panel>

          {/* Mathematical Assertions Breakdown */}
          {report.checks && report.checks.length > 0 && (
            <Panel
              title="Mathematical Assertions & Invariant Checks"
              subtitle="Granular evaluation of KKT optimality conditions and bounds"
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {report.checks.map((check, idx) => (
                  <VerificationCheck key={idx} check={check} />
                ))}
              </div>
            </Panel>
          )}

          {/* Raw JSON Certificate Accordion */}
          <Panel
            title="Raw Mathematical Certificate (JSON)"
            subtitle="Machine-readable proof artifact generated for audits and automated pipeline ingestion"
          >
            <div className="space-y-3">
              <div className="flex justify-end">
                <GhostButton size="sm" onClick={() => setShowJson(!showJson)}>
                  {showJson ? 'Collapse JSON Artifact' : 'Expand Full JSON Artifact'}
                </GhostButton>
              </div>

              {showJson && (
                <pre className="p-4 rounded-lg bg-[#0c0f17] border border-[#1d2332] font-mono text-xs text-neutral-300 overflow-x-auto max-h-96">
                  {JSON.stringify(report, null, 2)}
                </pre>
              )}
            </div>
          </Panel>
        </div>
      ) : (
        <EmptyState
          title="No Solution To Verify"
          description="Independent verification requires a candidate solution. Go to the Solve page, execute an optimization model, and inspect the resulting verification certificate here."
          action={
            <PrimaryButton onClick={() => onNavigate('solve')} icon={<ArrowRight className="w-4 h-4" />}>
              Go to Solve Control
            </PrimaryButton>
          }
        />
      )}
    </div>
  );
};
