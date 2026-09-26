import React, { useState } from 'react';
import {
  Copy,
  Lock,
} from 'lucide-react';
import { PageHeader } from '../components/PageHeader';
import { Panel } from '../components/Panel';
import { StatusBadge } from '../components/StatusBadge';
import { Metric, MetricGroup } from '../components/Metric';
import { VerificationCheck } from '../components/VerificationCheck';
import { PrimaryButton, GhostButton } from '../components/Buttons';
import { EmptyState } from '../components/EmptyState';
import type { SolveResult, ModelMetadata } from '../types/solver';
import type { NavTab } from '../layouts/Sidebar';

interface VerificationPageProps {
  activeModel: ModelMetadata;
  result: SolveResult | null;
  onNavigate: (tab: NavTab) => void;
}

export const VerificationPage: React.FC<VerificationPageProps> = ({
  activeModel,
  result,
  onNavigate,
}) => {
  const [copied, setCopied] = useState(false);
  const [showJson, setShowJson] = useState(false);

  const report = result?.verification_report;

  const handleCopyJson = () => {
    if (!report) return;
    navigator.clipboard.writeText(JSON.stringify(report, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Independent Mathematical Verification"
        subtitle="Zero-trust post-solve validation. Recomputes all residuals from raw model input data without trusting internal solver state."
        badge={
          report ? (
            <div className="flex items-center gap-2">
              <span className="text-xs text-neutral-400 font-mono">STATUS:</span>
              <StatusBadge status={report.passed ? 'VERIFIED_OPTIMAL' : 'VERIFICATION_FAILED'} size="md" />
            </div>
          ) : undefined
        }
      />

      {/* Architecture Rationale Banner */}
      <div className="rounded-xl border border-blue-900/40 bg-gradient-to-r from-blue-950/30 via-[#111624] to-[#0d1016] p-5">
        <div className="flex items-start gap-4">
          <div className="p-2.5 rounded-lg bg-blue-900/30 border border-blue-700/50 text-blue-400 shrink-0">
            <Lock className="w-5 h-5" />
          </div>
          <div className="space-y-1.5">
            <h2 className="text-sm font-semibold text-white">
              The Kavach-Opt Mathematical Trust Model
            </h2>
            <p className="text-xs text-neutral-300 leading-relaxed max-w-3xl">
              Commercial solvers frequently report “OPTIMAL” based on internal convergence flags that can be fooled
              by basis degradation or ill-conditioned refactorizations. Kavach-Opt introduces an independent,
              unentangled verification subsystem that discards all solver internal variables and re-evaluates
              primal feasibility \(\|Ax - b\|_\infty\), dual multipliers, bounds, complementary slackness, and integrality
              using 64-bit IEEE-754 precision.
            </p>
          </div>
        </div>
      </div>

      {/* Verification Pipeline Flow */}
      <Panel title="Verification Pipeline Architecture" subtitle="Four-stage air-gapped validation sequence">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-lg bg-[#0d1016] border border-[#202736] relative">
            <div className="text-[10px] font-mono text-neutral-400 mb-1">STAGE 1</div>
            <div className="text-xs font-semibold text-white mb-2">Original Raw Model</div>
            <p className="text-[11px] text-neutral-400">
              Unaltered canonical sparse matrix A, right-hand side b, and objective vector c.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-[#0d1016] border border-[#202736] relative">
            <div className="text-[10px] font-mono text-neutral-400 mb-1">STAGE 2</div>
            <div className="text-xs font-semibold text-white mb-2">Candidate Solution</div>
            <p className="text-[11px] text-neutral-400">
              Raw decision vector x* and dual multipliers y* exported by the optimization kernel.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-blue-950/20 border border-blue-800/40 relative">
            <div className="text-[10px] font-mono text-blue-400 mb-1">STAGE 3</div>
            <div className="text-xs font-semibold text-white mb-2">Independent Verifier</div>
            <p className="text-[11px] text-neutral-300">
              Independent SpMV re-multiplication without internal solver scaling factors.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-emerald-950/20 border border-emerald-800/40 relative">
            <div className="text-[10px] font-mono text-emerald-400 mb-1">STAGE 4</div>
            <div className="text-xs font-semibold text-white mb-2">Certified Certificate</div>
            <p className="text-[11px] text-neutral-300">
              Immutable certificate confirming feasibility within strictly bounded tolerances.
            </p>
          </div>
        </div>
      </Panel>

      {/* Main Verification Content */}
      {report ? (
        <div className="space-y-6">
          {/* Certificate Overview Card */}
          <Panel
            title="Verification Certificate Summary"
            subtitle={`Target: ${activeModel.name} · Certified Status: ${report.certified_status}`}
            badge={
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleCopyJson}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-[#1e2430] hover:bg-[#2a3140] text-xs font-mono text-neutral-200 transition-colors"
                >
                  <Copy className="w-3.5 h-3.5" />
                  <span>{copied ? 'Copied JSON' : 'Copy JSON'}</span>
                </button>
              </div>
            }
          >
            <div className="space-y-6">
              <MetricGroup columns={4}>
                <Metric
                  label="Certified Feasibility"
                  value={report.passed ? 'PASSED' : 'FAILED'}
                  highlight={report.passed ? 'success' : 'danger'}
                />
                <Metric
                  label="Max Primal Residual ||Ax - b||"
                  value={report.recomputed_primal_residual.toExponential(3)}
                  subtext="Tolerance: 1.0e-08"
                  highlight="success"
                />
                <Metric
                  label="Max Dual Residual"
                  value={report.recomputed_dual_residual.toExponential(3)}
                  subtext="Tolerance: 1.0e-08"
                  highlight="success"
                />
                <Metric
                  label="Complementary Slackness"
                  value={report.recomputed_complementary_slackness.toExponential(3)}
                  subtext="Condition: x_j * s_j == 0"
                  highlight="success"
                />
              </MetricGroup>

              <div className="p-3 rounded-lg bg-[#0d1016] border border-[#202736] font-mono text-xs text-neutral-300">
                <span className="text-neutral-400">Audit Trail: </span>
                {report.summary}
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
                  {showJson ? 'Collapse JSON' : 'Expand Full JSON Artifact'}
                </GhostButton>
              </div>

              {showJson && (
                <pre className="p-4 rounded-lg bg-[#090b0e] border border-[#1e2430] font-mono text-xs text-neutral-300 overflow-x-auto max-h-96">
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
            <PrimaryButton onClick={() => onNavigate('solve')}>
              Go to Solve Control
            </PrimaryButton>
          }
        />
      )}
    </div>
  );
};
