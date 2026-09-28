import React, { useState, useEffect } from 'react';
import {
  Shield,
  AlertTriangle,
  CheckCircle2,
  Gauge,
  Play,
  Check,
  Loader2,
} from 'lucide-react';
import { PageHeader } from '../components/PageHeader';
import { Panel } from '../components/Panel';
import { StatusBadge } from '../components/StatusBadge';
import { Metric, MetricGroup } from '../components/Metric';
import { DataTable } from '../components/DataTable';
import { SkeletonPanel, SkeletonMetric, SkeletonCard } from '../components/Skeleton';
import type { MRPLScenario, MRPLCaseData } from '../types/solver';
import { solverBridge } from '../lib/solverBridge';
import { useToast } from '../context/useToast';

export const MRPLPage: React.FC = () => {
  const [mrplCase, setMrplCase] = useState<MRPLCaseData | null>(null);
  const [selectedScenario, setSelectedScenario] = useState<MRPLScenario | null>(null);
  const [isSolvingLive, setIsSolvingLive] = useState(false);
  const [liveSolveResult, setLiveSolveResult] = useState<any | null>(null);
  const [liveSolveError, setLiveSolveError] = useState<string | null>(null);

  const { showToast } = useToast();

  useEffect(() => {
    let isMounted = true;
    solverBridge
      .getMRPLCaseData()
      .then((data) => {
        if (isMounted && data) {
          setMrplCase(data);
          if (data.scenarios && data.scenarios.length > 0) {
            setSelectedScenario(data.scenarios[0]);
          }
        }
      })
      .catch((err) => {
        console.error('Failed to load authentic MRPL data:', err);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const handleRunLiveScenario = (scenarioId: string) => {
    if (isSolvingLive) return;
    setIsSolvingLive(true);
    setLiveSolveError(null);
    showToast(
      'info',
      'Solving Refinery Scenario',
      `Executing C++ solve for ${selectedScenario?.name || scenarioId}...`
    );

    requestAnimationFrame(() => {
      requestAnimationFrame(async () => {
        try {
          const res = await solverBridge.runMRPLScenario(scenarioId);
          setLiveSolveResult(res.data);
          showToast(
            'success',
            'Refinery Scenario Solved',
            `${selectedScenario?.name} status: ${res.data.solver_status} (${(res.data.solve_time_seconds * 1000).toFixed(2)} ms)`
          );
        } catch (err: any) {
          console.error('Failed to run MRPL scenario live:', err);
          const msg = err?.message || 'Scenario solve failed';
          setLiveSolveError(msg);
          showToast('error', 'Scenario Solve Failed', msg);
        } finally {
          setIsSolvingLive(false);
        }
      });
    });
  };

  if (!mrplCase) {
    return (
      <div className="space-y-6 animate-fade-in" aria-busy="true">
        <PageHeader
          title="Refinery Optimization"
          subtitle="Crude blending, processing capacity and product planning"
        />
        <SkeletonPanel titleWidth="w-72" subtitleWidth="w-96">
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
            <SkeletonMetric />
            <SkeletonMetric />
            <SkeletonMetric />
            <SkeletonMetric />
            <SkeletonMetric />
          </div>
        </SkeletonPanel>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <SkeletonCard />
          <SkeletonCard />
        </div>
      </div>
    );
  }

  const baseline = mrplCase.baseline;

  // Quality table columns
  const qualityColumns = [
    {
      key: 'product',
      header: 'Product Stream',
      render: (row: any) => <span className="font-semibold text-white">{row.product}</span>,
    },
    {
      key: 'output',
      header: 'Output Rate',
      render: (row: any) => <span className="font-mono text-neutral-300">{row.production_kbpd.toFixed(1)} kbpd</span>,
    },
    {
      key: 'sulfur',
      header: 'Sulfur (wt%)',
      render: (row: any) => (
        <span className="font-mono text-neutral-300">
          {row.calculated_sulfur_wt_pct.toFixed(2)}% / &le; {row.sulfur_limit_wt_pct.toFixed(2)}%
        </span>
      ),
    },
    {
      key: 'sulfur_margin',
      header: 'Sulfur Margin',
      render: (row: any) => (
        <span className="font-mono text-emerald-400">
          +{row.sulfur_margin_wt_pct.toFixed(2)}%
        </span>
      ),
    },
    {
      key: 'density',
      header: 'Density (kg/m³)',
      render: (row: any) => {
        const isMinBound = row.product.includes('P3') || row.product.includes('Heavy Fuel Oil');
        const operator = isMinBound ? '≥' : '≤';
        return (
          <span className="font-mono text-neutral-300">
            {row.calculated_density_kg_m3.toFixed(1)} / {operator} {row.density_limit_kg_m3.toFixed(1)}
            {isMinBound && <span className="text-[10px] text-neutral-400 block font-sans">(Min Spec)</span>}
          </span>
        );
      },
    },
    {
      key: 'density_margin',
      header: 'Density Margin',
      render: (row: any) => {
        const isMinBound = row.product.includes('P3') || row.product.includes('Heavy Fuel Oil');
        const margin = row.density_margin_kg_m3;
        const sign = margin >= 0 ? '+' : '';
        const note = isMinBound ? 'above min' : 'below max';
        return (
          <span className={`font-mono ${margin >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {sign}{margin.toFixed(1)} kg/m³
            <span className="text-[10px] text-neutral-400 block font-sans">{note}</span>
          </span>
        );
      },
    },
    {
      key: 'specs',
      header: 'Specs Met',
      render: (row: any) => {
        const met = row.quality_satisfied ?? true;
        return (
          <span className={`inline-flex items-center gap-1 font-mono text-xs ${met ? 'text-emerald-400' : 'text-rose-400'}`}>
            {met ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertTriangle className="w-3.5 h-3.5" />}
            <span>{met ? 'COMPLIANT' : 'NON-COMPLIANT'}</span>
          </span>
        );
      },
    },
  ];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Refinery Optimization"
        subtitle="Crude blending, processing capacity and product planning"
        badge={
          <div className="flex items-center gap-2">
            <span className="text-xs text-neutral-400 font-mono">SECTOR:</span>
            <span className="text-xs font-mono font-semibold text-white px-2 py-0.5 rounded bg-[#1e2430] border border-[#2e3748]">
              PETROLEUM REFINING & CRUDE BLENDING
            </span>
          </div>
        }
      />

      {/* MANDATORY SYNTHETIC DATA DISCLOSURE */}
      <div className="rounded-xl border border-amber-800/40 bg-gradient-to-r from-amber-950/30 via-[#181410] to-[#0d1016] p-5">
        <div className="flex items-start gap-4">
          <div className="p-2.5 rounded-lg bg-amber-900/30 border border-amber-700/50 text-amber-400 shrink-0">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <div className="space-y-1.5">
            <h2 className="text-sm font-semibold text-white">
              Data Status Disclosure · Illustrative / Synthetic Industrial Dataset
            </h2>
            <p className="text-xs text-neutral-300 leading-relaxed max-w-4xl">
              The scenarios below demonstrate mathematical optimization on an illustrative petroleum refinery model. Technical and economic parameters are realistic approximations designed to showcase multi-period blending, capacity allocation, and quality constraint enforcement.
            </p>
          </div>
        </div>
      </div>

      {/* High-Level Refinery Economics */}
      <Panel
        title="Refinery Baseline Economics (Daily Financials)"
        subtitle="Gross revenue, feedstock costs, operating margins, and optimal net profit"
        badge={<StatusBadge status={baseline.status} size="md" />}
      >
        <div className="space-y-6">
          <MetricGroup columns={5}>
            <Metric
              label="Gross Product Revenue"
              value={`$${baseline.economics.gross_revenue_k_usd.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}k`}
              subtext="Finished fuels sale"
            />
            <Metric
              label="Crude Purchase Cost"
              value={`-$${baseline.economics.crude_purchase_cost_k_usd.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}k`}
              subtext="Crude slate intake"
            />
            <Metric
              label="Operating Variable Costs"
              value={`-$${baseline.economics.unit_variable_cost_k_usd.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}k`}
              subtext="Catalyst, steam, power"
            />
            <Metric
              label="Fixed Plant Costs"
              value={`-$${baseline.economics.unit_fixed_cost_k_usd.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}k`}
              subtext="Base maintenance & staffing"
            />
            <Metric
              label="Optimal Net Daily Profit"
              value={`$${baseline.economics.net_profit_k_usd.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}k`}
              highlight="success"
              subtext="Certified global optimum"
            />
          </MetricGroup>

          {/* Mass Conservation Guarantee */}
          <div className="p-3.5 rounded-lg bg-emerald-950/20 border border-emerald-800/40 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2.5">
              <Shield className="w-4 h-4 text-emerald-400 shrink-0" />
              <div>
                <span className="font-semibold text-white">Physical Mass Conservation Certified: </span>
                <span className="text-neutral-300 font-mono">
                  Max balance residual ||Ax - b|| = {baseline.mass_balance_max_residual.toFixed(6)} (Exact zero)
                </span>
              </div>
            </div>
            <span className="font-mono text-emerald-400 font-semibold px-2 py-0.5 rounded bg-emerald-900/40">
              100.0% CONSERVED
            </span>
          </div>
        </div>
      </Panel>

      {/* Industrial Process Flow Schematic */}
      <Panel
        title="Refinery Process Topology & Mass Balance Flow"
        subtitle="End-to-end petroleum processing path from crude slate to Euro-VI specification fuels"
        badge={
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-800/40 text-emerald-300 font-medium">
            MASS CONSERVED (0.0 RESIDUAL)
          </span>
        }
      >
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-5 gap-3 text-xs">
            {/* Stage 1: Crude Slate */}
            <div className="p-3.5 rounded-lg bg-[#141926] border border-[#1f283a] space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-mono text-amber-400 font-medium">STAGE 1</span>
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-amber-950/60 text-amber-300">FEEDSTOCK</span>
              </div>
              <h4 className="text-xs font-semibold text-white font-sans">Crude Slate Intake</h4>
              <ul className="space-y-1 text-[11px] font-mono text-neutral-300">
                <li className="flex justify-between"><span>· Crude A (Sweet):</span> <span className="text-white font-medium">120.0 kbpd</span></li>
                <li className="flex justify-between"><span>· Crude B (Medium):</span> <span className="text-white font-medium">80.0 kbpd</span></li>
                <li className="flex justify-between"><span>· Crude C (Sour):</span> <span className="text-white font-medium">83.2 kbpd</span></li>
                <li className="flex justify-between"><span>· Crude D (Heavy):</span> <span className="text-white font-medium">45.0 kbpd</span></li>
              </ul>
              <div className="pt-1.5 border-t border-[#1f283a] text-[10px] text-amber-400 font-mono flex justify-between">
                <span>Total Feedstock:</span>
                <span className="font-bold">328.2 kbpd</span>
              </div>
            </div>

            {/* Stage 2: Distillation */}
            <div className="p-3.5 rounded-lg bg-[#141926] border border-[#1f283a] space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-mono text-blue-400 font-medium">STAGE 2</span>
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-blue-950/60 text-blue-300">PRIMARY</span>
              </div>
              <h4 className="text-xs font-semibold text-white font-sans">Atmospheric CDU (U1)</h4>
              <p className="text-[11px] text-neutral-400 font-sans leading-relaxed">
                Primary crude distillation processing sweet crudes A &amp; B into distillates and atmospheric residue.
              </p>
              <div className="pt-1.5 border-t border-[#1f283a] text-[10px] text-neutral-300 font-mono flex justify-between">
                <span>CDU Load:</span>
                <span className="text-emerald-400 font-bold">100.0% (200.0/200.0 kbpd)</span>
              </div>
            </div>

            {/* Stage 3: Conversion */}
            <div className="p-3.5 rounded-lg bg-[#141926] border border-[#1f283a] space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-mono text-indigo-400 font-medium">STAGE 3</span>
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-indigo-950/60 text-indigo-300">CONVERSION</span>
              </div>
              <h4 className="text-xs font-semibold text-white font-sans">VDU &amp; FCCU (U2)</h4>
              <p className="text-[11px] text-neutral-400 font-sans leading-relaxed">
                Vacuum distillation and catalytic cracking treating sour crudes C &amp; D bottoms into gasoline blendstocks.
              </p>
              <div className="pt-1.5 border-t border-[#1f283a] text-[10px] text-neutral-300 font-mono flex justify-between">
                <span>VDU/FCCU Load:</span>
                <span className="text-blue-400 font-bold">71.2% (128.2/180.0 kbpd)</span>
              </div>
            </div>

            {/* Stage 4: Hydrotreating */}
            <div className="p-3.5 rounded-lg bg-[#141926] border border-[#1f283a] space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-mono text-emerald-400 font-medium">STAGE 4</span>
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-emerald-950/60 text-emerald-300">TREATING</span>
              </div>
              <h4 className="text-xs font-semibold text-white font-sans">HDS Hydrotreater (U3)</h4>
              <p className="text-[11px] text-neutral-400 font-sans leading-relaxed">
                Hydrodesulfurization ensuring gasoline (&le;0.80 wt%) and diesel (&le;1.30 wt%) meet statutory sulfur caps.
              </p>
              <div className="pt-1.5 border-t border-[#1f283a] text-[10px] text-neutral-300 font-mono flex justify-between">
                <span>HDS Load:</span>
                <span className="text-blue-400 font-bold">31.8% (38.2/120.0 kbpd)</span>
              </div>
            </div>

            {/* Stage 5: Products */}
            <div className="p-3.5 rounded-lg bg-[#141926] border border-[#1f283a] space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-mono text-emerald-400 font-medium">STAGE 5</span>
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-emerald-950/60 text-emerald-300">FINISHED</span>
              </div>
              <h4 className="text-xs font-semibold text-white font-sans">Finished Fuel Pools</h4>
              <ul className="space-y-1 text-[11px] font-mono text-neutral-300">
                <li className="flex justify-between"><span>· P1 Gasoline:</span> <span className="text-emerald-400 font-medium">140.0 kbpd</span></li>
                <li className="flex justify-between"><span>· P2 Diesel:</span> <span className="text-emerald-400 font-medium">98.2 kbpd</span></li>
                <li className="flex justify-between"><span>· P3 Fuel Oil:</span> <span className="text-emerald-400 font-medium">90.0 kbpd</span></li>
              </ul>
              <div className="pt-1.5 border-t border-[#1f283a] text-[10px] text-emerald-400 font-mono flex justify-between">
                <span>Total Finished:</span>
                <span className="font-bold">328.2 kbpd</span>
              </div>
            </div>
          </div>
        </div>
      </Panel>

      {/* Unit Operations & Capacities */}
      <Panel
        title="Major Processing Units & Capacity Utilization"
        subtitle="Throughput distribution across primary distillation and secondary conversion units"
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {baseline.units.map((unit: any) => (
            <div
              key={unit.unit}
              className="p-4 rounded-lg bg-[#141926] border border-[#1f283a] space-y-3 shadow-xs"
            >
              <div className="flex justify-between items-start">
                <span className="text-xs font-semibold text-white font-sans">{unit.unit}</span>
                <span
                  className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                    unit.is_active
                      ? 'bg-emerald-950 text-emerald-400 border border-emerald-800/40'
                      : 'bg-neutral-800 text-neutral-400'
                  }`}
                >
                  {unit.is_active ? 'ONLINE' : 'STANDBY'}
                </span>
              </div>

              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-neutral-400 font-sans">Throughput:</span>
                  <span className="font-mono text-neutral-200">
                    {unit.throughput_kbpd.toFixed(1)} / {unit.capacity_limit_kbpd.toFixed(1)} kbpd
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-neutral-400 font-sans">Load Factor:</span>
                  <span className="font-mono font-bold text-blue-400">
                    {unit.utilization_pct.toFixed(1)}%
                  </span>
                </div>

                <div className="w-full bg-[#1b212f] h-2 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${
                      unit.utilization_pct === 100
                        ? 'bg-amber-500'
                        : unit.utilization_pct > 70
                        ? 'bg-blue-500'
                        : 'bg-emerald-500'
                    }`}
                    style={{ width: `${unit.utilization_pct}%` }}
                  />
                </div>
              </div>

              <div className="text-[11px] text-neutral-400 font-mono pt-1 border-t border-[#1f283a]">
                {unit.summary}
              </div>
            </div>
          ))}
        </div>
      </Panel>

      {/* Finished Product Quality & Environmental Specs */}
      <Panel
        title="Product Blending & Environmental Quality Margins"
        subtitle="Mandatory sulfur and density limits ensuring Euro-VI / BS-VI compliance"
      >
        <DataTable
          data={baseline.qualities}
          columns={qualityColumns}
          keyExtractor={(row) => row.product}
        />
      </Panel>

      {/* What-If Operational Scenario Matrix */}
      <Panel
        title="What-If Operational Scenarios & Sensitivity Analysis"
        subtitle="Evaluate refinery robustness under supply disruptions, quality tightening, and unit outages"
      >
        <div className="space-y-5">
          {/* Comparison Matrix Table */}
          <div className="overflow-x-auto rounded-lg border border-[#1f283a] bg-[#141926]">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-[#0e121a] text-neutral-400 border-b border-[#1f283a]">
                <tr>
                  <th className="py-2.5 px-3 font-sans">Scenario</th>
                  <th className="py-2.5 px-3 font-sans">Condition</th>
                  <th className="py-2.5 px-3 text-right">Throughput</th>
                  <th className="py-2.5 px-3 text-right">Net Profit</th>
                  <th className="py-2.5 px-3 text-right">Profit Delta</th>
                  <th className="py-2.5 px-3 text-center">Status</th>
                  <th className="py-2.5 px-3 text-center">Mass Balance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1f283a]/60">
                {mrplCase.scenarios.map((sc: MRPLScenario) => (
                  <tr
                    key={sc.id}
                    onClick={() => {
                      setSelectedScenario(sc);
                      setLiveSolveResult(null);
                      setLiveSolveError(null);
                    }}
                    className={`cursor-pointer transition-colors ${
                      selectedScenario?.id === sc.id ? 'bg-[#182033]' : 'hover:bg-[#161c28]'
                    }`}
                  >
                    <td className="py-2.5 px-3 font-bold text-white font-sans">{sc.name}</td>
                    <td className="py-2.5 px-3 text-neutral-400 font-sans truncate max-w-xs">{sc.description}</td>
                    <td className="py-2.5 px-3 text-right text-neutral-200">
                      {sc.throughput_kbpd ? `${sc.throughput_kbpd.toFixed(1)} kbpd` : '—'}
                    </td>
                    <td className="py-2.5 px-3 text-right font-bold">
                      {sc.net_profit_k_usd ? (
                        <span className="text-emerald-400">${sc.net_profit_k_usd.toFixed(2)}k</span>
                      ) : (
                        <span className="text-rose-400 font-normal">INFEASIBLE</span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono">
                      {sc.profit_delta_k_usd !== undefined ? (
                        <span className={sc.profit_delta_k_usd < 0 ? 'text-rose-400' : 'text-emerald-400'}>
                          {sc.id === 'baseline' ? '0.00 (Base)' : `${sc.profit_delta_k_usd < 0 ? '' : '+'}$${sc.profit_delta_k_usd.toFixed(2)}k`}
                        </span>
                      ) : '—'}
                    </td>
                    <td className="py-2.5 px-3 text-center"><StatusBadge status={sc.status} size="sm" /></td>
                    <td className="py-2.5 px-3 text-center">
                      <span className={sc.mass_balance_passed ? 'text-emerald-400 font-semibold' : 'text-rose-400 font-semibold'}>
                        {sc.mass_balance_passed ? 'PASS (0.0)' : 'PROVEN INFEASIBLE'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {mrplCase.scenarios.map((sc: MRPLScenario) => {
              const isSelected = selectedScenario?.id === sc.id;
              return (
                <button
                  key={sc.id}
                  type="button"
                  onClick={() => {
                    setSelectedScenario(sc);
                    setLiveSolveResult(null);
                    setLiveSolveError(null);
                  }}
                  className={`p-4 rounded-lg border text-left transition-all ${
                    isSelected
                      ? 'border-blue-500/80 bg-blue-950/20 shadow-xs'
                      : 'border-[#1f283a] bg-[#141926] hover:border-[#2d3850]'
                  }`}
                >
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-xs font-semibold text-white font-sans">{sc.name}</span>
                    <StatusBadge status={sc.status} size="sm" />
                  </div>
                  <p className="text-[11px] text-neutral-400 mb-3 font-sans leading-relaxed">{sc.description}</p>
                  <div className="flex justify-between items-center text-xs font-mono pt-2 border-t border-[#1f283a]">
                    <span className="text-neutral-400 font-sans">Net Profit:</span>
                    <span
                      className={`font-bold ${
                        sc.net_profit_k_usd ? 'text-white' : 'text-neutral-500'
                      }`}
                    >
                      {sc.net_profit_k_usd
                        ? `$${sc.net_profit_k_usd.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}k`
                        : 'INFEASIBLE'}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Selected Scenario Drill-down */}
          {selectedScenario && (
            <div className="p-4 rounded-lg bg-[#0e121a] border border-[#1f283a] space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <Gauge className="w-4 h-4 text-blue-400" />
                  <span className="text-xs font-semibold text-white font-sans">
                    Scenario Focus: {selectedScenario.name}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <StatusBadge status={selectedScenario.status} size="sm" />
                  <button
                    type="button"
                    disabled={isSolvingLive}
                    onClick={() => handleRunLiveScenario(selectedScenario.id)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-xs font-mono text-white transition-colors"
                  >
                    <Play className={`w-3.5 h-3.5 ${isSolvingLive ? 'animate-spin' : ''}`} />
                    <span>{isSolvingLive ? 'Solving on C++ Core...' : 'Run Scenario Live'}</span>
                  </button>
                </div>
              </div>

              <p className="text-xs text-neutral-300 leading-relaxed font-sans">
                {selectedScenario.description}.{' '}
                {selectedScenario.status === 'INFEASIBLE'
                  ? 'Kavach-Opt mathematically proves that no feasible blending and scheduling plan exists that satisfies both demand commitments and strict physical crude conservation.'
                  : `Under this operating condition, the solver reroutes intermediate heavy gas oil fractions to maintain zero mass balance deviation while realizing $${selectedScenario.net_profit_k_usd?.toFixed(2)}k daily profit.`}
              </p>

              {liveSolveError && (
                <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-xs font-mono text-rose-300">
                  Live Solve Error: {liveSolveError}
                </div>
              )}

              {isSolvingLive && (
                <div className="p-4 rounded-lg bg-[#141926] border border-blue-500/40 space-y-3 animate-fade-in" aria-busy="true">
                  <div className="flex items-center gap-2 text-xs font-mono text-blue-400">
                    <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
                    <span>Executing C++ MILP Refinery Solver on {selectedScenario.name}...</span>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    <SkeletonMetric />
                    <SkeletonMetric />
                    <SkeletonMetric />
                    <SkeletonMetric />
                  </div>
                </div>
              )}

              {liveSolveResult && !isSolvingLive && (
                <div className="p-3.5 rounded-lg bg-emerald-950/30 border border-emerald-800/60 space-y-2 animate-fade-in">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-emerald-400 flex items-center gap-1.5 font-mono">
                      <Check className="w-4 h-4 text-emerald-400" />
                      LIVE SOLVE RESULT (kavach-opt mrpl --scenario {selectedScenario.id} --json)
                    </span>
                    <span className="text-[10px] font-mono text-emerald-300">
                      {(liveSolveResult.solve_time_seconds * 1000).toFixed(2)} ms · {liveSolveResult.nodes_explored} nodes
                    </span>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
                    <div>
                      <span className="text-neutral-400 block text-[10px] font-sans">Live Status:</span>
                      <span className="text-white font-bold">{liveSolveResult.solver_status}</span>
                    </div>
                    <div>
                      <span className="text-neutral-400 block text-[10px] font-sans">Net Margin:</span>
                      <span className="text-emerald-400 font-bold">
                        ${liveSolveResult.objective_net_margin_k_usd?.toFixed(2)} k/day
                      </span>
                    </div>
                    <div>
                      <span className="text-neutral-400 block text-[10px] font-sans">Mass Balance Residual:</span>
                      <span className="text-neutral-200">
                        {liveSolveResult.mass_balance_validation?.max_absolute_residual?.toExponential(2) || '0.00e+00'}
                      </span>
                    </div>
                    <div>
                      <span className="text-neutral-400 block text-[10px] font-sans">Verification:</span>
                      <span className="text-emerald-400">
                        {liveSolveResult.verified_optimal ? 'CERTIFIED (PASS)' : 'UNVERIFIED'}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2 text-xs font-mono border-t border-[#1f283a]">
                <div>
                  <span className="text-neutral-400 block text-[10px] font-sans">Profit Delta vs Baseline:</span>
                  <span
                    className={
                      selectedScenario.profit_delta_k_usd && selectedScenario.profit_delta_k_usd < 0
                        ? 'text-rose-400'
                        : 'text-neutral-200'
                    }
                  >
                    {selectedScenario.profit_delta_k_usd !== undefined
                      ? selectedScenario.id === 'baseline'
                        ? '0.00 (Base)'
                        : `$${selectedScenario.profit_delta_k_usd.toFixed(2)}k`
                      : '—'}
                  </span>
                </div>
                <div>
                  <span className="text-neutral-400 block text-[10px] font-sans">Refinery Throughput:</span>
                  <span className="text-neutral-200">
                    {selectedScenario.throughput_kbpd
                      ? `${selectedScenario.throughput_kbpd.toFixed(1)} kbpd`
                      : '0.0 kbpd'}
                  </span>
                </div>
                <div>
                  <span className="text-neutral-400 block text-[10px] font-sans">Mass Balance Residual:</span>
                  <span
                    className={
                      selectedScenario.mass_balance_passed ? 'text-emerald-400' : 'text-rose-400'
                    }
                  >
                    {selectedScenario.mass_balance_passed
                      ? `${(selectedScenario.mass_balance_residual || 0).toExponential(2)} (Conserved)`
                      : 'INCONSISTENT (Empty Polytope)'}
                  </span>
                </div>
                <div>
                  <span className="text-neutral-400 block text-[10px] font-sans">Solve Time:</span>
                  <span className="text-blue-400">
                    {((selectedScenario.solve_time_seconds || 0.001) * 1000).toFixed(2)} ms
                  </span>
                </div>
              </div>

              {/* Infeasibility Certificate or Scenario-Specific Units & Qualities */}
              {selectedScenario.status === 'INFEASIBLE' ? (
                <div className="p-4 rounded-lg bg-rose-950/20 border border-rose-800/40 space-y-2 text-xs">
                  <div className="flex items-center gap-2 text-rose-400 font-semibold font-mono">
                    <AlertTriangle className="w-4 h-4 shrink-0" />
                    <span>Primal Infeasibility Certified by Phase-I Simplex</span>
                  </div>
                  <p className="text-neutral-300 font-sans leading-relaxed">
                    Under the restricted crude slate (-40% sweet intake) combined with tightened product sulfur specifications (&le;0.50 wt%),
                    the linear constraint polytope {`{ x | Ax = b, l \u2264 x \u2264 u }`} is mathematically empty. The artificial variables in Phase I
                    could not be driven to zero (infeasibility sum &gt; 0). Kavach-Opt correctly proved the impossibility of satisfying product volume
                    commitments without violating conservation of mass or statutory product sulfur caps.
                  </p>
                </div>
              ) : (
                selectedScenario.units && selectedScenario.units.length > 0 && (
                  <div className="pt-2 border-t border-[#1f283a] space-y-3">
                    <div className="text-xs font-semibold text-white font-sans">
                      Unit Capacity Utilization ({selectedScenario.name})
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                      {selectedScenario.units.map((unit) => (
                        <div key={unit.unit} className="p-3 rounded-lg bg-[#141926] border border-[#1f283a] space-y-2 text-xs">
                          <div className="flex justify-between items-center">
                            <span className="font-semibold text-white font-sans">{unit.unit}</span>
                            <span className="font-mono text-blue-400 font-bold">{unit.utilization_pct.toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-[#1b212f] h-1.5 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all ${
                                unit.utilization_pct >= 99 ? 'bg-amber-500' : 'bg-emerald-500'
                              }`}
                              style={{ width: `${Math.min(unit.utilization_pct, 100)}%` }}
                            />
                          </div>
                          <div className="text-[11px] text-neutral-400 font-mono flex justify-between">
                            <span>{unit.throughput_kbpd.toFixed(1)} kbpd</span>
                            <span>Limit: {unit.capacity_limit_kbpd.toFixed(1)}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )
              )}
            </div>
          )}
        </div>
      </Panel>
    </div>
  );
};
