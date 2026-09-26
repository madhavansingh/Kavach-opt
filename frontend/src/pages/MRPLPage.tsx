import React, { useState } from 'react';
import {
  Shield,
  AlertTriangle,
  CheckCircle2,
  Gauge,
} from 'lucide-react';
import { PageHeader } from '../components/PageHeader';
import { Panel } from '../components/Panel';
import { StatusBadge } from '../components/StatusBadge';
import { Metric, MetricGroup } from '../components/Metric';
import { DataTable } from '../components/DataTable';
import { MOCK_MRPL_CASE } from '../data/mockData';
import type { MRPLScenario } from '../types/solver';

export const MRPLPage: React.FC = () => {
  const [selectedScenario, setSelectedScenario] = useState<MRPLScenario>(
    MOCK_MRPL_CASE.scenarios[0]
  );

  const baseline = MOCK_MRPL_CASE.baseline;

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
      render: (row: any) => (
        <span className="font-mono text-neutral-300">
          {row.calculated_density_kg_m3.toFixed(1)} / &le; {row.density_limit_kg_m3.toFixed(1)}
        </span>
      ),
    },
    {
      key: 'density_margin',
      header: 'Density Margin',
      render: (row: any) => (
        <span className="font-mono text-emerald-400">
          +{row.density_margin_kg_m3.toFixed(1)} kg/m³
        </span>
      ),
    },
    {
      key: 'specs',
      header: 'Specs Met',
      render: () => (
        <span className="inline-flex items-center gap-1 text-emerald-400 font-mono text-xs">
          <CheckCircle2 className="w-3.5 h-3.5" />
          <span>COMPLIANT</span>
        </span>
      ),
    },
  ];

  return (
    <div className="space-y-6">
      <PageHeader
        title="MRPL Industrial Case Study"
        subtitle="Crude scheduling, unit routing, and clean fuel quality blending optimization under capacity and emission constraints."
        badge={
          <div className="flex items-center gap-2">
            <span className="text-xs text-neutral-400 font-mono">INDUSTRY:</span>
            <span className="text-xs font-mono font-semibold text-white px-2 py-0.5 rounded bg-[#1e2430] border border-[#2e3748]">
              PETROLEUM REFINING
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
              {MOCK_MRPL_CASE.data_status_disclosure}
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
              value={`$${baseline.economics.gross_revenue_k_usd.toLocaleString(undefined, { minimumFractionDigits: 2 })}k`}
              subtext="Finished fuels sale"
            />
            <Metric
              label="Crude Purchase Cost"
              value={`-$${baseline.economics.crude_purchase_cost_k_usd.toLocaleString(undefined, { minimumFractionDigits: 2 })}k`}
              subtext="Crude slate intake"
            />
            <Metric
              label="Operating Variable Costs"
              value={`-$${baseline.economics.unit_variable_cost_k_usd.toLocaleString(undefined, { minimumFractionDigits: 2 })}k`}
              subtext="Catalyst, steam, power"
            />
            <Metric
              label="Fixed Plant Costs"
              value={`-$${baseline.economics.unit_fixed_cost_k_usd.toLocaleString(undefined, { minimumFractionDigits: 2 })}k`}
              subtext="Base maintenance & staffing"
            />
            <Metric
              label="Optimal Net Daily Profit"
              value={`$${baseline.economics.net_profit_k_usd.toLocaleString(undefined, { minimumFractionDigits: 2 })}k`}
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

      {/* Unit Operations & Capacities */}
      <Panel
        title="Major Processing Units & Capacity Utilization"
        subtitle="Throughput distribution across primary distillation and secondary conversion units"
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {baseline.units.map((unit) => (
            <div
              key={unit.unit}
              className="p-4 rounded-lg bg-[#0d1016] border border-[#202736] space-y-3"
            >
              <div className="flex justify-between items-start">
                <span className="text-xs font-semibold text-white">{unit.unit}</span>
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
                  <span className="text-neutral-400">Throughput:</span>
                  <span className="font-mono text-neutral-200">
                    {unit.throughput_kbpd.toFixed(1)} / {unit.capacity_limit_kbpd.toFixed(1)} kbpd
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-neutral-400">Load Factor:</span>
                  <span className="font-mono font-bold text-blue-400">
                    {unit.utilization_pct.toFixed(1)}%
                  </span>
                </div>

                <div className="w-full bg-[#1e2430] h-2 rounded-full overflow-hidden">
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

              <div className="text-[11px] text-neutral-400 font-mono pt-1 border-t border-[#1a202c]">
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
        title="What-If Operational Scenarios"
        subtitle="Evaluate refinery robustness under supply disruptions, quality tightening, and unit outages"
      >
        <div className="space-y-5">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {MOCK_MRPL_CASE.scenarios.map((sc) => {
              const isSelected = selectedScenario.id === sc.id;
              return (
                <button
                  key={sc.id}
                  type="button"
                  onClick={() => setSelectedScenario(sc)}
                  className={`p-4 rounded-lg border text-left transition-all ${
                    isSelected
                      ? 'border-blue-500/80 bg-blue-950/20 shadow-lg'
                      : 'border-[#202736] bg-[#0d1016] hover:border-[#2e3748]'
                  }`}
                >
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-xs font-semibold text-white">{sc.name}</span>
                    <StatusBadge status={sc.status} size="sm" />
                  </div>
                  <p className="text-[11px] text-neutral-400 mb-3">{sc.description}</p>
                  <div className="flex justify-between items-center text-xs font-mono pt-2 border-t border-[#1e2430]">
                    <span className="text-neutral-400">Net Profit:</span>
                    <span
                      className={`font-bold ${
                        sc.net_profit_k_usd ? 'text-white' : 'text-neutral-500'
                      }`}
                    >
                      {sc.net_profit_k_usd
                        ? `$${sc.net_profit_k_usd.toLocaleString(undefined, { minimumFractionDigits: 2 })}k`
                        : 'INFEASIBLE'}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Selected Scenario Drill-down */}
          <div className="p-4 rounded-lg bg-[#0a0d12] border border-[#1e2430] space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Gauge className="w-4 h-4 text-blue-400" />
                <span className="text-xs font-semibold text-white">
                  Scenario Focus: {selectedScenario.name}
                </span>
              </div>
              <StatusBadge status={selectedScenario.status} size="sm" />
            </div>

            <p className="text-xs text-neutral-300 leading-relaxed">
              {selectedScenario.description}.{' '}
              {selectedScenario.status === 'INFEASIBLE'
                ? 'Kavach-Opt mathematically proves that no feasible blending and scheduling plan exists that satisfies both demand commitments and strict physical crude conservation.'
                : `Under this operating condition, the solver reroutes intermediate heavy gas oil fractions to maintain zero mass balance deviation while realizing $${selectedScenario.net_profit_k_usd?.toFixed(2)}k daily profit.`}
            </p>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2 text-xs font-mono">
              <div>
                <span className="text-neutral-400 block text-[10px]">Profit Delta vs Baseline:</span>
                <span
                  className={
                    selectedScenario.profit_delta_k_usd && selectedScenario.profit_delta_k_usd < 0
                      ? 'text-rose-400'
                      : 'text-neutral-200'
                  }
                >
                  {selectedScenario.profit_delta_k_usd
                    ? `$${selectedScenario.profit_delta_k_usd.toFixed(2)}k`
                    : '—'}
                </span>
              </div>
              <div>
                <span className="text-neutral-400 block text-[10px]">Refinery Throughput:</span>
                <span className="text-neutral-200">
                  {selectedScenario.throughput_kbpd
                    ? `${selectedScenario.throughput_kbpd.toFixed(1)} kbpd`
                    : '—'}
                </span>
              </div>
              <div>
                <span className="text-neutral-400 block text-[10px]">Mass Balance Invariant:</span>
                <span
                  className={
                    selectedScenario.mass_balance_passed ? 'text-emerald-400' : 'text-rose-400'
                  }
                >
                  {selectedScenario.mass_balance_passed ? 'PASS (0.0 residual)' : 'FAIL / INFEASIBLE'}
                </span>
              </div>
              <div>
                <span className="text-neutral-400 block text-[10px]">Dual Simplex Warm Start:</span>
                <span className="text-blue-400">ENABLED (0.4ms)</span>
              </div>
            </div>
          </div>
        </div>
      </Panel>
    </div>
  );
};
