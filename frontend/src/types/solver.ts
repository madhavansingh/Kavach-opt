/**
 * @file solver.ts
 * @brief Core domain types for Kavach-Opt frontend.
 * Strictly reflects the C++ API, CLI JSON output, and verification structures.
 */

export type ProblemClass = 'LP' | 'MILP' | 'CONVEX_QP';

export type SolverStatus =
  | 'OPTIMAL'
  | 'FEASIBLE'
  | 'INFEASIBLE'
  | 'UNBOUNDED'
  | 'LIMIT_REACHED'
  | 'NUMERICAL_ERROR'
  | 'ERROR'
  | 'UNKNOWN';

export type VerificationStatus = 'CERTIFIED' | 'REJECTED' | 'UNVERIFIED' | 'WAITING';

export type BackendType = 'CPU_SEQUENTIAL' | 'CPU_MULTICORE' | 'CUDA_GPU';

export type PresolvePolicy = 'FULL' | 'NONE' | 'AUTO';

export type ScalingPolicy = 'AUTO' | 'ENABLED' | 'DISABLED';

export type NumericalRiskProfile = 'NORMAL' | 'ELEVATED' | 'HIGH';

export interface VerificationCheckItem {
  name: string;
  category: 'primal' | 'dual' | 'slackness' | 'integrality' | 'kkt' | 'objective';
  passed: boolean;
  residual: number;
  tolerance: number;
  description: string;
}

export interface VerificationReport {
  passed: boolean;
  certified_status: SolverStatus;
  recomputed_primal_residual: number;
  recomputed_dual_residual: number;
  recomputed_complementary_slackness: number;
  recomputed_integrality_violation?: number;
  recomputed_kkt_residual?: number;
  recomputed_objective: number;
  summary: string;
  checks?: VerificationCheckItem[];
}

export interface StrategyDecision {
  strategy_version: string;
  config_version: string;
  problem_class: ProblemClass;
  presolve_policy: PresolvePolicy;
  scaling_policy: ScalingPolicy;
  scaling_enabled: boolean;
  numerical_risk_profile: NumericalRiskProfile;
  recommended_backend: BackendType;
  recommended_cpu_threads: number;
  gpu_eligible_for_batched_spmv: boolean;
  gpu_routing_reason: string;
  rules_fired: string[];
  rule_explanations: string[];
  human_readable_rationale: string;
  decision_time_ms: number;
  total_strategy_time_ms: number;
}

export interface ModelMetadata {
  name: string;
  path?: string;
  format: 'MPS' | 'LP';
  num_variables: number;
  num_constraints: number;
  num_nonzeros: number;
  density: number;
  num_discrete_vars: number;
  problem_class: ProblemClass;
  dynamic_range: number;
  is_synthetic?: boolean;
}

export interface SolveResult {
  status: SolverStatus;
  objective_value: number;
  solve_time_seconds: number;
  iterations: number;
  nodes: number;
  optimality_gap?: number;
  backend_used: BackendType;
  verified: boolean;
  verification_report: VerificationReport;
  primal_solution?: Record<string, number>;
  dual_solution?: Record<string, number>;
  strategy_decision?: StrategyDecision;
}

export interface SolverSession {
  id: string;
  timestamp: string;
  model: ModelMetadata;
  status: 'IDLE' | 'SOLVING' | 'VERIFYING' | 'COMPLETED' | 'FAILED';
  result?: SolveResult;
  error?: string;
}

export interface BenchmarkItem {
  instance_name: string;
  suite: 'netlib' | 'miplib' | 'qp';
  rows: number;
  cols: number;
  nonzeros: number;
  problem_type: ProblemClass;
  status: SolverStatus;
  objective_value: number;
  solve_time_seconds: number;
  iterations: number;
  verification_passed: boolean;
  verified_status: string;
}

export interface MRPLProductQuality {
  product: string;
  production_kbpd: number;
  calculated_sulfur_wt_pct: number;
  sulfur_limit_wt_pct: number;
  sulfur_margin_wt_pct: number;
  calculated_density_kg_m3: number;
  density_limit_kg_m3: number;
  density_margin_kg_m3: number;
  quality_satisfied: boolean;
}

export interface MRPLUnitDecision {
  unit: string;
  is_active: boolean;
  throughput_kbpd: number;
  capacity_limit_kbpd: number;
  utilization_pct: number;
  summary: string;
}

export interface MRPLEconomics {
  gross_revenue_k_usd: number;
  crude_purchase_cost_k_usd: number;
  unit_variable_cost_k_usd: number;
  unit_fixed_cost_k_usd: number;
  total_cost_k_usd: number;
  net_profit_k_usd: number;
}

export interface MRPLScenario {
  id: string;
  name: string;
  description: string;
  status: SolverStatus;
  net_profit_k_usd?: number;
  profit_delta_k_usd?: number;
  throughput_kbpd?: number;
  mass_balance_passed: boolean;
}

export interface MRPLCaseData {
  data_status_disclosure: string;
  baseline: {
    model_name: string;
    num_vars: number;
    num_constrs: number;
    num_nonzeros: number;
    status: SolverStatus;
    net_profit_k_usd: number;
    lp_relaxation_objective: number;
    optimality_gap: number;
    nodes_explored: number;
    solve_time_seconds: number;
    mass_balance_max_residual: number;
    mass_balance_passed: boolean;
    economics: MRPLEconomics;
    qualities: MRPLProductQuality[];
    units: MRPLUnitDecision[];
  };
  scenarios: MRPLScenario[];
}

export interface SolveConfig {
  problem_class: ProblemClass;
  presolve_policy: PresolvePolicy;
  scaling_policy: ScalingPolicy;
  backend: BackendType;
  cpu_threads: number;
  primal_feasibility_tolerance: number;
  dual_feasibility_tolerance: number;
  optimality_tolerance: number;
  time_limit_seconds: number;
  max_iterations: number;
  max_nodes: number;
  enable_independent_verification: boolean;
}

