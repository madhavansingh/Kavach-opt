/**
 * @file integration.test.mjs
 * @brief Authoritative automated tests for the Kavach-Opt frontend adapter and API.
 *
 * Verifies live solver integration, model discovery, security gates,
 * problem-class execution, verification certificates, and benchmark artifacts.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  discoverModels,
  executeSolve,
  loadBenchmarks,
  loadMRPLData,
  getSystemHealth,
  sanitizeModelPath,
} from '../server/solverAdapter.mjs';
import { handleApiRequest } from '../server/apiRouter.mjs';

describe('Kavach-Opt Frontend Live Integration Suite', () => {
  describe('System Health & Binary Verification', () => {
    it('detects valid Kavach-Opt solver executable and returns version', async () => {
      const health = await getSystemHealth();
      assert.equal(health.status, 'ok');
      assert.equal(health.binary_present, true, 'Kavach-Opt binary must exist on disk');
      assert.ok(health.solver_binary.includes('kavach-opt'));
      assert.ok(
        health.version_info.toLowerCase().includes('kavach'),
        `Expected version info to contain 'kavach', got '${health.version_info}'`
      );
    });

    it('handles GET /api/health returning HTTP 200 and valid JSON', async () => {
      let statusCode = null;
      const headers = {};
      let body = '';
      const req = { method: 'GET', url: '/api/health' };
      const res = {
        set statusCode(code) { statusCode = code; },
        get statusCode() { return statusCode; },
        setHeader(name, val) { headers[name.toLowerCase()] = val; },
        end(data) { if (data) body += data; },
      };
      const handled = await handleApiRequest(req, res);
      assert.equal(handled, true);
      assert.equal(statusCode, 200);
      assert.equal(headers['content-type'], 'application/json');
      assert.equal(headers['cache-control'], 'no-cache, no-store, must-revalidate');
      const parsed = JSON.parse(body);
      assert.equal(parsed.status, 'ok');
      assert.equal(parsed.binary_present, true);
    });

    it('handles HEAD /api/health returning HTTP 200 with headers but no body', async () => {
      let statusCode = null;
      const headers = {};
      let body = '';
      const req = { method: 'HEAD', url: '/api/health' };
      const res = {
        set statusCode(code) { statusCode = code; },
        get statusCode() { return statusCode; },
        setHeader(name, val) { headers[name.toLowerCase()] = val; },
        end(data) { if (data) body += data; },
      };
      const handled = await handleApiRequest(req, res);
      assert.equal(handled, true);
      assert.equal(statusCode, 200);
      assert.equal(headers['content-type'], 'application/json');
      assert.equal(headers['cache-control'], 'no-cache, no-store, must-revalidate');
      assert.equal(body, '', 'HEAD request must return an empty body');
    });
  });

  describe('Model Discovery Subsystem', () => {
    it('discovers all authentic benchmark and case study models', async () => {
      const models = await discoverModels();
      assert.ok(models.length >= 15, `Expected >= 15 models, found ${models.length}`);

      const names = models.map((m) => m.name);
      assert.ok(names.includes('afiro.mps'), 'afiro.mps must be discovered');
      assert.ok(names.includes('knapsack.lp'), 'knapsack.lp must be discovered');
      assert.ok(names.includes('portfolio.lp'), 'portfolio.lp must be discovered');
      assert.ok(names.includes('mrpl_blending_capacity.lp'), 'MRPL model must be discovered');

      // Check problem class classifications
      const afiro = models.find((m) => m.name === 'afiro.mps');
      assert.equal(afiro.problem_class, 'LP');

      const knapsack = models.find((m) => m.name === 'knapsack.lp');
      assert.equal(knapsack.problem_class, 'MILP');

      const portfolio = models.find((m) => m.name === 'portfolio.lp');
      assert.equal(portfolio.problem_class, 'CONVEX_QP');

      const mrpl = models.find((m) => m.name === 'mrpl_blending_capacity.lp');
      assert.equal(mrpl.is_synthetic, true, 'MRPL model must be flagged as synthetic');
    });
  });

  describe('Security & Path Traversal Guards', () => {
    it('rejects paths escaping allowed benchmarks/ and examples/ directories', () => {
      assert.throws(() => sanitizeModelPath('../../CMakeLists.txt'), /Security Error/);
      assert.throws(() => sanitizeModelPath('/etc/passwd'), /Security Error/);
      assert.throws(() => sanitizeModelPath('../src/main.cpp'), /Security Error/);
      assert.throws(() => sanitizeModelPath(''), /Invalid model path/);
    });

    it('accepts safe relative model paths', () => {
      const res = sanitizeModelPath('benchmarks/netlib/afiro.mps');
      assert.equal(res.relativePath, 'benchmarks/netlib/afiro.mps');
      assert.ok(res.absolutePath.endsWith('afiro.mps'));
    });
  });

  describe('Live Solve Execution & Verification', () => {
    it('solves Netlib LP (afiro.mps) to verified optimality', async () => {
      const res = await executeSolve('benchmarks/netlib/afiro.mps', {
        presolve_policy: 'FULL',
        scaling_policy: 'AUTO',
      });

      assert.equal(res.success, true);
      const out = res.raw_output;
      assert.ok(out.result, 'Response must include result block');
      assert.equal(out.result.status, 'OPTIMAL');
      assert.ok(
        Math.abs(out.result.objective - -464.753142857143) < 1e-4,
        `Expected objective approx -464.753, got ${out.result.objective}`
      );
      assert.equal(out.result.verified, true);
      assert.ok(out.verification);
      assert.equal(out.verification.passed, true);
      assert.ok(out.verification.recomputed_primal_residual <= 1e-6);

      // Strategy Layer inspection
      assert.ok(out.strategy);
      assert.ok(out.strategy.rules_fired.length > 0);
      assert.ok(out.strategy.rules_fired.includes('R001') || out.strategy.rules_fired.includes('R003'));
      assert.ok(out.strategy.human_readable_rationale.length > 0);
    });

    it('solves MIPLIB MILP (knapsack.lp) to verified optimality', async () => {
      const res = await executeSolve('benchmarks/miplib/knapsack.lp', {
        backend: 'CPU_SEQUENTIAL',
      });

      assert.equal(res.success, true);
      const out = res.raw_output;
      assert.equal(out.result.status, 'OPTIMAL');
      assert.equal(out.result.objective, 41, `Expected knapsack objective 41, got ${out.result.objective}`);
      assert.equal(out.result.verified, true);
      assert.ok(out.verification.recomputed_integrality_violation <= 1e-5);
    });

    it('solves Convex QP (portfolio.lp) to verified optimality', async () => {
      const res = await executeSolve('benchmarks/qp/portfolio.lp');

      assert.equal(res.success, true);
      const out = res.raw_output;
      assert.equal(out.result.status, 'OPTIMAL');
      assert.equal(out.result.verified, true);
      assert.ok(out.verification.recomputed_kkt_residual <= 1e-6);
    });

    it('correctly reports infeasibility on infeasible_milp.lp without crashing', async () => {
      const res = await executeSolve('benchmarks/miplib/infeasible_milp.lp');

      assert.equal(res.success, true);
      const out = res.raw_output;
      assert.equal(out.result.status, 'INFEASIBLE');
      assert.equal(out.result.verified, false);
    });

    it('executes requested backend and reports hardware telemetry', async () => {
      const res = await executeSolve('benchmarks/netlib/afiro.mps', {
        backend: 'CUDA_GPU',
      });

      assert.equal(res.success, true);
      const out = res.raw_output;
      assert.equal(out.result.status, 'OPTIMAL');
      assert.ok(
        out.result.backend_used === 'CUDA_GPU' ||
        out.result.backend_used === 'CPU_SEQUENTIAL' ||
        out.result.backend_used === 'CPU_MULTICORE',
        `Expected valid backend string, got ${out.result.backend_used}`
      );
    });
  });

  describe('Benchmark Repository Artifacts', () => {
    it('loads authentic benchmark runs from repository JSON files', async () => {
      const benchmarks = await loadBenchmarks();
      assert.ok(benchmarks.length >= 10, `Expected >= 10 benchmarks, found ${benchmarks.length}`);

      for (const b of benchmarks) {
        assert.ok(b.instance_name);
        assert.ok(b.suite);
        assert.ok(typeof b.rows === 'number');
        assert.ok(typeof b.cols === 'number');
        assert.ok(typeof b.solve_time_seconds === 'number');
        assert.ok(
          b.status === 'OPTIMAL' || b.status === 'FEASIBLE' || b.status === 'INFEASIBLE',
          `Expected valid status, got ${b.status}`
        );
      }
    });
  });

  describe('MRPL Refinery Case Study Subsystem', () => {
    it('loads authentic baseline and scenario results with mandatory disclosures', async () => {
      const mrpl = await loadMRPLData();
      assert.ok(mrpl.data_status_disclosure.includes('DATA STATUS'));
      assert.ok(mrpl.data_status_disclosure.includes('illustrative/synthetic'));

      // Baseline checks
      assert.equal(mrpl.baseline.status, 'OPTIMAL');
      assert.ok(mrpl.baseline.net_profit_k_usd > 5000);
      assert.ok(mrpl.baseline.units.length >= 3);
      assert.ok(mrpl.baseline.qualities.length >= 3);

      // Scenarios checks
      assert.ok(mrpl.scenarios.length >= 4);
      const infeasibleScenario = mrpl.scenarios.find((s) => s.id === 'scenario_d_infeasible');
      assert.ok(infeasibleScenario, 'Scenario D must be present');
      assert.equal(infeasibleScenario.status, 'INFEASIBLE');
    });
  });
});
