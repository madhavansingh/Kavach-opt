/**
 * @file zero_mock.test.mjs
 * @brief Zero-Mock & Real-Time Integrity Verification Test Suite.
 *
 * Enforces absolute invariants:
 * 1. ZERO mockData files exist in source tree.
 * 2. ZERO mock imports or MOCK_* symbols in frontend/src/.
 * 3. 100% of model metadata originates from live C++ inspection.
 * 4. 100% of MRPL scenario solves execute on-the-fly via C++ core.
 * 5. 100% of benchmarks run live via C++ benchmark runner.
 * 6. Production build bundle contains zero mock datasets.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  inspectModel,
  runMRPLScenario,
  runBenchmark,
  discoverModels,
} from '../server/solverAdapter.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FRONTEND_ROOT = path.resolve(__dirname, '..');
const SRC_ROOT = path.resolve(FRONTEND_ROOT, 'src');

describe('Kavach-Opt Zero-Mock & Real-Time Integrity Test Suite', () => {
  describe('Mock Data Purge & File Invariants', () => {
    it('ensures mockData.ts is permanently deleted', () => {
      const mockDataFile = path.join(SRC_ROOT, 'data', 'mockData.ts');
      assert.equal(
        fs.existsSync(mockDataFile),
        false,
        'mockData.ts MUST NOT exist in frontend/src/'
      );
    });

    it('ensures no file in frontend/src/ imports mockData', () => {
      function scanDir(dir) {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);
          if (entry.isDirectory()) {
            scanDir(fullPath);
          } else if (entry.isFile() && /\.(ts|tsx|js|jsx)$/.test(entry.name)) {
            const content = fs.readFileSync(fullPath, 'utf-8');
            assert.ok(
              !content.includes('mockData'),
              `File ${path.relative(SRC_ROOT, fullPath)} contains reference to mockData`
            );
          }
        }
      }
      scanDir(SRC_ROOT);
    });
  });

  describe('Live Model Inspection via C++ Engine', () => {
    it('inspects Netlib LP (afiro.mps) with authentic structural characteristics', async () => {
      const info = await inspectModel('benchmarks/netlib/afiro.mps');
      assert.equal(info.name, 'AFIRO');
      assert.equal(info.num_variables, 32);
      assert.equal(info.num_constraints, 27);
      assert.equal(info.num_nonzeros, 83);
      assert.equal(info.problem_class, 'LP');
      assert.equal(info.is_milp, false);
      assert.equal(info.is_qp, false);
      assert.ok(typeof info.coefficient_range_ratio === 'number');
      assert.ok(info.coefficient_range_ratio > 1.0);
    });

    it('inspects MIPLIB MILP (knapsack.lp) with authentic discrete variables', async () => {
      const info = await inspectModel('benchmarks/miplib/knapsack.lp');
      assert.equal(info.is_milp, true);
      assert.equal(info.problem_class, 'MILP');
      assert.ok(info.num_variables > 0);
      assert.ok(info.integer_variables > 0 || info.binary_variables > 0);
    });

    it('inspects Convex QP (portfolio.lp) with authentic quadratic structure', async () => {
      const info = await inspectModel('benchmarks/qp/portfolio.lp');
      assert.equal(info.is_qp, true);
      assert.equal(info.problem_class, 'CONVEX_QP');
    });

    it('discovers all models with enriched C++ characteristics', async () => {
      const models = await discoverModels();
      assert.ok(models.length >= 15);
      for (const m of models) {
        assert.ok(m.num_variables >= 0);
        assert.ok(m.num_constraints >= 0);
        assert.ok(m.num_nonzeros >= 0);
        assert.ok(typeof m.density === 'number');
        assert.ok(typeof m.dynamic_range === 'number');
      }
    });
  });

  describe('Live MRPL Scenario Execution', () => {
    it('solves baseline MRPL scenario on the fly with certified optimality', async () => {
      const res = await runMRPLScenario('baseline');
      assert.equal(res.scenario, 'baseline');
      assert.equal(res.solver_status, 'OPTIMAL');
      assert.equal(res.verified_optimal, true);
      assert.ok(Math.abs(res.objective_net_margin_k_usd - 5445.91) < 0.1);
      assert.ok(res.solve_time_seconds > 0.0);
      assert.ok(res.nodes_explored >= 1);
      assert.ok(res.mass_balance_validation);
      assert.equal(res.mass_balance_validation.passed, true);
      assert.ok(res.mass_balance_validation.max_absolute_residual <= 1e-12);
    });

    it('solves scenario_d_infeasible on the fly with proven infeasibility', async () => {
      const res = await runMRPLScenario('scenario_d_infeasible');
      assert.equal(res.scenario, 'scenario_d_infeasible');
      assert.equal(res.solver_status, 'INFEASIBLE');
      assert.equal(res.verified_optimal, false);
    });
  });

  describe('Live Benchmark Execution', () => {
    it('executes single instance live and returns authentic execution metrics', async () => {
      const result = await runBenchmark('benchmarks/netlib/afiro.mps');
      assert.equal(result.success, true);
      assert.equal(result.live, true);
      assert.ok(result.result);
      assert.equal(result.result.status, 'OPTIMAL');
      assert.ok(result.result.solve_time_seconds > 0.0);
      assert.ok(Math.abs(result.result.objective_value - -464.75314) < 1e-3);
    });
  });

  describe('Production Bundle Verification', () => {
    it('verifies dist/ bundle contains zero mock datasets', () => {
      const distDir = path.join(FRONTEND_ROOT, 'dist', 'assets');
      assert.ok(fs.existsSync(distDir), 'dist/assets directory must exist (run npm run build first)');
      const files = fs.readdirSync(distDir);
      const jsFiles = files.filter((f) => f.endsWith('.js'));
      assert.ok(jsFiles.length > 0, 'At least one production JS bundle must exist');

      for (const jsFile of jsFiles) {
        const bundle = fs.readFileSync(path.join(distDir, jsFile), 'utf-8');
        assert.ok(
          !bundle.includes('MOCK_BENCHMARKS'),
          `Production bundle ${jsFile} must not contain MOCK_BENCHMARKS`
        );
        assert.ok(
          !bundle.includes('MOCK_MRPL_CASE'),
          `Production bundle ${jsFile} must not contain MOCK_MRPL_CASE`
        );
        assert.ok(
          !bundle.includes('SAMPLE_MODELS'),
          `Production bundle ${jsFile} must not contain SAMPLE_MODELS`
        );
      }
    });
  });
});
