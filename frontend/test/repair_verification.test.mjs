import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const SRC_DIR = path.resolve(__dirname, '../src');

test('Kavach-Opt Master Frontend Repair Validation', async (t) => {
  await t.test('Authoritative Version Consistency', () => {
    const constantsFile = fs.readFileSync(path.join(SRC_DIR, 'lib/constants.ts'), 'utf-8');
    assert.match(constantsFile, /PRODUCT_VERSION:\s*'v0\.1\.0'/);
    assert.match(constantsFile, /BUILD_TYPE:\s*'Release'/);

    // Verify no RC-1 or RC1 in TopBar, Sidebar, or OnboardingModal
    const topBar = fs.readFileSync(path.join(SRC_DIR, 'layouts/TopBar.tsx'), 'utf-8');
    const sidebar = fs.readFileSync(path.join(SRC_DIR, 'layouts/Sidebar.tsx'), 'utf-8');
    const onboarding = fs.readFileSync(path.join(SRC_DIR, 'components/OnboardingModal.tsx'), 'utf-8');

    assert.doesNotMatch(topBar, /v0\.1\.0-RC1/);
    assert.doesNotMatch(sidebar, />\s*RC-1\s*</);
    assert.doesNotMatch(onboarding, /'v0\.1\.0-RC1'/);
  });

  await t.test('Toast Lifecycle Deduplication Support', () => {
    const toastContext = fs.readFileSync(path.join(SRC_DIR, 'context/ToastContext.tsx'), 'utf-8');
    assert.match(toastContext, /lifecycleKey/);
    assert.match(toastContext, /dismissByLifecycleKey/);
    assert.match(toastContext, /prev\.filter\(\(t\)\s*=>\s*t\.lifecycleKey\s*!==\s*lifecycleKey\)/);

    const appTsx = fs.readFileSync(path.join(SRC_DIR, 'App.tsx'), 'utf-8');
    assert.match(appTsx, /solve_lifecycle/);
    assert.match(appTsx, /verify_lifecycle/);
  });

  await t.test('TopBar Header Active Run Cleanup', () => {
    const topBar = fs.readFileSync(path.join(SRC_DIR, 'layouts/TopBar.tsx'), 'utf-8');
    assert.match(topBar, /Last Run:/);
    assert.match(topBar, /Solving:/);
    assert.match(topBar, /isSolving\s*\?\s*'Solving:'\s*:\s*hasCompletedRun\s*\?\s*'Last Run:'\s*:\s*'Selected Model:'/);
  });

  await t.test('MRPL Refinery Consistency & Lower-Bound Density', () => {
    const mrplPage = fs.readFileSync(path.join(SRC_DIR, 'pages/MRPLPage.tsx'), 'utf-8');

    // Must not contain the hardcoded 280.0 kbpd stage text
    assert.doesNotMatch(mrplPage, /Total Intake:\s*280\.0\s*kbpd/);
    assert.doesNotMatch(mrplPage, /Total Output:\s*280\.0\s*kbpd/);

    // Must contain 328.2 kbpd authentic intake and output
    assert.match(mrplPage, /Total Feedstock:[\s\S]*?328\.2 kbpd/);
    assert.match(mrplPage, /Total Finished:[\s\S]*?328\.2 kbpd/);

    // Must handle P3 min density spec with >=
    assert.match(mrplPage, /row\.product\.includes\('P3'\)/);
    assert.match(mrplPage, /isMinBound\s*\?\s*'≥'\s*:\s*'≤'/);
    assert.match(mrplPage, /above min/);

    // Must not mention '10 ppm'
    assert.doesNotMatch(mrplPage, /10 ppm/);
  });

  await t.test('Benchmark Speedup & Metric Consistency', () => {
    const benchmarksPage = fs.readFileSync(path.join(SRC_DIR, 'pages/BenchmarksPage.tsx'), 'utf-8');

    // Badge must match the 8 workers card (3.13x)
    assert.match(benchmarksPage, /3\.13x MAX SPEEDUP \(8 WORKERS\)/);
    assert.doesNotMatch(benchmarksPage, /3\.04x MAX SPEEDUP/);

    // Aggregates must be calculated dynamically
    assert.match(benchmarksPage, /optimalityRate\s*=\s*totalCount\s*>\s*0/);
    assert.match(benchmarksPage, /verificationRate\s*=\s*totalCount\s*>\s*0/);
    assert.match(benchmarksPage, /medianRuntimeMs/);
  });

  await t.test('Verification Measurement Source & Tolerances', () => {
    const verPage = fs.readFileSync(path.join(SRC_DIR, 'pages/VerificationPage.tsx'), 'utf-8');

    assert.match(verPage, /DEFAULT_TOLERANCES/);
    assert.match(verPage, /Measurement Source/);
    assert.match(verPage, /Independent Verifier \(Recomputed SpMV\)/);
    assert.match(verPage, /Independent Verifier Primal Residual/);
  });

  await t.test('System Page Rigor & Kernel Registry', () => {
    const sysPage = fs.readFileSync(path.join(SRC_DIR, 'pages/SystemPage.tsx'), 'utf-8');

    // Unsupported claims removed
    assert.doesNotMatch(sysPage, /Gershgorin PSD verification/);
    assert.doesNotMatch(sysPage, /Subgradient bound pruning/);

    // Authentic implementations referenced
    assert.match(sysPage, /Cholesky curvature PSD verification/);
    assert.match(sysPage, /Continuous QP relaxation bound pruning/);
    assert.match(sysPage, /CUDA_KERNEL_REGISTRY/);
  });
});
