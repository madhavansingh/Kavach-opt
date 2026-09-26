import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '../..');
const frontendSrc = path.resolve(projectRoot, 'frontend/src');

describe('Kavach-Opt Real-Time UX & Interaction Invariants', () => {
  it('ensures all skeleton primitives exist and are properly exported', () => {
    const skeletonPath = path.resolve(frontendSrc, 'components/Skeleton.tsx');
    assert.ok(fs.existsSync(skeletonPath), 'Skeleton.tsx must exist');

    const content = fs.readFileSync(skeletonPath, 'utf8');
    assert.ok(content.includes('export const Skeleton'), 'Skeleton must be exported');
    assert.ok(content.includes('export const SkeletonText'), 'SkeletonText must be exported');
    assert.ok(content.includes('export const SkeletonMetric'), 'SkeletonMetric must be exported');
    assert.ok(content.includes('export const SkeletonCard'), 'SkeletonCard must be exported');
    assert.ok(content.includes('export const SkeletonTableRow'), 'SkeletonTableRow must be exported');
    assert.ok(content.includes('export const SkeletonChart'), 'SkeletonChart must be exported');
    assert.ok(content.includes('export const SkeletonPanel'), 'SkeletonPanel must be exported');
  });

  it('ensures SolverExecutionPanel exists with live state reporting', () => {
    const execPanelPath = path.resolve(frontendSrc, 'components/SolverExecutionPanel.tsx');
    assert.ok(fs.existsSync(execPanelPath), 'SolverExecutionPanel.tsx must exist');

    const content = fs.readFileSync(execPanelPath, 'utf8');
    assert.ok(content.includes('export const SolverExecutionPanel'), 'SolverExecutionPanel must be exported');
    assert.ok(content.includes('elapsedMs'), 'SolverExecutionPanel must measure real elapsed client time');
    assert.ok(!content.includes('0% → 20%'), 'Must not contain fake percentage milestones');
  });

  it('ensures Toast notification system exists and conforms to spec', () => {
    const toastContextPath = path.resolve(frontendSrc, 'context/ToastContext.tsx');
    const toastDefPath = path.resolve(frontendSrc, 'context/toastDef.ts');
    const useToastPath = path.resolve(frontendSrc, 'context/useToast.ts');

    assert.ok(fs.existsSync(toastContextPath), 'ToastContext.tsx must exist');
    assert.ok(fs.existsSync(toastDefPath), 'toastDef.ts must exist');
    assert.ok(fs.existsSync(useToastPath), 'useToast.ts must exist');

    const contextContent = fs.readFileSync(toastContextPath, 'utf8');
    assert.ok(contextContent.includes('export const ToastProvider'), 'ToastProvider must be exported');
  });

  it('ensures reduced-motion media query is enforced in CSS', () => {
    const cssPath = path.resolve(frontendSrc, 'index.css');
    assert.ok(fs.existsSync(cssPath), 'index.css must exist');

    const cssContent = fs.readFileSync(cssPath, 'utf8');
    assert.ok(cssContent.includes('prefers-reduced-motion'), 'Must include prefers-reduced-motion query');
    assert.ok(cssContent.includes('animate-fade-in'), 'Must define animate-fade-in');
    assert.ok(cssContent.includes('animate-indeterminate'), 'Must define animate-indeterminate');
  });

  it('ensures zero-mock invariant across all source files', () => {
    const files = fs.readdirSync(frontendSrc, { recursive: true })
      .filter((f) => typeof f === 'string' && (f.endsWith('.tsx') || f.endsWith('.ts')));

    for (const f of files) {
      const fullPath = path.resolve(frontendSrc, f);
      const content = fs.readFileSync(fullPath, 'utf8');
      assert.ok(!content.includes('mockData'), `File ${f} must not import or reference mockData`);
      assert.ok(!content.includes('PrototypeSolverBridge'), `File ${f} must not reference PrototypeSolverBridge`);
    }
  });

  it('ensures SolvePage implements the multi-state button and double-click protection', () => {
    const solvePagePath = path.resolve(frontendSrc, 'pages/SolvePage.tsx');
    const content = fs.readFileSync(solvePagePath, 'utf8');

    assert.ok(content.includes('SUBMITTING'), 'SolvePage must support SUBMITTING state');
    assert.ok(content.includes('RUNNING'), 'SolvePage must support RUNNING state');
    assert.ok(content.includes('COMPLETED'), 'SolvePage must support COMPLETED state');
    assert.ok(content.includes('FAILED'), 'SolvePage must support FAILED state');
    assert.ok(content.includes('requestAnimationFrame'), 'SolvePage must use requestAnimationFrame for immediate paint');
    assert.ok(content.includes('disabled={isSolving'), 'SolvePage button must be disabled during active solve');
  });
});
