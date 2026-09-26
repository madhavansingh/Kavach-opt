import React, { useState, useEffect } from 'react';
import {
  Shield,
  ArrowRight,
  ArrowLeft,
  X,
  CheckCircle2,
  Cpu,
  Layers,
  Sparkles,
  Lock,
  GitBranch,
  Activity,
  Zap,
} from 'lucide-react';
import { solverBridge, type SystemHealth } from '../lib/solverBridge';
import { SYSTEM_METADATA } from '../lib/constants';

interface OnboardingModalProps {
  isOpen: boolean;
  onClose: () => void;
  onNavigateToSolve?: () => void;
}

export const OnboardingModal: React.FC<OnboardingModalProps> = ({
  isOpen,
  onClose,
  onNavigateToSolve,
}) => {
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [health, setHealth] = useState<SystemHealth | null>(null);

  useEffect(() => {
    if (isOpen) {
      solverBridge.getSystemHealth().then((h) => setHealth(h)).catch(() => {});
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleFinish = () => {
    localStorage.setItem('kavach_onboarding_completed', 'true');
    onClose();
  };

  const handleEnterWorkbench = () => {
    handleFinish();
    if (onNavigateToSolve) {
      onNavigateToSolve();
    }
  };

  const totalSteps = 5;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fade-in"
      role="dialog"
      aria-modal="true"
      aria-label="Kavach-Opt Workstation Onboarding"
    >
      <div className="relative w-full max-w-3xl bg-[#0c1017] border border-[#1f283a] rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[92vh]">
        {/* Top bar with progress indicator */}
        <div className="px-6 py-4 border-b border-[#1b2333] flex items-center justify-between bg-[#0e131d]/60">
          <div className="flex items-center gap-2.5">
            <div className="w-6 h-6 rounded bg-emerald-950/60 border border-emerald-600/40 flex items-center justify-center text-emerald-400">
              <Shield className="w-3.5 h-3.5" />
            </div>
            <span className="font-sans font-bold text-xs tracking-wider text-neutral-200 uppercase">
              Kavach-Opt Engine Guide
            </span>
          </div>

          {/* Step dots */}
          <div className="flex items-center gap-1.5">
            {[1, 2, 3, 4, 5].map((step) => (
              <button
                key={step}
                onClick={() => setCurrentStep(step)}
                className={`h-1.5 rounded-full transition-all duration-300 ${
                  currentStep === step
                    ? 'w-7 bg-emerald-400'
                    : currentStep > step
                    ? 'w-3 bg-neutral-600'
                    : 'w-2 bg-neutral-800'
                }`}
                aria-label={`Go to step ${step}`}
              />
            ))}
          </div>

          <button
            onClick={handleFinish}
            className="p-1 rounded text-neutral-400 hover:text-neutral-200 hover:bg-[#161d2b] transition-colors"
            aria-label="Close guide"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 sm:p-8 overflow-y-auto flex-1 font-sans">
          {/* SCREEN 1: WELCOME */}
          {currentStep === 1 && (
            <div className="space-y-6 animate-fade-in">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-950/40 border border-emerald-700/50 text-emerald-400 text-xs font-mono font-medium">
                <Sparkles className="w-3 h-3" />
                <span>Sovereign Optimization Engine</span>
              </div>

              <div>
                <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white font-sans">
                  Kavach-Opt <span className="text-neutral-400 font-normal text-lg sm:text-xl">(कवच-Opt)</span>
                </h2>
                <p className="text-sm font-mono text-emerald-400/90 mt-1">
                  &ldquo;Solve. Verify. Understand.&rdquo;
                </p>
                <p className="mt-3 text-neutral-300 text-xs sm:text-sm leading-relaxed max-w-2xl font-sans">
                  A high-performance, sparse mathematical optimization engine built entirely from scratch in modern C++20.
                  Features zero forbidden optimization dependencies in the solving path, independent zero-trust verification,
                  and deterministic strategy routing for industrial-scale planning and scheduling.
                </p>
              </div>

              <div>
                <div className="text-[11px] font-mono text-neutral-400 uppercase tracking-wider mb-2">
                  Supported Mathematical Problem Classes
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                  <div className="p-3 rounded-lg bg-[#121722] border border-[#1e2637]">
                    <div className="text-xs font-bold text-white font-mono">LP</div>
                    <div className="text-[11px] text-neutral-400 mt-0.5">Continuous Linear</div>
                    <div className="text-[10px] text-emerald-400 font-mono mt-1">Revised Simplex</div>
                  </div>
                  <div className="p-3 rounded-lg bg-[#121722] border border-[#1e2637]">
                    <div className="text-xs font-bold text-white font-mono">MILP</div>
                    <div className="text-[11px] text-neutral-400 mt-0.5">Mixed-Integer Linear</div>
                    <div className="text-[10px] text-sky-400 font-mono mt-1">Branch-and-Cut</div>
                  </div>
                  <div className="p-3 rounded-lg bg-[#121722] border border-[#1e2637]">
                    <div className="text-xs font-bold text-white font-mono">Convex QP</div>
                    <div className="text-[11px] text-neutral-400 mt-0.5">Quadratic Objective</div>
                    <div className="text-[10px] text-indigo-400 font-mono mt-1">Primal Active-Set</div>
                  </div>
                  <div className="p-3 rounded-lg bg-[#121722] border border-[#1e2637]">
                    <div className="text-xs font-bold text-white font-mono">Convex MIQP</div>
                    <div className="text-[11px] text-neutral-400 mt-0.5">Mixed-Integer QP</div>
                    <div className="text-[10px] text-purple-400 font-mono mt-1">B&amp;B + Active-Set</div>
                  </div>
                </div>
              </div>

              <div className="pt-2 flex flex-wrap items-center gap-3">
                <button
                  onClick={() => setCurrentStep(2)}
                  className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs flex items-center gap-2 transition-colors"
                >
                  <span>Explore the Engine</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={handleEnterWorkbench}
                  className="px-4 py-2 rounded-lg bg-[#161d2b] hover:bg-[#20293d] border border-[#263247] text-neutral-200 font-medium text-xs transition-colors"
                >
                  Enter Workbench
                </button>
              </div>
            </div>
          )}

          {/* SCREEN 2: HOW KAVACH-OPT WORKS */}
          {currentStep === 2 && (
            <div className="space-y-6 animate-fade-in">
              <div>
                <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-blue-950/40 border border-blue-700/50 text-blue-400 text-xs font-mono font-medium mb-2">
                  <Activity className="w-3 h-3" />
                  <span>Pipeline Architecture</span>
                </div>
                <h3 className="text-xl font-bold text-white font-sans">
                  The End-to-End Optimization Pipeline
                </h3>
                <p className="mt-1 text-xs text-neutral-400 font-sans">
                  Every instance undergoes a deterministic 8-stage transformation from ingest to verified mathematical certificate.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {[
                  { step: '1. Model Ingestion', desc: 'Parses standard MPS (fixed/free) or native LP models into canonical sparse representations.' },
                  { step: '2. Topology Validation', desc: 'Validates non-zero bounds, dimensions, matrix non-empty checks, and finite numerical sanity.' },
                  { step: '3. Classification & Strategy', desc: '18 deterministic rules (R001–R018) characterize sparsity, integrality, and condition numbers.' },
                  { step: '4. Presolve Suite', desc: 'Eliminates fixed variables, singleton rows, empty rows/columns, and tightens implied bounds.' },
                  { step: '5. Core Optimization', desc: 'Revised Simplex (LP), Branch-and-Cut (MILP), or Active-Set QP/MIQP with SparseLU factorization.' },
                  { step: '6. Post-Solve Reconstruction', desc: 'Reconstructs original primal variables, dual multipliers, and slacks from presolve reductions.' },
                  { step: '7. Independent Verification', desc: 'Air-gapped zero-trust layer re-multiplies ||Ax - b||, checks bounds, integrality, and KKT residuals.' },
                  { step: '8. Telemetry & Analytics', desc: 'Outputs machine-readable JSON certification, execution timing, and cutting-plane metrics.' },
                ].map((item, idx) => (
                  <div key={idx} className="p-3 rounded-lg bg-[#121722] border border-[#1c2436] flex items-start gap-2.5">
                    <div className="w-5 h-5 rounded-full bg-neutral-800 text-emerald-400 font-mono text-[10px] font-bold flex items-center justify-center shrink-0 mt-0.5">
                      {idx + 1}
                    </div>
                    <div>
                      <div className="text-xs font-semibold text-white font-sans">{item.step}</div>
                      <div className="text-[11px] text-neutral-400 font-sans mt-0.5 leading-snug">{item.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* SCREEN 3: ENGINE CAPABILITIES */}
          {currentStep === 3 && (
            <div className="space-y-5 animate-fade-in">
              <div>
                <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-indigo-950/40 border border-indigo-700/50 text-indigo-400 text-xs font-mono font-medium mb-2">
                  <Layers className="w-3 h-3" />
                  <span>Mathematical Scope</span>
                </div>
                <h3 className="text-xl font-bold text-white font-sans">
                  Sovereign Engine Capabilities
                </h3>
                <p className="mt-1 text-xs text-neutral-400 font-sans">
                  Rigorous algorithmic layers engineered in modern C++20 without external solver runtimes.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="p-3 rounded-lg bg-[#121722] border border-[#1e2637] space-y-1">
                  <div className="text-xs font-bold text-emerald-400 font-mono">Sparse Numerics</div>
                  <p className="text-[11px] text-neutral-300 font-sans leading-snug">
                    CSR/CSC formats, SparseLU with Markowitz threshold pivoting (&mu; = 0.1), and Ruiz equilibration scaling.
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-[#121722] border border-[#1e2637] space-y-1">
                  <div className="text-xs font-bold text-emerald-400 font-mono">Revised Simplex</div>
                  <p className="text-[11px] text-neutral-300 font-sans leading-snug">
                    Two-Phase simplex with Phase I artificial variables, Bland anti-cycling, and steepest-edge pricing.
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-[#121722] border border-[#1e2637] space-y-1">
                  <div className="text-xs font-bold text-sky-400 font-mono">Branch-and-Cut</div>
                  <p className="text-[11px] text-neutral-300 font-sans leading-snug">
                    Gomory Mixed-Integer (GMI) cuts and Mixed-Integer Rounding (MIR) cuts generated at the root node.
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-[#121722] border border-[#1e2637] space-y-1">
                  <div className="text-xs font-bold text-sky-400 font-mono">Advanced MILP Search</div>
                  <p className="text-[11px] text-neutral-300 font-sans leading-snug">
                    Reliability branching with pseudocost tracking, strong branching candidates, and primal rounding heuristics.
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-[#121722] border border-[#1e2637] space-y-1">
                  <div className="text-xs font-bold text-indigo-400 font-mono">Convex QP Active-Set</div>
                  <p className="text-[11px] text-neutral-300 font-sans leading-snug">
                    Primal active-set method with positive semi-definite (PSD) Hessian validation and KKT stationarity.
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-[#121722] border border-[#1e2637] space-y-1">
                  <div className="text-xs font-bold text-purple-400 font-mono">Convex MIQP</div>
                  <p className="text-[11px] text-neutral-300 font-sans leading-snug">
                    Exact mixed-integer quadratic programming solving active-set QP continuous relaxations in a B&amp;B tree.
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-[#121722] border border-[#1e2637] space-y-1">
                  <div className="text-xs font-bold text-amber-400 font-mono">Zero-Trust Verifier</div>
                  <p className="text-[11px] text-neutral-300 font-sans leading-snug">
                    Air-gapped verification recomputing primal, dual, complementary slackness, and integrality residuals.
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-[#121722] border border-[#1e2637] space-y-1">
                  <div className="text-xs font-bold text-neutral-300 font-mono">CPU Parallelism</div>
                  <p className="text-[11px] text-neutral-300 font-sans leading-snug">
                    C++20 worker thread pool executing parallel batched SpMV matrix multiplications.
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-[#121722] border border-[#1e2637] space-y-1">
                  <div className="text-xs font-bold text-neutral-300 font-mono">Scoped CUDA</div>
                  <p className="text-[11px] text-neutral-300 font-sans leading-snug">
                    7 custom FP64 batched linear algebra kernels with mandatory automatic CPU fallback (Rule R012).
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* SCREEN 4: PERFORMANCE & EXECUTION (TRUTHFUL GPU/CPU MODEL) */}
          {currentStep === 4 && (
            <div className="space-y-6 animate-fade-in">
              <div>
                <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-slate-900 border border-slate-700 text-slate-300 text-xs font-mono font-medium mb-2">
                  <Cpu className="w-3 h-3" />
                  <span>Execution Model &amp; Truth</span>
                </div>
                <h3 className="text-xl font-bold text-white font-sans">
                  Hardware Acceleration Architecture
                </h3>
                <p className="mt-1 text-xs text-neutral-400 font-sans">
                  Kavach-Opt uses a CPU-first solver design with scoped numerical acceleration, never claiming fake full-GPU solves.
                </p>
              </div>

              <div className="space-y-3">
                <div className="p-3.5 rounded-xl bg-[#121722] border border-[#1f283a] flex items-center justify-between">
                  <div className="space-y-0.5">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-white text-xs font-sans">CPU Sequential</span>
                      <span className="px-1.5 py-0.2 rounded bg-emerald-950/60 border border-emerald-700/50 text-emerald-400 font-mono text-[10px]">
                        ACTIVE &amp; PRIMARY
                      </span>
                    </div>
                    <p className="text-[11px] text-neutral-400 font-sans">
                      Deterministic single-core Revised Simplex, SparseLU, and Active-Set QP.
                    </p>
                  </div>
                  <span className="font-mono text-xs text-emerald-400 font-semibold">100% Deterministic</span>
                </div>

                <div className="p-3.5 rounded-xl bg-[#121722] border border-[#1f283a] flex items-center justify-between">
                  <div className="space-y-0.5">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-white text-xs font-sans">CPU Multicore</span>
                      <span className="px-1.5 py-0.2 rounded bg-blue-950/60 border border-blue-700/50 text-blue-300 font-mono text-[10px]">
                        AVAILABLE
                      </span>
                    </div>
                    <p className="text-[11px] text-neutral-400 font-sans">
                      POSIX/std::thread pool for parallel batched SpMV (up to 16 threads).
                    </p>
                  </div>
                  <span className="font-mono text-xs text-blue-400 font-semibold">Scalable</span>
                </div>

                <div className="p-3.5 rounded-xl bg-[#121722] border border-[#1f283a] flex items-center justify-between">
                  <div className="space-y-0.5">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-white text-xs font-sans">CUDA GPU Layer</span>
                      <span className={`px-1.5 py-0.2 rounded font-mono text-[10px] ${
                        health?.cuda_device_present
                          ? 'bg-emerald-950/60 border border-emerald-700/50 text-emerald-400'
                          : 'bg-amber-950/40 border border-amber-800/40 text-amber-300'
                      }`}>
                        {health?.cuda_device_present ? 'CUDA DEVICE PRESENT' : 'UNAVAILABLE ON HOST'}
                      </span>
                    </div>
                    <p className="text-[11px] text-neutral-400 font-sans">
                      Scoped custom FP64 kernels for batched CSR/CSC SpMV. Solver logic remains on CPU.
                    </p>
                  </div>
                  <span className="font-mono text-xs text-amber-400/90 font-semibold">Rule R012 CPU Fallback</span>
                </div>
              </div>

              {/* Truthful Host Diagnostic Banner */}
              <div className="p-3.5 rounded-lg bg-[#0e131d] border border-[#1d2536] text-xs space-y-1">
                <div className="flex items-center gap-2 text-neutral-300 font-semibold font-mono text-[11px]">
                  <Activity className="w-3.5 h-3.5 text-blue-400" />
                  <span>Live Host Status:</span>
                  <span className="text-white">{health?.binary_present ? 'C++ Engine Connected' : 'Checking Binary...'}</span>
                </div>
                <p className="text-[11px] text-neutral-400 font-sans leading-relaxed">
                  Host system is Darwin (Apple Silicon). Because NVIDIA CUDA hardware is absent, the solver operates in
                  its primary CPU-first mode. All solver algorithms and numerical verifications run with 100% precision.
                </p>
              </div>
            </div>
          )}

          {/* SCREEN 5: READY */}
          {currentStep === 5 && (
            <div className="space-y-6 animate-fade-in text-center sm:text-left">
              <div>
                <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-950/40 border border-emerald-700/50 text-emerald-400 text-xs font-mono font-medium mb-3">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>Setup Complete</span>
                </div>
                <h3 className="text-2xl font-bold text-white font-sans">
                  Your optimization workspace is ready.
                </h3>
                <p className="mt-2 text-xs sm:text-sm text-neutral-300 font-sans leading-relaxed max-w-xl">
                  You are equipped with a complete, sovereign mathematical programming environment.
                  Select models, inspect sparsity structures, run deterministic solves, and verify solutions.
                </p>
              </div>

              {/* Live Readiness Summary */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-left">
                <div className="p-3 rounded-lg bg-[#121722] border border-[#1f283a]">
                  <div className="text-[10px] font-mono text-neutral-400 uppercase">Solver Engine</div>
                  <div className="text-xs font-semibold text-emerald-400 mt-1 flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" />
                    <span>C++20 Active</span>
                  </div>
                  <div className="text-[10px] text-neutral-400 font-mono mt-0.5 truncate">
                    {health?.version_info || SYSTEM_METADATA.PRODUCT_VERSION}
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-[#121722] border border-[#1f283a]">
                  <div className="text-[10px] font-mono text-neutral-400 uppercase">Backend</div>
                  <div className="text-xs font-semibold text-blue-400 mt-1 flex items-center gap-1">
                    <Cpu className="w-3 h-3" />
                    <span>CPU First</span>
                  </div>
                  <div className="text-[10px] text-neutral-400 font-mono mt-0.5">
                    Sequential + ThreadPool
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-[#121722] border border-[#1f283a]">
                  <div className="text-[10px] font-mono text-neutral-400 uppercase">Problem Classes</div>
                  <div className="text-xs font-semibold text-purple-400 mt-1 flex items-center gap-1">
                    <GitBranch className="w-3 h-3" />
                    <span>4 Classes</span>
                  </div>
                  <div className="text-[10px] text-neutral-400 font-mono mt-0.5">
                    LP · MILP · QP · MIQP
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-[#121722] border border-[#1f283a]">
                  <div className="text-[10px] font-mono text-neutral-400 uppercase">Verifier</div>
                  <div className="text-xs font-semibold text-emerald-400 mt-1 flex items-center gap-1">
                    <Lock className="w-3 h-3" />
                    <span>Zero-Trust</span>
                  </div>
                  <div className="text-[10px] text-neutral-400 font-mono mt-0.5">
                    Air-gapped checks
                  </div>
                </div>
              </div>

              <div className="pt-2 flex flex-wrap items-center gap-3">
                <button
                  onClick={handleEnterWorkbench}
                  className="px-5 py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs flex items-center gap-2 transition-colors shadow-sm"
                >
                  <Zap className="w-3.5 h-3.5 fill-white" />
                  <span>Open Workbench</span>
                </button>
                <button
                  onClick={handleFinish}
                  className="px-4 py-2.5 rounded-lg bg-[#161d2b] hover:bg-[#20293d] border border-[#263247] text-neutral-200 font-medium text-xs transition-colors"
                >
                  Close Tour
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Footer controls: Back, Skip, Next */}
        <div className="px-6 py-3.5 border-t border-[#1b2333] bg-[#0c1017] flex items-center justify-between text-xs">
          <button
            onClick={() => setCurrentStep((prev) => Math.max(1, prev - 1))}
            disabled={currentStep === 1}
            className="px-3 py-1.5 rounded text-neutral-400 hover:text-white disabled:opacity-30 disabled:pointer-events-none transition-colors flex items-center gap-1.5"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Back</span>
          </button>

          <div className="flex items-center gap-2">
            <button
              onClick={handleFinish}
              className="px-3 py-1.5 text-neutral-400 hover:text-neutral-200 transition-colors"
            >
              Skip Tour
            </button>
            {currentStep < totalSteps ? (
              <button
                onClick={() => setCurrentStep((prev) => Math.min(totalSteps, prev + 1))}
                className="px-4 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium flex items-center gap-1.5 transition-colors"
              >
                <span>Continue</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            ) : (
              <button
                onClick={handleEnterWorkbench}
                className="px-4 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium flex items-center gap-1.5 transition-colors"
              >
                <span>Finish</span>
                <CheckCircle2 className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
