/**
 * @file tokens.ts
 * @brief Centralized design tokens for Kavach-Opt industrial UI.
 */

export const colors = {
  // 3-Tier Layered Dark Neutrals (Engineering Workstation)
  surface: {
    canvas: '#0a0d12',    // Canvas background
    primary: '#11151f',   // Primary card / panel container
    elevated: '#161b28',  // Elevated inputs, highlights, sub-panels
    hover: '#1b2233',     // Interactive hover state
    active: '#20283d',    // Active / selected surface
    subtle: '#252f47',    // Accented subtle background
  },
  border: {
    subtle: '#1d2332',    // Quiet boundary
    default: '#262f42',   // Standard structural border
    focus: '#3b4763',     // Active focus border
    highlight: '#475569', // Highlighted edge
  },
  text: {
    primary: '#f1f5f9',   // Crisp white-slate
    secondary: '#94a3b8', // Descriptive labels / captions
    muted: '#64748b',     // Secondary details / timestamps
    faint: '#475569',     // Inactive / disabled text
  },
  // Semantic status colors - calibrated for high legibility without neon bleed
  semantic: {
    success: {
      text: '#34d399',      // emerald-400
      bg: 'rgba(16, 185, 129, 0.08)',
      border: 'rgba(16, 185, 129, 0.22)',
      dot: '#10b981',
    },
    warning: {
      text: '#fbbf24',      // amber-400
      bg: 'rgba(245, 158, 11, 0.08)',
      border: 'rgba(245, 158, 11, 0.22)',
      dot: '#f59e0b',
    },
    danger: {
      text: '#f87171',      // red-400
      bg: 'rgba(239, 68, 68, 0.08)',
      border: 'rgba(239, 68, 68, 0.22)',
      dot: '#ef4444',
    },
    info: {
      text: '#38bdf8',      // sky-400
      bg: 'rgba(14, 165, 233, 0.08)',
      border: 'rgba(14, 165, 233, 0.22)',
      dot: '#0ea5e9',
    },
    neutral: {
      text: '#94a3b8',      // slate-400
      bg: 'rgba(148, 163, 184, 0.06)',
      border: 'rgba(148, 163, 184, 0.18)',
      dot: '#64748b',
    },
  },
};

export const typography = {
  sans: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  mono: "'JetBrains Mono', 'SF Mono', Menlo, Monaco, Consolas, monospace",
};
