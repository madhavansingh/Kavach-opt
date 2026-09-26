/**
 * @file tokens.ts
 * @brief Centralized design tokens for Kavach-Opt industrial UI.
 */

export const colors = {
  // Layered dark neutrals (depth without pitch black)
  surface: {
    bg: '#0b0d11',
    card: '#11141a',
    panel: '#161a22',
    hover: '#1b202a',
    active: '#212733',
    subtle: '#2a3140',
  },
  border: {
    subtle: '#1e2430',
    default: '#272f3d',
    focus: '#3b4557',
  },
  text: {
    primary: '#f1f5f9',
    secondary: '#94a3b8',
    muted: '#64748b',
    faint: '#475569',
  },
  // Semantic status colors
  semantic: {
    success: {
      text: '#34d399',      // emerald-400
      bg: 'rgba(16, 185, 129, 0.1)',
      border: 'rgba(16, 185, 129, 0.25)',
      dot: '#10b981',
    },
    warning: {
      text: '#fbbf24',      // amber-400
      bg: 'rgba(245, 158, 11, 0.1)',
      border: 'rgba(245, 158, 11, 0.25)',
      dot: '#f59e0b',
    },
    danger: {
      text: '#f87171',      // red-400
      bg: 'rgba(239, 68, 68, 0.1)',
      border: 'rgba(239, 68, 68, 0.25)',
      dot: '#ef4444',
    },
    info: {
      text: '#38bdf8',      // sky-400
      bg: 'rgba(14, 165, 233, 0.1)',
      border: 'rgba(14, 165, 233, 0.25)',
      dot: '#0ea5e9',
    },
    neutral: {
      text: '#94a3b8',      // slate-400
      bg: 'rgba(148, 163, 184, 0.08)',
      border: 'rgba(148, 163, 184, 0.2)',
      dot: '#64748b',
    },
  },
};

export const typography = {
  sans: "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  mono: "'JetBrains Mono', 'SF Mono', Menlo, Monaco, Consolas, monospace",
};
