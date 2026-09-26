import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  icon?: React.ReactNode;
  size?: 'sm' | 'md' | 'lg';
  fullWidth?: boolean;
}

export const PrimaryButton: React.FC<ButtonProps> = ({
  children,
  icon,
  size = 'md',
  fullWidth = false,
  className = '',
  disabled,
  ...props
}) => {
  const sizeClasses =
    size === 'sm'
      ? 'px-3 py-1.5 text-xs'
      : size === 'lg'
      ? 'px-5 py-3 text-sm font-semibold'
      : 'px-4 py-2 text-xs font-medium';

  return (
    <button
      disabled={disabled}
      className={`inline-flex items-center justify-center gap-2 rounded-md font-sans tracking-wide transition-colors
        bg-emerald-600 hover:bg-emerald-500 active:bg-emerald-700 text-white shadow-sm
        focus:outline-none focus:ring-2 focus:ring-emerald-500/40 focus:ring-offset-1 focus:ring-offset-[#0b0d11]
        disabled:opacity-50 disabled:pointer-events-none ${sizeClasses} ${fullWidth ? 'w-full' : ''} ${className}`}
      {...props}
    >
      {icon && <span className="w-4 h-4">{icon}</span>}
      {children}
    </button>
  );
};

export const SecondaryButton: React.FC<ButtonProps> = ({
  children,
  icon,
  size = 'md',
  fullWidth = false,
  className = '',
  disabled,
  ...props
}) => {
  const sizeClasses =
    size === 'sm'
      ? 'px-3 py-1.5 text-xs'
      : size === 'lg'
      ? 'px-5 py-3 text-sm font-semibold'
      : 'px-4 py-2 text-xs font-medium';

  return (
    <button
      disabled={disabled}
      className={`inline-flex items-center justify-center gap-2 rounded-md font-sans tracking-wide transition-colors
        bg-[#161a22] hover:bg-[#1f242e] active:bg-[#252b37] text-neutral-200 border border-[#272f3d]
        focus:outline-none focus:ring-2 focus:ring-neutral-500/40 focus:ring-offset-1 focus:ring-offset-[#0b0d11]
        disabled:opacity-50 disabled:pointer-events-none ${sizeClasses} ${fullWidth ? 'w-full' : ''} ${className}`}
      {...props}
    >
      {icon && <span className="w-4 h-4 text-neutral-400">{icon}</span>}
      {children}
    </button>
  );
};

export const GhostButton: React.FC<ButtonProps> = ({
  children,
  icon,
  size = 'md',
  fullWidth = false,
  className = '',
  disabled,
  ...props
}) => {
  const sizeClasses =
    size === 'sm' ? 'px-2 py-1 text-xs' : 'px-3 py-1.5 text-xs font-medium';

  return (
    <button
      disabled={disabled}
      className={`inline-flex items-center justify-center gap-1.5 rounded font-sans tracking-wide transition-colors
        text-neutral-400 hover:text-neutral-200 hover:bg-[#161a22]
        disabled:opacity-50 disabled:pointer-events-none ${sizeClasses} ${fullWidth ? 'w-full' : ''} ${className}`}
      {...props}
    >
      {icon && <span className="w-3.5 h-3.5">{icon}</span>}
      {children}
    </button>
  );
};
