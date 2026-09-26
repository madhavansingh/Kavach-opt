import React, { useState, useCallback } from 'react';
import { CheckCircle2, AlertTriangle, Info, XCircle, X } from 'lucide-react';
import { ToastContext, type ToastItem, type ToastType } from './toastDef';

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const showToast = useCallback(
    (type: ToastType, title: string, message?: string, duration = 3500) => {
      const id = `${Date.now()}-${Math.random().toString(36).substring(2, 6)}`;
      const newToast: ToastItem = { id, type, title, message, duration };
      setToasts((prev) => [...prev.slice(-4), newToast]); // Limit to max 5 visible

      if (duration > 0) {
        setTimeout(() => {
          dismissToast(id);
        }, duration);
      }
    },
    [dismissToast]
  );

  return (
    <ToastContext.Provider value={{ showToast, dismissToast }}>
      {children}
      {/* Toast Render Container */}
      <div
        className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm w-full pointer-events-none px-4 sm:px-0"
        aria-live="polite"
        aria-atomic="true"
      >
        {toasts.map((toast) => {
          const typeStyles = {
            info: 'bg-[#11151f] border-blue-500/40 text-blue-200',
            success: 'bg-[#11151f] border-emerald-500/40 text-emerald-200',
            warning: 'bg-[#11151f] border-amber-500/40 text-amber-200',
            error: 'bg-[#11151f] border-rose-500/40 text-rose-200',
          }[toast.type];

          const IconComponent = {
            info: Info,
            success: CheckCircle2,
            warning: AlertTriangle,
            error: XCircle,
          }[toast.type];

          return (
            <div
              key={toast.id}
              className={`pointer-events-auto rounded-lg border p-3 shadow-lg flex items-start gap-2.5 transition-all duration-200 animate-fade-in ${typeStyles}`}
              role="alert"
            >
              <IconComponent className="w-4 h-4 shrink-0 mt-0.5" />
              <div className="flex-1 min-w-0">
                <div className="text-xs font-semibold text-white font-sans truncate">
                  {toast.title}
                </div>
                {toast.message && (
                  <div className="text-[11px] text-neutral-300 font-sans mt-0.5 leading-relaxed">
                    {toast.message}
                  </div>
                )}
              </div>
              <button
                type="button"
                onClick={() => dismissToast(toast.id)}
                className="text-neutral-400 hover:text-white transition-colors p-0.5 rounded shrink-0"
                aria-label="Dismiss notification"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
};
