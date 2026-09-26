import { createContext } from 'react';

export type ToastType = 'info' | 'success' | 'warning' | 'error';

export interface ToastItem {
  id: string;
  type: ToastType;
  title: string;
  message?: string;
  duration?: number;
  lifecycleKey?: string;
}

export interface ToastContextValue {
  showToast: (
    type: ToastType,
    title: string,
    message?: string,
    duration?: number,
    lifecycleKey?: string
  ) => string;
  dismissToast: (id: string) => void;
  dismissByLifecycleKey: (key: string) => void;
}

export const ToastContext = createContext<ToastContextValue | null>(null);
