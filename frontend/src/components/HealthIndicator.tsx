import { Info } from 'lucide-react';
import type { HealthResponse } from '../api/types';
import clsx from 'clsx';

interface HealthIndicatorProps {
  health: HealthResponse | null;
  error: string | null;
}

export function HealthIndicator({ health, error }: HealthIndicatorProps) {
  const isHealthy = health?.status === 'healthy';
  const isConnected = health !== null && !error;

  const statusLabel = isConnected
    ? isHealthy
      ? 'Healthy'
      : 'Degraded'
    : 'Disconnected';

  const tooltipLines: string[] = [];
  if (isConnected && health) {
    tooltipLines.push(`Model backend is ${isHealthy ? 'ready' : 'not fully ready'}`);
    tooltipLines.push(
      `GPU: ${health.gpu_memory_used_gb ?? '?'} / ${health.gpu_memory_total_gb ?? '?'} GB`,
    );
  } else {
    tooltipLines.push('Cannot reach model backend');
    tooltipLines.push('Check that the model pod is running');
  }

  return (
    <div className="relative flex items-center gap-2 text-xs font-mono group">
      <div
        className={clsx(
          'w-2 h-2 rounded-full',
          isConnected && isHealthy && 'bg-[var(--color-success)]',
          isConnected && !isHealthy && 'bg-[var(--color-warning)]',
          !isConnected && 'bg-[var(--color-danger)]',
        )}
      />
      <span className="text-[var(--color-text-secondary)]">{statusLabel}</span>
      <Info size={13} className="text-[var(--color-text-secondary)] cursor-help" />

      {/* Tooltip */}
      <div className="absolute right-0 top-full mt-2 w-80 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-tertiary)] px-3 py-2 text-xs text-[var(--color-text-secondary)] opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto transition-opacity z-50 shadow-lg">
        {tooltipLines.map((line, i) => (
          <p key={i} className={i > 0 ? 'mt-1' : ''}>{line}</p>
        ))}
        <hr className="my-1.5 border-[var(--color-border)]" />
        <p className="leading-relaxed">
          <span className="inline-block w-2 h-2 rounded-full bg-[var(--color-success)] mr-1 align-middle" /> Healthy &mdash; ready for inference<br />
          <span className="inline-block w-2 h-2 rounded-full bg-[var(--color-warning)] mr-1 align-middle" /> Degraded &mdash; model loading<br />
          <span className="inline-block w-2 h-2 rounded-full bg-[var(--color-danger)] mr-1 align-middle" /> Disconnected &mdash; backend unreachable
        </p>
      </div>
    </div>
  );
}
