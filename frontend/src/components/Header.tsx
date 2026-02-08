import { HealthIndicator } from './HealthIndicator';
import type { HealthResponse } from '../api/types';

interface HeaderProps {
  health: HealthResponse | null;
  healthError: string | null;
}

export function Header({ health, healthError }: HeaderProps) {
  return (
    <header className="flex items-center justify-between py-4 border-b border-[var(--color-border)]">
      <img
        src="/qwen-logo.png"
        alt="Qwen Image Edit"
        className="h-16"
      />
      <HealthIndicator health={health} error={healthError} />
    </header>
  );
}
