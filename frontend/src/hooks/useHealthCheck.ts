import { useEffect, useState } from 'react';
import { checkHealth } from '../api/client';
import type { HealthResponse } from '../api/types';

export function useHealthCheck(intervalMs = 10_000) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setInterval>;

    const poll = async () => {
      try {
        const data = await checkHealth(controller.signal);
        setHealth(data);
        setError(null);
      } catch (e) {
        if (!controller.signal.aborted) {
          setHealth(null);
          setError('Not connected');
        }
      }
    };

    poll();
    timer = setInterval(poll, intervalMs);

    return () => {
      controller.abort();
      clearInterval(timer);
    };
  }, [intervalMs]);

  return { health, error };
}
