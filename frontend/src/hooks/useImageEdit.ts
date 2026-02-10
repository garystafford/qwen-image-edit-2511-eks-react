import { useRef, useState } from 'react';
import { submitInference } from '../api/client';
import type { ImageOutput, InferenceRequest } from '../api/types';

export interface InferenceProgress {
  step: number;
  totalSteps: number;
}

export function useImageEdit() {
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<ImageOutput[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [elapsedTime, setElapsedTime] = useState<number | null>(null);
  const [progress, setProgress] = useState<InferenceProgress | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  const submit = async (request: InferenceRequest) => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;

    setIsLoading(true);
    setError(null);
    setResults([]);
    setElapsedTime(null);
    setProgress(null);

    try {
      const response = await submitInference(
        request,
        controller.signal,
        (step, totalSteps) => setProgress({ step, totalSteps }),
      );
      if (!response.success) {
        throw new Error(response.error || 'Unknown error');
      }
      setResults(response.images);
      setElapsedTime(response.total_time_seconds);
    } catch (e: unknown) {
      if (e instanceof Error && e.name !== 'AbortError') {
        setError(e.message);
      }
    } finally {
      if (!controller.signal.aborted) {
        setIsLoading(false);
        setProgress(null);
      }
    }
  };

  const cancel = () => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setIsLoading(false);
    setProgress(null);
  };

  const clearResults = () => {
    setResults([]);
    setError(null);
    setElapsedTime(null);
  };

  return { isLoading, results, error, elapsedTime, progress, submit, cancel, clearResults };
}
