import type { HealthResponse, InferenceRequest, InferenceResponse, StreamEvent } from './types';

const API_BASE = '/api/v1';

export async function checkHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`, { signal });
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function submitInference(
  request: InferenceRequest,
  signal?: AbortSignal,
  onProgress?: (step: number, totalSteps: number) => void,
): Promise<InferenceResponse> {
  const res = await fetch(`${API_BASE}/stream/infer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result: InferenceResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Parse complete SSE events from buffer
    while (true) {
      const idx = buffer.indexOf('\n\n');
      if (idx === -1) break;
      const raw = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);

      for (const line of raw.split('\n')) {
        if (!line.startsWith('data: ')) continue;
        const event: StreamEvent = JSON.parse(line.slice(6));
        switch (event.type) {
          case 'progress':
            onProgress?.(event.step, event.total_steps);
            break;
          case 'complete':
            result = {
              success: true,
              images: event.images,
              total_time_seconds: event.total_time_seconds,
              error: null,
            };
            break;
          case 'error':
            throw new Error(event.error);
        }
      }
    }
  }

  if (!result) throw new Error('Stream ended without result');
  return result;
}

export async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const base64 = result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
