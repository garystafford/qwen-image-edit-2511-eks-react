import type { HealthResponse, InferenceRequest, InferenceResponse } from './types';

const API_BASE = '/api/v1';

export async function checkHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`, { signal });
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function submitInference(
  request: InferenceRequest,
  signal?: AbortSignal,
): Promise<InferenceResponse> {
  const res = await fetch(`${API_BASE}/batch/infer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
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
