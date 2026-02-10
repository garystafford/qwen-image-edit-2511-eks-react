export interface ImageInput {
  data: string;
  filename?: string;
}

export interface ImageOutput {
  data: string;
  seed: number;
  index: number;
}

export interface InferenceRequest {
  images: ImageInput[];
  prompt: string;
  negative_prompt: string;
  seed: number;
  randomize_seed: boolean;
  guidance_scale: number;
  num_inference_steps: number;
  height: number;
  width: number;
  num_images_per_prompt: number;
  style_reference_mode: boolean;
}

export interface InferenceResponse {
  success: boolean;
  images: ImageOutput[];
  total_time_seconds: number;
  error: string | null;
}

export type StreamEvent =
  | { type: 'started'; total_steps: number }
  | { type: 'progress'; step: number; total_steps: number }
  | { type: 'heartbeat' }
  | { type: 'complete'; success: true; images: ImageOutput[]; total_time_seconds: number }
  | { type: 'error'; error: string };

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  gpu_available: boolean;
  gpu_memory_used_gb: number | null;
  gpu_memory_total_gb: number | null;
}

export interface UploadedImage {
  file: File;
  preview: string;
}

export interface AdvancedSettings {
  style_reference_mode: boolean;
  negative_prompt: string;
  seed: number;
  randomize_seed: boolean;
  guidance_scale: number;
  num_inference_steps: number;
  height: number;
  width: number;
  num_images_per_prompt: number;
}

export const DEFAULT_SETTINGS: AdvancedSettings = {
  style_reference_mode: true,
  negative_prompt:
    'blurry, out of focus, low resolution, low detail, low sharpness, double-image, soft edges, motion blur, depth of field blur, hazy, unclear, artifact, noisy, pixelated, compression artifacts, \'AI generated\' text and surrounding box',
  seed: 42,
  randomize_seed: true,
  guidance_scale: 3.0,
  num_inference_steps: 20,
  height: 1024,
  width: 1024,
  num_images_per_prompt: 1,
};
