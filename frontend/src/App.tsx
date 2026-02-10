import { useState, useRef, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { ImageUpload } from './components/ImageUpload';
import { ImagePreview } from './components/ImagePreview';
import { PromptInput } from './components/PromptInput';
import { ActionButtons } from './components/ActionButtons';
import { AdvancedSettings } from './components/AdvancedSettings';
import { ResultGallery } from './components/ResultGallery';
import { useHealthCheck } from './hooks/useHealthCheck';
import { useImageEdit } from './hooks/useImageEdit';
import { fileToBase64 } from './api/client';
import type { UploadedImage, AdvancedSettings as SettingsType } from './api/types';
import { DEFAULT_SETTINGS } from './api/types';

export default function App() {
  const { health, error: healthError } = useHealthCheck();
  const { isLoading, results, error, elapsedTime, progress, submit, cancel, clearResults } = useImageEdit();

  const [images, setImages] = useState<UploadedImage[]>([]);
  const [prompt, setPrompt] = useState('');
  const [settings, setSettings] = useState<SettingsType>(DEFAULT_SETTINGS);
  const [loadingSeconds, setLoadingSeconds] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval>>(undefined);

  useEffect(() => {
    if (!isLoading) {
      clearInterval(timerRef.current);
      return;
    }
    timerRef.current = setInterval(() => {
      setLoadingSeconds((s) => s + 1);
    }, 1000);
    return () => clearInterval(timerRef.current);
  }, [isLoading]);

  const handleImagesAdded = useCallback((files: File[]) => {
    const newImages = files.map((file) => ({
      file,
      preview: URL.createObjectURL(file),
    }));
    setImages((prev) => [...prev, ...newImages]);
  }, []);

  const handleRemoveImage = useCallback((index: number) => {
    setImages((prev) => {
      const removed = prev[index];
      URL.revokeObjectURL(removed.preview);
      return prev.filter((_, i) => i !== index);
    });
  }, []);

  const handleClear = useCallback(() => {
    images.forEach((img) => URL.revokeObjectURL(img.preview));
    setImages([]);
    setPrompt('');
    clearResults();
  }, [images, clearResults]);

  const handleSubmit = async () => {
    if (images.length === 0 || !prompt.trim()) return;
    setLoadingSeconds(0);

    const imageInputs = await Promise.all(
      images.map(async (img) => ({
        data: await fileToBase64(img.file),
        filename: img.file.name,
      })),
    );

    submit({
      images: imageInputs,
      prompt: prompt.trim(),
      negative_prompt: settings.negative_prompt,
      seed: settings.seed,
      randomize_seed: settings.randomize_seed,
      guidance_scale: settings.guidance_scale,
      num_inference_steps: settings.num_inference_steps,
      height: settings.height,
      width: settings.width,
      num_images_per_prompt: settings.num_images_per_prompt,
      style_reference_mode: settings.style_reference_mode,
    });
  };

  const canSubmit = images.length > 0 && prompt.trim().length > 0 && !isLoading;

  return (
    <div className="min-h-screen flex flex-col bg-[var(--color-bg-primary)] text-[var(--color-text-primary)]">
      <div className="flex-1 max-w-7xl w-full mx-auto px-4 pt-8 pb-1">
        <Header health={health} healthError={healthError} />

        <div className="mt-5 grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Left column: source images */}
          <div className="flex flex-col gap-4">
            <ImageUpload onImagesAdded={handleImagesAdded} disabled={isLoading} />
            <ImagePreview images={images} onRemove={handleRemoveImage} />
          </div>

          {/* Right column: results */}
          <ResultGallery
            results={results}
            isLoading={isLoading}
            elapsedTime={elapsedTime}
            error={error}
            loadingSeconds={loadingSeconds}
            progress={progress}
          />
        </div>

        <div className="mt-4 grid gap-4">
          <PromptInput
            value={prompt}
            onChange={setPrompt}
            onSubmit={handleSubmit}
            disabled={isLoading}
          />

          <AdvancedSettings settings={settings} onChange={setSettings} />

          <ActionButtons
            onEdit={handleSubmit}
            onCancel={cancel}
            onClear={handleClear}
            isLoading={isLoading}
            canSubmit={canSubmit}
          />
        </div>
      </div>

      <footer className="py-3">
        <div className="max-w-7xl mx-auto px-4 border-t border-[var(--color-border)] pt-3 flex items-center justify-center gap-2 text-xs text-[var(--color-text-secondary)]">
          <a
            href="https://www.linkedin.com/in/garystafford/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[var(--color-accent)] hover:text-[var(--color-accent-hover)] transition-colors"
          >
            Gary A. Stafford
          </a>
          <span>|</span>
          <a
            href="https://github.com/garystafford"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[var(--color-accent)] hover:text-[var(--color-accent-hover)] transition-colors"
          >
            GitHub
          </a>
          <span>|</span>
          <a
            href="https://huggingface.co/Qwen/Qwen-Image-Edit-2511"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[var(--color-accent)] hover:text-[var(--color-accent-hover)] transition-colors"
          >
            Model Card
          </a>
        </div>
      </footer>
    </div>
  );
}
