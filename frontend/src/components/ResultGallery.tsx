import { Download, Loader2 } from 'lucide-react';
import type { ImageOutput } from '../api/types';

interface ResultGalleryProps {
  results: ImageOutput[];
  isLoading: boolean;
  elapsedTime: number | null;
  error: string | null;
  loadingSeconds: number;
}

function downloadImage(base64: string, index: number) {
  const link = document.createElement('a');
  link.href = `data:image/png;base64,${base64}`;
  link.download = `qwen-edit-${index + 1}.png`;
  link.click();
}

export function ResultGallery({ results, isLoading, elapsedTime, error, loadingSeconds }: ResultGalleryProps) {
  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-secondary)] min-h-[300px] h-full flex flex-col">
      <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--color-border)]">
        <span className="text-sm text-[var(--color-text-secondary)]">Result</span>
        {elapsedTime !== null && (
          <span className="text-xs text-[var(--color-text-secondary)] font-mono">
            {elapsedTime.toFixed(1)}s
          </span>
        )}
      </div>

      <div className="flex-1 flex items-center justify-center p-4">
        {isLoading && (
          <div className="flex flex-col items-center gap-3 text-[var(--color-text-secondary)]">
            <Loader2 size={32} className="animate-spin text-[var(--color-accent)]" />
            <p className="text-sm">Generating...</p>
            <p className="text-xs font-mono tabular-nums">{loadingSeconds}s elapsed</p>
          </div>
        )}

        {error && !isLoading && (
          <div className="text-center">
            <p className="text-sm text-[var(--color-danger)]">{error}</p>
          </div>
        )}

        {!isLoading && !error && results.length === 0 && (
          <p className="text-sm text-[var(--color-text-secondary)]">
            Results will appear here
          </p>
        )}

        {!isLoading && results.length > 0 && (
          <div className="grid grid-cols-1 gap-4 w-full">
            {results.map((result) => (
              <div key={result.index} className="relative group">
                <img
                  src={`data:image/png;base64,${result.data}`}
                  alt={`Result ${result.index + 1}`}
                  className="w-full max-h-[400px] object-contain rounded-lg"
                />
                <div className="absolute top-2 right-2 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => downloadImage(result.data, result.index)}
                    className="p-2 rounded-lg bg-black/60 text-white hover:bg-black/80 transition-colors"
                    title="Download PNG"
                  >
                    <Download size={16} />
                  </button>
                </div>
                <div className="absolute bottom-2 left-2 text-xs font-mono text-white/70 bg-black/40 px-2 py-0.5 rounded">
                  seed: {result.seed}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
