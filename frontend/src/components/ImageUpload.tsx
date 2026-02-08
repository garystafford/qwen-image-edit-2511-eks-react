import { useCallback, useRef, useState } from 'react';
import { Upload } from 'lucide-react';
import clsx from 'clsx';

interface ImageUploadProps {
  onImagesAdded: (files: File[]) => void;
  disabled?: boolean;
}

const ACCEPTED_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/bmp', 'image/tiff'];

export function ImageUpload({ onImagesAdded, disabled }: ImageUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files) return;
      const valid = Array.from(files).filter((f) => ACCEPTED_TYPES.includes(f.type));
      if (valid.length > 0) onImagesAdded(valid);
    },
    [onImagesAdded],
  );

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (!disabled) handleFiles(e.dataTransfer.files);
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      className={clsx(
        'flex flex-col items-center justify-center gap-3 p-8 rounded-lg border-2 border-dashed cursor-pointer transition-all',
        isDragging
          ? 'border-[var(--color-accent)] bg-[var(--color-accent)]/10'
          : 'border-[var(--color-border)] hover:border-[var(--color-accent)]/50 hover:bg-[var(--color-bg-tertiary)]',
        disabled && 'opacity-50 cursor-not-allowed',
      )}
    >
      <Upload size={24} className="text-[var(--color-text-secondary)]" />
      <div className="text-center">
        <p className="text-sm text-[var(--color-text-primary)]">
          Drop images here or click to upload
        </p>
        <p className="text-xs text-[var(--color-text-secondary)] mt-1">
          PNG, JPEG, WebP supported
        </p>
      </div>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept="image/*"
        onChange={(e) => handleFiles(e.target.files)}
        className="hidden"
      />
    </div>
  );
}
