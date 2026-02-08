import { X } from 'lucide-react';
import clsx from 'clsx';
import type { UploadedImage } from '../api/types';

interface ImagePreviewProps {
  images: UploadedImage[];
  onRemove: (index: number) => void;
}

export function ImagePreview({ images, onRemove }: ImagePreviewProps) {
  if (images.length === 0) return null;

  const cols =
    images.length === 1
      ? 'grid-cols-1'
      : images.length === 2
        ? 'grid-cols-2'
        : 'grid-cols-3';

  return (
    <div className={clsx('grid gap-3', cols)}>
      {images.map((img, idx) => (
        <div key={img.preview} className="relative group rounded-lg overflow-hidden bg-[var(--color-bg-tertiary)]">
          <img
            src={img.preview}
            alt={img.file.name}
            className={clsx('w-full object-contain', images.length === 1 ? 'max-h-[300px]' : 'aspect-square')}
          />
          <button
            onClick={() => onRemove(idx)}
            className="absolute top-1 right-1 p-1 rounded-full bg-black/60 text-white opacity-0 group-hover:opacity-100 transition-opacity hover:bg-black/80"
          >
            <X size={14} />
          </button>
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent px-2 py-1">
            <p className="text-[10px] text-white truncate">{img.file.name}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
