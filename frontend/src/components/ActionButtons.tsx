import { Trash2 } from 'lucide-react';
import clsx from 'clsx';

interface ActionButtonsProps {
  onEdit: () => void;
  onCancel: () => void;
  onClear: () => void;
  isLoading: boolean;
  canSubmit: boolean;
}

export function ActionButtons({ onEdit, onCancel, onClear, isLoading, canSubmit }: ActionButtonsProps) {
  return (
    <div className="flex gap-3">
      {!isLoading ? (
        <button
          onClick={onEdit}
          disabled={!canSubmit}
          className={clsx(
            'flex-1 py-2.5 rounded-lg font-medium text-sm transition-all',
            canSubmit
              ? 'bg-[var(--color-accent)] hover:bg-[var(--color-accent-hover)] text-white'
              : 'bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)] cursor-not-allowed',
          )}
        >
          Edit!
        </button>
      ) : (
        <button
          onClick={onCancel}
          className="flex-1 py-2.5 rounded-lg font-medium text-sm bg-[var(--color-danger)] hover:bg-red-500 text-white transition-all"
        >
          Cancel
        </button>
      )}
      <button
        onClick={onClear}
        disabled={isLoading}
        className="px-4 py-2.5 rounded-lg text-sm border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)] hover:text-[var(--color-text-primary)] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        title="Clear images"
      >
        <Trash2 size={16} />
      </button>
    </div>
  );
}
