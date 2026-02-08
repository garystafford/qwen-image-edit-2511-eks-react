interface PromptInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  disabled?: boolean;
}

export function PromptInput({ value, onChange, onSubmit, disabled }: PromptInputProps) {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !disabled) {
      e.preventDefault();
      onSubmit();
    }
  };

  return (
    <div className="space-y-1">
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        rows={4}
        placeholder="Describe the edit instruction..."
        className="w-full px-4 py-3 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] text-sm text-[var(--color-text-primary)] placeholder:text-[var(--color-text-secondary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-accent)]/50 focus:border-[var(--color-accent)] transition-all disabled:opacity-50 resize-y min-h-[44px]"
      />
      <p className="text-[10px] text-[var(--color-text-secondary)]">
        Enter to submit &middot; Shift+Enter for newline &middot; Drag corner to resize
      </p>
    </div>
  );
}
