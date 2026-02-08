interface ToggleProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  info?: string;
}

export function Toggle({ label, checked, onChange, info }: ToggleProps) {
  return (
    <label className="flex items-start gap-3 cursor-pointer group">
      <div className="relative mt-0.5 shrink-0">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="sr-only peer"
        />
        <div className="w-10 h-5 rounded-full bg-[var(--color-border)] peer-checked:bg-[var(--color-accent)] transition-colors" />
        <div className="absolute left-0.5 top-0.5 w-4 h-4 rounded-full bg-white transition-transform peer-checked:translate-x-5" />
      </div>
      <div>
        <span className="text-sm text-[var(--color-text-primary)] group-hover:text-white transition-colors">
          {label}
        </span>
        {info && (
          <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">{info}</p>
        )}
      </div>
    </label>
  );
}
