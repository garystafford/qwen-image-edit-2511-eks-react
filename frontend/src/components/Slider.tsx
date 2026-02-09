import { useState, useRef, useEffect } from 'react';

interface SliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
}

export function Slider({ label, value, onChange, min, max, step }: SliderProps) {
  const pct = ((value - min) / (max - min)) * 100;
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing) inputRef.current?.select();
  }, [editing]);

  const startEditing = () => {
    setDraft(String(value));
    setEditing(true);
  };

  const commitEdit = () => {
    setEditing(false);
    const parsed = parseFloat(draft);
    if (isNaN(parsed)) return;
    const clamped = Math.min(max, Math.max(min, parsed));
    const snapped = Math.round(clamped / step) * step;
    // Round to avoid floating point drift
    const rounded = parseFloat(snapped.toFixed(10));
    onChange(rounded);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') commitEdit();
    if (e.key === 'Escape') setEditing(false);
  };

  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span className="text-[var(--color-text-secondary)]">{label}</span>
        {editing ? (
          <input
            ref={inputRef}
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={commitEdit}
            onKeyDown={handleKeyDown}
            className="w-24 text-right bg-[var(--color-bg-tertiary)] border border-[var(--color-accent)] rounded px-1 text-[var(--color-text-primary)] font-mono tabular-nums text-sm focus:outline-none"
          />
        ) : (
          <button
            onClick={startEditing}
            className="text-[var(--color-text-primary)] font-mono tabular-nums border-b border-dashed border-[var(--color-text-secondary)]/40 hover:border-[var(--color-accent)] hover:text-[var(--color-accent)] cursor-text transition-colors"
            title="Click to edit"
          >
            {value}
          </button>
        )}
      </div>
      <div className="relative h-6 flex items-center">
        {/* Track background */}
        <div className="absolute inset-x-0 h-1 rounded-full bg-[#5a5a7a]" />
        {/* Filled portion */}
        <div
          className="absolute left-0 h-1 rounded-full bg-[var(--color-accent)]"
          style={{ width: `${pct}%` }}
        />
        {/* Native input on top for interaction */}
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="slider-input absolute inset-0 w-full"
        />
      </div>
    </div>
  );
}
