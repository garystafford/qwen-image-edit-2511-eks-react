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

  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span className="text-[var(--color-text-secondary)]">{label}</span>
        <span className="text-[var(--color-text-primary)] font-mono tabular-nums">{value}</span>
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
