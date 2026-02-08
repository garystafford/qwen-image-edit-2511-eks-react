import { useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { Slider } from './Slider';
import { Toggle } from './Toggle';
import type { AdvancedSettings as SettingsType } from '../api/types';
import clsx from 'clsx';

interface AdvancedSettingsProps {
  settings: SettingsType;
  onChange: (settings: SettingsType) => void;
}

function gcd(a: number, b: number): number {
  return b === 0 ? a : gcd(b, a % b);
}

function aspectRatio(w: number, h: number): string {
  const d = gcd(w, h);
  return `${w / d}:${h / d}`;
}

export function AdvancedSettings({ settings, onChange }: AdvancedSettingsProps) {
  const [isOpen, setIsOpen] = useState(false);

  const update = <K extends keyof SettingsType>(key: K, value: SettingsType[K]) => {
    onChange({ ...settings, [key]: value });
  };

  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg-secondary)]">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-4 py-3 text-sm text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors"
      >
        <span>Advanced Settings</span>
        <ChevronDown
          size={16}
          className={clsx('transition-transform', isOpen && 'rotate-180')}
        />
      </button>

      {isOpen && (
        <div className="px-4 pb-4 space-y-5 border-t border-[var(--color-border)] pt-4">
          <Toggle
            label="Style Reference Mode"
            checked={settings.style_reference_mode}
            onChange={(v) => update('style_reference_mode', v)}
            info="When enabled: all but last image = style references, returns only edited last image."
          />

          <div className="space-y-1">
            <label className="text-sm text-[var(--color-text-secondary)]">
              Negative Prompt
            </label>
            <textarea
              value={settings.negative_prompt}
              onChange={(e) => update('negative_prompt', e.target.value)}
              rows={2}
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-tertiary)] border border-[var(--color-border)] text-sm text-[var(--color-text-primary)] placeholder:text-[var(--color-text-secondary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-accent)]/50 resize-none"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <Slider
              label="Seed"
              value={settings.seed}
              onChange={(v) => update('seed', v)}
              min={0}
              max={2147483647}
              step={1}
            />
            <div className="flex items-end pb-1">
              <Toggle
                label="Randomize"
                checked={settings.randomize_seed}
                onChange={(v) => update('randomize_seed', v)}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <Slider
              label="Guidance Scale"
              value={settings.guidance_scale}
              onChange={(v) => update('guidance_scale', v)}
              min={1.0}
              max={10.0}
              step={0.1}
            />
            <Slider
              label="Inference Steps"
              value={settings.num_inference_steps}
              onChange={(v) => update('num_inference_steps', v)}
              min={1}
              max={50}
              step={1}
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <Slider
              label="Height"
              value={settings.height}
              onChange={(v) => update('height', v)}
              min={256}
              max={2048}
              step={8}
            />
            <Slider
              label="Width"
              value={settings.width}
              onChange={(v) => update('width', v)}
              min={256}
              max={2048}
              step={8}
            />
          </div>
          <p className="text-xs text-[var(--color-text-secondary)] text-right -mt-3">
            Output: {settings.width}&times;{settings.height} ({aspectRatio(settings.width, settings.height)})
          </p>

          <Slider
            label="Images per Prompt"
            value={settings.num_images_per_prompt}
            onChange={(v) => update('num_images_per_prompt', v)}
            min={1}
            max={4}
            step={1}
          />
        </div>
      )}
    </div>
  );
}
