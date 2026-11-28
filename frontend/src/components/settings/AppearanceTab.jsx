import { useState } from 'react';
import {
  Check,
  RotateCcw,
  Type,
} from 'lucide-react';

import {
  THEME_CARDS,
  ACCENT_PRESETS,
  DENSITY_OPTIONS,
  SectionCard,
} from './SettingsPrimitives.jsx';
import { DEFAULT_SHORTCUTS } from '../ShortcutsModal.jsx';

const SHORTCUT_FIELDS = [
  ['nextPage', 'Next Page'],
  ['prevPage', 'Previous Page'],
  ['nextIssue', 'Next QC Issue'],
  ['findReplace', 'Find & Replace'],
  ['toggleCompare', 'Hold Original'],
  ['loupe', 'Loupe Magnifier'],
  ['readingPreview', 'Reading Preview'],
  ['copyPage', 'Copy Page'],
  ['runPipeline', 'Run Pipeline'],
];

export default function AppearanceTab({ ctx }) {
  const {
    theme,
    resolvedTheme,
    accentColor,
    uiDensity,
    setTheme,
    setAccentColor,
    setUiDensity,
    apiBaseUrl,
  } = ctx;

  const [shortcuts, setShortcuts] = useState(() => {
    try {
      const raw = localStorage.getItem('vellum.shortcuts');
      return raw ? { ...DEFAULT_SHORTCUTS, ...JSON.parse(raw) } : { ...DEFAULT_SHORTCUTS };
    } catch {
      return { ...DEFAULT_SHORTCUTS };
    }
  });

  const persistShortcuts = (nextMap) => {
    setShortcuts(nextMap);
    try {
      localStorage.setItem('vellum.shortcuts', JSON.stringify(nextMap));
      window.dispatchEvent(new CustomEvent('vellum:shortcuts-updated', { detail: nextMap }));
    } catch {
      // ignore storage errors
    }
    if (apiBaseUrl) {
      fetch(`${apiBaseUrl}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ keyboard_shortcuts: nextMap }),
      }).catch(() => {});
    }
  };

  return (
              <div className="space-y-4" data-testid="settings-panel-appearance">
                <SectionCard
                  title="Color Theme"
                  subtitle="Applies live across the workspace and syncs with your OS prefers-color-scheme when set to Same as System."
                >
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                    {THEME_CARDS.map((card) => {
                      const Icon = card.icon;
                      const isActive = theme === card.id;
                      return (
                        <button
                          key={card.id}
                          type="button"
                          data-testid={`theme-option-${card.id}`}
                          aria-pressed={isActive}
                          onClick={() => setTheme(card.id)}
                          className={`group rounded-studio border p-2.5 text-left transition-colors cursor-pointer space-y-2 ${
                            isActive
                              ? 'border-studio-accent bg-studio-accent/10'
                              : 'border-studio-border bg-studio-panel hover:border-studio-border-strong'
                          }`}
                        >
                          {/* Mini studio window preview */}
                          <div
                            className="h-12 rounded-studio-sm border overflow-hidden flex flex-col"
                            style={{
                              background: card.canvas,
                              borderColor: card.border,
                            }}
                          >
                            <div
                              className="h-3 border-b flex items-center gap-1 px-1.5"
                              style={{
                                background: card.panel,
                                borderColor: card.border,
                              }}
                            >
                              <span className="h-1.5 w-1.5 rounded-full bg-studio-accent" />
                              <span className="h-1 w-6 rounded-full bg-studio-border-strong/80" />
                            </div>
                            <div className="flex-1 flex">
                              <div
                                className="w-5 border-r"
                                style={{
                                  background: card.panel,
                                  borderColor: card.border,
                                }}
                              />
                              <div className="flex-1 flex items-center justify-center">
                                <span className="h-3 w-8 rounded-[2px] bg-studio-accent/80" />
                              </div>
                            </div>
                          </div>

                          <div className="flex items-center justify-between gap-1.5">
                            <div className="flex items-center gap-1.5 min-w-0">
                              <Icon
                                className={`h-3.5 w-3.5 shrink-0 ${
                                  isActive
                                    ? 'text-studio-accent'
                                    : 'text-studio-text-muted'
                                }`}
                              />
                              <span className="text-xs font-semibold text-studio-text truncate">
                                {card.label}
                              </span>
                            </div>
                            {isActive && (
                              <Check className="h-3.5 w-3.5 text-studio-accent shrink-0" />
                            )}
                          </div>
                          <p className="text-[10px] text-studio-text-muted leading-tight">
                            {card.id === 'system'
                              ? `Currently resolved: ${resolvedTheme}`
                              : card.subtitle}
                          </p>
                        </button>
                      );
                    })}
                  </div>
                </SectionCard>

                <SectionCard
                  title="Accent Color Preset"
                  subtitle="Highlights active stages, selected layers, primary actions, and focus rings live."
                >
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                    {ACCENT_PRESETS.map((preset) => {
                      const isActive = accentColor === preset.id;
                      const swatch =
                        resolvedTheme === 'light'
                          ? preset.lightHex
                          : preset.darkHex;
                      return (
                        <button
                          key={preset.id}
                          type="button"
                          data-testid={`accent-preset-${preset.id}`}
                          aria-pressed={isActive}
                          onClick={() => setAccentColor(preset.id)}
                          className={`rounded-studio border p-2.5 text-left transition-colors cursor-pointer flex items-center gap-2.5 ${
                            isActive
                              ? 'border-studio-accent bg-studio-accent/10'
                              : 'border-studio-border bg-studio-panel hover:border-studio-border-strong'
                          }`}
                        >
                          <span
                            className="h-6 w-6 rounded-studio-sm shrink-0 border border-black/20 flex items-center justify-center shadow-inner"
                            style={{ backgroundColor: swatch }}
                          >
                            {isActive && (
                              <Check className="h-3.5 w-3.5 text-white drop-shadow" />
                            )}
                          </span>
                          <div className="min-w-0">
                            <div className="text-xs font-semibold text-studio-text leading-tight">
                              {preset.label}
                            </div>
                            <div className="text-[10px] font-mono text-studio-text-muted uppercase">
                              {swatch}
                            </div>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </SectionCard>

                <SectionCard
                  title="UI Density"
                  subtitle="Adjusts toolbar spacing and control ergonomics across the editor."
                >
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                    {DENSITY_OPTIONS.map((opt) => {
                      const isActive = uiDensity === opt.id;
                      return (
                        <button
                          key={opt.id}
                          type="button"
                          data-testid={`density-option-${opt.id}`}
                          aria-pressed={isActive}
                          onClick={() => setUiDensity(opt.id)}
                          className={`rounded-studio border p-3 text-left transition-colors cursor-pointer flex items-start justify-between gap-2 ${
                            isActive
                              ? 'border-studio-accent bg-studio-accent/10'
                              : 'border-studio-border bg-studio-panel hover:border-studio-border-strong'
                          }`}
                        >
                          <div>
                            <div className="text-xs font-semibold text-studio-text">
                              {opt.label}
                            </div>
                            <div className="text-[11px] text-studio-text-muted mt-0.5">
                              {opt.desc}
                            </div>
                          </div>
                          <span
                            className={`mt-0.5 h-4 w-4 rounded-full border flex items-center justify-center shrink-0 ${
                              isActive
                                ? 'border-studio-accent bg-studio-accent text-studio-accent-fg'
                                : 'border-studio-border-strong'
                            }`}
                          >
                            {isActive && <Check className="h-2.5 w-2.5 stroke-[3]" />}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </SectionCard>

                <SectionCard
                  title="Keyboard Shortcuts"
                  subtitle="Click any shortcut input and press your desired key combination, or reset to defaults. Press ? anywhere for the cheat sheet."
                >
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {SHORTCUT_FIELDS.map(([key, label]) => (
                      <div key={key} className="flex items-center justify-between gap-2 text-xs">
                        <span className="text-studio-text-secondary truncate">{label}</span>
                        <input
                          type="text"
                          value={shortcuts[key] || ''}
                          onKeyDown={(e) => {
                            e.preventDefault();
                            if (['Control', 'Shift', 'Alt', 'Meta'].includes(e.key)) return;
                            const parts = [];
                            if (e.ctrlKey || e.metaKey) parts.push('Ctrl');
                            if (e.altKey) parts.push('Alt');
                            if (e.shiftKey) parts.push('Shift');
                            const mainKey = e.key.length === 1 ? e.key.toUpperCase() : e.key;
                            parts.push(mainKey);
                            persistShortcuts({ ...shortcuts, [key]: parts.join('+') });
                          }}
                          onChange={() => {}}
                          className="studio-input w-28 h-7 px-2 font-mono text-[11px] text-center cursor-pointer"
                        />
                      </div>
                    ))}
                  </div>
                  <div className="pt-2 flex justify-end">
                    <button
                      type="button"
                      onClick={() => persistShortcuts({ ...DEFAULT_SHORTCUTS })}
                      className="studio-btn h-7 px-2.5 text-[11px] gap-1"
                    >
                      <RotateCcw className="h-3 w-3" />
                      <span>Reset Defaults</span>
                    </button>
                  </div>
                </SectionCard>
              </div>
  );
}
