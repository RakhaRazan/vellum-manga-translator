import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  ArrowDown,
  ArrowUp,
  Check,
  CheckCircle2,
  Cpu,
  Download,
  Eraser,
  Eye,
  EyeOff,
  FileText,
  Globe,
  HardDrive,
  Image as ImageIcon,
  Languages,
  Layers,
  Loader2,
  Monitor,
  Moon,
  Palette,
  Plus,
  RefreshCw,
  RotateCcw,
  Save,
  Search,
  Server,
  Sliders,
  Sparkles,
  Sun,
  Trash2,
  Type,
  Upload,
  Wifi,
  X,
} from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import useEditorStore, {
  DEFAULT_API_BASE_URL,
  DEFAULT_DETECTION_MODELS,
  DEFAULT_FONT_FALLBACK_CHAIN,
  DEFAULT_INPAINTING_MODELS,
  DEFAULT_OCR_MODELS,
  DEFAULT_PROVIDER_PRESETS,
  DEFAULT_TYPESETTING_CONFIG,
  buildDefaultProviderProfiles,
  ensureFontFaceLoaded,
  formatFontFamilyChain,
  normalizeOptionalThreshold,
} from '../../store/useEditorStore.js';
import { formatExecutionProvider } from '../StatusBar.jsx';

import {
  TABS,
  THEME_CARDS,
  ACCENT_PRESETS,
  DENSITY_OPTIONS,
  LAMA_TILE_SIZES,
  READING_ORDER_OPTIONS,
  BACKEND_OPTIONS,
  REASONING_LEVELS,
  SOURCE_LANGUAGE_OPTIONS,
  INSTRUCTION_PRESETS,
  FONT_STYLE_FILTERS,
  ToggleSwitch,
  SectionCard,
  mergePipelineModelOptions,
  ModelStatusBadge,
  normalizeExecMode,
  StageHardwareBadge,
  glossaryDictToRows,
  glossaryRowsToDict,
} from './SettingsPrimitives.jsx';

export default function AppearanceTab({ ctx }) {
  const {
    theme,
    resolvedTheme,
    accentColor,
    uiDensity,
    setTheme,
    setAccentColor,
    setUiDensity,
  } = ctx;

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
              </div>
  );
}
