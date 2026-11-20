import React, { useRef, useState } from 'react';
import {
  BookOpen,
  Database,
  Download,
  FolderPlus,
  Languages,
  Loader2,
  Monitor,
  Moon,
  Redo2,
  RefreshCw,
  Sun,
  Undo2,
} from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import useEditorStore from '../store/useEditorStore.js';

const SOURCE_LANGUAGES = [
  { value: 'ja', label: 'JA · Manga' },
  { value: 'ko', label: 'KO · Manhwa' },
  { value: 'zh', label: 'ZH · Manhua' },
];

const TARGET_LANGUAGES = [
  { value: 'id', label: 'Indonesian' },
  { value: 'en', label: 'English' },
];

const VIEW_LAYER_ITEMS = [
  { key: 'original', label: 'Raw', title: 'Toggle Original Raw Page Layer' },
  { key: 'inpainted', label: 'Clean', title: 'Toggle LaMa Inpainted Layer' },
  { key: 'mask', label: 'Mask', title: 'Toggle Detection / Inpaint Mask Overlay' },
  { key: 'text', label: 'Text', title: 'Toggle Translated Text Layer' },
  { key: 'boxes', label: 'Boxes', title: 'Toggle Bounding Box Guides' },
];

const THEME_OPTIONS = [
  { value: 'dark', icon: Moon, label: 'Dark', title: 'Sumi Graphite (Dark)' },
  { value: 'light', icon: Sun, label: 'Light', title: 'Manga Paper (Light)' },
  { value: 'system', icon: Monitor, label: 'System', title: 'Follow System Theme' },
];

/**
 * Compact native desktop studio top toolbar (Koharu-inspired layout).
 * Houses multi-page import, Undo/Redo history controls, view layer toggles,
 * language selectors, Re-translate action, Dark/Light/System theme switcher,
 * local model inspector, and full-resolution export.
 */
export default function Toolbar({
  sourceLang,
  onSourceLangChange,
  targetLang,
  onTargetLangChange,
  onImportFiles,
  onRetranslate,
  onExportImage,
  onOpenModelsModal,
  isProcessing = false,
  isRetranslating = false,
  canRetranslate = false,
  canExport = false,
  isWorkspaceUnlocked = true,
  modelsStatus = null,
  batchStats = { total: 0, completed: 0, processing: 0, queued: 0 },
}) {
  const fileInputRef = useRef(null);
  const [exportFormat, setExportFormat] = useState('png');

  const {
    canUndo,
    canRedo,
    undoCount,
    redoCount,
    undo,
    redo,
    viewLayers,
    toggleViewLayer,
    theme,
    setTheme,
  } = useEditorStore(
    useShallow((s) => ({
      canUndo: s.history.past.length > 0,
      canRedo: s.history.future.length > 0,
      undoCount: s.history.past.length,
      redoCount: s.history.future.length,
      undo: s.undo,
      redo: s.redo,
      viewLayers: s.viewLayers,
      toggleViewLayer: s.toggleViewLayer,
      theme: s.theme,
      setTheme: s.setTheme,
    }))
  );

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0 && onImportFiles) {
      onImportFiles(files);
    }
    e.target.value = '';
  };

  const installedModelsCount = modelsStatus?.models
    ? Object.values(modelsStatus.models).filter((m) => m.installed).length
    : 4;
  const totalModelsCount = modelsStatus?.models
    ? Object.keys(modelsStatus.models).length
    : 4;

  return (
    <header className="h-11 shrink-0 border-b border-studio-border bg-studio-panel px-2.5 flex items-center justify-between gap-2 z-30 select-none">
      {/* Left: Studio Mark + Import + Undo/Redo */}
      <div className="flex items-center gap-2 min-w-0">
        <div className="flex items-center gap-2 pr-2 border-r border-studio-border shrink-0">
          <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-studio bg-studio-accent text-studio-accent-fg">
            <BookOpen className="h-3.5 w-3.5 stroke-[2.25]" />
          </div>
          <span className="hidden sm:inline text-xs font-semibold tracking-tight text-studio-text">
            Manga Studio
          </span>
        </div>

        {/* Hidden Multi-File Input */}
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept="image/jpeg,image/png,image/webp,image/*"
          onChange={handleFileChange}
          className="hidden"
          data-testid="import-image-input"
        />

        {/* Import Image Button */}
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={!isWorkspaceUnlocked}
          className="studio-btn-accent h-7 px-2.5 text-xs shrink-0"
          title="Import one or multiple comic pages into the workspace"
        >
          <FolderPlus className="h-3.5 w-3.5" />
          <span>Import</span>
          {batchStats.total > 0 && (
            <span className="ml-0.5 rounded-studio-sm bg-black/20 px-1 py-0.5 font-mono text-[10px] leading-none">
              {batchStats.completed}/{batchStats.total}
            </span>
          )}
        </button>

        {/* Active Batch Queue Indicator */}
        {(batchStats.processing > 0 || batchStats.queued > 0) && (
          <div className="hidden 2xl:inline-flex items-center gap-1.5 h-7 rounded-studio border border-studio-accent/40 bg-studio-accent/10 px-2 text-[11px] font-mono text-studio-accent shrink-0">
            <Loader2 className="h-3 w-3 animate-spin" />
            <span>
              {batchStats.completed}/{batchStats.total}
            </span>
          </div>
        )}

        <span className="h-4 w-px bg-studio-border shrink-0" aria-hidden="true" />

        {/* Undo / Redo History Controls */}
        <div
          className="inline-flex items-center rounded-studio border border-studio-border bg-studio-surface overflow-hidden shrink-0"
          role="group"
          aria-label="History controls"
        >
          <button
            type="button"
            onClick={undo}
            disabled={!canUndo}
            className="h-7 w-7 inline-flex items-center justify-center text-studio-text hover:bg-studio-hover disabled:opacity-35 disabled:pointer-events-none transition-colors cursor-pointer"
            title={canUndo ? `Undo (${undoCount}) — Ctrl+Z` : 'Undo (Ctrl+Z)'}
            aria-label="Undo"
          >
            <Undo2 className="h-3.5 w-3.5" />
          </button>
          <span className="h-4 w-px bg-studio-border" aria-hidden="true" />
          <button
            type="button"
            onClick={redo}
            disabled={!canRedo}
            className="h-7 w-7 inline-flex items-center justify-center text-studio-text hover:bg-studio-hover disabled:opacity-35 disabled:pointer-events-none transition-colors cursor-pointer"
            title={canRedo ? `Redo (${redoCount}) — Ctrl+Y` : 'Redo (Ctrl+Y)'}
            aria-label="Redo"
          >
            <Redo2 className="h-3.5 w-3.5" />
          </button>
        </div>

        {/* View Layer Toggles */}
        <div
          className="hidden xl:inline-flex items-center rounded-studio border border-studio-border bg-studio-surface p-0.5 gap-0.5 shrink-0"
          role="group"
          aria-label="View Layers"
        >
          {VIEW_LAYER_ITEMS.map((layer) => {
            const isActive = Boolean(viewLayers[layer.key]);
            return (
              <button
                key={layer.key}
                type="button"
                onClick={() => toggleViewLayer(layer.key)}
                title={layer.title}
                aria-pressed={isActive}
                className={`h-6 px-2 rounded-studio-sm text-[11px] font-medium transition-colors cursor-pointer ${
                  isActive
                    ? 'bg-studio-accent/15 text-studio-accent'
                    : 'text-studio-text-muted hover:text-studio-text hover:bg-studio-hover'
                }`}
              >
                {layer.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Center: Language Pipeline Selector + Re-translate */}
      <div className="flex items-center gap-1.5 shrink-0">
        <div className="flex items-center gap-1 h-7 rounded-studio border border-studio-border bg-studio-surface px-2">
          <Languages className="h-3.5 w-3.5 text-studio-accent shrink-0" />

          <select
            aria-label="Source Language"
            value={sourceLang}
            onChange={(e) => onSourceLangChange && onSourceLangChange(e.target.value)}
            disabled={!isWorkspaceUnlocked || isProcessing}
            className="bg-transparent text-xs font-medium text-studio-text focus:outline-none cursor-pointer disabled:opacity-50 pr-0.5"
          >
            {SOURCE_LANGUAGES.map((lang) => (
              <option key={lang.value} value={lang.value}>
                {lang.label}
              </option>
            ))}
          </select>

          <span className="text-studio-text-muted text-xs font-mono px-0.5">→</span>

          <select
            aria-label="Target Language"
            value={targetLang}
            onChange={(e) => onTargetLangChange && onTargetLangChange(e.target.value)}
            disabled={!isWorkspaceUnlocked || isProcessing}
            className="bg-transparent text-xs font-semibold text-studio-accent focus:outline-none cursor-pointer disabled:opacity-50 pr-0.5"
          >
            {TARGET_LANGUAGES.map((lang) => (
              <option key={lang.value} value={lang.value}>
                {lang.label}
              </option>
            ))}
          </select>
        </div>

        <button
          type="button"
          onClick={onRetranslate}
          disabled={!isWorkspaceUnlocked || !canRetranslate || isProcessing || isRetranslating}
          className="studio-btn h-7 px-2.5 text-xs font-medium"
          title="Re-translate dialogue blocks on the active page (/api/retranslate)"
        >
          <RefreshCw
            className={`h-3.5 w-3.5 text-studio-accent ${
              isRetranslating || isProcessing ? 'animate-spin' : ''
            }`}
          />
          <span className="hidden md:inline">
            {isRetranslating ? 'Translating…' : 'Re-translate'}
          </span>
        </button>
      </div>

      {/* Right: Theme Switcher (Dark / Light / System) + Local Models + Export */}
      <div className="flex items-center gap-2 shrink-0">
        {/* Dark / Light / System Theme Switcher */}
        <div
          className="inline-flex items-center rounded-studio border border-studio-border bg-studio-surface p-0.5 gap-0.5"
          role="group"
          aria-label="Theme"
        >
          {THEME_OPTIONS.map(({ value, icon: Icon, title }) => {
            const isSelected = theme === value;
            return (
              <button
                key={value}
                type="button"
                onClick={() => setTheme(value)}
                title={title}
                aria-label={title}
                aria-pressed={isSelected}
                className={`h-6 w-6 inline-flex items-center justify-center rounded-studio-sm transition-colors cursor-pointer ${
                  isSelected
                    ? 'bg-studio-accent text-studio-accent-fg'
                    : 'text-studio-text-muted hover:text-studio-text hover:bg-studio-hover'
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
              </button>
            );
          })}
        </div>

        {/* Local ONNX / OCR Models Inspector */}
        <button
          type="button"
          onClick={onOpenModelsModal}
          className="hidden sm:inline-flex studio-btn h-7 px-2 text-xs font-mono"
          title="Inspect local ONNX & OCR model weights"
        >
          <Database className="h-3.5 w-3.5 text-studio-accent" />
          <span className="text-[11px] tabular-nums">
            {installedModelsCount}/{totalModelsCount}
          </span>
        </button>

        {/* Export Split Control (.PNG / .JPG) */}
        <div className="inline-flex items-center h-7 rounded-studio border border-studio-border overflow-hidden bg-studio-surface">
          <select
            aria-label="Export Image Format"
            value={exportFormat}
            onChange={(e) => setExportFormat(e.target.value)}
            disabled={!isWorkspaceUnlocked || !canExport || isProcessing}
            className="h-full bg-studio-surface text-studio-text-secondary font-mono text-[11px] font-medium px-1.5 border-r border-studio-border focus:outline-none cursor-pointer disabled:opacity-40"
            title="Select full-resolution export format (.png or .jpg)"
          >
            <option value="png">.PNG</option>
            <option value="jpg">.JPG</option>
          </select>
          <button
            type="button"
            onClick={() => onExportImage && onExportImage(exportFormat)}
            disabled={!isWorkspaceUnlocked || !canExport || isProcessing}
            className="inline-flex items-center gap-1.5 h-full px-2.5 bg-studio-surface hover:bg-studio-hover text-studio-text hover:text-studio-accent disabled:opacity-40 disabled:pointer-events-none text-xs font-semibold transition-colors cursor-pointer"
            title="Export rendered comic page at full native resolution"
          >
            <Download className="h-3.5 w-3.5 text-studio-accent" />
            <span>Export</span>
          </button>
        </div>
      </div>
    </header>
  );
}
