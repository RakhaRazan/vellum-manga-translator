import React, { useMemo } from 'react';
import {
  Cpu,
  Languages,
  Layers,
  ListOrdered,
  Loader2,
  Play,
  Settings,
  Square,
} from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import useEditorStore, { DEFAULT_PROVIDER_PRESETS } from '../store/useEditorStore.js';
import { formatShortModelName } from './StatusBar.jsx';
import SeriesProfilePicker from './SeriesProfilePicker.jsx';

const SOURCE_LANGUAGES = [
  { value: 'ja', label: 'JA · Manga' },
  { value: 'ko', label: 'KO · Manhwa' },
  { value: 'zh', label: 'ZH · Manhua' },
];

const TRANSLATION_TONES = [
  { value: 'standard', label: 'Standard' },
  { value: 'casual', label: 'Casual' },
  { value: 'adult_doujinshi', label: 'Adult / Doujinshi (Uncensored)' },
];

const PIPELINE_STAGES = [
  { key: 'detect', label: 'Detect', title: 'Stage 1: Comic Text Detector (ONNX)' },
  { key: 'ocr', label: 'OCR', title: 'Stage 2: Multi-Language OCR Router' },
  { key: 'inpaint', label: 'Inpaint', title: 'Stage 3: Tile-Based LaMa Inpainting' },
  { key: 'translate', label: 'Translate', title: 'Stage 4: Context-Aware LLM Translation' },
];

/**
 * Compact 36px native studio control bar mounted directly below `MenuBar`.
 * Controls active LLM model, processing scope, pipeline stage toggles,
 * source/target languages, primary Run trigger, and Settings gear.
 */
export default function FrontControlBar({
  onRun,
  onCancel,
  onOpenSettings,
  onOpenBatchQueue,
  activeProfileSlug = '',
  onSelectProfileSlug,
  defaultSeriesTitle = '',
  profilesVersion = 0,
  onProfilesChanged,
  isProcessing = false,
  isStopping = false,
  isWorkspaceUnlocked = true,
  skipCompleted = true,
  onToggleSkipCompleted,
}) {
  const {
    apiBaseUrl,
    model,
    provider,
    scope,
    enabledStages,
    sourceLang,
    targetLang,
    translationTone,
    supportedTargetLanguages,
    lmStudio,
    activeProviderSummary,
    pagesCount,
    selectedPagesCount,
    hasActivePage,
    availableProviderPresets,
    providerModelsCache,
    setActiveProvider,
    setActiveModel,
    setPipelineScope,
    togglePipelineStage,
    setSourceLang,
    setTargetLang,
    setTranslationTone,
  } = useEditorStore(
    useShallow((s) => ({
      model: s.pipelineConfig.model,
      provider: s.pipelineConfig.provider,
      scope: s.pipelineConfig.scope,
      enabledStages: s.pipelineConfig.enabledStages,
      sourceLang: s.pipelineConfig.sourceLang,
      targetLang: s.pipelineConfig.targetLang,
      translationTone: s.pipelineConfig.translationTone || 'standard',
      supportedTargetLanguages: s.supportedTargetLanguages,
      lmStudio: s.runtimeStatus.lmStudio,
      activeProviderSummary: s.runtimeStatus.activeProviderSummary,
      pagesCount: s.pages.length,
      selectedPagesCount: s.selectedPageIds.length,
      hasActivePage: Boolean(s.activePageId),
      availableProviderPresets: s.availableProviderPresets,
      providerModelsCache: s.providerModelsCache,
      apiBaseUrl: s.apiBaseUrl,
      setActiveProvider: s.setActiveProvider,
      setActiveModel: s.setActiveModel,
      setPipelineScope: s.setPipelineScope,
      togglePipelineStage: s.togglePipelineStage,
      setSourceLang: s.setSourceLang,
      setTargetLang: s.setTargetLang,
      setTranslationTone: s.setTranslationTone,
    }))
  );

  const activeProviderId = provider || 'none';
  const providerList = useMemo(() => {
    const presets =
      Array.isArray(availableProviderPresets) && availableProviderPresets.length > 0
        ? availableProviderPresets
        : DEFAULT_PROVIDER_PRESETS;
    return presets;
  }, [availableProviderPresets]);

  // Deduplicated model dropdown options from live server + active config
  const modelOptions = useMemo(() => {
    const seen = new Set();
    const list = [];
    const addCandidate = (id) => {
      if (!id || typeof id !== 'string') return;
      const trimmed = id.trim();
      if (!trimmed || seen.has(trimmed)) return;
      seen.add(trimmed);
      list.push(trimmed);
    };

    addCandidate(model);
    addCandidate(activeProviderSummary?.model);
    const cached = providerModelsCache?.[activeProviderId];
    if (Array.isArray(cached?.models)) {
      cached.models.forEach(addCandidate);
    }
    if (activeProviderId === 'lmstudio') {
      addCandidate(lmStudio?.activeModel);
      addCandidate(lmStudio?.resolvedModelId);
      if (Array.isArray(lmStudio?.availableModels)) {
        lmStudio.availableModels.forEach((entry) => {
          if (typeof entry === 'string') {
            addCandidate(entry);
          } else if (entry && typeof entry.id === 'string') {
            addCandidate(entry.id);
          }
        });
      }
    }
    return list;
  }, [model, activeProviderSummary, providerModelsCache, activeProviderId, lmStudio]);

  const isLocalLmStudio = activeProviderId === 'lmstudio';
  const isNoLlm = activeProviderId === 'none';
  const isModelConnected = isNoLlm
    ? true
    : isLocalLmStudio
    ? Boolean(lmStudio?.connected)
    : Boolean(model);

  const activeStageList = PIPELINE_STAGES.filter((s) =>
    Boolean(enabledStages?.[s.key])
  );

  const hasTargetForScope =
    scope === 'project'
      ? pagesCount > 0
      : scope === 'selected_pages'
      ? selectedPagesCount > 0 || hasActivePage
      : hasActivePage;

  const canRun =
    isWorkspaceUnlocked &&
    !isProcessing &&
    hasTargetForScope &&
    activeStageList.length > 0;

  const scopeDisplayLabel =
    scope === 'project'
      ? `Project (${pagesCount})`
      : scope === 'selected_pages'
      ? `Selected Pages (${selectedPagesCount || (hasActivePage ? 1 : 0)})`
      : 'Current Page';

  const runTooltip =
    activeStageList.length === 0
      ? 'Enable at least one pipeline stage to run'
      : !hasTargetForScope
      ? 'Import a comic page first'
      : `Run ${activeStageList.map((s) => s.label).join(' → ')} on ${scopeDisplayLabel}`;

  return (
    <div
      role="toolbar"
      aria-label="Pipeline Control Bar"
      className="h-[var(--studio-controlbar-h)] min-h-[var(--studio-controlbar-h)] max-h-[var(--studio-controlbar-h)] shrink-0 border-b border-studio-border bg-studio-panel px-[var(--studio-pad-x)] flex items-center justify-between gap-[var(--studio-gap)] z-30 select-none overflow-x-auto transition-[height]"
    >
      {/* Left: 1-Click Provider Switcher + Active Model Selector + Scope Selector + Stage Pill Toggles */}
      <div className="flex items-center gap-2 min-w-0 shrink-0">
        {/* 1-Click Provider Switcher (preserves saved keys/URLs per provider) */}
        <div
          className="inline-flex items-center gap-1.5 h-6 rounded-studio border border-studio-border bg-studio-surface px-2 shrink-0"
          title="1-Click LLM Provider Switcher (saved keys & URLs are preserved)"
        >
          <span
            className={`h-1.5 w-1.5 rounded-full shrink-0 ${
              isNoLlm
                ? 'bg-studio-text-muted'
                : isModelConnected
                ? 'bg-studio-success'
                : 'bg-studio-danger'
            }`}
          />
          <select
            aria-label="Active Provider"
            data-testid="front-control-provider-select"
            value={activeProviderId}
            onChange={(e) => setActiveProvider(e.target.value)}
            disabled={isProcessing}
            className="bg-transparent text-[11px] font-semibold text-studio-text focus:outline-none cursor-pointer disabled:opacity-50 max-w-[125px] truncate"
          >
            {providerList.map((p) => (
              <option key={p.id} value={p.id}>
                {p.id === 'none' ? 'No-LLM (Manual)' : p.name}
              </option>
            ))}
          </select>
        </div>

        {/* Active Model Selector */}
        {!isNoLlm && (
          <div
            className="inline-flex items-center gap-1.5 h-6 rounded-studio border border-studio-border bg-studio-surface px-2 shrink-0"
            title={`Active LLM Model: ${model || 'None'} (${isModelConnected ? 'Connected' : 'Offline'})`}
          >
            <Cpu className="h-3 w-3 text-studio-text-muted shrink-0" />
            <select
              aria-label="Active Model"
              value={model || modelOptions[0] || ''}
              onChange={(e) => setActiveModel(e.target.value)}
              disabled={isProcessing}
              className="bg-transparent text-[11px] font-medium text-studio-text focus:outline-none cursor-pointer disabled:opacity-50 max-w-[140px] lg:max-w-[180px] truncate"
            >
              {modelOptions.length === 0 && (
                <option value="">Select model…</option>
              )}
              {modelOptions.map((m) => (
                <option key={m} value={m}>
                  {formatShortModelName(m)}
                </option>
              ))}
            </select>
          </div>
        )}

        <span className="h-4 w-px bg-studio-border shrink-0" aria-hidden="true" />

        {/* Scope Selector */}
        <div
          className="inline-flex items-center gap-1.5 h-6 rounded-studio border border-studio-border bg-studio-surface px-2 shrink-0"
          title="Execution scope for Run"
        >
          <Layers className="h-3 w-3 text-studio-accent shrink-0" />
          <select
            aria-label="Execution Scope"
            value={scope}
            onChange={(e) => setPipelineScope(e.target.value)}
            disabled={isProcessing}
            className="bg-transparent text-[11px] font-medium text-studio-text focus:outline-none cursor-pointer disabled:opacity-50"
          >
            <option value="current_page">Current Page</option>
            <option value="selected_pages">
              Selected Pages{selectedPagesCount > 0 ? ` (${selectedPagesCount})` : ''}
            </option>
            <option value="project">
              Project{pagesCount > 0 ? ` (${pagesCount})` : ''}
            </option>
          </select>
        </div>

        <span className="h-4 w-px bg-studio-border shrink-0" aria-hidden="true" />

        {/* Pipeline Stage Pill Toggles */}
        <div
          className="inline-flex items-center gap-1 shrink-0"
          role="group"
          aria-label="Pipeline Stages"
        >
          {PIPELINE_STAGES.map((stage) => {
            const isEnabled = Boolean(enabledStages?.[stage.key]);
            return (
              <button
                key={stage.key}
                type="button"
                onClick={() => togglePipelineStage(stage.key)}
                disabled={isProcessing}
                aria-pressed={isEnabled}
                title={stage.title}
                className={`inline-flex items-center gap-1.5 h-6 px-2 rounded-studio border text-[11px] font-medium transition-colors cursor-pointer disabled:opacity-50 ${
                  isEnabled
                    ? 'border-studio-accent/60 bg-studio-accent/15 text-studio-accent'
                    : 'border-studio-border bg-studio-surface text-studio-text-muted hover:text-studio-text hover:bg-studio-hover'
                }`}
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full shrink-0 ${
                    isEnabled ? 'bg-studio-accent' : 'bg-studio-border-strong'
                  }`}
                />
                <span>{stage.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Right: Tone Selector + Source & Target Language Dropdowns + Primary Run Button + Settings Gear */}
      <div className="flex items-center gap-2 shrink-0">
        {/* Tone / Genre Selector */}
        <div
          className="inline-flex items-center gap-1 h-6 rounded-studio border border-studio-border bg-studio-surface px-2"
          title="Translation Tone / Genre"
        >
          <span className="text-[10px] font-medium text-studio-text-muted">
            Tone
          </span>
          <select
            aria-label="Translation Tone"
            value={translationTone}
            onChange={(e) => setTranslationTone(e.target.value)}
            disabled={!isWorkspaceUnlocked || isProcessing}
            className="bg-transparent text-[11px] font-medium text-studio-text focus:outline-none cursor-pointer disabled:opacity-50 max-w-[140px]"
          >
            {TRANSLATION_TONES.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </select>
        </div>

        {/* Series Profile Picker */}
        <SeriesProfilePicker
          apiBaseUrl={apiBaseUrl}
          activeProfileSlug={activeProfileSlug}
          onSelectProfileSlug={onSelectProfileSlug}
          defaultTitle={defaultSeriesTitle}
          disabled={!isWorkspaceUnlocked || isProcessing}
          profilesVersion={profilesVersion}
          onProfilesChanged={onProfilesChanged}
        />

        {/* Source → Target Languages */}
        <div className="inline-flex items-center gap-1 h-6 rounded-studio border border-studio-border bg-studio-surface px-2">
          <Languages className="h-3 w-3 text-studio-accent shrink-0" />

          <select
            aria-label="Source Language"
            value={sourceLang}
            onChange={(e) => setSourceLang(e.target.value)}
            disabled={!isWorkspaceUnlocked || isProcessing}
            className="bg-transparent text-[11px] font-medium text-studio-text focus:outline-none cursor-pointer disabled:opacity-50"
          >
            {SOURCE_LANGUAGES.map((lang) => (
              <option key={lang.value} value={lang.value}>
                {lang.label}
              </option>
            ))}
          </select>

          <span className="text-studio-text-muted text-[11px] font-mono px-0.5">
            →
          </span>

          <select
            aria-label="Target Language"
            value={targetLang}
            onChange={(e) => setTargetLang(e.target.value)}
            disabled={!isWorkspaceUnlocked || isProcessing}
            className="bg-transparent text-[11px] font-semibold text-studio-accent focus:outline-none cursor-pointer disabled:opacity-50 max-w-[120px]"
          >
            {(supportedTargetLanguages || []).map((lang) => {
              const code = lang.code || lang.value;
              const label = lang.name || lang.label || code;
              return (
                <option key={code} value={code}>
                  {label}
                </option>
              );
            })}
          </select>
        </div>

        {/* Skip Completed (Resume) Toggle for Project batch */}
        {scope === 'project' && onToggleSkipCompleted && (
          <label
            title="Skip pages already marked done so a 500-page batch can resume after a stop or crash"
            className="flex items-center gap-1.5 text-[11px] text-studio-text-secondary hover:text-studio-text cursor-pointer select-none px-1.5 py-0.5 rounded border border-studio-border bg-studio-bg-deep/60 shrink-0"
          >
            <input
              type="checkbox"
              checked={Boolean(skipCompleted)}
              onChange={onToggleSkipCompleted}
              disabled={isProcessing}
              className="accent-studio-accent h-3 w-3 cursor-pointer"
            />
            <span>Skip Done</span>
          </label>
        )}

        {/* Primary Run Button */}
        <button
          type="button"
          data-testid="front-control-run-btn"
          onClick={onRun}
          disabled={!canRun}
          title={runTooltip}
          className="studio-btn-accent h-6 px-3 text-[11px] shrink-0"
        >
          {isProcessing ? (
            <>
              <Loader2 className="h-3 w-3 animate-spin" />
              <span>Running…</span>
            </>
          ) : (
            <>
              <Play className="h-3 w-3 fill-current" />
              <span>Run</span>
            </>
          )}
        </button>

        {/* Stop / Cancel Batch Button */}
        {isProcessing && onCancel && (
          <button
            type="button"
            data-testid="front-control-cancel-btn"
            onClick={onCancel}
            title="Stop batch immediately and preserve completed pages"
            className="h-6 px-2.5 text-[11px] font-medium rounded flex items-center gap-1 bg-red-600 hover:bg-red-500 text-white transition-colors cursor-pointer shrink-0"
          >
            <Square className="h-2.5 w-2.5 fill-current" />
            <span>{isStopping ? 'Stopping…' : 'Stop'}</span>
          </button>
        )}

        {/* Batch Series Queue Button */}
        {onOpenBatchQueue && (
          <button
            type="button"
            onClick={onOpenBatchQueue}
            aria-label="Batch Series Queue"
            title="Batch Series Queue (Multi-Manga Overnight Pipeline)"
            className="studio-btn h-6 w-6 p-0 shrink-0"
          >
            <ListOrdered className="h-3.5 w-3.5 text-studio-text-secondary" />
          </button>
        )}

        {/* Settings Gear Button */}
        <button
          type="button"
          onClick={onOpenSettings}
          aria-label="Settings"
          data-testid="open-settings-btn"
          title="Studio Settings (Ctrl+,)"
          className="studio-btn h-6 w-6 p-0 shrink-0"
        >
          <Settings className="h-3.5 w-3.5 text-studio-text-secondary" />
        </button>
      </div>
    </div>
  );
}
