import React from 'react';
import { Cpu, Crosshair, Loader2, Maximize2, Server } from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import useEditorStore from '../store/useEditorStore.js';

const PROVIDER_LABELS = {
  lmstudio: 'LM Studio',
  ollama: 'Ollama',
  openai: 'OpenAI',
  gemini: 'Gemini',
  anthropic: 'Anthropic',
  deepseek: 'DeepSeek',
  openrouter: 'OpenRouter',
  custom: 'Custom',
};

export function formatExecutionProvider(provider) {
  if (!provider) {
    return { label: 'Detecting…', short: '—', isGpu: false };
  }
  if (provider === 'CUDAExecutionProvider') {
    return { label: 'GPU · CUDA', short: 'CUDA', isGpu: true };
  }
  if (provider === 'DmlExecutionProvider') {
    return { label: 'GPU · DirectML', short: 'DirectML', isGpu: true };
  }
  if (provider === 'CPUExecutionProvider') {
    return { label: 'CPU', short: 'CPU', isGpu: false };
  }
  return {
    label: provider.replace(/ExecutionProvider$/i, ''),
    short: provider.replace(/ExecutionProvider$/i, ''),
    isGpu: /cuda|dml|tensorrt|rocm|coreml|gpu/i.test(provider),
  };
}

export function formatShortModelName(modelName) {
  if (!modelName) return 'No Model';
  const lower = String(modelName).toLowerCase();
  if (lower.includes('gemma 4 e2b') || lower.includes('gemma-4-e2b')) {
    return 'Gemma 4 E2B';
  }
  return modelName.length > 28 ? `${modelName.slice(0, 26)}…` : modelName;
}

/**
 * Compact 24px native desktop studio bottom status bar.
 * Displays live zoom %, cursor XY coordinates, active ONNX GPU/CPU provider,
 * and active LLM provider/model telemetry.
 */
export default function StatusBar({
  onResetZoom,
  onRefreshLlmStatus,
  isPollingLlm = false,
}) {
  const {
    zoom,
    cursorX,
    cursorY,
    executionProvider,
    lmStudio,
    activeProviderSummary,
    pipelineProvider,
    pipelineModel,
    pipelineBaseUrl,
    activePage,
    undoCount,
  } = useEditorStore(
    useShallow((s) => ({
      zoom: s.canvasView.zoom,
      cursorX: s.canvasView.cursorX,
      cursorY: s.canvasView.cursorY,
      executionProvider: s.runtimeStatus.executionProvider,
      lmStudio: s.runtimeStatus.lmStudio,
      activeProviderSummary: s.runtimeStatus.activeProviderSummary,
      pipelineProvider: s.pipelineConfig.provider,
      pipelineModel: s.pipelineConfig.model,
      pipelineBaseUrl: s.pipelineConfig.baseUrl,
      activePage: s.pages.find((p) => p.id === s.activePageId) || null,
      undoCount: s.history.past.length,
    }))
  );

  const zoomPercent = Math.round((zoom || 1) * 100);
  const hasCursor =
    typeof cursorX === 'number' && typeof cursorY === 'number';
  const providerMeta = formatExecutionProvider(executionProvider);

  const activeProviderId =
    activeProviderSummary?.provider || pipelineProvider || 'lmstudio';
  const providerDisplayName =
    PROVIDER_LABELS[activeProviderId] || activeProviderId;

  const isLocalLmStudio = activeProviderId === 'lmstudio';
  const isLlmConnected = isLocalLmStudio
    ? Boolean(lmStudio?.connected)
    : Boolean(activeProviderSummary?.model || pipelineModel);

  const fullModelName =
    (isLocalLmStudio
      ? lmStudio?.activeModel || lmStudio?.resolvedModelId
      : activeProviderSummary?.model) ||
    activePage?.model ||
    pipelineModel ||
    '';

  const shortModelLabel = isLlmConnected
    ? formatShortModelName(fullModelName)
    : 'Offline';

  const llmEndpoint =
    (isLocalLmStudio ? lmStudio?.endpoint : activeProviderSummary?.base_url) ||
    pipelineBaseUrl ||
    'http://localhost:1235/v1';

  const llmTooltip = isLlmConnected
    ? `LLM Provider: ${providerDisplayName}\nEndpoint: ${llmEndpoint}\nActive Model: ${fullModelName || 'Default'}\nClick to refresh status`
    : `LLM Provider: ${providerDisplayName} (Offline)\nEndpoint: ${llmEndpoint}\n${lmStudio?.error || 'Click to retry connection'}`;

  return (
    <footer
      className="h-6 min-h-[24px] max-h-[24px] shrink-0 border-t border-studio-border bg-studio-panel px-2.5 flex items-center justify-between gap-3 text-[11px] leading-none text-studio-text-secondary select-none z-30"
      role="status"
      aria-label="Studio Status Bar"
    >
      {/* Left: Zoom %, Cursor XY, and Optional Active Document Telemetry */}
      <div className="flex items-center gap-2 min-w-0">
        {/* Zoom % */}
        <button
          type="button"
          onClick={onResetZoom}
          disabled={!onResetZoom}
          className="inline-flex items-center gap-1 font-mono tabular-nums text-studio-text hover:text-studio-accent disabled:hover:text-studio-text transition-colors cursor-pointer disabled:cursor-default"
          title="Canvas zoom level (click to fit page)"
        >
          <Maximize2 className="h-3 w-3 text-studio-text-muted" />
          <span className="w-[38px] text-left">{zoomPercent}%</span>
        </button>

        <span className="h-3 w-px bg-studio-border shrink-0" aria-hidden="true" />

        {/* Cursor XY */}
        <div
          className="inline-flex items-center gap-1.5 font-mono tabular-nums text-studio-text-secondary"
          title="Cursor coordinates on canvas"
        >
          <Crosshair className="h-3 w-3 text-studio-text-muted shrink-0" />
          <span className="inline-flex items-center gap-2">
            <span className="w-[54px] text-left">
              <span className="text-studio-text-muted mr-0.5">X:</span>
              {hasCursor ? cursorX : '—'}
            </span>
            <span className="w-[54px] text-left">
              <span className="text-studio-text-muted mr-0.5">Y:</span>
              {hasCursor ? cursorY : '—'}
            </span>
          </span>
        </div>

        {/* Active page status, dimensions & block count */}
        {activePage && (
          <>
            <span
              className="h-3 w-px bg-studio-border shrink-0"
              aria-hidden="true"
            />
            {activePage.status === 'processing' ? (
              <div
                data-testid="statusbar-processing-message"
                className="inline-flex items-center gap-1.5 text-[11px] font-medium text-studio-accent truncate"
              >
                <Loader2 className="h-3 w-3 animate-spin shrink-0" />
                <span className="truncate">
                  {activePage.activeStage
                    ? `Processing (${activePage.activeStage})…`
                    : 'Processing…'}
                </span>
              </div>
            ) : activePage.status === 'error' ? (
              <div
                data-testid="statusbar-error-message"
                className="inline-flex items-center gap-1.5 text-[11px] font-medium text-studio-danger truncate"
                title={activePage.error || 'Processing failed'}
              >
                <span className="h-1.5 w-1.5 rounded-full bg-studio-danger shrink-0" />
                <span className="truncate">
                  Error — {activePage.error || 'Processing failed'}
                </span>
              </div>
            ) : activePage.status === 'idle' || activePage.status === 'queued' ? (
              <div
                data-testid="statusbar-idle-message"
                className="inline-flex items-center gap-1.5 text-[11px] font-medium text-studio-accent truncate"
                title="Configure Scope, Stages, Languages, and Pipeline Models, then press Run to start processing"
              >
                <span className="h-1.5 w-1.5 rounded-full bg-studio-accent shrink-0" />
                <span className="truncate">Ready — Press Run to process</span>
              </div>
            ) : (
              <div className="hidden md:flex items-center gap-2 font-mono tabular-nums text-studio-text-muted truncate">
                {activePage.dimensions && (
                  <span>
                    {activePage.dimensions.width}×{activePage.dimensions.height}px
                  </span>
                )}
                <span>
                  {activePage.blocks?.length || 0} block
                  {(activePage.blocks?.length || 0) === 1 ? '' : 's'}
                </span>
                {undoCount > 0 && (
                  <span className="text-studio-text-muted/80">
                    · {undoCount} edit{undoCount === 1 ? '' : 's'}
                  </span>
                )}
              </div>
            )}
          </>
        )}
      </div>

      {/* Right: ONNX GPU/CPU Provider + Active LLM Model */}
      <div className="flex items-center gap-2.5 shrink-0">
        {/* Hardware Execution Provider (CUDA / DirectML / CPU) */}
        <div
          className="inline-flex items-center gap-1.5 font-mono text-[11px]"
          title={`ONNX Runtime Execution Provider: ${executionProvider || 'Detecting…'}`}
        >
          <span
            className={`h-1.5 w-1.5 rounded-full shrink-0 ${
              providerMeta.isGpu
                ? 'bg-studio-accent'
                : executionProvider
                ? 'bg-studio-text-muted'
                : 'bg-studio-border-strong'
            }`}
          />
          <Cpu className="h-3 w-3 text-studio-text-muted shrink-0" />
          <span
            className={
              providerMeta.isGpu
                ? 'text-studio-text font-medium'
                : 'text-studio-text-secondary'
            }
          >
            {providerMeta.label}
          </span>
        </div>

        <span className="h-3 w-px bg-studio-border shrink-0" aria-hidden="true" />

        {/* Active LLM Model & Connection Status */}
        <button
          type="button"
          onClick={onRefreshLlmStatus}
          disabled={!onRefreshLlmStatus}
          className="inline-flex items-center gap-1.5 hover:text-studio-text transition-colors cursor-pointer disabled:cursor-default max-w-[260px]"
          title={llmTooltip}
        >
          <span
            className={`h-1.5 w-1.5 rounded-full shrink-0 ${
              isLlmConnected ? 'bg-studio-success' : 'bg-studio-danger'
            }`}
          />
          <Server className="h-3 w-3 text-studio-text-muted shrink-0 hidden sm:inline" />
          <span className="text-studio-text-muted hidden sm:inline shrink-0">
            {providerDisplayName}
          </span>
          <span className="text-studio-border-strong hidden sm:inline">·</span>
          <span
            className={`font-mono truncate ${
              isLlmConnected
                ? 'text-studio-text font-medium'
                : 'text-studio-danger'
            }`}
          >
            {shortModelLabel}
          </span>
          {isPollingLlm && (
            <Loader2 className="h-2.5 w-2.5 animate-spin text-studio-text-muted shrink-0" />
          )}
        </button>
      </div>
    </footer>
  );
}
