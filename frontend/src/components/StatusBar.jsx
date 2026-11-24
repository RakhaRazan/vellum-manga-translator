import React from 'react';
import { AlertTriangle, Cpu, Crosshair, Loader2, Maximize2, Server } from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import useEditorStore from '../store/useEditorStore.js';

const PROVIDER_LABELS = {
  none: 'Manual (No-LLM)',
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
    return { label: 'Detecting…', short: '-', isGpu: false };
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
  return modelName.length > 28 ? `${modelName.slice(0, 26)}…` : modelName;
}

export function formatShortDetectorName(detModel) {
  if (!detModel) return null;
  const lower = String(detModel).toLowerCase();
  if (lower.includes('yolo11') && lower.includes('int8')) return 'YOLO11m INT8';
  if (lower.includes('yolo11')) return 'YOLO11m-Seg';
  if (lower.includes('rfdetr')) return 'RF-DETR-2XL';
  if (lower.includes('comic-text-detector') || lower.includes('ctd')) return 'CTD';
  return detModel.length > 18 ? `${detModel.slice(0, 16)}…` : detModel;
}

/**
 * Compact 24px native desktop studio bottom status bar.
 * Displays live zoom %, cursor XY coordinates, active ONNX Runtime GPU/CPU provider,
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
    executionMode,
    detectorProvider,
    ocrProvider,
    inpainterProvider,
    detectorModel,
    lmStudio,
    activeProviderSummary,
    hardware,
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
      executionMode:
        s.pipelineConfig.executionMode || s.runtimeStatus.executionMode || 'hybrid',
      detectorProvider: s.runtimeStatus.detectorProvider,
      ocrProvider: s.runtimeStatus.ocrProvider,
      inpainterProvider: s.runtimeStatus.inpainterProvider,
      detectorModel:
        s.runtimeStatus.detectorModel || s.pipelineConfig.detectionModel,
      lmStudio: s.runtimeStatus.lmStudio,
      activeProviderSummary: s.runtimeStatus.activeProviderSummary,
      hardware: s.runtimeStatus.hardware,
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
  const resolvedHardwareProvider =
    hardware?.execution_provider || executionProvider;
  const providerMeta = formatExecutionProvider(resolvedHardwareProvider);
  const normExecMode =
    executionMode === 'cpu' || executionMode === 'cpu_only'
      ? 'cpu_only'
      : executionMode === 'gpu' || executionMode === 'force_gpu'
      ? 'force_gpu'
      : 'hybrid';
  const hybridStatusLabel =
    normExecMode === 'cpu_only'
      ? 'CPU'
      : providerMeta.short !== '-'
      ? providerMeta.short
      : normExecMode === 'force_gpu'
      ? 'GPU'
      : 'Hybrid';
  const hybridIsGpu = normExecMode !== 'cpu_only' && providerMeta.isGpu;

  const isDirectMlRfDetrWarning =
    normExecMode !== 'cpu_only' &&
    (resolvedHardwareProvider === 'DmlExecutionProvider' ||
      String(hardware?.backend || '').toLowerCase().includes('directml')) &&
    String(detectorModel || '').toLowerCase().includes('rfdetr');

  const activeProviderId =
    pipelineProvider || activeProviderSummary?.provider || 'none';
  const providerDisplayName =
    PROVIDER_LABELS[activeProviderId] || activeProviderId;

  const isNoLlm = activeProviderId === 'none';
  const isLocalLmStudio = activeProviderId === 'lmstudio';
  const isLlmConnected = isNoLlm
    ? true
    : isLocalLmStudio
    ? Boolean(lmStudio?.connected)
    : Boolean(activeProviderSummary?.model || pipelineModel);

  const fullModelName =
    (isLocalLmStudio
      ? lmStudio?.activeModel || lmStudio?.resolvedModelId
      : activeProviderSummary?.model) ||
    activePage?.model ||
    pipelineModel ||
    '';

  const shortModelLabel = isNoLlm
    ? 'Detect+Inpaint'
    : isLlmConnected
    ? formatShortModelName(fullModelName)
    : 'Offline';

  const llmEndpoint =
    (isLocalLmStudio ? lmStudio?.endpoint : activeProviderSummary?.base_url) ||
    pipelineBaseUrl ||
    (isLocalLmStudio ? 'http://localhost:1234/v1' : 'Not configured');

  const llmTooltip = isNoLlm
    ? 'Manual / No-LLM Mode: Detect + Inpaint only'
    : isLlmConnected
    ? `LLM Provider: ${providerDisplayName}\nEndpoint: ${llmEndpoint}\nActive Model: ${fullModelName || 'Default'}\nClick to refresh status`
    : `LLM Provider: ${providerDisplayName} (Offline)\nEndpoint: ${llmEndpoint}\n${lmStudio?.error || 'Click to retry connection'}`;

  return (
    <footer
      className="h-[var(--studio-status-h)] min-h-[var(--studio-status-h)] max-h-[var(--studio-status-h)] shrink-0 border-t border-studio-border bg-studio-panel px-[var(--studio-pad-x)] flex items-center justify-between gap-3 text-[length:var(--studio-font-size)] leading-none text-studio-text-secondary select-none z-30 transition-[height]"
      role="status"
      aria-label="Vellum Status Bar"
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
              {hasCursor ? cursorX : '-'}
            </span>
            <span className="w-[54px] text-left">
              <span className="text-studio-text-muted mr-0.5">Y:</span>
              {hasCursor ? cursorY : '-'}
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
                  Error - {activePage.error || 'Processing failed'}
                </span>
              </div>
            ) : activePage.status === 'idle' || activePage.status === 'queued' ? (
              <div
                data-testid="statusbar-idle-message"
                className="inline-flex items-center gap-1.5 text-[11px] font-medium text-studio-accent truncate"
                title="Configure Scope, Stages, Languages, and Pipeline Models, then press Run to start processing"
              >
                <span className="h-1.5 w-1.5 rounded-full bg-studio-accent shrink-0" />
                <span className="truncate">Ready - Press Run to process</span>
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

      {/* Right: ONNX GPU/CPU Provider + Active Detector + Active LLM Model */}
      <div className="flex items-center gap-2.5 shrink-0">
        {/* Hardware Execution Provider Badge (CUDA / DirectML / CPU) */}
        <div
          data-testid="statusbar-hardware-badge"
          className="inline-flex items-center gap-1.5 font-mono text-[11px]"
          title={`Active Engine: ${hybridStatusLabel}${hardware?.gpu_name ? ` (${hardware.gpu_name})` : ''}\nDetection: ${detectorProvider || resolvedHardwareProvider || 'CPU'}\nOCR: ${ocrProvider || 'CPUExecutionProvider'}\nInpainting: ${inpainterProvider || resolvedHardwareProvider || 'CPU'}\nActive Detector: ${detectorModel || 'Auto'}`}
        >
          <span
            className={`h-1.5 w-1.5 rounded-full shrink-0 ${
              hybridIsGpu
                ? 'bg-studio-accent'
                : resolvedHardwareProvider
                ? 'bg-studio-text-muted'
                : 'bg-studio-border-strong'
            }`}
          />
          <Cpu className="h-3 w-3 text-studio-text-muted shrink-0" />
          <span
            className={
              hybridIsGpu
                ? 'text-studio-text font-medium'
                : 'text-studio-text-secondary'
            }
          >
            {hybridStatusLabel}
          </span>
          {formatShortDetectorName(detectorModel) && (
            <>
              <span className="text-studio-border-strong hidden sm:inline">·</span>
              <span className="text-studio-text-secondary hidden sm:inline">
                {formatShortDetectorName(detectorModel)}
              </span>
            </>
          )}
          {isDirectMlRfDetrWarning && (
            <span
              data-testid="statusbar-directml-rfdetr-warning"
              className="inline-flex items-center gap-1 rounded-studio-sm border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-400"
              title="RF-DETR Seg 2XL on DirectML can be slow. Recommend YOLO11m-Seg for DirectML or the CUDA build for RF-DETR."
            >
              <AlertTriangle className="h-2.5 w-2.5 shrink-0" />
              <span className="hidden lg:inline">Use YOLO11m on DirectML</span>
            </span>
          )}
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
              isNoLlm
                ? 'bg-studio-text-muted'
                : isLlmConnected
                ? 'bg-studio-success'
                : 'bg-studio-danger'
            }`}
          />
          <Server className="h-3 w-3 text-studio-text-muted shrink-0 hidden sm:inline" />
          <span className="text-studio-text-muted hidden sm:inline shrink-0">
            {providerDisplayName}
          </span>
          <span className="text-studio-border-strong hidden sm:inline">·</span>
          <span
            className={`font-mono truncate ${
              isNoLlm
                ? 'text-studio-text-secondary'
                : isLlmConnected
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
