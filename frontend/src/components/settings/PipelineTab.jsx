import {
  Check,
  Cpu,
  Eraser,
  FileText,
  HardDrive,
  Layers,
  Loader2,
  Search,
  Sparkles,
} from 'lucide-react';

import {
  LAMA_TILE_SIZES,
  READING_ORDER_OPTIONS,
  BACKEND_OPTIONS,
  ToggleSwitch,
  SectionCard,
  ModelStatusBadge,
  normalizeExecMode,
  StageHardwareBadge,
} from './SettingsPrimitives.jsx';

export default function PipelineTab({ ctx }) {
  const {
    viewLayers,
    runtimeStatus,
    toggleViewLayer,
    resetViewLayers,
    apiBaseUrl,
    draftPipeline,
    setDraftPipeline,
    pipelineModelsData,
    isLoadingPipelineModels,
    isSwitchingBackend,
    requiresBackendRelaunch,
    isRelaunchingApp,
    handleSelectPreferredBackend,
    handleRelaunchForBackend,
    handleOptionalThresholdChange,
    handleOptionalThresholdBlur,
    detectionModelOptions,
    ocrModelOptions,
    inpaintingModelOptions,
    selectedDetectionModelInfo,
    selectedOcrModelInfo,
    selectedInpaintingModelInfo,
  } = ctx;

  return (
              <div className="space-y-4" data-testid="settings-panel-pipeline">
                <SectionCard
                  title="Processing"
                  subtitle="Select local AI models for layout detection, text recognition, and artwork inpainting."
                >
                  <div className="space-y-2.5" data-testid="processing-card-rows">
                    {/* Row 1 - Detection */}
                    <div
                      className="rounded-studio border border-studio-border bg-studio-panel p-3 flex flex-col sm:flex-row sm:items-start gap-3"
                      data-testid="processing-row-detection"
                    >
                      <div className="sm:w-56 shrink-0 flex items-start gap-2.5">
                        <span className="mt-0.5 h-7 w-7 rounded-studio border border-studio-border bg-studio-surface flex items-center justify-center text-studio-accent shrink-0">
                          <Search className="h-3.5 w-3.5" />
                        </span>
                        <div className="min-w-0">
                          <label
                            htmlFor="pipeline-detection-model"
                            className="text-xs font-semibold text-studio-text block leading-tight"
                          >
                            Detection
                          </label>
                          <p className="text-[11px] text-studio-text-muted mt-0.5 leading-snug">
                            Find text, speech bubbles, and panels.
                          </p>
                          <StageHardwareBadge
                            stage="detect"
                            draftMode={draftPipeline.executionMode}
                            runtimeStatus={runtimeStatus}
                            testId="detection-hardware-badge"
                          />
                        </div>
                      </div>

                      <div className="flex-1 min-w-0 space-y-2.5">
                        <div className="flex items-center gap-2">
                          <select
                            id="pipeline-detection-model"
                            data-testid="pipeline-detection-model-select"
                            value={
                              draftPipeline.detectionModel ||
                              'vellum-det-yolo11m-seg-1024-fp16'
                            }
                            onChange={(e) =>
                              setDraftPipeline((prev) => ({
                                ...prev,
                                detectionModel: e.target.value,
                              }))
                            }
                            className="studio-input flex-1 min-w-0 h-8 px-2.5 text-xs cursor-pointer"
                          >
                            {detectionModelOptions.map((opt) => (
                              <option key={opt.id} value={opt.id}>
                                {opt.label}
                              </option>
                            ))}
                          </select>
                          <ModelStatusBadge
                            model={selectedDetectionModelInfo}
                            isLoading={isLoadingPipelineModels}
                            testId="detection-model-status-badge"
                          />
                        </div>

                        {String(draftPipeline.detectionModel || '').toLowerCase().includes('yolo') && (
                          <div className="space-y-1.5 pt-1" data-testid="pipeline-detector-precision-group">
                            <div
                              className="inline-flex rounded-studio-sm border border-studio-border bg-studio-surface p-0.5 gap-0.5"
                              role="radiogroup"
                              aria-label="Detector Precision"
                            >
                              {[
                                { id: 'fast', label: 'Fast' },
                                { id: 'balanced', label: 'Balanced' },
                                { id: 'high_precision', label: 'High Precision' },
                              ].map((opt) => {
                                const active =
                                  (draftPipeline.detectorPrecision || 'balanced') === opt.id;
                                return (
                                  <button
                                    key={opt.id}
                                    type="button"
                                    role="radio"
                                    aria-checked={active}
                                    data-testid={`detector-precision-${opt.id}`}
                                    onClick={() =>
                                      setDraftPipeline((prev) => ({
                                        ...prev,
                                        detectorPrecision: opt.id,
                                      }))
                                    }
                                    className={`px-2.5 py-1 text-[11px] font-medium rounded-[3px] transition-colors cursor-pointer ${
                                      active
                                        ? 'bg-studio-accent text-studio-accent-fg shadow-sm'
                                        : 'text-studio-muted hover:text-studio-text hover:bg-studio-elevated/60'
                                    }`}
                                  >
                                    {opt.label}
                                  </button>
                                );
                              })}
                            </div>
                            <p className="text-[11px] leading-snug text-studio-muted">
                              Fast: single-pass (~450 ms) · Balanced (Recommended): +dark-bubble invert &amp; whisper rescue · High Precision: +Flip TTA &amp; spread split (~0.8 s).
                            </p>
                          </div>
                        )}

                        {/* 3 side-by-side numeric threshold inputs */}
                        <div className="grid grid-cols-3 gap-2">
                          {[
                            {
                              key: 'textThreshold',
                              id: 'pipeline-text-threshold',
                              label: 'Text threshold',
                              defaultHint:
                                selectedDetectionModelInfo?.defaultThresholds?.text,
                            },
                            {
                              key: 'bubbleThreshold',
                              id: 'pipeline-bubble-threshold',
                              label: 'Bubble threshold',
                              defaultHint:
                                selectedDetectionModelInfo?.defaultThresholds?.bubble,
                            },
                            {
                              key: 'panelThreshold',
                              id: 'pipeline-panel-threshold',
                              label: 'Panel threshold',
                              defaultHint:
                                selectedDetectionModelInfo?.defaultThresholds?.panel,
                            },
                          ].map((field) => (
                            <div key={field.key} className="space-y-1 min-w-0">
                              <label
                                htmlFor={field.id}
                                className="text-[10px] font-medium text-studio-text-secondary block truncate"
                              >
                                {field.label}
                              </label>
                              <input
                                id={field.id}
                                data-testid={field.id}
                                type="number"
                                inputMode="decimal"
                                min={0.05}
                                max={0.95}
                                step={0.01}
                                placeholder="Default"
                                title={
                                  typeof field.defaultHint === 'number'
                                    ? `Leave empty for model default (${field.defaultHint.toFixed(2)})`
                                    : 'Leave empty to use model default'
                                }
                                value={draftPipeline[field.key] ?? ''}
                                onChange={(e) =>
                                  handleOptionalThresholdChange(
                                    field.key,
                                    e.target.value
                                  )
                                }
                                onBlur={() =>
                                  handleOptionalThresholdBlur(field.key)
                                }
                                className="studio-input w-full h-7 px-2 text-xs font-mono tabular-nums select-text"
                              />
                            </div>
                          ))}
                        </div>

                        {/* Detector Engine v2 controls: Whisper/Borderless thresholds + Adaptive Tiling & SFX Filter */}
                        <div className="grid grid-cols-2 gap-2 pt-0.5">
                          {[
                            {
                              key: 'whisperThreshold',
                              id: 'pipeline-whisper-threshold',
                              label: 'Whisper threshold',
                              defaultHint: 0.25,
                            },
                            {
                              key: 'borderlessThreshold',
                              id: 'pipeline-borderless-threshold',
                              label: 'Borderless threshold',
                              defaultHint: 0.28,
                            },
                          ].map((field) => (
                            <div key={field.key} className="space-y-1 min-w-0">
                              <label
                                htmlFor={field.id}
                                className="text-[10px] font-medium text-studio-text-secondary block truncate"
                              >
                                {field.label}
                              </label>
                              <input
                                id={field.id}
                                data-testid={field.id}
                                type="number"
                                inputMode="decimal"
                                min={0.05}
                                max={0.95}
                                step={0.01}
                                placeholder="Default"
                                title={`Leave empty for model default (${field.defaultHint.toFixed(2)})`}
                                value={draftPipeline[field.key] ?? ''}
                                onChange={(e) =>
                                  handleOptionalThresholdChange(
                                    field.key,
                                    e.target.value
                                  )
                                }
                                onBlur={() =>
                                  handleOptionalThresholdBlur(field.key)
                                }
                                className="studio-input w-full h-7 px-2 text-xs font-mono tabular-nums select-text"
                              />
                            </div>
                          ))}
                        </div>

                        <div className="flex flex-wrap items-center gap-4 pt-1">
                          <label className="inline-flex items-center gap-2 cursor-pointer select-none">
                            <input
                              type="checkbox"
                              data-testid="pipeline-adaptive-tiling-toggle"
                              checked={Boolean(
                                draftPipeline.enableAdaptiveTiling ?? true
                              )}
                              onChange={(e) =>
                                setDraftPipeline((prev) => ({
                                  ...prev,
                                  enableAdaptiveTiling: e.target.checked,
                                }))
                              }
                              className="h-3.5 w-3.5 rounded border-studio-border accent-[var(--studio-accent)] cursor-pointer"
                            />
                            <span className="text-[11px] text-studio-text-secondary">
                              Adaptive Tiling (Webtoon &amp; Ultra-HD)
                            </span>
                          </label>

                          <label className="inline-flex items-center gap-2 cursor-pointer select-none">
                            <input
                              type="checkbox"
                              data-testid="pipeline-sfx-suppress-toggle"
                              checked={
                                Number(draftPipeline.sfxSuppressIou ?? 0.6) < 0.99
                              }
                              onChange={(e) =>
                                setDraftPipeline((prev) => ({
                                  ...prev,
                                  sfxSuppressIou: e.target.checked ? 0.6 : 1.0,
                                }))
                              }
                              className="h-3.5 w-3.5 rounded border-studio-border accent-[var(--studio-accent)] cursor-pointer"
                            />
                            <span className="text-[11px] text-studio-text-secondary">
                              SFX Suppression Filter
                            </span>
                          </label>
                        </div>
                      </div>
                    </div>

                    {/* Row 2 - OCR */}
                    <div
                      className="rounded-studio border border-studio-border bg-studio-panel p-3 flex flex-col sm:flex-row sm:items-center gap-3"
                      data-testid="processing-row-ocr"
                    >
                      <div className="sm:w-56 shrink-0 flex items-start gap-2.5">
                        <span className="mt-0.5 h-7 w-7 rounded-studio border border-studio-border bg-studio-surface flex items-center justify-center text-studio-accent shrink-0">
                          <FileText className="h-3.5 w-3.5" />
                        </span>
                        <div className="min-w-0">
                          <label
                            htmlFor="pipeline-ocr-model"
                            className="text-xs font-semibold text-studio-text block leading-tight"
                          >
                            OCR
                          </label>
                          <p className="text-[11px] text-studio-text-muted mt-0.5 leading-snug">
                            Read the source text inside detected regions.
                          </p>
                          <StageHardwareBadge
                            stage="ocr"
                            draftMode={draftPipeline.executionMode}
                            selectedModel={draftPipeline.ocrModel}
                            modelInfo={selectedOcrModelInfo}
                            runtimeStatus={runtimeStatus}
                            testId="ocr-hardware-badge"
                          />
                        </div>
                      </div>

                      <div className="flex-1 min-w-0 space-y-1.5">
                        <div className="flex items-center gap-2">
                          <select
                            id="pipeline-ocr-model"
                            data-testid="pipeline-ocr-model-select"
                            value={draftPipeline.ocrModel || 'auto'}
                            onChange={(e) =>
                              setDraftPipeline((prev) => ({
                                ...prev,
                                ocrModel: e.target.value,
                              }))
                            }
                            className="studio-input flex-1 min-w-0 h-8 px-2.5 text-xs cursor-pointer"
                          >
                            {ocrModelOptions.map((opt) => (
                              <option key={opt.id} value={opt.id}>
                                {opt.label}
                              </option>
                            ))}
                          </select>
                          <ModelStatusBadge
                            model={selectedOcrModelInfo}
                            isLoading={isLoadingPipelineModels}
                            testId="ocr-model-status-badge"
                          />
                        </div>
                        {(selectedOcrModelInfo?.id === 'paddleocr-vl-1.6' ||
                          selectedOcrModelInfo?.id === 'paddleocr-vl-gguf') && (
                          <p className="text-[10px] text-studio-text-muted leading-snug">
                            {selectedOcrModelInfo.lmStudioHint ||
                              'Uses LM Studio Vision-OCR endpoint with PaddleOCR-VL-1.6-GGUF loaded.'}
                          </p>
                        )}
                      </div>
                    </div>

                    {/* Row 3 - Inpainting */}
                    <div
                      className="rounded-studio border border-studio-border bg-studio-panel p-3 flex flex-col sm:flex-row sm:items-center gap-3"
                      data-testid="processing-row-inpainting"
                    >
                      <div className="sm:w-56 shrink-0 flex items-start gap-2.5">
                        <span className="mt-0.5 h-7 w-7 rounded-studio border border-studio-border bg-studio-surface flex items-center justify-center text-studio-accent shrink-0">
                          <Eraser className="h-3.5 w-3.5" />
                        </span>
                        <div className="min-w-0">
                          <label
                            htmlFor="pipeline-inpainting-model"
                            className="text-xs font-semibold text-studio-text block leading-tight"
                          >
                            Inpainting
                          </label>
                          <p className="text-[11px] text-studio-text-muted mt-0.5 leading-snug">
                            Rebuild artwork beneath removed text.
                          </p>
                          <StageHardwareBadge
                            stage="inpaint"
                            draftMode={draftPipeline.executionMode}
                            selectedModel={draftPipeline.inpaintingModel}
                            modelInfo={selectedInpaintingModelInfo}
                            runtimeStatus={runtimeStatus}
                            testId="inpainting-hardware-badge"
                          />
                        </div>
                      </div>

                      <div className="flex-1 min-w-0 space-y-2">
                        <div className="flex items-center gap-2">
                          <select
                            id="pipeline-inpainting-model"
                            data-testid="pipeline-inpainting-model-select"
                            value={draftPipeline.inpaintingModel || 'lama-manga'}
                            onChange={(e) =>
                              setDraftPipeline((prev) => ({
                                ...prev,
                                inpaintingModel: e.target.value,
                              }))
                            }
                            className="studio-input flex-1 min-w-0 h-8 px-2.5 text-xs cursor-pointer"
                          >
                            {inpaintingModelOptions.map((opt) => (
                              <option key={opt.id} value={opt.id}>
                                {opt.label}
                              </option>
                            ))}
                          </select>
                          <ModelStatusBadge
                            model={selectedInpaintingModelInfo}
                            isLoading={isLoadingPipelineModels}
                            testId="inpainting-model-status-badge"
                          />
                        </div>

                        {(draftPipeline.inpaintingModel === 'flux-2-klein' ||
                          draftPipeline.inpaintingModel === 'rorem-mixed') && (
                          <div
                            className="rounded-studio border border-studio-border/80 bg-studio-surface px-2.5 py-2 space-y-1"
                            data-testid="inpainting-diffusion-url-box"
                          >
                            <label
                              htmlFor="pipeline-inpaint-api-url"
                              className="block text-[10px] font-semibold uppercase tracking-wider text-studio-text-secondary"
                            >
                              Local Diffusion API URL
                            </label>
                            <input
                              id="pipeline-inpaint-api-url"
                              data-testid="pipeline-inpaint-api-url-input"
                              type="text"
                              value={
                                draftPipeline.inpaintApiUrl ??
                                'http://127.0.0.1:8188'
                              }
                              onChange={(e) =>
                                setDraftPipeline((prev) => ({
                                  ...prev,
                                  inpaintApiUrl: e.target.value,
                                }))
                              }
                              placeholder="http://127.0.0.1:8188 or http://127.0.0.1:7860"
                              className="studio-input w-full h-7 px-2 text-xs font-mono"
                            />
                            <p className="text-[10px] text-studio-text-muted leading-snug">
                              Connects to local ComfyUI / A1111 / sd-server diffusion worker on tight text-stroke ROIs (auto-falls back to GPU AOT/LaMa when offline).
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </SectionCard>

                <SectionCard
                  title="Execution Provider"
                  subtitle={`Detected hardware: ${runtimeStatus.hardware?.gpu_name || runtimeStatus.executionProvider || 'Auto-detecting…'} · OCR: ${runtimeStatus.ocrProvider || 'CPUExecutionProvider'}`}
                >
                  {/* 4-stop Hardware Execution Backend Segmented Control + Live Status Pill */}
                  {(() => {
                    const hw = runtimeStatus.hardware || {};
                    const availBackends = hw.available_backends || {
                      cuda: false,
                      directml:
                        runtimeStatus.executionProvider === 'DmlExecutionProvider' ||
                        (runtimeStatus.availableProviders || []).includes('DmlExecutionProvider'),
                      cpu: true,
                    };
                    const currentPref = String(
                      draftPipeline.preferredBackend || hw.preferred_backend || 'auto'
                    ).toLowerCase();
                    const activeBackend = String(
                      hw.active_backend ||
                        hw.backend ||
                        (runtimeStatus.executionProvider === 'CUDAExecutionProvider'
                          ? 'cuda'
                          : runtimeStatus.executionProvider === 'DmlExecutionProvider'
                          ? 'directml'
                          : 'cpu')
                    ).toLowerCase();
                    const gpuName = hw.gpu_name || 'GPU';
                    const cudaReason =
                      hw.cuda_unavailable_reason ||
                      'CUDA Runtime Not Installed';

                    let statusText = `${gpuName} • ${
                      activeBackend === 'cuda'
                        ? 'NVIDIA CUDA Ready'
                        : activeBackend === 'directml'
                        ? 'Universal GPU Active'
                        : 'CPU Mode'
                    }`;
                    let statusTone =
                      activeBackend === 'cuda' || activeBackend === 'directml'
                        ? 'success'
                        : 'neutral';

                    if (
                      (currentPref === 'cuda' || !availBackends.cuda) &&
                      !availBackends.cuda &&
                      activeBackend !== 'cpu'
                    ) {
                      if (currentPref === 'cuda') {
                        statusText = `${cudaReason} — Fallback to ${
                          activeBackend === 'directml' ? 'Universal GPU' : 'CPU'
                        }`;
                        statusTone = 'warning';
                      } else {
                        statusText = `${gpuName} • ${
                          activeBackend === 'directml' ? 'Universal GPU Ready' : 'CPU Mode'
                        }`;
                      }
                    } else if (currentPref === 'cpu') {
                      statusText = 'CPU ExecutionProvider Active (0 MB VRAM)';
                      statusTone = 'neutral';
                    }

                    const pillClasses =
                      statusTone === 'warning'
                        ? 'border-studio-warning/40 bg-studio-warning/10 text-studio-warning'
                        : statusTone === 'success'
                        ? 'border-studio-success/40 bg-studio-success/10 text-studio-success'
                        : 'border-studio-border-strong bg-studio-canvas text-studio-text-secondary';

                    const dotClasses =
                      statusTone === 'warning'
                        ? 'bg-studio-warning'
                        : statusTone === 'success'
                        ? 'bg-studio-success'
                        : 'bg-studio-text-muted';

                    return (
                      <div className="space-y-2 pb-2.5 border-b border-studio-border">
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div className="text-[11px] font-medium text-studio-text">
                            Hardware Execution Backend
                          </div>
                          <span
                            data-testid="hardware-backend-status-pill"
                            title={
                              !availBackends.cuda
                                ? cudaReason
                                : `Active ONNX provider: ${hw.execution_provider || runtimeStatus.executionProvider || activeBackend}`
                            }
                            className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[10px] font-mono font-medium ${pillClasses}`}
                          >
                            {isSwitchingBackend ? (
                              <Loader2 className="h-2.5 w-2.5 animate-spin" />
                            ) : (
                              <span className={`h-1.5 w-1.5 rounded-full ${dotClasses}`} />
                            )}
                            <span>{statusText}</span>
                          </span>
                        </div>

                        <div
                          role="radiogroup"
                          aria-label="Hardware Execution Backend"
                          className="grid grid-cols-4 gap-1 rounded-studio border border-studio-border bg-studio-canvas p-1"
                        >
                          {BACKEND_OPTIONS.map((opt) => {
                            const availabilityMap = {
                              auto: true,
                              directml: Boolean(availBackends.directml),
                              cuda: Boolean(availBackends.cuda),
                              cpu: true,
                            };
                            const isUnavailable = !availabilityMap[opt.id];
                            // If saved preferred_backend points to an unavailable backend, visually highlight the active fallback
                            const effectiveSelectedId = availabilityMap[currentPref]
                              ? currentPref
                              : availabilityMap[activeBackend]
                              ? activeBackend
                              : 'auto';
                            const isSelected = effectiveSelectedId === opt.id && !isUnavailable;
                            const tip =
                              opt.id === 'cuda' && isUnavailable
                                ? `NVIDIA CUDA disabled: ${cudaReason}`
                                : opt.id === 'cuda'
                                ? 'NVIDIA CUDA: maximum speed for NVIDIA RTX GPUs'
                                : opt.id === 'directml' && isUnavailable
                                ? 'Universal GPU disabled: Universal GPU provider is not bundled in this build'
                                : opt.id === 'directml'
                                ? 'Universal GPU: works across all modern Windows GPUs (AMD, Intel, and NVIDIA)'
                                : `Use ${opt.label} execution backend`;

                            return (
                              <button
                                key={opt.id}
                                type="button"
                                role="radio"
                                aria-checked={isSelected}
                                disabled={isSwitchingBackend || isUnavailable}
                                data-testid={`backend-seg-${opt.id}`}
                                title={tip}
                                onClick={() => {
                                  if (!isUnavailable) {
                                    handleSelectPreferredBackend(opt.id);
                                  }
                                }}
                                className={`relative flex items-center justify-center h-8 rounded-studio-sm text-xs font-medium transition-all ${
                                  isSelected
                                    ? 'bg-studio-accent text-studio-accent-fg font-semibold shadow-sm cursor-pointer'
                                    : isUnavailable
                                    ? 'bg-studio-panel/40 border border-studio-border/40 text-studio-text-muted opacity-45 cursor-not-allowed pointer-events-auto'
                                    : 'text-studio-text-secondary hover:bg-studio-hover hover:text-studio-text cursor-pointer'
                                }`}
                              >
                                <span>{opt.label}</span>
                              </button>
                            );
                          })}
                        </div>
                        {requiresBackendRelaunch && (
                          <div className="mt-2 flex items-center justify-between gap-2 rounded-studio border border-studio-accent/50 bg-studio-accent/10 px-2.5 py-2">
                            <p className="text-[11px] text-studio-text leading-snug">
                              Switching between NVIDIA CUDA and Universal GPU requires relaunching the matching runtime core.
                            </p>
                            <button
                              type="button"
                              disabled={isRelaunchingApp}
                              onClick={handleRelaunchForBackend}
                              data-testid="relaunch-backend-btn"
                              className="studio-btn-accent shrink-0 px-2.5 h-7 text-[11px]"
                            >
                              {isRelaunchingApp ? 'Restarting…' : 'Restart to Switch Backend'}
                            </button>
                          </div>
                        )}
                        {!availBackends.cuda && (
                          <p className="text-[10px] text-studio-text-muted leading-snug">
                            NVIDIA CUDA unavailable: {cudaReason}. ONNX sessions will run on{' '}
                            <span className="font-mono text-studio-text-secondary">
                              {availBackends.directml ? 'Universal GPU' : 'CPU'}
                            </span>
                            .
                          </p>
                        )}
                        <p className="text-[10px] text-studio-text-muted leading-snug">
                          <span className="font-medium text-studio-text-secondary">Universal GPU:</span> works across all modern Windows GPUs (AMD, Intel, and NVIDIA).{' '}
                          <span className="font-medium text-studio-text-secondary">NVIDIA CUDA:</span> maximum speed for NVIDIA RTX GPUs.
                        </p>
                      </div>
                    );
                  })()}

                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                    {[
                      {
                        id: 'hybrid',
                        label: 'Smart Hybrid',
                        badge: 'Recommended',
                        desc: 'Runs detection and cleanup on GPU, and text reading on CPU. Saves GPU memory for local AI models.',
                        icon: Cpu,
                      },
                      {
                        id: 'force_gpu',
                        label: 'Full GPU',
                        desc: 'Runs all stages on your GPU for maximum speed. Best if you have plenty of GPU memory.',
                        icon: Sparkles,
                      },
                      {
                        id: 'cpu_only',
                        label: 'CPU Only',
                        desc: 'Runs everything on your processor without using GPU memory. Slower, but works on any PC.',
                        icon: HardDrive,
                      },
                    ].map((opt) => {
                      const Icon = opt.icon;
                      const isActive =
                        normalizeExecMode(draftPipeline.executionMode) === opt.id;
                      return (
                        <button
                          key={opt.id}
                          type="button"
                          data-testid={`exec-mode-${opt.id}`}
                          aria-pressed={isActive}
                          onClick={() =>
                            setDraftPipeline((prev) => ({
                              ...prev,
                              executionMode: opt.id,
                            }))
                          }
                          className={`rounded-studio border p-3 text-left transition-colors cursor-pointer space-y-1 ${
                            isActive
                              ? 'border-studio-accent bg-studio-accent/10'
                              : 'border-studio-border bg-studio-panel hover:border-studio-border-strong'
                          }`}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-studio-text whitespace-nowrap">
                              <Icon
                                className={`h-3.5 w-3.5 shrink-0 ${
                                  isActive
                                    ? 'text-studio-accent'
                                    : 'text-studio-text-muted'
                                }`}
                              />
                              <span>{opt.label}</span>
                              {opt.badge && (
                                <span className="rounded-full border border-studio-accent/40 bg-studio-accent/15 px-1.5 py-0.5 text-[10px] font-medium leading-none text-studio-accent">
                                  {opt.badge}
                                </span>
                              )}
                            </span>
                            {isActive && (
                              <Check className="h-3.5 w-3.5 shrink-0 text-studio-accent" />
                            )}
                          </div>
                          <p className="text-[11px] text-studio-text-muted leading-snug">
                            {opt.desc}
                          </p>
                        </button>
                      );
                    })}
                  </div>
                </SectionCard>

                <SectionCard
                  title="Detection & Bounding Box Geometry"
                  subtitle="Controls Comic Text Detector sensitivity and mask dilation around detected speech bubbles."
                >
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {/* Detection Threshold */}
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between text-[11px]">
                        <label
                          htmlFor="pipeline-conf-threshold"
                          className="font-medium text-studio-text"
                        >
                          Detection Threshold
                        </label>
                        <span className="font-mono font-semibold text-studio-accent tabular-nums">
                          {Number(draftPipeline.confThreshold ?? 0.35).toFixed(2)}
                        </span>
                      </div>
                      <input
                        id="pipeline-conf-threshold"
                        type="range"
                        min={0.05}
                        max={0.95}
                        step={0.01}
                        value={draftPipeline.confThreshold ?? 0.35}
                        onChange={(e) =>
                          setDraftPipeline((prev) => ({
                            ...prev,
                            confThreshold: Number(e.target.value),
                          }))
                        }
                        className="w-full accent-[var(--color-accent)] cursor-pointer"
                      />
                      <p className="text-[10px] text-studio-text-muted">
                        Lower catches faint SFX/text; higher reduces false positives.
                      </p>
                    </div>

                    {/* Bbox Dilation Padding */}
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between text-[11px]">
                        <label
                          htmlFor="pipeline-bbox-padding"
                          className="font-medium text-studio-text"
                        >
                          BBox Dilation Padding
                        </label>
                        <span className="font-mono font-semibold text-studio-accent tabular-nums">
                          {Math.round(Number(draftPipeline.bboxPadding ?? 6))} px
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <input
                          id="pipeline-bbox-padding"
                          type="range"
                          min={0}
                          max={32}
                          step={1}
                          value={draftPipeline.bboxPadding ?? 6}
                          onChange={(e) =>
                            setDraftPipeline((prev) => ({
                              ...prev,
                              bboxPadding: Number(e.target.value),
                            }))
                          }
                          className="flex-1 accent-[var(--color-accent)] cursor-pointer"
                        />
                        <input
                          type="number"
                          min={0}
                          max={64}
                          value={draftPipeline.bboxPadding ?? 6}
                          onChange={(e) =>
                            setDraftPipeline((prev) => ({
                              ...prev,
                              bboxPadding: Math.max(
                                0,
                                Math.min(64, Number(e.target.value) || 0)
                              ),
                            }))
                          }
                          className="studio-input w-14 h-7 px-1.5 text-center font-mono text-xs tabular-nums"
                        />
                      </div>
                      <p className="text-[10px] text-studio-text-muted">
                        Expands masks/boxes outward to cleanly cover outlines & furigana.
                      </p>
                    </div>
                  </div>
                </SectionCard>

                <SectionCard
                  title="LaMa Inpainting & Reading Order"
                  subtitle="Configure tile patch size for LaMa ONNX reconstruction and speech bubble sorting order."
                >
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {/* LaMa Tile Size */}
                    <div className="space-y-1.5">
                      <span className="text-[11px] font-medium text-studio-text block">
                        LaMa Tile Size (px)
                      </span>
                      <div className="flex flex-wrap gap-1.5">
                        {LAMA_TILE_SIZES.map((sz) => {
                          const isActive = Number(draftPipeline.lamaTileSize ?? 512) === sz;
                          return (
                            <button
                              key={sz}
                              type="button"
                              onClick={() =>
                                setDraftPipeline((prev) => ({
                                  ...prev,
                                  lamaTileSize: sz,
                                }))
                              }
                              className={`h-7 px-2.5 rounded-studio border font-mono text-[11px] transition-colors cursor-pointer ${
                                isActive
                                  ? 'border-studio-accent bg-studio-accent text-studio-accent-fg font-semibold'
                                  : 'border-studio-border bg-studio-panel text-studio-text-secondary hover:bg-studio-hover hover:text-studio-text'
                              }`}
                            >
                              {sz}×{sz}
                            </button>
                          );
                        })}
                      </div>
                      <p className="text-[10px] text-studio-text-muted">
                        512×512 matches native LaMa ONNX training resolution.
                      </p>
                    </div>

                    {/* Reading Order */}
                    <div className="space-y-1.5">
                      <span className="text-[11px] font-medium text-studio-text block">
                        Reading Order
                      </span>
                      <div className="grid grid-cols-3 gap-1.5">
                        {READING_ORDER_OPTIONS.map((ro) => {
                          const isActive = (draftPipeline.readingOrder || 'auto') === ro.id;
                          return (
                            <button
                              key={ro.id}
                              type="button"
                              data-testid={`reading-order-${ro.id}`}
                              onClick={() =>
                                setDraftPipeline((prev) => ({
                                  ...prev,
                                  readingOrder: ro.id,
                                }))
                              }
                              className={`h-7 px-2 rounded-studio border text-xs font-medium transition-colors cursor-pointer ${
                                isActive
                                  ? 'border-studio-accent bg-studio-accent text-studio-accent-fg font-semibold'
                                  : 'border-studio-border bg-studio-panel text-studio-text-secondary hover:bg-studio-hover hover:text-studio-text'
                              }`}
                            >
                              {ro.label}
                            </button>
                          );
                        })}
                      </div>
                      <p className="text-[10px] text-studio-text-muted">
                        {
                          READING_ORDER_OPTIONS.find(
                            (r) => r.id === (draftPipeline.readingOrder || 'auto')
                          )?.desc
                        }
                      </p>
                    </div>
                  </div>
                </SectionCard>

                <SectionCard
                  title="Memory & VRAM Lifecycle"
                  subtitle="Native C++17 ONNX Runtime lazy-loading and session-unloading lifecycle."
                >
                  <div className="space-y-3">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <label
                          htmlFor="pipeline-auto-unload"
                          className="text-xs font-medium text-studio-text block"
                        >
                          Auto-Unload VRAM After Each Stage
                        </label>
                        <p className="text-[11px] text-studio-text-muted">
                          Immediately releases and resets ONNX Runtime sessions after Detect, OCR, and LaMa so idle VRAM and memory stay near zero.
                        </p>
                      </div>
                      <ToggleSwitch
                        id="pipeline-auto-unload"
                        label="Auto-Unload VRAM"
                        checked={Boolean(draftPipeline.autoUnloadVram)}
                        onChange={(val) =>
                          setDraftPipeline((prev) => ({
                            ...prev,
                            autoUnloadVram: val,
                          }))
                        }
                      />
                    </div>

                    <div className="h-px bg-studio-border" />

                    <div className="space-y-1.5">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <label
                          htmlFor="pipeline-batch-schedule"
                          className="text-xs font-medium text-studio-text"
                        >
                          Batch Execution Mode
                        </label>
                        <div
                          id="pipeline-batch-schedule"
                          role="group"
                          aria-label="Batch Execution Mode"
                          className="inline-flex rounded-lg border border-studio-border bg-studio-bg p-0.5"
                        >
                          {[
                            { id: 'auto', label: 'Auto' },
                            { id: 'pipelined', label: 'Pipelined' },
                            { id: 'stage-first', label: 'Stage-First' },
                          ].map((opt) => {
                            const active = (draftPipeline.batchScheduleMode || 'auto') === opt.id;
                            return (
                              <button
                                key={opt.id}
                                type="button"
                                onClick={() =>
                                  setDraftPipeline((prev) => ({
                                    ...prev,
                                    batchScheduleMode: opt.id,
                                  }))
                                }
                                className={`px-2.5 py-1 text-[11px] font-medium rounded-md transition-colors ${
                                  active
                                    ? 'bg-studio-accent text-studio-accent-fg shadow-sm'
                                    : 'text-studio-text-muted hover:text-studio-text'
                                }`}
                              >
                                {opt.label}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                      <p className="text-[11px] text-studio-text-muted">
                        Auto picks Pipelined (AHEAD=3) when VRAM &gt;= 1 GB or using Cloud API, and Stage-First (50-page chunks) on tight VRAM (&lt; 1 GB).
                      </p>
                    </div>

                    <div className="h-px bg-studio-border" />

                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <label
                          htmlFor="pipeline-refine-mask"
                          className="text-xs font-medium text-studio-text block"
                        >
                          Refine Binary Text Mask Before Inpainting
                        </label>
                        <p className="text-[11px] text-studio-text-muted">
                          Smooths jagged mask edges for cleaner screentone reconstruction.
                        </p>
                      </div>
                      <ToggleSwitch
                        id="pipeline-refine-mask"
                        label="Refine Binary Text Mask"
                        checked={Boolean(draftPipeline.refineMask)}
                        onChange={(val) =>
                          setDraftPipeline((prev) => ({
                            ...prev,
                            refineMask: val,
                          }))
                        }
                      />
                    </div>

                    <div className="h-px bg-studio-border" />

                    {/* v3.0.0 Studio Core Feature Flags */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      <div className="flex items-center justify-between gap-2">
                        <div>
                          <label
                            htmlFor="pipeline-3tier-inpaint"
                            className="text-xs font-medium text-studio-text block"
                          >
                            3-Tier CPU Inpainting
                          </label>
                          <p className="text-[11px] text-studio-text-muted">
                            Fast CPU flat/Telea fill before LaMa/GPU tiles.
                          </p>
                        </div>
                        <ToggleSwitch
                          id="pipeline-3tier-inpaint"
                          label="3-Tier CPU Inpainting"
                          checked={draftPipeline.enable3TierInpaint !== false}
                          onChange={(val) =>
                            setDraftPipeline((prev) => ({
                              ...prev,
                              enable3TierInpaint: val,
                            }))
                          }
                        />
                      </div>

                      <div className="flex items-center justify-between gap-2">
                        <div>
                          <label
                            htmlFor="pipeline-tone-match"
                            className="text-xs font-medium text-studio-text block"
                          >
                            Screentone Tone Match
                          </label>
                          <p className="text-[11px] text-studio-text-muted">
                            Matches boundary luminance/histogram to remove halos.
                          </p>
                        </div>
                        <ToggleSwitch
                          id="pipeline-tone-match"
                          label="Screentone Tone Match"
                          checked={draftPipeline.enableToneMatch !== false}
                          onChange={(val) =>
                            setDraftPipeline((prev) => ({
                              ...prev,
                              enableToneMatch: val,
                            }))
                          }
                        />
                      </div>

                      <div className="flex items-center justify-between gap-2">
                        <div>
                          <label
                            htmlFor="pipeline-cv-style"
                            className="text-xs font-medium text-studio-text block"
                          >
                            0-VRAM Auto Style/Outline
                          </label>
                          <p className="text-[11px] text-studio-text-muted">
                            Detects ink color, outline stroke, and tilt angle on CPU.
                          </p>
                        </div>
                        <ToggleSwitch
                          id="pipeline-cv-style"
                          label="0-VRAM Auto Style/Outline"
                          checked={draftPipeline.enableCvStyleDetect !== false}
                          onChange={(val) =>
                            setDraftPipeline((prev) => ({
                              ...prev,
                              enableCvStyleDetect: val,
                            }))
                          }
                        />
                      </div>

                      <div className="flex items-center justify-between gap-2">
                        <div>
                          <label
                            htmlFor="pipeline-waist-cut"
                            className="text-xs font-medium text-studio-text block"
                          >
                            Conjoined Balloon Waist-Cut
                          </label>
                          <p className="text-[11px] text-studio-text-muted">
                            Splits merged double-bubbles at narrow constriction waists.
                          </p>
                        </div>
                        <ToggleSwitch
                          id="pipeline-waist-cut"
                          label="Conjoined Balloon Waist-Cut"
                          checked={draftPipeline.enableWaistCut !== false}
                          onChange={(val) =>
                            setDraftPipeline((prev) => ({
                              ...prev,
                              enableWaistCut: val,
                            }))
                          }
                        />
                      </div>

                      <div className="flex items-center justify-between gap-2">
                        <div>
                          <label
                            htmlFor="pipeline-auto-deskew"
                            className="text-xs font-medium text-studio-text block"
                          >
                            Auto-Deskew Physical Scans
                          </label>
                          <p className="text-[11px] text-studio-text-muted">
                            Straightens tilted physical scan margins before detection.
                          </p>
                        </div>
                        <ToggleSwitch
                          id="pipeline-auto-deskew"
                          label="Auto-Deskew Physical Scans"
                          checked={Boolean(draftPipeline.autoDeskewScans)}
                          onChange={(val) =>
                            setDraftPipeline((prev) => ({
                              ...prev,
                              autoDeskewScans: val,
                            }))
                          }
                        />
                      </div>

                      <div className="flex items-center justify-between gap-2">
                        <div>
                          <label
                            htmlFor="pipeline-mask-furigana"
                            className="text-xs font-medium text-studio-text block"
                          >
                            Auto-Mask Furigana (OCR)
                          </label>
                          <p className="text-[11px] text-studio-text-muted">
                            Suppresses small ruby/furigana strokes to keep main kanji OCR clean.
                          </p>
                        </div>
                        <ToggleSwitch
                          id="pipeline-mask-furigana"
                          label="Auto-Mask Furigana (OCR)"
                          checked={draftPipeline.autoMaskFuriganaOcr !== false}
                          onChange={(val) =>
                            setDraftPipeline((prev) => ({
                              ...prev,
                              autoMaskFuriganaOcr: val,
                            }))
                          }
                        />
                      </div>

                      <div className="flex items-center justify-between gap-2">
                        <div>
                          <label
                            htmlFor="pipeline-skip-blank"
                            className="text-xs font-medium text-studio-text block"
                          >
                            Skip Blank & Duplicate Pages (pHash)
                          </label>
                          <p className="text-[11px] text-studio-text-muted">
                            Skips empty dividers and caches duplicate pages via 64-bit DCT pHash.
                          </p>
                        </div>
                        <ToggleSwitch
                          id="pipeline-skip-blank"
                          label="Skip Blank & Duplicate Pages (pHash)"
                          checked={draftPipeline.skipBlankPages !== false}
                          onChange={(val) =>
                            setDraftPipeline((prev) => ({
                              ...prev,
                              skipBlankPages: val,
                            }))
                          }
                        />
                      </div>

                      <div className="flex items-center justify-between gap-2">
                        <div>
                          <label
                            htmlFor="pipeline-smart-ssim"
                            className="text-xs font-medium text-studio-text block"
                          >
                            Smart Grayscale & SSIM Export
                          </label>
                          <p className="text-[11px] text-studio-text-muted">
                            Auto-detects B&W manga pages and optimizes JPEG quality via SSIM.
                          </p>
                        </div>
                        <ToggleSwitch
                          id="pipeline-smart-ssim"
                          label="Smart Grayscale & SSIM Export"
                          checked={draftPipeline.smartSsimExport !== false}
                          onChange={(val) =>
                            setDraftPipeline((prev) => ({
                              ...prev,
                              smartSsimExport: val,
                            }))
                          }
                        />
                      </div>
                    </div>

                    <div className="h-px bg-studio-border" />

                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <span className="text-xs font-medium text-studio-text block">
                          Purge Temp Image Cache
                        </span>
                        <p className="text-[11px] text-studio-text-muted">
                          Wipes unpinned temporary images in <code className="font-mono">./cache/images/</code> to reclaim SSD space between 3,000-page volumes (pinned project caches remain intact).
                        </p>
                      </div>
                      <button
                        type="button"
                        data-testid="pipeline-purge-temp-cache-btn"
                        onClick={async (e) => {
                          const btn = e.currentTarget;
                          const origText = btn.textContent;
                          btn.disabled = true;
                          btn.textContent = 'Purging…';
                          try {
                            const resp = await fetch(`${apiBaseUrl}/api/cache/clear`, {
                              method: 'POST',
                            });
                            const data = await resp.json().catch(() => ({}));
                            const count = data?.deleted_files ?? 0;
                            const mb = ((data?.freed_bytes ?? 0) / (1024 * 1024)).toFixed(1);
                            btn.textContent = `Cleared ${count} (${mb} MB)`;
                          } catch {
                            btn.textContent = 'Purge Failed';
                          } finally {
                            setTimeout(() => {
                              btn.disabled = false;
                              btn.textContent = origText || 'Purge Temp Image Cache';
                            }, 2500);
                          }
                        }}
                        className="studio-btn-secondary h-7 px-2.5 text-[11px] font-medium shrink-0 cursor-pointer"
                      >
                        Purge Temp Image Cache
                      </button>
                    </div>
                  </div>
                </SectionCard>

                <SectionCard
                  title="Default Canvas Layers"
                  subtitle="Choose which visual layers are enabled by default in the studio canvas."
                >
                  <div className="flex flex-wrap items-center justify-between gap-2.5">
                    <div
                      className="inline-flex items-center gap-0.5 rounded-studio border border-studio-border bg-studio-surface p-1"
                      role="group"
                      aria-label="Default Canvas Layer Toggles"
                    >
                      {[
                        { key: 'original', label: 'Raw', title: 'Toggle Original Image Layer' },
                        { key: 'inpainted', label: 'Clean', title: 'Toggle Inpainted Image Layer' },
                        { key: 'text', label: 'Text', title: 'Toggle Translated Text Layer' },
                        { key: 'boxes', label: 'Boxes', title: 'Toggle Bounding Boxes & Reading Order (#1/#2)' },
                        { key: 'mask', label: 'Mask', title: 'Toggle Inpaint Mask Overlay' },
                      ].map((layer) => {
                        const isOn = Boolean(viewLayers?.[layer.key]);
                        return (
                          <button
                            key={layer.key}
                            type="button"
                            onClick={() => toggleViewLayer(layer.key)}
                            aria-pressed={isOn}
                            title={layer.title}
                            data-testid={`settings-layer-toggle-${layer.key}`}
                            className={`h-6 px-2.5 rounded-studio-sm text-[11px] font-medium transition-colors cursor-pointer ${
                              isOn
                                ? 'bg-studio-accent/15 text-studio-accent border border-studio-accent/40'
                                : 'text-studio-text-muted hover:text-studio-text hover:bg-studio-hover border border-transparent'
                            }`}
                          >
                            {layer.label}
                          </button>
                        );
                      })}
                    </div>

                    <button
                      type="button"
                      onClick={() => resetViewLayers()}
                      data-testid="settings-layer-reset-btn"
                      className="studio-btn-secondary h-7 px-2.5 text-[11px] font-medium shrink-0 cursor-pointer"
                      title="Restore Raw/Clean/Text/Boxes = ON, Mask = OFF"
                    >
                      Reset Default
                    </button>
                  </div>
                </SectionCard>
              </div>
  );
}
