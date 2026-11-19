import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Cpu,
  Database,
  Download,
  FolderPlus,
  Languages,
  Layers,
  Loader2,
  RefreshCw,
  Server,
  Sparkles,
} from 'lucide-react';

const DEFAULT_API_BASE = 'http://localhost:8000';

const SOURCE_LANGUAGES = [
  { value: 'ja', label: 'Japanese (Manga)' },
  { value: 'ko', label: 'Korean (Manhwa)' },
  { value: 'zh', label: 'Chinese (Manhua)' },
];

const TARGET_LANGUAGES = [
  { value: 'id', label: 'Indonesian' },
  { value: 'en', label: 'English' },
];

function formatShortModelName(modelName) {
  if (!modelName) return 'No Model Loaded';
  const lower = modelName.toLowerCase();
  if (lower.includes('gemma 4 e2b') || lower.includes('gemma-4-e2b')) {
    return 'Gemma 4 E2B';
  }
  return modelName.length > 22 ? `${modelName.slice(0, 20)}…` : modelName;
}

/**
 * Top application toolbar for Manga/Manhwa/Manhua Comic Translator.
 * Includes multi-page batch image import, live LM Studio polling badge,
 * source & target language selectors, Re-translate action, and Export Image button.
 */
export default function Toolbar({
  apiBaseUrl = DEFAULT_API_BASE,
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
  const [lmStatus, setLmStatus] = useState(modelsStatus?.lm_studio || null);
  const [pollingLm, setPollingLm] = useState(false);
  const [exportFormat, setExportFormat] = useState('png');

  const fetchLmStudioStatus = useCallback(async () => {
    setPollingLm(true);
    try {
      const response = await fetch(`${apiBaseUrl}/api/lmstudio/status`, {
        method: 'GET',
        headers: { Accept: 'application/json' },
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      setLmStatus(data);
    } catch (err) {
      setLmStatus((prev) => ({
        connected: false,
        endpoint: prev?.endpoint || 'http://localhost:1235/v1',
        active_model: null,
        error: err?.message || 'LM Studio unreachable',
      }));
    } finally {
      setPollingLm(false);
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    if (modelsStatus?.lm_studio) {
      setLmStatus(modelsStatus.lm_studio);
    }
  }, [modelsStatus]);

  useEffect(() => {
    fetchLmStudioStatus();
    const intervalId = setInterval(fetchLmStudioStatus, 10000);
    return () => clearInterval(intervalId);
  }, [fetchLmStudioStatus]);

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0 && onImportFiles) {
      onImportFiles(files);
    }
    // Reset input so selecting the same file again still triggers onChange
    e.target.value = '';
  };

  const isLmConnected = Boolean(lmStatus?.connected);
  const fullModelName =
    lmStatus?.active_model ||
    lmStatus?.resolved_model_id ||
    'Gemma 4 E2B Uncensored HauhauCS Aggressive Q4 K P';
  const shortModelLabel = isLmConnected
    ? formatShortModelName(fullModelName)
    : 'Offline';

  const installedModelsCount = modelsStatus?.models
    ? Object.values(modelsStatus.models).filter((m) => m.installed).length
    : 4;
  const totalModelsCount = modelsStatus?.models
    ? Object.keys(modelsStatus.models).length
    : 4;
  const executionProvider =
    modelsStatus?.execution_provider === 'CUDAExecutionProvider' ? 'CUDA' : 'CPU';

  return (
    <header className="h-16 shrink-0 border-b border-slate-800/90 bg-slate-900/95 backdrop-blur-md px-4 flex items-center justify-between gap-3 z-30 select-none">
      {/* Left: Brand + Import Image Button (Multi-Page Batch Queue) */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2.5 pr-2 border-r border-slate-800">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 via-violet-500 to-purple-600 text-white shadow-lg shadow-indigo-500/25">
            <Sparkles className="h-5 w-5" />
          </div>
          <div className="hidden sm:block">
            <div className="flex items-center gap-1.5">
              <span className="text-sm font-bold tracking-tight text-slate-100">
                MangaFlow AI
              </span>
              <span className="rounded bg-indigo-500/15 border border-indigo-500/30 px-1.5 py-0.5 text-[10px] font-mono font-semibold text-indigo-300">
                {executionProvider}
              </span>
            </div>
            <p className="text-[11px] text-slate-400">Local Comic Translator</p>
          </div>
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
          className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 disabled:opacity-40 disabled:pointer-events-none px-3.5 py-2 text-xs font-semibold text-white shadow-lg shadow-indigo-600/25 transition-all cursor-pointer"
          title="Import one or multiple comic pages into the batch translation queue"
        >
          <FolderPlus className="h-4 w-4" />
          <span>Import Image</span>
          {batchStats.total > 0 && (
            <span className="ml-0.5 rounded-md bg-indigo-800/90 border border-indigo-400/30 px-1.5 py-0.5 font-mono text-[10px] text-indigo-100">
              {batchStats.completed}/{batchStats.total}
            </span>
          )}
        </button>

        {/* Batch Queue Active Indicator */}
        {(batchStats.processing > 0 || batchStats.queued > 0) && (
          <div className="hidden lg:inline-flex items-center gap-1.5 rounded-xl border border-indigo-500/30 bg-indigo-950/40 px-2.5 py-1.5 text-[11px] text-indigo-200">
            <Loader2 className="h-3.5 w-3.5 animate-spin text-indigo-400" />
            <span>
              Queue: {batchStats.completed}/{batchStats.total}
            </span>
          </div>
        )}
      </div>

      {/* Center: Source & Target Language Selectors + Re-translate Button */}
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1.5 rounded-xl border border-slate-800 bg-slate-950/90 px-2.5 py-1.5 shadow-inner">
          <Languages className="h-4 w-4 text-indigo-400 shrink-0" />

          {/* Source Language Dropdown */}
          <select
            aria-label="Source Language"
            value={sourceLang}
            onChange={(e) => onSourceLangChange && onSourceLangChange(e.target.value)}
            disabled={!isWorkspaceUnlocked || isProcessing}
            className="bg-transparent text-xs font-medium text-slate-200 focus:outline-none cursor-pointer disabled:opacity-50 pr-1"
          >
            {SOURCE_LANGUAGES.map((lang) => (
              <option key={lang.value} value={lang.value} className="bg-slate-900 text-slate-100">
                {lang.label}
              </option>
            ))}
          </select>

          <span className="text-slate-600 text-xs font-mono px-1">→</span>

          {/* Target Language Selector */}
          <select
            aria-label="Target Language"
            value={targetLang}
            onChange={(e) => onTargetLangChange && onTargetLangChange(e.target.value)}
            disabled={!isWorkspaceUnlocked || isProcessing}
            className="bg-transparent text-xs font-semibold text-indigo-300 focus:outline-none cursor-pointer disabled:opacity-50 pr-1"
          >
            {TARGET_LANGUAGES.map((lang) => (
              <option key={lang.value} value={lang.value} className="bg-slate-900 text-slate-100">
                {lang.label}
              </option>
            ))}
          </select>
        </div>

        {/* Re-translate Button */}
        <button
          type="button"
          onClick={onRetranslate}
          disabled={!isWorkspaceUnlocked || !canRetranslate || isProcessing || isRetranslating}
          className="inline-flex items-center gap-1.5 rounded-xl border border-slate-700/80 bg-slate-800/90 hover:bg-slate-700/90 hover:border-slate-600 disabled:opacity-40 disabled:pointer-events-none px-3.5 py-2 text-xs font-semibold text-slate-100 transition-all cursor-pointer"
          title="Re-translate dialogue bubbles on the current page via LM Studio (/api/retranslate)"
        >
          <RefreshCw
            className={`h-3.5 w-3.5 text-indigo-400 ${
              isRetranslating || isProcessing ? 'animate-spin' : ''
            }`}
          />
          <span>{isRetranslating ? 'Re-translating...' : 'Re-translate'}</span>
        </button>
      </div>

      {/* Right: Live LM Studio Polling Badge + Models Check + Export Image Button */}
      <div className="flex items-center gap-2.5">
        {/* Live LM Studio Status Badge polling /api/lmstudio/status */}
        <button
          type="button"
          onClick={fetchLmStudioStatus}
          className={`inline-flex items-center gap-2 rounded-xl border px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer ${
            isLmConnected
              ? 'border-emerald-500/30 bg-emerald-950/30 text-emerald-200 hover:bg-emerald-950/50'
              : 'border-rose-500/35 bg-rose-950/40 text-rose-200 hover:bg-rose-950/60'
          }`}
          title={
            isLmConnected
              ? `Connected to LM Studio (${lmStatus?.endpoint || 'http://localhost:1235/v1'})\nActive Model: ${fullModelName}\nClick to refresh status`
              : `LM Studio Offline (${lmStatus?.endpoint || 'http://localhost:1235/v1'})\n${lmStatus?.error || 'Ensure LM Studio server is running on port 1235'}`
          }
        >
          <span className="relative flex h-2 w-2">
            {isLmConnected && (
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            )}
            <span
              className={`relative inline-flex rounded-full h-2 w-2 ${
                isLmConnected ? 'bg-emerald-400' : 'bg-rose-500'
              }`}
            />
          </span>
          <Server className="h-3.5 w-3.5 text-slate-400 hidden xl:inline" />
          <span className="font-mono text-[11px] tracking-tight">
            localhost:1235 <span className="text-slate-500">•</span>{' '}
            <span className="font-sans font-semibold">{shortModelLabel}</span>
          </span>
          {pollingLm && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
        </button>

        {/* Local ONNX/OCR Models Status Button */}
        <button
          type="button"
          onClick={onOpenModelsModal}
          className="hidden md:inline-flex items-center gap-1.5 rounded-xl border border-slate-800 bg-slate-950/80 hover:border-slate-700 px-2.5 py-1.5 text-xs font-medium text-slate-300 transition-colors cursor-pointer"
          title="Inspect local ONNX & OCR model weights"
        >
          <Database className="h-3.5 w-3.5 text-indigo-400" />
          <span className="font-mono text-[11px]">
            {installedModelsCount}/{totalModelsCount}
          </span>
        </button>

        {/* Export Format Selector (.png / .jpg) + Export Image Button */}
        <div className="inline-flex items-center rounded-xl shadow-lg shadow-emerald-600/20 overflow-hidden border border-emerald-500/40">
          <select
            aria-label="Export Image Format"
            value={exportFormat}
            onChange={(e) => setExportFormat(e.target.value)}
            disabled={!isWorkspaceUnlocked || !canExport || isProcessing}
            className="bg-slate-900 text-emerald-300 font-mono text-[11px] font-semibold px-2 py-2 border-r border-emerald-500/30 focus:outline-none cursor-pointer disabled:opacity-40"
            title="Select full-resolution export format (.png or .jpg)"
          >
            <option value="png" className="bg-slate-900 text-slate-100">
              .PNG
            </option>
            <option value="jpg" className="bg-slate-900 text-slate-100">
              .JPG
            </option>
          </select>
          <button
            type="button"
            onClick={() => onExportImage && onExportImage(exportFormat)}
            disabled={!isWorkspaceUnlocked || !canExport || isProcessing}
            className="inline-flex items-center gap-1.5 bg-emerald-600 hover:bg-emerald-500 active:bg-emerald-700 disabled:opacity-40 disabled:pointer-events-none px-3.5 py-2 text-xs font-semibold text-white transition-all cursor-pointer"
            title="Export cleaned comic page with rendered translation text boxes at original resolution"
          >
            <Download className="h-4 w-4" />
            <span>Export Image</span>
          </button>
        </div>
      </div>
    </header>
  );
}
