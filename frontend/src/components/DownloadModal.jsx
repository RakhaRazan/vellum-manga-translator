import React, { useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Cpu,
  Database,
  Download,
  HardDrive,
  Loader2,
  RefreshCw,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const MODEL_METADATA = {
  comic_text_detector: {
    label: 'Comic Text Detector (ONNX)',
    description: 'Speech bubble bounding boxes, reading order & segmentation mask',
    defaultSizeMb: 90,
  },
  manga_ocr: {
    label: 'Manga OCR Base (Japanese)',
    description: 'Vision-Encoder-Decoder transformer for vertical & horizontal Japanese manga text',
    defaultSizeMb: 424,
  },
  rapid_ocr_ko_zh: {
    label: 'RapidOCR PP-OCRv4 (Korean & Chinese)',
    description: 'Multilingual detection, direction classification & recognition ONNX weights',
    defaultSizeMb: 41,
  },
  lama_inpaint: {
    label: 'LaMa Tile Inpainter (ONNX)',
    description: 'Resolution-robust large mask inpainting for lossless background cleaning',
    defaultSizeMb: 198,
  },
};

function formatMegabytes(bytes) {
  if (!bytes || bytes <= 0) return '0.0 MB';
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Dark-mode blocking modal overlay that connects to `/api/models/download` via SSE
 * when local CV/OCR models are incomplete, streaming real-time download progress
 * before unlocking the main editor workspace.
 */
export default function DownloadModal({
  isOpen,
  initialStatus,
  apiBaseUrl = API_BASE_URL,
  onModelsReady,
  onClose,
}) {
  const [modelsState, setModelsState] = useState(initialStatus?.models || {});
  const [executionProvider, setExecutionProvider] = useState(
    initialStatus?.execution_provider || 'Detecting...'
  );
  const [streamStatus, setStreamStatus] = useState('idle'); // idle | connecting | downloading | completed | error
  const [activeModelKey, setActiveModelKey] = useState(null);
  const [activeFile, setActiveFile] = useState(null);
  const [fileIndex, setFileIndex] = useState(null);
  const [fileCount, setFileCount] = useState(null);
  const [filePercent, setFilePercent] = useState(0);
  const [overallPercent, setOverallPercent] = useState(0);
  const [downloadedBytes, setDownloadedBytes] = useState(0);
  const [totalBytes, setTotalBytes] = useState(0);
  const [statusMessage, setStatusMessage] = useState(
    'Preparing local ONNX & OCR model verification...'
  );
  const [errorMessage, setErrorMessage] = useState(null);
  const [recentLogs, setRecentLogs] = useState([]);

  const eventSourceRef = useRef(null);

  useEffect(() => {
    if (initialStatus) {
      setModelsState(initialStatus.models || {});
      if (initialStatus.execution_provider) {
        setExecutionProvider(initialStatus.execution_provider);
      }
      if (initialStatus.all_ready) {
        setOverallPercent(100);
        setFilePercent(100);
        setStreamStatus('completed');
        setStatusMessage('All required local AI models are installed and verified offline.');
      }
    }
  }, [initialStatus]);

  const appendLog = (msg) => {
    if (!msg) return;
    setRecentLogs((prev) => {
      if (prev[prev.length - 1] === msg) return prev;
      return [...prev.slice(-5), msg];
    });
  };

  const refreshStatusAndFinish = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/models/status`);
      if (!response.ok) {
        throw new Error(`Status check failed with HTTP ${response.status}`);
      }
      const latestStatus = await response.json();
      setModelsState(latestStatus.models || {});
      if (latestStatus.execution_provider) {
        setExecutionProvider(latestStatus.execution_provider);
      }
      if (latestStatus.all_ready) {
        setStreamStatus('completed');
        setOverallPercent(100);
        setFilePercent(100);
        if (onModelsReady) {
          onModelsReady(latestStatus);
        }
      } else {
        setStreamStatus('error');
        setErrorMessage(
          'Some model files are still missing after the download stream ended. Please click Retry.'
        );
      }
    } catch (err) {
      setStreamStatus('error');
      setErrorMessage(err?.message || 'Failed to verify model status after download.');
    }
  };

  const startDownloadStream = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    setErrorMessage(null);
    setStreamStatus('connecting');
    setStatusMessage('Connecting to /api/models/download SSE stream...');
    appendLog('Connecting to http://localhost:8000/api/models/download ...');

    const sseUrl = `${apiBaseUrl}/api/models/download`;
    const es = new EventSource(sseUrl);
    eventSourceRef.current = es;

    es.onopen = () => {
      setStreamStatus('downloading');
    };

    es.onmessage = (event) => {
      if (!event.data) return;
      try {
        const payload = JSON.parse(event.data);

        if (payload.status === 'error') {
          setStreamStatus('error');
          setErrorMessage(payload.message || 'An error occurred while downloading model weights.');
          appendLog(payload.message || 'Download error');
          es.close();
          eventSourceRef.current = null;
          return;
        }

        if (typeof payload.overall_percent === 'number') {
          setOverallPercent(payload.overall_percent);
        }
        if (typeof payload.file_percent === 'number') {
          setFilePercent(payload.file_percent);
        }
        if (typeof payload.downloaded_bytes === 'number') {
          setDownloadedBytes(payload.downloaded_bytes);
        }
        if (typeof payload.total_bytes === 'number') {
          setTotalBytes(payload.total_bytes);
        }
        if (payload.model_key && payload.model_key !== 'all') {
          setActiveModelKey(payload.model_key);
        }
        if (payload.file !== undefined) {
          setActiveFile(payload.file);
        }
        if (payload.file_index !== undefined) {
          setFileIndex(payload.file_index);
        }
        if (payload.file_count !== undefined) {
          setFileCount(payload.file_count);
        }
        if (payload.message) {
          setStatusMessage(payload.message);
        }

        if (payload.status === 'starting' || payload.status === 'file_completed') {
          appendLog(payload.message);
        }

        if (payload.status === 'file_completed' && payload.model_key) {
          // Optimistically refresh status in background when a file finishes
          fetch(`${apiBaseUrl}/api/models/status`)
            .then((r) => (r.ok ? r.json() : null))
            .then((data) => {
              if (data?.models) {
                setModelsState(data.models);
              }
            })
            .catch(() => {});
        }

        if (payload.status === 'completed' || payload.status === 'skipped') {
          appendLog(payload.message || 'All models verified.');
          setActiveModelKey(null);
          setActiveFile(null);
          es.close();
          eventSourceRef.current = null;
          refreshStatusAndFinish();
        }
      } catch (parseErr) {
        console.error('Failed to parse SSE payload:', parseErr);
      }
    };

    es.onerror = () => {
      // EventSource fires onerror if connection closes unexpectedly before completion
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      setStreamStatus((prev) => {
        if (prev === 'completed') return prev;
        setErrorMessage(
          'Lost connection to backend SSE endpoint (http://localhost:8000/api/models/download). Ensure FastAPI server is running.'
        );
        return 'error';
      });
    };
  };

  useEffect(() => {
    if (!isOpen) {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      return;
    }

    // Automatically begin SSE download stream when modal opens and models are incomplete
    if (!initialStatus || !initialStatus.all_ready) {
      startDownloadStream();
    }

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  if (!isOpen) return null;

  const modelKeys = Object.keys(MODEL_METADATA);
  const installedCount = modelKeys.filter((k) => modelsState?.[k]?.installed).length;
  const isAllInstalled =
    installedCount === modelKeys.length && streamStatus === 'completed';

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/90 backdrop-blur-md p-4 select-none"
      role="dialog"
      aria-modal="true"
      aria-labelledby="download-modal-title"
    >
      <div className="w-full max-w-2xl rounded-2xl border border-slate-800 bg-slate-900/95 shadow-2xl shadow-black/80 overflow-hidden">
        {/* Top Accent Banner */}
        <div className="h-1.5 w-full bg-slate-800 overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-indigo-500 via-violet-500 to-emerald-400 transition-all duration-300 ease-out"
            style={{ width: `${Math.min(100, Math.max(2, overallPercent))}%` }}
          />
        </div>

        {/* Header */}
        <div className="px-6 pt-6 pb-4 border-b border-slate-800/80 flex items-start justify-between gap-4">
          <div className="flex items-start gap-3.5">
            <div className="mt-0.5 flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-indigo-500/15 border border-indigo-500/30 text-indigo-400">
              {streamStatus === 'completed' ? (
                <ShieldCheck className="h-6 w-6 text-emerald-400" />
              ) : streamStatus === 'error' ? (
                <AlertCircle className="h-6 w-6 text-rose-400" />
              ) : (
                <Download className="h-6 w-6 animate-bounce" />
              )}
            </div>
            <div>
              <div className="flex items-center gap-2.5 flex-wrap">
                <h2
                  id="download-modal-title"
                  className="text-lg font-semibold tracking-tight text-slate-100"
                >
                  {isAllInstalled
                    ? 'Local AI Models Ready'
                    : 'Initializing Local AI Models'}
                </h2>
                <span className="inline-flex items-center gap-1.5 rounded-full bg-slate-800/90 border border-slate-700/80 px-2.5 py-0.5 text-xs font-medium text-slate-300">
                  <Cpu className="h-3.5 w-3.5 text-indigo-400" />
                  {executionProvider}
                </span>
              </div>
              <p className="mt-1 text-xs text-slate-400">
                Required ONNX & OCR weights are verified locally in{' '}
                <code className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[11px] text-indigo-300">
                  models/
                </code>{' '}
                before unlocking the translation canvas.
              </p>
            </div>
          </div>

          <div className="text-right shrink-0">
            <span className="inline-flex items-center gap-1.5 rounded-lg bg-slate-800/70 border border-slate-700/60 px-2.5 py-1 text-xs font-mono text-slate-300">
              <Database className="h-3.5 w-3.5 text-violet-400" />
              {installedCount}/{modelKeys.length} Groups
            </span>
          </div>
        </div>

        {/* Body */}
        <div className="p-6 space-y-5">
          {/* Real-time Overall Progress Section */}
          <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-4 space-y-3">
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-2 font-medium text-slate-200 truncate">
                {streamStatus === 'downloading' || streamStatus === 'connecting' ? (
                  <Loader2 className="h-4 w-4 shrink-0 animate-spin text-indigo-400" />
                ) : streamStatus === 'completed' ? (
                  <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-400" />
                ) : (
                  <AlertCircle className="h-4 w-4 shrink-0 text-rose-400" />
                )}
                <span className="truncate">{statusMessage}</span>
              </div>
              <span className="font-mono font-semibold text-indigo-300 shrink-0 ml-3">
                {overallPercent.toFixed(1)}%
              </span>
            </div>

            {/* Overall Progress Bar */}
            <div className="h-3 w-full rounded-full bg-slate-800/90 overflow-hidden p-0.5 border border-slate-700/50">
              <div
                className={`h-full rounded-full transition-all duration-200 ease-out ${
                  streamStatus === 'error'
                    ? 'bg-rose-500'
                    : streamStatus === 'completed'
                    ? 'bg-emerald-500'
                    : 'bg-gradient-to-r from-indigo-500 via-violet-500 to-indigo-400'
                }`}
                style={{ width: `${Math.min(100, Math.max(0, overallPercent))}%` }}
              />
            </div>

            {/* Sub-bar & Byte Counter for Active File */}
            <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
              <div className="truncate">
                {activeFile ? (
                  <span>
                    File {fileIndex && fileCount ? `[${fileIndex}/${fileCount}]` : ''}:{' '}
                    <span className="text-slate-300">{activeFile}</span> ({filePercent.toFixed(1)}%)
                  </span>
                ) : (
                  <span>100% Offline Inference • Zero Cloud Dependency</span>
                )}
              </div>
              {totalBytes > 0 && (
                <div className="shrink-0 ml-2 text-slate-300">
                  {formatMegabytes(downloadedBytes)} / {formatMegabytes(totalBytes)}
                </div>
              )}
            </div>
          </div>

          {/* Model Groups Status Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {modelKeys.map((key) => {
              const meta = MODEL_METADATA[key];
              const groupStatus = modelsState?.[key];
              const isInstalled = Boolean(groupStatus?.installed);
              const isCurrentlyDownloading =
                activeModelKey === key && streamStatus === 'downloading';
              const sizeMb = groupStatus?.size_mb || meta.defaultSizeMb;
              const downloadedMb = groupStatus?.downloaded_mb ?? (isInstalled ? sizeMb : 0);

              return (
                <div
                  key={key}
                  className={`rounded-xl border p-3.5 transition-colors ${
                    isCurrentlyDownloading
                      ? 'border-indigo-500/60 bg-indigo-950/25'
                      : isInstalled
                      ? 'border-emerald-500/25 bg-slate-950/50'
                      : 'border-slate-800 bg-slate-950/40'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="font-medium text-xs text-slate-200 leading-snug">
                      {meta.label}
                    </div>
                    {isInstalled ? (
                      <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 border border-emerald-500/30 px-2 py-0.5 text-[10px] font-medium text-emerald-300 shrink-0">
                        <CheckCircle2 className="h-3 w-3" />
                        Ready
                      </span>
                    ) : isCurrentlyDownloading ? (
                      <span className="inline-flex items-center gap-1 rounded-full bg-indigo-500/20 border border-indigo-500/40 px-2 py-0.5 text-[10px] font-medium text-indigo-300 shrink-0">
                        <Loader2 className="h-3 w-3 animate-spin" />
                        Syncing
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/15 border border-amber-500/30 px-2 py-0.5 text-[10px] font-medium text-amber-300 shrink-0">
                        Pending
                      </span>
                    )}
                  </div>

                  <p className="mt-1 text-[11px] text-slate-400 line-clamp-2">
                    {meta.description}
                  </p>

                  <div className="mt-2.5 flex items-center justify-between text-[11px] font-mono text-slate-400">
                    <span className="inline-flex items-center gap-1">
                      <HardDrive className="h-3 w-3 text-slate-500" />
                      {isInstalled ? `${downloadedMb} MB` : `~${sizeMb} MB`}
                    </span>
                    {groupStatus?.missing_files?.length > 0 && (
                      <span className="text-amber-400/90">
                        {groupStatus.missing_files.length} file(s) missing
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Live SSE Activity Log */}
          {recentLogs.length > 0 && (
            <div className="rounded-xl border border-slate-800/80 bg-slate-950/90 px-3.5 py-2.5 font-mono text-[11px] text-slate-400 space-y-1 max-h-24 overflow-y-auto">
              {recentLogs.map((logItem, idx) => (
                <div key={idx} className="truncate">
                  <span className="text-indigo-400 mr-1.5">&gt;</span>
                  {logItem}
                </div>
              ))}
            </div>
          )}

          {/* Error Banner */}
          {errorMessage && (
            <div className="rounded-xl border border-rose-500/40 bg-rose-950/40 px-4 py-3 flex items-center justify-between gap-3">
              <div className="flex items-center gap-2.5 text-xs text-rose-200">
                <AlertCircle className="h-4 w-4 shrink-0 text-rose-400" />
                <span>{errorMessage}</span>
              </div>
              <button
                type="button"
                onClick={startDownloadStream}
                className="inline-flex items-center gap-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 px-3 py-1.5 text-xs font-medium text-white transition-colors shrink-0"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                Retry Download
              </button>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-slate-950/80 border-t border-slate-800/80 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Sparkles className="h-4 w-4 text-indigo-400" />
            <span>
              {isAllInstalled
                ? 'All models verified in local storage.'
                : 'Workspace will unlock automatically once all weights are verified.'}
            </span>
          </div>

          <div className="flex items-center gap-2.5">
            {streamStatus !== 'downloading' && streamStatus !== 'connecting' && !isAllInstalled && (
              <button
                type="button"
                onClick={startDownloadStream}
                className="inline-flex items-center gap-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 px-4 py-2 text-xs font-semibold text-white shadow-lg shadow-indigo-600/30 transition-all"
              >
                <Download className="h-3.5 w-3.5" />
                Download Missing Models
              </button>
            )}

            {isAllInstalled && onClose && (
              <button
                type="button"
                onClick={onClose}
                className="inline-flex items-center gap-1.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 px-4 py-2 text-xs font-semibold text-white shadow-lg shadow-emerald-600/25 transition-all"
              >
                <CheckCircle2 className="h-3.5 w-3.5" />
                Enter Workspace
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
