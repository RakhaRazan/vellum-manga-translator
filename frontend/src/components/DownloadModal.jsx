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
} from 'lucide-react';

import { DEFAULT_API_BASE_URL } from '../store/useEditorStore.js';

const API_BASE_URL = DEFAULT_API_BASE_URL;

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
 * Blocking desktop studio modal that connects to `/api/models/download` via SSE
 * when local CV/OCR models are incomplete, streaming real-time download progress
 * before unlocking the workspace.
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
    initialStatus?.execution_provider || 'Detecting…'
  );
  const [streamStatus, setStreamStatus] = useState('idle');
  const [activeModelKey, setActiveModelKey] = useState(null);
  const [activeFile, setActiveFile] = useState(null);
  const [fileIndex, setFileIndex] = useState(null);
  const [fileCount, setFileCount] = useState(null);
  const [filePercent, setFilePercent] = useState(0);
  const [overallPercent, setOverallPercent] = useState(0);
  const [downloadedBytes, setDownloadedBytes] = useState(0);
  const [totalBytes, setTotalBytes] = useState(0);
  const [statusMessage, setStatusMessage] = useState(
    'Preparing local ONNX & OCR model verification…'
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
    setStatusMessage('Connecting to /api/models/download SSE stream…');
    appendLog(`Connecting to ${apiBaseUrl}/api/models/download ...`);

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
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      setStreamStatus((prev) => {
        if (prev === 'completed') return prev;
        setErrorMessage(
          `Lost connection to backend SSE endpoint (${apiBaseUrl}/api/models/download). Ensure native C++17 backend server is running.`
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
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 select-none"
      role="dialog"
      aria-modal="true"
      aria-labelledby="download-modal-title"
    >
      <div className="w-full max-w-2xl rounded-studio-md border border-studio-border bg-studio-panel overflow-hidden">
        {/* Top 2px Progress Hairline */}
        <div className="h-0.5 w-full bg-studio-border overflow-hidden">
          <div
            className={`h-full transition-all duration-300 ease-out ${
              streamStatus === 'error'
                ? 'bg-studio-danger'
                : streamStatus === 'completed'
                ? 'bg-studio-success'
                : 'bg-studio-accent'
            }`}
            style={{ width: `${Math.min(100, Math.max(2, overallPercent))}%` }}
          />
        </div>

        {/* Header */}
        <div className="px-5 py-4 border-b border-studio-border flex items-start justify-between gap-4">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-studio border border-studio-border bg-studio-surface text-studio-accent">
              {streamStatus === 'completed' ? (
                <ShieldCheck className="h-5 w-5 text-studio-success" />
              ) : streamStatus === 'error' ? (
                <AlertCircle className="h-5 w-5 text-studio-danger" />
              ) : (
                <Download className="h-5 w-5" />
              )}
            </div>
            <div>
              <div className="flex items-center gap-2 flex-wrap">
                <h2
                  id="download-modal-title"
                  className="text-sm font-semibold tracking-tight text-studio-text"
                >
                  {isAllInstalled
                    ? 'Local AI Models Ready'
                    : 'Initializing Local AI Models'}
                </h2>
                <span className="inline-flex items-center gap-1 rounded-studio-sm bg-studio-surface border border-studio-border px-2 py-0.5 font-mono text-[11px] text-studio-text-secondary">
                  <Cpu className="h-3 w-3 text-studio-accent" />
                  {executionProvider}
                </span>
              </div>
              <p className="mt-1 text-xs text-studio-text-secondary">
                Required ONNX &amp; OCR weights are verified locally in{' '}
                <code className="rounded-studio-sm border border-studio-border bg-studio-surface px-1 py-0.5 font-mono text-[11px] text-studio-accent">
                  models/
                </code>{' '}
                before unlocking the studio canvas.
              </p>
            </div>
          </div>

          <div className="text-right shrink-0">
            <span className="inline-flex items-center gap-1.5 rounded-studio bg-studio-surface border border-studio-border px-2 py-1 text-[11px] font-mono text-studio-text-secondary tabular-nums">
              <Database className="h-3.5 w-3.5 text-studio-accent" />
              {installedCount}/{modelKeys.length} Groups
            </span>
          </div>
        </div>

        {/* Body */}
        <div className="p-5 space-y-4">
          {/* Overall Progress Section */}
          <div className="rounded-studio border border-studio-border bg-studio-surface p-3.5 space-y-2.5">
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-2 font-medium text-studio-text truncate">
                {streamStatus === 'downloading' || streamStatus === 'connecting' ? (
                  <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-studio-accent" />
                ) : streamStatus === 'completed' ? (
                  <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-studio-success" />
                ) : (
                  <AlertCircle className="h-3.5 w-3.5 shrink-0 text-studio-danger" />
                )}
                <span className="truncate">{statusMessage}</span>
              </div>
              <span className="font-mono font-semibold text-studio-accent shrink-0 ml-3 tabular-nums">
                {overallPercent.toFixed(1)}%
              </span>
            </div>

            <div className="h-2 w-full rounded-studio-sm bg-studio-panel overflow-hidden border border-studio-border">
              <div
                className={`h-full transition-all duration-200 ease-out ${
                  streamStatus === 'error'
                    ? 'bg-studio-danger'
                    : streamStatus === 'completed'
                    ? 'bg-studio-success'
                    : 'bg-studio-accent'
                }`}
                style={{ width: `${Math.min(100, Math.max(0, overallPercent))}%` }}
              />
            </div>

            <div className="flex items-center justify-between text-[11px] text-studio-text-muted font-mono">
              <div className="truncate">
                {activeFile ? (
                  <span>
                    File {fileIndex && fileCount ? `[${fileIndex}/${fileCount}]` : ''}:{' '}
                    <span className="text-studio-text">{activeFile}</span> ({filePercent.toFixed(1)}%)
                  </span>
                ) : (
                  <span>100% Offline Inference · Zero Cloud Dependency</span>
                )}
              </div>
              {totalBytes > 0 && (
                <div className="shrink-0 ml-2 text-studio-text-secondary tabular-nums">
                  {formatMegabytes(downloadedBytes)} / {formatMegabytes(totalBytes)}
                </div>
              )}
            </div>
          </div>

          {/* Model Groups Status Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
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
                  className={`rounded-studio border p-3 transition-colors ${
                    isCurrentlyDownloading
                      ? 'border-studio-accent bg-studio-accent/10'
                      : isInstalled
                      ? 'border-studio-border bg-studio-surface'
                      : 'border-studio-border bg-studio-surface/60'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="font-medium text-xs text-studio-text leading-snug">
                      {meta.label}
                    </div>
                    {isInstalled ? (
                      <span className="inline-flex items-center gap-1 rounded-studio-sm bg-studio-success/15 border border-studio-success/40 px-1.5 py-0.5 text-[10px] font-medium text-studio-success shrink-0">
                        <CheckCircle2 className="h-3 w-3" />
                        Ready
                      </span>
                    ) : isCurrentlyDownloading ? (
                      <span className="inline-flex items-center gap-1 rounded-studio-sm bg-studio-accent/15 border border-studio-accent/40 px-1.5 py-0.5 text-[10px] font-medium text-studio-accent shrink-0">
                        <Loader2 className="h-3 w-3 animate-spin" />
                        Syncing
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 rounded-studio-sm bg-studio-warning/15 border border-studio-warning/40 px-1.5 py-0.5 text-[10px] font-medium text-studio-warning shrink-0">
                        Pending
                      </span>
                    )}
                  </div>

                  <p className="mt-1 text-[11px] text-studio-text-secondary line-clamp-2">
                    {meta.description}
                  </p>

                  <div className="mt-2 flex items-center justify-between text-[11px] font-mono text-studio-text-muted tabular-nums">
                    <span className="inline-flex items-center gap-1">
                      <HardDrive className="h-3 w-3" />
                      {isInstalled ? `${downloadedMb} MB` : `~${sizeMb} MB`}
                    </span>
                    {groupStatus?.missing_files?.length > 0 && (
                      <span className="text-studio-warning">
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
            <div className="rounded-studio border border-studio-border bg-studio-canvas px-3 py-2 font-mono text-[11px] text-studio-text-secondary space-y-1 max-h-24 overflow-y-auto">
              {recentLogs.map((logItem, idx) => (
                <div key={idx} className="truncate">
                  <span className="text-studio-accent mr-1.5">&gt;</span>
                  {logItem}
                </div>
              ))}
            </div>
          )}

          {/* Error Banner */}
          {errorMessage && (
            <div className="rounded-studio border border-studio-danger/50 bg-studio-danger/10 px-3.5 py-2.5 flex items-center justify-between gap-3">
              <div className="flex items-center gap-2 text-xs text-studio-danger">
                <AlertCircle className="h-4 w-4 shrink-0" />
                <span>{errorMessage}</span>
              </div>
              <button
                type="button"
                onClick={startDownloadStream}
                className="studio-btn h-7 px-2.5 text-xs font-medium border-studio-danger/50 text-studio-danger hover:bg-studio-danger/15 shrink-0"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                Retry
              </button>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 bg-studio-surface border-t border-studio-border flex items-center justify-between gap-4">
          <div className="text-xs text-studio-text-secondary">
            {isAllInstalled
              ? 'All local weights verified in storage.'
              : 'Workspace unlocks automatically once all weights are verified.'}
          </div>

          <div className="flex items-center gap-2">
            {streamStatus !== 'downloading' &&
              streamStatus !== 'connecting' &&
              !isAllInstalled && (
                <button
                  type="button"
                  onClick={startDownloadStream}
                  className="studio-btn-accent h-7 px-3 text-xs"
                >
                  <Download className="h-3.5 w-3.5" />
                  Download Missing Models
                </button>
              )}

            {isAllInstalled && onClose && (
              <button
                type="button"
                onClick={onClose}
                className="studio-btn-accent h-7 px-3 text-xs"
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
