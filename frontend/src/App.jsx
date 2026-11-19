import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  Clock,
  FileImage,
  Layers,
  Loader2,
  RefreshCw,
  Trash2,
} from 'lucide-react';
import DownloadModal from './components/DownloadModal.jsx';
import Toolbar from './components/Toolbar.jsx';
import MangaCanvas from './components/MangaCanvas.jsx';
import TextSidebar, { COMIC_FONT_FAMILIES } from './components/TextSidebar.jsx';

const API_BASE_URL = 'http://localhost:8000';

/**
 * Normalize raw blocks returned from `/api/process-page` into canvas-ready block objects
 * with explicit `x`, `y`, `width`, `height`, `rotation`, and style properties.
 */
function normalizeBlocks(rawBlocks = []) {
  return rawBlocks.map((b, index) => {
    const bbox = Array.isArray(b.bbox) && b.bbox.length >= 4 ? b.bbox : [60, 60, 220, 140];
    const x1 = Number(bbox[0]) || 0;
    const y1 = Number(bbox[1]) || 0;
    const x2 = Number(bbox[2]) || x1 + 140;
    const y2 = Number(bbox[3]) || y1 + 60;
    const width = Math.max(36, x2 - x1);
    const height = Math.max(24, y2 - y1);
    const fontSize = b.font_size || b.style?.font_size || 18;
    const color = b.color || b.style?.color || '#000000';
    const align = b.align || b.style?.align || 'center';
    const fontFamily =
      b.font_family || b.style?.font_family || COMIC_FONT_FAMILIES[0].value;
    const strokeColor = b.stroke_color || b.style?.stroke_color || '#ffffff';
    const strokeWidth =
      b.stroke_width !== undefined
        ? b.stroke_width
        : b.style?.stroke_width !== undefined
        ? b.style.stroke_width
        : 4;
    const lineHeight = b.line_height || b.style?.line_height || 1.18;

    return {
      ...b,
      id: b.id || `block_${String(index + 1).padStart(3, '0')}`,
      order: b.order ?? index + 1,
      bbox: [x1, y1, x1 + width, y1 + height],
      x: typeof b.x === 'number' ? b.x : x1,
      y: typeof b.y === 'number' ? b.y : y1,
      width: typeof b.width === 'number' ? b.width : width,
      height: typeof b.height === 'number' ? b.height : height,
      rotation: typeof b.rotation === 'number' ? b.rotation : 0,
      original_text: b.original_text ?? b.source_text ?? '',
      source_text: b.source_text ?? b.original_text ?? '',
      translated_text: b.translated_text ?? '',
      font_family: fontFamily,
      font_size: fontSize,
      color,
      stroke_color: strokeColor,
      stroke_width: strokeWidth,
      align,
      line_height: lineHeight,
      style: {
        font_family: fontFamily,
        font_size: fontSize,
        color,
        stroke_color: strokeColor,
        stroke_width: strokeWidth,
        align,
        line_height: lineHeight,
      },
    };
  });
}

export default function App() {
  // Startup model verification state
  const [startupChecking, setStartupChecking] = useState(true);
  const [startupError, setStartupError] = useState(null);
  const [modelsStatus, setModelsStatus] = useState(null);
  const [isWorkspaceUnlocked, setIsWorkspaceUnlocked] = useState(false);
  const [isDownloadModalOpen, setIsDownloadModalOpen] = useState(false);

  // Translation language configuration
  const [sourceLang, setSourceLang] = useState('ja');
  const [targetLang, setTargetLang] = useState('id');

  // Multi-page batch queue & active editor state
  const [pages, setPages] = useState([]);
  const [activePageId, setActivePageId] = useState(null);
  const [selectedBlockId, setSelectedBlockId] = useState(null);
  const [isRetranslating, setIsRetranslating] = useState(false);
  const [statusBanner, setStatusBanner] = useState(null);

  const canvasRef = useRef(null);
  const isQueueWorkerRunningRef = useRef(false);

  // 1. Startup Check: Query GET http://localhost:8000/api/models/status (with retry while Tauri sidecar boots)
  const checkModelsStatus = useCallback(async () => {
    setStartupChecking(true);
    setStartupError(null);

    // If running inside Tauri shell and sidecar isn't up yet, allow @tauri-apps/plugin-shell fallback spawn
    if (typeof window !== 'undefined' && window.__TAURI_INTERNALS__) {
      import('@tauri-apps/plugin-shell')
        .then(({ Command }) => {
          const cmd = Command.sidecar('binaries/comic-backend');
          return cmd.spawn();
        })
        .then((child) => {
          window.__COMIC_BACKEND_CHILD__ = child;
        })
        .catch(() => {
          // Sidecar already spawned by Rust host in src-tauri/src/main.rs
        });
    }

    const maxAttempts = 12;
    let lastErr = null;

    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      try {
        const response = await fetch(`${API_BASE_URL}/api/models/status`, {
          method: 'GET',
          headers: { Accept: 'application/json' },
        });
        if (!response.ok) {
          throw new Error(`Backend returned HTTP ${response.status}`);
        }
        const data = await response.json();
        setModelsStatus(data);

        if (data.all_ready) {
          setIsWorkspaceUnlocked(true);
          setIsDownloadModalOpen(false);
        } else {
          setIsWorkspaceUnlocked(false);
          setIsDownloadModalOpen(true);
        }
        setStartupChecking(false);
        return;
      } catch (err) {
        lastErr = err;
        if (attempt < maxAttempts) {
          await new Promise((r) => setTimeout(r, 1200));
        }
      }
    }

    setStartupError(
      lastErr?.message ||
        'Unable to reach FastAPI backend at http://localhost:8000/api/models/status'
    );
    setIsWorkspaceUnlocked(false);
    setStartupChecking(false);
  }, []);

  useEffect(() => {
    checkModelsStatus();
    const handleUnload = () => {
      if (window.__COMIC_BACKEND_CHILD__ && typeof window.__COMIC_BACKEND_CHILD__.kill === 'function') {
        window.__COMIC_BACKEND_CHILD__.kill().catch(() => {});
      }
    };
    window.addEventListener('beforeunload', handleUnload);
    return () => window.removeEventListener('beforeunload', handleUnload);
  }, [checkModelsStatus]);

  const handleModelsReady = (updatedStatus) => {
    setModelsStatus(updatedStatus);
    setIsWorkspaceUnlocked(true);
    setIsDownloadModalOpen(false);
  };

  // Active page lookup
  const activePage = useMemo(
    () => pages.find((p) => p.id === activePageId) || null,
    [pages, activePageId]
  );

  // Batch queue statistics
  const batchStats = useMemo(() => {
    const total = pages.length;
    const completed = pages.filter((p) => p.status === 'done').length;
    const processing = pages.filter((p) => p.status === 'processing').length;
    const queued = pages.filter((p) => p.status === 'queued').length;
    return { total, completed, processing, queued };
  }, [pages]);

  // Enqueue one or multiple imported image files into the batch queue
  const handleImportFiles = useCallback(
    (fileList) => {
      const validFiles = Array.from(fileList || []).filter((f) =>
        f.type ? f.type.startsWith('image/') : /\.(jpe?g|png|webp|bmp)$/i.test(f.name)
      );
      if (validFiles.length === 0) return;

      const timestamp = Date.now();
      const newPages = validFiles.map((file, idx) => ({
        id: `page_${timestamp}_${idx}_${Math.random().toString(36).slice(2, 7)}`,
        name: file.name,
        file,
        originalUrl: URL.createObjectURL(file),
        cleanedImage: null,
        dimensions: null,
        blocks: [],
        status: 'queued', // queued | processing | done | error
        error: null,
        sourceLang,
        targetLang,
        model: null,
        tilesInpainted: 0,
      }));

      setPages((prev) => [...prev, ...newPages]);
      setActivePageId((prevActive) => prevActive || newPages[0].id);
      setSelectedBlockId(null);
      setStatusBanner(null);
    },
    [sourceLang, targetLang]
  );

  // Process a single page via POST http://localhost:8000/api/process-page
  const processSinglePage = useCallback(
    async (pageItem, overrideSourceLang, overrideTargetLang) => {
      const src = overrideSourceLang || pageItem.sourceLang || sourceLang;
      const tgt = overrideTargetLang || pageItem.targetLang || targetLang;

      setPages((prev) =>
        prev.map((p) =>
          p.id === pageItem.id
            ? { ...p, status: 'processing', error: null, sourceLang: src, targetLang: tgt }
            : p
        )
      );

      try {
        const formData = new FormData();
        formData.append('image', pageItem.file, pageItem.name);
        formData.append('source_lang', src);
        formData.append('target_lang', tgt);

        const response = await fetch(`${API_BASE_URL}/api/process-page`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          let errDetail = `HTTP ${response.status}`;
          try {
            const errJson = await response.json();
            if (errJson?.detail) errDetail = errJson.detail;
          } catch {
            // ignore JSON parse failure
          }
          throw new Error(errDetail);
        }

        const data = await response.json();
        const normalizedBlocks = normalizeBlocks(data.blocks || data.bubbles || []);

        setPages((prev) =>
          prev.map((p) =>
            p.id === pageItem.id
              ? {
                  ...p,
                  status: 'done',
                  error: null,
                  cleanedImage: data.cleaned_image || data.inpainted_image,
                  dimensions: data.dimensions || null,
                  blocks: normalizedBlocks,
                  sourceLang: data.source_lang || src,
                  targetLang: data.target_lang || tgt,
                  model: data.model || null,
                  tilesInpainted: data.tiles_inpainted ?? 0,
                }
              : p
          )
        );
      } catch (err) {
        const msg = err?.message || 'Failed to process comic page.';
        setPages((prev) =>
          prev.map((p) =>
            p.id === pageItem.id ? { ...p, status: 'error', error: msg } : p
          )
        );
        setStatusBanner({
          type: 'error',
          message: `Error processing ${pageItem.name}: ${msg}`,
        });
      }
    },
    [sourceLang, targetLang]
  );

  // Sequential batch queue processor (processes one queued page at a time to protect VRAM)
  useEffect(() => {
    if (!isWorkspaceUnlocked) return;
    if (isQueueWorkerRunningRef.current) return;

    const currentlyProcessing = pages.some((p) => p.status === 'processing');
    if (currentlyProcessing) return;

    const nextQueuedPage = pages.find((p) => p.status === 'queued');
    if (!nextQueuedPage) return;

    isQueueWorkerRunningRef.current = true;
    processSinglePage(nextQueuedPage).finally(() => {
      isQueueWorkerRunningRef.current = false;
    });
  }, [pages, isWorkspaceUnlocked, processSinglePage]);

  // Update a specific block's properties (position, size, original_text, translated_text, font family, size, color, white outline stroke)
  const handleUpdateBlock = useCallback(
    (blockId, updates) => {
      if (!activePageId) return;
      setPages((prev) =>
        prev.map((page) => {
          if (page.id !== activePageId) return page;
          return {
            ...page,
            blocks: (page.blocks || []).map((b) => {
              if (b.id !== blockId) return b;
              const nextFontFamily =
                updates.font_family ??
                updates.style?.font_family ??
                b.style?.font_family ??
                b.font_family ??
                COMIC_FONT_FAMILIES[0].value;
              const nextFontSize =
                updates.font_size ??
                updates.style?.font_size ??
                b.font_size ??
                18;
              const nextColor =
                updates.color ?? updates.style?.color ?? b.color ?? '#000000';
              const nextAlign =
                updates.align ?? updates.style?.align ?? b.align ?? 'center';
              const nextStrokeColor =
                updates.stroke_color ??
                updates.style?.stroke_color ??
                b.style?.stroke_color ??
                b.stroke_color ??
                '#ffffff';
              const nextStrokeWidth =
                updates.stroke_width !== undefined
                  ? updates.stroke_width
                  : updates.style?.stroke_width !== undefined
                  ? updates.style.stroke_width
                  : b.style?.stroke_width !== undefined
                  ? b.style.stroke_width
                  : 4;
              const nextLineHeight =
                updates.line_height ??
                updates.style?.line_height ??
                b.style?.line_height ??
                1.18;

              return {
                ...b,
                ...updates,
                font_family: nextFontFamily,
                font_size: nextFontSize,
                color: nextColor,
                stroke_color: nextStrokeColor,
                stroke_width: nextStrokeWidth,
                align: nextAlign,
                line_height: nextLineHeight,
                style: {
                  ...(b.style || {}),
                  ...(updates.style || {}),
                  font_family: nextFontFamily,
                  font_size: nextFontSize,
                  color: nextColor,
                  stroke_color: nextStrokeColor,
                  stroke_width: nextStrokeWidth,
                  align: nextAlign,
                  line_height: nextLineHeight,
                },
              };
            }),
          };
        })
      );
    },
    [activePageId]
  );

  // Apply typography style settings to all blocks on the current active page
  const handleApplyStyleToAll = useCallback(
    (styleUpdates) => {
      if (!activePageId) return;
      setPages((prev) =>
        prev.map((page) => {
          if (page.id !== activePageId) return page;
          return {
            ...page,
            blocks: (page.blocks || []).map((b) => ({
              ...b,
              font_family: styleUpdates.font_family ?? b.font_family,
              font_size: styleUpdates.font_size ?? b.font_size,
              color: styleUpdates.color ?? b.color,
              stroke_color: styleUpdates.stroke_color ?? b.stroke_color,
              stroke_width: styleUpdates.stroke_width ?? b.stroke_width,
              align: styleUpdates.align ?? b.align,
              line_height: styleUpdates.line_height ?? b.line_height,
              style: {
                ...(b.style || {}),
                ...styleUpdates,
              },
            })),
          };
        })
      );
      setStatusBanner({
        type: 'success',
        message: 'Applied typography and outline stroke settings to all bubbles on this page.',
      });
    },
    [activePageId]
  );

  // Re-translate current page dialogue bubbles via POST http://localhost:8000/api/retranslate
  const handleRetranslate = useCallback(async () => {
    if (!activePage) return;

    // If the page hasn't completed initial CV/OCR or had an error, run full processSinglePage
    if (!activePage.cleanedImage || !activePage.blocks || activePage.blocks.length === 0) {
      await processSinglePage(activePage, sourceLang, targetLang);
      return;
    }

    // If user switched source language (e.g. ja -> ko), re-run full OCR & detection pipeline
    if (activePage.sourceLang && activePage.sourceLang !== sourceLang) {
      await processSinglePage(activePage, sourceLang, targetLang);
      return;
    }

    setIsRetranslating(true);
    setStatusBanner(null);

    try {
      const payload = {
        source_lang: sourceLang,
        target_lang: targetLang,
        blocks: activePage.blocks.map((b) => ({
          id: b.id,
          order: b.order,
          bbox: b.bbox,
          polygon: b.polygon,
          original_text: b.original_text || b.source_text || '',
          source_text: b.source_text || b.original_text || '',
          translated_text: b.translated_text || '',
          font_size: b.font_size,
          color: b.color || '#000000',
          align: b.align || 'center',
        })),
      };

      const response = await fetch(`${API_BASE_URL}/api/retranslate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        let errDetail = `HTTP ${response.status}`;
        try {
          const errJson = await response.json();
          if (errJson?.detail) errDetail = errJson.detail;
        } catch {
          // ignore
        }
        throw new Error(errDetail);
      }

      const data = await response.json();
      const translationsMap = data.translations || {};
      const returnedBlocks = data.blocks || data.bubbles || [];
      const returnedById = Object.fromEntries(returnedBlocks.map((rb) => [rb.id, rb]));

      // Merge updated translations while preserving user's dragged/resized box geometry & typography
      setPages((prev) =>
        prev.map((p) => {
          if (p.id !== activePage.id) return p;
          return {
            ...p,
            targetLang: data.target_lang || targetLang,
            model: data.model || p.model,
            blocks: p.blocks.map((existingBlock) => {
              const updated = returnedById[existingBlock.id];
              const newText =
                translationsMap[existingBlock.id] ??
                updated?.translated_text ??
                existingBlock.translated_text;

              return {
                ...existingBlock,
                translated_text: newText,
              };
            }),
          };
        })
      );

      setStatusBanner({
        type: 'success',
        message: `Re-translated ${activePage.blocks.length} bubbles (${sourceLang.toUpperCase()} → ${targetLang.toUpperCase()}) via LM Studio.`,
      });
    } catch (err) {
      setStatusBanner({
        type: 'error',
        message: `Re-translation failed: ${err?.message || 'Unknown error'}`,
      });
    } finally {
      setIsRetranslating(false);
    }
  }, [activePage, sourceLang, targetLang, processSinglePage]);

  // Export current Konva canvas as a full-resolution .png or .jpg image
  const handleExportImage = useCallback(
    (format = 'png') => {
      if (!canvasRef.current || !activePage) return;
      const normalizedFmt =
        String(format).toLowerCase() === 'jpg' ||
        String(format).toLowerCase() === 'jpeg'
          ? 'jpg'
          : 'png';
      const dataUrl = canvasRef.current.exportImage(normalizedFmt, 0.98);
      if (!dataUrl) return;

      const baseName = activePage.name.replace(/\.[^/.]+$/, '') || 'comic_page';
      const filename = `${baseName}_${targetLang}_translated.${normalizedFmt}`;
      const link = document.createElement('a');
      link.download = filename;
      link.href = dataUrl;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      setStatusBanner({
        type: 'success',
        message: `Exported full-resolution ${filename} (${activePage.dimensions?.width || ''}×${activePage.dimensions?.height || ''}px).`,
      });
    },
    [activePage, targetLang]
  );

  // Add a custom manual text box onto the current page
  const handleAddTextBlock = useCallback(() => {
    if (!activePage) return;
    const nextIdx = (activePage.blocks?.length || 0) + 1;
    const newId = `block_custom_${Date.now().toString().slice(-4)}`;
    const centerW = Math.round((activePage.dimensions?.width || 800) / 2 - 80);
    const centerH = Math.round((activePage.dimensions?.height || 1200) / 2 - 40);

    const newBlock = {
      id: newId,
      order: nextIdx,
      bbox: [centerW, centerH, centerW + 160, centerH + 80],
      x: centerW,
      y: centerH,
      width: 160,
      height: 80,
      rotation: 0,
      original_text: '',
      source_text: '',
      translated_text: 'New Text',
      font_family: COMIC_FONT_FAMILIES[0].value,
      font_size: 20,
      color: '#000000',
      stroke_color: '#ffffff',
      stroke_width: 4,
      align: 'center',
      line_height: 1.18,
      style: {
        font_family: COMIC_FONT_FAMILIES[0].value,
        font_size: 20,
        color: '#000000',
        stroke_color: '#ffffff',
        stroke_width: 4,
        align: 'center',
        line_height: 1.18,
      },
    };

    setPages((prev) =>
      prev.map((p) =>
        p.id === activePage.id ? { ...p, blocks: [...(p.blocks || []), newBlock] } : p
      )
    );
    setSelectedBlockId(newId);
  }, [activePage]);

  // Delete a text block from the current page
  const handleDeleteBlock = useCallback(
    (blockId) => {
      if (!activePage) return;
      setPages((prev) =>
        prev.map((p) =>
          p.id === activePage.id
            ? { ...p, blocks: (p.blocks || []).filter((b) => b.id !== blockId) }
            : p
        )
      );
      if (selectedBlockId === blockId) {
        setSelectedBlockId(null);
      }
    },
    [activePage, selectedBlockId]
  );

  // Remove a page from the batch queue
  const handleRemovePage = (pageId) => {
    setPages((prev) => {
      const remaining = prev.filter((p) => p.id !== pageId);
      if (activePageId === pageId) {
        setActivePageId(remaining[0]?.id || null);
        setSelectedBlockId(null);
      }
      return remaining;
    });
  };

  // Load a bundled sample image (e.g. samples/sample_ja.jpg) and enqueue it
  const handleLoadSample = useCallback(
    async (filename = 'sample_ja.jpg', sampleLang = 'ja') => {
      try {
        setSourceLang(sampleLang);
        const resp = await fetch(`/samples/${filename}`);
        if (!resp.ok) {
          throw new Error(`Failed to load /samples/${filename}`);
        }
        const blob = await resp.blob();
        const file = new File([blob], filename, { type: blob.type || 'image/jpeg' });

        const timestamp = Date.now();
        const newPage = {
          id: `page_${timestamp}_0_${Math.random().toString(36).slice(2, 7)}`,
          name: filename,
          file,
          originalUrl: URL.createObjectURL(file),
          cleanedImage: null,
          dimensions: null,
          blocks: [],
          status: 'queued',
          error: null,
          sourceLang: sampleLang,
          targetLang,
          model: null,
          tilesInpainted: 0,
        };

        setPages((prev) => [...prev, newPage]);
        setActivePageId(newPage.id);
        setSelectedBlockId(null);
        setStatusBanner(null);
      } catch (err) {
        setStatusBanner({
          type: 'error',
          message: err?.message || `Could not load sample ${filename}`,
        });
      }
    },
    [targetLang]
  );

  const isCurrentPageProcessing = activePage?.status === 'processing';

  return (
    <div className="min-h-screen h-screen w-screen flex flex-col bg-slate-950 text-slate-100 overflow-hidden select-none">
      {/* Top Toolbar */}
      <Toolbar
        apiBaseUrl={API_BASE_URL}
        sourceLang={sourceLang}
        onSourceLangChange={setSourceLang}
        targetLang={targetLang}
        onTargetLangChange={setTargetLang}
        onImportFiles={handleImportFiles}
        onRetranslate={handleRetranslate}
        onExportImage={handleExportImage}
        onOpenModelsModal={() => setIsDownloadModalOpen(true)}
        isProcessing={isCurrentPageProcessing}
        isRetranslating={isRetranslating}
        canRetranslate={Boolean(activePage)}
        canExport={Boolean(activePage && (activePage.cleanedImage || activePage.originalUrl))}
        isWorkspaceUnlocked={isWorkspaceUnlocked}
        modelsStatus={modelsStatus}
        batchStats={batchStats}
      />

      {/* Status Notification Bar */}
      {statusBanner && (
        <div
          className={`px-4 py-1.5 text-xs flex items-center justify-between border-b ${
            statusBanner.type === 'error'
              ? 'bg-rose-950/70 border-rose-500/40 text-rose-200'
              : 'bg-emerald-950/60 border-emerald-500/30 text-emerald-200'
          }`}
        >
          <div className="flex items-center gap-2 truncate">
            {statusBanner.type === 'error' ? (
              <AlertCircle className="h-3.5 w-3.5 shrink-0 text-rose-400" />
            ) : (
              <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-emerald-400" />
            )}
            <span className="truncate">{statusBanner.message}</span>
          </div>
          <button
            type="button"
            onClick={() => setStatusBanner(null)}
            className="ml-4 text-[11px] text-slate-400 hover:text-slate-200"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Main Editor Workspace */}
      <main className="relative flex-1 flex overflow-hidden">
        {/* Left Sidebar: Multi-Page Batch Queue */}
        <aside className="w-60 shrink-0 border-r border-slate-800/80 bg-slate-900/60 flex flex-col overflow-hidden">
          <div className="p-3.5 border-b border-slate-800/80 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Layers className="h-4 w-4 text-indigo-400" />
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                Page Queue
              </span>
            </div>
            <span className="rounded-md bg-slate-800 px-2 py-0.5 font-mono text-[11px] text-slate-300">
              {pages.length}
            </span>
          </div>

          <div className="flex-1 overflow-y-auto p-2.5 space-y-2">
            {pages.length === 0 ? (
              <div className="rounded-xl border border-dashed border-slate-800/90 p-4 text-center space-y-1.5 my-2">
                <FileImage className="h-6 w-6 text-slate-600 mx-auto" />
                <p className="text-xs font-medium text-slate-400">Queue Empty</p>
                <p className="text-[11px] text-slate-500">
                  Import single or multiple comic pages to start batch translation.
                </p>
              </div>
            ) : (
              pages.map((p, idx) => {
                const isActive = p.id === activePageId;
                return (
                  <div
                    key={p.id}
                    onClick={() => {
                      setActivePageId(p.id);
                      setSelectedBlockId(null);
                    }}
                    className={`group relative rounded-xl border p-2.5 transition-all cursor-pointer flex items-center gap-2.5 ${
                      isActive
                        ? 'border-indigo-500/70 bg-indigo-950/35 shadow-md'
                        : 'border-slate-800/80 bg-slate-950/50 hover:border-slate-700'
                    }`}
                  >
                    {/* Page Thumbnail */}
                    <div className="h-12 w-9 shrink-0 rounded-lg bg-slate-900 border border-slate-800 overflow-hidden flex items-center justify-center">
                      {p.cleanedImage || p.originalUrl ? (
                        <img
                          src={p.cleanedImage || p.originalUrl}
                          alt={p.name}
                          className="h-full w-full object-cover"
                        />
                      ) : (
                        <FileImage className="h-4 w-4 text-slate-600" />
                      )}
                    </div>

                    {/* Page Metadata */}
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between gap-1">
                        <span className="text-xs font-medium text-slate-200 truncate">
                          {idx + 1}. {p.name}
                        </span>
                      </div>

                      <div className="mt-1 flex items-center gap-1.5 text-[11px]">
                        {p.status === 'done' && (
                          <span className="inline-flex items-center gap-1 text-emerald-400 font-medium">
                            <CheckCircle2 className="h-3 w-3" />
                            {p.blocks?.length || 0} bubbles
                          </span>
                        )}
                        {p.status === 'processing' && (
                          <span className="inline-flex items-center gap-1 text-indigo-300 font-medium">
                            <Loader2 className="h-3 w-3 animate-spin" />
                            Translating...
                          </span>
                        )}
                        {p.status === 'queued' && (
                          <span className="inline-flex items-center gap-1 text-amber-300 font-medium">
                            <Clock className="h-3 w-3" />
                            Queued
                          </span>
                        )}
                        {p.status === 'error' && (
                          <span
                            className="inline-flex items-center gap-1 text-rose-400 font-medium truncate"
                            title={p.error}
                          >
                            <AlertCircle className="h-3 w-3 shrink-0" />
                            Failed
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Remove Page Button */}
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRemovePage(p.id);
                      }}
                      className="opacity-0 group-hover:opacity-100 p-1 rounded-lg hover:bg-slate-800 text-slate-500 hover:text-rose-400 transition-all"
                      title="Remove page from queue"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                );
              })
            )}
          </div>
        </aside>

        {/* Center Interactive Konva Canvas */}
        <MangaCanvas
          ref={canvasRef}
          page={activePage}
          selectedBlockId={selectedBlockId}
          onSelectBlock={setSelectedBlockId}
          onUpdateBlock={handleUpdateBlock}
          onImportDrop={handleImportFiles}
          onLoadSample={handleLoadSample}
          isProcessing={isCurrentPageProcessing}
        />

        {/* Right Sidebar: Synchronized Text Blocks & Real-time Typography Customizer */}
        <TextSidebar
          page={activePage}
          selectedBlockId={selectedBlockId}
          onSelectBlock={setSelectedBlockId}
          onUpdateBlock={handleUpdateBlock}
          onApplyStyleToAll={handleApplyStyleToAll}
          onAddBlock={handleAddTextBlock}
          onDeleteBlock={handleDeleteBlock}
        />

        {/* Initial Startup Check Overlay (while querying /api/models/status or if backend is down) */}
        {(startupChecking || startupError) && !isDownloadModalOpen && (
          <div className="fixed inset-0 z-40 flex items-center justify-center bg-slate-950/90 backdrop-blur-md p-4">
            <div className="w-full max-w-md rounded-2xl border border-slate-800 bg-slate-900 p-6 text-center space-y-4 shadow-2xl">
              {startupChecking ? (
                <>
                  <Loader2 className="h-9 w-9 animate-spin text-indigo-400 mx-auto" />
                  <div className="space-y-1">
                    <h3 className="text-sm font-semibold text-slate-100">
                      Checking Local AI Models...
                    </h3>
                    <p className="text-xs text-slate-400 font-mono">
                      GET {API_BASE_URL}/api/models/status
                    </p>
                  </div>
                </>
              ) : (
                <>
                  <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-rose-500/15 border border-rose-500/30 text-rose-400">
                    <AlertTriangle className="h-6 w-6" />
                  </div>
                  <div className="space-y-1">
                    <h3 className="text-base font-semibold text-slate-100">
                      Backend Connection Failed
                    </h3>
                    <p className="text-xs text-slate-400">{startupError}</p>
                  </div>
                  <button
                    type="button"
                    onClick={checkModelsStatus}
                    className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 px-4 py-2 text-xs font-semibold text-white transition-colors"
                  >
                    <RefreshCw className="h-3.5 w-3.5" />
                    Retry Connection
                  </button>
                </>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Dark-mode Blocking Model Download & SSE Verification Modal */}
      <DownloadModal
        isOpen={isDownloadModalOpen}
        initialStatus={modelsStatus}
        apiBaseUrl={API_BASE_URL}
        onModelsReady={handleModelsReady}
        onClose={
          modelsStatus?.all_ready ? () => setIsDownloadModalOpen(false) : undefined
        }
      />
    </div>
  );
}
