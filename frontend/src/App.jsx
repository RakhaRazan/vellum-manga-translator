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
import { useShallow } from 'zustand/react/shallow';
import DownloadModal from './components/DownloadModal.jsx';
import SettingsModal from './components/SettingsModal.jsx';
import MenuBar from './components/MenuBar.jsx';
import FrontControlBar from './components/FrontControlBar.jsx';
import MangaCanvas from './components/MangaCanvas.jsx';
import TextSidebar from './components/TextSidebar.jsx';
import StatusBar from './components/StatusBar.jsx';
import useEditorStore, {
  bindSystemThemeListener,
  ensurePageFontsLoaded,
  normalizeBlocks,
} from './store/useEditorStore.js';

function resolveBackendUrl(url, apiBaseUrl) {
  if (!url) return null;
  if (/^(data:|blob:|https?:\/\/)/i.test(url)) return url;
  if (url.startsWith('/')) {
    const base = String(apiBaseUrl || '').replace(/\/+$/, '');
    return base ? `${base}${url}` : url;
  }
  return url;
}

const naturalSortCollator = new Intl.Collator(undefined, {
  numeric: true,
  sensitivity: 'base',
});

async function resolvePageImageFile(pageItem) {
  if (pageItem?.file instanceof File || pageItem?.file instanceof Blob) {
    return pageItem.file;
  }
  const srcUrl = pageItem?.originalUrl || pageItem?.cleanedImage;
  if (!srcUrl) return null;
  const resp = await fetch(srcUrl);
  if (!resp.ok) {
    throw new Error(`Could not fetch source image for ${pageItem.name} (HTTP ${resp.status})`);
  }
  const blob = await resp.blob();
  return new File([blob], pageItem.name || 'page.png', {
    type: blob.type || 'image/png',
  });
}

function triggerBlobDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function extractDataTransferFiles(dataTransfer) {
  const items = Array.from(dataTransfer?.items || []);
  const hasEntries = items.some(
    (it) => typeof it.webkitGetAsEntry === 'function'
  );
  if (!hasEntries) {
    return Array.from(dataTransfer?.files || []);
  }

  const droppedFiles = [];
  const walkEntry = (entry, prefix = '') =>
    new Promise((resolve) => {
      if (!entry) {
        resolve();
        return;
      }
      if (entry.isFile) {
        entry.file(
          (file) => {
            if (prefix && !file.webkitRelativePath) {
              try {
                Object.defineProperty(file, 'webkitRelativePath', {
                  value: `${prefix}${file.name}`,
                  configurable: true,
                });
              } catch {
                // ignore non-configurable property
              }
            }
            droppedFiles.push(file);
            resolve();
          },
          () => resolve()
        );
        return;
      }
      if (entry.isDirectory) {
        const reader = entry.createReader();
        const readBatch = () => {
          reader.readEntries(
            async (batch) => {
              if (!batch || batch.length === 0) {
                resolve();
                return;
              }
              for (const child of batch) {
                await walkEntry(child, `${prefix}${entry.name}/`);
              }
              readBatch();
            },
            () => resolve()
          );
        };
        readBatch();
        return;
      }
      resolve();
    });

  for (const item of items) {
    if (item.kind !== 'file') continue;
    const entry = item.webkitGetAsEntry?.();
    if (entry) {
      await walkEntry(entry);
    } else {
      const f = item.getAsFile?.();
      if (f) droppedFiles.push(f);
    }
  }

  return droppedFiles.length > 0
    ? droppedFiles
    : Array.from(dataTransfer?.files || []);
}

export default function App() {
  // Startup model verification & transient UI state
  const [startupChecking, setStartupChecking] = useState(true);
  const [startupError, setStartupError] = useState(null);
  const [isWorkspaceUnlocked, setIsWorkspaceUnlocked] = useState(false);
  const [isDownloadModalOpen, setIsDownloadModalOpen] = useState(false);
  const [isPipelineRunning, setIsPipelineRunning] = useState(false);
  const [busyLayerAction, setBusyLayerAction] = useState(null); // { blockId, stage: 'ocr' | 'translate' } | null
  const [isPollingLlm, setIsPollingLlm] = useState(false);
  const [statusBanner, setStatusBanner] = useState(null);

  const canvasRef = useRef(null);

  // Core studio state & actions from Zustand store
  const {
    pages,
    activePageId,
    selectedPageIds,
    selectedBlockId,
    zenMode,
    activeModal,
    pipelineConfig,
    viewLayers,
    modelsStatus,
    addPages,
    removePage,
    updatePage,
    setPageStatus,
    setActivePage,
    toggleSelectPage,
    selectPageRange,
    setSelectedBlock,
    updateBlock,
    addBlock,
    deleteBlock,
    applyStyleToAll,
    replaceProjectState,
    setSourceLang,
    hydratePipelineConfig,
    setSupportedTargetLanguages,
    setModelsStatus,
    setLmStudioStatus,
    setHealthStatus,
    openModal,
    closeModal,
  } = useEditorStore(
    useShallow((s) => ({
      pages: s.pages,
      activePageId: s.activePageId,
      selectedPageIds: s.selectedPageIds,
      selectedBlockId: s.selectedBlockId,
      zenMode: s.zenMode,
      activeModal: s.activeModal,
      pipelineConfig: s.pipelineConfig,
      viewLayers: s.viewLayers,
      modelsStatus: s.runtimeStatus.modelsStatus,
      addPages: s.addPages,
      removePage: s.removePage,
      updatePage: s.updatePage,
      setPageStatus: s.setPageStatus,
      setActivePage: s.setActivePage,
      toggleSelectPage: s.toggleSelectPage,
      selectPageRange: s.selectPageRange,
      setSelectedBlock: s.setSelectedBlock,
      updateBlock: s.updateBlock,
      addBlock: s.addBlock,
      deleteBlock: s.deleteBlock,
      applyStyleToAll: s.applyStyleToAll,
      replaceProjectState: s.replaceProjectState,
      setSourceLang: s.setSourceLang,
      hydratePipelineConfig: s.hydratePipelineConfig,
      setSupportedTargetLanguages: s.setSupportedTargetLanguages,
      setModelsStatus: s.setModelsStatus,
      setLmStudioStatus: s.setLmStudioStatus,
      setHealthStatus: s.setHealthStatus,
      openModal: s.openModal,
      closeModal: s.closeModal,
    }))
  );

  const { apiBaseUrl, sourceLang, targetLang } = pipelineConfig;

  // Bind Dark / Light / System OS media query synchronization
  useEffect(() => bindSystemThemeListener(), []);

  // Poll live LM Studio telemetry into store
  const fetchLlmAndHealthStatus = useCallback(async () => {
    setIsPollingLlm(true);
    try {
      const response = await fetch(`${apiBaseUrl}/api/lmstudio/status`, {
        method: 'GET',
        headers: { Accept: 'application/json' },
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      setLmStudioStatus(data);
    } catch (err) {
      setLmStudioStatus({
        connected: false,
        endpoint: pipelineConfig.baseUrl || 'http://localhost:1235/v1',
        active_model: null,
        error: err?.message || 'LM Studio unreachable',
      });
    } finally {
      setIsPollingLlm(false);
    }
  }, [apiBaseUrl, pipelineConfig.baseUrl, setLmStudioStatus]);

  // Hydrate persistent backend settings, languages & health metadata (non-blocking)
  const hydrateBackendContext = useCallback(async () => {
    try {
      const [settingsRes, healthRes, langsRes] = await Promise.allSettled([
        fetch(`${apiBaseUrl}/api/settings`, {
          headers: { Accept: 'application/json' },
        }),
        fetch(`${apiBaseUrl}/api/health`, {
          headers: { Accept: 'application/json' },
        }),
        fetch(`${apiBaseUrl}/api/languages`, {
          headers: { Accept: 'application/json' },
        }),
      ]);

      if (settingsRes.status === 'fulfilled' && settingsRes.value.ok) {
        const settingsData = await settingsRes.value.json();
        hydratePipelineConfig(settingsData);
      }
      if (healthRes.status === 'fulfilled' && healthRes.value.ok) {
        const healthData = await healthRes.value.json();
        setHealthStatus(healthData);
      }
      if (langsRes.status === 'fulfilled' && langsRes.value.ok) {
        const langsData = await langsRes.value.json();
        if (Array.isArray(langsData?.target_languages)) {
          setSupportedTargetLanguages(langsData.target_languages);
        }
      }
    } catch {
      // Non-fatal; fallback defaults in store remain active
    }
  }, [
    apiBaseUrl,
    hydratePipelineConfig,
    setHealthStatus,
    setSupportedTargetLanguages,
  ]);

  // 1. Startup Check: Query GET /api/models/status (with retry while FastAPI backend initializes)
  const checkModelsStatus = useCallback(async () => {
    setStartupChecking(true);
    setStartupError(null);

    const maxAttempts = 12;
    let lastErr = null;

    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      try {
        const response = await fetch(`${apiBaseUrl}/api/models/status`, {
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
        hydrateBackendContext();
        return;
      } catch (err) {
        lastErr = err;
        if (attempt < maxAttempts) {
          await new Promise((r) => setTimeout(r, 1200));
        }
      }
    }

    const displayOrigin =
      apiBaseUrl || (typeof window !== 'undefined' ? window.location.origin : '');
    setStartupError(
      lastErr?.message ||
        `Unable to reach FastAPI backend at ${displayOrigin}/api/models/status`
    );
    setIsWorkspaceUnlocked(false);
    setStartupChecking(false);
  }, [apiBaseUrl, setModelsStatus, hydrateBackendContext]);

  useEffect(() => {
    checkModelsStatus();
  }, [checkModelsStatus]);

  // Periodic LM Studio status polling
  useEffect(() => {
    fetchLlmAndHealthStatus();
    const intervalId = setInterval(fetchLlmAndHealthStatus, 10000);
    return () => clearInterval(intervalId);
  }, [fetchLlmAndHealthStatus]);

  const handleModelsReady = (updatedStatus) => {
    setModelsStatus(updatedStatus);
    setIsWorkspaceUnlocked(true);
    setIsDownloadModalOpen(false);
    hydrateBackendContext();
  };

  // Active page lookup
  const activePage = useMemo(
    () => pages.find((p) => p.id === activePageId) || null,
    [pages, activePageId]
  );

  // Queue-only import for one or multiple image files (Pages picker, Folder picker, or drag-and-drop)
  // Appends raw pages as `status: 'idle'` without triggering any AI processing until Run is clicked.
  const handleImportFiles = useCallback(
    (fileList) => {
      const validFiles = Array.from(fileList || [])
        .filter((f) =>
          f.type
            ? f.type.startsWith('image/')
            : /\.(jpe?g|png|webp|bmp)$/i.test(f.name)
        )
        .sort((a, b) =>
          naturalSortCollator.compare(
            a.webkitRelativePath || a.name,
            b.webkitRelativePath || b.name
          )
        );
      if (validFiles.length === 0) return;

      const timestamp = Date.now();
      const newPages = validFiles.map((file, idx) => ({
        id: `page_${timestamp}_${idx}_${Math.random().toString(36).slice(2, 7)}`,
        name: file.name,
        file,
        originalUrl: URL.createObjectURL(file),
        cleanedImage: null,
        inpaintedUrl: null,
        maskUrl: null,
        maskId: null,
        dimensions: null,
        blocks: [],
        status: 'idle',
        error: null,
        sourceLang,
        targetLang,
        model: null,
        tilesInpainted: 0,
      }));

      addPages(newPages, { activateFirst: true, recordHistory: true });
      setStatusBanner(null);
    },
    [sourceLang, targetLang, addPages]
  );

  // Import a .cbz or .zip comic archive via POST /api/import/archive (queue-only, no auto-run)
  const handleImportArchive = useCallback(
    async (archiveFile) => {
      if (!archiveFile) return;
      setStatusBanner(null);
      try {
        const formData = new FormData();
        formData.append('archive', archiveFile, archiveFile.name);
        const response = await fetch(`${apiBaseUrl}/api/import/archive`, {
          method: 'POST',
          body: formData,
        });
        if (!response.ok) {
          let detail = `HTTP ${response.status}`;
          try {
            const errJson = await response.json();
            if (errJson?.detail) detail = errJson.detail;
          } catch {
            // ignore
          }
          throw new Error(detail);
        }
        const data = await response.json();
        const extracted = data.pages || [];
        if (extracted.length === 0) {
          throw new Error('Archive contained no supported comic page images.');
        }

        const timestamp = Date.now();
        const importedPages = extracted.map((p, idx) => ({
          id: `page_arc_${timestamp}_${idx}_${Math.random().toString(36).slice(2, 6)}`,
          name: p.filename || `page_${String(idx + 1).padStart(3, '0')}.png`,
          file: null,
          originalUrl: resolveBackendUrl(p.image_url, apiBaseUrl),
          cleanedImage: null,
          inpaintedUrl: null,
          maskUrl: null,
          maskId: null,
          dimensions: p.dimensions || null,
          blocks: [],
          status: 'idle',
          error: null,
          sourceLang,
          targetLang,
          model: null,
          tilesInpainted: 0,
        }));

        addPages(importedPages, { activateFirst: true, recordHistory: true });
        setStatusBanner({
          type: 'success',
          message: `Imported ${importedPages.length} pages from ${data.archive_name || archiveFile.name}. Press Run to process.`,
        });
      } catch (err) {
        setStatusBanner({
          type: 'error',
          message: `Archive import failed: ${err?.message || 'Unknown error'}`,
        });
      }
    },
    [apiBaseUrl, sourceLang, targetLang, addPages]
  );

  // Unified queue-only handler for drag-and-drop of images, folders, or .cbz/.zip archives
  const handleImportDroppedFiles = useCallback(
    async (fileList) => {
      const files = Array.from(fileList || []);
      if (files.length === 0) return;

      const archiveFiles = files.filter((f) => /\.(cbz|zip)$/i.test(f.name || ''));
      const imageFiles = files.filter(
        (f) =>
          !/\.(cbz|zip)$/i.test(f.name || '') &&
          (f.type
            ? f.type.startsWith('image/')
            : /\.(jpe?g|png|webp|bmp)$/i.test(f.name || ''))
      );

      if (imageFiles.length > 0) {
        handleImportFiles(imageFiles);
      }
      for (const archive of archiveFiles) {
        await handleImportArchive(archive);
      }
    },
    [handleImportFiles, handleImportArchive]
  );

  const handleWorkspaceDrop = useCallback(
    async (e) => {
      e.preventDefault();
      if (!isWorkspaceUnlocked) return;
      const files = await extractDataTransferFiles(e.dataTransfer);
      if (files.length > 0) {
        handleImportDroppedFiles(files);
      }
    },
    [isWorkspaceUnlocked, handleImportDroppedFiles]
  );

  // Process a full page in one call via POST /api/process-page
  const processSinglePage = useCallback(
    async (pageItem, overrideSourceLang, overrideTargetLang) => {
      const src = overrideSourceLang || pageItem.sourceLang || sourceLang;
      const tgt = overrideTargetLang || pageItem.targetLang || targetLang;

      setPageStatus(pageItem.id, 'processing', {
        error: null,
        sourceLang: src,
        targetLang: tgt,
      });

      try {
        const imageFile = await resolvePageImageFile(pageItem);
        if (!imageFile) {
          throw new Error(`No source image available for ${pageItem.name}`);
        }

        const formData = new FormData();
        formData.append('image', imageFile, pageItem.name || 'page.png');
        formData.append('source_lang', src);
        formData.append('target_lang', tgt);

        const response = await fetch(`${apiBaseUrl}/api/process-page`, {
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
        const rawCleaned =
          data.cleaned_image_url ||
          data.inpainted_image_url ||
          data.cleaned_image ||
          data.inpainted_image;
        const resolvedMaskUrl = resolveBackendUrl(
          data.mask_url || data.mask,
          apiBaseUrl
        );

        const resolvedCleaned = resolveBackendUrl(rawCleaned, apiBaseUrl);

        updatePage(
          pageItem.id,
          {
            status: 'done',
            activeStage: null,
            error: null,
            cleanedImage: resolvedCleaned,
            inpaintedUrl: resolvedCleaned,
            maskId: data.mask_id || pageItem.maskId || null,
            maskUrl: resolvedMaskUrl || pageItem.maskUrl || null,
            dimensions: data.dimensions || pageItem.dimensions || null,
            blocks: normalizedBlocks,
            sourceLang: data.source_lang || src,
            targetLang: data.target_lang || tgt,
            model: data.model || null,
            tilesInpainted: data.tiles_inpainted ?? 0,
          },
          { recordHistory: true }
        );
      } catch (err) {
        const msg = err?.message || 'Failed to process comic page.';
        setPageStatus(pageItem.id, 'error', { activeStage: null, error: msg });
        setStatusBanner({
          type: 'error',
          message: `Error processing ${pageItem.name}: ${msg}`,
        });
      }
    },
    [apiBaseUrl, sourceLang, targetLang, setPageStatus, updatePage]
  );

  // Granular single-stage runner (`detect` | `ocr` | `inpaint` | `translate`)
  const runSingleStageOnPage = useCallback(
    async (pageId, stage, restrictedLayerIds = null) => {
      const currentPage = useEditorStore
        .getState()
        .pages.find((p) => p.id === pageId);
      if (!currentPage) return;

      const cfg = useEditorStore.getState().pipelineConfig;
      const src = cfg.sourceLang || 'ja';
      const tgt = cfg.targetLang || 'id';

      setPageStatus(pageId, 'processing', { activeStage: stage, error: null });

      try {
        if (stage === 'detect') {
          const imageFile = await resolvePageImageFile(currentPage);
          if (!imageFile) throw new Error('Source image missing for detection.');

          const formData = new FormData();
          formData.append('image', imageFile, currentPage.name || 'page.png');
          formData.append('source_lang', src);

          const resp = await fetch(`${apiBaseUrl}/api/stage/detect`, {
            method: 'POST',
            body: formData,
          });
          if (!resp.ok) {
            const errJson = await resp.json().catch(() => ({}));
            throw new Error(errJson?.detail || `HTTP ${resp.status}`);
          }
          const data = await resp.json();
          const detectedBlocks = normalizeBlocks(data.blocks || data.bubbles || []);
          updatePage(
            pageId,
            {
              status: 'done',
              activeStage: null,
              error: null,
              dimensions: data.dimensions || currentPage.dimensions || null,
              maskId: data.mask_id || null,
              maskUrl: resolveBackendUrl(data.mask_url || data.mask, apiBaseUrl),
              blocks: detectedBlocks,
              sourceLang: data.source_lang || src,
            },
            { recordHistory: true }
          );
          return;
        }

        const targetBlocks =
          restrictedLayerIds && restrictedLayerIds.length > 0
            ? (currentPage.blocks || []).filter((b) =>
                restrictedLayerIds.includes(b.id)
              )
            : currentPage.blocks || [];

        if (stage === 'ocr') {
          if (targetBlocks.length === 0) {
            throw new Error('No bounding boxes available for OCR. Run Detection first.');
          }
          const imageFile = await resolvePageImageFile(currentPage);
          if (!imageFile) throw new Error('Source image missing for OCR.');

          const liveOcrBlocks = targetBlocks.map((b) => {
            const hasLiveGeom =
              typeof b.x === 'number' &&
              typeof b.y === 'number' &&
              typeof b.width === 'number' &&
              typeof b.height === 'number';
            const liveBbox = hasLiveGeom
              ? [
                  Math.max(0, Math.round(b.x)),
                  Math.max(0, Math.round(b.y)),
                  Math.max(1, Math.round(b.x + b.width)),
                  Math.max(1, Math.round(b.y + b.height)),
                ]
              : b.bbox;
            return { ...b, bbox: liveBbox };
          });

          const formData = new FormData();
          formData.append('image', imageFile, currentPage.name || 'page.png');
          formData.append('source_lang', src);
          formData.append('bboxes', JSON.stringify(liveOcrBlocks));

          const resp = await fetch(`${apiBaseUrl}/api/stage/ocr`, {
            method: 'POST',
            body: formData,
          });
          if (!resp.ok) {
            const errJson = await resp.json().catch(() => ({}));
            throw new Error(errJson?.detail || `HTTP ${resp.status}`);
          }
          const data = await resp.json();
          const ocrById = Object.fromEntries(
            (data.blocks || data.bubbles || []).map((b) => [b.id, b])
          );
          updatePage(
            pageId,
            (existing) => ({
              status: 'done',
              activeStage: null,
              error: null,
              sourceLang: src,
              blocks: (existing.blocks || []).map((b) => {
                const hit = ocrById[b.id];
                if (!hit) return b;
                const text = hit.original_text ?? hit.source_text ?? b.original_text;
                return { ...b, original_text: text, source_text: text };
              }),
            }),
            { recordHistory: true }
          );
          return;
        }

        if (stage === 'inpaint') {
          const imageFile = await resolvePageImageFile(currentPage);
          if (!imageFile) throw new Error('Source image missing for inpainting.');

          const formData = new FormData();
          formData.append('image', imageFile, currentPage.name || 'page.png');
          if (currentPage.maskId) {
            formData.append('mask_id', currentPage.maskId);
          } else if (currentPage.maskUrl) {
            formData.append('mask_url', currentPage.maskUrl);
          }
          if (targetBlocks.length > 0) {
            formData.append(
              'bboxes',
              JSON.stringify(targetBlocks.map((b) => b.bbox))
            );
          } else if (!currentPage.maskId && !currentPage.maskUrl) {
            throw new Error('No mask or bounding boxes available for inpainting.');
          }

          const resp = await fetch(`${apiBaseUrl}/api/stage/inpaint`, {
            method: 'POST',
            body: formData,
          });
          if (!resp.ok) {
            const errJson = await resp.json().catch(() => ({}));
            throw new Error(errJson?.detail || `HTTP ${resp.status}`);
          }
          const data = await resp.json();
          const cleanedUrl =
            data.cleaned_image_url ||
            data.inpainted_image_url ||
            data.cleaned_image ||
            data.inpainted_image;
          const resolvedCleaned = resolveBackendUrl(cleanedUrl, apiBaseUrl);
          updatePage(
            pageId,
            {
              status: 'done',
              activeStage: null,
              error: null,
              cleanedImage: resolvedCleaned,
              inpaintedUrl: resolvedCleaned,
              dimensions: data.dimensions || currentPage.dimensions || null,
              tilesInpainted: data.tiles_inpainted ?? currentPage.tilesInpainted,
            },
            { recordHistory: true }
          );
          return;
        }

        if (stage === 'translate') {
          if (targetBlocks.length === 0) {
            throw new Error('No text blocks available to translate.');
          }
          const payload = {
            source_lang: src,
            target_lang: tgt,
            blocks: targetBlocks.map((b) => ({
              id: b.id,
              order: b.order,
              bbox: b.bbox,
              original_text: b.original_text || b.source_text || '',
              source_text: b.source_text || b.original_text || '',
              translated_text: b.translated_text || '',
            })),
            config: {
              provider: cfg.provider || undefined,
              model: cfg.model || undefined,
              temperature: cfg.temperature,
              top_p: cfg.topP ?? undefined,
              top_k: cfg.topK ?? undefined,
              max_tokens: cfg.maxTokens,
              frequency_penalty: cfg.frequencyPenalty ?? undefined,
              reasoning_effort: cfg.reasoningEffort || undefined,
              custom_instructions: cfg.customInstructions || undefined,
              glossary: cfg.glossary,
              use_vision: cfg.useVision,
              vision_max_side: cfg.visionMaxSide || undefined,
            },
          };

          const resp = await fetch(`${apiBaseUrl}/api/stage/translate`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              Accept: 'application/json',
            },
            body: JSON.stringify(payload),
          });
          if (!resp.ok) {
            const errJson = await resp.json().catch(() => ({}));
            throw new Error(errJson?.detail || `HTTP ${resp.status}`);
          }
          const data = await resp.json();
          const transMap = data.translations || {};
          const byId = Object.fromEntries(
            (data.blocks || data.bubbles || []).map((b) => [b.id, b])
          );
          updatePage(
            pageId,
            (existing) => ({
              status: 'done',
              activeStage: null,
              error: null,
              targetLang: data.target_lang || tgt,
              model: data.model || existing.model,
              blocks: (existing.blocks || []).map((b) => {
                const updated =
                  transMap[b.id] ?? byId[b.id]?.translated_text;
                if (updated === undefined) return b;
                return { ...b, translated_text: updated };
              }),
            }),
            { recordHistory: true }
          );
        }
      } catch (err) {
        const msg = err?.message || `Stage ${stage} failed.`;
        setPageStatus(pageId, 'error', { activeStage: null, error: msg });
        setStatusBanner({
          type: 'error',
          message: `${stage.toUpperCase()} failed on ${currentPage.name}: ${msg}`,
        });
        throw err;
      }
    },
    [apiBaseUrl, setPageStatus, updatePage]
  );

  // Execute a sequence of stages across one or more target pages
  const runStagesForPages = useCallback(
    async (targetPageIds, stagesToRun, restrictedLayerIds = null) => {
      if (!targetPageIds.length || !stagesToRun.length) return;
      setIsPipelineRunning(true);
      setStatusBanner(null);

      const allFourStages =
        stagesToRun.length === 4 &&
        ['detect', 'ocr', 'inpaint', 'translate'].every((s) =>
          stagesToRun.includes(s)
        );

      try {
        for (const pageId of targetPageIds) {
          const liveState = useEditorStore.getState();
          const pageObj = liveState.pages.find((p) => p.id === pageId);
          if (!pageObj) continue;

          if (allFourStages && !restrictedLayerIds) {
            await processSinglePage(
              pageObj,
              liveState.pipelineConfig.sourceLang,
              liveState.pipelineConfig.targetLang
            );
          } else {
            for (const stage of stagesToRun) {
              await runSingleStageOnPage(pageId, stage, restrictedLayerIds);
            }
          }
        }
        setStatusBanner({
          type: 'success',
          message: `Completed ${stagesToRun.join(' → ')} on ${targetPageIds.length} page${
            targetPageIds.length === 1 ? '' : 's'
          }.`,
        });
      } catch {
        // Error banner already reported by stage handler
      } finally {
        setIsPipelineRunning(false);
      }
    },
    [processSinglePage, runSingleStageOnPage]
  );

  // FrontControlBar primary Run handler
  const handleFrontControlRun = useCallback(() => {
    const state = useEditorStore.getState();
    const { scope, enabledStages } = state.pipelineConfig;
    const stagesToRun = ['detect', 'ocr', 'inpaint', 'translate'].filter((k) =>
      Boolean(enabledStages?.[k])
    );
    if (stagesToRun.length === 0) return;

    let targetIds = [];
    if (scope === 'project') {
      targetIds = state.pages.map((p) => p.id);
    } else if (scope === 'selected_pages') {
      targetIds =
        state.selectedPageIds.length > 0
          ? state.selectedPageIds
          : state.activePageId
          ? [state.activePageId]
          : [];
    } else if (state.activePageId) {
      targetIds = [state.activePageId];
    }

    runStagesForPages(targetIds, stagesToRun);
  }, [runStagesForPages]);

  const handleProcessProject = useCallback(() => {
    const state = useEditorStore.getState();
    const stages = ['detect', 'ocr', 'inpaint', 'translate'].filter((k) =>
      Boolean(state.pipelineConfig.enabledStages?.[k])
    );
    runStagesForPages(
      state.pages.map((p) => p.id),
      stages.length > 0 ? stages : ['detect', 'ocr', 'inpaint', 'translate']
    );
  }, [runStagesForPages]);

  const handleProcessSelectedPages = useCallback(() => {
    const state = useEditorStore.getState();
    const targetIds =
      state.selectedPageIds.length > 0
        ? state.selectedPageIds
        : state.activePageId
        ? [state.activePageId]
        : [];
    const stages = ['detect', 'ocr', 'inpaint', 'translate'].filter((k) =>
      Boolean(state.pipelineConfig.enabledStages?.[k])
    );
    runStagesForPages(
      targetIds,
      stages.length > 0 ? stages : ['detect', 'ocr', 'inpaint', 'translate']
    );
  }, [runStagesForPages]);

  const handleProcessSelectedLayers = useCallback(() => {
    const state = useEditorStore.getState();
    if (!state.activePageId) return;
    const layerIds =
      state.selectedLayerIds.length > 0
        ? state.selectedLayerIds
        : state.selectedBlockId
        ? [state.selectedBlockId]
        : [];
    if (layerIds.length === 0) return;
    const stages = ['ocr', 'inpaint', 'translate'].filter((k) =>
      Boolean(state.pipelineConfig.enabledStages?.[k])
    );
    runStagesForPages(
      [state.activePageId],
      stages.length > 0 ? stages : ['ocr', 'inpaint', 'translate'],
      layerIds
    );
  }, [runStagesForPages]);

  const handleRunSingleStage = useCallback(
    (stage) => {
      if (!activePageId) return;
      runStagesForPages([activePageId], [stage]);
    },
    [activePageId, runStagesForPages]
  );

  // Quick per-layer Re-OCR (/api/stage/ocr) & Re-Translate (/api/stage/translate) actions for TextSidebar
  const handleRunLayerAction = useCallback(
    async (blockId, stage) => {
      if (!activePageId || !blockId) return;
      setBusyLayerAction({ blockId, stage });
      setStatusBanner(null);
      try {
        await runSingleStageOnPage(activePageId, stage, [blockId]);
        setStatusBanner({
          type: 'success',
          message:
            stage === 'ocr'
              ? `Re-OCR completed for layer ${blockId}.`
              : `Re-translated layer ${blockId} successfully.`,
        });
      } catch {
        // Error banner already reported by runSingleStageOnPage
      } finally {
        setBusyLayerAction(null);
      }
    },
    [activePageId, runSingleStageOnPage]
  );

  const handleReOcrLayer = useCallback(
    (blockId) => handleRunLayerAction(blockId, 'ocr'),
    [handleRunLayerAction]
  );

  const handleReTranslateLayer = useCallback(
    (blockId) => handleRunLayerAction(blockId, 'translate'),
    [handleRunLayerAction]
  );

  // Apply typography style settings to all blocks on the active page
  const handleApplyStyleToAll = useCallback(
    (styleUpdates) => {
      if (!activePageId) return;
      applyStyleToAll(styleUpdates);
      setStatusBanner({
        type: 'success',
        message: 'Applied typography and stroke settings to all layers on this page.',
      });
    },
    [activePageId, applyStyleToAll]
  );

  // Export current Konva canvas as a full-resolution .png or .jpg image
  const handleExportImage = useCallback(
    async (format = 'png') => {
      if (!canvasRef.current || !activePage) return;
      await ensurePageFontsLoaded(activePage, null, apiBaseUrl);
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
        message: `Exported ${filename} (${activePage.dimensions?.width || ''}×${activePage.dimensions?.height || ''}px).`,
      });
    },
    [activePage, apiBaseUrl, targetLang]
  );

  // Export multi-page project as 1:1 native-resolution .cbz or .pdf via FastAPI exporter endpoints
  const handleExportMultiPage = useCallback(
    async (format) => {
      if (pages.length === 0) return;
      setStatusBanner(null);
      try {
        const exportStem = `mangaflow_${targetLang}`;
        let result = null;

        if (canvasRef.current?.exportArchive) {
          result = await canvasRef.current.exportArchive(pages, format, {
            filename: exportStem,
            imageFormat: 'png',
            quality: 0.98,
          });
        } else {
          const formData = new FormData();
          formData.append('filename', exportStem);
          for (let idx = 0; idx < pages.length; idx += 1) {
            const p = pages[idx];
            let blob = null;
            if (canvasRef.current?.exportPageBlob) {
              blob = await canvasRef.current.exportPageBlob(p, 'png', 0.98);
            }
            if (!blob) {
              const imgSrc = p.cleanedImage || p.originalUrl;
              if (imgSrc) {
                blob = await (await fetch(imgSrc)).blob();
              } else if (p.file) {
                blob = p.file;
              }
            }
            if (blob) {
              const pageFilename = `${String(idx + 1).padStart(3, '0')}_${p.name.replace(
                /\.[^/.]+$/,
                ''
              )}.png`;
              formData.append('images', blob, pageFilename);
            }
          }
          const resp = await fetch(`${apiBaseUrl}/api/export/${format}`, {
            method: 'POST',
            body: formData,
          });
          if (!resp.ok) {
            const errJson = await resp.json().catch(() => ({}));
            throw new Error(errJson?.detail || `HTTP ${resp.status}`);
          }
          result = {
            blob: await resp.blob(),
            filename: `${exportStem}.${format}`,
            pageCount: pages.length,
          };
        }

        triggerBlobDownload(result.blob, result.filename);
        setStatusBanner({
          type: 'success',
          message: `Exported ${result.pageCount} page${
            result.pageCount === 1 ? '' : 's'
          } at 1:1 native resolution to ${result.filename}.`,
        });
      } catch (err) {
        setStatusBanner({
          type: 'error',
          message: `Export .${format} failed: ${err?.message || 'Unknown error'}`,
        });
      }
    },
    [pages, apiBaseUrl, targetLang]
  );

  // Save Project (.mflow.json)
  const handleSaveProject = useCallback(() => {
    if (pages.length === 0) return;
    const { apiKey: _redactedKey, ...safePipelineConfig } = pipelineConfig;
    const projectPayload = {
      format: 'mangaflow-project',
      version: 2,
      savedAt: new Date().toISOString(),
      activePageId,
      viewLayers,
      pipelineConfig: safePipelineConfig,
      pages: pages.map((p) => ({
        id: p.id,
        name: p.name,
        originalUrl:
          p.originalUrl && !p.originalUrl.startsWith('blob:')
            ? p.originalUrl
            : null,
        cleanedImage: p.cleanedImage || p.inpaintedUrl || null,
        inpaintedUrl: p.inpaintedUrl || p.cleanedImage || null,
        maskUrl: p.maskUrl || null,
        maskId: p.maskId || null,
        dimensions: p.dimensions || null,
        blocks: p.blocks || [],
        status:
          p.status === 'processing' || p.status === 'queued'
            ? 'idle'
            : p.status,
        sourceLang: p.sourceLang,
        targetLang: p.targetLang,
        model: p.model,
        tilesInpainted: p.tilesInpainted ?? 0,
      })),
    };

    const blob = new Blob([JSON.stringify(projectPayload, null, 2)], {
      type: 'application/json',
    });
    triggerBlobDownload(blob, `mangaflow-project.mflow.json`);
    setStatusBanner({
      type: 'success',
      message: `Saved project (${pages.length} page${pages.length === 1 ? '' : 's'}) to mangaflow-project.mflow.json.`,
    });
  }, [pages, activePageId, viewLayers, pipelineConfig]);

  // Open Project (.mflow.json)
  const handleOpenProject = useCallback(
    async (file) => {
      if (!file) return;
      try {
        const text = await file.text();
        const parsed = JSON.parse(text);
        if (!parsed || !Array.isArray(parsed.pages)) {
          throw new Error('Invalid .mflow.json project file.');
        }
        replaceProjectState({
          pages: parsed.pages,
          activePageId: parsed.activePageId || null,
          pipelineConfig: parsed.pipelineConfig || null,
          viewLayers: parsed.viewLayers || null,
        });
        setStatusBanner({
          type: 'success',
          message: `Opened project ${file.name} (${parsed.pages.length} pages).`,
        });
      } catch (err) {
        setStatusBanner({
          type: 'error',
          message: `Failed to open project: ${err?.message || 'Invalid JSON'}`,
        });
      }
    },
    [replaceProjectState]
  );

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
        const file = new File([blob], filename, {
          type: blob.type || 'image/jpeg',
        });

        const timestamp = Date.now();
        const newPage = {
          id: `page_${timestamp}_0_${Math.random().toString(36).slice(2, 7)}`,
          name: filename,
          file,
          originalUrl: URL.createObjectURL(file),
          cleanedImage: null,
          inpaintedUrl: null,
          maskUrl: null,
          maskId: null,
          dimensions: null,
          blocks: [],
          status: 'idle',
          error: null,
          sourceLang: sampleLang,
          targetLang,
          model: null,
          tilesInpainted: 0,
        };

        addPages([newPage], { activateFirst: true, recordHistory: true });
        setStatusBanner(null);
      } catch (err) {
        setStatusBanner({
          type: 'error',
          message: err?.message || `Could not load sample ${filename}`,
        });
      }
    },
    [targetLang, setSourceLang, addPages]
  );

  const handleFitScreen = useCallback(() => {
    if (canvasRef.current?.fitToScreen) {
      canvasRef.current.fitToScreen();
    }
  }, []);

  const handleOpenSettings = useCallback(() => {
    openModal('settings');
  }, [openModal]);

  const handlePageClick = (e, pageId) => {
    if (e.shiftKey) {
      selectPageRange(pageId);
    } else if (e.ctrlKey || e.metaKey) {
      toggleSelectPage(pageId);
    } else {
      setActivePage(pageId);
    }
  };

  const isAnyProcessing =
    isPipelineRunning || pages.some((p) => p.status === 'processing');

  return (
    <div className="min-h-screen h-screen w-screen flex flex-col bg-studio-canvas text-studio-text overflow-hidden select-none">
      {/* Bar 1: Compact 28px Native Application Menu Bar */}
      <MenuBar
        onImportFiles={handleImportFiles}
        onImportArchive={handleImportArchive}
        onExportImage={handleExportImage}
        onExportCbz={() => handleExportMultiPage('cbz')}
        onExportPdf={() => handleExportMultiPage('pdf')}
        onSaveProject={handleSaveProject}
        onOpenProject={handleOpenProject}
        onProcessProject={handleProcessProject}
        onProcessSelectedPages={handleProcessSelectedPages}
        onProcessSelectedLayers={handleProcessSelectedLayers}
        onRunStage={handleRunSingleStage}
        onFitScreen={handleFitScreen}
        onOpenModelsModal={() => setIsDownloadModalOpen(true)}
        isProcessing={isAnyProcessing}
        isWorkspaceUnlocked={isWorkspaceUnlocked}
      />

      {/* Bar 2: Compact 36px Pipeline Front Control Bar */}
      <FrontControlBar
        onRun={handleFrontControlRun}
        onOpenSettings={handleOpenSettings}
        isProcessing={isAnyProcessing}
        isWorkspaceUnlocked={isWorkspaceUnlocked}
      />

      {/* Compact Notification Strip */}
      {statusBanner && (
        <div
          className={`px-3 py-1 text-[11px] flex items-center justify-between border-b border-studio-border ${
            statusBanner.type === 'error'
              ? 'bg-studio-danger/10 text-studio-danger'
              : 'bg-studio-success/10 text-studio-success'
          }`}
        >
          <div className="flex items-center gap-2 truncate">
            {statusBanner.type === 'error' ? (
              <AlertCircle className="h-3.5 w-3.5 shrink-0" />
            ) : (
              <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
            )}
            <span className="truncate font-medium">{statusBanner.message}</span>
          </div>
          <button
            type="button"
            onClick={() => setStatusBanner(null)}
            className="ml-4 text-[11px] text-studio-text-secondary hover:text-studio-text cursor-pointer"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Main Three-Pane Desktop Studio Workspace */}
      <main
        className="relative flex-1 min-h-0 flex overflow-hidden"
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleWorkspaceDrop}
      >
        {/* Left Sidebar: Multi-Page Filmstrip / Queue (collapsed in Zen Mode) */}
        {!zenMode && (
          <aside className="w-56 shrink-0 border-r border-studio-border bg-studio-panel flex flex-col overflow-hidden">
            <div className="h-9 px-3 border-b border-studio-border flex items-center justify-between shrink-0">
              <div className="flex items-center gap-1.5">
                <Layers className="h-3.5 w-3.5 text-studio-accent" />
                <span className="text-[11px] font-semibold uppercase tracking-wider text-studio-text">
                  Pages
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                {selectedPageIds.length > 1 && (
                  <span className="rounded-studio-sm bg-studio-accent/15 border border-studio-accent/40 px-1.5 py-0.5 font-mono text-[10px] leading-none text-studio-accent tabular-nums">
                    {selectedPageIds.length} sel
                  </span>
                )}
                <span className="rounded-studio-sm border border-studio-border bg-studio-surface px-1.5 py-0.5 font-mono text-[10px] leading-none text-studio-text-secondary tabular-nums">
                  {pages.length}
                </span>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
              {pages.length === 0 ? (
                <div className="rounded-studio border border-dashed border-studio-border p-4 text-center space-y-1.5 my-1">
                  <FileImage className="h-5 w-5 text-studio-text-muted mx-auto" />
                  <p className="text-xs font-medium text-studio-text-secondary">
                    No Pages Loaded
                  </p>
                  <p className="text-[11px] text-studio-text-muted leading-snug">
                    Use File → Import Pages (Ctrl+O) or drop images onto the canvas.
                  </p>
                </div>
              ) : (
                pages.map((p, idx) => {
                  const isActive = p.id === activePageId;
                  const isSelected = selectedPageIds.includes(p.id);
                  return (
                    <div
                      key={p.id}
                      onClick={(e) => handlePageClick(e, p.id)}
                      className={`group relative rounded-studio border p-2 transition-colors cursor-pointer flex items-center gap-2 ${
                        isActive
                          ? 'border-studio-accent bg-studio-accent/10'
                          : isSelected
                          ? 'border-studio-accent/50 bg-studio-surface'
                          : 'border-studio-border bg-studio-surface hover:border-studio-border-strong'
                      }`}
                    >
                      {isActive && (
                        <span
                          className="absolute left-0 top-1 bottom-1 w-0.5 bg-studio-accent"
                          aria-hidden="true"
                        />
                      )}

                      <div className="h-11 w-8 shrink-0 rounded-studio-sm bg-studio-canvas border border-studio-border overflow-hidden flex items-center justify-center">
                        {p.cleanedImage || p.originalUrl ? (
                          <img
                            src={p.cleanedImage || p.originalUrl}
                            alt={p.name}
                            className="h-full w-full object-cover"
                          />
                        ) : (
                          <FileImage className="h-3.5 w-3.5 text-studio-text-muted" />
                        )}
                      </div>

                      <div className="min-w-0 flex-1">
                        <div className="flex items-center justify-between gap-1">
                          <span className="text-xs font-medium text-studio-text truncate">
                            <span className="font-mono text-[10px] text-studio-text-muted mr-1 tabular-nums">
                              {String(idx + 1).padStart(2, '0')}
                            </span>
                            {p.name}
                          </span>
                        </div>

                        <div className="mt-1 flex items-center gap-1.5 text-[10px] font-mono">
                          {p.status === 'done' && (
                            <span className="inline-flex items-center gap-1 text-studio-success">
                              <CheckCircle2 className="h-3 w-3" />
                              {p.blocks?.length || 0} blocks
                            </span>
                          )}
                          {p.status === 'processing' && (
                            <span className="inline-flex items-center gap-1 text-studio-accent">
                              <Loader2 className="h-3 w-3 animate-spin" />
                              {p.activeStage ? `${p.activeStage}…` : 'Running…'}
                            </span>
                          )}
                          {(p.status === 'queued' || p.status === 'idle') && (
                            <span className="inline-flex items-center gap-1 text-studio-text-muted">
                              <Clock className="h-3 w-3" />
                              {p.status === 'queued' ? 'Queued' : 'Ready'}
                            </span>
                          )}
                          {p.status === 'error' && (
                            <span
                              className="inline-flex items-center gap-1 text-studio-danger truncate"
                              title={p.error}
                            >
                              <AlertCircle className="h-3 w-3 shrink-0" />
                              Failed
                            </span>
                          )}
                        </div>
                      </div>

                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          removePage(p.id);
                        }}
                        className="opacity-0 group-hover:opacity-100 p-1 rounded-studio-sm hover:bg-studio-hover text-studio-text-muted hover:text-studio-danger transition-colors cursor-pointer"
                        title="Remove page"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  );
                })
              )}
            </div>
          </aside>
        )}

        {/* Center Interactive 60fps Konva Canvas */}
        <div className="relative flex-1 min-w-0 h-full flex overflow-hidden">
          <MangaCanvas
            ref={canvasRef}
            page={activePage}
            selectedBlockId={selectedBlockId}
            onSelectBlock={setSelectedBlock}
            onUpdateBlock={updateBlock}
            onImportDrop={handleImportDroppedFiles}
            onLoadSample={handleLoadSample}
            isProcessing={activePage?.status === 'processing'}
          />
        </div>

        {/* Right Sidebar: Text Layers & Typography Inspector (collapsed in Zen Mode) */}
        {!zenMode && (
          <TextSidebar
            page={activePage}
            selectedBlockId={selectedBlockId}
            onSelectBlock={setSelectedBlock}
            onUpdateBlock={updateBlock}
            onApplyStyleToAll={handleApplyStyleToAll}
            onAddBlock={() => addBlock()}
            onDeleteBlock={deleteBlock}
            onReOcrLayer={handleReOcrLayer}
            onReTranslateLayer={handleReTranslateLayer}
            busyLayerAction={busyLayerAction}
            isProcessing={isAnyProcessing}
          />
        )}

        {/* Startup Check / Backend Error Overlay */}
        {(startupChecking || startupError) && !isDownloadModalOpen && (
          <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/65 p-4">
            <div className="w-full max-w-md rounded-studio-md border border-studio-border bg-studio-panel p-5 text-center space-y-3.5">
              {startupChecking ? (
                <>
                  <Loader2 className="h-7 w-7 animate-spin text-studio-accent mx-auto" />
                  <div className="space-y-1">
                    <h3 className="text-sm font-semibold text-studio-text">
                      Verifying Local AI Models…
                    </h3>
                    <p className="text-xs text-studio-text-muted font-mono">
                      GET {apiBaseUrl}/api/models/status
                    </p>
                  </div>
                </>
              ) : (
                <>
                  <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-studio border border-studio-danger/40 bg-studio-danger/10 text-studio-danger">
                    <AlertTriangle className="h-5 w-5" />
                  </div>
                  <div className="space-y-1">
                    <h3 className="text-sm font-semibold text-studio-text">
                      Backend Connection Failed
                    </h3>
                    <p className="text-xs text-studio-text-secondary">{startupError}</p>
                  </div>
                  <button
                    type="button"
                    onClick={checkModelsStatus}
                    className="studio-btn-accent h-8 px-3.5 text-xs"
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

      {/* Bottom 24px Native Studio Status Bar */}
      <StatusBar
        onResetZoom={activePage ? handleFitScreen : undefined}
        onRefreshLlmStatus={fetchLlmAndHealthStatus}
        isPollingLlm={isPollingLlm}
      />

      {/* Model Download & SSE Verification Modal */}
      <DownloadModal
        isOpen={isDownloadModalOpen}
        initialStatus={modelsStatus}
        apiBaseUrl={apiBaseUrl}
        onModelsReady={handleModelsReady}
        onClose={
          modelsStatus?.all_ready ? () => setIsDownloadModalOpen(false) : undefined
        }
      />

      {/* Koharu-style 5-Tab Studio Settings Modal */}
      <SettingsModal
        isOpen={activeModal === 'settings'}
        onClose={closeModal}
        onSaved={() => {
          hydrateBackendContext();
          fetchLlmAndHealthStatus();
        }}
        onApplyTypesettingToPage={handleApplyStyleToAll}
      />
    </div>
  );
}
