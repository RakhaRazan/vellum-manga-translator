import React, { Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  Clock,
  FileImage,
  Layers,
  Loader2,
  RefreshCw,
  Search,
  Trash2,
  X,
} from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import QuickProviderSetupModal from './components/QuickProviderSetupModal.jsx';
import MenuBar from './components/MenuBar.jsx';
import FrontControlBar from './components/FrontControlBar.jsx';
import MangaCanvas from './components/MangaCanvas.jsx';
import TextSidebar from './components/TextSidebar.jsx';
import StatusBar from './components/StatusBar.jsx';

const DownloadModal = React.lazy(() => import('./components/DownloadModal.jsx'));
const SettingsModal = React.lazy(() => import('./components/SettingsModal.jsx'));
const BatchQueueModal = React.lazy(() => import('./components/BatchQueueModal.jsx'));
import { applySeriesProfileToStore } from './components/SeriesProfilePicker.jsx';
import useEditorStore, {
  bindSystemThemeListener,
  ensurePageFontsLoaded,
  getPageQcFlags,
  mergeRedetectedBlocks,
  normalizeBlocks,
} from './store/useEditorStore.js';

import {
  resolveBackendUrl,
  naturalSortCollator,
  loadPersistedQueuePrefs,
  loadPersistedQueueItems,
  extractDataTransferFiles,
  AUTOSAVE_STORAGE_KEY,
  BATCH_QUEUE_STORAGE_KEY,
  BATCH_QUEUE_PREFS_KEY,
} from './utils/pipelineApi.js';
import { useBatchProcessor } from './hooks/useBatchProcessor.js';

class ModalErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, message: '' };
  }

  static getDerivedStateFromError(error) {
    return {
      hasError: true,
      message: error?.message || 'Unexpected UI error in modal.',
    };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div className="w-full max-w-md rounded-studio-md border border-studio-danger/40 bg-studio-panel p-4 shadow-2xl space-y-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-studio-danger">
              {this.props.title || 'Modal Render Error'}
            </h3>
            <p className="text-xs text-studio-text-secondary break-words font-mono">
              {this.state.message}
            </p>
            <div className="flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={() => {
                  this.setState({ hasError: false, message: '' });
                  this.props.onClose?.();
                }}
                className="studio-btn h-7 px-3 text-xs cursor-pointer"
              >
                Close
              </button>
              <button
                type="button"
                onClick={() => window.location.reload()}
                className="studio-btn-accent h-7 px-3 text-xs cursor-pointer"
              >
                Reload
              </button>
            </div>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  // Startup model verification & transient UI state
  const [startupChecking, setStartupChecking] = useState(true);
  const [startupError, setStartupError] = useState(null);
  const [isWorkspaceUnlocked, setIsWorkspaceUnlocked] = useState(false);
  const [isDownloadModalOpen, setIsDownloadModalOpen] = useState(false);
  const [quickProviderPrompt, setQuickProviderPrompt] = useState(null); // { onProceedTranslate, onProceedManual } | null
  const [isPipelineRunning, setIsPipelineRunning] = useState(false);
  const [pipelineProgress, setPipelineProgress] = useState(null);
  const [liveBackendStage, setLiveBackendStage] = useState(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [importBaseName, setImportBaseName] = useState('');
  const [busyLayerAction, setBusyLayerAction] = useState(null); // { blockId, stage: 'ocr' | 'translate' } | null
  const [isPollingLlm, setIsPollingLlm] = useState(false);
  const [statusBanner, setStatusBanner] = useState(null);
  const [skipCompleted, setSkipCompleted] = useState(true);
  const [autosaveSnapshot, setAutosaveSnapshot] = useState(null);
  const [isStopping, setIsStopping] = useState(false);

  const batchAbortControllerRef = useRef(null);
  const batchCancelRef = useRef(false);
  const wakePipelineWaitersRef = useRef(null);
  const projectCacheIdRef = useRef(`proj_${Date.now()}`);
  const canvasRef = useRef(null);
  const filmstripRef = useRef(null);
  const [filmstripScrollTop, setFilmstripScrollTop] = useState(0);
  const [filmstripViewportHeight, setFilmstripViewportHeight] = useState(640);
  const [pageFilter, setPageFilter] = useState('all'); // 'all' | 'issues'
  const [isFindReplaceOpen, setIsFindReplaceOpen] = useState(false);
  const [findText, setFindText] = useState('');
  const [replaceText, setReplaceText] = useState('');
  const [findCaseSensitive, setFindCaseSensitive] = useState(false);
  const [findWholeWord, setFindWholeWord] = useState(false);
  const [findReplaceStatus, setFindReplaceStatus] = useState('');
  const [activeProfileSlug, setActiveProfileSlug] = useState('');
  const [profilesVersion, setProfilesVersion] = useState(0);
  const [isBatchQueueOpen, setIsBatchQueueOpen] = useState(false);
  const [queueItems, setQueueItems] = useState(() => loadPersistedQueueItems());
  const [queueRunMode, setQueueRunMode] = useState(
    () => loadPersistedQueuePrefs().runMode || 'review'
  ); // 'review' | 'auto_export'
  const [queueExportFormat, setQueueExportFormat] = useState(
    () => loadPersistedQueuePrefs().exportFormat || 'cbz'
  ); // 'cbz' | 'pdf' | 'both'
  const [queueOutputDir, setQueueOutputDir] = useState(
    () => loadPersistedQueuePrefs().outputDir || ''
  );
  const [queueFilenamePattern, setQueueFilenamePattern] = useState(
    () => loadPersistedQueuePrefs().namePattern || '{name}'
  );

  useEffect(() => {
    try {
      localStorage.setItem(
        BATCH_QUEUE_PREFS_KEY,
        JSON.stringify({
          outputDir: queueOutputDir,
          namePattern: queueFilenamePattern,
          exportFormat: queueExportFormat,
          runMode: queueRunMode,
        })
      );
    } catch {
      // ignore localStorage errors
    }
  }, [queueOutputDir, queueFilenamePattern, queueExportFormat, queueRunMode]);
  const [queueSkipExisting, setQueueSkipExisting] = useState(true);
  const [queuePurgeCache, setQueuePurgeCache] = useState(true);
  const [queueShareContinuity, setQueueShareContinuity] = useState(true);
  const [isQueueRunning, setIsQueueRunning] = useState(false);
  const queueCancelRef = useRef(false);
  const activeQueueItemIdRef = useRef(null);
  const activeQueueItemMetaRef = useRef(null);
  const reimportedQueueItemIdsRef = useRef(new Set());

  useEffect(() => {
    try {
      const compact = queueItems.map((q) => ({
        id: q.id,
        projectId: q.projectId,
        kind: q.kind,
        name: q.name,
        stem: q.stem,
        path: q.path,
        profileSlug: q.profileSlug || '',
        status: q.status || 'queued',
        progressText: q.progressText || 'Queued',
        pageCount: q.pageCount ?? null,
        exportStatus: q.exportStatus || null,
        selected: q.selected !== false,
        error: q.error || null,
      }));
      localStorage.setItem(BATCH_QUEUE_STORAGE_KEY, JSON.stringify(compact));
    } catch {
      // ignore localStorage quota errors
    }
  }, [queueItems]);

  // Core studio state & actions from Zustand store
  const {
    pages,
    activePageId,
    selectedPageIds,
    selectedBlockId,
    zenMode,
    uiDensity,
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
    replaceAllTranslations,
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
      uiDensity: s.uiDensity || 'compact',
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
      replaceAllTranslations: s.replaceAllTranslations,
      setSourceLang: s.setSourceLang,
      hydratePipelineConfig: s.hydratePipelineConfig,
      setSupportedTargetLanguages: s.setSupportedTargetLanguages,
      setModelsStatus: s.setModelsStatus,
      setLmStudioStatus: s.setLmStudioStatus,
      setHealthStatus: s.setHealthStatus,
      setHardwareInfo: s.setHardwareInfo,
      openModal: s.openModal,
      closeModal: s.closeModal,
    }))
  );

  const { apiBaseUrl, sourceLang, targetLang } = pipelineConfig;

  // Bind Dark / Light / System OS media query synchronization
  useEffect(() => bindSystemThemeListener(), []);

  // Synchronize UI Density (`compact` vs `comfortable`) to root dataset attribute
  useEffect(() => {
    document.documentElement.dataset.density = uiDensity || 'compact';
  }, [uiDensity]);

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
        endpoint: pipelineConfig.baseUrl || '',
        active_model: null,
        error: err?.message || 'LM Studio unreachable',
      });
    } finally {
      setIsPollingLlm(false);
    }
  }, [apiBaseUrl, pipelineConfig.baseUrl, setLmStudioStatus]);

  // Hydrate persistent backend settings, languages, fonts & hardware capabilities (non-blocking)
  const hydrateBackendContext = useCallback(async () => {
    try {
      const [settingsRes, healthRes, langsRes, fontsRes, hwRes] = await Promise.allSettled([
        fetch(`${apiBaseUrl}/api/settings`, {
          headers: { Accept: 'application/json' },
        }),
        fetch(`${apiBaseUrl}/api/health`, {
          headers: { Accept: 'application/json' },
        }),
        fetch(`${apiBaseUrl}/api/languages`, {
          headers: { Accept: 'application/json' },
        }),
        fetch(`${apiBaseUrl}/api/fonts/list`, {
          headers: { Accept: 'application/json' },
        }),
        fetch(`${apiBaseUrl}/api/system/hardware`, {
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
      if (hwRes.status === 'fulfilled' && hwRes.value.ok) {
        const hwData = await hwRes.value.json();
        setHardwareInfo(hwData);
      }
      if (langsRes.status === 'fulfilled' && langsRes.value.ok) {
        const langsData = await langsRes.value.json();
        if (Array.isArray(langsData?.target_languages)) {
          setSupportedTargetLanguages(langsData.target_languages);
        }
      }
      if (fontsRes.status === 'fulfilled' && fontsRes.value.ok) {
        const fontsData = await fontsRes.value.json();
        useEditorStore.getState().setFontsCatalogCache(fontsData);
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

  // 1. Startup Check: Query GET /api/models/status (with retry while native C++17 backend initializes)
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
        `Unable to reach native C++17 backend at ${displayOrigin}/api/models/status`
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
  const autoMatchSeriesProfile = useCallback(
    async (title) => {
      const cleanTitle = String(title || '').trim();
      if (!cleanTitle) return;
      try {
        const resp = await fetch(`${apiBaseUrl}/api/series-profiles/match`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: cleanTitle }),
        });
        if (!resp.ok) return;
        const data = await resp.json();
        if (data?.matched && data?.profile?.slug) {
          setActiveProfileSlug(data.profile.slug);
          applySeriesProfileToStore(data.profile, apiBaseUrl, { syncBackend: true });
          setStatusBanner({
            type: 'success',
            message: `Matched Series Profile: ${data.profile.name}`,
          });
        }
      } catch {
        // ignore auto-match errors
      }
    },
    [apiBaseUrl]
  );

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

      const firstRel = validFiles[0].webkitRelativePath || '';
      const folderPart = firstRel.includes('/') ? firstRel.split('/')[0] : '';
      const firstStem = (validFiles[0].name || '').replace(/\.[^/.]+$/, '');
      const inferredName = folderPart || firstStem;
      if (inferredName) {
        setImportBaseName(inferredName);
      }

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
      if (inferredName) {
        autoMatchSeriesProfile(inferredName);
      }

      // Persist loose files into backend cache in background so .vellum.json Save/Open and autosave stay durable
      (async () => {
        const uploadedUrls = [];
        for (const page of newPages) {
          if (!page.file) continue;
          try {
            const fd = new FormData();
            fd.append('image', page.file, page.name || 'page.png');
            const resp = await fetch(`${apiBaseUrl}/api/cache/upload`, {
              method: 'POST',
              body: fd,
            });
            if (!resp.ok) continue;
            const data = await resp.json();
            if (data?.image_url) {
              uploadedUrls.push(data.image_url);
              const durableUrl = resolveBackendUrl(data.image_url, apiBaseUrl);
              updatePage(
                page.id,
                { originalUrl: durableUrl },
                { recordHistory: false }
              );
              if (page.originalUrl?.startsWith('blob:')) {
                try {
                  URL.revokeObjectURL(page.originalUrl);
                } catch {
                  // ignore
                }
              }
            }
          } catch {
            // Keep blob: URL fallback if upload fails
          }
        }
        if (uploadedUrls.length > 0) {
          fetch(`${apiBaseUrl}/api/projects_cache/pin`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              project_id: projectCacheIdRef.current,
              image_urls: uploadedUrls,
            }),
          }).catch(() => {});
        }
      })();
    },
    [apiBaseUrl, sourceLang, targetLang, addPages, updatePage]
  );

  // Import a .cbz or .zip comic archive via POST /api/import/archive (queue-only, no auto-run)
  const handleImportArchive = useCallback(
    async (archiveFile) => {
      if (!archiveFile) return;
      setStatusBanner(null);
      try {
        let response;
        const localPath =
          typeof archiveFile === 'string'
            ? archiveFile
            : typeof archiveFile?.path === 'string' && archiveFile.path.trim()
            ? archiveFile.path.trim()
            : null;
        if (localPath) {
          response = await fetch(`${apiBaseUrl}/api/import/archive-path`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: localPath }),
          });
        } else {
          const formData = new FormData();
          formData.append('archive', archiveFile, archiveFile.name);
          response = await fetch(`${apiBaseUrl}/api/import/archive`, {
            method: 'POST',
            body: formData,
          });
        }
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

        const archiveLabel = data.archive_name || archiveFile.name || '';
        const archiveStem = archiveLabel.replace(/\.[^/.]+$/, '');
        if (archiveStem) {
          setImportBaseName(archiveStem);
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
        const archiveUrls = extracted.map((p) => p.image_url).filter(Boolean);
        if (archiveUrls.length > 0) {
          fetch(`${apiBaseUrl}/api/projects_cache/pin`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              project_id: projectCacheIdRef.current,
              image_urls: archiveUrls,
            }),
          }).catch(() => {});
        }
        setStatusBanner({
          type: 'success',
          message: `Imported ${importedPages.length} pages from ${data.archive_name || archiveFile.name}. Press Run to process.`,
        });
        if (archiveStem) {
          autoMatchSeriesProfile(archiveStem);
        }
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

  const {
    runSingleStageOnPage,
    runStagesForPages,
    handleCancelBatch,
    handleStopBatchQueue,
    handleOpenQueueItemInEditor,
    handleBatchExportQueueItems,
    handleClearQueueAndPurgeCaches,
    handleStartBatchQueue,
  } = useBatchProcessor({
    activePageId,
    activeQueueItemIdRef,
    activeQueueItemMetaRef,
    apiBaseUrl,
    batchAbortControllerRef,
    batchCancelRef,
    canvasRef,
    importBaseName,
    isQueueRunning,
    pages,
    pipelineConfig,
    projectCacheIdRef,
    queueCancelRef,
    queueExportFormat,
    queueFilenamePattern,
    queueItems,
    queueOutputDir,
    queuePurgeCache,
    queueRunMode,
    queueShareContinuity,
    queueSkipExisting,
    reimportedQueueItemIdsRef,
    replaceProjectState,
    setActiveProfileSlug,
    setImportBaseName,
    setIsBatchQueueOpen,
    setIsPipelineRunning,
    setIsQueueRunning,
    setIsStopping,
    setPageStatus,
    setPipelineProgress,
    setQueueItems,
    setQuickProviderPrompt,
    setStatusBanner,
    updatePage,
    viewLayers,
    wakePipelineWaitersRef,
  });

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
      const candidates = skipCompleted
        ? state.pages.filter((p) => p.status !== 'done')
        : state.pages;
      if (state.pages.length > 0 && candidates.length === 0) {
        setStatusBanner({
          type: 'success',
          message: 'All pages are already marked done. Uncheck "Skip Done" to re-run them.',
        });
        return;
      }
      targetIds = candidates.map((p) => p.id);
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
  }, [runStagesForPages, skipCompleted]);

  const handleProcessProject = useCallback(() => {
    const state = useEditorStore.getState();
    const stages = ['detect', 'ocr', 'inpaint', 'translate'].filter((k) =>
      Boolean(state.pipelineConfig.enabledStages?.[k])
    );
    const candidates = skipCompleted
      ? state.pages.filter((p) => p.status !== 'done')
      : state.pages;
    if (state.pages.length > 0 && candidates.length === 0) {
      setStatusBanner({
        type: 'success',
        message: 'All pages are already marked done. Uncheck "Skip Done" to re-run them.',
      });
      return;
    }
    runStagesForPages(
      candidates.map((p) => p.id),
      stages.length > 0 ? stages : ['detect', 'ocr', 'inpaint', 'translate']
    );
  }, [runStagesForPages, skipCompleted]);

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
      setStatusBanner(null);
      try {
        const normalizedFmt =
          String(format).toLowerCase() === 'jpg' ||
          String(format).toLowerCase() === 'jpeg'
            ? 'jpg'
            : 'png';
        const baseName = activePage.name.replace(/\.[^/.]+$/, '') || importBaseName || 'comic_page';
        const filename = `${baseName}.${normalizedFmt}`;

        const picked = await pickNativeSavePath(apiBaseUrl, {
          defaultName: filename,
          defaultExt: normalizedFmt,
          title: `Save .${normalizedFmt.toUpperCase()} Image To...`,
        });
        if (picked.available && picked.cancelled) return;

        await ensurePageFontsLoaded(activePage, null, apiBaseUrl);
        const dataUrl = canvasRef.current.exportImage(normalizedFmt, 0.98);
        if (!dataUrl) return;

        const mimeType = normalizedFmt === 'jpg' ? 'image/jpeg' : 'image/png';
        const resp = await fetch(dataUrl);
        const blob = await resp.blob();

        if (picked.available && picked.dir && picked.filename) {
          const formData = new FormData();
          formData.append('file', blob, picked.filename);
          formData.append('output_dir', picked.dir);
          formData.append('output_filename', picked.filename);
          formData.append('overwrite', 'true');
          const saveResp = await fetch(`${apiBaseUrl}/api/export/save-file`, {
            method: 'POST',
            body: formData,
          });
          if (!saveResp.ok) {
            const errJson = await saveResp.json().catch(() => ({}));
            throw new Error(errJson?.detail || `HTTP ${saveResp.status}`);
          }
          const saveData = await saveResp.json();
          setStatusBanner({
            type: 'success',
            message: `Exported to ${saveData.saved_path || picked.path || filename} (${activePage.dimensions?.width || ''}×${activePage.dimensions?.height || ''}px).`,
          });
          return;
        }

        const saveRes = await saveBlobWithPicker(blob, filename, {
          description: normalizedFmt === 'jpg' ? 'JPEG Image' : 'PNG Image',
          accept: { [mimeType]: [`.${normalizedFmt}`] },
        });
        if (saveRes?.cancelled) return;

        setStatusBanner({
          type: 'success',
          message: `Exported ${saveRes.filename || filename} (${activePage.dimensions?.width || ''}×${activePage.dimensions?.height || ''}px).`,
        });
      } catch (err) {
        setStatusBanner({
          type: 'error',
          message: `Export .${format} failed: ${err?.message || 'Unknown error'}`,
        });
      }
    },
    [activePage, apiBaseUrl, importBaseName]
  );

  // Export multi-page project as 1:1 native-resolution .cbz or .pdf via native C++17 exporter endpoints
  const handleExportMultiPage = useCallback(
    async (format) => {
      if (pages.length === 0) return;
      setStatusBanner(null);
      try {
        const fmtLower = String(format).toLowerCase() === 'pdf' ? 'pdf' : 'cbz';
        const cleanBase = (importBaseName || '').trim();
        const exportStem = cleanBase || 'vellum';
        const defaultFilename = `${exportStem}.${fmtLower}`;

        const picked = await pickNativeSavePath(apiBaseUrl, {
          defaultName: defaultFilename,
          defaultExt: fmtLower,
          title: `Save .${fmtLower.toUpperCase()} Export To...`,
        });
        if (picked.available && picked.cancelled) return;

        let result = null;
        if (canvasRef.current?.exportArchive) {
          result = await canvasRef.current.exportArchive(pages, fmtLower, {
            filename: exportStem,
            imageFormat: 'png',
            quality: 0.98,
            ...(picked.available && picked.dir
              ? {
                  outputDir: picked.dir,
                  outputFilename: picked.filename || defaultFilename,
                  overwrite: true,
                }
              : {}),
            onProgress: ({ current, total, phase }) => {
              setStatusBanner({
                type: 'info',
                message:
                  phase === 'finalizing'
                    ? `Finalizing ${total}-page .${fmtLower.toUpperCase()} archive on disk...`
                    : `Exporting .${fmtLower.toUpperCase()}: rendering & uploading page ${current} / ${total}...`,
              });
            },
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
          const resp = await fetch(`${apiBaseUrl}/api/export/${fmtLower}`, {
            method: 'POST',
            body: formData,
          });
          if (!resp.ok) {
            const errJson = await resp.json().catch(() => ({}));
            throw new Error(errJson?.detail || `HTTP ${resp.status}`);
          }
          result = {
            blob: await resp.blob(),
            filename: defaultFilename,
            pageCount: pages.length,
          };
        }

        if (result?.savedPath) {
          setStatusBanner({
            type: 'success',
            message: `Exported ${result.pageCount} page${
              result.pageCount === 1 ? '' : 's'
            } at 1:1 native resolution to ${result.savedPath}.`,
          });
          return;
        }

        if (result?.downloadUrl) {
          const link = document.createElement('a');
          link.href = result.downloadUrl;
          link.download = result.filename || defaultFilename;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          setStatusBanner({
            type: 'success',
            message: `Exported ${result.pageCount} page${
              result.pageCount === 1 ? '' : 's'
            } at 1:1 native resolution to ${result.filename}.`,
          });
          return;
        }

        const mimeType =
          fmtLower === 'pdf' ? 'application/pdf' : 'application/vnd.comicbook+zip';
        const saveRes = await saveBlobWithPicker(result.blob, result.filename, {
          description: fmtLower === 'pdf' ? 'PDF Document' : 'CBZ Comic Archive',
          accept: { [mimeType]: [`.${fmtLower}`] },
        });
        if (saveRes?.cancelled) {
          setStatusBanner(null);
          return;
        }
        setStatusBanner({
          type: 'success',
          message: `Exported ${result.pageCount} page${
            result.pageCount === 1 ? '' : 's'
          } at 1:1 native resolution to ${saveRes.filename || result.filename}.`,
        });
      } catch (err) {
        setStatusBanner({
          type: 'error',
          message: `Export .${format} failed: ${err?.message || 'Unknown error'}`,
        });
      }
    },
    [pages, apiBaseUrl, importBaseName]
  );

  // Export single or multi-page layered Photoshop .psd (or .zip of .psd files) via native C++17 PSD writer
  const handleExportPsd = useCallback(async () => {
    if (pages.length === 0) return;
    setStatusBanner(null);
    try {
      const cleanBase = (importBaseName || '').trim();
      const singlePageName =
        pages.length === 1
          ? pages[0].name.replace(/\.[^/.]+$/, '') || cleanBase || 'comic_page'
          : cleanBase || 'vellum';
      const exportStem = singlePageName;
      const isMulti = pages.length > 1;
      const outExt = isMulti ? 'zip' : 'psd';
      const defaultFilename = `${exportStem}.${outExt}`;

      if (!canvasRef.current?.exportPsd) {
        throw new Error('Layered PSD exporter is not ready.');
      }

      const picked = await pickNativeSavePath(apiBaseUrl, {
        defaultName: defaultFilename,
        defaultExt: outExt,
        title: isMulti ? 'Save Layered PSD Archive (.zip) To...' : 'Save Layered .PSD To...',
      });
      if (picked.available && picked.cancelled) return;

      const result = await canvasRef.current.exportPsd(pages, {
        filename: exportStem,
        editableTextLayers: true,
        ...(picked.available && picked.dir
          ? {
              outputDir: picked.dir,
              outputFilename: picked.filename || defaultFilename,
              overwrite: true,
            }
          : {}),
        onProgress: ({ current, total, phase }) => {
          setStatusBanner({
            type: 'info',
            message:
              phase === 'finalizing'
                ? `Building ${total}-page layered .PSD on backend...`
                : `Exporting layered .PSD: preparing page ${current} / ${total}...`,
          });
        },
      });

      if (result?.savedPath) {
        setStatusBanner({
          type: 'success',
          message: `Exported ${result.pageCount} layered PSD page${
            result.pageCount === 1 ? '' : 's'
          } to ${result.savedPath}.`,
        });
        return;
      }

      const isZip = Boolean(result.isZip);
      const saveRes = await saveBlobWithPicker(result.blob, result.filename, {
        description: isZip ? 'Layered PSD Archive (.zip)' : 'Photoshop Document (.psd)',
        accept: isZip
          ? { 'application/zip': ['.zip'] }
          : { 'image/vnd.adobe.photoshop': ['.psd'] },
      });
      if (saveRes?.cancelled) {
        setStatusBanner(null);
        return;
      }

      setStatusBanner({
        type: 'success',
        message: `Exported ${result.pageCount} layered PSD page${
          result.pageCount === 1 ? '' : 's'
        } to ${saveRes.filename || result.filename}.`,
      });
    } catch (err) {
      setStatusBanner({
        type: 'error',
        message: `Export Layered .psd failed: ${err?.message || 'Unknown error'}`,
      });
    }
  }, [pages, apiBaseUrl, importBaseName]);

  // New Project / Clear Queue: reset workspace and clear backend LLM conversation memory
  const handleNewProject = useCallback(() => {
    const prevPid = projectCacheIdRef.current;
    if (prevPid) {
      fetch(`${apiBaseUrl}/api/conversation/${encodeURIComponent(prevPid)}`, {
        method: 'DELETE',
      }).catch(() => {});
    }
    projectCacheIdRef.current = `proj_${Date.now()}`;
    replaceProjectState({ pages: [], activePageId: null });
    setImportBaseName('');
    setPageFilter('all');
    setStatusBanner({
      type: 'success',
      message: 'Started a new project and cleared prior series conversation memory.',
    });
  }, [apiBaseUrl, replaceProjectState]);

  // Save Project (.vellum.json) with pinned cache images and full typesetting settings
  const handleSaveProject = useCallback(async () => {
    if (pages.length === 0) return;
    const { apiKey: _redactedKey, providerProfiles: rawProfiles, ...restPipelineConfig } = pipelineConfig;
    const safeProviderProfiles = rawProfiles && typeof rawProfiles === 'object'
      ? Object.fromEntries(
          Object.entries(rawProfiles).map(([pid, prof]) => [
            pid,
            prof && typeof prof === 'object' ? { ...prof, api_key: '' } : prof,
          ])
        )
      : undefined;
    const safePipelineConfig = {
      ...restPipelineConfig,
      ...(safeProviderProfiles ? { providerProfiles: safeProviderProfiles } : {}),
    };
    const projectSlug =
      (importBaseName || '').trim().replace(/[^a-zA-Z0-9_-]+/g, '_') ||
      `proj_${Date.now()}`;

    // Pin any /api/cache/images/... URLs into ./projects_cache/<project_id>/ so they are exempt from LRU pruning
    const urlsToPin = [];
    for (const p of pages) {
      for (const candidate of [p.originalUrl, p.cleanedImage, p.inpaintedUrl, p.maskUrl]) {
        if (typeof candidate === 'string' && candidate.includes('/api/cache/images/')) {
          urlsToPin.push(candidate);
        }
      }
    }
    let pinnedMap = {};
    if (urlsToPin.length > 0) {
      try {
        const pinResp = await fetch(`${apiBaseUrl}/api/cache/pin`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            project_id: projectSlug,
            image_urls: Array.from(new Set(urlsToPin)),
          }),
        });
        if (pinResp.ok) {
          const pinData = await pinResp.json();
          pinnedMap = pinData?.pinned_urls || {};
        }
      } catch {
        // Fallback to original cache URLs if pin fails
      }
    }

    const resolvePinnedUrl = (u) => {
      if (!u || typeof u !== 'string' || u.startsWith('blob:')) return null;
      return pinnedMap[u] ? resolveBackendUrl(pinnedMap[u], apiBaseUrl) : u;
    };

    const projectPayload = {
      format: 'vellum-project',
      version: 2,
      savedAt: new Date().toISOString(),
      activePageId,
      viewLayers,
      pipelineConfig: safePipelineConfig,
      typesetting: useEditorStore.getState().typesetting,
      pages: pages.map((p) => {
        const cleaned = resolvePinnedUrl(p.cleanedImage || p.inpaintedUrl);
        return {
          id: p.id,
          name: p.name,
          originalUrl: resolvePinnedUrl(p.originalUrl),
          cleanedImage: cleaned,
          inpaintedUrl: cleaned,
          maskUrl: resolvePinnedUrl(p.maskUrl),
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
        };
      }),
    };

    const blob = new Blob([JSON.stringify(projectPayload, null, 2)], {
      type: 'application/json',
    });
    const projectFilename = `${(importBaseName || '').trim() || 'vellum-project'}.vellum.json`;
    const saveRes = await saveBlobWithPicker(blob, projectFilename, {
      description: 'Vellum Project',
      accept: { 'application/json': ['.json'] },
    });
    if (saveRes?.cancelled) return;
    setStatusBanner({
      type: 'success',
      message: `Saved project (${pages.length} page${pages.length === 1 ? '' : 's'}) to ${saveRes.filename || projectFilename}.`,
    });
  }, [apiBaseUrl, pages, activePageId, viewLayers, pipelineConfig, importBaseName]);

  // Open Project (.vellum.json)
  const handleOpenProject = useCallback(
    async (file) => {
      if (!file) return;
      try {
        const text = await file.text();
        const parsed = JSON.parse(text);
        if (!parsed || !Array.isArray(parsed.pages)) {
          throw new Error('Invalid .vellum.json project file.');
        }
        replaceProjectState({
          pages: parsed.pages,
          activePageId: parsed.activePageId || null,
          pipelineConfig: parsed.pipelineConfig || null,
          viewLayers: parsed.viewLayers || null,
          typesetting: parsed.typesetting || null,
        });
        setAutosaveSnapshot(null);
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

  // Check for disk-backed crash recovery autosave on startup (with localStorage fallback)
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const resp = await fetch(`${apiBaseUrl}/api/projects_cache/autosave`);
        if (resp.ok) {
          const parsed = await resp.json();
          if (!cancelled && parsed && Array.isArray(parsed.pages) && parsed.pages.length > 0) {
            setAutosaveSnapshot(parsed);
            return;
          }
        }
      } catch {
        // Fallback to localStorage check below
      }
      try {
        const raw = localStorage.getItem(AUTOSAVE_STORAGE_KEY);
        if (!raw) return;
        const parsed = JSON.parse(raw);
        if (!cancelled && parsed && Array.isArray(parsed.pages) && parsed.pages.length > 0) {
          setAutosaveSnapshot(parsed);
        }
      } catch {
        // ignore corrupted autosave
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl]);

  // Debounced disk-backed crash autosave whenever pages or activePageId change
  useEffect(() => {
    if (pages.length === 0) return undefined;
    const debounceMs = isPipelineRunning ? 5000 : 1000;
    const timer = setTimeout(() => {
      try {
        const state = useEditorStore.getState();
        const { apiKey: _redactedKey, providerProfiles: rawProfiles, ...restPipelineConfig } = state.pipelineConfig;
        const safeProviderProfiles = rawProfiles && typeof rawProfiles === 'object'
          ? Object.fromEntries(
              Object.entries(rawProfiles).map(([pid, prof]) => [
                pid,
                prof && typeof prof === 'object' ? { ...prof, api_key: '' } : prof,
              ])
            )
          : undefined;
        const safePipelineConfig = {
          ...restPipelineConfig,
          ...(safeProviderProfiles ? { providerProfiles: safeProviderProfiles } : {}),
        };
        const savedAt = new Date().toISOString();
        const snapshot = {
          format: 'vellum-autosave',
          savedAt,
          importBaseName,
          activePageId: state.activePageId,
          viewLayers: state.viewLayers,
          pipelineConfig: safePipelineConfig,
          typesetting: state.typesetting,
          pages: state.pages.map((p) => ({
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
            blocks: (p.blocks || []).map((b) => {
              const { polygon: _poly, mask: _mask, ...lightBlock } = b || {};
              if (b?.mask && typeof b.mask === 'object') {
                const { polygon: _mpoly, ...lightMask } = b.mask;
                lightBlock.mask = lightMask;
              }
              return lightBlock;
            }),
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
        // Store only a tiny flag in localStorage to avoid WebView2 5-10 MB QuotaExceededError on 3,000-page volumes
        try {
          localStorage.setItem(
            AUTOSAVE_STORAGE_KEY,
            JSON.stringify({
              savedAt,
              title: importBaseName || 'Untitled',
              pageCount: state.pages.length,
            })
          );
        } catch {
          // ignore localStorage errors
        }
        fetch(`${apiBaseUrl}/api/projects_cache/autosave`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(snapshot),
        }).catch(() => {});
        const activePid = projectCacheIdRef.current;
        if (activePid && String(activePid).startsWith('qproj_')) {
          fetch(`${apiBaseUrl}/api/projects_cache/autosave`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ...snapshot, project_id: activePid }),
          }).catch(() => {});
        }
      } catch {
        // ignore serialization errors
      }
    }, debounceMs);
    return () => clearTimeout(timer);
  }, [pages, activePageId, importBaseName, apiBaseUrl, isPipelineRunning]);

  const handleRestoreAutosave = useCallback(() => {
    if (!autosaveSnapshot?.pages?.length) return;
    replaceProjectState({
      pages: autosaveSnapshot.pages,
      activePageId: autosaveSnapshot.activePageId || null,
      pipelineConfig: autosaveSnapshot.pipelineConfig || null,
      viewLayers: autosaveSnapshot.viewLayers || null,
      typesetting: autosaveSnapshot.typesetting || null,
    });
    if (autosaveSnapshot.importBaseName) {
      setImportBaseName(autosaveSnapshot.importBaseName);
    }
    setAutosaveSnapshot(null);
    setStatusBanner({
      type: 'success',
      message: `Restored last session (${autosaveSnapshot.pages.length} pages).`,
    });
  }, [autosaveSnapshot, replaceProjectState]);

  const handleDismissAutosave = useCallback(() => {
    setAutosaveSnapshot(null);
    try {
      localStorage.removeItem(AUTOSAVE_STORAGE_KEY);
    } catch {
      // ignore
    }
    fetch(`${apiBaseUrl}/api/projects_cache/autosave`, {
      method: 'DELETE',
    }).catch(() => {});
  }, [apiBaseUrl]);

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

        const sampleStem = filename.replace(/\.[^/.]+$/, '');
        if (sampleStem) {
          setImportBaseName(sampleStem);
        }

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

  useEffect(() => {
    const el = filmstripRef.current;
    if (!el) return undefined;
    const updateHeight = () => {
      if (el.clientHeight > 0) setFilmstripViewportHeight(el.clientHeight);
    };
    updateHeight();
    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver(updateHeight);
      ro.observe(el);
      return () => ro.disconnect();
    }
    window.addEventListener('resize', updateHeight);
    return () => window.removeEventListener('resize', updateHeight);
  }, [zenMode]);

  const FILMSTRIP_ITEM_HEIGHT = uiDensity === 'comfortable' ? 82 : 64;
  const selectedPageIdSet = useMemo(() => new Set(selectedPageIds), [selectedPageIds]);
  const FILMSTRIP_OVERSCAN = 4;

  const qcByPageId = useMemo(() => {
    const map = new Map();
    for (const p of pages) {
      map.set(p.id, getPageQcFlags(p, targetLang));
    }
    return map;
  }, [pages, targetLang]);

  const issuePages = useMemo(
    () => pages.filter((p) => qcByPageId.get(p.id)?.hasIssue),
    [pages, qcByPageId]
  );

  const displayedPages = useMemo(
    () => (pageFilter === 'issues' ? issuePages : pages),
    [pageFilter, issuePages, pages]
  );

  const jumpToIssuePage = useCallback(
    (direction = 1) => {
      if (issuePages.length === 0) return;
      const curIdx = pages.findIndex((p) => p.id === activePageId);
      let target = null;
      if (direction >= 0) {
        target =
          issuePages.find((p) => pages.findIndex((x) => x.id === p.id) > curIdx) ||
          issuePages[0];
      } else {
        const reversed = [...issuePages].reverse();
        target =
          reversed.find((p) => pages.findIndex((x) => x.id === p.id) < curIdx) ||
          reversed[0];
      }
      if (target) {
        setActivePage(target.id);
        const list = pageFilter === 'issues' ? issuePages : pages;
        const visIdx = list.findIndex((x) => x.id === target.id);
        if (visIdx >= 0 && filmstripRef.current) {
          const top = visIdx * FILMSTRIP_ITEM_HEIGHT;
          filmstripRef.current.scrollTop = Math.max(0, top - FILMSTRIP_ITEM_HEIGHT);
        }
      }
    },
    [issuePages, pages, activePageId, setActivePage, pageFilter, FILMSTRIP_ITEM_HEIGHT]
  );

  const findMatchStats = useMemo(() => {
    if (!isFindReplaceOpen || !findText) {
      return { matches: 0, pagesCount: 0 };
    }
    try {
      const escaped = findText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const pattern = findWholeWord ? `\\b${escaped}\\b` : escaped;
      const regex = new RegExp(pattern, findCaseSensitive ? 'g' : 'gi');
      let matches = 0;
      let pagesCount = 0;
      for (const p of pages) {
        let pageHasMatch = false;
        for (const b of p.blocks || []) {
          const txt = String(b.translated_text ?? '');
          if (!txt) continue;
          const m = txt.match(regex);
          if (m && m.length > 0) {
            matches += m.length;
            pageHasMatch = true;
          }
        }
        if (pageHasMatch) pagesCount += 1;
      }
      return { matches, pagesCount };
    } catch {
      return { matches: 0, pagesCount: 0 };
    }
  }, [isFindReplaceOpen, findText, findCaseSensitive, findWholeWord, pages]);

  useEffect(() => {
    const handleGlobalShortcuts = (e) => {
      const mod = e.ctrlKey || e.metaKey;
      if (mod && !e.altKey && String(e.key || '').toLowerCase() === 'h') {
        e.preventDefault();
        setFindReplaceStatus('');
        setIsFindReplaceOpen((prev) => !prev);
        return;
      }
      if (e.altKey && !mod && (e.key === ']' || e.code === 'BracketRight')) {
        e.preventDefault();
        jumpToIssuePage(1);
        return;
      }
      if (e.altKey && !mod && (e.key === '[' || e.code === 'BracketLeft')) {
        e.preventDefault();
        jumpToIssuePage(-1);
      }
    };
    window.addEventListener('keydown', handleGlobalShortcuts);
    return () => window.removeEventListener('keydown', handleGlobalShortcuts);
  }, [jumpToIssuePage]);

  const toThumbnailUrl = useCallback((url) => {
    if (!url || typeof url !== 'string') return url;
    if (url.startsWith('blob:') || url.startsWith('data:')) return url;
    const match = url.match(/^(.*)\/api\/cache\/images\/([^?#]+)(?:[?#].*)?$/i);
    if (!match) return url;
    const prefix = match[1] || '';
    const imageId = match[2];
    return `${prefix}/api/cache/thumb/${imageId}?w=180`;
  }, []);
  const filmstripStartIdx = Math.max(
    0,
    Math.floor(filmstripScrollTop / FILMSTRIP_ITEM_HEIGHT) - FILMSTRIP_OVERSCAN
  );
  const filmstripEndIdx = Math.min(
    displayedPages.length,
    Math.ceil(
      (filmstripScrollTop + filmstripViewportHeight) / FILMSTRIP_ITEM_HEIGHT
    ) + FILMSTRIP_OVERSCAN
  );

  const isAnyProcessing =
    isPipelineRunning || pages.some((p) => p.status === 'processing');

  // Mirror live page progress (p.X/Y) into the active BatchQueueModal row while queue is running
  useEffect(() => {
    const qid = activeQueueItemIdRef.current;
    if (!isQueueRunning || !qid || !pipelineProgress?.totalPages) return;
    const cur = pipelineProgress.currentPageIndex || 1;
    const tot = pipelineProgress.totalPages;
    const nextText = `Processing p.${cur}/${tot}`;
    setQueueItems((prev) => {
      const target = prev.find((q) => q.id === qid);
      if (!target || target.status !== 'processing' || target.progressText === nextText) {
        return prev;
      }
      return prev.map((q) => (q.id === qid ? { ...q, progressText: nextText } : q));
    });
  }, [isQueueRunning, pipelineProgress?.currentPageIndex, pipelineProgress?.totalPages]);

  useEffect(() => {
    if (!isAnyProcessing) {
      setElapsedSeconds(0);
      setLiveBackendStage(null);
      return;
    }
    const baseTime = pipelineProgress?.startedAt || Date.now();
    setElapsedSeconds(Math.max(0, Math.floor((Date.now() - baseTime) / 1000)));
    const timer = window.setInterval(() => {
      setElapsedSeconds(Math.max(0, Math.floor((Date.now() - baseTime) / 1000)));
    }, 1000);

    let cancelled = false;
    const pollStage = async () => {
      try {
        const res = await fetch(`${apiBaseUrl}/api/pipeline/status`);
        if (!res.ok || cancelled) return;
        const data = await res.json();
        if (!cancelled) {
          setLiveBackendStage(data?.active ? data : null);
        }
      } catch {
        // ignore transient poll errors
      }
    };
    pollStage();
    const stageTimer = window.setInterval(pollStage, 350);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
      window.clearInterval(stageTimer);
    };
  }, [isAnyProcessing, pipelineProgress?.startedAt, apiBaseUrl]);

  const activeProgressMeta = useMemo(() => {
    if (!isAnyProcessing) return null;
    const processingIdx = pages.findIndex((p) => p.status === 'processing');
    const processingPage = processingIdx >= 0 ? pages[processingIdx] : activePage;
    const currentPageIndex =
      pipelineProgress?.currentPageIndex ||
      (processingIdx >= 0 ? processingIdx + 1 : 1);
    const totalPages =
      pipelineProgress?.totalPages || Math.max(1, pages.length);
    const baseStageKey =
      pipelineProgress?.stage || processingPage?.activeStage || 'pipeline';
    const hasLiveStage =
      baseStageKey === 'pipeline' &&
      liveBackendStage?.active &&
      typeof liveBackendStage?.stage === 'string' &&
      liveBackendStage.stage.length > 0;
    const stageKey = hasLiveStage ? liveBackendStage.stage : baseStageKey;
    const stageIndex = hasLiveStage
      ? liveBackendStage.stage_index || 1
      : pipelineProgress?.stageIndex || 1;
    const totalStages = hasLiveStage
      ? liveBackendStage.total_stages || 4
      : pipelineProgress?.totalStages || 1;

    const totalSteps = Math.max(1, totalPages * totalStages);
    const completedSteps = (currentPageIndex - 1) * totalStages + (stageIndex - 0.5);
    const percent = Math.min(99, Math.max(6, Math.round((completedSteps / totalSteps) * 100)));

    const stageLabels = {
      detect: 'Detecting',
      ocr: 'OCR',
      inpaint: 'Inpainting',
      inpaint_translate: 'Inpaint + TL',
      translate: 'Translating',
      pipeline: 'Full Pipeline',
    };
    const stageLabel = stageLabels[stageKey] || String(stageKey).toUpperCase();

    const m = modelsStatus?.models;
    const lm = modelsStatus?.lm_studio;
    let modelName = '';
    if (hasLiveStage && liveBackendStage?.detail) {
      modelName = liveBackendStage.detail;
    } else if (stageKey === 'detect') {
      modelName =
        pipelineConfig.detectionModel ||
        m?.comic_text_detector?.active_filename ||
        m?.comic_text_detector?.files?.[0] ||
        m?.comic_text_detector?.active_model ||
        m?.comic_text_detector?.model_id ||
        'yolo11m-seg';
    } else if (stageKey === 'ocr') {
      modelName =
        sourceLang === 'ja'
          ? m?.manga_ocr?.active_filename ||
            m?.manga_ocr?.files?.[0] ||
            m?.manga_ocr?.active_model ||
            'manga-ocr-base'
          : m?.rapid_ocr_ko_zh?.active_filename ||
            m?.rapid_ocr_ko_zh?.files?.[0] ||
            m?.rapid_ocr_ko_zh?.active_model ||
            'RapidOCR PP-OCRv4';
    } else if (stageKey === 'inpaint') {
      modelName =
        pipelineConfig.inpaintingModel ||
        m?.lama_inpaint?.active_filename ||
        m?.lama_inpaint?.files?.[0] ||
        m?.lama_inpaint?.active_model ||
        'lama-manga';
    } else {
      modelName =
        lm?.active_model ||
        lm?.resolved_model_id ||
        pipelineConfig.model ||
        'LM Studio';
    }

    return {
      currentPageIndex,
      totalPages,
      stageLabel,
      modelName,
      percent,
    };
  }, [
    isAnyProcessing,
    pages,
    activePage,
    pipelineProgress,
    liveBackendStage,
    modelsStatus,
    pipelineConfig.detectionModel,
    pipelineConfig.inpaintingModel,
    pipelineConfig.model,
    sourceLang,
  ]);

  return (
    <div className="min-h-screen h-screen w-screen flex flex-col bg-studio-canvas text-studio-text overflow-hidden select-none">
      {/* Bar 1: Compact 28px Native Application Menu Bar */}
      <MenuBar
        onImportFiles={handleImportFiles}
        onImportArchive={handleImportArchive}
        onExportImage={handleExportImage}
        onExportCbz={() => handleExportMultiPage('cbz')}
        onExportPdf={() => handleExportMultiPage('pdf')}
        onExportPsd={handleExportPsd}
        onSaveProject={handleSaveProject}
        onOpenProject={handleOpenProject}
        onNewProject={handleNewProject}
        onProcessProject={handleProcessProject}
        onProcessSelectedPages={handleProcessSelectedPages}
        onProcessSelectedLayers={handleProcessSelectedLayers}
        onOpenBatchQueue={() => setIsBatchQueueOpen(true)}
        onRunStage={handleRunSingleStage}
        onFitScreen={handleFitScreen}
        onOpenModelsModal={() => setIsDownloadModalOpen(true)}
        isProcessing={isAnyProcessing}
        isWorkspaceUnlocked={isWorkspaceUnlocked}
      />

      {/* Bar 2: Compact 36px Pipeline Front Control Bar */}
      <FrontControlBar
        onRun={handleFrontControlRun}
        onCancel={handleCancelBatch}
        onOpenSettings={handleOpenSettings}
        onOpenBatchQueue={() => setIsBatchQueueOpen(true)}
        activeProfileSlug={activeProfileSlug}
        onSelectProfileSlug={setActiveProfileSlug}
        defaultSeriesTitle={importBaseName}
        profilesVersion={profilesVersion}
        onProfilesChanged={() => setProfilesVersion((v) => v + 1)}
        isProcessing={isAnyProcessing || isQueueRunning}
        isStopping={isStopping}
        isWorkspaceUnlocked={isWorkspaceUnlocked}
        skipCompleted={skipCompleted}
        onToggleSkipCompleted={() => setSkipCompleted((v) => !v)}
      />

      {/* Crash Recovery Autosave Prompt */}
      {autosaveSnapshot && pages.length === 0 && (
        <div className="px-3 py-1.5 text-[11px] flex items-center justify-between border-b border-studio-border bg-studio-accent/10 text-studio-text">
          <div className="flex items-center gap-2 truncate">
            <span className="font-semibold text-studio-accent">Restore Last Session:</span>
            <span className="truncate text-studio-text-secondary">
              {autosaveSnapshot.pages.length} page{autosaveSnapshot.pages.length === 1 ? '' : 's'} autosaved
              {autosaveSnapshot.savedAt ? ` (${new Date(autosaveSnapshot.savedAt).toLocaleTimeString()})` : ''}
            </span>
          </div>
          <div className="flex items-center gap-2 shrink-0 ml-4">
            <button
              type="button"
              onClick={handleRestoreAutosave}
              className="px-2 py-0.5 rounded bg-studio-accent text-studio-accent-fg font-medium hover:opacity-90 cursor-pointer"
            >
              Restore
            </button>
            <button
              type="button"
              onClick={handleDismissAutosave}
              className="text-studio-text-secondary hover:text-studio-text cursor-pointer"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

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
              <div className="flex items-center gap-1">
                {selectedPageIds.length > 1 && (
                  <span className="rounded-studio-sm bg-studio-accent/15 border border-studio-accent/40 px-1.5 py-0.5 font-mono text-[10px] leading-none text-studio-accent tabular-nums">
                    {selectedPageIds.length} sel
                  </span>
                )}
                <button
                  type="button"
                  onClick={() => setPageFilter('all')}
                  className={`rounded-studio-sm border px-1.5 py-0.5 font-mono text-[10px] leading-none tabular-nums cursor-pointer transition-colors ${
                    pageFilter === 'all'
                      ? 'border-studio-accent bg-studio-accent/15 text-studio-accent'
                      : 'border-studio-border bg-studio-surface text-studio-text-secondary hover:text-studio-text'
                  }`}
                  title="Show all pages"
                >
                  All {pages.length}
                </button>
                <button
                  type="button"
                  onClick={() => setPageFilter('issues')}
                  className={`rounded-studio-sm border px-1.5 py-0.5 font-mono text-[10px] leading-none tabular-nums cursor-pointer transition-colors ${
                    pageFilter === 'issues'
                      ? 'border-studio-warning bg-studio-warning/15 text-studio-warning'
                      : issuePages.length > 0
                      ? 'border-studio-warning/40 bg-studio-surface text-studio-warning hover:bg-studio-warning/10'
                      : 'border-studio-border bg-studio-surface text-studio-text-muted'
                  }`}
                  title="Show pages with QC issues (Alt+] / Alt+[ to jump)"
                >
                  Issues ({issuePages.length})
                </button>
              </div>
            </div>

            <div
              ref={filmstripRef}
              onScroll={(e) => setFilmstripScrollTop(e.currentTarget.scrollTop)}
              className="flex-1 overflow-y-auto p-2"
            >
              {displayedPages.length === 0 ? (
                <div className="rounded-studio border border-dashed border-studio-border p-4 text-center space-y-1.5 my-1">
                  <FileImage className="h-5 w-5 text-studio-text-muted mx-auto" />
                  <p className="text-xs font-medium text-studio-text-secondary">
                    {pages.length === 0 ? 'No Pages Loaded' : 'No QC Issues Found'}
                  </p>
                  <p className="text-[11px] text-studio-text-muted leading-snug">
                    {pages.length === 0
                      ? 'Use File → Import Pages (Ctrl+O) or drop images onto the canvas.'
                      : 'All processed pages passed CJK leak, echo-retry, and empty checks.'}
                  </p>
                </div>
              ) : (
                <div
                  style={{
                    height: `${displayedPages.length * FILMSTRIP_ITEM_HEIGHT}px`,
                    position: 'relative',
                  }}
                >
                  <div
                    style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      transform: `translate3d(0, ${filmstripStartIdx * FILMSTRIP_ITEM_HEIGHT}px, 0)`,
                      willChange: 'transform',
                    }}
                    className="space-y-1.5"
                  >
                    {displayedPages
                      .slice(filmstripStartIdx, filmstripEndIdx)
                      .map((p, relIdx) => {
                        const idx = pages.findIndex((x) => x.id === p.id);
                        const displayIdx = idx >= 0 ? idx : filmstripStartIdx + relIdx;
                        const isActive = p.id === activePageId;
                        const isSelected = selectedPageIdSet.has(p.id);
                        const qc = qcByPageId.get(p.id);
                        return (
                          <div
                            key={p.id}
                            onClick={(e) => handlePageClick(e, p.id)}
                            className={`group relative ${
                              uiDensity === 'comfortable' ? 'h-[74px] p-2.5' : 'h-[58px] p-1.5'
                            } rounded-studio border transition-colors cursor-pointer flex items-center gap-2 ${
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
                                  src={toThumbnailUrl(p.cleanedImage || p.originalUrl)}
                                  alt={p.name}
                                  loading="lazy"
                                  decoding="async"
                                  onError={(e) => {
                                    const rawUrl = p.cleanedImage || p.originalUrl;
                                    if (!rawUrl || e.currentTarget.dataset.fallback) return;
                                    e.currentTarget.dataset.fallback = '1';
                                    e.currentTarget.src = rawUrl;
                                  }}
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
                                    {String(displayIdx + 1).padStart(2, '0')}
                                  </span>
                                  {p.name}
                                </span>
                                {qc?.hasIssue && (
                                  <span
                                    title={
                                      qc.hasCjkLeak
                                        ? 'Untranslated CJK detected'
                                        : qc.hasTranslationAnomaly
                                        ? 'Translation length/punctuation anomaly (check TL)'
                                        : qc.hasLowOcrConfidence
                                        ? 'Low-confidence OCR detected (check source)'
                                        : qc.echoRetried
                                        ? 'Echo-retried page (check TL)'
                                        : 'Empty translation block'
                                    }
                                    className="h-2 w-2 rounded-full bg-studio-warning shrink-0"
                                  />
                                )}
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
                      })}
                  </div>
                </div>
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

          {/* Bottom-Right Live Progress Bar (Koharu-style, non-blocking) */}
          {activeProgressMeta && (
            <div
              data-testid="bottom-right-progress-bar"
              className="absolute bottom-3 right-3 z-30 w-72 rounded-studio-md border border-studio-border bg-studio-panel/95 backdrop-blur-md px-3 py-2 shadow-2xl pointer-events-none select-none"
            >
              <div className="flex items-center justify-between gap-2 text-[11px]">
                <div className="flex items-center gap-1.5 min-w-0">
                  <Loader2 className="h-3.5 w-3.5 animate-spin text-studio-accent shrink-0" />
                  <span className="font-semibold text-studio-text truncate">
                    Page {activeProgressMeta.currentPageIndex} / {activeProgressMeta.totalPages}
                  </span>
                </div>
                <span className="font-mono text-[10px] text-studio-text-muted tabular-nums shrink-0">
                  {elapsedSeconds}s
                </span>
              </div>

              <div className="mt-1 flex items-center justify-between gap-2 text-[10px] text-studio-text-secondary">
                <span className="font-medium text-studio-accent shrink-0">
                  {activeProgressMeta.stageLabel}
                </span>
                <span
                  className="font-mono truncate text-studio-text-muted"
                  title={activeProgressMeta.modelName}
                >
                  {activeProgressMeta.modelName}
                </span>
              </div>

              <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-studio-surface border border-studio-border">
                <div
                  className="h-full bg-studio-accent transition-all duration-300 ease-out"
                  style={{ width: `${activeProgressMeta.percent}%` }}
                />
              </div>
            </div>
          )}
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
            onOpenFindReplace={() => {
              setFindReplaceStatus('');
              setIsFindReplaceOpen(true);
            }}
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

      {/* Model Download & SSE Verification Modal (lazy-loaded when open) */}
      {isDownloadModalOpen && (
        <Suspense fallback={null}>
          <DownloadModal
            isOpen={isDownloadModalOpen}
            initialStatus={modelsStatus}
            apiBaseUrl={apiBaseUrl}
            onModelsReady={handleModelsReady}
            onClose={
              modelsStatus?.all_ready ? () => setIsDownloadModalOpen(false) : undefined
            }
          />
        </Suspense>
      )}

      {/* Quick Provider Setup when active_provider === "none" and Translate is triggered */}
      <QuickProviderSetupModal
        isOpen={Boolean(quickProviderPrompt)}
        onClose={() => setQuickProviderPrompt(null)}
        onConfiguredTranslate={() => quickProviderPrompt?.onProceedTranslate?.()}
        onRunManualNoLlm={() => quickProviderPrompt?.onProceedManual?.()}
        onOpenFullSettings={() => openModal('settings', 'providers')}
      />

      {/* Project-Wide Find & Replace Modal (Ctrl+H) */}
      {isFindReplaceOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/55 p-4"
          onClick={() => setIsFindReplaceOpen(false)}
        >
          <div
            className="w-full max-w-md rounded-studio-md border border-studio-border bg-studio-panel p-4 shadow-2xl space-y-3"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Search className="h-4 w-4 text-studio-accent" />
                <h3 className="text-xs font-semibold uppercase tracking-wider text-studio-text">
                  Project Find &amp; Replace
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setIsFindReplaceOpen(false)}
                className="p-1 rounded-studio-sm text-studio-text-muted hover:text-studio-text hover:bg-studio-hover cursor-pointer"
                aria-label="Close Find & Replace"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>

            <div className="space-y-2">
              <div>
                <label className="text-[11px] font-medium text-studio-text-secondary block mb-1">
                  Find in translated text
                </label>
                <input
                  type="text"
                  autoFocus
                  value={findText}
                  onChange={(e) => {
                    setFindText(e.target.value);
                    setFindReplaceStatus('');
                  }}
                  placeholder="e.g. Kuroda"
                  className="studio-input w-full h-8 px-2.5 text-xs"
                />
              </div>
              <div>
                <label className="text-[11px] font-medium text-studio-text-secondary block mb-1">
                  Replace with
                </label>
                <input
                  type="text"
                  value={replaceText}
                  onChange={(e) => {
                    setReplaceText(e.target.value);
                    setFindReplaceStatus('');
                  }}
                  placeholder="e.g. Kurota"
                  className="studio-input w-full h-8 px-2.5 text-xs"
                />
              </div>
            </div>

            <div className="flex items-center justify-between text-[11px]">
              <div className="flex items-center gap-3">
                <label className="inline-flex items-center gap-1.5 text-studio-text-secondary cursor-pointer">
                  <input
                    type="checkbox"
                    checked={findCaseSensitive}
                    onChange={(e) => setFindCaseSensitive(e.target.checked)}
                    className="accent-[var(--color-accent)]"
                  />
                  Case sensitive
                </label>
                <label className="inline-flex items-center gap-1.5 text-studio-text-secondary cursor-pointer">
                  <input
                    type="checkbox"
                    checked={findWholeWord}
                    onChange={(e) => setFindWholeWord(e.target.checked)}
                    className="accent-[var(--color-accent)]"
                  />
                  Whole word
                </label>
              </div>
              <span className="font-mono text-[10px] text-studio-text-muted tabular-nums">
                {findText
                  ? `${findMatchStats.matches} match${findMatchStats.matches === 1 ? '' : 'es'} / ${findMatchStats.pagesCount}p`
                  : 'Enter search text'}
              </span>
            </div>

            {findReplaceStatus && (
              <p className="text-[11px] font-medium text-studio-success">
                {findReplaceStatus}
              </p>
            )}

            <div className="flex items-center justify-end gap-2 pt-1">
              <button
                type="button"
                onClick={() => setIsFindReplaceOpen(false)}
                className="studio-btn h-7 px-3 text-xs"
              >
                Close
              </button>
              <button
                type="button"
                disabled={!findText || findMatchStats.matches === 0}
                onClick={() => {
                  const res = replaceAllTranslations(findText, replaceText, {
                    caseSensitive: findCaseSensitive,
                    wholeWord: findWholeWord,
                  });
                  setFindReplaceStatus(
                    `Replaced ${res.replacedBlocks} block${res.replacedBlocks === 1 ? '' : 's'} across ${res.affectedPages} page${res.affectedPages === 1 ? '' : 's'} (Ctrl+Z to undo).`
                  );
                }}
                className="h-7 px-3 rounded-studio-sm bg-studio-accent text-studio-accent-fg text-xs font-medium hover:opacity-90 disabled:opacity-40 cursor-pointer"
              >
                Replace All
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Lazy-Loaded Multi-Manga Editable & Overnight Batch Series Queue Modal */}
      {isBatchQueueOpen && (
        <ModalErrorBoundary
          title="Batch Series Queue Error"
          onClose={() => setIsBatchQueueOpen(false)}
        >
          <Suspense fallback={null}>
            <BatchQueueModal
              isOpen={isBatchQueueOpen}
              onClose={() => setIsBatchQueueOpen(false)}
              apiBaseUrl={apiBaseUrl}
              queueItems={queueItems}
              setQueueItems={setQueueItems}
              runMode={queueRunMode}
              setRunMode={setQueueRunMode}
              exportFormat={queueExportFormat}
              setExportFormat={setQueueExportFormat}
              outputDir={queueOutputDir}
              setOutputDir={setQueueOutputDir}
              filenamePattern={queueFilenamePattern}
              setFilenamePattern={setQueueFilenamePattern}
              skipExisting={queueSkipExisting}
              setSkipExisting={setQueueSkipExisting}
              purgeCacheBetween={queuePurgeCache}
              setPurgeCacheBetween={setQueuePurgeCache}
              shareSeriesContinuity={queueShareContinuity}
              setShareSeriesContinuity={setQueueShareContinuity}
              activeProjectId={projectCacheIdRef.current}
              isQueueRunning={isQueueRunning}
              onStartQueue={handleStartBatchQueue}
              onStopQueue={handleStopBatchQueue}
              onEditItem={handleOpenQueueItemInEditor}
              onBatchExportSelected={handleBatchExportQueueItems}
              onClearAndPurgeCaches={handleClearQueueAndPurgeCaches}
            />
          </Suspense>
        </ModalErrorBoundary>
      )}

      {/* 5-Tab Studio Settings Modal (lazy-loaded when open) */}
      {activeModal === 'settings' && (
        <ModalErrorBoundary title="Settings Modal Error" onClose={closeModal}>
          <Suspense fallback={null}>
            <SettingsModal
              isOpen={activeModal === 'settings'}
              onClose={closeModal}
              onSaved={() => {
                hydrateBackendContext();
                fetchLlmAndHealthStatus();
              }}
              onApplyTypesettingToPage={handleApplyStyleToAll}
            />
          </Suspense>
        </ModalErrorBoundary>
      )}
    </div>
  );
}
