import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  ArrowDown,
  ArrowUp,
  Check,
  CheckCircle2,
  Cpu,
  Download,
  Eraser,
  Eye,
  EyeOff,
  FileText,
  Globe,
  HardDrive,
  Image as ImageIcon,
  Languages,
  Layers,
  Loader2,
  Monitor,
  Moon,
  Palette,
  Plus,
  RefreshCw,
  RotateCcw,
  Save,
  Search,
  Server,
  Sliders,
  Sparkles,
  Sun,
  Trash2,
  Type,
  Upload,
  Wifi,
  X,
} from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import useEditorStore, {
  DEFAULT_API_BASE_URL,
  DEFAULT_DETECTION_MODELS,
  DEFAULT_FONT_FALLBACK_CHAIN,
  DEFAULT_INPAINTING_MODELS,
  DEFAULT_OCR_MODELS,
  DEFAULT_PROVIDER_PRESETS,
  DEFAULT_TYPESETTING_CONFIG,
  buildDefaultProviderProfiles,
  ensureFontFaceLoaded,
  formatFontFamilyChain,
  normalizeOptionalThreshold,
} from '../../store/useEditorStore.js';
import { formatExecutionProvider } from '../StatusBar.jsx';

import {
  TABS,
  THEME_CARDS,
  ACCENT_PRESETS,
  DENSITY_OPTIONS,
  LAMA_TILE_SIZES,
  READING_ORDER_OPTIONS,
  BACKEND_OPTIONS,
  REASONING_LEVELS,
  SOURCE_LANGUAGE_OPTIONS,
  INSTRUCTION_PRESETS,
  FONT_STYLE_FILTERS,
  ToggleSwitch,
  SectionCard,
  mergePipelineModelOptions,
  ModelStatusBadge,
  normalizeExecMode,
  StageHardwareBadge,
  glossaryDictToRows,
  glossaryRowsToDict,
} from './SettingsPrimitives.jsx';

export function useSettingsController({ isOpen, onClose, onSaved }) {
  const {
    theme,
    resolvedTheme,
    accentColor,
    uiDensity,
    settingsActiveTab,
    pipelineConfig,
    typesetting,
    viewLayers,
    availableProviderPresets,
    supportedTargetLanguages,
    providerModelsCache,
    pipelineModelsCatalog,
    fontsCatalogCache,
    runtimeStatus,
    hasActivePage,
    setTheme,
    setAccentColor,
    setUiDensity,
    setSettingsActiveTab,
    updatePipelineConfig,
    updateTypesetting,
    hydratePipelineConfig,
    persistSettingsToBackend,
    setProviderModelsResult,
    setPipelineModelsCatalog,
    setFontsCatalogCache,
    setModelsStatus,
    setHardwareInfo,
    setLmStudioStatus,
    toggleViewLayer,
    resetViewLayers,
  } = useEditorStore(
    useShallow((s) => ({
      theme: s.theme,
      resolvedTheme: s.resolvedTheme,
      accentColor: s.accentColor,
      uiDensity: s.uiDensity,
      settingsActiveTab: s.settingsActiveTab,
      pipelineConfig: s.pipelineConfig,
      typesetting: s.typesetting,
      viewLayers: s.viewLayers,
      availableProviderPresets: s.availableProviderPresets,
      supportedTargetLanguages: s.supportedTargetLanguages,
      providerModelsCache: s.providerModelsCache,
      pipelineModelsCatalog: s.pipelineModelsCatalog,
      fontsCatalogCache: s.fontsCatalogCache,
      runtimeStatus: s.runtimeStatus,
      hasActivePage: Boolean(s.activePageId),
      setTheme: s.setTheme,
      setAccentColor: s.setAccentColor,
      setUiDensity: s.setUiDensity,
      setSettingsActiveTab: s.setSettingsActiveTab,
      updatePipelineConfig: s.updatePipelineConfig,
      updateTypesetting: s.updateTypesetting,
      hydratePipelineConfig: s.hydratePipelineConfig,
      persistSettingsToBackend: s.persistSettingsToBackend,
      setProviderModelsResult: s.setProviderModelsResult,
      setPipelineModelsCatalog: s.setPipelineModelsCatalog,
      setFontsCatalogCache: s.setFontsCatalogCache,
      setModelsStatus: s.setModelsStatus,
      setHardwareInfo: s.setHardwareInfo,
      setLmStudioStatus: s.setLmStudioStatus,
      toggleViewLayer: s.toggleViewLayer,
      resetViewLayers: s.resetViewLayers,
    }))
  );

  const apiBaseUrl = pipelineConfig.apiBaseUrl ?? DEFAULT_API_BASE_URL;
  const activeTab = settingsActiveTab || 'appearance';

  // Local draft state so edits are smooth and saved cleanly to /api/settings
  const [draftPipeline, setDraftPipeline] = useState(() => ({ ...pipelineConfig }));
  const [draftTypesetting, setDraftTypesetting] = useState(() => ({
    ...typesetting,
    fontFallbackChain: [...(typesetting.fontFallbackChain || DEFAULT_FONT_FALLBACK_CHAIN)],
  }));
  const [glossaryRows, setGlossaryRows] = useState(() =>
    glossaryDictToRows(pipelineConfig.glossary)
  );
  const [bulkGlossaryMode, setBulkGlossaryMode] = useState(false);
  const [bulkGlossaryText, setBulkGlossaryText] = useState('');

  // Pipeline Processing models catalog & Koharu import state
  const [pipelineModelsData, setPipelineModelsData] = useState(
    pipelineModelsCatalog
  );
  const [isLoadingPipelineModels, setIsLoadingPipelineModels] = useState(false);
  const [isImportingKoharu, setIsImportingKoharu] = useState(false);
  const [koharuImportFeedback, setKoharuImportFeedback] = useState(null); // { type: 'success' | 'error', text: string }

  // Providers tab state
  const [selectedProviderId, setSelectedProviderId] = useState(
    pipelineConfig.provider || 'lmstudio'
  );
  const [showApiKeys, setShowApiKeys] = useState({});
  const [testingProviderId, setTestingProviderId] = useState(null);
  const [fetchingModelsFor, setFetchingModelsFor] = useState(null);

  // Fonts / Typesetting tab state
  const [fontsData, setFontsData] = useState(fontsCatalogCache);
  const [isLoadingFonts, setIsLoadingFonts] = useState(false);
  const [fontsError, setFontsError] = useState(null);
  const [fontSearch, setFontSearch] = useState('');
  const [fontSourceTab, setFontSourceTab] = useState('all'); // 'all' | 'installed' | 'system' | 'custom' | 'catalog'
  const [fontStyleFilter, setFontStyleFilter] = useState('all');
  const [downloadingFontId, setDownloadingFontId] = useState(null);
  const [uploadingFont, setUploadingFont] = useState(false);
  const [customChainFontInput, setCustomChainFontInput] = useState('');
  const fontUploadInputRef = useRef(null);

  // Modal lifecycle & save feedback
  const [isHydrating, setIsHydrating] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isSwitchingBackend, setIsSwitchingBackend] = useState(false);
  const [requiresBackendRelaunch, setRequiresBackendRelaunch] = useState(false);
  const [isRelaunchingApp, setIsRelaunchingApp] = useState(false);
  const [saveFeedback, setSaveFeedback] = useState(null); // { type: 'success' | 'error', text: string }
  const [settingsPath, setSettingsPath] = useState('');

  const handleSelectPreferredBackend = useCallback(
    async (backendId) => {
      setDraftPipeline((prev) => ({
        ...prev,
        preferredBackend: backendId,
      }));
      setIsSwitchingBackend(true);
      try {
        const resp = await fetch(`${apiBaseUrl}/api/system/backend`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'application/json',
          },
          body: JSON.stringify({
            preferred_backend: backendId,
          }),
        });
        if (resp.ok) {
          const hwData = await resp.json();
          setHardwareInfo(hwData);
          setRequiresBackendRelaunch(Boolean(hwData.requires_relaunch));
          updatePipelineConfig({
            preferredBackend: hwData.preferred_backend ?? backendId,
          });
        }
      } catch {
        // Fallback: draft value will still be saved via Save Settings
      } finally {
        setIsSwitchingBackend(false);
      }
    },
    [apiBaseUrl, setHardwareInfo, updatePipelineConfig]
  );

  const handleRelaunchForBackend = useCallback(async () => {
    setIsRelaunchingApp(true);
    try {
      await fetch(`${apiBaseUrl}/api/system/relaunch`, {
        method: 'POST',
        headers: { Accept: 'application/json' },
      });
    } catch {
      // Process exits immediately to trigger Dual-Core launcher restart
    }
  }, [apiBaseUrl]);

  const providerPresets = useMemo(
    () =>
      Array.isArray(availableProviderPresets) && availableProviderPresets.length > 0
        ? availableProviderPresets
        : DEFAULT_PROVIDER_PRESETS,
    [availableProviderPresets]
  );

  // Fetch pipeline model availability from GET /api/pipeline/models
  const fetchPipelineModels = useCallback(async () => {
    setIsLoadingPipelineModels(true);
    try {
      const resp = await fetch(`${apiBaseUrl}/api/pipeline/models`, {
        headers: { Accept: 'application/json' },
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setPipelineModelsData(data);
      setPipelineModelsCatalog(data);
      return data;
    } catch {
      return null;
    } finally {
      setIsLoadingPipelineModels(false);
    }
  }, [apiBaseUrl, setPipelineModelsCatalog]);

  // Import local Koharu models via POST /api/models/import-koharu and refresh statuses
  const handleImportKoharuModels = useCallback(async () => {
    setIsImportingKoharu(true);
    setKoharuImportFeedback(null);
    try {
      const resp = await fetch(`${apiBaseUrl}/api/models/import-koharu`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify({}),
      });
      const data = await resp.json().catch(() => ({}));

      if (data?.pipeline_models) {
        setPipelineModelsData(data.pipeline_models);
        setPipelineModelsCatalog(data.pipeline_models);
      }

      // Refresh live pipeline model availability & overall model status
      await fetchPipelineModels();
      fetch(`${apiBaseUrl}/api/models/status`, {
        headers: { Accept: 'application/json' },
      })
        .then((r) => (r.ok ? r.json() : null))
        .then((statusJson) => {
          if (statusJson) setModelsStatus(statusJson);
        })
        .catch(() => {});

      if (!resp.ok || data?.ok === false) {
        throw new Error(
          data?.detail ||
            data?.conversion_error ||
            (data?.status === 'cache_not_found'
              ? 'Local model cache directory was not found.'
              : `Local model import failed (HTTP ${resp.status}).`)
        );
      }

      const resultEntries = Object.values(data?.results || {});
      const newlyImported = resultEntries.filter(
        (r) => r?.status === 'imported' || r?.status === 'converted'
      ).length;
      const alreadyReady = resultEntries.filter(
        (r) => r?.status === 'already_installed'
      ).length;

      setKoharuImportFeedback({
        type: 'success',
        text:
          newlyImported > 0
            ? `Imported ${newlyImported} local model${newlyImported === 1 ? '' : 's'} and refreshed availability.`
            : alreadyReady > 0
            ? 'All available local models are already imported.'
            : 'Local model statuses refreshed.',
      });
    } catch (err) {
      setKoharuImportFeedback({
        type: 'error',
        text: err?.message || 'Failed to import local models.',
      });
    } finally {
      setIsImportingKoharu(false);
    }
  }, [apiBaseUrl, fetchPipelineModels, setModelsStatus, setPipelineModelsCatalog]);

  const handleOptionalThresholdChange = useCallback((fieldKey, rawValue) => {
    setDraftPipeline((prev) => ({
      ...prev,
      [fieldKey]: rawValue === '' || rawValue === null ? null : rawValue,
    }));
  }, []);

  const handleOptionalThresholdBlur = useCallback((fieldKey) => {
    setDraftPipeline((prev) => {
      const normalized = normalizeOptionalThreshold(prev[fieldKey]);
      if (prev[fieldKey] === normalized) return prev;
      return {
        ...prev,
        [fieldKey]: normalized,
      };
    });
  }, []);

  // Sync fresh settings from GET /api/settings and model availability from GET /api/pipeline/models when modal opens
  useEffect(() => {
    if (!isOpen) return;
    let cancelled = false;
    setSaveFeedback(null);
    setKoharuImportFeedback(null);

    const syncFromStore = () => {
      const s = useEditorStore.getState();
      setDraftPipeline({
        ...s.pipelineConfig,
        providerProfiles: {
          ...buildDefaultProviderProfiles(),
          ...(s.pipelineConfig.providerProfiles || {}),
        },
      });
      setDraftTypesetting({
        ...s.typesetting,
        fontFallbackChain: [
          ...(s.typesetting.fontFallbackChain || DEFAULT_FONT_FALLBACK_CHAIN),
        ],
      });
      setGlossaryRows(glossaryDictToRows(s.pipelineConfig.glossary));
      setSelectedProviderId(s.pipelineConfig.provider || 'lmstudio');
      if (s.pipelineModelsCatalog) {
        setPipelineModelsData(s.pipelineModelsCatalog);
      }
    };

    syncFromStore();
    setIsHydrating(true);

    Promise.allSettled([
      fetch(`${apiBaseUrl}/api/settings`, {
        headers: { Accept: 'application/json' },
      }).then((res) => (res.ok ? res.json() : null)),
      fetchPipelineModels(),
      fetch(`${apiBaseUrl}/api/models/status`, {
        headers: { Accept: 'application/json' },
      }).then((res) => (res.ok ? res.json() : null)),
      fetch(`${apiBaseUrl}/api/system/hardware`, {
        headers: { Accept: 'application/json' },
      }).then((res) => (res.ok ? res.json() : null)),
    ])
      .then(([settingsRes, modelsRes, statusRes, hwRes]) => {
        if (cancelled) return;
        const settingsData =
          settingsRes.status === 'fulfilled' ? settingsRes.value : null;
        const modelsData =
          modelsRes.status === 'fulfilled' ? modelsRes.value : null;
        const statusData =
          statusRes.status === 'fulfilled' ? statusRes.value : null;
        const hwData =
          hwRes.status === 'fulfilled' ? hwRes.value : null;
        if (statusData) {
          setModelsStatus(statusData);
        }
        if (hwData) {
          setHardwareInfo(hwData);
        }

        if (settingsData) {
          if (settingsData.settings_path) setSettingsPath(settingsData.settings_path);
          hydratePipelineConfig(settingsData);
          syncFromStore();
        } else if (modelsData?.active) {
          hydratePipelineConfig(modelsData.active);
          syncFromStore();
        }
      })
      .finally(() => {
        if (!cancelled) setIsHydrating(false);
      });

    return () => {
      cancelled = true;
    };
  }, [isOpen, apiBaseUrl, hydratePipelineConfig, fetchPipelineModels]);

  // Escape key closes modal
  useEffect(() => {
    if (!isOpen) return undefined;
    const handleKey = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose?.();
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isOpen, onClose]);

  // Fetch /api/fonts/list when entering Typesetting tab
  const fetchFontsList = useCallback(
    async (forceRefresh = false) => {
      setIsLoadingFonts(true);
      setFontsError(null);
      try {
        const url = `${apiBaseUrl}/api/fonts/list?include_catalog=true${
          forceRefresh ? '&refresh=true' : ''
        }`;
        const resp = await fetch(url, { headers: { Accept: 'application/json' } });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        setFontsData(data);
        setFontsCatalogCache(data);

        // Preload custom & downloaded fonts into document.fonts for instant preview
        const localFaces = (data.fonts || []).filter(
          (f) => f.category === 'custom' || f.category === 'downloaded'
        );
        localFaces.slice(0, 20).forEach((f) => {
          ensureFontFaceLoaded(f.family, f.url, apiBaseUrl);
        });
      } catch (err) {
        setFontsError(err?.message || 'Could not load font list from backend.');
      } finally {
        setIsLoadingFonts(false);
      }
    },
    [apiBaseUrl, setFontsCatalogCache]
  );

  useEffect(() => {
    if (isOpen && activeTab === 'typesetting' && !fontsData && !isLoadingFonts) {
      fetchFontsList(false);
    }
  }, [isOpen, activeTab, fontsData, isLoadingFonts, fetchFontsList]);

  // Helper: update a provider profile in draftPipeline
  const updateDraftProfile = useCallback((pid, patch) => {
    setDraftPipeline((prev) => {
      const preset =
        DEFAULT_PROVIDER_PRESETS.find((p) => p.id === pid) ||
        DEFAULT_PROVIDER_PRESETS[0];
      const existing = prev.providerProfiles?.[pid] || {
        base_url: preset.default_base_url,
        api_key: preset.default_api_key || '',
        model: preset.default_model,
      };
      const updated = { ...existing, ...patch };
      const nextProfiles = {
        ...(prev.providerProfiles || {}),
        [pid]: updated,
      };
      const isCurrent = prev.provider === pid;
      return {
        ...prev,
        providerProfiles: nextProfiles,
        ...(isCurrent
          ? {
              baseUrl: updated.base_url,
              apiKey: updated.api_key,
              model: updated.model,
            }
          : {}),
      };
    });
  }, []);

  // Switch active provider (in Tab 3 or Tab 4)
  const selectActiveProvider = useCallback((pid) => {
    setSelectedProviderId(pid);
    setDraftPipeline((prev) => {
      const preset =
        DEFAULT_PROVIDER_PRESETS.find((p) => p.id === pid) ||
        DEFAULT_PROVIDER_PRESETS[0];
      const profile = prev.providerProfiles?.[pid] || {
        base_url: preset.default_base_url,
        api_key: preset.default_api_key || '',
        model: preset.default_model,
      };
      return {
        ...prev,
        provider: pid,
        baseUrl: profile.base_url ?? preset.default_base_url,
        apiKey: profile.api_key ?? preset.default_api_key ?? '',
        model: profile.model ?? preset.default_model,
      };
    });
  }, []);

  // Test Connection / Fetch Models via POST /api/providers/test
  const handleTestOrFetchProvider = useCallback(
    async (pid, mode = 'test') => {
      const preset =
        providerPresets.find((p) => p.id === pid) || DEFAULT_PROVIDER_PRESETS[0];
      const profile = draftPipeline.providerProfiles?.[pid] || {
        base_url: preset.default_base_url,
        api_key: preset.default_api_key || '',
        model: preset.default_model,
      };

      if (mode === 'fetch') {
        setFetchingModelsFor(pid);
      } else {
        setTestingProviderId(pid);
      }

      try {
        const resp = await fetch(`${apiBaseUrl}/api/providers/test`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'application/json',
          },
          body: JSON.stringify({
            provider: pid,
            base_url: profile.base_url || preset.default_base_url,
            api_key: profile.api_key ?? '',
            model: profile.model || preset.default_model,
            timeout: 12.0,
          }),
        });
        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data?.detail || `HTTP ${resp.status}`);
        }

        setProviderModelsResult(pid, data);

        // If LM Studio, sync live runtime status badge too
        if (pid === 'lmstudio') {
          setLmStudioStatus({
            connected: Boolean(data.connected ?? data.ok),
            endpoint: data.endpoint,
            active_model: data.active_model,
            resolved_model_id: data.resolved_model_id,
            available_models: data.available_models || data.models || [],
            error: data.error,
          });
        }

        // If no model was set yet and models were returned, auto-select resolved or first model
        const fetchedList = data.models || data.available_models || [];
        if (data.ok && (!profile.model || mode === 'fetch') && fetchedList.length > 0) {
          const picked =
            (!profile.model && (data.active_model || data.resolved_model_id || fetchedList[0])) ||
            profile.model;
          if (picked && !profile.model) {
            updateDraftProfile(pid, { model: picked });
          }
        }
      } catch (err) {
        setProviderModelsResult(pid, {
          ok: false,
          connected: false,
          models: [],
          vision_models: [],
          error: err?.message || 'Connection failed',
          message: err?.message || 'Connection failed',
        });
      } finally {
        setTestingProviderId(null);
        setFetchingModelsFor(null);
      }
    },
    [
      apiBaseUrl,
      draftPipeline.providerProfiles,
      providerPresets,
      setLmStudioStatus,
      setProviderModelsResult,
      updateDraftProfile,
    ]
  );

  // Download a font from curated Google Fonts catalog
  const handleDownloadCatalogFont = useCallback(
    async (catalogItem) => {
      if (!catalogItem?.id) return;
      setDownloadingFontId(catalogItem.id);
      setFontsError(null);
      try {
        const resp = await fetch(`${apiBaseUrl}/api/fonts/download`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
          body: JSON.stringify({ fonts: [catalogItem.id] }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data?.detail || `HTTP ${resp.status}`);
        const firstRes = data.results?.[0];
        if (firstRes?.status === 'failed') {
          throw new Error(firstRes.error || `Failed to download ${catalogItem.family}`);
        }
        if (firstRes?.url) {
          await ensureFontFaceLoaded(catalogItem.family, firstRes.url, apiBaseUrl);
        }
        await fetchFontsList(true);
        setSaveFeedback({
          type: 'success',
          text: `Downloaded "${catalogItem.family}" for offline lettering.`,
        });
      } catch (err) {
        setFontsError(err?.message || `Download failed for ${catalogItem.family}`);
      } finally {
        setDownloadingFontId(null);
      }
    },
    [apiBaseUrl, fetchFontsList]
  );

  // Upload custom .ttf / .otf / .woff2 / .woff font
  const handleUploadCustomFont = useCallback(
    async (event) => {
      const file = event.target.files?.[0];
      event.target.value = '';
      if (!file) return;
      setUploadingFont(true);
      setFontsError(null);
      try {
        const formData = new FormData();
        formData.append('file', file, file.name);
        const resp = await fetch(`${apiBaseUrl}/api/fonts/upload`, {
          method: 'POST',
          body: formData,
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data?.detail || `HTTP ${resp.status}`);
        if (data?.family && data?.url) {
          await ensureFontFaceLoaded(data.family, data.url, apiBaseUrl);
          // Automatically place newly uploaded font at top of fallback chain
          setDraftTypesetting((prev) => {
            const filtered = (prev.fontFallbackChain || []).filter(
              (f) => f.toLowerCase() !== data.family.toLowerCase()
            );
            const nextChain = [data.family, ...filtered];
            return {
              ...prev,
              fontFallbackChain: nextChain,
              fontFamily: formatFontFamilyChain(nextChain),
            };
          });
        }
        await fetchFontsList(true);
        setSaveFeedback({
          type: 'success',
          text: `Imported "${data.family || file.name}" and added as primary font.`,
        });
      } catch (err) {
        setFontsError(err?.message || 'Failed to upload custom font.');
      } finally {
        setUploadingFont(false);
      }
    },
    [apiBaseUrl, fetchFontsList]
  );

  // Font fallback chain helpers
  const setChainPrimaryFont = useCallback((familyName, fontUrl = null) => {
    const clean = String(familyName || '').trim();
    if (!clean) return;
    if (fontUrl) ensureFontFaceLoaded(clean, fontUrl, apiBaseUrl);
    setDraftTypesetting((prev) => {
      const existing = prev.fontFallbackChain || [];
      const rest = existing.filter((f) => f.toLowerCase() !== clean.toLowerCase());
      const nextChain = [clean, ...rest];
      return {
        ...prev,
        fontFallbackChain: nextChain,
        fontFamily: formatFontFamilyChain(nextChain),
      };
    });
  }, [apiBaseUrl]);

  const appendChainFont = useCallback((familyName, fontUrl = null) => {
    const clean = String(familyName || '').trim();
    if (!clean) return;
    if (fontUrl) ensureFontFaceLoaded(clean, fontUrl, apiBaseUrl);
    setDraftTypesetting((prev) => {
      const existing = prev.fontFallbackChain || [];
      if (existing.some((f) => f.toLowerCase() === clean.toLowerCase())) {
        return prev;
      }
      // Insert right before trailing generic CSS fallback (e.g. sans-serif)
      const generics = new Set(['sans-serif', 'serif', 'monospace', 'cursive', 'system-ui']);
      const lastGenericIdx = existing.findIndex((f) => generics.has(f.toLowerCase()));
      let nextChain;
      if (lastGenericIdx >= 0) {
        nextChain = [
          ...existing.slice(0, lastGenericIdx),
          clean,
          ...existing.slice(lastGenericIdx),
        ];
      } else {
        nextChain = [...existing, clean];
      }
      return {
        ...prev,
        fontFallbackChain: nextChain,
        fontFamily: formatFontFamilyChain(nextChain),
      };
    });
  }, [apiBaseUrl]);

  const moveChainItem = useCallback((index, delta) => {
    setDraftTypesetting((prev) => {
      const list = [...(prev.fontFallbackChain || [])];
      const target = index + delta;
      if (target < 0 || target >= list.length) return prev;
      const [moved] = list.splice(index, 1);
      list.splice(target, 0, moved);
      return {
        ...prev,
        fontFallbackChain: list,
        fontFamily: formatFontFamilyChain(list),
      };
    });
  }, []);

  const removeChainItem = useCallback((index) => {
    setDraftTypesetting((prev) => {
      const list = [...(prev.fontFallbackChain || [])];
      if (list.length <= 1) return prev;
      list.splice(index, 1);
      return {
        ...prev,
        fontFallbackChain: list,
        fontFamily: formatFontFamilyChain(list),
      };
    });
  }, []);

  // Save all settings to Zustand store + localStorage + POST /api/settings
  const handleSaveAll = useCallback(async () => {
    setIsSaving(true);
    setSaveFeedback(null);

    const parsedGlossary = bulkGlossaryMode
      ? bulkGlossaryText
          .split('\n')
          .reduce((acc, line) => {
            const trimmed = line.trim();
            if (!trimmed || trimmed.startsWith('#')) return acc;
            for (const sep of ['->', '=>', '→', '=', ':']) {
              if (trimmed.includes(sep)) {
                const idx = trimmed.indexOf(sep);
                const lhs = trimmed.slice(0, idx).trim();
                const rhs = trimmed.slice(idx + sep.length).trim();
                if (lhs && rhs) acc[lhs] = rhs;
                break;
              }
            }
            return acc;
          }, {})
      : glossaryRowsToDict(glossaryRows);

    const activePid = draftPipeline.provider || 'lmstudio';
    const activeProfile = draftPipeline.providerProfiles?.[activePid] || {};
    const finalChain =
      draftTypesetting.fontFallbackChain?.length > 0
        ? [...draftTypesetting.fontFallbackChain]
        : [...DEFAULT_FONT_FALLBACK_CHAIN];
    const finalFontFamily = formatFontFamilyChain(finalChain);

    const mergedPipeline = {
      ...draftPipeline,
      detectionModel:
        draftPipeline.detectionModel || 'vellum-det-yolo11m-seg-1024-fp16',
      detectorPrecision: ['fast', 'balanced', 'high_precision'].includes(
        draftPipeline.detectorPrecision
      )
        ? draftPipeline.detectorPrecision
        : 'balanced',
      textThreshold: normalizeOptionalThreshold(draftPipeline.textThreshold),
      bubbleThreshold: normalizeOptionalThreshold(draftPipeline.bubbleThreshold),
      panelThreshold: normalizeOptionalThreshold(draftPipeline.panelThreshold),
      whisperThreshold: normalizeOptionalThreshold(draftPipeline.whisperThreshold),
      borderlessThreshold: normalizeOptionalThreshold(
        draftPipeline.borderlessThreshold
      ),
      enableAdaptiveTiling: Boolean(draftPipeline.enableAdaptiveTiling ?? true),
      sfxSuppressIou: Number(draftPipeline.sfxSuppressIou ?? 0.60),
      ocrModel: draftPipeline.ocrModel || 'auto',
      inpaintingModel: draftPipeline.inpaintingModel || 'lama-manga',
      inpaintApiUrl:
        (draftPipeline.inpaintApiUrl || 'http://127.0.0.1:8188').trim() ||
        'http://127.0.0.1:8188',
      baseUrl: activeProfile.base_url ?? draftPipeline.baseUrl,
      apiKey: activeProfile.api_key ?? draftPipeline.apiKey,
      model: activeProfile.model ?? draftPipeline.model,
      glossary: parsedGlossary,
    };

    const mergedTypesetting = {
      ...draftTypesetting,
      fontFallbackChain: finalChain,
      fontFamily: finalFontFamily,
    };

    // Keep draft normalized (e.g. clamped thresholds) and sync Zustand store & localStorage
    setDraftPipeline(mergedPipeline);
    updatePipelineConfig(mergedPipeline);
    updateTypesetting(mergedTypesetting);

    try {
      const saved = await persistSettingsToBackend();
      if (saved?.settings_path) setSettingsPath(saved.settings_path);
      fetchPipelineModels();
      setSaveFeedback({
        type: 'success',
        text: 'Settings saved and synced with backend.',
      });
      onSaved?.(saved);
    } catch (err) {
      setSaveFeedback({
        type: 'error',
        text: `Saved locally, but backend sync failed: ${err?.message || 'Unknown error'}`,
      });
    } finally {
      setIsSaving(false);
    }
  }, [
    bulkGlossaryMode,
    bulkGlossaryText,
    draftPipeline,
    draftTypesetting,
    fetchPipelineModels,
    glossaryRows,
    onSaved,
    persistSettingsToBackend,
    updatePipelineConfig,
    updateTypesetting,
  ]);

  const handleResetActiveTab = useCallback(() => {
    if (activeTab === 'appearance') {
      setTheme('dark');
      setAccentColor('amber');
      setUiDensity('compact');
    } else if (activeTab === 'pipeline') {
      setDraftPipeline((prev) => ({
        ...prev,
        detectionModel: 'vellum-det-yolo11m-seg-1024-fp16',
        detectorPrecision: 'balanced',
        textThreshold: null,
        bubbleThreshold: null,
        panelThreshold: null,
        whisperThreshold: null,
        borderlessThreshold: null,
        enableAdaptiveTiling: true,
        sfxSuppressIou: 0.60,
        ocrModel: 'auto',
        inpaintingModel: 'lama-manga',
        inpaintApiUrl: 'http://127.0.0.1:8188',
        executionMode: 'hybrid',
        preferredBackend: 'auto',
        confThreshold: 0.35,
        nmsThreshold: 0.45,
        maskThreshold: 0.5,
        bboxPadding: 6,
        lamaTileSize: 512,
        readingOrder: 'auto',
        autoUnloadVram: true,
        batchScheduleMode: 'auto',
        refineMask: true,
        autoDeskewScans: false,
        autoMaskFuriganaOcr: true,
        skipBlankPages: true,
        smartSsimExport: true,
      }));
    } else if (activeTab === 'providers') {
      const preset =
        providerPresets.find((p) => p.id === selectedProviderId) ||
        DEFAULT_PROVIDER_PRESETS[0];
      updateDraftProfile(selectedProviderId, {
        base_url: preset.default_base_url,
        api_key: preset.default_api_key || '',
        model: preset.default_model,
      });
    } else if (activeTab === 'translation') {
      setDraftPipeline((prev) => ({
        ...prev,
        sourceLang: 'ja',
        targetLang: 'id',
        translationTone: 'standard',
        honorificsPolicy: 'keep',
        temperature: 0.2,
        topP: 0.95,
        topK: 40,
        maxTokens: 2048,
        frequencyPenalty: 0.0,
        reasoningEffort: 'off',
        stripThinking: true,
        useVision: false,
        visionMaxSide: 1024,
        customInstructions: '',
        glossary: {},
      }));
      setGlossaryRows([]);
      setBulkGlossaryText('');
    } else if (activeTab === 'typesetting') {
      setDraftTypesetting({
        ...DEFAULT_TYPESETTING_CONFIG,
        fontFallbackChain: [...DEFAULT_FONT_FALLBACK_CHAIN],
      });
    }
    setSaveFeedback({
      type: 'success',
      text: `Reset ${activeTab} settings to defaults (click Save Settings to persist).`,
    });
  }, [
    activeTab,
    providerPresets,
    selectedProviderId,
    setAccentColor,
    setTheme,
    setUiDensity,
    updateDraftProfile,
  ]);

  // Derived pipeline processing model lists & availability for Tab 2
  const detectionModelOptions = useMemo(
    () =>
      mergePipelineModelOptions(
        DEFAULT_DETECTION_MODELS,
        pipelineModelsData?.detection
      ),
    [pipelineModelsData]
  );
  const ocrModelOptions = useMemo(
    () => mergePipelineModelOptions(DEFAULT_OCR_MODELS, pipelineModelsData?.ocr),
    [pipelineModelsData]
  );
  const inpaintingModelOptions = useMemo(
    () =>
      mergePipelineModelOptions(
        DEFAULT_INPAINTING_MODELS,
        pipelineModelsData?.inpainting
      ),
    [pipelineModelsData]
  );

  const selectedDetectionModelInfo = useMemo(
    () =>
      detectionModelOptions.find(
        (m) =>
          m.id === (draftPipeline.detectionModel || 'vellum-det-yolo11m-seg-1024-fp16')
      ) || detectionModelOptions[0],
    [detectionModelOptions, draftPipeline.detectionModel]
  );
  const selectedOcrModelInfo = useMemo(
    () =>
      ocrModelOptions.find((m) => m.id === (draftPipeline.ocrModel || 'auto')) ||
      ocrModelOptions[0],
    [ocrModelOptions, draftPipeline.ocrModel]
  );
  const selectedInpaintingModelInfo = useMemo(
    () =>
      inpaintingModelOptions.find(
        (m) => m.id === (draftPipeline.inpaintingModel || 'lama-manga')
      ) || inpaintingModelOptions[0],
    [inpaintingModelOptions, draftPipeline.inpaintingModel]
  );

  const hasNotInstalledModels = useMemo(() => {
    const allModels = [
      ...detectionModelOptions,
      ...ocrModelOptions,
      ...inpaintingModelOptions,
    ];
    return allModels.some(
      (m) => m && (m.installed === false || m.readyForInference === false)
    );
  }, [detectionModelOptions, ocrModelOptions, inpaintingModelOptions]);

  // Derived provider info for Tab 3
  const selectedPreset = useMemo(
    () =>
      providerPresets.find((p) => p.id === selectedProviderId) ||
      DEFAULT_PROVIDER_PRESETS[0],
    [providerPresets, selectedProviderId]
  );
  const selectedProfile = useMemo(() => {
    const existing = draftPipeline.providerProfiles?.[selectedProviderId];
    return {
      base_url: existing?.base_url ?? selectedPreset.default_base_url ?? '',
      api_key: existing?.api_key ?? selectedPreset.default_api_key ?? '',
      model: existing?.model ?? selectedPreset.default_model ?? '',
    };
  }, [draftPipeline.providerProfiles, selectedPreset, selectedProviderId]);
  const selectedProviderTest = providerModelsCache[selectedProviderId] || null;

  // Derived model options for Tab 4 (Translation)
  const translationModelOptions = useMemo(() => {
    const pid = draftPipeline.provider || 'lmstudio';
    const preset =
      providerPresets.find((p) => p.id === pid) || DEFAULT_PROVIDER_PRESETS[0];
    const cached = providerModelsCache[pid];
    const seen = new Set();
    const result = [];
    const add = (m) => {
      if (!m || typeof m !== 'string') return;
      const t = m.trim();
      if (!t || seen.has(t)) return;
      seen.add(t);
      result.push(t);
    };
    add(draftPipeline.model);
    add(draftPipeline.providerProfiles?.[pid]?.model);
    if (cached?.activeModel) add(cached.activeModel);
    if (cached?.resolvedModelId) add(cached.resolvedModelId);
    (cached?.models || []).forEach(add);
    if (pid === 'lmstudio') {
      add(runtimeStatus.lmStudio?.activeModel);
      add(runtimeStatus.lmStudio?.resolvedModelId);
      (runtimeStatus.lmStudio?.availableModels || []).forEach(add);
    }
    (preset.fallback_models || []).forEach(add);
    if (preset.default_model) add(preset.default_model);
    return result;
  }, [
    draftPipeline.model,
    draftPipeline.provider,
    draftPipeline.providerProfiles,
    providerModelsCache,
    providerPresets,
    runtimeStatus.lmStudio,
  ]);

  const normalizeFontQuery = (val) =>
    String(val || '').toLowerCase().replace(/[\s\-_]+/g, '');

  // Derived filtered fonts for Tab 5 (Typesetting)
  const { filteredInstalledFamilies, filteredCatalogItems } = useMemo(() => {
    const q = normalizeFontQuery(fontSearch.trim());
    const families = fontsData?.families || [];
    const catalog = fontsData?.catalog || [];

    const installed = families.filter((fam) => {
      if (fontSourceTab === 'catalog') return false;
      if (fontSourceTab === 'system' && fam.category !== 'system') return false;
      if (
        fontSourceTab === 'custom' &&
        fam.category !== 'custom' &&
        fam.category !== 'downloaded'
      ) {
        return false;
      }
      if (q && !normalizeFontQuery(fam.family).includes(q)) return false;
      return true;
    });

    const catItems = catalog.filter((item) => {
      if (
        fontSourceTab === 'installed' ||
        fontSourceTab === 'system' ||
        fontSourceTab === 'custom'
      ) {
        return false;
      }
      if (fontStyleFilter !== 'all' && item.style_category !== fontStyleFilter) {
        return false;
      }
      if (
        q &&
        !normalizeFontQuery(item.family).includes(q) &&
        !normalizeFontQuery(item.description).includes(q) &&
        !(item.scripts || []).some((sc) => normalizeFontQuery(sc).includes(q))
      ) {
        return false;
      }
      return true;
    });

    return {
      filteredInstalledFamilies: installed,
      filteredCatalogItems: catItems,
    };
  }, [fontsData, fontSearch, fontSourceTab, fontStyleFilter]);

  const installedFontSet = useMemo(() => {
    const families = fontsData?.families;
    if (!Array.isArray(families) || families.length === 0) return null;
    return new Set(families.map((f) => normalizeFontQuery(f.family)));
  }, [fontsData]);


  return {
    theme,
    resolvedTheme,
    accentColor,
    uiDensity,
    settingsActiveTab,
    pipelineConfig,
    typesetting,
    viewLayers,
    availableProviderPresets,
    supportedTargetLanguages,
    providerModelsCache,
    pipelineModelsCatalog,
    fontsCatalogCache,
    runtimeStatus,
    hasActivePage,
    setTheme,
    setAccentColor,
    setUiDensity,
    setSettingsActiveTab,
    updatePipelineConfig,
    updateTypesetting,
    hydratePipelineConfig,
    persistSettingsToBackend,
    setProviderModelsResult,
    setPipelineModelsCatalog,
    setFontsCatalogCache,
    setModelsStatus,
    setHardwareInfo,
    setLmStudioStatus,
    toggleViewLayer,
    resetViewLayers,
    apiBaseUrl,
    activeTab,
    draftPipeline,
    setDraftPipeline,
    draftTypesetting,
    setDraftTypesetting,
    glossaryRows,
    setGlossaryRows,
    bulkGlossaryMode,
    setBulkGlossaryMode,
    bulkGlossaryText,
    setBulkGlossaryText,
    pipelineModelsData,
    setPipelineModelsData,
    isLoadingPipelineModels,
    setIsLoadingPipelineModels,
    isImportingKoharu,
    setIsImportingKoharu,
    koharuImportFeedback,
    setKoharuImportFeedback,
    selectedProviderId,
    setSelectedProviderId,
    showApiKeys,
    setShowApiKeys,
    testingProviderId,
    setTestingProviderId,
    fetchingModelsFor,
    setFetchingModelsFor,
    fontsData,
    setFontsData,
    isLoadingFonts,
    setIsLoadingFonts,
    fontsError,
    setFontsError,
    fontSearch,
    setFontSearch,
    fontSourceTab,
    setFontSourceTab,
    fontStyleFilter,
    setFontStyleFilter,
    downloadingFontId,
    setDownloadingFontId,
    uploadingFont,
    setUploadingFont,
    customChainFontInput,
    setCustomChainFontInput,
    fontUploadInputRef,
    isHydrating,
    setIsHydrating,
    isSaving,
    setIsSaving,
    isSwitchingBackend,
    setIsSwitchingBackend,
    requiresBackendRelaunch,
    setRequiresBackendRelaunch,
    isRelaunchingApp,
    setIsRelaunchingApp,
    saveFeedback,
    setSaveFeedback,
    settingsPath,
    setSettingsPath,
    handleSelectPreferredBackend,
    handleRelaunchForBackend,
    providerPresets,
    fetchPipelineModels,
    handleImportKoharuModels,
    handleOptionalThresholdChange,
    handleOptionalThresholdBlur,
    fetchFontsList,
    updateDraftProfile,
    selectActiveProvider,
    handleTestOrFetchProvider,
    handleDownloadCatalogFont,
    handleUploadCustomFont,
    setChainPrimaryFont,
    appendChainFont,
    moveChainItem,
    removeChainItem,
    handleSaveAll,
    handleResetActiveTab,
    detectionModelOptions,
    ocrModelOptions,
    inpaintingModelOptions,
    selectedDetectionModelInfo,
    selectedOcrModelInfo,
    selectedInpaintingModelInfo,
    hasNotInstalledModels,
    selectedPreset,
    selectedProfile,
    selectedProviderTest,
    translationModelOptions,
    normalizeFontQuery,
    filteredInstalledFamilies,
    filteredCatalogItems,
    installedFontSet,
    isOpen,
    onClose,
    onSaved,
  };
}
