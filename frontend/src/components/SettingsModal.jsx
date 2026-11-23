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
} from '../store/useEditorStore.js';
import { formatExecutionProvider } from './StatusBar.jsx';

const TABS = [
  { id: 'appearance', label: 'Appearance', icon: Palette, desc: 'Theme, accent & density' },
  { id: 'pipeline', label: 'Pipeline', icon: Sliders, desc: 'GPU/CPU, detect & LaMa' },
  { id: 'providers', label: 'Providers', icon: Server, desc: 'LLM endpoints & keys' },
  { id: 'translation', label: 'Translation', icon: Languages, desc: 'Model, sampling & glossary' },
  { id: 'typesetting', label: 'Typesetting', icon: Type, desc: 'Fonts, fallback & style' },
];

const THEME_CARDS = [
  {
    id: 'dark',
    label: 'Dark',
    subtitle: 'Sumi Graphite',
    icon: Moon,
    canvas: '#111214',
    panel: '#181A1D',
    border: '#2A2D32',
    text: '#ECEEF2',
  },
  {
    id: 'light',
    label: 'Light',
    subtitle: 'Manga Paper',
    icon: Sun,
    canvas: '#E5E7EB',
    panel: '#F7F7F5',
    border: '#D4D6DC',
    text: '#16181D',
  },
  {
    id: 'system',
    label: 'Same as System',
    subtitle: 'OS prefers-color-scheme',
    icon: Monitor,
    canvas: 'linear-gradient(135deg, #111214 50%, #E5E7EB 50%)',
    panel: 'linear-gradient(135deg, #181A1D 50%, #F7F7F5 50%)',
    border: '#6E7480',
    text: '#ECEEF2',
  },
];

const ACCENT_PRESETS = [
  { id: 'amber', label: 'Amber', darkHex: '#E59A3C', lightHex: '#C97A16', desc: 'Warm studio amber' },
  { id: 'teal', label: 'Teal', darkHex: '#14B8A6', lightHex: '#0D9488', desc: 'Crisp cyan-teal' },
  { id: 'crimson', label: 'Crimson', darkHex: '#E11D48', lightHex: '#BE123C', desc: 'Bold manga crimson' },
  { id: 'slate', label: 'Slate', darkHex: '#5B9BD5', lightHex: '#3B76B0', desc: 'Cool slate blue' },
];

const DENSITY_OPTIONS = [
  {
    id: 'compact',
    label: 'Compact',
    desc: 'High-density native desktop toolbars & panels',
  },
  {
    id: 'comfortable',
    label: 'Comfortable',
    desc: 'Relaxed spacing with taller touch-friendly controls',
  },
];

const LAMA_TILE_SIZES = [256, 384, 512, 768, 1024];

const READING_ORDER_OPTIONS = [
  { id: 'auto', label: 'Auto', desc: 'JA → RTL · KO/ZH → LTR' },
  { id: 'rtl', label: 'RTL', desc: 'Right-to-Left (Manga)' },
  { id: 'ltr', label: 'LTR', desc: 'Left-to-Right (Manhwa/Manhua)' },
];

const REASONING_LEVELS = [
  { id: 'off', label: 'Off' },
  { id: 'low', label: 'Low' },
  { id: 'medium', label: 'Medium' },
  { id: 'high', label: 'High' },
];

const SOURCE_LANGUAGE_OPTIONS = [
  { code: 'ja', label: 'Japanese (Manga · JA)' },
  { code: 'ko', label: 'Korean (Manhwa · KO)' },
  { code: 'zh', label: 'Chinese (Manhua · ZH)' },
];

const INSTRUCTION_PRESETS = [
  {
    label: 'Natural Comic Dialogue',
    text: 'Keep dialogue punchy, conversational, and concise so it fits naturally inside comic speech bubbles.',
  },
  {
    label: 'Shounen Action',
    text: 'Match high-energy shounen battle tone. Use sharp exclamations and preserve character intensity.',
  },
  {
    label: 'Preserve Honorifics',
    text: 'Preserve original honorifics (-san, -senpai, -kun, -chan, oppa, hyung, shixiong) faithfully.',
  },
  {
    label: 'Adult / Doujinshi (Uncensored)',
    text: 'Use natural, expressive, uncensored colloquial language (including intimate dialogue, moans, and SFX) instead of stiff textbook phrasing.',
  },
];

const FONT_STYLE_FILTERS = [
  { id: 'all', label: 'All Styles' },
  { id: 'comic', label: 'Comic' },
  { id: 'sfx', label: 'SFX / Action' },
  { id: 'handwriting', label: 'Handwriting' },
  { id: 'webtoon', label: 'Webtoon' },
  { id: 'cjk_ja', label: 'JP' },
  { id: 'cjk_ko', label: 'KR' },
  { id: 'cjk_zh', label: 'ZH' },
];

function ToggleSwitch({ checked, onChange, disabled = false, label, id }) {
  return (
    <button
      id={id}
      type="button"
      role="switch"
      aria-checked={Boolean(checked)}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full border transition-colors cursor-pointer disabled:opacity-40 ${
        checked
          ? 'bg-studio-accent border-studio-accent'
          : 'bg-studio-canvas border-studio-border-strong'
      }`}
    >
      <span
        className={`inline-block h-3.5 w-3.5 transform rounded-full transition-transform ${
          checked
            ? 'translate-x-4 bg-studio-accent-fg'
            : 'translate-x-0.5 bg-studio-text-secondary'
        }`}
      />
    </button>
  );
}

function SectionCard({ title, subtitle, action, children }) {
  return (
    <section className="rounded-studio border border-studio-border bg-studio-surface p-3.5 space-y-3">
      {(title || action) && (
        <div className="flex items-start justify-between gap-2">
          <div>
            {title && (
              <h3 className="text-xs font-semibold text-studio-text leading-tight">
                {title}
              </h3>
            )}
            {subtitle && (
              <p className="text-[11px] text-studio-text-muted mt-0.5 leading-snug">
                {subtitle}
              </p>
            )}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

function mergePipelineModelOptions(defaults, apiEntries) {
  const apiMap = new Map();
  if (Array.isArray(apiEntries)) {
    for (const entry of apiEntries) {
      if (entry && typeof entry.id === 'string') {
        apiMap.set(entry.id, entry);
      }
    }
  }

  const seen = new Set();
  const merged = (defaults || []).map((item) => {
    seen.add(item.id);
    const live = apiMap.get(item.id);
    const installed =
      typeof live?.installed === 'boolean' ? live.installed : null;
    const readyForInference =
      typeof live?.ready_for_inference === 'boolean'
        ? live.ready_for_inference
        : installed;
    return {
      id: item.id,
      label: item.label || live?.name || item.name || item.id,
      hardwareTag: item.hardwareTag || live?.hardware_target || null,
      description: live?.description || item.description || '',
      installed,
      readyForInference,
      needsConversion: Boolean(live?.needs_conversion),
      koharuCached: Boolean(live?.koharu_cached),
      source: live?.source || null,
      sizeMb: typeof live?.size_mb === 'number' ? live.size_mb : null,
      defaultThresholds: live?.default_thresholds || null,
      lmStudioHint: live?.lm_studio_hint || null,
    };
  });

  if (Array.isArray(apiEntries)) {
    for (const live of apiEntries) {
      if (!live || typeof live.id !== 'string' || seen.has(live.id)) continue;
      seen.add(live.id);
      const installed =
        typeof live.installed === 'boolean' ? live.installed : null;
      const readyForInference =
        typeof live.ready_for_inference === 'boolean'
          ? live.ready_for_inference
          : installed;
      merged.push({
        id: live.id,
        label: live.name || live.id,
        hardwareTag: live.hardware_target || null,
        description: live.description || '',
        installed,
        readyForInference,
        needsConversion: Boolean(live.needs_conversion),
        koharuCached: Boolean(live.koharu_cached),
        source: live.source || null,
        sizeMb: typeof live.size_mb === 'number' ? live.size_mb : null,
        defaultThresholds: live.default_thresholds || null,
        lmStudioHint: live.lm_studio_hint || null,
      });
    }
  }

  return merged;
}

function ModelStatusBadge({ model, isLoading, testId }) {
  if (isLoading && (!model || model.installed === null)) {
    return (
      <span
        data-testid={testId}
        className="inline-flex items-center gap-1 rounded-studio-sm border border-studio-border bg-studio-surface px-2 py-0.5 text-[10px] font-mono text-studio-text-muted shrink-0"
      >
        <Loader2 className="h-2.5 w-2.5 animate-spin" />
        <span>Checking</span>
      </span>
    );
  }

  if (!model || model.installed === null) {
    return null;
  }

  const isReady = Boolean(model.installed && model.readyForInference !== false);
  if (isReady) {
    const details = [
      model.source ? model.source : 'local',
      model.sizeMb ? `${model.sizeMb} MB` : null,
    ]
      .filter(Boolean)
      .join(' · ');
    return (
      <span
        data-testid={testId}
        title={`Ready (${details})`}
        className="inline-flex items-center gap-1.5 rounded-studio-sm border border-studio-success/40 bg-studio-success/10 px-2 py-0.5 text-[10px] font-mono font-medium text-studio-success shrink-0"
      >
        <span className="h-1.5 w-1.5 rounded-full bg-studio-success" />
        <span>Ready</span>
      </span>
    );
  }

  const missingTip = model.needsConversion
    ? 'Cached in Koharu as .safetensors - click "Import Local Koharu Models" to convert to ONNX.'
    : 'Model weights are not installed locally.';

  return (
    <span
      data-testid={testId}
      title={missingTip}
      className="inline-flex items-center gap-1.5 rounded-studio-sm border border-studio-warning/40 bg-studio-warning/10 px-2 py-0.5 text-[10px] font-mono font-medium text-studio-warning shrink-0"
    >
      <span className="h-1.5 w-1.5 rounded-full bg-studio-warning" />
      <span>Missing .onnx</span>
    </span>
  );
}

function normalizeExecMode(mode) {
  const m = String(mode || 'hybrid').toLowerCase();
  if (m === 'cpu' || m === 'cpu_only') return 'cpu_only';
  if (m === 'gpu' || m === 'force_gpu') return 'force_gpu';
  return 'hybrid';
}

function StageHardwareBadge({
  stage,
  draftMode,
  selectedModel,
  modelInfo,
  runtimeStatus,
  testId,
}) {
  const mode = normalizeExecMode(draftMode);
  const liveProvider =
    stage === 'detect'
      ? runtimeStatus?.detectorProvider
      : stage === 'ocr'
      ? runtimeStatus?.ocrProvider
      : runtimeStatus?.inpainterProvider;

  let isGpu = false;
  let label = 'CPU (Fast AVX2)';

  const modelId = String(selectedModel || modelInfo?.id || '').toLowerCase();

  if (mode === 'cpu_only') {
    isGpu = false;
    label = 'CPU (Fast AVX2)';
  } else if (stage === 'ocr') {
    if (modelId === 'paddleocr-vl-1.6' || modelId === 'paddleocr-vl-gguf') {
      isGpu = true;
      label = 'VLM • Multi-Lang';
    } else if (modelId === 'manga-ocr') {
      isGpu = mode === 'force_gpu';
      label = mode === 'force_gpu' ? 'GPU • Japanese' : 'CPU • Japanese';
    } else if (modelId === 'baberu-ocr') {
      isGpu = mode === 'force_gpu';
      label = mode === 'force_gpu' ? 'GPU • JA/ZH/EN Fast' : 'CPU • JA/ZH/EN Fast';
    } else if (modelId === 'hayai-ocr') {
      isGpu = true;
      label = 'GPU/CPU • JA/ZH/KO/EN';
    } else if (mode === 'force_gpu') {
      isGpu = true;
      label =
        liveProvider === 'CUDAExecutionProvider'
          ? 'GPU (RTX CUDA)'
          : 'GPU (RTX/DirectML)';
    } else {
      isGpu = false;
      label = modelInfo?.hardwareTag || 'CPU (Fast AVX2)';
    }
  } else if (stage === 'inpaint') {
    isGpu = true;
    if (modelId === 'lama-manga') {
      label = 'GPU • Screentone';
    } else if (modelId === 'aot-inpainting') {
      label = 'GPU • Ultra-Fast';
    } else if (modelId === 'flux-2-klein') {
      label = 'Generative • 8GB+ VRAM';
    } else if (modelId === 'rorem-mixed') {
      label = 'Generative • Manga GGUF';
    } else {
      label =
        liveProvider === 'CUDAExecutionProvider'
          ? 'GPU (RTX CUDA)'
          : 'GPU (RTX/DirectML)';
    }
  } else {
    isGpu = true;
    label =
      liveProvider === 'CUDAExecutionProvider'
        ? 'GPU (RTX CUDA)'
        : 'GPU (RTX/DirectML)';
  }

  return (
    <span
      data-testid={testId}
      title={`Live execution target: ${liveProvider || label}`}
      className={`mt-1 inline-flex items-center gap-1 rounded-studio-sm border px-1.5 py-0.5 text-[10px] font-mono font-medium ${
        isGpu
          ? 'border-studio-accent/40 bg-studio-accent/10 text-studio-accent'
          : 'border-studio-border-strong bg-studio-canvas text-studio-text-secondary'
      }`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          isGpu ? 'bg-studio-accent' : 'bg-studio-text-muted'
        }`}
      />
      <span>{label}</span>
    </span>
  );
}

function glossaryDictToRows(glossaryObj) {
  if (!glossaryObj || typeof glossaryObj !== 'object') return [];
  return Object.entries(glossaryObj).map(([source, target], idx) => ({
    id: `g_${idx}_${source}`,
    source: String(source),
    target: String(target),
  }));
}

function glossaryRowsToDict(rows) {
  const dict = {};
  for (const row of rows || []) {
    const s = String(row?.source || '').trim();
    const t = String(row?.target || '').trim();
    if (s && t) {
      dict[s] = t;
    }
  }
  return dict;
}

/**
 * Koharu-style 5-Tab Settings Modal synced with `/api/settings` and `useEditorStore`.
 *
 * Tabs:
 * 1. Appearance  - Dark / Light / System theme, Accent presets, UI density
 * 2. Pipeline    - Auto GPU/CPU vs CPU Only, detection threshold, bbox padding, LaMa tile size, reading order, auto-unload VRAM
 * 3. Providers   - 8 Koharu-style providers (LM Studio :1235, Ollama, OpenAI, Gemini, Anthropic, DeepSeek, OpenRouter, Custom), Base URL, masked API key toggle, Test Connection, Fetch Models
 * 4. Translation - Provider & Model, Target Language++ (16 langs), Sampling sliders, Reasoning + strip <think>, Vision toggle, Instruction Bar, Glossary table
 * 5. Typesetting - Searchable Font Picker (/api/fonts/list + catalog downloader + Import .ttf/.otf), ordered Font Fallback Chain, auto-fit, default size, stroke, line height, ALL-CAPS
 */
export default function SettingsModal({
  isOpen,
  onClose,
  onSaved,
  onApplyTypesettingToPage,
}) {
  const {
    theme,
    resolvedTheme,
    accentColor,
    uiDensity,
    settingsActiveTab,
    pipelineConfig,
    typesetting,
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
    setLmStudioStatus,
  } = useEditorStore(
    useShallow((s) => ({
      theme: s.theme,
      resolvedTheme: s.resolvedTheme,
      accentColor: s.accentColor,
      uiDensity: s.uiDensity,
      settingsActiveTab: s.settingsActiveTab,
      pipelineConfig: s.pipelineConfig,
      typesetting: s.typesetting,
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
      setLmStudioStatus: s.setLmStudioStatus,
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
  const [saveFeedback, setSaveFeedback] = useState(null); // { type: 'success' | 'error', text: string }
  const [settingsPath, setSettingsPath] = useState('');

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
              ? 'Local Koharu cache directory was not found.'
              : `Koharu model import failed (HTTP ${resp.status}).`)
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
            ? `Imported ${newlyImported} local Koharu model${newlyImported === 1 ? '' : 's'} and refreshed availability.`
            : alreadyReady > 0
            ? 'All available local Koharu models are already imported.'
            : 'Local Koharu model statuses refreshed.',
      });
    } catch (err) {
      setKoharuImportFeedback({
        type: 'error',
        text: err?.message || 'Failed to import local Koharu models.',
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
    ])
      .then(([settingsRes, modelsRes, statusRes]) => {
        if (cancelled) return;
        const settingsData =
          settingsRes.status === 'fulfilled' ? settingsRes.value : null;
        const modelsData =
          modelsRes.status === 'fulfilled' ? modelsRes.value : null;
        const statusData =
          statusRes.status === 'fulfilled' ? statusRes.value : null;
        if (statusData) {
          setModelsStatus(statusData);
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
        confThreshold: 0.35,
        nmsThreshold: 0.45,
        maskThreshold: 0.5,
        bboxPadding: 6,
        lamaTileSize: 512,
        readingOrder: 'auto',
        autoUnloadVram: true,
        refineMask: true,
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

  if (!isOpen) return null;

  const hwProviderMeta = formatExecutionProvider(runtimeStatus.executionProvider);
  const currentTabMeta = TABS.find((t) => t.id === activeTab) || TABS[0];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-3 sm:p-5 select-none"
      role="dialog"
      aria-modal="true"
      aria-labelledby="settings-modal-title"
      data-testid="settings-modal"
      onClick={onClose}
    >
      <div
        className="w-full max-w-5xl h-[86vh] max-h-[760px] rounded-studio-md border border-studio-border bg-studio-panel shadow-2xl flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Top Header */}
        <header className="h-11 px-4 border-b border-studio-border bg-studio-panel flex items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-2.5 min-w-0">
            <span className="h-6 w-6 rounded-studio bg-studio-accent text-studio-accent-fg flex items-center justify-center shrink-0">
              <Sliders className="h-3.5 w-3.5" />
            </span>
            <div className="min-w-0">
              <h2
                id="settings-modal-title"
                className="text-xs font-semibold uppercase tracking-wider text-studio-text leading-none"
              >
                Studio Settings
              </h2>
              <p className="text-[10px] font-mono text-studio-text-muted truncate mt-0.5">
                {currentTabMeta.label} · {currentTabMeta.desc}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {isHydrating && (
              <span className="inline-flex items-center gap-1 text-[10px] font-mono text-studio-text-muted">
                <Loader2 className="h-3 w-3 animate-spin" />
                Syncing…
              </span>
            )}
            <button
              type="button"
              onClick={onClose}
              aria-label="Close Settings"
              className="h-7 w-7 inline-flex items-center justify-center rounded-studio border border-transparent hover:border-studio-border text-studio-text-muted hover:text-studio-text hover:bg-studio-hover transition-colors cursor-pointer"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </header>

        {/* Body: Koharu-style Left Vertical Tab Rail + Right Scrollable Content */}
        <div className="flex-1 min-h-0 flex flex-col md:flex-row overflow-hidden">
          {/* Left Navigation Rail */}
          <nav
            aria-label="Settings Categories"
            className="md:w-52 shrink-0 border-b md:border-b-0 md:border-r border-studio-border bg-studio-canvas/50 p-2 flex md:flex-col gap-1 overflow-x-auto md:overflow-y-auto"
          >
            {TABS.map((tab, idx) => {
              const Icon = tab.icon;
              const isSelected = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  type="button"
                  role="tab"
                  aria-selected={isSelected}
                  data-testid={`settings-tab-${tab.id}`}
                  onClick={() => setSettingsActiveTab(tab.id)}
                  className={`relative flex items-center gap-2.5 px-2.5 py-2 rounded-studio text-left transition-colors cursor-pointer shrink-0 md:w-full ${
                    isSelected
                      ? 'bg-studio-surface text-studio-text border border-studio-accent/50 shadow-sm'
                      : 'text-studio-text-secondary border border-transparent hover:bg-studio-hover hover:text-studio-text'
                  }`}
                >
                  {isSelected && (
                    <span
                      className="hidden md:block absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-r bg-studio-accent"
                      aria-hidden="true"
                    />
                  )}
                  <Icon
                    className={`h-4 w-4 shrink-0 ${
                      isSelected ? 'text-studio-accent' : 'text-studio-text-muted'
                    }`}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-1">
                      <span className="text-xs font-medium truncate">
                        {tab.label}
                      </span>
                      <span className="hidden md:inline font-mono text-[9px] text-studio-text-muted">
                        0{idx + 1}
                      </span>
                    </div>
                    <div className="hidden md:block text-[10px] text-studio-text-muted truncate">
                      {tab.desc}
                    </div>
                  </div>
                </button>
              );
            })}

            {/* Live Status Summary Card at bottom of sidebar */}
            <div className="hidden md:block mt-auto pt-3">
              <div className="rounded-studio border border-studio-border bg-studio-surface p-2.5 space-y-1.5 text-[10px] font-mono">
                <div className="text-studio-text-muted uppercase tracking-wider font-semibold">
                  Active Runtime
                </div>
                <div className="flex items-center justify-between gap-1">
                  <span className="text-studio-text-muted">Engine</span>
                  <span className="text-studio-text truncate">
                    {normalizeExecMode(draftPipeline.executionMode) === 'cpu_only'
                      ? 'CPU Only'
                      : normalizeExecMode(draftPipeline.executionMode) === 'force_gpu'
                      ? `GPU (${hwProviderMeta.short})`
                      : 'Hybrid GPU+CPU'}
                  </span>
                </div>
                <div className="flex items-center justify-between gap-1">
                  <span className="text-studio-text-muted">Provider</span>
                  <span className="text-studio-accent truncate">
                    {draftPipeline.provider}
                  </span>
                </div>
                <div className="flex items-center justify-between gap-1">
                  <span className="text-studio-text-muted">Target</span>
                  <span className="text-studio-text uppercase">
                    {draftPipeline.sourceLang} → {draftPipeline.targetLang}
                  </span>
                </div>
              </div>
            </div>
          </nav>

          {/* Right Content Pane */}
          <div className="flex-1 min-w-0 overflow-y-auto p-4 sm:p-5 space-y-4">
            {/* ==============================================================
             * TAB 1 - APPEARANCE
             * ============================================================== */}
            {activeTab === 'appearance' && (
              <div className="space-y-4" data-testid="settings-panel-appearance">
                <SectionCard
                  title="Color Theme"
                  subtitle="Applies live across the workspace and syncs with your OS prefers-color-scheme when set to Same as System."
                >
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                    {THEME_CARDS.map((card) => {
                      const Icon = card.icon;
                      const isActive = theme === card.id;
                      return (
                        <button
                          key={card.id}
                          type="button"
                          data-testid={`theme-option-${card.id}`}
                          aria-pressed={isActive}
                          onClick={() => setTheme(card.id)}
                          className={`group rounded-studio border p-2.5 text-left transition-colors cursor-pointer space-y-2 ${
                            isActive
                              ? 'border-studio-accent bg-studio-accent/10'
                              : 'border-studio-border bg-studio-panel hover:border-studio-border-strong'
                          }`}
                        >
                          {/* Mini studio window preview */}
                          <div
                            className="h-12 rounded-studio-sm border overflow-hidden flex flex-col"
                            style={{
                              background: card.canvas,
                              borderColor: card.border,
                            }}
                          >
                            <div
                              className="h-3 border-b flex items-center gap-1 px-1.5"
                              style={{
                                background: card.panel,
                                borderColor: card.border,
                              }}
                            >
                              <span className="h-1.5 w-1.5 rounded-full bg-studio-accent" />
                              <span className="h-1 w-6 rounded-full bg-studio-border-strong/80" />
                            </div>
                            <div className="flex-1 flex">
                              <div
                                className="w-5 border-r"
                                style={{
                                  background: card.panel,
                                  borderColor: card.border,
                                }}
                              />
                              <div className="flex-1 flex items-center justify-center">
                                <span className="h-3 w-8 rounded-[2px] bg-studio-accent/80" />
                              </div>
                            </div>
                          </div>

                          <div className="flex items-center justify-between gap-1.5">
                            <div className="flex items-center gap-1.5 min-w-0">
                              <Icon
                                className={`h-3.5 w-3.5 shrink-0 ${
                                  isActive
                                    ? 'text-studio-accent'
                                    : 'text-studio-text-muted'
                                }`}
                              />
                              <span className="text-xs font-semibold text-studio-text truncate">
                                {card.label}
                              </span>
                            </div>
                            {isActive && (
                              <Check className="h-3.5 w-3.5 text-studio-accent shrink-0" />
                            )}
                          </div>
                          <p className="text-[10px] text-studio-text-muted leading-tight">
                            {card.id === 'system'
                              ? `Currently resolved: ${resolvedTheme}`
                              : card.subtitle}
                          </p>
                        </button>
                      );
                    })}
                  </div>
                </SectionCard>

                <SectionCard
                  title="Accent Color Preset"
                  subtitle="Highlights active stages, selected layers, primary actions, and focus rings live."
                >
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                    {ACCENT_PRESETS.map((preset) => {
                      const isActive = accentColor === preset.id;
                      const swatch =
                        resolvedTheme === 'light'
                          ? preset.lightHex
                          : preset.darkHex;
                      return (
                        <button
                          key={preset.id}
                          type="button"
                          data-testid={`accent-preset-${preset.id}`}
                          aria-pressed={isActive}
                          onClick={() => setAccentColor(preset.id)}
                          className={`rounded-studio border p-2.5 text-left transition-colors cursor-pointer flex items-center gap-2.5 ${
                            isActive
                              ? 'border-studio-accent bg-studio-accent/10'
                              : 'border-studio-border bg-studio-panel hover:border-studio-border-strong'
                          }`}
                        >
                          <span
                            className="h-6 w-6 rounded-studio-sm shrink-0 border border-black/20 flex items-center justify-center shadow-inner"
                            style={{ backgroundColor: swatch }}
                          >
                            {isActive && (
                              <Check className="h-3.5 w-3.5 text-white drop-shadow" />
                            )}
                          </span>
                          <div className="min-w-0">
                            <div className="text-xs font-semibold text-studio-text leading-tight">
                              {preset.label}
                            </div>
                            <div className="text-[10px] font-mono text-studio-text-muted uppercase">
                              {swatch}
                            </div>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </SectionCard>

                <SectionCard
                  title="UI Density"
                  subtitle="Adjusts toolbar spacing and control ergonomics across the editor."
                >
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                    {DENSITY_OPTIONS.map((opt) => {
                      const isActive = uiDensity === opt.id;
                      return (
                        <button
                          key={opt.id}
                          type="button"
                          data-testid={`density-option-${opt.id}`}
                          aria-pressed={isActive}
                          onClick={() => setUiDensity(opt.id)}
                          className={`rounded-studio border p-3 text-left transition-colors cursor-pointer flex items-start justify-between gap-2 ${
                            isActive
                              ? 'border-studio-accent bg-studio-accent/10'
                              : 'border-studio-border bg-studio-panel hover:border-studio-border-strong'
                          }`}
                        >
                          <div>
                            <div className="text-xs font-semibold text-studio-text">
                              {opt.label}
                            </div>
                            <div className="text-[11px] text-studio-text-muted mt-0.5">
                              {opt.desc}
                            </div>
                          </div>
                          <span
                            className={`mt-0.5 h-4 w-4 rounded-full border flex items-center justify-center shrink-0 ${
                              isActive
                                ? 'border-studio-accent bg-studio-accent text-studio-accent-fg'
                                : 'border-studio-border-strong'
                            }`}
                          >
                            {isActive && <Check className="h-2.5 w-2.5 stroke-[3]" />}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </SectionCard>
              </div>
            )}

            {/* ==============================================================
             * TAB 2 - PIPELINE
             * ============================================================== */}
            {activeTab === 'pipeline' && (
              <div className="space-y-4" data-testid="settings-panel-pipeline">
                <SectionCard
                  title="Processing"
                  subtitle="Select local AI models for layout detection, text recognition, and artwork inpainting."
                  action={
                    hasNotInstalledModels ? (
                      <button
                        type="button"
                        data-testid="import-koharu-models-btn"
                        disabled={isImportingKoharu}
                        onClick={handleImportKoharuModels}
                        title={
                          pipelineModelsData?.koharu_cache_dir
                            ? `Import local Koharu models from ${pipelineModelsData.koharu_cache_dir}`
                            : 'Import local Koharu models into Vellum storage'
                        }
                        className="studio-btn h-7 px-2.5 text-[11px] border-studio-accent/50 text-studio-accent hover:bg-studio-accent/10 shrink-0"
                      >
                        {isImportingKoharu ? (
                          <>
                            <Loader2 className="h-3 w-3 animate-spin" />
                            <span>Importing…</span>
                          </>
                        ) : (
                          <>
                            <Download className="h-3 w-3" />
                            <span>Import Local Koharu Models</span>
                          </>
                        )}
                      </button>
                    ) : null
                  }
                >
                  {koharuImportFeedback && (
                    <div
                      data-testid="koharu-import-feedback"
                      className={`rounded-studio border px-2.5 py-2 text-[11px] flex items-center gap-2 ${
                        koharuImportFeedback.type === 'error'
                          ? 'border-studio-danger/40 bg-studio-danger/10 text-studio-danger'
                          : 'border-studio-success/40 bg-studio-success/10 text-studio-success'
                      }`}
                    >
                      {koharuImportFeedback.type === 'error' ? (
                        <AlertCircle className="h-3.5 w-3.5 shrink-0" />
                      ) : (
                        <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
                      )}
                      <span className="leading-snug">
                        {koharuImportFeedback.text}
                      </span>
                    </div>
                  )}

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
                  subtitle={`Detected hardware: ${runtimeStatus.executionProvider || 'Auto-detecting…'} · OCR: ${runtimeStatus.ocrProvider || 'CPUExecutionProvider'}`}
                >
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                    {[
                      {
                        id: 'hybrid',
                        label: 'Smart Hybrid (Recommended: Vision GPU + OCR CPU)',
                        desc: 'Runs YOLO11m-Seg & LaMa on GPU while keeping Manga-OCR / RapidOCR on fast AVX2 CPU, saving ~650 MB VRAM for LM Studio',
                        icon: Cpu,
                      },
                      {
                        id: 'force_gpu',
                        label: 'Force All GPU',
                        desc: 'Routes Detection, OCR encoders, and LaMa Inpainting onto GPU (CUDA / DirectML)',
                        icon: Sparkles,
                      },
                      {
                        id: 'cpu_only',
                        label: 'CPU Only',
                        desc: 'Forces all ONNX models (including INT8 detector) onto CPUExecutionProvider with zero VRAM use',
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
                            <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-studio-text">
                              <Icon
                                className={`h-3.5 w-3.5 ${
                                  isActive
                                    ? 'text-studio-accent'
                                    : 'text-studio-text-muted'
                                }`}
                              />
                              {opt.label}
                            </span>
                            {isActive && (
                              <Check className="h-3.5 w-3.5 text-studio-accent" />
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
              </div>
            )}

            {/* ==============================================================
             * TAB 3 - PROVIDERS (KOHARU-STYLE)
             * ============================================================== */}
            {activeTab === 'providers' && (
              <div
                className="grid grid-cols-1 lg:grid-cols-12 gap-3.5"
                data-testid="settings-panel-providers"
              >
                {/* Provider List Rail (8 Koharu-style providers) */}
                <div className="lg:col-span-4 space-y-1.5">
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-studio-text-muted px-1">
                    Providers ({providerPresets.length})
                  </div>
                  <div className="space-y-1">
                    {providerPresets.map((preset) => {
                      const isSelected = selectedProviderId === preset.id;
                      const isActiveEngine = draftPipeline.provider === preset.id;
                      const testInfo = providerModelsCache[preset.id];
                      const isConnected =
                        testInfo?.connected ??
                        (preset.id === 'lmstudio' && runtimeStatus.lmStudio?.connected);

                      return (
                        <button
                          key={preset.id}
                          type="button"
                          data-testid={`provider-card-${preset.id}`}
                          onClick={() => setSelectedProviderId(preset.id)}
                          className={`w-full rounded-studio border px-2.5 py-2 text-left transition-colors cursor-pointer flex items-center justify-between gap-2 ${
                            isSelected
                              ? 'border-studio-accent bg-studio-accent/10'
                              : 'border-studio-border bg-studio-surface hover:border-studio-border-strong'
                          }`}
                        >
                          <div className="min-w-0">
                            <div className="flex items-center gap-1.5">
                              <span
                                className={`h-1.5 w-1.5 rounded-full shrink-0 ${
                                  isConnected
                                    ? 'bg-studio-success'
                                    : testInfo && !testInfo.connected
                                    ? 'bg-studio-danger'
                                    : 'bg-studio-border-strong'
                                }`}
                              />
                              <span className="text-xs font-semibold text-studio-text truncate">
                                {preset.name}
                              </span>
                            </div>
                            <div className="text-[10px] font-mono text-studio-text-muted mt-0.5 flex items-center gap-1.5">
                              <span>{preset.is_local ? 'Local' : 'Cloud'}</span>
                              <span>·</span>
                              <span className="uppercase">{preset.protocol}</span>
                            </div>
                          </div>

                          {isActiveEngine && (
                            <span className="rounded-studio-sm bg-studio-accent text-studio-accent-fg px-1.5 py-0.5 text-[9px] font-mono font-bold uppercase shrink-0">
                              Active
                            </span>
                          )}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Selected Provider Configuration Pane */}
                <div className="lg:col-span-8 space-y-3">
                  <SectionCard
                    title={selectedPreset.name}
                    subtitle={
                      selectedPreset.is_local
                        ? `Local OpenAI-compatible inference server (default: ${selectedPreset.default_base_url})`
                        : selectedPreset.protocol === 'anthropic'
                        ? 'Native Anthropic Claude Messages API'
                        : 'Cloud OpenAI-compatible API endpoint'
                    }
                    action={
                      draftPipeline.provider === selectedProviderId ? (
                        <span className="inline-flex items-center gap-1 rounded-studio-sm border border-studio-accent/50 bg-studio-accent/15 px-2 py-1 text-[10px] font-mono font-semibold text-studio-accent">
                          <CheckCircle2 className="h-3 w-3" />
                          Active Provider
                        </span>
                      ) : (
                        <button
                          type="button"
                          onClick={() => selectActiveProvider(selectedProviderId)}
                          className="studio-btn-accent h-7 px-2.5 text-[11px]"
                        >
                          Use as Active
                        </button>
                      )
                    }
                  >
                    {/* Base URL */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <label
                          htmlFor="provider-base-url"
                          className="text-[11px] font-medium text-studio-text-secondary"
                        >
                          Base URL
                        </label>
                        {selectedPreset.default_base_url && (
                          <button
                            type="button"
                            onClick={() =>
                              updateDraftProfile(selectedProviderId, {
                                base_url: selectedPreset.default_base_url,
                              })
                            }
                            className="text-[10px] font-mono text-studio-accent hover:underline cursor-pointer"
                          >
                            Reset default
                          </button>
                        )}
                      </div>
                      <input
                        id="provider-base-url"
                        type="text"
                        value={selectedProfile.base_url}
                        onChange={(e) =>
                          updateDraftProfile(selectedProviderId, {
                            base_url: e.target.value,
                          })
                        }
                        placeholder={
                          selectedPreset.default_base_url ||
                          'https://your-endpoint.example/v1'
                        }
                        className="studio-input w-full h-8 px-2.5 text-xs font-mono select-text"
                      />
                    </div>

                    {/* Masked API Key with Show/Hide toggle */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <label
                          htmlFor="provider-api-key"
                          className="text-[11px] font-medium text-studio-text-secondary"
                        >
                          API Key{' '}
                          {!selectedPreset.requires_api_key && (
                            <span className="text-studio-text-muted font-normal">
                              (optional for local servers)
                            </span>
                          )}
                        </label>
                        <button
                          type="button"
                          onClick={() =>
                            setShowApiKeys((prev) => ({
                              ...prev,
                              [selectedProviderId]: !prev[selectedProviderId],
                            }))
                          }
                          aria-label={
                            showApiKeys[selectedProviderId]
                              ? 'Hide API Key'
                              : 'Show API Key'
                          }
                          className="inline-flex items-center gap-1 text-[10px] text-studio-text-secondary hover:text-studio-text cursor-pointer"
                        >
                          {showApiKeys[selectedProviderId] ? (
                            <>
                              <EyeOff className="h-3 w-3" />
                              Hide
                            </>
                          ) : (
                            <>
                              <Eye className="h-3 w-3" />
                              Show
                            </>
                          )}
                        </button>
                      </div>
                      <div className="relative flex items-center">
                        <input
                          id="provider-api-key"
                          type={showApiKeys[selectedProviderId] ? 'text' : 'password'}
                          value={selectedProfile.api_key}
                          onChange={(e) =>
                            updateDraftProfile(selectedProviderId, {
                              api_key: e.target.value,
                            })
                          }
                          placeholder={
                            selectedPreset.is_local
                              ? selectedPreset.default_api_key || 'lm-studio'
                              : 'sk-...'
                          }
                          autoComplete="off"
                          className="studio-input w-full h-8 pl-2.5 pr-9 text-xs font-mono select-text"
                        />
                        <button
                          type="button"
                          onClick={() =>
                            setShowApiKeys((prev) => ({
                              ...prev,
                              [selectedProviderId]: !prev[selectedProviderId],
                            }))
                          }
                          className="absolute right-1.5 h-6 w-6 inline-flex items-center justify-center rounded-studio-sm text-studio-text-muted hover:text-studio-text hover:bg-studio-hover cursor-pointer"
                          title={
                            showApiKeys[selectedProviderId]
                              ? 'Mask API Key'
                              : 'Reveal API Key'
                          }
                        >
                          {showApiKeys[selectedProviderId] ? (
                            <EyeOff className="h-3.5 w-3.5" />
                          ) : (
                            <Eye className="h-3.5 w-3.5" />
                          )}
                        </button>
                      </div>
                    </div>

                    {/* Default / Preferred Model */}
                    <div className="space-y-1">
                      <label
                        htmlFor="provider-default-model"
                        className="text-[11px] font-medium text-studio-text-secondary block"
                      >
                        Preferred Model ID
                      </label>
                      <input
                        id="provider-default-model"
                        type="text"
                        value={selectedProfile.model}
                        onChange={(e) =>
                          updateDraftProfile(selectedProviderId, {
                            model: e.target.value,
                          })
                        }
                        placeholder={selectedPreset.default_model || 'model-id'}
                        className="studio-input w-full h-8 px-2.5 text-xs font-mono select-text"
                      />
                    </div>

                    {/* Action Buttons: Test Connection + Fetch Models */}
                    <div className="flex flex-wrap items-center gap-2 pt-1">
                      <button
                        type="button"
                        data-testid="provider-test-connection-btn"
                        disabled={
                          testingProviderId === selectedProviderId ||
                          fetchingModelsFor === selectedProviderId
                        }
                        onClick={() =>
                          handleTestOrFetchProvider(selectedProviderId, 'test')
                        }
                        className="studio-btn-accent h-8 px-3 text-xs"
                      >
                        {testingProviderId === selectedProviderId ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <Wifi className="h-3.5 w-3.5" />
                        )}
                        <span>Test Connection</span>
                      </button>

                      <button
                        type="button"
                        data-testid="provider-fetch-models-btn"
                        disabled={
                          testingProviderId === selectedProviderId ||
                          fetchingModelsFor === selectedProviderId
                        }
                        onClick={() =>
                          handleTestOrFetchProvider(selectedProviderId, 'fetch')
                        }
                        className="studio-btn h-8 px-3 text-xs"
                      >
                        {fetchingModelsFor === selectedProviderId ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <RefreshCw className="h-3.5 w-3.5" />
                        )}
                        <span>Fetch Models</span>
                      </button>

                      {/* Capability pills */}
                      <div className="ml-auto flex items-center gap-1.5 text-[10px] font-mono text-studio-text-muted">
                        {selectedPreset.supports_vision && (
                          <span className="rounded-studio-sm border border-studio-border bg-studio-panel px-1.5 py-0.5">
                            Vision
                          </span>
                        )}
                        {selectedPreset.supports_json_schema && (
                          <span className="rounded-studio-sm border border-studio-border bg-studio-panel px-1.5 py-0.5">
                            JSON Schema
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Connection Test / Fetch Result Banner */}
                    {selectedProviderTest && (
                      <div
                        data-testid="provider-test-result"
                        className={`rounded-studio border p-2.5 text-xs space-y-2 ${
                          selectedProviderTest.connected
                            ? 'border-studio-success/40 bg-studio-success/10 text-studio-text'
                            : 'border-studio-danger/40 bg-studio-danger/10 text-studio-danger'
                        }`}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex items-center gap-1.5 font-medium">
                            {selectedProviderTest.connected ? (
                              <CheckCircle2 className="h-4 w-4 text-studio-success shrink-0" />
                            ) : (
                              <AlertCircle className="h-4 w-4 text-studio-danger shrink-0" />
                            )}
                            <span>
                              {selectedProviderTest.connected
                                ? selectedProviderTest.message ||
                                  `Connected (${selectedProviderTest.models.length} models)`
                                : selectedProviderTest.error ||
                                  selectedProviderTest.message ||
                                  'Connection failed'}
                            </span>
                          </div>
                          {typeof selectedProviderTest.latencyMs === 'number' && (
                            <span className="font-mono text-[10px] text-studio-text-muted shrink-0 tabular-nums">
                              {selectedProviderTest.latencyMs}ms
                            </span>
                          )}
                        </div>

                        {selectedProviderTest.connected &&
                          selectedProviderTest.resolvedModelId && (
                            <div className="text-[11px] font-mono text-studio-text-secondary">
                              Resolved model:{' '}
                              <span className="text-studio-text font-semibold">
                                {selectedProviderTest.activeModel ||
                                  selectedProviderTest.resolvedModelId}
                              </span>
                            </div>
                          )}

                        {/* Discovered Models Picker Chips */}
                        {selectedProviderTest.models?.length > 0 && (
                          <div className="space-y-1.5 pt-1 border-t border-studio-border/60">
                            <div className="text-[10px] font-mono uppercase tracking-wider text-studio-text-muted">
                              Fetched Models ({selectedProviderTest.models.length}) - click to select
                            </div>
                            <div className="flex flex-wrap gap-1.5 max-h-32 overflow-y-auto pr-1">
                              {selectedProviderTest.models.map((mid) => {
                                const isPicked = selectedProfile.model === mid;
                                const isVlm = (
                                  selectedProviderTest.visionModels || []
                                ).includes(mid);
                                return (
                                  <button
                                    key={mid}
                                    type="button"
                                    onClick={() =>
                                      updateDraftProfile(selectedProviderId, {
                                        model: mid,
                                      })
                                    }
                                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-studio-sm border font-mono text-[10px] transition-colors cursor-pointer ${
                                      isPicked
                                        ? 'border-studio-accent bg-studio-accent text-studio-accent-fg font-semibold'
                                        : 'border-studio-border bg-studio-panel text-studio-text hover:border-studio-accent/60'
                                    }`}
                                  >
                                    <span className="truncate max-w-[220px]">{mid}</span>
                                    {isVlm && (
                                      <span className="text-[9px] opacity-80">
                                        [VLM]
                                      </span>
                                    )}
                                  </button>
                                );
                              })}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </SectionCard>
                </div>
              </div>
            )}

            {/* ==============================================================
             * TAB 4 - TRANSLATION
             * ============================================================== */}
            {activeTab === 'translation' && (
              <div className="space-y-4" data-testid="settings-panel-translation">
                {/* Provider, Model & Target Language++ */}
                <SectionCard
                  title="Active Translator & Languages"
                  subtitle="Select LLM provider, model, source comic language, and target language (16 languages)."
                >
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {/* Active Provider */}
                    <div className="space-y-1">
                      <label
                        htmlFor="trans-provider"
                        className="text-[11px] font-medium text-studio-text-secondary block"
                      >
                        Active Provider
                      </label>
                      <select
                        id="trans-provider"
                        value={draftPipeline.provider || 'lmstudio'}
                        onChange={(e) => selectActiveProvider(e.target.value)}
                        className="studio-input w-full h-8 px-2.5 text-xs cursor-pointer"
                      >
                        {providerPresets.map((p) => (
                          <option key={p.id} value={p.id}>
                            {p.name}
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Active Model */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <label
                          htmlFor="trans-model"
                          className="text-[11px] font-medium text-studio-text-secondary"
                        >
                          Model
                        </label>
                        <button
                          type="button"
                          onClick={() =>
                            handleTestOrFetchProvider(
                              draftPipeline.provider || 'lmstudio',
                              'fetch'
                            )
                          }
                          disabled={
                            fetchingModelsFor ===
                            (draftPipeline.provider || 'lmstudio')
                          }
                          className="inline-flex items-center gap-1 text-[10px] font-mono text-studio-accent hover:underline cursor-pointer disabled:opacity-50"
                        >
                          {fetchingModelsFor ===
                          (draftPipeline.provider || 'lmstudio') ? (
                            <Loader2 className="h-2.5 w-2.5 animate-spin" />
                          ) : (
                            <RefreshCw className="h-2.5 w-2.5" />
                          )}
                          Fetch
                        </button>
                      </div>
                      <select
                        id="trans-model"
                        value={draftPipeline.model || ''}
                        onChange={(e) =>
                          updateDraftProfile(
                            draftPipeline.provider || 'lmstudio',
                            { model: e.target.value }
                          )
                        }
                        className="studio-input w-full h-8 px-2.5 text-xs font-mono cursor-pointer"
                      >
                        {translationModelOptions.map((m) => (
                          <option key={m} value={m}>
                            {m}
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Source Language */}
                    <div className="space-y-1">
                      <label
                        htmlFor="trans-source-lang"
                        className="text-[11px] font-medium text-studio-text-secondary block"
                      >
                        Source Comic Language
                      </label>
                      <select
                        id="trans-source-lang"
                        value={draftPipeline.sourceLang || 'ja'}
                        onChange={(e) =>
                          setDraftPipeline((prev) => ({
                            ...prev,
                            sourceLang: e.target.value,
                          }))
                        }
                        className="studio-input w-full h-8 px-2.5 text-xs cursor-pointer"
                      >
                        {SOURCE_LANGUAGE_OPTIONS.map((sl) => (
                          <option key={sl.code} value={sl.code}>
                            {sl.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Target Language++ (16 languages) */}
                    <div className="space-y-1">
                      <label
                        htmlFor="trans-target-lang"
                        className="text-[11px] font-medium text-studio-text-secondary flex items-center gap-1"
                      >
                        <Globe className="h-3 w-3 text-studio-accent" />
                        Target Language++ ({supportedTargetLanguages.length} languages)
                      </label>
                      <select
                        id="trans-target-lang"
                        value={draftPipeline.targetLang || 'id'}
                        onChange={(e) =>
                          setDraftPipeline((prev) => ({
                            ...prev,
                            targetLang: e.target.value,
                          }))
                        }
                        className="studio-input w-full h-8 px-2.5 text-xs cursor-pointer"
                      >
                        {supportedTargetLanguages.map((lang) => (
                          <option key={lang.code} value={lang.code}>
                            {lang.name}
                            {lang.native_name ? ` - ${lang.native_name}` : ''} ({lang.code})
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Tone / Genre Preset */}
                    <div className="space-y-1 sm:col-span-2">
                      <label
                        htmlFor="trans-tone"
                        className="text-[11px] font-medium text-studio-text-secondary block"
                      >
                        Tone / Genre
                      </label>
                      <select
                        id="trans-tone"
                        value={draftPipeline.translationTone || 'standard'}
                        onChange={(e) =>
                          setDraftPipeline((prev) => ({
                            ...prev,
                            translationTone: e.target.value,
                          }))
                        }
                        className="studio-input w-full h-8 px-2.5 text-xs cursor-pointer"
                      >
                        <option value="standard">Standard</option>
                        <option value="casual">Casual</option>
                        <option value="adult_doujinshi">
                          Adult / Doujinshi (Uncensored)
                        </option>
                      </select>
                      <p className="text-[10px] text-studio-text-muted">
                        Controls dialogue register, SFX expressiveness, and uncensored colloquial phrasing.
                      </p>
                    </div>
                  </div>
                </SectionCard>

                {/* Sampling Sliders */}
                <SectionCard
                  title="Sampling Parameters"
                  subtitle="Fine-tune creativity, nucleus sampling, token budget, and repetition control."
                >
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5">
                    {/* Temperature */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-studio-text-secondary font-medium">
                          temperature
                        </span>
                        <span className="font-mono text-studio-accent font-semibold tabular-nums">
                          {Number(draftPipeline.temperature ?? 0.2).toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        aria-label="Temperature"
                        min={0}
                        max={2}
                        step={0.05}
                        value={draftPipeline.temperature ?? 0.2}
                        onChange={(e) =>
                          setDraftPipeline((prev) => ({
                            ...prev,
                            temperature: Number(e.target.value),
                          }))
                        }
                        className="w-full accent-[var(--color-accent)] cursor-pointer"
                      />
                    </div>

                    {/* top_p */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-studio-text-secondary font-medium">
                          top_p
                        </span>
                        <span className="font-mono text-studio-accent font-semibold tabular-nums">
                          {Number(draftPipeline.topP ?? 0.95).toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        aria-label="Top P"
                        min={0}
                        max={1}
                        step={0.01}
                        value={draftPipeline.topP ?? 0.95}
                        onChange={(e) =>
                          setDraftPipeline((prev) => ({
                            ...prev,
                            topP: Number(e.target.value),
                          }))
                        }
                        className="w-full accent-[var(--color-accent)] cursor-pointer"
                      />
                    </div>

                    {/* top_k */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-studio-text-secondary font-medium">
                          top_k
                        </span>
                        <span className="font-mono text-studio-accent font-semibold tabular-nums">
                          {Math.round(Number(draftPipeline.topK ?? 40))}
                        </span>
                      </div>
                      <input
                        type="range"
                        aria-label="Top K"
                        min={0}
                        max={200}
                        step={1}
                        value={draftPipeline.topK ?? 40}
                        onChange={(e) =>
                          setDraftPipeline((prev) => ({
                            ...prev,
                            topK: Number(e.target.value),
                          }))
                        }
                        className="w-full accent-[var(--color-accent)] cursor-pointer"
                      />
                    </div>

                    {/* max_tokens */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-studio-text-secondary font-medium">
                          max_tokens
                        </span>
                        <span className="font-mono text-studio-accent font-semibold tabular-nums">
                          {Math.round(Number(draftPipeline.maxTokens ?? 2048))}
                        </span>
                      </div>
                      <input
                        type="range"
                        aria-label="Max Tokens"
                        min={256}
                        max={8192}
                        step={128}
                        value={Math.min(8192, draftPipeline.maxTokens ?? 2048)}
                        onChange={(e) =>
                          setDraftPipeline((prev) => ({
                            ...prev,
                            maxTokens: Number(e.target.value),
                          }))
                        }
                        className="w-full accent-[var(--color-accent)] cursor-pointer"
                      />
                    </div>

                    {/* frequency_penalty */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-studio-text-secondary font-medium">
                          frequency_penalty
                        </span>
                        <span className="font-mono text-studio-accent font-semibold tabular-nums">
                          {Number(draftPipeline.frequencyPenalty ?? 0).toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        aria-label="Frequency Penalty"
                        min={-2}
                        max={2}
                        step={0.05}
                        value={draftPipeline.frequencyPenalty ?? 0}
                        onChange={(e) =>
                          setDraftPipeline((prev) => ({
                            ...prev,
                            frequencyPenalty: Number(e.target.value),
                          }))
                        }
                        className="w-full accent-[var(--color-accent)] cursor-pointer"
                      />
                    </div>

                    {/* Timeout */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-studio-text-secondary font-medium">
                          timeout (sec)
                        </span>
                        <span className="font-mono text-studio-accent font-semibold tabular-nums">
                          {Math.round(Number(draftPipeline.timeout ?? 90))}s
                        </span>
                      </div>
                      <input
                        type="range"
                        aria-label="Request Timeout"
                        min={15}
                        max={240}
                        step={5}
                        value={draftPipeline.timeout ?? 90}
                        onChange={(e) =>
                          setDraftPipeline((prev) => ({
                            ...prev,
                            timeout: Number(e.target.value),
                          }))
                        }
                        className="w-full accent-[var(--color-accent)] cursor-pointer"
                      />
                    </div>
                  </div>
                </SectionCard>

                {/* Reasoning & Vision Controls */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-3.5">
                  <SectionCard
                    title="Reasoning Control"
                    subtitle="Configure thinking effort for reasoning models and strip <think> blocks."
                  >
                    <div className="space-y-3">
                      <div className="grid grid-cols-4 gap-1.5">
                        {REASONING_LEVELS.map((lvl) => {
                          const isActive =
                            (draftPipeline.reasoningEffort || 'off') === lvl.id;
                          return (
                            <button
                              key={lvl.id}
                              type="button"
                              onClick={() =>
                                setDraftPipeline((prev) => ({
                                  ...prev,
                                  reasoningEffort: lvl.id,
                                }))
                              }
                              className={`h-7 rounded-studio border text-xs font-medium transition-colors cursor-pointer ${
                                isActive
                                  ? 'border-studio-accent bg-studio-accent text-studio-accent-fg font-semibold'
                                  : 'border-studio-border bg-studio-panel text-studio-text-secondary hover:bg-studio-hover hover:text-studio-text'
                              }`}
                            >
                              {lvl.label}
                            </button>
                          );
                        })}
                      </div>

                      <div className="flex items-center justify-between gap-2 pt-1 border-t border-studio-border">
                        <div>
                          <label
                            htmlFor="trans-strip-think"
                            className="text-xs font-medium text-studio-text block"
                          >
                            Strip <code>&lt;think&gt;</code> Tags
                          </label>
                          <p className="text-[10px] text-studio-text-muted">
                            Removes chain-of-thought tags before parsing JSON dialogue.
                          </p>
                        </div>
                        <ToggleSwitch
                          id="trans-strip-think"
                          label="Strip think tags"
                          checked={Boolean(draftPipeline.stripThinking ?? true)}
                          onChange={(val) =>
                            setDraftPipeline((prev) => ({
                              ...prev,
                              stripThinking: val,
                            }))
                          }
                        />
                      </div>
                    </div>
                  </SectionCard>

                  <SectionCard
                    title="Multimodal Vision Context"
                    subtitle="Attach the comic page image so VLMs infer speaker gender, tone & expressions."
                  >
                    <div className="space-y-3">
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <ImageIcon className="h-4 w-4 text-studio-accent shrink-0" />
                          <div>
                            <label
                              htmlFor="trans-vision-toggle"
                              className="text-xs font-medium text-studio-text block"
                            >
                              Send Page Image (VLM)
                            </label>
                            <p className="text-[10px] text-studio-text-muted">
                              Auto-falls back to text-only if model rejects images.
                            </p>
                          </div>
                        </div>
                        <ToggleSwitch
                          id="trans-vision-toggle"
                          label="Enable Multimodal Vision"
                          checked={Boolean(draftPipeline.useVision)}
                          onChange={(val) =>
                            setDraftPipeline((prev) => ({
                              ...prev,
                              useVision: val,
                            }))
                          }
                        />
                      </div>

                      <div className="flex items-center justify-between gap-2 pt-1 border-t border-studio-border">
                        <label
                          htmlFor="trans-vision-max-side"
                          className="text-[11px] text-studio-text-secondary"
                        >
                          Max Image Side (px)
                        </label>
                        <select
                          id="trans-vision-max-side"
                          disabled={!draftPipeline.useVision}
                          value={draftPipeline.visionMaxSide ?? 1024}
                          onChange={(e) =>
                            setDraftPipeline((prev) => ({
                              ...prev,
                              visionMaxSide: Number(e.target.value),
                            }))
                          }
                          className="studio-input h-7 px-2 text-xs font-mono cursor-pointer disabled:opacity-40"
                        >
                          {[512, 768, 1024, 1280, 1536, 2048].map((px) => (
                            <option key={px} value={px}>
                              {px}px
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </SectionCard>
                </div>

                {/* Instruction Bar */}
                <SectionCard
                  title="Instruction Bar"
                  subtitle="Custom system prompt directives injected into every page translation request."
                  action={
                    <div className="flex flex-wrap gap-1">
                      {INSTRUCTION_PRESETS.map((pr) => (
                        <button
                          key={pr.label}
                          type="button"
                          onClick={() =>
                            setDraftPipeline((prev) => ({
                              ...prev,
                              customInstructions: prev.customInstructions
                                ? `${prev.customInstructions}\n${pr.text}`
                                : pr.text,
                            }))
                          }
                          className="rounded-studio-sm border border-studio-border bg-studio-panel hover:bg-studio-hover px-1.5 py-0.5 text-[10px] text-studio-text-secondary hover:text-studio-text cursor-pointer"
                        >
                          + {pr.label}
                        </button>
                      ))}
                    </div>
                  }
                >
                  <textarea
                    rows={3}
                    value={draftPipeline.customInstructions || ''}
                    onChange={(e) =>
                      setDraftPipeline((prev) => ({
                        ...prev,
                        customInstructions: e.target.value,
                      }))
                    }
                    placeholder="e.g., Translate in casual shounen Indonesian; keep character names in Japanese romaji; use 'gue/lu' for delinquent characters..."
                    className="studio-input w-full px-2.5 py-2 text-xs select-text resize-y font-sans"
                  />
                </SectionCard>

                {/* Glossary Table (Source -> Target) */}
                <SectionCard
                  title="Glossary (Source → Target)"
                  subtitle="Strict term overrides enforced across every dialogue block (character names, skills, locations)."
                  action={
                    <div className="flex items-center gap-1.5">
                      <button
                        type="button"
                        onClick={() => {
                          if (!bulkGlossaryMode) {
                            setBulkGlossaryText(
                              glossaryRows
                                .filter((r) => r.source || r.target)
                                .map((r) => `${r.source} -> ${r.target}`)
                                .join('\n')
                            );
                            setBulkGlossaryMode(true);
                          } else {
                            const parsed = [];
                            bulkGlossaryText.split('\n').forEach((line, idx) => {
                              const t = line.trim();
                              if (!t || t.startsWith('#')) return;
                              for (const sep of ['->', '=>', '→', '=', ':']) {
                                if (t.includes(sep)) {
                                  const pos = t.indexOf(sep);
                                  const lhs = t.slice(0, pos).trim();
                                  const rhs = t.slice(pos + sep.length).trim();
                                  if (lhs && rhs) {
                                    parsed.push({
                                      id: `gb_${Date.now()}_${idx}`,
                                      source: lhs,
                                      target: rhs,
                                    });
                                  }
                                  break;
                                }
                              }
                            });
                            setGlossaryRows(parsed);
                            setBulkGlossaryMode(false);
                          }
                        }}
                        className="studio-btn h-6 px-2 text-[10px]"
                      >
                        {bulkGlossaryMode ? 'Table View' : 'Bulk Edit'}
                      </button>
                      {!bulkGlossaryMode && (
                        <button
                          type="button"
                          onClick={() =>
                            setGlossaryRows((prev) => [
                              ...prev,
                              { id: `g_${Date.now()}`, source: '', target: '' },
                            ])
                          }
                          className="studio-btn-accent h-6 px-2 text-[10px]"
                        >
                          <Plus className="h-3 w-3" />
                          <span>Add Term</span>
                        </button>
                      )}
                    </div>
                  }
                >
                  {bulkGlossaryMode ? (
                    <textarea
                      rows={5}
                      value={bulkGlossaryText}
                      onChange={(e) => setBulkGlossaryText(e.target.value)}
                      placeholder={'先輩 -> senpai\n鬼殺隊 -> Demon Slayer Corps\n師兄 -> Shixiong'}
                      className="studio-input w-full px-2.5 py-2 text-xs font-mono select-text resize-y"
                    />
                  ) : glossaryRows.length === 0 ? (
                    <div className="rounded-studio border border-dashed border-studio-border p-3 text-center text-[11px] text-studio-text-muted">
                      No glossary entries yet. Click <strong>Add Term</strong> to map{' '}
                      <code>Source -&gt; Target</code> terms.
                    </div>
                  ) : (
                    <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                      <div className="grid grid-cols-12 gap-2 px-1 text-[10px] font-mono uppercase tracking-wider text-studio-text-muted">
                        <span className="col-span-5">Source Term</span>
                        <span className="col-span-1 text-center">→</span>
                        <span className="col-span-5">Target Translation</span>
                        <span className="col-span-1" />
                      </div>
                      {glossaryRows.map((row, idx) => (
                        <div
                          key={row.id}
                          className="grid grid-cols-12 gap-2 items-center"
                        >
                          <input
                            type="text"
                            value={row.source}
                            onChange={(e) =>
                              setGlossaryRows((prev) =>
                                prev.map((r, i) =>
                                  i === idx ? { ...r, source: e.target.value } : r
                                )
                              )
                            }
                            placeholder="先輩 / 师兄"
                            className="col-span-5 studio-input h-7 px-2 text-xs select-text"
                          />
                          <span className="col-span-1 text-center font-mono text-xs text-studio-text-muted">
                            →
                          </span>
                          <input
                            type="text"
                            value={row.target}
                            onChange={(e) =>
                              setGlossaryRows((prev) =>
                                prev.map((r, i) =>
                                  i === idx ? { ...r, target: e.target.value } : r
                                )
                              )
                            }
                            placeholder="senpai / shixiong"
                            className="col-span-5 studio-input h-7 px-2 text-xs select-text"
                          />
                          <button
                            type="button"
                            onClick={() =>
                              setGlossaryRows((prev) =>
                                prev.filter((_, i) => i !== idx)
                              )
                            }
                            title="Delete glossary entry"
                            className="col-span-1 h-7 w-7 inline-flex items-center justify-center rounded-studio text-studio-text-muted hover:text-studio-danger hover:bg-studio-danger/10 cursor-pointer"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </SectionCard>
              </div>
            )}

            {/* ==============================================================
             * TAB 5 - TYPESETTING
             * ============================================================== */}
            {activeTab === 'typesetting' && (
              <div className="space-y-4" data-testid="settings-panel-typesetting">
                {/* Ordered Font Fallback Chain + Live Speech Bubble Preview */}
                <SectionCard
                  title="Ordered Font Fallback Chain"
                  subtitle="Browser & Konva try each font in order from #1 downward when rendering Latin, accented, or CJK glyphs."
                  action={
                    <button
                      type="button"
                      onClick={() =>
                        setDraftTypesetting((prev) => ({
                          ...prev,
                          fontFallbackChain: [...DEFAULT_FONT_FALLBACK_CHAIN],
                          fontFamily: formatFontFamilyChain(
                            DEFAULT_FONT_FALLBACK_CHAIN
                          ),
                        }))
                      }
                      className="studio-btn h-6 px-2 text-[10px]"
                    >
                      <RotateCcw className="h-3 w-3" />
                      <span>Default Chain</span>
                    </button>
                  }
                >
                  <div className="grid grid-cols-1 lg:grid-cols-12 gap-3.5">
                    {/* Chain List */}
                    <div className="lg:col-span-7 space-y-2">
                      <div className="space-y-1">
                        {(draftTypesetting.fontFallbackChain || []).map(
                          (fontName, idx, arr) => (
                            <div
                              key={`${fontName}-${idx}`}
                              className="flex items-center justify-between gap-2 rounded-studio border border-studio-border bg-studio-panel px-2.5 py-1.5"
                            >
                              <div className="flex items-center gap-2 min-w-0">
                                <span className="h-4 min-w-[20px] px-1 rounded-studio-sm bg-studio-accent/15 text-studio-accent font-mono text-[10px] font-bold inline-flex items-center justify-center shrink-0">
                                  {idx + 1}
                                </span>
                                <span
                                  className="text-xs font-medium text-studio-text truncate"
                                  style={{ fontFamily: `"${fontName}", sans-serif` }}
                                >
                                  {fontName}
                                </span>
                                {idx === 0 && (
                                  <span className="rounded-studio-sm bg-studio-accent text-studio-accent-fg px-1 py-0.5 text-[9px] font-mono font-bold uppercase leading-none">
                                    Primary
                                  </span>
                                )}
                                {installedFontSet &&
                                  !['sans-serif', 'serif', 'monospace', 'cursive', 'fantasy', 'system-ui', 'ui-sans-serif', 'ui-serif', 'ui-monospace'].includes(
                                    String(fontName).trim().toLowerCase()
                                  ) &&
                                  !installedFontSet.has(normalizeFontQuery(fontName)) && (
                                    <span
                                      className="rounded-studio-sm border border-amber-500/40 bg-amber-500/10 text-amber-400 px-1.5 py-0.5 text-[9px] font-mono font-medium leading-none shrink-0"
                                      title={`"${fontName}" was not found in C:\\Windows\\Fonts or local custom fonts`}
                                    >
                                      Not Installed on PC
                                    </span>
                                  )}
                              </div>

                              <div className="flex items-center gap-0.5 shrink-0">
                                <button
                                  type="button"
                                  disabled={idx === 0}
                                  onClick={() => moveChainItem(idx, -1)}
                                  title="Move up"
                                  className="h-6 w-6 inline-flex items-center justify-center rounded-studio-sm text-studio-text-muted hover:text-studio-text hover:bg-studio-hover disabled:opacity-30 cursor-pointer"
                                >
                                  <ArrowUp className="h-3 w-3" />
                                </button>
                                <button
                                  type="button"
                                  disabled={idx === arr.length - 1}
                                  onClick={() => moveChainItem(idx, 1)}
                                  title="Move down"
                                  className="h-6 w-6 inline-flex items-center justify-center rounded-studio-sm text-studio-text-muted hover:text-studio-text hover:bg-studio-hover disabled:opacity-30 cursor-pointer"
                                >
                                  <ArrowDown className="h-3 w-3" />
                                </button>
                                <button
                                  type="button"
                                  disabled={arr.length <= 1}
                                  onClick={() => removeChainItem(idx)}
                                  title="Remove from chain"
                                  className="h-6 w-6 inline-flex items-center justify-center rounded-studio-sm text-studio-text-muted hover:text-studio-danger hover:bg-studio-danger/10 disabled:opacity-30 cursor-pointer"
                                >
                                  <Trash2 className="h-3 w-3" />
                                </button>
                              </div>
                            </div>
                          )
                        )}
                      </div>

                      {/* Manual Add Font to Fallback Chain */}
                      <div className="flex items-center gap-1.5">
                        <input
                          type="text"
                          value={customChainFontInput}
                          onChange={(e) => setCustomChainFontInput(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' && customChainFontInput.trim()) {
                              e.preventDefault();
                              appendChainFont(customChainFontInput.trim());
                              setCustomChainFontInput('');
                            }
                          }}
                          placeholder="Add font family name (e.g. CC Wild Words)..."
                          className="studio-input flex-1 h-7 px-2 text-xs select-text"
                        />
                        <button
                          type="button"
                          disabled={!customChainFontInput.trim()}
                          onClick={() => {
                            appendChainFont(customChainFontInput.trim());
                            setCustomChainFontInput('');
                          }}
                          className="studio-btn h-7 px-2.5 text-xs"
                        >
                          <Plus className="h-3 w-3" />
                          <span>Add</span>
                        </button>
                      </div>

                      <div className="text-[10px] font-mono text-studio-text-muted bg-studio-canvas/70 border border-studio-border rounded-studio-sm px-2 py-1 truncate">
                        font-family: {draftTypesetting.fontFamily}
                      </div>
                    </div>

                    {/* Live Bubble Preview */}
                    <div className="lg:col-span-5 rounded-studio border border-studio-border bg-studio-canvas p-3 flex flex-col items-center justify-center gap-2 min-h-[150px]">
                      <div className="text-[10px] font-mono uppercase tracking-wider text-studio-text-muted">
                        Live Bubble Preview
                      </div>
                      <div className="w-full max-w-[240px] rounded-[28px] border-2 border-neutral-900 bg-white px-4 py-3 text-center shadow-sm">
                        <div
                          style={{
                            fontFamily: draftTypesetting.fontFamily,
                            fontSize: `${Math.min(28, Math.max(12, draftTypesetting.defaultFontSize || 18))}px`,
                            lineHeight: draftTypesetting.defaultLineHeight || 1.18,
                            color: draftTypesetting.defaultTextColor || '#000000',
                            textTransform: draftTypesetting.allCaps
                              ? 'uppercase'
                              : 'none',
                            WebkitTextStroke:
                              (draftTypesetting.defaultStrokeWidth || 0) > 0
                                ? `${Math.min(2, (draftTypesetting.defaultStrokeWidth || 0) * 0.25)}px ${
                                    draftTypesetting.defaultStrokeColor || '#ffffff'
                                  }`
                                : undefined,
                            fontWeight: 700,
                          }}
                        >
                          What are you doing?! This place is dangerous!
                        </div>
                      </div>
                      {hasActivePage && onApplyTypesettingToPage && (
                        <button
                          type="button"
                          onClick={() =>
                            onApplyTypesettingToPage({
                              font_family: draftTypesetting.fontFamily,
                              font_size: draftTypesetting.defaultFontSize,
                              color: draftTypesetting.defaultTextColor,
                              stroke_color: draftTypesetting.defaultStrokeColor,
                              stroke_width: draftTypesetting.defaultStrokeWidth,
                              line_height: draftTypesetting.defaultLineHeight,
                              align: draftTypesetting.defaultAlign,
                              all_caps: draftTypesetting.allCaps,
                            })
                          }
                          className="studio-btn h-6 px-2.5 text-[10px]"
                        >
                          <Sparkles className="h-3 w-3 text-studio-accent" />
                          <span>Apply Defaults to Active Page</span>
                        </button>
                      )}
                    </div>
                  </div>
                </SectionCard>

                {/* Default Typography & Lettering Options */}
                <SectionCard
                  title="Default Lettering Style"
                  subtitle="Defaults applied to newly detected speech bubbles and manual text layers."
                >
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
                    {/* Default Font Size */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-studio-text-secondary font-medium">
                          Default Size
                        </span>
                        <span className="font-mono font-semibold text-studio-accent tabular-nums">
                          {draftTypesetting.defaultFontSize}px
                        </span>
                      </div>
                      <input
                        type="range"
                        aria-label="Default Font Size"
                        min={10}
                        max={48}
                        step={1}
                        value={draftTypesetting.defaultFontSize ?? 18}
                        onChange={(e) =>
                          setDraftTypesetting((prev) => ({
                            ...prev,
                            defaultFontSize: Number(e.target.value),
                          }))
                        }
                        className="w-full accent-[var(--color-accent)] cursor-pointer"
                      />
                    </div>

                    {/* Stroke Width */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-studio-text-secondary font-medium">
                          Stroke Width
                        </span>
                        <span className="font-mono font-semibold text-studio-accent tabular-nums">
                          {Number(draftTypesetting.defaultStrokeWidth ?? 4).toFixed(1)}px
                        </span>
                      </div>
                      <input
                        type="range"
                        aria-label="Default Stroke Width"
                        min={0}
                        max={12}
                        step={0.5}
                        value={draftTypesetting.defaultStrokeWidth ?? 4}
                        onChange={(e) =>
                          setDraftTypesetting((prev) => ({
                            ...prev,
                            defaultStrokeWidth: Number(e.target.value),
                          }))
                        }
                        className="w-full accent-[var(--color-accent)] cursor-pointer"
                      />
                    </div>

                    {/* Line Height */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-studio-text-secondary font-medium">
                          Line Height
                        </span>
                        <span className="font-mono font-semibold text-studio-accent tabular-nums">
                          {Number(draftTypesetting.defaultLineHeight ?? 1.18).toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        aria-label="Default Line Height"
                        min={0.9}
                        max={2.0}
                        step={0.02}
                        value={draftTypesetting.defaultLineHeight ?? 1.18}
                        onChange={(e) =>
                          setDraftTypesetting((prev) => ({
                            ...prev,
                            defaultLineHeight: Number(e.target.value),
                          }))
                        }
                        className="w-full accent-[var(--color-accent)] cursor-pointer"
                      />
                    </div>

                    {/* Text & Stroke Color */}
                    <div className="space-y-1">
                      <span className="text-[11px] text-studio-text-secondary font-medium block">
                        Fill / Stroke Color
                      </span>
                      <div className="flex items-center gap-2">
                        <label className="inline-flex items-center gap-1.5 text-[10px] font-mono text-studio-text-muted cursor-pointer">
                          <input
                            type="color"
                            aria-label="Default Text Fill Color"
                            value={draftTypesetting.defaultTextColor || '#000000'}
                            onChange={(e) =>
                              setDraftTypesetting((prev) => ({
                                ...prev,
                                defaultTextColor: e.target.value,
                              }))
                            }
                            className="h-6 w-6 rounded-studio-sm border border-studio-border bg-transparent cursor-pointer"
                          />
                          Fill
                        </label>
                        <label className="inline-flex items-center gap-1.5 text-[10px] font-mono text-studio-text-muted cursor-pointer">
                          <input
                            type="color"
                            aria-label="Default Text Stroke Color"
                            value={draftTypesetting.defaultStrokeColor || '#ffffff'}
                            onChange={(e) =>
                              setDraftTypesetting((prev) => ({
                                ...prev,
                                defaultStrokeColor: e.target.value,
                              }))
                            }
                            className="h-6 w-6 rounded-studio-sm border border-studio-border bg-transparent cursor-pointer"
                          />
                          Stroke
                        </label>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 border-t border-studio-border">
                    <div className="flex items-center justify-between gap-2">
                      <div>
                        <label
                          htmlFor="ts-auto-fit"
                          className="text-xs font-medium text-studio-text block"
                        >
                          Auto-Fit Font Size to Bubble
                        </label>
                        <p className="text-[10px] text-studio-text-muted">
                          Scales dialogue size automatically inside each bounding box.
                        </p>
                      </div>
                      <ToggleSwitch
                        id="ts-auto-fit"
                        label="Auto-Fit Font Size"
                        checked={Boolean(draftTypesetting.autoFitFontSize)}
                        onChange={(val) =>
                          setDraftTypesetting((prev) => ({
                            ...prev,
                            autoFitFontSize: val,
                          }))
                        }
                      />
                    </div>

                    <div className="flex items-center justify-between gap-2">
                      <div>
                        <label
                          htmlFor="ts-all-caps"
                          className="text-xs font-medium text-studio-text block"
                        >
                          ALL-CAPS Comic Lettering
                        </label>
                        <p className="text-[10px] text-studio-text-muted">
                          Renders dialogue text in classic uppercase manga lettering.
                        </p>
                      </div>
                      <ToggleSwitch
                        id="ts-all-caps"
                        label="ALL-CAPS Toggle"
                        checked={Boolean(draftTypesetting.allCaps)}
                        onChange={(val) =>
                          setDraftTypesetting((prev) => ({
                            ...prev,
                            allCaps: val,
                          }))
                        }
                      />
                    </div>
                  </div>
                </SectionCard>

                {/* Searchable Font Picker across /api/fonts/list + Catalog Downloader + Upload */}
                <SectionCard
                  title="Font Picker & Comic Catalog"
                  subtitle={
                    fontsData?.counts
                      ? `${fontsData.counts.families} installed families (${fontsData.counts.system} system, ${fontsData.counts.custom} custom, ${fontsData.counts.downloaded} downloaded) · ${fontsData.counts.catalog} online comic fonts`
                      : 'Browse Windows system fonts, download offline comic fonts, or import .ttf/.otf files.'
                  }
                  action={
                    <div className="flex items-center gap-1.5">
                      <input
                        ref={fontUploadInputRef}
                        type="file"
                        accept=".ttf,.otf,.woff2,.woff"
                        onChange={handleUploadCustomFont}
                        className="hidden"
                        data-testid="font-upload-input"
                      />
                      <button
                        type="button"
                        disabled={uploadingFont}
                        onClick={() => fontUploadInputRef.current?.click()}
                        className="studio-btn-accent h-7 px-2.5 text-[11px]"
                      >
                        {uploadingFont ? (
                          <Loader2 className="h-3 w-3 animate-spin" />
                        ) : (
                          <Upload className="h-3 w-3" />
                        )}
                        <span>Import .ttf/.otf</span>
                      </button>
                      <button
                        type="button"
                        disabled={isLoadingFonts}
                        onClick={() => fetchFontsList(true)}
                        title="Rescan system and local fonts"
                        className="studio-btn h-7 w-7 p-0"
                      >
                        <RefreshCw
                          className={`h-3.5 w-3.5 ${isLoadingFonts ? 'animate-spin' : ''}`}
                        />
                      </button>
                    </div>
                  }
                >
                  {/* Search & Source Filter Bar */}
                  <div className="flex flex-col sm:flex-row gap-2">
                    <div className="relative flex-1">
                      <Search className="h-3.5 w-3.5 text-studio-text-muted absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
                      <input
                        type="text"
                        value={fontSearch}
                        onChange={(e) => setFontSearch(e.target.value)}
                        placeholder="Search Windows fonts or 129 comic/CJK catalog fonts..."
                        className="studio-input w-full h-8 pl-8 pr-2.5 text-xs select-text"
                      />
                    </div>

                    <div className="inline-flex rounded-studio border border-studio-border bg-studio-panel p-0.5 gap-0.5 shrink-0 overflow-x-auto">
                      {[
                        { id: 'all', label: 'All' },
                        { id: 'installed', label: 'Installed' },
                        { id: 'system', label: 'Windows' },
                        { id: 'custom', label: 'Custom' },
                        { id: 'catalog', label: 'Online Catalog' },
                      ].map((tab) => (
                        <button
                          key={tab.id}
                          type="button"
                          onClick={() => setFontSourceTab(tab.id)}
                          className={`h-7 px-2 rounded-studio-sm text-[11px] font-medium transition-colors cursor-pointer shrink-0 ${
                            fontSourceTab === tab.id
                              ? 'bg-studio-accent text-studio-accent-fg font-semibold'
                              : 'text-studio-text-secondary hover:text-studio-text hover:bg-studio-hover'
                          }`}
                        >
                          {tab.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Catalog Style Sub-Filters */}
                  {(fontSourceTab === 'all' || fontSourceTab === 'catalog') && (
                    <div className="flex flex-wrap gap-1">
                      {FONT_STYLE_FILTERS.map((st) => (
                        <button
                          key={st.id}
                          type="button"
                          onClick={() => setFontStyleFilter(st.id)}
                          className={`px-2 py-0.5 rounded-studio-sm border text-[10px] font-medium transition-colors cursor-pointer ${
                            fontStyleFilter === st.id
                              ? 'border-studio-accent bg-studio-accent/15 text-studio-accent'
                              : 'border-studio-border bg-studio-panel text-studio-text-muted hover:text-studio-text'
                          }`}
                        >
                          {st.label}
                        </button>
                      ))}
                    </div>
                  )}

                  {fontsError && (
                    <div className="rounded-studio border border-studio-danger/40 bg-studio-danger/10 px-2.5 py-2 text-xs text-studio-danger flex items-center gap-2">
                      <AlertCircle className="h-3.5 w-3.5 shrink-0" />
                      <span>{fontsError}</span>
                    </div>
                  )}

                  {isLoadingFonts && !fontsData ? (
                    <div className="py-8 flex items-center justify-center gap-2 text-xs text-studio-text-muted">
                      <Loader2 className="h-4 w-4 animate-spin text-studio-accent" />
                      <span>Scanning Windows & local fonts…</span>
                    </div>
                  ) : (
                    <div className="space-y-3 max-h-64 overflow-y-auto pr-1">
                      {/* Installed Families Section */}
                      {filteredInstalledFamilies.length > 0 && (
                        <div className="space-y-1.5">
                          <div className="text-[10px] font-mono uppercase tracking-wider text-studio-text-muted">
                            Installed Fonts ({filteredInstalledFamilies.length})
                          </div>
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                            {filteredInstalledFamilies.slice(0, 80).map((fam) => {
                              const inChain = (
                                draftTypesetting.fontFallbackChain || []
                              ).some(
                                (f) => f.toLowerCase() === fam.family.toLowerCase()
                              );
                              return (
                                <div
                                  key={`${fam.category}-${fam.family}`}
                                  onMouseEnter={() => {
                                    if (fam.url) {
                                      ensureFontFaceLoaded(
                                        fam.family,
                                        fam.url,
                                        apiBaseUrl
                                      );
                                    }
                                  }}
                                  className="rounded-studio border border-studio-border bg-studio-panel px-2.5 py-2 flex items-center justify-between gap-2 hover:border-studio-border-strong transition-colors"
                                >
                                  <div className="min-w-0">
                                    <div
                                      className="text-xs font-semibold text-studio-text truncate"
                                      style={{
                                        fontFamily: `"${fam.family}", sans-serif`,
                                      }}
                                    >
                                      {fam.family}
                                    </div>
                                    <div className="text-[10px] font-mono text-studio-text-muted flex items-center gap-1.5">
                                      <span className="uppercase">
                                        {fam.category}
                                      </span>
                                      <span>·</span>
                                      <span>
                                        {fam.variants?.length || 1} style
                                        {(fam.variants?.length || 1) === 1 ? '' : 's'}
                                      </span>
                                    </div>
                                  </div>

                                  <div className="flex items-center gap-1 shrink-0">
                                    <button
                                      type="button"
                                      onClick={() =>
                                        setChainPrimaryFont(fam.family, fam.url)
                                      }
                                      className="studio-btn h-6 px-1.5 text-[10px]"
                                      title="Set as #1 Primary Font"
                                    >
                                      Primary
                                    </button>
                                    <button
                                      type="button"
                                      disabled={inChain}
                                      onClick={() =>
                                        appendChainFont(fam.family, fam.url)
                                      }
                                      className="studio-btn h-6 px-1.5 text-[10px]"
                                      title="Append to Fallback Chain"
                                    >
                                      {inChain ? 'In Chain' : '+ Chain'}
                                    </button>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {/* Online Curated Comic Catalog Section */}
                      {filteredCatalogItems.length > 0 && (
                        <div className="space-y-1.5">
                          <div className="text-[10px] font-mono uppercase tracking-wider text-studio-text-muted">
                            Online Comic Font Catalog ({filteredCatalogItems.length})
                          </div>
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                            {filteredCatalogItems.slice(0, 80).map((item) => {
                              const isDownloading = downloadingFontId === item.id;
                              return (
                                <div
                                  key={item.id}
                                  className="rounded-studio border border-studio-border bg-studio-panel px-2.5 py-2 flex items-center justify-between gap-2"
                                >
                                  <div className="min-w-0">
                                    <div className="flex items-center gap-1.5">
                                      <span className="text-xs font-semibold text-studio-text truncate">
                                        {item.family}
                                      </span>
                                      <span className="rounded-studio-sm border border-studio-border bg-studio-surface px-1 py-0.5 font-mono text-[9px] uppercase leading-none text-studio-text-muted shrink-0">
                                        {item.style_category}
                                      </span>
                                    </div>
                                    <div className="text-[10px] text-studio-text-muted truncate mt-0.5">
                                      {item.description}
                                    </div>
                                  </div>

                                  {item.installed ? (
                                    <button
                                      type="button"
                                      onClick={() =>
                                        setChainPrimaryFont(item.family, item.url)
                                      }
                                      className="studio-btn h-6 px-2 text-[10px] text-studio-success border-studio-success/40 shrink-0"
                                    >
                                      <Check className="h-3 w-3" />
                                      <span>Use</span>
                                    </button>
                                  ) : (
                                    <button
                                      type="button"
                                      disabled={isDownloading}
                                      onClick={() =>
                                        handleDownloadCatalogFont(item)
                                      }
                                      className="studio-btn h-6 px-2 text-[10px] shrink-0"
                                    >
                                      {isDownloading ? (
                                        <Loader2 className="h-3 w-3 animate-spin" />
                                      ) : (
                                        <Download className="h-3 w-3 text-studio-accent" />
                                      )}
                                      <span>Download</span>
                                    </button>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {filteredInstalledFamilies.length === 0 &&
                        filteredCatalogItems.length === 0 && (
                          <div className="py-6 text-center text-xs text-studio-text-muted">
                            No fonts matched "{fontSearch}". Try another search or import a local <code>.ttf</code>/<code>.otf</code> file.
                          </div>
                        )}
                    </div>
                  )}
                </SectionCard>
              </div>
            )}
          </div>
        </div>

        {/* Sticky Footer */}
        <footer className="px-4 py-2.5 border-t border-studio-border bg-studio-surface flex flex-wrap items-center justify-between gap-2 shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <button
              type="button"
              onClick={handleResetActiveTab}
              className="studio-btn h-7 px-2.5 text-[11px]"
              title={`Reset ${currentTabMeta.label} settings to defaults`}
            >
              <RotateCcw className="h-3 w-3" />
              <span>Reset Tab</span>
            </button>

            {saveFeedback ? (
              <span
                data-testid="settings-save-feedback"
                className={`inline-flex items-center gap-1.5 text-[11px] font-medium truncate ${
                  saveFeedback.type === 'error'
                    ? 'text-studio-danger'
                    : 'text-studio-success'
                }`}
              >
                {saveFeedback.type === 'error' ? (
                  <AlertCircle className="h-3.5 w-3.5 shrink-0" />
                ) : (
                  <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
                )}
                <span className="truncate">{saveFeedback.text}</span>
              </span>
            ) : (
              settingsPath && (
                <span
                  className="hidden sm:inline font-mono text-[10px] text-studio-text-muted truncate max-w-[300px]"
                  title={settingsPath}
                >
                  Synced: {settingsPath}
                </span>
              )
            )}
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={onClose}
              className="studio-btn h-7 px-3 text-xs"
            >
              Close
            </button>
            <button
              type="button"
              data-testid="settings-save-btn"
              disabled={isSaving}
              onClick={handleSaveAll}
              className="studio-btn-accent h-7 px-3.5 text-xs"
            >
              {isSaving ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  <span>Saving…</span>
                </>
              ) : (
                <>
                  <Save className="h-3.5 w-3.5" />
                  <span>Save Settings</span>
                </>
              )}
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}
