import {
  HardDrive,
  Languages,
  Loader2,
  Monitor,
  Moon,
  Palette,
  Server,
  Sliders,
  Sun,
  Type,
} from 'lucide-react';
import useEditorStore from '../../store/useEditorStore.js';
import { formatExecutionProvider } from '../StatusBar.jsx';
export { formatExecutionProvider };

export const TABS = [
  { id: 'appearance', label: 'Appearance', icon: Palette, desc: 'Theme, accent & density' },
  { id: 'pipeline', label: 'Pipeline', icon: Sliders, desc: 'GPU/CPU, detect & LaMa' },
  { id: 'providers', label: 'Providers', icon: Server, desc: 'LLM endpoints & keys' },
  { id: 'translation', label: 'Translation', icon: Languages, desc: 'Model, sampling & glossary' },
  { id: 'typesetting', label: 'Typesetting', icon: Type, desc: 'Fonts, fallback & style' },
  { id: 'storage', label: 'Storage & About', icon: HardDrive, desc: 'Disk usage, cache & backup' },
];

export const THEME_CARDS = [
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

export const ACCENT_PRESETS = [
  { id: 'amber', label: 'Amber', darkHex: '#E59A3C', lightHex: '#C97A16', desc: 'Warm studio amber' },
  { id: 'teal', label: 'Teal', darkHex: '#14B8A6', lightHex: '#0D9488', desc: 'Crisp cyan-teal' },
  { id: 'crimson', label: 'Crimson', darkHex: '#E11D48', lightHex: '#BE123C', desc: 'Bold manga crimson' },
  { id: 'slate', label: 'Slate', darkHex: '#5B9BD5', lightHex: '#3B76B0', desc: 'Cool slate blue' },
];

export const DENSITY_OPTIONS = [
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

export const LAMA_TILE_SIZES = [256, 384, 512, 768, 1024];

export const READING_ORDER_OPTIONS = [
  { id: 'auto', label: 'Auto', desc: 'JA → RTL · KO/ZH → LTR' },
  { id: 'rtl', label: 'RTL', desc: 'Right-to-Left (Manga)' },
  { id: 'ltr', label: 'LTR', desc: 'Left-to-Right (Manhwa/Manhua)' },
];

export const BACKEND_OPTIONS = [
  { id: 'auto', label: 'Auto' },
  { id: 'directml', label: 'Universal GPU' },
  { id: 'cuda', label: 'NVIDIA CUDA' },
  { id: 'cpu', label: 'CPU' },
];

export const REASONING_LEVELS = [
  { id: 'off', label: 'Off' },
  { id: 'low', label: 'Low' },
  { id: 'medium', label: 'Medium' },
  { id: 'high', label: 'High' },
];

export const SOURCE_LANGUAGE_OPTIONS = [
  { code: 'ja', label: 'Japanese (Manga · JA)' },
  { code: 'ko', label: 'Korean (Manhwa · KO)' },
  { code: 'zh', label: 'Chinese (Manhua · ZH)' },
];

export const INSTRUCTION_PRESETS = [
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

export const FONT_STYLE_FILTERS = [
  { id: 'all', label: 'All Styles' },
  { id: 'comic', label: 'Comic' },
  { id: 'sfx', label: 'SFX / Action' },
  { id: 'handwriting', label: 'Handwriting' },
  { id: 'webtoon', label: 'Webtoon' },
  { id: 'cjk_ja', label: 'JP' },
  { id: 'cjk_ko', label: 'KR' },
  { id: 'cjk_zh', label: 'ZH' },
];

export function ToggleSwitch({ checked, onChange, disabled = false, label, id }) {
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

export function SectionCard({ title, subtitle, action, children }) {
  return (
    <section className="rounded-studio border border-studio-border bg-studio-surface p-[var(--studio-card-pad)] space-y-3">
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

export function mergePipelineModelOptions(defaults, apiEntries) {
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
        source: live.source || null,
        sizeMb: typeof live.size_mb === 'number' ? live.size_mb : null,
        defaultThresholds: live.default_thresholds || null,
        lmStudioHint: live.lm_studio_hint || null,
      });
    }
  }

  return merged;
}

export function ModelStatusBadge({ model, isLoading, testId }) {
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
    ? 'Cached as .safetensors - click "Import Local Models..." to convert to ONNX.'
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

export function normalizeExecMode(mode) {
  const m = String(mode || 'hybrid').toLowerCase();
  if (m === 'cpu' || m === 'cpu_only') return 'cpu_only';
  if (m === 'gpu' || m === 'force_gpu') return 'force_gpu';
  return 'hybrid';
}

export function StageHardwareBadge({
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
          ? 'GPU (NVIDIA CUDA)'
          : 'GPU (Universal / CUDA)';
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
          ? 'GPU (NVIDIA CUDA)'
          : 'GPU (Universal / CUDA)';
    }
  } else {
    isGpu = true;
    label =
      liveProvider === 'CUDAExecutionProvider'
        ? 'GPU (NVIDIA CUDA)'
        : 'GPU (Universal / CUDA)';
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

export function glossaryDictToRows(glossaryObj) {
  if (!glossaryObj || typeof glossaryObj !== 'object') return [];
  return Object.entries(glossaryObj).map(([source, target], idx) => ({
    id: `g_${idx}_${source}`,
    source: String(source),
    target: String(target),
  }));
}

export function glossaryRowsToDict(rows) {
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
 * 5-Tab Settings Modal synced with `/api/settings` and `useEditorStore`.
 *
 * Tabs:
 * 1. Appearance  - Dark / Light / System theme, Accent presets, UI density
 * 2. Pipeline    - Auto GPU/CPU vs CPU Only, detection threshold, bbox padding, LaMa tile size, reading order, auto-unload VRAM
 * 3. Providers   - 8 built-in providers (LM Studio :1234, Ollama, OpenAI, Gemini, Anthropic, DeepSeek, OpenRouter, Custom), Base URL, masked API key toggle, Test Connection, Fetch Models
 * 4. Translation - Provider & Model, Target Language++ (16 langs), Sampling sliders, Reasoning + strip <think>, Vision toggle, Instruction Bar, Glossary table
 * 5. Typesetting - Searchable Font Picker (/api/fonts/list + catalog downloader + Import .ttf/.otf), ordered Font Fallback Chain, auto-fit, default size, stroke, line height, ALL-CAPS
 */
