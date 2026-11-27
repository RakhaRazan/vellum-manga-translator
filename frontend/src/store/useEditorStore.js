import { create } from 'zustand';
import {
  minimizeBoxOverlapEnergy,
  preserveBlockEditsWithHungarian,
} from '../utils/canvasMathOps.js';

export const MAX_HISTORY_STEPS = 30;
export const THEME_STORAGE_KEY = 'vellum.theme';
export const ACCENT_STORAGE_KEY = 'vellum.accent';
export const DENSITY_STORAGE_KEY = 'vellum.density';
export const SETTINGS_STORAGE_KEY = 'vellum.settings';
export const VALID_THEMES = ['dark', 'light', 'system'];
export const VALID_ACCENTS = ['amber', 'teal', 'crimson', 'slate'];
export const VALID_DENSITIES = ['compact', 'comfortable'];

export const DEFAULT_FONT_FALLBACK_CHAIN = [
  'CC Wild Words',
  'Anime Ace',
  'Segoe UI Symbol',
  'Noto Sans Symbols 2',
  'Noto Sans',
  'sans-serif',
];

export const DEFAULT_FONT_FAMILY =
  '"CC Wild Words", "Anime Ace", "Segoe UI Symbol", "Noto Sans Symbols 2", "Noto Sans", sans-serif';

export const DEFAULT_TYPESETTING_CONFIG = {
  fontFamily: DEFAULT_FONT_FAMILY,
  fontFallbackChain: [...DEFAULT_FONT_FALLBACK_CHAIN],
  autoFitFontSize: true,
  enableHyphenation: true,
  defaultFontSize: 18,
  defaultStrokeWidth: 4,
  defaultStrokeColor: '#ffffff',
  defaultTextColor: '#000000',
  defaultLineHeight: 1.18,
  defaultAlign: 'center',
  allCaps: false,
  autoTiltText: false,
};

export const DEFAULT_PROVIDER_PRESETS = [
  {
    id: 'none',
    name: 'None (Manual / No-LLM)',
    default_base_url: '',
    default_model: '',
    default_api_key: '',
    requires_api_key: false,
    requires_base_url: false,
    is_local: true,
    protocol: 'none',
    supports_json_schema: false,
    supports_vision: false,
    supports_top_k: false,
    supports_frequency_penalty: false,
    supports_reasoning_effort: false,
    fallback_models: [],
  },
  {
    id: 'lmstudio',
    name: 'LM Studio (Local)',
    default_base_url: 'http://localhost:1234/v1',
    default_model: '',
    default_api_key: '',
    requires_api_key: false,
    requires_base_url: true,
    is_local: true,
    protocol: 'openai',
    supports_json_schema: true,
    supports_vision: true,
    supports_top_k: true,
    supports_frequency_penalty: true,
    supports_reasoning_effort: true,
    fallback_models: [],
  },
  {
    id: 'ollama',
    name: 'Ollama (Local)',
    default_base_url: 'http://localhost:11434/v1',
    default_model: 'qwen2.5:7b',
    default_api_key: '',
    requires_api_key: false,
    requires_base_url: true,
    is_local: true,
    protocol: 'openai',
    supports_json_schema: true,
    supports_vision: true,
    supports_top_k: true,
    supports_frequency_penalty: true,
    supports_reasoning_effort: true,
    fallback_models: ['qwen2.5:7b', 'gemma3:4b', 'llama3.2-vision'],
  },
  {
    id: 'openai',
    name: 'OpenAI',
    default_base_url: 'https://api.openai.com/v1',
    default_model: 'gpt-4o-mini',
    default_api_key: '',
    requires_api_key: true,
    requires_base_url: true,
    is_local: false,
    protocol: 'openai',
    supports_json_schema: true,
    supports_vision: true,
    supports_top_k: false,
    supports_frequency_penalty: true,
    supports_reasoning_effort: true,
    fallback_models: ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1-mini', 'o4-mini'],
  },
  {
    id: 'gemini',
    name: 'Google Gemini',
    default_base_url: 'https://generativelanguage.googleapis.com/v1beta/openai',
    default_model: 'gemini-2.5-flash',
    default_api_key: '',
    requires_api_key: true,
    requires_base_url: true,
    is_local: false,
    protocol: 'openai',
    supports_json_schema: true,
    supports_vision: true,
    supports_top_k: false,
    supports_frequency_penalty: false,
    supports_reasoning_effort: true,
    fallback_models: ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash'],
  },
  {
    id: 'anthropic',
    name: 'Anthropic (Claude)',
    default_base_url: 'https://api.anthropic.com/v1',
    default_model: 'claude-sonnet-4-20250514',
    default_api_key: '',
    requires_api_key: true,
    requires_base_url: true,
    is_local: false,
    protocol: 'anthropic',
    supports_json_schema: false,
    supports_vision: true,
    supports_top_k: true,
    supports_frequency_penalty: false,
    supports_reasoning_effort: true,
    fallback_models: [
      'claude-sonnet-4-20250514',
      'claude-3-7-sonnet-latest',
      'claude-3-5-haiku-latest',
    ],
  },
  {
    id: 'deepseek',
    name: 'DeepSeek',
    default_base_url: 'https://api.deepseek.com/v1',
    default_model: 'deepseek-chat',
    default_api_key: '',
    requires_api_key: true,
    requires_base_url: true,
    is_local: false,
    protocol: 'openai',
    supports_json_schema: false,
    supports_vision: false,
    supports_top_k: false,
    supports_frequency_penalty: true,
    supports_reasoning_effort: false,
    fallback_models: ['deepseek-chat', 'deepseek-reasoner'],
  },
  {
    id: 'openrouter',
    name: 'OpenRouter',
    default_base_url: 'https://openrouter.ai/api/v1',
    default_model: 'google/gemini-2.5-flash',
    default_api_key: '',
    requires_api_key: true,
    requires_base_url: true,
    is_local: false,
    protocol: 'openai',
    supports_json_schema: true,
    supports_vision: true,
    supports_top_k: true,
    supports_frequency_penalty: true,
    supports_reasoning_effort: true,
    fallback_models: [
      'google/gemini-2.5-flash',
      'openai/gpt-4o-mini',
      'anthropic/claude-sonnet-4',
      'deepseek/deepseek-chat',
    ],
  },
  {
    id: 'custom',
    name: 'Custom (OpenAI-Compatible)',
    default_base_url: '',
    default_model: '',
    default_api_key: '',
    requires_api_key: false,
    requires_base_url: true,
    is_local: false,
    protocol: 'openai',
    supports_json_schema: true,
    supports_vision: true,
    supports_top_k: true,
    supports_frequency_penalty: true,
    supports_reasoning_effort: true,
    fallback_models: [],
  },
];

export function buildDefaultProviderProfiles() {
  return Object.fromEntries(
    DEFAULT_PROVIDER_PRESETS.map((p) => [
      p.id,
      {
        base_url: p.default_base_url,
        api_key: p.default_api_key || '',
        model: p.default_model,
      },
    ])
  );
}

export const MANGA_SYMBOL_FALLBACK_FONTS = [
  'Segoe UI Symbol',
  'Noto Sans Symbols 2',
  'Noto Sans',
  'Arial Unicode MS',
  'sans-serif',
];

const GENERIC_CSS_FONTS = new Set([
  'sans-serif',
  'serif',
  'monospace',
  'cursive',
  'fantasy',
  'system-ui',
  'ui-sans-serif',
  'ui-serif',
  'ui-monospace',
]);

export function parseFontFamilyList(fontFamilyStr) {
  if (!fontFamilyStr || typeof fontFamilyStr !== 'string') return [];
  return fontFamilyStr
    .split(',')
    .map((part) => part.trim().replace(/^["']+|["']+$/g, '').trim())
    .filter(Boolean);
}

export function extractPrimaryFontName(fontFamilyStr) {
  const parts = parseFontFamilyList(fontFamilyStr);
  return parts[0] || DEFAULT_FONT_FALLBACK_CHAIN[0];
}

export function formatFontFamilyChain(chain = []) {
  const formatted = [];
  const seen = new Set();
  for (const raw of chain || []) {
    const cleaned = String(raw || '').trim().replace(/^["']+|["']+$/g, '').trim();
    if (!cleaned) continue;
    const lower = cleaned.toLowerCase();
    if (seen.has(lower)) continue;
    seen.add(lower);
    if (GENERIC_CSS_FONTS.has(lower)) {
      formatted.push(lower);
    } else if (/[\s-]/.test(cleaned)) {
      formatted.push(`"${cleaned}"`);
    } else {
      formatted.push(cleaned);
    }
  }
  if (formatted.length === 0) return DEFAULT_FONT_FAMILY;
  if (!seen.has('sans-serif') && !seen.has('serif') && !seen.has('monospace')) {
    formatted.push('sans-serif');
  }
  return formatted.join(', ');
}

/**
 * Build a Konva-ready `fontFamily` string starting with the block's primary font,
 * followed by the user's configured Font Fallback Chain and symbol-safe fallbacks
 * so special manga symbols (`♡`, `♪`, `…`, `★`) never render as missing tofu boxes.
 */
export function buildKonvaFontFamily(
  primaryOrStack,
  fallbackChain = DEFAULT_FONT_FALLBACK_CHAIN
) {
  const primaryParts = Array.isArray(primaryOrStack)
    ? primaryOrStack
    : parseFontFamilyList(primaryOrStack);
  const configuredChain =
    Array.isArray(fallbackChain) && fallbackChain.length > 0
      ? fallbackChain
      : DEFAULT_FONT_FALLBACK_CHAIN;

  // Keep non-generic fonts ahead of generic CSS keywords (`sans-serif`, `serif`, etc.)
  const named = [];
  const generics = [];
  const seen = new Set();

  const pushFont = (raw) => {
    const cleaned = String(raw || '').trim().replace(/^["']+|["']+$/g, '').trim();
    if (!cleaned) return;
    const lower = cleaned.toLowerCase();
    if (seen.has(lower)) return;
    seen.add(lower);
    if (GENERIC_CSS_FONTS.has(lower)) {
      generics.push(lower);
    } else {
      named.push(cleaned);
    }
  };

  primaryParts.forEach(pushFont);
  configuredChain.forEach(pushFont);
  MANGA_SYMBOL_FALLBACK_FONTS.forEach(pushFont);

  if (generics.length === 0) {
    generics.push('sans-serif');
  }

  return formatFontFamilyChain([...named, ...generics]);
}

/**
 * Resolve the default backend API base URL:
 * - Honors `VITE_API_BASE_URL` when explicitly provided.
 * - Defaults to `''` (relative `/api/...` paths) so production builds served directly
 *   by the native C++17 HTTP backend (e.g. inside Microsoft Edge WebView2 on any loopback port) and
 *   Vite dev builds (via the `/api` proxy in `vite.config.js`) both work seamlessly
 *   against the active origin with zero hardcoded host/port coupling.
 */
export function resolveDefaultApiBaseUrl() {
  const envBase =
    typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE_URL
      ? String(import.meta.env.VITE_API_BASE_URL).trim().replace(/\/+$/, '')
      : '';
  return envBase || '';
}

export const DEFAULT_API_BASE_URL = resolveDefaultApiBaseUrl();

const LOADED_FONT_FACE_KEYS = new Set();
const IN_FLIGHT_FONT_LOADS = new Map();

function resolveFontAssetUrl(fontUrl, apiBaseUrl = DEFAULT_API_BASE_URL) {
  if (!fontUrl) return null;
  if (/^(https?:|data:|blob:)/i.test(fontUrl)) return fontUrl;
  const base = String(apiBaseUrl || '').replace(/\/+$/, '');
  return `${base}${fontUrl.startsWith('/') ? fontUrl : `/${fontUrl}`}`;
}

/**
 * Dynamically load a font binary from `/api/fonts/file/{font_name}` into `document.fonts`
 * using the browser `FontFace` API. For single-weight comic fonts, registers both normal (400)
 * and bold (700) descriptors so Konva `fontStyle="bold"` renders crisp native glyphs in
 * Chromium/WebView2 without falling back to system fonts.
 */
export async function ensureFontFaceLoaded(
  family,
  fontUrl,
  apiBaseUrl = DEFAULT_API_BASE_URL,
  { weight = null, italic = false, registerBoldAlias = true } = {}
) {
  if (
    typeof window === 'undefined' ||
    typeof FontFace === 'undefined' ||
    !family ||
    !fontUrl
  ) {
    return false;
  }
  const cleanFamily = String(family).trim().replace(/^["']+|["']+$/g, '');
  if (!cleanFamily || GENERIC_CSS_FONTS.has(cleanFamily.toLowerCase())) {
    return false;
  }
  const resolvedUrl = resolveFontAssetUrl(fontUrl, apiBaseUrl);
  if (!resolvedUrl) return false;

  const weightList = weight
    ? [String(weight)]
    : registerBoldAlias
    ? ['400', '700']
    : ['400'];
  const styleStr = italic ? 'italic' : 'normal';
  const cacheKey = `${cleanFamily.toLowerCase()}::${resolvedUrl}::${weightList.join(',')}::${styleStr}`;

  if (LOADED_FONT_FACE_KEYS.has(cacheKey)) return true;
  if (IN_FLIGHT_FONT_LOADS.has(cacheKey)) {
    return IN_FLIGHT_FONT_LOADS.get(cacheKey);
  }

  const loadPromise = (async () => {
    try {
      let addedAny = false;
      for (const w of weightList) {
        const descriptor = { weight: w, style: styleStr };
        const face = new FontFace(cleanFamily, `url("${resolvedUrl}")`, descriptor);
        const loaded = await face.load();
        document.fonts.add(loaded);
        addedAny = true;
      }
      if (addedAny) {
        LOADED_FONT_FACE_KEYS.add(cacheKey);
        // Trigger immediate Konva text re-measurement & layer batchDraw
        if (useEditorStore?.getState?.()?.bumpFontLoadVersion) {
          useEditorStore.getState().bumpFontLoadVersion();
        }
      }
      return addedAny;
    } catch {
      return false;
    } finally {
      IN_FLIGHT_FONT_LOADS.delete(cacheKey);
    }
  })();

  IN_FLIGHT_FONT_LOADS.set(cacheKey, loadPromise);
  return loadPromise;
}

/**
 * Load all relevant variants (or single-weight regular+bold alias) for a font family
 * from `/api/fonts/list` metadata so both normal preview and bold Konva canvas rendering match 1:1.
 */
export async function loadFontFamilyFromCatalog(
  familyGroupOrName,
  fontsCatalog = null,
  apiBaseUrl = DEFAULT_API_BASE_URL
) {
  if (!familyGroupOrName) return false;

  let group = null;
  if (typeof familyGroupOrName === 'string') {
    const clean = familyGroupOrName.trim().replace(/^["']+|["']+$/g, '');
    if (!clean || GENERIC_CSS_FONTS.has(clean.toLowerCase())) return false;
    const families = fontsCatalog?.families || [];
    group =
      families.find((f) => f.family?.toLowerCase() === clean.toLowerCase()) ||
      null;
    if (!group) {
      const fonts = fontsCatalog?.fonts || [];
      const single = fonts.find(
        (f) => f.family?.toLowerCase() === clean.toLowerCase()
      );
      if (single?.url) {
        return ensureFontFaceLoaded(single.family, single.url, apiBaseUrl, {
          registerBoldAlias: true,
        });
      }
      return false;
    }
  } else {
    group = familyGroupOrName;
  }

  const familyName = group.family;
  const variants = Array.isArray(group.variants) ? group.variants : [];
  if (variants.length === 0) {
    if (group.url) {
      return ensureFontFaceLoaded(familyName, group.url, apiBaseUrl, {
        registerBoldAlias: true,
      });
    }
    return false;
  }

  const regularVariant =
    variants.find((v) => !v.italic && (v.weight === 400 || v.weight === 500)) ||
    variants.find((v) => !v.italic) ||
    variants[0];
  const boldVariant = variants.find((v) => !v.italic && v.weight >= 600) || null;
  const italicVariant =
    variants.find((v) => v.italic && (v.weight === 400 || v.weight === 500)) ||
    variants.find((v) => v.italic && v.weight < 600) ||
    null;
  const boldItalicVariant =
    variants.find((v) => v.italic && v.weight >= 600) || null;

  const jobs = [];
  if (boldVariant && regularVariant && boldVariant.url !== regularVariant.url) {
    jobs.push(
      ensureFontFaceLoaded(familyName, regularVariant.url, apiBaseUrl, {
        weight: '400',
        italic: false,
        registerBoldAlias: false,
      })
    );
    jobs.push(
      ensureFontFaceLoaded(familyName, boldVariant.url, apiBaseUrl, {
        weight: '700',
        italic: false,
        registerBoldAlias: false,
      })
    );
  } else if (regularVariant?.url) {
    // Register both 400 and 700 descriptors for single-weight Roman/Regular font files
    // to prevent synthetic faux-bold clipping on Konva canvas & offscreen exports.
    jobs.push(
      ensureFontFaceLoaded(familyName, regularVariant.url, apiBaseUrl, {
        italic: Boolean(regularVariant.italic),
        registerBoldAlias: true,
      })
    );
  }

  if (italicVariant?.url && italicVariant.url !== regularVariant?.url) {
    jobs.push(
      ensureFontFaceLoaded(familyName, italicVariant.url, apiBaseUrl, {
        weight: '400',
        italic: true,
        registerBoldAlias: !boldItalicVariant,
      })
    );
  }

  if (
    boldItalicVariant?.url &&
    boldItalicVariant.url !== regularVariant?.url &&
    boldItalicVariant.url !== italicVariant?.url
  ) {
    jobs.push(
      ensureFontFaceLoaded(familyName, boldItalicVariant.url, apiBaseUrl, {
        weight: '700',
        italic: true,
        registerBoldAlias: false,
      })
    );
  }

  // Also load any remaining distinct variant files in the family group
  for (const v of variants) {
    if (
      v?.url &&
      v.url !== regularVariant?.url &&
      v.url !== boldVariant?.url &&
      v.url !== italicVariant?.url &&
      v.url !== boldItalicVariant?.url
    ) {
      jobs.push(
        ensureFontFaceLoaded(familyName, v.url, apiBaseUrl, {
          weight: String(v.weight || 400),
          italic: Boolean(v.italic),
          registerBoldAlias: false,
        })
      );
    }
  }

  const results = await Promise.all(jobs);
  return results.some(Boolean);
}

/**
 * Ensure all fonts referenced by a page's blocks and the active fallback chain
 * are loaded into `document.fonts` prior to 1:1 PNG / PDF / CBZ offscreen export.
 */
export async function ensurePageFontsLoaded(
  page,
  fontsCatalog = null,
  apiBaseUrl = DEFAULT_API_BASE_URL
) {
  if (typeof document === 'undefined' || !document.fonts) return;
  const state = useEditorStore.getState();
  const catalog = fontsCatalog || state.fontsCatalogCache;
  const chain = state.typesetting?.fontFallbackChain || DEFAULT_FONT_FALLBACK_CHAIN;

  const names = new Set();
  chain.forEach((f) => names.add(f));
  (page?.blocks || []).forEach((b) => {
    const raw = b?.style?.font_family || b?.font_family || '';
    parseFontFamilyList(raw).forEach((f) => names.add(f));
  });

  if (catalog) {
    await Promise.all(
      Array.from(names).map((fam) =>
        loadFontFamilyFromCatalog(fam, catalog, apiBaseUrl).catch(() => false)
      )
    );
  }
  try {
    await document.fonts.ready;
  } catch {
    // Ignore FontFaceSet errors
  }
}

export const DEFAULT_VIEW_LAYERS = {
  original: true,
  inpainted: true,
  mask: false,
  text: true,
  boxes: true,
  compareOriginal: false,
  splitCompare: false,
};

const CANVAS_LAYERS_STORAGE_KEY = 'vellum.canvas_layers.v1';

function loadPersistedViewLayers() {
  try {
    if (typeof window === 'undefined' || !window.localStorage) {
      return { ...DEFAULT_VIEW_LAYERS };
    }
    const raw = window.localStorage.getItem(CANVAS_LAYERS_STORAGE_KEY);
    if (!raw) return { ...DEFAULT_VIEW_LAYERS };
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') {
      return { ...DEFAULT_VIEW_LAYERS };
    }
    return {
      ...DEFAULT_VIEW_LAYERS,
      original: parsed.original !== undefined ? Boolean(parsed.original) : DEFAULT_VIEW_LAYERS.original,
      inpainted: parsed.inpainted !== undefined ? Boolean(parsed.inpainted) : DEFAULT_VIEW_LAYERS.inpainted,
      text: parsed.text !== undefined ? Boolean(parsed.text) : DEFAULT_VIEW_LAYERS.text,
      boxes: parsed.boxes !== undefined ? Boolean(parsed.boxes) : DEFAULT_VIEW_LAYERS.boxes,
      mask: parsed.mask !== undefined ? Boolean(parsed.mask) : DEFAULT_VIEW_LAYERS.mask,
      compareOriginal: false,
      splitCompare: false,
    };
  } catch {
    return { ...DEFAULT_VIEW_LAYERS };
  }
}

function savePersistedViewLayers(layers) {
  try {
    if (typeof window === 'undefined' || !window.localStorage || !layers) return;
    window.localStorage.setItem(
      CANVAS_LAYERS_STORAGE_KEY,
      JSON.stringify({
        original: Boolean(layers.original),
        inpainted: Boolean(layers.inpainted),
        text: Boolean(layers.text),
        boxes: Boolean(layers.boxes),
        mask: Boolean(layers.mask),
      })
    );
  } catch {
    // Ignore localStorage quota/security errors
  }
}

export const DEFAULT_ENABLED_STAGES = {
  detect: true,
  ocr: true,
  inpaint: true,
  translate: true,
};

export const DEFAULT_TARGET_LANGUAGES = [
  { code: 'id', name: 'Indonesian', native_name: 'Bahasa Indonesia', rtl: false },
  { code: 'en', name: 'English', native_name: 'English', rtl: false },
  { code: 'ja', name: 'Japanese', native_name: '日本語', rtl: false },
  { code: 'ko', name: 'Korean', native_name: '한국어', rtl: false },
  { code: 'zh-CN', name: 'Chinese (Simplified)', native_name: '简体中文', rtl: false },
  { code: 'zh-TW', name: 'Chinese (Traditional)', native_name: '繁體中文', rtl: false },
  { code: 'es', name: 'Spanish', native_name: 'Español', rtl: false },
  { code: 'fr', name: 'French', native_name: 'Français', rtl: false },
  { code: 'de', name: 'German', native_name: 'Deutsch', rtl: false },
  { code: 'pt', name: 'Portuguese', native_name: 'Português', rtl: false },
  { code: 'ru', name: 'Russian', native_name: 'Русский', rtl: false },
  { code: 'vi', name: 'Vietnamese', native_name: 'Tiếng Việt', rtl: false },
  { code: 'th', name: 'Thai', native_name: 'ไทย', rtl: false },
  { code: 'ms', name: 'Malay', native_name: 'Bahasa Melayu', rtl: false },
  { code: 'ar', name: 'Arabic', native_name: 'العربية', rtl: true },
  { code: 'it', name: 'Italian', native_name: 'Italiano', rtl: false },
];

export const DEFAULT_DETECTION_MODELS = [
  {
    id: 'vellum-det-yolo11m-seg-1024-fp16',
    label: 'YOLO11m-Seg · High-Speed GPU',
    name: 'YOLO11m-Seg · High-Speed GPU',
    description:
      'Primary 9-class 1024px layout & bubble segmentation (NVIDIA CUDA / Universal GPU FP16)',
  },
  {
    id: 'vellum-det-yolo11m-seg-1024-int8',
    label: 'YOLO11m-Seg · Lightweight CPU',
    name: 'YOLO11m-Seg · Lightweight CPU',
    description:
      'Static QDQ INT8 9-class 1024px detector optimized for CPU-only desktops',
  },
  {
    id: 'comic-text-detector',
    label: 'Comic Text Detector (Legacy)',
    name: 'Comic Text Detector (Legacy)',
    description:
      'Fast 1024px manga/manhwa/manhua text box & segmentation mask detector',
  },
];

export const DEFAULT_OCR_MODELS = [
  {
    id: 'auto',
    label: 'Auto (Language Router)',
    name: 'Auto (Language Router)',
    hardwareTag: 'CPU • Auto Router',
    description:
      'Automatically routes Japanese → Manga OCR and Korean/Chinese → RapidOCR / Hayai OCR',
  },
  {
    id: 'paddleocr-vl-1.6',
    label: 'PaddleOCR-VL 1.6 (VLM • Multi-Lang)',
    name: 'PaddleOCR-VL 1.6 (VLM • Multi-Lang)',
    hardwareTag: 'VLM • Multi-Lang',
    description:
      'Multilingual Vision-Language OCR via llama-server / LM Studio or llama-mtmd-cli worker',
  },
  {
    id: 'manga-ocr',
    label: 'Manga OCR (CPU • Japanese)',
    name: 'Manga OCR (CPU • Japanese)',
    hardwareTag: 'CPU • Japanese',
    description:
      'ViT + BERT autoregressive Japanese manga OCR (vertical tategaki & furigana)',
  },
  {
    id: 'baberu-ocr',
    label: 'Baberu OCR (CPU • JA/ZH/EN Fast)',
    name: 'Baberu OCR (CPU • JA/ZH/EN Fast)',
    hardwareTag: 'CPU • JA/ZH/EN Fast',
    description:
      'Compact DINOv2 + INT8 KV-cache autoregressive OCR for Japanese, Chinese & English',
  },
  {
    id: 'hayai-ocr',
    label: 'Hayai OCR (GPU/CPU • JA/ZH/KO/EN)',
    name: 'Hayai OCR (GPU/CPU • JA/ZH/KO/EN)',
    hardwareTag: 'GPU/CPU • JA/ZH/KO/EN',
    description:
      'High-throughput 150M SigLIP2 + 2D-mRoPE causal transformer OCR across JA, ZH, KO, EN',
  },
  {
    id: 'rapidocr-ppocrv4',
    label: 'RapidOCR PP-OCRv4 (CPU • KO/ZH)',
    name: 'RapidOCR PP-OCRv4 (CPU • KO/ZH)',
    hardwareTag: 'CPU • KO/ZH',
    description:
      'Fast CJK/Latin PP-OCRv4 ONNX recognition (Korean, Chinese & horizontal Japanese)',
  },
];

export const DEFAULT_INPAINTING_MODELS = [
  {
    id: 'lama-manga',
    label: 'LaMa · Manga & Screentone',
    name: 'LaMa · Manga & Screentone',
    hardwareTag: 'GPU • Screentone',
    description:
      'Manga-tuned Big-LaMa ONNX inpainter optimized for screentones & line art',
  },
  {
    id: 'aot-inpainting',
    label: 'AOT Inpainting (GPU • Ultra-Fast)',
    name: 'AOT Inpainting (GPU • Ultra-Fast)',
    hardwareTag: 'GPU • Ultra-Fast',
    description:
      'Lightweight 10-block AOT-GAN manga text-stroke inpainter (4-ch mask+RGB → 3-ch RGB, 512×512)',
  },
  {
    id: 'flux-2-klein',
    label: 'FLUX.2 Klein (Generative • 8GB+ VRAM)',
    name: 'FLUX.2 Klein (Generative • 8GB+ VRAM)',
    hardwareTag: 'Generative • 8GB+ VRAM',
    description:
      'Generative diffusion text-stroke ROI inpainter via local ComfyUI/A1111/sd-server adapter',
  },
  {
    id: 'rorem-mixed',
    label: 'RORem Mixed (Generative • Manga GGUF)',
    name: 'RORem Mixed (Generative • Manga GGUF)',
    hardwareTag: 'Generative • Manga GGUF',
    description:
      'Manga object/text removal diffusion adapter via local diffusion worker',
  },
  {
    id: 'lama-fp32',
    label: 'LaMa · High-Precision Universal',
    name: 'LaMa · High-Precision Universal',
    hardwareTag: 'GPU • FP32',
    description: 'General-purpose Big-LaMa FP32 ONNX inpainter (512×512 tiles)',
  },
];

export function normalizeOptionalThreshold(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) return null;
    const num = Number(trimmed);
    if (!Number.isFinite(num)) return null;
    const clamped = Math.min(0.95, Math.max(0.05, num));
    return Math.round(clamped * 10000) / 10000;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return null;
    const clamped = Math.min(0.95, Math.max(0.05, value));
    return Math.round(clamped * 10000) / 10000;
  }
  return null;
}

export const DEFAULT_PIPELINE_CONFIG = {
  apiBaseUrl: DEFAULT_API_BASE_URL,
  provider: 'none',
  baseUrl: '',
  apiKey: '',
  model: '',
  providerProfiles: buildDefaultProviderProfiles(),
  sourceLang: 'ja',
  targetLang: 'id',
  translationTone: 'standard', // 'standard' | 'casual' | 'adult_doujinshi' | 'formal' | 'shonen' | 'romcom'
  honorificsPolicy: 'keep', // 'keep' | 'localize' | 'drop'
  scope: 'current_page', // 'current_page' | 'selected_pages' | 'project'
  enabledStages: { ...DEFAULT_ENABLED_STAGES },
  temperature: 0.2,
  topP: 0.95,
  topK: 40,
  maxTokens: 2048,
  frequencyPenalty: 0.0,
  reasoningEffort: 'off',
  stripThinking: true,
  timeout: 90,
  useVision: false,
  visionMaxSide: 1024,
  customInstructions: '',
  glossary: {},
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
  executionMode: 'hybrid', // 'hybrid' | 'force_gpu' | 'cpu_only'
  preferredBackend: 'auto', // 'auto' | 'cuda' | 'directml' | 'cpu'
  confThreshold: 0.35,
  nmsThreshold: 0.45,
  maskThreshold: 0.5,
  bboxPadding: 6,
  lamaTileSize: 512,
  readingOrder: 'auto', // 'auto' | 'rtl' | 'ltr'
  autoUnloadVram: true,
  batchScheduleMode: 'auto', // 'auto' | 'pipelined' | 'stage-first'
  refineMask: true,
  autoProcessOnImport: false,
  enable3TierInpaint: true,
  enableToneMatch: true,
  enableCvStyleDetect: true,
  enableWaistCut: true,
  autoDeskewScans: false,
  autoMaskFuriganaOcr: true,
  skipBlankPages: true,
  smartSsimExport: true,
};

function computeBoxIntersectionArea(a, b) {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  if (ix2 <= ix1 || iy2 <= iy1) return 0;
  return (ix2 - ix1) * (iy2 - iy1);
}

function findContainingRegion(textBox, regions = [], minOverlap = 0.35) {
  if (!Array.isArray(regions) || regions.length === 0) return -1;
  const tw = Math.max(1, textBox[2] - textBox[0]);
  const th = Math.max(1, textBox[3] - textBox[1]);
  const textArea = tw * th;
  const cx = (textBox[0] + textBox[2]) * 0.5;
  const cy = (textBox[1] + textBox[3]) * 0.5;

  let bestIdx = -1;
  let bestScore = -1;
  let bestArea = Infinity;

  for (let i = 0; i < regions.length; i += 1) {
    const rb = regions[i]?.bbox;
    if (!Array.isArray(rb) || rb.length < 4) continue;
    const bx1 = Number(rb[0]) || 0;
    const by1 = Number(rb[1]) || 0;
    const bx2 = Number(rb[2]) || 0;
    const by2 = Number(rb[3]) || 0;
    const bw = bx2 - bx1;
    const bh = by2 - by1;
    if (bw <= 4 || bh <= 4) continue;

    const inter = computeBoxIntersectionArea(textBox, [bx1, by1, bx2, by2]);
    const overlap = inter / textArea;
    const centerInside =
      cx >= bx1 - 6 && cx <= bx2 + 6 && cy >= by1 - 6 && cy <= by2 + 6;
    if (overlap < minOverlap && !(centerInside && overlap >= 0.15)) {
      continue;
    }

    const area = bw * bh;
    if (
      overlap > bestScore + 0.05 ||
      (Math.abs(overlap - bestScore) <= 0.05 && area < bestArea)
    ) {
      bestIdx = i;
      bestScore = overlap;
      bestArea = area;
    }
  }
  return bestIdx;
}

export function rgbColorToHex(value) {
  if (!value) return null;
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : null;
  }
  if (Array.isArray(value) && value.length >= 3) {
    const r = Math.max(0, Math.min(255, Math.round(Number(value[0]) || 0)));
    const g = Math.max(0, Math.min(255, Math.round(Number(value[1]) || 0)));
    const b = Math.max(0, Math.min(255, Math.round(Number(value[2]) || 0)));
    return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
  }
  return null;
}

/**
 * Normalize raw dialogue blocks returned from `/api/process-page` or `/api/stage/*`
 * into canvas-ready block objects with explicit geometry & typography styles.
 */
const CJK_KANA_HANGUL_RE =
  /[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF\uFF66-\uFF9F\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF]/;

export function isNonCjkTargetLang(targetLang) {
  const t = String(targetLang || 'id').trim().toLowerCase();
  return !['ja', 'zh', 'zh-cn', 'zh-tw', 'ko'].includes(t);
}

export function isBlockCjkLeaked(block, targetLang) {
  if (!block || !isNonCjkTargetLang(targetLang)) return false;
  const src = String(block.original_text || block.source_text || '').trim();
  const trans = String(block.translated_text || '').trim();
  if (!src || !trans) return false;
  return CJK_KANA_HANGUL_RE.test(trans);
}

export function getPageQcFlags(page, fallbackTargetLang = 'id') {
  if (!page) {
    return {
      echoRetried: false,
      leakedBlockIds: [],
      lowConfidenceOcrIds: [],
      anomalyBlockIds: [],
      hasCjkLeak: false,
      hasEmptyBlock: false,
      hasLowOcrConfidence: false,
      hasTranslationAnomaly: false,
      hasIssue: false,
    };
  }
  const tgt = page.targetLang || fallbackTargetLang;
  const blocks = Array.isArray(page.blocks) ? page.blocks : [];
  const leakedBlockIds = Array.isArray(page.leakedBlockIds) ? page.leakedBlockIds : [];
  const lowConfidenceOcrIds = Array.isArray(page.lowConfidenceOcrIds)
    ? page.lowConfidenceOcrIds
    : blocks.filter((b) => Boolean(b?.low_ocr_confidence)).map((b) => b.id);
  const anomalyBlockIds = Array.isArray(page.anomalyBlockIds) ? page.anomalyBlockIds : [];
  const echoRetried = Boolean(page.echoRetried);
  const hasCjkLeak =
    leakedBlockIds.length > 0 ||
    blocks.some((b) => isBlockCjkLeaked(b, tgt));
  const hasEmptyBlock =
    page.status === 'done' &&
    blocks.some((b) => {
      const src = String(b?.original_text || b?.source_text || '').trim();
      const trans = String(b?.translated_text || '').trim();
      return Boolean(src) && !trans;
    });
  const hasLowOcrConfidence = lowConfidenceOcrIds.length > 0;
  const hasTranslationAnomaly = anomalyBlockIds.length > 0;
  return {
    echoRetried,
    leakedBlockIds,
    lowConfidenceOcrIds,
    anomalyBlockIds,
    hasCjkLeak,
    hasEmptyBlock,
    hasLowOcrConfidence,
    hasTranslationAnomaly,
    hasIssue:
      echoRetried ||
      hasCjkLeak ||
      hasEmptyBlock ||
      hasLowOcrConfidence ||
      hasTranslationAnomaly,
  };
}

export function resolveSemanticFontStyle(className) {
  const cls = String(className || '').trim().toLowerCase();
  if (
    cls === 'thoughtbubble' ||
    cls === 'thought' ||
    cls === 'whisperbubble' ||
    cls === 'whisper'
  ) {
    return 'italic';
  }
  if (cls === 'shoutbubble' || cls === 'shout' || cls === 'scream') {
    return 'bold';
  }
  return 'normal';
}

export function normalizeBlocks(rawBlocks = [], options = null) {
  const imgWidth = Number(options?.imgWidth) || 0;
  const targetLang = String(options?.targetLang || 'id').toLowerCase();
  const isHorizontalTarget = !['ja', 'ja-vert'].includes(targetLang);
  const layoutBubbles = Array.isArray(options?.layoutBubbles)
    ? options.layoutBubbles
    : [];
  const panels = Array.isArray(options?.panels) ? options.panels : [];

  const list = rawBlocks || [];
  const matchedBubbleIndices = list.map((b) => {
    const bbox =
      Array.isArray(b.bbox) && b.bbox.length >= 4
        ? b.bbox
        : [60, 60, 220, 140];
    const x1 = Number(bbox[0]) || 0;
    const y1 = Number(bbox[1]) || 0;
    const x2 = Number(bbox[2]) || x1 + 140;
    const y2 = Number(bbox[3]) || y1 + 60;
    return findContainingRegion([x1, y1, x2, y2], layoutBubbles, 0.35);
  });

  const bubbleUsageCount = new Map();
  for (const idx of matchedBubbleIndices) {
    if (idx >= 0) {
      bubbleUsageCount.set(idx, (bubbleUsageCount.get(idx) || 0) + 1);
    }
  }

  const normalized = list.map((b, index) => {
    const bbox =
      Array.isArray(b.bbox) && b.bbox.length >= 4
        ? b.bbox
        : [60, 60, 220, 140];
    const x1 = Number(bbox[0]) || 0;
    const y1 = Number(bbox[1]) || 0;
    const x2 = Number(bbox[2]) || x1 + 140;
    const y2 = Number(bbox[3]) || y1 + 60;
    const bboxWidth = Math.max(36, x2 - x1);
    const bboxHeight = Math.max(24, y2 - y1);

    let defaultX = x1;
    let defaultY = y1;
    let defaultWidth = bboxWidth;
    let defaultHeight = bboxHeight;

    const bubbleIdx = matchedBubbleIndices[index];
    const matchedBubble = bubbleIdx >= 0 ? layoutBubbles[bubbleIdx] : null;
    const panelIdx = findContainingRegion([x1, y1, x2, y2], panels, 0.45);
    const matchedPanel = panelIdx >= 0 ? panels[panelIdx] : null;

    const backendSafeBox =
      Array.isArray(b.safe_bbox) &&
      b.safe_bbox.length >= 4 &&
      Number(b.safe_bbox[2]) > Number(b.safe_bbox[0]) &&
      Number(b.safe_bbox[3]) > Number(b.safe_bbox[1])
        ? [
            Number(b.safe_bbox[0]) || 0,
            Number(b.safe_bbox[1]) || 0,
            Number(b.safe_bbox[2]) || 0,
            Number(b.safe_bbox[3]) || 0,
          ]
        : null;

    const rawBubbleBox =
      Array.isArray(b.bubble_bbox) &&
      b.bubble_bbox.length >= 4 &&
      Number(b.bubble_bbox[2]) > Number(b.bubble_bbox[0]) &&
      Number(b.bubble_bbox[3]) > Number(b.bubble_bbox[1])
        ? [
            Number(b.bubble_bbox[0]) || 0,
            Number(b.bubble_bbox[1]) || 0,
            Number(b.bubble_bbox[2]) || 0,
            Number(b.bubble_bbox[3]) || 0,
          ]
        : matchedBubble && Array.isArray(matchedBubble.bbox) && matchedBubble.bbox.length >= 4
        ? [
            Number(matchedBubble.bbox[0]) || 0,
            Number(matchedBubble.bbox[1]) || 0,
            Number(matchedBubble.bbox[2]) || 0,
            Number(matchedBubble.bbox[3]) || 0,
          ]
        : null;

    if (
      isHorizontalTarget &&
      backendSafeBox &&
      (bubbleIdx < 0 || (bubbleUsageCount.get(bubbleIdx) || 1) === 1)
    ) {
      // Strictly constrain layout box to backend safe_bbox
      defaultX = backendSafeBox[0];
      defaultY = backendSafeBox[1];
      defaultWidth = Math.max(20, backendSafeBox[2] - backendSafeBox[0]);
      defaultHeight = Math.max(20, backendSafeBox[3] - backendSafeBox[1]);
    } else if (isHorizontalTarget && rawBubbleBox) {
      // Fallback: inset bubble_bbox by 8% padding on all sides (never wider than the bubble)
      const [bx1, by1, bx2, by2] = rawBubbleBox;
      const bw = Math.max(20, bx2 - bx1);
      const bh = Math.max(20, by2 - by1);
      const padX = Math.round(bw * 0.08);
      const padY = Math.round(bh * 0.08);
      const innerX1 = bx1 + padX;
      const innerX2 = Math.max(innerX1 + 16, bx2 - padX);
      const innerY1 = by1 + padY;
      const innerY2 = Math.max(innerY1 + 16, by2 - padY);
      const innerW = Math.max(16, innerX2 - innerX1);
      const innerH = Math.max(16, innerY2 - innerY1);

      if (bubbleIdx < 0 || (bubbleUsageCount.get(bubbleIdx) || 1) === 1) {
        defaultX = innerX1;
        defaultY = innerY1;
        defaultWidth = innerW;
        defaultHeight = innerH;
      } else {
        // Multiple blocks sharing one bubble: clamp strictly inside inner bubble bounds
        defaultWidth = Math.max(16, Math.min(bboxWidth, innerW));
        defaultX = Math.max(
          innerX1,
          Math.min(x1, innerX2 - defaultWidth)
        );
        defaultY = Math.max(innerY1, Math.min(y1, innerY2 - 16));
        defaultHeight = Math.max(16, Math.min(bboxHeight, innerY2 - defaultY));
      }
    } else if (matchedPanel && Array.isArray(matchedPanel.bbox)) {
      // Clamp unbubbled text strictly inside its containing panel
      const px1 = (Number(matchedPanel.bbox[0]) || 0) + 4;
      const px2 = (Number(matchedPanel.bbox[2]) || px1 + bboxWidth) - 4;
      const panelAvailW = Math.max(20, px2 - px1);
      defaultWidth = Math.min(bboxWidth, panelAvailW);
      defaultX = Math.max(px1, Math.min(x1, px2 - defaultWidth));
    }

    const detectedTextColor = rgbColorToHex(b.text_color);
    const detectedOutlineObj =
      b.outline && typeof b.outline === 'object' ? b.outline : null;
    const detectedOutlineColor = detectedOutlineObj
      ? rgbColorToHex(detectedOutlineObj.color)
      : null;
    const detectedOutlineWidth =
      detectedOutlineObj && Number.isFinite(Number(detectedOutlineObj.width))
        ? Math.max(0, Number(detectedOutlineObj.width))
        : null;
    const detectedAngleDeg =
      typeof b.angle_deg === 'number' && Number.isFinite(b.angle_deg)
        ? Number(b.angle_deg.toFixed(2))
        : null;

    const fontSize = b.font_size || b.style?.font_size || 18;
    const color = b._userEditedColor
      ? b.color || b.style?.color || '#000000'
      : detectedTextColor || b.color || b.style?.color || '#000000';
    const align = b.align || b.style?.align || 'center';
    const fontFamily =
      b.font_family || b.style?.font_family || DEFAULT_FONT_FAMILY;
    const strokeColor = b._userEditedOutline
      ? b.stroke_color || b.style?.stroke_color || '#ffffff'
      : detectedOutlineColor || b.stroke_color || b.style?.stroke_color || '#ffffff';
    const strokeWidth = b._userEditedOutline
      ? b.stroke_width !== undefined
        ? b.stroke_width
        : b.style?.stroke_width !== undefined
        ? b.style.stroke_width
        : 4
      : detectedOutlineWidth !== null
      ? detectedOutlineWidth
      : b.stroke_width !== undefined
      ? b.stroke_width
      : b.style?.stroke_width !== undefined
      ? b.style.stroke_width
      : 4;
    const lineHeight = b.line_height || b.style?.line_height || 1.18;
    const explicitFontStyle = b.style?.font_style || b.font_style || null;
    const fontStyle = explicitFontStyle || resolveSemanticFontStyle(b.class_name || b.bubble_class);
    const resolvedRotation =
      typeof b.rotation === 'number'
        ? b.rotation
        : detectedAngleDeg !== null
        ? detectedAngleDeg
        : 0;

    return {
      ...b,
      id: b.id || `block_${String(index + 1).padStart(3, '0')}`,
      order: b.order ?? index + 1,
      bbox: [x1, y1, x1 + bboxWidth, y1 + bboxHeight],
      bubble_bbox: rawBubbleBox ? [...rawBubbleBox] : b.bubble_bbox,
      safe_bbox: backendSafeBox ? [...backendSafeBox] : b.safe_bbox,
      panel_bbox: matchedPanel?.bbox ? [...matchedPanel.bbox] : b.panel_bbox,
      polygon: Array.isArray(b.polygon)
        ? b.polygon.map((pt) => (Array.isArray(pt) ? [...pt] : pt))
        : undefined,
      room_polygon: Array.isArray(b.room_polygon) && b.room_polygon.length >= 3
        ? b.room_polygon.map((pt) => (Array.isArray(pt) ? [Number(pt[0]) || 0, Number(pt[1]) || 0] : pt))
        : undefined,
      text_color: Array.isArray(b.text_color) ? [...b.text_color] : b.text_color,
      outline: detectedOutlineObj
        ? {
            color: Array.isArray(detectedOutlineObj.color)
              ? [...detectedOutlineObj.color]
              : detectedOutlineObj.color,
            width: detectedOutlineWidth ?? 0,
          }
        : b.outline,
      angle_deg: detectedAngleDeg !== null ? detectedAngleDeg : b.angle_deg,
      x: typeof b.x === 'number' ? b.x : defaultX,
      y: typeof b.y === 'number' ? b.y : defaultY,
      width: typeof b.width === 'number' ? b.width : defaultWidth,
      height: typeof b.height === 'number' ? b.height : defaultHeight,
      rotation: resolvedRotation,
      original_text: b.original_text ?? b.source_text ?? '',
      source_text: b.source_text ?? b.original_text ?? '',
      translated_text: b.translated_text ?? '',
      font_family: fontFamily,
      font_size: fontSize,
      font_style: fontStyle,
      color,
      stroke_color: strokeColor,
      stroke_width: strokeWidth,
      align,
      line_height: lineHeight,
      style: {
        font_family: fontFamily,
        font_size: fontSize,
        font_style: fontStyle,
        color,
        stroke_color: strokeColor,
        stroke_width: strokeWidth,
        align,
        line_height: lineHeight,
      },
      _autoSized: typeof b.x !== 'number' && typeof b.width !== 'number',
    };
  });

  // Prevent adjacent auto-expanded boxes (e.g. side-by-side floating text outside bubbles)
  // from overlapping each other horizontally.
  const gap = 4;
  for (let i = 0; i < normalized.length; i += 1) {
    const a = normalized[i];
    if (!a._autoSized) continue;
    for (let j = i + 1; j < normalized.length; j += 1) {
      const b = normalized[j];
      if (!b._autoSized) continue;

      const ay1 = a.y;
      const ay2 = a.y + a.height;
      const by1 = b.y;
      const by2 = b.y + b.height;
      const vOverlap = Math.min(ay2, by2) - Math.max(ay1, by1);
      if (vOverlap <= Math.min(a.height, b.height) * 0.25) continue;

      const aOrigCenter = (a.bbox[0] + a.bbox[2]) * 0.5;
      const bOrigCenter = (b.bbox[0] + b.bbox[2]) * 0.5;
      const left = aOrigCenter <= bOrigCenter ? a : b;
      const right = aOrigCenter <= bOrigCenter ? b : a;

      if (left.x + left.width > right.x - gap && left.bbox[2] <= right.bbox[0] + 4) {
        const mid = Math.round((left.bbox[2] + right.bbox[0]) * 0.5);
        const leftMaxX2 = Math.max(left.bbox[2], mid - Math.floor(gap / 2));
        const rightMinX1 = Math.min(right.bbox[0], mid + Math.ceil(gap / 2));

        const newLeftW = Math.max(24, leftMaxX2 - left.x);
        if (newLeftW < left.width) {
          left.width = newLeftW;
        }
        const rightEnd = right.x + right.width;
        if (rightMinX1 > right.x) {
          const newRightW = Math.max(24, rightEnd - rightMinX1);
          right.x = rightMinX1;
          right.width = newRightW;
        }
      }
    }
  }

  // #14A: Spring-Mass Anti-Overlap Layout (12-step projected gradient descent)
  const imgHeight = Number(options?.imgHeight) || 0;
  if (imgWidth > 0 && imgHeight > 0 && normalized.length > 1) {
    const forEnergy = normalized.map((item) => ({
      ...item,
      manualPosition: !item._autoSized || Boolean(item.manualPosition),
    }));
    minimizeBoxOverlapEnergy(forEnergy, imgWidth, imgHeight, { steps: 12 });
    for (let i = 0; i < normalized.length; i += 1) {
      if (normalized[i]._autoSized) {
        normalized[i].x = forEnergy[i].x;
        normalized[i].y = forEnergy[i].y;
      }
    }
  }

  for (const item of normalized) {
    delete item._autoSized;
  }
  return normalized;
}

/**
 * #19: Preserve manual text/style/position edits across Re-Detect runs using
 * exact O(n^3) Hungarian assignment.
 */
export function mergeRedetectedBlocks(oldBlocks = [], newBlocks = [], imgWidth = 1000, imgHeight = 1400) {
  return preserveBlockEditsWithHungarian(oldBlocks, newBlocks, imgWidth, imgHeight, 0.85);
}

function cloneBlock(block) {
  return {
    ...block,
    bbox: Array.isArray(block.bbox) ? [...block.bbox] : block.bbox,
    polygon: Array.isArray(block.polygon)
      ? block.polygon.map((pt) => (Array.isArray(pt) ? [...pt] : pt))
      : block.polygon,
    room_polygon: Array.isArray(block.room_polygon)
      ? block.room_polygon.map((pt) => (Array.isArray(pt) ? [...pt] : pt))
      : block.room_polygon,
    outline:
      block.outline && typeof block.outline === 'object'
        ? {
            ...block.outline,
            color: Array.isArray(block.outline.color)
              ? [...block.outline.color]
              : block.outline.color,
          }
        : block.outline,
    style: block.style ? { ...block.style } : undefined,
  };
}

function clonePage(page) {
  const { runtimeMeta, runtime_meta, ...rest } = page || {};
  return {
    ...rest,
    dimensions: page.dimensions ? { ...page.dimensions } : null,
    blocks: Array.isArray(page.blocks) ? page.blocks.map(cloneBlock) : [],
    maskStrokes: Array.isArray(page.maskStrokes)
      ? page.maskStrokes.map((stroke) => ({
          ...stroke,
          points: Array.isArray(stroke.points) ? [...stroke.points] : [],
        }))
      : [],
  };
}

function createHistorySnapshot(pages, activePageId, selectedLayerIds) {
  // Structural sharing + strip transient runtime metadata; never clone base64 payloads.
  const lightweightPages = Array.isArray(pages)
    ? pages.map((p) => {
        if (!p || (!p.runtimeMeta && !p.runtime_meta)) return p;
        const { runtimeMeta, runtime_meta, ...rest } = p;
        return rest;
      })
    : [];
  return {
    pages: lightweightPages,
    activePageId: activePageId ?? null,
    selectedLayerIds: [...(selectedLayerIds || [])],
  };
}

function revokeUnusedBlobUrls(prevPages = [], nextPages = [], history = null) {
  if (typeof URL === 'undefined' || typeof URL.revokeObjectURL !== 'function') return;
  const activeUrls = new Set();
  const collect = (list) => {
    for (const p of list || []) {
      if (typeof p?.originalUrl === 'string' && p.originalUrl.startsWith('blob:')) {
        activeUrls.add(p.originalUrl);
      }
    }
  };
  collect(nextPages);
  if (history) {
    for (const snap of history.past || []) collect(snap?.pages);
    for (const snap of history.future || []) collect(snap?.pages);
  }
  for (const p of prevPages || []) {
    if (
      typeof p?.originalUrl === 'string' &&
      p.originalUrl.startsWith('blob:') &&
      !activeUrls.has(p.originalUrl)
    ) {
      try {
        URL.revokeObjectURL(p.originalUrl);
      } catch {
        // ignore
      }
    }
  }
}

function mergeBlockUpdates(block, updates) {
  const nextFontFamily =
    updates.font_family ??
    updates.style?.font_family ??
    block.style?.font_family ??
    block.font_family ??
    DEFAULT_FONT_FAMILY;
  const nextFontSize =
    updates.font_size ??
    updates.style?.font_size ??
    block.font_size ??
    18;
  const nextColor =
    updates.color ?? updates.style?.color ?? block.color ?? '#000000';
  const nextAlign =
    updates.align ?? updates.style?.align ?? block.align ?? 'center';
  const nextStrokeColor =
    updates.stroke_color ??
    updates.style?.stroke_color ??
    block.style?.stroke_color ??
    block.stroke_color ??
    '#ffffff';
  const nextStrokeWidth =
    updates.stroke_width !== undefined
      ? updates.stroke_width
      : updates.style?.stroke_width !== undefined
      ? updates.style.stroke_width
      : block.style?.stroke_width !== undefined
      ? block.style.stroke_width
      : 4;
  const nextLineHeight =
    updates.line_height ??
    updates.style?.line_height ??
    block.style?.line_height ??
    1.18;

  const didEditColor =
    updates.color !== undefined || updates.style?.color !== undefined;
  const didEditOutline =
    updates.stroke_color !== undefined ||
    updates.stroke_width !== undefined ||
    updates.style?.stroke_color !== undefined ||
    updates.style?.stroke_width !== undefined;

  return {
    ...block,
    ...updates,
    ...(didEditColor ? { _userEditedColor: true } : {}),
    ...(didEditOutline ? { _userEditedOutline: true } : {}),
    font_family: nextFontFamily,
    font_size: nextFontSize,
    color: nextColor,
    stroke_color: nextStrokeColor,
    stroke_width: nextStrokeWidth,
    align: nextAlign,
    line_height: nextLineHeight,
    style: {
      ...(block.style || {}),
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
}

/**
 * Resolve `'dark' | 'light' | 'system'` into a concrete `'dark' | 'light'` palette.
 */
export function resolveTheme(theme) {
  if (theme === 'light') return 'light';
  if (theme === 'dark') return 'dark';
  if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
    return window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light';
  }
  return 'dark';
}

/**
 * Apply the resolved Sumi Graphite (dark) or Manga Paper (light) theme classes
 * and `data-theme` attribute to `document.documentElement`.
 */
export function applyThemeToDocument(theme) {
  const resolved = resolveTheme(theme);
  if (typeof document !== 'undefined' && document.documentElement) {
    const root = document.documentElement;
    root.classList.remove('dark', 'light');
    root.classList.add(resolved);
    root.dataset.theme = resolved;
    root.style.colorScheme = resolved;
  }
  return resolved;
}

export function applyAccentToDocument(accentColor) {
  const valid = VALID_ACCENTS.includes(accentColor) ? accentColor : 'amber';
  if (typeof document !== 'undefined' && document.documentElement) {
    document.documentElement.dataset.accent = valid;
  }
  return valid;
}

export function applyDensityToDocument(uiDensity) {
  const valid = VALID_DENSITIES.includes(uiDensity) ? uiDensity : 'compact';
  if (typeof document !== 'undefined' && document.documentElement) {
    document.documentElement.dataset.density = valid;
  }
  return valid;
}

function readStoredValue(key, validList, fallback) {
  if (typeof window === 'undefined') return fallback;
  try {
    let saved = window.localStorage.getItem(key);
    if (!saved && key.startsWith('vellum.')) {
      const legacyKey = key.replace(/^vellum\./, 'mangaflow.');
      saved = window.localStorage.getItem(legacyKey);
    }
    if (validList.includes(saved)) {
      return saved;
    }
  } catch {
    // Ignore storage access errors in restricted webviews
  }
  return fallback;
}

function readStoredSettingsSnapshot() {
  if (typeof window === 'undefined') return null;
  try {
    let raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw && SETTINGS_STORAGE_KEY.startsWith('vellum.')) {
      raw = window.localStorage.getItem(SETTINGS_STORAGE_KEY.replace(/^vellum\./, 'mangaflow.'));
    }
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch {
    return null;
  }
}

function writeStoredSettingsSnapshot(snapshot) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(snapshot));
  } catch {
    // Ignore storage write errors
  }
}

let lastLocalAppearanceMutationAt = 0;

const storedSettingsSnapshot = readStoredSettingsSnapshot();
const initialTheme = readStoredValue(
  THEME_STORAGE_KEY,
  VALID_THEMES,
  storedSettingsSnapshot?.theme && VALID_THEMES.includes(storedSettingsSnapshot.theme)
    ? storedSettingsSnapshot.theme
    : 'dark'
);
const initialAccent = readStoredValue(
  ACCENT_STORAGE_KEY,
  VALID_ACCENTS,
  storedSettingsSnapshot?.accentColor && VALID_ACCENTS.includes(storedSettingsSnapshot.accentColor)
    ? storedSettingsSnapshot.accentColor
    : 'amber'
);
const initialDensity = readStoredValue(
  DENSITY_STORAGE_KEY,
  VALID_DENSITIES,
  storedSettingsSnapshot?.uiDensity && VALID_DENSITIES.includes(storedSettingsSnapshot.uiDensity)
    ? storedSettingsSnapshot.uiDensity
    : 'compact'
);

const initialResolvedTheme = applyThemeToDocument(initialTheme);
applyAccentToDocument(initialAccent);
applyDensityToDocument(initialDensity);

const { apiBaseUrl: _ignoredStoredApiBaseUrl, ...storedPipelineConfig } =
  storedSettingsSnapshot?.pipelineConfig || {};

const initialPipelineConfig = {
  ...DEFAULT_PIPELINE_CONFIG,
  ...storedPipelineConfig,
  apiBaseUrl: DEFAULT_API_BASE_URL,
  autoProcessOnImport: false,
  enabledStages: {
    ...DEFAULT_ENABLED_STAGES,
    ...(storedSettingsSnapshot?.pipelineConfig?.enabledStages || {}),
  },
  providerProfiles: {
    ...buildDefaultProviderProfiles(),
    ...(storedSettingsSnapshot?.pipelineConfig?.providerProfiles || {}),
  },
};

const initialTypesetting = {
  ...DEFAULT_TYPESETTING_CONFIG,
  ...(storedSettingsSnapshot?.typesetting || {}),
  fontFallbackChain:
    Array.isArray(storedSettingsSnapshot?.typesetting?.fontFallbackChain) &&
    storedSettingsSnapshot.typesetting.fontFallbackChain.length > 0
      ? [...storedSettingsSnapshot.typesetting.fontFallbackChain]
      : [...DEFAULT_FONT_FALLBACK_CHAIN],
};
initialTypesetting.fontFamily = formatFontFamilyChain(
  initialTypesetting.fontFallbackChain
);

const COALESCE_WINDOW_MS = 500;
let lastCoalesceKey = null;
let lastCoalesceTimestamp = 0;

function resetHistoryCoalescing() {
  lastCoalesceKey = null;
  lastCoalesceTimestamp = 0;
}

function buildNextHistoryState(
  state,
  nextPages,
  nextActivePageId,
  nextSelectedLayerIds,
  { recordHistory = true, coalesceKey = null } = {}
) {
  const nextPresent = createHistorySnapshot(
    nextPages,
    nextActivePageId,
    nextSelectedLayerIds
  );

  if (!recordHistory) {
    return {
      past: state.history.past,
      present: nextPresent,
      future: state.history.future,
    };
  }

  const now = Date.now();
  const shouldCoalesce =
    Boolean(coalesceKey) &&
    coalesceKey === lastCoalesceKey &&
    now - lastCoalesceTimestamp <= COALESCE_WINDOW_MS &&
    state.history.past.length > 0;

  if (shouldCoalesce) {
    lastCoalesceTimestamp = now;
    return {
      past: state.history.past,
      present: nextPresent,
      future: [],
    };
  }

  lastCoalesceKey = coalesceKey || null;
  lastCoalesceTimestamp = coalesceKey ? now : 0;

  const previousPresent =
    state.history.present ||
    createHistorySnapshot(
      state.pages,
      state.activePageId,
      state.selectedLayerIds
    );

  const nextPast = [...state.history.past, previousPresent].slice(
    -MAX_HISTORY_STEPS
  );

  return {
    past: nextPast,
    present: nextPresent,
    future: [],
  };
}

export const useEditorStore = create((set, get) => ({
  // ── Pages & Multi-Selection ────────────────────────────────────────────────
  pages: [],
  activePageId: null,
  selectedPageIds: [],

  // ── Layer / Bubble Selection & Active Tool ─────────────────────────────────
  selectedBlockId: null,
  selectedLayerIds: [],
  activeLayerType: 'text', // 'original' | 'mask' | 'inpainted' | 'text'
  activeTool: 'select', // 'select' | 'pan' | 'bbox' | 'brush' | 'eraser' | 'text'

  // ── Undo / Redo History Stack (max 50 steps) ───────────────────────────────
  history: {
    past: [],
    present: createHistorySnapshot([], null, []),
    future: [],
  },

  // ── View Layer Toggles & Workspace Chrome ──────────────────────────────────
  viewLayers: loadPersistedViewLayers(),
  zenMode: false,
  activeModal: null, // null | 'shortcuts' | 'about' | 'settings' | 'models'
  settingsActiveTab: 'appearance', // 'appearance' | 'pipeline' | 'providers' | 'translation' | 'typesetting'

  // ── Active Pipeline Configuration ──────────────────────────────────────────
  pipelineConfig: initialPipelineConfig,
  typesetting: initialTypesetting,
  availableProviderPresets: DEFAULT_PROVIDER_PRESETS,
  supportedTargetLanguages: DEFAULT_TARGET_LANGUAGES,
  providerModelsCache: {}, // { [providerId]: { models: [], visionModels: [], testedAt, connected, latencyMs, activeModel, resolvedModelId, error } }
  pipelineModelsCatalog: null, // cached response from GET /api/pipeline/models
  fontsCatalogCache: null, // cached response from GET /api/fonts/list
  fontLoadVersion: 0, // incremented whenever a FontFace finishes loading for instant Konva repaint

  // ── Hardware & LLM Runtime Status (shared by Toolbar & StatusBar) ──────────
  runtimeStatus: {
    executionProvider: null,
    executionMode: 'hybrid',
    detectorProvider: null,
    ocrProvider: null,
    inpainterProvider: null,
    detectorModel: null,
    availableProviders: [],
    modelsReady: false,
    modelsStatus: null,
    lmStudio: {
      connected: false,
      endpoint: DEFAULT_PIPELINE_CONFIG.baseUrl,
      activeModel: null,
      resolvedModelId: null,
      availableModels: [],
      error: null,
    },
    activeProviderSummary: null,
    hardware: null,
  },

  // ── Canvas Viewport Telemetry (for 24px StatusBar) ─────────────────────────
  canvasView: {
    zoom: 1,
    cursorX: null,
    cursorY: null,
  },

  // ── Appearance ('dark' | 'light' | 'system', accent preset, UI density) ────
  theme: initialTheme,
  resolvedTheme: initialResolvedTheme,
  accentColor: initialAccent,
  uiDensity: initialDensity,

  // ═══════════════════════════════════════════════════════════════════════════
  // Page Management Actions
  // ═══════════════════════════════════════════════════════════════════════════

  addPages: (newPages, options = {}) => {
    const items = Array.isArray(newPages) ? newPages : [newPages];
    if (items.length === 0) return;

    const { activateFirst = false, recordHistory = true } = options;

    set((state) => {
      const normalized = items.map((p) => {
        const resolvedInpainted = p.inpaintedUrl ?? p.cleanedImage ?? null;
        return {
          status: 'idle',
          cleanedImage: null,
          inpaintedUrl: null,
          maskUrl: null,
          maskId: null,
          maskStrokes: [],
          activeStage: null,
          ...p,
          cleanedImage: resolvedInpainted,
          inpaintedUrl: resolvedInpainted,
          blocks: normalizeBlocks(p.blocks || []),
        };
      });
      const nextPages = [...state.pages, ...normalized];
      const nextActivePageId =
        activateFirst || !state.activePageId
          ? normalized[0].id
          : state.activePageId;
      const nextSelectedPageIds =
        activateFirst || state.selectedPageIds.length === 0
          ? [nextActivePageId]
          : state.selectedPageIds;
      const nextSelectedLayerIds =
        activateFirst || !state.activePageId ? [] : state.selectedLayerIds;

      const history = buildNextHistoryState(
        state,
        nextPages,
        nextActivePageId,
        nextSelectedLayerIds,
        { recordHistory }
      );

      return {
        pages: nextPages,
        activePageId: nextActivePageId,
        selectedPageIds: nextSelectedPageIds,
        selectedLayerIds: nextSelectedLayerIds,
        selectedBlockId: nextSelectedLayerIds[0] ?? null,
        history,
      };
    });
  },

  removePage: (pageId, options = { recordHistory: true }) => {
    set((state) => {
      const nextPages = state.pages.filter((p) => p.id !== pageId);
      if (nextPages.length === state.pages.length) return state;

      const nextSelectedPageIds = state.selectedPageIds.filter(
        (id) => id !== pageId
      );
      const removedActive = state.activePageId === pageId;
      const nextActivePageId = removedActive
        ? nextPages[0]?.id ?? null
        : state.activePageId;
      const finalSelectedPageIds =
        nextSelectedPageIds.length > 0
          ? nextSelectedPageIds
          : nextActivePageId
          ? [nextActivePageId]
          : [];
      const nextSelectedLayerIds = removedActive ? [] : state.selectedLayerIds;

      const history = buildNextHistoryState(
        state,
        nextPages,
        nextActivePageId,
        nextSelectedLayerIds,
        { recordHistory: options.recordHistory ?? true }
      );
      revokeUnusedBlobUrls(state.pages, nextPages, history);

      return {
        pages: nextPages,
        activePageId: nextActivePageId,
        selectedPageIds: finalSelectedPageIds,
        selectedLayerIds: nextSelectedLayerIds,
        selectedBlockId: nextSelectedLayerIds[0] ?? null,
        history,
      };
    });
  },

  removeSelectedPages: () => {
    set((state) => {
      if (state.selectedPageIds.length === 0) return state;
      const removeSet = new Set(state.selectedPageIds);
      const nextPages = state.pages.filter((p) => !removeSet.has(p.id));
      const nextActivePageId =
        state.activePageId && !removeSet.has(state.activePageId)
          ? state.activePageId
          : nextPages[0]?.id ?? null;
      const nextSelectedPageIds = nextActivePageId ? [nextActivePageId] : [];
      const nextSelectedLayerIds =
        nextActivePageId === state.activePageId ? state.selectedLayerIds : [];

      const history = buildNextHistoryState(
        state,
        nextPages,
        nextActivePageId,
        nextSelectedLayerIds,
        { recordHistory: true }
      );
      revokeUnusedBlobUrls(state.pages, nextPages, history);

      return {
        pages: nextPages,
        activePageId: nextActivePageId,
        selectedPageIds: nextSelectedPageIds,
        selectedLayerIds: nextSelectedLayerIds,
        selectedBlockId: nextSelectedLayerIds[0] ?? null,
        history,
      };
    });
  },

  reorderPages: (fromIndex, toIndex) => {
    set((state) => {
      if (
        fromIndex === toIndex ||
        fromIndex < 0 ||
        toIndex < 0 ||
        fromIndex >= state.pages.length ||
        toIndex >= state.pages.length
      ) {
        return state;
      }
      const nextPages = [...state.pages];
      const [moved] = nextPages.splice(fromIndex, 1);
      nextPages.splice(toIndex, 0, moved);

      const history = buildNextHistoryState(
        state,
        nextPages,
        state.activePageId,
        state.selectedLayerIds,
        { recordHistory: true }
      );

      return { pages: nextPages, history };
    });
  },

  updatePage: (pageId, patchOrUpdater, options = { recordHistory: false }) => {
    set((state) => {
      let changed = false;
      const nextPages = state.pages.map((p) => {
        if (p.id !== pageId) return p;
        changed = true;
        const patch =
          typeof patchOrUpdater === 'function'
            ? patchOrUpdater(p)
            : patchOrUpdater;
        return { ...p, ...patch };
      });
      if (!changed) return state;

      const history = buildNextHistoryState(
        state,
        nextPages,
        state.activePageId,
        state.selectedLayerIds,
        {
          recordHistory: options.recordHistory ?? false,
          coalesceKey: options.coalesceKey ?? null,
        }
      );

      return { pages: nextPages, history };
    });
  },

  setPageStatus: (pageId, status, extraPatch = {}) => {
    get().updatePage(
      pageId,
      { status, ...extraPatch },
      { recordHistory: false }
    );
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // Page & Layer Selection Actions
  // ═══════════════════════════════════════════════════════════════════════════

  setActivePage: (pageId) => {
    resetHistoryCoalescing();
    set((state) => {
      const exists = pageId
        ? state.pages.some((p) => p.id === pageId)
        : false;
      const nextActiveId = exists ? pageId : null;
      const currentPresent = state.history.present;
      return {
        activePageId: nextActiveId,
        selectedPageIds: nextActiveId ? [nextActiveId] : [],
        selectedBlockId: null,
        selectedLayerIds: [],
        history: currentPresent
          ? {
              ...state.history,
              present: {
                ...currentPresent,
                activePageId: nextActiveId,
                selectedLayerIds: [],
              },
            }
          : state.history,
      };
    });
  },

  toggleSelectPage: (pageId) => {
    set((state) => {
      const exists = state.pages.some((p) => p.id === pageId);
      if (!exists) return state;

      const isSelected = state.selectedPageIds.includes(pageId);
      let nextSelected;
      if (isSelected) {
        nextSelected = state.selectedPageIds.filter((id) => id !== pageId);
        if (nextSelected.length === 0) {
          nextSelected = [pageId];
        }
      } else {
        nextSelected = [...state.selectedPageIds, pageId];
      }

      const nextActive = nextSelected.includes(state.activePageId)
        ? state.activePageId
        : nextSelected[nextSelected.length - 1] ?? null;

      return {
        selectedPageIds: nextSelected,
        activePageId: nextActive,
        selectedBlockId:
          nextActive === state.activePageId ? state.selectedBlockId : null,
        selectedLayerIds:
          nextActive === state.activePageId ? state.selectedLayerIds : [],
      };
    });
  },

  selectPageRange: (targetPageId) => {
    set((state) => {
      const targetIdx = state.pages.findIndex((p) => p.id === targetPageId);
      if (targetIdx === -1) return state;

      const anchorId =
        state.activePageId || state.selectedPageIds[0] || targetPageId;
      const anchorIdx = state.pages.findIndex((p) => p.id === anchorId);
      const start = Math.min(anchorIdx === -1 ? targetIdx : anchorIdx, targetIdx);
      const end = Math.max(anchorIdx === -1 ? targetIdx : anchorIdx, targetIdx);

      const rangeIds = state.pages.slice(start, end + 1).map((p) => p.id);
      return {
        selectedPageIds: rangeIds,
        activePageId: targetPageId,
        selectedBlockId:
          targetPageId === state.activePageId ? state.selectedBlockId : null,
        selectedLayerIds:
          targetPageId === state.activePageId ? state.selectedLayerIds : [],
      };
    });
  },

  selectAllPages: () => {
    set((state) => ({
      selectedPageIds: state.pages.map((p) => p.id),
    }));
  },

  selectAllLayers: () => {
    set((state) => {
      if (!state.activePageId) return state;
      const activePage = state.pages.find((p) => p.id === state.activePageId);
      const allIds = (activePage?.blocks || []).map((b) => b.id);
      return {
        selectedLayerIds: allIds,
        selectedBlockId: allIds[0] ?? null,
      };
    });
  },

  replaceProjectState: ({ pages = [], activePageId = null, pipelineConfig = null, viewLayers = null, typesetting = null } = {}) => {
    resetHistoryCoalescing();
    set((state) => {
      const normalizedPages = (pages || []).map((p, idx) => {
        const resolvedInpainted = p.inpaintedUrl || p.cleanedImage || null;
        const normalizedStatus =
          p.status === 'processing' || p.status === 'queued'
            ? 'idle'
            : p.status || (resolvedInpainted ? 'done' : 'idle');
        return {
          id: p.id || `page_loaded_${Date.now()}_${idx}`,
          name: p.name || `page_${idx + 1}.png`,
          file: p.file || null,
          originalUrl: p.originalUrl || null,
          cleanedImage: resolvedInpainted,
          inpaintedUrl: resolvedInpainted,
          maskUrl: p.maskUrl || null,
          maskId: p.maskId || null,
          maskStrokes: Array.isArray(p.maskStrokes) ? p.maskStrokes : [],
          dimensions: p.dimensions || null,
          blocks: normalizeBlocks(p.blocks || []),
          status: normalizedStatus,
          activeStage: null,
          error: null,
          sourceLang: p.sourceLang || state.pipelineConfig.sourceLang,
          targetLang: p.targetLang || state.pipelineConfig.targetLang,
          model: p.model || null,
          tilesInpainted: p.tilesInpainted ?? 0,
        };
      });
      const nextActiveId =
        activePageId && normalizedPages.some((p) => p.id === activePageId)
          ? activePageId
          : normalizedPages[0]?.id ?? null;

      const history = buildNextHistoryState(
        state,
        normalizedPages,
        nextActiveId,
        [],
        { recordHistory: true }
      );
      revokeUnusedBlobUrls(state.pages, normalizedPages, history);

      const nextPipelineConfig = pipelineConfig
        ? (() => {
            const { apiBaseUrl: _ignoredProjectApiBaseUrl, ...restPipeline } =
              pipelineConfig;
            return {
              ...state.pipelineConfig,
              ...restPipeline,
              apiBaseUrl: DEFAULT_API_BASE_URL,
              enabledStages: {
                ...state.pipelineConfig.enabledStages,
                ...(pipelineConfig.enabledStages || {}),
              },
            };
          })()
        : state.pipelineConfig;

      return {
        pages: normalizedPages,
        activePageId: nextActiveId,
        selectedPageIds: nextActiveId ? [nextActiveId] : [],
        selectedBlockId: null,
        selectedLayerIds: [],
        pipelineConfig: nextPipelineConfig,
        typesetting: typesetting
          ? { ...state.typesetting, ...typesetting }
          : state.typesetting,
        viewLayers: viewLayers
          ? { ...state.viewLayers, ...viewLayers }
          : state.viewLayers,
        history,
      };
    });
  },

  setSelectedBlock: (blockId) => {
    set({
      selectedBlockId: blockId ?? null,
      selectedLayerIds: blockId ? [blockId] : [],
    });
  },

  setSelectedLayers: (layerIds) => {
    const unique = Array.from(new Set((layerIds || []).filter(Boolean)));
    set({
      selectedLayerIds: unique,
      selectedBlockId: unique[0] ?? null,
    });
  },

  toggleSelectLayer: (layerId) => {
    if (!layerId) return;
    set((state) => {
      const exists = state.selectedLayerIds.includes(layerId);
      const next = exists
        ? state.selectedLayerIds.filter((id) => id !== layerId)
        : [...state.selectedLayerIds, layerId];
      return {
        selectedLayerIds: next,
        selectedBlockId: next[0] ?? null,
      };
    });
  },

  setActiveLayerType: (layerType) => {
    set({ activeLayerType: layerType });
  },

  setActiveTool: (tool) => {
    set({ activeTool: tool });
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // Block / Text Layer Editing Actions (History-Aware)
  // ═══════════════════════════════════════════════════════════════════════════

  updateBlock: (blockId, updates, options = {}) => {
    const {
      pageId = null,
      recordHistory = true,
      coalesceKey = null,
    } = options;

    set((state) => {
      const targetPageId = pageId || state.activePageId;
      if (!targetPageId) return state;

      let updatedAny = false;
      const nextPages = state.pages.map((page) => {
        if (page.id !== targetPageId) return page;
        const nextBlocks = (page.blocks || []).map((block) => {
          if (block.id !== blockId) return block;
          updatedAny = true;
          return mergeBlockUpdates(block, updates);
        });
        return { ...page, blocks: nextBlocks };
      });

      if (!updatedAny) return state;

      const history = buildNextHistoryState(
        state,
        nextPages,
        state.activePageId,
        state.selectedLayerIds,
        { recordHistory, coalesceKey }
      );

      return { pages: nextPages, history };
    });
  },

  updateBlocksBatch: (updatesById, options = {}) => {
    const {
      pageId = null,
      recordHistory = true,
      coalesceKey = null,
    } = options;

    if (!updatesById) return;
    const patchMap = Array.isArray(updatesById)
      ? Object.fromEntries(
          updatesById
            .filter((item) => item && item.id)
            .map(({ id, ...rest }) => [id, rest])
        )
      : updatesById;

    const patchKeys = Object.keys(patchMap);
    if (patchKeys.length === 0) return;

    set((state) => {
      const targetPageId = pageId || state.activePageId;
      if (!targetPageId) return state;

      let updatedAny = false;
      const nextPages = state.pages.map((page) => {
        if (page.id !== targetPageId) return page;
        const nextBlocks = (page.blocks || []).map((block) => {
          const patch = patchMap[block.id];
          if (!patch) return block;
          updatedAny = true;
          return mergeBlockUpdates(block, patch);
        });
        return { ...page, blocks: nextBlocks };
      });

      if (!updatedAny) return state;

      const history = buildNextHistoryState(
        state,
        nextPages,
        state.activePageId,
        state.selectedLayerIds,
        { recordHistory, coalesceKey }
      );

      return { pages: nextPages, history };
    });
  },

  addBlock: (blockOverride = null, pageId = null) => {
    let createdId = null;
    set((state) => {
      const targetPageId = pageId || state.activePageId;
      const page = state.pages.find((p) => p.id === targetPageId);
      if (!page) return state;

      const nextIdx = (page.blocks?.length || 0) + 1;
      const newId =
        blockOverride?.id || `block_custom_${Date.now().toString().slice(-4)}`;
      createdId = newId;

      const centerW = Math.round((page.dimensions?.width || 800) / 2 - 80);
      const centerH = Math.round((page.dimensions?.height || 1200) / 2 - 40);

      const ts = state.typesetting || DEFAULT_TYPESETTING_CONFIG;
      const newBlock = normalizeBlocks([
        {
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
          translated_text: ts.allCaps ? 'NEW TEXT' : 'New Text',
          font_family: ts.fontFamily || DEFAULT_FONT_FAMILY,
          font_size: ts.defaultFontSize || 18,
          color: ts.defaultTextColor || '#000000',
          stroke_color: ts.defaultStrokeColor || '#ffffff',
          stroke_width: ts.defaultStrokeWidth ?? 4,
          align: ts.defaultAlign || 'center',
          line_height: ts.defaultLineHeight || 1.18,
          ...(blockOverride || {}),
        },
      ])[0];

      const nextPages = state.pages.map((p) =>
        p.id === targetPageId
          ? { ...p, blocks: [...(p.blocks || []), newBlock] }
          : p
      );
      const nextSelectedLayerIds = [newId];
      const history = buildNextHistoryState(
        state,
        nextPages,
        state.activePageId,
        nextSelectedLayerIds,
        { recordHistory: true }
      );

      return {
        pages: nextPages,
        selectedBlockId: newId,
        selectedLayerIds: nextSelectedLayerIds,
        history,
      };
    });
    return createdId;
  },

  deleteBlock: (blockId, pageId = null) => {
    set((state) => {
      const targetPageId = pageId || state.activePageId;
      if (!targetPageId) return state;

      let removed = false;
      const nextPages = state.pages.map((p) => {
        if (p.id !== targetPageId) return p;
        const filtered = (p.blocks || []).filter((b) => b.id !== blockId);
        if (filtered.length !== (p.blocks || []).length) {
          removed = true;
        }
        return { ...p, blocks: filtered };
      });
      if (!removed) return state;

      const nextSelectedLayerIds = state.selectedLayerIds.filter(
        (id) => id !== blockId
      );
      const history = buildNextHistoryState(
        state,
        nextPages,
        state.activePageId,
        nextSelectedLayerIds,
        { recordHistory: true }
      );

      return {
        pages: nextPages,
        selectedLayerIds: nextSelectedLayerIds,
        selectedBlockId: nextSelectedLayerIds[0] ?? null,
        history,
      };
    });
  },

  deleteSelectedLayers: () => {
    set((state) => {
      if (!state.activePageId) return state;
      const targetIds =
        state.selectedLayerIds.length > 0
          ? state.selectedLayerIds
          : state.selectedBlockId
          ? [state.selectedBlockId]
          : [];
      if (targetIds.length === 0) return state;

      const removeSet = new Set(targetIds);
      const nextPages = state.pages.map((p) =>
        p.id === state.activePageId
          ? { ...p, blocks: (p.blocks || []).filter((b) => !removeSet.has(b.id)) }
          : p
      );
      const history = buildNextHistoryState(
        state,
        nextPages,
        state.activePageId,
        [],
        { recordHistory: true }
      );
      return {
        pages: nextPages,
        selectedLayerIds: [],
        selectedBlockId: null,
        history,
      };
    });
  },

  duplicateSelectedLayers: () => {
    set((state) => {
      if (!state.activePageId) return state;
      const activePage = state.pages.find((p) => p.id === state.activePageId);
      if (!activePage) return state;

      const targetIds =
        state.selectedLayerIds.length > 0
          ? state.selectedLayerIds
          : state.selectedBlockId
          ? [state.selectedBlockId]
          : [];
      if (targetIds.length === 0) return state;

      const targetSet = new Set(targetIds);
      const existingBlocks = activePage.blocks || [];
      const toDuplicate = existingBlocks.filter((b) => targetSet.has(b.id));
      if (toDuplicate.length === 0) return state;

      const maxOrder = existingBlocks.reduce(
        (max, b) => Math.max(max, Number(b.order) || 0),
        existingBlocks.length
      );
      const ts = Date.now().toString().slice(-4);
      const offset = 16;

      const duplicated = toDuplicate.map((block, idx) => {
        const cloned = cloneBlock(block);
        const newX = Math.round((cloned.x ?? cloned.bbox?.[0] ?? 0) + offset);
        const newY = Math.round((cloned.y ?? cloned.bbox?.[1] ?? 0) + offset);
        const w = cloned.width ?? Math.max(40, (cloned.bbox?.[2] ?? 140) - (cloned.bbox?.[0] ?? 0));
        const h = cloned.height ?? Math.max(24, (cloned.bbox?.[3] ?? 60) - (cloned.bbox?.[1] ?? 0));
        return {
          ...cloned,
          id: `${cloned.id}_copy_${ts}_${idx + 1}`,
          order: maxOrder + idx + 1,
          x: newX,
          y: newY,
          bbox: [newX, newY, newX + w, newY + h],
        };
      });

      const nextSelectedIds = duplicated.map((b) => b.id);
      const nextPages = state.pages.map((p) =>
        p.id === state.activePageId
          ? { ...p, blocks: [...existingBlocks, ...duplicated] }
          : p
      );

      const history = buildNextHistoryState(
        state,
        nextPages,
        state.activePageId,
        nextSelectedIds,
        { recordHistory: true }
      );

      return {
        pages: nextPages,
        selectedLayerIds: nextSelectedIds,
        selectedBlockId: nextSelectedIds[0] ?? null,
        history,
      };
    });
  },

  applyStyleToAll: (styleUpdates, pageId = null) => {
    set((state) => {
      const targetPageId = pageId || state.activePageId;
      if (!targetPageId) return state;

      const nextPages = state.pages.map((page) => {
        if (page.id !== targetPageId) return page;
        return {
          ...page,
          blocks: (page.blocks || []).map((b) =>
            mergeBlockUpdates(b, styleUpdates)
          ),
        };
      });

      const history = buildNextHistoryState(
        state,
        nextPages,
        state.activePageId,
        state.selectedLayerIds,
        { recordHistory: true }
      );

      return { pages: nextPages, history };
    });
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // Undo / Redo History Actions (past / present / future, max 50 steps)
  // ═══════════════════════════════════════════════════════════════════════════

  pushHistorySnapshot: () => {
    resetHistoryCoalescing();
    set((state) => {
      const currentSnapshot = createHistorySnapshot(
        state.pages,
        state.activePageId,
        state.selectedLayerIds
      );
      const previousPresent = state.history.present || currentSnapshot;
      return {
        history: {
          past: [...state.history.past, previousPresent].slice(
            -MAX_HISTORY_STEPS
          ),
          present: currentSnapshot,
          future: [],
        },
      };
    });
  },

  undo: () => {
    resetHistoryCoalescing();
    set((state) => {
      const { past, present, future } = state.history;
      if (past.length === 0) return state;

      const previous = past[past.length - 1];
      const nextPast = past.slice(0, -1);
      const currentSnapshot =
        present ||
        createHistorySnapshot(
          state.pages,
          state.activePageId,
          state.selectedLayerIds
        );
      const nextFuture = [currentSnapshot, ...future].slice(
        0,
        MAX_HISTORY_STEPS
      );

      const restoredPages = [...(previous.pages || [])];
      const restoredActivePageId = restoredPages.some(
        (p) => p.id === previous.activePageId
      )
        ? previous.activePageId
        : restoredPages[0]?.id ?? null;

      const activePageObj = restoredPages.find(
        (p) => p.id === restoredActivePageId
      );
      const validBlockIds = new Set(
        (activePageObj?.blocks || []).map((b) => b.id)
      );
      const restoredSelectedLayers = (previous.selectedLayerIds || []).filter(
        (id) => validBlockIds.has(id)
      );

      return {
        pages: restoredPages,
        activePageId: restoredActivePageId,
        selectedPageIds: restoredActivePageId ? [restoredActivePageId] : [],
        selectedLayerIds: restoredSelectedLayers,
        selectedBlockId: restoredSelectedLayers[0] ?? null,
        history: {
          past: nextPast,
          present: previous,
          future: nextFuture,
        },
      };
    });
  },

  redo: () => {
    resetHistoryCoalescing();
    set((state) => {
      const { past, present, future } = state.history;
      if (future.length === 0) return state;

      const [next, ...restFuture] = future;
      const currentSnapshot =
        present ||
        createHistorySnapshot(
          state.pages,
          state.activePageId,
          state.selectedLayerIds
        );
      const nextPast = [...past, currentSnapshot].slice(-MAX_HISTORY_STEPS);

      const restoredPages = [...(next.pages || [])];
      const restoredActivePageId = restoredPages.some(
        (p) => p.id === next.activePageId
      )
        ? next.activePageId
        : restoredPages[0]?.id ?? null;

      const activePageObj = restoredPages.find(
        (p) => p.id === restoredActivePageId
      );
      const validBlockIds = new Set(
        (activePageObj?.blocks || []).map((b) => b.id)
      );
      const restoredSelectedLayers = (next.selectedLayerIds || []).filter(
        (id) => validBlockIds.has(id)
      );

      return {
        pages: restoredPages,
        activePageId: restoredActivePageId,
        selectedPageIds: restoredActivePageId ? [restoredActivePageId] : [],
        selectedLayerIds: restoredSelectedLayers,
        selectedBlockId: restoredSelectedLayers[0] ?? null,
        history: {
          past: nextPast,
          present: next,
          future: restFuture,
        },
      };
    });
  },

  clearHistory: () => {
    resetHistoryCoalescing();
    set((state) => ({
      history: {
        past: [],
        present: createHistorySnapshot(
          state.pages,
          state.activePageId,
          state.selectedLayerIds
        ),
        future: [],
      },
    }));
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // View Layer Toggle Actions
  // ═══════════════════════════════════════════════════════════════════════════

  toggleViewLayer: (layerKey) => {
    set((state) => {
      if (!(layerKey in state.viewLayers)) return state;
      const nextLayers = {
        ...state.viewLayers,
        [layerKey]: !state.viewLayers[layerKey],
      };
      savePersistedViewLayers(nextLayers);
      return { viewLayers: nextLayers };
    });
  },

  setViewLayer: (layerKey, visible) => {
    set((state) => {
      if (!(layerKey in state.viewLayers)) return state;
      const nextLayers = {
        ...state.viewLayers,
        [layerKey]: Boolean(visible),
      };
      savePersistedViewLayers(nextLayers);
      return { viewLayers: nextLayers };
    });
  },

  setViewLayers: (partial) => {
    set((state) => {
      const nextLayers = {
        ...state.viewLayers,
        ...partial,
      };
      savePersistedViewLayers(nextLayers);
      return { viewLayers: nextLayers };
    });
  },

  resetViewLayers: () => {
    const defaults = { ...DEFAULT_VIEW_LAYERS };
    savePersistedViewLayers(defaults);
    set({ viewLayers: defaults });
  },

  toggleZenMode: () => {
    set((state) => ({ zenMode: !state.zenMode }));
  },

  setZenMode: (enabled) => {
    set({ zenMode: Boolean(enabled) });
  },

  openModal: (modalId, tabId = null) => {
    set((state) => ({
      activeModal: modalId,
      settingsActiveTab:
        modalId === 'settings' && tabId ? tabId : state.settingsActiveTab,
    }));
  },

  closeModal: () => {
    set({ activeModal: null });
  },

  setSettingsActiveTab: (tabId) => {
    set({ settingsActiveTab: tabId });
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // Active Pipeline Config & Typesetting Actions
  // ═══════════════════════════════════════════════════════════════════════════

  _writeLocalSnapshot: () => {
    const s = get();
    const { apiKey: _redactedKey, providerProfiles: rawProfiles, ...restPipelineConfig } = s.pipelineConfig || {};
    const safeProviderProfiles = rawProfiles && typeof rawProfiles === 'object'
      ? Object.fromEntries(
          Object.entries(rawProfiles).map(([pid, prof]) => [
            pid,
            prof && typeof prof === 'object' ? { ...prof, api_key: '' } : prof,
          ])
        )
      : undefined;
    writeStoredSettingsSnapshot({
      theme: s.theme,
      accentColor: s.accentColor,
      uiDensity: s.uiDensity,
      pipelineConfig: {
        ...restPipelineConfig,
        ...(safeProviderProfiles ? { providerProfiles: safeProviderProfiles } : {}),
      },
      typesetting: s.typesetting,
    });
  },

  updatePipelineConfig: (patch) => {
    set((state) => ({
      pipelineConfig: {
        ...state.pipelineConfig,
        ...patch,
      },
    }));
    get()._writeLocalSnapshot();
  },

  updateProviderProfile: (providerId, profilePatch) => {
    if (!providerId) return;
    set((state) => {
      const existing = state.pipelineConfig.providerProfiles?.[providerId] || {
        base_url: '',
        api_key: '',
        model: '',
      };
      const updatedProfile = { ...existing, ...profilePatch };
      const nextProfiles = {
        ...(state.pipelineConfig.providerProfiles || {}),
        [providerId]: updatedProfile,
      };
      const isCurrent = state.pipelineConfig.provider === providerId;
      return {
        pipelineConfig: {
          ...state.pipelineConfig,
          providerProfiles: nextProfiles,
          ...(isCurrent
            ? {
                baseUrl: updatedProfile.base_url ?? state.pipelineConfig.baseUrl,
                apiKey: updatedProfile.api_key ?? state.pipelineConfig.apiKey,
                model: updatedProfile.model ?? state.pipelineConfig.model,
              }
            : {}),
        },
      };
    });
    get()._writeLocalSnapshot();
  },

  updateTypesetting: (patch) => {
    set((state) => {
      const next = {
        ...state.typesetting,
        ...patch,
      };
      if (Array.isArray(patch?.fontFallbackChain)) {
        next.fontFallbackChain = [...patch.fontFallbackChain];
        next.fontFamily = formatFontFamilyChain(next.fontFallbackChain);
      }
      return { typesetting: next };
    });
    get()._writeLocalSnapshot();
  },

  setFontFallbackChain: (chain) => {
    if (!Array.isArray(chain)) return;
    const cleaned = chain
      .map((f) => String(f || '').trim())
      .filter(Boolean);
    const nextChain = cleaned.length > 0 ? cleaned : ['sans-serif'];
    set((state) => ({
      typesetting: {
        ...state.typesetting,
        fontFallbackChain: nextChain,
        fontFamily: formatFontFamilyChain(nextChain),
      },
    }));
    get()._writeLocalSnapshot();
  },

  setProviderModelsResult: (providerId, testResult) => {
    if (!providerId || !testResult) return;
    set((state) => ({
      providerModelsCache: {
        ...state.providerModelsCache,
        [providerId]: {
          connected: Boolean(testResult.connected ?? testResult.ok),
          models: Array.isArray(testResult.models)
            ? testResult.models
            : Array.isArray(testResult.available_models)
            ? testResult.available_models
            : [],
          visionModels: Array.isArray(testResult.vision_models)
            ? testResult.vision_models
            : [],
          activeModel: testResult.active_model ?? null,
          resolvedModelId: testResult.resolved_model_id ?? null,
          latencyMs: testResult.latency_ms ?? null,
          message: testResult.message ?? null,
          error: testResult.error ?? null,
          testedAt: Date.now(),
        },
      },
    }));
  },

  setPipelineModelsCatalog: (catalogData) => {
    if (!catalogData || typeof catalogData !== 'object') return;
    set((state) => ({
      pipelineModelsCatalog: catalogData,
      runtimeStatus: {
        ...state.runtimeStatus,
        executionProvider:
          catalogData.execution_provider ?? state.runtimeStatus.executionProvider,
        availableProviders:
          Array.isArray(catalogData.available_providers) &&
          catalogData.available_providers.length > 0
            ? catalogData.available_providers
            : state.runtimeStatus.availableProviders,
      },
    }));
  },

  setFontsCatalogCache: (catalogData) => {
    set({ fontsCatalogCache: catalogData });
    if (catalogData && typeof window !== 'undefined' && typeof FontFace !== 'undefined') {
      const chain =
        useEditorStore.getState()?.typesetting?.fontFallbackChain ||
        DEFAULT_FONT_FALLBACK_CHAIN;
      const preloadNames = new Set(['CC Wild Words', ...chain]);
      preloadNames.forEach((fam) => {
        loadFontFamilyFromCatalog(fam, catalogData, DEFAULT_API_BASE_URL).catch(
          () => false
        );
      });
    }
  },

  bumpFontLoadVersion: () => {
    set((state) => ({ fontLoadVersion: (state.fontLoadVersion || 0) + 1 }));
  },

  setPipelineScope: (scope) => {
    if (!['current_page', 'selected_pages', 'project'].includes(scope)) return;
    set((state) => ({
      pipelineConfig: {
        ...state.pipelineConfig,
        scope,
      },
    }));
  },

  togglePipelineStage: (stageKey) => {
    set((state) => {
      if (!(stageKey in state.pipelineConfig.enabledStages)) return state;
      return {
        pipelineConfig: {
          ...state.pipelineConfig,
          enabledStages: {
            ...state.pipelineConfig.enabledStages,
            [stageKey]: !state.pipelineConfig.enabledStages[stageKey],
          },
        },
      };
    });
  },

  setPipelineStages: (partial) => {
    set((state) => ({
      pipelineConfig: {
        ...state.pipelineConfig,
        enabledStages: {
          ...state.pipelineConfig.enabledStages,
          ...partial,
        },
      },
    }));
  },

  setActiveProvider: (providerId, { syncBackend = true } = {}) => {
    const nextPid = String(providerId || 'none').trim() || 'none';
    set((state) => {
      const prevPid = state.pipelineConfig.provider || 'none';
      const profiles = { ...(state.pipelineConfig.providerProfiles || {}) };
      if (prevPid !== 'none') {
        profiles[prevPid] = {
          base_url: state.pipelineConfig.baseUrl ?? profiles[prevPid]?.base_url ?? '',
          api_key: state.pipelineConfig.apiKey ?? profiles[prevPid]?.api_key ?? '',
          model: state.pipelineConfig.model ?? profiles[prevPid]?.model ?? '',
        };
      }
      const preset = DEFAULT_PROVIDER_PRESETS.find((p) => p.id === nextPid);
      const nextProfile = profiles[nextPid] || {
        base_url: preset?.default_base_url || '',
        api_key: preset?.default_api_key || '',
        model: preset?.default_model || '',
      };
      if (nextPid !== 'none' && !profiles[nextPid]) {
        profiles[nextPid] = nextProfile;
      }
      return {
        pipelineConfig: {
          ...state.pipelineConfig,
          provider: nextPid,
          baseUrl: nextPid === 'none' ? '' : nextProfile.base_url ?? preset?.default_base_url ?? '',
          apiKey: nextPid === 'none' ? '' : nextProfile.api_key ?? '',
          model: nextPid === 'none' ? '' : nextProfile.model ?? preset?.default_model ?? '',
          providerProfiles: profiles,
        },
      };
    });
    get()._writeLocalSnapshot();
    if (syncBackend) {
      const s = get();
      const apiBaseUrl = s.pipelineConfig.apiBaseUrl ?? DEFAULT_API_BASE_URL;
      fetch(`${apiBaseUrl}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          active_provider: nextPid,
          provider: nextPid,
          providers: s.pipelineConfig.providerProfiles,
        }),
      }).catch(() => {});
    }
  },

  setActiveModel: (model) => {
    set((state) => {
      const pid = state.pipelineConfig.provider || 'none';
      const nextProfiles = {
        ...(state.pipelineConfig.providerProfiles || {}),
        [pid]: {
          ...(state.pipelineConfig.providerProfiles?.[pid] || {
            base_url: state.pipelineConfig.baseUrl,
            api_key: state.pipelineConfig.apiKey,
          }),
          model,
        },
      };
      return {
        pipelineConfig: {
          ...state.pipelineConfig,
          model,
          providerProfiles: nextProfiles,
        },
      };
    });
    get()._writeLocalSnapshot();
  },

  setSupportedTargetLanguages: (langs) => {
    if (!Array.isArray(langs) || langs.length === 0) return;
    set({ supportedTargetLanguages: langs });
  },

  setSourceLang: (sourceLang) => {
    set((state) => ({
      pipelineConfig: {
        ...state.pipelineConfig,
        sourceLang,
      },
    }));
    get()._writeLocalSnapshot();
  },

  setTargetLang: (targetLang) => {
    set((state) => ({
      pipelineConfig: {
        ...state.pipelineConfig,
        targetLang,
      },
    }));
    get()._writeLocalSnapshot();
  },

  setTranslationTone: (translationTone) => {
    set((state) => ({
      pipelineConfig: {
        ...state.pipelineConfig,
        translationTone: translationTone || 'standard',
      },
    }));
    get()._writeLocalSnapshot();
  },

  setHonorificsPolicy: (honorificsPolicy) => {
    set((state) => ({
      pipelineConfig: {
        ...state.pipelineConfig,
        honorificsPolicy: honorificsPolicy || 'keep',
      },
    }));
    get()._writeLocalSnapshot();
  },

  replaceAllTranslations: (
    findText,
    replaceText,
    { caseSensitive = false, wholeWord = false } = {}
  ) => {
    const query = String(findText ?? '');
    if (!query) return { replacedBlocks: 0, affectedPages: 0 };
    const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const pattern = wholeWord ? `\\b${escaped}\\b` : escaped;
    const flags = caseSensitive ? 'g' : 'gi';
    const regex = new RegExp(pattern, flags);
    const replacement = String(replaceText ?? '');

    let replacedBlocks = 0;
    let affectedPages = 0;

    set((state) => {
      let anyChanged = false;
      const nextPages = state.pages.map((page) => {
        const blocks = page.blocks || [];
        let pageChanged = false;
        const nextBlocks = blocks.map((block) => {
          const orig = String(block.translated_text ?? '');
          if (!orig) return block;
          regex.lastIndex = 0;
          if (!regex.test(orig)) return block;
          regex.lastIndex = 0;
          const updated = orig.replace(regex, replacement);
          if (updated === orig) return block;
          pageChanged = true;
          anyChanged = true;
          replacedBlocks += 1;
          return { ...block, translated_text: updated };
        });
        if (!pageChanged) return page;
        affectedPages += 1;
        return { ...page, blocks: nextBlocks };
      });

      if (!anyChanged) return state;

      const history = buildNextHistoryState(
        state,
        nextPages,
        state.activePageId,
        state.selectedLayerIds,
        { recordHistory: true }
      );
      return { pages: nextPages, history };
    });

    return { replacedBlocks, affectedPages };
  },

  hydratePipelineConfig: (backendSettings, { forceAppearance = false } = {}) => {
    if (!backendSettings || typeof backendSettings !== 'object') return;
    const recentlyMutatedAppearance =
      !forceAppearance && Date.now() - lastLocalAppearanceMutationAt < 3000;

    set((state) => {
      const nextTheme =
        !recentlyMutatedAppearance &&
        VALID_THEMES.includes(backendSettings.theme)
          ? backendSettings.theme
          : state.theme;
      const nextAccent =
        !recentlyMutatedAppearance &&
        VALID_ACCENTS.includes(backendSettings.accent_color)
          ? backendSettings.accent_color
          : state.accentColor;
      const nextDensity =
        !recentlyMutatedAppearance &&
        VALID_DENSITIES.includes(backendSettings.ui_density)
          ? backendSettings.ui_density
          : state.uiDensity;

      const resolvedTheme = applyThemeToDocument(nextTheme);
      applyAccentToDocument(nextAccent);
      applyDensityToDocument(nextDensity);

      try {
        if (typeof window !== 'undefined') {
          window.localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
          window.localStorage.setItem(ACCENT_STORAGE_KEY, nextAccent);
          window.localStorage.setItem(DENSITY_STORAGE_KEY, nextDensity);
        }
      } catch {
        // Ignore storage write errors
      }

      const mergedProfiles = {
        ...buildDefaultProviderProfiles(),
        ...(state.pipelineConfig.providerProfiles || {}),
        ...(backendSettings.providers && typeof backendSettings.providers === 'object'
          ? backendSettings.providers
          : {}),
      };

      const nextChain =
        Array.isArray(backendSettings.font_fallback_chain) &&
        backendSettings.font_fallback_chain.length > 0
          ? [...backendSettings.font_fallback_chain]
          : state.typesetting.fontFallbackChain;

      const nextTypesetting = {
        ...state.typesetting,
        fontFallbackChain: nextChain,
        fontFamily:
          backendSettings.font_family || formatFontFamilyChain(nextChain),
        autoFitFontSize:
          backendSettings.auto_fit_font_size !== undefined
            ? Boolean(backendSettings.auto_fit_font_size)
            : state.typesetting.autoFitFontSize,
        enableHyphenation:
          backendSettings.enable_hyphenation !== undefined
            ? Boolean(backendSettings.enable_hyphenation)
            : state.typesetting.enableHyphenation ?? true,
        defaultFontSize:
          backendSettings.default_font_size ??
          state.typesetting.defaultFontSize,
        defaultStrokeWidth:
          backendSettings.default_stroke_width !== undefined
            ? Number(backendSettings.default_stroke_width)
            : state.typesetting.defaultStrokeWidth,
        defaultStrokeColor:
          backendSettings.default_stroke_color ??
          state.typesetting.defaultStrokeColor,
        defaultTextColor:
          backendSettings.default_text_color ??
          state.typesetting.defaultTextColor,
        defaultLineHeight:
          backendSettings.default_line_height !== undefined
            ? Number(backendSettings.default_line_height)
            : state.typesetting.defaultLineHeight,
        defaultAlign:
          backendSettings.default_align ?? state.typesetting.defaultAlign,
        allCaps:
          backendSettings.all_caps !== undefined
            ? Boolean(backendSettings.all_caps)
            : state.typesetting.allCaps,
        autoTiltText:
          backendSettings.auto_tilt_text !== undefined
            ? Boolean(backendSettings.auto_tilt_text)
            : state.typesetting.autoTiltText ?? false,
      };

      const nextPipeline = {
        ...state.pipelineConfig,
        provider:
          backendSettings.active_provider ??
          backendSettings.provider ??
          state.pipelineConfig.provider,
        baseUrl: backendSettings.base_url ?? state.pipelineConfig.baseUrl,
        apiKey: backendSettings.api_key ?? state.pipelineConfig.apiKey,
        model: backendSettings.model ?? state.pipelineConfig.model,
        providerProfiles: mergedProfiles,
        sourceLang:
          backendSettings.source_lang ?? state.pipelineConfig.sourceLang,
        targetLang:
          backendSettings.target_lang ?? state.pipelineConfig.targetLang,
        temperature:
          backendSettings.temperature ?? state.pipelineConfig.temperature,
        topP:
          backendSettings.top_p !== undefined
            ? backendSettings.top_p
            : state.pipelineConfig.topP,
        topK:
          backendSettings.top_k !== undefined
            ? backendSettings.top_k
            : state.pipelineConfig.topK,
        maxTokens:
          backendSettings.max_tokens ?? state.pipelineConfig.maxTokens,
        frequencyPenalty:
          backendSettings.frequency_penalty !== undefined
            ? backendSettings.frequency_penalty
            : state.pipelineConfig.frequencyPenalty,
        reasoningEffort:
          backendSettings.reasoning_effort ??
          state.pipelineConfig.reasoningEffort,
        stripThinking:
          backendSettings.strip_thinking !== undefined
            ? Boolean(backendSettings.strip_thinking)
            : state.pipelineConfig.stripThinking,
        timeout: backendSettings.timeout ?? state.pipelineConfig.timeout,
        useVision:
          backendSettings.use_vision !== undefined
            ? Boolean(backendSettings.use_vision)
            : state.pipelineConfig.useVision,
        visionMaxSide:
          backendSettings.vision_max_side ?? state.pipelineConfig.visionMaxSide,
        customInstructions:
          backendSettings.custom_instructions ??
          state.pipelineConfig.customInstructions,
        translationTone:
          backendSettings.translation_tone ??
          state.pipelineConfig.translationTone ??
          'standard',
        honorificsPolicy:
          backendSettings.honorifics_policy ??
          state.pipelineConfig.honorificsPolicy ??
          'keep',
        glossary:
          backendSettings.glossary && typeof backendSettings.glossary === 'object'
            ? { ...backendSettings.glossary }
            : state.pipelineConfig.glossary,
        detectionModel:
          backendSettings.detection_model ?? state.pipelineConfig.detectionModel,
        detectorPrecision: ['fast', 'balanced', 'high_precision'].includes(
          backendSettings.detector_precision
        )
          ? backendSettings.detector_precision
          : state.pipelineConfig.detectorPrecision || 'balanced',
        textThreshold:
          backendSettings.text_threshold !== undefined
            ? normalizeOptionalThreshold(backendSettings.text_threshold)
            : state.pipelineConfig.textThreshold,
        bubbleThreshold:
          backendSettings.bubble_threshold !== undefined
            ? normalizeOptionalThreshold(backendSettings.bubble_threshold)
            : state.pipelineConfig.bubbleThreshold,
        panelThreshold:
          backendSettings.panel_threshold !== undefined
            ? normalizeOptionalThreshold(backendSettings.panel_threshold)
            : state.pipelineConfig.panelThreshold,
        whisperThreshold:
          backendSettings.whisper_threshold !== undefined
            ? normalizeOptionalThreshold(backendSettings.whisper_threshold)
            : state.pipelineConfig.whisperThreshold,
        borderlessThreshold:
          backendSettings.borderless_threshold !== undefined
            ? normalizeOptionalThreshold(backendSettings.borderless_threshold)
            : state.pipelineConfig.borderlessThreshold,
        enableAdaptiveTiling:
          backendSettings.enable_adaptive_tiling !== undefined
            ? Boolean(backendSettings.enable_adaptive_tiling)
            : state.pipelineConfig.enableAdaptiveTiling,
        sfxSuppressIou:
          backendSettings.sfx_suppress_iou !== undefined
            ? Number(backendSettings.sfx_suppress_iou)
            : state.pipelineConfig.sfxSuppressIou,
        ocrModel:
          backendSettings.ocr_model ?? state.pipelineConfig.ocrModel,
        inpaintingModel:
          backendSettings.inpainting_model ?? state.pipelineConfig.inpaintingModel,
        inpaintApiUrl:
          backendSettings.inpaint_api_url ?? state.pipelineConfig.inpaintApiUrl,
        executionMode:
          backendSettings.execution_mode ?? state.pipelineConfig.executionMode,
        preferredBackend:
          backendSettings.preferred_backend ??
          state.pipelineConfig.preferredBackend ??
          'auto',
        confThreshold:
          backendSettings.conf_threshold ?? state.pipelineConfig.confThreshold,
        nmsThreshold:
          backendSettings.nms_threshold ?? state.pipelineConfig.nmsThreshold,
        maskThreshold:
          backendSettings.mask_threshold ?? state.pipelineConfig.maskThreshold,
        bboxPadding:
          backendSettings.bbox_padding ?? state.pipelineConfig.bboxPadding,
        lamaTileSize:
          backendSettings.lama_tile_size ?? state.pipelineConfig.lamaTileSize,
        readingOrder:
          backendSettings.reading_order ?? state.pipelineConfig.readingOrder,
        autoUnloadVram:
          backendSettings.auto_unload_vram !== undefined
            ? Boolean(backendSettings.auto_unload_vram)
            : state.pipelineConfig.autoUnloadVram,
        batchScheduleMode: ['auto', 'pipelined', 'stage-first'].includes(
          backendSettings.batch_schedule_mode
        )
          ? backendSettings.batch_schedule_mode
          : state.pipelineConfig.batchScheduleMode || 'auto',
        refineMask:
          backendSettings.refine_mask !== undefined
            ? Boolean(backendSettings.refine_mask)
            : state.pipelineConfig.refineMask,
        enable3TierInpaint:
          backendSettings.enable_3tier_inpaint !== undefined
            ? Boolean(backendSettings.enable_3tier_inpaint)
            : state.pipelineConfig.enable3TierInpaint ?? true,
        enableToneMatch:
          backendSettings.enable_tone_match !== undefined
            ? Boolean(backendSettings.enable_tone_match)
            : state.pipelineConfig.enableToneMatch ?? true,
        enableCvStyleDetect:
          backendSettings.enable_cv_style_detect !== undefined
            ? Boolean(backendSettings.enable_cv_style_detect)
            : state.pipelineConfig.enableCvStyleDetect ?? true,
        enableWaistCut:
          backendSettings.enable_waist_cut !== undefined
            ? Boolean(backendSettings.enable_waist_cut)
            : state.pipelineConfig.enableWaistCut ?? true,
        autoDeskewScans:
          backendSettings.auto_deskew_scans !== undefined
            ? Boolean(backendSettings.auto_deskew_scans)
            : state.pipelineConfig.autoDeskewScans ?? false,
        autoMaskFuriganaOcr:
          backendSettings.auto_mask_furigana_ocr !== undefined
            ? Boolean(backendSettings.auto_mask_furigana_ocr)
            : state.pipelineConfig.autoMaskFuriganaOcr ?? true,
        skipBlankPages:
          backendSettings.skip_blank_pages !== undefined
            ? Boolean(backendSettings.skip_blank_pages)
            : state.pipelineConfig.skipBlankPages ?? true,
        smartSsimExport:
          backendSettings.smart_ssim_export !== undefined
            ? Boolean(backendSettings.smart_ssim_export)
            : state.pipelineConfig.smartSsimExport ?? true,
      };

      const nextPresets =
        Array.isArray(backendSettings.available_providers) &&
        backendSettings.available_providers.length > 0
          ? backendSettings.available_providers.map((bp) => {
              const fallback = DEFAULT_PROVIDER_PRESETS.find(
                (dp) => dp.id === bp.id
              );
              return { ...(fallback || {}), ...bp };
            })
          : state.availableProviderPresets;

      return {
        theme: nextTheme,
        resolvedTheme,
        accentColor: nextAccent,
        uiDensity: nextDensity,
        availableProviderPresets: nextPresets,
        supportedTargetLanguages:
          Array.isArray(backendSettings.supported_target_languages) &&
          backendSettings.supported_target_languages.length > 0
            ? backendSettings.supported_target_languages
            : state.supportedTargetLanguages,
        pipelineConfig: nextPipeline,
        typesetting: nextTypesetting,
      };
    });

    get()._writeLocalSnapshot();
  },

  buildBackendSettingsPayload: () => {
    const s = get();
    const p = s.pipelineConfig;
    const t = s.typesetting;
    return {
      active_provider: p.provider || 'none',
      provider: p.provider || 'none',
      base_url: p.baseUrl,
      api_key: p.apiKey,
      model: p.model,
      timeout: Number(p.timeout) || 90,
      source_lang: p.sourceLang,
      target_lang: p.targetLang,
      translation_tone: p.translationTone || 'standard',
      honorifics_policy: p.honorificsPolicy || 'keep',
      temperature: Number(p.temperature),
      top_p: p.topP === null || p.topP === '' ? null : Number(p.topP),
      top_k: p.topK === null || p.topK === '' ? null : Math.round(Number(p.topK)),
      max_tokens: Math.round(Number(p.maxTokens) || 2048),
      frequency_penalty:
        p.frequencyPenalty === null || p.frequencyPenalty === ''
          ? null
          : Number(p.frequencyPenalty),
      reasoning_effort: p.reasoningEffort || 'off',
      strip_thinking: Boolean(p.stripThinking),
      use_vision: Boolean(p.useVision),
      vision_max_side: Math.round(Number(p.visionMaxSide) || 1024),
      custom_instructions: p.customInstructions || '',
      glossary: p.glossary || {},
      providers: p.providerProfiles || buildDefaultProviderProfiles(),
      theme: s.theme,
      accent_color: s.accentColor,
      ui_density: s.uiDensity,
      detection_model: p.detectionModel || 'vellum-det-yolo11m-seg-1024-fp16',
      detector_precision: ['fast', 'balanced', 'high_precision'].includes(
        p.detectorPrecision
      )
        ? p.detectorPrecision
        : 'balanced',
      text_threshold: normalizeOptionalThreshold(p.textThreshold),
      bubble_threshold: normalizeOptionalThreshold(p.bubbleThreshold),
      panel_threshold: normalizeOptionalThreshold(p.panelThreshold),
      whisper_threshold: normalizeOptionalThreshold(p.whisperThreshold),
      borderless_threshold: normalizeOptionalThreshold(p.borderlessThreshold),
      enable_adaptive_tiling: Boolean(p.enableAdaptiveTiling ?? true),
      sfx_suppress_iou: Number(p.sfxSuppressIou ?? 0.60),
      ocr_model: p.ocrModel || 'auto',
      inpainting_model: p.inpaintingModel || 'lama-manga',
      inpaint_api_url: p.inpaintApiUrl || 'http://127.0.0.1:8188',
      execution_mode: p.executionMode || 'hybrid',
      preferred_backend: p.preferredBackend || 'auto',
      conf_threshold: Number(p.confThreshold ?? 0.35),
      nms_threshold: Number(p.nmsThreshold ?? 0.45),
      mask_threshold: Number(p.maskThreshold ?? 0.5),
      bbox_padding: Math.round(Number(p.bboxPadding ?? 6)),
      lama_tile_size: Math.round(Number(p.lamaTileSize ?? 512)),
      reading_order: p.readingOrder || 'auto',
      auto_unload_vram: Boolean(p.autoUnloadVram),
      batch_schedule_mode: p.batchScheduleMode || 'auto',
      refine_mask: Boolean(p.refineMask),
      enable_3tier_inpaint: Boolean(p.enable3TierInpaint ?? true),
      enable_tone_match: Boolean(p.enableToneMatch ?? true),
      enable_cv_style_detect: Boolean(p.enableCvStyleDetect ?? true),
      enable_waist_cut: Boolean(p.enableWaistCut ?? true),
      auto_deskew_scans: Boolean(p.autoDeskewScans ?? false),
      auto_mask_furigana_ocr: Boolean(p.autoMaskFuriganaOcr ?? true),
      skip_blank_pages: Boolean(p.skipBlankPages ?? true),
      smart_ssim_export: Boolean(p.smartSsimExport ?? true),
      font_family: t.fontFamily || formatFontFamilyChain(t.fontFallbackChain),
      font_fallback_chain: [...(t.fontFallbackChain || DEFAULT_FONT_FALLBACK_CHAIN)],
      auto_fit_font_size: Boolean(t.autoFitFontSize),
      enable_hyphenation: Boolean(t.enableHyphenation ?? true),
      default_font_size: Math.round(Number(t.defaultFontSize ?? 18)),
      default_stroke_width: Number(t.defaultStrokeWidth ?? 4),
      default_stroke_color: t.defaultStrokeColor || '#ffffff',
      default_text_color: t.defaultTextColor || '#000000',
      default_line_height: Number(t.defaultLineHeight ?? 1.18),
      default_align: t.defaultAlign || 'center',
      all_caps: Boolean(t.allCaps),
      auto_tilt_text: Boolean(t.autoTiltText ?? false),
    };
  },

  persistSettingsToBackend: async (partialOverride = null) => {
    const state = get();
    const apiBaseUrl = state.pipelineConfig.apiBaseUrl ?? DEFAULT_API_BASE_URL;
    const body =
      partialOverride && typeof partialOverride === 'object'
        ? partialOverride
        : state.buildBackendSettingsPayload();

    const resp = await fetch(`${apiBaseUrl}/api/settings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try {
        const errJson = await resp.json();
        if (errJson?.detail) detail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      } catch {
        // ignore
      }
      throw new Error(detail);
    }
    const saved = await resp.json();
    get().hydratePipelineConfig(saved, { forceAppearance: true });
    return saved;
  },

  resetPipelineConfig: () => {
    set({
      pipelineConfig: {
        ...DEFAULT_PIPELINE_CONFIG,
        enabledStages: { ...DEFAULT_ENABLED_STAGES },
        providerProfiles: buildDefaultProviderProfiles(),
      },
      typesetting: {
        ...DEFAULT_TYPESETTING_CONFIG,
        fontFallbackChain: [...DEFAULT_FONT_FALLBACK_CHAIN],
      },
    });
    get()._writeLocalSnapshot();
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // Runtime Status & Canvas Telemetry Actions
  // ═══════════════════════════════════════════════════════════════════════════

  setModelsStatus: (data) => {
    if (!data) return;
    set((state) => ({
      runtimeStatus: {
        ...state.runtimeStatus,
        executionProvider:
          data.execution_provider ?? state.runtimeStatus.executionProvider,
        executionMode:
          data.execution_mode ?? state.runtimeStatus.executionMode,
        detectorProvider:
          data.detector_provider ?? state.runtimeStatus.detectorProvider,
        ocrProvider:
          data.ocr_provider ?? state.runtimeStatus.ocrProvider,
        inpainterProvider:
          data.inpainter_provider ?? state.runtimeStatus.inpainterProvider,
        detectorModel:
          data.detector_model ?? state.runtimeStatus.detectorModel,
        availableProviders:
          data.available_providers ?? state.runtimeStatus.availableProviders,
        modelsReady:
          data.all_ready !== undefined
            ? Boolean(data.all_ready)
            : state.runtimeStatus.modelsReady,
        modelsStatus: data,
        lmStudio: data.lm_studio
          ? {
              connected: Boolean(data.lm_studio.connected),
              endpoint:
                data.lm_studio.endpoint ||
                state.runtimeStatus.lmStudio.endpoint,
              activeModel: data.lm_studio.active_model ?? null,
              resolvedModelId: data.lm_studio.resolved_model_id ?? null,
              availableModels: data.lm_studio.available_models ?? [],
              error: data.lm_studio.error ?? null,
            }
          : state.runtimeStatus.lmStudio,
      },
    }));
  },

  setLmStudioStatus: (data) => {
    if (!data) return;
    set((state) => ({
      runtimeStatus: {
        ...state.runtimeStatus,
        lmStudio: {
          connected: Boolean(data.connected),
          endpoint: data.endpoint || state.runtimeStatus.lmStudio.endpoint,
          activeModel: data.active_model ?? null,
          resolvedModelId: data.resolved_model_id ?? null,
          availableModels: data.available_models ?? [],
          error: data.error ?? null,
        },
      },
    }));
  },

  setHardwareInfo: (hwData) => {
    if (!hwData || typeof hwData !== 'object') return;
    set((state) => ({
      runtimeStatus: {
        ...state.runtimeStatus,
        executionProvider:
          hwData.execution_provider ?? state.runtimeStatus.executionProvider,
        executionMode:
          hwData.execution_mode ?? state.runtimeStatus.executionMode,
        availableProviders:
          Array.isArray(hwData.available_providers) && hwData.available_providers.length > 0
            ? hwData.available_providers
            : state.runtimeStatus.availableProviders,
        hardware: hwData,
      },
    }));
  },

  setHealthStatus: (data) => {
    if (!data) return;
    set((state) => ({
      runtimeStatus: {
        ...state.runtimeStatus,
        executionProvider:
          data.execution_provider ?? state.runtimeStatus.executionProvider,
        executionMode:
          data.execution_mode ?? state.runtimeStatus.executionMode,
        detectorProvider:
          data.detector_provider ?? state.runtimeStatus.detectorProvider,
        ocrProvider:
          data.ocr_provider ?? state.runtimeStatus.ocrProvider,
        inpainterProvider:
          data.inpainter_provider ?? state.runtimeStatus.inpainterProvider,
        detectorModel:
          data.detector_model ?? state.runtimeStatus.detectorModel,
        availableProviders:
          data.available_providers ?? state.runtimeStatus.availableProviders,
        modelsReady:
          data.models_ready !== undefined
            ? Boolean(data.models_ready)
            : state.runtimeStatus.modelsReady,
        activeProviderSummary:
          data.active_provider ?? state.runtimeStatus.activeProviderSummary,
        lmStudio: data.lm_studio
          ? {
              connected: Boolean(data.lm_studio.connected),
              endpoint:
                data.lm_studio.endpoint ||
                state.runtimeStatus.lmStudio.endpoint,
              activeModel: data.lm_studio.active_model ?? null,
              resolvedModelId: data.lm_studio.resolved_model_id ?? null,
              availableModels: data.lm_studio.available_models ?? [],
              error: data.lm_studio.error ?? null,
            }
          : state.runtimeStatus.lmStudio,
      },
    }));
  },

  setCanvasZoom: (zoom) => {
    const numeric = Number(zoom);
    if (!Number.isFinite(numeric) || numeric <= 0) return;
    const clamped = Number(Math.min(4.5, Math.max(0.15, numeric)).toFixed(4));
    set((state) => {
      if (state.canvasView.zoom === clamped) return state;
      return {
        canvasView: {
          ...state.canvasView,
          zoom: clamped,
        },
      };
    });
  },

  zoomIn: () => {
    const current = get().canvasView.zoom || 1;
    get().setCanvasZoom(current * 1.18);
  },

  zoomOut: () => {
    const current = get().canvasView.zoom || 1;
    get().setCanvasZoom(current / 1.18);
  },

  setCursorPosition: (x, y) => {
    set((state) => {
      if (state.canvasView.cursorX === x && state.canvasView.cursorY === y) {
        return state;
      }
      return {
        canvasView: {
          ...state.canvasView,
          cursorX: x,
          cursorY: y,
        },
      };
    });
  },

  clearCursorPosition: () => {
    set((state) => {
      if (
        state.canvasView.cursorX === null &&
        state.canvasView.cursorY === null
      ) {
        return state;
      }
      return {
        canvasView: {
          ...state.canvasView,
          cursorX: null,
          cursorY: null,
        },
      };
    });
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // Appearance Actions (Theme, Accent Preset, UI Density)
  // ═══════════════════════════════════════════════════════════════════════════

  setTheme: (nextTheme, { syncBackend = true } = {}) => {
    if (!VALID_THEMES.includes(nextTheme)) return;
    lastLocalAppearanceMutationAt = Date.now();
    try {
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
      }
    } catch {
      // Ignore storage persistence errors
    }
    const resolvedTheme = applyThemeToDocument(nextTheme);
    set({ theme: nextTheme, resolvedTheme });
    get()._writeLocalSnapshot();
    if (syncBackend) {
      const apiBaseUrl =
        get().pipelineConfig.apiBaseUrl ?? DEFAULT_API_BASE_URL;
      fetch(`${apiBaseUrl}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ theme: nextTheme }),
      }).catch(() => {});
    }
  },

  setAccentColor: (nextAccent, { syncBackend = true } = {}) => {
    if (!VALID_ACCENTS.includes(nextAccent)) return;
    lastLocalAppearanceMutationAt = Date.now();
    try {
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(ACCENT_STORAGE_KEY, nextAccent);
      }
    } catch {
      // Ignore storage persistence errors
    }
    const applied = applyAccentToDocument(nextAccent);
    set({ accentColor: applied });
    get()._writeLocalSnapshot();
    if (syncBackend) {
      const apiBaseUrl =
        get().pipelineConfig.apiBaseUrl ?? DEFAULT_API_BASE_URL;
      fetch(`${apiBaseUrl}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ accent_color: applied }),
      }).catch(() => {});
    }
  },

  setUiDensity: (nextDensity, { syncBackend = true } = {}) => {
    if (!VALID_DENSITIES.includes(nextDensity)) return;
    lastLocalAppearanceMutationAt = Date.now();
    try {
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(DENSITY_STORAGE_KEY, nextDensity);
      }
    } catch {
      // Ignore storage persistence errors
    }
    const applied = applyDensityToDocument(nextDensity);
    set({ uiDensity: applied });
    get()._writeLocalSnapshot();
    if (syncBackend) {
      const apiBaseUrl =
        get().pipelineConfig.apiBaseUrl ?? DEFAULT_API_BASE_URL;
      fetch(`${apiBaseUrl}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ui_density: applied }),
      }).catch(() => {});
    }
  },

  cycleTheme: () => {
    const current = get().theme;
    const nextIdx = (VALID_THEMES.indexOf(current) + 1) % VALID_THEMES.length;
    get().setTheme(VALID_THEMES[nextIdx]);
  },

  syncSystemTheme: () => {
    const { theme, accentColor, uiDensity } = get();
    const resolvedTheme = applyThemeToDocument(theme);
    applyAccentToDocument(accentColor);
    applyDensityToDocument(uiDensity);
    set((state) =>
      state.resolvedTheme === resolvedTheme ? state : { resolvedTheme }
    );
  },
}));

/**
 * Subscribe to OS `prefers-color-scheme` changes so `theme === 'system'` updates
 * immediately when Windows/macOS switches between Dark and Light mode.
 * Returns a cleanup function suitable for `useEffect`.
 */
export function bindSystemThemeListener() {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return () => {};
  }

  // Ensure DOM reflects current store theme on mount
  useEditorStore.getState().syncSystemTheme();

  const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
  const handleChange = () => {
    useEditorStore.getState().syncSystemTheme();
  };

  if (typeof mediaQuery.addEventListener === 'function') {
    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }
  mediaQuery.addListener(handleChange);
  return () => mediaQuery.removeListener(handleChange);
}

export default useEditorStore;
