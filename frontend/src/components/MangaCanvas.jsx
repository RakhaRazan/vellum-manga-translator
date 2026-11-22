import React, {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from 'react';
import Konva from 'konva';
import {
  Stage,
  Layer,
  Group as KonvaGroup,
  Image as KonvaImage,
  Line as KonvaLine,
  Text as KonvaText,
  Rect as KonvaRect,
  Transformer,
} from 'react-konva';
import {
  ChevronsLeftRight,
  Columns2,
  Eye,
  FileImage,
  Hand,
  Loader2,
  Maximize2,
  MousePointer2,
  Redo2,
  Sparkles,
  Type,
  Undo2,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import useEditorStore, {
  DEFAULT_API_BASE_URL,
  DEFAULT_FONT_FAMILY,
  buildKonvaFontFamily,
  ensurePageFontsLoaded,
} from '../store/useEditorStore.js';

// ============================================================================
// Constants & Shared Helpers
// ============================================================================

const MIN_ZOOM = 0.15;
const MAX_ZOOM = 4.5;
const MAX_IMAGE_CACHE_ENTRIES = 36;
const MAX_MASK_CACHE_ENTRIES = 16;

const IMAGE_ELEMENT_CACHE = new Map();
const MASK_CANVAS_CACHE = new Map();

function clampZoom(scale) {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, scale));
}

function clampSplitRatio(ratio) {
  return Math.min(0.98, Math.max(0.02, ratio));
}

/**
 * Resolve relative `/api/...` URLs against `apiBaseUrl` while preserving
 * `blob:`, `data:`, and absolute `http(s):` URLs.
 */
export function resolveImageSrc(src, apiBaseUrl = DEFAULT_API_BASE_URL) {
  if (!src || typeof src !== 'string') return null;
  const trimmed = src.trim();
  if (!trimmed) return null;
  if (/^(blob:|data:|https?:\/\/)/i.test(trimmed)) {
    return trimmed;
  }
  if (trimmed.startsWith('/api/')) {
    const base = String(apiBaseUrl || '').replace(/\/+$/, '');
    return base ? `${base}${trimmed}` : trimmed;
  }
  return trimmed;
}

/**
 * Prefer lightweight cached `/api/cache/images/...` URLs over inline base64 strings
 * to keep canvas rendering & memory usage light at 60fps.
 */
export function pickInpaintedSrc(page, apiBaseUrl = DEFAULT_API_BASE_URL) {
  if (!page) return null;
  const candidates = [
    page.cleanedImageUrl,
    page.inpaintedImageUrl,
    page.inpaintedUrl,
    page.cleaned_image_url,
    page.inpainted_image_url,
    page.cleanedImage,
    page.inpaintedImage,
  ].filter((c) => typeof c === 'string' && c.trim().length > 0);

  if (candidates.length === 0) return null;

  // Prefer cached URL paths (`/api/cache/images/...` or `http...`) over base64 data URIs
  const cachedFirst = candidates.find(
    (c) => !c.trim().toLowerCase().startsWith('data:')
  );
  return resolveImageSrc(cachedFirst || candidates[0], apiBaseUrl);
}

/**
 * Load an `HTMLImageElement` with async decoding, CORS support for `/api/cache/images/...`,
 * and bounded in-memory LRU caching to eliminate decode stutter on toggle/page switch.
 */
export function loadImageElement(src) {
  if (!src) return Promise.resolve(null);

  const cached = IMAGE_ELEMENT_CACHE.get(src);
  if (cached && cached.complete && cached.naturalWidth > 0) {
    // Refresh LRU order
    IMAGE_ELEMENT_CACHE.delete(src);
    IMAGE_ELEMENT_CACHE.set(src, cached);
    return Promise.resolve(cached);
  }

  return new Promise((resolve) => {
    const loadedImageEl = new window.Image();
    loadedImageEl.decoding = 'async';
    if (!/^(data:|blob:)/i.test(src)) {
      loadedImageEl.crossOrigin = 'anonymous';
    }
    loadedImageEl.onload = () => {
      if (IMAGE_ELEMENT_CACHE.size >= MAX_IMAGE_CACHE_ENTRIES) {
        const oldestKey = IMAGE_ELEMENT_CACHE.keys().next().value;
        if (oldestKey) IMAGE_ELEMENT_CACHE.delete(oldestKey);
      }
      IMAGE_ELEMENT_CACHE.set(src, loadedImageEl);
      resolve(loadedImageEl);
    };
    loadedImageEl.onerror = () => {
      resolve(null);
    };
    loadedImageEl.src = src;
  });
}

function useLoadedImage(src) {
  const [state, setState] = useState(() => {
    if (!src) return { src: null, img: null };
    const cached = IMAGE_ELEMENT_CACHE.get(src);
    return cached && cached.complete && cached.naturalWidth > 0
      ? { src, img: cached }
      : { src, img: null };
  });

  useEffect(() => {
    if (!src) {
      setState({ src: null, img: null });
      return;
    }

    const cached = IMAGE_ELEMENT_CACHE.get(src);
    if (cached && cached.complete && cached.naturalWidth > 0) {
      setState((prev) =>
        prev.src === src && prev.img === cached ? prev : { src, img: cached }
      );
      return;
    }

    let cancelled = false;
    loadImageElement(src).then((img) => {
      if (!cancelled) {
        setState({ src, img });
      }
    });

    return () => {
      cancelled = true;
    };
  }, [src]);

  return state.src === src ? state.img : null;
}

/**
 * Convert a binary `(H, W)` grayscale mask PNG (`/api/cache/images/mask_....png`)
 * into a transparent offscreen `<canvas>` tinted crimson where mask pixels > 0.
 * Cached by `maskSrc` so toggling the Inpaint Mask layer is instantaneous at 60fps.
 */
function useMaskOverlayCanvas(maskSrc, maskImg) {
  return useMemo(() => {
    if (!maskSrc || !maskImg) return null;
    const cached = MASK_CANVAS_CACHE.get(maskSrc);
    if (cached) return cached;

    const w = maskImg.naturalWidth || maskImg.width;
    const h = maskImg.naturalHeight || maskImg.height;
    if (!w || !h) return null;

    const offscreen = document.createElement('canvas');
    offscreen.width = w;
    offscreen.height = h;
    const ctx = offscreen.getContext('2d', { willReadFrequently: true });
    if (!ctx) return null;

    try {
      ctx.drawImage(maskImg, 0, 0, w, h);
      const imageData = ctx.getImageData(0, 0, w, h);
      const data = imageData.data;

      for (let i = 0; i < data.length; i += 4) {
        const lum = data[i];
        if (lum > 32) {
          // Vivid crimson/rose translucent overlay for masked text regions
          data[i] = 239;
          data[i + 1] = 68;
          data[i + 2] = 68;
          data[i + 3] = Math.round((lum / 255) * 145);
        } else {
          data[i] = 0;
          data[i + 1] = 0;
          data[i + 2] = 0;
          data[i + 3] = 0;
        }
      }

      ctx.putImageData(imageData, 0, 0);
      if (MASK_CANVAS_CACHE.size >= MAX_MASK_CACHE_ENTRIES) {
        const oldestKey = MASK_CANVAS_CACHE.keys().next().value;
        if (oldestKey) MASK_CANVAS_CACHE.delete(oldestKey);
      }
      MASK_CANVAS_CACHE.set(maskSrc, offscreen);
      return offscreen;
    } catch {
      // Fallback if CORS prevented getImageData
      return null;
    }
  }, [maskSrc, maskImg]);
}

/**
 * Extract canonical geometry `{ x, y, width, height, rotation }` from a block.
 * Strictly constrains unpositioned blocks to `safe_bbox` (or `bubble_bbox` inset by 12%),
 * never expanding wider than the actual speech bubble.
 */
export function getBlockGeometry(block) {
  const rotation = typeof block?.rotation === 'number' ? block.rotation : 0;

  if (
    typeof block?.x === 'number' &&
    typeof block?.y === 'number' &&
    typeof block?.width === 'number' &&
    typeof block?.height === 'number'
  ) {
    return {
      x: block.x,
      y: block.y,
      width: Math.max(20, block.width),
      height: Math.max(20, block.height),
      rotation,
    };
  }

  if (
    Array.isArray(block?.safe_bbox) &&
    block.safe_bbox.length >= 4 &&
    Number(block.safe_bbox[2]) > Number(block.safe_bbox[0]) &&
    Number(block.safe_bbox[3]) > Number(block.safe_bbox[1])
  ) {
    const sx1 = Number(block.safe_bbox[0]) || 0;
    const sy1 = Number(block.safe_bbox[1]) || 0;
    const sx2 = Number(block.safe_bbox[2]) || sx1 + 40;
    const sy2 = Number(block.safe_bbox[3]) || sy1 + 24;
    return {
      x: sx1,
      y: sy1,
      width: Math.max(20, sx2 - sx1),
      height: Math.max(20, sy2 - sy1),
      rotation,
    };
  }

  if (
    Array.isArray(block?.bubble_bbox) &&
    block.bubble_bbox.length >= 4 &&
    Number(block.bubble_bbox[2]) > Number(block.bubble_bbox[0]) &&
    Number(block.bubble_bbox[3]) > Number(block.bubble_bbox[1])
  ) {
    const bx1 = Number(block.bubble_bbox[0]) || 0;
    const by1 = Number(block.bubble_bbox[1]) || 0;
    const bx2 = Number(block.bubble_bbox[2]) || bx1 + 40;
    const by2 = Number(block.bubble_bbox[3]) || by1 + 24;
    const bw = Math.max(20, bx2 - bx1);
    const bh = Math.max(20, by2 - by1);
    const padX = Math.round(bw * 0.08);
    const padY = Math.round(bh * 0.08);
    const ix1 = bx1 + padX;
    const iy1 = by1 + padY;
    const ix2 = Math.max(ix1 + 16, bx2 - padX);
    const iy2 = Math.max(iy1 + 16, by2 - padY);
    return {
      x: ix1,
      y: iy1,
      width: Math.max(16, ix2 - ix1),
      height: Math.max(16, iy2 - iy1),
      rotation,
    };
  }

  const x = typeof block?.x === 'number' ? block.x : block?.bbox?.[0] ?? 0;
  const y = typeof block?.y === 'number' ? block.y : block?.bbox?.[1] ?? 0;
  const width =
    typeof block?.width === 'number'
      ? Math.max(20, block.width)
      : Math.max(20, (block?.bbox?.[2] ?? 140) - (block?.bbox?.[0] ?? 0));
  const height =
    typeof block?.height === 'number'
      ? Math.max(20, block.height)
      : Math.max(20, (block?.bbox?.[3] ?? 60) - (block?.bbox?.[1] ?? 0));

  return { x, y, width, height, rotation };
}

let _measureCanvasCtx = null;
function getMeasureCanvasCtx() {
  if (!_measureCanvasCtx && typeof document !== 'undefined') {
    const c = document.createElement('canvas');
    _measureCanvasCtx = c.getContext('2d');
  }
  return _measureCanvasCtx;
}

/**
 * Compute resolution-proportional font bounds from actual page pixel width.
 * On a 2580px-wide page with defaultFontSize=18: effectiveMaxFont ~ 46px, effectiveMinFont ~ 34px.
 */
export function getResolutionFontBounds(imgWidth = 1000, defaultFontSize = 18) {
  const resScale = Math.max(1, (Number(imgWidth) || 1000) / 1000);
  const effectiveMaxFont = Math.round((Number(defaultFontSize) || 18) * resScale);
  const effectiveMinFont = Math.max(12, Math.round(13 * resScale));
  return {
    resScale,
    effectiveMaxFont: Math.max(effectiveMinFont, effectiveMaxFont),
    effectiveMinFont,
  };
}

/**
 * 2D iterative auto-fit solver (width + height) using offscreen Canvas `measureText`
 * word-wrapping. Steps `fontSize` down from `effectiveMaxFont` by 1px until BOTH:
 *   - widest wrapped line <= boxWidth * 0.94
 *   - total wrapped paragraph height <= boxHeight * 0.94
 * If a dense paragraph still overflows at `effectiveMinFont`, steps further down safely
 * so text never overflows outside speech bubble ink borders.
 */
export function fitTextToBox(
  text,
  boxWidth,
  boxHeight,
  fontFamily,
  effectiveMaxFont = 18,
  effectiveMinFontOrOpts = 12,
  extraOpts = {}
) {
  const opts =
    typeof effectiveMinFontOrOpts === 'object' && effectiveMinFontOrOpts !== null
      ? effectiveMinFontOrOpts
      : { minFontSize: effectiveMinFontOrOpts, ...extraOpts };
  const lineHeight = opts.lineHeight ?? 1.18;
  const cleanText = String(text || '').trim();
  const minSize = Math.max(10, Math.round(Number(opts.minFontSize) || 12));
  const startSize = Math.max(minSize, Math.round(Number(effectiveMaxFont) || 18));
  const ctx = getMeasureCanvasCtx();
  if (!ctx || !cleanText) {
    return { fontSize: startSize, text: cleanText };
  }

  const availW = Math.max(12, Math.min(boxWidth * 0.94, boxWidth - 4));
  const availH = Math.max(12, Math.min(boxHeight * 0.94, boxHeight - 4));
  const paragraphs = cleanText.split(/\r?\n/);

  const breakLongWord = (word) => {
    const chunks = [];
    let chunk = '';
    for (let i = 0; i < word.length; i += 1) {
      const next = chunk + word[i];
      if (chunk.length > 0 && ctx.measureText(next).width > availW) {
        chunks.push(chunk);
        chunk = word[i];
      } else {
        chunk = next;
      }
    }
    if (chunk) chunks.push(chunk);
    return chunks;
  };

  const simulateWrap = (size, allowWordBreak) => {
    ctx.font = `bold ${size}px ${fontFamily || 'sans-serif'}`;
    const lines = [];
    let maxLineWidth = 0;
    let hadOversizedWord = false;

    for (const para of paragraphs) {
      const words = para.split(/\s+/).filter(Boolean);
      if (words.length === 0) {
        lines.push('');
        continue;
      }
      let currentLine = '';
      for (let i = 0; i < words.length; i += 1) {
        const word = words[i];
        const wordW = ctx.measureText(word).width;
        if (wordW > availW) {
          if (!allowWordBreak) {
            hadOversizedWord = true;
            return { fits: false, lines, maxLineWidth: wordW, hadOversizedWord };
          }
          if (currentLine) {
            const lw = ctx.measureText(currentLine).width;
            if (lw > maxLineWidth) maxLineWidth = lw;
            lines.push(currentLine);
            currentLine = '';
          }
          const pieces = breakLongWord(word);
          for (let p = 0; p < pieces.length - 1; p += 1) {
            const pw = ctx.measureText(pieces[p]).width;
            if (pw > maxLineWidth) maxLineWidth = pw;
            lines.push(pieces[p]);
          }
          currentLine = pieces[pieces.length - 1] || '';
          continue;
        }

        const candidate = currentLine ? `${currentLine} ${word}` : word;
        const candW = ctx.measureText(candidate).width;
        if (candW <= availW) {
          currentLine = candidate;
        } else {
          if (currentLine) {
            const lw = ctx.measureText(currentLine).width;
            if (lw > maxLineWidth) maxLineWidth = lw;
            lines.push(currentLine);
          }
          currentLine = word;
        }
      }
      if (currentLine) {
        const lw = ctx.measureText(currentLine).width;
        if (lw > maxLineWidth) maxLineWidth = lw;
        lines.push(currentLine);
      }
    }

    const totalHeight = lines.length * size * lineHeight;
    const fits = maxLineWidth <= availW && totalHeight <= availH;
    return { fits, lines, maxLineWidth, totalHeight, hadOversizedWord };
  };

  // Pass 1: Step down fontSize cleanly from effectiveMaxFont to effectiveMinFont without mid-word breaks
  for (let size = startSize; size >= minSize; size -= 1) {
    const res = simulateWrap(size, false);
    if (res.fits) {
      return { fontSize: size, text: cleanText };
    }
  }

  // Pass 2: Safety overflow prevention for unusually cramped/dense bubbles (never spill outside ink borders)
  const hardMinSize = Math.max(10, Math.round(minSize * 0.5));
  for (let size = minSize - 1; size >= hardMinSize; size -= 1) {
    const res = simulateWrap(size, false);
    if (res.fits) {
      return { fontSize: size, text: cleanText };
    }
  }

  // Pass 3: Allow clean long-word breaking if a single word still exceeded availW
  for (let size = startSize; size >= hardMinSize; size -= 1) {
    const res = simulateWrap(size, true);
    if (res.fits) {
      return { fontSize: size, text: res.lines.join('\n') };
    }
  }

  const fallback = simulateWrap(hardMinSize, true);
  return { fontSize: hardMinSize, text: fallback.lines.join('\n') || cleanText };
}

/**
 * Extract canonical Konva typography props shared 1:1 between live canvas and native export.
 */
export function getBlockTextStyle(block, geom = null, imgWidth = 1000) {
  const ts = useEditorStore.getState().typesetting;
  const defaultFontSize = Number(ts?.defaultFontSize) || 18;
  const { effectiveMaxFont, effectiveMinFont } = getResolutionFontBounds(
    imgWidth,
    defaultFontSize
  );
  const rawFontFamily =
    block?.style?.font_family || block?.font_family || ts?.fontFamily || DEFAULT_FONT_FAMILY;
  const fontFamily = buildKonvaFontFamily(rawFontFamily, ts?.fontFallbackChain);
  const fill = block?.color || block?.style?.color || ts?.defaultTextColor || '#000000';
  const strokeColor =
    block?.style?.stroke_color || block?.stroke_color || ts?.defaultStrokeColor || '#ffffff';
  const strokeWidth =
    block?.style?.stroke_width !== undefined
      ? block.style.stroke_width
      : block?.stroke_width !== undefined
      ? block.stroke_width
      : ts?.defaultStrokeWidth ?? 4;
  const align = block?.align || block?.style?.align || ts?.defaultAlign || 'center';
  const lineHeight = block?.style?.line_height || block?.line_height || ts?.defaultLineHeight || 1.18;
  const useAllCaps =
    block?.all_caps !== undefined
      ? Boolean(block.all_caps)
      : block?.style?.all_caps !== undefined
      ? Boolean(block.style.all_caps)
      : Boolean(ts?.allCaps);
  const rawText = block?.translated_text || '';
  let displayText = useAllCaps ? rawText.toUpperCase() : rawText;

  let fontSize = effectiveMaxFont;
  if (ts?.autoFitFontSize !== false && geom && displayText) {
    const fitted = fitTextToBox(
      displayText,
      geom.width,
      geom.height,
      fontFamily,
      effectiveMaxFont,
      effectiveMinFont,
      { lineHeight }
    );
    fontSize = fitted.fontSize;
    displayText = fitted.text;
  } else {
    const manualSize = Number(block?.font_size || block?.style?.font_size);
    if (manualSize && manualSize !== defaultFontSize) {
      fontSize = Math.max(10, Math.round(manualSize));
    }
  }

  return {
    text: displayText,
    fontSize,
    fontFamily,
    fontStyle: 'bold',
    fill,
    stroke: strokeWidth > 0 ? strokeColor : undefined,
    strokeWidth,
    fillAfterStrokeEnabled: true,
    lineJoin: 'round',
    align,
    verticalAlign: 'middle',
    lineHeight,
    wrap: 'word',
    padding: 2,
  };
}

// ============================================================================
// 1:1 Native-Resolution Offscreen Konva Exporter (PNG / JPG / CBZ / PDF)
// ============================================================================

function buildOffscreenExportStage(page, baseImageEl, format = 'png') {
  const width = Math.round(
    page?.dimensions?.width ||
      baseImageEl?.naturalWidth ||
      baseImageEl?.width ||
      850
  );
  const height = Math.round(
    page?.dimensions?.height ||
      baseImageEl?.naturalHeight ||
      baseImageEl?.height ||
      1200
  );

  const isJpeg =
    String(format).toLowerCase() === 'jpg' ||
    String(format).toLowerCase() === 'jpeg';

  const container = document.createElement('div');
  const stage = new Konva.Stage({
    container,
    width,
    height,
  });

  const bgLayer = new Konva.Layer({ listening: false });
  if (isJpeg) {
    bgLayer.add(
      new Konva.Rect({
        x: 0,
        y: 0,
        width,
        height,
        fill: '#ffffff',
        listening: false,
      })
    );
  }
  bgLayer.add(
    new Konva.Image({
      image: baseImageEl,
      x: 0,
      y: 0,
      width,
      height,
      listening: false,
    })
  );
  stage.add(bgLayer);

  const textLayer = new Konva.Layer({ listening: false });
  const blocks = page?.blocks || [];
  for (let i = 0; i < blocks.length; i += 1) {
    const block = blocks[i];
    const geom = getBlockGeometry(block, width, page?.targetLang);
    const style = getBlockTextStyle(block, geom, width);
    if (!style.text) continue;
    textLayer.add(
      new Konva.Text({
        x: geom.x,
        y: geom.y,
        width: geom.width,
        height: geom.height,
        rotation: geom.rotation,
        ...style,
        listening: false,
        shadowForStrokeEnabled: false,
        perfectDrawEnabled: false,
      })
    );
  }
  stage.add(textLayer);
  stage.draw();

  return { stage, width, height, isJpeg };
}

/**
 * Synchronously composite a page + translated text at exact 1:1 native resolution
 * using a detached offscreen Konva stage (zero live stage flicker).
 */
export function compositePageToDataUrl(
  page,
  baseImageEl,
  { format = 'png', quality = 0.98, pixelRatio = null } = {}
) {
  if (!page || !baseImageEl) return null;
  const { stage, width, height, isJpeg } = buildOffscreenExportStage(
    page,
    baseImageEl,
    format
  );
  const resolvedPixelRatio =
    pixelRatio ?? (Math.max(width, height) > 2800 ? 1.5 : 2);
  try {
    return stage.toDataURL({
      x: 0,
      y: 0,
      width,
      height,
      pixelRatio: resolvedPixelRatio,
      mimeType: isJpeg ? 'image/jpeg' : 'image/png',
      quality: isJpeg ? quality : 1,
    });
  } finally {
    stage.destroy();
  }
}

/**
 * Asynchronously composite ANY page (active or background) at high-DPI resolution
 * into a lossless PNG or JPEG `Blob` suitable for single download, `.cbz`, or `.pdf` export.
 */
export async function compositePageToBlob(
  page,
  {
    format = 'png',
    quality = 0.98,
    pixelRatio = null,
    apiBaseUrl = DEFAULT_API_BASE_URL,
    preloadedImage = null,
  } = {}
) {
  if (!page) return null;

  await ensurePageFontsLoaded(page, null, apiBaseUrl);

  let baseImg = preloadedImage;
  if (!baseImg) {
    const preferredSrc =
      pickInpaintedSrc(page, apiBaseUrl) ||
      resolveImageSrc(page.originalUrl, apiBaseUrl);
    if (!preferredSrc) return null;
    baseImg = await loadImageElement(preferredSrc);
    if (!baseImg) return null;
  }

  const { stage, width, height, isJpeg } = buildOffscreenExportStage(
    page,
    baseImg,
    format
  );
  const resolvedPixelRatio =
    pixelRatio ?? (Math.max(width, height) > 2800 ? 1.5 : 2);

  try {
    const canvas = stage.toCanvas({
      x: 0,
      y: 0,
      width,
      height,
      pixelRatio: resolvedPixelRatio,
    });
    const mimeType = isJpeg ? 'image/jpeg' : 'image/png';
    const blob = await new Promise((resolve) => {
      canvas.toBlob(
        (b) => resolve(b),
        mimeType,
        isJpeg ? quality : undefined
      );
    });
    // Explicitly release Chromium GPU canvas backing store immediately (critical for 1,500+ page exports)
    canvas.width = 0;
    canvas.height = 0;
    return blob;
  } finally {
    stage.destroy();
  }
}

/**
 * Composite multiple pages at exact 1:1 native resolution and submit to native C++17 backend
 * `/api/export/cbz` or `/api/export/pdf`.
 */
export async function exportPagesArchive(
  pages,
  archiveFormat = 'cbz',
  {
    apiBaseUrl = DEFAULT_API_BASE_URL,
    filename = 'vellum_export',
    imageFormat = 'png',
    quality = 0.98,
    activePageId = null,
    activeBaseImage = null,
    onProgress = null,
  } = {}
) {
  if (!Array.isArray(pages) || pages.length === 0) {
    throw new Error('No pages available to export.');
  }

  const fmt = String(archiveFormat).toLowerCase() === 'pdf' ? 'pdf' : 'cbz';
  const ext =
    String(imageFormat).toLowerCase() === 'jpg' ||
    String(imageFormat).toLowerCase() === 'jpeg'
      ? 'jpg'
      : 'png';
  const base = String(apiBaseUrl || '').replace(/\/+$/, '');
  const padWidth = Math.max(4, String(pages.length).length);

  // Use disk-backed O(1)-RAM streamed export session for large batches (>10 pages, up to 10,000+ pages)
  if (pages.length > 10) {
    const startResp = await fetch(`${base}/api/export/session/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        format: fmt,
        filename,
        image_format: ext,
        jpeg_quality: Math.round(quality * 100),
      }),
    });
    if (!startResp.ok) {
      const errJson = await startResp.json().catch(() => ({}));
      throw new Error(errJson?.detail || `Export session start HTTP ${startResp.status}`);
    }
    const { session_id: sessionId } = await startResp.json();
    let uploadedCount = 0;

    try {
      for (let idx = 0; idx < pages.length; idx += 1) {
        const p = pages[idx];
        if (onProgress) {
          onProgress({ current: idx + 1, total: pages.length, phase: 'rendering' });
        }
        const preloaded = p.id === activePageId ? activeBaseImage : null;
        let blob = await compositePageToBlob(p, {
          format: ext,
          quality,
          apiBaseUrl,
          preloadedImage: preloaded,
        });
        if (!blob && p.file) {
          blob = p.file;
        }
        if (!blob) continue;

        const safeStem = (p.name || `page_${idx + 1}`).replace(/\.[^/.]+$/, '');
        const pageFilename = `${String(idx + 1).padStart(padWidth, '0')}_${safeStem}.${ext}`;

        const pageForm = new FormData();
        pageForm.append('index', String(idx));
        pageForm.append('filename', pageFilename);
        pageForm.append('image', blob, pageFilename);

        let pageResp = null;
        for (let attempt = 0; attempt < 3; attempt += 1) {
          try {
            pageResp = await fetch(
              `${base}/api/export/session/${encodeURIComponent(sessionId)}/page`,
              { method: 'POST', body: pageForm }
            );
            if (pageResp.ok) break;
          } catch (netErr) {
            if (attempt === 2) throw netErr;
          }
          await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
        }
        if (!pageResp || !pageResp.ok) {
          const errJson = await pageResp?.json().catch(() => ({}));
          throw new Error(
            errJson?.detail ||
              `Export page ${idx + 1} upload failed (HTTP ${pageResp?.status || 'network'})`
          );
        }
        uploadedCount += 1;

        // Yield to UI event loop periodically so WebView stays responsive over 1,500+ pages
        if ((idx & 3) === 3) {
          await new Promise((r) => setTimeout(r, 0));
        }
      }

      if (uploadedCount === 0) {
        throw new Error('No renderable page images found for export.');
      }

      if (onProgress) {
        onProgress({ current: uploadedCount, total: pages.length, phase: 'finalizing' });
      }
      const finResp = await fetch(
        `${base}/api/export/session/${encodeURIComponent(sessionId)}/finalize`,
        { method: 'POST' }
      );
      if (!finResp.ok) {
        const errJson = await finResp.json().catch(() => ({}));
        throw new Error(errJson?.detail || `Export finalize HTTP ${finResp.status}`);
      }

      const outBlob = await finResp.blob();
      return {
        blob: outBlob,
        filename: `${filename}.${fmt}`,
        pageCount: uploadedCount,
      };
    } catch (err) {
      fetch(`${base}/api/export/session/${encodeURIComponent(sessionId)}/cancel`, {
        method: 'POST',
      }).catch(() => {});
      throw err;
    }
  }

  const formData = new FormData();
  formData.append('filename', filename);
  formData.append('image_format', ext);
  formData.append('jpeg_quality', String(Math.round(quality * 100)));

  let appendedCount = 0;
  for (let idx = 0; idx < pages.length; idx += 1) {
    const p = pages[idx];
    if (onProgress) {
      onProgress({ current: idx + 1, total: pages.length, phase: 'rendering' });
    }
    const preloaded = p.id === activePageId ? activeBaseImage : null;
    let blob = await compositePageToBlob(p, {
      format: ext,
      quality,
      apiBaseUrl,
      preloadedImage: preloaded,
    });

    if (!blob && p.file) {
      blob = p.file;
    }

    if (blob) {
      const safeStem = (p.name || `page_${idx + 1}`).replace(/\.[^/.]+$/, '');
      const pageFilename = `${String(idx + 1).padStart(padWidth, '0')}_${safeStem}.${ext}`;
      formData.append('images', blob, pageFilename);
      appendedCount += 1;
    }
  }

  if (appendedCount === 0) {
    throw new Error('No renderable page images found for export.');
  }

  const resp = await fetch(`${base}/api/export/${fmt}`, {
    method: 'POST',
    body: formData,
  });

  if (!resp.ok) {
    const errJson = await resp.json().catch(() => ({}));
    throw new Error(errJson?.detail || `Export HTTP ${resp.status}`);
  }

  const outBlob = await resp.blob();
  return {
    blob: outBlob,
    filename: `${filename}.${fmt}`,
    pageCount: appendedCount,
  };
}

// ============================================================================
// Memoized Konva Sub-Components (Zero Sibling Re-Renders)
// ============================================================================

/**
 * Memoized individual Konva Text Node.
 * Editing, dragging, or styling one bubble only re-renders THAT node,
 * never sibling text nodes on the page.
 */
const MemoTextNode = React.memo(
  function MemoTextNode({
    block,
    imgWidth,
    targetLang,
    fontRenderKey,
    isInteractive,
    isEditingInline,
    registerNodeRef,
    onHoverChange,
    onBlockMouseDown,
    onBlockClick,
    onBlockDoubleClick,
    onBlockDragStart,
    onBlockDragMove,
    onBlockDragEnd,
  }) {
    const geom = getBlockGeometry(block, imgWidth, targetLang);
    const style = getBlockTextStyle(block, geom, imgWidth);
    void fontRenderKey;

    const handleRef = useCallback(
      (node) => {
        registerNodeRef(block.id, node);
      },
      [block.id, registerNodeRef]
    );

    return (
      <KonvaText
        ref={handleRef}
        id={block.id}
        name="translated-text-box"
        x={geom.x}
        y={geom.y}
        width={geom.width}
        height={geom.height}
        rotation={geom.rotation}
        text={style.text}
        fontSize={style.fontSize}
        fontFamily={style.fontFamily}
        fontStyle={style.fontStyle}
        fill={style.fill}
        stroke={style.stroke}
        strokeWidth={style.strokeWidth}
        fillAfterStrokeEnabled={true}
        lineJoin="round"
        align={style.align}
        verticalAlign="middle"
        lineHeight={style.lineHeight}
        wrap="word"
        padding={2}
        opacity={isEditingInline ? 0.08 : 1}
        draggable={isInteractive && !isEditingInline}
        listening={isInteractive}
        shadowForStrokeEnabled={false}
        perfectDrawEnabled={false}
        onMouseEnter={() => onHoverChange(block.id, true)}
        onMouseLeave={() => onHoverChange(block.id, false)}
        onMouseDown={(e) => onBlockMouseDown(block.id, e)}
        onTouchStart={(e) => onBlockMouseDown(block.id, e)}
        onClick={(e) => onBlockClick(block.id, e)}
        onTap={(e) => onBlockClick(block.id, e)}
        onDblClick={(e) => onBlockDoubleClick(block.id, e)}
        onDblTap={(e) => onBlockDoubleClick(block.id, e)}
        onDragStart={(e) => onBlockDragStart(block.id, e)}
        onDragMove={(e) => onBlockDragMove(block.id, e)}
        onDragEnd={(e) => onBlockDragEnd(block.id, e)}
      />
    );
  },
  (prev, next) =>
    prev.block === next.block &&
    prev.imgWidth === next.imgWidth &&
    prev.fontRenderKey === next.fontRenderKey &&
    prev.isInteractive === next.isInteractive &&
    prev.isEditingInline === next.isEditingInline &&
    prev.registerNodeRef === next.registerNodeRef &&
    prev.onHoverChange === next.onHoverChange &&
    prev.onBlockMouseDown === next.onBlockMouseDown &&
    prev.onBlockClick === next.onBlockClick &&
    prev.onBlockDoubleClick === next.onBlockDoubleClick &&
    prev.onBlockDragStart === next.onBlockDragStart &&
    prev.onBlockDragMove === next.onBlockDragMove &&
    prev.onBlockDragEnd === next.onBlockDragEnd
);

/**
 * Memoized Bounding Box + Reading Order `#1/#2` Badge Guide (rendered on non-listening guide layer).
 */
const MemoBoxGuideNode = React.memo(
  function MemoBoxGuideNode({
    block,
    orderNumber,
    isSelected,
    isHovered,
    registerGuideRef,
  }) {
    const geom = getBlockGeometry(block);
    const labelText = `#${orderNumber}`;
    const badgeWidth = Math.max(24, labelText.length * 8 + 10);
    const badgeHeight = 18;
    const badgeY = geom.y >= badgeHeight + 2 ? -badgeHeight - 2 : 2;

    const strokeColor = isSelected
      ? '#E59A3C'
      : isHovered
      ? '#38BDF8'
      : 'rgba(229, 154, 60, 0.72)';
    const fillColor = isSelected
      ? 'rgba(229, 154, 60, 0.10)'
      : isHovered
      ? 'rgba(56, 189, 248, 0.08)'
      : 'rgba(229, 154, 60, 0.03)';
    const badgeBg = isSelected
      ? '#E59A3C'
      : isHovered
      ? '#0284C7'
      : 'rgba(24, 26, 29, 0.88)';
    const badgeFg = isSelected ? '#111214' : '#F8FAFC';

    const handleRef = useCallback(
      (node) => {
        registerGuideRef(block.id, node);
      },
      [block.id, registerGuideRef]
    );

    return (
      <KonvaGroup
        ref={handleRef}
        x={geom.x}
        y={geom.y}
        rotation={geom.rotation}
        listening={false}
      >
        <KonvaRect
          name="guide-box-rect"
          x={0}
          y={0}
          width={geom.width}
          height={geom.height}
          stroke={strokeColor}
          strokeWidth={isSelected ? 2 : 1.5}
          dash={isSelected ? undefined : [6, 4]}
          fill={fillColor}
          cornerRadius={3}
          listening={false}
          perfectDrawEnabled={false}
        />
        <KonvaRect
          x={0}
          y={badgeY}
          width={badgeWidth}
          height={badgeHeight}
          fill={badgeBg}
          stroke={isSelected ? '#C97A16' : strokeColor}
          strokeWidth={1}
          cornerRadius={3}
          listening={false}
          perfectDrawEnabled={false}
        />
        <KonvaText
          x={0}
          y={badgeY}
          width={badgeWidth}
          height={badgeHeight}
          text={labelText}
          fontSize={11}
          fontFamily='"JetBrains Mono", Consolas, monospace'
          fontStyle="bold"
          fill={badgeFg}
          align="center"
          verticalAlign="middle"
          listening={false}
          perfectDrawEnabled={false}
        />
      </KonvaGroup>
    );
  },
  (prev, next) =>
    prev.block === next.block &&
    prev.orderNumber === next.orderNumber &&
    prev.isSelected === next.isSelected &&
    prev.isHovered === next.isHovered &&
    prev.registerGuideRef === next.registerGuideRef
);

/**
 * Memoized fallback Inpaint Mask polygon/rect when binary mask PNG is unavailable.
 */
const MemoMaskFallbackShape = React.memo(function MemoMaskFallbackShape({
  block,
}) {
  if (Array.isArray(block.polygon) && block.polygon.length >= 3) {
    const flatPoints = block.polygon.flatMap((pt) => [
      Number(pt[0]) || 0,
      Number(pt[1]) || 0,
    ]);
    return (
      <KonvaLine
        points={flatPoints}
        closed={true}
        fill="rgba(239, 68, 68, 0.45)"
        stroke="rgba(239, 68, 68, 0.78)"
        strokeWidth={1.5}
        listening={false}
        perfectDrawEnabled={false}
      />
    );
  }
  const geom = getBlockGeometry(block);
  return (
    <KonvaRect
      x={geom.x}
      y={geom.y}
      width={geom.width}
      height={geom.height}
      rotation={geom.rotation}
      fill="rgba(239, 68, 68, 0.45)"
      stroke="rgba(239, 68, 68, 0.78)"
      strokeWidth={1.5}
      cornerRadius={2}
      listening={false}
      perfectDrawEnabled={false}
    />
  );
});

/**
 * Memoized subtle selection/hover outline displayed when Bounding Boxes guide layer is hidden.
 */
const MemoSelectionOutline = React.memo(function MemoSelectionOutline({
  block,
  isSelected,
}) {
  const geom = getBlockGeometry(block);
  return (
    <KonvaRect
      x={geom.x}
      y={geom.y}
      width={geom.width}
      height={geom.height}
      rotation={geom.rotation}
      stroke={isSelected ? '#E59A3C' : '#38BDF8'}
      strokeWidth={isSelected ? 1.75 : 1.25}
      dash={isSelected ? undefined : [5, 4]}
      fill={
        isSelected ? 'rgba(229, 154, 60, 0.08)' : 'rgba(56, 189, 248, 0.05)'
      }
      cornerRadius={3}
      listening={false}
      perfectDrawEnabled={false}
    />
  );
});

// ============================================================================
// Main High-Performance Konva Studio Canvas (v2)
// ============================================================================

const LAYER_HUD_TOGGLES = [
  { key: 'original', label: 'Raw', title: 'Toggle Original Image Layer' },
  { key: 'inpainted', label: 'Clean', title: 'Toggle Inpainted Image Layer' },
  { key: 'text', label: 'Text', title: 'Toggle Translated Text Layer' },
  {
    key: 'boxes',
    label: 'Boxes',
    title: 'Toggle Bounding Boxes & Reading Order (#1/#2)',
  },
  { key: 'mask', label: 'Mask', title: 'Toggle Inpaint Mask Overlay' },
];

const MangaCanvas = forwardRef(function MangaCanvas(
  {
    page,
    selectedBlockId: propSelectedBlockId,
    onSelectBlock,
    onUpdateBlock,
    onImportDrop,
    onLoadSample,
    isProcessing = false,
  },
  ref
) {
  // ── Zustand Store Subscriptions (fine-grained shallow selectors) ───────────
  const {
    storeSelectedBlockId,
    selectedLayerIds,
    activeTool,
    viewLayers,
    apiBaseUrl,
    resolvedTheme,
    storeZoom,
    fontLoadVersion,
    typesettingFontFamily,
    typesettingAllCaps,
    typesettingDefaultFontSize,
    typesettingAutoFit,
    canUndo,
    canRedo,
    undoCount,
    redoCount,
    setSelectedBlock,
    setSelectedLayers,
    toggleSelectLayer,
    setActiveTool,
    updateBlock,
    updateBlocksBatch,
    addBlock,
    undo,
    redo,
    toggleViewLayer,
    setCanvasZoom,
    setCursorPosition,
    clearCursorPosition,
  } = useEditorStore(
    useShallow((s) => ({
      storeSelectedBlockId: s.selectedBlockId,
      selectedLayerIds: s.selectedLayerIds,
      activeTool: s.activeTool,
      viewLayers: s.viewLayers,
      apiBaseUrl: s.pipelineConfig.apiBaseUrl,
      resolvedTheme: s.resolvedTheme,
      storeZoom: s.canvasView.zoom,
      fontLoadVersion: s.fontLoadVersion,
      typesettingFontFamily: s.typesetting?.fontFamily || '',
      typesettingAllCaps: Boolean(s.typesetting?.allCaps),
      typesettingDefaultFontSize: Number(s.typesetting?.defaultFontSize) || 18,
      typesettingAutoFit: s.typesetting?.autoFitFontSize !== false,
      canUndo: s.history.past.length > 0,
      canRedo: s.history.future.length > 0,
      undoCount: s.history.past.length,
      redoCount: s.history.future.length,
      setSelectedBlock: s.setSelectedBlock,
      setSelectedLayers: s.setSelectedLayers,
      toggleSelectLayer: s.toggleSelectLayer,
      setActiveTool: s.setActiveTool,
      updateBlock: s.updateBlock,
      updateBlocksBatch: s.updateBlocksBatch,
      addBlock: s.addBlock,
      undo: s.undo,
      redo: s.redo,
      toggleViewLayer: s.toggleViewLayer,
      setCanvasZoom: s.setCanvasZoom,
      setCursorPosition: s.setCursorPosition,
      clearCursorPosition: s.clearCursorPosition,
    }))
  );

  // Merge store selection with prop fallback
  const effectiveSelectedIds = useMemo(() => {
    if (selectedLayerIds && selectedLayerIds.length > 0) {
      return selectedLayerIds;
    }
    const primary = propSelectedBlockId ?? storeSelectedBlockId;
    return primary ? [primary] : [];
  }, [selectedLayerIds, propSelectedBlockId, storeSelectedBlockId]);

  const selectedIdSet = useMemo(
    () => new Set(effectiveSelectedIds),
    [effectiveSelectedIds]
  );

  // ── DOM & Konva Node Refs ──────────────────────────────────────────────────
  const containerRef = useRef(null);
  const stageRef = useRef(null);
  const baseLayerRef = useRef(null);
  const maskLayerRef = useRef(null);
  const textLayerRef = useRef(null);
  const boxesLayerRef = useRef(null);
  const uiLayerRef = useRef(null);
  const transformerRef = useRef(null);

  // Clip groups for Split-Curtain Before/After Compare (mutated imperatively at 60fps)
  const inpaintClipGroupRef = useRef(null);
  const maskClipGroupRef = useRef(null);
  const textClipGroupRef = useRef(null);
  const boxesClipGroupRef = useRef(null);

  // Imperative overlay refs (mutated in rAF with zero React re-renders)
  const marqueeRectRef = useRef(null);
  const drawPreviewRectRef = useRef(null);
  const splitLineOverlayRef = useRef(null);
  const splitHandleRef = useRef(null);
  const splitLeftBadgeRef = useRef(null);
  const splitRightBadgeRef = useRef(null);
  const zoomLabelRef = useRef(null);

  // Maps of Konva node refs for text nodes & bounding box guide groups
  const textNodeRefs = useRef({});
  const guideNodeRefs = useRef({});

  const fontRenderKey = `${fontLoadVersion}|${typesettingFontFamily}|${typesettingAllCaps}|${typesettingDefaultFontSize}|${typesettingAutoFit}`;

  // Repaint Konva text nodes instantaneously whenever a dynamic FontFace finishes loading
  // or the global Font Fallback Chain / ALL-CAPS typesetting config changes.
  useEffect(() => {
    const nodes = Object.values(textNodeRefs.current || {});
    nodes.forEach((node) => {
      if (node && typeof node._setTextData === 'function') {
        node._setTextData();
      }
    });
    if (textLayerRef.current) {
      textLayerRef.current.batchDraw();
    }
    if (transformerRef.current) {
      transformerRef.current.forceUpdate();
      uiLayerRef.current?.batchDraw();
    }
  }, [fontRenderKey]);

  // Imperative 60fps Transform & Gesture State Refs
  const transformRef = useRef({ scale: 1, x: 0, y: 0 });
  const targetTransformRef = useRef({ scale: 1, x: 0, y: 0 });
  const transformRafRef = useRef(null);
  const lastZoomSyncTimeRef = useRef(0);
  const lastSyncedZoomRef = useRef(1);

  const splitRatioRef = useRef(0.5);
  const splitDragRafRef = useRef(null);
  const isDraggingSplitRef = useRef(false);

  const panStateRef = useRef(null);
  const panRafRef = useRef(null);
  const marqueeStateRef = useRef(null);
  const drawBoxStateRef = useRef(null);
  const cursorRafRef = useRef(null);
  const pendingCursorRef = useRef(null);

  const dragSessionRef = useRef(null);
  const didDragBlockRef = useRef(false);
  const isSpacePanningRef = useRef(false);

  // ── Minimal React UI State ─────────────────────────────────────────────────
  const [viewport, setViewport] = useState({ width: 900, height: 700 });
  const [hoveredBlockId, setHoveredBlockId] = useState(null);
  const [isHoldComparingLocal, setIsHoldComparingLocal] = useState(false);
  const [isSpacePanning, setIsSpacePanning] = useState(false);
  const [isDraggingFileOver, setIsDraggingFileOver] = useState(false);
  const [splitPercentAria, setSplitPercentAria] = useState(50);
  const [inlineEditState, setInlineEditState] = useState(null); // { id, text, style }

  // ── Cached Image & Mask Loading (Zero Base64) ──────────────────────────────
  const inpaintedSrc = useMemo(
    () => pickInpaintedSrc(page, apiBaseUrl),
    [page, apiBaseUrl]
  );
  const originalSrc = useMemo(
    () => resolveImageSrc(page?.originalUrl, apiBaseUrl),
    [page?.originalUrl, apiBaseUrl]
  );
  const maskSrc = useMemo(
    () => resolveImageSrc(page?.maskUrl, apiBaseUrl),
    [page?.maskUrl, apiBaseUrl]
  );

  const inpaintedImageEl = useLoadedImage(inpaintedSrc);
  const rawOriginalImageEl = useLoadedImage(originalSrc);
  const maskImgEl = useLoadedImage(maskSrc);
  const maskOverlayCanvas = useMaskOverlayCanvas(maskSrc, maskImgEl);

  const exportBaseImage = inpaintedImageEl || rawOriginalImageEl;
  const hasRenderablePage = Boolean(
    page && (rawOriginalImageEl || inpaintedImageEl)
  );

  const imageWidth =
    page?.dimensions?.width ||
    exportBaseImage?.naturalWidth ||
    exportBaseImage?.width ||
    850;
  const imageHeight =
    page?.dimensions?.height ||
    exportBaseImage?.naturalHeight ||
    exportBaseImage?.height ||
    1200;

  // ── Derived View & Layer Visibility Flags ──────────────────────────────────
  const isDarkTheme = resolvedTheme !== 'light';
  const isHoldComparing =
    isHoldComparingLocal || Boolean(viewLayers.compareOriginal);
  const isSplitCompare = Boolean(viewLayers.splitCompare) && !isHoldComparing;

  const showOriginalLayer =
    Boolean(rawOriginalImageEl) &&
    (isHoldComparing || isSplitCompare || Boolean(viewLayers.original));
  const showInpaintedLayer =
    !isHoldComparing &&
    Boolean(inpaintedImageEl) &&
    Boolean(viewLayers.inpainted);
  const showMaskLayer = !isHoldComparing && Boolean(viewLayers.mask);
  const showTextLayer = !isHoldComparing && Boolean(viewLayers.text);
  const showBoxesLayer = !isHoldComparing && Boolean(viewLayers.boxes);

  const isPanMode = activeTool === 'pan' || isSpacePanning;
  const isDrawTextMode = activeTool === 'text' && !isSpacePanning;
  const isTextInteractive =
    !isHoldComparing && !isPanMode && !isDrawTextMode && showTextLayer;

  // Keep live refs for stable callbacks (prevents MemoTextNode re-renders!)
  const blocks = useMemo(() => page?.blocks || [], [page?.blocks]);
  const blocksRef = useRef(blocks);
  blocksRef.current = blocks;

  const selectedIdsRef = useRef(effectiveSelectedIds);
  selectedIdsRef.current = effectiveSelectedIds;

  const activeToolRef = useRef(activeTool);
  activeToolRef.current = activeTool;

  const isSplitCompareRef = useRef(isSplitCompare);
  isSplitCompareRef.current = isSplitCompare;

  const isHoldComparingRef = useRef(isHoldComparing);
  isHoldComparingRef.current = isHoldComparing;

  const imageDimsRef = useRef({ width: imageWidth, height: imageHeight });
  imageDimsRef.current = { width: imageWidth, height: imageHeight };

  const viewportRef = useRef(viewport);
  viewportRef.current = viewport;

  const pageRef = useRef(page);
  pageRef.current = page;

  const inlineEditStateRef = useRef(inlineEditState);
  inlineEditStateRef.current = inlineEditState;

  // ── Imperative Overlay & Split-Curtain Synchronizer (60fps, Zero Re-Renders) ──
  const applySplitClip = useCallback((ratio) => {
    const groups = [
      inpaintClipGroupRef.current,
      maskClipGroupRef.current,
      textClipGroupRef.current,
      boxesClipGroupRef.current,
    ];
    const { width: imgW, height: imgH } = imageDimsRef.current;

    if (!isSplitCompareRef.current) {
      for (let i = 0; i < groups.length; i += 1) {
        if (groups[i]) groups[i].clip(null);
      }
    } else {
      const splitX = ratio * imgW;
      const clipRect = {
        x: splitX,
        y: 0,
        width: Math.max(0, imgW - splitX),
        height: imgH,
      };
      for (let i = 0; i < groups.length; i += 1) {
        if (groups[i]) groups[i].clip(clipRect);
      }
    }
  }, []);

  const syncOverlayPositions = useCallback(() => {
    const { scale, x, y } = transformRef.current;
    const { width: imgW, height: imgH } = imageDimsRef.current;
    const vp = viewportRef.current;

    // 1. Update zoom % readout text directly without React re-render
    if (zoomLabelRef.current) {
      zoomLabelRef.current.textContent = `${Math.round(scale * 100)}%`;
    }

    // 2. Position Split-Curtain vertical divider & handle glued to comic page coordinates
    if (isSplitCompareRef.current && splitLineOverlayRef.current) {
      const ratio = splitRatioRef.current;
      const screenX = x + ratio * imgW * scale;
      const pageTop = Math.max(0, y);
      const pageBottom = Math.min(vp.height, y + imgH * scale);
      const visibleHeight = Math.max(48, pageBottom - pageTop);

      const el = splitLineOverlayRef.current;
      el.style.transform = `translate3d(${screenX}px, ${pageTop}px, 0)`;
      el.style.height = `${visibleHeight}px`;

      if (splitHandleRef.current) {
        const handleY = Math.min(
          Math.max(visibleHeight / 2, 44),
          Math.max(44, visibleHeight - 68)
        );
        splitHandleRef.current.style.transform = `translate3d(-50%, ${handleY - 18}px, 0)`;
      }

      if (splitLeftBadgeRef.current) {
        splitLeftBadgeRef.current.style.opacity = ratio < 0.14 ? '0' : '1';
      }
      if (splitRightBadgeRef.current) {
        splitRightBadgeRef.current.style.opacity = ratio > 0.86 ? '0' : '1';
      }
    }
  }, []);

  const applyStageTransformImmediate = useCallback(
    (nextTransform, { syncStore = false } = {}) => {
      const clampedScale = clampZoom(nextTransform.scale);
      const next = {
        scale: clampedScale,
        x: nextTransform.x,
        y: nextTransform.y,
      };
      transformRef.current = next;
      targetTransformRef.current = next;

      const stage = stageRef.current;
      if (stage) {
        stage.scale({ x: next.scale, y: next.scale });
        stage.position({ x: next.x, y: next.y });
        stage.batchDraw();
      }

      syncOverlayPositions();

      if (syncStore) {
        const rounded = Number(next.scale.toFixed(4));
        lastSyncedZoomRef.current = rounded;
        lastZoomSyncTimeRef.current = performance.now();
        setCanvasZoom(rounded);
      }
    },
    [setCanvasZoom, syncOverlayPositions]
  );

  // Smooth rAF interpolation loop for butter-smooth 60fps mouse-wheel & menu zoom
  const scheduleTransformLoop = useCallback(() => {
    if (transformRafRef.current !== null) return;

    const step = (now) => {
      transformRafRef.current = null;
      const cur = transformRef.current;
      const tgt = targetTransformRef.current;

      const dScale = tgt.scale - cur.scale;
      const dX = tgt.x - cur.x;
      const dY = tgt.y - cur.y;

      const isSettled =
        Math.abs(dScale) < 0.0006 &&
        Math.abs(dX) < 0.35 &&
        Math.abs(dY) < 0.35;

      const lerpFactor = 0.36;
      const next = isSettled
        ? { scale: tgt.scale, x: tgt.x, y: tgt.y }
        : {
            scale: cur.scale + dScale * lerpFactor,
            x: cur.x + dX * lerpFactor,
            y: cur.y + dY * lerpFactor,
          };

      transformRef.current = next;

      const stage = stageRef.current;
      if (stage) {
        stage.scale({ x: next.scale, y: next.scale });
        stage.position({ x: next.x, y: next.y });
        stage.batchDraw();
      }

      syncOverlayPositions();

      // Throttle Zustand zoom updates so StatusBar stays live without React frame drops
      if (isSettled || now - lastZoomSyncTimeRef.current >= 80) {
        const rounded = Number(next.scale.toFixed(4));
        if (Math.abs(rounded - lastSyncedZoomRef.current) > 0.0005) {
          lastSyncedZoomRef.current = rounded;
          lastZoomSyncTimeRef.current = now;
          setCanvasZoom(rounded);
        }
      }

      if (!isSettled) {
        transformRafRef.current = window.requestAnimationFrame(step);
      }
    };

    transformRafRef.current = window.requestAnimationFrame(step);
  }, [setCanvasZoom, syncOverlayPositions]);

  // Helper to convert screen/stage pointer to 1:1 native comic page coordinates
  const stagePointToPage = useCallback((stagePoint) => {
    const { scale, x, y } = transformRef.current;
    return {
      x: (stagePoint.x - x) / scale,
      y: (stagePoint.y - y) / scale,
    };
  }, []);

  // ── Viewport Resize Observer ───────────────────────────────────────────────
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const updateSize = () => {
      const rect = el.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        setViewport((prev) => {
          const w = Math.floor(rect.width);
          const h = Math.floor(rect.height);
          return prev.width === w && prev.height === h ? prev : { width: w, height: h };
        });
      }
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Re-sync Split-Curtain & overlays whenever viewport resizes or splitCompare toggles
  useEffect(() => {
    applySplitClip(splitRatioRef.current);
    stageRef.current?.batchDraw();
    syncOverlayPositions();
  }, [
    viewport.width,
    viewport.height,
    isSplitCompare,
    showOriginalLayer,
    showInpaintedLayer,
    showMaskLayer,
    showTextLayer,
    showBoxesLayer,
    imageWidth,
    imageHeight,
    inpaintedImageEl,
    maskOverlayCanvas,
    applySplitClip,
    syncOverlayPositions,
  ]);

  // ── Fit Page to Viewport ───────────────────────────────────────────────────
  const fitPageToViewport = useCallback(() => {
    const vp = viewportRef.current;
    const { width: imgW, height: imgH } = imageDimsRef.current;
    if (vp.width <= 0 || vp.height <= 0 || imgW <= 0 || imgH <= 0) return;

    if (transformRafRef.current !== null) {
      window.cancelAnimationFrame(transformRafRef.current);
      transformRafRef.current = null;
    }

    const padX = 56;
    const padY = 64;
    const usableWidth = Math.max(180, vp.width - padX * 2);
    const usableHeight = Math.max(180, vp.height - padY * 2);

    const computedFitScale = clampZoom(
      Math.min(1.15, Math.min(usableWidth / imgW, usableHeight / imgH))
    );
    const originX = Math.round((vp.width - imgW * computedFitScale) / 2);
    const originY = Math.round((vp.height - imgH * computedFitScale) / 2);

    applyStageTransformImmediate(
      { scale: computedFitScale, x: originX, y: originY },
      { syncStore: true }
    );
  }, [applyStageTransformImmediate]);

  // Smoothly zoom toward target scale centered on viewport center
  const animateZoomTo = useCallback(
    (targetScale) => {
      const vp = viewportRef.current;
      const curTarget = targetTransformRef.current;
      const nextScale = clampZoom(targetScale);
      const centerX = vp.width / 2;
      const centerY = vp.height / 2;
      const anchorPageX = (centerX - curTarget.x) / curTarget.scale;
      const anchorPageY = (centerY - curTarget.y) / curTarget.scale;

      targetTransformRef.current = {
        scale: nextScale,
        x: centerX - anchorPageX * nextScale,
        y: centerY - anchorPageY * nextScale,
      };
      scheduleTransformLoop();
    },
    [scheduleTransformLoop]
  );

  // Auto-fit once per loaded page (never resets zoom when inpainted image arrives for same page!)
  const lastFittedPageIdRef = useRef(null);
  useEffect(() => {
    if (!page?.id) {
      lastFittedPageIdRef.current = null;
      return;
    }
    if (hasRenderablePage && stageRef.current && lastFittedPageIdRef.current !== page.id) {
      lastFittedPageIdRef.current = page.id;
      fitPageToViewport();
    }
  }, [page?.id, hasRenderablePage, fitPageToViewport]);

  // Respond to external zoom commands (MenuBar View -> Zoom In / Out / 100%, Ctrl++ / Ctrl+- / Ctrl+1)
  useEffect(() => {
    if (!hasRenderablePage || !stageRef.current) return;
    if (Math.abs(storeZoom - lastSyncedZoomRef.current) > 0.001) {
      lastSyncedZoomRef.current = storeZoom;
      animateZoomTo(storeZoom);
    }
  }, [storeZoom, hasRenderablePage, animateZoomTo]);

  // Cleanup all rAF loops on unmount
  useEffect(
    () => () => {
      if (transformRafRef.current !== null) {
        window.cancelAnimationFrame(transformRafRef.current);
      }
      if (panRafRef.current !== null) {
        window.cancelAnimationFrame(panRafRef.current);
      }
      if (splitDragRafRef.current !== null) {
        window.cancelAnimationFrame(splitDragRafRef.current);
      }
      if (cursorRafRef.current !== null) {
        window.cancelAnimationFrame(cursorRafRef.current);
      }
    },
    []
  );

  // ── Konva Transformer Multi-Select Attachment ──────────────────────────────
  useEffect(() => {
    const tr = transformerRef.current;
    if (!tr) return;

    if (!isHoldComparing && showTextLayer && !inlineEditState && effectiveSelectedIds.length > 0) {
      const nodes = effectiveSelectedIds
        .map((id) => textNodeRefs.current[id])
        .filter(Boolean);
      tr.nodes(nodes);
    } else {
      tr.nodes([]);
    }
    tr.getLayer()?.batchDraw();
  }, [effectiveSelectedIds, isHoldComparing, showTextLayer, inlineEditState, blocks]);

  // ── Keyboard Shortcuts (Space Pan, V/H Tools, Esc, Arrow Nudge, \ Compare) ─
  useEffect(() => {
    const isEditable = (el) =>
      el &&
      (el.tagName === 'INPUT' ||
        el.tagName === 'TEXTAREA' ||
        el.tagName === 'SELECT' ||
        el.isContentEditable);

    const handleKeyDown = (e) => {
      if (isEditable(e.target)) return;

      if (e.code === 'Space' && !e.repeat && !e.ctrlKey && !e.metaKey && !e.altKey) {
        e.preventDefault();
        isSpacePanningRef.current = true;
        setIsSpacePanning(true);
        return;
      }

      if (e.key === '\\' && !e.repeat && !e.ctrlKey && !e.metaKey && !e.altKey) {
        if (pageRef.current) {
          setIsHoldComparingLocal(true);
        }
        return;
      }

      if (e.ctrlKey || e.metaKey || e.altKey) return;

      const lower = e.key.toLowerCase();
      if (lower === 'v') {
        setActiveTool('select');
        return;
      }
      if (lower === 'h') {
        setActiveTool('pan');
        return;
      }
      if (e.key === 'Escape') {
        if (inlineEditState) {
          setInlineEditState(null);
          return;
        }
        if (activeToolRef.current !== 'select') {
          setActiveTool('select');
          return;
        }
        if (selectedIdsRef.current.length > 0) {
          setSelectedLayers([]);
          if (onSelectBlock) onSelectBlock(null);
        }
        return;
      }

      // Arrow-key nudging for single or multi-selected text blocks
      if (
        ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key) &&
        selectedIdsRef.current.length > 0 &&
        pageRef.current
      ) {
        e.preventDefault();
        const step = e.shiftKey ? 10 : 1;
        const dx =
          e.key === 'ArrowRight' ? step : e.key === 'ArrowLeft' ? -step : 0;
        const dy =
          e.key === 'ArrowDown' ? step : e.key === 'ArrowUp' ? -step : 0;

        const patches = {};
        const selectedSet = new Set(selectedIdsRef.current);
        blocksRef.current.forEach((b) => {
          if (!selectedSet.has(b.id)) return;
          const g = getBlockGeometry(b);
          const nx = Math.round(g.x + dx);
          const ny = Math.round(g.y + dy);
          patches[b.id] = {
            x: nx,
            y: ny,
            bbox: [nx, ny, nx + g.width, ny + g.height],
          };
        });
        if (Object.keys(patches).length > 0) {
          updateBlocksBatch(patches, {
            recordHistory: true,
            coalesceKey: `nudge:${selectedIdsRef.current.join(',')}`,
          });
        }
      }
    };

    const handleKeyUp = (e) => {
      if (e.code === 'Space') {
        isSpacePanningRef.current = false;
        setIsSpacePanning(false);
      }
      if (e.key === '\\') {
        setIsHoldComparingLocal(false);
      }
    };

    const handleWindowBlur = () => {
      isSpacePanningRef.current = false;
      setIsSpacePanning(false);
      setIsHoldComparingLocal(false);
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    window.addEventListener('blur', handleWindowBlur);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      window.removeEventListener('blur', handleWindowBlur);
    };
  }, [inlineEditState, onSelectBlock, setActiveTool, setSelectedLayers, updateBlocksBatch]);

  // ── Imperative Ref Handle (Single PNG/JPG, Multi-Page CBZ/PDF, Fit/Zoom) ──
  useImperativeHandle(
    ref,
    () => ({
      exportImage: (format = 'png', quality = 0.98) => {
        if (!page || !exportBaseImage) return null;
        return compositePageToDataUrl(page, exportBaseImage, {
          format,
          quality,
        });
      },
      exportPageBlob: async (targetPage, format = 'png', quality = 0.98) => {
        const resolvedPage = targetPage || page;
        if (!resolvedPage) return null;
        const preloaded =
          resolvedPage.id === page?.id ? exportBaseImage : null;
        return compositePageToBlob(resolvedPage, {
          format,
          quality,
          apiBaseUrl,
          preloadedImage: preloaded,
        });
      },
      exportArchive: async (targetPages, archiveFormat = 'cbz', options = {}) =>
        exportPagesArchive(targetPages, archiveFormat, {
          apiBaseUrl,
          activePageId: page?.id ?? null,
          activeBaseImage: exportBaseImage,
          ...options,
        }),
      fitToScreen: fitPageToViewport,
      zoomTo: animateZoomTo,
      getTransform: () => ({ ...transformRef.current }),
    }),
    [page, exportBaseImage, apiBaseUrl, fitPageToViewport, animateZoomTo]
  );

  // ── Stable Node Registration Callbacks (Never Break React.memo) ───────────
  const registerTextNodeRef = useCallback((id, node) => {
    if (node) {
      textNodeRefs.current[id] = node;
    } else {
      delete textNodeRefs.current[id];
    }
  }, []);

  const registerGuideRef = useCallback((id, node) => {
    if (node) {
      guideNodeRefs.current[id] = node;
    } else {
      delete guideNodeRefs.current[id];
    }
  }, []);

  // Helper to sync a bounding box guide imperatively during drag/transform (60fps)
  const syncGuideNodeToTextNode = useCallback((id, textNode) => {
    const guide = guideNodeRefs.current[id];
    if (!guide || !textNode) return;
    guide.position({ x: textNode.x(), y: textNode.y() });
    guide.rotation(textNode.rotation());
    const boxRect = guide.findOne('.guide-box-rect');
    if (boxRect) {
      boxRect.width(Math.max(32, textNode.width() * Math.abs(textNode.scaleX())));
      boxRect.height(Math.max(20, textNode.height() * Math.abs(textNode.scaleY())));
    }
  }, []);

  // ── Stable Text Block Event Handlers ───────────────────────────────────────
  const handleBlockHoverChange = useCallback((id, entered) => {
    setHoveredBlockId((prev) => {
      if (entered) return id;
      return prev === id ? null : prev;
    });
  }, []);

  const handleBlockMouseDown = useCallback(
    (id, e) => {
      // Allow middle-button panning even over a text bubble
      if (e.evt?.button === 1) return;
      e.cancelBubble = true;
      didDragBlockRef.current = false;

      const isMultiMod = Boolean(e.evt?.shiftKey || e.evt?.ctrlKey || e.evt?.metaKey);
      if (isMultiMod) {
        toggleSelectLayer(id);
        return;
      }

      if (!selectedIdsRef.current.includes(id)) {
        setSelectedBlock(id);
        if (onSelectBlock) onSelectBlock(id);
      }
    },
    [onSelectBlock, setSelectedBlock, toggleSelectLayer]
  );

  const handleBlockClick = useCallback(
    (id, e) => {
      if (e.evt?.button === 1) return;
      e.cancelBubble = true;
      const isMultiMod = Boolean(e.evt?.shiftKey || e.evt?.ctrlKey || e.evt?.metaKey);
      if (isMultiMod || didDragBlockRef.current) return;

      // Narrow multi-selection to clicked bubble if user clicked without dragging
      if (selectedIdsRef.current.length > 1 && selectedIdsRef.current.includes(id)) {
        setSelectedBlock(id);
        if (onSelectBlock) onSelectBlock(id);
      }
    },
    [onSelectBlock, setSelectedBlock]
  );

  const handleBlockDoubleClick = useCallback((id, e) => {
    e.cancelBubble = true;
    const block = blocksRef.current.find((b) => b.id === id);
    if (!block) return;
    setInlineEditState({
      id: block.id,
      text: block.translated_text || '',
    });
  }, []);

  const handleBlockDragStart = useCallback((id, e) => {
    e.cancelBubble = true;
    didDragBlockRef.current = true;

    const currentSelection = selectedIdsRef.current.includes(id)
      ? selectedIdsRef.current
      : [id];

    // If a session is already initialized (e.g. Transformer dragging multiple nodes), keep it
    if (dragSessionRef.current) return;

    const starts = {};
    for (let i = 0; i < currentSelection.length; i += 1) {
      const sid = currentSelection[i];
      const node = textNodeRefs.current[sid];
      if (node) {
        starts[sid] = {
          x: node.x(),
          y: node.y(),
          width: node.width(),
          height: node.height(),
        };
      }
    }

    dragSessionRef.current = {
      leaderId: id,
      ids: currentSelection,
      starts,
    };
  }, []);

  const handleBlockDragMove = useCallback(
    (id, e) => {
      e.cancelBubble = true;
      const session = dragSessionRef.current;
      const node = e.target;

      if (session && session.leaderId === id && session.ids.length > 1) {
        const leaderStart = session.starts[id];
        if (leaderStart) {
          const dx = node.x() - leaderStart.x;
          const dy = node.y() - leaderStart.y;
          for (let i = 0; i < session.ids.length; i += 1) {
            const peerId = session.ids[i];
            if (peerId === id) continue;
            const peerNode = textNodeRefs.current[peerId];
            const peerStart = session.starts[peerId];
            if (peerNode && peerStart && !peerNode.isDragging()) {
              peerNode.position({
                x: peerStart.x + dx,
                y: peerStart.y + dy,
              });
            }
            if (peerNode) {
              syncGuideNodeToTextNode(peerId, peerNode);
            }
          }
          textLayerRef.current?.batchDraw();
          uiLayerRef.current?.batchDraw();
        }
      }

      syncGuideNodeToTextNode(id, node);
      boxesLayerRef.current?.batchDraw();
    },
    [syncGuideNodeToTextNode]
  );

  const handleBlockDragEnd = useCallback(
    (id, e) => {
      e.cancelBubble = true;
      const session = dragSessionRef.current;
      const targetIds = session?.ids?.length ? session.ids : [id];

      // Wait until all nodes in a Transformer multi-drag have completed dragEnd
      const anyStillDragging = targetIds.some((sid) => {
        const n = textNodeRefs.current[sid];
        return n && n !== e.target && n.isDragging();
      });
      if (anyStillDragging) return;

      const patches = {};
      for (let i = 0; i < targetIds.length; i += 1) {
        const sid = targetIds[i];
        const node = textNodeRefs.current[sid];
        if (!node) continue;

        const newX = Math.round(node.x());
        const newY = Math.round(node.y());
        const start = session?.starts?.[sid];
        if (
          start &&
          Math.round(start.x) === newX &&
          Math.round(start.y) === newY
        ) {
          continue;
        }

        const bw = Math.round(node.width());
        const bh = Math.round(node.height());
        node.position({ x: newX, y: newY });
        syncGuideNodeToTextNode(sid, node);

        patches[sid] = {
          x: newX,
          y: newY,
          bbox: [newX, newY, newX + bw, newY + bh],
        };
      }

      dragSessionRef.current = null;
      const changedIds = Object.keys(patches);
      if (changedIds.length === 1 && onUpdateBlock) {
        // Record 1 Undo/Redo snapshot on dragEnd
        onUpdateBlock(changedIds[0], patches[changedIds[0]], {
          recordHistory: true,
        });
      } else if (changedIds.length > 0) {
        // Atomic multi-select batch commit -> 1 Undo/Redo snapshot
        updateBlocksBatch(patches, { recordHistory: true });
      }
    },
    [onUpdateBlock, syncGuideNodeToTextNode, updateBlocksBatch]
  );

  // ── Konva Transformer Single & Multi-Select Resize / Rotate ────────────────
  const handleTransformerTransform = useCallback(() => {
    const tr = transformerRef.current;
    if (!tr) return;
    const nodes = tr.nodes();

    if (nodes.length === 1) {
      // Reflow single text box smoothly during resize without distorting font aspect
      const node = nodes[0];
      const sx = node.scaleX();
      const sy = node.scaleY();
      if (sx !== 1 || sy !== 1) {
        node.scaleX(1);
        node.scaleY(1);
        node.width(Math.max(32, node.width() * Math.abs(sx)));
        node.height(Math.max(20, node.height() * Math.abs(sy)));
      }
      syncGuideNodeToTextNode(node.id(), node);
    } else {
      for (let i = 0; i < nodes.length; i += 1) {
        syncGuideNodeToTextNode(nodes[i].id(), nodes[i]);
      }
    }
    boxesLayerRef.current?.batchDraw();
  }, [syncGuideNodeToTextNode]);

  const handleTransformerEnd = useCallback(() => {
    const tr = transformerRef.current;
    if (!tr) return;
    const nodes = tr.nodes();
    if (nodes.length === 0) return;

    const ts = useEditorStore.getState().typesetting;
    const patches = {};
    for (let i = 0; i < nodes.length; i += 1) {
      const node = nodes[i];
      const sx = node.scaleX();
      const sy = node.scaleY();
      node.scaleX(1);
      node.scaleY(1);

      const newX = Math.round(node.x());
      const newY = Math.round(node.y());
      const newWidth = Math.round(Math.max(20, node.width() * Math.abs(sx)));
      const newHeight = Math.round(Math.max(20, node.height() * Math.abs(sy)));
      const newRotation = Number(node.rotation().toFixed(2));

      node.position({ x: newX, y: newY });
      node.width(newWidth);
      node.height(newHeight);

      const block = (blocks || []).find((b) => b.id === node.id());
      const { effectiveMaxFont, effectiveMinFont } = getResolutionFontBounds(
        imageDimsRef.current.width,
        ts?.defaultFontSize || 18
      );
      let nextFontSize = effectiveMaxFont;
      if (block && ts?.autoFitFontSize !== false) {
        const rawFontFamily =
          block?.style?.font_family || block?.font_family || ts?.fontFamily || DEFAULT_FONT_FAMILY;
        const fontFamily = buildKonvaFontFamily(rawFontFamily, ts?.fontFallbackChain);
        const lineHeight =
          block?.style?.line_height || block?.line_height || ts?.defaultLineHeight || 1.18;
        const useAllCaps =
          block?.all_caps !== undefined
            ? Boolean(block.all_caps)
            : block?.style?.all_caps !== undefined
            ? Boolean(block.style.all_caps)
            : Boolean(ts?.allCaps);
        const rawText = block?.translated_text || '';
        const displayText = useAllCaps ? rawText.toUpperCase() : rawText;
        if (displayText) {
          const fitted = fitTextToBox(
            displayText,
            newWidth,
            newHeight,
            fontFamily,
            effectiveMaxFont,
            effectiveMinFont,
            { lineHeight }
          );
          nextFontSize = fitted.fontSize;
          node.fontSize(fitted.fontSize);
          node.text(fitted.text);
        }
      } else if (block) {
        nextFontSize = block?.font_size || block?.style?.font_size || effectiveMaxFont;
      }

      syncGuideNodeToTextNode(node.id(), node);

      patches[node.id()] = {
        x: newX,
        y: newY,
        width: newWidth,
        height: newHeight,
        rotation: newRotation,
        bbox: [newX, newY, newX + newWidth, newY + newHeight],
        font_size: nextFontSize,
        style: {
          ...(block?.style || {}),
          font_size: nextFontSize,
        },
      };
    }

    textLayerRef.current?.batchDraw();
    boxesLayerRef.current?.batchDraw();

    const changedIds = Object.keys(patches);
    if (changedIds.length === 1 && onUpdateBlock) {
      onUpdateBlock(changedIds[0], patches[changedIds[0]], {
        recordHistory: true,
      });
    } else if (changedIds.length > 0) {
      updateBlocksBatch(patches, { recordHistory: true });
    }
  }, [blocks, onUpdateBlock, syncGuideNodeToTextNode, updateBlocksBatch]);

  // ── 60fps Wheel Zoom (rAF Smooth Interpolation) ────────────────────────────
  const handleWheel = useCallback(
    (e) => {
      e.evt.preventDefault();
      const stage = stageRef.current;
      if (!stage) return;

      const pointer = stage.getPointerPosition();
      if (!pointer) return;

      const curTarget = targetTransformRef.current;
      const oldScale = curTarget.scale;

      // Normalize delta across mouse wheel & high-precision trackpads
      const rawDeltaY = e.evt.deltaY;
      const clampedDelta = Math.max(-180, Math.min(180, rawDeltaY));
      const zoomFactor = Math.exp(-clampedDelta * 0.0018);
      const newScale = clampZoom(oldScale * zoomFactor);

      const mousePointTo = {
        x: (pointer.x - curTarget.x) / oldScale,
        y: (pointer.y - curTarget.y) / oldScale,
      };

      targetTransformRef.current = {
        scale: newScale,
        x: pointer.x - mousePointTo.x * newScale,
        y: pointer.y - mousePointTo.y * newScale,
      };

      scheduleTransformLoop();
    },
    [scheduleTransformLoop]
  );

  // ── 60fps rAF Panning (Pan Tool, Space+Drag, Middle-Mouse Drag) ────────────
  const startPanGesture = useCallback(
    (clientX, clientY) => {
      if (transformRafRef.current !== null) {
        window.cancelAnimationFrame(transformRafRef.current);
        transformRafRef.current = null;
      }

      const startPos = {
        clientX,
        clientY,
        stageX: transformRef.current.x,
        stageY: transformRef.current.y,
        latestClientX: clientX,
        latestClientY: clientY,
      };
      panStateRef.current = startPos;

      const container = containerRef.current;
      if (container) container.style.cursor = 'grabbing';

      const handleMove = (ev) => {
        if (!panStateRef.current) return;
        panStateRef.current.latestClientX = ev.clientX;
        panStateRef.current.latestClientY = ev.clientY;

        if (panRafRef.current !== null) return;
        panRafRef.current = window.requestAnimationFrame(() => {
          panRafRef.current = null;
          const ps = panStateRef.current;
          if (!ps) return;
          const nextX = ps.stageX + (ps.latestClientX - ps.clientX);
          const nextY = ps.stageY + (ps.latestClientY - ps.clientY);
          const scale = transformRef.current.scale;
          transformRef.current = { scale, x: nextX, y: nextY };
          targetTransformRef.current = { scale, x: nextX, y: nextY };

          const stage = stageRef.current;
          if (stage) {
            stage.position({ x: nextX, y: nextY });
            stage.batchDraw();
          }
          syncOverlayPositions();
        });
      };

      const handleUp = () => {
        if (panRafRef.current !== null) {
          window.cancelAnimationFrame(panRafRef.current);
          panRafRef.current = null;
        }
        const ps = panStateRef.current;
        if (ps) {
          const nextX = ps.stageX + (ps.latestClientX - ps.clientX);
          const nextY = ps.stageY + (ps.latestClientY - ps.clientY);
          const scale = transformRef.current.scale;
          transformRef.current = { scale, x: nextX, y: nextY };
          targetTransformRef.current = { scale, x: nextX, y: nextY };
          if (stageRef.current) {
            stageRef.current.position({ x: nextX, y: nextY });
            stageRef.current.batchDraw();
          }
          syncOverlayPositions();
        }
        panStateRef.current = null;
        if (container) container.style.cursor = '';
        window.removeEventListener('pointermove', handleMove);
        window.removeEventListener('pointerup', handleUp);
        window.removeEventListener('pointercancel', handleUp);
      };

      window.addEventListener('pointermove', handleMove);
      window.addEventListener('pointerup', handleUp);
      window.addEventListener('pointercancel', handleUp);
    },
    [syncOverlayPositions]
  );

  // ── Inline Text Editor Commit (declared before stage pointer handlers) ─────
  const commitInlineEdit = useCallback(() => {
    const current = inlineEditStateRef.current;
    if (!current) return;
    inlineEditStateRef.current = null;
    setInlineEditState(null);
    const existing = blocksRef.current.find((b) => b.id === current.id);
    if (existing && existing.translated_text !== current.text) {
      if (onUpdateBlock) {
        onUpdateBlock(
          current.id,
          { translated_text: current.text },
          { recordHistory: true }
        );
      } else {
        updateBlock(
          current.id,
          { translated_text: current.text },
          { recordHistory: true }
        );
      }
    }
  }, [onUpdateBlock, updateBlock]);

  // ── Stage Pointer Handlers (Pan, Marquee Multi-Select, Draw Manual Text Box) ─
  const handleStagePointerDown = useCallback(
    (e) => {
      const evt = e.evt;
      const stage = stageRef.current;
      if (!stage) return;

      // Commit any active inline edit when clicking outside
      if (inlineEditState) {
        commitInlineEdit();
      }

      // Middle mouse button ALWAYS pans smoothly in rAF
      if (evt?.button === 1) {
        evt.preventDefault();
        startPanGesture(evt.clientX, evt.clientY);
        return;
      }

      // Only handle primary button for pan/select/draw
      if (evt?.button !== undefined && evt.button !== 0) return;

      // Pan mode (Pan tool or Spacebar held)
      if (isPanMode) {
        startPanGesture(evt.clientX, evt.clientY);
        return;
      }

      const clickedEmpty = e.target === stage;
      const pointer = stage.getPointerPosition();
      if (!pointer) return;
      const pagePt = stagePointToPage(pointer);

      // Manual Text Box Draw Tool
      if (isDrawTextMode) {
        drawBoxStateRef.current = {
          startX: pagePt.x,
          startY: pagePt.y,
          currX: pagePt.x,
          currY: pagePt.y,
        };
        const preview = drawPreviewRectRef.current;
        if (preview) {
          preview.setAttrs({
            x: pagePt.x,
            y: pagePt.y,
            width: 0,
            height: 0,
            visible: true,
          });
          uiLayerRef.current?.batchDraw();
        }
        return;
      }

      // Select Tool on Empty Stage: Start Rubber-Band Marquee Multi-Select
      if (clickedEmpty) {
        marqueeStateRef.current = {
          startX: pagePt.x,
          startY: pagePt.y,
          currX: pagePt.x,
          currY: pagePt.y,
          additive: Boolean(evt?.shiftKey || evt?.ctrlKey || evt?.metaKey),
          moved: false,
        };
        const marquee = marqueeRectRef.current;
        if (marquee) {
          marquee.setAttrs({
            x: pagePt.x,
            y: pagePt.y,
            width: 0,
            height: 0,
            visible: true,
          });
          uiLayerRef.current?.batchDraw();
        }
      }
    },
    [
      commitInlineEdit,
      inlineEditState,
      isDrawTextMode,
      isPanMode,
      stagePointToPage,
      startPanGesture,
    ]
  );

  const handleStagePointerMove = useCallback(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const pointer = stage.getPointerPosition();
    if (!pointer) return;

    const pagePt = stagePointToPage(pointer);

    // Update StatusBar 1:1 image coordinates throttled via rAF
    pendingCursorRef.current = {
      x: Math.round(pagePt.x),
      y: Math.round(pagePt.y),
    };
    if (cursorRafRef.current === null) {
      cursorRafRef.current = window.requestAnimationFrame(() => {
        cursorRafRef.current = null;
        const pt = pendingCursorRef.current;
        if (pt) {
          setCursorPosition(pt.x, pt.y);
        }
      });
    }

    // Update Manual Text Box Draw Preview imperatively (zero React re-render)
    if (drawBoxStateRef.current) {
      const ds = drawBoxStateRef.current;
      ds.currX = pagePt.x;
      ds.currY = pagePt.y;
      const preview = drawPreviewRectRef.current;
      if (preview) {
        preview.setAttrs({
          x: Math.min(ds.startX, ds.currX),
          y: Math.min(ds.startY, ds.currY),
          width: Math.abs(ds.currX - ds.startX),
          height: Math.abs(ds.currY - ds.startY),
        });
        uiLayerRef.current?.batchDraw();
      }
      return;
    }

    // Update Rubber-Band Marquee Selection Rectangle imperatively (zero React re-render)
    if (marqueeStateRef.current) {
      const ms = marqueeStateRef.current;
      ms.currX = pagePt.x;
      ms.currY = pagePt.y;
      if (
        Math.abs(ms.currX - ms.startX) > 4 ||
        Math.abs(ms.currY - ms.startY) > 4
      ) {
        ms.moved = true;
      }
      const marquee = marqueeRectRef.current;
      if (marquee) {
        marquee.setAttrs({
          x: Math.min(ms.startX, ms.currX),
          y: Math.min(ms.startY, ms.currY),
          width: Math.abs(ms.currX - ms.startX),
          height: Math.abs(ms.currY - ms.startY),
        });
        uiLayerRef.current?.batchDraw();
      }
    }
  }, [setCursorPosition, stagePointToPage]);

  const handleStagePointerUp = useCallback(() => {
    // 1. Finalize Manual Text Box Creation
    if (drawBoxStateRef.current) {
      const ds = drawBoxStateRef.current;
      drawBoxStateRef.current = null;
      if (drawPreviewRectRef.current) {
        drawPreviewRectRef.current.visible(false);
        uiLayerRef.current?.batchDraw();
      }

      const rawW = Math.abs(ds.currX - ds.startX);
      const rawH = Math.abs(ds.currY - ds.startY);
      const { width: imgW, height: imgH } = imageDimsRef.current;

      let boxX;
      let boxY;
      let boxW;
      let boxH;
      if (rawW >= 14 && rawH >= 14) {
        boxX = Math.round(Math.max(0, Math.min(imgW - 40, Math.min(ds.startX, ds.currX))));
        boxY = Math.round(Math.max(0, Math.min(imgH - 28, Math.min(ds.startY, ds.currY))));
        boxW = Math.round(Math.max(48, Math.min(imgW - boxX, rawW)));
        boxH = Math.round(Math.max(32, Math.min(imgH - boxY, rawH)));
      } else {
        boxW = 180;
        boxH = 72;
        boxX = Math.round(Math.max(0, Math.min(imgW - boxW, ds.startX - boxW / 2)));
        boxY = Math.round(Math.max(0, Math.min(imgH - boxH, ds.startY - boxH / 2)));
      }

      addBlock({
        x: boxX,
        y: boxY,
        width: boxW,
        height: boxH,
        bbox: [boxX, boxY, boxX + boxW, boxY + boxH],
        translated_text: 'New Text',
      });
      setActiveTool('select');
      return;
    }

    // 2. Finalize Rubber-Band Marquee Multi-Select (or clear selection on plain empty click)
    if (marqueeStateRef.current) {
      const ms = marqueeStateRef.current;
      marqueeStateRef.current = null;
      if (marqueeRectRef.current) {
        marqueeRectRef.current.visible(false);
        uiLayerRef.current?.batchDraw();
      }

      if (!ms.moved) {
        if (!ms.additive) {
          setSelectedLayers([]);
          if (onSelectBlock) onSelectBlock(null);
        }
        return;
      }

      const selBox = {
        x: Math.min(ms.startX, ms.currX),
        y: Math.min(ms.startY, ms.currY),
        width: Math.abs(ms.currX - ms.startX),
        height: Math.abs(ms.currY - ms.startY),
      };

      const hits = blocksRef.current
        .filter((b) => {
          const g = getBlockGeometry(b);
          return Konva.Util.haveIntersection(selBox, {
            x: g.x,
            y: g.y,
            width: g.width,
            height: g.height,
          });
        })
        .map((b) => b.id);

      const nextIds = ms.additive
        ? Array.from(new Set([...selectedIdsRef.current, ...hits]))
        : hits;
      setSelectedLayers(nextIds);
      if (onSelectBlock) onSelectBlock(nextIds[0] ?? null);
    }
  }, [addBlock, onSelectBlock, setActiveTool, setSelectedLayers]);

  const handleStagePointerLeave = useCallback(() => {
    if (cursorRafRef.current !== null) {
      window.cancelAnimationFrame(cursorRafRef.current);
      cursorRafRef.current = null;
    }
    pendingCursorRef.current = null;
    clearCursorPosition();
  }, [clearCursorPosition]);

  // ── Split-Curtain Before/After Divider Drag (60fps rAF) ────────────────────
  const updateSplitCurtainFromClientX = useCallback(
    (clientX) => {
      const container = containerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const localX = clientX - rect.left;
      const { scale, x } = transformRef.current;
      const pageX = (localX - x) / scale;
      const nextRatio = clampSplitRatio(pageX / imageDimsRef.current.width);
      splitRatioRef.current = nextRatio;

      if (splitDragRafRef.current !== null) return;
      splitDragRafRef.current = window.requestAnimationFrame(() => {
        splitDragRafRef.current = null;
        applySplitClip(splitRatioRef.current);
        stageRef.current?.batchDraw();
        syncOverlayPositions();
      });
    },
    [applySplitClip, syncOverlayPositions]
  );

  const handleSplitPointerDown = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      isDraggingSplitRef.current = true;
      e.currentTarget.setPointerCapture?.(e.pointerId);
      updateSplitCurtainFromClientX(e.clientX);
    },
    [updateSplitCurtainFromClientX]
  );

  const handleSplitPointerMove = useCallback(
    (e) => {
      if (!isDraggingSplitRef.current) return;
      e.preventDefault();
      e.stopPropagation();
      updateSplitCurtainFromClientX(e.clientX);
    },
    [updateSplitCurtainFromClientX]
  );

  const handleSplitPointerUp = useCallback(
    (e) => {
      if (!isDraggingSplitRef.current) return;
      e.stopPropagation();
      isDraggingSplitRef.current = false;
      e.currentTarget.releasePointerCapture?.(e.pointerId);
      if (splitDragRafRef.current !== null) {
        window.cancelAnimationFrame(splitDragRafRef.current);
        splitDragRafRef.current = null;
      }
      applySplitClip(splitRatioRef.current);
      stageRef.current?.batchDraw();
      syncOverlayPositions();
      setSplitPercentAria(Math.round(splitRatioRef.current * 100));
    },
    [applySplitClip, syncOverlayPositions]
  );

  const handleSplitKeyDown = useCallback(
    (e) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      e.preventDefault();
      const delta = (e.shiftKey ? 0.08 : 0.02) * (e.key === 'ArrowRight' ? 1 : -1);
      splitRatioRef.current = clampSplitRatio(splitRatioRef.current + delta);
      applySplitClip(splitRatioRef.current);
      stageRef.current?.batchDraw();
      syncOverlayPositions();
      setSplitPercentAria(Math.round(splitRatioRef.current * 100));
    },
    [applySplitClip, syncOverlayPositions]
  );

  const handleSplitDoubleClick = useCallback(
    (e) => {
      e.stopPropagation();
      splitRatioRef.current = 0.5;
      applySplitClip(0.5);
      stageRef.current?.batchDraw();
      syncOverlayPositions();
      setSplitPercentAria(50);
    },
    [applySplitClip, syncOverlayPositions]
  );

  // ── Inline Text Editor Positioning ─────────────────────────────────────────
  const inlineEditBoxStyle = useMemo(() => {
    if (!inlineEditState) return null;
    const block = blocks.find((b) => b.id === inlineEditState.id);
    if (!block) return null;
    const geom = getBlockGeometry(block);
    const style = getBlockTextStyle(block, geom, imageWidth);
    const { scale, x, y } = transformRef.current;

    return {
      left: `${x + geom.x * scale}px`,
      top: `${y + geom.y * scale}px`,
      width: `${Math.max(48, geom.width * scale)}px`,
      height: `${Math.max(32, geom.height * scale)}px`,
      fontSize: `${Math.max(10, style.fontSize * scale)}px`,
      fontFamily: style.fontFamily,
      textAlign: style.align,
      lineHeight: style.lineHeight,
      color: style.fill,
      transform: geom.rotation ? `rotate(${geom.rotation}deg)` : undefined,
      transformOrigin: 'top left',
    };
  }, [inlineEditState, blocks, imageWidth]);

  // ── File Drag & Drop Import ────────────────────────────────────────────────
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingFileOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingFileOver(false);
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingFileOver(false);
    if (!onImportDrop) return;

    const items = Array.from(e.dataTransfer?.items || []);
    const hasEntries = items.some(
      (it) => typeof it.webkitGetAsEntry === 'function'
    );

    let droppedFiles = [];
    if (hasEntries) {
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
    }

    if (droppedFiles.length === 0) {
      droppedFiles = Array.from(e.dataTransfer?.files || []);
    }

    if (droppedFiles.length > 0) {
      onImportDrop(droppedFiles);
    }
  };

  // Cursor style reflecting active canvas mode
  const canvasCursorClass = isPanMode
    ? 'cursor-grab active:cursor-grabbing'
    : isDrawTextMode
    ? 'cursor-crosshair'
    : 'cursor-default';

  return (
    <div
      ref={containerRef}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`relative flex-1 h-full w-full bg-studio-canvas overflow-hidden select-none ${canvasCursorClass}`}
      style={{
        backgroundImage: isDarkTheme
          ? 'radial-gradient(rgba(110, 116, 128, 0.22) 1px, transparent 1px)'
          : 'radial-gradient(rgba(118, 124, 138, 0.28) 1px, transparent 1px)',
        backgroundSize: '20px 20px',
      }}
      data-testid="manga-canvas-container"
    >
      {/* File Drag & Drop Overlay */}
      {isDraggingFileOver && (
        <div className="absolute inset-4 z-40 rounded-studio-md border-2 border-dashed border-studio-accent bg-studio-panel/90 backdrop-blur-sm flex flex-col items-center justify-center gap-3 pointer-events-none">
          <FileImage className="h-10 w-10 text-studio-accent animate-bounce" />
          <p className="text-xs font-semibold text-studio-text">
            Drop Comic Page(s), Folder, or .cbz / .zip to Import
          </p>
        </div>
      )}

      {/* Empty Workspace State when no page is loaded */}
      {!page && (
        <div className="h-full w-full flex items-center justify-center p-6">
          <div className="max-w-md w-full rounded-studio-md border border-dashed border-studio-border-strong bg-studio-panel/90 p-7 text-center space-y-4">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-studio border border-studio-accent/35 bg-studio-accent/10 text-studio-accent">
              <FileImage className="h-6 w-6" />
            </div>
            <div className="space-y-1.5">
              <h2 className="text-sm font-semibold text-studio-text">
                No Comic Page Loaded
              </h2>
              <p className="text-xs text-studio-text-secondary leading-relaxed">
                Use <span className="text-studio-accent font-semibold">File → Import Pages (Ctrl+O)</span> or
                drag &amp; drop Manga, Manhwa, or Manhua pages directly onto the canvas.
              </p>
            </div>
            <div className="pt-1 flex flex-wrap items-center justify-center gap-1.5 text-[11px] font-mono text-studio-text-secondary">
              <span className="rounded-studio-sm bg-studio-surface border border-studio-border px-2 py-0.5">
                1. ONNX Detect
              </span>
              <span className="rounded-studio-sm bg-studio-surface border border-studio-border px-2 py-0.5">
                2. OCR &amp; LaMa Inpaint
              </span>
              <span className="rounded-studio-sm bg-studio-surface border border-studio-border px-2 py-0.5">
                3. LLM Translate
              </span>
            </div>
            {onLoadSample && (
              <div className="pt-2.5 border-t border-studio-border flex flex-wrap items-center justify-center gap-2">
                <button
                  type="button"
                  onClick={() => onLoadSample('sample_ja.jpg', 'ja')}
                  className="studio-btn-accent h-7 px-3 text-xs"
                >
                  <Sparkles className="h-3.5 w-3.5" />
                  Try sample_ja.jpg (JA)
                </button>
                <button
                  type="button"
                  onClick={() => onLoadSample('sample_ko.jpg', 'ko')}
                  className="studio-btn h-7 px-2.5 text-xs"
                >
                  sample_ko.jpg (KO)
                </button>
                <button
                  type="button"
                  onClick={() => onLoadSample('sample_zh.jpg', 'zh')}
                  className="studio-btn h-7 px-2.5 text-xs"
                >
                  sample_zh.jpg (ZH)
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Konva 60fps Multi-Layer Interactive Stage */}
      {hasRenderablePage && (
        <Stage
          ref={stageRef}
          width={viewport.width}
          height={viewport.height}
          onWheel={handleWheel}
          onMouseDown={handleStagePointerDown}
          onMouseMove={handleStagePointerMove}
          onMouseUp={handleStagePointerUp}
          onMouseLeave={handleStagePointerLeave}
          onTouchStart={handleStagePointerDown}
        >
          {/* LAYER 1: Dedicated Static Base / Inpainted Image Layer — listening={false} for 60fps */}
          <Layer ref={baseLayerRef} listening={false} perfectDrawEnabled={false}>
            {/* Subtle page backdrop & drop shadow */}
            <KonvaRect
              x={0}
              y={0}
              width={imageWidth}
              height={imageHeight}
              fill={isDarkTheme ? '#0B0C0E' : '#FFFFFF'}
              shadowColor="#000000"
              shadowBlur={isDarkTheme ? 28 : 16}
              shadowOpacity={isDarkTheme ? 0.55 : 0.16}
              shadowForStrokeEnabled={false}
              perfectDrawEnabled={false}
            />

            {/* Neutral surface when both Original and Inpainted layers are toggled off */}
            {!showOriginalLayer && !showInpaintedLayer && (
              <KonvaRect
                x={0}
                y={0}
                width={imageWidth}
                height={imageHeight}
                fill={isDarkTheme ? '#181A1D' : '#F7F7F5'}
                stroke={isDarkTheme ? '#2A2D32' : '#D4D6DC'}
                strokeWidth={2}
                perfectDrawEnabled={false}
              />
            )}

            {/* Original Raw Comic Page */}
            {rawOriginalImageEl && showOriginalLayer && (
              <KonvaImage
                name="original-comic-image"
                image={rawOriginalImageEl}
                x={0}
                y={0}
                width={imageWidth}
                height={imageHeight}
                perfectDrawEnabled={false}
              />
            )}

            {/* LaMa Inpainted Layer (loaded from /api/cache/images/...; clipped right of curtain in Split-Curtain mode) */}
            {inpaintedImageEl && showInpaintedLayer && (
              <KonvaGroup ref={inpaintClipGroupRef} listening={false}>
                <KonvaImage
                  name="inpainted-comic-image"
                  image={inpaintedImageEl}
                  x={0}
                  y={0}
                  width={imageWidth}
                  height={imageHeight}
                  perfectDrawEnabled={false}
                />
              </KonvaGroup>
            )}
          </Layer>

          {/* LAYER 2: Dedicated Inpaint Mask Overlay Layer — listening={false} */}
          <Layer
            ref={maskLayerRef}
            listening={false}
            visible={showMaskLayer}
            perfectDrawEnabled={false}
          >
            <KonvaGroup ref={maskClipGroupRef} listening={false}>
              {maskOverlayCanvas ? (
                <KonvaImage
                  name="inpaint-mask-overlay"
                  image={maskOverlayCanvas}
                  x={0}
                  y={0}
                  width={imageWidth}
                  height={imageHeight}
                  perfectDrawEnabled={false}
                />
              ) : (
                blocks.map((block) => (
                  <MemoMaskFallbackShape key={`mask_${block.id}`} block={block} />
                ))
              )}

              {(page?.maskStrokes || []).map((stroke, idx) => (
                <KonvaLine
                  key={stroke.id || `mask_stroke_${idx}`}
                  points={stroke.points || []}
                  stroke={
                    stroke.mode === 'eraser'
                      ? 'rgba(0, 0, 0, 1)'
                      : 'rgba(239, 68, 68, 0.58)'
                  }
                  strokeWidth={stroke.size || 24}
                  lineCap="round"
                  lineJoin="round"
                  globalCompositeOperation={
                    stroke.mode === 'eraser' ? 'destination-out' : 'source-over'
                  }
                  perfectDrawEnabled={false}
                />
              ))}
            </KonvaGroup>
          </Layer>

          {/* LAYER 3: Memoized Translated Text Elements (Isolated per-bubble re-rendering) */}
          <Layer ref={textLayerRef} visible={showTextLayer}>
            <KonvaGroup ref={textClipGroupRef}>
              {blocks.map((block) => (
                <MemoTextNode
                  key={block.id}
                  block={block}
                  imgWidth={imageWidth}
                  targetLang={page?.targetLang}
                  fontRenderKey={fontRenderKey}
                  isInteractive={isTextInteractive}
                  isEditingInline={inlineEditState?.id === block.id}
                  registerNodeRef={registerTextNodeRef}
                  onHoverChange={handleBlockHoverChange}
                  onBlockMouseDown={handleBlockMouseDown}
                  onBlockClick={handleBlockClick}
                  onBlockDoubleClick={handleBlockDoubleClick}
                  onBlockDragStart={handleBlockDragStart}
                  onBlockDragMove={handleBlockDragMove}
                  onBlockDragEnd={handleBlockDragEnd}
                />
              ))}
            </KonvaGroup>
          </Layer>

          {/* LAYER 4: Bounding Boxes & Reading Order #1/#2 Guides — listening={false} */}
          <Layer
            ref={boxesLayerRef}
            listening={false}
            visible={showBoxesLayer}
            perfectDrawEnabled={false}
          >
            <KonvaGroup ref={boxesClipGroupRef} listening={false}>
              {blocks.map((block, index) => (
                <MemoBoxGuideNode
                  key={`guide_${block.id}`}
                  block={block}
                  orderNumber={block.order ?? index + 1}
                  isSelected={selectedIdSet.has(block.id)}
                  isHovered={hoveredBlockId === block.id}
                  registerGuideRef={registerGuideRef}
                />
              ))}
            </KonvaGroup>
          </Layer>

          {/* LAYER 5: Interactive Editing Overlay (Selection Outlines, Marquee, Draw Preview, Transformer) */}
          <Layer ref={uiLayerRef} visible={!isHoldComparing}>
            {!showBoxesLayer &&
              blocks.map((block) => {
                const isSelected = selectedIdSet.has(block.id);
                const isHovered = hoveredBlockId === block.id;
                if (isSelected || !isHovered) return null;
                return (
                  <MemoSelectionOutline
                    key={`outline_${block.id}`}
                    block={block}
                    isSelected={false}
                  />
                );
              })}

            {/* Marquee Multi-Select Rectangle */}
            <KonvaRect
              ref={marqueeRectRef}
              visible={false}
              fill="rgba(229, 154, 60, 0.14)"
              stroke="#E59A3C"
              strokeWidth={1.5}
              dash={[5, 3]}
              listening={false}
              perfectDrawEnabled={false}
            />

            {/* Manual Text Box Creation Preview Rectangle */}
            <KonvaRect
              ref={drawPreviewRectRef}
              visible={false}
              fill="rgba(56, 189, 248, 0.14)"
              stroke="#38BDF8"
              strokeWidth={2}
              dash={[6, 4]}
              cornerRadius={3}
              listening={false}
              perfectDrawEnabled={false}
            />

            {/* Single & Multi-Select Konva Transformer */}
            <Transformer
              ref={transformerRef}
              rotateEnabled={true}
              keepRatio={false}
              enabledAnchors={[
                'top-left',
                'top-center',
                'top-right',
                'middle-left',
                'middle-right',
                'bottom-left',
                'bottom-center',
                'bottom-right',
              ]}
              borderStroke="#E59A3C"
              borderStrokeWidth={1.5}
              anchorStroke="#C97A16"
              anchorFill="#FFFBEB"
              anchorSize={8}
              anchorCornerRadius={2}
              padding={2}
              onTransform={handleTransformerTransform}
              onTransformEnd={handleTransformerEnd}
              boundBoxFunc={(oldBox, newBox) => {
                if (
                  Math.abs(newBox.width) < 32 ||
                  Math.abs(newBox.height) < 20
                ) {
                  return oldBox;
                }
                return newBox;
              }}
            />
          </Layer>
        </Stage>
      )}

      {/* Draggable Split-Curtain Before/After Compare Divider (60fps Imperative Overlay) */}
      {hasRenderablePage && isSplitCompare && (
        <div
          className="absolute inset-0 z-20 pointer-events-none overflow-hidden"
          data-testid="split-curtain-overlay"
        >
          <div
            ref={splitLineOverlayRef}
            className="absolute top-0 left-0 w-0 will-change-transform"
          >
            {/* Generous pointer hit strip along vertical divider */}
            <div
              role="slider"
              tabIndex={0}
              aria-label="Before and After Split Curtain Divider"
              aria-orientation="horizontal"
              aria-valuemin={2}
              aria-valuemax={98}
              aria-valuenow={splitPercentAria}
              data-testid="split-curtain-handle"
              onPointerDown={handleSplitPointerDown}
              onPointerMove={handleSplitPointerMove}
              onPointerUp={handleSplitPointerUp}
              onPointerCancel={handleSplitPointerUp}
              onKeyDown={handleSplitKeyDown}
              onDoubleClick={handleSplitDoubleClick}
              className="pointer-events-auto absolute top-0 -left-3 w-6 h-full cursor-ew-resize group focus:outline-none"
              title="Drag to compare Original (left) vs Translated (right) · Double-click to reset 50%"
            >
              {/* Crisp 2px Warm Amber Vertical Curtain Hairline */}
              <div className="mx-auto w-0.5 h-full bg-studio-accent shadow-[0_0_8px_rgba(0,0,0,0.65)]" />

              {/* Tactile Center Grab Handle */}
              <div
                ref={splitHandleRef}
                className="absolute left-1/2 top-0 h-9 w-9 rounded-full border-2 border-studio-accent bg-studio-panel/95 text-studio-accent shadow-lg flex items-center justify-center group-hover:scale-105 group-focus:ring-2 group-focus:ring-studio-accent/50 transition-transform"
              >
                <ChevronsLeftRight className="h-4 w-4 stroke-[2.25]" />
              </div>
            </div>

            {/* BEFORE / AFTER Labels anchored cleanly to either side of the curtain */}
            <div
              ref={splitLeftBadgeRef}
              className="absolute top-3 right-3 px-2 py-0.5 rounded-studio-sm border border-studio-border bg-studio-panel/90 text-[10px] font-mono font-semibold uppercase tracking-wider text-studio-text shadow-md whitespace-nowrap transition-opacity"
            >
              Before · Original
            </div>
            <div
              ref={splitRightBadgeRef}
              className="absolute top-3 left-3 px-2 py-0.5 rounded-studio-sm border border-studio-accent/60 bg-studio-panel/90 text-[10px] font-mono font-semibold uppercase tracking-wider text-studio-accent shadow-md whitespace-nowrap transition-opacity"
            >
              After · Translated
            </div>
          </div>
        </div>
      )}

      {/* Inline Textarea Editor on Double-Click */}
      {inlineEditState && inlineEditBoxStyle && (
        <textarea
          autoFocus
          value={inlineEditState.text}
          onChange={(e) =>
            setInlineEditState((prev) =>
              prev ? { ...prev, text: e.target.value } : null
            )
          }
          onBlur={commitInlineEdit}
          onKeyDown={(e) => {
            if (e.key === 'Escape') {
              e.preventDefault();
              setInlineEditState(null);
            } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
              e.preventDefault();
              commitInlineEdit();
            }
          }}
          style={inlineEditBoxStyle}
          className="absolute z-30 rounded-studio-sm border-2 border-studio-accent bg-white/95 p-1 font-bold shadow-lg outline-none resize-none select-text"
          data-testid="inline-text-editor"
        />
      )}

      {/* Floating Top-Left Canvas Tool & Undo/Redo Strip */}
      {hasRenderablePage && (
        <div className="absolute top-3 left-3 z-20 flex items-center gap-1 rounded-studio border border-studio-border bg-studio-panel/95 backdrop-blur-sm p-1 shadow-lg">
          {/* Canvas Tools: Select (V), Pan (H), Draw Text Box */}
          <div
            className="inline-flex items-center gap-0.5"
            role="group"
            aria-label="Canvas Tools"
          >
            <button
              type="button"
              onClick={() => setActiveTool('select')}
              aria-pressed={activeTool === 'select' && !isSpacePanning}
              title="Select & Marquee Multi-Select Tool (V) · Shift+Click to multi-select"
              data-testid="tool-select-btn"
              className={`h-6 w-6 inline-flex items-center justify-center rounded-studio-sm transition-colors cursor-pointer ${
                activeTool === 'select' && !isSpacePanning
                  ? 'bg-studio-accent text-studio-accent-fg'
                  : 'text-studio-text-secondary hover:text-studio-text hover:bg-studio-hover'
              }`}
            >
              <MousePointer2 className="h-3.5 w-3.5" />
            </button>
            <button
              type="button"
              onClick={() => setActiveTool('pan')}
              aria-pressed={isPanMode}
              title="Pan Canvas Tool (H, or hold Space / Middle-click drag)"
              data-testid="tool-pan-btn"
              className={`h-6 w-6 inline-flex items-center justify-center rounded-studio-sm transition-colors cursor-pointer ${
                isPanMode
                  ? 'bg-studio-accent text-studio-accent-fg'
                  : 'text-studio-text-secondary hover:text-studio-text hover:bg-studio-hover'
              }`}
            >
              <Hand className="h-3.5 w-3.5" />
            </button>
            <button
              type="button"
              onClick={() =>
                setActiveTool(activeTool === 'text' ? 'select' : 'text')
              }
              aria-pressed={activeTool === 'text'}
              title="Draw Manual Text Box on Canvas (Click or drag on page)"
              data-testid="tool-text-btn"
              className={`h-6 px-1.5 inline-flex items-center gap-1 rounded-studio-sm text-[11px] font-medium transition-colors cursor-pointer ${
                activeTool === 'text'
                  ? 'bg-studio-accent text-studio-accent-fg'
                  : 'text-studio-text-secondary hover:text-studio-text hover:bg-studio-hover'
              }`}
            >
              <Type className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Text Box</span>
            </button>
          </div>

          <span className="h-4 w-px bg-studio-border mx-0.5" aria-hidden="true" />

          {/* Undo / Redo Snapshot Buttons */}
          <div
            className="inline-flex items-center gap-0.5"
            role="group"
            aria-label="Canvas History"
          >
            <button
              type="button"
              onClick={undo}
              disabled={!canUndo}
              title={canUndo ? `Undo (${undoCount}) — Ctrl+Z` : 'Undo (Ctrl+Z)'}
              aria-label="Undo"
              data-testid="canvas-undo-btn"
              className="h-6 w-6 inline-flex items-center justify-center rounded-studio-sm text-studio-text-secondary hover:text-studio-text hover:bg-studio-hover disabled:opacity-35 disabled:pointer-events-none transition-colors cursor-pointer"
            >
              <Undo2 className="h-3.5 w-3.5" />
            </button>
            <button
              type="button"
              onClick={redo}
              disabled={!canRedo}
              title={canRedo ? `Redo (${redoCount}) — Ctrl+Y` : 'Redo (Ctrl+Y)'}
              aria-label="Redo"
              data-testid="canvas-redo-btn"
              className="h-6 w-6 inline-flex items-center justify-center rounded-studio-sm text-studio-text-secondary hover:text-studio-text hover:bg-studio-hover disabled:opacity-35 disabled:pointer-events-none transition-colors cursor-pointer"
            >
              <Redo2 className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      )}

      {/* Floating Top-Right Layer Visibility Toggle Strip (Synced with useEditorStore.viewLayers) */}
      {hasRenderablePage && (
        <div
          className="absolute top-3 right-3 z-20 flex items-center gap-0.5 rounded-studio border border-studio-border bg-studio-panel/95 backdrop-blur-sm p-1 shadow-lg"
          role="group"
          aria-label="Canvas Layer Toggles"
        >
          {LAYER_HUD_TOGGLES.map((layer) => {
            const isOn = Boolean(viewLayers[layer.key]);
            return (
              <button
                key={layer.key}
                type="button"
                onClick={() => toggleViewLayer(layer.key)}
                aria-pressed={isOn}
                title={layer.title}
                data-testid={`layer-toggle-${layer.key}`}
                className={`h-6 px-2 rounded-studio-sm text-[11px] font-medium transition-colors cursor-pointer ${
                  isOn
                    ? 'bg-studio-accent/15 text-studio-accent border border-studio-accent/40'
                    : 'text-studio-text-muted hover:text-studio-text hover:bg-studio-hover border border-transparent'
                }`}
              >
                {layer.label}
              </button>
            );
          })}
        </div>
      )}

      {/* Active Hold-Compare Banner */}
      {page && isHoldComparing && (
        <div className="absolute top-12 left-1/2 -translate-x-1/2 z-30 pointer-events-none">
          <div className="inline-flex items-center gap-2 rounded-studio border border-studio-accent/50 bg-studio-panel/95 backdrop-blur-md px-3.5 py-1 text-xs font-semibold text-studio-accent shadow-lg">
            <Eye className="h-3.5 w-3.5" />
            <span>Comparing Original Raw Page (Release to return)</span>
          </div>
        </div>
      )}

      {/* Draw Text Box Hint Pill */}
      {hasRenderablePage && isDrawTextMode && (
        <div className="absolute top-12 left-1/2 -translate-x-1/2 z-20 pointer-events-none">
          <div className="inline-flex items-center gap-1.5 rounded-studio border border-sky-400/50 bg-studio-panel/95 px-3 py-1 text-[11px] font-medium text-sky-400 shadow-lg">
            <Type className="h-3.5 w-3.5" />
            <span>Click or drag on page to place a manual text box (Esc to cancel)</span>
          </div>
        </div>
      )}

      {/* Floating Bottom HUD: Hold Compare + Split-Curtain Compare + Smooth Zoom + Telemetry */}
      {hasRenderablePage && (
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-20 flex items-center gap-1.5 rounded-studio border border-studio-border bg-studio-panel/95 backdrop-blur-md px-2 py-1 shadow-xl">
          {/* Hold to Compare Original */}
          <button
            type="button"
            onMouseDown={() => setIsHoldComparingLocal(true)}
            onMouseUp={() => setIsHoldComparingLocal(false)}
            onMouseLeave={() => setIsHoldComparingLocal(false)}
            onTouchStart={() => setIsHoldComparingLocal(true)}
            onTouchEnd={() => setIsHoldComparingLocal(false)}
            data-testid="hold-compare-btn"
            className={`inline-flex items-center gap-1.5 h-6 rounded-studio-sm border px-2 text-[11px] font-medium transition-colors cursor-pointer select-none ${
              isHoldComparing
                ? 'border-studio-accent bg-studio-accent/20 text-studio-accent'
                : 'border-studio-border bg-studio-surface hover:bg-studio-hover text-studio-text'
            }`}
            title="Press and hold (or hold \) to view the original raw comic page"
          >
            <Eye className="h-3 w-3 text-studio-accent" />
            <span className="hidden sm:inline">Hold Original</span>
          </button>

          {/* Split-Curtain Before/After Compare Toggle */}
          <button
            type="button"
            onClick={() => toggleViewLayer('splitCompare')}
            aria-pressed={Boolean(viewLayers.splitCompare)}
            data-testid="split-compare-toggle-btn"
            className={`inline-flex items-center gap-1.5 h-6 rounded-studio-sm border px-2 text-[11px] font-medium transition-colors cursor-pointer select-none ${
              viewLayers.splitCompare
                ? 'border-studio-accent bg-studio-accent text-studio-accent-fg font-semibold'
                : 'border-studio-border bg-studio-surface hover:bg-studio-hover text-studio-text'
            }`}
            title="Toggle draggable Split-Curtain Before/After Compare divider"
          >
            <Columns2 className="h-3 w-3" />
            <span>Split Compare</span>
          </button>

          <div className="h-4 w-px bg-studio-border" />

          {/* Smooth Zoom Controls */}
          <div className="flex items-center gap-0.5">
            <button
              type="button"
              onClick={() => animateZoomTo(targetTransformRef.current.scale / 1.2)}
              className="inline-flex items-center justify-center h-6 w-6 rounded-studio-sm hover:bg-studio-hover text-studio-text-secondary hover:text-studio-text transition-colors cursor-pointer"
              title="Zoom Out (Ctrl+-)"
              aria-label="Zoom Out"
            >
              <ZoomOut className="h-3.5 w-3.5" />
            </button>

            <button
              type="button"
              onClick={fitPageToViewport}
              className="px-1.5 h-6 rounded-studio-sm hover:bg-studio-hover font-mono text-[11px] font-medium text-studio-text tabular-nums transition-colors cursor-pointer min-w-[44px]"
              title="Click to Fit Page to Screen (Ctrl+0)"
            >
              <span ref={zoomLabelRef}>
                {Math.round(transformRef.current.scale * 100)}%
              </span>
            </button>

            <button
              type="button"
              onClick={() => animateZoomTo(targetTransformRef.current.scale * 1.2)}
              className="inline-flex items-center justify-center h-6 w-6 rounded-studio-sm hover:bg-studio-hover text-studio-text-secondary hover:text-studio-text transition-colors cursor-pointer"
              title="Zoom In (Ctrl++)"
              aria-label="Zoom In"
            >
              <ZoomIn className="h-3.5 w-3.5" />
            </button>

            <button
              type="button"
              onClick={fitPageToViewport}
              className="inline-flex items-center justify-center h-6 w-6 rounded-studio-sm hover:bg-studio-hover text-studio-text-muted hover:text-studio-text transition-colors cursor-pointer"
              title="Fit to Viewport (Ctrl+0)"
              aria-label="Fit to Viewport"
            >
              <Maximize2 className="h-3.5 w-3.5" />
            </button>
          </div>

          <div className="hidden md:flex items-center gap-1.5 pl-1.5 border-l border-studio-border text-[10px] font-mono text-studio-text-muted tabular-nums">
            <span>
              {imageWidth}×{imageHeight}px · {blocks.length}
            </span>
            {effectiveSelectedIds.length > 1 && (
              <span className="rounded-studio-sm bg-studio-accent/15 border border-studio-accent/40 px-1 py-0.5 text-studio-accent font-semibold">
                {effectiveSelectedIds.length} selected
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
});

export default MangaCanvas;
