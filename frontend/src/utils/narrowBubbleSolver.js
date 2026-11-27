/**
 * Narrow-Bubble Typesetting Solver (Ignacio + Koharu 7-Step Copyfitting + DP Line Breaker)
 *
 * Designed for narrow/irregular manga speech bubbles:
 *  1. Polygon Band Sampling: intersects horizontal width at 5 evenly-spaced interior
 *     points PLUS all polygon vertices inside [v0 - air, v0 + pitch + air] so sharp
 *     bubble tails never intrude on text.
 *  2. Single-Axis Centering (`roomAxis`): locks ONE vertical center axis maximizing
 *     (covered_bands, symmetric_span) to eliminate zig-zag alignment in slanted bubbles.
 *  3. 7-Step Copyfitting Hierarchy:
 *     - Scale 1.0 whole words
 *     - Shrink whole words down to FLOOR = 0.5 (8-point sweep + 8 binary-halving steps)
 *     - Halve margin (inset / 2.0) inside polygon rooms before hyphenating
 *     - Syllable hyphenation ONLY for Latin words >= 6 letters (HYPHENATION_KEEPS = 2),
 *       strictly skipping CJK, numbers, and symbols
 *  4. Dynamic Programming Line Breaker: sweeps line count 1..64 with DP state
 *     (line_idx, seg_end, consecutive_hyphen_run < 3) minimizing
 *     break_penalty (0.0 punct/end, 2.0 mid-phrase, 1000.0 hyphen)
 *     + quadratic slack_penalty (100.0 * ((room - used) / room)^2)
 *     + overflow_penalty (1e6).
 */

import { polylabel } from './canvasMathOps.js';

export const COPYFIT_FLOOR = 0.5;
export const HYPHENATION_MIN_WORD_LEN = 6;
export const HYPHENATION_KEEPS = 2;
export const MAX_CONSECUTIVE_HYPHENS = 3; // state < 3 -> 0, 1, 2
export const MAX_DP_LINES = 64;
export const SWEEP_POINTS = 8;
export const BINARY_HALVING_STEPS = 8;

export const BREAK_PENALTY_PUNCT_OR_END = 0.0;
export const BREAK_PENALTY_MID_PHRASE = 2.0;
export const BREAK_PENALTY_HYPHEN = 1000.0;
export const SLACK_PENALTY_WEIGHT = 100.0;
export const OVERFLOW_PENALTY = 1e6;

const EPS = 1e-4;

// Offscreen 2D canvas context for text width measurement
let _sharedMeasureCtx = null;
function getDefaultMeasureCtx() {
  if (!_sharedMeasureCtx && typeof document !== 'undefined') {
    const canvas = document.createElement('canvas');
    _sharedMeasureCtx = canvas.getContext('2d');
  }
  return _sharedMeasureCtx;
}

/**
 * Create a cached text-width measurer for a given Canvas font string.
 */
function createFontMeasurer(fontSize, fontFamily, fontStyle = 'normal', customMeasureFn = null) {
  const normStyle = String(fontStyle || 'normal').trim() || 'normal';
  const fontSpec = `${normStyle && normStyle !== 'normal' ? `${normStyle} ` : ''}${fontSize}px ${fontFamily || 'sans-serif'}`;
  const cache = new Map();

  if (typeof customMeasureFn === 'function') {
    return (str) => {
      if (!str) return 0;
      const hit = cache.get(str);
      if (hit !== undefined) return hit;
      const w = Number(customMeasureFn(str, fontSize, fontFamily, fontStyle)) || 0;
      cache.set(str, w);
      return w;
    };
  }

  const ctx = getDefaultMeasureCtx();
  if (!ctx) {
    // Deterministic headless/Node fallback (~0.56em average glyph advance)
    return (str) => {
      if (!str) return 0;
      const hit = cache.get(str);
      if (hit !== undefined) return hit;
      let w = 0;
      for (let i = 0; i < str.length; i += 1) {
        const code = str.charCodeAt(i);
        if (code === 32) {
          w += fontSize * 0.28;
        } else if (code > 0x2e80) {
          w += fontSize * 0.98;
        } else {
          w += fontSize * 0.56;
        }
      }
      cache.set(str, w);
      return w;
    };
  }

  ctx.font = fontSpec;
  return (str) => {
    if (!str) return 0;
    const hit = cache.get(str);
    if (hit !== undefined) return hit;
    if (ctx.font !== fontSpec) {
      ctx.font = fontSpec;
    }
    const w = ctx.measureText(str).width;
    cache.set(str, w);
    return w;
  };
}

// ============================================================================
// Polygon & Room Geometry Helpers
// ============================================================================

/**
 * Normalize a polygon from `[[x, y], ...]` or `[{x, y}, ...]` to `Array<[number, number]>`.
 */
export function normalizePolygon(rawPoly) {
  if (!Array.isArray(rawPoly) || rawPoly.length < 3) return null;
  const pts = [];
  for (let i = 0; i < rawPoly.length; i += 1) {
    const p = rawPoly[i];
    let x;
    let y;
    if (Array.isArray(p) && p.length >= 2) {
      x = Number(p[0]);
      y = Number(p[1]);
    } else if (p && typeof p === 'object') {
      x = Number(p.x);
      y = Number(p.y);
    } else {
      continue;
    }
    if (Number.isFinite(x) && Number.isFinite(y)) {
      pts.push([x, y]);
    }
  }
  return pts.length >= 3 ? pts : null;
}

/**
 * Convert `[x1, y1, x2, y2]` or `{x, y, width, height}` into a 4-vertex rectangle polygon.
 */
export function rectToPolygon(rect) {
  if (!rect) return null;
  let x1;
  let y1;
  let x2;
  let y2;
  if (Array.isArray(rect) && rect.length >= 4) {
    x1 = Number(rect[0]);
    y1 = Number(rect[1]);
    x2 = Number(rect[2]);
    y2 = Number(rect[3]);
  } else if (typeof rect === 'object') {
    x1 = Number(rect.x) || 0;
    y1 = Number(rect.y) || 0;
    x2 = x1 + Math.max(1, Number(rect.width) || 0);
    y2 = y1 + Math.max(1, Number(rect.height) || 0);
  } else {
    return null;
  }
  if (!Number.isFinite(x1) || !Number.isFinite(y1) || !Number.isFinite(x2) || !Number.isFinite(y2)) {
    return null;
  }
  if (x2 <= x1 || y2 <= y1) return null;
  return [
    [x1, y1],
    [x2, y1],
    [x2, y2],
    [x1, y2],
  ];
}

function getPolygonBounds(poly) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < poly.length; i += 1) {
    const [x, y] = poly[i];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  return {
    minX,
    minY,
    maxX,
    maxY,
    width: Math.max(0, maxX - minX),
    height: Math.max(0, maxY - minY),
    centerX: (minX + maxX) * 0.5,
    centerY: (minY + maxY) * 0.5,
  };
}

/**
 * Intersect a horizontal scanline at `y` with a closed polygon.
 * Returns the widest interior horizontal interval `[left, right]` (or null if none).
 */
export function scanlinePolygonInterval(poly, y, preferCenterX = null) {
  const n = poly.length;
  const xs = [];

  for (let i = 0, j = n - 1; i < n; j = i, i += 1) {
    const [x1, y1] = poly[j];
    const [x2, y2] = poly[i];
    // Standard half-open scanline rule avoids double-counting vertices
    if ((y1 <= y && y2 > y) || (y2 <= y && y1 > y)) {
      const t = (y - y1) / (y2 - y1);
      xs.push(x1 + t * (x2 - x1));
    }
  }

  if (xs.length < 2) return null;
  xs.sort((a, b) => a - b);

  let bestLeft = 0;
  let bestRight = 0;
  let bestSpan = -1;
  let bestDist = Infinity;

  for (let k = 0; k + 1 < xs.length; k += 2) {
    const l = xs[k];
    const r = xs[k + 1];
    const span = r - l;
    if (span <= EPS) continue;

    if (preferCenterX !== null) {
      const mid = (l + r) * 0.5;
      const dist = Math.abs(mid - preferCenterX);
      // Prefer significantly wider interval, otherwise closest to room center
      if (span > bestSpan * 1.15 || (span >= bestSpan * 0.85 && dist < bestDist)) {
        bestLeft = l;
        bestRight = r;
        bestSpan = span;
        bestDist = dist;
      }
    } else if (span > bestSpan) {
      bestLeft = l;
      bestRight = r;
      bestSpan = span;
    }
  }

  return bestSpan > EPS ? [bestLeft, bestRight] : null;
}

/**
 * Sample horizontal room width across a candidate line band `[v0 - air, v0 + pitch + air]`.
 *
 * Intersects at:
 *  - 5 evenly-spaced interior points inside the band
 *  - PLUS all polygon vertices whose Y lies inside the band (nudged +/- eps to capture
 *    inward concave vertices and sharp bubble tails)
 *
 * Returns `{ left, right, width, top, bottom }` after applying `inset` on both sides.
 */
export function samplePolygonBand(poly, bounds, v0, pitch, air, inset) {
  const bandTop = v0 - air;
  const bandBottom = v0 + pitch + air;
  const bandSpan = Math.max(EPS, bandBottom - bandTop);

  // If band extends outside the polygon's vertical range, clamp interior check:
  // if the core line [v0, v0 + pitch] itself is outside [minY, maxY], width is 0.
  if (v0 < bounds.minY - EPS || v0 + pitch > bounds.maxY + EPS) {
    return { left: bounds.centerX, right: bounds.centerX, width: 0, top: v0, bottom: v0 + pitch };
  }

  const clampedTop = Math.max(bounds.minY + EPS, bandTop);
  const clampedBottom = Math.min(bounds.maxY - EPS, bandBottom);
  if (clampedBottom <= clampedTop) {
    return { left: bounds.centerX, right: bounds.centerX, width: 0, top: v0, bottom: v0 + pitch };
  }

  const sampleYs = [];
  // 5 evenly-spaced interior points
  for (let k = 1; k <= 5; k += 1) {
    const frac = k / 6;
    const sy = Math.min(clampedBottom, Math.max(clampedTop, bandTop + frac * bandSpan));
    sampleYs.push(sy);
  }

  // PLUS all polygon vertices inside [bandTop, bandBottom]
  for (let i = 0; i < poly.length; i += 1) {
    const vy = poly[i][1];
    if (vy >= bandTop && vy <= bandBottom) {
      const yMinus = Math.max(clampedTop, vy - 0.05);
      const yPlus = Math.min(clampedBottom, vy + 0.05);
      if (yMinus >= clampedTop && yMinus <= clampedBottom) sampleYs.push(yMinus);
      if (yPlus >= clampedTop && yPlus <= clampedBottom) sampleYs.push(yPlus);
    }
  }

  let intersectLeft = -Infinity;
  let intersectRight = Infinity;

  for (let i = 0; i < sampleYs.length; i += 1) {
    const iv = scanlinePolygonInterval(poly, sampleYs[i], bounds.centerX);
    if (!iv) {
      return { left: bounds.centerX, right: bounds.centerX, width: 0, top: v0, bottom: v0 + pitch };
    }
    if (iv[0] > intersectLeft) intersectLeft = iv[0];
    if (iv[1] < intersectRight) intersectRight = iv[1];
    if (intersectRight <= intersectLeft) {
      return { left: bounds.centerX, right: bounds.centerX, width: 0, top: v0, bottom: v0 + pitch };
    }
  }

  const left = intersectLeft + inset;
  const right = intersectRight - inset;
  const width = Math.max(0, right - left);
  return {
    left,
    right,
    width,
    top: v0,
    bottom: v0 + pitch,
  };
}

/**
 * Single-Axis Centering (`roomAxis`):
 * Evaluate band midpoints and lock ONE vertical center axis maximizing
 * `(covered_bands, symmetric_span)` to eliminate zig-zag alignment in slanted bubbles.
 */
export function computeRoomAxis(bands, fallbackCenterX, poleX = null) {
  const candidates = [];
  const validMids = [];

  if (typeof poleX === 'number' && Number.isFinite(poleX)) {
    candidates.push(poleX);
  }

  for (let i = 0; i < bands.length; i += 1) {
    const b = bands[i];
    if (b.width > EPS) {
      const mid = (b.left + b.right) * 0.5;
      candidates.push(mid);
      validMids.push(mid);
    }
  }

  if (validMids.length === 0) {
    return {
      roomAxis: fallbackCenterX,
      coveredBands: 0,
      symmetricSpan: 0,
      bandWidths: bands.map(() => 0),
    };
  }

  // Also test the median midpoint as a robust candidate
  validMids.sort((a, b) => a - b);
  candidates.push(validMids[Math.floor(validMids.length / 2)]);

  let bestAxis = validMids[0];
  let bestCovered = -1;
  let bestSymSpan = -1;
  let bestMinSpan = -1;

  for (let c = 0; c < candidates.length; c += 1) {
    const axis = candidates[c];
    let covered = 0;
    let symSpan = 0;
    let minSpan = Infinity;

    for (let i = 0; i < bands.length; i += 1) {
      const b = bands[i];
      if (b.width <= EPS) continue;
      const half = Math.min(axis - b.left, b.right - axis);
      if (half > EPS) {
        const span = 2 * half;
        covered += 1;
        symSpan += span;
        if (span < minSpan) minSpan = span;
      }
    }

    if (covered === 0) minSpan = 0;

    if (
      covered > bestCovered ||
      (covered === bestCovered && symSpan > bestSymSpan + 1e-3) ||
      (covered === bestCovered && Math.abs(symSpan - bestSymSpan) <= 1e-3 && minSpan > bestMinSpan)
    ) {
      bestAxis = axis;
      bestCovered = covered;
      bestSymSpan = symSpan;
      bestMinSpan = minSpan;
    }
  }

  const bandWidths = bands.map((b) => {
    if (b.width <= EPS) return 0;
    const half = Math.min(bestAxis - b.left, b.right - bestAxis);
    return half > EPS ? 2 * half : 0;
  });

  return {
    roomAxis: bestAxis,
    coveredBands: bestCovered,
    symmetricSpan: bestSymSpan,
    bandWidths,
  };
}

// ============================================================================
// Weak-Script Protection & Latin Syllable Hyphenation
// ============================================================================

const CJK_OR_DIGIT_RE = /[\u2E80-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF\uFF00-\uFFEF0-9]/;
const LATIN_WORD_RE = /^[A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF]+$/;
const VOWEL_RE = /[aeiouyAEIOUY\u00C0-\u00FF]/;
const PUNCT_END_RE = /[.,!?;:…—–\-'"”’)\]\}!?。、！？♡♥♪♫★☆]$/;
const EDGE_STRIP_RE = /^([^A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF]*)([A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF]+)([^A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF]*)$/;

const INSEPARABLE_DIGRAPHS = new Set([
  'ch', 'sh', 'th', 'ph', 'wh', 'ng', 'ny', 'kh', 'sy', 'qu', 'ck', 'gh',
  'bl', 'br', 'cl', 'cr', 'dr', 'fl', 'fr', 'gl', 'gr', 'pl', 'pr', 'tr', 'st', 'sp', 'sk',
]);

/**
 * Weak-script symbols (`♡`, `♪`, `…`, etc.) that should stay attached to their
 * neighboring word rather than becoming orphaned standalone tokens.
 */
const WEAK_SCRIPT_ONLY_RE = /^[♡♥♪♫♬★☆…—–・〜～!?.,;:]+$/;

/**
 * Split a Latin word (>= 6 letters) into syllable segments keeping >= 2 chars per side
 * (`HYPHENATION_KEEPS = 2`). Strictly skips CJK, numbers, and symbol-containing words.
 */
export function hyphenateLatinWord(word) {
  if (!word || word.length < HYPHENATION_MIN_WORD_LEN) return [word];
  if (CJK_OR_DIGIT_RE.test(word)) return [word];

  const match = EDGE_STRIP_RE.exec(word);
  if (!match) return [word];

  const [, leadPunct, core, trailPunct] = match;
  if (core.length < HYPHENATION_MIN_WORD_LEN || !LATIN_WORD_RE.test(core)) {
    return [word];
  }

  const lower = core.toLowerCase();
  const len = core.length;
  const splitPoints = new Set();

  const isVowel = (idx) => idx >= 0 && idx < len && VOWEL_RE.test(lower[idx]);
  const canSplitAt = (pos) => pos >= HYPHENATION_KEEPS && len - pos >= HYPHENATION_KEEPS;

  // Common English & Indonesian prefixes/suffixes
  const prefixes = ['meng', 'meny', 'ber', 'ter', 'per', 'mem', 'men', 'pem', 'pen', 'dis', 'pre', 'pro', 'sub', 'non', 'mis', 'out', 'over', 'un', 're', 'in', 'im'];
  for (let i = 0; i < prefixes.length; i += 1) {
    const p = prefixes[i];
    if (lower.startsWith(p) && canSplitAt(p.length)) {
      splitPoints.add(p.length);
      break;
    }
  }

  const suffixes = ['tion', 'sion', 'ment', 'ness', 'able', 'ible', 'less', 'full', 'ing', 'kan', 'nya', 'lah', 'kah', 'pun'];
  for (let i = 0; i < suffixes.length; i += 1) {
    const s = suffixes[i];
    const pos = len - s.length;
    if (lower.endsWith(s) && canSplitAt(pos)) {
      splitPoints.add(pos);
      break;
    }
  }

  // Vowel-consonant syllable boundary rules (V-CV and VC-CV)
  for (let i = 1; i < len - 1; i += 1) {
    if (!isVowel(i - 1)) continue;
    // Count consecutive consonants starting at i
    let cEnd = i;
    while (cEnd < len && !isVowel(cEnd)) {
      cEnd += 1;
    }
    const cCount = cEnd - i;
    if (cCount === 0 || cEnd >= len) continue;

    if (cCount === 1) {
      // V-CV split before the single consonant
      if (canSplitAt(i)) splitPoints.add(i);
    } else if (cCount === 2) {
      const pair = lower.slice(i, i + 2);
      if (INSEPARABLE_DIGRAPHS.has(pair)) {
        if (canSplitAt(i)) splitPoints.add(i);
      } else if (canSplitAt(i + 1)) {
        splitPoints.add(i + 1);
      }
    } else {
      // 3+ consonants: keep inseparable final digraph together if present
      const tailPair = lower.slice(cEnd - 2, cEnd);
      const splitPos = INSEPARABLE_DIGRAPHS.has(tailPair) ? cEnd - 2 : i + 1;
      if (canSplitAt(splitPos)) splitPoints.add(splitPos);
    }
  }

  const sorted = Array.from(splitPoints).sort((a, b) => a - b);
  if (sorted.length === 0) return [word];

  // Filter out splits that would create tiny 1-char interior fragments
  const cleanSplits = [];
  let prev = 0;
  for (let i = 0; i < sorted.length; i += 1) {
    const pos = sorted[i];
    if (pos - prev >= HYPHENATION_KEEPS && len - pos >= HYPHENATION_KEEPS) {
      cleanSplits.push(pos);
      prev = pos;
    }
  }
  if (cleanSplits.length === 0) return [word];

  const parts = [];
  let start = 0;
  for (let i = 0; i < cleanSplits.length; i += 1) {
    parts.push(core.slice(start, cleanSplits[i]));
    start = cleanSplits[i];
  }
  parts.push(core.slice(start));

  parts[0] = leadPunct + parts[0];
  parts[parts.length - 1] = parts[parts.length - 1] + trailPunct;
  return parts;
}

/**
 * Tokenize text into DP segments.
 * Preserves weak-script symbols attached to their neighbor token so `♡`, `♪`, `…`
 * inherit neighbor layout cleanly and never strand as orphans.
 */
function tokenizeForDP(text, enableHyphenation) {
  const normalized = String(text || '').replace(/\r\n?/g, '\n').trim();
  if (!normalized) return [];

  const paragraphs = normalized.split('\n');
  const segments = [];

  for (let p = 0; p < paragraphs.length; p += 1) {
    const rawWords = paragraphs[p].trim().split(/\s+/).filter(Boolean);
    if (rawWords.length === 0) continue;

    // Glue standalone weak-script symbols (e.g. "hello ♡" -> "hello ♡") onto previous word
    const words = [];
    for (let w = 0; w < rawWords.length; w += 1) {
      const token = rawWords[w];
      if (words.length > 0 && WEAK_SCRIPT_ONLY_RE.test(token)) {
        words[words.length - 1] = `${words[words.length - 1]} ${token}`;
      } else {
        words.push(token);
      }
    }

    for (let w = 0; w < words.length; w += 1) {
      const word = words[w];
      const isLastInPara = w === words.length - 1;
      const isHardBreak = isLastInPara && p < paragraphs.length - 1;
      const parts = enableHyphenation ? hyphenateLatinWord(word) : [word];

      for (let k = 0; k < parts.length; k += 1) {
        const isWordEnd = k === parts.length - 1;
        const partText = parts[k];
        segments.push({
          text: partText,
          spaceBefore: w > 0 && k === 0,
          isWordEnd,
          isHardBreak: isWordEnd && isHardBreak,
          endsWithPunctuation: isWordEnd && PUNCT_END_RE.test(partText),
        });
      }
    }
  }

  return segments;
}

/**
 * Reconstruct a single line string from segments `[start .. end - 1]`.
 * Appends `"-"` if the ending segment is a mid-word hyphenation break.
 */
function buildLineString(segments, start, end) {
  let out = '';
  for (let i = start; i < end; i += 1) {
    const seg = segments[i];
    if (i > start && seg.spaceBefore) {
      out += ' ';
    }
    out += seg.text;
  }
  if (end > start && !segments[end - 1].isWordEnd) {
    out += '-';
  }
  return out;
}

// ============================================================================
// Dynamic Programming Line Breaker (L = 1..64, state: (line, seg_end, hrun < 3))
// ============================================================================

/**
 * Run the DP line breaker for a fixed fontSize & inset, sweeping line counts `1..64`.
 */
function solveDPLayoutAtScale({
  segments,
  poly,
  bounds,
  pole = null,
  fontSize,
  lineHeight,
  fontFamily,
  fontStyle,
  align,
  inset,
  measureFn,
}) {
  const numSegs = segments.length;
  if (numSegs === 0) {
    return {
      overflow: false,
      cost: 0,
      fontSize,
      pitch: fontSize * lineHeight,
      roomAxis: bounds.centerX,
      lines: [],
    };
  }

  const pitch = fontSize * lineHeight;
  const air = Math.max(0.75, Math.min(fontSize * 0.12, inset * 0.35));
  const vInset = Math.max(0, Math.min(inset * 0.5, bounds.height * 0.12));
  const usableHeight = Math.max(0, bounds.height - 2 * vInset);

  // Count mandatory hard line breaks
  let minLines = 1;
  for (let i = 0; i < numSegs - 1; i += 1) {
    if (segments[i].isHardBreak) minLines += 1;
  }

  const maxVertLines = Math.floor((usableHeight + 0.5) / pitch);
  const maxL = Math.min(MAX_DP_LINES, numSegs, Math.max(minLines, maxVertLines));

  // Precompute line string & width lookup for (start, end) pairs when needed
  const lineTextCache = new Map();
  const getLineMetrics = (start, end) => {
    const key = start * 1024 + end;
    let entry = lineTextCache.get(key);
    if (entry === undefined) {
      const text = buildLineString(segments, start, end);
      const width = measureFn(text);
      entry = { text, width };
      lineTextCache.set(key, entry);
    }
    return entry;
  };

  // Precompute whether [start..end-1] crosses an internal hard break (forbidden on one line)
  const hasInternalHardBreak = (start, end) => {
    for (let i = start; i < end - 1; i += 1) {
      if (segments[i].isHardBreak) return true;
    }
    return false;
  };

  let bestFeasible = null;
  let bestFallback = null;

  const startL = Math.max(1, minLines);
  const endL = Math.max(startL, maxL);

  for (let L = startL; L <= endL; L += 1) {
    const totalTextH = L * pitch;
    const verticalOverflow = totalTextH > usableHeight + 0.5;
    // #6 Polylabel-seeded vertical band centering for non-convex / spiky / cloud bubbles
    const anchorCenterY =
      pole &&
      pole.radius > 0 &&
      pole.y - totalTextH * 0.5 >= bounds.minY + vInset - 2 &&
      pole.y + totalTextH * 0.5 <= bounds.maxY - vInset + 2
        ? pole.y
        : bounds.minY + bounds.height * 0.5;
    const vStart = Math.max(
      bounds.minY,
      Math.min(bounds.maxY - totalTextH, anchorCenterY - totalTextH * 0.5)
    );

    // Sample all L bands against polygon
    const bands = new Array(L);
    for (let li = 0; li < L; li += 1) {
      const v0 = vStart + li * pitch;
      bands[li] = samplePolygonBand(poly, bounds, v0, pitch, air, inset);
    }

    // Lock single vertical center axis
    const { roomAxis, bandWidths } = computeRoomAxis(
      bands,
      pole && pole.radius > 0 ? pole.x : bounds.centerX,
      pole && pole.radius > 0 ? pole.x : null
    );

    // Choose effective room width per band based on alignment
    const effectiveWidths =
      align === 'center'
        ? bandWidths
        : bands.map((b) => b.width);

    // DP state: dp[lineIdx][segEnd][hRun] where hRun in {0, 1, 2}
    const strideSeg = MAX_CONSECUTIVE_HYPHENS;
    const strideLine = (numSegs + 1) * strideSeg;
    const totalStates = (L + 1) * strideLine;

    const dpCost = new Float64Array(totalStates);
    const dpPrevSeg = new Int32Array(totalStates);
    const dpPrevH = new Int32Array(totalStates);
    dpCost.fill(Infinity);
    dpPrevSeg.fill(-1);
    dpPrevH.fill(-1);

    dpCost[0] = 0.0; // (line=0, seg=0, hRun=0)

    for (let li = 0; li < L; li += 1) {
      const room = effectiveWidths[li];
      const baseCur = li * strideLine;
      const baseNext = (li + 1) * strideLine;

      // Remaining lines after this one: L - 1 - li
      const minSegStart = li;
      const maxSegStart = numSegs - (L - li);

      for (let prevEnd = minSegStart; prevEnd <= maxSegStart; prevEnd += 1) {
        for (let prevH = 0; prevH < MAX_CONSECUTIVE_HYPHENS; prevH += 1) {
          const prevCost = dpCost[baseCur + prevEnd * strideSeg + prevH];
          if (!Number.isFinite(prevCost)) continue;

          // If we are already worse than bestFeasible and no feasible path can beat it, prune
          if (bestFeasible && prevCost >= bestFeasible.cost) continue;

          const maxNextEnd = li === L - 1 ? numSegs : numSegs - (L - 1 - li);
          const minNextEnd = li === L - 1 ? numSegs : prevEnd + 1;

          for (let segEnd = minNextEnd; segEnd <= maxNextEnd; segEnd += 1) {
            if (hasInternalHardBreak(prevEnd, segEnd)) break;

            const lastSeg = segments[segEnd - 1];
            const isHyphenBreak = !lastSeg.isWordEnd;
            const nextH = isHyphenBreak ? prevH + 1 : 0;
            if (nextH >= MAX_CONSECUTIVE_HYPHENS) continue;

            const { width: used } = getLineMetrics(prevEnd, segEnd);

            let linePenalty = 0.0;
            if (room <= EPS || used > room + 0.25) {
              const excess = Math.max(0, used - Math.max(0, room));
              linePenalty += OVERFLOW_PENALTY + excess * 1e3;
            } else {
              const slackRatio = (room - used) / room;
              linePenalty += SLACK_PENALTY_WEIGHT * slackRatio * slackRatio;
            }

            // Break penalty
            if (segEnd === numSegs || lastSeg.isHardBreak || lastSeg.endsWithPunctuation) {
              linePenalty += BREAK_PENALTY_PUNCT_OR_END;
            } else if (isHyphenBreak) {
              linePenalty += BREAK_PENALTY_HYPHEN;
            } else {
              linePenalty += BREAK_PENALTY_MID_PHRASE;
            }

            const totalCost = prevCost + linePenalty;
            const nextIdx = baseNext + segEnd * strideSeg + nextH;
            if (totalCost < dpCost[nextIdx]) {
              dpCost[nextIdx] = totalCost;
              dpPrevSeg[nextIdx] = prevEnd;
              dpPrevH[nextIdx] = prevH;
            }

            // Early exit inner loop: once a line already overflows significantly, adding more
            // segments to the same line only increases overflow (unless it's the last line)
            if (li < L - 1 && room > EPS && used > room * 1.15) {
              break;
            }
          }
        }
      }
    }

    // Find best terminal state at (L, numSegs, hRun)
    let bestH = -1;
    let bestTerminalCost = Infinity;
    const termBase = L * strideLine + numSegs * strideSeg;
    for (let h = 0; h < MAX_CONSECUTIVE_HYPHENS; h += 1) {
      const c = dpCost[termBase + h];
      if (c < bestTerminalCost) {
        bestTerminalCost = c;
        bestH = h;
      }
    }

    if (bestH < 0 || !Number.isFinite(bestTerminalCost)) continue;

    const totalCostWithVert =
      bestTerminalCost + (verticalOverflow ? OVERFLOW_PENALTY + (totalTextH - usableHeight) * 1e3 : 0);

    // Reconstruct lines
    const lineRanges = new Array(L);
    let curSeg = numSegs;
    let curH = bestH;
    for (let li = L; li >= 1; li -= 1) {
      const idx = li * strideLine + curSeg * strideSeg + curH;
      const pSeg = dpPrevSeg[idx];
      const pH = dpPrevH[idx];
      lineRanges[li - 1] = [pSeg, curSeg];
      curSeg = pSeg;
      curH = pH;
    }

    let hasOverflow = verticalOverflow;
    const lines = new Array(L);
    for (let li = 0; li < L; li += 1) {
      const [s, e] = lineRanges[li];
      const { text, width: measuredWidth } = getLineMetrics(s, e);
      const b = bands[li];
      const room = effectiveWidths[li];
      if (room <= EPS || measuredWidth > room + 0.25) {
        hasOverflow = true;
      }

      let x;
      if (align === 'center') {
        const half = Math.max(0, room * 0.5);
        x = roomAxis - half;
      } else if (align === 'right') {
        x = b.left;
      } else {
        x = b.left;
      }

      lines[li] = {
        text,
        x,
        y: b.top,
        width: Math.max(measuredWidth, room),
        roomWidth: room,
        measuredWidth,
        bandLeft: b.left,
        bandRight: b.right,
      };
    }

    const candidate = {
      overflow: hasOverflow,
      cost: totalCostWithVert,
      fontSize,
      pitch,
      roomAxis,
      lines,
    };

    if (!hasOverflow) {
      if (!bestFeasible || candidate.cost < bestFeasible.cost) {
        bestFeasible = candidate;
      }
    }
    if (!bestFallback || candidate.cost < bestFallback.cost) {
      bestFallback = candidate;
    }
  }

  return bestFeasible || bestFallback || {
    overflow: true,
    cost: OVERFLOW_PENALTY,
    fontSize,
    pitch,
    roomAxis: bounds.centerX,
    lines: [
      {
        text: buildLineString(segments, 0, segments.length),
        x: bounds.minX + inset,
        y: bounds.minY + (bounds.height - pitch) * 0.5,
        width: Math.max(1, bounds.width - 2 * inset),
        roomWidth: Math.max(1, bounds.width - 2 * inset),
        measuredWidth: measureFn(buildLineString(segments, 0, segments.length)),
        bandLeft: bounds.minX + inset,
        bandRight: bounds.maxX - inset,
      },
    ],
  };
}

// ============================================================================
// 7-Step Copyfitting Hierarchy (Ignacio + Koharu)
// ============================================================================

/**
 * Shrink scale from `1.0` down to `floorScale` using an 8-point sweep
 * followed by 8 binary-halving refinement steps.
 */
function shrinkWithSweepAndBinaryHalving(evalAtScale, floorScale = COPYFIT_FLOOR) {
  let loScale = null;
  let loResult = null;
  let hiScale = 1.0;

  // 8-point sweep from 1.0 down to floorScale
  for (let k = 1; k <= SWEEP_POINTS; k += 1) {
    const scale = 1.0 - (k / SWEEP_POINTS) * (1.0 - floorScale);
    const res = evalAtScale(scale);
    if (!res.overflow) {
      loScale = scale;
      loResult = res;
      break;
    }
    hiScale = scale;
  }

  if (!loResult) return null;

  // 8 binary-halving steps between loScale (fits) and hiScale (overflows)
  let low = loScale;
  let high = hiScale;
  let best = loResult;
  let bestScale = loScale;

  for (let step = 0; step < BINARY_HALVING_STEPS; step += 1) {
    const mid = (low + high) * 0.5;
    const res = evalAtScale(mid);
    if (!res.overflow) {
      low = mid;
      best = res;
      bestScale = mid;
    } else {
      high = mid;
    }
  }

  return { scale: bestScale, result: best };
}

// Bounded LRU memoization cache for solved layouts (protects 60fps canvas renders)
const MAX_SOLVER_CACHE = 256;
const SOLVER_CACHE = new Map();

function makePolygonSignature(poly, boxX, boxY) {
  if (!poly) return 'rect';
  let sig = '';
  for (let i = 0; i < poly.length; i += 1) {
    // Store relative to box origin so pure translation/drag hits the cache!
    const rx = Math.round((poly[i][0] - boxX) * 2);
    const ry = Math.round((poly[i][1] - boxY) * 2);
    sig += `${rx},${ry};`;
  }
  return sig;
}

/**
 * Main Narrow-Bubble Typesetting Solver entry point.
 *
 * @param {object} params
 * @param {string} params.text - Text to typeset
 * @param {Array<[number,number]>|null} [params.roomPolygon] - Optional detector room_polygon in page coords
 * @param {Array<number>|object|null} [params.safeBbox] - Fallback safe_bbox `[x1, y1, x2, y2]` or `{x, y, width, height}`
 * @param {{x:number,y:number,width:number,height:number}} [params.box] - Block display box for local coordinate offsets
 * @param {number} [params.baseFontSize=18] - Target font size at scale=1.0
 * @param {number} [params.minFontSize=10] - Hard minimum font size for emergency fallback
 * @param {number} [params.lineHeight=1.18]
 * @param {string} [params.fontFamily='sans-serif']
 * @param {string} [params.fontStyle='bold']
 * @param {'left'|'center'|'right'} [params.align='center']
 * @param {boolean} [params.enableHyphenation=true]
 * @param {number} [params.inset] - Horizontal margin inset in px (auto-derived if omitted)
 * @param {number} [params.strokeWidth=0] - Extra stroke padding awareness
 * @param {boolean} [params.lockFontSize=false] - When true (manualFontSize), solve at scale=1.0 only
 * @param {Function} [params.measureText] - Optional custom text width measurer (for tests)
 * @param {string} [params.fontRenderKey=''] - Cache-buster when FontFace loads
 */
export function solveNarrowBubbleLayout({
  text,
  roomPolygon = null,
  safeBbox = null,
  box = null,
  baseFontSize = 18,
  minFontSize = 10,
  lineHeight = 1.18,
  fontFamily = 'sans-serif',
  fontStyle = 'normal',
  align = 'center',
  enableHyphenation = true,
  inset = null,
  strokeWidth = 0,
  lockFontSize = false,
  measureText = null,
  fontRenderKey = '',
} = {}) {
  const cleanText = String(text || '').trim();
  const normalizedPoly = normalizePolygon(roomPolygon);
  const isPolygonRoom = Boolean(normalizedPoly);
  const fallbackPoly = rectToPolygon(safeBbox) || rectToPolygon(box) || [
    [0, 0],
    [140, 0],
    [140, 60],
    [0, 60],
  ];
  const poly = normalizedPoly || fallbackPoly;
  const bounds = getPolygonBounds(poly);
  const pole = isPolygonRoom ? polylabel(poly, 1.0) : null;

  const originX = box ? Number(box.x) || 0 : bounds.minX;
  const originY = box ? Number(box.y) || 0 : bounds.minY;
  const boxW = box ? Math.max(1, Number(box.width) || bounds.width) : bounds.width;
  const boxH = box ? Math.max(1, Number(box.height) || bounds.height) : bounds.height;

  const targetFontSize = Math.max(8, Number(baseFontSize) || 18);
  const hardMinFont = Math.max(7, Math.min(targetFontSize, Number(minFontSize) || 10));
  const resolvedLineHeight = Math.max(0.95, Number(lineHeight) || 1.18);
  const strokePad = Math.max(0, Number(strokeWidth) || 0);
  const baseInset =
    typeof inset === 'number' && Number.isFinite(inset)
      ? Math.max(0, inset)
      : Math.max(2, Math.min(bounds.width * 0.08, 4 + strokePad * 0.5));

  if (!cleanText) {
    return {
      fontSize: targetFontSize,
      scale: 1.0,
      lineHeight: resolvedLineHeight,
      pitch: targetFontSize * resolvedLineHeight,
      roomAxis: bounds.centerX,
      localAxisX: bounds.centerX - originX,
      lines: [],
      joinedText: '',
      overflow: false,
      stepUsed: 0,
      usedHalvedMargin: false,
      usedHyphenation: false,
    };
  }

  // Translation-invariant LRU cache lookup (skipped if custom measureText provided)
  const useCache = typeof measureText !== 'function';
  let cacheKey = null;
  if (useCache) {
    const polySig = makePolygonSignature(poly, originX, originY);
    cacheKey = [
      cleanText,
      Math.round(boxW * 2),
      Math.round(boxH * 2),
      polySig,
      targetFontSize,
      hardMinFont,
      resolvedLineHeight.toFixed(3),
      fontFamily,
      fontStyle,
      align,
      enableHyphenation ? 1 : 0,
      baseInset.toFixed(2),
      lockFontSize ? 1 : 0,
      fontRenderKey,
    ].join('|');

    const cached = SOLVER_CACHE.get(cacheKey);
    if (cached) {
      // Refresh LRU & re-project page-space coords from local coords (handles drag with 0 solve cost)
      SOLVER_CACHE.delete(cacheKey);
      SOLVER_CACHE.set(cacheKey, cached);
      return {
        ...cached,
        roomAxis: originX + cached.localAxisX,
        lines: cached.lines.map((l) => ({
          ...l,
          x: originX + l.localX,
          y: originY + l.localY,
          bandLeft: originX + l.localBandLeft,
          bandRight: originX + l.localBandRight,
        })),
      };
    }
  }

  const wholeWordSegments = tokenizeForDP(cleanText, false);
  let hyphenSegments = null;
  const getHyphenSegments = () => {
    if (!hyphenSegments) {
      hyphenSegments = tokenizeForDP(cleanText, true);
    }
    return hyphenSegments;
  };

  const evalConfig = (scale, segs, curInset) => {
    const fs = Math.max(6, Math.round(targetFontSize * scale * 10) / 10);
    const measurer = createFontMeasurer(fs, fontFamily, fontStyle, measureText);
    return solveDPLayoutAtScale({
      segments: segs,
      poly,
      bounds,
      pole,
      fontSize: fs,
      lineHeight: resolvedLineHeight,
      fontFamily,
      fontStyle,
      align,
      inset: curInset,
      measureFn: measurer,
    });
  };

  let chosen = null;
  let chosenScale = 1.0;
  let stepUsed = 1;
  let usedHalvedMargin = false;
  let usedHyphenation = false;

  // Manual locked font size: solve directly at scale = 1.0
  if (lockFontSize) {
    chosen = evalConfig(1.0, wholeWordSegments, baseInset);
    chosenScale = 1.0;
    stepUsed = 1;
  } else {
    // ── Step 1: scale = 1.0 with whole words only ────────────────────────────
    const step1 = evalConfig(1.0, wholeWordSegments, baseInset);
    if (!step1.overflow) {
      chosen = step1;
      chosenScale = 1.0;
      stepUsed = 1;
    }

    // ── Step 2: Shrink whole words down to FLOOR = 0.5 (8-pt sweep + 8 binary)
    if (!chosen) {
      const step2 = shrinkWithSweepAndBinaryHalving(
        (s) => evalConfig(s, wholeWordSegments, baseInset),
        COPYFIT_FLOOR
      );
      if (step2) {
        chosen = step2.result;
        chosenScale = step2.scale;
        stepUsed = 2;
      }
    }

    // ── Step 3 & 4: Inside polygon room, halve margin (inset / 2.0) before hyphenating
    const halvedInset = isPolygonRoom ? baseInset * 0.5 : baseInset;
    if (!chosen && isPolygonRoom) {
      const step3 = evalConfig(1.0, wholeWordSegments, halvedInset);
      if (!step3.overflow) {
        chosen = step3;
        chosenScale = 1.0;
        stepUsed = 3;
        usedHalvedMargin = true;
      } else {
        const step4 = shrinkWithSweepAndBinaryHalving(
          (s) => evalConfig(s, wholeWordSegments, halvedInset),
          COPYFIT_FLOOR
        );
        if (step4) {
          chosen = step4.result;
          chosenScale = step4.scale;
          stepUsed = 4;
          usedHalvedMargin = true;
        }
      }
    }

    // ── Step 5 & 6: Enable Latin syllable hyphenation (>= 6 letters, keeps >= 2)
    if (!chosen && enableHyphenation) {
      const hSegs = getHyphenSegments();
      const hasAnySplit = hSegs.length > wholeWordSegments.length;
      if (hasAnySplit) {
        const step5 = evalConfig(1.0, hSegs, halvedInset);
        if (!step5.overflow) {
          chosen = step5;
          chosenScale = 1.0;
          stepUsed = 5;
          usedHalvedMargin = isPolygonRoom;
          usedHyphenation = true;
        } else {
          const step6 = shrinkWithSweepAndBinaryHalving(
            (s) => evalConfig(s, hSegs, halvedInset),
            COPYFIT_FLOOR
          );
          if (step6) {
            chosen = step6.result;
            chosenScale = step6.scale;
            stepUsed = 6;
            usedHalvedMargin = isPolygonRoom;
            usedHyphenation = true;
          }
        }
      }
    }

    // ── Step 7: Emergency below-floor shrink fallback (never spill outside ink walls)
    if (!chosen) {
      const segs = enableHyphenation ? getHyphenSegments() : wholeWordSegments;
      const minScale = Math.min(COPYFIT_FLOOR, Math.max(0.25, hardMinFont / targetFontSize));
      const emergency =
        minScale < COPYFIT_FLOOR
          ? shrinkWithSweepAndBinaryHalving(
              (s) => evalConfig(s, segs, halvedInset),
              minScale
            )
          : null;
      if (emergency) {
        chosen = emergency.result;
        chosenScale = emergency.scale;
      } else {
        chosen = evalConfig(minScale, segs, halvedInset);
        chosenScale = minScale;
      }
      stepUsed = 7;
      usedHalvedMargin = isPolygonRoom;
      usedHyphenation = Boolean(enableHyphenation && segs.length > wholeWordSegments.length);
    }
  }

  const finalFontSize = Math.max(6, Math.round(chosen.fontSize));
  const finalLines = chosen.lines.map((l) => ({
    text: l.text,
    x: l.x,
    y: l.y,
    localX: l.x - originX,
    localY: l.y - originY,
    width: l.width,
    roomWidth: l.roomWidth,
    measuredWidth: l.measuredWidth,
    bandLeft: l.bandLeft,
    bandRight: l.bandRight,
    localBandLeft: l.bandLeft - originX,
    localBandRight: l.bandRight - originX,
  }));

  const result = {
    fontSize: finalFontSize,
    scale: chosenScale,
    lineHeight: resolvedLineHeight,
    pitch: finalFontSize * resolvedLineHeight,
    roomAxis: chosen.roomAxis,
    localAxisX: chosen.roomAxis - originX,
    lines: finalLines,
    joinedText: finalLines.map((l) => l.text).join('\n'),
    overflow: Boolean(chosen.overflow),
    stepUsed,
    usedHalvedMargin,
    usedHyphenation,
  };

  if (useCache && cacheKey) {
    if (SOLVER_CACHE.size >= MAX_SOLVER_CACHE) {
      const oldestKey = SOLVER_CACHE.keys().next().value;
      if (oldestKey !== undefined) SOLVER_CACHE.delete(oldestKey);
    }
    SOLVER_CACHE.set(cacheKey, result);
  }

  return result;
}

/**
 * Clear the solver memoization cache (e.g. for tests or font reloads).
 */
export function clearNarrowBubbleSolverCache() {
  SOLVER_CACHE.clear();
}
