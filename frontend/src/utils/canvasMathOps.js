/**
 * Vellum Math Wave M5 — Pure JS Frontend Computational Geometry, Assignment & Layout Ops
 *
 * Implements:
 *   #6  - Polylabel (Branch-and-Bound Quadtree Pole of Inaccessibility, precision = 1.0 px)
 *   #14 - Spring-Mass Anti-Overlap Projected Gradient Descent + Medial Axis Cubic Bezier Fit
 *   #19 - Exact O(n^3) Hungarian (Kuhn-Munkres) Assignment for Re-Detect Edit Preservation
 */

// ============================================================================
// #6 — Polylabel (Pole of Inaccessibility Quadtree)
// ============================================================================

function pointToSegmentDistSq(px, py, ax, ay, bx, by) {
  let x = ax;
  let y = ay;
  let dx = bx - ax;
  let dy = by - ay;
  if (dx !== 0 || dy !== 0) {
    const t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy);
    if (t > 1) {
      x = bx;
      y = by;
    } else if (t > 0) {
      x += dx * t;
      y += dy * t;
    }
  }
  dx = px - x;
  dy = py - y;
  return dx * dx + dy * dy;
}

function pointToPolygonDist(x, y, poly) {
  let inside = false;
  let minDistSq = Infinity;
  const n = poly.length;
  for (let i = 0, j = n - 1; i < n; j = i, i += 1) {
    const [ax, ay] = poly[i];
    const [bx, by] = poly[j];
    if ((ay > y) !== (by > y) && x < ((bx - ax) * (y - ay)) / (by - ay) + ax) {
      inside = !inside;
    }
    const dSq = pointToSegmentDistSq(x, y, ax, ay, bx, by);
    if (dSq < minDistSq) minDistSq = dSq;
  }
  const dist = Math.sqrt(minDistSq);
  return inside ? dist : -dist;
}

function getPolygonCentroid(poly) {
  let area = 0;
  let cx = 0;
  let cy = 0;
  const n = poly.length;
  for (let i = 0, j = n - 1; i < n; j = i, i += 1) {
    const [ax, ay] = poly[i];
    const [bx, by] = poly[j];
    const f = ax * by - bx * ay;
    cx += (ax + bx) * f;
    cy += (ay + by) * f;
    area += f * 3;
  }
  if (Math.abs(area) < 1e-6) {
    let sx = 0;
    let sy = 0;
    for (let i = 0; i < n; i += 1) {
      sx += poly[i][0];
      sy += poly[i][1];
    }
    return [sx / Math.max(1, n), sy / Math.max(1, n)];
  }
  return [cx / area, cy / area];
}

class MaxHeap {
  constructor() {
    this.data = [];
  }
  push(item) {
    this.data.push(item);
    let i = this.data.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.data[p].max >= this.data[i].max) break;
      const tmp = this.data[p];
      this.data[p] = this.data[i];
      this.data[i] = tmp;
      i = p;
    }
  }
  pop() {
    const top = this.data[0];
    const last = this.data.pop();
    if (this.data.length > 0) {
      this.data[0] = last;
      let i = 0;
      const len = this.data.length;
      while (true) {
        let best = i;
        const l = (i << 1) + 1;
        const r = l + 1;
        if (l < len && this.data[l].max > this.data[best].max) best = l;
        if (r < len && this.data[r].max > this.data[best].max) best = r;
        if (best === i) break;
        const tmp = this.data[i];
        this.data[i] = this.data[best];
        this.data[best] = tmp;
        i = best;
      }
    }
    return top;
  }
  get length() {
    return this.data.length;
  }
}

/**
 * Find Pole of Inaccessibility (x*, y*) and max inscribed radius r* inside a polygon.
 *
 * @param {Array<[number, number]>} poly
 * @param {number} [precision=1.0]
 * @returns {{ x: number, y: number, radius: number }}
 */
export function polylabel(poly, precision = 1.0) {
  if (!Array.isArray(poly) || poly.length < 3) {
    return { x: 0, y: 0, radius: 0 };
  }
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < poly.length; i += 1) {
    const [px, py] = poly[i];
    if (px < minX) minX = px;
    if (py < minY) minY = py;
    if (px > maxX) maxX = px;
    if (py > maxY) maxY = py;
  }

  const width = maxX - minX;
  const height = maxY - minY;
  const cellSize = Math.min(width, height);
  if (cellSize <= 1e-3) {
    return { x: (minX + maxX) * 0.5, y: (minY + maxY) * 0.5, radius: 0 };
  }

  let h = cellSize * 0.5;
  const makeCell = (x, y, half) => {
    const d = pointToPolygonDist(x, y, poly);
    return { x, y, h: half, d, max: d + half * Math.SQRT2 };
  };

  const heap = new MaxHeap();
  for (let x = minX; x < maxX; x += cellSize) {
    for (let y = minY; y < maxY; y += cellSize) {
      heap.push(makeCell(x + h, y + h, h));
    }
  }

  const [cx, cy] = getPolygonCentroid(poly);
  let bestCell = makeCell(cx, cy, 0);
  const bboxCell = makeCell(minX + width * 0.5, minY + height * 0.5, 0);
  if (bboxCell.d > bestCell.d) bestCell = bboxCell;

  let iters = 0;
  const maxIters = 1024;
  while (heap.length > 0 && iters < maxIters) {
    iters += 1;
    const cell = heap.pop();
    if (cell.d > bestCell.d) bestCell = cell;
    if (cell.max - bestCell.d <= precision) continue;

    h = cell.h * 0.5;
    if (h < precision * 0.25) continue;
    heap.push(makeCell(cell.x - h, cell.y - h, h));
    heap.push(makeCell(cell.x + h, cell.y - h, h));
    heap.push(makeCell(cell.x - h, cell.y + h, h));
    heap.push(makeCell(cell.x + h, cell.y + h, h));
  }

  return {
    x: bestCell.x,
    y: bestCell.y,
    radius: Math.max(0, bestCell.d),
  };
}

// ============================================================================
// #14A — Spring-Mass Anti-Overlap Layout (Projected Gradient Descent)
// ============================================================================

/**
 * Minimize E(x) = sum w_a ||x_i - x_i^(0)||^2 + sum_{i<j} w_o Area(B_i cap B_j)^2 + sum w_b OOB(B_i)^2
 * over 12 projected gradient descent steps.
 * Manually positioned boxes (`manualPosition === true` or `!_autoSized`) stay fixed as obstacles.
 */
export function minimizeBoxOverlapEnergy(boxes, pageWidth, pageHeight, opts = {}) {
  if (!Array.isArray(boxes) || boxes.length <= 1) return boxes;
  const W = Number(pageWidth) || 0;
  const H = Number(pageHeight) || 0;
  if (W <= 0 || H <= 0) return boxes;

  const steps = opts.steps ?? 12;
  const wAnchor = opts.wAnchor ?? 0.35;
  const wOverlap = opts.wOverlap ?? 1.0;
  const wBounds = opts.wBounds ?? 2.0;
  const pad = opts.pad ?? 4;
  const n = boxes.length;

  const x0 = new Float64Array(n);
  const y0 = new Float64Array(n);
  const cx = new Float64Array(n);
  const cy = new Float64Array(n);
  const bw = new Float64Array(n);
  const bh = new Float64Array(n);
  const movable = new Uint8Array(n);

  for (let i = 0; i < n; i += 1) {
    const b = boxes[i];
    const x = Number(b.x) || 0;
    const y = Number(b.y) || 0;
    const w = Math.max(16, Number(b.width) || 40);
    const h = Math.max(16, Number(b.height) || 24);
    x0[i] = x;
    y0[i] = y;
    cx[i] = x;
    cy[i] = y;
    bw[i] = w;
    bh[i] = h;
    movable[i] = b.manualPosition ? 0 : 1;
  }

  const gx = new Float64Array(n);
  const gy = new Float64Array(n);

  for (let step = 0; step < steps; step += 1) {
    gx.fill(0);
    gy.fill(0);

    for (let i = 0; i < n; i += 1) {
      if (!movable[i]) continue;
      // Anchor spring force
      gx[i] += 2 * wAnchor * (cx[i] - x0[i]);
      gy[i] += 2 * wAnchor * (cy[i] - y0[i]);

      // Page boundary penalty
      if (cx[i] < 0) gx[i] += 2 * wBounds * cx[i];
      if (cx[i] + bw[i] > W) gx[i] += 2 * wBounds * (cx[i] + bw[i] - W);
      if (cy[i] < 0) gy[i] += 2 * wBounds * cy[i];
      if (cy[i] + bh[i] > H) gy[i] += 2 * wBounds * (cy[i] + bh[i] - H);
    }

    // Pairwise overlap repulsion
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        if (!movable[i] && !movable[j]) continue;
        const ix = Math.min(cx[i] + bw[i] + pad, cx[j] + bw[j] + pad) - Math.max(cx[i], cx[j]);
        const iy = Math.min(cy[i] + bh[i] + pad, cy[j] + bh[j] + pad) - Math.max(cy[i], cy[j]);
        if (ix <= 0 || iy <= 0) continue;

        const cix = cx[i] + bw[i] * 0.5;
        const ciy = cy[i] + bh[i] * 0.5;
        const cjx = cx[j] + bw[j] * 0.5;
        const cjy = cy[j] + bh[j] * 0.5;

        // Push along minimum-penetration axis
        if (ix < iy) {
          const dir = cix <= cjx ? -1 : 1;
          const f = wOverlap * ix;
          if (movable[i]) gx[i] -= dir * f;
          if (movable[j]) gx[j] += dir * f;
        } else {
          const dir = ciy <= cjy ? -1 : 1;
          const f = wOverlap * iy;
          if (movable[i]) gy[i] -= dir * f;
          if (movable[j]) gy[j] += dir * f;
        }
      }
    }

    const lr = 0.28;
    for (let i = 0; i < n; i += 1) {
      if (!movable[i]) continue;
      cx[i] = Math.max(0, Math.min(W - bw[i], cx[i] - lr * gx[i]));
      cy[i] = Math.max(0, Math.min(H - bh[i], cy[i] - lr * gy[i]));
    }
  }

  for (let i = 0; i < n; i += 1) {
    if (movable[i]) {
      boxes[i].x = Math.round(cx[i]);
      boxes[i].y = Math.round(cy[i]);
    }
  }
  return boxes;
}

// ============================================================================
// #14B — Medial Axis + Least-Squares Cubic Bezier SFX Curve
// ============================================================================

/**
 * Fit a 4-point Cubic Bezier curve along the medial spine of a polygon (or box)
 * via closed-form least squares.
 *
 * @param {Array<[number, number]>|null} poly
 * @param {{ x: number, y: number, width: number, height: number }|null} [fallbackBox]
 * @returns {{ p0:[number,number], p1:[number,number], p2:[number,number], p3:[number,number], svgPath:string }}
 */
export function fitCubicBezierFromPolygon(poly, fallbackBox = null) {
  let pts = Array.isArray(poly) && poly.length >= 3 ? poly : null;
  if (!pts && fallbackBox) {
    const x = Number(fallbackBox.x) || 0;
    const y = Number(fallbackBox.y) || 0;
    const w = Math.max(10, Number(fallbackBox.width) || 100);
    const h = Math.max(10, Number(fallbackBox.height) || 40);
    pts = [
      [x, y],
      [x + w, y],
      [x + w, y + h],
      [x, y + h],
    ];
  }
  if (!pts) {
    return {
      p0: [0, 20],
      p1: [40, 20],
      p2: [80, 20],
      p3: [120, 20],
      svgPath: 'M 0 20 C 40 20, 80 20, 120 20',
    };
  }

  // PCA major axis of polygon vertices
  let mx = 0;
  let my = 0;
  for (let i = 0; i < pts.length; i += 1) {
    mx += pts[i][0];
    my += pts[i][1];
  }
  mx /= pts.length;
  my /= pts.length;

  let cxx = 0;
  let cyy = 0;
  let cxy = 0;
  for (let i = 0; i < pts.length; i += 1) {
    const dx = pts[i][0] - mx;
    const dy = pts[i][1] - my;
    cxx += dx * dx;
    cyy += dy * dy;
    cxy += dx * dy;
  }
  const theta = 0.5 * Math.atan2(2 * cxy, cxx - cyy);
  let ux = Math.cos(theta);
  let uy = Math.sin(theta);
  // Keep left-to-right orientation for readable text paths
  if (ux < 0 || (Math.abs(ux) < 1e-4 && uy < 0)) {
    ux = -ux;
    uy = -uy;
  }
  const vx = -uy;
  const vy = ux;

  // Project vertices onto (u, v) frame and bin into medial spine slices
  let minU = Infinity;
  let maxU = -Infinity;
  const proj = pts.map(([px, py]) => {
    const dx = px - mx;
    const dy = py - my;
    const u = dx * ux + dy * uy;
    const v = dx * vx + dy * vy;
    if (u < minU) minU = u;
    if (u > maxU) maxU = u;
    return [u, v];
  });

  const spanU = Math.max(1, maxU - minU);
  const K = 8;
  const spine = [];
  for (let k = 0; k < K; k += 1) {
    const uLo = minU + (k / K) * spanU;
    const uHi = minU + ((k + 1) / K) * spanU;
    const uMid = (uLo + uHi) * 0.5;
    let sumV = 0;
    let cnt = 0;
    for (let i = 0; i < proj.length; i += 1) {
      if (proj[i][0] >= uLo - 1e-4 && proj[i][0] <= uHi + 1e-4) {
        sumV += proj[i][1];
        cnt += 1;
      }
    }
    const vMid = cnt > 0 ? sumV / cnt : 0;
    spine.push({
      t: K > 1 ? k / (K - 1) : 0.5,
      x: mx + uMid * ux + vMid * vx,
      y: my + uMid * uy + vMid * vy,
    });
  }

  const p0 = [spine[0].x, spine[0].y];
  const p3 = [spine[spine.length - 1].x, spine[spine.length - 1].y];

  // Closed-form 2x2 least squares for interior control points P1, P2
  let a11 = 0;
  let a12 = 0;
  let a22 = 0;
  let rx1 = 0;
  let ry1 = 0;
  let rx2 = 0;
  let ry2 = 0;
  for (let i = 0; i < spine.length; i += 1) {
    const t = spine[i].t;
    const omt = 1 - t;
    const b0 = omt * omt * omt;
    const b1 = 3 * omt * omt * t;
    const b2 = 3 * omt * t * t;
    const b3 = t * t * t;
    const tx = spine[i].x - b0 * p0[0] - b3 * p3[0];
    const ty = spine[i].y - b0 * p0[1] - b3 * p3[1];
    a11 += b1 * b1;
    a12 += b1 * b2;
    a22 += b2 * b2;
    rx1 += b1 * tx;
    ry1 += b1 * ty;
    rx2 += b2 * tx;
    ry2 += b2 * ty;
  }

  const det = a11 * a22 - a12 * a12;
  let p1;
  let p2;
  if (Math.abs(det) > 1e-6) {
    p1 = [(rx1 * a22 - rx2 * a12) / det, (ry1 * a22 - ry2 * a12) / det];
    p2 = [(a11 * rx2 - a12 * rx1) / det, (a11 * ry2 - a12 * ry1) / det];
  } else {
    p1 = [p0[0] + (p3[0] - p0[0]) / 3, p0[1] + (p3[1] - p0[1]) / 3];
    p2 = [p0[0] + (2 * (p3[0] - p0[0])) / 3, p0[1] + (2 * (p3[1] - p0[1])) / 3];
  }

  const f = (v) => Math.round(v * 10) / 10;
  const svgPath = `M ${f(p0[0])} ${f(p0[1])} C ${f(p1[0])} ${f(p1[1])}, ${f(p2[0])} ${f(p2[1])}, ${f(p3[0])} ${f(p3[1])}`;
  return { p0, p1, p2, p3, svgPath };
}

// ============================================================================
// #19 — Exact O(n^3) Hungarian Algorithm for Re-Detect Edit Preservation
// ============================================================================

/**
 * Solve minimum-cost bipartite assignment using O(n^3) Kuhn-Munkres (Hungarian).
 * Rejects pairs whose cost exceeds `maxCost`.
 *
 * @param {number[][]} costMatrix - N x M matrix
 * @param {number} [maxCost=0.85]
 * @returns {Array<{ oldIdx: number, newIdx: number, cost: number }>}
 */
export function hungarianAssign(costMatrix, maxCost = 0.85) {
  const n = costMatrix.length;
  const m = n > 0 ? costMatrix[0].length : 0;
  if (n === 0 || m === 0) return [];

  const dim = Math.max(n, m);
  const INF = 1e9;
  const padCost = maxCost + 10;

  const u = new Float64Array(dim + 1);
  const v = new Float64Array(dim + 1);
  const p = new Int32Array(dim + 1);
  const way = new Int32Array(dim + 1);

  for (let i = 1; i <= dim; i += 1) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(dim + 1).fill(INF);
    const used = new Uint8Array(dim + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = INF;
      let j1 = 0;
      for (let j = 1; j <= dim; j += 1) {
        if (used[j]) continue;
        const c = i0 <= n && j <= m ? costMatrix[i0 - 1][j - 1] : padCost;
        const cur = c - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= dim; j += 1) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);

    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }

  const matches = [];
  for (let j = 1; j <= m; j += 1) {
    const i = p[j];
    if (i >= 1 && i <= n) {
      const c = costMatrix[i - 1][j - 1];
      if (c <= maxCost) {
        matches.push({ oldIdx: i - 1, newIdx: j - 1, cost: c });
      }
    }
  }
  return matches;
}

function getBlockBox(b) {
  if (Array.isArray(b?.bbox) && b.bbox.length >= 4) {
    const x1 = Number(b.bbox[0]) || 0;
    const y1 = Number(b.bbox[1]) || 0;
    const x2 = Number(b.bbox[2]) || x1 + 40;
    const y2 = Number(b.bbox[3]) || y1 + 24;
    return [x1, y1, Math.max(x1 + 1, x2), Math.max(y1 + 1, y2)];
  }
  const x1 = Number(b?.x) || 0;
  const y1 = Number(b?.y) || 0;
  const w = Math.max(1, Number(b?.width) || 40);
  const h = Math.max(1, Number(b?.height) || 24);
  return [x1, y1, x1 + w, y1 + h];
}

function boxIoU(a, b) {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  if (ix2 <= ix1 || iy2 <= iy1) return 0;
  const inter = (ix2 - ix1) * (iy2 - iy1);
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / Math.max(1e-6, areaA + areaB - inter);
}

/**
 * Preserve manual text/style/position edits across Re-Detect runs via Hungarian assignment.
 */
export function preserveBlockEditsWithHungarian(
  oldBlocks = [],
  newBlocks = [],
  imgWidth = 1000,
  imgHeight = 1400,
  maxCost = 0.85
) {
  if (!Array.isArray(oldBlocks) || oldBlocks.length === 0) return newBlocks;
  if (!Array.isArray(newBlocks) || newBlocks.length === 0) return newBlocks;

  const diag = Math.max(1, Math.hypot(Number(imgWidth) || 1000, Number(imgHeight) || 1400));
  const costMatrix = oldBlocks.map((oldB) => {
    const a = getBlockBox(oldB);
    const acx = (a[0] + a[2]) * 0.5;
    const acy = (a[1] + a[3]) * 0.5;
    return newBlocks.map((newB) => {
      const b = getBlockBox(newB);
      const bcx = (b[0] + b[2]) * 0.5;
      const bcy = (b[1] + b[3]) * 0.5;
      const iou = boxIoU(a, b);
      const dist = Math.hypot(acx - bcx, acy - bcy) / diag;
      return 1 - iou + 0.35 * dist;
    });
  });

  const matches = hungarianAssign(costMatrix, maxCost);
  if (matches.length === 0) return newBlocks;

  const out = [...newBlocks];
  for (const { oldIdx, newIdx } of matches) {
    const oldB = oldBlocks[oldIdx];
    const newB = out[newIdx];
    if (!oldB || !newB) continue;

    const merged = { ...newB };
    // Preserve translated text & OCR text if user had edited or translated it
    if (typeof oldB.translated_text === 'string' && oldB.translated_text.trim() !== '') {
      merged.translated_text = oldB.translated_text;
    }
    if (typeof oldB.original_text === 'string' && oldB.original_text.trim() !== '' && !newB.original_text) {
      merged.original_text = oldB.original_text;
      merged.source_text = oldB.source_text || oldB.original_text;
    }
    // Preserve custom typography overrides
    if (oldB.manualFontSize) {
      merged.manualFontSize = true;
      merged.font_size = oldB.font_size;
    }
    if (oldB.font_family) merged.font_family = oldB.font_family;
    if (oldB.font_style) merged.font_style = oldB.font_style;
    if (oldB.color) merged.color = oldB.color;
    if (oldB.stroke_color) merged.stroke_color = oldB.stroke_color;
    if (typeof oldB.stroke_width === 'number') merged.stroke_width = oldB.stroke_width;
    if (oldB.align) merged.align = oldB.align;
    if (typeof oldB.line_height === 'number') merged.line_height = oldB.line_height;
    if (oldB.curvedPath) merged.curvedPath = true;
    if (oldB.style && typeof oldB.style === 'object') {
      merged.style = { ...(newB.style || {}), ...oldB.style };
    }
    // Preserve manual position / nudge
    if (oldB.manualPosition) {
      merged.manualPosition = true;
      merged.x = oldB.x;
      merged.y = oldB.y;
      merged.width = oldB.width;
      merged.height = oldB.height;
      if (typeof oldB.rotation === 'number') merged.rotation = oldB.rotation;
    }
    out[newIdx] = merged;
  }
  return out;
}
