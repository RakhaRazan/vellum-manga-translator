import React, { useEffect, useRef } from 'react';

const LOUPE_SIZE = 220;
const HALF_SIZE = LOUPE_SIZE / 2;
const MIN_LOUPE_ZOOM = 2;
const MAX_LOUPE_ZOOM = 5;

function isEditableTarget(target) {
  if (!target) return false;
  const tag = target.tagName;
  return (
    tag === 'INPUT' ||
    tag === 'TEXTAREA' ||
    tag === 'SELECT' ||
    Boolean(target.isContentEditable)
  );
}

/**
 * 60fps circular magnifier activated while holding the `Z` key.
 * Samples the live Konva stage canvases via requestAnimationFrame with zero React state updates.
 */
export default function LoupeOverlay({ stageRef, containerRef }) {
  const wrapperRef = useRef(null);
  const canvasRef = useRef(null);
  const stateRef = useRef({
    active: false,
    zoom: 3,
    clientX: 0,
    clientY: 0,
    hasPointer: false,
    rafId: null,
  });

  useEffect(() => {
    const drawFrame = () => {
      const st = stateRef.current;
      st.rafId = null;
      const wrapper = wrapperRef.current;
      const outCanvas = canvasRef.current;
      const container = containerRef?.current;
      const stage = stageRef?.current;

      if (!st.active || !st.hasPointer || !wrapper || !outCanvas || !container || !stage) {
        if (wrapper) wrapper.style.display = 'none';
        return;
      }

      const rect = container.getBoundingClientRect();
      const localX = st.clientX - rect.left;
      const localY = st.clientY - rect.top;

      if (localX < 0 || localY < 0 || localX > rect.width || localY > rect.height) {
        wrapper.style.display = 'none';
        return;
      }

      wrapper.style.display = 'block';
      wrapper.style.transform = `translate3d(${Math.round(localX - HALF_SIZE)}px, ${Math.round(localY - HALF_SIZE)}px, 0)`;

      const dpr = window.devicePixelRatio || 1;
      const targetPx = Math.round(LOUPE_SIZE * dpr);
      if (outCanvas.width !== targetPx || outCanvas.height !== targetPx) {
        outCanvas.width = targetPx;
        outCanvas.height = targetPx;
      }

      const ctx = outCanvas.getContext('2d');
      if (!ctx) return;

      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, targetPx, targetPx);

      ctx.beginPath();
      ctx.arc(targetPx / 2, targetPx / 2, targetPx / 2, 0, Math.PI * 2);
      ctx.clip();

      ctx.fillStyle = '#0f1115';
      ctx.fillRect(0, 0, targetPx, targetPx);

      const stageContent = stage.content || stage.container?.();
      const canvases = stageContent ? stageContent.querySelectorAll('canvas') : [];
      const srcRadiusCss = HALF_SIZE / st.zoom;

      for (let i = 0; i < canvases.length; i += 1) {
        const src = canvases[i];
        if (!src || src.width === 0 || src.height === 0) continue;
        const scaleX = src.width / Math.max(1, rect.width);
        const scaleY = src.height / Math.max(1, rect.height);
        const sx = (localX - srcRadiusCss) * scaleX;
        const sy = (localY - srcRadiusCss) * scaleY;
        const sw = srcRadiusCss * 2 * scaleX;
        const sh = srcRadiusCss * 2 * scaleY;
        try {
          ctx.drawImage(src, sx, sy, sw, sh, 0, 0, targetPx, targetPx);
        } catch {
          // Ignore transient tainted/unready canvas frames
        }
      }

      // Subtle center crosshair
      const cx = targetPx / 2;
      const cy = targetPx / 2;
      const arm = 7 * dpr;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
      ctx.lineWidth = Math.max(1, dpr);
      ctx.beginPath();
      ctx.moveTo(cx - arm, cy);
      ctx.lineTo(cx + arm, cy);
      ctx.moveTo(cx, cy - arm);
      ctx.lineTo(cx, cy + arm);
      ctx.stroke();

      // Zoom level pill at bottom of loupe
      const label = `${st.zoom.toFixed(1).replace(/\.0$/, '')}x`;
      ctx.font = `600 ${Math.round(11 * dpr)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
      const tw = ctx.measureText(label).width;
      const pw = tw + 12 * dpr;
      const ph = 18 * dpr;
      const px = cx - pw / 2;
      const py = targetPx - ph - 10 * dpr;
      ctx.fillStyle = 'rgba(15, 17, 21, 0.82)';
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(px, py, pw, ph, 9 * dpr);
      } else {
        ctx.rect(px, py, pw, ph);
      }
      ctx.fill();
      ctx.fillStyle = '#f8fafc';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, cx, py + ph / 2);

      ctx.restore();
    };

    const scheduleDraw = () => {
      const st = stateRef.current;
      if (!st.rafId) {
        st.rafId = window.requestAnimationFrame(drawFrame);
      }
    };

    const handleKeyDown = (e) => {
      if (e.code !== 'KeyZ' || e.ctrlKey || e.metaKey || e.altKey) return;
      if (isEditableTarget(document.activeElement)) return;
      if (!stateRef.current.active) {
        stateRef.current.active = true;
        scheduleDraw();
      }
    };

    const handleKeyUp = (e) => {
      if (e.code !== 'KeyZ') return;
      stateRef.current.active = false;
      if (wrapperRef.current) wrapperRef.current.style.display = 'none';
    };

    const handleBlur = () => {
      stateRef.current.active = false;
      if (wrapperRef.current) wrapperRef.current.style.display = 'none';
    };

    const handlePointerMove = (e) => {
      const st = stateRef.current;
      st.clientX = e.clientX;
      st.clientY = e.clientY;
      st.hasPointer = true;
      if (st.active) scheduleDraw();
    };

    const handleWheel = (e) => {
      const st = stateRef.current;
      if (!st.active) return;
      e.preventDefault();
      e.stopPropagation();
      const delta = e.deltaY < 0 ? 0.5 : -0.5;
      st.zoom = Math.min(MAX_LOUPE_ZOOM, Math.max(MIN_LOUPE_ZOOM, st.zoom + delta));
      scheduleDraw();
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    window.addEventListener('blur', handleBlur);
    window.addEventListener('pointermove', handlePointerMove, { passive: true });
    window.addEventListener('wheel', handleWheel, { passive: false, capture: true });

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      window.removeEventListener('blur', handleBlur);
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('wheel', handleWheel, { capture: true });
      if (stateRef.current.rafId) {
        window.cancelAnimationFrame(stateRef.current.rafId);
      }
    };
  }, [stageRef, containerRef]);

  return (
    <div
      ref={wrapperRef}
      style={{
        display: 'none',
        width: `${LOUPE_SIZE}px`,
        height: `${LOUPE_SIZE}px`,
        position: 'absolute',
        top: 0,
        left: 0,
        pointerEvents: 'none',
        zIndex: 45,
        borderRadius: '9999px',
        boxShadow: '0 14px 36px rgba(0, 0, 0, 0.55), 0 0 0 2px rgba(255, 255, 255, 0.85)',
        overflow: 'hidden',
      }}
      aria-hidden="true"
    >
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: '100%', display: 'block' }}
      />
    </div>
  );
}
