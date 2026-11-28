import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  ChevronLeft,
  ChevronRight,
  Columns2,
  FileImage,
  ScrollText,
  X,
} from 'lucide-react';

const WEBTOON_MAX_WIDTH = 820;

function PagePreviewSurface({ page, maxHeightClass = 'max-h-[calc(100vh-88px)]', fullWidth = false }) {
  if (!page) return null;
  const imgSrc = page.inpaintedUrl || page.originalUrl;
  const w = Math.max(1, page.width || 800);
  const h = Math.max(1, page.height || 1200);
  const visibleBlocks = (page.blocks || []).filter(
    (b) => !b.ignored && String(b.translated_text || '').trim() !== ''
  );

  return (
    <div
      className={`relative select-none overflow-hidden rounded shadow-2xl bg-black ${
        fullWidth ? 'w-full' : ''
      }`}
      style={{ aspectRatio: `${w} / ${h}` }}
    >
      {imgSrc ? (
        <img
          src={imgSrc}
          alt={page.filename || 'Page preview'}
          className={`block w-full h-full object-contain ${maxHeightClass}`}
          draggable={false}
        />
      ) : (
        <div className="w-full h-full flex items-center justify-center text-xs text-slate-400">
          No image
        </div>
      )}

      {visibleBlocks.map((b) => {
        const leftPct = (b.x / w) * 100;
        const topPct = (b.y / h) * 100;
        const widthPct = (b.width / w) * 100;
        const heightPct = (b.height / h) * 100;
        const fontSizePx = Math.max(9, Math.min(28, (b.fontSize || 16) * 0.85));
        const textColor = b.textColor || '#000000';
        const strokeColor = b.strokeColor || '#ffffff';

        return (
          <div
            key={b.id}
            style={{
              position: 'absolute',
              left: `${leftPct}%`,
              top: `${topPct}%`,
              width: `${widthPct}%`,
              height: `${heightPct}%`,
              display: 'flex',
              alignItems: 'center',
              justifyContent:
                b.align === 'left'
                  ? 'flex-start'
                  : b.align === 'right'
                    ? 'flex-end'
                    : 'center',
              textAlign: b.align || 'center',
              color: textColor,
              fontFamily: b.fontFamily || 'sans-serif',
              fontSize: `${fontSizePx}px`,
              lineHeight: b.lineHeight || 1.18,
              fontWeight: 600,
              padding: '2px 3px',
              overflow: 'hidden',
              wordBreak: 'break-word',
              textShadow: `-1px -1px 0 ${strokeColor}, 1px -1px 0 ${strokeColor}, -1px 1px 0 ${strokeColor}, 1px 1px 0 ${strokeColor}`,
              transform: b.rotation ? `rotate(${b.rotation}deg)` : undefined,
              pointerEvents: 'none',
            }}
          >
            <span>{b.translated_text}</span>
          </div>
        );
      })}
    </div>
  );
}

/**
 * Distraction-free reading preview modal (`F11` or top-bar Preview button).
 * Supports Single Page, Dual Spread (RTL/LTR), and Virtualized Webtoon Vertical Scroll.
 */
export default function ReadingPreviewModal({
  isOpen,
  onClose,
  pages = [],
  activePageId,
  onSelectPage,
  readingDirection = 'rtl',
}) {
  const [mode, setMode] = useState('single'); // 'single' | 'spread' | 'webtoon'
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(800);
  const webtoonContainerRef = useRef(null);

  const activeIndex = useMemo(() => {
    const idx = pages.findIndex((p) => p.id === activePageId);
    return idx >= 0 ? idx : 0;
  }, [pages, activePageId]);

  const currentPage = pages[activeIndex] || null;

  const goStep = (delta) => {
    if (pages.length === 0) return;
    const step = mode === 'spread' ? delta * 2 : delta;
    const nextIdx = Math.max(0, Math.min(pages.length - 1, activeIndex + step));
    const next = pages[nextIdx];
    if (next && onSelectPage) onSelectPage(next.id);
  };

  useEffect(() => {
    if (!isOpen) return;
    const onKeyDown = (e) => {
      if (e.key === 'Escape' || e.key === 'F11') {
        e.preventDefault();
        onClose();
        return;
      }
      if (mode === 'webtoon') return;
      const isRtl = readingDirection === 'rtl';
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        goStep(isRtl ? -1 : 1);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        goStep(isRtl ? 1 : -1);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  });

  useEffect(() => {
    if (!isOpen || mode !== 'webtoon') return;
    const el = webtoonContainerRef.current;
    if (!el) return;
    setViewportHeight(el.clientHeight || 800);
    const onResize = () => setViewportHeight(el.clientHeight || 800);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [isOpen, mode]);

  // Compute virtualized webtoon layout metrics
  const webtoonLayout = useMemo(() => {
    let y = 0;
    const items = pages.map((p, idx) => {
      const w = Math.max(1, p.width || 800);
      const h = Math.max(1, p.height || 1200);
      const scaledH = Math.round((h / w) * WEBTOON_MAX_WIDTH);
      const item = { page: p, idx, top: y, height: scaledH };
      y += scaledH + 12;
      return item;
    });
    return { items, totalHeight: y };
  }, [pages]);

  if (!isOpen) return null;

  const spreadPages = (() => {
    if (mode !== 'spread' || pages.length === 0) return [];
    const second = pages[activeIndex + 1];
    const pair = second ? [currentPage, second] : [currentPage];
    return readingDirection === 'rtl' ? [...pair].reverse() : pair;
  })();

  const visibleWebtoonItems =
    mode === 'webtoon'
      ? webtoonLayout.items.filter(
          (item) =>
            item.top + item.height >= scrollTop - viewportHeight * 0.8 &&
            item.top <= scrollTop + viewportHeight * 1.8
        )
      : [];

  return (
    <div
      className="fixed inset-0 z-50 bg-neutral-950/95 text-slate-100 flex flex-col"
      role="dialog"
      aria-modal="true"
      aria-label="Reading Preview"
    >
      {/* Floating compact top bar */}
      <div className="h-12 shrink-0 px-4 flex items-center justify-between border-b border-white/10 bg-neutral-950/80 backdrop-blur">
        <div className="flex items-center gap-2 text-xs text-slate-300">
          <span className="font-semibold text-white">Reading Preview</span>
          <span className="text-slate-500">|</span>
          <span>
            {pages.length > 0 ? `Page ${activeIndex + 1} / ${pages.length}` : 'No pages'}
          </span>
          <span className="hidden sm:inline text-slate-500 uppercase text-[10px]">
            ({readingDirection})
          </span>
        </div>

        <div className="flex items-center gap-1 bg-white/5 p-1 rounded-lg border border-white/10">
          <button
            type="button"
            onClick={() => setMode('single')}
            className={`px-2.5 py-1 rounded text-xs flex items-center gap-1.5 transition-colors ${
              mode === 'single'
                ? 'bg-sky-600 text-white'
                : 'text-slate-300 hover:bg-white/10'
            }`}
          >
            <FileImage className="w-3.5 h-3.5" />
            <span>Single</span>
          </button>
          <button
            type="button"
            onClick={() => setMode('spread')}
            className={`px-2.5 py-1 rounded text-xs flex items-center gap-1.5 transition-colors ${
              mode === 'spread'
                ? 'bg-sky-600 text-white'
                : 'text-slate-300 hover:bg-white/10'
            }`}
          >
            <Columns2 className="w-3.5 h-3.5" />
            <span>Spread</span>
          </button>
          <button
            type="button"
            onClick={() => setMode('webtoon')}
            className={`px-2.5 py-1 rounded text-xs flex items-center gap-1.5 transition-colors ${
              mode === 'webtoon'
                ? 'bg-sky-600 text-white'
                : 'text-slate-300 hover:bg-white/10'
            }`}
          >
            <ScrollText className="w-3.5 h-3.5" />
            <span>Webtoon</span>
          </button>
        </div>

        <button
          type="button"
          onClick={onClose}
          className="px-2.5 py-1 rounded-md text-xs text-slate-300 hover:text-white hover:bg-white/10 flex items-center gap-1"
          title="Exit Preview (Esc / F11)"
        >
          <X className="w-4 h-4" />
          <span className="hidden sm:inline">Esc</span>
        </button>
      </div>

      {/* Main reading viewport */}
      {mode === 'webtoon' ? (
        <div
          ref={webtoonContainerRef}
          onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
          className="flex-1 overflow-y-auto px-4 py-4"
        >
          <div
            className="mx-auto relative"
            style={{
              maxWidth: `${WEBTOON_MAX_WIDTH}px`,
              height: `${webtoonLayout.totalHeight}px`,
            }}
          >
            {visibleWebtoonItems.map((item) => (
              <div
                key={item.page.id}
                style={{
                  position: 'absolute',
                  top: `${item.top}px`,
                  left: 0,
                  right: 0,
                  height: `${item.height}px`,
                }}
              >
                <PagePreviewSurface
                  page={item.page}
                  fullWidth
                  maxHeightClass="max-h-none"
                />
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="flex-1 relative flex items-center justify-center p-4 overflow-hidden">
          <button
            type="button"
            onClick={() => goStep(readingDirection === 'rtl' ? 1 : -1)}
            className="absolute left-3 z-10 p-2 rounded-full bg-black/50 hover:bg-black/80 text-white border border-white/10"
            title="Previous"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>

          <div
            className={`flex items-center justify-center gap-3 max-w-full max-h-full ${
              mode === 'spread' ? 'w-full' : ''
            }`}
          >
            {mode === 'single' ? (
              <div className="h-[calc(100vh-88px)] flex items-center justify-center">
                <PagePreviewSurface page={currentPage} />
              </div>
            ) : (
              spreadPages.map((p) => (
                <div
                  key={p.id}
                  className="h-[calc(100vh-88px)] flex items-center justify-center"
                >
                  <PagePreviewSurface page={p} />
                </div>
              ))
            )}
          </div>

          <button
            type="button"
            onClick={() => goStep(readingDirection === 'rtl' ? -1 : 1)}
            className="absolute right-3 z-10 p-2 rounded-full bg-black/50 hover:bg-black/80 text-white border border-white/10"
            title="Next"
          >
            <ChevronRight className="w-5 h-5" />
          </button>
        </div>
      )}
    </div>
  );
}
