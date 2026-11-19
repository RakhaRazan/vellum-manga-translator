import React, { useEffect, useRef } from 'react';
import {
  AlignCenter,
  AlignLeft,
  AlignRight,
  CopyCheck,
  MessageSquare,
  Plus,
  RotateCcw,
  Sparkles,
  Trash2,
  Type,
} from 'lucide-react';

export const COMIC_FONT_FAMILIES = [
  {
    label: 'Comic Neue (Default Comic)',
    value: '"Comic Neue", "Anime Ace", "Comic Sans MS", sans-serif',
  },
  {
    label: 'Anime Ace / Classic Manga',
    value: '"Anime Ace", "Comic Neue", "Comic Sans MS", cursive, sans-serif',
  },
  {
    label: 'Inter Bold (Clean Webtoon)',
    value: 'Inter, system-ui, -apple-system, sans-serif',
  },
  {
    label: 'Impact (SFX / Action)',
    value: 'Impact, "Arial Black", sans-serif',
  },
  {
    label: 'Arial Black (Heavy Dialogue)',
    value: '"Arial Black", Gadget, sans-serif',
  },
  {
    label: 'Georgia Serif (Narration)',
    value: 'Georgia, Cambria, "Times New Roman", serif',
  },
  {
    label: 'JetBrains Mono (System / UIBox)',
    value: '"JetBrains Mono", monospace',
  },
];

const TEXT_COLOR_PRESETS = [
  { label: 'Black', hex: '#000000' },
  { label: 'White', hex: '#ffffff' },
  { label: 'Crimson', hex: '#dc2626' },
  { label: 'Indigo', hex: '#4338ca' },
  { label: 'Amber', hex: '#d97706' },
];

const STROKE_COLOR_PRESETS = [
  { label: 'Pure White', hex: '#ffffff' },
  { label: 'Pitch Black', hex: '#000000' },
  { label: 'Soft Cream', hex: '#fef3c7' },
  { label: 'Sky Halo', hex: '#e0f2fe' },
];

/**
 * Right-hand sidebar displaying all detected speech bubble text blocks synchronized
 * with the selected box on `MangaCanvas`.
 * Supports real-time editing of `original_text`, `translated_text`, font family,
 * font size, text color, alignment, and white outline stroke.
 */
export default function TextSidebar({
  page,
  selectedBlockId,
  onSelectBlock,
  onUpdateBlock,
  onApplyStyleToAll,
  onAddBlock,
  onDeleteBlock,
}) {
  const cardRefs = useRef({});
  const blocks = page?.blocks || [];
  const selectedBlock = blocks.find((b) => b.id === selectedBlockId) || null;

  // Automatically scroll the selected block card into view when selected from MangaCanvas
  useEffect(() => {
    if (selectedBlockId && cardRefs.current[selectedBlockId]) {
      cardRefs.current[selectedBlockId].scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
      });
    }
  }, [selectedBlockId]);

  const currentFontFamily =
    selectedBlock?.style?.font_family ||
    selectedBlock?.font_family ||
    COMIC_FONT_FAMILIES[0].value;
  const currentFontSize =
    selectedBlock?.font_size || selectedBlock?.style?.font_size || 18;
  const currentTextColor =
    selectedBlock?.color || selectedBlock?.style?.color || '#000000';
  const currentStrokeColor =
    selectedBlock?.style?.stroke_color || selectedBlock?.stroke_color || '#ffffff';
  const currentStrokeWidth =
    selectedBlock?.style?.stroke_width !== undefined
      ? selectedBlock.style.stroke_width
      : selectedBlock?.stroke_width !== undefined
      ? selectedBlock.stroke_width
      : 4;
  const currentAlign =
    selectedBlock?.align || selectedBlock?.style?.align || 'center';
  const currentLineHeight =
    selectedBlock?.style?.line_height || selectedBlock?.line_height || 1.18;

  return (
    <aside className="w-80 xl:w-96 shrink-0 border-l border-slate-800/90 bg-slate-900/75 backdrop-blur-md flex flex-col overflow-hidden select-none">
      {/* Sidebar Header */}
      <div className="p-3.5 border-b border-slate-800/90 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <MessageSquare className="h-4 w-4 text-indigo-400" />
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-200">
            Text Blocks
          </span>
          <span className="rounded-md bg-slate-800 border border-slate-700/70 px-1.5 py-0.5 font-mono text-[10px] text-indigo-300">
            {blocks.length}
          </span>
        </div>

        {page && (
          <button
            type="button"
            onClick={onAddBlock}
            className="inline-flex items-center gap-1 rounded-lg bg-indigo-600/20 hover:bg-indigo-600/35 border border-indigo-500/30 px-2.5 py-1 text-[11px] font-semibold text-indigo-200 transition-colors cursor-pointer"
            title="Add a new draggable text box to the canvas"
          >
            <Plus className="h-3.5 w-3.5 text-indigo-400" />
            <span>Add Bubble</span>
          </button>
        )}
      </div>

      {/* Real-Time Selected Block Editor & Typography Customizer */}
      {selectedBlock ? (
        <div className="p-3.5 border-b border-slate-800/90 bg-slate-950/85 space-y-3 overflow-y-auto max-h-[58%]">
          {/* Selected Bubble Header */}
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-1.5 min-w-0">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-indigo-500/20 border border-indigo-500/40 font-mono text-[11px] font-bold text-indigo-300">
                #{selectedBlock.order}
              </span>
              <span className="text-xs font-semibold text-slate-200 truncate font-mono">
                {selectedBlock.id}
              </span>
            </div>

            <div className="flex items-center gap-1">
              {onApplyStyleToAll && (
                <button
                  type="button"
                  onClick={() =>
                    onApplyStyleToAll({
                      font_family: currentFontFamily,
                      font_size: currentFontSize,
                      color: currentTextColor,
                      stroke_color: currentStrokeColor,
                      stroke_width: currentStrokeWidth,
                      align: currentAlign,
                      line_height: currentLineHeight,
                    })
                  }
                  className="inline-flex items-center gap-1 rounded-lg bg-slate-800/90 hover:bg-slate-700 border border-slate-700/80 px-2 py-1 text-[10px] font-medium text-slate-300 transition-colors cursor-pointer"
                  title="Apply this bubble's font, colors, and outline stroke settings to all bubbles on this page"
                >
                  <CopyCheck className="h-3 w-3 text-indigo-400" />
                  <span>Apply Style to All</span>
                </button>
              )}

              <button
                type="button"
                onClick={() => onDeleteBlock && onDeleteBlock(selectedBlock.id)}
                className="inline-flex items-center justify-center h-6 w-6 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-rose-950/50 transition-colors cursor-pointer"
                title="Delete selected text block"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>

          {/* Editable Original OCR Text */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <label
                htmlFor="sidebar-original-text"
                className="text-[11px] font-medium text-slate-400"
              >
                Original Text (OCR)
              </label>
              <span className="text-[10px] font-mono text-slate-500">
                {selectedBlock.text_direction || 'auto'}
              </span>
            </div>
            <textarea
              id="sidebar-original-text"
              rows={2}
              value={selectedBlock.original_text ?? ''}
              onChange={(e) =>
                onUpdateBlock(selectedBlock.id, {
                  original_text: e.target.value,
                  source_text: e.target.value,
                })
              }
              className="w-full rounded-xl border border-slate-800 bg-slate-900/90 px-3 py-1.5 text-xs text-slate-200 focus:border-indigo-500 focus:outline-none select-text resize-none font-sans"
              placeholder="Original extracted source text..."
            />
          </div>

          {/* Editable Translated Text */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <label
                htmlFor="sidebar-translated-text"
                className="text-[11px] font-semibold text-indigo-300 flex items-center gap-1"
              >
                <Sparkles className="h-3 w-3 text-indigo-400" />
                Translated Text
              </label>
              <span className="text-[10px] font-mono text-slate-500">
                {(selectedBlock.translated_text || '').length} chars
              </span>
            </div>
            <textarea
              id="sidebar-translated-text"
              rows={3}
              value={selectedBlock.translated_text ?? ''}
              onChange={(e) =>
                onUpdateBlock(selectedBlock.id, {
                  translated_text: e.target.value,
                })
              }
              className="w-full rounded-xl border border-indigo-500/40 bg-slate-900 px-3 py-2 text-xs font-medium text-slate-100 focus:border-indigo-400 focus:outline-none select-text resize-none"
              placeholder="Type translated dialogue..."
            />
          </div>

          {/* Font Family Selector */}
          <div className="space-y-1">
            <label
              htmlFor="sidebar-font-family"
              className="text-[11px] font-medium text-slate-400 flex items-center gap-1"
            >
              <Type className="h-3 w-3 text-indigo-400" />
              Font Family
            </label>
            <select
              id="sidebar-font-family"
              value={currentFontFamily}
              onChange={(e) =>
                onUpdateBlock(selectedBlock.id, {
                  font_family: e.target.value,
                })
              }
              className="w-full rounded-xl border border-slate-800 bg-slate-900 px-2.5 py-1.5 text-xs text-slate-200 focus:border-indigo-500 focus:outline-none cursor-pointer"
            >
              {COMIC_FONT_FAMILIES.map((f) => (
                <option key={f.label} value={f.value} className="bg-slate-900 text-slate-100">
                  {f.label}
                </option>
              ))}
            </select>
          </div>

          {/* Font Size & Text Alignment */}
          <div className="grid grid-cols-12 gap-2.5 items-end">
            <div className="col-span-7 space-y-1">
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-slate-400 font-medium">Font Size</span>
                <span className="font-mono font-semibold text-indigo-300">
                  {currentFontSize}px
                </span>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min={10}
                  max={64}
                  value={currentFontSize}
                  onChange={(e) =>
                    onUpdateBlock(selectedBlock.id, {
                      font_size: Number(e.target.value),
                    })
                  }
                  className="w-full accent-indigo-500 cursor-pointer"
                />
                <input
                  type="number"
                  min={8}
                  max={96}
                  value={currentFontSize}
                  onChange={(e) =>
                    onUpdateBlock(selectedBlock.id, {
                      font_size: Math.max(8, Math.min(96, Number(e.target.value) || 18)),
                    })
                  }
                  className="w-12 rounded-lg border border-slate-800 bg-slate-900 px-1.5 py-1 text-center font-mono text-xs text-slate-200 focus:border-indigo-500 focus:outline-none"
                />
              </div>
            </div>

            <div className="col-span-5 space-y-1">
              <span className="text-[11px] font-medium text-slate-400 block">
                Alignment
              </span>
              <div className="flex rounded-xl border border-slate-800 bg-slate-900 p-0.5">
                {[
                  { value: 'left', icon: AlignLeft, title: 'Align Left' },
                  { value: 'center', icon: AlignCenter, title: 'Align Center' },
                  { value: 'right', icon: AlignRight, title: 'Align Right' },
                ].map(({ value, icon: Icon, title }) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => onUpdateBlock(selectedBlock.id, { align: value })}
                    title={title}
                    className={`flex-1 flex items-center justify-center py-1 rounded-lg text-xs transition-colors cursor-pointer ${
                      currentAlign === value
                        ? 'bg-indigo-600 text-white shadow'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <Icon className="h-3.5 w-3.5" />
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Text Color & White Outline Stroke Customizer */}
          <div className="rounded-xl border border-slate-800/90 bg-slate-900/60 p-2.5 space-y-2.5">
            {/* Text Fill Color */}
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <input
                  type="color"
                  aria-label="Text Color"
                  value={currentTextColor}
                  onChange={(e) =>
                    onUpdateBlock(selectedBlock.id, { color: e.target.value })
                  }
                  className="h-6 w-7 rounded border border-slate-700 bg-transparent cursor-pointer"
                />
                <div>
                  <div className="text-[11px] font-medium text-slate-200">
                    Text Color
                  </div>
                  <div className="font-mono text-[10px] text-slate-400 uppercase">
                    {currentTextColor}
                  </div>
                </div>
              </div>

              {/* Quick Text Color Swatches */}
              <div className="flex items-center gap-1">
                {TEXT_COLOR_PRESETS.map((preset) => (
                  <button
                    key={preset.hex}
                    type="button"
                    onClick={() =>
                      onUpdateBlock(selectedBlock.id, { color: preset.hex })
                    }
                    title={`Text Color: ${preset.label}`}
                    className={`h-4 w-4 rounded-full border transition-transform cursor-pointer ${
                      currentTextColor.toLowerCase() === preset.hex
                        ? 'border-indigo-400 scale-110 ring-2 ring-indigo-500/40'
                        : 'border-slate-600 hover:scale-105'
                    }`}
                    style={{ backgroundColor: preset.hex }}
                  />
                ))}
              </div>
            </div>

            <div className="h-px bg-slate-800/80" />

            {/* White Outline Stroke Controls */}
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <input
                    type="color"
                    aria-label="Outline Stroke Color"
                    value={currentStrokeColor}
                    onChange={(e) =>
                      onUpdateBlock(selectedBlock.id, {
                        stroke_color: e.target.value,
                      })
                    }
                    className="h-6 w-7 rounded border border-slate-700 bg-transparent cursor-pointer"
                  />
                  <div>
                    <div className="text-[11px] font-medium text-slate-200">
                      White Outline Stroke
                    </div>
                    <div className="font-mono text-[10px] text-slate-400">
                      {currentStrokeColor.toUpperCase()} • {currentStrokeWidth}px
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-1">
                  {STROKE_COLOR_PRESETS.map((preset) => (
                    <button
                      key={preset.hex}
                      type="button"
                      onClick={() =>
                        onUpdateBlock(selectedBlock.id, {
                          stroke_color: preset.hex,
                        })
                      }
                      title={`Outline Color: ${preset.label}`}
                      className={`h-4 w-4 rounded-full border transition-transform cursor-pointer ${
                        currentStrokeColor.toLowerCase() === preset.hex
                          ? 'border-indigo-400 scale-110 ring-2 ring-indigo-500/40'
                          : 'border-slate-600 hover:scale-105'
                      }`}
                      style={{ backgroundColor: preset.hex }}
                    />
                  ))}
                  <button
                    type="button"
                    onClick={() =>
                      onUpdateBlock(selectedBlock.id, {
                        stroke_color: '#ffffff',
                        stroke_width: 4,
                      })
                    }
                    title="Reset to default 4px pure white outline stroke"
                    className="ml-1 p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors cursor-pointer"
                  >
                    <RotateCcw className="h-3 w-3" />
                  </button>
                </div>
              </div>

              {/* Outline Stroke Width Slider */}
              <div className="flex items-center gap-2.5">
                <span className="text-[10px] font-mono text-slate-400 shrink-0 w-16">
                  Width: {currentStrokeWidth}px
                </span>
                <input
                  type="range"
                  min={0}
                  max={12}
                  step={0.5}
                  value={currentStrokeWidth}
                  onChange={(e) =>
                    onUpdateBlock(selectedBlock.id, {
                      stroke_width: Number(e.target.value),
                    })
                  }
                  className="flex-1 accent-indigo-500 cursor-pointer"
                />
              </div>
            </div>
          </div>

          {/* Box Geometry Readout */}
          <div className="flex items-center justify-between text-[10px] font-mono text-slate-500 px-1">
            <span>
              Pos: ({selectedBlock.x}, {selectedBlock.y})
            </span>
            <span>
              Box: {selectedBlock.width}×{selectedBlock.height}px
            </span>
          </div>
        </div>
      ) : (
        page &&
        blocks.length > 0 && (
          <div className="px-4 py-2.5 border-b border-slate-800/80 bg-slate-950/50 text-[11px] text-slate-400">
            Select a speech bubble on the canvas or from the list below to edit its text and
            typography.
          </div>
        )
      )}

      {/* Synchronized List of All Detected Speech Bubbles */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {!page ? (
          <div className="text-center py-12 px-4 text-xs text-slate-500 space-y-1">
            <p className="font-medium text-slate-400">No Page Active</p>
            <p className="text-[11px]">
              Import a comic page to detect and edit speech bubbles.
            </p>
          </div>
        ) : blocks.length === 0 ? (
          <div className="text-center py-12 px-4 text-xs text-slate-500">
            {page.status === 'processing'
              ? 'Detecting & translating speech bubbles...'
              : 'No dialogue bubbles on this page. Click "Add Bubble" to place one manually.'}
          </div>
        ) : (
          blocks.map((block) => {
            const isSelected = block.id === selectedBlockId;
            return (
              <div
                key={block.id}
                ref={(el) => {
                  if (el) {
                    cardRefs.current[block.id] = el;
                  } else {
                    delete cardRefs.current[block.id];
                  }
                }}
                onClick={() => onSelectBlock && onSelectBlock(block.id)}
                className={`rounded-xl border p-3 transition-all cursor-pointer space-y-2 ${
                  isSelected
                    ? 'border-indigo-500 bg-indigo-950/35 shadow-md shadow-indigo-950/50 ring-1 ring-indigo-500/30'
                    : 'border-slate-800/90 bg-slate-950/60 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between text-[11px]">
                  <div className="flex items-center gap-1.5">
                    <span
                      className={`inline-flex items-center justify-center h-4 px-1.5 rounded font-mono text-[10px] font-bold ${
                        isSelected
                          ? 'bg-indigo-500 text-white'
                          : 'bg-slate-800 text-indigo-300'
                      }`}
                    >
                      #{block.order}
                    </span>
                    <span className="font-mono text-slate-400 text-[11px]">
                      {block.id}
                    </span>
                  </div>
                  <span className="font-mono text-[10px] text-slate-500">
                    {block.font_size}px • {block.width}×{block.height}
                  </span>
                </div>

                {/* Inline Original Text Preview / Quick Edit */}
                <div className="text-[11px] text-slate-400 bg-slate-900/90 border border-slate-800/80 rounded-lg px-2.5 py-1.5 font-sans select-text break-words">
                  <span className="text-[10px] font-mono uppercase text-slate-500 mr-1.5">
                    OCR:
                  </span>
                  {block.original_text || (
                    <span className="italic text-slate-600">Empty</span>
                  )}
                </div>

                {/* Inline Translated Text Preview */}
                <div className="text-xs font-medium text-slate-100 select-text break-words pl-0.5">
                  {block.translated_text || (
                    <span className="italic text-slate-500">Empty translation</span>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </aside>
  );
}
