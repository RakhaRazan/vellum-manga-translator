import React, {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from 'react';
import {
  Stage,
  Layer,
  Image as KonvaImage,
  Text as KonvaText,
  Rect as KonvaRect,
  Transformer,
} from 'react-konva';
import {
  Eye,
  FileImage,
  Loader2,
  Maximize2,
  Move,
  Sparkles,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';

/**
 * Custom hook to load an HTMLImageElement from a URL or base64 data URI.
 */
function useLoadedImage(src) {
  const [image, setImage] = useState(null);

  useEffect(() => {
    if (!src) {
      setImage(null);
      return;
    }
    let isCancelled = false;
    const img = new window.Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      if (!isCancelled) {
        setImage(img);
      }
    };
    img.onerror = () => {
      if (!isCancelled) {
        setImage(null);
      }
    };
    img.src = src;

    return () => {
      isCancelled = true;
    };
  }, [src]);

  return image;
}

/**
 * Interactive Konva Canvas for Manga/Manhwa/Manhua Translation.
 * - Layer 1 (Background): Cleaned LaMa-inpainted image (or original image while holding "Hold to Compare Original")
 * - Layer 2 (Typography): Draggable & Resizable translated text boxes rendered at native resolution
 * - Layer 3 (UI Overlay): Hover/selection bounding boxes and Konva Transformer handles (excluded during export)
 * - Export: Lossless full-resolution .png / .jpg export at exact original image dimensions
 */
const MangaCanvas = forwardRef(function MangaCanvas(
  {
    page,
    selectedBlockId,
    onSelectBlock,
    onUpdateBlock,
    onImportDrop,
    onLoadSample,
    isProcessing = false,
  },
  ref
) {
  const containerRef = useRef(null);
  const stageRef = useRef(null);
  const backdropShadowRef = useRef(null);
  const exportWhiteBgRef = useRef(null);
  const baseImageNodeRef = useRef(null);
  const textLayerRef = useRef(null);
  const uiLayerRef = useRef(null);
  const transformerRef = useRef(null);
  const textNodeRefs = useRef({});

  const [viewport, setViewport] = useState({ width: 900, height: 700 });
  const [stageTransform, setStageTransform] = useState({
    scale: 1,
    x: 0,
    y: 0,
  });
  const [hoveredBlockId, setHoveredBlockId] = useState(null);
  const [isComparingOriginal, setIsComparingOriginal] = useState(false);
  const [isDraggingFileOver, setIsDraggingFileOver] = useState(false);

  const cleanedImgEl = useLoadedImage(page?.cleanedImage || null);
  const originalImgEl = useLoadedImage(page?.originalUrl || null);

  const exportBaseImage = cleanedImgEl || originalImgEl;
  const activeBaseImage =
    isComparingOriginal && originalImgEl
      ? originalImgEl
      : exportBaseImage;

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

  // Measure container viewport size
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const updateSize = () => {
      const rect = el.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        setViewport({
          width: Math.floor(rect.width),
          height: Math.floor(rect.height),
        });
      }
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Fit comic page into viewport cleanly
  const fitPageToViewport = useCallback(() => {
    if (!activeBaseImage || viewport.width <= 0 || viewport.height <= 0) return;
    const padding = 48;
    const availW = Math.max(200, viewport.width - padding * 2);
    const availH = Math.max(200, viewport.height - padding * 2);

    const scaleX = availW / imageWidth;
    const scaleY = availH / imageHeight;
    const fitScale = Math.min(1.15, Math.max(0.15, Math.min(scaleX, scaleY)));

    const centeredX = Math.round((viewport.width - imageWidth * fitScale) / 2);
    const centeredY = Math.round((viewport.height - imageHeight * fitScale) / 2);

    setStageTransform({
      scale: Number(fitScale.toFixed(4)),
      x: centeredX,
      y: centeredY,
    });
  }, [activeBaseImage, viewport.width, viewport.height, imageWidth, imageHeight]);

  // Auto-fit when switching to a different page or when its image first loads
  const lastFittedPageIdRef = useRef(null);
  useEffect(() => {
    if (page?.id && activeBaseImage && lastFittedPageIdRef.current !== page.id) {
      lastFittedPageIdRef.current = page.id;
      fitPageToViewport();
    }
  }, [page?.id, activeBaseImage, fitPageToViewport]);

  // Attach Konva Transformer to currently selected text node
  useEffect(() => {
    const tr = transformerRef.current;
    if (!tr) return;

    if (
      selectedBlockId &&
      !isComparingOriginal &&
      textNodeRefs.current[selectedBlockId]
    ) {
      tr.nodes([textNodeRefs.current[selectedBlockId]]);
      tr.getLayer()?.batchDraw();
    } else {
      tr.nodes([]);
      tr.getLayer()?.batchDraw();
    }
  }, [selectedBlockId, isComparingOriginal, page?.blocks]);

  // Expose lossless full-resolution .png / .jpg export to parent via ref
  useImperativeHandle(
    ref,
    () => ({
      /**
       * Composites the inpainted background layer and Konva text layer
       * at exact original image dimensions (imageWidth x imageHeight, pixelRatio: 1)
       * without UI handles, shadows, or viewport zoom clipping.
       * @param {'png' | 'jpg' | 'jpeg'} format
       * @param {number} quality - JPEG quality (0.0 to 1.0), defaults to 0.98
       * @returns {string | null} Base64 data URL
       */
      exportImage: (format = 'png', quality = 0.98) => {
        const stage = stageRef.current;
        if (!stage || !exportBaseImage) return null;

        const isJpeg =
          String(format).toLowerCase() === 'jpg' ||
          String(format).toLowerCase() === 'jpeg';
        const mimeType = isJpeg ? 'image/jpeg' : 'image/png';

        // 1. Save current viewport transform & layer visibility
        const prevScaleX = stage.scaleX();
        const prevScaleY = stage.scaleY();
        const prevX = stage.x();
        const prevY = stage.y();
        const prevWidth = stage.width();
        const prevHeight = stage.height();

        // 2. Hide non-content UI elements (selection boxes, Transformer, viewport drop shadow)
        if (uiLayerRef.current) uiLayerRef.current.hide();
        if (backdropShadowRef.current) backdropShadowRef.current.hide();
        if (exportWhiteBgRef.current) {
          if (isJpeg) {
            exportWhiteBgRef.current.show();
          } else {
            exportWhiteBgRef.current.hide();
          }
        }

        // Ensure text layer is visible and base image is the inpainted image
        if (textLayerRef.current) textLayerRef.current.show();
        if (baseImageNodeRef.current) {
          baseImageNodeRef.current.image(exportBaseImage);
        }

        // 3. Set stage to exact 1:1 original comic dimensions
        stage.scale({ x: 1, y: 1 });
        stage.position({ x: 0, y: 0 });
        stage.width(imageWidth);
        stage.height(imageHeight);
        stage.draw();

        // 4. Capture composite data URL at full native resolution
        const dataUrl = stage.toDataURL({
          x: 0,
          y: 0,
          width: imageWidth,
          height: imageHeight,
          pixelRatio: 1,
          mimeType,
          quality: isJpeg ? quality : 1,
        });

        // 5. Restore viewport transform & UI overlay immediately
        if (exportWhiteBgRef.current) exportWhiteBgRef.current.hide();
        if (backdropShadowRef.current) backdropShadowRef.current.show();
        if (uiLayerRef.current) uiLayerRef.current.show();
        if (baseImageNodeRef.current) {
          baseImageNodeRef.current.image(activeBaseImage);
        }
        if (textLayerRef.current) {
          textLayerRef.current.visible(!isComparingOriginal);
        }

        stage.scale({ x: prevScaleX, y: prevScaleY });
        stage.position({ x: prevX, y: prevY });
        stage.width(prevWidth);
        stage.height(prevHeight);
        stage.draw();

        return dataUrl;
      },
      fitToScreen: fitPageToViewport,
    }),
    [
      exportBaseImage,
      activeBaseImage,
      isComparingOriginal,
      imageWidth,
      imageHeight,
      fitPageToViewport,
    ]
  );

  // Smooth mouse-wheel zoom centered on pointer position
  const handleWheel = (e) => {
    e.evt.preventDefault();
    const stage = stageRef.current;
    if (!stage) return;

    const oldScale = stageTransform.scale;
    const pointer = stage.getPointerPosition();
    if (!pointer) return;

    const mousePointTo = {
      x: (pointer.x - stageTransform.x) / oldScale,
      y: (pointer.y - stageTransform.y) / oldScale,
    };

    const zoomDirection = e.evt.deltaY < 0 ? 1 : -1;
    const scaleBy = 1.12;
    const unclampedScale =
      zoomDirection > 0 ? oldScale * scaleBy : oldScale / scaleBy;
    const newScale = Math.min(4.5, Math.max(0.15, unclampedScale));

    const newPos = {
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    };

    setStageTransform({
      scale: Number(newScale.toFixed(4)),
      x: Math.round(newPos.x),
      y: Math.round(newPos.y),
    });
  };

  const adjustZoom = (factor) => {
    const oldScale = stageTransform.scale;
    const newScale = Math.min(4.5, Math.max(0.15, oldScale * factor));
    const centerX = viewport.width / 2;
    const centerY = viewport.height / 2;
    const pointTo = {
      x: (centerX - stageTransform.x) / oldScale,
      y: (centerY - stageTransform.y) / oldScale,
    };

    setStageTransform({
      scale: Number(newScale.toFixed(4)),
      x: Math.round(centerX - pointTo.x * newScale),
      y: Math.round(centerY - pointTo.y * newScale),
    });
  };

  // Deselect when clicking on empty stage or base comic image
  const handleStageMouseDown = (e) => {
    const clickedOnEmpty =
      e.target === e.target.getStage() ||
      e.target.name() === 'base-comic-image';
    if (clickedOnEmpty && onSelectBlock) {
      onSelectBlock(null);
    }
  };

  // Drag-and-drop image files onto canvas
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

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingFileOver(false);
    const files = Array.from(e.dataTransfer?.files || []).filter((f) =>
      f.type.startsWith('image/')
    );
    if (files.length > 0 && onImportDrop) {
      onImportDrop(files);
    }
  };

  const blocks = page?.blocks || [];

  return (
    <div
      ref={containerRef}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className="relative flex-1 h-full w-full bg-slate-950 overflow-hidden select-none"
      style={{
        backgroundImage:
          'radial-gradient(rgba(71, 85, 105, 0.28) 1px, transparent 1px)',
        backgroundSize: '20px 20px',
      }}
    >
      {/* File Drag & Drop Overlay */}
      {isDraggingFileOver && (
        <div className="absolute inset-4 z-40 rounded-2xl border-2 border-dashed border-indigo-400 bg-indigo-950/80 backdrop-blur-sm flex flex-col items-center justify-center gap-3 pointer-events-none">
          <FileImage className="h-12 w-12 text-indigo-300 animate-bounce" />
          <p className="text-sm font-semibold text-indigo-100">
            Drop Comic Page(s) to Import &amp; Translate
          </p>
        </div>
      )}

      {/* Empty Workspace State when no page is loaded */}
      {!page && (
        <div className="h-full w-full flex items-center justify-center p-6">
          <div className="max-w-md w-full rounded-2xl border border-dashed border-slate-800 bg-slate-900/60 backdrop-blur-sm p-8 text-center space-y-4 shadow-2xl">
            <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-indigo-500/10 border border-indigo-500/25 text-indigo-400">
              <FileImage className="h-7 w-7" />
            </div>
            <div className="space-y-1.5">
              <h2 className="text-base font-semibold text-slate-100">
                No Comic Page Loaded
              </h2>
              <p className="text-xs text-slate-400 leading-relaxed">
                Click <span className="text-indigo-300 font-semibold">Import Image</span> in the top
                toolbar or drag &amp; drop Manga, Manhwa, or Manhua pages directly onto this canvas.
              </p>
            </div>
            <div className="pt-1 flex flex-wrap items-center justify-center gap-2 text-[11px] font-mono text-slate-400">
              <span className="rounded-lg bg-slate-800/80 border border-slate-700/60 px-2.5 py-1">
                1. ONNX Detection
              </span>
              <span className="rounded-lg bg-slate-800/80 border border-slate-700/60 px-2.5 py-1">
                2. OCR &amp; LaMa Inpaint
              </span>
              <span className="rounded-lg bg-slate-800/80 border border-slate-700/60 px-2.5 py-1">
                3. Gemma 4 E2B Translation
              </span>
            </div>
            {onLoadSample && (
              <div className="pt-2 border-t border-slate-800/80 flex flex-wrap items-center justify-center gap-2">
                <button
                  type="button"
                  onClick={() => onLoadSample('sample_ja.jpg', 'ja')}
                  className="inline-flex items-center gap-1.5 rounded-xl bg-indigo-600/20 hover:bg-indigo-600/35 border border-indigo-500/40 px-3 py-1.5 text-xs font-semibold text-indigo-200 transition-colors cursor-pointer"
                >
                  <Sparkles className="h-3.5 w-3.5 text-indigo-400" />
                  Try sample_ja.jpg (JA)
                </button>
                <button
                  type="button"
                  onClick={() => onLoadSample('sample_ko.jpg', 'ko')}
                  className="inline-flex items-center gap-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-300 transition-colors cursor-pointer"
                >
                  sample_ko.jpg (KO)
                </button>
                <button
                  type="button"
                  onClick={() => onLoadSample('sample_zh.jpg', 'zh')}
                  className="inline-flex items-center gap-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-300 transition-colors cursor-pointer"
                >
                  sample_zh.jpg (ZH)
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Konva Interactive Canvas */}
      {page && activeBaseImage && (
        <Stage
          ref={stageRef}
          width={viewport.width}
          height={viewport.height}
          scaleX={stageTransform.scale}
          scaleY={stageTransform.scale}
          x={stageTransform.x}
          y={stageTransform.y}
          draggable
          onWheel={handleWheel}
          onMouseDown={handleStageMouseDown}
          onTouchStart={handleStageMouseDown}
          onDragEnd={(e) => {
            if (e.target === stageRef.current) {
              setStageTransform((prev) => ({
                ...prev,
                x: Math.round(e.target.x()),
                y: Math.round(e.target.y()),
              }));
            }
          }}
          className="cursor-grab active:cursor-grabbing"
        >
          {/* Layer 1: Base Cleaned Inpainted Image (or Original Image during Hold-to-Compare) */}
          <Layer>
            <KonvaRect
              ref={backdropShadowRef}
              x={0}
              y={0}
              width={imageWidth}
              height={imageHeight}
              fill="#0f172a"
              shadowColor="#000000"
              shadowBlur={32}
              shadowOpacity={0.65}
              listening={false}
            />
            <KonvaRect
              ref={exportWhiteBgRef}
              x={0}
              y={0}
              width={imageWidth}
              height={imageHeight}
              fill="#ffffff"
              visible={false}
              listening={false}
            />
            <KonvaImage
              ref={baseImageNodeRef}
              name="base-comic-image"
              image={activeBaseImage}
              x={0}
              y={0}
              width={imageWidth}
              height={imageHeight}
            />
          </Layer>

          {/* Layer 2: Pure Translated Text Elements (Composited during Export) */}
          <Layer ref={textLayerRef} visible={!isComparingOriginal}>
            {blocks.map((block) => {
              const bx =
                typeof block.x === 'number'
                  ? block.x
                  : block.bbox?.[0] ?? 0;
              const by =
                typeof block.y === 'number'
                  ? block.y
                  : block.bbox?.[1] ?? 0;
              const bw =
                typeof block.width === 'number'
                  ? block.width
                  : Math.max(40, (block.bbox?.[2] ?? 140) - (block.bbox?.[0] ?? 0));
              const bh =
                typeof block.height === 'number'
                  ? block.height
                  : Math.max(24, (block.bbox?.[3] ?? 60) - (block.bbox?.[1] ?? 0));
              const rotation = block.rotation || 0;

              const fontSize = block.font_size || block.style?.font_size || 18;
              const fontFamily =
                block.style?.font_family ||
                block.font_family ||
                '"Comic Neue", "Anime Ace", "Comic Sans MS", sans-serif';
              const textColor = block.color || block.style?.color || '#000000';
              const strokeColor =
                block.style?.stroke_color || block.stroke_color || '#ffffff';
              const strokeWidth =
                block.style?.stroke_width !== undefined
                  ? block.style.stroke_width
                  : block.stroke_width !== undefined
                  ? block.stroke_width
                  : 4;
              const textAlign = block.align || block.style?.align || 'center';
              const lineHeight =
                block.style?.line_height || block.line_height || 1.18;

              return (
                <KonvaText
                  key={block.id}
                  ref={(node) => {
                    if (node) {
                      textNodeRefs.current[block.id] = node;
                    } else {
                      delete textNodeRefs.current[block.id];
                    }
                  }}
                  id={block.id}
                  name="translated-text-box"
                  x={bx}
                  y={by}
                  width={bw}
                  height={bh}
                  rotation={rotation}
                  text={block.translated_text || ''}
                  fontSize={fontSize}
                  fontFamily={fontFamily}
                  fontStyle="bold"
                  fill={textColor}
                  stroke={strokeWidth > 0 ? strokeColor : undefined}
                  strokeWidth={strokeWidth}
                  fillAfterStrokeEnabled={true}
                  lineJoin="round"
                  align={textAlign}
                  verticalAlign="middle"
                  lineHeight={lineHeight}
                  wrap="word"
                  padding={4}
                  draggable
                  onMouseEnter={(e) => {
                    setHoveredBlockId(block.id);
                    const container = e.target.getStage()?.container();
                    if (container) container.style.cursor = 'move';
                  }}
                  onMouseLeave={(e) => {
                    setHoveredBlockId((prev) =>
                      prev === block.id ? null : prev
                    );
                    const container = e.target.getStage()?.container();
                    if (container) container.style.cursor = 'grab';
                  }}
                  onClick={(e) => {
                    e.cancelBubble = true;
                    if (onSelectBlock) onSelectBlock(block.id);
                  }}
                  onTap={(e) => {
                    e.cancelBubble = true;
                    if (onSelectBlock) onSelectBlock(block.id);
                  }}
                  onDragStart={(e) => {
                    e.cancelBubble = true;
                    if (onSelectBlock) onSelectBlock(block.id);
                  }}
                  onDragEnd={(e) => {
                    e.cancelBubble = true;
                    const node = e.target;
                    const newX = Math.round(node.x());
                    const newY = Math.round(node.y());
                    if (onUpdateBlock) {
                      onUpdateBlock(block.id, {
                        x: newX,
                        y: newY,
                        bbox: [newX, newY, newX + bw, newY + bh],
                      });
                    }
                  }}
                  onTransform={(e) => {
                    const node = e.target;
                    const scaleX = node.scaleX();
                    const scaleY = node.scaleY();
                    if (scaleX !== 1 || scaleY !== 1) {
                      node.scaleX(1);
                      node.scaleY(1);
                      node.width(Math.max(32, node.width() * scaleX));
                      node.height(Math.max(20, node.height() * scaleY));
                    }
                  }}
                  onTransformEnd={(e) => {
                    const node = e.target;
                    const scaleX = node.scaleX();
                    const scaleY = node.scaleY();
                    node.scaleX(1);
                    node.scaleY(1);

                    const newX = Math.round(node.x());
                    const newY = Math.round(node.y());
                    const newWidth = Math.round(
                      Math.max(32, node.width() * scaleX)
                    );
                    const newHeight = Math.round(
                      Math.max(20, node.height() * scaleY)
                    );
                    const newRotation = Number(node.rotation().toFixed(2));

                    if (onUpdateBlock) {
                      onUpdateBlock(block.id, {
                        x: newX,
                        y: newY,
                        width: newWidth,
                        height: newHeight,
                        rotation: newRotation,
                        bbox: [
                          newX,
                          newY,
                          newX + newWidth,
                          newY + newHeight,
                        ],
                      });
                    }
                  }}
                />
              );
            })}
          </Layer>

          {/* Layer 3: Interactive UI Overlay (Selection Outlines & Konva Transformer - Hidden on Export) */}
          <Layer ref={uiLayerRef} visible={!isComparingOriginal}>
            {blocks.map((block) => {
              const isSelected = selectedBlockId === block.id;
              const isHovered = hoveredBlockId === block.id;
              if (!isSelected && !isHovered) return null;

              const bx =
                typeof block.x === 'number'
                  ? block.x
                  : block.bbox?.[0] ?? 0;
              const by =
                typeof block.y === 'number'
                  ? block.y
                  : block.bbox?.[1] ?? 0;
              const bw =
                typeof block.width === 'number'
                  ? block.width
                  : Math.max(40, (block.bbox?.[2] ?? 140) - (block.bbox?.[0] ?? 0));
              const bh =
                typeof block.height === 'number'
                  ? block.height
                  : Math.max(24, (block.bbox?.[3] ?? 60) - (block.bbox?.[1] ?? 0));
              const rotation = block.rotation || 0;

              return (
                <KonvaRect
                  key={`highlight_${block.id}`}
                  x={bx}
                  y={by}
                  width={bw}
                  height={bh}
                  rotation={rotation}
                  stroke={isSelected ? '#6366f1' : '#38bdf8'}
                  strokeWidth={isSelected ? 2 : 1.5}
                  dash={isSelected ? undefined : [5, 4]}
                  fill={
                    isSelected
                      ? 'rgba(99, 102, 241, 0.08)'
                      : 'rgba(56, 189, 248, 0.05)'
                  }
                  cornerRadius={4}
                  listening={false}
                />
              );
            })}

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
              borderStroke="#6366f1"
              borderStrokeWidth={1.5}
              anchorStroke="#4f46e5"
              anchorFill="#eef2ff"
              anchorSize={8}
              anchorCornerRadius={2}
              padding={2}
              boundBoxFunc={(oldBox, newBox) => {
                if (newBox.width < 32 || newBox.height < 20) {
                  return oldBox;
                }
                return newBox;
              }}
            />
          </Layer>
        </Stage>
      )}

      {/* Active Comparison Mode Top Banner */}
      {page && isComparingOriginal && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-30 pointer-events-none">
          <div className="inline-flex items-center gap-2 rounded-full border border-amber-400/40 bg-amber-950/90 backdrop-blur-md px-4 py-1.5 text-xs font-semibold text-amber-200 shadow-xl">
            <Eye className="h-4 w-4 text-amber-400" />
            <span>Comparing Original Raw Page (Release to view translation)</span>
          </div>
        </div>
      )}

      {/* Pipeline Processing Overlay */}
      {page && (isProcessing || page.status === 'processing') && (
        <div className="absolute inset-0 z-30 flex items-center justify-center bg-slate-950/75 backdrop-blur-sm">
          <div className="rounded-2xl border border-indigo-500/30 bg-slate-900/95 px-6 py-5 shadow-2xl flex items-center gap-4 max-w-md">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-indigo-500/15 border border-indigo-500/30 text-indigo-400">
              <Loader2 className="h-6 w-6 animate-spin" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-semibold text-slate-100">
                  Processing Comic Page...
                </h3>
                <Sparkles className="h-3.5 w-3.5 text-indigo-400" />
              </div>
              <p className="mt-0.5 text-xs text-slate-400">
                Running ONNX Detector → OCR → LaMa Tile Inpainting → Gemma 4 E2B Translation
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Floating Bottom HUD: Hold to Compare Original + Zoom Controls */}
      {page && activeBaseImage && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 flex items-center gap-2.5 rounded-2xl border border-slate-800/90 bg-slate-900/90 backdrop-blur-md px-3 py-1.5 shadow-2xl">
          {/* Press-and-Hold "Hold to Compare Original" Button */}
          <button
            type="button"
            onMouseDown={() => setIsComparingOriginal(true)}
            onMouseUp={() => setIsComparingOriginal(false)}
            onMouseLeave={() => setIsComparingOriginal(false)}
            onTouchStart={() => setIsComparingOriginal(true)}
            onTouchEnd={() => setIsComparingOriginal(false)}
            className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-1.5 text-xs font-semibold transition-all cursor-pointer select-none ${
              isComparingOriginal
                ? 'border-amber-400/60 bg-amber-500/20 text-amber-200 shadow-inner'
                : 'border-slate-700/80 bg-slate-800/90 hover:bg-slate-700/80 text-slate-200'
            }`}
            title="Press and hold to temporarily view the original untranslated comic image"
          >
            <Eye className="h-3.5 w-3.5 text-indigo-400" />
            <span>Hold to Compare Original</span>
          </button>

          <div className="h-4 w-px bg-slate-800" />

          {/* Zoom Controls */}
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => adjustZoom(1 / 1.18)}
              className="inline-flex items-center justify-center h-7 w-7 rounded-lg hover:bg-slate-800 text-slate-300 transition-colors cursor-pointer"
              title="Zoom Out"
            >
              <ZoomOut className="h-3.5 w-3.5" />
            </button>

            <button
              type="button"
              onClick={fitPageToViewport}
              className="px-2 py-1 rounded-lg hover:bg-slate-800 font-mono text-xs font-medium text-slate-200 transition-colors cursor-pointer"
              title="Click to Fit Page to Screen"
            >
              {Math.round(stageTransform.scale * 100)}%
            </button>

            <button
              type="button"
              onClick={() => adjustZoom(1.18)}
              className="inline-flex items-center justify-center h-7 w-7 rounded-lg hover:bg-slate-800 text-slate-300 transition-colors cursor-pointer"
              title="Zoom In"
            >
              <ZoomIn className="h-3.5 w-3.5" />
            </button>

            <button
              type="button"
              onClick={fitPageToViewport}
              className="inline-flex items-center justify-center h-7 w-7 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors cursor-pointer"
              title="Fit to Viewport"
            >
              <Maximize2 className="h-3.5 w-3.5" />
            </button>
          </div>

          <div className="hidden sm:flex items-center gap-1.5 pl-1 border-l border-slate-800 text-[11px] font-mono text-slate-400">
            <Move className="h-3 w-3 text-slate-500" />
            <span>
              {imageWidth}×{imageHeight}px • {blocks.length} bubbles
            </span>
          </div>
        </div>
      )}
    </div>
  );
});

export default MangaCanvas;
