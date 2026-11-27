import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlignCenter,
  AlignLeft,
  AlignRight,
  AlertTriangle,
  CaseUpper,
  Check,
  ChevronDown,
  CopyCheck,
  Download,
  Loader2,
  MessageSquare,
  Plus,
  RefreshCw,
  RotateCcw,
  ScanSearch,
  Search,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Type,
  Upload,
  X,
} from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import useEditorStore, {
  DEFAULT_API_BASE_URL,
  DEFAULT_FONT_FALLBACK_CHAIN,
  buildKonvaFontFamily,
  ensureFontFaceLoaded,
  extractPrimaryFontName,
  isBlockCjkLeaked,
  loadFontFamilyFromCatalog,
} from '../store/useEditorStore.js';

export const COMIC_FONT_FAMILIES = [
  {
    label: 'CC Wild Words (Default Manga)',
    value:
      '"CC Wild Words", "Anime Ace", "Segoe UI Symbol", "Noto Sans Symbols 2", "Noto Sans", sans-serif',
  },
  {
    label: 'Comic Neue (Classic Comic)',
    value: '"Comic Neue", "CC Wild Words", "Anime Ace", "Comic Sans MS", sans-serif',
  },
  {
    label: 'Anime Ace / Classic Manga',
    value: '"Anime Ace", "CC Wild Words", "Comic Neue", "Comic Sans MS", cursive, sans-serif',
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
  { label: 'Slate Blue', hex: '#2563eb' },
  { label: 'Warm Amber', hex: '#c97a16' },
];

const STROKE_COLOR_PRESETS = [
  { label: 'Pure White', hex: '#ffffff' },
  { label: 'Pitch Black', hex: '#000000' },
  { label: 'Soft Cream', hex: '#fef3c7' },
  { label: 'Cool Mist', hex: '#e2e8f0' },
];

const FONT_FILTER_TABS = [
  { id: 'all', label: 'All' },
  { id: 'comic', label: 'Custom' },
  { id: 'system', label: 'Windows' },
  { id: 'catalog', label: 'Catalog' },
];

/**
 * Right-hand inspector sidebar displaying detected speech bubble text blocks
 * synchronized with `MangaCanvas`.
 *
 * Upgraded v2 Studio features:
 * - Crisp Sumi Graphite / Manga Paper desktop studio styling
 * - Fast searchable Font Picker connected to `GET /api/fonts/list`, `POST /api/fonts/upload`,
 *   `POST /api/fonts/download`, and dynamic `FontFace` loading from `/api/fonts/file/{font_name}`
 * - Automatic Konva Font Fallback Chain assembly so manga symbols (`♡`, `♪`, `…`) never box out
 * - Quick per-layer "Re-OCR Layer" (`/api/stage/ocr`) & "Re-Translate Layer" (`/api/stage/translate`) actions
 */
export default function TextSidebar({
  page,
  selectedBlockId,
  onSelectBlock,
  onUpdateBlock,
  onApplyStyleToAll,
  onAddBlock,
  onDeleteBlock,
  onReOcrLayer,
  onReTranslateLayer,
  onOpenFindReplace,
  busyLayerAction = null, // { blockId: string, stage: 'ocr' | 'translate' } | null
  isProcessing = false,
}) {
  const cardRefs = useRef({});
  const fontPickerRef = useRef(null);
  const fontSearchInputRef = useRef(null);
  const fontUploadInputRef = useRef(null);

  const {
    selectedLayerIds,
    toggleSelectLayer,
    typesetting,
    apiBaseUrl,
    fontsCatalogCache,
    setFontsCatalogCache,
    openModal,
  } = useEditorStore(
    useShallow((s) => ({
      selectedLayerIds: s.selectedLayerIds,
      toggleSelectLayer: s.toggleSelectLayer,
      typesetting: s.typesetting,
      apiBaseUrl: s.pipelineConfig.apiBaseUrl ?? DEFAULT_API_BASE_URL,
      fontsCatalogCache: s.fontsCatalogCache,
      setFontsCatalogCache: s.setFontsCatalogCache,
      openModal: s.openModal,
    }))
  );

  // Searchable Font Picker state
  const [isFontPickerOpen, setIsFontPickerOpen] = useState(false);
  const [fontSearch, setFontSearch] = useState('');
  const [fontFilterTab, setFontFilterTab] = useState('all'); // 'all' | 'comic' | 'system' | 'catalog'
  const [isLoadingFontsList, setIsLoadingFontsList] = useState(false);
  const [loadingFontFamily, setLoadingFontFamily] = useState(null);
  const [downloadingCatalogId, setDownloadingCatalogId] = useState(null);
  const [uploadingCustomFont, setUploadingCustomFont] = useState(false);
  const [fontNotice, setFontNotice] = useState(null);

  const blocks = page?.blocks || [];
  const selectedBlock = blocks.find((b) => b.id === selectedBlockId) || null;
  const selectedIdSet = new Set(
    selectedLayerIds && selectedLayerIds.length > 0
      ? selectedLayerIds
      : selectedBlockId
      ? [selectedBlockId]
      : []
  );

  const fallbackChain = useMemo(
    () =>
      Array.isArray(typesetting?.fontFallbackChain) &&
      typesetting.fontFallbackChain.length > 0
        ? typesetting.fontFallbackChain
        : DEFAULT_FONT_FALLBACK_CHAIN,
    [typesetting?.fontFallbackChain]
  );

  // Scroll active layer card into view when selection changes
  useEffect(() => {
    if (selectedBlockId && cardRefs.current[selectedBlockId]) {
      cardRefs.current[selectedBlockId].scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
      });
    }
  }, [selectedBlockId]);

  // Fetch `/api/fonts/list` on mount (or refresh) and cache in Zustand store
  const fetchFontsCatalog = useCallback(
    async (forceRefresh = false) => {
      setIsLoadingFontsList(true);
      setFontNotice(null);
      try {
        const url = `${apiBaseUrl}/api/fonts/list?include_catalog=true${
          forceRefresh ? '&refresh=true' : ''
        }`;
        const resp = await fetch(url, { headers: { Accept: 'application/json' } });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        setFontsCatalogCache(data);

        // Preload local custom/downloaded comic fonts & active fallback chain fonts into FontFaceSet
        const localFamilies = (data.families || []).filter(
          (f) => f.category === 'custom' || f.category === 'downloaded'
        );
        localFamilies.slice(0, 16).forEach((fam) => {
          loadFontFamilyFromCatalog(fam, data, apiBaseUrl);
        });
      } catch (err) {
        setFontNotice({
          type: 'error',
          text: err?.message || 'Unable to load /api/fonts/list',
        });
      } finally {
        setIsLoadingFontsList(false);
      }
    },
    [apiBaseUrl, setFontsCatalogCache]
  );

  useEffect(() => {
    if (!fontsCatalogCache && !isLoadingFontsList) {
      fetchFontsCatalog(false);
    }
  }, [fontsCatalogCache, isLoadingFontsList, fetchFontsCatalog]);

  // Ensure currently selected block's primary font is dynamically loaded via FontFace API
  const currentFontFamilyRaw =
    selectedBlock?.style?.font_family ||
    selectedBlock?.font_family ||
    typesetting?.fontFamily ||
    COMIC_FONT_FAMILIES[0].value;
  const currentPrimaryFont = extractPrimaryFontName(currentFontFamilyRaw);
  const effectiveKonvaFontStack = useMemo(
    () => buildKonvaFontFamily(currentFontFamilyRaw, fallbackChain),
    [currentFontFamilyRaw, fallbackChain]
  );

  useEffect(() => {
    if (!fontsCatalogCache || !currentPrimaryFont) return;
    loadFontFamilyFromCatalog(currentPrimaryFont, fontsCatalogCache, apiBaseUrl);
  }, [fontsCatalogCache, currentPrimaryFont, apiBaseUrl]);

  // Close font picker on click outside or Escape
  useEffect(() => {
    if (!isFontPickerOpen) return undefined;
    const handlePointerDown = (e) => {
      if (fontPickerRef.current && !fontPickerRef.current.contains(e.target)) {
        setIsFontPickerOpen(false);
      }
    };
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        setIsFontPickerOpen(false);
      }
    };
    window.addEventListener('mousedown', handlePointerDown);
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('mousedown', handlePointerDown);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isFontPickerOpen]);

  // Focus search input when opening font picker
  useEffect(() => {
    if (isFontPickerOpen) {
      window.setTimeout(() => fontSearchInputRef.current?.focus(), 20);
    }
  }, [isFontPickerOpen]);

  // Select an installed font family -> dynamically load via FontFace API + apply with Fallback Chain
  const handleSelectFontFamily = useCallback(
    async (familyGroupOrName, directUrl = null) => {
      if (!selectedBlock) return;
      const familyName =
        typeof familyGroupOrName === 'string'
          ? familyGroupOrName.trim()
          : familyGroupOrName?.family;
      if (!familyName) return;

      const fullKonvaStack = buildKonvaFontFamily(familyName, fallbackChain);
      onUpdateBlock(selectedBlock.id, { font_family: fullKonvaStack });
      setIsFontPickerOpen(false);
      setFontNotice(null);

      setLoadingFontFamily(familyName);
      try {
        if (directUrl) {
          await ensureFontFaceLoaded(familyName, directUrl, apiBaseUrl, {
            registerBoldAlias: true,
          });
        } else if (typeof familyGroupOrName === 'object' && familyGroupOrName) {
          await loadFontFamilyFromCatalog(
            familyGroupOrName,
            fontsCatalogCache,
            apiBaseUrl
          );
        } else {
          await loadFontFamilyFromCatalog(
            familyName,
            fontsCatalogCache,
            apiBaseUrl
          );
        }
      } finally {
        setLoadingFontFamily(null);
      }
    },
    [selectedBlock, fallbackChain, onUpdateBlock, apiBaseUrl, fontsCatalogCache]
  );

  // One-click download from online comic catalog & immediate apply to selected layer
  const handleDownloadAndApplyCatalogFont = useCallback(
    async (catalogItem) => {
      if (!catalogItem?.id || !selectedBlock) return;
      setDownloadingCatalogId(catalogItem.id);
      setFontNotice(null);
      try {
        const resp = await fetch(`${apiBaseUrl}/api/fonts/download`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'application/json',
          },
          body: JSON.stringify({ fonts: [catalogItem.id] }),
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data?.detail || `HTTP ${resp.status}`);
        const itemRes = data.results?.[0];
        if (itemRes?.status === 'failed') {
          throw new Error(itemRes.error || `Failed to download ${catalogItem.family}`);
        }
        const fontUrl = itemRes?.url || catalogItem.url;
        if (fontUrl) {
          await ensureFontFaceLoaded(catalogItem.family, fontUrl, apiBaseUrl, {
            registerBoldAlias: true,
          });
        }
        const fullStack = buildKonvaFontFamily(catalogItem.family, fallbackChain);
        onUpdateBlock(selectedBlock.id, { font_family: fullStack });
        setIsFontPickerOpen(false);
        await fetchFontsCatalog(true);
      } catch (err) {
        setFontNotice({
          type: 'error',
          text: err?.message || `Download failed for ${catalogItem.family}`,
        });
      } finally {
        setDownloadingCatalogId(null);
      }
    },
    [apiBaseUrl, fallbackChain, fetchFontsCatalog, onUpdateBlock, selectedBlock]
  );

  // Quick upload .ttf/.otf directly from the sidebar Font Picker
  const handleQuickUploadFont = useCallback(
    async (e) => {
      const file = e.target.files?.[0];
      e.target.value = '';
      if (!file) return;
      setUploadingCustomFont(true);
      setFontNotice(null);
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
          await ensureFontFaceLoaded(data.family, data.url, apiBaseUrl, {
            registerBoldAlias: true,
          });
          if (selectedBlock) {
            const fullStack = buildKonvaFontFamily(data.family, fallbackChain);
            onUpdateBlock(selectedBlock.id, { font_family: fullStack });
          }
          setIsFontPickerOpen(false);
        }
        await fetchFontsCatalog(true);
      } catch (err) {
        setFontNotice({
          type: 'error',
          text: err?.message || 'Failed to upload custom font.',
        });
      } finally {
        setUploadingCustomFont(false);
      }
    },
    [apiBaseUrl, fallbackChain, fetchFontsCatalog, onUpdateBlock, selectedBlock]
  );

  const normalizeFontQuery = (val) =>
    String(val || '').toLowerCase().replace(/[\s\-_]+/g, '');

  // Filtered installed families + online catalog items for fast searching
  const { installedResults, catalogResults } = useMemo(() => {
    const q = normalizeFontQuery(fontSearch.trim());
    const families = fontsCatalogCache?.families || [];
    const catalog = fontsCatalogCache?.catalog || [];

    // Sort custom & downloaded comic fonts ahead of system fonts
    const priorityScore = (cat) =>
      cat === 'custom' ? 0 : cat === 'downloaded' ? 1 : 2;

    const installed = families
      .filter((fam) => {
        if (fontFilterTab === 'catalog') return false;
        if (
          fontFilterTab === 'comic' &&
          fam.category !== 'custom' &&
          fam.category !== 'downloaded'
        ) {
          return false;
        }
        if (fontFilterTab === 'system' && fam.category !== 'system') {
          return false;
        }
        if (q && !normalizeFontQuery(fam.family).includes(q)) return false;
        return true;
      })
      .sort((a, b) => {
        const diff = priorityScore(a.category) - priorityScore(b.category);
        if (diff !== 0) return diff;
        return a.family.localeCompare(b.family);
      });

    const showCatalog =
      fontFilterTab === 'catalog' || (fontFilterTab === 'all' && q.length > 0);
    const catFiltered = showCatalog
      ? catalog.filter((item) => {
          if (item.installed && fontFilterTab === 'all') return false;
          if (!q) return true;
          return (
            normalizeFontQuery(item.family).includes(q) ||
            normalizeFontQuery(item.description).includes(q) ||
            normalizeFontQuery(item.style_category).includes(q)
          );
        })
      : [];

    return {
      installedResults: installed.slice(0, 60),
      catalogResults: catFiltered.slice(0, 40),
    };
  }, [fontsCatalogCache, fontSearch, fontFilterTab]);

  // Identify missing fallback fonts that aren't installed on PC
  const missingFallbackFonts = useMemo(() => {
    const families = fontsCatalogCache?.families;
    if (!Array.isArray(families) || families.length === 0) return [];
    const genericCss = new Set([
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
    const installedSet = new Set(
      families.map((f) => normalizeFontQuery(f.family))
    );
    return (fallbackChain || []).filter(
      (name) =>
        name &&
        !genericCss.has(String(name).trim().toLowerCase()) &&
        !installedSet.has(normalizeFontQuery(name))
    );
  }, [fontsCatalogCache, fallbackChain]);

  // Identify category badge for current primary font
  const currentFontCategory = useMemo(() => {
    const match = (fontsCatalogCache?.families || []).find(
      (f) => f.family?.toLowerCase() === currentPrimaryFont.toLowerCase()
    );
    return match?.category || null;
  }, [fontsCatalogCache, currentPrimaryFont]);

  const currentFontSize =
    selectedBlock?.font_size ||
    selectedBlock?.style?.font_size ||
    typesetting?.defaultFontSize ||
    18;
  const currentTextColor =
    selectedBlock?.color ||
    selectedBlock?.style?.color ||
    typesetting?.defaultTextColor ||
    '#000000';
  const currentStrokeColor =
    selectedBlock?.style?.stroke_color ||
    selectedBlock?.stroke_color ||
    typesetting?.defaultStrokeColor ||
    '#ffffff';
  const currentStrokeWidth =
    selectedBlock?.style?.stroke_width !== undefined
      ? selectedBlock.style.stroke_width
      : selectedBlock?.stroke_width !== undefined
      ? selectedBlock.stroke_width
      : typesetting?.defaultStrokeWidth ?? 4;
  const currentAlign =
    selectedBlock?.align ||
    selectedBlock?.style?.align ||
    typesetting?.defaultAlign ||
    'center';
  const currentLineHeight =
    selectedBlock?.style?.line_height ||
    selectedBlock?.line_height ||
    typesetting?.defaultLineHeight ||
    1.18;
  const currentAllCaps =
    selectedBlock?.all_caps !== undefined
      ? Boolean(selectedBlock.all_caps)
      : selectedBlock?.style?.all_caps !== undefined
      ? Boolean(selectedBlock.style.all_caps)
      : Boolean(typesetting?.allCaps);
  const currentRotation =
    typeof selectedBlock?.rotation === 'number' && Number.isFinite(selectedBlock.rotation)
      ? selectedBlock.rotation
      : typesetting?.autoTiltText &&
        typeof selectedBlock?.angle_deg === 'number' &&
        Number.isFinite(selectedBlock.angle_deg)
      ? selectedBlock.angle_deg
      : 0;

  const isSelectedLayerOcrBusy =
    busyLayerAction?.blockId === selectedBlock?.id &&
    busyLayerAction?.stage === 'ocr';
  const isSelectedLayerTranslateBusy =
    busyLayerAction?.blockId === selectedBlock?.id &&
    busyLayerAction?.stage === 'translate';
  const isLayerStageDisabled =
    isProcessing || Boolean(busyLayerAction) || !page;

  return (
    <aside className="w-72 xl:w-80 shrink-0 border-l border-studio-border bg-studio-panel flex flex-col overflow-hidden select-none">
      {/* Hidden Custom Font Upload Input */}
      <input
        ref={fontUploadInputRef}
        type="file"
        accept=".ttf,.otf,.woff2,.woff"
        onChange={handleQuickUploadFont}
        className="hidden"
        data-testid="sidebar-font-upload-input"
      />

      {/* Inspector Header */}
      <div className="h-9 px-3 border-b border-studio-border bg-studio-panel flex items-center justify-between gap-2 shrink-0">
        <div className="flex items-center gap-1.5">
          <MessageSquare className="h-3.5 w-3.5 text-studio-accent" />
          <span className="text-[11px] font-semibold uppercase tracking-wider text-studio-text">
            Text Layers
          </span>
          <span className="rounded-studio-sm border border-studio-border bg-studio-surface px-1.5 py-0.5 font-mono text-[10px] leading-none text-studio-text-secondary tabular-nums">
            {blocks.length}
          </span>
        </div>

        <div className="flex items-center gap-1">
          {onOpenFindReplace && (
            <button
              type="button"
              onClick={onOpenFindReplace}
              className="studio-btn h-6 w-6 p-0 justify-center"
              title="Find & Replace across project (Ctrl+H)"
              aria-label="Find & Replace"
            >
              <Search className="h-3 w-3 text-studio-text-secondary" />
            </button>
          )}
          {page && (
            <button
              type="button"
              onClick={onAddBlock}
              className="studio-btn h-6 px-2 text-[11px] font-medium"
              title="Add a new text layer to the canvas (T)"
            >
              <Plus className="h-3 w-3 text-studio-accent" />
              <span>Add</span>
            </button>
          )}
        </div>
      </div>

      {/* Selected Block Inspector & Typography Controls */}
      {selectedBlock ? (
        <div className="p-3 border-b border-studio-border bg-studio-panel space-y-2.5 overflow-y-auto max-h-[64%]">
          {/* Selected Bubble Header */}
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-1.5 min-w-0">
              <span className="inline-flex h-5 px-1.5 shrink-0 items-center justify-center rounded-studio-sm bg-studio-accent text-studio-accent-fg font-mono text-[10px] font-bold">
                #{selectedBlock.order}
              </span>
              <span className="text-[11px] font-medium text-studio-text-secondary truncate font-mono">
                {selectedBlock.id}
              </span>
            </div>

            <div className="flex items-center gap-1">
              {onApplyStyleToAll && (
                <button
                  type="button"
                  onClick={() =>
                    onApplyStyleToAll({
                      font_family: effectiveKonvaFontStack,
                      font_size: currentFontSize,
                      color: currentTextColor,
                      stroke_color: currentStrokeColor,
                      stroke_width: currentStrokeWidth,
                      align: currentAlign,
                      line_height: currentLineHeight,
                      all_caps: currentAllCaps,
                    })
                  }
                  className="studio-btn h-6 px-1.5 text-[10px] font-medium"
                  title="Apply this layer's font chain, typography & stroke settings to all layers on this page"
                >
                  <CopyCheck className="h-3 w-3 text-studio-accent" />
                  <span>Apply All</span>
                </button>
              )}

              <button
                type="button"
                onClick={() => onDeleteBlock && onDeleteBlock(selectedBlock.id)}
                className="inline-flex items-center justify-center h-6 w-6 rounded-studio border border-transparent hover:border-studio-danger/40 text-studio-text-muted hover:text-studio-danger hover:bg-studio-danger/10 transition-colors cursor-pointer"
                title="Delete selected text layer (Del)"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>

          {/* Per-Layer Quick Stage Actions: Re-OCR Layer & Re-Translate Layer */}
          <div className="grid grid-cols-2 gap-1.5">
            <button
              type="button"
              data-testid="layer-reocr-btn"
              disabled={isLayerStageDisabled || !onReOcrLayer}
              onClick={() => onReOcrLayer?.(selectedBlock.id)}
              title="Re-run OCR on this layer's bounding box only (/api/stage/ocr)"
              className="studio-btn h-7 px-2 text-[11px] font-medium"
            >
              {isSelectedLayerOcrBusy ? (
                <Loader2 className="h-3 w-3 animate-spin text-studio-accent shrink-0" />
              ) : (
                <ScanSearch className="h-3 w-3 text-studio-accent shrink-0" />
              )}
              <span className="truncate">
                {isSelectedLayerOcrBusy ? 'OCR…' : 'Re-OCR Layer'}
              </span>
            </button>

            <button
              type="button"
              data-testid="layer-retranslate-btn"
              disabled={
                isLayerStageDisabled ||
                !onReTranslateLayer ||
                !(selectedBlock.original_text || selectedBlock.source_text || '').trim()
              }
              onClick={() => onReTranslateLayer?.(selectedBlock.id)}
              title="Re-translate this layer using the active LLM provider (/api/stage/translate)"
              className="studio-btn h-7 px-2 text-[11px] font-medium border-studio-accent/40 hover:border-studio-accent"
            >
              {isSelectedLayerTranslateBusy ? (
                <Loader2 className="h-3 w-3 animate-spin text-studio-accent shrink-0" />
              ) : (
                <Sparkles className="h-3 w-3 text-studio-accent shrink-0" />
              )}
              <span className="truncate">
                {isSelectedLayerTranslateBusy ? 'Translating…' : 'Re-Translate'}
              </span>
            </button>
          </div>

          {/* Editable Original OCR Text */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <label
                htmlFor="sidebar-original-text"
                className="text-[11px] font-medium text-studio-text-secondary"
              >
                Source (OCR)
              </label>
              <span className="text-[10px] font-mono text-studio-text-muted">
                {selectedBlock.text_direction || 'auto'}
              </span>
            </div>
            <textarea
              id="sidebar-original-text"
              rows={2}
              value={selectedBlock.original_text ?? ''}
              onChange={(e) =>
                onUpdateBlock(
                  selectedBlock.id,
                  {
                    original_text: e.target.value,
                    source_text: e.target.value,
                  },
                  { coalesceKey: `orig:${selectedBlock.id}` }
                )
              }
              className="studio-input w-full px-2.5 py-1.5 text-xs select-text resize-none font-sans"
              placeholder="Original extracted source text..."
            />
          </div>

          {/* Editable Translated Text */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <label
                htmlFor="sidebar-translated-text"
                className="text-[11px] font-semibold text-studio-accent"
              >
                Translation
              </label>
              <span className="text-[10px] font-mono text-studio-text-muted tabular-nums">
                {(selectedBlock.translated_text || '').length} chars
              </span>
            </div>
            <textarea
              id="sidebar-translated-text"
              rows={3}
              value={selectedBlock.translated_text ?? ''}
              onChange={(e) =>
                onUpdateBlock(
                  selectedBlock.id,
                  {
                    translated_text: e.target.value,
                  },
                  { coalesceKey: `trans:${selectedBlock.id}` }
                )
              }
              className="studio-input w-full px-2.5 py-1.5 text-xs font-medium border-studio-accent/50 select-text resize-none"
              placeholder="Type translated dialogue (supports ♡ ♪ …)..."
            />
          </div>

          {/* Searchable Font Picker (/api/fonts/list + dynamic FontFace loader + Fallback Chain) */}
          <div className="space-y-1 relative" ref={fontPickerRef}>
            <div className="flex items-center justify-between gap-1">
              <span className="text-[11px] font-medium text-studio-text-secondary flex items-center gap-1">
                <Type className="h-3 w-3 text-studio-accent" />
                Font Family
              </span>
              <button
                type="button"
                onClick={() => openModal('settings', 'typesetting')}
                className="inline-flex items-center gap-1 text-[10px] font-mono text-studio-text-muted hover:text-studio-accent transition-colors cursor-pointer"
                title="Configure ordered Font Fallback Chain & comic font catalog"
              >
                <SlidersHorizontal className="h-2.5 w-2.5" />
                <span>Chain</span>
              </button>
            </div>

            {/* Picker Trigger Button */}
            <button
              type="button"
              data-testid="sidebar-font-picker-trigger"
              aria-expanded={isFontPickerOpen}
              aria-haspopup="listbox"
              onClick={() => setIsFontPickerOpen((prev) => !prev)}
              className={`studio-input w-full h-8 px-2.5 flex items-center justify-between gap-2 text-left cursor-pointer ${
                isFontPickerOpen ? 'border-studio-accent' : ''
              }`}
            >
              <span className="flex items-center gap-1.5 min-w-0">
                <span
                  className="text-xs font-semibold text-studio-text truncate"
                  style={{ fontFamily: effectiveKonvaFontStack }}
                >
                  {currentPrimaryFont}
                </span>
                {currentFontCategory && (
                  <span className="rounded-studio-sm border border-studio-border bg-studio-panel px-1 py-0.5 font-mono text-[9px] uppercase leading-none text-studio-text-muted shrink-0">
                    {currentFontCategory === 'system'
                      ? 'Win'
                      : currentFontCategory === 'downloaded'
                      ? 'DL'
                      : 'Custom'}
                  </span>
                )}
              </span>

              <span className="flex items-center gap-1 shrink-0">
                {loadingFontFamily && (
                  <Loader2 className="h-3 w-3 animate-spin text-studio-accent" />
                )}
                <ChevronDown
                  className={`h-3.5 w-3.5 text-studio-text-muted transition-transform ${
                    isFontPickerOpen ? 'rotate-180' : ''
                  }`}
                />
              </span>
            </button>

            {/* Active Fallback Chain Readout */}
            <div
              className="flex items-center justify-between gap-1 text-[10px] font-mono text-studio-text-muted px-0.5"
              title={`Konva fontFamily: ${effectiveKonvaFontStack}`}
            >
              <span className="truncate">
                Fallback: {fallbackChain.slice(0, 3).join(' → ')} → ♡♪…
              </span>
              {missingFallbackFonts.length > 0 && (
                <span
                  className="shrink-0 rounded-studio-sm border border-amber-500/40 bg-amber-500/10 text-amber-400 px-1 py-0.5 text-[9px] font-mono leading-none"
                  title={`Missing on PC (C:\\Windows\\Fonts): ${missingFallbackFonts.join(', ')}`}
                >
                  Not Installed on PC
                </span>
              )}
            </div>

            {/* Searchable Font Picker Popover */}
            {isFontPickerOpen && (
              <div
                data-testid="sidebar-font-picker-panel"
                className="mt-1 rounded-studio border border-studio-border-strong bg-studio-panel shadow-xl p-2 space-y-2 z-40"
              >
                {/* Search Input + Quick Upload / Refresh */}
                <div className="flex items-center gap-1">
                  <div className="relative flex-1">
                    <Search className="h-3 w-3 text-studio-text-muted absolute left-2 top-1/2 -translate-y-1/2 pointer-events-none" />
                    <input
                      ref={fontSearchInputRef}
                      type="text"
                      data-testid="sidebar-font-search-input"
                      value={fontSearch}
                      onChange={(e) => setFontSearch(e.target.value)}
                      placeholder="Search Windows or comic fonts…"
                      className="studio-input w-full h-7 pl-6 pr-6 text-[11px] select-text"
                    />
                    {fontSearch && (
                      <button
                        type="button"
                        onClick={() => setFontSearch('')}
                        aria-label="Clear font search"
                        className="absolute right-1.5 top-1/2 -translate-y-1/2 text-studio-text-muted hover:text-studio-text cursor-pointer"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    )}
                  </div>

                  <button
                    type="button"
                    disabled={uploadingCustomFont}
                    onClick={() => fontUploadInputRef.current?.click()}
                    title="Import local .ttf / .otf comic font"
                    className="studio-btn h-7 w-7 p-0 shrink-0"
                  >
                    {uploadingCustomFont ? (
                      <Loader2 className="h-3 w-3 animate-spin text-studio-accent" />
                    ) : (
                      <Upload className="h-3 w-3 text-studio-accent" />
                    )}
                  </button>

                  <button
                    type="button"
                    disabled={isLoadingFontsList}
                    onClick={() => fetchFontsCatalog(true)}
                    title="Rescan /api/fonts/list"
                    className="studio-btn h-7 w-7 p-0 shrink-0"
                  >
                    <RefreshCw
                      className={`h-3 w-3 ${isLoadingFontsList ? 'animate-spin' : ''}`}
                    />
                  </button>
                </div>

                {/* Source Category Tabs */}
                <div className="grid grid-cols-4 gap-0.5 rounded-studio border border-studio-border bg-studio-surface p-0.5">
                  {FONT_FILTER_TABS.map((tab) => (
                    <button
                      key={tab.id}
                      type="button"
                      onClick={() => setFontFilterTab(tab.id)}
                      className={`h-5 rounded-studio-sm text-[10px] font-medium transition-colors cursor-pointer ${
                        fontFilterTab === tab.id
                          ? 'bg-studio-accent text-studio-accent-fg font-semibold'
                          : 'text-studio-text-secondary hover:text-studio-text hover:bg-studio-hover'
                      }`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>

                {fontNotice && (
                  <div className="rounded-studio-sm border border-studio-danger/40 bg-studio-danger/10 px-2 py-1 text-[10px] text-studio-danger">
                    {fontNotice.text}
                  </div>
                )}

                {/* Scrollable Font List */}
                <div
                  role="listbox"
                  aria-label="Available Fonts"
                  className="max-h-52 overflow-y-auto space-y-1 pr-0.5"
                >
                  {isLoadingFontsList && !fontsCatalogCache ? (
                    <div className="py-5 flex items-center justify-center gap-1.5 text-[11px] text-studio-text-muted">
                      <Loader2 className="h-3.5 w-3.5 animate-spin text-studio-accent" />
                      <span>Loading fonts…</span>
                    </div>
                  ) : (
                    <>
                      {/* Installed Local / Windows System Fonts */}
                      {installedResults.map((fam) => {
                        const isActive =
                          fam.family.toLowerCase() ===
                          currentPrimaryFont.toLowerCase();
                        return (
                          <button
                            key={`${fam.category}:${fam.family}`}
                            type="button"
                            role="option"
                            aria-selected={isActive}
                            onMouseEnter={() =>
                              loadFontFamilyFromCatalog(
                                fam,
                                fontsCatalogCache,
                                apiBaseUrl
                              )
                            }
                            onClick={() => handleSelectFontFamily(fam)}
                            className={`w-full rounded-studio-sm border px-2 py-1.5 text-left transition-colors cursor-pointer flex items-center justify-between gap-2 ${
                              isActive
                                ? 'border-studio-accent bg-studio-accent/15 text-studio-text'
                                : 'border-transparent bg-studio-surface hover:border-studio-border-strong hover:bg-studio-hover text-studio-text'
                            }`}
                          >
                            <div className="min-w-0 flex-1">
                              <div
                                className="text-xs font-semibold truncate leading-tight"
                                style={{
                                  fontFamily: `"${fam.family}", sans-serif`,
                                }}
                              >
                                {fam.family}
                              </div>
                              <div
                                className="text-[10px] text-studio-text-muted truncate"
                                style={{
                                  fontFamily: buildKonvaFontFamily(
                                    fam.family,
                                    fallbackChain
                                  ),
                                }}
                              >
                                Comic Text! ♡ ♪ …
                              </div>
                            </div>

                            <div className="flex items-center gap-1 shrink-0">
                              <span className="rounded-studio-sm border border-studio-border bg-studio-panel px-1 py-0.5 font-mono text-[9px] uppercase leading-none text-studio-text-muted">
                                {fam.category === 'system'
                                  ? 'Win'
                                  : fam.category === 'downloaded'
                                  ? 'DL'
                                  : 'Custom'}
                              </span>
                              {isActive && (
                                <Check className="h-3 w-3 text-studio-accent" />
                              )}
                            </div>
                          </button>
                        );
                      })}

                      {/* Online Curated Comic Catalog Results (1-click Download & Use) */}
                      {catalogResults.length > 0 && (
                        <div className="pt-1 space-y-1">
                          <div className="px-1 text-[9px] font-mono uppercase tracking-wider text-studio-text-muted">
                            Online Comic Catalog ({catalogResults.length})
                          </div>
                          {catalogResults.map((item) => {
                            const isDownloading =
                              downloadingCatalogId === item.id;
                            return (
                              <div
                                key={item.id}
                                className="rounded-studio-sm border border-studio-border bg-studio-surface px-2 py-1.5 flex items-center justify-between gap-2"
                              >
                                <div className="min-w-0">
                                  <div className="text-[11px] font-semibold text-studio-text truncate">
                                    {item.family}
                                  </div>
                                  <div className="text-[9px] text-studio-text-muted truncate">
                                    {item.description}
                                  </div>
                                </div>

                                {item.installed ? (
                                  <button
                                    type="button"
                                    onClick={() =>
                                      handleSelectFontFamily(
                                        item.family,
                                        item.url
                                      )
                                    }
                                    className="studio-btn h-6 px-1.5 text-[10px] text-studio-success border-studio-success/40 shrink-0"
                                  >
                                    Use
                                  </button>
                                ) : (
                                  <button
                                    type="button"
                                    disabled={isDownloading}
                                    onClick={() =>
                                      handleDownloadAndApplyCatalogFont(item)
                                    }
                                    className="studio-btn h-6 px-1.5 text-[10px] shrink-0"
                                    title={`Download ${item.family} from Google Fonts & apply`}
                                  >
                                    {isDownloading ? (
                                      <Loader2 className="h-2.5 w-2.5 animate-spin" />
                                    ) : (
                                      <Download className="h-2.5 w-2.5 text-studio-accent" />
                                    )}
                                    <span>Get</span>
                                  </button>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      )}

                      {installedResults.length === 0 &&
                        catalogResults.length === 0 && (
                          <div className="py-4 text-center text-[11px] text-studio-text-muted">
                            No fonts matched "{fontSearch}". Switch to{' '}
                            <button
                              type="button"
                              onClick={() => setFontFilterTab('catalog')}
                              className="text-studio-accent underline cursor-pointer"
                            >
                              Catalog
                            </button>{' '}
                            or import a <code>.ttf/.otf</code>.
                          </div>
                        )}
                    </>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Font Size, Alignment & ALL-CAPS */}
          <div className="grid grid-cols-12 gap-2 items-end">
            <div className="col-span-6 space-y-1">
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-studio-text-secondary font-medium">Size</span>
                <span className="font-mono font-semibold text-studio-accent tabular-nums">
                  {currentFontSize}px
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <input
                  type="range"
                  min={10}
                  max={64}
                  value={currentFontSize}
                  onChange={(e) =>
                    onUpdateBlock(
                      selectedBlock.id,
                      { font_size: Number(e.target.value), manualFontSize: true },
                      { coalesceKey: `fontsize:${selectedBlock.id}` }
                    )
                  }
                  className="w-full accent-[var(--color-accent)] cursor-pointer"
                />
                <input
                  type="number"
                  min={8}
                  max={96}
                  value={currentFontSize}
                  onChange={(e) =>
                    onUpdateBlock(
                      selectedBlock.id,
                      {
                        font_size: Math.max(
                          8,
                          Math.min(96, Number(e.target.value) || 18)
                        ),
                        manualFontSize: true,
                      },
                      { coalesceKey: `fontsize:${selectedBlock.id}` }
                    )
                  }
                  className="studio-input w-11 h-7 px-1 text-center font-mono text-xs tabular-nums"
                />
              </div>
            </div>

            <div className="col-span-4 space-y-1">
              <span className="text-[11px] font-medium text-studio-text-secondary block">
                Align
              </span>
              <div className="flex h-7 rounded-studio border border-studio-border bg-studio-surface p-0.5 gap-0.5">
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
                    className={`flex-1 flex items-center justify-center rounded-studio-sm transition-colors cursor-pointer ${
                      currentAlign === value
                        ? 'bg-studio-accent text-studio-accent-fg'
                        : 'text-studio-text-muted hover:text-studio-text hover:bg-studio-hover'
                    }`}
                  >
                    <Icon className="h-3 w-3" />
                  </button>
                ))}
              </div>
            </div>

            <div className="col-span-2 space-y-1">
              <span className="text-[11px] font-medium text-studio-text-secondary block text-center">
                Caps
              </span>
              <button
                type="button"
                aria-pressed={currentAllCaps}
                onClick={() =>
                  onUpdateBlock(selectedBlock.id, {
                    all_caps: !currentAllCaps,
                    style: { all_caps: !currentAllCaps },
                  })
                }
                title="Toggle ALL-CAPS comic lettering for this layer"
                className={`w-full h-7 rounded-studio border flex items-center justify-center transition-colors cursor-pointer ${
                  currentAllCaps
                    ? 'border-studio-accent bg-studio-accent text-studio-accent-fg'
                    : 'border-studio-border bg-studio-surface text-studio-text-muted hover:text-studio-text hover:bg-studio-hover'
                }`}
              >
                <CaseUpper className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>

          {/* Line Height Slider */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono text-studio-text-muted shrink-0 w-16 tabular-nums">
              Line {Number(currentLineHeight).toFixed(2)}
            </span>
            <input
              type="range"
              aria-label="Line Height"
              min={0.85}
              max={2.0}
              step={0.02}
              value={currentLineHeight}
              onChange={(e) =>
                onUpdateBlock(
                  selectedBlock.id,
                  { line_height: Number(e.target.value) },
                  { coalesceKey: `lineheight:${selectedBlock.id}` }
                )
              }
              className="flex-1 accent-[var(--color-accent)] cursor-pointer"
            />
          </div>

          {/* Rotation Slider + Quick 0° Reset */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono text-studio-text-muted shrink-0 w-16 tabular-nums">
              Rot {Math.round(currentRotation)}°
            </span>
            <input
              type="range"
              aria-label="Rotation Angle"
              min={-45}
              max={45}
              step={1}
              value={Math.max(-45, Math.min(45, Math.round(currentRotation)))}
              onChange={(e) =>
                onUpdateBlock(
                  selectedBlock.id,
                  { rotation: Number(e.target.value) },
                  { coalesceKey: `rotation:${selectedBlock.id}` }
                )
              }
              className="flex-1 accent-[var(--color-accent)] cursor-pointer"
            />
            <button
              type="button"
              onClick={() =>
                onUpdateBlock(selectedBlock.id, { rotation: 0 })
              }
              disabled={Math.abs(currentRotation) < 0.01}
              title="Reset text rotation to 0°"
              className="px-1.5 h-5 rounded-studio-sm border border-studio-border bg-studio-surface hover:bg-studio-hover text-[10px] font-mono text-studio-text-secondary hover:text-studio-text disabled:opacity-35 disabled:pointer-events-none cursor-pointer transition-colors"
            >
              0°
            </button>
          </div>

          {/* Text Color & Outline Stroke Section */}
          <div className="rounded-studio border border-studio-border bg-studio-surface p-2 space-y-2">
            {/* Fill Color */}
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <input
                  type="color"
                  aria-label="Text Color"
                  value={currentTextColor}
                  onChange={(e) =>
                    onUpdateBlock(
                      selectedBlock.id,
                      { color: e.target.value },
                      { coalesceKey: `color:${selectedBlock.id}` }
                    )
                  }
                  className="h-6 w-6 rounded-studio-sm border border-studio-border bg-transparent cursor-pointer"
                />
                <div>
                  <div className="text-[11px] font-medium text-studio-text leading-tight">
                    Fill
                  </div>
                  <div className="font-mono text-[10px] text-studio-text-muted uppercase leading-tight">
                    {currentTextColor}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-1">
                {TEXT_COLOR_PRESETS.map((preset) => (
                  <button
                    key={preset.hex}
                    type="button"
                    onClick={() =>
                      onUpdateBlock(selectedBlock.id, { color: preset.hex })
                    }
                    title={`Text Color: ${preset.label}`}
                    className={`h-4 w-4 rounded-studio-sm border transition-colors cursor-pointer ${
                      currentTextColor.toLowerCase() === preset.hex
                        ? 'border-studio-accent ring-1 ring-studio-accent'
                        : 'border-studio-border-strong'
                    }`}
                    style={{ backgroundColor: preset.hex }}
                  />
                ))}
              </div>
            </div>

            <div className="h-px bg-studio-border" />

            {/* Outline Stroke */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <input
                    type="color"
                    aria-label="Outline Stroke Color"
                    value={currentStrokeColor}
                    onChange={(e) =>
                      onUpdateBlock(
                        selectedBlock.id,
                        { stroke_color: e.target.value },
                        { coalesceKey: `strokecolor:${selectedBlock.id}` }
                      )
                    }
                    className="h-6 w-6 rounded-studio-sm border border-studio-border bg-transparent cursor-pointer"
                  />
                  <div>
                    <div className="text-[11px] font-medium text-studio-text leading-tight">
                      Stroke
                    </div>
                    <div className="font-mono text-[10px] text-studio-text-muted leading-tight tabular-nums">
                      {currentStrokeColor.toUpperCase()} · {currentStrokeWidth}px
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
                      className={`h-4 w-4 rounded-studio-sm border transition-colors cursor-pointer ${
                        currentStrokeColor.toLowerCase() === preset.hex
                          ? 'border-studio-accent ring-1 ring-studio-accent'
                          : 'border-studio-border-strong'
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
                    title="Reset to default 4px white stroke"
                    className="ml-0.5 p-1 rounded-studio-sm hover:bg-studio-hover text-studio-text-muted hover:text-studio-text transition-colors cursor-pointer"
                  >
                    <RotateCcw className="h-3 w-3" />
                  </button>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono text-studio-text-muted shrink-0 w-16 tabular-nums">
                  Width {currentStrokeWidth}px
                </span>
                <input
                  type="range"
                  min={0}
                  max={12}
                  step={0.5}
                  value={currentStrokeWidth}
                  onChange={(e) =>
                    onUpdateBlock(
                      selectedBlock.id,
                      { stroke_width: Number(e.target.value) },
                      { coalesceKey: `strokewidth:${selectedBlock.id}` }
                    )
                  }
                  className="flex-1 accent-[var(--color-accent)] cursor-pointer"
                />
              </div>
            </div>
          </div>

          {/* Geometry Readout */}
          <div className="flex items-center justify-between text-[10px] font-mono text-studio-text-muted tabular-nums px-0.5">
            <span>
              XY ({selectedBlock.x}, {selectedBlock.y})
            </span>
            <span>
              {selectedBlock.width}×{selectedBlock.height}px
            </span>
          </div>
        </div>
      ) : (
        page &&
        blocks.length > 0 && (
          <div className="px-3 py-2 border-b border-studio-border bg-studio-surface/50 text-[11px] text-studio-text-muted">
            Select a text block on the canvas or below to edit typography & run per-layer OCR/translation.
          </div>
        )
      )}

      {/* Synchronized List of All Detected Speech Bubbles */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
        {!page ? (
          <div className="text-center py-10 px-4 text-xs text-studio-text-muted space-y-1">
            <p className="font-medium text-studio-text-secondary">No Active Page</p>
            <p className="text-[11px]">
              Import a comic page to inspect and edit dialogue layers.
            </p>
          </div>
        ) : blocks.length === 0 ? (
          <div className="text-center py-10 px-4 text-xs text-studio-text-muted">
            {page.status === 'processing'
              ? 'Detecting & translating speech bubbles…'
              : 'No text layers on this page. Click "Add" to create one.'}
          </div>
        ) : (
          blocks.map((block) => {
            const isPrimary = block.id === selectedBlockId;
            const isSelected = selectedIdSet.has(block.id);
            const blockOcrBusy =
              busyLayerAction?.blockId === block.id &&
              busyLayerAction?.stage === 'ocr';
            const blockTransBusy =
              busyLayerAction?.blockId === block.id &&
              busyLayerAction?.stage === 'translate';
            const blockPrimaryFont = extractPrimaryFontName(
              block.style?.font_family || block.font_family || typesetting?.fontFamily
            );
            const blockLeaked =
              (Array.isArray(page?.leakedBlockIds) && page.leakedBlockIds.includes(block.id)) ||
              isBlockCjkLeaked(block, page?.targetLang);
            const blockEmptyTrans =
              page?.status === 'done' &&
              Boolean(String(block.original_text || block.source_text || '').trim()) &&
              !String(block.translated_text || '').trim();
            const blockHasQcFlag = blockLeaked || blockEmptyTrans;

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
                onClick={(e) => {
                  if (e.shiftKey || e.ctrlKey || e.metaKey) {
                    toggleSelectLayer(block.id);
                  } else if (onSelectBlock) {
                    onSelectBlock(block.id);
                  }
                }}
                className={`group rounded-studio border p-2 transition-colors cursor-pointer space-y-1.5 ${
                  isPrimary
                    ? 'border-studio-accent bg-studio-accent/10'
                    : isSelected
                    ? 'border-studio-accent/60 bg-studio-accent/5'
                    : 'border-studio-border bg-studio-surface hover:border-studio-border-strong'
                }`}
              >
                <div className="flex items-center justify-between gap-1 text-[11px]">
                  <div className="flex items-center gap-1.5 min-w-0">
                    <span
                      className={`inline-flex items-center justify-center h-4 px-1.5 rounded-studio-sm font-mono text-[10px] font-bold shrink-0 ${
                        isSelected
                          ? 'bg-studio-accent text-studio-accent-fg'
                          : 'bg-studio-hover text-studio-text-secondary'
                      }`}
                    >
                      #{block.order}
                    </span>
                    <span className="font-mono text-studio-text-muted text-[10px] truncate">
                      {blockPrimaryFont}
                    </span>
                    {blockHasQcFlag && (
                      <span
                        title={
                          blockLeaked
                            ? 'Untranslated CJK/source characters detected'
                            : 'Empty translation'
                        }
                        className="inline-flex items-center gap-0.5 rounded-studio-sm bg-studio-warning/15 border border-studio-warning/40 px-1 py-0.2 text-[9px] font-mono text-studio-warning shrink-0"
                      >
                        <AlertTriangle className="h-2.5 w-2.5" />
                        {blockLeaked ? 'CJK' : 'Empty'}
                      </span>
                    )}
                  </div>

                  {/* Quick Hover Per-Layer Stage Triggers + Size Readout */}
                  <div className="flex items-center gap-1 shrink-0">
                    <button
                      type="button"
                      disabled={isLayerStageDisabled || !onReOcrLayer}
                      onClick={(e) => {
                        e.stopPropagation();
                        onReOcrLayer?.(block.id);
                      }}
                      title="Re-OCR this layer (/api/stage/ocr)"
                      className={`h-5 w-5 inline-flex items-center justify-center rounded-studio-sm hover:bg-studio-hover text-studio-text-muted hover:text-studio-accent transition-opacity cursor-pointer disabled:opacity-30 ${
                        blockOcrBusy || isPrimary
                          ? 'opacity-100'
                          : 'opacity-0 group-hover:opacity-100'
                      }`}
                    >
                      {blockOcrBusy ? (
                        <Loader2 className="h-3 w-3 animate-spin text-studio-accent" />
                      ) : (
                        <ScanSearch className="h-3 w-3" />
                      )}
                    </button>
                    <button
                      type="button"
                      disabled={
                        isLayerStageDisabled ||
                        !onReTranslateLayer ||
                        !(block.original_text || block.source_text || '').trim()
                      }
                      onClick={(e) => {
                        e.stopPropagation();
                        onReTranslateLayer?.(block.id);
                      }}
                      title="Re-Translate this layer (/api/stage/translate)"
                      className={`h-5 w-5 inline-flex items-center justify-center rounded-studio-sm hover:bg-studio-hover text-studio-text-muted hover:text-studio-accent transition-opacity cursor-pointer disabled:opacity-30 ${
                        blockTransBusy || isPrimary
                          ? 'opacity-100'
                          : 'opacity-0 group-hover:opacity-100'
                      }`}
                    >
                      {blockTransBusy ? (
                        <Loader2 className="h-3 w-3 animate-spin text-studio-accent" />
                      ) : (
                        <Sparkles className="h-3 w-3" />
                      )}
                    </button>
                    <span className="font-mono text-[10px] text-studio-text-muted tabular-nums">
                      {block.font_size}px
                    </span>
                  </div>
                </div>

                {/* Source OCR Text Preview */}
                <div className="text-[11px] text-studio-text-secondary bg-studio-panel border border-studio-border rounded-studio-sm px-2 py-1 font-sans select-text break-words">
                  <span className="text-[9px] font-mono uppercase text-studio-text-muted mr-1.5">
                    OCR
                  </span>
                  {block.original_text || (
                    <span className="italic text-studio-text-muted">Empty</span>
                  )}
                </div>

                {/* Translated Text Preview */}
                <div className="text-xs font-medium text-studio-text select-text break-words px-0.5">
                  {block.translated_text || (
                    <span className="italic text-studio-text-muted">
                      Empty translation
                    </span>
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
