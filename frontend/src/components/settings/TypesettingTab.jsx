import {
  AlertCircle,
  ArrowDown,
  ArrowUp,
  Check,
  Download,
  Loader2,
  Plus,
  RefreshCw,
  RotateCcw,
  Search,
  Sparkles,
  Trash2,
  Upload,
} from 'lucide-react';
import {
  DEFAULT_FONT_FALLBACK_CHAIN,
  ensureFontFaceLoaded,
  formatFontFamilyChain,
} from '../../store/useEditorStore.js';

import {
  FONT_STYLE_FILTERS,
  ToggleSwitch,
  SectionCard,
} from './SettingsPrimitives.jsx';

export default function TypesettingTab({ ctx }) {
  const {
    typesetting,
    hasActivePage,
    onApplyTypesettingToPage,
    apiBaseUrl,
    draftTypesetting,
    setDraftTypesetting,
    fontsData,
    isLoadingFonts,
    fontsError,
    fontSearch,
    setFontSearch,
    fontSourceTab,
    setFontSourceTab,
    fontStyleFilter,
    setFontStyleFilter,
    downloadingFontId,
    uploadingFont,
    customChainFontInput,
    setCustomChainFontInput,
    fontUploadInputRef,
    fetchFontsList,
    handleDownloadCatalogFont,
    handleUploadCustomFont,
    setChainPrimaryFont,
    appendChainFont,
    moveChainItem,
    removeChainItem,
    normalizeFontQuery,
    filteredInstalledFamilies,
    filteredCatalogItems,
    installedFontSet,
  } = ctx;

  return (
              <div className="space-y-4" data-testid="settings-panel-typesetting">
                {/* Ordered Font Fallback Chain + Live Speech Bubble Preview */}
                <SectionCard
                  title="Ordered Font Fallback Chain"
                  subtitle="Browser & Konva try each font in order from #1 downward when rendering Latin, accented, or CJK glyphs."
                  action={
                    <button
                      type="button"
                      onClick={() =>
                        setDraftTypesetting((prev) => ({
                          ...prev,
                          fontFallbackChain: [...DEFAULT_FONT_FALLBACK_CHAIN],
                          fontFamily: formatFontFamilyChain(
                            DEFAULT_FONT_FALLBACK_CHAIN
                          ),
                        }))
                      }
                      className="studio-btn h-6 px-2 text-[10px]"
                    >
                      <RotateCcw className="h-3 w-3" />
                      <span>Default Chain</span>
                    </button>
                  }
                >
                  <div className="grid grid-cols-1 lg:grid-cols-12 gap-3.5">
                    {/* Chain List */}
                    <div className="lg:col-span-7 space-y-2">
                      <div className="space-y-1">
                        {(draftTypesetting.fontFallbackChain || []).map(
                          (fontName, idx, arr) => (
                            <div
                              key={`${fontName}-${idx}`}
                              className="flex items-center justify-between gap-2 rounded-studio border border-studio-border bg-studio-panel px-2.5 py-1.5"
                            >
                              <div className="flex items-center gap-2 min-w-0">
                                <span className="h-4 min-w-[20px] px-1 rounded-studio-sm bg-studio-accent/15 text-studio-accent font-mono text-[10px] font-bold inline-flex items-center justify-center shrink-0">
                                  {idx + 1}
                                </span>
                                <span
                                  className="text-xs font-medium text-studio-text truncate"
                                  style={{ fontFamily: `"${fontName}", sans-serif` }}
                                >
                                  {fontName}
                                </span>
                                {idx === 0 && (
                                  <span className="rounded-studio-sm bg-studio-accent text-studio-accent-fg px-1 py-0.5 text-[9px] font-mono font-bold uppercase leading-none">
                                    Primary
                                  </span>
                                )}
                                {installedFontSet &&
                                  !['sans-serif', 'serif', 'monospace', 'cursive', 'fantasy', 'system-ui', 'ui-sans-serif', 'ui-serif', 'ui-monospace'].includes(
                                    String(fontName).trim().toLowerCase()
                                  ) &&
                                  !installedFontSet.has(normalizeFontQuery(fontName)) && (
                                    <span
                                      className="rounded-studio-sm border border-amber-500/40 bg-amber-500/10 text-amber-400 px-1.5 py-0.5 text-[9px] font-mono font-medium leading-none shrink-0"
                                      title={`"${fontName}" was not found in C:\\Windows\\Fonts or local custom fonts`}
                                    >
                                      Not Installed on PC
                                    </span>
                                  )}
                              </div>

                              <div className="flex items-center gap-0.5 shrink-0">
                                <button
                                  type="button"
                                  disabled={idx === 0}
                                  onClick={() => moveChainItem(idx, -1)}
                                  title="Move up"
                                  className="h-6 w-6 inline-flex items-center justify-center rounded-studio-sm text-studio-text-muted hover:text-studio-text hover:bg-studio-hover disabled:opacity-30 cursor-pointer"
                                >
                                  <ArrowUp className="h-3 w-3" />
                                </button>
                                <button
                                  type="button"
                                  disabled={idx === arr.length - 1}
                                  onClick={() => moveChainItem(idx, 1)}
                                  title="Move down"
                                  className="h-6 w-6 inline-flex items-center justify-center rounded-studio-sm text-studio-text-muted hover:text-studio-text hover:bg-studio-hover disabled:opacity-30 cursor-pointer"
                                >
                                  <ArrowDown className="h-3 w-3" />
                                </button>
                                <button
                                  type="button"
                                  disabled={arr.length <= 1}
                                  onClick={() => removeChainItem(idx)}
                                  title="Remove from chain"
                                  className="h-6 w-6 inline-flex items-center justify-center rounded-studio-sm text-studio-text-muted hover:text-studio-danger hover:bg-studio-danger/10 disabled:opacity-30 cursor-pointer"
                                >
                                  <Trash2 className="h-3 w-3" />
                                </button>
                              </div>
                            </div>
                          )
                        )}
                      </div>

                      {/* Manual Add Font to Fallback Chain */}
                      <div className="flex items-center gap-1.5">
                        <input
                          type="text"
                          value={customChainFontInput}
                          onChange={(e) => setCustomChainFontInput(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' && customChainFontInput.trim()) {
                              e.preventDefault();
                              appendChainFont(customChainFontInput.trim());
                              setCustomChainFontInput('');
                            }
                          }}
                          placeholder="Add font family name (e.g. CC Wild Words)..."
                          className="studio-input flex-1 h-7 px-2 text-xs select-text"
                        />
                        <button
                          type="button"
                          disabled={!customChainFontInput.trim()}
                          onClick={() => {
                            appendChainFont(customChainFontInput.trim());
                            setCustomChainFontInput('');
                          }}
                          className="studio-btn h-7 px-2.5 text-xs"
                        >
                          <Plus className="h-3 w-3" />
                          <span>Add</span>
                        </button>
                      </div>

                      <div className="text-[10px] font-mono text-studio-text-muted bg-studio-canvas/70 border border-studio-border rounded-studio-sm px-2 py-1 truncate">
                        font-family: {draftTypesetting.fontFamily}
                      </div>
                    </div>

                    {/* Live Bubble Preview */}
                    <div className="lg:col-span-5 rounded-studio border border-studio-border bg-studio-canvas p-3 flex flex-col items-center justify-center gap-2 min-h-[150px]">
                      <div className="text-[10px] font-mono uppercase tracking-wider text-studio-text-muted">
                        Live Bubble Preview
                      </div>
                      <div className="w-full max-w-[240px] rounded-[28px] border-2 border-neutral-900 bg-white px-4 py-3 text-center shadow-sm">
                        <div
                          style={{
                            fontFamily: draftTypesetting.fontFamily,
                            fontSize: `${Math.min(28, Math.max(12, draftTypesetting.defaultFontSize || 18))}px`,
                            lineHeight: draftTypesetting.defaultLineHeight || 1.18,
                            color: draftTypesetting.defaultTextColor || '#000000',
                            textTransform: draftTypesetting.allCaps
                              ? 'uppercase'
                              : 'none',
                            WebkitTextStroke:
                              (draftTypesetting.defaultStrokeWidth || 0) > 0
                                ? `${Math.min(2, (draftTypesetting.defaultStrokeWidth || 0) * 0.25)}px ${
                                    draftTypesetting.defaultStrokeColor || '#ffffff'
                                  }`
                                : undefined,
                            fontWeight: 700,
                          }}
                        >
                          What are you doing?! This place is dangerous!
                        </div>
                      </div>
                      {hasActivePage && onApplyTypesettingToPage && (
                        <button
                          type="button"
                          onClick={() =>
                            onApplyTypesettingToPage({
                              font_family: draftTypesetting.fontFamily,
                              font_size: draftTypesetting.defaultFontSize,
                              color: draftTypesetting.defaultTextColor,
                              stroke_color: draftTypesetting.defaultStrokeColor,
                              stroke_width: draftTypesetting.defaultStrokeWidth,
                              line_height: draftTypesetting.defaultLineHeight,
                              align: draftTypesetting.defaultAlign,
                              all_caps: draftTypesetting.allCaps,
                            })
                          }
                          className="studio-btn h-6 px-2.5 text-[10px]"
                        >
                          <Sparkles className="h-3 w-3 text-studio-accent" />
                          <span>Apply Defaults to Active Page</span>
                        </button>
                      )}
                    </div>
                  </div>
                </SectionCard>

                {/* Default Typography & Lettering Options */}
                <SectionCard
                  title="Default Lettering Style"
                  subtitle="Defaults applied to newly detected speech bubbles and manual text layers."
                >
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
                    {/* Default Font Size */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-studio-text-secondary font-medium">
                          Default Size
                        </span>
                        <span className="font-mono font-semibold text-studio-accent tabular-nums">
                          {draftTypesetting.defaultFontSize}px
                        </span>
                      </div>
                      <input
                        type="range"
                        aria-label="Default Font Size"
                        min={10}
                        max={48}
                        step={1}
                        value={draftTypesetting.defaultFontSize ?? 18}
                        onChange={(e) =>
                          setDraftTypesetting((prev) => ({
                            ...prev,
                            defaultFontSize: Number(e.target.value),
                          }))
                        }
                        className="w-full accent-[var(--color-accent)] cursor-pointer"
                      />
                    </div>

                    {/* Stroke Width */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-studio-text-secondary font-medium">
                          Stroke Width
                        </span>
                        <span className="font-mono font-semibold text-studio-accent tabular-nums">
                          {Number(draftTypesetting.defaultStrokeWidth ?? 4).toFixed(1)}px
                        </span>
                      </div>
                      <input
                        type="range"
                        aria-label="Default Stroke Width"
                        min={0}
                        max={12}
                        step={0.5}
                        value={draftTypesetting.defaultStrokeWidth ?? 4}
                        onChange={(e) =>
                          setDraftTypesetting((prev) => ({
                            ...prev,
                            defaultStrokeWidth: Number(e.target.value),
                          }))
                        }
                        className="w-full accent-[var(--color-accent)] cursor-pointer"
                      />
                    </div>

                    {/* Line Height */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-studio-text-secondary font-medium">
                          Line Height
                        </span>
                        <span className="font-mono font-semibold text-studio-accent tabular-nums">
                          {Number(draftTypesetting.defaultLineHeight ?? 1.18).toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        aria-label="Default Line Height"
                        min={0.9}
                        max={2.0}
                        step={0.02}
                        value={draftTypesetting.defaultLineHeight ?? 1.18}
                        onChange={(e) =>
                          setDraftTypesetting((prev) => ({
                            ...prev,
                            defaultLineHeight: Number(e.target.value),
                          }))
                        }
                        className="w-full accent-[var(--color-accent)] cursor-pointer"
                      />
                    </div>

                    {/* Text & Stroke Color */}
                    <div className="space-y-1">
                      <span className="text-[11px] text-studio-text-secondary font-medium block">
                        Fill / Stroke Color
                      </span>
                      <div className="flex items-center gap-2">
                        <label className="inline-flex items-center gap-1.5 text-[10px] font-mono text-studio-text-muted cursor-pointer">
                          <input
                            type="color"
                            aria-label="Default Text Fill Color"
                            value={draftTypesetting.defaultTextColor || '#000000'}
                            onChange={(e) =>
                              setDraftTypesetting((prev) => ({
                                ...prev,
                                defaultTextColor: e.target.value,
                              }))
                            }
                            className="h-6 w-6 rounded-studio-sm border border-studio-border bg-transparent cursor-pointer"
                          />
                          Fill
                        </label>
                        <label className="inline-flex items-center gap-1.5 text-[10px] font-mono text-studio-text-muted cursor-pointer">
                          <input
                            type="color"
                            aria-label="Default Text Stroke Color"
                            value={draftTypesetting.defaultStrokeColor || '#ffffff'}
                            onChange={(e) =>
                              setDraftTypesetting((prev) => ({
                                ...prev,
                                defaultStrokeColor: e.target.value,
                              }))
                            }
                            className="h-6 w-6 rounded-studio-sm border border-studio-border bg-transparent cursor-pointer"
                          />
                          Stroke
                        </label>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 border-t border-studio-border">
                    <div className="flex items-center justify-between gap-2">
                      <div>
                        <label
                          htmlFor="ts-auto-fit"
                          className="text-xs font-medium text-studio-text block"
                        >
                          Auto-Fit Font Size to Bubble
                        </label>
                        <p className="text-[10px] text-studio-text-muted">
                          Scales dialogue size automatically inside each bounding box.
                        </p>
                      </div>
                      <ToggleSwitch
                        id="ts-auto-fit"
                        label="Auto-Fit Font Size"
                        checked={Boolean(draftTypesetting.autoFitFontSize)}
                        onChange={(val) =>
                          setDraftTypesetting((prev) => ({
                            ...prev,
                            autoFitFontSize: val,
                          }))
                        }
                      />
                    </div>

                    <div className="flex items-center justify-between gap-2">
                      <div>
                        <label
                          htmlFor="ts-all-caps"
                          className="text-xs font-medium text-studio-text block"
                        >
                          ALL-CAPS Comic Lettering
                        </label>
                        <p className="text-[10px] text-studio-text-muted">
                          Renders dialogue text in classic uppercase manga lettering.
                        </p>
                      </div>
                      <ToggleSwitch
                        id="ts-all-caps"
                        label="ALL-CAPS Toggle"
                        checked={Boolean(draftTypesetting.allCaps)}
                        onChange={(val) =>
                          setDraftTypesetting((prev) => ({
                            ...prev,
                            allCaps: val,
                          }))
                        }
                      />
                    </div>

                    <div className="flex items-center justify-between gap-2 sm:col-span-2 pt-1 border-t border-studio-border/60">
                      <div>
                        <label
                          htmlFor="ts-enable-hyphenation"
                          className="text-xs font-medium text-studio-text block"
                        >
                          Narrow-Bubble Hyphenation
                        </label>
                        <p className="text-[10px] text-studio-text-muted">
                          Allows syllable hyphenation on Latin words (≥6 letters) as a last resort in narrow vertical bubbles.
                        </p>
                      </div>
                      <ToggleSwitch
                        id="ts-enable-hyphenation"
                        label="Narrow-Bubble Hyphenation"
                        checked={draftTypesetting.enableHyphenation !== false}
                        onChange={(val) =>
                          setDraftTypesetting((prev) => ({
                            ...prev,
                            enableHyphenation: val,
                          }))
                        }
                      />
                    </div>

                    <div className="flex items-center justify-between gap-2 sm:col-span-2 pt-1 border-t border-studio-border/60">
                      <div>
                        <label
                          htmlFor="ts-auto-tilt-text"
                          className="text-xs font-medium text-studio-text block"
                        >
                          Auto-Tilt Text Boxes
                        </label>
                        <p className="text-[10px] text-studio-text-muted">
                          Automatically rotate typeset text to match tilted manga speech bubbles or slanted SFX. Keep off for straight horizontal lettering.
                        </p>
                      </div>
                      <ToggleSwitch
                        id="ts-auto-tilt-text"
                        label="Auto-Tilt Text Boxes"
                        checked={Boolean(draftTypesetting.autoTiltText)}
                        onChange={(val) =>
                          setDraftTypesetting((prev) => ({
                            ...prev,
                            autoTiltText: val,
                          }))
                        }
                      />
                    </div>
                  </div>
                </SectionCard>

                {/* Searchable Font Picker across /api/fonts/list + Catalog Downloader + Upload */}
                <SectionCard
                  title="Font Picker & Comic Catalog"
                  subtitle={
                    fontsData?.counts
                      ? `${fontsData.counts.families} installed families (${fontsData.counts.system} system, ${fontsData.counts.custom} custom, ${fontsData.counts.downloaded} downloaded) · ${fontsData.counts.catalog} online comic fonts`
                      : 'Browse Windows system fonts, download offline comic fonts, or import .ttf/.otf files.'
                  }
                  action={
                    <div className="flex items-center gap-1.5">
                      <input
                        ref={fontUploadInputRef}
                        type="file"
                        accept=".ttf,.otf,.woff2,.woff"
                        onChange={handleUploadCustomFont}
                        className="hidden"
                        data-testid="font-upload-input"
                      />
                      <button
                        type="button"
                        disabled={uploadingFont}
                        onClick={() => fontUploadInputRef.current?.click()}
                        className="studio-btn-accent h-7 px-2.5 text-[11px]"
                      >
                        {uploadingFont ? (
                          <Loader2 className="h-3 w-3 animate-spin" />
                        ) : (
                          <Upload className="h-3 w-3" />
                        )}
                        <span>Import .ttf/.otf</span>
                      </button>
                      <button
                        type="button"
                        disabled={isLoadingFonts}
                        onClick={() => fetchFontsList(true)}
                        title="Rescan system and local fonts"
                        className="studio-btn h-7 w-7 p-0"
                      >
                        <RefreshCw
                          className={`h-3.5 w-3.5 ${isLoadingFonts ? 'animate-spin' : ''}`}
                        />
                      </button>
                    </div>
                  }
                >
                  {/* Search & Source Filter Bar */}
                  <div className="flex flex-col sm:flex-row gap-2">
                    <div className="relative flex-1">
                      <Search className="h-3.5 w-3.5 text-studio-text-muted absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
                      <input
                        type="text"
                        value={fontSearch}
                        onChange={(e) => setFontSearch(e.target.value)}
                        placeholder="Search Windows fonts or 129 comic/CJK catalog fonts..."
                        className="studio-input w-full h-8 pl-8 pr-2.5 text-xs select-text"
                      />
                    </div>

                    <div className="inline-flex rounded-studio border border-studio-border bg-studio-panel p-0.5 gap-0.5 shrink-0 overflow-x-auto">
                      {[
                        { id: 'all', label: 'All' },
                        { id: 'installed', label: 'Installed' },
                        { id: 'system', label: 'Windows' },
                        { id: 'custom', label: 'Custom' },
                        { id: 'catalog', label: 'Online Catalog' },
                      ].map((tab) => (
                        <button
                          key={tab.id}
                          type="button"
                          onClick={() => setFontSourceTab(tab.id)}
                          className={`h-7 px-2 rounded-studio-sm text-[11px] font-medium transition-colors cursor-pointer shrink-0 ${
                            fontSourceTab === tab.id
                              ? 'bg-studio-accent text-studio-accent-fg font-semibold'
                              : 'text-studio-text-secondary hover:text-studio-text hover:bg-studio-hover'
                          }`}
                        >
                          {tab.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Catalog Style Sub-Filters */}
                  {(fontSourceTab === 'all' || fontSourceTab === 'catalog') && (
                    <div className="flex flex-wrap gap-1">
                      {FONT_STYLE_FILTERS.map((st) => (
                        <button
                          key={st.id}
                          type="button"
                          onClick={() => setFontStyleFilter(st.id)}
                          className={`px-2 py-0.5 rounded-studio-sm border text-[10px] font-medium transition-colors cursor-pointer ${
                            fontStyleFilter === st.id
                              ? 'border-studio-accent bg-studio-accent/15 text-studio-accent'
                              : 'border-studio-border bg-studio-panel text-studio-text-muted hover:text-studio-text'
                          }`}
                        >
                          {st.label}
                        </button>
                      ))}
                    </div>
                  )}

                  {fontsError && (
                    <div className="rounded-studio border border-studio-danger/40 bg-studio-danger/10 px-2.5 py-2 text-xs text-studio-danger flex items-center gap-2">
                      <AlertCircle className="h-3.5 w-3.5 shrink-0" />
                      <span>{fontsError}</span>
                    </div>
                  )}

                  {isLoadingFonts && !fontsData ? (
                    <div className="py-8 flex items-center justify-center gap-2 text-xs text-studio-text-muted">
                      <Loader2 className="h-4 w-4 animate-spin text-studio-accent" />
                      <span>Scanning Windows & local fonts…</span>
                    </div>
                  ) : (
                    <div className="space-y-3 max-h-64 overflow-y-auto pr-1">
                      {/* Installed Families Section */}
                      {filteredInstalledFamilies.length > 0 && (
                        <div className="space-y-1.5">
                          <div className="text-[10px] font-mono uppercase tracking-wider text-studio-text-muted">
                            Installed Fonts ({filteredInstalledFamilies.length})
                          </div>
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                            {filteredInstalledFamilies.slice(0, 80).map((fam) => {
                              const inChain = (
                                draftTypesetting.fontFallbackChain || []
                              ).some(
                                (f) => f.toLowerCase() === fam.family.toLowerCase()
                              );
                              return (
                                <div
                                  key={`${fam.category}-${fam.family}`}
                                  onMouseEnter={() => {
                                    if (fam.url) {
                                      ensureFontFaceLoaded(
                                        fam.family,
                                        fam.url,
                                        apiBaseUrl
                                      );
                                    }
                                  }}
                                  className="rounded-studio border border-studio-border bg-studio-panel px-2.5 py-2 flex items-center justify-between gap-2 hover:border-studio-border-strong transition-colors"
                                >
                                  <div className="min-w-0">
                                    <div
                                      className="text-xs font-semibold text-studio-text truncate"
                                      style={{
                                        fontFamily: `"${fam.family}", sans-serif`,
                                      }}
                                    >
                                      {fam.family}
                                    </div>
                                    <div className="text-[10px] font-mono text-studio-text-muted flex items-center gap-1.5">
                                      <span className="uppercase">
                                        {fam.category}
                                      </span>
                                      <span>·</span>
                                      <span>
                                        {fam.variants?.length || 1} style
                                        {(fam.variants?.length || 1) === 1 ? '' : 's'}
                                      </span>
                                    </div>
                                  </div>

                                  <div className="flex items-center gap-1 shrink-0">
                                    <button
                                      type="button"
                                      onClick={() =>
                                        setChainPrimaryFont(fam.family, fam.url)
                                      }
                                      className="studio-btn h-6 px-1.5 text-[10px]"
                                      title="Set as #1 Primary Font"
                                    >
                                      Primary
                                    </button>
                                    <button
                                      type="button"
                                      disabled={inChain}
                                      onClick={() =>
                                        appendChainFont(fam.family, fam.url)
                                      }
                                      className="studio-btn h-6 px-1.5 text-[10px]"
                                      title="Append to Fallback Chain"
                                    >
                                      {inChain ? 'In Chain' : '+ Chain'}
                                    </button>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {/* Online Curated Comic Catalog Section */}
                      {filteredCatalogItems.length > 0 && (
                        <div className="space-y-1.5">
                          <div className="text-[10px] font-mono uppercase tracking-wider text-studio-text-muted">
                            Online Comic Font Catalog ({filteredCatalogItems.length})
                          </div>
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                            {filteredCatalogItems.slice(0, 80).map((item) => {
                              const isDownloading = downloadingFontId === item.id;
                              return (
                                <div
                                  key={item.id}
                                  className="rounded-studio border border-studio-border bg-studio-panel px-2.5 py-2 flex items-center justify-between gap-2"
                                >
                                  <div className="min-w-0">
                                    <div className="flex items-center gap-1.5">
                                      <span className="text-xs font-semibold text-studio-text truncate">
                                        {item.family}
                                      </span>
                                      <span className="rounded-studio-sm border border-studio-border bg-studio-surface px-1 py-0.5 font-mono text-[9px] uppercase leading-none text-studio-text-muted shrink-0">
                                        {item.style_category}
                                      </span>
                                    </div>
                                    <div className="text-[10px] text-studio-text-muted truncate mt-0.5">
                                      {item.description}
                                    </div>
                                  </div>

                                  {item.installed ? (
                                    <button
                                      type="button"
                                      onClick={() =>
                                        setChainPrimaryFont(item.family, item.url)
                                      }
                                      className="studio-btn h-6 px-2 text-[10px] text-studio-success border-studio-success/40 shrink-0"
                                    >
                                      <Check className="h-3 w-3" />
                                      <span>Use</span>
                                    </button>
                                  ) : (
                                    <button
                                      type="button"
                                      disabled={isDownloading}
                                      onClick={() =>
                                        handleDownloadCatalogFont(item)
                                      }
                                      className="studio-btn h-6 px-2 text-[10px] shrink-0"
                                    >
                                      {isDownloading ? (
                                        <Loader2 className="h-3 w-3 animate-spin" />
                                      ) : (
                                        <Download className="h-3 w-3 text-studio-accent" />
                                      )}
                                      <span>Download</span>
                                    </button>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {filteredInstalledFamilies.length === 0 &&
                        filteredCatalogItems.length === 0 && (
                          <div className="py-6 text-center text-xs text-studio-text-muted">
                            No fonts matched "{fontSearch}". Try another search or import a local <code>.ttf</code>/<code>.otf</code> file.
                          </div>
                        )}
                    </div>
                  )}
                </SectionCard>
              </div>
  );
}
