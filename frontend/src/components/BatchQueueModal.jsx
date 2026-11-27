import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ArrowDown,
  ArrowUp,
  Download,
  FileArchive,
  FolderOpen,
  FolderSearch,
  Layers,
  Loader2,
  Pencil,
  Play,
  Square,
  Trash2,
  X,
} from 'lucide-react';

const STATUS_BADGES = {
  queued: 'border-studio-border bg-studio-surface text-studio-text-muted',
  importing: 'border-studio-accent/40 bg-studio-accent/10 text-studio-accent',
  processing: 'border-studio-accent/50 bg-studio-accent/15 text-studio-accent',
  ready: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-400',
  exporting: 'border-studio-accent/40 bg-studio-accent/10 text-studio-accent',
  done: 'border-studio-success/40 bg-studio-success/10 text-studio-success',
  skipped: 'border-studio-border bg-studio-surface text-studio-text-secondary',
  error: 'border-studio-danger/40 bg-studio-danger/10 text-studio-danger',
};

function stripArchiveExt(name) {
  return String(name || '').replace(/\.(cbz|zip|cbr|rar|pdf)$/i, '');
}

export default function BatchQueueModal({
  isOpen,
  onClose,
  apiBaseUrl,
  queueItems,
  setQueueItems,
  runMode,
  setRunMode,
  exportFormat,
  setExportFormat,
  outputDir,
  setOutputDir,
  filenamePattern,
  setFilenamePattern,
  skipExisting,
  setSkipExisting,
  purgeCacheBetween,
  setPurgeCacheBetween,
  shareSeriesContinuity,
  setShareSeriesContinuity,
  activeProjectId,
  isQueueRunning,
  onStartQueue,
  onStopQueue,
  onEditItem,
  onBatchExportSelected,
  onClearAndPurgeCaches,
}) {
  const [profiles, setProfiles] = useState([]);
  const [busyAdd, setBusyAdd] = useState(false);

  const base = String(apiBaseUrl || '').replace(/\/+$/, '');

  const fetchProfiles = useCallback(async () => {
    try {
      const resp = await fetch(`${base}/api/series-profiles`, {
        headers: { Accept: 'application/json' },
      });
      if (!resp.ok) return;
      const data = await resp.json();
      if (Array.isArray(data?.profiles)) {
        setProfiles(data.profiles);
      }
    } catch {
      // ignore
    }
  }, [base]);

  useEffect(() => {
    if (isOpen) {
      fetchProfiles();
    }
  }, [isOpen, fetchProfiles]);

  const matchProfileForTitle = useCallback(
    async (title) => {
      try {
        const resp = await fetch(`${base}/api/series-profiles/match`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title }),
        });
        if (!resp.ok) return '';
        const data = await resp.json();
        return data?.matched && data?.profile?.slug ? data.profile.slug : '';
      } catch {
        return '';
      }
    },
    [base]
  );

  const enqueueEntries = useCallback(
    async (rawItems) => {
      if (!rawItems || rawItems.length === 0) return;
      const existingPaths = new Set(queueItems.map((q) => q.path));
      const newOnes = [];
      for (let i = 0; i < rawItems.length; i += 1) {
        const raw = rawItems[i];
        if (!raw?.path || existingPaths.has(raw.path)) continue;
        existingPaths.add(raw.path);
        const title = raw.name || raw.path.split(/[\\/]/).pop() || `Item_${i + 1}`;
        const matchedSlug = await matchProfileForTitle(title);
        const uid = `q_${Date.now()}_${i}_${Math.random().toString(36).slice(2, 6)}`;
        newOnes.push({
          id: uid,
          projectId: `qproj_${uid}`,
          kind: raw.kind === 'folder' ? 'folder' : 'archive',
          name: title,
          stem: stripArchiveExt(title),
          path: raw.path,
          pageCount: raw.page_count ?? null,
          profileSlug: matchedSlug,
          status: 'queued',
          progressText: 'Queued',
          exportStatus: null,
          selected: true,
          error: null,
        });
      }
      if (newOnes.length > 0) {
        setQueueItems((prev) => [...prev, ...newOnes]);
      }
    },
    [queueItems, matchProfileForTitle, setQueueItems]
  );

  const pickPaths = async (mode, title) => {
    const resp = await fetch(`${base}/api/system/pick-paths`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode, ...(title ? { title } : {}) }),
    });
    if (!resp.ok) return [];
    const data = await resp.json();
    if (data?.cancelled || !Array.isArray(data?.paths)) return [];
    return data.paths;
  };

  const handleAddCbzFiles = async () => {
    if (busyAdd || isQueueRunning) return;
    setBusyAdd(true);
    try {
      const paths = await pickPaths('files');
      await enqueueEntries(
        paths.map((p) => ({
          kind: 'archive',
          name: p.split(/[\\/]/).pop() || p,
          path: p,
        }))
      );
    } finally {
      setBusyAdd(false);
    }
  };

  const handleAddFolder = async () => {
    if (busyAdd || isQueueRunning) return;
    setBusyAdd(true);
    try {
      const paths = await pickPaths('folder', 'Select Manga Folder');
      if (paths[0]) {
        await enqueueEntries([
          {
            kind: 'folder',
            name: paths[0].split(/[\\/]/).pop() || paths[0],
            path: paths[0],
          },
        ]);
      }
    } finally {
      setBusyAdd(false);
    }
  };

  const handleScanRoot = async () => {
    if (busyAdd || isQueueRunning) return;
    setBusyAdd(true);
    try {
      const paths = await pickPaths('folder', 'Select Root Folder to Scan');
      if (!paths[0]) return;
      const resp = await fetch(`${base}/api/import/scan-batch-root`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder: paths[0] }),
      });
      if (!resp.ok) return;
      const data = await resp.json();
      if (Array.isArray(data?.items)) {
        await enqueueEntries(data.items);
      }
    } finally {
      setBusyAdd(false);
    }
  };

  const handlePickOutputDir = async () => {
    if (isQueueRunning) return;
    const paths = await pickPaths('folder', 'Select Output Folder');
    if (paths[0]) {
      setOutputDir(paths[0]);
    }
  };

  const handleClickBatchExport = async () => {
    if (isQueueRunning) return;
    const exportable = queueItems.filter(
      (q) => q.selected !== false && (q.status === 'ready' || q.status === 'done')
    );
    if (exportable.length === 0) return;

    const paths = await pickPaths('folder', 'Save Batch Export To Folder');
    if (!paths[0]) return; // user cancelled cleanly
    setOutputDir(paths[0]);
    onBatchExportSelected?.(exportable, exportFormat, paths[0]);
  };

  const moveItem = (idx, delta) => {
    const target = idx + delta;
    if (target < 0 || target >= queueItems.length) return;
    setQueueItems((prev) => {
      const next = [...prev];
      const [item] = next.splice(idx, 1);
      next.splice(target, 0, item);
      return next;
    });
  };

  const toggleSelectAll = (checked) => {
    setQueueItems((prev) => prev.map((q) => ({ ...q, selected: checked })));
  };

  const checkedItems = useMemo(
    () => queueItems.filter((q) => q.selected !== false),
    [queueItems]
  );

  const stats = useMemo(() => {
    let ready = 0;
    let done = 0;
    let skipped = 0;
    let error = 0;
    let selectedCount = 0;
    let selectedExportable = 0;
    let allSelected = queueItems.length > 0;
    for (const q of queueItems) {
      if (q.status === 'ready') ready += 1;
      else if (q.status === 'done') done += 1;
      else if (q.status === 'skipped') skipped += 1;
      else if (q.status === 'error') error += 1;
      if (q.selected === false) {
        allSelected = false;
      } else {
        selectedCount += 1;
        if (q.status === 'ready' || q.status === 'done') {
          selectedExportable += 1;
        }
      }
    }
    return {
      ready,
      done,
      skipped,
      error,
      selectedCount,
      selectedExportable,
      allSelected,
      someSelected: selectedCount > 0 && selectedCount < queueItems.length,
      total: queueItems.length,
    };
  }, [queueItems]);

  const canStartQueue = !isQueueRunning && stats.selectedCount > 0;

  const handleClickStartQueue = async () => {
    if (!canStartQueue) return;
    const selectedIdList = checkedItems.map((q) => q.id);
    if (runMode === 'auto_export' && !outputDir.trim()) {
      const paths = await pickPaths('folder', 'Select Auto-Export Output Folder');
      if (!paths[0]) return; // user cancelled cleanly
      setOutputDir(paths[0]);
      onStartQueue?.(profiles, selectedIdList, paths[0]);
      return;
    }
    onStartQueue?.(profiles, selectedIdList);
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onClick={() => !isQueueRunning && onClose?.()}
    >
      <div
        className="w-full max-w-5xl max-h-[88vh] flex flex-col rounded-studio-md border border-studio-border bg-studio-panel shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="px-4 py-3 border-b border-studio-border flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-studio-accent" />
            <div>
              <h2 className="text-xs font-semibold uppercase tracking-wider text-studio-text">
                Batch Series Queue (Multi-Manga Editable & Overnight Pipeline)
              </h2>
              <p className="text-[11px] text-studio-text-muted">
                Queue multiple .cbz archives or folders, process 1-by-1, edit any title in Vellum, and batch-export to .cbz / .pdf.
              </p>
            </div>
          </div>
          <button
            type="button"
            disabled={isQueueRunning}
            onClick={onClose}
            className="p-1 rounded-studio-sm text-studio-text-muted hover:text-studio-text hover:bg-studio-hover disabled:opacity-40 cursor-pointer"
            aria-label="Close Batch Queue"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Source Buttons + Mode/Format Controls */}
        <div className="px-4 py-3 border-b border-studio-border bg-studio-surface/50 space-y-2.5 shrink-0">
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              disabled={busyAdd || isQueueRunning}
              onClick={handleAddCbzFiles}
              className="studio-btn h-7 px-2.5 text-[11px] font-medium"
            >
              <FileArchive className="h-3.5 w-3.5 text-studio-accent" />
              <span>Add .cbz Files</span>
            </button>
            <button
              type="button"
              disabled={busyAdd || isQueueRunning}
              onClick={handleAddFolder}
              className="studio-btn h-7 px-2.5 text-[11px] font-medium"
            >
              <FolderOpen className="h-3.5 w-3.5 text-studio-accent" />
              <span>Add Manga Folder</span>
            </button>
            <button
              type="button"
              disabled={busyAdd || isQueueRunning}
              onClick={handleScanRoot}
              className="studio-btn h-7 px-2.5 text-[11px] font-medium"
            >
              {busyAdd ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin text-studio-accent" />
              ) : (
                <FolderSearch className="h-3.5 w-3.5 text-studio-accent" />
              )}
              <span>Scan Root Folder</span>
            </button>

            <div className="ml-auto flex items-center gap-2 text-[11px] font-mono text-studio-text-secondary">
              <span>
                {stats.ready + stats.done}/{stats.total} processed
                {stats.ready > 0 ? ` · ${stats.ready} ready` : ''}
                {stats.done > 0 ? ` · ${stats.done} exported` : ''}
                {stats.skipped > 0 ? ` · ${stats.skipped} skipped` : ''}
                {stats.error > 0 ? ` · ${stats.error} err` : ''}
              </span>
            </div>
          </div>

          {/* Run Mode + Export Format Segmented Controls */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-1.5">
              <span className="text-[11px] text-studio-text-secondary shrink-0">
                Run Mode:
              </span>
              <div className="inline-flex rounded-studio-sm border border-studio-border bg-studio-panel p-0.5">
                {[
                  { id: 'review', label: 'Process & Keep Editable' },
                  { id: 'auto_export', label: 'Process & Auto-Export' },
                ].map((opt) => (
                  <button
                    key={opt.id}
                    type="button"
                    disabled={isQueueRunning}
                    onClick={() => setRunMode(opt.id)}
                    className={`h-6 px-2.5 rounded-studio-sm text-[11px] font-medium transition-colors cursor-pointer ${
                      runMode === opt.id
                        ? 'bg-studio-accent/20 text-studio-accent border border-studio-accent/40'
                        : 'text-studio-text-secondary hover:text-studio-text border border-transparent'
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center gap-1.5">
              <span className="text-[11px] text-studio-text-secondary shrink-0">
                Format:
              </span>
              <div className="inline-flex rounded-studio-sm border border-studio-border bg-studio-panel p-0.5">
                {[
                  { id: 'cbz', label: 'CBZ' },
                  { id: 'pdf', label: 'PDF' },
                  { id: 'both', label: 'CBZ + PDF' },
                ].map((opt) => (
                  <button
                    key={opt.id}
                    type="button"
                    disabled={isQueueRunning}
                    onClick={() => setExportFormat(opt.id)}
                    className={`h-6 px-2.5 rounded-studio-sm text-[11px] font-medium transition-colors cursor-pointer ${
                      exportFormat === opt.id
                        ? 'bg-studio-accent/20 text-studio-accent border border-studio-accent/40'
                        : 'text-studio-text-secondary hover:text-studio-text border border-transparent'
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-12 gap-2 items-center">
            <div className="md:col-span-7 flex items-center gap-1.5">
              <span className="text-[11px] text-studio-text-secondary shrink-0">
                Output Folder:
              </span>
              <input
                type="text"
                value={outputDir}
                onChange={(e) => setOutputDir(e.target.value)}
                disabled={isQueueRunning}
                placeholder={
                  runMode === 'review'
                    ? 'Optional in Review Mode (chosen when you click Batch Export Selected)…'
                    : 'Required for Auto-Export (select destination folder)…'
                }
                className="studio-input flex-1 h-7 px-2 text-[11px] font-mono"
              />
              <button
                type="button"
                disabled={isQueueRunning}
                onClick={handlePickOutputDir}
                className="studio-btn h-7 px-2 text-[11px] shrink-0"
              >
                Browse…
              </button>
            </div>
            <div className="md:col-span-5 flex items-center gap-1.5">
              <span className="text-[11px] text-studio-text-secondary shrink-0">
                Name Pattern:
              </span>
              <input
                type="text"
                value={filenamePattern}
                onChange={(e) => setFilenamePattern(e.target.value)}
                disabled={isQueueRunning}
                placeholder="{name}"
                className="studio-input flex-1 h-7 px-2 text-[11px] font-mono"
              />
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-4 text-[11px] text-studio-text-secondary">
            {runMode === 'auto_export' && (
              <>
                <label className="inline-flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={skipExisting}
                    onChange={(e) => setSkipExisting(e.target.checked)}
                    disabled={isQueueRunning}
                    className="accent-[var(--color-accent)]"
                  />
                  Skip if output already exists (Resume Queue)
                </label>
                <label className="inline-flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={purgeCacheBetween}
                    onChange={(e) => setPurgeCacheBetween(e.target.checked)}
                    disabled={isQueueRunning}
                    className="accent-[var(--color-accent)]"
                  />
                  Purge temp image cache between titles
                </label>
              </>
            )}
            <label className="inline-flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={shareSeriesContinuity}
                onChange={(e) => setShareSeriesContinuity(e.target.checked)}
                disabled={isQueueRunning}
                className="accent-[var(--color-accent)]"
              />
              Share series continuity across volumes of same profile
            </label>
          </div>
        </div>

        {/* Queue Rows */}
        <div className="flex-1 overflow-y-auto p-3 space-y-1.5">
          {queueItems.length > 0 && (
            <div className="flex items-center justify-between px-2.5 pb-1 text-[10px] text-studio-text-muted">
              <label className="inline-flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={stats.allSelected}
                  ref={(el) => {
                    if (el) el.indeterminate = stats.someSelected;
                  }}
                  onChange={(e) => toggleSelectAll(e.target.checked)}
                  disabled={isQueueRunning}
                  className="accent-[var(--color-accent)]"
                />
                Select all ({stats.selectedCount}/{stats.total} selected for process & export)
              </label>
              <span>Click Edit on any row to open it in the Vellum editor</span>
            </div>
          )}

          {queueItems.length === 0 ? (
            <div className="rounded-studio border border-dashed border-studio-border p-10 text-center space-y-1.5">
              <Layers className="h-6 w-6 text-studio-text-muted mx-auto" />
              <p className="text-xs font-medium text-studio-text-secondary">
                Queue is empty
              </p>
              <p className="text-[11px] text-studio-text-muted">
                Use &ldquo;Add .cbz Files&rdquo;, &ldquo;Add Manga Folder&rdquo;, or &ldquo;Scan Root Folder&rdquo; to batch-queue titles.
              </p>
            </div>
          ) : (
            queueItems.map((item, idx) => {
              const isActiveInEditor =
                Boolean(activeProjectId) && item.projectId === activeProjectId;
              return (
                <div
                  key={item.id}
                  className={`rounded-studio border px-2.5 py-2 flex items-center gap-2 text-[11px] ${
                    isActiveInEditor
                      ? 'border-studio-accent/60 bg-studio-accent/5'
                      : 'border-studio-border bg-studio-surface'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={item.selected !== false}
                    disabled={isQueueRunning}
                    onChange={(e) => {
                      const checked = e.target.checked;
                      setQueueItems((prev) =>
                        prev.map((q) => (q.id === item.id ? { ...q, selected: checked } : q))
                      );
                    }}
                    className="accent-[var(--color-accent)] shrink-0 cursor-pointer"
                    title="Include in Batch Export"
                  />

                  <span className="font-mono text-[10px] text-studio-text-muted w-6 shrink-0 tabular-nums">
                    {String(idx + 1).padStart(2, '0')}
                  </span>

                  <span className="rounded-studio-sm border border-studio-border bg-studio-panel px-1.5 py-0.5 font-mono text-[9px] uppercase text-studio-text-secondary shrink-0">
                    {item.kind === 'folder' ? 'Folder' : 'CBZ'}
                  </span>

                  <div className="min-w-0 flex-1">
                    <div className="font-medium text-studio-text truncate flex items-center gap-1.5" title={item.path}>
                      <span className="truncate">{item.name}</span>
                      {isActiveInEditor && (
                        <span className="rounded bg-studio-accent/20 text-studio-accent px-1 py-0.2 text-[9px] font-mono shrink-0">
                          OPEN
                        </span>
                      )}
                    </div>
                    <div className="text-[10px] text-studio-text-muted truncate font-mono">
                      {item.path}
                    </div>
                  </div>

                  {/* Per-Row Profile Selector */}
                  <select
                    value={item.profileSlug || ''}
                    disabled={isQueueRunning}
                    onChange={(e) => {
                      const nextSlug = e.target.value;
                      setQueueItems((prev) =>
                        prev.map((q) => (q.id === item.id ? { ...q, profileSlug: nextSlug } : q))
                      );
                    }}
                    className="studio-input h-6 px-1.5 text-[11px] w-36 shrink-0 cursor-pointer"
                    title="Series Profile for this title"
                  >
                    <option value="">No Profile</option>
                    {profiles.map((p) => (
                      <option key={p.slug} value={p.slug}>
                        {p.name}
                      </option>
                    ))}
                  </select>

                  {/* Status Badge */}
                  <span
                    title={item.error || item.progressText}
                    className={`rounded-studio-sm border px-2 py-0.5 font-mono text-[10px] shrink-0 max-w-[140px] truncate ${
                      STATUS_BADGES[item.status] || STATUS_BADGES.queued
                    }`}
                  >
                    {item.progressText || item.status}
                  </span>

                  {/* Per-Row Edit Button */}
                  {!isQueueRunning && (
                    <button
                      type="button"
                      onClick={() => onEditItem?.(item, profiles)}
                      className="studio-btn h-6 px-2 text-[10px] font-medium shrink-0 inline-flex items-center gap-1 cursor-pointer"
                      title="Open this manga in the Vellum editor"
                    >
                      <Pencil className="h-3 w-3 text-studio-accent" />
                      <span>Edit</span>
                    </button>
                  )}

                  {/* Reorder & Remove */}
                  {!isQueueRunning && (
                    <div className="flex items-center gap-0.5 shrink-0">
                      <button
                        type="button"
                        disabled={idx === 0}
                        onClick={() => moveItem(idx, -1)}
                        className="p-1 rounded hover:bg-studio-hover text-studio-text-muted hover:text-studio-text disabled:opacity-30 cursor-pointer"
                        title="Move up"
                      >
                        <ArrowUp className="h-3 w-3" />
                      </button>
                      <button
                        type="button"
                        disabled={idx === queueItems.length - 1}
                        onClick={() => moveItem(idx, 1)}
                        className="p-1 rounded hover:bg-studio-hover text-studio-text-muted hover:text-studio-text disabled:opacity-30 cursor-pointer"
                        title="Move down"
                      >
                        <ArrowDown className="h-3 w-3" />
                      </button>
                      <button
                        type="button"
                        onClick={() =>
                          setQueueItems((prev) => prev.filter((q) => q.id !== item.id))
                        }
                        className="p-1 rounded hover:bg-studio-hover text-studio-text-muted hover:text-studio-danger cursor-pointer"
                        title="Remove from queue"
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>

        {/* Footer Actions */}
        <div className="px-4 py-2.5 border-t border-studio-border bg-studio-panel flex flex-wrap items-center justify-between gap-2 shrink-0">
          <div className="flex items-center gap-2">
            {queueItems.length > 0 && !isQueueRunning && (
              <button
                type="button"
                onClick={() => onClearAndPurgeCaches?.()}
                className="studio-btn h-7 px-2.5 text-[11px] text-studio-danger border-studio-danger/30 hover:bg-studio-danger/10 inline-flex items-center gap-1 cursor-pointer"
                title="Clear queue list and delete saved project caches from disk"
              >
                <Trash2 className="h-3 w-3" />
                <span>Clear Queue & Purge Saved Caches</span>
              </button>
            )}
          </div>

          <div className="flex items-center gap-2">
            {isQueueRunning ? (
              <button
                type="button"
                onClick={onStopQueue}
                className="h-7 px-3 rounded-studio-sm bg-red-600 hover:bg-red-500 text-white text-xs font-medium inline-flex items-center gap-1.5 cursor-pointer"
              >
                <Square className="h-3 w-3 fill-current" />
                <span>Stop Queue</span>
              </button>
            ) : (
              <>
                <button
                  type="button"
                  onClick={onClose}
                  className="studio-btn h-7 px-3 text-xs"
                >
                  Close
                </button>
                <button
                  type="button"
                  disabled={stats.selectedExportable === 0}
                  onClick={handleClickBatchExport}
                  className="studio-btn h-7 px-3 text-xs font-medium inline-flex items-center gap-1.5 disabled:opacity-40 cursor-pointer"
                  title="Pick one destination folder and export all selected Ready/Exported manga"
                >
                  <Download className="h-3.5 w-3.5 text-studio-accent" />
                  <span>
                    Batch Export Selected ({stats.selectedExportable})…
                  </span>
                </button>
                <button
                  type="button"
                  disabled={!canStartQueue}
                  onClick={handleClickStartQueue}
                  title={
                    stats.selectedCount === 0
                      ? 'Check at least 1 manga row above to process'
                      : runMode === 'auto_export' && !outputDir.trim()
                        ? 'Will prompt for Output Folder, then process & export selected manga'
                        : 'Process selected manga'
                  }
                  className="studio-btn-accent h-7 px-3 text-xs font-medium inline-flex items-center gap-1.5 disabled:opacity-40 cursor-pointer"
                >
                  <Play className="h-3 w-3 fill-current" />
                  <span>
                    {stats.selectedCount === 0
                      ? 'Select at least 1 manga'
                      : runMode === 'review'
                        ? `Process & Keep Editable Selected (${stats.selectedCount})`
                        : `Process & Auto-Export Selected (${stats.selectedCount})`}
                  </span>
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
