import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Download, HardDrive, Loader2, Trash2, Upload } from 'lucide-react';
import { SectionCard } from './SettingsPrimitives.jsx';

function formatBytes(bytes) {
  const b = Number(bytes) || 0;
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1024 * 1024 * 1024) return `${(b / (1024 * 1024)).toFixed(1)} MB`;
  return `${(b / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export default function StorageAboutTab({ ctx }) {
  const { apiBaseUrl } = ctx;
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [cleaning, setCleaning] = useState(false);
  const [backupBusy, setBackupBusy] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const importInputRef = useRef(null);

  const loadStats = useCallback(async () => {
    if (!apiBaseUrl) return;
    setLoading(true);
    try {
      const resp = await fetch(`${apiBaseUrl}/api/storage/stats`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const json = await resp.json();
      setStats(json);
    } catch (err) {
      setFeedback({ type: 'error', text: err?.message || 'Failed to inspect disk usage' });
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    loadStats();
  }, [loadStats]);

  const handleCleanup = async () => {
    if (!window.confirm('Delete cached images and project files older than 7 days?')) return;
    setCleaning(true);
    setFeedback(null);
    try {
      const resp = await fetch(`${apiBaseUrl}/api/storage/cleanup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_age_days: 7 }),
      });
      if (!resp.ok) throw new Error(`Cleanup HTTP ${resp.status}`);
      const json = await resp.json();
      setFeedback({
        type: 'success',
        text: `Cleaned ${json.deleted_files ?? 0} files (${formatBytes(json.freed_bytes ?? 0)} freed).`,
      });
      await loadStats();
    } catch (err) {
      setFeedback({ type: 'error', text: err?.message || 'Cache cleanup failed' });
    } finally {
      setCleaning(false);
    }
  };

  const handleExportBackup = async () => {
    setBackupBusy(true);
    setFeedback(null);
    try {
      const resp = await fetch(`${apiBaseUrl}/api/backup/export?include_keys=false`);
      if (!resp.ok) throw new Error(`Backup export HTTP ${resp.status}`);
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `vellum_backup_${new Date().toISOString().slice(0, 10)}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      setFeedback({ type: 'success', text: 'Portable backup exported (API keys excluded).' });
    } catch (err) {
      setFeedback({ type: 'error', text: err?.message || 'Backup export failed' });
    } finally {
      setBackupBusy(false);
    }
  };

  const handleImportBackup = async (e) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    setBackupBusy(true);
    setFeedback(null);
    try {
      const formData = new FormData();
      formData.append('file', file, file.name);
      const resp = await fetch(`${apiBaseUrl}/api/backup/import`, {
        method: 'POST',
        body: formData,
      });
      if (!resp.ok) throw new Error(`Backup import HTTP ${resp.status}`);
      setFeedback({ type: 'success', text: 'Backup imported successfully. Restart or reload settings to apply.' });
      await loadStats();
    } catch (err) {
      setFeedback({ type: 'error', text: err?.message || 'Backup import failed' });
    } finally {
      setBackupBusy(false);
    }
  };

  const buckets = [
    ['Cache', stats?.cache_bytes ?? 0],
    ['Projects Cache', stats?.projects_cache_bytes ?? 0],
    ['Models', stats?.models_bytes ?? 0],
    ['Release', stats?.release_bytes ?? 0],
  ];
  const totalBytes = Math.max(
    1,
    stats?.total_bytes ?? buckets.reduce((acc, [, v]) => acc + Number(v || 0), 0)
  );

  return (
    <div className="space-y-4" data-testid="settings-panel-storage">
      <div className="rounded-studio border border-studio-border bg-studio-panel px-3 py-2.5 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-studio-text">Vellum</span>
          <span className="px-1.5 py-0.5 rounded bg-studio-accent/15 text-studio-accent font-mono text-[10px] font-semibold">
            v3.0.0
          </span>
        </div>
        <span className="text-[11px] font-mono text-studio-text-muted">
          Native C++17 + ONNX Runtime · Zero Python
        </span>
      </div>

      <SectionCard
        title="Disk Usage Meter"
        subtitle="Inspect local Vellum storage footprint and prune stale temporary cache files."
      >
        {loading && !stats ? (
          <div className="py-4 flex items-center gap-2 text-xs text-studio-text-muted">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            <span>Scanning disk usage…</span>
          </div>
        ) : (
          <div className="space-y-3">
            {buckets.map(([label, val]) => {
              const pct = Math.min(100, Math.round(((Number(val) || 0) / totalBytes) * 100));
              return (
                <div key={label} className="space-y-1 text-xs">
                  <div className="flex items-center justify-between font-mono text-[11px]">
                    <span className="text-studio-text-secondary flex items-center gap-1.5">
                      <HardDrive className="h-3 w-3 text-studio-text-muted" />
                      {label}
                    </span>
                    <span className="text-studio-text tabular-nums">{formatBytes(val)}</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-studio-surface overflow-hidden border border-studio-border/60">
                    <div
                      className="h-full bg-studio-accent transition-all"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}

            <div className="pt-2 flex items-center justify-between gap-2 border-t border-studio-border">
              <span className="text-[11px] font-mono text-studio-text-muted">
                Total: {formatBytes(stats?.total_bytes ?? totalBytes)}
              </span>
              <button
                type="button"
                disabled={cleaning}
                onClick={handleCleanup}
                className="studio-btn h-7 px-2.5 text-xs gap-1.5 text-amber-300"
              >
                {cleaning ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                <span>Clean Cache &gt; 7 Days</span>
              </button>
            </div>
          </div>
        )}
      </SectionCard>

      <SectionCard
        title="Portable Backup"
        subtitle="Export or restore settings, series profiles, and glossaries as a single .zip archive (API keys excluded by default)."
      >
        <input
          ref={importInputRef}
          type="file"
          accept=".zip"
          onChange={handleImportBackup}
          className="hidden"
        />
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            disabled={backupBusy}
            onClick={handleExportBackup}
            className="studio-btn-accent h-8 px-3 text-xs gap-1.5"
          >
            <Download className="h-3.5 w-3.5" />
            <span>Export Backup (.zip)</span>
          </button>
          <button
            type="button"
            disabled={backupBusy}
            onClick={() => importInputRef.current?.click()}
            className="studio-btn h-8 px-3 text-xs gap-1.5"
          >
            <Upload className="h-3.5 w-3.5" />
            <span>Import Backup (.zip)</span>
          </button>
        </div>
      </SectionCard>

      {feedback && (
        <div
          className={`text-xs px-3 py-2 rounded-studio border ${
            feedback.type === 'error'
              ? 'border-studio-danger/40 bg-studio-danger/10 text-studio-danger'
              : 'border-studio-success/40 bg-studio-success/10 text-studio-success'
          }`}
        >
          {feedback.text}
        </div>
      )}
    </div>
  );
}
