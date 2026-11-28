import React, { useCallback, useEffect, useState } from 'react';
import { Activity, Loader2, RotateCcw, X } from 'lucide-react';

const RATE_STORAGE_KEY = 'vellum.telemetry.ratePer1M';

function formatInt(n) {
  const v = Number(n) || 0;
  return v.toLocaleString();
}

export default function TelemetryPopover({
  isOpen,
  onClose,
  apiBaseUrl,
  provider = 'none',
}) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [ratePer1M, setRatePer1M] = useState(() => {
    try {
      const saved = localStorage.getItem(RATE_STORAGE_KEY);
      return saved !== null ? saved : '0.15';
    } catch {
      return '0.15';
    }
  });

  const fetchTelemetry = useCallback(async () => {
    if (!apiBaseUrl) return;
    setLoading(true);
    setError('');
    try {
      const resp = await fetch(`${apiBaseUrl}/api/telemetry/llm`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const json = await resp.json();
      setData(json);
    } catch (err) {
      setError(err?.message || 'Failed to load LLM telemetry');
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    if (isOpen) {
      fetchTelemetry();
    }
  }, [isOpen, fetchTelemetry]);

  if (!isOpen) return null;

  const isLocal =
    provider === 'lmstudio' || provider === 'ollama' || provider === 'none';
  const promptTokens = Number(data?.prompt_tokens ?? data?.input_tokens ?? 0);
  const completionTokens = Number(
    data?.completion_tokens ?? data?.output_tokens ?? 0
  );
  const totalTokens = Number(
    data?.total_tokens ?? promptTokens + completionTokens
  );
  const kvHitPct = Number(
    data?.kv_cache_hit_pct ?? data?.cache_hit_pct ?? 0
  );
  const pagesPerMin = Number(data?.pages_per_min ?? 0);
  const numericRate = Math.max(0, Number(ratePer1M) || 0);
  const estimatedCost = isLocal ? 0 : (totalTokens / 1_000_000) * numericRate;

  const handleRateChange = (val) => {
    setRatePer1M(val);
    try {
      localStorage.setItem(RATE_STORAGE_KEY, val);
    } catch {
      // ignore storage errors
    }
  };

  const handleReset = async () => {
    try {
      await fetch(`${apiBaseUrl}/api/telemetry/llm/reset`, { method: 'POST' });
      await fetchTelemetry();
    } catch (err) {
      setError(err?.message || 'Reset failed');
    }
  };

  return (
    <div
      className="fixed bottom-8 right-3 z-50 w-72 rounded-studio-md border border-studio-border bg-studio-panel shadow-2xl p-3 text-xs text-studio-text select-none"
      role="dialog"
      aria-label="LLM Telemetry"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="flex items-center justify-between gap-2 pb-2 border-b border-studio-border">
        <div className="flex items-center gap-1.5 font-semibold text-studio-text">
          <Activity className="h-3.5 w-3.5 text-studio-accent" />
          <span>LLM Telemetry</span>
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={handleReset}
            title="Reset telemetry counters"
            className="h-6 px-1.5 rounded-studio-sm hover:bg-studio-hover text-studio-text-muted hover:text-studio-text inline-flex items-center gap-1 text-[10px] cursor-pointer"
          >
            <RotateCcw className="h-3 w-3" />
            <span>Reset</span>
          </button>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close telemetry"
            className="h-6 w-6 rounded-studio-sm hover:bg-studio-hover text-studio-text-muted hover:text-studio-text inline-flex items-center justify-center cursor-pointer"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {loading && !data ? (
        <div className="py-5 flex items-center justify-center gap-2 text-studio-text-muted">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          <span>Loading telemetry…</span>
        </div>
      ) : error ? (
        <div className="py-3 text-[11px] text-studio-danger">{error}</div>
      ) : (
        <div className="pt-2.5 space-y-2 font-mono text-[11px]">
          <div className="flex items-center justify-between">
            <span className="text-studio-text-muted">Prompt Tokens</span>
            <span className="tabular-nums font-medium">{formatInt(promptTokens)}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-studio-text-muted">Completion Tokens</span>
            <span className="tabular-nums font-medium">{formatInt(completionTokens)}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-studio-text-muted">KV-Cache Hit</span>
            <span className="tabular-nums font-semibold text-studio-accent">
              {kvHitPct.toFixed(1)}%
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-studio-text-muted">Throughput</span>
            <span className="tabular-nums">{pagesPerMin.toFixed(2)} pages/min</span>
          </div>

          <div className="pt-2 border-t border-studio-border space-y-1.5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-studio-text-muted">Rate ($/1M tok)</span>
              {isLocal ? (
                <span className="text-studio-success font-semibold">$0.00 Local</span>
              ) : (
                <input
                  type="number"
                  min="0"
                  step="0.01"
                  value={ratePer1M}
                  onChange={(e) => handleRateChange(e.target.value)}
                  className="studio-input w-20 h-6 px-1.5 text-right text-[11px]"
                />
              )}
            </div>
            <div className="flex items-center justify-between">
              <span className="text-studio-text-muted">Est. Session Cost</span>
              <span className="tabular-nums font-semibold text-studio-text">
                {isLocal ? '$0.00 Local' : `$${estimatedCost.toFixed(4)}`}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
