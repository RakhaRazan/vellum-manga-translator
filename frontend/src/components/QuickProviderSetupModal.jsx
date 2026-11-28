import React, { useEffect, useState } from 'react';
import {
  CheckCircle2,
  Cloud,
  Cpu,
  Eye,
  EyeOff,
  Loader2,
  PenTool,
  Server,
  Sliders,
  X,
} from 'lucide-react';
import useEditorStore, {
  DEFAULT_API_BASE_URL,
  DEFAULT_PROVIDER_PRESETS,
} from '../store/useEditorStore.js';

const CATEGORY_CARDS = [
  {
    id: 'cloud',
    title: 'Cloud API',
    subtitle: 'OpenAI, Gemini, Anthropic, DeepSeek, OpenRouter',
    icon: Cloud,
    providers: ['openai', 'gemini', 'anthropic', 'deepseek', 'openrouter'],
  },
  {
    id: 'local',
    title: 'Local LLM',
    subtitle: 'LM Studio (:1234) or Ollama (:11434)',
    icon: Cpu,
    providers: ['lmstudio', 'ollama'],
  },
  {
    id: 'custom',
    title: 'Custom OpenAI-Compatible',
    subtitle: 'Any self-hosted or proxy /v1 endpoint',
    icon: Server,
    providers: ['custom'],
  },
  {
    id: 'manual',
    title: 'Manual / No-LLM Mode',
    subtitle: 'Detect + Inpaint only (manual lettering)',
    icon: PenTool,
    providers: ['none'],
  },
];

/**
 * Provider-agnostic Quick Provider Setup card opened when `active_provider === "none"`
 * and the user triggers Translate. Never assumes LM Studio is required.
 */
export default function QuickProviderSetupModal({
  isOpen,
  onClose,
  onConfiguredTranslate,
  onRunManualNoLlm,
  onOpenFullSettings,
}) {
  const pipelineConfig = useEditorStore((s) => s.pipelineConfig);
  const setActiveProvider = useEditorStore((s) => s.setActiveProvider);
  const updateProviderProfile = useEditorStore((s) => s.updateProviderProfile);
  const setActiveModel = useEditorStore((s) => s.setActiveModel);
  const togglePipelineStage = useEditorStore((s) => s.togglePipelineStage);
  const persistSettingsToBackend = useEditorStore((s) => s.persistSettingsToBackend);

  const apiBaseUrl = pipelineConfig?.apiBaseUrl ?? DEFAULT_API_BASE_URL;

  const [category, setCategory] = useState('cloud');
  const [providerId, setProviderId] = useState('openai');
  const [baseUrl, setBaseUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [fetchedModels, setFetchedModels] = useState([]);
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState(null);
  const [isSaving, setIsSaving] = useState(false);

  const loadProviderDraft = (pid) => {
    const preset = DEFAULT_PROVIDER_PRESETS.find((p) => p.id === pid);
    const saved = pipelineConfig?.providerProfiles?.[pid] || {};
    setProviderId(pid);
    setBaseUrl(saved.base_url ?? preset?.default_base_url ?? '');
    setApiKey(saved.api_key ?? '');
    setModel(saved.model ?? preset?.default_model ?? '');
    setFetchedModels([]);
    setTestResult(null);
  };

  useEffect(() => {
    if (!isOpen) return;
    setCategory('cloud');
    loadProviderDraft('openai');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  if (!isOpen) return null;

  const activeCategory =
    CATEGORY_CARDS.find((c) => c.id === category) || CATEGORY_CARDS[0];
  const preset = DEFAULT_PROVIDER_PRESETS.find((p) => p.id === providerId);

  const handleCategorySelect = (cat) => {
    setCategory(cat.id);
    if (cat.id === 'manual') {
      setProviderId('none');
      setTestResult(null);
      return;
    }
    loadProviderDraft(cat.providers[0]);
  };

  const handleTestAndFetch = async () => {
    if (providerId === 'none') return;
    setIsTesting(true);
    setTestResult(null);
    try {
      const resp = await fetch(`${apiBaseUrl}/api/providers/models`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({
          provider: providerId,
          base_url: baseUrl.trim(),
          api_key: apiKey,
        }),
      });
      const data = await resp.json().catch(() => ({}));
      const models = Array.isArray(data.models) ? data.models : [];
      setFetchedModels(models);
      if (data.ok) {
        if (!model && models.length > 0) {
          setModel(models[0]);
        }
        setTestResult({
          ok: true,
          message: `Connected (${models.length} model${models.length === 1 ? '' : 's'} found)`,
        });
      } else {
        setTestResult({
          ok: false,
          message: data.error || data.message || data.detail || 'Connection failed',
        });
      }
    } catch (err) {
      setTestResult({
        ok: false,
        message: err?.message || 'Unable to reach provider endpoint',
      });
    } finally {
      setIsTesting(false);
    }
  };

  const handleSaveAndTranslate = async () => {
    if (category === 'manual' || providerId === 'none') {
      setActiveProvider('none', { syncBackend: true });
      if (pipelineConfig?.enabledStages?.translate) {
        togglePipelineStage('translate');
      }
      onClose();
      onRunManualNoLlm?.();
      return;
    }

    setIsSaving(true);
    try {
      updateProviderProfile(providerId, {
        base_url: baseUrl.trim(),
        api_key: apiKey,
        model: model.trim(),
      });
      setActiveProvider(providerId, { syncBackend: false });
      if (model.trim()) {
        setActiveModel(model.trim());
      }
      await persistSettingsToBackend().catch(() => {});
      onClose();
      onConfiguredTranslate?.();
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Quick Provider Setup"
      data-testid="quick-provider-setup-modal"
    >
      <div className="w-full max-w-2xl rounded-studio-md border border-studio-border-strong bg-studio-panel shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-studio-border px-4 py-3 bg-studio-surface/60">
          <div>
            <h2 className="text-sm font-semibold text-studio-text">
              Quick Provider Setup
            </h2>
            <p className="text-[11px] text-studio-text-muted mt-0.5">
              No translation provider is active. Choose a Cloud API, Local LLM, Custom endpoint, or run Manual / No-LLM mode.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close Quick Provider Setup"
            className="h-7 w-7 inline-flex items-center justify-center rounded-studio-sm text-studio-text-muted hover:text-studio-text hover:bg-studio-hover cursor-pointer"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="p-4 space-y-4">
          {/* 4 Equal Category Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {CATEGORY_CARDS.map((cat) => {
              const Icon = cat.icon;
              const isSelected = category === cat.id;
              return (
                <button
                  key={cat.id}
                  type="button"
                  data-testid={`quick-provider-cat-${cat.id}`}
                  onClick={() => handleCategorySelect(cat)}
                  className={`flex items-start gap-2.5 rounded-studio border p-3 text-left transition-colors cursor-pointer ${
                    isSelected
                      ? 'border-studio-accent bg-studio-accent/10'
                      : 'border-studio-border bg-studio-surface hover:border-studio-border-strong'
                  }`}
                >
                  <div
                    className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-studio-sm border ${
                      isSelected
                        ? 'border-studio-accent/50 bg-studio-accent/20 text-studio-accent'
                        : 'border-studio-border bg-studio-panel text-studio-text-secondary'
                    }`}
                  >
                    <Icon className="h-3.5 w-3.5" />
                  </div>
                  <div className="min-w-0">
                    <div className="text-xs font-semibold text-studio-text">
                      {cat.title}
                    </div>
                    <div className="text-[11px] text-studio-text-muted leading-snug mt-0.5">
                      {cat.subtitle}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Configuration Form for Selected Category */}
          {category !== 'manual' ? (
            <div className="rounded-studio border border-studio-border bg-studio-surface/60 p-3.5 space-y-3">
              {activeCategory.providers.length > 1 && (
                <div className="flex flex-wrap items-center gap-1.5">
                  {activeCategory.providers.map((pid) => {
                    const p = DEFAULT_PROVIDER_PRESETS.find((x) => x.id === pid);
                    const active = providerId === pid;
                    return (
                      <button
                        key={pid}
                        type="button"
                        onClick={() => loadProviderDraft(pid)}
                        className={`h-7 px-2.5 rounded-studio-sm border text-[11px] font-medium transition-colors cursor-pointer ${
                          active
                            ? 'border-studio-accent bg-studio-accent text-studio-accent-fg font-semibold'
                            : 'border-studio-border bg-studio-panel text-studio-text-secondary hover:text-studio-text'
                        }`}
                      >
                        {p?.name || pid}
                      </button>
                    );
                  })}
                </div>
              )}

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                <div className="space-y-1">
                  <label className="text-[11px] font-medium text-studio-text-secondary block">
                    Base URL
                  </label>
                  <input
                    type="text"
                    value={baseUrl}
                    onChange={(e) => setBaseUrl(e.target.value)}
                    placeholder={preset?.default_base_url || 'https://api.example.com/v1'}
                    className="studio-input w-full h-8 px-2.5 text-xs font-mono select-text"
                  />
                </div>

                <div className="space-y-1">
                  <label className="text-[11px] font-medium text-studio-text-secondary block">
                    API Key {preset?.requires_api_key ? '' : '(optional)'}
                  </label>
                  <div className="relative flex items-center">
                    <input
                      type={showKey ? 'text' : 'password'}
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder={preset?.requires_api_key ? 'sk-...' : 'Optional'}
                      autoComplete="off"
                      className="studio-input w-full h-8 pl-2.5 pr-8 text-xs font-mono select-text"
                    />
                    <button
                      type="button"
                      onClick={() => setShowKey((v) => !v)}
                      aria-label={showKey ? 'Hide API Key' : 'Show API Key'}
                      className="absolute right-1.5 h-6 w-6 inline-flex items-center justify-center rounded-studio-sm text-studio-text-muted hover:text-studio-text cursor-pointer"
                    >
                      {showKey ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                    </button>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-12 gap-2 items-end">
                <div className="sm:col-span-8 space-y-1">
                  <label className="text-[11px] font-medium text-studio-text-secondary block">
                    Model
                  </label>
                  {fetchedModels.length > 0 ? (
                    <select
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      className="studio-input w-full h-8 px-2 text-xs font-mono"
                    >
                      {fetchedModels.map((m) => (
                        <option key={m} value={m}>
                          {m}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="text"
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      placeholder={preset?.default_model || 'model-id'}
                      className="studio-input w-full h-8 px-2.5 text-xs font-mono select-text"
                    />
                  )}
                </div>
                <div className="sm:col-span-4">
                  <button
                    type="button"
                    onClick={handleTestAndFetch}
                    disabled={isTesting}
                    className="studio-btn w-full h-8 px-3 text-xs justify-center"
                  >
                    {isTesting && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                    <span>Test &amp; Fetch Models</span>
                  </button>
                </div>
              </div>

              {testResult && (
                <div
                  className={`rounded-studio-sm border px-2.5 py-1.5 text-[11px] flex items-center gap-1.5 ${
                    testResult.ok
                      ? 'border-studio-success/40 bg-studio-success/10 text-studio-text'
                      : 'border-studio-danger/40 bg-studio-danger/10 text-studio-danger'
                  }`}
                >
                  <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
                  <span>{testResult.message}</span>
                </div>
              )}
            </div>
          ) : (
            <div className="rounded-studio border border-studio-border bg-studio-surface/60 p-3.5 text-xs text-studio-text-secondary">
              Runs <strong className="text-studio-text">Detect → OCR → Inpaint</strong> only, leaving bubbles clean for manual lettering without connecting any LLM.
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between gap-2 border-t border-studio-border px-4 py-2.5 bg-studio-surface/40">
          <button
            type="button"
            onClick={() => {
              onClose();
              onOpenFullSettings?.();
            }}
            className="studio-btn h-7 px-2.5 text-[11px]"
          >
            <Sliders className="h-3 w-3" />
            <span>Open Settings → Providers</span>
          </button>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClose}
              className="studio-btn h-7 px-3 text-xs"
            >
              Cancel
            </button>
            <button
              type="button"
              data-testid="quick-provider-confirm-btn"
              disabled={isSaving}
              onClick={handleSaveAndTranslate}
              className="studio-btn-accent h-7 px-3.5 text-xs"
            >
              {isSaving && <Loader2 className="h-3 w-3 animate-spin" />}
              <span>
                {category === 'manual'
                  ? 'Run Detect + Inpaint Only'
                  : 'Save & Translate'}
              </span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
