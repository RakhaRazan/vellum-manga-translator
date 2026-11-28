import {
  Globe,
  Image as ImageIcon,
  Languages,
  Loader2,
  Plus,
  RefreshCw,
  Sliders,
  Trash2,
  Type,
} from 'lucide-react';
import useEditorStore from '../../store/useEditorStore.js';

import {
  REASONING_LEVELS,
  SOURCE_LANGUAGE_OPTIONS,
  INSTRUCTION_PRESETS,
  ToggleSwitch,
  SectionCard,
} from './SettingsPrimitives.jsx';

export default function TranslationTab({ ctx }) {
  const {
    supportedTargetLanguages,
    apiBaseUrl,
    draftPipeline,
    setDraftPipeline,
    glossaryRows,
    setGlossaryRows,
    bulkGlossaryMode,
    setBulkGlossaryMode,
    bulkGlossaryText,
    setBulkGlossaryText,
    fetchingModelsFor,
    providerPresets,
    updateDraftProfile,
    selectActiveProvider,
    handleTestOrFetchProvider,
    translationModelOptions,
  } = ctx;

  return (
              <div className="space-y-4" data-testid="settings-panel-translation">
                {/* Provider, Model & Target Language++ */}
                <SectionCard
                  title="Active Translator & Languages"
                  subtitle="Select LLM provider, model, source comic language, and target language (16 languages)."
                >
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {/* Active Provider */}
                    <div className="space-y-1">
                      <label
                        htmlFor="trans-provider"
                        className="text-[11px] font-medium text-studio-text-secondary block"
                      >
                        Active Provider
                      </label>
                      <select
                        id="trans-provider"
                        value={draftPipeline.provider || 'lmstudio'}
                        onChange={(e) => selectActiveProvider(e.target.value)}
                        className="studio-input w-full h-8 px-2.5 text-xs cursor-pointer"
                      >
                        {providerPresets.map((p) => (
                          <option key={p.id} value={p.id}>
                            {p.name}
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Active Model */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <label
                          htmlFor="trans-model"
                          className="text-[11px] font-medium text-studio-text-secondary"
                        >
                          Model
                        </label>
                        <button
                          type="button"
                          onClick={() =>
                            handleTestOrFetchProvider(
                              draftPipeline.provider || 'lmstudio',
                              'fetch'
                            )
                          }
                          disabled={
                            fetchingModelsFor ===
                            (draftPipeline.provider || 'lmstudio')
                          }
                          className="inline-flex items-center gap-1 text-[10px] font-mono text-studio-accent hover:underline cursor-pointer disabled:opacity-50"
                        >
                          {fetchingModelsFor ===
                          (draftPipeline.provider || 'lmstudio') ? (
                            <Loader2 className="h-2.5 w-2.5 animate-spin" />
                          ) : (
                            <RefreshCw className="h-2.5 w-2.5" />
                          )}
                          Fetch
                        </button>
                      </div>
                      <select
                        id="trans-model"
                        value={draftPipeline.model || ''}
                        onChange={(e) =>
                          updateDraftProfile(
                            draftPipeline.provider || 'lmstudio',
                            { model: e.target.value }
                          )
                        }
                        className="studio-input w-full h-8 px-2.5 text-xs font-mono cursor-pointer"
                      >
                        {translationModelOptions.map((m) => (
                          <option key={m} value={m}>
                            {m}
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Source Language */}
                    <div className="space-y-1">
                      <label
                        htmlFor="trans-source-lang"
                        className="text-[11px] font-medium text-studio-text-secondary block"
                      >
                        Source Comic Language
                      </label>
                      <select
                        id="trans-source-lang"
                        value={draftPipeline.sourceLang || 'ja'}
                        onChange={(e) =>
                          setDraftPipeline((prev) => ({
                            ...prev,
                            sourceLang: e.target.value,
                          }))
                        }
                        className="studio-input w-full h-8 px-2.5 text-xs cursor-pointer"
                      >
                        {SOURCE_LANGUAGE_OPTIONS.map((sl) => (
                          <option key={sl.code} value={sl.code}>
                            {sl.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Target Language++ (16 languages) */}
                    <div className="space-y-1">
                      <label
                        htmlFor="trans-target-lang"
                        className="text-[11px] font-medium text-studio-text-secondary flex items-center gap-1"
                      >
                        <Globe className="h-3 w-3 text-studio-accent" />
                        Target Language++ ({supportedTargetLanguages.length} languages)
                      </label>
                      <select
                        id="trans-target-lang"
                        value={draftPipeline.targetLang || 'id'}
                        onChange={(e) =>
                          setDraftPipeline((prev) => ({
                            ...prev,
                            targetLang: e.target.value,
                          }))
                        }
                        className="studio-input w-full h-8 px-2.5 text-xs cursor-pointer"
                      >
                        {supportedTargetLanguages.map((lang) => (
                          <option key={lang.code} value={lang.code}>
                            {lang.name}
                            {lang.native_name ? ` - ${lang.native_name}` : ''} ({lang.code})
                          </option>
                        ))}
                      </select>
                    </div>

                    {/* Tone / Genre Preset */}
                    <div className="space-y-1 sm:col-span-2">
                      <label
                        htmlFor="trans-tone"
                        className="text-[11px] font-medium text-studio-text-secondary block"
                      >
                        Tone / Genre
                      </label>
                      <select
                        id="trans-tone"
                        value={draftPipeline.translationTone || 'standard'}
                        onChange={(e) =>
                          setDraftPipeline((prev) => ({
                            ...prev,
                            translationTone: e.target.value,
                          }))
                        }
                        className="studio-input w-full h-8 px-2.5 text-xs cursor-pointer"
                      >
                        <option value="standard">Standard</option>
                        <option value="casual">Casual</option>
                        <option value="formal">Formal / Historical</option>
                        <option value="shonen">Action / Shonen</option>
                        <option value="romcom">Romcom / Expressive</option>
                        <option value="adult_doujinshi">
                          Adult / Doujinshi (Uncensored)
                        </option>
                      </select>
                      <p className="text-[10px] text-studio-text-muted">
                        Controls dialogue register, SFX expressiveness, and uncensored colloquial phrasing.
                      </p>
                    </div>

                    {/* Honorifics Policy */}
                    <div className="space-y-1.5 sm:col-span-2">
                      <span className="text-[11px] font-medium text-studio-text-secondary block">
                        Honorifics Policy
                      </span>
                      <div className="grid grid-cols-3 gap-1.5">
                        {[
                          { id: 'keep', label: 'Keep (-san/-kun)' },
                          { id: 'localize', label: 'Localize (Kak/Mr.)' },
                          { id: 'drop', label: 'Drop' },
                        ].map((opt) => {
                          const active = (draftPipeline.honorificsPolicy || 'keep') === opt.id;
                          return (
                            <button
                              key={opt.id}
                              type="button"
                              onClick={() => {
                                setDraftPipeline((prev) => ({
                                  ...prev,
                                  honorificsPolicy: opt.id,
                                }));
                                useEditorStore.getState().setHonorificsPolicy(opt.id);
                                fetch(`${apiBaseUrl}/api/settings`, {
                                  method: 'POST',
                                  headers: { 'Content-Type': 'application/json' },
                                  body: JSON.stringify({ honorifics_policy: opt.id }),
                                }).catch(() => {});
                              }}
                              className={`h-8 px-2 rounded-studio-sm border text-[11px] font-medium transition-colors cursor-pointer ${
                                active
                                  ? 'border-studio-accent bg-studio-accent/15 text-studio-accent'
                                  : 'border-studio-border bg-studio-surface text-studio-text-secondary hover:text-studio-text'
                              }`}
                            >
                              {opt.label}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </SectionCard>

                {/* Sampling Sliders */}
                <SectionCard
                  title="Sampling Parameters"
                  subtitle="Fine-tune creativity, nucleus sampling, token budget, and repetition control."
                >
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5">
                    {/* Temperature */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-studio-text-secondary font-medium">
                          temperature
                        </span>
                        <span className="font-mono text-studio-accent font-semibold tabular-nums">
                          {Number(draftPipeline.temperature ?? 0.2).toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        aria-label="Temperature"
                        min={0}
                        max={2}
                        step={0.05}
                        value={draftPipeline.temperature ?? 0.2}
                        onChange={(e) =>
                          setDraftPipeline((prev) => ({
                            ...prev,
                            temperature: Number(e.target.value),
                          }))
                        }
                        className="w-full accent-[var(--color-accent)] cursor-pointer"
                      />
                    </div>

                    {/* top_p */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-studio-text-secondary font-medium">
                          top_p
                        </span>
                        <span className="font-mono text-studio-accent font-semibold tabular-nums">
                          {Number(draftPipeline.topP ?? 0.95).toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        aria-label="Top P"
                        min={0}
                        max={1}
                        step={0.01}
                        value={draftPipeline.topP ?? 0.95}
                        onChange={(e) =>
                          setDraftPipeline((prev) => ({
                            ...prev,
                            topP: Number(e.target.value),
                          }))
                        }
                        className="w-full accent-[var(--color-accent)] cursor-pointer"
                      />
                    </div>

                    {/* top_k */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-studio-text-secondary font-medium">
                          top_k
                        </span>
                        <span className="font-mono text-studio-accent font-semibold tabular-nums">
                          {Math.round(Number(draftPipeline.topK ?? 40))}
                        </span>
                      </div>
                      <input
                        type="range"
                        aria-label="Top K"
                        min={0}
                        max={200}
                        step={1}
                        value={draftPipeline.topK ?? 40}
                        onChange={(e) =>
                          setDraftPipeline((prev) => ({
                            ...prev,
                            topK: Number(e.target.value),
                          }))
                        }
                        className="w-full accent-[var(--color-accent)] cursor-pointer"
                      />
                    </div>

                    {/* max_tokens */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-studio-text-secondary font-medium">
                          max_tokens
                        </span>
                        <span className="font-mono text-studio-accent font-semibold tabular-nums">
                          {Math.round(Number(draftPipeline.maxTokens ?? 2048))}
                        </span>
                      </div>
                      <input
                        type="range"
                        aria-label="Max Tokens"
                        min={256}
                        max={8192}
                        step={128}
                        value={Math.min(8192, draftPipeline.maxTokens ?? 2048)}
                        onChange={(e) =>
                          setDraftPipeline((prev) => ({
                            ...prev,
                            maxTokens: Number(e.target.value),
                          }))
                        }
                        className="w-full accent-[var(--color-accent)] cursor-pointer"
                      />
                    </div>

                    {/* frequency_penalty */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-studio-text-secondary font-medium">
                          frequency_penalty
                        </span>
                        <span className="font-mono text-studio-accent font-semibold tabular-nums">
                          {Number(draftPipeline.frequencyPenalty ?? 0).toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        aria-label="Frequency Penalty"
                        min={-2}
                        max={2}
                        step={0.05}
                        value={draftPipeline.frequencyPenalty ?? 0}
                        onChange={(e) =>
                          setDraftPipeline((prev) => ({
                            ...prev,
                            frequencyPenalty: Number(e.target.value),
                          }))
                        }
                        className="w-full accent-[var(--color-accent)] cursor-pointer"
                      />
                    </div>

                    {/* Timeout */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-studio-text-secondary font-medium">
                          timeout (sec)
                        </span>
                        <span className="font-mono text-studio-accent font-semibold tabular-nums">
                          {Math.round(Number(draftPipeline.timeout ?? 90))}s
                        </span>
                      </div>
                      <input
                        type="range"
                        aria-label="Request Timeout"
                        min={15}
                        max={240}
                        step={5}
                        value={draftPipeline.timeout ?? 90}
                        onChange={(e) =>
                          setDraftPipeline((prev) => ({
                            ...prev,
                            timeout: Number(e.target.value),
                          }))
                        }
                        className="w-full accent-[var(--color-accent)] cursor-pointer"
                      />
                    </div>
                  </div>
                </SectionCard>

                {/* Reasoning & Vision Controls */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-3.5">
                  <SectionCard
                    title="Reasoning Control"
                    subtitle="Configure thinking effort for reasoning models and filter internal thought blocks."
                  >
                    <div className="space-y-3">
                      <div className="grid grid-cols-4 gap-1.5">
                        {REASONING_LEVELS.map((lvl) => {
                          const isActive =
                            (draftPipeline.reasoningEffort || 'off') === lvl.id;
                          return (
                            <button
                              key={lvl.id}
                              type="button"
                              onClick={() =>
                                setDraftPipeline((prev) => ({
                                  ...prev,
                                  reasoningEffort: lvl.id,
                                }))
                              }
                              className={`h-7 rounded-studio border text-xs font-medium transition-colors cursor-pointer ${
                                isActive
                                  ? 'border-studio-accent bg-studio-accent text-studio-accent-fg font-semibold'
                                  : 'border-studio-border bg-studio-panel text-studio-text-secondary hover:bg-studio-hover hover:text-studio-text'
                              }`}
                            >
                              {lvl.label}
                            </button>
                          );
                        })}
                      </div>

                      <div className="flex items-center justify-between gap-2 pt-1 border-t border-studio-border">
                        <div>
                          <label
                            htmlFor="trans-strip-think"
                            className="text-xs font-medium text-studio-text block"
                          >
                            Hide AI Thinking Process
                          </label>
                          <p className="text-[10px] text-studio-text-muted">
                            Automatically filters out internal model thought blocks before parsing translated dialogue.
                          </p>
                        </div>
                        <ToggleSwitch
                          id="trans-strip-think"
                          label="Hide AI Thinking Process"
                          checked={Boolean(draftPipeline.stripThinking ?? true)}
                          onChange={(val) =>
                            setDraftPipeline((prev) => ({
                              ...prev,
                              stripThinking: val,
                            }))
                          }
                        />
                      </div>
                    </div>
                  </SectionCard>

                  <SectionCard
                    title="Multimodal Vision Context"
                    subtitle="Attach the comic page image so VLMs infer speaker gender, tone & expressions."
                  >
                    <div className="space-y-3">
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <ImageIcon className="h-4 w-4 text-studio-accent shrink-0" />
                          <div>
                            <label
                              htmlFor="trans-vision-toggle"
                              className="text-xs font-medium text-studio-text block"
                            >
                              Send Page Image (VLM)
                            </label>
                            <p className="text-[10px] text-studio-text-muted">
                              Auto-falls back to text-only if model rejects images.
                            </p>
                          </div>
                        </div>
                        <ToggleSwitch
                          id="trans-vision-toggle"
                          label="Enable Multimodal Vision"
                          checked={Boolean(draftPipeline.useVision)}
                          onChange={(val) =>
                            setDraftPipeline((prev) => ({
                              ...prev,
                              useVision: val,
                            }))
                          }
                        />
                      </div>

                      <div className="flex items-center justify-between gap-2 pt-1 border-t border-studio-border">
                        <label
                          htmlFor="trans-vision-max-side"
                          className="text-[11px] text-studio-text-secondary"
                        >
                          Max Image Side (px)
                        </label>
                        <select
                          id="trans-vision-max-side"
                          disabled={!draftPipeline.useVision}
                          value={draftPipeline.visionMaxSide ?? 1024}
                          onChange={(e) =>
                            setDraftPipeline((prev) => ({
                              ...prev,
                              visionMaxSide: Number(e.target.value),
                            }))
                          }
                          className="studio-input h-7 px-2 text-xs font-mono cursor-pointer disabled:opacity-40"
                        >
                          {[512, 768, 1024, 1280, 1536, 2048].map((px) => (
                            <option key={px} value={px}>
                              {px}px
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  </SectionCard>
                </div>

                {/* Instruction Bar */}
                <SectionCard
                  title="Instruction Bar"
                  subtitle="Custom system prompt directives injected into every page translation request."
                  action={
                    <div className="flex flex-wrap gap-1">
                      {INSTRUCTION_PRESETS.map((pr) => (
                        <button
                          key={pr.label}
                          type="button"
                          onClick={() =>
                            setDraftPipeline((prev) => ({
                              ...prev,
                              customInstructions: prev.customInstructions
                                ? `${prev.customInstructions}\n${pr.text}`
                                : pr.text,
                            }))
                          }
                          className="rounded-studio-sm border border-studio-border bg-studio-panel hover:bg-studio-hover px-1.5 py-0.5 text-[10px] text-studio-text-secondary hover:text-studio-text cursor-pointer"
                        >
                          + {pr.label}
                        </button>
                      ))}
                    </div>
                  }
                >
                  <textarea
                    rows={3}
                    value={draftPipeline.customInstructions || ''}
                    onChange={(e) =>
                      setDraftPipeline((prev) => ({
                        ...prev,
                        customInstructions: e.target.value,
                      }))
                    }
                    placeholder="e.g., Translate in casual shounen Indonesian; keep character names in Japanese romaji; use 'gue/lu' for delinquent characters..."
                    className="studio-input w-full px-2.5 py-2 text-xs select-text resize-y font-sans"
                  />
                </SectionCard>

                {/* Glossary Table (Source -> Target) */}
                <SectionCard
                  title="Glossary (Source → Target)"
                  subtitle="Strict term overrides enforced across every dialogue block (character names, skills, locations)."
                  action={
                    <div className="flex items-center gap-1.5">
                      <button
                        type="button"
                        onClick={() => {
                          if (!bulkGlossaryMode) {
                            setBulkGlossaryText(
                              glossaryRows
                                .filter((r) => r.source || r.target)
                                .map((r) => `${r.source} -> ${r.target}`)
                                .join('\n')
                            );
                            setBulkGlossaryMode(true);
                          } else {
                            const parsed = [];
                            bulkGlossaryText.split('\n').forEach((line, idx) => {
                              const t = line.trim();
                              if (!t || t.startsWith('#')) return;
                              for (const sep of ['->', '=>', '→', '=', ':']) {
                                if (t.includes(sep)) {
                                  const pos = t.indexOf(sep);
                                  const lhs = t.slice(0, pos).trim();
                                  const rhs = t.slice(pos + sep.length).trim();
                                  if (lhs && rhs) {
                                    parsed.push({
                                      id: `gb_${Date.now()}_${idx}`,
                                      source: lhs,
                                      target: rhs,
                                    });
                                  }
                                  break;
                                }
                              }
                            });
                            setGlossaryRows(parsed);
                            setBulkGlossaryMode(false);
                          }
                        }}
                        className="studio-btn h-6 px-2 text-[10px]"
                      >
                        {bulkGlossaryMode ? 'Table View' : 'Bulk Edit'}
                      </button>
                      {!bulkGlossaryMode && (
                        <button
                          type="button"
                          onClick={() =>
                            setGlossaryRows((prev) => [
                              ...prev,
                              { id: `g_${Date.now()}`, source: '', target: '' },
                            ])
                          }
                          className="studio-btn-accent h-6 px-2 text-[10px]"
                        >
                          <Plus className="h-3 w-3" />
                          <span>Add Term</span>
                        </button>
                      )}
                    </div>
                  }
                >
                  {bulkGlossaryMode ? (
                    <textarea
                      rows={5}
                      value={bulkGlossaryText}
                      onChange={(e) => setBulkGlossaryText(e.target.value)}
                      placeholder={'先輩 -> senpai\n鬼殺隊 -> Demon Slayer Corps\n師兄 -> Shixiong'}
                      className="studio-input w-full px-2.5 py-2 text-xs font-mono select-text resize-y"
                    />
                  ) : glossaryRows.length === 0 ? (
                    <div className="rounded-studio border border-dashed border-studio-border p-3 text-center text-[11px] text-studio-text-muted">
                      No glossary entries yet. Click <strong>Add Term</strong> to map{' '}
                      <code>Source -&gt; Target</code> terms.
                    </div>
                  ) : (
                    <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                      <div className="grid grid-cols-12 gap-2 px-1 text-[10px] font-mono uppercase tracking-wider text-studio-text-muted">
                        <span className="col-span-5">Source Term</span>
                        <span className="col-span-1 text-center">→</span>
                        <span className="col-span-5">Target Translation</span>
                        <span className="col-span-1" />
                      </div>
                      {glossaryRows.map((row, idx) => (
                        <div
                          key={row.id}
                          className="grid grid-cols-12 gap-2 items-center"
                        >
                          <input
                            type="text"
                            value={row.source}
                            onChange={(e) =>
                              setGlossaryRows((prev) =>
                                prev.map((r, i) =>
                                  i === idx ? { ...r, source: e.target.value } : r
                                )
                              )
                            }
                            placeholder="先輩 / 师兄"
                            className="col-span-5 studio-input h-7 px-2 text-xs select-text"
                          />
                          <span className="col-span-1 text-center font-mono text-xs text-studio-text-muted">
                            →
                          </span>
                          <input
                            type="text"
                            value={row.target}
                            onChange={(e) =>
                              setGlossaryRows((prev) =>
                                prev.map((r, i) =>
                                  i === idx ? { ...r, target: e.target.value } : r
                                )
                              )
                            }
                            placeholder="senpai / shixiong"
                            className="col-span-5 studio-input h-7 px-2 text-xs select-text"
                          />
                          <button
                            type="button"
                            onClick={() =>
                              setGlossaryRows((prev) =>
                                prev.filter((_, i) => i !== idx)
                              )
                            }
                            title="Delete glossary entry"
                            className="col-span-1 h-7 w-7 inline-flex items-center justify-center rounded-studio text-studio-text-muted hover:text-studio-danger hover:bg-studio-danger/10 cursor-pointer"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </SectionCard>
              </div>
  );
}
