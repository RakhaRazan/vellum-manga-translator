import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  ArrowDown,
  ArrowUp,
  Check,
  CheckCircle2,
  Cpu,
  Download,
  Eraser,
  Eye,
  EyeOff,
  FileText,
  Globe,
  HardDrive,
  Image as ImageIcon,
  Languages,
  Layers,
  Loader2,
  Monitor,
  Moon,
  Palette,
  Plus,
  RefreshCw,
  RotateCcw,
  Save,
  Search,
  Server,
  Sliders,
  Sparkles,
  Sun,
  Trash2,
  Type,
  Upload,
  Wifi,
  X,
} from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import useEditorStore, {
  DEFAULT_API_BASE_URL,
  DEFAULT_DETECTION_MODELS,
  DEFAULT_FONT_FALLBACK_CHAIN,
  DEFAULT_INPAINTING_MODELS,
  DEFAULT_OCR_MODELS,
  DEFAULT_PROVIDER_PRESETS,
  DEFAULT_TYPESETTING_CONFIG,
  buildDefaultProviderProfiles,
  ensureFontFaceLoaded,
  formatFontFamilyChain,
  normalizeOptionalThreshold,
} from '../../store/useEditorStore.js';
import { formatExecutionProvider } from '../StatusBar.jsx';

import {
  TABS,
  THEME_CARDS,
  ACCENT_PRESETS,
  DENSITY_OPTIONS,
  LAMA_TILE_SIZES,
  READING_ORDER_OPTIONS,
  BACKEND_OPTIONS,
  REASONING_LEVELS,
  SOURCE_LANGUAGE_OPTIONS,
  INSTRUCTION_PRESETS,
  FONT_STYLE_FILTERS,
  ToggleSwitch,
  SectionCard,
  mergePipelineModelOptions,
  ModelStatusBadge,
  normalizeExecMode,
  StageHardwareBadge,
  glossaryDictToRows,
  glossaryRowsToDict,
} from './SettingsPrimitives.jsx';

export default function ProvidersTab({ ctx }) {
  const {
    providerModelsCache,
    runtimeStatus,
    draftPipeline,
    selectedProviderId,
    setSelectedProviderId,
    showApiKeys,
    setShowApiKeys,
    testingProviderId,
    fetchingModelsFor,
    providerPresets,
    updateDraftProfile,
    selectActiveProvider,
    handleTestOrFetchProvider,
    selectedPreset,
    selectedProfile,
    selectedProviderTest,
  } = ctx;

  return (
              <div
                className="grid grid-cols-1 lg:grid-cols-12 gap-3.5"
                data-testid="settings-panel-providers"
              >
                {/* Provider List Rail (8 Koharu-style providers) */}
                <div className="lg:col-span-4 space-y-1.5">
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-studio-text-muted px-1">
                    Providers ({providerPresets.length})
                  </div>
                  <div className="space-y-1">
                    {providerPresets.map((preset) => {
                      const isSelected = selectedProviderId === preset.id;
                      const isActiveEngine = draftPipeline.provider === preset.id;
                      const testInfo = providerModelsCache[preset.id];
                      const isConnected =
                        testInfo?.connected ??
                        (preset.id === 'lmstudio' && runtimeStatus.lmStudio?.connected);

                      return (
                        <button
                          key={preset.id}
                          type="button"
                          data-testid={`provider-card-${preset.id}`}
                          onClick={() => setSelectedProviderId(preset.id)}
                          className={`w-full rounded-studio border px-2.5 py-2 text-left transition-colors cursor-pointer flex items-center justify-between gap-2 ${
                            isSelected
                              ? 'border-studio-accent bg-studio-accent/10'
                              : 'border-studio-border bg-studio-surface hover:border-studio-border-strong'
                          }`}
                        >
                          <div className="min-w-0">
                            <div className="flex items-center gap-1.5">
                              <span
                                className={`h-1.5 w-1.5 rounded-full shrink-0 ${
                                  isConnected
                                    ? 'bg-studio-success'
                                    : testInfo && !testInfo.connected
                                    ? 'bg-studio-danger'
                                    : 'bg-studio-border-strong'
                                }`}
                              />
                              <span className="text-xs font-semibold text-studio-text truncate">
                                {preset.name}
                              </span>
                            </div>
                            <div className="text-[10px] font-mono text-studio-text-muted mt-0.5 flex items-center gap-1.5">
                              <span>{preset.is_local ? 'Local' : 'Cloud'}</span>
                              <span>·</span>
                              <span className="uppercase">{preset.protocol}</span>
                            </div>
                          </div>

                          {isActiveEngine && (
                            <span className="rounded-studio-sm bg-studio-accent text-studio-accent-fg px-1.5 py-0.5 text-[9px] font-mono font-bold uppercase shrink-0">
                              Active
                            </span>
                          )}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Selected Provider Configuration Pane */}
                <div className="lg:col-span-8 space-y-3">
                  <SectionCard
                    title={selectedPreset.name}
                    subtitle={
                      selectedPreset.is_local
                        ? `Local OpenAI-compatible inference server (default: ${selectedPreset.default_base_url})`
                        : selectedPreset.protocol === 'anthropic'
                        ? 'Native Anthropic Claude Messages API'
                        : 'Cloud OpenAI-compatible API endpoint'
                    }
                    action={
                      draftPipeline.provider === selectedProviderId ? (
                        <span className="inline-flex items-center gap-1 rounded-studio-sm border border-studio-accent/50 bg-studio-accent/15 px-2 py-1 text-[10px] font-mono font-semibold text-studio-accent">
                          <CheckCircle2 className="h-3 w-3" />
                          Active Provider
                        </span>
                      ) : (
                        <button
                          type="button"
                          onClick={() => selectActiveProvider(selectedProviderId)}
                          className="studio-btn-accent h-7 px-2.5 text-[11px]"
                        >
                          Use as Active
                        </button>
                      )
                    }
                  >
                    {/* Base URL */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <label
                          htmlFor="provider-base-url"
                          className="text-[11px] font-medium text-studio-text-secondary"
                        >
                          Base URL
                        </label>
                        {selectedPreset.default_base_url && (
                          <button
                            type="button"
                            onClick={() =>
                              updateDraftProfile(selectedProviderId, {
                                base_url: selectedPreset.default_base_url,
                              })
                            }
                            className="text-[10px] font-mono text-studio-accent hover:underline cursor-pointer"
                          >
                            Reset default
                          </button>
                        )}
                      </div>
                      <input
                        id="provider-base-url"
                        type="text"
                        value={selectedProfile.base_url}
                        onChange={(e) =>
                          updateDraftProfile(selectedProviderId, {
                            base_url: e.target.value,
                          })
                        }
                        placeholder={
                          selectedPreset.default_base_url ||
                          'https://your-endpoint.example/v1'
                        }
                        className="studio-input w-full h-8 px-2.5 text-xs font-mono select-text"
                      />
                    </div>

                    {/* Masked API Key with Show/Hide toggle */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <label
                          htmlFor="provider-api-key"
                          className="text-[11px] font-medium text-studio-text-secondary"
                        >
                          API Key{' '}
                          {!selectedPreset.requires_api_key && (
                            <span className="text-studio-text-muted font-normal">
                              (optional for local servers)
                            </span>
                          )}
                        </label>
                        <button
                          type="button"
                          onClick={() =>
                            setShowApiKeys((prev) => ({
                              ...prev,
                              [selectedProviderId]: !prev[selectedProviderId],
                            }))
                          }
                          aria-label={
                            showApiKeys[selectedProviderId]
                              ? 'Hide API Key'
                              : 'Show API Key'
                          }
                          className="inline-flex items-center gap-1 text-[10px] text-studio-text-secondary hover:text-studio-text cursor-pointer"
                        >
                          {showApiKeys[selectedProviderId] ? (
                            <>
                              <EyeOff className="h-3 w-3" />
                              Hide
                            </>
                          ) : (
                            <>
                              <Eye className="h-3 w-3" />
                              Show
                            </>
                          )}
                        </button>
                      </div>
                      <div className="relative flex items-center">
                        <input
                          id="provider-api-key"
                          type={showApiKeys[selectedProviderId] ? 'text' : 'password'}
                          value={selectedProfile.api_key}
                          onChange={(e) =>
                            updateDraftProfile(selectedProviderId, {
                              api_key: e.target.value,
                            })
                          }
                          placeholder={
                            selectedPreset.is_local
                              ? 'Optional'
                              : 'API key...'
                          }
                          autoComplete="off"
                          className="studio-input w-full h-8 pl-2.5 pr-9 text-xs font-mono select-text"
                        />
                        <button
                          type="button"
                          onClick={() =>
                            setShowApiKeys((prev) => ({
                              ...prev,
                              [selectedProviderId]: !prev[selectedProviderId],
                            }))
                          }
                          className="absolute right-1.5 h-6 w-6 inline-flex items-center justify-center rounded-studio-sm text-studio-text-muted hover:text-studio-text hover:bg-studio-hover cursor-pointer"
                          title={
                            showApiKeys[selectedProviderId]
                              ? 'Mask API Key'
                              : 'Reveal API Key'
                          }
                        >
                          {showApiKeys[selectedProviderId] ? (
                            <EyeOff className="h-3.5 w-3.5" />
                          ) : (
                            <Eye className="h-3.5 w-3.5" />
                          )}
                        </button>
                      </div>
                    </div>

                    {/* Default / Preferred Model + Universal Fetch Models Dropdown */}
                    <div className="space-y-1.5">
                      <label
                        htmlFor="provider-default-model"
                        className="text-[11px] font-medium text-studio-text-secondary block"
                      >
                        Preferred Model ID
                      </label>
                      {selectedProviderTest?.models?.length > 0 && (
                        <select
                          aria-label="Fetched Models Dropdown"
                          data-testid="provider-fetched-models-select"
                          value={
                            selectedProviderTest.models.includes(selectedProfile.model)
                              ? selectedProfile.model
                              : ''
                          }
                          onChange={(e) => {
                            if (e.target.value) {
                              updateDraftProfile(selectedProviderId, {
                                model: e.target.value,
                              });
                            }
                          }}
                          className="studio-input w-full h-8 px-2 text-xs font-mono"
                        >
                          <option value="">
                            Select from {selectedProviderTest.models.length} fetched model(s)…
                          </option>
                          {selectedProviderTest.models.map((mid) => (
                            <option key={mid} value={mid}>
                              {mid}
                            </option>
                          ))}
                        </select>
                      )}
                      <input
                        id="provider-default-model"
                        type="text"
                        value={selectedProfile.model}
                        onChange={(e) =>
                          updateDraftProfile(selectedProviderId, {
                            model: e.target.value,
                          })
                        }
                        placeholder={selectedPreset.default_model || 'model-id'}
                        className="studio-input w-full h-8 px-2.5 text-xs font-mono select-text"
                      />
                    </div>

                    {/* Action Buttons: Test Connection + Fetch Models */}
                    <div className="flex flex-wrap items-center gap-2 pt-1">
                      <button
                        type="button"
                        data-testid="provider-test-connection-btn"
                        disabled={
                          testingProviderId === selectedProviderId ||
                          fetchingModelsFor === selectedProviderId
                        }
                        onClick={() =>
                          handleTestOrFetchProvider(selectedProviderId, 'test')
                        }
                        className="studio-btn-accent h-8 px-3 text-xs"
                      >
                        {testingProviderId === selectedProviderId ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <Wifi className="h-3.5 w-3.5" />
                        )}
                        <span>Test Connection</span>
                      </button>

                      <button
                        type="button"
                        data-testid="provider-fetch-models-btn"
                        disabled={
                          testingProviderId === selectedProviderId ||
                          fetchingModelsFor === selectedProviderId
                        }
                        onClick={() =>
                          handleTestOrFetchProvider(selectedProviderId, 'fetch')
                        }
                        className="studio-btn h-8 px-3 text-xs"
                      >
                        {fetchingModelsFor === selectedProviderId ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <RefreshCw className="h-3.5 w-3.5" />
                        )}
                        <span>Fetch Models</span>
                      </button>

                      {/* Capability pills */}
                      <div className="ml-auto flex items-center gap-1.5 text-[10px] font-mono text-studio-text-muted">
                        {selectedPreset.supports_vision && (
                          <span className="rounded-studio-sm border border-studio-border bg-studio-panel px-1.5 py-0.5">
                            Vision
                          </span>
                        )}
                        {selectedPreset.supports_json_schema && (
                          <span className="rounded-studio-sm border border-studio-border bg-studio-panel px-1.5 py-0.5">
                            JSON Schema
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Connection Test / Fetch Result Banner */}
                    {selectedProviderTest && (
                      <div
                        data-testid="provider-test-result"
                        className={`rounded-studio border p-2.5 text-xs space-y-2 ${
                          selectedProviderTest.connected
                            ? 'border-studio-success/40 bg-studio-success/10 text-studio-text'
                            : 'border-studio-danger/40 bg-studio-danger/10 text-studio-danger'
                        }`}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex items-center gap-1.5 font-medium">
                            {selectedProviderTest.connected ? (
                              <CheckCircle2 className="h-4 w-4 text-studio-success shrink-0" />
                            ) : (
                              <AlertCircle className="h-4 w-4 text-studio-danger shrink-0" />
                            )}
                            <span>
                              {selectedProviderTest.connected
                                ? selectedProviderTest.message ||
                                  `Connected (${selectedProviderTest.models.length} models)`
                                : selectedProviderTest.error ||
                                  selectedProviderTest.message ||
                                  'Connection failed'}
                            </span>
                          </div>
                          {typeof selectedProviderTest.latencyMs === 'number' && (
                            <span className="font-mono text-[10px] text-studio-text-muted shrink-0 tabular-nums">
                              {selectedProviderTest.latencyMs}ms
                            </span>
                          )}
                        </div>

                        {selectedProviderTest.connected &&
                          selectedProviderTest.resolvedModelId && (
                            <div className="text-[11px] font-mono text-studio-text-secondary">
                              Resolved model:{' '}
                              <span className="text-studio-text font-semibold">
                                {selectedProviderTest.activeModel ||
                                  selectedProviderTest.resolvedModelId}
                              </span>
                            </div>
                          )}

                        {/* Discovered Models Picker Chips */}
                        {selectedProviderTest.models?.length > 0 && (
                          <div className="space-y-1.5 pt-1 border-t border-studio-border/60">
                            <div className="text-[10px] font-mono uppercase tracking-wider text-studio-text-muted">
                              Fetched Models ({selectedProviderTest.models.length}) - click to select
                            </div>
                            <div className="flex flex-wrap gap-1.5 max-h-32 overflow-y-auto pr-1">
                              {selectedProviderTest.models.map((mid) => {
                                const isPicked = selectedProfile.model === mid;
                                const isVlm = (
                                  selectedProviderTest.visionModels || []
                                ).includes(mid);
                                return (
                                  <button
                                    key={mid}
                                    type="button"
                                    onClick={() =>
                                      updateDraftProfile(selectedProviderId, {
                                        model: mid,
                                      })
                                    }
                                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-studio-sm border font-mono text-[10px] transition-colors cursor-pointer ${
                                      isPicked
                                        ? 'border-studio-accent bg-studio-accent text-studio-accent-fg font-semibold'
                                        : 'border-studio-border bg-studio-panel text-studio-text hover:border-studio-accent/60'
                                    }`}
                                  >
                                    <span className="truncate max-w-[220px]">{mid}</span>
                                    {isVlm && (
                                      <span className="text-[9px] opacity-80">
                                        [VLM]
                                      </span>
                                    )}
                                  </button>
                                );
                              })}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </SectionCard>
                </div>
              </div>
  );
}
