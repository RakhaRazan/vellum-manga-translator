import React from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Cpu,
  HardDrive,
  Loader2,
  RotateCcw,
  Save,
  Sliders,
  X,
} from 'lucide-react';
import { TABS, normalizeExecMode } from './settings/SettingsPrimitives.jsx';
import { useSettingsController } from './settings/useSettingsController.js';
import AppearanceTab from './settings/AppearanceTab.jsx';
import PipelineTab from './settings/PipelineTab.jsx';
import ProvidersTab from './settings/ProvidersTab.jsx';
import TranslationTab from './settings/TranslationTab.jsx';
import TypesettingTab from './settings/TypesettingTab.jsx';

export default function SettingsModal({ isOpen, onClose, onSaved }) {
  const ctx = useSettingsController({ isOpen, onClose, onSaved });
  const {
    runtimeStatus,
    setSettingsActiveTab,
    activeTab,
    draftPipeline,
    isHydrating,
    isSaving,
    saveFeedback,
    settingsPath,
    handleSaveAll,
    handleResetActiveTab,
  } = ctx;

  if (!isOpen) return null;

  const hwProviderMeta = formatExecutionProvider(runtimeStatus.executionProvider);
  const currentTabMeta = TABS.find((t) => t.id === activeTab) || TABS[0];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-3 sm:p-5 select-none"
      role="dialog"
      aria-modal="true"
      aria-labelledby="settings-modal-title"
      data-testid="settings-modal"
      onClick={onClose}
    >
      <div
        className="w-full max-w-5xl h-[86vh] max-h-[760px] rounded-studio-md border border-studio-border bg-studio-panel shadow-2xl flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Top Header */}
        <header className="h-11 px-4 border-b border-studio-border bg-studio-panel flex items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-2.5 min-w-0">
            <span className="h-6 w-6 rounded-studio bg-studio-accent text-studio-accent-fg flex items-center justify-center shrink-0">
              <Sliders className="h-3.5 w-3.5" />
            </span>
            <div className="min-w-0">
              <h2
                id="settings-modal-title"
                className="text-xs font-semibold uppercase tracking-wider text-studio-text leading-none"
              >
                Studio Settings
              </h2>
              <p className="text-[10px] font-mono text-studio-text-muted truncate mt-0.5">
                {currentTabMeta.label} · {currentTabMeta.desc}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {isHydrating && (
              <span className="inline-flex items-center gap-1 text-[10px] font-mono text-studio-text-muted">
                <Loader2 className="h-3 w-3 animate-spin" />
                Syncing…
              </span>
            )}
            <button
              type="button"
              onClick={onClose}
              aria-label="Close Settings"
              className="h-7 w-7 inline-flex items-center justify-center rounded-studio border border-transparent hover:border-studio-border text-studio-text-muted hover:text-studio-text hover:bg-studio-hover transition-colors cursor-pointer"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </header>

        {/* Body: Koharu-style Left Vertical Tab Rail + Right Scrollable Content */}
        <div className="flex-1 min-h-0 flex flex-col md:flex-row overflow-hidden">
          {/* Left Navigation Rail */}
          <nav
            aria-label="Settings Categories"
            className="md:w-52 shrink-0 border-b md:border-b-0 md:border-r border-studio-border bg-studio-canvas/50 p-2 flex md:flex-col gap-1 overflow-x-auto md:overflow-y-auto"
          >
            {TABS.map((tab, idx) => {
              const Icon = tab.icon;
              const isSelected = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  type="button"
                  role="tab"
                  aria-selected={isSelected}
                  data-testid={`settings-tab-${tab.id}`}
                  onClick={() => setSettingsActiveTab(tab.id)}
                  className={`relative flex items-center gap-2.5 px-2.5 py-2 rounded-studio text-left transition-colors cursor-pointer shrink-0 md:w-full ${
                    isSelected
                      ? 'bg-studio-surface text-studio-text border border-studio-accent/50 shadow-sm'
                      : 'text-studio-text-secondary border border-transparent hover:bg-studio-hover hover:text-studio-text'
                  }`}
                >
                  {isSelected && (
                    <span
                      className="hidden md:block absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-r bg-studio-accent"
                      aria-hidden="true"
                    />
                  )}
                  <Icon
                    className={`h-4 w-4 shrink-0 ${
                      isSelected ? 'text-studio-accent' : 'text-studio-text-muted'
                    }`}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-1">
                      <span className="text-xs font-medium truncate">
                        {tab.label}
                      </span>
                      <span className="hidden md:inline font-mono text-[9px] text-studio-text-muted">
                        0{idx + 1}
                      </span>
                    </div>
                    <div className="hidden md:block text-[10px] text-studio-text-muted truncate">
                      {tab.desc}
                    </div>
                  </div>
                </button>
              );
            })}

            {/* Live Status Summary Card at bottom of sidebar */}
            <div className="hidden md:block mt-auto pt-3">
              <div className="rounded-studio border border-studio-border bg-studio-surface p-2.5 space-y-1.5 text-[10px] font-mono">
                <div className="text-studio-text-muted uppercase tracking-wider font-semibold">
                  Active Runtime
                </div>
                <div className="flex items-center justify-between gap-1">
                  <span className="text-studio-text-muted">Engine</span>
                  <span className="text-studio-text truncate">
                    {normalizeExecMode(draftPipeline.executionMode) === 'cpu_only'
                      ? 'CPU Only'
                      : normalizeExecMode(draftPipeline.executionMode) === 'force_gpu'
                      ? `GPU (${hwProviderMeta.short})`
                      : 'Hybrid GPU+CPU'}
                  </span>
                </div>
                <div className="flex items-center justify-between gap-1">
                  <span className="text-studio-text-muted">Provider</span>
                  <span className="text-studio-accent truncate">
                    {draftPipeline.provider}
                  </span>
                </div>
                <div className="flex items-center justify-between gap-1">
                  <span className="text-studio-text-muted">Target</span>
                  <span className="text-studio-text uppercase">
                    {draftPipeline.sourceLang} → {draftPipeline.targetLang}
                  </span>
                </div>
              </div>
            </div>
          </nav>

          {/* Right Content Pane */}
          <div className="flex-1 min-w-0 overflow-y-auto p-4 sm:p-5 space-y-4">
            {/* ==============================================================
             * TAB 1 - APPEARANCE
             * ============================================================== */}
            {activeTab === 'appearance' && <AppearanceTab ctx={ctx} />}
            {activeTab === 'pipeline' && <PipelineTab ctx={ctx} />}
            {activeTab === 'providers' && <ProvidersTab ctx={ctx} />}
            {activeTab === 'translation' && <TranslationTab ctx={ctx} />}
            {activeTab === 'typesetting' && <TypesettingTab ctx={ctx} />}
          </div>
        </div>

        {/* Sticky Footer */}
        <footer className="px-4 py-2.5 border-t border-studio-border bg-studio-surface flex flex-wrap items-center justify-between gap-2 shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <button
              type="button"
              onClick={handleResetActiveTab}
              className="studio-btn h-7 px-2.5 text-[11px]"
              title={`Reset ${currentTabMeta.label} settings to defaults`}
            >
              <RotateCcw className="h-3 w-3" />
              <span>Reset Tab</span>
            </button>

            {saveFeedback ? (
              <span
                data-testid="settings-save-feedback"
                className={`inline-flex items-center gap-1.5 text-[11px] font-medium truncate ${
                  saveFeedback.type === 'error'
                    ? 'text-studio-danger'
                    : 'text-studio-success'
                }`}
              >
                {saveFeedback.type === 'error' ? (
                  <AlertCircle className="h-3.5 w-3.5 shrink-0" />
                ) : (
                  <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
                )}
                <span className="truncate">{saveFeedback.text}</span>
              </span>
            ) : (
              settingsPath && (
                <span
                  className="hidden sm:inline font-mono text-[10px] text-studio-text-muted truncate max-w-[300px]"
                  title={settingsPath}
                >
                  Synced: {settingsPath}
                </span>
              )
            )}
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={onClose}
              className="studio-btn h-7 px-3 text-xs"
            >
              Close
            </button>
            <button
              type="button"
              data-testid="settings-save-btn"
              disabled={isSaving}
              onClick={handleSaveAll}
              className="studio-btn-accent h-7 px-3.5 text-xs"
            >
              {isSaving ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  <span>Saving…</span>
                </>
              ) : (
                <>
                  <Save className="h-3.5 w-3.5" />
                  <span>Save Settings</span>
                </>
              )}
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}
