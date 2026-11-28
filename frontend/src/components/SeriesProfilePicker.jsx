import React, { useCallback, useEffect, useRef, useState } from 'react';
import { BookMarked, BookmarkPlus, Check, Trash2, X } from 'lucide-react';
import useEditorStore from '../store/useEditorStore.js';

export async function applySeriesProfileToStore(profile, apiBaseUrl, { syncBackend = true } = {}) {
  if (!profile || typeof profile !== 'object') return;
  const glossaryMap = {};
  if (Array.isArray(profile.glossary)) {
    for (const row of profile.glossary) {
      const s = String(row?.src || '').trim();
      const d = String(row?.dst || '').trim();
      if (s && d) glossaryMap[s] = d;
    }
  } else if (profile.glossary && typeof profile.glossary === 'object') {
    Object.assign(glossaryMap, profile.glossary);
  }

  const state = useEditorStore.getState();
  state.hydratePipelineConfig({
    glossary: glossaryMap,
    custom_instructions: profile.custom_instructions ?? '',
    translation_tone: profile.translation_tone || 'standard',
    honorifics_policy: profile.honorifics_policy || 'keep',
    source_lang: profile.source_lang || state.pipelineConfig.sourceLang || 'ja',
    target_lang: profile.target_lang || state.pipelineConfig.targetLang || 'id',
    detector_precision: profile.detector_precision || state.pipelineConfig.detectorPrecision || 'balanced',
  });

  if (syncBackend && apiBaseUrl) {
    const payload = useEditorStore.getState().buildBackendSettingsPayload();
    try {
      await fetch(`${String(apiBaseUrl).replace(/\/+$/, '')}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
    } catch {
      // ignore network errors when syncing settings
    }
  }
}

export default function SeriesProfilePicker({
  apiBaseUrl,
  activeProfileSlug = '',
  onSelectProfileSlug,
  defaultTitle = '',
  disabled = false,
  profilesVersion = 0,
  onProfilesChanged,
}) {
  const [profiles, setProfiles] = useState([]);
  const [isPopoverOpen, setIsPopoverOpen] = useState(false);
  const [nameInput, setNameInput] = useState('');
  const [keywordsInput, setKeywordsInput] = useState('');
  const [saving, setSaving] = useState(false);
  const popoverRef = useRef(null);

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
      // ignore offline probe errors
    }
  }, [base]);

  useEffect(() => {
    fetchProfiles();
  }, [fetchProfiles, profilesVersion]);

  useEffect(() => {
    if (!isPopoverOpen) return undefined;
    const onOutside = (e) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target)) {
        setIsPopoverOpen(false);
      }
    };
    window.addEventListener('mousedown', onOutside);
    return () => window.removeEventListener('mousedown', onOutside);
  }, [isPopoverOpen]);

  const handleSelectChange = (e) => {
    const slug = e.target.value;
    onSelectProfileSlug?.(slug);
    if (!slug) return;
    const found = profiles.find((p) => p.slug === slug);
    if (found) {
      applySeriesProfileToStore(found, base, { syncBackend: true });
    }
  };

  const openSavePopover = () => {
    const current = profiles.find((p) => p.slug === activeProfileSlug);
    setNameInput(current?.name || defaultTitle || 'My Manga Series');
    setKeywordsInput(
      Array.isArray(current?.match_keywords) ? current.match_keywords.join(', ') : ''
    );
    setIsPopoverOpen(true);
  };

  const handleSaveProfile = async (e) => {
    e.preventDefault();
    const cleanName = nameInput.trim();
    if (!cleanName || saving) return;
    setSaving(true);
    try {
      const cfg = useEditorStore.getState().pipelineConfig;
      const glossaryArr = Object.entries(cfg.glossary || {})
        .filter(([s, d]) => String(s).trim() && String(d).trim())
        .map(([src, dst]) => ({ src: String(src).trim(), dst: String(dst).trim() }));
      const matchKeywords = keywordsInput
        .split(',')
        .map((k) => k.trim())
        .filter(Boolean);

      const resp = await fetch(`${base}/api/series-profiles`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: cleanName,
          match_keywords: matchKeywords,
          glossary: glossaryArr,
          custom_instructions: cfg.customInstructions || '',
          translation_tone: cfg.translationTone || 'standard',
          honorifics_policy: cfg.honorificsPolicy || 'keep',
          source_lang: cfg.sourceLang || 'ja',
          target_lang: cfg.targetLang || 'id',
          detector_precision: cfg.detectorPrecision || 'balanced',
        }),
      });
      if (resp.ok) {
        const data = await resp.json();
        if (data?.profile?.slug) {
          onSelectProfileSlug?.(data.profile.slug);
        }
        await fetchProfiles();
        onProfilesChanged?.();
        setIsPopoverOpen(false);
      }
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteProfile = async (slug, ev) => {
    ev.stopPropagation();
    if (!slug) return;
    try {
      const resp = await fetch(
        `${base}/api/series-profiles/${encodeURIComponent(slug)}`,
        { method: 'DELETE' }
      );
      if (resp.ok) {
        if (activeProfileSlug === slug) {
          onSelectProfileSlug?.('');
        }
        await fetchProfiles();
        onProfilesChanged?.();
      }
    } catch {
      // ignore
    }
  };

  return (
    <div className="relative inline-flex items-center gap-1" ref={popoverRef}>
      <div
        className="inline-flex items-center gap-1 h-6 rounded-studio border border-studio-border bg-studio-surface px-1.5"
        title="Active Per-Series Profile (Glossary, Instructions, Tone & Honorifics)"
      >
        <BookMarked className="h-3 w-3 text-studio-accent shrink-0" />
        <select
          aria-label="Series Profile"
          value={activeProfileSlug}
          onChange={handleSelectChange}
          disabled={disabled}
          className="bg-transparent text-[11px] font-medium text-studio-text focus:outline-none cursor-pointer disabled:opacity-50 max-w-[130px]"
        >
          <option value="">Default (No Profile)</option>
          {profiles.map((p) => (
            <option key={p.slug} value={p.slug}>
              {p.name}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={openSavePopover}
          disabled={disabled}
          title="Save Current Settings as Series Profile"
          className="p-0.5 rounded hover:bg-studio-hover text-studio-text-secondary hover:text-studio-accent cursor-pointer disabled:opacity-40"
        >
          <BookmarkPlus className="h-3 w-3" />
        </button>
      </div>

      {isPopoverOpen && (
        <div className="absolute left-0 top-full mt-1 z-50 w-72 rounded-studio-md border border-studio-border bg-studio-panel p-3 shadow-2xl space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-studio-text">
              Save Series Profile
            </span>
            <button
              type="button"
              onClick={() => setIsPopoverOpen(false)}
              className="p-0.5 rounded text-studio-text-muted hover:text-studio-text cursor-pointer"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>

          <form onSubmit={handleSaveProfile} className="space-y-2">
            <div>
              <label className="text-[10px] font-medium text-studio-text-secondary block mb-0.5">
                Series Name
              </label>
              <input
                type="text"
                autoFocus
                value={nameInput}
                onChange={(e) => setNameInput(e.target.value)}
                placeholder="e.g. One Piece"
                className="studio-input w-full h-7 px-2 text-xs"
              />
            </div>
            <div>
              <label className="text-[10px] font-medium text-studio-text-secondary block mb-0.5">
                Match Keywords (comma-separated)
              </label>
              <input
                type="text"
                value={keywordsInput}
                onChange={(e) => setKeywordsInput(e.target.value)}
                placeholder="e.g. OP, Mugiwara, Eiichiro"
                className="studio-input w-full h-7 px-2 text-xs"
              />
            </div>
            <button
              type="submit"
              disabled={!nameInput.trim() || saving}
              className="w-full h-7 rounded-studio-sm bg-studio-accent text-studio-accent-fg text-xs font-medium inline-flex items-center justify-center gap-1 hover:opacity-90 disabled:opacity-40 cursor-pointer"
            >
              <Check className="h-3 w-3" />
              <span>{saving ? 'Saving…' : 'Save Current as Series Profile'}</span>
            </button>
          </form>

          {profiles.length > 0 && (
            <div className="pt-1.5 border-t border-studio-border space-y-1 max-h-32 overflow-y-auto">
              <div className="text-[10px] font-medium text-studio-text-muted">
                Saved Profiles ({profiles.length})
              </div>
              {profiles.map((p) => (
                <div
                  key={p.slug}
                  className="flex items-center justify-between gap-1 text-[11px] px-1.5 py-1 rounded bg-studio-surface"
                >
                  <button
                    type="button"
                    onClick={() => {
                      onSelectProfileSlug?.(p.slug);
                      applySeriesProfileToStore(p, base, { syncBackend: true });
                      setIsPopoverOpen(false);
                    }}
                    className="truncate text-left flex-1 text-studio-text hover:text-studio-accent cursor-pointer"
                  >
                    {p.name}
                  </button>
                  <button
                    type="button"
                    onClick={(ev) => handleDeleteProfile(p.slug, ev)}
                    title={`Delete ${p.name}`}
                    className="p-0.5 rounded text-studio-text-muted hover:text-studio-danger cursor-pointer"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
