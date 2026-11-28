import React, { useEffect } from 'react';
import { Keyboard, X } from 'lucide-react';

export const DEFAULT_SHORTCUTS = {
  nextPage: 'PageDown',
  prevPage: 'PageUp',
  nextIssue: 'Alt+]',
  findReplace: 'Ctrl+H',
  toggleCompare: '\\',
  loupe: 'Z',
  readingPreview: 'F11',
  copyPage: 'Ctrl+Shift+C',
  runPipeline: 'Ctrl+Enter',
};

const SHORTCUT_LABELS = [
  ['nextPage', 'Next Page'],
  ['prevPage', 'Previous Page'],
  ['nextIssue', 'Next QC Issue'],
  ['findReplace', 'Find & Replace'],
  ['toggleCompare', 'Hold Original Compare'],
  ['loupe', 'Hold 60fps Loupe'],
  ['readingPreview', 'Distraction-Free Reading Preview'],
  ['copyPage', 'Copy Rendered Page to Clipboard'],
  ['runPipeline', 'Run Active Pipeline'],
];

export default function ShortcutsModal({ isOpen, onClose, shortcuts = DEFAULT_SHORTCUTS }) {
  useEffect(() => {
    if (!isOpen) return undefined;
    const onKeyDown = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const merged = { ...DEFAULT_SHORTCUTS, ...(shortcuts || {}) };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 select-none"
      role="dialog"
      aria-modal="true"
      aria-label="Keyboard Shortcuts"
      onClick={onClose}
    >
      <div
        className="w-full max-w-md rounded-studio-md border border-studio-border bg-studio-panel shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="h-10 px-4 border-b border-studio-border flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-studio-text">
            <Keyboard className="h-4 w-4 text-studio-accent" />
            <span>Keyboard Shortcuts</span>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close Shortcuts"
            className="h-7 w-7 rounded-studio inline-flex items-center justify-center text-studio-text-muted hover:text-studio-text hover:bg-studio-hover cursor-pointer"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="p-4 space-y-2 max-h-[70vh] overflow-y-auto text-xs">
          {SHORTCUT_LABELS.map(([key, label]) => (
            <div
              key={key}
              className="flex items-center justify-between gap-3 py-1.5 border-b border-studio-border/50 last:border-b-0"
            >
              <span className="text-studio-text-secondary">{label}</span>
              <kbd className="px-2 py-0.5 rounded-studio-sm border border-studio-border bg-studio-surface font-mono text-[11px] text-studio-text">
                {merged[key]}
              </kbd>
            </div>
          ))}
          <div className="flex items-center justify-between gap-3 py-1.5">
            <span className="text-studio-text-secondary">Show This Cheat Sheet</span>
            <kbd className="px-2 py-0.5 rounded-studio-sm border border-studio-border bg-studio-surface font-mono text-[11px] text-studio-text">
              ?
            </kbd>
          </div>
        </div>
      </div>
    </div>
  );
}
