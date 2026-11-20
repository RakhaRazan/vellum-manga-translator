import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Archive,
  BookOpen,
  Check,
  Columns2,
  Copy,
  Cpu,
  Database,
  Download,
  ExternalLink,
  FileArchive,
  FileCode2,
  FileImage,
  FileText,
  FolderOpen,
  Info,
  Keyboard,
  Layers,
  Maximize2,
  Monitor,
  Moon,
  Play,
  PlusSquare,
  Redo2,
  Save,
  ScanSearch,
  Settings,
  Sparkles,
  Sun,
  Trash2,
  Type,
  Undo2,
  Wand2,
  X,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import useEditorStore from '../store/useEditorStore.js';
import { formatExecutionProvider } from './StatusBar.jsx';

const MENU_ORDER = ['file', 'edit', 'process', 'view', 'help'];

const THEME_OPTIONS = [
  { value: 'dark', icon: Moon, title: 'Sumi Graphite (Dark)' },
  { value: 'light', icon: Sun, title: 'Manga Paper (Light)' },
  { value: 'system', icon: Monitor, title: 'Follow System Theme' },
];

const SHORTCUT_GROUPS = [
  {
    title: 'File',
    items: [
      { keys: 'Ctrl+O', action: 'Import Pages' },
      { keys: 'Ctrl+Shift+O', action: 'Open Project (.mflow.json)' },
      { keys: 'Ctrl+S', action: 'Save Project (.mflow.json)' },
      { keys: 'Ctrl+Shift+E', action: 'Export Page (.png)' },
    ],
  },
  {
    title: 'Edit',
    items: [
      { keys: 'Ctrl+Z', action: 'Undo' },
      { keys: 'Ctrl+Y', action: 'Redo' },
      { keys: 'Ctrl+A', action: 'Select All Layers' },
      { keys: 'Ctrl+D', action: 'Duplicate Selected' },
      { keys: 'Del', action: 'Delete Selected' },
      { keys: 'T', action: 'Add Manual Text Box' },
    ],
  },
  {
    title: 'Process',
    items: [
      { keys: 'Ctrl+Shift+Enter', action: 'Process Project' },
      { keys: 'Alt+1', action: 'Run Detection' },
      { keys: 'Alt+2', action: 'Run OCR' },
      { keys: 'Alt+3', action: 'Run Inpainting' },
      { keys: 'Alt+4', action: 'Run Translation' },
    ],
  },
  {
    title: 'View & Help',
    items: [
      { keys: 'Ctrl++ / Ctrl+-', action: 'Zoom In / Zoom Out' },
      { keys: 'Ctrl+0', action: 'Fit Screen' },
      { keys: 'Ctrl+1', action: 'Zoom 100%' },
      { keys: 'Tab', action: 'Toggle Zen Mode' },
      { keys: '?', action: 'Keyboard Shortcuts' },
    ],
  },
];

function MenuSeparator() {
  return <div className="my-1 h-px bg-studio-border" role="separator" />;
}

function MenuSectionLabel({ children }) {
  return (
    <div className="px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-studio-text-muted">
      {children}
    </div>
  );
}

function MenuItem({
  icon: Icon,
  label,
  shortcut,
  checked = null,
  disabled = false,
  danger = false,
  onClick,
  href = null,
}) {
  const baseClass = `w-full px-2.5 h-7 flex items-center justify-between gap-4 text-[11px] rounded-studio-sm transition-colors text-left ${
    disabled
      ? 'opacity-40 pointer-events-none text-studio-text-muted'
      : danger
      ? 'text-studio-danger hover:bg-studio-danger/10 cursor-pointer'
      : 'text-studio-text hover:bg-studio-hover cursor-pointer'
  }`;

  const content = (
    <>
      <span className="flex items-center gap-2 min-w-0">
        {checked !== null ? (
          <span className="w-3.5 h-3.5 inline-flex items-center justify-center shrink-0">
            {checked && <Check className="h-3.5 w-3.5 text-studio-accent" />}
          </span>
        ) : Icon ? (
          <Icon
            className={`h-3.5 w-3.5 shrink-0 ${
              danger ? 'text-studio-danger' : 'text-studio-text-secondary'
            }`}
          />
        ) : (
          <span className="w-3.5 h-3.5 shrink-0" />
        )}
        <span className="truncate">{label}</span>
      </span>
      {shortcut && (
        <span className="font-mono text-[10px] text-studio-text-muted shrink-0">
          {shortcut}
        </span>
      )}
    </>
  );

  if (href) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        onClick={onClick}
        className={baseClass}
        role="menuitem"
      >
        {content}
      </a>
    );
  }

  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={baseClass}
      role="menuitem"
    >
      {content}
    </button>
  );
}

/**
 * Compact 28px native desktop application menu bar (`File`, `Edit`, `Process`, `View`, `Help`),
 * wired directly to `useEditorStore`, complete with global keyboard shortcuts,
 * Keyboard Shortcuts modal, and About Vellum modal.
 */
export default function MenuBar({
  onImportFiles,
  onImportArchive,
  onExportImage,
  onExportCbz,
  onExportPdf,
  onSaveProject,
  onOpenProject,
  onProcessProject,
  onProcessSelectedPages,
  onProcessSelectedLayers,
  onRunStage,
  onFitScreen,
  onOpenModelsModal,
  isProcessing = false,
  isWorkspaceUnlocked = true,
  githubUrl = 'https://github.com/vellum-ai/vellum',
  defaultOpenMenu = null,
}) {
  const [openMenu, setOpenMenu] = useState(defaultOpenMenu);
  const menuBarRef = useRef(null);
  const pageFilesInputRef = useRef(null);
  const folderInputRef = useRef(null);
  const archiveInputRef = useRef(null);
  const projectInputRef = useRef(null);

  const {
    pagesCount,
    activePage,
    selectedPagesCount,
    selectedLayersCount,
    canUndo,
    canRedo,
    viewLayers,
    zenMode,
    theme,
    activeModal,
    executionProvider,
    modelsStatus,
    undo,
    redo,
    selectAllLayers,
    deleteSelectedLayers,
    duplicateSelectedLayers,
    addBlock,
    toggleViewLayer,
    toggleZenMode,
    setTheme,
    zoomIn,
    zoomOut,
    setCanvasZoom,
    openModal,
    closeModal,
  } = useEditorStore(
    useShallow((s) => ({
      pagesCount: s.pages.length,
      activePage: s.pages.find((p) => p.id === s.activePageId) || null,
      selectedPagesCount: s.selectedPageIds.length,
      selectedLayersCount:
        s.selectedLayerIds.length > 0
          ? s.selectedLayerIds.length
          : s.selectedBlockId
          ? 1
          : 0,
      canUndo: s.history.past.length > 0,
      canRedo: s.history.future.length > 0,
      viewLayers: s.viewLayers,
      zenMode: s.zenMode,
      theme: s.theme,
      activeModal: s.activeModal,
      executionProvider: s.runtimeStatus.executionProvider,
      modelsStatus: s.runtimeStatus.modelsStatus,
      undo: s.undo,
      redo: s.redo,
      selectAllLayers: s.selectAllLayers,
      deleteSelectedLayers: s.deleteSelectedLayers,
      duplicateSelectedLayers: s.duplicateSelectedLayers,
      addBlock: s.addBlock,
      toggleViewLayer: s.toggleViewLayer,
      toggleZenMode: s.toggleZenMode,
      setTheme: s.setTheme,
      zoomIn: s.zoomIn,
      zoomOut: s.zoomOut,
      setCanvasZoom: s.setCanvasZoom,
      openModal: s.openModal,
      closeModal: s.closeModal,
    }))
  );

  const hasPages = pagesCount > 0;
  const hasActivePage = Boolean(activePage);
  const canExportPage = Boolean(
    activePage &&
      (activePage.cleanedImage || activePage.inpaintedUrl || activePage.originalUrl)
  );
  const hasBlocksOnPage = Boolean(activePage?.blocks?.length);

  const runAndClose = useCallback((fn) => {
    setOpenMenu(null);
    if (typeof fn === 'function') {
      fn();
    }
  }, []);

  // Attach non-standard folder upload attributes imperatively to avoid React DOM prop warnings
  useEffect(() => {
    if (folderInputRef.current) {
      folderInputRef.current.setAttribute('webkitdirectory', '');
      folderInputRef.current.setAttribute('directory', '');
    }
  }, []);

  // Close dropdown on click outside
  useEffect(() => {
    if (!openMenu) return undefined;
    const handlePointerDown = (e) => {
      if (menuBarRef.current && !menuBarRef.current.contains(e.target)) {
        setOpenMenu(null);
      }
    };
    window.addEventListener('mousedown', handlePointerDown);
    return () => window.removeEventListener('mousedown', handlePointerDown);
  }, [openMenu]);

  // Global desktop keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        if (openMenu) {
          e.preventDefault();
          setOpenMenu(null);
          return;
        }
        if (activeModal) {
          e.preventDefault();
          closeModal();
          return;
        }
      }

      if (openMenu && (e.key === 'ArrowRight' || e.key === 'ArrowLeft')) {
        e.preventDefault();
        const idx = MENU_ORDER.indexOf(openMenu);
        const nextIdx =
          e.key === 'ArrowRight'
            ? (idx + 1) % MENU_ORDER.length
            : (idx - 1 + MENU_ORDER.length) % MENU_ORDER.length;
        setOpenMenu(MENU_ORDER[nextIdx]);
        return;
      }

      const target = e.target;
      const isEditableField =
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable);

      if (isEditableField) return;

      const mod = e.ctrlKey || e.metaKey;
      const key = e.key.toLowerCase();

      // Help shortcuts
      if ((e.key === '?' || e.key === 'F1') && !mod && !e.altKey) {
        e.preventDefault();
        openModal('shortcuts');
        return;
      }

      // Zen Mode (Tab)
      if (e.key === 'Tab' && !mod && !e.altKey && !activeModal) {
        e.preventDefault();
        toggleZenMode();
        return;
      }

      // Add Manual Text Box (T)
      if (key === 't' && !mod && !e.altKey && !e.shiftKey && hasActivePage && !activeModal) {
        e.preventDefault();
        addBlock();
        return;
      }

      // Delete Selected Layers (Delete / Backspace)
      if ((e.key === 'Delete' || e.key === 'Backspace') && !mod && !e.altKey && !activeModal) {
        if (selectedLayersCount > 0) {
          e.preventDefault();
          deleteSelectedLayers();
        }
        return;
      }

      // Stage shortcuts (Alt+1..4)
      if (e.altKey && !mod && !isProcessing && isWorkspaceUnlocked && hasActivePage) {
        if (e.key === '1') {
          e.preventDefault();
          onRunStage?.('detect');
          return;
        }
        if (e.key === '2') {
          e.preventDefault();
          onRunStage?.('ocr');
          return;
        }
        if (e.key === '3') {
          e.preventDefault();
          onRunStage?.('inpaint');
          return;
        }
        if (e.key === '4') {
          e.preventDefault();
          onRunStage?.('translate');
          return;
        }
      }

      if (!mod) return;

      // File shortcuts
      if (key === 'o' && !e.shiftKey && isWorkspaceUnlocked) {
        e.preventDefault();
        pageFilesInputRef.current?.click();
        return;
      }
      if (key === 'o' && e.shiftKey && isWorkspaceUnlocked) {
        e.preventDefault();
        projectInputRef.current?.click();
        return;
      }
      if (key === 's' && !e.shiftKey) {
        e.preventDefault();
        if (hasPages) onSaveProject?.();
        return;
      }
      if (key === 'e' && e.shiftKey) {
        e.preventDefault();
        if (canExportPage) onExportImage?.('png');
        return;
      }

      // Settings shortcut (Ctrl+,)
      if (e.key === ',' && !e.shiftKey) {
        e.preventDefault();
        openModal('settings');
        return;
      }

      // Edit shortcuts
      if (key === 'z' && !e.shiftKey) {
        e.preventDefault();
        undo();
        return;
      }
      if (key === 'y' || (key === 'z' && e.shiftKey)) {
        e.preventDefault();
        redo();
        return;
      }
      if (key === 'a' && !e.shiftKey && hasBlocksOnPage) {
        e.preventDefault();
        selectAllLayers();
        return;
      }
      if (key === 'd' && !e.shiftKey && selectedLayersCount > 0) {
        e.preventDefault();
        duplicateSelectedLayers();
        return;
      }

      // Process Project shortcut (Ctrl+Shift+Enter)
      if (e.key === 'Enter' && e.shiftKey && hasPages && !isProcessing && isWorkspaceUnlocked) {
        e.preventDefault();
        onProcessProject?.();
        return;
      }

      // View zoom shortcuts
      if (e.key === '=' || e.key === '+') {
        e.preventDefault();
        zoomIn();
        return;
      }
      if (e.key === '-') {
        e.preventDefault();
        zoomOut();
        return;
      }
      if (e.key === '0') {
        e.preventDefault();
        onFitScreen?.();
        return;
      }
      if (e.key === '1') {
        e.preventDefault();
        setCanvasZoom(1);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    openMenu,
    activeModal,
    closeModal,
    openModal,
    toggleZenMode,
    hasActivePage,
    hasPages,
    hasBlocksOnPage,
    canExportPage,
    selectedLayersCount,
    isProcessing,
    isWorkspaceUnlocked,
    addBlock,
    deleteSelectedLayers,
    duplicateSelectedLayers,
    selectAllLayers,
    undo,
    redo,
    zoomIn,
    zoomOut,
    setCanvasZoom,
    onFitScreen,
    onSaveProject,
    onExportImage,
    onProcessProject,
    onRunStage,
  ]);

  const handleFileInput = (e, cb) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0 && cb) {
      cb(files);
    }
    e.target.value = '';
  };

  const installedModelsCount = modelsStatus?.models
    ? Object.values(modelsStatus.models).filter((m) => m.installed).length
    : 4;
  const totalModelsCount = modelsStatus?.models
    ? Object.keys(modelsStatus.models).length
    : 4;
  const providerMeta = formatExecutionProvider(executionProvider);

  const renderTrigger = (id, label) => {
    const isOpen = openMenu === id;
    return (
      <button
        type="button"
        onClick={() => setOpenMenu((prev) => (prev === id ? null : id))}
        onMouseEnter={() => {
          if (openMenu && openMenu !== id) {
            setOpenMenu(id);
          }
        }}
        aria-haspopup="menu"
        aria-expanded={isOpen}
        className={`h-6 px-2 rounded-studio-sm text-[11px] font-medium transition-colors cursor-pointer ${
          isOpen
            ? 'bg-studio-accent/15 text-studio-accent'
            : 'text-studio-text-secondary hover:text-studio-text hover:bg-studio-hover'
        }`}
      >
        {label}
      </button>
    );
  };

  return (
    <>
      <nav
        ref={menuBarRef}
        aria-label="Application Menu Bar"
        className="h-7 min-h-[28px] max-h-[28px] shrink-0 border-b border-studio-border bg-studio-panel px-2 flex items-center justify-between gap-2 z-40 select-none"
      >
        {/* Hidden File Inputs for Menu Commands */}
        <input
          ref={pageFilesInputRef}
          type="file"
          multiple
          accept="image/jpeg,image/png,image/webp,image/bmp,image/*"
          onChange={(e) => handleFileInput(e, onImportFiles)}
          className="hidden"
          data-testid="menu-import-pages-input"
        />
        <input
          ref={folderInputRef}
          type="file"
          multiple
          onChange={(e) => handleFileInput(e, onImportFiles)}
          className="hidden"
          data-testid="menu-import-folder-input"
        />
        <input
          ref={archiveInputRef}
          type="file"
          accept=".cbz,.zip,application/zip,application/vnd.comicbook+zip"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file && onImportArchive) onImportArchive(file);
            e.target.value = '';
          }}
          className="hidden"
          data-testid="menu-import-archive-input"
        />
        <input
          ref={projectInputRef}
          type="file"
          accept=".json,.mflow.json,application/json"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file && onOpenProject) onOpenProject(file);
            e.target.value = '';
          }}
          className="hidden"
          data-testid="menu-open-project-input"
        />

        {/* Left: Compact Brand Mark + Native Top-Level Menus */}
        <div className="flex items-center gap-0.5 min-w-0" role="menubar">
          <div className="flex items-center gap-1.5 pr-2 mr-1 border-r border-studio-border shrink-0">
            <img
              src="/vellum-logo.svg"
              alt="Vellum"
              className="h-4 w-4 shrink-0"
            />
            <span className="text-[11px] font-semibold tracking-tight text-studio-text hidden sm:inline">
              Vellum
            </span>
          </div>

          {/* FILE MENU */}
          <div className="relative">
            {renderTrigger('file', 'File')}
            {openMenu === 'file' && (
              <div
                role="menu"
                className="absolute left-0 top-full mt-0.5 w-60 rounded-studio border border-studio-border bg-studio-panel p-1 shadow-lg z-50"
              >
                <MenuItem
                  icon={FileImage}
                  label="Import Pages…"
                  shortcut="Ctrl+O"
                  disabled={!isWorkspaceUnlocked}
                  onClick={() => runAndClose(() => pageFilesInputRef.current?.click())}
                />
                <MenuItem
                  icon={FolderOpen}
                  label="Import Folder…"
                  disabled={!isWorkspaceUnlocked}
                  onClick={() => runAndClose(() => folderInputRef.current?.click())}
                />
                <MenuItem
                  icon={FileArchive}
                  label="Import .cbz / .zip…"
                  disabled={!isWorkspaceUnlocked}
                  onClick={() => runAndClose(() => archiveInputRef.current?.click())}
                />
                <MenuSeparator />
                <MenuItem
                  icon={Download}
                  label="Export Page (.png)"
                  shortcut="Ctrl+Shift+E"
                  disabled={!canExportPage || isProcessing}
                  onClick={() => runAndClose(() => onExportImage?.('png'))}
                />
                <MenuItem
                  icon={Download}
                  label="Export Page (.jpg)"
                  disabled={!canExportPage || isProcessing}
                  onClick={() => runAndClose(() => onExportImage?.('jpg'))}
                />
                <MenuItem
                  icon={Archive}
                  label="Export .cbz"
                  disabled={!hasPages || isProcessing}
                  onClick={() => runAndClose(onExportCbz)}
                />
                <MenuItem
                  icon={FileText}
                  label="Export .pdf"
                  disabled={!hasPages || isProcessing}
                  onClick={() => runAndClose(onExportPdf)}
                />
                <MenuSeparator />
                <MenuItem
                  icon={Save}
                  label="Save Project (.mflow.json)"
                  shortcut="Ctrl+S"
                  disabled={!hasPages}
                  onClick={() => runAndClose(onSaveProject)}
                />
                <MenuItem
                  icon={FileCode2}
                  label="Open Project…"
                  shortcut="Ctrl+Shift+O"
                  disabled={!isWorkspaceUnlocked}
                  onClick={() => runAndClose(() => projectInputRef.current?.click())}
                />
              </div>
            )}
          </div>

          {/* EDIT MENU */}
          <div className="relative">
            {renderTrigger('edit', 'Edit')}
            {openMenu === 'edit' && (
              <div
                role="menu"
                className="absolute left-0 top-full mt-0.5 w-56 rounded-studio border border-studio-border bg-studio-panel p-1 shadow-lg z-50"
              >
                <MenuItem
                  icon={Undo2}
                  label="Undo"
                  shortcut="Ctrl+Z"
                  disabled={!canUndo}
                  onClick={() => runAndClose(undo)}
                />
                <MenuItem
                  icon={Redo2}
                  label="Redo"
                  shortcut="Ctrl+Y"
                  disabled={!canRedo}
                  onClick={() => runAndClose(redo)}
                />
                <MenuSeparator />
                <MenuItem
                  icon={Layers}
                  label="Select All Layers"
                  shortcut="Ctrl+A"
                  disabled={!hasBlocksOnPage}
                  onClick={() => runAndClose(selectAllLayers)}
                />
                <MenuItem
                  icon={Copy}
                  label="Duplicate Selected"
                  shortcut="Ctrl+D"
                  disabled={selectedLayersCount === 0}
                  onClick={() => runAndClose(duplicateSelectedLayers)}
                />
                <MenuItem
                  icon={Trash2}
                  label="Delete Selected"
                  shortcut="Del"
                  danger
                  disabled={selectedLayersCount === 0}
                  onClick={() => runAndClose(deleteSelectedLayers)}
                />
                <MenuSeparator />
                <MenuItem
                  icon={PlusSquare}
                  label="Add Manual Text Box"
                  shortcut="T"
                  disabled={!hasActivePage}
                  onClick={() => runAndClose(() => addBlock())}
                />
                <MenuSeparator />
                <MenuItem
                  icon={Settings}
                  label="Settings…"
                  shortcut="Ctrl+,"
                  onClick={() => runAndClose(() => openModal('settings'))}
                />
              </div>
            )}
          </div>

          {/* PROCESS MENU */}
          <div className="relative">
            {renderTrigger('process', 'Process')}
            {openMenu === 'process' && (
              <div
                role="menu"
                className="absolute left-0 top-full mt-0.5 w-64 rounded-studio border border-studio-border bg-studio-panel p-1 shadow-lg z-50"
              >
                <MenuItem
                  icon={Play}
                  label={`Process Project${hasPages ? ` (${pagesCount})` : ''}`}
                  shortcut="Ctrl+Shift+Enter"
                  disabled={!isWorkspaceUnlocked || !hasPages || isProcessing}
                  onClick={() => runAndClose(onProcessProject)}
                />
                <MenuItem
                  icon={Play}
                  label={`Process Selected Pages${
                    selectedPagesCount > 0 ? ` (${selectedPagesCount})` : ''
                  }`}
                  disabled={
                    !isWorkspaceUnlocked ||
                    (selectedPagesCount === 0 && !hasActivePage) ||
                    isProcessing
                  }
                  onClick={() => runAndClose(onProcessSelectedPages)}
                />
                <MenuItem
                  icon={Layers}
                  label={`Process Selected Layers${
                    selectedLayersCount > 0 ? ` (${selectedLayersCount})` : ''
                  }`}
                  disabled={
                    !isWorkspaceUnlocked ||
                    !hasActivePage ||
                    selectedLayersCount === 0 ||
                    isProcessing
                  }
                  onClick={() => runAndClose(onProcessSelectedLayers)}
                />
                <MenuSeparator />
                <MenuSectionLabel>Single Stages</MenuSectionLabel>
                <MenuItem
                  icon={ScanSearch}
                  label="Run Detection"
                  shortcut="Alt+1"
                  disabled={!isWorkspaceUnlocked || !hasActivePage || isProcessing}
                  onClick={() => runAndClose(() => onRunStage?.('detect'))}
                />
                <MenuItem
                  icon={Type}
                  label="Run OCR"
                  shortcut="Alt+2"
                  disabled={!isWorkspaceUnlocked || !hasActivePage || isProcessing}
                  onClick={() => runAndClose(() => onRunStage?.('ocr'))}
                />
                <MenuItem
                  icon={Wand2}
                  label="Run Inpainting"
                  shortcut="Alt+3"
                  disabled={!isWorkspaceUnlocked || !hasActivePage || isProcessing}
                  onClick={() => runAndClose(() => onRunStage?.('inpaint'))}
                />
                <MenuItem
                  icon={Sparkles}
                  label="Run Translation"
                  shortcut="Alt+4"
                  disabled={!isWorkspaceUnlocked || !hasActivePage || isProcessing}
                  onClick={() => runAndClose(() => onRunStage?.('translate'))}
                />
              </div>
            )}
          </div>

          {/* VIEW MENU */}
          <div className="relative">
            {renderTrigger('view', 'View')}
            {openMenu === 'view' && (
              <div
                role="menu"
                className="absolute left-0 top-full mt-0.5 w-72 rounded-studio border border-studio-border bg-studio-panel p-1 shadow-lg z-50"
              >
                <MenuItem
                  icon={ZoomIn}
                  label="Zoom In"
                  shortcut="Ctrl++"
                  onClick={() => runAndClose(zoomIn)}
                />
                <MenuItem
                  icon={ZoomOut}
                  label="Zoom Out"
                  shortcut="Ctrl+-"
                  onClick={() => runAndClose(zoomOut)}
                />
                <MenuItem
                  icon={Maximize2}
                  label="Fit Screen"
                  shortcut="Ctrl+0"
                  onClick={() => runAndClose(onFitScreen)}
                />
                <MenuItem
                  label="100%"
                  shortcut="Ctrl+1"
                  onClick={() => runAndClose(() => setCanvasZoom(1))}
                />
                <MenuSeparator />
                <MenuSectionLabel>Layer Toggles</MenuSectionLabel>
                <MenuItem
                  label="Original Image"
                  checked={Boolean(viewLayers.original)}
                  onClick={() => toggleViewLayer('original')}
                />
                <MenuItem
                  label="Inpainted Layer"
                  checked={Boolean(viewLayers.inpainted)}
                  onClick={() => toggleViewLayer('inpainted')}
                />
                <MenuItem
                  label="Translated Text"
                  checked={Boolean(viewLayers.text)}
                  onClick={() => toggleViewLayer('text')}
                />
                <MenuItem
                  label="Bounding Boxes & Reading Order #1/#2"
                  checked={Boolean(viewLayers.boxes)}
                  onClick={() => toggleViewLayer('boxes')}
                />
                <MenuItem
                  label="Inpaint Mask"
                  checked={Boolean(viewLayers.mask)}
                  onClick={() => toggleViewLayer('mask')}
                />
                <MenuSeparator />
                <MenuItem
                  icon={Columns2}
                  label="Split-Curtain Before/After Compare"
                  checked={Boolean(viewLayers.splitCompare)}
                  onClick={() => toggleViewLayer('splitCompare')}
                />
                <MenuItem
                  label="Zen Mode"
                  shortcut="Tab"
                  checked={zenMode}
                  onClick={() => runAndClose(toggleZenMode)}
                />
              </div>
            )}
          </div>

          {/* HELP MENU */}
          <div className="relative">
            {renderTrigger('help', 'Help')}
            {openMenu === 'help' && (
              <div
                role="menu"
                className="absolute left-0 top-full mt-0.5 w-56 rounded-studio border border-studio-border bg-studio-panel p-1 shadow-lg z-50"
              >
                <MenuItem
                  icon={ExternalLink}
                  label="GitHub Repository"
                  href={githubUrl}
                  onClick={() => setOpenMenu(null)}
                />
                <MenuSeparator />
                <MenuItem
                  icon={Keyboard}
                  label="Keyboard Shortcuts…"
                  shortcut="?"
                  onClick={() => runAndClose(() => openModal('shortcuts'))}
                />
                <MenuItem
                  icon={Info}
                  label="About Vellum"
                  onClick={() => runAndClose(() => openModal('about'))}
                />
              </div>
            )}
          </div>
        </div>

        {/* Right: Zen indicator, Models button, and Dark/Light/System Theme Switcher */}
        <div className="flex items-center gap-1.5 shrink-0">
          {zenMode && (
            <button
              type="button"
              onClick={toggleZenMode}
              className="h-5 px-1.5 rounded-studio-sm border border-studio-accent/50 bg-studio-accent/15 text-[10px] font-mono font-medium text-studio-accent hover:bg-studio-accent/25 transition-colors cursor-pointer"
              title="Exit Zen Mode (Tab)"
            >
              ZEN · Tab
            </button>
          )}

          {/* Local Models Count Button */}
          <button
            type="button"
            onClick={onOpenModelsModal}
            className="hidden sm:inline-flex items-center gap-1 h-5 px-1.5 rounded-studio-sm border border-studio-border bg-studio-surface hover:bg-studio-hover text-[10px] font-mono text-studio-text-secondary transition-colors cursor-pointer"
            title="Inspect local ONNX & OCR model weights"
          >
            <Database className="h-3 w-3 text-studio-accent" />
            <span className="tabular-nums">
              {installedModelsCount}/{totalModelsCount}
            </span>
          </button>

          {/* Compact Dark / Light / System Theme Switcher */}
          <div
            className="inline-flex items-center h-5 rounded-studio-sm border border-studio-border bg-studio-surface p-0.5 gap-0.5"
            role="group"
            aria-label="Theme"
          >
            {THEME_OPTIONS.map(({ value, icon: Icon, title }) => {
              const isSelected = theme === value;
              return (
                <button
                  key={value}
                  type="button"
                  onClick={() => setTheme(value)}
                  title={title}
                  aria-label={title}
                  aria-pressed={isSelected}
                  className={`h-4 w-4 inline-flex items-center justify-center rounded-[2px] transition-colors cursor-pointer ${
                    isSelected
                      ? 'bg-studio-accent text-studio-accent-fg'
                      : 'text-studio-text-muted hover:text-studio-text hover:bg-studio-hover'
                  }`}
                >
                  <Icon className="h-2.5 w-2.5" />
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Keyboard Shortcuts Modal */}
      {activeModal === 'shortcuts' && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/65 p-4 select-none"
          role="dialog"
          aria-modal="true"
          aria-labelledby="shortcuts-modal-title"
          onClick={closeModal}
        >
          <div
            className="w-full max-w-xl rounded-studio-md border border-studio-border bg-studio-panel overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="h-10 px-4 border-b border-studio-border flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Keyboard className="h-4 w-4 text-studio-accent" />
                <h2
                  id="shortcuts-modal-title"
                  className="text-xs font-semibold uppercase tracking-wider text-studio-text"
                >
                  Keyboard Shortcuts
                </h2>
              </div>
              <button
                type="button"
                onClick={closeModal}
                aria-label="Close Keyboard Shortcuts"
                className="h-6 w-6 inline-flex items-center justify-center rounded-studio-sm text-studio-text-muted hover:text-studio-text hover:bg-studio-hover cursor-pointer"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>

            <div className="p-4 grid grid-cols-1 sm:grid-cols-2 gap-3 max-h-[70vh] overflow-y-auto">
              {SHORTCUT_GROUPS.map((group) => (
                <div
                  key={group.title}
                  className="rounded-studio border border-studio-border bg-studio-surface p-3 space-y-2"
                >
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-studio-accent">
                    {group.title}
                  </div>
                  <div className="space-y-1.5">
                    {group.items.map((item) => (
                      <div
                        key={item.action}
                        className="flex items-center justify-between gap-2 text-[11px]"
                      >
                        <span className="text-studio-text-secondary">
                          {item.action}
                        </span>
                        <kbd className="rounded-studio-sm border border-studio-border bg-studio-panel px-1.5 py-0.5 font-mono text-[10px] text-studio-text shrink-0">
                          {item.keys}
                        </kbd>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <div className="px-4 py-2.5 border-t border-studio-border bg-studio-surface flex items-center justify-between text-[11px] text-studio-text-muted">
              <span>Press Esc to close</span>
              <button
                type="button"
                onClick={closeModal}
                className="studio-btn h-7 px-3 text-xs"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* About Vellum Modal */}
      {activeModal === 'about' && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/65 p-4 select-none"
          role="dialog"
          aria-modal="true"
          aria-labelledby="about-modal-title"
          onClick={closeModal}
        >
          <div
            className="w-full max-w-md rounded-studio-md border border-studio-border bg-studio-panel overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="h-10 px-4 border-b border-studio-border flex items-center justify-between">
              <div className="flex items-center gap-2">
                <img src="/vellum-logo.svg" alt="" className="h-4 w-4 shrink-0" />
                <h2
                  id="about-modal-title"
                  className="text-xs font-semibold uppercase tracking-wider text-studio-text"
                >
                  About Vellum
                </h2>
              </div>
              <button
                type="button"
                onClick={closeModal}
                aria-label="Close About dialog"
                className="h-6 w-6 inline-flex items-center justify-center rounded-studio-sm text-studio-text-muted hover:text-studio-text hover:bg-studio-hover cursor-pointer"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>

            <div className="p-4 space-y-3 text-xs">
              <div className="flex items-center gap-3">
                <img
                  src="/vellum-logo.svg"
                  alt="Vellum"
                  className="h-10 w-10 shrink-0"
                />
                <div>
                  <div className="text-sm font-semibold text-studio-text">
                    Vellum
                  </div>
                  <div className="text-[11px] font-mono text-studio-text-muted">
                    v2 Lightweight Local Architecture
                  </div>
                </div>
              </div>

              <p className="text-studio-text-secondary leading-relaxed">
                Local-first, offline-capable studio for translating Japanese
                Manga, Korean Manhwa, and Chinese Manhua using Comic Text
                Detector (ONNX), manga-ocr / RapidOCR, tile-based LaMa
                inpainting, and context-aware LLM translation.
              </p>

              <div className="rounded-studio border border-studio-border bg-studio-surface p-2.5 space-y-1.5 font-mono text-[11px]">
                <div className="flex items-center justify-between">
                  <span className="text-studio-text-muted flex items-center gap-1.5">
                    <Cpu className="h-3 w-3 text-studio-accent" />
                    Runtime Provider
                  </span>
                  <span className="text-studio-text">{providerMeta.label}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-studio-text-muted flex items-center gap-1.5">
                    <Database className="h-3 w-3 text-studio-accent" />
                    Local Weights
                  </span>
                  <span className="text-studio-text">
                    {installedModelsCount}/{totalModelsCount} Verified
                  </span>
                </div>
              </div>
            </div>

            <div className="px-4 py-2.5 border-t border-studio-border bg-studio-surface flex items-center justify-between">
              <a
                href={githubUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 text-[11px] font-medium text-studio-accent hover:underline"
              >
                <ExternalLink className="h-3 w-3" />
                GitHub
              </a>
              <button
                type="button"
                onClick={closeModal}
                className="studio-btn h-7 px-3 text-xs"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
