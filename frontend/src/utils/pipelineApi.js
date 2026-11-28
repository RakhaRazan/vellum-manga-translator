// Vellum — Standalone Pipeline API, Retry & File/Queue Utilities

export function resolveBackendUrl(url, apiBaseUrl) {
  if (!url) return null;
  if (/^(data:|blob:|https?:\/\/)/i.test(url)) return url;
  if (url.startsWith('/')) {
    const base = String(apiBaseUrl || '').replace(/\/+$/, '');
    return base ? `${base}${url}` : url;
  }
  return url;
}

export const naturalSortCollator = new Intl.Collator(undefined, {
  numeric: true,
  sensitivity: 'base',
});

export async function resolvePageImageFile(pageItem) {
  if (pageItem?.file instanceof File || pageItem?.file instanceof Blob) {
    return pageItem.file;
  }
  const srcUrl = pageItem?.originalUrl || pageItem?.cleanedImage;
  if (!srcUrl) return null;
  const resp = await fetch(srcUrl);
  if (!resp.ok) {
    throw new Error(`Could not fetch source image for ${pageItem.name} (HTTP ${resp.status})`);
  }
  const blob = await resp.blob();
  return new File([blob], pageItem.name || 'page.png', {
    type: blob.type || 'image/png',
  });
}

export const BATCH_QUEUE_STORAGE_KEY = 'vellum.batch_queue.v1';
export const BATCH_QUEUE_PREFS_KEY = 'vellum.batch_queue_prefs.v1';

export function loadPersistedQueuePrefs() {
  try {
    const raw = localStorage.getItem(BATCH_QUEUE_PREFS_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

export function loadPersistedQueueItems() {
  try {
    const raw = localStorage.getItem(BATCH_QUEUE_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((it) => it && it.id && it.path)
      .map((it) => ({
        ...it,
        projectId: it.projectId || `qproj_${String(it.id).replace(/[^a-zA-Z0-9_-]/g, '')}`,
        selected: it.selected !== false,
        status:
          it.status === 'importing' || it.status === 'processing' || it.status === 'exporting'
            ? 'queued'
            : it.status || 'queued',
        progressText:
          it.status === 'importing' || it.status === 'processing' || it.status === 'exporting'
            ? 'Queued'
            : it.progressText || 'Queued',
      }));
  } catch {
    return [];
  }
}

export function formatBatchExportFilename(pattern, stem, ext) {
  const cleanStem = String(stem || 'vellum').trim() || 'vellum';
  const rawPattern = String(pattern || '{name}').trim() || '{name}';
  let out = rawPattern.replace(/\{name\}/gi, cleanStem);
  out = out.replace(/\.(cbz|pdf)$/i, '');
  return `${out}.${ext}`;
}

export async function pickNativeSavePath(apiBaseUrl, { defaultName, defaultExt, title } = {}) {
  try {
    const resp = await fetch(`${apiBaseUrl}/api/system/pick-paths`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mode: 'save_file',
        default_name: defaultName || '',
        default_ext: defaultExt || '',
        title: title || 'Save Export File To...',
      }),
    });
    if (!resp.ok) {
      return { available: false };
    }
    const data = await resp.json();
    return {
      available: true,
      cancelled: Boolean(data?.cancelled),
      path: data?.path || null,
      dir: data?.dir || null,
      filename: data?.filename || null,
    };
  } catch {
    return { available: false };
  }
}

export async function saveBlobWithPicker(blob, filename, { description, accept } = {}) {
  if (typeof window !== 'undefined' && typeof window.showSaveFilePicker === 'function') {
    try {
      const opts = { suggestedName: filename };
      if (accept) {
        opts.types = [{ description: description || 'File', accept }];
      }
      const handle = await window.showSaveFilePicker(opts);
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      return { saved: true, filename: handle.name || filename };
    } catch (err) {
      if (err?.name === 'AbortError') {
        return { saved: false, cancelled: true };
      }
      // Fall back to standard download if File System Access API is blocked
    }
  }
  triggerBlobDownload(blob, filename);
  return { saved: true, filename };
}

export function triggerBlobDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function extractDataTransferFiles(dataTransfer) {
  const items = Array.from(dataTransfer?.items || []);
  const hasEntries = items.some(
    (it) => typeof it.webkitGetAsEntry === 'function'
  );
  if (!hasEntries) {
    return Array.from(dataTransfer?.files || []);
  }

  const droppedFiles = [];
  const walkEntry = (entry, prefix = '') =>
    new Promise((resolve) => {
      if (!entry) {
        resolve();
        return;
      }
      if (entry.isFile) {
        entry.file(
          (file) => {
            if (prefix && !file.webkitRelativePath) {
              try {
                Object.defineProperty(file, 'webkitRelativePath', {
                  value: `${prefix}${file.name}`,
                  configurable: true,
                });
              } catch {
                // ignore non-configurable property
              }
            }
            droppedFiles.push(file);
            resolve();
          },
          () => resolve()
        );
        return;
      }
      if (entry.isDirectory) {
        const reader = entry.createReader();
        const readBatch = () => {
          reader.readEntries(
            async (batch) => {
              if (!batch || batch.length === 0) {
                resolve();
                return;
              }
              for (const child of batch) {
                await walkEntry(child, `${prefix}${entry.name}/`);
              }
              readBatch();
            },
            () => resolve()
          );
        };
        readBatch();
        return;
      }
      resolve();
    });

  for (const item of items) {
    if (item.kind !== 'file') continue;
    const entry = item.webkitGetAsEntry?.();
    if (entry) {
      await walkEntry(entry);
    } else {
      const f = item.getAsFile?.();
      if (f) droppedFiles.push(f);
    }
  }

  return droppedFiles.length > 0
    ? droppedFiles
    : Array.from(dataTransfer?.files || []);
}

export const AUTOSAVE_STORAGE_KEY = 'vellum.autosave.v1';

export async function fetchWithRetry(url, init = {}, { retries = 1, signal, isCancelled } = {}) {
  const maxAttempts = Math.max(1, retries);
  let lastErrDetail = '';
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    if (signal?.aborted || isCancelled?.()) {
      throw new DOMException('Aborted', 'AbortError');
    }
    try {
      const response = await fetch(url, { ...init, signal });
      if (response.ok) return response;
      lastErrDetail = `HTTP ${response.status}`;
      try {
        const errJson = await response.json();
        if (errJson?.detail) lastErrDetail = errJson.detail;
      } catch {
        // ignore JSON parse failure
      }
      const isTransient =
        response.status === 502 ||
        response.status === 503 ||
        response.status === 504 ||
        response.status === 429;
      if (!isTransient || attempt === maxAttempts - 1) {
        throw new Error(lastErrDetail);
      }
    } catch (err) {
      if (err?.name === 'AbortError' || signal?.aborted || isCancelled?.()) {
        throw err;
      }
      if (attempt === maxAttempts - 1) {
        throw err;
      }
    }
    if (signal?.aborted || isCancelled?.()) {
      throw new DOMException('Aborted', 'AbortError');
    }
    await new Promise((resolve, reject) => {
      const timer = setTimeout(resolve, 1200 * (attempt + 1));
      if (signal) {
        signal.addEventListener(
          'abort',
          () => {
            clearTimeout(timer);
            reject(new DOMException('Aborted', 'AbortError'));
          },
          { once: true }
        );
      }
    });
  }
  throw new Error(lastErrDetail || 'Request failed');
}

