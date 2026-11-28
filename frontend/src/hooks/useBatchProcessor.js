import { useCallback } from 'react';
import { applySeriesProfileToStore } from '../components/SeriesProfilePicker.jsx';
import useEditorStore, {
  mergeRedetectedBlocks,
  normalizeBlocks,
} from '../store/useEditorStore.js';
import {
  BATCH_QUEUE_STORAGE_KEY,
  resolveBackendUrl,
  resolvePageImageFile,
  formatBatchExportFilename,
  fetchWithRetry,
} from '../utils/pipelineApi.js';

/**
 * Encapsulates single-page processing, per-stage execution, Wave M5 Little's Law
 * EWMA+AIMD pipelined batch scheduling, autosave checkpointing, and multi-batch queue runs.
 */
export function useBatchProcessor({
  activePageId,
  activeQueueItemIdRef,
  activeQueueItemMetaRef,
  apiBaseUrl,
  batchAbortControllerRef,
  batchCancelRef,
  canvasRef,
  importBaseName,
  isQueueRunning,
  pages,
  pipelineConfig,
  projectCacheIdRef,
  queueCancelRef,
  queueExportFormat,
  queueFilenamePattern,
  queueItems,
  queueOutputDir,
  queuePurgeCache,
  queueRunMode,
  queueShareContinuity,
  queueSkipExisting,
  reimportedQueueItemIdsRef,
  replaceProjectState,
  setActiveProfileSlug,
  setImportBaseName,
  setIsBatchQueueOpen,
  setIsPipelineRunning,
  setIsQueueRunning,
  setIsStopping,
  setPageStatus,
  setPipelineProgress,
  setQueueItems,
  setQuickProviderPrompt,
  setStatusBanner,
  sourceLang,
  targetLang,
  updatePage,
  viewLayers,
  wakePipelineWaitersRef,
}) {
  // Process a full page in one call via POST /api/process-page
  const processSinglePage = useCallback(
    async (pageItem, overrideSourceLang, overrideTargetLang, batchOpts = null) => {
      const src = overrideSourceLang || pageItem.sourceLang || sourceLang;
      const tgt = overrideTargetLang || pageItem.targetLang || targetLang;
      const prevStatus = pageItem.status === 'processing' ? 'idle' : (pageItem.status || 'idle');
      const activeSignal = batchOpts?.signal || batchAbortControllerRef.current?.signal;

      setPageStatus(pageItem.id, 'processing', {
        error: null,
        sourceLang: src,
        targetLang: tgt,
      });

      try {
        const imageFile = await resolvePageImageFile(pageItem);
        if (!imageFile) {
          throw new Error(`No source image available for ${pageItem.name}`);
        }

        const cfg = useEditorStore.getState().pipelineConfig;
        const formData = new FormData();
        formData.append('image', imageFile, pageItem.name || 'page.png');
        formData.append('source_lang', src);
        formData.append('target_lang', tgt);
        formData.append('translation_tone', cfg.translationTone || 'standard');
        formData.append('honorifics_policy', cfg.honorificsPolicy || 'keep');
        formData.append('project_id', projectCacheIdRef.current);
        if (cfg.customInstructions && String(cfg.customInstructions).trim()) {
          formData.append('custom_instructions', String(cfg.customInstructions).trim());
        }
        if (cfg.glossary && typeof cfg.glossary === 'object' && Object.keys(cfg.glossary).length > 0) {
          formData.append('glossary', JSON.stringify(cfg.glossary));
        }
        if (batchOpts) {
          formData.append('keep_alive', batchOpts.keepAlive ? 'true' : 'false');
          formData.append('batch_is_last', batchOpts.isLastPage ? 'true' : 'false');
        }

        const response = await fetchWithRetry(
          `${apiBaseUrl}/api/process-page`,
          { method: 'POST', body: formData },
          {
            retries: batchOpts?.isBatch ? 3 : 1,
            signal: activeSignal,
            isCancelled: () => batchCancelRef.current,
          }
        );

        const data = await response.json();
        if (activeSignal?.aborted || batchCancelRef.current) {
          throw new DOMException('Aborted', 'AbortError');
        }
        const normalizedBlocks = normalizeBlocks(data.blocks || data.bubbles || [], {
          imgWidth: data.dimensions?.width || pageItem.dimensions?.width,
          imgHeight: data.dimensions?.height || pageItem.dimensions?.height,
          targetLang: data.target_lang || tgt,
          layoutBubbles: data.layout_bubbles || [],
          panels: data.panels || [],
        }).map((b) => ({
          ...b,
          initial_translated_text: b.initial_translated_text ?? b.translated_text ?? '',
        }));
        const rawCleaned =
          data.cleaned_image_url ||
          data.inpainted_image_url ||
          data.cleaned_image ||
          data.inpainted_image;
        const resolvedMaskUrl = resolveBackendUrl(
          data.mask_url || data.mask,
          apiBaseUrl
        );

        const resolvedCleaned = resolveBackendUrl(rawCleaned, apiBaseUrl);

        updatePage(
          pageItem.id,
          {
            status: 'done',
            activeStage: null,
            error: null,
            cleanedImage: resolvedCleaned,
            inpaintedUrl: resolvedCleaned,
            maskId: data.mask_id || pageItem.maskId || null,
            maskUrl: resolvedMaskUrl || pageItem.maskUrl || null,
            dimensions: data.dimensions || pageItem.dimensions || null,
            blocks: normalizedBlocks,
            sourceLang: data.source_lang || src,
            targetLang: data.target_lang || tgt,
            model: data.model || null,
            tilesInpainted: data.tiles_inpainted ?? 0,
            echoRetried: Boolean(data.echo_retried),
            refusalDetected: Boolean(data.refusal_detected),
            leakedBlockIds: Array.isArray(data.leaked_block_ids) ? data.leaked_block_ids : [],
            lowConfidenceOcrIds: Array.isArray(data.low_confidence_ocr_ids) ? data.low_confidence_ocr_ids : [],
            anomalyBlockIds: Array.isArray(data.anomaly_block_ids) ? data.anomaly_block_ids : [],
          },
          { recordHistory: !batchOpts?.isBatch }
        );
        if (data?.refusal_detected && !batchOpts?.isBatch) {
          setStatusBanner({
            type: 'error',
            message: 'LLM refusal detected on 1+ bubbles — check Issues badge and retry with a different prompt/model.',
          });
        }
        const urlsToPin = [
          pageItem.originalUrl,
          rawCleaned,
          data.mask_url || data.mask,
        ].filter((u) => typeof u === 'string' && u && !u.startsWith('blob:') && !u.startsWith('data:'));
        if (urlsToPin.length > 0) {
          fetch(`${apiBaseUrl}/api/projects_cache/pin`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              project_id: projectCacheIdRef.current,
              image_urls: urlsToPin,
            }),
          }).catch(() => {});
        }
        return { ok: true };
      } catch (err) {
        if (err?.name === 'AbortError' || batchOpts?.signal?.aborted || batchCancelRef.current) {
          setPageStatus(pageItem.id, prevStatus, { activeStage: null, error: null });
          return { ok: false, aborted: true };
        }
        const msg = err?.message || 'Failed to process comic page.';
        setPageStatus(pageItem.id, 'error', { activeStage: null, error: msg });
        if (!batchOpts?.isBatch) {
          setStatusBanner({
            type: 'error',
            message: `Error processing ${pageItem.name}: ${msg}`,
          });
        }
        return { ok: false, error: msg };
      }
    },
    [apiBaseUrl, sourceLang, targetLang, setPageStatus, updatePage]
  );

  // Granular single-stage runner (`detect` | `ocr` | `inpaint` | `translate`)
  const runSingleStageOnPage = useCallback(
    async (pageId, stage, restrictedLayerIds = null, batchOpts = null) => {
      const currentPage = useEditorStore
        .getState()
        .pages.find((p) => p.id === pageId);
      if (!currentPage) return;

      const cfg = useEditorStore.getState().pipelineConfig;
      const src = cfg.sourceLang || 'ja';
      const tgt = cfg.targetLang || 'id';
      const prevStatus = currentPage.status === 'processing' ? 'idle' : (currentPage.status || 'idle');
      const activeSignal = batchOpts?.signal || batchAbortControllerRef.current?.signal;
      const isCancelled = () => batchCancelRef.current;

      setPageStatus(pageId, 'processing', { activeStage: stage, error: null });

      try {
        if (activeSignal?.aborted || isCancelled()) {
          throw new DOMException('Aborted', 'AbortError');
        }
        if (stage === 'detect') {
          let imageFile = null;
          try {
            imageFile = await resolvePageImageFile(currentPage);
          } catch (fetchErr) {
            const qMeta = activeQueueItemMetaRef.current;
            const is404 = String(fetchErr?.message || '').includes('404');
            if (
              is404 &&
              qMeta?.path &&
              !reimportedQueueItemIdsRef.current.has(qMeta.id)
            ) {
              reimportedQueueItemIdsRef.current.add(qMeta.id);
              const impUrl =
                qMeta.kind === 'folder'
                  ? `${apiBaseUrl}/api/import/folder-path`
                  : `${apiBaseUrl}/api/import/archive-path`;
              const impResp = await fetch(impUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: qMeta.path }),
              });
              if (impResp.ok) {
                const impData = await impResp.json();
                const byName = new Map(
                  (impData?.pages || []).map((ep) => [
                    String(ep.filename || '').toLowerCase(),
                    resolveBackendUrl(ep.image_url, apiBaseUrl),
                  ])
                );
                const curAll = useEditorStore.getState().pages;
                const remapped = curAll.map((pg) => {
                  const freshUrl = byName.get(String(pg.name || '').toLowerCase());
                  return freshUrl ? { ...pg, originalUrl: freshUrl } : pg;
                });
                const pinnedRemapped = await pinPagesToProjectCache(
                  qMeta.projectId || projectCacheIdRef.current,
                  remapped
                );
                for (const pg of pinnedRemapped) {
                  if (pg.originalUrl) {
                    updatePage(pg.id, { originalUrl: pg.originalUrl }, { recordHistory: false });
                  }
                }
                const refreshed = pinnedRemapped.find((pg) => pg.id === pageId) || currentPage;
                imageFile = await resolvePageImageFile(refreshed);
              }
            }
            if (!imageFile) throw fetchErr;
          }
          if (!imageFile) throw new Error('Source image missing for detection.');

          const formData = new FormData();
          formData.append('image', imageFile, currentPage.name || 'page.png');
          if (batchOpts) {
            formData.append('keep_alive', batchOpts.keepAlive ? 'true' : 'false');
            formData.append('batch_is_last', batchOpts.isLastPage ? 'true' : 'false');
          }
          formData.append('source_lang', src);
          formData.append('detector_precision', cfg.detectorPrecision || 'balanced');
          formData.append('skip_blank_pages', cfg.skipBlankPages !== false ? 'true' : 'false');

          const resp = await fetchWithRetry(
            `${apiBaseUrl}/api/stage/detect`,
            { method: 'POST', body: formData },
            { retries: batchOpts?.isBatch ? 3 : 1, signal: activeSignal, isCancelled }
          );
          const data = await resp.json();
          if (activeSignal?.aborted || isCancelled()) {
            throw new DOMException('Aborted', 'AbortError');
          }
          const imgW = data.dimensions?.width || currentPage.dimensions?.width || data.width;
          const imgH = data.dimensions?.height || currentPage.dimensions?.height || data.height;
          const rawDetectedBlocks = data.skipped
            ? []
            : normalizeBlocks(data.blocks || data.bubbles || [], {
                imgWidth: imgW,
                imgHeight: imgH,
                targetLang: tgt,
                layoutBubbles: data.layout_bubbles || [],
                panels: data.panels || [],
              });
          const detectedBlocks =
            !data.skipped && (currentPage.blocks || []).length > 0
              ? mergeRedetectedBlocks(currentPage.blocks, rawDetectedBlocks, imgW, imgH)
              : rawDetectedBlocks;
          updatePage(
            pageId,
            {
              status: 'done',
              activeStage: null,
              error: null,
              dimensions: data.dimensions || currentPage.dimensions || null,
              maskId: data.mask_id || null,
              maskUrl: resolveBackendUrl(data.mask_url || data.mask, apiBaseUrl),
              blocks: detectedBlocks,
              sourceLang: data.source_lang || src,
            },
            { recordHistory: !batchOpts?.isBatch }
          );
          return;
        }

        const rawTargetBlocks =
          restrictedLayerIds && restrictedLayerIds.length > 0
            ? (currentPage.blocks || []).filter((b) =>
                restrictedLayerIds.includes(b.id)
              )
            : currentPage.blocks || [];
        const targetBlocks = rawTargetBlocks.filter((b) => !b.ignored);

        if (stage === 'ocr') {
          if (targetBlocks.length === 0) {
            throw new Error('No bounding boxes available for OCR. Run Detection first.');
          }
          const imageFile = await resolvePageImageFile(currentPage);
          if (!imageFile) throw new Error('Source image missing for OCR.');

          const liveOcrBlocks = targetBlocks.map((b) => {
            const hasLiveGeom =
              typeof b.x === 'number' &&
              typeof b.y === 'number' &&
              typeof b.width === 'number' &&
              typeof b.height === 'number';
            const liveBbox = hasLiveGeom
              ? [
                  Math.max(0, Math.round(b.x)),
                  Math.max(0, Math.round(b.y)),
                  Math.max(1, Math.round(b.x + b.width)),
                  Math.max(1, Math.round(b.y + b.height)),
                ]
              : b.bbox;
            return { ...b, bbox: liveBbox };
          });

          const formData = new FormData();
          formData.append('image', imageFile, currentPage.name || 'page.png');
          if (batchOpts) {
            formData.append('keep_alive', batchOpts.keepAlive ? 'true' : 'false');
            formData.append('batch_is_last', batchOpts.isLastPage ? 'true' : 'false');
          }
          formData.append('source_lang', src);
          formData.append('detector_precision', cfg.detectorPrecision || 'balanced');
          formData.append('bboxes', JSON.stringify(liveOcrBlocks));

          const resp = await fetchWithRetry(
            `${apiBaseUrl}/api/stage/ocr`,
            { method: 'POST', body: formData },
            { retries: batchOpts?.isBatch ? 3 : 1, signal: activeSignal, isCancelled }
          );
          const data = await resp.json();
          if (activeSignal?.aborted || isCancelled()) {
            throw new DOMException('Aborted', 'AbortError');
          }
          const ocrById = Object.fromEntries(
            (data.blocks || data.bubbles || []).map((b) => [b.id, b])
          );
          updatePage(
            pageId,
            (existing) => ({
              status: 'done',
              activeStage: null,
              error: null,
              sourceLang: src,
              lowConfidenceOcrIds: Array.isArray(data.low_confidence_ocr_ids) ? data.low_confidence_ocr_ids : [],
              blocks: (existing.blocks || []).map((b) => {
                const hit = ocrById[b.id];
                if (!hit) return b;
                const text = hit.original_text ?? hit.source_text ?? b.original_text;
                return {
                  ...b,
                  original_text: text,
                  source_text: text,
                  ocr_entropy: hit.ocr_entropy,
                  low_ocr_confidence: Boolean(hit.low_ocr_confidence),
                };
              }),
            }),
            { recordHistory: !batchOpts?.isBatch }
          );
          return;
        }

        if (stage === 'inpaint') {
          const imageFile = await resolvePageImageFile(currentPage);
          if (!imageFile) throw new Error('Source image missing for inpainting.');

          const formData = new FormData();
          formData.append('image', imageFile, currentPage.name || 'page.png');
          if (batchOpts) {
            formData.append('keep_alive', batchOpts.keepAlive ? 'true' : 'false');
            formData.append('batch_is_last', batchOpts.isLastPage ? 'true' : 'false');
          }
          if (currentPage.maskId) {
            formData.append('mask_id', currentPage.maskId);
          } else if (currentPage.maskUrl) {
            formData.append('mask_url', currentPage.maskUrl);
          }
          if (targetBlocks.length > 0) {
            formData.append(
              'bboxes',
              JSON.stringify(targetBlocks.map((b) => b.bbox))
            );
            formData.append(
              'blocks',
              JSON.stringify(
                targetBlocks.map((b) => ({
                  id: b.id,
                  bbox: b.bbox,
                  safe_bbox: b.safe_bbox || undefined,
                  class_name: b.class_name || b.bubble_class || undefined,
                }))
              )
            );
          } else if (!currentPage.maskId && !currentPage.maskUrl) {
            throw new Error('No mask or bounding boxes available for inpainting.');
          }

          const resp = await fetchWithRetry(
            `${apiBaseUrl}/api/stage/inpaint`,
            { method: 'POST', body: formData },
            { retries: batchOpts?.isBatch ? 3 : 1, signal: activeSignal, isCancelled }
          );
          const data = await resp.json();
          if (activeSignal?.aborted || isCancelled()) {
            throw new DOMException('Aborted', 'AbortError');
          }
          const cleanedUrl =
            data.cleaned_image_url ||
            data.inpainted_image_url ||
            data.cleaned_image ||
            data.inpainted_image;
          const resolvedCleaned = resolveBackendUrl(cleanedUrl, apiBaseUrl);
          updatePage(
            pageId,
            {
              status: 'done',
              activeStage: null,
              error: null,
              cleanedImage: resolvedCleaned,
              inpaintedUrl: resolvedCleaned,
              dimensions: data.dimensions || currentPage.dimensions || null,
              tilesInpainted: data.tiles_inpainted ?? currentPage.tilesInpainted,
            },
            { recordHistory: !batchOpts?.isBatch }
          );
          if (cleanedUrl && !String(cleanedUrl).startsWith('data:')) {
            fetch(`${apiBaseUrl}/api/projects_cache/pin`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                project_id: projectCacheIdRef.current,
                image_urls: [cleanedUrl],
              }),
            }).catch(() => {});
          }
          return;
        }

        if (stage === 'translate') {
          if (targetBlocks.length === 0) {
            throw new Error('No text blocks available to translate.');
          }
          const payload = {
            project_id: projectCacheIdRef.current,
            source_lang: src,
            target_lang: tgt,
            translation_tone: cfg.translationTone || 'standard',
            honorifics_policy: cfg.honorificsPolicy || 'keep',
            blocks: targetBlocks.map((b) => ({
              id: b.id,
              order: b.order,
              bbox: b.bbox,
              bubble_class: b.class_name || b.bubble_class || undefined,
              panel_id: typeof b.panel_id === 'number' ? b.panel_id : undefined,
              original_text: b.original_text || b.source_text || '',
              source_text: b.source_text || b.original_text || '',
              translated_text: b.translated_text || '',
            })),
            config: {
              provider: cfg.provider || undefined,
              model: cfg.model || undefined,
              temperature: cfg.temperature,
              top_p: cfg.topP ?? undefined,
              top_k: cfg.topK ?? undefined,
              max_tokens: cfg.maxTokens,
              frequency_penalty: cfg.frequencyPenalty ?? undefined,
              reasoning_effort: cfg.reasoningEffort || undefined,
              translation_tone: cfg.translationTone || 'standard',
              honorifics_policy: cfg.honorificsPolicy || 'keep',
              custom_instructions: cfg.customInstructions || undefined,
              glossary: cfg.glossary,
              use_vision: cfg.useVision,
              vision_max_side: cfg.visionMaxSide || undefined,
            },
          };

          const resp = await fetchWithRetry(
            `${apiBaseUrl}/api/stage/translate`,
            {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                Accept: 'application/json',
              },
              body: JSON.stringify(payload),
            },
            { retries: batchOpts?.isBatch ? 3 : 1, signal: activeSignal, isCancelled }
          );
          const data = await resp.json();
          if (activeSignal?.aborted || isCancelled()) {
            throw new DOMException('Aborted', 'AbortError');
          }
          const transMap = data.translations || {};
          const byId = Object.fromEntries(
            (data.blocks || data.bubbles || []).map((b) => [b.id, b])
          );
          updatePage(
            pageId,
            (existing) => ({
              status: 'done',
              activeStage: null,
              error: null,
              targetLang: data.target_lang || tgt,
              model: data.model || existing.model,
              echoRetried: Boolean(data.echo_retried),
              refusalDetected: Boolean(data.refusal_detected),
              leakedBlockIds: Array.isArray(data.leaked_block_ids) ? data.leaked_block_ids : [],
              anomalyBlockIds: Array.isArray(data.anomaly_block_ids) ? data.anomaly_block_ids : [],
              blocks: (existing.blocks || []).map((b) => {
                if (b.ignored) return b;
                const updated =
                  transMap[b.id] ?? byId[b.id]?.translated_text;
                if (updated === undefined) return b;
                return {
                  ...b,
                  translated_text: updated,
                  initial_translated_text: b.initial_translated_text ?? updated,
                };
              }),
            }),
            { recordHistory: !batchOpts?.isBatch }
          );
          if (data?.refusal_detected && !batchOpts?.isBatch) {
            setStatusBanner({
              type: 'error',
              message: 'LLM refusal detected on 1+ bubbles — check Issues badge and retry with a different prompt/model.',
            });
          }
        }
      } catch (err) {
        if (err?.name === 'AbortError' || batchOpts?.signal?.aborted || batchCancelRef.current) {
          setPageStatus(pageId, prevStatus, { activeStage: null, error: null });
          throw err;
        }
        const msg = err?.message || `Stage ${stage} failed.`;
        setPageStatus(pageId, 'error', { activeStage: null, error: msg });
        setStatusBanner({
          type: 'error',
          message: `${stage.toUpperCase()} failed on ${currentPage.name}: ${msg}`,
        });
        throw err;
      }
    },
    [apiBaseUrl, setPageStatus, updatePage]
  );

  // Execute a sequence of stages across one or more target pages
  const runStagesForPages = useCallback(
    async (targetPageIds, stagesToRun, restrictedLayerIds = null) => {
      const storePages = useEditorStore.getState().pages || [];
      targetPageIds = targetPageIds.filter(
        (pid) => !storePages.find((p) => p.id === pid)?.locked
      );
      if (!targetPageIds.length || !stagesToRun.length) return;

      const activePid = useEditorStore.getState().pipelineConfig?.provider || 'none';
      if (stagesToRun.includes('translate') && activePid === 'none') {
        const nonTranslateStages = stagesToRun.filter((s) => s !== 'translate');
        setQuickProviderPrompt({
          onProceedTranslate: () => runStagesForPages(targetPageIds, stagesToRun, restrictedLayerIds),
          onProceedManual:
            nonTranslateStages.length > 0
              ? () => runStagesForPages(targetPageIds, nonTranslateStages, restrictedLayerIds)
              : null,
        });
        return;
      }

      const controller = new AbortController();
      batchAbortControllerRef.current = controller;
      batchCancelRef.current = false;
      setIsStopping(false);
      const { signal } = controller;

      setIsPipelineRunning(true);
      setStatusBanner(null);

      const allFourStages =
        stagesToRun.length === 4 &&
        ['detect', 'ocr', 'inpaint', 'translate'].every((s) =>
          stagesToRun.includes(s)
        );

      const startedAt = Date.now();
      const totalPages = targetPageIds.length;
      const totalStages = allFourStages && !restrictedLayerIds ? 1 : stagesToRun.length;

      setPipelineProgress({
        currentPageIndex: 1,
        totalPages,
        stage: allFourStages && !restrictedLayerIds ? 'pipeline' : stagesToRun[0],
        stageIndex: 1,
        totalStages,
        startedAt,
      });

      fetch(`${apiBaseUrl}/api/system/keep-awake`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: true }),
      }).catch(() => {});
      fetch(`${apiBaseUrl}/api/system/taskbar-progress`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state: 'normal', value: 0, max: Math.max(1, totalPages) }),
      }).catch(() => {});

      const isMultiPageBatch = totalPages > 1;
      let completedFinalPage = false;
      let succeededCount = 0;
      let failedCount = 0;
      let aborted = false;

      const persistAutosaveCheckpoint = () => {
        try {
          const state = useEditorStore.getState();
          if (!state.pages?.length) return;
          const { apiKey: _redactedKey, providerProfiles: rawProfiles, ...restPipelineConfig } = state.pipelineConfig;
          const safeProviderProfiles = rawProfiles && typeof rawProfiles === 'object'
            ? Object.fromEntries(
                Object.entries(rawProfiles).map(([pid, prof]) => [
                  pid,
                  prof && typeof prof === 'object' ? { ...prof, api_key: '' } : prof,
                ])
              )
            : undefined;
          const savedAt = new Date().toISOString();
          const snapshot = {
            format: 'vellum-autosave',
            savedAt,
            importBaseName,
            activePageId: state.activePageId,
            viewLayers: state.viewLayers,
            pipelineConfig: {
              ...restPipelineConfig,
              ...(safeProviderProfiles ? { providerProfiles: safeProviderProfiles } : {}),
            },
            typesetting: state.typesetting,
            pages: state.pages.map((p) => ({
              id: p.id,
              name: p.name,
              originalUrl:
                p.originalUrl && !p.originalUrl.startsWith('blob:')
                  ? p.originalUrl
                  : null,
              cleanedImage: p.cleanedImage || p.inpaintedUrl || null,
              inpaintedUrl: p.inpaintedUrl || p.cleanedImage || null,
              maskUrl: p.maskUrl || null,
              maskId: p.maskId || null,
              dimensions: p.dimensions || null,
              blocks: (p.blocks || []).map((b) => {
                const { polygon: _poly, mask: _mask, ...lightBlock } = b || {};
                if (b?.mask && typeof b.mask === 'object') {
                  const { polygon: _mpoly, ...lightMask } = b.mask;
                  lightBlock.mask = lightMask;
                }
                return lightBlock;
              }),
              status:
                p.status === 'processing' || p.status === 'queued'
                  ? 'idle'
                  : p.status,
              sourceLang: p.sourceLang,
              targetLang: p.targetLang,
              model: p.model,
              tilesInpainted: p.tilesInpainted ?? 0,
            })),
          };
          fetch(`${apiBaseUrl}/api/projects_cache/autosave`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(snapshot),
          }).catch(() => {});
        } catch {
          // ignore checkpoint serialization errors
        }
      };

      const resolveBatchSchedulePlan = async () => {
        const configured = useEditorStore.getState().pipelineConfig?.batchScheduleMode || 'auto';
        if (configured === 'stage-first') {
          return { mode: 'stage-first', ahead: 3 };
        }
        const CLOUD_PROVIDERS = new Set([
          'openai',
          'gemini',
          'anthropic',
          'deepseek',
          'openrouter',
        ]);
        let mode = configured === 'pipelined' ? 'pipelined' : 'pipelined';
        let ahead = 3;
        try {
          const hwResp = await fetch(`${apiBaseUrl}/api/system/hardware`, { signal });
          if (hwResp.ok) {
            const hw = await hwResp.json();
            const freeVramMb = Number(hw?.free_vram_mb ?? 0);
            const ramLoadPct = Number(hw?.ram_load_pct ?? hw?.memory_load_percent ?? 0);
            if (configured === 'auto') {
              if (activePid !== 'none' && !CLOUD_PROVIDERS.has(activePid) && freeVramMb < 1024) {
                mode = 'stage-first';
              }
              if (ramLoadPct >= 85) {
                ahead = 1;
              }
            }
          }
        } catch {
          // Fall back to pipelined if hardware probe fails
        }
        return { mode, ahead };
      };

      const unloadStageBestEffort = async (stage) => {
        if (batchCancelRef.current || signal.aborted) return;
        try {
          await fetch(`${apiBaseUrl}/api/system/unload-stage`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ stage }),
            signal,
          });
        } catch {
          // Best-effort VRAM release
        }
      };

      try {
        if (isMultiPageBatch) {
          const { mode: effectiveMode, ahead: batchAhead } = await resolveBatchSchedulePlan();
          if (signal.aborted) {
            aborted = true;
          } else if (effectiveMode === 'stage-first') {
            const CHUNK_SIZE = 50;
            const orderedStages = ['detect', 'ocr', 'inpaint', 'translate'].filter((s) =>
              stagesToRun.includes(s)
            );
            for (const s of stagesToRun) {
              if (!orderedStages.includes(s)) orderedStages.push(s);
            }

            for (let chunkStart = 0; chunkStart < totalPages; chunkStart += CHUNK_SIZE) {
              if (batchCancelRef.current || signal.aborted) {
                aborted = true;
                break;
              }
              const chunkIds = targetPageIds.slice(chunkStart, chunkStart + CHUNK_SIZE);
              const chunkFailed = new Set();

              for (let sIdx = 0; sIdx < orderedStages.length; sIdx += 1) {
                if (batchCancelRef.current || signal.aborted) {
                  aborted = true;
                  break;
                }
                const stage = orderedStages[sIdx];
                for (let cIdx = 0; cIdx < chunkIds.length; cIdx += 1) {
                  if (batchCancelRef.current || signal.aborted) {
                    aborted = true;
                    break;
                  }
                  const pageId = chunkIds[cIdx];
                  if (chunkFailed.has(pageId)) continue;
                  setPipelineProgress({
                    currentPageIndex: chunkStart + cIdx + 1,
                    totalPages,
                    stage,
                    stageIndex: sIdx + 1,
                    totalStages: orderedStages.length,
                    startedAt,
                  });
                  try {
                    await runSingleStageOnPage(pageId, stage, restrictedLayerIds, {
                      keepAlive: true,
                      isLastPage: false,
                      isBatch: true,
                      signal,
                    });
                  } catch (err) {
                    if (err?.name === 'AbortError' || signal.aborted || batchCancelRef.current) {
                      aborted = true;
                      break;
                    }
                    chunkFailed.add(pageId);
                  }
                }
                if (aborted || batchCancelRef.current || signal.aborted) {
                  aborted = true;
                  break;
                }
                if (stage === 'detect' || stage === 'ocr' || stage === 'inpaint') {
                  await unloadStageBestEffort(stage);
                }
              }

              if (aborted) break;
              failedCount += chunkFailed.size;
              succeededCount += chunkIds.length - chunkFailed.size;
              if (chunkStart + chunkIds.length >= totalPages) {
                completedFinalPage = true;
              }
              persistAutosaveCheckpoint();
            }
          } else if (
            stagesToRun.includes('translate') &&
            stagesToRun.some((s) => s !== 'translate')
          ) {
            // #20 Little's Law + EWMA + AIMD Adaptive Pipeline Scheduler:
            // Tracks T_vision and T_llm via EWMA (alpha = 0.25), computes
            // L_target = clamp(ceil(T_llm / T_vision), 1, L_cap), and adapts L_cap in [1, 4] via AIMD.
            const EWMA_ALPHA = 0.25;
            let ewmaVisionMs = 450;
            let ewmaLlmMs = 900;
            let lCap = batchAhead <= 1 ? 1 : 3;
            let smoothPages = 0;
            let lastRamLoadPct = 0;
            let lastFreeVramMb = 4096;
            const getAdaptiveAhead = () => {
              const ratio = ewmaLlmMs / Math.max(50, ewmaVisionMs);
              return Math.max(1, Math.min(lCap, Math.ceil(ratio)));
            };
            const maybeProbeHardwareAimd = async (pageIdx) => {
              if (pageIdx % 4 !== 0) return;
              try {
                const hwResp = await fetch(`${apiBaseUrl}/api/system/hardware`, { signal });
                if (hwResp.ok) {
                  const hw = await hwResp.json();
                  lastFreeVramMb = Number(hw?.free_vram_mb ?? lastFreeVramMb);
                  lastRamLoadPct = Number(hw?.ram_load_pct ?? hw?.memory_load_percent ?? lastRamLoadPct);
                  if (lastRamLoadPct >= 85 || lastFreeVramMb < 900) {
                    lCap = Math.max(1, Math.floor(lCap / 2));
                    smoothPages = 0;
                  }
                }
              } catch {
                // Keep previous telemetry on transient probe failure
              }
            };

            const visionStages = ['detect', 'ocr', 'inpaint'].filter((s) =>
              stagesToRun.includes(s)
            );
            // Store only lightweight pageIdx/pageId/ok metadata in readyQueue (never blobs/base64).
            const readyQueue = [];
            let producerDone = false;
            const waiters = new Set();
            const notify = () => {
              if (waiters.size === 0) return;
              const pending = Array.from(waiters);
              waiters.clear();
              for (const fn of pending) fn();
            };
            wakePipelineWaitersRef.current = notify;
            const waitSignal = () =>
              new Promise((resolve) => {
                if (batchCancelRef.current || signal.aborted) {
                  resolve();
                  return;
                }
                waiters.add(resolve);
              });

            const producer = (async () => {
              try {
                for (let pageIdx = 0; pageIdx < totalPages; pageIdx += 1) {
                  while (readyQueue.length >= getAdaptiveAhead() && !signal.aborted && !batchCancelRef.current) {
                    await waitSignal();
                  }
                  if (batchCancelRef.current || signal.aborted) break;

                  await maybeProbeHardwareAimd(pageIdx);
                  const tVisStart = performance.now();
                  const pageId = targetPageIds[pageIdx];
                  let ok = true;
                  let zeroBubbles = false;
                  for (let sIdx = 0; sIdx < visionStages.length; sIdx += 1) {
                    if (batchCancelRef.current || signal.aborted) {
                      ok = false;
                      break;
                    }
                    const stage = visionStages[sIdx];
                    try {
                      await runSingleStageOnPage(pageId, stage, restrictedLayerIds, {
                        keepAlive: true,
                        isLastPage: false,
                        isBatch: true,
                        signal,
                      });
                      if (stage === 'detect') {
                        const detectedPage = useEditorStore
                          .getState()
                          .pages.find((p) => p.id === pageId);
                        if ((detectedPage?.blocks || []).length === 0) {
                          zeroBubbles = true;
                          break;
                        }
                      }
                    } catch (err) {
                      if (err?.name === 'AbortError' || signal.aborted || batchCancelRef.current) {
                        ok = false;
                        break;
                      }
                      ok = false;
                      break;
                    }
                  }
                  if (ok && !zeroBubbles) {
                    const dtVis = Math.max(10, performance.now() - tVisStart);
                    ewmaVisionMs = (1 - EWMA_ALPHA) * ewmaVisionMs + EWMA_ALPHA * dtVis;
                  }
                  if (batchCancelRef.current || signal.aborted) break;
                  readyQueue.push({ pageIdx, pageId, ok, zeroBubbles });
                  notify();
                }
              } finally {
                producerDone = true;
                notify();
              }
            })();

            const consumer = (async () => {
              while (!signal.aborted && !batchCancelRef.current) {
                while (
                  readyQueue.length === 0 &&
                  !producerDone &&
                  !signal.aborted &&
                  !batchCancelRef.current
                ) {
                  await waitSignal();
                }
                if (batchCancelRef.current || signal.aborted) break;
                if (readyQueue.length === 0 && producerDone) break;

                const item = readyQueue.shift();
                notify();
                if (!item) continue;

                const isLastPage = item.pageIdx === totalPages - 1;
                if (!item.ok) {
                  failedCount += 1;
                  smoothPages = 0;
                  if (isLastPage) completedFinalPage = true;
                  continue;
                }
                if (item.zeroBubbles) {
                  succeededCount += 1;
                  if (isLastPage) completedFinalPage = true;
                  continue;
                }

                if (batchCancelRef.current || signal.aborted) break;

                setPipelineProgress({
                  currentPageIndex: item.pageIdx + 1,
                  totalPages,
                  stage: 'translate',
                  stageIndex: stagesToRun.length,
                  totalStages: stagesToRun.length,
                  startedAt,
                });
                const tLlmStart = performance.now();
                try {
                  await runSingleStageOnPage(item.pageId, 'translate', restrictedLayerIds, {
                    keepAlive: !isLastPage,
                    isLastPage,
                    isBatch: true,
                    signal,
                  });
                  const dtLlm = Math.max(10, performance.now() - tLlmStart);
                  ewmaLlmMs = (1 - EWMA_ALPHA) * ewmaLlmMs + EWMA_ALPHA * dtLlm;
                  succeededCount += 1;
                  smoothPages += 1;
                  if (smoothPages >= 5 && lastRamLoadPct < 78 && lastFreeVramMb >= 1200) {
                    lCap = Math.min(4, lCap + 1);
                    smoothPages = 0;
                  }
                } catch (err) {
                  if (err?.name === 'AbortError' || signal.aborted || batchCancelRef.current) {
                    aborted = true;
                    break;
                  }
                  failedCount += 1;
                  smoothPages = 0;
                }
                if (isLastPage) {
                  completedFinalPage = true;
                }
              }
            })();

            await Promise.all([producer, consumer]);
            wakePipelineWaitersRef.current = null;
            if (signal.aborted || batchCancelRef.current) {
              aborted = true;
            }
          } else {
            // Sequential fallback for single-stage/vision-only pipelined batches
            for (let pageIdx = 0; pageIdx < targetPageIds.length; pageIdx += 1) {
              if (batchCancelRef.current || signal.aborted) {
                aborted = true;
                break;
              }
              const pageId = targetPageIds[pageIdx];
              const isLastPage = pageIdx === targetPageIds.length - 1;
              let pageFailed = false;
              for (let stageIdx = 0; stageIdx < stagesToRun.length; stageIdx += 1) {
                if (batchCancelRef.current || signal.aborted) {
                  aborted = true;
                  break;
                }
                const stage = stagesToRun[stageIdx];
                const isFinalCall = isLastPage && stageIdx === stagesToRun.length - 1;
                setPipelineProgress({
                  currentPageIndex: pageIdx + 1,
                  totalPages,
                  stage,
                  stageIndex: stageIdx + 1,
                  totalStages,
                  startedAt,
                });
                try {
                  await runSingleStageOnPage(pageId, stage, restrictedLayerIds, {
                    keepAlive: !isFinalCall,
                    isLastPage: isFinalCall,
                    isBatch: true,
                    signal,
                  });
                  if (stage === 'detect' && stagesToRun.length > 1) {
                    const detectedPage = useEditorStore
                      .getState()
                      .pages.find((p) => p.id === pageId);
                    if ((detectedPage?.blocks || []).length === 0) {
                      break;
                    }
                  }
                } catch (err) {
                  if (err?.name === 'AbortError' || signal.aborted || batchCancelRef.current) {
                    aborted = true;
                    break;
                  }
                  pageFailed = true;
                  break;
                }
              }
              if (aborted) break;
              if (pageFailed) failedCount += 1;
              else succeededCount += 1;
              if (isLastPage) completedFinalPage = true;
            }
          }
        } else {
          for (let pageIdx = 0; pageIdx < targetPageIds.length; pageIdx += 1) {
            if (signal.aborted) {
              aborted = true;
              break;
            }
            const pageId = targetPageIds[pageIdx];
            const liveState = useEditorStore.getState();
            const pageObj = liveState.pages.find((p) => p.id === pageId);
            if (!pageObj) continue;

            const isLastPage = pageIdx === targetPageIds.length - 1;
            const keepAlive = false;

            if (allFourStages && !restrictedLayerIds) {
              setPipelineProgress({
                currentPageIndex: pageIdx + 1,
                totalPages,
                stage: 'pipeline',
                stageIndex: 1,
                totalStages: 1,
                startedAt,
              });
              const pageRes = await processSinglePage(
                pageObj,
                liveState.pipelineConfig.sourceLang,
                liveState.pipelineConfig.targetLang,
                { keepAlive, isLastPage, isBatch: false, signal }
              );
              if (pageRes?.aborted || signal.aborted) {
                aborted = true;
                break;
              }
              if (pageRes?.ok) {
                succeededCount += 1;
              } else {
                failedCount += 1;
              }
            } else {
              let pageFailed = false;
              for (let stageIdx = 0; stageIdx < stagesToRun.length; stageIdx += 1) {
                if (signal.aborted) {
                  aborted = true;
                  break;
                }
                const stage = stagesToRun[stageIdx];
                setPipelineProgress({
                  currentPageIndex: pageIdx + 1,
                  totalPages,
                  stage,
                  stageIndex: stageIdx + 1,
                  totalStages,
                  startedAt,
                });
                try {
                  await runSingleStageOnPage(pageId, stage, restrictedLayerIds, {
                    keepAlive: false,
                    isLastPage: true,
                    isBatch: false,
                    signal,
                  });
                  if (stage === 'detect' && stagesToRun.length > 1) {
                    const detectedPage = useEditorStore
                      .getState()
                      .pages.find((p) => p.id === pageId);
                    if ((detectedPage?.blocks || []).length === 0) {
                      break;
                    }
                  }
                } catch (err) {
                  if (err?.name === 'AbortError' || signal.aborted) {
                    aborted = true;
                    break;
                  }
                  pageFailed = true;
                  throw new Error('Single-page stage failed');
                }
              }
              if (aborted) break;
              if (pageFailed) {
                failedCount += 1;
              } else {
                succeededCount += 1;
              }
            }
            if (isLastPage) {
              completedFinalPage = true;
            }
          }
        }
        if (aborted || signal.aborted) {
          setStatusBanner({
            type: 'warning',
            message: `Batch stopped (${succeededCount}/${totalPages} completed - progress preserved).`,
          });
        } else if (failedCount > 0) {
          setStatusBanner({
            type: 'warning',
            message: `Completed ${succeededCount}/${totalPages} pages (${failedCount} failed - inspect error badges to retry).`,
          });
        } else {
          const singleZeroBubble =
            totalPages === 1 &&
            stagesToRun.includes('detect') &&
            (useEditorStore
              .getState()
              .pages.find((p) => p.id === targetPageIds[0])?.blocks?.length || 0) === 0;
          setStatusBanner(
            singleZeroBubble
              ? {
                  type: 'info',
                  message: 'Done (0 bubbles — draw boxes manually if needed).',
                }
              : {
                  type: 'success',
                  message: `Completed ${stagesToRun.join(' → ')} on ${totalPages} page${
                    totalPages === 1 ? '' : 's'
                  }.`,
                }
          );
          if (data?.refusal_detected && !batchOpts?.isBatch) {
            setStatusBanner({
              type: 'error',
              message: 'LLM refusal detected on 1+ bubbles — check Issues badge and retry with a different prompt/model.',
            });
          }
        }
      } catch {
        // Error banner already reported by stage handler
      } finally {
        wakePipelineWaitersRef.current = null;
        batchCancelRef.current = false;
        if (batchAbortControllerRef.current === controller) {
          batchAbortControllerRef.current = null;
        }
        if (
          isMultiPageBatch &&
          (!completedFinalPage || failedCount > 0 || aborted) &&
          useEditorStore.getState().pipelineConfig.autoUnloadVram
        ) {
          fetch(`${apiBaseUrl}/api/pipeline/unload`, { method: 'POST' }).catch(() => {});
        }
        setIsStopping(false);
        setIsPipelineRunning(false);
        setPipelineProgress(null);
        if (!activeQueueItemIdRef.current) {
          fetch(`${apiBaseUrl}/api/system/keep-awake`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: false }),
          }).catch(() => {});
          fetch(`${apiBaseUrl}/api/system/taskbar-progress`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ state: 'none', value: 0, max: 100 }),
          }).catch(() => {});
        }
      }
      return { aborted: aborted || signal.aborted, failedCount, succeededCount };
    },
    [apiBaseUrl, processSinglePage, runSingleStageOnPage]
  );

  const handleCancelBatch = useCallback(() => {
    queueCancelRef.current = true;
    batchCancelRef.current = true;
    setIsStopping(true);
    batchAbortControllerRef.current?.abort();
    wakePipelineWaitersRef.current?.();
  }, []);

  const handleStopBatchQueue = useCallback(() => {
    queueCancelRef.current = true;
    handleCancelBatch();
  }, [handleCancelBatch]);

  const buildAutosaveSnapshot = useCallback(
    (overrideProjectId = null, overrideTitle = null, overridePages = null) => {
      const state = useEditorStore.getState();
      const targetPages = Array.isArray(overridePages) ? overridePages : state.pages;
      const { apiKey: _redactedKey, providerProfiles: rawProfiles, ...restPipelineConfig } =
        state.pipelineConfig;
      const safeProviderProfiles =
        rawProfiles && typeof rawProfiles === 'object'
          ? Object.fromEntries(
              Object.entries(rawProfiles).map(([pid, prof]) => [
                pid,
                prof && typeof prof === 'object' ? { ...prof, api_key: '' } : prof,
              ])
            )
          : undefined;
      const safePipelineConfig = {
        ...restPipelineConfig,
        ...(safeProviderProfiles ? { providerProfiles: safeProviderProfiles } : {}),
      };
      return {
        format: 'vellum-autosave',
        project_id: overrideProjectId || projectCacheIdRef.current || '',
        savedAt: new Date().toISOString(),
        importBaseName: overrideTitle ?? importBaseName,
        activePageId: state.activePageId,
        viewLayers: state.viewLayers,
        pipelineConfig: safePipelineConfig,
        typesetting: state.typesetting,
        pages: targetPages.map((p) => ({
          id: p.id,
          name: p.name,
          originalUrl:
            p.originalUrl && !p.originalUrl.startsWith('blob:')
              ? p.originalUrl
              : null,
          cleanedImage: p.cleanedImage || p.inpaintedUrl || null,
          inpaintedUrl: p.inpaintedUrl || p.cleanedImage || null,
          maskUrl: p.maskUrl || null,
          maskId: p.maskId || null,
          dimensions: p.dimensions || null,
          blocks: (p.blocks || []).map((b) => {
            const { polygon: _poly, mask: _mask, ...lightBlock } = b || {};
            if (b?.mask && typeof b.mask === 'object') {
              const { polygon: _mpoly, ...lightMask } = b.mask;
              lightBlock.mask = lightMask;
            }
            return lightBlock;
          }),
          status:
            p.status === 'processing' || p.status === 'queued'
              ? 'idle'
              : p.status,
          sourceLang: p.sourceLang,
          targetLang: p.targetLang,
          model: p.model,
          tilesInpainted: p.tilesInpainted ?? 0,
        })),
      };
    },
    [importBaseName]
  );

  const pinPagesToProjectCache = useCallback(
    async (projectId, pageList) => {
      if (!projectId || !Array.isArray(pageList) || pageList.length === 0) return pageList;
      const urls = new Set();
      for (const p of pageList) {
        for (const u of [p.originalUrl, p.cleanedImage, p.inpaintedUrl, p.maskUrl]) {
          if (u && typeof u === 'string' && !u.startsWith('data:') && !u.startsWith('blob:')) {
            urls.add(u);
          }
        }
      }
      if (urls.size === 0) return pageList;
      try {
        const resp = await fetch(`${apiBaseUrl}/api/projects_cache/pin`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            project_id: projectId,
            image_urls: Array.from(urls),
          }),
        });
        if (!resp.ok) return pageList;
        const data = await resp.json();
        const pinnedMap = data?.pinned_urls || {};
        const remap = (u) => {
          if (!u) return u;
          const mapped = pinnedMap[u];
          return mapped ? resolveBackendUrl(mapped, apiBaseUrl) : u;
        };
        return pageList.map((p) => ({
          ...p,
          originalUrl: remap(p.originalUrl),
          cleanedImage: remap(p.cleanedImage),
          inpaintedUrl: remap(p.inpaintedUrl),
          maskUrl: remap(p.maskUrl),
        }));
      } catch {
        return pageList;
      }
    },
    [apiBaseUrl]
  );

  const flushCurrentProjectAutosave = useCallback(async () => {
    const curPages = useEditorStore.getState().pages;
    const pid = projectCacheIdRef.current;
    if (!pid || curPages.length === 0) return false;
    try {
      const snapshot = buildAutosaveSnapshot(pid, importBaseName, curPages);
      const resp = await fetch(`${apiBaseUrl}/api/projects_cache/autosave`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(snapshot),
      });
      return resp.ok;
    } catch {
      return false;
    }
  }, [apiBaseUrl, buildAutosaveSnapshot, importBaseName]);

  const handleOpenQueueItemInEditor = useCallback(
    async (item, profilesList = []) => {
      if (!item || isQueueRunning) return;
      try {
        await flushCurrentProjectAutosave();

        const itemPid =
          item.projectId || `qproj_${String(item.id).replace(/[^a-zA-Z0-9_-]/g, '')}`;

        if (item.profileSlug) {
          const matchedProf = profilesList.find((p) => p.slug === item.profileSlug);
          if (matchedProf) {
            setActiveProfileSlug(matchedProf.slug);
            await applySeriesProfileToStore(matchedProf, apiBaseUrl, { syncBackend: true });
          }
        }

        // Try loading isolated per-project autosave first (with 404 auto-reimport guard)
        try {
          const resp = await fetch(
            `${apiBaseUrl}/api/projects_cache/autosave?project_id=${encodeURIComponent(itemPid)}`
          );
          if (resp.ok) {
            const parsed = await resp.json();
            if (parsed && Array.isArray(parsed.pages) && parsed.pages.length > 0) {
              const firstUrl = parsed.pages[0]?.originalUrl;
              let firstOk = true;
              if (firstUrl && !String(firstUrl).startsWith('data:')) {
                try {
                  const probe = await fetch(firstUrl, { method: 'HEAD' });
                  if (probe.status === 404) firstOk = false;
                } catch {
                  // ignore network probe error
                }
              }
              if (firstOk) {
                projectCacheIdRef.current = itemPid;
                replaceProjectState({
                  pages: parsed.pages,
                  activePageId: parsed.activePageId || parsed.pages[0]?.id || null,
                  viewLayers: parsed.viewLayers || null,
                  typesetting: parsed.typesetting || null,
                });
                setImportBaseName(item.stem || item.name);
                setIsBatchQueueOpen(false);
                setStatusBanner({
                  type: 'success',
                  message: `Opened "${item.name}" (${parsed.pages.length} pages) in editor.`,
                });
                return;
              }
            }
          }
        } catch {
          // Fall through to fresh import below
        }

        // Not imported yet -> import, pin, save initial autosave, and open in editor
        const impUrl =
          item.kind === 'folder'
            ? `${apiBaseUrl}/api/import/folder-path`
            : `${apiBaseUrl}/api/import/archive-path`;
        const impResp = await fetch(impUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: item.path }),
        });
        if (!impResp.ok) {
          const errJ = await impResp.json().catch(() => ({}));
          throw new Error(errJ?.detail || `Import HTTP ${impResp.status}`);
        }
        const impData = await impResp.json();
        const extracted = impData.pages || [];
        if (extracted.length === 0) {
          throw new Error('No pages found in source');
        }

        const ts = Date.now();
        const curCfg = useEditorStore.getState().pipelineConfig;
        const rawImported = extracted.map((p, idx) => ({
          id: `page_q_${ts}_${idx}_${Math.random().toString(36).slice(2, 6)}`,
          name: p.filename || `page_${String(idx + 1).padStart(3, '0')}.png`,
          file: null,
          originalUrl: resolveBackendUrl(p.image_url, apiBaseUrl),
          cleanedImage: null,
          inpaintedUrl: null,
          maskUrl: null,
          maskId: null,
          dimensions: p.dimensions || null,
          blocks: [],
          status: 'idle',
          error: null,
          sourceLang: curCfg.sourceLang || 'ja',
          targetLang: curCfg.targetLang || 'id',
          model: null,
          tilesInpainted: 0,
        }));
        const pinnedPages = await pinPagesToProjectCache(itemPid, rawImported);

        projectCacheIdRef.current = itemPid;
        replaceProjectState({
          pages: pinnedPages,
          activePageId: pinnedPages[0]?.id || null,
        });
        setImportBaseName(item.stem || item.name);

        const initialSnap = buildAutosaveSnapshot(itemPid, item.stem || item.name, pinnedPages);
        await fetch(`${apiBaseUrl}/api/projects_cache/autosave`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(initialSnap),
        }).catch(() => {});

        setQueueItems((prev) =>
          prev.map((q) =>
            q.id === item.id
              ? {
                  ...q,
                  projectId: itemPid,
                  pageCount: pinnedPages.length,
                  status: 'ready',
                  progressText: `Ready to Edit (${pinnedPages.length}p)`,
                  error: null,
                }
              : q
          )
        );
        setIsBatchQueueOpen(false);
        setStatusBanner({
          type: 'success',
          message: `Imported and opened "${item.name}" (${pinnedPages.length} pages) for editing.`,
        });
      } catch (err) {
        setStatusBanner({
          type: 'error',
          message: `Failed to open "${item.name}": ${err?.message || 'Unknown error'}`,
        });
      }
    },
    [
      isQueueRunning,
      flushCurrentProjectAutosave,
      apiBaseUrl,
      replaceProjectState,
      pinPagesToProjectCache,
      buildAutosaveSnapshot,
    ]
  );

  const handleBatchExportQueueItems = useCallback(
    async (selectedItems, format = 'cbz', outputDir = '') => {
      const targetDir = String(outputDir || '').trim();
      if (!targetDir || !Array.isArray(selectedItems) || selectedItems.length === 0 || isQueueRunning) {
        return;
      }
      if (!canvasRef.current?.exportArchive) {
        setStatusBanner({ type: 'error', message: 'Canvas archive exporter is not ready.' });
        return;
      }

      await flushCurrentProjectAutosave();
      queueCancelRef.current = false;
      setIsQueueRunning(true);

      const formats = format === 'both' ? ['cbz', 'pdf'] : [format === 'pdf' ? 'pdf' : 'cbz'];
      const updateQueueRow = (id, patch) => {
        setQueueItems((prev) => prev.map((q) => (q.id === id ? { ...q, ...patch } : q)));
      };

      let exportedCount = 0;
      try {
        for (let i = 0; i < selectedItems.length; i += 1) {
          if (queueCancelRef.current) break;
          const item = selectedItems[i];
          const itemPid =
            item.projectId || `qproj_${String(item.id).replace(/[^a-zA-Z0-9_-]/g, '')}`;
          const cleanStem = item.stem || item.name || `manga_${i + 1}`;

          try {
            updateQueueRow(item.id, {
              status: 'exporting',
              progressText: `Exporting (${formats.join('+').toUpperCase()})…`,
              error: null,
            });

            let itemPages = null;
            if (
              projectCacheIdRef.current === itemPid &&
              useEditorStore.getState().pages.length > 0
            ) {
              itemPages = useEditorStore.getState().pages;
            } else {
              const resp = await fetch(
                `${apiBaseUrl}/api/projects_cache/autosave?project_id=${encodeURIComponent(itemPid)}`
              );
              if (!resp.ok) {
                throw new Error(`Autosave load HTTP ${resp.status}`);
              }
              const parsed = await resp.json();
              itemPages = Array.isArray(parsed?.pages) ? parsed.pages : [];
            }

            if (!itemPages || itemPages.length === 0) {
              throw new Error('No saved pages found — process or open this title first.');
            }

            for (const ext of formats) {
              if (queueCancelRef.current) break;
              const outFilename = formatBatchExportFilename(queueFilenamePattern, cleanStem, ext);
              await canvasRef.current.exportArchive(itemPages, ext, {
                filename: cleanStem,
                imageFormat: 'png',
                quality: 0.98,
                outputDir: targetDir,
                outputFilename: outFilename,
                overwrite: true,
                silent: true,
              });
            }

            if (queueCancelRef.current) {
              updateQueueRow(item.id, { status: 'ready', progressText: 'Stopped' });
              break;
            }

            exportedCount += 1;
            updateQueueRow(item.id, {
              status: 'done',
              exportStatus: 'Exported',
              progressText: `Exported (${itemPages.length}p)`,
            });

            // Yield between titles so offscreen canvases release cleanly
            await new Promise((r) => setTimeout(r, 30));
          } catch (err) {
            updateQueueRow(item.id, {
              status: 'error',
              progressText: 'Export Error',
              error: err?.message || 'Batch export failed',
            });
          }
        }
      } finally {
        setIsQueueRunning(false);
        setStatusBanner({
          type: queueCancelRef.current ? 'warning' : 'success',
          message: queueCancelRef.current
            ? `Batch export stopped (${exportedCount}/${selectedItems.length} exported).`
            : `Batch exported ${exportedCount}/${selectedItems.length} manga (${formats
                .join(' + ')
                .toUpperCase()}) to ${targetDir}.`,
        });
      }
    },
    [apiBaseUrl, flushCurrentProjectAutosave, isQueueRunning, queueFilenamePattern]
  );

  const handleClearQueueAndPurgeCaches = useCallback(async () => {
    if (isQueueRunning) return;
    if (
      typeof window !== 'undefined' &&
      !window.confirm(
        'Clear all items from the Batch Queue and delete their saved project caches from disk?'
      )
    ) {
      return;
    }
    const itemsToPurge = [...queueItems];
    for (const it of itemsToPurge) {
      if (!it.projectId) continue;
      await fetch(
        `${apiBaseUrl}/api/projects_cache/autosave?project_id=${encodeURIComponent(
          it.projectId
        )}&purge_dir=true`,
        { method: 'DELETE' }
      ).catch(() => {});
    }
    setQueueItems([]);
    try {
      localStorage.removeItem(BATCH_QUEUE_STORAGE_KEY);
    } catch {
      // ignore
    }
    setStatusBanner({
      type: 'info',
      message: `Cleared batch queue and purged ${itemsToPurge.length} saved project cache(s).`,
    });
  }, [apiBaseUrl, isQueueRunning, queueItems]);

  const handleStartBatchQueue = useCallback(
    async (profilesList = [], targetIds = null, overrideOutputDir = null) => {
      const isAutoExport = queueRunMode === 'auto_export';
      const targetDir = String(overrideOutputDir ?? queueOutputDir).trim();
      if (queueItems.length === 0 || isQueueRunning) return;
      if (isAutoExport && !targetDir) return;

      const idSet = Array.isArray(targetIds) ? new Set(targetIds) : null;
      const snapshotItems = idSet
        ? queueItems.filter((q) => idSet.has(q.id))
        : queueItems.filter((q) => q.selected !== false);
      if (snapshotItems.length === 0) return;

      await flushCurrentProjectAutosave();
      queueCancelRef.current = false;
      reimportedQueueItemIdsRef.current.clear();
      setIsQueueRunning(true);
      fetch(`${apiBaseUrl}/api/system/keep-awake`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: true }),
      }).catch(() => {});
      fetch(`${apiBaseUrl}/api/system/taskbar-progress`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state: 'normal', value: 0, max: Math.max(1, snapshotItems.length) }),
      }).catch(() => {});

      const formats =
        queueExportFormat === 'both'
          ? ['cbz', 'pdf']
          : [queueExportFormat === 'pdf' ? 'pdf' : 'cbz'];

      const existingOutNames = new Set();
      if (isAutoExport && queueSkipExisting && targetDir) {
        try {
          const scanResp = await fetch(`${apiBaseUrl}/api/import/scan-batch-root`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder: targetDir }),
          });
          if (scanResp.ok) {
            const scanData = await scanResp.json();
            for (const it of scanData?.items || []) {
              if (it?.name) existingOutNames.add(String(it.name).toLowerCase());
            }
          }
        } catch {
          // ignore if output dir does not exist yet
        }
      }

      const updateQueueRow = (id, patch) => {
        setQueueItems((prev) =>
          prev.map((q) => (q.id === id ? { ...q, ...patch } : q))
        );
      };

      let completedTitles = 0;

      try {
        for (let qIdx = 0; qIdx < snapshotItems.length; qIdx += 1) {
          if (queueCancelRef.current) break;
          const item = snapshotItems[qIdx];
          if (
            (isAutoExport && item.status === 'done') ||
            (!isAutoExport && (item.status === 'ready' || item.status === 'done'))
          ) {
            completedTitles += 1;
            continue;
          }

          const cleanStem = item.stem || item.name;
          const expectedFiles = formats.map((ext) =>
            formatBatchExportFilename(queueFilenamePattern, cleanStem, ext)
          );

          if (
            isAutoExport &&
            queueSkipExisting &&
            expectedFiles.every((fn) => existingOutNames.has(fn.toLowerCase()))
          ) {
            updateQueueRow(item.id, { status: 'skipped', progressText: 'Skipped (Exists)' });
            continue;
          }

          const itemPid =
            item.projectId || `qproj_${String(item.id).replace(/[^a-zA-Z0-9_-]/g, '')}`;
          const useSharedSeries = Boolean(queueShareContinuity && item.profileSlug);
          const convPid = useSharedSeries ? `series_${item.profileSlug}` : itemPid;
          projectCacheIdRef.current = convPid;
          activeQueueItemMetaRef.current = {
            id: item.id,
            path: item.path,
            kind: item.kind,
            projectId: itemPid,
          };

          if (item.profileSlug) {
            const matchedProf = profilesList.find((p) => p.slug === item.profileSlug);
            if (matchedProf) {
              setActiveProfileSlug(matchedProf.slug);
              await applySeriesProfileToStore(matchedProf, apiBaseUrl, { syncBackend: true });
            }
          }

          try {
            updateQueueRow(item.id, {
              projectId: itemPid,
              status: 'importing',
              progressText: 'Importing…',
              error: null,
            });
            const impUrl =
              item.kind === 'folder'
                ? `${apiBaseUrl}/api/import/folder-path`
                : `${apiBaseUrl}/api/import/archive-path`;
            const impResp = await fetch(impUrl, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ path: item.path }),
            });
            if (!impResp.ok) {
              const errJ = await impResp.json().catch(() => ({}));
              throw new Error(errJ?.detail || `Import HTTP ${impResp.status}`);
            }
            const impData = await impResp.json();
            const extracted = impData.pages || [];
            if (extracted.length === 0) {
              throw new Error('No pages found in source');
            }

            const ts = Date.now();
            const curCfg = useEditorStore.getState().pipelineConfig;
            const rawImportedPages = extracted.map((p, idx) => ({
              id: `page_q_${ts}_${idx}_${Math.random().toString(36).slice(2, 6)}`,
              name: p.filename || `page_${String(idx + 1).padStart(3, '0')}.png`,
              file: null,
              originalUrl: resolveBackendUrl(p.image_url, apiBaseUrl),
              cleanedImage: null,
              inpaintedUrl: null,
              maskUrl: null,
              maskId: null,
              dimensions: p.dimensions || null,
              blocks: [],
              status: 'idle',
              error: null,
              sourceLang: curCfg.sourceLang || 'ja',
              targetLang: curCfg.targetLang || 'id',
              model: null,
              tilesInpainted: 0,
            }));

            // Pin newly imported source pages immediately BEFORE detection so cache purges can never cause HTTP 404
            const importedPages = await pinPagesToProjectCache(itemPid, rawImportedPages);

            replaceProjectState({
              pages: importedPages,
              activePageId: importedPages[0]?.id || null,
            });
            setImportBaseName(cleanStem);

            activeQueueItemIdRef.current = item.id;
            updateQueueRow(item.id, {
              pageCount: importedPages.length,
              status: 'processing',
              progressText: `Processing p.1/${importedPages.length}`,
            });

            const stages = ['detect', 'ocr', 'inpaint', 'translate'].filter((k) =>
              Boolean(curCfg.enabledStages?.[k])
            );
            const runRes = await runStagesForPages(
              importedPages.map((p) => p.id),
              stages.length > 0 ? stages : ['detect', 'ocr', 'inpaint', 'translate']
            );
            activeQueueItemIdRef.current = null;
            if (runRes?.aborted || queueCancelRef.current) {
              updateQueueRow(item.id, { status: 'queued', progressText: 'Stopped' });
              break;
            }

            // Pin cleaned/mask outputs & save isolated per-project autosave so this title stays editable
            const latestPages = useEditorStore.getState().pages;
            const pinnedPages = await pinPagesToProjectCache(itemPid, latestPages);
            const snap = buildAutosaveSnapshot(itemPid, cleanStem, pinnedPages);
            await fetch(`${apiBaseUrl}/api/projects_cache/autosave`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(snap),
            });

            if (isAutoExport) {
              updateQueueRow(item.id, {
                status: 'exporting',
                progressText: `Exporting (${formats.join('+').toUpperCase()})…`,
              });
              if (!canvasRef.current?.exportArchive) {
                throw new Error('Canvas archive exporter is not ready');
              }
              for (const ext of formats) {
                const outFilename = formatBatchExportFilename(
                  queueFilenamePattern,
                  cleanStem,
                  ext
                );
                await canvasRef.current.exportArchive(pinnedPages, ext, {
                  filename: cleanStem,
                  imageFormat: 'png',
                  quality: 0.98,
                  outputDir: targetDir,
                  outputFilename: outFilename,
                  overwrite: false,
                  silent: true,
                });
                existingOutNames.add(outFilename.toLowerCase());
              }
              completedTitles += 1;
              updateQueueRow(item.id, {
                projectId: itemPid,
                pageCount: pinnedPages.length,
                status: 'done',
                exportStatus: 'Exported',
                progressText: `Exported (${pinnedPages.length}p)`,
              });
            } else {
              completedTitles += 1;
              updateQueueRow(item.id, {
                projectId: itemPid,
                pageCount: pinnedPages.length,
                status: 'ready',
                progressText: `Ready to Edit (${pinnedPages.length}p)`,
                selected: true,
              });
            }
          } catch (err) {
            if (queueCancelRef.current) {
              updateQueueRow(item.id, { status: 'queued', progressText: 'Stopped' });
              break;
            }
            updateQueueRow(item.id, {
              status: 'error',
              progressText: 'Error',
              error: err?.message || 'Queue item failed',
            });
          } finally {
            activeQueueItemIdRef.current = null;
            activeQueueItemMetaRef.current = null;
            if (!useSharedSeries) {
              await fetch(`${apiBaseUrl}/api/conversation/${encodeURIComponent(convPid)}`, {
                method: 'DELETE',
              }).catch(() => {});
            }
            // Await cache clear strictly AFTER export finishes (Auto-Export mode only) so it never races the next title's import
            if (isAutoExport && queuePurgeCache) {
              await fetch(`${apiBaseUrl}/api/cache/clear`, { method: 'POST' }).catch(() => {});
            }
            replaceProjectState({ pages: [], activePageId: null });
          }
        }
      } finally {
        activeQueueItemIdRef.current = null;
        activeQueueItemMetaRef.current = null;
        setIsQueueRunning(false);
        fetch(`${apiBaseUrl}/api/system/keep-awake`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled: false }),
        }).catch(() => {});
        fetch(`${apiBaseUrl}/api/system/taskbar-progress`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ state: 'none', value: 0, max: 100 }),
        }).catch(() => {});
        setStatusBanner({
          type: queueCancelRef.current ? 'warning' : 'success',
          message: queueCancelRef.current
            ? `Batch Series Queue stopped (${completedTitles}/${snapshotItems.length} completed).`
            : isAutoExport
              ? `Batch Series Queue finished (${completedTitles}/${snapshotItems.length} exported to ${targetDir}).`
              : `Batch Series Queue finished (${completedTitles}/${snapshotItems.length} ready to edit).`,
        });
      }
    },
    [
      queueRunMode,
      queueOutputDir,
      queueItems,
      isQueueRunning,
      queueExportFormat,
      queueSkipExisting,
      queueFilenamePattern,
      queueShareContinuity,
      queuePurgeCache,
      apiBaseUrl,
      flushCurrentProjectAutosave,
      pinPagesToProjectCache,
      buildAutosaveSnapshot,
      replaceProjectState,
      runStagesForPages,
    ]
  );


  return {
    processSinglePage,
    runSingleStageOnPage,
    runStagesForPages,
    handleCancelBatch,
    handleStopBatchQueue,
    buildAutosaveSnapshot,
    pinPagesToProjectCache,
    flushCurrentProjectAutosave,
    handleOpenQueueItemInEditor,
    handleBatchExportQueueItems,
    handleClearQueueAndPurgeCaches,
    handleStartBatchQueue,
  };
}
