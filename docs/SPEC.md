# Manga/Manhwa/Manhua Comic Translator — System Specification (v2)

## 1. Overview & Project Vision

This project is a high-performance, local-first, offline-capable desktop/web application for translating Japanese Manga (`ja`), Korean Manhwa (`ko`), and Chinese Manhua (`zh`) into Indonesian (`id`) or English (`en`).

### Key Principles (v2 Lightweight Architecture)
- **100% Local & Privacy-Preserving**: Zero cloud dependencies during standard translation operations.
- **Hardware Agnostic with Safe Fallback**: Auto-detects ONNX Runtime GPU acceleration in priority order (`CUDAExecutionProvider` → `DmlExecutionProvider` via `onnxruntime-directml` → `CPUExecutionProvider`), with DirectML-safe `SessionOptions` and automatic runtime CPU fallback on GPU/VRAM failure.
- **Lazy Loading & Aggressive RAM/VRAM Efficiency**: Every ONNX session (`ComicTextDetector`, `RapidOCR`, `LamaInpainter`) and Japanese `manga-ocr` model is lazily loaded on demand and explicitly unloaded (`unload()` + `gc.collect()`) immediately after each stage/page finishes.
- **Zero-Base64 Image Delivery**: Processed inpainted images and binary masks are saved atomically to a bounded local temp/cache directory and served via lightweight URLs (`/api/cache/images/{image_id}.png`) using FastAPI `FileResponse` (reducing JSON responses from megabytes to ~5 KB, with optional `include_base64=true` backward compatibility).
- **Both Full-Page & Granular Stage APIs**: Supports one-shot full-page translation (`POST /api/process-page`) as well as independent single-stage endpoints (`POST /api/stage/detect`, `/api/stage/ocr`, `/api/stage/inpaint`, `/api/stage/translate`) for interactive human-in-the-loop editing.
- **Context-Aware LLM Translation**: Translates full-page dialogue blocks in a single request via local LM Studio (`http://localhost:1235/v1`) with Structured JSON Schema output, optional glossary/custom instructions, and optional multimodal page-image context (with automatic text-only fallback).
- **Lossless Inpainting & Interactive Canvas**: Backgrounds and screentones are reconstructed via tile-based LaMa (`512x512` ROI patches); translated text is rendered and edited interactively via `react-konva`.
- **Desktop Packaging**: Bundled as a standalone Windows executable (`.exe`) using a Tauri desktop shell and a PyInstaller-frozen FastAPI sidecar.

---

## 2. Directory Structure

```text
Manga/
├── .gitignore
├── pyrightconfig.json                # Type-checking config targeting backend/venv
├── start_dev.bat                     # Local dev launcher (FastAPI :8000 + Vite :5173)
├── docs/
│   └── SPEC.md                       # v2 architecture, endpoints, and JSON contracts
├── backend/
│   ├── main.py                       # FastAPI entry point, image cache, full-page & /api/stage/* routes
│   ├── requirements.txt              # Backend Python dependencies
│   ├── test_models.py                # Offline ModelManager verification
│   ├── test_pipeline.py              # End-to-end JA/KO/ZH CV + OCR + LaMa verification
│   ├── test_api.py                   # Live HTTP API verification (JA/KO/ZH + FileResponse cache)
│   └── core/
│       ├── __init__.py
│       ├── model_manager.py          # Offline weight verification, SSE downloader, ORT provider/session factory
│       ├── detector.py               # Comic Text Detector (ONNX) + reading-order sorter + binary mask
│       ├── ocr.py                    # OCR Router: manga-ocr (ja, post-OCR unload) & RapidOCR PP-OCRv4 (ko, zh)
│       ├── inpainter.py              # Tile-based LaMa Inpainting (ONNX, 512x512 ROI patches)
│       ├── translator.py             # LM Studio client (OpenAI SDK, JSON Schema, optional VLM/glossary)
│       └── exporter.py               # .cbz/.zip natural-order importer & .cbz/.pdf (img2pdf) exporter
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js                # Dev server (:5173) with /api proxy -> :8000
│   ├── src/
│   │   ├── App.jsx                   # Batch queue, pipeline client, state management
│   │   ├── main.jsx
│   │   └── components/
│   │       ├── MangaCanvas.jsx       # react-konva multi-layer editor & full-res export
│   │       ├── Toolbar.jsx           # Language selectors, model/provider badge, export
│   │       ├── TextSidebar.jsx       # Dialogue editor & typography controls
│   │       └── DownloadModal.jsx     # First-launch SSE model download progress UI
│   └── src-tauri/                    # Tauri Rust host & sidecar configuration
├── models/                           # Local offline weights (comictextdetector, lama, rapidocr, manga-ocr-base)
├── samples/                          # Sample comic pages (sample_ja.jpg, sample_ko.jpg, sample_zh.jpg)
└── packaging/
    ├── build_backend.py              # PyInstaller build driver
    ├── comic-backend.spec            # PyInstaller sidecar spec
    └── tauri/                        # Tauri packaging templates
```

---

## 3. Pipeline Architecture (v2)

Each stage runs sequentially with lazy initialization and immediate session unloading so peak RAM/VRAM stays minimal. Callers may either run the entire pipeline in one request (`POST /api/process-page`) or drive each stage independently (`POST /api/stage/*`) to let users add/remove/adjust bounding boxes and masks between steps.

```text
[Raw Page Image]
       │
       ▼
[Stage 1: Comic Text Detector (ONNX)] ─── POST /api/stage/detect
  • Lazy-loads models/comictextdetector.pt.onnx (CUDA -> DirectML -> CPU)
  • Outputs reading-order sorted bboxes (`rtl` for ja; `ltr` for ko/zh) + cached binary mask PNG
  • Immediately unloads Detector ONNX session
       │
       ├───► (Optional Interactive Review: User edits/selects bboxes & mask)
       ▼
[Stage 2: Multi-Language OCR Router] ──── POST /api/stage/ocr
  ├── ja ──────► manga-ocr (models/manga-ocr-base) -> explicitly unloaded from RAM on completion
  └── ko / zh ──► RapidOCR (PP-OCRv4 ONNX, CUDA/DirectML -> CPU) -> sessions unloaded on completion
       │
       ▼
[Stage 3: Tile-Based LaMa Inpainter] ──── POST /api/stage/inpaint
  • Lazy-loads models/lama_fp32.onnx (512x512 ROI tiles, CUDA -> DirectML -> CPU)
  • Saves lossless PNG to local temp/cache -> returns `/api/cache/images/{id}.png`
  • Immediately unloads LaMa ONNX session
       │
       ▼
[Stage 4: Context-Aware LLM Translator] ─ POST /api/stage/translate (or /api/retranslate)
  • Queries local LM Studio (`http://localhost:1235/v1`) with Structured JSON Schema
  • Supports optional page image (VLM context w/ text-only fallback), glossary, custom instructions, model/temp overrides
       │
       ▼
[Interactive Canvas (react-konva)]
  ├── Layer 1: Inpainted Background fetched via GET /api/cache/images/{id}.png
  └── Layer 2: Editable Text Elements (Draggable, Resizable, Styled, Full-Res Export)
```

---

## 4. API Endpoints (FastAPI @ `localhost:8000`)

### 4.1 System, Model Management & Image Cache
- `GET /api/health`
  - Reports server status, active `execution_provider` (`CUDAExecutionProvider`, `DmlExecutionProvider`, or `CPUExecutionProvider`), `available_providers`, `models_ready`, and LM Studio status.
- `GET /api/models/status`
  - Offline verification of all required local CV/OCR model files in `models/` plus LM Studio connectivity.
- `GET /api/models/download` (alias `GET /api/models/download-stream`)
  - Server-Sent Events (SSE) stream reporting real-time Hugging Face Hub download progress for any missing model weights.
- `GET /api/lmstudio/status`
  - Pings `http://localhost:1235/v1/models` and returns connection status, active/resolved model ID, and available models.
- `GET /api/cache/images/{image_id}.png`
  - Serves cached inpainted PNGs and binary masks from the local temp/cache directory via `FileResponse` (`media_type="image/png"`, `Cache-Control: public, max-age=3600`). Validates `image_id` strictly (`^[A-Za-z0-9_-]{1,128}$`) against path traversal.

### 4.2 Granular Single-Stage Endpoints (v2)
- `POST /api/stage/detect` (alias `POST /api/pipeline/detect`)
  - **Content-Type**: `multipart/form-data`
  - **Fields**: `image` (or `file`), `source_lang` (`ja`/`ko`/`zh`, default `ja`), optional `conf_threshold`, `nms_threshold`, `mask_threshold`, `include_mask_base64` (`false` default).
  - **Returns**: Reading-order sorted `bubbles`/`blocks`, `reading_order` (`"rtl"` for `ja`, `"ltr"` for `ko`/`zh`), `ordered_ids`, and cached binary mask (`mask_id`, `mask_url`, `mask`, plus optional `mask_base64`).
- `POST /api/stage/ocr` (alias `POST /api/pipeline/ocr`)
  - **Content-Type**: `multipart/form-data`
  - **Fields**: `image` (or `file`), `bboxes` (or `blocks`/`bubbles`) as a JSON string containing either `[[x1, y1, x2, y2], ...]` or bubble objects from `/api/stage/detect`, and `source_lang` (`ja`/`ko`/`zh`, default `ja`).
  - **Returns**: Extracted `texts` array and structured `blocks`/`bubbles` with `original_text` and `source_text`. Unloads `manga-ocr` / RapidOCR sessions immediately after completion.
- `POST /api/stage/inpaint` (alias `POST /api/pipeline/inpaint`)
  - **Content-Type**: `multipart/form-data`
  - **Fields**: `image` (or `file`), plus at least one of: uploaded `mask` file, cached `mask_id`/`mask_url` (from `/api/stage/detect`), or selected `bboxes` (JSON string). Optional `tile_padding`, `refine_mask` (`true` default), `include_base64` (`false` default).
  - **Behavior**: If both a page mask and selected `bboxes` are given, restricts inpainting strictly to the selected boxes. If only `bboxes` are given, synthesizes a mask for those regions.
  - **Returns**: Lightweight cached PNG URL (`/api/cache/images/{id}.png`) in `cleaned_image`, `inpainted_image`, `cleaned_image_url`, and `inpainted_image_url`.
- `POST /api/stage/translate`
  - **Content-Type**: Accepts **either** `application/json` (`StageTranslateRequest`) **or** `multipart/form-data` (for attaching an optional page image file alongside JSON `blocks` and `config`).
  - **Fields**: `blocks` (or `bubbles`/`dialogues`), `source_lang` (`ja`/`ko`/`zh`), `target_lang` (`id`/`en`), optional page image (`image`/`file` upload, `image_url`, or `image_base64`), and optional `config` (`model`, `base_url`, `api_key`, `temperature`, `max_tokens`, `timeout`, `custom_instructions`, `glossary`, `use_vision`).
  - **Returns**: Translated `blocks`/`bubbles`, `translations` ID map, `dialogues`, active `model`, `has_page_image`, and `page_image_dimensions`.

### 4.3 Full-Page & Retranslation Endpoints (Preserved)
- `POST /api/process-page` (alias `POST /api/pipeline/process-page`)
  - Full sequential pipeline in one call: Detect → Reading-Order Sort → OCR → LaMa Tile Inpaint → LM Studio Translate, unloading every ONNX/OCR session after each stage.
  - Accepts `image` (or `file`), `source_lang`, `target_lang`, and optional `include_base64` (`false` default; set `true` for backward-compatible inline base64 data URI).
- `POST /api/retranslate` (alias `POST /api/pipeline/translate`)
  - Re-translates existing dialogue blocks when switching target languages (`id` ↔ `en`) without re-running CV/OCR/Inpainting.

### 4.4 Archive Import & Multi-Page Export Endpoints (v2)
- `POST /api/import/archive`
  - **Content-Type**: `multipart/form-data` (`archive` or `file` upload: `.cbz` or `.zip`).
  - Extracts valid comic page images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`) sorted in natural page order (`page_2.jpg` before `page_10.jpg`), ignoring macOS/hidden artifacts, caches them locally, and returns ordered page metadata with `/api/cache/images/{id}.{ext}` URLs.
- `POST /api/export/cbz`
  - **Content-Type**: `multipart/form-data` (ordered `images`/`files` uploads) or `application/json` (ordered `pages`/`images` of base64 data URIs or `/api/cache/images/...` URLs).
  - Returns a downloadable `.cbz` archive (`application/vnd.comicbook+zip`) with zero-padded natural-order page filenames.
- `POST /api/export/pdf`
  - **Content-Type**: Same as `POST /api/export/cbz`.
  - Uses lightweight `img2pdf` (lossless JPEG/opaque-PNG passthrough, automatic alpha flattening for RGBA PNGs) and returns a downloadable multi-page `.pdf` (`application/pdf`).

---

## 5. JSON Data Contracts (v2)

### 5.1 Bubble Definition (Core Unit of Data)
```json
{
  "id": "bubble_001",
  "order": 1,
  "bbox": [552, 113, 830, 196],
  "polygon": [[552, 113], [830, 113], [830, 196], [552, 196]],
  "original_text": "何をしているんだ？ ここは危険だぞ！",
  "source_text": "何をしているんだ？ ここは危険だぞ！",
  "translated_text": "Apa yang kamu lakukan? Di sini berbahaya!",
  "font_size": 19,
  "color": "#000000",
  "align": "center",
  "confidence": 0.9412,
  "text_direction": "horizontal",
  "style": {
    "font_family": "Anime Ace",
    "font_size": 19,
    "color": "#000000",
    "stroke_color": "#ffffff",
    "stroke_width": 2,
    "align": "center",
    "line_height": 1.2
  }
}
```

### 5.2 Structured Translation Schema (LM Studio Contract)
Sent as `response_format` JSON Schema to LM Studio's OpenAI-compatible Chat Completions endpoint:
```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "comic_translation_response",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {
        "dialogues": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "id": { "type": "string" },
              "translated_text": { "type": "string" }
            },
            "required": ["id", "translated_text"],
            "additionalProperties": false
          }
        }
      },
      "required": ["dialogues"],
      "additionalProperties": false
    }
  }
}
```

### 5.3 Complete Page Processing Response (`POST /api/process-page`)
```json
{
  "page_id": "sample_ja",
  "dimensions": { "width": 1000, "height": 1400 },
  "source_lang": "ja",
  "target_lang": "id",
  "source_language": "ja",
  "target_language": "id",
  "model": "gemma-4-e2b-uncensored-hauhaucs-aggressive",
  "execution_provider": "CPUExecutionProvider",
  "tiles_inpainted": 4,
  "cleaned_image_id": "3e2eb55408624c4f8df366aeb90fb637",
  "cleaned_image_url": "/api/cache/images/3e2eb55408624c4f8df366aeb90fb637.png",
  "inpainted_image_url": "/api/cache/images/3e2eb55408624c4f8df366aeb90fb637.png",
  "cleaned_image": "/api/cache/images/3e2eb55408624c4f8df366aeb90fb637.png",
  "inpainted_image": "/api/cache/images/3e2eb55408624c4f8df366aeb90fb637.png",
  "blocks": [],
  "bubbles": []
}
```
*(When `include_base64=true` is passed, `cleaned_image` and `inpainted_image` contain `data:image/png;base64,...` while `cleaned_image_url` and `inpainted_image_url` still return `/api/cache/images/{id}.png`.)*

### 5.4 Granular Stage Contracts (`POST /api/stage/*`)

#### `POST /api/stage/detect` Response
```json
{
  "page_id": "sample_ja",
  "dimensions": { "width": 1000, "height": 1400 },
  "source_lang": "ja",
  "source_language": "ja",
  "reading_order": "rtl",
  "execution_provider": "CPUExecutionProvider",
  "mask_id": "mask_9f86d081884c...",
  "mask_url": "/api/cache/images/mask_9f86d081884c....png",
  "mask": "/api/cache/images/mask_9f86d081884c....png",
  "ordered_ids": ["bubble_001", "bubble_002", "bubble_003", "bubble_004"],
  "bubbles": [
    {
      "id": "bubble_001",
      "order": 1,
      "bbox": [552, 113, 830, 196],
      "polygon": [[552, 113], [830, 113], [830, 196], [552, 196]],
      "confidence": 0.9412,
      "text_direction": "horizontal"
    }
  ],
  "blocks": []
}
```

#### `POST /api/stage/ocr` Response
```json
{
  "source_lang": "ja",
  "source_language": "ja",
  "engine": "manga-ocr",
  "count": 2,
  "texts": [
    "何をしているんだ？ ここは危険だぞ！",
    "まさか．．． ここで会うとはな！"
  ],
  "blocks": [
    {
      "id": "bubble_001",
      "order": 1,
      "bbox": [552, 113, 830, 196],
      "polygon": [[552, 113], [830, 113], [830, 196], [552, 196]],
      "original_text": "何をしているんだ？ ここは危険だぞ！",
      "source_text": "何をしているんだ？ ここは危険だぞ！",
      "confidence": 0.9412,
      "text_direction": "horizontal"
    }
  ],
  "bubbles": []
}
```

#### `POST /api/stage/inpaint` Response
```json
{
  "dimensions": { "width": 1000, "height": 1400 },
  "execution_provider": "CPUExecutionProvider",
  "tiles_inpainted": 4,
  "cleaned_image_id": "9b8264a30d674fb2b3e788d85ccc7c26",
  "cleaned_image_url": "/api/cache/images/9b8264a30d674fb2b3e788d85ccc7c26.png",
  "inpainted_image_url": "/api/cache/images/9b8264a30d674fb2b3e788d85ccc7c26.png",
  "cleaned_image": "/api/cache/images/9b8264a30d674fb2b3e788d85ccc7c26.png",
  "inpainted_image": "/api/cache/images/9b8264a30d674fb2b3e788d85ccc7c26.png"
}
```

#### `POST /api/stage/translate` Request (JSON) & Response
```json
{
  "source_lang": "ja",
  "target_lang": "id",
  "blocks": [
    { "id": "bubble_001", "bbox": [552, 113, 830, 196], "original_text": "何をしているんだ？" }
  ],
  "image_url": "/api/cache/images/9b8264a30d674fb2b3e788d85ccc7c26.png",
  "config": {
    "model": "Gemma 4 E2B Uncensored HauhauCS Aggressive Q4 K P",
    "temperature": 0.2,
    "max_tokens": 2048,
    "custom_instructions": "Keep dialogue punchy for shounen action.",
    "glossary": { "先輩": "senpai" },
    "use_vision": true
  }
}
```
Response:
```json
{
  "model": "gemma-4-e2b-uncensored-hauhaucs-aggressive",
  "source_lang": "ja",
  "target_lang": "id",
  "source_language": "ja",
  "target_language": "id",
  "has_page_image": true,
  "page_image_dimensions": { "width": 1000, "height": 1400 },
  "translations": { "bubble_001": "Apa yang kamu lakukan?" },
  "dialogues": [
    {
      "id": "bubble_001",
      "source_text": "何をしているんだ？",
      "translated_text": "Apa yang kamu lakukan?"
    }
  ],
  "blocks": [],
  "bubbles": []
}
```

### 5.5 Model Status Response (`GET /api/models/status`)
```json
{
  "all_ready": true,
  "execution_provider": "CPUExecutionProvider",
  "models_dir": "C:\\FILE PROYEKAN\\Manga\\models",
  "models": {
    "comic_text_detector": { "installed": true, "size_mb": 90, "downloaded_mb": 90.28, "files": [], "missing_files": [] },
    "manga_ocr": { "installed": true, "size_mb": 424, "downloaded_mb": 423.66, "files": [], "missing_files": [] },
    "rapid_ocr_ko_zh": { "installed": true, "size_mb": 41, "downloaded_mb": 40.7, "files": [], "missing_files": [] },
    "lama_inpaint": { "installed": true, "size_mb": 198, "downloaded_mb": 198.41, "files": [], "missing_files": [] }
  },
  "lm_studio": {
    "connected": true,
    "endpoint": "http://localhost:1235/v1",
    "active_model": "Gemma 4 E2B Uncensored HauhauCS Aggressive Q4 K P",
    "resolved_model_id": "gemma-4-e2b-uncensored-hauhaucs-aggressive",
    "preferred_model": "Gemma 4 E2B Uncensored HauhauCS Aggressive Q4 K P",
    "available_models": []
  }
}
```

---

## 6. Execution Strategy, Hardware Acceleration & RAM/VRAM Management

1. **Provider Auto-Detection & Fallback**:
   - `get_preferred_execution_providers()` checks `onnxruntime.get_available_providers()` in order:
     1. `CUDAExecutionProvider` (NVIDIA CUDA via `onnxruntime-gpu`)
     2. `DmlExecutionProvider` (Windows DirectML GPU acceleration via `onnxruntime-directml`, configured with `enable_mem_pattern=False` and `ORT_SEQUENTIAL`)
     3. `CPUExecutionProvider` (universal safe fallback)
   - Both session initialization (`create_onnx_session`) and runtime inference automatically fall back to `CPUExecutionProvider` if a GPU provider fails or runs out of VRAM.
2. **Lazy Loading & Per-Stage Session Unloading**:
   - Models are initialized only when their stage is invoked.
   - `ComicTextDetector.unload()`, `OCRRouter.unload()` (`unload_manga_ocr()` + `unload_rapid_ocr()`), and `LamaInpainter.unload()` run inside `finally:` blocks after every stage/page so zero ONNX sessions or PyTorch `manga-ocr` weights remain resident while idle.
3. **Japanese `manga-ocr` Memory Strategy**:
   - `manga-ocr` (`kha-white/manga-ocr-base`) runs as a local PyTorch `VisionEncoderDecoderModel` for maximum vertical tategaki and furigana accuracy; `unload_manga_ocr()` explicitly clears `.model`, `.processor`, and `.tokenizer`, calls `gc.collect()`, and flushes `torch.cuda.empty_cache()` immediately after Japanese OCR completes.
4. **Tile-Based Inpainting & Bounded Disk Cache**:
   - LaMa processes `512x512` ROI tiles around detected/selected boxes one tile at a time and composites only masked pixels back onto the original image.
   - Cached PNGs are written atomically (`*.tmp` → `os.replace`) and pruned automatically at `MAX_CACHED_IMAGES = 100`.

---

## 7. Packaging & Desktop Distribution (Windows `.exe`)

- **PyInstaller**:
  - Compiles FastAPI and ONNX Runtime dependencies into a single sidecar binary (`comic-backend.exe`).
  - Bundles `rapidocr_onnxruntime` and `manga_ocr` assets cleanly for offline execution.
- **Tauri Shell**:
  - Native Windows lightweight UI wrapper rendering the React + Konva frontend.
  - Controls the `comic-backend` sidecar lifecycle (starts on launch, terminates cleanly on window exit).
  - Resolves persistent model storage via `MANGA_MODELS_DIR` or `%LOCALAPPDATA%/ComicTranslator/models` when frozen, with automatic seeding from adjacent local `models/` directories.
