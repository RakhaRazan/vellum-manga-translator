# Vellum Comic Translator - System Specification (v3.0.0 Native C++17)

## 1. Overview & Project Vision

This project is a high-performance, local-first, offline-capable desktop application for translating Japanese Manga (`ja`), Korean Manhwa (`ko`), and Chinese Manhua (`zh`) into Indonesian (`id`) or English (`en`).

### Key Principles (v3.0.0 Native C++17 Architecture)
- **100% Native C++17 & Offline-First**: Zero Python runtime dependency. All computer vision, OCR, and inpainting models execute directly inside the native process via the official ONNX Runtime C++ API (`onnxruntime_cxx_api.h`).
- **Smart Hybrid Hardware Routing (`CUDAExecutionProvider` → `DmlExecutionProvider` → `CPUExecutionProvider`)**: In default `"hybrid"` mode, heavy 2D spatial vision models (Detector Engine v2 & LaMa/AOT Inpainting) run on the GPU while lightweight/autoregressive OCR (`Manga OCR`, `Baberu OCR`, `RapidOCR`) runs on CPU AVX2—saving ~650 MB VRAM for local LLMs in LM Studio. Also supports `"force_gpu"` and `"cpu_only"` modes.
- **100% Portable Relative Paths**: All runtime directories (`./models/`, `./fonts/`, `./cache/`, `./projects_cache/`, and `./settings.json`) resolve relative to the portable application root via `util::get_portable_root_dir()`.
- **Detector Engine v2 (`backend-cpp/src/detector/`)**: Supports `Vellum YOLO11m-Seg` (9-class FP16 GPU & INT8 CPU) and `ComicTextDetector` with Adaptive Tiling (`SAHI + WBF`) for tall Webtoons, interior mask-inscribed `safe_bbox`, IoA panel association, SFX suppression, and cycle-safe DAG topological reading order (`RTL` / `LTR`).
- **Multi-Engine OCR & Text-Stroke-Only Inpainting**: Routes across `Manga OCR` (JA), `Baberu OCR` (JA/ZH/EN), `Hayai OCR` (JA/ZH/KO/EN), `RapidOCR PP-OCRv4` (KO/ZH), and `PaddleOCR-VL 1.6` (GGUF VLM). Inpainting (`LaMa Manga`, `AOT Inpainting`, `FLUX.2 Klein`, `RORem Mixed`) erases only the `3px`-eroded interior text-stroke mask so speech bubble borders remain intact.
- **10,000-Page Single-Volume Endurance & 64-Bit ZIP64 (`>4 GB`) I/O**: Supports 64-bit disk-backed `.cbz` import (`mz_zip_reader_init_cfile`, `POST /api/import/archive-path`, up to `25,000` pages / `32 GiB`), explicit **ZIP64 (`>4 GB`)** `.cbz` export (`mz_zip_writer_init_cfile_v2` with `MZ_ZIP_FLAG_WRITE_ZIP64`), 10 GB-guarded PDF 1.4 streaming export, `50,000`-file image cache with auto-pinned `./projects_cache/<project_id>/`, and polygon-stripped disk-backed autosave (`./projects_cache/autosave.vellum.json`).

---

## 2. Directory Structure

```text
Vellum/
├── start_dev.bat                     # Dev launcher (Headless C++ :8000 + Vite :5173 HMR)
├── Vellum.exe                        # One-click native Win32 launcher (<100 KB)
├── docs/
│   └── SPEC.md                       # Native C++17 architecture, endpoints, and JSON contracts
├── backend-cpp/                      # 100% Native C++17 backend & WebView2 host
│   ├── CMakeLists.txt                # MSVC x64 build script (cpp-httplib, nlohmann/json, ONNX Runtime, webview)
│   └── src/
│       ├── main.cpp                  # cpp-httplib HTTP server, route registrations, WebView2 window
│       ├── launcher.cpp              # Minimal native root launcher
│       ├── settings_manager.hpp/.cpp # Persistent JSON settings manager
│       ├── font_manager.hpp/.cpp     # DirectWrite system font scanner & TTF/OTF loader
│       ├── exporter.hpp/.cpp         # miniz-based CBZ exporter & native PDF generator
│       ├── utils.hpp                 # Win32, string, UTF-8/UTF-16, base64, filesystem utils
│       └── pipeline/
│           ├── model_manager.hpp/.cpp # ONNX Runtime session factory, download manager, safetensors loader
│           ├── detector.hpp/.cpp     # ComicTextDetector (ONNX Runtime C++)
│           ├── ocr.hpp/.cpp          # manga-ocr & RapidOCR PP-OCRv4 (ONNX Runtime C++)
│           ├── inpainter.hpp/.cpp    # Tile-based LaMa inpainter (ONNX Runtime C++ + stb_image)
│           └── translator.hpp/.cpp   # Multi-provider LLM client (LM Studio, Ollama, OpenAI, Gemini, etc.)
├── frontend/                         # Modern React 19 + Konva + Zustand UI
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js                # Dev server (:5173) with /api proxy -> :8000
│   └── src/
│       ├── App.jsx                   # Batch queue, pipeline client, state management
│       ├── main.jsx
│       ├── components/
│       │   ├── MangaCanvas.jsx       # react-konva multi-layer editor & full-res export
│       │   ├── MenuBar.jsx           # App menu bar, keyboard shortcuts, About dialog
│       │   ├── Toolbar.jsx           # Language selectors, model/provider badge, export
│       │   ├── TextSidebar.jsx       # Dialogue editor & typography controls
│       │   ├── SettingsModal.jsx     # Appearance, pipeline, provider, translation, typesetting
│       │   ├── StatusBar.jsx         # Bottom bar: zoom, coords, runtime provider, LLM status
│       │   └── DownloadModal.jsx     # SSE model download progress UI
│       └── store/
│           └── useEditorStore.js     # Centralized Zustand store
├── models/                           # Local offline weights (.onnx)
├── samples/                          # Sample comic pages (sample_ja.jpg, sample_ko.jpg, sample_zh.jpg)
├── packaging/
│   ├── build_cpp_release.bat         # Automated one-click MSVC C++17 + Vite release packager
│   └── icon.ico                      # Win32 application icon
└── Vellum_Runtime/                   # Release bundle folder containing Vellum.exe & ONNX Runtime DLLs
```

---

## 3. Pipeline Architecture (v2)

Each stage runs sequentially with lazy initialization and immediate session unloading so peak RAM/VRAM stays minimal. Callers may either run the entire pipeline in one request (`POST /api/process-page`) or drive each stage independently (`POST /api/stage/*`) to let users add/remove/adjust bounding boxes and masks between steps.

```text
[Raw Page Image]
       │
       ▼
[Stage 1: YOLO11m-Seg / ComicTextDetector (ONNX Runtime C++)] ─── POST /api/stage/detect
  • Lazy-loads models/vellum-det-yolo11m-seg-1024-fp16.onnx or comictextdetector.pt.onnx (CUDA -> DirectML -> CPU)
  • Outputs reading-order sorted bboxes (`rtl` for ja; `ltr` for ko/zh) + cached binary mask PNG
  • Immediately resets Detector ONNX Runtime session
       │
       ├───► (Optional Interactive Review: User edits/selects bboxes & mask)
       ▼
[Stage 2: Multi-Language OCR Router (ONNX Runtime C++)] ──── POST /api/stage/ocr
  ├── ja ──────► manga-ocr ONNX -> session explicitly reset from memory on completion
  └── ko / zh ──► RapidOCR (PP-OCRv4 ONNX, CUDA/DirectML -> CPU) -> sessions reset on completion
       │
       ▼
[Stage 3: Tile-Based LaMa Inpainter (ONNX Runtime C++)] ──── POST /api/stage/inpaint
  • Lazy-loads models/lama_manga.onnx or lama_fp32.onnx (512x512 ROI tiles, CUDA -> DirectML -> CPU)
  • Saves lossless PNG to local temp/cache -> returns `/api/cache/images/{id}.png`
  • Immediately resets LaMa ONNX Runtime session
       │
       ▼
[Stage 4: Multi-Provider Context-Aware LLM Translator] ─ POST /api/stage/translate (or /api/retranslate)
  • Queries LM Studio (`http://localhost:1234/v1`), Ollama, or Cloud APIs (OpenAI, Gemini, Anthropic, DeepSeek, etc.)
  • Structured JSON Schema output, optional page image vision context, glossary, and custom instructions
       │
       ▼
[Interactive Canvas (react-konva)]
  ├── Layer 1: Inpainted Background fetched via GET /api/cache/images/{id}.png
  └── Layer 2: Editable Text Elements (Draggable, Resizable, Styled, Full-Res Export)
```

---

## 4. API Endpoints (Native C++17 `cpp-httplib` @ `localhost:8000`)

### 4.1 System, Model Management & Image Cache
- `GET /api/health`
  - Reports server status, active `execution_provider` (`CUDAExecutionProvider`, `DmlExecutionProvider`, or `CPUExecutionProvider`), `available_providers`, `models_ready`, and LM Studio status.
- `GET /api/models/status`
  - Offline verification of all required local CV/OCR model files in `models/` plus LM Studio connectivity.
- `GET /api/models/download` (alias `GET /api/models/download-stream`)
  - Server-Sent Events (SSE) stream reporting real-time Hugging Face Hub download progress for any missing model weights.
- `GET /api/lmstudio/status`
  - Pings `http://localhost:1234/v1/models` and returns connection status, active/resolved model ID, and available models.
- `GET /api/cache/images/{image_id}.png`
  - Serves cached inpainted PNGs and binary masks from the local temp/cache directory via native `cpp-httplib` file streaming (`media_type="image/png"`, `Cache-Control: public, max-age=3600`). Validates `image_id` strictly (`^[A-Za-z0-9_-]{1,128}$`) against path traversal.

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
  - Uses native C++17 PDF generator (`miniz` + `stb_image` for alpha flattening, lossless JPEG/PNG stream passthrough) and returns a downloadable multi-page `.pdf` (`application/pdf`).

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
  "model": "qwen2.5-7b-instruct",
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
    "model": "qwen2.5-7b-instruct",
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
  "model": "qwen2.5-7b-instruct",
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
  "models_dir": "./models",
  "models": {
    "comic_text_detector": { "installed": true, "size_mb": 90, "downloaded_mb": 90.28, "files": [], "missing_files": [] },
    "manga_ocr": { "installed": true, "size_mb": 424, "downloaded_mb": 423.66, "files": [], "missing_files": [] },
    "rapid_ocr_ko_zh": { "installed": true, "size_mb": 41, "downloaded_mb": 40.7, "files": [], "missing_files": [] },
    "lama_inpaint": { "installed": true, "size_mb": 198, "downloaded_mb": 198.41, "files": [], "missing_files": [] }
  },
  "lm_studio": {
    "connected": true,
    "endpoint": "http://localhost:1234/v1",
    "active_model": "qwen2.5-7b-instruct",
    "resolved_model_id": "qwen2.5-7b-instruct",
    "preferred_model": "qwen2.5-7b-instruct",
    "available_models": []
  }
}
```

---

## 6. Execution Strategy, Hardware Acceleration & Memory Lifecycle

1. **Provider Auto-Detection & Fallback**:
   - `ModelManager::get_available_providers()` queries the ONNX Runtime C++ API in order:
     1. `CUDAExecutionProvider` (NVIDIA CUDA via prebuilt `onnxruntime-gpu`)
     2. `DmlExecutionProvider` (Windows DirectML GPU acceleration via DirectX 12)
     3. `CPUExecutionProvider` (universal safe fallback)
   - DirectML sessions are created with `enable_mem_pattern = false` and `ORT_SEQUENTIAL` execution mode for maximum memory stability.
   - If GPU initialization or allocation fails at runtime, the session automatically falls back to `CPUExecutionProvider`.
2. **Lazy Loading & Explicit C++ Session Reset**:
   - Models are instantiated into `std::unique_ptr<Ort::Session>` only when their specific stage is invoked.
   - Calling `.reset()` on each `Ort::Session` immediately returns host and GPU VRAM to the operating system after each page/stage finishes.
3. **Japanese `manga-ocr` ONNX Strategy**:
   - `manga-ocr` runs via an optimized ONNX Runtime vision-encoder/text-decoder graph directly in C++ (eliminating Python, PyTorch, and `transformers` runtime dependencies completely).
4. **Tile-Based Inpainting & Bounded Disk Cache**:
   - LaMa processes `512x512` ROI tiles around detected text boxes one tile at a time and composites only masked pixels back onto the original image via `stb_image` / `stb_image_write`.
   - Cached PNGs are written atomically (`*.tmp` → `fs::rename`) and pruned automatically at `MAX_CACHED_IMAGES = 100`.

---

## 7. Packaging & Desktop Distribution (100% Native Windows Release)

- **Pure C++17 Desktop Executable (`Vellum.exe`)**:
  - Compiled using MSVC C++17 with `/O2`, `/GL`, `/Gy`, and static C/C++ runtime linking (`MultiThreaded` release).
  - Single standalone binary embedded with Microsoft Edge WebView2 control and `cpp-httplib` loopback server on `127.0.0.1:8000`.
  - Production frontend is compiled via `npm run build` (`frontend/dist/`) and served directly by the native backend into the WebView2 canvas.
- **Zero Python Runtime Dependency**:
  - The distributed release package contains only native binaries (`Vellum.exe`, `Vellum_Runtime/`, ONNX Runtime DLLs, DirectML DLLs).
  - End users do **not** need Python, PyTorch, CUDA Toolkit, node.js, or Electron installed.
- **One-Click Automated Packager**:
  - `packaging/build_cpp_release.bat` automates frontend building, CMake Release compilation, DLL staging into `Vellum_Runtime/`, and ZIP packaging (`release/Vellum-v3.0.0-Windows-x64-Universal.zip`).

---

## 8. Vellum v2.5 Studio Core Spec

Vellum v2.5 refines the native C++17 studio core around hardware-safe GPU/CPU execution, zero-secret provider configuration, pure C++17 computer vision heuristics, tiered inpainting, and print-ready export:

1. **Dual-Backend ONNX (`DML`/`CUDA`/`CPU` + `nvcuda.dll` Safe Probe)**:
   - Probes `nvcuda.dll` via `LoadLibraryExW` / `GetProcAddress` before registering `CUDAExecutionProvider` so systems without an NVIDIA driver never fail on missing CUDA DLLs.
   - Routes sessions cleanly across `DmlExecutionProvider` (DirectX 12), `CUDAExecutionProvider`, and `CPUExecutionProvider` with automatic CPU fallback.
2. **Provider-Agnostic LLM Profiles (`"none"` Manual Mode + DPAPI Key Encryption)**:
   - Supports neutral per-provider profiles (`custom`, `lmstudio` on optional `http://localhost:1234/v1`, `ollama`, `openai`, `gemini`, `anthropic`, `deepseek`, `openrouter`) plus `"none"` manual mode for 100% offline/manual lettering without an LLM.
   - API keys default to `""` and are encrypted at rest on Windows using DPAPI (`CryptProtectData` / `CryptUnprotectData`) before persisting to `settings.json`.
3. **Pure C++17 CV Core**:
   - **5-Stage Otsu Mask**: Luminance grayscale conversion, local contrast normalization, adaptive Otsu thresholding, morphological dilation/erosion, and interior stroke isolation without external CV libraries.
   - **0-VRAM Sobel 4θ Typography**: Pure CPU Sobel gradient orientation histograms (`0°`, `45°`, `90°`, `135°`) for text-direction and stroke-density analysis at zero VRAM cost.
   - **Waist-Cutting**: Splits merged vertical/horizontal speech-bubble components at narrow constriction waists.
   - **Panel-First Reading Order**: Associates bubbles to detected comic panels via IoA and resolves `RTL` (Japanese) or `LTR` (Korean/Chinese) reading order per panel.
4. **3-Tier CPU-First Inpainting + Tone Match**:
   - **Tier 1**: Fast CPU flat/median/telea fill for uniform speech-bubble interiors.
   - **Tier 2**: Local CPU-first ONNX LaMa / AOT tile inpainting (`512x512` ROI).
   - **Tier 3**: GPU-accelerated ONNX or external API inpainting for complex screentones, paired with boundary **Tone Match** histogram/luminance blending to eliminate seam halos.
5. **Narrow-Bubble DP Typesetter + Rubylith Brush**:
   - Dynamic-programming line-breaking typesetter that minimizes raggedness and overflow inside narrow or irregular speech bubbles.
   - Interactive **Rubylith Brush** overlay in the canvas editor for manual mask painting, erasing, and targeted re-inpainting.
6. **Layered `.psd` Export**:
   - Native C++17 Adobe Photoshop `.psd` writer exporting the original page, cleaned/inpainted background, and editable rasterized/grouped text layers so letterers can fine-tune in Photoshop or Clip Studio Paint.
