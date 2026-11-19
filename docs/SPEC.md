# Manga/Manhwa/Manhua Comic Translator - System Specification

## 1. Overview & Project Vision

This project is a high-performance, local-first, offline-capable desktop/web application for translating Japanese Manga (`ja`), Korean Manhwa (`ko`), and Chinese Manhua (`zh`).

### Key Principles
- **100% Local & Privacy-Preserving**: No cloud dependencies during standard translation operations.
- **Hardware Agnostic**: Accelerated via ONNX Runtime with CUDA GPU acceleration when available, with automatic CPU fallback.
- **Context-Aware Translation**: Full-page dialogue extraction translated in one LLM prompt using local GGUF models via LM Studio with strict Structured JSON outputs.
- **Lossless Inpainting & Interactive Canvas**: Backgrounds and screentones restored via LaMa inpainting; text rendered cleanly and edited interactively via `react-konva`.
- **Desktop Packaging**: Bundled as a standalone Windows executable (`.exe`) using a Tauri desktop shell and a PyInstaller-frozen FastAPI sidecar.

---

## 2. Directory Structure

```text
Manga/
├── .gitignore
├── docs/
│   └── SPEC.md                       # Complete architecture and API specifications
├── backend/
│   ├── app.py                        # FastAPI entry point & route registration
│   ├── config.py                     # Environment variables, model paths, defaults
│   ├── core/
│   │   ├── __init__.py
│   │   ├── detector.py               # Comic Text Detector (ONNX)
│   │   ├── ocr_router.py             # OCR Router: manga-ocr (ja) & rapidocr (ko, zh)
│   │   ├── inpainter.py              # LaMa Inpainting (ONNX)
│   │   ├── translator.py             # LM Studio client (OpenAI SDK + JSON Schema)
│   │   ├── downloader.py             # Hugging Face model weight downloader with SSE
│   │   └── utils.py                  # Image conversion, geometry, reading order sort
│   └── requirements.txt              # Backend dependencies
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── src/
│       ├── App.tsx
│       ├── main.tsx
│       ├── components/
│       │   ├── CanvasEditor.tsx      # react-konva multi-layer editor
│       │   ├── Toolbar.tsx           # Model settings, source language, page loader
│       │   ├── TextProperties.tsx    # Font, color, stroke, align, size controls
│       │   └── DownloadModal.tsx     # First-launch model download progress bar
│       ├── hooks/
│       │   └── usePipeline.ts        # API client and pipeline state manager
│       └── types/
│           └── pipeline.ts           # TypeScript interfaces matching backend contracts
├── models/                           # Default directory for offline model weights (.onnx, etc.)
├── samples/                          # Sample raw comic pages (Manga, Manhwa, Manhua)
└── packaging/
    ├── pyinstaller/
    │   └── backend.spec              # PyInstaller bundle spec for FastAPI sidecar
    └── tauri/
        ├── tauri.conf.json           # Tauri window, sidecar, and build configuration
        └── src-tauri/                # Tauri Rust host code
```

---

## 3. Pipeline Architecture

```
[Raw Page Image]
       │
       ▼
[1. Comic Text Detector (ONNX)]
       │
       ├───► Bounding Boxes & Text Mask (Polygon/Mask Array)
       │
       ▼
[2. Bubble Reading-Order Sorter]
       │ (Manga: Right-to-Left / Top-to-Bottom)
       │ (Manhwa/Manhua: Left-to-Right / Top-to-Bottom)
       │
       ├───► Sorted Text Crops
       │
       ▼
[3. Multi-Language OCR Router]
       ├── ja ──────► manga-ocr (kha-white/manga-ocr-base)
       └── ko / zh ──► RapidOCR (PP-OCRv4 ONNX weights)
       │
       ├───► Extracted Source Text per Bubble ID
       │
       ▼
[4. LLM Translation Engine (LM Studio @ localhost:1235)]
       │ (Full-page context prompt + Structured JSON Output)
       │
       ├───► Translated Text per Bubble ID
       │
       ▼
[5. Inpainter (LaMa ONNX)] ◄─── Uses Detector's Text Mask + Raw Image
       │
       ├───► Inpainted Background Image (Text Erased)
       │
       ▼
[6. Interactive Canvas (react-konva)]
       ├── Layer 1: Inpainted Background
       └── Layer 2: Editable Text Elements (Draggable, Resizable, Styled)
```

---

## 4. API Endpoints (FastAPI @ `localhost:8000`)

### 4.1 System & Model Management
* `GET /api/health`
  * Verifies server uptime, execution provider (`CUDAExecutionProvider` or `CPUExecutionProvider`), and LM Studio connectivity.
* `GET /api/models/status`
  * Checks presence of required model weights in `models/` directory.
* `GET /api/models/download-stream` (Server-Sent Events)
  * Streams real-time download and extraction progress for missing Hugging Face model weights.

### 4.2 Pipeline Operations
* `POST /api/pipeline/detect`
  * Accepts an image file; returns detected text bounding boxes and polygon masks.
* `POST /api/pipeline/ocr`
  * Accepts image + bounding boxes + source language (`ja`, `ko`, `zh`); returns OCR text per box.
* `POST /api/pipeline/translate`
  * Accepts array of bubble source texts; queries LM Studio with JSON Schema; returns translated texts.
* `POST /api/pipeline/inpaint`
  * Accepts image + mask; returns base64 or binary cleaned image.
* `POST /api/pipeline/process-page`
  * Full end-to-end pipeline in one call: Detect $\rightarrow$ Order $\rightarrow$ OCR $\rightarrow$ Translate $\rightarrow$ Inpaint.

---

## 5. JSON Data Contracts

### 5.1 Bubble Definition (Core Unit of Data)
```json
{
  "id": "bubble_001",
  "order": 1,
  "bbox": [120, 85, 340, 260],
  "polygon": [
    [120, 85],
    [340, 85],
    [340, 260],
    [120, 260]
  ],
  "source_text": "何をしているんだ？",
  "translated_text": "What are you doing?",
  "style": {
    "font_family": "Anime Ace",
    "font_size": 18,
    "color": "#000000",
    "stroke_color": "#ffffff",
    "stroke_width": 2,
    "align": "center",
    "line_height": 1.2
  }
}
```

### 5.2 Structured Translation Schema (LM Studio Contract)
Sent as the `response_format` JSON Schema to the OpenAI SDK endpoint:
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
              "translated_text": { "type": "string" },
              "notes": { "type": "string" }
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

### 5.3 Complete Page Processing Response (`POST /api/pipeline/process-page`)
```json
{
  "page_id": "page_001",
  "dimensions": {
    "width": 1080,
    "height": 1920
  },
  "source_language": "ja",
  "target_language": "en",
  "inpainted_image": "data:image/png;base64,iVBORw0KGgoAAA...",
  "bubbles": [
    {
      "id": "bubble_001",
      "order": 1,
      "bbox": [750, 100, 920, 280],
      "polygon": [[750, 100], [920, 100], [920, 280], [750, 280]],
      "source_text": "何をしているんだ？",
      "translated_text": "What are you doing?",
      "style": {
        "font_family": "Anime Ace",
        "font_size": 18,
        "color": "#000000",
        "stroke_color": "#ffffff",
        "stroke_width": 2,
        "align": "center",
        "line_height": 1.2
      }
    }
  ]
}
```

### 5.4 Model Status Response (`GET /api/models/status`)
```json
{
  "all_ready": false,
  "execution_provider": "CUDAExecutionProvider",
  "models": {
    "comic_text_detector": { "installed": true, "size_mb": 95 },
    "manga_ocr": { "installed": false, "size_mb": 420 },
    "rapid_ocr_ko_zh": { "installed": true, "size_mb": 35 },
    "lama_inpaint": { "installed": false, "size_mb": 210 }
  },
  "lm_studio": {
    "connected": true,
    "endpoint": "http://localhost:1235/v1",
    "active_model": "qwen2.5-7b-instruct"
  }
}
```

---

## 6. Execution Strategy & VRAM Management

1. **Lazy Model Loading**:
   - Models are initialized upon first inference request or pre-warmed sequentially.
2. **Sequential Stage Execution**:
   - Detection, OCR, and Inpainting execute sequentially in memory to prevent peak VRAM spikes.
3. **Low-VRAM Fallback Mode**:
   - If available GPU VRAM is $\le$ 4 GB or CUDA throws OOM, OCR and Inpainting automatically fall back to CPU execution (`CPUExecutionProvider`) while keeping Detection on GPU, leaving headroom for the LM Studio model.
4. **Tile-Based Inpainting**:
   - Rather than feeding high-resolution full webtoon images ($1080 \times 8000+$) directly into LaMa, crops are taken around bounding boxes with a 32px padding boundary, inpainted, and patched back onto the original canvas.

---

## 7. Packaging & Desktop Distribution (Windows `.exe`)

- **PyInstaller**:
  - Compiles FastAPI and ONNX Runtime dependencies into a single sidecar binary (`manga-backend.exe`).
  - `--collect-all rapidocr_onnxruntime` and `--collect-data manga_ocr` to prevent missing dynamic assets.
- **Tauri Shell**:
  - Native Windows lightweight UI wrapper rendering the React frontend.
  - Controls the `manga-backend.exe` sidecar lifecycle (starts on launch, terminates gracefully on window exit).
  - Configures model storage paths dynamically inside `%APPDATA%/MangaTranslator/models` or a local `./models/` folder.
