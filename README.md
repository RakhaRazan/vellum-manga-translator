# Vellum

A local desktop app for translating and typesetting Japanese manga, Korean manhwa, and Chinese manhua on Windows.

It runs as a 100% native Windows desktop application with a pure modern C++17 backend (ONNX Runtime C++ API + Microsoft Edge WebView2) and a React + Konva canvas editor. `Vellum.exe` runs completely standalone with **zero Python runtime dependency** (no Python, PyTorch, CUDA toolkit, or Electron installation required). The whole Windows release zip is under 8 MB.

## Download

Grab the latest `Vellum-Windows-x64.zip` from [Releases](https://github.com/HoldMyAss/vellum-manga-translator/releases), extract it, and run `Vellum.exe`.

Requirements: Windows 10/11 (x64) with Microsoft Edge WebView2 (already included on most Windows systems).

On first launch, Vellum will download the ONNX models it needs (detection, OCR, and inpainting) into `%LOCALAPPDATA%\ComicTranslator\models`, or you can drop them into a `models/` folder next to `Vellum.exe`. If you already use Koharu, Vellum can import its local model cache automatically from **Settings -> Pipeline**.

## What it does

- **Import & batch queue:** Open single images, folders, or `.cbz`/`.zip` archives. Nothing runs automatically on import; pick the pages and pipeline stages you want (`Detect`, `OCR`, `Inpaint`, `Translate`) and hit **Run**.
- **Detection:** Supports Koharu RF-DETR-Seg-2XL and ComicTextDetector (ONNX), with automatic reading-order sorting (RTL for Japanese, LTR for Korean/Chinese).
- **OCR:** Uses `manga-ocr` (ONNX) for Japanese, `RapidOCR` (PP-OCRv4) for Korean and Chinese, or optional `PaddleOCR-VL-1.6` GGUF via LM Studio.
- **Inpainting:** Tile-based LaMa inpainting (`lama_manga.onnx` or `lama_fp32.onnx`) over detected text masks.
- **Translation:** Works with local models (LM Studio, Ollama) or cloud APIs (OpenAI, Gemini, Anthropic, DeepSeek, OpenRouter, or any OpenAI-compatible endpoint). Supports custom prompts, glossaries, and optional page-image vision context.
- **Typesetting editor:** Edit, drag, resize, or draw text boxes directly on the canvas, manage local/system/Google fonts, hold to compare against the original page, and export to PNG, JPG, CBZ, or PDF.

## Setting up a translator (LM Studio)

By default, Vellum looks for a local server from [LM Studio](https://lmstudio.ai/) on port `1235`:

1. Load a model in LM Studio.
2. Open the **Developer / Local Server** tab, set the port to `1235`, enable **CORS**, and start the server.
3. Vellum will pick it up automatically.

If you prefer Ollama or a cloud API key (OpenAI, Gemini, Claude, DeepSeek, OpenRouter, etc.), configure it under **Settings -> Providers**.

## Models

Vellum runs ONNX Runtime with `CUDA -> DirectML -> CPU` fallback (you can force CPU-only or toggle VRAM auto-unload in **Settings -> Pipeline**).

| Stage | Models |
| :--- | :--- |
| Detection | `koharu-layout-rfdetr-seg-2xl-1152.onnx`, `comictextdetector.pt.onnx` |
| OCR | `manga-ocr` (JA), `RapidOCR PP-OCRv4` (KO/ZH), `PaddleOCR-VL-1.6` GGUF (optional) |
| Inpainting | `lama_manga.onnx`, `lama_fp32.onnx` |

If you have an existing Koharu Hugging Face cache with `.safetensors` weights and want to convert them locally to `.onnx`, an optional standalone developer conversion script is provided in `packaging/convert_koharu_to_onnx.bat` / `packaging/convert_koharu_to_onnx.py` (which requires Python with `torch`, `safetensors`, `onnx`, `transformers`, and `timm`). Note that this is strictly an optional developer helper—standard Vellum users do **not** need Python installed, as pre-converted ONNX models download directly.

## Building from source

### Prerequisites
- Windows 10/11 (x64)
- Visual Studio 2022 (or Build Tools) with **Desktop development with C++** and CMake
- Node.js 18+

All C++ dependencies (`cpp-httplib`, `nlohmann/json`, `webview`, WebView2, ONNX Runtime, `miniz`, `stb`) are fetched automatically by CMake. No `vcpkg` needed.

### Release build

```bat
packaging\build_cpp_release.bat
```

This builds the React frontend, compiles `backend-cpp` in Release mode, copies the runtime into `Vellum_Runtime/`, places the launcher `Vellum.exe` in the project root, and outputs `release/Vellum-Windows-x64.zip`.

### Dev mode (hot reload)

After compiling the C++ backend once, run:

```bat
start_dev.bat
```

This launches the C++ backend in headless mode (`127.0.0.1:8000`) alongside the Vite dev server (`127.0.0.1:5173`).

### Manual CMake build

```powershell
cd frontend
npm install
npm run build
cd ..\backend-cpp
cmake -B build -A x64
cmake --build build --config Release --parallel
```

To build against CUDA or CPU-only ONNX Runtime instead of DirectML, pass `-DVELLUM_ORT_BACKEND=CUDA` or `-DVELLUM_ORT_BACKEND=CPU` (or `-DMANGAFLOW_ORT_BACKEND=...`) to CMake.

### Useful flags & env vars

- `--headless` / `VELLUM_HEADLESS=1` (or `MANGAFLOW_HEADLESS=1`) - run the HTTP backend without opening the WebView2 window
- `VELLUM_DEBUG=1` (or `MANGAFLOW_DEBUG=1`) - enable WebView2 DevTools (`F12`)
- `VELLUM_PORT=8000` (or `MANGAFLOW_PORT=8000`) - override the local backend port
- `VELLUM_FRONTEND_DIST=<path>` - specify an explicit directory for `index.html` frontend assets
- `MANGA_MODELS_DIR=<path>` - use a custom models directory
- `KOHARU_HF_CACHE=<path>` - point to a custom Koharu Hugging Face cache

## Project layout

- `backend-cpp/` - C++17 HTTP server, WebView2 host, ONNX inference pipeline, and CBZ/PDF exporter
- `frontend/` - React 19, Zustand, Tailwind CSS, and `react-konva` editor UI
- `packaging/` - Windows release script, icon, and optional Koharu ONNX converter
- `docs/` - Architecture notes and API spec (`SPEC.md`)
- `samples/` - Sample JA / KO / ZH pages for testing

## Credits

Vellum does not bundle third-party model weights. Supported models are downloaded from their original sources or imported from your local cache, and remain subject to their respective licenses:

- [Koharu (`mayocream`)](https://huggingface.co/mayocream) - `koharu-layout-rfdetr-seg-2xl-1152`, `lama-manga`, and `manga-ocr` weights
- [comic-text-detector](https://github.com/dmMaze/comic-text-detector) - ComicTextDetector model
- [manga-ocr](https://github.com/kha-white/manga-ocr) - Japanese manga OCR
- [RapidOCR](https://github.com/RapidAI/RapidOCR) / [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Korean & Chinese OCR models
- [LaMa](https://github.com/advimman/lama) - Image inpainting architecture
