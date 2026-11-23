# Vellum

A high-performance, 100% portable local desktop studio for translating and typesetting Japanese manga, Korean manhwa, and Chinese manhua on Windows.

Built with a pure modern **C++17** backend (**ONNX Runtime C++ API** + **Microsoft Edge WebView2**) and a **React + Konva** canvas editor, `Vellum.exe` runs completely standalone with **zero Python runtime dependency** (no Python, PyTorch, CUDA toolkit, or Electron installation required).

## Download

Grab the latest `Vellum-Windows-x64.zip` from [Releases](https://github.com/HoldMyAss/vellum-manga-translator/releases), extract it anywhere, and double-click `Vellum.exe`.

- **Requirements:** Windows 10/11 (x64) with Microsoft Edge WebView2 Runtime (pre-installed on Windows 11).
- **100% Portable Relative Paths:** All runtime data resolves strictly relative to `Vellum.exe` (`./models/`, `./fonts/`, `./cache/`, `./projects_cache/`, and `./settings.json`)—zero `%LOCALAPPDATA%` or Registry clutter.
- On first launch, Vellum automatically checks `./models/` and lets you download missing ONNX models directly from **Settings -> Pipeline** (or import an existing local Koharu Hugging Face cache).

## Architecture & Scale Highlights (v2.0)

- **Smart Hybrid Hardware Routing (`CUDA -> DirectML -> CPU`):** Automatically routes heavy 2D spatial vision models (Detector & Inpainter) to your GPU (`CUDAExecutionProvider` / `DmlExecutionProvider`) while keeping lightweight/autoregressive OCR (`Manga OCR`, `Baberu OCR`, `RapidOCR`) on CPU AVX2—leaving maximum VRAM free for local LLMs in LM Studio. Also supports `Force All GPU` and `CPU Only` modes.
- **Detector Engine v2 & Adaptive Tiling:**
  - Supports **Vellum YOLO11m-Seg** (9-class instance segmentation, FP16 GPU & INT8 CPU), **Koharu Layout RF-DETR-Seg 2XL** (`High-Context • Heavy CPU/CUDA`), and **ComicTextDetector**.
  - Includes **Adaptive Tiling (SAHI + WBF)** for ultra-tall Webtoon strips (`800x12000px+`) and single-tile preservation for high-resolution tankobon pages (`2580x3643px`), interior mask-inscribed `safe_bbox`, IoA panel association, SFX suppression filtering, and cycle-safe DAG topological reading order (`RTL` for Japanese, `LTR` for Korean/Chinese).
- **Multi-Engine OCR Router:** Supports **Manga OCR** (JA vertical/Kanji specialist), **Baberu OCR** (fast `JA`/`ZH`/`EN` + SFX), **Hayai OCR** (aspect-ratio-preserving `JA`/`ZH`/`KO`/`EN`), **RapidOCR PP-OCRv4** (`KO`/`ZH`), and **PaddleOCR-VL 1.6** (`.gguf` VLM via local `llama-server` / LM Studio).
- **Text-Stroke-Only Screentone Inpainting:** Strictly separates the full container `bubble_mask` (used only for `safe_bbox` typesetting) from the `3px`-eroded interior text-stroke mask (used for inpainting), preserving crisp black bubble borders while erasing 100% of original CJK strokes across **LaMa Manga** (`lama_manga.onnx` / `lama_fp32.onnx`), **AOT Inpainting** (`aot-inpainting.onnx`), **FLUX.2 Klein**, and **RORem Mixed**.
- **Contextual LLM Translation:** Connects to **LM Studio** (`localhost:1235`), **Ollama**, **OpenAI**, **Gemini**, **Anthropic**, **DeepSeek**, **OpenRouter**, or any OpenAI-compatible endpoint with `<think>` tag stripping, O(N) JSON salvage recovery, custom prompts, glossaries, uncensored `adult` localization tone, and optional multimodal page vision.
- **10,000-Page Single-Volume Endurance & 64-Bit ZIP64 (`>4 GB`) I/O:** Interactive 60fps `react-konva` editor with 2D resolution-proportional bubble auto-fit, per-block manual `font_size` overrides, Windows & Google Fonts manager with symbol fallbacks (`♡ ♪ …`), Split-Curtain Before/After compare, virtualized filmstrip, polygon-stripped disk-backed autosave (`./projects_cache/autosave.vellum.json`), auto-pinned project cache (`50,000`-file LRU ceiling + one-click temp cache purge), Stop/Cancel & Skip-Done Resume batch controls, `O(1)`-RAM 64-bit disk-streaming `.cbz` import (`up to 25,000 pages / 32 GiB`), and explicit **ZIP64 (`>4 GB`)** `.cbz` & `.pdf` export tested up to 10,000-page volumes.

## Setting Up a Translator (LM Studio)

By default, Vellum connects to a local server from [LM Studio](https://lmstudio.ai/) on port `1235`:

1. Load a model in LM Studio.
2. Open the **Developer / Local Server** tab, set the port to `1235`, enable **CORS**, and click **Start Server**.
3. Vellum detects it automatically.

For Ollama or cloud APIs (OpenAI, Gemini, Anthropic, DeepSeek, OpenRouter), configure your endpoint and key under **Settings -> Providers**.

## Supported Models

| Stage | Model | ID / Weights | Languages / Role | Execution Target |
| :--- | :--- | :--- | :--- | :--- |
| **Detection** | **Vellum YOLO11m-Seg (FP16)** | `vellum-det-yolo11m-seg-1024-fp16.onnx` | 9-class bubble, panel & SFX segmentation (`SAHI+WBF`, `safe_bbox`, IoA, DAG order) | GPU (`CUDA` / `DirectML`) |
| **Detection** | **Vellum YOLO11m-Seg (INT8)** | `vellum-det-yolo11m-seg-1024-int8.onnx` | 9-class static QDQ INT8 detector for CPU-only systems | CPU AVX2 |
| **Detection** | **Koharu Layout RF-DETR-Seg 2XL** | `koharu-layout-rfdetr-seg-2xl-1152.onnx` | 1152px high-context layout, panel, bubble & SFX segmentation | Heavy CPU / CUDA |
| **Detection** | **ComicTextDetector** | `comictextdetector.pt.onnx` | 1024px text box & stroke mask detector | GPU (`CUDA` / `DirectML`) / CPU |
| **OCR** | **Manga OCR** | `manga-ocr/` (`encoder` + `decoder`) | Japanese (`JA`) vertical tategaki, furigana & Kanji specialist | CPU AVX2 (Hybrid) / GPU |
| **OCR** | **Baberu OCR** | `baberu-ocr/` (`vision_int4` + `decoder_int8`) | Fast `JA` / `ZH` / `EN` dialogue & stylized SFX OCR | CPU AVX2 (Hybrid) / GPU |
| **OCR** | **Hayai OCR** | `hayai-ocr/` (`encoder` + `decoder`) | Aspect-ratio-preserving `JA` / `ZH` / `KO` / `EN` manga & webtoon OCR | GPU / CPU |
| **OCR** | **RapidOCR PP-OCRv4** | `rapidocr/` (`det` + `cls` + `rec`) | Korean (`KO`) & Chinese (`ZH`) manhwa/manhua line recognition | CPU AVX2 (Hybrid) / GPU |
| **OCR** | **PaddleOCR-VL 1.6** | `paddleocr-vl-gguf/` (`.gguf` + `mmproj`) | Multilingual Vision-Language OCR (`JA` / `KO` / `ZH` / `EN`) | Local VLM (`llama-server` / LM Studio) |
| **Inpainting** | **LaMa Manga** | `lama_manga.onnx` / `lama_fp32.onnx` | 512×512 tile-based screentone & line-art text-stroke erasure | GPU (`CUDA` / `DirectML`) / CPU |
| **Inpainting** | **AOT Inpainting** | `aot-inpainting.onnx` | Ultra-fast 10-block AOT-GAN manga text-stroke inpainter | GPU (`CUDA` / `DirectML`) / CPU |
| **Inpainting** | **FLUX.2 Klein** | `flux-2-klein` (Diffusion API + LaMa fallback) | Generative diffusion ROI inpainter for complex backgrounds | Local Diffusion API (8GB+ VRAM) |
| **Inpainting** | **RORem Mixed** | `rorem-mixed` (Diffusion API + LaMa/AOT fallback) | Generative manga text/object removal GGUF adapter | Local Diffusion API |

An optional developer conversion helper for local Koharu `.safetensors` caches is included in `packaging/convert_koharu_to_onnx.bat` / `packaging/convert_koharu_to_onnx.py` (requires Python with `torch`, `safetensors`, `onnx`, `transformers`, and `timm`). Standard Vellum users do **not** need Python installed.

## Building from Source

### Prerequisites

- Windows 10/11 (x64)
- Visual Studio 2022 (or Build Tools) with **Desktop development with C++** and CMake
- Node.js 18+

All C++ dependencies (`cpp-httplib`, `nlohmann/json`, `webview`, WebView2, ONNX Runtime, `miniz`, `stb`) are fetched automatically by CMake. No `vcpkg` required.

### Release Build

```bat
packaging\build_cpp_release.bat
```

## Credits & Acknowledgments

Vellum stands on the shoulders of the open-source manga translation, OCR, and computer vision community:

- **[Koharu](https://github.com/mayocream/koharu)** (`mayocream`) - Inspiration for the modern local-first manga translation workflow, `koharu-layout-rfdetr-seg-2xl-1152`, and `lama-manga` weights.
- **[ogkalu](https://huggingface.co/ogkalu)** - Comic and manga speech-bubble / layout detection datasets and YOLO segmentation research.
- **[comic-text-detector](https://github.com/dmMaze/comic-text-detector)** (`dmMaze`) - Foundational manga text block and stroke mask detection architecture.
- **[manga-ocr](https://github.com/kha-white/manga-ocr)** (`kha-white`) - Japanese vertical/horizontal manga OCR model.
- **[baberu-ocr](https://huggingface.co/genshiai-daichi/baberu-ocr)** (`genshiai-daichi/baberu-ocr`) - Lightweight quantized `JA`/`ZH`/`EN` and SFX manga OCR weights.
- **[hayai-ocr-v2](https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2)** (`JustANormalTinkerer/hayai-ocr-v2`) - Aspect-ratio-preserving multilingual manga and webtoon OCR architecture.
- **[RapidOCR](https://github.com/RapidAI/RapidOCR)** & **[PaddleOCR / PaddleOCR-VL 1.6](https://github.com/PaddlePaddle/PaddleOCR)** (`PaddlePaddle`) - PP-OCRv4 Korean/Chinese recognition and PaddleOCR-VL-1.6 GGUF Vision-Language OCR models.
- **[LaMa](https://github.com/advimman/lama)** (`advimman/lama`) - Resolution-robust Large Mask Inpainting architecture.
- **[aot-inpainting](https://huggingface.co/mayocream/aot-inpainting)** (`mayocream/aot-inpainting`) - Fast AOT-GAN manga text-stroke inpainting model.
- **[RORem-mixed-GGUF](https://huggingface.co/mayocream/RORem-mixed-GGUF)** (`mayocream/RORem-mixed-GGUF`) - Generative manga text and object removal diffusion weights.
- **[FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)** (`black-forest-labs/FLUX.2-klein-4B`) - High-speed rectified-flow generative inpainting model.
