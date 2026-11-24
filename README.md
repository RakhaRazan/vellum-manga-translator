# Vellum v2.5 Studio Core

A high-performance, 100% portable local desktop studio for translating, cleaning, and typesetting Japanese manga, Korean manhwa, and Chinese manhua on Windows.

Built with a pure modern **C++17** backend (**ONNX Runtime C++ API** + **Microsoft Edge WebView2**) and a **React + Konva** canvas editor, `Vellum.exe` runs completely standalone with **zero Python runtime dependency** (no Python, PyTorch, CUDA toolkit, or Electron installation required).

---

## Download (Unified Dual-Core Portable Release)

Download the latest portable archive from [Releases](https://github.com/HoldMyAss/vellum-manga-translator/releases), extract anywhere, and double-click `Vellum.exe`:

| Variant | Archive | Backends Included | Best For |
| :--- | :--- | :--- | :--- |
| **Universal Dual-Core** (Recommended) | `Vellum-v2.5-Windows-x64-Universal.zip` | **CUDA 12.9 + DirectML + CPU** | All Windows x64 PCs — automatically launches `Vellum_Core.exe` (CUDA) on NVIDIA RTX GPUs or `Vellum_Core_DML.exe` (DirectML) on AMD Radeon, Intel Arc/iGPU, GTX, or CPU-only desktops, with 1-click backend switching in Settings. |

- **Requirements:** Windows 10/11 (x64) with Microsoft Edge WebView2 Runtime (pre-installed on Windows 11).
- **100% Portable Relative Paths:** All runtime data resolves strictly relative to `Vellum.exe` (`./models/`, `./fonts/`, `./cache/`, `./projects_cache/`, and `./settings.json`)—zero `%LOCALAPPDATA%` or Registry clutter.
- **Dynamic VRAM Management:** Configures ONNX Runtime CUDA with `kSameAsRequested` allocation (`arena_extend_strategy = 1`) and live DXGI 1.4 VRAM monitoring so vision models coexist cleanly with local LLMs on 8 GB GPUs.
- On first launch, Vellum checks `./models/` and lets you download ONNX models directly from **Settings → Pipeline** (or import existing local `.onnx` / Hugging Face model weights).

---

## Key Capabilities (v2.5 Studio Core)

### 1. Provider-Agnostic LLM Translation & DPAPI Security
- **Zero-Lock-In Providers:** Works out of the box with **Cloud APIs** (`OpenAI`, `Google Gemini`, `Anthropic Claude`, `DeepSeek`, `OpenRouter`), **Local LLMs** (`LM Studio` on `localhost:1234`, `Ollama` on `localhost:11434`), **Custom OpenAI-Compatible** endpoints, or **`None` (Manual / No-LLM)** mode for pure CV/inpainting/typesetting workflows.
- **Windows DPAPI Secret Protection:** `active_provider` defaults to `"none"`. All non-empty API keys written to `settings.json` are encrypted at rest using Windows `CryptProtectData` (`"dpapi:..."`), while project saves (`.vellum.json`), crash-recovery autosaves, and `localStorage` strip all API keys.
- Includes `<think>` tag stripping, `O(N)` JSON salvage recovery, custom prompts, glossaries, uncensored localization tones, and optional multimodal page vision.

### 2. 0-VRAM Classic CV Typography & Layout Geometry
- **Classic CV Style Detection:** Pure-C++ Otsu + Sauvola adaptive binarization, stroke-width estimation, vertical/horizontal text orientation, dominant ink color, outline detection, and skew angle (`angle_deg`, `estimated_font_size`, `text_color`, `outline`) running in **~7 ms/page on CPU** with **+0 MB VRAM**.
- **Conjoined Balloon Waist-Cutting & RDP `room_polygon`:** Separates merged speech balloons via convexity-defect waist chords and squared-Euclidean EDT watershed partitioning, simplified via Ramer–Douglas–Peucker (`<= 32` vertices) for lean project serialization.
- **Smart Hybrid Hardware Routing (`CUDA → DirectML → CPU`):** Routes 2D vision models (Detector & Inpainter) to GPU while keeping autoregressive/lightweight OCR (`Manga OCR`, `Baberu OCR`, `RapidOCR`) on CPU AVX2.

### 3. 3-Tier CPU-First Inpainting + Tone Match
- **Tier 1 (Flat Median Fill) & Tier 2 (Pure-C++ Telea FMM):** Plain white and smooth-gradient speech bubbles are erased directly on CPU in milliseconds, skipping LaMa GPU inference entirely (`gpu_skipped = true` when all bubbles resolve in Tier 1–2).
- **Tier 3 (Neural LaMa / AOT / Diffusion + Tone Match):** Textured/screentone bubbles route to `LaMa Manga`, `AOT Inpainting`, `FLUX.2 Klein`, or `RORem Mixed`, followed by `4×3` Ridge-regularized affine color matching and separable Gaussian residual diffusion (`σ = 24 px`) to eliminate tile seams.
- **Text-Stroke-Only Masking:** Separates the container `bubble_mask` (typesetting) from the `3px`-eroded interior text-stroke mask (inpainting) to preserve crisp bubble outlines.

### 4. Narrow-Bubble DP Typesetter & Spot-Remove Brush
- **Dynamic Programming Line-Breaker:** Shapes translated dialogue inside narrow or tapered `room_polygon` contours with Knuth–Plass-style penalty optimization, automatic hyphenation, and 2D resolution-proportional auto-fit.
- **Spot-Remove Brush & Interactive Canvas:** 60fps `react-konva` editor with a non-destructive spot-remove brush, Split-Curtain Before/After compare, Windows & Google Fonts manager with manga symbol fallbacks (`♡ ♪ …`), and virtualized filmstrip.

### 5. Layered `.psd` Export & 10,000-Page Endurance
- **Native Layered `.psd` Writer:** Exports editable Photoshop documents with `Background`, non-destructive `Cleanup` diff patch, and `Text` layers—including `luni` Unicode names, `TySh` (`EngineData` vector type), and `lfx2` (`FrFX` outline stroke) tagged blocks.
- **10,000-Page Volume I/O:** `O(1)`-RAM disk-streaming `.cbz` import (up to `25,000` pages / `32 GiB`), explicit **ZIP64 (`>4 GB`)** `.cbz` export, lossless `/DCTDecode` multi-page **PDF 1.4** export, and lean disk-backed crash-recovery autosave (`./projects_cache/autosave.vellum.json`).

---

## Supported Models

| Stage | Model | ID / Weights | Languages / Role | Execution Target |
| :--- | :--- | :--- | :--- | :--- |
| **Detection** | **Vellum YOLO11m-Seg (FP16)** | `vellum-det-yolo11m-seg-1024-fp16.onnx` | 9-class bubble, panel & SFX segmentation (`SAHI+WBF`, `safe_bbox`, IoA, DAG order) | GPU (`CUDA` / `DirectML`) |
| **Detection** | **Vellum YOLO11m-Seg (INT8)** | `vellum-det-yolo11m-seg-1024-int8.onnx` | 9-class static QDQ INT8 detector for CPU-only systems | CPU AVX2 |
| **Detection** | **RF-DETR-Seg 2XL (High-Context)** | `koharu-layout-rfdetr-seg-2xl-1152.onnx` | 1152px high-context layout, panel, bubble & SFX segmentation | CUDA / Heavy CPU |
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

---

## Building from Source

### Prerequisites
- Windows 10/11 (x64)
- Visual Studio 2022 (or Build Tools) with **Desktop development with C++** and CMake
- Node.js 18+
- *(Optional, for CUDA variant)* NVIDIA CUDA Toolkit 12.x + cuDNN 9.x

All C++ dependencies (`cpp-httplib`, `nlohmann/json`, `webview`, WebView2, ONNX Runtime, `miniz`, `stb`) are fetched automatically by CMake. No `vcpkg` required.

### Dual-Core Release Build
```bat
REM Build Unified Dual-Core (CUDA + DirectML + CPU) distribution:
packaging\build_cpp_release.bat AUTO

REM Build CUDA-only or DirectML-only distribution:
packaging\build_cpp_release.bat CUDA
packaging\build_cpp_release.bat DML
```

---

## Credits & Acknowledgments

Vellum stands on the shoulders of the open-source manga translation, OCR, and computer vision community:

- **[Koharu](https://github.com/mayocream/koharu)** (`mayocream`) — Inspiration for the modern local-first manga translation workflow, `koharu-layout-rfdetr-seg-2xl-1152`, `lama-manga`, `aot-inpainting`, and `RORem-mixed-GGUF` weights.
- **[ogkalu](https://huggingface.co/ogkalu)** — Comic and manga speech-bubble / layout detection datasets and YOLO segmentation research.
- **[comic-text-detector](https://github.com/dmMaze/comic-text-detector)** (`dmMaze`) — Foundational manga text block and stroke mask detection architecture.
- **[manga-ocr](https://github.com/kha-white/manga-ocr)** (`kha-white`) — Japanese vertical/horizontal manga OCR model.
- **[baberu-ocr](https://huggingface.co/genshiai-daichi/baberu-ocr)** (`genshiai-daichi/baberu-ocr`) — Lightweight quantized `JA`/`ZH`/`EN` and SFX manga OCR weights.
- **[hayai-ocr-v2](https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2)** (`JustANormalTinkerer/hayai-ocr-v2`) — Aspect-ratio-preserving multilingual manga and webtoon OCR architecture.
- **[RapidOCR](https://github.com/RapidAI/RapidOCR)** & **[PaddleOCR / PaddleOCR-VL 1.6](https://github.com/PaddlePaddle/PaddleOCR)** (`PaddlePaddle`) — PP-OCRv4 Korean/Chinese recognition and PaddleOCR-VL-1.6 GGUF Vision-Language OCR models.
- **[LaMa](https://github.com/advimman/lama)** (`advimman/lama`) — Resolution-robust Large Mask Inpainting architecture.
- **[aot-inpainting](https://huggingface.co/mayocream/aot-inpainting)** (`mayocream/aot-inpainting`) — Fast AOT-GAN manga text-stroke inpainting model.
- **[RORem-mixed-GGUF](https://huggingface.co/mayocream/RORem-mixed-GGUF)** (`mayocream/RORem-mixed-GGUF`) — Generative manga text and object removal diffusion weights.
- **[FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)** (`black-forest-labs/FLUX.2-klein-4B`) — High-speed rectified-flow generative inpainting model.
