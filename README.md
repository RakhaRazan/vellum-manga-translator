# Vellum v3.0.0

A 100% portable, local-first desktop studio for translating, cleaning, and typesetting Japanese manga (`JA`), Korean manhwa (`KO`), and Chinese manhua (`ZH`) on Windows.

Built with a pure modern **C++17** backend (**ONNX Runtime C++ API** + **Microsoft Edge WebView2**) and a **React 19 + Konva** canvas editor, `Vellum.exe` runs completely standalone with **zero Python runtime dependency** (no Python, PyTorch, CUDA toolkit, or Electron installation required).

---

## Download (Dual-Core Portable Releases)

Download the latest portable archive from [Releases] (https://github.com/HoldMyAss/vellum-manga-translator/releases), extract anywhere, and double-click `Vellum.exe`:

| Variant | Archive | Backends Included | Best For |
| :--- | :--- | :--- | :--- |
| **Universal Dual-Core** (Recommended) | `Vellum-v3.0.0-Windows-x64-Universal.zip` | **CUDA 12.9 + DirectML + CPU** | All Windows x64 PCs - automatically launches `Vellum_Core.exe` (CUDA) on NVIDIA RTX GPUs or `Vellum_Core_DML.exe` (DirectML) on AMD Radeon, Intel Arc/iGPU, GTX, or CPU-only desktops, with 1-click backend switching in Settings. |

- **Requirements:** Windows 10/11 (x64) with Microsoft Edge WebView2 Runtime (pre-installed on Windows 11).
- **100% Portable Relative Paths:** All runtime data resolves strictly relative to `Vellum.exe` (`./models/`, `./fonts/`, `./cache/`, `./projects_cache/`, and `./settings.json`)—zero `%LOCALAPPDATA%` or Registry clutter.
- **Dynamic VRAM Management:** Configures ONNX Runtime CUDA with `kSameAsRequested` allocation (`arena_extend_strategy = 1`) and live DXGI 1.4 VRAM monitoring so vision models coexist cleanly with local LLMs on 8 GB GPUs.
- On first launch, Vellum checks `./models/` and lets you download ONNX models directly from **Settings → Pipeline**.

---

## Key Capabilities (v3.0.0)

### 1. Provider-Agnostic LLM Translation, Refusal Guard & DPAPI Security
- **Zero-Lock-In Providers:** Works out of the box with **Cloud APIs** (`OpenAI`, `Google Gemini`, `Anthropic-Claude`, `DeepSeek`, `OpenRouter`), **Local LLMs** (`LM Studio` on `localhost:1234`, `Ollama` on default `localhost:11434`), **Custom OpenAI-Compatible** endpoints, or **`None` (Manual / No-LLM)** mode for pure CV/inpainting/typesetting workflows.
- **Windows DPAPI Secret Protection:** `active_provider` defaults to `"none"`. All non-empty API keys written to `settings.json` are encrypted at rest using Windows `CryptProtectData` (`"dpapi:..."`), while project saves (`.vellum.json`), crash-recovery autosaves, and `localStorage` strip all API keys.
- **LLM Refusal Guard & Auto-QC:** Automatically detects standard LLM refusal responses, flags affected bubbles in the `Issues` badge, and blocks refusal turns from polluting conversation history. Includes `<think>` tag stripping, `O(N)` JSON salvage recovery, custom prompts, Aho-Corasick + Okapi BM25 glossaries/memory, and optional multimodal page vision.

### 2. Full JA / KO / ZH Detection & Batched/Parallel OCR
- **Detector Engine v2 (`YOLO11m-Seg` + `ComicTextDetector`):** 9-class bubble, panel, and SFX segmentation with `SAHI+WBF` adaptive tiling for tall Webtoons, interior mask-inscribed `safe_bbox`, IoA panel association, and cycle-safe DAG topological reading order (`RTL` for JA, `LTR` for KO/ZH).
- **Multi-Engine OCR:** Supports `Manga OCR` (JA vertical tategaki & furigana), `RapidOCR PP-OCRv4` (KO/ZH), `Baberu OCR` (JA/ZH/EN), `Hayai OCR` (JA/ZH/KO/EN), and `PaddleOCR-VL 1.6` (local VLM), with vertical CJK/Hangul column re-packing, Perona-Malik parallel denoising, and CUDA/DirectML dynamic mini-batching.

### 3. 3-Tier + Tier 2.5 FFT + Poisson Inpainting
- **Tier 1 (Flat Median Fill) & Tier 2 (Pure-C++ Telea FMM):** Plain white and smooth-gradient speech bubbles are erased directly on CPU in milliseconds, skipping GPU inference entirely (`gpu_skipped = true`).
- **Tier 2.5 (FFT Screentone Detection) & Tier 3 (Neural LaMa / AOT / Diffusion + Poisson SOR):** Textured/screentone bubbles route to `LaMa Manga`, `AOT Inpainting`, `FLUX.2 Klein`, or `RORem Mixed`, followed by `4×3` Ridge-regularized affine color matching and Poisson SOR seam blending.
- **Text-Stroke-Only Masking:** Separates container `bubble_mask` (typesetting) from the `3px`-eroded interior text-stroke mask (inpainting) to preserve crisp bubble outlines.

### 4. 21 0-VRAM Math Models
- **21 Pure-C++/JS 0-VRAM Math Models:** Covers Sauvola + Otsu binarization, convexity-defect waist cutting, EDT watershed room polygons, RDP simplification, Perona-Malik denoising, SWT furigana masking, Hungarian block matching, Mahalanobis translation Auto-QC, and Knuth–Plass DP narrow-bubble line breaking.
- **Scanlator Studio Tools:**
  - **Restore-to-Raw Brush & Spot Clean:** Paint stray artifacts out via `/api/inpaint/region` or switch to **Restore Raw** to paint original pixels back over accidental overwrites at 0 ms latency with full Undo/Redo.
  - **1-Click `Fix Bubble`:** Runs single-block OCR → Region Inpaint → Translate in one click from the inspector sidebar.
  - **Inline `**bold**` & `*italic*` Runs:** Renders inline emphasis cleanly on the Konva canvas with two-pass outside strokes.
  - **Loupe (`Z`), Reading Preview (`F11`), Click-to-Order, Batch Series Queue (w/ Minimize-to-Pill), Token Telemetry, & Portable Backup (`.zip`).**

### 5. Layered `.psd` Export & 10,000-Page Endurance
- **Native Layered `.psd` Writer:** Exports editable Photoshop documents with `Background`, non-destructive `Cleanup` diff patch, and `Text` layers—including `luni` Unicode names, `TySh` (`EngineData` vector type), and `lfx2` (`FrFX` outline stroke) tagged blocks.
- **10,000-Page Volume I/O:** `O(1)`-RAM disk-streaming `.cbz` import (up to `25,000` pages / `32 GiB`), explicit **ZIP64 (`>4 GB`)** `.cbz` export, lossless `/DCTDecode` multi-page **PDF 1.4** export, and lean disk-backed crash-recovery autosave (`./projects_cache/autosave.vellum.json`).

---

## Supported Models

| Stage | Model | ID / Weights | Languages / Role | Execution Target |
| :--- | :--- | :--- | :--- | :--- |
| **Detection** | **Vellum YOLO11m-Seg (FP16)** | `vellum-det-yolo11m-seg-1024-fp16.onnx` | 9-class bubble, panel & SFX segmentation (`SAHI+WBF`, `safe_bbox`, IoA, DAG order) | GPU (`CUDA` / `DirectML`) |
| **Detection** | **Vellum YOLO11m-Seg (INT8)** | `vellum-det-yolo11m-seg-1024-int8.onnx` | 9-class static QDQ INT8 detector for CPU-only systems | CPU AVX2 |
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

## Acknowledgments / Model Credits

Vellum is built upon open-source computer vision, OCR, and inpainting model weights from the community:

- **[YOLO11m-Seg](https://huggingface.co/ogkalu)** (`ogkalu` / Ultralytics datasets) - Comic and manga speech-bubble, panel, and SFX segmentation research.
- **[comic-text-detector](https://github.com/dmMaze/comic-text-detector)** (`dmMaze`) - Manga text block and stroke mask detection weights.
- **[manga-ocr](https://github.com/kha-white/manga-ocr)** (`kha-white`) - Japanese vertical/horizontal manga OCR model.
- **[RapidOCR](https://github.com/RapidAI/RapidOCR)** & **[PaddleOCR / PaddleOCR-VL 1.6](https://github.com/PaddlePaddle/PaddleOCR)** (`PaddlePaddle`) - PP-OCRv4 Korean/Chinese recognition and PaddleOCR-VL-1.6 GGUF Vision-Language OCR models.
- **[baberu-ocr](https://huggingface.co/genshiai-daichi/baberu-ocr)** (`genshiai-daichi/baberu-ocr`) - Lightweight quantized `JA`/`ZH`/`EN` and SFX manga OCR weights.
- **[hayai-ocr-v2](https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2)** (`JustANormalTinkerer/hayai-ocr-v2`) - Aspect-ratio-preserving multilingual manga and webtoon OCR architecture.
- **[LaMa](https://github.com/advimman/lama)** (`advimman/lama` & `lama-manga`) - Resolution-robust Large Mask Inpainting architecture.
- **[aot-inpainting](https://huggingface.co/mayocream/aot-inpainting)** (`mayocream/aot-inpainting`) - Fast AOT-GAN manga text-stroke inpainting model.
- **[RORem-mixed-GGUF](https://huggingface.co/mayocream/RORem-mixed-GGUF)** (`mayocream/RORem-mixed-GGUF`) - Generative manga text and object removal diffusion weights.
- **[FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)** (`black-forest-labs/FLUX.2-klein-4B`) - High-speed rectified-flow generative inpainting model.
