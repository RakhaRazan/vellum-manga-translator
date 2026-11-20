#pragma once

// MangaFlow AI — Native C++17 Multi-Language OCR Router (Phase 10.2)
//
// Routes speech bubble OCR per `source_lang` using ONNX Runtime C++ API (`Ort::Session`)
// and `stb_image.h` (zero OpenCV / Python dependency):
//   - `ja`: Japanese `manga-ocr` ViT encoder + BERT autoregressive decoder ONNX + C++ `vocab.txt` tokenizer
//   - `ko`: Korean PP-OCRv4 recognition + Multilingual PP-OCRv3 detection + orientation classifier ONNX
//   - `zh`: Chinese PP-OCRv4 recognition + PP-OCRv4 detection + orientation classifier ONNX
//
// Embedded character dictionaries are read directly from each PP-OCRv4 ONNX model's custom
// metadata (`character`), and all ONNX sessions are lazily loaded and immediately unloaded on
// completion (`auto_unload = true`) to minimize idle RAM/VRAM.

#include "model_manager.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mangaflow::pipeline {

struct OcrBubbleInput {
    std::string id;
    int order = 0;
    std::array<int, 4> bbox{0, 0, 0, 0};  // [x1, y1, x2, y2]
    std::vector<std::array<int, 2>> polygon;
    float confidence = 1.0f;
    std::string text_direction = "horizontal";  // "horizontal" | "vertical"
};

struct OcrBubbleOutput {
    std::string id;
    int order = 0;
    std::array<int, 4> bbox{0, 0, 0, 0};
    std::vector<std::array<int, 2>> polygon;
    std::string original_text;
    std::string source_text;
    float confidence = 1.0f;
    std::string text_direction = "horizontal";

    nlohmann::json to_json() const;
};

struct OcrOptions {
    std::string source_lang = "ja";
    std::string ocr_model = "auto";  // "auto" | "manga-ocr" | "paddleocr-vl-gguf" | "rapidocr-ppocrv4"
    std::string lmstudio_base_url = "http://localhost:1235/v1";
    std::string lmstudio_api_key = "lm-studio";
    bool auto_unload = true;
    bool force_cpu = false;
};

struct OcrStageResult {
    std::string source_lang;
    std::string engine;           // "manga-ocr" | "rapidocr" | "paddleocr-vl-gguf"
    std::string ocr_model;        // resolved canonical ID: "manga-ocr" | "rapidocr-ppocrv4" | "paddleocr-vl-gguf"
    std::string requested_model;  // requested option: "auto" | "manga-ocr" | "rapidocr-ppocrv4" | "paddleocr-vl-gguf"
    std::string execution_provider = "CPUExecutionProvider";
    std::vector<OcrBubbleOutput> blocks;
};

class OcrEngine {
public:
    explicit OcrEngine(const ModelManager& model_manager);

    // Normalize & resolve OCR model selection ("auto" | "manga-ocr" | "paddleocr-vl-gguf" | "rapidocr-ppocrv4")
    static std::string normalize_ocr_model(const std::string& raw);
    static std::string resolve_effective_ocr_model(
        const std::string& ocr_model,
        const std::string& source_lang);

    // Parse JSON array/object of `[x1, y1, x2, y2]` lists or bubble objects (matching `POST /api/stage/ocr`).
    static std::vector<OcrBubbleInput> parse_bubbles_json(const std::string& raw_json);

    // Decode raw image bytes (JPG/PNG/WebP/BMP) via stb_image and run OCR on all bubbles.
    OcrStageResult recognize_from_bytes(
        const uint8_t* image_bytes,
        size_t num_bytes,
        const std::vector<OcrBubbleInput>& bubbles,
        const OcrOptions& options = {});

    // Run OCR on an interleaved RGB uint8 page image `(height * width * 3)`.
    OcrStageResult recognize_rgb(
        const uint8_t* rgb_data,
        int width,
        int height,
        const std::vector<OcrBubbleInput>& bubbles,
        const OcrOptions& options = {});

    // Explicitly release all loaded ONNX OCR sessions and dictionaries.
    void unload();
    bool is_loaded() const noexcept {
        return manga_enc_session_.has_value() || manga_dec_session_.has_value() ||
               rapid_det_session_.has_value() || rapid_cls_session_.has_value() ||
               rapid_rec_session_.has_value();
    }

private:
    struct RgbImage {
        int width = 0;
        int height = 0;
        std::vector<uint8_t> data;  // RGB uint8
    };

    // Japanese manga-ocr ONNX encoder + decoder + vocab.txt
    void ensure_manga_ocr_loaded(bool force_cpu);
    std::string recognize_ja_crop(const RgbImage& raw_crop, const std::string& text_direction);
    std::string run_manga_ocr_single(const RgbImage& crop);
    std::string decode_manga_ocr_tokens(const std::vector<int64_t>& token_ids) const;
    static std::string postprocess_manga_ocr_text(const std::string& raw_utf8);

    // Korean / Chinese / CJK PP-OCRv4 ONNX (det + cls + rec)
    void ensure_rapid_ocr_loaded(const std::string& lang, bool force_cpu);
    std::string recognize_rapid_crop(const RgbImage& raw_crop);
    std::pair<std::string, float> run_ppocr_rec(const RgbImage& line_crop);
    RgbImage apply_ppocr_cls_if_needed(const RgbImage& line_crop);
    std::vector<std::array<int, 4>> run_ppocr_det(const RgbImage& crop);

    // PaddleOCR-VL-1.6-GGUF Vision-OCR via local LM Studio endpoint
    std::string resolve_paddleocr_vl_lmstudio_model(const OcrOptions& options) const;
    std::string recognize_paddleocr_vl_crop(
        const RgbImage& raw_crop,
        const std::string& model_id,
        const OcrOptions& options) const;

    // Pure C++ crop cleaning & multi-line strip segmentation
    static RgbImage crop_bbox_clean(
        const uint8_t* rgb_data,
        int img_w,
        int img_h,
        const std::array<int, 4>& bbox);
    static void erase_border_touching_ink(RgbImage& crop);
    static RgbImage pad_rgb_white(
        const RgbImage& src,
        int top,
        int bottom,
        int left,
        int right);
    static std::vector<RgbImage> extract_horizontal_line_strips(
        const RgbImage& crop,
        int min_line_height = 16);

    const ModelManager& model_manager_;

    // Lazy Japanese manga-ocr state
    std::optional<OrtSessionHandle> manga_enc_session_;
    std::optional<OrtSessionHandle> manga_dec_session_;
    std::vector<std::string> manga_vocab_;
    bool manga_force_cpu_ = false;

    // Lazy RapidOCR PP-OCRv4 state (for active `ko` or `cjk` profile)
    std::string rapid_loaded_lang_;
    std::optional<OrtSessionHandle> rapid_det_session_;
    std::optional<OrtSessionHandle> rapid_cls_session_;
    std::optional<OrtSessionHandle> rapid_rec_session_;
    std::vector<std::string> rapid_charset_;
    bool rapid_force_cpu_ = false;
};

}  // namespace mangaflow::pipeline
