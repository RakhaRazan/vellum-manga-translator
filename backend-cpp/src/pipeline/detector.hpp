#pragma once

// Vellum — Native C++17 Text & Layout Detector (Phase 10.1)
//
// Supports:
//   1. Primary : Koharu RF-DETR-Seg-2XL (`koharu-layout-rfdetr-seg-2xl-1152.onnx`)
//   2. Fallback: Comic Text Detector (`comictextdetector.pt.onnx`)
//
// Uses ONNX Runtime C++ API (`onnxruntime_cxx_api.h`) with automatic CUDA/DirectML -> CPU
// fallback and single-header `stb_image.h` / `stb_image_write.h` (zero OpenCV dependency).
// Outputs reading-order sorted speech bubbles (`rtl` for `ja`, `ltr` for `ko`/`zh`) and a
// binary `(H, W)` uint8 text segmentation mask (`0` or `255`) ready for LaMa inpainting.

#include "model_manager.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace mangaflow::pipeline {

struct DetectedBubble {
    std::string id;
    int order = 0;
    std::array<int, 4> bbox{0, 0, 0, 0};         // [x1, y1, x2, y2] tight text bbox
    std::array<int, 4> bubble_bbox{0, 0, 0, 0};  // [x1, y1, x2, y2] enclosing speech bubble / traced interior
    std::array<int, 4> safe_bbox{0, 0, 0, 0};    // [x1, y1, x2, y2] 8% H / 8% V safe inset inside bubble_bbox
    std::array<std::array<int, 2>, 4> polygon{};
    float confidence = 0.0f;
    std::string text_direction = "horizontal";  // "horizontal" | "vertical"
    std::string category = "text";              // "text" | "onomatopoeia"

    nlohmann::json to_json() const;
};

struct LayoutRegion {
    std::string id;
    int order = 0;
    std::array<int, 4> bbox{0, 0, 0, 0};  // [x1, y1, x2, y2]
    float confidence = 0.0f;
    std::string category;  // "bubble" | "panel"

    nlohmann::json to_json() const;
};

struct AppliedDetectionThresholds {
    float text = 0.25f;
    float onomatopoeia = 0.25f;
    std::optional<float> bubble;
    std::optional<float> panel;
    float nms = 0.40f;
    float mask = 0.50f;

    nlohmann::json to_json() const;
};

struct DetectionOptions {
    std::string source_language = "ja";
    std::string reading_order = "auto";         // "auto" | "rtl" | "ltr"
    std::string detection_model;                // "koharu-layout-rfdetr-seg-2xl" | "comic-text-detector"
    std::optional<float> text_threshold;        // nullopt = model default; overrides class 0/1 & CTD conf
    std::optional<float> bubble_threshold;      // nullopt = model default (0.50); overrides RF-DETR class 2
    std::optional<float> panel_threshold;       // nullopt = model default (0.50); overrides RF-DETR class 3
    std::optional<float> conf_threshold;        // explicit per-request legacy override
    float ctd_default_conf_threshold = 0.35f;   // settings.conf_threshold used by CTD when text_threshold is null
    std::optional<float> nms_threshold;
    std::optional<float> mask_threshold;
    int box_padding = 6;
    int dilate_kernel_size = 5;
    bool restrict_mask_to_bboxes = true;
    bool auto_unload = true;
    bool force_cpu = false;
    bool allow_model_fallback = true;
    std::string detector_backend = "auto";      // legacy alias: "auto" | "rfdetr" | "ctd"
};

struct DetectionResult {
    int width = 0;
    int height = 0;
    std::string execution_provider = "CPUExecutionProvider";
    DetectorBackend backend = DetectorBackend::KoharuRfDetr;
    std::string model_id = "koharu-layout-rfdetr-seg-2xl";
    std::string model_path;
    std::string reading_order = "rtl";
    AppliedDetectionThresholds applied_thresholds;
    std::vector<DetectedBubble> bubbles;
    std::vector<LayoutRegion> speech_bubbles;
    std::vector<LayoutRegion> panels;
    std::vector<uint8_t> mask;  // Row-major (height * width), values 0 or 255
};

class TextDetector {
public:
    explicit TextDetector(const ModelManager& model_manager);

    // Decode raw image bytes (JPG/PNG/WebP/BMP via stb_image) and run detection.
    DetectionResult detect_from_bytes(
        const uint8_t* image_bytes,
        size_t num_bytes,
        const DetectionOptions& options = {});

    // Load an image file from disk and run detection.
    DetectionResult detect_from_file(
        const std::filesystem::path& image_path,
        const DetectionOptions& options = {});

    // Run detection directly on an interleaved RGB uint8 buffer (height * width * 3).
    DetectionResult detect_rgb(
        const uint8_t* rgb_data,
        int width,
        int height,
        const DetectionOptions& options = {});

    // Encode a binary `(height * width)` mask as PNG bytes (1-channel grayscale).
    static std::vector<uint8_t> encode_mask_png(
        const std::vector<uint8_t>& mask,
        int width,
        int height);

    // Base64-encode raw bytes (helper for optional `include_mask_base64=true`).
    static std::string base64_encode(const uint8_t* data, size_t size);
    static std::string mask_to_data_uri(
        const std::vector<uint8_t>& mask,
        int width,
        int height);

    // Normalize comic language code (`ja` / `ko` / `zh`); throws std::invalid_argument on unknown language.
    static std::string normalize_source_language(const std::string& raw);

    // Explicitly unload any cached ONNX session from RAM/VRAM.
    void unload();
    bool is_loaded() const noexcept { return cached_session_.has_value(); }

private:
    struct RawBox {
        std::array<int, 4> bbox{0, 0, 0, 0};
        float score = 0.0f;
        std::string direction = "horizontal";
        std::string category = "text";
        int source_query_index = -1;  // Used by RF-DETR to map back to instance mask slice
    };

    OrtSessionHandle& ensure_session(
        const ResolvedDetectorModel& resolved,
        bool force_cpu);

    DetectionResult run_rfdetr(
        const uint8_t* rgb_data,
        int orig_w,
        int orig_h,
        const ResolvedDetectorModel& resolved,
        const DetectionOptions& options);

    DetectionResult run_comic_text_detector(
        const uint8_t* rgb_data,
        int orig_w,
        int orig_h,
        const ResolvedDetectorModel& resolved,
        const DetectionOptions& options);

    static std::vector<size_t> nms_xyxy(
        const std::vector<std::array<float, 4>>& boxes,
        const std::vector<float>& scores,
        float iou_threshold);

    static std::vector<DetectedBubble> sort_reading_order(
        const std::vector<RawBox>& raw_boxes,
        const std::string& source_language,
        int page_height,
        std::vector<int>* sorted_query_indices_out = nullptr,
        const std::string& reading_order = "auto",
        const std::vector<LayoutRegion>* panels = nullptr);

    static std::vector<LayoutRegion> sort_layout_regions(
        const std::vector<std::pair<std::array<int, 4>, float>>& raw_regions,
        const std::string& category,
        const std::string& id_prefix,
        bool rtl,
        int page_height);

    static void dilate_binary_mask_ellipse(
        std::vector<uint8_t>& mask,
        int width,
        int height,
        int kernel_size);

    static void populate_bubble_and_safe_bboxes(
        std::vector<DetectedBubble>& bubbles,
        const uint8_t* rgb_data,
        int orig_w,
        int orig_h,
        const std::vector<LayoutRegion>* speech_bubbles,
        const std::vector<LayoutRegion>* panels,
        const std::vector<uint8_t>* text_mask = nullptr);

    const ModelManager& model_manager_;
    std::optional<OrtSessionHandle> cached_session_;
    std::filesystem::path cached_model_path_;
    bool cached_force_cpu_ = false;
};

}  // namespace mangaflow::pipeline
