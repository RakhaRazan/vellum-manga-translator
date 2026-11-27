#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace vellum::detector {

/// 9-class layout and text taxonomy for Detector Engine v2.
enum class ClassId : int {
    SpeechBubble       = 0,
    ThoughtBubble      = 1,
    NarrationBox       = 2,
    ShoutBubble        = 3,
    WhisperBubble      = 4,
    BorderlessDialogue = 5,
    TextRegion         = 6,
    Panel              = 7,
    SfxIgnore          = 8,
};

inline constexpr int kNumClasses = 9;

inline const char* class_id_to_string(ClassId id) noexcept {
    switch (id) {
        case ClassId::SpeechBubble:       return "SpeechBubble";
        case ClassId::ThoughtBubble:      return "ThoughtBubble";
        case ClassId::NarrationBox:       return "NarrationBox";
        case ClassId::ShoutBubble:        return "ShoutBubble";
        case ClassId::WhisperBubble:      return "WhisperBubble";
        case ClassId::BorderlessDialogue: return "BorderlessDialogue";
        case ClassId::TextRegion:         return "TextRegion";
        case ClassId::Panel:              return "Panel";
        case ClassId::SfxIgnore:          return "SfxIgnore";
        default:                          return "Unknown";
    }
}

inline bool is_dialogue_class(ClassId id) noexcept {
    const int v = static_cast<int>(id);
    return v >= static_cast<int>(ClassId::SpeechBubble) &&
           v <= static_cast<int>(ClassId::BorderlessDialogue);
}

enum class ReadingDirection {
    RTL,  // Japanese manga (right-to-left)
    LTR   // Korean manhwa / Chinese manhua (left-to-right)
};

enum class BackendType {
    Auto,
    Yolo11Seg,
    RfDetrSeg,
    LegacyCtd
};

enum class DetectorPrecision {
    Fast,           // 1x YOLO pass + always-active zero-cost upgrades
    Balanced,       // + Dual-Threshold Rescue + Selective Polarity-Invert TTA
    HighPrecision   // + Horizontal Flip TTA + Panel-Guided Zoom + Optional DBNet Borderless Rescue
};

inline const char* detector_precision_to_string(DetectorPrecision p) noexcept {
    switch (p) {
        case DetectorPrecision::Fast:          return "fast";
        case DetectorPrecision::Balanced:      return "balanced";
        case DetectorPrecision::HighPrecision: return "high_precision";
        default:                               return "balanced";
    }
}

inline DetectorPrecision parse_detector_precision(const std::string& raw) noexcept {
    std::string s;
    s.reserve(raw.size());
    for (char ch : raw) {
        if (ch >= 'A' && ch <= 'Z') {
            s.push_back(static_cast<char>(ch - 'A' + 'a'));
        } else if (ch == '-') {
            s.push_back('_');
        } else if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n') {
            s.push_back(ch);
        }
    }
    if (s == "fast") {
        return DetectorPrecision::Fast;
    }
    if (s == "high_precision" || s == "high" || s == "max" || s == "accurate") {
        return DetectorPrecision::HighPrecision;
    }
    return DetectorPrecision::Balanced;
}

struct Point2f {
    float x = 0.0f;
    float y = 0.0f;
};

struct BBox {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;

    float width() const noexcept { return std::max(0.0f, x2 - x1); }
    float height() const noexcept { return std::max(0.0f, y2 - y1); }
    float area() const noexcept { return width() * height(); }
    float cx() const noexcept { return 0.5f * (x1 + x2); }
    float cy() const noexcept { return 0.5f * (y1 + y2); }
    bool is_valid() const noexcept { return x2 > x1 && y2 > y1; }
};

struct InstanceMask {
    std::vector<uint8_t> data;  // Local cropped 8-bit binary mask (0 or 255), row-major (width * height)
    int width = 0;
    int height = 0;
    int offset_x = 0;           // Top-left X in full image coordinates
    int offset_y = 0;           // Top-left Y in full image coordinates
    std::vector<Point2f> polygon;  // Simplified polygon in full image coordinates
    double area_px = 0.0;
    std::string mask_file;

    bool empty() const noexcept {
        return data.empty() || width <= 0 || height <= 0;
    }
};

struct PanelDetection {
    int id = 0;
    int reading_order = 0;
    float confidence = 0.0f;
    BBox bbox;
    BBox bbox_norm;
};

struct BubbleOutlineInfo {
    std::array<uint8_t, 3> color{255, 255, 255};
    int width = 0;
};

struct BubbleDetection {
    int id = 0;
    int reading_order = 0;
    int panel_id = -1;          // -1 = unassociated
    int speaker_cluster_id = -1; // -1 = unclustered / non-dialogue
    bool is_cross_panel = false;
    ClassId class_id = ClassId::SpeechBubble;
    std::string class_name;
    float confidence = 0.0f;
    BBox bbox;
    BBox bbox_norm;
    BBox safe_bbox;
    InstanceMask mask;
    std::string crop_file;
    // Optional v2.5 CV geometry & typography metadata
    std::vector<Point2f> room_polygon;
    std::optional<float> angle_deg;
    std::optional<float> estimated_font_size;
    std::optional<std::array<uint8_t, 3>> text_color;
    std::optional<BubbleOutlineInfo> outline;
};

struct DetectorConfig {
    std::array<float, kNumClasses> class_thresholds{
        0.42f,  // 0: SpeechBubble
        0.35f,  // 1: ThoughtBubble
        0.40f,  // 2: NarrationBox
        0.35f,  // 3: ShoutBubble
        0.25f,  // 4: WhisperBubble
        0.28f,  // 5: BorderlessDialogue
        0.35f,  // 6: TextRegion
        0.45f,  // 7: Panel
        0.35f   // 8: SfxIgnore
    };
    int max_detections = 150;
    float nms_iou_threshold = 0.55f;
    float sfx_suppress_iou = 0.60f;
    bool enable_adaptive_tiling = true;
    float tile_aspect_trigger = 2.5f;
    int tile_size = 1024;
    float tile_overlap = 0.20f;
    bool keep_session_alive = true;
    std::string export_dir;

    ReadingDirection reading_direction = ReadingDirection::RTL;
    BackendType backend = BackendType::Auto;
    std::string model_path;
    bool force_cpu = false;
    bool enable_cv_style_detect = true;
    bool enable_waist_cut = true;
    bool auto_deskew_scans = false;
    DetectorPrecision precision = DetectorPrecision::Balanced;
    bool is_subpass = false;
};

struct RuntimeMeta {
    BackendType backend = BackendType::Auto;
    std::string model_name;
    std::string execution_provider;
    double inference_ms = 0.0;
    double postproc_ms = 0.0;
    int tiles_processed = 1;
};

struct PageDetectionResult {
    std::string schema_version = "2.0";
    std::string filename;
    int image_width = 0;
    int image_height = 0;
    ReadingDirection reading_direction = ReadingDirection::RTL;
    RuntimeMeta runtime_meta;
    std::vector<PanelDetection> panels;
    std::vector<BubbleDetection> detections;
};

}  // namespace vellum::detector

namespace vellum {
namespace det = detector;
}  // namespace vellum

