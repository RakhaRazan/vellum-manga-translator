#include "rfdetr_seg_backend.hpp"
#include "geometry_postproc.hpp"

#include "../pipeline/detector.hpp"
#include "../pipeline/model_manager.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <optional>
#include <stdexcept>

namespace vellum::detector {

namespace {

InstanceMask cropMaskFromPage(
    const std::vector<uint8_t>& page_mask,
    int page_w,
    int page_h,
    const BBox& bbox) {
    InstanceMask out;
    if (page_mask.size() < static_cast<size_t>(page_w) * static_cast<size_t>(page_h) ||
        !bbox.is_valid()) {
        return out;
    }

    const int x1 = std::clamp(static_cast<int>(std::floor(bbox.x1)), 0, page_w - 1);
    const int y1 = std::clamp(static_cast<int>(std::floor(bbox.y1)), 0, page_h - 1);
    const int x2 = std::clamp(static_cast<int>(std::ceil(bbox.x2)), x1 + 1, page_w);
    const int y2 = std::clamp(static_cast<int>(std::ceil(bbox.y2)), y1 + 1, page_h);

    out.offset_x = x1;
    out.offset_y = y1;
    out.width = x2 - x1;
    out.height = y2 - y1;
    out.data.assign(static_cast<size_t>(out.width) * static_cast<size_t>(out.height), 0);

    double area = 0.0;
    for (int y = 0; y < out.height; ++y) {
        const size_t src_row = static_cast<size_t>(y1 + y) * static_cast<size_t>(page_w);
        const size_t dst_row = static_cast<size_t>(y) * static_cast<size_t>(out.width);
        for (int x = 0; x < out.width; ++x) {
            const uint8_t val = page_mask[src_row + static_cast<size_t>(x1 + x)];
            if (val > 0) {
                out.data[dst_row + static_cast<size_t>(x)] = 255;
                area += 1.0;
            }
        }
    }
    out.area_px = area;
    return out;
}

}  // namespace

RfDetrSegBackend::RfDetrSegBackend() = default;
RfDetrSegBackend::~RfDetrSegBackend() = default;

bool RfDetrSegBackend::load(const DetectorConfig& config) {
    std::optional<std::filesystem::path> models_dir_opt;
    if (!config.model_path.empty()) {
        const auto p = std::filesystem::u8path(config.model_path);
        if (std::filesystem::is_directory(p)) {
            models_dir_opt = p;
        } else if (p.has_parent_path()) {
            models_dir_opt = p.parent_path();
        }
    }

    model_manager_ = std::make_unique<mangaflow::pipeline::ModelManager>(models_dir_opt);
    text_detector_ = std::make_unique<mangaflow::pipeline::TextDetector>(*model_manager_);
    active_provider_ = mangaflow::pipeline::ModelManager::get_primary_provider(config.force_cpu);
    model_name_ = (config.backend == BackendType::LegacyCtd)
        ? "comic-text-detector"
        : "koharu-layout-rfdetr-seg-2xl";
    loaded_ = true;
    return true;
}

PageDetectionResult RfDetrSegBackend::detectPage(
    const uint8_t* rgbData,
    int width,
    int height,
    const std::string& filename,
    const DetectorConfig& config) {
    if (!rgbData || width <= 0 || height <= 0) {
        throw std::invalid_argument("RfDetrSegBackend::detectPage: invalid RGB buffer or dimensions.");
    }
    if (!loaded_ && !load(config)) {
        throw std::runtime_error("RfDetrSegBackend::detectPage: failed to initialize ModelManager.");
    }

    const auto t_start = std::chrono::steady_clock::now();

    const bool is_ctd = (config.backend == BackendType::LegacyCtd);
    mangaflow::pipeline::DetectionOptions opts;
    opts.detection_model = is_ctd ? "comic-text-detector" : "koharu-layout-rfdetr-seg-2xl";
    opts.detector_backend = is_ctd ? "ctd" : "rfdetr";
    opts.source_language = (config.reading_direction == ReadingDirection::RTL) ? "ja" : "ko";
    opts.reading_order = (config.reading_direction == ReadingDirection::RTL) ? "rtl" : "ltr";
    opts.force_cpu = config.force_cpu;
    opts.auto_unload = !config.keep_session_alive;
    opts.allow_model_fallback = !is_ctd;
    opts.nms_threshold = config.nms_iou_threshold;
    opts.conf_threshold = config.class_thresholds[static_cast<size_t>(ClassId::SpeechBubble)];
    opts.text_threshold = config.class_thresholds[static_cast<size_t>(ClassId::SpeechBubble)];
    opts.bubble_threshold = config.class_thresholds[static_cast<size_t>(ClassId::SpeechBubble)];
    opts.panel_threshold = config.class_thresholds[static_cast<size_t>(ClassId::Panel)];

    const mangaflow::pipeline::DetectionResult legacy =
        text_detector_->detect_rgb(rgbData, width, height, opts);

    const auto t_infer = std::chrono::steady_clock::now();

    active_provider_ = legacy.execution_provider;
    model_name_ = legacy.model_id;

    PageDetectionResult result;
    result.schema_version = "2.0";
    result.filename = filename;
    result.image_width = width;
    result.image_height = height;
    result.reading_direction = config.reading_direction;
    result.runtime_meta.backend = is_ctd ? BackendType::LegacyCtd : BackendType::RfDetrSeg;
    result.runtime_meta.model_name = model_name_;
    result.runtime_meta.execution_provider = active_provider_;
    result.runtime_meta.tiles_processed = 1;

    // Map panels -> ClassId::Panel (7)
    int next_panel_id = 0;
    for (const auto& p : legacy.panels) {
        BBox box{
            static_cast<float>(p.bbox[0]),
            static_cast<float>(p.bbox[1]),
            static_cast<float>(p.bbox[2]),
            static_cast<float>(p.bbox[3])
        };
        if (!box.is_valid()) continue;

        PanelDetection pd;
        pd.id = next_panel_id++;
        pd.reading_order = p.order;
        pd.confidence = p.confidence;
        pd.bbox = box;
        pd.bbox_norm = BBox{
            box.x1 / static_cast<float>(width),
            box.y1 / static_cast<float>(height),
            box.x2 / static_cast<float>(width),
            box.y2 / static_cast<float>(height)
        };
        result.panels.push_back(std::move(pd));
    }

    // Map detected bubbles & text regions into 9-class taxonomy
    int next_det_id = 0;
    for (const auto& b : legacy.bubbles) {
        const bool is_sfx = (b.category == "onomatopoeia");
        const auto& src_box = is_sfx ? b.bbox : b.bubble_bbox;

        BBox box{
            static_cast<float>(src_box[0]),
            static_cast<float>(src_box[1]),
            static_cast<float>(src_box[2]),
            static_cast<float>(src_box[3])
        };
        if (!box.is_valid()) {
            box = BBox{
                static_cast<float>(b.bbox[0]),
                static_cast<float>(b.bbox[1]),
                static_cast<float>(b.bbox[2]),
                static_cast<float>(b.bbox[3])
            };
        }
        if (!box.is_valid()) continue;

        ClassId cls = ClassId::SpeechBubble;
        if (is_sfx) {
            cls = ClassId::SfxIgnore;
        } else {
            // Check if enclosed in a detected speech bubble region
            bool has_bubble_region = false;
            const BBox text_box{
                static_cast<float>(b.bbox[0]),
                static_cast<float>(b.bbox[1]),
                static_cast<float>(b.bbox[2]),
                static_cast<float>(b.bbox[3])
            };
            for (const auto& sb : legacy.speech_bubbles) {
                const BBox sb_box{
                    static_cast<float>(sb.bbox[0]),
                    static_cast<float>(sb.bbox[1]),
                    static_cast<float>(sb.bbox[2]),
                    static_cast<float>(sb.bbox[3])
                };
                if (computeIoA(text_box, sb_box) >= 0.50f) {
                    has_bubble_region = true;
                    break;
                }
            }
            cls = has_bubble_region ? ClassId::SpeechBubble : ClassId::BorderlessDialogue;
        }

        BubbleDetection det;
        det.id = next_det_id++;
        det.reading_order = b.order;
        det.class_id = cls;
        det.class_name = class_id_to_string(cls);
        det.confidence = b.confidence;
        det.bbox = box;
        det.bbox_norm = BBox{
            box.x1 / static_cast<float>(width),
            box.y1 / static_cast<float>(height),
            box.x2 / static_cast<float>(width),
            box.y2 / static_cast<float>(height)
        };
        det.safe_bbox = BBox{
            static_cast<float>(b.safe_bbox[0]),
            static_cast<float>(b.safe_bbox[1]),
            static_cast<float>(b.safe_bbox[2]),
            static_cast<float>(b.safe_bbox[3])
        };
        det.mask = cropMaskFromPage(legacy.mask, width, height, box);
        det.mask.polygon.reserve(4);
        for (const auto& pt : b.polygon) {
            det.mask.polygon.push_back(Point2f{
                static_cast<float>(pt[0]),
                static_cast<float>(pt[1])
            });
        }

        result.detections.push_back(std::move(det));
    }

    runGeometryPostproc(result, config);

    const auto t_end = std::chrono::steady_clock::now();
    result.runtime_meta.inference_ms =
        std::chrono::duration<double, std::milli>(t_infer - t_start).count();
    result.runtime_meta.postproc_ms =
        std::chrono::duration<double, std::milli>(t_end - t_infer).count();

    return result;
}

void RfDetrSegBackend::unload() {
    if (text_detector_) {
        text_detector_->unload();
    }
    loaded_ = false;
}

std::string RfDetrSegBackend::activeModelName() const {
    return model_name_;
}

std::string RfDetrSegBackend::executionProvider() const {
    return active_provider_;
}

}  // namespace vellum::detector
