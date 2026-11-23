#include "export_artifacts.hpp"

#include <stb_image_write.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

namespace vellum::detector {

namespace {

constexpr int kCropPaddingPx = 4;

const char* readingDirectionToString(ReadingDirection dir) noexcept {
    return (dir == ReadingDirection::RTL) ? "RTL" : "LTR";
}

const char* backendTypeToString(BackendType backend) noexcept {
    switch (backend) {
        case BackendType::Auto:       return "auto";
        case BackendType::Yolo11Seg:  return "yolo11_seg";
        case BackendType::RfDetrSeg:  return "rfdetr_seg";
        case BackendType::LegacyCtd:  return "legacy_ctd";
        default:                      return "unknown";
    }
}

nlohmann::json bboxToJson(const BBox& b) {
    return nlohmann::json::array({b.x1, b.y1, b.x2, b.y2});
}

std::string extractFileStem(const std::string& filename) {
    if (filename.empty()) {
        return "page";
    }
    std::filesystem::path p = std::filesystem::u8path(filename);
    std::string stem = p.stem().u8string();
    return stem.empty() ? "page" : stem;
}

}  // namespace

nlohmann::json toSchemaV2Json(const PageDetectionResult& result) {
    nlohmann::json panels_json = nlohmann::json::array();
    for (const auto& p : result.panels) {
        panels_json.push_back({
            {"id", p.id},
            {"reading_order", p.reading_order},
            {"confidence", p.confidence},
            {"bbox", bboxToJson(p.bbox)},
            {"bbox_norm", bboxToJson(p.bbox_norm)}
        });
    }

    nlohmann::json detections_json = nlohmann::json::array();
    for (const auto& d : result.detections) {
        nlohmann::json poly_json = nlohmann::json::array();
        for (const auto& pt : d.mask.polygon) {
            poly_json.push_back(nlohmann::json::array({pt.x, pt.y}));
        }

        detections_json.push_back({
            {"id", d.id},
            {"reading_order", d.reading_order},
            {"panel_id", d.panel_id},
            {"is_cross_panel", d.is_cross_panel},
            {"class_id", static_cast<int>(d.class_id)},
            {"class_name", d.class_name.empty() ? class_id_to_string(d.class_id) : d.class_name},
            {"confidence", d.confidence},
            {"bbox", bboxToJson(d.bbox)},
            {"bbox_norm", bboxToJson(d.bbox_norm)},
            {"safe_bbox", bboxToJson(d.safe_bbox)},
            {"mask", {
                {"offset_x", d.mask.offset_x},
                {"offset_y", d.mask.offset_y},
                {"width", d.mask.width},
                {"height", d.mask.height},
                {"area_px", d.mask.area_px},
                {"polygon", std::move(poly_json)},
                {"mask_file", d.mask.mask_file}
            }},
            {"crop_file", d.crop_file}
        });
    }

    return nlohmann::json{
        {"schema_version", result.schema_version.empty() ? "2.0" : result.schema_version},
        {"image", {
            {"filename", result.filename},
            {"width", result.image_width},
            {"height", result.image_height},
            {"reading_direction", readingDirectionToString(result.reading_direction)}
        }},
        {"runtime_meta", {
            {"backend", backendTypeToString(result.runtime_meta.backend)},
            {"model_name", result.runtime_meta.model_name},
            {"execution_provider", result.runtime_meta.execution_provider},
            {"inference_ms", result.runtime_meta.inference_ms},
            {"postproc_ms", result.runtime_meta.postproc_ms},
            {"tiles_processed", result.runtime_meta.tiles_processed}
        }},
        {"panels", std::move(panels_json)},
        {"detections", std::move(detections_json)}
    };
}

bool exportPageArtifacts(
    PageDetectionResult& result,
    const uint8_t* rgbData,
    int width,
    int height,
    const DetectorConfig& config) {
    if (config.export_dir.empty()) {
        return false;
    }

    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path export_root = fs::u8path(config.export_dir);
    const fs::path crops_dir = export_root / "crops";
    const fs::path masks_dir = export_root / "masks";

    fs::create_directories(crops_dir, ec);
    if (ec) return false;
    fs::create_directories(masks_dir, ec);
    if (ec) return false;

    const std::string stem = extractFileStem(result.filename);

    for (auto& det : result.detections) {
        const std::string id_str = std::to_string(det.id);

        // 1. Write cropped RGB PNG with 4px clamped padding
        if (rgbData && width > 0 && height > 0 && det.bbox.is_valid()) {
            const int cx1 = std::clamp(static_cast<int>(std::floor(det.bbox.x1)) - kCropPaddingPx, 0, width - 1);
            const int cy1 = std::clamp(static_cast<int>(std::floor(det.bbox.y1)) - kCropPaddingPx, 0, height - 1);
            const int cx2 = std::clamp(static_cast<int>(std::ceil(det.bbox.x2)) + kCropPaddingPx, cx1 + 1, width);
            const int cy2 = std::clamp(static_cast<int>(std::ceil(det.bbox.y2)) + kCropPaddingPx, cy1 + 1, height);
            const int cw = cx2 - cx1;
            const int ch = cy2 - cy1;

            if (cw > 0 && ch > 0) {
                std::vector<uint8_t> crop_buf(static_cast<size_t>(cw) * static_cast<size_t>(ch) * 3ULL);
                const size_t row_bytes = static_cast<size_t>(cw) * 3ULL;
                for (int y = 0; y < ch; ++y) {
                    const size_t src_off = (static_cast<size_t>(cy1 + y) * static_cast<size_t>(width) +
                                            static_cast<size_t>(cx1)) * 3ULL;
                    std::memcpy(crop_buf.data() + static_cast<size_t>(y) * row_bytes,
                                rgbData + src_off, row_bytes);
                }
                const std::string rel_crop = "crops/" + id_str + "_crop.png";
                const fs::path crop_path = export_root / fs::u8path(rel_crop);
                if (stbi_write_png(crop_path.u8string().c_str(), cw, ch, 3, crop_buf.data(), cw * 3) != 0) {
                    det.crop_file = rel_crop;
                }
            }
        }

        // 2. Write 8-bit binary mask PNG
        if (!det.mask.empty()) {
            const std::string rel_mask = "masks/" + id_str + "_mask.png";
            const fs::path mask_path = export_root / fs::u8path(rel_mask);
            if (stbi_write_png(mask_path.u8string().c_str(),
                               det.mask.width,
                               det.mask.height,
                               1,
                               det.mask.data.data(),
                               det.mask.width) != 0) {
                det.mask.mask_file = rel_mask;
            }
        }
    }

    // 3. Write Schema 2.0 JSON
    const fs::path json_path = export_root / fs::u8path(stem + ".detections.json");
    std::ofstream ofs(json_path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        return false;
    }
    const nlohmann::json payload = toSchemaV2Json(result);
    ofs << payload.dump(2) << '\n';
    return ofs.good();
}

}  // namespace vellum::detector
