#include "adaptive_tiler.hpp"
#include "geometry_postproc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace vellum::detector {

namespace {

constexpr int kHighResTriggerPx = 5000;

struct WbfBubbleCluster {
    ClassId class_id = ClassId::SpeechBubble;
    BBox fused_bbox;
    double weight_sum = 0.0;
    float max_confidence = 0.0f;
    std::vector<BubbleDetection> members;
};

struct WbfPanelCluster {
    BBox fused_bbox;
    double weight_sum = 0.0;
    float max_confidence = 0.0f;
    int count = 0;
};

void cropRgbTile(
    const uint8_t* rgb,
    int full_w,
    const AdaptiveTiler::TileRect& tile,
    std::vector<uint8_t>& out_rgb) {
    out_rgb.resize(static_cast<size_t>(tile.width) * static_cast<size_t>(tile.height) * 3ULL);
    const size_t row_bytes = static_cast<size_t>(tile.width) * 3ULL;
    for (int y = 0; y < tile.height; ++y) {
        const size_t src_off = (static_cast<size_t>(tile.y + y) * static_cast<size_t>(full_w) +
                                static_cast<size_t>(tile.x)) * 3ULL;
        const size_t dst_off = static_cast<size_t>(y) * row_bytes;
        std::memcpy(out_rgb.data() + dst_off, rgb + src_off, row_bytes);
    }
}

InstanceMask unionClusterMasks(
    const BBox& fused_bbox,
    const std::vector<BubbleDetection>& members,
    int page_w,
    int page_h) {
    InstanceMask out;
    if (!fused_bbox.is_valid()) {
        return out;
    }

    const int x1 = std::clamp(static_cast<int>(std::floor(fused_bbox.x1)), 0, page_w - 1);
    const int y1 = std::clamp(static_cast<int>(std::floor(fused_bbox.y1)), 0, page_h - 1);
    const int x2 = std::clamp(static_cast<int>(std::ceil(fused_bbox.x2)), x1 + 1, page_w);
    const int y2 = std::clamp(static_cast<int>(std::ceil(fused_bbox.y2)), y1 + 1, page_h);

    const int mw = x2 - x1;
    const int mh = y2 - y1;
    if (mw <= 0 || mh <= 0) {
        return out;
    }

    bool any_mask = false;
    for (const auto& m : members) {
        if (!m.mask.empty()) {
            any_mask = true;
            break;
        }
    }
    if (!any_mask) {
        return out;
    }

    out.offset_x = x1;
    out.offset_y = y1;
    out.width = mw;
    out.height = mh;
    out.data.assign(static_cast<size_t>(mw) * static_cast<size_t>(mh), 0);

    for (const auto& m : members) {
        if (m.mask.empty()) continue;
        const int sx0 = std::max(x1, m.mask.offset_x);
        const int sy0 = std::max(y1, m.mask.offset_y);
        const int sx1 = std::min(x2, m.mask.offset_x + m.mask.width);
        const int sy1 = std::min(y2, m.mask.offset_y + m.mask.height);
        for (int gy = sy0; gy < sy1; ++gy) {
            const size_t src_row = static_cast<size_t>(gy - m.mask.offset_y) *
                                   static_cast<size_t>(m.mask.width);
            const size_t dst_row = static_cast<size_t>(gy - y1) * static_cast<size_t>(mw);
            for (int gx = sx0; gx < sx1; ++gx) {
                if (m.mask.data[src_row + static_cast<size_t>(gx - m.mask.offset_x)] != 0) {
                    out.data[dst_row + static_cast<size_t>(gx - x1)] = 255;
                }
            }
        }
    }

    double area = 0.0;
    for (uint8_t v : out.data) {
        if (v != 0) area += 1.0;
    }
    out.area_px = area;

    // Preserve polygon from highest-confidence member if present
    if (!members.empty()) {
        out.polygon = members.front().mask.polygon;
    }
    return out;
}

std::vector<BubbleDetection> fuseBubblesWbf(
    std::vector<BubbleDetection>& raw_dets,
    float iou_threshold,
    int page_w,
    int page_h) {
    std::sort(raw_dets.begin(), raw_dets.end(),
              [](const BubbleDetection& a, const BubbleDetection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<WbfBubbleCluster> clusters;
    for (auto& det : raw_dets) {
        if (!det.bbox.is_valid()) continue;

        int best_idx = -1;
        float best_iou = iou_threshold;
        for (size_t i = 0; i < clusters.size(); ++i) {
            if (clusters[i].class_id != det.class_id) continue;
            const float iou = computeIoU(det.bbox, clusters[i].fused_bbox);
            if (iou >= best_iou) {
                best_iou = iou;
                best_idx = static_cast<int>(i);
            }
        }

        const double w = std::max(1e-4, static_cast<double>(det.confidence));
        if (best_idx < 0) {
            WbfBubbleCluster cl;
            cl.class_id = det.class_id;
            cl.fused_bbox = det.bbox;
            cl.weight_sum = w;
            cl.max_confidence = det.confidence;
            cl.members.push_back(std::move(det));
            clusters.push_back(std::move(cl));
        } else {
            auto& cl = clusters[static_cast<size_t>(best_idx)];
            const double new_w = cl.weight_sum + w;
            cl.fused_bbox.x1 = static_cast<float>(
                (cl.fused_bbox.x1 * cl.weight_sum + det.bbox.x1 * w) / new_w);
            cl.fused_bbox.y1 = static_cast<float>(
                (cl.fused_bbox.y1 * cl.weight_sum + det.bbox.y1 * w) / new_w);
            cl.fused_bbox.x2 = static_cast<float>(
                (cl.fused_bbox.x2 * cl.weight_sum + det.bbox.x2 * w) / new_w);
            cl.fused_bbox.y2 = static_cast<float>(
                (cl.fused_bbox.y2 * cl.weight_sum + det.bbox.y2 * w) / new_w);
            cl.weight_sum = new_w;
            cl.max_confidence = std::max(cl.max_confidence, det.confidence);
            cl.members.push_back(std::move(det));
        }
    }

    std::vector<BubbleDetection> fused;
    fused.reserve(clusters.size());
    int next_id = 0;
    for (auto& cl : clusters) {
        BubbleDetection out;
        out.id = next_id++;
        out.class_id = cl.class_id;
        out.class_name = class_id_to_string(cl.class_id);
        out.confidence = cl.max_confidence;
        out.bbox = cl.fused_bbox;
        out.bbox_norm = BBox{
            out.bbox.x1 / static_cast<float>(page_w),
            out.bbox.y1 / static_cast<float>(page_h),
            out.bbox.x2 / static_cast<float>(page_w),
            out.bbox.y2 / static_cast<float>(page_h)
        };
        out.mask = unionClusterMasks(out.bbox, cl.members, page_w, page_h);
        fused.push_back(std::move(out));
    }
    return fused;
}

std::vector<PanelDetection> fusePanelsWbf(
    std::vector<PanelDetection>& raw_panels,
    float iou_threshold,
    int page_w,
    int page_h) {
    std::sort(raw_panels.begin(), raw_panels.end(),
              [](const PanelDetection& a, const PanelDetection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<WbfPanelCluster> clusters;
    for (const auto& p : raw_panels) {
        if (!p.bbox.is_valid()) continue;

        int best_idx = -1;
        float best_iou = iou_threshold;
        for (size_t i = 0; i < clusters.size(); ++i) {
            const float iou = computeIoU(p.bbox, clusters[i].fused_bbox);
            if (iou >= best_iou) {
                best_iou = iou;
                best_idx = static_cast<int>(i);
            }
        }

        const double w = std::max(1e-4, static_cast<double>(p.confidence));
        if (best_idx < 0) {
            clusters.push_back(WbfPanelCluster{p.bbox, w, p.confidence, 1});
        } else {
            auto& cl = clusters[static_cast<size_t>(best_idx)];
            const double new_w = cl.weight_sum + w;
            cl.fused_bbox.x1 = static_cast<float>(
                (cl.fused_bbox.x1 * cl.weight_sum + p.bbox.x1 * w) / new_w);
            cl.fused_bbox.y1 = static_cast<float>(
                (cl.fused_bbox.y1 * cl.weight_sum + p.bbox.y1 * w) / new_w);
            cl.fused_bbox.x2 = static_cast<float>(
                (cl.fused_bbox.x2 * cl.weight_sum + p.bbox.x2 * w) / new_w);
            cl.fused_bbox.y2 = static_cast<float>(
                (cl.fused_bbox.y2 * cl.weight_sum + p.bbox.y2 * w) / new_w);
            cl.weight_sum = new_w;
            cl.max_confidence = std::max(cl.max_confidence, p.confidence);
            ++cl.count;
        }
    }

    std::vector<PanelDetection> fused;
    fused.reserve(clusters.size());
    int next_id = 0;
    for (const auto& cl : clusters) {
        PanelDetection out;
        out.id = next_id++;
        out.confidence = cl.max_confidence;
        out.bbox = cl.fused_bbox;
        out.bbox_norm = BBox{
            out.bbox.x1 / static_cast<float>(page_w),
            out.bbox.y1 / static_cast<float>(page_h),
            out.bbox.x2 / static_cast<float>(page_w),
            out.bbox.y2 / static_cast<float>(page_h)
        };
        fused.push_back(std::move(out));
    }
    return fused;
}

}  // namespace

bool AdaptiveTiler::shouldTile(
    int width,
    int height,
    const DetectorConfig& config) noexcept {
    if (!config.enable_adaptive_tiling || width <= 0 || height <= 0) {
        return false;
    }
    const float aspect = static_cast<float>(height) / static_cast<float>(width);
    return (aspect > config.tile_aspect_trigger) ||
           (std::max(width, height) > kHighResTriggerPx);
}

std::vector<AdaptiveTiler::TileRect> AdaptiveTiler::computeTileGrid(
    int width,
    int height,
    const DetectorConfig& config) {
    const int tile_size = std::max(256, config.tile_size);
    const float overlap = std::clamp(config.tile_overlap, 0.0f, 0.75f);
    const int stride = std::max(1, static_cast<int>(std::round(
        static_cast<float>(tile_size) * (1.0f - overlap))));

    auto buildAxisStarts = [tile_size, stride](int total_len) {
        std::vector<int> starts;
        if (total_len <= tile_size) {
            starts.push_back(0);
            return starts;
        }
        for (int pos = 0; pos + tile_size < total_len; pos += stride) {
            starts.push_back(pos);
        }
        const int last = total_len - tile_size;
        if (starts.empty() || starts.back() != last) {
            starts.push_back(last);
        }
        return starts;
    };

    const std::vector<int> xs = buildAxisStarts(width);
    const std::vector<int> ys = buildAxisStarts(height);

    std::vector<TileRect> grid;
    grid.reserve(xs.size() * ys.size());
    for (int y : ys) {
        const int th = std::min(tile_size, height - y);
        for (int x : xs) {
            const int tw = std::min(tile_size, width - x);
            grid.push_back(TileRect{x, y, tw, th});
        }
    }
    return grid;
}

PageDetectionResult AdaptiveTiler::detectPage(
    IDetectorEngine& engine,
    const uint8_t* rgbData,
    int width,
    int height,
    const std::string& filename,
    const DetectorConfig& config) {
    if (!rgbData || width <= 0 || height <= 0) {
        throw std::invalid_argument("AdaptiveTiler::detectPage: invalid RGB buffer or dimensions.");
    }

    if (!shouldTile(width, height, config)) {
        return engine.detectPage(rgbData, width, height, filename, config);
    }

    const auto tiles = computeTileGrid(width, height, config);
    if (tiles.size() <= 1) {
        return engine.detectPage(rgbData, width, height, filename, config);
    }

    const auto t_start = std::chrono::steady_clock::now();

    // Keep session alive across tiles within a single page, then honor config.keep_session_alive.
    DetectorConfig tile_cfg = config;
    tile_cfg.keep_session_alive = true;

    std::vector<uint8_t> tile_rgb;
    std::vector<PanelDetection> global_panels;
    std::vector<BubbleDetection> global_dets;
    double total_infer_ms = 0.0;
    RuntimeMeta last_meta;

    for (const auto& tile : tiles) {
        cropRgbTile(rgbData, width, tile, tile_rgb);
        PageDetectionResult tile_res =
            engine.detectPage(tile_rgb.data(), tile.width, tile.height, filename, tile_cfg);

        total_infer_ms += tile_res.runtime_meta.inference_ms;
        last_meta = tile_res.runtime_meta;

        const float dx = static_cast<float>(tile.x);
        const float dy = static_cast<float>(tile.y);

        for (auto& p : tile_res.panels) {
            p.bbox.x1 += dx;
            p.bbox.y1 += dy;
            p.bbox.x2 += dx;
            p.bbox.y2 += dy;
            global_panels.push_back(std::move(p));
        }

        for (auto& d : tile_res.detections) {
            d.bbox.x1 += dx;
            d.bbox.y1 += dy;
            d.bbox.x2 += dx;
            d.bbox.y2 += dy;
            d.mask.offset_x += tile.x;
            d.mask.offset_y += tile.y;
            for (auto& pt : d.mask.polygon) {
                pt.x += dx;
                pt.y += dy;
            }
            global_dets.push_back(std::move(d));
        }
    }

    if (!config.keep_session_alive) {
        engine.unload();
    }

    const auto t_fuse_start = std::chrono::steady_clock::now();

    PageDetectionResult result;
    result.schema_version = "2.0";
    result.filename = filename;
    result.image_width = width;
    result.image_height = height;
    result.reading_direction = config.reading_direction;
    result.runtime_meta = last_meta;
    result.runtime_meta.tiles_processed = static_cast<int>(tiles.size());
    result.runtime_meta.inference_ms = total_infer_ms;

    result.panels = fusePanelsWbf(global_panels, config.nms_iou_threshold, width, height);
    result.detections = fuseBubblesWbf(global_dets, config.nms_iou_threshold, width, height);

    runGeometryPostproc(result, config);

    const auto t_end = std::chrono::steady_clock::now();
    result.runtime_meta.postproc_ms =
        std::chrono::duration<double, std::milli>(t_end - t_fuse_start).count();

    return result;
}

}  // namespace vellum::detector
