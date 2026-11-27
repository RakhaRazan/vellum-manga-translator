#include "adaptive_tiler.hpp"
#include "geometry_postproc.hpp"
#include "math_det_ops.hpp"
#include "yolo_seg_backend.hpp"

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
    for (auto& cl : clusters) {
        BubbleDetection out;
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

    // Same-class containment suppression pass: drop split fragments straddling tile seams
    // when intersection / min(areaA, areaB) >= 0.70, keeping the higher-confidence/more complete box.
    std::sort(fused.begin(), fused.end(), [](const BubbleDetection& a, const BubbleDetection& b) {
        if (std::abs(a.confidence - b.confidence) > 1e-4f) {
            return a.confidence > b.confidence;
        }
        return a.bbox.area() > b.bbox.area();
    });

    std::vector<bool> suppressed(fused.size(), false);
    for (size_t i = 0; i < fused.size(); ++i) {
        if (suppressed[i]) continue;
        const BBox& a = fused[i].bbox;
        const float area_a = a.area();
        if (area_a <= 0.0f) continue;

        for (size_t j = i + 1; j < fused.size(); ++j) {
            if (suppressed[j] || fused[j].class_id != fused[i].class_id) continue;
            const BBox& b = fused[j].bbox;
            const float area_b = b.area();
            if (area_b <= 0.0f) continue;

            const float ix1 = std::max(a.x1, b.x1);
            const float iy1 = std::max(a.y1, b.y1);
            const float ix2 = std::min(a.x2, b.x2);
            const float iy2 = std::min(a.y2, b.y2);
            if (ix2 <= ix1 || iy2 <= iy1) continue;

            const float inter = (ix2 - ix1) * (iy2 - iy1);
            const float min_area = std::min(area_a, area_b);
            if ((inter / min_area) >= 0.70f) {
                suppressed[j] = true;
            }
        }
    }

    std::vector<BubbleDetection> final_fused;
    final_fused.reserve(fused.size());
    int next_id = 0;
    for (size_t i = 0; i < fused.size(); ++i) {
        if (suppressed[i]) continue;
        fused[i].id = next_id++;
        final_fused.push_back(std::move(fused[i]));
    }
    return final_fused;
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

float computeDarkPixelRatio(const uint8_t* rgb, int width, int height) noexcept {
    if (!rgb || width <= 0 || height <= 0) return 0.0f;
    const int step = 4;
    size_t total = 0;
    size_t dark = 0;
    for (int y = 0; y < height; y += step) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(width) * 3ULL;
        for (int x = 0; x < width; x += step) {
            const size_t idx = row + static_cast<size_t>(x) * 3ULL;
            const int lum = (77 * static_cast<int>(rgb[idx]) +
                             150 * static_cast<int>(rgb[idx + 1]) +
                             29 * static_cast<int>(rgb[idx + 2])) >> 8;
            if (lum < 60) ++dark;
            ++total;
        }
    }
    return (total > 0) ? (static_cast<float>(dark) / static_cast<float>(total)) : 0.0f;
}

void buildInvertedRgb(const uint8_t* rgb, int width, int height, std::vector<uint8_t>& out) {
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height) * 3ULL;
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<uint8_t>(255u - rgb[i]);
    }
}

void buildHFlipRgb(const uint8_t* rgb, int width, int height, std::vector<uint8_t>& out) {
    out.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3ULL);
    for (int y = 0; y < height; ++y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(width) * 3ULL;
        for (int x = 0; x < width; ++x) {
            const size_t src = row + static_cast<size_t>(width - 1 - x) * 3ULL;
            const size_t dst = row + static_cast<size_t>(x) * 3ULL;
            out[dst + 0] = rgb[src + 0];
            out[dst + 1] = rgb[src + 1];
            out[dst + 2] = rgb[src + 2];
        }
    }
}

void mirrorResultHorizontally(PageDetectionResult& res, int page_w) {
    const float fw = static_cast<float>(page_w);
    for (auto& p : res.panels) {
        const float nx1 = std::clamp(fw - p.bbox.x2, 0.0f, fw);
        const float nx2 = std::clamp(fw - p.bbox.x1, 0.0f, fw);
        p.bbox.x1 = nx1;
        p.bbox.x2 = nx2;
    }
    for (auto& d : res.detections) {
        const float nx1 = std::clamp(fw - d.bbox.x2, 0.0f, fw);
        const float nx2 = std::clamp(fw - d.bbox.x1, 0.0f, fw);
        d.bbox.x1 = nx1;
        d.bbox.x2 = nx2;
        if (!d.mask.empty()) {
            d.mask.offset_x = std::clamp(page_w - (d.mask.offset_x + d.mask.width), 0, std::max(0, page_w - d.mask.width));
            for (int y = 0; y < d.mask.height; ++y) {
                uint8_t* row = d.mask.data.data() + static_cast<size_t>(y) * static_cast<size_t>(d.mask.width);
                std::reverse(row, row + d.mask.width);
            }
            for (auto& pt : d.mask.polygon) {
                pt.x = std::clamp(fw - pt.x, 0.0f, fw);
            }
        }
    }
}

std::vector<AdaptiveTiler::TileRect> buildSpreadSplitCrops(int width, int height) {
    // Fast 2-way Left/Right half split with 10% center overlap for double-page spreads.
    const int half_w = width / 2;
    const int overlap_half = std::max(1, static_cast<int>(std::round(static_cast<float>(width) * 0.05f)));
    const int left_w = std::min(width, half_w + overlap_half);
    const int right_x = std::max(0, half_w - overlap_half);
    const int right_w = width - right_x;
    return {
        AdaptiveTiler::TileRect{0, 0, left_w, height},
        AdaptiveTiler::TileRect{right_x, 0, right_w, height}
    };
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

    // #18: Opt-In RANSAC + 3x3 Homography Deskew & Lightweight TPS Spine Dewarp (0 ms when disabled).
    DeskewResult deskew_res;
    const uint8_t* active_rgb = rgbData;
    DetectorConfig active_cfg = config;
    if (config.auto_deskew_scans && !config.is_subpass) {
        deskew_res = deskewAndDewarpScan(rgbData, width, height);
        if (deskew_res.applied && deskew_res.warped_rgb.size() ==
            static_cast<size_t>(width) * static_cast<size_t>(height) * 3ULL) {
            active_rgb = deskew_res.warped_rgb.data();
        }
        active_cfg.auto_deskew_scans = false;
    }

    if (!shouldTile(width, height, active_cfg)) {
        if (active_cfg.precision == DetectorPrecision::Fast) {
            return engine.detectPage(active_rgb, width, height, filename, active_cfg);
        }

        // Balanced / HighPrecision multi-pass orchestration with WBF fusion.
        DetectorConfig sub_cfg = active_cfg;
        sub_cfg.keep_session_alive = true;
        sub_cfg.is_subpass = true;

        std::vector<PanelDetection> global_panels;
        std::vector<BubbleDetection> global_dets;
        double total_infer_ms = 0.0;
        int passes_run = 0;

        auto collectPass = [&](PageDetectionResult&& pass_res) {
            total_infer_ms += pass_res.runtime_meta.inference_ms;
            ++passes_run;
            for (auto& p : pass_res.panels) {
                global_panels.push_back(std::move(p));
            }
            for (auto& d : pass_res.detections) {
                global_dets.push_back(std::move(d));
            }
        };

        // Pass 1: Global page pass (includes Dual-Threshold Rescue inside YOLO backend).
        PageDetectionResult base_res = engine.detectPage(active_rgb, width, height, filename, sub_cfg);
        RuntimeMeta last_meta = base_res.runtime_meta;
        collectPass(std::move(base_res));

        // Pass 2 (Balanced & HighPrecision): Selective Polarity-Invert TTA on dark pages (>= 12% dark pixels).
        if (computeDarkPixelRatio(active_rgb, width, height) >= 0.12f) {
            std::vector<uint8_t> inv_rgb;
            buildInvertedRgb(active_rgb, width, height, inv_rgb);
            collectPass(engine.detectPage(inv_rgb.data(), width, height, filename, sub_cfg));
        }

        // HighPrecision passes: Horizontal Flip TTA + Spread-Only 2-way Left/Right split (10% overlap).
        if (active_cfg.precision == DetectorPrecision::HighPrecision) {
            std::vector<uint8_t> flip_rgb;
            buildHFlipRgb(active_rgb, width, height, flip_rgb);
            PageDetectionResult flip_res = engine.detectPage(flip_rgb.data(), width, height, filename, sub_cfg);
            mirrorResultHorizontally(flip_res, width);
            collectPass(std::move(flip_res));

            const bool is_double_page_spread =
                static_cast<float>(width) > static_cast<float>(height) * 1.15f;
            if (is_double_page_spread) {
                const auto zoom_crops = buildSpreadSplitCrops(width, height);
                std::vector<uint8_t> crop_rgb;
                for (const auto& crop : zoom_crops) {
                    cropRgbTile(active_rgb, width, crop, crop_rgb);
                    PageDetectionResult crop_res =
                        engine.detectPage(crop_rgb.data(), crop.width, crop.height, filename, sub_cfg);
                    const float dx = static_cast<float>(crop.x);
                    const float dy = static_cast<float>(crop.y);
                    for (auto& p : crop_res.panels) {
                        p.bbox.x1 += dx;
                        p.bbox.y1 += dy;
                        p.bbox.x2 += dx;
                        p.bbox.y2 += dy;
                    }
                    for (auto& d : crop_res.detections) {
                        d.bbox.x1 += dx;
                        d.bbox.y1 += dy;
                        d.bbox.x2 += dx;
                        d.bbox.y2 += dy;
                        d.mask.offset_x += crop.x;
                        d.mask.offset_y += crop.y;
                        for (auto& pt : d.mask.polygon) {
                            pt.x += dx;
                            pt.y += dy;
                        }
                    }
                    collectPass(std::move(crop_res));
                }
            }
        }

        if (!active_cfg.keep_session_alive) {
            engine.unload();
        }

        const auto t_fuse_start = std::chrono::steady_clock::now();

        PageDetectionResult result;
        result.schema_version = "2.0";
        result.filename = filename;
        result.image_width = width;
        result.image_height = height;
        result.reading_direction = active_cfg.reading_direction;
        result.runtime_meta = last_meta;
        result.runtime_meta.tiles_processed = passes_run;
        result.runtime_meta.inference_ms = total_infer_ms;

        result.panels = fusePanelsWbf(global_panels, active_cfg.nms_iou_threshold, width, height);
        result.detections = fuseBubblesWbf(global_dets, active_cfg.nms_iou_threshold, width, height);

        runGeometryPostproc(result, active_cfg, active_rgb);
        Yolo11SegBackend::applyV25CvStack(result, active_rgb, width, height, active_cfg);

        const auto t_end = std::chrono::steady_clock::now();
        result.runtime_meta.postproc_ms =
            std::chrono::duration<double, std::milli>(t_end - t_fuse_start).count();
        return result;
    }

    // #8: 1D DP Minimum-Energy Seam Webtoon Tiling for tall strips so tiles slice
    // through empty gutters instead of bisecting speech bubbles.
    std::vector<TileRect> tiles;
    const bool is_webtoon_strip =
        static_cast<float>(height) > static_cast<float>(width) * active_cfg.tile_aspect_trigger;
    if (is_webtoon_strip) {
        const int target_h = std::max(256, active_cfg.tile_size);
        const int min_h = std::max(256, (target_h * 6) / 10);
        const int max_h = (target_h * 14) / 10;
        const std::vector<int> seam_ys =
            computeWebtoonSeamSplits(active_rgb, width, height, target_h, min_h, max_h);

        const int tile_w = std::max(256, active_cfg.tile_size);
        const float overlap = std::clamp(active_cfg.tile_overlap, 0.0f, 0.75f);
        const int x_stride = std::max(1, static_cast<int>(std::round(static_cast<float>(tile_w) * (1.0f - overlap))));
        std::vector<int> xs;
        if (width <= tile_w) {
            xs.push_back(0);
        } else {
            for (int pos = 0; pos + tile_w < width; pos += x_stride) xs.push_back(pos);
            if (xs.empty() || xs.back() != width - tile_w) xs.push_back(width - tile_w);
        }

        constexpr int kSeamGuardPx = 16;
        for (size_t k = 0; k + 1 < seam_ys.size(); ++k) {
            const int y0 = std::max(0, seam_ys[k] - (k > 0 ? kSeamGuardPx : 0));
            const int y1 = std::min(height, seam_ys[k + 1] + (k + 2 < seam_ys.size() ? kSeamGuardPx : 0));
            const int th = y1 - y0;
            if (th <= 0) continue;
            for (int x : xs) {
                const int tw = std::min(tile_w, width - x);
                tiles.push_back(TileRect{x, y0, tw, th});
            }
        }
    } else {
        tiles = computeTileGrid(width, height, active_cfg);
    }

    if (tiles.size() <= 1) {
        return engine.detectPage(active_rgb, width, height, filename, active_cfg);
    }

    const auto t_start = std::chrono::steady_clock::now();

    // Keep session alive across tiles within a single page, then honor config.keep_session_alive.
    DetectorConfig tile_cfg = active_cfg;
    tile_cfg.keep_session_alive = true;
    tile_cfg.is_subpass = true;

    std::vector<uint8_t> tile_rgb;
    std::vector<PanelDetection> global_panels;
    std::vector<BubbleDetection> global_dets;
    double total_infer_ms = 0.0;
    RuntimeMeta last_meta;

    for (const auto& tile : tiles) {
        cropRgbTile(active_rgb, width, tile, tile_rgb);
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

    if (!active_cfg.keep_session_alive) {
        engine.unload();
    }

    const auto t_fuse_start = std::chrono::steady_clock::now();

    PageDetectionResult result;
    result.schema_version = "2.0";
    result.filename = filename;
    result.image_width = width;
    result.image_height = height;
    result.reading_direction = active_cfg.reading_direction;
    result.runtime_meta = last_meta;
    result.runtime_meta.tiles_processed = static_cast<int>(tiles.size());
    result.runtime_meta.inference_ms = total_infer_ms;

    result.panels = fusePanelsWbf(global_panels, active_cfg.nms_iou_threshold, width, height);
    result.detections = fuseBubblesWbf(global_dets, active_cfg.nms_iou_threshold, width, height);

    runGeometryPostproc(result, active_cfg, active_rgb);
    Yolo11SegBackend::applyV25CvStack(result, active_rgb, width, height, active_cfg);

    const auto t_end = std::chrono::steady_clock::now();
    result.runtime_meta.postproc_ms =
        std::chrono::duration<double, std::milli>(t_end - t_fuse_start).count();

    return result;
}

}  // namespace vellum::detector
