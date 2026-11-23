#include "geometry_postproc.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vellum::detector {

namespace {

constexpr float kMinBoxDimensionPx = 12.0f;
constexpr double kMinMaskAreaRatio = 0.00015;  // 0.015% of page area
constexpr float kSafeBoxInnerPadding = 0.05f;
constexpr float kPanelIoAAssignThreshold = 0.45f;
constexpr float kPanelIoACrossThreshold = 0.20f;
constexpr float kDagOverlapTolerance = 0.15f;

float intersectionArea(const BBox& a, const BBox& b) noexcept {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    if (ix2 <= ix1 || iy2 <= iy1) {
        return 0.0f;
    }
    return (ix2 - ix1) * (iy2 - iy1);
}

BBox fallbackInsetBBox(const BBox& bbox) noexcept {
    const float w = bbox.width();
    const float h = bbox.height();
    if (w <= 0.0f || h <= 0.0f) {
        return bbox;
    }
    const float pad_x = w * kSafeBoxInnerPadding;
    const float pad_y = h * kSafeBoxInnerPadding;
    BBox safe{
        bbox.x1 + pad_x,
        bbox.y1 + pad_y,
        bbox.x2 - pad_x,
        bbox.y2 - pad_y
    };
    if (safe.x2 <= safe.x1) {
        const float mid_x = bbox.cx();
        safe.x1 = std::max(bbox.x1, mid_x - 0.5f);
        safe.x2 = std::min(bbox.x2, mid_x + 0.5f);
    }
    if (safe.y2 <= safe.y1) {
        const float mid_y = bbox.cy();
        safe.y1 = std::max(bbox.y1, mid_y - 0.5f);
        safe.y2 = std::min(bbox.y2, mid_y + 0.5f);
    }
    return safe;
}

double effectiveMaskAreaPx(const BubbleDetection& det) noexcept {
    if (det.mask.area_px > 0.0) {
        return det.mask.area_px;
    }
    if (!det.mask.empty()) {
        double count = 0.0;
        const size_t total = static_cast<size_t>(det.mask.width) *
                             static_cast<size_t>(det.mask.height);
        const size_t n = std::min(total, det.mask.data.size());
        for (size_t i = 0; i < n; ++i) {
            if (det.mask.data[i] != 0) {
                count += 1.0;
            }
        }
        return count;
    }
    return static_cast<double>(det.bbox.area());
}

Point2f computeMaskCentroid(const BubbleDetection& det) noexcept {
    if (!det.mask.empty()) {
        const int w = det.mask.width;
        const int h = det.mask.height;
        const size_t expected = static_cast<size_t>(w) * static_cast<size_t>(h);
        if (det.mask.data.size() >= expected) {
            double sum_x = 0.0;
            double sum_y = 0.0;
            double count = 0.0;
            for (int y = 0; y < h; ++y) {
                const size_t row_off = static_cast<size_t>(y) * static_cast<size_t>(w);
                for (int x = 0; x < w; ++x) {
                    if (det.mask.data[row_off + static_cast<size_t>(x)] != 0) {
                        sum_x += static_cast<double>(x) + 0.5;
                        sum_y += static_cast<double>(y) + 0.5;
                        count += 1.0;
                    }
                }
            }
            if (count > 0.0) {
                return Point2f{
                    static_cast<float>(det.mask.offset_x) + static_cast<float>(sum_x / count),
                    static_cast<float>(det.mask.offset_y) + static_cast<float>(sum_y / count)
                };
            }
        }
    }
    return Point2f{det.bbox.cx(), det.bbox.cy()};
}

struct DagNode {
    size_t original_index = 0;
    BBox bbox;
    Point2f centroid;
};

// Cycle-safe DAG topological sort with deterministic tie-breaking.
std::vector<size_t> topologicalSortDag(
    const std::vector<DagNode>& nodes,
    ReadingDirection direction) {
    const size_t n = nodes.size();
    if (n == 0) {
        return {};
    }
    if (n == 1) {
        return {0};
    }

    std::vector<std::vector<size_t>> adj(n);
    std::vector<int> in_degree(n, 0);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const BBox& a = nodes[i].bbox;
            const BBox& b = nodes[j].bbox;
            const Point2f& ca = nodes[i].centroid;
            const Point2f& cb = nodes[j].centroid;

            const float h_min = std::min(a.height(), b.height());
            const float v_overlap = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));

            if (h_min <= 0.0f || v_overlap <= kDagOverlapTolerance * h_min) {
                // Vertical dominance
                if (ca.y < cb.y) {
                    adj[i].push_back(j);
                    ++in_degree[j];
                } else if (cb.y < ca.y) {
                    adj[j].push_back(i);
                    ++in_degree[i];
                }
            } else {
                // Same horizontal band: apply reading direction
                if (direction == ReadingDirection::RTL) {
                    if (ca.x > cb.x) {
                        adj[i].push_back(j);
                        ++in_degree[j];
                    } else if (cb.x > ca.x) {
                        adj[j].push_back(i);
                        ++in_degree[i];
                    }
                } else {
                    if (ca.x < cb.x) {
                        adj[i].push_back(j);
                        ++in_degree[j];
                    } else if (cb.x < ca.x) {
                        adj[j].push_back(i);
                        ++in_degree[i];
                    }
                }
            }
        }
    }

    auto tieBreakBefore = [direction, &nodes](size_t lhs, size_t rhs) noexcept {
        const Point2f& ca = nodes[lhs].centroid;
        const Point2f& cb = nodes[rhs].centroid;
        if (std::abs(ca.y - cb.y) > 1e-3f) {
            return ca.y < cb.y;  // Top-to-bottom
        }
        if (direction == ReadingDirection::RTL) {
            if (std::abs(ca.x - cb.x) > 1e-3f) {
                return ca.x > cb.x;  // Right-to-left
            }
        } else {
            if (std::abs(ca.x - cb.x) > 1e-3f) {
                return ca.x < cb.x;  // Left-to-right
            }
        }
        return nodes[lhs].original_index < nodes[rhs].original_index;
    };

    std::vector<bool> visited(n, false);
    std::vector<size_t> order;
    order.reserve(n);

    while (order.size() < n) {
        // Pick an unvisited node with in_degree == 0 (or minimum in_degree if a cycle exists).
        size_t best = n;
        int best_deg = 0;

        for (size_t i = 0; i < n; ++i) {
            if (visited[i]) {
                continue;
            }
            const int deg = std::max(0, in_degree[i]);
            if (best == n || deg < best_deg || (deg == best_deg && tieBreakBefore(i, best))) {
                best = i;
                best_deg = deg;
            }
        }

        if (best == n) {
            break;
        }

        visited[best] = true;
        order.push_back(best);
        for (size_t nxt : adj[best]) {
            if (!visited[nxt] && in_degree[nxt] > 0) {
                --in_degree[nxt];
            }
        }
    }

    return order;
}

}  // namespace

float computeIoU(const BBox& a, const BBox& b) noexcept {
    const float inter = intersectionArea(a, b);
    if (inter <= 0.0f) {
        return 0.0f;
    }
    const float uni = a.area() + b.area() - inter;
    return (uni > 0.0f) ? (inter / uni) : 0.0f;
}

float computeIoA(const BBox& inner, const BBox& outer) noexcept {
    const float inner_area = inner.area();
    if (inner_area <= 0.0f) {
        return 0.0f;
    }
    const float inter = intersectionArea(inner, outer);
    return inter / inner_area;
}

void filterSfxAndMicroArtifacts(
    std::vector<BubbleDetection>& detections,
    int imageWidth,
    int imageHeight,
    const DetectorConfig& config) {
    const double page_area =
        (imageWidth > 0 && imageHeight > 0)
            ? static_cast<double>(imageWidth) * static_cast<double>(imageHeight)
            : 0.0;
    const double min_mask_area = page_area * kMinMaskAreaRatio;

    // Collect SfxIgnore references before filtering so suppression checks all SFX boxes.
    std::vector<BubbleDetection> sfx_boxes;
    for (const auto& det : detections) {
        if (det.class_id == ClassId::SfxIgnore && det.bbox.is_valid()) {
            sfx_boxes.push_back(det);
        }
    }

    std::vector<BubbleDetection> kept;
    kept.reserve(detections.size());

    for (auto& det : detections) {
        if (!det.bbox.is_valid()) {
            continue;
        }

        // Micro-artifact filtering
        if (det.bbox.width() < kMinBoxDimensionPx || det.bbox.height() < kMinBoxDimensionPx) {
            continue;
        }
        if (min_mask_area > 0.0 && effectiveMaskAreaPx(det) < min_mask_area) {
            continue;
        }

        // SFX suppression for dialogue candidates (0..5)
        if (is_dialogue_class(det.class_id)) {
            bool suppressed = false;
            for (const auto& sfx : sfx_boxes) {
                if (sfx.confidence > det.confidence &&
                    computeIoU(det.bbox, sfx.bbox) > config.sfx_suppress_iou) {
                    suppressed = true;
                    break;
                }
            }
            if (suppressed) {
                continue;
            }
        }

        kept.push_back(std::move(det));
    }

    detections = std::move(kept);
}

BBox computeSafeBBox(const BBox& bbox, const InstanceMask& mask) noexcept {
    if (!bbox.is_valid()) {
        return bbox;
    }
    if (mask.empty()) {
        return fallbackInsetBBox(bbox);
    }

    const int w = mask.width;
    const int h = mask.height;
    const size_t total = static_cast<size_t>(w) * static_cast<size_t>(h);
    if (mask.data.size() < total) {
        return fallbackInsetBBox(bbox);
    }

    // 3x3 morphological closing: dilation followed by erosion.
    std::vector<uint8_t> dilated(total, 0);
    for (int y = 0; y < h; ++y) {
        const int y0 = std::max(0, y - 1);
        const int y1 = std::min(h - 1, y + 1);
        for (int x = 0; x < w; ++x) {
            const int x0 = std::max(0, x - 1);
            const int x1 = std::min(w - 1, x + 1);
            uint8_t any_fg = 0;
            for (int ny = y0; ny <= y1 && !any_fg; ++ny) {
                const size_t row = static_cast<size_t>(ny) * static_cast<size_t>(w);
                for (int nx = x0; nx <= x1; ++nx) {
                    if (mask.data[row + static_cast<size_t>(nx)] != 0) {
                        any_fg = 1;
                        break;
                    }
                }
            }
            dilated[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] = any_fg;
        }
    }

    std::vector<uint8_t> closed(total, 0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t all_fg = 1;
            for (int dy = -1; dy <= 1 && all_fg; ++dy) {
                const int ny = y + dy;
                if (ny < 0 || ny >= h) {
                    all_fg = 0;
                    break;
                }
                const size_t row = static_cast<size_t>(ny) * static_cast<size_t>(w);
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx;
                    if (nx < 0 || nx >= w || dilated[row + static_cast<size_t>(nx)] == 0) {
                        all_fg = 0;
                        break;
                    }
                }
            }
            closed[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] = all_fg;
        }
    }

    // Erode the closed bubble mask inward by 3px so the interior contour stays off the black ink border.
    constexpr int kErodeR = 3;
    constexpr int kErodeR2 = kErodeR * kErodeR;
    std::vector<uint8_t> eroded(total, 0);
    int mask_x_min = w, mask_y_min = h, mask_x_max = -1, mask_y_max = -1;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (closed[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] == 0) {
                continue;
            }
            uint8_t keep = 1;
            for (int dy = -kErodeR; dy <= kErodeR && keep; ++dy) {
                const int ny = y + dy;
                if (ny < 0 || ny >= h) {
                    keep = 0;
                    break;
                }
                const size_t row = static_cast<size_t>(ny) * static_cast<size_t>(w);
                for (int dx = -kErodeR; dx <= kErodeR; ++dx) {
                    if (dx * dx + dy * dy > kErodeR2) continue;
                    const int nx = x + dx;
                    if (nx < 0 || nx >= w || closed[row + static_cast<size_t>(nx)] == 0) {
                        keep = 0;
                        break;
                    }
                }
            }
            if (keep) {
                eroded[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] = 1;
                mask_x_min = std::min(mask_x_min, x);
                mask_y_min = std::min(mask_y_min, y);
                mask_x_max = std::max(mask_x_max, x);
                mask_y_max = std::max(mask_y_max, y);
            }
        }
    }

    // Fallback to closed mask if the bubble was too small for 3px erosion
    const std::vector<uint8_t>& active_mask = (mask_x_max > mask_x_min && mask_y_max > mask_y_min)
        ? eroded
        : closed;
    if (mask_x_max <= mask_x_min || mask_y_max <= mask_y_min) {
        mask_x_min = w;
        mask_y_min = h;
        mask_x_max = -1;
        mask_y_max = -1;
        for (int y = 0; y < h; ++y) {
            const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
            for (int x = 0; x < w; ++x) {
                if (closed[row + static_cast<size_t>(x)] != 0) {
                    mask_x_min = std::min(mask_x_min, x);
                    mask_y_min = std::min(mask_y_min, y);
                    mask_x_max = std::max(mask_x_max, x);
                    mask_y_max = std::max(mask_y_max, y);
                }
            }
        }
    }

    // Fast row-histogram largest inscribed axis-aligned rectangle over active_mask.
    std::vector<int> heights(static_cast<size_t>(w), 0);
    std::vector<int> stack;
    stack.reserve(static_cast<size_t>(w) + 1);

    int best_area = 0;
    int best_x1 = 0;
    int best_y1 = 0;
    int best_x2 = 0;
    int best_y2 = 0;

    for (int y = 0; y < h; ++y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        for (int x = 0; x < w; ++x) {
            heights[static_cast<size_t>(x)] =
                (active_mask[row + static_cast<size_t>(x)] != 0)
                    ? (heights[static_cast<size_t>(x)] + 1)
                    : 0;
        }

        stack.clear();
        for (int x = 0; x <= w; ++x) {
            const int cur_h = (x < w) ? heights[static_cast<size_t>(x)] : 0;
            while (!stack.empty() && heights[static_cast<size_t>(stack.back())] > cur_h) {
                const int rect_h = heights[static_cast<size_t>(stack.back())];
                stack.pop_back();
                const int left = stack.empty() ? 0 : (stack.back() + 1);
                const int right = x - 1;
                const int rect_w = right - left + 1;
                const int area = rect_w * rect_h;
                if (area > best_area) {
                    best_area = area;
                    best_x1 = left;
                    best_x2 = right + 1;
                    best_y1 = y - rect_h + 1;
                    best_y2 = y + 1;
                }
            }
            stack.push_back(x);
        }
    }

    if (best_area <= 0 || mask_x_max <= mask_x_min || mask_y_max <= mask_y_min) {
        return fallbackInsetBBox(bbox);
    }

    // Oval and jagged shout bubbles often have a conservative largest-inscribed-rectangle
    // (~65-70% of mask span). Relax safe_bbox proportions so it covers ~80% (78%-82%) of the
    // eroded bubble_mask interior width & height, while remaining strictly inside the eroded
    // mask at its horizontal and vertical mid-bands.
    const float eroded_w = static_cast<float>(mask_x_max - mask_x_min + 1);
    const float eroded_h = static_cast<float>(mask_y_max - mask_y_min + 1);
    const float target_w = eroded_w * 0.80f;
    const float target_h = eroded_h * 0.80f;

    // Measure actual eroded mask span across the central 50% band so we never cross concave walls
    int band_x1 = mask_x_min;
    int band_x2 = mask_x_max + 1;
    const int cy_mid = (mask_y_min + mask_y_max) / 2;
    const int band_y_half = std::max(1, static_cast<int>(eroded_h * 0.22f));
    for (int y = std::max(mask_y_min, cy_mid - band_y_half);
         y <= std::min(mask_y_max, cy_mid + band_y_half); ++y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        int row_l = -1, row_r = -1;
        for (int x = mask_x_min; x <= mask_x_max; ++x) {
            if (active_mask[row + static_cast<size_t>(x)] != 0) {
                if (row_l < 0) row_l = x;
                row_r = x;
            }
        }
        if (row_l >= 0 && row_r >= row_l) {
            band_x1 = std::max(band_x1, row_l);
            band_x2 = std::min(band_x2, row_r + 1);
        }
    }

    int band_y1 = mask_y_min;
    int band_y2 = mask_y_max + 1;
    const int cx_mid = (mask_x_min + mask_x_max) / 2;
    const int band_x_half = std::max(1, static_cast<int>(eroded_w * 0.22f));
    for (int x = std::max(mask_x_min, cx_mid - band_x_half);
         x <= std::min(mask_x_max, cx_mid + band_x_half); ++x) {
        int col_t = -1, col_b = -1;
        for (int y = mask_y_min; y <= mask_y_max; ++y) {
            if (active_mask[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] != 0) {
                if (col_t < 0) col_t = y;
                col_b = y;
            }
        }
        if (col_t >= 0 && col_b >= col_t) {
            band_y1 = std::max(band_y1, col_t);
            band_y2 = std::min(band_y2, col_b + 1);
        }
    }

    const float center_x = 0.5f * static_cast<float>(mask_x_min + mask_x_max + 1);
    const float center_y = 0.5f * static_cast<float>(mask_y_min + mask_y_max + 1);

    float fx1 = std::min(static_cast<float>(best_x1), center_x - 0.5f * target_w);
    float fx2 = std::max(static_cast<float>(best_x2), center_x + 0.5f * target_w);
    float fy1 = std::min(static_cast<float>(best_y1), center_y - 0.5f * target_h);
    float fy2 = std::max(static_cast<float>(best_y2), center_y + 0.5f * target_h);

    // Clamp strictly inside the eroded mask bounds and central band limits
    const float lim_x1 = static_cast<float>((band_x2 > band_x1 + 4) ? band_x1 : mask_x_min);
    const float lim_x2 = static_cast<float>((band_x2 > band_x1 + 4) ? band_x2 : (mask_x_max + 1));
    const float lim_y1 = static_cast<float>((band_y2 > band_y1 + 4) ? band_y1 : mask_y_min);
    const float lim_y2 = static_cast<float>((band_y2 > band_y1 + 4) ? band_y2 : (mask_y_max + 1));

    fx1 = std::clamp(fx1, lim_x1, lim_x2 - 1.0f);
    fx2 = std::clamp(fx2, fx1 + 1.0f, lim_x2);
    fy1 = std::clamp(fy1, lim_y1, lim_y2 - 1.0f);
    fy2 = std::clamp(fy2, fy1 + 1.0f, lim_y2);

    BBox safe{
        static_cast<float>(mask.offset_x) + fx1,
        static_cast<float>(mask.offset_y) + fy1,
        static_cast<float>(mask.offset_x) + fx2,
        static_cast<float>(mask.offset_y) + fy2
    };

    // Clamp safely inside the outer bbox.
    safe.x1 = std::max(bbox.x1, safe.x1);
    safe.y1 = std::max(bbox.y1, safe.y1);
    safe.x2 = std::min(bbox.x2, safe.x2);
    safe.y2 = std::min(bbox.y2, safe.y2);

    if (!safe.is_valid()) {
        return fallbackInsetBBox(bbox);
    }
    return safe;
}

void associateBubblesWithPanels(
    std::vector<BubbleDetection>& detections,
    const std::vector<PanelDetection>& panels) noexcept {
    for (auto& det : detections) {
        int best_panel_id = -1;
        float best_ioa = 0.0f;
        int significant_overlaps = 0;

        for (const auto& panel : panels) {
            const float ioa = computeIoA(det.bbox, panel.bbox);
            if (ioa >= kPanelIoACrossThreshold) {
                ++significant_overlaps;
            }
            if (ioa > best_ioa) {
                best_ioa = ioa;
                best_panel_id = panel.id;
            }
        }

        det.is_cross_panel = (significant_overlaps >= 2);
        det.panel_id = (best_ioa >= kPanelIoAAssignThreshold) ? best_panel_id : -1;
    }
}

void sortReadingOrder(
    std::vector<PanelDetection>& panels,
    std::vector<BubbleDetection>& detections,
    ReadingDirection direction) {
    // 1. Sort panels via cycle-safe DAG topological sort.
    if (!panels.empty()) {
        std::vector<DagNode> panel_nodes;
        panel_nodes.reserve(panels.size());
        for (size_t i = 0; i < panels.size(); ++i) {
            panel_nodes.push_back(DagNode{
                i,
                panels[i].bbox,
                Point2f{panels[i].bbox.cx(), panels[i].bbox.cy()}
            });
        }

        const std::vector<size_t> panel_order = topologicalSortDag(panel_nodes, direction);
        std::vector<PanelDetection> sorted_panels;
        sorted_panels.reserve(panels.size());
        for (size_t rank = 0; rank < panel_order.size(); ++rank) {
            PanelDetection p = std::move(panels[panel_order[rank]]);
            p.reading_order = static_cast<int>(rank);
            sorted_panels.push_back(std::move(p));
        }
        panels = std::move(sorted_panels);
    }

    if (detections.empty()) {
        return;
    }

    // 2. Partition detections by panel_id.
    std::unordered_map<int, std::vector<size_t>> by_panel;
    std::vector<size_t> unassociated;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (detections[i].panel_id >= 0) {
            by_panel[detections[i].panel_id].push_back(i);
        } else {
            unassociated.push_back(i);
        }
    }

    auto sortGroup = [&detections, direction](const std::vector<size_t>& indices) {
        std::vector<DagNode> nodes;
        nodes.reserve(indices.size());
        for (size_t idx : indices) {
            nodes.push_back(DagNode{
                idx,
                detections[idx].bbox,
                computeMaskCentroid(detections[idx])
            });
        }
        const std::vector<size_t> local_order = topologicalSortDag(nodes, direction);
        std::vector<size_t> result;
        result.reserve(local_order.size());
        for (size_t k : local_order) {
            result.push_back(indices[k]);
        }
        return result;
    };

    std::vector<size_t> final_indices;
    final_indices.reserve(detections.size());

    // Emit detections panel-by-panel in sorted panel reading_order.
    for (const auto& panel : panels) {
        auto it = by_panel.find(panel.id);
        if (it != by_panel.end()) {
            auto sorted_group = sortGroup(it->second);
            final_indices.insert(final_indices.end(), sorted_group.begin(), sorted_group.end());
            by_panel.erase(it);
        }
    }

    // Fallback for any panel_id not present in `panels`.
    for (const auto& kv : by_panel) {
        auto sorted_group = sortGroup(kv.second);
        final_indices.insert(final_indices.end(), sorted_group.begin(), sorted_group.end());
    }

    // Order unassociated bubbles globally.
    if (!unassociated.empty()) {
        auto sorted_unassoc = sortGroup(unassociated);
        final_indices.insert(final_indices.end(), sorted_unassoc.begin(), sorted_unassoc.end());
    }

    std::vector<BubbleDetection> sorted_detections;
    sorted_detections.reserve(detections.size());
    for (size_t rank = 0; rank < final_indices.size(); ++rank) {
        BubbleDetection d = std::move(detections[final_indices[rank]]);
        d.reading_order = static_cast<int>(rank);
        sorted_detections.push_back(std::move(d));
    }
    detections = std::move(sorted_detections);
}

void runGeometryPostproc(
    PageDetectionResult& result,
    const DetectorConfig& config) {
    filterSfxAndMicroArtifacts(result.detections, result.image_width, result.image_height, config);

    // Only compute mask-inscribed safe_bbox when the backend produces solid interior
    // bubble masks (YOLO11m-Seg). RF-DETR-Seg / Legacy CTD store thin text-stroke
    // inpainting masks in `det.mask`, so running largest-inscribed-rectangle over them
    // would collapse `safe_bbox` to a tiny stroke fragment.
    const bool has_solid_bubble_mask =
        (result.runtime_meta.backend == BackendType::Yolo11Seg);
    for (auto& det : result.detections) {
        if (has_solid_bubble_mask) {
            det.safe_bbox = computeSafeBBox(det.bbox, det.mask);
        } else if (!det.safe_bbox.is_valid()) {
            const float bw = det.bbox.width();
            const float bh = det.bbox.height();
            const float pad_x = bw * 0.08f;
            const float pad_y = bh * 0.08f;
            BBox inset{
                det.bbox.x1 + pad_x,
                det.bbox.y1 + pad_y,
                det.bbox.x2 - pad_x,
                det.bbox.y2 - pad_y
            };
            det.safe_bbox = inset.is_valid() ? inset : det.bbox;
        }

        // Clamp safe_bbox strictly inside det.bbox with at least 6px inset from
        // black bubble outline walls when the bubble is large enough.
        const float min_inset = (det.bbox.width() > 16.0f && det.bbox.height() > 16.0f) ? 6.0f : 2.0f;
        const float lim_x1 = det.bbox.x1 + min_inset;
        const float lim_y1 = det.bbox.y1 + min_inset;
        const float lim_x2 = det.bbox.x2 - min_inset;
        const float lim_y2 = det.bbox.y2 - min_inset;
        if (lim_x2 > lim_x1 && lim_y2 > lim_y1) {
            BBox clamped{
                std::clamp(det.safe_bbox.x1, lim_x1, lim_x2 - 1.0f),
                std::clamp(det.safe_bbox.y1, lim_y1, lim_y2 - 1.0f),
                std::clamp(det.safe_bbox.x2, lim_x1 + 1.0f, lim_x2),
                std::clamp(det.safe_bbox.y2, lim_y1 + 1.0f, lim_y2)
            };
            if (clamped.is_valid()) {
                det.safe_bbox = clamped;
            }
        }
    }

    associateBubblesWithPanels(result.detections, result.panels);
    sortReadingOrder(result.panels, result.detections, result.reading_direction);
}

std::vector<uint8_t> buildTextStrokeInpaintMask(
    const uint8_t* rgbImage,
    int width,
    int height,
    const std::vector<BubbleDetection>& detections,
    const std::vector<uint8_t>* optionalCtdStrokeMask) {
    if (!rgbImage || width <= 0 || height <= 0) {
        return {};
    }

    const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<uint8_t> out_mask(total, 0);

    auto lumaAt = [rgbImage, width, height](int x, int y) noexcept -> int {
        const int cx = std::clamp(x, 0, width - 1);
        const int cy = std::clamp(y, 0, height - 1);
        const size_t off = (static_cast<size_t>(cy) * static_cast<size_t>(width) + static_cast<size_t>(cx)) * 3ULL;
        const int r = rgbImage[off + 0];
        const int g = rgbImage[off + 1];
        const int b = rgbImage[off + 2];
        return (299 * r + 587 * g + 114 * b + 500) / 1000;
    };

    static constexpr int kDx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static constexpr int kDy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    for (const auto& det : detections) {
        if (!is_dialogue_class(det.class_id)) {
            continue;
        }
        if (!det.bbox.is_valid()) {
            continue;
        }

        // Outer ROI spanned by det.bbox
        const int rx1 = std::clamp(static_cast<int>(std::floor(det.bbox.x1)), 0, width - 1);
        const int ry1 = std::clamp(static_cast<int>(std::floor(det.bbox.y1)), 0, height - 1);
        const int rx2 = std::clamp(static_cast<int>(std::ceil(det.bbox.x2)), rx1 + 1, width);
        const int ry2 = std::clamp(static_cast<int>(std::ceil(det.bbox.y2)), ry1 + 1, height);

        const int rw = rx2 - rx1;
        const int rh = ry2 - ry1;
        if (rw < 8 || rh < 8) {
            continue;
        }

        const size_t roi_total = static_cast<size_t>(rw) * static_cast<size_t>(rh);

        // 1. Construct the valid interior erase zone from the full segmentation InstanceMask
        //    (`bubble_mask`) eroded inward by 3px (protecting the black ink border of the bubble
        //    while covering 100% of top/bottom tapered tips and side columns outside safe_bbox).
        //    If InstanceMask is empty or sparse, fall back to det.bbox inset by 4px.
        std::vector<uint8_t> interior_zone(roi_total, 0);
        int zone_pixels = 0;

        const bool has_valid_mask =
            !det.mask.empty() &&
            det.mask.data.size() >= static_cast<size_t>(det.mask.width) * static_cast<size_t>(det.mask.height) &&
            effectiveMaskAreaPx(det) >= 0.25 * static_cast<double>(det.bbox.area());

        if (has_valid_mask) {
            std::vector<uint8_t> raw_bubble_mask(roi_total, 0);
            for (int ly = 0; ly < rh; ++ly) {
                const int gy = ry1 + ly;
                const int my = gy - det.mask.offset_y;
                if (my < 0 || my >= det.mask.height) continue;
                const size_t mrow = static_cast<size_t>(my) * static_cast<size_t>(det.mask.width);
                const size_t rrow = static_cast<size_t>(ly) * static_cast<size_t>(rw);
                for (int lx = 0; lx < rw; ++lx) {
                    const int gx = rx1 + lx;
                    const int mx = gx - det.mask.offset_x;
                    if (mx < 0 || mx >= det.mask.width) continue;
                    if (det.mask.data[mrow + static_cast<size_t>(mx)] != 0) {
                        raw_bubble_mask[rrow + static_cast<size_t>(lx)] = 1;
                    }
                }
            }

            // Erode bubble_mask inward by 3px (circular radius=3)
            constexpr int kErodeR = 3;
            constexpr int kErodeR2 = kErodeR * kErodeR;
            for (int ly = kErodeR; ly < rh - kErodeR; ++ly) {
                const size_t rrow = static_cast<size_t>(ly) * static_cast<size_t>(rw);
                for (int lx = kErodeR; lx < rw - kErodeR; ++lx) {
                    if (raw_bubble_mask[rrow + static_cast<size_t>(lx)] == 0) {
                        continue;
                    }
                    bool all_inside = true;
                    for (int dy = -kErodeR; dy <= kErodeR && all_inside; ++dy) {
                        const size_t nrow = static_cast<size_t>(ly + dy) * static_cast<size_t>(rw);
                        for (int dx = -kErodeR; dx <= kErodeR; ++dx) {
                            if (dx * dx + dy * dy > kErodeR2) continue;
                            if (raw_bubble_mask[nrow + static_cast<size_t>(lx + dx)] == 0) {
                                all_inside = false;
                                break;
                            }
                        }
                    }
                    if (all_inside) {
                        interior_zone[rrow + static_cast<size_t>(lx)] = 1;
                        ++zone_pixels;
                    }
                }
            }
        }

        // Fallback if InstanceMask is empty or eroded away: use det.bbox inset by 4px
        if (zone_pixels < 16) {
            std::fill(interior_zone.begin(), interior_zone.end(), 0);
            zone_pixels = 0;
            const int inset = (rw > 12 && rh > 12) ? 4 : 2;
            for (int ly = inset; ly < rh - inset; ++ly) {
                const size_t rrow = static_cast<size_t>(ly) * static_cast<size_t>(rw);
                for (int lx = inset; lx < rw - inset; ++lx) {
                    interior_zone[rrow + static_cast<size_t>(lx)] = 1;
                    ++zone_pixels;
                }
            }
        }

        if (zone_pixels < 16) {
            continue;
        }

        // 2. Compute local background median luma inside the eroded bubble interior
        std::array<int, 256> hist{};
        for (int ly = 0; ly < rh; ++ly) {
            const int gy = ry1 + ly;
            const size_t rrow = static_cast<size_t>(ly) * static_cast<size_t>(rw);
            for (int lx = 0; lx < rw; ++lx) {
                if (interior_zone[rrow + static_cast<size_t>(lx)] == 0) continue;
                ++hist[static_cast<size_t>(lumaAt(rx1 + lx, gy))];
            }
        }
        int accum = 0;
        int bg_median = 255;
        const int half = zone_pixels / 2;
        for (int b = 0; b < 256; ++b) {
            accum += hist[static_cast<size_t>(b)];
            if (accum >= half) {
                bg_median = b;
                break;
            }
        }

        // 3. Detect all dark text strokes (luma < 145 or |luma - bg_median| > 40 on light backgrounds,
        //    plus inverted light text luma > 180 on dark boxes) across the ENTIRE eroded bubble interior,
        //    and union any optional CTD text-stroke mask inside the same eroded interior.
        const bool dark_bg = (bg_median < 128);
        std::vector<uint8_t> raw_strokes(roi_total, 0);

        for (int ly = 0; ly < rh; ++ly) {
            const int gy = ry1 + ly;
            const size_t row_off = static_cast<size_t>(ly) * static_cast<size_t>(rw);
            for (int lx = 0; lx < rw; ++lx) {
                const size_t idx = row_off + static_cast<size_t>(lx);
                if (interior_zone[idx] == 0) {
                    continue;
                }
                const int gx = rx1 + lx;
                const int luma = lumaAt(gx, gy);
                const int contrast = std::abs(luma - bg_median);
                bool is_stroke = false;
                if (dark_bg) {
                    is_stroke = (luma > 180 && contrast > 40);
                } else {
                    is_stroke = (luma < 145) || (luma < bg_median && contrast > 40);
                }
                if (!is_stroke && optionalCtdStrokeMask && optionalCtdStrokeMask->size() == total) {
                    if ((*optionalCtdStrokeMask)[static_cast<size_t>(gy) * static_cast<size_t>(width) + static_cast<size_t>(gx)] > 0) {
                        is_stroke = true;
                    }
                }
                if (is_stroke) {
                    raw_strokes[idx] = 1;
                }
            }
        }

        // 4. Connected-component filtering:
        //    - Filter out tiny 1-4px screentone noise dots (comp.size() <= 4).
        //    - Protect bubble outline segments that may protrude slightly into the eroded zone:
        //      only drop a boundary-touching component if it is a long wall contour touching
        //      the eroded perimeter for many pixels, while keeping characters/punctuation
        //      (e.g. た, ます, だ, す, いや, ！！) near the tapered tips or side columns.
        std::vector<uint8_t> visited(roi_total, 0);
        std::vector<uint8_t> clean_strokes(roi_total, 0);
        std::vector<int> comp;
        std::vector<int> queue;

        for (int sy = 0; sy < rh; ++sy) {
            for (int sx = 0; sx < rw; ++sx) {
                const size_t start = static_cast<size_t>(sy) * static_cast<size_t>(rw) + static_cast<size_t>(sx);
                if (visited[start] || raw_strokes[start] == 0) {
                    continue;
                }

                comp.clear();
                queue.clear();
                visited[start] = 1;
                queue.push_back(static_cast<int>(start));
                int boundary_contacts = 0;
                int min_cx = sx, max_cx = sx, min_cy = sy, max_cy = sy;

                size_t q_head = 0;
                while (q_head < queue.size()) {
                    const int cur = queue[q_head++];
                    comp.push_back(cur);
                    const int cx = cur % rw;
                    const int cy = cur / rw;
                    min_cx = std::min(min_cx, cx);
                    max_cx = std::max(max_cx, cx);
                    min_cy = std::min(min_cy, cy);
                    max_cy = std::max(max_cy, cy);

                    bool on_zone_edge = false;
                    for (int k = 0; k < 8; ++k) {
                        const int nx = cx + kDx[k];
                        const int ny = cy + kDy[k];
                        if (nx < 0 || nx >= rw || ny < 0 || ny >= rh) {
                            on_zone_edge = true;
                            continue;
                        }
                        const size_t nidx = static_cast<size_t>(ny) * static_cast<size_t>(rw) + static_cast<size_t>(nx);
                        if (interior_zone[nidx] == 0) {
                            on_zone_edge = true;
                        } else if (!visited[nidx] && raw_strokes[nidx] != 0) {
                            visited[nidx] = 1;
                            queue.push_back(static_cast<int>(nidx));
                        }
                    }
                    if (on_zone_edge) {
                        ++boundary_contacts;
                    }
                }

                // Drop tiny 1-4px screentone noise dots
                if (comp.size() <= 4) {
                    continue;
                }

                // If a component hugs the eroded perimeter across a large span (>45% of bubble width/height
                // AND >35% of its pixels are on the perimeter edge), it is part of the bubble border wall,
                // not an interior character.
                const int comp_w = max_cx - min_cx + 1;
                const int comp_h = max_cy - min_cy + 1;
                const bool is_border_wall_arc =
                    (boundary_contacts > 24) &&
                    (static_cast<double>(boundary_contacts) / static_cast<double>(comp.size()) > 0.35) &&
                    (comp_w > rw / 2 || comp_h > rh / 2);
                if (is_border_wall_arc) {
                    continue;
                }

                for (int idx : comp) {
                    clean_strokes[static_cast<size_t>(idx)] = 1;
                }
            }
        }

        // 5. Dilate clean text strokes by 4px (circular radius=4), clamped strictly inside
        //    the 3px-eroded bubble_mask interior (`interior_zone`) so every character,
        //    dakuten (゛), and furigana mark is completely covered without touching bubble walls.
        constexpr int kDilateRadius = 4;
        constexpr int kR2 = kDilateRadius * kDilateRadius + 1;
        for (int ly = 0; ly < rh; ++ly) {
            const size_t row_off = static_cast<size_t>(ly) * static_cast<size_t>(rw);
            for (int lx = 0; lx < rw; ++lx) {
                if (clean_strokes[row_off + static_cast<size_t>(lx)] == 0) {
                    continue;
                }
                for (int dy = -kDilateRadius; dy <= kDilateRadius; ++dy) {
                    const int ny = ly + dy;
                    if (ny < 0 || ny >= rh) continue;
                    const int gy = ry1 + ny;
                    const size_t nrow = static_cast<size_t>(ny) * static_cast<size_t>(rw);
                    for (int dx = -kDilateRadius; dx <= kDilateRadius; ++dx) {
                        if (dx * dx + dy * dy > kR2) continue;
                        const int nx = lx + dx;
                        if (nx < 0 || nx >= rw) continue;
                        if (interior_zone[nrow + static_cast<size_t>(nx)] == 0) continue;
                        const int gx = rx1 + nx;
                        if (gx < 0 || gx >= width || gy < 0 || gy >= height) continue;
                        out_mask[static_cast<size_t>(gy) * static_cast<size_t>(width) + static_cast<size_t>(gx)] = 255;
                    }
                }
            }
        }
    }

    return out_mask;
}

}  // namespace vellum::detector
