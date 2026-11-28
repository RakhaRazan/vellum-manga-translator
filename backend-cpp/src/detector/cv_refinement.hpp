#pragma once

// Vellum - Pure C++17 Detection Consolidation & 5-Stage Otsu + Plain-Paper Glyph Mask Refinement
//
// Zero OpenCV / Leptonica external dependencies; operates directly on uint8_t buffers
// in native page pixel coordinates.

#include <algorithm>
#include <cstdint>
#include <vector>

namespace vellum::detector {

/// Axis-aligned integer rectangle in native page pixel coordinates: [x1, x2) x [y1, y2).
struct RectI {
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;

    int width() const noexcept { return std::max(0, x2 - x1); }
    int height() const noexcept { return std::max(0, y2 - y1); }
    int64_t area() const noexcept {
        return static_cast<int64_t>(width()) * static_cast<int64_t>(height());
    }
    bool empty() const noexcept { return x2 <= x1 || y2 <= y1; }

    bool contains(int x, int y) const noexcept {
        return x >= x1 && x < x2 && y >= y1 && y < y2;
    }

    RectI clamped(int page_w, int page_h) const noexcept {
        const int cx1 = std::clamp(x1, 0, page_w);
        const int cy1 = std::clamp(y1, 0, page_h);
        const int cx2 = std::clamp(x2, cx1, page_w);
        const int cy2 = std::clamp(y2, cy1, page_h);
        return RectI{cx1, cy1, cx2, cy2};
    }

    RectI intersected(const RectI& o) const noexcept {
        const int ix1 = std::max(x1, o.x1);
        const int iy1 = std::max(y1, o.y1);
        const int ix2 = std::min(x2, o.x2);
        const int iy2 = std::min(y2, o.y2);
        if (ix2 <= ix1 || iy2 <= iy1) {
            return RectI{0, 0, 0, 0};
        }
        return RectI{ix1, iy1, ix2, iy2};
    }

    RectI united(const RectI& o) const noexcept {
        if (empty()) return o;
        if (o.empty()) return *this;
        return RectI{
            std::min(x1, o.x1),
            std::min(y1, o.y1),
            std::max(x2, o.x2),
            std::max(y2, o.y2)
        };
    }

    RectI expanded(int pad_x, int pad_y, int page_w, int page_h) const noexcept {
        return RectI{x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y}.clamped(page_w, page_h);
    }
};

/// Raw RF-DETR text/onomatopoeia candidate prior to per-query deduplication.
struct RawTextCandidate {
    int query_idx = -1;
    int class_id = 0;  // 0 = text, 1 = onomatopoeia
    float score = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

/// A text or onomatopoeia block entering SFX consolidation and 5-stage mask refinement.
struct RefinementBlock {
    int original_index = 0;
    int query_idx = -1;
    int class_id = 0;          // 0 = text, 1 = onomatopoeia
    float score = 0.0f;
    RectI bbox;                // Core detection bounding box in native page pixels
    bool is_vertical = false;  // True if reading axis is vertical (Y), false if horizontal (X)

    // Input raw binary mask (0 or 255) cropped to raw_mask_rect (row-major).
    RectI raw_mask_rect;
    std::vector<uint8_t> raw_mask;

    // Output refined binary mask (0 or 255) cropped to refined_mask_rect (row-major).
    RectI refined_mask_rect;
    std::vector<uint8_t> refined_mask;
};

/// Tuning parameters for consolidation and 5-stage mask refinement.
struct RefinementConfig {
    float sfx_merge_overlap = 0.50f;
    double reach_radius_1024 = 6.0;
    double l2_margin_1024 = 2.0;
    double otsu_score_fraction = 0.1;
    double plain_paper_ratio = 0.99;
    int plain_paper_rgb_tol = 32;
    int final_grow_px = 3;
};

/// Scale a morphology radius defined at 1024px long side to native page dimensions:
/// scaled(r, W, H) = clamp(round(max(W, H) / 1024.0 * r), 1, 127).
int scaledMorphRadius(double r, int page_w, int page_h) noexcept;

/// Enforce `one_label_per_query` for RF-DETR text/onomatopoeia candidates:
/// on the same `query_idx`, `text` (class_id == 0) always beats `onomatopoeia` (class_id == 1).
/// Same-class duplicates on the same query keep the higher score.
std::vector<RawTextCandidate> enforceOneLabelPerQuery(
    const std::vector<RawTextCandidate>& candidates);

/// Merge overlapping SFX (`class_id == 1`) blocks whenever their shared glyph overlap
/// `inter_px / min(area_a, area_b) >= min_glyph_overlap` (default 0.50).
/// Non-SFX (`class_id == 0`) blocks are preserved unchanged.
void mergeOverlappingSfxBlocks(
    std::vector<RefinementBlock>& blocks,
    float min_glyph_overlap = 0.50f);

/// Single-tile Otsu threshold with `score_fraction` (default 0.1):
/// finds the lowest threshold `t <= t_max` whose between-class variance is
/// `>= (1.0 - score_fraction) * max_variance`.
uint8_t computeOtsuThreshold1Tile(
    const uint8_t* gray,
    int width,
    int height,
    double score_fraction = 0.1) noexcept;

/// Run smallest-box-wins occupancy claiming + 5-stage Otsu & plain-paper glyph mask
/// refinement on all blocks in native page pixel coordinates.
///
/// Populates each block's `refined_mask` and `refined_mask_rect`, and returns the
/// combined full-page binary mask (`page_w * page_h`, values 0 or 255).
std::vector<uint8_t> refineTextMasks(
    const uint8_t* rgb_data,
    int page_w,
    int page_h,
    std::vector<RefinementBlock>& blocks,
    const RefinementConfig& cfg = {});

}  // namespace vellum::detector
