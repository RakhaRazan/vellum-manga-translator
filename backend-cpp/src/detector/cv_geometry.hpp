#pragma once

// Vellum - Pure C++17 Conjoined Balloon Waist-Cutting, Watershed & Panel-First Reading Order
//
// Zero external CV dependencies; operates directly on uint8_t buffers and integer/float geometry:
//   1. Conjoined Balloon Waist-Cutting (convex hull + convexity defects + chord separation)
//      + Squared-Euclidean Distance-Transform Watershed + RDP Polygon Simplification (<= 32 pts)
//   2. Classic CV Panel Fallback (<5 ms gutter projection + rectangular contour extraction)
//   3. Panel-First Reading Order (90% nesting, 25% overlap / 50% vertical alignment,
//      recursive H/V guillotine cuts, and progressive-erosion pairwise tournament fallback)

#include "cv_refinement.hpp"
#include "detector_types.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace vellum::detector {

/// Input text block for waist-cutting and watershed room partitioning.
struct RoomSeedBlock {
    int block_index = 0;
    RectI bbox;
    // Optional seed pixels (native page coordinates) used for 90% side test and watershed EDT.
    // If empty, sampled points inside `bbox` are used.
    std::vector<std::array<int, 2>> seed_pixels;
};

/// Simplify a closed 2D polygon using Ramer-Douglas-Peucker with
/// `epsilon = eps_ratio * perimeter`, capped at `max_vertices` (default 32).
std::vector<Point2f> simplifyPolygonRDP(
    const std::vector<Point2f>& closed_poly,
    double eps_ratio = 0.001,
    int max_vertices = 32);

/// Partition a single bubble contour / binary mask enclosing >= 1 text blocks into disjoint
/// simplified `room_polygon`s (one per block in `seeds`, <= 32 vertices each):
///   - When `enable_waist_cut` is true and `seeds.size() >= 2`:
///     computes convex hull & convexity defects (depth >= 0.04 * sqrt(Area)),
///     tests candidate chords sorted by `chord_length / sqrt(min(A1, A_total - A1))`
///     requiring >= 90% of each block's seeds on opposite sides, and recursively splits.
///   - Any remaining multi-text room is partitioned via squared-Euclidean EDT watershed
///     with smallest-box priority.
std::vector<std::vector<Point2f>> partitionBubbleRooms(
    const uint8_t* bubble_mask,
    const RectI& mask_rect,
    const std::vector<RoomSeedBlock>& seeds,
    bool enable_waist_cut = true,
    double rdp_eps_ratio = 0.001,
    int max_vertices = 32);

/// Build per-block `room_polygon`s for all dialogue/text blocks on a page by flood-filling
/// bright/plain bubble interiors (or using enclosing speech-bubble regions) and running
/// waist-cutting + watershed on any bubble enclosing >= 2 text blocks.
std::vector<std::vector<Point2f>> computePageRoomPolygons(
    const uint8_t* rgb_data,
    int page_w,
    int page_h,
    const std::vector<RoomSeedBlock>& text_blocks,
    const std::vector<RectI>& bubble_regions,
    const std::vector<uint8_t>* text_mask,
    bool enable_waist_cut = true);

/// Fast (<5 ms) pure-C++ classic CV panel finder (white/black gutter projection +
/// rectangular contour extraction) used as fallback when YOLO11m-Seg or CTD provides no panels.
std::vector<RectI> detectPanelsClassicCV(
    const uint8_t* rgb_data,
    int page_w,
    int page_h);

/// Panel-first reading order sorting for text blocks:
///   - Nests panels with >= 90% containment.
///   - Assigns text blocks to panels (>= 25% overlap or >= 50% vertical alignment beside nearest panel).
///   - Sorts panels and blocks via recursive horizontal/vertical guillotine cuts (`rtl` true for ja,
///     false for ko/zh), falling back to pairwise tournament with progressive box erosion
///     (0%, 10%, 20%, 30%, 40%, 50%) for diagonal/L-shaped layouts.
/// Returns permutation indices into `text_boxes` in reading order.
std::vector<size_t> sortReadingOrderPanelFirst(
    const std::vector<RectI>& text_boxes,
    const std::vector<RectI>& panel_boxes,
    bool rtl);

}  // namespace vellum::detector
