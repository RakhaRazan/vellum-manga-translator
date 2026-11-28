#pragma once

// Vellum Math Wave M1 — Pure C++17, 0 MB VRAM Geometric & Signal Processing Ops
//
// Implements:
//   #1  - O(N) Integral-Image Guided Image Filter for pixel-accurate bubble mask snapping
//   #2  - Radon/Hough White Gutter Panel Detection + Exact Bitmask DP Reading Order (N <= 12)
//   #8  - 1D DP Minimum-Energy Seam Webtoon Tiling
//   #15 - Discrete Curvature + Ray-Casting Speaker Clustering
//   #16 - Anisotropic DBSCAN + Bresenham Line-of-Sight Barrier for fragmented text merging
//   #18 - Opt-In RANSAC + 3x3 Homography Deskew & Lightweight TPS Spine Dewarp

#include "detector_types.hpp"

#include <cstdint>
#include <vector>

namespace vellum::detector {

// -----------------------------------------------------------------------------
// #1 — O(N) Integral-Image Guided Image Filter
// -----------------------------------------------------------------------------
/// Edge-preserving O(N) Guided Image Filter using double-precision integral images.
///   a_k = (mean(I*p) - mu_k * p_bar_k) / (sigma_k^2 + eps)
///   b_k = p_bar_k - a_k * mu_k
///   q   = mean(a) * I + mean(b)
/// `guide_I` and `src_p` must have `width * height` floats in [0, 1].
void applyGuidedFilter(
    const float* guide_I,
    const float* src_p,
    int width,
    int height,
    int radius,
    float eps,
    std::vector<float>& out_q);

// -----------------------------------------------------------------------------
// #2 — Radon/Hough Gutter Panel Detection + Exact Bitmask DP Reading Order
// -----------------------------------------------------------------------------
/// Detect straight and angled white/black page gutters via lightweight downscaled
/// Radon/Hough line projection (rho = x cos(theta) + y sin(theta)) to extract
/// deterministic PanelDetection boxes when the neural detector returns no panels.
std::vector<PanelDetection> detectPanelsRadonHough(
    const uint8_t* rgbData,
    int width,
    int height);

/// Exact Bitmask DP (Minimum Feedback Arc Set) ordering for N <= 12 nodes.
/// Maximizes sum of satisfied directed edge weights w(v -> u):
///   DP[S] = max_{u in S} (DP[S \ {u}] + sum_{v in S \ {u}} w(v -> u))
/// `weights` is a row-major N x N matrix where weights[v * N + u] = w(v -> u).
/// `tie_order` provides a deterministic priority rank (lower = earlier on tie).
/// Returns optimal permutation of indices [0..N-1], or empty if N == 0 or N > 12.
std::vector<size_t> solveReadingOrderBitmaskDP(
    size_t n,
    const std::vector<float>& weights,
    const std::vector<size_t>& tie_order);

// -----------------------------------------------------------------------------
// #8 — 1D DP Minimum-Energy Seam Webtoon Tiling
// -----------------------------------------------------------------------------
/// Compute optimal vertical tile split Y-coordinates [0, y_1, ..., height] for tall
/// Webtoon strips using 1D DP over downscaled row energy:
///   E(y) = mean(|I_x| + |I_y|) + beta * sigma_row(y)
/// minimizing E(y) + gamma * ((y - y' - H_target) / H_target)^2 within [min_h, max_h].
std::vector<int> computeWebtoonSeamSplits(
    const uint8_t* rgbData,
    int width,
    int height,
    int target_h,
    int min_h,
    int max_h,
    float beta = 0.5f,
    float gamma = 4.0f);

// -----------------------------------------------------------------------------
// #15 — Discrete Curvature + Ray-Casting Speaker Clustering
// -----------------------------------------------------------------------------
/// Find sharp outward speech-bubble tail tips (interior angle < 45 deg) from each
/// bubble's `room_polygon` / mask contour, cast rays along outward bisectors within
/// the same panel, and cluster bubbles whose rays converge (or share conjoined necks)
/// into `BubbleDetection::speaker_cluster_id`.
void clusterSpeakersByTailRays(
    std::vector<BubbleDetection>& detections,
    const std::vector<PanelDetection>& panels);

// -----------------------------------------------------------------------------
// #16 — Anisotropic DBSCAN + Bresenham Line-of-Sight Barrier
// -----------------------------------------------------------------------------
/// Check if the integer Bresenham line between (x0, y0) and (x1, y1) crosses a thick
/// dark panel or bubble border ink stroke.
bool hasBresenhamInkBarrier(
    const uint8_t* rgbData,
    int width,
    int height,
    int x0,
    int y0,
    int x1,
    int y1) noexcept;

/// Merge fragmented BorderlessDialogue / TextRegion detections using font-normalized
/// anisotropic DBSCAN distance <= 1.5 with scale ratio in [0.75, 1.35], gated by
/// Bresenham Line-of-Sight ink barrier checks.
void mergeFragmentedTextRegionsDBSCAN(
    std::vector<BubbleDetection>& detections,
    const uint8_t* rgbData,
    int width,
    int height,
    ReadingDirection direction);

// -----------------------------------------------------------------------------
// #18 — Opt-In RANSAC + 3x3 Homography Deskew & Lightweight TPS Spine Dewarp
// -----------------------------------------------------------------------------
struct DeskewResult {
    bool applied = false;
    float skew_deg = 0.0f;
    bool tps_applied = false;
    std::vector<uint8_t> warped_rgb;
};

/// Estimate dominant page skew from panel border segments via RANSAC (valid correction
/// clamped to 0.8 deg <= |theta| <= 6.0 deg), warp via bilinear 3x3 Homography, and
/// apply lightweight Thin-Plate Spline (U(r) = r^2 ln r) only if top/bottom panel
/// border curvature exceeds threshold.
DeskewResult deskewAndDewarpScan(
    const uint8_t* rgbData,
    int width,
    int height);

}  // namespace vellum::detector
