#pragma once

// Vellum - Pure C++17 Zero-VRAM Typography Metadata Extractor
//
// Linear O(W * H) CPU algorithms with zero external CV dependencies:
//   1. Main-Body Component Filter (filters out tiny furigana & specks)
//   2. Trig-Free Sobel 4-theta Tilt & Orientation Estimator
//   3. Stroke Width (Area + 3-4 Chamfer DT) & Rotated Projection Font Size
//   4. Core Ink Color & Felzenszwalb-Huttenlocher EDT Distance-Ring Outline Detector

#include "cv_refinement.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace vellum::detector {

/// Detected text stroke outline metadata.
struct OutlineStyle {
    std::array<uint8_t, 3> color{255, 255, 255};
    int width = 0;  // Estimated outline width in pixels (1..max_ring)
};

/// Extracted per-block typography & styling metadata.
struct TypographyStyle {
    float angle_deg = 0.0f;                      // Tilt angle in degrees [-45, 45], 0 when suppressed
    float coherence_r4 = 0.0f;                   // Sobel 4-theta coherence R_4 in [0, 1]
    float stroke_width = 0.0f;                   // Estimated stroke width in pixels
    float estimated_font_size = 0.0f;            // Estimated single-line font size in pixels
    std::array<uint8_t, 3> text_color{0, 0, 0};  // Median core ink RGB [r, g, b]
    std::optional<OutlineStyle> outline;         // Present when a distinct outline ring is detected
    bool valid = false;
};

/// Linear-time O(W * H) Felzenszwalb-Huttenlocher 2D Exact Squared Euclidean Distance Transform.
/// Computes squared Euclidean distance from every pixel to the nearest foreground pixel (`fg != 0`).
/// Output `dist_sq` has size `width * height`. Returns false if `fg` has no foreground pixels.
bool computeEuclideanDistanceSqFH(
    const uint8_t* fg,
    int width,
    int height,
    std::vector<float>& dist_sq);

/// Extract typography metadata from explicit native-resolution `ink_mask` and `block_mask`.
TypographyStyle analyzeBlockTypography(
    const uint8_t* rgb_data,
    int page_w,
    int page_h,
    const RectI& bbox,
    bool is_vertical,
    bool is_onomatopoeia,
    const RectI& ink_mask_rect,
    const std::vector<uint8_t>& ink_mask,
    const RectI& block_mask_rect,
    const std::vector<uint8_t>& block_mask);

/// Convenience entry point when only RGB image, core `bbox`, and optional seed/refined mask
/// are available: computes Otsu ink mask + solid `block_mask` locally and runs full analysis.
TypographyStyle analyzeBlockTypographyFromBox(
    const uint8_t* rgb_data,
    int page_w,
    int page_h,
    const RectI& bbox,
    bool is_vertical,
    bool is_onomatopoeia,
    const RectI* optional_mask_rect = nullptr,
    const std::vector<uint8_t>* optional_mask = nullptr);

}  // namespace vellum::detector
