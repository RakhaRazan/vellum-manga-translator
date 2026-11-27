#pragma once

// Vellum Math Wave M2 — Pure C++17, 0 MB VRAM OCR & Typography Signal Processing Ops
//
// Implements:
//   #3  - O(1) Integral-Image Sauvola Thresholding + 2D Structure Tensor Halftone Filter
//         + 1D Ink Projection Trim & Symmetric 1:1 Square Padding
//   #9  - Perceptual OKLab Color Space + 2-3 Component EM GMM for Color Manhwa/Webtoon Text
//   #10 - Perona-Malik Anisotropic Diffusion Denoiser (gated by noise stddev, 0 ms on clean scans)
//   #11 - Stroke Width Transform (SWT) Opt-In Furigana Masking (temporary OCR crop only)
//   #17 - Compact Japanese Visual Confusable Markov Prior for Beam Search (B=3)

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mangaflow::pipeline::math_ocr {

// -----------------------------------------------------------------------------
// #3 — O(1) Integral-Image Sauvola + 2D Structure Tensor + 1:1 Square Padding
// -----------------------------------------------------------------------------

/// O(1) per-pixel Sauvola adaptive thresholding using double-precision integral images
/// of I and I^2: T = mu * (1 + k * (sigma / R - 1)).
/// Output `out_mask` has size `width * height` (1 = ink, 0 = background).
void computeSauvolaMask(
    const uint8_t* gray,
    int width,
    int height,
    std::vector<uint8_t>& out_mask,
    int window = 15,
    float k = 0.2f,
    float R = 128.0f,
    bool light_on_dark = false);

/// Compute 2D Structure Tensor J anisotropy coherence C = ((l1 - l2) / (l1 + l2 + eps))^2
/// per pixel in O(N) using integral-image box smoothing (radius = 2).
void computeStructureTensorCoherence(
    const uint8_t* gray,
    int width,
    int height,
    std::vector<float>& out_coherence);

/// Filter isotropic halftone screentone dots (low structure tensor coherence) out of a
/// binary ink mask while preserving directional text strokes (high coherence).
void filterHalftoneDotsByStructureTensor(
    const uint8_t* gray,
    int width,
    int height,
    std::vector<uint8_t>& ink_mask,
    float min_coherence = 0.14f);

/// Blank pixels outside an optional bubble polygon (in crop-local coords), trim excess
/// whitespace via 1D ink projection, and pad symmetrically to a 1:1 square
/// (`side = max(trimmed_w, trimmed_h) + pad_margin`) using median background RGB.
struct SquarePaddedRgb {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> data;
};

SquarePaddedRgb trimAndSquarePadRgb(
    const uint8_t* rgb,
    int width,
    int height,
    const std::vector<std::array<int, 2>>* local_polygon = nullptr,
    int pad_margin = 16);

// -----------------------------------------------------------------------------
// #9 — Perceptual OKLab + 2-3 Component EM GMM
// -----------------------------------------------------------------------------

struct OklabColor {
    float L = 0.0f;
    float a = 0.0f;
    float b = 0.0f;
};

OklabColor srgbToOklab(uint8_t r, uint8_t g, uint8_t b) noexcept;
float oklabDeltaE(const OklabColor& c1, const OklabColor& c2) noexcept;

struct OklabGmmResult {
    bool used_gmm = false;               // True if ROI had sufficient chroma variance for OKLab GMM
    std::vector<uint8_t> ink_mask;       // width * height (1 = text ink, 0 = bg/outline)
    std::array<uint8_t, 3> text_rgb{0, 0, 0};
    std::array<uint8_t, 3> bg_rgb{255, 255, 255};
    bool has_outline_cluster = false;
    std::array<uint8_t, 3> outline_rgb{255, 255, 255};
};

/// Run 6-iteration 2-3 component diagonal EM GMM in perceptual OKLab space on a colored
/// bubble ROI. Returns `used_gmm = false` immediately on standard B&W manga bubbles.
OklabGmmResult segmentColorBubbleOklabGmm(
    const uint8_t* page_rgb,
    int page_w,
    int page_h,
    int roi_x1,
    int roi_y1,
    int roi_x2,
    int roi_y2,
    int core_x1,
    int core_y1,
    int core_x2,
    int core_y2);

// -----------------------------------------------------------------------------
// #10 — Perona-Malik Anisotropic Diffusion Denoiser
// -----------------------------------------------------------------------------

/// Estimate background noise stddev; if above `noise_thresh`, apply `iters` iterations
/// of discrete Perona-Malik anisotropic diffusion (g(s) = 1 / (1 + (s / K)^2), K = 18)
/// in-place on `rgb`. Returns true if diffusion was applied (0 ms on clean digital scans).
bool denoiseCropPeronaMalikIfNoisy(
    uint8_t* rgb,
    int width,
    int height,
    float noise_thresh = 8.5f,
    float K = 18.0f,
    int iters = 3);

// -----------------------------------------------------------------------------
// #11 — Stroke Width Transform (SWT) Opt-In Furigana Masking
// -----------------------------------------------------------------------------

/// For Japanese vertical OCR crops, run fast ray-cast SWT on ink edges to identify
/// thin small right-adjacent ruby (furigana) components (w_furi <= 0.55 * w_body and
/// h_furi <= 0.45 * h_body) and mask them with local background RGB in-place.
/// Returns the number of masked furigana components.
int maskFuriganaSwtInCrop(
    uint8_t* rgb,
    int width,
    int height);

// -----------------------------------------------------------------------------
// #17 — Compact Japanese Visual Confusable Markov Prior
// -----------------------------------------------------------------------------

/// Compute log-probability prior adjustment for transitioning `prev_tok -> cand_tok`
/// on high-frequency Japanese visual confusable pairs:
///   ソ/ン, シ/ツ, ー/一, 口/ロ, 力/カ, 工/エ
/// based on surrounding script continuity (Katakana vs Kanji/Hiragana).
float jaConfusableTransitionBonus(
    std::string_view prev_tok,
    std::string_view cand_tok) noexcept;

/// Bidirectional sequence-level script-continuity score for final beam reranking.
float jaConfusableSequenceBonus(const std::vector<std::string_view>& tokens) noexcept;

}  // namespace mangaflow::pipeline::math_ocr
