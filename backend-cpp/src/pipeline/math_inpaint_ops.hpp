#pragma once

// Vellum Math Wave M3 — Pure C++17, 0 MB VRAM Inpainting, Blending & Smart Export Ops
//
// Implements:
//   #4A - Tier 2.5 CPU Periodic Screentone Synthesis via 2D Autocorrelation (phase-locked lattice)
//   #4B - Discrete Screened Poisson Blending via Red-Black SOR (omega=1.75, max 60 iters)
//   #12 - Criminisi Isophote-Driven Exemplar + PatchMatch Inpainting (7x7 patches)
//   #21 - Hasler-Susstrunk Colorfulness + Fast Luminance SSIM for Smart Grayscale & Quality Export

#include <cstdint>
#include <vector>

namespace mangaflow::pipeline::math_inpaint {

// -----------------------------------------------------------------------------
// #4A — Tier 2.5 CPU Periodic Screentone Synthesis (2D Autocorrelation Lattice)
// -----------------------------------------------------------------------------

/// Analyze the unmasked exterior ring around `[bx1, by1, bx2, by2]` via normalized
/// 2D autocorrelation over shifts 3..24 px. If a strong periodic peak exists (halftone
/// dots or parallel hatching), extracts lattice basis vector(s) v1, (v2) and fills every
/// hole pixel (x, y) by phase-locked sampling from the nearest valid exterior pixel.
/// Returns true if the hole was completely synthesized on CPU (allowing Tier 3 GPU skip).
bool synthesizePeriodicScreentone(
    uint8_t* canvas_rgb,
    const uint8_t* hole_mask,
    const uint8_t* all_masks,
    int page_w,
    int page_h,
    int bx1,
    int by1,
    int bx2,
    int by2);

// -----------------------------------------------------------------------------
// #4B — Discrete Screened Poisson Blending via Red-Black SOR
// -----------------------------------------------------------------------------

/// Solve 4*f(i,j) - f(i-1,j) - f(i+1,j) - f(i,j-1) - f(i,j+1) = div(v)(i,j)
/// with Dirichlet boundary f|_{dOmega} = dst_rgb|_{dOmega} inside the tight mask bbox
/// using Red-Black Successive Over-Relaxation (omega = 1.75, max 60 iters, tol = 1e-3).
/// Only executes when tight mask bbox <= 512x512; returns true if blended.
bool poissonBlendRegionSOR(
    uint8_t* dst_rgb,
    const uint8_t* src_rgb,
    const uint8_t* mask,
    int width,
    int height,
    int roi_x1,
    int roi_y1,
    int roi_x2,
    int roi_y2,
    float omega = 1.75f,
    int max_iters = 60,
    float tol = 1e-3f);

// -----------------------------------------------------------------------------
// #12 — Criminisi Isophote-Driven Exemplar + PatchMatch Inpainting
// -----------------------------------------------------------------------------

/// Compact CPU Criminisi (P(p) = C(p) * |grad(I)^perp . n_p| / alpha) + PatchMatch
/// exemplar inpainter for small structured non-periodic holes (area <= max_hole_area).
/// When `require_high_coherence` is true, only fires when exterior Structure-Tensor
/// coherence confirms directional linework (speedlines/hatching).
bool inpaintCriminisiPatchMatch(
    uint8_t* canvas_rgb,
    const uint8_t* hole_mask,
    const uint8_t* all_masks,
    int page_w,
    int page_h,
    int bx1,
    int by1,
    int bx2,
    int by2,
    int patch_radius = 3,
    int max_hole_area = 4096,
    bool require_high_coherence = true);

// -----------------------------------------------------------------------------
// #21 — Hasler-Susstrunk Colorfulness + Fast Luminance SSIM
// -----------------------------------------------------------------------------

/// Compute Hasler-Susstrunk colorfulness metric M = sqrt(var_rg + var_yb) + 0.3 * sqrt(mu_rg^2 + mu_yb^2)
/// on a fast stride-`stride` pixel sample. Values M < 8.0 indicate B&W grayscale manga.
float computeHaslerSusstrunkColorfulness(
    const uint8_t* rgb,
    int width,
    int height,
    int stride = 4) noexcept;

/// Convert RGB uint8 buffer to 8-bit grayscale luminance buffer.
void rgbToGray8(
    const uint8_t* rgb,
    int width,
    int height,
    std::vector<uint8_t>& out_gray);

/// Compute fast mean blockwise luminance SSIM in [0, 1] between two 8-bit grayscale
/// images of identical dimensions (subsampled to <= 256px long side for < 1 ms speed).
float computeFastLuminanceSSIM(
    const uint8_t* ref_gray,
    const uint8_t* test_gray,
    int width,
    int height) noexcept;

}  // namespace mangaflow::pipeline::math_inpaint
