#include "math_inpaint_ops.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace mangaflow::pipeline::math_inpaint {

namespace {

inline int clampi(int v, int lo, int hi) noexcept {
    return std::max(lo, std::min(hi, v));
}

inline uint8_t rgbToLuma(uint8_t r, uint8_t g, uint8_t b) noexcept {
    return static_cast<uint8_t>((77 * static_cast<int>(r) +
                                 150 * static_cast<int>(g) +
                                 29 * static_cast<int>(b)) >> 8);
}

inline float computeExteriorCoherence(
    const uint8_t* gray,
    const uint8_t* valid,
    int w,
    int h) noexcept {
    double sum_c = 0.0;
    int cnt = 0;
    for (int y = 2; y + 2 < h; ++y) {
        const size_t rc = static_cast<size_t>(y) * static_cast<size_t>(w);
        for (int x = 2; x + 2 < w; ++x) {
            const size_t idx = rc + static_cast<size_t>(x);
            if (!valid[idx]) continue;
            // Local 3x3 structure tensor over valid neighbors
            double jxx = 0.0, jyy = 0.0, jxy = 0.0;
            int n_ok = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                const size_t r0 = static_cast<size_t>(y + dy - 1) * static_cast<size_t>(w);
                const size_t r1 = static_cast<size_t>(y + dy) * static_cast<size_t>(w);
                const size_t r2 = static_cast<size_t>(y + dy + 1) * static_cast<size_t>(w);
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx;
                    const size_t nidx = r1 + static_cast<size_t>(nx);
                    if (!valid[nidx]) continue;
                    const double gx = 0.5 * (static_cast<double>(gray[r1 + static_cast<size_t>(nx + 1)]) -
                                             static_cast<double>(gray[r1 + static_cast<size_t>(nx - 1)]));
                    const double gy = 0.5 * (static_cast<double>(gray[r2 + static_cast<size_t>(nx)]) -
                                             static_cast<double>(gray[r0 + static_cast<size_t>(nx)]));
                    jxx += gx * gx;
                    jyy += gy * gy;
                    jxy += gx * gy;
                    ++n_ok;
                }
            }
            if (n_ok < 5) continue;
            const double tr = jxx + jyy;
            if (tr < 200.0) continue;  // Skip flat regions
            const double diff = jxx - jyy;
            const double disc = std::sqrt(diff * diff + 4.0 * jxy * jxy);
            const double ratio = disc / (tr + 1e-3);
            sum_c += ratio * ratio;
            ++cnt;
        }
    }
    return (cnt >= 16) ? static_cast<float>(sum_c / static_cast<double>(cnt)) : 0.0f;
}

}  // namespace

// =============================================================================
// #4A — Tier 2.5 CPU Periodic Screentone Synthesis (2D Autocorrelation Lattice)
// =============================================================================

bool synthesizePeriodicScreentone(
    uint8_t* canvas_rgb,
    const uint8_t* hole_mask,
    const uint8_t* all_masks,
    int page_w,
    int page_h,
    int bx1,
    int by1,
    int bx2,
    int by2) {
    if (!canvas_rgb || !hole_mask || !all_masks || page_w <= 0 || page_h <= 0) {
        return false;
    }
    bx1 = clampi(bx1, 0, page_w);
    by1 = clampi(by1, 0, page_h);
    bx2 = clampi(bx2, bx1, page_w);
    by2 = clampi(by2, by1, page_h);
    if (bx2 <= bx1 || by2 <= by1) return false;

    constexpr int kPad = 32;
    const int lx0 = clampi(bx1 - kPad, 0, page_w);
    const int ly0 = clampi(by1 - kPad, 0, page_h);
    const int lx1 = clampi(bx2 + kPad, 0, page_w);
    const int ly1 = clampi(by2 + kPad, 0, page_h);
    const int lw = lx1 - lx0;
    const int lh = ly1 - ly0;
    if (lw < 16 || lh < 16) return false;

    const size_t ltotal = static_cast<size_t>(lw) * static_cast<size_t>(lh);
    std::vector<float> gray(ltotal, 0.0f);
    std::vector<uint8_t> valid(ltotal, 0);
    std::vector<uint8_t> is_hole(ltotal, 0);

    double sum_g = 0.0;
    double sum_g2 = 0.0;
    int valid_cnt = 0;
    int hole_cnt = 0;

    for (int ly = 0; ly < lh; ++ly) {
        const int gy = ly0 + ly;
        const size_t prow = static_cast<size_t>(gy) * static_cast<size_t>(page_w);
        const size_t lrow = static_cast<size_t>(ly) * static_cast<size_t>(lw);
        for (int lx = 0; lx < lw; ++lx) {
            const int gx = lx0 + lx;
            const size_t pidx = prow + static_cast<size_t>(gx);
            const size_t lidx = lrow + static_cast<size_t>(lx);
            const uint8_t* px = canvas_rgb + pidx * 3ULL;
            const float lum = static_cast<float>(rgbToLuma(px[0], px[1], px[2]));
            gray[lidx] = lum;

            const bool in_box = (gx >= bx1 && gx < bx2 && gy >= by1 && gy < by2);
            if (in_box && hole_mask[pidx] > 0) {
                is_hole[lidx] = 1;
                ++hole_cnt;
            } else if (all_masks[pidx] == 0) {
                valid[lidx] = 1;
                sum_g += static_cast<double>(lum);
                sum_g2 += static_cast<double>(lum) * static_cast<double>(lum);
                ++valid_cnt;
            }
        }
    }

    if (hole_cnt == 0 || valid_cnt < 96) {
        return false;
    }

    const double mean_g = sum_g / static_cast<double>(valid_cnt);
    const double var_g = std::max(0.0, (sum_g2 / static_cast<double>(valid_cnt)) - mean_g * mean_g);
    // Screentones / hatching have substantial periodic contrast (std >= 12 luma levels).
    if (var_g < 144.0) {
        return false;
    }

    // Zero-mean normalized grayscale for autocorrelation.
    std::vector<float> zm(ltotal, 0.0f);
    for (size_t i = 0; i < ltotal; ++i) {
        if (valid[i]) {
            zm[i] = gray[i] - static_cast<float>(mean_g);
        }
    }

    // Normalized 2D Autocorrelation over candidate shifts 3 <= hypot(dx, dy) <= 24 px.
    constexpr int kMaxShift = 24;
    constexpr int kMinShiftSq = 9;    // 3^2
    constexpr int kMaxShiftSq = 576;  // 24^2
    const int step = (lw * lh > 160 * 160) ? 2 : 1;

    struct ShiftPeak {
        int dx = 0;
        int dy = 0;
        float corr = 0.0f;
    };
    std::vector<ShiftPeak> peaks;
    peaks.reserve(512);

    double pos_corr_sum = 0.0;
    int pos_corr_cnt = 0;

    for (int dy = 0; dy <= kMaxShift; ++dy) {
        const int dx_start = (dy == 0) ? 3 : -kMaxShift;
        for (int dx = dx_start; dx <= kMaxShift; ++dx) {
            const int d2 = dx * dx + dy * dy;
            if (d2 < kMinShiftSq || d2 > kMaxShiftSq) continue;

            const int y_lo = 0;
            const int y_hi = lh - dy;
            const int x_lo = std::max(0, -dx);
            const int x_hi = std::min(lw, lw - dx);
            if (y_hi <= y_lo || x_hi <= x_lo) continue;

            double num = 0.0;
            double den1 = 0.0;
            double den2 = 0.0;
            int pairs = 0;

            for (int y = y_lo; y < y_hi; y += step) {
                const size_t r1 = static_cast<size_t>(y) * static_cast<size_t>(lw);
                const size_t r2 = static_cast<size_t>(y + dy) * static_cast<size_t>(lw);
                for (int x = x_lo; x < x_hi; x += step) {
                    const size_t i1 = r1 + static_cast<size_t>(x);
                    const size_t i2 = r2 + static_cast<size_t>(x + dx);
                    if (!valid[i1] || !valid[i2]) continue;
                    const double a = static_cast<double>(zm[i1]);
                    const double b = static_cast<double>(zm[i2]);
                    num += a * b;
                    den1 += a * a;
                    den2 += b * b;
                    ++pairs;
                }
            }

            if (pairs < 48 || den1 < 1e-3 || den2 < 1e-3) continue;
            const float corr = static_cast<float>(num / std::sqrt(den1 * den2));
            if (corr > 0.0f) {
                pos_corr_sum += static_cast<double>(corr);
                ++pos_corr_cnt;
            }
            if (corr >= 0.48f) {
                peaks.push_back(ShiftPeak{dx, dy, corr});
            }
        }
    }

    if (peaks.empty()) {
        return false;
    }

    std::sort(peaks.begin(), peaks.end(), [](const ShiftPeak& a, const ShiftPeak& b) {
        if (std::abs(a.corr - b.corr) > 0.04f) return a.corr > b.corr;
        // Prefer shorter fundamental period among equally strong harmonics
        return (a.dx * a.dx + a.dy * a.dy) < (b.dx * b.dx + b.dy * b.dy);
    });

    const float mean_pos_corr = (pos_corr_cnt > 0)
        ? static_cast<float>(pos_corr_sum / static_cast<double>(pos_corr_cnt))
        : 0.2f;
    const ShiftPeak v1 = peaks.front();
    const float peak_to_mean = v1.corr / std::max(0.05f, mean_pos_corr);

    // Require strong periodic lattice signature (either high absolute correlation or sharp peak-to-mean).
    if (v1.corr < 0.58f || (v1.corr < 0.72f && peak_to_mean < 2.0f)) {
        return false;
    }

    // Find optional second linearly independent lattice basis vector v2 (angle 35..145 deg from v1).
    bool has_v2 = false;
    ShiftPeak v2{0, 0, 0.0f};
    const float len1 = std::hypot(static_cast<float>(v1.dx), static_cast<float>(v1.dy));
    for (size_t i = 1; i < peaks.size(); ++i) {
        const auto& c = peaks[i];
        if (c.corr < 0.48f) break;
        const float len2 = std::hypot(static_cast<float>(c.dx), static_cast<float>(c.dy));
        const float cos_ang = std::abs(static_cast<float>(v1.dx * c.dx + v1.dy * c.dy) / std::max(1e-4f, len1 * len2));
        if (cos_ang <= 0.82f) {  // >= ~35 deg separation
            v2 = c;
            has_v2 = true;
            break;
        }
    }

    // Precompute lattice search offsets sorted by Euclidean distance from origin.
    struct LatticeOff {
        int ox = 0;
        int oy = 0;
        int d2 = 0;
    };
    std::vector<LatticeOff> offsets;
    const int k_max = has_v2 ? 12 : 20;
    offsets.reserve(static_cast<size_t>(has_v2 ? (2 * k_max + 1) * (2 * k_max + 1) : 2 * k_max));
    if (has_v2) {
        for (int k1 = -k_max; k1 <= k_max; ++k1) {
            for (int k2 = -k_max; k2 <= k_max; ++k2) {
                if (k1 == 0 && k2 == 0) continue;
                const int ox = k1 * v1.dx + k2 * v2.dx;
                const int oy = k1 * v1.dy + k2 * v2.dy;
                if (std::abs(ox) >= lw || std::abs(oy) >= lh) continue;
                offsets.push_back(LatticeOff{ox, oy, ox * ox + oy * oy});
            }
        }
    } else {
        for (int k1 = -k_max; k1 <= k_max; ++k1) {
            if (k1 == 0) continue;
            const int ox = k1 * v1.dx;
            const int oy = k1 * v1.dy;
            if (std::abs(ox) >= lw || std::abs(oy) >= lh) continue;
            offsets.push_back(LatticeOff{ox, oy, ox * ox + oy * oy});
        }
    }
    std::sort(offsets.begin(), offsets.end(), [](const LatticeOff& a, const LatticeOff& b) {
        return a.d2 < b.d2;
    });

    // Phase-locked sampling into temporary output buffer (commit only if >= 90% coverage).
    std::vector<uint8_t> synth_rgb(ltotal * 3ULL, 0);
    std::vector<uint8_t> synth_ok(ltotal, 0);
    int matched = 0;

    for (int ly = 0; ly < lh; ++ly) {
        const size_t lrow = static_cast<size_t>(ly) * static_cast<size_t>(lw);
        for (int lx = 0; lx < lw; ++lx) {
            const size_t lidx = lrow + static_cast<size_t>(lx);
            if (!is_hole[lidx]) continue;

            for (const auto& off : offsets) {
                const int sx = lx + off.ox;
                const int sy = ly + off.oy;
                if (sx < 0 || sx >= lw || sy < 0 || sy >= lh) continue;
                const size_t sidx = static_cast<size_t>(sy) * static_cast<size_t>(lw) + static_cast<size_t>(sx);
                if (!valid[sidx]) continue;

                const size_t p_src = (static_cast<size_t>(ly0 + sy) * static_cast<size_t>(page_w) +
                                      static_cast<size_t>(lx0 + sx)) * 3ULL;
                synth_rgb[lidx * 3ULL + 0] = canvas_rgb[p_src + 0];
                synth_rgb[lidx * 3ULL + 1] = canvas_rgb[p_src + 1];
                synth_rgb[lidx * 3ULL + 2] = canvas_rgb[p_src + 2];
                synth_ok[lidx] = 1;
                ++matched;
                break;
            }
        }
    }

    if (matched * 10 < hole_cnt * 9) {
        return false;  // Insufficient lattice coverage -> let Tier 3 neural handle it
    }

    // Commit phase-locked synthesized pixels to canvas_rgb.
    for (int ly = 0; ly < lh; ++ly) {
        const int gy = ly0 + ly;
        const size_t lrow = static_cast<size_t>(ly) * static_cast<size_t>(lw);
        const size_t prow = static_cast<size_t>(gy) * static_cast<size_t>(page_w);
        for (int lx = 0; lx < lw; ++lx) {
            const size_t lidx = lrow + static_cast<size_t>(lx);
            if (!is_hole[lidx]) continue;
            const size_t pidx = (prow + static_cast<size_t>(lx0 + lx)) * 3ULL;
            if (synth_ok[lidx]) {
                canvas_rgb[pidx + 0] = synth_rgb[lidx * 3ULL + 0];
                canvas_rgb[pidx + 1] = synth_rgb[lidx * 3ULL + 1];
                canvas_rgb[pidx + 2] = synth_rgb[lidx * 3ULL + 2];
            } else {
                // Fallback for rare unreached pixel (<10%): copy nearest already-synthesized neighbor
                for (int r = 1; r <= 8; ++r) {
                    bool found = false;
                    for (int dy = -r; dy <= r && !found; ++dy) {
                        const int ny = ly + dy;
                        if (ny < 0 || ny >= lh) continue;
                        for (int dx = -r; dx <= r && !found; ++dx) {
                            const int nx = lx + dx;
                            if (nx < 0 || nx >= lw) continue;
                            const size_t nidx = static_cast<size_t>(ny) * static_cast<size_t>(lw) + static_cast<size_t>(nx);
                            if (synth_ok[nidx]) {
                                canvas_rgb[pidx + 0] = synth_rgb[nidx * 3ULL + 0];
                                canvas_rgb[pidx + 1] = synth_rgb[nidx * 3ULL + 1];
                                canvas_rgb[pidx + 2] = synth_rgb[nidx * 3ULL + 2];
                                found = true;
                            } else if (valid[nidx]) {
                                const size_t p_src = (static_cast<size_t>(ly0 + ny) * static_cast<size_t>(page_w) +
                                                      static_cast<size_t>(lx0 + nx)) * 3ULL;
                                canvas_rgb[pidx + 0] = canvas_rgb[p_src + 0];
                                canvas_rgb[pidx + 1] = canvas_rgb[p_src + 1];
                                canvas_rgb[pidx + 2] = canvas_rgb[p_src + 2];
                                found = true;
                            }
                        }
                    }
                    if (found) break;
                }
            }
        }
    }

    return true;
}

// =============================================================================
// #4B — Discrete Screened Poisson Blending via Red-Black SOR
// =============================================================================

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
    float omega,
    int max_iters,
    float tol) {
    if (!dst_rgb || !src_rgb || !mask || width <= 2 || height <= 2) {
        return false;
    }

    roi_x1 = clampi(roi_x1, 0, width);
    roi_y1 = clampi(roi_y1, 0, height);
    roi_x2 = clampi(roi_x2, roi_x1, width);
    roi_y2 = clampi(roi_y2, roi_y1, height);

    // Find tight bounding box of active mask pixels inside ROI.
    int mx1 = roi_x2, my1 = roi_y2, mx2 = roi_x1, my2 = roi_y1;
    int active_cnt = 0;
    for (int y = roi_y1; y < roi_y2; ++y) {
        const uint8_t* mrow = mask + static_cast<size_t>(y) * static_cast<size_t>(width);
        for (int x = roi_x1; x < roi_x2; ++x) {
            if (mrow[x] > 0) {
                if (x < mx1) mx1 = x;
                if (x + 1 > mx2) mx2 = x + 1;
                if (y < my1) my1 = y;
                if (y + 1 > my2) my2 = y + 1;
                ++active_cnt;
            }
        }
    }
    if (active_cnt == 0) return false;

    const int tight_w = mx2 - mx1;
    const int tight_h = my2 - my1;
    if (tight_w > 512 || tight_h > 512) {
        return false;  // Guard: only run Poisson SOR on patches <= 512x512
    }

    // Expand by 1px for Dirichlet boundary f|_{dOmega} = dst_rgb|_{dOmega}.
    const int bx0 = std::max(0, mx1 - 1);
    const int by0 = std::max(0, my1 - 1);
    const int bx1 = std::min(width, mx2 + 1);
    const int by1 = std::min(height, my2 + 1);
    const int bw = bx1 - bx0;
    const int bh = by1 - by0;
    if (bw < 3 || bh < 3) return false;

    const size_t btotal = static_cast<size_t>(bw) * static_cast<size_t>(bh);
    std::vector<uint8_t> local_mask(btotal, 0);
    for (int ly = 0; ly < bh; ++ly) {
        const int gy = by0 + ly;
        const uint8_t* mrow = mask + static_cast<size_t>(gy) * static_cast<size_t>(width);
        uint8_t* lrow = local_mask.data() + static_cast<size_t>(ly) * static_cast<size_t>(bw);
        for (int lx = 0; lx < bw; ++lx) {
            const int gx = bx0 + lx;
            // Pixels touching outer image border stay Dirichlet-fixed to prevent out-of-bounds stencil
            if (ly > 0 && ly + 1 < bh && lx > 0 && lx + 1 < bw && mrow[gx] > 0) {
                lrow[lx] = 1;
            }
        }
    }

    std::vector<float> f(btotal, 0.0f);
    std::vector<float> div_v(btotal, 0.0f);
    const float safe_omega = std::clamp(omega, 1.0f, 1.92f);
    const float abs_tol = std::max(1e-4f, tol) * 255.0f;

    for (int ch = 0; ch < 3; ++ch) {
        // Initialize f and precompute guidance divergence div(v) = 4*g - sum(g_nb) from src_rgb
        for (int ly = 0; ly < bh; ++ly) {
            const int gy = by0 + ly;
            const size_t prow = static_cast<size_t>(gy) * static_cast<size_t>(width);
            const size_t lrow = static_cast<size_t>(ly) * static_cast<size_t>(bw);
            for (int lx = 0; lx < bw; ++lx) {
                const int gx = bx0 + lx;
                const size_t pidx = (prow + static_cast<size_t>(gx)) * 3ULL + static_cast<size_t>(ch);
                const size_t lidx = lrow + static_cast<size_t>(lx);
                f[lidx] = local_mask[lidx]
                    ? static_cast<float>(src_rgb[pidx])
                    : static_cast<float>(dst_rgb[pidx]);
            }
        }

        for (int ly = 1; ly + 1 < bh; ++ly) {
            const int gy = by0 + ly;
            const size_t pr_u = static_cast<size_t>(gy - 1) * static_cast<size_t>(width);
            const size_t pr_c = static_cast<size_t>(gy) * static_cast<size_t>(width);
            const size_t pr_d = static_cast<size_t>(gy + 1) * static_cast<size_t>(width);
            const size_t lrow = static_cast<size_t>(ly) * static_cast<size_t>(bw);
            for (int lx = 1; lx + 1 < bw; ++lx) {
                const size_t lidx = lrow + static_cast<size_t>(lx);
                if (!local_mask[lidx]) continue;
                const int gx = bx0 + lx;
                const float gc = static_cast<float>(src_rgb[(pr_c + static_cast<size_t>(gx)) * 3ULL + static_cast<size_t>(ch)]);
                const float gl = static_cast<float>(src_rgb[(pr_c + static_cast<size_t>(gx - 1)) * 3ULL + static_cast<size_t>(ch)]);
                const float gr = static_cast<float>(src_rgb[(pr_c + static_cast<size_t>(gx + 1)) * 3ULL + static_cast<size_t>(ch)]);
                const float gu = static_cast<float>(src_rgb[(pr_u + static_cast<size_t>(gx)) * 3ULL + static_cast<size_t>(ch)]);
                const float gd = static_cast<float>(src_rgb[(pr_d + static_cast<size_t>(gx)) * 3ULL + static_cast<size_t>(ch)]);
                div_v[lidx] = 4.0f * gc - gl - gr - gu - gd;
            }
        }

        // Red-Black SOR iterations
        for (int iter = 0; iter < max_iters; ++iter) {
            float max_delta = 0.0f;
            for (int color = 0; color < 2; ++color) {
                for (int ly = 1; ly + 1 < bh; ++ly) {
                    const size_t lrow = static_cast<size_t>(ly) * static_cast<size_t>(bw);
                    const int start_x = 1 + ((ly + color) & 1);
                    for (int lx = start_x; lx + 1 < bw; lx += 2) {
                        const size_t lidx = lrow + static_cast<size_t>(lx);
                        if (!local_mask[lidx]) continue;
                        const float gs = 0.25f * (
                            f[lidx - 1] + f[lidx + 1] +
                            f[lidx - static_cast<size_t>(bw)] + f[lidx + static_cast<size_t>(bw)] +
                            div_v[lidx]);
                        const float delta = safe_omega * (gs - f[lidx]);
                        f[lidx] += delta;
                        const float ad = std::abs(delta);
                        if (ad > max_delta) max_delta = ad;
                    }
                }
            }
            if (max_delta < abs_tol) {
                break;
            }
        }

        // Write blended channel back to dst_rgb strictly inside active mask
        for (int ly = 1; ly + 1 < bh; ++ly) {
            const int gy = by0 + ly;
            const size_t prow = static_cast<size_t>(gy) * static_cast<size_t>(width);
            const size_t lrow = static_cast<size_t>(ly) * static_cast<size_t>(bw);
            for (int lx = 1; lx + 1 < bw; ++lx) {
                const size_t lidx = lrow + static_cast<size_t>(lx);
                if (!local_mask[lidx]) continue;
                const int gx = bx0 + lx;
                const size_t pidx = (prow + static_cast<size_t>(gx)) * 3ULL + static_cast<size_t>(ch);
                dst_rgb[pidx] = static_cast<uint8_t>(clampi(static_cast<int>(std::lround(f[lidx])), 0, 255));
            }
        }
    }

    return true;
}

// =============================================================================
// #12 — Criminisi Isophote-Driven Exemplar + PatchMatch Inpainting
// =============================================================================

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
    int patch_radius,
    int max_hole_area,
    bool require_high_coherence) {
    if (!canvas_rgb || !hole_mask || !all_masks || page_w <= 0 || page_h <= 0) {
        return false;
    }

    bx1 = clampi(bx1, 0, page_w);
    by1 = clampi(by1, 0, page_h);
    bx2 = clampi(bx2, bx1, page_w);
    by2 = clampi(by2, by1, page_h);
    if (bx2 <= bx1 || by2 <= by1) return false;

    int hole_area = 0;
    for (int y = by1; y < by2; ++y) {
        const uint8_t* hrow = hole_mask + static_cast<size_t>(y) * static_cast<size_t>(page_w);
        for (int x = bx1; x < bx2; ++x) {
            if (hrow[x] > 0) ++hole_area;
        }
    }
    if (hole_area == 0 || hole_area > max_hole_area) {
        return false;
    }

    const int pr = std::clamp(patch_radius, 2, 4);  // 5x5, 7x7, or 9x9
    constexpr int kPad = 28;
    const int lx0 = clampi(bx1 - kPad, 0, page_w);
    const int ly0 = clampi(by1 - kPad, 0, page_h);
    const int lx1 = clampi(bx2 + kPad, 0, page_w);
    const int ly1 = clampi(by2 + kPad, 0, page_h);
    const int lw = lx1 - lx0;
    const int lh = ly1 - ly0;
    if (lw < 2 * pr + 8 || lh < 2 * pr + 8) return false;

    const size_t ltotal = static_cast<size_t>(lw) * static_cast<size_t>(lh);
    std::vector<uint8_t> work_rgb(ltotal * 3ULL, 0);
    std::vector<uint8_t> gray(ltotal, 0);
    std::vector<uint8_t> filled(ltotal, 1);       // 1 = known, 0 = hole
    std::vector<uint8_t> valid_src_px(ltotal, 0); // 1 = originally valid exterior pixel
    std::vector<float> conf(ltotal, 1.0f);

    for (int ly = 0; ly < lh; ++ly) {
        const int gy = ly0 + ly;
        const size_t prow = static_cast<size_t>(gy) * static_cast<size_t>(page_w);
        const size_t lrow = static_cast<size_t>(ly) * static_cast<size_t>(lw);
        for (int lx = 0; lx < lw; ++lx) {
            const int gx = lx0 + lx;
            const size_t pidx = prow + static_cast<size_t>(gx);
            const size_t lidx = lrow + static_cast<size_t>(lx);
            work_rgb[lidx * 3ULL + 0] = canvas_rgb[pidx * 3ULL + 0];
            work_rgb[lidx * 3ULL + 1] = canvas_rgb[pidx * 3ULL + 1];
            work_rgb[lidx * 3ULL + 2] = canvas_rgb[pidx * 3ULL + 2];
            gray[lidx] = rgbToLuma(work_rgb[lidx * 3ULL + 0], work_rgb[lidx * 3ULL + 1], work_rgb[lidx * 3ULL + 2]);

            const bool in_box = (gx >= bx1 && gx < bx2 && gy >= by1 && gy < by2);
            if (in_box && hole_mask[pidx] > 0) {
                filled[lidx] = 0;
                conf[lidx] = 0.0f;
            } else if (all_masks[pidx] == 0) {
                valid_src_px[lidx] = 1;
            }
        }
    }

    if (require_high_coherence) {
        const float coh = computeExteriorCoherence(gray.data(), valid_src_px.data(), lw, lh);
        if (coh < 0.24f) {
            return false;  // Not strongly structured linework -> let Tier 3 neural handle it
        }
    }

    // Collect valid source patch centers (entire (2*pr+1)x(2*pr+1) patch must be valid_src_px == 1).
    std::vector<int> src_centers;
    src_centers.reserve(ltotal / 2);
    for (int y = pr; y + pr < lh; ++y) {
        for (int x = pr; x + pr < lw; ++x) {
            bool all_ok = true;
            for (int dy = -pr; dy <= pr && all_ok; ++dy) {
                const size_t row = static_cast<size_t>(y + dy) * static_cast<size_t>(lw);
                for (int dx = -pr; dx <= pr; ++dx) {
                    if (!valid_src_px[row + static_cast<size_t>(x + dx)]) {
                        all_ok = false;
                        break;
                    }
                }
            }
            if (all_ok) {
                src_centers.push_back(y * lw + x);
            }
        }
    }
    if (src_centers.size() < 8) {
        return false;
    }

    std::vector<uint8_t> is_valid_src_center(ltotal, 0);
    for (int c : src_centers) {
        is_valid_src_center[static_cast<size_t>(c)] = 1;
    }

    // PatchMatch offset cache for neighbor propagation
    std::vector<int> match_x(ltotal, -1);
    std::vector<int> match_y(ltotal, -1);

    uint32_t rng = 0xA511E9B3u ^ static_cast<uint32_t>(lw * 17 + lh * 31);
    auto nextRand = [&rng]() noexcept -> uint32_t {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    };

    auto evalPatchSSD = [&](int px, int py, int sx, int sy, int best_ssd) noexcept -> int {
        int ssd = 0;
        for (int dy = -pr; dy <= pr; ++dy) {
            const int ty = py + dy;
            if (ty < 0 || ty >= lh) continue;
            const size_t trow = static_cast<size_t>(ty) * static_cast<size_t>(lw);
            const size_t srow = static_cast<size_t>(sy + dy) * static_cast<size_t>(lw);
            for (int dx = -pr; dx <= pr; ++dx) {
                const int tx = px + dx;
                if (tx < 0 || tx >= lw) continue;
                const size_t tidx = trow + static_cast<size_t>(tx);
                if (!filled[tidx]) continue;
                const size_t sidx = srow + static_cast<size_t>(sx + dx);
                const int dr = static_cast<int>(work_rgb[tidx * 3ULL + 0]) - static_cast<int>(work_rgb[sidx * 3ULL + 0]);
                const int dg = static_cast<int>(work_rgb[tidx * 3ULL + 1]) - static_cast<int>(work_rgb[sidx * 3ULL + 1]);
                const int db = static_cast<int>(work_rgb[tidx * 3ULL + 2]) - static_cast<int>(work_rgb[sidx * 3ULL + 2]);
                ssd += dr * dr + dg * dg + db * db;
                if (ssd >= best_ssd) return best_ssd;
            }
        }
        return ssd;
    };

    int remaining = hole_area;
    int max_steps = std::max(64, hole_area);
    static constexpr int kDx4[4] = {-1, 1, 0, 0};
    static constexpr int kDy4[4] = {0, 0, -1, 1};

    while (remaining > 0 && max_steps-- > 0) {
        // 1. Find highest-priority front pixel p* via Criminisi P(p) = C(p) * (D(p) + 0.05).
        int best_px = -1, best_py = -1;
        float best_prio = -1.0f;
        float best_conf = 0.0f;

        for (int y = 1; y + 1 < lh; ++y) {
            const size_t row = static_cast<size_t>(y) * static_cast<size_t>(lw);
            for (int x = 1; x + 1 < lw; ++x) {
                const size_t idx = row + static_cast<size_t>(x);
                if (filled[idx]) continue;
                // Check if on fill front (4-adjacent to a filled pixel)
                if (!filled[idx - 1] && !filled[idx + 1] &&
                    !filled[idx - static_cast<size_t>(lw)] && !filled[idx + static_cast<size_t>(lw)]) {
                    continue;
                }

                // Confidence C(p)
                float c_sum = 0.0f;
                int c_cnt = 0;
                float max_g2 = -1.0f;
                float iso_x = 0.0f, iso_y = 0.0f;

                for (int dy = -pr; dy <= pr; ++dy) {
                    const int ty = y + dy;
                    if (ty <= 0 || ty + 1 >= lh) continue;
                    const size_t trow = static_cast<size_t>(ty) * static_cast<size_t>(lw);
                    for (int dx = -pr; dx <= pr; ++dx) {
                        const int tx = x + dx;
                        if (tx <= 0 || tx + 1 >= lw) continue;
                        const size_t tidx = trow + static_cast<size_t>(tx);
                        c_sum += conf[tidx];
                        ++c_cnt;
                        if (filled[tidx] && filled[tidx - 1] && filled[tidx + 1] &&
                            filled[tidx - static_cast<size_t>(lw)] && filled[tidx + static_cast<size_t>(lw)]) {
                            const float gx = 0.5f * (static_cast<float>(gray[tidx + 1]) - static_cast<float>(gray[tidx - 1]));
                            const float gy = 0.5f * (static_cast<float>(gray[tidx + static_cast<size_t>(lw)]) -
                                                     static_cast<float>(gray[tidx - static_cast<size_t>(lw)]));
                            const float g2 = gx * gx + gy * gy;
                            if (g2 > max_g2) {
                                max_g2 = g2;
                                iso_x = -gy;
                                iso_y = gx;
                            }
                        }
                    }
                }
                const float Cp = (c_cnt > 0) ? (c_sum / static_cast<float>(c_cnt)) : 0.0f;

                // Boundary normal n_p = normalize(grad(filled))
                const float nx = static_cast<float>(filled[idx + 1]) - static_cast<float>(filled[idx - 1]);
                const float ny = static_cast<float>(filled[idx + static_cast<size_t>(lw)]) -
                                 static_cast<float>(filled[idx - static_cast<size_t>(lw)]);
                const float nlen = std::hypot(nx, ny);
                const float Dp = (nlen > 1e-4f)
                    ? (std::abs(iso_x * (nx / nlen) + iso_y * (ny / nlen)) / 255.0f)
                    : 0.0f;

                const float prio = Cp * (Dp + 0.05f);
                if (prio > best_prio) {
                    best_prio = prio;
                    best_px = x;
                    best_py = y;
                    best_conf = Cp;
                }
            }
        }

        if (best_px < 0) break;

        // 2. PatchMatch search: neighbor propagation + random seed + shrinking window refinement.
        int best_sx = src_centers[0] % lw;
        int best_sy = src_centers[0] / lw;
        int best_ssd = std::numeric_limits<int>::max();

        auto tryCandidate = [&](int sx, int sy) {
            if (sx < pr || sx + pr >= lw || sy < pr || sy + pr >= lh) return;
            if (!is_valid_src_center[static_cast<size_t>(sy) * static_cast<size_t>(lw) + static_cast<size_t>(sx)]) return;
            const int ssd = evalPatchSSD(best_px, best_py, sx, sy, best_ssd);
            if (ssd < best_ssd) {
                best_ssd = ssd;
                best_sx = sx;
                best_sy = sy;
            }
        };

        // Neighbor propagation from previously matched adjacent patches
        for (int k = 0; k < 4; ++k) {
            const int nx = best_px + kDx4[k] * pr;
            const int ny = best_py + kDy4[k] * pr;
            if (nx >= 0 && nx < lw && ny >= 0 && ny < lh) {
                const size_t nidx = static_cast<size_t>(ny) * static_cast<size_t>(lw) + static_cast<size_t>(nx);
                if (match_x[nidx] >= 0) {
                    tryCandidate(match_x[nidx] - kDx4[k] * pr, match_y[nidx] - kDy4[k] * pr);
                }
            }
        }

        // Global random probes across valid source centers
        const size_t num_probes = std::min<size_t>(src_centers.size(), 48);
        for (size_t p = 0; p < num_probes; ++p) {
            const int c = src_centers[nextRand() % src_centers.size()];
            tryCandidate(c % lw, c / lw);
        }

        // Shrinking random search window around current best match
        for (int rad = std::max(lw, lh) / 3; rad >= 1; rad /= 2) {
            for (int t = 0; t < 4; ++t) {
                const int rx = best_sx + static_cast<int>(nextRand() % static_cast<uint32_t>(2 * rad + 1)) - rad;
                const int ry = best_sy + static_cast<int>(nextRand() % static_cast<uint32_t>(2 * rad + 1)) - rad;
                tryCandidate(rx, ry);
            }
        }

        // 3. Copy matched exemplar patch into unknown pixels around (best_px, best_py)
        const size_t pidx_c = static_cast<size_t>(best_py) * static_cast<size_t>(lw) + static_cast<size_t>(best_px);
        match_x[pidx_c] = best_sx;
        match_y[pidx_c] = best_sy;

        for (int dy = -pr; dy <= pr; ++dy) {
            const int ty = best_py + dy;
            if (ty < 0 || ty >= lh) continue;
            const size_t trow = static_cast<size_t>(ty) * static_cast<size_t>(lw);
            const size_t srow = static_cast<size_t>(best_sy + dy) * static_cast<size_t>(lw);
            for (int dx = -pr; dx <= pr; ++dx) {
                const int tx = best_px + dx;
                if (tx < 0 || tx >= lw) continue;
                const size_t tidx = trow + static_cast<size_t>(tx);
                if (filled[tidx]) continue;
                const size_t sidx = srow + static_cast<size_t>(best_sx + dx);
                work_rgb[tidx * 3ULL + 0] = work_rgb[sidx * 3ULL + 0];
                work_rgb[tidx * 3ULL + 1] = work_rgb[sidx * 3ULL + 1];
                work_rgb[tidx * 3ULL + 2] = work_rgb[sidx * 3ULL + 2];
                gray[tidx] = gray[sidx];
                filled[tidx] = 1;
                conf[tidx] = best_conf;
                --remaining;
            }
        }
    }

    if (remaining > 0) {
        return false;
    }

    // Commit filled hole pixels to canvas_rgb.
    for (int gy = by1; gy < by2; ++gy) {
        const size_t prow = static_cast<size_t>(gy) * static_cast<size_t>(page_w);
        const size_t lrow = static_cast<size_t>(gy - ly0) * static_cast<size_t>(lw);
        for (int gx = bx1; gx < bx2; ++gx) {
            const size_t pidx = prow + static_cast<size_t>(gx);
            if (hole_mask[pidx] == 0) continue;
            const size_t lidx = lrow + static_cast<size_t>(gx - lx0);
            canvas_rgb[pidx * 3ULL + 0] = work_rgb[lidx * 3ULL + 0];
            canvas_rgb[pidx * 3ULL + 1] = work_rgb[lidx * 3ULL + 1];
            canvas_rgb[pidx * 3ULL + 2] = work_rgb[lidx * 3ULL + 2];
        }
    }

    return true;
}

// =============================================================================
// #21 — Hasler-Susstrunk Colorfulness + Fast Luminance SSIM
// =============================================================================

float computeHaslerSusstrunkColorfulness(
    const uint8_t* rgb,
    int width,
    int height,
    int stride) noexcept {
    if (!rgb || width <= 0 || height <= 0) return 0.0f;
    const int step = std::max(1, stride);

    double sum_rg = 0.0, sum_rg2 = 0.0;
    double sum_yb = 0.0, sum_yb2 = 0.0;
    size_t cnt = 0;

    for (int y = 0; y < height; y += step) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(width) * 3ULL;
        for (int x = 0; x < width; x += step) {
            const uint8_t* px = rgb + row + static_cast<size_t>(x) * 3ULL;
            const double r = static_cast<double>(px[0]);
            const double g = static_cast<double>(px[1]);
            const double b = static_cast<double>(px[2]);
            const double rg = r - g;
            const double yb = 0.5 * (r + g) - b;
            sum_rg += rg;
            sum_rg2 += rg * rg;
            sum_yb += yb;
            sum_yb2 += yb * yb;
            ++cnt;
        }
    }
    if (cnt == 0) return 0.0f;

    const double inv = 1.0 / static_cast<double>(cnt);
    const double mu_rg = sum_rg * inv;
    const double mu_yb = sum_yb * inv;
    const double var_rg = std::max(0.0, sum_rg2 * inv - mu_rg * mu_rg);
    const double var_yb = std::max(0.0, sum_yb2 * inv - mu_yb * mu_yb);

    const double sigma_rgyb = std::sqrt(var_rg + var_yb);
    const double mu_rgyb = std::sqrt(mu_rg * mu_rg + mu_yb * mu_yb);
    return static_cast<float>(sigma_rgyb + 0.3 * mu_rgyb);
}

void rgbToGray8(
    const uint8_t* rgb,
    int width,
    int height,
    std::vector<uint8_t>& out_gray) {
    if (!rgb || width <= 0 || height <= 0) {
        out_gray.clear();
        return;
    }
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    out_gray.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* px = rgb + i * 3ULL;
        out_gray[i] = rgbToLuma(px[0], px[1], px[2]);
    }
}

float computeFastLuminanceSSIM(
    const uint8_t* ref_gray,
    const uint8_t* test_gray,
    int width,
    int height) noexcept {
    if (!ref_gray || !test_gray || width < 8 || height < 8) return 1.0f;

    // Sample 8x8 blocks across the image (stepped so <= ~512 blocks are evaluated -> < 0.5 ms).
    const int ds = std::max(1, std::max(width, height) / 256);
    const int blk_stride = 8 * ds;
    constexpr double kC1 = (0.01 * 255.0) * (0.01 * 255.0);
    constexpr double kC2 = (0.03 * 255.0) * (0.03 * 255.0);

    double sum_ssim = 0.0;
    int blocks = 0;

    for (int y = 0; y + 8 <= height; y += blk_stride) {
        for (int x = 0; x + 8 <= width; x += blk_stride) {
            double sx = 0.0, sy = 0.0, sx2 = 0.0, sy2 = 0.0, sxy = 0.0;
            for (int dy = 0; dy < 8; ++dy) {
                const size_t row = static_cast<size_t>(y + dy) * static_cast<size_t>(width) + static_cast<size_t>(x);
                for (int dx = 0; dx < 8; ++dx) {
                    const double vx = static_cast<double>(ref_gray[row + static_cast<size_t>(dx)]);
                    const double vy = static_cast<double>(test_gray[row + static_cast<size_t>(dx)]);
                    sx += vx;
                    sy += vy;
                    sx2 += vx * vx;
                    sy2 += vy * vy;
                    sxy += vx * vy;
                }
            }
            constexpr double inv64 = 1.0 / 64.0;
            const double mx = sx * inv64;
            const double my = sy * inv64;
            const double vx = std::max(0.0, sx2 * inv64 - mx * mx);
            const double vy = std::max(0.0, sy2 * inv64 - my * my);
            const double vxy = sxy * inv64 - mx * my;

            const double num = (2.0 * mx * my + kC1) * (2.0 * vxy + kC2);
            const double den = (mx * mx + my * my + kC1) * (vx + vy + kC2);
            sum_ssim += (den > 1e-12) ? (num / den) : 1.0;
            ++blocks;
        }
    }

    return (blocks > 0) ? static_cast<float>(std::clamp(sum_ssim / static_cast<double>(blocks), 0.0, 1.0)) : 1.0f;
}

}  // namespace mangaflow::pipeline::math_inpaint
