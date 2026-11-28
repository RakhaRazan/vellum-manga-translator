#include "yolo_seg_backend.hpp"
#include "cv_geometry.hpp"
#include "cv_refinement.hpp"
#include "cv_typography.hpp"
#include "geometry_postproc.hpp"
#include "math_det_ops.hpp"
#include "../pipeline/ort_provider.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <thread>
#include <vector>

namespace vellum::detector {

namespace {

constexpr int kYoloInputSize = 1024;
constexpr int kMaskChannels = 32;
constexpr float kLetterboxPadVal = 114.0f / 255.0f;
constexpr float kMaskThreshold = 0.50f;
constexpr float kDualRescueFloor = 0.14f;
constexpr float kHighConfCriticBypass = 0.45f;
constexpr float kContainmentSuppressThresh = 0.78f;

// Lightweight high-contrast edge/outline critic for zero-bubble rescue on stylized
// screentone / hollow-text balloons crossed by speed lines (zero extra ONNX forward pass).
bool passesHollowOutlineCritic(
    const uint8_t* rgbData,
    int width,
    int height,
    const BBox& bbox) noexcept {
    if (!rgbData || width <= 0 || height <= 0 || !bbox.is_valid()) {
        return false;
    }
    const int raw_x1 = std::clamp(static_cast<int>(std::floor(bbox.x1)), 0, width - 1);
    const int raw_y1 = std::clamp(static_cast<int>(std::floor(bbox.y1)), 0, height - 1);
    const int raw_x2 = std::clamp(static_cast<int>(std::ceil(bbox.x2)), raw_x1 + 1, width);
    const int raw_y2 = std::clamp(static_cast<int>(std::ceil(bbox.y2)), raw_y1 + 1, height);
    const int bw = raw_x2 - raw_x1;
    const int bh = raw_y2 - raw_y1;
    if (bw < 14 || bh < 14) {
        return false;
    }
    const double box_area = static_cast<double>(bw) * static_cast<double>(bh);
    const double page_area = static_cast<double>(width) * static_cast<double>(height);
    if (page_area <= 0.0 || (box_area / page_area) > 0.45) {
        return false;
    }

    const int inset_x = std::clamp(bw / 10, 1, 8);
    const int inset_y = std::clamp(bh / 10, 1, 8);
    const int ix1 = raw_x1 + inset_x;
    const int iy1 = raw_y1 + inset_y;
    const int ix2 = std::max(ix1 + 2, raw_x2 - inset_x);
    const int iy2 = std::max(iy1 + 2, raw_y2 - inset_y);
    const int iw = ix2 - ix1;
    const int ih = iy2 - iy1;

    const int step_x = std::max(1, (iw + 79) / 80);
    const int step_y = std::max(1, (ih + 79) / 80);
    const int gw = (iw + step_x - 1) / step_x;
    const int gh = (ih + step_y - 1) / step_y;
    if (gw < 4 || gh < 4) {
        return false;
    }

    std::vector<uint8_t> gray(static_cast<size_t>(gw) * static_cast<size_t>(gh));
    int min_lum = 255;
    int max_lum = 0;
    for (int gy = 0; gy < gh; ++gy) {
        const int sy = std::min(iy2 - 1, iy1 + gy * step_y);
        const size_t src_row = static_cast<size_t>(sy) * static_cast<size_t>(width);
        const size_t dst_row = static_cast<size_t>(gy) * static_cast<size_t>(gw);
        for (int gx = 0; gx < gw; ++gx) {
            const int sx = std::min(ix2 - 1, ix1 + gx * step_x);
            const size_t idx = (src_row + static_cast<size_t>(sx)) * 3ULL;
            const int lum = (77 * static_cast<int>(rgbData[idx]) +
                             150 * static_cast<int>(rgbData[idx + 1]) +
                             29 * static_cast<int>(rgbData[idx + 2])) >> 8;
            gray[dst_row + static_cast<size_t>(gx)] = static_cast<uint8_t>(lum);
            min_lum = std::min(min_lum, lum);
            max_lum = std::max(max_lum, lum);
        }
    }
    if ((max_lum - min_lum) < 45) {
        return false;
    }

    std::vector<uint8_t> edge_map(gray.size(), 0);
    int edge_count = 0;
    for (int y = 0; y < gh; ++y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(gw);
        for (int x = 0; x < gw; ++x) {
            const int v = static_cast<int>(gray[row + static_cast<size_t>(x)]);
            const int dx = (x + 1 < gw) ? std::abs(v - static_cast<int>(gray[row + static_cast<size_t>(x + 1)])) : 0;
            const int dy = (y + 1 < gh) ? std::abs(v - static_cast<int>(gray[row + static_cast<size_t>(gw) + static_cast<size_t>(x)])) : 0;
            if (dx >= 38 || dy >= 38) {
                edge_map[row + static_cast<size_t>(x)] = 1;
                ++edge_count;
            }
        }
    }

    const float edge_ratio = static_cast<float>(edge_count) / static_cast<float>(gray.size());
    if (edge_ratio < 0.025f || edge_ratio > 0.55f) {
        return false;
    }

    // Require localized stroke clusters (not flat uniform screentone/background texture).
    float cell_edges[16]{};
    int cell_count = 0;
    float sum_edge = 0.0f;
    float max_cell = 0.0f;
    for (int cy = 0; cy < 4; ++cy) {
        const int y0 = (cy * gh) / 4;
        const int y1 = ((cy + 1) * gh) / 4;
        for (int cx = 0; cx < 4; ++cx) {
            const int x0 = (cx * gw) / 4;
            const int x1 = ((cx + 1) * gw) / 4;
            int ce = 0;
            int ct = 0;
            for (int y = y0; y < y1; ++y) {
                const size_t row = static_cast<size_t>(y) * static_cast<size_t>(gw);
                for (int x = x0; x < x1; ++x) {
                    ce += edge_map[row + static_cast<size_t>(x)];
                    ++ct;
                }
            }
            const float r = (ct > 0) ? (static_cast<float>(ce) / static_cast<float>(ct)) : 0.0f;
            cell_edges[cell_count++] = r;
            sum_edge += r;
            max_cell = std::max(max_cell, r);
        }
    }
    const float mean_edge = sum_edge / static_cast<float>(cell_count);
    float var = 0.0f;
    for (int i = 0; i < cell_count; ++i) {
        const float d = cell_edges[i] - mean_edge;
        var += d * d;
    }
    const float std_edge = std::sqrt(var / static_cast<float>(cell_count));
    return (std_edge >= 0.015f) || (max_cell >= 0.08f && edge_ratio <= 0.40f);
}

bool isPageNonBlank(const uint8_t* rgbData, int width, int height) noexcept {
    if (!rgbData || width <= 0 || height <= 0) return false;
    const int step_x = std::max(1, width / 64);
    const int step_y = std::max(1, height / 64);
    int min_lum = 255;
    int max_lum = 0;
    for (int y = 0; y < height; y += step_y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(width);
        for (int x = 0; x < width; x += step_x) {
            const size_t idx = (row + static_cast<size_t>(x)) * 3ULL;
            const int lum = (77 * static_cast<int>(rgbData[idx]) +
                             150 * static_cast<int>(rgbData[idx + 1]) +
                             29 * static_cast<int>(rgbData[idx + 2])) >> 8;
            min_lum = std::min(min_lum, lum);
            max_lum = std::max(max_lum, lum);
            if ((max_lum - min_lum) >= 30) {
                return true;
            }
        }
    }
    return false;
}

// IEEE-754 float32 <-> float16 conversion helpers for FP16 ONNX models.
uint16_t floatToHalfBits(float value) noexcept {
    uint32_t f = 0;
    std::memcpy(&f, &value, sizeof(f));
    const uint32_t sign = (f >> 16) & 0x8000u;
    const int32_t exp = static_cast<int32_t>((f >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mant = f & 0x007FFFFFu;

    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        const uint32_t m = mant | 0x00800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        return static_cast<uint16_t>(sign | (m >> shift));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

float halfBitsToFloat(uint16_t h) noexcept {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t f = 0;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            uint32_t m = mant;
            int32_t e = 127 - 15 + 1;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x03FFu;
            f = sign | (static_cast<uint32_t>(e) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

inline float sigmoidf(float x) noexcept {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

struct TrimRect {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
};

struct LetterboxInfo {
    float scale = 1.0f;
    int pad_left = 0;
    int pad_top = 0;
    int new_w = kYoloInputSize;
    int new_h = kYoloInputSize;
};

inline int rgbLuma(uint8_t r, uint8_t g, uint8_t b) noexcept {
    return (77 * static_cast<int>(r) + 150 * static_cast<int>(g) + 29 * static_cast<int>(b)) >> 8;
}

// Auto-Margin Trim: scan outer uniform white/black margins, clamped to max 25% per edge.
TrimRect computeAutoMarginTrim(const uint8_t* rgb, int width, int height) noexcept {
    TrimRect full{0, 0, width, height};
    if (!rgb || width < 128 || height < 128) {
        return full;
    }

    auto classifyRow = [&](int y) -> int {
        const int step = std::max(1, width / 128);
        int total = 0;
        int white = 0;
        int black = 0;
        const size_t row_off = static_cast<size_t>(y) * static_cast<size_t>(width) * 3ULL;
        for (int x = 0; x < width; x += step) {
            const size_t idx = row_off + static_cast<size_t>(x) * 3ULL;
            const int lum = rgbLuma(rgb[idx], rgb[idx + 1], rgb[idx + 2]);
            if (lum >= 242) ++white;
            else if (lum <= 18) ++black;
            ++total;
        }
        if (total == 0) return 0;
        if (white * 100 >= total * 98) return 1;
        if (black * 100 >= total * 98) return 2;
        return 0;
    };

    auto classifyCol = [&](int x) -> int {
        const int step = std::max(1, height / 128);
        int total = 0;
        int white = 0;
        int black = 0;
        for (int y = 0; y < height; y += step) {
            const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(width) +
                                static_cast<size_t>(x)) * 3ULL;
            const int lum = rgbLuma(rgb[idx], rgb[idx + 1], rgb[idx + 2]);
            if (lum >= 242) ++white;
            else if (lum <= 18) ++black;
            ++total;
        }
        if (total == 0) return 0;
        if (white * 100 >= total * 98) return 1;
        if (black * 100 >= total * 98) return 2;
        return 0;
    };

    const int max_trim_x = width / 4;
    const int max_trim_y = height / 4;

    int trim_top = 0;
    const int top_pol = classifyRow(0);
    if (top_pol != 0) {
        while (trim_top < max_trim_y && classifyRow(trim_top) == top_pol) {
            ++trim_top;
        }
    }

    int trim_bottom = 0;
    const int bot_pol = classifyRow(height - 1);
    if (bot_pol != 0) {
        while (trim_bottom < max_trim_y && classifyRow(height - 1 - trim_bottom) == bot_pol) {
            ++trim_bottom;
        }
    }

    int trim_left = 0;
    const int left_pol = classifyCol(0);
    if (left_pol != 0) {
        while (trim_left < max_trim_x && classifyCol(trim_left) == left_pol) {
            ++trim_left;
        }
    }

    int trim_right = 0;
    const int right_pol = classifyCol(width - 1);
    if (right_pol != 0) {
        while (trim_right < max_trim_x && classifyCol(width - 1 - trim_right) == right_pol) {
            ++trim_right;
        }
    }

    // Only trim strictly uniform borders >= 3% of page width/height to prevent box drift.
    const int min_trim_x = (width * 3) / 100;
    const int min_trim_y = (height * 3) / 100;
    if (trim_top < min_trim_y) trim_top = 0;
    if (trim_bottom < min_trim_y) trim_bottom = 0;
    if (trim_left < min_trim_x) trim_left = 0;
    if (trim_right < min_trim_x) trim_right = 0;

    // Leave a tiny 2px guard margin if trimmed so edge-touching bubbles aren't clipped.
    if (trim_top > 2) trim_top -= 2;
    if (trim_bottom > 2) trim_bottom -= 2;
    if (trim_left > 2) trim_left -= 2;
    if (trim_right > 2) trim_right -= 2;

    const int crop_w = width - trim_left - trim_right;
    const int crop_h = height - trim_top - trim_bottom;
    if (crop_w < 64 || crop_h < 64) {
        return full;
    }
    return TrimRect{trim_left, trim_top, crop_w, crop_h};
}

LetterboxInfo computeLetterbox(int src_w, int src_h, int dst_w, int dst_h) noexcept {
    LetterboxInfo info;
    info.scale = std::min(static_cast<float>(dst_w) / static_cast<float>(src_w),
                          static_cast<float>(dst_h) / static_cast<float>(src_h));
    info.new_w = std::max(1, static_cast<int>(std::round(static_cast<float>(src_w) * info.scale)));
    info.new_h = std::max(1, static_cast<int>(std::round(static_cast<float>(src_h) * info.scale)));
    info.pad_left = (dst_w - info.new_w) / 2;
    info.pad_top = (dst_h - info.new_h) / 2;
    return info;
}

void letterboxRgbToNchwFloat(
    const uint8_t* rgb,
    int full_w,
    const TrimRect& trim,
    const LetterboxInfo& lb,
    int dst_w,
    int dst_h,
    std::vector<float>& out_nchw) {
    const size_t plane = static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h);
    out_nchw.assign(3ULL * plane, kLetterboxPadVal);

    float* ch0 = out_nchw.data();
    float* ch1 = out_nchw.data() + plane;
    float* ch2 = out_nchw.data() + 2ULL * plane;

    const int src_w = trim.w;
    const int src_h = trim.h;
    const float scale_x = static_cast<float>(src_w) / static_cast<float>(lb.new_w);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(lb.new_h);
    const bool is_downscale = (src_w > lb.new_w || src_h > lb.new_h);

    if (is_downscale) {
        // Area-averaging (box) downscale to eliminate screentone moire artifacts.
        for (int dy = 0; dy < lb.new_h; ++dy) {
            const float sy0_f = static_cast<float>(dy) * scale_y;
            const float sy1_f = static_cast<float>(dy + 1) * scale_y;
            const int iy0 = std::clamp(static_cast<int>(std::floor(sy0_f)), 0, src_h - 1);
            const int iy1 = std::clamp(static_cast<int>(std::ceil(sy1_f)) - 1, iy0, src_h - 1);
            const size_t dst_row = static_cast<size_t>(dy + lb.pad_top) * static_cast<size_t>(dst_w);

            for (int dx = 0; dx < lb.new_w; ++dx) {
                const float sx0_f = static_cast<float>(dx) * scale_x;
                const float sx1_f = static_cast<float>(dx + 1) * scale_x;
                const int ix0 = std::clamp(static_cast<int>(std::floor(sx0_f)), 0, src_w - 1);
                const int ix1 = std::clamp(static_cast<int>(std::ceil(sx1_f)) - 1, ix0, src_w - 1);

                float sum_r = 0.0f;
                float sum_g = 0.0f;
                float sum_b = 0.0f;
                float sum_w = 0.0f;

                for (int iy = iy0; iy <= iy1; ++iy) {
                    const float wy = std::min(sy1_f, static_cast<float>(iy + 1)) -
                                     std::max(sy0_f, static_cast<float>(iy));
                    if (wy <= 0.0f) continue;
                    const size_t src_row = static_cast<size_t>(trim.y + iy) * static_cast<size_t>(full_w);
                    for (int ix = ix0; ix <= ix1; ++ix) {
                        const float wx = std::min(sx1_f, static_cast<float>(ix + 1)) -
                                         std::max(sx0_f, static_cast<float>(ix));
                        const float w = wy * wx;
                        if (w <= 0.0f) continue;
                        const size_t idx = (src_row + static_cast<size_t>(trim.x + ix)) * 3ULL;
                        sum_r += w * static_cast<float>(rgb[idx + 0]);
                        sum_g += w * static_cast<float>(rgb[idx + 1]);
                        sum_b += w * static_cast<float>(rgb[idx + 2]);
                        sum_w += w;
                    }
                }

                const float inv = (sum_w > 0.0f) ? (1.0f / (sum_w * 255.0f)) : 0.0f;
                const size_t dst_idx = dst_row + static_cast<size_t>(dx + lb.pad_left);
                ch0[dst_idx] = sum_r * inv;
                ch1[dst_idx] = sum_g * inv;
                ch2[dst_idx] = sum_b * inv;
            }
        }
        return;
    }

    for (int dy = 0; dy < lb.new_h; ++dy) {
        const float sy_f = (static_cast<float>(dy) + 0.5f) * scale_y - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(sy_f)), 0, src_h - 1);
        const int y1 = std::clamp(y0 + 1, 0, src_h - 1);
        const float wy1 = std::clamp(sy_f - static_cast<float>(y0), 0.0f, 1.0f);
        const float wy0 = 1.0f - wy1;

        const size_t dst_row = static_cast<size_t>(dy + lb.pad_top) * static_cast<size_t>(dst_w);
        const size_t src_row0 = static_cast<size_t>(trim.y + y0) * static_cast<size_t>(full_w);
        const size_t src_row1 = static_cast<size_t>(trim.y + y1) * static_cast<size_t>(full_w);

        for (int dx = 0; dx < lb.new_w; ++dx) {
            const float sx_f = (static_cast<float>(dx) + 0.5f) * scale_x - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(sx_f)), 0, src_w - 1);
            const int x1 = std::clamp(x0 + 1, 0, src_w - 1);
            const float wx1 = std::clamp(sx_f - static_cast<float>(x0), 0.0f, 1.0f);
            const float wx0 = 1.0f - wx1;

            const size_t idx00 = (src_row0 + static_cast<size_t>(trim.x + x0)) * 3ULL;
            const size_t idx01 = (src_row0 + static_cast<size_t>(trim.x + x1)) * 3ULL;
            const size_t idx10 = (src_row1 + static_cast<size_t>(trim.x + x0)) * 3ULL;
            const size_t idx11 = (src_row1 + static_cast<size_t>(trim.x + x1)) * 3ULL;

            const float w00 = wy0 * wx0;
            const float w01 = wy0 * wx1;
            const float w10 = wy1 * wx0;
            const float w11 = wy1 * wx1;

            const size_t dst_idx = dst_row + static_cast<size_t>(dx + lb.pad_left);
            ch0[dst_idx] = (w00 * rgb[idx00 + 0] + w01 * rgb[idx01 + 0] +
                            w10 * rgb[idx10 + 0] + w11 * rgb[idx11 + 0]) / 255.0f;
            ch1[dst_idx] = (w00 * rgb[idx00 + 1] + w01 * rgb[idx01 + 1] +
                            w10 * rgb[idx10 + 1] + w11 * rgb[idx11 + 1]) / 255.0f;
            ch2[dst_idx] = (w00 * rgb[idx00 + 2] + w01 * rgb[idx01 + 2] +
                            w10 * rgb[idx10 + 2] + w11 * rgb[idx11 + 2]) / 255.0f;
        }
    }
}

float computeDIoU(const BBox& a, const BBox& b) noexcept {
    const float iou = computeIoU(a, b);
    const float dx = a.cx() - b.cx();
    const float dy = a.cy() - b.cy();
    const float center_dist2 = dx * dx + dy * dy;
    const float enc_w = std::max(a.x2, b.x2) - std::min(a.x1, b.x1);
    const float enc_h = std::max(a.y2, b.y2) - std::min(a.y1, b.y1);
    const float c2 = enc_w * enc_w + enc_h * enc_h;
    if (c2 <= 1e-6f) {
        return iou;
    }
    return iou - (center_dist2 / c2);
}

float computeContainmentRatio(const BBox& a, const BBox& b) noexcept {
    const float min_area = std::min(a.area(), b.area());
    if (min_area <= 1e-6f) {
        return 0.0f;
    }
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    if (ix2 <= ix1 || iy2 <= iy1) {
        return 0.0f;
    }
    return ((ix2 - ix1) * (iy2 - iy1)) / min_area;
}

struct RawYoloCandidate {
    BBox lb_bbox;  // In 1024x1024 letterbox coordinates
    int class_id = 0;
    float confidence = 0.0f;
    bool is_rescue = false;
    std::array<float, kMaskChannels> mask_coeffs{};
};

// Extract simplified boundary polygon in full image coordinates from a cropped binary mask.
std::vector<Point2f> extractSimplifiedPolygon(const InstanceMask& mask) {
    std::vector<Point2f> poly;
    if (mask.empty()) {
        return poly;
    }

    // Collect left-most and right-most foreground points per row to form a clean hull/contour
    // without OpenCV, then subsample to <= 32 vertices.
    std::vector<Point2f> left_edge;
    std::vector<Point2f> right_edge;
    const int step = std::max(1, mask.height / 16);

    for (int y = 0; y < mask.height; y += step) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(mask.width);
        int min_x = -1;
        int max_x = -1;
        for (int x = 0; x < mask.width; ++x) {
            if (mask.data[row + static_cast<size_t>(x)] != 0) {
                if (min_x < 0) min_x = x;
                max_x = x;
            }
        }
        if (min_x >= 0) {
            left_edge.push_back(Point2f{
                static_cast<float>(mask.offset_x + min_x),
                static_cast<float>(mask.offset_y + y)
            });
            if (max_x != min_x) {
                right_edge.push_back(Point2f{
                    static_cast<float>(mask.offset_x + max_x),
                    static_cast<float>(mask.offset_y + y)
                });
            }
        }
    }

    poly.reserve(left_edge.size() + right_edge.size());
    for (const auto& pt : left_edge) {
        poly.push_back(pt);
    }
    for (auto it = right_edge.rbegin(); it != right_edge.rend(); ++it) {
        poly.push_back(*it);
    }
    return poly;
}

std::vector<float> extractTensorAsFloat(const Ort::Value& val) {
    const auto info = val.GetTensorTypeAndShapeInfo();
    const size_t count = info.GetElementCount();
    const auto elem_type = info.GetElementType();

    std::vector<float> out(count);
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const auto* ptr = val.GetTensorData<Ort::Float16_t>();
        for (size_t i = 0; i < count; ++i) {
            out[i] = halfBitsToFloat(ptr[i].val);
        }
    } else {
        const auto* ptr = val.GetTensorData<float>();
        std::memcpy(out.data(), ptr, count * sizeof(float));
    }
    return out;
}

}  // namespace

Ort::Env& Yolo11SegBackend::sharedEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "VellumYolo11Seg");
    return env;
}

int Yolo11SegBackend::queryPhysicalCoreCount() noexcept {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len > 0) {
        std::vector<uint8_t> buf(len);
        if (GetLogicalProcessorInformationEx(
                RelationProcessorCore,
                reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
                &len)) {
            int cores = 0;
            DWORD offset = 0;
            while (offset < len) {
                const auto* info =
                    reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + offset);
                if (info->Relationship == RelationProcessorCore) {
                    ++cores;
                }
                offset += info->Size;
            }
            if (cores > 0) {
                return cores;
            }
        }
    }
    const unsigned int hw = std::thread::hardware_concurrency();
    return static_cast<int>(std::max(1u, hw));
}

bool Yolo11SegBackend::load(const DetectorConfig& config) {
    if (config.model_path.empty()) {
        return false;
    }

    if (session_ && loaded_model_path_ == config.model_path &&
        loaded_force_cpu_ == config.force_cpu) {
        return true;
    }

    unload();

    const std::filesystem::path model_fs_path = std::filesystem::u8path(config.model_path);
    if (!std::filesystem::exists(model_fs_path)) {
        return false;
    }

    const int physical_cores = queryPhysicalCoreCount();
    const int gpu_host_threads = std::max(1, std::min(2, physical_cores / 2));

    const std::vector<std::string> providers =
        mangaflow::pipeline::get_ordered_execution_providers(config.force_cpu);

    auto tryCreateSession = [&](const std::string& provider) -> std::unique_ptr<Ort::Session> {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        opts.SetInterOpNumThreads(1);
        mangaflow::pipeline::enable_shared_env_allocators(opts);

        if (provider == "CPUExecutionProvider") {
            opts.SetIntraOpNumThreads(physical_cores);
            opts.AddConfigEntry("session.intra_op.allow_spinning", "1");
            return std::make_unique<Ort::Session>(sharedEnv(), model_fs_path.c_str(), opts);
        }

        opts.SetIntraOpNumThreads(gpu_host_threads);
        opts.AddConfigEntry("session.intra_op.allow_spinning", "0");

        if (!mangaflow::pipeline::append_gpu_execution_provider(opts, provider, /*device_id=*/0)) {
            return nullptr;
        }
        return std::make_unique<Ort::Session>(sharedEnv(), model_fs_path.c_str(), opts);
    };

    for (const auto& provider : providers) {
        try {
            auto sess = tryCreateSession(provider);
            if (sess) {
                session_ = std::move(sess);
                active_provider_ = provider;
                break;
            }
        } catch (...) {
            // Fall through to next execution provider
        }
    }

    if (!session_) {
        return false;
    }

    Ort::AllocatorWithDefaultOptions allocator;
    if (session_->GetInputCount() < 1 || session_->GetOutputCount() < 2) {
        unload();
        return false;
    }

    {
        auto in_name = session_->GetInputNameAllocated(0, allocator);
        input_name_ = in_name.get();
        const auto type_info = session_->GetInputTypeInfo(0);
        const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_elem_type_ = tensor_info.GetElementType();
        const auto shape = tensor_info.GetShape();
        if (shape.size() == 4) {
            if (shape[2] > 0) input_height_ = static_cast<int>(shape[2]);
            if (shape[3] > 0) input_width_ = static_cast<int>(shape[3]);
        }
    }

    output_names_.clear();
    const size_t out_count = session_->GetOutputCount();
    output_names_.reserve(out_count);
    for (size_t i = 0; i < out_count; ++i) {
        auto out_name = session_->GetOutputNameAllocated(i, allocator);
        output_names_.emplace_back(out_name.get());
    }

    loaded_model_path_ = config.model_path;
    loaded_force_cpu_ = config.force_cpu;
    model_name_ = model_fs_path.stem().u8string();
    return true;
}

PageDetectionResult Yolo11SegBackend::detectPage(
    const uint8_t* rgbData,
    int width,
    int height,
    const std::string& filename,
    const DetectorConfig& config) {
    if (!rgbData || width <= 0 || height <= 0) {
        throw std::invalid_argument("Yolo11SegBackend::detectPage: invalid RGB buffer or dimensions.");
    }

    if (!session_ && !load(config)) {
        throw std::runtime_error("Yolo11SegBackend::detectPage: failed to load ONNX model.");
    }

    const auto t_start = std::chrono::steady_clock::now();

    // #18: Opt-In RANSAC + 3x3 Homography Deskew & Lightweight TPS Spine Dewarp (0 ms when disabled).
    DeskewResult deskew_res;
    const uint8_t* active_rgb = rgbData;
    if (config.auto_deskew_scans && !config.is_subpass) {
        deskew_res = deskewAndDewarpScan(rgbData, width, height);
        if (deskew_res.applied && deskew_res.warped_rgb.size() ==
            static_cast<size_t>(width) * static_cast<size_t>(height) * 3ULL) {
            active_rgb = deskew_res.warped_rgb.data();
        }
    }

    const int dst_w = (input_width_ > 0) ? input_width_ : kYoloInputSize;
    const int dst_h = (input_height_ > 0) ? input_height_ : kYoloInputSize;
    // Keep Auto-Margin Trim disabled in Fast mode for zero-drift 1-pass stability.
    const TrimRect trim = (config.precision == DetectorPrecision::Fast)
        ? TrimRect{0, 0, width, height}
        : computeAutoMarginTrim(active_rgb, width, height);
    const LetterboxInfo lb = computeLetterbox(trim.w, trim.h, dst_w, dst_h);

    std::vector<float> input_fp32;
    letterboxRgbToNchwFloat(active_rgb, width, trim, lb, dst_w, dst_h, input_fp32);

    const std::array<int64_t, 4> input_shape{1, 3, dst_h, dst_w};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Float16_t> input_fp16;
    Ort::Value input_tensor{nullptr};
    if (input_elem_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        input_fp16.resize(input_fp32.size());
        for (size_t i = 0; i < input_fp32.size(); ++i) {
            input_fp16[i] = Ort::Float16_t(input_fp32[i]);
        }
        input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
            mem_info, input_fp16.data(), input_fp16.size(), input_shape.data(), input_shape.size());
    } else {
        input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_fp32.data(), input_fp32.size(), input_shape.data(), input_shape.size());
    }

    const char* in_names[] = {input_name_.c_str()};
    std::vector<const char*> out_names;
    out_names.reserve(output_names_.size());
    for (const auto& s : output_names_) {
        out_names.push_back(s.c_str());
    }

    static std::atomic<uint64_t> s_yolo_runs{0};
    const bool want_shrink = mangaflow::pipeline::should_shrink_arena_now(s_yolo_runs, 5);
    const bool is_cuda_ep = (active_provider_ == "CUDAExecutionProvider");
    Ort::RunOptions run_opts = want_shrink
        ? mangaflow::pipeline::make_shrink_run_options(true, is_cuda_ep)
        : Ort::RunOptions{nullptr};

    std::vector<Ort::Value> outputs;
    try {
        outputs = session_->Run(
            run_opts,
            in_names, &input_tensor, 1,
            out_names.data(), out_names.size());
    } catch (const Ort::Exception& ex) {
        std::fprintf(stderr, "[Vellum][YOLO11m-Seg] %s inference failed: %s\n", active_provider_.c_str(), ex.what());
        if (active_provider_ == "CPUExecutionProvider") {
            throw;
        }
        DetectorConfig cpu_cfg = config;
        cpu_cfg.force_cpu = true;
        if (!load(cpu_cfg) || !session_) {
            throw;
        }
        Ort::RunOptions cpu_ro = want_shrink
            ? mangaflow::pipeline::make_shrink_run_options(true, false)
            : Ort::RunOptions{nullptr};
        outputs = session_->Run(
            cpu_ro,
            in_names, &input_tensor, 1,
            out_names.data(), out_names.size());
    }

    if (mangaflow::pipeline::query_system_ram_stats().ram_load_pct >= mangaflow::pipeline::kRamPressureGuardPct) {
        mangaflow::pipeline::trim_process_working_set_throttled(15);
    }

    const auto t_infer = std::chrono::steady_clock::now();

    // Locate output0 (detections: 3D) and output1 (protos: 4D)
    size_t det_idx = 0;
    size_t proto_idx = 1;
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto shape = outputs[i].GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() == 3) det_idx = i;
        if (shape.size() == 4) proto_idx = i;
    }

    const auto det_shape = outputs[det_idx].GetTensorTypeAndShapeInfo().GetShape();
    const auto proto_shape = outputs[proto_idx].GetTensorTypeAndShapeInfo().GetShape();
    const std::vector<float> det_data = extractTensorAsFloat(outputs[det_idx]);
    const std::vector<float> proto_data = extractTensorAsFloat(outputs[proto_idx]);

    // Infer num_classes dynamically from output0: [1, 4 + num_classes + 32, num_anchors]
    const int64_t dim1 = (det_shape.size() >= 2) ? det_shape[1] : 0;
    const int64_t dim2 = (det_shape.size() >= 3) ? det_shape[2] : 0;
    const bool channels_first = (dim1 <= dim2);
    const int channels = static_cast<int>(channels_first ? dim1 : dim2);
    const int num_anchors = static_cast<int>(channels_first ? dim2 : dim1);
    const int num_classes = std::max(0, channels - 4 - kMaskChannels);

    const int proto_c = (proto_shape.size() == 4) ? static_cast<int>(proto_shape[1]) : kMaskChannels;
    const int proto_h = (proto_shape.size() == 4) ? static_cast<int>(proto_shape[2]) : 256;
    const int proto_w = (proto_shape.size() == 4) ? static_cast<int>(proto_shape[3]) : 256;

    std::vector<RawYoloCandidate> candidates;
    if (num_classes > 0 && num_anchors > 0) {
        auto getDetVal = [&](int ch, int anchor) -> float {
            if (channels_first) {
                return det_data[static_cast<size_t>(ch) * static_cast<size_t>(num_anchors) +
                                static_cast<size_t>(anchor)];
            }
            return det_data[static_cast<size_t>(anchor) * static_cast<size_t>(channels) +
                            static_cast<size_t>(ch)];
        };

        const bool enable_dual_rescue = (config.precision != DetectorPrecision::Fast);

        for (int a = 0; a < num_anchors; ++a) {
            int best_cls = -1;
            float best_score = 0.0f;
            bool best_is_rescue = false;
            for (int c = 0; c < num_classes; ++c) {
                const float s = getDetVal(4 + c, a);
                const float thresh = (c < kNumClasses)
                    ? config.class_thresholds[static_cast<size_t>(c)]
                    : 0.35f;
                const bool is_panel_or_sfx =
                    (num_classes == kNumClasses) &&
                    (c == static_cast<int>(ClassId::Panel) || c == static_cast<int>(ClassId::SfxIgnore));
                const float floor_thresh = (enable_dual_rescue && !is_panel_or_sfx)
                    ? std::min(thresh, kDualRescueFloor)
                    : thresh;
                if (s >= floor_thresh && s > best_score) {
                    best_score = s;
                    best_cls = c;
                    best_is_rescue = (s < thresh);
                }
            }
            if (best_cls < 0) {
                continue;
            }

            const float cx = getDetVal(0, a);
            const float cy = getDetVal(1, a);
            const float bw = getDetVal(2, a);
            const float bh = getDetVal(3, a);

            RawYoloCandidate cand;
            cand.lb_bbox = BBox{cx - 0.5f * bw, cy - 0.5f * bh, cx + 0.5f * bw, cy + 0.5f * bh};
            cand.class_id = best_cls;
            cand.confidence = best_score;
            cand.is_rescue = best_is_rescue;
            for (int m = 0; m < kMaskChannels; ++m) {
                cand.mask_coeffs[static_cast<size_t>(m)] = getDetVal(4 + num_classes + m, a);
            }
            candidates.push_back(cand);
        }
    }

    // Class-aware DIoU-NMS so tightly adjacent/conjoined bubbles with distinct centers survive.
    std::sort(candidates.begin(), candidates.end(),
              [](const RawYoloCandidate& a, const RawYoloCandidate& b) {
                  return a.confidence > b.confidence;
              });

    const int max_det = std::max(150, config.max_detections);
    std::vector<RawYoloCandidate> nms_kept;
    nms_kept.reserve(std::min(static_cast<int>(candidates.size()), max_det));
    std::vector<bool> suppressed(candidates.size(), false);

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        nms_kept.push_back(candidates[i]);
        if (static_cast<int>(nms_kept.size()) >= max_det) break;

        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j] || candidates[i].class_id != candidates[j].class_id) continue;
            if (computeDIoU(candidates[i].lb_bbox, candidates[j].lb_bbox) > config.nms_iou_threshold ||
                computeContainmentRatio(candidates[i].lb_bbox, candidates[j].lb_bbox) >= kContainmentSuppressThresh) {
                suppressed[j] = true;
            }
        }
    }

    PageDetectionResult result;
    result.schema_version = "2.0";
    result.filename = filename;
    result.image_width = width;
    result.image_height = height;
    result.reading_direction = config.reading_direction;
    result.runtime_meta.backend = BackendType::Yolo11Seg;
    result.runtime_meta.model_name = model_name_;
    result.runtime_meta.execution_provider = active_provider_;
    result.runtime_meta.tiles_processed = 1;

    const float proto_scale_x = static_cast<float>(proto_w) / static_cast<float>(dst_w);
    const float proto_scale_y = static_cast<float>(proto_h) / static_cast<float>(dst_h);
    const size_t proto_plane = static_cast<size_t>(proto_h) * static_cast<size_t>(proto_w);
    const int use_c = std::min(proto_c, kMaskChannels);

    int next_panel_id = 0;
    int next_det_id = 0;

    std::vector<float> proto_prob_crop;

    auto emitDetection = [&](const RawYoloCandidate& cand, ClassId cls, const BBox& orig_box, const BBox& norm_box) {
        BubbleDetection det;
        det.id = next_det_id++;
        det.class_id = cls;
        det.class_name = class_id_to_string(cls);
        det.confidence = cand.confidence;
        det.bbox = orig_box;
        det.bbox_norm = norm_box;

        // #1: O(N) Guided Image Filtering on upsampled 256x256 float probability mask inside
        // an expanded local bbox ROI guided by full-res grayscale I so masks snap to ink walls.
        const int bx1 = std::clamp(static_cast<int>(std::floor(orig_box.x1)), 0, width - 1);
        const int by1 = std::clamp(static_cast<int>(std::floor(orig_box.y1)), 0, height - 1);
        const int bx2 = std::clamp(static_cast<int>(std::ceil(orig_box.x2)), bx1 + 1, width);
        const int by2 = std::clamp(static_cast<int>(std::ceil(orig_box.y2)), by1 + 1, height);
        const int mw = bx2 - bx1;
        const int mh = by2 - by1;

        if (mw > 0 && mh > 0 && !proto_data.empty()) {
            det.mask.offset_x = bx1;
            det.mask.offset_y = by1;
            det.mask.width = mw;
            det.mask.height = mh;
            det.mask.data.assign(static_cast<size_t>(mw) * static_cast<size_t>(mh), 0);

            constexpr int kGuidePad = 4;
            const int ex1 = std::max(0, bx1 - kGuidePad);
            const int ey1 = std::max(0, by1 - kGuidePad);
            const int ex2 = std::min(width, bx2 + kGuidePad);
            const int ey2 = std::min(height, by2 + kGuidePad);
            const int ew = ex2 - ex1;
            const int eh = ey2 - ey1;

            const float lb_x1 = (static_cast<float>(ex1 - trim.x) + 0.5f) * lb.scale + static_cast<float>(lb.pad_left);
            const float lb_y1 = (static_cast<float>(ey1 - trim.y) + 0.5f) * lb.scale + static_cast<float>(lb.pad_top);
            const float lb_x2 = (static_cast<float>(ex2 - 1 - trim.x) + 0.5f) * lb.scale + static_cast<float>(lb.pad_left);
            const float lb_y2 = (static_cast<float>(ey2 - 1 - trim.y) + 0.5f) * lb.scale + static_cast<float>(lb.pad_top);

            const int px_min = std::clamp(static_cast<int>(std::floor(lb_x1 * proto_scale_x - 0.5f)), 0, proto_w - 1);
            const int py_min = std::clamp(static_cast<int>(std::floor(lb_y1 * proto_scale_y - 0.5f)), 0, proto_h - 1);
            const int px_max = std::clamp(static_cast<int>(std::ceil(lb_x2 * proto_scale_x - 0.5f)), px_min, proto_w - 1);
            const int py_max = std::clamp(static_cast<int>(std::ceil(lb_y2 * proto_scale_y - 0.5f)), py_min, proto_h - 1);
            const int pcw = px_max - px_min + 1;
            const int pch = py_max - py_min + 1;

            proto_prob_crop.resize(static_cast<size_t>(pcw) * static_cast<size_t>(pch));
            for (int py = py_min; py <= py_max; ++py) {
                const size_t dst_prow = static_cast<size_t>(py - py_min) * static_cast<size_t>(pcw);
                const size_t proto_row = static_cast<size_t>(py) * static_cast<size_t>(proto_w);
                for (int px = px_min; px <= px_max; ++px) {
                    float logit = 0.0f;
                    const size_t p_off = proto_row + static_cast<size_t>(px);
                    for (int c = 0; c < use_c; ++c) {
                        logit += cand.mask_coeffs[static_cast<size_t>(c)] *
                                 proto_data[static_cast<size_t>(c) * proto_plane + p_off];
                    }
                    proto_prob_crop[dst_prow + static_cast<size_t>(px - px_min)] = sigmoidf(logit);
                }
            }

            const bool run_guided = (ew >= 8 && eh >= 8 && static_cast<int64_t>(ew) * static_cast<int64_t>(eh) <= 600000);
            if (run_guided) {
                const size_t en = static_cast<size_t>(ew) * static_cast<size_t>(eh);
                std::vector<float> guide_I(en);
                std::vector<float> upsampled_p(en);

                for (int y = 0; y < eh; ++y) {
                    const int gy = ey1 + y;
                    const float crop_y = static_cast<float>(gy - trim.y) + 0.5f;
                    const float lb_y = crop_y * lb.scale + static_cast<float>(lb.pad_top);
                    const float py_f = lb_y * proto_scale_y - 0.5f - static_cast<float>(py_min);
                    const int ly0 = std::clamp(static_cast<int>(std::floor(py_f)), 0, pch - 1);
                    const int ly1 = std::clamp(ly0 + 1, 0, pch - 1);
                    const float wy1 = std::clamp(py_f - static_cast<float>(ly0), 0.0f, 1.0f);
                    const float wy0 = 1.0f - wy1;

                    const size_t prow0 = static_cast<size_t>(ly0) * static_cast<size_t>(pcw);
                    const size_t prow1 = static_cast<size_t>(ly1) * static_cast<size_t>(pcw);
                    const size_t rgb_row = static_cast<size_t>(gy) * static_cast<size_t>(width) * 3ULL;
                    const size_t erow = static_cast<size_t>(y) * static_cast<size_t>(ew);

                    for (int x = 0; x < ew; ++x) {
                        const int gx = ex1 + x;
                        const size_t rgb_off = rgb_row + static_cast<size_t>(gx) * 3ULL;
                        guide_I[erow + static_cast<size_t>(x)] =
                            static_cast<float>(rgbLuma(active_rgb[rgb_off], active_rgb[rgb_off + 1], active_rgb[rgb_off + 2])) / 255.0f;

                        const float crop_x = static_cast<float>(gx - trim.x) + 0.5f;
                        const float lb_x = crop_x * lb.scale + static_cast<float>(lb.pad_left);
                        const float px_f = lb_x * proto_scale_x - 0.5f - static_cast<float>(px_min);
                        const int lx0 = std::clamp(static_cast<int>(std::floor(px_f)), 0, pcw - 1);
                        const int lx1 = std::clamp(lx0 + 1, 0, pcw - 1);
                        const float wx1 = std::clamp(px_f - static_cast<float>(lx0), 0.0f, 1.0f);
                        const float wx0 = 1.0f - wx1;

                        upsampled_p[erow + static_cast<size_t>(x)] =
                            wy0 * (wx0 * proto_prob_crop[prow0 + static_cast<size_t>(lx0)] +
                                   wx1 * proto_prob_crop[prow0 + static_cast<size_t>(lx1)]) +
                            wy1 * (wx0 * proto_prob_crop[prow1 + static_cast<size_t>(lx0)] +
                                   wx1 * proto_prob_crop[prow1 + static_cast<size_t>(lx1)]);
                    }
                }

                std::vector<float> filtered_q;
                applyGuidedFilter(guide_I.data(), upsampled_p.data(), ew, eh, /*radius=*/4, /*eps=*/1e-3f, filtered_q);

                const int off_x = bx1 - ex1;
                const int off_y = by1 - ey1;
                double area_px = 0.0;
                for (int y = 0; y < mh; ++y) {
                    const size_t q_row = static_cast<size_t>(y + off_y) * static_cast<size_t>(ew) + static_cast<size_t>(off_x);
                    const size_t dst_row = static_cast<size_t>(y) * static_cast<size_t>(mw);
                    for (int x = 0; x < mw; ++x) {
                        if (filtered_q[q_row + static_cast<size_t>(x)] >= kMaskThreshold) {
                            det.mask.data[dst_row + static_cast<size_t>(x)] = 255;
                            area_px += 1.0;
                        }
                    }
                }
                det.mask.area_px = area_px;
            } else {
                double area_px = 0.0;
                for (int y = 0; y < mh; ++y) {
                    const float crop_y = static_cast<float>(by1 + y - trim.y) + 0.5f;
                    const float lb_y = crop_y * lb.scale + static_cast<float>(lb.pad_top);
                    const float py_f = lb_y * proto_scale_y - 0.5f - static_cast<float>(py_min);
                    const int ly0 = std::clamp(static_cast<int>(std::floor(py_f)), 0, pch - 1);
                    const int ly1 = std::clamp(ly0 + 1, 0, pch - 1);
                    const float wy1 = std::clamp(py_f - static_cast<float>(ly0), 0.0f, 1.0f);
                    const float wy0 = 1.0f - wy1;

                    const size_t prow0 = static_cast<size_t>(ly0) * static_cast<size_t>(pcw);
                    const size_t prow1 = static_cast<size_t>(ly1) * static_cast<size_t>(pcw);
                    const size_t dst_row = static_cast<size_t>(y) * static_cast<size_t>(mw);
                    for (int x = 0; x < mw; ++x) {
                        const float crop_x = static_cast<float>(bx1 + x - trim.x) + 0.5f;
                        const float lb_x = crop_x * lb.scale + static_cast<float>(lb.pad_left);
                        const float px_f = lb_x * proto_scale_x - 0.5f - static_cast<float>(px_min);
                        const int lx0 = std::clamp(static_cast<int>(std::floor(px_f)), 0, pcw - 1);
                        const int lx1 = std::clamp(lx0 + 1, 0, pcw - 1);
                        const float wx1 = std::clamp(px_f - static_cast<float>(lx0), 0.0f, 1.0f);
                        const float wx0 = 1.0f - wx1;

                        const float prob =
                            wy0 * (wx0 * proto_prob_crop[prow0 + static_cast<size_t>(lx0)] +
                                   wx1 * proto_prob_crop[prow0 + static_cast<size_t>(lx1)]) +
                            wy1 * (wx0 * proto_prob_crop[prow1 + static_cast<size_t>(lx0)] +
                                   wx1 * proto_prob_crop[prow1 + static_cast<size_t>(lx1)]);

                        if (prob >= kMaskThreshold) {
                            det.mask.data[dst_row + static_cast<size_t>(x)] = 255;
                            area_px += 1.0;
                        }
                    }
                }
                det.mask.area_px = area_px;
            }
            det.mask.polygon = extractSimplifiedPolygon(det.mask);
        }

        result.detections.push_back(std::move(det));
    };

    struct CriticRejectedCand {
        RawYoloCandidate cand;
        ClassId cls;
        BBox orig_box;
        BBox norm_box;
    };
    std::vector<CriticRejectedCand> critic_rejected;

    for (const auto& cand : nms_kept) {
        // Un-letterbox box and add back (trim.x, trim.y) to native page coordinates
        const float ox1 = std::clamp(
            (cand.lb_bbox.x1 - static_cast<float>(lb.pad_left)) / lb.scale + static_cast<float>(trim.x),
            0.0f, static_cast<float>(width));
        const float oy1 = std::clamp(
            (cand.lb_bbox.y1 - static_cast<float>(lb.pad_top)) / lb.scale + static_cast<float>(trim.y),
            0.0f, static_cast<float>(height));
        const float ox2 = std::clamp(
            (cand.lb_bbox.x2 - static_cast<float>(lb.pad_left)) / lb.scale + static_cast<float>(trim.x),
            0.0f, static_cast<float>(width));
        const float oy2 = std::clamp(
            (cand.lb_bbox.y2 - static_cast<float>(lb.pad_top)) / lb.scale + static_cast<float>(trim.y),
            0.0f, static_cast<float>(height));
        BBox orig_box{ox1, oy1, ox2, oy2};
        if (!orig_box.is_valid()) {
            continue;
        }

        BBox norm_box{
            ox1 / static_cast<float>(width),
            oy1 / static_cast<float>(height),
            ox2 / static_cast<float>(width),
            oy2 / static_cast<float>(height)
        };

        // Single-class or compact multi-class open-source manga bubble models (e.g. 1..3 bubble classes)
        // map cleanly to ClassId::SpeechBubble, while full 9-class Vellum models preserve their taxonomy.
        ClassId cls = ClassId::SpeechBubble;
        if (num_classes == kNumClasses && cand.class_id >= 0 && cand.class_id < kNumClasses) {
            cls = static_cast<ClassId>(cand.class_id);
        } else if (num_classes > kNumClasses && cand.class_id >= kNumClasses) {
            cls = ClassId::TextRegion;
        }

        if (cls == ClassId::Panel) {
            PanelDetection p;
            p.id = next_panel_id++;
            p.confidence = cand.confidence;
            p.bbox = orig_box;
            p.bbox_norm = norm_box;
            result.panels.push_back(std::move(p));
            continue;
        }

        // 0-VRAM CV Ink Critic: reject empty white regions, solid black blocks, or screentone-only FPs.
        // Never drop high-confidence (>= 0.60) speech bubbles (e.g. containing small "..." punctuation).
        if (cls != ClassId::SfxIgnore) {
            const bool bypass_critic = (!cand.is_rescue && cand.confidence >= kHighConfCriticBypass);
            if (!bypass_critic && !passesInkCritic(rgbData, width, height, orig_box, cand.is_rescue)) {
                critic_rejected.push_back(CriticRejectedCand{cand, cls, orig_box, norm_box});
                continue;
            }
        }

        emitDetection(cand, cls, orig_box, norm_box);
    }

    // Zero-bubble rescue pass (0 extra ONNX forward passes): when standard pass yields 0 speech/text
    // bubbles on a non-blank page, re-evaluate critic-rejected stylized/hollow-text candidates at a
    // slightly relaxed confidence floor (0.18 balanced/fast, 0.14 high_precision) using outline/edge density.
    const bool has_non_sfx_bubble = std::any_of(
        result.detections.begin(),
        result.detections.end(),
        [](const BubbleDetection& d) { return d.class_id != ClassId::SfxIgnore; });
    if (!has_non_sfx_bubble && !critic_rejected.empty() && isPageNonBlank(rgbData, width, height)) {
        const float zero_rescue_conf =
            (config.precision == DetectorPrecision::HighPrecision) ? 0.14f : 0.18f;
        for (const auto& rc : critic_rejected) {
            if (rc.cand.confidence < zero_rescue_conf) {
                continue;
            }
            if (passesHollowOutlineCritic(rgbData, width, height, rc.orig_box)) {
                emitDetection(rc.cand, rc.cls, rc.orig_box, rc.norm_box);
            }
        }
    }

    if (!config.is_subpass) {
        runGeometryPostproc(result, config, active_rgb);
        applyV25CvStack(result, active_rgb, width, height, config);
    }

    const auto t_end = std::chrono::steady_clock::now();
    result.runtime_meta.inference_ms =
        std::chrono::duration<double, std::milli>(t_infer - t_start).count();
    result.runtime_meta.postproc_ms =
        std::chrono::duration<double, std::milli>(t_end - t_infer).count();

    if (!config.keep_session_alive && !config.is_subpass) {
        unload();
    }

    return result;
}

bool Yolo11SegBackend::passesInkCritic(
    const uint8_t* rgbData,
    int width,
    int height,
    const BBox& bbox,
    bool strict_rescue) noexcept {
    if (!rgbData || width <= 0 || height <= 0 || !bbox.is_valid()) {
        return false;
    }

    const int raw_x1 = std::clamp(static_cast<int>(std::floor(bbox.x1)), 0, width - 1);
    const int raw_y1 = std::clamp(static_cast<int>(std::floor(bbox.y1)), 0, height - 1);
    const int raw_x2 = std::clamp(static_cast<int>(std::ceil(bbox.x2)), raw_x1 + 1, width);
    const int raw_y2 = std::clamp(static_cast<int>(std::ceil(bbox.y2)), raw_y1 + 1, height);
    const int bw = raw_x2 - raw_x1;
    const int bh = raw_y2 - raw_y1;
    if (bw < 6 || bh < 6) {
        return false;
    }
    if (strict_rescue && (bw < 10 || bh < 10)) {
        return false;
    }

    // Inset by ~10% so bubble outlines do not pollute the interior ink check.
    const int inset_x = (bw >= 12) ? std::clamp(bw / 10, 1, 8) : 0;
    const int inset_y = (bh >= 12) ? std::clamp(bh / 10, 1, 8) : 0;
    const int ix1 = raw_x1 + inset_x;
    const int iy1 = raw_y1 + inset_y;
    const int ix2 = std::max(ix1 + 1, raw_x2 - inset_x);
    const int iy2 = std::max(iy1 + 1, raw_y2 - inset_y);
    const int iw = ix2 - ix1;
    const int ih = iy2 - iy1;

    // Sample onto a compact <= 80x80 grid (< 0.2 ms/box).
    const int step_x = std::max(1, (iw + 79) / 80);
    const int step_y = std::max(1, (ih + 79) / 80);
    const int gw = (iw + step_x - 1) / step_x;
    const int gh = (ih + step_y - 1) / step_y;
    if (gw <= 0 || gh <= 0) {
        return false;
    }

    std::vector<uint8_t> gray(static_cast<size_t>(gw) * static_cast<size_t>(gh));
    std::vector<uint8_t> border_luma;
    border_luma.reserve(static_cast<size_t>(2 * (gw + gh)));

    for (int gy = 0; gy < gh; ++gy) {
        const int sy = std::min(iy2 - 1, iy1 + gy * step_y);
        const size_t src_row = static_cast<size_t>(sy) * static_cast<size_t>(width);
        const size_t dst_row = static_cast<size_t>(gy) * static_cast<size_t>(gw);
        for (int gx = 0; gx < gw; ++gx) {
            const int sx = std::min(ix2 - 1, ix1 + gx * step_x);
            const size_t idx = (src_row + static_cast<size_t>(sx)) * 3ULL;
            const uint8_t lum = static_cast<uint8_t>(rgbLuma(rgbData[idx], rgbData[idx + 1], rgbData[idx + 2]));
            gray[dst_row + static_cast<size_t>(gx)] = lum;
            if (gy == 0 || gy == gh - 1 || gx == 0 || gx == gw - 1) {
                border_luma.push_back(lum);
            }
        }
    }

    int bg_luma = 240;
    if (!border_luma.empty()) {
        const size_t mid = border_luma.size() / 2;
        std::nth_element(border_luma.begin(), border_luma.begin() + static_cast<ptrdiff_t>(mid), border_luma.end());
        bg_luma = static_cast<int>(border_luma[mid]);
    }

    const bool dark_bg = (bg_luma < 128);
    const int thresh = dark_bg
        ? std::clamp(bg_luma + 42, 65, 195)
        : std::clamp(bg_luma - 42, 60, 195);

    std::vector<uint8_t> ink(gray.size(), 0);
    int ink_count = 0;
    for (size_t i = 0; i < gray.size(); ++i) {
        const int v = static_cast<int>(gray[i]);
        const bool is_ink = dark_bg ? (v > thresh) : (v < thresh);
        if (is_ink) {
            ink[i] = 1;
            ++ink_count;
        }
    }

    const float total_px = static_cast<float>(gray.size());
    const float ink_ratio = static_cast<float>(ink_count) / total_px;
    if (ink_ratio < 0.005f || ink_ratio > 0.78f) {
        return false;
    }

    // Count distinct 4-connected ink components (size >= 2 px) to filter single-speck noise.
    std::vector<uint8_t> visited(ink.size(), 0);
    std::vector<int> queue;
    queue.reserve(256);
    int distinct_components = 0;

    for (int y = 0; y < gh; ++y) {
        for (int x = 0; x < gw; ++x) {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(gw) + static_cast<size_t>(x);
            if (!ink[idx] || visited[idx]) continue;

            visited[idx] = 1;
            queue.clear();
            queue.push_back(static_cast<int>(idx));
            size_t head = 0;
            while (head < queue.size()) {
                const int cur = queue[head++];
                const int cy = cur / gw;
                const int cx = cur % gw;
                const int nx[4] = {cx - 1, cx + 1, cx, cx};
                const int ny[4] = {cy, cy, cy - 1, cy + 1};
                for (int k = 0; k < 4; ++k) {
                    if (nx[k] < 0 || nx[k] >= gw || ny[k] < 0 || ny[k] >= gh) continue;
                    const size_t nidx = static_cast<size_t>(ny[k]) * static_cast<size_t>(gw) + static_cast<size_t>(nx[k]);
                    if (ink[nidx] && !visited[nidx]) {
                        visited[nidx] = 1;
                        queue.push_back(static_cast<int>(nidx));
                    }
                }
            }
            if (queue.size() >= 2) {
                ++distinct_components;
            }
        }
    }

    if (distinct_components < 1) {
        return false;
    }
    if (strict_rescue && distinct_components < 2 && ink_ratio < 0.02f) {
        return false;
    }

    // Uniform screentone-only check: low spatial variance across 4x4 sub-grid + high 1px flip rate.
    if (gw >= 8 && gh >= 8) {
        float cell_ratios[16]{};
        int cell_count = 0;
        float min_cell = 1.0f;
        float sum_cell = 0.0f;
        for (int cy = 0; cy < 4; ++cy) {
            const int y0 = (cy * gh) / 4;
            const int y1 = ((cy + 1) * gh) / 4;
            for (int cx = 0; cx < 4; ++cx) {
                const int x0 = (cx * gw) / 4;
                const int x1 = ((cx + 1) * gw) / 4;
                int c_ink = 0;
                int c_tot = 0;
                for (int y = y0; y < y1; ++y) {
                    const size_t row = static_cast<size_t>(y) * static_cast<size_t>(gw);
                    for (int x = x0; x < x1; ++x) {
                        c_ink += ink[row + static_cast<size_t>(x)];
                        ++c_tot;
                    }
                }
                const float r = (c_tot > 0) ? (static_cast<float>(c_ink) / static_cast<float>(c_tot)) : 0.0f;
                cell_ratios[cell_count++] = r;
                min_cell = std::min(min_cell, r);
                sum_cell += r;
            }
        }
        const float mean_cell = sum_cell / static_cast<float>(cell_count);
        float var = 0.0f;
        for (int i = 0; i < cell_count; ++i) {
            const float d = cell_ratios[i] - mean_cell;
            var += d * d;
        }
        const float std_cell = std::sqrt(var / static_cast<float>(cell_count));

        int transitions = 0;
        int pairs = 0;
        for (int y = 0; y < gh; ++y) {
            const size_t row = static_cast<size_t>(y) * static_cast<size_t>(gw);
            for (int x = 0; x + 1 < gw; ++x) {
                transitions += (ink[row + static_cast<size_t>(x)] != ink[row + static_cast<size_t>(x + 1)]) ? 1 : 0;
                ++pairs;
            }
        }
        for (int y = 0; y + 1 < gh; ++y) {
            const size_t row0 = static_cast<size_t>(y) * static_cast<size_t>(gw);
            const size_t row1 = static_cast<size_t>(y + 1) * static_cast<size_t>(gw);
            for (int x = 0; x < gw; ++x) {
                transitions += (ink[row0 + static_cast<size_t>(x)] != ink[row1 + static_cast<size_t>(x)]) ? 1 : 0;
                ++pairs;
            }
        }
        const float trans_rate = (pairs > 0) ? (static_cast<float>(transitions) / static_cast<float>(pairs)) : 0.0f;

        if (std_cell < 0.045f && min_cell > 0.05f && trans_rate > 0.38f) {
            return false;
        }
    }

    return true;
}

void Yolo11SegBackend::applyV25CvStack(
    PageDetectionResult& result,
    const uint8_t* rgbData,
    int width,
    int height,
    const DetectorConfig& config) {
    // Wire full 0-VRAM v2.5 CV stack (5-stage Otsu glyph refinement, waist-cut watershed
    // room polygons <= 32 vertices, and typography/color/outline metadata) into YOLO11m-Seg.
    if (result.detections.empty() || !rgbData || width <= 0 || height <= 0) {
        return;
    }

    std::vector<RefinementBlock> ref_blocks;
    ref_blocks.reserve(result.detections.size());
    std::vector<RoomSeedBlock> seeds;
    seeds.reserve(result.detections.size());
    std::vector<RectI> bubble_rects;
    bubble_rects.reserve(result.detections.size());

    const bool is_rtl = (config.reading_direction == ReadingDirection::RTL);
    for (size_t i = 0; i < result.detections.size(); ++i) {
        const auto& d = result.detections[i];
        const BBox& core = d.safe_bbox.is_valid() ? d.safe_bbox : d.bbox;
        const RectI core_rect = RectI{
            static_cast<int>(std::floor(core.x1)),
            static_cast<int>(std::floor(core.y1)),
            static_cast<int>(std::ceil(core.x2)),
            static_cast<int>(std::ceil(core.y2))
        }.clamped(width, height);
        const RectI full_rect = RectI{
            static_cast<int>(std::floor(d.bbox.x1)),
            static_cast<int>(std::floor(d.bbox.y1)),
            static_cast<int>(std::ceil(d.bbox.x2)),
            static_cast<int>(std::ceil(d.bbox.y2))
        }.clamped(width, height);

        RefinementBlock rb;
        rb.original_index = static_cast<int>(i);
        rb.class_id = (d.class_id == ClassId::SfxIgnore) ? 1 : 0;
        rb.score = d.confidence;
        rb.bbox = core_rect;
        rb.is_vertical = is_rtl && (core_rect.height() >= core_rect.width());
        if (!d.mask.empty()) {
            rb.raw_mask_rect = RectI{
                d.mask.offset_x,
                d.mask.offset_y,
                d.mask.offset_x + d.mask.width,
                d.mask.offset_y + d.mask.height
            }.clamped(width, height);
            rb.raw_mask = d.mask.data;
        }
        ref_blocks.push_back(std::move(rb));

        RoomSeedBlock seed;
        seed.block_index = static_cast<int>(i);
        seed.bbox = core_rect;
        seeds.push_back(std::move(seed));

        if (!full_rect.empty()) {
            bubble_rects.push_back(full_rect);
        }
    }

    // 1. 5-Stage Otsu + Plain-Paper Glyph Mask Refinement (zero bubble-border bleed)
    const std::vector<uint8_t> refined_page_mask =
        refineTextMasks(rgbData, width, height, ref_blocks);

    // 2. Conjoined Balloon Waist-Cutting + Watershed Room Polygons (<= 32 vertices)
    const auto room_polys = computePageRoomPolygons(
        rgbData, width, height, seeds, bubble_rects, &refined_page_mask, config.enable_waist_cut);
    for (size_t i = 0; i < result.detections.size() && i < room_polys.size(); ++i) {
        result.detections[i].room_polygon = room_polys[i];
    }

    // #15: Discrete Curvature + Ray-Casting Speaker Clustering
    assignSpeakerClusters(result.detections, result.panels);

    // 3. 0-VRAM CV Typography & Colorimetry (tilt, font size, fill color, outline)
    // Pass nullptr for optional_mask so Otsu extracts interior text strokes strictly
    // inside `safe_bbox` (avoiding YOLO11m-Seg's full-bubble wall mask skewing A_max).
    if (config.enable_cv_style_detect) {
        for (size_t i = 0; i < result.detections.size(); ++i) {
            auto& d = result.detections[i];
            RectI tb = ref_blocks[i].bbox;
            // Inset by 2px when box is large enough so stray bubble border pixels are excluded
            if (tb.width() >= 20 && tb.height() >= 20) {
                tb = RectI{tb.x1 + 2, tb.y1 + 2, tb.x2 - 2, tb.y2 - 2}.clamped(width, height);
            }
            const bool is_vert = is_rtl && (tb.height() > static_cast<int>(tb.width() * 1.15));
            const bool is_sfx = (d.class_id == ClassId::SfxIgnore);
            const auto style = analyzeBlockTypographyFromBox(
                rgbData, width, height, tb, is_vert, is_sfx, nullptr, nullptr);
            if (style.valid) {
                d.angle_deg = style.angle_deg;
                d.estimated_font_size = style.estimated_font_size;
                d.text_color = style.text_color;
                if (style.outline.has_value()) {
                    d.outline = BubbleOutlineInfo{style.outline->color, style.outline->width};
                } else {
                    d.outline = std::nullopt;
                }
            }
        }
    }
}

void Yolo11SegBackend::unload() {
    session_.reset();
    loaded_model_path_.clear();
    input_name_.clear();
    output_names_.clear();
}

std::string Yolo11SegBackend::activeModelName() const {
    return model_name_;
}

std::string Yolo11SegBackend::executionProvider() const {
    return active_provider_;
}

}  // namespace vellum::detector
