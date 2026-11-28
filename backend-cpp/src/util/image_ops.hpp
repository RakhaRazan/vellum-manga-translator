#pragma once

// Shared low-level image & encoding utilities for Vellum C++17 pipeline stages.
// Consolidates stb_image RAII ownership, bilinear RGB resizing, and re-exports util::base64_encode.

#include "../utils.hpp"
#include <stb_image.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace mangaflow::util {

struct StbiDeleter {
    void operator()(unsigned char* p) const noexcept {
        if (p) stbi_image_free(p);
    }
};
using StbiPtr = std::unique_ptr<unsigned char, StbiDeleter>;

// Bilinear resize of interleaved RGB uint8 image `(src_w, src_h)` -> `(dst_w, dst_h)`
// using align_corners=false pixel-center mapping (matches PIL BILINEAR & cv2.INTER_LINEAR).
inline std::vector<uint8_t> resize_rgb_bilinear(
    const uint8_t* src,
    int src_w,
    int src_h,
    int dst_w,
    int dst_h) {
    std::vector<uint8_t> dst(static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h) * 3ULL, 0);
    if (!src || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        return dst;
    }

    if (src_w == dst_w && src_h == dst_h) {
        std::memcpy(dst.data(), src, dst.size());
        return dst;
    }

    const float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);

    std::vector<int> x0_lut(static_cast<size_t>(dst_w));
    std::vector<int> x1_lut(static_cast<size_t>(dst_w));
    std::vector<float> wx1_lut(static_cast<size_t>(dst_w));

    const float max_sx = static_cast<float>(src_w - 1);
    const float max_sy = static_cast<float>(src_h - 1);
    for (int x = 0; x < dst_w; ++x) {
        const float sx = std::clamp((static_cast<float>(x) + 0.5f) * scale_x - 0.5f, 0.0f, max_sx);
        const int x0 = static_cast<int>(std::floor(sx));
        const int x1 = std::min(x0 + 1, src_w - 1);
        x0_lut[static_cast<size_t>(x)] = x0;
        x1_lut[static_cast<size_t>(x)] = x1;
        wx1_lut[static_cast<size_t>(x)] = sx - static_cast<float>(x0);
    }

    for (int y = 0; y < dst_h; ++y) {
        const float sy = std::clamp((static_cast<float>(y) + 0.5f) * scale_y - 0.5f, 0.0f, max_sy);
        const int y0 = static_cast<int>(std::floor(sy));
        const int y1 = std::min(y0 + 1, src_h - 1);
        const float wy1 = sy - static_cast<float>(y0);
        const float wy0 = 1.0f - wy1;

        const uint8_t* row0 = src + static_cast<size_t>(y0) * static_cast<size_t>(src_w) * 3ULL;
        const uint8_t* row1 = src + static_cast<size_t>(y1) * static_cast<size_t>(src_w) * 3ULL;
        uint8_t* out_row = dst.data() + static_cast<size_t>(y) * static_cast<size_t>(dst_w) * 3ULL;

        for (int x = 0; x < dst_w; ++x) {
            const int x0 = x0_lut[static_cast<size_t>(x)];
            const int x1 = x1_lut[static_cast<size_t>(x)];
            const float wx1 = wx1_lut[static_cast<size_t>(x)];
            const float wx0 = 1.0f - wx1;

            const float w00 = wx0 * wy0;
            const float w10 = wx1 * wy0;
            const float w01 = wx0 * wy1;
            const float w11 = wx1 * wy1;

            const uint8_t* p00 = row0 + static_cast<size_t>(x0) * 3ULL;
            const uint8_t* p10 = row0 + static_cast<size_t>(x1) * 3ULL;
            const uint8_t* p01 = row1 + static_cast<size_t>(x0) * 3ULL;
            const uint8_t* p11 = row1 + static_cast<size_t>(x1) * 3ULL;

            uint8_t* out_px = out_row + static_cast<size_t>(x) * 3ULL;
            for (int c = 0; c < 3; ++c) {
                const float val = w00 * static_cast<float>(p00[c]) +
                                  w10 * static_cast<float>(p10[c]) +
                                  w01 * static_cast<float>(p01[c]) +
                                  w11 * static_cast<float>(p11[c]);
                out_px[c] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(val)), 0, 255));
            }
        }
    }

    return dst;
}

}  // namespace mangaflow::util
