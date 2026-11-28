#include "math_ocr_ops.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

namespace mangaflow::pipeline::math_ocr {

namespace {

inline uint8_t rgbToLuma(uint8_t r, uint8_t g, uint8_t b) noexcept {
    return static_cast<uint8_t>((77 * static_cast<int>(r) +
                                 150 * static_cast<int>(g) +
                                 29 * static_cast<int>(b)) >> 8);
}

inline uint8_t medianHist256(const std::array<uint32_t, 256>& hist, uint32_t total) noexcept {
    if (total == 0) return 255;
    const uint32_t half = total / 2;
    uint32_t acc = 0;
    for (int i = 0; i < 256; ++i) {
        acc += hist[static_cast<size_t>(i)];
        if (acc > half) return static_cast<uint8_t>(i);
    }
    return 255;
}

void boxMeanFloatIntegral(
    const float* src,
    int w,
    int h,
    int r,
    std::vector<double>& sat,
    std::vector<float>& dst) {
    const int sw = w + 1;
    const int sh = h + 1;
    sat.assign(static_cast<size_t>(sw) * static_cast<size_t>(sh), 0.0);
    for (int y = 0; y < h; ++y) {
        const size_t srow = static_cast<size_t>(y) * static_cast<size_t>(w);
        const size_t prev = static_cast<size_t>(y) * static_cast<size_t>(sw);
        const size_t cur = static_cast<size_t>(y + 1) * static_cast<size_t>(sw);
        double row_sum = 0.0;
        for (int x = 0; x < w; ++x) {
            row_sum += static_cast<double>(src[srow + static_cast<size_t>(x)]);
            sat[cur + static_cast<size_t>(x + 1)] = sat[prev + static_cast<size_t>(x + 1)] + row_sum;
        }
    }

    dst.resize(static_cast<size_t>(w) * static_cast<size_t>(h));
    for (int y = 0; y < h; ++y) {
        const int y0 = std::max(0, y - r);
        const int y1 = std::min(h - 1, y + r);
        const size_t r0 = static_cast<size_t>(y0) * static_cast<size_t>(sw);
        const size_t r1 = static_cast<size_t>(y1 + 1) * static_cast<size_t>(sw);
        const size_t drow = static_cast<size_t>(y) * static_cast<size_t>(w);
        const double dy = static_cast<double>(y1 - y0 + 1);
        for (int x = 0; x < w; ++x) {
            const int x0 = std::max(0, x - r);
            const int x1 = std::min(w - 1, x + r);
            const double sum =
                sat[r1 + static_cast<size_t>(x1 + 1)] -
                sat[r0 + static_cast<size_t>(x1 + 1)] -
                sat[r1 + static_cast<size_t>(x0)] +
                sat[r0 + static_cast<size_t>(x0)];
            dst[drow + static_cast<size_t>(x)] =
                static_cast<float>(sum / (dy * static_cast<double>(x1 - x0 + 1)));
        }
    }
}

bool pointInPolyI(int px, int py, const std::vector<std::array<int, 2>>& poly) noexcept {
    const size_t n = poly.size();
    if (n < 3) return true;
    bool inside = false;
    const float fx = static_cast<float>(px) + 0.5f;
    const float fy = static_cast<float>(py) + 0.5f;
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const float xi = static_cast<float>(poly[i][0]);
        const float yi = static_cast<float>(poly[i][1]);
        const float xj = static_cast<float>(poly[j][0]);
        const float yj = static_cast<float>(poly[j][1]);
        const bool inter = ((yi > fy) != (yj > fy)) &&
                           (fx < (xj - xi) * (fy - yi) / (yj - yi + 1e-12f) + xi);
        if (inter) inside = !inside;
    }
    return inside;
}

inline float srgbToLinear(uint8_t u) noexcept {
    const float c = static_cast<float>(u) / 255.0f;
    return (c <= 0.04045f) ? (c / 12.92f) : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

inline uint8_t linearToSrgb(float c) noexcept {
    const float clamped = std::clamp(c, 0.0f, 1.0f);
    const float s = (clamped <= 0.0031308f)
        ? (12.92f * clamped)
        : (1.055f * std::pow(clamped, 1.0f / 2.4f) - 0.055f);
    return static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(s * 255.0f)), 0, 255));
}

std::array<uint8_t, 3> oklabToSrgb(const OklabColor& c) noexcept {
    const float l_ = c.L + 0.3963377774f * c.a + 0.2158037573f * c.b;
    const float m_ = c.L - 0.1055613458f * c.a - 0.0638541728f * c.b;
    const float s_ = c.L - 0.0894841775f * c.a - 1.2914855480f * c.b;

    const float l = l_ * l_ * l_;
    const float m = m_ * m_ * m_;
    const float s = s_ * s_ * s_;

    const float r = +4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s;
    const float g = -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s;
    const float b = -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s;
    return {linearToSrgb(r), linearToSrgb(g), linearToSrgb(b)};
}

enum class ScriptKind {
    Other,
    Katakana,
    Hiragana,
    Kanji,
    Prolonged,
};

uint32_t firstUtf8Codepoint(std::string_view s) noexcept {
    if (s.empty()) return 0;
    const unsigned char b0 = static_cast<unsigned char>(s[0]);
    if (b0 < 0x80) return b0;
    if ((b0 & 0xE0) == 0xC0 && s.size() >= 2) {
        return ((b0 & 0x1Fu) << 6) | (static_cast<unsigned char>(s[1]) & 0x3Fu);
    }
    if ((b0 & 0xF0) == 0xE0 && s.size() >= 3) {
        return ((b0 & 0x0Fu) << 12) |
               ((static_cast<unsigned char>(s[1]) & 0x3Fu) << 6) |
               (static_cast<unsigned char>(s[2]) & 0x3Fu);
    }
    if ((b0 & 0xF8) == 0xF0 && s.size() >= 4) {
        return ((b0 & 0x07u) << 18) |
               ((static_cast<unsigned char>(s[1]) & 0x3Fu) << 12) |
               ((static_cast<unsigned char>(s[2]) & 0x3Fu) << 6) |
               (static_cast<unsigned char>(s[3]) & 0x3Fu);
    }
    return 0;
}

ScriptKind classifyCp(uint32_t cp) noexcept {
    if (cp == 0x30FC) return ScriptKind::Prolonged;               // ー
    if (cp >= 0x30A1 && cp <= 0x30FA) return ScriptKind::Katakana;
    if (cp >= 0x3041 && cp <= 0x309F) return ScriptKind::Hiragana;
    if ((cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF)) return ScriptKind::Kanji;
    return ScriptKind::Other;
}

}  // namespace

// =============================================================================
// #3 — O(1) Integral-Image Sauvola + 2D Structure Tensor + 1:1 Square Padding
// =============================================================================

void computeSauvolaMask(
    const uint8_t* gray,
    int width,
    int height,
    std::vector<uint8_t>& out_mask,
    int window,
    float k,
    float R,
    bool light_on_dark) {
    if (!gray || width <= 0 || height <= 0) {
        out_mask.clear();
        return;
    }

    const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);
    const int r = std::max(1, window / 2);
    const int sw = width + 1;
    const int sh = height + 1;

    std::vector<double> sat(static_cast<size_t>(sw) * static_cast<size_t>(sh), 0.0);
    std::vector<double> sat2(static_cast<size_t>(sw) * static_cast<size_t>(sh), 0.0);

    for (int y = 0; y < height; ++y) {
        const size_t grow = static_cast<size_t>(y) * static_cast<size_t>(width);
        const size_t prev = static_cast<size_t>(y) * static_cast<size_t>(sw);
        const size_t cur = static_cast<size_t>(y + 1) * static_cast<size_t>(sw);
        double rs = 0.0;
        double rs2 = 0.0;
        for (int x = 0; x < width; ++x) {
            const int raw = static_cast<int>(gray[grow + static_cast<size_t>(x)]);
            const double val = static_cast<double>(light_on_dark ? (255 - raw) : raw);
            rs += val;
            rs2 += val * val;
            sat[cur + static_cast<size_t>(x + 1)] = sat[prev + static_cast<size_t>(x + 1)] + rs;
            sat2[cur + static_cast<size_t>(x + 1)] = sat2[prev + static_cast<size_t>(x + 1)] + rs2;
        }
    }

    out_mask.assign(total, 0);
    const double inv_R = 1.0 / std::max(1.0, static_cast<double>(R));
    const double kd = static_cast<double>(k);

    for (int y = 0; y < height; ++y) {
        const int y0 = std::max(0, y - r);
        const int y1 = std::min(height - 1, y + r);
        const size_t r0 = static_cast<size_t>(y0) * static_cast<size_t>(sw);
        const size_t r1 = static_cast<size_t>(y1 + 1) * static_cast<size_t>(sw);
        const size_t grow = static_cast<size_t>(y) * static_cast<size_t>(width);
        const double dy = static_cast<double>(y1 - y0 + 1);

        for (int x = 0; x < width; ++x) {
            const int x0 = std::max(0, x - r);
            const int x1 = std::min(width - 1, x + r);
            const double area = dy * static_cast<double>(x1 - x0 + 1);
            const double s1 =
                sat[r1 + static_cast<size_t>(x1 + 1)] - sat[r0 + static_cast<size_t>(x1 + 1)] -
                sat[r1 + static_cast<size_t>(x0)] + sat[r0 + static_cast<size_t>(x0)];
            const double s2 =
                sat2[r1 + static_cast<size_t>(x1 + 1)] - sat2[r0 + static_cast<size_t>(x1 + 1)] -
                sat2[r1 + static_cast<size_t>(x0)] + sat2[r0 + static_cast<size_t>(x0)];

            const double mu = s1 / area;
            const double var = std::max(0.0, (s2 / area) - mu * mu);
            const double sigma = std::sqrt(var);
            const double thresh = mu * (1.0 + kd * (sigma * inv_R - 1.0));

            const int raw = static_cast<int>(gray[grow + static_cast<size_t>(x)]);
            const double val = static_cast<double>(light_on_dark ? (255 - raw) : raw);
            if (val < thresh && val < 230.0 && (mu - val) >= 12.0) {
                out_mask[grow + static_cast<size_t>(x)] = 1;
            }
        }
    }
}

void computeStructureTensorCoherence(
    const uint8_t* gray,
    int width,
    int height,
    std::vector<float>& out_coherence) {
    if (!gray || width <= 0 || height <= 0) {
        out_coherence.clear();
        return;
    }

    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<float> jxx(n, 0.0f);
    std::vector<float> jyy(n, 0.0f);
    std::vector<float> jxy(n, 0.0f);

    for (int y = 0; y < height; ++y) {
        const int ym = std::max(0, y - 1);
        const int yp = std::min(height - 1, y + 1);
        const size_t rm = static_cast<size_t>(ym) * static_cast<size_t>(width);
        const size_t rc = static_cast<size_t>(y) * static_cast<size_t>(width);
        const size_t rp = static_cast<size_t>(yp) * static_cast<size_t>(width);
        for (int x = 0; x < width; ++x) {
            const int xm = std::max(0, x - 1);
            const int xp = std::min(width - 1, x + 1);
            const float gx = 0.5f * (static_cast<float>(gray[rc + static_cast<size_t>(xp)]) -
                                     static_cast<float>(gray[rc + static_cast<size_t>(xm)]));
            const float gy = 0.5f * (static_cast<float>(gray[rp + static_cast<size_t>(x)]) -
                                     static_cast<float>(gray[rm + static_cast<size_t>(x)]));
            const size_t idx = rc + static_cast<size_t>(x);
            jxx[idx] = gx * gx;
            jyy[idx] = gy * gy;
            jxy[idx] = gx * gy;
        }
    }

    std::vector<double> sat;
    std::vector<float> sxx, syy, sxy;
    boxMeanFloatIntegral(jxx.data(), width, height, /*r=*/2, sat, sxx);
    boxMeanFloatIntegral(jyy.data(), width, height, /*r=*/2, sat, syy);
    boxMeanFloatIntegral(jxy.data(), width, height, /*r=*/2, sat, sxy);

    out_coherence.resize(n);
    constexpr float kEps = 1e-3f;
    for (size_t i = 0; i < n; ++i) {
        const float tr = sxx[i] + syy[i];
        const float diff = sxx[i] - syy[i];
        const float disc = std::sqrt(diff * diff + 4.0f * sxy[i] * sxy[i]);
        // (l1 - l2) / (l1 + l2 + eps) == disc / (tr + eps)
        const float ratio = disc / (tr + kEps);
        out_coherence[i] = std::clamp(ratio * ratio, 0.0f, 1.0f);
    }
}

void filterHalftoneDotsByStructureTensor(
    const uint8_t* gray,
    int width,
    int height,
    std::vector<uint8_t>& ink_mask,
    float min_coherence) {
    if (!gray || width < 6 || height < 6) return;
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (ink_mask.size() < n) return;

    std::vector<float> coherence;
    computeStructureTensorCoherence(gray, width, height, coherence);

    static constexpr int kDx8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static constexpr int kDy8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    std::vector<uint8_t> visited(n, 0);
    std::vector<int> queue;
    queue.reserve(128);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int start = y * width + x;
            if (ink_mask[static_cast<size_t>(start)] == 0 || visited[static_cast<size_t>(start)] != 0) {
                continue;
            }
            queue.clear();
            queue.push_back(start);
            visited[static_cast<size_t>(start)] = 1;

            int min_x = x, max_x = x, min_y = y, max_y = y;
            float sum_c = 0.0f;
            size_t head = 0;
            while (head < queue.size()) {
                const int cur = queue[head++];
                const int cx = cur % width;
                const int cy = cur / width;
                min_x = std::min(min_x, cx);
                max_x = std::max(max_x, cx);
                min_y = std::min(min_y, cy);
                max_y = std::max(max_y, cy);
                sum_c += coherence[static_cast<size_t>(cur)];

                for (int k = 0; k < 8; ++k) {
                    const int nx = cx + kDx8[k];
                    const int ny = cy + kDy8[k];
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                    const int nidx = ny * width + nx;
                    if (ink_mask[static_cast<size_t>(nidx)] != 0 && visited[static_cast<size_t>(nidx)] == 0) {
                        visited[static_cast<size_t>(nidx)] = 1;
                        queue.push_back(nidx);
                    }
                }
            }

            const int area = static_cast<int>(queue.size());
            const int cw = max_x - min_x + 1;
            const int ch = max_y - min_y + 1;
            const float mean_c = sum_c / static_cast<float>(std::max(1, area));
            const float aspect = static_cast<float>(std::max(cw, ch)) / static_cast<float>(std::max(1, std::min(cw, ch)));

            // Reject small isotropic screentone dots (circular, low tensor coherence).
            if (area <= 42 && aspect <= 1.65f && mean_c < min_coherence) {
                for (int idx : queue) {
                    ink_mask[static_cast<size_t>(idx)] = 0;
                }
            }
        }
    }
}

SquarePaddedRgb trimAndSquarePadRgb(
    const uint8_t* rgb,
    int width,
    int height,
    const std::vector<std::array<int, 2>>* local_polygon,
    int pad_margin) {
    SquarePaddedRgb out;
    if (!rgb || width <= 0 || height <= 0) {
        out.width = 32;
        out.height = 32;
        out.data.assign(32ULL * 32ULL * 3ULL, 255);
        return out;
    }

    const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<uint8_t> work(rgb, rgb + total * 3ULL);

    // Estimate median background RGB from outer 10% border pixels.
    std::array<uint32_t, 256> hr{}, hg{}, hb{}, hl{};
    uint32_t bcnt = 0;
    const int bx = std::max(1, width / 10);
    const int by = std::max(1, height / 10);
    for (int y = 0; y < height; ++y) {
        const bool ye = (y < by || y >= height - by);
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(width);
        for (int x = 0; x < width; ++x) {
            if (!ye && x >= bx && x < width - bx) continue;
            const uint8_t* px = work.data() + (row + static_cast<size_t>(x)) * 3ULL;
            ++hr[px[0]];
            ++hg[px[1]];
            ++hb[px[2]];
            ++hl[rgbToLuma(px[0], px[1], px[2])];
            ++bcnt;
        }
    }
    const uint8_t bg_r = medianHist256(hr, bcnt);
    const uint8_t bg_g = medianHist256(hg, bcnt);
    const uint8_t bg_b = medianHist256(hb, bcnt);
    const int bg_luma = static_cast<int>(medianHist256(hl, bcnt));
    const bool dark_bg = (bg_luma < 110);

    // Blank pixels strictly outside optional bubble polygon (leaving a 1px guard).
    if (local_polygon != nullptr && local_polygon->size() >= 4) {
        for (int y = 0; y < height; ++y) {
            const size_t row = static_cast<size_t>(y) * static_cast<size_t>(width);
            for (int x = 0; x < width; ++x) {
                if (!pointInPolyI(x, y, *local_polygon)) {
                    uint8_t* px = work.data() + (row + static_cast<size_t>(x)) * 3ULL;
                    px[0] = bg_r;
                    px[1] = bg_g;
                    px[2] = bg_b;
                }
            }
        }
    }

    // 1D ink projection to trim excess whitespace symmetrically without clipping glyphs.
    std::vector<int> row_proj(static_cast<size_t>(height), 0);
    std::vector<int> col_proj(static_cast<size_t>(width), 0);
    for (int y = 0; y < height; ++y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(width);
        for (int x = 0; x < width; ++x) {
            const uint8_t* px = work.data() + (row + static_cast<size_t>(x)) * 3ULL;
            const int lum = static_cast<int>(rgbToLuma(px[0], px[1], px[2]));
            const bool is_ink = dark_bg ? (lum > bg_luma + 30) : (lum < bg_luma - 30);
            if (is_ink) {
                ++row_proj[static_cast<size_t>(y)];
                ++col_proj[static_cast<size_t>(x)];
            }
        }
    }

    int tx1 = 0, tx2 = width, ty1 = 0, ty2 = height;
    int first_x = -1, last_x = -1;
    for (int x = 0; x < width; ++x) {
        if (col_proj[static_cast<size_t>(x)] >= 1) {
            if (first_x < 0) first_x = x;
            last_x = x;
        }
    }
    int first_y = -1, last_y = -1;
    for (int y = 0; y < height; ++y) {
        if (row_proj[static_cast<size_t>(y)] >= 1) {
            if (first_y < 0) first_y = y;
            last_y = y;
        }
    }

    if (first_x >= 0 && last_x >= first_x && first_y >= 0 && last_y >= first_y) {
        const int inner_pad = 4;
        tx1 = std::max(0, first_x - inner_pad);
        tx2 = std::min(width, last_x + 1 + inner_pad);
        ty1 = std::max(0, first_y - inner_pad);
        ty2 = std::min(height, last_y + 1 + inner_pad);
    }

    const int tw = std::max(1, tx2 - tx1);
    const int th = std::max(1, ty2 - ty1);

    // Pad symmetrically to a 1:1 square: side = max(tw, th) + pad_margin.
    const int side = std::max(tw, th) + std::max(8, pad_margin);
    const int ox = (side - tw) / 2;
    const int oy = (side - th) / 2;

    out.width = side;
    out.height = side;
    out.data.resize(static_cast<size_t>(side) * static_cast<size_t>(side) * 3ULL);
    for (size_t i = 0; i < static_cast<size_t>(side) * static_cast<size_t>(side); ++i) {
        out.data[i * 3ULL + 0] = bg_r;
        out.data[i * 3ULL + 1] = bg_g;
        out.data[i * 3ULL + 2] = bg_b;
    }

    const size_t copy_bytes = static_cast<size_t>(tw) * 3ULL;
    for (int y = 0; y < th; ++y) {
        const size_t src_off = (static_cast<size_t>(ty1 + y) * static_cast<size_t>(width) +
                                static_cast<size_t>(tx1)) * 3ULL;
        const size_t dst_off = (static_cast<size_t>(oy + y) * static_cast<size_t>(side) +
                                static_cast<size_t>(ox)) * 3ULL;
        std::memcpy(out.data.data() + dst_off, work.data() + src_off, copy_bytes);
    }

    return out;
}

// =============================================================================
// #9 — Perceptual OKLab + 2-3 Component EM GMM
// =============================================================================

OklabColor srgbToOklab(uint8_t r, uint8_t g, uint8_t b) noexcept {
    const float lr = srgbToLinear(r);
    const float lg = srgbToLinear(g);
    const float lb = srgbToLinear(b);

    const float l = 0.4122214708f * lr + 0.5363325363f * lg + 0.0514459929f * lb;
    const float m = 0.2119034982f * lr + 0.6806995451f * lg + 0.1073969566f * lb;
    const float s = 0.0883024619f * lr + 0.2817188376f * lg + 0.6299787005f * lb;

    const float l_ = std::cbrt(std::max(0.0f, l));
    const float m_ = std::cbrt(std::max(0.0f, m));
    const float s_ = std::cbrt(std::max(0.0f, s));

    return OklabColor{
        0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_,
        1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_,
        0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_
    };
}

float oklabDeltaE(const OklabColor& c1, const OklabColor& c2) noexcept {
    const float dL = c1.L - c2.L;
    const float da = c1.a - c2.a;
    const float db = c1.b - c2.b;
    return std::sqrt(dL * dL + da * da + db * db);
}

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
    int core_y2) {
    OklabGmmResult res;
    if (!page_rgb || page_w <= 0 || page_h <= 0) return res;

    roi_x1 = std::clamp(roi_x1, 0, page_w);
    roi_y1 = std::clamp(roi_y1, 0, page_h);
    roi_x2 = std::clamp(roi_x2, roi_x1, page_w);
    roi_y2 = std::clamp(roi_y2, roi_y1, page_h);
    const int rw = roi_x2 - roi_x1;
    const int rh = roi_y2 - roi_y1;
    if (rw < 8 || rh < 8) return res;

    // 1. Subsample up to ~1024 pixels in OKLab and check chroma variance.
    const int total_px = rw * rh;
    const int step = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(total_px) / 1024.0)));

    struct Sample {
        OklabColor lab;
        bool in_core = false;
    };
    std::vector<Sample> samples;
    samples.reserve(1200);

    double sum_c = 0.0;
    double sum_c2 = 0.0;
    float max_c = 0.0f;

    for (int ly = 0; ly < rh; ly += step) {
        const int gy = roi_y1 + ly;
        const size_t rgb_row = static_cast<size_t>(gy) * static_cast<size_t>(page_w) * 3ULL;
        const bool y_in = (gy >= core_y1 && gy < core_y2);
        for (int lx = 0; lx < rw; lx += step) {
            const int gx = roi_x1 + lx;
            const uint8_t* px = page_rgb + rgb_row + static_cast<size_t>(gx) * 3ULL;
            const OklabColor lab = srgbToOklab(px[0], px[1], px[2]);
            const float chroma = std::hypot(lab.a, lab.b);
            sum_c += static_cast<double>(chroma);
            sum_c2 += static_cast<double>(chroma) * static_cast<double>(chroma);
            max_c = std::max(max_c, chroma);
            samples.push_back(Sample{lab, y_in && (gx >= core_x1 && gx < core_x2)});
        }
    }

    if (samples.size() < 24) return res;
    const double mean_c = sum_c / static_cast<double>(samples.size());
    const double var_c = std::max(0.0, (sum_c2 / static_cast<double>(samples.size())) - mean_c * mean_c);
    const double std_c = std::sqrt(var_c);

    // Fast exit for standard grayscale / B&W manga bubbles.
    if (max_c < 0.075f || std_c < 0.022) {
        return res;
    }

    // 2. Initialize 3-component diagonal GMM in OKLab:
    //    k=0: outer-ring (background) mean, k=1: farthest point (text), k=2: farthest from {0,1} (outline).
    constexpr int K = 3;
    std::array<OklabColor, K> mu{};
    std::array<OklabColor, K> var{};
    std::array<float, K> pi{0.55f, 0.30f, 0.15f};

    // Component 0: mean of non-core (outer ring) samples
    double bgL = 0.0, bga = 0.0, bgb = 0.0;
    int bg_n = 0;
    for (const auto& s : samples) {
        if (!s.in_core) {
            bgL += s.lab.L;
            bga += s.lab.a;
            bgb += s.lab.b;
            ++bg_n;
        }
    }
    if (bg_n > 0) {
        mu[0] = OklabColor{static_cast<float>(bgL / bg_n), static_cast<float>(bga / bg_n), static_cast<float>(bgb / bg_n)};
    } else {
        mu[0] = samples.front().lab;
    }

    // Component 1: core sample farthest from mu[0]
    float best_d1 = -1.0f;
    mu[1] = mu[0];
    for (const auto& s : samples) {
        if (!s.in_core) continue;
        const float d = oklabDeltaE(s.lab, mu[0]);
        if (d > best_d1) {
            best_d1 = d;
            mu[1] = s.lab;
        }
    }
    if (best_d1 < 0.08f) {
        return res;  // Not enough color separation
    }

    // Component 2: sample farthest from min(dist(mu[0]), dist(mu[1]))
    float best_d2 = -1.0f;
    mu[2] = OklabColor{0.5f * (mu[0].L + mu[1].L), 0.5f * (mu[0].a + mu[1].a), 0.5f * (mu[0].b + mu[1].b)};
    for (const auto& s : samples) {
        const float d = std::min(oklabDeltaE(s.lab, mu[0]), oklabDeltaE(s.lab, mu[1]));
        if (d > best_d2) {
            best_d2 = d;
            mu[2] = s.lab;
        }
    }

    const int active_k = (best_d2 >= 0.07f) ? 3 : 2;
    if (active_k == 2) {
        pi = {0.65f, 0.35f, 0.0f};
    }
    for (int k = 0; k < active_k; ++k) {
        var[static_cast<size_t>(k)] = OklabColor{0.02f, 0.01f, 0.01f};
    }

    // 3. 6-iteration diagonal EM GMM in OKLab
    const size_t ns = samples.size();
    std::vector<std::array<float, K>> resp(ns);
    constexpr float kVarFloor = 2e-4f;

    for (int iter = 0; iter < 6; ++iter) {
        // E-step
        for (size_t i = 0; i < ns; ++i) {
            const OklabColor& x = samples[i].lab;
            float max_lp = -1e30f;
            std::array<float, K> lp{};
            for (int k = 0; k < active_k; ++k) {
                const auto& m = mu[static_cast<size_t>(k)];
                const auto& v = var[static_cast<size_t>(k)];
                const float dL = x.L - m.L;
                const float da = x.a - m.a;
                const float db = x.b - m.b;
                const float log_det = std::log(v.L * v.a * v.b);
                const float mahal = (dL * dL) / v.L + (da * da) / v.a + (db * db) / v.b;
                lp[static_cast<size_t>(k)] = std::log(std::max(1e-4f, pi[static_cast<size_t>(k)])) -
                                             0.5f * (log_det + mahal);
                max_lp = std::max(max_lp, lp[static_cast<size_t>(k)]);
            }
            float sum_exp = 0.0f;
            for (int k = 0; k < active_k; ++k) {
                resp[i][static_cast<size_t>(k)] = std::exp(lp[static_cast<size_t>(k)] - max_lp);
                sum_exp += resp[i][static_cast<size_t>(k)];
            }
            const float inv = (sum_exp > 1e-12f) ? (1.0f / sum_exp) : (1.0f / static_cast<float>(active_k));
            for (int k = 0; k < active_k; ++k) {
                resp[i][static_cast<size_t>(k)] *= inv;
            }
        }

        // M-step
        for (int k = 0; k < active_k; ++k) {
            double nk = 0.0, sL = 0.0, sa = 0.0, sb = 0.0;
            for (size_t i = 0; i < ns; ++i) {
                const double r = static_cast<double>(resp[i][static_cast<size_t>(k)]);
                nk += r;
                sL += r * samples[i].lab.L;
                sa += r * samples[i].lab.a;
                sb += r * samples[i].lab.b;
            }
            if (nk < 1.0) continue;
            pi[static_cast<size_t>(k)] = static_cast<float>(nk / static_cast<double>(ns));
            mu[static_cast<size_t>(k)] = OklabColor{
                static_cast<float>(sL / nk),
                static_cast<float>(sa / nk),
                static_cast<float>(sb / nk)
            };
            double vL = 0.0, va = 0.0, vb = 0.0;
            const auto& m = mu[static_cast<size_t>(k)];
            for (size_t i = 0; i < ns; ++i) {
                const double r = static_cast<double>(resp[i][static_cast<size_t>(k)]);
                const double dL = samples[i].lab.L - m.L;
                const double da = samples[i].lab.a - m.a;
                const double db = samples[i].lab.b - m.b;
                vL += r * dL * dL;
                va += r * da * da;
                vb += r * db * db;
            }
            var[static_cast<size_t>(k)] = OklabColor{
                std::max(kVarFloor, static_cast<float>(vL / nk)),
                std::max(kVarFloor, static_cast<float>(va / nk)),
                std::max(kVarFloor, static_cast<float>(vb / nk))
            };
        }
    }

    // 4. Identify background cluster (highest outer-ring responsibility) and text cluster
    //    (highest core responsibility with largest DeltaE_OK from bg).
    std::array<float, K> outer_mass{};
    std::array<float, K> core_mass{};
    for (size_t i = 0; i < ns; ++i) {
        for (int k = 0; k < active_k; ++k) {
            if (samples[i].in_core) core_mass[static_cast<size_t>(k)] += resp[i][static_cast<size_t>(k)];
            else outer_mass[static_cast<size_t>(k)] += resp[i][static_cast<size_t>(k)];
        }
    }

    int bg_k = 0;
    for (int k = 1; k < active_k; ++k) {
        if (outer_mass[static_cast<size_t>(k)] > outer_mass[static_cast<size_t>(bg_k)]) {
            bg_k = k;
        }
    }

    int text_k = -1;
    float best_text_score = -1.0f;
    for (int k = 0; k < active_k; ++k) {
        if (k == bg_k) continue;
        const float de = oklabDeltaE(mu[static_cast<size_t>(k)], mu[static_cast<size_t>(bg_k)]);
        const float score = de * std::sqrt(std::max(1.0f, core_mass[static_cast<size_t>(k)]));
        if (score > best_text_score) {
            best_text_score = score;
            text_k = k;
        }
    }
    if (text_k < 0 || oklabDeltaE(mu[static_cast<size_t>(text_k)], mu[static_cast<size_t>(bg_k)]) < 0.08f) {
        return res;
    }

    int outline_k = -1;
    if (active_k == 3) {
        for (int k = 0; k < 3; ++k) {
            if (k != bg_k && k != text_k) {
                const float de_bg = oklabDeltaE(mu[static_cast<size_t>(k)], mu[static_cast<size_t>(bg_k)]);
                const float de_tx = oklabDeltaE(mu[static_cast<size_t>(k)], mu[static_cast<size_t>(text_k)]);
                if (de_bg >= 0.08f && de_tx >= 0.08f && pi[static_cast<size_t>(k)] >= 0.03f) {
                    outline_k = k;
                }
            }
        }
    }

    // 5. Assign full ROI pixels to MAP OKLab cluster (inside core_box for ink_mask).
    res.ink_mask.assign(static_cast<size_t>(rw) * static_cast<size_t>(rh), 0);
    int ink_cnt = 0;
    for (int ly = 0; ly < rh; ++ly) {
        const int gy = roi_y1 + ly;
        if (gy < core_y1 || gy >= core_y2) continue;
        const size_t rgb_row = static_cast<size_t>(gy) * static_cast<size_t>(page_w) * 3ULL;
        const size_t mrow = static_cast<size_t>(ly) * static_cast<size_t>(rw);
        for (int lx = 0; lx < rw; ++lx) {
            const int gx = roi_x1 + lx;
            if (gx < core_x1 || gx >= core_x2) continue;
            const uint8_t* px = page_rgb + rgb_row + static_cast<size_t>(gx) * 3ULL;
            const OklabColor lab = srgbToOklab(px[0], px[1], px[2]);

            int best_c = bg_k;
            float best_d = oklabDeltaE(lab, mu[static_cast<size_t>(bg_k)]);
            for (int k = 0; k < active_k; ++k) {
                if (k == bg_k) continue;
                const float d = oklabDeltaE(lab, mu[static_cast<size_t>(k)]);
                if (d < best_d) {
                    best_d = d;
                    best_c = k;
                }
            }
            if (best_c == text_k) {
                res.ink_mask[mrow + static_cast<size_t>(lx)] = 1;
                ++ink_cnt;
            }
        }
    }

    if (ink_cnt < 8) {
        return res;
    }

    res.used_gmm = true;
    res.bg_rgb = oklabToSrgb(mu[static_cast<size_t>(bg_k)]);
    res.text_rgb = oklabToSrgb(mu[static_cast<size_t>(text_k)]);
    if (outline_k >= 0) {
        res.has_outline_cluster = true;
        res.outline_rgb = oklabToSrgb(mu[static_cast<size_t>(outline_k)]);
    }
    return res;
}

// =============================================================================
// #10 — Perona-Malik Anisotropic Diffusion Denoiser
// =============================================================================

bool denoiseCropPeronaMalikIfNoisy(
    uint8_t* rgb,
    int width,
    int height,
    float noise_thresh,
    float K,
    int iters) {
    if (!rgb || width < 12 || height < 12 || iters <= 0) {
        return false;
    }

    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<uint8_t> luma(n);
    std::array<uint32_t, 256> hist{};
    for (size_t i = 0; i < n; ++i) {
        const uint8_t y = rgbToLuma(rgb[i * 3ULL], rgb[i * 3ULL + 1], rgb[i * 3ULL + 2]);
        luma[i] = y;
        ++hist[y];
    }
    const int bg_med = static_cast<int>(medianHist256(hist, static_cast<uint32_t>(n)));

    // Estimate background noise stddev using flat background pixels within +-35 of median luma
    // (2D discrete Laplacian residual ignores smooth gradients and measures JPEG ringing/grain).
    double sum_sq = 0.0;
    int bg_cnt = 0;
    for (int y = 1; y + 1 < height; ++y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(width);
        for (int x = 1; x + 1 < width; ++x) {
            const size_t idx = row + static_cast<size_t>(x);
            const int c = static_cast<int>(luma[idx]);
            if (std::abs(c - bg_med) > 35) continue;
            const int n_v = static_cast<int>(luma[idx - static_cast<size_t>(width)]);
            const int s_v = static_cast<int>(luma[idx + static_cast<size_t>(width)]);
            const int w_v = static_cast<int>(luma[idx - 1]);
            const int e_v = static_cast<int>(luma[idx + 1]);
            if (std::abs(n_v - bg_med) > 35 || std::abs(s_v - bg_med) > 35 ||
                std::abs(w_v - bg_med) > 35 || std::abs(e_v - bg_med) > 35) {
                continue;
            }
            const double lap = static_cast<double>(c) - 0.25 * static_cast<double>(n_v + s_v + w_v + e_v);
            sum_sq += lap * lap;
            ++bg_cnt;
        }
    }

    if (bg_cnt < 16) return false;
    // Laplacian residual std scales by sqrt(1.25) relative to white noise sigma.
    const float noise_std = static_cast<float>(std::sqrt(sum_sq / static_cast<double>(bg_cnt)) / 1.118034);
    if (noise_std < noise_thresh) {
        return false;  // Clean digital scan -> 0 ms diffusion overhead!
    }

    // 3-iteration 4-neighbor Perona-Malik PDE per RGB channel:
    //   I^{t+1} = I^t + lambda * sum_d g(|grad_d|) * grad_d, g(s) = 1 / (1 + (s/K)^2)
    const float inv_K2 = 1.0f / std::max(1.0f, K * K);
    constexpr float kLambda = 0.20f;

    std::vector<float> cur(n);
    std::vector<float> nxt(n);

    for (int ch = 0; ch < 3; ++ch) {
        for (size_t i = 0; i < n; ++i) {
            cur[i] = static_cast<float>(rgb[i * 3ULL + static_cast<size_t>(ch)]);
        }

        for (int it = 0; it < iters; ++it) {
            nxt = cur;
            for (int y = 1; y + 1 < height; ++y) {
                const size_t row = static_cast<size_t>(y) * static_cast<size_t>(width);
                for (int x = 1; x + 1 < width; ++x) {
                    const size_t idx = row + static_cast<size_t>(x);
                    const float c = cur[idx];
                    const float dn = cur[idx - static_cast<size_t>(width)] - c;
                    const float ds = cur[idx + static_cast<size_t>(width)] - c;
                    const float dw = cur[idx - 1] - c;
                    const float de = cur[idx + 1] - c;

                    const float gn = 1.0f / (1.0f + dn * dn * inv_K2);
                    const float gs = 1.0f / (1.0f + ds * ds * inv_K2);
                    const float gw = 1.0f / (1.0f + dw * dw * inv_K2);
                    const float ge = 1.0f / (1.0f + de * de * inv_K2);

                    nxt[idx] = c + kLambda * (gn * dn + gs * ds + gw * dw + ge * de);
                }
            }
            cur.swap(nxt);
        }

        for (size_t i = 0; i < n; ++i) {
            rgb[i * 3ULL + static_cast<size_t>(ch)] =
                static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(cur[i])), 0, 255));
        }
    }

    return true;
}

// =============================================================================
// #11 — Stroke Width Transform (SWT) Opt-In Furigana Masking
// =============================================================================

int maskFuriganaSwtInCrop(
    uint8_t* rgb,
    int width,
    int height) {
    if (!rgb || width < 24 || height < 32) return 0;

    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<uint8_t> gray(n);
    std::array<uint32_t, 256> hr{}, hg{}, hb{}, hl{};
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* px = rgb + i * 3ULL;
        const uint8_t y = rgbToLuma(px[0], px[1], px[2]);
        gray[i] = y;
        ++hl[y];
        ++hr[px[0]];
        ++hg[px[1]];
        ++hb[px[2]];
    }
    const int bg_luma = static_cast<int>(medianHist256(hl, static_cast<uint32_t>(n)));
    const bool light_on_dark = (bg_luma < 110);
    const uint8_t bg_r = medianHist256(hr, static_cast<uint32_t>(n));
    const uint8_t bg_g = medianHist256(hg, static_cast<uint32_t>(n));
    const uint8_t bg_b = medianHist256(hb, static_cast<uint32_t>(n));

    std::vector<uint8_t> ink;
    computeSauvolaMask(gray.data(), width, height, ink, /*window=*/15, /*k=*/0.2f, /*R=*/128.0f, light_on_dark);

    // 1. Fast ray-cast SWT on ink pixels along inward gradient direction.
    constexpr float kSwtInf = 1e6f;
    std::vector<float> swt(n, kSwtInf);
    constexpr int kMaxRay = 24;

    for (int y = 1; y + 1 < height; ++y) {
        const size_t rm = static_cast<size_t>(y - 1) * static_cast<size_t>(width);
        const size_t rc = static_cast<size_t>(y) * static_cast<size_t>(width);
        const size_t rp = static_cast<size_t>(y + 1) * static_cast<size_t>(width);
        for (int x = 1; x + 1 < width; ++x) {
            const size_t idx = rc + static_cast<size_t>(x);
            if (ink[idx] == 0) continue;
            // Check if boundary pixel of ink
            if (ink[idx - 1] && ink[idx + 1] && ink[rm + static_cast<size_t>(x)] && ink[rp + static_cast<size_t>(x)]) {
                continue;
            }
            const float gx = static_cast<float>(gray[rc + static_cast<size_t>(x + 1)]) -
                             static_cast<float>(gray[rc + static_cast<size_t>(x - 1)]);
            const float gy = static_cast<float>(gray[rp + static_cast<size_t>(x)]) -
                             static_cast<float>(gray[rm + static_cast<size_t>(x)]);
            const float mag = std::hypot(gx, gy);
            if (mag < 12.0f) continue;

            // Inward direction into ink: dark text -> negative gradient; light text -> positive gradient
            const float sign = light_on_dark ? 1.0f : -1.0f;
            const float dx = sign * gx / mag;
            const float dy = sign * gy / mag;

            std::vector<size_t> ray_pts;
            ray_pts.reserve(static_cast<size_t>(kMaxRay));
            ray_pts.push_back(idx);
            float ray_len = 0.0f;
            bool hit_opp = false;

            for (int step = 1; step <= kMaxRay; ++step) {
                const int nx = static_cast<int>(std::lround(static_cast<float>(x) + static_cast<float>(step) * dx));
                const int ny = static_cast<int>(std::lround(static_cast<float>(y) + static_cast<float>(step) * dy));
                if (nx <= 0 || nx + 1 >= width || ny <= 0 || ny + 1 >= height) break;
                const size_t nidx = static_cast<size_t>(ny) * static_cast<size_t>(width) + static_cast<size_t>(nx);
                if (ink[nidx] == 0) {
                    ray_len = static_cast<float>(step);
                    hit_opp = true;
                    break;
                }
                ray_pts.push_back(nidx);
            }
            if (hit_opp && ray_len >= 1.0f) {
                for (size_t p : ray_pts) {
                    if (ray_len < swt[p]) swt[p] = ray_len;
                }
            }
        }
    }

    // 2. Extract 8-connected ink components and compute median SWT width per component.
    struct Comp {
        std::vector<int> pixels;
        int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        float med_swt = 0.0f;
    };
    static constexpr int kDx8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static constexpr int kDy8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    std::vector<uint8_t> visited(n, 0);
    std::vector<Comp> comps;
    std::vector<int> q;
    q.reserve(128);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int start = y * width + x;
            if (ink[static_cast<size_t>(start)] == 0 || visited[static_cast<size_t>(start)] != 0) continue;
            q.clear();
            q.push_back(start);
            visited[static_cast<size_t>(start)] = 1;
            int min_x = x, max_x = x, min_y = y, max_y = y;
            size_t head = 0;
            while (head < q.size()) {
                const int cur = q[head++];
                const int cx = cur % width;
                const int cy = cur / width;
                min_x = std::min(min_x, cx);
                max_x = std::max(max_x, cx);
                min_y = std::min(min_y, cy);
                max_y = std::max(max_y, cy);
                for (int k = 0; k < 8; ++k) {
                    const int nx = cx + kDx8[k];
                    const int ny = cy + kDy8[k];
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                    const int nidx = ny * width + nx;
                    if (ink[static_cast<size_t>(nidx)] != 0 && visited[static_cast<size_t>(nidx)] == 0) {
                        visited[static_cast<size_t>(nidx)] = 1;
                        q.push_back(nidx);
                    }
                }
            }
            if (q.size() < 4) continue;
            std::vector<float> sws;
            sws.reserve(q.size());
            for (int p : q) {
                if (swt[static_cast<size_t>(p)] < kSwtInf * 0.5f) {
                    sws.push_back(swt[static_cast<size_t>(p)]);
                }
            }
            float med_sw = 2.0f;
            if (!sws.empty()) {
                std::nth_element(sws.begin(), sws.begin() + static_cast<ptrdiff_t>(sws.size() / 2), sws.end());
                med_sw = sws[sws.size() / 2];
            }
            comps.push_back(Comp{q, min_x, min_y, max_x + 1, max_y + 1, med_sw});
        }
    }

    if (comps.size() < 3) return 0;

    // 3. Estimate body stroke width w_body and body char height h_body (75th percentile).
    std::vector<float> body_ws;
    std::vector<float> body_hs;
    for (const auto& c : comps) {
        if (c.pixels.size() >= 16) {
            body_ws.push_back(c.med_swt);
            body_hs.push_back(static_cast<float>(c.y2 - c.y1));
        }
    }
    if (body_ws.size() < 2) return 0;
    std::sort(body_ws.begin(), body_ws.end());
    std::sort(body_hs.begin(), body_hs.end());
    const float w_body = body_ws[(body_ws.size() * 3) / 4];
    const float h_body = body_hs[(body_hs.size() * 3) / 4];
    if (w_body < 2.2f || h_body < 14.0f) return 0;

    // 4. Identify thin small right-adjacent ruby (furigana) components:
    //    w_furi <= 0.55 * w_body AND h_furi <= 0.45 * h_body, sitting immediately right of a body char.
    int masked_count = 0;
    std::vector<uint8_t> erase_mask(n, 0);

    for (size_t i = 0; i < comps.size(); ++i) {
        const auto& c = comps[i];
        const float cw = static_cast<float>(c.x2 - c.x1);
        const float ch = static_cast<float>(c.y2 - c.y1);
        if (c.med_swt > 0.55f * w_body) continue;
        if (ch > 0.45f * h_body || cw > 0.55f * h_body) continue;

        bool right_of_body = false;
        for (size_t j = 0; j < comps.size(); ++j) {
            if (i == j) continue;
            const auto& b = comps[j];
            const float bh = static_cast<float>(b.y2 - b.y1);
            if (bh < 0.65f * h_body) continue;
            // Ruby sits to the right of `b` (x1 >= b.x2 - 2, gap <= 0.65 * h_body) with vertical overlap
            const int v_ov = std::min(c.y2, b.y2) - std::max(c.y1, b.y1);
            if (v_ov >= 2 && c.x1 >= b.x2 - 2 && static_cast<float>(c.x1 - b.x2) <= 0.65f * h_body) {
                right_of_body = true;
                break;
            }
        }
        if (!right_of_body) continue;

        ++masked_count;
        for (int p : c.pixels) {
            const int px = p % width;
            const int py = p / width;
            for (int dy = -1; dy <= 1; ++dy) {
                const int ny = py + dy;
                if (ny < 0 || ny >= height) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = px + dx;
                    if (nx < 0 || nx >= width) continue;
                    erase_mask[static_cast<size_t>(ny) * static_cast<size_t>(width) + static_cast<size_t>(nx)] = 1;
                }
            }
        }
    }

    if (masked_count > 0) {
        for (size_t i = 0; i < n; ++i) {
            if (erase_mask[i]) {
                rgb[i * 3ULL + 0] = bg_r;
                rgb[i * 3ULL + 1] = bg_g;
                rgb[i * 3ULL + 2] = bg_b;
            }
        }
    }
    return masked_count;
}

// =============================================================================
// #17 — Compact Japanese Visual Confusable Markov Prior
// =============================================================================

float jaConfusableTransitionBonus(
    std::string_view prev_tok,
    std::string_view cand_tok) noexcept {
    const uint32_t cur = firstUtf8Codepoint(cand_tok);
    if (cur == 0) return 0.0f;
    const uint32_t prev = firstUtf8Codepoint(prev_tok);
    const ScriptKind pk = classifyCp(prev);

    // Pair 1: ー (0x30FC, Katakana prolonged mark) vs 一 (0x4E00, Kanji ichi)
    if (cur == 0x30FC) {
        if (pk == ScriptKind::Katakana || pk == ScriptKind::Hiragana || pk == ScriptKind::Prolonged) return +0.45f;
        if (pk == ScriptKind::Kanji) return -0.25f;
        return 0.0f;
    }
    if (cur == 0x4E00) {
        if (pk == ScriptKind::Katakana) return -0.45f;
        if (pk == ScriptKind::Kanji) return +0.30f;
        return 0.0f;
    }

    // Pair 2: ロ/カ/エ (Katakana: 0x30ED, 0x30AB, 0x30A8) vs 口/力/工 (Kanji: 0x53E3, 0x529B, 0x5DE5)
    if (cur == 0x30ED || cur == 0x30AB || cur == 0x30A8) {
        if (pk == ScriptKind::Katakana || pk == ScriptKind::Prolonged) return +0.45f;
        if (pk == ScriptKind::Kanji) return -0.35f;
        return 0.0f;
    }
    if (cur == 0x53E3 || cur == 0x529B || cur == 0x5DE5) {
        if (pk == ScriptKind::Katakana || pk == ScriptKind::Prolonged) return -0.45f;
        if (pk == ScriptKind::Kanji) return +0.35f;
        return 0.0f;
    }

    // Pair 3: ソ (0x30BD) / ン (0x30F3) and シ (0x30B7) / ツ (0x30C4)
    if (cur == 0x30F3) {
        // ン cannot start a Japanese Katakana word (prev is empty/Other/punct)
        if (pk == ScriptKind::Other) return -0.40f;
        if (pk == ScriptKind::Katakana) return +0.20f;
        return 0.0f;
    }
    if (cur == 0x30BD || cur == 0x30B7 || cur == 0x30C4) {
        if (pk == ScriptKind::Katakana || pk == ScriptKind::Prolonged) return +0.20f;
        return 0.0f;
    }

    return 0.0f;
}

float jaConfusableSequenceBonus(const std::vector<std::string_view>& tokens) noexcept {
    const size_t n = tokens.size();
    if (n == 0) return 0.0f;

    float bonus = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const std::string_view prev = (i > 0) ? tokens[i - 1] : std::string_view{};
        const std::string_view cur = tokens[i];
        const std::string_view next = (i + 1 < n) ? tokens[i + 1] : std::string_view{};

        bonus += jaConfusableTransitionBonus(prev, cur);

        // Right-hand lookahead for word-initial confusables (e.g. ロボット vs 口ボット)
        const uint32_t cp = firstUtf8Codepoint(cur);
        const ScriptKind nk = classifyCp(firstUtf8Codepoint(next));
        if (cp == 0x30ED || cp == 0x30AB || cp == 0x30A8) {
            if (nk == ScriptKind::Katakana || nk == ScriptKind::Prolonged) bonus += 0.35f;
            else if (nk == ScriptKind::Kanji) bonus -= 0.25f;
        } else if (cp == 0x53E3 || cp == 0x529B || cp == 0x5DE5) {
            if (nk == ScriptKind::Katakana || nk == ScriptKind::Prolonged) bonus -= 0.35f;
            else if (nk == ScriptKind::Kanji) bonus += 0.25f;
        }
    }
    return bonus;
}

}  // namespace mangaflow::pipeline::math_ocr
