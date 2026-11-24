#include "cv_typography.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace vellum::detector {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr float kEdtInf = 1e12f;

inline uint8_t rgbToLuma(uint8_t r, uint8_t g, uint8_t b) noexcept {
    return static_cast<uint8_t>((299 * static_cast<int>(r) +
                                 587 * static_cast<int>(g) +
                                 114 * static_cast<int>(b) + 500) / 1000);
}

inline uint8_t sampleBinaryAt(
    const std::vector<uint8_t>& mask,
    const RectI& rect,
    int x,
    int y) noexcept {
    if (!rect.contains(x, y)) return 0;
    const size_t idx = static_cast<size_t>(y - rect.y1) * static_cast<size_t>(rect.width()) +
                       static_cast<size_t>(x - rect.x1);
    return (idx < mask.size() && mask[idx] != 0) ? 1 : 0;
}

uint8_t medianFromHist256(const std::array<uint32_t, 256>& hist, uint32_t total) noexcept {
    if (total == 0) return 0;
    const uint32_t half = total / 2;
    uint32_t acc = 0;
    for (int i = 0; i < 256; ++i) {
        acc += hist[static_cast<size_t>(i)];
        if (acc > half) return static_cast<uint8_t>(i);
    }
    return 255;
}

// Separable O(W * H) L_infinity erosion of radius `r` (out-of-bounds = 0).
void erodeLInf(
    const std::vector<uint8_t>& src,
    std::vector<uint8_t>& dst,
    int w,
    int h,
    int r) {
    const size_t total = static_cast<size_t>(w) * static_cast<size_t>(h);
    dst.assign(total, 0);
    if (w <= 0 || h <= 0 || src.size() < total) return;
    if (r <= 0) {
        for (size_t i = 0; i < total; ++i) dst[i] = (src[i] != 0) ? 1 : 0;
        return;
    }

    const int need = 2 * r + 1;
    if (w < need || h < need) return;

    std::vector<uint8_t> tmp(total, 0);
    for (int y = 0; y < h; ++y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        int count = 0;
        const int init_hi = std::min(w - 1, r);
        for (int x = 0; x <= init_hi; ++x) {
            if (src[row + static_cast<size_t>(x)] != 0) ++count;
        }
        for (int x = 0; x < w; ++x) {
            if (x >= r && x + r < w && count == need) {
                tmp[row + static_cast<size_t>(x)] = 1;
            }
            const int out_x = x - r;
            const int in_x = x + r + 1;
            if (out_x >= 0 && src[row + static_cast<size_t>(out_x)] != 0) --count;
            if (in_x < w && src[row + static_cast<size_t>(in_x)] != 0) ++count;
        }
    }

    for (int x = 0; x < w; ++x) {
        int count = 0;
        const int init_hi = std::min(h - 1, r);
        for (int y = 0; y <= init_hi; ++y) {
            if (tmp[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] != 0) {
                ++count;
            }
        }
        for (int y = 0; y < h; ++y) {
            if (y >= r && y + r < h && count == need) {
                dst[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] = 1;
            }
            const int out_y = y - r;
            const int in_y = y + r + 1;
            if (out_y >= 0 &&
                tmp[static_cast<size_t>(out_y) * static_cast<size_t>(w) + static_cast<size_t>(x)] != 0) {
                --count;
            }
            if (in_y < h &&
                tmp[static_cast<size_t>(in_y) * static_cast<size_t>(w) + static_cast<size_t>(x)] != 0) {
                ++count;
            }
        }
    }
}

// Separable O(W * H) L_infinity dilation of radius `r`.
void dilateLInf(
    const std::vector<uint8_t>& src,
    std::vector<uint8_t>& dst,
    int w,
    int h,
    int r) {
    const size_t total = static_cast<size_t>(w) * static_cast<size_t>(h);
    dst.assign(total, 0);
    if (w <= 0 || h <= 0 || src.size() < total) return;
    if (r <= 0) {
        for (size_t i = 0; i < total; ++i) dst[i] = (src[i] != 0) ? 1 : 0;
        return;
    }

    std::vector<uint8_t> tmp(total, 0);
    for (int y = 0; y < h; ++y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        int count = 0;
        const int init_hi = std::min(w - 1, r);
        for (int x = 0; x <= init_hi; ++x) {
            if (src[row + static_cast<size_t>(x)] != 0) ++count;
        }
        for (int x = 0; x < w; ++x) {
            if (count > 0) tmp[row + static_cast<size_t>(x)] = 1;
            const int out_x = x - r;
            const int in_x = x + r + 1;
            if (out_x >= 0 && src[row + static_cast<size_t>(out_x)] != 0) --count;
            if (in_x < w && src[row + static_cast<size_t>(in_x)] != 0) ++count;
        }
    }

    for (int x = 0; x < w; ++x) {
        int count = 0;
        const int init_hi = std::min(h - 1, r);
        for (int y = 0; y <= init_hi; ++y) {
            if (tmp[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] != 0) {
                ++count;
            }
        }
        for (int y = 0; y < h; ++y) {
            if (count > 0) {
                dst[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] = 1;
            }
            const int out_y = y - r;
            const int in_y = y + r + 1;
            if (out_y >= 0 &&
                tmp[static_cast<size_t>(out_y) * static_cast<size_t>(w) + static_cast<size_t>(x)] != 0) {
                --count;
            }
            if (in_y < h &&
                tmp[static_cast<size_t>(in_y) * static_cast<size_t>(w) + static_cast<size_t>(x)] != 0) {
                ++count;
            }
        }
    }
}

// 1D Felzenszwalb-Huttenlocher exact squared Euclidean distance transform.
void edt1d(
    const float* f,
    int n,
    float* d,
    std::vector<int>& v,
    std::vector<float>& z) {
    int k = -1;
    for (int q = 0; q < n; ++q) {
        if (f[q] >= kEdtInf * 0.5f) continue;
        if (k < 0) {
            k = 0;
            v[0] = q;
            z[0] = -kEdtInf;
            z[1] = +kEdtInf;
            continue;
        }

        const float fq_qq = f[q] + static_cast<float>(q) * static_cast<float>(q);
        float s = 0.0f;
        while (true) {
            const int vk = v[static_cast<size_t>(k)];
            const float fvk_vv = f[vk] + static_cast<float>(vk) * static_cast<float>(vk);
            s = (fq_qq - fvk_vv) / (2.0f * static_cast<float>(q - vk));
            if (k > 0 && s <= z[static_cast<size_t>(k)]) {
                --k;
            } else {
                break;
            }
        }
        ++k;
        v[static_cast<size_t>(k)] = q;
        z[static_cast<size_t>(k)] = s;
        z[static_cast<size_t>(k + 1)] = +kEdtInf;
    }

    if (k < 0) {
        for (int q = 0; q < n; ++q) d[q] = kEdtInf;
        return;
    }

    int idx = 0;
    for (int q = 0; q < n; ++q) {
        while (z[static_cast<size_t>(idx + 1)] < static_cast<float>(q)) {
            ++idx;
        }
        const float dx = static_cast<float>(q - v[static_cast<size_t>(idx)]);
        d[q] = dx * dx + f[v[static_cast<size_t>(idx)]];
    }
}

// Stage 1: Main-Body Component Filter (removes tiny furigana & specks).
std::vector<uint8_t> filterMainBodyComponents(
    const std::vector<uint8_t>& ink,
    int w,
    int h) {
    const size_t total = static_cast<size_t>(w) * static_cast<size_t>(h);
    if (w <= 0 || h <= 0 || ink.size() < total) return {};

    struct Comp {
        std::vector<int> pixels;
        int max_dim = 0;
        int area = 0;
    };

    static constexpr int kDx8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static constexpr int kDy8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    std::vector<uint8_t> visited(total, 0);
    std::vector<Comp> comps;
    std::vector<int> queue;
    queue.reserve(128);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int start = y * w + x;
            if (ink[static_cast<size_t>(start)] == 0 || visited[static_cast<size_t>(start)] != 0) {
                continue;
            }
            queue.clear();
            queue.push_back(start);
            visited[static_cast<size_t>(start)] = 1;

            int min_x = x, max_x = x, min_y = y, max_y = y;
            size_t head = 0;
            while (head < queue.size()) {
                const int cur = queue[head++];
                const int cx = cur % w;
                const int cy = cur / w;
                if (cx < min_x) min_x = cx;
                if (cx > max_x) max_x = cx;
                if (cy < min_y) min_y = cy;
                if (cy > max_y) max_y = cy;

                for (int k = 0; k < 8; ++k) {
                    const int nx = cx + kDx8[k];
                    const int ny = cy + kDy8[k];
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                    const int nidx = ny * w + nx;
                    if (ink[static_cast<size_t>(nidx)] != 0 && visited[static_cast<size_t>(nidx)] == 0) {
                        visited[static_cast<size_t>(nidx)] = 1;
                        queue.push_back(nidx);
                    }
                }
            }

            Comp c;
            c.pixels = queue;
            c.area = static_cast<int>(queue.size());
            c.max_dim = std::max(max_x - min_x + 1, max_y - min_y + 1);
            comps.push_back(std::move(c));
        }
    }

    if (comps.empty()) {
        return std::vector<uint8_t>(total, 0);
    }

    std::vector<int> dims;
    std::vector<int> areas;
    dims.reserve(comps.size());
    areas.reserve(comps.size());
    for (const auto& c : comps) {
        if (c.area >= 4 && c.max_dim >= 3) {
            dims.push_back(c.max_dim);
            areas.push_back(c.area);
        }
    }

    if (dims.empty()) {
        // Everything was tiny; keep all foreground pixels as-is.
        std::vector<uint8_t> out(total, 0);
        for (size_t i = 0; i < total; ++i) out[i] = (ink[i] != 0) ? 1 : 0;
        return out;
    }

    std::sort(dims.begin(), dims.end());
    std::sort(areas.begin(), areas.end());

    // 75th percentile reference resists skew from multiple furigana/punctuation specks.
    const int ref_dim = dims[(dims.size() * 3) / 4];
    const int ref_area = areas[(areas.size() * 3) / 4];
    const int min_dim = std::max(4, static_cast<int>(std::floor(0.40 * static_cast<double>(ref_dim))));
    const int min_area = std::max(4, static_cast<int>(std::floor(0.18 * static_cast<double>(ref_area))));

    std::vector<uint8_t> main_ink(total, 0);
    int kept_pixels = 0;
    for (const auto& c : comps) {
        if (c.area >= 4 && (c.max_dim >= min_dim || c.area >= min_area)) {
            for (int idx : c.pixels) {
                main_ink[static_cast<size_t>(idx)] = 1;
            }
            kept_pixels += c.area;
        }
    }

    if (kept_pixels == 0) {
        for (size_t i = 0; i < total; ++i) main_ink[i] = (ink[i] != 0) ? 1 : 0;
    }
    return main_ink;
}

// Stage 3a: Stroke width estimate = average(area-based, 3-4 chamfer distance transform).
float estimateStrokeWidth(const std::vector<uint8_t>& main_ink, int w, int h) {
    const size_t total = static_cast<size_t>(w) * static_cast<size_t>(h);
    if (w <= 0 || h <= 0 || main_ink.size() < total) return 1.0f;

    // 1. Area-based estimate: w_area = 2 * Area / Perimeter (4-neighbor fg-bg edges).
    int64_t area = 0;
    int64_t perim = 0;
    for (int y = 0; y < h; ++y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        for (int x = 0; x < w; ++x) {
            if (main_ink[row + static_cast<size_t>(x)] == 0) continue;
            ++area;
            if (x == 0 || main_ink[row + static_cast<size_t>(x - 1)] == 0) ++perim;
            if (x + 1 == w || main_ink[row + static_cast<size_t>(x + 1)] == 0) ++perim;
            if (y == 0 || main_ink[row - static_cast<size_t>(w) + static_cast<size_t>(x)] == 0) ++perim;
            if (y + 1 == h || main_ink[row + static_cast<size_t>(w) + static_cast<size_t>(x)] == 0) ++perim;
        }
    }
    if (area == 0 || perim == 0) return 1.0f;
    const double w_area = std::max(1.0, (2.0 * static_cast<double>(area)) / static_cast<double>(perim));

    // 2. 2-pass linear 3-4 Chamfer Distance Transform inward from background.
    constexpr int kInfChamfer = 1000000;
    std::vector<int> dt(total, 0);
    for (size_t i = 0; i < total; ++i) {
        dt[i] = (main_ink[i] != 0) ? kInfChamfer : 0;
    }

    // Treat out-of-bounds as background (distance 0 -> border fg has distance 3).
    for (int y = 0; y < h; ++y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        for (int x = 0; x < w; ++x) {
            if (dt[row + static_cast<size_t>(x)] == 0) continue;
            int d = dt[row + static_cast<size_t>(x)];
            const int d_l = (x > 0) ? dt[row + static_cast<size_t>(x - 1)] : 0;
            const int d_u = (y > 0) ? dt[row - static_cast<size_t>(w) + static_cast<size_t>(x)] : 0;
            const int d_ul = (x > 0 && y > 0) ? dt[row - static_cast<size_t>(w) + static_cast<size_t>(x - 1)] : 0;
            const int d_ur = (x + 1 < w && y > 0) ? dt[row - static_cast<size_t>(w) + static_cast<size_t>(x + 1)] : 0;
            d = std::min(d, d_l + 3);
            d = std::min(d, d_u + 3);
            d = std::min(d, d_ul + 4);
            d = std::min(d, d_ur + 4);
            dt[row + static_cast<size_t>(x)] = d;
        }
    }

    for (int y = h - 1; y >= 0; --y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        for (int x = w - 1; x >= 0; --x) {
            if (dt[row + static_cast<size_t>(x)] == 0) continue;
            int d = dt[row + static_cast<size_t>(x)];
            const int d_r = (x + 1 < w) ? dt[row + static_cast<size_t>(x + 1)] : 0;
            const int d_d = (y + 1 < h) ? dt[row + static_cast<size_t>(w) + static_cast<size_t>(x)] : 0;
            const int d_dr = (x + 1 < w && y + 1 < h) ? dt[row + static_cast<size_t>(w) + static_cast<size_t>(x + 1)] : 0;
            const int d_dl = (x > 0 && y + 1 < h) ? dt[row + static_cast<size_t>(w) + static_cast<size_t>(x - 1)] : 0;
            d = std::min(d, d_r + 3);
            d = std::min(d, d_d + 3);
            d = std::min(d, d_dr + 4);
            d = std::min(d, d_dl + 4);
            dt[row + static_cast<size_t>(x)] = d;
        }
    }

    // Collect medial-axis (local-maximum) chamfer distances.
    std::vector<int> ridge_dists;
    ridge_dists.reserve(static_cast<size_t>(area / 2 + 16));
    static constexpr int kDx8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static constexpr int kDy8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    for (int y = 0; y < h; ++y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        for (int x = 0; x < w; ++x) {
            const int d = dt[row + static_cast<size_t>(x)];
            if (d <= 0) continue;
            bool is_ridge = true;
            for (int k = 0; k < 8; ++k) {
                const int nx = x + kDx8[k];
                const int ny = y + kDy8[k];
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    if (dt[static_cast<size_t>(ny) * static_cast<size_t>(w) + static_cast<size_t>(nx)] > d) {
                        is_ridge = false;
                        break;
                    }
                }
            }
            if (is_ridge) {
                ridge_dists.push_back(d);
            }
        }
    }

    double w_chamfer = w_area;
    if (!ridge_dists.empty()) {
        std::sort(ridge_dists.begin(), ridge_dists.end());
        const double med_r = static_cast<double>(ridge_dists[ridge_dists.size() / 2]) / 3.0;
        w_chamfer = std::max(1.0, 2.0 * med_r - 0.5);
    }

    return static_cast<float>(0.5 * (w_area + w_chamfer));
}

// Stage 3b: Project ink pixels along rotated reading axis, discard slices with < 3 px ink
// or < 50% median ink, and take median cross-axis span as `estimated_font_size`.
// Segments multi-line blocks into individual line bands along the cross axis first so
// multi-line bubbles return single-line font size rather than total block height/width.
float estimateFontSizeProjected(
    const std::vector<uint8_t>& main_ink,
    int w,
    int h,
    bool is_vertical,
    float angle_deg) {
    if (w <= 0 || h <= 0) return 0.0f;

    const double rad = static_cast<double>(angle_deg) * (kPi / 180.0);
    const double ca = std::cos(rad);
    const double sa = std::sin(rad);

    // Reading axis `a` and cross axis `b`:
    // Horizontal: a = (cos, sin), b = (-sin, cos)
    // Vertical:   a = (-sin, cos), b = (cos, sin)
    const double ax = is_vertical ? -sa : ca;
    const double ay = is_vertical ? ca : sa;
    const double bx = is_vertical ? ca : -sa;
    const double by = is_vertical ? sa : ca;

    const double cx0 = 0.5 * static_cast<double>(w);
    const double cy0 = 0.5 * static_cast<double>(h);

    struct ProjPt {
        int s = 0;  // reading-axis bin
        int t = 0;  // cross-axis coord
    };
    std::vector<ProjPt> pts;
    pts.reserve(static_cast<size_t>(w * h / 4 + 64));

    int min_s = std::numeric_limits<int>::max();
    int max_s = std::numeric_limits<int>::min();
    int min_t = std::numeric_limits<int>::max();
    int max_t = std::numeric_limits<int>::min();

    for (int y = 0; y < h; ++y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        const double dy = static_cast<double>(y) - cy0;
        for (int x = 0; x < w; ++x) {
            if (main_ink[row + static_cast<size_t>(x)] == 0) continue;
            const double dx = static_cast<double>(x) - cx0;
            const int s = static_cast<int>(std::lround(dx * ax + dy * ay));
            const int t = static_cast<int>(std::lround(dx * bx + dy * by));
            pts.push_back(ProjPt{s, t});
            if (s < min_s) min_s = s;
            if (s > max_s) max_s = s;
            if (t < min_t) min_t = t;
            if (t > max_t) max_t = t;
        }
    }

    if (pts.empty() || max_t < min_t || max_s < min_s) {
        return 0.0f;
    }

    // Segment into individual line bands along the cross axis `t`.
    const int t_len = max_t - min_t + 1;
    std::vector<int> t_hist(static_cast<size_t>(t_len), 0);
    for (const auto& p : pts) {
        ++t_hist[static_cast<size_t>(p.t - min_t)];
    }

    struct Band {
        int t0 = 0;
        int t1 = 0;
    };
    std::vector<Band> bands;
    int cur_start = -1;
    for (int i = 0; i < t_len; ++i) {
        if (t_hist[static_cast<size_t>(i)] > 0) {
            if (cur_start < 0) cur_start = i;
        } else if (cur_start >= 0) {
            if (i - cur_start >= 4) {
                bands.push_back(Band{min_t + cur_start, min_t + i - 1});
            }
            cur_start = -1;
        }
    }
    if (cur_start >= 0 && t_len - cur_start >= 4) {
        bands.push_back(Band{min_t + cur_start, min_t + t_len - 1});
    }
    if (bands.empty()) {
        bands.push_back(Band{min_t, max_t});
    }

    // Collect valid reading-axis slices across each line band.
    const int s_len = max_s - min_s + 1;
    std::vector<int> slice_counts(static_cast<size_t>(s_len));
    std::vector<int> slice_tmin(static_cast<size_t>(s_len));
    std::vector<int> slice_tmax(static_cast<size_t>(s_len));

    std::vector<int> all_spans;
    all_spans.reserve(static_cast<size_t>(s_len) * bands.size());

    for (const auto& band : bands) {
        std::fill(slice_counts.begin(), slice_counts.end(), 0);
        std::fill(slice_tmin.begin(), slice_tmin.end(), std::numeric_limits<int>::max());
        std::fill(slice_tmax.begin(), slice_tmax.end(), std::numeric_limits<int>::min());

        for (const auto& p : pts) {
            if (p.t < band.t0 || p.t > band.t1) continue;
            const size_t si = static_cast<size_t>(p.s - min_s);
            ++slice_counts[si];
            if (p.t < slice_tmin[si]) slice_tmin[si] = p.t;
            if (p.t > slice_tmax[si]) slice_tmax[si] = p.t;
        }

        // Gather counts for slices with >= 3 px ink to compute median ink per slice.
        std::vector<int> valid_counts;
        valid_counts.reserve(static_cast<size_t>(s_len));
        for (int i = 0; i < s_len; ++i) {
            if (slice_counts[static_cast<size_t>(i)] >= 3) {
                valid_counts.push_back(slice_counts[static_cast<size_t>(i)]);
            }
        }
        if (valid_counts.empty()) continue;

        std::sort(valid_counts.begin(), valid_counts.end());
        const double med_ink = static_cast<double>(valid_counts[valid_counts.size() / 2]);
        const double min_ink_thresh = 0.5 * med_ink;

        for (int i = 0; i < s_len; ++i) {
            const int cnt = slice_counts[static_cast<size_t>(i)];
            if (cnt < 3 || static_cast<double>(cnt) < min_ink_thresh) continue;
            const int span = slice_tmax[static_cast<size_t>(i)] - slice_tmin[static_cast<size_t>(i)] + 1;
            if (span > 0) {
                all_spans.push_back(span);
            }
        }
    }

    if (all_spans.empty()) {
        // Fallback to median band height
        std::vector<int> band_spans;
        for (const auto& b : bands) band_spans.push_back(b.t1 - b.t0 + 1);
        std::sort(band_spans.begin(), band_spans.end());
        return static_cast<float>(band_spans[band_spans.size() / 2]);
    }

    std::sort(all_spans.begin(), all_spans.end());
    return static_cast<float>(all_spans[all_spans.size() / 2]);
}

}  // namespace

// -----------------------------------------------------------------------------
// Public API Implementations
// -----------------------------------------------------------------------------

bool computeEuclideanDistanceSqFH(
    const uint8_t* fg,
    int width,
    int height,
    std::vector<float>& dist_sq) {
    if (!fg || width <= 0 || height <= 0) {
        dist_sq.clear();
        return false;
    }

    const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);
    dist_sq.assign(total, kEdtInf);
    bool any_fg = false;
    for (size_t i = 0; i < total; ++i) {
        if (fg[i] != 0) {
            dist_sq[i] = 0.0f;
            any_fg = true;
        }
    }
    if (!any_fg) {
        return false;
    }

    const int max_dim = std::max(width, height);
    std::vector<float> f_buf(static_cast<size_t>(max_dim));
    std::vector<float> d_buf(static_cast<size_t>(max_dim));
    std::vector<int> v_buf(static_cast<size_t>(max_dim));
    std::vector<float> z_buf(static_cast<size_t>(max_dim + 1));

    // Pass 1: transform along columns
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            f_buf[static_cast<size_t>(y)] =
                dist_sq[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)];
        }
        edt1d(f_buf.data(), height, d_buf.data(), v_buf, z_buf);
        for (int y = 0; y < height; ++y) {
            dist_sq[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] =
                d_buf[static_cast<size_t>(y)];
        }
    }

    // Pass 2: transform along rows
    for (int y = 0; y < height; ++y) {
        float* row = dist_sq.data() + static_cast<size_t>(y) * static_cast<size_t>(width);
        std::memcpy(f_buf.data(), row, static_cast<size_t>(width) * sizeof(float));
        edt1d(f_buf.data(), width, row, v_buf, z_buf);
    }

    return true;
}

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
    const std::vector<uint8_t>& block_mask) {
    TypographyStyle out;
    if (!rgb_data || page_w <= 0 || page_h <= 0) {
        return out;
    }

    const RectI core_box = bbox.clamped(page_w, page_h);
    if (core_box.empty() || ink_mask_rect.empty() || ink_mask.empty()) {
        return out;
    }

    const int iw = ink_mask_rect.width();
    const int ih = ink_mask_rect.height();
    if (ink_mask.size() < static_cast<size_t>(iw) * static_cast<size_t>(ih)) {
        return out;
    }

    // -------------------------------------------------------------------------
    // 1. Main-Body Component Filter
    // -------------------------------------------------------------------------
    const std::vector<uint8_t> main_ink = filterMainBodyComponents(ink_mask, iw, ih);

    // -------------------------------------------------------------------------
    // 2. Trig-Free Sobel 4-theta Tilt & Orientation
    // -------------------------------------------------------------------------
    const int reach = scaledMorphRadius(6.0, page_w, page_h);
    const int erode_r = reach + 1;

    if (!block_mask_rect.empty() &&
        block_mask.size() >= static_cast<size_t>(block_mask_rect.width()) *
                                 static_cast<size_t>(block_mask_rect.height())) {
        const int bw = block_mask_rect.width();
        const int bh = block_mask_rect.height();
        std::vector<uint8_t> eroded_bm;
        erodeLInf(block_mask, eroded_bm, bw, bh, erode_r);

        // Compute grayscale on block_mask_rect padded by 1px for 3x3 Sobel
        const RectI sobel_rect = block_mask_rect.expanded(1, 1, page_w, page_h);
        const int sw = sobel_rect.width();
        const int sh = sobel_rect.height();
        std::vector<uint8_t> gray(static_cast<size_t>(sw) * static_cast<size_t>(sh), 0);
        for (int sy = 0; sy < sh; ++sy) {
            const int gy = sobel_rect.y1 + sy;
            const size_t rgb_row = static_cast<size_t>(gy) * static_cast<size_t>(page_w) * 3ULL;
            const size_t grow = static_cast<size_t>(sy) * static_cast<size_t>(sw);
            for (int sx = 0; sx < sw; ++sx) {
                const int gx = sobel_rect.x1 + sx;
                const uint8_t* px = rgb_data + rgb_row + static_cast<size_t>(gx) * 3ULL;
                gray[grow + static_cast<size_t>(sx)] = rgbToLuma(px[0], px[1], px[2]);
            }
        }

        double sum_m = 0.0;
        double sum_c4 = 0.0;
        double sum_s4 = 0.0;

        // Restrict Sobel 4-theta sampling strictly to eroded interior ink pixels
        // (excluding curved/spiky speech-bubble borders from polluting tilt angles).
        std::vector<uint8_t> ink_near;
        dilateLInf(main_ink, ink_near, iw, ih, 1);

        for (int by = 0; by < bh; ++by) {
            const int gy = block_mask_rect.y1 + by;
            if (gy <= sobel_rect.y1 || gy >= sobel_rect.y2 - 1) continue;
            if (gy < ink_mask_rect.y1 || gy >= ink_mask_rect.y2) continue;
            const int sy = gy - sobel_rect.y1;
            const int iy = gy - ink_mask_rect.y1;
            const size_t r0 = static_cast<size_t>(sy - 1) * static_cast<size_t>(sw);
            const size_t r1 = static_cast<size_t>(sy) * static_cast<size_t>(sw);
            const size_t r2 = static_cast<size_t>(sy + 1) * static_cast<size_t>(sw);
            const size_t brow = static_cast<size_t>(by) * static_cast<size_t>(bw);
            const size_t irow = static_cast<size_t>(iy) * static_cast<size_t>(iw);

            for (int bx = 0; bx < bw; ++bx) {
                if (eroded_bm[brow + static_cast<size_t>(bx)] == 0) continue;
                const int gx_coord = block_mask_rect.x1 + bx;
                if (gx_coord <= sobel_rect.x1 || gx_coord >= sobel_rect.x2 - 1) continue;
                if (gx_coord < ink_mask_rect.x1 || gx_coord >= ink_mask_rect.x2) continue;
                const int ix = gx_coord - ink_mask_rect.x1;
                if (ink_near[irow + static_cast<size_t>(ix)] == 0) continue;
                const int sx = gx_coord - sobel_rect.x1;

                const double p00 = static_cast<double>(gray[r0 + static_cast<size_t>(sx - 1)]);
                const double p01 = static_cast<double>(gray[r0 + static_cast<size_t>(sx)]);
                const double p02 = static_cast<double>(gray[r0 + static_cast<size_t>(sx + 1)]);
                const double p10 = static_cast<double>(gray[r1 + static_cast<size_t>(sx - 1)]);
                const double p12 = static_cast<double>(gray[r1 + static_cast<size_t>(sx + 1)]);
                const double p20 = static_cast<double>(gray[r2 + static_cast<size_t>(sx - 1)]);
                const double p21 = static_cast<double>(gray[r2 + static_cast<size_t>(sx)]);
                const double p22 = static_cast<double>(gray[r2 + static_cast<size_t>(sx + 1)]);

                const double gx = (-p00 + p02) + 2.0 * (-p10 + p12) + (-p20 + p22);
                const double gy_v = (-p00 - 2.0 * p01 - p02) + (p20 + 2.0 * p21 + p22);

                const double gx2 = gx * gx;
                const double gy2 = gy_v * gy_v;
                const double m = gx2 + gy2;
                if (m <= 1e-6) continue;

                const double u = gx2 - gy2;
                const double v = 2.0 * gx * gy_v;
                const double c4 = (u * u - v * v) / m;
                const double s4 = (2.0 * u * v) / m;

                sum_m += m;
                sum_c4 += c4;
                sum_s4 += s4;
            }
        }

        if (sum_m > 1e-6) {
            const double r4 = std::hypot(sum_c4, sum_s4) / sum_m;
            out.coherence_r4 = static_cast<float>(r4);

            // Strict deadzone & coherence gate: clamp all angles < 12 deg or coherence < 0.45 to 0.0f.
            // Standard dialogue/narration requires unambiguously strong ink tilt (>= 14 deg, R >= 0.50).
            if (r4 >= 0.45) {
                const double theta_rad = 0.25 * std::atan2(sum_s4, sum_c4);
                double deg = theta_rad * (180.0 / kPi);

                const double box_w = static_cast<double>(core_box.width());
                const double box_h = static_cast<double>(core_box.height());
                const double aspect = std::max(box_w, box_h) / std::max(1.0, std::min(box_w, box_h));

                const double min_tilt_deg = is_onomatopoeia ? 12.0 : 14.0;
                const double min_coherence = is_onomatopoeia ? 0.45 : 0.50;
                const bool suppress_small = (std::abs(deg) < min_tilt_deg) || (r4 < min_coherence);
                const bool suppress_square = (aspect < 2.0) && !(is_onomatopoeia && r4 >= 0.55);
                if (suppress_small || suppress_square) {
                    deg = 0.0;
                }
                out.angle_deg = static_cast<float>(deg);
            } else {
                out.angle_deg = 0.0f;
            }
        }
    }

    // -------------------------------------------------------------------------
    // 3. Stroke Width & Font Size
    // -------------------------------------------------------------------------
    out.stroke_width = estimateStrokeWidth(main_ink, iw, ih);
    out.estimated_font_size =
        estimateFontSizeProjected(main_ink, iw, ih, is_vertical, out.angle_deg);

    // -------------------------------------------------------------------------
    // 4. Core Ink Color & Distance-Ring Outline Detector
    // -------------------------------------------------------------------------
    std::vector<uint8_t> core_ink;
    if (out.stroke_width >= 3.0f) {
        erodeLInf(main_ink, core_ink, iw, ih, 1);
        int core_cnt = 0;
        for (uint8_t v : core_ink) {
            if (v != 0) ++core_cnt;
        }
        if (core_cnt == 0) {
            core_ink = main_ink;
        }
    } else {
        core_ink = main_ink;
    }

    {
        std::array<uint32_t, 256> hr{}, hg{}, hb{};
        uint32_t cnt = 0;
        for (int ly = 0; ly < ih; ++ly) {
            const int gy = ink_mask_rect.y1 + ly;
            const size_t rgb_row = static_cast<size_t>(gy) * static_cast<size_t>(page_w) * 3ULL;
            const size_t irow = static_cast<size_t>(ly) * static_cast<size_t>(iw);
            for (int lx = 0; lx < iw; ++lx) {
                if (core_ink[irow + static_cast<size_t>(lx)] == 0) continue;
                const int gx = ink_mask_rect.x1 + lx;
                const uint8_t* px = rgb_data + rgb_row + static_cast<size_t>(gx) * 3ULL;
                ++hr[px[0]];
                ++hg[px[1]];
                ++hb[px[2]];
                ++cnt;
            }
        }
        if (cnt > 0) {
            out.text_color = {
                medianFromHist256(hr, cnt),
                medianFromHist256(hg, cnt),
                medianFromHist256(hb, cnt)
            };
            out.valid = true;
        }
    }

    // Distance-Ring Outline Detector via Felzenszwalb-Huttenlocher EDT
    const int max_ring = std::clamp(static_cast<int>(std::lround(out.stroke_width)), 2, 8);
    const RectI edt_rect = ink_mask_rect.expanded(max_ring + 2, max_ring + 2, page_w, page_h);
    const int ew = edt_rect.width();
    const int eh = edt_rect.height();

    if (ew > 0 && eh > 0) {
        std::vector<uint8_t> edt_fg(static_cast<size_t>(ew) * static_cast<size_t>(eh), 0);
        for (int ly = 0; ly < ih; ++ly) {
            const int ey = ink_mask_rect.y1 + ly - edt_rect.y1;
            const size_t srow = static_cast<size_t>(ly) * static_cast<size_t>(iw);
            const size_t drow = static_cast<size_t>(ey) * static_cast<size_t>(ew);
            for (int lx = 0; lx < iw; ++lx) {
                if (ink_mask[srow + static_cast<size_t>(lx)] != 0) {
                    const int ex = ink_mask_rect.x1 + lx - edt_rect.x1;
                    edt_fg[drow + static_cast<size_t>(ex)] = 1;
                }
            }
        }

        std::vector<float> dist_sq;
        if (computeEuclideanDistanceSqFH(edt_fg.data(), ew, eh, dist_sq)) {
            std::vector<std::array<uint32_t, 256>> ring_r(static_cast<size_t>(max_ring));
            std::vector<std::array<uint32_t, 256>> ring_g(static_cast<size_t>(max_ring));
            std::vector<std::array<uint32_t, 256>> ring_b(static_cast<size_t>(max_ring));
            std::vector<uint32_t> ring_cnt(static_cast<size_t>(max_ring), 0);

            const float max_dist_sq =
                static_cast<float>(max_ring) + 0.5f;
            const float max_d2 = max_dist_sq * max_dist_sq;

            for (int ey = 0; ey < eh; ++ey) {
                const int gy = edt_rect.y1 + ey;
                const size_t rgb_row = static_cast<size_t>(gy) * static_cast<size_t>(page_w) * 3ULL;
                const size_t erow = static_cast<size_t>(ey) * static_cast<size_t>(ew);
                for (int ex = 0; ex < ew; ++ex) {
                    const size_t eidx = erow + static_cast<size_t>(ex);
                    if (edt_fg[eidx] != 0) continue;
                    const float d2 = dist_sq[eidx];
                    if (d2 <= 0.0f || d2 > max_d2) continue;
                    const int r = static_cast<int>(std::floor(std::sqrt(d2) + 0.5f));
                    if (r < 1 || r > max_ring) continue;

                    const int gx = edt_rect.x1 + ex;
                    const uint8_t* px = rgb_data + rgb_row + static_cast<size_t>(gx) * 3ULL;
                    const size_t ri = static_cast<size_t>(r - 1);
                    ++ring_r[ri][px[0]];
                    ++ring_g[ri][px[1]];
                    ++ring_b[ri][px[2]];
                    ++ring_cnt[ri];
                }
            }

            std::vector<std::array<uint8_t, 3>> ring_colors(static_cast<size_t>(max_ring));
            bool all_rings_populated = true;
            for (int r = 0; r < max_ring; ++r) {
                if (ring_cnt[static_cast<size_t>(r)] == 0) {
                    all_rings_populated = false;
                    break;
                }
                ring_colors[static_cast<size_t>(r)] = {
                    medianFromHist256(ring_r[static_cast<size_t>(r)], ring_cnt[static_cast<size_t>(r)]),
                    medianFromHist256(ring_g[static_cast<size_t>(r)], ring_cnt[static_cast<size_t>(r)]),
                    medianFromHist256(ring_b[static_cast<size_t>(r)], ring_cnt[static_cast<size_t>(r)])
                };
            }

            if (all_rings_populated) {
                const auto& bg_col = ring_colors[static_cast<size_t>(max_ring - 1)];
                std::vector<double> d_ring(static_cast<size_t>(max_ring), 0.0);
                for (int r = 0; r < max_ring; ++r) {
                    const double dr = static_cast<double>(ring_colors[static_cast<size_t>(r)][0]) -
                                      static_cast<double>(bg_col[0]);
                    const double dg = static_cast<double>(ring_colors[static_cast<size_t>(r)][1]) -
                                      static_cast<double>(bg_col[1]);
                    const double db = static_cast<double>(ring_colors[static_cast<size_t>(r)][2]) -
                                      static_cast<double>(bg_col[2]);
                    d_ring[static_cast<size_t>(r)] = std::sqrt(dr * dr + dg * dg + db * db);
                }

                double max_drop = 0.0;
                int best_r = 0;
                for (int r = 1; r < max_ring; ++r) {
                    const double drop = d_ring[static_cast<size_t>(r - 1)] - d_ring[static_cast<size_t>(r)];
                    if (drop > max_drop) {
                        max_drop = drop;
                        best_r = r;
                    }
                }

                if (max_drop >= 20.0 && best_r >= 1) {
                    // Guard against 1px font anti-aliasing on plain backgrounds:
                    // if width == 1 and ring_colors[0] lies on the interior of the RGB line
                    // segment between text_color and bg_col, it is an anti-aliased edge rather
                    // than a distinct stroke outline.
                    bool is_antialias_fringe = false;
                    if (best_r == 1) {
                        const double vx = static_cast<double>(bg_col[0]) - static_cast<double>(out.text_color[0]);
                        const double vy = static_cast<double>(bg_col[1]) - static_cast<double>(out.text_color[1]);
                        const double vz = static_cast<double>(bg_col[2]) - static_cast<double>(out.text_color[2]);
                        const double v_len2 = vx * vx + vy * vy + vz * vz;
                        if (v_len2 > 1.0) {
                            const double px = static_cast<double>(ring_colors[0][0]) - static_cast<double>(out.text_color[0]);
                            const double py = static_cast<double>(ring_colors[0][1]) - static_cast<double>(out.text_color[1]);
                            const double pz = static_cast<double>(ring_colors[0][2]) - static_cast<double>(out.text_color[2]);
                            const double t = (px * vx + py * vy + pz * vz) / v_len2;
                            if (t > 0.0 && t < 1.0) {
                                const double dx = px - t * vx;
                                const double dy = py - t * vy;
                                const double dz = pz - t * vz;
                                const double perp_dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                                if (perp_dist < 15.0) {
                                    is_antialias_fringe = true;
                                }
                            }
                        }
                    }

                    if (!is_antialias_fringe) {
                        OutlineStyle ol;
                        ol.color = ring_colors[0];
                        ol.width = best_r;
                        out.outline = ol;
                    }
                }
            }
        }
    }

    return out;
}

TypographyStyle analyzeBlockTypographyFromBox(
    const uint8_t* rgb_data,
    int page_w,
    int page_h,
    const RectI& bbox,
    bool is_vertical,
    bool is_onomatopoeia,
    const RectI* optional_mask_rect,
    const std::vector<uint8_t>* optional_mask) {
    if (!rgb_data || page_w <= 0 || page_h <= 0) {
        return {};
    }

    const RectI core_box = bbox.clamped(page_w, page_h);
    if (core_box.empty()) return {};

    const int reach = scaledMorphRadius(6.0, page_w, page_h);
    const RectI otsu_rect = core_box.expanded(reach, reach, page_w, page_h);
    const int ow = otsu_rect.width();
    const int oh = otsu_rect.height();
    const size_t ototal = static_cast<size_t>(ow) * static_cast<size_t>(oh);

    std::vector<uint8_t> gray(ototal, 0);
    for (int oy = 0; oy < oh; ++oy) {
        const int gy = otsu_rect.y1 + oy;
        const size_t rgb_row = static_cast<size_t>(gy) * static_cast<size_t>(page_w) * 3ULL;
        const size_t orow = static_cast<size_t>(oy) * static_cast<size_t>(ow);
        for (int ox = 0; ox < ow; ++ox) {
            const int gx = otsu_rect.x1 + ox;
            const uint8_t* px = rgb_data + rgb_row + static_cast<size_t>(gx) * 3ULL;
            gray[orow + static_cast<size_t>(ox)] = rgbToLuma(px[0], px[1], px[2]);
        }
    }

    const uint8_t thresh = computeOtsuThreshold1Tile(gray.data(), ow, oh, 0.1);

    // Determine ink polarity from pixels outside optional_mask (or outside core_box).
    int ring_total = 0;
    int ring_dark = 0;
    const bool has_opt_mask =
        (optional_mask_rect != nullptr && optional_mask != nullptr &&
         !optional_mask_rect->empty() && !optional_mask->empty());

    for (int oy = 0; oy < oh; ++oy) {
        const int gy = otsu_rect.y1 + oy;
        const size_t orow = static_cast<size_t>(oy) * static_cast<size_t>(ow);
        for (int ox = 0; ox < ow; ++ox) {
            const int gx = otsu_rect.x1 + ox;
            const bool inside = has_opt_mask
                ? (sampleBinaryAt(*optional_mask, *optional_mask_rect, gx, gy) != 0)
                : core_box.contains(gx, gy);
            if (inside) continue;
            ++ring_total;
            if (gray[orow + static_cast<size_t>(ox)] <= thresh) ++ring_dark;
        }
    }
    const bool light_on_dark =
        (ring_total > 0) && (static_cast<double>(ring_dark) / static_cast<double>(ring_total) > 0.5);

    std::vector<uint8_t> ink_mask(ototal, 0);
    for (int oy = 0; oy < oh; ++oy) {
        const int gy = otsu_rect.y1 + oy;
        const size_t orow = static_cast<size_t>(oy) * static_cast<size_t>(ow);
        for (int ox = 0; ox < ow; ++ox) {
            const int gx = otsu_rect.x1 + ox;
            const bool inside = has_opt_mask
                ? (sampleBinaryAt(*optional_mask, *optional_mask_rect, gx, gy) != 0)
                : core_box.contains(gx, gy);
            if (!inside) continue;
            const uint8_t luma = gray[orow + static_cast<size_t>(ox)];
            const bool is_ink = light_on_dark ? (luma > thresh) : (luma <= thresh);
            if (is_ink) {
                ink_mask[orow + static_cast<size_t>(ox)] = 1;
            }
        }
    }

    // Build solid `block_mask` via L_infinity dilation of `ink_mask` by `reach` (matching
    // Stage 2 solid block mask so eroding by `reach + 1` isolates the glyph interior).
    std::vector<uint8_t> block_mask;
    dilateLInf(ink_mask, block_mask, ow, oh, reach);

    return analyzeBlockTypography(
        rgb_data,
        page_w,
        page_h,
        core_box,
        is_vertical,
        is_onomatopoeia,
        otsu_rect,
        ink_mask,
        otsu_rect,
        block_mask);
}

}  // namespace vellum::detector
