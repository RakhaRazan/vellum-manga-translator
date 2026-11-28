#include "math_det_ops.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vellum::detector {

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kDegToRad = kPi / 180.0f;
constexpr float kRadToDeg = 180.0f / kPi;

inline int rgbLumaInt(const uint8_t* px) noexcept {
    return (77 * static_cast<int>(px[0]) +
            150 * static_cast<int>(px[1]) +
            29 * static_cast<int>(px[2])) >> 8;
}

inline int rgbLumaAt(const uint8_t* rgb, int w, int h, int x, int y) noexcept {
    const int cx = std::clamp(x, 0, w - 1);
    const int cy = std::clamp(y, 0, h - 1);
    const size_t off = (static_cast<size_t>(cy) * static_cast<size_t>(w) +
                        static_cast<size_t>(cx)) * 3ULL;
    return rgbLumaInt(rgb + off);
}

// O(N) integral-image box mean filter (exact boundary clamping).
void boxMeanIntegral(
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
        const size_t src_row = static_cast<size_t>(y) * static_cast<size_t>(w);
        const size_t sat_prev = static_cast<size_t>(y) * static_cast<size_t>(sw);
        const size_t sat_cur = static_cast<size_t>(y + 1) * static_cast<size_t>(sw);
        double row_sum = 0.0;
        for (int x = 0; x < w; ++x) {
            row_sum += static_cast<double>(src[src_row + static_cast<size_t>(x)]);
            sat[sat_cur + static_cast<size_t>(x + 1)] =
                sat[sat_prev + static_cast<size_t>(x + 1)] + row_sum;
        }
    }

    dst.resize(static_cast<size_t>(w) * static_cast<size_t>(h));
    for (int y = 0; y < h; ++y) {
        const int y0 = std::max(0, y - r);
        const int y1 = std::min(h - 1, y + r);
        const size_t sat_r0 = static_cast<size_t>(y0) * static_cast<size_t>(sw);
        const size_t sat_r1 = static_cast<size_t>(y1 + 1) * static_cast<size_t>(sw);
        const size_t dst_row = static_cast<size_t>(y) * static_cast<size_t>(w);
        const double dy = static_cast<double>(y1 - y0 + 1);

        for (int x = 0; x < w; ++x) {
            const int x0 = std::max(0, x - r);
            const int x1 = std::min(w - 1, x + r);
            const double sum =
                sat[sat_r1 + static_cast<size_t>(x1 + 1)] -
                sat[sat_r0 + static_cast<size_t>(x1 + 1)] -
                sat[sat_r1 + static_cast<size_t>(x0)] +
                sat[sat_r0 + static_cast<size_t>(x0)];
            const double area = dy * static_cast<double>(x1 - x0 + 1);
            dst[dst_row + static_cast<size_t>(x)] = static_cast<float>(sum / area);
        }
    }
}

struct DisjointSet {
    std::vector<int> parent;
    explicit DisjointSet(size_t n) : parent(n) {
        std::iota(parent.begin(), parent.end(), 0);
    }
    int find(int x) noexcept {
        while (parent[static_cast<size_t>(x)] != x) {
            parent[static_cast<size_t>(x)] = parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
            x = parent[static_cast<size_t>(x)];
        }
        return x;
    }
    void unite(int a, int b) noexcept {
        const int ra = find(a);
        const int rb = find(b);
        if (ra != rb) {
            parent[static_cast<size_t>(rb)] = ra;
        }
    }
};

// Solve small dense linear system A * x = b using Gaussian elimination with partial pivoting.
bool solveLinearSystem(std::vector<double>& A, std::vector<double>& b, int n, std::vector<double>& x) {
    x.assign(static_cast<size_t>(n), 0.0);
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double max_abs = std::abs(A[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(col)]);
        for (int row = col + 1; row < n; ++row) {
            const double v = std::abs(A[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)]);
            if (v > max_abs) {
                max_abs = v;
                pivot = row;
            }
        }
        if (max_abs < 1e-12) {
            return false;
        }
        if (pivot != col) {
            for (int c = col; c < n; ++c) {
                std::swap(
                    A[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(c)],
                    A[static_cast<size_t>(pivot) * static_cast<size_t>(n) + static_cast<size_t>(c)]);
            }
            std::swap(b[static_cast<size_t>(col)], b[static_cast<size_t>(pivot)]);
        }
        const double diag = A[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(col)];
        for (int row = col + 1; row < n; ++row) {
            const double factor = A[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] / diag;
            if (std::abs(factor) < 1e-15) continue;
            for (int c = col; c < n; ++c) {
                A[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(c)] -=
                    factor * A[static_cast<size_t>(col) * static_cast<size_t>(n) + static_cast<size_t>(c)];
            }
            b[static_cast<size_t>(row)] -= factor * b[static_cast<size_t>(col)];
        }
    }
    for (int row = n - 1; row >= 0; --row) {
        double sum = b[static_cast<size_t>(row)];
        for (int c = row + 1; c < n; ++c) {
            sum -= A[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(c)] * x[static_cast<size_t>(c)];
        }
        x[static_cast<size_t>(row)] =
            sum / A[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(row)];
    }
    return true;
}

inline double tpsKernel(double r2) noexcept {
    if (r2 <= 1e-12) return 0.0;
    return 0.5 * r2 * std::log(r2);  // r^2 * ln(r)
}

}  // namespace

// =============================================================================
// #1 — O(N) Integral-Image Guided Image Filter
// =============================================================================
void applyGuidedFilter(
    const float* guide_I,
    const float* src_p,
    int width,
    int height,
    int radius,
    float eps,
    std::vector<float>& out_q) {
    if (!guide_I || !src_p || width <= 0 || height <= 0) {
        out_q.clear();
        return;
    }

    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    const int r = std::max(1, radius);
    const float safe_eps = std::max(1e-6f, eps);

    std::vector<double> sat;
    std::vector<float> mean_I;
    std::vector<float> mean_p;
    std::vector<float> corr_II;
    std::vector<float> corr_Ip;
    std::vector<float> buf(n);

    boxMeanIntegral(guide_I, width, height, r, sat, mean_I);
    boxMeanIntegral(src_p, width, height, r, sat, mean_p);

    for (size_t i = 0; i < n; ++i) {
        buf[i] = guide_I[i] * guide_I[i];
    }
    boxMeanIntegral(buf.data(), width, height, r, sat, corr_II);

    for (size_t i = 0; i < n; ++i) {
        buf[i] = guide_I[i] * src_p[i];
    }
    boxMeanIntegral(buf.data(), width, height, r, sat, corr_Ip);

    std::vector<float> a(n);
    std::vector<float> b(n);
    for (size_t i = 0; i < n; ++i) {
        const float var_I = std::max(0.0f, corr_II[i] - mean_I[i] * mean_I[i]);
        const float cov_Ip = corr_Ip[i] - mean_I[i] * mean_p[i];
        const float ak = cov_Ip / (var_I + safe_eps);
        a[i] = ak;
        b[i] = mean_p[i] - ak * mean_I[i];
    }

    std::vector<float> mean_a;
    std::vector<float> mean_b;
    boxMeanIntegral(a.data(), width, height, r, sat, mean_a);
    boxMeanIntegral(b.data(), width, height, r, sat, mean_b);

    out_q.resize(n);
    for (size_t i = 0; i < n; ++i) {
        out_q[i] = std::clamp(mean_a[i] * guide_I[i] + mean_b[i], 0.0f, 1.0f);
    }
}

// =============================================================================
// #2 — Radon/Hough Gutter Panel Detection
// =============================================================================
std::vector<PanelDetection> detectPanelsRadonHough(
    const uint8_t* rgbData,
    int width,
    int height) {
    if (!rgbData || width < 64 || height < 64) {
        return {};
    }

    // Downscale to <= 320px long side for ~1-2 ms execution.
    const int long_side = std::max(width, height);
    const int ds = std::max(1, (long_side + 319) / 320);
    const int sw = std::max(1, width / ds);
    const int sh = std::max(1, height / ds);
    const size_t stotal = static_cast<size_t>(sw) * static_cast<size_t>(sh);

    std::vector<uint8_t> gray(stotal, 255);
    for (int sy = 0; sy < sh; ++sy) {
        const int gy = std::min(height - 1, sy * ds + ds / 2);
        const size_t src_row = static_cast<size_t>(gy) * static_cast<size_t>(width) * 3ULL;
        const size_t dst_row = static_cast<size_t>(sy) * static_cast<size_t>(sw);
        for (int sx = 0; sx < sw; ++sx) {
            const int gx = std::min(width - 1, sx * ds + ds / 2);
            gray[dst_row + static_cast<size_t>(sx)] =
                static_cast<uint8_t>(rgbLumaInt(rgbData + src_row + static_cast<size_t>(gx) * 3ULL));
        }
    }

    // Estimate page gutter polarity from outer 2% border.
    std::array<uint32_t, 256> bhist{};
    uint32_t bcnt = 0;
    const int bx_strip = std::max(2, sw / 50);
    const int by_strip = std::max(2, sh / 50);
    for (int y = 0; y < sh; ++y) {
        const bool ye = (y < by_strip || y >= sh - by_strip);
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(sw);
        for (int x = 0; x < sw; ++x) {
            if (ye || x < bx_strip || x >= sw - bx_strip) {
                ++bhist[gray[row + static_cast<size_t>(x)]];
                ++bcnt;
            }
        }
    }
    int med_border = 255;
    uint32_t acc = 0;
    for (int i = 0; i < 256; ++i) {
        acc += bhist[static_cast<size_t>(i)];
        if (acc > bcnt / 2) {
            med_border = i;
            break;
        }
    }
    const bool black_gutter = (med_border <= 70);

    std::vector<uint8_t> is_gutter(stotal, 0);
    for (size_t i = 0; i < stotal; ++i) {
        is_gutter[i] = (black_gutter ? (gray[i] <= 42) : (gray[i] >= 228)) ? 1 : 0;
    }

    // Radon/Hough line projection: rho = x * cos(theta) + y * sin(theta)
    // Scan near-horizontal (80..100 deg, step 2 deg) and near-vertical (-10..+10 deg, step 2 deg)
    // to detect both straight and angled gutters across the page.
    std::vector<uint8_t> gutter_mask = is_gutter;
    const int max_rho = static_cast<int>(std::ceil(std::hypot(static_cast<double>(sw), static_cast<double>(sh)))) + 2;
    const int rho_bins = 2 * max_rho + 1;

    auto projectAngles = [&](int deg_start, int deg_end, int deg_step, float min_ratio) {
        std::vector<int> gutter_votes(static_cast<size_t>(rho_bins), 0);
        std::vector<int> total_votes(static_cast<size_t>(rho_bins), 0);

        for (int deg = deg_start; deg <= deg_end; deg += deg_step) {
            const float rad = static_cast<float>(deg) * kDegToRad;
            const float ct = std::cos(rad);
            const float st = std::sin(rad);

            std::fill(gutter_votes.begin(), gutter_votes.end(), 0);
            std::fill(total_votes.begin(), total_votes.end(), 0);

            for (int y = 0; y < sh; ++y) {
                const size_t row = static_cast<size_t>(y) * static_cast<size_t>(sw);
                const float y_st = static_cast<float>(y) * st;
                for (int x = 0; x < sw; ++x) {
                    const int rbin = static_cast<int>(std::lround(static_cast<float>(x) * ct + y_st)) + max_rho;
                    if (rbin >= 0 && rbin < rho_bins) {
                        ++total_votes[static_cast<size_t>(rbin)];
                        gutter_votes[static_cast<size_t>(rbin)] += is_gutter[row + static_cast<size_t>(x)];
                    }
                }
            }

            const int min_chord = std::min(sw, sh) / 2;
            for (int b = 1; b + 1 < rho_bins; ++b) {
                const int tot = total_votes[static_cast<size_t>(b)];
                if (tot < min_chord) continue;
                const float ratio = static_cast<float>(gutter_votes[static_cast<size_t>(b)]) / static_cast<float>(tot);
                if (ratio < min_ratio) continue;
                // Local peak suppression along rho
                const float r_prev = (total_votes[static_cast<size_t>(b - 1)] > 0)
                    ? (static_cast<float>(gutter_votes[static_cast<size_t>(b - 1)]) / static_cast<float>(total_votes[static_cast<size_t>(b - 1)]))
                    : 0.0f;
                const float r_next = (total_votes[static_cast<size_t>(b + 1)] > 0)
                    ? (static_cast<float>(gutter_votes[static_cast<size_t>(b + 1)]) / static_cast<float>(total_votes[static_cast<size_t>(b + 1)]))
                    : 0.0f;
                if (ratio < r_prev || ratio < r_next) continue;

                // Carve this detected gutter line (2px wide) into gutter_mask
                const float rho = static_cast<float>(b - max_rho);
                if (std::abs(st) >= std::abs(ct)) {
                    for (int x = 0; x < sw; ++x) {
                        const int y = static_cast<int>(std::lround((rho - static_cast<float>(x) * ct) / st));
                        for (int dy = -1; dy <= 1; ++dy) {
                            const int yy = y + dy;
                            if (yy >= 0 && yy < sh) {
                                gutter_mask[static_cast<size_t>(yy) * static_cast<size_t>(sw) + static_cast<size_t>(x)] = 1;
                            }
                        }
                    }
                } else {
                    for (int y = 0; y < sh; ++y) {
                        const int x = static_cast<int>(std::lround((rho - static_cast<float>(y) * st) / ct));
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int xx = x + dx;
                            if (xx >= 0 && xx < sw) {
                                gutter_mask[static_cast<size_t>(y) * static_cast<size_t>(sw) + static_cast<size_t>(xx)] = 1;
                            }
                        }
                    }
                }
            }
        }
    };

    // Detect global horizontal & vertical (including slightly angled) gutters.
    projectAngles(80, 100, 2, 0.86f);
    projectAngles(-10, 10, 2, 0.86f);

    // Flood-fill exterior gutter from page edges so white bubbles inside panels stay solid.
    std::vector<uint8_t> exterior(stotal, 0);
    std::vector<int> queue;
    queue.reserve(stotal / 4);
    auto pushExt = [&](int x, int y) {
        const int idx = y * sw + x;
        if (gutter_mask[static_cast<size_t>(idx)] != 0 && exterior[static_cast<size_t>(idx)] == 0) {
            exterior[static_cast<size_t>(idx)] = 1;
            queue.push_back(idx);
        }
    };
    for (int x = 0; x < sw; ++x) { pushExt(x, 0); pushExt(x, sh - 1); }
    for (int y = 1; y < sh - 1; ++y) { pushExt(0, y); pushExt(sw - 1, y); }

    static constexpr int kDx4[4] = {-1, 1, 0, 0};
    static constexpr int kDy4[4] = {0, 0, -1, 1};
    size_t head = 0;
    while (head < queue.size()) {
        const int cur = queue[head++];
        const int cx = cur % sw;
        const int cy = cur / sw;
        for (int k = 0; k < 4; ++k) {
            const int nx = cx + kDx4[k];
            const int ny = cy + kDy4[k];
            if (nx >= 0 && nx < sw && ny >= 0 && ny < sh) pushExt(nx, ny);
        }
    }

    // Extract 4-connected solid panel components.
    const int min_pw = std::max(8, sw / 10);
    const int min_ph = std::max(8, sh / 12);
    const int64_t min_area = static_cast<int64_t>(stotal) / 45;

    struct BoxI { int x1, y1, x2, y2; };
    std::vector<BoxI> raw_panels;
    std::vector<uint8_t> visited(stotal, 0);

    for (int y = 0; y < sh; ++y) {
        for (int x = 0; x < sw; ++x) {
            const int start = y * sw + x;
            if (exterior[static_cast<size_t>(start)] != 0 || visited[static_cast<size_t>(start)] != 0) {
                continue;
            }
            queue.clear();
            queue.push_back(start);
            visited[static_cast<size_t>(start)] = 1;
            int x1 = x, x2 = x, y1 = y, y2 = y;

            head = 0;
            while (head < queue.size()) {
                const int cur = queue[head++];
                const int cx = cur % sw;
                const int cy = cur / sw;
                if (cx < x1) x1 = cx;
                if (cx > x2) x2 = cx;
                if (cy < y1) y1 = cy;
                if (cy > y2) y2 = cy;
                for (int k = 0; k < 4; ++k) {
                    const int nx = cx + kDx4[k];
                    const int ny = cy + kDy4[k];
                    if (nx < 0 || nx >= sw || ny < 0 || ny >= sh) continue;
                    const int nidx = ny * sw + nx;
                    if (exterior[static_cast<size_t>(nidx)] == 0 && visited[static_cast<size_t>(nidx)] == 0) {
                        visited[static_cast<size_t>(nidx)] = 1;
                        queue.push_back(nidx);
                    }
                }
            }
            const int pw = x2 - x1 + 1;
            const int ph = y2 - y1 + 1;
            if (pw >= min_pw && ph >= min_ph && static_cast<int64_t>(pw) * static_cast<int64_t>(ph) >= min_area) {
                raw_panels.push_back(BoxI{x1, y1, x2 + 1, y2 + 1});
            }
        }
    }

    // Recursive local gutter-projection split on oversized panels (catches T-junction gutters).
    std::vector<BoxI> final_boxes;
    std::vector<BoxI> work = std::move(raw_panels);
    int split_guard = 0;
    while (!work.empty() && split_guard < 32) {
        ++split_guard;
        const BoxI r = work.back();
        work.pop_back();
        const int rw = r.x2 - r.x1;
        const int rh = r.y2 - r.y1;
        bool did_split = false;

        if (rh > sh / 3) {
            int run_start = -1, best_y = -1, best_run = 0;
            for (int y = r.y1 + min_ph; y < r.y2 - min_ph; ++y) {
                const size_t row = static_cast<size_t>(y) * static_cast<size_t>(sw);
                int gpx = 0;
                for (int x = r.x1; x < r.x2; ++x) {
                    gpx += is_gutter[row + static_cast<size_t>(x)];
                }
                if (gpx >= static_cast<int>(0.90f * static_cast<float>(rw))) {
                    if (run_start < 0) run_start = y;
                } else if (run_start >= 0) {
                    const int run = y - run_start;
                    if (run >= 2 && run > best_run) {
                        best_run = run;
                        best_y = run_start + run / 2;
                    }
                    run_start = -1;
                }
            }
            if (best_y > r.y1 + min_ph && best_y < r.y2 - min_ph) {
                work.push_back(BoxI{r.x1, r.y1, r.x2, best_y});
                work.push_back(BoxI{r.x1, best_y, r.x2, r.y2});
                did_split = true;
            }
        }

        if (!did_split && rw > sw / 3) {
            int run_start = -1, best_x = -1, best_run = 0;
            for (int x = r.x1 + min_pw; x < r.x2 - min_pw; ++x) {
                int gpx = 0;
                for (int y = r.y1; y < r.y2; ++y) {
                    gpx += is_gutter[static_cast<size_t>(y) * static_cast<size_t>(sw) + static_cast<size_t>(x)];
                }
                if (gpx >= static_cast<int>(0.90f * static_cast<float>(rh))) {
                    if (run_start < 0) run_start = x;
                } else if (run_start >= 0) {
                    const int run = x - run_start;
                    if (run >= 2 && run > best_run) {
                        best_run = run;
                        best_x = run_start + run / 2;
                    }
                    run_start = -1;
                }
            }
            if (best_x > r.x1 + min_pw && best_x < r.x2 - min_pw) {
                work.push_back(BoxI{r.x1, r.y1, best_x, r.y2});
                work.push_back(BoxI{best_x, r.y1, r.x2, r.y2});
                did_split = true;
            }
        }

        if (!did_split) {
            final_boxes.push_back(r);
        }
    }

    std::vector<PanelDetection> out;
    out.reserve(final_boxes.size());
    const float fw = static_cast<float>(width);
    const float fh = static_cast<float>(height);
    int next_id = 0;
    for (const auto& b : final_boxes) {
        PanelDetection p;
        p.id = next_id++;
        p.confidence = 0.85f;
        p.bbox = BBox{
            std::clamp(static_cast<float>(b.x1 * ds), 0.0f, fw),
            std::clamp(static_cast<float>(b.y1 * ds), 0.0f, fh),
            std::clamp(static_cast<float>(b.x2 * ds), 0.0f, fw),
            std::clamp(static_cast<float>(b.y2 * ds), 0.0f, fh)
        };
        if (!p.bbox.is_valid()) continue;
        p.bbox_norm = BBox{p.bbox.x1 / fw, p.bbox.y1 / fh, p.bbox.x2 / fw, p.bbox.y2 / fh};
        out.push_back(std::move(p));
    }
    return out;
}

// =============================================================================
// #2 — Exact Bitmask DP Reading Order (Minimum Feedback Arc Set, N <= 12)
// =============================================================================
std::vector<size_t> solveReadingOrderBitmaskDP(
    size_t n,
    const std::vector<float>& weights,
    const std::vector<size_t>& tie_order) {
    if (n == 0 || n > 12 || weights.size() < n * n) {
        return {};
    }
    if (n == 1) {
        return {0};
    }

    const uint32_t full_mask = (1u << n) - 1u;
    const size_t num_states = (size_t{1} << n);
    constexpr float kNegInf = -1e30f;

    std::vector<float> dp(num_states, kNegInf);
    std::vector<int8_t> parent(num_states, -1);
    dp[0] = 0.0f;

    for (uint32_t mask = 1; mask <= full_mask; ++mask) {
        float best_val = kNegInf;
        int best_u = -1;

        for (size_t u = 0; u < n; ++u) {
            const uint32_t bit = 1u << u;
            if ((mask & bit) == 0) continue;
            const uint32_t prev = mask ^ bit;
            if (dp[prev] <= kNegInf * 0.5f) continue;

            // Placing `u` at the end of `prev` satisfies all directed edges (v -> u) for v in prev.
            float gain = 0.0f;
            for (size_t v = 0; v < n; ++v) {
                if ((prev & (1u << v)) != 0) {
                    gain += weights[v * n + u];
                }
            }
            const float cand = dp[prev] + gain;
            const bool better = (cand > best_val + 1e-5f);
            const bool tie = (std::abs(cand - best_val) <= 1e-5f) &&
                             (best_u >= 0) &&
                             (tie_order.size() == n
                                  ? (tie_order[u] > tie_order[static_cast<size_t>(best_u)])
                                  : (u > static_cast<size_t>(best_u)));
            // On tie for the LAST element `u`, prefer the one with LARGER tie_order (so earlier items come first).
            if (best_u < 0 || better || tie) {
                best_val = cand;
                best_u = static_cast<int>(u);
            }
        }

        dp[mask] = best_val;
        parent[mask] = static_cast<int8_t>(best_u);
    }

    std::vector<size_t> order(n, 0);
    uint32_t cur = full_mask;
    for (size_t pos = n; pos-- > 0;) {
        const int u = static_cast<int>(parent[cur]);
        if (u < 0) {
            std::iota(order.begin(), order.end(), size_t{0});
            return order;
        }
        order[pos] = static_cast<size_t>(u);
        cur ^= (1u << static_cast<uint32_t>(u));
    }
    return order;
}

// =============================================================================
// #8 — 1D DP Minimum-Energy Seam Webtoon Tiling
// =============================================================================
std::vector<int> computeWebtoonSeamSplits(
    const uint8_t* rgbData,
    int width,
    int height,
    int target_h,
    int min_h,
    int max_h,
    float beta,
    float gamma) {
    if (!rgbData || width <= 0 || height <= 0) {
        return {0, std::max(0, height)};
    }
    const int safe_target = std::max(256, target_h);
    const int safe_min = std::clamp(min_h, 128, safe_target);
    const int safe_max = std::max(safe_target, max_h);

    if (height <= safe_max) {
        return {0, height};
    }

    // Downsample horizontally to <= 256 cols and vertically by `sy` so DP has <= 1200 states.
    const int sx = std::max(1, width / 256);
    const int sy = std::max(2, (height + 1199) / 1200);
    const int cols = (width + sx - 1) / sx;
    const int rows = (height + sy - 1) / sy;

    std::vector<float> prev_luma(static_cast<size_t>(cols), 0.0f);
    std::vector<float> cur_luma(static_cast<size_t>(cols), 0.0f);
    std::vector<float> raw_energy(static_cast<size_t>(rows), 0.0f);

    for (int r = 0; r < rows; ++r) {
        const int y = std::min(height - 1, r * sy);
        const size_t row_off = static_cast<size_t>(y) * static_cast<size_t>(width) * 3ULL;

        double sum_l = 0.0;
        double sum_l2 = 0.0;
        double sum_grad = 0.0;

        for (int c = 0; c < cols; ++c) {
            const int x = std::min(width - 1, c * sx);
            const float lum = static_cast<float>(rgbLumaInt(rgbData + row_off + static_cast<size_t>(x) * 3ULL)) / 255.0f;
            cur_luma[static_cast<size_t>(c)] = lum;
            sum_l += static_cast<double>(lum);
            sum_l2 += static_cast<double>(lum) * static_cast<double>(lum);

            const float gx = (c > 0) ? std::abs(lum - cur_luma[static_cast<size_t>(c - 1)]) : 0.0f;
            const float gy = (r > 0) ? std::abs(lum - prev_luma[static_cast<size_t>(c)]) : 0.0f;
            sum_grad += static_cast<double>(gx + gy);
        }

        const double mean_l = sum_l / static_cast<double>(cols);
        const double var_l = std::max(0.0, (sum_l2 / static_cast<double>(cols)) - mean_l * mean_l);
        const float sigma_row = static_cast<float>(std::sqrt(var_l));
        const float mean_grad = static_cast<float>(sum_grad / static_cast<double>(cols));
        raw_energy[static_cast<size_t>(r)] = mean_grad + beta * sigma_row;

        std::swap(prev_luma, cur_luma);
    }

    // Smooth energy over 3 adjacent rows to avoid single-row noise seams.
    std::vector<float> energy(static_cast<size_t>(rows), 0.0f);
    for (int r = 0; r < rows; ++r) {
        const int r0 = std::max(0, r - 1);
        const int r1 = std::min(rows - 1, r + 1);
        float s = 0.0f;
        for (int k = r0; k <= r1; ++k) s += raw_energy[static_cast<size_t>(k)];
        energy[static_cast<size_t>(r)] = s / static_cast<float>(r1 - r0 + 1);
    }

    // 1D DP over downscaled row indices [0..rows].
    const int min_step = std::max(1, safe_min / sy);
    const int max_step = std::max(min_step + 1, safe_max / sy);
    const float target_step = static_cast<float>(safe_target) / static_cast<float>(sy);

    constexpr float kInf = 1e30f;
    std::vector<float> dp(static_cast<size_t>(rows) + 1, kInf);
    std::vector<int> prev_row(static_cast<size_t>(rows) + 1, -1);
    dp[0] = 0.0f;

    for (int r = min_step; r <= rows; ++r) {
        const bool is_end = (r == rows);
        const int lo = std::max(0, r - max_step);
        const int hi = is_end ? std::max(0, r - std::max(1, min_step / 2)) : (r - min_step);
        const float seam_e = is_end ? 0.0f : energy[static_cast<size_t>(r)];

        float best_cost = kInf;
        int best_p = -1;
        for (int p = lo; p <= hi; ++p) {
            if (dp[static_cast<size_t>(p)] >= kInf * 0.5f) continue;
            const float diff = (static_cast<float>(r - p) - target_step) / target_step;
            const float len_pen = is_end ? (0.25f * gamma * diff * diff) : (gamma * diff * diff);
            const float cost = dp[static_cast<size_t>(p)] + seam_e + len_pen;
            if (cost < best_cost) {
                best_cost = cost;
                best_p = p;
            }
        }
        dp[static_cast<size_t>(r)] = best_cost;
        prev_row[static_cast<size_t>(r)] = best_p;
    }

    if (prev_row[static_cast<size_t>(rows)] < 0) {
        // Fallback uniform splits
        std::vector<int> uniform{0};
        for (int y = safe_target; y < height; y += safe_target) {
            uniform.push_back(y);
        }
        uniform.push_back(height);
        return uniform;
    }

    std::vector<int> splits;
    for (int cur = rows; cur > 0; cur = prev_row[static_cast<size_t>(cur)]) {
        splits.push_back(std::min(height, cur * sy));
        if (prev_row[static_cast<size_t>(cur)] < 0) break;
    }
    splits.push_back(0);
    std::reverse(splits.begin(), splits.end());
    splits.back() = height;
    return splits;
}

// =============================================================================
// #15 — Discrete Curvature + Ray-Casting Speaker Clustering
// =============================================================================
void clusterSpeakersByTailRays(
    std::vector<BubbleDetection>& detections,
    const std::vector<PanelDetection>& panels) {
    const size_t n = detections.size();
    if (n == 0) return;

    for (auto& d : detections) {
        d.speaker_cluster_id = -1;
    }

    std::unordered_map<int, BBox> panel_boxes;
    for (const auto& p : panels) {
        panel_boxes[p.id] = p.bbox;
    }

    struct TailRay {
        bool valid = false;
        Point2f origin;
        Point2f dir;  // Unit outward bisector
    };

    std::vector<TailRay> rays(n);
    constexpr float kMaxTailAngleRad = 45.0f * kDegToRad;

    for (size_t i = 0; i < n; ++i) {
        const auto& d = detections[i];
        if (!is_dialogue_class(d.class_id)) continue;

        const auto& poly = !d.room_polygon.empty() ? d.room_polygon : d.mask.polygon;
        const size_t m = poly.size();
        if (m < 4) continue;

        float cx = 0.0f, cy = 0.0f;
        for (const auto& pt : poly) {
            cx += pt.x;
            cy += pt.y;
        }
        cx /= static_cast<float>(m);
        cy /= static_cast<float>(m);

        float best_angle = kMaxTailAngleRad;
        TailRay best_ray;

        for (size_t k = 0; k < m; ++k) {
            const Point2f& prev = poly[(k + m - 1) % m];
            const Point2f& cur = poly[k];
            const Point2f& next = poly[(k + 1) % m];

            const float ux = prev.x - cur.x;
            const float uy = prev.y - cur.y;
            const float wx = next.x - cur.x;
            const float wy = next.y - cur.y;
            const float lu = std::hypot(ux, uy);
            const float lw = std::hypot(wx, wy);
            if (lu < 3.0f || lw < 3.0f) continue;

            const float nux = ux / lu;
            const float nuy = uy / lu;
            const float nwx = wx / lw;
            const float nwy = wy / lw;

            const float dot = std::clamp(nux * nwx + nuy * nwy, -1.0f, 1.0f);
            const float theta = std::acos(dot);
            if (theta >= best_angle) continue;

            // Outward bisector points opposite to (nu + nw).
            const float bx = -(nux + nwx);
            const float by = -(nuy + nwy);
            const float lb = std::hypot(bx, by);
            if (lb < 1e-4f) continue;

            const float dir_x = bx / lb;
            const float dir_y = by / lb;

            // Verify tip protrudes outward away from bubble centroid.
            const float vx = cur.x - cx;
            const float vy = cur.y - cy;
            if ((dir_x * vx + dir_y * vy) <= 0.0f) continue;

            const float d_cur = std::hypot(vx, vy);
            const float d_prev = std::hypot(prev.x - cx, prev.y - cy);
            const float d_next = std::hypot(next.x - cx, next.y - cy);
            if (d_cur <= 0.5f * (d_prev + d_next)) continue;

            best_angle = theta;
            best_ray = TailRay{true, cur, Point2f{dir_x, dir_y}};
        }

        rays[i] = best_ray;
    }

    // Group dialogue detections by panel_id and cluster via Union-Find.
    DisjointSet dsu(n);

    for (size_t i = 0; i < n; ++i) {
        if (!is_dialogue_class(detections[i].class_id)) continue;
        const int pid = detections[i].panel_id;

        float panel_diag = 400.0f;
        BBox panel_box{0.0f, 0.0f, 1e6f, 1e6f};
        if (auto it = panel_boxes.find(pid); it != panel_boxes.end()) {
            panel_box = it->second;
            panel_diag = std::max(80.0f, std::hypot(panel_box.width(), panel_box.height()));
        }
        const float prox_radius = std::clamp(0.18f * panel_diag, 40.0f, 220.0f);
        const float max_ray_t = 0.85f * panel_diag;

        for (size_t j = i + 1; j < n; ++j) {
            if (!is_dialogue_class(detections[j].class_id)) continue;
            if (detections[j].panel_id != pid) continue;

            // 1. Conjoined-neck check: nearly touching/overlapping bboxes in same panel.
            const BBox& a = detections[i].bbox;
            const BBox& b = detections[j].bbox;
            const float gap_x = std::max(0.0f, std::max(a.x1, b.x1) - std::min(a.x2, b.x2));
            const float gap_y = std::max(0.0f, std::max(a.y1, b.y1) - std::min(a.y2, b.y2));
            const float gap = std::hypot(gap_x, gap_y);
            const float min_diag = std::min(std::hypot(a.width(), a.height()),
                                            std::hypot(b.width(), b.height()));
            if (gap <= 0.10f * min_diag) {
                dsu.unite(static_cast<int>(i), static_cast<int>(j));
                continue;
            }

            // 2. Ray-ray convergence inside the same panel.
            if (rays[i].valid && rays[j].valid) {
                const Point2f& o1 = rays[i].origin;
                const Point2f& d1 = rays[i].dir;
                const Point2f& o2 = rays[j].origin;
                const Point2f& d2 = rays[j].dir;

                const float rx = o2.x - o1.x;
                const float ry = o2.y - o1.y;
                const float det = d1.x * (-d2.y) - d1.y * (-d2.x);  // d1 x (-d2)
                if (std::abs(det) > 0.08f) {
                    const float t1 = (rx * (-d2.y) - ry * (-d2.x)) / det;
                    const float t2 = (d1.x * ry - d1.y * rx) / det;
                    if (t1 > 0.0f && t2 > 0.0f && t1 <= max_ray_t && t2 <= max_ray_t) {
                        const float px = 0.5f * ((o1.x + t1 * d1.x) + (o2.x + t2 * d2.x));
                        const float py = 0.5f * ((o1.y + t1 * d1.y) + (o2.y + t2 * d2.y));
                        const float margin = 0.10f * panel_diag;
                        if (px >= panel_box.x1 - margin && px <= panel_box.x2 + margin &&
                            py >= panel_box.y1 - margin && py <= panel_box.y2 + margin) {
                            const float dist_rays = std::hypot(
                                (o1.x + t1 * d1.x) - (o2.x + t2 * d2.x),
                                (o1.y + t1 * d1.y) - (o2.y + t2 * d2.y));
                            if (dist_rays <= prox_radius) {
                                dsu.unite(static_cast<int>(i), static_cast<int>(j));
                            }
                        }
                    }
                } else {
                    // Nearly parallel converging tails close together
                    const float origin_dist = std::hypot(rx, ry);
                    const float dir_dot = d1.x * d2.x + d1.y * d2.y;
                    if (origin_dist <= prox_radius && dir_dot > 0.7f) {
                        dsu.unite(static_cast<int>(i), static_cast<int>(j));
                    }
                }
            }
        }
    }

    // Assign compact cluster IDs 0, 1, 2, ... in detection order.
    std::unordered_map<int, int> root_to_cluster;
    int next_cluster = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!is_dialogue_class(detections[i].class_id)) continue;
        const int root = dsu.find(static_cast<int>(i));
        auto it = root_to_cluster.find(root);
        if (it == root_to_cluster.end()) {
            root_to_cluster[root] = next_cluster;
            detections[i].speaker_cluster_id = next_cluster++;
        } else {
            detections[i].speaker_cluster_id = it->second;
        }
    }
}

// =============================================================================
// #16 — Anisotropic DBSCAN + Bresenham Line-of-Sight Barrier
// =============================================================================
bool hasBresenhamInkBarrier(
    const uint8_t* rgbData,
    int width,
    int height,
    int x0,
    int y0,
    int x1,
    int y1) noexcept {
    if (!rgbData || width <= 0 || height <= 0) {
        return false;
    }

    x0 = std::clamp(x0, 0, width - 1);
    y0 = std::clamp(y0, 0, height - 1);
    x1 = std::clamp(x1, 0, width - 1);
    y1 = std::clamp(y1, 0, height - 1);

    const int dx = std::abs(x1 - x0);
    const int dy = std::abs(y1 - y0);
    const int total_steps = std::max(dx, dy);
    if (total_steps < 6) {
        return false;
    }

    const int sx = (x0 < x1) ? 1 : -1;
    const int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    // Perpendicular unit step for checking whether a dark hit is a continuous border stroke
    // rather than an isolated glyph stroke.
    const float len = std::max(1.0f, std::hypot(static_cast<float>(x1 - x0), static_cast<float>(y1 - y0)));
    const float perp_x = -static_cast<float>(y1 - y0) / len;
    const float perp_y = static_cast<float>(x1 - x0) / len;

    const int skip_margin = std::max(2, total_steps / 8);
    int cx = x0;
    int cy = y0;
    int dark_run = 0;

    for (int step = 0; step <= total_steps; ++step) {
        if (step >= skip_margin && step <= total_steps - skip_margin) {
            const int lum = rgbLumaAt(rgbData, width, height, cx, cy);
            if (lum < 65) {
                ++dark_run;
                if (dark_run >= 2) {
                    // Check perpendicular continuity (+-4 px): a panel/bubble border continues along the normal.
                    int perp_dark = 0;
                    for (int k = -4; k <= 4; ++k) {
                        if (k == 0) continue;
                        const int px = static_cast<int>(std::lround(static_cast<float>(cx) + static_cast<float>(k) * perp_x));
                        const int py = static_cast<int>(std::lround(static_cast<float>(cy) + static_cast<float>(k) * perp_y));
                        if (rgbLumaAt(rgbData, width, height, px, py) < 75) {
                            ++perp_dark;
                        }
                    }
                    if (perp_dark >= 6) {
                        return true;
                    }
                }
                if (dark_run >= 6) {
                    return true;
                }
            } else {
                dark_run = 0;
            }
        }

        if (cx == x1 && cy == y1) break;
        const int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            cx += sx;
        }
        if (e2 < dx) {
            err += dx;
            cy += sy;
        }
    }

    return false;
}

void mergeFragmentedTextRegionsDBSCAN(
    std::vector<BubbleDetection>& detections,
    const uint8_t* rgbData,
    int width,
    int height,
    ReadingDirection direction) {
    const size_t n = detections.size();
    if (n < 2) return;

    auto isMergeCandidate = [](ClassId id) noexcept {
        return id == ClassId::BorderlessDialogue || id == ClassId::TextRegion;
    };

    std::vector<size_t> cand_idx;
    cand_idx.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (isMergeCandidate(detections[i].class_id) && detections[i].bbox.is_valid()) {
            cand_idx.push_back(i);
        }
    }
    const size_t m = cand_idx.size();
    if (m < 2) return;

    // Anisotropic scaling factors: vertical manga columns stretch along Y; horizontal along X.
    const float alpha_x = (direction == ReadingDirection::RTL) ? 1.25f : 1.65f;
    const float alpha_y = (direction == ReadingDirection::RTL) ? 1.65f : 1.25f;

    auto fontScale = [](const BubbleDetection& d) noexcept -> float {
        if (d.estimated_font_size.has_value() && *d.estimated_font_size >= 8.0f) {
            return *d.estimated_font_size;
        }
        return std::max(10.0f, std::min(d.bbox.width(), d.bbox.height()));
    };

    DisjointSet dsu(m);
    for (size_t a = 0; a < m; ++a) {
        const auto& da = detections[cand_idx[a]];
        const float ha = fontScale(da);
        for (size_t b = a + 1; b < m; ++b) {
            const auto& db = detections[cand_idx[b]];
            const float hb = fontScale(db);

            const float ratio = ha / std::max(1.0f, hb);
            if (ratio < 0.75f || ratio > 1.35f) continue;

            const float hf = 0.5f * (ha + hb);
            const float gap_x = std::max(0.0f, std::max(da.bbox.x1, db.bbox.x1) - std::min(da.bbox.x2, db.bbox.x2));
            const float gap_y = std::max(0.0f, std::max(da.bbox.y1, db.bbox.y1) - std::min(da.bbox.y2, db.bbox.y2));

            const float nx = gap_x / (alpha_x * hf);
            const float ny = gap_y / (alpha_y * hf);
            const float d_aniso = std::hypot(nx, ny);
            if (d_aniso > 1.5f) continue;

            if (rgbData && width > 0 && height > 0) {
                if (hasBresenhamInkBarrier(
                        rgbData, width, height,
                        static_cast<int>(std::lround(da.bbox.cx())),
                        static_cast<int>(std::lround(da.bbox.cy())),
                        static_cast<int>(std::lround(db.bbox.cx())),
                        static_cast<int>(std::lround(db.bbox.cy())))) {
                    continue;
                }
            }

            dsu.unite(static_cast<int>(a), static_cast<int>(b));
        }
    }

    // Group candidate indices by cluster root.
    std::unordered_map<int, std::vector<size_t>> clusters;
    for (size_t i = 0; i < m; ++i) {
        clusters[dsu.find(static_cast<int>(i))].push_back(cand_idx[i]);
    }

    bool any_merged = false;
    for (const auto& [root, members] : clusters) {
        if (members.size() > 1) {
            any_merged = true;
            break;
        }
    }
    if (!any_merged) return;

    std::vector<uint8_t> consumed(n, 0);
    std::vector<BubbleDetection> merged_list;
    merged_list.reserve(n);

    for (const auto& [root, members] : clusters) {
        if (members.size() == 1) continue;

        BubbleDetection acc = detections[members[0]];
        consumed[members[0]] = 1;
        for (size_t k = 1; k < members.size(); ++k) {
            const auto& other = detections[members[k]];
            consumed[members[k]] = 1;
            acc.bbox.x1 = std::min(acc.bbox.x1, other.bbox.x1);
            acc.bbox.y1 = std::min(acc.bbox.y1, other.bbox.y1);
            acc.bbox.x2 = std::max(acc.bbox.x2, other.bbox.x2);
            acc.bbox.y2 = std::max(acc.bbox.y2, other.bbox.y2);
            acc.confidence = std::max(acc.confidence, other.confidence);
            if (other.class_id == ClassId::BorderlessDialogue) {
                acc.class_id = ClassId::BorderlessDialogue;
                acc.class_name = class_id_to_string(acc.class_id);
            }
        }
        if (width > 0 && height > 0) {
            acc.bbox_norm = BBox{
                acc.bbox.x1 / static_cast<float>(width),
                acc.bbox.y1 / static_cast<float>(height),
                acc.bbox.x2 / static_cast<float>(width),
                acc.bbox.y2 / static_cast<float>(height)
            };
        }
        // Clear stale single-fragment mask so computeSafeBBox uses the merged bbox cleanly.
        acc.mask = InstanceMask{};
        merged_list.push_back(std::move(acc));
    }

    std::vector<BubbleDetection> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (!consumed[i]) {
            out.push_back(std::move(detections[i]));
        }
    }
    for (auto& d : merged_list) {
        out.push_back(std::move(d));
    }
    detections = std::move(out);
}

// =============================================================================
// #18 — Opt-In RANSAC + 3x3 Homography Deskew & Lightweight TPS Spine Dewarp
// =============================================================================
DeskewResult deskewAndDewarpScan(
    const uint8_t* rgbData,
    int width,
    int height) {
    DeskewResult res;
    if (!rgbData || width < 128 || height < 128) {
        return res;
    }

    // 1. Downscale to <= 400px long side and collect strong near-horizontal / near-vertical edge points.
    const int ds = std::max(1, (std::max(width, height) + 399) / 400);
    const int sw = width / ds;
    const int sh = height / ds;
    if (sw < 32 || sh < 32) return res;

    std::vector<uint8_t> gray(static_cast<size_t>(sw) * static_cast<size_t>(sh));
    for (int y = 0; y < sh; ++y) {
        const int gy = std::min(height - 1, y * ds);
        const size_t srow = static_cast<size_t>(gy) * static_cast<size_t>(width) * 3ULL;
        const size_t drow = static_cast<size_t>(y) * static_cast<size_t>(sw);
        for (int x = 0; x < sw; ++x) {
            const int gx = std::min(width - 1, x * ds);
            gray[drow + static_cast<size_t>(x)] =
                static_cast<uint8_t>(rgbLumaInt(rgbData + srow + static_cast<size_t>(gx) * 3ULL));
        }
    }

    struct EdgePt { float x; float y; };
    std::vector<EdgePt> h_edges;  // Strong vertical gradient -> horizontal panel borders
    h_edges.reserve(static_cast<size_t>(sw) * 8ULL);

    for (int y = 2; y < sh - 2; ++y) {
        const size_t r0 = static_cast<size_t>(y - 1) * static_cast<size_t>(sw);
        const size_t r1 = static_cast<size_t>(y) * static_cast<size_t>(sw);
        const size_t r2 = static_cast<size_t>(y + 1) * static_cast<size_t>(sw);
        for (int x = 2; x < sw - 2; ++x) {
            const int gx = std::abs(static_cast<int>(gray[r1 + static_cast<size_t>(x + 1)]) -
                                    static_cast<int>(gray[r1 + static_cast<size_t>(x - 1)]));
            const int gy = std::abs(static_cast<int>(gray[r2 + static_cast<size_t>(x)]) -
                                    static_cast<int>(gray[r0 + static_cast<size_t>(x)]));
            if (gy >= 70 && gy > gx * 2) {
                h_edges.push_back(EdgePt{static_cast<float>(x), static_cast<float>(y)});
            }
        }
    }
    if (h_edges.size() < 40) {
        return res;
    }

    // 2. Deterministic RANSAC over horizontal border edge pairs to estimate dominant page skew angle.
    //    Sample pairs with dx >= sw / 6 and |angle| <= 7 deg.
    uint32_t rng = 0x9E3779B9u ^ static_cast<uint32_t>(width * 31 + height);
    auto nextRand = [&rng]() noexcept -> uint32_t {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    };

    const size_t ne = h_edges.size();
    constexpr int kRansacIters = 180;
    constexpr float kInlierTolPx = 1.6f;

    int best_inliers = 0;
    float best_angle_deg = 0.0f;
    std::vector<float> candidate_angles;
    candidate_angles.reserve(kRansacIters);

    for (int iter = 0; iter < kRansacIters; ++iter) {
        const EdgePt& p1 = h_edges[nextRand() % ne];
        const EdgePt& p2 = h_edges[nextRand() % ne];
        const float dx = p2.x - p1.x;
        const float dy = p2.y - p1.y;
        if (std::abs(dx) < static_cast<float>(sw) * 0.16f) continue;

        const float ang = std::atan2(dy, dx) * kRadToDeg;
        const float norm_ang = (ang > 90.0f) ? (ang - 180.0f) : ((ang < -90.0f) ? (ang + 180.0f) : ang);
        if (std::abs(norm_ang) > 7.5f) continue;

        // Line equation: a*x + b*y + c = 0 (unit normal)
        const float len = std::hypot(dx, dy);
        const float a = -dy / len;
        const float b = dx / len;
        const float c = -(a * p1.x + b * p1.y);

        int inliers = 0;
        const size_t step = std::max<size_t>(1, ne / 512);
        for (size_t k = 0; k < ne; k += step) {
            const float dist = std::abs(a * h_edges[k].x + b * h_edges[k].y + c);
            if (dist <= kInlierTolPx) {
                ++inliers;
            }
        }
        if (inliers > best_inliers) {
            best_inliers = inliers;
            best_angle_deg = norm_ang;
        }
    }

    // Require sufficient consensus and clamp valid correction to [0.8 deg, 6.0 deg].
    if (best_inliers < 18 || std::abs(best_angle_deg) < 0.8f || std::abs(best_angle_deg) > 6.0f) {
        return res;
    }

    // 3. Bilinear 3x3 Homography warp (inverse rotation around page center by -best_angle_deg).
    const float rad = best_angle_deg * kDegToRad;
    const float c = std::cos(rad);
    const float s = std::sin(rad);
    const float cx = 0.5f * static_cast<float>(width - 1);
    const float cy = 0.5f * static_cast<float>(height - 1);

    // Inverse homography H^{-1}: maps dst (x, y) -> src (sx, sy)
    // H^{-1} = T(cx,cy) * R(+rad) * T(-cx,-cy)
    const float h00 = c,  h01 = -s, h02 = cx - c * cx + s * cy;
    const float h10 = s,  h11 =  c, h12 = cy - s * cx - c * cy;

    // 4. Check top & bottom panel border curvature for optional lightweight TPS spine dewarp.
    //    Sample top-most and bottom-most horizontal border y across 6 horizontal bands.
    constexpr int kBands = 6;
    std::array<float, kBands> top_y{};
    std::array<float, kBands> bot_y{};
    std::array<bool, kBands> top_ok{};
    std::array<bool, kBands> bot_ok{};
    top_y.fill(static_cast<float>(sh));
    bot_y.fill(0.0f);

    for (const auto& pt : h_edges) {
        // Deskewed downscaled coordinates
        const float dx = pt.x - 0.5f * static_cast<float>(sw);
        const float dy = pt.y - 0.5f * static_cast<float>(sh);
        const float rx = c * dx + s * dy + 0.5f * static_cast<float>(sw);
        const float ry = -s * dx + c * dy + 0.5f * static_cast<float>(sh);
        const int band = std::clamp(static_cast<int>(rx * static_cast<float>(kBands) / static_cast<float>(sw)), 0, kBands - 1);
        if (ry > static_cast<float>(sh) * 0.02f && ry < static_cast<float>(sh) * 0.25f) {
            if (ry < top_y[static_cast<size_t>(band)]) {
                top_y[static_cast<size_t>(band)] = ry;
                top_ok[static_cast<size_t>(band)] = true;
            }
        } else if (ry > static_cast<float>(sh) * 0.75f && ry < static_cast<float>(sh) * 0.98f) {
            if (ry > bot_y[static_cast<size_t>(band)]) {
                bot_y[static_cast<size_t>(band)] = ry;
                bot_ok[static_cast<size_t>(band)] = true;
            }
        }
    }

    auto measureCurvature = [&](const std::array<float, kBands>& ys, const std::array<bool, kBands>& ok, float& out_med) -> float {
        std::vector<float> vals;
        for (int b = 0; b < kBands; ++b) {
            if (ok[static_cast<size_t>(b)]) vals.push_back(ys[static_cast<size_t>(b)]);
        }
        if (vals.size() < 4) return 0.0f;
        std::sort(vals.begin(), vals.end());
        out_med = vals[vals.size() / 2];
        float max_dev = 0.0f;
        for (float v : vals) max_dev = std::max(max_dev, std::abs(v - out_med));
        return max_dev / static_cast<float>(sh);
    };

    float top_med = 0.0f, bot_med = 0.0f;
    const float top_curv = measureCurvature(top_y, top_ok, top_med);
    const float bot_curv = measureCurvature(bot_y, bot_ok, bot_med);
    const bool need_tps = (std::max(top_curv, bot_curv) > 0.015f);

    // Optional coarse 32x32 TPS vertical displacement field dy_grid (evaluated from U(r)=r^2 ln r).
    constexpr int kGrid = 32;
    std::vector<float> tps_dy_grid;
    bool tps_ok = false;

    if (need_tps) {
        std::vector<Point2f> ctrl_pts;
        std::vector<double> ctrl_dy;
        for (int b = 0; b < kBands; ++b) {
            const float bx = (static_cast<float>(b) + 0.5f) / static_cast<float>(kBands) * static_cast<float>(width);
            if (top_ok[static_cast<size_t>(b)]) {
                ctrl_pts.push_back(Point2f{bx, top_med * static_cast<float>(ds)});
                ctrl_dy.push_back(static_cast<double>(top_y[static_cast<size_t>(b)] - top_med) * static_cast<double>(ds));
            }
            if (bot_ok[static_cast<size_t>(b)]) {
                ctrl_pts.push_back(Point2f{bx, bot_med * static_cast<float>(ds)});
                ctrl_dy.push_back(static_cast<double>(bot_y[static_cast<size_t>(b)] - bot_med) * static_cast<double>(ds));
            }
        }
        const int K = static_cast<int>(ctrl_pts.size());
        if (K >= 6) {
            const int M = K + 3;
            std::vector<double> L(static_cast<size_t>(M) * static_cast<size_t>(M), 0.0);
            std::vector<double> rhs(static_cast<size_t>(M), 0.0);
            const double reg = 1e-2 * static_cast<double>(width) * static_cast<double>(width);

            for (int i = 0; i < K; ++i) {
                rhs[static_cast<size_t>(i)] = ctrl_dy[static_cast<size_t>(i)];
                for (int j = 0; j < K; ++j) {
                    if (i == j) {
                        L[static_cast<size_t>(i) * static_cast<size_t>(M) + static_cast<size_t>(j)] = reg;
                    } else {
                        const double dx = static_cast<double>(ctrl_pts[static_cast<size_t>(i)].x - ctrl_pts[static_cast<size_t>(j)].x);
                        const double dy = static_cast<double>(ctrl_pts[static_cast<size_t>(i)].y - ctrl_pts[static_cast<size_t>(j)].y);
                        L[static_cast<size_t>(i) * static_cast<size_t>(M) + static_cast<size_t>(j)] = tpsKernel(dx * dx + dy * dy);
                    }
                }
                L[static_cast<size_t>(i) * static_cast<size_t>(M) + static_cast<size_t>(K + 0)] = 1.0;
                L[static_cast<size_t>(i) * static_cast<size_t>(M) + static_cast<size_t>(K + 1)] = ctrl_pts[static_cast<size_t>(i)].x;
                L[static_cast<size_t>(i) * static_cast<size_t>(M) + static_cast<size_t>(K + 2)] = ctrl_pts[static_cast<size_t>(i)].y;
                L[static_cast<size_t>(K + 0) * static_cast<size_t>(M) + static_cast<size_t>(i)] = 1.0;
                L[static_cast<size_t>(K + 1) * static_cast<size_t>(M) + static_cast<size_t>(i)] = ctrl_pts[static_cast<size_t>(i)].x;
                L[static_cast<size_t>(K + 2) * static_cast<size_t>(M) + static_cast<size_t>(i)] = ctrl_pts[static_cast<size_t>(i)].y;
            }

            std::vector<double> sol;
            if (solveLinearSystem(L, rhs, M, sol)) {
                tps_dy_grid.assign(static_cast<size_t>(kGrid + 1) * static_cast<size_t>(kGrid + 1), 0.0f);
                const float max_shift = 0.04f * static_cast<float>(height);
                for (int gy = 0; gy <= kGrid; ++gy) {
                    const double py = static_cast<double>(gy) / static_cast<double>(kGrid) * static_cast<double>(height - 1);
                    for (int gx = 0; gx <= kGrid; ++gx) {
                        const double px = static_cast<double>(gx) / static_cast<double>(kGrid) * static_cast<double>(width - 1);
                        double val = sol[static_cast<size_t>(K + 0)] +
                                     sol[static_cast<size_t>(K + 1)] * px +
                                     sol[static_cast<size_t>(K + 2)] * py;
                        for (int i = 0; i < K; ++i) {
                            const double dx = px - static_cast<double>(ctrl_pts[static_cast<size_t>(i)].x);
                            const double dy = py - static_cast<double>(ctrl_pts[static_cast<size_t>(i)].y);
                            val += sol[static_cast<size_t>(i)] * tpsKernel(dx * dx + dy * dy);
                        }
                        tps_dy_grid[static_cast<size_t>(gy) * static_cast<size_t>(kGrid + 1) + static_cast<size_t>(gx)] =
                            std::clamp(static_cast<float>(val), -max_shift, max_shift);
                    }
                }
                tps_ok = true;
            }
        }
    }

    // 5. Render bilinearly warped RGB image.
    res.warped_rgb.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3ULL, 255);
    const float inv_w = static_cast<float>(kGrid) / static_cast<float>(std::max(1, width - 1));
    const float inv_h = static_cast<float>(kGrid) / static_cast<float>(std::max(1, height - 1));

    for (int y = 0; y < height; ++y) {
        const size_t dst_row = static_cast<size_t>(y) * static_cast<size_t>(width) * 3ULL;
        const float fy = static_cast<float>(y);
        for (int x = 0; x < width; ++x) {
            const float fx = static_cast<float>(x);
            float tps_dy = 0.0f;
            if (tps_ok) {
                const float gx_f = fx * inv_w;
                const float gy_f = fy * inv_h;
                const int gx0 = std::clamp(static_cast<int>(gx_f), 0, kGrid - 1);
                const int gy0 = std::clamp(static_cast<int>(gy_f), 0, kGrid - 1);
                const float tx = gx_f - static_cast<float>(gx0);
                const float ty = gy_f - static_cast<float>(gy0);
                const size_t r0 = static_cast<size_t>(gy0) * static_cast<size_t>(kGrid + 1);
                const size_t r1 = static_cast<size_t>(gy0 + 1) * static_cast<size_t>(kGrid + 1);
                const float v00 = tps_dy_grid[r0 + static_cast<size_t>(gx0)];
                const float v01 = tps_dy_grid[r0 + static_cast<size_t>(gx0 + 1)];
                const float v10 = tps_dy_grid[r1 + static_cast<size_t>(gx0)];
                const float v11 = tps_dy_grid[r1 + static_cast<size_t>(gx0 + 1)];
                tps_dy = (1.0f - ty) * ((1.0f - tx) * v00 + tx * v01) +
                         ty * ((1.0f - tx) * v10 + tx * v11);
            }

            const float sx = h00 * fx + h01 * (fy + tps_dy) + h02;
            const float sy = h10 * fx + h11 * (fy + tps_dy) + h12;
            const int x0 = static_cast<int>(std::floor(sx));
            const int y0 = static_cast<int>(std::floor(sy));

            if (x0 < 0 || y0 < 0 || x0 + 1 >= width || y0 + 1 >= height) {
                continue;  // Leave white border padding (255)
            }

            const float wx1 = sx - static_cast<float>(x0);
            const float wy1 = sy - static_cast<float>(y0);
            const float wx0 = 1.0f - wx1;
            const float wy0 = 1.0f - wy1;

            const size_t sr0 = static_cast<size_t>(y0) * static_cast<size_t>(width);
            const size_t sr1 = static_cast<size_t>(y0 + 1) * static_cast<size_t>(width);
            const size_t i00 = (sr0 + static_cast<size_t>(x0)) * 3ULL;
            const size_t i01 = (sr0 + static_cast<size_t>(x0 + 1)) * 3ULL;
            const size_t i10 = (sr1 + static_cast<size_t>(x0)) * 3ULL;
            const size_t i11 = (sr1 + static_cast<size_t>(x0 + 1)) * 3ULL;

            const float w00 = wy0 * wx0;
            const float w01 = wy0 * wx1;
            const float w10 = wy1 * wx0;
            const float w11 = wy1 * wx1;

            const size_t d_off = dst_row + static_cast<size_t>(x) * 3ULL;
            for (int ch = 0; ch < 3; ++ch) {
                const float v = w00 * rgbData[i00 + ch] + w01 * rgbData[i01 + ch] +
                                w10 * rgbData[i10 + ch] + w11 * rgbData[i11 + ch];
                res.warped_rgb[d_off + ch] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(v)), 0, 255));
            }
        }
    }

    res.applied = true;
    res.skew_deg = best_angle_deg;
    res.tps_applied = tps_ok;
    return res;
}

}  // namespace vellum::detector
