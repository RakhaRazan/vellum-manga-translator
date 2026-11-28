#include <httplib.h>

#include "inpainter.hpp"
#include "detector.hpp"
#include "math_inpaint_ops.hpp"
#include "ort_provider.hpp"
#include "../detector/cv_typography.hpp"
#include "../util/image_ops.hpp"
#include "../utils.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100 4244 4456 4996)
#endif
#include <stb_image.h>
#include <stb_image_write.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <utility>

namespace mangaflow::pipeline {

namespace {

using util::StbiPtr;
using util::resize_rgb_bilinear;

struct ScopeInpainterUnload {
    LamaInpainter* inpainter = nullptr;
    bool enabled = false;
    ~ScopeInpainterUnload() {
        if (enabled && inpainter) {
            inpainter->unload();
        }
    }
};

inline int clampi(int v, int lo, int hi) {
    return std::max(lo, std::min(hi, v));
}

inline float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

inline uint8_t rgb_to_luma(uint8_t r, uint8_t g, uint8_t b) {
    const float y = 0.299f * static_cast<float>(r) +
                    0.587f * static_cast<float>(g) +
                    0.114f * static_cast<float>(b);
    return static_cast<uint8_t>(clampi(static_cast<int>(std::lround(y)), 0, 255));
}

std::vector<uint8_t> resize_mask_nearest(
    const uint8_t* src,
    int src_w,
    int src_h,
    int dst_w,
    int dst_h) {
    std::vector<uint8_t> dst(static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h), 0);
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        return dst;
    }
    if (src_w == dst_w && src_h == dst_h) {
        std::memcpy(dst.data(), src, dst.size());
        return dst;
    }
    const float sx_ratio = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float sy_ratio = static_cast<float>(src_h) / static_cast<float>(dst_h);
    for (int y = 0; y < dst_h; ++y) {
        const int sy = clampi(static_cast<int>(std::floor((static_cast<float>(y) + 0.5f) * sy_ratio)), 0, src_h - 1);
        const uint8_t* srow = src + static_cast<size_t>(sy) * static_cast<size_t>(src_w);
        uint8_t* drow = dst.data() + static_cast<size_t>(y) * static_cast<size_t>(dst_w);
        for (int x = 0; x < dst_w; ++x) {
            const int sx = clampi(static_cast<int>(std::floor((static_cast<float>(x) + 0.5f) * sx_ratio)), 0, src_w - 1);
            drow[x] = (srow[sx] > 127) ? 255 : 0;
        }
    }
    return dst;
}

// Fast O(W*H) separable rectangular binary dilation (`k_w x k_h`).
std::vector<uint8_t> dilate_mask_rect(
    const std::vector<uint8_t>& src,
    int width,
    int height,
    int radius) {
    if (radius <= 0 || width <= 0 || height <= 0) {
        return src;
    }
    const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<uint8_t> horiz(total, 0);

    // Horizontal pass with sliding active count
    for (int y = 0; y < height; ++y) {
        const uint8_t* srow = src.data() + static_cast<size_t>(y) * static_cast<size_t>(width);
        uint8_t* hrow = horiz.data() + static_cast<size_t>(y) * static_cast<size_t>(width);
        int active = 0;
        for (int x = 0; x <= std::min(radius, width - 1); ++x) {
            if (srow[x] > 0) ++active;
        }
        for (int x = 0; x < width; ++x) {
            hrow[x] = (active > 0) ? 255 : 0;
            const int leave = x - radius;
            const int enter = x + radius + 1;
            if (leave >= 0 && srow[leave] > 0) --active;
            if (enter < width && srow[enter] > 0) ++active;
        }
    }

    // Vertical pass with sliding active count
    std::vector<uint8_t> out(total, 0);
    for (int x = 0; x < width; ++x) {
        int active = 0;
        for (int y = 0; y <= std::min(radius, height - 1); ++y) {
            if (horiz[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] > 0) {
                ++active;
            }
        }
        for (int y = 0; y < height; ++y) {
            out[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] =
                (active > 0) ? 255 : 0;
            const int leave = y - radius;
            const int enter = y + radius + 1;
            if (leave >= 0 && horiz[static_cast<size_t>(leave) * static_cast<size_t>(width) + static_cast<size_t>(x)] > 0) {
                --active;
            }
            if (enter < height && horiz[static_cast<size_t>(enter) * static_cast<size_t>(width) + static_cast<size_t>(x)] > 0) {
                ++active;
            }
        }
    }
    return out;
}

void dilate_mask_ellipse_inplace(
    std::vector<uint8_t>& mask,
    int width,
    int height,
    int kernel_size) {
    if (kernel_size <= 1 || width <= 0 || height <= 0) {
        return;
    }
    const int radius = kernel_size / 2;
    if (radius <= 0) return;

    std::vector<std::pair<int, int>> offsets;
    const float r2 = static_cast<float>(radius * radius) + 0.25f;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (static_cast<float>(dx * dx + dy * dy) <= r2) {
                offsets.emplace_back(dx, dy);
            }
        }
    }

    std::vector<uint8_t> dilated(mask.size(), 0);
    for (int y = 0; y < height; ++y) {
        const size_t row_off = static_cast<size_t>(y) * static_cast<size_t>(width);
        for (int x = 0; x < width; ++x) {
            if (mask[row_off + static_cast<size_t>(x)] == 0) {
                continue;
            }
            for (const auto& [dx, dy] : offsets) {
                const int nx = x + dx;
                const int ny = y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    dilated[static_cast<size_t>(ny) * static_cast<size_t>(width) + static_cast<size_t>(nx)] = 255;
                }
            }
        }
    }
    mask.swap(dilated);
}

void stbi_write_vec_callback(void* context, void* data, int size) {
    if (!context || !data || size <= 0) return;
    auto* vec = static_cast<std::vector<uint8_t>*>(context);
    const auto* bytes = static_cast<const uint8_t*>(data);
    vec->insert(vec->end(), bytes, bytes + size);
}

// -----------------------------------------------------------------------------
// Pure-C++ Telea Fast-Marching Method (FMM) Inpainter (Tier 2 CPU-First)
// -----------------------------------------------------------------------------

enum TeleaState : uint8_t {
    kTeleaKnown = 0,
    kTeleaBand = 1,
    kTeleaInside = 2,
};

inline float solve_eikonal_2d(
    int x1, int y1, int x2, int y2,
    int w, int h,
    const std::vector<uint8_t>& state,
    const std::vector<float>& dist) {
    constexpr float kLarge = 1.0e6f;
    float t1 = kLarge;
    float t2 = kLarge;
    if (x1 >= 0 && x1 < w && y1 >= 0 && y1 < h) {
        const size_t i1 = static_cast<size_t>(y1) * static_cast<size_t>(w) + static_cast<size_t>(x1);
        if (state[i1] == kTeleaKnown) t1 = dist[i1];
    }
    if (x2 >= 0 && x2 < w && y2 >= 0 && y2 < h) {
        const size_t i2 = static_cast<size_t>(y2) * static_cast<size_t>(w) + static_cast<size_t>(x2);
        if (state[i2] == kTeleaKnown) t2 = dist[i2];
    }
    if (t1 >= kLarge && t2 >= kLarge) return kLarge;
    if (t1 >= kLarge) return t2 + 1.0f;
    if (t2 >= kLarge) return t1 + 1.0f;

    const float diff = t1 - t2;
    const float r = 2.0f - diff * diff;
    if (r >= 0.0f) {
        const float s = 0.5f * (t1 + t2 + std::sqrt(r));
        if (s >= t1 && s >= t2) return s;
    }
    return std::min(t1, t2) + 1.0f;
}

void telea_inpaint_crop(
    std::vector<uint8_t>& crop_rgb,
    const std::vector<uint8_t>& hole_mask,
    const std::vector<uint8_t>& exclude_mask,
    int w,
    int h,
    float radius) {
    if (w <= 0 || h <= 0) return;
    const size_t total = static_cast<size_t>(w) * static_cast<size_t>(h);
    constexpr float kLarge = 1.0e6f;
    const int rad_i = std::max(1, static_cast<int>(std::ceil(radius)));
    const float rad_sq = radius * radius;

    std::vector<uint8_t> state(total, kTeleaKnown);
    std::vector<float> dist(total, 0.0f);
    std::vector<float> img_f(total * 3ULL, 0.0f);
    for (size_t i = 0; i < total; ++i) {
        img_f[i * 3 + 0] = static_cast<float>(crop_rgb[i * 3 + 0]);
        img_f[i * 3 + 1] = static_cast<float>(crop_rgb[i * 3 + 1]);
        img_f[i * 3 + 2] = static_cast<float>(crop_rgb[i * 3 + 2]);
        if (hole_mask[i] > 0) {
            state[i] = kTeleaInside;
            dist[i] = kLarge;
        }
    }

    struct HeapItem {
        float t;
        int idx;
        bool operator>(const HeapItem& o) const noexcept { return t > o.t; }
    };
    std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> heap;

    static constexpr int kDx4[4] = {-1, 1, 0, 0};
    static constexpr int kDy4[4] = {0, 0, -1, 1};

    auto compute_t = [&](int x, int y) -> float {
        float t = kLarge;
        t = std::min(t, solve_eikonal_2d(x - 1, y, x, y - 1, w, h, state, dist));
        t = std::min(t, solve_eikonal_2d(x + 1, y, x, y - 1, w, h, state, dist));
        t = std::min(t, solve_eikonal_2d(x - 1, y, x, y + 1, w, h, state, dist));
        t = std::min(t, solve_eikonal_2d(x + 1, y, x, y + 1, w, h, state, dist));
        return t;
    };

    // Seed narrow band with INSIDE pixels adjacent to KNOWN pixels
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x);
            if (state[idx] != kTeleaInside) continue;
            bool touches_known = false;
            for (int k = 0; k < 4; ++k) {
                const int nx = x + kDx4[k];
                const int ny = y + kDy4[k];
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    const size_t nidx = static_cast<size_t>(ny) * static_cast<size_t>(w) + static_cast<size_t>(nx);
                    if (state[nidx] == kTeleaKnown) {
                        touches_known = true;
                        break;
                    }
                }
            }
            if (touches_known) {
                state[idx] = kTeleaBand;
                dist[idx] = 1.0f;
                heap.push({1.0f, static_cast<int>(idx)});
            }
        }
    }

    auto inpaint_pixel = [&](int qx, int qy) {
        const size_t qidx = static_cast<size_t>(qy) * static_cast<size_t>(w) + static_cast<size_t>(qx);
        const float tq = dist[qidx];

        // Estimate gradT at q from known 4-neighbors
        float grad_tx = 0.0f;
        float grad_ty = 0.0f;
        const bool has_l = (qx > 0 && state[qidx - 1] == kTeleaKnown);
        const bool has_r = (qx + 1 < w && state[qidx + 1] == kTeleaKnown);
        if (has_l && has_r) grad_tx = 0.5f * (dist[qidx + 1] - dist[qidx - 1]);
        else if (has_r) grad_tx = dist[qidx + 1] - tq;
        else if (has_l) grad_tx = tq - dist[qidx - 1];

        const bool has_u = (qy > 0 && state[qidx - static_cast<size_t>(w)] == kTeleaKnown);
        const bool has_d = (qy + 1 < h && state[qidx + static_cast<size_t>(w)] == kTeleaKnown);
        if (has_u && has_d) grad_ty = 0.5f * (dist[qidx + static_cast<size_t>(w)] - dist[qidx - static_cast<size_t>(w)]);
        else if (has_d) grad_ty = dist[qidx + static_cast<size_t>(w)] - tq;
        else if (has_u) grad_ty = tq - dist[qidx - static_cast<size_t>(w)];

        double num[3] = {0.0, 0.0, 0.0};
        double w_sum = 0.0;

        const int py0 = std::max(0, qy - rad_i);
        const int py1 = std::min(h - 1, qy + rad_i);
        const int px0 = std::max(0, qx - rad_i);
        const int px1 = std::min(w - 1, qx + rad_i);

        for (int py = py0; py <= py1; ++py) {
            const float ry = static_cast<float>(qy - py);
            const size_t prow = static_cast<size_t>(py) * static_cast<size_t>(w);
            for (int px = px0; px <= px1; ++px) {
                const float rx = static_cast<float>(qx - px);
                const float d2 = rx * rx + ry * ry;
                if (d2 <= 0.0f || d2 > rad_sq) continue;
                const size_t pidx = prow + static_cast<size_t>(px);
                if (state[pidx] != kTeleaKnown) continue;
                if (!exclude_mask.empty() && exclude_mask[pidx] > 0) continue;

                const float d = std::sqrt(d2);
                float w_dir = std::fabs(rx * grad_tx + ry * grad_ty) / d;
                if (w_dir < 1.0e-6f) w_dir = 1.0e-6f;
                const float w_dst = 1.0f / d2;
                const float w_lev = 1.0f / (1.0f + std::fabs(tq - dist[pidx]));
                const double wt = static_cast<double>(w_dir) * static_cast<double>(w_dst) * static_cast<double>(w_lev);

                // First-order Taylor gradient extrapolation from known neighbors of p
                const bool pl = (px > 0 && state[pidx - 1] == kTeleaKnown &&
                                 (exclude_mask.empty() || exclude_mask[pidx - 1] == 0));
                const bool pr = (px + 1 < w && state[pidx + 1] == kTeleaKnown &&
                                 (exclude_mask.empty() || exclude_mask[pidx + 1] == 0));
                const bool pu = (py > 0 && state[pidx - static_cast<size_t>(w)] == kTeleaKnown &&
                                 (exclude_mask.empty() || exclude_mask[pidx - static_cast<size_t>(w)] == 0));
                const bool pd = (py + 1 < h && state[pidx + static_cast<size_t>(w)] == kTeleaKnown &&
                                 (exclude_mask.empty() || exclude_mask[pidx + static_cast<size_t>(w)] == 0));

                for (int c = 0; c < 3; ++c) {
                    const float ip = img_f[pidx * 3 + c];
                    float gx_i = 0.0f;
                    float gy_i = 0.0f;
                    if (pl && pr) {
                        gx_i = 0.5f * (img_f[(pidx + 1) * 3 + c] - img_f[(pidx - 1) * 3 + c]);
                    } else if (pr) {
                        gx_i = img_f[(pidx + 1) * 3 + c] - ip;
                    } else if (pl) {
                        gx_i = ip - img_f[(pidx - 1) * 3 + c];
                    }
                    if (pu && pd) {
                        gy_i = 0.5f * (img_f[(pidx + static_cast<size_t>(w)) * 3 + c] -
                                       img_f[(pidx - static_cast<size_t>(w)) * 3 + c]);
                    } else if (pd) {
                        gy_i = img_f[(pidx + static_cast<size_t>(w)) * 3 + c] - ip;
                    } else if (pu) {
                        gy_i = ip - img_f[(pidx - static_cast<size_t>(w)) * 3 + c];
                    }
                    const float est = clampf(ip + gx_i * rx + gy_i * ry, 0.0f, 255.0f);
                    num[c] += wt * static_cast<double>(est);
                }
                w_sum += wt;
            }
        }

        if (w_sum > 1.0e-12) {
            for (int c = 0; c < 3; ++c) {
                img_f[qidx * 3 + c] = clampf(static_cast<float>(num[c] / w_sum), 0.0f, 255.0f);
            }
        }
    };

    while (!heap.empty()) {
        const HeapItem top = heap.top();
        heap.pop();
        const size_t idx = static_cast<size_t>(top.idx);
        if (state[idx] == kTeleaKnown) continue;

        const int cx = top.idx % w;
        const int cy = top.idx / w;
        inpaint_pixel(cx, cy);
        state[idx] = kTeleaKnown;

        for (int k = 0; k < 4; ++k) {
            const int nx = cx + kDx4[k];
            const int ny = cy + kDy4[k];
            if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
            const size_t nidx = static_cast<size_t>(ny) * static_cast<size_t>(w) + static_cast<size_t>(nx);
            if (state[nidx] == kTeleaKnown) continue;
            const float nt = compute_t(nx, ny);
            if (nt < dist[nidx]) {
                dist[nidx] = nt;
                state[nidx] = kTeleaBand;
                heap.push({nt, static_cast<int>(nidx)});
            }
        }
    }

    for (size_t i = 0; i < total; ++i) {
        if (hole_mask[i] > 0) {
            crop_rgb[i * 3 + 0] = static_cast<uint8_t>(clampi(static_cast<int>(std::lround(img_f[i * 3 + 0])), 0, 255));
            crop_rgb[i * 3 + 1] = static_cast<uint8_t>(clampi(static_cast<int>(std::lround(img_f[i * 3 + 1])), 0, 255));
            crop_rgb[i * 3 + 2] = static_cast<uint8_t>(clampi(static_cast<int>(std::lround(img_f[i * 3 + 2])), 0, 255));
        }
    }
}

// -----------------------------------------------------------------------------
// Seam Elimination / Tone Match: 4x3 Ridge Affine Color Regression +
// Separable 1D Gaussian Residual Diffusion (sigma = 24.0 px)
// -----------------------------------------------------------------------------

void separable_gaussian_blur_f32(
    const std::vector<float>& src,
    std::vector<float>& dst,
    int w,
    int h,
    float sigma) {
    const size_t total = static_cast<size_t>(w) * static_cast<size_t>(h);
    dst.assign(total, 0.0f);
    if (w <= 0 || h <= 0 || sigma <= 0.0f) {
        dst = src;
        return;
    }

    const int radius = std::max(1, static_cast<int>(std::ceil(3.0 * static_cast<double>(sigma))));
    std::vector<float> kernel(static_cast<size_t>(radius + 1), 0.0f);
    const double inv_2sig2 = 1.0 / (2.0 * static_cast<double>(sigma) * static_cast<double>(sigma));
    double ksum = 1.0;
    kernel[0] = 1.0f;
    for (int d = 1; d <= radius; ++d) {
        const double val = std::exp(-static_cast<double>(d * d) * inv_2sig2);
        kernel[static_cast<size_t>(d)] = static_cast<float>(val);
        ksum += 2.0 * val;
    }
    const float inv_ksum = static_cast<float>(1.0 / ksum);
    for (float& kv : kernel) kv *= inv_ksum;

    std::vector<float> tmp(total, 0.0f);

    // Horizontal pass (clamp-to-edge)
    for (int y = 0; y < h; ++y) {
        const float* srow = src.data() + static_cast<size_t>(y) * static_cast<size_t>(w);
        float* trow = tmp.data() + static_cast<size_t>(y) * static_cast<size_t>(w);
        for (int x = 0; x < w; ++x) {
            float acc = srow[x] * kernel[0];
            for (int d = 1; d <= radius; ++d) {
                const int xl = (x - d < 0) ? 0 : (x - d);
                const int xr = (x + d >= w) ? (w - 1) : (x + d);
                acc += (srow[xl] + srow[xr]) * kernel[static_cast<size_t>(d)];
            }
            trow[x] = acc;
        }
    }

    // Vertical pass (clamp-to-edge)
    for (int y = 0; y < h; ++y) {
        float* drow = dst.data() + static_cast<size_t>(y) * static_cast<size_t>(w);
        const float* crow = tmp.data() + static_cast<size_t>(y) * static_cast<size_t>(w);
        const float k0 = kernel[0];
        for (int x = 0; x < w; ++x) {
            drow[x] = crow[x] * k0;
        }
        for (int d = 1; d <= radius; ++d) {
            const int yu = (y - d < 0) ? 0 : (y - d);
            const int yd = (y + d >= h) ? (h - 1) : (y + d);
            const float* urow = tmp.data() + static_cast<size_t>(yu) * static_cast<size_t>(w);
            const float* lrow = tmp.data() + static_cast<size_t>(yd) * static_cast<size_t>(w);
            const float kd = kernel[static_cast<size_t>(d)];
            for (int x = 0; x < w; ++x) {
                drow[x] += (urow[x] + lrow[x]) * kd;
            }
        }
    }
}

bool solve_4x4_ridge_affine(
    const double A_in[4][4],
    const double B_in[4][3],
    double W_out[4][3]) {
    double M[4][7]{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) M[r][c] = A_in[r][c];
        for (int c = 0; c < 3; ++c) M[r][4 + c] = B_in[r][c];
    }

    double det = 1.0;
    for (int col = 0; col < 4; ++col) {
        int pivot = col;
        double max_abs = std::fabs(M[col][col]);
        for (int r = col + 1; r < 4; ++r) {
            const double v = std::fabs(M[r][col]);
            if (v > max_abs) {
                max_abs = v;
                pivot = r;
            }
        }
        if (max_abs < 1.0e-12) {
            return false;
        }
        if (pivot != col) {
            for (int c = col; c < 7; ++c) {
                std::swap(M[col][c], M[pivot][c]);
            }
            det = -det;
        }
        const double diag = M[col][col];
        det *= diag;
        for (int r = col + 1; r < 4; ++r) {
            const double factor = M[r][col] / diag;
            M[r][col] = 0.0;
            for (int c = col + 1; c < 7; ++c) {
                M[r][c] -= factor * M[col][c];
            }
        }
    }

    if (!std::isfinite(det) || std::fabs(det) < 1.0e-8) {
        return false;
    }

    // Back-substitution for all 3 output channels
    for (int ch = 0; ch < 3; ++ch) {
        for (int r = 3; r >= 0; --r) {
            double rhs = M[r][4 + ch];
            for (int c = r + 1; c < 4; ++c) {
                rhs -= M[r][c] * W_out[c][ch];
            }
            W_out[r][ch] = rhs / M[r][r];
            if (!std::isfinite(W_out[r][ch])) {
                return false;
            }
        }
    }
    return true;
}

bool apply_tone_match(
    const uint8_t* src_rgb,
    std::vector<uint8_t>& pred_rgb,
    const uint8_t* mask,
    int w,
    int h) {
    if (!src_rgb || !mask || w <= 0 || h <= 0) return false;
    const size_t total = static_cast<size_t>(w) * static_cast<size_t>(h);
    if (pred_rgb.size() < total * 3ULL) return false;

    size_t k_count = 0;
    double A[4][4]{};
    double B[4][3]{};
    static constexpr double kInv255 = 1.0 / 255.0;

    for (size_t i = 0; i < total; ++i) {
        if (mask[i] != 0) continue;
        ++k_count;
        const double x[4] = {
            static_cast<double>(pred_rgb[i * 3 + 0]) * kInv255,
            static_cast<double>(pred_rgb[i * 3 + 1]) * kInv255,
            static_cast<double>(pred_rgb[i * 3 + 2]) * kInv255,
            1.0
        };
        const double y[3] = {
            static_cast<double>(src_rgb[i * 3 + 0]) * kInv255,
            static_cast<double>(src_rgb[i * 3 + 1]) * kInv255,
            static_cast<double>(src_rgb[i * 3 + 2]) * kInv255
        };
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                A[r][c] += x[r] * x[c];
            }
            for (int ch = 0; ch < 3; ++ch) {
                B[r][ch] += x[r] * y[ch];
            }
        }
    }

    // No-op guard: require at least 16 unmasked context pixels
    if (k_count < 16) {
        return false;
    }

    // Normalize normal equations by k_count so ridge lambda is invariant to tile size,
    // and regularize toward the identity transform (I_3) on the 3x3 RGB diagonal ONLY
    // (intercept A[3][3] unregularized). This prevents ill-conditioning on grayscale /
    // rank-1 manga tiles where R ~= G ~= B would otherwise cause large off-diagonal
    // cancellation weights and clamping artifacts inside the inpainted hole.
    const double inv_k = 1.0 / static_cast<double>(k_count);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            A[r][c] *= inv_k;
        }
        for (int ch = 0; ch < 3; ++ch) {
            B[r][ch] *= inv_k;
        }
    }
    static constexpr double kRidgeLambda = 0.01;
    for (int c = 0; c < 3; ++c) {
        A[c][c] += kRidgeLambda;
        B[c][c] += kRidgeLambda;
    }

    double W[4][3]{};
    if (!solve_4x4_ridge_affine(A, B, W)) {
        return false;
    }

    // Compute affine-matched image and context residuals r_c = I_src - I_affine
    std::vector<float> aff_r(total, 0.0f);
    std::vector<float> aff_g(total, 0.0f);
    std::vector<float> aff_b(total, 0.0f);
    std::vector<float> weight_map(total, 0.0f);
    std::vector<float> res_r(total, 0.0f);
    std::vector<float> res_g(total, 0.0f);
    std::vector<float> res_b(total, 0.0f);

    for (size_t i = 0; i < total; ++i) {
        const double p0 = static_cast<double>(pred_rgb[i * 3 + 0]) * kInv255;
        const double p1 = static_cast<double>(pred_rgb[i * 3 + 1]) * kInv255;
        const double p2 = static_cast<double>(pred_rgb[i * 3 + 2]) * kInv255;

        const float ar = static_cast<float>(W[0][0] * p0 + W[1][0] * p1 + W[2][0] * p2 + W[3][0]);
        const float ag = static_cast<float>(W[0][1] * p0 + W[1][1] * p1 + W[2][1] * p2 + W[3][1]);
        const float ab = static_cast<float>(W[0][2] * p0 + W[1][2] * p1 + W[2][2] * p2 + W[3][2]);
        aff_r[i] = ar;
        aff_g[i] = ag;
        aff_b[i] = ab;

        if (mask[i] == 0) {
            weight_map[i] = 1.0f;
            res_r[i] = static_cast<float>(static_cast<double>(src_rgb[i * 3 + 0]) * kInv255) - ar;
            res_g[i] = static_cast<float>(static_cast<double>(src_rgb[i * 3 + 1]) * kInv255) - ag;
            res_b[i] = static_cast<float>(static_cast<double>(src_rgb[i * 3 + 2]) * kInv255) - ab;
        }
    }

    // Diffuse residuals across the hole via separable 1D Gaussian blur (sigma = 24.0 px)
    static constexpr float kToneMatchSigma = 24.0f;
    std::vector<float> blur_w;
    std::vector<float> blur_rr;
    std::vector<float> blur_rg;
    std::vector<float> blur_rb;
    separable_gaussian_blur_f32(weight_map, blur_w, w, h, kToneMatchSigma);
    separable_gaussian_blur_f32(res_r, blur_rr, w, h, kToneMatchSigma);
    separable_gaussian_blur_f32(res_g, blur_rg, w, h, kToneMatchSigma);
    separable_gaussian_blur_f32(res_b, blur_rb, w, h, kToneMatchSigma);

    for (size_t i = 0; i < total; ++i) {
        if (mask[i] == 0) continue;
        float out_r = aff_r[i];
        float out_g = aff_g[i];
        float out_b = aff_b[i];
        if (blur_w[i] > 1.0e-3f) {
            const float inv_w = 1.0f / blur_w[i];
            out_r += blur_rr[i] * inv_w;
            out_g += blur_rg[i] * inv_w;
            out_b += blur_rb[i] * inv_w;
        }
        pred_rgb[i * 3 + 0] = static_cast<uint8_t>(clampi(static_cast<int>(std::lround(clampf(out_r, 0.0f, 1.0f) * 255.0f)), 0, 255));
        pred_rgb[i * 3 + 1] = static_cast<uint8_t>(clampi(static_cast<int>(std::lround(clampf(out_g, 0.0f, 1.0f) * 255.0f)), 0, 255));
        pred_rgb[i * 3 + 2] = static_cast<uint8_t>(clampi(static_cast<int>(std::lround(clampf(out_b, 0.0f, 1.0f) * 255.0f)), 0, 255));
    }

    return true;
}

}  // namespace

// -----------------------------------------------------------------------------
// Static Image / Mask Decoding & PNG Encoding Helpers
// -----------------------------------------------------------------------------

LamaInpainter::LamaInpainter(const ModelManager& model_manager)
    : model_manager_(model_manager) {}

void LamaInpainter::unload() {
    cached_session_.reset();
    cached_model_path_.clear();
    cached_force_cpu_ = false;
    cached_aot_session_.reset();
    cached_aot_path_.clear();
    cached_aot_force_cpu_ = false;
}

std::vector<uint8_t> LamaInpainter::decode_rgb_from_bytes(
    const uint8_t* image_bytes,
    size_t num_bytes,
    int& width_out,
    int& height_out) {
    if (!image_bytes || num_bytes == 0) {
        throw std::invalid_argument("Uploaded image data is empty.");
    }
    if (num_bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Image payload exceeds maximum supported size.");
    }

    int w = 0;
    int h = 0;
    int comp = 0;
    StbiPtr rgb(stbi_load_from_memory(
        image_bytes,
        static_cast<int>(num_bytes),
        &w,
        &h,
        &comp,
        3));
    if (!rgb || w <= 0 || h <= 0) {
        rgb.reset();
        std::vector<uint8_t> wic_rgb;
        if (util::decode_image_wic(image_bytes, num_bytes, 3, wic_rgb, w, h) &&
            !wic_rgb.empty() && w > 0 && h > 0) {
            width_out = w;
            height_out = h;
            return wic_rgb;
        }
        const char* reason = stbi_failure_reason();
        throw std::invalid_argument(
            std::string("Failed to decode image via stb_image: ") + (reason ? reason : "invalid format"));
    }

    width_out = w;
    height_out = h;
    const size_t byte_len = static_cast<size_t>(w) * static_cast<size_t>(h) * 3ULL;
    return std::vector<uint8_t>(rgb.get(), rgb.get() + byte_len);
}

std::vector<uint8_t> LamaInpainter::decode_mask_from_bytes(
    const uint8_t* mask_bytes,
    size_t num_bytes,
    int target_w,
    int target_h) {
    if (!mask_bytes || num_bytes == 0) {
        throw std::invalid_argument("Mask image data is empty.");
    }
    if (num_bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Mask payload exceeds maximum supported size.");
    }

    int mw = 0;
    int mh = 0;
    int comp = 0;
    StbiPtr gray(stbi_load_from_memory(
        mask_bytes,
        static_cast<int>(num_bytes),
        &mw,
        &mh,
        &comp,
        1));
    if (!gray || mw <= 0 || mh <= 0) {
        const char* reason = stbi_failure_reason();
        throw std::invalid_argument(
            std::string("Failed to decode mask image via stb_image: ") + (reason ? reason : "invalid format"));
    }

    std::vector<uint8_t> out;
    if (mw == target_w && mh == target_h) {
        const size_t total = static_cast<size_t>(mw) * static_cast<size_t>(mh);
        out.resize(total);
        for (size_t i = 0; i < total; ++i) {
            out[i] = (gray.get()[i] > 127) ? 255 : 0;
        }
    } else {
        out = resize_mask_nearest(gray.get(), mw, mh, target_w, target_h);
    }
    return out;
}

std::vector<uint8_t> LamaInpainter::synthesize_mask_from_bboxes(
    const std::vector<std::array<int, 4>>& bboxes,
    int width,
    int height) {
    std::vector<uint8_t> mask(static_cast<size_t>(width) * static_cast<size_t>(height), 0);
    for (const auto& box : bboxes) {
        const int x1 = clampi(box[0], 0, width);
        const int y1 = clampi(box[1], 0, height);
        const int x2 = clampi(box[2], 0, width);
        const int y2 = clampi(box[3], 0, height);
        if (x2 <= x1 || y2 <= y1) continue;
        for (int y = y1; y < y2; ++y) {
            std::memset(
                mask.data() + static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x1),
                255,
                static_cast<size_t>(x2 - x1));
        }
    }
    return mask;
}

void LamaInpainter::restrict_mask_to_bboxes(
    std::vector<uint8_t>& mask,
    const std::vector<std::array<int, 4>>& bboxes,
    int width,
    int height,
    int pad) {
    std::vector<uint8_t> roi_mask(mask.size(), 0);
    for (const auto& box : bboxes) {
        const int x1 = clampi(box[0] - pad, 0, width);
        const int y1 = clampi(box[1] - pad, 0, height);
        const int x2 = clampi(box[2] + pad, 0, width);
        const int y2 = clampi(box[3] + pad, 0, height);
        if (x2 <= x1 || y2 <= y1) continue;
        for (int y = y1; y < y2; ++y) {
            std::memset(
                roi_mask.data() + static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x1),
                255,
                static_cast<size_t>(x2 - x1));
        }
    }
    for (size_t i = 0; i < mask.size(); ++i) {
        mask[i] = (mask[i] > 0 && roi_mask[i] > 0) ? 255 : 0;
    }
}

std::vector<uint8_t> LamaInpainter::encode_rgb_png(
    const std::vector<uint8_t>& rgb,
    int width,
    int height) {
    if (width <= 0 || height <= 0 ||
        rgb.size() < static_cast<size_t>(width) * static_cast<size_t>(height) * 3ULL) {
        throw std::invalid_argument("Invalid RGB image buffer for PNG encoding.");
    }
    std::vector<uint8_t> png_bytes;
    png_bytes.reserve(static_cast<size_t>(width) * static_cast<size_t>(height));
    const int ok = stbi_write_png_to_func(
        &stbi_write_vec_callback,
        &png_bytes,
        width,
        height,
        3,
        rgb.data(),
        width * 3);
    if (!ok || png_bytes.empty()) {
        throw std::runtime_error("Failed to encode inpainted RGB image to PNG.");
    }
    return png_bytes;
}

std::string LamaInpainter::rgb_to_data_uri(
    const std::vector<uint8_t>& rgb,
    int width,
    int height) {
    const auto png = encode_rgb_png(rgb, width, height);
    return "data:image/png;base64," + util::base64_encode(png.data(), png.size());
}

void LamaInpainter::composite_feathered_region(
    std::vector<uint8_t>& dst_rgb,
    const std::vector<uint8_t>& src_rgb,
    const std::vector<uint8_t>& mask,
    int width,
    int height,
    const std::array<int, 4>& roi,
    float feather_px) {
    if (width <= 0 || height <= 0) return;
    const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (dst_rgb.size() < total * 3ULL || src_rgb.size() < total * 3ULL || mask.size() < total) {
        return;
    }

    const int pad = std::max(2, static_cast<int>(std::ceil(std::max(0.0f, feather_px))) + 2);
    const int lx0 = clampi(roi[0] - pad, 0, width);
    const int ly0 = clampi(roi[1] - pad, 0, height);
    const int lx1 = clampi(roi[2] + pad, 0, width);
    const int ly1 = clampi(roi[3] + pad, 0, height);
    const int lw = lx1 - lx0;
    const int lh = ly1 - ly0;
    if (lw <= 0 || lh <= 0) return;

    // #4B: Discrete Screened Poisson Blending via Red-Black SOR for seamless Spot Clean
    // patches (<= 512x512), falling back to EDT alpha feathering for larger regions.
    if (feather_px > 0.0f &&
        math_inpaint::poissonBlendRegionSOR(
            dst_rgb.data(), src_rgb.data(), mask.data(),
            width, height, lx0, ly0, lx1, ly1)) {
        return;
    }

    if (feather_px <= 0.0f) {
        for (int ly = 0; ly < lh; ++ly) {
            const int gy = ly0 + ly;
            const uint8_t* mrow = mask.data() + static_cast<size_t>(gy) * static_cast<size_t>(width);
            const uint8_t* srow = src_rgb.data() + static_cast<size_t>(gy) * static_cast<size_t>(width) * 3ULL;
            uint8_t* drow = dst_rgb.data() + static_cast<size_t>(gy) * static_cast<size_t>(width) * 3ULL;
            for (int lx = 0; lx < lw; ++lx) {
                const int gx = lx0 + lx;
                if (mrow[gx] > 0) {
                    drow[gx * 3 + 0] = srow[gx * 3 + 0];
                    drow[gx * 3 + 1] = srow[gx * 3 + 1];
                    drow[gx * 3 + 2] = srow[gx * 3 + 2];
                }
            }
        }
        return;
    }

    // Distance from each masked pixel to the nearest unmasked pixel using Felzenszwalb-Huttenlocher 2D EDT
    std::vector<uint8_t> bg_mask(static_cast<size_t>(lw) * static_cast<size_t>(lh), 0);
    for (int ly = 0; ly < lh; ++ly) {
        const uint8_t* mrow = mask.data() + static_cast<size_t>(ly0 + ly) * static_cast<size_t>(width) + static_cast<size_t>(lx0);
        uint8_t* brow = bg_mask.data() + static_cast<size_t>(ly) * static_cast<size_t>(lw);
        for (int lx = 0; lx < lw; ++lx) {
            brow[lx] = (mrow[lx] == 0) ? 255 : 0;
        }
    }

    std::vector<float> dist_sq;
    const bool has_bg = vellum::detector::computeEuclideanDistanceSqFH(bg_mask.data(), lw, lh, dist_sq);
    const float inv_feather = 1.0f / feather_px;

    for (int ly = 0; ly < lh; ++ly) {
        const int gy = ly0 + ly;
        const uint8_t* mrow = mask.data() + static_cast<size_t>(gy) * static_cast<size_t>(width);
        const uint8_t* srow = src_rgb.data() + static_cast<size_t>(gy) * static_cast<size_t>(width) * 3ULL;
        uint8_t* drow = dst_rgb.data() + static_cast<size_t>(gy) * static_cast<size_t>(width) * 3ULL;
        const size_t lrow = static_cast<size_t>(ly) * static_cast<size_t>(lw);
        for (int lx = 0; lx < lw; ++lx) {
            const int gx = lx0 + lx;
            if (mrow[gx] == 0) continue;
            const float alpha = has_bg
                ? clampf(std::sqrt(std::max(0.0f, dist_sq[lrow + static_cast<size_t>(lx)])) * inv_feather, 0.0f, 1.0f)
                : 1.0f;
            const float inv_a = 1.0f - alpha;
            for (int c = 0; c < 3; ++c) {
                const float blended = alpha * static_cast<float>(srow[gx * 3 + c]) +
                                      inv_a * static_cast<float>(drow[gx * 3 + c]);
                drow[gx * 3 + c] = static_cast<uint8_t>(clampi(static_cast<int>(std::lround(blended)), 0, 255));
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Mask Connected-Component ROI Extraction & Interior Ink Refinement
// -----------------------------------------------------------------------------

std::vector<std::array<int, 4>> LamaInpainter::extract_bboxes_from_mask(
    const std::vector<uint8_t>& mask,
    int width,
    int height) {
    bool has_any = false;
    for (uint8_t v : mask) {
        if (v > 0) {
            has_any = true;
            break;
        }
    }
    if (!has_any) return {};

    // Group nearby characters/lines within the same bubble via 25x25 rectangular dilation (radius=12)
    const std::vector<uint8_t> grouped = dilate_mask_rect(mask, width, height, 12);
    const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<uint8_t> visited(total, 0);
    std::vector<std::array<int, 4>> bboxes;

    static constexpr int kDx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static constexpr int kDy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t start = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            if (visited[start] || grouped[start] == 0) continue;

            int min_x = x, max_x = x, min_y = y, max_y = y;
            int area = 0;
            std::queue<int> q;
            visited[start] = 1;
            q.push(static_cast<int>(start));

            while (!q.empty()) {
                const int cur = q.front();
                q.pop();
                const int cx = cur % width;
                const int cy = cur / width;
                min_x = std::min(min_x, cx);
                max_x = std::max(max_x, cx);
                min_y = std::min(min_y, cy);
                max_y = std::max(max_y, cy);
                ++area;

                for (int k = 0; k < 8; ++k) {
                    const int nx = cx + kDx[k];
                    const int ny = cy + kDy[k];
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                    const size_t nidx = static_cast<size_t>(ny) * static_cast<size_t>(width) + static_cast<size_t>(nx);
                    if (!visited[nidx] && grouped[nidx] > 0) {
                        visited[nidx] = 1;
                        q.push(static_cast<int>(nidx));
                    }
                }
            }

            const int bw = max_x - min_x + 1;
            const int bh = max_y - min_y + 1;
            if (area >= 16 && bw >= 4 && bh >= 4) {
                bboxes.push_back({min_x, min_y, std::min(width, max_x + 1), std::min(height, max_y + 1)});
            }
        }
    }
    return bboxes;
}

std::vector<uint8_t> LamaInpainter::refine_bubble_mask(
    const uint8_t* rgb_data,
    const std::vector<uint8_t>& base_mask,
    const std::vector<std::array<int, 4>>& bboxes,
    int width,
    int height,
    int dilate_kernel_size) {
    std::vector<uint8_t> refined = base_mask;
    std::vector<uint8_t> global_protected_wall(static_cast<size_t>(width) * static_cast<size_t>(height), 0);

    static constexpr int kDx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static constexpr int kDy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    for (const auto& box : bboxes) {
        const int x1 = clampi(box[0], 0, width);
        const int y1 = clampi(box[1], 0, height);
        const int x2 = clampi(box[2], 0, width);
        const int y2 = clampi(box[3], 0, height);
        const int bw = x2 - x1;
        const int bh = y2 - y1;
        if (bw < 10 || bh < 10) continue;

        const size_t sub_total = static_cast<size_t>(bw) * static_cast<size_t>(bh);
        std::vector<uint8_t> luma_grid(sub_total, 255);
        std::vector<uint8_t> dark_wall_cand(sub_total, 0);
        for (int y = 0; y < bh; ++y) {
            const uint8_t* srow = rgb_data +
                                  (static_cast<size_t>(y1 + y) * static_cast<size_t>(width) +
                                   static_cast<size_t>(x1)) * 3ULL;
            for (int x = 0; x < bw; ++x) {
                const uint8_t* px = srow + static_cast<size_t>(x) * 3ULL;
                const uint8_t lum = rgb_to_luma(px[0], px[1], px[2]);
                const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(bw) + static_cast<size_t>(x);
                luma_grid[idx] = lum;
                if (lum < 170) {
                    dark_wall_cand[idx] = 1;
                }
            }
        }

        // Flood-fill border-touching dark ink (luma < 170) so speech bubble borders are preserved
        std::vector<uint8_t> border_ink(sub_total, 0);
        std::queue<int> q;
        auto mark_border = [&](int lx, int ly) {
            const size_t idx = static_cast<size_t>(ly) * static_cast<size_t>(bw) + static_cast<size_t>(lx);
            if (dark_wall_cand[idx] && !border_ink[idx]) {
                border_ink[idx] = 1;
                q.push(static_cast<int>(idx));
            }
        };
        for (int lx = 0; lx < bw; ++lx) {
            mark_border(lx, 0);
            mark_border(lx, bh - 1);
        }
        for (int ly = 0; ly < bh; ++ly) {
            mark_border(0, ly);
            mark_border(bw - 1, ly);
        }
        while (!q.empty()) {
            const int cur = q.front();
            q.pop();
            const int cx = cur % bw;
            const int cy = cur / bw;
            for (int k = 0; k < 8; ++k) {
                const int nx = cx + kDx[k];
                const int ny = cy + kDy[k];
                if (nx >= 0 && nx < bw && ny >= 0 && ny < bh) {
                    mark_border(nx, ny);
                }
            }
        }

        // 2px safety dilation around border_ink so text dilation never eats into bubble outlines
        std::vector<uint8_t> protected_wall(sub_total, 0);
        for (int ly = 0; ly < bh; ++ly) {
            for (int lx = 0; lx < bw; ++lx) {
                if (!border_ink[static_cast<size_t>(ly) * static_cast<size_t>(bw) + static_cast<size_t>(lx)]) continue;
                for (int dy = -2; dy <= 2; ++dy) {
                    const int ny = ly + dy;
                    if (ny < 0 || ny >= bh) continue;
                    for (int dx = -2; dx <= 2; ++dx) {
                        if (dx * dx + dy * dy > 4) continue;
                        const int nx = lx + dx;
                        if (nx >= 0 && nx < bw) {
                            protected_wall[static_cast<size_t>(ny) * static_cast<size_t>(bw) + static_cast<size_t>(nx)] = 1;
                        }
                    }
                }
            }
        }

        // Add all non-border interior text strokes & punctuation (luma < 215) into `refined`
        for (int ly = 0; ly < bh; ++ly) {
            uint8_t* mrow = refined.data() + static_cast<size_t>(y1 + ly) * static_cast<size_t>(width) + static_cast<size_t>(x1);
            uint8_t* prow = global_protected_wall.data() + static_cast<size_t>(y1 + ly) * static_cast<size_t>(width) + static_cast<size_t>(x1);
            for (int lx = 0; lx < bw; ++lx) {
                const size_t idx = static_cast<size_t>(ly) * static_cast<size_t>(bw) + static_cast<size_t>(lx);
                if (protected_wall[idx]) {
                    mrow[lx] = 0;
                    prow[lx] = 1;
                } else if (luma_grid[idx] < 215) {
                    mrow[lx] = 255;
                }
            }
        }
    }

    if (dilate_kernel_size > 1) {
        dilate_mask_ellipse_inplace(refined, width, height, dilate_kernel_size);
    }
    for (size_t i = 0; i < refined.size(); ++i) {
        if (global_protected_wall[i]) {
            refined[i] = 0;
        }
    }
    return refined;
}

// -----------------------------------------------------------------------------
// LaMa ONNX Session & Tile-Based 512x512 ROI Execution
// -----------------------------------------------------------------------------

OrtSessionHandle& LamaInpainter::ensure_session(
    const ResolvedLamaModel& resolved,
    bool force_cpu) {
    if (cached_session_.has_value() &&
        cached_model_path_ == resolved.model_path &&
        cached_force_cpu_ == force_cpu &&
        (!force_cpu || cached_session_->active_provider == "CPUExecutionProvider")) {
        return *cached_session_;
    }
    cached_session_.reset();
    cached_session_ = ModelManager::create_ort_session(resolved.model_path, force_cpu);
    cached_model_path_ = resolved.model_path;
    cached_force_cpu_ = force_cpu;
    return *cached_session_;
}

OrtSessionHandle& LamaInpainter::ensure_aot_session(
    const ResolvedAotModel& resolved,
    bool force_cpu) {
    if (cached_aot_session_.has_value() &&
        cached_aot_path_ == resolved.model_path &&
        cached_aot_force_cpu_ == force_cpu &&
        (!force_cpu || cached_aot_session_->active_provider == "CPUExecutionProvider")) {
        return *cached_aot_session_;
    }
    cached_aot_session_.reset();
    cached_aot_session_ = ModelManager::create_ort_session(resolved.model_path, force_cpu);
    cached_aot_path_ = resolved.model_path;
    cached_aot_force_cpu_ = force_cpu;
    return *cached_aot_session_;
}

std::vector<uint8_t> LamaInpainter::run_aot_tile_512(
    const uint8_t* tile_rgb,
    const uint8_t* tile_mask,
    const ResolvedAotModel& resolved,
    bool force_cpu) {
    constexpr size_t kPlane = static_cast<size_t>(kTileSize) * static_cast<size_t>(kTileSize);
    // 4-channel input [1, 4, 512, 512]: ch0 = binary mask [0, 1], ch1..3 = RGB [0, 1]
    std::vector<float> in_tensor(4ULL * kPlane);
    static constexpr float kInv255 = 1.0f / 255.0f;
    float* ch_m = in_tensor.data();
    float* ch_r = in_tensor.data() + kPlane;
    float* ch_g = in_tensor.data() + 2ULL * kPlane;
    float* ch_b = in_tensor.data() + 3ULL * kPlane;

    for (size_t i = 0; i < kPlane; ++i) {
        const uint8_t* px = tile_rgb + i * 3ULL;
        const bool is_masked = (tile_mask[i] > 127);
        const float m = is_masked ? 1.0f : 0.0f;
        const float keep = is_masked ? 0.0f : 1.0f;
        ch_m[i] = m;
        ch_r[i] = ((static_cast<float>(px[0]) * kInv255) * 2.0f - 1.0f) * keep;
        ch_g[i] = ((static_cast<float>(px[1]) * kInv255) * 2.0f - 1.0f) * keep;
        ch_b[i] = ((static_cast<float>(px[2]) * kInv255) * 2.0f - 1.0f) * keep;
    }

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 4> in_shape = {1, 4, kTileSize, kTileSize};

    auto run_once = [&](OrtSessionHandle& handle) -> std::vector<Ort::Value> {
        Ort::Value ort_in = Ort::Value::CreateTensor<float>(
            mem_info, in_tensor.data(), in_tensor.size(), in_shape.data(), in_shape.size());
        const char* in_names[] = {"input"};
        const char* out_names[] = {"output"};
        const bool is_cuda = (handle.active_provider == "CUDAExecutionProvider");
        Ort::RunOptions ro = make_shrink_run_options(true, is_cuda);
        return handle.session->Run(
            ro, in_names, &ort_in, 1, out_names, 1);
    };

    std::vector<Ort::Value> outputs;
    {
        OrtSessionHandle& handle = ensure_aot_session(resolved, force_cpu);
        try {
            outputs = run_once(handle);
        } catch (const Ort::Exception&) {
            if (handle.active_provider == "CPUExecutionProvider") {
                throw;
            }
            OrtSessionHandle& cpu_handle = ensure_aot_session(resolved, /*force_cpu=*/true);
            outputs = run_once(cpu_handle);
        }
    }

    const float* out_ptr = outputs[0].GetTensorData<float>();
    const size_t elem_cnt = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    if (elem_cnt < 3ULL * kPlane) {
        throw std::runtime_error("Unexpected AOT ONNX output tensor element count.");
    }

    float min_v = 0.0f;
    float max_v = 0.0f;
    for (size_t i = 0; i < 3ULL * kPlane; i += 64) {
        if (out_ptr[i] < min_v) min_v = out_ptr[i];
        if (out_ptr[i] > max_v) max_v = out_ptr[i];
    }
    const bool is_tanh_range = (min_v < -0.05f && max_v <= 1.5f);
    const float out_scale = (max_v <= 1.5f) ? 255.0f : 1.0f;

    std::vector<uint8_t> out_rgb(3ULL * kPlane);
    const float* o0 = out_ptr;
    const float* o1 = out_ptr + kPlane;
    const float* o2 = out_ptr + 2ULL * kPlane;

    for (size_t i = 0; i < kPlane; ++i) {
        uint8_t* dst = out_rgb.data() + i * 3ULL;
        const float r_val = is_tanh_range ? ((o0[i] + 1.0f) * 127.5f) : (o0[i] * out_scale);
        const float g_val = is_tanh_range ? ((o1[i] + 1.0f) * 127.5f) : (o1[i] * out_scale);
        const float b_val = is_tanh_range ? ((o2[i] + 1.0f) * 127.5f) : (o2[i] * out_scale);
        dst[0] = static_cast<uint8_t>(clampi(static_cast<int>(std::lround(r_val)), 0, 255));
        dst[1] = static_cast<uint8_t>(clampi(static_cast<int>(std::lround(g_val)), 0, 255));
        dst[2] = static_cast<uint8_t>(clampi(static_cast<int>(std::lround(b_val)), 0, 255));
    }
    return out_rgb;
}

std::vector<uint8_t> LamaInpainter::run_tile_512(
    const uint8_t* tile_rgb,
    const uint8_t* tile_mask,
    const ResolvedLamaModel& resolved,
    bool force_cpu) {
    constexpr size_t kPlane = static_cast<size_t>(kTileSize) * static_cast<size_t>(kTileSize);
    std::vector<float> img_tensor(3ULL * kPlane);
    std::vector<float> mask_tensor(kPlane);

    static constexpr float kInv255 = 1.0f / 255.0f;
    float* ch0 = img_tensor.data();
    float* ch1 = img_tensor.data() + kPlane;
    float* ch2 = img_tensor.data() + 2ULL * kPlane;

    for (size_t i = 0; i < kPlane; ++i) {
        const uint8_t* px = tile_rgb + i * 3ULL;
        ch0[i] = static_cast<float>(px[0]) * kInv255;
        ch1[i] = static_cast<float>(px[1]) * kInv255;
        ch2[i] = static_cast<float>(px[2]) * kInv255;
        mask_tensor[i] = (tile_mask[i] > 127) ? 1.0f : 0.0f;
    }

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 4> img_shape = {1, 3, kTileSize, kTileSize};
    const std::array<int64_t, 4> mask_shape = {1, 1, kTileSize, kTileSize};

    auto run_once = [&](OrtSessionHandle& handle) -> std::vector<Ort::Value> {
        std::array<Ort::Value, 2> inputs = {
            Ort::Value::CreateTensor<float>(mem_info, img_tensor.data(), img_tensor.size(), img_shape.data(), img_shape.size()),
            Ort::Value::CreateTensor<float>(mem_info, mask_tensor.data(), mask_tensor.size(), mask_shape.data(), mask_shape.size()),
        };
        const char* in_names[] = {"image", "mask"};
        const char* out_names[] = {"output"};
        const bool is_cuda = (handle.active_provider == "CUDAExecutionProvider");
        Ort::RunOptions ro = make_shrink_run_options(true, is_cuda);
        return handle.session->Run(
            ro, in_names, inputs.data(), inputs.size(), out_names, 1);
    };

    std::vector<Ort::Value> outputs;
    {
        OrtSessionHandle& handle = ensure_session(resolved, force_cpu);
        try {
            outputs = run_once(handle);
        } catch (const Ort::Exception&) {
            if (handle.active_provider == "CPUExecutionProvider") {
                throw;
            }
            OrtSessionHandle& cpu_handle = ensure_session(resolved, /*force_cpu=*/true);
            outputs = run_once(cpu_handle);
        }
    }

    const float* out_ptr = outputs[0].GetTensorData<float>();
    const size_t elem_cnt = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    if (elem_cnt < 3ULL * kPlane) {
        throw std::runtime_error("Unexpected LaMa ONNX output tensor element count.");
    }

    // Auto-detect output scale ([0, 255] standard vs [0, 1] fallback)
    float min_v = 0.0f;
    float max_v = 0.0f;
    for (size_t i = 0; i < 3ULL * kPlane; i += 64) {
        if (out_ptr[i] < min_v) min_v = out_ptr[i];
        if (out_ptr[i] > max_v) max_v = out_ptr[i];
    }
    const bool is_tanh_range = (min_v < -0.05f && max_v <= 1.5f);
    const float out_scale = (max_v <= 1.5f) ? 255.0f : 1.0f;

    std::vector<uint8_t> out_rgb(3ULL * kPlane);
    const float* o0 = out_ptr;
    const float* o1 = out_ptr + kPlane;
    const float* o2 = out_ptr + 2ULL * kPlane;

    for (size_t i = 0; i < kPlane; ++i) {
        uint8_t* dst = out_rgb.data() + i * 3ULL;
        const float r_val = is_tanh_range ? ((o0[i] + 1.0f) * 127.5f) : (o0[i] * out_scale);
        const float g_val = is_tanh_range ? ((o1[i] + 1.0f) * 127.5f) : (o1[i] * out_scale);
        const float b_val = is_tanh_range ? ((o2[i] + 1.0f) * 127.5f) : (o2[i] * out_scale);
        dst[0] = static_cast<uint8_t>(clampi(static_cast<int>(std::lround(r_val)), 0, 255));
        dst[1] = static_cast<uint8_t>(clampi(static_cast<int>(std::lround(g_val)), 0, 255));
        dst[2] = static_cast<uint8_t>(clampi(static_cast<int>(std::lround(b_val)), 0, 255));
    }
    return out_rgb;
}

bool LamaInpainter::inpaint_roi(
    std::vector<uint8_t>& canvas,
    int orig_w,
    int orig_h,
    const std::vector<uint8_t>& full_mask,
    const std::array<int, 4>& bbox,
    int padding,
    int tile_size,
    const ResolvedLamaModel& resolved,
    bool force_cpu,
    bool enable_tone_match,
    bool* tone_match_applied_out) {
    const int win_size = clampi(tile_size, 256, 1024);
    const int x1 = clampi(bbox[0] - padding, 0, orig_w);
    const int y1 = clampi(bbox[1] - padding, 0, orig_h);
    const int x2 = clampi(bbox[2] + padding, 0, orig_w);
    const int y2 = clampi(bbox[3] + padding, 0, orig_h);
    const int roi_w = x2 - x1;
    const int roi_h = y2 - y1;
    if (roi_w <= 0 || roi_h <= 0) {
        return false;
    }

    // Check whether this ROI contains any active mask pixel
    bool has_mask_px = false;
    for (int y = y1; y < y2 && !has_mask_px; ++y) {
        const uint8_t* mrow = full_mask.data() + static_cast<size_t>(y) * static_cast<size_t>(orig_w);
        for (int x = x1; x < x2; ++x) {
            if (mrow[x] > 0) {
                has_mask_px = true;
                break;
            }
        }
    }
    if (!has_mask_px) {
        return false;
    }

    // Case 1: Padded ROI fits inside the configured context window (`win_size x win_size`;
    // 1:1 native 512x512 when win_size == 512, or resize-to-512-and-restore for custom tile sizes)
    if (roi_w <= win_size && roi_h <= win_size && orig_w >= win_size && orig_h >= win_size) {
        const int cx = (x1 + x2) / 2;
        const int cy = (y1 + y2) / 2;
        const int wx1 = clampi(cx - win_size / 2, 0, orig_w - win_size);
        const int wy1 = clampi(cy - win_size / 2, 0, orig_h - win_size);

        std::vector<uint8_t> tile_rgb(static_cast<size_t>(win_size) * static_cast<size_t>(win_size) * 3ULL);
        for (int ty = 0; ty < win_size; ++ty) {
            std::memcpy(
                tile_rgb.data() + static_cast<size_t>(ty) * static_cast<size_t>(win_size) * 3ULL,
                canvas.data() + (static_cast<size_t>(wy1 + ty) * static_cast<size_t>(orig_w) + static_cast<size_t>(wx1)) * 3ULL,
                static_cast<size_t>(win_size) * 3ULL);
        }

        // Restrict tile mask strictly to the current padded ROI
        std::vector<uint8_t> tile_mask(static_cast<size_t>(win_size) * static_cast<size_t>(win_size), 0);
        const int rel_x1 = x1 - wx1;
        const int rel_y1 = y1 - wy1;
        for (int ry = 0; ry < roi_h; ++ry) {
            std::memcpy(
                tile_mask.data() + static_cast<size_t>(rel_y1 + ry) * static_cast<size_t>(win_size) + static_cast<size_t>(rel_x1),
                full_mask.data() + static_cast<size_t>(y1 + ry) * static_cast<size_t>(orig_w) + static_cast<size_t>(x1),
                static_cast<size_t>(roi_w));
        }

        std::vector<uint8_t> inpainted_tile;
        if (win_size == kTileSize) {
            inpainted_tile = run_tile_512(tile_rgb.data(), tile_mask.data(), resolved, force_cpu);
        } else {
            const auto resized_rgb = resize_rgb_bilinear(tile_rgb.data(), win_size, win_size, kTileSize, kTileSize);
            const auto resized_mask = resize_mask_nearest(tile_mask.data(), win_size, win_size, kTileSize, kTileSize);
            const auto inpainted_512 = run_tile_512(resized_rgb.data(), resized_mask.data(), resolved, force_cpu);
            inpainted_tile = resize_rgb_bilinear(inpainted_512.data(), kTileSize, kTileSize, win_size, win_size);
        }

        if (enable_tone_match) {
            if (apply_tone_match(tile_rgb.data(), inpainted_tile, tile_mask.data(), win_size, win_size) &&
                tone_match_applied_out) {
                *tone_match_applied_out = true;
            }
        }

        // #4B: Discrete Screened Poisson Blending on Tier 3 LaMa patch (<= 512x512)
        // so gradient/color seams vanish cleanly before compositing onto canvas.
        if (math_inpaint::poissonBlendRegionSOR(
                tile_rgb.data(), inpainted_tile.data(), tile_mask.data(),
                win_size, win_size, 0, 0, win_size, win_size)) {
            inpainted_tile.swap(tile_rgb);
        }

        // Lossless composite: patch ONLY pixels where tile_mask > 0
        for (int ty = 0; ty < win_size; ++ty) {
            const uint8_t* mrow = tile_mask.data() + static_cast<size_t>(ty) * static_cast<size_t>(win_size);
            const uint8_t* srow = inpainted_tile.data() + static_cast<size_t>(ty) * static_cast<size_t>(win_size) * 3ULL;
            uint8_t* drow = canvas.data() +
                            (static_cast<size_t>(wy1 + ty) * static_cast<size_t>(orig_w) +
                             static_cast<size_t>(wx1)) * 3ULL;
            for (int tx = 0; tx < win_size; ++tx) {
                if (mrow[tx] > 0) {
                    drow[tx * 3 + 0] = srow[tx * 3 + 0];
                    drow[tx * 3 + 1] = srow[tx * 3 + 1];
                    drow[tx * 3 + 2] = srow[tx * 3 + 2];
                }
            }
        }
        return true;
    }

    // Case 2: Oversized ROI (>win_size) or small image (<win_size) -> resize ROI to 512x512 and restore
    std::vector<uint8_t> roi_rgb(static_cast<size_t>(roi_w) * static_cast<size_t>(roi_h) * 3ULL);
    std::vector<uint8_t> roi_mask(static_cast<size_t>(roi_w) * static_cast<size_t>(roi_h));
    for (int ry = 0; ry < roi_h; ++ry) {
        std::memcpy(
            roi_rgb.data() + static_cast<size_t>(ry) * static_cast<size_t>(roi_w) * 3ULL,
            canvas.data() + (static_cast<size_t>(y1 + ry) * static_cast<size_t>(orig_w) + static_cast<size_t>(x1)) * 3ULL,
            static_cast<size_t>(roi_w) * 3ULL);
        std::memcpy(
            roi_mask.data() + static_cast<size_t>(ry) * static_cast<size_t>(roi_w),
            full_mask.data() + static_cast<size_t>(y1 + ry) * static_cast<size_t>(orig_w) + static_cast<size_t>(x1),
            static_cast<size_t>(roi_w));
    }

    const std::vector<uint8_t> resized_rgb = resize_rgb_bilinear(roi_rgb.data(), roi_w, roi_h, kTileSize, kTileSize);
    const std::vector<uint8_t> resized_mask = resize_mask_nearest(roi_mask.data(), roi_w, roi_h, kTileSize, kTileSize);

    const std::vector<uint8_t> inpainted_512 =
        run_tile_512(resized_rgb.data(), resized_mask.data(), resolved, force_cpu);
    std::vector<uint8_t> restored_roi =
        resize_rgb_bilinear(inpainted_512.data(), kTileSize, kTileSize, roi_w, roi_h);

    if (enable_tone_match) {
        if (apply_tone_match(roi_rgb.data(), restored_roi, roi_mask.data(), roi_w, roi_h) &&
            tone_match_applied_out) {
            *tone_match_applied_out = true;
        }
    }

    for (int ry = 0; ry < roi_h; ++ry) {
        const uint8_t* mrow = roi_mask.data() + static_cast<size_t>(ry) * static_cast<size_t>(roi_w);
        const uint8_t* srow = restored_roi.data() + static_cast<size_t>(ry) * static_cast<size_t>(roi_w) * 3ULL;
        uint8_t* drow = canvas.data() +
                        (static_cast<size_t>(y1 + ry) * static_cast<size_t>(orig_w) +
                         static_cast<size_t>(x1)) * 3ULL;
        for (int rx = 0; rx < roi_w; ++rx) {
            if (mrow[rx] > 0) {
                drow[rx * 3 + 0] = srow[rx * 3 + 0];
                drow[rx * 3 + 1] = srow[rx * 3 + 1];
                drow[rx * 3 + 2] = srow[rx * 3 + 2];
            }
        }
    }
    return true;
}

InpaintResult LamaInpainter::inpaint_rgb(
    const uint8_t* rgb_data,
    int width,
    int height,
    const std::vector<uint8_t>& binary_mask,
    const InpaintOptions& options) {
    if (!rgb_data || width <= 0 || height <= 0) {
        throw std::invalid_argument("Invalid RGB image buffer or dimensions.");
    }
    if (binary_mask.size() < static_cast<size_t>(width) * static_cast<size_t>(height)) {
        throw std::invalid_argument("Binary mask size does not match image dimensions.");
    }

    const std::string requested_model = options.inpainting_model.empty()
        ? std::string("lama-manga")
        : options.inpainting_model;
    const bool wants_aot = (requested_model == "aot-inpainting");
    const bool wants_diffusion = (requested_model == "flux-2-klein" || requested_model == "rorem-mixed");

    const int eff_tile_size = clampi(options.tile_size, 256, 1024);

    const std::vector<std::array<int, 4>> mask_boxes =
        extract_bboxes_from_mask(binary_mask, width, height);
    const std::vector<std::array<int, 4>>& roi_boxes =
        (options.bboxes.has_value() && !options.bboxes->empty())
            ? *options.bboxes
            : mask_boxes;

    std::vector<uint8_t> working_mask = (options.refine_mask && !roi_boxes.empty())
        ? refine_bubble_mask(rgb_data, binary_mask, roi_boxes, width, height, options.dilate_kernel_size)
        : binary_mask;

    const size_t rgb_bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3ULL;
    std::vector<uint8_t> canvas(rgb_data, rgb_data + rgb_bytes);

    int tier1_blocks = 0;
    int tier2_blocks = 0;
    int tier3_blocks = 0;
    int tiles_count = 0;
    bool tone_match_applied = false;

    std::vector<std::array<int, 4>> neural_boxes;

    if (options.enable_3tier_inpaint) {
        // Immutable snapshot of all blocks' masks so ring sampling always excludes other blocks' masks
        const std::vector<uint8_t> all_blocks_mask = working_mask;

        constexpr int kRingPx = 4;
        constexpr int kBubbleCorePx = 3;
        constexpr size_t kMinRingPixels = 16;
        constexpr double kMaxMeanDev = 10.0;
        constexpr double kMaxOutlierDev = 64.0;
        constexpr int kTeleaCollar = 8;
        constexpr float kTeleaRadius = 5.0f;

        const std::vector<std::array<int, 4>>& block_list = !roi_boxes.empty() ? roi_boxes : mask_boxes;

        struct RingSample {
            int x;
            int y;
            uint8_t r;
            uint8_t g;
            uint8_t b;
        };

        const int dilate_pad = options.refine_mask
            ? std::max(0, options.dilate_kernel_size / 2)
            : 0;

        for (const auto& box : block_list) {
            const int bx1 = clampi(box[0] - dilate_pad, 0, width);
            const int by1 = clampi(box[1] - dilate_pad, 0, height);
            const int bx2 = clampi(box[2] + dilate_pad, 0, width);
            const int by2 = clampi(box[3] + dilate_pad, 0, height);
            if (bx2 <= bx1 || by2 <= by1) continue;

            bool has_fg = false;
            for (int y = by1; y < by2 && !has_fg; ++y) {
                const uint8_t* mrow = working_mask.data() + static_cast<size_t>(y) * static_cast<size_t>(width);
                for (int x = bx1; x < bx2; ++x) {
                    if (mrow[x] > 0) {
                        has_fg = true;
                        break;
                    }
                }
            }
            if (!has_fg) continue;

            // Local padded window for ring sampling + bubble wall erosion
            const int pad = kRingPx + kBubbleCorePx + 4;
            const int lx0 = clampi(bx1 - pad, 0, width);
            const int ly0 = clampi(by1 - pad, 0, height);
            const int lx1 = clampi(bx2 + pad, 0, width);
            const int ly1 = clampi(by2 + pad, 0, height);
            const int lw = lx1 - lx0;
            const int lh = ly1 - ly0;
            if (lw <= 0 || lh <= 0) {
                neural_boxes.push_back(box);
                continue;
            }

            const size_t ltotal = static_cast<size_t>(lw) * static_cast<size_t>(lh);
            std::vector<uint8_t> local_block_mask(ltotal, 0);
            for (int gy = by1; gy < by2; ++gy) {
                const uint8_t* mrow = working_mask.data() + static_cast<size_t>(gy) * static_cast<size_t>(width);
                uint8_t* lrow = local_block_mask.data() + static_cast<size_t>(gy - ly0) * static_cast<size_t>(lw);
                for (int gx = bx1; gx < bx2; ++gx) {
                    if (mrow[gx] > 0) {
                        lrow[gx - lx0] = 255;
                    }
                }
            }

            std::vector<float> dist_to_block_sq;
            if (!vellum::detector::computeEuclideanDistanceSqFH(local_block_mask.data(), lw, lh, dist_to_block_sq)) {
                continue;
            }

            // First estimate rough ring luma to handle both light and dark bubbles when detecting bubble walls
            std::array<int, 256> rough_luma_hist{};
            int rough_cnt = 0;
            const float ring_sq = static_cast<float>(kRingPx * kRingPx);
            for (int ly = 0; ly < lh; ++ly) {
                const int gy = ly0 + ly;
                const uint8_t* amrow = all_blocks_mask.data() + static_cast<size_t>(gy) * static_cast<size_t>(width);
                const uint8_t* crow = canvas.data() + static_cast<size_t>(gy) * static_cast<size_t>(width) * 3ULL;
                const size_t lrow = static_cast<size_t>(ly) * static_cast<size_t>(lw);
                for (int lx = 0; lx < lw; ++lx) {
                    const float d2 = dist_to_block_sq[lrow + static_cast<size_t>(lx)];
                    if (d2 <= 0.0f || d2 > ring_sq) continue;
                    const int gx = lx0 + lx;
                    if (amrow[gx] > 0) continue;
                    const uint8_t* px = crow + static_cast<size_t>(gx) * 3ULL;
                    ++rough_luma_hist[rgb_to_luma(px[0], px[1], px[2])];
                    ++rough_cnt;
                }
            }
            if (rough_cnt < static_cast<int>(kMinRingPixels)) {
                neural_boxes.push_back(box);
                continue;
            }
            uint8_t rough_median_luma = 255;
            {
                int acc = 0;
                const int target = rough_cnt / 2;
                for (int i = 0; i < 256; ++i) {
                    acc += rough_luma_hist[static_cast<size_t>(i)];
                    if (acc >= target) {
                        rough_median_luma = static_cast<uint8_t>(i);
                        break;
                    }
                }
            }

            // Detect bubble walls (dark/bright outline strokes & sharp edges outside all_blocks_mask)
            // and erode BUBBLE_CORE = 3 px inward from bubble walls
            std::vector<uint8_t> wall_mask(ltotal, 0);
            bool any_wall = false;
            const bool is_light_bubble = (rough_median_luma >= 128);
            for (int ly = 0; ly < lh; ++ly) {
                const int gy = ly0 + ly;
                const uint8_t* amrow = all_blocks_mask.data() + static_cast<size_t>(gy) * static_cast<size_t>(width);
                const uint8_t* crow = canvas.data() + static_cast<size_t>(gy) * static_cast<size_t>(width) * 3ULL;
                const size_t lrow = static_cast<size_t>(ly) * static_cast<size_t>(lw);
                for (int lx = 0; lx < lw; ++lx) {
                    const int gx = lx0 + lx;
                    if (amrow[gx] > 0) continue;
                    const uint8_t* px = crow + static_cast<size_t>(gx) * 3ULL;
                    const uint8_t luma = rgb_to_luma(px[0], px[1], px[2]);
                    bool is_wall = (is_light_bubble && luma < 160) || (!is_light_bubble && luma > 160);
                    if (!is_wall) {
                        static constexpr int kDx4[4] = {-1, 1, 0, 0};
                        static constexpr int kDy4[4] = {0, 0, -1, 1};
                        for (int k = 0; k < 4; ++k) {
                            const int ngx = gx + kDx4[k];
                            const int ngy = gy + kDy4[k];
                            if (ngx < 0 || ngx >= width || ngy < 0 || ngy >= height) continue;
                            if (all_blocks_mask[static_cast<size_t>(ngy) * static_cast<size_t>(width) + static_cast<size_t>(ngx)] > 0) {
                                continue;
                            }
                            const uint8_t* npx = canvas.data() +
                                                 (static_cast<size_t>(ngy) * static_cast<size_t>(width) + static_cast<size_t>(ngx)) * 3ULL;
                            const int diff = std::max({
                                std::abs(static_cast<int>(px[0]) - static_cast<int>(npx[0])),
                                std::abs(static_cast<int>(px[1]) - static_cast<int>(npx[1])),
                                std::abs(static_cast<int>(px[2]) - static_cast<int>(npx[2]))
                            });
                            if (diff > 48) {
                                is_wall = true;
                                break;
                            }
                        }
                    }
                    if (is_wall) {
                        wall_mask[lrow + static_cast<size_t>(lx)] = 255;
                        any_wall = true;
                    }
                }
            }

            std::vector<float> dist_to_wall_sq;
            const bool has_walls = any_wall &&
                                   vellum::detector::computeEuclideanDistanceSqFH(wall_mask.data(), lw, lh, dist_to_wall_sq);
            const float core_sq = static_cast<float>(kBubbleCorePx * kBubbleCorePx);

            // Sample valid exterior ring pixels (RING = 4 px, eroded BUBBLE_CORE = 3 px from walls,
            // excluding this block's mask and all other blocks' masks)
            std::vector<RingSample> samples;
            samples.reserve(static_cast<size_t>(rough_cnt));
            std::array<int, 256> r_hist{};
            std::array<int, 256> g_hist{};
            std::array<int, 256> b_hist{};

            for (int ly = 0; ly < lh; ++ly) {
                const int gy = ly0 + ly;
                const uint8_t* amrow = all_blocks_mask.data() + static_cast<size_t>(gy) * static_cast<size_t>(width);
                const uint8_t* crow = canvas.data() + static_cast<size_t>(gy) * static_cast<size_t>(width) * 3ULL;
                const size_t lrow = static_cast<size_t>(ly) * static_cast<size_t>(lw);
                for (int lx = 0; lx < lw; ++lx) {
                    const size_t lidx = lrow + static_cast<size_t>(lx);
                    const float d2 = dist_to_block_sq[lidx];
                    if (d2 <= 0.0f || d2 > ring_sq) continue;
                    const int gx = lx0 + lx;
                    if (amrow[gx] > 0) continue;
                    if (has_walls && dist_to_wall_sq[lidx] <= core_sq) continue;
                    const uint8_t* px = crow + static_cast<size_t>(gx) * 3ULL;
                    samples.push_back({gx, gy, px[0], px[1], px[2]});
                    ++r_hist[px[0]];
                    ++g_hist[px[1]];
                    ++b_hist[px[2]];
                }
            }

            const bool ring_is_clean_flat =
                (samples.size() >= kMinRingPixels) &&
                (samples.size() * 2ULL >= static_cast<size_t>(rough_cnt));

            if (ring_is_clean_flat) {
                auto hist_median = [&samples](const std::array<int, 256>& h) -> uint8_t {
                    size_t acc = 0;
                    const size_t target = samples.size() / 2;
                    for (int i = 0; i < 256; ++i) {
                        acc += static_cast<size_t>(h[static_cast<size_t>(i)]);
                        if (acc > target) return static_cast<uint8_t>(i);
                    }
                    return 255;
                };

                // Tier 1: Flat Median
                const uint8_t med_r = hist_median(r_hist);
                const uint8_t med_g = hist_median(g_hist);
                const uint8_t med_b = hist_median(b_hist);

                double dev_sum = 0.0;
                int max_dev = 0;
                for (const auto& s : samples) {
                    const int d = std::max({
                        std::abs(static_cast<int>(s.r) - static_cast<int>(med_r)),
                        std::abs(static_cast<int>(s.g) - static_cast<int>(med_g)),
                        std::abs(static_cast<int>(s.b) - static_cast<int>(med_b))
                    });
                    dev_sum += static_cast<double>(d);
                    if (d > max_dev) max_dev = d;
                }
                const double mean_dev = dev_sum / static_cast<double>(samples.size());

                if (mean_dev <= kMaxMeanDev && static_cast<double>(max_dev) < kMaxOutlierDev) {
                    for (int gy = by1; gy < by2; ++gy) {
                        uint8_t* mrow = working_mask.data() + static_cast<size_t>(gy) * static_cast<size_t>(width);
                        uint8_t* crow = canvas.data() + static_cast<size_t>(gy) * static_cast<size_t>(width) * 3ULL;
                        for (int gx = bx1; gx < bx2; ++gx) {
                            if (mrow[gx] > 0) {
                                crow[gx * 3 + 0] = med_r;
                                crow[gx * 3 + 1] = med_g;
                                crow[gx * 3 + 2] = med_b;
                                mrow[gx] = 0;
                            }
                        }
                    }
                    ++tier1_blocks;
                    continue;
                }

                // Tier 2: Linear Plane f_c(x, y) = c_bar + a_c*(x - x_bar) + b_c*(y - y_bar) + Telea FMM
                const double inv_n = 1.0 / static_cast<double>(samples.size());
                double mean_x = 0.0, mean_y = 0.0;
                double mean_c[3] = {0.0, 0.0, 0.0};
                for (const auto& s : samples) {
                    mean_x += static_cast<double>(s.x);
                    mean_y += static_cast<double>(s.y);
                    mean_c[0] += static_cast<double>(s.r);
                    mean_c[1] += static_cast<double>(s.g);
                    mean_c[2] += static_cast<double>(s.b);
                }
                mean_x *= inv_n;
                mean_y *= inv_n;
                mean_c[0] *= inv_n;
                mean_c[1] *= inv_n;
                mean_c[2] *= inv_n;

                double sxx = 0.0, sxy = 0.0, syy = 0.0;
                double sxc[3] = {0.0, 0.0, 0.0};
                double syc[3] = {0.0, 0.0, 0.0};
                for (const auto& s : samples) {
                    const double dx = static_cast<double>(s.x) - mean_x;
                    const double dy = static_cast<double>(s.y) - mean_y;
                    sxx += dx * dx;
                    sxy += dx * dy;
                    syy += dy * dy;
                    const double dc[3] = {
                        static_cast<double>(s.r) - mean_c[0],
                        static_cast<double>(s.g) - mean_c[1],
                        static_cast<double>(s.b) - mean_c[2]
                    };
                    for (int c = 0; c < 3; ++c) {
                        sxc[c] += dx * dc[c];
                        syc[c] += dy * dc[c];
                    }
                }

                const double det = sxx * syy - sxy * sxy;
                double ac[3] = {0.0, 0.0, 0.0};
                double bc[3] = {0.0, 0.0, 0.0};
                if (std::fabs(det) > 1.0e-6) {
                    const double inv_det = 1.0 / det;
                    for (int c = 0; c < 3; ++c) {
                        ac[c] = (syy * sxc[c] - sxy * syc[c]) * inv_det;
                        bc[c] = (sxx * syc[c] - sxy * sxc[c]) * inv_det;
                    }
                }

                double res_sum = 0.0;
                double max_res = 0.0;
                for (const auto& s : samples) {
                    const double dx = static_cast<double>(s.x) - mean_x;
                    const double dy = static_cast<double>(s.y) - mean_y;
                    const double pr = mean_c[0] + ac[0] * dx + bc[0] * dy;
                    const double pg = mean_c[1] + ac[1] * dx + bc[1] * dy;
                    const double pb = mean_c[2] + ac[2] * dx + bc[2] * dy;
                    const double r_err = std::max({
                        std::fabs(static_cast<double>(s.r) - pr),
                        std::fabs(static_cast<double>(s.g) - pg),
                        std::fabs(static_cast<double>(s.b) - pb)
                    });
                    res_sum += r_err;
                    if (r_err > max_res) max_res = r_err;
                }
                const double mean_res = res_sum * inv_n;

                if (mean_res <= kMaxMeanDev && max_res < kMaxOutlierDev) {
                    const int cx0 = clampi(bx1 - kTeleaCollar, 0, width);
                    const int cy0 = clampi(by1 - kTeleaCollar, 0, height);
                    const int cx1 = clampi(bx2 + kTeleaCollar, 0, width);
                    const int cy1 = clampi(by2 + kTeleaCollar, 0, height);
                    const int cw = cx1 - cx0;
                    const int ch = cy1 - cy0;
                    if (cw > 0 && ch > 0) {
                        const size_t ctotal = static_cast<size_t>(cw) * static_cast<size_t>(ch);
                        std::vector<uint8_t> crop_rgb(ctotal * 3ULL, 0);
                        std::vector<uint8_t> hole_mask(ctotal, 0);
                        std::vector<uint8_t> exclude_mask(ctotal, 0);

                        for (int cy = 0; cy < ch; ++cy) {
                            const int gy = cy0 + cy;
                            std::memcpy(
                                crop_rgb.data() + static_cast<size_t>(cy) * static_cast<size_t>(cw) * 3ULL,
                                canvas.data() + (static_cast<size_t>(gy) * static_cast<size_t>(width) + static_cast<size_t>(cx0)) * 3ULL,
                                static_cast<size_t>(cw) * 3ULL);
                            const uint8_t* wmrow = working_mask.data() + static_cast<size_t>(gy) * static_cast<size_t>(width);
                            const uint8_t* amrow = all_blocks_mask.data() + static_cast<size_t>(gy) * static_cast<size_t>(width);
                            for (int cx = 0; cx < cw; ++cx) {
                                const int gx = cx0 + cx;
                                const size_t cidx = static_cast<size_t>(cy) * static_cast<size_t>(cw) + static_cast<size_t>(cx);
                                const bool in_box = (gx >= bx1 && gx < bx2 && gy >= by1 && gy < by2);
                                if (in_box && wmrow[gx] > 0) {
                                    hole_mask[cidx] = 255;
                                } else if (amrow[gx] > 0 && wmrow[gx] > 0) {
                                    exclude_mask[cidx] = 255;
                                }
                            }
                        }

                        telea_inpaint_crop(crop_rgb, hole_mask, exclude_mask, cw, ch, kTeleaRadius);

                        for (int gy = by1; gy < by2; ++gy) {
                            uint8_t* wmrow = working_mask.data() + static_cast<size_t>(gy) * static_cast<size_t>(width);
                            uint8_t* drow = canvas.data() + static_cast<size_t>(gy) * static_cast<size_t>(width) * 3ULL;
                            const uint8_t* srow = crop_rgb.data() + static_cast<size_t>(gy - cy0) * static_cast<size_t>(cw) * 3ULL;
                            for (int gx = bx1; gx < bx2; ++gx) {
                                if (wmrow[gx] > 0) {
                                    const int lx = gx - cx0;
                                    drow[gx * 3 + 0] = srow[lx * 3 + 0];
                                    drow[gx * 3 + 1] = srow[lx * 3 + 1];
                                    drow[gx * 3 + 2] = srow[lx * 3 + 2];
                                    wmrow[gx] = 0;
                                }
                            }
                        }
                        ++tier2_blocks;
                        continue;
                    }
                }
            }

            // #4A: Tier 2.5 CPU Periodic Screentone Synthesis (2D Autocorrelation Lattice)
            // Phase-locked halftone/hatching fill keeps GPU sessions idle when periodic.
            if (math_inpaint::synthesizePeriodicScreentone(
                    canvas.data(), working_mask.data(), all_blocks_mask.data(),
                    width, height, bx1, by1, bx2, by2)) {
                for (int gy = by1; gy < by2; ++gy) {
                    uint8_t* wmrow = working_mask.data() + static_cast<size_t>(gy) * static_cast<size_t>(width);
                    for (int gx = bx1; gx < bx2; ++gx) {
                        wmrow[gx] = 0;
                    }
                }
                ++tier2_blocks;
                continue;
            }

            // #12: Criminisi Isophote-Driven Exemplar + PatchMatch for small structured
            // non-periodic holes (speedlines/hatching <= 4096 px^2, or CPU-only fallback).
            if (math_inpaint::inpaintCriminisiPatchMatch(
                    canvas.data(), working_mask.data(), all_blocks_mask.data(),
                    width, height, bx1, by1, bx2, by2,
                    /*patch_radius=*/3, /*max_hole_area=*/4096,
                    /*require_high_coherence=*/!options.force_cpu)) {
                for (int gy = by1; gy < by2; ++gy) {
                    uint8_t* wmrow = working_mask.data() + static_cast<size_t>(gy) * static_cast<size_t>(width);
                    for (int gx = bx1; gx < bx2; ++gx) {
                        wmrow[gx] = 0;
                    }
                }
                ++tier2_blocks;
                continue;
            }

            // Tier 3: Textured/screentone block requires neural inpainting
            neural_boxes.push_back(box);
        }

        // Check if any masked pixels remain outside block_list
        if (neural_boxes.empty()) {
            const auto leftover_boxes = extract_bboxes_from_mask(working_mask, width, height);
            for (const auto& lb : leftover_boxes) {
                neural_boxes.push_back(lb);
            }
        }

        // Tier 3 Lazy Bypass: if 100% of blocks finished in Tier 1-2, skip GPU/ONNX completely
        // while leaving any existing Ort::Session idle (no session destroy/recreate thrashing).
        if (neural_boxes.empty()) {
            InpaintResult res;
            res.width = width;
            res.height = height;
            res.tiles_processed = 0;
            res.tile_size = eff_tile_size;
            res.tier1_blocks = tier1_blocks;
            res.tier2_blocks = tier2_blocks;
            res.tier3_blocks = 0;
            res.gpu_skipped = true;
            res.tone_match_applied = false;
            res.execution_provider = "CPUExecutionProvider";
            res.model_id = requested_model;
            res.is_manga_lama = false;
            res.model_path = "";
            res.rgb = std::move(canvas);
            return res;
        }
    } else {
        neural_boxes = !roi_boxes.empty() ? roi_boxes : mask_boxes;
    }

    // Resolve ONNX weights lazily only when Tier 3 neural inpainting is actually needed
    std::optional<ResolvedAotModel> resolved_aot;
    if (wants_aot || wants_diffusion) {
        resolved_aot = model_manager_.resolve_aot_model();
    }

    const auto resolved = model_manager_.resolve_lama_model(
        (wants_aot || wants_diffusion) ? "lama-manga" : requested_model,
        /*allow_fallback=*/true);
    if (wants_aot && !resolved_aot.has_value() && !resolved.has_value()) {
        throw std::runtime_error(
            "Requested AOT Inpainting ONNX weights ('aot-inpainting.onnx') not found in " +
            util::path_to_utf8(model_manager_.models_dir()) + ".");
    }
    if (!wants_aot && !resolved.has_value() && !resolved_aot.has_value()) {
        throw std::runtime_error(
            "Requested Inpainting ONNX weights ('" + requested_model + "') not found in " +
            util::path_to_utf8(model_manager_.models_dir()) +
            ". Download or import local models via Settings -> Pipeline first.");
    }

    // Tier 3 uses 256px context windows when 3-tier CPU-first mode is enabled
    const int neural_tile_size = options.enable_3tier_inpaint ? 256 : eff_tile_size;

    ScopeInpainterUnload unload_guard{this, options.auto_unload};

    // Cluster nearby neural ROIs whose combined padded bounding box fits inside a single
    // `neural_tile_size x neural_tile_size` context window.
    std::vector<std::array<int, 4>> merged_rois;
    if (width >= neural_tile_size && height >= neural_tile_size && !neural_boxes.empty()) {
        struct PaddedRoi {
            int x1, y1, x2, y2;
        };
        std::vector<PaddedRoi> padded;
        padded.reserve(neural_boxes.size());
        for (const auto& box : neural_boxes) {
            const int x1 = clampi(box[0] - options.tile_padding, 0, width);
            const int y1 = clampi(box[1] - options.tile_padding, 0, height);
            const int x2 = clampi(box[2] + options.tile_padding, 0, width);
            const int y2 = clampi(box[3] + options.tile_padding, 0, height);
            if (x2 <= x1 || y2 <= y1) continue;

            bool has_mask = false;
            for (int y = y1; y < y2 && !has_mask; ++y) {
                const uint8_t* mrow = working_mask.data() + static_cast<size_t>(y) * static_cast<size_t>(width);
                for (int x = x1; x < x2; ++x) {
                    if (mrow[x] > 0) {
                        has_mask = true;
                        break;
                    }
                }
            }
            if (has_mask) {
                padded.push_back({x1, y1, x2, y2});
            }
        }

        std::sort(padded.begin(), padded.end(), [](const PaddedRoi& a, const PaddedRoi& b) {
            return (a.y1 != b.y1) ? (a.y1 < b.y1) : (a.x1 < b.x1);
        });

        std::vector<PaddedRoi> clusters;
        for (const auto& r : padded) {
            const int rw = r.x2 - r.x1;
            const int rh = r.y2 - r.y1;
            if (rw > neural_tile_size || rh > neural_tile_size) {
                clusters.push_back(r);
                continue;
            }
            int best_idx = -1;
            long long best_waste = std::numeric_limits<long long>::max();
            for (size_t ci = 0; ci < clusters.size(); ++ci) {
                const auto& c = clusters[ci];
                if ((c.x2 - c.x1) > neural_tile_size || (c.y2 - c.y1) > neural_tile_size) {
                    continue;
                }
                const int ux1 = std::min(c.x1, r.x1);
                const int uy1 = std::min(c.y1, r.y1);
                const int ux2 = std::max(c.x2, r.x2);
                const int uy2 = std::max(c.y2, r.y2);
                if ((ux2 - ux1) <= neural_tile_size && (uy2 - uy1) <= neural_tile_size) {
                    const long long union_area = static_cast<long long>(ux2 - ux1) * static_cast<long long>(uy2 - uy1);
                    const long long prev_area = static_cast<long long>(c.x2 - c.x1) * static_cast<long long>(c.y2 - c.y1);
                    const long long waste = union_area - prev_area;
                    if (waste < best_waste) {
                        best_waste = waste;
                        best_idx = static_cast<int>(ci);
                    }
                }
            }
            if (best_idx >= 0) {
                auto& c = clusters[static_cast<size_t>(best_idx)];
                c.x1 = std::min(c.x1, r.x1);
                c.y1 = std::min(c.y1, r.y1);
                c.x2 = std::max(c.x2, r.x2);
                c.y2 = std::max(c.y2, r.y2);
            } else {
                clusters.push_back(r);
            }
        }

        merged_rois.reserve(clusters.size());
        for (const auto& c : clusters) {
            merged_rois.push_back({c.x1, c.y1, c.x2, c.y2});
        }
    }

    tier3_blocks = static_cast<int>(neural_boxes.size());
    std::string provider_used = ModelManager::get_primary_provider(options.force_cpu);

    // Reusable scratch buffers hoisted across ROI tiles to avoid per-tile heap churn
    std::vector<uint8_t> scratch_roi_rgb;
    std::vector<uint8_t> scratch_roi_mask;
    std::vector<uint8_t> scratch_tile_rgb;
    std::vector<uint8_t> scratch_tile_mask;
    std::vector<uint8_t> scratch_resized_rgb;
    std::vector<uint8_t> scratch_resized_mask;
    std::vector<uint8_t> scratch_inpainted_512;
    std::vector<uint8_t> scratch_restored;

    // Helper for AOT or Diffusion ROI inpainting (strictly compositing ONLY working_mask > 0 pixels)
    auto run_custom_roi = [&](const std::array<int, 4>& box, int pad) -> bool {
        const int x1 = clampi(box[0] - pad, 0, width);
        const int y1 = clampi(box[1] - pad, 0, height);
        const int x2 = clampi(box[2] + pad, 0, width);
        const int y2 = clampi(box[3] + pad, 0, height);
        const int roi_w = x2 - x1;
        const int roi_h = y2 - y1;
        if (roi_w <= 0 || roi_h <= 0) return false;

        bool has_mask_px = false;
        for (int y = y1; y < y2 && !has_mask_px; ++y) {
            const uint8_t* mrow = working_mask.data() + static_cast<size_t>(y) * static_cast<size_t>(width);
            for (int x = x1; x < x2; ++x) {
                if (mrow[x] > 0) {
                    has_mask_px = true;
                    break;
                }
            }
        }
        if (!has_mask_px) return false;

        scratch_roi_rgb.resize(static_cast<size_t>(roi_w) * static_cast<size_t>(roi_h) * 3ULL);
        scratch_roi_mask.resize(static_cast<size_t>(roi_w) * static_cast<size_t>(roi_h));
        auto& roi_rgb = scratch_roi_rgb;
        auto& roi_mask = scratch_roi_mask;
        for (int ry = 0; ry < roi_h; ++ry) {
            std::memcpy(
                roi_rgb.data() + static_cast<size_t>(ry) * static_cast<size_t>(roi_w) * 3ULL,
                canvas.data() + (static_cast<size_t>(y1 + ry) * static_cast<size_t>(width) + static_cast<size_t>(x1)) * 3ULL,
                static_cast<size_t>(roi_w) * 3ULL);
            std::memcpy(
                roi_mask.data() + static_cast<size_t>(ry) * static_cast<size_t>(roi_w),
                working_mask.data() + static_cast<size_t>(y1 + ry) * static_cast<size_t>(width) + static_cast<size_t>(x1),
                static_cast<size_t>(roi_w));
        }

        // 1. If FLUX.2 Klein or RORem Mixed is selected, probe local diffusion HTTP adapter
        if (wants_diffusion) {
            for (const std::string& raw_ep : {options.inpaint_api_url, std::string("http://127.0.0.1:7860")}) {
                std::string host = "127.0.0.1";
                int port = 8188;
                std::string s = util::trim_ascii(raw_ep);
                if (s.rfind("http://", 0) == 0) s = s.substr(7);
                if (const auto slash = s.find('/'); slash != std::string::npos) s = s.substr(0, slash);
                if (const auto colon = s.rfind(':'); colon != std::string::npos) {
                    host = s.substr(0, colon);
                    try { port = std::stoi(s.substr(colon + 1)); } catch (...) {}
                } else if (!s.empty()) {
                    host = s;
                }
                try {
                    httplib::Client cli(host, port);
                    cli.set_connection_timeout(0, 350000);
                    cli.set_read_timeout(30, 0);
                    const auto img_png = encode_rgb_png(roi_rgb, roi_w, roi_h);
                    const auto mask_png = TextDetector::encode_mask_png(roi_mask, roi_w, roi_h);
                    const nlohmann::json req_json = {
                        {"model", requested_model},
                        {"init_images", nlohmann::json::array({util::base64_encode(img_png.data(), img_png.size())})},
                        {"mask", util::base64_encode(mask_png.data(), mask_png.size())},
                        {"width", roi_w},
                        {"height", roi_h},
                        {"denoising_strength", 0.65},
                        {"steps", 8},
                    };
                    auto resp = cli.Post("/sdapi/v1/img2img", req_json.dump(), "application/json");
                    if (resp && resp->status >= 200 && resp->status < 300) {
                        const auto body = nlohmann::json::parse(resp->body, nullptr, false);
                        if (body.is_object() && body.contains("images") && body["images"].is_array() && !body["images"].empty()) {
                            cli.Post("/sdapi/v1/unload-checkpoint", "{}", "application/json");
                            cli.Post("/free", R"({"unload_models":true,"free_memory":true})", "application/json");
                        }
                    }
                } catch (...) {
                }
            }
        }

        // 2. Run AOT Inpainting ONNX ([1, 4, 512, 512] -> [1, 3, 512, 512]) when available
        if ((wants_aot || wants_diffusion) && resolved_aot.has_value()) {
            const int win_size = clampi(neural_tile_size, 256, 1024);
            if (roi_w <= win_size && roi_h <= win_size && width >= win_size && height >= win_size) {
                const int cx = (x1 + x2) / 2;
                const int cy = (y1 + y2) / 2;
                const int wx1 = clampi(cx - win_size / 2, 0, width - win_size);
                const int wy1 = clampi(cy - win_size / 2, 0, height - win_size);

                scratch_tile_rgb.resize(static_cast<size_t>(win_size) * static_cast<size_t>(win_size) * 3ULL);
                auto& tile_rgb = scratch_tile_rgb;
                for (int ty = 0; ty < win_size; ++ty) {
                    std::memcpy(
                        tile_rgb.data() + static_cast<size_t>(ty) * static_cast<size_t>(win_size) * 3ULL,
                        canvas.data() + (static_cast<size_t>(wy1 + ty) * static_cast<size_t>(width) + static_cast<size_t>(wx1)) * 3ULL,
                        static_cast<size_t>(win_size) * 3ULL);
                }

                scratch_tile_mask.assign(static_cast<size_t>(win_size) * static_cast<size_t>(win_size), 0);
                auto& tile_mask = scratch_tile_mask;
                const int rel_x1 = x1 - wx1;
                const int rel_y1 = y1 - wy1;
                for (int ry = 0; ry < roi_h; ++ry) {
                    std::memcpy(
                        tile_mask.data() + static_cast<size_t>(rel_y1 + ry) * static_cast<size_t>(win_size) + static_cast<size_t>(rel_x1),
                        working_mask.data() + static_cast<size_t>(y1 + ry) * static_cast<size_t>(width) + static_cast<size_t>(x1),
                        static_cast<size_t>(roi_w));
                }

                if (win_size == kTileSize) {
                    scratch_restored = run_aot_tile_512(tile_rgb.data(), tile_mask.data(), *resolved_aot, options.force_cpu);
                } else {
                    scratch_resized_rgb = resize_rgb_bilinear(tile_rgb.data(), win_size, win_size, kTileSize, kTileSize);
                    scratch_resized_mask = resize_mask_nearest(tile_mask.data(), win_size, win_size, kTileSize, kTileSize);
                    scratch_inpainted_512 = run_aot_tile_512(scratch_resized_rgb.data(), scratch_resized_mask.data(), *resolved_aot, options.force_cpu);
                    scratch_restored = resize_rgb_bilinear(scratch_inpainted_512.data(), kTileSize, kTileSize, win_size, win_size);
                }
                auto& inpainted_tile = scratch_restored;

                if (options.enable_tone_match) {
                    if (apply_tone_match(tile_rgb.data(), inpainted_tile, tile_mask.data(), win_size, win_size)) {
                        tone_match_applied = true;
                    }
                }

                if (math_inpaint::poissonBlendRegionSOR(
                        tile_rgb.data(), inpainted_tile.data(), tile_mask.data(),
                        win_size, win_size, 0, 0, win_size, win_size)) {
                    inpainted_tile.swap(tile_rgb);
                }

                for (int ty = 0; ty < win_size; ++ty) {
                    const uint8_t* mrow = tile_mask.data() + static_cast<size_t>(ty) * static_cast<size_t>(win_size);
                    const uint8_t* srow = inpainted_tile.data() + static_cast<size_t>(ty) * static_cast<size_t>(win_size) * 3ULL;
                    uint8_t* drow = canvas.data() + (static_cast<size_t>(wy1 + ty) * static_cast<size_t>(width) + static_cast<size_t>(wx1)) * 3ULL;
                    for (int tx = 0; tx < win_size; ++tx) {
                        if (mrow[tx] > 0) {
                            drow[tx * 3 + 0] = srow[tx * 3 + 0];
                            drow[tx * 3 + 1] = srow[tx * 3 + 1];
                            drow[tx * 3 + 2] = srow[tx * 3 + 2];
                        }
                    }
                }
                return true;
            }

            scratch_resized_rgb = resize_rgb_bilinear(roi_rgb.data(), roi_w, roi_h, kTileSize, kTileSize);
            scratch_resized_mask = resize_mask_nearest(roi_mask.data(), roi_w, roi_h, kTileSize, kTileSize);
            scratch_inpainted_512 = run_aot_tile_512(scratch_resized_rgb.data(), scratch_resized_mask.data(), *resolved_aot, options.force_cpu);
            scratch_restored = resize_rgb_bilinear(scratch_inpainted_512.data(), kTileSize, kTileSize, roi_w, roi_h);
            auto& restored = scratch_restored;

            if (options.enable_tone_match) {
                if (apply_tone_match(roi_rgb.data(), restored, roi_mask.data(), roi_w, roi_h)) {
                    tone_match_applied = true;
                }
            }

            if (math_inpaint::poissonBlendRegionSOR(
                    roi_rgb.data(), restored.data(), roi_mask.data(),
                    roi_w, roi_h, 0, 0, roi_w, roi_h)) {
                restored.swap(roi_rgb);
            }

            for (int ry = 0; ry < roi_h; ++ry) {
                const uint8_t* mrow = roi_mask.data() + static_cast<size_t>(ry) * static_cast<size_t>(roi_w);
                const uint8_t* srow = restored.data() + static_cast<size_t>(ry) * static_cast<size_t>(roi_w) * 3ULL;
                uint8_t* drow = canvas.data() + (static_cast<size_t>(y1 + ry) * static_cast<size_t>(width) + static_cast<size_t>(x1)) * 3ULL;
                for (int rx = 0; rx < roi_w; ++rx) {
                    if (mrow[rx] > 0) {
                        drow[rx * 3 + 0] = srow[rx * 3 + 0];
                        drow[rx * 3 + 1] = srow[rx * 3 + 1];
                        drow[rx * 3 + 2] = srow[rx * 3 + 2];
                    }
                }
            }
            return true;
        }

        if (resolved.has_value()) {
            return inpaint_roi(
                canvas, width, height, working_mask, box, pad, neural_tile_size,
                *resolved, options.force_cpu, options.enable_tone_match, &tone_match_applied);
        }
        return false;
    };

    if (!merged_rois.empty()) {
        for (const auto& box : merged_rois) {
            if (wants_aot || wants_diffusion) {
                if (run_custom_roi(box, 0)) ++tiles_count;
            } else if (resolved.has_value() && inpaint_roi(
                    canvas,
                    width,
                    height,
                    working_mask,
                    box,
                    /*padding=*/0,
                    neural_tile_size,
                    *resolved,
                    options.force_cpu,
                    options.enable_tone_match,
                    &tone_match_applied)) {
                ++tiles_count;
            }
        }
    } else {
        for (const auto& box : neural_boxes) {
            if (wants_aot || wants_diffusion) {
                if (run_custom_roi(box, options.tile_padding)) ++tiles_count;
            } else if (resolved.has_value() && inpaint_roi(
                    canvas,
                    width,
                    height,
                    working_mask,
                    box,
                    options.tile_padding,
                    neural_tile_size,
                    *resolved,
                    options.force_cpu,
                    options.enable_tone_match,
                    &tone_match_applied)) {
                ++tiles_count;
            }
        }
    }

    if (tiles_count == 0) {
        // No neural inference actually executed -> do not thrash any existing Ort::Session
        unload_guard.enabled = false;
    } else if (cached_aot_session_.has_value()) {
        provider_used = cached_aot_session_->active_provider;
    } else if (cached_session_.has_value()) {
        provider_used = cached_session_->active_provider;
    }

    if (tiles_count > 0 &&
        (wants_diffusion || query_system_ram_stats().ram_load_pct >= kRamPressureGuardPct)) {
        unload();
        trim_process_working_set_throttled(15);
    }

    InpaintResult res;
    res.width = width;
    res.height = height;
    res.tiles_processed = tiles_count;
    res.tile_size = neural_tile_size;
    res.tier1_blocks = tier1_blocks;
    res.tier2_blocks = tier2_blocks;
    res.tier3_blocks = (tiles_count > 0) ? tier3_blocks : 0;
    res.gpu_skipped = (tiles_count == 0);
    res.tone_match_applied = tone_match_applied;
    res.execution_provider = (tiles_count == 0) ? std::string("CPUExecutionProvider") : provider_used;
    res.model_id = (wants_aot || wants_diffusion)
        ? requested_model
        : (resolved.has_value() ? resolved->model_id() : requested_model);
    res.is_manga_lama = resolved.has_value() ? resolved->is_manga_lama : false;
    res.model_path = (wants_aot && resolved_aot.has_value())
        ? util::path_to_utf8(resolved_aot->model_path)
        : (resolved.has_value() ? util::path_to_utf8(resolved->model_path) : "");
    res.rgb = std::move(canvas);
    return res;
}

}  // namespace mangaflow::pipeline
