#include "inpainter.hpp"
#include "detector.hpp"
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

struct StbiDeleter {
    void operator()(unsigned char* p) const noexcept {
        if (p) stbi_image_free(p);
    }
};
using StbiPtr = std::unique_ptr<unsigned char, StbiDeleter>;

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

std::vector<uint8_t> resize_rgb_bilinear(
    const uint8_t* src,
    int src_w,
    int src_h,
    int dst_w,
    int dst_h) {
    std::vector<uint8_t> dst(static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h) * 3ULL, 0);
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
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
    for (int x = 0; x < dst_w; ++x) {
        const float sx = clampf((static_cast<float>(x) + 0.5f) * scale_x - 0.5f, 0.0f, static_cast<float>(src_w - 1));
        const int x0 = static_cast<int>(std::floor(sx));
        x0_lut[static_cast<size_t>(x)] = x0;
        x1_lut[static_cast<size_t>(x)] = std::min(x0 + 1, src_w - 1);
        wx1_lut[static_cast<size_t>(x)] = sx - static_cast<float>(x0);
    }

    for (int y = 0; y < dst_h; ++y) {
        const float sy = clampf((static_cast<float>(y) + 0.5f) * scale_y - 0.5f, 0.0f, static_cast<float>(src_h - 1));
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
                out_px[c] = static_cast<uint8_t>(clampi(static_cast<int>(std::lround(val)), 0, 255));
            }
        }
    }
    return dst;
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
    return "data:image/png;base64," + TextDetector::base64_encode(png.data(), png.size());
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
        std::vector<uint8_t> ink(sub_total, 0);
        for (int y = 0; y < bh; ++y) {
            const uint8_t* srow = rgb_data +
                                  (static_cast<size_t>(y1 + y) * static_cast<size_t>(width) +
                                   static_cast<size_t>(x1)) * 3ULL;
            for (int x = 0; x < bw; ++x) {
                const uint8_t* px = srow + static_cast<size_t>(x) * 3ULL;
                if (rgb_to_luma(px[0], px[1], px[2]) < 210) {
                    ink[static_cast<size_t>(y) * static_cast<size_t>(bw) + static_cast<size_t>(x)] = 1;
                }
            }
        }

        // Flood-fill border-touching ink so speech bubble borders are preserved
        std::vector<uint8_t> border_ink(sub_total, 0);
        std::queue<int> q;
        auto mark_border = [&](int lx, int ly) {
            const size_t idx = static_cast<size_t>(ly) * static_cast<size_t>(bw) + static_cast<size_t>(lx);
            if (ink[idx] && !border_ink[idx]) {
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

        // Add all non-border interior text strokes & punctuation into `refined`
        for (int ly = 0; ly < bh; ++ly) {
            uint8_t* mrow = refined.data() + static_cast<size_t>(y1 + ly) * static_cast<size_t>(width) + static_cast<size_t>(x1);
            for (int lx = 0; lx < bw; ++lx) {
                const size_t idx = static_cast<size_t>(ly) * static_cast<size_t>(bw) + static_cast<size_t>(lx);
                if (ink[idx] && !border_ink[idx]) {
                    mrow[lx] = 255;
                }
            }
        }
    }

    if (dilate_kernel_size > 1) {
        dilate_mask_ellipse_inplace(refined, width, height, dilate_kernel_size);
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
    unload();
    cached_session_ = ModelManager::create_ort_session(resolved.model_path, force_cpu);
    cached_model_path_ = resolved.model_path;
    cached_force_cpu_ = force_cpu;
    return *cached_session_;
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
        return handle.session->Run(
            Ort::RunOptions{nullptr}, in_names, inputs.data(), inputs.size(), out_names, 1);
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
    float peak = 0.0f;
    for (size_t i = 0; i < 3ULL * kPlane; i += 64) {
        if (out_ptr[i] > peak) peak = out_ptr[i];
    }
    const float out_scale = (peak <= 1.5f) ? 255.0f : 1.0f;

    std::vector<uint8_t> out_rgb(3ULL * kPlane);
    const float* o0 = out_ptr;
    const float* o1 = out_ptr + kPlane;
    const float* o2 = out_ptr + 2ULL * kPlane;

    for (size_t i = 0; i < kPlane; ++i) {
        uint8_t* dst = out_rgb.data() + i * 3ULL;
        dst[0] = static_cast<uint8_t>(clampi(static_cast<int>(std::lround(o0[i] * out_scale)), 0, 255));
        dst[1] = static_cast<uint8_t>(clampi(static_cast<int>(std::lround(o1[i] * out_scale)), 0, 255));
        dst[2] = static_cast<uint8_t>(clampi(static_cast<int>(std::lround(o2[i] * out_scale)), 0, 255));
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
    bool force_cpu) {
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
    const std::vector<uint8_t> restored_roi =
        resize_rgb_bilinear(inpainted_512.data(), kTileSize, kTileSize, roi_w, roi_h);

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
    const auto resolved = model_manager_.resolve_lama_model(
        requested_model,
        options.allow_model_fallback);
    if (!resolved.has_value()) {
        throw std::runtime_error(
            "Requested LaMa Inpainting ONNX weights ('" + requested_model + "') not found in " +
            util::path_to_utf8(model_manager_.models_dir()) +
            ". Call POST /api/models/import-koharu or GET /api/models/download first.");
    }

    const int eff_tile_size = clampi(options.tile_size, 256, 1024);

    const std::vector<std::array<int, 4>> roi_boxes = options.bboxes.has_value()
        ? *options.bboxes
        : extract_bboxes_from_mask(binary_mask, width, height);

    const std::vector<uint8_t> working_mask = (options.refine_mask && !roi_boxes.empty())
        ? refine_bubble_mask(rgb_data, binary_mask, roi_boxes, width, height, options.dilate_kernel_size)
        : binary_mask;

    const size_t rgb_bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3ULL;
    std::vector<uint8_t> canvas(rgb_data, rgb_data + rgb_bytes);

    ScopeInpainterUnload unload_guard{this, options.auto_unload};

    int tiles_count = 0;
    std::string provider_used = ModelManager::get_primary_provider(options.force_cpu);

    for (const auto& box : roi_boxes) {
        if (inpaint_roi(
                canvas,
                width,
                height,
                working_mask,
                box,
                options.tile_padding,
                eff_tile_size,
                *resolved,
                options.force_cpu)) {
            ++tiles_count;
        }
    }
    if (cached_session_.has_value()) {
        provider_used = cached_session_->active_provider;
    }

    InpaintResult res;
    res.width = width;
    res.height = height;
    res.tiles_processed = tiles_count;
    res.tile_size = eff_tile_size;
    res.execution_provider = provider_used;
    res.model_id = resolved->model_id();
    res.is_koharu_manga = resolved->is_koharu_manga;
    res.model_path = util::path_to_utf8(resolved->model_path);
    res.rgb = std::move(canvas);
    return res;
}

}  // namespace mangaflow::pipeline
