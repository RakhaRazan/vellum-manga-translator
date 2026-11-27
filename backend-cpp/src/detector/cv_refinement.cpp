#include "cv_refinement.hpp"
#include "../pipeline/math_ocr_ops.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <numeric>
#include <unordered_map>
#include <utility>

namespace vellum::detector {

namespace {

inline uint8_t rgbToLuma(uint8_t r, uint8_t g, uint8_t b) noexcept {
    return static_cast<uint8_t>((299 * static_cast<int>(r) +
                                 587 * static_cast<int>(g) +
                                 114 * static_cast<int>(b) + 500) / 1000);
}

inline uint8_t sampleMaskAt(
    const std::vector<uint8_t>& mask,
    const RectI& rect,
    int x,
    int y) noexcept {
    if (!rect.contains(x, y)) return 0;
    const size_t idx = static_cast<size_t>(y - rect.y1) * static_cast<size_t>(rect.width()) +
                       static_cast<size_t>(x - rect.x1);
    return (idx < mask.size() && mask[idx] != 0) ? 1 : 0;
}

// Separable O(W * H) L_infinity (square) binary dilation of radius `r`.
// Input/output values are 0 or 1.
void dilateLInfBinary(
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

    // Horizontal pass
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

    // Vertical pass
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

// Separable O(W * H) L_infinity (square) binary erosion of radius `r`.
// Out-of-bounds pixels are treated as 0.
void erodeLInfBinary(
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
    if (w < need || h < need) {
        return;  // Entire buffer erodes to 0
    }

    std::vector<uint8_t> tmp(total, 0);

    // Horizontal pass
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

    // Vertical pass
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

// L_infinity morphological closing (dilate then erode) with zero-padding of `r`
// so foreground touching ROI edges does not get artificially eroded.
void closeLInfBinary(
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

    const int pw = w + 2 * r;
    const int ph = h + 2 * r;
    std::vector<uint8_t> padded(static_cast<size_t>(pw) * static_cast<size_t>(ph), 0);
    for (int y = 0; y < h; ++y) {
        const size_t src_row = static_cast<size_t>(y) * static_cast<size_t>(w);
        const size_t dst_row = static_cast<size_t>(y + r) * static_cast<size_t>(pw) + static_cast<size_t>(r);
        for (int x = 0; x < w; ++x) {
            if (src[src_row + static_cast<size_t>(x)] != 0) {
                padded[dst_row + static_cast<size_t>(x)] = 1;
            }
        }
    }

    std::vector<uint8_t> dilated;
    std::vector<uint8_t> eroded;
    dilateLInfBinary(padded, dilated, pw, ph, r);
    erodeLInfBinary(dilated, eroded, pw, ph, r);

    for (int y = 0; y < h; ++y) {
        const size_t src_row = static_cast<size_t>(y + r) * static_cast<size_t>(pw) + static_cast<size_t>(r);
        const size_t dst_row = static_cast<size_t>(y) * static_cast<size_t>(w);
        std::memcpy(dst.data() + dst_row, eroded.data() + src_row, static_cast<size_t>(w));
    }
}

// Euclidean L_2 (disk) binary dilation of radius `r`.
void dilateL2Binary(
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

    std::vector<std::pair<int, int>> offsets;
    const int r2 = r * r;
    offsets.reserve(static_cast<size_t>((2 * r + 1) * (2 * r + 1)));
    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            if (dx * dx + dy * dy <= r2) {
                offsets.emplace_back(dx, dy);
            }
        }
    }

    for (int y = 0; y < h; ++y) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
        for (int x = 0; x < w; ++x) {
            if (src[row + static_cast<size_t>(x)] == 0) continue;
            for (const auto& [dx, dy] : offsets) {
                const int nx = x + dx;
                const int ny = y + dy;
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    dst[static_cast<size_t>(ny) * static_cast<size_t>(w) + static_cast<size_t>(nx)] = 1;
                }
            }
        }
    }
}

// 4-connected interior hole flood-fill on a binary mask (0 or 1).
// If `restrict_rect` is non-empty, only holes inside `restrict_rect` (in local coordinates) are filled.
void fillInteriorHoles4(
    std::vector<uint8_t>& mask,
    int w,
    int h,
    const RectI* restrict_local_rect = nullptr) {
    if (w <= 2 || h <= 2) return;
    const size_t total = static_cast<size_t>(w) * static_cast<size_t>(h);
    if (mask.size() < total) return;

    std::vector<uint8_t> exterior(total, 0);
    std::vector<int> queue;
    queue.reserve(static_cast<size_t>(w + h) * 4ULL);

    auto pushIfZero = [&](int x, int y) {
        const int idx = y * w + x;
        if (mask[static_cast<size_t>(idx)] == 0 && exterior[static_cast<size_t>(idx)] == 0) {
            exterior[static_cast<size_t>(idx)] = 1;
            queue.push_back(idx);
        }
    };

    for (int x = 0; x < w; ++x) {
        pushIfZero(x, 0);
        pushIfZero(x, h - 1);
    }
    for (int y = 1; y < h - 1; ++y) {
        pushIfZero(0, y);
        pushIfZero(w - 1, y);
    }

    size_t head = 0;
    while (head < queue.size()) {
        const int idx = queue[head++];
        const int x = idx % w;
        const int y = idx / w;
        if (x > 0) pushIfZero(x - 1, y);
        if (x + 1 < w) pushIfZero(x + 1, y);
        if (y > 0) pushIfZero(x, y - 1);
        if (y + 1 < h) pushIfZero(x, y + 1);
    }

    if (restrict_local_rect) {
        const RectI r = restrict_local_rect->clamped(w, h);
        for (int y = r.y1; y < r.y2; ++y) {
            const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w);
            for (int x = r.x1; x < r.x2; ++x) {
                const size_t idx = row + static_cast<size_t>(x);
                if (mask[idx] == 0 && exterior[idx] == 0) {
                    mask[idx] = 1;
                }
            }
        }
    } else {
        for (size_t i = 0; i < total; ++i) {
            if (mask[i] == 0 && exterior[i] == 0) {
                mask[i] = 1;
            }
        }
    }
}

int medianFromHist256(const std::array<uint32_t, 256>& hist, uint32_t total) noexcept {
    if (total == 0) return 255;
    const uint32_t half = total / 2;
    uint32_t acc = 0;
    for (int i = 0; i < 256; ++i) {
        acc += hist[static_cast<size_t>(i)];
        if (acc > half) return i;
    }
    return 255;
}

struct DisjointSet {
    std::vector<int> parent;
    explicit DisjointSet(int n) : parent(static_cast<size_t>(n)) {
        std::iota(parent.begin(), parent.end(), 0);
    }
    int find(int x) {
        while (parent[static_cast<size_t>(x)] != x) {
            parent[static_cast<size_t>(x)] = parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
            x = parent[static_cast<size_t>(x)];
        }
        return x;
    }
    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a != b) parent[static_cast<size_t>(b)] = a;
    }
};

}  // namespace

// -----------------------------------------------------------------------------
// Public API Implementations
// -----------------------------------------------------------------------------

int scaledMorphRadius(double r, int page_w, int page_h) noexcept {
    const double long_side = static_cast<double>(std::max(page_w, page_h));
    const int val = static_cast<int>(std::lround((long_side / 1024.0) * r));
    return std::clamp(val, 1, 127);
}

std::vector<RawTextCandidate> enforceOneLabelPerQuery(
    const std::vector<RawTextCandidate>& candidates) {
    if (candidates.empty()) return {};

    // Map query_idx -> index in `kept`.
    std::unordered_map<int, size_t> query_to_kept;
    query_to_kept.reserve(candidates.size());

    std::vector<RawTextCandidate> kept;
    kept.reserve(candidates.size());

    for (const auto& c : candidates) {
        if (c.query_idx < 0) {
            kept.push_back(c);
            continue;
        }
        auto it = query_to_kept.find(c.query_idx);
        if (it == query_to_kept.end()) {
            query_to_kept.emplace(c.query_idx, kept.size());
            kept.push_back(c);
        } else {
            auto& existing = kept[it->second];
            // `text` (class_id == 0) always beats `onomatopoeia` (class_id == 1) on the same query.
            if (c.class_id == 0 && existing.class_id != 0) {
                existing = c;
            } else if (c.class_id == existing.class_id && c.score > existing.score) {
                existing = c;
            }
        }
    }

    return kept;
}

void mergeOverlappingSfxBlocks(
    std::vector<RefinementBlock>& blocks,
    float min_glyph_overlap) {
    std::vector<size_t> sfx_indices;
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (blocks[i].class_id == 1) {
            sfx_indices.push_back(i);
        }
    }
    if (sfx_indices.size() <= 1) return;

    const int m = static_cast<int>(sfx_indices.size());
    std::vector<int64_t> mask_areas(static_cast<size_t>(m), 0);
    for (int i = 0; i < m; ++i) {
        const auto& b = blocks[sfx_indices[static_cast<size_t>(i)]];
        int64_t cnt = 0;
        for (uint8_t v : b.raw_mask) {
            if (v != 0) ++cnt;
        }
        mask_areas[static_cast<size_t>(i)] = cnt;
    }

    DisjointSet dsu(m);
    for (int i = 0; i < m; ++i) {
        const auto& bi = blocks[sfx_indices[static_cast<size_t>(i)]];
        const int64_t area_i = mask_areas[static_cast<size_t>(i)];
        for (int j = i + 1; j < m; ++j) {
            const auto& bj = blocks[sfx_indices[static_cast<size_t>(j)]];
            const int64_t area_j = mask_areas[static_cast<size_t>(j)];

            float overlap = 0.0f;
            if (area_i > 0 && area_j > 0) {
                const RectI inter = bi.raw_mask_rect.intersected(bj.raw_mask_rect);
                if (!inter.empty()) {
                    int64_t inter_px = 0;
                    for (int y = inter.y1; y < inter.y2; ++y) {
                        const size_t ri = static_cast<size_t>(y - bi.raw_mask_rect.y1) *
                                          static_cast<size_t>(bi.raw_mask_rect.width());
                        const size_t rj = static_cast<size_t>(y - bj.raw_mask_rect.y1) *
                                          static_cast<size_t>(bj.raw_mask_rect.width());
                        for (int x = inter.x1; x < inter.x2; ++x) {
                            const uint8_t vi = bi.raw_mask[ri + static_cast<size_t>(x - bi.raw_mask_rect.x1)];
                            const uint8_t vj = bj.raw_mask[rj + static_cast<size_t>(x - bj.raw_mask_rect.x1)];
                            if (vi != 0 && vj != 0) ++inter_px;
                        }
                    }
                    const int64_t denom = std::min(area_i, area_j);
                    if (denom > 0) {
                        overlap = static_cast<float>(static_cast<double>(inter_px) / static_cast<double>(denom));
                    }
                }
            } else {
                // Fallback to bbox overlap if raw masks are empty.
                const RectI inter = bi.bbox.intersected(bj.bbox);
                const int64_t denom = std::min(bi.bbox.area(), bj.bbox.area());
                if (!inter.empty() && denom > 0) {
                    overlap = static_cast<float>(static_cast<double>(inter.area()) / static_cast<double>(denom));
                }
            }

            if (overlap >= min_glyph_overlap) {
                dsu.unite(i, j);
            }
        }
    }

    std::unordered_map<int, std::vector<size_t>> groups;
    for (int i = 0; i < m; ++i) {
        groups[dsu.find(i)].push_back(sfx_indices[static_cast<size_t>(i)]);
    }

    std::vector<uint8_t> consumed(blocks.size(), 0);
    std::vector<RefinementBlock> out;
    out.reserve(blocks.size());

    for (size_t idx = 0; idx < blocks.size(); ++idx) {
        if (blocks[idx].class_id != 1) {
            out.push_back(std::move(blocks[idx]));
            continue;
        }
        if (consumed[idx]) continue;

        // Find this SFX block's group
        int sfx_pos = -1;
        for (int i = 0; i < m; ++i) {
            if (sfx_indices[static_cast<size_t>(i)] == idx) {
                sfx_pos = i;
                break;
            }
        }
        const int root = dsu.find(sfx_pos);
        const auto& members = groups[root];
        for (size_t mem : members) consumed[mem] = 1;

        if (members.size() == 1) {
            out.push_back(std::move(blocks[idx]));
            continue;
        }

        // Merge member SFX blocks
        RefinementBlock merged = blocks[members[0]];
        RectI union_bbox = merged.bbox;
        RectI union_mrect = merged.raw_mask_rect;
        float max_score = merged.score;

        for (size_t k = 1; k < members.size(); ++k) {
            const auto& mb = blocks[members[k]];
            union_bbox = union_bbox.united(mb.bbox);
            union_mrect = union_mrect.united(mb.raw_mask_rect);
            if (mb.score > max_score) {
                max_score = mb.score;
                merged.query_idx = mb.query_idx;
            }
        }

        merged.bbox = union_bbox;
        merged.score = max_score;
        merged.is_vertical =
            union_bbox.height() > static_cast<int>(std::lround(static_cast<double>(union_bbox.width()) * 1.2));
        merged.raw_mask_rect = union_mrect;
        merged.raw_mask.assign(
            static_cast<size_t>(union_mrect.width()) * static_cast<size_t>(union_mrect.height()), 0);

        if (!union_mrect.empty()) {
            const int uw = union_mrect.width();
            for (size_t mem : members) {
                const auto& mb = blocks[mem];
                if (mb.raw_mask.empty() || mb.raw_mask_rect.empty()) continue;
                const int mw = mb.raw_mask_rect.width();
                for (int y = mb.raw_mask_rect.y1; y < mb.raw_mask_rect.y2; ++y) {
                    const size_t srow = static_cast<size_t>(y - mb.raw_mask_rect.y1) * static_cast<size_t>(mw);
                    const size_t drow = static_cast<size_t>(y - union_mrect.y1) * static_cast<size_t>(uw);
                    for (int x = mb.raw_mask_rect.x1; x < mb.raw_mask_rect.x2; ++x) {
                        if (mb.raw_mask[srow + static_cast<size_t>(x - mb.raw_mask_rect.x1)] != 0) {
                            merged.raw_mask[drow + static_cast<size_t>(x - union_mrect.x1)] = 255;
                        }
                    }
                }
            }
        }

        out.push_back(std::move(merged));
    }

    blocks = std::move(out);
}

uint8_t computeOtsuThreshold1Tile(
    const uint8_t* gray,
    int width,
    int height,
    double score_fraction) noexcept {
    if (!gray || width <= 0 || height <= 0) return 128;
    const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);

    std::array<uint32_t, 256> hist{};
    for (size_t i = 0; i < total; ++i) {
        ++hist[gray[i]];
    }

    double sum_all = 0.0;
    for (int i = 0; i < 256; ++i) {
        sum_all += static_cast<double>(i) * static_cast<double>(hist[static_cast<size_t>(i)]);
    }

    const double inv_n = 1.0 / static_cast<double>(total);
    std::array<double, 256> variances{};
    double max_var = -1.0;
    int best_t = 128;

    uint32_t w0_cnt = 0;
    double sum0 = 0.0;

    for (int t = 0; t < 255; ++t) {
        w0_cnt += hist[static_cast<size_t>(t)];
        if (w0_cnt == 0) continue;
        const uint32_t w1_cnt = static_cast<uint32_t>(total) - w0_cnt;
        if (w1_cnt == 0) break;

        sum0 += static_cast<double>(t) * static_cast<double>(hist[static_cast<size_t>(t)]);
        const double w0 = static_cast<double>(w0_cnt) * inv_n;
        const double w1 = static_cast<double>(w1_cnt) * inv_n;
        const double m0 = sum0 / static_cast<double>(w0_cnt);
        const double m1 = (sum_all - sum0) / static_cast<double>(w1_cnt);
        const double diff = m0 - m1;
        const double var_b = w0 * w1 * diff * diff;
        variances[static_cast<size_t>(t)] = var_b;
        if (var_b > max_var) {
            max_var = var_b;
            best_t = t;
        }
    }

    if (max_var <= 1e-6) {
        return 128;
    }

    const double sf = std::clamp(score_fraction, 0.0, 1.0);
    if (sf <= 0.0) {
        return static_cast<uint8_t>(best_t);
    }

    // Leptonica-style 1-tile Otsu with score_fraction:
    // select the lowest threshold `t <= best_t` with variance >= (1.0 - score_fraction) * max_var.
    const double min_score = (1.0 - sf) * max_var;
    int chosen_t = best_t;
    for (int t = 0; t <= best_t; ++t) {
        if (variances[static_cast<size_t>(t)] >= min_score) {
            chosen_t = t;
            break;
        }
    }
    return static_cast<uint8_t>(chosen_t);
}

std::vector<uint8_t> refineTextMasks(
    const uint8_t* rgb_data,
    int page_w,
    int page_h,
    std::vector<RefinementBlock>& blocks,
    const RefinementConfig& cfg) {
    if (!rgb_data || page_w <= 0 || page_h <= 0) {
        return {};
    }

    const size_t page_pixels = static_cast<size_t>(page_w) * static_cast<size_t>(page_h);
    std::vector<uint8_t> page_mask(page_pixels, 0);
    if (blocks.empty()) {
        return page_mask;
    }

    // Smallest-Box-Wins: sort indices from smallest bbox area to largest.
    const size_t n = blocks.size();
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        const int64_t area_a = blocks[a].bbox.area();
        const int64_t area_b = blocks[b].bbox.area();
        if (area_a != area_b) return area_a < area_b;
        if (blocks[a].score != blocks[b].score) return blocks[a].score > blocks[b].score;
        if (blocks[a].bbox.y1 != blocks[b].bbox.y1) return blocks[a].bbox.y1 < blocks[b].bbox.y1;
        return blocks[a].bbox.x1 < blocks[b].bbox.x1;
    });

    // Rank in smallest-to-largest processing order (smaller rank = processed earlier).
    std::vector<size_t> process_rank(n, 0);
    for (size_t r = 0; r < n; ++r) {
        process_rank[order[r]] = r;
    }

    // Global claimed occupancy bitmap so large boxes never steal glyphs from small adjacent notes.
    std::vector<uint8_t> claimed(page_pixels, 0);

    const int reach = scaledMorphRadius(cfg.reach_radius_1024, page_w, page_h);
    const int margin = scaledMorphRadius(cfg.l2_margin_1024, page_w, page_h);
    const int grow = std::max(0, cfg.final_grow_px);

    static constexpr int kDx8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static constexpr int kDy8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    static constexpr int kDx4[4] = {-1, 1, 0, 0};
    static constexpr int kDy4[4] = {0, 0, -1, 1};

    for (size_t rank = 0; rank < n; ++rank) {
        const size_t bi = order[rank];
        RefinementBlock& blk = blocks[bi];
        blk.refined_mask.clear();
        blk.refined_mask_rect = RectI{0, 0, 0, 0};

        const RectI core_box = blk.bbox.clamped(page_w, page_h);
        if (core_box.empty()) {
            continue;
        }

        const int bw = core_box.width();
        const int bh = core_box.height();
        const int line = std::max(1, std::min(bw, bh));

        // ---------------------------------------------------------------------
        // Stage 1: Directional Clip & Speck Filter
        // Expand `line = min(w, h)` along reading axis and `reach = scaled(6.0)` across it;
        // retain only 8-connected components touching the core box.
        // ---------------------------------------------------------------------
        const int pad_x = blk.is_vertical ? reach : line;
        const int pad_y = blk.is_vertical ? line : reach;
        const RectI lined_rect = core_box.expanded(pad_x, pad_y, page_w, page_h);
        const int lw = lined_rect.width();
        const int lh = lined_rect.height();
        const size_t ltotal = static_cast<size_t>(lw) * static_cast<size_t>(lh);

        std::vector<uint8_t> clipped(ltotal, 0);
        int clipped_count = 0;
        for (int ly = 0; ly < lh; ++ly) {
            const int gy = lined_rect.y1 + ly;
            const size_t page_row = static_cast<size_t>(gy) * static_cast<size_t>(page_w);
            const size_t lrow = static_cast<size_t>(ly) * static_cast<size_t>(lw);
            for (int lx = 0; lx < lw; ++lx) {
                const int gx = lined_rect.x1 + lx;
                if (claimed[page_row + static_cast<size_t>(gx)] != 0) continue;
                if (sampleMaskAt(blk.raw_mask, blk.raw_mask_rect, gx, gy) != 0) {
                    clipped[lrow + static_cast<size_t>(lx)] = 1;
                    ++clipped_count;
                }
            }
        }

        std::vector<uint8_t> stage1_mask(ltotal, 0);
        int stage1_count = 0;

        if (clipped_count > 0) {
            std::vector<uint8_t> visited(ltotal, 0);
            std::vector<int> queue;
            queue.reserve(256);

            for (int ly = 0; ly < lh; ++ly) {
                for (int lx = 0; lx < lw; ++lx) {
                    const int start_idx = ly * lw + lx;
                    if (clipped[static_cast<size_t>(start_idx)] == 0 ||
                        visited[static_cast<size_t>(start_idx)] != 0) {
                        continue;
                    }

                    queue.clear();
                    queue.push_back(start_idx);
                    visited[static_cast<size_t>(start_idx)] = 1;
                    bool touches_core = false;

                    size_t head = 0;
                    while (head < queue.size()) {
                        const int cur = queue[head++];
                        const int cx = cur % lw;
                        const int cy = cur / lw;
                        if (core_box.contains(lined_rect.x1 + cx, lined_rect.y1 + cy)) {
                            touches_core = true;
                        }
                        for (int k = 0; k < 8; ++k) {
                            const int nx = cx + kDx8[k];
                            const int ny = cy + kDy8[k];
                            if (nx < 0 || nx >= lw || ny < 0 || ny >= lh) continue;
                            const int nidx = ny * lw + nx;
                            if (clipped[static_cast<size_t>(nidx)] != 0 &&
                                visited[static_cast<size_t>(nidx)] == 0) {
                                visited[static_cast<size_t>(nidx)] = 1;
                                queue.push_back(nidx);
                            }
                        }
                    }

                    if (touches_core) {
                        for (int idx : queue) {
                            stage1_mask[static_cast<size_t>(idx)] = 1;
                        }
                        stage1_count += static_cast<int>(queue.size());
                    }
                }
            }
        }

        // Fallback if low-res instance mask missed tiny text inside core_box:
        // seed stage1_mask with unclaimed pixels of core_box so Stage 3 Otsu can still find ink.
        if (stage1_count == 0) {
            for (int gy = core_box.y1; gy < core_box.y2; ++gy) {
                const size_t page_row = static_cast<size_t>(gy) * static_cast<size_t>(page_w);
                const size_t lrow = static_cast<size_t>(gy - lined_rect.y1) * static_cast<size_t>(lw);
                for (int gx = core_box.x1; gx < core_box.x2; ++gx) {
                    if (claimed[page_row + static_cast<size_t>(gx)] == 0) {
                        stage1_mask[lrow + static_cast<size_t>(gx - lined_rect.x1)] = 1;
                        ++stage1_count;
                    }
                }
            }
            if (stage1_count == 0) {
                continue;
            }
        }

        // ---------------------------------------------------------------------
        // Stage 2: Solid Block Mask
        // L_infinity morphological closing (radius `reach`) + 4-connected interior hole flood-fill.
        // ---------------------------------------------------------------------
        std::vector<uint8_t> block_mask;
        closeLInfBinary(stage1_mask, block_mask, lw, lh, reach);
        fillInteriorHoles4(block_mask, lw, lh, nullptr);

        // Compute tight bounding box of block_mask in page coordinates.
        int bm_x1 = lined_rect.x2;
        int bm_y1 = lined_rect.y2;
        int bm_x2 = lined_rect.x1;
        int bm_y2 = lined_rect.y1;
        for (int ly = 0; ly < lh; ++ly) {
            const size_t lrow = static_cast<size_t>(ly) * static_cast<size_t>(lw);
            for (int lx = 0; lx < lw; ++lx) {
                if (block_mask[lrow + static_cast<size_t>(lx)] != 0) {
                    const int gx = lined_rect.x1 + lx;
                    const int gy = lined_rect.y1 + ly;
                    if (gx < bm_x1) bm_x1 = gx;
                    if (gx + 1 > bm_x2) bm_x2 = gx + 1;
                    if (gy < bm_y1) bm_y1 = gy;
                    if (gy + 1 > bm_y2) bm_y2 = gy + 1;
                }
            }
        }
        RectI block_bounds = (bm_x2 > bm_x1 && bm_y2 > bm_y1)
            ? RectI{bm_x1, bm_y1, bm_x2, bm_y2}.united(core_box)
            : core_box;

        // ---------------------------------------------------------------------
        // Stage 3: 1-Tile Otsu Ink Polarity & Stroke-Tip Recovery
        // ---------------------------------------------------------------------
        const RectI otsu_rect = block_bounds.expanded(reach, reach, page_w, page_h);
        const int ow = otsu_rect.width();
        const int oh = otsu_rect.height();
        const size_t ototal = static_cast<size_t>(ow) * static_cast<size_t>(oh);

        std::vector<uint8_t> otsu_gray(ototal, 0);
        for (int oy = 0; oy < oh; ++oy) {
            const int gy = otsu_rect.y1 + oy;
            const size_t rgb_row = static_cast<size_t>(gy) * static_cast<size_t>(page_w) * 3ULL;
            const size_t orow = static_cast<size_t>(oy) * static_cast<size_t>(ow);
            for (int ox = 0; ox < ow; ++ox) {
                const int gx = otsu_rect.x1 + ox;
                const uint8_t* px = rgb_data + rgb_row + static_cast<size_t>(gx) * 3ULL;
                otsu_gray[orow + static_cast<size_t>(ox)] = rgbToLuma(px[0], px[1], px[2]);
            }
        }

        const uint8_t otsu_thresh =
            computeOtsuThreshold1Tile(otsu_gray.data(), ow, oh, cfg.otsu_score_fraction);

        // Auto-detect dark-on-light vs light-on-dark ink polarity by comparing dark ratio
        // in `otsu_rect \ block_mask`.
        int ring_total = 0;
        int ring_dark = 0;
        for (int oy = 0; oy < oh; ++oy) {
            const int gy = otsu_rect.y1 + oy;
            const size_t orow = static_cast<size_t>(oy) * static_cast<size_t>(ow);
            for (int ox = 0; ox < ow; ++ox) {
                const int gx = otsu_rect.x1 + ox;
                if (sampleMaskAt(block_mask, lined_rect, gx, gy) != 0) continue;
                ++ring_total;
                if (otsu_gray[orow + static_cast<size_t>(ox)] <= otsu_thresh) {
                    ++ring_dark;
                }
            }
        }
        const double dark_ratio = (ring_total > 0)
            ? (static_cast<double>(ring_dark) / static_cast<double>(ring_total))
            : 0.0;
        const bool light_on_dark = (dark_ratio > 0.5);

        auto isInkLuma = [otsu_thresh, light_on_dark](uint8_t luma) noexcept -> bool {
            return light_on_dark ? (luma > otsu_thresh) : (luma <= otsu_thresh);
        };

        // Partition Otsu ink inside `otsu_rect` into `interior_ink` and `outer_ring_ink`.
        std::vector<uint8_t> interior_ink(ototal, 0);
        std::vector<uint8_t> outer_ring_ink(ototal, 0);
        for (int oy = 0; oy < oh; ++oy) {
            const int gy = otsu_rect.y1 + oy;
            const size_t page_row = static_cast<size_t>(gy) * static_cast<size_t>(page_w);
            const size_t orow = static_cast<size_t>(oy) * static_cast<size_t>(ow);
            for (int ox = 0; ox < ow; ++ox) {
                const int gx = otsu_rect.x1 + ox;
                if (claimed[page_row + static_cast<size_t>(gx)] != 0) continue;
                if (!isInkLuma(otsu_gray[orow + static_cast<size_t>(ox)])) continue;
                if (sampleMaskAt(block_mask, lined_rect, gx, gy) != 0) {
                    interior_ink[orow + static_cast<size_t>(ox)] = 1;
                } else {
                    outer_ring_ink[orow + static_cast<size_t>(ox)] = 1;
                }
            }
        }

        // Recover outer-ring ink components ONLY if they touch interior ink AND do NOT
        // touch the outer ring boundary (prevents leaking onto bubble borders).
        std::vector<uint8_t> otsu_glyph = interior_ink;
        {
            std::vector<uint8_t> visited(ototal, 0);
            std::vector<int> queue;
            queue.reserve(128);

            for (int oy = 0; oy < oh; ++oy) {
                for (int ox = 0; ox < ow; ++ox) {
                    const int start = oy * ow + ox;
                    if (outer_ring_ink[static_cast<size_t>(start)] == 0 ||
                        visited[static_cast<size_t>(start)] != 0) {
                        continue;
                    }

                    queue.clear();
                    queue.push_back(start);
                    visited[static_cast<size_t>(start)] = 1;
                    bool touches_interior = false;
                    bool touches_boundary = false;

                    size_t head = 0;
                    while (head < queue.size()) {
                        const int cur = queue[head++];
                        const int cx = cur % ow;
                        const int cy = cur / ow;
                        if (cx == 0 || cx == ow - 1 || cy == 0 || cy == oh - 1) {
                            touches_boundary = true;
                        }
                        for (int k = 0; k < 8; ++k) {
                            const int nx = cx + kDx8[k];
                            const int ny = cy + kDy8[k];
                            if (nx < 0 || nx >= ow || ny < 0 || ny >= oh) continue;
                            const int nidx = ny * ow + nx;
                            if (interior_ink[static_cast<size_t>(nidx)] != 0) {
                                touches_interior = true;
                            }
                            if (outer_ring_ink[static_cast<size_t>(nidx)] != 0 &&
                                visited[static_cast<size_t>(nidx)] == 0) {
                                visited[static_cast<size_t>(nidx)] = 1;
                                queue.push_back(nidx);
                            }
                        }
                    }

                    if (touches_interior && !touches_boundary) {
                        for (int idx : queue) {
                            otsu_glyph[static_cast<size_t>(idx)] = 1;
                        }
                    }
                }
            }
        }

        // #3: 2D Structure Tensor anisotropy filter — reject isotropic halftone screentone dots
        // (low coherence) while preserving directional text strokes (high coherence).
        mangaflow::pipeline::math_ocr::filterHalftoneDotsByStructureTensor(
            otsu_gray.data(), ow, oh, otsu_glyph, /*min_coherence=*/0.14f);

        // ---------------------------------------------------------------------
        // Stage 4: Plain-Paper Punctuation Recovery
        // If >= 99% of background non-ink pixels fall within +/- 32 RGB of median paper color,
        // recover missed ink components inside `lined_rect` with max(wc, hc) <= line
        // touching the dominant paper component 4-way.
        // ---------------------------------------------------------------------
        const RectI work_base = lined_rect.united(otsu_rect);
        const RectI work_rect = work_base.expanded(margin + grow, margin + grow, page_w, page_h);
        const int ww = work_rect.width();
        const int wh = work_rect.height();
        const size_t wtotal = static_cast<size_t>(ww) * static_cast<size_t>(wh);

        std::vector<uint8_t> glyph_work(wtotal, 0);
        for (int oy = 0; oy < oh; ++oy) {
            const size_t orow = static_cast<size_t>(oy) * static_cast<size_t>(ow);
            const size_t wrow = static_cast<size_t>(otsu_rect.y1 + oy - work_rect.y1) * static_cast<size_t>(ww);
            for (int ox = 0; ox < ow; ++ox) {
                if (otsu_glyph[orow + static_cast<size_t>(ox)] != 0) {
                    glyph_work[wrow + static_cast<size_t>(otsu_rect.x1 + ox - work_rect.x1)] = 1;
                }
            }
        }

        {
            std::array<uint32_t, 256> hist_r{}, hist_g{}, hist_b{};
            uint32_t bg_count = 0;
            for (int oy = 0; oy < oh; ++oy) {
                const int gy = otsu_rect.y1 + oy;
                const size_t rgb_row = static_cast<size_t>(gy) * static_cast<size_t>(page_w) * 3ULL;
                const size_t orow = static_cast<size_t>(oy) * static_cast<size_t>(ow);
                for (int ox = 0; ox < ow; ++ox) {
                    if (isInkLuma(otsu_gray[orow + static_cast<size_t>(ox)])) continue;
                    const int gx = otsu_rect.x1 + ox;
                    const uint8_t* px = rgb_data + rgb_row + static_cast<size_t>(gx) * 3ULL;
                    ++hist_r[px[0]];
                    ++hist_g[px[1]];
                    ++hist_b[px[2]];
                    ++bg_count;
                }
            }

            if (bg_count > 0) {
                const int med_r = medianFromHist256(hist_r, bg_count);
                const int med_g = medianFromHist256(hist_g, bg_count);
                const int med_b = medianFromHist256(hist_b, bg_count);
                const int tol = cfg.plain_paper_rgb_tol;

                uint32_t plain_count = 0;
                for (int oy = 0; oy < oh; ++oy) {
                    const int gy = otsu_rect.y1 + oy;
                    const size_t rgb_row = static_cast<size_t>(gy) * static_cast<size_t>(page_w) * 3ULL;
                    const size_t orow = static_cast<size_t>(oy) * static_cast<size_t>(ow);
                    for (int ox = 0; ox < ow; ++ox) {
                        if (isInkLuma(otsu_gray[orow + static_cast<size_t>(ox)])) continue;
                        const int gx = otsu_rect.x1 + ox;
                        const uint8_t* px = rgb_data + rgb_row + static_cast<size_t>(gx) * 3ULL;
                        if (std::abs(static_cast<int>(px[0]) - med_r) <= tol &&
                            std::abs(static_cast<int>(px[1]) - med_g) <= tol &&
                            std::abs(static_cast<int>(px[2]) - med_b) <= tol) {
                            ++plain_count;
                        }
                    }
                }

                const double plain_frac =
                    static_cast<double>(plain_count) / static_cast<double>(bg_count);

                if (plain_frac >= cfg.plain_paper_ratio) {
                    // Build paper_mask and missed_ink_mask over `lined_rect`.
                    std::vector<uint8_t> paper_mask(ltotal, 0);
                    std::vector<uint8_t> missed_ink(ltotal, 0);
                    for (int ly = 0; ly < lh; ++ly) {
                        const int gy = lined_rect.y1 + ly;
                        const size_t page_row = static_cast<size_t>(gy) * static_cast<size_t>(page_w);
                        const size_t rgb_row = page_row * 3ULL;
                        const size_t lrow = static_cast<size_t>(ly) * static_cast<size_t>(lw);
                        const size_t wrow = static_cast<size_t>(gy - work_rect.y1) * static_cast<size_t>(ww);
                        for (int lx = 0; lx < lw; ++lx) {
                            const int gx = lined_rect.x1 + lx;
                            const uint8_t* px = rgb_data + rgb_row + static_cast<size_t>(gx) * 3ULL;
                            const uint8_t luma = rgbToLuma(px[0], px[1], px[2]);
                            if (!isInkLuma(luma)) {
                                if (std::abs(static_cast<int>(px[0]) - med_r) <= tol &&
                                    std::abs(static_cast<int>(px[1]) - med_g) <= tol &&
                                    std::abs(static_cast<int>(px[2]) - med_b) <= tol) {
                                    paper_mask[lrow + static_cast<size_t>(lx)] = 1;
                                }
                            } else if (claimed[page_row + static_cast<size_t>(gx)] == 0 &&
                                       glyph_work[wrow + static_cast<size_t>(gx - work_rect.x1)] == 0) {
                                missed_ink[lrow + static_cast<size_t>(lx)] = 1;
                            }
                        }
                    }

                    // Label 4-connected paper components in `lined_rect` and pick dominant paper
                    // component (largest overlapping `core_box`, fallback to overall largest).
                    std::vector<int> paper_label(ltotal, 0);
                    std::vector<int> queue;
                    queue.reserve(512);
                    int next_label = 1;
                    int best_core_label = 0;
                    size_t best_core_size = 0;
                    int best_any_label = 0;
                    size_t best_any_size = 0;

                    for (int ly = 0; ly < lh; ++ly) {
                        for (int lx = 0; lx < lw; ++lx) {
                            const int start = ly * lw + lx;
                            if (paper_mask[static_cast<size_t>(start)] == 0 ||
                                paper_label[static_cast<size_t>(start)] != 0) {
                                continue;
                            }
                            const int lbl = next_label++;
                            queue.clear();
                            queue.push_back(start);
                            paper_label[static_cast<size_t>(start)] = lbl;
                            bool overlaps_core = false;

                            size_t head = 0;
                            while (head < queue.size()) {
                                const int cur = queue[head++];
                                const int cx = cur % lw;
                                const int cy = cur / lw;
                                if (core_box.contains(lined_rect.x1 + cx, lined_rect.y1 + cy)) {
                                    overlaps_core = true;
                                }
                                for (int k = 0; k < 4; ++k) {
                                    const int nx = cx + kDx4[k];
                                    const int ny = cy + kDy4[k];
                                    if (nx < 0 || nx >= lw || ny < 0 || ny >= lh) continue;
                                    const int nidx = ny * lw + nx;
                                    if (paper_mask[static_cast<size_t>(nidx)] != 0 &&
                                        paper_label[static_cast<size_t>(nidx)] == 0) {
                                        paper_label[static_cast<size_t>(nidx)] = lbl;
                                        queue.push_back(nidx);
                                    }
                                }
                            }

                            if (queue.size() > best_any_size) {
                                best_any_size = queue.size();
                                best_any_label = lbl;
                            }
                            if (overlaps_core && queue.size() > best_core_size) {
                                best_core_size = queue.size();
                                best_core_label = lbl;
                            }
                        }
                    }

                    const int dom_paper = (best_core_label != 0) ? best_core_label : best_any_label;

                    if (dom_paper != 0) {
                        // Find 8-connected components of missed ink inside `lined_rect`.
                        std::vector<uint8_t> visited(ltotal, 0);
                        for (int ly = 0; ly < lh; ++ly) {
                            for (int lx = 0; lx < lw; ++lx) {
                                const int start = ly * lw + lx;
                                if (missed_ink[static_cast<size_t>(start)] == 0 ||
                                    visited[static_cast<size_t>(start)] != 0) {
                                    continue;
                                }

                                queue.clear();
                                queue.push_back(start);
                                visited[static_cast<size_t>(start)] = 1;

                                int c_min_x = lx, c_max_x = lx;
                                int c_min_y = ly, c_max_y = ly;
                                bool touches_dom_paper_4 = false;
                                bool touches_lined_border = false;
                                bool all_inside_core = true;

                                size_t head = 0;
                                while (head < queue.size()) {
                                    const int cur = queue[head++];
                                    const int cx = cur % lw;
                                    const int cy = cur / lw;
                                    if (cx < c_min_x) c_min_x = cx;
                                    if (cx > c_max_x) c_max_x = cx;
                                    if (cy < c_min_y) c_min_y = cy;
                                    if (cy > c_max_y) c_max_y = cy;

                                    if (cx == 0 || cx == lw - 1 || cy == 0 || cy == lh - 1) {
                                        touches_lined_border = true;
                                    }
                                    if (!core_box.contains(lined_rect.x1 + cx, lined_rect.y1 + cy)) {
                                        all_inside_core = false;
                                    }

                                    // Check 4-way adjacency to dominant paper component
                                    for (int k = 0; k < 4; ++k) {
                                        const int nx = cx + kDx4[k];
                                        const int ny = cy + kDy4[k];
                                        if (nx < 0 || nx >= lw || ny < 0 || ny >= lh) continue;
                                        if (paper_label[static_cast<size_t>(ny * lw + nx)] == dom_paper) {
                                            touches_dom_paper_4 = true;
                                        }
                                    }

                                    // Expand 8-connected missed ink component
                                    for (int k = 0; k < 8; ++k) {
                                        const int nx = cx + kDx8[k];
                                        const int ny = cy + kDy8[k];
                                        if (nx < 0 || nx >= lw || ny < 0 || ny >= lh) continue;
                                        const int nidx = ny * lw + nx;
                                        if (missed_ink[static_cast<size_t>(nidx)] != 0 &&
                                            visited[static_cast<size_t>(nidx)] == 0) {
                                            visited[static_cast<size_t>(nidx)] = 1;
                                            queue.push_back(nidx);
                                        }
                                    }
                                }

                                const int wc = c_max_x - c_min_x + 1;
                                const int hc = c_max_y - c_min_y + 1;
                                const bool border_safe = (!touches_lined_border || all_inside_core);

                                if (std::max(wc, hc) <= line && touches_dom_paper_4 && border_safe) {
                                    for (int idx : queue) {
                                        const int cx = idx % lw;
                                        const int cy = idx / lw;
                                        const int wx = lined_rect.x1 + cx - work_rect.x1;
                                        const int wy = lined_rect.y1 + cy - work_rect.y1;
                                        glyph_work[static_cast<size_t>(wy) * static_cast<size_t>(ww) +
                                                   static_cast<size_t>(wx)] = 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // ---------------------------------------------------------------------
        // Stage 5: Hole Fill, L2 Dilation, Neighbor Clip & Final Grow
        // ---------------------------------------------------------------------

        // 5a. Fill closed glyph holes inside `bbox`
        const RectI core_local{
            core_box.x1 - work_rect.x1,
            core_box.y1 - work_rect.y1,
            core_box.x2 - work_rect.x1,
            core_box.y2 - work_rect.y1
        };
        fillInteriorHoles4(glyph_work, ww, wh, &core_local);

        // 5b. Euclidean L2 dilation (`margin = scaled(2.0)`)
        std::vector<uint8_t> dilated_l2;
        dilateL2Binary(glyph_work, dilated_l2, ww, wh, margin);

        // 5c. Clip against neighbor block bboxes
        for (size_t nb_idx = 0; nb_idx < n; ++nb_idx) {
            if (nb_idx == bi) continue;
            const RectI nb_box = blocks[nb_idx].bbox.clamped(page_w, page_h);
            const RectI inter = work_rect.intersected(nb_box);
            if (inter.empty()) continue;

            // If neighbor was processed earlier (smaller box), clip everywhere in neighbor's bbox.
            // If neighbor is larger (processed later), preserve pixels inside our own core_box
            // so a large enclosing box never erases a small nested note.
            const bool neighbor_has_priority = (process_rank[nb_idx] < rank);
            for (int gy = inter.y1; gy < inter.y2; ++gy) {
                const size_t wrow = static_cast<size_t>(gy - work_rect.y1) * static_cast<size_t>(ww);
                for (int gx = inter.x1; gx < inter.x2; ++gx) {
                    if (neighbor_has_priority || !core_box.contains(gx, gy)) {
                        dilated_l2[wrow + static_cast<size_t>(gx - work_rect.x1)] = 0;
                    }
                }
            }
        }

        // 5d. Final L_infinity dilation `GROW = 3` px
        std::vector<uint8_t> final_work;
        dilateLInfBinary(dilated_l2, final_work, ww, wh, grow);

        // Claim pixels via global `claimed` occupancy bitmap and write to `page_mask`.
        int r_min_x = work_rect.x2;
        int r_min_y = work_rect.y2;
        int r_max_x = work_rect.x1;
        int r_max_y = work_rect.y1;

        for (int wy = 0; wy < wh; ++wy) {
            const int gy = work_rect.y1 + wy;
            const size_t page_row = static_cast<size_t>(gy) * static_cast<size_t>(page_w);
            const size_t wrow = static_cast<size_t>(wy) * static_cast<size_t>(ww);
            for (int wx = 0; wx < ww; ++wx) {
                const size_t widx = wrow + static_cast<size_t>(wx);
                if (final_work[widx] == 0) continue;

                const int gx = work_rect.x1 + wx;
                const size_t pidx = page_row + static_cast<size_t>(gx);
                if (claimed[pidx] != 0) {
                    final_work[widx] = 0;  // Already claimed by a smaller box
                    continue;
                }
                claimed[pidx] = 1;
                page_mask[pidx] = 255;

                if (gx < r_min_x) r_min_x = gx;
                if (gx + 1 > r_max_x) r_max_x = gx + 1;
                if (gy < r_min_y) r_min_y = gy;
                if (gy + 1 > r_max_y) r_max_y = gy + 1;
            }
        }

        // Crop per-block refined mask tightly to its non-zero bounding box.
        if (r_max_x > r_min_x && r_max_y > r_min_y) {
            blk.refined_mask_rect = RectI{r_min_x, r_min_y, r_max_x, r_max_y};
            const int rw = blk.refined_mask_rect.width();
            const int rh = blk.refined_mask_rect.height();
            blk.refined_mask.assign(static_cast<size_t>(rw) * static_cast<size_t>(rh), 0);
            for (int gy = r_min_y; gy < r_max_y; ++gy) {
                const size_t srow = static_cast<size_t>(gy - work_rect.y1) * static_cast<size_t>(ww);
                const size_t drow = static_cast<size_t>(gy - r_min_y) * static_cast<size_t>(rw);
                for (int gx = r_min_x; gx < r_max_x; ++gx) {
                    if (final_work[srow + static_cast<size_t>(gx - work_rect.x1)] != 0) {
                        blk.refined_mask[drow + static_cast<size_t>(gx - r_min_x)] = 255;
                    }
                }
            }
        }
    }

    return page_mask;
}

}  // namespace vellum::detector
