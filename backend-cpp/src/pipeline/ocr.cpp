#include <httplib.h>

#include "ocr.hpp"
#include "detector.hpp"
#include "math_ocr_ops.hpp"
#include "math_nlp_qc.hpp"
#include "ort_provider.hpp"
#include "translator.hpp"
#include "../detector/cv_refinement.hpp"
#include "../util/image_ops.hpp"
#include "../utils.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100 4244 4456 4996)
#endif
#include <stb_image.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace mangaflow::pipeline {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

using util::StbiPtr;
using util::resize_rgb_bilinear;

int ocr_cpu_worker_count() noexcept {
    const int hw = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    return std::clamp(hw / 2, 2, 4);
}

int ocr_cpu_intra_threads_per_worker() noexcept {
    const int hw = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    const int workers = ocr_cpu_worker_count();
    return std::max(1, hw / workers);
}

struct ScopeOcrUnload {
    OcrEngine* engine = nullptr;
    bool enabled = false;
    ~ScopeOcrUnload() {
        if (enabled && engine) {
            engine->unload();
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

int parse_json_int(const json& val, const char* field_name) {
    if (val.is_boolean()) {
        throw std::invalid_argument(std::string("Expected integer for '") + field_name + "', got boolean.");
    }
    if (val.is_number_integer()) {
        return val.get<int>();
    }
    if (val.is_number_float()) {
        return static_cast<int>(std::lround(val.get<double>()));
    }
    if (val.is_string()) {
        try {
            return static_cast<int>(std::lround(std::stod(val.get<std::string>())));
        } catch (...) {
        }
    }
    throw std::invalid_argument(std::string("Expected integer for '") + field_name + "'.");
}

float parse_json_float(const json& val, float default_val) {
    if (val.is_boolean()) return default_val;
    if (val.is_number()) return val.get<float>();
    if (val.is_string()) {
        try {
            return std::stof(val.get<std::string>());
        } catch (...) {
            return default_val;
        }
    }
    return default_val;
}

std::array<int, 4> parse_bbox_array(const json& raw_box) {
    if (!raw_box.is_array() || raw_box.size() < 4) {
        throw std::invalid_argument("Expected 4-element bbox array [x1, y1, x2, y2].");
    }
    int x1 = parse_json_int(raw_box[0], "x1");
    int y1 = parse_json_int(raw_box[1], "y1");
    int x2 = parse_json_int(raw_box[2], "x2");
    int y2 = parse_json_int(raw_box[3], "y2");
    if (x2 < x1) std::swap(x1, x2);
    if (y2 < y1) std::swap(y1, y2);
    return {x1, y1, x2, y2};
}

std::vector<std::array<int, 2>> bbox_to_polygon(const std::array<int, 4>& bb) {
    return {{bb[0], bb[1]}, {bb[2], bb[1]}, {bb[2], bb[3]}, {bb[0], bb[3]}};
}

std::vector<std::string> load_vocab_lines(const fs::path& vocab_path) {
    std::ifstream ifs(vocab_path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open manga-ocr vocab.txt: " + util::path_to_utf8(vocab_path));
    }
    std::vector<std::string> lines;
    lines.reserve(6200);
    std::string line;
    bool first = true;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (first && line.size() >= 3 &&
            static_cast<unsigned char>(line[0]) == 0xEF &&
            static_cast<unsigned char>(line[1]) == 0xBB &&
            static_cast<unsigned char>(line[2]) == 0xBF) {
            line.erase(0, 3);
        }
        first = false;
        lines.push_back(std::move(line));
    }
    return lines;
}

std::vector<std::string> load_rapidocr_charset_from_session(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::ModelMetadata meta = session.GetModelMetadata();
    auto raw_val = meta.LookupCustomMetadataMapAllocated("character", alloc);
    if (!raw_val || raw_val.get() == nullptr) {
        throw std::runtime_error("PP-OCRv4 recognition ONNX model is missing embedded 'character' metadata.");
    }

    const std::string blob(raw_val.get());
    std::vector<std::string> charset;
    charset.reserve(7000);
    charset.emplace_back("");  // Index 0 = CTC blank

    size_t start = 0;
    while (start <= blob.size()) {
        const size_t nl = blob.find('\n', start);
        const size_t end = (nl == std::string::npos) ? blob.size() : nl;
        std::string entry = blob.substr(start, end - start);
        if (!entry.empty() && entry.back() == '\r') {
            entry.pop_back();
        }
        if (nl != std::string::npos || !entry.empty()) {
            charset.push_back(std::move(entry));
        }
        if (nl == std::string::npos) {
            break;
        }
        start = nl + 1;
    }

    charset.emplace_back(" ");  // Trailing space token (use_space_char=True in RapidOCR CTCLabelDecode)
    return charset;
}

void append_utf8_codepoint(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::vector<uint32_t> decode_utf8_codepoints(std::string_view s) {
    std::vector<uint32_t> cps;
    cps.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char b0 = static_cast<unsigned char>(s[i]);
        if (b0 < 0x80) {
            cps.push_back(b0);
            i += 1;
        } else if ((b0 & 0xE0) == 0xC0 && i + 1 < s.size()) {
            const uint32_t cp = ((b0 & 0x1Fu) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3Fu);
            cps.push_back(cp);
            i += 2;
        } else if ((b0 & 0xF0) == 0xE0 && i + 2 < s.size()) {
            const uint32_t cp = ((b0 & 0x0Fu) << 12) |
                                ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 6) |
                                (static_cast<unsigned char>(s[i + 2]) & 0x3Fu);
            cps.push_back(cp);
            i += 3;
        } else if ((b0 & 0xF8) == 0xF0 && i + 3 < s.size()) {
            const uint32_t cp = ((b0 & 0x07u) << 18) |
                                ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 12) |
                                ((static_cast<unsigned char>(s[i + 2]) & 0x3Fu) << 6) |
                                (static_cast<unsigned char>(s[i + 3]) & 0x3Fu);
            cps.push_back(cp);
            i += 4;
        } else {
            i += 1;
        }
    }
    return cps;
}

struct ParsedLocalEndpoint {
    bool valid = false;
    std::string host;
    int port = 1234;
    std::string base_path = "/v1";
};

ParsedLocalEndpoint parse_local_http_endpoint(const std::string& raw_url) {
    ParsedLocalEndpoint out;
    std::string s = util::trim_ascii(raw_url);
    if (s.empty()) {
        s = "http://localhost:1234/v1";
    }
    const std::string lower = util::to_lower_ascii(s);
    if (lower.rfind("http://", 0) == 0) {
        s = s.substr(7);
        out.port = 80;
    } else if (lower.rfind("https://", 0) == 0) {
        return out;  // Local LM Studio Vision-OCR uses plain HTTP loopback
    }
    std::string authority = s;
    if (const size_t slash = s.find('/'); slash != std::string::npos) {
        authority = s.substr(0, slash);
        out.base_path = s.substr(slash);
    } else {
        out.base_path = "/v1";
    }
    while (out.base_path.size() > 1 && out.base_path.back() == '/') {
        out.base_path.pop_back();
    }
    if (const size_t colon = authority.rfind(':'); colon != std::string::npos) {
        out.host = authority.substr(0, colon);
        try {
            out.port = std::stoi(authority.substr(colon + 1));
        } catch (...) {
            return out;
        }
    } else {
        out.host = authority;
    }
    out.valid = !out.host.empty() && out.port > 0 && out.port <= 65535;
    return out;
}

// -----------------------------------------------------------------------------
// Neighbor-Mask Blanking Helpers (prevents adjacent dialogue from leaking into OCR)
// -----------------------------------------------------------------------------

struct BlockGlyphMask {
    std::array<int, 4> rect{0, 0, 0, 0};  // [x1, y1, x2, y2] in page coords
    std::vector<uint8_t> mask;            // owned (x2-x1)*(y2-y1), 0 or 255
    const std::vector<uint8_t>* external_mask = nullptr;  // non-owning view into b.glyph_mask

    bool empty() const noexcept {
        return external_mask ? external_mask->empty() : mask.empty();
    }
    const uint8_t* data() const noexcept {
        return external_mask ? external_mask->data() : mask.data();
    }
};

std::vector<BlockGlyphMask> build_block_glyph_masks(
    const uint8_t* rgb_data,
    int img_w,
    int img_h,
    const std::vector<OcrBubbleInput>& bubbles,
    const OcrOptions& options) {
    std::vector<BlockGlyphMask> masks(bubbles.size());
    const bool has_page_mask =
        (options.page_glyph_mask != nullptr &&
         options.page_glyph_mask_w == img_w &&
         options.page_glyph_mask_h == img_h &&
         options.page_glyph_mask->size() >= static_cast<size_t>(img_w) * static_cast<size_t>(img_h));

    static constexpr int kDx8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static constexpr int kDy8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    for (size_t i = 0; i < bubbles.size(); ++i) {
        const auto& b = bubbles[i];

        // Priority 1: Explicit per-block glyph mask (non-owning const view — zero copy)
        const int gw = b.glyph_mask_rect[2] - b.glyph_mask_rect[0];
        const int gh = b.glyph_mask_rect[3] - b.glyph_mask_rect[1];
        if (gw > 0 && gh > 0 && b.glyph_mask.size() >= static_cast<size_t>(gw) * static_cast<size_t>(gh)) {
            masks[i].rect = b.glyph_mask_rect;
            masks[i].external_mask = &b.glyph_mask;
            continue;
        }

        const int x1 = clampi(b.bbox[0], 0, img_w);
        const int y1 = clampi(b.bbox[1], 0, img_h);
        const int x2 = clampi(b.bbox[2], 0, img_w);
        const int y2 = clampi(b.bbox[3], 0, img_h);
        const int bw = x2 - x1;
        const int bh = y2 - y1;
        if (bw <= 0 || bh <= 0) continue;

        masks[i].rect = {x1, y1, x2, y2};
        const size_t total = static_cast<size_t>(bw) * static_cast<size_t>(bh);
        masks[i].mask.assign(total, 0);

        // Priority 2: Slice from full-page text-stroke mask if provided
        if (has_page_mask) {
            bool any_px = false;
            for (int y = 0; y < bh; ++y) {
                const uint8_t* prow = options.page_glyph_mask->data() +
                                      static_cast<size_t>(y1 + y) * static_cast<size_t>(img_w) +
                                      static_cast<size_t>(x1);
                uint8_t* drow = masks[i].mask.data() + static_cast<size_t>(y) * static_cast<size_t>(bw);
                for (int x = 0; x < bw; ++x) {
                    if (prow[x] > 0) {
                        drow[x] = 255;
                        any_px = true;
                    }
                }
            }
            if (any_px) continue;
        }

        // Priority 3: Synthesize tight Otsu glyph mask inside bbox (excluding border-touching bubble wall ink)
        std::vector<uint8_t> gray(total, 255);
        std::array<int, 256> hist{};
        for (int y = 0; y < bh; ++y) {
            const uint8_t* srow = rgb_data +
                                  (static_cast<size_t>(y1 + y) * static_cast<size_t>(img_w) +
                                   static_cast<size_t>(x1)) * 3ULL;
            uint8_t* grow = gray.data() + static_cast<size_t>(y) * static_cast<size_t>(bw);
            for (int x = 0; x < bw; ++x) {
                const uint8_t* px = srow + static_cast<size_t>(x) * 3ULL;
                const uint8_t luma = rgb_to_luma(px[0], px[1], px[2]);
                grow[x] = luma;
                ++hist[luma];
            }
        }

        int acc = 0;
        const int half = static_cast<int>(total) / 2;
        uint8_t median_luma = 255;
        for (int v = 0; v < 256; ++v) {
            acc += hist[static_cast<size_t>(v)];
            if (acc >= half) {
                median_luma = static_cast<uint8_t>(v);
                break;
            }
        }
        const uint8_t thresh = vellum::detector::computeOtsuThreshold1Tile(gray.data(), bw, bh, 0.1);
        const bool dark_text = (median_luma >= 128);

        std::vector<uint8_t> raw_ink(total, 0);
        for (size_t k = 0; k < total; ++k) {
            const bool is_fg = dark_text ? (gray[k] <= thresh && gray[k] < 220)
                                         : (gray[k] >= thresh && gray[k] > 35);
            if (is_fg) raw_ink[k] = 1;
        }

        // Exclude ink connected to the bbox perimeter (bubble walls), keeping interior text glyphs
        std::vector<uint8_t> border_ink(total, 0);
        std::queue<int> q;
        auto push_b = [&](int lx, int ly) {
            const size_t idx = static_cast<size_t>(ly) * static_cast<size_t>(bw) + static_cast<size_t>(lx);
            if (raw_ink[idx] && !border_ink[idx]) {
                border_ink[idx] = 1;
                q.push(static_cast<int>(idx));
            }
        };
        for (int lx = 0; lx < bw; ++lx) {
            push_b(lx, 0);
            push_b(lx, bh - 1);
        }
        for (int ly = 0; ly < bh; ++ly) {
            push_b(0, ly);
            push_b(bw - 1, ly);
        }
        while (!q.empty()) {
            const int cur = q.front();
            q.pop();
            const int cx = cur % bw;
            const int cy = cur / bw;
            for (int d = 0; d < 8; ++d) {
                const int nx = cx + kDx8[d];
                const int ny = cy + kDy8[d];
                if (nx >= 0 && nx < bw && ny >= 0 && ny < bh) {
                    push_b(nx, ny);
                }
            }
        }

        // Dilate interior glyph mask by 1px to cover anti-aliased fringes
        for (int ly = 0; ly < bh; ++ly) {
            for (int lx = 0; lx < bw; ++lx) {
                const size_t idx = static_cast<size_t>(ly) * static_cast<size_t>(bw) + static_cast<size_t>(lx);
                if (!raw_ink[idx] || border_ink[idx]) continue;
                for (int dy = -1; dy <= 1; ++dy) {
                    const int ny = ly + dy;
                    if (ny < 0 || ny >= bh) continue;
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nx = lx + dx;
                        if (nx < 0 || nx >= bw) continue;
                        masks[i].mask[static_cast<size_t>(ny) * static_cast<size_t>(bw) + static_cast<size_t>(nx)] = 255;
                    }
                }
            }
        }
    }

    return masks;
}

bool blank_neighbor_masks_in_crop(
    std::vector<uint8_t>& crop_rgb,
    int crop_w,
    int crop_h,
    const std::array<int, 4>& crop_rect,
    size_t self_idx,
    const std::vector<BlockGlyphMask>& all_masks) {
    if (crop_w <= 0 || crop_h <= 0 || all_masks.size() <= 1) return false;
    const int cx1 = crop_rect[0];
    const int cy1 = crop_rect[1];
    const int cx2 = crop_rect[2];
    const int cy2 = crop_rect[3];
    if (cx2 - cx1 != crop_w || cy2 - cy1 != crop_h) return false;

    const size_t total = static_cast<size_t>(crop_w) * static_cast<size_t>(crop_h);
    std::vector<uint8_t> own_mask(total, 0);
    std::vector<uint8_t> other_mask(total, 0);

    // Paint self glyph mask into `own_mask`
    if (self_idx < all_masks.size() && !all_masks[self_idx].empty()) {
        const auto& sm = all_masks[self_idx];
        const int mw = sm.rect[2] - sm.rect[0];
        const int ix1 = std::max(cx1, sm.rect[0]);
        const int iy1 = std::max(cy1, sm.rect[1]);
        const int ix2 = std::min(cx2, sm.rect[2]);
        const int iy2 = std::min(cy2, sm.rect[3]);
        for (int gy = iy1; gy < iy2; ++gy) {
            const uint8_t* srow = sm.data() + static_cast<size_t>(gy - sm.rect[1]) * static_cast<size_t>(mw);
            uint8_t* drow = own_mask.data() + static_cast<size_t>(gy - cy1) * static_cast<size_t>(crop_w);
            for (int gx = ix1; gx < ix2; ++gx) {
                if (srow[gx - sm.rect[0]] > 0) {
                    drow[gx - cx1] = 255;
                }
            }
        }
    }

    // Union all OTHER blocks' glyph masks overlapping this crop (excluding own_mask)
    bool has_other = false;
    for (size_t j = 0; j < all_masks.size(); ++j) {
        if (j == self_idx || all_masks[j].empty()) continue;
        const auto& om = all_masks[j];
        const int ix1 = std::max(cx1, om.rect[0]);
        const int iy1 = std::max(cy1, om.rect[1]);
        const int ix2 = std::min(cx2, om.rect[2]);
        const int iy2 = std::min(cy2, om.rect[3]);
        if (ix2 <= ix1 || iy2 <= iy1) continue;

        const int mw = om.rect[2] - om.rect[0];
        for (int gy = iy1; gy < iy2; ++gy) {
            const uint8_t* srow = om.data() + static_cast<size_t>(gy - om.rect[1]) * static_cast<size_t>(mw);
            const uint8_t* orow = own_mask.data() + static_cast<size_t>(gy - cy1) * static_cast<size_t>(crop_w);
            uint8_t* drow = other_mask.data() + static_cast<size_t>(gy - cy1) * static_cast<size_t>(crop_w);
            for (int gx = ix1; gx < ix2; ++gx) {
                const int lx = gx - cx1;
                if (srow[gx - om.rect[0]] > 0 && orow[lx] == 0) {
                    drow[lx] = 255;
                    has_other = true;
                }
            }
        }
    }

    if (!has_other) return false;

    // Compute median RGB of unmasked background pixels inside this crop
    std::array<int, 256> r_hist{};
    std::array<int, 256> g_hist{};
    std::array<int, 256> b_hist{};
    size_t bg_count = 0;
    for (size_t i = 0; i < total; ++i) {
        if (own_mask[i] == 0 && other_mask[i] == 0) {
            const uint8_t* px = crop_rgb.data() + i * 3ULL;
            ++r_hist[px[0]];
            ++g_hist[px[1]];
            ++b_hist[px[2]];
            ++bg_count;
        }
    }

    auto hist_median = [bg_count](const std::array<int, 256>& h) -> uint8_t {
        if (bg_count == 0) return 255;
        size_t acc = 0;
        const size_t target = bg_count / 2;
        for (int v = 0; v < 256; ++v) {
            acc += static_cast<size_t>(h[static_cast<size_t>(v)]);
            if (acc > target) return static_cast<uint8_t>(v);
        }
        return 255;
    };

    const uint8_t med_r = hist_median(r_hist);
    const uint8_t med_g = hist_median(g_hist);
    const uint8_t med_b = hist_median(b_hist);

    for (size_t i = 0; i < total; ++i) {
        if (other_mask[i] > 0) {
            uint8_t* px = crop_rgb.data() + i * 3ULL;
            px[0] = med_r;
            px[1] = med_g;
            px[2] = med_b;
        }
    }
    return true;
}

}  // namespace

// -----------------------------------------------------------------------------
// OcrBubbleOutput JSON Serialization
// -----------------------------------------------------------------------------

json OcrBubbleOutput::to_json() const {
    const double conf_rounded = std::round(static_cast<double>(confidence) * 10000.0) / 10000.0;
    const double ent_rounded = std::round(static_cast<double>(ocr_entropy) * 10000.0) / 10000.0;
    json poly = json::array();
    for (const auto& pt : polygon) {
        poly.push_back(json::array({pt[0], pt[1]}));
    }
    return {
        {"id", id},
        {"order", order},
        {"bbox", {bbox[0], bbox[1], bbox[2], bbox[3]}},
        {"polygon", std::move(poly)},
        {"original_text", original_text},
        {"source_text", source_text},
        {"confidence", conf_rounded},
        {"ocr_entropy", ent_rounded},
        {"low_ocr_confidence", low_ocr_confidence},
        {"text_direction", text_direction},
    };
}

// -----------------------------------------------------------------------------
// Bubbles JSON Parser (`POST /api/stage/ocr` & `POST /api/stage/inpaint`)
// -----------------------------------------------------------------------------

std::vector<OcrBubbleInput> OcrEngine::parse_bubbles_json(const std::string& raw_json) {
    const json parsed = json::parse(raw_json, nullptr, false);
    if (parsed.is_discarded()) {
        throw std::invalid_argument("Invalid JSON in bboxes/blocks/bubbles field.");
    }

    const json* arr = nullptr;
    if (parsed.is_array()) {
        arr = &parsed;
    } else if (parsed.is_object()) {
        for (const char* key : {"bboxes", "blocks", "bubbles"}) {
            if (parsed.contains(key) && parsed[key].is_array()) {
                arr = &parsed[key];
                break;
            }
        }
    }
    if (!arr) {
        throw std::invalid_argument("Expected a JSON array of bounding boxes or bubble objects.");
    }

    std::vector<OcrBubbleInput> out;
    out.reserve(arr->size());

    for (size_t i = 0; i < arr->size(); ++i) {
        const int idx = static_cast<int>(i + 1);
        const auto& entry = (*arr)[i];

        char default_id[32]{};
        std::snprintf(default_id, sizeof(default_id), "bubble_%03d", idx);

        OcrBubbleInput item;
        if (entry.is_object()) {
            if (!entry.contains("bbox")) {
                throw std::invalid_argument("Bubble entry #" + std::to_string(idx) + " is missing 'bbox'.");
            }
            item.bbox = parse_bbox_array(entry["bbox"]);

            if (entry.contains("id") && entry["id"].is_string() && !util::trim_ascii(entry["id"].get<std::string>()).empty()) {
                item.id = util::trim_ascii(entry["id"].get<std::string>());
            } else {
                item.id = default_id;
            }

            item.order = (entry.contains("order") && !entry["order"].is_null())
                ? parse_json_int(entry["order"], "order")
                : idx;
            item.confidence = entry.contains("confidence")
                ? parse_json_float(entry["confidence"], 1.0f)
                : 1.0f;

            std::string raw_dir;
            if (entry.contains("text_direction") && entry["text_direction"].is_string()) {
                raw_dir = util::to_lower_ascii(util::trim_ascii(entry["text_direction"].get<std::string>()));
            }
            if (raw_dir == "vertical" || raw_dir == "horizontal") {
                item.text_direction = raw_dir;
            } else {
                const int bw = std::max(1, item.bbox[2] - item.bbox[0]);
                const int bh = std::max(1, item.bbox[3] - item.bbox[1]);
                item.text_direction = (bh > static_cast<int>(std::lround(bw * 1.2))) ? "vertical" : "horizontal";
            }

            if (entry.contains("polygon") && entry["polygon"].is_array()) {
                for (const auto& pt : entry["polygon"]) {
                    if (pt.is_array() && pt.size() >= 2) {
                        item.polygon.push_back({parse_json_int(pt[0], "px"), parse_json_int(pt[1], "py")});
                    }
                }
            }
            if (item.polygon.size() < 4) {
                item.polygon = bbox_to_polygon(item.bbox);
            }
        } else if (entry.is_array()) {
            item.bbox = parse_bbox_array(entry);
            item.id = default_id;
            item.order = idx;
            item.confidence = 1.0f;
            const int bw = std::max(1, item.bbox[2] - item.bbox[0]);
            const int bh = std::max(1, item.bbox[3] - item.bbox[1]);
            item.text_direction = (bh > static_cast<int>(std::lround(bw * 1.2))) ? "vertical" : "horizontal";
            item.polygon = bbox_to_polygon(item.bbox);
        } else {
            throw std::invalid_argument("Bubble entry #" + std::to_string(idx) + " must be a [x1,y1,x2,y2] array or object.");
        }

        out.push_back(std::move(item));
    }

    return out;
}

// -----------------------------------------------------------------------------
// OcrEngine Construction, Unloading & Crop Utilities
// -----------------------------------------------------------------------------

OcrEngine::OcrEngine(const ModelManager& model_manager)
    : model_manager_(model_manager) {}

std::string OcrEngine::normalize_ocr_model(const std::string& raw) {
    const std::string cleaned = util::to_lower_ascii(util::trim_ascii(raw));
    if (cleaned.empty() || cleaned == "auto" || cleaned == "default") {
        return "auto";
    }
    if (cleaned == "manga-ocr" || cleaned == "manga_ocr" || cleaned == "mangaocr" ||
        cleaned == "koharu-manga-ocr" || cleaned == "koharu_manga_ocr") {
        return "manga-ocr";
    }
    if (cleaned == "baberu-ocr" || cleaned == "baberu_ocr" || cleaned == "baberu" ||
        cleaned == "baberuocr") {
        return "baberu-ocr";
    }
    if (cleaned == "hayai-ocr" || cleaned == "hayai_ocr" || cleaned == "hayai" ||
        cleaned == "hayaiocr" || cleaned == "hayai-ocr-v2" || cleaned == "hayai_ocr_v2") {
        return "hayai-ocr";
    }
    if (cleaned == "paddleocr-vl-1.6" || cleaned == "paddleocr_vl_1.6" ||
        cleaned == "paddleocr-vl-gguf" || cleaned == "paddleocr_vl_gguf" ||
        cleaned == "paddleocr-vl" || cleaned == "paddleocr_vl" ||
        cleaned == "paddle-vl" || cleaned == "paddle_vl" ||
        cleaned == "paddleocr-vl-1.6-gguf") {
        return "paddleocr-vl-1.6";
    }
    if (cleaned == "rapidocr-ppocrv4" || cleaned == "rapidocr_ppocrv4" ||
        cleaned == "rapidocr" || cleaned == "rapid-ocr" || cleaned == "rapid_ocr" ||
        cleaned == "ppocrv4" || cleaned == "pp-ocrv4") {
        return "rapidocr-ppocrv4";
    }
    throw std::invalid_argument(
        "Unsupported ocr_model '" + raw +
        "'. Supported options: 'auto', 'paddleocr-vl-1.6', 'manga-ocr', 'baberu-ocr', 'hayai-ocr', 'rapidocr-ppocrv4'.");
}

std::string OcrEngine::resolve_effective_ocr_model(
    const std::string& ocr_model,
    const std::string& source_lang) {
    const std::string norm = normalize_ocr_model(ocr_model);
    const std::string lang = TextDetector::normalize_source_language(source_lang);
    if (norm == "auto") {
        return (lang == "ja") ? "manga-ocr" : "rapidocr-ppocrv4";
    }
    // Language-safe OCR routing guard:
    // - Korean (`ko`): manga-ocr and baberu-ocr do not support Hangul -> force rapidocr-ppocrv4
    //   (keep hayai-ocr or paddleocr-vl-1.6 when explicitly chosen).
    if (lang == "ko" && (norm == "manga-ocr" || norm == "baberu-ocr")) {
        return "rapidocr-ppocrv4";
    }
    // - Chinese (`zh`): manga-ocr is Japanese-only -> force rapidocr-ppocrv4
    //   (keep baberu-ocr, hayai-ocr, or paddleocr-vl-1.6 when explicitly chosen).
    if (lang == "zh" && norm == "manga-ocr") {
        return "rapidocr-ppocrv4";
    }
    return norm;
}

void OcrEngine::unload() {
    manga_enc_session_.reset();
    manga_dec_session_.reset();
    manga_vocab_.clear();
    manga_force_cpu_ = false;
    baberu_vis_session_.reset();
    baberu_pre_session_.reset();
    baberu_stp_session_.reset();
    baberu_id2ch_.clear();
    baberu_symbol_ids_.clear();
    hayai_enc_session_.reset();
    hayai_dec_session_.reset();
    hayai_id2tok_.clear();
    hayai_force_cpu_ = false;
    rapid_det_session_.reset();
    rapid_cls_session_.reset();
    rapid_rec_session_.reset();
    rapid_charset_.clear();
    rapid_loaded_lang_.clear();
    rapid_force_cpu_ = false;
}

OcrEngine::RgbImage OcrEngine::pad_rgb_white(
    const RgbImage& src,
    int top,
    int bottom,
    int left,
    int right) {
    RgbImage dst;
    dst.width = src.width + left + right;
    dst.height = src.height + top + bottom;
    dst.data.assign(static_cast<size_t>(dst.width) * static_cast<size_t>(dst.height) * 3ULL, 255);

    const size_t row_bytes = static_cast<size_t>(src.width) * 3ULL;
    for (int y = 0; y < src.height; ++y) {
        const uint8_t* srow = src.data.data() + static_cast<size_t>(y) * row_bytes;
        uint8_t* drow = dst.data.data() +
                        (static_cast<size_t>(y + top) * static_cast<size_t>(dst.width) +
                         static_cast<size_t>(left)) * 3ULL;
        std::memcpy(drow, srow, row_bytes);
    }
    return dst;
}

void OcrEngine::erase_border_touching_ink(RgbImage& crop) {
    const int w = crop.width;
    const int h = crop.height;
    if (w < 12 || h < 12) return;

    const size_t total = static_cast<size_t>(w) * static_cast<size_t>(h);
    std::vector<uint8_t> ink(total, 0);
    std::array<uint32_t, 256> bg_r{}, bg_g{}, bg_b{};
    uint32_t bg_count = 0;
    for (size_t i = 0; i < total; ++i) {
        const uint8_t* px = crop.data.data() + i * 3ULL;
        const uint8_t luma = rgb_to_luma(px[0], px[1], px[2]);
        if (luma < 200) {
            ink[i] = 1;
        } else {
            ++bg_r[px[0]];
            ++bg_g[px[1]];
            ++bg_b[px[2]];
            ++bg_count;
        }
    }

    auto median_hist = [](const std::array<uint32_t, 256>& hist, uint32_t cnt) -> uint8_t {
        if (cnt == 0) return 255;
        const uint32_t half = cnt / 2;
        uint32_t acc = 0;
        for (int v = 0; v < 256; ++v) {
            acc += hist[static_cast<size_t>(v)];
            if (acc > half) return static_cast<uint8_t>(v);
        }
        return 255;
    };
    const uint8_t fill_r = median_hist(bg_r, bg_count);
    const uint8_t fill_g = median_hist(bg_g, bg_count);
    const uint8_t fill_b = median_hist(bg_b, bg_count);

    // Component-aware 8-connected flood-fill from the 2px outer perimeter:
    // ONLY erase components that clearly represent speech-bubble walls / panel borders
    // (spanning > 35% of an edge OR confined strictly to the outer 15% margin band),
    // so legitimate characters near bubble edges are never clipped.
    std::vector<uint8_t> visited(total, 0);
    std::vector<uint8_t> erase_mask(total, 0);
    static constexpr int kDx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static constexpr int kDy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    const int margin_x = std::max(2, static_cast<int>(std::round(w * 0.15)));
    const int margin_y = std::max(2, static_cast<int>(std::round(h * 0.15)));
    const int span_thresh_x = static_cast<int>(std::round(w * 0.35));
    const int span_thresh_y = static_cast<int>(std::round(h * 0.35));

    auto process_seed = [&](int sx, int sy) {
        const size_t sidx = static_cast<size_t>(sy) * static_cast<size_t>(w) + static_cast<size_t>(sx);
        if (!ink[sidx] || visited[sidx]) return;

        std::queue<int> q;
        std::vector<size_t> comp;
        visited[sidx] = 1;
        q.push(static_cast<int>(sidx));

        int min_x = sx, max_x = sx, min_y = sy, max_y = sy;
        bool confined_to_outer_band = true;

        while (!q.empty()) {
            const int cur = q.front();
            q.pop();
            comp.push_back(static_cast<size_t>(cur));
            const int cx = cur % w;
            const int cy = cur / w;
            min_x = std::min(min_x, cx);
            max_x = std::max(max_x, cx);
            min_y = std::min(min_y, cy);
            max_y = std::max(max_y, cy);
            if (cx >= margin_x && cx < w - margin_x && cy >= margin_y && cy < h - margin_y) {
                confined_to_outer_band = false;
            }
            for (int k = 0; k < 8; ++k) {
                const int nx = cx + kDx[k];
                const int ny = cy + kDy[k];
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                    const size_t nidx = static_cast<size_t>(ny) * static_cast<size_t>(w) + static_cast<size_t>(nx);
                    if (ink[nidx] && !visited[nidx]) {
                        visited[nidx] = 1;
                        q.push(static_cast<int>(nidx));
                    }
                }
            }
        }

        const int comp_w = max_x - min_x + 1;
        const int comp_h = max_y - min_y + 1;
        // Guard: never erase a massive component covering > 60% of total crop pixels (e.g. inverted dark bubble)
        if (comp.size() * 5ULL > total * 3ULL) {
            return;
        }
        const bool is_wall_arc =
            (comp_w >= span_thresh_x) ||
            (comp_h >= span_thresh_y) ||
            confined_to_outer_band;
        if (is_wall_arc) {
            for (size_t idx : comp) {
                erase_mask[idx] = 1;
            }
        }
    };

    for (int x = 0; x < w; ++x) {
        process_seed(x, 0);
        process_seed(x, 1);
        process_seed(x, h - 1);
        process_seed(x, h - 2);
    }
    for (int y = 0; y < h; ++y) {
        process_seed(0, y);
        process_seed(1, y);
        process_seed(w - 1, y);
        process_seed(w - 2, y);
    }

    // 3x3 dilation to clear anti-aliased gray halo around erased bubble outline strokes
    std::vector<uint8_t> dilated(total, 0);
    bool any_erased = false;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (!erase_mask[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)]) {
                continue;
            }
            any_erased = true;
            for (int dy = -1; dy <= 1; ++dy) {
                const int ny = y + dy;
                if (ny < 0 || ny >= h) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx;
                    if (nx < 0 || nx >= w) continue;
                    dilated[static_cast<size_t>(ny) * static_cast<size_t>(w) + static_cast<size_t>(nx)] = 1;
                }
            }
        }
    }
    if (!any_erased) return;

    for (size_t i = 0; i < total; ++i) {
        if (dilated[i]) {
            uint8_t* px = crop.data.data() + i * 3ULL;
            px[0] = fill_r;
            px[1] = fill_g;
            px[2] = fill_b;
        }
    }
}

OcrEngine::RgbImage OcrEngine::crop_bbox_clean(
    const uint8_t* rgb_data,
    int img_w,
    int img_h,
    const std::array<int, 4>& bbox,
    bool erase_border_ink,
    std::array<int, 4>* out_crop_rect) {
    int x1 = clampi(bbox[0], 0, img_w);
    int y1 = clampi(bbox[1], 0, img_h);
    int x2 = clampi(bbox[2], 0, img_w);
    int y2 = clampi(bbox[3], 0, img_h);

    RgbImage crop;
    if (x2 <= x1 || y2 <= y1) {
        if (out_crop_rect) *out_crop_rect = {x1, y1, x1, y1};
        crop.width = 32;
        crop.height = 32;
        crop.data.assign(32ULL * 32ULL * 3ULL, 255);
        return crop;
    }

    // Defensive clipped-glyph expansion: if an input bbox edge slices directly through dark text
    // strokes inside a white speech bubble, expand up to 20px outward ONLY when a clean bright
    // ink-free gap is reached (never expanding across speech-bubble outlines or dark screentones).
    auto row_ink_count = [&](int y, int xa, int xb) -> int {
        int cnt = 0;
        const uint8_t* row = rgb_data + static_cast<size_t>(y) * static_cast<size_t>(img_w) * 3ULL;
        for (int x = xa; x < xb; ++x) {
            const uint8_t* px = row + static_cast<size_t>(x) * 3ULL;
            if (rgb_to_luma(px[0], px[1], px[2]) < 185) ++cnt;
        }
        return cnt;
    };
    auto col_ink_count = [&](int x, int ya, int yb) -> int {
        int cnt = 0;
        for (int y = ya; y < yb; ++y) {
            const uint8_t* px =
                rgb_data + (static_cast<size_t>(y) * static_cast<size_t>(img_w) + static_cast<size_t>(x)) * 3ULL;
            if (rgb_to_luma(px[0], px[1], px[2]) < 185) ++cnt;
        }
        return cnt;
    };

    constexpr int kMaxEdgeExpand = 20;
    const int init_w = x2 - x1;
    const int init_h = y2 - y1;
    const int mx = std::max(0, init_w / 10);
    const int my = std::max(0, init_h / 10);
    const int inner_xa = x1 + mx;
    const int inner_xb = std::max(inner_xa + 1, x2 - mx);
    const int inner_ya = y1 + my;
    const int inner_yb = std::max(inner_ya + 1, y2 - my);
    const int max_row_ink = std::max(4, static_cast<int>((inner_xb - inner_xa) * 0.65));
    const int max_col_ink = std::max(4, static_cast<int>((inner_yb - inner_ya) * 0.65));

    // Top edge
    if (const int c0 = row_ink_count(y1, inner_xa, inner_xb); c0 > 0 && c0 < max_row_ink) {
        int clean_streak = 0;
        for (int step = 1; step <= kMaxEdgeExpand && (y1 - step) >= 0; ++step) {
            const int c = row_ink_count(y1 - step, inner_xa, inner_xb);
            if (c >= max_row_ink) break;
            if (c == 0) {
                ++clean_streak;
                if (clean_streak >= 2) {
                    y1 = std::max(0, y1 - step - 1);
                    break;
                }
            } else {
                clean_streak = 0;
            }
        }
    }
    // Bottom edge
    if (const int c0 = row_ink_count(y2 - 1, inner_xa, inner_xb); c0 > 0 && c0 < max_row_ink) {
        int clean_streak = 0;
        for (int step = 1; step <= kMaxEdgeExpand && (y2 - 1 + step) < img_h; ++step) {
            const int c = row_ink_count(y2 - 1 + step, inner_xa, inner_xb);
            if (c >= max_row_ink) break;
            if (c == 0) {
                ++clean_streak;
                if (clean_streak >= 2) {
                    y2 = std::min(img_h, y2 + step + 1);
                    break;
                }
            } else {
                clean_streak = 0;
            }
        }
    }
    // Left edge
    if (const int c0 = col_ink_count(x1, inner_ya, inner_yb); c0 > 0 && c0 < max_col_ink) {
        int clean_streak = 0;
        for (int step = 1; step <= kMaxEdgeExpand && (x1 - step) >= 0; ++step) {
            const int c = col_ink_count(x1 - step, inner_ya, inner_yb);
            if (c >= max_col_ink) break;
            if (c == 0) {
                ++clean_streak;
                if (clean_streak >= 2) {
                    x1 = std::max(0, x1 - step - 1);
                    break;
                }
            } else {
                clean_streak = 0;
            }
        }
    }
    // Right edge
    if (const int c0 = col_ink_count(x2 - 1, inner_ya, inner_yb); c0 > 0 && c0 < max_col_ink) {
        int clean_streak = 0;
        for (int step = 1; step <= kMaxEdgeExpand && (x2 - 1 + step) < img_w; ++step) {
            const int c = col_ink_count(x2 - 1 + step, inner_ya, inner_yb);
            if (c >= max_col_ink) break;
            if (c == 0) {
                ++clean_streak;
                if (clean_streak >= 2) {
                    x2 = std::min(img_w, x2 + step + 1);
                    break;
                }
            } else {
                clean_streak = 0;
            }
        }
    }

    crop.width = x2 - x1;
    crop.height = y2 - y1;
    crop.data.resize(static_cast<size_t>(crop.width) * static_cast<size_t>(crop.height) * 3ULL);

    const size_t row_bytes = static_cast<size_t>(crop.width) * 3ULL;
    for (int y = 0; y < crop.height; ++y) {
        const uint8_t* srow = rgb_data +
                              (static_cast<size_t>(y1 + y) * static_cast<size_t>(img_w) +
                               static_cast<size_t>(x1)) * 3ULL;
        uint8_t* drow = crop.data.data() + static_cast<size_t>(y) * row_bytes;
        std::memcpy(drow, srow, row_bytes);
    }

    if (out_crop_rect) {
        *out_crop_rect = {x1, y1, x2, y2};
    }

    if (erase_border_ink) {
        erase_border_touching_ink(crop);
    }
    return crop;
}

std::vector<OcrEngine::RgbImage> OcrEngine::extract_horizontal_line_strips(
    const RgbImage& crop,
    int min_line_height) {
    const int w = crop.width;
    const int h = crop.height;
    if (static_cast<double>(w) < static_cast<double>(h) * 1.1) {
        return {};
    }

    const int margin_x = std::max(1, static_cast<int>(w * 0.10));
    std::vector<int> row_counts(static_cast<size_t>(h), 0);
    for (int y = 0; y < h; ++y) {
        int cnt = 0;
        const uint8_t* row = crop.data.data() + static_cast<size_t>(y) * static_cast<size_t>(w) * 3ULL;
        for (int x = margin_x; x < w - margin_x; ++x) {
            const uint8_t* px = row + static_cast<size_t>(x) * 3ULL;
            if (rgb_to_luma(px[0], px[1], px[2]) < 180) {
                ++cnt;
            }
        }
        row_counts[static_cast<size_t>(y)] = cnt;
    }

    bool in_line = false;
    int start_y = 0;
    std::vector<std::pair<int, int>> spans;
    for (int y = 0; y < h; ++y) {
        if (row_counts[static_cast<size_t>(y)] >= 3 && !in_line) {
            in_line = true;
            start_y = y;
        } else if (row_counts[static_cast<size_t>(y)] < 3 && in_line) {
            in_line = false;
            if (y - start_y >= min_line_height) {
                spans.emplace_back(start_y, y);
            }
        }
    }
    if (in_line && (h - start_y) >= min_line_height) {
        spans.emplace_back(start_y, h);
    }

    if (spans.size() < 2) {
        return {};
    }

    std::vector<RgbImage> strips;
    strips.reserve(spans.size());
    for (const auto& [sy1, sy2] : spans) {
        const int y1 = std::max(0, sy1 - 5);
        const int y2 = std::min(h, sy2 + 5);

        int min_x = w;
        int max_x = -1;
        for (int y = y1; y < y2; ++y) {
            const uint8_t* row = crop.data.data() + static_cast<size_t>(y) * static_cast<size_t>(w) * 3ULL;
            for (int x = 0; x < w; ++x) {
                const uint8_t* px = row + static_cast<size_t>(x) * 3ULL;
                if (rgb_to_luma(px[0], px[1], px[2]) < 180) {
                    if (x < min_x) min_x = x;
                    if (x > max_x) max_x = x;
                }
            }
        }
        if (max_x < min_x) continue;

        const int x1 = std::max(0, min_x - 8);
        const int x2 = std::min(w, max_x + 8);
        if (x2 - x1 < 16) continue;

        RgbImage sub;
        sub.width = x2 - x1;
        sub.height = y2 - y1;
        sub.data.resize(static_cast<size_t>(sub.width) * static_cast<size_t>(sub.height) * 3ULL);
        const size_t rb = static_cast<size_t>(sub.width) * 3ULL;
        for (int y = 0; y < sub.height; ++y) {
            std::memcpy(
                sub.data.data() + static_cast<size_t>(y) * rb,
                crop.data.data() + (static_cast<size_t>(y1 + y) * static_cast<size_t>(w) + static_cast<size_t>(x1)) * 3ULL,
                rb);
        }
        strips.push_back(pad_rgb_white(sub, 10, 10, 16, 16));
    }

    return (strips.size() >= 2) ? strips : std::vector<RgbImage>{};
}

// -----------------------------------------------------------------------------
// Japanese `manga-ocr` ONNX Encoder/Decoder + C++ Vocabulary Tokenizer
// -----------------------------------------------------------------------------

void OcrEngine::ensure_manga_ocr_loaded(bool force_cpu) {
    if (manga_enc_session_.has_value() &&
        manga_dec_session_.has_value() &&
        !manga_vocab_.empty() &&
        manga_force_cpu_ == force_cpu) {
        return;
    }

    manga_enc_session_.reset();
    manga_dec_session_.reset();
    manga_vocab_.clear();

    const auto resolved = model_manager_.resolve_manga_ocr_models();
    if (!resolved.has_value()) {
        throw std::runtime_error(
            "Japanese manga-ocr ONNX weights (manga-ocr/encoder_model.onnx, decoder_model.onnx, vocab.txt) "
            "not found in " + util::path_to_utf8(model_manager_.models_dir()) +
            ". Download them via Settings -> Pipeline (GET /api/models/download) first.");
    }

    const int per_worker_threads = ocr_cpu_intra_threads_per_worker();
    manga_vocab_ = load_vocab_lines(resolved->vocab_txt);
    manga_enc_session_ = ModelManager::create_ort_session(resolved->encoder_onnx, force_cpu, per_worker_threads);

    // Note: decoder_model.onnx has dynamic `seq_len` (1..N per step). On DirectML, dynamic shapes
    // trigger shader recompilation per token step, whereas CPU executes the 2-layer BERT decoder
    // in ~2 ms/step with zero shader stalls. Keep CUDA if available, otherwise CPU for decoder.
    const bool dec_force_cpu = force_cpu || (manga_enc_session_->active_provider == "DmlExecutionProvider");
    manga_dec_session_ = ModelManager::create_ort_session(resolved->decoder_onnx, dec_force_cpu, per_worker_threads);
    manga_force_cpu_ = force_cpu;
}

bool OcrEngine::manga_supports_dynamic_batch() const {
    if (!manga_enc_session_.has_value() || !manga_dec_session_.has_value()) {
        return false;
    }
    try {
        const auto enc_shape =
            manga_enc_session_->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        const auto dec_ids_shape =
            manga_dec_session_->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        const auto dec_hid_shape =
            manga_dec_session_->session->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
        return !enc_shape.empty() && enc_shape[0] <= 0 &&
               !dec_ids_shape.empty() && dec_ids_shape[0] <= 0 &&
               !dec_hid_shape.empty() && dec_hid_shape[0] <= 0;
    } catch (...) {
        return false;
    }
}

std::string OcrEngine::postprocess_manga_ocr_text(const std::string& raw_utf8) {
    // 1. Strip all ASCII whitespace
    std::string no_ws;
    no_ws.reserve(raw_utf8.size());
    for (unsigned char c : raw_utf8) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') continue;
        no_ws.push_back(static_cast<char>(c));
    }

    // 2. Decode UTF-8 codepoints for ellipsis normalization & ASCII -> full-width conversion
    auto cps = decode_utf8_codepoints(no_ws);

    // Replace U+2026 ('…') with three '.' codepoints (matching manga_ocr.post_process)
    std::vector<uint32_t> expanded;
    expanded.reserve(cps.size() + 8);
    for (uint32_t cp : cps) {
        if (cp == 0x2026u) {
            expanded.push_back('.');
            expanded.push_back('.');
            expanded.push_back('.');
        } else {
            expanded.push_back(cp);
        }
    }

    // Normalize runs of 2+ consecutive '・' (U+30FB) or '.' into matching '.' runs
    size_t i = 0;
    while (i < expanded.size()) {
        if (expanded[i] == 0x30FBu || expanded[i] == '.') {
            size_t j = i;
            while (j < expanded.size() && (expanded[j] == 0x30FBu || expanded[j] == '.')) {
                ++j;
            }
            if (j - i >= 2) {
                for (size_t k = i; k < j; ++k) {
                    expanded[k] = '.';
                }
            }
            i = j;
        } else {
            ++i;
        }
    }

    // Convert half-width ASCII (0x21..0x7E) to full-width (0xFF01..0xFF5E) matching jaconv.h2z
    std::string out;
    out.reserve(expanded.size() * 3);
    for (uint32_t cp : expanded) {
        if (cp >= 0x21u && cp <= 0x7Eu) {
            append_utf8_codepoint(out, cp + 0xFEE0u);
        } else {
            append_utf8_codepoint(out, cp);
        }
    }
    return out;
}

std::string OcrEngine::decode_manga_ocr_tokens(const std::vector<int64_t>& token_ids) const {
    std::string raw;
    raw.reserve(token_ids.size() * 3);
    for (int64_t tid : token_ids) {
        if (tid <= 4 || static_cast<size_t>(tid) >= manga_vocab_.size()) {
            continue;  // Skip [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4
        }
        const std::string& tok = manga_vocab_[static_cast<size_t>(tid)];
        if (tok.empty()) continue;
        if (tok.size() >= 2 && tok.front() == '[' && tok.back() == ']') continue;
        if (tok.rfind("<unused", 0) == 0) continue;

        if (tok.size() >= 2 && tok[0] == '#' && tok[1] == '#') {
            raw.append(tok.data() + 2, tok.size() - 2);
        } else {
            raw.append(tok);
        }
    }
    return postprocess_manga_ocr_text(raw);
}

std::string OcrEngine::run_manga_ocr_single(const RgbImage& crop, float* out_entropy) {
    if (out_entropy) *out_entropy = 0.0f;
    constexpr int kImgSize = 224;
    constexpr int64_t kBosTokenId = 2;   // [CLS]
    constexpr int64_t kEosTokenId = 3;   // [SEP]
    constexpr int64_t kPadTokenId = 0;   // [PAD]
    constexpr size_t kMaxTokens = 300;

    // 1. Convert to grayscale-replicated RGB (`img.convert("L").convert("RGB")`), resize 224x224
    RgbImage gray_rgb;
    gray_rgb.width = crop.width;
    gray_rgb.height = crop.height;
    gray_rgb.data.resize(crop.data.size());
    const size_t npix = static_cast<size_t>(crop.width) * static_cast<size_t>(crop.height);
    for (size_t i = 0; i < npix; ++i) {
        const uint8_t* p = crop.data.data() + i * 3ULL;
        const uint8_t luma = rgb_to_luma(p[0], p[1], p[2]);
        uint8_t* q = gray_rgb.data.data() + i * 3ULL;
        q[0] = luma;
        q[1] = luma;
        q[2] = luma;
    }

    const std::vector<uint8_t> resized =
        resize_rgb_bilinear(gray_rgb.data.data(), gray_rgb.width, gray_rgb.height, kImgSize, kImgSize);

    constexpr size_t kPlane = static_cast<size_t>(kImgSize) * static_cast<size_t>(kImgSize);
    std::vector<float> pixel_values(3ULL * kPlane);
    for (size_t i = 0; i < kPlane; ++i) {
        const float v = (static_cast<float>(resized[i * 3ULL]) / 255.0f - 0.5f) / 0.5f;
        pixel_values[i] = v;
        pixel_values[kPlane + i] = v;
        pixel_values[2ULL * kPlane + i] = v;
    }

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 4> enc_in_shape = {1, 3, kImgSize, kImgSize};
    Ort::Value enc_input = Ort::Value::CreateTensor<float>(
        mem_info, pixel_values.data(), pixel_values.size(), enc_in_shape.data(), enc_in_shape.size());

    // 2. Run ViT Encoder (`pixel_values` -> `last_hidden_state` [1, 197, 768])
    static std::atomic<uint64_t> s_ocr_enc_runs{0};
    const bool shrink_enc = should_shrink_arena_now(s_ocr_enc_runs, 5);
    auto run_encoder = [&](OrtSessionHandle& handle) -> std::vector<Ort::Value> {
        const char* in_names[] = {"pixel_values"};
        const char* out_names[] = {"last_hidden_state"};
        const bool is_cuda = (handle.active_provider == "CUDAExecutionProvider");
        Ort::RunOptions ro = shrink_enc
            ? make_shrink_run_options(true, is_cuda)
            : Ort::RunOptions{nullptr};
        return handle.session->Run(ro, in_names, &enc_input, 1, out_names, 1);
    };

    std::vector<Ort::Value> enc_outputs;
    try {
        enc_outputs = run_encoder(*manga_enc_session_);
    } catch (const Ort::Exception&) {
        std::lock_guard<std::mutex> lock(manga_fallback_mutex_);
        if (manga_enc_session_->active_provider == "CPUExecutionProvider") {
            throw;
        }
        const auto resolved = model_manager_.resolve_manga_ocr_models();
        manga_enc_session_ = ModelManager::create_ort_session(
            resolved->encoder_onnx, /*force_cpu=*/true, ocr_cpu_intra_threads_per_worker());
        enc_outputs = run_encoder(*manga_enc_session_);
    }

    const auto enc_shape = enc_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const size_t enc_elem_count = enc_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    const float* enc_ptr = enc_outputs[0].GetTensorData<float>();
    std::vector<float> enc_hidden(enc_ptr, enc_ptr + enc_elem_count);

    Ort::Value enc_hidden_tensor = Ort::Value::CreateTensor<float>(
        mem_info, enc_hidden.data(), enc_hidden.size(), enc_shape.data(), enc_shape.size());

    // 3. #17: Lightweight Width-3 Beam Search (B=3) + Compact JA Confusable Markov Prior
    //    with early-exit EOS guard (batches active beams in 1 ORT call when dynamic batching is supported).
    constexpr size_t kBeamWidth = 3;
    const bool can_batch_beams = manga_supports_dynamic_batch();

    struct MangaBeam {
        std::vector<int64_t> tokens;
        float score = 0.0f;
        float entropy_sum = 0.0f;
        int entropy_steps = 0;
        bool finished = false;
    };

    auto tokView = [&](int64_t tid) noexcept -> std::string_view {
        if (tid >= 0 && static_cast<size_t>(tid) < manga_vocab_.size()) {
            return manga_vocab_[static_cast<size_t>(tid)];
        }
        return {};
    };

    std::vector<MangaBeam> beams;
    beams.push_back(MangaBeam{{kBosTokenId}, 0.0f, 0.0f, 0, false});
    std::vector<MangaBeam> completed;
    completed.reserve(kBeamWidth * 2);

    const char* dec_in_names[] = {"input_ids", "encoder_hidden_states"};
    const char* dec_out_names[] = {"logits"};

    const size_t enc_seq = (enc_shape.size() >= 2) ? static_cast<size_t>(enc_shape[1]) : 197;
    const size_t enc_dim = (enc_shape.size() >= 3) ? static_cast<size_t>(enc_shape[2]) : 768;
    const size_t enc_stride = enc_seq * enc_dim;

    for (size_t step = 0; step < kMaxTokens && !beams.empty(); ++step) {
        const size_t cur_b = beams.size();
        const size_t cur_seq = step + 1ULL;

        std::vector<MangaBeam> candidates;
        candidates.reserve(cur_b * kBeamWidth);

        auto expandBeamFromLogits = [&](const MangaBeam& beam, const float* last_logits, size_t vocab_sz) {
            float max_l = -std::numeric_limits<float>::infinity();
            for (size_t v = 0; v < vocab_sz; ++v) {
                if (last_logits[v] > max_l) max_l = last_logits[v];
            }
            double sum_exp = 0.0;
            for (size_t v = 0; v < vocab_sz; ++v) {
                sum_exp += std::exp(static_cast<double>(last_logits[v] - max_l));
            }
            const float log_z = max_l + static_cast<float>(std::log(std::max(1e-12, sum_exp)));
            const float step_entropy = math_nlp::computeLogitEntropyBits(last_logits, vocab_sz);

            // Find top-kBeamWidth raw tokens
            std::array<std::pair<float, int64_t>, kBeamWidth> topk{};
            for (auto& p : topk) p = {-std::numeric_limits<float>::infinity(), kEosTokenId};
            for (size_t v = 0; v < vocab_sz; ++v) {
                const float val = last_logits[v];
                if (val > topk.back().first) {
                    topk.back() = {val, static_cast<int64_t>(v)};
                    for (size_t k = kBeamWidth - 1; k > 0 && topk[k].first > topk[k - 1].first; --k) {
                        std::swap(topk[k], topk[k - 1]);
                    }
                }
            }

            const std::string_view prev_tok =
                (beam.tokens.size() > 1) ? tokView(beam.tokens.back()) : std::string_view{};
            const float top1_prob = std::exp(topk[0].first - log_z);
            const bool any_confusable =
                (math_ocr::jaConfusableTransitionBonus(prev_tok, tokView(topk[0].second)) != 0.0f) ||
                (math_ocr::jaConfusableTransitionBonus(prev_tok, tokView(topk[1].second)) != 0.0f);
            // Fast-path: when single active beam is >96% confident and non-confusable, keep 1 branch.
            const size_t expand_k = (cur_b == 1 && top1_prob >= 0.96f && !any_confusable) ? 1 : kBeamWidth;

            for (size_t k = 0; k < expand_k; ++k) {
                const int64_t tid = topk[k].second;
                const float logp = topk[k].first - log_z;
                const float prior = math_ocr::jaConfusableTransitionBonus(prev_tok, tokView(tid));
                MangaBeam next = beam;
                next.score += logp + prior;
                if (tid == kEosTokenId || tid == kPadTokenId) {
                    if (next.entropy_steps == 0) {
                        next.entropy_sum += step_entropy;
                        next.entropy_steps += 1;
                    }
                    next.finished = true;
                    completed.push_back(std::move(next));
                } else {
                    next.entropy_sum += step_entropy;
                    next.entropy_steps += 1;
                    next.tokens.push_back(tid);
                    candidates.push_back(std::move(next));
                }
            }
        };

        if (can_batch_beams && cur_b > 1) {
            std::vector<int64_t> ids_buf(cur_b * cur_seq);
            std::vector<float> enc_buf(cur_b * enc_stride);
            for (size_t b = 0; b < cur_b; ++b) {
                std::memcpy(ids_buf.data() + b * cur_seq, beams[b].tokens.data(), cur_seq * sizeof(int64_t));
                std::memcpy(enc_buf.data() + b * enc_stride, enc_hidden.data(), enc_stride * sizeof(float));
            }
            const std::array<int64_t, 2> ids_shape = {static_cast<int64_t>(cur_b), static_cast<int64_t>(cur_seq)};
            const std::array<int64_t, 3> hid_shape = {
                static_cast<int64_t>(cur_b), static_cast<int64_t>(enc_seq), static_cast<int64_t>(enc_dim)
            };
            std::array<Ort::Value, 2> dec_inputs = {
                Ort::Value::CreateTensor<int64_t>(mem_info, ids_buf.data(), ids_buf.size(), ids_shape.data(), ids_shape.size()),
                Ort::Value::CreateTensor<float>(mem_info, enc_buf.data(), enc_buf.size(), hid_shape.data(), hid_shape.size()),
            };
            auto dec_outputs = manga_dec_session_->session->Run(
                Ort::RunOptions{nullptr}, dec_in_names, dec_inputs.data(), dec_inputs.size(), dec_out_names, 1);
            const auto logits_shape = dec_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            if (logits_shape.size() != 3 || logits_shape[2] <= 0) break;
            const size_t out_seq = static_cast<size_t>(logits_shape[1]);
            const size_t vocab_sz = static_cast<size_t>(logits_shape[2]);
            const float* logits_data = dec_outputs[0].GetTensorData<float>();
            for (size_t b = 0; b < cur_b; ++b) {
                expandBeamFromLogits(beams[b], logits_data + (b * out_seq + (out_seq - 1ULL)) * vocab_sz, vocab_sz);
            }
        } else {
            for (size_t b = 0; b < cur_b; ++b) {
                const std::array<int64_t, 2> ids_shape = {1, static_cast<int64_t>(cur_seq)};
                std::array<Ort::Value, 2> dec_inputs = {
                    Ort::Value::CreateTensor<int64_t>(
                        mem_info, beams[b].tokens.data(), beams[b].tokens.size(), ids_shape.data(), ids_shape.size()),
                    Ort::Value::CreateTensor<float>(
                        mem_info, enc_hidden.data(), enc_hidden.size(), enc_shape.data(), enc_shape.size()),
                };
                auto dec_outputs = manga_dec_session_->session->Run(
                    Ort::RunOptions{nullptr}, dec_in_names, dec_inputs.data(), dec_inputs.size(), dec_out_names, 1);
                const auto logits_shape = dec_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
                if (logits_shape.size() != 3 || logits_shape[2] <= 0) break;
                const size_t seq_len = static_cast<size_t>(logits_shape[1]);
                const size_t vocab_sz = static_cast<size_t>(logits_shape[2]);
                const float* last_logits = dec_outputs[0].GetTensorData<float>() + (seq_len - 1ULL) * vocab_sz;
                expandBeamFromLogits(beams[b], last_logits, vocab_sz);
            }
        }

        std::sort(candidates.begin(), candidates.end(), [](const MangaBeam& a, const MangaBeam& b) {
            return a.score > b.score;
        });
        if (candidates.size() > kBeamWidth) {
            candidates.resize(kBeamWidth);
        }
        beams = std::move(candidates);

        // Early-exit EOS guard: stop once we have >= kBeamWidth completed beams or top completed beats best active
        if (!completed.empty()) {
            const float best_done = std::max_element(
                completed.begin(), completed.end(),
                [](const MangaBeam& a, const MangaBeam& b) { return a.score < b.score; })->score;
            if (beams.empty() || completed.size() >= kBeamWidth || best_done >= beams.front().score + 0.5f) {
                break;
            }
        }
    }

    if (completed.empty()) {
        completed = std::move(beams);
    }

    // Rerank final beams with length normalization + bidirectional JA confusable sequence prior.
    const MangaBeam* best_beam = nullptr;
    float best_final = -std::numeric_limits<float>::infinity();
    for (const auto& b : completed) {
        std::vector<std::string_view> tviews;
        tviews.reserve(b.tokens.size());
        for (size_t i = 1; i < b.tokens.size(); ++i) {
            tviews.push_back(tokView(b.tokens[i]));
        }
        const float len_norm = std::pow(static_cast<float>(std::max<size_t>(1, tviews.size())), 0.65f);
        const float final_score = (b.score + math_ocr::jaConfusableSequenceBonus(tviews)) / len_norm;
        if (!best_beam || final_score > best_final) {
            best_final = final_score;
            best_beam = &b;
        }
    }

    if (out_entropy && best_beam && best_beam->entropy_steps > 0) {
        *out_entropy = best_beam->entropy_sum / static_cast<float>(best_beam->entropy_steps);
    }
    return best_beam ? decode_manga_ocr_tokens(best_beam->tokens) : std::string{};
}

std::vector<std::string> OcrEngine::run_manga_ocr_batch(
    const std::vector<RgbImage>& crops,
    size_t max_batch,
    std::vector<float>* out_entropies) {
    constexpr int kImgSize = 224;
    constexpr int64_t kBosTokenId = 2;   // [CLS]
    constexpr int64_t kEosTokenId = 3;   // [SEP]
    constexpr int64_t kPadTokenId = 0;   // [PAD]
    constexpr size_t kMaxTokens = 300;
    constexpr size_t kPlane = static_cast<size_t>(kImgSize) * static_cast<size_t>(kImgSize);

    std::vector<std::string> results(crops.size());
    if (out_entropies) {
        out_entropies->assign(crops.size(), 0.0f);
    }
    if (crops.empty()) {
        return results;
    }

    const size_t chunk_cap = std::max<size_t>(1, max_batch);
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const char* enc_in_names[] = {"pixel_values"};
    const char* enc_out_names[] = {"last_hidden_state"};
    const char* dec_in_names[] = {"input_ids", "encoder_hidden_states"};
    const char* dec_out_names[] = {"logits"};

    for (size_t start = 0; start < crops.size(); start += chunk_cap) {
        const size_t batch_n = std::min(chunk_cap, crops.size() - start);
        std::vector<float> pixel_values(batch_n * 3ULL * kPlane);

        for (size_t b = 0; b < batch_n; ++b) {
            const auto& crop = crops[start + b];
            RgbImage gray_rgb;
            gray_rgb.width = crop.width;
            gray_rgb.height = crop.height;
            gray_rgb.data.resize(crop.data.size());
            const size_t npix = static_cast<size_t>(crop.width) * static_cast<size_t>(crop.height);
            for (size_t i = 0; i < npix; ++i) {
                const uint8_t* p = crop.data.data() + i * 3ULL;
                const uint8_t luma = rgb_to_luma(p[0], p[1], p[2]);
                uint8_t* q = gray_rgb.data.data() + i * 3ULL;
                q[0] = luma;
                q[1] = luma;
                q[2] = luma;
            }

            const std::vector<uint8_t> resized =
                resize_rgb_bilinear(gray_rgb.data.data(), gray_rgb.width, gray_rgb.height, kImgSize, kImgSize);
            float* dst = pixel_values.data() + b * 3ULL * kPlane;
            for (size_t i = 0; i < kPlane; ++i) {
                const float v = (static_cast<float>(resized[i * 3ULL]) / 255.0f - 0.5f) / 0.5f;
                dst[i] = v;
                dst[kPlane + i] = v;
                dst[2ULL * kPlane + i] = v;
            }
        }

        const std::array<int64_t, 4> enc_in_shape = {
            static_cast<int64_t>(batch_n), 3, kImgSize, kImgSize
        };
        Ort::Value enc_input = Ort::Value::CreateTensor<float>(
            mem_info, pixel_values.data(), pixel_values.size(), enc_in_shape.data(), enc_in_shape.size());

        static std::atomic<uint64_t> s_ocr_batch_runs{0};
        const bool shrink_batch = should_shrink_arena_now(s_ocr_batch_runs, 5);
        const bool is_cuda_enc = (manga_enc_session_->active_provider == "CUDAExecutionProvider");
        Ort::RunOptions enc_ro = shrink_batch
            ? make_shrink_run_options(true, is_cuda_enc)
            : Ort::RunOptions{nullptr};
        auto enc_outputs = manga_enc_session_->session->Run(
            enc_ro, enc_in_names, &enc_input, 1, enc_out_names, 1);
        if (query_system_ram_stats().ram_load_pct >= kRamPressureGuardPct) {
            trim_process_working_set_throttled(15);
        }

        const auto enc_shape = enc_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        if (enc_shape.size() != 3 || static_cast<size_t>(enc_shape[0]) != batch_n ||
            enc_shape[1] <= 0 || enc_shape[2] <= 0) {
            throw std::runtime_error("Unexpected batched manga-ocr encoder output shape.");
        }
        const size_t enc_seq = static_cast<size_t>(enc_shape[1]);
        const size_t enc_dim = static_cast<size_t>(enc_shape[2]);
        const size_t enc_stride = enc_seq * enc_dim;
        const float* enc_ptr = enc_outputs[0].GetTensorData<float>();
        std::vector<float> enc_hidden(enc_ptr, enc_ptr + batch_n * enc_stride);

        std::vector<std::vector<int64_t>> token_seqs(batch_n);
        std::vector<float> ent_sums(batch_n, 0.0f);
        std::vector<int> ent_counts(batch_n, 0);
        std::vector<size_t> active_slots(batch_n);
        for (size_t b = 0; b < batch_n; ++b) {
            token_seqs[b].reserve(64);
            token_seqs[b].push_back(kBosTokenId);
            active_slots[b] = b;
        }

        std::vector<int64_t> ids_buf;
        std::vector<float> active_enc_buf;
        std::vector<size_t> next_active;
        next_active.reserve(batch_n);

        for (size_t step = 0; step < kMaxTokens && !active_slots.empty(); ++step) {
            const size_t cur_b = active_slots.size();
            const size_t cur_seq = step + 1ULL;

            ids_buf.resize(cur_b * cur_seq);
            active_enc_buf.resize(cur_b * enc_stride);
            for (size_t j = 0; j < cur_b; ++j) {
                const size_t slot = active_slots[j];
                std::memcpy(ids_buf.data() + j * cur_seq, token_seqs[slot].data(), cur_seq * sizeof(int64_t));
                std::memcpy(
                    active_enc_buf.data() + j * enc_stride,
                    enc_hidden.data() + slot * enc_stride,
                    enc_stride * sizeof(float));
            }

            const std::array<int64_t, 2> ids_shape = {
                static_cast<int64_t>(cur_b), static_cast<int64_t>(cur_seq)
            };
            const std::array<int64_t, 3> hid_shape = {
                static_cast<int64_t>(cur_b), static_cast<int64_t>(enc_seq), static_cast<int64_t>(enc_dim)
            };

            std::array<Ort::Value, 2> dec_inputs = {
                Ort::Value::CreateTensor<int64_t>(
                    mem_info, ids_buf.data(), ids_buf.size(), ids_shape.data(), ids_shape.size()),
                Ort::Value::CreateTensor<float>(
                    mem_info, active_enc_buf.data(), active_enc_buf.size(), hid_shape.data(), hid_shape.size()),
            };

            auto dec_outputs = manga_dec_session_->session->Run(
                Ort::RunOptions{nullptr},
                dec_in_names,
                dec_inputs.data(),
                dec_inputs.size(),
                dec_out_names,
                1);

            const auto logits_shape = dec_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            if (logits_shape.size() != 3 || static_cast<size_t>(logits_shape[0]) != cur_b ||
                logits_shape[1] <= 0 || logits_shape[2] <= 0) {
                break;
            }
            const size_t out_seq = static_cast<size_t>(logits_shape[1]);
            const size_t vocab_sz = static_cast<size_t>(logits_shape[2]);
            const float* logits_data = dec_outputs[0].GetTensorData<float>();

            next_active.clear();
            for (size_t j = 0; j < cur_b; ++j) {
                const size_t slot = active_slots[j];
                const float* last_logits = logits_data + (j * out_seq + (out_seq - 1ULL)) * vocab_sz;
                int64_t best_id = kEosTokenId;
                float best_val = -std::numeric_limits<float>::infinity();
                for (size_t v = 0; v < vocab_sz; ++v) {
                    if (last_logits[v] > best_val) {
                        best_val = last_logits[v];
                        best_id = static_cast<int64_t>(v);
                    }
                }
                if (best_id != kEosTokenId && best_id != kPadTokenId) {
                    if (out_entropies) {
                        ent_sums[slot] += math_nlp::computeLogitEntropyBits(last_logits, vocab_sz);
                        ent_counts[slot] += 1;
                    }
                    token_seqs[slot].push_back(best_id);
                    next_active.push_back(slot);
                }
            }
            active_slots.swap(next_active);
        }

        for (size_t b = 0; b < batch_n; ++b) {
            results[start + b] = decode_manga_ocr_tokens(token_seqs[b]);
            if (out_entropies && ent_counts[b] > 0) {
                (*out_entropies)[start + b] = ent_sums[b] / static_cast<float>(ent_counts[b]);
            }
        }
    }

    return results;
}

std::string OcrEngine::recognize_ja_crop(
    const RgbImage& raw_crop,
    const std::string& text_direction,
    float* out_entropy) {
    if (out_entropy) *out_entropy = 0.0f;
    if (text_direction == "horizontal") {
        const auto strips = extract_horizontal_line_strips(raw_crop);
        if (!strips.empty()) {
            std::string combined;
            float ent_sum = 0.0f;
            int ent_cnt = 0;
            for (const auto& strip : strips) {
                float strip_ent = 0.0f;
                const std::string t = util::trim_ascii(run_manga_ocr_single(strip, out_entropy ? &strip_ent : nullptr));
                if (!t.empty()) {
                    if (!combined.empty()) combined.push_back(' ');
                    combined.append(t);
                    ent_sum += strip_ent;
                    ent_cnt += 1;
                }
            }
            if (!combined.empty()) {
                if (out_entropy && ent_cnt > 0) *out_entropy = ent_sum / static_cast<float>(ent_cnt);
                return combined;
            }
        }
    }

    // #3: 1D Ink Projection Trim + Symmetric 1:1 Square Padding (side = max(w, h) + 16)
    // so narrow vertical Japanese bubbles are never horizontally stretched before 224x224 ViT resize.
    const auto sq = math_ocr::trimAndSquarePadRgb(
        raw_crop.data.data(), raw_crop.width, raw_crop.height, nullptr, /*pad_margin=*/16);
    RgbImage padded;
    padded.width = sq.width;
    padded.height = sq.height;
    padded.data = sq.data;
    return util::trim_ascii(run_manga_ocr_single(padded, out_entropy));
}

// -----------------------------------------------------------------------------
// Baberu OCR (`genshiai-daichi/baberu-ocr` ONNX Vision + INT8 KV-Cache Decoder)
// -----------------------------------------------------------------------------

void OcrEngine::ensure_baberu_ocr_loaded() {
    if (baberu_vis_session_.has_value() &&
        baberu_pre_session_.has_value() &&
        baberu_stp_session_.has_value() &&
        !baberu_id2ch_.empty()) {
        return;
    }

    baberu_vis_session_.reset();
    baberu_pre_session_.reset();
    baberu_stp_session_.reset();
    baberu_id2ch_.clear();
    baberu_symbol_ids_.clear();

    const auto resolved = model_manager_.resolve_baberu_ocr_models();
    if (!resolved.has_value()) {
        throw std::runtime_error(
            "Baberu OCR ONNX weights (baberu-ocr/vision_int4.onnx, decoder_prefill_int8.onnx, decoder_step_int8.onnx, vocab.json) "
            "not found in " + util::path_to_utf8(model_manager_.models_dir()) + ".");
    }

    const auto vocab_bytes = util::read_file_bytes(resolved->vocab_json);
    if (!vocab_bytes.has_value() || vocab_bytes->empty()) {
        throw std::runtime_error("Failed to read Baberu OCR vocab.json.");
    }
    const json charset = json::parse(vocab_bytes->begin(), vocab_bytes->end(), nullptr, false);
    if (charset.is_discarded() || !charset.is_array()) {
        throw std::runtime_error("Invalid JSON array in Baberu OCR vocab.json.");
    }

    baberu_id2ch_.assign(charset.size() + 4, "");
    for (size_t i = 0; i < charset.size(); ++i) {
        if (charset[i].is_string()) {
            const std::string ch = charset[i].get<std::string>();
            baberu_id2ch_[i + 4] = ch;
            if (ch == "ー" || ch == "ｰ" || ch == "〜" || ch == "~" || ch == "." || ch == "…") {
                baberu_symbol_ids_.insert(static_cast<int64_t>(i + 4));
            }
        }
    }

    // Keep Baberu OCR strictly on fast AVX2 CPUExecutionProvider to conserve RTX VRAM;
    // size intra-op threads per worker so the multi-core worker pool never oversubscribes cores.
    const int per_worker_threads = ocr_cpu_intra_threads_per_worker();
    baberu_vis_session_ = ModelManager::create_ort_session(resolved->vision_onnx, /*force_cpu=*/true, per_worker_threads);
    baberu_pre_session_ = ModelManager::create_ort_session(resolved->decoder_prefill_onnx, /*force_cpu=*/true, per_worker_threads);
    baberu_stp_session_ = ModelManager::create_ort_session(resolved->decoder_step_onnx, /*force_cpu=*/true, per_worker_threads);
    baberu_vis_dynamic_batch_ = false;
    try {
        const auto vis_in_shape = baberu_vis_session_->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (vis_in_shape.size() == 4 && vis_in_shape[0] <= 0) {
            baberu_vis_dynamic_batch_ = true;
        }
    } catch (...) {
        baberu_vis_dynamic_batch_ = false;
    }
}

std::string OcrEngine::decode_baberu_from_vision_embeds(
    const float* vis_data,
    const std::vector<int64_t>& vis_shape,
    float* out_entropy) {
    if (out_entropy) *out_entropy = 0.0f;
    if (!vis_data || vis_shape.size() < 2) return "";

    constexpr int64_t kBosId = 1;
    constexpr int64_t kEosId = 2;
    constexpr int64_t kPadId = 0;
    constexpr int kMaxNewTokens = 256;
    constexpr float kRepetitionPenalty = 1.2f;

    size_t vis_elems = 1;
    for (int64_t d : vis_shape) vis_elems *= static_cast<size_t>(std::max<int64_t>(1, d));
    std::vector<float> vis_buf(vis_data, vis_data + vis_elems);
    const int64_t vis_seq_len = vis_shape[1];

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    int64_t bos_token = kBosId;
    const std::array<int64_t, 2> id_shape = {1, 1};
    std::array<Ort::Value, 2> pre_inputs = {
        Ort::Value::CreateTensor<float>(mem_info, vis_buf.data(), vis_buf.size(), vis_shape.data(), vis_shape.size()),
        Ort::Value::CreateTensor<int64_t>(mem_info, &bos_token, 1, id_shape.data(), id_shape.size()),
    };
    const char* pre_in_names[] = {"vision_embeds", "input_ids"};
    const char* pre_out_names[] = {
        "logits",
        "present_k0", "present_k1", "present_k2", "present_k3", "present_k4", "present_k5",
        "present_v0", "present_v1", "present_v2", "present_v3", "present_v4", "present_v5",
    };
    auto cur_outputs = baberu_pre_session_->session->Run(
        Ort::RunOptions{nullptr}, pre_in_names, pre_inputs.data(), pre_inputs.size(), pre_out_names, 13);

    // Conditional Beam Search: stays at 1x greedy speed when p1 >= 0.85 (and >= 0.92 for JA confusables),
    // escalating to width-3 beam search + JA confusable Markov prior only on uncertain steps.
    constexpr size_t kBeamWidth = 3;

    struct KvTensorBuf {
        std::vector<int64_t> shape;
        std::vector<float> data;
    };
    struct BaberuBeam {
        std::vector<int64_t> generated;
        std::unordered_set<int64_t> seen;
        std::vector<float> logits;
        std::array<KvTensorBuf, 12> kv;
        float score = 0.0f;
        float entropy_sum = 0.0f;
        int entropy_steps = 0;
        bool finished = false;
    };

    auto extractStepOutputs = [](const std::vector<Ort::Value>& outs,
                                 std::vector<float>& out_logits,
                                 std::array<KvTensorBuf, 12>& out_kv) -> bool {
        if (outs.size() < 13) return false;
        const auto ls = outs[0].GetTensorTypeAndShapeInfo().GetShape();
        if (ls.size() != 3 || ls[2] <= 0) return false;
        const size_t seq_len = static_cast<size_t>(ls[1]);
        const size_t vocab_sz = static_cast<size_t>(ls[2]);
        const float* lptr = outs[0].GetTensorData<float>() + (seq_len - 1ULL) * vocab_sz;
        out_logits.assign(lptr, lptr + vocab_sz);
        for (size_t k = 0; k < 12; ++k) {
            const auto info = outs[k + 1].GetTensorTypeAndShapeInfo();
            out_kv[k].shape = info.GetShape();
            const size_t cnt = info.GetElementCount();
            const float* ptr = outs[k + 1].GetTensorData<float>();
            out_kv[k].data.assign(ptr, ptr + cnt);
        }
        return true;
    };

    auto baberuTokView = [&](int64_t tid) noexcept -> std::string_view {
        if (tid >= 4 && static_cast<size_t>(tid) < baberu_id2ch_.size()) {
            return baberu_id2ch_[static_cast<size_t>(tid)];
        }
        return {};
    };

    BaberuBeam init_beam;
    init_beam.seen.insert(kBosId);
    if (!extractStepOutputs(cur_outputs, init_beam.logits, init_beam.kv)) {
        return "";
    }

    std::vector<BaberuBeam> beams;
    beams.push_back(std::move(init_beam));
    std::vector<BaberuBeam> completed;
    completed.reserve(kBeamWidth * 2);

    int64_t pos = vis_seq_len + 1;
    const char* stp_in_names[] = {
        "input_ids", "position_ids",
        "past_k0", "past_k1", "past_k2", "past_k3", "past_k4", "past_k5",
        "past_v0", "past_v1", "past_v2", "past_v3", "past_v4", "past_v5",
    };

    for (int step = 0; step < kMaxNewTokens && !beams.empty(); ++step) {
        struct PendingCand {
            size_t parent_idx = 0;
            int64_t tok = kEosId;
            float score = 0.0f;
            float step_entropy = 0.0f;
            bool is_eos = false;
        };
        std::vector<PendingCand> pending;
        pending.reserve(beams.size() * kBeamWidth);

        for (size_t bi = 0; bi < beams.size(); ++bi) {
            const auto& beam = beams[bi];
            const size_t vocab_sz = beam.logits.size();
            if (vocab_sz == 0) continue;

            std::vector<float> logits = beam.logits;
            for (int64_t tid : beam.seen) {
                if (tid >= 0 && static_cast<size_t>(tid) < vocab_sz) {
                    float s = logits[static_cast<size_t>(tid)];
                    logits[static_cast<size_t>(tid)] = (s < 0.0f) ? (s * kRepetitionPenalty) : (s / kRepetitionPenalty);
                }
            }
            if (!beam.generated.empty()) {
                const int64_t last = beam.generated.back();
                const int cap = baberu_symbol_ids_.count(last) ? 16 : 12;
                int run = 0;
                for (auto it = beam.generated.rbegin(); it != beam.generated.rend() && *it == last; ++it) {
                    ++run;
                }
                if (run >= cap && last >= 0 && static_cast<size_t>(last) < vocab_sz) {
                    logits[static_cast<size_t>(last)] = -std::numeric_limits<float>::infinity();
                }
            }

            float max_l = -std::numeric_limits<float>::infinity();
            for (size_t v = 0; v < vocab_sz; ++v) {
                if (logits[v] > max_l) max_l = logits[v];
            }
            double sum_exp = 0.0;
            for (size_t v = 0; v < vocab_sz; ++v) {
                if (std::isfinite(logits[v])) {
                    sum_exp += std::exp(static_cast<double>(logits[v] - max_l));
                }
            }
            const float log_z = max_l + static_cast<float>(std::log(std::max(1e-12, sum_exp)));

            std::array<std::pair<float, int64_t>, kBeamWidth> topk{};
            for (auto& p : topk) p = {-std::numeric_limits<float>::infinity(), kEosId};
            for (size_t v = 0; v < vocab_sz; ++v) {
                if (logits[v] > topk.back().first) {
                    topk.back() = {logits[v], static_cast<int64_t>(v)};
                    for (size_t k = kBeamWidth - 1; k > 0 && topk[k].first > topk[k - 1].first; --k) {
                        std::swap(topk[k], topk[k - 1]);
                    }
                }
            }

            const std::string_view prev_tok =
                beam.generated.empty() ? std::string_view{} : baberuTokView(beam.generated.back());
            const float top1_prob = std::exp(topk[0].first - log_z);
            const bool any_conf =
                (math_ocr::jaConfusableTransitionBonus(prev_tok, baberuTokView(topk[0].second)) != 0.0f) ||
                (math_ocr::jaConfusableTransitionBonus(prev_tok, baberuTokView(topk[1].second)) != 0.0f);
            // Fast-path 1x greedy when p1 >= 0.85 (and >= 0.92 if JA confusable); escalate to B=3 only when uncertain.
            const bool stay_greedy = (beams.size() == 1) && (top1_prob >= (any_conf ? 0.92f : 0.85f));
            const size_t expand_k = stay_greedy ? 1 : kBeamWidth;

            const float step_entropy = math_nlp::computeLogitEntropyBits(logits.data(), vocab_sz);
            for (size_t k = 0; k < expand_k; ++k) {
                const int64_t tid = topk[k].second;
                const float logp = topk[k].first - log_z;
                const float prior = math_ocr::jaConfusableTransitionBonus(prev_tok, baberuTokView(tid));
                const bool is_eos = (tid == kEosId || tid == kPadId);
                pending.push_back(PendingCand{bi, tid, beam.score + logp + prior, step_entropy, is_eos});
            }
        }

        std::sort(pending.begin(), pending.end(), [](const PendingCand& a, const PendingCand& b) {
            return a.score > b.score;
        });

        std::vector<BaberuBeam> next_beams;
        next_beams.reserve(kBeamWidth);
        const int64_t cur_pos = pos++;

        for (const auto& pc : pending) {
            if (pc.is_eos) {
                BaberuBeam done;
                done.generated = beams[pc.parent_idx].generated;
                done.score = pc.score;
                done.entropy_sum = beams[pc.parent_idx].entropy_sum;
                done.entropy_steps = beams[pc.parent_idx].entropy_steps;
                if (done.entropy_steps == 0) {
                    done.entropy_sum += pc.step_entropy;
                    done.entropy_steps += 1;
                }
                done.finished = true;
                completed.push_back(std::move(done));
                continue;
            }
            if (next_beams.size() >= kBeamWidth) continue;

            BaberuBeam nb;
            nb.generated = beams[pc.parent_idx].generated;
            nb.generated.push_back(pc.tok);
            nb.seen = beams[pc.parent_idx].seen;
            nb.seen.insert(pc.tok);
            nb.score = pc.score;
            nb.entropy_sum = beams[pc.parent_idx].entropy_sum + pc.step_entropy;
            nb.entropy_steps = beams[pc.parent_idx].entropy_steps + 1;

            if (static_cast<int>(nb.generated.size()) >= kMaxNewTokens) {
                nb.finished = true;
                completed.push_back(std::move(nb));
                continue;
            }

            int64_t next_tok = pc.tok;
            int64_t pos_val = cur_pos;
            std::vector<Ort::Value> stp_inputs;
            stp_inputs.reserve(14);
            stp_inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem_info, &next_tok, 1, id_shape.data(), id_shape.size()));
            stp_inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem_info, &pos_val, 1, id_shape.data(), id_shape.size()));
            auto& parent_kv = beams[pc.parent_idx].kv;
            for (size_t k = 0; k < 12; ++k) {
                stp_inputs.push_back(Ort::Value::CreateTensor<float>(
                    mem_info, parent_kv[k].data.data(), parent_kv[k].data.size(),
                    parent_kv[k].shape.data(), parent_kv[k].shape.size()));
            }
            auto step_outs = baberu_stp_session_->session->Run(
                Ort::RunOptions{nullptr}, stp_in_names, stp_inputs.data(), stp_inputs.size(), pre_out_names, 13);
            if (extractStepOutputs(step_outs, nb.logits, nb.kv)) {
                next_beams.push_back(std::move(nb));
            }
        }

        beams = std::move(next_beams);
        if (!completed.empty()) {
            const float best_done = std::max_element(
                completed.begin(), completed.end(),
                [](const BaberuBeam& a, const BaberuBeam& b) { return a.score < b.score; })->score;
            if (beams.empty() || completed.size() >= kBeamWidth || best_done >= beams.front().score + 0.5f) {
                break;
            }
        }
    }

    if (completed.empty()) {
        completed = std::move(beams);
    }

    const BaberuBeam* best_beam = nullptr;
    float best_final = -std::numeric_limits<float>::infinity();
    for (const auto& b : completed) {
        std::vector<std::string_view> tviews;
        tviews.reserve(b.generated.size());
        for (int64_t tid : b.generated) {
            tviews.push_back(baberuTokView(tid));
        }
        const float len_norm = std::pow(static_cast<float>(std::max<size_t>(1, tviews.size())), 0.65f);
        const float final_score = (b.score + math_ocr::jaConfusableSequenceBonus(tviews)) / len_norm;
        if (!best_beam || final_score > best_final) {
            best_final = final_score;
            best_beam = &b;
        }
    }

    std::string out;
    if (best_beam) {
        if (out_entropy && best_beam->entropy_steps > 0) {
            *out_entropy = best_beam->entropy_sum / static_cast<float>(best_beam->entropy_steps);
        }
        for (int64_t tid : best_beam->generated) {
            if (tid >= 4 && static_cast<size_t>(tid) < baberu_id2ch_.size()) {
                out.append(baberu_id2ch_[static_cast<size_t>(tid)]);
            }
        }
    }
    return util::trim_ascii(out);
}

std::string OcrEngine::recognize_baberu_crop(const RgbImage& raw_crop, float* out_entropy) {
    if (out_entropy) *out_entropy = 0.0f;
    ensure_baberu_ocr_loaded();

    constexpr int kImgSize = 224;
    // #3: 1D Ink Projection Trim + Symmetric 1:1 Square Padding (side = max(w, h) + 16)
    const auto sq = math_ocr::trimAndSquarePadRgb(
        raw_crop.data.data(), raw_crop.width, raw_crop.height, nullptr, /*pad_margin=*/16);
    const std::vector<uint8_t> resized =
        resize_rgb_bilinear(sq.data.data(), sq.width, sq.height, kImgSize, kImgSize);

    // DINOv2 / ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    constexpr size_t kPlane = static_cast<size_t>(kImgSize) * static_cast<size_t>(kImgSize);
    std::vector<float> pixel_values(3ULL * kPlane);
    for (size_t i = 0; i < kPlane; ++i) {
        const float r = static_cast<float>(resized[i * 3ULL + 0]) / 255.0f;
        const float g = static_cast<float>(resized[i * 3ULL + 1]) / 255.0f;
        const float b = static_cast<float>(resized[i * 3ULL + 2]) / 255.0f;
        pixel_values[i] = (r - 0.485f) / 0.229f;
        pixel_values[kPlane + i] = (g - 0.456f) / 0.224f;
        pixel_values[2ULL * kPlane + i] = (b - 0.406f) / 0.225f;
    }

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 4> vis_in_shape = {1, 3, kImgSize, kImgSize};
    Ort::Value vis_in = Ort::Value::CreateTensor<float>(
        mem_info, pixel_values.data(), pixel_values.size(), vis_in_shape.data(), vis_in_shape.size());

    const char* vis_in_names[] = {"pixel_values"};
    const char* vis_out_names[] = {"vision_embeds"};
    auto vis_outputs = baberu_vis_session_->session->Run(
        Ort::RunOptions{nullptr}, vis_in_names, &vis_in, 1, vis_out_names, 1);

    const auto vis_shape = vis_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    return decode_baberu_from_vision_embeds(vis_outputs[0].GetTensorData<float>(), vis_shape, out_entropy);
}

// -----------------------------------------------------------------------------
// Hayai OCR v2.1 (`JustANormalTinkerer/hayai-ocr-v2` SigLIP2 + 2D-mRoPE Decoder)
// -----------------------------------------------------------------------------

void OcrEngine::ensure_hayai_ocr_loaded(bool force_cpu) {
    if (hayai_enc_session_.has_value() &&
        hayai_dec_session_.has_value() &&
        !hayai_id2tok_.empty() &&
        hayai_force_cpu_ == force_cpu) {
        return;
    }

    hayai_enc_session_.reset();
    hayai_dec_session_.reset();
    hayai_id2tok_.clear();

    const auto resolved = model_manager_.resolve_hayai_ocr_models();
    if (!resolved.has_value()) {
        throw std::runtime_error(
            "Hayai OCR ONNX weights (hayai-ocr/encoder.onnx, decoder.onnx, vocab.json) "
            "not found in " + util::path_to_utf8(model_manager_.models_dir()) + ".");
    }

    const auto vocab_bytes = util::read_file_bytes(resolved->vocab_json);
    if (!vocab_bytes.has_value() || vocab_bytes->empty()) {
        throw std::runtime_error("Failed to read Hayai OCR vocab.json.");
    }
    const json vj = json::parse(vocab_bytes->begin(), vocab_bytes->end(), nullptr, false);
    if (vj.is_discarded()) {
        throw std::runtime_error("Invalid JSON in Hayai OCR vocab.json.");
    }

    if (vj.is_object() && vj.contains("tokens") && vj["tokens"].is_array()) {
        hayai_bos_id_ = vj.value("bos_id", 1);
        hayai_eos_id_ = vj.value("eos_id", 2);
        hayai_pad_id_ = vj.value("pad_id", 0);
        const auto& arr = vj["tokens"];
        hayai_id2tok_.resize(arr.size());
        for (size_t i = 0; i < arr.size(); ++i) {
            if (arr[i].is_string()) {
                hayai_id2tok_[i] = arr[i].get<std::string>();
            }
        }
    } else if (vj.is_array()) {
        hayai_id2tok_.resize(vj.size());
        for (size_t i = 0; i < vj.size(); ++i) {
            if (vj[i].is_string()) {
                hayai_id2tok_[i] = vj[i].get<std::string>();
            }
        }
    }

    const int per_worker_threads = ocr_cpu_intra_threads_per_worker();
    hayai_enc_session_ = ModelManager::create_ort_session(resolved->encoder_onnx, force_cpu, force_cpu ? per_worker_threads : 0);
    // Autoregressive decoder uses dynamic sequence length 1..N; run on CPU when DirectML is active
    // to avoid per-step DirectML shader compilation stalls.
    const bool dec_cpu = force_cpu || (hayai_enc_session_->active_provider == "DmlExecutionProvider");
    hayai_dec_session_ = ModelManager::create_ort_session(resolved->decoder_onnx, dec_cpu, dec_cpu ? per_worker_threads : 0);
    hayai_force_cpu_ = force_cpu;
}

std::string OcrEngine::recognize_hayai_crop(const RgbImage& raw_crop, float* out_entropy) {
    if (out_entropy) *out_entropy = 0.0f;
    constexpr int kImgSize = 256;
    constexpr size_t kMaxTokens = 96;

    const RgbImage padded = pad_rgb_white(raw_crop, 6, 6, 6, 6);
    const std::vector<uint8_t> resized =
        resize_rgb_bilinear(padded.data.data(), padded.width, padded.height, kImgSize, kImgSize);

    // SigLIP2 normalization: (px / 255.0 - 0.5) / 0.5
    constexpr size_t kPlane = static_cast<size_t>(kImgSize) * static_cast<size_t>(kImgSize);
    std::vector<float> pixel_values(3ULL * kPlane);
    for (size_t i = 0; i < kPlane; ++i) {
        pixel_values[i] = (static_cast<float>(resized[i * 3ULL + 0]) / 255.0f - 0.5f) / 0.5f;
        pixel_values[kPlane + i] = (static_cast<float>(resized[i * 3ULL + 1]) / 255.0f - 0.5f) / 0.5f;
        pixel_values[2ULL * kPlane + i] = (static_cast<float>(resized[i * 3ULL + 2]) / 255.0f - 0.5f) / 0.5f;
    }

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 4> enc_in_shape = {1, 3, kImgSize, kImgSize};
    Ort::Value enc_in = Ort::Value::CreateTensor<float>(
        mem_info, pixel_values.data(), pixel_values.size(), enc_in_shape.data(), enc_in_shape.size());

    const char* enc_in_names[] = {"pixel_values"};
    const char* enc_out_names[] = {"visual_features"};
    std::vector<Ort::Value> enc_outputs;
    try {
        enc_outputs = hayai_enc_session_->session->Run(
            Ort::RunOptions{nullptr}, enc_in_names, &enc_in, 1, enc_out_names, 1);
    } catch (const Ort::Exception&) {
        if (hayai_enc_session_->active_provider == "CPUExecutionProvider") {
            throw;
        }
        const auto resolved = model_manager_.resolve_hayai_ocr_models();
        hayai_enc_session_ = ModelManager::create_ort_session(resolved->encoder_onnx, /*force_cpu=*/true);
        enc_outputs = hayai_enc_session_->session->Run(
            Ort::RunOptions{nullptr}, enc_in_names, &enc_in, 1, enc_out_names, 1);
    }

    const auto vis_shape = enc_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const size_t vis_elems = enc_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    const float* vis_ptr = enc_outputs[0].GetTensorData<float>();
    std::vector<float> vis_feat(vis_ptr, vis_ptr + vis_elems);

    std::vector<int64_t> token_ids;
    token_ids.reserve(kMaxTokens + 1);
    token_ids.push_back(hayai_bos_id_);

    const char* dec_in_names[] = {"visual_features", "text_token_ids"};
    const char* dec_out_names[] = {"logits"};

    float ent_sum = 0.0f;
    int ent_cnt = 0;
    for (size_t step = 0; step < kMaxTokens; ++step) {
        const std::array<int64_t, 2> ids_shape = {1, static_cast<int64_t>(token_ids.size())};
        std::array<Ort::Value, 2> dec_inputs = {
            Ort::Value::CreateTensor<float>(mem_info, vis_feat.data(), vis_feat.size(), vis_shape.data(), vis_shape.size()),
            Ort::Value::CreateTensor<int64_t>(mem_info, token_ids.data(), token_ids.size(), ids_shape.data(), ids_shape.size()),
        };

        auto dec_outputs = hayai_dec_session_->session->Run(
            Ort::RunOptions{nullptr}, dec_in_names, dec_inputs.data(), dec_inputs.size(), dec_out_names, 1);

        const auto l_shape = dec_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        if (l_shape.size() != 3 || l_shape[2] <= 0) break;
        const size_t seq_len = static_cast<size_t>(l_shape[1]);
        const size_t vocab_sz = static_cast<size_t>(l_shape[2]);
        const float* last_logits = dec_outputs[0].GetTensorData<float>() + (seq_len - 1ULL) * vocab_sz;

        int64_t best_id = hayai_eos_id_;
        float best_val = -std::numeric_limits<float>::infinity();
        for (size_t v = 0; v < vocab_sz; ++v) {
            if (last_logits[v] > best_val) {
                best_val = last_logits[v];
                best_id = static_cast<int64_t>(v);
            }
        }
        if (best_id == hayai_eos_id_ || best_id == hayai_pad_id_) {
            break;
        }
        if (out_entropy) {
            ent_sum += math_nlp::computeLogitEntropyBits(last_logits, vocab_sz);
            ent_cnt += 1;
        }
        token_ids.push_back(best_id);
    }
    if (out_entropy && ent_cnt > 0) {
        *out_entropy = ent_sum / static_cast<float>(ent_cnt);
    }

    std::string out;
    for (size_t i = 1; i < token_ids.size(); ++i) {
        const int64_t tid = token_ids[i];
        if (tid == hayai_eos_id_ || tid == hayai_pad_id_ || tid == hayai_bos_id_) continue;
        if (tid >= 0 && static_cast<size_t>(tid) < hayai_id2tok_.size()) {
            out.append(hayai_id2tok_[static_cast<size_t>(tid)]);
        }
    }
    return util::trim_ascii(out);
}

// -----------------------------------------------------------------------------
// Korean (`ko`) & Chinese (`zh`) PP-OCRv4 ONNX (det + cls + rec)
// -----------------------------------------------------------------------------

void OcrEngine::ensure_rapid_ocr_loaded(const std::string& lang, bool force_cpu) {
    const std::string profile = (lang == "ko") ? "ko" : "cjk";
    // PP-OCRv4 det/rec use variable tensor widths per bubble/line; avoid DirectML per-shape shader
    // recompilation by preferring CUDA when available or fast CPU otherwise.
    const std::string primary = ModelManager::get_primary_provider(force_cpu);
    const bool use_cpu = force_cpu || (primary == "DmlExecutionProvider");

    if (rapid_loaded_lang_ == profile &&
        rapid_force_cpu_ == use_cpu &&
        rapid_det_session_.has_value() &&
        rapid_cls_session_.has_value() &&
        rapid_rec_session_.has_value() &&
        !rapid_charset_.empty()) {
        return;
    }

    rapid_det_session_.reset();
    rapid_cls_session_.reset();
    rapid_rec_session_.reset();
    rapid_charset_.clear();
    rapid_loaded_lang_.clear();

    const auto paths = model_manager_.resolve_rapidocr_paths(lang);
    if (!paths.has_value()) {
        throw std::runtime_error(
            "RapidOCR (" + lang + ") PP-OCRv4 ONNX weights are missing. Call GET /api/models/download first.");
    }

    const int per_worker_threads = ocr_cpu_intra_threads_per_worker();
    rapid_det_session_ = ModelManager::create_ort_session(paths->det_model_path, use_cpu, per_worker_threads);
    rapid_cls_session_ = ModelManager::create_ort_session(paths->cls_model_path, use_cpu, per_worker_threads);
    rapid_rec_session_ = ModelManager::create_ort_session(paths->rec_model_path, use_cpu, per_worker_threads);
    rapid_charset_ = load_rapidocr_charset_from_session(*rapid_rec_session_->session);
    rapid_loaded_lang_ = profile;
    rapid_force_cpu_ = use_cpu;
    rapid_rec_dynamic_batch_ = false;
    try {
        const auto rec_in_shape = rapid_rec_session_->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (rec_in_shape.size() == 4 && rec_in_shape[0] <= 0 && rec_in_shape[3] <= 0) {
            rapid_rec_dynamic_batch_ = true;
        }
    } catch (...) {
        rapid_rec_dynamic_batch_ = false;
    }
}

std::vector<std::pair<std::string, float>> OcrEngine::run_ppocr_rec_batch(
    const std::vector<RgbImage>& line_crops,
    size_t max_batch) {
    constexpr int kRecH = 48;
    constexpr int kMinRecW = 320;

    std::vector<std::pair<std::string, float>> results(line_crops.size(), {"", 0.0f});
    if (line_crops.empty()) return results;
    if (!rapid_rec_dynamic_batch_ || line_crops.size() == 1) {
        for (size_t i = 0; i < line_crops.size(); ++i) {
            results[i] = run_ppocr_rec(line_crops[i]);
        }
        return results;
    }

    // Sort strip indices by resized width so mini-batches have minimal right-padding waste
    std::vector<std::pair<int, size_t>> order;
    order.reserve(line_crops.size());
    for (size_t i = 0; i < line_crops.size(); ++i) {
        const float ratio = static_cast<float>(std::max(1, line_crops[i].width)) /
                            static_cast<float>(std::max(1, line_crops[i].height));
        const int rw = std::max(1, static_cast<int>(std::ceil(static_cast<float>(kRecH) * ratio)));
        order.push_back({rw, i});
    }
    std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name = rapid_rec_session_->session->GetInputNameAllocated(0, alloc);
    auto out_name = rapid_rec_session_->session->GetOutputNameAllocated(0, alloc);
    const char* in_names[] = {in_name.get()};
    const char* out_names[] = {out_name.get()};

    const size_t chunk_cap = std::max<size_t>(1, max_batch);
    for (size_t start = 0; start < order.size(); start += chunk_cap) {
        const size_t batch_n = std::min(chunk_cap, order.size() - start);
        int max_rw = kMinRecW;
        for (size_t b = 0; b < batch_n; ++b) {
            if (order[start + b].first > max_rw) max_rw = order[start + b].first;
        }
        const int batch_w = ((max_rw + 7) / 8) * 8;  // align to multiple of 8
        const size_t plane = static_cast<size_t>(kRecH) * static_cast<size_t>(batch_w);
        const size_t img_stride = 3ULL * plane;
        // White background padding: (255 / 255 - 0.5) / 0.5 = +1.0f
        std::vector<float> tensor(batch_n * img_stride, 1.0f);

        for (size_t b = 0; b < batch_n; ++b) {
            const int rw = order[start + b].first;
            const auto& crop = line_crops[order[start + b].second];
            const std::vector<uint8_t> resized =
                resize_rgb_bilinear(crop.data.data(), crop.width, crop.height, rw, kRecH);
            float* ch0 = tensor.data() + b * img_stride;
            float* ch1 = ch0 + plane;
            float* ch2 = ch0 + 2ULL * plane;
            for (int y = 0; y < kRecH; ++y) {
                const uint8_t* srow = resized.data() + static_cast<size_t>(y) * static_cast<size_t>(rw) * 3ULL;
                const size_t drow = static_cast<size_t>(y) * static_cast<size_t>(batch_w);
                for (int x = 0; x < rw; ++x) {
                    const uint8_t* px = srow + static_cast<size_t>(x) * 3ULL;
                    const size_t idx = drow + static_cast<size_t>(x);
                    ch0[idx] = (static_cast<float>(px[2]) / 255.0f - 0.5f) / 0.5f;
                    ch1[idx] = (static_cast<float>(px[1]) / 255.0f - 0.5f) / 0.5f;
                    ch2[idx] = (static_cast<float>(px[0]) / 255.0f - 0.5f) / 0.5f;
                }
            }
        }

        const std::array<int64_t, 4> in_shape = {static_cast<int64_t>(batch_n), 3, kRecH, batch_w};
        Ort::Value ort_in = Ort::Value::CreateTensor<float>(
            mem_info, tensor.data(), tensor.size(), in_shape.data(), in_shape.size());

        auto outputs = rapid_rec_session_->session->Run(
            Ort::RunOptions{nullptr}, in_names, &ort_in, 1, out_names, 1);

        const auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        if (out_shape.size() != 3 || static_cast<size_t>(out_shape[0]) != batch_n ||
            out_shape[1] <= 0 || out_shape[2] <= 0) {
            continue;
        }
        const size_t steps = static_cast<size_t>(out_shape[1]);
        const size_t vocab_sz = static_cast<size_t>(out_shape[2]);
        const float* all_probs = outputs[0].GetTensorData<float>();

        for (size_t b = 0; b < batch_n; ++b) {
            const float* probs = all_probs + b * steps * vocab_sz;
            std::string text;
            double score_sum = 0.0;
            size_t score_cnt = 0;
            size_t prev_idx = 0;
            bool has_prev = false;

            for (size_t t = 0; t < steps; ++t) {
                const float* row = probs + t * vocab_sz;
                size_t best_idx = 0;
                float best_p = row[0];
                for (size_t c = 1; c < vocab_sz; ++c) {
                    if (row[c] > best_p) {
                        best_p = row[c];
                        best_idx = c;
                    }
                }
                const bool is_dup = has_prev && (best_idx == prev_idx);
                has_prev = true;
                prev_idx = best_idx;
                if (best_idx == 0 || is_dup) continue;
                if (best_idx < rapid_charset_.size()) {
                    text.append(rapid_charset_[best_idx]);
                    score_sum += static_cast<double>(best_p);
                    ++score_cnt;
                }
            }
            const float mean_conf = (score_cnt > 0) ? static_cast<float>(score_sum / static_cast<double>(score_cnt)) : 0.0f;
            results[order[start + b].second] = {util::trim_ascii(text), mean_conf};
        }
    }
    return results;
}

std::pair<std::string, float> OcrEngine::run_ppocr_rec(const RgbImage& line_crop) {
    constexpr int kRecH = 48;
    constexpr int kMinRecW = 320;

    const float ratio = static_cast<float>(std::max(1, line_crop.width)) /
                        static_cast<float>(std::max(1, line_crop.height));
    const int resized_w = std::max(1, static_cast<int>(std::ceil(static_cast<float>(kRecH) * ratio)));
    const int padded_w = std::max(kMinRecW, resized_w);

    const std::vector<uint8_t> resized =
        resize_rgb_bilinear(line_crop.data.data(), line_crop.width, line_crop.height, resized_w, kRecH);

    const size_t plane = static_cast<size_t>(kRecH) * static_cast<size_t>(padded_w);
    std::vector<float> tensor(3ULL * plane, 0.0f);

    float* ch0 = tensor.data();
    float* ch1 = tensor.data() + plane;
    float* ch2 = tensor.data() + 2ULL * plane;
    for (int y = 0; y < kRecH; ++y) {
        const uint8_t* srow = resized.data() + static_cast<size_t>(y) * static_cast<size_t>(resized_w) * 3ULL;
        const size_t drow = static_cast<size_t>(y) * static_cast<size_t>(padded_w);
        for (int x = 0; x < resized_w; ++x) {
            const uint8_t* px = srow + static_cast<size_t>(x) * 3ULL;
            const size_t idx = drow + static_cast<size_t>(x);
            // PaddleOCR BGR normalization `(val / 255.0 - 0.5) / 0.5`
            ch0[idx] = (static_cast<float>(px[2]) / 255.0f - 0.5f) / 0.5f;
            ch1[idx] = (static_cast<float>(px[1]) / 255.0f - 0.5f) / 0.5f;
            ch2[idx] = (static_cast<float>(px[0]) / 255.0f - 0.5f) / 0.5f;
        }
    }

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 4> in_shape = {1, 3, kRecH, padded_w};
    Ort::Value ort_in = Ort::Value::CreateTensor<float>(
        mem_info, tensor.data(), tensor.size(), in_shape.data(), in_shape.size());

    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name = rapid_rec_session_->session->GetInputNameAllocated(0, alloc);
    auto out_name = rapid_rec_session_->session->GetOutputNameAllocated(0, alloc);
    const char* in_names[] = {in_name.get()};
    const char* out_names[] = {out_name.get()};

    static std::atomic<uint64_t> s_rapid_rec_runs{0};
    const bool shrink_rec = should_shrink_arena_now(s_rapid_rec_runs, 5);
    const bool is_cuda_rec = (rapid_rec_session_->active_provider == "CUDAExecutionProvider");
    Ort::RunOptions rec_ro = shrink_rec
        ? make_shrink_run_options(true, is_cuda_rec)
        : Ort::RunOptions{nullptr};
    auto outputs = rapid_rec_session_->session->Run(
        rec_ro, in_names, &ort_in, 1, out_names, 1);
    if (query_system_ram_stats().ram_load_pct >= kRamPressureGuardPct) {
        trim_process_working_set_throttled(15);
    }

    const auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (out_shape.size() != 3 || out_shape[1] <= 0 || out_shape[2] <= 0) {
        return {"", 0.0f};
    }

    const size_t steps = static_cast<size_t>(out_shape[1]);
    const size_t vocab_sz = static_cast<size_t>(out_shape[2]);
    const float* probs = outputs[0].GetTensorData<float>();

    // Greedy CTC decoding (collapse consecutive duplicates, drop blank index 0)
    std::string text;
    double score_sum = 0.0;
    size_t score_cnt = 0;
    size_t prev_idx = 0;
    bool has_prev = false;

    for (size_t t = 0; t < steps; ++t) {
        const float* row = probs + t * vocab_sz;
        size_t best_idx = 0;
        float best_p = row[0];
        for (size_t c = 1; c < vocab_sz; ++c) {
            if (row[c] > best_p) {
                best_p = row[c];
                best_idx = c;
            }
        }
        const bool is_dup = has_prev && (best_idx == prev_idx);
        has_prev = true;
        prev_idx = best_idx;

        if (best_idx == 0 || is_dup) {
            continue;
        }
        if (best_idx < rapid_charset_.size()) {
            text.append(rapid_charset_[best_idx]);
            score_sum += static_cast<double>(best_p);
            ++score_cnt;
        }
    }

    const float mean_conf = (score_cnt > 0) ? static_cast<float>(score_sum / static_cast<double>(score_cnt)) : 0.0f;
    return {util::trim_ascii(text), mean_conf};
}

OcrEngine::RgbImage OcrEngine::apply_ppocr_cls_if_needed(const RgbImage& line_crop) {
    if (!rapid_cls_session_.has_value() || line_crop.width < 8 || line_crop.height < 8) {
        return line_crop;
    }

    constexpr int kClsH = 48;
    constexpr int kClsW = 192;
    const float ratio = static_cast<float>(line_crop.width) / static_cast<float>(line_crop.height);
    const int resized_w = clampi(static_cast<int>(std::ceil(static_cast<float>(kClsH) * ratio)), 1, kClsW);

    const std::vector<uint8_t> resized =
        resize_rgb_bilinear(line_crop.data.data(), line_crop.width, line_crop.height, resized_w, kClsH);

    constexpr size_t kPlane = static_cast<size_t>(kClsH) * static_cast<size_t>(kClsW);
    std::vector<float> tensor(3ULL * kPlane, 0.0f);
    for (int y = 0; y < kClsH; ++y) {
        const uint8_t* srow = resized.data() + static_cast<size_t>(y) * static_cast<size_t>(resized_w) * 3ULL;
        const size_t drow = static_cast<size_t>(y) * static_cast<size_t>(kClsW);
        for (int x = 0; x < resized_w; ++x) {
            const uint8_t* px = srow + static_cast<size_t>(x) * 3ULL;
            const size_t idx = drow + static_cast<size_t>(x);
            tensor[idx] = (static_cast<float>(px[2]) / 255.0f - 0.5f) / 0.5f;
            tensor[kPlane + idx] = (static_cast<float>(px[1]) / 255.0f - 0.5f) / 0.5f;
            tensor[2ULL * kPlane + idx] = (static_cast<float>(px[0]) / 255.0f - 0.5f) / 0.5f;
        }
    }

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 4> in_shape = {1, 3, kClsH, kClsW};
    Ort::Value ort_in = Ort::Value::CreateTensor<float>(
        mem_info, tensor.data(), tensor.size(), in_shape.data(), in_shape.size());

    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name = rapid_cls_session_->session->GetInputNameAllocated(0, alloc);
    auto out_name = rapid_cls_session_->session->GetOutputNameAllocated(0, alloc);
    const char* in_names[] = {in_name.get()};
    const char* out_names[] = {out_name.get()};

    auto outputs = rapid_cls_session_->session->Run(
        Ort::RunOptions{nullptr}, in_names, &ort_in, 1, out_names, 1);

    const auto elem_count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    if (elem_count < 2) {
        return line_crop;
    }
    const float* scores = outputs[0].GetTensorData<float>();
    if (scores[1] > scores[0] && scores[1] >= 0.90f) {
        // Rotate 180 degrees
        RgbImage rotated;
        rotated.width = line_crop.width;
        rotated.height = line_crop.height;
        rotated.data.resize(line_crop.data.size());
        const size_t total = static_cast<size_t>(line_crop.width) * static_cast<size_t>(line_crop.height);
        for (size_t i = 0; i < total; ++i) {
            const uint8_t* sp = line_crop.data.data() + (total - 1ULL - i) * 3ULL;
            uint8_t* dp = rotated.data.data() + i * 3ULL;
            dp[0] = sp[0];
            dp[1] = sp[1];
            dp[2] = sp[2];
        }
        return rotated;
    }
    return line_crop;
}

std::vector<std::array<int, 4>> OcrEngine::run_ppocr_det(const RgbImage& crop) {
    if (!rapid_det_session_.has_value() || crop.width < 8 || crop.height < 8) {
        return {};
    }

    // Resize to multiple of 32 (gentle upscale for small bubbles, cap at 960px)
    const int min_dim = std::min(crop.width, crop.height);
    const int max_dim = std::max(crop.width, crop.height);
    float scale = 1.0f;
    if (min_dim < 160) {
        scale = std::min(3.0f, 160.0f / static_cast<float>(min_dim));
    }
    if (static_cast<float>(max_dim) * scale > 960.0f) {
        scale = 960.0f / static_cast<float>(max_dim);
    }

    const int det_w = std::max(32, static_cast<int>(std::lround((static_cast<float>(crop.width) * scale) / 32.0f)) * 32);
    const int det_h = std::max(32, static_cast<int>(std::lround((static_cast<float>(crop.height) * scale) / 32.0f)) * 32);

    const std::vector<uint8_t> resized =
        resize_rgb_bilinear(crop.data.data(), crop.width, crop.height, det_w, det_h);

    const size_t plane = static_cast<size_t>(det_h) * static_cast<size_t>(det_w);
    std::vector<float> tensor(3ULL * plane, 0.0f);
    for (size_t i = 0; i < plane; ++i) {
        const uint8_t* px = resized.data() + i * 3ULL;
        tensor[i] = (static_cast<float>(px[2]) / 255.0f - 0.5f) / 0.5f;
        tensor[plane + i] = (static_cast<float>(px[1]) / 255.0f - 0.5f) / 0.5f;
        tensor[2ULL * plane + i] = (static_cast<float>(px[0]) / 255.0f - 0.5f) / 0.5f;
    }

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 4> in_shape = {1, 3, det_h, det_w};
    Ort::Value ort_in = Ort::Value::CreateTensor<float>(
        mem_info, tensor.data(), tensor.size(), in_shape.data(), in_shape.size());

    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name = rapid_det_session_->session->GetInputNameAllocated(0, alloc);
    auto out_name = rapid_det_session_->session->GetOutputNameAllocated(0, alloc);
    const char* in_names[] = {in_name.get()};
    const char* out_names[] = {out_name.get()};

    auto outputs = rapid_det_session_->session->Run(
        Ort::RunOptions{nullptr}, in_names, &ort_in, 1, out_names, 1);

    const auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (out_shape.size() != 4 || out_shape[2] <= 0 || out_shape[3] <= 0) {
        return {};
    }

    const int mh = static_cast<int>(out_shape[2]);
    const int mw = static_cast<int>(out_shape[3]);
    const float* prob_map = outputs[0].GetTensorData<float>();

    constexpr float kBinThresh = 0.30f;
    constexpr float kBoxThresh = 0.50f;
    constexpr float kUnclipRatio = 1.60f;

    const size_t total = static_cast<size_t>(mh) * static_cast<size_t>(mw);
    std::vector<uint8_t> visited(total, 0);
    std::vector<std::array<int, 4>> boxes;

    static constexpr int kDx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static constexpr int kDy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    for (int y = 0; y < mh; ++y) {
        for (int x = 0; x < mw; ++x) {
            const size_t start_idx = static_cast<size_t>(y) * static_cast<size_t>(mw) + static_cast<size_t>(x);
            if (visited[start_idx] || prob_map[start_idx] < kBinThresh) {
                continue;
            }

            int min_x = x, max_x = x, min_y = y, max_y = y;
            double prob_sum = 0.0;
            int count = 0;

            std::queue<int> q;
            visited[start_idx] = 1;
            q.push(static_cast<int>(start_idx));

            while (!q.empty()) {
                const int cur = q.front();
                q.pop();
                const int cx = cur % mw;
                const int cy = cur / mw;
                min_x = std::min(min_x, cx);
                max_x = std::max(max_x, cx);
                min_y = std::min(min_y, cy);
                max_y = std::max(max_y, cy);
                prob_sum += static_cast<double>(prob_map[cur]);
                ++count;

                for (int k = 0; k < 8; ++k) {
                    const int nx = cx + kDx[k];
                    const int ny = cy + kDy[k];
                    if (nx < 0 || nx >= mw || ny < 0 || ny >= mh) continue;
                    const size_t nidx = static_cast<size_t>(ny) * static_cast<size_t>(mw) + static_cast<size_t>(nx);
                    if (!visited[nidx] && prob_map[nidx] >= kBinThresh) {
                        visited[nidx] = 1;
                        q.push(static_cast<int>(nidx));
                    }
                }
            }

            if (count < 12) continue;
            const double mean_score = prob_sum / static_cast<double>(count);
            if (mean_score < kBoxThresh) continue;

            const float bw = static_cast<float>(max_x - min_x + 1);
            const float bh = static_cast<float>(max_y - min_y + 1);
            if (std::min(bw, bh) < 3.0f) continue;

            // DBNet rectangular unclip expansion
            const float margin = (bw * bh * kUnclipRatio) / std::max(1.0f, 2.0f * (bw + bh));
            const float sx = static_cast<float>(crop.width) / static_cast<float>(mw);
            const float sy = static_cast<float>(crop.height) / static_cast<float>(mh);

            const int rx1 = clampi(static_cast<int>(std::floor((static_cast<float>(min_x) - margin) * sx)), 0, crop.width);
            const int ry1 = clampi(static_cast<int>(std::floor((static_cast<float>(min_y) - margin) * sy)), 0, crop.height);
            const int rx2 = clampi(static_cast<int>(std::ceil((static_cast<float>(max_x + 1) + margin) * sx)), 0, crop.width);
            const int ry2 = clampi(static_cast<int>(std::ceil((static_cast<float>(max_y + 1) + margin) * sy)), 0, crop.height);

            if ((rx2 - rx1) >= 8 && (ry2 - ry1) >= 8) {
                boxes.push_back({rx1, ry1, rx2, ry2});
            }
        }
    }

    std::sort(boxes.begin(), boxes.end(), [](const std::array<int, 4>& a, const std::array<int, 4>& b) {
        const int row_a = static_cast<int>(std::lround(0.5 * (a[1] + a[3]) / 18.0));
        const int row_b = static_cast<int>(std::lround(0.5 * (b[1] + b[3]) / 18.0));
        if (row_a != row_b) return row_a < row_b;
        return (a[0] + a[2]) < (b[0] + b[2]);
    });
    return boxes;
}

namespace {

int count_target_script_chars(const std::string& utf8, const std::string& lang) {
    const auto cps = decode_utf8_codepoints(utf8);
    int cnt = 0;
    for (uint32_t cp : cps) {
        const bool is_hangul =
            (cp >= 0xAC00u && cp <= 0xD7AFu) ||
            (cp >= 0x1100u && cp <= 0x11FFu) ||
            (cp >= 0x3130u && cp <= 0x318Fu);
        const bool is_hanzi =
            (cp >= 0x4E00u && cp <= 0x9FFFu) ||
            (cp >= 0x3400u && cp <= 0x4DBFu) ||
            (cp >= 0xF900u && cp <= 0xFAFFu);
        if (lang == "ko") {
            if (is_hangul || is_hanzi) ++cnt;
        } else if (lang == "zh" || lang == "cjk") {
            if (is_hanzi) ++cnt;
        } else {
            if (is_hangul || is_hanzi || (cp >= 0x3040u && cp <= 0x30FFu)) ++cnt;
        }
    }
    return cnt;
}

void compute_crop_ink_mask(
    const uint8_t* rgb,
    int w,
    int h,
    std::vector<uint8_t>& out_ink,
    float& out_ink_ratio) {
    out_ink.assign(static_cast<size_t>(w) * static_cast<size_t>(h), 0);
    out_ink_ratio = 0.0f;
    if (!rgb || w <= 0 || h <= 0) return;

    // Sample border pixels to detect light-on-dark vs dark-on-light polarity
    std::vector<uint8_t> border_luma;
    border_luma.reserve(static_cast<size_t>(w + h) * 2ULL);
    auto luma_at = [&](int x, int y) -> uint8_t {
        const uint8_t* p = rgb + (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * 3ULL;
        return static_cast<uint8_t>((77 * p[0] + 150 * p[1] + 29 * p[2]) >> 8);
    };
    for (int x = 0; x < w; ++x) {
        border_luma.push_back(luma_at(x, 0));
        if (h > 1) border_luma.push_back(luma_at(x, h - 1));
    }
    for (int y = 1; y < h - 1; ++y) {
        border_luma.push_back(luma_at(0, y));
        if (w > 1) border_luma.push_back(luma_at(w - 1, y));
    }
    std::nth_element(border_luma.begin(), border_luma.begin() + border_luma.size() / 2, border_luma.end());
    const int bg_luma = border_luma.empty() ? 255 : static_cast<int>(border_luma[border_luma.size() / 2]);
    const bool inverted = (bg_luma < 100);

    const int thresh = inverted
        ? std::min(220, bg_luma + 42)
        : std::max(45, std::min(205, bg_luma - 38));

    size_t ink_cnt = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int l = static_cast<int>(luma_at(x, y));
            const bool is_ink = inverted ? (l > thresh) : (l < thresh);
            if (is_ink) {
                out_ink[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] = 1;
                ++ink_cnt;
            }
        }
    }
    out_ink_ratio = static_cast<float>(ink_cnt) / static_cast<float>(static_cast<size_t>(w) * static_cast<size_t>(h));
}

}  // namespace

std::string OcrEngine::recognize_rapid_crop(const RgbImage& raw_crop) {
    if (raw_crop.width <= 0 || raw_crop.height <= 0 || raw_crop.data.empty()) {
        return {};
    }

    // Helper: tight ink crop with white padding (for zero-box DBNet bypass)
    auto make_tight_ink_crop = [](const RgbImage& src, const std::vector<uint8_t>& ink) -> RgbImage {
        int min_x = src.width, min_y = src.height, max_x = -1, max_y = -1;
        for (int y = 0; y < src.height; ++y) {
            const size_t row = static_cast<size_t>(y) * static_cast<size_t>(src.width);
            for (int x = 0; x < src.width; ++x) {
                if (ink[row + static_cast<size_t>(x)]) {
                    if (x < min_x) min_x = x;
                    if (x > max_x) max_x = x;
                    if (y < min_y) min_y = y;
                    if (y > max_y) max_y = y;
                }
            }
        }
        if (max_x < min_x || max_y < min_y) {
            return pad_rgb_white(src, 10, 10, 10, 10);
        }
        const int cw = max_x - min_x + 1;
        const int ch = max_y - min_y + 1;
        RgbImage sub;
        sub.width = cw;
        sub.height = ch;
        sub.data.resize(static_cast<size_t>(cw) * static_cast<size_t>(ch) * 3ULL);
        const size_t rb = static_cast<size_t>(cw) * 3ULL;
        for (int y = 0; y < ch; ++y) {
            std::memcpy(
                sub.data.data() + static_cast<size_t>(y) * rb,
                src.data.data() + (static_cast<size_t>(min_y + y) * static_cast<size_t>(src.width) + static_cast<size_t>(min_x)) * 3ULL,
                rb);
        }
        return pad_rgb_white(sub, 8, 8, 10, 10);
    };

    // Helper: Vertical Upright-Glyph Horizontal Strip Re-Packing (NO 90-deg rotation)
    // Segments vertical columns (ordered RTL) and radical-safe upright glyphs, then packs
    // upright glyphs left-to-right onto a 48px white horizontal strip for PP-OCRv4 rec.
    auto repack_vertical_to_horizontal_strips = [&](const RgbImage& vcrop) -> std::vector<RgbImage> {
        RgbImage cleaned = vcrop;
        erase_border_touching_ink(cleaned);

        std::vector<uint8_t> ink;
        float ratio = 0.0f;
        compute_crop_ink_mask(cleaned.data.data(), cleaned.width, cleaned.height, ink, ratio);
        if (ratio < 0.005f) return {};

        const int W = cleaned.width;
        const int H = cleaned.height;

        // 0. Remove large bubble-wall / panel-frame connected components (even if inset 2-6px from crop edge)
        {
            std::vector<uint8_t> visited(static_cast<size_t>(W) * static_cast<size_t>(H), 0);
            std::vector<int> qx, qy;
            qx.reserve(256);
            qy.reserve(256);
            for (int sy = 0; sy < H; ++sy) {
                for (int sx = 0; sx < W; ++sx) {
                    const size_t sidx = static_cast<size_t>(sy) * static_cast<size_t>(W) + static_cast<size_t>(sx);
                    if (!ink[sidx] || visited[sidx]) continue;
                    qx.clear();
                    qy.clear();
                    qx.push_back(sx);
                    qy.push_back(sy);
                    visited[sidx] = 1;
                    int min_x = sx, max_x = sx, min_y = sy, max_y = sy;
                    size_t head = 0;
                    while (head < qx.size()) {
                        const int cx = qx[head];
                        const int cy = qy[head];
                        ++head;
                        if (cx < min_x) min_x = cx;
                        if (cx > max_x) max_x = cx;
                        if (cy < min_y) min_y = cy;
                        if (cy > max_y) max_y = cy;
                        const int dx4[4] = {-1, 1, 0, 0};
                        const int dy4[4] = {0, 0, -1, 1};
                        for (int d = 0; d < 4; ++d) {
                            const int nx = cx + dx4[d];
                            const int ny = cy + dy4[d];
                            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
                            const size_t nidx = static_cast<size_t>(ny) * static_cast<size_t>(W) + static_cast<size_t>(nx);
                            if (ink[nidx] && !visited[nidx]) {
                                visited[nidx] = 1;
                                qx.push_back(nx);
                                qy.push_back(ny);
                            }
                        }
                    }
                    const int cw = max_x - min_x + 1;
                    const int ch = max_y - min_y + 1;
                    const bool is_frame_or_wall =
                        (static_cast<float>(ch) > static_cast<float>(H) * 0.72f && static_cast<float>(cw) > static_cast<float>(W) * 0.55f) ||
                        (static_cast<float>(ch) > static_cast<float>(H) * 0.85f);
                    if (is_frame_or_wall) {
                        for (size_t k = 0; k < qx.size(); ++k) {
                            const size_t pidx = static_cast<size_t>(qy[k]) * static_cast<size_t>(W) + static_cast<size_t>(qx[k]);
                            ink[pidx] = 0;
                            cleaned.data[pidx * 3ULL + 0] = 255;
                            cleaned.data[pidx * 3ULL + 1] = 255;
                            cleaned.data[pidx * 3ULL + 2] = 255;
                        }
                    }
                }
            }
        }

        // 1. 1D X-axis ink projection -> vertical columns
        std::vector<int> col_proj(static_cast<size_t>(W), 0);
        for (int y = 0; y < H; ++y) {
            const size_t row = static_cast<size_t>(y) * static_cast<size_t>(W);
            for (int x = 0; x < W; ++x) {
                col_proj[static_cast<size_t>(x)] += ink[row + static_cast<size_t>(x)];
            }
        }

        struct Span { int a = 0; int b = 0; };
        std::vector<Span> raw_cols;
        int x = 0;
        while (x < W) {
            while (x < W && col_proj[static_cast<size_t>(x)] == 0) ++x;
            if (x >= W) break;
            const int start = x;
            while (x < W && col_proj[static_cast<size_t>(x)] > 0) ++x;
            raw_cols.push_back({start, x});
        }
        // Merge tiny horizontal gaps (<= 3 px) inside the same column (e.g. left/right radicals)
        std::vector<Span> cols;
        for (const auto& sp : raw_cols) {
            if (!cols.empty() && (sp.a - cols.back().b) <= 3) {
                cols.back().b = sp.b;
            } else {
                cols.push_back(sp);
            }
        }
        // Filter noise columns and sort Right-to-Left (RTL)
        std::vector<Span> valid_cols;
        for (const auto& c : cols) {
            const int cw = c.b - c.a;
            if (cw < 6) continue;
            int col_ink = 0;
            for (int cx = c.a; cx < c.b; ++cx) col_ink += col_proj[static_cast<size_t>(cx)];
            if (col_ink < 12) continue;
            valid_cols.push_back(c);
        }
        if (valid_cols.empty()) return {};
        std::sort(valid_cols.begin(), valid_cols.end(), [](const Span& lhs, const Span& rhs) {
            return lhs.a > rhs.a;  // Right-to-Left column order
        });

        constexpr int kStripH = 48;
        constexpr int kGlyphTargetH = 38;
        constexpr int kGapX = 5;
        constexpr int kMarginX = 10;

        std::vector<RgbImage> packed_strips;
        for (const auto& col : valid_cols) {
            const int col_w = col.b - col.a;
            // 2. 1D Y-axis ink projection inside this vertical column
            std::vector<int> row_proj(static_cast<size_t>(H), 0);
            for (int y = 0; y < H; ++y) {
                const size_t row = static_cast<size_t>(y) * static_cast<size_t>(W);
                int sum = 0;
                for (int cx = col.a; cx < col.b; ++cx) {
                    sum += ink[row + static_cast<size_t>(cx)];
                }
                row_proj[static_cast<size_t>(y)] = sum;
            }

            std::vector<Span> glyphs;
            int y = 0;
            while (y < H) {
                while (y < H && row_proj[static_cast<size_t>(y)] == 0) ++y;
                if (y >= H) break;
                const int start = y;
                while (y < H && row_proj[static_cast<size_t>(y)] > 0) ++y;
                glyphs.push_back({start, y});
            }
            if (glyphs.empty()) continue;

            // 3. Radical-safe merge: merge adjacent vertical fragments whose combined height <= 1.15 * col_w
            const float max_char_h = std::max(14.0f, 1.15f * static_cast<float>(col_w));
            const int max_rad_gap = std::max(4, col_w / 3);
            std::vector<Span> merged;
            for (const auto& g : glyphs) {
                if (!merged.empty()) {
                    const int gap = g.a - merged.back().b;
                    const int combined_h = g.b - merged.back().a;
                    if (gap <= max_rad_gap && static_cast<float>(combined_h) <= max_char_h) {
                        merged.back().b = g.b;
                        continue;
                    }
                }
                merged.push_back(g);
            }

            // Extract tight horizontal bounds [gx1, gx2] per upright glyph
            struct GlyphBox { int x1, y1, x2, y2, dst_w, dst_h; };
            std::vector<GlyphBox> gboxes;
            int total_strip_w = kMarginX * 2;
            for (const auto& g : merged) {
                const int gh = g.b - g.a;
                if (gh < 5) continue;
                int gx1 = col.b, gx2 = col.a;
                int g_ink = 0;
                for (int gy = g.a; gy < g.b; ++gy) {
                    const size_t row = static_cast<size_t>(gy) * static_cast<size_t>(W);
                    for (int cx = col.a; cx < col.b; ++cx) {
                        if (ink[row + static_cast<size_t>(cx)]) {
                            if (cx < gx1) gx1 = cx;
                            if (cx > gx2) gx2 = cx;
                            ++g_ink;
                        }
                    }
                }
                if (gx2 < gx1 || g_ink < 8) continue;
                const int gw = gx2 - gx1 + 1;
                const float scale_h = static_cast<float>(kGlyphTargetH) / static_cast<float>(gh);
                const float scale_w = (static_cast<float>(kGlyphTargetH) * 1.15f) / static_cast<float>(gw);
                const float scale = std::min(scale_h, scale_w);
                const int dw = std::max(4, static_cast<int>(std::lround(static_cast<float>(gw) * scale)));
                const int dh = std::max(4, std::min(kStripH - 4, static_cast<int>(std::lround(static_cast<float>(gh) * scale))));
                if (!gboxes.empty()) total_strip_w += kGapX;
                total_strip_w += dw;
                gboxes.push_back({gx1, g.a, gx2 + 1, g.b, dw, dh});
            }
            if (gboxes.empty()) continue;

            // 4. Pack upright glyphs left-to-right onto white 48px horizontal strip (bilinear resample)
            RgbImage strip;
            strip.width = std::max(48, total_strip_w);
            strip.height = kStripH;
            strip.data.assign(static_cast<size_t>(strip.width) * static_cast<size_t>(strip.height) * 3ULL, 255);

            int cur_x = kMarginX;
            for (const auto& gb : gboxes) {
                const int sw = gb.x2 - gb.x1;
                const int sh = gb.y2 - gb.y1;
                const int off_y = (kStripH - gb.dst_h) / 2;
                for (int dy = 0; dy < gb.dst_h; ++dy) {
                    const float sy = (static_cast<float>(dy) + 0.5f) * static_cast<float>(sh) / static_cast<float>(gb.dst_h) - 0.5f;
                    const int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, sh - 1);
                    const int y1 = std::clamp(y0 + 1, 0, sh - 1);
                    const float fy = std::clamp(sy - static_cast<float>(y0), 0.0f, 1.0f);
                    for (int dx = 0; dx < gb.dst_w; ++dx) {
                        const float sx = (static_cast<float>(dx) + 0.5f) * static_cast<float>(sw) / static_cast<float>(gb.dst_w) - 0.5f;
                        const int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, sw - 1);
                        const int x1 = std::clamp(x0 + 1, 0, sw - 1);
                        const float fx = std::clamp(sx - static_cast<float>(x0), 0.0f, 1.0f);

                        const uint8_t* p00 = cleaned.data.data() + (static_cast<size_t>(gb.y1 + y0) * static_cast<size_t>(W) + static_cast<size_t>(gb.x1 + x0)) * 3ULL;
                        const uint8_t* p10 = cleaned.data.data() + (static_cast<size_t>(gb.y1 + y0) * static_cast<size_t>(W) + static_cast<size_t>(gb.x1 + x1)) * 3ULL;
                        const uint8_t* p01 = cleaned.data.data() + (static_cast<size_t>(gb.y1 + y1) * static_cast<size_t>(W) + static_cast<size_t>(gb.x1 + x0)) * 3ULL;
                        const uint8_t* p11 = cleaned.data.data() + (static_cast<size_t>(gb.y1 + y1) * static_cast<size_t>(W) + static_cast<size_t>(gb.x1 + x1)) * 3ULL;

                        uint8_t* dst_p = strip.data.data() + (static_cast<size_t>(off_y + dy) * static_cast<size_t>(strip.width) + static_cast<size_t>(cur_x + dx)) * 3ULL;
                        for (int c = 0; c < 3; ++c) {
                            const float top = static_cast<float>(p00[c]) * (1.0f - fx) + static_cast<float>(p10[c]) * fx;
                            const float bot = static_cast<float>(p01[c]) * (1.0f - fx) + static_cast<float>(p11[c]) * fx;
                            const float val = top * (1.0f - fy) + bot * fy;
                            dst_p[c] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(val)), 0, 255));
                        }
                    }
                }
                cur_x += gb.dst_w + kGapX;
            }
            packed_strips.push_back(std::move(strip));
        }
        return packed_strips;
    };

    auto run_vertical_repack_rec = [&](const RgbImage& vcrop) -> std::pair<std::string, float> {
        const auto vstrips = repack_vertical_to_horizontal_strips(vcrop);
        if (vstrips.empty()) return {"", 0.0f};
        std::string joined;
        double conf_sum = 0.0;
        int conf_cnt = 0;
        for (const auto& st : vstrips) {
            const auto [txt, conf] = run_ppocr_rec(st);
            const std::string t = util::trim_ascii(txt);
            if (!t.empty() && conf >= 0.25f) {
                if (!joined.empty() && rapid_loaded_lang_ == "ko") {
                    joined.push_back(' ');
                }
                joined.append(t);
                conf_sum += static_cast<double>(conf);
                ++conf_cnt;
            }
        }
        return {util::trim_ascii(joined), conf_cnt > 0 ? static_cast<float>(conf_sum / conf_cnt) : 0.0f};
    };

    const bool is_ko_or_zh = (rapid_loaded_lang_ == "ko" || rapid_loaded_lang_ == "zh" || rapid_loaded_lang_ == "cjk");
    const bool is_tall_crop = (static_cast<float>(raw_crop.height) > static_cast<float>(raw_crop.width) * 1.25f);

    std::vector<uint8_t> crop_ink;
    float ink_ratio = 0.0f;
    compute_crop_ink_mask(raw_crop.data.data(), raw_crop.width, raw_crop.height, crop_ink, ink_ratio);

    std::string horiz_best_text;
    float horiz_best_conf = 0.0f;

    // 0. Smart DBNet Fast-Path Bypass:
    //    (a) Narrow vertical single-column ZH/KO bubble (h > w * 2.0): skip DBNet -> vertical re-packer.
    //    (b) Single-line horizontal crop (w > h * 2.2 && h <= 72 && 1 continuous Y ink band): skip DBNet -> tight crop rec.
    if (is_ko_or_zh && static_cast<float>(raw_crop.height) > static_cast<float>(raw_crop.width) * 2.0f && ink_ratio >= 0.015f) {
        const auto vstrips = repack_vertical_to_horizontal_strips(raw_crop);
        if (vstrips.size() == 1) {
            const auto [vtxt, vconf] = run_ppocr_rec(vstrips[0]);
            const std::string vt = util::trim_ascii(vtxt);
            if (!vt.empty() && vconf >= 0.30f && count_target_script_chars(vt, rapid_loaded_lang_) >= 2) {
                return vt;
            }
            if (!vt.empty()) {
                horiz_best_text = vt;
                horiz_best_conf = vconf;
            }
        }
    } else if (static_cast<float>(raw_crop.width) > static_cast<float>(raw_crop.height) * 2.2f &&
               raw_crop.height <= 72 && ink_ratio >= 0.01f) {
        int y_bands = 0;
        bool in_band = false;
        for (int y = 0; y < raw_crop.height; ++y) {
            bool has_ink = false;
            const size_t row = static_cast<size_t>(y) * static_cast<size_t>(raw_crop.width);
            for (int x = 0; x < raw_crop.width; ++x) {
                if (crop_ink[row + static_cast<size_t>(x)]) { has_ink = true; break; }
            }
            if (has_ink && !in_band) { ++y_bands; in_band = true; }
            else if (!has_ink) { in_band = false; }
        }
        if (y_bands == 1) {
            const RgbImage tight = make_tight_ink_crop(raw_crop, crop_ink);
            const auto [htxt, hconf] = run_ppocr_rec(tight);
            const std::string ht = util::trim_ascii(htxt);
            if (!ht.empty() && hconf >= 0.65f && (!is_ko_or_zh || count_target_script_chars(ht, rapid_loaded_lang_) >= 1)) {
                return ht;
            }
        }
    }

    // 1. Fast path: horizontal line-strip segmentation -> direct PP-OCRv4 recognition per line
    if (!is_tall_crop || !is_ko_or_zh) {
        const auto strips = extract_horizontal_line_strips(raw_crop);
        if (!strips.empty()) {
            std::vector<std::string> texts;
            double score_sum = 0.0;
            for (const auto& strip : strips) {
                const auto [txt, conf] = run_ppocr_rec(strip);
                if (!txt.empty()) {
                    texts.push_back(txt);
                    score_sum += static_cast<double>(conf);
                }
            }
            if (!texts.empty()) {
                const float mean_conf = static_cast<float>(score_sum / static_cast<double>(texts.size()));
                std::string joined;
                for (size_t i = 0; i < texts.size(); ++i) {
                    if (i > 0) joined.push_back(' ');
                    joined.append(texts[i]);
                }
                joined = util::trim_ascii(joined);
                if (mean_conf >= 0.70f && (!is_ko_or_zh || count_target_script_chars(joined, rapid_loaded_lang_) >= 1)) {
                    return joined;
                }
                horiz_best_text = joined;
                horiz_best_conf = mean_conf;
            }
        }
    }

    // 2. Full-crop DBNet detection + vertical-box re-packing + orientation cls + PP-OCRv4 rec
    const RgbImage padded = pad_rgb_white(raw_crop, 18, 18, 18, 18);
    const auto det_boxes = run_ppocr_det(padded);
    if (!det_boxes.empty()) {
        std::string joined;
        double conf_sum = 0.0;
        int conf_cnt = 0;
        for (const auto& bb : det_boxes) {
            RgbImage sub;
            sub.width = bb[2] - bb[0];
            sub.height = bb[3] - bb[1];
            sub.data.resize(static_cast<size_t>(sub.width) * static_cast<size_t>(sub.height) * 3ULL);
            const size_t rb = static_cast<size_t>(sub.width) * 3ULL;
            for (int y = 0; y < sub.height; ++y) {
                std::memcpy(
                    sub.data.data() + static_cast<size_t>(y) * rb,
                    padded.data.data() +
                        (static_cast<size_t>(bb[1] + y) * static_cast<size_t>(padded.width) +
                         static_cast<size_t>(bb[0])) * 3ULL,
                    rb);
            }

            std::string txt;
            float conf = 0.0f;
            // Case (a): DBNet emitted a tall vertical text box (h > w * 1.35) for ZH/KO
            if (is_ko_or_zh && static_cast<float>(sub.height) > static_cast<float>(sub.width) * 1.35f) {
                auto [vtxt, vconf] = run_vertical_repack_rec(sub);
                auto [htxt, hconf] = run_ppocr_rec(sub);
                const int vc = count_target_script_chars(vtxt, rapid_loaded_lang_);
                const int hc = count_target_script_chars(htxt, rapid_loaded_lang_);
                if (vc > hc || (vc == hc && vconf >= hconf)) {
                    txt = std::move(vtxt);
                    conf = vconf;
                } else {
                    txt = std::move(htxt);
                    conf = hconf;
                }
            } else {
                std::tie(txt, conf) = run_ppocr_rec(sub);
                if (rapid_loaded_lang_ != "ko" && conf < 0.85f) {
                    const RgbImage oriented = apply_ppocr_cls_if_needed(sub);
                    if (oriented.data != sub.data) {
                        const auto [rot_txt, rot_conf] = run_ppocr_rec(oriented);
                        if (!rot_txt.empty() && rot_conf > conf + 0.10f) {
                            txt = rot_txt;
                            conf = rot_conf;
                        }
                    }
                }
            }
            if (!txt.empty() && conf >= 0.28f) {
                if (!joined.empty()) joined.push_back(' ');
                joined.append(txt);
                conf_sum += static_cast<double>(conf);
                ++conf_cnt;
            }
        }
        if (!joined.empty()) {
            const float db_conf = conf_cnt > 0 ? static_cast<float>(conf_sum / conf_cnt) : 0.0f;
            const int db_chars = count_target_script_chars(joined, rapid_loaded_lang_);
            const int prev_chars = count_target_script_chars(horiz_best_text, rapid_loaded_lang_);
            if (db_chars > prev_chars || (db_chars == prev_chars && db_conf >= horiz_best_conf)) {
                horiz_best_text = util::trim_ascii(joined);
                horiz_best_conf = db_conf;
            }
        }
    } else if (ink_ratio >= 0.015f) {
        // Zero-Box DBNet Fallback: bypass DBNet and run tight ink-cropped horizontal strip directly
        const RgbImage tight = make_tight_ink_crop(raw_crop, crop_ink);
        const auto [fb_txt, fb_conf] = run_ppocr_rec(tight);
        const int fb_chars = count_target_script_chars(fb_txt, rapid_loaded_lang_);
        const int prev_chars = count_target_script_chars(horiz_best_text, rapid_loaded_lang_);
        if (!fb_txt.empty() && (fb_chars > prev_chars || (fb_chars == prev_chars && fb_conf > horiz_best_conf))) {
            horiz_best_text = util::trim_ascii(fb_txt);
            horiz_best_conf = fb_conf;
        }
    }

    // 3. Case (b): Tall vertical bubble crop for ZH/KO -> run Vertical-Column Horizontal Strip Re-Packing
    //    and pick the best candidate by (target_script_chars, mean_conf).
    if (is_ko_or_zh && is_tall_crop && ink_ratio >= 0.015f) {
        const int h_chars = count_target_script_chars(horiz_best_text, rapid_loaded_lang_);
        if (h_chars < 2 || horiz_best_conf < 0.80f) {
            const auto [vert_txt, vert_conf] = run_vertical_repack_rec(raw_crop);
            const int v_chars = count_target_script_chars(vert_txt, rapid_loaded_lang_);
            if (v_chars > h_chars || (v_chars == h_chars && vert_conf > horiz_best_conf)) {
                return vert_txt;
            }
        }
    }

    if (!horiz_best_text.empty()) {
        return horiz_best_text;
    }

    // 4. Final single-line recognition fallback
    const auto [fallback_txt, _] = run_ppocr_rec(padded);
    return util::trim_ascii(fallback_txt);
}

std::vector<std::string> OcrEngine::recognize_rapid_page_batched(const std::vector<RgbImage>& clean_crops) {
    const size_t n = clean_crops.size();
    std::vector<std::string> out_texts(n);
    if (n == 0) return out_texts;

    // Phase 1: Collect candidate horizontal strips per bubble (using fast horizontal strip extraction
    // or DBNet when needed) WITHOUT running rec yet.
    struct StripRef {
        size_t owner = 0;
        RgbImage img;
    };
    std::vector<StripRef> all_strips;
    all_strips.reserve(n * 3ULL);
    std::vector<bool> needs_full_fallback(n, false);

    for (size_t idx = 0; idx < n; ++idx) {
        const auto& crop = clean_crops[idx];
        if (crop.width <= 0 || crop.height <= 0 || crop.data.empty()) continue;
        const bool is_ko_or_zh = (rapid_loaded_lang_ == "ko" || rapid_loaded_lang_ == "zh" || rapid_loaded_lang_ == "cjk");
        const bool is_tall = (static_cast<float>(crop.height) > static_cast<float>(crop.width) * 1.25f);
        if (is_ko_or_zh && is_tall) {
            // Let tall vertical bubbles use the dedicated vertical upright-glyph re-packer path
            needs_full_fallback[idx] = true;
            continue;
        }
        auto strips = extract_horizontal_line_strips(crop);
        if (!strips.empty()) {
            for (auto& s : strips) {
                all_strips.push_back({idx, std::move(s)});
            }
        } else {
            needs_full_fallback[idx] = true;
        }
    }

    // Phase 2: Width-sorted dynamic mini-batching (up to B=16 strips per ONNX rec call)
    if (!all_strips.empty()) {
        std::vector<RgbImage> strip_imgs;
        strip_imgs.reserve(all_strips.size());
        for (auto& sr : all_strips) strip_imgs.push_back(std::move(sr.img));

        const auto batch_res = run_ppocr_rec_batch(strip_imgs, /*max_batch=*/16);
        std::vector<double> conf_sum(n, 0.0);
        std::vector<int> conf_cnt(n, 0);
        for (size_t k = 0; k < batch_res.size(); ++k) {
            const size_t owner = all_strips[k].owner;
            const std::string t = util::trim_ascii(batch_res[k].first);
            if (!t.empty()) {
                if (!out_texts[owner].empty()) out_texts[owner].push_back(' ');
                out_texts[owner].append(t);
                conf_sum[owner] += static_cast<double>(batch_res[k].second);
                conf_cnt[owner] += 1;
            }
        }
        for (size_t idx = 0; idx < n; ++idx) {
            if (!needs_full_fallback[idx]) {
                const float mean_c = (conf_cnt[idx] > 0) ? static_cast<float>(conf_sum[idx] / conf_cnt[idx]) : 0.0f;
                if (out_texts[idx].empty() || mean_c < 0.70f) {
                    needs_full_fallback[idx] = true;
                }
            }
        }
    }

    // Phase 3: Run full DBNet / vertical re-pack fallback ONLY on bubbles that needed it
    std::vector<size_t> fb_indices;
    for (size_t idx = 0; idx < n; ++idx) {
        if (needs_full_fallback[idx]) fb_indices.push_back(idx);
    }
    if (!fb_indices.empty()) {
        const bool is_cpu = (rapid_rec_session_->active_provider == "CPUExecutionProvider");
        if (is_cpu && fb_indices.size() > 1) {
            const size_t num_workers = std::min(fb_indices.size(), static_cast<size_t>(ocr_cpu_worker_count()));
            std::atomic<size_t> next_k{0};
            std::vector<std::thread> workers;
            workers.reserve(num_workers);
            for (size_t w = 0; w < num_workers; ++w) {
                workers.emplace_back([&]() {
                    while (true) {
                        const size_t k = next_k.fetch_add(1, std::memory_order_relaxed);
                        if (k >= fb_indices.size()) break;
                        const size_t idx = fb_indices[k];
                        const std::string fb = recognize_rapid_crop(clean_crops[idx]);
                        if (!fb.empty()) out_texts[idx] = fb;
                    }
                });
            }
            for (auto& th : workers) {
                if (th.joinable()) th.join();
            }
        } else {
            for (size_t idx : fb_indices) {
                const std::string fb = recognize_rapid_crop(clean_crops[idx]);
                if (!fb.empty()) out_texts[idx] = fb;
            }
        }
    }

    return out_texts;
}

// -----------------------------------------------------------------------------
// Public Recognition API
// -----------------------------------------------------------------------------

OcrStageResult OcrEngine::recognize_from_bytes(
    const uint8_t* image_bytes,
    size_t num_bytes,
    const std::vector<OcrBubbleInput>& bubbles,
    const OcrOptions& options) {
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
            return recognize_rgb(wic_rgb.data(), w, h, bubbles, options);
        }
        const char* reason = stbi_failure_reason();
        throw std::invalid_argument(
            std::string("Failed to decode image via stb_image: ") + (reason ? reason : "invalid format"));
    }

    return recognize_rgb(rgb.get(), w, h, bubbles, options);
}

std::string OcrEngine::resolve_paddleocr_vl_lmstudio_model(const OcrOptions& options) const {
    const auto local_gguf = model_manager_.resolve_paddleocr_vl_gguf();
    const std::string gguf_hint = local_gguf.has_value()
        ? (" Local GGUF: " + util::path_to_utf8(local_gguf->model_gguf) + ".")
        : " Place PaddleOCR-VL-1.6-GGUF in ./models/paddleocr-vl-gguf/ or load it in LM Studio.";

    const auto ep = parse_local_http_endpoint(options.lmstudio_base_url);
    if (!ep.valid) {
        throw std::runtime_error(
            "PaddleOCR-VL-1.6-GGUF Vision-OCR requires a local HTTP LM Studio endpoint (e.g. http://localhost:1234/v1)." +
            gguf_hint);
    }

    httplib::Client cli(ep.host, ep.port);
    cli.set_connection_timeout(3, 0);
    cli.set_read_timeout(5, 0);
    httplib::Headers headers = {{"Accept", "application/json"}};
    if (!options.lmstudio_api_key.empty()) {
        headers.emplace("Authorization", "Bearer " + options.lmstudio_api_key);
    }

    const auto res = cli.Get((ep.base_path + "/models").c_str(), headers);
    if (!res || res->status < 200 || res->status >= 300) {
        throw std::runtime_error(
            "PaddleOCR-VL-1.6-GGUF requires LM Studio running at " + options.lmstudio_base_url +
            " with PaddleOCR-VL-1.6-GGUF loaded." + gguf_hint +
            " Switch ocr_model to 'auto', 'manga-ocr', or 'rapidocr-ppocrv4' for direct ONNX Runtime OCR.");
    }

    const json body = json::parse(res->body, nullptr, false);
    std::string matched_model;
    if (!body.is_discarded() && body.is_object() && body.contains("data") && body["data"].is_array()) {
        for (const auto& entry : body["data"]) {
            if (!entry.is_object() || !entry.contains("id") || !entry["id"].is_string()) continue;
            const std::string id = entry["id"].get<std::string>();
            const std::string lower = util::to_lower_ascii(id);
            if (lower.find("paddleocr") != std::string::npos || lower.find("paddle-ocr") != std::string::npos) {
                matched_model = id;
                break;
            }
        }
    }

    if (matched_model.empty()) {
        throw std::runtime_error(
            "LM Studio is reachable at " + options.lmstudio_base_url +
            ", but PaddleOCR-VL-1.6-GGUF is not loaded." + gguf_hint +
            " Load PaddleOCR-VL-1.6-GGUF in LM Studio or switch ocr_model to 'auto', 'manga-ocr', or 'rapidocr-ppocrv4'.");
    }

    return matched_model;
}

std::string OcrEngine::recognize_paddleocr_vl_crop(
    const RgbImage& raw_crop,
    const std::string& model_id,
    const OcrOptions& options) const {
    const auto ep = parse_local_http_endpoint(options.lmstudio_base_url);
    if (!ep.valid) {
        throw std::runtime_error("Invalid LM Studio base URL for PaddleOCR-VL-GGUF Vision-OCR.");
    }

    const RgbImage padded = pad_rgb_white(raw_crop, 8, 8, 8, 8);
    const std::string data_uri = ComicTranslator::encode_rgb_for_vision(
        padded.data.data(), padded.width, padded.height, 1024, 92);

    const json payload = {
        {"model", model_id},
        {"temperature", 0.0},
        {"max_tokens", 512},
        {"messages", json::array({
            {
                {"role", "user"},
                {"content", json::array({
                    {{"type", "image_url"}, {"image_url", {{"url", data_uri}}}},
                    {{"type", "text"}, {"text", "OCR:"}},
                })},
            },
        })},
    };

    httplib::Client cli(ep.host, ep.port);
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(60, 0);
    httplib::Headers headers = {{"Accept", "application/json"}};
    if (!options.lmstudio_api_key.empty()) {
        headers.emplace("Authorization", "Bearer " + options.lmstudio_api_key);
    }

    const auto res = cli.Post(
        (ep.base_path + "/chat/completions").c_str(),
        headers,
        payload.dump(-1, ' ', false, json::error_handler_t::replace),
        "application/json");
    if (!res || res->status < 200 || res->status >= 300) {
        throw std::runtime_error(
            "PaddleOCR-VL-1.6-GGUF Vision-OCR request to LM Studio failed (HTTP " +
            std::to_string(res ? res->status : 0) + ").");
    }

    const json body = json::parse(res->body, nullptr, false);
    if (body.is_discarded() || !body.is_object() ||
        !body.contains("choices") || !body["choices"].is_array() || body["choices"].empty()) {
        return {};
    }
    const auto& msg = body["choices"][0];
    if (!msg.is_object() || !msg.contains("message") || !msg["message"].is_object() ||
        !msg["message"].contains("content") || !msg["message"]["content"].is_string()) {
        return {};
    }
    return util::trim_ascii(ComicTranslator::strip_thinking_tags(msg["message"]["content"].get<std::string>()));
}

OcrStageResult OcrEngine::recognize_rgb(
    const uint8_t* rgb_data,
    int width,
    int height,
    const std::vector<OcrBubbleInput>& bubbles,
    const OcrOptions& options) {
    if (!rgb_data || width <= 0 || height <= 0) {
        throw std::invalid_argument("Invalid RGB image buffer or dimensions.");
    }

    const std::string lang = TextDetector::normalize_source_language(options.source_lang);
    const std::string requested_model = normalize_ocr_model(options.ocr_model);
    std::string effective_model = resolve_effective_ocr_model(requested_model, lang);
    // If an optional model explicitly chosen for ko/zh isn't installed, fall back cleanly to rapidocr-ppocrv4
    if ((lang == "ko" || lang == "zh") && effective_model == "hayai-ocr" &&
        !model_manager_.resolve_hayai_ocr_models().has_value()) {
        effective_model = "rapidocr-ppocrv4";
    }
    if (lang == "zh" && effective_model == "baberu-ocr" &&
        !model_manager_.resolve_baberu_ocr_models().has_value()) {
        effective_model = "rapidocr-ppocrv4";
    }
    ScopeOcrUnload unload_guard{this, options.auto_unload};

    OcrStageResult result;
    result.source_lang = lang;
    result.requested_model = requested_model;
    result.ocr_model = effective_model;
    result.blocks.reserve(bubbles.size());

    if (bubbles.empty()) {
        result.engine = (effective_model == "rapidocr-ppocrv4") ? "rapidocr" : effective_model;
        result.execution_provider = ModelManager::get_primary_provider(options.force_cpu);
        return result;
    }

    std::string paddle_vl_model_id;
    bool paddle_vl_fallback_to_local = false;
    if (effective_model == "manga-ocr") {
        result.engine = "manga-ocr";
        ensure_manga_ocr_loaded(options.force_cpu);
        result.execution_provider = manga_enc_session_->active_provider;
    } else if (effective_model == "baberu-ocr") {
        result.engine = "baberu-ocr";
        ensure_baberu_ocr_loaded();
        result.execution_provider = "CPUExecutionProvider";
    } else if (effective_model == "hayai-ocr") {
        result.engine = "hayai-ocr";
        // Hayai OCR v2.1 is allowed onto GPU/CPU
        ensure_hayai_ocr_loaded(false);
        result.execution_provider = hayai_enc_session_->active_provider;
    } else if (effective_model == "rapidocr-ppocrv4") {
        result.engine = "rapidocr";
        ensure_rapid_ocr_loaded(lang, options.force_cpu);
        result.execution_provider = rapid_rec_session_->active_provider;
    } else {
        result.engine = "paddleocr-vl-1.6";
        try {
            paddle_vl_model_id = resolve_paddleocr_vl_lmstudio_model(options);
            result.execution_provider = "LMStudioVisionOCR";
        } catch (...) {
            // Graceful fallback when PaddleOCR-VL vision server is not loaded in LM Studio
            paddle_vl_fallback_to_local = true;
            if (lang != "ko" && model_manager_.resolve_baberu_ocr_models().has_value()) {
                ensure_baberu_ocr_loaded();
                result.ocr_model = "baberu-ocr";
                result.engine = "baberu-ocr";
                result.execution_provider = "CPUExecutionProvider";
            } else if (lang == "ja" && model_manager_.resolve_manga_ocr_models().has_value()) {
                ensure_manga_ocr_loaded(options.force_cpu);
                result.ocr_model = "manga-ocr";
                result.engine = "manga-ocr";
                result.execution_provider = manga_enc_session_->active_provider;
            } else {
                ensure_rapid_ocr_loaded(lang, options.force_cpu);
                result.ocr_model = "rapidocr-ppocrv4";
                result.engine = "rapidocr";
                result.execution_provider = rapid_rec_session_->active_provider;
            }
        }
    }

    // Precompute per-block glyph masks so overlapping neighbor dialogue is blanked
    // with the crop's median unmasked background RGB before ONNX OCR inference.
    const std::vector<BlockGlyphMask> block_glyph_masks =
        build_block_glyph_masks(rgb_data, width, height, bubbles, options);

    // Component-aware bubble-wall border cleanup removes curved bubble outlines
    // (which otherwise cause `へへ`/`ノ`/`〈〉` hallucinations) while preserving interior glyphs.
    const bool clean_border =
        (effective_model == "rapidocr-ppocrv4" ||
         effective_model == "manga-ocr" ||
         effective_model == "baberu-ocr" ||
         effective_model == "hayai-ocr");
    std::vector<RgbImage> clean_crops;
    clean_crops.reserve(bubbles.size());
    for (size_t idx = 0; idx < bubbles.size(); ++idx) {
        std::array<int, 4> crop_rect{0, 0, 0, 0};
        RgbImage clean_crop = crop_bbox_clean(
            rgb_data, width, height, bubbles[idx].bbox, clean_border, &crop_rect);
        if (blank_neighbor_masks_in_crop(
                clean_crop.data, clean_crop.width, clean_crop.height,
                crop_rect, idx, block_glyph_masks)) {
            ++result.neighbor_blanked_blocks;
        }
        // #11: Stroke Width Transform (SWT) Opt-In Furigana Masking (temporary OCR crop ONLY;
        // never alters inpainting text_mask).
        if (lang == "ja" && options.auto_mask_furigana_ocr &&
            bubbles[idx].text_direction == "vertical") {
            math_ocr::maskFuriganaSwtInCrop(
                clean_crop.data.data(), clean_crop.width, clean_crop.height);
        }
        clean_crops.push_back(std::move(clean_crop));
    }

    // #10: Perona-Malik Anisotropic Diffusion Denoiser (0 ms on clean digital scans;
    // smooths JPEG ringing/grain on degraded scans while preserving sharp text edges).
    // Deferred per-crop right before inference (runs in parallel inside the worker pool).
    std::vector<uint8_t> crop_denoised(bubbles.size(), 0);
    auto ensure_crop_denoised = [&](size_t idx) {
        if (!crop_denoised[idx]) {
            math_ocr::denoiseCropPeronaMalikIfNoisy(
                clean_crops[idx].data.data(), clean_crops[idx].width, clean_crops[idx].height);
            crop_denoised[idx] = 1;
        }
    };
    auto denoise_all_crops_for_batch = [&]() {
        if (bubbles.size() <= 1) {
            if (!bubbles.empty()) ensure_crop_denoised(0);
            return;
        }
        const size_t n_workers = std::min(bubbles.size(), static_cast<size_t>(ocr_cpu_worker_count()));
        std::atomic<size_t> cur{0};
        std::vector<std::thread> threads;
        threads.reserve(n_workers);
        for (size_t w = 0; w < n_workers; ++w) {
            threads.emplace_back([&]() {
                while (true) {
                    const size_t i = cur.fetch_add(1, std::memory_order_relaxed);
                    if (i >= bubbles.size()) break;
                    ensure_crop_denoised(i);
                }
            });
        }
        for (auto& th : threads) {
            if (th.joinable()) th.join();
        }
    };

    const bool using_manga =
        (effective_model == "manga-ocr") ||
        (paddle_vl_fallback_to_local && !baberu_vis_session_.has_value() && manga_enc_session_.has_value());
    const bool using_rapid =
        (effective_model == "rapidocr-ppocrv4") ||
        (paddle_vl_fallback_to_local && !baberu_vis_session_.has_value() && !manga_enc_session_.has_value());

    std::vector<std::string> recognized_texts(bubbles.size());
    std::vector<float> recognized_entropies(bubbles.size(), 0.0f);
    bool handled_by_cuda_batch = false;

    auto is_gpu_batch_provider = [](const std::string& ep) noexcept {
        return ep == "CUDAExecutionProvider" || ep == "DmlExecutionProvider";
    };

    if (using_manga && bubbles.size() > 1 &&
        manga_enc_session_.has_value() && manga_dec_session_.has_value() &&
        is_gpu_batch_provider(manga_enc_session_->active_provider) &&
        (is_gpu_batch_provider(manga_dec_session_->active_provider) ||
         manga_enc_session_->active_provider == "DmlExecutionProvider") &&
        manga_supports_dynamic_batch()) {
        denoise_all_crops_for_batch();
        try {
            std::vector<RgbImage> batch_crops;
            std::vector<size_t> crop_owners;
            std::vector<bool> used_strips(bubbles.size(), false);
            batch_crops.reserve(bubbles.size() * 2ULL);
            crop_owners.reserve(bubbles.size() * 2ULL);

            for (size_t idx = 0; idx < bubbles.size(); ++idx) {
                if (bubbles[idx].text_direction == "horizontal") {
                    auto strips = extract_horizontal_line_strips(clean_crops[idx]);
                    if (!strips.empty()) {
                        used_strips[idx] = true;
                        for (auto& s : strips) {
                            batch_crops.push_back(std::move(s));
                            crop_owners.push_back(idx);
                        }
                        continue;
                    }
                }
                const auto sq = math_ocr::trimAndSquarePadRgb(
                    clean_crops[idx].data.data(), clean_crops[idx].width, clean_crops[idx].height, nullptr, 16);
                batch_crops.push_back(RgbImage{sq.width, sq.height, sq.data});
                crop_owners.push_back(idx);
            }

            std::vector<float> batch_entropies;
            const auto batch_texts = run_manga_ocr_batch(batch_crops, /*max_batch=*/8, &batch_entropies);
            std::vector<float> owner_ent_sum(bubbles.size(), 0.0f);
            std::vector<int> owner_ent_cnt(bubbles.size(), 0);
            for (size_t k = 0; k < batch_texts.size(); ++k) {
                const std::string t = util::trim_ascii(batch_texts[k]);
                if (!t.empty()) {
                    const size_t owner = crop_owners[k];
                    std::string& dst = recognized_texts[owner];
                    if (!dst.empty()) dst.push_back(' ');
                    dst.append(t);
                    if (k < batch_entropies.size()) {
                        owner_ent_sum[owner] += batch_entropies[k];
                        owner_ent_cnt[owner] += 1;
                    }
                }
            }
            for (size_t idx = 0; idx < bubbles.size(); ++idx) {
                if (owner_ent_cnt[idx] > 0) {
                    recognized_entropies[idx] = owner_ent_sum[idx] / static_cast<float>(owner_ent_cnt[idx]);
                }
            }

            // Fallback for horizontal bubbles whose line strips all yielded empty text
            std::vector<RgbImage> fallback_crops;
            std::vector<size_t> fallback_owners;
            for (size_t idx = 0; idx < bubbles.size(); ++idx) {
                if (used_strips[idx] && recognized_texts[idx].empty()) {
                    fallback_crops.push_back(pad_rgb_white(clean_crops[idx], 8, 8, 8, 8));
                    fallback_owners.push_back(idx);
                }
            }
            if (!fallback_crops.empty()) {
                std::vector<float> fb_entropies;
                const auto fb_texts = run_manga_ocr_batch(fallback_crops, /*max_batch=*/8, &fb_entropies);
                for (size_t k = 0; k < fb_texts.size(); ++k) {
                    recognized_texts[fallback_owners[k]] = util::trim_ascii(fb_texts[k]);
                    if (k < fb_entropies.size()) {
                        recognized_entropies[fallback_owners[k]] = fb_entropies[k];
                    }
                }
            }
            handled_by_cuda_batch = true;
        } catch (const Ort::Exception&) {
            // Static batch or CUDA dynamic-batch error: fall back cleanly to worker pool
            std::fill(recognized_texts.begin(), recognized_texts.end(), std::string{});
            std::fill(recognized_entropies.begin(), recognized_entropies.end(), 0.0f);
            handled_by_cuda_batch = false;
        }
    }

    const bool using_baberu =
        (effective_model == "baberu-ocr") ||
        (paddle_vl_fallback_to_local && baberu_vis_session_.has_value());
    const bool using_hayai = (effective_model == "hayai-ocr");

    // Fast path for RapidOCR dynamic rec mini-batching (B <= 16 width-sorted)
    if (!handled_by_cuda_batch && using_rapid && rapid_rec_dynamic_batch_ && bubbles.size() >= 2) {
        denoise_all_crops_for_batch();
        try {
            recognized_texts = recognize_rapid_page_batched(clean_crops);
            handled_by_cuda_batch = true;
        } catch (const Ort::Exception&) {
            std::fill(recognized_texts.begin(), recognized_texts.end(), std::string{});
            handled_by_cuda_batch = false;
        }
    }

    // Optional Vision Encoder mini-batching for Baberu OCR when running on CUDA/DirectML with dynamic batch dim
    if (!handled_by_cuda_batch && using_baberu && baberu_vis_dynamic_batch_ && bubbles.size() >= 2 &&
        baberu_vis_session_.has_value() && is_gpu_batch_provider(baberu_vis_session_->active_provider)) {
        denoise_all_crops_for_batch();
        try {
            constexpr int kImgSize = 224;
            constexpr size_t kPlane = static_cast<size_t>(kImgSize) * static_cast<size_t>(kImgSize);
            constexpr size_t kImgStride = 3ULL * kPlane;
            constexpr size_t kMaxBatch = 8;
            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            const char* vis_in_names[] = {"pixel_values"};
            const char* vis_out_names[] = {"vision_embeds"};

            for (size_t start = 0; start < bubbles.size(); start += kMaxBatch) {
                const size_t bn = std::min(kMaxBatch, bubbles.size() - start);
                std::vector<float> batch_px(bn * kImgStride);
                for (size_t b = 0; b < bn; ++b) {
                    const auto& c = clean_crops[start + b];
                    const auto sq = math_ocr::trimAndSquarePadRgb(c.data.data(), c.width, c.height, nullptr, 16);
                    const auto res = resize_rgb_bilinear(sq.data.data(), sq.width, sq.height, kImgSize, kImgSize);
                    float* dst = batch_px.data() + b * kImgStride;
                    for (size_t i = 0; i < kPlane; ++i) {
                        dst[i] = (static_cast<float>(res[i * 3ULL + 0]) / 255.0f - 0.485f) / 0.229f;
                        dst[kPlane + i] = (static_cast<float>(res[i * 3ULL + 1]) / 255.0f - 0.456f) / 0.224f;
                        dst[2ULL * kPlane + i] = (static_cast<float>(res[i * 3ULL + 2]) / 255.0f - 0.406f) / 0.225f;
                    }
                }
                const std::array<int64_t, 4> in_shape = {static_cast<int64_t>(bn), 3, kImgSize, kImgSize};
                Ort::Value vis_in = Ort::Value::CreateTensor<float>(
                    mem_info, batch_px.data(), batch_px.size(), in_shape.data(), in_shape.size());
                auto vis_outs = baberu_vis_session_->session->Run(
                    Ort::RunOptions{nullptr}, vis_in_names, &vis_in, 1, vis_out_names, 1);
                const auto out_shape = vis_outs[0].GetTensorTypeAndShapeInfo().GetShape();
                if (out_shape.size() < 2 || static_cast<size_t>(out_shape[0]) != bn) {
                    throw std::runtime_error("Unexpected batched Baberu vision output shape");
                }
                std::vector<int64_t> single_shape = out_shape;
                single_shape[0] = 1;
                size_t per_item_elems = 1;
                for (size_t d = 1; d < out_shape.size(); ++d) {
                    per_item_elems *= static_cast<size_t>(std::max<int64_t>(1, out_shape[d]));
                }
                const float* all_vis = vis_outs[0].GetTensorData<float>();
                for (size_t b = 0; b < bn; ++b) {
                    recognized_texts[start + b] = decode_baberu_from_vision_embeds(
                        all_vis + b * per_item_elems, single_shape, &recognized_entropies[start + b]);
                }
            }
            handled_by_cuda_batch = true;
        } catch (...) {
            std::fill(recognized_texts.begin(), recognized_texts.end(), std::string{});
            std::fill(recognized_entropies.begin(), recognized_entropies.end(), 0.0f);
            handled_by_cuda_batch = false;
        }
    }

    if (!handled_by_cuda_batch) {
        const bool hayai_is_cpu = using_hayai &&
            hayai_enc_session_.has_value() && hayai_dec_session_.has_value() &&
            hayai_enc_session_->active_provider == "CPUExecutionProvider" &&
            hayai_dec_session_->active_provider == "CPUExecutionProvider";
        const bool rapid_is_cpu = using_rapid &&
            rapid_rec_session_.has_value() &&
            rapid_rec_session_->active_provider == "CPUExecutionProvider";
        const bool can_parallel_pool =
            (using_manga || rapid_is_cpu || using_baberu || hayai_is_cpu) && bubbles.size() > 1;
        if (can_parallel_pool) {
            const size_t num_workers = std::min(
                bubbles.size(),
                static_cast<size_t>(ocr_cpu_worker_count()));
            std::atomic<size_t> next_idx{0};
            std::mutex err_mutex;
            std::exception_ptr first_err = nullptr;
            std::vector<std::thread> workers;
            workers.reserve(num_workers);

            for (size_t w = 0; w < num_workers; ++w) {
                workers.emplace_back([&]() {
                    try {
                        while (true) {
                            const size_t idx = next_idx.fetch_add(1, std::memory_order_relaxed);
                            if (idx >= bubbles.size()) break;
                            ensure_crop_denoised(idx);
                            if (using_manga) {
                                recognized_texts[idx] =
                                    recognize_ja_crop(clean_crops[idx], bubbles[idx].text_direction, &recognized_entropies[idx]);
                            } else if (using_baberu) {
                                recognized_texts[idx] =
                                    recognize_baberu_crop(clean_crops[idx], &recognized_entropies[idx]);
                            } else if (using_hayai) {
                                recognized_texts[idx] =
                                    recognize_hayai_crop(clean_crops[idx], &recognized_entropies[idx]);
                            } else {
                                recognized_texts[idx] = recognize_rapid_crop(clean_crops[idx]);
                            }
                        }
                    } catch (...) {
                        std::lock_guard<std::mutex> lock(err_mutex);
                        if (!first_err) first_err = std::current_exception();
                    }
                });
            }
            for (auto& th : workers) {
                if (th.joinable()) th.join();
            }
            if (first_err) {
                std::rethrow_exception(first_err);
            }
        } else {
            for (size_t idx = 0; idx < bubbles.size(); ++idx) {
                ensure_crop_denoised(idx);
                const auto& b = bubbles[idx];
                const auto& clean_crop = clean_crops[idx];
                if (effective_model == "manga-ocr") {
                    recognized_texts[idx] = recognize_ja_crop(clean_crop, b.text_direction, &recognized_entropies[idx]);
                } else if (effective_model == "baberu-ocr") {
                    recognized_texts[idx] = recognize_baberu_crop(clean_crop, &recognized_entropies[idx]);
                } else if (effective_model == "hayai-ocr") {
                    recognized_texts[idx] = recognize_hayai_crop(clean_crop, &recognized_entropies[idx]);
                } else if (effective_model == "rapidocr-ppocrv4") {
                    recognized_texts[idx] = recognize_rapid_crop(clean_crop);
                } else if (!paddle_vl_fallback_to_local) {
                    recognized_texts[idx] = recognize_paddleocr_vl_crop(clean_crop, paddle_vl_model_id, options);
                } else if (baberu_vis_session_.has_value()) {
                    recognized_texts[idx] = recognize_baberu_crop(clean_crop, &recognized_entropies[idx]);
                } else if (manga_enc_session_.has_value()) {
                    recognized_texts[idx] = recognize_ja_crop(clean_crop, b.text_direction, &recognized_entropies[idx]);
                } else {
                    recognized_texts[idx] = recognize_rapid_crop(clean_crop);
                }
            }
        }
    }

    auto sanitize_ocr_text = [](const std::string& raw, const std::string& /*source_lang*/) -> std::string {
        if (raw.empty()) return {};
        const auto cps = decode_utf8_codepoints(raw);
        if (cps.empty()) return {};

        auto is_punct_or_space = [](uint32_t cp) -> bool {
            if (cp <= 0x20u || cp == 0x3000u) return true;
            if ((cp >= 0x21u && cp <= 0x2Fu) || (cp >= 0x3Au && cp <= 0x40u) ||
                (cp >= 0x5Bu && cp <= 0x60u) || (cp >= 0x7Bu && cp <= 0x7Eu)) {
                return true;
            }
            if ((cp >= 0x3001u && cp <= 0x303Fu) || (cp >= 0xFF01u && cp <= 0xFF0Fu) ||
                (cp >= 0xFF1Au && cp <= 0xFF20u) || (cp >= 0xFF3Bu && cp <= 0xFF40u) ||
                (cp >= 0xFF5Bu && cp <= 0xFF65u) || cp == 0x2026u || cp == 0x30FBu ||
                cp == 0x30FCu || cp == 0xFF70u) {
                return true;
            }
            return false;
        };

        // 1. Strip zero-width & ASCII control chars
        std::vector<uint32_t> filtered;
        filtered.reserve(cps.size());
        for (uint32_t cp : cps) {
            if (cp == 0x200Bu || cp == 0x200Cu || cp == 0x200Du || cp == 0xFEFFu) {
                continue;
            }
            if (cp < 0x20u || cp == 0x7Fu) {
                if (cp == '\n' || cp == '\r' || cp == '\t') {
                    filtered.push_back(' ');
                }
                continue;
            }
            filtered.push_back(cp);
        }

        // 2. Collapse runaway OCR hallucination loops
        std::vector<uint32_t> collapsed;
        collapsed.reserve(filtered.size());
        size_t i = 0;
        while (i < filtered.size()) {
            const uint32_t cp = filtered[i];
            size_t j = i + 1;
            while (j < filtered.size() && filtered[j] == cp) {
                ++j;
            }
            const size_t run = j - i;
            if ((cp == 0x30FCu || cp == 0xFF70u) && run >= 5) {
                // ー / ｰ >= 5 -> ーーー
                for (int k = 0; k < 3; ++k) collapsed.push_back(0x30FCu);
            } else if (cp == 0x30FBu && run >= 3) {
                // ・ >= 3 -> …
                collapsed.push_back(0x2026u);
            } else if (!is_punct_or_space(cp) && run >= 8) {
                // Same non-punctuation char repeated >= 8 -> cap at 4
                for (int k = 0; k < 4; ++k) collapsed.push_back(cp);
            } else {
                for (size_t k = i; k < j; ++k) collapsed.push_back(filtered[k]);
            }
            i = j;
        }

        std::string out;
        out.reserve(collapsed.size() * 3);
        for (uint32_t cp : collapsed) {
            append_utf8_codepoint(out, cp);
        }
        out = util::trim_ascii(out);
        if (out.empty()) return {};

        // 3. Drop pure single-dot/artifact noise
        const auto final_cps = decode_utf8_codepoints(out);
        if (final_cps.size() == 1) {
            const uint32_t c = final_cps[0];
            if (c == '.' || c == '\'' || c == ',' || c == '-' || c == '"' || c == '`' ||
                c == '_' || c == ':' || c == ';' || c == 0x30FBu || c == 0x3002u ||
                c == 0x3001u || c == 0xFF0Eu || c == 0xFF0Cu || c == 0x30FCu || c == 0xFF70u) {
                return {};
            }
        }
        return out;
    };

    auto is_exempt_short_qc_bubble = [](const std::string& text) -> bool {
        if (text.empty()) return true;
        const auto cps = decode_utf8_codepoints(text);
        if (cps.empty()) return true;
        auto is_punct_sym = [](uint32_t cp) noexcept -> bool {
            if (cp <= 0x20u) return true;
            if ((cp >= 0x21u && cp <= 0x2Fu) || (cp >= 0x3Au && cp <= 0x40u) ||
                (cp >= 0x5Bu && cp <= 0x60u) || (cp >= 0x7Bu && cp <= 0x7Eu)) return true;
            if ((cp >= 0x2000u && cp <= 0x206Fu) || (cp >= 0x2600u && cp <= 0x27BFu)) return true;
            if ((cp >= 0x3000u && cp <= 0x303Fu) || (cp >= 0xFF01u && cp <= 0xFF0Fu) ||
                (cp >= 0xFF1Au && cp <= 0xFF20u) || (cp >= 0xFF3Bu && cp <= 0xFF40u) ||
                (cp >= 0xFF5Bu && cp <= 0xFF65u)) return true;
            return false;
        };
        auto is_cjk_ideograph = [](uint32_t cp) noexcept -> bool {
            return (cp >= 0x4E00u && cp <= 0x9FFFu) ||
                   (cp >= 0x3400u && cp <= 0x4DBFu) ||
                   (cp >= 0xF900u && cp <= 0xFAFFu);
        };
        size_t non_punct = 0;
        bool has_kanji_hanzi = false;
        for (uint32_t cp : cps) {
            if (!is_punct_sym(cp)) {
                ++non_punct;
                if (is_cjk_ideograph(cp)) has_kanji_hanzi = true;
            }
        }
        // Exempt pure punctuation/symbols ("...", "…", "!?", "！？", "!!", "?")
        // and <= 2 non-kanji/non-hanzi reaction characters ("あ", "っ!", "으").
        return (non_punct == 0) || (non_punct <= 2 && !has_kanji_hanzi);
    };

    for (size_t idx = 0; idx < bubbles.size(); ++idx) {
        const auto& b = bubbles[idx];
        std::string sanitized = sanitize_ocr_text(recognized_texts[idx], lang);
        OcrBubbleOutput out;
        out.id = b.id;
        out.order = b.order;
        out.bbox = b.bbox;
        out.polygon = b.polygon;
        out.original_text = sanitized;
        out.source_text = std::move(sanitized);
        out.confidence = b.confidence;
        out.ocr_entropy = recognized_entropies[idx];
        out.low_ocr_confidence =
            (!is_exempt_short_qc_bubble(out.source_text) && out.ocr_entropy > 1.75f);
        if (out.low_ocr_confidence) {
            result.low_confidence_ocr_ids.push_back(out.id);
        }
        out.text_direction = b.text_direction;
        result.blocks.push_back(std::move(out));
    }

    return result;
}

}  // namespace mangaflow::pipeline
