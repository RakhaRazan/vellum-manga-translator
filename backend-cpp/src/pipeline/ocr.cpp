#include <httplib.h>

#include "ocr.hpp"
#include "detector.hpp"
#include "translator.hpp"
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
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace mangaflow::pipeline {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct StbiDeleter {
    void operator()(unsigned char* p) const noexcept {
        if (p) stbi_image_free(p);
    }
};
using StbiPtr = std::unique_ptr<unsigned char, StbiDeleter>;

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

std::vector<uint8_t> resize_rgb_bilinear(
    const uint8_t* src,
    int src_w,
    int src_h,
    int dst_w,
    int dst_h) {
    std::vector<uint8_t> dst(static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h) * 3ULL, 255);
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
    int port = 1235;
    std::string base_path = "/v1";
};

ParsedLocalEndpoint parse_local_http_endpoint(const std::string& raw_url) {
    ParsedLocalEndpoint out;
    std::string s = util::trim_ascii(raw_url);
    if (s.empty()) {
        s = "http://localhost:1235/v1";
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

}  // namespace

// -----------------------------------------------------------------------------
// OcrBubbleOutput JSON Serialization
// -----------------------------------------------------------------------------

json OcrBubbleOutput::to_json() const {
    const double conf_rounded = std::round(static_cast<double>(confidence) * 10000.0) / 10000.0;
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
    if (cleaned == "paddleocr-vl-gguf" || cleaned == "paddleocr_vl_gguf" ||
        cleaned == "paddleocr-vl" || cleaned == "paddleocr_vl" ||
        cleaned == "paddle-vl" || cleaned == "paddle_vl" ||
        cleaned == "paddleocr-vl-1.6-gguf") {
        return "paddleocr-vl-gguf";
    }
    if (cleaned == "rapidocr-ppocrv4" || cleaned == "rapidocr_ppocrv4" ||
        cleaned == "rapidocr" || cleaned == "rapid-ocr" || cleaned == "rapid_ocr" ||
        cleaned == "ppocrv4" || cleaned == "pp-ocrv4") {
        return "rapidocr-ppocrv4";
    }
    throw std::invalid_argument(
        "Unsupported ocr_model '" + raw +
        "'. Supported options: 'auto', 'manga-ocr', 'paddleocr-vl-gguf', 'rapidocr-ppocrv4'.");
}

std::string OcrEngine::resolve_effective_ocr_model(
    const std::string& ocr_model,
    const std::string& source_lang) {
    const std::string norm = normalize_ocr_model(ocr_model);
    if (norm != "auto") {
        return norm;
    }
    const std::string lang = TextDetector::normalize_source_language(source_lang);
    return (lang == "ja") ? "manga-ocr" : "rapidocr-ppocrv4";
}

void OcrEngine::unload() {
    manga_enc_session_.reset();
    manga_dec_session_.reset();
    manga_vocab_.clear();
    manga_force_cpu_ = false;
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
    for (size_t i = 0; i < total; ++i) {
        const uint8_t* px = crop.data.data() + i * 3ULL;
        if (rgb_to_luma(px[0], px[1], px[2]) < 210) {
            ink[i] = 1;
        }
    }

    // Flood-fill 8-connected ink pixels starting strictly from the 1px outer perimeter
    std::vector<uint8_t> border_mask(total, 0);
    std::queue<int> q;
    auto push_if_ink = [&](int x, int y) {
        const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x);
        if (ink[idx] && !border_mask[idx]) {
            border_mask[idx] = 1;
            q.push(static_cast<int>(idx));
        }
    };

    for (int x = 0; x < w; ++x) {
        push_if_ink(x, 0);
        push_if_ink(x, h - 1);
    }
    for (int y = 0; y < h; ++y) {
        push_if_ink(0, y);
        push_if_ink(w - 1, y);
    }

    if (q.empty()) {
        return;
    }

    static constexpr int kDx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static constexpr int kDy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    while (!q.empty()) {
        const int cur = q.front();
        q.pop();
        const int cx = cur % w;
        const int cy = cur / w;
        for (int k = 0; k < 8; ++k) {
            const int nx = cx + kDx[k];
            const int ny = cy + kDy[k];
            if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                push_if_ink(nx, ny);
            }
        }
    }

    // 3x3 dilation to clear anti-aliased gray halo around erased bubble outline strokes
    std::vector<uint8_t> dilated(total, 0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (!border_mask[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)]) {
                continue;
            }
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

    for (size_t i = 0; i < total; ++i) {
        if (dilated[i]) {
            uint8_t* px = crop.data.data() + i * 3ULL;
            px[0] = 255;
            px[1] = 255;
            px[2] = 255;
        }
    }
}

OcrEngine::RgbImage OcrEngine::crop_bbox_clean(
    const uint8_t* rgb_data,
    int img_w,
    int img_h,
    const std::array<int, 4>& bbox) {
    int x1 = clampi(bbox[0], 0, img_w);
    int y1 = clampi(bbox[1], 0, img_h);
    int x2 = clampi(bbox[2], 0, img_w);
    int y2 = clampi(bbox[3], 0, img_h);

    RgbImage crop;
    if (x2 <= x1 || y2 <= y1) {
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

    erase_border_touching_ink(crop);
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
            ". Call POST /api/models/import-koharu or run packaging/convert_koharu_to_onnx.bat first.");
    }

    manga_vocab_ = load_vocab_lines(resolved->vocab_txt);
    manga_enc_session_ = ModelManager::create_ort_session(resolved->encoder_onnx, force_cpu);

    // Note: decoder_model.onnx has dynamic `seq_len` (1..N per step). On DirectML, dynamic shapes
    // trigger shader recompilation per token step, whereas CPU executes the 2-layer BERT decoder
    // in ~2 ms/step with zero shader stalls. Keep CUDA if available, otherwise CPU for decoder.
    const bool dec_force_cpu = force_cpu || (manga_enc_session_->active_provider == "DmlExecutionProvider");
    manga_dec_session_ = ModelManager::create_ort_session(resolved->decoder_onnx, dec_force_cpu);
    manga_force_cpu_ = force_cpu;
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

std::string OcrEngine::run_manga_ocr_single(const RgbImage& crop) {
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
    auto run_encoder = [&](OrtSessionHandle& handle) -> std::vector<Ort::Value> {
        const char* in_names[] = {"pixel_values"};
        const char* out_names[] = {"last_hidden_state"};
        return handle.session->Run(Ort::RunOptions{nullptr}, in_names, &enc_input, 1, out_names, 1);
    };

    std::vector<Ort::Value> enc_outputs;
    try {
        enc_outputs = run_encoder(*manga_enc_session_);
    } catch (const Ort::Exception&) {
        if (manga_enc_session_->active_provider == "CPUExecutionProvider") {
            throw;
        }
        const auto resolved = model_manager_.resolve_manga_ocr_models();
        manga_enc_session_ = ModelManager::create_ort_session(resolved->encoder_onnx, /*force_cpu=*/true);
        enc_outputs = run_encoder(*manga_enc_session_);
    }

    const auto enc_shape = enc_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const size_t enc_elem_count = enc_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    const float* enc_ptr = enc_outputs[0].GetTensorData<float>();
    std::vector<float> enc_hidden(enc_ptr, enc_ptr + enc_elem_count);

    Ort::Value enc_hidden_tensor = Ort::Value::CreateTensor<float>(
        mem_info, enc_hidden.data(), enc_hidden.size(), enc_shape.data(), enc_shape.size());

    // 3. Autoregressive greedy BERT decoder loop (`input_ids` + `encoder_hidden_states` -> `logits`)
    std::vector<int64_t> token_ids;
    token_ids.reserve(64);
    token_ids.push_back(kBosTokenId);

    const char* dec_in_names[] = {"input_ids", "encoder_hidden_states"};
    const char* dec_out_names[] = {"logits"};

    for (size_t step = 0; step < kMaxTokens; ++step) {
        const std::array<int64_t, 2> ids_shape = {1, static_cast<int64_t>(token_ids.size())};
        Ort::Value ids_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info, token_ids.data(), token_ids.size(), ids_shape.data(), ids_shape.size());

        std::array<Ort::Value, 2> dec_inputs = {
            std::move(ids_tensor),
            Ort::Value::CreateTensor<float>(
                mem_info, enc_hidden.data(), enc_hidden.size(), enc_shape.data(), enc_shape.size()),
        };

        auto dec_outputs = manga_dec_session_->session->Run(
            Ort::RunOptions{nullptr},
            dec_in_names,
            dec_inputs.data(),
            dec_inputs.size(),
            dec_out_names,
            1);

        const auto logits_shape = dec_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        if (logits_shape.size() != 3 || logits_shape[2] <= 0) {
            break;
        }
        const size_t seq_len = static_cast<size_t>(logits_shape[1]);
        const size_t vocab_sz = static_cast<size_t>(logits_shape[2]);
        const float* last_logits = dec_outputs[0].GetTensorData<float>() + (seq_len - 1ULL) * vocab_sz;

        int64_t best_id = kEosTokenId;
        float best_val = -std::numeric_limits<float>::infinity();
        for (size_t v = 0; v < vocab_sz; ++v) {
            if (last_logits[v] > best_val) {
                best_val = last_logits[v];
                best_id = static_cast<int64_t>(v);
            }
        }

        if (best_id == kEosTokenId || best_id == kPadTokenId) {
            break;
        }
        token_ids.push_back(best_id);
    }

    return decode_manga_ocr_tokens(token_ids);
}

std::string OcrEngine::recognize_ja_crop(const RgbImage& raw_crop, const std::string& text_direction) {
    if (text_direction == "horizontal") {
        const auto strips = extract_horizontal_line_strips(raw_crop);
        if (!strips.empty()) {
            std::string combined;
            for (const auto& strip : strips) {
                const std::string t = util::trim_ascii(run_manga_ocr_single(strip));
                if (!t.empty()) {
                    if (!combined.empty()) combined.push_back(' ');
                    combined.append(t);
                }
            }
            if (!combined.empty()) {
                return combined;
            }
        }
    }

    const RgbImage padded = pad_rgb_white(raw_crop, 8, 8, 8, 8);
    return util::trim_ascii(run_manga_ocr_single(padded));
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

    rapid_det_session_ = ModelManager::create_ort_session(paths->det_model_path, use_cpu);
    rapid_cls_session_ = ModelManager::create_ort_session(paths->cls_model_path, use_cpu);
    rapid_rec_session_ = ModelManager::create_ort_session(paths->rec_model_path, use_cpu);
    rapid_charset_ = load_rapidocr_charset_from_session(*rapid_rec_session_->session);
    rapid_loaded_lang_ = profile;
    rapid_force_cpu_ = use_cpu;
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

    auto outputs = rapid_rec_session_->session->Run(
        Ort::RunOptions{nullptr}, in_names, &ort_in, 1, out_names, 1);

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

std::string OcrEngine::recognize_rapid_crop(const RgbImage& raw_crop) {
    // 1. Fast path: horizontal line-strip segmentation -> direct PP-OCRv4 recognition per line
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
        if (!texts.empty() && (score_sum / static_cast<double>(texts.size())) >= 0.70) {
            std::string joined;
            for (size_t i = 0; i < texts.size(); ++i) {
                if (i > 0) joined.push_back(' ');
                joined.append(texts[i]);
            }
            return util::trim_ascii(joined);
        }
    }

    // 2. Full-crop DBNet detection + orientation cls + PP-OCRv4 recognition fallback
    const RgbImage padded = pad_rgb_white(raw_crop, 18, 18, 18, 18);
    const auto det_boxes = run_ppocr_det(padded);
    if (!det_boxes.empty()) {
        std::string joined;
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
            auto [txt, conf] = run_ppocr_rec(sub);
            // Only consult the Chinese/English 180-degree orientation classifier for `zh` when
            // upright recognition confidence is low, and only keep the rotated output if rec
            // confidence genuinely improves. (Applying ch_ppocr_mobile_v2.0_cls to Korean Hangul
            // frequently misclassifies upright Hangul lines as upside-down.)
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
            if (!txt.empty() && conf >= 0.30f) {
                if (!joined.empty()) joined.push_back(' ');
                joined.append(txt);
            }
        }
        if (!joined.empty()) {
            return util::trim_ascii(joined);
        }
    }

    // 3. Single-line recognition fallback when detector finds no boxes on a tight crop
    const auto [fallback_txt, _] = run_ppocr_rec(padded);
    return util::trim_ascii(fallback_txt);
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
        : " Call POST /api/models/import-koharu to import PaddleOCR-VL-1.6-GGUF first.";

    const auto ep = parse_local_http_endpoint(options.lmstudio_base_url);
    if (!ep.valid) {
        throw std::runtime_error(
            "PaddleOCR-VL-1.6-GGUF Vision-OCR requires a local HTTP LM Studio endpoint (e.g. http://localhost:1235/v1)." +
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
    const std::string effective_model = resolve_effective_ocr_model(requested_model, lang);
    ScopeOcrUnload unload_guard{this, options.auto_unload};

    OcrStageResult result;
    result.source_lang = lang;
    result.requested_model = requested_model;
    result.ocr_model = effective_model;
    result.blocks.reserve(bubbles.size());

    std::string paddle_vl_model_id;
    if (effective_model == "manga-ocr") {
        result.engine = "manga-ocr";
        ensure_manga_ocr_loaded(options.force_cpu);
        result.execution_provider = manga_enc_session_->active_provider;
    } else if (effective_model == "rapidocr-ppocrv4") {
        result.engine = "rapidocr";
        ensure_rapid_ocr_loaded(lang, options.force_cpu);
        result.execution_provider = rapid_rec_session_->active_provider;
    } else {
        result.engine = "paddleocr-vl-gguf";
        paddle_vl_model_id = resolve_paddleocr_vl_lmstudio_model(options);
        result.execution_provider = "LMStudioVisionOCR";
    }

    for (const auto& b : bubbles) {
        const RgbImage clean_crop = crop_bbox_clean(rgb_data, width, height, b.bbox);
        std::string text;
        if (effective_model == "manga-ocr") {
            text = recognize_ja_crop(clean_crop, b.text_direction);
        } else if (effective_model == "rapidocr-ppocrv4") {
            text = recognize_rapid_crop(clean_crop);
        } else {
            text = recognize_paddleocr_vl_crop(clean_crop, paddle_vl_model_id, options);
        }

        OcrBubbleOutput out;
        out.id = b.id;
        out.order = b.order;
        out.bbox = b.bbox;
        out.polygon = b.polygon;
        out.original_text = text;
        out.source_text = std::move(text);
        out.confidence = b.confidence;
        out.text_direction = b.text_direction;
        result.blocks.push_back(std::move(out));
    }

    return result;
}

}  // namespace mangaflow::pipeline
