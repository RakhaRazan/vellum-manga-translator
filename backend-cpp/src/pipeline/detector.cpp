#include "detector.hpp"
#include "../detector/cv_geometry.hpp"
#include "../detector/cv_refinement.hpp"
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
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace mangaflow::pipeline {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

using util::StbiPtr;
using util::resize_rgb_bilinear;

struct ScopeUnload {
    TextDetector* detector = nullptr;
    bool enabled = false;
    ~ScopeUnload() {
        if (enabled && detector) {
            detector->unload();
        }
    }
};

inline float sigmoidf(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

inline float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

inline int clampi(int v, int lo, int hi) {
    return std::max(lo, std::min(hi, v));
}

// Bilinear sample from a 2D float map `(src_w, src_h)` at continuous coords `(sx, sy)`.
inline float sample_bilinear_f32(
    const float* map,
    int src_w,
    int src_h,
    float sx,
    float sy) {
    const float cx = clampf(sx, 0.0f, static_cast<float>(src_w - 1));
    const float cy = clampf(sy, 0.0f, static_cast<float>(src_h - 1));
    const int x0 = static_cast<int>(std::floor(cx));
    const int y0 = static_cast<int>(std::floor(cy));
    const int x1 = std::min(x0 + 1, src_w - 1);
    const int y1 = std::min(y0 + 1, src_h - 1);
    const float wx1 = cx - static_cast<float>(x0);
    const float wy1 = cy - static_cast<float>(y0);
    const float wx0 = 1.0f - wx1;
    const float wy0 = 1.0f - wy1;

    const float* r0 = map + static_cast<size_t>(y0) * static_cast<size_t>(src_w);
    const float* r1 = map + static_cast<size_t>(y1) * static_cast<size_t>(src_w);
    return wx0 * wy0 * r0[x0] + wx1 * wy0 * r0[x1] + wx0 * wy1 * r1[x0] + wx1 * wy1 * r1[x1];
}

void stbi_write_vec_callback(void* context, void* data, int size) {
    if (!context || !data || size <= 0) return;
    auto* vec = static_cast<std::vector<uint8_t>*>(context);
    const auto* bytes = static_cast<const uint8_t*>(data);
    vec->insert(vec->end(), bytes, bytes + size);
}

bool resolve_is_rtl(const std::string& source_language, const std::string& reading_order) {
    const std::string ro = util::to_lower_ascii(util::trim_ascii(reading_order));
    if (ro == "rtl") return true;
    if (ro == "ltr") return false;
    return util::to_lower_ascii(util::trim_ascii(source_language)) == "ja";
}

double round_4(double v) {
    return std::round(v * 10000.0) / 10000.0;
}

}  // namespace

// -----------------------------------------------------------------------------
// DetectedBubble / LayoutRegion / AppliedDetectionThresholds JSON Serialization
// -----------------------------------------------------------------------------

json DetectedBubble::to_json() const {
    const double conf_rounded = round_4(static_cast<double>(confidence));
    json poly = json::array();
    for (const auto& pt : polygon) {
        poly.push_back(json::array({pt[0], pt[1]}));
    }
    json j = {
        {"id", id},
        {"order", order},
        {"bbox", {bbox[0], bbox[1], bbox[2], bbox[3]}},
        {"bubble_bbox", {bubble_bbox[0], bubble_bbox[1], bubble_bbox[2], bubble_bbox[3]}},
        {"safe_bbox", {safe_bbox[0], safe_bbox[1], safe_bbox[2], safe_bbox[3]}},
        {"polygon", std::move(poly)},
        {"confidence", conf_rounded},
        {"text_direction", text_direction},
        {"category", category},
    };
    if (!room_polygon.empty()) {
        json rpoly = json::array();
        for (const auto& pt : room_polygon) {
            rpoly.push_back(json::array({round_4(pt[0]), round_4(pt[1])}));
        }
        j["room_polygon"] = std::move(rpoly);
    }
    if (angle_deg.has_value()) {
        j["angle_deg"] = round_4(static_cast<double>(*angle_deg));
    }
    if (estimated_font_size.has_value()) {
        j["estimated_font_size"] = round_4(static_cast<double>(*estimated_font_size));
    }
    if (text_color.has_value()) {
        j["text_color"] = json::array({
            static_cast<int>((*text_color)[0]),
            static_cast<int>((*text_color)[1]),
            static_cast<int>((*text_color)[2])
        });
    }
    if (outline.has_value()) {
        j["outline"] = {
            {"color", json::array({
                static_cast<int>(outline->color[0]),
                static_cast<int>(outline->color[1]),
                static_cast<int>(outline->color[2])
            })},
            {"width", outline->width}
        };
    } else if (angle_deg.has_value()) {
        j["outline"] = nullptr;
    }
    return j;
}

json LayoutRegion::to_json() const {
    return {
        {"id", id},
        {"order", order},
        {"bbox", {bbox[0], bbox[1], bbox[2], bbox[3]}},
        {"confidence", round_4(static_cast<double>(confidence))},
        {"category", category},
    };
}

json AppliedDetectionThresholds::to_json() const {
    return {
        {"text", round_4(static_cast<double>(text))},
        {"onomatopoeia", round_4(static_cast<double>(onomatopoeia))},
        {"bubble", bubble.has_value() ? json(round_4(static_cast<double>(*bubble))) : json(nullptr)},
        {"panel", panel.has_value() ? json(round_4(static_cast<double>(*panel))) : json(nullptr)},
        {"nms", round_4(static_cast<double>(nms))},
        {"mask", round_4(static_cast<double>(mask))},
    };
}

// -----------------------------------------------------------------------------
// TextDetector Construction & Static Utilities
// -----------------------------------------------------------------------------

TextDetector::TextDetector(const ModelManager& model_manager)
    : model_manager_(model_manager) {}

void TextDetector::unload() {
    cached_session_.reset();
    cached_model_path_.clear();
    cached_force_cpu_ = false;
}

std::string TextDetector::normalize_source_language(const std::string& raw) {
    const std::string s = util::to_lower_ascii(util::trim_ascii(raw));
    if (s.empty() || s == "ja" || s == "jp" || s == "jpn" || s == "japanese" || s == "manga") {
        return "ja";
    }
    if (s == "ko" || s == "kr" || s == "kor" || s == "korean" || s == "manhwa") {
        return "ko";
    }
    if (s == "zh" || s == "cn" || s == "zho" || s == "chinese" || s == "manhua" ||
        s == "zh-cn" || s == "zh-tw" || s == "zh_cn" || s == "zh_tw") {
        return "zh";
    }
    throw std::invalid_argument(
        "Unsupported source_lang '" + raw + "'. Supported comic source languages: 'ja', 'ko', 'zh'.");
}

std::vector<uint8_t> TextDetector::encode_mask_png(
    const std::vector<uint8_t>& mask,
    int width,
    int height) {
    if (width <= 0 || height <= 0 ||
        mask.size() < static_cast<size_t>(width) * static_cast<size_t>(height)) {
        throw std::invalid_argument("Invalid binary mask dimensions for PNG encoding.");
    }
    std::vector<uint8_t> png_bytes;
    png_bytes.reserve(static_cast<size_t>(width) * static_cast<size_t>(height) / 16 + 1024);
    const int ok = stbi_write_png_to_func(
        &stbi_write_vec_callback,
        &png_bytes,
        width,
        height,
        1,
        mask.data(),
        width);
    if (!ok || png_bytes.empty()) {
        throw std::runtime_error("Failed to encode binary text mask to PNG.");
    }
    return png_bytes;
}

std::string TextDetector::base64_encode(const uint8_t* data, size_t size) {
    return util::base64_encode(data, size);
}

std::string TextDetector::mask_to_data_uri(
    const std::vector<uint8_t>& mask,
    int width,
    int height) {
    const auto png = encode_mask_png(mask, width, height);
    return "data:image/png;base64," + base64_encode(png.data(), png.size());
}

// -----------------------------------------------------------------------------
// NMS, Reading Order Sort & Elliptical Morphological Dilation
// -----------------------------------------------------------------------------

std::vector<size_t> TextDetector::nms_xyxy(
    const std::vector<std::array<float, 4>>& boxes,
    const std::vector<float>& scores,
    float iou_threshold) {
    const size_t n = boxes.size();
    if (n == 0) return {};

    std::vector<float> areas(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        areas[i] = std::max(0.0f, boxes[i][2] - boxes[i][0]) *
                   std::max(0.0f, boxes[i][3] - boxes[i][1]);
    }

    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return scores[a] > scores[b];
    });

    std::vector<size_t> keep;
    keep.reserve(n);
    std::vector<uint8_t> suppressed(n, 0);

    for (size_t oi = 0; oi < n; ++oi) {
        const size_t i = order[oi];
        if (suppressed[i]) continue;
        keep.push_back(i);

        const auto& bi = boxes[i];
        const float area_i = areas[i];
        for (size_t oj = oi + 1; oj < n; ++oj) {
            const size_t j = order[oj];
            if (suppressed[j]) continue;

            const auto& bj = boxes[j];
            const float xx1 = std::max(bi[0], bj[0]);
            const float yy1 = std::max(bi[1], bj[1]);
            const float xx2 = std::min(bi[2], bj[2]);
            const float yy2 = std::min(bi[3], bj[3]);

            const float inter = std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1);
            if (inter <= 0.0f) continue;

            const float area_j = areas[j];
            const float iou = inter / (area_i + area_j - inter + 1e-6f);
            const float io_min = inter / (std::min(area_i, area_j) + 1e-6f);
            if (iou > iou_threshold || io_min > 0.75f) {
                suppressed[j] = 1;
            }
        }
    }

    return keep;
}

std::vector<LayoutRegion> TextDetector::sort_layout_regions(
    const std::vector<std::pair<std::array<int, 4>, float>>& raw_regions,
    const std::string& category,
    const std::string& id_prefix,
    bool rtl,
    int page_height) {
    if (raw_regions.empty()) {
        return {};
    }

    std::vector<int> heights;
    heights.reserve(raw_regions.size());
    for (const auto& r : raw_regions) {
        heights.push_back(std::max(1, r.first[3] - r.first[1]));
    }
    std::sort(heights.begin(), heights.end());
    const double median_h = static_cast<double>(heights[heights.size() / 2]);
    const double row_tolerance = std::max(median_h * 0.6, static_cast<double>(page_height) * 0.05);

    std::vector<size_t> by_cy(raw_regions.size());
    std::iota(by_cy.begin(), by_cy.end(), 0);
    std::sort(by_cy.begin(), by_cy.end(), [&](size_t a, size_t b) {
        const double cy_a = 0.5 * static_cast<double>(raw_regions[a].first[1] + raw_regions[a].first[3]);
        const double cy_b = 0.5 * static_cast<double>(raw_regions[b].first[1] + raw_regions[b].first[3]);
        return cy_a < cy_b;
    });

    std::vector<std::vector<size_t>> rows;
    for (size_t idx : by_cy) {
        const double cy = 0.5 * static_cast<double>(raw_regions[idx].first[1] + raw_regions[idx].first[3]);
        if (rows.empty()) {
            rows.push_back({idx});
            continue;
        }
        auto& last_row = rows.back();
        double sum_cy = 0.0;
        for (size_t r_idx : last_row) {
            sum_cy += 0.5 * static_cast<double>(raw_regions[r_idx].first[1] + raw_regions[r_idx].first[3]);
        }
        const double mean_cy = sum_cy / static_cast<double>(last_row.size());
        if (std::abs(cy - mean_cy) <= row_tolerance) {
            last_row.push_back(idx);
        } else {
            rows.push_back({idx});
        }
    }

    std::vector<LayoutRegion> out;
    out.reserve(raw_regions.size());
    int order_num = 1;
    for (auto& row : rows) {
        std::sort(row.begin(), row.end(), [&](size_t a, size_t b) {
            const double cx_a = 0.5 * static_cast<double>(raw_regions[a].first[0] + raw_regions[a].first[2]);
            const double cx_b = 0.5 * static_cast<double>(raw_regions[b].first[0] + raw_regions[b].first[2]);
            return rtl ? (cx_a > cx_b) : (cx_a < cx_b);
        });
        for (size_t idx : row) {
            char id_buf[64]{};
            std::snprintf(id_buf, sizeof(id_buf), "%s_%03d", id_prefix.c_str(), order_num);
            LayoutRegion reg;
            reg.id = id_buf;
            reg.order = order_num;
            reg.bbox = raw_regions[idx].first;
            reg.confidence = raw_regions[idx].second;
            reg.category = category;
            out.push_back(std::move(reg));
            ++order_num;
        }
    }
    return out;
}

std::vector<DetectedBubble> TextDetector::sort_reading_order(
    const std::vector<RawBox>& raw_boxes,
    const std::string& source_language,
    int page_height,
    std::vector<int>* sorted_query_indices_out,
    const std::string& reading_order,
    const std::vector<LayoutRegion>* panels) {
    if (sorted_query_indices_out) {
        sorted_query_indices_out->clear();
    }
    if (raw_boxes.empty()) {
        return {};
    }

    const bool rtl = resolve_is_rtl(source_language, reading_order);

    auto order_indices_by_rows = [&](const std::vector<size_t>& subset,
                                     const std::function<std::array<int, 4>(size_t)>& get_box,
                                     double tol_factor) -> std::vector<size_t> {
        if (subset.size() <= 1) {
            return subset;
        }
        std::vector<int> heights;
        heights.reserve(subset.size());
        for (size_t idx : subset) {
            const auto bb = get_box(idx);
            heights.push_back(std::max(1, bb[3] - bb[1]));
        }
        std::sort(heights.begin(), heights.end());
        const double median_h = static_cast<double>(heights[heights.size() / 2]);
        const double row_tolerance = std::max(median_h * tol_factor, static_cast<double>(page_height) * 0.05);

        std::vector<size_t> by_cy = subset;
        std::sort(by_cy.begin(), by_cy.end(), [&](size_t a, size_t b) {
            const auto ba = get_box(a);
            const auto bb = get_box(b);
            return (ba[1] + ba[3]) < (bb[1] + bb[3]);
        });

        std::vector<std::vector<size_t>> rows;
        for (size_t idx : by_cy) {
            const auto bb = get_box(idx);
            const double cy = 0.5 * static_cast<double>(bb[1] + bb[3]);
            if (rows.empty()) {
                rows.push_back({idx});
                continue;
            }
            auto& last_row = rows.back();
            double sum_cy = 0.0;
            for (size_t r_idx : last_row) {
                const auto rb = get_box(r_idx);
                sum_cy += 0.5 * static_cast<double>(rb[1] + rb[3]);
            }
            const double mean_cy = sum_cy / static_cast<double>(last_row.size());
            if (std::abs(cy - mean_cy) <= row_tolerance) {
                last_row.push_back(idx);
            } else {
                rows.push_back({idx});
            }
        }

        std::vector<size_t> ordered;
        ordered.reserve(subset.size());
        for (auto& row : rows) {
            std::sort(row.begin(), row.end(), [&](size_t a, size_t b) {
                const auto ba = get_box(a);
                const auto bb = get_box(b);
                const double cx_a = 0.5 * static_cast<double>(ba[0] + ba[2]);
                const double cx_b = 0.5 * static_cast<double>(bb[0] + bb[2]);
                return rtl ? (cx_a > cx_b) : (cx_a < cx_b);
            });
            for (size_t idx : row) {
                ordered.push_back(idx);
            }
        }
        return ordered;
    };

    std::vector<size_t> final_order;
    final_order.reserve(raw_boxes.size());

    if (panels != nullptr && panels->size() >= 2) {
        // Panel-aware reading order: group each text box into its smallest containing panel,
        // sort panel groups + unassigned boxes in page reading order, then sort text boxes within each panel.
        struct SortGroup {
            std::array<int, 4> anchor_bbox{0, 0, 0, 0};
            std::vector<size_t> box_indices;
        };

        std::vector<std::vector<size_t>> panel_buckets(panels->size());
        std::vector<size_t> unassigned;

        for (size_t i = 0; i < raw_boxes.size(); ++i) {
            const auto& bb = raw_boxes[i].bbox;
            const double cx = 0.5 * static_cast<double>(bb[0] + bb[2]);
            const double cy = 0.5 * static_cast<double>(bb[1] + bb[3]);

            int best_panel = -1;
            double best_area = std::numeric_limits<double>::infinity();
            for (size_t p = 0; p < panels->size(); ++p) {
                const auto& pb = (*panels)[p].bbox;
                if (cx >= static_cast<double>(pb[0]) && cx <= static_cast<double>(pb[2]) &&
                    cy >= static_cast<double>(pb[1]) && cy <= static_cast<double>(pb[3])) {
                    const double area = static_cast<double>(std::max(1, pb[2] - pb[0])) *
                                        static_cast<double>(std::max(1, pb[3] - pb[1]));
                    if (area < best_area) {
                        best_area = area;
                        best_panel = static_cast<int>(p);
                    }
                }
            }
            if (best_panel >= 0) {
                panel_buckets[static_cast<size_t>(best_panel)].push_back(i);
            } else {
                unassigned.push_back(i);
            }
        }

        std::vector<SortGroup> groups;
        for (size_t p = 0; p < panels->size(); ++p) {
            if (!panel_buckets[p].empty()) {
                groups.push_back({(*panels)[p].bbox, std::move(panel_buckets[p])});
            }
        }
        for (size_t idx : unassigned) {
            groups.push_back({raw_boxes[idx].bbox, {idx}});
        }

        std::vector<size_t> group_ids(groups.size());
        std::iota(group_ids.begin(), group_ids.end(), 0);
        const auto ordered_groups = order_indices_by_rows(
            group_ids,
            [&](size_t gi) { return groups[gi].anchor_bbox; },
            0.6);

        for (size_t gi : ordered_groups) {
            const auto ordered_inner = order_indices_by_rows(
                groups[gi].box_indices,
                [&](size_t bi) { return raw_boxes[bi].bbox; },
                0.8);
            for (size_t bi : ordered_inner) {
                final_order.push_back(bi);
            }
        }
    } else {
        std::vector<size_t> all_ids(raw_boxes.size());
        std::iota(all_ids.begin(), all_ids.end(), 0);
        final_order = order_indices_by_rows(
            all_ids,
            [&](size_t bi) { return raw_boxes[bi].bbox; },
            0.8);
    }

    std::vector<DetectedBubble> sorted;
    sorted.reserve(raw_boxes.size());
    if (sorted_query_indices_out) {
        sorted_query_indices_out->reserve(raw_boxes.size());
    }

    int order_num = 1;
    for (size_t idx : final_order) {
        const auto& rb = raw_boxes[idx];
        const int x1 = rb.bbox[0];
        const int y1 = rb.bbox[1];
        const int x2 = rb.bbox[2];
        const int y2 = rb.bbox[3];

        char id_buf[32]{};
        std::snprintf(id_buf, sizeof(id_buf), "bubble_%03d", order_num);

        DetectedBubble bubble;
        bubble.id = id_buf;
        bubble.order = order_num;
        bubble.bbox = {x1, y1, x2, y2};
        bubble.polygon = {{{x1, y1}, {x2, y1}, {x2, y2}, {x1, y2}}};
        bubble.confidence = rb.score;
        bubble.text_direction = rb.direction;
        bubble.category = rb.category;
        sorted.push_back(std::move(bubble));

        if (sorted_query_indices_out) {
            sorted_query_indices_out->push_back(rb.source_query_index);
        }
        ++order_num;
    }

    return sorted;
}

void TextDetector::dilate_binary_mask_ellipse(
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

// -----------------------------------------------------------------------------
// Session Lifecycle & Public Entry Points
// -----------------------------------------------------------------------------

OrtSessionHandle& TextDetector::ensure_session(
    const ResolvedDetectorModel& resolved,
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

DetectionResult TextDetector::detect_from_bytes(
    const uint8_t* image_bytes,
    size_t num_bytes,
    const DetectionOptions& options) {
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
            return detect_rgb(wic_rgb.data(), w, h, options);
        }
        const char* reason = stbi_failure_reason();
        throw std::invalid_argument(
            std::string("Failed to decode image via stb_image: ") + (reason ? reason : "invalid format"));
    }

    return detect_rgb(rgb.get(), w, h, options);
}

DetectionResult TextDetector::detect_from_file(
    const fs::path& image_path,
    const DetectionOptions& options) {
    const auto bytes = util::read_file_bytes(image_path);
    if (!bytes.has_value() || bytes->empty()) {
        throw std::runtime_error("Failed to read image file: " + util::path_to_utf8(image_path));
    }
    return detect_from_bytes(bytes->data(), bytes->size(), options);
}

DetectionResult TextDetector::detect_rgb(
    const uint8_t* rgb_data,
    int width,
    int height,
    const DetectionOptions& options) {
    if (!rgb_data || width <= 0 || height <= 0) {
        throw std::invalid_argument("Invalid RGB image buffer or dimensions.");
    }

    DetectionOptions normalized_opts = options;
    normalized_opts.source_language = normalize_source_language(options.source_language);

    const std::string requested_model = !util::trim_ascii(normalized_opts.detection_model).empty()
        ? normalized_opts.detection_model
        : normalized_opts.detector_backend;
    const auto resolved = model_manager_.resolve_detector_model(
        requested_model,
        normalized_opts.allow_model_fallback);
    if (!resolved.has_value()) {
        throw std::runtime_error(
            "Requested text detector ONNX weights ('" + requested_model + "') were not found in " +
            util::path_to_utf8(model_manager_.models_dir()) +
            ". Place the standalone .onnx model in ./models or call GET /api/models/download first.");
    }

    ScopeUnload unload_guard{this, normalized_opts.auto_unload};

    return run_comic_text_detector(rgb_data, width, height, *resolved, normalized_opts);
}

void TextDetector::populate_bubble_and_safe_bboxes(
    std::vector<DetectedBubble>& bubbles,
    const uint8_t* rgb_data,
    int orig_w,
    int orig_h,
    const std::vector<LayoutRegion>* speech_bubbles,
    const std::vector<LayoutRegion>* panels,
    const std::vector<uint8_t>* text_mask) {
    if (bubbles.empty() || orig_w <= 0 || orig_h <= 0) {
        return;
    }

    auto pixel_luma = [&](int x, int y) -> uint8_t {
        if (!rgb_data) return 255;
        const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(orig_w) + static_cast<size_t>(x)) * 3ULL;
        const float yv = 0.299f * static_cast<float>(rgb_data[idx + 0]) +
                         0.587f * static_cast<float>(rgb_data[idx + 1]) +
                         0.114f * static_cast<float>(rgb_data[idx + 2]);
        return static_cast<uint8_t>(clampi(static_cast<int>(std::lround(yv)), 0, 255));
    };

    auto is_dark_border = [&](int x, int y) -> bool {
        if (x < 0 || x >= orig_w || y < 0 || y >= orig_h) return true;
        if (text_mask && text_mask->size() >= static_cast<size_t>(orig_w) * static_cast<size_t>(orig_h)) {
            if ((*text_mask)[static_cast<size_t>(y) * static_cast<size_t>(orig_w) + static_cast<size_t>(x)] > 0) {
                return false;  // Ignore text strokes when tracing outward to bubble/frame ink
            }
        }
        return pixel_luma(x, y) < 90;
    };

    auto inter_area = [](const std::array<int, 4>& a, const std::array<int, 4>& b) -> double {
        const int ix1 = std::max(a[0], b[0]);
        const int iy1 = std::max(a[1], b[1]);
        const int ix2 = std::min(a[2], b[2]);
        const int iy2 = std::min(a[3], b[3]);
        if (ix2 <= ix1 || iy2 <= iy1) return 0.0;
        return static_cast<double>(ix2 - ix1) * static_cast<double>(iy2 - iy1);
    };

    for (auto& b : bubbles) {
        const auto& tb = b.bbox;
        const int tw = std::max(1, tb[2] - tb[0]);
        const int th = std::max(1, tb[3] - tb[1]);
        const double text_area = static_cast<double>(tw) * static_cast<double>(th);
        const int cx = (tb[0] + tb[2]) / 2;
        const int cy = (tb[1] + tb[3]) / 2;

        // 1. Match enclosing RF-DETR speech bubble where >50% of text_bbox lies inside bubble_bbox
        bool matched_bubble = false;
        std::array<int, 4> resolved_bubble = tb;
        if (speech_bubbles && !speech_bubbles->empty()) {
            double best_overlap = 0.50;
            double best_area = std::numeric_limits<double>::max();
            for (const auto& lb : *speech_bubbles) {
                const double ov = inter_area(tb, lb.bbox) / text_area;
                if (ov <= 0.50) continue;
                const double lb_area = static_cast<double>(std::max(1, lb.bbox[2] - lb.bbox[0])) *
                                       static_cast<double>(std::max(1, lb.bbox[3] - lb.bbox[1]));
                if (ov > best_overlap + 0.05 || (std::abs(ov - best_overlap) <= 0.05 && lb_area < best_area)) {
                    best_overlap = ov;
                    best_area = lb_area;
                    resolved_bubble = lb.bbox;
                    matched_bubble = true;
                }
            }
        }

        // Helper: trace outward from text_bbox center along 3 horizontal scanlines (25%, 50%, 75% of text_bbox.height)
        // until hitting dark ink border pixels (luma < 90) or search boundaries, returning the conservative inner median.
        struct HorizontalTrace {
            int left = 0;
            int right = 0;
            int left_hits = 0;
            int right_hits = 0;
        };
        auto trace_horizontal_walls = [&](int search_x1, int search_x2) -> HorizontalTrace {
            const std::array<int, 3> y_scans = {
                clampi(tb[1] + th / 4, 0, orig_h - 1),
                clampi(tb[1] + th / 2, 0, orig_h - 1),
                clampi(tb[1] + (th * 3) / 4, 0, orig_h - 1),
            };
            const int start_left = text_mask ? cx : (tb[0] - 1);
            const int start_right = text_mask ? cx : tb[2];

            std::array<int, 3> left_walls{};
            std::array<int, 3> right_walls{};
            int left_hits = 0;
            int right_hits = 0;

            for (size_t si = 0; si < 3; ++si) {
                const int sy = y_scans[si];

                int lx = search_x1;
                for (int x = clampi(start_left, search_x1, orig_w - 1); x >= search_x1; --x) {
                    if (is_dark_border(x, sy)) {
                        lx = x + 1;
                        ++left_hits;
                        break;
                    }
                }
                left_walls[si] = lx;

                int rx = search_x2;
                for (int x = clampi(start_right, 0, std::max(0, search_x2 - 1)); x < search_x2; ++x) {
                    if (is_dark_border(x, sy)) {
                        rx = x;
                        ++right_hits;
                        break;
                    }
                }
                right_walls[si] = rx;
            }

            std::sort(left_walls.begin(), left_walls.end());
            std::sort(right_walls.begin(), right_walls.end());
            return HorizontalTrace{left_walls[1], right_walls[1], left_hits, right_hits};
        };

        // 2. White-interior bubble wall tracing:
        //    - When RF-DETR matched a bubble, tighten oversized horizontal bounds (e.g. long tails) to actual ink walls.
        //    - Otherwise (CTD / narration boxes / unbubbled text), trace outward within panel bounds and safe maxWidth.
        if (matched_bubble && rgb_data != nullptr) {
            const int rb_w = std::max(1, resolved_bubble[2] - resolved_bubble[0]);
            const auto traced = trace_horizontal_walls(resolved_bubble[0], resolved_bubble[2]);
            if (traced.left_hits >= 2 && traced.right_hits >= 2) {
                const int tx1 = std::max(resolved_bubble[0], traced.left);
                const int tx2 = std::min(resolved_bubble[2], traced.right);
                const int min_valid_w = std::max(tw, static_cast<int>(std::lround(static_cast<double>(rb_w) * 0.45)));
                if ((tx2 - tx1) >= min_valid_w) {
                    resolved_bubble[0] = tx1;
                    resolved_bubble[2] = tx2;
                }
            }
        } else if (!matched_bubble && rgb_data != nullptr) {
            int lim_x1 = 0;
            int lim_y1 = 0;
            int lim_x2 = orig_w;
            int lim_y2 = orig_h;

            if (panels && !panels->empty()) {
                double best_ov = 0.50;
                double best_area = std::numeric_limits<double>::max();
                for (const auto& p : *panels) {
                    const double ov = inter_area(tb, p.bbox) / text_area;
                    if (ov <= 0.50) continue;
                    const double pa = static_cast<double>(std::max(1, p.bbox[2] - p.bbox[0])) *
                                      static_cast<double>(std::max(1, p.bbox[3] - p.bbox[1]));
                    if (ov > best_ov + 0.05 || (std::abs(ov - best_ov) <= 0.05 && pa < best_area)) {
                        best_ov = ov;
                        best_area = pa;
                        lim_x1 = p.bbox[0];
                        lim_y1 = p.bbox[1];
                        lim_x2 = p.bbox[2];
                        lim_y2 = p.bbox[3];
                    }
                }
            }

            // Clamp safely to maxWidth = min(text_bbox.height * 0.68, img_width * 0.24) (never narrower than text_bbox)
            const double max_w_cap = std::min(
                static_cast<double>(th) * 0.68,
                static_cast<double>(orig_w) * 0.24);
            const int max_w = std::max(tw, static_cast<int>(std::lround(max_w_cap)));
            const int max_h = std::max(
                th,
                static_cast<int>(std::lround(std::min(
                    static_cast<double>(th) * 1.45,
                    static_cast<double>(orig_h) * 0.28))));

            const int search_x1 = std::max(lim_x1, cx - max_w / 2);
            const int search_x2 = std::min(lim_x2, cx + (max_w + 1) / 2);
            const int search_y1 = std::max(lim_y1, cy - max_h / 2);
            const int search_y2 = std::min(lim_y2, cy + (max_h + 1) / 2);

            const auto h_trace = trace_horizontal_walls(search_x1, search_x2);
            int bx1 = h_trace.left;
            int bx2 = h_trace.right;
            if ((bx2 - bx1) < tw) {
                bx1 = tb[0];
                bx2 = tb[2];
            }

            // Trace vertical walls along 3 vertical scanlines (25%, 50%, 75% of text_bbox.width)
            const std::array<int, 3> x_scans = {
                clampi(tb[0] + tw / 4, 0, orig_w - 1),
                clampi(tb[0] + tw / 2, 0, orig_w - 1),
                clampi(tb[0] + (tw * 3) / 4, 0, orig_w - 1),
            };
            const int start_top = text_mask ? cy : (tb[1] - 1);
            const int start_bottom = text_mask ? cy : tb[3];

            std::array<int, 3> top_walls{};
            std::array<int, 3> bottom_walls{};
            for (size_t si = 0; si < 3; ++si) {
                const int sx = x_scans[si];

                int ty = search_y1;
                for (int y = clampi(start_top, search_y1, orig_h - 1); y >= search_y1; --y) {
                    if (is_dark_border(sx, y)) {
                        ty = y + 1;
                        break;
                    }
                }
                top_walls[si] = ty;

                int by = search_y2;
                for (int y = clampi(start_bottom, 0, std::max(0, search_y2 - 1)); y < search_y2; ++y) {
                    if (is_dark_border(sx, y)) {
                        by = y;
                        break;
                    }
                }
                bottom_walls[si] = by;
            }
            std::sort(top_walls.begin(), top_walls.end());
            std::sort(bottom_walls.begin(), bottom_walls.end());
            int by1 = top_walls[1];
            int by2 = bottom_walls[1];
            if ((by2 - by1) < th) {
                by1 = tb[1];
                by2 = tb[3];
            }

            if ((bx2 - bx1) > max_w) {
                const int mid_x = (bx1 + bx2) / 2;
                bx1 = clampi(mid_x - max_w / 2, lim_x1, lim_x2);
                bx2 = clampi(bx1 + max_w, lim_x1, lim_x2);
            }
            if ((by2 - by1) > max_h) {
                const int mid_y = (by1 + by2) / 2;
                by1 = clampi(mid_y - max_h / 2, lim_y1, lim_y2);
                by2 = clampi(by1 + max_h, lim_y1, lim_y2);
            }

            resolved_bubble = {
                clampi(bx1, 0, orig_w),
                clampi(by1, 0, orig_h),
                clampi(std::max(bx1 + 1, bx2), 0, orig_w),
                clampi(std::max(by1 + 1, by2), 0, orig_h),
            };
        }

        b.bubble_bbox = resolved_bubble;

        // 3. Balanced 8% inner margin inside interior bubble box so horizontal lines never touch curved oval borders
        const int bw = std::max(1, b.bubble_bbox[2] - b.bubble_bbox[0]);
        const int bh = std::max(1, b.bubble_bbox[3] - b.bubble_bbox[1]);
        const int pad_x = static_cast<int>(std::lround(static_cast<double>(bw) * 0.08));
        const int pad_y = static_cast<int>(std::lround(static_cast<double>(bh) * 0.08));

        int sx1 = b.bubble_bbox[0] + pad_x;
        int sx2 = b.bubble_bbox[2] - pad_x;
        if ((sx2 - sx1) < 16) {
            const int min_w = std::min(bw, 16);
            const int mid_x = (b.bubble_bbox[0] + b.bubble_bbox[2]) / 2;
            sx1 = clampi(mid_x - min_w / 2, b.bubble_bbox[0], b.bubble_bbox[2]);
            sx2 = clampi(sx1 + min_w, sx1 + 1, b.bubble_bbox[2]);
        }

        int sy1 = b.bubble_bbox[1] + pad_y;
        int sy2 = b.bubble_bbox[3] - pad_y;
        if ((sy2 - sy1) < 16) {
            const int min_h = std::min(bh, 16);
            const int mid_y = (b.bubble_bbox[1] + b.bubble_bbox[3]) / 2;
            sy1 = clampi(mid_y - min_h / 2, b.bubble_bbox[1], b.bubble_bbox[3]);
            sy2 = clampi(sy1 + min_h, sy1 + 1, b.bubble_bbox[3]);
        }

        b.safe_bbox = {
            clampi(sx1, 0, orig_w),
            clampi(sy1, 0, orig_h),
            clampi(std::max(sx1 + 1, sx2), 0, orig_w),
            clampi(std::max(sy1 + 1, sy2), 0, orig_h),
        };
    }
}

// -----------------------------------------------------------------------------
// Backend 2: Comic Text Detector Fallback (`comictextdetector.pt.onnx`)
// -----------------------------------------------------------------------------

DetectionResult TextDetector::run_comic_text_detector(
    const uint8_t* rgb_data,
    int orig_w,
    int orig_h,
    const ResolvedDetectorModel& resolved,
    const DetectionOptions& options) {
    constexpr int kInputSize = 1024;
    const float conf_t = options.text_threshold.has_value()
        ? clampf(*options.text_threshold, 0.05f, 0.95f)
        : (options.conf_threshold.has_value()
               ? clampf(*options.conf_threshold, 0.05f, 0.95f)
               : clampf(options.ctd_default_conf_threshold, 0.05f, 0.95f));
    const float nms_t = options.nms_threshold.value_or(0.40f);
    const float mask_t = clampf(options.mask_threshold.value_or(0.35f), 0.01f, 0.99f);
    const bool is_rtl = resolve_is_rtl(options.source_language, options.reading_order);

    const float scale = std::min(
        static_cast<float>(kInputSize) / static_cast<float>(orig_h),
        static_cast<float>(kInputSize) / static_cast<float>(orig_w));
    const int new_w = std::max(1, static_cast<int>(std::lround(static_cast<float>(orig_w) * scale)));
    const int new_h = std::max(1, static_cast<int>(std::lround(static_cast<float>(orig_h) * scale)));

    const std::vector<uint8_t> resized = resize_rgb_bilinear(rgb_data, orig_w, orig_h, new_w, new_h);

    const size_t plane = static_cast<size_t>(kInputSize) * static_cast<size_t>(kInputSize);
    std::vector<float> input_tensor(3ULL * plane, 0.0f);
    static constexpr float kInv255 = 1.0f / 255.0f;

    float* ch0 = input_tensor.data();
    float* ch1 = input_tensor.data() + plane;
    float* ch2 = input_tensor.data() + 2ULL * plane;

    for (int y = 0; y < new_h; ++y) {
        const uint8_t* src_row = resized.data() + static_cast<size_t>(y) * static_cast<size_t>(new_w) * 3ULL;
        const size_t dst_row = static_cast<size_t>(y) * static_cast<size_t>(kInputSize);
        for (int x = 0; x < new_w; ++x) {
            const uint8_t* px = src_row + static_cast<size_t>(x) * 3ULL;
            const size_t idx = dst_row + static_cast<size_t>(x);
            ch0[idx] = static_cast<float>(px[0]) * kInv255;
            ch1[idx] = static_cast<float>(px[1]) * kInv255;
            ch2[idx] = static_cast<float>(px[2]) * kInv255;
        }
    }

    const std::array<int64_t, 4> input_shape = {1, 3, kInputSize, kInputSize};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value ort_input = Ort::Value::CreateTensor<float>(
        mem_info, input_tensor.data(), input_tensor.size(), input_shape.data(), input_shape.size());

    auto run_inference = [&](OrtSessionHandle& handle) -> std::vector<Ort::Value> {
        Ort::AllocatorWithDefaultOptions alloc;
        auto in_name = handle.session->GetInputNameAllocated(0, alloc);
        auto out0 = handle.session->GetOutputNameAllocated(0, alloc);
        auto out1 = handle.session->GetOutputNameAllocated(1, alloc);
        auto out2 = handle.session->GetOutputNameAllocated(2, alloc);
        const char* input_names[] = {in_name.get()};
        const char* output_names[] = {out0.get(), out1.get(), out2.get()};
        return handle.session->Run(
            Ort::RunOptions{nullptr},
            input_names,
            &ort_input,
            1,
            output_names,
            3);
    };

    std::vector<Ort::Value> outputs;
    std::string provider_used = "CPUExecutionProvider";
    {
        OrtSessionHandle& handle = ensure_session(resolved, options.force_cpu);
        try {
            outputs = run_inference(handle);
            provider_used = handle.active_provider;
        } catch (const Ort::Exception& ex) {
            std::fprintf(stderr, "[Vellum][RF-DETR] %s inference failed: %s\n", handle.active_provider.c_str(), ex.what());
            if (handle.active_provider == "CPUExecutionProvider") {
                throw;
            }
            OrtSessionHandle& cpu_handle = ensure_session(resolved, /*force_cpu=*/true);
            outputs = run_inference(cpu_handle);
            provider_used = cpu_handle.active_provider;
        }
    }

    // Decode `blk` ([1, 64512, 7])
    const auto blk_info = outputs[0].GetTensorTypeAndShapeInfo();
    const size_t total_blk_elems = blk_info.GetElementCount();
    const size_t num_rows = total_blk_elems / 7ULL;
    const float* blk_ptr = outputs[0].GetTensorData<float>();

    std::vector<std::array<float, 4>> boxes_xyxy;
    std::vector<float> scores;
    std::vector<int> cls_ids;

    for (size_t r = 0; r < num_rows; ++r) {
        const float* row = blk_ptr + r * 7ULL;
        const float obj_conf = row[4];
        if (obj_conf < conf_t) continue;
        const float cls0 = row[5];
        const float cls1 = row[6];
        const int cls_id = (cls0 >= cls1) ? 0 : 1;
        const float score = obj_conf * std::max(cls0, cls1);
        if (score < conf_t) continue;

        const float cx = row[0] / scale;
        const float cy = row[1] / scale;
        const float bw = row[2] / scale;
        const float bh = row[3] / scale;

        boxes_xyxy.push_back({cx - bw * 0.5f, cy - bh * 0.5f, cx + bw * 0.5f, cy + bh * 0.5f});
        scores.push_back(score);
        cls_ids.push_back(cls_id);
    }

    const auto keep_indices = nms_xyxy(boxes_xyxy, scores, nms_t);
    std::vector<RawBox> raw_boxes;
    raw_boxes.reserve(keep_indices.size());

    for (size_t k : keep_indices) {
        const auto& b = boxes_xyxy[k];
        const int bx1 = clampi(static_cast<int>(std::lround(b[0])) - options.box_padding, 0, orig_w);
        const int by1 = clampi(static_cast<int>(std::lround(b[1])) - options.box_padding, 0, orig_h);
        const int bx2 = clampi(static_cast<int>(std::lround(b[2])) + options.box_padding, 0, orig_w);
        const int by2 = clampi(static_cast<int>(std::lround(b[3])) + options.box_padding, 0, orig_h);
        if ((bx2 - bx1) < 8 || (by2 - by1) < 8) continue;

        const bool is_vertical =
            (cls_ids[k] == 0) && ((by2 - by1) > static_cast<int>(std::lround(static_cast<double>(bx2 - bx1) * 1.2)));
        RawBox rb;
        rb.bbox = {bx1, by1, bx2, by2};
        rb.score = scores[k];
        rb.direction = is_vertical ? "vertical" : "horizontal";
        rb.category = "text";
        raw_boxes.push_back(std::move(rb));
    }

    auto bubbles = sort_reading_order(
        raw_boxes,
        options.source_language,
        orig_h,
        nullptr,
        options.reading_order,
        nullptr);

    // Fuse `seg` ([1, 1, 1024, 1024]) and `det` ([1, 2, 1024, 1024] channel 0) into binary `(orig_h, orig_w)` mask
    const float* seg_ptr = outputs[1].GetTensorData<float>();
    const float* det_ptr = outputs[2].GetTensorData<float>();

    std::vector<float> fused_crop(static_cast<size_t>(new_w) * static_cast<size_t>(new_h), 0.0f);
    for (int y = 0; y < new_h; ++y) {
        const size_t src_row = static_cast<size_t>(y) * static_cast<size_t>(kInputSize);
        const size_t dst_row = static_cast<size_t>(y) * static_cast<size_t>(new_w);
        for (int x = 0; x < new_w; ++x) {
            const size_t sidx = src_row + static_cast<size_t>(x);
            fused_crop[dst_row + static_cast<size_t>(x)] = std::max(seg_ptr[sidx], det_ptr[sidx] * 0.85f);
        }
    }

    std::vector<uint8_t> binary_mask(static_cast<size_t>(orig_w) * static_cast<size_t>(orig_h), 0);
    const float rx = static_cast<float>(new_w) / static_cast<float>(orig_w);
    const float ry = static_cast<float>(new_h) / static_cast<float>(orig_h);
    for (int y = 0; y < orig_h; ++y) {
        const float sy = (static_cast<float>(y) + 0.5f) * ry - 0.5f;
        uint8_t* dst_row = binary_mask.data() + static_cast<size_t>(y) * static_cast<size_t>(orig_w);
        for (int x = 0; x < orig_w; ++x) {
            const float sx = (static_cast<float>(x) + 0.5f) * rx - 0.5f;
            if (sample_bilinear_f32(fused_crop.data(), new_w, new_h, sx, sy) >= mask_t) {
                dst_row[x] = 255;
            }
        }
    }

    if (options.dilate_kernel_size > 1) {
        dilate_binary_mask_ellipse(binary_mask, orig_w, orig_h, options.dilate_kernel_size);
    }

    if (options.restrict_mask_to_bboxes && !bubbles.empty()) {
        std::vector<uint8_t> roi_mask(binary_mask.size(), 0);
        const int pad = std::max(8, options.box_padding);
        for (const auto& b : bubbles) {
            const int rx1 = clampi(b.bbox[0] - pad, 0, orig_w);
            const int ry1 = clampi(b.bbox[1] - pad, 0, orig_h);
            const int rx2 = clampi(b.bbox[2] + pad, 0, orig_w);
            const int ry2 = clampi(b.bbox[3] + pad, 0, orig_h);
            for (int y = ry1; y < ry2; ++y) {
                std::memset(
                    roi_mask.data() + static_cast<size_t>(y) * static_cast<size_t>(orig_w) + static_cast<size_t>(rx1),
                    255,
                    static_cast<size_t>(rx2 - rx1));
            }
        }
        for (size_t i = 0; i < binary_mask.size(); ++i) {
            binary_mask[i] &= roi_mask[i];
        }
    }

    populate_bubble_and_safe_bboxes(
        bubbles, rgb_data, orig_w, orig_h, nullptr, nullptr, &binary_mask);

    // Classic CV panel fallback + panel-first reading order + room_polygon + typography for CTD
    if (!bubbles.empty()) {
        const auto fallback_panels = vellum::detector::detectPanelsClassicCV(rgb_data, orig_w, orig_h);
        std::vector<vellum::detector::RectI> text_rects;
        text_rects.reserve(bubbles.size());
        for (const auto& b : bubbles) {
            text_rects.push_back(vellum::detector::RectI{b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3]});
        }
        const auto perm = vellum::detector::sortReadingOrderPanelFirst(text_rects, fallback_panels, is_rtl);
        if (perm.size() == bubbles.size()) {
            std::vector<DetectedBubble> reordered;
            reordered.reserve(bubbles.size());
            int ord = 1;
            for (size_t idx : perm) {
                DetectedBubble b = std::move(bubbles[idx]);
                char id_buf[32]{};
                std::snprintf(id_buf, sizeof(id_buf), "bubble_%03d", ord);
                b.id = id_buf;
                b.order = ord++;
                reordered.push_back(std::move(b));
            }
            bubbles = std::move(reordered);
        }

        std::vector<vellum::detector::RoomSeedBlock> seeds;
        std::vector<vellum::detector::RectI> bubble_rects;
        seeds.reserve(bubbles.size());
        bubble_rects.reserve(bubbles.size());
        for (size_t i = 0; i < bubbles.size(); ++i) {
            vellum::detector::RoomSeedBlock s;
            s.block_index = static_cast<int>(i);
            s.bbox = vellum::detector::RectI{
                bubbles[i].bbox[0], bubbles[i].bbox[1], bubbles[i].bbox[2], bubbles[i].bbox[3]
            };
            seeds.push_back(std::move(s));
            bubble_rects.push_back(vellum::detector::RectI{
                bubbles[i].bubble_bbox[0], bubbles[i].bubble_bbox[1],
                bubbles[i].bubble_bbox[2], bubbles[i].bubble_bbox[3]
            });
        }
        const auto room_polys = vellum::detector::computePageRoomPolygons(
            rgb_data, orig_w, orig_h, seeds, bubble_rects, &binary_mask, options.enable_waist_cut);
        for (size_t i = 0; i < bubbles.size() && i < room_polys.size(); ++i) {
            bubbles[i].room_polygon.reserve(room_polys[i].size());
            for (const auto& pt : room_polys[i]) {
                bubbles[i].room_polygon.push_back({pt.x, pt.y});
            }
        }

        if (options.enable_cv_style_detect) {
            const vellum::detector::RectI page_rect{0, 0, orig_w, orig_h};
            for (auto& b : bubbles) {
                const vellum::detector::RectI tb{b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3]};
                const bool is_vert = (b.text_direction == "vertical");
                const auto style = vellum::detector::analyzeBlockTypographyFromBox(
                    rgb_data, orig_w, orig_h, tb, is_vert, false, &page_rect, &binary_mask);
                if (style.valid) {
                    b.angle_deg = style.angle_deg;
                    b.estimated_font_size = style.estimated_font_size;
                    b.text_color = style.text_color;
                    if (style.outline.has_value()) {
                        b.outline = DetectedOutline{style.outline->color, style.outline->width};
                    } else {
                        b.outline = std::nullopt;
                    }
                }
            }
        }
    }

    DetectionResult result;
    result.width = orig_w;
    result.height = orig_h;
    result.execution_provider = provider_used;
    result.backend = DetectorBackend::ComicTextDetector;
    result.model_id = detector_model_id(DetectorBackend::ComicTextDetector);
    result.model_path = util::path_to_utf8(resolved.model_path);
    result.reading_order = is_rtl ? "rtl" : "ltr";
    result.applied_thresholds.text = conf_t;
    result.applied_thresholds.onomatopoeia = conf_t;
    result.applied_thresholds.bubble = std::nullopt;
    result.applied_thresholds.panel = std::nullopt;
    result.applied_thresholds.nms = nms_t;
    result.applied_thresholds.mask = mask_t;
    result.bubbles = std::move(bubbles);
    result.mask = std::move(binary_mask);
    return result;
}

}  // namespace mangaflow::pipeline
