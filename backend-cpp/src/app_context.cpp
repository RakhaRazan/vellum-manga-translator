#include "app_context.hpp"
#include "utils.hpp"

#include <windows.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mangaflow {

fs::path get_executable_dir() {
    std::wstring buffer(MAX_PATH, L'\0');
    while (true) {
        const DWORD len = GetModuleFileNameW(
            nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (len == 0) {
            return fs::current_path();
        }
        if (len < buffer.size()) {
            buffer.resize(static_cast<size_t>(len));
            break;
        }
        buffer.resize(buffer.size() * 2);
    }
    std::error_code ec;
    fs::path exe_path = fs::weakly_canonical(fs::path(buffer), ec);
    if (ec) {
        exe_path = fs::path(buffer);
    }
    return exe_path.parent_path();
}

std::optional<fs::path> resolve_frontend_dist(const fs::path& exe_dir) {
    auto is_valid_dist = [](const fs::path& candidate) -> bool {
        std::error_code ec;
        return fs::is_directory(candidate, ec) &&
               fs::is_regular_file(candidate / L"index.html", ec);
    };

    auto env_dist = util::get_env_wide(L"VELLUM_FRONTEND_DIST");
    if (!env_dist.has_value()) {
        env_dist = util::get_env_wide(L"MANGAFLOW_FRONTEND_DIST");
    }
    if (env_dist.has_value()) {
        fs::path candidate(*env_dist);
        if (is_valid_dist(candidate)) {
            std::error_code ec;
            auto canonical = fs::weakly_canonical(candidate, ec);
            return ec ? candidate : canonical;
        }
    }

    fs::path cursor = exe_dir;
    for (int depth = 0; depth < 6; ++depth) {
        for (const fs::path& candidate : {
                 cursor / L"frontend" / L"dist",
                 cursor / L"frontend_dist",
             }) {
            if (is_valid_dist(candidate)) {
                std::error_code ec;
                auto canonical = fs::weakly_canonical(candidate, ec);
                return ec ? candidate : canonical;
            }
        }
        fs::path parent = cursor.parent_path();
        if (parent.empty() || parent == cursor) {
            break;
        }
        cursor = parent;
    }

    std::error_code cwd_ec;
    fs::path cwd_candidate = fs::current_path(cwd_ec) / L"frontend" / L"dist";
    if (!cwd_ec && is_valid_dist(cwd_candidate)) {
        std::error_code ec;
        auto canonical = fs::weakly_canonical(cwd_candidate, ec);
        return ec ? cwd_candidate : canonical;
    }

    return std::nullopt;
}

// -----------------------------------------------------------------------------
// Binary Mask & Cleaned Image Cache Helpers (`/api/cache/images/{id}.png`)
// -----------------------------------------------------------------------------

std::pair<std::string, std::string> save_mask_png_to_cache(
    const std::vector<uint8_t>& png_bytes) {
    const fs::path cache_dir = get_image_cache_dir();
    const std::string mask_id = "mask_" + util::generate_uuid_hex();
    const fs::path final_path = cache_dir / util::utf8_to_wide(mask_id + ".png");

    std::string write_err;
    if (!util::atomic_write_bytes(final_path, png_bytes.data(), png_bytes.size(), &write_err)) {
        throw std::runtime_error("Failed to save binary mask PNG to image cache: " + write_err);
    }

    prune_image_cache(cache_dir);
    return {mask_id, "/api/cache/images/" + mask_id + ".png"};
}

std::pair<std::string, std::string> save_inpainted_png_to_cache(
    const std::vector<uint8_t>& png_bytes) {
    const fs::path cache_dir = get_image_cache_dir();
    const std::string image_id = util::generate_uuid_hex();
    const fs::path final_path = cache_dir / util::utf8_to_wide(image_id + ".png");

    std::string write_err;
    if (!util::atomic_write_bytes(final_path, png_bytes.data(), png_bytes.size(), &write_err)) {
        throw std::runtime_error("Failed to save cleaned PNG to image cache: " + write_err);
    }

    prune_image_cache(cache_dir);
    return {image_id, "/api/cache/images/" + image_id + ".png"};
}

// -----------------------------------------------------------------------------
// Outbound LM Studio Health Probe (delegates to Multi-Provider ComicTranslator)
// -----------------------------------------------------------------------------

json probe_lm_studio(SettingsManager& settings_mgr, double timeout_seconds) {
    return pipeline::ComicTranslator::get_lmstudio_status(
               settings_mgr.load(),
               timeout_seconds)
        .to_json();
}

// -----------------------------------------------------------------------------
// Stage 4 / Retranslate / Full-Page Pipeline Helpers (SPEC.md Sections 5.1-5.4)
// -----------------------------------------------------------------------------

size_t count_utf8_codepoints(const std::string& s) {
    size_t count = 0;
    for (unsigned char c : s) {
        if ((c & 0xC0U) != 0x80U) {
            ++count;
        }
    }
    return count;
}

int estimate_font_size(const std::array<int, 4>& bbox, const std::string& text) {
    const int bw = std::max(24, bbox[2] - bbox[0]);
    const int bh = std::max(24, bbox[3] - bbox[1]);
    const double usable_area = (static_cast<double>(bw) * 0.82) * (static_cast<double>(bh) * 0.78);
    const size_t char_count = std::max<size_t>(1, count_utf8_codepoints(util::trim_ascii(text)));
    const int estimated = static_cast<int>(
        std::sqrt(usable_area / (static_cast<double>(char_count) * 0.85)));
    return std::max(14, std::min(28, estimated));
}

json bbox_to_polygon_json(const std::array<int, 4>& bbox) {
    return json::array({
        json::array({bbox[0], bbox[1]}),
        json::array({bbox[2], bbox[1]}),
        json::array({bbox[2], bbox[3]}),
        json::array({bbox[0], bbox[3]}),
    });
}

json build_default_style_json(
    int font_size,
    const std::string& color,
    const std::string& align) {
    return {
        {"font_family", "CC Wild Words"},
        {"font_size", font_size},
        {"color", color},
        {"stroke_color", "#ffffff"},
        {"stroke_width", 2},
        {"align", align},
        {"line_height", 1.2},
    };
}


std::string json_scalar_str(const json& v) {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_null()) return {};
    return v.dump();
}

std::vector<TranslateBlockItemInput> parse_translate_block_inputs(const json& raw_blocks) {
    if (!raw_blocks.is_array()) {
        throw std::invalid_argument("Expected a JSON array of dialogue blocks.");
    }
    std::vector<TranslateBlockItemInput> items;
    items.reserve(raw_blocks.size());

    for (size_t i = 0; i < raw_blocks.size(); ++i) {
        const auto& entry = raw_blocks[i];
        if (!entry.is_object()) {
            throw std::invalid_argument("Each block entry must be a JSON object.");
        }

        TranslateBlockItemInput item;
        if (entry.contains("id") && !entry["id"].is_null()) {
            item.id = util::trim_ascii(json_scalar_str(entry["id"]));
        }
        if (item.id.empty()) {
            char buf[32]{};
            std::snprintf(buf, sizeof(buf), "bubble_%03zu", i + 1);
            item.id = buf;
        }

        if (entry.contains("order") && entry["order"].is_number()) {
            item.order = static_cast<int>(std::llround(entry["order"].get<double>()));
        } else {
            item.order = static_cast<int>(i + 1);
        }

        if (entry.contains("bbox") && entry["bbox"].is_array() && entry["bbox"].size() >= 4) {
            for (size_t k = 0; k < 4; ++k) {
                if (entry["bbox"][k].is_number()) {
                    item.bbox[k] = static_cast<int>(std::llround(entry["bbox"][k].get<double>()));
                }
            }
            if (item.bbox[2] < item.bbox[0]) std::swap(item.bbox[0], item.bbox[2]);
            if (item.bbox[3] < item.bbox[1]) std::swap(item.bbox[1], item.bbox[3]);
        }

        if (entry.contains("polygon") && entry["polygon"].is_array() && !entry["polygon"].empty()) {
            item.polygon = entry["polygon"];
        } else {
            item.polygon = bbox_to_polygon_json(item.bbox);
        }

        if (entry.contains("original_text") && !entry["original_text"].is_null()) {
            item.original_text = json_scalar_str(entry["original_text"]);
        } else if (entry.contains("source_text") && !entry["source_text"].is_null()) {
            item.original_text = json_scalar_str(entry["source_text"]);
        } else if (entry.contains("text") && !entry["text"].is_null()) {
            item.original_text = json_scalar_str(entry["text"]);
        }

        if (entry.contains("font_size") && entry["font_size"].is_number()) {
            const int fs_val = static_cast<int>(std::llround(entry["font_size"].get<double>()));
            if (fs_val > 0) item.font_size = fs_val;
        }
        if (entry.contains("color") && entry["color"].is_string() &&
            !util::trim_ascii(entry["color"].get<std::string>()).empty()) {
            item.color = util::trim_ascii(entry["color"].get<std::string>());
        }
        if (entry.contains("align") && entry["align"].is_string()) {
            const std::string al = util::to_lower_ascii(util::trim_ascii(entry["align"].get<std::string>()));
            if (al == "left" || al == "center" || al == "right") {
                item.align = al;
            }
        }
        if (entry.contains("class_name") && entry["class_name"].is_string()) {
            item.bubble_class = util::trim_ascii(entry["class_name"].get<std::string>());
        } else if (entry.contains("bubble_class") && entry["bubble_class"].is_string()) {
            item.bubble_class = util::trim_ascii(entry["bubble_class"].get<std::string>());
        }
        if (entry.contains("panel_id") && entry["panel_id"].is_number()) {
            item.panel_id = static_cast<int>(std::llround(entry["panel_id"].get<double>()));
        }
        if (entry.contains("speaker_id") && entry["speaker_id"].is_number()) {
            item.speaker_id = static_cast<int>(std::llround(entry["speaker_id"].get<double>()));
        } else if (entry.contains("speaker_cluster_id") && entry["speaker_cluster_id"].is_number()) {
            item.speaker_id = static_cast<int>(std::llround(entry["speaker_cluster_id"].get<double>()));
        }
        if (entry.contains("confidence") && entry["confidence"].is_number()) {
            item.confidence = entry["confidence"].get<double>();
        }
        if (entry.contains("text_direction") && entry["text_direction"].is_string()) {
            const std::string td = util::to_lower_ascii(util::trim_ascii(entry["text_direction"].get<std::string>()));
            if (td == "vertical" || td == "horizontal") {
                item.text_direction = td;
            }
        }

        items.push_back(std::move(item));
    }

    return items;
}

json build_translated_blocks_json(
    const std::vector<TranslateBlockItemInput>& items,
    const std::map<std::string, std::string>& trans_map) {
    json out = json::array();
    for (const auto& item : items) {
        const auto it = trans_map.find(item.id);
        const std::string new_trans = (it != trans_map.end()) ? it->second : item.original_text;
        const std::string sizing_text = new_trans.empty() ? item.original_text : new_trans;
        const int fsize = item.font_size.has_value()
            ? *item.font_size
            : estimate_font_size(item.bbox, sizing_text);

        out.push_back({
            {"id", item.id},
            {"order", item.order},
            {"bbox", json::array({item.bbox[0], item.bbox[1], item.bbox[2], item.bbox[3]})},
            {"polygon", item.polygon},
            {"original_text", item.original_text},
            {"source_text", item.original_text},
            {"translated_text", new_trans},
            {"font_size", fsize},
            {"color", item.color},
            {"align", item.align},
            {"confidence", item.confidence},
            {"text_direction", item.text_direction},
            {"style", build_default_style_json(fsize, item.color, item.align)},
        });
    }
    return out;
}

std::string extract_page_id_from_filename(const std::string& filename) {
    std::string page_id = "page_001";
    if (!filename.empty()) {
        std::string fname = filename;
        std::replace(fname.begin(), fname.end(), '\\', '/');
        if (const auto slash = fname.rfind('/'); slash != std::string::npos) {
            fname = fname.substr(slash + 1);
        }
        const std::string stem = util::path_to_utf8(fs::path(util::utf8_to_wide(fname)).stem());
        if (!stem.empty()) {
            page_id = stem;
        }
    }
    return page_id;
}


DecodedOptionalImage decode_optional_page_image(
    const std::optional<std::string>& uploaded_bytes,
    const std::optional<std::string>& image_url,
    const std::optional<std::string>& image_base64) {
    DecodedOptionalImage out;
    if (uploaded_bytes.has_value() && !uploaded_bytes->empty()) {
        out.rgb = pipeline::LamaInpainter::decode_rgb_from_bytes(
            reinterpret_cast<const uint8_t*>(uploaded_bytes->data()),
            uploaded_bytes->size(),
            out.width,
            out.height);
        out.valid = true;
        return out;
    }

    if (image_url.has_value() && !util::trim_ascii(*image_url).empty()) {
        const fs::path cached_path = resolve_cached_image_path(*image_url);
        const auto bytes = util::read_file_bytes(cached_path);
        if (!bytes.has_value() || bytes->empty()) {
            throw std::runtime_error("Failed to read cached page image from disk.");
        }
        out.rgb = pipeline::LamaInpainter::decode_rgb_from_bytes(
            bytes->data(), bytes->size(), out.width, out.height);
        out.valid = true;
        return out;
    }

    if (image_base64.has_value() && !util::trim_ascii(*image_base64).empty()) {
        std::string b64 = util::trim_ascii(*image_base64);
        if (b64.size() >= 5 &&
            util::to_lower_ascii(b64.substr(0, 5)) == "data:") {
            const size_t comma = b64.find(',');
            if (comma != std::string::npos) {
                b64 = b64.substr(comma + 1);
            }
        }
        const auto decoded = util::base64_decode(b64);
        if (!decoded.has_value() || decoded->empty()) {
            throw std::invalid_argument("Failed to decode 'image_base64': invalid Base64 data.");
        }
        out.rgb = pipeline::LamaInpainter::decode_rgb_from_bytes(
            decoded->data(), decoded->size(), out.width, out.height);
        out.valid = true;
        return out;
    }

    return out;
}

// -----------------------------------------------------------------------------
// Static Frontend Asset Serving (`frontend/dist` at `/`)
// -----------------------------------------------------------------------------

std::string mime_type_for_extension(const std::string& raw_ext) {
    static const std::unordered_map<std::string, std::string> kMimeTypes = {
        {".html", "text/html; charset=utf-8"},
        {".htm", "text/html; charset=utf-8"},
        {".js", "application/javascript; charset=utf-8"},
        {".mjs", "application/javascript; charset=utf-8"},
        {".css", "text/css; charset=utf-8"},
        {".json", "application/json; charset=utf-8"},
        {".map", "application/json; charset=utf-8"},
        {".svg", "image/svg+xml"},
        {".png", "image/png"},
        {".jpg", "image/jpeg"},
        {".jpeg", "image/jpeg"},
        {".webp", "image/webp"},
        {".gif", "image/gif"},
        {".bmp", "image/bmp"},
        {".ico", "image/x-icon"},
        {".woff2", "font/woff2"},
        {".woff", "font/woff"},
        {".ttf", "font/ttf"},
        {".otf", "font/otf"},
        {".wasm", "application/wasm"},
        {".txt", "text/plain; charset=utf-8"},
    };
    const std::string ext = util::to_lower_ascii(raw_ext);
    const auto it = kMimeTypes.find(ext);
    return (it != kMimeTypes.end()) ? it->second : "application/octet-stream";
}

bool serve_file_from_disk(
    const fs::path& file_path,
    const std::string& request_path,
    httplib::Response& res) {
    const auto bytes = util::read_file_bytes(file_path);
    if (!bytes.has_value()) {
        return false;
    }

    const std::string ext = util::path_to_utf8(file_path.extension());
    const std::string mime = mime_type_for_extension(ext);
    if (util::to_lower_ascii(ext) == ".html" || util::to_lower_ascii(ext) == ".htm") {
        res.set_header("Cache-Control", "no-cache");
    } else if (request_path.rfind("/assets/", 0) == 0) {
        res.set_header("Cache-Control", "public, max-age=31536000, immutable");
    } else {
        res.set_header("Cache-Control", "public, max-age=3600");
    }
    res.set_content(
        reinterpret_cast<const char*>(bytes->data()),
        bytes->size(),
        mime);
    return true;
}

// -----------------------------------------------------------------------------
// Export Request Parser (supports both `multipart/form-data` & `application/json`)
// -----------------------------------------------------------------------------


bool parse_bool_param(const std::string& raw, bool default_val) {
    const std::string t = util::to_lower_ascii(util::trim_ascii(raw));
    if (t == "1" || t == "true" || t == "yes" || t == "on") return true;
    if (t == "0" || t == "false" || t == "no" || t == "off") return false;
    return default_val;
}

ParsedExportRequest parse_export_pages_request(const httplib::Request& req) {
    ParsedExportRequest parsed;

    if (req.is_multipart_form_data()) {
        std::vector<httplib::FormData> raw_uploads;
        for (const char* field : {"images", "files", "pages", "image", "file"}) {
            auto items = req.form.get_files(field);
            for (auto& item : items) {
                raw_uploads.push_back(std::move(item));
            }
        }
        if (raw_uploads.empty()) {
            throw ComicExportError(
                "No page images uploaded. Provide ordered multipart file fields named 'images' or 'files'.");
        }

        parsed.pages.reserve(raw_uploads.size());
        for (size_t i = 0; i < raw_uploads.size(); ++i) {
            const auto& u = raw_uploads[i];
            if (u.content.empty()) {
                throw ComicExportError(
                    "Uploaded page image #" + std::to_string(i + 1) +
                    " (" + (u.filename.empty() ? "unnamed" : u.filename) + ") is empty.");
            }
            std::vector<uint8_t> bytes(u.content.begin(), u.content.end());
            std::optional<std::string> fname =
                u.filename.empty() ? std::nullopt : std::optional<std::string>(u.filename);
            parsed.pages.push_back({std::move(bytes), std::move(fname)});
        }

        if (req.form.has_field("filename")) {
            const std::string v = util::trim_ascii(req.form.get_field("filename"));
            if (!v.empty()) parsed.export_name = v;
        } else if (req.form.has_field("title")) {
            const std::string v = util::trim_ascii(req.form.get_field("title"));
            if (!v.empty()) parsed.export_name = v;
        }

        if (req.form.has_field("image_format")) {
            parsed.image_format = parse_export_image_format(req.form.get_field("image_format"));
        }
        if (req.form.has_field("jpeg_quality")) {
            const std::string q_str = util::trim_ascii(req.form.get_field("jpeg_quality"));
            if (!q_str.empty()) {
                try {
                    parsed.jpeg_quality = std::max(50, std::min(100, std::stoi(q_str)));
                } catch (...) {
                    throw ComicExportError("Invalid integer for 'jpeg_quality': '" + q_str + "'");
                }
            }
        }
        return parsed;
    }

    if (req.body.empty()) {
        throw ComicExportError("Empty export request body.");
    }

    const json body = json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.is_object()) {
        throw ComicExportError("Invalid JSON export request body.");
    }

    const json* raw_items = nullptr;
    if (body.contains("pages") && body["pages"].is_array() && !body["pages"].empty()) {
        raw_items = &body["pages"];
    } else if (body.contains("images") && body["images"].is_array() && !body["images"].empty()) {
        raw_items = &body["images"];
    }
    if (!raw_items) {
        throw ComicExportError("Provide a non-empty ordered list in 'pages' or 'images'.");
    }

    parsed.pages.reserve(raw_items->size());
    for (size_t i = 0; i < raw_items->size(); ++i) {
        const int page_idx = static_cast<int>(i + 1);
        const auto& item = (*raw_items)[i];
        if (item.is_string()) {
            auto bytes = decode_export_image_reference(item.get<std::string>(), page_idx);
            parsed.pages.push_back({std::move(bytes), std::nullopt});
        } else if (item.is_object()) {
            std::string ref;
            for (const char* key : {"image", "url", "image_url", "image_id"}) {
                if (item.contains(key) && item[key].is_string() && !item[key].get<std::string>().empty()) {
                    ref = item[key].get<std::string>();
                    break;
                }
            }
            if (ref.empty()) {
                throw ComicExportError(
                    "Page " + std::to_string(page_idx) +
                    " object must include 'image', 'url', 'image_url', or 'image_id'.");
            }
            auto bytes = decode_export_image_reference(ref, page_idx);
            std::optional<std::string> fname;
            if (item.contains("filename") && item["filename"].is_string() && !item["filename"].get<std::string>().empty()) {
                fname = item["filename"].get<std::string>();
            }
            parsed.pages.push_back({std::move(bytes), std::move(fname)});
        } else {
            throw ComicExportError("Page " + std::to_string(page_idx) + " entry must be a string or object.");
        }
    }

    if (body.contains("filename") && body["filename"].is_string() && !util::trim_ascii(body["filename"].get<std::string>()).empty()) {
        parsed.export_name = util::trim_ascii(body["filename"].get<std::string>());
    } else if (body.contains("title") && body["title"].is_string() && !util::trim_ascii(body["title"].get<std::string>()).empty()) {
        parsed.export_name = util::trim_ascii(body["title"].get<std::string>());
    }
    if (body.contains("image_format") && body["image_format"].is_string()) {
        parsed.image_format = parse_export_image_format(body["image_format"].get<std::string>());
    }
    if (body.contains("jpeg_quality") && body["jpeg_quality"].is_number()) {
        parsed.jpeg_quality = std::max(50, std::min(100, body["jpeg_quality"].get<int>()));
    }

    return parsed;
}

// -----------------------------------------------------------------------------
// Application Context & HTTP Route Registration
// -----------------------------------------------------------------------------





BatchKeepAliveHint parse_batch_keep_alive(const httplib::Request& req) {
    BatchKeepAliveHint hint;
    for (const char* key : {"keep_alive", "keep_sessions_alive", "batch_mode"}) {
        if (req.form.has_field(key)) {
            const std::string raw = util::trim_ascii(req.form.get_field(key));
            if (!raw.empty()) {
                hint.keep_alive = parse_bool_param(raw, false);
                break;
            }
        }
    }
    if (req.form.has_field("auto_unload")) {
        const std::string raw = util::trim_ascii(req.form.get_field("auto_unload"));
        if (!raw.empty()) {
            hint.keep_alive = !parse_bool_param(raw, true);
        }
    }
    for (const char* last_key : {"batch_is_last", "unload_after"}) {
        if (req.form.has_field(last_key)) {
            const std::string raw = util::trim_ascii(req.form.get_field(last_key));
            if (!raw.empty() && parse_bool_param(raw, false)) {
                hint.unload_after = true;
                hint.keep_alive = false;
                break;
            }
        }
    }
    return hint;
}

std::optional<float> parse_optional_float_field(const httplib::Request& req, const char* name) {
    if (!req.form.has_field(name)) {
        return std::nullopt;
    }
    const std::string raw = util::trim_ascii(req.form.get_field(name));
    if (raw.empty()) {
        return std::nullopt;
    }
    try {
        return std::stof(raw);
    } catch (...) {
        throw std::invalid_argument("Invalid floating-point number for field '" + std::string(name) + "': '" + raw + "'");
    }
}

void apply_optional_nullable_threshold_field(
    const httplib::Request& req,
    const char* name,
    std::optional<float>& target) {
    if (!req.form.has_field(name)) {
        return;
    }
    const std::string raw = util::trim_ascii(req.form.get_field(name));
    const std::string lower = util::to_lower_ascii(raw);
    if (lower.empty() || lower == "null" || lower == "default" || lower == "auto" || lower == "none") {
        target = std::nullopt;
        return;
    }
    try {
        const float val = std::stof(raw);
        if (!std::isfinite(val)) {
            throw std::invalid_argument("Non-finite threshold for field '" + std::string(name) + "'.");
        }
        target = std::max(0.05f, std::min(0.95f, val));
    } catch (const std::invalid_argument&) {
        throw;
    } catch (...) {
        throw std::invalid_argument("Invalid threshold for field '" + std::string(name) + "': '" + raw + "'");
    }
}

void populate_detection_options_from_settings(
    pipeline::DetectionOptions& opts,
    const json& settings) {
    opts.detection_model = SettingsManager::normalize_detection_model(
        settings.value("detection_model", std::string("vellum-det-yolo11m-seg-1024-fp16")));
    opts.allow_model_fallback = true;

    if (settings.contains("text_threshold") && settings["text_threshold"].is_number()) {
        opts.text_threshold = settings["text_threshold"].get<float>();
    } else {
        opts.text_threshold = std::nullopt;
    }
    if (settings.contains("bubble_threshold") && settings["bubble_threshold"].is_number()) {
        opts.bubble_threshold = settings["bubble_threshold"].get<float>();
    } else {
        opts.bubble_threshold = std::nullopt;
    }
    if (settings.contains("panel_threshold") && settings["panel_threshold"].is_number()) {
        opts.panel_threshold = settings["panel_threshold"].get<float>();
    } else {
        opts.panel_threshold = std::nullopt;
    }

    opts.ctd_default_conf_threshold = static_cast<float>(settings.value("conf_threshold", 0.35));
    opts.nms_threshold = static_cast<float>(settings.value("nms_threshold", 0.45));
    opts.mask_threshold = static_cast<float>(settings.value("mask_threshold", 0.50));
    opts.box_padding = settings.value("bbox_padding", 6);
    opts.reading_order = settings.value("reading_order", std::string("auto"));
    opts.auto_unload = settings.value("auto_unload_vram", true);
    opts.enable_cv_style_detect = settings.value("enable_cv_style_detect", true);
    opts.enable_waist_cut = settings.value("enable_waist_cut", true);
    const std::string det_exec_mode = settings.value("execution_mode", std::string("hybrid"));
    opts.force_cpu = (det_exec_mode == "cpu_only" || det_exec_mode == "cpu");
}

void populate_ocr_options_from_settings(
    pipeline::OcrOptions& opts,
    const json& settings) {
    opts.ocr_model = SettingsManager::normalize_ocr_model(
        settings.value("ocr_model", std::string("auto")));
    opts.auto_unload = settings.value("auto_unload_vram", true);
    opts.auto_mask_furigana_ocr = settings.value("auto_mask_furigana_ocr", true);
    // Smart Hybrid Routing:
    // - "hybrid" (default): keep autoregressive manga-ocr & lightweight RapidOCR on CPU (saves ~650 MB VRAM, avoids DML stalls).
    // - "force_gpu": allow OCR encoder/models onto GPU.
    // - "cpu_only": force all models onto CPU.
    const std::string ocr_exec_mode = settings.value("execution_mode", std::string("hybrid"));
    opts.force_cpu = (ocr_exec_mode != "force_gpu" && ocr_exec_mode != "gpu");

    std::string lm_url = pipeline::kDefaultLmStudioBaseUrl;
    std::string lm_key = pipeline::kDefaultLmStudioApiKey;
    if (settings.value("provider", std::string("lmstudio")) == "lmstudio") {
        if (settings.contains("base_url") && settings["base_url"].is_string() &&
            !util::trim_ascii(settings["base_url"].get<std::string>()).empty()) {
            lm_url = util::trim_ascii(settings["base_url"].get<std::string>());
        }
        if (settings.contains("api_key") && settings["api_key"].is_string()) {
            lm_key = util::trim_ascii(settings["api_key"].get<std::string>());
        }
    } else if (settings.contains("providers") && settings["providers"].is_object() &&
               settings["providers"].contains("lmstudio") && settings["providers"]["lmstudio"].is_object()) {
        const auto& lm = settings["providers"]["lmstudio"];
        if (lm.contains("base_url") && lm["base_url"].is_string() &&
            !util::trim_ascii(lm["base_url"].get<std::string>()).empty()) {
            lm_url = util::trim_ascii(lm["base_url"].get<std::string>());
        }
        if (lm.contains("api_key") && lm["api_key"].is_string()) {
            lm_key = util::trim_ascii(lm["api_key"].get<std::string>());
        }
    }
    opts.lmstudio_base_url = std::move(lm_url);
    opts.lmstudio_api_key = std::move(lm_key);
}

void populate_inpaint_options_from_settings(
    pipeline::InpaintOptions& opts,
    const json& settings) {
    opts.inpainting_model = SettingsManager::normalize_inpainting_model(
        settings.value("inpainting_model", std::string("lama-manga")));
    opts.inpaint_api_url = settings.value("inpaint_api_url", std::string("http://127.0.0.1:8188"));
    opts.allow_model_fallback = true;
    opts.tile_size = settings.value("lama_tile_size", 512);
    opts.refine_mask = settings.value("refine_mask", true);
    opts.auto_unload = settings.value("auto_unload_vram", true);
    opts.enable_3tier_inpaint = settings.value("enable_3tier_inpaint", true);
    opts.enable_tone_match = settings.value("enable_tone_match", true);
    const std::string inp_exec_mode = settings.value("execution_mode", std::string("hybrid"));
    opts.force_cpu = (inp_exec_mode == "cpu_only" || inp_exec_mode == "cpu");
}

json extract_multipart_translation_overrides(const httplib::Request& req) {
    json cfg = json::object();

    if (req.form.has_field("config")) {
        const std::string raw_cfg = util::trim_ascii(req.form.get_field("config"));
        if (!raw_cfg.empty()) {
            const json parsed = json::parse(raw_cfg, nullptr, false);
            if (parsed.is_discarded() || !parsed.is_object()) {
                throw std::invalid_argument("'config' form field must be a valid JSON object.");
            }
            cfg = parsed;
        }
    }

    for (const char* str_key : {
             "provider", "model", "base_url", "api_key",
             "reasoning_effort", "translation_tone", "honorifics_policy", "custom_instructions"}) {
        if (req.form.has_field(str_key)) {
            const std::string val = util::trim_ascii(req.form.get_field(str_key));
            if (!val.empty()) {
                cfg[str_key] = val;
            }
        }
    }

    for (const char* dbl_key : {"temperature", "top_p", "frequency_penalty", "timeout"}) {
        if (req.form.has_field(dbl_key)) {
            const std::string raw = util::trim_ascii(req.form.get_field(dbl_key));
            if (!raw.empty()) {
                try {
                    cfg[dbl_key] = std::stod(raw);
                } catch (...) {
                    throw std::invalid_argument(
                        "Invalid float for '" + std::string(dbl_key) + "': '" + raw + "'");
                }
            }
        }
    }

    for (const char* int_key : {"top_k", "max_tokens", "vision_max_side"}) {
        if (req.form.has_field(int_key)) {
            const std::string raw = util::trim_ascii(req.form.get_field(int_key));
            if (!raw.empty()) {
                try {
                    cfg[int_key] = std::stoi(raw);
                } catch (...) {
                    throw std::invalid_argument(
                        "Invalid integer for '" + std::string(int_key) + "': '" + raw + "'");
                }
            }
        }
    }

    if (req.form.has_field("glossary")) {
        const std::string raw_g = util::trim_ascii(req.form.get_field("glossary"));
        if (!raw_g.empty()) {
            cfg["glossary"] = raw_g;
        }
    }

    if (req.form.has_field("use_vision")) {
        const std::string raw_v = util::trim_ascii(req.form.get_field("use_vision"));
        if (!raw_v.empty()) {
            cfg["use_vision"] = parse_bool_param(raw_v, false);
        }
    }

    if (req.form.has_field("strip_thinking")) {
        const std::string raw_st = util::trim_ascii(req.form.get_field("strip_thinking"));
        if (!raw_st.empty()) {
            cfg["strip_thinking"] = parse_bool_param(raw_st, true);
        }
    }

    return cfg;
}

void apply_cors_headers(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization, Accept");
}

void send_json(httplib::Response& res, const json& payload, int status) {
    apply_cors_headers(res);
    res.status = status;
    res.set_content(
        payload.dump(-1, ' ', false, json::error_handler_t::replace),
        "application/json; charset=utf-8");
}


}  // namespace mangaflow
