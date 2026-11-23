// Vellum - Native Modern C++17 Backend & WebView2 Desktop Host (Phase 10.3)
//
// Runs an embedded cpp-httplib HTTP server on a background std::thread
// (127.0.0.1:8000, or an OS-assigned free loopback port if busy), serves the
// compiled React v2 frontend bundle (`frontend/dist`) at `/` alongside `/api/*`
// JSON/binary/SSE endpoints (Health, Full 4-Model Koharu Cache & HTTPS Model
// Management, Native ONNX Runtime Text & Layout Detection, Multi-Language OCR,
// Tile-Based LaMa Inpainting, Multi-Provider LLM Translation via cpp-httplib +
// WinHTTP, Full End-to-End Page Processing, Persistent Settings, Local/System/
// Google Fonts, Image Cache, CBZ/ZIP Archive Import, and CBZ / Native PDF 1.4
// Export), and hosts the UI inside a native Microsoft Edge WebView2 desktop
// window.

#include <httplib.h>

#include <windows.h>
#include <dwmapi.h>

#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4005)  // webview.h unconditionally defines WIN32_LEAN_AND_MEAN
#endif
#include "webview/webview.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "detector/adaptive_tiler.hpp"
#include "detector/detector_engine.hpp"
#include "detector/export_artifacts.hpp"
#include "detector/geometry_postproc.hpp"
#include "exporter.hpp"
#include "font_manager.hpp"
#include "pipeline/detector.hpp"
#include "pipeline/inpainter.hpp"
#include "pipeline/model_manager.hpp"
#include "pipeline/ocr.hpp"
#include "pipeline/translator.hpp"
#include "settings_manager.hpp"
#include "utils.hpp"

#include <stb_image.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace mangaflow {

constexpr const char* kHost = "127.0.0.1";
constexpr int kDefaultPort = 8000;
constexpr const char* kWindowTitle = "Vellum";
constexpr int kWindowWidth = 1440;
constexpr int kWindowHeight = 900;
constexpr int kWindowMinWidth = 1024;
constexpr int kWindowMinHeight = 680;
constexpr const char* kAppVersion = "2.0.0";

// -----------------------------------------------------------------------------
// Executable-Relative Path Resolution
// -----------------------------------------------------------------------------

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

json probe_lm_studio(SettingsManager& settings_mgr, double timeout_seconds = 3.0) {
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
    const std::string& color = "#000000",
    const std::string& align = "center") {
    return {
        {"font_family", "Anime Ace"},
        {"font_size", font_size},
        {"color", color},
        {"stroke_color", "#ffffff"},
        {"stroke_width", 2},
        {"align", align},
        {"line_height", 1.2},
    };
}

struct TranslateBlockItemInput {
    std::string id;
    int order = 0;
    std::array<int, 4> bbox{0, 0, 160, 80};
    json polygon;
    std::string original_text;
    std::optional<int> font_size;
    std::string color = "#000000";
    std::string align = "center";
    double confidence = 1.0;
    std::string text_direction = "horizontal";
};

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

struct DecodedOptionalImage {
    std::vector<uint8_t> rgb;
    int width = 0;
    int height = 0;
    bool valid = false;
};

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

struct ParsedExportRequest {
    std::vector<ExportPageInput> pages;
    std::optional<std::string> export_name;
    ExportImageFormat image_format = ExportImageFormat::Original;
    int jpeg_quality = 95;
};

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

struct ExportSessionState {
    std::string id;
    std::string format = "cbz";  // "cbz" | "pdf"
    std::optional<std::string> export_name;
    ExportImageFormat image_format = ExportImageFormat::Original;
    int jpeg_quality = 95;
    fs::path session_dir;
    std::map<int, DiskExportPageEntry> pages;
};

struct CompletedExportState {
    fs::path file_path;
    fs::path session_dir;
    std::string download_name;
    std::string mime_type;
    uint64_t file_size = 0;
    size_t page_count = 0;
};

struct AppContext {
    fs::path exe_dir;
    fs::path frontend_dist_dir;
    std::string ort_version;
    std::vector<std::string> available_providers;
    std::unique_ptr<pipeline::ModelManager> model_manager;
    std::unique_ptr<pipeline::TextDetector> detector;
    std::unique_ptr<vellum::detector::IDetectorEngine> detector_v2;
    vellum::detector::BackendType detector_v2_backend = vellum::detector::BackendType::Auto;
    std::unique_ptr<pipeline::OcrEngine> ocr_engine;
    std::unique_ptr<pipeline::LamaInpainter> inpainter;
    SettingsManager settings_manager;
    FontManager font_manager;
    std::mutex inference_mutex;
    std::mutex import_mutex;
    std::mutex export_sessions_mutex;
    std::unordered_map<std::string, ExportSessionState> export_sessions;
    std::unordered_map<std::string, CompletedExportState> completed_exports;

    // Live per-stage progress reporting for UI polling (`GET /api/pipeline/status`)
    std::mutex progress_mutex;
    bool stage_active = false;
    std::string active_stage;
    std::string active_stage_detail;
    int active_stage_index = 0;
    int active_total_stages = 4;

    void set_live_stage(const std::string& stage, const std::string& detail, int idx, int total) {
        std::lock_guard<std::mutex> lock(progress_mutex);
        stage_active = true;
        active_stage = stage;
        active_stage_detail = detail;
        active_stage_index = idx;
        active_total_stages = total;
    }

    void clear_live_stage() {
        std::lock_guard<std::mutex> lock(progress_mutex);
        stage_active = false;
        active_stage.clear();
        active_stage_detail.clear();
        active_stage_index = 0;
    }

    // Batch Session Keep-Alive & Deferred Idle VRAM Unload Watchdog
    std::mutex unload_timer_mutex;
    std::condition_variable unload_timer_cv;
    uint64_t unload_generation = 0;
    bool idle_unload_pending = false;
    bool watchdog_stop = false;
    std::chrono::steady_clock::time_point idle_unload_deadline{};
    std::thread unload_watchdog_thread;

    void unload_all_sessions() {
        std::lock_guard<std::mutex> lock(inference_mutex);
        if (detector) detector->unload();
        if (detector_v2) detector_v2->unload();
        if (ocr_engine) ocr_engine->unload();
        if (inpainter) inpainter->unload();
        pipeline::ModelManager::trim_process_heap();
    }

    void cancel_pending_unload() {
        std::lock_guard<std::mutex> lock(unload_timer_mutex);
        ++unload_generation;
        idle_unload_pending = false;
    }

    void schedule_deferred_unload(int delay_seconds) {
        {
            std::lock_guard<std::mutex> lock(unload_timer_mutex);
            ++unload_generation;
            idle_unload_pending = true;
            idle_unload_deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(std::max(1, delay_seconds));
        }
        unload_timer_cv.notify_one();
    }

    void start_unload_watchdog() {
        unload_watchdog_thread = std::thread([this]() {
            std::unique_lock<std::mutex> lock(unload_timer_mutex);
            while (!watchdog_stop) {
                if (!idle_unload_pending) {
                    unload_timer_cv.wait(lock, [this]() {
                        return watchdog_stop || idle_unload_pending;
                    });
                    continue;
                }
                const uint64_t gen = unload_generation;
                const auto deadline = idle_unload_deadline;
                if (unload_timer_cv.wait_until(lock, deadline, [this, gen]() {
                        return watchdog_stop || !idle_unload_pending || unload_generation != gen;
                    })) {
                    continue;
                }
                idle_unload_pending = false;
                lock.unlock();
                unload_all_sessions();
                lock.lock();
            }
        });
    }

    void stop_unload_watchdog() {
        {
            std::lock_guard<std::mutex> lock(unload_timer_mutex);
            watchdog_stop = true;
            idle_unload_pending = false;
        }
        unload_timer_cv.notify_all();
        if (unload_watchdog_thread.joinable()) {
            unload_watchdog_thread.join();
        }
        unload_all_sessions();
    }
};

struct BatchKeepAliveHint {
    std::optional<bool> keep_alive;
    bool unload_after = false;
};

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
        settings.value("detection_model", std::string("koharu-layout-rfdetr-seg-2xl")));
    opts.allow_model_fallback = (opts.detection_model == "koharu-layout-rfdetr-seg-2xl");

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
    const std::string det_exec_mode = settings.value("execution_mode", std::string("hybrid"));
    opts.force_cpu = (det_exec_mode == "cpu_only" || det_exec_mode == "cpu");
}

void populate_ocr_options_from_settings(
    pipeline::OcrOptions& opts,
    const json& settings) {
    opts.ocr_model = SettingsManager::normalize_ocr_model(
        settings.value("ocr_model", std::string("auto")));
    opts.auto_unload = settings.value("auto_unload_vram", true);
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
             "reasoning_effort", "translation_tone", "custom_instructions"}) {
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

void send_json(httplib::Response& res, const json& payload, int status = 200) {
    apply_cors_headers(res);
    res.status = status;
    res.set_content(
        payload.dump(-1, ' ', false, json::error_handler_t::replace),
        "application/json; charset=utf-8");
}

void register_routes(httplib::Server& server, AppContext& ctx) {
    // CORS preflight for /api/*
    server.Options(R"(/api/.*)", [](const httplib::Request&, httplib::Response& res) {
        apply_cors_headers(res);
        res.status = 204;
    });

    // -------------------------------------------------------------------------
    // System, Health, Models, Koharu Cache, Providers & Languages
    // -------------------------------------------------------------------------

    server.Get("/api/ping", [](const httplib::Request&, httplib::Response& res) {
        apply_cors_headers(res);
        res.set_content(R"({"status":"ok"})", "application/json; charset=utf-8");
    });

    server.Get("/api/pipeline/status", [&ctx](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(ctx.progress_mutex);
        send_json(res, {
            {"active", ctx.stage_active},
            {"stage", ctx.active_stage},
            {"detail", ctx.active_stage_detail},
            {"stage_index", ctx.active_stage_index},
            {"total_stages", ctx.active_total_stages},
        });
    });

    server.Get("/api/health", [&ctx](const httplib::Request&, httplib::Response& res) {
        const json current_settings = ctx.settings_manager.load();
        const json models_status = ctx.model_manager->get_status_json();
        const auto lm_status = probe_lm_studio(ctx.settings_manager);
        const std::string sel_det = current_settings.value("detection_model", std::string("koharu-layout-rfdetr-seg-2xl"));
        const auto det = ctx.model_manager->resolve_detector_model(sel_det, /*allow_fallback=*/true);
        const auto hw = pipeline::ModelManager::query_hardware_profile();
        const json payload = {
            {"status", "ok"},
            {"backend", "cpp-native"},
            {"version", kAppVersion},
            {"onnxruntime_version", ctx.ort_version},
            {"execution_provider", pipeline::ModelManager::get_primary_provider()},
            {"available_providers", ctx.available_providers},
            {"hardware", {
                {"total_ram_mb", hw.total_ram_mb},
                {"available_ram_mb", hw.available_ram_mb},
                {"memory_load_percent", hw.memory_load_percent},
                {"dedicated_vram_mb", hw.dedicated_vram_mb},
                {"logical_cpus", hw.logical_cpus},
                {"gpu_name", hw.gpu_name},
                {"low_memory_mode", hw.low_memory_mode},
            }},
            {"models_ready", models_status.value("all_ready", false)},
            {"models_dir", models_status.value("models_dir", std::string(""))},
            {"detector_backend", det.has_value() ? json(pipeline::detector_backend_name(det->backend)) : json(nullptr)},
            {"detector_model", det.has_value() ? json(pipeline::detector_model_id(det->backend)) : json(nullptr)},
            {"pipeline_models", {
                {"detection_model", sel_det},
                {"ocr_model", current_settings.value("ocr_model", std::string("auto"))},
                {"inpainting_model", current_settings.value("inpainting_model", std::string("lama-manga"))},
            }},
            {"active_provider", ctx.settings_manager.get_active_provider_summary()},
            {"lm_studio", lm_status},
        };
        send_json(res, payload);
    });

    server.Get("/api/models/status", [&ctx](const httplib::Request&, httplib::Response& res) {
        const json current_settings = ctx.settings_manager.load();
        const std::string exec_mode = current_settings.value("execution_mode", std::string("hybrid"));
        const bool cpu_only = (exec_mode == "cpu_only" || exec_mode == "cpu");
        const bool ocr_cpu = (exec_mode != "force_gpu" && exec_mode != "gpu");

        std::string detector_provider = pipeline::ModelManager::get_primary_provider(cpu_only);
        if (ctx.detector_v2 && !ctx.detector_v2->executionProvider().empty()) {
            detector_provider = ctx.detector_v2->executionProvider();
        }
        const std::string ocr_provider = pipeline::ModelManager::get_primary_provider(ocr_cpu);
        const std::string inpainter_provider = pipeline::ModelManager::get_primary_provider(cpu_only);

        json payload = ctx.model_manager->get_status_json();
        payload["execution_mode"] = exec_mode;
        payload["detector_provider"] = detector_provider;
        payload["ocr_provider"] = ocr_provider;
        payload["inpainter_provider"] = inpainter_provider;
        payload["lm_studio"] = probe_lm_studio(ctx.settings_manager);
        send_json(res, payload);
    });

    server.Get("/api/models/koharu", [&ctx](const httplib::Request&, httplib::Response& res) {
        send_json(res, ctx.model_manager->get_koharu_status_json());
    });

    server.Get("/api/pipeline/models", [&ctx](const httplib::Request&, httplib::Response& res) {
        const json current_settings = ctx.settings_manager.load();
        send_json(res, ctx.model_manager->get_pipeline_models_json(current_settings));
    });

    auto handle_unload_sessions = [&ctx](const httplib::Request&, httplib::Response& res) {
        ctx.cancel_pending_unload();
        ctx.unload_all_sessions();
        send_json(res, {
            {"status", "ok"},
            {"unloaded", true},
        });
    };
    server.Post("/api/pipeline/unload", handle_unload_sessions);
    server.Post("/api/models/unload", handle_unload_sessions);

    server.Post("/api/models/import-koharu", [&ctx](const httplib::Request& req, httplib::Response& res) {
        pipeline::KoharuImportOptions opts;
        opts.search_root_dir = ctx.exe_dir;

        if (!util::trim_ascii(req.body).empty()) {
            const json body = json::parse(req.body, nullptr, false);
            if (body.is_discarded() || !body.is_object()) {
                send_json(res, {{"detail", "Invalid JSON object in /api/models/import-koharu request body."}}, 400);
                return;
            }
            if (body.contains("koharu_cache_dir") && body["koharu_cache_dir"].is_string()) {
                const std::string custom_cache = util::trim_ascii(body["koharu_cache_dir"].get<std::string>());
                if (!custom_cache.empty()) {
                    opts.koharu_cache_dir = fs::path(util::utf8_to_wide(custom_cache));
                }
            }
            if (body.contains("models")) {
                if (body["models"].is_array()) {
                    for (const auto& entry : body["models"]) {
                        if (entry.is_string()) {
                            opts.models.push_back(entry.get<std::string>());
                        }
                    }
                } else if (body["models"].is_string()) {
                    std::istringstream iss(body["models"].get<std::string>());
                    std::string tok;
                    while (std::getline(iss, tok, ',')) {
                        const std::string cleaned = util::trim_ascii(tok);
                        if (!cleaned.empty()) opts.models.push_back(cleaned);
                    }
                }
            }
            if (body.contains("force") && body["force"].is_boolean()) {
                opts.force = body["force"].get<bool>();
            }
            if (body.contains("convert") && body["convert"].is_boolean()) {
                opts.convert_safetensors = body["convert"].get<bool>();
            }
            if (body.contains("prefer_hardlink") && body["prefer_hardlink"].is_boolean()) {
                opts.prefer_hardlink = body["prefer_hardlink"].get<bool>();
            }
        }

        try {
            json import_res;
            {
                std::lock_guard<std::mutex> lock(ctx.import_mutex);
                import_res = ctx.model_manager->import_koharu_models(opts);
            }
            import_res["pipeline_models"] =
                ctx.model_manager->get_pipeline_models_json(ctx.settings_manager.load());

            const int status_code =
                (import_res.value("status", std::string("")) == "cache_not_found") ? 404 : 200;
            send_json(res, import_res, status_code);
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", std::string("Koharu model import failed: ") + ex.what()}}, 500);
        }
    });

    auto handle_models_download = [&ctx](const httplib::Request&, httplib::Response& res) {
        apply_cors_headers(res);
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_header("X-Accel-Buffering", "no");

        res.set_chunked_content_provider(
            "text/event-stream",
            [&ctx](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                try {
                    ctx.model_manager->download_missing_models([&sink](const json& evt) -> bool {
                        if (!sink.is_writable()) {
                            return false;
                        }
                        const std::string line = "data: " + evt.dump() + "\n\n";
                        return sink.write(line.data(), line.size());
                    });
                } catch (const std::exception& ex) {
                    if (sink.is_writable()) {
                        const json err_evt = {
                            {"status", "error"},
                            {"model_key", "all"},
                            {"file", nullptr},
                            {"file_percent", 0.0},
                            {"overall_percent", 0.0},
                            {"downloaded_bytes", 0},
                            {"total_bytes", 0},
                            {"offline", false},
                            {"message", std::string("Model download error: ") + ex.what()},
                        };
                        const std::string line = "data: " + err_evt.dump() + "\n\n";
                        sink.write(line.data(), line.size());
                    }
                }
                sink.done();
                return true;
            });
    };
    server.Get("/api/models/download", handle_models_download);
    server.Get("/api/models/download-stream", handle_models_download);

    // -------------------------------------------------------------------------
    // Stage 1: Text & Layout Detection (`POST /api/stage/detect`, `/api/pipeline/detect`)
    // -------------------------------------------------------------------------

    auto handle_stage_detect = [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!req.is_multipart_form_data()) {
            send_json(res, {{"detail", "Expected multipart/form-data upload with field 'image' or 'file'."}}, 400);
            return;
        }

        httplib::FormData upload;
        if (req.form.has_file("image")) {
            upload = req.form.get_file("image");
        } else if (req.form.has_file("file")) {
            upload = req.form.get_file("file");
        } else {
            send_json(res, {{"detail", "Missing uploaded image file. Provide multipart form field 'image' or 'file'."}}, 400);
            return;
        }

        if (upload.content.empty()) {
            send_json(res, {{"detail", "Uploaded image file is empty."}}, 400);
            return;
        }

        const json current_settings = ctx.settings_manager.load();
        pipeline::DetectionOptions opts;
        populate_detection_options_from_settings(opts, current_settings);
        bool include_mask_base64 = false;
        try {
            const std::string raw_lang = req.form.has_field("source_lang")
                ? req.form.get_field("source_lang")
                : (req.form.has_field("source_language")
                       ? req.form.get_field("source_language")
                       : current_settings.value("source_lang", std::string("ja")));
            opts.source_language = pipeline::TextDetector::normalize_source_language(raw_lang);

            if (auto cf = parse_optional_float_field(req, "conf_threshold"); cf.has_value()) {
                opts.conf_threshold = cf;
            }
            if (auto nf = parse_optional_float_field(req, "nms_threshold"); nf.has_value()) {
                opts.nms_threshold = nf;
            }
            if (auto mf = parse_optional_float_field(req, "mask_threshold"); mf.has_value()) {
                opts.mask_threshold = mf;
            }

            apply_optional_nullable_threshold_field(req, "text_threshold", opts.text_threshold);
            apply_optional_nullable_threshold_field(req, "bubble_threshold", opts.bubble_threshold);
            apply_optional_nullable_threshold_field(req, "panel_threshold", opts.panel_threshold);

            if (req.form.has_field("detection_model")) {
                const std::string dm = util::trim_ascii(req.form.get_field("detection_model"));
                if (!dm.empty()) {
                    opts.detection_model = SettingsManager::normalize_detection_model(dm);
                    opts.allow_model_fallback = false;
                }
            } else if (req.form.has_field("detector_backend")) {
                const std::string db = util::trim_ascii(req.form.get_field("detector_backend"));
                if (!db.empty()) {
                    opts.detector_backend = db;
                    if (util::to_lower_ascii(db) != "auto") {
                        opts.detection_model = SettingsManager::normalize_detection_model(db);
                        opts.allow_model_fallback = false;
                    }
                }
            }

            if (req.form.has_field("bbox_padding") || req.form.has_field("box_padding")) {
                const std::string pad_str = util::trim_ascii(
                    req.form.has_field("bbox_padding") ? req.form.get_field("bbox_padding") : req.form.get_field("box_padding"));
                if (!pad_str.empty()) {
                    try {
                        opts.box_padding = std::max(0, std::min(64, std::stoi(pad_str)));
                    } catch (...) {
                        throw std::invalid_argument("Invalid integer for 'bbox_padding': '" + pad_str + "'");
                    }
                }
            }
            if (req.form.has_field("reading_order")) {
                const std::string ro = util::to_lower_ascii(util::trim_ascii(req.form.get_field("reading_order")));
                if (ro == "auto" || ro == "rtl" || ro == "ltr") {
                    opts.reading_order = ro;
                }
            }
            if (req.form.has_field("include_mask_base64")) {
                include_mask_base64 = parse_bool_param(req.form.get_field("include_mask_base64"), false);
            }
        } catch (const std::invalid_argument& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
            return;
        }

        std::string export_dir;
        if (req.form.has_field("export_dir")) {
            export_dir = util::trim_ascii(req.form.get_field("export_dir"));
        }

        // Auto-resolve Detector Engine v2:
        //   1. ./models/vellum-det-yolo11m-seg-1024-fp16.onnx (or -int8.onnx on CPU-only)
        //   2. koharu-layout-rfdetr-seg-2xl-1152.onnx
        //   3. comictextdetector.pt.onnx
        const std::string det_selector = !opts.detection_model.empty() ? opts.detection_model : opts.detector_backend;
        const std::string sel_lower = util::to_lower_ascii(det_selector);

        auto find_yolo_model = [&](bool prefer_int8) -> std::optional<fs::path> {
            const std::array<const char*, 2> order = prefer_int8
                ? std::array<const char*, 2>{"vellum-det-yolo11m-seg-1024-int8.onnx", "vellum-det-yolo11m-seg-1024-fp16.onnx"}
                : std::array<const char*, 2>{"vellum-det-yolo11m-seg-1024-fp16.onnx", "vellum-det-yolo11m-seg-1024-int8.onnx"};
            std::vector<fs::path> roots{ctx.model_manager->models_dir()};
            if (ctx.model_manager->fallback_models_dir().has_value()) {
                roots.push_back(*ctx.model_manager->fallback_models_dir());
            }
            std::error_code ec;
            for (const char* fname : order) {
                for (const auto& root : roots) {
                    const fs::path candidate = root / util::utf8_to_wide(fname);
                    if (fs::is_regular_file(candidate, ec) && !ec && fs::file_size(candidate, ec) > 1024ULL * 1024ULL) {
                        return candidate;
                    }
                }
            }
            return std::nullopt;
        };

        const bool is_cpu_only = opts.force_cpu ||
            (pipeline::ModelManager::get_primary_provider(opts.force_cpu) == "CPUExecutionProvider");
        std::optional<fs::path> yolo_path;
        if (sel_lower.empty() || sel_lower == "auto" || sel_lower.find("yolo") != std::string::npos) {
            yolo_path = find_yolo_model(is_cpu_only || sel_lower.find("int8") != std::string::npos);
        }

        vellum::detector::BackendType target_backend = vellum::detector::BackendType::Yolo11Seg;
        std::string resolved_model_path;
        if (yolo_path.has_value()) {
            target_backend = vellum::detector::BackendType::Yolo11Seg;
            resolved_model_path = util::path_to_utf8(*yolo_path);
        } else {
            const auto legacy_det = ctx.model_manager->resolve_detector_model(det_selector, opts.allow_model_fallback);
            if (!legacy_det.has_value()) {
                send_json(
                    res,
                    {{"detail",
                      "Requested detector ONNX weights ('" + det_selector +
                          "') are missing. Call POST /api/models/import-koharu (for local Koharu cache) "
                          "or GET /api/models/download first."}},
                    503);
                return;
            }
            target_backend = (legacy_det->backend == pipeline::DetectorBackend::KoharuRfDetr)
                ? vellum::detector::BackendType::RfDetrSeg
                : vellum::detector::BackendType::LegacyCtd;
            resolved_model_path = util::path_to_utf8(legacy_det->model_path);
        }

        const BatchKeepAliveHint keep_hint = parse_batch_keep_alive(req);
        const bool want_auto_unload = opts.auto_unload;
        if (keep_hint.keep_alive.value_or(false)) {
            opts.auto_unload = false;
        }

        try {
            int img_w = 0;
            int img_h = 0;
            int img_c = 0;
            std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> rgb(
                stbi_load_from_memory(
                    reinterpret_cast<const stbi_uc*>(upload.content.data()),
                    static_cast<int>(upload.content.size()),
                    &img_w, &img_h, &img_c, 3),
                &stbi_image_free);
            if (!rgb || img_w <= 0 || img_h <= 0) {
                send_json(res, {{"detail", "Failed to decode uploaded image bytes."}}, 400);
                return;
            }

            vellum::detector::DetectorConfig v2_cfg;
            v2_cfg.model_path = resolved_model_path;
            v2_cfg.backend = target_backend;
            v2_cfg.force_cpu = opts.force_cpu;
            v2_cfg.keep_session_alive = !opts.auto_unload;
            v2_cfg.export_dir = export_dir;
            v2_cfg.enable_adaptive_tiling = current_settings.value("enable_adaptive_tiling", true);
            v2_cfg.sfx_suppress_iou = static_cast<float>(current_settings.value("sfx_suppress_iou", 0.60));
            if (current_settings.contains("whisper_threshold") && current_settings["whisper_threshold"].is_number()) {
                v2_cfg.class_thresholds[static_cast<size_t>(vellum::detector::ClassId::WhisperBubble)] =
                    current_settings["whisper_threshold"].get<float>();
            }
            if (current_settings.contains("borderless_threshold") && current_settings["borderless_threshold"].is_number()) {
                v2_cfg.class_thresholds[static_cast<size_t>(vellum::detector::ClassId::BorderlessDialogue)] =
                    current_settings["borderless_threshold"].get<float>();
            }
            if (opts.reading_order == "ltr") {
                v2_cfg.reading_direction = vellum::detector::ReadingDirection::LTR;
            } else if (opts.reading_order == "rtl") {
                v2_cfg.reading_direction = vellum::detector::ReadingDirection::RTL;
            } else {
                v2_cfg.reading_direction = (opts.source_language == "ja")
                    ? vellum::detector::ReadingDirection::RTL
                    : vellum::detector::ReadingDirection::LTR;
            }
            if (opts.nms_threshold.has_value()) {
                v2_cfg.nms_iou_threshold = *opts.nms_threshold;
            }
            if (opts.conf_threshold.has_value()) {
                v2_cfg.class_thresholds[0] = *opts.conf_threshold;
            } else if (opts.text_threshold.has_value()) {
                v2_cfg.class_thresholds[0] = *opts.text_threshold;
            }
            if (opts.bubble_threshold.has_value()) {
                v2_cfg.class_thresholds[0] = *opts.bubble_threshold;
            }
            if (opts.panel_threshold.has_value()) {
                v2_cfg.class_thresholds[static_cast<size_t>(vellum::detector::ClassId::Panel)] = *opts.panel_threshold;
            }

            ctx.cancel_pending_unload();
            vellum::detector::PageDetectionResult v2_res;
            {
                std::lock_guard<std::mutex> lock(ctx.inference_mutex);
                if (!ctx.detector_v2 || ctx.detector_v2_backend != target_backend) {
                    if (ctx.detector_v2) {
                        ctx.detector_v2->unload();
                    }
                    ctx.detector_v2 = vellum::detector::DetectorFactory::create(target_backend);
                    ctx.detector_v2_backend = target_backend;
                }
                ctx.detector_v2->load(v2_cfg);
                v2_res = vellum::detector::AdaptiveTiler::detectPage(
                    *ctx.detector_v2, rgb.get(), img_w, img_h, upload.filename, v2_cfg);
            }
            if (!v2_cfg.export_dir.empty()) {
                vellum::detector::exportPageArtifacts(v2_res, rgb.get(), img_w, img_h, v2_cfg);
            }
            if (want_auto_unload) {
                if (keep_hint.unload_after) {
                    ctx.unload_all_sessions();
                } else if (keep_hint.keep_alive.value_or(false)) {
                    ctx.schedule_deferred_unload(60);
                }
            }

            // Build tight text-stroke inpainting mask strictly inside each dialogue safe_bbox
            // (never pass full YOLO/RF-DETR bubble segmentation masks into LaMa inpainting).
            const std::vector<uint8_t> page_mask =
                vellum::detector::buildTextStrokeInpaintMask(
                    rgb.get(), img_w, img_h, v2_res.detections);

            const auto mask_png = pipeline::TextDetector::encode_mask_png(page_mask, img_w, img_h);
            const auto [mask_id, mask_url] = save_mask_png_to_cache(mask_png);

            std::optional<std::string> mask_b64;
            if (include_mask_base64) {
                mask_b64 = "data:image/png;base64," +
                           pipeline::TextDetector::base64_encode(mask_png.data(), mask_png.size());
            }

            json bubbles_json = json::array();
            json ordered_ids = json::array();
            json layout_bubbles_json = json::array();
            for (const auto& d : v2_res.detections) {
                const std::string id_str = "b_" + std::to_string(d.id);
                const int x1 = static_cast<int>(std::lround(d.bbox.x1));
                const int y1 = static_cast<int>(std::lround(d.bbox.y1));
                const int x2 = static_cast<int>(std::lround(d.bbox.x2));
                const int y2 = static_cast<int>(std::lround(d.bbox.y2));
                const int sx1 = static_cast<int>(std::lround(d.safe_bbox.x1));
                const int sy1 = static_cast<int>(std::lround(d.safe_bbox.y1));
                const int sx2 = static_cast<int>(std::lround(d.safe_bbox.x2));
                const int sy2 = static_cast<int>(std::lround(d.safe_bbox.y2));

                json poly_json = json::array();
                for (const auto& pt : d.mask.polygon) {
                    poly_json.push_back({static_cast<int>(std::lround(pt.x)), static_cast<int>(std::lround(pt.y))});
                }

                const std::string legacy_cat =
                    (d.class_id == vellum::detector::ClassId::SfxIgnore) ? "onomatopoeia" : "text";
                const std::string text_dir =
                    (v2_res.reading_direction == vellum::detector::ReadingDirection::RTL && (y2 - y1) >= (x2 - x1))
                        ? "vertical"
                        : "horizontal";

                bubbles_json.push_back({
                    {"id", id_str},
                    {"order", d.reading_order},
                    {"reading_order", d.reading_order},
                    {"bbox", {x1, y1, x2, y2}},
                    {"bubble_bbox", {x1, y1, x2, y2}},
                    {"safe_bbox", {sx1, sy1, sx2, sy2}},
                    {"polygon", std::move(poly_json)},
                    {"confidence", d.confidence},
                    {"text_direction", text_dir},
                    {"category", legacy_cat},
                    {"class_id", static_cast<int>(d.class_id)},
                    {"class_name", d.class_name},
                    {"panel_id", d.panel_id},
                    {"is_cross_panel", d.is_cross_panel},
                });
                ordered_ids.push_back(id_str);

                if (vellum::detector::is_dialogue_class(d.class_id)) {
                    layout_bubbles_json.push_back({
                        {"id", "lb_" + std::to_string(d.id)},
                        {"order", d.reading_order},
                        {"bbox", {x1, y1, x2, y2}},
                        {"confidence", d.confidence},
                        {"category", "bubble"},
                    });
                }
            }

            json panels_json = json::array();
            for (const auto& p : v2_res.panels) {
                panels_json.push_back({
                    {"id", "p_" + std::to_string(p.id)},
                    {"panel_id", p.id},
                    {"order", p.reading_order},
                    {"reading_order", p.reading_order},
                    {"bbox", {static_cast<int>(std::lround(p.bbox.x1)),
                              static_cast<int>(std::lround(p.bbox.y1)),
                              static_cast<int>(std::lround(p.bbox.x2)),
                              static_cast<int>(std::lround(p.bbox.y2))}},
                    {"bbox_norm", {p.bbox_norm.x1, p.bbox_norm.y1, p.bbox_norm.x2, p.bbox_norm.y2}},
                    {"confidence", p.confidence},
                    {"category", "panel"},
                });
            }

            std::string page_id = "page_001";
            if (!upload.filename.empty()) {
                std::string fname = upload.filename;
                std::replace(fname.begin(), fname.end(), '\\', '/');
                if (const auto slash = fname.rfind('/'); slash != std::string::npos) {
                    fname = fname.substr(slash + 1);
                }
                const std::string stem = util::path_to_utf8(fs::path(util::utf8_to_wide(fname)).stem());
                if (!stem.empty()) {
                    page_id = stem;
                }
            }

            const nlohmann::json v2_schema = vellum::detector::toSchemaV2Json(v2_res);
            json payload = {
                {"schema_version", "2.0"},
                {"page_id", page_id},
                {"dimensions", {{"width", img_w}, {"height", img_h}}},
                {"source_lang", opts.source_language},
                {"source_language", opts.source_language},
                {"reading_order", v2_res.reading_direction == vellum::detector::ReadingDirection::RTL ? "rtl" : "ltr"},
                {"execution_provider", v2_res.runtime_meta.execution_provider},
                {"detector_backend", v2_schema["runtime_meta"]["backend"]},
                {"detector_model", v2_res.runtime_meta.model_name},
                {"model_path", resolved_model_path},
                {"runtime_meta", v2_schema["runtime_meta"]},
                {"mask_id", mask_id},
                {"mask_url", mask_url},
                {"mask", mask_b64.has_value() ? *mask_b64 : mask_url},
                {"ordered_ids", std::move(ordered_ids)},
                {"bubbles", bubbles_json},
                {"blocks", std::move(bubbles_json)},
                {"layout_bubbles", std::move(layout_bubbles_json)},
                {"panels", std::move(panels_json)},
                {"detections", v2_schema["detections"]},
            };
            if (mask_b64.has_value()) {
                payload["mask_base64"] = *mask_b64;
            }
            send_json(res, payload);
        } catch (const std::invalid_argument& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", std::string("Text detection failed: ") + ex.what()}}, 500);
        }
    };
    server.Post("/api/stage/detect", handle_stage_detect);
    server.Post("/api/pipeline/detect", handle_stage_detect);

    // -------------------------------------------------------------------------
    // Stage 2: Multi-Language OCR Router (`POST /api/stage/ocr`, `/api/pipeline/ocr`)
    // -------------------------------------------------------------------------

    auto handle_stage_ocr = [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!req.is_multipart_form_data()) {
            send_json(res, {{"detail", "Expected multipart/form-data upload with field 'image' or 'file'."}}, 400);
            return;
        }

        httplib::FormData upload;
        if (req.form.has_file("image")) {
            upload = req.form.get_file("image");
        } else if (req.form.has_file("file")) {
            upload = req.form.get_file("file");
        } else {
            send_json(res, {{"detail", "Missing uploaded image file. Provide multipart form field 'image' or 'file'."}}, 400);
            return;
        }
        if (upload.content.empty()) {
            send_json(res, {{"detail", "Uploaded image file is empty."}}, 400);
            return;
        }

        std::string raw_boxes_json;
        if (req.form.has_field("bboxes")) {
            raw_boxes_json = req.form.get_field("bboxes");
        } else if (req.form.has_field("blocks")) {
            raw_boxes_json = req.form.get_field("blocks");
        } else if (req.form.has_field("bubbles")) {
            raw_boxes_json = req.form.get_field("bubbles");
        }
        if (util::trim_ascii(raw_boxes_json).empty()) {
            send_json(
                res,
                {{"detail", "Missing selected bounding boxes. Provide JSON array in form field 'bboxes' (or 'blocks'/'bubbles')."}},
                400);
            return;
        }

        const json current_settings = ctx.settings_manager.load();
        pipeline::OcrOptions opts;
        populate_ocr_options_from_settings(opts, current_settings);
        std::vector<pipeline::OcrBubbleInput> parsed_bubbles;
        std::string effective_ocr_model;
        try {
            const std::string raw_lang = req.form.has_field("source_lang")
                ? req.form.get_field("source_lang")
                : (req.form.has_field("source_language")
                       ? req.form.get_field("source_language")
                       : current_settings.value("source_lang", std::string("ja")));
            opts.source_lang = pipeline::TextDetector::normalize_source_language(raw_lang);
            if (req.form.has_field("ocr_model")) {
                const std::string om = util::trim_ascii(req.form.get_field("ocr_model"));
                if (!om.empty()) {
                    opts.ocr_model = SettingsManager::normalize_ocr_model(om);
                }
            }
            effective_ocr_model = pipeline::OcrEngine::resolve_effective_ocr_model(
                opts.ocr_model, opts.source_lang);
            parsed_bubbles = pipeline::OcrEngine::parse_bubbles_json(raw_boxes_json);
        } catch (const std::invalid_argument& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
            return;
        }

        if (effective_ocr_model == "manga-ocr") {
            if (!ctx.model_manager->resolve_manga_ocr_models().has_value()) {
                send_json(
                    res,
                    {{"detail",
                      "Japanese manga-ocr ONNX weights are missing. Call POST /api/models/import-koharu "
                      "or GET /api/models/download first."}},
                    503);
                return;
            }
        } else if (effective_ocr_model == "baberu-ocr") {
            if (!ctx.model_manager->resolve_baberu_ocr_models().has_value()) {
                send_json(res, {{"detail", "Baberu OCR ONNX weights are missing in ./models/baberu-ocr/."}}, 503);
                return;
            }
        } else if (effective_ocr_model == "hayai-ocr") {
            if (!ctx.model_manager->resolve_hayai_ocr_models().has_value()) {
                send_json(res, {{"detail", "Hayai OCR ONNX weights are missing in ./models/hayai-ocr/."}}, 503);
                return;
            }
        } else if (effective_ocr_model == "rapidocr-ppocrv4") {
            if (!ctx.model_manager->resolve_rapidocr_paths(opts.source_lang).has_value()) {
                send_json(
                    res,
                    {{"detail",
                      "RapidOCR (" + opts.source_lang + ") PP-OCRv4 ONNX weights are missing. "
                      "Call GET /api/models/download first."}},
                    503);
                return;
            }
        }

        const BatchKeepAliveHint keep_hint = parse_batch_keep_alive(req);
        const bool want_auto_unload = opts.auto_unload;
        if (keep_hint.keep_alive.value_or(false)) {
            opts.auto_unload = false;
        }

        try {
            ctx.cancel_pending_unload();
            pipeline::OcrStageResult ocr_res;
            {
                std::lock_guard<std::mutex> lock(ctx.inference_mutex);
                ocr_res = ctx.ocr_engine->recognize_from_bytes(
                    reinterpret_cast<const uint8_t*>(upload.content.data()),
                    upload.content.size(),
                    parsed_bubbles,
                    opts);
            }
            if (want_auto_unload) {
                if (keep_hint.unload_after) {
                    ctx.unload_all_sessions();
                } else if (keep_hint.keep_alive.value_or(false)) {
                    ctx.schedule_deferred_unload(60);
                }
            }

            json blocks_json = json::array();
            json texts_json = json::array();
            for (const auto& blk : ocr_res.blocks) {
                blocks_json.push_back(blk.to_json());
                texts_json.push_back(blk.source_text);
            }

            send_json(res, {
                {"source_lang", ocr_res.source_lang},
                {"source_language", ocr_res.source_lang},
                {"engine", ocr_res.engine},
                {"ocr_model", ocr_res.ocr_model},
                {"requested_ocr_model", ocr_res.requested_model},
                {"execution_provider", ocr_res.execution_provider},
                {"count", ocr_res.blocks.size()},
                {"texts", std::move(texts_json)},
                {"blocks", blocks_json},
                {"bubbles", std::move(blocks_json)},
            });
        } catch (const std::invalid_argument& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        } catch (const std::exception& ex) {
            const int err_status = (effective_ocr_model == "paddleocr-vl-gguf") ? 503 : 500;
            send_json(res, {{"detail", std::string("OCR failed: ") + ex.what()}}, err_status);
        }
    };
    server.Post("/api/stage/ocr", handle_stage_ocr);
    server.Post("/api/pipeline/ocr", handle_stage_ocr);

    // -------------------------------------------------------------------------
    // Stage 3: Tile-Based LaMa Inpainter (`POST /api/stage/inpaint`, `/api/pipeline/inpaint`)
    // -------------------------------------------------------------------------

    auto handle_stage_inpaint = [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!req.is_multipart_form_data()) {
            send_json(res, {{"detail", "Expected multipart/form-data upload with field 'image' or 'file'."}}, 400);
            return;
        }

        httplib::FormData upload;
        if (req.form.has_file("image")) {
            upload = req.form.get_file("image");
        } else if (req.form.has_file("file")) {
            upload = req.form.get_file("file");
        } else {
            send_json(res, {{"detail", "Missing uploaded image file. Provide multipart form field 'image' or 'file'."}}, 400);
            return;
        }
        if (upload.content.empty()) {
            send_json(res, {{"detail", "Uploaded image file is empty."}}, 400);
            return;
        }

        const json current_settings = ctx.settings_manager.load();
        pipeline::InpaintOptions opts;
        populate_inpaint_options_from_settings(opts, current_settings);
        try {
            if (req.form.has_field("inpainting_model")) {
                const std::string im = util::trim_ascii(req.form.get_field("inpainting_model"));
                if (!im.empty()) {
                    opts.inpainting_model = SettingsManager::normalize_inpainting_model(im);
                    opts.allow_model_fallback = false;
                }
            }
            if (req.form.has_field("lama_tile_size") || req.form.has_field("tile_size")) {
                const std::string ts_str = util::trim_ascii(
                    req.form.has_field("lama_tile_size") ? req.form.get_field("lama_tile_size") : req.form.get_field("tile_size"));
                if (!ts_str.empty()) {
                    try {
                        opts.tile_size = std::max(256, std::min(1024, std::stoi(ts_str)));
                    } catch (...) {
                        throw std::invalid_argument("Invalid integer for 'lama_tile_size': '" + ts_str + "'");
                    }
                }
            }
        } catch (const std::invalid_argument& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
            return;
        }

        const bool has_inpaint_weights =
            (opts.inpainting_model == "aot-inpainting")
                ? (ctx.model_manager->resolve_aot_model().has_value() || ctx.model_manager->resolve_lama_model().has_value())
                : (opts.inpainting_model == "flux-2-klein" || opts.inpainting_model == "rorem-mixed")
                ? (ctx.model_manager->resolve_aot_model().has_value() || ctx.model_manager->resolve_lama_model().has_value())
                : ctx.model_manager->resolve_lama_model(opts.inpainting_model, opts.allow_model_fallback).has_value();
        if (!has_inpaint_weights) {
            send_json(
                res,
                {{"detail",
                  "Requested Inpainting ONNX weights ('" + opts.inpainting_model +
                      "') are missing. Call POST /api/models/import-koharu or GET /api/models/download first."}},
                503);
            return;
        }

        int orig_w = 0;
        int orig_h = 0;
        std::vector<uint8_t> rgb_image;
        try {
            rgb_image = pipeline::LamaInpainter::decode_rgb_from_bytes(
                reinterpret_cast<const uint8_t*>(upload.content.data()),
                upload.content.size(),
                orig_w,
                orig_h);
        } catch (const std::invalid_argument& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
            return;
        }

        std::optional<std::vector<std::array<int, 4>>> selected_bboxes;
        std::string raw_boxes_json;
        if (req.form.has_field("bboxes")) {
            raw_boxes_json = req.form.get_field("bboxes");
        } else if (req.form.has_field("blocks")) {
            raw_boxes_json = req.form.get_field("blocks");
        } else if (req.form.has_field("bubbles")) {
            raw_boxes_json = req.form.get_field("bubbles");
        }
        if (!util::trim_ascii(raw_boxes_json).empty()) {
            try {
                const auto parsed = pipeline::OcrEngine::parse_bubbles_json(raw_boxes_json);
                std::vector<std::array<int, 4>> boxes;
                boxes.reserve(parsed.size());
                for (const auto& b : parsed) {
                    boxes.push_back(b.bbox);
                }
                selected_bboxes = std::move(boxes);
            } catch (const std::invalid_argument& ex) {
                send_json(res, {{"detail", ex.what()}}, 400);
                return;
            }
        }

        std::vector<uint8_t> working_mask;
        bool has_explicit_mask = false;

        try {
            if (req.form.has_file("mask")) {
                const auto mask_upload = req.form.get_file("mask");
                if (mask_upload.content.empty()) {
                    send_json(res, {{"detail", "Uploaded mask file is empty."}}, 400);
                    return;
                }
                working_mask = pipeline::LamaInpainter::decode_mask_from_bytes(
                    reinterpret_cast<const uint8_t*>(mask_upload.content.data()),
                    mask_upload.content.size(),
                    orig_w,
                    orig_h);
                has_explicit_mask = true;
            } else {
                std::string mask_ref;
                if (req.form.has_field("mask_id") && !util::trim_ascii(req.form.get_field("mask_id")).empty()) {
                    mask_ref = util::trim_ascii(req.form.get_field("mask_id"));
                } else if (req.form.has_field("mask_url") && !util::trim_ascii(req.form.get_field("mask_url")).empty()) {
                    mask_ref = util::trim_ascii(req.form.get_field("mask_url"));
                }
                if (!mask_ref.empty()) {
                    const fs::path cached_mask_path = resolve_cached_image_path(mask_ref);
                    const auto mask_file_bytes = util::read_file_bytes(cached_mask_path);
                    if (!mask_file_bytes.has_value() || mask_file_bytes->empty()) {
                        send_json(res, {{"detail", "Failed to read cached mask image from disk."}}, 404);
                        return;
                    }
                    working_mask = pipeline::LamaInpainter::decode_mask_from_bytes(
                        mask_file_bytes->data(), mask_file_bytes->size(), orig_w, orig_h);
                    has_explicit_mask = true;
                }
            }
        } catch (const ComicExportError& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
            return;
        } catch (const std::invalid_argument& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
            return;
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", ex.what()}}, 404);
            return;
        }

        if (!has_explicit_mask) {
            if (!selected_bboxes.has_value() || selected_bboxes->empty()) {
                send_json(
                    res,
                    {{"detail", "Provide at least one of 'mask', 'mask_id', 'mask_url', or 'bboxes'."}},
                    400);
                return;
            }
            working_mask = pipeline::LamaInpainter::synthesize_mask_from_bboxes(*selected_bboxes, orig_w, orig_h);
        } else if (selected_bboxes.has_value()) {
            pipeline::LamaInpainter::restrict_mask_to_bboxes(working_mask, *selected_bboxes, orig_w, orig_h, 8);
        }

        opts.bboxes = selected_bboxes;
        bool refine_mask = opts.refine_mask;
        bool include_base64 = false;
        if (req.form.has_field("tile_padding")) {
            const std::string tp_str = util::trim_ascii(req.form.get_field("tile_padding"));
            if (!tp_str.empty()) {
                try {
                    opts.tile_padding = std::max(0, std::min(128, std::stoi(tp_str)));
                } catch (...) {
                    send_json(res, {{"detail", "Invalid integer for 'tile_padding': '" + tp_str + "'"}}, 400);
                    return;
                }
            }
        }
        if (req.form.has_field("refine_mask")) {
            refine_mask = parse_bool_param(req.form.get_field("refine_mask"), refine_mask);
        }
        if (req.form.has_field("include_base64")) {
            include_base64 = parse_bool_param(req.form.get_field("include_base64"), false);
        }
        // Only run refine_bubble_mask when a synthetic bounding-box mask was built (!has_explicit_mask);
        // when an explicit text-stroke mask is uploaded/cached, preserve it without re-thresholding the full bbox.
        opts.refine_mask = (!has_explicit_mask) || (req.form.has_field("refine_mask") && refine_mask);

        const BatchKeepAliveHint keep_hint = parse_batch_keep_alive(req);
        const bool want_auto_unload = opts.auto_unload;
        if (keep_hint.keep_alive.value_or(false)) {
            opts.auto_unload = false;
        }

        try {
            ctx.cancel_pending_unload();
            pipeline::InpaintResult inpaint_res;
            {
                std::lock_guard<std::mutex> lock(ctx.inference_mutex);
                inpaint_res = ctx.inpainter->inpaint_rgb(rgb_image.data(), orig_w, orig_h, working_mask, opts);
            }
            if (want_auto_unload) {
                if (keep_hint.unload_after) {
                    ctx.unload_all_sessions();
                } else if (keep_hint.keep_alive.value_or(false)) {
                    ctx.schedule_deferred_unload(60);
                }
            }

            const auto png_bytes = pipeline::LamaInpainter::encode_rgb_png(
                inpaint_res.rgb, inpaint_res.width, inpaint_res.height);
            const auto [cleaned_id, cleaned_url] = save_inpainted_png_to_cache(png_bytes);

            const std::string image_payload = include_base64
                ? ("data:image/png;base64," +
                   pipeline::TextDetector::base64_encode(png_bytes.data(), png_bytes.size()))
                : cleaned_url;

            send_json(res, {
                {"dimensions", {{"width", inpaint_res.width}, {"height", inpaint_res.height}}},
                {"execution_provider", inpaint_res.execution_provider},
                {"inpainting_model", inpaint_res.model_id},
                {"model_path", inpaint_res.model_path},
                {"tile_size", inpaint_res.tile_size},
                {"is_koharu_manga", inpaint_res.is_koharu_manga},
                {"tiles_inpainted", inpaint_res.tiles_processed},
                {"cleaned_image_id", cleaned_id},
                {"cleaned_image_url", cleaned_url},
                {"inpainted_image_url", cleaned_url},
                {"cleaned_image", image_payload},
                {"inpainted_image", image_payload},
            });
        } catch (const std::invalid_argument& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", std::string("LaMa inpainting failed: ") + ex.what()}}, 500);
        }
    };
    server.Post("/api/stage/inpaint", handle_stage_inpaint);
    server.Post("/api/pipeline/inpaint", handle_stage_inpaint);

    // -------------------------------------------------------------------------
    // Stage 4: Context-Aware Multi-Provider Translation (`POST /api/stage/translate`)
    // -------------------------------------------------------------------------

    server.Post("/api/stage/translate", [&ctx](const httplib::Request& req, httplib::Response& res) {
        json raw_blocks = json::array();
        json overrides = json::object();
        std::string raw_src = "ja";
        std::string raw_tgt = "id";
        std::optional<std::string> uploaded_img_bytes;
        std::optional<std::string> image_url;
        std::optional<std::string> image_base64;

        try {
            if (req.is_multipart_form_data()) {
                std::string raw_blocks_str;
                if (req.form.has_field("blocks")) {
                    raw_blocks_str = req.form.get_field("blocks");
                } else if (req.form.has_field("bubbles")) {
                    raw_blocks_str = req.form.get_field("bubbles");
                } else if (req.form.has_field("dialogues")) {
                    raw_blocks_str = req.form.get_field("dialogues");
                }
                if (!util::trim_ascii(raw_blocks_str).empty()) {
                    raw_blocks = json::parse(raw_blocks_str, nullptr, false);
                    if (raw_blocks.is_discarded() || !raw_blocks.is_array()) {
                        send_json(res, {{"detail", "'blocks' form field must be a valid JSON array."}}, 400);
                        return;
                    }
                }

                overrides = extract_multipart_translation_overrides(req);

                if (req.form.has_file("image")) {
                    uploaded_img_bytes = req.form.get_file("image").content;
                } else if (req.form.has_file("file")) {
                    uploaded_img_bytes = req.form.get_file("file").content;
                }

                if (req.form.has_field("source_language") && !util::trim_ascii(req.form.get_field("source_language")).empty()) {
                    raw_src = req.form.get_field("source_language");
                } else if (req.form.has_field("source_lang") && !util::trim_ascii(req.form.get_field("source_lang")).empty()) {
                    raw_src = req.form.get_field("source_lang");
                }

                if (req.form.has_field("target_language") && !util::trim_ascii(req.form.get_field("target_language")).empty()) {
                    raw_tgt = req.form.get_field("target_language");
                } else if (req.form.has_field("target_lang") && !util::trim_ascii(req.form.get_field("target_lang")).empty()) {
                    raw_tgt = req.form.get_field("target_lang");
                }

                if (req.form.has_field("image_url") && !util::trim_ascii(req.form.get_field("image_url")).empty()) {
                    image_url = req.form.get_field("image_url");
                }
                if (req.form.has_field("image_base64") && !util::trim_ascii(req.form.get_field("image_base64")).empty()) {
                    image_base64 = req.form.get_field("image_base64");
                }
            } else {
                if (util::trim_ascii(req.body).empty()) {
                    send_json(res, {{"detail", "Empty request body."}}, 400);
                    return;
                }
                const json body = json::parse(req.body, nullptr, false);
                if (body.is_discarded() || !body.is_object()) {
                    send_json(res, {{"detail", "Invalid JSON request body."}}, 400);
                    return;
                }

                if (body.contains("blocks") && body["blocks"].is_array() && !body["blocks"].empty()) {
                    raw_blocks = body["blocks"];
                } else if (body.contains("bubbles") && body["bubbles"].is_array() && !body["bubbles"].empty()) {
                    raw_blocks = body["bubbles"];
                } else if (body.contains("dialogues") && body["dialogues"].is_array() && !body["dialogues"].empty()) {
                    raw_blocks = body["dialogues"];
                } else if (body.contains("blocks") && body["blocks"].is_array()) {
                    raw_blocks = body["blocks"];
                }

                if (body.contains("source_language") && body["source_language"].is_string() &&
                    !util::trim_ascii(body["source_language"].get<std::string>()).empty()) {
                    raw_src = body["source_language"].get<std::string>();
                } else if (body.contains("source_lang") && body["source_lang"].is_string() &&
                           !util::trim_ascii(body["source_lang"].get<std::string>()).empty()) {
                    raw_src = body["source_lang"].get<std::string>();
                }

                if (body.contains("target_language") && body["target_language"].is_string() &&
                    !util::trim_ascii(body["target_language"].get<std::string>()).empty()) {
                    raw_tgt = body["target_language"].get<std::string>();
                } else if (body.contains("target_lang") && body["target_lang"].is_string() &&
                           !util::trim_ascii(body["target_lang"].get<std::string>()).empty()) {
                    raw_tgt = body["target_lang"].get<std::string>();
                }

                if (body.contains("image_url") && body["image_url"].is_string() &&
                    !util::trim_ascii(body["image_url"].get<std::string>()).empty()) {
                    image_url = body["image_url"].get<std::string>();
                }
                if (body.contains("image_base64") && body["image_base64"].is_string() &&
                    !util::trim_ascii(body["image_base64"].get<std::string>()).empty()) {
                    image_base64 = body["image_base64"].get<std::string>();
                }

                if (body.contains("config") && body["config"].is_object()) {
                    overrides = body["config"];
                }
                for (const char* top_key : {
                         "provider", "model", "base_url", "api_key",
                         "temperature", "top_p", "top_k", "max_tokens",
                         "frequency_penalty", "reasoning_effort", "translation_tone",
                         "strip_thinking", "use_vision", "vision_max_side",
                         "custom_instructions", "glossary", "timeout"}) {
                    if (body.contains(top_key) && !body[top_key].is_null() && !overrides.contains(top_key)) {
                        overrides[top_key] = body[top_key];
                    }
                }
            }

            const std::string src_lang = pipeline::ComicTranslator::normalize_source_language(raw_src);
            const std::string tgt_lang = pipeline::ComicTranslator::normalize_target_language(raw_tgt);
            const auto input_items = parse_translate_block_inputs(raw_blocks);

            const json current_settings = ctx.settings_manager.load();
            const auto translator_cfg = pipeline::TranslatorConfig::from_settings_with_overrides(
                current_settings, overrides);
            pipeline::ComicTranslator translator(translator_cfg);

            DecodedOptionalImage page_img = decode_optional_page_image(
                uploaded_img_bytes, image_url, image_base64);

            std::optional<std::string> page_image_data_uri;
            json page_image_dims = nullptr;
            if (page_img.valid) {
                page_image_dims = {{"width", page_img.width}, {"height", page_img.height}};
                if (translator.config().use_vision) {
                    page_image_data_uri = pipeline::ComicTranslator::encode_rgb_for_vision(
                        page_img.rgb.data(),
                        page_img.width,
                        page_img.height,
                        translator.config().vision_max_side);
                }
            }

            std::vector<pipeline::DialogueBlockInput> dialogue_inputs;
            dialogue_inputs.reserve(input_items.size());
            for (const auto& item : input_items) {
                dialogue_inputs.push_back({item.id, item.original_text});
            }

            pipeline::TranslateRequestOptions req_opts;
            req_opts.source_lang = src_lang;
            req_opts.target_lang = tgt_lang;
            req_opts.page_image_data_uri = page_image_data_uri;
            req_opts.use_vision = translator.config().use_vision;

            const auto translation_res = translator.translate_blocks(dialogue_inputs, req_opts);
            json translated_blocks = build_translated_blocks_json(
                input_items, translation_res.translations);

            json dialogues_json = json::array();
            for (const auto& d : translation_res.dialogues) {
                dialogues_json.push_back(d.to_json());
            }

            send_json(res, {
                {"provider", translation_res.provider},
                {"model", translation_res.model},
                {"vision_used", translation_res.vision_used},
                {"reasoning_effort", translation_res.reasoning_effort},
                {"source_lang", src_lang},
                {"target_lang", tgt_lang},
                {"source_language", src_lang},
                {"target_language", tgt_lang},
                {"has_page_image", page_img.valid},
                {"page_image_dimensions", page_image_dims},
                {"translations", translation_res.translations},
                {"dialogues", std::move(dialogues_json)},
                {"blocks", translated_blocks},
                {"bubbles", std::move(translated_blocks)},
            });
        } catch (const ComicExportError& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        } catch (const std::invalid_argument& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        } catch (const pipeline::ProviderConnectionError& ex) {
            send_json(res, {{"detail", ex.what()}}, 503);
        } catch (const pipeline::ProviderTranslationError& ex) {
            send_json(res, {{"detail", ex.what()}}, 502);
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", std::string("Translation failed: ") + ex.what()}}, 500);
        }
    });

    // -------------------------------------------------------------------------
    // Retranslate Existing Dialogue Blocks (`POST /api/retranslate`, `/api/pipeline/translate`)
    // -------------------------------------------------------------------------

    auto handle_retranslate = [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (util::trim_ascii(req.body).empty()) {
            send_json(res, {{"detail", "Empty request body."}}, 400);
            return;
        }
        const json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.is_object()) {
            send_json(res, {{"detail", "Invalid JSON request body."}}, 400);
            return;
        }

        try {
            std::string raw_src = "ja";
            if (body.contains("source_language") && body["source_language"].is_string() &&
                !util::trim_ascii(body["source_language"].get<std::string>()).empty()) {
                raw_src = body["source_language"].get<std::string>();
            } else if (body.contains("source_lang") && body["source_lang"].is_string() &&
                       !util::trim_ascii(body["source_lang"].get<std::string>()).empty()) {
                raw_src = body["source_lang"].get<std::string>();
            }

            std::string raw_tgt = "id";
            if (body.contains("target_language") && body["target_language"].is_string() &&
                !util::trim_ascii(body["target_language"].get<std::string>()).empty()) {
                raw_tgt = body["target_language"].get<std::string>();
            } else if (body.contains("target_lang") && body["target_lang"].is_string() &&
                       !util::trim_ascii(body["target_lang"].get<std::string>()).empty()) {
                raw_tgt = body["target_lang"].get<std::string>();
            }

            const std::string src_lang = pipeline::ComicTranslator::normalize_source_language(raw_src);
            const std::string tgt_lang = pipeline::ComicTranslator::normalize_target_language(raw_tgt);

            json raw_blocks = json::array();
            if (body.contains("blocks") && body["blocks"].is_array() && !body["blocks"].empty()) {
                raw_blocks = body["blocks"];
            } else if (body.contains("bubbles") && body["bubbles"].is_array() && !body["bubbles"].empty()) {
                raw_blocks = body["bubbles"];
            } else if (body.contains("dialogues") && body["dialogues"].is_array() && !body["dialogues"].empty()) {
                raw_blocks = body["dialogues"];
            }
            const auto input_items = parse_translate_block_inputs(raw_blocks);

            json overrides = json::object();
            if (body.contains("config") && body["config"].is_object()) {
                overrides = body["config"];
            }
            for (const char* top_key : {
                     "provider", "model", "base_url", "api_key",
                     "temperature", "top_p", "top_k", "max_tokens",
                     "frequency_penalty", "reasoning_effort", "translation_tone",
                     "strip_thinking", "use_vision", "vision_max_side",
                     "custom_instructions", "glossary", "timeout"}) {
                if (body.contains(top_key) && !body[top_key].is_null()) {
                    overrides[top_key] = body[top_key];
                }
            }

            const json current_settings = ctx.settings_manager.load();
            const auto translator_cfg = pipeline::TranslatorConfig::from_settings_with_overrides(
                current_settings, overrides);
            pipeline::ComicTranslator translator(translator_cfg);

            std::optional<std::string> page_image_data_uri;
            if (translator.config().use_vision) {
                std::optional<std::string> image_url;
                std::optional<std::string> image_base64;
                if (body.contains("image_url") && body["image_url"].is_string() &&
                    !util::trim_ascii(body["image_url"].get<std::string>()).empty()) {
                    image_url = body["image_url"].get<std::string>();
                }
                if (body.contains("image_base64") && body["image_base64"].is_string() &&
                    !util::trim_ascii(body["image_base64"].get<std::string>()).empty()) {
                    image_base64 = body["image_base64"].get<std::string>();
                }
                DecodedOptionalImage page_img = decode_optional_page_image(
                    std::nullopt, image_url, image_base64);
                if (page_img.valid) {
                    page_image_data_uri = pipeline::ComicTranslator::encode_rgb_for_vision(
                        page_img.rgb.data(),
                        page_img.width,
                        page_img.height,
                        translator.config().vision_max_side);
                }
            }

            std::vector<pipeline::DialogueBlockInput> dialogue_inputs;
            dialogue_inputs.reserve(input_items.size());
            for (const auto& item : input_items) {
                dialogue_inputs.push_back({item.id, item.original_text});
            }

            pipeline::TranslateRequestOptions req_opts;
            req_opts.source_lang = src_lang;
            req_opts.target_lang = tgt_lang;
            req_opts.page_image_data_uri = page_image_data_uri;
            req_opts.use_vision = translator.config().use_vision;

            const auto translation_res = translator.translate_blocks(dialogue_inputs, req_opts);
            json retranslated_blocks = build_translated_blocks_json(
                input_items, translation_res.translations);

            json dialogues_json = json::array();
            for (const auto& d : translation_res.dialogues) {
                dialogues_json.push_back(d.to_json());
            }

            send_json(res, {
                {"provider", translation_res.provider},
                {"model", translation_res.model},
                {"vision_used", translation_res.vision_used},
                {"reasoning_effort", translation_res.reasoning_effort},
                {"source_lang", src_lang},
                {"target_lang", tgt_lang},
                {"source_language", src_lang},
                {"target_language", tgt_lang},
                {"translations", translation_res.translations},
                {"dialogues", std::move(dialogues_json)},
                {"blocks", retranslated_blocks},
                {"bubbles", std::move(retranslated_blocks)},
            });
        } catch (const ComicExportError& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        } catch (const std::invalid_argument& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        } catch (const pipeline::ProviderConnectionError& ex) {
            send_json(res, {{"detail", ex.what()}}, 503);
        } catch (const pipeline::ProviderTranslationError& ex) {
            send_json(res, {{"detail", ex.what()}}, 502);
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", std::string("Retranslation failed: ") + ex.what()}}, 500);
        }
    };
    server.Post("/api/retranslate", handle_retranslate);
    server.Post("/api/pipeline/translate", handle_retranslate);

    // -------------------------------------------------------------------------
    // Full End-to-End Page Pipeline (`POST /api/process-page`, `/api/pipeline/process-page`)
    // Detect -> Reading-Order Sort -> OCR -> Tile-Based LaMa Inpaint -> LLM Translate
    // -------------------------------------------------------------------------

    auto handle_process_page = [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!req.is_multipart_form_data()) {
            send_json(res, {{"detail", "Expected multipart/form-data upload with field 'image' or 'file'."}}, 400);
            return;
        }

        httplib::FormData upload;
        if (req.form.has_file("image")) {
            upload = req.form.get_file("image");
        } else if (req.form.has_file("file")) {
            upload = req.form.get_file("file");
        } else {
            send_json(res, {{"detail", "Missing uploaded image file. Provide multipart form field 'image' or 'file'."}}, 400);
            return;
        }
        if (upload.content.empty()) {
            send_json(res, {{"detail", "Uploaded image file is empty."}}, 400);
            return;
        }

        const json current_settings = ctx.settings_manager.load();

        std::string src_lang;
        std::string tgt_lang;
        bool include_base64 = false;
        pipeline::DetectionOptions det_opts;
        pipeline::OcrOptions ocr_opts;
        pipeline::InpaintOptions inpaint_opts;
        populate_detection_options_from_settings(det_opts, current_settings);
        populate_ocr_options_from_settings(ocr_opts, current_settings);
        populate_inpaint_options_from_settings(inpaint_opts, current_settings);
        std::string effective_ocr_model;
        json translation_overrides = json::object();

        try {
            std::string raw_src = current_settings.value("source_lang", std::string("ja"));
            if (req.form.has_field("source_language") && !util::trim_ascii(req.form.get_field("source_language")).empty()) {
                raw_src = req.form.get_field("source_language");
            } else if (req.form.has_field("source_lang") && !util::trim_ascii(req.form.get_field("source_lang")).empty()) {
                raw_src = req.form.get_field("source_lang");
            }

            std::string raw_tgt = current_settings.value("target_lang", std::string("id"));
            if (req.form.has_field("target_language") && !util::trim_ascii(req.form.get_field("target_language")).empty()) {
                raw_tgt = req.form.get_field("target_language");
            } else if (req.form.has_field("target_lang") && !util::trim_ascii(req.form.get_field("target_lang")).empty()) {
                raw_tgt = req.form.get_field("target_lang");
            }

            src_lang = pipeline::ComicTranslator::normalize_source_language(raw_src);
            tgt_lang = pipeline::ComicTranslator::normalize_target_language(raw_tgt);

            if (req.form.has_field("include_base64")) {
                include_base64 = parse_bool_param(req.form.get_field("include_base64"), false);
            }
            if (req.has_param("include_base64")) {
                include_base64 = include_base64 || parse_bool_param(req.get_param_value("include_base64"), false);
            }

            det_opts.source_language = src_lang;
            if (auto cf = parse_optional_float_field(req, "conf_threshold"); cf.has_value()) {
                det_opts.conf_threshold = cf;
            }
            if (auto nf = parse_optional_float_field(req, "nms_threshold"); nf.has_value()) {
                det_opts.nms_threshold = nf;
            }
            if (auto mf = parse_optional_float_field(req, "mask_threshold"); mf.has_value()) {
                det_opts.mask_threshold = mf;
            }
            apply_optional_nullable_threshold_field(req, "text_threshold", det_opts.text_threshold);
            apply_optional_nullable_threshold_field(req, "bubble_threshold", det_opts.bubble_threshold);
            apply_optional_nullable_threshold_field(req, "panel_threshold", det_opts.panel_threshold);

            if (req.form.has_field("detection_model")) {
                const std::string dm = util::trim_ascii(req.form.get_field("detection_model"));
                if (!dm.empty()) {
                    det_opts.detection_model = SettingsManager::normalize_detection_model(dm);
                    det_opts.allow_model_fallback = false;
                }
            } else if (req.form.has_field("detector_backend")) {
                const std::string db = util::trim_ascii(req.form.get_field("detector_backend"));
                if (!db.empty()) {
                    det_opts.detector_backend = db;
                    if (util::to_lower_ascii(db) != "auto") {
                        det_opts.detection_model = SettingsManager::normalize_detection_model(db);
                        det_opts.allow_model_fallback = false;
                    }
                }
            }
            if (req.form.has_field("bbox_padding") || req.form.has_field("box_padding")) {
                const std::string pad_str = util::trim_ascii(
                    req.form.has_field("bbox_padding") ? req.form.get_field("bbox_padding") : req.form.get_field("box_padding"));
                if (!pad_str.empty()) {
                    try {
                        det_opts.box_padding = std::max(0, std::min(64, std::stoi(pad_str)));
                    } catch (...) {
                        throw std::invalid_argument("Invalid integer for 'bbox_padding': '" + pad_str + "'");
                    }
                }
            }
            if (req.form.has_field("reading_order")) {
                const std::string ro = util::to_lower_ascii(util::trim_ascii(req.form.get_field("reading_order")));
                if (ro == "auto" || ro == "rtl" || ro == "ltr") {
                    det_opts.reading_order = ro;
                }
            }

            ocr_opts.source_lang = src_lang;
            if (req.form.has_field("ocr_model")) {
                const std::string om = util::trim_ascii(req.form.get_field("ocr_model"));
                if (!om.empty()) {
                    ocr_opts.ocr_model = SettingsManager::normalize_ocr_model(om);
                }
            }
            effective_ocr_model = pipeline::OcrEngine::resolve_effective_ocr_model(
                ocr_opts.ocr_model, src_lang);

            if (req.form.has_field("inpainting_model")) {
                const std::string im = util::trim_ascii(req.form.get_field("inpainting_model"));
                if (!im.empty()) {
                    inpaint_opts.inpainting_model = SettingsManager::normalize_inpainting_model(im);
                    inpaint_opts.allow_model_fallback = false;
                }
            }
            if (req.form.has_field("lama_tile_size") || req.form.has_field("tile_size")) {
                const std::string ts_str = util::trim_ascii(
                    req.form.has_field("lama_tile_size") ? req.form.get_field("lama_tile_size") : req.form.get_field("tile_size"));
                if (!ts_str.empty()) {
                    try {
                        inpaint_opts.tile_size = std::max(256, std::min(1024, std::stoi(ts_str)));
                    } catch (...) {
                        throw std::invalid_argument("Invalid integer for 'lama_tile_size': '" + ts_str + "'");
                    }
                }
            }
            if (req.form.has_field("tile_padding")) {
                const std::string tp_str = util::trim_ascii(req.form.get_field("tile_padding"));
                if (!tp_str.empty()) {
                    try {
                        inpaint_opts.tile_padding = std::max(0, std::min(128, std::stoi(tp_str)));
                    } catch (...) {
                        throw std::invalid_argument("Invalid integer for 'tile_padding': '" + tp_str + "'");
                    }
                }
            }
            if (req.form.has_field("refine_mask")) {
                inpaint_opts.refine_mask = parse_bool_param(req.form.get_field("refine_mask"), inpaint_opts.refine_mask);
            }

            translation_overrides = extract_multipart_translation_overrides(req);
        } catch (const std::invalid_argument& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
            return;
        }

        // Verify local CV/OCR/Inpainting models before decoding/inferencing
        const std::string det_selector =
            !det_opts.detection_model.empty() ? det_opts.detection_model : det_opts.detector_backend;
        const std::string det_sel_lower = util::to_lower_ascii(det_selector);
        const bool is_cpu_det = det_opts.force_cpu ||
            (pipeline::ModelManager::get_primary_provider(det_opts.force_cpu) == "CPUExecutionProvider");

        auto find_yolo_model = [&](bool prefer_int8) -> std::optional<fs::path> {
            const std::array<const char*, 2> order = prefer_int8
                ? std::array<const char*, 2>{"vellum-det-yolo11m-seg-1024-int8.onnx", "vellum-det-yolo11m-seg-1024-fp16.onnx"}
                : std::array<const char*, 2>{"vellum-det-yolo11m-seg-1024-fp16.onnx", "vellum-det-yolo11m-seg-1024-int8.onnx"};
            std::vector<fs::path> roots{ctx.model_manager->models_dir()};
            if (ctx.model_manager->fallback_models_dir().has_value()) {
                roots.push_back(*ctx.model_manager->fallback_models_dir());
            }
            std::error_code ec;
            for (const char* fname : order) {
                for (const auto& root : roots) {
                    const fs::path candidate = root / util::utf8_to_wide(fname);
                    if (fs::is_regular_file(candidate, ec) && !ec && fs::file_size(candidate, ec) > 1024ULL * 1024ULL) {
                        return candidate;
                    }
                }
            }
            return std::nullopt;
        };

        std::optional<fs::path> yolo_path;
        if (det_sel_lower.empty() || det_sel_lower == "auto" || det_sel_lower.find("yolo") != std::string::npos) {
            yolo_path = find_yolo_model(is_cpu_det || det_sel_lower.find("int8") != std::string::npos);
        }

        vellum::detector::BackendType target_det_backend = vellum::detector::BackendType::Yolo11Seg;
        std::string resolved_det_model_path;
        if (yolo_path.has_value()) {
            target_det_backend = vellum::detector::BackendType::Yolo11Seg;
            resolved_det_model_path = util::path_to_utf8(*yolo_path);
        } else {
            const auto legacy_det = ctx.model_manager->resolve_detector_model(det_selector, det_opts.allow_model_fallback);
            if (!legacy_det.has_value()) {
                send_json(res, {{"detail", "Requested detector ONNX weights ('" + det_selector + "') are missing. Call POST /api/models/import-koharu or GET /api/models/download first."}}, 503);
                return;
            }
            target_det_backend = (legacy_det->backend == pipeline::DetectorBackend::KoharuRfDetr)
                ? vellum::detector::BackendType::RfDetrSeg
                : vellum::detector::BackendType::LegacyCtd;
            resolved_det_model_path = util::path_to_utf8(legacy_det->model_path);
        }
        if (effective_ocr_model == "manga-ocr") {
            if (!ctx.model_manager->resolve_manga_ocr_models().has_value()) {
                send_json(res, {{"detail", "Japanese manga-ocr ONNX weights are missing. Call POST /api/models/import-koharu or GET /api/models/download first."}}, 503);
                return;
            }
        } else if (effective_ocr_model == "baberu-ocr") {
            if (!ctx.model_manager->resolve_baberu_ocr_models().has_value()) {
                send_json(res, {{"detail", "Baberu OCR ONNX weights are missing in ./models/baberu-ocr/."}}, 503);
                return;
            }
        } else if (effective_ocr_model == "hayai-ocr") {
            if (!ctx.model_manager->resolve_hayai_ocr_models().has_value()) {
                send_json(res, {{"detail", "Hayai OCR ONNX weights are missing in ./models/hayai-ocr/."}}, 503);
                return;
            }
        } else if (effective_ocr_model == "baberu-ocr") {
            if (!ctx.model_manager->resolve_baberu_ocr_models().has_value()) {
                send_json(res, {{"detail", "Baberu OCR ONNX weights are missing in ./models/baberu-ocr/."}}, 503);
                return;
            }
        } else if (effective_ocr_model == "hayai-ocr") {
            if (!ctx.model_manager->resolve_hayai_ocr_models().has_value()) {
                send_json(res, {{"detail", "Hayai OCR ONNX weights are missing in ./models/hayai-ocr/."}}, 503);
                return;
            }
        } else if (effective_ocr_model == "rapidocr-ppocrv4") {
            if (!ctx.model_manager->resolve_rapidocr_paths(src_lang).has_value()) {
                send_json(res, {{"detail", "RapidOCR (" + src_lang + ") PP-OCRv4 ONNX weights are missing. Call GET /api/models/download first."}}, 503);
                return;
            }
        }
        const bool has_page_inpaint_weights =
            (inpaint_opts.inpainting_model == "aot-inpainting" ||
             inpaint_opts.inpainting_model == "flux-2-klein" ||
             inpaint_opts.inpainting_model == "rorem-mixed")
                ? (ctx.model_manager->resolve_aot_model().has_value() || ctx.model_manager->resolve_lama_model().has_value())
                : ctx.model_manager->resolve_lama_model(inpaint_opts.inpainting_model, inpaint_opts.allow_model_fallback).has_value();
        if (!has_page_inpaint_weights) {
            send_json(res, {{"detail", "Requested Inpainting ONNX weights ('" + inpaint_opts.inpainting_model + "') are missing. Call POST /api/models/import-koharu or GET /api/models/download first."}}, 503);
            return;
        }

        std::unique_ptr<pipeline::ComicTranslator> translator;
        try {
            const auto translator_cfg = pipeline::TranslatorConfig::from_settings_with_overrides(
                current_settings, translation_overrides);
            translator = std::make_unique<pipeline::ComicTranslator>(translator_cfg);
        } catch (const std::invalid_argument& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
            return;
        }

        int orig_w = 0;
        int orig_h = 0;
        std::vector<uint8_t> rgb_image;
        try {
            rgb_image = pipeline::LamaInpainter::decode_rgb_from_bytes(
                reinterpret_cast<const uint8_t*>(upload.content.data()),
                upload.content.size(),
                orig_w,
                orig_h);
        } catch (const std::invalid_argument& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
            return;
        }

        vellum::detector::PageDetectionResult v2_det_result;
        std::vector<uint8_t> page_mask;
        std::unordered_map<std::string, std::array<int, 4>> bubble_box_by_id;
        std::unordered_map<std::string, std::array<int, 4>> safe_box_by_id;
        pipeline::OcrStageResult ocr_result;
        pipeline::InpaintResult inpaint_result;
        std::string mask_id;
        std::string mask_url;
        std::string cleaned_image_id;
        std::string cleaned_image_url;
        std::string cleaned_image_payload;

        const BatchKeepAliveHint keep_hint = parse_batch_keep_alive(req);
        const bool want_auto_unload = current_settings.value("auto_unload_vram", true);
        const auto hw_profile = pipeline::ModelManager::query_hardware_profile();
        const bool constrain_memory =
            want_auto_unload &&
            (hw_profile.low_memory_mode || pipeline::ModelManager::is_under_memory_pressure());

        // Adaptive RAM/VRAM policy:
        // - Standard/High-headroom PCs (e.g. >=12 GB RAM & >=6 GB VRAM): keep all 3 Ort::Sessions
        //   cached across pages in a batch and unload VRAM once after the final page.
        // - Low-spec PCs (<10 GB RAM or <5 GB VRAM) or live high RAM pressure (>=86% used / <1.5 GB free
        //   while LM Studio is loaded): unload each stage immediately so peak VRAM stays <600 MB.
        det_opts.auto_unload = constrain_memory;
        ocr_opts.auto_unload = constrain_memory;
        inpaint_opts.auto_unload = constrain_memory;

        pipeline::PageTranslationResult translation_res;
        std::future<pipeline::PageTranslationResult> trans_future;
        bool trans_started = false;

        struct LiveStageGuard {
            AppContext& app;
            ~LiveStageGuard() { app.clear_live_stage(); }
        } live_stage_guard{ctx};

        try {
            ctx.cancel_pending_unload();
            // Stages 1 -> 2 -> 3 serialized under inference_mutex with warm Ort::Session reuse
            {
                std::lock_guard<std::mutex> lock(ctx.inference_mutex);

                // Stage 1: Detector Engine v2 + Adaptive Tiler (YOLO11m-Seg / RF-DETR-Seg / Legacy CTD)
                ctx.set_live_stage("detect", det_selector, 1, 4);
                vellum::detector::DetectorConfig v2_cfg;
                v2_cfg.model_path = resolved_det_model_path;
                v2_cfg.backend = target_det_backend;
                v2_cfg.force_cpu = det_opts.force_cpu;
                v2_cfg.keep_session_alive = !det_opts.auto_unload;
                v2_cfg.enable_adaptive_tiling = current_settings.value("enable_adaptive_tiling", true);
                v2_cfg.sfx_suppress_iou = static_cast<float>(current_settings.value("sfx_suppress_iou", 0.60));
                if (current_settings.contains("whisper_threshold") && current_settings["whisper_threshold"].is_number()) {
                    v2_cfg.class_thresholds[static_cast<size_t>(vellum::detector::ClassId::WhisperBubble)] =
                        current_settings["whisper_threshold"].get<float>();
                }
                if (current_settings.contains("borderless_threshold") && current_settings["borderless_threshold"].is_number()) {
                    v2_cfg.class_thresholds[static_cast<size_t>(vellum::detector::ClassId::BorderlessDialogue)] =
                        current_settings["borderless_threshold"].get<float>();
                }
                if (det_opts.reading_order == "ltr") {
                    v2_cfg.reading_direction = vellum::detector::ReadingDirection::LTR;
                } else if (det_opts.reading_order == "rtl") {
                    v2_cfg.reading_direction = vellum::detector::ReadingDirection::RTL;
                } else {
                    v2_cfg.reading_direction = (src_lang == "ja")
                        ? vellum::detector::ReadingDirection::RTL
                        : vellum::detector::ReadingDirection::LTR;
                }
                if (det_opts.nms_threshold.has_value()) {
                    v2_cfg.nms_iou_threshold = *det_opts.nms_threshold;
                }
                if (det_opts.conf_threshold.has_value()) {
                    v2_cfg.class_thresholds[0] = *det_opts.conf_threshold;
                } else if (det_opts.text_threshold.has_value()) {
                    v2_cfg.class_thresholds[0] = *det_opts.text_threshold;
                }
                if (det_opts.bubble_threshold.has_value()) {
                    v2_cfg.class_thresholds[0] = *det_opts.bubble_threshold;
                }
                if (det_opts.panel_threshold.has_value()) {
                    v2_cfg.class_thresholds[static_cast<size_t>(vellum::detector::ClassId::Panel)] = *det_opts.panel_threshold;
                }

                if (!ctx.detector_v2 || ctx.detector_v2_backend != target_det_backend) {
                    if (ctx.detector_v2) {
                        ctx.detector_v2->unload();
                    }
                    ctx.detector_v2 = vellum::detector::DetectorFactory::create(target_det_backend);
                    ctx.detector_v2_backend = target_det_backend;
                }
                ctx.detector_v2->load(v2_cfg);
                v2_det_result = vellum::detector::AdaptiveTiler::detectPage(
                    *ctx.detector_v2, rgb_image.data(), orig_w, orig_h, upload.filename, v2_cfg);

                // Build tight text-stroke inpainting mask strictly inside each dialogue safe_bbox
                // (never pass full YOLO/RF-DETR bubble segmentation masks into LaMa inpainting).
                page_mask = vellum::detector::buildTextStrokeInpaintMask(
                    rgb_image.data(), orig_w, orig_h, v2_det_result.detections);

                const auto mask_png = pipeline::TextDetector::encode_mask_png(
                    page_mask, orig_w, orig_h);
                std::tie(mask_id, mask_url) = save_mask_png_to_cache(mask_png);

                // Stage 2: Multi-Language OCR Router (`manga-ocr` / `rapidocr-ppocrv4` / `paddleocr-vl-gguf`)
                ctx.set_live_stage("ocr", ocr_opts.ocr_model.empty() ? "manga-ocr" : ocr_opts.ocr_model, 2, 4);
                std::vector<pipeline::OcrBubbleInput> ocr_inputs;
                ocr_inputs.reserve(v2_det_result.detections.size());
                std::vector<std::array<int, 4>> detected_bboxes;
                detected_bboxes.reserve(v2_det_result.detections.size());

                for (const auto& d : v2_det_result.detections) {
                    if (d.class_id == vellum::detector::ClassId::SfxIgnore) {
                        continue;
                    }
                    const std::string bid = "b_" + std::to_string(d.id);
                    const int x1 = static_cast<int>(std::lround(d.bbox.x1));
                    const int y1 = static_cast<int>(std::lround(d.bbox.y1));
                    const int x2 = static_cast<int>(std::lround(d.bbox.x2));
                    const int y2 = static_cast<int>(std::lround(d.bbox.y2));
                    const int sx1 = static_cast<int>(std::lround(d.safe_bbox.x1));
                    const int sy1 = static_cast<int>(std::lround(d.safe_bbox.y1));
                    const int sx2 = static_cast<int>(std::lround(d.safe_bbox.x2));
                    const int sy2 = static_cast<int>(std::lround(d.safe_bbox.y2));

                    pipeline::OcrBubbleInput item;
                    item.id = bid;
                    item.order = d.reading_order;
                    item.bbox = {x1, y1, x2, y2};
                    for (const auto& pt : d.mask.polygon) {
                        item.polygon.push_back({static_cast<int>(std::lround(pt.x)), static_cast<int>(std::lround(pt.y))});
                    }
                    if (item.polygon.empty()) {
                        item.polygon = {
                            {x1, y1}, {x2, y1}, {x2, y2}, {x1, y2}
                        };
                    }
                    item.confidence = d.confidence;
                    item.text_direction =
                        (v2_det_result.reading_direction == vellum::detector::ReadingDirection::RTL &&
                         (y2 - y1) >= (x2 - x1))
                            ? "vertical"
                            : "horizontal";
                    bubble_box_by_id[bid] = item.bbox;
                    safe_box_by_id[bid] = {sx1, sy1, sx2, sy2};
                    detected_bboxes.push_back(item.bbox);
                    ocr_inputs.push_back(std::move(item));
                }

                ocr_result = ctx.ocr_engine->recognize_rgb(
                    rgb_image.data(), orig_w, orig_h, ocr_inputs, ocr_opts);

                // Launch Stage 4 (LLM Translation) asynchronously in parallel with Stage 3 (LaMa Inpainting)
                // since translation only depends on OCR text (plus optional compressed vision JPEG).
                const std::string active_llm_name = !translator->config().preferred_model.empty()
                    ? translator->config().preferred_model
                    : translator->config().provider;
                ctx.set_live_stage("inpaint_translate", inpaint_opts.inpainting_model + " + " + active_llm_name, 3, 4);
                std::vector<pipeline::DialogueBlockInput> async_dialogues;
                async_dialogues.reserve(ocr_result.blocks.size());
                for (const auto& b : ocr_result.blocks) {
                    async_dialogues.push_back({b.id, b.source_text});
                }
                std::optional<std::string> async_vision_uri;
                if (translator->config().use_vision) {
                    try {
                        async_vision_uri = pipeline::ComicTranslator::encode_rgb_for_vision(
                            rgb_image.data(),
                            orig_w,
                            orig_h,
                            translator->config().vision_max_side);
                    } catch (...) {
                        async_vision_uri = std::nullopt;
                    }
                }
                pipeline::TranslateRequestOptions async_req_opts;
                async_req_opts.source_lang = src_lang;
                async_req_opts.target_lang = tgt_lang;
                async_req_opts.page_image_data_uri = std::move(async_vision_uri);
                async_req_opts.use_vision = translator->config().use_vision;

                trans_future = std::async(
                    std::launch::async,
                    [&translator, dialogues = std::move(async_dialogues), req_opts = std::move(async_req_opts)]() {
                        return translator->translate_blocks(dialogues, req_opts);
                    });
                trans_started = true;

                // Stage 3: Tile-Based LaMa Inpainting + Solid-Bubble Clean Erase
                inpaint_opts.bboxes = detected_bboxes;
                inpaint_opts.refine_mask = false;
                inpaint_result = ctx.inpainter->inpaint_rgb(
                    rgb_image.data(), orig_w, orig_h, page_mask, inpaint_opts);
                ctx.set_live_stage("translate", active_llm_name, 4, 4);

                // Unload VRAM immediately once after the final page of a batch (or under memory pressure)
                if (want_auto_unload &&
                    (constrain_memory ||
                     keep_hint.unload_after ||
                     (keep_hint.keep_alive.has_value() && !*keep_hint.keep_alive))) {
                    if (ctx.detector) ctx.detector->unload();
                    if (ctx.detector_v2) ctx.detector_v2->unload();
                    if (ctx.ocr_engine) ctx.ocr_engine->unload();
                    if (ctx.inpainter) ctx.inpainter->unload();
                    pipeline::ModelManager::trim_process_heap();
                }
            }

            if (want_auto_unload && !constrain_memory) {
                if (keep_hint.keep_alive.has_value() && *keep_hint.keep_alive) {
                    // Safety fallback: unload VRAM if batch client disconnects mid-batch
                    ctx.schedule_deferred_unload(60);
                } else if (!keep_hint.unload_after && !keep_hint.keep_alive.has_value()) {
                    // Automatic batch keep-alive: keep Ort::Sessions hot for back-to-back pages,
                    // then unload once after 15s of inactivity when the batch finishes.
                    ctx.schedule_deferred_unload(15);
                }
            }

            const auto cleaned_png = pipeline::LamaInpainter::encode_rgb_png(
                inpaint_result.rgb, inpaint_result.width, inpaint_result.height);
            std::tie(cleaned_image_id, cleaned_image_url) = save_inpainted_png_to_cache(cleaned_png);
            cleaned_image_payload = include_base64
                ? ("data:image/png;base64," +
                   pipeline::TextDetector::base64_encode(cleaned_png.data(), cleaned_png.size()))
                : cleaned_image_url;

            // Free large per-page uncompressed image/mask buffers before waiting on Stage 4 LLM HTTP
            page_mask.clear();
            page_mask.shrink_to_fit();
            inpaint_result.rgb.clear();
            inpaint_result.rgb.shrink_to_fit();
            rgb_image.clear();
            rgb_image.shrink_to_fit();
        } catch (const std::invalid_argument& ex) {
            if (trans_started && trans_future.valid()) {
                try { trans_future.wait(); } catch (...) {}
            }
            send_json(res, {{"detail", ex.what()}}, 400);
            return;
        } catch (const std::exception& ex) {
            if (trans_started && trans_future.valid()) {
                try { trans_future.wait(); } catch (...) {}
            }
            send_json(res, {{"detail", std::string("Page CV/OCR/Inpaint pipeline failed: ") + ex.what()}}, 500);
            return;
        }

        // Collect Stage 4 LLM Translation result (already running in parallel with Stage 3)
        try {
            translation_res = trans_future.get();
        } catch (const pipeline::ProviderConnectionError& ex) {
            send_json(res, {{"detail", ex.what()}}, 503);
            return;
        } catch (const pipeline::ProviderTranslationError& ex) {
            send_json(res, {{"detail", ex.what()}}, 502);
            return;
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", std::string("Translation failed: ") + ex.what()}}, 500);
            return;
        }

        json processed_blocks = json::array();
        for (const auto& b : ocr_result.blocks) {
            const auto it = translation_res.translations.find(b.id);
            const std::string trans_text =
                (it != translation_res.translations.end()) ? it->second : b.source_text;
            const std::string sizing_text = trans_text.empty() ? b.source_text : trans_text;

            std::array<int, 4> bubble_bb = b.bbox;
            std::array<int, 4> safe_bb = b.bbox;
            if (const auto bit = bubble_box_by_id.find(b.id); bit != bubble_box_by_id.end()) {
                bubble_bb = bit->second;
            }
            if (const auto sit = safe_box_by_id.find(b.id); sit != safe_box_by_id.end()) {
                safe_bb = sit->second;
            }
            const int fsize = estimate_font_size(safe_bb, sizing_text);

            json poly_json = json::array();
            for (const auto& pt : b.polygon) {
                poly_json.push_back(json::array({pt[0], pt[1]}));
            }
            if (poly_json.empty()) {
                poly_json = bbox_to_polygon_json(b.bbox);
            }

            processed_blocks.push_back({
                {"id", b.id},
                {"order", b.order},
                {"bbox", json::array({b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3]})},
                {"bubble_bbox", json::array({bubble_bb[0], bubble_bb[1], bubble_bb[2], bubble_bb[3]})},
                {"safe_bbox", json::array({safe_bb[0], safe_bb[1], safe_bb[2], safe_bb[3]})},
                {"polygon", std::move(poly_json)},
                {"original_text", b.source_text},
                {"source_text", b.source_text},
                {"translated_text", trans_text},
                {"font_size", fsize},
                {"color", "#000000"},
                {"align", "center"},
                {"confidence", b.confidence},
                {"text_direction", b.text_direction},
                {"style", build_default_style_json(fsize, "#000000", "center")},
            });
        }

        const nlohmann::json v2_schema = vellum::detector::toSchemaV2Json(v2_det_result);
        json layout_bubbles_json = json::array();
        for (const auto& d : v2_det_result.detections) {
            if (vellum::detector::is_dialogue_class(d.class_id)) {
                layout_bubbles_json.push_back({
                    {"id", "lb_" + std::to_string(d.id)},
                    {"order", d.reading_order},
                    {"bbox", {static_cast<int>(std::lround(d.bbox.x1)),
                              static_cast<int>(std::lround(d.bbox.y1)),
                              static_cast<int>(std::lround(d.bbox.x2)),
                              static_cast<int>(std::lround(d.bbox.y2))}},
                    {"confidence", d.confidence},
                    {"category", "bubble"},
                });
            }
        }
        json panels_json = json::array();
        for (const auto& p : v2_det_result.panels) {
            panels_json.push_back({
                {"id", "p_" + std::to_string(p.id)},
                {"panel_id", p.id},
                {"order", p.reading_order},
                {"reading_order", p.reading_order},
                {"bbox", {static_cast<int>(std::lround(p.bbox.x1)),
                          static_cast<int>(std::lround(p.bbox.y1)),
                          static_cast<int>(std::lround(p.bbox.x2)),
                          static_cast<int>(std::lround(p.bbox.y2))}},
                {"bbox_norm", {p.bbox_norm.x1, p.bbox_norm.y1, p.bbox_norm.x2, p.bbox_norm.y2}},
                {"confidence", p.confidence},
                {"category", "panel"},
            });
        }

        const std::string page_id = extract_page_id_from_filename(upload.filename);
        const std::string active_exec_mode = current_settings.value("execution_mode", std::string("hybrid"));
        json combined_runtime_meta = v2_schema["runtime_meta"];
        combined_runtime_meta["execution_mode"] = active_exec_mode;
        combined_runtime_meta["detector_provider"] = v2_det_result.runtime_meta.execution_provider;
        combined_runtime_meta["ocr_provider"] = ocr_result.execution_provider;
        combined_runtime_meta["inpainter_provider"] = inpaint_result.execution_provider;
        combined_runtime_meta["detector_model"] = v2_det_result.runtime_meta.model_name;
        combined_runtime_meta["ocr_model"] = ocr_result.ocr_model;
        combined_runtime_meta["inpainting_model"] = inpaint_result.model_id;

        send_json(res, {
            {"schema_version", "2.0"},
            {"page_id", page_id},
            {"dimensions", {{"width", orig_w}, {"height", orig_h}}},
            {"source_lang", src_lang},
            {"target_lang", tgt_lang},
            {"source_language", src_lang},
            {"target_language", tgt_lang},
            {"reading_order", v2_det_result.reading_direction == vellum::detector::ReadingDirection::RTL ? "rtl" : "ltr"},
            {"execution_mode", active_exec_mode},
            {"execution_provider", v2_det_result.runtime_meta.execution_provider},
            {"detector_provider", v2_det_result.runtime_meta.execution_provider},
            {"ocr_provider", ocr_result.execution_provider},
            {"inpainter_provider", inpaint_result.execution_provider},
            {"runtime_meta", std::move(combined_runtime_meta)},
            {"detector_backend", v2_schema["runtime_meta"]["backend"]},
            {"detector_model", v2_det_result.runtime_meta.model_name},
            {"tiles_detected", v2_det_result.runtime_meta.tiles_processed},
            {"provider", translation_res.provider},
            {"model", translation_res.model},
            {"vision_used", translation_res.vision_used},
            {"reasoning_effort", translation_res.reasoning_effort},
            {"ocr_engine", ocr_result.engine},
            {"ocr_model", ocr_result.ocr_model},
            {"inpainting_model", inpaint_result.model_id},
            {"tiles_inpainted", inpaint_result.tiles_processed},
            {"cleaned_image_id", cleaned_image_id},
            {"cleaned_image_url", cleaned_image_url},
            {"inpainted_image_url", cleaned_image_url},
            {"cleaned_image", cleaned_image_payload},
            {"inpainted_image", cleaned_image_payload},
            {"mask_id", mask_id},
            {"mask_url", mask_url},
            {"layout_bubbles", std::move(layout_bubbles_json)},
            {"panels", std::move(panels_json)},
            {"blocks", processed_blocks},
            {"bubbles", std::move(processed_blocks)},
        });
    };
    server.Post("/api/process-page", handle_process_page);
    server.Post("/api/pipeline/process-page", handle_process_page);

    server.Get("/api/lmstudio/status", [&ctx](const httplib::Request&, httplib::Response& res) {
        send_json(res, probe_lm_studio(ctx.settings_manager, 5.0));
    });

    server.Get("/api/providers", [&ctx](const httplib::Request&, httplib::Response& res) {
        const json current = ctx.settings_manager.load();
        send_json(res, {
            {"active_provider", current.value("provider", std::string("lmstudio"))},
            {"providers", SettingsManager::list_provider_presets()},
            {"supported_reasoning_efforts", SettingsManager::list_reasoning_efforts()},
            {"target_languages", SettingsManager::list_target_languages()},
        });
    });

    server.Post("/api/providers/test", [&ctx](const httplib::Request& req, httplib::Response& res) {
        json body = json::object();
        if (!util::trim_ascii(req.body).empty()) {
            body = json::parse(req.body, nullptr, false);
            if (body.is_discarded() || !body.is_object()) {
                send_json(res, {{"detail", "Invalid JSON object in /api/providers/test request body."}}, 400);
                return;
            }
        }

        const json settings = ctx.settings_manager.load();
        std::string prov_id;
        try {
            if (body.contains("provider") && body["provider"].is_string() &&
                !util::trim_ascii(body["provider"].get<std::string>()).empty()) {
                prov_id = pipeline::ComicTranslator::normalize_provider_id(
                    body["provider"].get<std::string>());
            } else {
                prov_id = pipeline::ComicTranslator::normalize_provider_id(
                    settings.value("provider", std::string("lmstudio")));
            }
        } catch (const std::invalid_argument& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
            return;
        }

        const auto& preset = pipeline::ComicTranslator::get_provider_spec(prov_id);
        const json* saved_profile = nullptr;
        if (settings.contains("providers") && settings["providers"].is_object() &&
            settings["providers"].contains(prov_id) && settings["providers"][prov_id].is_object()) {
            saved_profile = &settings["providers"][prov_id];
        }

        const std::string active_provider = settings.value("provider", std::string("lmstudio"));

        std::string base_url;
        if (body.contains("base_url") && body["base_url"].is_string() &&
            !util::trim_ascii(body["base_url"].get<std::string>()).empty()) {
            base_url = util::trim_ascii(body["base_url"].get<std::string>());
        } else if (prov_id == active_provider &&
                   settings.contains("base_url") && settings["base_url"].is_string() &&
                   !util::trim_ascii(settings["base_url"].get<std::string>()).empty()) {
            base_url = util::trim_ascii(settings["base_url"].get<std::string>());
        } else if (saved_profile && saved_profile->contains("base_url") &&
                   (*saved_profile)["base_url"].is_string() &&
                   !util::trim_ascii((*saved_profile)["base_url"].get<std::string>()).empty()) {
            base_url = util::trim_ascii((*saved_profile)["base_url"].get<std::string>());
        } else {
            base_url = preset.default_base_url;
        }

        std::string api_key;
        if (body.contains("api_key") && body["api_key"].is_string()) {
            api_key = util::trim_ascii(body["api_key"].get<std::string>());
        } else if (prov_id == active_provider &&
                   settings.contains("api_key") && settings["api_key"].is_string()) {
            api_key = util::trim_ascii(settings["api_key"].get<std::string>());
        } else if (saved_profile && saved_profile->contains("api_key") &&
                   (*saved_profile)["api_key"].is_string()) {
            api_key = util::trim_ascii((*saved_profile)["api_key"].get<std::string>());
        } else {
            api_key = preset.default_api_key;
        }

        std::string preferred_model;
        if (body.contains("model") && body["model"].is_string() &&
            !util::trim_ascii(body["model"].get<std::string>()).empty()) {
            preferred_model = util::trim_ascii(body["model"].get<std::string>());
        } else if (prov_id == active_provider &&
                   settings.contains("model") && settings["model"].is_string() &&
                   !util::trim_ascii(settings["model"].get<std::string>()).empty()) {
            preferred_model = util::trim_ascii(settings["model"].get<std::string>());
        } else if (saved_profile && saved_profile->contains("model") &&
                   (*saved_profile)["model"].is_string() &&
                   !util::trim_ascii((*saved_profile)["model"].get<std::string>()).empty()) {
            preferred_model = util::trim_ascii((*saved_profile)["model"].get<std::string>());
        } else {
            preferred_model = preset.default_model;
        }

        double timeout_sec = 10.0;
        if (body.contains("timeout") && body["timeout"].is_number()) {
            timeout_sec = std::max(1.0, std::min(60.0, body["timeout"].get<double>()));
        }

        const auto test_res = pipeline::ComicTranslator::test_provider_connection(
            prov_id, base_url, api_key, preferred_model, timeout_sec);
        send_json(res, test_res.to_json());
    });

    server.Get("/api/languages", [](const httplib::Request&, httplib::Response& res) {
        send_json(res, {
            {"source_languages", SettingsManager::list_source_languages()},
            {"target_languages", SettingsManager::list_target_languages()},
        });
    });

    // -------------------------------------------------------------------------
    // Persistent Settings (`GET /api/settings`, `POST /api/settings`)
    // -------------------------------------------------------------------------

    server.Get("/api/settings", [&ctx](const httplib::Request&, httplib::Response& res) {
        const json current = ctx.settings_manager.load();
        send_json(res, ctx.settings_manager.format_api_response(current));
    });

    server.Post("/api/settings", [&ctx](const httplib::Request& req, httplib::Response& res) {
        const json patch = json::parse(req.body, nullptr, false);
        if (patch.is_discarded() || !patch.is_object()) {
            send_json(res, {{"detail", "Invalid JSON object in settings request body."}}, 400);
            return;
        }
        try {
            const json updated = ctx.settings_manager.update(patch);
            if (updated.value("auto_unload_vram", true)) {
                std::lock_guard<std::mutex> lock(ctx.inference_mutex);
                if (ctx.detector) ctx.detector->unload();
                if (ctx.ocr_engine) ctx.ocr_engine->unload();
                if (ctx.inpainter) ctx.inpainter->unload();
            }
            send_json(res, ctx.settings_manager.format_api_response(updated));
        } catch (const std::invalid_argument& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", ex.what()}}, 500);
        }
    });

    // -------------------------------------------------------------------------
    // Fonts Management (`/api/fonts/*`)
    // -------------------------------------------------------------------------

    server.Get("/api/fonts/list", [&ctx](const httplib::Request& req, httplib::Response& res) {
        std::optional<std::string> category;
        if (req.has_param("category")) {
            const std::string cat = util::to_lower_ascii(util::trim_ascii(req.get_param_value("category")));
            if (!cat.empty()) {
                if (cat == "system" || cat == "custom" || cat == "downloaded") {
                    category = cat;
                } else {
                    send_json(
                        res,
                        {{"detail", "Invalid font category '" + cat + "'. Expected 'system', 'custom', or 'downloaded'."}},
                        400);
                    return;
                }
            }
        }
        const bool include_catalog = req.has_param("include_catalog")
            ? parse_bool_param(req.get_param_value("include_catalog"), true)
            : true;
        const bool refresh = req.has_param("refresh")
            ? parse_bool_param(req.get_param_value("refresh"), false)
            : false;

        send_json(res, ctx.font_manager.list_fonts(category, include_catalog, refresh));
    });

    server.Get(R"(/api/fonts/file/(.+))", [&ctx](const httplib::Request& req, httplib::Response& res) {
        const std::string font_name = req.matches[1].str();
        try {
            const auto resolved = ctx.font_manager.resolve_font_file(font_name);
            const auto bytes = util::read_file_bytes(resolved.path);
            if (!bytes.has_value()) {
                send_json(res, {{"detail", "Failed to read font file from disk."}}, 500);
                return;
            }
            apply_cors_headers(res);
            res.set_header("Cache-Control", "public, max-age=86400");
            res.set_header("Content-Disposition", "inline; filename=\"" + resolved.filename + "\"");
            res.set_content(
                reinterpret_cast<const char*>(bytes->data()),
                bytes->size(),
                resolved.media_type);
        } catch (const FontValidationError& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        } catch (const FontNotFoundError& ex) {
            send_json(res, {{"detail", ex.what()}}, 404);
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", ex.what()}}, 500);
        }
    });

    server.Post("/api/fonts/upload", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (!req.is_multipart_form_data()) {
            send_json(res, {{"detail", "Expected multipart/form-data upload with field 'file' or 'font'."}}, 400);
            return;
        }
        httplib::FormData upload;
        if (req.form.has_file("file")) {
            upload = req.form.get_file("file");
        } else if (req.form.has_file("font")) {
            upload = req.form.get_file("font");
        } else {
            send_json(res, {{"detail", "Missing uploaded font file. Provide multipart form field 'file' or 'font'."}}, 400);
            return;
        }

        if (util::trim_ascii(upload.filename).empty()) {
            send_json(res, {{"detail", "Uploaded font file must include a filename (e.g. 'CCWildWords.ttf')."}}, 400);
            return;
        }
        if (upload.content.empty()) {
            send_json(res, {{"detail", "Uploaded font file is empty."}}, 400);
            return;
        }

        std::optional<std::string> family;
        if (req.form.has_field("family")) {
            const std::string f = util::trim_ascii(req.form.get_field("family"));
            if (!f.empty()) family = f;
        }
        bool overwrite = true;
        if (req.form.has_field("overwrite")) {
            overwrite = parse_bool_param(req.form.get_field("overwrite"), true);
        }

        try {
            std::vector<uint8_t> raw_bytes(upload.content.begin(), upload.content.end());
            const json entry = ctx.font_manager.save_uploaded_font(upload.filename, raw_bytes, family, overwrite);
            send_json(res, {
                {"status", "ok"},
                {"filename", entry["filename"]},
                {"family", entry["family"]},
                {"category", entry["category"]},
                {"url", entry["url"]},
                {"font", entry},
            });
        } catch (const FontValidationError& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", ex.what()}}, 500);
        }
    });

    server.Post("/api/fonts/download", [&ctx](const httplib::Request& req, httplib::Response& res) {
        const json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.is_object()) {
            send_json(res, {{"detail", "Invalid JSON body for /api/fonts/download."}}, 400);
            return;
        }

        std::vector<std::string> requested;
        std::unordered_set<std::string> seen;
        auto push_unique = [&](const std::string& s) {
            const std::string cleaned = util::trim_ascii(s);
            if (!cleaned.empty() && seen.insert(cleaned).second) {
                requested.push_back(cleaned);
            }
        };

        if (body.contains("fonts") && body["fonts"].is_array()) {
            for (const auto& item : body["fonts"]) {
                if (item.is_string()) push_unique(item.get<std::string>());
            }
        }
        if (body.contains("font") && body["font"].is_string()) {
            push_unique(body["font"].get<std::string>());
        }
        if (body.contains("family") && body["family"].is_string()) {
            push_unique(body["family"].get<std::string>());
        }

        if (requested.empty()) {
            send_json(
                res,
                {{"detail", "Provide at least one font name or catalog ID in 'fonts', 'font', or 'family'."}},
                400);
            return;
        }

        std::optional<std::string> weight;
        if (body.contains("weight") && body["weight"].is_string() && !util::trim_ascii(body["weight"].get<std::string>()).empty()) {
            weight = util::trim_ascii(body["weight"].get<std::string>());
        }
        const bool overwrite = body.contains("overwrite") && body["overwrite"].is_boolean()
            ? body["overwrite"].get<bool>()
            : false;

        try {
            send_json(res, ctx.font_manager.download_catalog_fonts(requested, weight, overwrite));
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", ex.what()}}, 500);
        }
    });

    // -------------------------------------------------------------------------
    // Cached Images (`/api/cache/images/{image_id}.{ext}` and `/api/cache/images/{image_id}`)
    // -------------------------------------------------------------------------

    auto serve_cached_image_ref = [](const std::string& ref, httplib::Response& res) {
        try {
            const fs::path path = resolve_cached_image_path(ref);
            const auto bytes = util::read_file_bytes(path);
            if (!bytes.has_value()) {
                send_json(res, {{"detail", "Failed to read cached image."}}, 500);
                return;
            }
            std::string ext = util::to_lower_ascii(util::path_to_utf8(path.extension()));
            if (!ext.empty() && ext.front() == '.') {
                ext.erase(ext.begin());
            }
            const std::string fname = util::path_to_utf8(path.filename());
            apply_cors_headers(res);
            res.set_header("Cache-Control", "public, max-age=3600");
            res.set_header("Content-Disposition", "inline; filename=\"" + fname + "\"");
            res.set_content(
                reinterpret_cast<const char*>(bytes->data()),
                bytes->size(),
                media_type_for_cache_ext(ext));
        } catch (const ComicExportError& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", ex.what()}}, 404);
        }
    };

    server.Get(R"(/api/cache/images/([A-Za-z0-9_-]{1,128})\.([A-Za-z0-9]+))",
        [serve_cached_image_ref](const httplib::Request& req, httplib::Response& res) {
            serve_cached_image_ref(req.matches[1].str() + "." + util::to_lower_ascii(req.matches[2].str()), res);
        });

    server.Get(R"(/api/cache/images/([A-Za-z0-9_-]{1,128}))",
        [serve_cached_image_ref](const httplib::Request& req, httplib::Response& res) {
            serve_cached_image_ref(req.matches[1].str(), res);
        });

    // Pin project images into `./projects_cache/<project_id>/` (exempt from 50,000-file LRU pruning)
    auto handle_pin_project_cache = [](const httplib::Request& req, httplib::Response& res) {
        const json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.is_object()) {
            send_json(res, {{"detail", "Invalid JSON object for /api/projects_cache/pin."}}, 400);
            return;
        }
        const std::string raw_pid = util::trim_ascii(body.value("project_id", std::string("")));
        if (raw_pid.empty() || raw_pid.size() > 64) {
            send_json(res, {{"detail", "Valid 'project_id' (1-64 chars) is required."}}, 400);
            return;
        }
        for (unsigned char c : raw_pid) {
            if (!std::isalnum(c) && c != '_' && c != '-') {
                send_json(res, {{"detail", "Invalid characters in 'project_id'."}}, 400);
                return;
            }
        }
        if (!body.contains("image_urls") || !body["image_urls"].is_array()) {
            send_json(res, {{"detail", "Provide 'image_urls' array."}}, 400);
            return;
        }

        const fs::path proj_dir = get_projects_cache_dir() / util::utf8_to_wide(raw_pid);
        std::error_code ec;
        fs::create_directories(proj_dir, ec);
        if (ec) {
            send_json(res, {{"detail", "Failed to create project cache directory."}}, 500);
            return;
        }

        json pinned_urls = json::object();
        size_t pinned_count = 0;
        for (const auto& item : body["image_urls"]) {
            if (!item.is_string()) continue;
            const std::string ref = util::trim_ascii(item.get<std::string>());
            if (ref.empty()) continue;
            if (ref.find("/api/projects_cache/") != std::string::npos) {
                pinned_urls[ref] = ref;
                ++pinned_count;
                continue;
            }
            try {
                const fs::path src = resolve_cached_image_path(ref);
                const std::string fname = util::path_to_utf8(src.filename());
                const fs::path dst = proj_dir / src.filename();
                if (!fs::exists(dst, ec)) {
                    if (!CreateHardLinkW(dst.c_str(), src.c_str(), nullptr)) {
                        CopyFileW(src.c_str(), dst.c_str(), FALSE);
                    }
                }
                if (fs::is_regular_file(dst, ec)) {
                    pinned_urls[ref] = "/api/projects_cache/" + raw_pid + "/" + fname;
                    ++pinned_count;
                }
            } catch (...) {
                // Skip unresolvable references without failing the batch
            }
        }

        send_json(res, {
            {"status", "ok"},
            {"project_id", raw_pid},
            {"pinned_count", pinned_count},
            {"pinned_urls", std::move(pinned_urls)},
        });
    };
    server.Post("/api/cache/pin", handle_pin_project_cache);
    server.Post("/api/projects_cache/pin", handle_pin_project_cache);

    // Wipe unpinned `./cache/images/` files on user demand
    server.Post("/api/cache/clear", [](const httplib::Request&, httplib::Response& res) {
        const fs::path cache_dir = get_image_cache_dir();
        std::error_code ec;
        size_t deleted_files = 0;
        std::uintmax_t freed_bytes = 0;
        if (fs::is_directory(cache_dir, ec)) {
            for (const auto& entry : fs::directory_iterator(cache_dir, ec)) {
                if (ec) break;
                std::error_code f_ec;
                if (!entry.is_regular_file(f_ec) || f_ec) continue;
                const std::string fname = util::path_to_utf8(entry.path().filename());
                if (fname == ".gitkeep") continue;
                const auto fsize = entry.file_size(f_ec);
                if (!f_ec) {
                    freed_bytes += fsize;
                }
                if (fs::remove(entry.path(), f_ec) && !f_ec) {
                    ++deleted_files;
                }
            }
        }
        send_json(res, {
            {"status", "ok"},
            {"deleted_files", deleted_files},
            {"freed_bytes", freed_bytes},
        });
    });

    // Disk-backed crash-recovery autosave (`./projects_cache/autosave.vellum.json`)
    server.Post("/api/projects_cache/autosave", [](const httplib::Request& req, httplib::Response& res) {
        if (util::trim_ascii(req.body).empty()) {
            send_json(res, {{"detail", "Empty autosave request body."}}, 400);
            return;
        }
        const json parsed = json::parse(req.body, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object()) {
            send_json(res, {{"detail", "Invalid JSON object for /api/projects_cache/autosave."}}, 400);
            return;
        }
        const fs::path autosave_path = get_projects_cache_dir() / L"autosave.vellum.json";
        std::string write_err;
        if (!util::atomic_write_bytes(
                autosave_path,
                reinterpret_cast<const uint8_t*>(req.body.data()),
                req.body.size(),
                &write_err)) {
            send_json(res, {{"detail", write_err.empty() ? "Failed to write autosave file." : write_err}}, 500);
            return;
        }
        const size_t page_count = (parsed.contains("pages") && parsed["pages"].is_array())
            ? parsed["pages"].size()
            : 0;
        send_json(res, {
            {"status", "ok"},
            {"saved_bytes", req.body.size()},
            {"page_count", page_count},
        });
    });

    server.Get("/api/projects_cache/autosave", [](const httplib::Request&, httplib::Response& res) {
        const fs::path autosave_path = get_projects_cache_dir() / L"autosave.vellum.json";
        std::error_code ec;
        if (!fs::is_regular_file(autosave_path, ec)) {
            send_json(res, {{"status", "empty"}, {"pages", json::array()}}, 200);
            return;
        }
        const auto bytes = util::read_file_bytes(autosave_path);
        if (!bytes.has_value() || bytes->empty()) {
            send_json(res, {{"status", "empty"}, {"pages", json::array()}}, 200);
            return;
        }
        apply_cors_headers(res);
        res.status = 200;
        res.set_content(
            reinterpret_cast<const char*>(bytes->data()),
            bytes->size(),
            "application/json; charset=utf-8");
    });

    server.Delete("/api/projects_cache/autosave", [](const httplib::Request&, httplib::Response& res) {
        const fs::path autosave_path = get_projects_cache_dir() / L"autosave.vellum.json";
        std::error_code ec;
        const bool removed = fs::remove(autosave_path, ec);
        send_json(res, {{"status", "ok"}, {"removed", removed}});
    });

    server.Get(R"(/api/projects_cache/([A-Za-z0-9_-]{1,64})/([A-Za-z0-9_.-]{1,160}))",
        [](const httplib::Request& req, httplib::Response& res) {
            try {
                const fs::path path = resolve_pinned_project_image_path(
                    req.matches[1].str(), req.matches[2].str());
                const auto bytes = util::read_file_bytes(path);
                if (!bytes.has_value()) {
                    send_json(res, {{"detail", "Failed to read pinned project image."}}, 500);
                    return;
                }
                std::string ext = util::to_lower_ascii(util::path_to_utf8(path.extension()));
                if (!ext.empty() && ext.front() == '.') {
                    ext.erase(ext.begin());
                }
                apply_cors_headers(res);
                res.set_header("Cache-Control", "public, max-age=86400");
                res.set_header("Content-Disposition", "inline; filename=\"" + req.matches[2].str() + "\"");
                res.set_content(
                    reinterpret_cast<const char*>(bytes->data()),
                    bytes->size(),
                    media_type_for_cache_ext(ext));
            } catch (const ComicExportError& ex) {
                send_json(res, {{"detail", ex.what()}}, 400);
            } catch (const std::exception& ex) {
                send_json(res, {{"detail", ex.what()}}, 404);
            }
        });

    server.Post(
        "/api/cache/upload",
        [&](const httplib::Request& req, httplib::Response& res) {
            if (!req.is_multipart_form_data()) {
                send_json(res, {{"detail", "Expected multipart/form-data upload with field 'image' or 'file'."}}, 400);
                return;
            }
            httplib::FormData upload;
            if (req.form.has_file("image")) {
                upload = req.form.get_file("image");
            } else if (req.form.has_file("file")) {
                upload = req.form.get_file("file");
            } else {
                send_json(res, {{"detail", "Missing uploaded image file. Provide multipart form field 'image' or 'file'."}}, 400);
                return;
            }
            if (upload.content.empty()) {
                send_json(res, {{"detail", "Uploaded image file is empty."}}, 400);
                return;
            }

            std::string ext = util::to_lower_ascii(fs::path(upload.filename).extension().string());
            if (ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".webp" && ext != ".bmp") {
                ext = ".png";
            }
            const std::string image_id = "imp_" + util::generate_uuid_hex() + ext;
            const fs::path cache_dir = get_image_cache_dir();
            const fs::path dest_path = cache_dir / image_id;
            std::string write_err;
            if (!util::atomic_write_bytes(dest_path, upload.content.data(), upload.content.size(), &write_err)) {
                send_json(res, {{"detail", write_err.empty() ? "Failed to cache uploaded image." : write_err}}, 500);
                return;
            }
            prune_image_cache(cache_dir);
            send_json(res, {
                {"image_id", image_id},
                {"image_url", "/api/cache/images/" + image_id},
            });
        });

    // -------------------------------------------------------------------------
    // Archive Import & Multi-Page Export (`/api/import/archive`, `/api/export/cbz`, `/api/export/pdf`)
    // -------------------------------------------------------------------------

    server.Post("/api/import/archive", [](const httplib::Request& req, httplib::Response& res) {
        if (!req.is_multipart_form_data()) {
            send_json(res, {{"detail", "Expected multipart/form-data with field 'archive' or 'file'."}}, 400);
            return;
        }
        httplib::FormData upload;
        if (req.form.has_file("archive")) {
            upload = req.form.get_file("archive");
        } else if (req.form.has_file("file")) {
            upload = req.form.get_file("file");
        } else {
            send_json(res, {{"detail", "Missing uploaded archive file. Provide multipart form field 'archive' or 'file'."}}, 400);
            return;
        }

        std::string archive_name = upload.filename.empty() ? "comic.cbz" : upload.filename;
        std::replace(archive_name.begin(), archive_name.end(), '\\', '/');
        if (const auto slash = archive_name.rfind('/'); slash != std::string::npos) {
            archive_name = archive_name.substr(slash + 1);
        }
        const std::string suffix = util::to_lower_ascii(
            util::path_to_utf8(fs::path(util::utf8_to_wide(archive_name)).extension()));
        if (!suffix.empty() && suffix != ".cbz" && suffix != ".zip") {
            send_json(
                res,
                {{"detail", "Unsupported archive format '" + suffix + "'. Upload a .cbz or .zip archive."}},
                400);
            return;
        }
        if (upload.content.empty()) {
            send_json(res, {{"detail", "Uploaded archive file is empty."}}, 400);
            return;
        }

        try {
            const fs::path cache_dir = get_image_cache_dir();
            std::error_code dir_ec;
            fs::create_directories(cache_dir, dir_ec);
            const fs::path tmp_archive = cache_dir / util::utf8_to_wide("tmp_upload_" + util::generate_uuid_hex() + ".cbz");
            std::string write_err;
            if (!util::atomic_write_bytes(
                    tmp_archive,
                    reinterpret_cast<const uint8_t*>(upload.content.data()),
                    upload.content.size(),
                    &write_err)) {
                send_json(res, {{"detail", "Failed to stage uploaded archive on disk: " + write_err}}, 500);
                return;
            }
            struct TempUploadGuard {
                fs::path p;
                ~TempUploadGuard() {
                    std::error_code ec;
                    fs::remove(p, ec);
                }
            } tmp_guard{tmp_archive};

            const auto extracted = extract_archive_pages_from_file(tmp_archive, cache_dir);

            json pages_json = json::array();
            for (const auto& p : extracted) {
                pages_json.push_back({
                    {"index", p.index},
                    {"filename", p.filename},
                    {"archive_path", p.archive_path},
                    {"image_id", p.image_id},
                    {"image_url", "/api/cache/images/" + p.image_id + "." + p.extension},
                    {"dimensions", {{"width", p.dimensions.width}, {"height", p.dimensions.height}}},
                    {"size_bytes", p.size_bytes},
                });
            }

            send_json(res, {
                {"archive_name", archive_name},
                {"total_pages", extracted.size()},
                {"pages", std::move(pages_json)},
            });
        } catch (const ArchiveImportError& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", std::string("Archive import failed: ") + ex.what()}}, 500);
        }
    });

    server.Post("/api/import/archive-path", [](const httplib::Request& req, httplib::Response& res) {
        const json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.is_object()) {
            send_json(res, {{"detail", "Invalid JSON object for /api/import/archive-path."}}, 400);
            return;
        }
        const std::string raw_path = util::trim_ascii(body.value("path", std::string("")));
        if (raw_path.empty()) {
            send_json(res, {{"detail", "Provide non-empty 'path' to a local .cbz or .zip file."}}, 400);
            return;
        }
        const fs::path archive_path(util::utf8_to_wide(raw_path));
        const std::string suffix = util::to_lower_ascii(util::path_to_utf8(archive_path.extension()));
        if (suffix != ".cbz" && suffix != ".zip") {
            send_json(res, {{"detail", "Unsupported archive format '" + suffix + "'. Expected .cbz or .zip."}}, 400);
            return;
        }

        try {
            const fs::path cache_dir = get_image_cache_dir();
            const auto extracted = extract_archive_pages_from_file(archive_path, cache_dir);
            json pages_json = json::array();
            for (const auto& p : extracted) {
                pages_json.push_back({
                    {"index", p.index},
                    {"filename", p.filename},
                    {"archive_path", p.archive_path},
                    {"image_id", p.image_id},
                    {"image_url", "/api/cache/images/" + p.image_id + "." + p.extension},
                    {"dimensions", {{"width", p.dimensions.width}, {"height", p.dimensions.height}}},
                    {"size_bytes", p.size_bytes},
                });
            }
            send_json(res, {
                {"archive_name", util::path_to_utf8(archive_path.filename())},
                {"total_pages", extracted.size()},
                {"pages", std::move(pages_json)},
            });
        } catch (const ArchiveImportError& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", std::string("Archive import failed: ") + ex.what()}}, 500);
        }
    });

    server.Post("/api/export/cbz", [](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto parsed = parse_export_pages_request(req);
            const auto cbz_bytes = build_cbz_bytes(parsed.pages, parsed.image_format, parsed.jpeg_quality);
            const std::string download_name = sanitize_export_filename(parsed.export_name, "cbz");

            apply_cors_headers(res);
            res.set_header("Content-Disposition", "attachment; filename=\"" + download_name + "\"");
            res.set_header("X-Page-Count", std::to_string(parsed.pages.size()));
            res.set_content(
                reinterpret_cast<const char*>(cbz_bytes.data()),
                cbz_bytes.size(),
                "application/vnd.comicbook+zip");
        } catch (const ComicExportError& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", std::string("CBZ export failed: ") + ex.what()}}, 500);
        }
    });

    server.Post("/api/export/pdf", [](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto parsed = parse_export_pages_request(req);
            const auto pdf_bytes = build_pdf_bytes(parsed.pages);
            const std::string download_name = sanitize_export_filename(parsed.export_name, "pdf");

            apply_cors_headers(res);
            res.set_header("Content-Disposition", "attachment; filename=\"" + download_name + "\"");
            res.set_header("X-Page-Count", std::to_string(parsed.pages.size()));
            res.set_content(
                reinterpret_cast<const char*>(pdf_bytes.data()),
                pdf_bytes.size(),
                "application/pdf");
        } catch (const ComicExportError& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", std::string("PDF export failed: ") + ex.what()}}, 500);
        }
    });

    // -------------------------------------------------------------------------
    // Streamed Disk-Backed Export Sessions (O(1) RAM for 1,500 - 10,000+ Pages)
    //   POST /api/export/session/start
    //   POST /api/export/session/{id}/page
    //   POST /api/export/session/{id}/finalize
    //   POST /api/export/session/{id}/cancel
    // -------------------------------------------------------------------------

    server.Post("/api/export/session/start", [&ctx](const httplib::Request& req, httplib::Response& res) {
        try {
            json body = json::object();
            if (!util::trim_ascii(req.body).empty()) {
                body = json::parse(req.body, nullptr, false);
                if (body.is_discarded() || !body.is_object()) {
                    send_json(res, {{"detail", "Invalid JSON object for /api/export/session/start."}}, 400);
                    return;
                }
            }

            ExportSessionState st;
            st.id = "exp_" + util::generate_uuid_hex();
            const std::string fmt = util::to_lower_ascii(util::trim_ascii(body.value("format", std::string("cbz"))));
            st.format = (fmt == "pdf") ? "pdf" : "cbz";
            if (body.contains("filename") && body["filename"].is_string() &&
                !util::trim_ascii(body["filename"].get<std::string>()).empty()) {
                st.export_name = util::trim_ascii(body["filename"].get<std::string>());
            }
            if (body.contains("image_format") && body["image_format"].is_string()) {
                st.image_format = parse_export_image_format(body["image_format"].get<std::string>());
            }
            if (body.contains("jpeg_quality") && body["jpeg_quality"].is_number()) {
                st.jpeg_quality = std::max(50, std::min(100, body["jpeg_quality"].get<int>()));
            }

            st.session_dir = get_export_sessions_dir() / util::utf8_to_wide(st.id);
            std::error_code ec;
            fs::create_directories(st.session_dir, ec);
            if (ec) {
                throw std::runtime_error("Failed to create temporary export session directory.");
            }

            const std::string sid = st.id;
            {
                std::lock_guard<std::mutex> lock(ctx.export_sessions_mutex);
                ctx.export_sessions[sid] = std::move(st);
            }

            send_json(res, {
                {"status", "ok"},
                {"session_id", sid},
            });
        } catch (const ComicExportError& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", std::string("Failed to start export session: ") + ex.what()}}, 500);
        }
    });

    server.Post(R"(/api/export/session/([A-Za-z0-9_-]{1,64})/page)",
        [&ctx](const httplib::Request& req, httplib::Response& res) {
            const std::string sid = req.matches[1].str();
            if (!req.is_multipart_form_data()) {
                send_json(res, {{"detail", "Expected multipart/form-data with field 'image'."}}, 400);
                return;
            }
            httplib::FormData upload;
            if (req.form.has_file("image")) {
                upload = req.form.get_file("image");
            } else if (req.form.has_file("file")) {
                upload = req.form.get_file("file");
            } else {
                send_json(res, {{"detail", "Missing 'image' file in export page upload."}}, 400);
                return;
            }
            if (upload.content.empty()) {
                send_json(res, {{"detail", "Uploaded export page image is empty."}}, 400);
                return;
            }

            int page_index = -1;
            if (req.form.has_field("index")) {
                try {
                    page_index = std::stoi(util::trim_ascii(req.form.get_field("index")));
                } catch (...) {
                    send_json(res, {{"detail", "Invalid integer for 'index'."}}, 400);
                    return;
                }
            }

            std::optional<std::string> page_filename;
            if (req.form.has_field("filename") && !util::trim_ascii(req.form.get_field("filename")).empty()) {
                page_filename = util::trim_ascii(req.form.get_field("filename"));
            } else if (!upload.filename.empty()) {
                page_filename = upload.filename;
            }

            fs::path session_dir;
            int assigned_index = 0;
            {
                std::lock_guard<std::mutex> lock(ctx.export_sessions_mutex);
                auto it = ctx.export_sessions.find(sid);
                if (it == ctx.export_sessions.end()) {
                    send_json(res, {{"detail", "Export session not found or already finalized."}}, 404);
                    return;
                }
                session_dir = it->second.session_dir;
                assigned_index = (page_index >= 0)
                    ? page_index
                    : (it->second.pages.empty() ? 0 : (it->second.pages.rbegin()->first + 1));
            }

            char fname_buf[64]{};
            std::snprintf(fname_buf, sizeof(fname_buf), "p_%06d.img", assigned_index);
            const fs::path page_path = session_dir / util::utf8_to_wide(fname_buf);

            std::string write_err;
            if (!util::atomic_write_bytes(
                    page_path,
                    reinterpret_cast<const uint8_t*>(upload.content.data()),
                    upload.content.size(),
                    &write_err)) {
                send_json(res, {{"detail", "Failed to write export page to disk: " + write_err}}, 500);
                return;
            }

            size_t count_now = 0;
            {
                std::lock_guard<std::mutex> lock(ctx.export_sessions_mutex);
                auto it = ctx.export_sessions.find(sid);
                if (it == ctx.export_sessions.end()) {
                    std::error_code ec;
                    fs::remove(page_path, ec);
                    send_json(res, {{"detail", "Export session closed during upload."}}, 404);
                    return;
                }
                it->second.pages[assigned_index] = DiskExportPageEntry{page_path, std::move(page_filename)};
                count_now = it->second.pages.size();
            }

            send_json(res, {
                {"status", "ok"},
                {"index", assigned_index},
                {"received_pages", count_now},
            });
        });

    server.Post(R"(/api/export/session/([A-Za-z0-9_-]{1,64})/finalize)",
        [&ctx](const httplib::Request& req, httplib::Response& res) {
            const std::string sid = req.matches[1].str();
            ExportSessionState st;
            {
                std::lock_guard<std::mutex> lock(ctx.export_sessions_mutex);
                auto it = ctx.export_sessions.find(sid);
                if (it == ctx.export_sessions.end()) {
                    send_json(res, {{"detail", "Export session not found or already finalized."}}, 404);
                    return;
                }
                st = std::move(it->second);
                ctx.export_sessions.erase(it);
            }

            auto cleanup_dir = [dir = st.session_dir]() {
                std::error_code ec;
                fs::remove_all(dir, ec);
            };

            try {
                std::vector<DiskExportPageEntry> ordered_pages;
                ordered_pages.reserve(st.pages.size());
                for (auto& kv : st.pages) {
                    ordered_pages.push_back(std::move(kv.second));
                }
                if (ordered_pages.empty()) {
                    cleanup_dir();
                    send_json(res, {{"detail", "Export session has no uploaded pages."}}, 400);
                    return;
                }

                const std::string download_name = sanitize_export_filename(st.export_name, st.format);
                const fs::path out_file = st.session_dir / util::utf8_to_wide("output." + st.format);
                const std::string mime = (st.format == "pdf")
                    ? "application/pdf"
                    : "application/vnd.comicbook+zip";

                if (st.format == "pdf") {
                    build_pdf_to_file(ordered_pages, out_file);
                } else {
                    build_cbz_to_file(ordered_pages, out_file, st.image_format, st.jpeg_quality);
                }

                std::error_code sz_ec;
                const uint64_t file_size = fs::file_size(out_file, sz_ec);
                if (sz_ec || file_size == 0) {
                    cleanup_dir();
                    send_json(res, {{"detail", "Finalized export file is missing or empty."}}, 500);
                    return;
                }

                bool return_url = req.has_param("return_url") &&
                    parse_bool_param(req.get_param_value("return_url"), false);
                if (!return_url && !util::trim_ascii(req.body).empty()) {
                    const json body = json::parse(req.body, nullptr, false);
                    if (body.is_object() && body.contains("return_url") && body["return_url"].is_boolean()) {
                        return_url = body["return_url"].get<bool>();
                    }
                }

                if (return_url) {
                    CompletedExportState done;
                    done.file_path = out_file;
                    done.session_dir = st.session_dir;
                    done.download_name = download_name;
                    done.mime_type = mime;
                    done.file_size = file_size;
                    done.page_count = ordered_pages.size();
                    {
                        std::lock_guard<std::mutex> lock(ctx.export_sessions_mutex);
                        ctx.completed_exports[sid] = std::move(done);
                    }
                    send_json(res, {
                        {"status", "ok"},
                        {"session_id", sid},
                        {"download_url", "/api/export/session/" + sid + "/download"},
                        {"filename", download_name},
                        {"size_bytes", file_size},
                        {"page_count", ordered_pages.size()},
                    });
                    return;
                }

                auto stream_file = std::make_shared<std::ifstream>(out_file, std::ios::binary);
                if (!stream_file->is_open()) {
                    cleanup_dir();
                    send_json(res, {{"detail", "Failed to open finalized export file for streaming."}}, 500);
                    return;
                }

                apply_cors_headers(res);
                res.set_header("Content-Disposition", "attachment; filename=\"" + download_name + "\"");
                res.set_header("X-Page-Count", std::to_string(ordered_pages.size()));

                const fs::path dir_to_clean = st.session_dir;
                res.set_content_provider(
                    static_cast<size_t>(file_size),
                    mime,
                    [stream_file](size_t offset, size_t length, httplib::DataSink& sink) -> bool {
                        constexpr size_t kChunk = 1024ULL * 1024ULL;  // 1 MiB stream chunks
                        std::vector<char> buf(std::min(length, kChunk));
                        stream_file->clear();
                        stream_file->seekg(static_cast<std::streamoff>(offset), std::ios::beg);
                        size_t remaining = length;
                        while (remaining > 0) {
                            const size_t to_read = std::min(remaining, buf.size());
                            stream_file->read(buf.data(), static_cast<std::streamsize>(to_read));
                            const std::streamsize got = stream_file->gcount();
                            if (got <= 0) {
                                return false;
                            }
                            if (!sink.write(buf.data(), static_cast<size_t>(got))) {
                                return false;
                            }
                            remaining -= static_cast<size_t>(got);
                        }
                        return true;
                    },
                    [stream_file, dir_to_clean](bool /*success*/) {
                        stream_file->close();
                        std::error_code ec;
                        fs::remove_all(dir_to_clean, ec);
                    });
            } catch (const ComicExportError& ex) {
                cleanup_dir();
                send_json(res, {{"detail", ex.what()}}, 400);
            } catch (const std::exception& ex) {
                cleanup_dir();
                send_json(res, {{"detail", std::string("Streamed export failed: ") + ex.what()}}, 500);
            }
        });

    server.Get(R"(/api/export/session/([A-Za-z0-9_-]{1,64})/download)",
        [&ctx](const httplib::Request& req, httplib::Response& res) {
            const std::string sid = req.matches[1].str();
            CompletedExportState done;
            {
                std::lock_guard<std::mutex> lock(ctx.export_sessions_mutex);
                auto it = ctx.completed_exports.find(sid);
                if (it == ctx.completed_exports.end()) {
                    send_json(res, {{"detail", "Completed export not found or already downloaded."}}, 404);
                    return;
                }
                done = std::move(it->second);
                ctx.completed_exports.erase(it);
            }

            auto cleanup_dir = [dir = done.session_dir]() {
                std::error_code ec;
                fs::remove_all(dir, ec);
            };

            auto stream_file = std::make_shared<std::ifstream>(done.file_path, std::ios::binary);
            if (!stream_file->is_open()) {
                cleanup_dir();
                send_json(res, {{"detail", "Failed to open finalized export file for download."}}, 500);
                return;
            }

            apply_cors_headers(res);
            res.set_header("Content-Disposition", "attachment; filename=\"" + done.download_name + "\"");
            res.set_header("X-Page-Count", std::to_string(done.page_count));

            const fs::path dir_to_clean = done.session_dir;
            res.set_content_provider(
                static_cast<size_t>(done.file_size),
                done.mime_type,
                [stream_file](size_t offset, size_t length, httplib::DataSink& sink) -> bool {
                    constexpr size_t kChunk = 1024ULL * 1024ULL;  // 1 MiB stream chunks
                    std::vector<char> buf(std::min(length, kChunk));
                    stream_file->clear();
                    stream_file->seekg(static_cast<std::streamoff>(offset), std::ios::beg);
                    size_t remaining = length;
                    while (remaining > 0) {
                        const size_t to_read = std::min(remaining, buf.size());
                        stream_file->read(buf.data(), static_cast<std::streamsize>(to_read));
                        const std::streamsize got = stream_file->gcount();
                        if (got <= 0) {
                            return false;
                        }
                        if (!sink.write(buf.data(), static_cast<size_t>(got))) {
                            return false;
                        }
                        remaining -= static_cast<size_t>(got);
                    }
                    return true;
                },
                [stream_file, dir_to_clean](bool /*success*/) {
                    stream_file->close();
                    std::error_code ec;
                    fs::remove_all(dir_to_clean, ec);
                });
        });

    server.Post(R"(/api/export/session/([A-Za-z0-9_-]{1,64})/cancel)",
        [&ctx](const httplib::Request& req, httplib::Response& res) {
            const std::string sid = req.matches[1].str();
            fs::path dir;
            {
                std::lock_guard<std::mutex> lock(ctx.export_sessions_mutex);
                auto it = ctx.export_sessions.find(sid);
                if (it != ctx.export_sessions.end()) {
                    dir = it->second.session_dir;
                    ctx.export_sessions.erase(it);
                }
            }
            if (!dir.empty()) {
                std::error_code ec;
                fs::remove_all(dir, ec);
            }
            send_json(res, {{"status", "ok"}, {"cancelled", true}});
        });

    // Clean JSON fallback for unknown /api/* routes
    server.Get(R"(/api/.*)", [](const httplib::Request& req, httplib::Response& res) {
        send_json(
            res,
            {{"detail", "API route not found: " + req.path}},
            404);
    });
    server.Post(R"(/api/.*)", [](const httplib::Request& req, httplib::Response& res) {
        send_json(
            res,
            {{"detail", "API route not found: " + req.path}},
            404);
    });

    // Static frontend asset handler (`frontend/dist` mounted at `/`)
    server.Get(R"(/(.*))", [&ctx](const httplib::Request& req, httplib::Response& res) {
        std::string rel = req.path;
        if (!rel.empty() && rel.front() == '/') {
            rel.erase(rel.begin());
        }

        if (rel == "api" || rel.rfind("api/", 0) == 0) {
            send_json(res, {{"detail", "Not found"}}, 404);
            return;
        }

        fs::path target = rel.empty()
            ? (ctx.frontend_dist_dir / L"index.html")
            : (ctx.frontend_dist_dir / util::utf8_to_wide(rel));

        std::error_code ec;
        fs::path canonical_target = fs::weakly_canonical(target, ec);
        if (ec || !util::is_path_within_root(ctx.frontend_dist_dir, canonical_target)) {
            res.status = 403;
            res.set_content("Forbidden", "text/plain; charset=utf-8");
            return;
        }

        if (fs::is_directory(canonical_target, ec)) {
            canonical_target /= L"index.html";
        }

        if (!fs::is_regular_file(canonical_target, ec)) {
            if (!target.has_extension()) {
                canonical_target = ctx.frontend_dist_dir / L"index.html";
            } else {
                res.status = 404;
                res.set_content("Not Found", "text/plain; charset=utf-8");
                return;
            }
        }

        if (!serve_file_from_disk(canonical_target, req.path, res)) {
            res.status = 500;
            res.set_content("Failed to read static asset", "text/plain; charset=utf-8");
        }
    });
}

// -----------------------------------------------------------------------------
// Server Binding & Readiness Synchronization
// -----------------------------------------------------------------------------

bool is_loopback_port_available(const char* host, int port) {
    if (port <= 0 || port > 65535) {
        return false;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<u_short>(port));
    if (InetPtonA(AF_INET, host, &addr.sin_addr) != 1) {
        return false;
    }

    // Fast path: check exclusive bind first (<0.1 ms).
    SOCKET test_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (test_sock == INVALID_SOCKET) {
        return false;
    }
    const int exclusive = 1;
    setsockopt(
        test_sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
        reinterpret_cast<const char*>(&exclusive), sizeof(exclusive));
    const bool can_bind =
        bind(test_sock, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) == 0;
    closesocket(test_sock);
    if (!can_bind) {
        return false;
    }

    // Secondary guard for listeners bound on 0.0.0.0 with SO_REUSEADDR.
    SOCKET probe = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (probe != INVALID_SOCKET) {
        u_long nonblocking = 1;
        ioctlsocket(probe, FIONBIO, &nonblocking);
        const int conn_res = connect(
            probe, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr));
        bool in_use = (conn_res == 0);
        if (!in_use && WSAGetLastError() == WSAEWOULDBLOCK) {
            fd_set write_fds{};
            fd_set except_fds{};
            FD_ZERO(&write_fds);
            FD_ZERO(&except_fds);
            FD_SET(probe, &write_fds);
            FD_SET(probe, &except_fds);
            timeval tv{};
            tv.tv_sec = 0;
            tv.tv_usec = 30000;  // 30 ms is plenty for loopback TCP SYN-ACK
            if (select(0, nullptr, &write_fds, &except_fds, &tv) > 0 &&
                FD_ISSET(probe, &write_fds)) {
                int so_err = 0;
                int opt_len = sizeof(so_err);
                if (getsockopt(
                        probe, SOL_SOCKET, SO_ERROR,
                        reinterpret_cast<char*>(&so_err), &opt_len) == 0 &&
                    so_err == 0) {
                    in_use = true;
                }
            }
        }
        closesocket(probe);
        if (in_use) {
            return false;
        }
    }

    return true;
}

int bind_server_port(httplib::Server& server, int preferred_port) {
    server.set_socket_options([](socket_t sock) {
        const int exclusive = 1;
        setsockopt(
            sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
            reinterpret_cast<const char*>(&exclusive), sizeof(exclusive));
    });

    if (is_loopback_port_available(kHost, preferred_port)) {
        if (server.bind_to_port(kHost, preferred_port)) {
            return preferred_port;
        }
        server.stop();
    }
    return server.bind_to_any_port(kHost);
}

bool wait_for_server_ready(int port, std::atomic<bool>& server_failed, int timeout_ms = 5000) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    while (std::chrono::steady_clock::now() < deadline) {
        if (server_failed.load()) {
            return false;
        }
        httplib::Client probe(kHost, port);
        probe.set_connection_timeout(0, 150000);
        probe.set_read_timeout(0, 250000);
        if (auto res = probe.Get("/api/ping"); res && res->status == 200) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

// -----------------------------------------------------------------------------
// Native Win32 Window Styling
// -----------------------------------------------------------------------------

void apply_dark_window_theme(void* native_window) {
    if (!native_window) return;
    HWND hwnd = static_cast<HWND>(native_window);
    const BOOL use_dark_mode = TRUE;
    if (FAILED(DwmSetWindowAttribute(hwnd, 20, &use_dark_mode, sizeof(use_dark_mode)))) {
        DwmSetWindowAttribute(hwnd, 19, &use_dark_mode, sizeof(use_dark_mode));
    }
}

void apply_window_icon(void* native_window) {
    if (!native_window) return;
    HWND hwnd = static_cast<HWND>(native_window);
    HINSTANCE hinst = GetModuleHandleW(nullptr);
    if (HICON hicon_big = static_cast<HICON>(LoadImageW(
            hinst, MAKEINTRESOURCEW(101), IMAGE_ICON,
            GetSystemMetrics(SM_CXICON), GetSystemMetrics(SM_CYICON), LR_DEFAULTCOLOR))) {
        SendMessageW(hwnd, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(hicon_big));
    }
    if (HICON hicon_small = static_cast<HICON>(LoadImageW(
            hinst, MAKEINTRESOURCEW(101), IMAGE_ICON,
            GetSystemMetrics(SM_CXSMICON), GetSystemMetrics(SM_CYSMICON), LR_DEFAULTCOLOR))) {
        SendMessageW(hwnd, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(hicon_small));
    }
}

void show_error_dialog(const std::wstring& message) {
    MessageBoxW(
        nullptr,
        message.c_str(),
        L"Vellum - Startup Error",
        MB_OK | MB_ICONERROR);
}

}  // namespace mangaflow

void register_cuda_dll_paths(const fs::path& exe_dir) {
    std::vector<fs::path> roots;
    roots.push_back(exe_dir / "cuda");
    if (auto cuda_env = mangaflow::util::get_env_wide(L"CUDA_PATH"); cuda_env.has_value() && !cuda_env->empty()) {
        roots.emplace_back(*cuda_env);
    }
    roots.emplace_back(L"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA");
    if (auto local_app = mangaflow::util::get_env_wide(L"LOCALAPPDATA"); local_app.has_value() && !local_app->empty()) {
        roots.emplace_back(fs::path(*local_app) / L"LM-Studio");
    }
    if (auto user_prof = mangaflow::util::get_env_wide(L"USERPROFILE"); user_prof.has_value() && !user_prof->empty()) {
        roots.emplace_back(fs::path(*user_prof) / L".lmstudio");
    }

    std::wstring extra_paths;
    std::error_code ec;
    for (const auto& root : roots) {
        if (!fs::is_directory(root, ec) || ec) continue;
        for (const auto& entry : fs::recursive_directory_iterator(
                 root, fs::directory_options::skip_permission_denied, ec)) {
            if (ec) break;
            if (!entry.is_directory(ec) || ec) continue;
            const std::wstring name = entry.path().filename().wstring();
            if (name == L"bin" || name == L"x86_64" || name == L"cuda12") {
                extra_paths += entry.path().wstring();
                extra_paths += L";";
            }
        }
    }

    if (!extra_paths.empty()) {
        const DWORD len = GetEnvironmentVariableW(L"PATH", nullptr, 0);
        std::wstring cur_path(len > 0 ? len : 0, L'\0');
        if (len > 0) {
            GetEnvironmentVariableW(L"PATH", cur_path.data(), len);
            if (!cur_path.empty() && cur_path.back() == L'\0') cur_path.pop_back();
        }
        const std::wstring combined = extra_paths + cur_path;
        SetEnvironmentVariableW(L"PATH", combined.c_str());
    }
}

// -----------------------------------------------------------------------------
// Win32 GUI Entry Point (Zero Black Console Window)
// -----------------------------------------------------------------------------

int WINAPI WinMain(
    HINSTANCE /*hInst*/,
    HINSTANCE /*hPrevInst*/,
    LPSTR lpCmdLine,
    int /*nCmdShow*/) {
    using namespace mangaflow;

    const std::string cmd_line = lpCmdLine ? std::string(lpCmdLine) : std::string();
    const bool headless = util::env_flag_enabled(L"VELLUM_HEADLESS") ||
                          util::env_flag_enabled(L"MANGAFLOW_HEADLESS") ||
                          cmd_line.find("--headless") != std::string::npos;

    const fs::path exe_dir = get_executable_dir();
    register_cuda_dll_paths(exe_dir);
    const auto frontend_dist = resolve_frontend_dist(exe_dir);
    if (!frontend_dist.has_value() && !headless) {
        show_error_dialog(
            L"Compiled React v2 frontend bundle (`frontend/dist/index.html`) was not found.\n\n"
            L"Please run `npm run build` inside the `frontend/` directory first.");
        return 1;
    }

    AppContext ctx;
    ctx.exe_dir = exe_dir;
    ctx.frontend_dist_dir = frontend_dist.value_or(exe_dir);
    ctx.ort_version = pipeline::ModelManager::get_ort_version();
    ctx.available_providers = pipeline::ModelManager::get_available_providers();
    ctx.model_manager = std::make_unique<pipeline::ModelManager>(
        std::nullopt,
        pipeline::ModelManager::find_workspace_models_dir(exe_dir),
        std::nullopt);
    ctx.detector = std::make_unique<pipeline::TextDetector>(*ctx.model_manager);
    ctx.ocr_engine = std::make_unique<pipeline::OcrEngine>(*ctx.model_manager);
    ctx.inpainter = std::make_unique<pipeline::LamaInpainter>(*ctx.model_manager);
    ctx.start_unload_watchdog();

    // Pre-load persistent settings so execution_mode (`MANGA_FORCE_CPU`) is synced at startup
    const json initial_settings = ctx.settings_manager.load();
    const std::string init_exec_mode = initial_settings.value("execution_mode", std::string("hybrid"));
    if (init_exec_mode == "cpu_only" || init_exec_mode == "cpu") {
        util::set_env_utf8(L"MANGA_FORCE_CPU", "1");
    }

    httplib::Server server;
    server.set_read_timeout(300, 0);
    server.set_write_timeout(600, 0);
    server.set_keep_alive_max_count(10000);
    server.set_payload_max_length(16ULL * 1024ULL * 1024ULL * 1024ULL);  // 16 GiB for multi-GB 10,000-page archives & exports
    register_routes(server, ctx);

    int preferred_port = kDefaultPort;
    auto env_port = util::get_env_utf8(L"VELLUM_PORT");
    if (!env_port) {
        env_port = util::get_env_utf8(L"MANGAFLOW_PORT");
    }
    if (env_port) {
        try {
            preferred_port = std::stoi(*env_port);
        } catch (...) {
            preferred_port = kDefaultPort;
        }
    }

    const int bound_port = bind_server_port(server, preferred_port);
    if (bound_port <= 0) {
        if (!headless) {
            show_error_dialog(L"Failed to bind local loopback HTTP server on 127.0.0.1.");
        }
        return 1;
    }

    std::atomic<bool> server_failed{false};
    std::thread server_thread([&server, &server_failed]() {
        try {
            if (!server.listen_after_bind()) {
                server_failed.store(true);
            }
        } catch (...) {
            server_failed.store(true);
        }
    });

    auto stop_server = [&server, &server_thread, &ctx]() {
        server.stop();
        if (server_thread.joinable()) {
            server_thread.join();
        }
        ctx.stop_unload_watchdog();
    };

    if (!wait_for_server_ready(bound_port, server_failed)) {
        stop_server();
        if (!headless) {
            show_error_dialog(L"Embedded C++ HTTP backend failed to respond on /api/health.");
        }
        return 1;
    }

    if (headless) {
        // Pure HTTP backend mode for automated verification / CLI hosting
        if (server_thread.joinable()) {
            server_thread.join();
        }
        ctx.stop_unload_watchdog();
        return server_failed.load() ? 1 : 0;
    }

    const std::string app_url =
        "http://" + std::string(kHost) + ":" + std::to_string(bound_port) + "/";

    int exit_code = 0;
    try {
        const bool debug_devtools = util::env_flag_enabled(L"VELLUM_DEBUG") ||
                                    util::env_flag_enabled(L"MANGAFLOW_DEBUG");
        webview::webview window(debug_devtools, nullptr);

        if (auto native_win = window.window(); native_win.ok()) {
            apply_dark_window_theme(native_win.value());
            apply_window_icon(native_win.value());
        }

        window.set_title(kWindowTitle);
        window.set_size(kWindowMinWidth, kWindowMinHeight, WEBVIEW_HINT_MIN);
        window.set_size(kWindowWidth, kWindowHeight, WEBVIEW_HINT_NONE);
        window.navigate(app_url);
        window.run();
    } catch (const webview::exception& ex) {
        show_error_dialog(
            L"Failed to initialize native Edge WebView2 window:\n\n" +
            util::utf8_to_wide(ex.what()));
        exit_code = 1;
    } catch (const std::exception& ex) {
        show_error_dialog(
            L"Unexpected runtime error:\n\n" +
            util::utf8_to_wide(ex.what()));
        exit_code = 1;
    }

    stop_server();
    return exit_code;
}
