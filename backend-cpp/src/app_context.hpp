#pragma once

// Vellum - Shared Server Application Context & HTTP/JSON Helpers

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "detector/detector_engine.hpp"
#include "exporter.hpp"
#include "font_manager.hpp"
#include "pipeline/detector.hpp"
#include "pipeline/inpainter.hpp"
#include "pipeline/math_nlp_qc.hpp"
#include "pipeline/model_manager.hpp"
#include "pipeline/ocr.hpp"
#include "pipeline/translator.hpp"
#include "settings_manager.hpp"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mangaflow {

namespace fs = std::filesystem;
using json = nlohmann::json;

constexpr const char* kHost = "127.0.0.1";
constexpr int kDefaultPort = 8000;
constexpr const char* kWindowTitle = "Vellum";
constexpr int kWindowWidth = 1440;
constexpr int kWindowHeight = 900;
constexpr int kWindowMinWidth = 1024;
constexpr int kWindowMinHeight = 680;
constexpr const char* kAppVersion = "2.0.0";

struct TranslateBlockItemInput {
    std::string id;
    int order = 0;
    std::array<int, 4> bbox{0, 0, 0, 0};
    json polygon;
    std::string original_text;
    std::optional<int> font_size;
    std::string color = "#000000";
    std::string align = "center";
    double confidence = 1.0;
    std::string text_direction = "horizontal";
    std::string bubble_class;
    int panel_id = -1;
    int speaker_id = -1;
};

struct DecodedOptionalImage {
    bool valid = false;
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;
};

struct ParsedExportRequest {
    std::vector<ExportPageInput> pages;
    std::optional<std::string> export_name;
    ExportImageFormat image_format = ExportImageFormat::Original;
    int jpeg_quality = 95;
};

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
    std::shared_ptr<pipeline::math_nlp::PHashLruCache> preflight_phash_cache =
        std::make_shared<pipeline::math_nlp::PHashLruCache>(2048);
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

    void unload_all_sessions_unlocked() {
        if (detector) detector->unload();
        if (detector_v2) detector_v2->unload();
        if (ocr_engine) ocr_engine->unload();
        if (inpainter) inpainter->unload();
        pipeline::ModelManager::trim_process_heap();
    }

    void unload_all_sessions() {
        std::lock_guard<std::mutex> lock(inference_mutex);
        unload_all_sessions_unlocked();
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

// Shared Path, Image Cache, JSON & HTTP Helpers (defined in app_context.cpp)
fs::path get_executable_dir();
std::optional<fs::path> resolve_frontend_dist(const fs::path& exe_dir);
std::pair<std::string, std::string> save_mask_png_to_cache(const std::vector<uint8_t>& png_bytes);
std::pair<std::string, std::string> save_inpainted_png_to_cache(const std::vector<uint8_t>& png_bytes);
json probe_lm_studio(SettingsManager& settings_mgr, double timeout_seconds = 3.0);
size_t count_utf8_codepoints(const std::string& s);
int estimate_font_size(const std::array<int, 4>& bbox, const std::string& text);
json bbox_to_polygon_json(const std::array<int, 4>& bbox);
json build_default_style_json(int font_size, const std::string& color = "#000000", const std::string& align = "center");
std::string json_scalar_str(const json& v);
std::vector<TranslateBlockItemInput> parse_translate_block_inputs(const json& raw_blocks);
json build_translated_blocks_json(
    const std::vector<TranslateBlockItemInput>& items,
    const std::map<std::string, std::string>& trans_map);
std::string extract_page_id_from_filename(const std::string& filename);
DecodedOptionalImage decode_optional_page_image(
    const std::optional<std::string>& uploaded_bytes,
    const std::optional<std::string>& image_url,
    const std::optional<std::string>& image_base64);
std::string mime_type_for_extension(const std::string& raw_ext);
bool serve_file_from_disk(const fs::path& file_path, const std::string& request_path, httplib::Response& res);
bool parse_bool_param(const std::string& raw, bool default_val);
ParsedExportRequest parse_export_pages_request(const httplib::Request& req);
BatchKeepAliveHint parse_batch_keep_alive(const httplib::Request& req);
std::optional<float> parse_optional_float_field(const httplib::Request& req, const char* name);
void apply_optional_nullable_threshold_field(
    const httplib::Request& req,
    const char* name,
    std::optional<float>& target);
void populate_detection_options_from_settings(pipeline::DetectionOptions& opts, const json& settings);
void populate_ocr_options_from_settings(pipeline::OcrOptions& opts, const json& settings);
void populate_inpaint_options_from_settings(pipeline::InpaintOptions& opts, const json& settings);
json extract_multipart_translation_overrides(const httplib::Request& req);
void apply_cors_headers(httplib::Response& res);
void send_json(httplib::Response& res, const json& payload, int status = 200);

}  // namespace mangaflow
