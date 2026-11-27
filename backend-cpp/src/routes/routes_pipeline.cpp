#include "routes_pipeline.hpp"
#include "../detector/adaptive_tiler.hpp"
#include "../detector/export_artifacts.hpp"
#include "../detector/geometry_postproc.hpp"
#include "../pipeline/ort_provider.hpp"
#include "../utils.hpp"

#include <stb_image.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mangaflow::routes {

void register_pipeline_routes(httplib::Server& server, AppContext& ctx) {
    // Stage 1: Text & Layout Detection (`POST /api/stage/detect`, `/api/pipeline/detect`)
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // Stage 0: Fast CPU Preflight — Shannon Entropy Blank-Page Guard + 64-Bit 2D DCT pHash Cache (#7)
    // -------------------------------------------------------------------------
    auto preflight_phash_cache = ctx.preflight_phash_cache;

    auto handle_stage_preflight = [preflight_phash_cache](const httplib::Request& req, httplib::Response& res) {
        const auto t_start = std::chrono::steady_clock::now();
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

        const std::string target_lang = req.form.has_field("target_lang")
            ? util::trim_ascii(req.form.get_field("target_lang"))
            : (req.form.has_field("target_language") ? util::trim_ascii(req.form.get_field("target_language")) : std::string("id"));
        const std::string profile_slug = req.form.has_field("profile_slug")
            ? util::trim_ascii(req.form.get_field("profile_slug"))
            : std::string("default");
        float entropy_thresh = 0.22f;
        if (auto et = parse_optional_float_field(req, "entropy_threshold"); et.has_value()) {
            entropy_thresh = *et;
        }
        int hamming_thresh = 4;
        if (req.form.has_field("hamming_threshold")) {
            try {
                hamming_thresh = std::clamp(std::stoi(util::trim_ascii(req.form.get_field("hamming_threshold"))), 0, 16);
            } catch (...) {}
        }

        int img_w = 0, img_h = 0, img_c = 0;
        std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> rgb(
            stbi_load_from_memory(
                reinterpret_cast<const stbi_uc*>(upload.content.data()),
                static_cast<int>(upload.content.size()),
                &img_w, &img_h, &img_c, 3),
            &stbi_image_free);
        std::vector<uint8_t> wic_rgb;
        const uint8_t* rgb_ptr = rgb.get();
        if (!rgb_ptr || img_w <= 0 || img_h <= 0) {
            rgb.reset();
            if (util::decode_image_wic(
                    reinterpret_cast<const uint8_t*>(upload.content.data()),
                    upload.content.size(), 3, wic_rgb, img_w, img_h) &&
                !wic_rgb.empty() && img_w > 0 && img_h > 0) {
                rgb_ptr = wic_rgb.data();
            }
        }
        if (!rgb_ptr || img_w <= 0 || img_h <= 0) {
            send_json(res, {{"detail", "Failed to decode uploaded image bytes."}}, 400);
            return;
        }

        const auto t_math = std::chrono::steady_clock::now();
        const float entropy = pipeline::math_nlp::computeBlankPageEntropy(rgb_ptr, img_w, img_h);
        const uint64_t phash = pipeline::math_nlp::computeDctPHash64(rgb_ptr, img_w, img_h);
        const std::string phash_hex = pipeline::math_nlp::formatPHashHex(phash);
        const double math_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_math).count();

        if (entropy < entropy_thresh) {
            const double elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_start).count();
            send_json(res, {
                {"skipped", true},
                {"reason", "blank_page"},
                {"cache_hit", false},
                {"entropy", std::round(static_cast<double>(entropy) * 10000.0) / 10000.0},
                {"phash", phash_hex},
                {"width", img_w},
                {"height", img_h},
                {"blocks", json::array()},
                {"bubbles", json::array()},
                {"panels", json::array()},
                {"math_ms", std::round(math_ms * 1000.0) / 1000.0},
                {"elapsed_ms", std::round(elapsed_ms * 1000.0) / 1000.0},
            });
            return;
        }

        if (req.form.has_field("store_result")) {
            const std::string payload = req.form.get_field("store_result");
            if (!payload.empty()) {
                json parsed_payload = json::parse(payload, nullptr, false);
                if (parsed_payload.is_discarded()) {
                    parsed_payload = payload;
                }
                preflight_phash_cache->put(target_lang, profile_slug, phash, std::move(parsed_payload));
                const double elapsed_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t_start).count();
                send_json(res, {
                    {"skipped", false},
                    {"cache_hit", false},
                    {"stored", true},
                    {"entropy", std::round(static_cast<double>(entropy) * 10000.0) / 10000.0},
                    {"phash", phash_hex},
                    {"math_ms", std::round(math_ms * 1000.0) / 1000.0},
                    {"elapsed_ms", std::round(elapsed_ms * 1000.0) / 1000.0},
                });
                return;
            }
        }

        if (auto hit = preflight_phash_cache->lookup(target_lang, profile_slug, phash, hamming_thresh); hit.has_value()) {
            const double elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_start).count();
            send_json(res, {
                {"skipped", false},
                {"cache_hit", true},
                {"reason", "phash_duplicate"},
                {"entropy", std::round(static_cast<double>(entropy) * 10000.0) / 10000.0},
                {"phash", phash_hex},
                {"matched_phash", pipeline::math_nlp::formatPHashHex(hit->matched_hash)},
                {"hamming_distance", hit->hamming_distance},
                {"cached_result", std::move(hit->payload)},
                {"math_ms", std::round(math_ms * 1000.0) / 1000.0},
                {"elapsed_ms", std::round(elapsed_ms * 1000.0) / 1000.0},
            });
            return;
        }

        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_start).count();
        send_json(res, {
            {"skipped", false},
            {"cache_hit", false},
            {"entropy", std::round(static_cast<double>(entropy) * 10000.0) / 10000.0},
            {"phash", phash_hex},
            {"width", img_w},
            {"height", img_h},
            {"math_ms", std::round(math_ms * 1000.0) / 1000.0},
            {"elapsed_ms", std::round(elapsed_ms * 1000.0) / 1000.0},
        });
    };
    server.Post("/api/stage/preflight", handle_stage_preflight);
    server.Post("/api/pipeline/preflight", handle_stage_preflight);

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

        std::string detector_precision = current_settings.value("detector_precision", std::string("balanced"));
        if (req.form.has_field("detector_precision")) {
            const std::string dp = util::trim_ascii(req.form.get_field("detector_precision"));
            if (!dp.empty()) {
                detector_precision = SettingsManager::normalize_detector_precision(dp);
            }
        } else if (req.has_param("detector_precision")) {
            const std::string dp = util::trim_ascii(req.get_param_value("detector_precision"));
            if (!dp.empty()) {
                detector_precision = SettingsManager::normalize_detector_precision(dp);
            }
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
        if (sel_lower.empty() || sel_lower == "auto" ||
            sel_lower.find("yolo") != std::string::npos ||
            sel_lower.find("rfdetr") != std::string::npos) {
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
                          "') are missing. Download or import local models via Settings -> Pipeline first."}},
                    503);
                return;
            }
            target_backend = vellum::detector::BackendType::LegacyCtd;
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
            std::vector<uint8_t> wic_rgb;
            const uint8_t* rgb_ptr = rgb.get();
            if (!rgb_ptr || img_w <= 0 || img_h <= 0) {
                rgb.reset();
                if (util::decode_image_wic(
                        reinterpret_cast<const uint8_t*>(upload.content.data()),
                        upload.content.size(),
                        3,
                        wic_rgb,
                        img_w,
                        img_h) &&
                    !wic_rgb.empty() && img_w > 0 && img_h > 0) {
                    rgb_ptr = wic_rgb.data();
                }
            }
            if (!rgb_ptr || img_w <= 0 || img_h <= 0) {
                const char* stbi_err = stbi_failure_reason();
                const bool looks_webp = util::is_webp_bytes(
                    reinterpret_cast<const uint8_t*>(upload.content.data()),
                    upload.content.size());
                const std::string detail = std::string("Failed to decode uploaded image bytes (") +
                    (stbi_err ? stbi_err : "unrecognized/corrupt format") +
                    (looks_webp ? "; WebP WIC fallback failed" : "; WIC fallback failed") + ").";
                send_json(res, {{"detail", detail}}, 400);
                return;
            }

            // #7: Fast Shannon Entropy Blank-Page Guard before invoking GPU ONNX
            bool skip_blank_pages = false;
            if (req.form.has_field("skip_blank_pages")) {
                skip_blank_pages = parse_bool_param(req.form.get_field("skip_blank_pages"), false);
            } else if (req.has_param("skip_blank_pages")) {
                skip_blank_pages = parse_bool_param(req.get_param_value("skip_blank_pages"), false);
            }
            if (skip_blank_pages) {
                const float page_entropy = pipeline::math_nlp::computeBlankPageEntropy(rgb_ptr, img_w, img_h);
                if (page_entropy < 0.22f) {
                    const uint64_t phash = pipeline::math_nlp::computeDctPHash64(rgb_ptr, img_w, img_h);
                    send_json(res, {
                        {"skipped", true},
                        {"reason", "blank_page"},
                        {"entropy", std::round(static_cast<double>(page_entropy) * 10000.0) / 10000.0},
                        {"phash", pipeline::math_nlp::formatPHashHex(phash)},
                        {"width", img_w},
                        {"height", img_h},
                        {"count", 0},
                        {"bubbles", json::array()},
                        {"blocks", json::array()},
                        {"panels", json::array()},
                    });
                    return;
                }
            }

            vellum::detector::DetectorConfig v2_cfg;
            v2_cfg.model_path = resolved_model_path;
            v2_cfg.backend = target_backend;
            v2_cfg.force_cpu = opts.force_cpu;
            v2_cfg.keep_session_alive = !opts.auto_unload;
            v2_cfg.export_dir = export_dir;
            v2_cfg.enable_adaptive_tiling = current_settings.value("enable_adaptive_tiling", true);
            v2_cfg.enable_cv_style_detect = current_settings.value("enable_cv_style_detect", true);
            v2_cfg.enable_waist_cut = current_settings.value("enable_waist_cut", true);
            v2_cfg.auto_deskew_scans = current_settings.value("auto_deskew_scans", false);
            v2_cfg.precision = vellum::detector::parse_detector_precision(detector_precision);
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
                if (pipeline::is_vram_tight()) {
                    if (ctx.ocr_engine) ctx.ocr_engine->unload();
                    if (ctx.inpainter) ctx.inpainter->unload();
                }
                if (!ctx.detector_v2 || ctx.detector_v2_backend != target_backend) {
                    if (ctx.detector_v2) {
                        ctx.detector_v2->unload();
                    }
                    ctx.detector_v2 = vellum::detector::DetectorFactory::create(target_backend);
                    ctx.detector_v2_backend = target_backend;
                }
                ctx.detector_v2->load(v2_cfg);
                v2_res = vellum::detector::AdaptiveTiler::detectPage(
                    *ctx.detector_v2, rgb_ptr, img_w, img_h, upload.filename, v2_cfg);
            }
            if (!v2_cfg.export_dir.empty()) {
                vellum::detector::exportPageArtifacts(v2_res, rgb_ptr, img_w, img_h, v2_cfg);
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
                    rgb_ptr, img_w, img_h, v2_res.detections);
            rgb.reset();  // Release uncompressed RGB image buffer before PNG/JSON encoding
            std::vector<uint8_t>().swap(wic_rgb);

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

                json bubble_obj = {
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
                    {"speaker_id", d.speaker_cluster_id},
                    {"is_cross_panel", d.is_cross_panel},
                };
                if (!d.room_polygon.empty()) {
                    json rpoly = json::array();
                    for (const auto& pt : d.room_polygon) {
                        rpoly.push_back({pt.x, pt.y});
                    }
                    bubble_obj["room_polygon"] = std::move(rpoly);
                }
                if (d.angle_deg.has_value()) {
                    bubble_obj["angle_deg"] = *d.angle_deg;
                }
                if (d.estimated_font_size.has_value()) {
                    bubble_obj["estimated_font_size"] = *d.estimated_font_size;
                }
                if (d.text_color.has_value()) {
                    bubble_obj["text_color"] = {
                        static_cast<int>((*d.text_color)[0]),
                        static_cast<int>((*d.text_color)[1]),
                        static_cast<int>((*d.text_color)[2]),
                    };
                }
                if (d.outline.has_value()) {
                    bubble_obj["outline"] = {
                        {"color", {
                            static_cast<int>(d.outline->color[0]),
                            static_cast<int>(d.outline->color[1]),
                            static_cast<int>(d.outline->color[2]),
                        }},
                        {"width", d.outline->width},
                    };
                }
                bubbles_json.push_back(std::move(bubble_obj));
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
                      "Japanese manga-ocr ONNX weights are missing. Download or import local models via Settings -> Pipeline first."}},
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
                if (pipeline::is_vram_tight()) {
                    if (ctx.detector) ctx.detector->unload();
                    if (ctx.detector_v2) ctx.detector_v2->unload();
                    if (ctx.inpainter) ctx.inpainter->unload();
                }
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
                {"neighbor_blanked_blocks", ocr_res.neighbor_blanked_blocks},
                {"low_confidence_ocr_ids", ocr_res.low_confidence_ocr_ids},
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
            if (req.form.has_field("enable_3tier_inpaint")) {
                opts.enable_3tier_inpaint = parse_bool_param(
                    req.form.get_field("enable_3tier_inpaint"), opts.enable_3tier_inpaint);
            }
            if (req.form.has_field("enable_tone_match")) {
                opts.enable_tone_match = parse_bool_param(
                    req.form.get_field("enable_tone_match"), opts.enable_tone_match);
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
        if (!opts.enable_3tier_inpaint && !has_inpaint_weights) {
            send_json(
                res,
                {{"detail",
                  "Requested Inpainting ONNX weights ('" + opts.inpainting_model +
                      "') are missing. Download or import local models via Settings -> Pipeline first."}},
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
            // Unify with /api/process-page: build tight bubble-wall-protected text-stroke mask
            std::vector<vellum::detector::BubbleDetection> synth_dets;
            synth_dets.reserve(selected_bboxes->size());
            int det_id = 0;
            for (const auto& bx : *selected_bboxes) {
                vellum::detector::BubbleDetection d;
                d.id = det_id++;
                d.class_id = vellum::detector::ClassId::SpeechBubble;
                d.bbox = vellum::detector::BBox{
                    static_cast<float>(bx[0]),
                    static_cast<float>(bx[1]),
                    static_cast<float>(bx[2]),
                    static_cast<float>(bx[3])};
                synth_dets.push_back(std::move(d));
            }
            working_mask = vellum::detector::buildTextStrokeInpaintMask(
                rgb_image.data(), orig_w, orig_h, synth_dets);
            if (working_mask.size() != static_cast<size_t>(orig_w) * static_cast<size_t>(orig_h)) {
                working_mask = pipeline::LamaInpainter::synthesize_mask_from_bboxes(*selected_bboxes, orig_w, orig_h);
            }
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
                if (pipeline::is_vram_tight()) {
                    if (ctx.detector) ctx.detector->unload();
                    if (ctx.detector_v2) ctx.detector_v2->unload();
                    if (ctx.ocr_engine) ctx.ocr_engine->unload();
                }
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
                {"tier1_blocks", inpaint_res.tier1_blocks},
                {"tier2_blocks", inpaint_res.tier2_blocks},
                {"tier3_blocks", inpaint_res.tier3_blocks},
                {"gpu_skipped", inpaint_res.gpu_skipped},
                {"tone_match_applied", inpaint_res.tone_match_applied},
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
    // Manual Spot-Remove Endpoint (`POST /api/inpaint/region`)
    // Accepts `{ page_id / image, mask_rect, mask_png_base64 }`, runs 3-Tier +
    // Tone Match on that region, and composites with a 4 px Euclidean feather
    // (Felzenszwalb DT) returning the updated cleaned image URL.
    // -------------------------------------------------------------------------

    server.Post("/api/inpaint/region", [&ctx](const httplib::Request& req, httplib::Response& res) {
        if (req.body.empty()) {
            send_json(res, {{"detail", "Empty JSON request body for /api/inpaint/region."}}, 400);
            return;
        }
        const json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.is_object()) {
            send_json(res, {{"detail", "Request body must be a valid JSON object."}}, 400);
            return;
        }

        // 1. Resolve source image from `image`, `image_url`, `cleaned_image_url`, `image_id`, or `page_id`
        std::string img_ref;
        for (const char* key : {"image", "image_url", "cleaned_image_url", "inpainted_image_url", "image_id", "page_id"}) {
            if (body.contains(key) && body[key].is_string()) {
                const std::string v = util::trim_ascii(body[key].get<std::string>());
                if (!v.empty()) {
                    img_ref = v;
                    break;
                }
            }
        }
        if (img_ref.empty()) {
            send_json(res, {{"detail", "Provide 'page_id' or 'image' (cached image ID, URL, or base64 data URI)."}}, 400);
            return;
        }

        int orig_w = 0;
        int orig_h = 0;
        std::vector<uint8_t> rgb_image;
        try {
            const std::vector<uint8_t> img_bytes = decode_export_image_reference(img_ref, 1);
            rgb_image = pipeline::LamaInpainter::decode_rgb_from_bytes(
                img_bytes.data(), img_bytes.size(), orig_w, orig_h);
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", std::string("Failed to resolve source image: ") + ex.what()}}, 400);
            return;
        }

        // 2. Parse optional `mask_rect` (`[x1, y1, x2, y2]` or `{x, y, width/w, height/h}` or `{x1, y1, x2, y2}`)
        std::optional<std::array<int, 4>> mask_rect;
        if (body.contains("mask_rect") && !body["mask_rect"].is_null()) {
            const auto& mr = body["mask_rect"];
            if (mr.is_array() && mr.size() >= 4) {
                int x1 = static_cast<int>(std::llround(mr[0].get<double>()));
                int y1 = static_cast<int>(std::llround(mr[1].get<double>()));
                int x2 = static_cast<int>(std::llround(mr[2].get<double>()));
                int y2 = static_cast<int>(std::llround(mr[3].get<double>()));
                if (x2 < x1) std::swap(x1, x2);
                if (y2 < y1) std::swap(y1, y2);
                x1 = std::max(0, std::min(orig_w, x1));
                y1 = std::max(0, std::min(orig_h, y1));
                x2 = std::max(0, std::min(orig_w, x2));
                y2 = std::max(0, std::min(orig_h, y2));
                if (x2 > x1 && y2 > y1) {
                    mask_rect = std::array<int, 4>{x1, y1, x2, y2};
                }
            } else if (mr.is_object()) {
                int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
                if (mr.contains("x1") && mr.contains("y1") && mr.contains("x2") && mr.contains("y2")) {
                    x1 = static_cast<int>(std::llround(mr["x1"].get<double>()));
                    y1 = static_cast<int>(std::llround(mr["y1"].get<double>()));
                    x2 = static_cast<int>(std::llround(mr["x2"].get<double>()));
                    y2 = static_cast<int>(std::llround(mr["y2"].get<double>()));
                } else if (mr.contains("x") && mr.contains("y") &&
                           (mr.contains("width") || mr.contains("w")) &&
                           (mr.contains("height") || mr.contains("h"))) {
                    x1 = static_cast<int>(std::llround(mr["x"].get<double>()));
                    y1 = static_cast<int>(std::llround(mr["y"].get<double>()));
                    const int rw = static_cast<int>(std::llround(
                        (mr.contains("width") ? mr["width"] : mr["w"]).get<double>()));
                    const int rh = static_cast<int>(std::llround(
                        (mr.contains("height") ? mr["height"] : mr["h"]).get<double>()));
                    x2 = x1 + std::max(0, rw);
                    y2 = y1 + std::max(0, rh);
                } else {
                    send_json(res, {{"detail", "Invalid 'mask_rect' object format."}}, 400);
                    return;
                }
                if (x2 < x1) std::swap(x1, x2);
                if (y2 < y1) std::swap(y1, y2);
                x1 = std::max(0, std::min(orig_w, x1));
                y1 = std::max(0, std::min(orig_h, y1));
                x2 = std::max(0, std::min(orig_w, x2));
                y2 = std::max(0, std::min(orig_h, y2));
                if (x2 > x1 && y2 > y1) {
                    mask_rect = std::array<int, 4>{x1, y1, x2, y2};
                }
            } else {
                send_json(res, {{"detail", "'mask_rect' must be a 4-element array or object."}}, 400);
                return;
            }
        }

        // 3. Parse optional `mask_png_base64` (supports both white-on-black grayscale and RGBA brush PNGs)
        std::vector<uint8_t> region_mask(static_cast<size_t>(orig_w) * static_cast<size_t>(orig_h), 0);
        bool has_mask_png = false;
        if (body.contains("mask_png_base64") && body["mask_png_base64"].is_string()) {
            std::string b64 = util::trim_ascii(body["mask_png_base64"].get<std::string>());
            if (!b64.empty()) {
                if (b64.size() >= 5 && util::to_lower_ascii(b64.substr(0, 5)) == "data:") {
                    const size_t comma = b64.find(',');
                    if (comma != std::string::npos) {
                        b64 = b64.substr(comma + 1);
                    }
                }
                const auto decoded = util::base64_decode(b64);
                if (!decoded.has_value() || decoded->empty()) {
                    send_json(res, {{"detail", "Failed to decode 'mask_png_base64': invalid Base64 data."}}, 400);
                    return;
                }
                int mw = 0, mh = 0, mcomp = 0;
                unsigned char* rgba = stbi_load_from_memory(
                    decoded->data(), static_cast<int>(decoded->size()), &mw, &mh, &mcomp, 4);
                if (!rgba || mw <= 0 || mh <= 0) {
                    if (rgba) stbi_image_free(rgba);
                    send_json(res, {{"detail", "Failed to decode PNG from 'mask_png_base64'."}}, 400);
                    return;
                }

                // Detect whether mask uses alpha channel (transparent brush overlay) or luminance
                bool has_transparency = false;
                const size_t mtotal = static_cast<size_t>(mw) * static_cast<size_t>(mh);
                if (mcomp == 2 || mcomp == 4) {
                    for (size_t i = 0; i < mtotal; ++i) {
                        if (rgba[i * 4 + 3] < 250) {
                            has_transparency = true;
                            break;
                        }
                    }
                }
                std::vector<uint8_t> decoded_bin(mtotal, 0);
                for (size_t i = 0; i < mtotal; ++i) {
                    const uint8_t r = rgba[i * 4 + 0];
                    const uint8_t g = rgba[i * 4 + 1];
                    const uint8_t b = rgba[i * 4 + 2];
                    const uint8_t a = rgba[i * 4 + 3];
                    if (has_transparency) {
                        decoded_bin[i] = (a > 127) ? 255 : 0;
                    } else {
                        const int luma = (77 * r + 150 * g + 29 * b) >> 8;
                        decoded_bin[i] = (luma > 127) ? 255 : 0;
                    }
                }
                stbi_image_free(rgba);

                if (mw == orig_w && mh == orig_h) {
                    region_mask = std::move(decoded_bin);
                    if (mask_rect.has_value()) {
                        pipeline::LamaInpainter::restrict_mask_to_bboxes(
                            region_mask, {*mask_rect}, orig_w, orig_h, 0);
                    }
                } else if (mask_rect.has_value()) {
                    const int rx1 = (*mask_rect)[0];
                    const int ry1 = (*mask_rect)[1];
                    const int rw = (*mask_rect)[2] - rx1;
                    const int rh = (*mask_rect)[3] - ry1;
                    const float sx_ratio = static_cast<float>(mw) / static_cast<float>(rw);
                    const float sy_ratio = static_cast<float>(mh) / static_cast<float>(rh);
                    for (int ry = 0; ry < rh; ++ry) {
                        const int sy = std::max(0, std::min(mh - 1, static_cast<int>(std::floor((ry + 0.5f) * sy_ratio))));
                        uint8_t* drow = region_mask.data() + static_cast<size_t>(ry1 + ry) * static_cast<size_t>(orig_w) + static_cast<size_t>(rx1);
                        const uint8_t* srow = decoded_bin.data() + static_cast<size_t>(sy) * static_cast<size_t>(mw);
                        for (int rx = 0; rx < rw; ++rx) {
                            const int sx = std::max(0, std::min(mw - 1, static_cast<int>(std::floor((rx + 0.5f) * sx_ratio))));
                            drow[rx] = srow[sx];
                        }
                    }
                } else {
                    const float sx_ratio = static_cast<float>(mw) / static_cast<float>(orig_w);
                    const float sy_ratio = static_cast<float>(mh) / static_cast<float>(orig_h);
                    for (int y = 0; y < orig_h; ++y) {
                        const int sy = std::max(0, std::min(mh - 1, static_cast<int>(std::floor((y + 0.5f) * sy_ratio))));
                        uint8_t* drow = region_mask.data() + static_cast<size_t>(y) * static_cast<size_t>(orig_w);
                        const uint8_t* srow = decoded_bin.data() + static_cast<size_t>(sy) * static_cast<size_t>(mw);
                        for (int x = 0; x < orig_w; ++x) {
                            const int sx = std::max(0, std::min(mw - 1, static_cast<int>(std::floor((x + 0.5f) * sx_ratio))));
                            drow[x] = srow[sx];
                        }
                    }
                }
                has_mask_png = true;
            }
        }

        if (!has_mask_png) {
            if (!mask_rect.has_value()) {
                send_json(res, {{"detail", "Provide 'mask_rect' and/or 'mask_png_base64'."}}, 400);
                return;
            }
            region_mask = pipeline::LamaInpainter::synthesize_mask_from_bboxes({*mask_rect}, orig_w, orig_h);
        }

        // Compute tight active ROI bounds of `region_mask`
        int min_x = orig_w, min_y = orig_h, max_x = -1, max_y = -1;
        for (int y = 0; y < orig_h; ++y) {
            const uint8_t* mrow = region_mask.data() + static_cast<size_t>(y) * static_cast<size_t>(orig_w);
            for (int x = 0; x < orig_w; ++x) {
                if (mrow[x] > 0) {
                    if (x < min_x) min_x = x;
                    if (x > max_x) max_x = x;
                    if (y < min_y) min_y = y;
                    if (y > max_y) max_y = y;
                }
            }
        }
        if (max_x < min_x || max_y < min_y) {
            send_json(res, {{"detail", "Provided region mask contains no active pixels."}}, 400);
            return;
        }
        const std::array<int, 4> active_roi{min_x, min_y, max_x + 1, max_y + 1};

        // 4. Configure 3-Tier + Tone Match options
        const json current_settings = ctx.settings_manager.load();
        pipeline::InpaintOptions opts;
        populate_inpaint_options_from_settings(opts, current_settings);
        opts.bboxes = std::vector<std::array<int, 4>>{active_roi};
        opts.refine_mask = false;  // Respect exact manual brush / spot-remove mask
        try {
            if (body.contains("inpainting_model") && body["inpainting_model"].is_string()) {
                const std::string im = util::trim_ascii(body["inpainting_model"].get<std::string>());
                if (!im.empty()) {
                    opts.inpainting_model = SettingsManager::normalize_inpainting_model(im);
                }
            }
            if (body.contains("enable_3tier_inpaint") && body["enable_3tier_inpaint"].is_boolean()) {
                opts.enable_3tier_inpaint = body["enable_3tier_inpaint"].get<bool>();
            }
            if (body.contains("enable_tone_match") && body["enable_tone_match"].is_boolean()) {
                opts.enable_tone_match = body["enable_tone_match"].get<bool>();
            }
        } catch (const std::invalid_argument& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
            return;
        }

        const bool include_base64 = body.value("include_base64", false);
        constexpr float kFeatherPx = 4.0f;

        try {
            ctx.cancel_pending_unload();
            pipeline::InpaintResult inpaint_res;
            {
                std::lock_guard<std::mutex> lock(ctx.inference_mutex);
                inpaint_res = ctx.inpainter->inpaint_rgb(rgb_image.data(), orig_w, orig_h, region_mask, opts);
            }

            // 5. Composite with a 4 px Euclidean feather (Felzenszwalb DT)
            std::vector<uint8_t> composited = rgb_image;
            pipeline::LamaInpainter::composite_feathered_region(
                composited, inpaint_res.rgb, region_mask, orig_w, orig_h, active_roi, kFeatherPx);

            const auto png_bytes = pipeline::LamaInpainter::encode_rgb_png(composited, orig_w, orig_h);
            const auto [cleaned_id, cleaned_url] = save_inpainted_png_to_cache(png_bytes);
            const std::string image_payload = include_base64
                ? ("data:image/png;base64," +
                   pipeline::TextDetector::base64_encode(png_bytes.data(), png_bytes.size()))
                : cleaned_url;

            send_json(res, {
                {"dimensions", {{"width", orig_w}, {"height", orig_h}}},
                {"mask_rect", json::array({active_roi[0], active_roi[1], active_roi[2], active_roi[3]})},
                {"feather_px", kFeatherPx},
                {"execution_provider", inpaint_res.execution_provider},
                {"inpainting_model", inpaint_res.model_id},
                {"tiles_inpainted", inpaint_res.tiles_processed},
                {"tier1_blocks", inpaint_res.tier1_blocks},
                {"tier2_blocks", inpaint_res.tier2_blocks},
                {"tier3_blocks", inpaint_res.tier3_blocks},
                {"gpu_skipped", inpaint_res.gpu_skipped},
                {"tone_match_applied", inpaint_res.tone_match_applied},
                {"cleaned_image_id", cleaned_id},
                {"cleaned_image_url", cleaned_url},
                {"inpainted_image_url", cleaned_url},
                {"cleaned_image", image_payload},
                {"inpainted_image", image_payload},
            });
        } catch (const std::invalid_argument& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", std::string("Region inpainting failed: ") + ex.what()}}, 500);
        }
    });

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
                         "honorifics_policy",
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
                pipeline::DialogueBlockInput di;
                di.id = item.id;
                di.source_text = item.original_text;
                di.bubble_class = item.bubble_class;
                di.panel_id = item.panel_id;
                di.speaker_id = item.speaker_id;
                dialogue_inputs.push_back(std::move(di));
            }

            pipeline::TranslateRequestOptions req_opts;
            req_opts.source_lang = src_lang;
            req_opts.target_lang = tgt_lang;
            req_opts.page_image_data_uri = page_image_data_uri;
            req_opts.use_vision = translator.config().use_vision;
            if (req.is_multipart_form_data()) {
                if (req.form.has_field("project_id")) {
                    req_opts.project_id = util::trim_ascii(req.form.get_field("project_id"));
                }
                if (req.form.has_field("use_history")) {
                    req_opts.use_history = parse_bool_param(req.form.get_field("use_history"), true);
                }
            } else {
                const json req_body = json::parse(req.body, nullptr, false);
                if (!req_body.is_discarded() && req_body.is_object()) {
                    if (req_body.contains("project_id") && req_body["project_id"].is_string()) {
                        req_opts.project_id = util::trim_ascii(req_body["project_id"].get<std::string>());
                    }
                    if (req_body.contains("use_history") && req_body["use_history"].is_boolean()) {
                        req_opts.use_history = req_body["use_history"].get<bool>();
                    }
                }
            }

            const auto translation_res = translator.translate_blocks(dialogue_inputs, req_opts);
            json translated_blocks = build_translated_blocks_json(
                input_items, translation_res.translations);

            json dialogues_json = json::array();
            for (const auto& d : translation_res.dialogues) {
                dialogues_json.push_back(d.to_json());
            }

            send_json(res, {
                {"status", translation_res.status},
                {"provider", translation_res.provider},
                {"model", translation_res.model},
                {"vision_used", translation_res.vision_used},
                {"echo_retried", translation_res.echo_retried},
                {"history_compacted", translation_res.history_compacted},
                {"leaked_block_ids", translation_res.leaked_block_ids},
                {"anomaly_block_ids", translation_res.anomaly_block_ids},
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
                     "honorifics_policy",
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
                pipeline::DialogueBlockInput di;
                di.id = item.id;
                di.source_text = item.original_text;
                di.bubble_class = item.bubble_class;
                di.panel_id = item.panel_id;
                di.speaker_id = item.speaker_id;
                dialogue_inputs.push_back(std::move(di));
            }

            pipeline::TranslateRequestOptions req_opts;
            req_opts.source_lang = src_lang;
            req_opts.target_lang = tgt_lang;
            req_opts.page_image_data_uri = page_image_data_uri;
            req_opts.use_vision = translator.config().use_vision;
            if (body.contains("project_id") && body["project_id"].is_string()) {
                req_opts.project_id = util::trim_ascii(body["project_id"].get<std::string>());
            }
            if (body.contains("use_history") && body["use_history"].is_boolean()) {
                req_opts.use_history = body["use_history"].get<bool>();
            }

            const auto translation_res = translator.translate_blocks(dialogue_inputs, req_opts);
            json retranslated_blocks = build_translated_blocks_json(
                input_items, translation_res.translations);

            json dialogues_json = json::array();
            for (const auto& d : translation_res.dialogues) {
                dialogues_json.push_back(d.to_json());
            }

            send_json(res, {
                {"status", translation_res.status},
                {"provider", translation_res.provider},
                {"model", translation_res.model},
                {"vision_used", translation_res.vision_used},
                {"echo_retried", translation_res.echo_retried},
                {"history_compacted", translation_res.history_compacted},
                {"leaked_block_ids", translation_res.leaked_block_ids},
                {"anomaly_block_ids", translation_res.anomaly_block_ids},
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
        if (det_sel_lower.empty() || det_sel_lower == "auto" ||
            det_sel_lower.find("yolo") != std::string::npos ||
            det_sel_lower.find("rfdetr") != std::string::npos) {
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
                send_json(res, {{"detail", "Requested detector ONNX weights ('" + det_selector + "') are missing. Download or import local models via Settings -> Pipeline first."}}, 503);
                return;
            }
            target_det_backend = vellum::detector::BackendType::LegacyCtd;
            resolved_det_model_path = util::path_to_utf8(legacy_det->model_path);
        }
        if (effective_ocr_model == "manga-ocr") {
            if (!ctx.model_manager->resolve_manga_ocr_models().has_value()) {
                send_json(res, {{"detail", "Japanese manga-ocr ONNX weights are missing. Download or import local models via Settings -> Pipeline first."}}, 503);
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
            send_json(res, {{"detail", "Requested Inpainting ONNX weights ('" + inpaint_opts.inpainting_model + "') are missing. Download or import local models via Settings -> Pipeline first."}}, 503);
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
        std::unordered_map<std::string, size_t> det_index_by_id;
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
                v2_cfg.enable_cv_style_detect = current_settings.value("enable_cv_style_detect", true);
                v2_cfg.enable_waist_cut = current_settings.value("enable_waist_cut", true);
                v2_cfg.auto_deskew_scans = current_settings.value("auto_deskew_scans", false);
                {
                    std::string auto_prec = current_settings.value("detector_precision", std::string("balanced"));
                    if (req.form.has_field("detector_precision")) {
                        const std::string dp = util::trim_ascii(req.form.get_field("detector_precision"));
                        if (!dp.empty()) {
                            auto_prec = SettingsManager::normalize_detector_precision(dp);
                        }
                    }
                    v2_cfg.precision = vellum::detector::parse_detector_precision(auto_prec);
                }
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

                for (size_t di = 0; di < v2_det_result.detections.size(); ++di) {
                    const auto& d = v2_det_result.detections[di];
                    if (d.class_id == vellum::detector::ClassId::SfxIgnore) {
                        continue;
                    }
                    const std::string bid = "b_" + std::to_string(d.id);
                    det_index_by_id[bid] = di;
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

                ocr_opts.page_glyph_mask = &page_mask;
                ocr_opts.page_glyph_mask_w = orig_w;
                ocr_opts.page_glyph_mask_h = orig_h;
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
                    pipeline::DialogueBlockInput di;
                    di.id = b.id;
                    di.source_text = b.source_text;
                    const auto dit = det_index_by_id.find(b.id);
                    if (dit != det_index_by_id.end() && dit->second < v2_det_result.detections.size()) {
                        const auto& det = v2_det_result.detections[dit->second];
                        di.bubble_class = det.class_name;
                        di.panel_id = det.panel_id;
                        di.speaker_id = det.speaker_cluster_id;
                    }
                    async_dialogues.push_back(std::move(di));
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
                if (req.form.has_field("project_id")) {
                    async_req_opts.project_id = util::trim_ascii(req.form.get_field("project_id"));
                } else if (translation_overrides.contains("project_id") &&
                           translation_overrides["project_id"].is_string()) {
                    async_req_opts.project_id = util::trim_ascii(
                        translation_overrides["project_id"].get<std::string>());
                }
                if (req.form.has_field("use_history")) {
                    async_req_opts.use_history = parse_bool_param(req.form.get_field("use_history"), true);
                } else if (translation_overrides.contains("use_history") &&
                           translation_overrides["use_history"].is_boolean()) {
                    async_req_opts.use_history = translation_overrides["use_history"].get<bool>();
                }

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

            json block_obj = {
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
            };
            if (const auto dit = det_index_by_id.find(b.id);
                dit != det_index_by_id.end() && dit->second < v2_det_result.detections.size()) {
                const auto& d = v2_det_result.detections[dit->second];
                block_obj["speaker_id"] = d.speaker_cluster_id;
                if (!d.room_polygon.empty()) {
                    json rpoly = json::array();
                    for (const auto& pt : d.room_polygon) {
                        rpoly.push_back({pt.x, pt.y});
                    }
                    block_obj["room_polygon"] = std::move(rpoly);
                }
                if (d.angle_deg.has_value()) {
                    block_obj["angle_deg"] = *d.angle_deg;
                }
                if (d.estimated_font_size.has_value()) {
                    block_obj["estimated_font_size"] = *d.estimated_font_size;
                }
                if (d.text_color.has_value()) {
                    block_obj["text_color"] = {
                        static_cast<int>((*d.text_color)[0]),
                        static_cast<int>((*d.text_color)[1]),
                        static_cast<int>((*d.text_color)[2]),
                    };
                }
                if (d.outline.has_value()) {
                    block_obj["outline"] = {
                        {"color", {
                            static_cast<int>(d.outline->color[0]),
                            static_cast<int>(d.outline->color[1]),
                            static_cast<int>(d.outline->color[2]),
                        }},
                        {"width", d.outline->width},
                    };
                }
            }
            processed_blocks.push_back(std::move(block_obj));
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
            {"translation_status", translation_res.status},
            {"echo_retried", translation_res.echo_retried},
            {"history_compacted", translation_res.history_compacted},
            {"leaked_block_ids", translation_res.leaked_block_ids},
            {"anomaly_block_ids", translation_res.anomaly_block_ids},
            {"low_confidence_ocr_ids", ocr_result.low_confidence_ocr_ids},
            {"model", translation_res.model},
            {"vision_used", translation_res.vision_used},
            {"reasoning_effort", translation_res.reasoning_effort},
            {"ocr_engine", ocr_result.engine},
            {"ocr_model", ocr_result.ocr_model},
            {"inpainting_model", inpaint_result.model_id},
            {"tiles_inpainted", inpaint_result.tiles_processed},
            {"tier1_blocks", inpaint_result.tier1_blocks},
            {"tier2_blocks", inpaint_result.tier2_blocks},
            {"tier3_blocks", inpaint_result.tier3_blocks},
            {"gpu_skipped", inpaint_result.gpu_skipped},
            {"tone_match_applied", inpaint_result.tone_match_applied},
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

}

}  // namespace mangaflow::routes
