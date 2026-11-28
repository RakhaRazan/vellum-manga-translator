#include "routes_system.hpp"
#include "../utils.hpp"
#include "../pipeline/ort_provider.hpp"

#include <windows.h>
#include <shobjidl.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace mangaflow::routes {

void register_system_routes(httplib::Server& server, AppContext& ctx) {
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
        const std::string sel_det = SettingsManager::normalize_detection_model(
            current_settings.value("detection_model", std::string("vellum-det-yolo11m-seg-1024-fp16")));
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

    server.Get("/api/system/hardware", [&ctx](const httplib::Request&, httplib::Response& res) {
        const json current_settings = ctx.settings_manager.load();
        const std::string exec_mode = current_settings.value("execution_mode", std::string("hybrid"));
        const std::string pref_backend = current_settings.value("preferred_backend", std::string("auto"));
        pipeline::set_preferred_backend(pref_backend);
        const bool cpu_only = (exec_mode == "cpu_only" || exec_mode == "cpu");
        const auto caps = pipeline::detect_hardware_capabilities(cpu_only);
        json payload = pipeline::hardware_capabilities_to_json(caps);
        payload["execution_mode"] = exec_mode;
        payload["onnxruntime_version"] = ctx.ort_version;
        send_json(res, payload);
    });

    server.Post("/api/system/backend", [&ctx](const httplib::Request& req, httplib::Response& res) {
        const json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.is_object()) {
            send_json(res, {{"detail", "Invalid JSON object in backend request body."}}, 400);
            return;
        }
        try {
            json patch = json::object();
            if (body.contains("preferred_backend") && !body["preferred_backend"].is_null()) {
                patch["preferred_backend"] = pipeline::normalize_preferred_backend(
                    body["preferred_backend"].get<std::string>());
            } else if (body.contains("backend") && !body["backend"].is_null()) {
                patch["preferred_backend"] = pipeline::normalize_preferred_backend(
                    body["backend"].get<std::string>());
            }
            if (body.contains("execution_mode") && !body["execution_mode"].is_null()) {
                patch["execution_mode"] = body["execution_mode"];
            }
            const json updated = ctx.settings_manager.update(patch);
            const std::string exec_mode = updated.value("execution_mode", std::string("hybrid"));
            const std::string pref_backend = updated.value("preferred_backend", std::string("auto"));
            pipeline::set_preferred_backend(pref_backend);

            // Cleanly unload all cached ONNX sessions so the next run initializes on the new backend
            ctx.cancel_pending_unload();
            ctx.unload_all_sessions();

            const bool cpu_only = (exec_mode == "cpu_only" || exec_mode == "cpu");
            const auto caps = pipeline::detect_hardware_capabilities(cpu_only);
            const bool has_dml_in_proc =
                std::find(caps.available_providers.begin(), caps.available_providers.end(), "DmlExecutionProvider") !=
                caps.available_providers.end();
            const bool has_cuda_in_proc =
                std::find(caps.available_providers.begin(), caps.available_providers.end(), "CUDAExecutionProvider") !=
                caps.available_providers.end();
            const bool requires_relaunch =
                (pref_backend == "directml" && !has_dml_in_proc && caps.directml_available) ||
                (pref_backend == "cuda" && !has_cuda_in_proc && caps.cuda_available) ||
                (pref_backend == "auto" && !has_cuda_in_proc && caps.cuda_available && caps.cuda_driver_present);

            json payload = pipeline::hardware_capabilities_to_json(caps);
            payload["execution_mode"] = exec_mode;
            payload["onnxruntime_version"] = ctx.ort_version;
            payload["requires_relaunch"] = requires_relaunch;
            payload["ok"] = true;
            send_json(res, payload);
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        }
    });

    server.Post("/api/system/relaunch", [&ctx](const httplib::Request&, httplib::Response& res) {
        send_json(res, {{"ok", true}, {"relaunching", true}});
        std::thread([]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            ExitProcess(42);
        }).detach();
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

    server.Post("/api/system/unload-stage", [&ctx](const httplib::Request& req, httplib::Response& res) {
        std::string stage = "all";
        if (!util::trim_ascii(req.body).empty()) {
            const json body = json::parse(req.body, nullptr, false);
            if (body.is_discarded() || !body.is_object()) {
                send_json(res, {{"detail", "Invalid JSON object in /api/system/unload-stage request body."}}, 400);
                return;
            }
            if (body.contains("stage") && body["stage"].is_string()) {
                stage = util::to_lower_ascii(util::trim_ascii(body["stage"].get<std::string>()));
            }
        }

        if (stage != "detect" && stage != "ocr" && stage != "inpaint" && stage != "all") {
            send_json(res, {
                {"detail", "Invalid stage '" + stage + "'. Expected one of: detect, ocr, inpaint, all."},
            }, 400);
            return;
        }

        ctx.cancel_pending_unload();
        {
            std::lock_guard<std::mutex> lock(ctx.inference_mutex);
            if (stage == "detect") {
                if (ctx.detector) ctx.detector->unload();
                if (ctx.detector_v2) ctx.detector_v2->unload();
                pipeline::ModelManager::trim_process_heap();
            } else if (stage == "ocr") {
                if (ctx.ocr_engine) ctx.ocr_engine->unload();
                pipeline::ModelManager::trim_process_heap();
            } else if (stage == "inpaint") {
                if (ctx.inpainter) ctx.inpainter->unload();
                pipeline::ModelManager::trim_process_heap();
            } else {
                ctx.unload_all_sessions_unlocked();
            }
        }

        const uint64_t free_vram_mb = pipeline::query_free_vram_mb();
        send_json(res, {
            {"ok", true},
            {"stage", stage},
            {"free_vram_mb", free_vram_mb},
        });
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

    server.Get("/api/lmstudio/status", [&ctx](const httplib::Request&, httplib::Response& res) {
        send_json(res, probe_lm_studio(ctx.settings_manager, 5.0));
    });

    server.Get("/api/providers", [&ctx](const httplib::Request&, httplib::Response& res) {
        const json current = ctx.settings_manager.load();
        send_json(res, {
            {"active_provider", current.value("active_provider", current.value("provider", std::string("none")))},
            {"providers", SettingsManager::list_provider_presets()},
            {"supported_reasoning_efforts", SettingsManager::list_reasoning_efforts()},
            {"target_languages", SettingsManager::list_target_languages()},
        });
    });

    auto resolve_provider_request_profile = [&ctx](
        const json& body,
        std::string& prov_id,
        std::string& base_url,
        std::string& api_key,
        std::string& preferred_model,
        double& timeout_sec) {
        const json settings = ctx.settings_manager.load();
        const std::string active_provider = pipeline::ComicTranslator::normalize_provider_id(
            settings.value("active_provider", settings.value("provider", std::string("none"))));

        if (body.contains("provider") && body["provider"].is_string() &&
            !util::trim_ascii(body["provider"].get<std::string>()).empty()) {
            prov_id = pipeline::ComicTranslator::normalize_provider_id(
                body["provider"].get<std::string>());
        } else {
            prov_id = active_provider;
        }

        const auto& preset = pipeline::ComicTranslator::get_provider_spec(prov_id);
        const json* saved_profile = nullptr;
        if (settings.contains("providers") && settings["providers"].is_object() &&
            settings["providers"].contains(prov_id) && settings["providers"][prov_id].is_object()) {
            saved_profile = &settings["providers"][prov_id];
        }

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

        // API key is decrypted in memory only by SettingsManager/ComicTranslator; never logged
        if (body.contains("api_key") && body["api_key"].is_string() &&
            !util::trim_ascii(body["api_key"].get<std::string>()).empty()) {
            api_key = util::trim_ascii(body["api_key"].get<std::string>());
        } else if (prov_id == active_provider &&
                   settings.contains("api_key") && settings["api_key"].is_string() &&
                   !util::trim_ascii(settings["api_key"].get<std::string>()).empty()) {
            api_key = util::trim_ascii(settings["api_key"].get<std::string>());
        } else if (saved_profile && saved_profile->contains("api_key") &&
                   (*saved_profile)["api_key"].is_string() &&
                   !util::trim_ascii((*saved_profile)["api_key"].get<std::string>()).empty()) {
            api_key = util::trim_ascii((*saved_profile)["api_key"].get<std::string>());
        } else {
            api_key = preset.default_api_key;
        }

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

        timeout_sec = 10.0;
        if (body.contains("timeout") && body["timeout"].is_number()) {
            timeout_sec = std::max(1.0, std::min(60.0, body["timeout"].get<double>()));
        }
    };

    server.Post("/api/providers/test", [resolve_provider_request_profile](const httplib::Request& req, httplib::Response& res) {
        json body = json::object();
        if (!util::trim_ascii(req.body).empty()) {
            body = json::parse(req.body, nullptr, false);
            if (body.is_discarded() || !body.is_object()) {
                send_json(res, {{"detail", "Invalid JSON object in /api/providers/test request body."}}, 400);
                return;
            }
        }

        std::string prov_id, base_url, api_key, preferred_model;
        double timeout_sec = 10.0;
        try {
            resolve_provider_request_profile(body, prov_id, base_url, api_key, preferred_model, timeout_sec);
        } catch (const std::invalid_argument& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
            return;
        }

        const auto test_res = pipeline::ComicTranslator::test_provider_connection(
            prov_id, base_url, api_key, preferred_model, timeout_sec);
        send_json(res, test_res.to_json());
    });

    auto handle_provider_models = [resolve_provider_request_profile](const httplib::Request& req, httplib::Response& res) {
        json body = json::object();
        if (!util::trim_ascii(req.body).empty()) {
            body = json::parse(req.body, nullptr, false);
            if (body.is_discarded() || !body.is_object()) {
                send_json(res, {{"detail", "Invalid JSON object in /api/providers/models request body."}}, 400);
                return;
            }
        }
        if (req.has_param("provider") && !body.contains("provider")) {
            body["provider"] = req.get_param_value("provider");
        }
        if (req.has_param("base_url") && !body.contains("base_url")) {
            body["base_url"] = req.get_param_value("base_url");
        }

        std::string prov_id, base_url, api_key, preferred_model;
        double timeout_sec = 10.0;
        try {
            resolve_provider_request_profile(body, prov_id, base_url, api_key, preferred_model, timeout_sec);
        } catch (const std::invalid_argument& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
            return;
        }

        const auto& spec = pipeline::ComicTranslator::get_provider_spec(prov_id);
        const std::string resolved_url = pipeline::ComicTranslator::normalize_provider_base_url(prov_id, base_url);

        if (prov_id == "none") {
            send_json(res, {
                {"ok", false},
                {"status", "provider_not_configured"},
                {"provider", "none"},
                {"provider_name", spec.name},
                {"endpoint", ""},
                {"models", json::array()},
                {"vision_models", json::array()},
                {"model_count", 0},
                {"latency_ms", 0},
                {"message", "No translation provider configured (active_provider is 'none')."},
            });
            return;
        }

        const bool include_embeddings = body.value("include_embeddings", false);
        const auto t0 = std::chrono::steady_clock::now();
        try {
            const auto models = pipeline::ComicTranslator::list_provider_models(
                prov_id, resolved_url, api_key, timeout_sec, include_embeddings);
            std::vector<std::string> vision_models;
            for (const auto& m : models) {
                if (pipeline::ComicTranslator::is_likely_vision_model(m, prov_id)) {
                    vision_models.push_back(m);
                }
            }
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0);
            send_json(res, {
                {"ok", true},
                {"status", "ok"},
                {"provider", prov_id},
                {"provider_name", spec.name},
                {"endpoint", resolved_url},
                {"models", models},
                {"available_models", models},
                {"vision_models", vision_models},
                {"model_count", models.size()},
                {"latency_ms", static_cast<int>(elapsed.count())},
            });
        } catch (const pipeline::ProviderConnectionError& ex) {
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0);
            send_json(res, {
                {"ok", false},
                {"status", "error"},
                {"provider", prov_id},
                {"provider_name", spec.name},
                {"endpoint", resolved_url},
                {"models", json::array()},
                {"vision_models", json::array()},
                {"model_count", 0},
                {"latency_ms", static_cast<int>(elapsed.count())},
                {"error", ex.what()},
            });
        }
    };
    server.Post("/api/providers/models", handle_provider_models);
    server.Get("/api/providers/models", handle_provider_models);

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
            const json prev = ctx.settings_manager.load();
            const json updated = ctx.settings_manager.update(patch);
            const bool backend_changed =
                updated.value("preferred_backend", std::string("auto")) !=
                    prev.value("preferred_backend", std::string("auto")) ||
                updated.value("execution_mode", std::string("hybrid")) !=
                    prev.value("execution_mode", std::string("hybrid"));
            if (backend_changed || updated.value("auto_unload_vram", true)) {
                ctx.cancel_pending_unload();
                ctx.unload_all_sessions();
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

    // Native Win32 IFileOpenDialog / IFileSaveDialog Path Picker (`POST /api/system/pick-paths`)
    server.Post("/api/system/pick-paths", [](const httplib::Request& req, httplib::Response& res) {
        std::string mode = "files";
        std::string default_name;
        std::string default_ext;
        std::string dialog_title;
        if (!util::trim_ascii(req.body).empty()) {
            const json body = json::parse(req.body, nullptr, false);
            if (!body.is_discarded() && body.is_object()) {
                if (body.contains("mode") && body["mode"].is_string()) {
                    const std::string m = util::to_lower_ascii(util::trim_ascii(body["mode"].get<std::string>()));
                    if (m == "folder" || m == "files" || m == "save_file") {
                        mode = m;
                    }
                }
                if (body.contains("default_name") && body["default_name"].is_string()) {
                    default_name = util::trim_ascii(body["default_name"].get<std::string>());
                }
                if (body.contains("default_ext") && body["default_ext"].is_string()) {
                    default_ext = util::to_lower_ascii(util::trim_ascii(body["default_ext"].get<std::string>()));
                    if (!default_ext.empty() && default_ext.front() == '.') {
                        default_ext.erase(default_ext.begin());
                    }
                }
                if (body.contains("title") && body["title"].is_string()) {
                    dialog_title = util::trim_ascii(body["title"].get<std::string>());
                }
            }
        }

        std::vector<std::string> picked;
        std::string saved_path;
        std::string saved_dir;
        std::string saved_filename;
        bool cancelled = false;
        std::string err_msg;

        auto run_dialog = [&]() {
            const HRESULT hr_com = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
            const bool should_uninit = SUCCEEDED(hr_com);
            if (FAILED(hr_com) && hr_com != RPC_E_CHANGED_MODE) {
                err_msg = "Failed to initialize COM for native file dialog.";
                return;
            }
            struct ComGuard {
                bool active = false;
                ~ComGuard() { if (active) CoUninitialize(); }
            } com_guard{should_uninit};

            if (mode == "save_file") {
                IFileSaveDialog* save_dlg = nullptr;
                HRESULT hr = CoCreateInstance(
                    CLSID_FileSaveDialog, nullptr, CLSCTX_ALL, IID_PPV_ARGS(&save_dlg));
                if (FAILED(hr) || !save_dlg) {
                    err_msg = "Failed to create Win32 IFileSaveDialog.";
                    return;
                }
                struct SaveDlgGuard {
                    IFileSaveDialog* p = nullptr;
                    ~SaveDlgGuard() { if (p) p->Release(); }
                } save_guard{save_dlg};

                DWORD opts = 0;
                if (SUCCEEDED(save_dlg->GetOptions(&opts))) {
                    opts |= FOS_FORCEFILESYSTEM | FOS_OVERWRITEPROMPT | FOS_PATHMUSTEXIST | FOS_NOCHANGEDIR;
                    save_dlg->SetOptions(opts);
                }

                const std::wstring wtitle = dialog_title.empty()
                    ? L"Save Export File To..."
                    : util::utf8_to_wide(dialog_title);
                save_dlg->SetTitle(wtitle.c_str());

                std::wstring filter_label = L"All Files (*.*)";
                std::wstring filter_spec = L"*.*";
                if (default_ext == "cbz") {
                    filter_label = L"CBZ Comic Archive (*.cbz)";
                    filter_spec = L"*.cbz";
                } else if (default_ext == "pdf") {
                    filter_label = L"PDF Document (*.pdf)";
                    filter_spec = L"*.pdf";
                } else if (default_ext == "psd") {
                    filter_label = L"Photoshop Document (*.psd)";
                    filter_spec = L"*.psd";
                } else if (default_ext == "zip") {
                    filter_label = L"ZIP Archive (*.zip)";
                    filter_spec = L"*.zip";
                } else if (default_ext == "png") {
                    filter_label = L"PNG Image (*.png)";
                    filter_spec = L"*.png";
                } else if (default_ext == "jpg" || default_ext == "jpeg") {
                    filter_label = L"JPEG Image (*.jpg;*.jpeg)";
                    filter_spec = L"*.jpg;*.jpeg";
                }
                if (!default_ext.empty()) {
                    const COMDLG_FILTERSPEC filters[] = {
                        {filter_label.c_str(), filter_spec.c_str()},
                        {L"All Files (*.*)", L"*.*"},
                    };
                    save_dlg->SetFileTypes(2, filters);
                    save_dlg->SetFileTypeIndex(1);
                    const std::wstring wext = util::utf8_to_wide(default_ext);
                    save_dlg->SetDefaultExtension(wext.c_str());
                }
                if (!default_name.empty()) {
                    const std::wstring wname = util::utf8_to_wide(default_name);
                    save_dlg->SetFileName(wname.c_str());
                }

                hr = save_dlg->Show(nullptr);
                if (hr == HRESULT_FROM_WIN32(ERROR_CANCELLED)) {
                    cancelled = true;
                    return;
                }
                if (FAILED(hr)) {
                    err_msg = "Save file dialog failed.";
                    return;
                }

                IShellItem* item = nullptr;
                if (SUCCEEDED(save_dlg->GetResult(&item)) && item) {
                    PWSTR psz = nullptr;
                    if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &psz)) && psz) {
                        const fs::path p(psz);
                        saved_path = util::path_to_utf8(p);
                        saved_dir = util::path_to_utf8(p.parent_path());
                        saved_filename = util::path_to_utf8(p.filename());
                        picked.push_back(saved_path);
                        CoTaskMemFree(psz);
                    }
                    item->Release();
                }
                return;
            }

            IFileOpenDialog* dlg = nullptr;
            HRESULT hr = CoCreateInstance(
                CLSID_FileOpenDialog, nullptr, CLSCTX_ALL, IID_PPV_ARGS(&dlg));
            if (FAILED(hr) || !dlg) {
                err_msg = "Failed to create Win32 IFileOpenDialog.";
                return;
            }
            struct DlgGuard {
                IFileOpenDialog* p = nullptr;
                ~DlgGuard() { if (p) p->Release(); }
            } dlg_guard{dlg};

            DWORD opts = 0;
            if (SUCCEEDED(dlg->GetOptions(&opts))) {
                opts |= FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST;
                if (mode == "folder") {
                    opts |= FOS_PICKFOLDERS;
                } else {
                    opts |= FOS_ALLOWMULTISELECT | FOS_FILEMUSTEXIST;
                }
                dlg->SetOptions(opts);
            }

            if (mode == "folder") {
                const std::wstring wtitle = dialog_title.empty()
                    ? L"Select Folder"
                    : util::utf8_to_wide(dialog_title);
                dlg->SetTitle(wtitle.c_str());
            } else {
                const std::wstring wtitle = dialog_title.empty()
                    ? L"Select Comic Archives (.cbz / .zip)"
                    : util::utf8_to_wide(dialog_title);
                dlg->SetTitle(wtitle.c_str());
                const COMDLG_FILTERSPEC filters[] = {
                    {L"Comic Archives (*.cbz;*.zip)", L"*.cbz;*.zip"},
                    {L"All Files (*.*)", L"*.*"},
                };
                dlg->SetFileTypes(2, filters);
                dlg->SetFileTypeIndex(1);
            }

            hr = dlg->Show(nullptr);
            if (hr == HRESULT_FROM_WIN32(ERROR_CANCELLED)) {
                cancelled = true;
                return;
            }
            if (FAILED(hr)) {
                err_msg = "File picker dialog failed.";
                return;
            }

            if (mode == "folder") {
                IShellItem* item = nullptr;
                if (SUCCEEDED(dlg->GetResult(&item)) && item) {
                    PWSTR psz = nullptr;
                    if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &psz)) && psz) {
                        picked.push_back(util::wide_to_utf8(psz));
                        CoTaskMemFree(psz);
                    }
                    item->Release();
                }
            } else {
                IShellItemArray* arr = nullptr;
                if (SUCCEEDED(dlg->GetResults(&arr)) && arr) {
                    DWORD count = 0;
                    if (SUCCEEDED(arr->GetCount(&count))) {
                        for (DWORD i = 0; i < count; ++i) {
                            IShellItem* item = nullptr;
                            if (SUCCEEDED(arr->GetItemAt(i, &item)) && item) {
                                PWSTR psz = nullptr;
                                if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &psz)) && psz) {
                                    picked.push_back(util::wide_to_utf8(psz));
                                    CoTaskMemFree(psz);
                                }
                                item->Release();
                            }
                        }
                    }
                    arr->Release();
                }
            }
        };

        // Run dialog on a dedicated STA thread so COM apartment mode never conflicts with httplib worker
        std::thread dlg_thread(run_dialog);
        dlg_thread.join();

        if (!err_msg.empty()) {
            send_json(res, {{"ok", false}, {"detail", err_msg}}, 500);
            return;
        }
        if (mode == "save_file") {
            if (cancelled || saved_path.empty()) {
                send_json(res, {{"ok", true}, {"cancelled", true}});
                return;
            }
            send_json(res, {
                {"ok", true},
                {"cancelled", false},
                {"path", saved_path},
                {"dir", saved_dir},
                {"filename", saved_filename},
                {"paths", picked},
            });
            return;
        }
        send_json(res, {
            {"ok", true},
            {"cancelled", cancelled},
            {"paths", picked},
        });
    });

    // #6: LLM Token, KV-Cache & Cost Telemetry
    server.Get("/api/telemetry/llm", [](const httplib::Request&, httplib::Response& res) {
        send_json(res, pipeline::ComicTranslator::get_session_telemetry());
    });
    server.Post("/api/telemetry/llm/reset", [](const httplib::Request&, httplib::Response& res) {
        pipeline::ComicTranslator::reset_session_telemetry();
        send_json(res, {{"ok", true}});
    });

    // #12: Win32 Anti-Sleep Wake-Lock & Taskbar Progress
    server.Post("/api/system/keep-awake", [](const httplib::Request& req, httplib::Response& res) {
        bool enabled = false;
        const json body = json::parse(req.body, nullptr, false);
        if (!body.is_discarded() && body.is_object() && body.contains("enabled") && body["enabled"].is_boolean()) {
            enabled = body["enabled"].get<bool>();
        }
        const EXECUTION_STATE prev = SetThreadExecutionState(
            enabled ? (ES_CONTINUOUS | ES_SYSTEM_REQUIRED) : ES_CONTINUOUS);
        send_json(res, {
            {"ok", prev != 0},
            {"enabled", enabled},
        });
    });

    server.Post("/api/system/taskbar-progress", [&ctx](const httplib::Request& req, httplib::Response& res) {
        const json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.is_object()) {
            send_json(res, {{"detail", "Invalid JSON body."}}, 400);
            return;
        }
        const std::string state = util::to_lower_ascii(util::trim_ascii(body.value("state", std::string("none"))));
        const uint64_t val = body.contains("value") && body["value"].is_number()
            ? static_cast<uint64_t>(std::max<int64_t>(0, body["value"].get<int64_t>())) : 0ULL;
        const uint64_t total = body.contains("total") && body["total"].is_number()
            ? static_cast<uint64_t>(std::max<int64_t>(1, body["total"].get<int64_t>())) : 100ULL;

        HWND hwnd = static_cast<HWND>(ctx.main_hwnd.load());
        bool applied = false;
        if (hwnd && IsWindow(hwnd)) {
            std::thread([hwnd, state, val, total, &applied]() {
                const HRESULT hr_init = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
                ITaskbarList3* tb = nullptr;
                if (SUCCEEDED(CoCreateInstance(CLSID_TaskbarList, nullptr, CLSCTX_INPROC_SERVER,
                                               IID_ITaskbarList3, reinterpret_cast<void**>(&tb))) && tb) {
                    if (SUCCEEDED(tb->HrInit())) {
                        TBPFLAG flag = TBPF_NOPROGRESS;
                        if (state == "normal") flag = TBPF_NORMAL;
                        else if (state == "paused") flag = TBPF_PAUSED;
                        else if (state == "error") flag = TBPF_ERROR;
                        tb->SetProgressState(hwnd, flag);
                        if (flag != TBPF_NOPROGRESS) {
                            tb->SetProgressValue(hwnd, std::min(val, total), total);
                        }
                        applied = true;
                    }
                    tb->Release();
                }
                if (SUCCEEDED(hr_init)) CoUninitialize();
            }).join();
        }
        send_json(res, {
            {"ok", true},
            {"applied", applied},
            {"state", state},
            {"value", val},
            {"total", total},
        });
    });

    // #15: On-Demand Disk Space Meter & Safe Cache Cleanup
    auto resolve_storage_dir = [&ctx](const std::string& name) -> fs::path {
        const fs::path cwd_dir = fs::current_path() / util::utf8_to_wide(name);
        std::error_code ec;
        if (fs::exists(cwd_dir, ec)) return cwd_dir;
        const fs::path exe_rel = ctx.exe_dir / util::utf8_to_wide(name);
        if (fs::exists(exe_rel, ec)) return exe_rel;
        return cwd_dir;
    };

    server.Get("/api/storage/stats", [resolve_storage_dir](const httplib::Request&, httplib::Response& res) {
        auto scan_dir = [](const fs::path& dir) -> json {
            uint64_t total_bytes = 0;
            uint64_t file_count = 0;
            std::error_code ec;
            if (fs::is_directory(dir, ec)) {
                for (const auto& entry : fs::recursive_directory_iterator(
                         dir, fs::directory_options::skip_permission_denied, ec)) {
                    if (entry.is_regular_file(ec)) {
                        total_bytes += static_cast<uint64_t>(entry.file_size(ec));
                        ++file_count;
                    }
                }
            }
            return {
                {"path", util::path_to_utf8(dir)},
                {"bytes", total_bytes},
                {"mb", std::round((static_cast<double>(total_bytes) / (1024.0 * 1024.0)) * 100.0) / 100.0},
                {"files", file_count},
            };
        };
        const json cache_s = scan_dir(resolve_storage_dir("cache"));
        const json proj_s = scan_dir(resolve_storage_dir("projects_cache"));
        const json models_s = scan_dir(resolve_storage_dir("models"));
        const json release_s = scan_dir(resolve_storage_dir("release"));
        const uint64_t total_bytes =
            cache_s["bytes"].get<uint64_t>() + proj_s["bytes"].get<uint64_t>() +
            models_s["bytes"].get<uint64_t>() + release_s["bytes"].get<uint64_t>();
        send_json(res, {
            {"ok", true},
            {"cache", cache_s},
            {"projects_cache", proj_s},
            {"models", models_s},
            {"release", release_s},
            {"total_bytes", total_bytes},
            {"total_mb", std::round((static_cast<double>(total_bytes) / (1024.0 * 1024.0)) * 100.0) / 100.0},
        });
    });

    server.Post("/api/storage/cleanup", [resolve_storage_dir](const httplib::Request& req, httplib::Response& res) {
        int max_age_days = 7;
        std::vector<std::string> targets = {"cache", "projects_cache"};
        const json body = json::parse(req.body, nullptr, false);
        if (!body.is_discarded() && body.is_object()) {
            if (body.contains("max_age_days") && body["max_age_days"].is_number()) {
                max_age_days = std::max(0, body["max_age_days"].get<int>());
            }
            if (body.contains("targets") && body["targets"].is_array()) {
                targets.clear();
                for (const auto& t : body["targets"]) {
                    if (t.is_string()) {
                        const std::string s = util::to_lower_ascii(util::trim_ascii(t.get<std::string>()));
                        // Strictly whitelist ONLY cache and projects_cache
                        if (s == "cache" || s == "projects_cache") {
                            targets.push_back(s);
                        }
                    }
                }
            }
        }

        const auto now_ft = fs::file_time_type::clock::now();
        const auto cutoff = now_ft - std::chrono::hours(24LL * static_cast<int64_t>(max_age_days));
        uint64_t freed_bytes = 0;
        uint64_t deleted_files = 0;
        std::error_code ec;

        for (const auto& tname : targets) {
            const fs::path dir = resolve_storage_dir(tname);
            if (!fs::is_directory(dir, ec)) continue;
            std::vector<fs::path> to_delete;
            for (const auto& entry : fs::recursive_directory_iterator(
                     dir, fs::directory_options::skip_permission_denied, ec)) {
                if (!entry.is_regular_file(ec)) continue;
                const std::string fname = util::path_to_utf8(entry.path().filename());
                if (fname == ".gitkeep") continue;
                const auto mtime = entry.last_write_time(ec);
                if (!ec && (max_age_days == 0 || mtime <= cutoff)) {
                    to_delete.push_back(entry.path());
                }
            }
            for (const auto& p : to_delete) {
                const uint64_t sz = static_cast<uint64_t>(fs::file_size(p, ec));
                if (fs::remove(p, ec) && !ec) {
                    freed_bytes += sz;
                    ++deleted_files;
                }
            }
        }

        send_json(res, {
            {"ok", true},
            {"max_age_days", max_age_days},
            {"freed_bytes", freed_bytes},
            {"deleted_files", deleted_files},
        });
    });

}

}  // namespace mangaflow::routes
