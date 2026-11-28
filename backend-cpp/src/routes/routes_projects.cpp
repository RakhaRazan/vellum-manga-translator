#include "routes_projects.hpp"
#include "../utils.hpp"

#include <stb_image.h>
#include <stb_image_write.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace mangaflow::routes {

void register_projects_routes(httplib::Server& server, AppContext& ctx) {
    (void)ctx;
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

    // Lightweight JPEG thumbnails (`/api/cache/thumb/{id}?w=180`, clamped w in 64..320)
    auto serve_cached_thumb = [](const std::string& ref, const httplib::Request& req, httplib::Response& res) {
        try {
            int target_w = 180;
            if (req.has_param("w")) {
                try {
                    target_w = std::stoi(req.get_param_value("w"));
                } catch (...) {
                    target_w = 180;
                }
            }
            target_w = std::clamp(target_w, 64, 320);
            std::string content_type = "image/jpeg";
            const auto thumb_bytes = get_or_create_cached_thumbnail(ref, target_w, /*jpeg_quality=*/75, &content_type);
            apply_cors_headers(res);
            res.set_header(
                "Cache-Control",
                content_type == "image/webp" ? "public, max-age=3600" : "public, max-age=86400");
            res.set_content(
                reinterpret_cast<const char*>(thumb_bytes.data()),
                thumb_bytes.size(),
                content_type);
        } catch (const ComicExportError& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", ex.what()}}, 404);
        }
    };

    server.Get(R"(/api/cache/thumb/([A-Za-z0-9_-]{1,128})\.([A-Za-z0-9]+))",
        [serve_cached_thumb](const httplib::Request& req, httplib::Response& res) {
            serve_cached_thumb(req.matches[1].str() + "." + util::to_lower_ascii(req.matches[2].str()), req, res);
        });

    server.Get(R"(/api/cache/thumb/([A-Za-z0-9_-]{1,128}))",
        [serve_cached_thumb](const httplib::Request& req, httplib::Response& res) {
            serve_cached_thumb(req.matches[1].str(), req, res);
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

    // Disk-backed crash-recovery & per-project autosave
    // - Default (no project_id): `./projects_cache/autosave.vellum.json`
    // - Isolated (`project_id` set): `./projects_cache/<sanitized_project_id>/autosave.json`
    auto resolve_autosave_target = [](const std::string& raw_pid, std::string& out_sanitized) -> fs::path {
        const std::string trimmed = util::trim_ascii(raw_pid);
        if (trimmed.empty()) {
            out_sanitized.clear();
            return get_projects_cache_dir() / L"autosave.vellum.json";
        }
        out_sanitized = pipeline::ComicTranslator::sanitize_project_id(trimmed);
        return get_projects_cache_dir() / util::utf8_to_wide(out_sanitized) / L"autosave.json";
    };

    server.Post("/api/projects_cache/autosave", [resolve_autosave_target](const httplib::Request& req, httplib::Response& res) {
        if (util::trim_ascii(req.body).empty()) {
            send_json(res, {{"detail", "Empty autosave request body."}}, 400);
            return;
        }
        const json parsed = json::parse(req.body, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object()) {
            send_json(res, {{"detail", "Invalid JSON object for /api/projects_cache/autosave."}}, 400);
            return;
        }
        std::string raw_pid;
        if (parsed.contains("project_id") && parsed["project_id"].is_string()) {
            raw_pid = parsed["project_id"].get<std::string>();
        } else if (req.has_param("project_id")) {
            raw_pid = req.get_param_value("project_id");
        }
        std::string sanitized_pid;
        const fs::path autosave_path = resolve_autosave_target(raw_pid, sanitized_pid);
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
            {"project_id", sanitized_pid},
            {"saved_bytes", req.body.size()},
            {"page_count", page_count},
        });
    });

    server.Get("/api/projects_cache/autosave", [resolve_autosave_target](const httplib::Request& req, httplib::Response& res) {
        const std::string raw_pid = req.has_param("project_id") ? req.get_param_value("project_id") : std::string();
        std::string sanitized_pid;
        const fs::path autosave_path = resolve_autosave_target(raw_pid, sanitized_pid);
        std::error_code ec;
        if (!fs::is_regular_file(autosave_path, ec)) {
            send_json(res, {{"status", "empty"}, {"project_id", sanitized_pid}, {"pages", json::array()}}, 200);
            return;
        }
        const auto bytes = util::read_file_bytes(autosave_path);
        if (!bytes.has_value() || bytes->empty()) {
            send_json(res, {{"status", "empty"}, {"project_id", sanitized_pid}, {"pages", json::array()}}, 200);
            return;
        }
        apply_cors_headers(res);
        res.status = 200;
        res.set_content(
            reinterpret_cast<const char*>(bytes->data()),
            bytes->size(),
            "application/json; charset=utf-8");
    });

    server.Delete("/api/projects_cache/autosave", [resolve_autosave_target](const httplib::Request& req, httplib::Response& res) {
        const std::string raw_pid = req.has_param("project_id") ? req.get_param_value("project_id") : std::string();
        const bool purge_dir = req.has_param("purge_dir") && parse_bool_param(req.get_param_value("purge_dir"), false);
        std::string sanitized_pid;
        const fs::path autosave_path = resolve_autosave_target(raw_pid, sanitized_pid);
        std::error_code ec;
        bool removed = false;
        if (purge_dir && !sanitized_pid.empty()) {
            const fs::path proj_dir = get_projects_cache_dir() / util::utf8_to_wide(sanitized_pid);
            if (fs::exists(proj_dir, ec) && !ec) {
                const auto count = fs::remove_all(proj_dir, ec);
                removed = (!ec && count > 0);
            }
        } else {
            removed = fs::remove(autosave_path, ec);
        }
        send_json(res, {{"status", "ok"}, {"project_id", sanitized_pid}, {"removed", removed}});
    });

    // Clear per-project LLM conversation history (RAM + disk `conversation.json`)
    server.Delete(R"(/api/conversation/([A-Za-z0-9_.-]{1,64}))",
        [](const httplib::Request& req, httplib::Response& res) {
            const std::string raw_pid = util::trim_ascii(req.matches[1].str());
            const std::string sanitized = pipeline::ComicTranslator::sanitize_project_id(raw_pid);
            const fs::path conv_path = pipeline::ComicTranslator::conversation_file_path(raw_pid);
            std::error_code ec;
            const bool existed = !conv_path.empty() && fs::is_regular_file(conv_path, ec);
            pipeline::ComicTranslator::clear_project_conversation(raw_pid);
            send_json(res, {
                {"status", "ok"},
                {"project_id", raw_pid},
                {"sanitized_project_id", sanitized},
                {"file_removed", existed},
            });
        });

    // -------------------------------------------------------------------------
    // Per-Series Profiles Storage (`./series_profiles/<slug>.json`)
    // -------------------------------------------------------------------------
    auto get_series_profiles_dir = []() -> fs::path {
        const fs::path dir = util::get_portable_root_dir() / L"series_profiles";
        std::error_code ec;
        fs::create_directories(dir, ec);
        return dir;
    };

    auto sanitize_profile_slug = [](const std::string& raw) -> std::string {
        const std::string lower = util::to_lower_ascii(util::trim_ascii(raw));
        std::string out;
        out.reserve(std::min<size_t>(lower.size(), 64));
        bool prev_sep = false;
        for (unsigned char c : lower) {
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
                out.push_back(static_cast<char>(c));
                prev_sep = false;
            } else if (c == '-' || c == '_') {
                if (!out.empty() && !prev_sep) {
                    out.push_back(static_cast<char>(c));
                    prev_sep = true;
                }
            } else {
                if (!out.empty() && !prev_sep) {
                    out.push_back('_');
                    prev_sep = true;
                }
            }
            if (out.size() >= 64) break;
        }
        while (!out.empty() && (out.back() == '_' || out.back() == '-')) {
            out.pop_back();
        }
        return out;
    };

    auto iso8601_utc_now = []() -> std::string {
        const auto now = std::chrono::system_clock::now();
        const std::time_t tt = std::chrono::system_clock::to_time_t(now);
        std::tm utc_tm{};
        gmtime_s(&utc_tm, &tt);
        char buf[32]{};
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &utc_tm);
        return std::string(buf);
    };

    auto normalize_match_title = [](const std::string& raw) -> std::string {
        std::string s = util::to_lower_ascii(util::trim_ascii(raw));
        // Strip common archive extensions
        for (const char* ext : {".cbz", ".zip", ".cbr", ".rar", ".pdf"}) {
            const size_t elen = std::strlen(ext);
            if (s.size() > elen && s.compare(s.size() - elen, elen, ext) == 0) {
                s.erase(s.size() - elen);
                break;
            }
        }
        // Replace punctuation/separators with spaces
        std::string spaced;
        spaced.reserve(s.size());
        for (unsigned char c : s) {
            if (std::isalnum(c) || c >= 0x80) {
                spaced.push_back(static_cast<char>(c));
            } else {
                spaced.push_back(' ');
            }
        }
        // Tokenize and drop volume/chapter/number noise tokens
        std::istringstream iss(spaced);
        std::string tok;
        std::vector<std::string> kept;
        while (iss >> tok) {
            if (tok == "vol" || tok == "volume" || tok == "ch" || tok == "chap" ||
                tok == "chapter" || tok == "ep" || tok == "episode" || tok == "part" ||
                tok == "book" || tok == "raw" || tok == "jp" || tok == "jpn" || tok == "en" || tok == "id") {
                continue;
            }
            bool all_digits = !tok.empty() && std::all_of(tok.begin(), tok.end(), [](unsigned char c) {
                return std::isdigit(c) != 0;
            });
            if (all_digits) continue;
            // Drop v01 / c012 / ch05 / vol03 / ep12
            if ((tok.size() >= 2 && (tok[0] == 'v' || tok[0] == 'c') &&
                 std::all_of(tok.begin() + 1, tok.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) ||
                (tok.size() >= 3 && (tok.rfind("ch", 0) == 0 || tok.rfind("ep", 0) == 0) &&
                 std::all_of(tok.begin() + 2, tok.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) ||
                (tok.size() >= 4 && tok.rfind("vol", 0) == 0 &&
                 std::all_of(tok.begin() + 3, tok.end(), [](unsigned char c) { return std::isdigit(c) != 0; }))) {
                continue;
            }
            kept.push_back(std::move(tok));
        }
        std::ostringstream oss;
        for (size_t i = 0; i < kept.size(); ++i) {
            if (i > 0) oss << ' ';
            oss << kept[i];
        }
        return oss.str();
    };

    auto load_all_series_profiles = [get_series_profiles_dir]() -> std::vector<json> {
        std::vector<json> profiles;
        const fs::path dir = get_series_profiles_dir();
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(dir, ec)) {
            if (ec) break;
            if (!entry.is_regular_file(ec) || ec) continue;
            const std::string ext = util::to_lower_ascii(util::path_to_utf8(entry.path().extension()));
            if (ext != ".json") continue;
            const auto bytes = util::read_file_bytes(entry.path());
            if (!bytes.has_value() || bytes->empty()) continue;
            const json parsed = json::parse(bytes->begin(), bytes->end(), nullptr, false);
            if (!parsed.is_discarded() && parsed.is_object() && parsed.contains("slug") && parsed["slug"].is_string()) {
                profiles.push_back(parsed);
            }
        }
        std::sort(profiles.begin(), profiles.end(), [](const json& a, const json& b) {
            const std::string ua = a.value("updated_at", std::string(""));
            const std::string ub = b.value("updated_at", std::string(""));
            if (ua != ub) return ua > ub;
            return a.value("name", std::string("")) < b.value("name", std::string(""));
        });
        return profiles;
    };

    server.Get("/api/series-profiles", [load_all_series_profiles](const httplib::Request&, httplib::Response& res) {
        const auto profiles = load_all_series_profiles();
        send_json(res, {
            {"ok", true},
            {"profiles", profiles},
        });
    });

    server.Post("/api/series-profiles",
        [get_series_profiles_dir, sanitize_profile_slug, iso8601_utc_now](
            const httplib::Request& req, httplib::Response& res) {
            const json body = json::parse(req.body, nullptr, false);
            if (body.is_discarded() || !body.is_object()) {
                send_json(res, {{"detail", "Invalid JSON object for /api/series-profiles."}}, 400);
                return;
            }
            const std::string raw_name = util::trim_ascii(body.value("name", std::string("")));
            const std::string raw_slug = util::trim_ascii(body.value("slug", std::string("")));
            const std::string slug = sanitize_profile_slug(!raw_slug.empty() ? raw_slug : raw_name);
            if (slug.empty()) {
                send_json(res, {{"detail", "Provide a valid 'slug' or 'name' (alphanumeric/underscore/hyphen)."}}, 400);
                return;
            }
            const std::string name = raw_name.empty() ? slug : raw_name;

            json match_keywords = json::array();
            if (body.contains("match_keywords") && body["match_keywords"].is_array()) {
                for (const auto& kw : body["match_keywords"]) {
                    if (kw.is_string()) {
                        const std::string k = util::trim_ascii(kw.get<std::string>());
                        if (!k.empty()) match_keywords.push_back(k);
                    }
                }
            }

            json glossary_arr = json::array();
            if (body.contains("glossary")) {
                if (body["glossary"].is_array()) {
                    for (const auto& item : body["glossary"]) {
                        if (!item.is_object()) continue;
                        const std::string src = util::trim_ascii(item.value("src", std::string("")));
                        const std::string dst = util::trim_ascii(item.value("dst", std::string("")));
                        if (!src.empty() && !dst.empty()) {
                            glossary_arr.push_back({{"src", src}, {"dst", dst}});
                        }
                    }
                } else if (body["glossary"].is_object()) {
                    for (auto it = body["glossary"].begin(); it != body["glossary"].end(); ++it) {
                        const std::string src = util::trim_ascii(it.key());
                        const std::string dst = it.value().is_string() ? util::trim_ascii(it.value().get<std::string>()) : "";
                        if (!src.empty() && !dst.empty()) {
                            glossary_arr.push_back({{"src", src}, {"dst", dst}});
                        }
                    }
                }
            }

            const json profile = {
                {"slug", slug},
                {"name", name},
                {"match_keywords", std::move(match_keywords)},
                {"glossary", std::move(glossary_arr)},
                {"custom_instructions", util::trim_ascii(body.value("custom_instructions", std::string("")))},
                {"translation_tone", pipeline::ComicTranslator::normalize_translation_tone(
                    body.value("translation_tone", std::string("standard")))},
                {"honorifics_policy", pipeline::ComicTranslator::normalize_honorifics_policy(
                    body.value("honorifics_policy", std::string("keep")))},
                {"source_lang", pipeline::ComicTranslator::normalize_source_language(
                    body.value("source_lang", std::string("ja")))},
                {"target_lang", pipeline::ComicTranslator::normalize_target_language(
                    body.value("target_lang", std::string("id")))},
                {"detector_precision", SettingsManager::normalize_detector_precision(
                    body.value("detector_precision", std::string("balanced")))},
                {"updated_at", iso8601_utc_now()},
            };

            const fs::path target_path = get_series_profiles_dir() / util::utf8_to_wide(slug + ".json");
            const std::string serialized = profile.dump(2, ' ', false, json::error_handler_t::replace);
            std::string write_err;
            if (!util::atomic_write_bytes(
                    target_path,
                    reinterpret_cast<const uint8_t*>(serialized.data()),
                    serialized.size(),
                    &write_err)) {
                send_json(res, {{"detail", write_err.empty() ? "Failed to save series profile." : write_err}}, 500);
                return;
            }
            send_json(res, {
                {"ok", true},
                {"profile", profile},
            });
        });

    server.Post("/api/series-profiles/match",
        [load_all_series_profiles, normalize_match_title](
            const httplib::Request& req, httplib::Response& res) {
            const json body = json::parse(req.body, nullptr, false);
            if (body.is_discarded() || !body.is_object()) {
                send_json(res, {{"detail", "Invalid JSON object for /api/series-profiles/match."}}, 400);
                return;
            }
            const std::string raw_title = util::trim_ascii(body.value("title", std::string("")));
            const std::string norm_title = normalize_match_title(raw_title);
            if (norm_title.empty()) {
                send_json(res, {{"ok", true}, {"matched", false}, {"profile", nullptr}});
                return;
            }

            auto score_candidate = [&](const std::string& cand_raw) -> int {
                const std::string norm_cand = normalize_match_title(cand_raw);
                if (norm_cand.empty()) return 0;
                if (norm_title == norm_cand) return 100;
                if (norm_title.find(norm_cand) != std::string::npos ||
                    norm_cand.find(norm_title) != std::string::npos) {
                    const size_t min_len = std::min(norm_title.size(), norm_cand.size());
                    const size_t max_len = std::max(norm_title.size(), norm_cand.size());
                    return 65 + static_cast<int>((30ULL * min_len) / std::max<size_t>(1, max_len));
                }
                // Token Jaccard overlap
                std::unordered_set<std::string> t1, t2;
                {
                    std::istringstream a(norm_title);
                    std::string w;
                    while (a >> w) t1.insert(w);
                }
                {
                    std::istringstream b(norm_cand);
                    std::string w;
                    while (b >> w) t2.insert(w);
                }
                if (t1.empty() || t2.empty()) return 0;
                size_t inter = 0;
                for (const auto& w : t1) {
                    if (t2.count(w)) ++inter;
                }
                if (inter == 0) return 0;
                const size_t uni = t1.size() + t2.size() - inter;
                return static_cast<int>((85ULL * inter) / std::max<size_t>(1, uni));
            };

            const auto profiles = load_all_series_profiles();
            int best_score = 0;
            json best_profile = nullptr;
            for (const auto& prof : profiles) {
                int s = std::max(
                    score_candidate(prof.value("name", std::string(""))),
                    score_candidate(prof.value("slug", std::string(""))));
                if (prof.contains("match_keywords") && prof["match_keywords"].is_array()) {
                    for (const auto& kw : prof["match_keywords"]) {
                        if (kw.is_string()) {
                            s = std::max(s, score_candidate(kw.get<std::string>()));
                        }
                    }
                }
                if (s > best_score) {
                    best_score = s;
                    best_profile = prof;
                }
            }

            const bool matched = (best_score >= 50 && best_profile.is_object());
            send_json(res, {
                {"ok", true},
                {"matched", matched},
                {"score", best_score},
                {"normalized_title", norm_title},
                {"profile", matched ? best_profile : json(nullptr)},
            });
        });

    server.Delete(R"(/api/series-profiles/([A-Za-z0-9_-]{1,64}))",
        [get_series_profiles_dir, sanitize_profile_slug](
            const httplib::Request& req, httplib::Response& res) {
            const std::string slug = sanitize_profile_slug(req.matches[1].str());
            if (slug.empty()) {
                send_json(res, {{"detail", "Invalid series profile slug."}}, 400);
                return;
            }
            const fs::path target_path = get_series_profiles_dir() / util::utf8_to_wide(slug + ".json");
            std::error_code ec;
            const bool removed = fs::remove(target_path, ec);
            send_json(res, {
                {"ok", true},
                {"slug", slug},
                {"removed", removed},
            });
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

    // Local Folder-Path Image Importer (O(1) RAM streaming copy into cache)
    server.Post("/api/import/folder-path", [](const httplib::Request& req, httplib::Response& res) {
        const json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.is_object()) {
            send_json(res, {{"detail", "Invalid JSON object for /api/import/folder-path."}}, 400);
            return;
        }
        const std::string raw_path = util::trim_ascii(body.value("path", std::string("")));
        if (raw_path.empty()) {
            send_json(res, {{"detail", "Provide non-empty 'path' to a local image folder."}}, 400);
            return;
        }
        const fs::path folder_path(util::utf8_to_wide(raw_path));
        try {
            const fs::path cache_dir = get_image_cache_dir();
            const auto imported = import_folder_pages_from_dir(folder_path, cache_dir);
            json pages_json = json::array();
            for (const auto& p : imported) {
                const std::string url = "/api/cache/images/" + p.image_id + "." + p.extension;
                pages_json.push_back({
                    {"index", p.index},
                    {"filename", p.filename},
                    {"image_id", p.image_id},
                    {"originalUrl", url},
                    {"image_url", url},
                    {"width", p.dimensions.width},
                    {"height", p.dimensions.height},
                    {"dimensions", {{"width", p.dimensions.width}, {"height", p.dimensions.height}}},
                    {"size_bytes", p.size_bytes},
                });
            }
            send_json(res, {
                {"ok", true},
                {"folder_name", util::path_to_utf8(folder_path.filename())},
                {"total_pages", imported.size()},
                {"pages", std::move(pages_json)},
            });
        } catch (const ArchiveImportError& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", std::string("Folder import failed: ") + ex.what()}}, 500);
        }
    });

    // Scan a root directory for immediate `.cbz`/`.zip` archives and 1-level image subfolders
    server.Post("/api/import/scan-batch-root", [](const httplib::Request& req, httplib::Response& res) {
        const json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.is_object()) {
            send_json(res, {{"detail", "Invalid JSON object for /api/import/scan-batch-root."}}, 400);
            return;
        }
        std::string raw_folder = util::trim_ascii(body.value("folder", std::string("")));
        if (raw_folder.empty()) {
            raw_folder = util::trim_ascii(body.value("path", std::string("")));
        }
        if (raw_folder.empty()) {
            send_json(res, {{"detail", "Provide non-empty 'folder' path to scan."}}, 400);
            return;
        }
        const fs::path root_path(util::utf8_to_wide(raw_folder));
        std::error_code ec;
        if (!fs::is_directory(root_path, ec) || ec) {
            send_json(res, {{"detail", "Root folder was not found on disk."}}, 400);
            return;
        }

        auto is_supported_img_ext = [](const std::string& ext_lower) -> bool {
            return ext_lower == ".jpg" || ext_lower == ".jpeg" ||
                   ext_lower == ".png" || ext_lower == ".webp" || ext_lower == ".bmp";
        };

        struct ScannedBatchItem {
            std::string kind;  // "archive" | "folder"
            std::string name;
            std::string path;
            int page_count = -1;
        };
        std::vector<ScannedBatchItem> items;

        for (const auto& entry : fs::directory_iterator(root_path, ec)) {
            if (ec) break;
            std::error_code item_ec;
            const fs::path p = entry.path();
            if (entry.is_regular_file(item_ec) && !item_ec) {
                const std::string ext = util::to_lower_ascii(util::path_to_utf8(p.extension()));
                if (ext == ".cbz" || ext == ".zip") {
                    items.push_back({
                        "archive",
                        util::path_to_utf8(p.filename()),
                        util::path_to_utf8(p),
                        -1,
                    });
                }
            } else if (entry.is_directory(item_ec) && !item_ec) {
                int img_count = 0;
                std::error_code sub_ec;
                for (const auto& sub : fs::directory_iterator(p, sub_ec)) {
                    if (sub_ec) break;
                    std::error_code f_ec;
                    if (!sub.is_regular_file(f_ec) || f_ec) continue;
                    const std::string sext = util::to_lower_ascii(util::path_to_utf8(sub.path().extension()));
                    if (is_supported_img_ext(sext)) {
                        ++img_count;
                    }
                }
                if (img_count > 0) {
                    items.push_back({
                        "folder",
                        util::path_to_utf8(p.filename()),
                        util::path_to_utf8(p),
                        img_count,
                    });
                }
            }
        }

        std::sort(items.begin(), items.end(), [](const ScannedBatchItem& a, const ScannedBatchItem& b) {
            return natural_path_less(a.name, b.name);
        });

        json out_items = json::array();
        for (const auto& it : items) {
            out_items.push_back({
                {"kind", it.kind},
                {"name", it.name},
                {"path", it.path},
                {"page_count", it.page_count >= 0 ? json(it.page_count) : json(nullptr)},
            });
        }
        send_json(res, {
            {"ok", true},
            {"root", util::path_to_utf8(root_path)},
            {"total_items", out_items.size()},
            {"items", std::move(out_items)},
        });
    });

    // #5: Quick Page Image Ops (`POST /api/cache/transform`: rotate_cw/ccw/180 & split_spread)
    server.Post("/api/cache/transform", [](const httplib::Request& req, httplib::Response& res) {
        const json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.is_object()) {
            send_json(res, {{"detail", "Invalid JSON body."}}, 400);
            return;
        }
        std::string image_id = util::trim_ascii(body.value("image_id", std::string("")));
        const std::string op = util::to_lower_ascii(util::trim_ascii(body.value("op", std::string(""))));
        const std::string reading_dir = util::to_lower_ascii(util::trim_ascii(body.value("reading_direction", std::string("rtl"))));

        if (image_id.empty()) {
            send_json(res, {{"detail", "Missing image_id."}}, 400);
            return;
        }
        // Allow passing full URL `/api/cache/images/<id>.png` or bare `<id>`
        const std::string kPrefix = "/api/cache/images/";
        if (const size_t pos = image_id.rfind(kPrefix); pos != std::string::npos) {
            image_id = image_id.substr(pos + kPrefix.size());
        }

        if (op != "rotate_cw" && op != "rotate_ccw" && op != "rotate_180" && op != "split_spread") {
            send_json(res, {{"detail", "Unsupported op. Expected rotate_cw, rotate_ccw, rotate_180, or split_spread."}}, 400);
            return;
        }

        fs::path src_path;
        try {
            src_path = resolve_cached_image_path(image_id);
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", ex.what()}}, 404);
            return;
        }

        const auto file_bytes = util::read_file_bytes(src_path);
        if (!file_bytes.has_value() || file_bytes->empty()) {
            send_json(res, {{"detail", "Failed to read cached source image."}}, 500);
            return;
        }

        int w = 0, h = 0, ch = 0;
        std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> stbi_rgb(
            stbi_load_from_memory(file_bytes->data(), static_cast<int>(file_bytes->size()), &w, &h, &ch, 3),
            &stbi_image_free);
        std::vector<uint8_t> wic_rgb;
        const uint8_t* rgb = stbi_rgb.get();
        if (!rgb || w <= 0 || h <= 0) {
            stbi_rgb.reset();
            if (util::decode_image_wic(file_bytes->data(), file_bytes->size(), 3, wic_rgb, w, h) &&
                !wic_rgb.empty() && w > 0 && h > 0) {
                rgb = wic_rgb.data();
            }
        }
        if (!rgb || w <= 0 || h <= 0) {
            send_json(res, {{"detail", "Failed to decode cached source image."}}, 400);
            return;
        }

        auto save_rgb_as_cached_png = [](const uint8_t* data, int dw, int dh) -> json {
            std::vector<uint8_t> png_buf;
            auto write_cb = [](void* context, void* ptr, int size) {
                auto* out = static_cast<std::vector<uint8_t>*>(context);
                const auto* b = static_cast<const uint8_t*>(ptr);
                out->insert(out->end(), b, b + size);
            };
            if (!stbi_write_png_to_func(write_cb, &png_buf, dw, dh, 3, data, dw * 3) || png_buf.empty()) {
                throw std::runtime_error("Failed to encode transformed PNG.");
            }
            const fs::path cache_dir = get_image_cache_dir();
            const std::string new_id = util::generate_uuid_hex();
            const fs::path dst_path = cache_dir / util::utf8_to_wide(new_id + ".png");
            std::string write_err;
            if (!util::atomic_write_bytes(dst_path, png_buf.data(), png_buf.size(), &write_err)) {
                throw std::runtime_error("Failed to save transformed image: " + write_err);
            }
            return {
                {"image_id", new_id},
                {"url", "/api/cache/images/" + new_id + ".png"},
                {"width", dw},
                {"height", dh},
            };
        };

        try {
            json out_images = json::array();
            if (op == "split_spread") {
                if (w < 4) {
                    send_json(res, {{"detail", "Image is too narrow to split."}}, 400);
                    return;
                }
                const int mid = w / 2;
                const int left_w = mid;
                const int right_w = w - mid;
                std::vector<uint8_t> left_buf(static_cast<size_t>(left_w) * static_cast<size_t>(h) * 3ULL);
                std::vector<uint8_t> right_buf(static_cast<size_t>(right_w) * static_cast<size_t>(h) * 3ULL);
                for (int y = 0; y < h; ++y) {
                    const uint8_t* src_row = rgb + static_cast<size_t>(y) * static_cast<size_t>(w) * 3ULL;
                    std::memcpy(left_buf.data() + static_cast<size_t>(y) * static_cast<size_t>(left_w) * 3ULL,
                                src_row, static_cast<size_t>(left_w) * 3ULL);
                    std::memcpy(right_buf.data() + static_cast<size_t>(y) * static_cast<size_t>(right_w) * 3ULL,
                                src_row + static_cast<size_t>(mid) * 3ULL, static_cast<size_t>(right_w) * 3ULL);
                }
                const json left_obj = save_rgb_as_cached_png(left_buf.data(), left_w, h);
                const json right_obj = save_rgb_as_cached_png(right_buf.data(), right_w, h);
                if (reading_dir == "ltr") {
                    out_images.push_back(left_obj);
                    out_images.push_back(right_obj);
                } else {
                    out_images.push_back(right_obj);
                    out_images.push_back(left_obj);
                }
            } else {
                const int nw = (op == "rotate_180") ? w : h;
                const int nh = (op == "rotate_180") ? h : w;
                std::vector<uint8_t> rot(static_cast<size_t>(nw) * static_cast<size_t>(nh) * 3ULL);
                for (int y = 0; y < h; ++y) {
                    for (int x = 0; x < w; ++x) {
                        int dx = x, dy = y;
                        if (op == "rotate_cw") {
                            dx = h - 1 - y;
                            dy = x;
                        } else if (op == "rotate_ccw") {
                            dx = y;
                            dy = w - 1 - x;
                        } else {
                            dx = w - 1 - x;
                            dy = h - 1 - y;
                        }
                        const uint8_t* sp = rgb + (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * 3ULL;
                        uint8_t* dp = rot.data() + (static_cast<size_t>(dy) * static_cast<size_t>(nw) + static_cast<size_t>(dx)) * 3ULL;
                        dp[0] = sp[0];
                        dp[1] = sp[1];
                        dp[2] = sp[2];
                    }
                }
                out_images.push_back(save_rgb_as_cached_png(rot.data(), nw, nh));
            }
            send_json(res, {
                {"ok", true},
                {"op", op},
                {"reading_direction", reading_dir},
                {"images", std::move(out_images)},
            });
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", ex.what()}}, 500);
        }
    });

}

}  // namespace mangaflow::routes
