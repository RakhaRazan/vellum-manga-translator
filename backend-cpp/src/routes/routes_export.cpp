#include "routes_export.hpp"
#include "../psd_writer.hpp"
#include "../utils.hpp"

#include <miniz.h>
#include <stb_image.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace mangaflow::routes {

void register_export_routes(httplib::Server& server, AppContext& ctx) {
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

    server.Post("/api/export/psd", [](const httplib::Request& req, httplib::Response& res) {
        try {
            auto decode_rgba = [](const std::vector<uint8_t>& raw, int& w, int& h, const std::string& label) -> std::vector<uint8_t> {
                int ch = 0;
                unsigned char* px = stbi_load_from_memory(
                    raw.data(), static_cast<int>(raw.size()), &w, &h, &ch, 4);
                if (!px || w <= 0 || h <= 0) {
                    throw ComicExportError("Failed to decode image for PSD (" + label + ").");
                }
                std::vector<uint8_t> out(px, px + static_cast<size_t>(w) * static_cast<size_t>(h) * 4U);
                stbi_image_free(px);
                return out;
            };

            auto decode_gray = [](const std::vector<uint8_t>& raw, int& w, int& h) -> std::vector<uint8_t> {
                int ch = 0;
                unsigned char* px = stbi_load_from_memory(
                    raw.data(), static_cast<int>(raw.size()), &w, &h, &ch, 1);
                if (!px || w <= 0 || h <= 0) {
                    return {};
                }
                std::vector<uint8_t> out(px, px + static_cast<size_t>(w) * static_cast<size_t>(h));
                stbi_image_free(px);
                return out;
            };

            auto pick_str = [](const json& obj, std::initializer_list<const char*> keys) -> std::string {
                for (const char* k : keys) {
                    if (obj.contains(k) && obj[k].is_string()) {
                        const std::string v = util::trim_ascii(obj[k].get<std::string>());
                        if (!v.empty()) return v;
                    }
                }
                return {};
            };

            if (req.body.empty()) {
                throw ComicExportError("Empty PSD export request body.");
            }
            const json body = json::parse(req.body, nullptr, false);
            if (body.is_discarded() || !body.is_object()) {
                throw ComicExportError("Invalid JSON request body for /api/export/psd.");
            }

            const json* raw_pages = nullptr;
            if (body.contains("pages") && body["pages"].is_array() && !body["pages"].empty()) {
                raw_pages = &body["pages"];
            }
            if (!raw_pages) {
                throw ComicExportError("Provide a non-empty 'pages' array for /api/export/psd.");
            }

            std::optional<std::string> export_name;
            if (const std::string fn = pick_str(body, {"filename", "title"}); !fn.empty()) {
                export_name = fn;
            }
            const std::string output_dir = pick_str(body, {"output_dir"});
            std::optional<std::string> output_filename_opt;
            if (const std::string ofn = pick_str(body, {"output_filename"}); !ofn.empty()) {
                output_filename_opt = ofn;
            }
            const bool overwrite =
                body.contains("overwrite") && body["overwrite"].is_boolean()
                    ? body["overwrite"].get<bool>()
                    : false;
            const bool editable_text_layers =
                body.contains("editable_text_layers") && body["editable_text_layers"].is_boolean()
                    ? body["editable_text_layers"].get<bool>()
                    : false;

            std::vector<std::pair<std::string, std::vector<uint8_t>>> page_psds;
            page_psds.reserve(raw_pages->size());
            const size_t pad_width = std::max<size_t>(3, std::to_string(raw_pages->size()).size());

            for (size_t i = 0; i < raw_pages->size(); ++i) {
                const int page_num = static_cast<int>(i + 1);
                const auto& p = (*raw_pages)[i];
                if (!p.is_object()) {
                    throw ComicExportError("Page " + std::to_string(page_num) + " must be a JSON object.");
                }

                const std::string orig_ref = pick_str(p, {"original_image", "original", "image", "image_url"});
                if (orig_ref.empty()) {
                    throw ComicExportError("Page " + std::to_string(page_num) + " is missing 'original_image'.");
                }
                const auto orig_bytes = decode_export_image_reference(orig_ref, page_num);
                int doc_w = 0, doc_h = 0;
                const auto orig_rgba = decode_rgba(orig_bytes, doc_w, doc_h, "original_image");

                std::vector<vellum::psd::PsdLayer> layers;

                // Layer 0: Background (original raw page image, opaque RGB)
                {
                    vellum::psd::PsdLayer bg;
                    bg.name = "Background";
                    bg.left = 0;
                    bg.top = 0;
                    bg.width = doc_w;
                    bg.height = doc_h;
                    bg.has_alpha = false;
                    bg.pixels.resize(static_cast<size_t>(doc_w) * static_cast<size_t>(doc_h) * 3U);
                    for (size_t px = 0; px < static_cast<size_t>(doc_w) * static_cast<size_t>(doc_h); ++px) {
                        bg.pixels[px * 3 + 0] = orig_rgba[px * 4 + 0];
                        bg.pixels[px * 3 + 1] = orig_rgba[px * 4 + 1];
                        bg.pixels[px * 3 + 2] = orig_rgba[px * 4 + 2];
                    }
                    layers.push_back(std::move(bg));
                }

                // Layer 1: Cleanup (non-destructive diff patch + transparency)
                {
                    const std::string clean_ref = pick_str(p, {"cleaned_image", "inpainted_image", "cleaned"});
                    if (!clean_ref.empty() && clean_ref != orig_ref) {
                        const auto clean_bytes = decode_export_image_reference(clean_ref, page_num);
                        int cw = 0, ch = 0;
                        const auto clean_rgba = decode_rgba(clean_bytes, cw, ch, "cleaned_image");

                        std::vector<uint8_t> mask_gray;
                        const std::string mask_ref = pick_str(p, {"mask_image", "mask_url", "mask"});
                        if (!mask_ref.empty()) {
                            try {
                                const auto mask_bytes = decode_export_image_reference(mask_ref, page_num);
                                int mw = 0, mh = 0;
                                auto decoded_mask = decode_gray(mask_bytes, mw, mh);
                                if (mw == doc_w && mh == doc_h) {
                                    mask_gray = std::move(decoded_mask);
                                }
                            } catch (...) {
                                // Ignore optional mask decode errors; fall back to pixel diff
                            }
                        }

                        if (cw == doc_w && ch == doc_h) {
                            layers.push_back(vellum::psd::build_cleanup_diff_layer(
                                orig_rgba.data(),
                                clean_rgba.data(),
                                doc_w,
                                doc_h,
                                mask_gray.empty() ? nullptr : mask_gray.data(),
                                2,
                                "Cleanup"));
                        } else {
                            vellum::psd::PsdLayer clean_layer;
                            clean_layer.name = "Cleanup";
                            clean_layer.left = 0;
                            clean_layer.top = 0;
                            clean_layer.width = cw;
                            clean_layer.height = ch;
                            clean_layer.has_alpha = true;
                            clean_layer.pixels = clean_rgba;
                            layers.push_back(std::move(clean_layer));
                        }
                    } else {
                        vellum::psd::PsdLayer empty_clean;
                        empty_clean.name = "Cleanup";
                        empty_clean.has_alpha = true;
                        layers.push_back(std::move(empty_clean));
                    }
                }

                // Layer 2+: Text (per-block layers with exact (x, y) offsets, or full-page Text layer)
                bool added_text_layer = false;
                const json* raw_tlayers = nullptr;
                for (const char* tkey : {"text_layers", "blocks", "bubbles"}) {
                    if (p.contains(tkey) && p[tkey].is_array() && !p[tkey].empty()) {
                        raw_tlayers = &p[tkey];
                        break;
                    }
                }
                if (raw_tlayers != nullptr) {
                    const auto& tlayers = *raw_tlayers;
                    for (size_t ti = 0; ti < tlayers.size(); ++ti) {
                        const auto& tl = tlayers[ti];
                        if (!tl.is_object()) continue;

                        vellum::psd::PsdLayer layer;
                        std::string lname = pick_str(tl, {"name", "id"});
                        if (lname.empty()) {
                            char nb[32]{};
                            std::snprintf(nb, sizeof(nb), "Text %02zu", ti + 1);
                            lname = nb;
                        }
                        layer.name = lname;
                        layer.has_alpha = true;

                        // Extract bounding coordinates from explicit x/y/width/height or fallback `bbox: [x1, y1, x2, y2]`
                        double bx = 0.0, by = 0.0, bw = 0.0, bh = 0.0;
                        if (tl.contains("bbox") && tl["bbox"].is_array() && tl["bbox"].size() >= 4 &&
                            tl["bbox"][0].is_number() && tl["bbox"][1].is_number() &&
                            tl["bbox"][2].is_number() && tl["bbox"][3].is_number()) {
                            bx = tl["bbox"][0].get<double>();
                            by = tl["bbox"][1].get<double>();
                            bw = std::max(1.0, tl["bbox"][2].get<double>() - bx);
                            bh = std::max(1.0, tl["bbox"][3].get<double>() - by);
                        }
                        const int32_t lx = tl.contains("x") && tl["x"].is_number()
                            ? static_cast<int32_t>(std::llround(tl["x"].get<double>()))
                            : static_cast<int32_t>(std::llround(bx));
                        const int32_t ly = tl.contains("y") && tl["y"].is_number()
                            ? static_cast<int32_t>(std::llround(tl["y"].get<double>()))
                            : static_cast<int32_t>(std::llround(by));
                        if (tl.contains("opacity") && tl["opacity"].is_number()) {
                            const double op = std::max(0.0, std::min(1.0, tl["opacity"].get<double>()));
                            layer.opacity = static_cast<uint8_t>(std::llround(op * 255.0));
                        }

                        const std::string img_ref = pick_str(tl, {"image", "raster_image"});
                        if (!img_ref.empty()) {
                            const auto img_bytes = decode_export_image_reference(img_ref, page_num);
                            int tw = 0, th = 0;
                            auto trgba = decode_rgba(img_bytes, tw, th, "text_layer");
                            layer.left = lx;
                            layer.top = ly;
                            layer.width = tw;
                            layer.height = th;
                            layer.pixels = std::move(trgba);
                        }

                        if (editable_text_layers) {
                            const std::string txt = pick_str(tl, {"text", "translated_text", "original_text"});
                            if (!txt.empty()) {
                                vellum::psd::TypeLayerSpec spec;
                                spec.text = txt;
                                const std::string font = pick_str(tl, {"font_family", "font"});
                                if (!font.empty()) spec.font_name = font;
                                if (tl.contains("font_size") && tl["font_size"].is_number()) {
                                    spec.font_size = std::max(4.0, tl["font_size"].get<double>());
                                }
                                if (tl.contains("line_height") && tl["line_height"].is_number()) {
                                    spec.leading = spec.font_size * std::max(0.5, tl["line_height"].get<double>());
                                }
                                const std::string col = pick_str(tl, {"color", "fill"});
                                if (!col.empty()) {
                                    spec.fill_color = vellum::psd::parse_hex_color(col, {0, 0, 0});
                                }
                                const std::string al = util::to_lower_ascii(pick_str(tl, {"align"}));
                                if (al == "left") spec.align = vellum::psd::TextAlign::Left;
                                else if (al == "right") spec.align = vellum::psd::TextAlign::Right;
                                else spec.align = vellum::psd::TextAlign::Center;

                                spec.faux_bold = tl.contains("bold") && tl["bold"].is_boolean() && tl["bold"].get<bool>();
                                spec.faux_italic = tl.contains("italic") && tl["italic"].is_boolean() && tl["italic"].get<bool>();
                                spec.vertical = tl.contains("vertical") && tl["vertical"].is_boolean() && tl["vertical"].get<bool>();
                                spec.box_x = tl.contains("box_x") && tl["box_x"].is_number()
                                    ? tl["box_x"].get<double>() : static_cast<double>(lx);
                                spec.box_y = tl.contains("box_y") && tl["box_y"].is_number()
                                    ? tl["box_y"].get<double>() : static_cast<double>(ly);
                                spec.box_width = tl.contains("width") && tl["width"].is_number()
                                    ? std::max(1.0, tl["width"].get<double>())
                                    : (bw > 0.0 ? bw : static_cast<double>(std::max<int32_t>(1, layer.width)));
                                spec.box_height = tl.contains("height") && tl["height"].is_number()
                                    ? std::max(1.0, tl["height"].get<double>())
                                    : (bh > 0.0 ? bh : static_cast<double>(std::max<int32_t>(1, layer.height)));
                                if (tl.contains("rotation") && tl["rotation"].is_number()) {
                                    spec.rotation_deg = tl["rotation"].get<double>();
                                }
                                if (tl.contains("stroke_width") && tl["stroke_width"].is_number() &&
                                    tl["stroke_width"].get<double>() > 0.0) {
                                    spec.outline.enabled = true;
                                    spec.outline.width_px = tl["stroke_width"].get<double>();
                                    const std::string scol = pick_str(tl, {"stroke_color", "stroke"});
                                    spec.outline.color = vellum::psd::parse_hex_color(scol, {255, 255, 255});
                                }
                                layer.type_spec = std::move(spec);
                            }
                        }

                        layers.push_back(std::move(layer));
                        added_text_layer = true;
                    }
                } else if (const std::string text_ref = pick_str(p, {"text_image", "text_layer"}); !text_ref.empty()) {
                    const auto text_bytes = decode_export_image_reference(text_ref, page_num);
                    int tw = 0, th = 0;
                    const auto text_rgba = decode_rgba(text_bytes, tw, th, "text_image");
                    layers.push_back(vellum::psd::build_cropped_rgba_layer(text_rgba.data(), tw, th, "Text"));
                    added_text_layer = true;
                }

                if (!added_text_layer) {
                    vellum::psd::PsdLayer empty_text;
                    empty_text.name = "Text";
                    empty_text.has_alpha = true;
                    layers.push_back(std::move(empty_text));
                }

                vellum::psd::PsdDocumentOptions psd_opts;
                psd_opts.width = doc_w;
                psd_opts.height = doc_h;
                psd_opts.channels = 3;
                psd_opts.dpi = 72.0;
                psd_opts.editable_text_layers = editable_text_layers;

                auto psd_bytes = vellum::psd::write_psd_bytes(psd_opts, layers);

                std::ostringstream prefix_ss;
                prefix_ss << std::setw(static_cast<int>(pad_width)) << std::setfill('0') << page_num;
                const std::string page_fn = pick_str(p, {"filename", "name"});
                const std::string entry_name = page_fn.empty()
                    ? (prefix_ss.str() + ".psd")
                    : (prefix_ss.str() + "_" + sanitize_export_filename(page_fn, "psd", "page"));
                page_psds.emplace_back(entry_name, std::move(psd_bytes));
            }

            const bool is_zip = (page_psds.size() > 1);
            const std::string out_ext = is_zip ? "zip" : "psd";
            const std::vector<uint8_t> final_bytes = is_zip
                ? build_zip_from_named_buffers(page_psds)
                : std::move(page_psds[0].second);
            const std::string download_name = sanitize_export_filename(
                output_filename_opt.has_value() ? output_filename_opt : export_name,
                out_ext,
                is_zip ? "vellum_psd" : "comic_page");

            if (!output_dir.empty()) {
                const fs::path dest_dir(util::utf8_to_wide(output_dir));
                std::error_code dir_ec;
                fs::create_directories(dest_dir, dir_ec);
                if (dir_ec) {
                    send_json(res, {{"detail", "Failed to create output_dir: " + dir_ec.message()}}, 500);
                    return;
                }
                fs::path dest_path = dest_dir / util::utf8_to_wide(download_name);
                if (!overwrite) {
                    const fs::path stem = dest_path.stem();
                    const fs::path ext = dest_path.extension();
                    int suffix_num = 1;
                    std::error_code ex_ec;
                    while (fs::exists(dest_path, ex_ec) && !ex_ec && suffix_num < 10000) {
                        dest_path = dest_dir / (stem.wstring() + L"_" + std::to_wstring(suffix_num) + ext.wstring());
                        ++suffix_num;
                    }
                }
                std::string write_err;
                if (!util::atomic_write_bytes(dest_path, final_bytes.data(), final_bytes.size(), &write_err)) {
                    send_json(res, {{"detail", "Failed to write PSD export to disk: " + write_err}}, 500);
                    return;
                }
                send_json(res, {
                    {"ok", true},
                    {"status", "ok"},
                    {"saved_path", util::path_to_utf8(dest_path)},
                    {"filename", util::path_to_utf8(dest_path.filename())},
                    {"size_bytes", final_bytes.size()},
                    {"page_count", page_psds.size()},
                    {"is_zip", is_zip},
                });
                return;
            }

            apply_cors_headers(res);
            res.set_header("X-Page-Count", std::to_string(page_psds.size()));
            res.set_header("Content-Disposition", "attachment; filename=\"" + download_name + "\"");
            res.set_content(
                reinterpret_cast<const char*>(final_bytes.data()),
                final_bytes.size(),
                is_zip ? "application/zip" : "image/vnd.adobe.photoshop");
        } catch (const vellum::psd::PsdWriteError& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        } catch (const ComicExportError& ex) {
            send_json(res, {{"detail", ex.what()}}, 400);
        } catch (const std::exception& ex) {
            send_json(res, {{"detail", std::string("PSD export failed: ") + ex.what()}}, 500);
        }
    });

    // Direct-to-disk binary file writer for manual Save As exports (`.png`, `.jpg`, etc.)
    server.Post("/api/export/save-file", [](const httplib::Request& req, httplib::Response& res) {
        if (!req.is_multipart_form_data()) {
            send_json(res, {{"detail", "Expected multipart/form-data for /api/export/save-file."}}, 400);
            return;
        }
        httplib::FormData upload;
        if (req.form.has_file("file")) {
            upload = req.form.get_file("file");
        } else if (req.form.has_file("image")) {
            upload = req.form.get_file("image");
        } else {
            send_json(res, {{"detail", "Missing 'file' or 'image' upload."}}, 400);
            return;
        }
        const std::string output_dir = req.form.has_field("output_dir")
            ? util::trim_ascii(req.form.get_field("output_dir"))
            : std::string();
        const std::string raw_filename = req.form.has_field("output_filename")
            ? util::trim_ascii(req.form.get_field("output_filename"))
            : util::trim_ascii(upload.filename);
        if (output_dir.empty() || raw_filename.empty()) {
            send_json(res, {{"detail", "Both 'output_dir' and 'output_filename' are required."}}, 400);
            return;
        }
        bool overwrite = true;
        if (req.form.has_field("overwrite")) {
            overwrite = parse_bool_param(req.form.get_field("overwrite"), true);
        }

        const fs::path dest_dir(util::utf8_to_wide(output_dir));
        std::error_code dir_ec;
        fs::create_directories(dest_dir, dir_ec);
        if (dir_ec) {
            send_json(res, {{"detail", "Failed to create output_dir: " + dir_ec.message()}}, 500);
            return;
        }
        std::string ext = "png";
        std::string stem_only = raw_filename;
        if (const auto dot = raw_filename.rfind('.'); dot != std::string::npos && dot + 1 < raw_filename.size()) {
            ext = util::to_lower_ascii(raw_filename.substr(dot + 1));
            stem_only = raw_filename.substr(0, dot);
        }
        const std::string safe_name = sanitize_export_filename(stem_only, ext, "comic_page");
        fs::path dest_path = dest_dir / util::utf8_to_wide(safe_name);
        if (!overwrite) {
            const fs::path stem = dest_path.stem();
            const fs::path ext_p = dest_path.extension();
            int suffix_num = 1;
            std::error_code ex_ec;
            while (fs::exists(dest_path, ex_ec) && !ex_ec && suffix_num < 10000) {
                dest_path = dest_dir / (stem.wstring() + L"_" + std::to_wstring(suffix_num) + ext_p.wstring());
                ++suffix_num;
            }
        }

        std::string write_err;
        if (!util::atomic_write_bytes(dest_path, upload.content.data(), upload.content.size(), &write_err)) {
            send_json(res, {{"detail", "Failed to save file to disk: " + write_err}}, 500);
            return;
        }
        send_json(res, {
            {"ok", true},
            {"saved_path", util::path_to_utf8(dest_path)},
            {"filename", util::path_to_utf8(dest_path.filename())},
            {"size_bytes", upload.content.size()},
        });
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
                std::string output_dir_str;
                std::optional<std::string> output_filename_opt;
                bool overwrite = false;
                if (!util::trim_ascii(req.body).empty()) {
                    const json body = json::parse(req.body, nullptr, false);
                    if (body.is_object()) {
                        if (!return_url && body.contains("return_url") && body["return_url"].is_boolean()) {
                            return_url = body["return_url"].get<bool>();
                        }
                        if (body.contains("output_dir") && body["output_dir"].is_string()) {
                            output_dir_str = util::trim_ascii(body["output_dir"].get<std::string>());
                        }
                        if (body.contains("output_filename") && body["output_filename"].is_string()) {
                            const std::string fn = util::trim_ascii(body["output_filename"].get<std::string>());
                            if (!fn.empty()) output_filename_opt = fn;
                        }
                        if (body.contains("overwrite") && body["overwrite"].is_boolean()) {
                            overwrite = body["overwrite"].get<bool>();
                        }
                    }
                }

                if (!output_dir_str.empty()) {
                    const fs::path dest_dir(util::utf8_to_wide(output_dir_str));
                    std::error_code dir_ec;
                    fs::create_directories(dest_dir, dir_ec);
                    if (dir_ec) {
                        cleanup_dir();
                        send_json(res, {{"detail", "Failed to create output_dir: " + dir_ec.message()}}, 500);
                        return;
                    }
                    const std::string base_name = output_filename_opt.has_value()
                        ? sanitize_export_filename(output_filename_opt, st.format)
                        : download_name;
                    fs::path dest_path = dest_dir / util::utf8_to_wide(base_name);
                    if (!overwrite) {
                        const fs::path stem = dest_path.stem();
                        const fs::path ext = dest_path.extension();
                        int suffix_num = 1;
                        std::error_code ex_ec;
                        while (fs::exists(dest_path, ex_ec) && !ex_ec && suffix_num < 10000) {
                            dest_path = dest_dir / (stem.wstring() + L"_" + std::to_wstring(suffix_num) + ext.wstring());
                            ++suffix_num;
                        }
                    }
                    std::error_code cp_ec;
                    fs::copy_file(out_file, dest_path, fs::copy_options::overwrite_existing, cp_ec);
                    if (cp_ec) {
                        cleanup_dir();
                        send_json(res, {{"detail", "Failed to write finalized export to output_dir: " + cp_ec.message()}}, 500);
                        return;
                    }

                    CompletedExportState done;
                    done.file_path = out_file;
                    done.session_dir = st.session_dir;
                    done.download_name = util::path_to_utf8(dest_path.filename());
                    done.mime_type = mime;
                    done.file_size = file_size;
                    done.page_count = ordered_pages.size();
                    {
                        std::lock_guard<std::mutex> lock(ctx.export_sessions_mutex);
                        ctx.completed_exports[sid] = std::move(done);
                    }
                    send_json(res, {
                        {"ok", true},
                        {"status", "ok"},
                        {"session_id", sid},
                        {"saved_path", util::path_to_utf8(dest_path)},
                        {"download_url", "/api/export/session/" + sid + "/download"},
                        {"filename", util::path_to_utf8(dest_path.filename())},
                        {"size_bytes", file_size},
                        {"page_count", ordered_pages.size()},
                    });
                    return;
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

    // #16: Portable Settings + Series Profiles + Fonts Backup Bundle (.zip)
    server.Get("/api/backup/export", [&ctx](const httplib::Request& req, httplib::Response& res) {
        bool include_keys = false;
        if (req.has_param("include_keys")) {
            include_keys = parse_bool_param(req.get_param_value("include_keys"), false);
        }

        mz_zip_archive zip{};
        if (!mz_zip_writer_init_heap(&zip, 0, 256 * 1024)) {
            send_json(res, {{"detail", "Failed to initialize backup ZIP writer."}}, 500);
            return;
        }

        auto strip_keys_recursive = [](auto&& self, json& node) -> void {
            if (node.is_object()) {
                for (auto it = node.begin(); it != node.end(); ++it) {
                    if (it.key() == "api_key" && it.value().is_string()) {
                        it.value() = "";
                    } else {
                        self(self, it.value());
                    }
                }
            } else if (node.is_array()) {
                for (auto& el : node) self(self, el);
            }
        };

        json settings_copy = ctx.settings_manager.load();
        if (!include_keys) {
            strip_keys_recursive(strip_keys_recursive, settings_copy);
        }
        const std::string settings_str = settings_copy.dump(2, ' ', false, json::error_handler_t::replace);
        mz_zip_writer_add_mem(&zip, "settings.json", settings_str.data(), settings_str.size(), MZ_BEST_SPEED);

        std::error_code ec;
        const fs::path profiles_dir = fs::current_path() / L"series_profiles";
        if (fs::is_directory(profiles_dir, ec)) {
            for (const auto& entry : fs::directory_iterator(profiles_dir, ec)) {
                if (!entry.is_regular_file(ec)) continue;
                const std::string ext = util::to_lower_ascii(util::path_to_utf8(entry.path().extension()));
                if (ext != ".json") continue;
                const auto bytes = util::read_file_bytes(entry.path());
                if (bytes.has_value() && !bytes->empty()) {
                    const std::string arc_name = "series_profiles/" + util::path_to_utf8(entry.path().filename());
                    mz_zip_writer_add_mem(&zip, arc_name.c_str(), bytes->data(), bytes->size(), MZ_BEST_SPEED);
                }
            }
        }

        const fs::path fonts_dir = fs::current_path() / L"fonts";
        if (fs::is_directory(fonts_dir, ec)) {
            for (const auto& entry : fs::directory_iterator(fonts_dir, ec)) {
                if (!entry.is_regular_file(ec)) continue;
                const std::string ext = util::to_lower_ascii(util::path_to_utf8(entry.path().extension()));
                if (ext != ".ttf" && ext != ".otf" && ext != ".ttc" && ext != ".woff" && ext != ".woff2") continue;
                const auto bytes = util::read_file_bytes(entry.path());
                if (bytes.has_value() && !bytes->empty()) {
                    const std::string arc_name = "fonts/" + util::path_to_utf8(entry.path().filename());
                    mz_zip_writer_add_mem(&zip, arc_name.c_str(), bytes->data(), bytes->size(), MZ_BEST_SPEED);
                }
            }
        }

        void* zip_buf = nullptr;
        size_t zip_size = 0;
        const bool ok = mz_zip_writer_finalize_heap_archive(&zip, &zip_buf, &zip_size) != 0;
        mz_zip_writer_end(&zip);
        if (!ok || !zip_buf || zip_size == 0) {
            if (zip_buf) mz_free(zip_buf);
            send_json(res, {{"detail", "Failed to finalize backup ZIP archive."}}, 500);
            return;
        }

        apply_cors_headers(res);
        res.set_header("Content-Disposition", "attachment; filename=\"vellum-backup.zip\"");
        res.set_content(reinterpret_cast<const char*>(zip_buf), zip_size, "application/zip");
        mz_free(zip_buf);
    });

    server.Post("/api/backup/import", [&ctx](const httplib::Request& req, httplib::Response& res) {
        std::string zip_bytes;
        if (req.is_multipart_form_data()) {
            if (req.form.has_file("file")) {
                zip_bytes = req.form.get_file("file").content;
            } else if (req.form.has_file("backup")) {
                zip_bytes = req.form.get_file("backup").content;
            }
        } else {
            zip_bytes = req.body;
        }
        if (zip_bytes.empty()) {
            send_json(res, {{"detail", "Missing backup .zip payload."}}, 400);
            return;
        }

        mz_zip_archive zip{};
        if (!mz_zip_reader_init_mem(&zip, zip_bytes.data(), zip_bytes.size(), 0)) {
            send_json(res, {{"detail", "Invalid or corrupted backup .zip archive."}}, 400);
            return;
        }

        auto is_safe_entry_name = [](const std::string& p) -> bool {
            if (p.empty() || p[0] == '/' || p[0] == '\\') return false;
            if (p.find("..") != std::string::npos || p.find('\\') != std::string::npos || p.find(':') != std::string::npos) {
                return false;
            }
            return true;
        };

        int imported_profiles = 0;
        int imported_fonts = 0;
        bool imported_settings = false;
        std::error_code ec;

        const mz_uint num_files = mz_zip_reader_get_num_files(&zip);
        for (mz_uint i = 0; i < num_files; ++i) {
            mz_zip_archive_file_stat st{};
            if (!mz_zip_reader_file_stat(&zip, i, &st) || st.m_is_directory) continue;
            const std::string name(st.m_filename);
            if (!is_safe_entry_name(name)) continue;

            size_t out_sz = 0;
            void* raw = mz_zip_reader_extract_to_heap(&zip, i, &out_sz, 0);
            if (!raw) continue;
            std::vector<uint8_t> data(static_cast<const uint8_t*>(raw), static_cast<const uint8_t*>(raw) + out_sz);
            mz_free(raw);

            if (name == "settings.json") {
                const json incoming = json::parse(data.begin(), data.end(), nullptr, false);
                if (!incoming.is_discarded() && incoming.is_object()) {
                    json merged = ctx.settings_manager.load();
                    for (auto it = incoming.begin(); it != incoming.end(); ++it) {
                        if (it.key() == "api_key") {
                            if (it.value().is_string() && !util::trim_ascii(it.value().get<std::string>()).empty()) {
                                merged["api_key"] = it.value();
                            }
                            continue;
                        }
                        if (it.key() == "providers" && it.value().is_object() && merged.contains("providers") && merged["providers"].is_object()) {
                            for (auto pit = it.value().begin(); pit != it.value().end(); ++pit) {
                                if (!pit.value().is_object()) continue;
                                json p_obj = pit.value();
                                if ((!p_obj.contains("api_key") || !p_obj["api_key"].is_string() ||
                                     util::trim_ascii(p_obj["api_key"].get<std::string>()).empty()) &&
                                    merged["providers"].contains(pit.key()) &&
                                    merged["providers"][pit.key()].is_object() &&
                                    merged["providers"][pit.key()].contains("api_key")) {
                                    p_obj["api_key"] = merged["providers"][pit.key()]["api_key"];
                                }
                                merged["providers"][pit.key()] = std::move(p_obj);
                            }
                            continue;
                        }
                        merged[it.key()] = it.value();
                    }
                    ctx.settings_manager.update(merged);
                    imported_settings = true;
                }
            } else if (name.rfind("series_profiles/", 0) == 0) {
                const std::string fname = name.substr(16);
                if (fname.find('/') == std::string::npos &&
                    util::to_lower_ascii(util::path_to_utf8(fs::path(util::utf8_to_wide(fname)).extension())) == ".json") {
                    const fs::path dir = fs::current_path() / L"series_profiles";
                    fs::create_directories(dir, ec);
                    std::string werr;
                    if (util::atomic_write_bytes(dir / util::utf8_to_wide(fname), data.data(), data.size(), &werr)) {
                        ++imported_profiles;
                    }
                }
            } else if (name.rfind("fonts/", 0) == 0) {
                const std::string fname = name.substr(6);
                const std::string ext = util::to_lower_ascii(util::path_to_utf8(fs::path(util::utf8_to_wide(fname)).extension()));
                if (fname.find('/') == std::string::npos &&
                    (ext == ".ttf" || ext == ".otf" || ext == ".ttc" || ext == ".woff" || ext == ".woff2")) {
                    const fs::path dir = fs::current_path() / L"fonts";
                    fs::create_directories(dir, ec);
                    std::string werr;
                    if (util::atomic_write_bytes(dir / util::utf8_to_wide(fname), data.data(), data.size(), &werr)) {
                        ++imported_fonts;
                    }
                }
            }
        }
        mz_zip_reader_end(&zip);

        send_json(res, {
            {"ok", true},
            {"imported_settings", imported_settings},
            {"imported_profiles", imported_profiles},
            {"imported_fonts", imported_fonts},
        });
    });

}

}  // namespace mangaflow::routes
