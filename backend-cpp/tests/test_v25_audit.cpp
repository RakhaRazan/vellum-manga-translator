#include "../src/detector/cv_geometry.hpp"
#include "../src/detector/cv_refinement.hpp"
#include "../src/detector/cv_typography.hpp"
#include "../src/exporter.hpp"
#include "../src/pipeline/detector.hpp"
#include "../src/pipeline/inpainter.hpp"
#include "../src/pipeline/model_manager.hpp"
#include "../src/pipeline/ocr.hpp"
#include "../src/pipeline/ort_provider.hpp"
#include "../src/settings_manager.hpp"
#include "../src/utils.hpp"

#include <miniz.h>
#include <nlohmann/json.hpp>
#include <stb_image.h>
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

int g_pass = 0;
int g_fail = 0;

#define AUDIT_CHECK(cond, msg)                                                 \
    do {                                                                       \
        if (cond) {                                                            \
            ++g_pass;                                                          \
            std::cout << "  [PASS] " << (msg) << "\n";                         \
        } else {                                                               \
            ++g_fail;                                                          \
            std::cerr << "  [FAIL] " << (msg) << "\n";                         \
        }                                                                      \
    } while (0)

double median_val(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

bool load_rgb(const fs::path& p, std::vector<uint8_t>& rgb, int& w, int& h) {
    int c = 0;
    unsigned char* raw = stbi_load(p.u8string().c_str(), &w, &h, &c, 3);
    if (!raw || w <= 0 || h <= 0) return false;
    rgb.assign(raw, raw + static_cast<size_t>(w) * static_cast<size_t>(h) * 3ULL);
    stbi_image_free(raw);
    return true;
}

// -----------------------------------------------------------------------------
// Audit 1: Phase A CV Latency, 3-Tier CPU Inpainting Bypass & Tone Match
// -----------------------------------------------------------------------------
void run_audit_1_cv_and_inpainting(const fs::path& repo_root) {
    std::cout << "\n============================================================\n";
    std::cout << "AUDIT 1: CV Phase A Latency, 3-Tier Inpainting & Tone Match\n";
    std::cout << "============================================================\n";

    mangaflow::pipeline::ModelManager mm(repo_root / L"models");
    mangaflow::pipeline::TextDetector det(mm);
    mangaflow::pipeline::LamaInpainter inp(mm);

    const std::vector<std::pair<std::string, std::string>> samples = {
        {"sample_ja.jpg", "ja"},
        {"sample_ko.jpg", "ko"},
        {"sample_zh.jpg", "zh"},
    };

    int total_real_bubbles = 0;
    int total_real_t1 = 0;
    int total_real_t2 = 0;
    int total_real_t3 = 0;
    int total_real_gpu_skipped_pages = 0;

    for (const auto& [fname, lang] : samples) {
        const fs::path img_path = repo_root / L"samples" / fs::u8path(fname);
        std::vector<uint8_t> rgb;
        int w = 0, h = 0;
        if (!load_rgb(img_path, rgb, w, h)) {
            AUDIT_CHECK(false, "Failed to load " + fname);
            continue;
        }

        // Run detector once to get real bubbles/panels/mask (using CTD for fast CPU/GPU detection)
        mangaflow::pipeline::DetectionOptions dopts;
        dopts.detection_model = "comic-text-detector";
        dopts.detector_backend = "ctd";
        dopts.source_language = lang;
        dopts.reading_order = (lang == "ja") ? "rtl" : "ltr";
        dopts.auto_unload = false;
        dopts.enable_cv_style_detect = true;
        dopts.enable_waist_cut = true;
        const auto dres = det.detect_rgb(rgb.data(), w, h, dopts);

        // Measure pure Phase A CV CPU latency over 5 runs on the real image + real detections
        const bool is_rtl = (lang == "ja");
        std::vector<vellum::detector::RectI> text_rects;
        std::vector<vellum::detector::RoomSeedBlock> seeds;
        for (size_t i = 0; i < dres.bubbles.size(); ++i) {
            const auto& b = dres.bubbles[i];
            vellum::detector::RectI r{b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3]};
            text_rects.push_back(r);
            vellum::detector::RoomSeedBlock s;
            s.block_index = static_cast<int>(i);
            s.bbox = r;
            seeds.push_back(s);
        }
        std::vector<vellum::detector::RectI> bubble_rects;
        for (const auto& sb : dres.speech_bubbles) {
            bubble_rects.push_back({sb.bbox[0], sb.bbox[1], sb.bbox[2], sb.bbox[3]});
        }

        std::vector<double> phase_a_ms;
        size_t max_poly_pts = 0;
        int valid_styles = 0;
        for (int run = 0; run < 5; ++run) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto panel_rects = vellum::detector::detectPanelsClassicCV(rgb.data(), w, h);
            const auto perm = vellum::detector::sortReadingOrderPanelFirst(text_rects, panel_rects, is_rtl);
            (void)perm;
            const auto room_polys = vellum::detector::computePageRoomPolygons(
                rgb.data(), w, h, seeds, bubble_rects, &dres.mask, /*enable_waist_cut=*/true);
            const vellum::detector::RectI page_rect{0, 0, w, h};
            int styles_this_run = 0;
            for (const auto& b : dres.bubbles) {
                const vellum::detector::RectI tb{b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3]};
                const auto st = vellum::detector::analyzeBlockTypographyFromBox(
                    rgb.data(), w, h, tb, b.text_direction == "vertical",
                    b.category == "onomatopoeia", &page_rect, &dres.mask);
                if (st.valid) ++styles_this_run;
            }
            const auto t1 = std::chrono::steady_clock::now();
            phase_a_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            valid_styles = styles_this_run;
            for (const auto& rp : room_polys) {
                max_poly_pts = std::max(max_poly_pts, rp.size());
            }
        }

        const double med_cv_ms = median_val(phase_a_ms);
        std::cout << "  [" << fname << "] " << w << "x" << h
                  << " | bubbles=" << dres.bubbles.size()
                  << " | Phase A CPU p50=" << std::fixed << std::setprecision(2) << med_cv_ms << " ms (+0 MB VRAM)"
                  << " | max_room_poly_pts=" << max_poly_pts
                  << " | styles_valid=" << valid_styles << "/" << dres.bubbles.size() << "\n";
        AUDIT_CHECK(max_poly_pts <= 32, fname + " room_polygon <= 32 vertices");

        // Run 3-Tier CPU Inpainting on real sample
        std::vector<std::array<int, 4>> bboxes;
        for (const auto& b : dres.bubbles) bboxes.push_back(b.bbox);

        mangaflow::pipeline::InpaintOptions iopts;
        iopts.inpainting_model = "lama-manga";
        iopts.enable_3tier_inpaint = true;
        iopts.enable_tone_match = true;
        iopts.refine_mask = true;
        iopts.auto_unload = false;
        iopts.bboxes = bboxes;

        const auto t_inp0 = std::chrono::steady_clock::now();
        const auto ires = inp.inpaint_rgb(rgb.data(), w, h, dres.mask, iopts);
        const auto t_inp1 = std::chrono::steady_clock::now();
        const double inp_ms = std::chrono::duration<double, std::milli>(t_inp1 - t_inp0).count();

        const int btot = ires.tier1_blocks + ires.tier2_blocks + ires.tier3_blocks;
        const double bypass_pct = (btot > 0) ? (100.0 * (ires.tier1_blocks + ires.tier2_blocks) / btot) : 0.0;
        total_real_bubbles += btot;
        total_real_t1 += ires.tier1_blocks;
        total_real_t2 += ires.tier2_blocks;
        total_real_t3 += ires.tier3_blocks;
        if (ires.gpu_skipped) ++total_real_gpu_skipped_pages;

        std::cout << "  [" << fname << "] 3-Tier Inpaint: T1=" << ires.tier1_blocks
                  << " T2=" << ires.tier2_blocks << " T3=" << ires.tier3_blocks
                  << " | bypass=" << std::fixed << std::setprecision(1) << bypass_pct << "%"
                  << " | gpu_skipped=" << (ires.gpu_skipped ? "true" : "false")
                  << " | tiles=" << ires.tiles_processed
                  << " | tone_match=" << (ires.tone_match_applied ? "true" : "false")
                  << " | time=" << std::setprecision(2) << inp_ms << " ms"
                  << " | ep=" << ires.execution_provider << "\n";
    }

    const double overall_bypass = (total_real_bubbles > 0)
        ? (100.0 * (total_real_t1 + total_real_t2) / total_real_bubbles)
        : 0.0;
    std::cout << "  [Real Samples Summary] Total blocks=" << total_real_bubbles
              << " | T1=" << total_real_t1 << " T2=" << total_real_t2 << " T3=" << total_real_t3
              << " | Overall CPU Bypass=" << std::fixed << std::setprecision(1) << overall_bypass << "%"
              << " | GPU-skipped pages=" << total_real_gpu_skipped_pages << "/3\n";

    // Synthetic Test 1: Plain white bubble -> MUST finish in Tier 1 with gpu_skipped == true
    {
        const int W = 512, H = 512;
        std::vector<uint8_t> img(W * H * 3, 250);
        std::vector<uint8_t> mask(W * H, 0);
        // Draw dark text inside [180..320, 180..320]
        for (int y = 210; y < 290; ++y) {
            for (int x = 210; x < 290; ++x) {
                const size_t idx = static_cast<size_t>(y * W + x);
                img[idx * 3 + 0] = 15;
                img[idx * 3 + 1] = 15;
                img[idx * 3 + 2] = 15;
                mask[idx] = 255;
            }
        }
        mangaflow::pipeline::InpaintOptions iopts;
        iopts.enable_3tier_inpaint = true;
        iopts.enable_tone_match = true;
        iopts.refine_mask = false;
        iopts.bboxes = std::vector<std::array<int, 4>>{{200, 200, 300, 300}};
        const auto res = inp.inpaint_rgb(img.data(), W, H, mask, iopts);
        AUDIT_CHECK(res.tier1_blocks == 1 && res.tier3_blocks == 0 && res.gpu_skipped && res.tiles_processed == 0,
                    "Synthetic plain-white bubble finishes in Tier 1 without invoking LaMa GPU");
    }

    // Synthetic Test 2: Linear gradient bubble -> MUST finish in Tier 2 (Telea FMM) with gpu_skipped == true
    {
        const int W = 512, H = 512;
        std::vector<uint8_t> img(W * H * 3, 0);
        std::vector<uint8_t> mask(W * H, 0);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const size_t idx = static_cast<size_t>(y * W + x);
                const uint8_t v = static_cast<uint8_t>(std::clamp(140 + (x - 150) / 3 + (y - 150) / 4, 0, 255));
                img[idx * 3 + 0] = v;
                img[idx * 3 + 1] = static_cast<uint8_t>(std::min(255, v + 5));
                img[idx * 3 + 2] = static_cast<uint8_t>(std::max(0, v - 5));
            }
        }
        // Put masked hole in center where gradient range > 20 (exceeds Tier 1 mean_dev=10, fits linear plane in Tier 2)
        for (int y = 200; y < 320; ++y) {
            for (int x = 180; x < 340; ++x) {
                const size_t idx = static_cast<size_t>(y * W + x);
                img[idx * 3 + 0] = 10;
                img[idx * 3 + 1] = 10;
                img[idx * 3 + 2] = 10;
                mask[idx] = 255;
            }
        }
        mangaflow::pipeline::InpaintOptions iopts;
        iopts.enable_3tier_inpaint = true;
        iopts.enable_tone_match = true;
        iopts.refine_mask = false;
        iopts.bboxes = std::vector<std::array<int, 4>>{{180, 200, 340, 320}};
        const auto res = inp.inpaint_rgb(img.data(), W, H, mask, iopts);
        AUDIT_CHECK(res.tier2_blocks == 1 && res.tier3_blocks == 0 && res.gpu_skipped && res.tiles_processed == 0,
                    "Synthetic linear-gradient bubble finishes in Tier 2 (Telea FMM) without invoking LaMa GPU");
    }

    // Synthetic Test 3: Screentone/textured bubble -> Tier 3 LaMa + Tone Match (verify zero NaN/clamping artifacts)
    {
        const int W = 512, H = 512;
        std::vector<uint8_t> img(W * H * 3, 0);
        std::vector<uint8_t> mask(W * H, 0);
        // Screentone checkerboard pattern (high variance -> triggers Tier 3 LaMa + Tone Match)
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const size_t idx = static_cast<size_t>(y * W + x);
                const bool dot = (((x / 4) + (y / 4)) % 2) == 0;
                const uint8_t v = dot ? 215 : 135;
                img[idx * 3 + 0] = v;
                img[idx * 3 + 1] = static_cast<uint8_t>(v - 10);
                img[idx * 3 + 2] = static_cast<uint8_t>(v - 5);
            }
        }
        for (int y = 216; y < 296; ++y) {
            for (int x = 216; x < 296; ++x) {
                const size_t idx = static_cast<size_t>(y * W + x);
                img[idx * 3 + 0] = 5;
                img[idx * 3 + 1] = 5;
                img[idx * 3 + 2] = 5;
                mask[idx] = 255;
            }
        }
        mangaflow::pipeline::InpaintOptions iopts;
        iopts.inpainting_model = "lama-manga";
        iopts.enable_3tier_inpaint = true;
        iopts.enable_tone_match = true;
        iopts.refine_mask = false;
        iopts.auto_unload = true;
        iopts.bboxes = std::vector<std::array<int, 4>>{{216, 216, 296, 296}};
        const auto res = inp.inpaint_rgb(img.data(), W, H, mask, iopts);

        // Verify output pixels inside hole have no extreme 0/255 clamping artifacts (expected mean ~170)
        int clamped_zero = 0;
        int clamped_255 = 0;
        double sum_luma = 0.0;
        int hole_cnt = 0;
        for (int y = 216; y < 296; ++y) {
            for (int x = 216; x < 296; ++x) {
                const size_t idx = static_cast<size_t>(y * W + x) * 3ULL;
                const uint8_t r = res.rgb[idx + 0];
                const uint8_t g = res.rgb[idx + 1];
                const uint8_t b = res.rgb[idx + 2];
                if (r == 0 && g == 0 && b == 0) ++clamped_zero;
                if (r == 255 && g == 255 && b == 255) ++clamped_255;
                sum_luma += (0.299 * r + 0.587 * g + 0.114 * b);
                ++hole_cnt;
            }
        }
        const double mean_luma = hole_cnt > 0 ? (sum_luma / hole_cnt) : 0.0;
        std::cout << "  [Tone Match Screentone Test] T3=" << res.tier3_blocks
                  << " tiles=" << res.tiles_processed
                  << " tone_match=" << (res.tone_match_applied ? "true" : "false")
                  << " mean_luma=" << std::fixed << std::setprecision(1) << mean_luma
                  << " clamped_0=" << clamped_zero << " clamped_255=" << clamped_255 << "\n";
        AUDIT_CHECK(res.tier3_blocks == 1 && res.tiles_processed == 1 && res.tone_match_applied &&
                        clamped_zero == 0 && clamped_255 == 0 && mean_luma > 120.0 && mean_luma < 220.0,
                    "Tier 3 LaMa + Tone Match executes with zero NaN / zero 0-or-255 clamping artifacts");
    }

    det.unload();
    inp.unload();
}

// -----------------------------------------------------------------------------
// Audit 2: Hardware Safety & CUDA 8 GB Guard
// -----------------------------------------------------------------------------
void run_audit_2_hardware_and_cuda_guard(const fs::path& repo_root) {
    std::cout << "\n============================================================\n";
    std::cout << "AUDIT 2: Hardware Safety, CUDA 8 GB Guard & Hybrid OCR CPU\n";
    std::cout << "============================================================\n";

    const bool has_nvcuda = mangaflow::pipeline::probe_nvcuda_dll();
    const auto caps = mangaflow::pipeline::detect_hardware_capabilities(/*force_cpu=*/false);
    const auto caps_cpu = mangaflow::pipeline::detect_hardware_capabilities(/*force_cpu=*/true);

    std::cout << "  nvcuda.dll present   : " << (has_nvcuda ? "true" : "false") << "\n";
    std::cout << "  Detected GPU         : " << caps.gpu_name << " (" << caps.vram_mb << " MB VRAM)\n";
    std::cout << "  System RAM           : " << caps.total_ram_mb << " MB (avail " << caps.available_ram_mb << " MB)\n";
    std::cout << "  Primary EP (auto)    : " << caps.execution_provider << " (backend=" << caps.backend << ")\n";
    std::cout << "  Primary EP (cpu)     : " << caps_cpu.execution_provider << "\n";
    std::cout << "  CUDA mem limit (def) : " << caps.cuda_mem_limit_mb << " MB\n";

    AUDIT_CHECK(caps_cpu.execution_provider == "CPUExecutionProvider",
                "force_cpu=true resolves cleanly to CPUExecutionProvider");

    // Verify safe CUDA append never crashes even on non-CUDA / DML-only builds
    {
        Ort::SessionOptions opts;
        const bool cuda_ok = mangaflow::pipeline::append_gpu_execution_provider(
            opts, "CUDAExecutionProvider", /*device_id=*/0);
        std::cout << "  append_gpu_execution_provider(CUDA) returned: " << (cuda_ok ? "true" : "false")
                  << " (zero crash)\n";
        const auto safe_eps = mangaflow::pipeline::get_safe_available_providers();
        const bool has_cuda_ep =
            std::find(safe_eps.begin(), safe_eps.end(), "CUDAExecutionProvider") != safe_eps.end();
        if (!has_nvcuda || !has_cuda_ep) {
            AUDIT_CHECK(!cuda_ok, "CUDA provider probe returns false cleanly without crashing when CUDA EP unavailable");
        } else {
            AUDIT_CHECK(cuda_ok, "CUDA provider appended safely with kSameAsRequested + gpu_mem_limit");
        }
    }

    // Verify configurable gpu_mem_limit clamping
    {
        const uint64_t orig = mangaflow::pipeline::get_cuda_mem_limit_mb();
        AUDIT_CHECK(orig == mangaflow::pipeline::kDefaultCudaMemLimitMb,
                    "Default CUDA VRAM cap is 3584 MB (3.5 GB)");
        mangaflow::pipeline::set_cuda_mem_limit_mb(100);
        const uint64_t lo = mangaflow::pipeline::get_cuda_mem_limit_mb();
        mangaflow::pipeline::set_cuda_mem_limit_mb(999999);
        const uint64_t hi = mangaflow::pipeline::get_cuda_mem_limit_mb();
        mangaflow::pipeline::set_cuda_mem_limit_mb(orig);
        AUDIT_CHECK(lo == 512 && hi == 65536 &&
                        mangaflow::pipeline::get_cuda_mem_limit_bytes() == static_cast<size_t>(orig) * 1024ULL * 1024ULL,
                    "CUDA gpu_mem_limit clamping [512 MB, 65536 MB] and byte conversion verified");
    }

    // Verify OCR stays on CPU in Smart Hybrid mode (main.cpp:913 -> force_cpu = (mode != "force_gpu"))
    {
        mangaflow::pipeline::ModelManager mm(repo_root / L"models");
        mangaflow::pipeline::OcrEngine ocr(mm);
        std::vector<uint8_t> rgb;
        int w = 0, h = 0;
        if (load_rgb(repo_root / L"samples" / L"sample_ko.jpg", rgb, w, h)) {
            mangaflow::pipeline::OcrBubbleInput b;
            b.id = "b_0";
            b.order = 0;
            b.bbox = {50, 50, 260, 260};
            b.polygon = {{50, 50}, {260, 50}, {260, 260}, {50, 260}};
            b.text_direction = "horizontal";

            mangaflow::pipeline::OcrOptions oopts;
            oopts.source_lang = "ko";
            oopts.ocr_model = "rapidocr-ppocrv4";
            oopts.force_cpu = true;  // Smart Hybrid default
            oopts.auto_unload = true;

            const auto ores = ocr.recognize_rgb(rgb.data(), w, h, {b}, oopts);
            std::cout << "  [Smart Hybrid OCR] model=" << ores.ocr_model
                      << " ep=" << ores.execution_provider << "\n";
            AUDIT_CHECK(ores.execution_provider == "CPUExecutionProvider",
                        "OCR stays on CPUExecutionProvider in Smart Hybrid mode");
        }
    }
}

// -----------------------------------------------------------------------------
// Audit 3: 10,000-Page Endurance & Format Integrity (RDP JSON + ZIP64 >4GB + PDF)
// -----------------------------------------------------------------------------
void run_audit_3_endurance_and_formats(const fs::path& repo_root) {
    std::cout << "\n============================================================\n";
    std::cout << "AUDIT 3: 10,000-Page Endurance, ZIP64 (>4 GB) CBZ & PDF Tests\n";
    std::cout << "============================================================\n";

    // 3A. Synthetic 10,000-page autosave serialization with RDP-simplified room_polygon
    {
        constexpr int kPages = 10000;
        constexpr int kBubblesPerPage = 6;

        // Build a realistic 200-vertex oval contour and simplify it with real simplifyPolygonRDP
        std::vector<vellum::detector::Point2f> raw_oval;
        raw_oval.reserve(200);
        for (int i = 0; i < 200; ++i) {
            const double ang = (2.0 * 3.14159265358979323846 * i) / 200.0;
            raw_oval.push_back({
                static_cast<float>(200.0 + 95.0 * std::cos(ang)),
                static_cast<float>(300.0 + 135.0 * std::sin(ang))
            });
        }
        const auto simplified = vellum::detector::simplifyPolygonRDP(raw_oval, 0.008, 32);
        std::cout << "  [RDP Polygon] raw=" << raw_oval.size()
                  << " pts -> simplified=" << simplified.size() << " pts\n";
        AUDIT_CHECK(simplified.size() >= 3 && simplified.size() <= 32,
                    "RDP-simplified room_polygon is bounded to <= 32 vertices");

        json rpoly_json = json::array();
        for (const auto& pt : simplified) {
            rpoly_json.push_back({std::round(pt.x * 10.0) / 10.0, std::round(pt.y * 10.0) / 10.0});
        }

        const auto t_build0 = std::chrono::steady_clock::now();
        json pages = json::array();
        for (int p = 0; p < kPages; ++p) {
            json blocks = json::array();
            for (int b = 0; b < kBubblesPerPage; ++b) {
                blocks.push_back({
                    {"id", "b_" + std::to_string(b)},
                    {"order", b},
                    {"bbox", {100 + b * 20, 150 + b * 30, 290 + b * 20, 420 + b * 30}},
                    {"safe_bbox", {112 + b * 20, 165 + b * 30, 278 + b * 20, 405 + b * 30}},
                    {"room_polygon", rpoly_json},
                    {"original_text", "テスト台詞"},
                    {"translated_text", "Sample dialogue line."},
                    {"font_size", 18},
                    {"angle_deg", 0.0},
                    {"text_color", {12, 12, 12}},
                });
            }
            pages.push_back({
                {"id", "page_" + std::to_string(p + 1)},
                {"name", "page_" + std::to_string(p + 1) + ".jpg"},
                {"originalUrl", "/api/cache/images/imp_" + std::to_string(p + 1) + ".jpg"},
                {"inpaintedUrl", "/api/cache/images/clean_" + std::to_string(p + 1) + ".png"},
                {"dimensions", {{"width", 1000}, {"height", 1400}}},
                {"status", "done"},
                {"blocks", std::move(blocks)},
            });
        }
        json snapshot = {
            {"format", "vellum-autosave"},
            {"savedAt", "2026-11-24T00:00:00.000Z"},
            {"pages", std::move(pages)},
        };
        const auto t_ser0 = std::chrono::steady_clock::now();
        const std::string dumped = snapshot.dump();
        const auto t_ser1 = std::chrono::steady_clock::now();

        const auto t_parse0 = std::chrono::steady_clock::now();
        const json reparsed = json::parse(dumped);
        const auto t_parse1 = std::chrono::steady_clock::now();

        const double size_mb = static_cast<double>(dumped.size()) / (1024.0 * 1024.0);
        const double ser_ms = std::chrono::duration<double, std::milli>(t_ser1 - t_ser0).count();
        const double parse_ms = std::chrono::duration<double, std::milli>(t_parse1 - t_parse0).count();
        const double total_ms = std::chrono::duration<double, std::milli>(t_ser1 - t_build0).count();

        // Estimate unsimplified 200-pt bloat for comparison
        const double unsimplified_extra_mb =
            (static_cast<double>(kPages) * kBubblesPerPage * (200 - simplified.size()) * 14.0) / (1024.0 * 1024.0);

        std::cout << "  [10,000-Page Autosave] pages=" << kPages
                  << " (60,000 bubbles) | JSON size=" << std::fixed << std::setprecision(2) << size_mb << " MB"
                  << " (vs ~" << (size_mb + unsimplified_extra_mb) << " MB unsimplified)"
                  << " | serialize=" << ser_ms << " ms | parse=" << parse_ms << " ms | total=" << total_ms << " ms\n";
        AUDIT_CHECK(reparsed["pages"].size() == kPages && size_mb < 60.0,
                    "10,000-page autosave JSON stays lean (<60 MB) and round-trips cleanly");
    }

    // 3B. ZIP64 (>4 GiB) .cbz unit test using real build_cbz_to_file + mz_zip_reader_init_cfile
    {
        const fs::path tmp_dir = fs::temp_directory_path() / L"vellum_audit_zip64";
        std::error_code ec;
        fs::create_directories(tmp_dir, ec);

        // Create one valid ~4.1 MiB high-entropy PNG file (1400x1000 noisy RGB -> ~4.2 MiB PNG)
        const int PW = 1400, PH = 1000;
        std::vector<uint8_t> noise_rgb(static_cast<size_t>(PW) * static_cast<size_t>(PH) * 3ULL);
        uint32_t rng = 0xDEADBEEFU;
        for (auto& b : noise_rgb) {
            rng = rng * 1664525U + 1013904223U;
            b = static_cast<uint8_t>((rng >> 24) & 0xFFU);
        }
        const fs::path sample_png = tmp_dir / L"page_4mb.png";
        const int ok_png = stbi_write_png(sample_png.u8string().c_str(), PW, PH, 3, noise_rgb.data(), PW * 3);
        const uint64_t single_sz = ok_png ? static_cast<uint64_t>(fs::file_size(sample_png, ec)) : 0;

        // Compute how many copies exceed 4 GiB (4,294,967,296 bytes) by ~100 MiB
        constexpr uint64_t kFourGiB = 4ULL * 1024ULL * 1024ULL * 1024ULL;
        const size_t copies = (single_sz > 0)
            ? static_cast<size_t>((kFourGiB + 100ULL * 1024ULL * 1024ULL) / single_sz) + 1
            : 0;

        const fs::path cbz_path = tmp_dir / L"test_zip64_over_4gb.cbz";
        std::vector<mangaflow::DiskExportPageEntry> entries;
        entries.reserve(copies);
        for (size_t i = 0; i < copies; ++i) {
            entries.push_back({sample_png, "p_" + std::to_string(i + 1) + ".png"});
        }

        const auto t_z0 = std::chrono::steady_clock::now();
        mangaflow::build_cbz_to_file(entries, cbz_path, mangaflow::ExportImageFormat::Original, 95);
        const auto t_z1 = std::chrono::steady_clock::now();
        const double z_sec = std::chrono::duration<double>(t_z1 - t_z0).count();
        const uint64_t cbz_sz = static_cast<uint64_t>(fs::file_size(cbz_path, ec));

        // Open and verify with miniz ZIP64 reader
        bool reader_ok = false;
        mz_uint num_files = 0;
        bool first_match = false;
        bool last_match = false;
        const auto orig_bytes = mangaflow::util::read_file_bytes(sample_png);

        FILE* fp = nullptr;
        if (_wfopen_s(&fp, cbz_path.c_str(), L"rb") == 0 && fp) {
            mz_zip_archive zip{};
            if (mz_zip_reader_init_cfile(&zip, fp, cbz_sz, 0)) {
                reader_ok = true;
                num_files = mz_zip_reader_get_num_files(&zip);
                if (num_files == copies && orig_bytes.has_value()) {
                    size_t sz0 = 0, sz_last = 0;
                    void* p0 = mz_zip_reader_extract_to_heap(&zip, 0, &sz0, 0);
                    void* p_last = mz_zip_reader_extract_to_heap(&zip, num_files - 1, &sz_last, 0);
                    if (p0 && sz0 == orig_bytes->size() &&
                        std::memcmp(p0, orig_bytes->data(), sz0) == 0) {
                        first_match = true;
                    }
                    if (p_last && sz_last == orig_bytes->size() &&
                        std::memcmp(p_last, orig_bytes->data(), sz_last) == 0) {
                        last_match = true;
                    }
                    if (p0) mz_free(p0);
                    if (p_last) mz_free(p_last);
                }
                mz_zip_reader_end(&zip);
            }
            std::fclose(fp);
        }

        std::cout << "  [ZIP64 >4 GiB CBZ] single_png=" << std::fixed << std::setprecision(2)
                  << (static_cast<double>(single_sz) / (1024.0 * 1024.0)) << " MiB"
                  << " x " << copies << " entries = "
                  << (static_cast<double>(cbz_sz) / (1024.0 * 1024.0 * 1024.0)) << " GiB"
                  << " | write_time=" << z_sec << " s"
                  << " | read_entries=" << num_files
                  << " | first_match=" << (first_match ? "true" : "false")
                  << " | last_match=" << (last_match ? "true" : "false") << "\n";

        AUDIT_CHECK(cbz_sz > kFourGiB && reader_ok && num_files == copies && first_match && last_match,
                    "ZIP64 (>4 GiB) .cbz write + read round-trip verified (including >4 GiB offset entry)");

        fs::remove_all(tmp_dir, ec);
    }

    // 3C. Native PDF 1.4 multi-page export unit test
    {
        const fs::path tmp_pdf = fs::temp_directory_path() / L"vellum_audit_test.pdf";
        std::vector<mangaflow::DiskExportPageEntry> pages = {
            {repo_root / L"samples" / L"sample_ja.jpg", "sample_ja.jpg"},
            {repo_root / L"samples" / L"sample_ko.jpg", "sample_ko.jpg"},
            {repo_root / L"samples" / L"sample_zh.jpg", "sample_zh.jpg"},
        };
        mangaflow::build_pdf_to_file(pages, tmp_pdf);
        const auto pdf_bytes = mangaflow::util::read_file_bytes(tmp_pdf);
        bool pdf_ok = false;
        if (pdf_bytes.has_value() && pdf_bytes->size() > 1024) {
            const std::string s(reinterpret_cast<const char*>(pdf_bytes->data()), pdf_bytes->size());
            pdf_ok = (s.rfind("%PDF-1.4\n", 0) == 0) &&
                     (s.find("/Count 3") != std::string::npos) &&
                     (s.find("/DCTDecode") != std::string::npos) &&
                     (s.find("xref\n") != std::string::npos) &&
                     (s.find("%%EOF\n") != std::string::npos);
            std::cout << "  [PDF 1.4 Export] 3 pages -> " << pdf_bytes->size() << " bytes\n";
        }
        AUDIT_CHECK(pdf_ok, "Multi-page PDF 1.4 export (lossless /DCTDecode + xref + %%EOF) verified");
        std::error_code ec;
        fs::remove(tmp_pdf, ec);
    }
}

// -----------------------------------------------------------------------------
// Audit 4: Security & DPAPI Settings Encryption Verification
// -----------------------------------------------------------------------------
void run_audit_4_security_dpapi() {
    std::cout << "\n============================================================\n";
    std::cout << "AUDIT 4: DPAPI Settings Encryption & Default Provider Check\n";
    std::cout << "============================================================\n";

    const json defs = mangaflow::SettingsManager::build_default_settings();
    AUDIT_CHECK(defs.value("active_provider", std::string("")) == "none" &&
                    defs.value("provider", std::string("")) == "none" &&
                    defs.value("api_key", std::string("X")).empty(),
                "Default settings have active_provider='none', provider='none', api_key=''");

    const fs::path tmp_settings = fs::temp_directory_path() / L"vellum_audit_settings.json";
    std::error_code ec;
    fs::remove(tmp_settings, ec);

    const std::string test_secret = "sk-audit-dpapi-test-9876543210abcdef";
    {
        mangaflow::SettingsManager mgr(tmp_settings);
        const json updated = mgr.update({
            {"active_provider", "openai"},
            {"api_key", test_secret},
        });
        AUDIT_CHECK(updated.value("api_key", std::string("")) == test_secret,
                    "In-memory SettingsManager::update returns decrypted plaintext api_key");
    }

    // Verify raw file on disk has dpapi: prefix and zero plaintext leakage
    const auto raw_bytes = mangaflow::util::read_file_bytes(tmp_settings);
    bool raw_encrypted = false;
    if (raw_bytes.has_value()) {
        const std::string raw_str(reinterpret_cast<const char*>(raw_bytes->data()), raw_bytes->size());
        const json on_disk = json::parse(raw_str);
        const std::string top_key = on_disk.value("api_key", std::string(""));
        const std::string prof_key = on_disk["providers"]["openai"].value("api_key", std::string(""));
        raw_encrypted = (raw_str.find(test_secret) == std::string::npos) &&
                        (top_key.rfind("dpapi:", 0) == 0) &&
                        (prof_key.rfind("dpapi:", 0) == 0);
        std::cout << "  [DPAPI On-Disk] api_key prefix='" << top_key.substr(0, 12)
                  << "...' len=" << top_key.size() << "\n";
    }
    AUDIT_CHECK(raw_encrypted, "settings.json on disk encrypts non-empty API keys with 'dpapi:' (zero plaintext)");

    // Reload with a fresh SettingsManager and verify lossless DPAPI decryption
    {
        mangaflow::SettingsManager mgr2(tmp_settings);
        const json reloaded = mgr2.load(/*force_reload=*/true);
        AUDIT_CHECK(reloaded.value("api_key", std::string("")) == test_secret &&
                        reloaded["providers"]["openai"].value("api_key", std::string("")) == test_secret,
                    "Fresh SettingsManager::load decrypts 'dpapi:' API keys losslessly");
    }

    fs::remove(tmp_settings, ec);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        fs::path repo_root = fs::current_path();
        if (argc >= 2) {
            repo_root = fs::u8path(argv[1]);
        }
        std::cout << "[test_v25_audit] Repo root: " << repo_root.u8string() << "\n";

        run_audit_1_cv_and_inpainting(repo_root);
        run_audit_2_hardware_and_cuda_guard(repo_root);
        run_audit_3_endurance_and_formats(repo_root);
        run_audit_4_security_dpapi();

        std::cout << "\n============================================================\n";
        std::cout << "AUDIT SUMMARY: " << g_pass << " PASSED, " << g_fail << " FAILED\n";
        std::cout << "============================================================\n";
        return (g_fail == 0) ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "[FATAL] Unhandled exception: " << ex.what() << "\n";
        return 1;
    }
}
