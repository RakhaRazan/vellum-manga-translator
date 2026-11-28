#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>

#include "adaptive_tiler.hpp"
#include "cv_typography.hpp"
#include "detector_engine.hpp"
#include "export_artifacts.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::string toLowerAscii(std::string s) {
    for (char& ch : s) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return s;
}

bool isSupportedImageExt(const fs::path& p) {
    const std::string ext = toLowerAscii(p.extension().u8string());
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".webp" || ext == ".bmp";
}

double percentileMs(std::vector<double> values, double pct) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double idx = (pct / 100.0) * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = std::min(values.size() - 1, lo + 1);
    const double frac = idx - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

double queryPeakRamMb() noexcept {
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
    }
    return 0.0;
}

void printUsage(const char* argv0) {
    std::cout << "Usage: " << argv0
              << " --model <path.onnx> --input <file_or_dir>"
              << " [--dir <RTL|LTR>] [--out <export_dir>] [--runs <N>] [--tiling <0|1>]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path;
    std::string input_path;
    std::string dir_str = "RTL";
    std::string out_dir;
    int runs = 1;
    bool enable_tiling = true;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto nextVal = [&](std::string& dst) -> bool {
            if (i + 1 >= argc) return false;
            dst = argv[++i];
            return true;
        };
        if (arg == "--model") {
            if (!nextVal(model_path)) { printUsage(argv[0]); return 1; }
        } else if (arg == "--input") {
            if (!nextVal(input_path)) { printUsage(argv[0]); return 1; }
        } else if (arg == "--dir") {
            if (!nextVal(dir_str)) { printUsage(argv[0]); return 1; }
        } else if (arg == "--out") {
            if (!nextVal(out_dir)) { printUsage(argv[0]); return 1; }
        } else if (arg == "--runs") {
            std::string v;
            if (!nextVal(v)) { printUsage(argv[0]); return 1; }
            runs = std::max(1, std::stoi(v));
        } else if (arg == "--tiling") {
            std::string v;
            if (!nextVal(v)) { printUsage(argv[0]); return 1; }
            enable_tiling = (v != "0" && toLowerAscii(v) != "false");
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
    }

    if (model_path.empty() || input_path.empty()) {
        printUsage(argv[0]);
        return 1;
    }

    const fs::path in_fs = fs::u8path(input_path);
    std::vector<fs::path> image_files;
    if (fs::is_directory(in_fs)) {
        for (const auto& entry : fs::directory_iterator(in_fs)) {
            if (entry.is_regular_file() && isSupportedImageExt(entry.path())) {
                image_files.push_back(entry.path());
            }
        }
        std::sort(image_files.begin(), image_files.end());
    } else if (fs::is_regular_file(in_fs)) {
        image_files.push_back(in_fs);
    } else {
        std::cerr << "Error: input path not found: " << input_path << "\n";
        return 1;
    }

    if (image_files.empty()) {
        std::cerr << "Error: no supported images found at: " << input_path << "\n";
        return 1;
    }

    const std::string model_lower = toLowerAscii(fs::u8path(model_path).filename().u8string());
    vellum::detector::BackendType backend = vellum::detector::BackendType::Yolo11Seg;
    if (model_lower.find("comictextdetector") != std::string::npos) {
        backend = vellum::detector::BackendType::LegacyCtd;
    }

    vellum::detector::DetectorConfig cfg;
    cfg.model_path = model_path;
    cfg.backend = backend;
    cfg.reading_direction = (toLowerAscii(dir_str) == "ltr")
        ? vellum::detector::ReadingDirection::LTR
        : vellum::detector::ReadingDirection::RTL;
    cfg.enable_adaptive_tiling = enable_tiling;
    cfg.keep_session_alive = true;
    cfg.export_dir = out_dir;

    auto engine = vellum::detector::DetectorFactory::create(backend);
    if (!engine->load(cfg)) {
        std::cerr << "Error: failed to load detector model: " << model_path << "\n";
        return 1;
    }

    std::cout << "[vellum_det_bench] Model   : " << engine->activeModelName() << "\n";
    std::cout << "[vellum_det_bench] Provider: " << engine->executionProvider() << "\n";
    std::cout << "[vellum_det_bench] Dir     : " << (cfg.reading_direction == vellum::detector::ReadingDirection::RTL ? "RTL" : "LTR") << "\n";
    std::cout << "[vellum_det_bench] Tiling  : " << (enable_tiling ? "ON" : "OFF") << "\n";
    std::cout << "[vellum_det_bench] Runs    : " << runs << "\n";

    std::vector<double> all_infer_ms;
    std::vector<double> all_post_ms;
    std::array<int, vellum::detector::kNumClasses> total_class_counts{};

    for (const auto& img_path : image_files) {
        int w = 0, h = 0, c = 0;
        std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> rgb(
            stbi_load(img_path.u8string().c_str(), &w, &h, &c, 3),
            &stbi_image_free);
        if (!rgb || w <= 0 || h <= 0) {
            std::cerr << "Warning: failed to load image: " << img_path.u8string() << "\n";
            continue;
        }

        vellum::detector::PageDetectionResult last_res;
        std::vector<double> img_infer_ms;
        std::vector<double> img_post_ms;
        img_infer_ms.reserve(static_cast<size_t>(runs));
        img_post_ms.reserve(static_cast<size_t>(runs));

        for (int r = 0; r < runs; ++r) {
            last_res = vellum::detector::AdaptiveTiler::detectPage(
                *engine, rgb.get(), w, h, img_path.filename().u8string(), cfg);
            img_infer_ms.push_back(last_res.runtime_meta.inference_ms);
            img_post_ms.push_back(last_res.runtime_meta.postproc_ms);
            all_infer_ms.push_back(last_res.runtime_meta.inference_ms);
            all_post_ms.push_back(last_res.runtime_meta.postproc_ms);
        }

        std::array<int, vellum::detector::kNumClasses> img_counts{};
        img_counts[static_cast<size_t>(vellum::detector::ClassId::Panel)] =
            static_cast<int>(last_res.panels.size());
        for (const auto& det : last_res.detections) {
            const int cid = static_cast<int>(det.class_id);
            if (cid >= 0 && cid < vellum::detector::kNumClasses) {
                ++img_counts[static_cast<size_t>(cid)];
            }
        }
        for (size_t k = 0; k < img_counts.size(); ++k) {
            total_class_counts[k] += img_counts[k];
        }

        if (!out_dir.empty()) {
            if (!vellum::detector::exportPageArtifacts(last_res, rgb.get(), w, h, cfg)) {
                std::cerr << "Warning: failed to export artifacts to " << out_dir << "\n";
            }
        }

        std::cout << "\n--- " << img_path.filename().u8string()
                  << " (" << w << "x" << h << ", tiles=" << last_res.runtime_meta.tiles_processed << ") ---\n";
        std::cout << "  Inference p50 / p95 : "
                  << percentileMs(img_infer_ms, 50.0) << " ms / "
                  << percentileMs(img_infer_ms, 95.0) << " ms\n";
        std::cout << "  Postproc  p50       : " << percentileMs(img_post_ms, 50.0) << " ms\n";
        std::cout << "  Detections          : panels=" << last_res.panels.size()
                  << ", bubbles/regions=" << last_res.detections.size() << "\n";
        for (const auto& det : last_res.detections) {
            std::cout << "    det[" << det.id << "] order=" << det.reading_order
                      << " panel=" << det.panel_id
                      << " (" << det.class_name << "): "
                      << "room_polygon=" << det.room_polygon.size() << " pts";
            if (det.angle_deg.has_value()) {
                std::cout << ", {angle_deg: " << *det.angle_deg
                          << ", estimated_font_size: " << det.estimated_font_size.value_or(0.0f);
                if (det.text_color.has_value()) {
                    std::cout << ", text_color: [" << static_cast<int>((*det.text_color)[0])
                              << "," << static_cast<int>((*det.text_color)[1])
                              << "," << static_cast<int>((*det.text_color)[2]) << "]";
                }
                std::cout << ", outline: ";
                if (det.outline.has_value()) {
                    std::cout << "{color: [" << static_cast<int>(det.outline->color[0])
                              << "," << static_cast<int>(det.outline->color[1])
                              << "," << static_cast<int>(det.outline->color[2])
                              << "], width: " << det.outline->width << "}";
                } else {
                    std::cout << "null";
                }
                std::cout << "}";
            }
            std::cout << "\n";
        }
    }

    std::cout << "\n=== Summary ===\n";
    std::cout << "  Provider            : " << engine->executionProvider() << "\n";
    std::cout << "  Inference p50 / p95 : "
              << percentileMs(all_infer_ms, 50.0) << " ms / "
              << percentileMs(all_infer_ms, 95.0) << " ms\n";
    std::cout << "  Postproc p50        : " << percentileMs(all_post_ms, 50.0) << " ms\n";
    std::cout << "  Peak RAM            : " << queryPeakRamMb() << " MB\n";
    std::cout << "  Counts per class    :\n";
    for (int k = 0; k < vellum::detector::kNumClasses; ++k) {
        std::cout << "    [" << k << "] "
                  << vellum::detector::class_id_to_string(static_cast<vellum::detector::ClassId>(k))
                  << " = " << total_class_counts[static_cast<size_t>(k)] << "\n";
    }

    engine->unload();
    return 0;
}
