#pragma once

// Vellum - Native C++17 Tile-Based LaMa ONNX Inpainter (Phase 10.2)
//
// Uses local LaMa ONNX weights via `ModelManager::resolve_lama_model()`
// (prefers Koharu's manga screentone `lama_manga.onnx`, falls back to `lama_fp32.onnx`).
//
// Features:
// - Lazy `Ort::Session` creation (`CUDA` -> `DirectML` -> `CPU` fallback) + immediate `unload()`.
// - Tile-based `512x512` ROI inpainting (1:1 native pixel resolution for ROIs <= 512px,
//   resize-and-restore for oversized ROIs) processed sequentially one tile at a time.
// - 100% lossless background compositing (only masked pixels are patched onto the original RGB page).
// - Interior ink mask refinement preserving speech bubble borders while removing stray punctuation/halos.
// - Single-header `stb_image` / `stb_image_write` PNG encoding (zero OpenCV / Python dependency).

#include "model_manager.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mangaflow::pipeline {

struct InpaintOptions {
    std::optional<std::vector<std::array<int, 4>>> bboxes;
    std::string inpainting_model = "lama-manga";  // "lama-manga" | "aot-inpainting" | "flux-2-klein" | "rorem-mixed" | "lama-fp32" | "auto"
    std::string inpaint_api_url = "http://127.0.0.1:8188";
    int tile_size = 512;                          // Context ROI window size (256..1024; ONNX native is 512x512)
    int tile_padding = 32;
    int dilate_kernel_size = 9;
    bool refine_mask = true;
    bool auto_unload = true;
    bool force_cpu = false;
    bool allow_model_fallback = true;
};

struct InpaintResult {
    int width = 0;
    int height = 0;
    int tiles_processed = 0;
    int tile_size = 512;
    std::string execution_provider = "CPUExecutionProvider";
    std::string model_id = "lama-manga";
    bool is_koharu_manga = false;
    std::string model_path;
    std::vector<uint8_t> rgb;  // Row-major (height * width * 3), RGB uint8
};

class LamaInpainter {
public:
    static constexpr int kTileSize = 512;

    explicit LamaInpainter(const ModelManager& model_manager);

    // Decode raw image bytes (JPG/PNG/WebP/BMP) -> RGB uint8 buffer `(width * height * 3)`.
    static std::vector<uint8_t> decode_rgb_from_bytes(
        const uint8_t* image_bytes,
        size_t num_bytes,
        int& width_out,
        int& height_out);

    // Decode raw mask image bytes -> binary `(target_w * target_h)` uint8 (`0` or `255`).
    static std::vector<uint8_t> decode_mask_from_bytes(
        const uint8_t* mask_bytes,
        size_t num_bytes,
        int target_w,
        int target_h);

    // Synthesize a binary mask (`0` or `255`) filled strictly over `bboxes`.
    static std::vector<uint8_t> synthesize_mask_from_bboxes(
        const std::vector<std::array<int, 4>>& bboxes,
        int width,
        int height);

    // Restrict an existing binary mask strictly to padded `bboxes`.
    static void restrict_mask_to_bboxes(
        std::vector<uint8_t>& mask,
        const std::vector<std::array<int, 4>>& bboxes,
        int width,
        int height,
        int pad = 8);

    // Run tile-based LaMa inpainting on an RGB image + binary `(width * height)` mask.
    InpaintResult inpaint_rgb(
        const uint8_t* rgb_data,
        int width,
        int height,
        const std::vector<uint8_t>& binary_mask,
        const InpaintOptions& options = {});

    // Encode RGB `(width * height * 3)` buffer into lossless PNG bytes via `stb_image_write`.
    static std::vector<uint8_t> encode_rgb_png(
        const std::vector<uint8_t>& rgb,
        int width,
        int height);

    // Encode RGB buffer as `data:image/png;base64,...` (optional backward compatibility).
    static std::string rgb_to_data_uri(
        const std::vector<uint8_t>& rgb,
        int width,
        int height);

    // Explicitly unload any cached LaMa / AOT ONNX session from RAM/VRAM.
    void unload();
    bool is_loaded() const noexcept { return cached_session_.has_value() || cached_aot_session_.has_value(); }

private:
    OrtSessionHandle& ensure_session(const ResolvedLamaModel& resolved, bool force_cpu);
    OrtSessionHandle& ensure_aot_session(const ResolvedAotModel& resolved, bool force_cpu);

    // Execute LaMa or AOT ONNX on a single `512x512` RGB tile + binary mask -> inpainted `512x512` RGB.
    std::vector<uint8_t> run_tile_512(
        const uint8_t* tile_rgb,
        const uint8_t* tile_mask,
        const ResolvedLamaModel& resolved,
        bool force_cpu);

    std::vector<uint8_t> run_aot_tile_512(
        const uint8_t* tile_rgb,
        const uint8_t* tile_mask,
        const ResolvedAotModel& resolved,
        bool force_cpu);

    // Inpaint a single padded ROI `[x1, y1, x2, y2]` and composite only masked pixels onto `canvas`.
    bool inpaint_roi(
        std::vector<uint8_t>& canvas,
        int orig_w,
        int orig_h,
        const std::vector<uint8_t>& full_mask,
        const std::array<int, 4>& bbox,
        int padding,
        int tile_size,
        const ResolvedLamaModel& resolved,
        bool force_cpu);

    static std::vector<std::array<int, 4>> extract_bboxes_from_mask(
        const std::vector<uint8_t>& mask,
        int width,
        int height);

    static std::vector<uint8_t> refine_bubble_mask(
        const uint8_t* rgb_data,
        const std::vector<uint8_t>& base_mask,
        const std::vector<std::array<int, 4>>& bboxes,
        int width,
        int height,
        int dilate_kernel_size);

    const ModelManager& model_manager_;
    std::optional<OrtSessionHandle> cached_session_;
    std::filesystem::path cached_model_path_;
    bool cached_force_cpu_ = false;
    std::optional<OrtSessionHandle> cached_aot_session_;
    std::filesystem::path cached_aot_path_;
    bool cached_aot_force_cpu_ = false;
};

}  // namespace mangaflow::pipeline
