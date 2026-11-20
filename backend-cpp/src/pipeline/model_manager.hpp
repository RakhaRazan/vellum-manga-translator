#pragma once

// MangaFlow AI — Native C++17 Model Manager & ONNX Runtime Session Factory (Phase 10.1)
//
// Features:
// - Persistent model storage at `<portable_root>\models` (or `MANGA_MODELS_DIR`).
// - Full 4-model Koharu local Hugging Face cache auto-discovery
//   (`C:\FILE SOFTWARE\Koharu\store\hugging-face\models\<repo>\snapshots\<hash>\`):
//     1. mayocream--koharu-layout-rfdetr-seg-2xl-1152 (layout & text segmentation)
//     2. mayocream--lama-manga (manga screentone LaMa inpainter)
//     3. mayocream--manga-ocr (Japanese OCR)
//     4. PaddlePaddle--PaddleOCR-VL-1.6-GGUF (Vision-Language OCR GGUF for LM Studio)
// - Optional read-only fallback scan of workspace `models/` for pre-existing ONNX assets.
// - Automatic HTTPS download fallback (`GET /api/models/download` SSE progress stream via WinHTTP)
//   for public GitHub users who do not have Koharu installed.
// - Hardware-agnostic ONNX Runtime C++ session factory (`CUDA` -> `DirectML` -> `CPU` fallback).

#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mangaflow::pipeline {

enum class DetectorBackend {
    KoharuRfDetr,
    ComicTextDetector,
};

inline std::string detector_backend_name(DetectorBackend backend) {
    switch (backend) {
        case DetectorBackend::KoharuRfDetr:
            return "koharu_rfdetr";
        case DetectorBackend::ComicTextDetector:
            return "comic_text_detector";
    }
    return "comic_text_detector";
}

inline std::string detector_model_id(DetectorBackend backend) {
    switch (backend) {
        case DetectorBackend::KoharuRfDetr:
            return "koharu-layout-rfdetr-seg-2xl";
        case DetectorBackend::ComicTextDetector:
            return "comic-text-detector";
    }
    return "comic-text-detector";
}

struct ResolvedDetectorModel {
    DetectorBackend backend = DetectorBackend::ComicTextDetector;
    std::filesystem::path model_path;
    std::optional<std::filesystem::path> config_path;
    std::string source;
};

struct ResolvedLamaModel {
    std::filesystem::path model_path;
    std::string source;
    bool is_koharu_manga = false;

    std::string model_id() const {
        return is_koharu_manga ? "lama-manga" : "lama-fp32";
    }
};

struct ResolvedMangaOcrModel {
    std::filesystem::path encoder_onnx;
    std::filesystem::path decoder_onnx;
    std::filesystem::path vocab_txt;
    std::optional<std::filesystem::path> config_json;
    std::string source;
};

struct ResolvedPaddleOcrVlGguf {
    std::filesystem::path model_gguf;
    std::filesystem::path mmproj_gguf;
    std::string source;
};

struct RapidOcrPaths {
    std::filesystem::path det_model_path;
    std::filesystem::path rec_model_path;
    std::filesystem::path cls_model_path;
};

struct KoharuSnapshotStatus {
    std::string key;
    std::string folder_name;
    std::string role;
    bool cache_found = false;
    std::filesystem::path snapshot_dir;
    std::string detected_format = "missing";  // "onnx" | "safetensors" | "gguf" | "missing"
    std::vector<std::string> snapshot_files;
    std::uintmax_t snapshot_bytes = 0;
    bool ready_for_inference = false;
    bool needs_conversion = false;
    std::vector<std::string> target_files;
};

struct KoharuImportOptions {
    std::optional<std::filesystem::path> koharu_cache_dir;
    std::vector<std::string> models;  // Empty = all 4 Koharu models
    bool force = false;
    bool convert_safetensors = true;
    bool prefer_hardlink = true;
    std::optional<std::filesystem::path> search_root_dir;
};

struct OrtSessionHandle {
    std::unique_ptr<Ort::Session> session;
    std::string active_provider = "CPUExecutionProvider";
};

struct SystemHardwareProfile {
    uint64_t total_ram_mb = 0;
    uint64_t available_ram_mb = 0;
    uint32_t memory_load_percent = 0;
    uint64_t dedicated_vram_mb = 0;
    uint32_t logical_cpus = 1;
    bool low_memory_mode = false;  // True on <10 GB RAM or <5 GB VRAM systems
    std::string gpu_name;
};

class ModelManager {
public:
    using ProgressCallback = std::function<bool(const nlohmann::json& event)>;

    explicit ModelManager(
        std::optional<std::filesystem::path> models_dir = std::nullopt,
        std::optional<std::filesystem::path> fallback_workspace_models_dir = std::nullopt,
        std::optional<std::filesystem::path> koharu_cache_dir = std::nullopt);

    // Persistent storage & cache roots
    static std::filesystem::path default_models_dir();
    static std::filesystem::path default_koharu_cache_dir();
    static std::optional<std::filesystem::path> find_workspace_models_dir(
        const std::filesystem::path& exe_dir);

    // Process-wide ONNX Runtime environment & provider discovery (`CUDA` -> `DirectML` -> `CPU`)
    static Ort::Env& shared_ort_env();
    static std::string get_ort_version();
    static std::vector<std::string> get_available_providers();
    static std::vector<std::string> get_preferred_providers(bool force_cpu = false);
    static std::string get_primary_provider(bool force_cpu = false);
    static OrtSessionHandle create_ort_session(
        const std::filesystem::path& model_path,
        bool force_cpu = false);

    // Live hardware & memory pressure telemetry (adaptive RAM/VRAM/CPU budgeting)
    static SystemHardwareProfile query_hardware_profile();
    static bool is_under_memory_pressure();
    static void trim_process_heap();

    // Koharu 4-model Hugging Face cache auto-discovery (`<model_dir>/snapshots/<hash>/`)
    static std::optional<std::filesystem::path> resolve_koharu_snapshot(
        const std::filesystem::path& cache_root,
        const std::string& folder_name);
    std::vector<KoharuSnapshotStatus> inspect_koharu_cache() const;
    nlohmann::json get_koharu_status_json() const;

    // Dynamic pipeline model catalog (`GET /api/pipeline/models`) & Koharu importer (`POST /api/models/import-koharu`)
    nlohmann::json get_pipeline_models_json(
        const nlohmann::json& active_settings = nlohmann::json::object()) const;
    nlohmann::json import_koharu_models(const KoharuImportOptions& options = {}) const;

    // Model resolvers (prefer requested model; optional fallback when allow_fallback=true)
    std::optional<ResolvedDetectorModel> resolve_detector_model(
        const std::string& preferred_backend = "auto",
        bool allow_fallback = false) const;
    std::optional<ResolvedLamaModel> resolve_lama_model(
        const std::string& preferred_model = "auto",
        bool allow_fallback = true) const;
    std::optional<ResolvedMangaOcrModel> resolve_manga_ocr_models() const;
    std::optional<ResolvedPaddleOcrVlGguf> resolve_paddleocr_vl_gguf() const;
    std::optional<RapidOcrPaths> resolve_rapidocr_paths(const std::string& language = "zh") const;

    // Offline readiness & SPEC.md Section 5.5 status payload
    bool is_all_ready() const;
    nlohmann::json get_status_json() const;

    // Public GitHub fallback: stream missing Hugging Face weights over HTTPS (WinHTTP) into persistent dir
    void download_missing_models(const ProgressCallback& on_event) const;

    const std::filesystem::path& models_dir() const noexcept { return models_dir_; }
    const std::optional<std::filesystem::path>& fallback_models_dir() const noexcept {
        return fallback_models_dir_;
    }
    const std::filesystem::path& koharu_cache_dir() const noexcept { return koharu_cache_dir_; }

private:
    struct DownloadFileSpec {
        std::string model_key;
        std::string repo_id;
        std::string remote_filename;
        std::string local_rel_path;
        std::uintmax_t expected_bytes;
        std::uintmax_t min_valid_bytes;
    };

    std::optional<std::filesystem::path> find_valid_file(
        const std::string& rel_path,
        std::uintmax_t min_valid_bytes,
        std::string* source_out = nullptr) const;
    std::vector<DownloadFileSpec> get_missing_download_specs() const;
    bool download_single_file_winhttp(
        const DownloadFileSpec& spec,
        const std::filesystem::path& target_path,
        size_t file_index,
        size_t file_count,
        std::uintmax_t& cumulative_downloaded,
        std::uintmax_t& total_expected_bytes,
        const ProgressCallback& on_event,
        std::string& error_out) const;

    std::filesystem::path models_dir_;
    std::optional<std::filesystem::path> fallback_models_dir_;
    std::filesystem::path koharu_cache_dir_;
};

}  // namespace mangaflow::pipeline
