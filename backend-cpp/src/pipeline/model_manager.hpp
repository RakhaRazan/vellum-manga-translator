#pragma once

// Vellum - Native C++17 Model Manager & ONNX Runtime Session Factory
//
// Features:
// - Persistent model storage at `<portable_root>\models` (or `MANGA_MODELS_DIR`).
// - Optional read-only fallback scan of workspace `models/` for pre-existing ONNX assets.
// - Automatic HTTPS download fallback (`GET /api/models/download` SSE progress stream via WinHTTP).
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
    ComicTextDetector,
};

inline std::string detector_backend_name(DetectorBackend backend) {
    switch (backend) {
        case DetectorBackend::ComicTextDetector:
            return "comic_text_detector";
    }
    return "comic_text_detector";
}

inline std::string detector_model_id(DetectorBackend backend) {
    switch (backend) {
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
    bool is_manga_lama = false;

    std::string model_id() const {
        return is_manga_lama ? "lama-manga" : "lama-fp32";
    }
};

struct ResolvedMangaOcrModel {
    std::filesystem::path encoder_onnx;
    std::filesystem::path decoder_onnx;
    std::filesystem::path vocab_txt;
    std::optional<std::filesystem::path> config_json;
    std::string source;
};

struct ResolvedBaberuOcrModel {
    std::filesystem::path vision_onnx;
    std::filesystem::path decoder_prefill_onnx;
    std::filesystem::path decoder_step_onnx;
    std::filesystem::path vocab_json;
    std::string source;
};

struct ResolvedHayaiOcrModel {
    std::filesystem::path encoder_onnx;
    std::filesystem::path decoder_onnx;
    std::filesystem::path vocab_json;
    std::string source;
};

struct ResolvedAotModel {
    std::filesystem::path model_path;
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
        std::optional<std::filesystem::path> fallback_workspace_models_dir = std::nullopt);

    // Persistent storage & cache roots
    static std::filesystem::path default_models_dir();
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
        bool force_cpu = false,
        int cpu_intra_threads_override = 0);

    // Live hardware & memory pressure telemetry (adaptive RAM/VRAM/CPU budgeting)
    static SystemHardwareProfile query_hardware_profile();
    static bool is_under_memory_pressure();
    static void trim_process_heap();

    // Dynamic pipeline model catalog (`GET /api/pipeline/models`)
    nlohmann::json get_pipeline_models_json(
        const nlohmann::json& active_settings = nlohmann::json::object()) const;

    // Model resolvers (prefer requested model; optional fallback when allow_fallback=true)
    std::optional<ResolvedDetectorModel> resolve_detector_model(
        const std::string& preferred_backend = "auto",
        bool allow_fallback = false) const;
    std::optional<ResolvedLamaModel> resolve_lama_model(
        const std::string& preferred_model = "auto",
        bool allow_fallback = true) const;
    std::optional<ResolvedAotModel> resolve_aot_model() const;
    std::optional<ResolvedMangaOcrModel> resolve_manga_ocr_models() const;
    std::optional<ResolvedBaberuOcrModel> resolve_baberu_ocr_models() const;
    std::optional<ResolvedHayaiOcrModel> resolve_hayai_ocr_models() const;
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
};

}  // namespace mangaflow::pipeline
