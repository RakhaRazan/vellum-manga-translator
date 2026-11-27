#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <mutex>
#include <optional>
#include <string>

namespace mangaflow {

class SettingsManager {
public:
    explicit SettingsManager(std::optional<std::filesystem::path> custom_path = std::nullopt);

    // Resolve default path: `MANGA_SETTINGS_PATH` env var or `<portable_root>\settings.json`.
    static std::filesystem::path default_settings_path();

    // Thread-safe load from disk (cached after first load unless force_reload=true).
    nlohmann::json load(bool force_reload = false);

    // Merge a partial or full JSON patch onto current settings, validate, and persist atomically.
    // Throws std::invalid_argument on validation errors, std::runtime_error on disk write errors.
    nlohmann::json update(const nlohmann::json& patch);

    // Build complete response payload for `GET /api/settings` & `POST /api/settings`.
    nlohmann::json format_api_response(const nlohmann::json& settings) const;

    // Convenience summaries for `/api/health`, `/api/providers`, `/api/languages`, and `/api/pipeline/models`.
    nlohmann::json get_active_provider_summary();
    static nlohmann::json list_provider_presets();
    static nlohmann::json list_source_languages();
    static nlohmann::json list_target_languages();
    static nlohmann::json list_reasoning_efforts();
    static nlohmann::json list_detection_models();
    static nlohmann::json list_ocr_models();
    static nlohmann::json list_inpainting_models();
    static std::string normalize_detection_model(const std::string& raw);
    static std::string normalize_detector_precision(const std::string& raw);
    static std::string normalize_ocr_model(const std::string& raw);
    static std::string normalize_inpainting_model(const std::string& raw);
    static nlohmann::json build_default_settings();

    const std::filesystem::path& path() const noexcept { return path_; }

private:
    nlohmann::json sanitize_and_merge(const nlohmann::json& raw, const nlohmann::json& base) const;
    void save_locked(const nlohmann::json& settings);

    std::filesystem::path path_;
    mutable std::mutex mutex_;
    std::optional<nlohmann::json> cache_;
};

}  // namespace mangaflow
