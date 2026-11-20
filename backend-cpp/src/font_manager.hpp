#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mangaflow {

class FontValidationError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

class FontNotFoundError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct ResolvedFontFile {
    std::filesystem::path path;
    std::string media_type;
    std::string filename;
};

class FontManager {
public:
    explicit FontManager(std::optional<std::filesystem::path> custom_fonts_dir = std::nullopt);

    // Resolve default local fonts dir: `MANGA_FONTS_DIR` or `%LOCALAPPDATA%\ComicTranslator\fonts`.
    static std::filesystem::path default_fonts_dir();
    static std::vector<std::filesystem::path> windows_system_font_dirs();

    // `GET /api/fonts/list`
    nlohmann::json list_fonts(
        const std::optional<std::string>& category = std::nullopt,
        bool include_catalog = true,
        bool force_refresh = false);

    // `GET /api/fonts/file/{font_name}`
    // Throws FontValidationError on invalid/unsafe identifier, FontNotFoundError if not found.
    ResolvedFontFile resolve_font_file(const std::string& font_name);

    // `POST /api/fonts/upload`
    // Throws FontValidationError on invalid file/extension/magic, std::runtime_error on I/O failure.
    nlohmann::json save_uploaded_font(
        const std::string& filename,
        const std::vector<uint8_t>& raw_bytes,
        const std::optional<std::string>& override_family = std::nullopt,
        bool overwrite = true);

    // `POST /api/fonts/download`
    nlohmann::json download_catalog_fonts(
        const std::vector<std::string>& requested_fonts,
        const std::optional<std::string>& weight = std::nullopt,
        bool overwrite = false);

    const std::filesystem::path& fonts_dir() const noexcept { return fonts_dir_; }

private:
    struct IndexedFont {
        nlohmann::json entry;
        std::filesystem::path path;
        std::string family_lower;
        std::string full_name_lower;
        std::string filename_lower;
        std::string stem_lower;
        std::string category;
        int weight = 400;
        bool italic = false;
    };

    struct SystemDirFingerprint {
        std::string path_lower;
        int64_t mtime_ticks = 0;
        size_t entry_count = 0;
        bool operator==(const SystemDirFingerprint& o) const {
            return path_lower == o.path_lower &&
                   mtime_ticks == o.mtime_ticks &&
                   entry_count == o.entry_count;
        }
    };

    void ensure_dirs() const;
    nlohmann::json load_downloaded_manifest() const;
    void save_downloaded_manifest(const nlohmann::json& manifest) const;
    std::optional<IndexedFont> build_indexed_font(
        const std::filesystem::path& path,
        const std::string& category,
        const std::optional<std::string>& override_family = std::nullopt) const;
    std::vector<IndexedFont> scan_local_fonts() const;
    std::vector<IndexedFont> scan_system_fonts(bool force_refresh);
    std::vector<IndexedFont> collect_all_installed(bool force_refresh);
    nlohmann::json group_by_family(const std::vector<IndexedFont>& entries) const;

    std::filesystem::path fonts_dir_;
    std::filesystem::path downloaded_dir_;
    std::filesystem::path custom_subdir_;
    std::filesystem::path manifest_path_;

    mutable std::mutex mutex_;
    std::vector<SystemDirFingerprint> system_cache_key_;
    std::vector<IndexedFont> system_cache_;
};

}  // namespace mangaflow
