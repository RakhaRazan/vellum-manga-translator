#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mangaflow {

class ArchiveImportError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

class ComicExportError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

struct PageDimensions {
    int width = 0;
    int height = 0;
};

struct ExtractedArchivePage {
    int index = 0;
    std::string filename;
    std::string archive_path;
    std::string image_id;
    std::string extension;
    std::filesystem::path saved_path;
    PageDimensions dimensions;
    size_t size_bytes = 0;
};

struct ExportPageInput {
    std::vector<uint8_t> image_bytes;
    std::optional<std::string> filename;
};

struct DiskExportPageEntry {
    std::filesystem::path file_path;
    std::optional<std::string> filename;
};

enum class ExportImageFormat {
    Original,
    Png,
    Jpg,
};

// Local image cache utilities (`/api/cache/images/{image_id}.{ext}`) & pinned project cache
std::filesystem::path get_image_cache_dir();
std::filesystem::path get_thumb_cache_dir();
std::filesystem::path get_projects_cache_dir();
std::filesystem::path get_export_sessions_dir();
void prune_image_cache(const std::filesystem::path& cache_dir, size_t max_files = 50000);
std::filesystem::path resolve_cached_image_path(const std::string& id_or_url);
std::vector<uint8_t> get_or_create_cached_thumbnail(
    const std::string& id_or_url,
    int target_width = 180,
    int jpeg_quality = 75,
    std::string* out_content_type = nullptr);
std::filesystem::path resolve_pinned_project_image_path(
    const std::string& project_id,
    const std::string& filename);
std::string media_type_for_cache_ext(const std::string& ext_lower);

// Natural sorting & filename sanitization
bool natural_path_less(const std::string& a, const std::string& b);
std::string sanitize_export_filename(
    const std::optional<std::string>& raw_name,
    const std::string& extension,
    const std::string& default_stem = "translated_comic");
ExportImageFormat parse_export_image_format(const std::optional<std::string>& raw);

// Decode a JSON export image reference (`data:image/...;base64,...`, raw base64, or `/api/cache/images/...`)
std::vector<uint8_t> decode_export_image_reference(const std::string& ref, int page_index);

constexpr size_t kMaxArchivePages = 25000;
constexpr uint64_t kMaxArchiveUncompressedBytes = 32ULL * 1024ULL * 1024ULL * 1024ULL;  // 32 GiB

// `POST /api/import/archive` - extract valid comic page images from `.cbz`/`.zip` in natural page order
std::vector<ExtractedArchivePage> extract_archive_pages(
    const std::vector<uint8_t>& archive_bytes,
    const std::filesystem::path& output_dir,
    size_t max_pages = kMaxArchivePages);

// `POST /api/import/archive-path` - O(1) RAM 64-bit disk-backed `.cbz`/`.zip` extractor
std::vector<ExtractedArchivePage> extract_archive_pages_from_file(
    const std::filesystem::path& archive_path,
    const std::filesystem::path& output_dir,
    size_t max_pages = kMaxArchivePages);

// `POST /api/import/folder-path` - O(1) RAM local folder image importer in natural page order
std::vector<ExtractedArchivePage> import_folder_pages_from_dir(
    const std::filesystem::path& folder_path,
    const std::filesystem::path& output_dir,
    size_t max_pages = kMaxArchivePages);

// `POST /api/export/cbz` - package ordered pages into a downloadable `.cbz` archive
std::vector<uint8_t> build_cbz_bytes(
    const std::vector<ExportPageInput>& pages,
    ExportImageFormat image_format = ExportImageFormat::Original,
    int jpeg_quality = 95,
    bool smart_ssim_export = true);

// Disk-backed O(1)-RAM `.cbz` writer for large multi-thousand-page batches
void build_cbz_to_file(
    const std::vector<DiskExportPageEntry>& pages,
    const std::filesystem::path& output_path,
    ExportImageFormat image_format = ExportImageFormat::Original,
    int jpeg_quality = 95,
    bool smart_ssim_export = true);

// `POST /api/export/pdf` - native C++ PDF 1.4 multi-page writer (JPEG DCTDecode + lossless FlateDecode)
std::vector<uint8_t> build_pdf_bytes(
    const std::vector<ExportPageInput>& pages,
    bool smart_ssim_export = true);

// Disk-backed O(1)-RAM PDF 1.4 writer for large multi-thousand-page batches
void build_pdf_to_file(
    const std::vector<DiskExportPageEntry>& pages,
    const std::filesystem::path& output_path,
    bool smart_ssim_export = true);

// Package arbitrary named binary buffers (e.g. multi-page `.psd` files) into a `.zip` archive
std::vector<uint8_t> build_zip_from_named_buffers(
    const std::vector<std::pair<std::string, std::vector<uint8_t>>>& entries);

}  // namespace mangaflow
