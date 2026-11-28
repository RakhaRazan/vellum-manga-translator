#include "exporter.hpp"
#include "pipeline/math_inpaint_ops.hpp"
#include "utils.hpp"

#include <miniz.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100 4244 4456 4996)
#endif
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <unordered_set>

namespace mangaflow {

namespace fs = std::filesystem;

namespace {

constexpr size_t kMaxSingleImageBytes = 64ULL * 1024ULL * 1024ULL;                   // 64 MiB

inline mz_bool mz_zip_writer_init_cfile_v2(mz_zip_archive* pZip, FILE* pFile, mz_uint flags) {
    return mz_zip_writer_init_cfile(pZip, pFile, flags);
}

struct StbiDeleter {
    void operator()(unsigned char* ptr) const noexcept {
        if (ptr) stbi_image_free(ptr);
    }
};
using StbiBuffer = std::unique_ptr<unsigned char, StbiDeleter>;

bool is_safe_image_id(const std::string& id) {
    if (id.empty() || id.size() > 128) return false;
    for (unsigned char c : id) {
        if (!std::isalnum(c) && c != '_' && c != '-') {
            return false;
        }
    }
    return true;
}

bool is_cacheable_ext(const std::string& ext_lower) {
    return ext_lower == "png" || ext_lower == "jpg" ||
           ext_lower == "jpeg" || ext_lower == "webp";
}

bool is_supported_archive_image_ext(const std::string& ext_lower) {
    return ext_lower == ".png" || ext_lower == ".jpg" ||
           ext_lower == ".jpeg" || ext_lower == ".webp" || ext_lower == ".bmp";
}

bool is_safe_archive_member(const std::string& member_name) {
    std::string norm = member_name;
    std::replace(norm.begin(), norm.end(), '\\', '/');
    if (norm.empty() || norm.front() == '/' || norm.back() == '/') {
        return false;
    }

    std::vector<std::string> parts;
    std::istringstream iss(norm);
    std::string seg;
    while (std::getline(iss, seg, '/')) {
        if (!seg.empty()) {
            parts.push_back(seg);
        }
    }
    if (parts.empty()) return false;

    for (const auto& part : parts) {
        if (part == "." || part == ".." || part == "__MACOSX" || part.front() == '.') {
            return false;
        }
    }

    const fs::path leaf(util::utf8_to_wide(parts.back()));
    const std::string ext_lower = util::to_lower_ascii(util::path_to_utf8(leaf.extension()));
    return is_supported_archive_image_ext(ext_lower);
}

enum class DetectedImageFormat {
    Unknown,
    Jpeg,
    Png,
    Webp,
    Bmp,
};

DetectedImageFormat detect_image_magic(const std::vector<uint8_t>& data) {
    if (data.size() >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF) {
        return DetectedImageFormat::Jpeg;
    }
    if (data.size() >= 8 &&
        data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47 &&
        data[4] == 0x0D && data[5] == 0x0A && data[6] == 0x1A && data[7] == 0x0A) {
        return DetectedImageFormat::Png;
    }
    if (data.size() >= 12 &&
        std::memcmp(data.data(), "RIFF", 4) == 0 &&
        std::memcmp(data.data() + 8, "WEBP", 4) == 0) {
        return DetectedImageFormat::Webp;
    }
    if (data.size() >= 2 && data[0] == 'B' && data[1] == 'M') {
        return DetectedImageFormat::Bmp;
    }
    return DetectedImageFormat::Unknown;
}

uint16_t u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}

uint32_t u24_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16);
}

uint32_t u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

uint16_t u16_be(const uint8_t* p) {
    return static_cast<uint16_t>((static_cast<uint16_t>(p[0]) << 8) | p[1]);
}

// Parse VP8 / VP8L / VP8X WebP container header for exact pixel dimensions.
bool parse_webp_dimensions(const std::vector<uint8_t>& data, int& width, int& height) {
    if (data.size() < 30 ||
        std::memcmp(data.data(), "RIFF", 4) != 0 ||
        std::memcmp(data.data() + 8, "WEBP", 4) != 0) {
        return false;
    }

    size_t pos = 12;
    while (pos + 8 <= data.size()) {
        const uint8_t* chunk = data.data() + pos;
        const uint32_t chunk_size = u32_le(chunk + 4);
        const size_t payload = pos + 8;
        if (payload > data.size()) break;

        if (std::memcmp(chunk, "VP8X", 4) == 0 && payload + 10 <= data.size()) {
            width = static_cast<int>(u24_le(data.data() + payload + 4) + 1);
            height = static_cast<int>(u24_le(data.data() + payload + 7) + 1);
            return width > 0 && height > 0;
        }
        if (std::memcmp(chunk, "VP8 ", 4) == 0 && payload + 10 <= data.size()) {
            if (data[payload + 3] == 0x9D && data[payload + 4] == 0x01 && data[payload + 5] == 0x2A) {
                width = static_cast<int>(u16_le(data.data() + payload + 6) & 0x3FFF);
                height = static_cast<int>(u16_le(data.data() + payload + 8) & 0x3FFF);
                return width > 0 && height > 0;
            }
        }
        if (std::memcmp(chunk, "VP8L", 4) == 0 && payload + 5 <= data.size() && data[payload] == 0x2F) {
            const uint32_t bits = u32_le(data.data() + payload + 1);
            width = static_cast<int>((bits & 0x3FFFU) + 1);
            height = static_cast<int>(((bits >> 14) & 0x3FFFU) + 1);
            return width > 0 && height > 0;
        }

        const size_t padded_size = static_cast<size_t>(chunk_size) + (chunk_size & 1U);
        if (payload + padded_size < pos) break;
        pos = payload + padded_size;
    }
    return false;
}

struct JpegHeaderInfo {
    int width = 0;
    int height = 0;
    int components = 0;
    int precision = 0;
    bool is_adobe_cmyk = false;
};

// Parse JPEG SOF0/SOF1/SOF2 marker to allow lossless /DCTDecode passthrough into PDF 1.4.
bool parse_jpeg_header(const std::vector<uint8_t>& data, JpegHeaderInfo& info) {
    if (data.size() < 4 || data[0] != 0xFF || data[1] != 0xD8) {
        return false;
    }

    size_t pos = 2;
    bool found_sof = false;
    while (pos + 2 <= data.size()) {
        // Seek to marker 0xFF
        while (pos < data.size() && data[pos] != 0xFF) {
            ++pos;
        }
        while (pos < data.size() && data[pos] == 0xFF) {
            ++pos;
        }
        if (pos >= data.size()) break;

        const uint8_t marker = data[pos++];
        if (marker == 0xD9 || marker == 0xDA) {  // EOI or SOS (scan data follows)
            break;
        }
        if ((marker >= 0xD0 && marker <= 0xD7) || marker == 0x01 || marker == 0x00) {
            continue;
        }
        if (pos + 2 > data.size()) break;
        const uint16_t seg_len = u16_be(data.data() + pos);
        if (seg_len < 2 || pos + seg_len > data.size()) {
            break;
        }

        // Adobe APP14 marker (checks CMYK YCCK inversion flag)
        if (marker == 0xEE && seg_len >= 14 &&
            std::memcmp(data.data() + pos + 2, "Adobe", 5) == 0) {
            info.is_adobe_cmyk = true;
        }

        // SOF0 (Baseline), SOF1 (Extended Sequential), SOF2 (Progressive)
        if ((marker == 0xC0 || marker == 0xC1 || marker == 0xC2) && seg_len >= 9) {
            info.precision = data[pos + 2];
            info.height = static_cast<int>(u16_be(data.data() + pos + 3));
            info.width = static_cast<int>(u16_be(data.data() + pos + 5));
            info.components = data[pos + 7];
            found_sof = true;
        }
        pos += seg_len;
    }

    return found_sof && info.width > 0 && info.height > 0 &&
           info.precision == 8 &&
           (info.components == 1 || info.components == 3 || info.components == 4);
}

void stbi_write_vec_callback(void* context, void* data, int size) {
    if (!context || !data || size <= 0) return;
    auto* vec = static_cast<std::vector<uint8_t>*>(context);
    const auto* bytes = static_cast<const uint8_t*>(data);
    vec->insert(vec->end(), bytes, bytes + size);
}

// Flatten decoded 1/2/3/4-channel pixels onto opaque white (RGB or Grayscale).
std::vector<uint8_t> flatten_pixels_to_rgb(
    const uint8_t* pixels,
    int width,
    int height,
    int channels) {
    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<uint8_t> rgb(pixel_count * 3);

    if (channels == 3) {
        std::memcpy(rgb.data(), pixels, pixel_count * 3);
    } else if (channels == 4) {
        for (size_t i = 0; i < pixel_count; ++i) {
            const uint32_t r = pixels[i * 4 + 0];
            const uint32_t g = pixels[i * 4 + 1];
            const uint32_t b = pixels[i * 4 + 2];
            const uint32_t a = pixels[i * 4 + 3];
            const uint32_t inv_a = 255U - a;
            rgb[i * 3 + 0] = static_cast<uint8_t>((r * a + 255U * inv_a + 127U) / 255U);
            rgb[i * 3 + 1] = static_cast<uint8_t>((g * a + 255U * inv_a + 127U) / 255U);
            rgb[i * 3 + 2] = static_cast<uint8_t>((b * a + 255U * inv_a + 127U) / 255U);
        }
    } else if (channels == 2) {
        for (size_t i = 0; i < pixel_count; ++i) {
            const uint32_t g = pixels[i * 2 + 0];
            const uint32_t a = pixels[i * 2 + 1];
            const uint8_t v = static_cast<uint8_t>((g * a + 255U * (255U - a) + 127U) / 255U);
            rgb[i * 3 + 0] = v;
            rgb[i * 3 + 1] = v;
            rgb[i * 3 + 2] = v;
        }
    } else {
        for (size_t i = 0; i < pixel_count; ++i) {
            const uint8_t v = pixels[i];
            rgb[i * 3 + 0] = v;
            rgb[i * 3 + 1] = v;
            rgb[i * 3 + 2] = v;
        }
    }
    return rgb;
}

std::vector<uint8_t> encode_rgb_png(const uint8_t* rgb, int width, int height, bool smart_gray = false) {
    std::vector<uint8_t> out;
    out.reserve(static_cast<size_t>(width) * static_cast<size_t>(height));
    if (smart_gray &&
        pipeline::math_inpaint::computeHaslerSusstrunkColorfulness(rgb, width, height, 4) < 8.0f) {
        std::vector<uint8_t> gray;
        pipeline::math_inpaint::rgbToGray8(rgb, width, height, gray);
        const int ok = stbi_write_png_to_func(
            stbi_write_vec_callback, &out, width, height, 1, gray.data(), width);
        if (ok && !out.empty()) {
            return out;
        }
        out.clear();
    }
    const int ok = stbi_write_png_to_func(
        stbi_write_vec_callback, &out, width, height, 3, rgb, width * 3);
    if (!ok || out.empty()) {
        throw ComicExportError("Failed to encode image as PNG.");
    }
    return out;
}

std::vector<uint8_t> encode_rgb_jpg(const uint8_t* rgb, int width, int height, int quality, bool smart_ssim = false) {
    const int clamped_q = std::max(50, std::min(100, quality));
    const bool use_gray = smart_ssim &&
        (pipeline::math_inpaint::computeHaslerSusstrunkColorfulness(rgb, width, height, 4) < 8.0f);

    std::vector<uint8_t> gray;
    if (use_gray || smart_ssim) {
        pipeline::math_inpaint::rgbToGray8(rgb, width, height, gray);
    }

    auto encodeAt = [&](int q) -> std::vector<uint8_t> {
        std::vector<uint8_t> buf;
        buf.reserve(static_cast<size_t>(width) * static_cast<size_t>(height) / 4);
        const int ok = use_gray
            ? stbi_write_jpg_to_func(stbi_write_vec_callback, &buf, width, height, 1, gray.data(), q)
            : stbi_write_jpg_to_func(stbi_write_vec_callback, &buf, width, height, 3, rgb, q);
        if (!ok || buf.empty()) {
            throw ComicExportError("Failed to encode image as JPEG.");
        }
        return buf;
    };

    if (!smart_ssim || width < 16 || height < 16) {
        return encodeAt(clamped_q);
    }

    // #21: 4-step binary search over q in [72, min(95, clamped_q)] targeting luminance SSIM >= 0.985
    int lo = 72;
    int hi = std::max(lo, std::min(95, clamped_q));
    std::vector<uint8_t> best_bytes;

    for (int step = 0; step < 4 && lo <= hi; ++step) {
        const int mid = (lo + hi) / 2;
        std::vector<uint8_t> cand = encodeAt(mid);
        int dw = 0, dh = 0, dc = 0;
        StbiBuffer dec(stbi_load_from_memory(cand.data(), static_cast<int>(cand.size()), &dw, &dh, &dc, 1));
        if (dec && dw == width && dh == height) {
            const float ssim = pipeline::math_inpaint::computeFastLuminanceSSIM(
                gray.data(), dec.get(), width, height);
            if (ssim >= 0.985f) {
                best_bytes = std::move(cand);
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        } else {
            best_bytes = std::move(cand);
            break;
        }
    }

    if (!best_bytes.empty()) {
        return best_bytes;
    }
    return encodeAt(std::max(lo, std::min(95, clamped_q)));
}

struct NaturalSegment {
    struct Token {
        bool is_number = false;
        uint64_t num = 0;
        std::string text;
    };
    std::vector<Token> tokens;
};

std::vector<NaturalSegment> build_natural_key(const std::string& path_str) {
    std::string norm = path_str;
    std::replace(norm.begin(), norm.end(), '\\', '/');
    while (!norm.empty() && norm.front() == '/') norm.erase(norm.begin());
    while (!norm.empty() && norm.back() == '/') norm.pop_back();

    std::vector<NaturalSegment> segments;
    std::istringstream iss(norm);
    std::string seg_str;
    while (std::getline(iss, seg_str, '/')) {
        NaturalSegment seg;
        const std::string lower = util::to_lower_ascii(seg_str);
        size_t i = 0;
        while (i < lower.size()) {
            if (std::isdigit(static_cast<unsigned char>(lower[i]))) {
                size_t start = i;
                uint64_t val = 0;
                while (i < lower.size() && std::isdigit(static_cast<unsigned char>(lower[i]))) {
                    const uint64_t digit = static_cast<uint64_t>(lower[i] - '0');
                    if (val <= (UINT64_MAX - digit) / 10ULL) {
                        val = val * 10ULL + digit;
                    }
                    ++i;
                }
                seg.tokens.push_back({true, val, lower.substr(start, i - start)});
            } else {
                size_t start = i;
                while (i < lower.size() && !std::isdigit(static_cast<unsigned char>(lower[i]))) {
                    ++i;
                }
                seg.tokens.push_back({false, 0, lower.substr(start, i - start)});
            }
        }
        segments.push_back(std::move(seg));
    }
    return segments;
}

std::string sanitize_page_stem(const std::optional<std::string>& filename) {
    if (!filename.has_value() || filename->empty()) {
        return {};
    }
    std::string norm = *filename;
    std::replace(norm.begin(), norm.end(), '\\', '/');
    const fs::path p(util::utf8_to_wide(norm));
    const std::string raw_stem = util::path_to_utf8(p.stem());

    std::string clean;
    clean.reserve(raw_stem.size());
    for (unsigned char c : raw_stem) {
        if (std::isalnum(c) || c == '.' || c == '_' || c == ' ' || c == '-') {
            clean.push_back(static_cast<char>(c));
        } else {
            clean.push_back('_');
        }
    }
    while (!clean.empty() && (clean.front() == ' ' || clean.front() == '.' || clean.front() == '_')) {
        clean.erase(clean.begin());
    }
    while (!clean.empty() && (clean.back() == ' ' || clean.back() == '.' || clean.back() == '_')) {
        clean.pop_back();
    }
    if (clean.size() > 64) {
        clean.resize(64);
    }
    return clean;
}

std::string format_pdf_number(double val) {
    char buf[64]{};
    std::snprintf(buf, sizeof(buf), "%.2f", val);
    std::string s(buf);
    if (s.find('.') != std::string::npos) {
        while (!s.empty() && s.back() == '0') s.pop_back();
        if (!s.empty() && s.back() == '.') s.pop_back();
    }
    return s;
}

}  // namespace

// -----------------------------------------------------------------------------
// Image Cache Management
// -----------------------------------------------------------------------------

fs::path get_image_cache_dir() {
    fs::path dir;
    if (auto env_cache = util::get_env_wide(L"MANGA_CACHE_DIR")) {
        dir = fs::path(*env_cache);
    } else {
        dir = util::get_comic_translator_app_dir() / L"cache" / L"images";
    }
    std::error_code ec;
    fs::create_directories(dir, ec);
    fs::path canon = fs::weakly_canonical(dir, ec);
    return ec ? dir : canon;
}

fs::path get_export_sessions_dir() {
    fs::path dir = get_image_cache_dir().parent_path() / L"exports";
    std::error_code ec;
    fs::create_directories(dir, ec);
    fs::path canon = fs::weakly_canonical(dir, ec);
    return ec ? dir : canon;
}

fs::path get_thumb_cache_dir() {
    fs::path dir = get_image_cache_dir().parent_path() / L"thumbs";
    std::error_code ec;
    fs::create_directories(dir, ec);
    fs::path canon = fs::weakly_canonical(dir, ec);
    return ec ? dir : canon;
}

fs::path get_projects_cache_dir() {
    fs::path dir = util::get_comic_translator_app_dir() / L"projects_cache";
    std::error_code ec;
    fs::create_directories(dir, ec);
    fs::path canon = fs::weakly_canonical(dir, ec);
    return ec ? dir : canon;
}

fs::path resolve_pinned_project_image_path(
    const std::string& project_id,
    const std::string& filename) {
    if (!is_safe_image_id(project_id) || filename.empty() || filename.size() > 160) {
        throw ComicExportError("Invalid pinned project image reference.");
    }
    const fs::path leaf(util::utf8_to_wide(filename));
    const std::string stem = util::path_to_utf8(leaf.stem());
    std::string ext = util::to_lower_ascii(util::path_to_utf8(leaf.extension()));
    if (!ext.empty() && ext.front() == '.') {
        ext.erase(ext.begin());
    }
    if (!is_safe_image_id(stem) || !is_cacheable_ext(ext) ||
        filename.find('/') != std::string::npos || filename.find('\\') != std::string::npos) {
        throw ComicExportError("Invalid pinned project image filename.");
    }

    const fs::path proj_dir = get_projects_cache_dir() / util::utf8_to_wide(project_id);
    std::error_code ec;
    const fs::path cand = fs::weakly_canonical(proj_dir / leaf, ec);
    if (ec || !fs::is_regular_file(cand, ec) ||
        !util::is_path_within_root(proj_dir, cand)) {
        throw std::runtime_error("Pinned project image not found.");
    }
    return cand;
}

void prune_image_cache(const fs::path& cache_dir, size_t max_files) {
    // Throttle directory scans during high-volume multi-thousand-page batches
    static std::atomic<uint64_t> s_prune_calls{0};
    if ((s_prune_calls.fetch_add(1, std::memory_order_relaxed) & 63ULL) != 0ULL) {
        return;
    }
    std::error_code ec;
    std::vector<std::pair<fs::file_time_type, fs::path>> files;
    for (const auto& entry : fs::directory_iterator(cache_dir, ec)) {
        if (ec) break;
        std::error_code f_ec;
        if (!entry.is_regular_file(f_ec) || f_ec) continue;
        std::string ext = util::to_lower_ascii(util::path_to_utf8(entry.path().extension()));
        if (!ext.empty() && ext.front() == '.') ext.erase(ext.begin());
        if (!is_cacheable_ext(ext)) continue;
        const auto mtime = entry.last_write_time(f_ec);
        if (!f_ec) {
            files.emplace_back(mtime, entry.path());
        }
    }
    if (files.size() <= max_files) {
        return;
    }
    std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    const size_t to_delete = files.size() - max_files;
    for (size_t i = 0; i < to_delete; ++i) {
        fs::remove(files[i].second, ec);
    }
}

std::string media_type_for_cache_ext(const std::string& ext_lower) {
    if (ext_lower == "png") return "image/png";
    if (ext_lower == "jpg" || ext_lower == "jpeg") return "image/jpeg";
    if (ext_lower == "webp") return "image/webp";
    return "application/octet-stream";
}

fs::path resolve_cached_image_path(const std::string& id_or_url) {
    std::string cleaned = util::trim_ascii(id_or_url);
    const std::string prefix = "/api/cache/images/";
    if (const auto pos = cleaned.find(prefix); pos != std::string::npos) {
        cleaned = cleaned.substr(pos + prefix.size());
    }
    if (const auto qpos = cleaned.find('?'); qpos != std::string::npos) {
        cleaned.resize(qpos);
    }

    std::optional<std::string> requested_ext;
    if (const auto dot = cleaned.rfind('.'); dot != std::string::npos) {
        std::string ext = util::to_lower_ascii(cleaned.substr(dot + 1));
        if (!is_cacheable_ext(ext)) {
            throw ComicExportError("Unsupported cached image extension: '" + ext + "'");
        }
        cleaned.resize(dot);
        requested_ext = std::move(ext);
    }

    if (!is_safe_image_id(cleaned)) {
        throw ComicExportError("Invalid cached image identifier: '" + id_or_url + "'");
    }

    const fs::path cache_root = get_image_cache_dir();
    std::vector<std::string> exts;
    if (requested_ext.has_value()) {
        exts.push_back(*requested_ext);
    } else {
        exts = {"png", "jpg", "jpeg", "webp"};
    }

    for (const auto& ext : exts) {
        std::error_code ec;
        const fs::path cand = fs::weakly_canonical(
            cache_root / util::utf8_to_wide(cleaned + "." + ext), ec);
        if (!ec && fs::is_regular_file(cand, ec) &&
            util::to_lower_ascii(util::path_to_utf8(cand.parent_path())) ==
                util::to_lower_ascii(util::path_to_utf8(cache_root))) {
            return cand;
        }
    }

    // Fallback: search pinned project directories under `./projects_cache/<project_id>/`
    // so pinned editable manga never return HTTP 404 after a temp cache purge.
    const fs::path proj_root = get_projects_cache_dir();
    std::error_code p_ec;
    if (fs::is_directory(proj_root, p_ec) && !p_ec) {
        for (const auto& entry : fs::directory_iterator(proj_root, p_ec)) {
            if (p_ec) break;
            std::error_code d_ec;
            if (!entry.is_directory(d_ec) || d_ec) continue;
            for (const auto& ext : exts) {
                const fs::path cand = entry.path() / util::utf8_to_wide(cleaned + "." + ext);
                if (fs::is_regular_file(cand, d_ec) && !d_ec) {
                    return cand;
                }
            }
        }
    }

    throw std::runtime_error("Cached image '" + cleaned + "' not found.");
}

std::vector<uint8_t> get_or_create_cached_thumbnail(
    const std::string& id_or_url,
    int target_width,
    int jpeg_quality,
    std::string* out_content_type) {
    if (out_content_type) {
        *out_content_type = "image/jpeg";
    }
    const int clamped_w = std::clamp(target_width, 64, 320);
    const int clamped_q = std::clamp(jpeg_quality, 30, 95);

    std::string cleaned = util::trim_ascii(id_or_url);
    if (const auto qpos = cleaned.find('?'); qpos != std::string::npos) {
        cleaned.resize(qpos);
    }
    if (const auto slash = cleaned.rfind('/'); slash != std::string::npos) {
        cleaned = cleaned.substr(slash + 1);
    }
    std::string stem = cleaned;
    if (const auto dot = stem.rfind('.'); dot != std::string::npos) {
        stem.resize(dot);
    }
    if (!is_safe_image_id(stem)) {
        throw ComicExportError("Invalid thumbnail image identifier: '" + id_or_url + "'");
    }

    const fs::path thumb_dir = get_thumb_cache_dir();
    const fs::path thumb_path = thumb_dir / util::utf8_to_wide(stem + "_" + std::to_string(clamped_w) + ".jpg");

    {
        std::error_code ec;
        if (fs::is_regular_file(thumb_path, ec) && !ec) {
            if (auto cached = util::read_file_bytes(thumb_path); cached.has_value() && !cached->empty()) {
                return std::move(*cached);
            }
        }
    }

    // Locate full-resolution source in `./cache/images/` or pinned `./projects_cache/<project_id>/`
    fs::path src_path;
    try {
        src_path = resolve_cached_image_path(cleaned);
    } catch (...) {
        const fs::path proj_root = get_projects_cache_dir();
        std::error_code ec;
        const std::vector<std::string> exts = {"png", "jpg", "jpeg", "webp"};
        for (const auto& entry : fs::directory_iterator(proj_root, ec)) {
            if (ec) break;
            std::error_code d_ec;
            if (!entry.is_directory(d_ec) || d_ec) continue;
            for (const auto& ext : exts) {
                const fs::path cand = entry.path() / util::utf8_to_wide(stem + "." + ext);
                if (fs::is_regular_file(cand, d_ec) && !d_ec) {
                    src_path = cand;
                    break;
                }
            }
            if (!src_path.empty()) break;
        }
    }
    if (src_path.empty()) {
        throw std::runtime_error("Cached image '" + stem + "' not found.");
    }

    auto src_bytes = util::read_file_bytes(src_path);
    if (!src_bytes.has_value() || src_bytes->empty()) {
        throw std::runtime_error("Failed to read cached image for thumbnail.");
    }

    int src_w = 0, src_h = 0, src_c = 0;
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> rgb(
        stbi_load_from_memory(
            src_bytes->data(),
            static_cast<int>(src_bytes->size()),
            &src_w, &src_h, &src_c, 3),
        &stbi_image_free);
    std::vector<uint8_t> wic_rgb;
    const uint8_t* decoded_rgb_ptr = rgb.get();
    if (!decoded_rgb_ptr || src_w <= 0 || src_h <= 0) {
        rgb.reset();
        if (util::decode_image_wic(
                src_bytes->data(), src_bytes->size(), 3, wic_rgb, src_w, src_h) &&
            !wic_rgb.empty() && src_w > 0 && src_h > 0) {
            decoded_rgb_ptr = wic_rgb.data();
        } else if (out_content_type &&
                   util::is_webp_bytes(src_bytes->data(), src_bytes->size())) {
            int webp_w = 0, webp_h = 0;
            if (parse_webp_dimensions(*src_bytes, webp_w, webp_h)) {
                *out_content_type = "image/webp";
                return std::move(*src_bytes);
            }
        }
    }
    src_bytes.reset();  // Free compressed file bytes immediately after decode

    if (!decoded_rgb_ptr || src_w <= 0 || src_h <= 0) {
        throw std::runtime_error("Failed to decode image for thumbnail.");
    }

    const int dst_w = std::min(src_w, clamped_w);
    const int dst_h = std::max(1, static_cast<int>(std::lround(
        static_cast<double>(src_h) * (static_cast<double>(dst_w) / static_cast<double>(src_w)))));

    std::vector<uint8_t> thumb_rgb(static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h) * 3ULL);
    if (dst_w == src_w && dst_h == src_h) {
        std::memcpy(thumb_rgb.data(), decoded_rgb_ptr, thumb_rgb.size());
    } else {
        const float sx_scale = static_cast<float>(src_w) / static_cast<float>(dst_w);
        const float sy_scale = static_cast<float>(src_h) / static_cast<float>(dst_h);
        const uint8_t* src_ptr = decoded_rgb_ptr;
        for (int dy = 0; dy < dst_h; ++dy) {
            const float sy0 = static_cast<float>(dy) * sy_scale;
            const float sy1 = static_cast<float>(dy + 1) * sy_scale;
            const int iy0 = std::clamp(static_cast<int>(std::floor(sy0)), 0, src_h - 1);
            const int iy1 = std::clamp(static_cast<int>(std::ceil(sy1)) - 1, iy0, src_h - 1);
            const size_t dst_row = static_cast<size_t>(dy) * static_cast<size_t>(dst_w) * 3ULL;
            for (int dx = 0; dx < dst_w; ++dx) {
                const float sx0 = static_cast<float>(dx) * sx_scale;
                const float sx1 = static_cast<float>(dx + 1) * sx_scale;
                const int ix0 = std::clamp(static_cast<int>(std::floor(sx0)), 0, src_w - 1);
                const int ix1 = std::clamp(static_cast<int>(std::ceil(sx1)) - 1, ix0, src_w - 1);
                uint32_t sr = 0, sg = 0, sb = 0, cnt = 0;
                for (int iy = iy0; iy <= iy1; ++iy) {
                    const size_t srow = static_cast<size_t>(iy) * static_cast<size_t>(src_w) * 3ULL;
                    for (int ix = ix0; ix <= ix1; ++ix) {
                        const size_t sidx = srow + static_cast<size_t>(ix) * 3ULL;
                        sr += src_ptr[sidx + 0];
                        sg += src_ptr[sidx + 1];
                        sb += src_ptr[sidx + 2];
                        ++cnt;
                    }
                }
                const size_t didx = dst_row + static_cast<size_t>(dx) * 3ULL;
                thumb_rgb[didx + 0] = static_cast<uint8_t>(cnt > 0 ? (sr / cnt) : 0);
                thumb_rgb[didx + 1] = static_cast<uint8_t>(cnt > 0 ? (sg / cnt) : 0);
                thumb_rgb[didx + 2] = static_cast<uint8_t>(cnt > 0 ? (sb / cnt) : 0);
            }
        }
    }
    rgb.reset();  // Free full-res RGB buffer before JPEG encoding

    std::vector<uint8_t> jpg_out;
    jpg_out.reserve(16384);
    auto write_cb = [](void* ctx, void* data, int size) {
        if (ctx && data && size > 0) {
            auto* vec = static_cast<std::vector<uint8_t>*>(ctx);
            const auto* bytes = static_cast<const uint8_t*>(data);
            vec->insert(vec->end(), bytes, bytes + size);
        }
    };
    const int ok = stbi_write_jpg_to_func(
        write_cb, &jpg_out, dst_w, dst_h, 3, thumb_rgb.data(), clamped_q);
    if (!ok || jpg_out.empty()) {
        throw std::runtime_error("Failed to encode JPEG thumbnail.");
    }

    // Persist to `./cache/thumbs/{id}_{w}.jpg` and LRU-prune
    {
        std::ofstream ofs(thumb_path, std::ios::binary | std::ios::trunc);
        if (ofs) {
            ofs.write(reinterpret_cast<const char*>(jpg_out.data()),
                      static_cast<std::streamsize>(jpg_out.size()));
        }
        prune_image_cache(thumb_dir, 4000);
    }

    return jpg_out;
}

// -----------------------------------------------------------------------------
// Natural Sort & Filename Helpers
// -----------------------------------------------------------------------------

bool natural_path_less(const std::string& a, const std::string& b) {
    const auto ka = build_natural_key(a);
    const auto kb = build_natural_key(b);
    const size_t seg_count = std::min(ka.size(), kb.size());
    for (size_t s = 0; s < seg_count; ++s) {
        const auto& ta = ka[s].tokens;
        const auto& tb = kb[s].tokens;
        const size_t tok_count = std::min(ta.size(), tb.size());
        for (size_t t = 0; t < tok_count; ++t) {
            if (ta[t].is_number != tb[t].is_number) {
                return ta[t].is_number && !tb[t].is_number;
            }
            if (ta[t].is_number) {
                if (ta[t].num != tb[t].num) return ta[t].num < tb[t].num;
                if (ta[t].text.size() != tb[t].text.size()) {
                    return ta[t].text.size() < tb[t].text.size();
                }
            } else {
                if (ta[t].text != tb[t].text) return ta[t].text < tb[t].text;
            }
        }
        if (ta.size() != tb.size()) return ta.size() < tb.size();
    }
    if (ka.size() != kb.size()) return ka.size() < kb.size();
    return a < b;
}

std::string sanitize_export_filename(
    const std::optional<std::string>& raw_name,
    const std::string& extension,
    const std::string& default_stem) {
    std::string raw = raw_name.has_value() ? util::trim_ascii(*raw_name) : "";
    if (!raw.empty()) {
        std::replace(raw.begin(), raw.end(), '\\', '/');
        if (const auto slash = raw.rfind('/'); slash != std::string::npos) {
            raw = raw.substr(slash + 1);
        }
    }

    const std::string lower = util::to_lower_ascii(raw);
    for (const char* suffix : {".cbz", ".pdf", ".zip", ".psd"}) {
        const size_t slen = std::strlen(suffix);
        if (lower.size() >= slen && lower.compare(lower.size() - slen, slen, suffix) == 0) {
            raw.resize(raw.size() - slen);
            break;
        }
    }

    std::string cleaned;
    cleaned.reserve(raw.size());
    for (unsigned char c : raw) {
        const bool is_win32_illegal =
            (c < 0x20u || c == 0x7Fu ||
             c == '<' || c == '>' || c == ':' || c == '"' ||
             c == '/' || c == '\\' || c == '|' || c == '?' || c == '*');
        if (is_win32_illegal) {
            cleaned.push_back('_');
        } else {
            cleaned.push_back(static_cast<char>(c));
        }
    }
    while (!cleaned.empty() && (cleaned.front() == ' ' || cleaned.front() == '.' || cleaned.front() == '_')) {
        cleaned.erase(cleaned.begin());
    }
    while (!cleaned.empty() && (cleaned.back() == ' ' || cleaned.back() == '.' || cleaned.back() == '_')) {
        cleaned.pop_back();
    }
    // Guard against Windows reserved device names (CON, PRN, AUX, NUL, COM1..9, LPT1..9)
    const std::string stem_upper = [&]() {
        std::string u;
        u.reserve(cleaned.size());
        for (unsigned char ch : cleaned) {
            if (ch == '.') break;
            u.push_back(static_cast<char>(std::toupper(ch)));
        }
        return u;
    }();
    if (stem_upper == "CON" || stem_upper == "PRN" || stem_upper == "AUX" || stem_upper == "NUL" ||
        (stem_upper.size() == 4 &&
         (stem_upper.rfind("COM", 0) == 0 || stem_upper.rfind("LPT", 0) == 0) &&
         stem_upper[3] >= '1' && stem_upper[3] <= '9')) {
        cleaned = "_" + cleaned;
    }
    if (cleaned.empty()) {
        cleaned = default_stem;
    }
    if (cleaned.size() > 120) {
        size_t cut = 120;
        // Never split a multi-byte UTF-8 continuation byte (10xxxxxx)
        while (cut > 0 && (static_cast<unsigned char>(cleaned[cut]) & 0xC0u) == 0x80u) {
            --cut;
        }
        cleaned.resize(cut);
        while (!cleaned.empty() && (cleaned.back() == ' ' || cleaned.back() == '.' || cleaned.back() == '_')) {
            cleaned.pop_back();
        }
        if (cleaned.empty()) cleaned = default_stem;
    }
    return cleaned + "." + extension;
}

ExportImageFormat parse_export_image_format(const std::optional<std::string>& raw) {
    if (!raw.has_value()) {
        return ExportImageFormat::Original;
    }
    const std::string cleaned = util::to_lower_ascii(util::trim_ascii(*raw));
    if (cleaned.empty() || cleaned == "original" || cleaned == "auto") {
        return ExportImageFormat::Original;
    }
    if (cleaned == "png") {
        return ExportImageFormat::Png;
    }
    if (cleaned == "jpg" || cleaned == "jpeg") {
        return ExportImageFormat::Jpg;
    }
    throw ComicExportError(
        "Unsupported image_format '" + *raw + "'. Expected 'original', 'png', or 'jpg'.");
}

std::vector<uint8_t> decode_export_image_reference(const std::string& ref, int page_index) {
    const std::string cleaned = util::trim_ascii(ref);
    if (cleaned.empty()) {
        throw ComicExportError("Page " + std::to_string(page_index) + " image reference is empty.");
    }

    const bool looks_like_cache_url = (cleaned.rfind("/api/cache/images/", 0) == 0);
    std::string stem_check = cleaned;
    if (const auto dot = stem_check.rfind('.'); dot != std::string::npos) {
        stem_check.resize(dot);
    }
    const bool looks_like_cache_id =
        !looks_like_cache_url &&
        cleaned.find(',') == std::string::npos &&
        cleaned.size() <= 134 &&
        is_safe_image_id(stem_check);

    if (looks_like_cache_url || looks_like_cache_id) {
        try {
            const fs::path cached = resolve_cached_image_path(cleaned);
            auto bytes = util::read_file_bytes(cached);
            if (bytes.has_value() && !bytes->empty()) {
                return *bytes;
            }
        } catch (const std::exception& ex) {
            if (looks_like_cache_url) {
                throw ComicExportError(ex.what());
            }
            // Otherwise fall through to base64 decode attempt.
        }
    }

    std::string_view b64_view = cleaned;
    if (cleaned.size() > 5 &&
        util::to_lower_ascii(cleaned.substr(0, 5)) == "data:") {
        const auto comma = cleaned.find(',');
        if (comma == std::string::npos) {
            throw ComicExportError("Page " + std::to_string(page_index) + " contains malformed data URI.");
        }
        b64_view = std::string_view(cleaned).substr(comma + 1);
    }

    auto decoded = util::base64_decode(b64_view);
    if (!decoded.has_value() || decoded->empty()) {
        throw ComicExportError(
            "Page " + std::to_string(page_index) + " contains invalid base64 or image URL.");
    }
    return *decoded;
}

// -----------------------------------------------------------------------------
// Archive Import (`POST /api/import/archive` & `/api/import/archive-path`) via miniz + stb_image
// -----------------------------------------------------------------------------

namespace {

std::vector<ExtractedArchivePage> extract_archive_pages_from_zip_handle(
    mz_zip_archive& zip,
    const fs::path& output_dir,
    size_t max_pages) {
    struct CandidateEntry {
        mz_uint index = 0;
        std::string member_name;
        uint64_t uncompressed_size = 0;
    };

    const mz_uint total_files = mz_zip_reader_get_num_files(&zip);
    std::vector<CandidateEntry> candidates;
    for (mz_uint i = 0; i < total_files; ++i) {
        if (mz_zip_reader_is_file_a_directory(&zip, i)) {
            continue;
        }
        mz_zip_archive_file_stat st{};
        if (!mz_zip_reader_file_stat(&zip, i, &st)) {
            continue;
        }
        const std::string name(st.m_filename);
        if (st.m_uncomp_size == 0 || !is_safe_archive_member(name)) {
            continue;
        }
        const std::string lower_name = util::to_lower_ascii(name);
        if (lower_name.size() >= 9 && lower_name.rfind("thumbs.db") == lower_name.size() - 9) {
            continue;
        }
        candidates.push_back({i, name, st.m_uncomp_size});
    }

    if (candidates.empty()) {
        throw ArchiveImportError(
            "No supported comic page images (.png, .jpg, .jpeg, .webp, .bmp) found inside the archive.");
    }

    std::sort(candidates.begin(), candidates.end(), [](const CandidateEntry& a, const CandidateEntry& b) {
        return natural_path_less(a.member_name, b.member_name);
    });

    if (candidates.size() > max_pages) {
        throw ArchiveImportError(
            "Archive contains " + std::to_string(candidates.size()) +
            " images, exceeding limit of " + std::to_string(max_pages) + " pages.");
    }

    uint64_t declared_total = 0;
    for (const auto& c : candidates) {
        declared_total += c.uncompressed_size;
        if (declared_total > kMaxArchiveUncompressedBytes) {
            throw ArchiveImportError("Archive uncompressed size exceeds the 32 GiB safety limit.");
        }
    }

    std::vector<ExtractedArchivePage> extracted;
    extracted.reserve(candidates.size());
    uint64_t cumulative_bytes = 0;

    for (size_t idx = 0; idx < candidates.size(); ++idx) {
        const auto& cand = candidates[idx];
        size_t extracted_size = 0;
        void* raw_ptr = mz_zip_reader_extract_to_heap(&zip, cand.index, &extracted_size, 0);
        if (!raw_ptr) {
            throw ArchiveImportError("Failed to read '" + cand.member_name + "' from archive.");
        }
        std::vector<uint8_t> raw_bytes(
            static_cast<const uint8_t*>(raw_ptr),
            static_cast<const uint8_t*>(raw_ptr) + extracted_size);
        mz_free(raw_ptr);

        if (raw_bytes.empty()) {
            throw ArchiveImportError("Archive entry '" + cand.member_name + "' is empty.");
        }
        if (raw_bytes.size() > kMaxSingleImageBytes) {
            throw ArchiveImportError("Archive entry '" + cand.member_name + "' exceeds 64 MiB limit.");
        }
        cumulative_bytes += static_cast<uint64_t>(raw_bytes.size());
        if (cumulative_bytes > kMaxArchiveUncompressedBytes) {
            throw ArchiveImportError("Archive extracted payload exceeds the 32 GiB safety limit.");
        }

        const DetectedImageFormat fmt = detect_image_magic(raw_bytes);
        int width = 0;
        int height = 0;
        std::string out_ext;
        std::vector<uint8_t> out_bytes;

        if (fmt == DetectedImageFormat::Webp) {
            if (!parse_webp_dimensions(raw_bytes, width, height)) {
                throw ArchiveImportError("Archive entry '" + cand.member_name + "' is not a valid WebP image.");
            }
            out_ext = "webp";
            out_bytes = std::move(raw_bytes);
        } else if (fmt == DetectedImageFormat::Jpeg || fmt == DetectedImageFormat::Png) {
            int channels = 0;
            if (!stbi_info_from_memory(
                    raw_bytes.data(), static_cast<int>(raw_bytes.size()), &width, &height, &channels) ||
                width <= 0 || height <= 0) {
                throw ArchiveImportError("Archive entry '" + cand.member_name + "' is not a valid image.");
            }
            out_ext = (fmt == DetectedImageFormat::Jpeg) ? "jpg" : "png";
            out_bytes = std::move(raw_bytes);
        } else {
            int channels = 0;
            StbiBuffer decoded(stbi_load_from_memory(
                raw_bytes.data(), static_cast<int>(raw_bytes.size()), &width, &height, &channels, 0));
            if (!decoded || width <= 0 || height <= 0) {
                throw ArchiveImportError("Archive entry '" + cand.member_name + "' is not a valid image.");
            }
            const auto rgb = flatten_pixels_to_rgb(decoded.get(), width, height, channels);
            out_ext = "png";
            out_bytes = encode_rgb_png(rgb.data(), width, height);
        }

        const std::string image_id = "imp_" + util::generate_uuid_hex();
        const fs::path final_path = output_dir / util::utf8_to_wide(image_id + "." + out_ext);
        std::string write_err;
        if (!util::atomic_write_bytes(final_path, out_bytes.data(), out_bytes.size(), &write_err)) {
            throw std::runtime_error("Failed to write cached archive page: " + write_err);
        }

        std::string clean_member = cand.member_name;
        std::replace(clean_member.begin(), clean_member.end(), '\\', '/');
        const auto slash = clean_member.rfind('/');
        const std::string leaf_name =
            (slash == std::string::npos) ? clean_member : clean_member.substr(slash + 1);

        extracted.push_back({
            static_cast<int>(idx + 1),
            leaf_name,
            clean_member,
            image_id,
            out_ext,
            final_path,
            {width, height},
            out_bytes.size(),
        });
    }

    prune_image_cache(output_dir);
    return extracted;
}

}  // namespace

std::vector<ExtractedArchivePage> extract_archive_pages_from_file(
    const fs::path& archive_path,
    const fs::path& output_dir,
    size_t max_pages) {
    std::error_code ec;
    if (!fs::is_regular_file(archive_path, ec) || ec) {
        throw ArchiveImportError("Archive file was not found on disk.");
    }
    const uint64_t file_size = static_cast<uint64_t>(fs::file_size(archive_path, ec));
    if (ec || file_size == 0) {
        throw ArchiveImportError("Archive file on disk is empty.");
    }

    fs::create_directories(output_dir, ec);

    FILE* fp = nullptr;
    if (_wfopen_s(&fp, archive_path.c_str(), L"rb") != 0 || !fp) {
        throw ArchiveImportError("Failed to open archive file for reading.");
    }
    struct FileGuard {
        FILE* f = nullptr;
        ~FileGuard() {
            if (f) std::fclose(f);
        }
    } file_guard{fp};

    mz_zip_archive zip{};
    if (!mz_zip_reader_init_cfile(&zip, fp, file_size, 0)) {
        throw ArchiveImportError("File is not a valid .cbz or .zip archive.");
    }
    struct ZipReaderGuard {
        mz_zip_archive* z;
        ~ZipReaderGuard() { mz_zip_reader_end(z); }
    } guard{&zip};

    return extract_archive_pages_from_zip_handle(zip, output_dir, max_pages);
}

std::vector<ExtractedArchivePage> import_folder_pages_from_dir(
    const fs::path& folder_path,
    const fs::path& output_dir,
    size_t max_pages) {
    std::error_code ec;
    if (!fs::is_directory(folder_path, ec) || ec) {
        throw ArchiveImportError("Folder path was not found on disk.");
    }

    struct FolderImageCand {
        fs::path file_path;
        std::string filename;
        std::string ext_lower;
    };
    std::vector<FolderImageCand> candidates;
    for (const auto& entry : fs::directory_iterator(folder_path, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec) || ec) continue;
        std::error_code sz_ec;
        if (entry.file_size(sz_ec) == 0 || sz_ec) continue;
        const fs::path p = entry.path();
        const std::string fname = util::path_to_utf8(p.filename());
        if (fname.empty() || fname.front() == '.' || fname.rfind("._", 0) == 0) continue;
        const std::string fname_lower = util::to_lower_ascii(fname);
        if (fname_lower == "thumbs.db" || fname_lower == ".ds_store" || fname_lower == "desktop.ini") continue;
        const std::string ext = util::to_lower_ascii(util::path_to_utf8(p.extension()));
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".webp" || ext == ".bmp") {
            candidates.push_back({p, fname, ext});
        }
    }

    if (candidates.empty()) {
        throw ArchiveImportError("No supported image files (.jpg, .jpeg, .png, .webp, .bmp) found in folder.");
    }
    if (candidates.size() > max_pages) {
        throw ArchiveImportError(
            "Folder contains " + std::to_string(candidates.size()) +
            " images, exceeding the maximum of " + std::to_string(max_pages) + " pages.");
    }

    std::sort(candidates.begin(), candidates.end(), [](const FolderImageCand& a, const FolderImageCand& b) {
        return natural_path_less(a.filename, b.filename);
    });

    fs::create_directories(output_dir, ec);

    std::vector<ExtractedArchivePage> imported;
    imported.reserve(candidates.size());

    for (size_t idx = 0; idx < candidates.size(); ++idx) {
        const auto& cand = candidates[idx];
        int width = 0;
        int height = 0;
        int channels = 0;
        std::string out_ext;
        const std::string image_id = "imp_" + util::generate_uuid_hex();
        fs::path final_path;
        size_t size_bytes = 0;

        if (cand.ext_lower == ".jpg" || cand.ext_lower == ".jpeg" || cand.ext_lower == ".png") {
            FILE* fp = nullptr;
            if (_wfopen_s(&fp, cand.file_path.c_str(), L"rb") != 0 || !fp) {
                throw ArchiveImportError("Failed to open image file '" + cand.filename + "'.");
            }
            const int ok = stbi_info_from_file(fp, &width, &height, &channels);
            std::fclose(fp);
            if (!ok || width <= 0 || height <= 0) {
                throw ArchiveImportError("File '" + cand.filename + "' is not a valid image.");
            }
            out_ext = (cand.ext_lower == ".png") ? "png" : "jpg";
            final_path = output_dir / util::utf8_to_wide(image_id + "." + out_ext);
            fs::copy_file(cand.file_path, final_path, fs::copy_options::overwrite_existing, ec);
            if (ec) {
                throw std::runtime_error("Failed to copy image '" + cand.filename + "' into cache: " + ec.message());
            }
            size_bytes = static_cast<size_t>(fs::file_size(final_path, ec));
        } else if (cand.ext_lower == ".webp") {
            std::ifstream ifs(cand.file_path, std::ios::binary);
            std::vector<uint8_t> header(64, 0);
            ifs.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()));
            header.resize(static_cast<size_t>(ifs.gcount()));
            ifs.close();
            if (!parse_webp_dimensions(header, width, height) || width <= 0 || height <= 0) {
                throw ArchiveImportError("File '" + cand.filename + "' is not a valid WebP image.");
            }
            out_ext = "webp";
            final_path = output_dir / util::utf8_to_wide(image_id + "." + out_ext);
            fs::copy_file(cand.file_path, final_path, fs::copy_options::overwrite_existing, ec);
            if (ec) {
                throw std::runtime_error("Failed to copy WebP '" + cand.filename + "' into cache: " + ec.message());
            }
            size_bytes = static_cast<size_t>(fs::file_size(final_path, ec));
        } else {
            // BMP -> convert to PNG
            FILE* fp = nullptr;
            if (_wfopen_s(&fp, cand.file_path.c_str(), L"rb") != 0 || !fp) {
                throw ArchiveImportError("Failed to open BMP file '" + cand.filename + "'.");
            }
            StbiBuffer decoded(stbi_load_from_file(fp, &width, &height, &channels, 0));
            std::fclose(fp);
            if (!decoded || width <= 0 || height <= 0) {
                throw ArchiveImportError("File '" + cand.filename + "' is not a valid BMP image.");
            }
            const auto rgb = flatten_pixels_to_rgb(decoded.get(), width, height, channels);
            const auto out_bytes = encode_rgb_png(rgb.data(), width, height);
            out_ext = "png";
            final_path = output_dir / util::utf8_to_wide(image_id + "." + out_ext);
            std::string write_err;
            if (!util::atomic_write_bytes(final_path, out_bytes.data(), out_bytes.size(), &write_err)) {
                throw std::runtime_error("Failed to write cached BMP page: " + write_err);
            }
            size_bytes = out_bytes.size();
        }

        imported.push_back({
            static_cast<int>(idx + 1),
            cand.filename,
            cand.filename,
            image_id,
            out_ext,
            final_path,
            {width, height},
            size_bytes,
        });
    }

    prune_image_cache(output_dir);
    return imported;
}

std::vector<ExtractedArchivePage> extract_archive_pages(
    const std::vector<uint8_t>& archive_bytes,
    const fs::path& output_dir,
    size_t max_pages) {
    if (archive_bytes.empty()) {
        throw ArchiveImportError("Uploaded archive file is empty.");
    }

    std::error_code dir_ec;
    fs::create_directories(output_dir, dir_ec);

    const fs::path tmp_archive = output_dir / util::utf8_to_wide("tmp_import_" + util::generate_uuid_hex() + ".cbz");
    std::string write_err;
    if (!util::atomic_write_bytes(tmp_archive, archive_bytes.data(), archive_bytes.size(), &write_err)) {
        throw ArchiveImportError("Failed to stage temporary archive file on disk: " + write_err);
    }
    struct TempFileGuard {
        fs::path p;
        ~TempFileGuard() {
            std::error_code ec;
            fs::remove(p, ec);
        }
    } tmp_guard{tmp_archive};

    return extract_archive_pages_from_file(tmp_archive, output_dir, max_pages);
}

// -----------------------------------------------------------------------------
// CBZ Archive Export (`POST /api/export/cbz`) via miniz + stb_image
// -----------------------------------------------------------------------------

std::vector<uint8_t> build_cbz_bytes(
    const std::vector<ExportPageInput>& pages,
    ExportImageFormat image_format,
    int jpeg_quality,
    bool smart_ssim_export) {
    if (pages.empty()) {
        throw ComicExportError("At least one page image is required to export a .cbz archive.");
    }
    if (pages.size() > kMaxArchivePages) {
        throw ComicExportError(
            "Cannot export " + std::to_string(pages.size()) +
            " pages; maximum is " + std::to_string(kMaxArchivePages) + ".");
    }

    const size_t pad_width = std::max<size_t>(3, std::to_string(pages.size()).size());

    mz_zip_archive zip{};
    if (!mz_zip_writer_init_heap(&zip, 0, 1024ULL * 1024ULL)) {
        throw ComicExportError("Failed to initialize in-memory CBZ writer.");
    }
    bool finalized = false;
    struct ZipWriterGuard {
        mz_zip_archive* z;
        bool* done;
        ~ZipWriterGuard() {
            if (!*done) {
                mz_zip_writer_end(z);
            }
        }
    } guard{&zip, &finalized};

    for (size_t i = 0; i < pages.size(); ++i) {
        const int page_num = static_cast<int>(i + 1);
        const auto& raw = pages[i].image_bytes;
        if (raw.empty()) {
            throw ComicExportError("Page " + std::to_string(page_num) + " image payload is empty.");
        }
        if (raw.size() > kMaxSingleImageBytes) {
            throw ComicExportError("Page " + std::to_string(page_num) + " exceeds maximum allowed image size.");
        }

        const DetectedImageFormat fmt = detect_image_magic(raw);
        std::vector<uint8_t> encoded;
        std::string ext;

        if (image_format == ExportImageFormat::Original && fmt == DetectedImageFormat::Webp) {
            int w = 0, h = 0;
            if (!parse_webp_dimensions(raw, w, h)) {
                throw ComicExportError("Page " + std::to_string(page_num) + " is not a valid WebP image.");
            }
            encoded = raw;
            ext = "webp";
        } else {
            int w = 0, h = 0, channels = 0;
            StbiBuffer decoded(stbi_load_from_memory(
                raw.data(), static_cast<int>(raw.size()), &w, &h, &channels, 0));
            if (!decoded || w <= 0 || h <= 0) {
                throw ComicExportError("Page " + std::to_string(page_num) + " is not a valid image.");
            }

            if (image_format == ExportImageFormat::Original) {
                if (fmt == DetectedImageFormat::Jpeg) {
                    encoded = raw;
                    ext = "jpg";
                } else if (fmt == DetectedImageFormat::Png) {
                    encoded = raw;
                    ext = "png";
                } else {
                    const auto rgb = flatten_pixels_to_rgb(decoded.get(), w, h, channels);
                    encoded = encode_rgb_png(rgb.data(), w, h, smart_ssim_export);
                    ext = "png";
                }
            } else if (image_format == ExportImageFormat::Jpg) {
                const auto rgb = flatten_pixels_to_rgb(decoded.get(), w, h, channels);
                encoded = encode_rgb_jpg(rgb.data(), w, h, jpeg_quality, smart_ssim_export);
                ext = "jpg";
            } else {  // ExportImageFormat::Png
                if (fmt == DetectedImageFormat::Png && !smart_ssim_export) {
                    encoded = raw;
                    ext = "png";
                } else {
                    const auto rgb = flatten_pixels_to_rgb(decoded.get(), w, h, channels);
                    encoded = encode_rgb_png(rgb.data(), w, h, smart_ssim_export);
                    ext = "png";
                }
            }
        }

        std::ostringstream prefix_ss;
        prefix_ss << std::setw(static_cast<int>(pad_width)) << std::setfill('0') << page_num;
        const std::string prefix = prefix_ss.str();
        const std::string stem = sanitize_page_stem(pages[i].filename);
        const std::string entry_name =
            (!stem.empty() && stem.rfind(prefix, 0) != 0)
            ? (prefix + "_" + stem + "." + ext)
            : (prefix + "." + ext);

        if (!mz_zip_writer_add_mem(
                &zip,
                entry_name.c_str(),
                encoded.data(),
                encoded.size(),
                MZ_NO_COMPRESSION)) {
            throw ComicExportError("Failed to add '" + entry_name + "' to CBZ archive.");
        }
    }

    void* out_buf = nullptr;
    size_t out_size = 0;
    if (!mz_zip_writer_finalize_heap_archive(&zip, &out_buf, &out_size) || !out_buf) {
        throw ComicExportError("Failed to finalize CBZ archive.");
    }
    mz_zip_writer_end(&zip);
    finalized = true;

    std::vector<uint8_t> result(
        static_cast<const uint8_t*>(out_buf),
        static_cast<const uint8_t*>(out_buf) + out_size);
    mz_free(out_buf);
    return result;
}

// -----------------------------------------------------------------------------
// Native C++ PDF 1.4 Multi-Page Writer (`POST /api/export/pdf`)
// -----------------------------------------------------------------------------

std::vector<uint8_t> build_pdf_bytes(
    const std::vector<ExportPageInput>& pages,
    bool smart_ssim_export) {
    if (pages.empty()) {
        throw ComicExportError("At least one page image is required to export a .pdf document.");
    }
    if (pages.size() > kMaxArchivePages) {
        throw ComicExportError(
            "Cannot export " + std::to_string(pages.size()) +
            " pages; maximum is " + std::to_string(kMaxArchivePages) + ".");
    }

    struct PdfImageStream {
        int width = 0;
        int height = 0;
        std::string color_space;  // "/DeviceRGB", "/DeviceGray", or "/DeviceCMYK"
        std::string filter;       // "/DCTDecode" or "/FlateDecode"
        bool invert_cmyk = false;
        std::vector<uint8_t> stream_data;
    };

    std::vector<PdfImageStream> prepared;
    prepared.reserve(pages.size());

    for (size_t i = 0; i < pages.size(); ++i) {
        const int page_num = static_cast<int>(i + 1);
        const auto& raw = pages[i].image_bytes;
        if (raw.empty()) {
            throw ComicExportError("Page " + std::to_string(page_num) + " image payload is empty.");
        }
        if (raw.size() > kMaxSingleImageBytes) {
            throw ComicExportError("Page " + std::to_string(page_num) + " exceeds maximum allowed image size.");
        }

        const DetectedImageFormat fmt = detect_image_magic(raw);
        PdfImageStream img;

        // 1. Direct lossless JPEG passthrough via /DCTDecode (zero recompression)
        JpegHeaderInfo jpeg_info{};
        int probe_w = 0, probe_h = 0, probe_c = 0;
        if (fmt == DetectedImageFormat::Jpeg &&
            parse_jpeg_header(raw, jpeg_info) &&
            stbi_info_from_memory(raw.data(), static_cast<int>(raw.size()), &probe_w, &probe_h, &probe_c)) {
            img.width = jpeg_info.width;
            img.height = jpeg_info.height;
            img.filter = "/DCTDecode";
            if (jpeg_info.components == 1) {
                img.color_space = "/DeviceGray";
            } else if (jpeg_info.components == 4) {
                img.color_space = "/DeviceCMYK";
                img.invert_cmyk = jpeg_info.is_adobe_cmyk;
            } else {
                img.color_space = "/DeviceRGB";
            }
            img.stream_data = raw;
            prepared.push_back(std::move(img));
            continue;
        }

        // 2. PNG / BMP / RGBA -> decode, flatten alpha onto white, zlib-compress for lossless /FlateDecode
        int w = 0, h = 0, channels = 0;
        StbiBuffer decoded(stbi_load_from_memory(
            raw.data(), static_cast<int>(raw.size()), &w, &h, &channels, 0));
        if (!decoded || w <= 0 || h <= 0) {
            throw ComicExportError("Page " + std::to_string(page_num) + " is not a valid image.");
        }

        std::vector<uint8_t> raw_pixels;
        if (channels == 1) {
            img.color_space = "/DeviceGray";
            raw_pixels.assign(decoded.get(), decoded.get() + static_cast<size_t>(w) * static_cast<size_t>(h));
        } else if (channels == 2) {
            img.color_space = "/DeviceGray";
            const size_t px_count = static_cast<size_t>(w) * static_cast<size_t>(h);
            raw_pixels.resize(px_count);
            for (size_t p = 0; p < px_count; ++p) {
                const uint32_t g = decoded.get()[p * 2 + 0];
                const uint32_t a = decoded.get()[p * 2 + 1];
                raw_pixels[p] = static_cast<uint8_t>((g * a + 255U * (255U - a) + 127U) / 255U);
            }
        } else {
            const auto rgb = flatten_pixels_to_rgb(decoded.get(), w, h, channels);
            if (smart_ssim_export &&
                pipeline::math_inpaint::computeHaslerSusstrunkColorfulness(rgb.data(), w, h, 4) < 8.0f) {
                img.color_space = "/DeviceGray";
                pipeline::math_inpaint::rgbToGray8(rgb.data(), w, h, raw_pixels);
            } else {
                img.color_space = "/DeviceRGB";
                raw_pixels = rgb;
            }
        }

        mz_ulong comp_bound = mz_compressBound(static_cast<mz_ulong>(raw_pixels.size()));
        std::vector<uint8_t> compressed(static_cast<size_t>(comp_bound));
        const int zres = mz_compress2(
            compressed.data(),
            &comp_bound,
            raw_pixels.data(),
            static_cast<mz_ulong>(raw_pixels.size()),
            MZ_DEFAULT_COMPRESSION);
        if (zres != MZ_OK) {
            throw ComicExportError("Failed to Flate-compress page " + std::to_string(page_num) + " for PDF.");
        }
        compressed.resize(static_cast<size_t>(comp_bound));

        img.width = w;
        img.height = h;
        img.filter = "/FlateDecode";
        img.stream_data = std::move(compressed);
        prepared.push_back(std::move(img));
    }

    // Assemble PDF 1.4 binary document:
    //   Obj 1             : /Catalog -> /Pages 2 0 R
    //   Obj 2             : /Pages   -> /Kids [5 0 R, 8 0 R, ...] /Count N
    //   Obj 3 + i*3       : Image XObject (/Im0)
    //   Obj 4 + i*3       : Page Content Stream
    //   Obj 5 + i*3       : /Page Dictionary
    const size_t page_count = prepared.size();
    const size_t total_objects = 2 + page_count * 3;
    std::vector<size_t> obj_offsets(total_objects + 1, 0);

    std::vector<uint8_t> pdf;
    size_t estimated_size = 4096;
    for (const auto& img : prepared) {
        estimated_size += img.stream_data.size() + 512;
    }
    pdf.reserve(estimated_size);

    auto append_str = [&pdf](std::string_view s) {
        pdf.insert(pdf.end(), s.begin(), s.end());
    };
    auto append_bytes = [&pdf](const std::vector<uint8_t>& b) {
        pdf.insert(pdf.end(), b.begin(), b.end());
    };
    auto begin_obj = [&](size_t obj_id) {
        obj_offsets[obj_id] = pdf.size();
        append_str(std::to_string(obj_id) + " 0 obj\n");
    };

    // PDF 1.4 Header + binary comment marker
    append_str("%PDF-1.4\n");
    pdf.push_back('%');
    pdf.push_back(0xE2);
    pdf.push_back(0xE3);
    pdf.push_back(0xCF);
    pdf.push_back(0xD3);
    pdf.push_back('\n');

    // Object 1: Document Catalog
    begin_obj(1);
    append_str("<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    // Object 2: Pages Tree
    begin_obj(2);
    {
        std::ostringstream kids;
        kids << "<< /Type /Pages /Kids [";
        for (size_t i = 0; i < page_count; ++i) {
            if (i > 0) kids << ' ';
            kids << (5 + i * 3) << " 0 R";
        }
        kids << "] /Count " << page_count << " >>\nendobj\n";
        append_str(kids.str());
    }

    // Per-page objects (Image XObject, Content Stream, Page Dictionary)
    for (size_t i = 0; i < page_count; ++i) {
        const auto& img = prepared[i];
        const size_t img_obj_id = 3 + i * 3;
        const size_t content_obj_id = 4 + i * 3;
        const size_t page_obj_id = 5 + i * 3;

        // Match standard 96 DPI -> 72 PDF points per inch
        const std::string pt_w = format_pdf_number(static_cast<double>(img.width) * 72.0 / 96.0);
        const std::string pt_h = format_pdf_number(static_cast<double>(img.height) * 72.0 / 96.0);

        // Image XObject
        begin_obj(img_obj_id);
        {
            std::ostringstream dict;
            dict << "<< /Type /XObject /Subtype /Image"
                 << " /Width " << img.width
                 << " /Height " << img.height
                 << " /ColorSpace " << img.color_space
                 << " /BitsPerComponent 8"
                 << " /Filter " << img.filter;
            if (img.invert_cmyk) {
                dict << " /Decode [1 0 1 0 1 0 1 0]";
            }
            dict << " /Length " << img.stream_data.size()
                 << " >>\nstream\n";
            append_str(dict.str());
            append_bytes(img.stream_data);
            append_str("\nendstream\nendobj\n");
        }

        // Page Content Stream
        const std::string content_stream =
            "q\n" + pt_w + " 0 0 " + pt_h + " 0 0 cm\n/Im0 Do\nQ\n";
        begin_obj(content_obj_id);
        append_str(
            "<< /Length " + std::to_string(content_stream.size()) +
            " >>\nstream\n" + content_stream + "endstream\nendobj\n");

        // Page Dictionary
        begin_obj(page_obj_id);
        {
            std::ostringstream page_dict;
            page_dict << "<< /Type /Page /Parent 2 0 R"
                      << " /MediaBox [0 0 " << pt_w << " " << pt_h << "]"
                      << " /Resources << /XObject << /Im0 " << img_obj_id << " 0 R >> >>"
                      << " /Contents " << content_obj_id << " 0 R"
                      << " >>\nendobj\n";
            append_str(page_dict.str());
        }
    }

    // Cross-reference table (PDF 1.4 strict 20-byte entries)
    const size_t xref_offset = pdf.size();
    append_str("xref\n0 " + std::to_string(total_objects + 1) + "\n");
    append_str("0000000000 65535 f \n");
    for (size_t obj_id = 1; obj_id <= total_objects; ++obj_id) {
        char line[32]{};
        std::snprintf(
            line, sizeof(line), "%010llu 00000 n \n",
            static_cast<unsigned long long>(obj_offsets[obj_id]));
        append_str(line);
    }

    // Trailer & EOF
    append_str(
        "trailer\n<< /Size " + std::to_string(total_objects + 1) +
        " /Root 1 0 R >>\nstartxref\n" +
        std::to_string(xref_offset) + "\n%%EOF\n");

    return pdf;
}

// -----------------------------------------------------------------------------
// Disk-Backed O(1)-RAM CBZ & PDF Writers (Supports 1,500 - 10,000+ Pages)
// -----------------------------------------------------------------------------

void build_cbz_to_file(
    const std::vector<DiskExportPageEntry>& pages,
    const fs::path& output_path,
    ExportImageFormat image_format,
    int jpeg_quality,
    bool smart_ssim_export) {
    if (pages.empty()) {
        throw ComicExportError("At least one page image is required to export a .cbz archive.");
    }
    if (pages.size() > kMaxArchivePages) {
        throw ComicExportError(
            "Cannot export " + std::to_string(pages.size()) +
            " pages; maximum is " + std::to_string(kMaxArchivePages) + ".");
    }

    const size_t pad_width = std::max<size_t>(4, std::to_string(pages.size()).size());

    FILE* fp = nullptr;
    if (_wfopen_s(&fp, output_path.c_str(), L"wb") != 0 || !fp) {
        throw ComicExportError("Failed to create output CBZ file on disk.");
    }
    struct FileGuard {
        FILE* f = nullptr;
        ~FileGuard() {
            if (f) std::fclose(f);
        }
    } file_guard{fp};

    mz_zip_archive zip{};
    if (!mz_zip_writer_init_cfile_v2(&zip, fp, MZ_ZIP_FLAG_WRITE_ZIP64)) {
        throw ComicExportError("Failed to initialize disk-backed CBZ writer.");
    }
    bool finalized = false;
    struct ZipWriterGuard {
        mz_zip_archive* z;
        bool* done;
        ~ZipWriterGuard() {
            if (!*done) {
                mz_zip_writer_end(z);
            }
        }
    } zip_guard{&zip, &finalized};

    for (size_t i = 0; i < pages.size(); ++i) {
        const int page_num = static_cast<int>(i + 1);
        const auto raw_opt = util::read_file_bytes(pages[i].file_path);
        if (!raw_opt.has_value() || raw_opt->empty()) {
            throw ComicExportError("Page " + std::to_string(page_num) + " image file is missing or empty.");
        }
        const auto& raw = *raw_opt;
        if (raw.size() > kMaxSingleImageBytes) {
            throw ComicExportError("Page " + std::to_string(page_num) + " exceeds maximum allowed image size.");
        }

        const DetectedImageFormat fmt = detect_image_magic(raw);
        const uint8_t* write_ptr = raw.data();
        size_t write_size = raw.size();
        std::vector<uint8_t> reencoded;
        std::string ext;

        // Zero-copy passthrough when target format matches source (avoids 1,500+ full pixel decodes)
        if (image_format == ExportImageFormat::Original && fmt == DetectedImageFormat::Webp) {
            int w = 0, h = 0;
            if (!parse_webp_dimensions(raw, w, h)) {
                throw ComicExportError("Page " + std::to_string(page_num) + " is not a valid WebP image.");
            }
            ext = "webp";
        } else if ((image_format == ExportImageFormat::Original &&
                    (fmt == DetectedImageFormat::Png || fmt == DetectedImageFormat::Jpeg)) ||
                   (image_format == ExportImageFormat::Png && fmt == DetectedImageFormat::Png)) {
            int w = 0, h = 0, c = 0;
            if (!stbi_info_from_memory(raw.data(), static_cast<int>(raw.size()), &w, &h, &c) ||
                w <= 0 || h <= 0) {
                throw ComicExportError("Page " + std::to_string(page_num) + " is not a valid image.");
            }
            ext = (fmt == DetectedImageFormat::Jpeg) ? "jpg" : "png";
        } else {
            int w = 0, h = 0, channels = 0;
            StbiBuffer decoded(stbi_load_from_memory(
                raw.data(), static_cast<int>(raw.size()), &w, &h, &channels, 0));
            if (!decoded || w <= 0 || h <= 0) {
                throw ComicExportError("Page " + std::to_string(page_num) + " is not a valid image.");
            }
            const auto rgb = flatten_pixels_to_rgb(decoded.get(), w, h, channels);
            if (image_format == ExportImageFormat::Jpg) {
                reencoded = encode_rgb_jpg(rgb.data(), w, h, jpeg_quality, smart_ssim_export);
                ext = "jpg";
            } else {
                reencoded = encode_rgb_png(rgb.data(), w, h, smart_ssim_export);
                ext = "png";
            }
            write_ptr = reencoded.data();
            write_size = reencoded.size();
        }

        std::ostringstream prefix_ss;
        prefix_ss << std::setw(static_cast<int>(pad_width)) << std::setfill('0') << page_num;
        const std::string prefix = prefix_ss.str();
        const std::string stem = sanitize_page_stem(pages[i].filename);
        const std::string entry_name =
            (!stem.empty() && stem.rfind(prefix, 0) != 0)
            ? (prefix + "_" + stem + "." + ext)
            : (prefix + "." + ext);

        if (!mz_zip_writer_add_mem(
                &zip,
                entry_name.c_str(),
                write_ptr,
                write_size,
                MZ_NO_COMPRESSION)) {
            throw ComicExportError("Failed to add '" + entry_name + "' to CBZ archive.");
        }
    }

    if (!mz_zip_writer_finalize_archive(&zip)) {
        throw ComicExportError("Failed to finalize disk-backed CBZ archive.");
    }
    mz_zip_writer_end(&zip);
    finalized = true;
}

void build_pdf_to_file(
    const std::vector<DiskExportPageEntry>& pages,
    const fs::path& output_path,
    bool smart_ssim_export) {
    if (pages.empty()) {
        throw ComicExportError("At least one page image is required to export a .pdf document.");
    }
    if (pages.size() > kMaxArchivePages) {
        throw ComicExportError(
            "Cannot export " + std::to_string(pages.size()) +
            " pages; maximum is " + std::to_string(kMaxArchivePages) + ".");
    }

    std::ofstream ofs(output_path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        throw ComicExportError("Failed to create output PDF file on disk.");
    }

    const size_t page_count = pages.size();
    const size_t total_objects = 2 + page_count * 3;
    std::vector<uint64_t> obj_offsets(total_objects + 1, 0);
    uint64_t bytes_written = 0;

    auto write_raw = [&](const void* data, size_t len) {
        if (len == 0) return;
        ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(len));
        if (!ofs.good()) {
            throw ComicExportError("Disk write error while building PDF document.");
        }
        bytes_written += static_cast<uint64_t>(len);
    };
    auto write_str = [&](std::string_view s) {
        write_raw(s.data(), s.size());
    };
    auto begin_obj = [&](size_t obj_id) {
        obj_offsets[obj_id] = bytes_written;
        write_str(std::to_string(obj_id) + " 0 obj\n");
    };

    // PDF 1.4 Header + binary comment marker
    write_str("%PDF-1.4\n");
    const uint8_t bin_hdr[6] = {'%', 0xE2, 0xE3, 0xCF, 0xD3, '\n'};
    write_raw(bin_hdr, sizeof(bin_hdr));

    // Object 1: Document Catalog
    begin_obj(1);
    write_str("<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    // Object 2: Pages Tree (object IDs are deterministic: 5 + i*3)
    begin_obj(2);
    {
        std::ostringstream kids;
        kids << "<< /Type /Pages /Kids [";
        for (size_t i = 0; i < page_count; ++i) {
            if (i > 0) kids << ' ';
            kids << (5 + i * 3) << " 0 R";
        }
        kids << "] /Count " << page_count << " >>\nendobj\n";
        write_str(kids.str());
    }

    // Stream each page from disk one at a time -> O(1) RAM regardless of page count
    for (size_t i = 0; i < page_count; ++i) {
        const int page_num = static_cast<int>(i + 1);
        const auto raw_opt = util::read_file_bytes(pages[i].file_path);
        if (!raw_opt.has_value() || raw_opt->empty()) {
            throw ComicExportError("Page " + std::to_string(page_num) + " image file is missing or empty.");
        }
        const auto& raw = *raw_opt;
        if (raw.size() > kMaxSingleImageBytes) {
            throw ComicExportError("Page " + std::to_string(page_num) + " exceeds maximum allowed image size.");
        }

        const DetectedImageFormat fmt = detect_image_magic(raw);
        int img_w = 0;
        int img_h = 0;
        std::string color_space;
        std::string filter;
        bool invert_cmyk = false;
        const uint8_t* stream_ptr = nullptr;
        size_t stream_len = 0;
        std::vector<uint8_t> compressed_buf;

        JpegHeaderInfo jpeg_info{};
        int probe_w = 0, probe_h = 0, probe_c = 0;
        if (fmt == DetectedImageFormat::Jpeg &&
            parse_jpeg_header(raw, jpeg_info) &&
            stbi_info_from_memory(raw.data(), static_cast<int>(raw.size()), &probe_w, &probe_h, &probe_c)) {
            img_w = jpeg_info.width;
            img_h = jpeg_info.height;
            filter = "/DCTDecode";
            if (jpeg_info.components == 1) {
                color_space = "/DeviceGray";
            } else if (jpeg_info.components == 4) {
                color_space = "/DeviceCMYK";
                invert_cmyk = jpeg_info.is_adobe_cmyk;
            } else {
                color_space = "/DeviceRGB";
            }
            stream_ptr = raw.data();
            stream_len = raw.size();
        } else {
            int w = 0, h = 0, channels = 0;
            StbiBuffer decoded(stbi_load_from_memory(
                raw.data(), static_cast<int>(raw.size()), &w, &h, &channels, 0));
            if (!decoded || w <= 0 || h <= 0) {
                throw ComicExportError("Page " + std::to_string(page_num) + " is not a valid image.");
            }
            std::vector<uint8_t> raw_pixels;
            if (channels == 1) {
                color_space = "/DeviceGray";
                raw_pixels.assign(decoded.get(), decoded.get() + static_cast<size_t>(w) * static_cast<size_t>(h));
            } else if (channels == 2) {
                color_space = "/DeviceGray";
                const size_t px_count = static_cast<size_t>(w) * static_cast<size_t>(h);
                raw_pixels.resize(px_count);
                for (size_t p = 0; p < px_count; ++p) {
                    const uint32_t g = decoded.get()[p * 2 + 0];
                    const uint32_t a = decoded.get()[p * 2 + 1];
                    raw_pixels[p] = static_cast<uint8_t>((g * a + 255U * (255U - a) + 127U) / 255U);
                }
            } else {
                const auto rgb = flatten_pixels_to_rgb(decoded.get(), w, h, channels);
                if (smart_ssim_export &&
                    pipeline::math_inpaint::computeHaslerSusstrunkColorfulness(rgb.data(), w, h, 4) < 8.0f) {
                    color_space = "/DeviceGray";
                    pipeline::math_inpaint::rgbToGray8(rgb.data(), w, h, raw_pixels);
                } else {
                    color_space = "/DeviceRGB";
                    raw_pixels = rgb;
                }
            }

            mz_ulong comp_bound = mz_compressBound(static_cast<mz_ulong>(raw_pixels.size()));
            compressed_buf.resize(static_cast<size_t>(comp_bound));
            const int zres = mz_compress2(
                compressed_buf.data(),
                &comp_bound,
                raw_pixels.data(),
                static_cast<mz_ulong>(raw_pixels.size()),
                MZ_BEST_SPEED);
            if (zres != MZ_OK) {
                throw ComicExportError("Failed to Flate-compress page " + std::to_string(page_num) + " for PDF.");
            }
            compressed_buf.resize(static_cast<size_t>(comp_bound));
            img_w = w;
            img_h = h;
            filter = "/FlateDecode";
            stream_ptr = compressed_buf.data();
            stream_len = compressed_buf.size();
        }

        const size_t img_obj_id = 3 + i * 3;
        const size_t content_obj_id = 4 + i * 3;
        const size_t page_obj_id = 5 + i * 3;

        const std::string pt_w = format_pdf_number(static_cast<double>(img_w) * 72.0 / 96.0);
        const std::string pt_h = format_pdf_number(static_cast<double>(img_h) * 72.0 / 96.0);

        begin_obj(img_obj_id);
        {
            std::ostringstream dict;
            dict << "<< /Type /XObject /Subtype /Image"
                 << " /Width " << img_w
                 << " /Height " << img_h
                 << " /ColorSpace " << color_space
                 << " /BitsPerComponent 8"
                 << " /Filter " << filter;
            if (invert_cmyk) {
                dict << " /Decode [1 0 1 0 1 0 1 0]";
            }
            dict << " /Length " << stream_len << " >>\nstream\n";
            write_str(dict.str());
            write_raw(stream_ptr, stream_len);
            write_str("\nendstream\nendobj\n");
        }

        const std::string content_stream =
            "q\n" + pt_w + " 0 0 " + pt_h + " 0 0 cm\n/Im0 Do\nQ\n";
        begin_obj(content_obj_id);
        write_str(
            "<< /Length " + std::to_string(content_stream.size()) +
            " >>\nstream\n" + content_stream + "endstream\nendobj\n");

        begin_obj(page_obj_id);
        {
            std::ostringstream page_dict;
            page_dict << "<< /Type /Page /Parent 2 0 R"
                      << " /MediaBox [0 0 " << pt_w << " " << pt_h << "]"
                      << " /Resources << /XObject << /Im0 " << img_obj_id << " 0 R >> >>"
                      << " /Contents " << content_obj_id << " 0 R"
                      << " >>\nendobj\n";
            write_str(page_dict.str());
        }
    }

    // Cross-reference table (PDF 1.4 10-digit xref offset guard = 10 GB)
    if (bytes_written >= 10000000000ULL) {
        throw ComicExportError(
            "PDF export exceeds the 10 GB PDF 1.4 cross-reference offset limit. Please export as .cbz (ZIP64) or split the volume.");
    }
    const uint64_t xref_offset = bytes_written;
    write_str("xref\n0 " + std::to_string(total_objects + 1) + "\n");
    write_str("0000000000 65535 f \n");
    for (size_t obj_id = 1; obj_id <= total_objects; ++obj_id) {
        char line[32]{};
        std::snprintf(
            line, sizeof(line), "%010llu 00000 n \n",
            static_cast<unsigned long long>(obj_offsets[obj_id]));
        write_str(line);
    }

    write_str(
        "trailer\n<< /Size " + std::to_string(total_objects + 1) +
        " /Root 1 0 R >>\nstartxref\n" +
        std::to_string(xref_offset) + "\n%%EOF\n");
    ofs.flush();
    if (!ofs.good()) {
        throw ComicExportError("Failed to flush PDF file to disk.");
    }
}

std::vector<uint8_t> build_zip_from_named_buffers(
    const std::vector<std::pair<std::string, std::vector<uint8_t>>>& entries) {
    if (entries.empty()) {
        throw ComicExportError("At least one file entry is required to build a .zip archive.");
    }
    mz_zip_archive zip{};
    if (!mz_zip_writer_init_heap(&zip, 0, 1024ULL * 1024ULL)) {
        throw ComicExportError("Failed to initialize in-memory ZIP writer.");
    }
    bool finalized = false;
    struct ZipWriterGuard {
        mz_zip_archive* z;
        bool* done;
        ~ZipWriterGuard() {
            if (!*done) {
                mz_zip_writer_end(z);
            }
        }
    } guard{&zip, &finalized};

    for (const auto& [name, bytes] : entries) {
        if (!mz_zip_writer_add_mem(
                &zip,
                name.c_str(),
                bytes.data(),
                bytes.size(),
                MZ_BEST_SPEED)) {
            throw ComicExportError("Failed to add '" + name + "' to ZIP archive.");
        }
    }

    void* out_buf = nullptr;
    size_t out_size = 0;
    if (!mz_zip_writer_finalize_heap_archive(&zip, &out_buf, &out_size) || !out_buf) {
        throw ComicExportError("Failed to finalize ZIP archive.");
    }
    mz_zip_writer_end(&zip);
    finalized = true;

    std::vector<uint8_t> result(
        static_cast<const uint8_t*>(out_buf),
        static_cast<const uint8_t*>(out_buf) + out_size);
    mz_free(out_buf);
    return result;
}

}  // namespace mangaflow
