#include "font_manager.hpp"
#include "utils.hpp"

#include <winhttp.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <regex>
#include <unordered_map>
#include <unordered_set>

namespace mangaflow {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr size_t kMaxFontFileBytes = 50ULL * 1024ULL * 1024ULL;
constexpr size_t kMinFontFileBytes = 64ULL;

struct CatalogFontSpec {
    std::string id;
    std::string family;
    std::string style_category;
    std::vector<std::string> scripts;
    std::string description;
};

std::string slugify_font_id(const std::string& family) {
    std::string out;
    bool prev_dash = true;
    for (unsigned char c : family) {
        if (std::isalnum(c)) {
            out.push_back(static_cast<char>(std::tolower(c)));
            prev_dash = false;
        } else if (!prev_dash) {
            out.push_back('-');
            prev_dash = true;
        }
    }
    while (!out.empty() && out.back() == '-') {
        out.pop_back();
    }
    return out.empty() ? "font" : out;
}

const std::vector<CatalogFontSpec>& curated_google_fonts() {
    static const std::vector<CatalogFontSpec> kCatalog = []() {
        struct RawItem {
            const char* family;
            const char* cat;
            std::vector<std::string> scripts;
            const char* desc;
        };
        const std::vector<RawItem> raw = {
            // --- Comic & Manga Dialogue (25) ---
            {"Comic Neue", "comic", {"latin"}, "Clean classic comic book speech bubble font"},
            {"Bangers", "comic", {"latin"}, "Bold comic book impact and shouting font"},
            {"Patrick Hand", "comic", {"latin", "vietnamese"}, "Warm hand-lettered dialogue bubbles"},
            {"Patrick Hand SC", "comic", {"latin", "vietnamese"}, "Small-caps manga lettering style"},
            {"Architects Daughter", "comic", {"latin"}, "Crisp hand-drawn comic dialogue"},
            {"Kalam", "comic", {"latin", "devanagari"}, "Expressive hand-lettered comic script"},
            {"Neucha", "comic", {"latin", "cyrillic"}, "Casual comic speech bubble lettering"},
            {"Dekko", "comic", {"latin", "devanagari"}, "Clean friendly comic speech font"},
            {"Short Stack", "comic", {"latin"}, "Quirky rounded indie comic dialogue"},
            {"Boogaloo", "comic", {"latin"}, "Retro bouncy cartoon and comic lettering"},
            {"Fredoka", "comic", {"latin", "hebrew"}, "Smooth rounded modern comic dialogue"},
            {"Bubblegum Sans", "comic", {"latin"}, "Playful cartoon speech bubbles"},
            {"Luckiest Guy", "comic", {"latin"}, "Heavy bold cartoon shout and title"},
            {"Permanent Marker", "comic", {"latin"}, "Thick felt-tip marker shout and emphasis"},
            {"Chewy", "comic", {"latin"}, "Bouncy comedic dialogue and bubbles"},
            {"Sniglet", "comic", {"latin"}, "Soft rounded comedy speech bubbles"},
            {"Baloo 2", "comic", {"latin", "vietnamese"}, "Rounded friendly webtoon and comic font"},
            {"Changa One", "comic", {"latin"}, "Punchy heavy comic emphasis"},
            {"Carter One", "comic", {"latin"}, "Expressive bold comic dialogue"},
            {"Lilita One", "comic", {"latin"}, "Chunky bold comic shout and title"},
            {"Titan One", "comic", {"latin"}, "Ultra-bold impact bubble lettering"},
            {"Freckle Face", "comic", {"latin"}, "Fun whimsical indie comic font"},
            {"Coming Soon", "comic", {"latin"}, "Neat classic comic strip lettering"},
            {"Walter Turncoat", "comic", {"latin"}, "Sketchy hand-inked comic dialogue"},
            {"Schoolbell", "comic", {"latin"}, "Light casual slice-of-life comic"},
            // --- Action, Horror & SFX Display (22) ---
            {"Anton", "sfx", {"latin", "vietnamese"}, "Tall heavy shout and sound effect"},
            {"Bebas Neue", "sfx", {"latin"}, "Condensed bold narration and action SFX"},
            {"Bungee", "sfx", {"latin", "vietnamese"}, "Blocky heavy comic sound effects"},
            {"Bungee Shade", "sfx", {"latin", "vietnamese"}, "3D shadowed comic title and SFX"},
            {"Creepster", "sfx", {"latin"}, "Spooky horror manga title and SFX"},
            {"Nosifer", "sfx", {"latin"}, "Dripping horror sound effects"},
            {"Eater", "sfx", {"latin"}, "Jagged monster and thriller SFX"},
            {"Butcherman", "sfx", {"latin"}, "Gritty dark horror shout font"},
            {"Shojumaru", "sfx", {"latin"}, "Japanese brush-inspired action display"},
            {"Rampart One", "sfx", {"latin", "japanese"}, "3D outlined manga SFX and chapter title"},
            {"Black Ops One", "sfx", {"latin"}, "Stencil military and mecha SFX"},
            {"Russo One", "sfx", {"latin", "cyrillic"}, "Heavy sci-fi and battle shout"},
            {"Oswald", "sfx", {"latin", "cyrillic", "vietnamese"}, "Narrow dramatic narration and shout"},
            {"Teko", "sfx", {"latin", "devanagari"}, "Tall bold condensed action SFX"},
            {"Staatliches", "sfx", {"latin"}, "Bold graphic caption and shout"},
            {"Righteous", "sfx", {"latin"}, "Stylized sci-fi comic display"},
            {"Press Start 2P", "sfx", {"latin", "cyrillic"}, "Retro 8-bit RPG game UI box"},
            {"Silkscreen", "sfx", {"latin"}, "Crisp pixel status window for isekai manhwa"},
            {"Orbitron", "sfx", {"latin"}, "Futuristic cyberpunk HUD and system box"},
            {"Audiowide", "sfx", {"latin"}, "Mecha and sci-fi narration box"},
            {"Rubik Mono One", "sfx", {"latin", "cyrillic"}, "Ultra-heavy impact BOOM sound effect"},
            {"Sigmar One", "sfx", {"latin", "vietnamese"}, "Heavy rounded cartoon impact SFX"},
            // --- Handwriting, Whispers & Margin Notes (24) ---
            {"Caveat", "handwriting", {"latin", "cyrillic"}, "Natural handwritten inner monologue"},
            {"Caveat Brush", "handwriting", {"latin"}, "Expressive brush margin notes"},
            {"Indie Flower", "handwriting", {"latin"}, "Airy slice-of-life handwritten notes"},
            {"Shadows Into Light", "handwriting", {"latin"}, "Delicate handwritten whisper text"},
            {"Shadows Into Light Two", "handwriting", {"latin"}, "Clean readable handwritten dialogue"},
            {"Amatic SC", "handwriting", {"latin", "cyrillic", "vietnamese"}, "Tall sketchy manga aside and whisper"},
            {"Annie Use Your Telescope", "handwriting", {"latin"}, "Cute tall handwritten background note"},
            {"Gloria Hallelujah", "handwriting", {"latin"}, "Expressive comic strip handwriting"},
            {"Gochi Hand", "handwriting", {"latin"}, "Casual teenager journal handwriting"},
            {"Handlee", "handwriting", {"latin"}, "Neat rounded handwritten notes"},
            {"Just Another Hand", "handwriting", {"latin"}, "Narrow margin scribbles and whispers"},
            {"Klee One", "handwriting", {"japanese", "latin"}, "Neat textbook script supporting Japanese"},
            {"Nanum Pen Script", "handwriting", {"korean", "latin"}, "Expressive pen script supporting Korean"},
            {"Nanum Brush Script", "handwriting", {"korean", "latin"}, "Dynamic brush script supporting Korean"},
            {"Reenie Beanie", "handwriting", {"latin"}, "Soft handwritten diary and whisper"},
            {"Rock Salt", "handwriting", {"latin"}, "Rough gritty scratchy scrawl"},
            {"Covered By Your Grace", "handwriting", {"latin"}, "Casual felt-tip pen side note"},
            {"Homemade Apple", "handwriting", {"latin"}, "Flowing cursive letter and flashback note"},
            {"Nothing You Could Do", "handwriting", {"latin"}, "Light sketchy margin commentary"},
            {"Sue Ellen Francisco", "handwriting", {"latin"}, "Tall whimsical handwritten aside"},
            {"Pangolin", "handwriting", {"latin", "cyrillic", "vietnamese"}, "Comic handwritten hybrid with Cyrillic"},
            {"Marck Script", "handwriting", {"latin", "cyrillic"}, "Flowing handwritten script with Cyrillic"},
            {"Mali", "handwriting", {"thai", "latin", "vietnamese"}, "Playful handwritten comic font with Thai"},
            {"Sriracha", "handwriting", {"thai", "latin", "vietnamese"}, "Expressive comic script with Thai"},
            // --- Japanese Manga CJK (18) ---
            {"Noto Sans JP", "cjk_ja", {"japanese", "latin"}, "Clean universal Japanese manga gothic"},
            {"Noto Serif JP", "cjk_ja", {"japanese", "latin"}, "Dramatic narration and horror Japanese mincho"},
            {"Kosugi Maru", "cjk_ja", {"japanese", "latin"}, "Classic rounded Japanese manga bubble font"},
            {"Kosugi", "cjk_ja", {"japanese", "latin"}, "Clean Japanese gothic speech lettering"},
            {"M PLUS Rounded 1c", "cjk_ja", {"japanese", "latin"}, "Modern rounded Japanese dialogue font"},
            {"M PLUS 1p", "cjk_ja", {"japanese", "latin"}, "Crisp modern Japanese comic sans"},
            {"Zen Maru Gothic", "cjk_ja", {"japanese", "latin"}, "Soft friendly rounded Japanese bubbles"},
            {"Zen Kaku Gothic New", "cjk_ja", {"japanese", "latin"}, "Modern clean manga speech gothic"},
            {"Zen Kurenaido", "cjk_ja", {"japanese", "latin"}, "Handwritten Japanese manga margin notes"},
            {"Yomogi", "cjk_ja", {"japanese", "latin"}, "Cute handwritten Japanese background text"},
            {"Yusei Magic", "cjk_ja", {"japanese", "latin"}, "Bold marker-style Japanese manga font"},
            {"Hachi Maru Pop", "cjk_ja", {"japanese", "latin"}, "Playful bubbly pop Japanese lettering"},
            {"Dela Gothic One", "cjk_ja", {"japanese", "latin"}, "Ultra-heavy Japanese manga SFX and shout"},
            {"Reggae One", "cjk_ja", {"japanese", "latin"}, "Spiky shonen action Japanese SFX"},
            {"RocknRoll One", "cjk_ja", {"japanese", "latin"}, "Energetic pop action Japanese display"},
            {"DotGothic16", "cjk_ja", {"japanese", "latin"}, "Pixel RPG status window Japanese font"},
            {"Shippori Mincho", "cjk_ja", {"japanese", "latin"}, "Literary classic Japanese narration mincho"},
            {"Kaisei Opti", "cjk_ja", {"japanese", "latin"}, "Stylized Japanese book and narration serif"},
            // --- Korean Manhwa CJK (17) ---
            {"Noto Sans KR", "cjk_ko", {"korean", "latin"}, "Clean universal Korean webtoon gothic"},
            {"Noto Serif KR", "cjk_ko", {"korean", "latin"}, "Dramatic narration Korean serif"},
            {"Jua", "cjk_ko", {"korean", "latin"}, "Playful rounded Korean comic dialogue"},
            {"Do Hyeon", "cjk_ko", {"korean", "latin"}, "Bold impactful Korean manhwa shout"},
            {"Black Han Sans", "cjk_ko", {"korean", "latin"}, "Ultra-heavy Korean battle SFX"},
            {"Gaegu", "cjk_ko", {"korean", "latin"}, "Cute handwritten Korean manhwa notes"},
            {"Hi Melody", "cjk_ko", {"korean", "latin"}, "Casual handwritten Korean speech"},
            {"Gamja Flower", "cjk_ko", {"korean", "latin"}, "Playful quirky Korean comic lettering"},
            {"Poor Story", "cjk_ko", {"korean", "latin"}, "Charming indie Korean webtoon font"},
            {"Cute Font", "cjk_ko", {"korean", "latin"}, "Tall cute Korean comic dialogue"},
            {"Yeon Sung", "cjk_ko", {"korean", "latin"}, "Traditional Korean brush-style dialogue"},
            {"Stylish", "cjk_ko", {"korean", "latin"}, "Clean expressive Korean webtoon lettering"},
            {"Song Myung", "cjk_ko", {"korean", "latin"}, "Classic literary Korean narration serif"},
            {"Gowun Dodum", "cjk_ko", {"korean", "latin"}, "Soft readable Korean webtoon sans"},
            {"Gowun Batang", "cjk_ko", {"korean", "latin"}, "Warm literary Korean narration"},
            {"Dongle", "cjk_ko", {"korean", "latin"}, "Soft rounded bubbly Korean dialogue"},
            {"Single Day", "cjk_ko", {"korean", "latin"}, "Expressive hand-drawn Korean lettering"},
            // --- Chinese Manhua CJK (11) ---
            {"Noto Sans SC", "cjk_zh", {"chinese-simplified", "latin"}, "Clean Simplified Chinese manhua sans"},
            {"Noto Sans TC", "cjk_zh", {"chinese-traditional", "latin"}, "Clean Traditional Chinese manhua sans"},
            {"Noto Serif SC", "cjk_zh", {"chinese-simplified", "latin"}, "Classic Simplified Chinese narration"},
            {"Noto Serif TC", "cjk_zh", {"chinese-traditional", "latin"}, "Classic Traditional Chinese narration"},
            {"ZCOOL KuaiLe", "cjk_zh", {"chinese-simplified", "latin"}, "Fun rounded comic Simplified Chinese"},
            {"ZCOOL QingKe HuangYou", "cjk_zh", {"chinese-simplified", "latin"}, "Sharp modern comic Chinese display"},
            {"ZCOOL XiaoWei", "cjk_zh", {"chinese-simplified", "latin"}, "Elegant literary Chinese manhua font"},
            {"Ma Shan Zheng", "cjk_zh", {"chinese-simplified", "latin"}, "Bold martial-arts brush calligraphy"},
            {"Zhi Mang Xing", "cjk_zh", {"chinese-simplified", "latin"}, "Flowing wuxia/xianxia brush script"},
            {"Liu Jian Mao Cao", "cjk_zh", {"chinese-simplified", "latin"}, "Wild cursive wuxia action SFX"},
            {"Long Cang", "cjk_zh", {"chinese-simplified", "latin"}, "Expressive Chinese handwritten notes"},
            // --- Clean Webtoon, Narration & Multilingual (12) ---
            {"Inter", "webtoon", {"latin", "cyrillic", "vietnamese"}, "Clean modern webtoon dialogue sans"},
            {"Poppins", "webtoon", {"latin", "devanagari"}, "Geometric clean comic speech sans"},
            {"Nunito", "webtoon", {"latin", "cyrillic", "vietnamese"}, "Friendly rounded webtoon lettering"},
            {"Quicksand", "webtoon", {"latin", "vietnamese"}, "Light airy geometric dialogue"},
            {"Rubik", "webtoon", {"latin", "cyrillic", "arabic", "hebrew"}, "Versatile rounded multilingual sans"},
            {"Plus Jakarta Sans", "webtoon", {"latin", "vietnamese"}, "Crisp Indonesian and SEA webtoon sans"},
            {"Lexend", "webtoon", {"latin", "vietnamese"}, "High-readability speech bubble sans"},
            {"Merriweather", "webtoon", {"latin", "cyrillic", "vietnamese"}, "Classic comic narration box serif"},
            {"Playfair Display", "webtoon", {"latin", "cyrillic", "vietnamese"}, "Elegant fantasy and romance narration"},
            {"Cairo", "webtoon", {"arabic", "latin"}, "Clean modern Arabic comic lettering"},
            {"Tajawal", "webtoon", {"arabic", "latin"}, "Friendly Arabic speech bubble font"},
            {"Kanit", "webtoon", {"thai", "latin", "vietnamese"}, "Clean modern Thai and Latin webtoon sans"},
        };
        std::vector<CatalogFontSpec> out;
        out.reserve(raw.size());
        for (const auto& item : raw) {
            out.push_back({
                slugify_font_id(item.family),
                item.family,
                item.cat,
                item.scripts,
                item.desc,
            });
        }
        return out;
    }();
    return kCatalog;
}

const CatalogFontSpec* find_catalog_spec(const std::string& query_lower) {
    for (const auto& spec : curated_google_fonts()) {
        if (spec.id == query_lower || util::to_lower_ascii(spec.family) == query_lower) {
            return &spec;
        }
    }
    return nullptr;
}

bool is_supported_font_extension(const std::string& ext_lower) {
    return ext_lower == ".ttf" || ext_lower == ".otf" ||
           ext_lower == ".woff2" || ext_lower == ".woff";
}

std::string format_for_extension(const std::string& ext_lower) {
    if (ext_lower == ".ttf") return "ttf";
    if (ext_lower == ".otf") return "otf";
    if (ext_lower == ".woff2") return "woff2";
    if (ext_lower == ".woff") return "woff";
    return {};
}

std::string media_type_for_font_ext(const std::string& ext_lower) {
    if (ext_lower == ".ttf") return "font/ttf";
    if (ext_lower == ".otf") return "font/otf";
    if (ext_lower == ".woff2") return "font/woff2";
    if (ext_lower == ".woff") return "font/woff";
    return "application/octet-stream";
}

std::string build_font_url(const std::string& filename) {
    return "/api/fonts/file/" + util::url_encode_component(filename);
}

int category_rank(const std::string& category) {
    if (category == "custom") return 0;
    if (category == "downloaded") return 1;
    return 2;
}

bool has_sfnt_magic(const uint8_t* hdr) {
    const bool ttf = (hdr[0] == 0x00 && hdr[1] == 0x01 && hdr[2] == 0x00 && hdr[3] == 0x00);
    const bool otf = (std::memcmp(hdr, "OTTO", 4) == 0);
    const bool apple = (std::memcmp(hdr, "true", 4) == 0);
    const bool typ1 = (std::memcmp(hdr, "typ1", 4) == 0);
    return ttf || otf || apple || typ1;
}

void validate_font_bytes(const std::vector<uint8_t>& raw_bytes, const std::string& ext_lower) {
    if (!is_supported_font_extension(ext_lower)) {
        throw FontValidationError(
            "Unsupported font extension '" + ext_lower + "'. Supported formats: .ttf, .otf, .woff2, .woff.");
    }
    if (raw_bytes.size() < kMinFontFileBytes) {
        throw FontValidationError("Font file is empty or truncated.");
    }
    if (raw_bytes.size() > kMaxFontFileBytes) {
        throw FontValidationError("Font file exceeds maximum allowed size (50 MB).");
    }
    const uint8_t* hdr = raw_bytes.data();
    if (ext_lower == ".ttf" || ext_lower == ".otf") {
        if (!has_sfnt_magic(hdr)) {
            throw FontValidationError("Invalid TrueType/OpenType font header for '" + ext_lower + "' file.");
        }
    } else if (ext_lower == ".woff") {
        if (std::memcmp(hdr, "wOFF", 4) != 0) {
            throw FontValidationError("Invalid WOFF font header.");
        }
    } else if (ext_lower == ".woff2") {
        if (std::memcmp(hdr, "wOF2", 4) != 0) {
            throw FontValidationError("Invalid WOFF2 font header.");
        }
    }
}

std::string sanitize_font_filename(const std::string& filename) {
    std::string norm = filename;
    std::replace(norm.begin(), norm.end(), '\\', '/');
    const auto slash = norm.rfind('/');
    const std::string base = util::trim_ascii(
        slash == std::string::npos ? norm : norm.substr(slash + 1));
    if (base.empty()) {
        throw FontValidationError("Font filename cannot be empty.");
    }

    const fs::path p = fs::path(util::utf8_to_wide(base));
    const std::string ext_lower = util::to_lower_ascii(util::path_to_utf8(p.extension()));
    if (!is_supported_font_extension(ext_lower)) {
        throw FontValidationError(
            "Unsupported font file format '" + ext_lower + "'. Upload a .ttf, .otf, .woff2, or .woff file.");
    }

    const std::string raw_stem = util::path_to_utf8(p.stem());
    std::string clean_stem;
    clean_stem.reserve(raw_stem.size());
    for (unsigned char c : raw_stem) {
        if (std::isalnum(c) || c == '.' || c == '_' || c == ' ' || c == '-') {
            clean_stem.push_back(static_cast<char>(c));
        } else {
            clean_stem.push_back('_');
        }
    }
    while (!clean_stem.empty() &&
           (clean_stem.front() == ' ' || clean_stem.front() == '.' || clean_stem.front() == '_')) {
        clean_stem.erase(clean_stem.begin());
    }
    while (!clean_stem.empty() &&
           (clean_stem.back() == ' ' || clean_stem.back() == '.' || clean_stem.back() == '_')) {
        clean_stem.pop_back();
    }
    if (clean_stem.empty()) {
        clean_stem = "custom_font";
    }
    return clean_stem + ext_lower;
}

uint16_t read_u16_be(const uint8_t* buf, size_t offset) {
    return static_cast<uint16_t>((static_cast<uint16_t>(buf[offset]) << 8) | buf[offset + 1]);
}

uint32_t read_u32_be(const uint8_t* buf, size_t offset) {
    return (static_cast<uint32_t>(buf[offset]) << 24) |
           (static_cast<uint32_t>(buf[offset + 1]) << 16) |
           (static_cast<uint32_t>(buf[offset + 2]) << 8) |
           static_cast<uint32_t>(buf[offset + 3]);
}

std::string decode_sfnt_name_bytes(
    const uint8_t* data,
    size_t len,
    uint16_t platform_id) {
    if (!data || len == 0) {
        return {};
    }
    if (platform_id == 3 || platform_id == 0) {
        // UTF-16-BE
        const size_t wchar_count = len / 2;
        std::wstring wide;
        wide.reserve(wchar_count);
        for (size_t i = 0; i < wchar_count; ++i) {
            const wchar_t ch = static_cast<wchar_t>(read_u16_be(data, i * 2));
            if (ch != L'\0') {
                wide.push_back(ch);
            }
        }
        return util::trim_ascii(util::wide_to_utf8(wide));
    }

    // Platform 1 (Mac Roman) / Latin-1 fallback -> UTF-8
    std::string out;
    out.reserve(len);
    for (size_t i = 0; i < len; ++i) {
        const uint8_t b = data[i];
        if (b == 0) continue;
        if (b < 0x80) {
            out.push_back(static_cast<char>(b));
        } else {
            out.push_back(static_cast<char>(0xC0 | (b >> 6)));
            out.push_back(static_cast<char>(0x80 | (b & 0x3F)));
        }
    }
    return util::trim_ascii(out);
}

struct ParsedSfntNames {
    std::optional<std::string> family;
    std::optional<std::string> subfamily;
    std::optional<std::string> full_name;
};

ParsedSfntNames parse_sfnt_name_table(const fs::path& path) {
    ParsedSfntNames out;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        return out;
    }

    uint8_t header[12]{};
    if (!ifs.read(reinterpret_cast<char*>(header), 12)) {
        return out;
    }
    if (!has_sfnt_magic(header)) {
        return out;
    }

    const uint16_t num_tables = read_u16_be(header, 4);
    if (num_tables == 0 || num_tables > 256) {
        return out;
    }

    std::vector<uint8_t> table_dir(static_cast<size_t>(num_tables) * 16);
    if (!ifs.read(reinterpret_cast<char*>(table_dir.data()), static_cast<std::streamsize>(table_dir.size()))) {
        return out;
    }

    uint32_t name_offset = 0;
    uint32_t name_length = 0;
    for (uint16_t i = 0; i < num_tables; ++i) {
        const size_t pos = static_cast<size_t>(i) * 16;
        if (std::memcmp(table_dir.data() + pos, "name", 4) == 0) {
            name_offset = read_u32_be(table_dir.data(), pos + 8);
            name_length = read_u32_be(table_dir.data(), pos + 12);
            break;
        }
    }

    if (name_offset == 0 || name_length < 6 || name_length > 2'000'000U) {
        return out;
    }

    ifs.seekg(static_cast<std::streamoff>(name_offset), std::ios::beg);
    std::vector<uint8_t> name_data(name_length);
    if (!ifs.read(reinterpret_cast<char*>(name_data.data()), static_cast<std::streamsize>(name_length))) {
        return out;
    }

    const uint16_t count = read_u16_be(name_data.data(), 2);
    const uint16_t string_offset = read_u16_be(name_data.data(), 4);
    if (count == 0 || 6ULL + static_cast<size_t>(count) * 12ULL > name_data.size()) {
        return out;
    }

    std::unordered_map<uint16_t, std::pair<int, std::string>> best;
    for (uint16_t i = 0; i < count; ++i) {
        const size_t rec = 6 + static_cast<size_t>(i) * 12;
        const uint16_t platform_id = read_u16_be(name_data.data(), rec);
        const uint16_t language_id = read_u16_be(name_data.data(), rec + 4);
        const uint16_t name_id = read_u16_be(name_data.data(), rec + 6);
        const uint16_t length = read_u16_be(name_data.data(), rec + 8);
        const uint16_t offset = read_u16_be(name_data.data(), rec + 10);

        if (length == 0 || (name_id != 1 && name_id != 2 && name_id != 4 && name_id != 16 && name_id != 17)) {
            continue;
        }
        const size_t str_start = static_cast<size_t>(string_offset) + static_cast<size_t>(offset);
        if (str_start + length > name_data.size()) {
            continue;
        }

        const std::string decoded = decode_sfnt_name_bytes(name_data.data() + str_start, length, platform_id);
        if (decoded.empty()) {
            continue;
        }

        int priority = 40;
        if (platform_id == 3 && language_id == 0x0409) {
            priority = 100;
        } else if (platform_id == 3) {
            priority = 80;
        } else if (platform_id == 0) {
            priority = 70;
        } else if (platform_id == 1 && language_id == 0) {
            priority = 60;
        }

        const auto it = best.find(name_id);
        if (it == best.end() || priority > it->second.first) {
            best[name_id] = {priority, decoded};
        }
    }

    if (best.count(16)) out.family = best[16].second;
    else if (best.count(1)) out.family = best[1].second;

    if (best.count(17)) out.subfamily = best[17].second;
    else if (best.count(2)) out.subfamily = best[2].second;

    if (best.count(4)) out.full_name = best[4].second;
    return out;
}

std::pair<int, bool> infer_weight_and_italic(const std::string& subfamily, const std::string& stem) {
    const std::string text = util::to_lower_ascii(subfamily + " " + stem);
    const std::string stem_lower = util::to_lower_ascii(stem);

    bool is_italic = (text.find("italic") != std::string::npos ||
                      text.find("oblique") != std::string::npos ||
                      text.find("slanted") != std::string::npos);
    if (!is_italic && stem_lower.size() >= 2 && stem_lower.back() == 'i') {
        const char prev = stem_lower[stem_lower.size() - 2];
        if (prev == '-' || prev == '_' || prev == ' ') {
            is_italic = true;
        }
    }

    auto has_any = [&](std::initializer_list<const char*> toks) {
        for (const char* t : toks) {
            if (text.find(t) != std::string::npos) return true;
        }
        return false;
    };

    auto ends_with = [](const std::string& s, const char* suffix) {
        const size_t slen = std::strlen(suffix);
        return s.size() >= slen && s.compare(s.size() - slen, slen, suffix) == 0;
    };

    int weight = 400;
    if (has_any({"black", "heavy", "ultra", "extrablack"})) {
        weight = 900;
    } else if (has_any({"extrabold", "extra bold", "ultrabold"})) {
        weight = 800;
    } else if (has_any({"semibold", "semi bold", "demibold", "demi bold"})) {
        weight = 600;
    } else if (text.find("bold") != std::string::npos ||
               ends_with(stem_lower, "bd") ||
               ends_with(stem_lower, "bdi") ||
               ends_with(stem_lower, "bdit")) {
        weight = 700;
    } else if (text.find("medium") != std::string::npos) {
        weight = 500;
    } else if (has_any({"extralight", "extra light", "ultralight"})) {
        weight = 200;
    } else if (has_any({"thin", "hairline"})) {
        weight = 100;
    } else if (text.find("light") != std::string::npos) {
        weight = 300;
    }
    return {weight, is_italic};
}

// -----------------------------------------------------------------------------
// Native Windows WinHTTP HTTPS Client (zero OpenSSL dependency)
// -----------------------------------------------------------------------------

struct WinHttpHandle {
    HINTERNET h = nullptr;
    WinHttpHandle() = default;
    explicit WinHttpHandle(HINTERNET handle) : h(handle) {}
    ~WinHttpHandle() {
        if (h) WinHttpCloseHandle(h);
    }
    WinHttpHandle(const WinHttpHandle&) = delete;
    WinHttpHandle& operator=(const WinHttpHandle&) = delete;
    explicit operator bool() const noexcept { return h != nullptr; }
};

struct HttpDownloadResult {
    int status_code = 0;
    std::vector<uint8_t> body;
    std::string error;
};

HttpDownloadResult winhttp_get(
    const std::string& url,
    const std::wstring& user_agent,
    int timeout_ms = 20000) {
    HttpDownloadResult result;
    const std::wstring wurl = util::utf8_to_wide(url);

    URL_COMPONENTS uc{};
    uc.dwStructSize = sizeof(uc);
    wchar_t host_buf[256]{};
    wchar_t path_buf[2048]{};
    uc.lpszHostName = host_buf;
    uc.dwHostNameLength = static_cast<DWORD>(std::size(host_buf));
    uc.lpszUrlPath = path_buf;
    uc.dwUrlPathLength = static_cast<DWORD>(std::size(path_buf));

    if (!WinHttpCrackUrl(wurl.c_str(), static_cast<DWORD>(wurl.size()), 0, &uc)) {
        result.error = "Invalid URL: " + url;
        return result;
    }

    WinHttpHandle session(WinHttpOpen(
        user_agent.c_str(),
        WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
        WINHTTP_NO_PROXY_NAME,
        WINHTTP_NO_PROXY_BYPASS,
        0));
    if (!session) {
        result.error = "WinHttpOpen failed (error " + std::to_string(GetLastError()) + ")";
        return result;
    }

    WinHttpSetTimeouts(session.h, timeout_ms, timeout_ms, timeout_ms, timeout_ms);

    const std::wstring host(uc.lpszHostName, uc.dwHostNameLength);
    WinHttpHandle connect(WinHttpConnect(session.h, host.c_str(), uc.nPort, 0));
    if (!connect) {
        result.error = "WinHttpConnect failed for host (error " + std::to_string(GetLastError()) + ")";
        return result;
    }

    std::wstring full_path(uc.lpszUrlPath, uc.dwUrlPathLength);
    if (uc.dwExtraInfoLength > 0 && uc.lpszExtraInfo) {
        full_path.append(uc.lpszExtraInfo, uc.dwExtraInfoLength);
    }

    const DWORD flags = (uc.nScheme == INTERNET_SCHEME_HTTPS) ? WINHTTP_FLAG_SECURE : 0;
    WinHttpHandle request(WinHttpOpenRequest(
        connect.h, L"GET", full_path.c_str(), nullptr,
        WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, flags));
    if (!request) {
        result.error = "WinHttpOpenRequest failed (error " + std::to_string(GetLastError()) + ")";
        return result;
    }

    DWORD redirect_policy = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
    WinHttpSetOption(request.h, WINHTTP_OPTION_REDIRECT_POLICY, &redirect_policy, sizeof(redirect_policy));

    if (!WinHttpSendRequest(request.h, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
        !WinHttpReceiveResponse(request.h, nullptr)) {
        result.error = "WinHTTP network request failed (error " + std::to_string(GetLastError()) + ")";
        return result;
    }

    DWORD status = 0;
    DWORD status_sz = sizeof(status);
    WinHttpQueryHeaders(
        request.h,
        WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
        WINHTTP_HEADER_NAME_BY_INDEX,
        &status,
        &status_sz,
        WINHTTP_NO_HEADER_INDEX);
    result.status_code = static_cast<int>(status);

    while (true) {
        DWORD avail = 0;
        if (!WinHttpQueryDataAvailable(request.h, &avail)) {
            result.error = "WinHttpQueryDataAvailable failed (error " + std::to_string(GetLastError()) + ")";
            return result;
        }
        if (avail == 0) {
            break;
        }
        if (result.body.size() + avail > kMaxFontFileBytes) {
            result.error = "Downloaded response exceeded 50 MB limit.";
            return result;
        }
        const size_t prev_size = result.body.size();
        result.body.resize(prev_size + avail);
        DWORD read_bytes = 0;
        if (!WinHttpReadData(request.h, result.body.data() + prev_size, avail, &read_bytes)) {
            result.error = "WinHttpReadData failed (error " + std::to_string(GetLastError()) + ")";
            return result;
        }
        result.body.resize(prev_size + read_bytes);
    }

    return result;
}

std::string resolve_google_font_binary_url(
    const std::string& family,
    const std::optional<std::string>& weight) {
    std::vector<std::string> queries;
    if (weight.has_value() && !util::trim_ascii(*weight).empty()) {
        queries.push_back(family + ":wght@" + util::trim_ascii(*weight));
    }
    queries.push_back(family);
    queries.push_back(family + ":wght@400");
    queries.push_back(family + ":wght@500");
    queries.push_back(family + ":wght@700");

    std::unordered_set<std::string> tried;
    std::string last_error = "Font '" + family + "' not found on Google Fonts.";
    static const std::regex kUrlRe(R"(url\((https://[^)'"]+)\))", std::regex::icase);

    for (const auto& q : queries) {
        if (!tried.insert(q).second) continue;

        const std::string css_url =
            "https://fonts.googleapis.com/css2?family=" + util::url_encode_component(q);
        // Mozilla/4.0 instructs Google Fonts CSS2 to serve direct unsubsetted .ttf/.otf files.
        const auto resp = winhttp_get(css_url, L"Mozilla/4.0", 15000);
        if (!resp.error.empty()) {
            throw std::runtime_error("Network error contacting Google Fonts for '" + family + "': " + resp.error);
        }
        if (resp.status_code != 200) {
            last_error = "Google Fonts returned HTTP " + std::to_string(resp.status_code) + " for '" + q + "'.";
            continue;
        }

        const std::string css_text(resp.body.begin(), resp.body.end());
        std::vector<std::string> urls;
        for (auto it = std::sregex_iterator(css_text.begin(), css_text.end(), kUrlRe);
             it != std::sregex_iterator(); ++it) {
            urls.push_back((*it)[1].str());
        }
        if (urls.empty()) {
            last_error = "Google Fonts CSS for '" + q + "' did not contain a downloadable font URL.";
            continue;
        }

        for (const auto& cand : urls) {
            const std::string lower = util::to_lower_ascii(cand);
            if ((lower.size() >= 4 && lower.compare(lower.size() - 4, 4, ".ttf") == 0) ||
                (lower.size() >= 4 && lower.compare(lower.size() - 4, 4, ".otf") == 0)) {
                return cand;
            }
        }
        return urls.front();
    }

    throw std::runtime_error(last_error);
}

}  // namespace

fs::path FontManager::default_fonts_dir() {
    if (auto env_dir = util::get_env_wide(L"MANGA_FONTS_DIR")) {
        return fs::path(*env_dir);
    }
    return util::get_comic_translator_app_dir() / L"fonts";
}

std::vector<fs::path> FontManager::windows_system_font_dirs() {
    std::vector<fs::path> candidates;
    const std::wstring windir = util::get_env_wide(L"WINDIR").value_or(L"C:\\Windows");
    candidates.push_back(fs::path(windir) / L"Fonts");
    candidates.push_back(util::get_local_app_data_dir() / L"Microsoft" / L"Windows" / L"Fonts");

    std::vector<fs::path> existing;
    std::unordered_set<std::string> seen;
    for (const auto& dir : candidates) {
        std::error_code ec;
        if (!fs::is_directory(dir, ec) || ec) continue;
        fs::path resolved = fs::weakly_canonical(dir, ec);
        if (ec) resolved = dir;
        const std::string key = util::to_lower_ascii(util::path_to_utf8(resolved));
        if (seen.insert(key).second) {
            existing.push_back(resolved);
        }
    }
    return existing;
}

FontManager::FontManager(std::optional<fs::path> custom_fonts_dir) {
    fs::path base = custom_fonts_dir.has_value() ? *custom_fonts_dir : default_fonts_dir();
    std::error_code ec;
    fs::path canon = fs::weakly_canonical(base, ec);
    fonts_dir_ = ec ? base : canon;
    downloaded_dir_ = fonts_dir_ / L"downloaded";
    custom_subdir_ = fonts_dir_ / L"custom";
    manifest_path_ = downloaded_dir_ / L"manifest.json";
}

void FontManager::ensure_dirs() const {
    std::error_code ec;
    fs::create_directories(fonts_dir_, ec);
    fs::create_directories(downloaded_dir_, ec);
}

json FontManager::load_downloaded_manifest() const {
    std::error_code ec;
    if (!fs::is_regular_file(manifest_path_, ec) || ec) {
        return json::object();
    }
    const auto bytes = util::read_file_bytes(manifest_path_);
    if (!bytes.has_value()) {
        return json::object();
    }
    const json parsed = json::parse(bytes->begin(), bytes->end(), nullptr, false);
    return parsed.is_object() ? parsed : json::object();
}

void FontManager::save_downloaded_manifest(const json& manifest) const {
    ensure_dirs();
    const std::string serialized = manifest.dump(2) + "\n";
    std::string err;
    if (!util::atomic_write_bytes(manifest_path_, serialized.data(), serialized.size(), &err)) {
        throw std::runtime_error("Failed to persist font manifest: " + err);
    }
}

std::optional<FontManager::IndexedFont> FontManager::build_indexed_font(
    const fs::path& path,
    const std::string& category,
    const std::optional<std::string>& override_family) const {
    const std::string ext_lower = util::to_lower_ascii(util::path_to_utf8(path.extension()));
    const std::string fmt = format_for_extension(ext_lower);
    if (fmt.empty()) {
        return std::nullopt;
    }

    std::error_code ec;
    const auto fsize = fs::file_size(path, ec);
    if (ec || fsize < kMinFontFileBytes) {
        return std::nullopt;
    }

    const ParsedSfntNames sfnt =
        (ext_lower == ".ttf" || ext_lower == ".otf") ? parse_sfnt_name_table(path) : ParsedSfntNames{};

    const std::string filename = util::path_to_utf8(path.filename());
    const std::string stem = util::path_to_utf8(path.stem());

    const bool has_override = override_family.has_value() && !util::trim_ascii(*override_family).empty();
    std::string family;
    if (has_override) {
        family = util::trim_ascii(*override_family);
    } else if (sfnt.family.has_value() && !sfnt.family->empty()) {
        family = *sfnt.family;
    } else {
        std::string pretty = stem;
        for (char& ch : pretty) {
            if (ch == '_' || ch == '-') ch = ' ';
        }
        family = util::trim_ascii(pretty);
        if (family.empty()) family = stem;
    }

    std::string subfamily =
        (sfnt.subfamily.has_value() && !sfnt.subfamily->empty()) ? *sfnt.subfamily : "Regular";
    const std::string fam_check = util::to_lower_ascii(family);
    if (fam_check.rfind("cc wild words", 0) == 0 || fam_check.rfind("ccwildwords", 0) == 0) {
        family = "CC Wild Words";
    }
    std::string full_name;
    if (!has_override && sfnt.full_name.has_value() && !sfnt.full_name->empty()) {
        full_name = *sfnt.full_name;
    } else {
        full_name = (util::to_lower_ascii(subfamily) == "regular") ? family : (family + " " + subfamily);
    }

    const auto [weight, is_italic] = infer_weight_and_italic(subfamily, stem);

    IndexedFont idx;
    idx.path = path;
    idx.category = category;
    idx.family_lower = util::to_lower_ascii(family);
    idx.full_name_lower = util::to_lower_ascii(full_name);
    idx.filename_lower = util::to_lower_ascii(filename);
    idx.stem_lower = util::to_lower_ascii(stem);
    idx.weight = weight;
    idx.italic = is_italic;
    idx.entry = {
        {"id", category + ":" + filename},
        {"family", family},
        {"subfamily", subfamily},
        {"full_name", full_name},
        {"filename", filename},
        {"category", category},
        {"format", fmt},
        {"weight", weight},
        {"italic", is_italic},
        {"size_bytes", static_cast<uint64_t>(fsize)},
        {"url", build_font_url(filename)},
    };
    return idx;
}

std::vector<FontManager::IndexedFont> FontManager::scan_local_fonts() const {
    std::vector<IndexedFont> entries;
    std::unordered_set<std::string> seen_names;

    auto scan_dir = [&](const fs::path& dir, const std::string& category, const json* manifest) {
        std::error_code ec;
        if (!fs::is_directory(dir, ec) || ec) return;

        std::vector<fs::path> files;
        for (const auto& dirent : fs::directory_iterator(dir, ec)) {
            if (ec) break;
            std::error_code f_ec;
            if (!dirent.is_regular_file(f_ec) || f_ec) continue;
            const std::string ext = util::to_lower_ascii(util::path_to_utf8(dirent.path().extension()));
            if (!is_supported_font_extension(ext)) continue;
            files.push_back(dirent.path());
        }
        std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
            return util::to_lower_ascii(util::path_to_utf8(a.filename())) <
                   util::to_lower_ascii(util::path_to_utf8(b.filename()));
        });

        for (const auto& p : files) {
            const std::string fname = util::path_to_utf8(p.filename());
            const std::string name_key = util::to_lower_ascii(fname);
            if (seen_names.count(name_key)) continue;

            std::optional<std::string> override_fam;
            if (manifest && manifest->contains(fname) && (*manifest)[fname].is_object()) {
                const std::string f = (*manifest)[fname].value("family", std::string(""));
                if (!f.empty()) override_fam = f;
            }

            if (auto item = build_indexed_font(p, category, override_fam)) {
                seen_names.insert(name_key);
                entries.push_back(std::move(*item));
            }
        }
    };

    scan_dir(fonts_dir_, "custom", nullptr);
    scan_dir(custom_subdir_, "custom", nullptr);

    // Also scan immediate subdirectories inside `./fonts/` (e.g. `./fonts/CC Wild Words/`)
    std::error_code sub_ec;
    if (fs::is_directory(fonts_dir_, sub_ec) && !sub_ec) {
        const std::string dl_name = util::to_lower_ascii(util::path_to_utf8(downloaded_dir_.filename()));
        const std::string cust_name = util::to_lower_ascii(util::path_to_utf8(custom_subdir_.filename()));
        for (const auto& dirent : fs::directory_iterator(fonts_dir_, sub_ec)) {
            if (sub_ec) break;
            std::error_code d_ec;
            if (!dirent.is_directory(d_ec) || d_ec) continue;
            const std::string sub_name = util::to_lower_ascii(util::path_to_utf8(dirent.path().filename()));
            if (sub_name == dl_name || sub_name == cust_name) continue;
            scan_dir(dirent.path(), "custom", nullptr);
        }
    }

    const json manifest = load_downloaded_manifest();
    scan_dir(downloaded_dir_, "downloaded", &manifest);
    return entries;
}

std::vector<FontManager::IndexedFont> FontManager::scan_system_fonts(bool force_refresh) {
    const auto sys_dirs = windows_system_font_dirs();

    std::vector<SystemDirFingerprint> fingerprint;
    fingerprint.reserve(sys_dirs.size());
    for (const auto& dir : sys_dirs) {
        std::error_code ec;
        const auto mtime = fs::last_write_time(dir, ec);
        size_t count = 0;
        for (auto it = fs::directory_iterator(dir, ec); !ec && it != fs::directory_iterator(); ++it) {
            ++count;
        }
        fingerprint.push_back({
            util::to_lower_ascii(util::path_to_utf8(dir)),
            ec ? 0LL : static_cast<int64_t>(mtime.time_since_epoch().count()),
            count,
        });
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!force_refresh && !system_cache_.empty() && system_cache_key_ == fingerprint) {
            return system_cache_;
        }
    }

    std::vector<IndexedFont> entries;
    std::unordered_set<std::string> seen_names;

    // Scan per-user Windows fonts first, then C:\Windows\Fonts
    std::vector<fs::path> ordered_dirs = sys_dirs;
    if (ordered_dirs.size() > 1) {
        std::reverse(ordered_dirs.begin(), ordered_dirs.end());
    }

    for (const auto& dir : ordered_dirs) {
        std::error_code ec;
        std::vector<fs::path> files;
        for (const auto& dirent : fs::directory_iterator(dir, ec)) {
            if (ec) break;
            std::error_code f_ec;
            if (!dirent.is_regular_file(f_ec) || f_ec) continue;
            const std::string ext = util::to_lower_ascii(util::path_to_utf8(dirent.path().extension()));
            if (ext != ".ttf" && ext != ".otf") continue;
            files.push_back(dirent.path());
        }
        std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
            return util::to_lower_ascii(util::path_to_utf8(a.filename())) <
                   util::to_lower_ascii(util::path_to_utf8(b.filename()));
        });

        for (const auto& p : files) {
            const std::string key = util::to_lower_ascii(util::path_to_utf8(p.filename()));
            if (seen_names.count(key)) continue;
            if (auto item = build_indexed_font(p, "system")) {
                seen_names.insert(key);
                entries.push_back(std::move(*item));
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        system_cache_key_ = std::move(fingerprint);
        system_cache_ = entries;
    }
    return entries;
}

std::vector<FontManager::IndexedFont> FontManager::collect_all_installed(bool force_refresh) {
    auto combined = scan_local_fonts();
    std::unordered_set<std::string> seen;
    for (const auto& item : combined) {
        seen.insert(item.filename_lower);
    }

    auto sys_entries = scan_system_fonts(force_refresh);
    for (auto& item : sys_entries) {
        if (seen.insert(item.filename_lower).second) {
            combined.push_back(std::move(item));
        }
    }

    std::sort(combined.begin(), combined.end(), [](const IndexedFont& a, const IndexedFont& b) {
        const int ra = category_rank(a.category);
        const int rb = category_rank(b.category);
        if (ra != rb) return ra < rb;
        if (a.family_lower != b.family_lower) return a.family_lower < b.family_lower;
        if (a.italic != b.italic) return !a.italic && b.italic;
        const int wa = std::abs(a.weight - 400);
        const int wb = std::abs(b.weight - 400);
        if (wa != wb) return wa < wb;
        return a.filename_lower < b.filename_lower;
    });
    return combined;
}

json FontManager::group_by_family(const std::vector<IndexedFont>& entries) const {
    std::unordered_map<std::string, std::vector<const IndexedFont*>> grouped;
    for (const auto& item : entries) {
        grouped[item.family_lower].push_back(&item);
    }

    struct GroupSort {
        int rank;
        std::string family_lower;
        json obj;
    };
    std::vector<GroupSort> groups;
    groups.reserve(grouped.size());

    for (auto& [fam_key, items] : grouped) {
        std::sort(items.begin(), items.end(), [](const IndexedFont* a, const IndexedFont* b) {
            const int ra = category_rank(a->category);
            const int rb = category_rank(b->category);
            if (ra != rb) return ra < rb;
            if (a->italic != b->italic) return !a->italic && b->italic;
            const int wa = std::abs(a->weight - 400);
            const int wb = std::abs(b->weight - 400);
            if (wa != wb) return wa < wb;
            return a->filename_lower < b->filename_lower;
        });

        const IndexedFont* primary = items.front();
        json variants = json::array();
        for (const IndexedFont* v : items) {
            variants.push_back({
                {"subfamily", v->entry["subfamily"]},
                {"filename", v->entry["filename"]},
                {"format", v->entry["format"]},
                {"weight", v->entry["weight"]},
                {"italic", v->entry["italic"]},
                {"url", v->entry["url"]},
            });
        }
        groups.push_back({
            category_rank(primary->category),
            primary->family_lower,
            {
                {"family", primary->entry["family"]},
                {"category", primary->entry["category"]},
                {"default_file", primary->entry["filename"]},
                {"url", primary->entry["url"]},
                {"variants", std::move(variants)},
            },
        });
    }

    std::sort(groups.begin(), groups.end(), [](const GroupSort& a, const GroupSort& b) {
        if (a.rank != b.rank) return a.rank < b.rank;
        return a.family_lower < b.family_lower;
    });

    json out = json::array();
    for (auto& g : groups) {
        out.push_back(std::move(g.obj));
    }
    return out;
}

json FontManager::list_fonts(
    const std::optional<std::string>& category,
    bool include_catalog,
    bool force_refresh) {
    const auto all_items = collect_all_installed(force_refresh);

    size_t system_count = 0;
    size_t custom_count = 0;
    size_t downloaded_count = 0;
    for (const auto& item : all_items) {
        if (item.category == "system") ++system_count;
        else if (item.category == "custom") ++custom_count;
        else if (item.category == "downloaded") ++downloaded_count;
    }

    std::vector<IndexedFont> filtered;
    filtered.reserve(all_items.size());
    for (const auto& item : all_items) {
        if (!category.has_value() || item.category == *category) {
            filtered.push_back(item);
        }
    }

    json fonts_json = json::array();
    for (const auto& item : filtered) {
        fonts_json.push_back(item.entry);
    }

    json families_json = group_by_family(filtered);

    std::unordered_map<std::string, const IndexedFont*> installed_by_family;
    for (const auto& item : all_items) {
        if (!installed_by_family.count(item.family_lower)) {
            installed_by_family[item.family_lower] = &item;
        }
    }

    json catalog_json = json::array();
    const auto& catalog = curated_google_fonts();
    if (include_catalog) {
        for (const auto& spec : catalog) {
            const auto it = installed_by_family.find(util::to_lower_ascii(spec.family));
            const bool installed = (it != installed_by_family.end());
            catalog_json.push_back({
                {"id", spec.id},
                {"family", spec.family},
                {"style_category", spec.style_category},
                {"scripts", spec.scripts},
                {"description", spec.description},
                {"installed", installed},
                {"filename", installed ? it->second->entry["filename"] : json(nullptr)},
                {"url", installed ? it->second->entry["url"] : json(nullptr)},
            });
        }
    }

    json sys_dirs_json = json::array();
    for (const auto& d : windows_system_font_dirs()) {
        sys_dirs_json.push_back(util::path_to_utf8(d));
    }

    return {
        {"fonts_dir", util::path_to_utf8(fonts_dir_)},
        {"system_font_dirs", std::move(sys_dirs_json)},
        {"counts", {
            {"total", filtered.size()},
            {"families", families_json.size()},
            {"system", system_count},
            {"custom", custom_count},
            {"downloaded", downloaded_count},
            {"catalog", catalog.size()},
        }},
        {"fonts", std::move(fonts_json)},
        {"families", std::move(families_json)},
        {"catalog", std::move(catalog_json)},
    };
}

ResolvedFontFile FontManager::resolve_font_file(const std::string& font_name) {
    const std::string cleaned = util::trim_ascii(util::url_decode_component(font_name));
    if (cleaned.empty() ||
        cleaned.find('/') != std::string::npos ||
        cleaned.find('\\') != std::string::npos ||
        cleaned.find("..") != std::string::npos ||
        cleaned.find('\0') != std::string::npos) {
        throw FontValidationError("Invalid font identifier: '" + font_name + "'");
    }

    std::vector<fs::path> allowed_roots;
    auto add_root = [&](const fs::path& p) {
        std::error_code ec;
        fs::path c = fs::weakly_canonical(p, ec);
        allowed_roots.push_back(ec ? p : c);
    };
    add_root(fonts_dir_);
    add_root(downloaded_dir_);
    add_root(custom_subdir_);
    for (const auto& d : windows_system_font_dirs()) {
        add_root(d);
    }

    auto is_within_allowed = [&](const fs::path& candidate) -> std::optional<fs::path> {
        std::error_code ec;
        if (!fs::is_regular_file(candidate, ec) || ec) return std::nullopt;
        fs::path resolved = fs::weakly_canonical(candidate, ec);
        if (ec) return std::nullopt;
        const fs::path fonts_canon = fs::weakly_canonical(fonts_dir_, ec);
        if (!ec && util::is_path_within_root(fonts_canon, resolved)) {
            return resolved;
        }
        const std::string parent_lower = util::to_lower_ascii(util::path_to_utf8(resolved.parent_path()));
        for (const auto& root : allowed_roots) {
            if (parent_lower == util::to_lower_ascii(util::path_to_utf8(root))) {
                return resolved;
            }
        }
        return std::nullopt;
    };

    const auto all_items = collect_all_installed(false);
    const std::string lower_query = util::to_lower_ascii(cleaned);

    auto make_resolved = [](const fs::path& p) -> ResolvedFontFile {
        const std::string ext = util::to_lower_ascii(util::path_to_utf8(p.extension()));
        return {p, media_type_for_font_ext(ext), util::path_to_utf8(p.filename())};
    };

    // 1. Exact filename match
    for (const auto& item : all_items) {
        if (item.filename_lower == lower_query) {
            if (auto resolved = is_within_allowed(item.path)) return make_resolved(*resolved);
        }
    }
    // 2. Stem match
    for (const auto& item : all_items) {
        if (item.stem_lower == lower_query) {
            if (auto resolved = is_within_allowed(item.path)) return make_resolved(*resolved);
        }
    }
    // 3. Family or full name match
    for (const auto& item : all_items) {
        if (item.family_lower == lower_query || item.full_name_lower == lower_query) {
            if (auto resolved = is_within_allowed(item.path)) return make_resolved(*resolved);
        }
    }

    throw FontNotFoundError("Font '" + cleaned + "' not found in local or system font directories.");
}

json FontManager::save_uploaded_font(
    const std::string& filename,
    const std::vector<uint8_t>& raw_bytes,
    const std::optional<std::string>& override_family,
    bool overwrite) {
    const std::string safe_name = sanitize_font_filename(filename);
    const fs::path safe_rel = fs::path(util::utf8_to_wide(safe_name));
    const std::string ext_lower = util::to_lower_ascii(util::path_to_utf8(safe_rel.extension()));
    validate_font_bytes(raw_bytes, ext_lower);

    ensure_dirs();
    std::error_code ec;
    const fs::path root_canon = fs::weakly_canonical(fonts_dir_, ec);
    const fs::path target_path = fs::weakly_canonical(fonts_dir_ / safe_rel, ec);
    if (!util::is_path_within_root(root_canon, target_path) ||
        util::to_lower_ascii(util::path_to_utf8(target_path.parent_path())) !=
            util::to_lower_ascii(util::path_to_utf8(root_canon))) {
        throw FontValidationError("Invalid target font path.");
    }

    if (fs::exists(target_path, ec) && !overwrite) {
        throw FontValidationError(
            "Custom font '" + safe_name + "' already exists. Set overwrite=true to replace it.");
    }

    std::string write_err;
    if (!util::atomic_write_bytes(target_path, raw_bytes.data(), raw_bytes.size(), &write_err)) {
        throw std::runtime_error("Failed to save uploaded font: " + write_err);
    }

    auto indexed = build_indexed_font(target_path, "custom", override_family);
    if (!indexed.has_value()) {
        throw FontValidationError("Failed to index uploaded font '" + safe_name + "'.");
    }
    return indexed->entry;
}

json FontManager::download_catalog_fonts(
    const std::vector<std::string>& requested_fonts,
    const std::optional<std::string>& weight,
    bool overwrite) {
    ensure_dirs();
    json results = json::array();
    int downloaded_count = 0;
    int skipped_count = 0;
    int failed_count = 0;

    json manifest = load_downloaded_manifest();
    bool manifest_changed = false;

    for (const auto& raw_req : requested_fonts) {
        const std::string req_clean = util::trim_ascii(raw_req);
        if (req_clean.empty()) continue;

        const CatalogFontSpec* spec = find_catalog_spec(util::to_lower_ascii(req_clean));
        const std::string family_name = spec ? spec->family : req_clean;
        const std::optional<std::string> catalog_id =
            spec ? std::optional<std::string>(spec->id) : std::nullopt;

        std::string safe_base;
        for (unsigned char c : family_name) {
            if (std::isalnum(c) || c == '.' || c == '_' || c == '-') {
                safe_base.push_back(static_cast<char>(c));
            } else {
                safe_base.push_back('_');
            }
        }
        while (!safe_base.empty() && (safe_base.front() == '_' || safe_base.front() == '.')) {
            safe_base.erase(safe_base.begin());
        }
        while (!safe_base.empty() && (safe_base.back() == '_' || safe_base.back() == '.')) {
            safe_base.pop_back();
        }
        if (safe_base.empty()) safe_base = "google_font";

        if (weight.has_value()) {
            const std::string w = util::trim_ascii(*weight);
            if (!w.empty() && w != "400") {
                std::string w_clean;
                for (unsigned char c : w) {
                    if (std::isalnum(c)) w_clean.push_back(static_cast<char>(c));
                }
                if (!w_clean.empty()) {
                    safe_base += "_" + w_clean;
                }
            }
        }

        // Check already downloaded
        std::optional<fs::path> existing_match;
        if (!overwrite) {
            for (const char* ext : {".ttf", ".otf", ".woff2", ".woff"}) {
                const fs::path cand = downloaded_dir_ / util::utf8_to_wide(safe_base + ext);
                std::error_code ec;
                if (fs::is_regular_file(cand, ec) && !ec && fs::file_size(cand, ec) >= kMinFontFileBytes) {
                    existing_match = cand;
                    break;
                }
            }
        }

        if (existing_match.has_value()) {
            const std::string fname = util::path_to_utf8(existing_match->filename());
            std::error_code ec;
            const auto fsize = fs::file_size(*existing_match, ec);
            auto indexed = build_indexed_font(*existing_match, "downloaded", family_name);
            ++skipped_count;
            json item_res = {
                {"requested", req_clean},
                {"family", family_name},
                {"status", "skipped"},
                {"filename", fname},
                {"url", build_font_url(fname)},
                {"size_bytes", static_cast<uint64_t>(fsize)},
            };
            if (indexed.has_value()) {
                item_res["font"] = indexed->entry;
            }
            results.push_back(std::move(item_res));
            continue;
        }

        try {
            const std::string binary_url = resolve_google_font_binary_url(family_name, weight);
            const auto dl_resp = winhttp_get(binary_url, L"Vellum/2.0", 30000);
            if (!dl_resp.error.empty()) {
                throw std::runtime_error("Download failed: " + dl_resp.error);
            }
            if (dl_resp.status_code != 200) {
                throw std::runtime_error("Font binary server returned HTTP " + std::to_string(dl_resp.status_code));
            }

            const auto& font_bytes = dl_resp.body;
            std::string url_no_query = binary_url;
            if (const auto qpos = url_no_query.find('?'); qpos != std::string::npos) {
                url_no_query.resize(qpos);
            }
            std::string url_ext = util::to_lower_ascii(
                util::path_to_utf8(fs::path(util::utf8_to_wide(url_no_query)).extension()));
            if (!is_supported_font_extension(url_ext)) {
                if (font_bytes.size() >= 4 && std::memcmp(font_bytes.data(), "OTTO", 4) == 0) url_ext = ".otf";
                else if (font_bytes.size() >= 4 && std::memcmp(font_bytes.data(), "wOF2", 4) == 0) url_ext = ".woff2";
                else if (font_bytes.size() >= 4 && std::memcmp(font_bytes.data(), "wOFF", 4) == 0) url_ext = ".woff";
                else url_ext = ".ttf";
            }

            validate_font_bytes(font_bytes, url_ext);

            const std::string final_filename = safe_base + url_ext;
            const fs::path final_path = downloaded_dir_ / util::utf8_to_wide(final_filename);
            std::string write_err;
            if (!util::atomic_write_bytes(final_path, font_bytes.data(), font_bytes.size(), &write_err)) {
                throw std::runtime_error("Failed to save downloaded font: " + write_err);
            }

            manifest[final_filename] = {
                {"family", family_name},
                {"catalog_id", catalog_id.has_value() ? json(*catalog_id) : json(nullptr)},
                {"source_url", binary_url},
            };
            manifest_changed = true;

            auto indexed = build_indexed_font(final_path, "downloaded", family_name);
            ++downloaded_count;
            json dl_item = {
                {"requested", req_clean},
                {"family", family_name},
                {"status", "downloaded"},
                {"filename", final_filename},
                {"url", build_font_url(final_filename)},
                {"size_bytes", static_cast<uint64_t>(font_bytes.size())},
            };
            if (indexed.has_value()) {
                dl_item["font"] = indexed->entry;
            }
            results.push_back(std::move(dl_item));
        } catch (const std::exception& ex) {
            ++failed_count;
            results.push_back({
                {"requested", req_clean},
                {"family", family_name},
                {"status", "failed"},
                {"filename", nullptr},
                {"url", nullptr},
                {"size_bytes", 0},
                {"error", ex.what()},
            });
        }
    }

    if (manifest_changed) {
        try {
            save_downloaded_manifest(manifest);
        } catch (...) {
        }
    }

    return {
        {"downloaded_count", downloaded_count},
        {"skipped_count", skipped_count},
        {"failed_count", failed_count},
        {"results", std::move(results)},
    };
}

}  // namespace mangaflow
