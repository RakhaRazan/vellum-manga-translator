#include "settings_manager.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mangaflow {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct ProviderPreset {
    const char* id;
    const char* name;
    const char* default_base_url;
    const char* default_model;
    const char* default_api_key;
    bool requires_api_key;
    bool requires_base_url;
    bool is_local;
    const char* protocol;
    bool supports_json_schema;
    bool supports_vision;
    bool supports_top_k;
    bool supports_frequency_penalty;
    bool supports_reasoning_effort;
};

struct TargetLanguageSpec {
    const char* code;
    const char* name;
    const char* native_name;
    bool rtl;
    std::vector<const char*> aliases;
};

const std::vector<ProviderPreset>& provider_presets() {
    static const std::vector<ProviderPreset> kPresets = {
        {"lmstudio", "LM Studio (Local)", "http://localhost:1235/v1",
         "Gemma 4 E2B Uncensored HauhauCS Aggressive Q4 K P", "lm-studio",
         false, true, true, "openai", true, true, true, true, true},
        {"ollama", "Ollama (Local)", "http://localhost:11434/v1",
         "qwen2.5:7b", "ollama",
         false, true, true, "openai", true, true, true, true, true},
        {"openai", "OpenAI", "https://api.openai.com/v1",
         "gpt-4o-mini", "",
         true, true, false, "openai", true, true, false, true, true},
        {"gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
         "gemini-2.5-flash", "",
         true, true, false, "openai", true, true, false, false, true},
        {"anthropic", "Anthropic (Claude)", "https://api.anthropic.com/v1",
         "claude-sonnet-4-20250514", "",
         true, true, false, "anthropic", false, true, true, false, true},
        {"deepseek", "DeepSeek", "https://api.deepseek.com/v1",
         "deepseek-chat", "",
         true, true, false, "openai", false, false, false, true, false},
        {"openrouter", "OpenRouter", "https://openrouter.ai/api/v1",
         "google/gemini-2.5-flash", "",
         true, true, false, "openai", true, true, true, true, true},
        {"custom", "Custom (OpenAI-Compatible)", "",
         "", "",
         false, true, false, "openai", true, true, true, true, true},
    };
    return kPresets;
}

const ProviderPreset& find_provider_preset(const std::string& id) {
    for (const auto& preset : provider_presets()) {
        if (id == preset.id) {
            return preset;
        }
    }
    return provider_presets().front();
}

std::string normalize_provider_id(const std::string& raw) {
    const std::string cleaned = util::to_lower_ascii(util::trim_ascii(raw));
    if (cleaned.empty()) {
        return "lmstudio";
    }
    static const std::unordered_map<std::string, std::string> kAliases = {
        {"lmstudio", "lmstudio"},
        {"lm-studio", "lmstudio"},
        {"lm_studio", "lmstudio"},
        {"lm studio", "lmstudio"},
        {"ollama", "ollama"},
        {"openai", "openai"},
        {"gpt", "openai"},
        {"gemini", "gemini"},
        {"google", "gemini"},
        {"google-gemini", "gemini"},
        {"google_gemini", "gemini"},
        {"anthropic", "anthropic"},
        {"claude", "anthropic"},
        {"deepseek", "deepseek"},
        {"deep-seek", "deepseek"},
        {"openrouter", "openrouter"},
        {"open-router", "openrouter"},
        {"open_router", "openrouter"},
        {"custom", "custom"},
        {"openai-compatible", "custom"},
        {"openai_compatible", "custom"},
    };
    const auto it = kAliases.find(cleaned);
    if (it == kAliases.end()) {
        throw std::invalid_argument("Unsupported provider '" + raw + "'.");
    }
    return it->second;
}

std::string normalize_provider_base_url(const std::string& provider_id, const std::string& raw_url) {
    const auto& preset = find_provider_preset(provider_id);
    std::string cleaned = util::trim_ascii(raw_url);
    while (!cleaned.empty() && cleaned.back() == '/') {
        cleaned.pop_back();
    }
    if (cleaned.empty()) {
        std::string fallback = preset.default_base_url;
        while (!fallback.empty() && fallback.back() == '/') {
            fallback.pop_back();
        }
        return fallback;
    }
    if (provider_id == "lmstudio" || provider_id == "ollama") {
        const std::string lower = util::to_lower_ascii(cleaned);
        const bool has_v1_suffix = lower.size() >= 3 && lower.compare(lower.size() - 3, 3, "/v1") == 0;
        if (!has_v1_suffix && lower.find("/v1/") == std::string::npos &&
            lower.find("/api/") == std::string::npos) {
            cleaned += "/v1";
        }
    }
    return cleaned;
}

const std::vector<TargetLanguageSpec>& target_language_specs() {
    static const std::vector<TargetLanguageSpec> kLangs = {
        {"id", "Indonesian", "Bahasa Indonesia", false,
         {"id", "ind", "indonesian", "bahasa", "bahasa indonesia", "indo"}},
        {"en", "English", "English", false,
         {"en", "en-us", "en-gb", "eng", "english"}},
        {"ja", "Japanese", "日本語", false,
         {"ja", "jp", "jpn", "japanese", "日本語"}},
        {"ko", "Korean", "한국어", false,
         {"ko", "kr", "kor", "korean", "한국어"}},
        {"zh-CN", "Chinese (Simplified)", "简体中文", false,
         {"zh-cn", "zh_cn", "zh-hans", "zh_hans", "zh", "zho", "chi", "cn",
          "chinese", "chinese simplified", "simplified chinese",
          "chinese (simplified)", "简体中文", "中文"}},
        {"zh-TW", "Chinese (Traditional)", "繁體中文", false,
         {"zh-tw", "zh_tw", "zh-hant", "zh_hant", "zh-hk", "tw",
          "chinese traditional", "traditional chinese",
          "chinese (traditional)", "繁體中文"}},
        {"es", "Spanish", "Español", false,
         {"es", "es-es", "es-mx", "es-419", "spa", "spanish", "español", "espanol"}},
        {"fr", "French", "Français", false,
         {"fr", "fr-fr", "fra", "fre", "french", "français", "francais"}},
        {"de", "German", "Deutsch", false,
         {"de", "de-de", "deu", "ger", "german", "deutsch"}},
        {"pt", "Portuguese", "Português", false,
         {"pt", "pt-br", "pt-pt", "por", "portuguese", "português", "portugues"}},
        {"ru", "Russian", "Русский", false,
         {"ru", "rus", "russian", "русский"}},
        {"vi", "Vietnamese", "Tiếng Việt", false,
         {"vi", "vn", "vie", "vietnamese", "tiếng việt", "tieng viet"}},
        {"th", "Thai", "ไทย", false,
         {"th", "tha", "thai", "ไทย", "ภาษาไทย"}},
        {"ms", "Malay", "Bahasa Melayu", false,
         {"ms", "my", "msa", "may", "malay", "bahasa melayu", "melayu"}},
        {"ar", "Arabic", "العربية", true,
         {"ar", "ara", "arabic", "العربية"}},
        {"it", "Italian", "Italiano", false,
         {"it", "ita", "italian", "italiano"}},
    };
    return kLangs;
}

std::string normalize_target_language(const std::string& raw) {
    const std::string cleaned = util::to_lower_ascii(util::trim_ascii(raw));
    for (const auto& spec : target_language_specs()) {
        for (const char* alias : spec.aliases) {
            if (cleaned == alias) {
                return spec.code;
            }
        }
    }
    throw std::invalid_argument("Unsupported target_lang '" + raw + "'.");
}

std::string normalize_source_language(const std::string& raw) {
    const std::string sl = util::to_lower_ascii(util::trim_ascii(raw));
    if (sl == "ja" || sl == "jp" || sl == "japanese") return "ja";
    if (sl == "ko" || sl == "kr" || sl == "korean") return "ko";
    if (sl == "zh" || sl == "cn" || sl == "chinese") return "zh";
    throw std::invalid_argument("Unsupported source_lang '" + raw + "'. Must be 'ja', 'ko', or 'zh'.");
}

std::string normalize_reasoning_effort(const std::string& raw) {
    const std::string cleaned = util::to_lower_ascii(util::trim_ascii(raw));
    if (cleaned.empty() || cleaned == "off" || cleaned == "none" ||
        cleaned == "disabled" || cleaned == "false" || cleaned == "0") {
        return "off";
    }
    if (cleaned == "low" || cleaned == "minimal") return "low";
    if (cleaned == "medium" || cleaned == "med" || cleaned == "default") return "medium";
    if (cleaned == "high" || cleaned == "max" || cleaned == "xhigh") return "high";
    throw std::invalid_argument("Unsupported reasoning_effort '" + raw + "'.");
}

double clamp_double(const json& val, double low, double high, double fallback) {
    if (val.is_boolean() || val.is_null()) {
        return fallback;
    }
    double num = fallback;
    if (val.is_number()) {
        num = val.get<double>();
    } else if (val.is_string()) {
        try {
            num = std::stod(util::trim_ascii(val.get<std::string>()));
        } catch (...) {
            return fallback;
        }
    } else {
        return fallback;
    }
    if (!std::isfinite(num)) {
        return fallback;
    }
    return std::max(low, std::min(high, num));
}

json clamp_optional_double(const json& val, double low, double high, const json& fallback) {
    if (val.is_null()) {
        return nullptr;
    }
    if (val.is_string()) {
        const std::string trimmed = util::to_lower_ascii(util::trim_ascii(val.get<std::string>()));
        if (trimmed.empty() || trimmed == "null" || trimmed == "default" ||
            trimmed == "auto" || trimmed == "none") {
            return nullptr;
        }
    }
    if (val.is_boolean()) {
        return fallback;
    }
    const double fb = fallback.is_number() ? fallback.get<double>() : low;
    if (val.is_number()) {
        const double num = val.get<double>();
        if (!std::isfinite(num)) {
            return fallback.is_null() ? json(nullptr) : json(fb);
        }
        return std::max(low, std::min(high, num));
    }
    if (val.is_string()) {
        try {
            const double num = std::stod(util::trim_ascii(val.get<std::string>()));
            if (!std::isfinite(num)) {
                return fallback.is_null() ? json(nullptr) : json(fb);
            }
            return std::max(low, std::min(high, num));
        } catch (...) {
            return fallback;
        }
    }
    return fallback.is_null() ? json(nullptr) : json(fb);
}

int clamp_int(const json& val, int low, int high, int fallback) {
    if (val.is_boolean() || val.is_null()) {
        return fallback;
    }
    int num = fallback;
    if (val.is_number_integer()) {
        num = val.get<int>();
    } else if (val.is_number_float()) {
        num = static_cast<int>(std::llround(val.get<double>()));
    } else if (val.is_string()) {
        try {
            num = static_cast<int>(std::llround(std::stod(util::trim_ascii(val.get<std::string>()))));
        } catch (...) {
            return fallback;
        }
    } else {
        return fallback;
    }
    return std::max(low, std::min(high, num));
}

json clamp_optional_int(const json& val, int low, int high, const json& fallback) {
    if (val.is_null() || (val.is_string() && util::trim_ascii(val.get<std::string>()).empty())) {
        return nullptr;
    }
    if (val.is_boolean()) {
        return fallback;
    }
    if (val.is_number()) {
        const int num = static_cast<int>(std::llround(val.get<double>()));
        return std::max(low, std::min(high, num));
    }
    if (val.is_string()) {
        try {
            const int num = static_cast<int>(std::llround(std::stod(util::trim_ascii(val.get<std::string>()))));
            return std::max(low, std::min(high, num));
        } catch (...) {
            return fallback;
        }
    }
    return fallback;
}

bool coerce_bool(const json& val, bool fallback) {
    if (val.is_boolean()) {
        return val.get<bool>();
    }
    if (val.is_number_integer()) {
        return val.get<int>() != 0;
    }
    if (val.is_string()) {
        const std::string t = util::to_lower_ascii(util::trim_ascii(val.get<std::string>()));
        if (t == "1" || t == "true" || t == "yes" || t == "on") return true;
        if (t == "0" || t == "false" || t == "no" || t == "off") return false;
    }
    return fallback;
}

std::string json_to_string(const json& val) {
    if (val.is_string()) return val.get<std::string>();
    if (val.is_null()) return {};
    return val.dump();
}

json parse_glossary(const json& raw) {
    json parsed = json::object();
    if (raw.is_null()) {
        return parsed;
    }
    if (raw.is_object()) {
        for (auto it = raw.begin(); it != raw.end(); ++it) {
            const std::string src = util::trim_ascii(it.key());
            const std::string tgt = it.value().is_null() ? "" : util::trim_ascii(json_to_string(it.value()));
            if (!src.empty() && !tgt.empty()) {
                parsed[src] = tgt;
            }
        }
        return parsed;
    }
    if (raw.is_string()) {
        const std::string text = util::trim_ascii(raw.get<std::string>());
        if (text.empty()) {
            return parsed;
        }
        if (text.front() == '{' || text.front() == '[') {
            const json nested = json::parse(text, nullptr, false);
            if (!nested.is_discarded()) {
                return parse_glossary(nested);
            }
        }
        std::istringstream iss(text);
        std::string line;
        while (std::getline(iss, line)) {
            const std::string trimmed = util::trim_ascii(line);
            if (trimmed.empty() || trimmed[0] == '#' ||
                (trimmed.size() >= 2 && trimmed[0] == '/' && trimmed[1] == '/')) {
                continue;
            }
            static const std::vector<std::string> kSeps = {"->", "=>", "→", "=", ":"};
            for (const auto& sep : kSeps) {
                const auto pos = trimmed.find(sep);
                if (pos != std::string::npos) {
                    const std::string lhs = util::trim_ascii(trimmed.substr(0, pos));
                    const std::string rhs = util::trim_ascii(trimmed.substr(pos + sep.size()));
                    if (!lhs.empty() && !rhs.empty()) {
                        parsed[lhs] = rhs;
                    }
                    break;
                }
            }
        }
        return parsed;
    }
    if (raw.is_array()) {
        for (const auto& entry : raw) {
            if (entry.is_object()) {
                std::string src;
                std::string tgt;
                if (entry.contains("source") && !entry["source"].is_null()) {
                    src = util::trim_ascii(json_to_string(entry["source"]));
                } else if (entry.contains("src") && !entry["src"].is_null()) {
                    src = util::trim_ascii(json_to_string(entry["src"]));
                }
                if (entry.contains("target") && !entry["target"].is_null()) {
                    tgt = util::trim_ascii(json_to_string(entry["target"]));
                } else if (entry.contains("tgt") && !entry["tgt"].is_null()) {
                    tgt = util::trim_ascii(json_to_string(entry["tgt"]));
                }
                if (!src.empty() && !tgt.empty()) {
                    parsed[src] = tgt;
                }
            } else if (entry.is_string()) {
                const json sub = parse_glossary(entry);
                for (auto it = sub.begin(); it != sub.end(); ++it) {
                    parsed[it.key()] = it.value();
                }
            }
        }
    }
    return parsed;
}

std::string strip_quotes_and_spaces(std::string s) {
    s = util::trim_ascii(s);
    while (!s.empty() && (s.front() == '"' || s.front() == '\'')) {
        s.erase(s.begin());
        s = util::trim_ascii(s);
    }
    while (!s.empty() && (s.back() == '"' || s.back() == '\'')) {
        s.pop_back();
        s = util::trim_ascii(s);
    }
    return s;
}

std::string format_font_family_stack(const std::vector<std::string>& chain) {
    static const std::unordered_set<std::string> kGenerics = {
        "sans-serif", "serif", "monospace", "cursive", "fantasy", "system-ui", "ui-sans-serif"
    };
    std::vector<std::string> formatted;
    std::unordered_set<std::string> seen;
    for (const auto& item : chain) {
        const std::string cleaned = strip_quotes_and_spaces(item);
        if (cleaned.empty()) continue;
        const std::string lower = util::to_lower_ascii(cleaned);
        if (seen.count(lower)) continue;
        seen.insert(lower);
        if (kGenerics.count(lower)) {
            formatted.push_back(lower);
        } else if (cleaned.find(' ') != std::string::npos || cleaned.find('-') != std::string::npos) {
            formatted.push_back("\"" + cleaned + "\"");
        } else {
            formatted.push_back(cleaned);
        }
    }
    if (formatted.empty()) {
        return R"("CC Wild Words", "Anime Ace", "Comic Neue", "Noto Sans", sans-serif)";
    }
    if (!seen.count("sans-serif") && !seen.count("serif") && !seen.count("monospace")) {
        formatted.push_back("sans-serif");
    }
    std::string joined;
    for (size_t i = 0; i < formatted.size(); ++i) {
        if (i > 0) joined += ", ";
        joined += formatted[i];
    }
    return joined;
}

json build_default_provider_profiles() {
    json profiles = json::object();
    for (const auto& preset : provider_presets()) {
        profiles[preset.id] = {
            {"base_url", preset.default_base_url},
            {"api_key", preset.default_api_key},
            {"model", preset.default_model},
        };
    }
    return profiles;
}

}  // namespace

fs::path SettingsManager::default_settings_path() {
    if (auto env_path = util::get_env_wide(L"MANGA_SETTINGS_PATH")) {
        return fs::path(*env_path);
    }
    return util::get_portable_root_dir() / L"settings.json";
}

SettingsManager::SettingsManager(std::optional<fs::path> custom_path)
    : path_(custom_path.has_value() ? *custom_path : default_settings_path()) {}

json SettingsManager::list_provider_presets() {
    json arr = json::array();
    for (const auto& p : provider_presets()) {
        arr.push_back({
            {"id", p.id},
            {"name", p.name},
            {"default_base_url", p.default_base_url},
            {"default_model", p.default_model},
            {"requires_api_key", p.requires_api_key},
            {"requires_base_url", p.requires_base_url},
            {"is_local", p.is_local},
            {"protocol", p.protocol},
            {"supports_json_schema", p.supports_json_schema},
            {"supports_vision", p.supports_vision},
            {"supports_top_k", p.supports_top_k},
            {"supports_frequency_penalty", p.supports_frequency_penalty},
            {"supports_reasoning_effort", p.supports_reasoning_effort},
        });
    }
    return arr;
}

json SettingsManager::list_source_languages() {
    return json::array({
        {{"code", "ja"}, {"name", "Japanese (Manga)"}, {"native_name", "日本語"}, {"reading_order", "rtl"}},
        {{"code", "ko"}, {"name", "Korean (Manhwa)"}, {"native_name", "한국어"}, {"reading_order", "ltr"}},
        {{"code", "zh"}, {"name", "Chinese (Manhua)"}, {"native_name", "中文"}, {"reading_order", "ltr"}},
    });
}

json SettingsManager::list_target_languages() {
    json arr = json::array();
    for (const auto& spec : target_language_specs()) {
        arr.push_back({
            {"code", spec.code},
            {"name", spec.name},
            {"native_name", spec.native_name},
            {"rtl", spec.rtl},
        });
    }
    return arr;
}

json SettingsManager::list_reasoning_efforts() {
    return json::array({"off", "low", "medium", "high"});
}

json SettingsManager::list_detection_models() {
    return json::array({
        "vellum-det-yolo11m-seg-1024-fp16",
        "vellum-det-yolo11m-seg-1024-int8",
        "koharu-layout-rfdetr-seg-2xl",
        "comic-text-detector",
    });
}

json SettingsManager::list_ocr_models() {
    return json::array({
        "auto",
        "paddleocr-vl-1.6",
        "manga-ocr",
        "baberu-ocr",
        "hayai-ocr",
        "rapidocr-ppocrv4",
    });
}

json SettingsManager::list_inpainting_models() {
    return json::array({
        "lama-manga",
        "aot-inpainting",
        "flux-2-klein",
        "rorem-mixed",
        "lama-fp32",
    });
}

std::string SettingsManager::normalize_detection_model(const std::string& raw) {
    const std::string cleaned = util::to_lower_ascii(util::trim_ascii(raw));
    if (cleaned == "vellum-det-yolo11m-seg-1024-fp16" ||
        cleaned == "vellum-det-yolo11m-seg-1024-fp16.onnx" ||
        cleaned == "vellum_det_yolo11m_fp16" ||
        cleaned == "yolo11m-seg-fp16" ||
        cleaned == "yolo11-seg") {
        return "vellum-det-yolo11m-seg-1024-fp16";
    }
    if (cleaned == "vellum-det-yolo11m-seg-1024-int8" ||
        cleaned == "vellum-det-yolo11m-seg-1024-int8.onnx" ||
        cleaned == "vellum_det_yolo11m_int8" ||
        cleaned == "yolo11m-seg-int8") {
        return "vellum-det-yolo11m-seg-1024-int8";
    }
    if (cleaned.empty() || cleaned == "default" || cleaned == "auto" ||
        cleaned == "koharu-layout-rfdetr-seg-2xl" ||
        cleaned == "koharu-layout-rfdetr-seg-2xl-1152" ||
        cleaned == "koharu_layout_rfdetr_seg_2xl" ||
        cleaned == "koharu_rfdetr" ||
        cleaned == "koharu-rfdetr" ||
        cleaned == "rfdetr" ||
        cleaned == "rf-detr" ||
        cleaned == "rfdetr-seg-2xl" ||
        cleaned == "rfdetr-seg-2xlarge") {
        return "koharu-layout-rfdetr-seg-2xl";
    }
    if (cleaned == "comic-text-detector" ||
        cleaned == "comic_text_detector" ||
        cleaned == "comictextdetector" ||
        cleaned == "ctd") {
        return "comic-text-detector";
    }
    throw std::invalid_argument(
        "Unsupported detection_model '" + raw +
        "'. Supported options: 'vellum-det-yolo11m-seg-1024-fp16', 'vellum-det-yolo11m-seg-1024-int8', 'koharu-layout-rfdetr-seg-2xl', 'comic-text-detector'.");
}

std::string SettingsManager::normalize_ocr_model(const std::string& raw) {
    const std::string cleaned = util::to_lower_ascii(util::trim_ascii(raw));
    if (cleaned.empty() || cleaned == "auto" || cleaned == "default") {
        return "auto";
    }
    if (cleaned == "manga-ocr" || cleaned == "manga_ocr" || cleaned == "mangaocr" ||
        cleaned == "koharu-manga-ocr" || cleaned == "koharu_manga_ocr") {
        return "manga-ocr";
    }
    if (cleaned == "baberu-ocr" || cleaned == "baberu_ocr" || cleaned == "baberu" ||
        cleaned == "baberuocr") {
        return "baberu-ocr";
    }
    if (cleaned == "hayai-ocr" || cleaned == "hayai_ocr" || cleaned == "hayai" ||
        cleaned == "hayaiocr" || cleaned == "hayai-ocr-v2" || cleaned == "hayai_ocr_v2") {
        return "hayai-ocr";
    }
    if (cleaned == "paddleocr-vl-1.6" || cleaned == "paddleocr_vl_1.6" ||
        cleaned == "paddleocr-vl-gguf" || cleaned == "paddleocr_vl_gguf" ||
        cleaned == "paddleocr-vl" || cleaned == "paddleocr_vl" ||
        cleaned == "paddle-vl" || cleaned == "paddle_vl" ||
        cleaned == "paddleocr-vl-1.6-gguf") {
        return "paddleocr-vl-1.6";
    }
    if (cleaned == "rapidocr-ppocrv4" || cleaned == "rapidocr_ppocrv4" ||
        cleaned == "rapidocr" || cleaned == "rapid-ocr" || cleaned == "rapid_ocr" ||
        cleaned == "ppocrv4" || cleaned == "pp-ocrv4") {
        return "rapidocr-ppocrv4";
    }
    throw std::invalid_argument(
        "Unsupported ocr_model '" + raw +
        "'. Supported options: 'auto', 'paddleocr-vl-1.6', 'manga-ocr', 'baberu-ocr', 'hayai-ocr', 'rapidocr-ppocrv4'.");
}

std::string SettingsManager::normalize_inpainting_model(const std::string& raw) {
    const std::string cleaned = util::to_lower_ascii(util::trim_ascii(raw));
    if (cleaned.empty() || cleaned == "default" || cleaned == "auto" ||
        cleaned == "lama-manga" || cleaned == "lama_manga" ||
        cleaned == "koharu-lama-manga" || cleaned == "koharu_lama_manga") {
        return "lama-manga";
    }
    if (cleaned == "aot-inpainting" || cleaned == "aot_inpainting" ||
        cleaned == "aot" || cleaned == "aot-manga") {
        return "aot-inpainting";
    }
    if (cleaned == "flux-2-klein" || cleaned == "flux_2_klein" ||
        cleaned == "flux.2-klein" || cleaned == "flux2-klein" ||
        cleaned == "flux-klein" || cleaned == "flux") {
        return "flux-2-klein";
    }
    if (cleaned == "rorem-mixed" || cleaned == "rorem_mixed" ||
        cleaned == "rorem" || cleaned == "rorem-mixed-gguf" ||
        cleaned == "rorem_mixed_gguf") {
        return "rorem-mixed";
    }
    if (cleaned == "lama-fp32" || cleaned == "lama_fp32" ||
        cleaned == "lama" || cleaned == "fp32") {
        return "lama-fp32";
    }
    throw std::invalid_argument(
        "Unsupported inpainting_model '" + raw +
        "'. Supported options: 'lama-manga', 'aot-inpainting', 'flux-2-klein', 'rorem-mixed', 'lama-fp32'.");
}

json SettingsManager::build_default_settings() {
    const auto& lm = find_provider_preset("lmstudio");
    return {
        {"provider", "lmstudio"},
        {"base_url", lm.default_base_url},
        {"api_key", lm.default_api_key},
        {"model", lm.default_model},
        {"timeout", 90.0},
        {"source_lang", "ja"},
        {"target_lang", "id"},
        {"translation_tone", "standard"},
        {"temperature", 0.2},
        {"top_p", 0.95},
        {"top_k", 40},
        {"max_tokens", 1024},
        {"frequency_penalty", 0.0},
        {"reasoning_effort", "off"},
        {"strip_thinking", true},
        {"use_vision", false},
        {"vision_max_side", 1024},
        {"custom_instructions", ""},
        {"glossary", json::object()},
        {"providers", build_default_provider_profiles()},
        // Tab 1: Appearance
        {"theme", "dark"},
        {"accent_color", "amber"},
        {"ui_density", "compact"},
        // Tab 2: Pipeline (Vellum YOLO11m-Seg default + Smart Hybrid GPU/CPU settings)
        {"detection_model", "vellum-det-yolo11m-seg-1024-fp16"},
        {"text_threshold", nullptr},
        {"bubble_threshold", nullptr},
        {"panel_threshold", nullptr},
        {"whisper_threshold", nullptr},
        {"borderless_threshold", nullptr},
        {"enable_adaptive_tiling", true},
        {"sfx_suppress_iou", 0.60},
        {"ocr_model", "auto"},
        {"inpainting_model", "lama-manga"},
        {"inpaint_api_url", "http://127.0.0.1:8188"},
        {"execution_mode", "hybrid"},
        {"conf_threshold", 0.35},
        {"nms_threshold", 0.45},
        {"mask_threshold", 0.5},
        {"bbox_padding", 6},
        {"lama_tile_size", 512},
        {"reading_order", "auto"},
        {"auto_unload_vram", true},
        {"refine_mask", true},
        // Tab 5: Typesetting
        {"font_family", R"("CC Wild Words", "Anime Ace", "Comic Neue", "Noto Sans", sans-serif)"},
        {"font_fallback_chain", json::array({"CC Wild Words", "Anime Ace", "Comic Neue", "Noto Sans", "sans-serif"})},
        {"auto_fit_font_size", true},
        {"default_font_size", 18},
        {"default_stroke_width", 4.0},
        {"default_stroke_color", "#ffffff"},
        {"default_text_color", "#000000"},
        {"default_line_height", 1.18},
        {"default_align", "center"},
        {"all_caps", false},
    };
}

json SettingsManager::sanitize_and_merge(const json& raw, const json& base) const {
    if (!raw.is_object()) {
        throw std::invalid_argument("Settings payload must be a JSON object.");
    }

    json current = base.is_object() ? base : build_default_settings();
    json profiles = build_default_provider_profiles();

    // Preserve existing saved per-provider profiles
    if (current.contains("providers") && current["providers"].is_object()) {
        for (auto it = current["providers"].begin(); it != current["providers"].end(); ++it) {
            if (!it.value().is_object()) continue;
            try {
                const std::string norm_pid = normalize_provider_id(it.key());
                const auto& preset = find_provider_preset(norm_pid);
                profiles[norm_pid] = {
                    {"base_url", it.value().value("base_url", std::string(preset.default_base_url))},
                    {"api_key", it.value().value("api_key", std::string(preset.default_api_key))},
                    {"model", it.value().value("model", std::string(preset.default_model))},
                };
            } catch (...) {
            }
        }
    }

    // Apply patched per-provider profiles
    if (raw.contains("providers") && raw["providers"].is_object()) {
        for (auto it = raw["providers"].begin(); it != raw["providers"].end(); ++it) {
            if (!it.value().is_object()) continue;
            try {
                const std::string norm_pid = normalize_provider_id(it.key());
                json& cur_prof = profiles[norm_pid];
                if (it.value().contains("base_url") && !it.value()["base_url"].is_null()) {
                    cur_prof["base_url"] = util::trim_ascii(json_to_string(it.value()["base_url"]));
                }
                if (it.value().contains("api_key") && !it.value()["api_key"].is_null()) {
                    cur_prof["api_key"] = util::trim_ascii(json_to_string(it.value()["api_key"]));
                }
                if (it.value().contains("model") && !it.value()["model"].is_null()) {
                    cur_prof["model"] = util::trim_ascii(json_to_string(it.value()["model"]));
                }
            } catch (...) {
            }
        }
    }

    const std::string prev_provider = current.value("provider", std::string("lmstudio"));
    std::string active_provider = prev_provider;
    if (raw.contains("provider") && !raw["provider"].is_null()) {
        active_provider = normalize_provider_id(json_to_string(raw["provider"]));
    }

    const bool provider_switched = (active_provider != prev_provider);
    const auto& preset = find_provider_preset(active_provider);
    const json active_profile = profiles[active_provider];

    std::string base_url;
    if (raw.contains("base_url") && !raw["base_url"].is_null()) {
        base_url = normalize_provider_base_url(active_provider, json_to_string(raw["base_url"]));
    } else if (provider_switched) {
        base_url = normalize_provider_base_url(active_provider, active_profile.value("base_url", std::string("")));
    } else {
        base_url = normalize_provider_base_url(active_provider, current.value("base_url", std::string("")));
    }

    std::string api_key;
    if (raw.contains("api_key") && !raw["api_key"].is_null()) {
        api_key = util::trim_ascii(json_to_string(raw["api_key"]));
        if (api_key.empty() && std::string(preset.default_api_key).size() > 0) {
            api_key = preset.default_api_key;
        }
    } else if (provider_switched) {
        api_key = active_profile.value("api_key", std::string(preset.default_api_key));
    } else {
        api_key = current.value("api_key", std::string(preset.default_api_key));
    }

    std::string model;
    if (raw.contains("model") && !raw["model"].is_null()) {
        model = util::trim_ascii(json_to_string(raw["model"]));
        if (model.empty() && !provider_switched) {
            model = preset.default_model;
        }
    } else if (provider_switched) {
        model = active_profile.value("model", std::string(""));
        if (model.empty()) model = preset.default_model;
    } else {
        model = current.value("model", std::string(""));
        if (model.empty()) model = preset.default_model;
    }

    profiles[active_provider] = {
        {"base_url", base_url},
        {"api_key", api_key},
        {"model", model},
    };

    std::string source_lang = current.value("source_lang", std::string("ja"));
    if (raw.contains("source_lang") && !raw["source_lang"].is_null()) {
        source_lang = normalize_source_language(json_to_string(raw["source_lang"]));
    }

    std::string target_lang = current.value("target_lang", std::string("id"));
    if (raw.contains("target_lang") && !raw["target_lang"].is_null()) {
        target_lang = normalize_target_language(json_to_string(raw["target_lang"]));
    }

    std::string translation_tone = current.value("translation_tone", std::string("standard"));
    if (raw.contains("translation_tone") && !raw["translation_tone"].is_null()) {
        const std::string tt = util::to_lower_ascii(util::trim_ascii(json_to_string(raw["translation_tone"])));
        if (tt.empty() || tt == "standard" || tt == "default" || tt == "normal") {
            translation_tone = "standard";
        } else if (tt == "casual" || tt == "slang" || tt == "colloquial") {
            translation_tone = "casual";
        } else if (tt == "adult_doujinshi" || tt == "adult-doujinshi") {
            translation_tone = "adult_doujinshi";
        } else if (tt == "adult" || tt == "doujinshi" || tt == "nsfw" || tt == "uncensored") {
            translation_tone = "adult";
        }
    }

    std::string reasoning_effort = current.value("reasoning_effort", std::string("off"));
    if (raw.contains("reasoning_effort") && !raw["reasoning_effort"].is_null()) {
        reasoning_effort = normalize_reasoning_effort(json_to_string(raw["reasoning_effort"]));
    }

    const double temperature = (raw.contains("temperature") && !raw["temperature"].is_null())
        ? clamp_double(raw["temperature"], 0.0, 2.0, current.value("temperature", 0.2))
        : current.value("temperature", 0.2);

    const json top_p = raw.contains("top_p")
        ? clamp_optional_double(raw["top_p"], 0.0, 1.0, current.value("top_p", json(0.95)))
        : current.value("top_p", json(0.95));

    const json top_k = raw.contains("top_k")
        ? clamp_optional_int(raw["top_k"], 0, 500, current.value("top_k", json(40)))
        : current.value("top_k", json(40));

    const int max_tokens = (raw.contains("max_tokens") && !raw["max_tokens"].is_null())
        ? clamp_int(raw["max_tokens"], 64, 32768, current.value("max_tokens", 1024))
        : current.value("max_tokens", 1024);

    const json frequency_penalty = raw.contains("frequency_penalty")
        ? clamp_optional_double(raw["frequency_penalty"], -2.0, 2.0, current.value("frequency_penalty", json(0.0)))
        : current.value("frequency_penalty", json(0.0));

    const double timeout = (raw.contains("timeout") && !raw["timeout"].is_null())
        ? clamp_double(raw["timeout"], 5.0, 300.0, current.value("timeout", 90.0))
        : current.value("timeout", 90.0);

    const bool use_vision = (raw.contains("use_vision") && !raw["use_vision"].is_null())
        ? coerce_bool(raw["use_vision"], current.value("use_vision", false))
        : current.value("use_vision", false);

    const int vision_max_side = (raw.contains("vision_max_side") && !raw["vision_max_side"].is_null())
        ? clamp_int(raw["vision_max_side"], 256, 2048, current.value("vision_max_side", 1024))
        : current.value("vision_max_side", 1024);

    const std::string custom_instructions =
        (raw.contains("custom_instructions") && !raw["custom_instructions"].is_null())
        ? util::trim_ascii(json_to_string(raw["custom_instructions"]))
        : current.value("custom_instructions", std::string(""));

    const json glossary = raw.contains("glossary")
        ? parse_glossary(raw["glossary"])
        : current.value("glossary", json::object());

    const bool strip_thinking = (raw.contains("strip_thinking") && !raw["strip_thinking"].is_null())
        ? coerce_bool(raw["strip_thinking"], current.value("strip_thinking", true))
        : current.value("strip_thinking", true);

    // Tab 1: Appearance
    std::string theme = current.value("theme", std::string("dark"));
    if (raw.contains("theme") && !raw["theme"].is_null()) {
        const std::string t = util::to_lower_ascii(util::trim_ascii(json_to_string(raw["theme"])));
        if (t == "dark" || t == "light" || t == "system") theme = t;
    }

    std::string accent_color = current.value("accent_color", std::string("amber"));
    if (raw.contains("accent_color") && !raw["accent_color"].is_null()) {
        const std::string a = util::to_lower_ascii(util::trim_ascii(json_to_string(raw["accent_color"])));
        if (a == "amber" || a == "teal" || a == "crimson" || a == "slate") accent_color = a;
    }

    std::string ui_density = current.value("ui_density", std::string("compact"));
    if (raw.contains("ui_density") && !raw["ui_density"].is_null()) {
        const std::string d = util::to_lower_ascii(util::trim_ascii(json_to_string(raw["ui_density"])));
        if (d == "compact" || d == "comfortable") ui_density = d;
    }

    // Tab 2: Pipeline
    std::string detection_model = current.value("detection_model", std::string("koharu-layout-rfdetr-seg-2xl"));
    if (raw.contains("detection_model") && !raw["detection_model"].is_null()) {
        detection_model = normalize_detection_model(json_to_string(raw["detection_model"]));
    }

    const json text_threshold = raw.contains("text_threshold")
        ? clamp_optional_double(raw["text_threshold"], 0.05, 0.95, current.value("text_threshold", json(nullptr)))
        : current.value("text_threshold", json(nullptr));

    const json bubble_threshold = raw.contains("bubble_threshold")
        ? clamp_optional_double(raw["bubble_threshold"], 0.05, 0.95, current.value("bubble_threshold", json(nullptr)))
        : current.value("bubble_threshold", json(nullptr));

    const json panel_threshold = raw.contains("panel_threshold")
        ? clamp_optional_double(raw["panel_threshold"], 0.05, 0.95, current.value("panel_threshold", json(nullptr)))
        : current.value("panel_threshold", json(nullptr));

    const json whisper_threshold = raw.contains("whisper_threshold")
        ? clamp_optional_double(raw["whisper_threshold"], 0.05, 0.95, current.value("whisper_threshold", json(nullptr)))
        : current.value("whisper_threshold", json(nullptr));

    const json borderless_threshold = raw.contains("borderless_threshold")
        ? clamp_optional_double(raw["borderless_threshold"], 0.05, 0.95, current.value("borderless_threshold", json(nullptr)))
        : current.value("borderless_threshold", json(nullptr));

    const bool enable_adaptive_tiling = (raw.contains("enable_adaptive_tiling") && !raw["enable_adaptive_tiling"].is_null())
        ? coerce_bool(raw["enable_adaptive_tiling"], current.value("enable_adaptive_tiling", true))
        : current.value("enable_adaptive_tiling", true);

    const double sfx_suppress_iou = (raw.contains("sfx_suppress_iou") && !raw["sfx_suppress_iou"].is_null())
        ? clamp_double(raw["sfx_suppress_iou"], 0.05, 1.0, current.value("sfx_suppress_iou", 0.60))
        : current.value("sfx_suppress_iou", 0.60);

    std::string ocr_model = current.value("ocr_model", std::string("auto"));
    if (raw.contains("ocr_model") && !raw["ocr_model"].is_null()) {
        ocr_model = normalize_ocr_model(json_to_string(raw["ocr_model"]));
    }

    std::string inpainting_model = current.value("inpainting_model", std::string("lama-manga"));
    if (raw.contains("inpainting_model") && !raw["inpainting_model"].is_null()) {
        inpainting_model = normalize_inpainting_model(json_to_string(raw["inpainting_model"]));
    }

    std::string inpaint_api_url = current.value("inpaint_api_url", std::string("http://127.0.0.1:8188"));
    if (raw.contains("inpaint_api_url") && !raw["inpaint_api_url"].is_null()) {
        const std::string u = util::trim_ascii(json_to_string(raw["inpaint_api_url"]));
        if (!u.empty()) inpaint_api_url = u;
    }

    std::string execution_mode = current.value("execution_mode", std::string("hybrid"));
    if (execution_mode == "auto" || execution_mode.empty()) {
        execution_mode = "hybrid";
    } else if (execution_mode == "cpu") {
        execution_mode = "cpu_only";
    } else if (execution_mode == "gpu") {
        execution_mode = "force_gpu";
    }
    if (raw.contains("execution_mode") && !raw["execution_mode"].is_null()) {
        const std::string em = util::to_lower_ascii(util::trim_ascii(json_to_string(raw["execution_mode"])));
        if (em == "cpu" || em == "cpu_only" || em == "cpu only" || em == "cpu-only") {
            execution_mode = "cpu_only";
        } else if (em == "gpu" || em == "force_gpu" || em == "force gpu" || em == "force-gpu") {
            execution_mode = "force_gpu";
        } else {
            execution_mode = "hybrid";
        }
    }
    util::set_env_utf8(L"MANGA_FORCE_CPU", execution_mode == "cpu_only" ? "1" : "0");

    const double conf_threshold = (raw.contains("conf_threshold") && !raw["conf_threshold"].is_null())
        ? clamp_double(raw["conf_threshold"], 0.05, 0.95, current.value("conf_threshold", 0.35))
        : current.value("conf_threshold", 0.35);

    const double nms_threshold = (raw.contains("nms_threshold") && !raw["nms_threshold"].is_null())
        ? clamp_double(raw["nms_threshold"], 0.05, 0.95, current.value("nms_threshold", 0.45))
        : current.value("nms_threshold", 0.45);

    const double mask_threshold = (raw.contains("mask_threshold") && !raw["mask_threshold"].is_null())
        ? clamp_double(raw["mask_threshold"], 0.05, 0.95, current.value("mask_threshold", 0.5))
        : current.value("mask_threshold", 0.5);

    const int bbox_padding = (raw.contains("bbox_padding") && !raw["bbox_padding"].is_null())
        ? clamp_int(raw["bbox_padding"], 0, 64, current.value("bbox_padding", 6))
        : current.value("bbox_padding", 6);

    const int lama_tile_size = (raw.contains("lama_tile_size") && !raw["lama_tile_size"].is_null())
        ? clamp_int(raw["lama_tile_size"], 256, 1024, current.value("lama_tile_size", 512))
        : current.value("lama_tile_size", 512);

    std::string reading_order = current.value("reading_order", std::string("auto"));
    if (raw.contains("reading_order") && !raw["reading_order"].is_null()) {
        const std::string ro = util::to_lower_ascii(util::trim_ascii(json_to_string(raw["reading_order"])));
        if (ro == "auto" || ro == "rtl" || ro == "ltr") reading_order = ro;
    }

    const bool auto_unload_vram = (raw.contains("auto_unload_vram") && !raw["auto_unload_vram"].is_null())
        ? coerce_bool(raw["auto_unload_vram"], current.value("auto_unload_vram", true))
        : current.value("auto_unload_vram", true);

    const bool refine_mask = (raw.contains("refine_mask") && !raw["refine_mask"].is_null())
        ? coerce_bool(raw["refine_mask"], current.value("refine_mask", true))
        : current.value("refine_mask", true);

    // Tab 5: Typesetting
    std::vector<std::string> font_fallback_chain;
    if (current.contains("font_fallback_chain") && current["font_fallback_chain"].is_array()) {
        for (const auto& item : current["font_fallback_chain"]) {
            if (item.is_string()) font_fallback_chain.push_back(item.get<std::string>());
        }
    }
    if (font_fallback_chain.empty()) {
        font_fallback_chain = {"CC Wild Words", "Anime Ace", "Comic Neue", "Noto Sans", "sans-serif"};
    }

    bool chain_updated = false;
    if (raw.contains("font_fallback_chain") && raw["font_fallback_chain"].is_array()) {
        std::vector<std::string> cleaned_chain;
        std::unordered_set<std::string> seen;
        for (const auto& entry : raw["font_fallback_chain"]) {
            const std::string s = strip_quotes_and_spaces(json_to_string(entry));
            const std::string lower = util::to_lower_ascii(s);
            if (!s.empty() && !seen.count(lower)) {
                seen.insert(lower);
                cleaned_chain.push_back(s);
            }
        }
        if (!cleaned_chain.empty()) {
            font_fallback_chain = std::move(cleaned_chain);
            chain_updated = true;
        }
    }

    std::string font_family = current.value(
        "font_family",
        std::string(R"("CC Wild Words", "Anime Ace", "Comic Neue", "Noto Sans", sans-serif)"));
    if (raw.contains("font_family") && !raw["font_family"].is_null()) {
        const std::string ff = util::trim_ascii(json_to_string(raw["font_family"]));
        if (!ff.empty()) {
            font_family = ff;
            if (!chain_updated) {
                std::vector<std::string> parsed_chain;
                std::istringstream iss(font_family);
                std::string part;
                while (std::getline(iss, part, ',')) {
                    const std::string cleaned = strip_quotes_and_spaces(part);
                    if (!cleaned.empty()) parsed_chain.push_back(cleaned);
                }
                if (!parsed_chain.empty()) {
                    font_fallback_chain = std::move(parsed_chain);
                }
            }
        }
    } else if (chain_updated) {
        font_family = format_font_family_stack(font_fallback_chain);
    }

    const bool auto_fit_font_size = (raw.contains("auto_fit_font_size") && !raw["auto_fit_font_size"].is_null())
        ? coerce_bool(raw["auto_fit_font_size"], current.value("auto_fit_font_size", true))
        : current.value("auto_fit_font_size", true);

    const int default_font_size = (raw.contains("default_font_size") && !raw["default_font_size"].is_null())
        ? clamp_int(raw["default_font_size"], 8, 96, current.value("default_font_size", 18))
        : current.value("default_font_size", 18);

    const double default_stroke_width = (raw.contains("default_stroke_width") && !raw["default_stroke_width"].is_null())
        ? clamp_double(raw["default_stroke_width"], 0.0, 16.0, current.value("default_stroke_width", 4.0))
        : current.value("default_stroke_width", 4.0);

    std::string default_stroke_color = current.value("default_stroke_color", std::string("#ffffff"));
    if (raw.contains("default_stroke_color") && !raw["default_stroke_color"].is_null()) {
        const std::string c = util::trim_ascii(json_to_string(raw["default_stroke_color"]));
        if (!c.empty()) default_stroke_color = c;
    }

    std::string default_text_color = current.value("default_text_color", std::string("#000000"));
    if (raw.contains("default_text_color") && !raw["default_text_color"].is_null()) {
        const std::string c = util::trim_ascii(json_to_string(raw["default_text_color"]));
        if (!c.empty()) default_text_color = c;
    }

    const double default_line_height = (raw.contains("default_line_height") && !raw["default_line_height"].is_null())
        ? clamp_double(raw["default_line_height"], 0.8, 2.5, current.value("default_line_height", 1.18))
        : current.value("default_line_height", 1.18);

    std::string default_align = current.value("default_align", std::string("center"));
    if (raw.contains("default_align") && !raw["default_align"].is_null()) {
        const std::string al = util::to_lower_ascii(util::trim_ascii(json_to_string(raw["default_align"])));
        if (al == "left" || al == "center" || al == "right") default_align = al;
    }

    const bool all_caps = (raw.contains("all_caps") && !raw["all_caps"].is_null())
        ? coerce_bool(raw["all_caps"], current.value("all_caps", false))
        : current.value("all_caps", false);

    return {
        {"provider", active_provider},
        {"base_url", base_url},
        {"api_key", api_key},
        {"model", model},
        {"timeout", timeout},
        {"source_lang", source_lang},
        {"target_lang", target_lang},
        {"translation_tone", translation_tone},
        {"temperature", temperature},
        {"top_p", top_p},
        {"top_k", top_k},
        {"max_tokens", max_tokens},
        {"frequency_penalty", frequency_penalty},
        {"reasoning_effort", reasoning_effort},
        {"strip_thinking", strip_thinking},
        {"use_vision", use_vision},
        {"vision_max_side", vision_max_side},
        {"custom_instructions", custom_instructions},
        {"glossary", glossary},
        {"providers", profiles},
        {"theme", theme},
        {"accent_color", accent_color},
        {"ui_density", ui_density},
        {"detection_model", detection_model},
        {"text_threshold", text_threshold},
        {"bubble_threshold", bubble_threshold},
        {"panel_threshold", panel_threshold},
        {"whisper_threshold", whisper_threshold},
        {"borderless_threshold", borderless_threshold},
        {"enable_adaptive_tiling", enable_adaptive_tiling},
        {"sfx_suppress_iou", sfx_suppress_iou},
        {"ocr_model", ocr_model},
        {"inpainting_model", inpainting_model},
        {"inpaint_api_url", inpaint_api_url},
        {"execution_mode", execution_mode},
        {"conf_threshold", conf_threshold},
        {"nms_threshold", nms_threshold},
        {"mask_threshold", mask_threshold},
        {"bbox_padding", bbox_padding},
        {"lama_tile_size", lama_tile_size},
        {"reading_order", reading_order},
        {"auto_unload_vram", auto_unload_vram},
        {"refine_mask", refine_mask},
        {"font_family", font_family},
        {"font_fallback_chain", font_fallback_chain},
        {"auto_fit_font_size", auto_fit_font_size},
        {"default_font_size", default_font_size},
        {"default_stroke_width", default_stroke_width},
        {"default_stroke_color", default_stroke_color},
        {"default_text_color", default_text_color},
        {"default_line_height", default_line_height},
        {"default_align", default_align},
        {"all_caps", all_caps},
    };
}

json SettingsManager::load(bool force_reload) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cache_.has_value() && !force_reload) {
        return *cache_;
    }

    const json defaults = build_default_settings();
    std::error_code ec;
    if (!fs::is_regular_file(path_, ec) || ec) {
        cache_ = defaults;
        return *cache_;
    }

    const auto bytes = util::read_file_bytes(path_);
    if (!bytes.has_value()) {
        cache_ = defaults;
        return *cache_;
    }

    try {
        const json parsed = json::parse(bytes->begin(), bytes->end());
        if (parsed.is_object()) {
            cache_ = sanitize_and_merge(parsed, defaults);
        } else {
            cache_ = defaults;
        }
    } catch (...) {
        cache_ = defaults;
    }

    return *cache_;
}

void SettingsManager::save_locked(const json& settings) {
    const std::string serialized = settings.dump(2) + "\n";
    std::string err;
    if (!util::atomic_write_bytes(path_, serialized.data(), serialized.size(), &err)) {
        throw std::runtime_error("Failed to persist settings to disk: " + err);
    }
    cache_ = settings;
}

json SettingsManager::update(const json& patch) {
    std::lock_guard<std::mutex> lock(mutex_);
    json current;
    if (cache_.has_value()) {
        current = *cache_;
    } else {
        const json defaults = build_default_settings();
        std::error_code ec;
        if (fs::is_regular_file(path_, ec) && !ec) {
            if (const auto bytes = util::read_file_bytes(path_)) {
                try {
                    const json parsed = json::parse(bytes->begin(), bytes->end());
                    current = parsed.is_object() ? sanitize_and_merge(parsed, defaults) : defaults;
                } catch (...) {
                    current = defaults;
                }
            } else {
                current = defaults;
            }
        } else {
            current = defaults;
        }
    }

    const json merged = sanitize_and_merge(patch, current);
    save_locked(merged);
    return merged;
}

json SettingsManager::format_api_response(const json& settings) const {
    json resp = settings;
    resp["settings_path"] = util::path_to_utf8(path_);
    resp["available_providers"] = list_provider_presets();
    resp["supported_target_languages"] = list_target_languages();
    resp["supported_reasoning_efforts"] = list_reasoning_efforts();
    resp["supported_detection_models"] = list_detection_models();
    resp["supported_ocr_models"] = list_ocr_models();
    resp["supported_inpainting_models"] = list_inpainting_models();
    return resp;
}

json SettingsManager::get_active_provider_summary() {
    const json s = load();
    const std::string pid = s.value("provider", std::string("lmstudio"));
    const auto& preset = find_provider_preset(pid);
    return {
        {"provider", pid},
        {"provider_name", preset.name},
        {"endpoint", s.value("base_url", std::string(preset.default_base_url))},
        {"model", s.value("model", std::string(preset.default_model))},
        {"use_vision", s.value("use_vision", false)},
        {"reasoning_effort", s.value("reasoning_effort", std::string("off"))},
        {"target_lang", s.value("target_lang", std::string("id"))},
        {"translation_tone", s.value("translation_tone", std::string("standard"))},
    };
}

}  // namespace mangaflow
