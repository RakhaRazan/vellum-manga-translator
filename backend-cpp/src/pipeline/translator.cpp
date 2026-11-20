#include <httplib.h>

#include "translator.hpp"
#include "../utils.hpp"

#include <winhttp.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100 4244 4456 4996)
#endif
#include <stb_image.h>
#include <stb_image_write.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace mangaflow::pipeline {

using json = nlohmann::json;

namespace {

constexpr size_t kMaxHttpResponseBytes = 32ULL * 1024ULL * 1024ULL;  // 32 MiB

struct StbiDeleter {
    void operator()(unsigned char* p) const noexcept {
        if (p) stbi_image_free(p);
    }
};
using StbiPtr = std::unique_ptr<unsigned char, StbiDeleter>;

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

std::string safe_json_dump(const json& obj) {
    return obj.dump(-1, ' ', false, json::error_handler_t::replace);
}

std::string json_scalar_to_string(const json& val) {
    if (val.is_string()) return val.get<std::string>();
    if (val.is_null()) return {};
    return safe_json_dump(val);
}

double clamp_d(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

int clamp_i(int v, int lo, int hi) {
    return std::max(lo, std::min(hi, v));
}

std::optional<double> parse_optional_double(const json& val) {
    if (val.is_null() || val.is_boolean()) return std::nullopt;
    if (val.is_number()) return val.get<double>();
    if (val.is_string()) {
        const std::string s = util::trim_ascii(val.get<std::string>());
        if (s.empty()) return std::nullopt;
        try {
            return std::stod(s);
        } catch (...) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

std::optional<int> parse_optional_int(const json& val) {
    if (val.is_null() || val.is_boolean()) return std::nullopt;
    if (val.is_number_integer()) return val.get<int>();
    if (val.is_number_float()) return static_cast<int>(std::llround(val.get<double>()));
    if (val.is_string()) {
        const std::string s = util::trim_ascii(val.get<std::string>());
        if (s.empty()) return std::nullopt;
        try {
            return static_cast<int>(std::llround(std::stod(s)));
        } catch (...) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

std::optional<bool> parse_optional_bool(const json& val) {
    if (val.is_boolean()) return val.get<bool>();
    if (val.is_number_integer()) return val.get<int>() != 0;
    if (val.is_string()) {
        const std::string t = util::to_lower_ascii(util::trim_ascii(val.get<std::string>()));
        if (t == "1" || t == "true" || t == "yes" || t == "on") return true;
        if (t == "0" || t == "false" || t == "no" || t == "off") return false;
    }
    return std::nullopt;
}

std::string base64_encode_bytes(const uint8_t* data, size_t size) {
    static constexpr char kChars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((size + 2) / 3) * 4);
    size_t i = 0;
    while (i + 3 <= size) {
        const uint32_t triple = (static_cast<uint32_t>(data[i]) << 16) |
                                (static_cast<uint32_t>(data[i + 1]) << 8) |
                                static_cast<uint32_t>(data[i + 2]);
        out.push_back(kChars[(triple >> 18) & 0x3F]);
        out.push_back(kChars[(triple >> 12) & 0x3F]);
        out.push_back(kChars[(triple >> 6) & 0x3F]);
        out.push_back(kChars[triple & 0x3F]);
        i += 3;
    }
    const size_t rem = size - i;
    if (rem == 1) {
        const uint32_t v = static_cast<uint32_t>(data[i]) << 16;
        out.push_back(kChars[(v >> 18) & 0x3F]);
        out.push_back(kChars[(v >> 12) & 0x3F]);
        out.push_back('=');
        out.push_back('=');
    } else if (rem == 2) {
        const uint32_t v = (static_cast<uint32_t>(data[i]) << 16) |
                           (static_cast<uint32_t>(data[i + 1]) << 8);
        out.push_back(kChars[(v >> 18) & 0x3F]);
        out.push_back(kChars[(v >> 12) & 0x3F]);
        out.push_back(kChars[(v >> 6) & 0x3F]);
        out.push_back('=');
    }
    return out;
}

std::string collapse_whitespace(std::string_view sv) {
    std::string out;
    out.reserve(sv.size());
    bool in_ws = true;
    for (unsigned char c : sv) {
        if (std::isspace(c)) {
            if (!in_ws) {
                out.push_back(' ');
                in_ws = true;
            }
        } else {
            out.push_back(static_cast<char>(c));
            in_ws = false;
        }
    }
    if (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

bool starts_with_icase(std::string_view s, std::string_view prefix) {
    if (s.size() < prefix.size()) return false;
    for (size_t i = 0; i < prefix.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(s[i])) !=
            std::tolower(static_cast<unsigned char>(prefix[i]))) {
            return false;
        }
    }
    return true;
}

size_t find_icase(const std::string& haystack, const std::string& needle, size_t start = 0) {
    if (needle.empty() || start >= haystack.size() || needle.size() > haystack.size() - start) {
        return std::string::npos;
    }
    const size_t limit = haystack.size() - needle.size();
    for (size_t i = start; i <= limit; ++i) {
        bool match = true;
        for (size_t j = 0; j < needle.size(); ++j) {
            if (std::tolower(static_cast<unsigned char>(haystack[i + j])) !=
                std::tolower(static_cast<unsigned char>(needle[j]))) {
                match = false;
                break;
            }
        }
        if (match) return i;
    }
    return std::string::npos;
}

// -----------------------------------------------------------------------------
// Static Provider & Target Language Registries (Koharu-Style)
// -----------------------------------------------------------------------------

const std::vector<ProviderSpec>& all_provider_specs() {
    static const std::vector<ProviderSpec> kSpecs = {
        {
            "lmstudio", "LM Studio (Local)", kDefaultLmStudioBaseUrl,
            kDefaultLmStudioModel, kDefaultLmStudioApiKey,
            false, true, true, "openai",
            true, true, true, true, true,
            {},
        },
        {
            "ollama", "Ollama (Local)", kDefaultOllamaBaseUrl,
            "qwen2.5:7b", "ollama",
            false, true, true, "openai",
            true, true, true, true, true,
            {},
        },
        {
            "openai", "OpenAI", "https://api.openai.com/v1",
            "gpt-4o-mini", "",
            true, true, false, "openai",
            true, true, false, true, true,
            {},
        },
        {
            "gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
            "gemini-2.5-flash", "",
            true, true, false, "openai",
            true, true, false, false, true,
            {},
        },
        {
            "anthropic", "Anthropic (Claude)", "https://api.anthropic.com/v1",
            "claude-sonnet-4-20250514", "",
            true, true, false, "anthropic",
            false, true, true, false, true,
            {},
        },
        {
            "deepseek", "DeepSeek", "https://api.deepseek.com/v1",
            "deepseek-chat", "",
            true, true, false, "openai",
            false, false, false, true, false,
            {},
        },
        {
            "openrouter", "OpenRouter", "https://openrouter.ai/api/v1",
            "google/gemini-2.5-flash", "",
            true, true, false, "openai",
            true, true, true, true, true,
            {
                {"HTTP-Referer", "https://github.com/comic-translator"},
                {"X-Title", "MangaFlow Comic Translator"},
            },
        },
        {
            "custom", "Custom (OpenAI-Compatible)", "",
            "", "",
            false, true, false, "openai",
            true, true, true, true, true,
            {},
        },
    };
    return kSpecs;
}

const std::vector<TargetLanguageSpec>& all_target_languages() {
    static const std::vector<TargetLanguageSpec> kLangs = {
        {
            "id", "Indonesian", "Bahasa Indonesia", false,
            "Write natural, expressive spoken comic Indonesian (Bahasa Indonesia). "
            "Keep character tone, emotion, and casual slang natural for speech bubbles "
            "(use 'aku/kamu', 'kau', or casual 'gue/lu' when the speaker is rough/slangy; "
            "never sound stiff or machine-translated).",
            {"id", "ind", "indonesian", "bahasa", "bahasa indonesia", "indo"},
        },
        {
            "en", "English", "English", false,
            "Write natural, idiomatic comic-book English. "
            "Use punchy phrasing, natural contractions, and expressive slang that fits each character's emotion and tone.",
            {"en", "en-us", "en-gb", "eng", "english"},
        },
        {
            "ja", "Japanese", "日本語", false,
            "Write natural, expressive Japanese comic dialogue (日本語) matching each speaker's register, "
            "pronouns, and sentence-ending particles naturally for manga bubbles.",
            {"ja", "jp", "jpn", "japanese", "日本語"},
        },
        {
            "ko", "Korean", "한국어", false,
            "Write natural, expressive Korean comic dialogue (한국어) with appropriate speech levels "
            "(반말/존댓말) matching character relationships and scene tension.",
            {"ko", "kr", "kor", "korean", "한국어"},
        },
        {
            "zh-CN", "Chinese (Simplified)", "简体中文", false,
            "Write natural, idiomatic Simplified Chinese comic dialogue (简体中文) with lively pacing, "
            "authentic colloquial expressions, and tone fitting each character.",
            {"zh-cn", "zh_cn", "zh-hans", "zh_hans", "zh", "zho", "chi", "cn",
             "chinese", "chinese simplified", "simplified chinese",
             "chinese (simplified)", "简体中文", "中文"},
        },
        {
            "zh-TW", "Chinese (Traditional)", "繁體中文", false,
            "Write natural, idiomatic Traditional Chinese comic dialogue (繁體中文, Taiwan/HK comic style) "
            "with lively pacing and authentic spoken phrasing.",
            {"zh-tw", "zh_tw", "zh-hant", "zh_hant", "zh-hk", "tw",
             "chinese traditional", "traditional chinese",
             "chinese (traditional)", "繁體中文"},
        },
        {
            "es", "Spanish", "Español", false,
            "Write natural, expressive comic-book Spanish (Español) using colloquial phrasing, "
            "appropriate register (tú/usted), and punchy dialogue suited for speech bubbles.",
            {"es", "es-es", "es-mx", "es-419", "spa", "spanish", "español", "espanol"},
        },
        {
            "fr", "French", "Français", false,
            "Write natural, expressive comic-book French (Français) with authentic spoken tone, "
            "appropriate tutoiement/vouvoiement, and crisp bubble-friendly phrasing.",
            {"fr", "fr-fr", "fra", "fre", "french", "français", "francais"},
        },
        {
            "de", "German", "Deutsch", false,
            "Write natural, idiomatic comic-book German (Deutsch) with lively spoken phrasing, "
            "natural du/Sie register, and punchy dialogue.",
            {"de", "de-de", "deu", "ger", "german", "deutsch"},
        },
        {
            "pt", "Portuguese", "Português", false,
            "Write natural, expressive comic-book Portuguese (Português) with lively colloquial phrasing "
            "and authentic emotional nuance suitable for speech bubbles.",
            {"pt", "pt-br", "pt-pt", "por", "portuguese", "português", "portugues"},
        },
        {
            "ru", "Russian", "Русский", false,
            "Write natural, expressive comic-book Russian (Русский) with vivid colloquial tone, "
            "appropriate ты/вы register, and punchy dialogue.",
            {"ru", "rus", "russian", "русский"},
        },
        {
            "vi", "Vietnamese", "Tiếng Việt", false,
            "Write natural, expressive comic-book Vietnamese (Tiếng Việt) using authentic pronouns "
            "(tôi/cậu, tao/mày, anh/em, etc.) that accurately reflect character relationships and emotion.",
            {"vi", "vn", "vie", "vietnamese", "tiếng việt", "tieng viet"},
        },
        {
            "th", "Thai", "ไทย", false,
            "Write natural, expressive comic-book Thai (ภาษาไทย) with fitting pronouns, particles "
            "(ครับ/ค่ะ/นะ/เว้ย), and lively colloquial rhythm for comic bubbles.",
            {"th", "tha", "thai", "ไทย", "ภาษาไทย"},
        },
        {
            "ms", "Malay", "Bahasa Melayu", false,
            "Write natural, expressive comic-book Malay (Bahasa Melayu) using natural spoken phrasing "
            "(aku/kau, saya/awak) matching each character's tone and emotion.",
            {"ms", "my", "msa", "may", "malay", "bahasa melayu", "melayu"},
        },
        {
            "ar", "Arabic", "العربية", true,
            "Write natural, expressive comic-book Arabic (العربية) with clear, vivid, and concise phrasing "
            "suitable for comic speech bubbles while capturing each character's emotion.",
            {"ar", "ara", "arabic", "العربية"},
        },
        {
            "it", "Italian", "Italiano", false,
            "Write natural, expressive comic-book Italian (Italiano) with idiomatic spoken tone, "
            "appropriate tu/Lei register, and crisp bubble-friendly dialogue.",
            {"it", "ita", "italian", "italiano"},
        },
    };
    return kLangs;
}

// -----------------------------------------------------------------------------
// Dual HTTP (cpp-httplib) & HTTPS (WinHTTP) Transport Layer
// -----------------------------------------------------------------------------

struct ParsedUrl {
    bool valid = false;
    bool is_https = false;
    std::string host;
    int port = 80;
    std::string path_and_query = "/";
};

ParsedUrl parse_http_url(const std::string& raw_url) {
    ParsedUrl out;
    std::string s = util::trim_ascii(raw_url);
    if (s.empty()) return out;

    if (starts_with_icase(s, "https://")) {
        out.is_https = true;
        out.port = 443;
        s = s.substr(8);
    } else if (starts_with_icase(s, "http://")) {
        out.is_https = false;
        out.port = 80;
        s = s.substr(7);
    } else {
        out.is_https = false;
        out.port = 80;
    }

    std::string authority;
    const size_t slash_pos = s.find('/');
    const size_t query_pos = s.find('?');
    size_t split_pos = std::string::npos;
    if (slash_pos != std::string::npos && query_pos != std::string::npos) {
        split_pos = std::min(slash_pos, query_pos);
    } else if (slash_pos != std::string::npos) {
        split_pos = slash_pos;
    } else if (query_pos != std::string::npos) {
        split_pos = query_pos;
    }

    if (split_pos != std::string::npos) {
        authority = s.substr(0, split_pos);
        out.path_and_query = s.substr(split_pos);
        if (out.path_and_query.empty() || out.path_and_query[0] != '/') {
            out.path_and_query = "/" + out.path_and_query;
        }
    } else {
        authority = s;
        out.path_and_query = "/";
    }

    if (authority.empty()) return out;

    // Handle bracketed IPv6 host `[::1]:1235`
    if (authority.front() == '[') {
        const size_t close_br = authority.find(']');
        if (close_br == std::string::npos) return out;
        out.host = authority.substr(1, close_br - 1);
        if (close_br + 1 < authority.size() && authority[close_br + 1] == ':') {
            try {
                out.port = std::stoi(authority.substr(close_br + 2));
            } catch (...) {
                return out;
            }
        }
    } else {
        const size_t colon = authority.rfind(':');
        if (colon != std::string::npos) {
            out.host = authority.substr(0, colon);
            try {
                out.port = std::stoi(authority.substr(colon + 1));
            } catch (...) {
                return out;
            }
        } else {
            out.host = authority;
        }
    }

    if (out.host.empty() || out.port <= 0 || out.port > 65535) {
        return out;
    }
    out.valid = true;
    return out;
}

struct HttpResponse {
    int status_code = 0;
    std::string body;
    bool transport_error = false;
    bool timed_out = false;
    std::string error_message;
};

HttpResponse http_request_cpp_httplib(
    const std::string& method,
    const ParsedUrl& parsed,
    const std::vector<std::pair<std::string, std::string>>& headers,
    const std::string& body,
    double timeout_seconds) {
    auto run_once = [&](const std::string& connect_host) -> HttpResponse {
        HttpResponse out;
        try {
            httplib::Client client(connect_host, parsed.port);
            const double conn_sec = std::min(10.0, std::max(1.0, timeout_seconds));
            const int conn_s = static_cast<int>(conn_sec);
            const int conn_us = static_cast<int>((conn_sec - conn_s) * 1'000'000.0);
            const int rw_s = static_cast<int>(std::max(1.0, timeout_seconds));
            const int rw_us = static_cast<int>((std::max(1.0, timeout_seconds) - rw_s) * 1'000'000.0);

            client.set_connection_timeout(conn_s, conn_us);
            client.set_read_timeout(rw_s, rw_us);
            client.set_write_timeout(std::max(10, rw_s), rw_us);
            client.set_follow_location(method == "GET");

            httplib::Headers hmap;
            std::string content_type = "application/json";
            for (const auto& [k, v] : headers) {
                if (util::to_lower_ascii(k) == "content-type") {
                    content_type = v;
                } else {
                    hmap.emplace(k, v);
                }
            }

            httplib::Result res = (method == "POST")
                ? client.Post(parsed.path_and_query, hmap, body, content_type)
                : client.Get(parsed.path_and_query, hmap);

            if (!res) {
                out.transport_error = true;
                const auto err = res.error();
                out.timed_out = (err == httplib::Error::ConnectionTimeout ||
                                 err == httplib::Error::Read ||
                                 err == httplib::Error::Write);
                out.error_message = httplib::to_string(err);
                return out;
            }

            out.status_code = res->status;
            out.body = std::move(res->body);
            return out;
        } catch (const std::exception& ex) {
            out.transport_error = true;
            out.error_message = ex.what();
            return out;
        }
    };

    const bool is_localhost = (util::to_lower_ascii(parsed.host) == "localhost");
    if (is_localhost) {
        // Prefer IPv4 loopback on Windows so local servers don't stall on ::1
        HttpResponse v4 = run_once("127.0.0.1");
        if (!v4.transport_error) {
            return v4;
        }
        HttpResponse fallback = run_once("localhost");
        return fallback.transport_error ? v4 : fallback;
    }

    return run_once(parsed.host);
}

HttpResponse http_request_winhttp(
    const std::string& method,
    const ParsedUrl& parsed,
    const std::vector<std::pair<std::string, std::string>>& headers,
    const std::string& body,
    double timeout_seconds) {
    HttpResponse out;

    const int total_ms = clamp_i(static_cast<int>(std::llround(timeout_seconds * 1000.0)), 1000, 300000);
    const int resolve_ms = std::min(15000, total_ms);
    const int connect_ms = std::min(15000, total_ms);
    const int send_ms = std::max(30000, total_ms);
    const int receive_ms = total_ms;

    WinHttpHandle session(WinHttpOpen(
        L"MangaFlowAI/2.0",
        WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
        WINHTTP_NO_PROXY_NAME,
        WINHTTP_NO_PROXY_BYPASS,
        0));
    if (!session) {
        out.transport_error = true;
        out.error_message = "WinHttpOpen failed (error " + std::to_string(GetLastError()) + ")";
        return out;
    }

    WinHttpSetTimeouts(session.h, resolve_ms, connect_ms, send_ms, receive_ms);

    const std::wstring whost = util::utf8_to_wide(parsed.host);
    WinHttpHandle connect(WinHttpConnect(
        session.h,
        whost.c_str(),
        static_cast<INTERNET_PORT>(parsed.port),
        0));
    if (!connect) {
        out.transport_error = true;
        out.error_message = "WinHttpConnect failed (error " + std::to_string(GetLastError()) + ")";
        return out;
    }

    const std::wstring wmethod = util::utf8_to_wide(method);
    const std::wstring wpath = util::utf8_to_wide(parsed.path_and_query);
    const DWORD flags = parsed.is_https ? WINHTTP_FLAG_SECURE : 0;

    WinHttpHandle request(WinHttpOpenRequest(
        connect.h,
        wmethod.c_str(),
        wpath.c_str(),
        nullptr,
        WINHTTP_NO_REFERER,
        WINHTTP_DEFAULT_ACCEPT_TYPES,
        flags));
    if (!request) {
        out.transport_error = true;
        out.error_message = "WinHttpOpenRequest failed (error " + std::to_string(GetLastError()) + ")";
        return out;
    }

    // Apply timeouts explicitly to the request handle as well (overrides 30s default receive timeout)
    WinHttpSetTimeouts(request.h, resolve_ms, connect_ms, send_ms, receive_ms);

    if (method == "GET") {
        DWORD redirect_policy = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
        WinHttpSetOption(
            request.h, WINHTTP_OPTION_REDIRECT_POLICY, &redirect_policy, sizeof(redirect_policy));
    }

    std::string raw_header_block;
    for (const auto& [k, v] : headers) {
        raw_header_block += k;
        raw_header_block += ": ";
        raw_header_block += v;
        raw_header_block += "\r\n";
    }
    if (!raw_header_block.empty()) {
        const std::wstring wheaders = util::utf8_to_wide(raw_header_block);
        WinHttpAddRequestHeaders(
            request.h,
            wheaders.c_str(),
            static_cast<DWORD>(wheaders.size()),
            WINHTTP_ADDREQ_FLAG_ADD | WINHTTP_ADDREQ_FLAG_REPLACE);
    }

    const DWORD total_len = static_cast<DWORD>(body.size());
    if (!WinHttpSendRequest(
            request.h,
            WINHTTP_NO_ADDITIONAL_HEADERS,
            0,
            WINHTTP_NO_REQUEST_DATA,
            0,
            total_len,
            0)) {
        const DWORD err = GetLastError();
        out.transport_error = true;
        out.timed_out = (err == ERROR_WINHTTP_TIMEOUT);
        out.error_message = "WinHttpSendRequest failed (error " + std::to_string(err) + ")";
        return out;
    }

    // Stream request payload in 64 KiB chunks (handles large base64 vision payloads reliably)
    size_t sent = 0;
    while (sent < body.size()) {
        const DWORD chunk = static_cast<DWORD>(
            std::min<size_t>(65536ULL, body.size() - sent));
        DWORD written = 0;
        if (!WinHttpWriteData(request.h, body.data() + sent, chunk, &written)) {
            const DWORD err = GetLastError();
            out.transport_error = true;
            out.timed_out = (err == ERROR_WINHTTP_TIMEOUT);
            out.error_message = "WinHttpWriteData failed (error " + std::to_string(err) + ")";
            return out;
        }
        if (written == 0) {
            break;
        }
        sent += written;
    }

    if (!WinHttpReceiveResponse(request.h, nullptr)) {
        const DWORD err = GetLastError();
        out.transport_error = true;
        out.timed_out = (err == ERROR_WINHTTP_TIMEOUT);
        out.error_message = "WinHttpReceiveResponse failed (error " + std::to_string(err) + ")";
        return out;
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
    out.status_code = static_cast<int>(status);

    while (true) {
        DWORD avail = 0;
        if (!WinHttpQueryDataAvailable(request.h, &avail)) {
            const DWORD err = GetLastError();
            out.transport_error = true;
            out.timed_out = (err == ERROR_WINHTTP_TIMEOUT);
            out.error_message = "WinHttpQueryDataAvailable failed (error " + std::to_string(err) + ")";
            return out;
        }
        if (avail == 0) {
            break;
        }
        if (out.body.size() + avail > kMaxHttpResponseBytes) {
            out.transport_error = true;
            out.error_message = "HTTP response exceeded 32 MiB limit.";
            return out;
        }
        const size_t prev_size = out.body.size();
        out.body.resize(prev_size + avail);
        DWORD read_bytes = 0;
        if (!WinHttpReadData(request.h, out.body.data() + prev_size, avail, &read_bytes)) {
            const DWORD err = GetLastError();
            out.transport_error = true;
            out.timed_out = (err == ERROR_WINHTTP_TIMEOUT);
            out.error_message = "WinHttpReadData failed (error " + std::to_string(err) + ")";
            return out;
        }
        out.body.resize(prev_size + read_bytes);
    }

    return out;
}

HttpResponse perform_http_request(
    const std::string& method,
    const std::string& full_url,
    const std::vector<std::pair<std::string, std::string>>& headers,
    const std::string& body,
    double timeout_seconds) {
    const ParsedUrl parsed = parse_http_url(full_url);
    if (!parsed.valid) {
        HttpResponse bad;
        bad.transport_error = true;
        bad.error_message = "Invalid URL: '" + full_url + "'";
        return bad;
    }

    if (parsed.is_https) {
        return http_request_winhttp(method, parsed, headers, body, timeout_seconds);
    }
    return http_request_cpp_httplib(method, parsed, headers, body, timeout_seconds);
}

std::string extract_http_error_detail(const HttpResponse& resp) {
    const json parsed = json::parse(resp.body, nullptr, false);
    if (!parsed.is_discarded() && parsed.is_object()) {
        if (parsed.contains("error")) {
            const auto& err = parsed["error"];
            if (err.is_object() && err.contains("message") && err["message"].is_string()) {
                const std::string msg = util::trim_ascii(err["message"].get<std::string>());
                if (!msg.empty()) return msg;
            } else if (err.is_string()) {
                const std::string msg = util::trim_ascii(err.get<std::string>());
                if (!msg.empty()) return msg;
            }
        }
        for (const char* k : {"message", "detail"}) {
            if (parsed.contains(k) && parsed[k].is_string()) {
                const std::string msg = util::trim_ascii(parsed[k].get<std::string>());
                if (!msg.empty()) return msg;
            }
        }
    }

    std::string snippet = collapse_whitespace(resp.body);
    if (snippet.size() > 240) {
        snippet.resize(240);
    }
    return snippet.empty() ? ("HTTP " + std::to_string(resp.status_code)) : snippet;
}

std::string strip_trailing_v1(const std::string& base_url) {
    std::string s = util::trim_ascii(base_url);
    while (!s.empty() && s.back() == '/') {
        s.pop_back();
    }
    const std::string lower = util::to_lower_ascii(s);
    if (lower.size() >= 3 && lower.compare(lower.size() - 3, 3, "/v1") == 0) {
        s.resize(s.size() - 3);
    }
    while (!s.empty() && s.back() == '/') {
        s.pop_back();
    }
    return s;
}

std::pair<std::optional<std::string>, std::unordered_set<std::string>>
fetch_lmstudio_v0_metadata(const std::string& base_url, double timeout_seconds) {
    std::optional<std::string> loaded_model;
    std::unordered_set<std::string> vlm_models;

    const std::string root = strip_trailing_v1(base_url);
    if (root.empty()) {
        return {loaded_model, vlm_models};
    }

    const HttpResponse resp = perform_http_request(
        "GET",
        root + "/api/v0/models",
        {{"Accept", "application/json"}},
        "",
        std::min(3.0, std::max(0.5, timeout_seconds)));

    if (resp.transport_error || resp.status_code != 200) {
        return {loaded_model, vlm_models};
    }

    const json payload = json::parse(resp.body, nullptr, false);
    if (payload.is_discarded() || !payload.is_object() ||
        !payload.contains("data") || !payload["data"].is_array()) {
        return {loaded_model, vlm_models};
    }

    for (const auto& entry : payload["data"]) {
        if (!entry.is_object()) continue;
        const std::string model_id = entry.contains("id") && entry["id"].is_string()
            ? util::trim_ascii(entry["id"].get<std::string>())
            : "";
        if (model_id.empty()) continue;

        const std::string state = entry.contains("state") && entry["state"].is_string()
            ? util::to_lower_ascii(util::trim_ascii(entry["state"].get<std::string>()))
            : "";
        const std::string model_type = entry.contains("type") && entry["type"].is_string()
            ? util::to_lower_ascii(util::trim_ascii(entry["type"].get<std::string>()))
            : "";

        if (model_type == "embeddings" || ComicTranslator::is_embedding_model(model_id)) {
            continue;
        }
        if (model_type == "vlm") {
            vlm_models.insert(model_id);
        }
        if (!loaded_model.has_value() && state == "loaded") {
            loaded_model = model_id;
        }
    }

    return {loaded_model, vlm_models};
}

std::vector<std::string> fetch_ollama_native_tags(
    const std::string& base_url,
    double timeout_seconds) {
    std::vector<std::string> names;
    const std::string root = strip_trailing_v1(base_url);
    if (root.empty()) return names;

    const HttpResponse resp = perform_http_request(
        "GET",
        root + "/api/tags",
        {{"Accept", "application/json"}},
        "",
        timeout_seconds);
    if (resp.transport_error || resp.status_code != 200) {
        return names;
    }

    const json payload = json::parse(resp.body, nullptr, false);
    if (payload.is_discarded() || !payload.is_object() ||
        !payload.contains("models") || !payload["models"].is_array()) {
        return names;
    }

    for (const auto& entry : payload["models"]) {
        if (!entry.is_object()) continue;
        std::string name;
        if (entry.contains("name") && entry["name"].is_string()) {
            name = util::trim_ascii(entry["name"].get<std::string>());
        } else if (entry.contains("model") && entry["model"].is_string()) {
            name = util::trim_ascii(entry["model"].get<std::string>());
        }
        if (!name.empty()) {
            names.push_back(std::move(name));
        }
    }
    return names;
}

// -----------------------------------------------------------------------------
// Model Tokenization & Fuzzy Scoring Helpers
// -----------------------------------------------------------------------------

std::vector<std::string> tokenize_model_name(const std::string& name) {
    std::vector<std::string> tokens;
    std::string cur;
    for (unsigned char c : name) {
        if (std::isalnum(c)) {
            cur.push_back(static_cast<char>(std::tolower(c)));
        } else if (!cur.empty()) {
            tokens.push_back(std::move(cur));
            cur.clear();
        }
    }
    if (!cur.empty()) {
        tokens.push_back(std::move(cur));
    }
    return tokens;
}

std::string compact_alnum_lower(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (std::isalnum(c)) {
            out.push_back(static_cast<char>(std::tolower(c)));
        }
    }
    return out;
}

// -----------------------------------------------------------------------------
// Vision Image Resizing & JPEG Encoding Helpers
// -----------------------------------------------------------------------------

std::vector<uint8_t> resize_rgb_bilinear(
    const uint8_t* src,
    int src_w,
    int src_h,
    int dst_w,
    int dst_h) {
    std::vector<uint8_t> dst(static_cast<size_t>(dst_w) * static_cast<size_t>(dst_h) * 3ULL, 0);
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        return dst;
    }
    if (src_w == dst_w && src_h == dst_h) {
        std::memcpy(dst.data(), src, dst.size());
        return dst;
    }

    const float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);

    for (int dy = 0; dy < dst_h; ++dy) {
        const float sy = (static_cast<float>(dy) + 0.5f) * scale_y - 0.5f;
        const int y0 = clamp_i(static_cast<int>(std::floor(sy)), 0, src_h - 1);
        const int y1 = std::min(src_h - 1, y0 + 1);
        const float wy = std::max(0.0f, std::min(1.0f, sy - static_cast<float>(y0)));

        const uint8_t* row0 = src + static_cast<size_t>(y0) * static_cast<size_t>(src_w) * 3ULL;
        const uint8_t* row1 = src + static_cast<size_t>(y1) * static_cast<size_t>(src_w) * 3ULL;
        uint8_t* dst_row = dst.data() + static_cast<size_t>(dy) * static_cast<size_t>(dst_w) * 3ULL;

        for (int dx = 0; dx < dst_w; ++dx) {
            const float sx = (static_cast<float>(dx) + 0.5f) * scale_x - 0.5f;
            const int x0 = clamp_i(static_cast<int>(std::floor(sx)), 0, src_w - 1);
            const int x1 = std::min(src_w - 1, x0 + 1);
            const float wx = std::max(0.0f, std::min(1.0f, sx - static_cast<float>(x0)));

            const uint8_t* p00 = row0 + static_cast<size_t>(x0) * 3ULL;
            const uint8_t* p01 = row0 + static_cast<size_t>(x1) * 3ULL;
            const uint8_t* p10 = row1 + static_cast<size_t>(x0) * 3ULL;
            const uint8_t* p11 = row1 + static_cast<size_t>(x1) * 3ULL;

            for (int c = 0; c < 3; ++c) {
                const float top = static_cast<float>(p00[c]) * (1.0f - wx) +
                                  static_cast<float>(p01[c]) * wx;
                const float bot = static_cast<float>(p10[c]) * (1.0f - wx) +
                                  static_cast<float>(p11[c]) * wx;
                const float val = top * (1.0f - wy) + bot * wy;
                dst_row[static_cast<size_t>(dx) * 3ULL + static_cast<size_t>(c)] =
                    static_cast<uint8_t>(clamp_i(static_cast<int>(std::lround(val)), 0, 255));
            }
        }
    }
    return dst;
}

void stbi_write_vec_cb(void* ctx, void* data, int size) {
    if (!ctx || !data || size <= 0) return;
    auto* vec = static_cast<std::vector<uint8_t>*>(ctx);
    const auto* bytes = static_cast<const uint8_t*>(data);
    vec->insert(vec->end(), bytes, bytes + size);
}

// Parse `data:<mime>;base64,<payload>` -> `{media_type, clean_b64}`
std::optional<std::pair<std::string, std::string>> parse_image_data_uri(const std::string& uri) {
    const std::string trimmed = util::trim_ascii(uri);
    if (!starts_with_icase(trimmed, "data:image/")) {
        return std::nullopt;
    }
    const size_t semi = trimmed.find(';');
    const size_t comma = trimmed.find(',');
    if (semi == std::string::npos || comma == std::string::npos || comma < semi) {
        return std::nullopt;
    }
    const std::string mime = trimmed.substr(5, semi - 5);
    const std::string meta = util::to_lower_ascii(trimmed.substr(semi + 1, comma - semi - 1));
    if (meta.find("base64") == std::string::npos) {
        return std::nullopt;
    }
    std::string b64;
    b64.reserve(trimmed.size() - comma - 1);
    for (size_t i = comma + 1; i < trimmed.size(); ++i) {
        if (!std::isspace(static_cast<unsigned char>(trimmed[i]))) {
            b64.push_back(trimmed[i]);
        }
    }
    if (b64.empty()) return std::nullopt;
    return std::make_pair(mime, std::move(b64));
}

// -----------------------------------------------------------------------------
// Pure C++ O(N) JSON Cleaning, Fence Stripping & Salvage Helpers
// -----------------------------------------------------------------------------

std::string remove_paired_tag_icase(
    const std::string& input,
    const std::string& open_tag,
    const std::string& close_tag) {
    std::string out;
    out.reserve(input.size());
    size_t pos = 0;
    while (pos < input.size()) {
        const size_t start = find_icase(input, open_tag, pos);
        if (start == std::string::npos) {
            out.append(input, pos, std::string::npos);
            break;
        }
        out.append(input, pos, start - pos);
        const size_t close_pos = find_icase(input, close_tag, start + open_tag.size());
        if (close_pos == std::string::npos) {
            // Unclosed tag: keep remainder intact for the unclosed-tag pass to inspect
            out.append(input, start, std::string::npos);
            break;
        }
        pos = close_pos + close_tag.size();
    }
    return out;
}

// Remove trailing commas immediately before `}` or `]` outside JSON string literals.
std::string remove_trailing_json_commas(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool in_str = false;
    bool escaped = false;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (in_str) {
            out.push_back(c);
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_str = false;
            }
            continue;
        }
        if (c == '"') {
            in_str = true;
            out.push_back(c);
            continue;
        }
        if (c == ',') {
            size_t j = i + 1;
            while (j < s.size() && std::isspace(static_cast<unsigned char>(s[j]))) {
                ++j;
            }
            if (j < s.size() && (s[j] == '}' || s[j] == ']')) {
                continue;  // drop trailing comma
            }
        }
        out.push_back(c);
    }
    return out;
}

std::string strip_fences_and_noise(const std::string& think_stripped) {
    std::string text = util::trim_ascii(think_stripped);
    if (text.empty()) return {};

    // Extract markdown ```json ... ``` or ``` ... ``` code fence if present
    const size_t fence_start = text.find("```");
    if (fence_start != std::string::npos) {
        size_t body_start = fence_start + 3;
        // Skip optional language label (e.g. `json`) up to newline or whitespace
        while (body_start < text.size() &&
               (std::isalpha(static_cast<unsigned char>(text[body_start])) ||
                text[body_start] == ' ' || text[body_start] == '\t')) {
            ++body_start;
        }
        if (body_start < text.size() && text[body_start] == '\r') ++body_start;
        if (body_start < text.size() && text[body_start] == '\n') ++body_start;

        const size_t fence_end = text.find("```", body_start);
        if (fence_end != std::string::npos && fence_end > body_start) {
            const std::string fenced = util::trim_ascii(text.substr(body_start, fence_end - body_start));
            if (!fenced.empty()) {
                text = fenced;
            }
        }
    }

    const size_t first_brace = text.find('{');
    const size_t first_bracket = text.find('[');
    size_t start_idx = std::string::npos;
    if (first_brace != std::string::npos && first_bracket != std::string::npos) {
        start_idx = std::min(first_brace, first_bracket);
    } else if (first_brace != std::string::npos) {
        start_idx = first_brace;
    } else if (first_bracket != std::string::npos) {
        start_idx = first_bracket;
    }

    if (start_idx != std::string::npos) {
        const size_t last_brace = text.rfind('}');
        const size_t last_bracket = text.rfind(']');
        size_t end_idx = std::string::npos;
        if (last_brace != std::string::npos && last_bracket != std::string::npos) {
            end_idx = std::max(last_brace, last_bracket);
        } else if (last_brace != std::string::npos) {
            end_idx = last_brace;
        } else if (last_bracket != std::string::npos) {
            end_idx = last_bracket;
        }
        if (end_idx != std::string::npos && end_idx > start_idx) {
            text = text.substr(start_idx, end_idx - start_idx + 1);
        }
    }

    text = remove_trailing_json_commas(text);
    return util::trim_ascii(text);
}

void extract_mapping_from_parsed_json(
    const json& parsed,
    std::vector<std::pair<std::string, std::string>>& ordered_out) {
    auto add_pair = [&](const std::string& k, const std::string& v) {
        const std::string ck = util::trim_ascii(k);
        const std::string cv = util::trim_ascii(v);
        if (ck.empty()) return;
        for (auto& existing : ordered_out) {
            if (existing.first == ck) {
                existing.second = cv;
                return;
            }
        }
        ordered_out.emplace_back(ck, cv);
    };

    static const std::array<const char*, 4> kTextKeys = {
        "translated_text", "translation", "text", "target_text"
    };

    if (parsed.is_object()) {
        for (const char* list_key : {"dialogues", "translations", "blocks", "items", "data", "results"}) {
            if (parsed.contains(list_key) &&
                (parsed[list_key].is_array() || parsed[list_key].is_object())) {
                std::vector<std::pair<std::string, std::string>> sub;
                extract_mapping_from_parsed_json(parsed[list_key], sub);
                if (!sub.empty()) {
                    ordered_out = std::move(sub);
                    return;
                }
            }
        }

        if (parsed.contains("id")) {
            const std::string item_id = util::trim_ascii(json_scalar_to_string(parsed["id"]));
            for (const char* tkey : kTextKeys) {
                if (parsed.contains(tkey) && parsed[tkey].is_string() && !item_id.empty()) {
                    add_pair(item_id, parsed[tkey].get<std::string>());
                    return;
                }
            }
        }

        for (auto it = parsed.begin(); it != parsed.end(); ++it) {
            const std::string k = util::trim_ascii(it.key());
            if (k.empty()) continue;
            if (it.value().is_string()) {
                add_pair(k, it.value().get<std::string>());
            } else if (it.value().is_object()) {
                for (const char* tkey : kTextKeys) {
                    if (it.value().contains(tkey) && it.value()[tkey].is_string()) {
                        add_pair(k, it.value()[tkey].get<std::string>());
                        break;
                    }
                }
            }
        }
        return;
    }

    if (parsed.is_array()) {
        for (const auto& entry : parsed) {
            if (!entry.is_object()) continue;
            if (entry.contains("id") && !entry["id"].is_null()) {
                const std::string item_id = util::trim_ascii(json_scalar_to_string(entry["id"]));
                if (!item_id.empty()) {
                    for (const char* tkey : kTextKeys) {
                        if (entry.contains(tkey) && entry[tkey].is_string()) {
                            add_pair(item_id, entry[tkey].get<std::string>());
                            break;
                        }
                    }
                }
            } else if (entry.size() == 1) {
                const auto it = entry.begin();
                if (it.value().is_string()) {
                    add_pair(it.key(), it.value().get<std::string>());
                }
            }
        }
    }
}

// Parse a JSON string literal starting at `quote_pos` (where `s[quote_pos] == '"'`),
// advancing `end_pos` past the closing quote. Returns std::nullopt if malformed.
std::optional<std::string> read_json_string_literal(
    const std::string& s,
    size_t quote_pos,
    size_t& end_pos) {
    if (quote_pos >= s.size() || s[quote_pos] != '"') {
        return std::nullopt;
    }
    size_t i = quote_pos + 1;
    bool escaped = false;
    while (i < s.size()) {
        const char c = s[i];
        if (escaped) {
            escaped = false;
        } else if (c == '\\') {
            escaped = true;
        } else if (c == '"') {
            const std::string literal = s.substr(quote_pos, i - quote_pos + 1);
            end_pos = i + 1;
            const json parsed = json::parse(literal, nullptr, false);
            if (!parsed.is_discarded() && parsed.is_string()) {
                return util::trim_ascii(parsed.get<std::string>());
            }
            // Fallback manual unescape
            std::string raw = s.substr(quote_pos + 1, i - quote_pos - 1);
            std::string unesc;
            unesc.reserve(raw.size());
            for (size_t k = 0; k < raw.size(); ++k) {
                if (raw[k] == '\\' && k + 1 < raw.size()) {
                    const char nxt = raw[++k];
                    if (nxt == 'n' || nxt == 'r') unesc.push_back(' ');
                    else if (nxt == 't') unesc.push_back(' ');
                    else unesc.push_back(nxt);
                } else {
                    unesc.push_back(raw[k]);
                }
            }
            return util::trim_ascii(unesc);
        }
        ++i;
    }
    return std::nullopt;
}

size_t skip_ws(const std::string& s, size_t pos) {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
        ++pos;
    }
    return pos;
}

// Pure C++ salvage extractor for truncated/malformed LLM JSON output (zero regex backtracking).
std::vector<std::pair<std::string, std::string>> salvage_translations_from_text(
    const std::string& text,
    const std::vector<std::string>& expected_ids) {
    std::vector<std::pair<std::string, std::string>> salvaged;
    auto upsert = [&](const std::string& k, const std::string& v) {
        if (k.empty()) return;
        for (auto& p : salvaged) {
            if (p.first == k) {
                p.second = v;
                return;
            }
        }
        salvaged.emplace_back(k, v);
    };

    static const std::array<std::string, 4> kFieldNames = {
        "\"translated_text\"", "\"translation\"", "\"text\"", "\"target_text\""
    };

    // 1. Scan for `"id" : "<id>" ... "<translated_text|...>" : "<val>"`
    size_t pos = 0;
    while (pos < text.size()) {
        const size_t id_key = text.find("\"id\"", pos);
        if (id_key == std::string::npos) break;
        size_t cur = skip_ws(text, id_key + 4);
        if (cur >= text.size() || text[cur] != ':') {
            pos = id_key + 4;
            continue;
        }
        cur = skip_ws(text, cur + 1);
        size_t after_id = cur;
        auto bid_opt = read_json_string_literal(text, cur, after_id);
        if (!bid_opt.has_value() || bid_opt->empty()) {
            pos = id_key + 4;
            continue;
        }

        // Find the next object boundary or next `"id"` key so we don't pair across entries
        const size_t next_id = text.find("\"id\"", after_id);
        const size_t search_end = (next_id == std::string::npos) ? text.size() : next_id;

        size_t best_field_pos = std::string::npos;
        size_t best_field_len = 0;
        for (const auto& fn : kFieldNames) {
            const size_t fp = text.find(fn, after_id);
            if (fp != std::string::npos && fp < search_end &&
                (best_field_pos == std::string::npos || fp < best_field_pos)) {
                best_field_pos = fp;
                best_field_len = fn.size();
            }
        }

        if (best_field_pos != std::string::npos) {
            size_t vc = skip_ws(text, best_field_pos + best_field_len);
            if (vc < search_end && text[vc] == ':') {
                vc = skip_ws(text, vc + 1);
                size_t after_val = vc;
                if (auto val_opt = read_json_string_literal(text, vc, after_val)) {
                    upsert(*bid_opt, *val_opt);
                    pos = after_val;
                    continue;
                }
            }
        }
        pos = after_id;
    }

    // 2. Scan for direct `"<expected_id>" : "<val>"`
    for (const auto& exp_id : expected_ids) {
        bool already = false;
        for (const auto& p : salvaged) {
            if (p.first == exp_id && !p.second.empty()) {
                already = true;
                break;
            }
        }
        if (already) continue;

        const std::string needle = "\"" + exp_id + "\"";
        size_t p = 0;
        while ((p = text.find(needle, p)) != std::string::npos) {
            size_t cur = skip_ws(text, p + needle.size());
            if (cur < text.size() && text[cur] == ':') {
                cur = skip_ws(text, cur + 1);
                size_t after_val = cur;
                if (auto val_opt = read_json_string_literal(text, cur, after_val)) {
                    upsert(exp_id, *val_opt);
                    break;
                }
            }
            p += needle.size();
        }
    }

    return salvaged;
}

bool is_ascii_word_char(unsigned char c) {
    return std::isalnum(c) || c == '_';
}

bool is_pure_ascii_alnum_term(const std::string& term) {
    if (term.empty()) return false;
    for (unsigned char c : term) {
        if (c >= 0x80) return false;
        if (!std::isalnum(c) && c != '_' && c != '-' && c != ' ') return false;
    }
    return true;
}

}  // namespace

// -----------------------------------------------------------------------------
// JSON Serializers
// -----------------------------------------------------------------------------

json TranslatedDialogueItem::to_json() const {
    return {
        {"id", id},
        {"source_text", source_text},
        {"translated_text", translated_text},
    };
}

json PageTranslationResult::to_json() const {
    json diag_arr = json::array();
    for (const auto& d : dialogues) {
        diag_arr.push_back(d.to_json());
    }
    return {
        {"provider", provider},
        {"model", model},
        {"source_language", source_language},
        {"target_language", target_language},
        {"vision_used", vision_used},
        {"reasoning_effort", reasoning_effort},
        {"translations", translations},
        {"dialogues", std::move(diag_arr)},
    };
}

json ProviderTestResult::to_json() const {
    json out = {
        {"ok", ok},
        {"connected", connected},
        {"provider", provider},
        {"provider_name", provider_name},
        {"endpoint", endpoint},
        {"active_model", active_model.has_value() ? json(*active_model) : json(nullptr)},
        {"resolved_model_id", resolved_model_id.has_value() ? json(*resolved_model_id) : json(nullptr)},
        {"preferred_model", preferred_model},
        {"models", models},
        {"available_models", models},
        {"vision_models", vision_models},
        {"model_count", models.size()},
        {"latency_ms", latency_ms},
        {"message", message},
    };
    if (error.has_value() && !error->empty()) {
        out["error"] = *error;
    }
    return out;
}

json LmStudioStatusResult::to_json() const {
    json out = {
        {"connected", connected},
        {"endpoint", endpoint},
        {"active_model", active_model.has_value() ? json(*active_model) : json(nullptr)},
        {"resolved_model_id", resolved_model_id.has_value() ? json(*resolved_model_id) : json(nullptr)},
        {"preferred_model", preferred_model},
        {"available_models", available_models},
    };
    if (error.has_value() && !error->empty()) {
        out["error"] = *error;
    }
    return out;
}

// -----------------------------------------------------------------------------
// TranslatorConfig Builders
// -----------------------------------------------------------------------------

TranslatorConfig TranslatorConfig::from_settings(const json& settings) {
    TranslatorConfig cfg;
    if (!settings.is_object()) {
        return cfg;
    }

    if (settings.contains("provider") && settings["provider"].is_string()) {
        cfg.provider = ComicTranslator::normalize_provider_id(settings["provider"].get<std::string>());
    }
    const auto& spec = ComicTranslator::get_provider_spec(cfg.provider);

    cfg.base_url = settings.contains("base_url") && settings["base_url"].is_string()
        ? ComicTranslator::normalize_provider_base_url(cfg.provider, settings["base_url"].get<std::string>())
        : spec.default_base_url;
    cfg.api_key = settings.contains("api_key") && settings["api_key"].is_string()
        ? ComicTranslator::resolve_provider_api_key(cfg.provider, settings["api_key"].get<std::string>())
        : spec.default_api_key;
    cfg.preferred_model = settings.contains("model") && settings["model"].is_string()
        ? util::trim_ascii(settings["model"].get<std::string>())
        : "";
    if (cfg.preferred_model.empty()) {
        cfg.preferred_model = spec.default_model;
    }

    // Optional environment overrides when using LM Studio
    if (cfg.provider == "lmstudio") {
        if (auto env_url = util::get_env_utf8(L"LMSTUDIO_BASE_URL"); env_url.has_value() && !env_url->empty()) {
            cfg.base_url = ComicTranslator::normalize_provider_base_url("lmstudio", *env_url);
        }
        if (auto env_model = util::get_env_utf8(L"LMSTUDIO_MODEL"); env_model.has_value() && !env_model->empty()) {
            cfg.preferred_model = util::trim_ascii(*env_model);
        }
    }

    if (settings.contains("temperature")) {
        if (auto v = parse_optional_double(settings["temperature"])) {
            cfg.temperature = clamp_d(*v, 0.0, 2.0);
        }
    }
    if (settings.contains("top_p")) {
        if (settings["top_p"].is_null()) {
            cfg.top_p = std::nullopt;
        } else if (auto v = parse_optional_double(settings["top_p"])) {
            cfg.top_p = clamp_d(*v, 0.0, 1.0);
        }
    }
    if (settings.contains("top_k")) {
        if (settings["top_k"].is_null()) {
            cfg.top_k = std::nullopt;
        } else if (auto v = parse_optional_int(settings["top_k"])) {
            cfg.top_k = clamp_i(*v, 0, 500);
        }
    }
    if (settings.contains("max_tokens")) {
        if (auto v = parse_optional_int(settings["max_tokens"])) {
            cfg.max_tokens = clamp_i(*v, 64, 32768);
        }
    }
    if (settings.contains("frequency_penalty")) {
        if (settings["frequency_penalty"].is_null()) {
            cfg.frequency_penalty = std::nullopt;
        } else if (auto v = parse_optional_double(settings["frequency_penalty"])) {
            cfg.frequency_penalty = clamp_d(*v, -2.0, 2.0);
        }
    }
    if (settings.contains("reasoning_effort") && settings["reasoning_effort"].is_string()) {
        cfg.reasoning_effort = ComicTranslator::normalize_reasoning_effort(
            settings["reasoning_effort"].get<std::string>());
    }
    if (settings.contains("strip_thinking")) {
        if (auto v = parse_optional_bool(settings["strip_thinking"])) {
            cfg.strip_thinking = *v;
        }
    }
    if (settings.contains("use_vision")) {
        if (auto v = parse_optional_bool(settings["use_vision"])) {
            cfg.use_vision = *v;
        }
    }
    if (settings.contains("vision_max_side")) {
        if (auto v = parse_optional_int(settings["vision_max_side"])) {
            cfg.vision_max_side = clamp_i(*v, 256, 2048);
        }
    }
    if (settings.contains("custom_instructions") && settings["custom_instructions"].is_string()) {
        cfg.custom_instructions = util::trim_ascii(settings["custom_instructions"].get<std::string>());
    }
    if (settings.contains("glossary")) {
        cfg.glossary = ComicTranslator::parse_glossary(settings["glossary"]);
    }
    if (settings.contains("timeout")) {
        if (auto v = parse_optional_double(settings["timeout"])) {
            cfg.timeout_seconds = clamp_d(*v, 5.0, 300.0);
        }
    }
    return cfg;
}

TranslatorConfig TranslatorConfig::from_settings_with_overrides(
    const json& settings,
    const json& overrides) {
    TranslatorConfig cfg = from_settings(settings);
    if (!overrides.is_object() || overrides.empty()) {
        return cfg;
    }

    const std::string base_provider = cfg.provider;
    std::string target_provider = base_provider;
    if (overrides.contains("provider") && overrides["provider"].is_string() &&
        !util::trim_ascii(overrides["provider"].get<std::string>()).empty()) {
        target_provider = ComicTranslator::normalize_provider_id(
            overrides["provider"].get<std::string>());
    }

    const bool switching_provider = (target_provider != base_provider);
    const auto& preset = ComicTranslator::get_provider_spec(target_provider);

    const json* saved_profile = nullptr;
    if (settings.is_object() && settings.contains("providers") && settings["providers"].is_object() &&
        settings["providers"].contains(target_provider) &&
        settings["providers"][target_provider].is_object()) {
        saved_profile = &settings["providers"][target_provider];
    }

    cfg.provider = target_provider;

    if (overrides.contains("base_url") && overrides["base_url"].is_string() &&
        !util::trim_ascii(overrides["base_url"].get<std::string>()).empty()) {
        cfg.base_url = ComicTranslator::normalize_provider_base_url(
            target_provider, overrides["base_url"].get<std::string>());
    } else if (switching_provider) {
        const std::string prof_url =
            (saved_profile && saved_profile->contains("base_url") && (*saved_profile)["base_url"].is_string())
            ? (*saved_profile)["base_url"].get<std::string>()
            : preset.default_base_url;
        cfg.base_url = ComicTranslator::normalize_provider_base_url(target_provider, prof_url);
    }

    if (overrides.contains("api_key") && overrides["api_key"].is_string() &&
        !util::trim_ascii(overrides["api_key"].get<std::string>()).empty()) {
        cfg.api_key = util::trim_ascii(overrides["api_key"].get<std::string>());
    } else if (switching_provider) {
        const std::string prof_key =
            (saved_profile && saved_profile->contains("api_key") && (*saved_profile)["api_key"].is_string())
            ? (*saved_profile)["api_key"].get<std::string>()
            : preset.default_api_key;
        cfg.api_key = ComicTranslator::resolve_provider_api_key(target_provider, prof_key);
    }

    if (overrides.contains("model") && overrides["model"].is_string() &&
        !util::trim_ascii(overrides["model"].get<std::string>()).empty()) {
        cfg.preferred_model = util::trim_ascii(overrides["model"].get<std::string>());
    } else if (switching_provider) {
        const std::string prof_model =
            (saved_profile && saved_profile->contains("model") && (*saved_profile)["model"].is_string())
            ? util::trim_ascii((*saved_profile)["model"].get<std::string>())
            : "";
        cfg.preferred_model = prof_model.empty() ? preset.default_model : prof_model;
    }

    if (overrides.contains("temperature") && !overrides["temperature"].is_null()) {
        if (auto v = parse_optional_double(overrides["temperature"])) {
            cfg.temperature = clamp_d(*v, 0.0, 2.0);
        }
    }
    if (overrides.contains("top_p")) {
        if (overrides["top_p"].is_null()) {
            // Preserve base when null in partial override unless explicitly cleared
        } else if (auto v = parse_optional_double(overrides["top_p"])) {
            cfg.top_p = clamp_d(*v, 0.0, 1.0);
        }
    }
    if (overrides.contains("top_k")) {
        if (!overrides["top_k"].is_null()) {
            if (auto v = parse_optional_int(overrides["top_k"])) {
                cfg.top_k = clamp_i(*v, 0, 500);
            }
        }
    }
    if (overrides.contains("max_tokens") && !overrides["max_tokens"].is_null()) {
        if (auto v = parse_optional_int(overrides["max_tokens"])) {
            cfg.max_tokens = clamp_i(*v, 64, 32768);
        }
    }
    if (overrides.contains("frequency_penalty") && !overrides["frequency_penalty"].is_null()) {
        if (auto v = parse_optional_double(overrides["frequency_penalty"])) {
            cfg.frequency_penalty = clamp_d(*v, -2.0, 2.0);
        }
    }
    if (overrides.contains("reasoning_effort") && overrides["reasoning_effort"].is_string() &&
        !util::trim_ascii(overrides["reasoning_effort"].get<std::string>()).empty()) {
        cfg.reasoning_effort = ComicTranslator::normalize_reasoning_effort(
            overrides["reasoning_effort"].get<std::string>());
    }
    if (overrides.contains("strip_thinking") && !overrides["strip_thinking"].is_null()) {
        if (auto v = parse_optional_bool(overrides["strip_thinking"])) {
            cfg.strip_thinking = *v;
        }
    }
    if (overrides.contains("use_vision") && !overrides["use_vision"].is_null()) {
        if (auto v = parse_optional_bool(overrides["use_vision"])) {
            cfg.use_vision = *v;
        }
    }
    if (overrides.contains("vision_max_side") && !overrides["vision_max_side"].is_null()) {
        if (auto v = parse_optional_int(overrides["vision_max_side"])) {
            cfg.vision_max_side = clamp_i(*v, 256, 2048);
        }
    }
    if (overrides.contains("custom_instructions") && overrides["custom_instructions"].is_string()) {
        cfg.custom_instructions = util::trim_ascii(overrides["custom_instructions"].get<std::string>());
    }
    if (overrides.contains("glossary") && !overrides["glossary"].is_null()) {
        const auto extra = ComicTranslator::parse_glossary(overrides["glossary"]);
        for (const auto& [k, v] : extra) {
            cfg.glossary[k] = v;
        }
    }
    if (overrides.contains("timeout") && !overrides["timeout"].is_null()) {
        if (auto v = parse_optional_double(overrides["timeout"])) {
            cfg.timeout_seconds = clamp_d(*v, 5.0, 300.0);
        }
    }
    return cfg;
}

// -----------------------------------------------------------------------------
// ComicTranslator Static Normalization & Helpers
// -----------------------------------------------------------------------------

std::string ComicTranslator::normalize_provider_id(const std::string& raw) {
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
        throw std::invalid_argument(
            "Unsupported provider '" + raw +
            "'. Must be one of: lmstudio, ollama, openai, gemini, anthropic, deepseek, openrouter, custom.");
    }
    return it->second;
}

const ProviderSpec& ComicTranslator::get_provider_spec(const std::string& provider_id) {
    const std::string norm = normalize_provider_id(provider_id);
    for (const auto& spec : all_provider_specs()) {
        if (spec.id == norm) {
            return spec;
        }
    }
    return all_provider_specs().front();
}

std::string ComicTranslator::normalize_provider_base_url(
    const std::string& provider_id,
    const std::string& raw_base_url) {
    const std::string norm = normalize_provider_id(provider_id);
    const auto& spec = get_provider_spec(norm);
    std::string cleaned = util::trim_ascii(raw_base_url);
    while (!cleaned.empty() && cleaned.back() == '/') {
        cleaned.pop_back();
    }
    if (cleaned.empty()) {
        std::string fallback = spec.default_base_url;
        while (!fallback.empty() && fallback.back() == '/') {
            fallback.pop_back();
        }
        return fallback;
    }

    if (norm == "lmstudio" || norm == "ollama") {
        const std::string lower = util::to_lower_ascii(cleaned);
        const bool has_v1 = lower.size() >= 3 && lower.compare(lower.size() - 3, 3, "/v1") == 0;
        if (!has_v1 && lower.find("/v1/") == std::string::npos &&
            lower.find("/api/") == std::string::npos) {
            cleaned += "/v1";
        }
    }
    return cleaned;
}

std::string ComicTranslator::resolve_provider_api_key(
    const std::string& provider_id,
    const std::string& raw_api_key) {
    const std::string cleaned = util::trim_ascii(raw_api_key);
    if (!cleaned.empty()) return cleaned;
    return get_provider_spec(provider_id).default_api_key;
}

std::string ComicTranslator::normalize_source_language(const std::string& raw) {
    const std::string cleaned = util::to_lower_ascii(util::trim_ascii(raw));
    if (cleaned.empty() || cleaned == "ja" || cleaned == "jp" || cleaned == "jpn" || cleaned == "japanese") {
        return "ja";
    }
    if (cleaned == "ko" || cleaned == "kr" || cleaned == "kor" || cleaned == "korean") {
        return "ko";
    }
    if (cleaned == "zh" || cleaned == "cn" || cleaned == "zho" || cleaned == "chi" ||
        cleaned == "zh-cn" || cleaned == "zh-tw" || cleaned == "chinese") {
        return "zh";
    }
    throw std::invalid_argument(
        "Unsupported source_lang '" + raw + "'. Must be 'ja', 'ko', or 'zh'.");
}

std::string ComicTranslator::normalize_target_language(const std::string& raw) {
    const std::string cleaned = util::to_lower_ascii(util::trim_ascii(raw));
    for (const auto& spec : all_target_languages()) {
        for (const auto& alias : spec.aliases) {
            if (cleaned == alias) {
                return spec.code;
            }
        }
    }
    throw std::invalid_argument(
        "Unsupported target_lang '" + raw +
        "'. Supported codes: id, en, ja, ko, zh-CN, zh-TW, es, fr, de, pt, ru, vi, th, ms, ar, it.");
}

const TargetLanguageSpec& ComicTranslator::get_target_language_spec(const std::string& code) {
    const std::string norm = normalize_target_language(code);
    for (const auto& spec : all_target_languages()) {
        if (spec.code == norm) {
            return spec;
        }
    }
    return all_target_languages().front();
}

std::string ComicTranslator::normalize_reasoning_effort(const std::string& raw) {
    const std::string cleaned = util::to_lower_ascii(util::trim_ascii(raw));
    if (cleaned.empty() || cleaned == "off" || cleaned == "none" ||
        cleaned == "disabled" || cleaned == "false" || cleaned == "0") {
        return "off";
    }
    if (cleaned == "low" || cleaned == "minimal") return "low";
    if (cleaned == "medium" || cleaned == "med" || cleaned == "default") return "medium";
    if (cleaned == "high" || cleaned == "max" || cleaned == "xhigh") return "high";
    throw std::invalid_argument(
        "Unsupported reasoning_effort '" + raw + "'. Must be one of: off, low, medium, high.");
}

std::map<std::string, std::string> ComicTranslator::parse_glossary(const json& raw) {
    std::map<std::string, std::string> parsed;
    if (raw.is_null()) return parsed;

    if (raw.is_object()) {
        for (auto it = raw.begin(); it != raw.end(); ++it) {
            const std::string src = util::trim_ascii(it.key());
            const std::string tgt = it.value().is_null()
                ? ""
                : util::trim_ascii(json_scalar_to_string(it.value()));
            if (!src.empty() && !tgt.empty()) {
                parsed[src] = tgt;
            }
        }
        return parsed;
    }

    if (raw.is_string()) {
        const std::string text = util::trim_ascii(raw.get<std::string>());
        if (text.empty()) return parsed;
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
            static const std::array<std::string, 5> kSeps = {"->", "=>", "→", "=", ":"};
            for (const auto& sep : kSeps) {
                const size_t pos = trimmed.find(sep);
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
                    src = util::trim_ascii(json_scalar_to_string(entry["source"]));
                } else if (entry.contains("src") && !entry["src"].is_null()) {
                    src = util::trim_ascii(json_scalar_to_string(entry["src"]));
                }
                if (entry.contains("target") && !entry["target"].is_null()) {
                    tgt = util::trim_ascii(json_scalar_to_string(entry["target"]));
                } else if (entry.contains("tgt") && !entry["tgt"].is_null()) {
                    tgt = util::trim_ascii(json_scalar_to_string(entry["tgt"]));
                }
                if (!src.empty() && !tgt.empty()) {
                    parsed[src] = tgt;
                }
            } else if (entry.is_string()) {
                const auto sub = parse_glossary(entry);
                for (const auto& [k, v] : sub) {
                    parsed[k] = v;
                }
            }
        }
    }
    return parsed;
}

bool ComicTranslator::is_embedding_model(const std::string& model_id) {
    const std::string lower = util::to_lower_ascii(model_id);
    if (lower.find("embed") != std::string::npos || lower.rfind("text-embedding", 0) == 0) {
        return true;
    }
    static const std::array<const char*, 8> kNonChat = {
        "whisper", "tts-", "dall-e", "text-moderation",
        "omni-moderation", "babbage-002", "davinci-002", "rerank",
    };
    for (const char* marker : kNonChat) {
        if (lower.find(marker) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool ComicTranslator::is_likely_vision_model(
    const std::string& model_id,
    const std::string& provider_id) {
    const std::string lower = util::to_lower_ascii(model_id);
    if (provider_id == "anthropic" &&
        (lower.find("claude-3") != std::string::npos ||
         lower.find("claude-4") != std::string::npos ||
         lower.find("claude-sonnet") != std::string::npos ||
         lower.find("claude-opus") != std::string::npos ||
         lower.find("claude-haiku") != std::string::npos)) {
        return true;
    }
    if (provider_id == "gemini" &&
        lower.find("gemini") != std::string::npos &&
        lower.find("embed") == std::string::npos) {
        return true;
    }
    static const std::array<const char*, 31> kMarkers = {
        "vision", "vl", "vlm",
        "gemma-3", "gemma3", "gemma-4", "gemma4",
        "qwen2.5-vl", "qwen2-vl", "qwen3.5", "qwen3-vl",
        "pixtral", "ministral-3", "llava", "minicpm-v", "internvl",
        "gpt-4o", "gpt-4.1", "gpt-4-turbo", "gpt-5",
        "o1", "o3", "o4",
        "claude-3", "claude-4", "claude-sonnet", "claude-opus",
        "gemini-1.5", "gemini-2", "gemini-3", "kimi-vl",
    };
    for (const char* marker : kMarkers) {
        if (lower.find(marker) != std::string::npos) {
            return true;
        }
    }
    return false;
}

double ComicTranslator::score_model_match(
    const std::string& preferred_name,
    const std::string& candidate_id) {
    const std::string pref_lower = util::to_lower_ascii(util::trim_ascii(preferred_name));
    const std::string cand_lower = util::to_lower_ascii(util::trim_ascii(candidate_id));
    if (pref_lower.empty() || cand_lower.empty()) return 0.0;
    if (pref_lower == cand_lower) return 1.0;

    const std::string pref_compact = compact_alnum_lower(pref_lower);
    const std::string cand_compact = compact_alnum_lower(cand_lower);
    if (!pref_compact.empty() && !cand_compact.empty() && pref_compact == cand_compact) {
        return 0.99;
    }

    const auto pref_tokens = tokenize_model_name(pref_lower);
    const auto cand_tokens = tokenize_model_name(cand_lower);
    if (pref_tokens.empty() || cand_tokens.empty()) return 0.0;

    static const std::unordered_set<std::string> kQuantTokens = {
        "q2", "q3", "q4", "q5", "q6", "q8",
        "k", "p", "m", "s", "l", "xs", "xxs",
        "0", "1", "fp16", "bf16", "fp32", "gguf", "imat", "iq4", "iq3",
    };

    std::unordered_set<std::string> pref_core;
    for (const auto& t : pref_tokens) {
        if (!kQuantTokens.count(t)) pref_core.insert(t);
    }
    if (pref_core.empty()) {
        pref_core.insert(pref_tokens.begin(), pref_tokens.end());
    }

    std::unordered_set<std::string> cand_core;
    for (const auto& t : cand_tokens) {
        if (!kQuantTokens.count(t)) cand_core.insert(t);
    }
    if (cand_core.empty()) {
        cand_core.insert(cand_tokens.begin(), cand_tokens.end());
    }

    size_t common = 0;
    for (const auto& t : pref_core) {
        if (cand_core.count(t)) ++common;
    }
    if (common == 0) return 0.0;

    const size_t union_sz = pref_core.size() + cand_core.size() - common;
    const double core_jaccard = static_cast<double>(common) / static_cast<double>(std::max<size_t>(1, union_sz));
    const double cand_cov = static_cast<double>(common) / static_cast<double>(std::max<size_t>(1, cand_core.size()));
    const double pref_cov = static_cast<double>(common) / static_cast<double>(std::max<size_t>(1, pref_core.size()));

    double score = (core_jaccard * 0.5) + (cand_cov * 0.3) + (pref_cov * 0.2);
    if (!cand_compact.empty() &&
        (pref_compact.find(cand_compact) != std::string::npos ||
         cand_compact.find(pref_compact) != std::string::npos)) {
        score = std::min(0.98, score + 0.1);
    }
    return score;
}

std::optional<std::string> ComicTranslator::format_display_model_name(
    const std::string& provider_id,
    const std::optional<std::string>& resolved_model_id,
    const std::string& preferred_model) {
    if (!resolved_model_id.has_value()) {
        return std::nullopt;
    }
    const std::string res_lower = util::to_lower_ascii(*resolved_model_id);
    const std::string pref_lower = util::to_lower_ascii(preferred_model);
    if (provider_id == "lmstudio" &&
        res_lower.find("gemma-4-e2b-uncensored-hauhaucs-aggressive") != std::string::npos &&
        (pref_lower.empty() ||
         pref_lower.find("gemma 4 e2b") != std::string::npos ||
         pref_lower.find("gemma-4-e2b") != std::string::npos)) {
        return std::string(kDefaultLmStudioModel);
    }
    return resolved_model_id;
}

// -----------------------------------------------------------------------------
// Provider Discovery, Connection Test & LM Studio Status
// -----------------------------------------------------------------------------

std::vector<std::string> ComicTranslator::list_provider_models(
    const std::string& provider,
    const std::string& base_url,
    const std::string& api_key,
    double timeout_seconds,
    bool include_embeddings) {
    const std::string prov_id = normalize_provider_id(provider);
    const auto& spec = get_provider_spec(prov_id);
    const std::string resolved_url = normalize_provider_base_url(prov_id, base_url);
    const std::string resolved_key = resolve_provider_api_key(prov_id, api_key);

    if (spec.requires_base_url && resolved_url.empty()) {
        throw ProviderConnectionError("Base URL is required for " + spec.name + ".");
    }
    if (spec.requires_api_key && resolved_key.empty()) {
        throw ProviderConnectionError(
            "API key is required for " + spec.name +
            ". Provide an API key to test or use this provider.");
    }

    const std::string models_url = resolved_url + "/models";
    std::vector<std::pair<std::string, std::string>> headers = {
        {"Accept", "application/json"},
    };
    for (const auto& kv : spec.extra_headers) {
        headers.push_back(kv);
    }
    if (spec.protocol == "anthropic") {
        headers.emplace_back("x-api-key", resolved_key);
        headers.emplace_back("anthropic-version", kDefaultAnthropicVersion);
    } else if (!resolved_key.empty()) {
        headers.emplace_back("Authorization", "Bearer " + resolved_key);
    }

    const HttpResponse resp = perform_http_request("GET", models_url, headers, "", timeout_seconds);
    if (resp.transport_error) {
        if (resp.timed_out) {
            std::ostringstream oss;
            oss << "Timed out after " << timeout_seconds << "s while connecting to "
                << spec.name << " at " << models_url << ".";
            throw ProviderConnectionError(oss.str());
        }
        std::string hint;
        if (prov_id == "lmstudio") {
            hint = "Ensure LM Studio is open and the Local Inference Server is running (default port 1235).";
        } else if (prov_id == "ollama") {
            hint = "Ensure Ollama is running locally (default port 11434).";
        } else {
            hint = "Verify network connectivity and Base URL (" + resolved_url + ").";
        }
        throw ProviderConnectionError(
            "Cannot connect to " + spec.name + " at " + models_url + ". " + hint);
    }

    if (resp.status_code == 401 || resp.status_code == 403) {
        throw ProviderConnectionError(
            "Authentication failed for " + spec.name +
            " (HTTP " + std::to_string(resp.status_code) + "): " +
            extract_http_error_detail(resp));
    }

    if (resp.status_code >= 400) {
        if (prov_id == "ollama") {
            auto tags = fetch_ollama_native_tags(resolved_url, timeout_seconds);
            if (!tags.empty()) {
                if (include_embeddings) return tags;
                std::vector<std::string> filtered;
                for (const auto& m : tags) {
                    if (!is_embedding_model(m)) filtered.push_back(m);
                }
                return filtered.empty() ? tags : filtered;
            }
        }
        throw ProviderConnectionError(
            spec.name + " returned HTTP " + std::to_string(resp.status_code) +
            " at " + models_url + ": " + extract_http_error_detail(resp));
    }

    const json payload = json::parse(resp.body, nullptr, false);
    if (payload.is_discarded()) {
        std::string snippet = resp.body.substr(0, std::min<size_t>(200, resp.body.size()));
        throw ProviderConnectionError(
            spec.name + " at " + models_url + " returned non-JSON payload: " + snippet);
    }

    std::vector<std::string> raw_ids;
    auto push_id = [&](std::string mid) {
        mid = util::trim_ascii(mid);
        if (mid.empty()) return;
        if (prov_id == "gemini" && mid.rfind("models/", 0) == 0) {
            mid = mid.substr(7);
        }
        raw_ids.push_back(std::move(mid));
    };

    if (payload.is_object()) {
        const json* arr = nullptr;
        if (payload.contains("data") && payload["data"].is_array()) {
            arr = &payload["data"];
        } else if (payload.contains("models") && payload["models"].is_array()) {
            arr = &payload["models"];
        }
        if (arr) {
            for (const auto& item : *arr) {
                if (item.is_object()) {
                    if (item.contains("id") && item["id"].is_string()) {
                        push_id(item["id"].get<std::string>());
                    } else if (item.contains("name") && item["name"].is_string()) {
                        push_id(item["name"].get<std::string>());
                    }
                } else if (item.is_string()) {
                    push_id(item.get<std::string>());
                }
            }
        }
    } else if (payload.is_array()) {
        for (const auto& item : payload) {
            if (item.is_string()) {
                push_id(item.get<std::string>());
            } else if (item.is_object() && item.contains("id") && item["id"].is_string()) {
                push_id(item["id"].get<std::string>());
            }
        }
    } else {
        throw ProviderConnectionError("Unexpected response format from " + models_url + ".");
    }

    if (raw_ids.empty() && prov_id == "ollama") {
        raw_ids = fetch_ollama_native_tags(resolved_url, timeout_seconds);
    }

    // Deduplicate while preserving order
    std::vector<std::string> deduped;
    std::unordered_set<std::string> seen;
    for (auto& id : raw_ids) {
        if (seen.insert(id).second) {
            deduped.push_back(std::move(id));
        }
    }

    if (include_embeddings) {
        return deduped;
    }
    std::vector<std::string> filtered;
    for (const auto& m : deduped) {
        if (!is_embedding_model(m)) {
            filtered.push_back(m);
        }
    }
    return filtered.empty() ? deduped : filtered;
}

std::string ComicTranslator::resolve_provider_model(
    const std::string& provider,
    const std::string& preferred_model,
    const std::string& base_url,
    const std::string& api_key,
    double timeout_seconds,
    const std::optional<std::vector<std::string>>& available_models) {
    const std::string prov_id = normalize_provider_id(provider);
    const auto& spec = get_provider_spec(prov_id);
    const std::string resolved_url = normalize_provider_base_url(prov_id, base_url);
    std::string pref_clean = util::trim_ascii(preferred_model);
    if (pref_clean.empty()) {
        pref_clean = spec.default_model;
    }
    if (prov_id == "gemini" && pref_clean.rfind("models/", 0) == 0) {
        pref_clean = pref_clean.substr(7);
    }

    // For cloud/remote providers, skip redundant /models roundtrip during inference when model is known
    if (!spec.is_local && !available_models.has_value() && !pref_clean.empty()) {
        return pref_clean;
    }

    const std::vector<std::string> all_models = available_models.has_value()
        ? *available_models
        : list_provider_models(prov_id, resolved_url, api_key, timeout_seconds, true);

    std::vector<std::string> llm_models;
    for (const auto& m : all_models) {
        if (!is_embedding_model(m)) {
            llm_models.push_back(m);
        }
    }
    const auto& candidates = llm_models.empty() ? all_models : llm_models;

    if (candidates.empty()) {
        if (!pref_clean.empty() && !spec.is_local) {
            return pref_clean;
        }
        throw ProviderConnectionError(
            spec.name + " at " + resolved_url +
            " is reachable, but no models are installed or loaded.");
    }

    // 1. Exact or case-insensitive match
    if (!pref_clean.empty()) {
        const std::string pref_lower = util::to_lower_ascii(pref_clean);
        for (const auto& cand : candidates) {
            if (cand == pref_clean || util::to_lower_ascii(cand) == pref_lower) {
                return cand;
            }
        }

        // 2. Fuzzy token match (e.g. "Gemma 4 E2B Uncensored HauhauCS Aggressive Q4 K P"
        //    -> "gemma-4-e2b-uncensored-hauhaucs-aggressive")
        double best_score = -1.0;
        std::string best_cand;
        for (const auto& cand : candidates) {
            const double sc = score_model_match(pref_clean, cand);
            if (sc > best_score) {
                best_score = sc;
                best_cand = cand;
            }
        }
        if (best_score >= 0.50 && !best_cand.empty()) {
            return best_cand;
        }
    }

    // 3. For LM Studio, prefer the currently loaded LLM from `/api/v0/models`
    if (prov_id == "lmstudio") {
        const auto [loaded_v0, _] = fetch_lmstudio_v0_metadata(
            resolved_url, std::min(3.0, timeout_seconds));
        if (loaded_v0.has_value()) {
            for (const auto& cand : candidates) {
                if (cand == *loaded_v0) {
                    return *loaded_v0;
                }
            }
        }
    }

    // 4. For cloud/remote providers, trust explicitly configured model ID even if absent from /models
    if (!pref_clean.empty() && !spec.is_local) {
        return pref_clean;
    }

    return candidates.front();
}

ProviderTestResult ComicTranslator::test_provider_connection(
    const std::string& provider,
    const std::string& base_url,
    const std::string& api_key,
    const std::string& preferred_model,
    double timeout_seconds) {
    const std::string prov_id = normalize_provider_id(provider);
    const auto& spec = get_provider_spec(prov_id);
    const std::string resolved_url = normalize_provider_base_url(prov_id, base_url);
    std::string pref_model = util::trim_ascii(preferred_model);
    if (pref_model.empty()) {
        pref_model = spec.default_model;
    }

    ProviderTestResult res;
    res.provider = prov_id;
    res.provider_name = spec.name;
    res.endpoint = resolved_url;
    res.preferred_model = pref_model;

    const auto t0 = std::chrono::steady_clock::now();
    try {
        const auto all_models = list_provider_models(
            prov_id, resolved_url, api_key, timeout_seconds, true);
        std::vector<std::string> chat_models;
        for (const auto& m : all_models) {
            if (!is_embedding_model(m)) {
                chat_models.push_back(m);
            }
        }
        const auto& listed = chat_models.empty() ? all_models : chat_models;

        std::unordered_set<std::string> vlm_from_server;
        if (prov_id == "lmstudio") {
            auto [_, vlms] = fetch_lmstudio_v0_metadata(
                resolved_url, std::min(3.0, timeout_seconds));
            vlm_from_server = std::move(vlms);
        }

        std::vector<std::string> vision_models;
        for (const auto& m : listed) {
            if (vlm_from_server.count(m) || is_likely_vision_model(m, prov_id)) {
                vision_models.push_back(m);
            }
        }

        std::optional<std::string> resolved_id;
        if (!listed.empty()) {
            resolved_id = resolve_provider_model(
                prov_id, pref_model, resolved_url, api_key, timeout_seconds, all_models);
        } else if (!pref_model.empty() && !spec.is_local) {
            resolved_id = pref_model;
        }

        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0);
        res.latency_ms = static_cast<int>(elapsed.count());

        if (listed.empty() && spec.is_local) {
            res.ok = false;
            res.connected = false;
            res.message = "Connected to " + spec.name + ", but no models are installed or loaded.";
            res.error = spec.name + " at " + resolved_url + " has no models available.";
            return res;
        }

        res.ok = true;
        res.connected = true;
        res.resolved_model_id = resolved_id;
        res.active_model = format_display_model_name(prov_id, resolved_id, pref_model);
        res.models = listed;
        res.vision_models = std::move(vision_models);
        res.message = "Connected to " + spec.name + ". Found " +
                      std::to_string(listed.size()) + " model(s).";
        return res;
    } catch (const ProviderConnectionError& ex) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0);
        res.latency_ms = static_cast<int>(elapsed.count());
        res.ok = false;
        res.connected = false;
        res.message = ex.what();
        res.error = ex.what();
        return res;
    }
}

LmStudioStatusResult ComicTranslator::get_lmstudio_status(
    const json& settings,
    double timeout_seconds) {
    std::string lm_url = kDefaultLmStudioBaseUrl;
    std::string lm_key = kDefaultLmStudioApiKey;
    std::string preferred_model = kDefaultLmStudioModel;

    if (settings.is_object()) {
        const std::string active_prov = settings.value("provider", std::string("lmstudio"));
        if (active_prov == "lmstudio") {
            if (settings.contains("base_url") && settings["base_url"].is_string() &&
                !util::trim_ascii(settings["base_url"].get<std::string>()).empty()) {
                lm_url = settings["base_url"].get<std::string>();
            }
            if (settings.contains("api_key") && settings["api_key"].is_string() &&
                !util::trim_ascii(settings["api_key"].get<std::string>()).empty()) {
                lm_key = settings["api_key"].get<std::string>();
            }
            if (settings.contains("model") && settings["model"].is_string() &&
                !util::trim_ascii(settings["model"].get<std::string>()).empty()) {
                preferred_model = util::trim_ascii(settings["model"].get<std::string>());
            }
        } else if (settings.contains("providers") && settings["providers"].is_object() &&
                   settings["providers"].contains("lmstudio") &&
                   settings["providers"]["lmstudio"].is_object()) {
            const auto& prof = settings["providers"]["lmstudio"];
            if (prof.contains("base_url") && prof["base_url"].is_string() &&
                !util::trim_ascii(prof["base_url"].get<std::string>()).empty()) {
                lm_url = prof["base_url"].get<std::string>();
            }
            if (prof.contains("api_key") && prof["api_key"].is_string() &&
                !util::trim_ascii(prof["api_key"].get<std::string>()).empty()) {
                lm_key = prof["api_key"].get<std::string>();
            }
            if (prof.contains("model") && prof["model"].is_string() &&
                !util::trim_ascii(prof["model"].get<std::string>()).empty()) {
                preferred_model = util::trim_ascii(prof["model"].get<std::string>());
            }
        }
    }

    if (auto env_url = util::get_env_utf8(L"LMSTUDIO_BASE_URL"); env_url.has_value() && !env_url->empty()) {
        lm_url = *env_url;
    }
    if (auto env_model = util::get_env_utf8(L"LMSTUDIO_MODEL"); env_model.has_value() && !env_model->empty()) {
        preferred_model = util::trim_ascii(*env_model);
    }

    const std::string resolved_url = normalize_provider_base_url("lmstudio", lm_url);

    LmStudioStatusResult status;
    status.endpoint = resolved_url;
    status.preferred_model = preferred_model;

    try {
        const auto models = list_provider_models(
            "lmstudio", resolved_url, lm_key, timeout_seconds, true);
        const std::string resolved_id = resolve_provider_model(
            "lmstudio", preferred_model, resolved_url, lm_key, timeout_seconds, models);
        status.connected = true;
        status.resolved_model_id = resolved_id;
        status.active_model = format_display_model_name(
            "lmstudio", status.resolved_model_id, preferred_model);
        status.available_models = models;
    } catch (const ProviderConnectionError& ex) {
        status.connected = false;
        status.error = ex.what();
    } catch (const std::exception& ex) {
        status.connected = false;
        status.error = ex.what();
    }

    return status;
}

// -----------------------------------------------------------------------------
// Prompting, Vision Encoding, Reasoning Stripping & JSON Extraction
// -----------------------------------------------------------------------------

json ComicTranslator::comic_translation_json_schema() {
    return {
        {"type", "json_schema"},
        {"json_schema", {
            {"name", "comic_translation_response"},
            {"strict", true},
            {"schema", {
                {"type", "object"},
                {"properties", {
                    {"dialogues", {
                        {"type", "array"},
                        {"items", {
                            {"type", "object"},
                            {"properties", {
                                {"id", {{"type", "string"}}},
                                {"translated_text", {{"type", "string"}}},
                            }},
                            {"required", json::array({"id", "translated_text"})},
                            {"additionalProperties", false},
                        }},
                    }},
                }},
                {"required", json::array({"dialogues"})},
                {"additionalProperties", false},
            }},
        }},
    };
}

std::string ComicTranslator::build_system_prompt(
    const std::string& source_lang,
    const std::string& target_lang,
    const std::string& reasoning_effort,
    const std::string& custom_instructions,
    const std::map<std::string, std::string>& glossary,
    bool has_page_image) {
    const std::string src_norm = normalize_source_language(source_lang);
    const auto& tgt_spec = get_target_language_spec(target_lang);
    const std::string effort_norm = normalize_reasoning_effort(reasoning_effort);

    std::string source_desc;
    std::string honorific_rule;
    if (src_norm == "ja") {
        source_desc = "Japanese Manga (ja)";
        honorific_rule =
            "Preserve Japanese honorifics naturally (e.g., -san, -senpai, -kun, -chan, -sama, -sensei "
            "such as 'Tanaka-senpai', 'Sato-san') when appropriate for the target language.";
    } else if (src_norm == "ko") {
        source_desc = "Korean Manhwa (ko)";
        honorific_rule =
            "Preserve Korean honorifics and kinship/address terms naturally "
            "(e.g., oppa, hyung, sunbae, noona, unnie, -nim, dongsaeng) when appropriate for the target language.";
    } else {
        source_desc = "Chinese Manhua (zh)";
        honorific_rule =
            "Preserve Chinese martial and familial honorifics accurately "
            "(e.g., 哥哥 -> 'gege', 师兄 -> 'shixiong', 师姐 -> 'shijie', 师父 -> 'shifu', 师弟 -> 'shidi', 姐姐 -> 'jiejie') "
            "when translating into non-Chinese languages.";
    }

    const std::string target_desc =
        tgt_spec.name + " (" + tgt_spec.native_name + ", code: " + tgt_spec.code + ")";
    const std::string thinking_directive = (effort_norm == "off")
        ? "Do not think, analyze, or explain. "
        : "Keep any internal reasoning brief, then output strictly valid JSON. ";

    std::ostringstream oss;
    oss << "You are an expert " << source_desc << " to " << target_desc << " comic translator. "
        << thinking_directive
        << "Translate the full page of dialogue blocks from " << source_desc << " into " << target_desc << ".\n"
        << "Rules:\n"
        << "1. Read all blocks together as one continuous scene so conversational context, pronouns, and speaker tone remain consistent across bubbles.\n"
        << "2. " << tgt_spec.style_rule << "\n"
        << "3. " << honorific_rule << "\n"
        << "4. Output ONLY valid JSON in the exact format: "
        << "{\"dialogues\": [{\"id\": \"<block_id>\", \"translated_text\": \"<translation>\"}]}. "
        << "Keep every input block 'id' unchanged.";

    if (!glossary.empty()) {
        std::vector<std::string> pairs;
        for (const auto& [src_term, tgt_term] : glossary) {
            const std::string s = util::trim_ascii(src_term);
            const std::string t = util::trim_ascii(tgt_term);
            if (!s.empty() && !t.empty()) {
                pairs.push_back("- " + s + " -> " + t);
            }
        }
        if (!pairs.empty()) {
            oss << "\nGlossary (MUST follow strictly — always translate these terms as specified):";
            for (const auto& line : pairs) {
                oss << "\n" << line;
            }
        }
    }

    const std::string ci = util::trim_ascii(custom_instructions);
    if (!ci.empty()) {
        oss << "\nCustom Instructions (MUST follow): " << ci;
    }

    if (has_page_image) {
        oss << "\nVisual Context: Use the attached comic page image to infer speaker gender, expressions, "
            << "action intensity, relationships, and scene mood so every bubble matches the artwork.";
    }

    return oss.str();
}

std::string ComicTranslator::strip_thinking_tags(const std::string& raw_content) {
    std::string text = util::trim_ascii(raw_content);
    if (text.empty()) return {};

    // 1. Strip paired reasoning blocks (O(N), case-insensitive, zero regex stack risk)
    text = remove_paired_tag_icase(text, "<think>", "</think>");
    text = remove_paired_tag_icase(text, "<thought>", "</thought>");
    text = remove_paired_tag_icase(text, "<|thought|>", "<|thought|>");
    text = remove_paired_tag_icase(text, "<|channel>thought", "<channel|>");

    // 2. Handle an unclosed reasoning tag if a model omitted the closing tag before emitting JSON
    static const std::array<std::string, 4> kOpenTags = {
        "<think>", "<thought>", "<|thought|>", "<|channel>thought"
    };
    size_t unclosed_pos = std::string::npos;
    size_t unclosed_len = 0;
    for (const auto& tag : kOpenTags) {
        const size_t p = find_icase(text, tag, 0);
        if (p != std::string::npos && (unclosed_pos == std::string::npos || p < unclosed_pos)) {
            unclosed_pos = p;
            unclosed_len = tag.size();
        }
    }

    if (unclosed_pos != std::string::npos) {
        const std::string prefix = util::trim_ascii(text.substr(0, unclosed_pos));
        const std::string remainder = text.substr(unclosed_pos + unclosed_len);
        const size_t fence_idx = remainder.find("```");
        const size_t brace_idx = remainder.find('{');
        size_t best_idx = std::string::npos;
        if (fence_idx != std::string::npos && brace_idx != std::string::npos) {
            best_idx = std::min(fence_idx, brace_idx);
        } else if (fence_idx != std::string::npos) {
            best_idx = fence_idx;
        } else if (brace_idx != std::string::npos) {
            best_idx = brace_idx;
        }

        if (best_idx != std::string::npos) {
            const std::string after_thought = util::trim_ascii(remainder.substr(best_idx));
            text = prefix.empty() ? after_thought : (prefix + "\n" + after_thought);
        } else if (!prefix.empty()) {
            text = prefix;
        }
    }

    return util::trim_ascii(text);
}

std::map<std::string, std::string> ComicTranslator::extract_and_clean_translations(
    const std::string& raw_content,
    const std::vector<DialogueBlockInput>& input_blocks,
    bool strip_thinking) {
    std::vector<std::string> expected_ids;
    expected_ids.reserve(input_blocks.size());
    for (const auto& b : input_blocks) {
        expected_ids.push_back(b.id);
    }

    const std::string think_stripped = strip_thinking
        ? strip_thinking_tags(raw_content)
        : util::trim_ascii(raw_content);
    const std::string cleaned = strip_fences_and_noise(think_stripped);

    std::vector<std::pair<std::string, std::string>> extracted_ordered;
    if (!cleaned.empty()) {
        json parsed = json::parse(cleaned, nullptr, false);
        if (!parsed.is_discarded()) {
            extract_mapping_from_parsed_json(parsed, extracted_ordered);
        } else {
            // Retry after replacing raw ASCII control characters (preserving UTF-8 bytes >= 0x80)
            std::string sanitized = cleaned;
            for (char& ch : sanitized) {
                const unsigned char uc = static_cast<unsigned char>(ch);
                if (uc < 0x20) {
                    ch = ' ';
                }
            }
            parsed = json::parse(sanitized, nullptr, false);
            if (!parsed.is_discarded()) {
                extract_mapping_from_parsed_json(parsed, extracted_ordered);
            }
        }
    }

    if (extracted_ordered.empty() || extracted_ordered.size() < expected_ids.size()) {
        const auto salvaged = salvage_translations_from_text(think_stripped, expected_ids);
        for (const auto& [k, v] : salvaged) {
            bool found_non_empty = false;
            for (auto& existing : extracted_ordered) {
                if (existing.first == k) {
                    if (existing.second.empty()) existing.second = v;
                    found_non_empty = true;
                    break;
                }
            }
            if (!found_non_empty) {
                extracted_ordered.emplace_back(k, v);
            }
        }
    }

    std::unordered_map<std::string, std::string> extracted_map;
    std::vector<std::string> values_in_order;
    values_in_order.reserve(extracted_ordered.size());
    for (const auto& [k, v] : extracted_ordered) {
        extracted_map[k] = v;
        values_in_order.push_back(v);
    }

    std::map<std::string, std::string> result;
    for (size_t idx = 0; idx < input_blocks.size(); ++idx) {
        const std::string& bid = input_blocks[idx].id;
        const std::string& src = input_blocks[idx].source_text;
        if (src.empty()) {
            result[bid] = "";
            continue;
        }

        if (auto it = extracted_map.find(bid); it != extracted_map.end() && !it->second.empty()) {
            result[bid] = it->second;
            continue;
        }

        std::optional<std::string> matched;
        const std::string bid_lower = util::to_lower_ascii(bid);
        for (const auto& [ek, ev] : extracted_ordered) {
            if (!ev.empty() && util::to_lower_ascii(ek) == bid_lower) {
                matched = ev;
                break;
            }
        }

        if (!matched.has_value()) {
            // Trailing digits match (e.g. "bubble_001" <-> "001", "1", "bubble_1")
            size_t digit_end = bid.size();
            size_t digit_start = digit_end;
            while (digit_start > 0 && std::isdigit(static_cast<unsigned char>(bid[digit_start - 1]))) {
                --digit_start;
            }
            if (digit_start < digit_end) {
                const std::string num_str = bid.substr(digit_start);
                std::string num_int_str;
                try {
                    num_int_str = std::to_string(std::stoi(num_str));
                } catch (...) {
                    num_int_str = num_str;
                }
                const std::array<std::string, 3> candidates = {
                    num_str, num_int_str, "bubble_" + num_int_str
                };
                for (const auto& ck : candidates) {
                    if (auto it = extracted_map.find(ck); it != extracted_map.end() && !it->second.empty()) {
                        matched = it->second;
                        break;
                    }
                }
            }
        }

        if (!matched.has_value() &&
            values_in_order.size() == input_blocks.size() &&
            !values_in_order[idx].empty()) {
            matched = values_in_order[idx];
        }

        result[bid] = matched.value_or(src);
    }

    return result;
}

std::string ComicTranslator::apply_glossary_replacements(
    const std::string& text,
    const std::map<std::string, std::string>& glossary) {
    if (text.empty() || glossary.empty()) {
        return text;
    }

    // Sort longest source key first so longer phrases win before substrings
    std::vector<std::pair<std::string, std::string>> rules;
    rules.reserve(glossary.size());
    for (const auto& [k, v] : glossary) {
        const std::string sk = util::trim_ascii(k);
        const std::string sv = util::trim_ascii(v);
        if (!sk.empty() && !sv.empty() && sk != sv) {
            rules.emplace_back(sk, sv);
        }
    }
    if (rules.empty()) return text;

    std::sort(rules.begin(), rules.end(), [](const auto& a, const auto& b) {
        return a.first.size() > b.first.size();
    });

    std::string out = text;
    for (const auto& [src, tgt] : rules) {
        const bool ascii_word = is_pure_ascii_alnum_term(src);
        const size_t src_in_tgt = tgt.find(src);

        size_t pos = 0;
        while ((pos = out.find(src, pos)) != std::string::npos) {
            // Word boundary check for pure ASCII terms (prevents corrupting substrings)
            if (ascii_word) {
                const bool left_ok = (pos == 0) ||
                    !is_ascii_word_char(static_cast<unsigned char>(out[pos - 1]));
                const size_t after = pos + src.size();
                const bool right_ok = (after >= out.size()) ||
                    !is_ascii_word_char(static_cast<unsigned char>(out[after]));
                if (!left_ok || !right_ok) {
                    pos += src.size();
                    continue;
                }
            }

            // Avoid double-replacing when `src` is a substring of `tgt` already present at this position
            if (src_in_tgt != std::string::npos && pos >= src_in_tgt) {
                const size_t tgt_start = pos - src_in_tgt;
                if (tgt_start + tgt.size() <= out.size() &&
                    out.compare(tgt_start, tgt.size(), tgt) == 0) {
                    pos = tgt_start + tgt.size();
                    continue;
                }
            }

            out.replace(pos, src.size(), tgt);
            pos += tgt.size();
        }
    }

    return out;
}

std::string ComicTranslator::encode_rgb_for_vision(
    const uint8_t* rgb_data,
    int width,
    int height,
    int max_side,
    int jpeg_quality) {
    if (!rgb_data || width <= 0 || height <= 0) {
        throw std::invalid_argument("Invalid RGB image buffer for vision encoding.");
    }

    const int clamped_max = clamp_i(max_side, 256, 2048);
    const int clamped_q = clamp_i(jpeg_quality, 50, 98);
    const int longest = std::max(width, height);

    int out_w = width;
    int out_h = height;
    std::vector<uint8_t> resized_buf;
    const uint8_t* encode_ptr = rgb_data;

    if (longest > clamped_max) {
        const double scale = static_cast<double>(clamped_max) / static_cast<double>(longest);
        out_w = std::max(1, static_cast<int>(std::llround(width * scale)));
        out_h = std::max(1, static_cast<int>(std::llround(height * scale)));
        resized_buf = resize_rgb_bilinear(rgb_data, width, height, out_w, out_h);
        encode_ptr = resized_buf.data();
    }

    std::vector<uint8_t> jpg_bytes;
    jpg_bytes.reserve(static_cast<size_t>(out_w) * static_cast<size_t>(out_h) / 4ULL + 4096ULL);
    const int ok = stbi_write_jpg_to_func(
        stbi_write_vec_cb,
        &jpg_bytes,
        out_w,
        out_h,
        3,
        encode_ptr,
        clamped_q);
    if (!ok || jpg_bytes.empty()) {
        throw std::runtime_error("Failed to encode page image to JPEG for Vision LLM.");
    }

    return "data:image/jpeg;base64," + base64_encode_bytes(jpg_bytes.data(), jpg_bytes.size());
}

std::string ComicTranslator::encode_bytes_for_vision(
    const uint8_t* image_bytes,
    size_t num_bytes,
    int max_side,
    int jpeg_quality) {
    if (!image_bytes || num_bytes == 0) {
        throw std::invalid_argument("Empty image buffer for vision encoding.");
    }
    int w = 0;
    int h = 0;
    int comp = 0;
    StbiPtr pixels(stbi_load_from_memory(
        image_bytes,
        static_cast<int>(num_bytes),
        &w,
        &h,
        &comp,
        3));
    if (!pixels || w <= 0 || h <= 0) {
        throw std::invalid_argument("Failed to decode image bytes for Vision LLM.");
    }
    return encode_rgb_for_vision(pixels.get(), w, h, max_side, jpeg_quality);
}

// -----------------------------------------------------------------------------
// ComicTranslator Instance Methods
// -----------------------------------------------------------------------------

ComicTranslator::ComicTranslator(const TranslatorConfig& config)
    : config_(config) {
    config_.provider = normalize_provider_id(config_.provider);
    spec_ = &get_provider_spec(config_.provider);
    config_.base_url = normalize_provider_base_url(config_.provider, config_.base_url);
    config_.api_key = resolve_provider_api_key(config_.provider, config_.api_key);
    config_.preferred_model = util::trim_ascii(config_.preferred_model);
    if (config_.preferred_model.empty()) {
        config_.preferred_model = spec_->default_model;
    }
    config_.temperature = clamp_d(config_.temperature, 0.0, 2.0);
    if (config_.top_p.has_value()) {
        config_.top_p = clamp_d(*config_.top_p, 0.0, 1.0);
    }
    if (config_.top_k.has_value()) {
        config_.top_k = clamp_i(*config_.top_k, 0, 500);
    }
    config_.max_tokens = clamp_i(config_.max_tokens, 64, 32768);
    if (config_.frequency_penalty.has_value()) {
        config_.frequency_penalty = clamp_d(*config_.frequency_penalty, -2.0, 2.0);
    }
    config_.reasoning_effort = normalize_reasoning_effort(config_.reasoning_effort);
    config_.vision_max_side = clamp_i(config_.vision_max_side, 256, 2048);
    config_.custom_instructions = util::trim_ascii(config_.custom_instructions);
    config_.timeout_seconds = clamp_d(config_.timeout_seconds, 5.0, 300.0);
}

std::string ComicTranslator::resolve_model(bool force_refresh) {
    if (resolved_model_.has_value() && !force_refresh) {
        return *resolved_model_;
    }
    const std::string resolved = resolve_provider_model(
        config_.provider,
        config_.preferred_model,
        config_.base_url,
        config_.api_key,
        std::min(10.0, config_.timeout_seconds));
    resolved_model_ = resolved;
    return resolved;
}

std::pair<std::string, bool> ComicTranslator::call_anthropic_messages(
    const std::string& model_id,
    const std::string& system_prompt,
    const std::string& user_payload,
    double temperature,
    std::optional<double> top_p,
    std::optional<int> top_k,
    int max_tokens,
    const std::string& reasoning_effort,
    const std::optional<std::string>& page_image_data_uri) const {
    if (config_.api_key.empty()) {
        throw ProviderConnectionError("API key is required for Anthropic (Claude).");
    }
    if (config_.base_url.empty()) {
        throw ProviderConnectionError("Base URL is required for Anthropic (Claude).");
    }

    const std::string messages_url = config_.base_url + "/messages";
    const std::vector<std::pair<std::string, std::string>> headers = {
        {"x-api-key", config_.api_key},
        {"anthropic-version", kDefaultAnthropicVersion},
        {"Content-Type", "application/json"},
        {"Accept", "application/json"},
    };

    const json text_content = json::array({
        {{"type", "text"}, {"text", user_payload}},
    });

    std::optional<json> vision_content;
    if (page_image_data_uri.has_value() && !page_image_data_uri->empty()) {
        if (auto parsed_uri = parse_image_data_uri(*page_image_data_uri)) {
            vision_content = json::array({
                {
                    {"type", "image"},
                    {"source", {
                        {"type", "base64"},
                        {"media_type", parsed_uri->first},
                        {"data", parsed_uri->second},
                    }},
                },
                {{"type", "text"}, {"text", user_payload}},
            });
        }
    }

    auto thinking_budget_for = [](const std::string& eff) -> int {
        if (eff == "low") return 1024;
        if (eff == "medium") return 4096;
        if (eff == "high") return 8192;
        return 0;
    };

    auto post_payload = [&](
        const json& content_blocks,
        bool include_thinking,
        bool include_optional_sampling) -> HttpResponse {
        const int budget = include_thinking ? thinking_budget_for(reasoning_effort) : 0;
        const int eff_max_tokens = (budget > 0) ? std::max(max_tokens, budget + 1024) : max_tokens;

        json body = {
            {"model", model_id},
            {"max_tokens", eff_max_tokens},
            {"system", system_prompt},
            {"messages", json::array({
                {{"role", "user"}, {"content", content_blocks}},
            })},
        };

        if (budget > 0) {
            body["thinking"] = {{"type", "enabled"}, {"budget_tokens", budget}};
        } else {
            body["temperature"] = temperature;
            if (include_optional_sampling) {
                if (top_k.has_value() && *top_k > 0) {
                    body["top_k"] = *top_k;
                } else if (top_p.has_value()) {
                    body["top_p"] = *top_p;
                }
            }
        }

        const HttpResponse resp = perform_http_request(
            "POST", messages_url, headers, safe_json_dump(body), config_.timeout_seconds);
        if (resp.transport_error) {
            if (resp.timed_out) {
                std::ostringstream oss;
                oss << "Anthropic request to " << messages_url << " timed out after "
                    << config_.timeout_seconds << "s.";
                throw ProviderConnectionError(oss.str());
            }
            throw ProviderConnectionError(
                "Failed to reach Anthropic API at " + messages_url + ": " + resp.error_message);
        }
        return resp;
    };

    auto run_with_fallbacks = [&](const json& content_blocks) -> HttpResponse {
        const std::array<std::pair<bool, bool>, 3> attempts = {{
            {reasoning_effort != "off", true},
            {false, true},
            {false, false},
        }};
        std::vector<std::pair<bool, bool>> seen;
        HttpResponse last;
        for (const auto& cfg_pair : attempts) {
            if (std::find(seen.begin(), seen.end(), cfg_pair) != seen.end()) {
                continue;
            }
            seen.push_back(cfg_pair);
            last = post_payload(content_blocks, cfg_pair.first, cfg_pair.second);
            if (last.status_code != 400) {
                return last;
            }
        }
        return last;
    };

    bool vision_used = false;
    HttpResponse resp;
    if (vision_content.has_value()) {
        resp = run_with_fallbacks(*vision_content);
        if (resp.status_code == 400) {
            resp = run_with_fallbacks(text_content);
            vision_used = false;
        } else {
            vision_used = (resp.status_code < 400);
        }
    } else {
        resp = run_with_fallbacks(text_content);
    }

    if (resp.status_code == 401 || resp.status_code == 403) {
        throw ProviderConnectionError(
            "Anthropic authentication failed (HTTP " + std::to_string(resp.status_code) + "): " +
            extract_http_error_detail(resp));
    }
    if (resp.status_code >= 400) {
        throw ProviderTranslationError(
            "Anthropic returned HTTP " + std::to_string(resp.status_code) + ": " +
            extract_http_error_detail(resp));
    }

    const json payload = json::parse(resp.body, nullptr, false);
    if (payload.is_discarded() || !payload.is_object()) {
        throw ProviderTranslationError("Anthropic returned unexpected or non-JSON response payload.");
    }
    if (!payload.contains("content") || !payload["content"].is_array()) {
        throw ProviderTranslationError("Anthropic returned empty content array.");
    }

    std::string combined;
    for (const auto& part : payload["content"]) {
        if (!part.is_object()) continue;
        if (part.value("type", std::string("")) == "text" &&
            part.contains("text") && part["text"].is_string()) {
            const std::string t = util::trim_ascii(part["text"].get<std::string>());
            if (!t.empty()) {
                if (!combined.empty()) combined += "\n";
                combined += t;
            }
        }
    }

    return {combined, vision_used};
}

std::pair<std::string, bool> ComicTranslator::call_openai_chat_completion(
    const std::string& model_id,
    const std::string& system_prompt,
    const std::string& user_payload,
    double temperature,
    std::optional<double> top_p,
    std::optional<int> top_k,
    int max_tokens,
    std::optional<double> frequency_penalty,
    const std::string& reasoning_effort,
    const std::optional<std::string>& page_image_data_uri) const {
    if (config_.base_url.empty()) {
        throw ProviderConnectionError("Provider '" + config_.provider + "' is missing a valid Base URL.");
    }
    if (spec_->requires_api_key && config_.api_key.empty()) {
        throw ProviderConnectionError("API key is required for " + spec_->name + ".");
    }

    const std::string completions_url = config_.base_url + "/chat/completions";
    std::vector<std::pair<std::string, std::string>> headers = {
        {"Content-Type", "application/json"},
        {"Accept", "application/json"},
    };
    for (const auto& kv : spec_->extra_headers) {
        headers.push_back(kv);
    }
    if (!config_.api_key.empty()) {
        headers.emplace_back("Authorization", "Bearer " + config_.api_key);
    }

    const json text_messages = json::array({
        {{"role", "system"}, {"content", system_prompt}},
        {{"role", "user"}, {"content", user_payload}},
    });

    std::optional<json> vision_messages;
    if (page_image_data_uri.has_value() && !page_image_data_uri->empty()) {
        vision_messages = json::array({
            {{"role", "system"}, {"content", system_prompt}},
            {
                {"role", "user"},
                {"content", json::array({
                    {{"type", "text"}, {"text", user_payload}},
                    {{"type", "image_url"}, {"image_url", {{"url", *page_image_data_uri}}}},
                })},
            },
        });
    }

    const std::string openai_effort = (reasoning_effort == "off") ? "none" : reasoning_effort;

    auto send_body = [&](const json& body) -> HttpResponse {
        const HttpResponse resp = perform_http_request(
            "POST", completions_url, headers, safe_json_dump(body), config_.timeout_seconds);
        if (resp.transport_error) {
            if (resp.timed_out) {
                std::ostringstream oss;
                oss << spec_->name << " request to " << config_.base_url
                    << " timed out after " << config_.timeout_seconds << "s.";
                throw ProviderConnectionError(oss.str());
            }
            std::string hint;
            if (config_.provider == "lmstudio") {
                hint = "Make sure LM Studio is running and listening on port 1235.";
            } else if (config_.provider == "ollama") {
                hint = "Make sure Ollama is running locally.";
            } else {
                hint = "Verify Base URL and network connectivity (" + config_.base_url + ").";
            }
            throw ProviderConnectionError(
                "Failed to reach " + spec_->name + " at " + config_.base_url + ". " + hint);
        }
        return resp;
    };

    auto try_messages = [&](const json& messages) -> HttpResponse {
        // Attempt 1: Full structured schema + sampling + reasoning_effort + top_k + frequency_penalty
        {
            json body = {
                {"model", model_id},
                {"messages", messages},
                {"response_format", spec_->supports_json_schema
                                        ? comic_translation_json_schema()
                                        : json({{"type", "json_object"}})},
                {"temperature", temperature},
                {"max_tokens", max_tokens},
            };
            if (top_p.has_value()) {
                body["top_p"] = *top_p;
            }
            if (spec_->supports_top_k && top_k.has_value() && *top_k > 0) {
                body["top_k"] = *top_k;
            }
            if (spec_->supports_frequency_penalty && frequency_penalty.has_value()) {
                body["frequency_penalty"] = *frequency_penalty;
            }
            if (spec_->supports_reasoning_effort) {
                body["reasoning_effort"] = openai_effort;
            }

            HttpResponse resp = send_body(body);
            if (resp.status_code != 400) {
                return resp;
            }
        }

        // Attempt 2: Preserve strict JSON schema without optional reasoning_effort / top_k / frequency_penalty
        if (spec_->supports_json_schema) {
            json body = {
                {"model", model_id},
                {"messages", messages},
                {"response_format", comic_translation_json_schema()},
                {"temperature", temperature},
                {"max_tokens", max_tokens},
            };
            if (top_p.has_value()) {
                body["top_p"] = *top_p;
            }
            HttpResponse resp = send_body(body);
            if (resp.status_code != 400) {
                return resp;
            }
        }

        // Attempt 3: `json_object` + basic sampling
        {
            json body = {
                {"model", model_id},
                {"messages", messages},
                {"response_format", {{"type", "json_object"}}},
                {"temperature", temperature},
                {"max_tokens", max_tokens},
            };
            if (top_p.has_value()) {
                body["top_p"] = *top_p;
            }
            HttpResponse resp = send_body(body);
            if (resp.status_code != 400) {
                return resp;
            }
        }

        // Attempt 4: Plain chat completion with basic temperature + max_tokens
        {
            json body = {
                {"model", model_id},
                {"messages", messages},
                {"temperature", temperature},
                {"max_tokens", max_tokens},
            };
            HttpResponse resp = send_body(body);
            if (resp.status_code != 400) {
                return resp;
            }
        }

        // Attempt 5: Reasoning models (e.g. OpenAI o1/o3/o4) that require `max_completion_tokens`
        {
            json body = {
                {"model", model_id},
                {"messages", messages},
                {"max_completion_tokens", max_tokens},
            };
            return send_body(body);
        }
    };

    bool vision_used = false;
    HttpResponse resp;
    if (vision_messages.has_value()) {
        resp = try_messages(*vision_messages);
        if (resp.status_code == 400) {
            // Graceful fallback to text-only if the active model rejects multimodal payloads
            resp = try_messages(text_messages);
            vision_used = false;
        } else {
            vision_used = (resp.status_code < 400);
        }
    } else {
        resp = try_messages(text_messages);
        vision_used = false;
    }

    if (resp.status_code == 401 || resp.status_code == 403) {
        throw ProviderConnectionError(
            "Authentication failed for " + spec_->name + " (" + config_.base_url + "): " +
            extract_http_error_detail(resp));
    }
    if (resp.status_code >= 400) {
        throw ProviderTranslationError(
            spec_->name + " returned HTTP " + std::to_string(resp.status_code) + ": " +
            extract_http_error_detail(resp));
    }

    const json payload = json::parse(resp.body, nullptr, false);
    if (payload.is_discarded() || !payload.is_object()) {
        throw ProviderTranslationError(
            spec_->name + " returned non-JSON chat completion response.");
    }
    if (!payload.contains("choices") || !payload["choices"].is_array() ||
        payload["choices"].empty()) {
        throw ProviderTranslationError(
            spec_->name + " returned an empty choices array.");
    }

    const auto& first_choice = payload["choices"][0];
    std::string content_str;
    if (first_choice.is_object() && first_choice.contains("message") &&
        first_choice["message"].is_object()) {
        const auto& msg = first_choice["message"];
        if (msg.contains("content")) {
            if (msg["content"].is_string()) {
                content_str = util::trim_ascii(msg["content"].get<std::string>());
            } else if (msg["content"].is_array()) {
                for (const auto& part : msg["content"]) {
                    if (part.is_object() && part.contains("text") && part["text"].is_string()) {
                        const std::string t = util::trim_ascii(part["text"].get<std::string>());
                        if (!t.empty()) {
                            if (!content_str.empty()) content_str += "\n";
                            content_str += t;
                        }
                    }
                }
            }
        }
        if (content_str.empty()) {
            for (const char* rk : {"reasoning_content", "reasoning"}) {
                if (msg.contains(rk) && msg[rk].is_string()) {
                    const std::string r = util::trim_ascii(msg[rk].get<std::string>());
                    if (!r.empty()) {
                        content_str = r;
                        break;
                    }
                }
            }
        }
    }

    return {content_str, vision_used};
}

PageTranslationResult ComicTranslator::translate_blocks(
    const std::vector<DialogueBlockInput>& blocks,
    const TranslateRequestOptions& options) {
    const std::string src_lang = normalize_source_language(options.source_lang);
    const std::string tgt_lang = normalize_target_language(options.target_lang);

    std::vector<DialogueBlockInput> normalized_blocks;
    normalized_blocks.reserve(blocks.size());
    for (size_t i = 0; i < blocks.size(); ++i) {
        std::string bid = util::trim_ascii(blocks[i].id);
        if (bid.empty()) {
            char buf[32]{};
            std::snprintf(buf, sizeof(buf), "bubble_%03zu", i + 1);
            bid = buf;
        }
        normalized_blocks.push_back({bid, collapse_whitespace(blocks[i].source_text)});
    }

    const std::string model_id = resolve_model();

    const std::string eff_reasoning = options.reasoning_effort.has_value()
        ? normalize_reasoning_effort(*options.reasoning_effort)
        : config_.reasoning_effort;
    const double eff_temp = options.temperature.has_value()
        ? clamp_d(*options.temperature, 0.0, 2.0)
        : config_.temperature;
    const std::optional<double> eff_top_p = options.top_p.has_value()
        ? std::optional<double>(clamp_d(*options.top_p, 0.0, 1.0))
        : config_.top_p;
    const std::optional<int> eff_top_k = options.top_k.has_value()
        ? std::optional<int>(clamp_i(*options.top_k, 0, 500))
        : config_.top_k;
    const int eff_max_tokens = options.max_tokens.has_value()
        ? clamp_i(*options.max_tokens, 64, 32768)
        : config_.max_tokens;
    const std::optional<double> eff_freq_pen = options.frequency_penalty.has_value()
        ? std::optional<double>(clamp_d(*options.frequency_penalty, -2.0, 2.0))
        : config_.frequency_penalty;
    const bool eff_use_vision = options.use_vision.value_or(config_.use_vision);

    PageTranslationResult result;
    result.provider = config_.provider;
    result.model = model_id;
    result.source_language = src_lang;
    result.target_language = tgt_lang;
    result.vision_used = false;
    result.reasoning_effort = eff_reasoning;

    if (normalized_blocks.empty()) {
        return result;
    }

    std::vector<DialogueBlockInput> non_empty_blocks;
    for (const auto& b : normalized_blocks) {
        if (!b.source_text.empty()) {
            non_empty_blocks.push_back(b);
        }
    }
    if (non_empty_blocks.empty()) {
        for (const auto& b : normalized_blocks) {
            result.translations[b.id] = "";
            result.dialogues.push_back({b.id, "", ""});
        }
        return result;
    }

    // Merge instance-level glossary with per-request glossary overrides
    std::map<std::string, std::string> merged_glossary = config_.glossary;
    if (options.glossary.has_value()) {
        for (const auto& [k, v] : *options.glossary) {
            const std::string sk = util::trim_ascii(k);
            const std::string sv = util::trim_ascii(v);
            if (!sk.empty() && !sv.empty()) {
                merged_glossary[sk] = sv;
            }
        }
    }

    const std::string eff_instructions = options.custom_instructions.has_value()
        ? util::trim_ascii(*options.custom_instructions)
        : config_.custom_instructions;

    std::optional<std::string> resolved_image_uri;
    if (eff_use_vision && options.page_image_data_uri.has_value() &&
        !util::trim_ascii(*options.page_image_data_uri).empty()) {
        resolved_image_uri = util::trim_ascii(*options.page_image_data_uri);
    }

    const std::string system_prompt = build_system_prompt(
        src_lang,
        tgt_lang,
        eff_reasoning,
        eff_instructions,
        merged_glossary,
        resolved_image_uri.has_value());

    json dialogues_json = json::array();
    for (const auto& b : non_empty_blocks) {
        dialogues_json.push_back({
            {"id", b.id},
            {"source_text", b.source_text},
        });
    }
    const std::string user_payload = safe_json_dump({{"dialogues", std::move(dialogues_json)}});

    std::string raw_output;
    bool vision_used = false;
    if (spec_->protocol == "anthropic") {
        std::tie(raw_output, vision_used) = call_anthropic_messages(
            model_id,
            system_prompt,
            user_payload,
            eff_temp,
            eff_top_p,
            eff_top_k,
            eff_max_tokens,
            eff_reasoning,
            resolved_image_uri);
    } else {
        std::tie(raw_output, vision_used) = call_openai_chat_completion(
            model_id,
            system_prompt,
            user_payload,
            eff_temp,
            eff_top_p,
            eff_top_k,
            eff_max_tokens,
            eff_freq_pen,
            eff_reasoning,
            resolved_image_uri);
    }

    auto translation_map = extract_and_clean_translations(
        raw_output,
        normalized_blocks,
        config_.strip_thinking);

    // Apply deterministic post-translation glossary replacement rules if any source terms remain
    if (!merged_glossary.empty()) {
        for (auto& [bid, translated] : translation_map) {
            if (!translated.empty()) {
                translated = apply_glossary_replacements(translated, merged_glossary);
            }
        }
    }

    result.vision_used = vision_used;
    result.translations = translation_map;
    result.dialogues.reserve(normalized_blocks.size());
    for (const auto& b : normalized_blocks) {
        const auto it = translation_map.find(b.id);
        const std::string trans = (it != translation_map.end()) ? it->second : b.source_text;
        result.dialogues.push_back({b.id, b.source_text, trans});
    }

    return result;
}

}  // namespace mangaflow::pipeline
