#pragma once

// Shared lightweight Windows/UTF-8/filesystem/encoding utilities for MangaFlow AI C++17 modules.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mangaflow::util {

namespace fs = std::filesystem;

inline std::string wide_to_utf8(const std::wstring& wide) {
    if (wide.empty()) {
        return {};
    }
    const int needed = WideCharToMultiByte(
        CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()),
        nullptr, 0, nullptr, nullptr);
    if (needed <= 0) {
        return {};
    }
    std::string out(static_cast<size_t>(needed), '\0');
    WideCharToMultiByte(
        CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()),
        out.data(), needed, nullptr, nullptr);
    return out;
}

inline std::wstring utf8_to_wide(std::string_view utf8) {
    if (utf8.empty()) {
        return {};
    }
    const int needed = MultiByteToWideChar(
        CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
    if (needed <= 0) {
        return {};
    }
    std::wstring out(static_cast<size_t>(needed), L'\0');
    MultiByteToWideChar(
        CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()),
        out.data(), needed);
    return out;
}

inline std::string path_to_utf8(const fs::path& path) {
    return wide_to_utf8(path.wstring());
}

inline std::optional<std::wstring> get_env_wide(const wchar_t* name) {
    const DWORD len = GetEnvironmentVariableW(name, nullptr, 0);
    if (len == 0) {
        return std::nullopt;
    }
    std::wstring buffer(static_cast<size_t>(len), L'\0');
    const DWORD written = GetEnvironmentVariableW(name, buffer.data(), len);
    if (written == 0) {
        return std::nullopt;
    }
    buffer.resize(static_cast<size_t>(written));
    return buffer;
}

inline std::optional<std::string> get_env_utf8(const wchar_t* name) {
    auto wide = get_env_wide(name);
    if (!wide.has_value()) {
        return std::nullopt;
    }
    return wide_to_utf8(*wide);
}

inline void set_env_utf8(const wchar_t* name, const std::string& value) {
    const std::wstring wide_val = utf8_to_wide(value);
    SetEnvironmentVariableW(name, wide_val.c_str());
}

inline std::string to_lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

inline std::string trim_ascii(std::string_view sv) {
    size_t start = 0;
    while (start < sv.size() && std::isspace(static_cast<unsigned char>(sv[start]))) {
        ++start;
    }
    size_t end = sv.size();
    while (end > start && std::isspace(static_cast<unsigned char>(sv[end - 1]))) {
        --end;
    }
    return std::string(sv.substr(start, end - start));
}

inline bool env_flag_enabled(const wchar_t* name) {
    auto val = get_env_utf8(name);
    if (!val.has_value()) {
        return false;
    }
    const std::string lowered = to_lower_ascii(trim_ascii(*val));
    return lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on";
}

inline fs::path get_local_app_data_dir() {
    if (auto lad = get_env_wide(L"LOCALAPPDATA")) {
        return fs::path(*lad);
    }
    if (auto profile = get_env_wide(L"USERPROFILE")) {
        return fs::path(*profile) / L"AppData" / L"Local";
    }
    std::error_code ec;
    return fs::temp_directory_path(ec);
}

inline fs::path get_executable_dir() {
    std::wstring buffer(MAX_PATH, L'\0');
    while (true) {
        const DWORD len = GetModuleFileNameW(
            nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (len == 0) {
            std::error_code ec;
            return fs::current_path(ec);
        }
        if (len < buffer.size()) {
            buffer.resize(static_cast<size_t>(len));
            break;
        }
        buffer.resize(buffer.size() * 2);
    }
    std::error_code ec;
    fs::path exe_path = fs::weakly_canonical(fs::path(buffer), ec);
    if (ec) {
        exe_path = fs::path(buffer);
    }
    return exe_path.parent_path();
}

inline fs::path get_portable_root_dir() {
    const fs::path exe_dir = get_executable_dir();
    auto has_root_markers = [](const fs::path& dir) -> bool {
        std::error_code ec;
        return fs::is_directory(dir / L"models", ec) ||
               fs::is_regular_file(dir / L"settings.json", ec) ||
               fs::is_directory(dir / L"backend-cpp", ec);
    };

    if (has_root_markers(exe_dir)) {
        return exe_dir;
    }

    // Support running from `Manga/MangaFlow_Cpp/` or `Manga/backend-cpp/build/Release/`
    // by walking up to the portable root `Manga/` directory if present.
    fs::path cursor = exe_dir;
    for (int depth = 0; depth < 4; ++depth) {
        fs::path parent = cursor.parent_path();
        if (parent.empty() || parent == cursor) {
            break;
        }
        if (has_root_markers(parent)) {
            return parent;
        }
        cursor = parent;
    }

    return exe_dir;
}

inline fs::path get_comic_translator_app_dir() {
    return get_portable_root_dir();
}

inline std::string generate_uuid_hex() {
    thread_local std::mt19937_64 rng([]() {
        std::random_device rd;
        std::seed_seq seq{rd(), rd(), rd(), rd()};
        return std::mt19937_64(seq);
    }());
    std::uniform_int_distribution<uint64_t> dist;
    const uint64_t hi = dist(rng);
    const uint64_t lo = dist(rng);

    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(32, '0');
    for (int i = 0; i < 16; ++i) {
        out[static_cast<size_t>(i)] = kHex[(hi >> (60 - i * 4)) & 0xF];
        out[static_cast<size_t>(i + 16)] = kHex[(lo >> (60 - i * 4)) & 0xF];
    }
    return out;
}

inline bool atomic_write_bytes(
    const fs::path& target_path,
    const void* data,
    size_t size,
    std::string* error_out = nullptr) {
    std::error_code ec;
    if (target_path.has_parent_path()) {
        fs::create_directories(target_path.parent_path(), ec);
        if (ec) {
            if (error_out) {
                *error_out = "Failed to create directory: " + ec.message();
            }
            return false;
        }
    }

    const fs::path tmp_path = fs::path(target_path.wstring() + L"." + utf8_to_wide(generate_uuid_hex().substr(0, 8)) + L".tmp");
    {
        std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open()) {
            if (error_out) {
                *error_out = "Failed to open temporary file for writing: " + path_to_utf8(tmp_path);
            }
            return false;
        }
        if (size > 0) {
            ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
        }
        ofs.flush();
        if (!ofs.good()) {
            ofs.close();
            fs::remove(tmp_path, ec);
            if (error_out) {
                *error_out = "Failed while writing temporary file: " + path_to_utf8(tmp_path);
            }
            return false;
        }
    }

    if (!MoveFileExW(
            tmp_path.c_str(),
            target_path.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const DWORD win_err = GetLastError();
        fs::remove(tmp_path, ec);
        if (error_out) {
            *error_out = "Atomic replace failed (Win32 error " + std::to_string(win_err) + ")";
        }
        return false;
    }
    return true;
}

inline std::optional<std::vector<uint8_t>> read_file_bytes(const fs::path& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        return std::nullopt;
    }
    const auto sz = ifs.tellg();
    if (sz < 0) {
        return std::nullopt;
    }
    std::vector<uint8_t> data(static_cast<size_t>(sz));
    ifs.seekg(0, std::ios::beg);
    if (sz > 0 && !ifs.read(reinterpret_cast<char*>(data.data()), sz)) {
        return std::nullopt;
    }
    return data;
}

inline std::string url_encode_component(std::string_view input) {
    static constexpr char kHex[] = "0123456789ABCDEF";
    std::string out;
    out.reserve(input.size() * 2);
    for (unsigned char c : input) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~') {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('%');
            out.push_back(kHex[(c >> 4) & 0xF]);
            out.push_back(kHex[c & 0xF]);
        }
    }
    return out;
}

inline std::string url_decode_component(std::string_view input) {
    auto hex_val = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    std::string out;
    out.reserve(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i] == '%' && i + 2 < input.size()) {
            const int hi = hex_val(input[i + 1]);
            const int lo = hex_val(input[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(static_cast<char>((hi << 4) | lo));
                i += 2;
                continue;
            }
        }
        out.push_back(input[i]);
    }
    return out;
}

inline std::optional<std::vector<uint8_t>> base64_decode(std::string_view encoded) {
    static const std::array<int8_t, 256> kTable = []() {
        std::array<int8_t, 256> t{};
        t.fill(-1);
        for (int i = 0; i < 26; ++i) {
            t[static_cast<size_t>('A' + i)] = static_cast<int8_t>(i);
            t[static_cast<size_t>('a' + i)] = static_cast<int8_t>(26 + i);
        }
        for (int i = 0; i < 10; ++i) {
            t[static_cast<size_t>('0' + i)] = static_cast<int8_t>(52 + i);
        }
        t[static_cast<size_t>('+')] = 62;
        t[static_cast<size_t>('/')] = 63;
        t[static_cast<size_t>('-')] = 62;  // URL-safe tolerance
        t[static_cast<size_t>('_')] = 63;
        return t;
    }();

    std::vector<uint8_t> out;
    out.reserve((encoded.size() * 3) / 4);

    uint32_t accum = 0;
    int bits = -8;
    for (unsigned char c : encoded) {
        if (std::isspace(c)) {
            continue;
        }
        if (c == '=') {
            break;
        }
        const int8_t val = kTable[c];
        if (val < 0) {
            return std::nullopt;
        }
        accum = (accum << 6) | static_cast<uint32_t>(val);
        bits += 6;
        if (bits >= 0) {
            out.push_back(static_cast<uint8_t>((accum >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

inline bool is_path_within_root(const fs::path& root_canonical, const fs::path& candidate_canonical) {
    auto root_it = root_canonical.begin();
    auto cand_it = candidate_canonical.begin();
    for (; root_it != root_canonical.end(); ++root_it, ++cand_it) {
        if (cand_it == candidate_canonical.end()) {
            return false;
        }
        if (to_lower_ascii(path_to_utf8(*root_it)) != to_lower_ascii(path_to_utf8(*cand_it))) {
            return false;
        }
    }
    return true;
}

}  // namespace mangaflow::util
