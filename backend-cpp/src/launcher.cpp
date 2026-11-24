// Vellum - Dual-Core Native Win32 Launcher (`Vellum.exe`)
//
// Zero-console, zero-.NET, <5ms native Win32 launcher placed in the root
// folder. Reads `preferred_backend` from `settings.json` and probes
// `nvcuda.dll` (System32) to launch either:
//   - `Vellum_Runtime\Vellum_Core.exe`         (NVIDIA CUDA + CPU core)
//   - `Vellum_Runtime\dml\Vellum_Core_DML.exe` (DirectML + CPU core)
//
// Supports seamless backend switching: if a GUI core exits with exit code 42
// (`POST /api/system/relaunch`), the launcher re-reads `settings.json` and
// immediately spawns the newly selected runtime core.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>

namespace {

constexpr DWORD kRelaunchExitCode = 42;

std::wstring get_executable_dir() {
    std::wstring buf(MAX_PATH, L'\0');
    while (true) {
        const DWORD len = GetModuleFileNameW(
            nullptr, buf.data(), static_cast<DWORD>(buf.size()));
        if (len == 0) {
            return L".";
        }
        if (len < buf.size()) {
            buf.resize(static_cast<size_t>(len));
            break;
        }
        buf.resize(buf.size() * 2);
    }
    const size_t slash = buf.find_last_of(L"\\/");
    return (slash == std::wstring::npos) ? L"." : buf.substr(0, slash);
}

bool file_exists(const std::wstring& path) {
    const DWORD attrs = GetFileAttributesW(path.c_str());
    return attrs != INVALID_FILE_ATTRIBUTES &&
           (attrs & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

bool probe_nvcuda_dll() noexcept {
    HMODULE mod = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (mod != nullptr) {
        FreeLibrary(mod);
        return true;
    }
    return false;
}

std::string to_lower_ascii(std::string s) {
    for (char& ch : s) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }
    return s;
}

std::string read_preferred_backend_from_settings(const std::wstring& root_dir) {
    const std::wstring settings_path = root_dir + L"\\settings.json";
    std::ifstream ifs(settings_path, std::ios::binary);
    if (!ifs.is_open()) {
        return "auto";
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    const std::string content = oss.str();

    const std::string key = "\"preferred_backend\"";
    const size_t pos = content.find(key);
    if (pos == std::string::npos) {
        return "auto";
    }
    const size_t colon = content.find(':', pos + key.size());
    if (colon == std::string::npos) {
        return "auto";
    }
    const size_t q1 = content.find('"', colon + 1);
    if (q1 == std::string::npos) {
        return "auto";
    }
    const size_t q2 = content.find('"', q1 + 1);
    if (q2 == std::string::npos || q2 <= q1 + 1) {
        return "auto";
    }
    return to_lower_ascii(content.substr(q1 + 1, q2 - q1 - 1));
}

bool has_headless_flag(PWSTR lpCmdLine) {
    if (lpCmdLine == nullptr || lpCmdLine[0] == L'\0') {
        return false;
    }
    std::wstring cmd(lpCmdLine);
    for (wchar_t& ch : cmd) {
        if (ch >= L'A' && ch <= L'Z') {
            ch = static_cast<wchar_t>(ch - L'A' + L'a');
        }
    }
    return cmd.find(L"--headless") != std::wstring::npos;
}

}  // namespace

int WINAPI wWinMain(
    HINSTANCE /*hInstance*/,
    HINSTANCE /*hPrevInstance*/,
    PWSTR lpCmdLine,
    int /*nCmdShow*/) {
    const std::wstring root_dir = get_executable_dir();
    const std::wstring runtime_dir = root_dir + L"\\Vellum_Runtime";
    const std::wstring dml_dir = runtime_dir + L"\\dml";
    const bool headless = has_headless_flag(lpCmdLine);
    const bool has_nvidia_driver = probe_nvcuda_dll();

    for (int relaunch_attempt = 0; relaunch_attempt < 32; ++relaunch_attempt) {
        const std::string pref = read_preferred_backend_from_settings(root_dir);

        const std::wstring cuda_core = file_exists(runtime_dir + L"\\Vellum_Core.exe")
            ? (runtime_dir + L"\\Vellum_Core.exe")
            : (runtime_dir + L"\\Vellum.exe");
        const std::wstring dml_core = dml_dir + L"\\Vellum_Core_DML.exe";

        const bool has_cuda_core = file_exists(cuda_core);
        const bool has_dml_core = file_exists(dml_core);

        bool use_dml = false;
        if (pref == "directml" || pref == "dml") {
            use_dml = has_dml_core;
        } else if (pref == "auto" && !has_nvidia_driver && has_dml_core) {
            use_dml = true;
        } else if (!has_cuda_core && has_dml_core) {
            use_dml = true;
        }

        const std::wstring target_exe = use_dml ? dml_core : cuda_core;
        const std::wstring work_dir = use_dml ? dml_dir : runtime_dir;

        if (!file_exists(target_exe)) {
            const std::wstring msg =
                L"Vellum runtime binary was not found at:\n" +
                target_exe +
                L"\n\nPlease rebuild using packaging\\build_cpp_release.bat.";
            MessageBoxW(nullptr, msg.c_str(), L"Vellum", MB_OK | MB_ICONERROR);
            return 1;
        }

        std::wstring cmdline = L"\"" + target_exe + L"\"";
        if (lpCmdLine != nullptr && lpCmdLine[0] != L'\0') {
            cmdline.push_back(L' ');
            cmdline.append(lpCmdLine);
        }

        STARTUPINFOW si{};
        si.cb = sizeof(si);
        PROCESS_INFORMATION pi{};

        const BOOL ok = CreateProcessW(
            target_exe.c_str(),
            cmdline.data(),
            nullptr,
            nullptr,
            FALSE,
            0,
            nullptr,
            work_dir.c_str(),
            &si,
            &pi);

        if (!ok) {
            const DWORD err = GetLastError();
            const std::wstring msg =
                L"Failed to launch Vellum (Win32 error " +
                std::to_wstring(err) + L"):\n" + target_exe;
            MessageBoxW(nullptr, msg.c_str(), L"Vellum", MB_OK | MB_ICONERROR);
            return 1;
        }

        CloseHandle(pi.hThread);
        if (headless) {
            CloseHandle(pi.hProcess);
            return 0;
        }

        // In GUI mode, wait for child exit so exit code 42 seamlessly relaunches the switched backend core.
        WaitForSingleObject(pi.hProcess, INFINITE);
        DWORD exit_code = 0;
        GetExitCodeProcess(pi.hProcess, &exit_code);
        CloseHandle(pi.hProcess);

        if (exit_code != kRelaunchExitCode) {
            return static_cast<int>(exit_code);
        }
    }

    return 0;
}
