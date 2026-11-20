// Vellum — One-Click Native Win32 Launcher (`Vellum.exe`)
//
// Zero-console, zero-.NET, <5ms native Win32 launcher placed in the root
// folder. Double-clicking it immediately spawns
// `Vellum_Runtime\Vellum.exe` with `Vellum_Runtime\` as its working
// directory and forwards any CLI arguments (such as `--headless`).

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <string>

namespace {

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

}  // namespace

int WINAPI wWinMain(
    HINSTANCE /*hInstance*/,
    HINSTANCE /*hPrevInstance*/,
    PWSTR lpCmdLine,
    int /*nCmdShow*/) {
    const std::wstring root_dir = get_executable_dir();
    const std::wstring app_dir = root_dir + L"\\Vellum_Runtime";
    const std::wstring target_exe = app_dir + L"\\Vellum.exe";

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
        app_dir.c_str(),
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
    CloseHandle(pi.hProcess);
    return 0;
}
