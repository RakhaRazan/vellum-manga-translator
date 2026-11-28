// Vellum - Native Modern C++17 Backend & WebView2 Desktop Host

#include <httplib.h>
#include <windows.h>
#include <dwmapi.h>
#include <onnxruntime_cxx_api.h>
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4005)
#endif
#include "webview/webview.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "app_context.hpp"
#include "pipeline/ort_provider.hpp"
#include "routes/routes_export.hpp"
#include "routes/routes_pipeline.hpp"
#include "routes/routes_projects.hpp"
#include "routes/routes_system.hpp"
#include "utils.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace mangaflow {

void register_routes(httplib::Server& server, AppContext& ctx) {
    routes::register_system_routes(server, ctx);
    routes::register_pipeline_routes(server, ctx);
    routes::register_projects_routes(server, ctx);
    routes::register_export_routes(server, ctx);

    // Clean JSON fallback for unknown /api/* routes
    server.Get(R"(/api/.*)", [](const httplib::Request& req, httplib::Response& res) {
        send_json(res, {{"detail", "API route not found: " + req.path}}, 404);
    });
    server.Post(R"(/api/.*)", [](const httplib::Request& req, httplib::Response& res) {
        send_json(res, {{"detail", "API route not found: " + req.path}}, 404);
    });

    // Static frontend asset handler (`frontend/dist` mounted at `/`)
    server.Get(R"(/(.*))", [&ctx](const httplib::Request& req, httplib::Response& res) {
        std::string rel = req.path;
        if (!rel.empty() && rel.front() == '/') {
            rel.erase(rel.begin());
        }

        if (rel == "api" || rel.rfind("api/", 0) == 0) {
            send_json(res, {{"detail", "Not found"}}, 404);
            return;
        }

        fs::path target = rel.empty()
            ? (ctx.frontend_dist_dir / L"index.html")
            : (ctx.frontend_dist_dir / util::utf8_to_wide(rel));

        std::error_code ec;
        fs::path canonical_target = fs::weakly_canonical(target, ec);
        if (ec || !util::is_path_within_root(ctx.frontend_dist_dir, canonical_target)) {
            res.status = 403;
            res.set_content("Forbidden", "text/plain; charset=utf-8");
            return;
        }

        if (fs::is_directory(canonical_target, ec)) {
            canonical_target /= L"index.html";
        }

        if (!fs::is_regular_file(canonical_target, ec)) {
            if (!target.has_extension()) {
                canonical_target = ctx.frontend_dist_dir / L"index.html";
            } else {
                res.status = 404;
                res.set_content("Not Found", "text/plain; charset=utf-8");
                return;
            }
        }

        if (!serve_file_from_disk(canonical_target, req.path, res)) {
            res.status = 500;
            res.set_content("Failed to read static asset", "text/plain; charset=utf-8");
        }
    });
}

// -----------------------------------------------------------------------------
// Server Binding & Readiness Synchronization
// -----------------------------------------------------------------------------

bool is_loopback_port_available(const char* host, int port) {
    if (port <= 0 || port > 65535) {
        return false;
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<u_short>(port));
    if (InetPtonA(AF_INET, host, &addr.sin_addr) != 1) {
        return false;
    }

    // Fast path: check exclusive bind first (<0.1 ms).
    SOCKET test_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (test_sock == INVALID_SOCKET) {
        return false;
    }
    const int exclusive = 1;
    setsockopt(
        test_sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
        reinterpret_cast<const char*>(&exclusive), sizeof(exclusive));
    const bool can_bind =
        bind(test_sock, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) == 0;
    closesocket(test_sock);
    if (!can_bind) {
        return false;
    }

    // Secondary guard for listeners bound on 0.0.0.0 with SO_REUSEADDR.
    SOCKET probe = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (probe != INVALID_SOCKET) {
        u_long nonblocking = 1;
        ioctlsocket(probe, FIONBIO, &nonblocking);
        const int conn_res = connect(
            probe, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr));
        bool in_use = (conn_res == 0);
        if (!in_use && WSAGetLastError() == WSAEWOULDBLOCK) {
            fd_set write_fds{};
            fd_set except_fds{};
            FD_ZERO(&write_fds);
            FD_ZERO(&except_fds);
            FD_SET(probe, &write_fds);
            FD_SET(probe, &except_fds);
            timeval tv{};
            tv.tv_sec = 0;
            tv.tv_usec = 30000;  // 30 ms is plenty for loopback TCP SYN-ACK
            if (select(0, nullptr, &write_fds, &except_fds, &tv) > 0 &&
                FD_ISSET(probe, &write_fds)) {
                int so_err = 0;
                int opt_len = sizeof(so_err);
                if (getsockopt(
                        probe, SOL_SOCKET, SO_ERROR,
                        reinterpret_cast<char*>(&so_err), &opt_len) == 0 &&
                    so_err == 0) {
                    in_use = true;
                }
            }
        }
        closesocket(probe);
        if (in_use) {
            return false;
        }
    }

    return true;
}

int bind_server_port(httplib::Server& server, int preferred_port) {
    server.set_socket_options([](socket_t sock) {
        const int exclusive = 1;
        setsockopt(
            sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
            reinterpret_cast<const char*>(&exclusive), sizeof(exclusive));
    });

    if (is_loopback_port_available(kHost, preferred_port)) {
        if (server.bind_to_port(kHost, preferred_port)) {
            return preferred_port;
        }
        server.stop();
    }
    return server.bind_to_any_port(kHost);
}

bool wait_for_server_ready(int port, std::atomic<bool>& server_failed, int timeout_ms = 5000) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    while (std::chrono::steady_clock::now() < deadline) {
        if (server_failed.load()) {
            return false;
        }
        httplib::Client probe(kHost, port);
        probe.set_connection_timeout(0, 150000);
        probe.set_read_timeout(0, 250000);
        if (auto res = probe.Get("/api/ping"); res && res->status == 200) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

// -----------------------------------------------------------------------------
// Native Win32 Window Styling
// -----------------------------------------------------------------------------

void apply_dark_window_theme(void* native_window) {
    if (!native_window) return;
    HWND hwnd = static_cast<HWND>(native_window);
    const BOOL use_dark_mode = TRUE;
    if (FAILED(DwmSetWindowAttribute(hwnd, 20, &use_dark_mode, sizeof(use_dark_mode)))) {
        DwmSetWindowAttribute(hwnd, 19, &use_dark_mode, sizeof(use_dark_mode));
    }
}

void apply_window_icon(void* native_window) {
    if (!native_window) return;
    HWND hwnd = static_cast<HWND>(native_window);
    HINSTANCE hinst = GetModuleHandleW(nullptr);
    if (HICON hicon_big = static_cast<HICON>(LoadImageW(
            hinst, MAKEINTRESOURCEW(101), IMAGE_ICON,
            GetSystemMetrics(SM_CXICON), GetSystemMetrics(SM_CYICON), LR_DEFAULTCOLOR))) {
        SendMessageW(hwnd, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(hicon_big));
    }
    if (HICON hicon_small = static_cast<HICON>(LoadImageW(
            hinst, MAKEINTRESOURCEW(101), IMAGE_ICON,
            GetSystemMetrics(SM_CXSMICON), GetSystemMetrics(SM_CYSMICON), LR_DEFAULTCOLOR))) {
        SendMessageW(hwnd, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(hicon_small));
    }
}

void show_error_dialog(const std::wstring& message) {
    MessageBoxW(
        nullptr,
        message.c_str(),
        L"Vellum - Startup Error",
        MB_OK | MB_ICONERROR);
}

}  // namespace mangaflow

void register_cuda_dll_paths(const fs::path& exe_dir) {
    std::vector<fs::path> roots;
    roots.push_back(exe_dir / "cuda");
    if (auto cuda_env = mangaflow::util::get_env_wide(L"CUDA_PATH"); cuda_env.has_value() && !cuda_env->empty()) {
        roots.emplace_back(*cuda_env);
    }
    roots.emplace_back(L"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA");
    if (auto local_app = mangaflow::util::get_env_wide(L"LOCALAPPDATA"); local_app.has_value() && !local_app->empty()) {
        roots.emplace_back(fs::path(*local_app) / L"LM-Studio");
    }
    if (auto user_prof = mangaflow::util::get_env_wide(L"USERPROFILE"); user_prof.has_value() && !user_prof->empty()) {
        roots.emplace_back(fs::path(*user_prof) / L".lmstudio");
    }

    std::wstring extra_paths;
    std::error_code ec;
    for (const auto& root : roots) {
        if (!fs::is_directory(root, ec) || ec) continue;
        for (const auto& entry : fs::recursive_directory_iterator(
                 root, fs::directory_options::skip_permission_denied, ec)) {
            if (ec) break;
            if (!entry.is_directory(ec) || ec) continue;
            const std::wstring name = entry.path().filename().wstring();
            if (name == L"bin" || name == L"x86_64" || name == L"cuda12") {
                extra_paths += entry.path().wstring();
                extra_paths += L";";
            }
        }
    }

    if (!extra_paths.empty()) {
        const DWORD len = GetEnvironmentVariableW(L"PATH", nullptr, 0);
        std::wstring cur_path(len > 0 ? len : 0, L'\0');
        if (len > 0) {
            GetEnvironmentVariableW(L"PATH", cur_path.data(), len);
            if (!cur_path.empty() && cur_path.back() == L'\0') cur_path.pop_back();
        }
        const std::wstring combined = extra_paths + cur_path;
        SetEnvironmentVariableW(L"PATH", combined.c_str());
    }
}

// -----------------------------------------------------------------------------
// Win32 GUI Entry Point (Zero Black Console Window)
// -----------------------------------------------------------------------------

int WINAPI WinMain(
    HINSTANCE /*hInst*/,
    HINSTANCE /*hPrevInst*/,
    LPSTR lpCmdLine,
    int /*nCmdShow*/) {
    using namespace mangaflow;

    const std::string cmd_line = lpCmdLine ? std::string(lpCmdLine) : std::string();
    const bool headless = util::env_flag_enabled(L"VELLUM_HEADLESS") ||
                          util::env_flag_enabled(L"MANGAFLOW_HEADLESS") ||
                          cmd_line.find("--headless") != std::string::npos;

    const fs::path exe_dir = get_executable_dir();
    register_cuda_dll_paths(exe_dir);
    const auto frontend_dist = resolve_frontend_dist(exe_dir);
    if (!frontend_dist.has_value() && !headless) {
        show_error_dialog(
            L"Compiled React v2 frontend bundle (`frontend/dist/index.html`) was not found.\n\n"
            L"Please run `npm run build` inside the `frontend/` directory first.");
        return 1;
    }

    AppContext ctx;
    ctx.exe_dir = exe_dir;
    ctx.frontend_dist_dir = frontend_dist.value_or(exe_dir);
    ctx.ort_version = pipeline::ModelManager::get_ort_version();
    ctx.available_providers = pipeline::ModelManager::get_available_providers();
    ctx.model_manager = std::make_unique<pipeline::ModelManager>(
        std::nullopt,
        pipeline::ModelManager::find_workspace_models_dir(exe_dir));
    ctx.detector = std::make_unique<pipeline::TextDetector>(*ctx.model_manager);
    ctx.ocr_engine = std::make_unique<pipeline::OcrEngine>(*ctx.model_manager);
    ctx.inpainter = std::make_unique<pipeline::LamaInpainter>(*ctx.model_manager);
    ctx.start_unload_watchdog();

    // Pre-load persistent settings so execution_mode (`MANGA_FORCE_CPU`) & preferred_backend are synced at startup
    const json initial_settings = ctx.settings_manager.load();
    const std::string init_exec_mode = initial_settings.value("execution_mode", std::string("hybrid"));
    if (init_exec_mode == "cpu_only" || init_exec_mode == "cpu") {
        util::set_env_utf8(L"MANGA_FORCE_CPU", "1");
    }
    pipeline::set_preferred_backend(
        initial_settings.value("preferred_backend", std::string("auto")));

    httplib::Server server;
    server.set_read_timeout(300, 0);
    server.set_write_timeout(600, 0);
    server.set_keep_alive_max_count(10000);
    server.set_payload_max_length(16ULL * 1024ULL * 1024ULL * 1024ULL);  // 16 GiB for multi-GB 10,000-page archives & exports
    register_routes(server, ctx);

    int preferred_port = kDefaultPort;
    auto env_port = util::get_env_utf8(L"VELLUM_PORT");
    if (!env_port) {
        env_port = util::get_env_utf8(L"MANGAFLOW_PORT");
    }
    if (env_port) {
        try {
            preferred_port = std::stoi(*env_port);
        } catch (...) {
            preferred_port = kDefaultPort;
        }
    }

    const int bound_port = bind_server_port(server, preferred_port);
    if (bound_port <= 0) {
        if (!headless) {
            show_error_dialog(L"Failed to bind local loopback HTTP server on 127.0.0.1.");
        }
        return 1;
    }

    std::atomic<bool> server_failed{false};
    std::thread server_thread([&server, &server_failed]() {
        try {
            if (!server.listen_after_bind()) {
                server_failed.store(true);
            }
        } catch (...) {
            server_failed.store(true);
        }
    });

    auto stop_server = [&server, &server_thread, &ctx]() {
        server.stop();
        if (server_thread.joinable()) {
            server_thread.join();
        }
        ctx.main_hwnd.store(nullptr);
        SetThreadExecutionState(ES_CONTINUOUS);
        ctx.stop_unload_watchdog();
    };

    if (!wait_for_server_ready(bound_port, server_failed)) {
        stop_server();
        if (!headless) {
            show_error_dialog(L"Embedded C++ HTTP backend failed to respond on /api/health.");
        }
        return 1;
    }

    if (headless) {
        // Pure HTTP backend mode for automated verification / CLI hosting
        if (server_thread.joinable()) {
            server_thread.join();
        }
        ctx.stop_unload_watchdog();
        return server_failed.load() ? 1 : 0;
    }

    const std::string app_url =
        "http://" + std::string(kHost) + ":" + std::to_string(bound_port) + "/";

    int exit_code = 0;
    try {
        // WebView2 Memory Discipline: cap V8 old-space heap and allow timely background GC
        {
            const std::wstring kMemArgs =
                L"--js-flags=--max-old-space-size=1024 --disable-background-timer-throttling=false";
            std::wstring cur_args;
            if (auto existing = util::get_env_wide(L"WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS");
                existing.has_value() && !existing->empty()) {
                cur_args = *existing;
            }
            if (cur_args.find(L"--max-old-space-size=") == std::wstring::npos) {
                if (!cur_args.empty()) cur_args.push_back(L' ');
                cur_args += kMemArgs;
                SetEnvironmentVariableW(L"WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", cur_args.c_str());
            }
        }

        const bool debug_devtools = util::env_flag_enabled(L"VELLUM_DEBUG") ||
                                    util::env_flag_enabled(L"MANGAFLOW_DEBUG");
        webview::webview window(debug_devtools, nullptr);

        if (auto native_win = window.window(); native_win.ok()) {
            ctx.main_hwnd.store(native_win.value());
            apply_dark_window_theme(native_win.value());
            apply_window_icon(native_win.value());
        }

        window.set_title(kWindowTitle);
        window.set_size(kWindowMinWidth, kWindowMinHeight, WEBVIEW_HINT_MIN);
        window.set_size(kWindowWidth, kWindowHeight, WEBVIEW_HINT_NONE);
        window.navigate(app_url);
        window.run();
    } catch (const webview::exception& ex) {
        show_error_dialog(
            L"Failed to initialize native Edge WebView2 window:\n\n" +
            util::utf8_to_wide(ex.what()));
        exit_code = 1;
    } catch (const std::exception& ex) {
        show_error_dialog(
            L"Unexpected runtime error:\n\n" +
            util::utf8_to_wide(ex.what()));
        exit_code = 1;
    }

    stop_server();
    return exit_code;
}
