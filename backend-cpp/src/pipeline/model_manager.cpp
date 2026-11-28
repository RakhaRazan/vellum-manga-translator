#include "model_manager.hpp"
#include "ort_provider.hpp"
#include "../utils.hpp"

#include <dxgi.h>
#include <malloc.h>
#include <winhttp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace mangaflow::pipeline {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr const char* kCtdFile = "comictextdetector.pt.onnx";

constexpr const char* kLamaMangaFile = "lama_manga.onnx";
constexpr const char* kLamaFallbackFile = "lama_fp32.onnx";

constexpr const char* kMangaOcrEncoderFile = "manga-ocr/encoder_model.onnx";
constexpr const char* kMangaOcrDecoderFile = "manga-ocr/decoder_model.onnx";
constexpr const char* kMangaOcrVocabFile = "manga-ocr/vocab.txt";
constexpr const char* kMangaOcrConfigFile = "manga-ocr/config.json";

constexpr const char* kPaddleVlModelFile = "paddleocr-vl-gguf/PaddleOCR-VL-1.6-GGUF.gguf";
constexpr const char* kPaddleVlMmprojFile = "paddleocr-vl-gguf/PaddleOCR-VL-1.6-GGUF-mmproj.gguf";

constexpr std::uintmax_t kMinCtdBytes = 50'000'000ULL;
constexpr std::uintmax_t kMinLamaBytes = 100'000'000ULL;
constexpr std::uintmax_t kMinMangaOcrEncBytes = 100'000'000ULL;
constexpr std::uintmax_t kMinMangaOcrDecBytes = 30'000'000ULL;
constexpr std::uintmax_t kMinVocabBytes = 1'000ULL;
constexpr std::uintmax_t kMinGgufBytes = 100'000'000ULL;

struct WinHttpHandle {
    HINTERNET h = nullptr;
    WinHttpHandle() = default;
    explicit WinHttpHandle(HINTERNET handle) : h(handle) {}
    ~WinHttpHandle() {
        if (h) {
            WinHttpCloseHandle(h);
        }
    }
    WinHttpHandle(const WinHttpHandle&) = delete;
    WinHttpHandle& operator=(const WinHttpHandle&) = delete;
    explicit operator bool() const noexcept { return h != nullptr; }
};

double round_2(double v) {
    return std::round(v * 100.0) / 100.0;
}

double round_1(double v) {
    return std::round(v * 10.0) / 10.0;
}

std::string format_mb(std::uintmax_t bytes, int decimals = 1) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(decimals)
        << (static_cast<double>(bytes) / (1024.0 * 1024.0));
    return oss.str();
}

bool is_valid_regular_file(const fs::path& path, std::uintmax_t min_bytes, std::uintmax_t* out_size = nullptr) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec) || ec) {
        return false;
    }
    const auto sz = fs::file_size(path, ec);
    if (ec || sz < min_bytes) {
        return false;
    }
    if (out_size) {
        *out_size = sz;
    }
    return true;
}

std::optional<fs::path> find_first_with_ext(
    const fs::path& dir,
    const std::string& ext_lower,
    std::uintmax_t min_bytes,
    const std::string& name_contains_lower = "",
    const std::string& name_excludes_lower = "") {
    std::error_code ec;
    if (!fs::is_directory(dir, ec) || ec) {
        return std::nullopt;
    }
    std::vector<fs::path> matches;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        std::error_code f_ec;
        if (!entry.is_regular_file(f_ec) || f_ec) continue;
        const std::string fname = util::path_to_utf8(entry.path().filename());
        if (fname.empty() || fname.front() == '.') continue;
        const std::string ext = util::to_lower_ascii(util::path_to_utf8(entry.path().extension()));
        if (ext != ext_lower) continue;
        const std::string lower_name = util::to_lower_ascii(fname);
        if (!name_contains_lower.empty() && lower_name.find(name_contains_lower) == std::string::npos) {
            continue;
        }
        if (!name_excludes_lower.empty() && lower_name.find(name_excludes_lower) != std::string::npos) {
            continue;
        }
        const auto sz = fs::file_size(entry.path(), f_ec);
        if (!f_ec && sz >= min_bytes) {
            matches.push_back(entry.path());
        }
    }
    if (matches.empty()) {
        return std::nullopt;
    }
    std::sort(matches.begin(), matches.end());
    return matches.front();
}

struct StandardModelSpec {
    const char* model_key;
    const char* repo_id;
    const char* remote_filename;
    const char* local_rel_path;
    std::uintmax_t expected_bytes;
    std::uintmax_t min_valid_bytes;
};

const std::vector<StandardModelSpec>& standard_download_specs() {
    static const std::vector<StandardModelSpec> kSpecs = {
        // 1. Comic Text Detector ONNX fallback (~90.28 MB)
        {
            "comic_text_detector",
            "mayocream/comic-text-detector-onnx",
            "comic-text-detector.onnx",
            "comictextdetector.pt.onnx",
            94'662'990ULL,
            50'000'000ULL,
        },
        // 2. LaMa Inpainting ONNX fallback (~198.41 MB)
        {
            "lama_inpaint",
            "Carve/LaMa-ONNX",
            "lama_fp32.onnx",
            "lama_fp32.onnx",
            208'049'867ULL,
            100'000'000ULL,
        },
        // 3. RapidOCR PP-OCRv4 Korean & Chinese (~40.70 MB total)
        {
            "rapid_ocr_ko_zh",
            "pitapo/rapidocr",
            "onnx/PP-OCRv4/det/ch_PP-OCRv4_det_infer.onnx",
            "rapidocr/ch_PP-OCRv4_det_infer.onnx",
            4'747'796ULL,
            2'000'000ULL,
        },
        {
            "rapid_ocr_ko_zh",
            "pitapo/rapidocr",
            "onnx/PP-OCRv4/det/Multilingual_PP-OCRv3_det_infer.onnx",
            "rapidocr/Multilingual_PP-OCRv3_det_infer.onnx",
            2'423'876ULL,
            1'000'000ULL,
        },
        {
            "rapid_ocr_ko_zh",
            "pitapo/rapidocr",
            "onnx/PP-OCRv4/rec/ch_PP-OCRv4_rec_infer.onnx",
            "rapidocr/ch_PP-OCRv4_rec_infer.onnx",
            10'852'784ULL,
            5'000'000ULL,
        },
        {
            "rapid_ocr_ko_zh",
            "pitapo/rapidocr",
            "onnx/PP-OCRv4/rec/korean_PP-OCRv4_rec_infer.onnx",
            "rapidocr/korean_PP-OCRv4_rec_infer.onnx",
            24'065'974ULL,
            10'000'000ULL,
        },
        {
            "rapid_ocr_ko_zh",
            "pitapo/rapidocr",
            "onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx",
            "rapidocr/ch_ppocr_mobile_v2.0_cls_infer.onnx",
            585'540ULL,
            100'000ULL,
        },
        // 4. Japanese Manga OCR fallback weights (kha-white/manga-ocr-base)
        {
            "manga_ocr",
            "kha-white/manga-ocr-base",
            "config.json",
            "manga-ocr-base/config.json",
            75'610ULL,
            100ULL,
        },
        {
            "manga_ocr",
            "kha-white/manga-ocr-base",
            "preprocessor_config.json",
            "manga-ocr-base/preprocessor_config.json",
            228ULL,
            50ULL,
        },
        {
            "manga_ocr",
            "kha-white/manga-ocr-base",
            "special_tokens_map.json",
            "manga-ocr-base/special_tokens_map.json",
            112ULL,
            20ULL,
        },
        {
            "manga_ocr",
            "kha-white/manga-ocr-base",
            "tokenizer_config.json",
            "manga-ocr-base/tokenizer_config.json",
            486ULL,
            50ULL,
        },
        {
            "manga_ocr",
            "kha-white/manga-ocr-base",
            "vocab.txt",
            "manga-ocr-base/vocab.txt",
            24'130ULL,
            1'000ULL,
        },
        {
            "manga_ocr",
            "kha-white/manga-ocr-base",
            "pytorch_model.bin",
            "manga-ocr-base/pytorch_model.bin",
            444'135'113ULL,
            400'000'000ULL,
        },
    };
    return kSpecs;
}

std::string build_hf_resolve_url(const std::string& repo_id, const std::string& remote_filename) {
    return "https://huggingface.co/" + repo_id + "/resolve/main/" + remote_filename;
}

bool copy_or_link_file_atomic(
    const fs::path& src,
    const fs::path& dst,
    bool prefer_hardlink,
    std::string& action_out,
    std::string& error_out) {
    std::error_code ec;
    if (dst.has_parent_path()) {
        fs::create_directories(dst.parent_path(), ec);
        if (ec) {
            error_out = "Failed to create directory: " + ec.message();
            return false;
        }
    }

    const fs::path tmp_path = fs::path(
        dst.wstring() + L"." + util::utf8_to_wide(util::generate_uuid_hex().substr(0, 8)) + L".part");

    if (prefer_hardlink) {
        if (CreateHardLinkW(tmp_path.c_str(), src.c_str(), nullptr)) {
            if (MoveFileExW(tmp_path.c_str(), dst.c_str(), MOVEFILE_REPLACE_EXISTING)) {
                action_out = "linked";
                return true;
            }
            fs::remove(tmp_path, ec);
        }
    }

    if (!CopyFileW(src.c_str(), tmp_path.c_str(), FALSE)) {
        const DWORD win_err = GetLastError();
        fs::remove(tmp_path, ec);
        error_out = "CopyFileW failed for " + util::path_to_utf8(src) +
                    " (Win32 error " + std::to_string(win_err) + ")";
        return false;
    }

    if (!MoveFileExW(
            tmp_path.c_str(),
            dst.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const DWORD win_err = GetLastError();
        fs::remove(tmp_path, ec);
        error_out = "MoveFileExW failed for " + util::path_to_utf8(dst) +
                    " (Win32 error " + std::to_string(win_err) + ")";
        return false;
    }

    action_out = "copied";
    return true;
}

std::optional<fs::path> find_koharu_converter_script(const std::optional<fs::path>& search_root) {
    auto check_from = [](fs::path cursor) -> std::optional<fs::path> {
        for (int depth = 0; depth < 7; ++depth) {
            const fs::path candidate = cursor / L"packaging" / L"convert_koharu_to_onnx.py";
            std::error_code ec;
            if (fs::is_regular_file(candidate, ec) && !ec) {
                fs::path canon = fs::weakly_canonical(candidate, ec);
                return ec ? candidate : canon;
            }
            const fs::path parent = cursor.parent_path();
            if (parent.empty() || parent == cursor) {
                break;
            }
            cursor = parent;
        }
        return std::nullopt;
    };

    if (search_root.has_value()) {
        if (auto found = check_from(*search_root)) {
            return found;
        }
    }
    std::error_code cwd_ec;
    if (auto cwd = fs::current_path(cwd_ec); !cwd_ec) {
        if (auto found = check_from(cwd)) {
            return found;
        }
    }
    return std::nullopt;
}

std::optional<fs::path> find_python_executable(const fs::path& repo_root) {
    std::error_code ec;
    auto env_py = util::get_env_wide(L"VELLUM_PYTHON_EXE");
    if (!env_py.has_value() || env_py->empty()) {
        env_py = util::get_env_wide(L"MANGA_PYTHON_EXE");
    }
    if (env_py.has_value() && !env_py->empty()) {
        const fs::path p(*env_py);
        if (fs::is_regular_file(p, ec) && !ec) {
            return p;
        }
    }

    for (const fs::path& candidate : {
             repo_root / L"backend" / L"venv" / L"Scripts" / L"python.exe",
             repo_root / L".venv" / L"Scripts" / L"python.exe",
             repo_root / L"venv" / L"Scripts" / L"python.exe",
         }) {
        if (fs::is_regular_file(candidate, ec) && !ec) {
            fs::path canon = fs::weakly_canonical(candidate, ec);
            return ec ? candidate : canon;
        }
    }

    wchar_t found_buf[MAX_PATH * 4]{};
    const DWORD len = SearchPathW(
        nullptr, L"python.exe", nullptr, static_cast<DWORD>(std::size(found_buf)), found_buf, nullptr);
    if (len > 0 && len < std::size(found_buf)) {
        return fs::path(found_buf);
    }
    return std::nullopt;
}

std::wstring quote_win_arg(const std::wstring& arg) {
    if (arg.empty()) {
        return L"\"\"";
    }
    if (arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        return arg;
    }
    std::wstring out = L"\"";
    size_t backslashes = 0;
    for (wchar_t ch : arg) {
        if (ch == L'\\') {
            ++backslashes;
        } else if (ch == L'"') {
            out.append(backslashes * 2 + 1, L'\\');
            out.push_back(L'"');
            backslashes = 0;
        } else {
            out.append(backslashes, L'\\');
            out.push_back(ch);
            backslashes = 0;
        }
    }
    out.append(backslashes * 2, L'\\');
    out.push_back(L'"');
    return out;
}

bool run_subprocess_capture(
    const std::vector<std::wstring>& argv,
    const fs::path& working_dir,
    int& exit_code_out,
    std::string& output_out,
    std::string& error_out) {
    exit_code_out = -1;
    output_out.clear();

    if (argv.empty()) {
        error_out = "Empty subprocess argv.";
        return false;
    }

    std::wstring cmdline;
    for (size_t i = 0; i < argv.size(); ++i) {
        if (i > 0) cmdline.push_back(L' ');
        cmdline += quote_win_arg(argv[i]);
    }

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE read_pipe = nullptr;
    HANDLE write_pipe = nullptr;
    if (!CreatePipe(&read_pipe, &write_pipe, &sa, 0)) {
        error_out = "CreatePipe failed (Win32 error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = write_pipe;
    si.hStdError = write_pipe;
    si.hStdInput = nullptr;

    PROCESS_INFORMATION pi{};
    std::vector<wchar_t> cmd_mut(cmdline.begin(), cmdline.end());
    cmd_mut.push_back(L'\0');

    const BOOL created = CreateProcessW(
        argv[0].c_str(),
        cmd_mut.data(),
        nullptr,
        nullptr,
        TRUE,
        CREATE_NO_WINDOW,
        nullptr,
        working_dir.empty() ? nullptr : working_dir.c_str(),
        &si,
        &pi);

    CloseHandle(write_pipe);
    if (!created) {
        const DWORD win_err = GetLastError();
        CloseHandle(read_pipe);
        error_out = "CreateProcessW failed for " + util::path_to_utf8(fs::path(argv[0])) +
                    " (Win32 error " + std::to_string(win_err) + ")";
        return false;
    }

    constexpr size_t kMaxCapturedOutput = 64ULL * 1024ULL;
    char buf[4096];
    DWORD bytes_read = 0;
    while (ReadFile(read_pipe, buf, sizeof(buf), &bytes_read, nullptr) && bytes_read > 0) {
        if (output_out.size() < kMaxCapturedOutput) {
            const size_t keep = std::min<size_t>(bytes_read, kMaxCapturedOutput - output_out.size());
            output_out.append(buf, keep);
        }
    }

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD raw_exit = 1;
    GetExitCodeProcess(pi.hProcess, &raw_exit);
    exit_code_out = static_cast<int>(raw_exit);

    CloseHandle(read_pipe);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return true;
}

bool move_stage_dir_contents(const fs::path& src_root, const fs::path& dst_root, std::string& error_out) {
    std::error_code ec;
    if (!fs::is_directory(src_root, ec) || ec) {
        return true;
    }
    for (const auto& entry : fs::recursive_directory_iterator(src_root, ec)) {
        if (ec) {
            error_out = "Failed iterating staging directory: " + ec.message();
            return false;
        }
        std::error_code f_ec;
        if (!entry.is_regular_file(f_ec) || f_ec) {
            continue;
        }
        const fs::path rel = fs::relative(entry.path(), src_root, f_ec);
        if (f_ec || rel.empty()) {
            continue;
        }
        const fs::path dst = dst_root / rel;
        if (dst.has_parent_path()) {
            fs::create_directories(dst.parent_path(), f_ec);
        }
        if (!MoveFileExW(
                entry.path().c_str(),
                dst.c_str(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED | MOVEFILE_WRITE_THROUGH)) {
            const DWORD win_err = GetLastError();
            error_out = "Failed moving staged file " + util::path_to_utf8(rel) +
                        " into models directory (Win32 error " + std::to_string(win_err) + ")";
            return false;
        }
    }
    return true;
}

json load_rfdetr_default_thresholds_json(const std::optional<fs::path>& cfg_path) {
    json thresholds = {
        {"text", 0.25},
        {"onomatopoeia", 0.20},
        {"bubble", 0.50},
        {"panel", 0.50},
    };
    if (!cfg_path.has_value()) {
        return thresholds;
    }
    const auto bytes = util::read_file_bytes(*cfg_path);
    if (!bytes.has_value()) {
        return thresholds;
    }
    const json parsed = json::parse(bytes->begin(), bytes->end(), nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object() ||
        !parsed.contains("recommended_thresholds") || !parsed["recommended_thresholds"].is_object()) {
        return thresholds;
    }
    const auto& rt = parsed["recommended_thresholds"];
    for (const char* k : {"text", "onomatopoeia", "bubble", "panel"}) {
        if (rt.contains(k) && rt[k].is_number()) {
            thresholds[k] = round_2(rt[k].get<double>());
        }
    }
    return thresholds;
}

bool snapshot_has_safetensors(const fs::path& snap_dir) {
    return find_first_with_ext(snap_dir, ".safetensors", 1024ULL).has_value();
}

}  // namespace

// -----------------------------------------------------------------------------
// Path & Cache Root Discovery
// -----------------------------------------------------------------------------

fs::path ModelManager::default_models_dir() {
    if (auto env_dir = util::get_env_wide(L"MANGA_MODELS_DIR"); env_dir.has_value() && !env_dir->empty()) {
        std::error_code ec;
        fs::path p = fs::weakly_canonical(fs::path(*env_dir), ec);
        return ec ? fs::path(*env_dir) : p;
    }
    fs::path persistent = util::get_comic_translator_app_dir() / L"models";
    std::error_code ec;
    fs::path canon = fs::weakly_canonical(persistent, ec);
    return ec ? persistent : canon;
}

std::optional<fs::path> ModelManager::find_workspace_models_dir(const fs::path& exe_dir) {
    fs::path cursor = exe_dir;
    for (int depth = 0; depth < 6; ++depth) {
        fs::path candidate = cursor / L"models";
        std::error_code ec;
        if (fs::is_directory(candidate, ec) && !ec) {
            fs::path canon = fs::weakly_canonical(candidate, ec);
            return ec ? candidate : canon;
        }
        fs::path parent = cursor.parent_path();
        if (parent.empty() || parent == cursor) {
            break;
        }
        cursor = parent;
    }

    std::error_code cwd_ec;
    fs::path cwd_models = fs::current_path(cwd_ec) / L"models";
    if (!cwd_ec && fs::is_directory(cwd_models, cwd_ec) && !cwd_ec) {
        fs::path canon = fs::weakly_canonical(cwd_models, cwd_ec);
        return cwd_ec ? cwd_models : canon;
    }
    return std::nullopt;
}

// -----------------------------------------------------------------------------
// ONNX Runtime Environment & Session Factory
// -----------------------------------------------------------------------------

Ort::Env& ModelManager::shared_ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Vellum");
    return env;
}

std::string ModelManager::get_ort_version() {
    try {
        return Ort::GetVersionString();
    } catch (...) {
        return "unknown";
    }
}

std::vector<std::string> ModelManager::get_available_providers() {
    return get_safe_available_providers();
}

std::vector<std::string> ModelManager::get_preferred_providers(bool force_cpu) {
    return get_ordered_execution_providers(force_cpu);
}

std::string ModelManager::get_primary_provider(bool force_cpu) {
    return get_primary_execution_provider(force_cpu);
}

SystemHardwareProfile ModelManager::query_hardware_profile() {
    const auto caps = detect_hardware_capabilities(/*force_cpu=*/false);
    SystemHardwareProfile prof;
    prof.total_ram_mb = caps.total_ram_mb;
    prof.available_ram_mb = caps.available_ram_mb;
    prof.memory_load_percent = caps.memory_load_percent;
    prof.dedicated_vram_mb = caps.vram_mb;
    prof.logical_cpus = caps.logical_cpus;
    prof.gpu_name = caps.gpu_name;
    prof.low_memory_mode = caps.low_memory_mode;
    return prof;
}

bool ModelManager::is_under_memory_pressure() {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) {
        return false;
    }
    const uint64_t avail_mb = static_cast<uint64_t>(ms.ullAvailPhys) / (1024ULL * 1024ULL);
    return (ms.dwMemoryLoad >= 86) || (avail_mb < 1536ULL);
}

void ModelManager::trim_process_heap() {
    _heapmin();
}

OrtSessionHandle ModelManager::create_ort_session(
    const fs::path& model_path,
    bool force_cpu,
    int cpu_intra_threads_override) {
    std::error_code ec;
    if (!fs::is_regular_file(model_path, ec) || ec) {
        throw std::runtime_error("ONNX model file not found: " + util::path_to_utf8(model_path));
    }

    Ort::Env& env = shared_ort_env();
    const auto preferred = get_preferred_providers(force_cpu);

    // Cap CPU intra-op threads to physical CPU core count to avoid HT contention while keeping UI/LM Studio responsive
    auto query_physical_cores = []() -> int {
        DWORD len = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
        if (len > 0) {
            std::vector<uint8_t> buf(len);
            if (GetLogicalProcessorInformationEx(
                    RelationProcessorCore,
                    reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
                    &len)) {
                int cores = 0;
                DWORD offset = 0;
                while (offset < len) {
                    const auto* info =
                        reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + offset);
                    if (info->Relationship == RelationProcessorCore) {
                        ++cores;
                    }
                    offset += info->Size;
                }
                if (cores > 0) {
                    return cores;
                }
            }
        }
        const unsigned int hw = std::thread::hardware_concurrency();
        return static_cast<int>(std::max(1u, hw > 1 ? (hw / 2u) : 1u));
    };
    static const int kPhysicalCores = query_physical_cores();
    const int cpu_intra_threads = (cpu_intra_threads_override > 0)
        ? std::max(1, cpu_intra_threads_override)
        : std::max(1, kPhysicalCores);
    const int gpu_host_threads = std::max(1, std::min(2, kPhysicalCores / 2));

    auto apply_common_resource_opts = [](Ort::SessionOptions& opts, bool is_cpu_only) {
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        opts.SetInterOpNumThreads(1);
        // Enable thread spinning on CPU-only sessions for faster transformer/decoder throughput,
        // disable spinning on GPU sessions so CPU doesn't busy-wait for GPU kernels.
        opts.AddConfigEntry("session.intra_op.allow_spinning", is_cpu_only ? "1" : "0");
        opts.AddConfigEntry("session.inter_op.allow_spinning", "0");
        opts.AddConfigEntry("session.set_denormal_as_zero", "1");
        enable_shared_env_allocators(opts);
        if (is_cpu_only) {
            opts.AddConfigEntry("session.memory.enable_memory_arena_shrinkage", "cpu:0");
        }
    };

    auto build_cpu_session = [&]() -> OrtSessionHandle {
        Ort::SessionOptions opts;
        apply_common_resource_opts(opts, /*is_cpu_only=*/true);
        opts.SetIntraOpNumThreads(cpu_intra_threads);
        OrtSessionHandle handle;
        handle.session = std::make_unique<Ort::Session>(env, model_path.c_str(), opts);
        handle.active_provider = "CPUExecutionProvider";
        return handle;
    };

    for (const auto& provider : preferred) {
        if (provider == "CPUExecutionProvider") {
            return build_cpu_session();
        }

        try {
            Ort::SessionOptions opts;
            apply_common_resource_opts(opts, /*is_cpu_only=*/false);
            opts.SetIntraOpNumThreads(gpu_host_threads);

            if (!append_gpu_execution_provider(opts, provider, /*device_id=*/0)) {
                continue;
            }

            OrtSessionHandle handle;
            handle.session = std::make_unique<Ort::Session>(env, model_path.c_str(), opts);
            handle.active_provider = provider;
            return handle;
        } catch (const std::exception& ex) {
            std::fprintf(stderr, "[Vellum][ORT] Provider %s failed: %s\n", provider.c_str(), ex.what());
            // Fall through to next provider (ultimately CPUExecutionProvider)
        } catch (...) {
            // Fall through to next provider (ultimately CPUExecutionProvider)
        }
    }

    return build_cpu_session();
}

// -----------------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------------

ModelManager::ModelManager(
    std::optional<fs::path> models_dir,
    std::optional<fs::path> fallback_workspace_models_dir) {
    models_dir_ = models_dir.has_value() ? *models_dir : default_models_dir();

    std::error_code ec;
    fs::create_directories(models_dir_, ec);
    if (auto canon = fs::weakly_canonical(models_dir_, ec); !ec) {
        models_dir_ = canon;
    }

    if (fallback_workspace_models_dir.has_value()) {
        std::error_code f_ec;
        if (fs::is_directory(*fallback_workspace_models_dir, f_ec) && !f_ec) {
            fs::path f_canon = fs::weakly_canonical(*fallback_workspace_models_dir, f_ec);
            fs::path resolved = f_ec ? *fallback_workspace_models_dir : f_canon;
            if (util::to_lower_ascii(util::path_to_utf8(resolved)) !=
                util::to_lower_ascii(util::path_to_utf8(models_dir_))) {
                fallback_models_dir_ = resolved;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Dynamic Pipeline Model Catalog (`GET /api/pipeline/models`)
// -----------------------------------------------------------------------------

json ModelManager::get_pipeline_models_json(const json& active_settings) const {

    // -------------------------------------------------------------------------
    // 1. Detection Models
    // -------------------------------------------------------------------------
    json detection_models = json::array();

    // 1b. comic-text-detector
    {
        bool installed = false;
        std::string source = "missing";
        std::uintmax_t bytes = 0;
        json path_val = nullptr;
        json files = json::array();

        if (auto found = find_valid_file(kCtdFile, kMinCtdBytes)) {
            installed = true;
            source = "local";
            is_valid_regular_file(*found, 1ULL, &bytes);
            path_val = util::path_to_utf8(*found);
            files.push_back(util::path_to_utf8(found->filename()));
        }

        detection_models.push_back({
            {"id", "comic-text-detector"},
            {"name", "Comic Text Detector (Legacy)"},
            {"description", "Fast 1024px manga/manhwa/manhua text box & segmentation mask detector"},
            {"installed", installed},
            {"source", source},
            {"ready_for_inference", installed},
            {"needs_conversion", false},
            {"size_mb", round_2(static_cast<double>(bytes) / (1024.0 * 1024.0))},
            {"path", std::move(path_val)},
            {"files", std::move(files)},
            {"supports_thresholds", json::array({"text_threshold", "conf_threshold", "nms_threshold", "mask_threshold"})},
            {"default_thresholds", {{"text", 0.35}, {"bubble", nullptr}, {"panel", nullptr}}},
        });
    }

    // 1c & 1d. Vellum Detector Engine v2 (YOLO11m-Seg FP16 / INT8)
    for (const auto& [id, name, desc, filename] : {
             std::tuple<const char*, const char*, const char*, const char*>{
                 "vellum-det-yolo11m-seg-1024-fp16",
                 "YOLO11m-Seg · High-Speed GPU",
                 "9-class 1024px layout, bubble, panel & SFX segmentation (GPU FP16)",
                 "vellum-det-yolo11m-seg-1024-fp16.onnx"},
             std::tuple<const char*, const char*, const char*, const char*>{
                 "vellum-det-yolo11m-seg-1024-int8",
                 "YOLO11m-Seg · Lightweight CPU",
                 "9-class 1024px layout, bubble, panel & SFX segmentation (CPU INT8)",
                 "vellum-det-yolo11m-seg-1024-int8.onnx"}}) {
        bool installed = false;
        std::string source = "missing";
        std::uintmax_t bytes = 0;
        json path_val = nullptr;
        json files = json::array();

        if (auto found = find_valid_file(filename, 1024ULL * 1024ULL)) {
            installed = true;
            source = "local";
            is_valid_regular_file(*found, 1ULL, &bytes);
            path_val = util::path_to_utf8(*found);
            files.push_back(util::path_to_utf8(found->filename()));
        }

        detection_models.push_back({
            {"id", id},
            {"name", name},
            {"description", desc},
            {"installed", installed},
            {"source", source},
            {"ready_for_inference", installed},
            {"needs_conversion", false},
            {"size_mb", round_2(static_cast<double>(bytes) / (1024.0 * 1024.0))},
            {"path", std::move(path_val)},
            {"files", std::move(files)},
            {"supports_thresholds", json::array({"conf_threshold", "nms_threshold"})},
            {"default_thresholds", {{"text", 0.42}, {"bubble", 0.42}, {"panel", 0.45}}},
        });
    }

    // -------------------------------------------------------------------------
    // 2. OCR Models
    // -------------------------------------------------------------------------
    json ocr_models = json::array();

    // 2a. manga-ocr status (computed first so "auto" can reference it)
    bool manga_ocr_installed = false;
    bool manga_ocr_needs_conv = false;
    std::string manga_ocr_source = "missing";
    std::uintmax_t manga_ocr_bytes = 0;
    json manga_ocr_path = nullptr;
    json manga_ocr_files = json::array();
    {
        auto enc = find_valid_file(kMangaOcrEncoderFile, kMinMangaOcrEncBytes);
        auto dec = find_valid_file(kMangaOcrDecoderFile, kMinMangaOcrDecBytes);
        auto vocab = find_valid_file(kMangaOcrVocabFile, kMinVocabBytes);
        if (enc.has_value() && dec.has_value() && vocab.has_value()) {
            manga_ocr_installed = true;
            manga_ocr_source = "local";
            std::uintmax_t sz = 0;
            if (is_valid_regular_file(*enc, 1ULL, &sz)) manga_ocr_bytes += sz;
            if (is_valid_regular_file(*dec, 1ULL, &sz)) manga_ocr_bytes += sz;
            if (is_valid_regular_file(*vocab, 1ULL, &sz)) manga_ocr_bytes += sz;
            manga_ocr_path = util::path_to_utf8(enc->parent_path());
            manga_ocr_files = {kMangaOcrEncoderFile, kMangaOcrDecoderFile, kMangaOcrVocabFile};
        }
    }

    // 2b. rapidocr-ppocrv4 status
    bool rapidocr_installed = false;
    std::string rapidocr_source = "missing";
    std::uintmax_t rapidocr_bytes = 0;
    json rapidocr_path = nullptr;
    json rapidocr_files = json::array();
    {
        bool all_present = true;
        bool any_present = false;
        fs::path first_dir;
        for (const auto& spec : standard_download_specs()) {
            if (std::string(spec.model_key) != "rapid_ocr_ko_zh") continue;
            if (auto found = find_valid_file(spec.local_rel_path, spec.min_valid_bytes)) {
                any_present = true;
                if (first_dir.empty()) first_dir = found->parent_path();
                std::uintmax_t sz = 0;
                if (is_valid_regular_file(*found, 1ULL, &sz)) rapidocr_bytes += sz;
                rapidocr_files.push_back(spec.local_rel_path);
            } else {
                all_present = false;
            }
        }
        if (all_present && any_present) {
            rapidocr_installed = true;
            rapidocr_source = "local";
            rapidocr_path = util::path_to_utf8(first_dir);
        } else if (resolve_rapidocr_paths("zh").has_value() || resolve_rapidocr_paths("ko").has_value()) {
            rapidocr_installed = true;
            rapidocr_source = "local";
            rapidocr_path = util::path_to_utf8(first_dir);
        }
    }

    // 2c. paddleocr-vl-gguf status
    bool paddle_vl_installed = false;
    std::string paddle_vl_source = "missing";
    std::uintmax_t paddle_vl_bytes = 0;
    json paddle_vl_path = nullptr;
    json paddle_vl_files = json::array();
    {
        std::optional<fs::path> local_model;
        std::optional<fs::path> local_mmproj;
        for (const auto& base_dir : {std::optional<fs::path>(models_dir_), fallback_models_dir_}) {
            if (!base_dir.has_value()) continue;
            const fs::path local_dir = *base_dir / L"paddleocr-vl-gguf";
            local_model = find_first_with_ext(local_dir, ".gguf", kMinGgufBytes, "", "mmproj");
            local_mmproj = find_first_with_ext(local_dir, ".gguf", kMinGgufBytes, "mmproj");
            if (local_model.has_value()) break;
        }
        if (local_model.has_value()) {
            paddle_vl_installed = true;
            paddle_vl_source = "local";
            std::uintmax_t sz = 0;
            if (is_valid_regular_file(*local_model, 1ULL, &sz)) paddle_vl_bytes += sz;
            paddle_vl_files.push_back(util::path_to_utf8(local_model->filename()));
            if (local_mmproj.has_value() && is_valid_regular_file(*local_mmproj, 1ULL, &sz)) {
                paddle_vl_bytes += sz;
                paddle_vl_files.push_back(util::path_to_utf8(local_mmproj->filename()));
            }
            paddle_vl_path = util::path_to_utf8(*local_model);
        }
    }

    // 2d. baberu-ocr status (./models/baberu-ocr/)
    bool baberu_installed = false;
    std::string baberu_source = "missing";
    std::uintmax_t baberu_bytes = 0;
    json baberu_path = nullptr;
    json baberu_files = json::array();
    if (auto baberu = resolve_baberu_ocr_models(); baberu.has_value()) {
        baberu_installed = true;
        baberu_source = baberu->source;
        std::uintmax_t sz = 0;
        for (const auto& p : {baberu->vision_onnx, baberu->decoder_prefill_onnx, baberu->decoder_step_onnx, baberu->vocab_json}) {
            if (is_valid_regular_file(p, 1ULL, &sz)) baberu_bytes += sz;
            baberu_files.push_back(util::path_to_utf8(p.filename()));
        }
        baberu_path = util::path_to_utf8(baberu->vision_onnx.parent_path());
    }

    // 2e. hayai-ocr status (./models/hayai-ocr/)
    bool hayai_installed = false;
    std::string hayai_source = "missing";
    std::uintmax_t hayai_bytes = 0;
    json hayai_path = nullptr;
    json hayai_files = json::array();
    if (auto hayai = resolve_hayai_ocr_models(); hayai.has_value()) {
        hayai_installed = true;
        hayai_source = hayai->source;
        std::uintmax_t sz = 0;
        for (const auto& p : {hayai->encoder_onnx, hayai->decoder_onnx, hayai->vocab_json}) {
            if (is_valid_regular_file(p, 1ULL, &sz)) hayai_bytes += sz;
            hayai_files.push_back(util::path_to_utf8(p.filename()));
        }
        hayai_path = util::path_to_utf8(hayai->encoder_onnx.parent_path());
    }

    // Push OCR entries in canonical option order: auto, paddleocr-vl-1.6, manga-ocr, baberu-ocr, hayai-ocr, rapidocr-ppocrv4
    {
        const bool auto_installed = manga_ocr_installed || baberu_installed || hayai_installed || rapidocr_installed;
        std::string auto_source = "missing";
        if (auto_installed) {
            auto_source = "local";
        } else if (manga_ocr_source == "koharu-cache") {
            auto_source = "koharu-cache";
        }

        ocr_models.push_back({
            {"id", "auto"},
            {"name", "Auto (Language Router)"},
            {"description", "Automatically routes Japanese → manga-ocr and Korean/Chinese → RapidOCR / Hayai OCR"},
            {"installed", auto_installed},
            {"source", auto_source},
            {"ready_for_inference", auto_installed},
            {"needs_conversion", !manga_ocr_installed && manga_ocr_needs_conv},
            {"size_mb", round_2(static_cast<double>(manga_ocr_bytes + rapidocr_bytes) / (1024.0 * 1024.0))},
            {"path", nullptr},
            {"files", json::array()},
            {"hardware_target", "CPU (Fast AVX2)"},
            {"routing", {{"ja", "manga-ocr"}, {"ko", "rapidocr-ppocrv4"}, {"zh", "rapidocr-ppocrv4"}}},
        });
    }

    ocr_models.push_back({
        {"id", "paddleocr-vl-1.6"},
        {"name", "PaddleOCR-VL 1.6 (VLM • Multi-Lang)"},
        {"description", "Multilingual Vision-Language OCR via llama-server / LM Studio or llama-mtmd-cli worker"},
        {"installed", paddle_vl_installed},
        {"source", paddle_vl_source},
        {"ready_for_inference", paddle_vl_installed},
        {"needs_conversion", false},
        {"size_mb", round_2(static_cast<double>(paddle_vl_bytes) / (1024.0 * 1024.0))},
        {"path", std::move(paddle_vl_path)},
        {"files", std::move(paddle_vl_files)},
        {"runtime", "lmstudio-vision"},
        {"hardware_target", "VLM • Multi-Lang"},
        {"lm_studio_hint", "Load PaddleOCR-VL-1.6-GGUF in llama-server / LM Studio (http://localhost:1234/v1) for Vision-OCR mode"},
        {"languages", json::array({"ja", "ko", "zh", "en"})},
    });

    ocr_models.push_back({
        {"id", "manga-ocr"},
        {"name", "Manga OCR (CPU • Japanese)"},
        {"description", "ViT + BERT autoregressive Japanese manga OCR (vertical tategaki & furigana)"},
        {"installed", manga_ocr_installed},
        {"source", manga_ocr_source},
        {"ready_for_inference", manga_ocr_installed},
        {"needs_conversion", manga_ocr_needs_conv},
        {"size_mb", round_2(static_cast<double>(manga_ocr_bytes) / (1024.0 * 1024.0))},
        {"path", std::move(manga_ocr_path)},
        {"files", std::move(manga_ocr_files)},
        {"hardware_target", "CPU • Japanese"},
        {"languages", json::array({"ja"})},
    });

    ocr_models.push_back({
        {"id", "baberu-ocr"},
        {"name", "Baberu OCR (CPU • JA/ZH/EN Fast)"},
        {"description", "Compact DINOv2 + INT8 KV-cache autoregressive OCR for Japanese, Chinese & English"},
        {"installed", baberu_installed},
        {"source", baberu_source},
        {"ready_for_inference", baberu_installed},
        {"needs_conversion", false},
        {"size_mb", round_2(static_cast<double>(baberu_bytes) / (1024.0 * 1024.0))},
        {"path", std::move(baberu_path)},
        {"files", std::move(baberu_files)},
        {"hardware_target", "CPU • JA/ZH/EN Fast"},
        {"languages", json::array({"ja", "zh", "en"})},
    });

    ocr_models.push_back({
        {"id", "hayai-ocr"},
        {"name", "Hayai OCR (GPU/CPU • JA/ZH/KO/EN)"},
        {"description", "High-throughput 150M SigLIP2 + 2D-mRoPE causal transformer OCR across JA, ZH, KO, EN"},
        {"installed", hayai_installed},
        {"source", hayai_source},
        {"ready_for_inference", hayai_installed},
        {"needs_conversion", false},
        {"size_mb", round_2(static_cast<double>(hayai_bytes) / (1024.0 * 1024.0))},
        {"path", std::move(hayai_path)},
        {"files", std::move(hayai_files)},
        {"hardware_target", "GPU/CPU • JA/ZH/KO/EN"},
        {"languages", json::array({"ja", "zh", "ko", "en"})},
    });

    ocr_models.push_back({
        {"id", "rapidocr-ppocrv4"},
        {"name", "RapidOCR PP-OCRv4 (CPU • KO/ZH)"},
        {"description", "Fast CJK/Latin PP-OCRv4 ONNX recognition (Korean, Chinese & horizontal Japanese)"},
        {"installed", rapidocr_installed},
        {"source", rapidocr_source},
        {"ready_for_inference", rapidocr_installed},
        {"needs_conversion", false},
        {"size_mb", round_2(static_cast<double>(rapidocr_bytes) / (1024.0 * 1024.0))},
        {"path", std::move(rapidocr_path)},
        {"files", std::move(rapidocr_files)},
        {"hardware_target", "CPU • KO/ZH"},
        {"languages", json::array({"ko", "zh", "ja"})},
    });

    // -------------------------------------------------------------------------
    // 3. Inpainting Models
    // -------------------------------------------------------------------------
    json inpainting_models = json::array();

    // 3a. lama-manga
    {
        bool installed = false;
        bool ready = false;
        std::string source = "missing";
        std::uintmax_t bytes = 0;
        json path_val = nullptr;
        json files = json::array();

        if (auto found = find_valid_file(kLamaMangaFile, kMinLamaBytes)) {
            installed = true;
            ready = true;
            source = "local";
            is_valid_regular_file(*found, 1ULL, &bytes);
            path_val = util::path_to_utf8(*found);
            files.push_back(util::path_to_utf8(found->filename()));
        }

        inpainting_models.push_back({
            {"id", "lama-manga"},
            {"name", "LaMa · Manga & Screentone"},
            {"description", "Manga-tuned Big-LaMa ONNX inpainter optimized for screentones & line art"},
            {"installed", installed},
            {"source", source},
            {"ready_for_inference", ready},
            {"needs_conversion", false},
            {"size_mb", round_2(static_cast<double>(bytes) / (1024.0 * 1024.0))},
            {"path", std::move(path_val)},
            {"files", std::move(files)},
            {"hardware_target", "GPU • Screentone"},
        });
    }

    // 3b. aot-inpainting (./models/aot-inpainting.onnx)
    {
        bool installed = false;
        std::string source = "missing";
        std::uintmax_t bytes = 0;
        json path_val = nullptr;
        json files = json::array();

        if (auto aot = resolve_aot_model(); aot.has_value()) {
            installed = true;
            source = aot->source;
            is_valid_regular_file(aot->model_path, 1ULL, &bytes);
            path_val = util::path_to_utf8(aot->model_path);
            files.push_back(util::path_to_utf8(aot->model_path.filename()));
        }

        inpainting_models.push_back({
            {"id", "aot-inpainting"},
            {"name", "AOT Inpainting (GPU • Ultra-Fast)"},
            {"description", "Lightweight 10-block AOT-GAN manga text-stroke inpainter (4-ch mask+RGB → 3-ch RGB, 512×512)"},
            {"installed", installed},
            {"source", source},
            {"ready_for_inference", installed},
            {"needs_conversion", false},
            {"size_mb", round_2(static_cast<double>(bytes) / (1024.0 * 1024.0))},
            {"path", std::move(path_val)},
            {"files", std::move(files)},
            {"hardware_target", "GPU • Ultra-Fast"},
        });
    }

    // 3c. flux-2-klein (./models/flux-2-klein/ + local diffusion HTTP/CLI adapter)
    {
        bool installed = false;
        std::string source = "diffusion-api";
        std::uintmax_t bytes = 0;
        json path_val = nullptr;
        json files = json::array();
        for (const auto& base_dir : {std::optional<fs::path>(models_dir_), fallback_models_dir_}) {
            if (!base_dir.has_value()) continue;
            const fs::path fdir = *base_dir / L"flux-2-klein";
            if (auto f = find_first_with_ext(fdir, ".gguf", 1024ULL)) {
                installed = true;
                source = "local";
                is_valid_regular_file(*f, 1ULL, &bytes);
                path_val = util::path_to_utf8(*f);
                files.push_back(util::path_to_utf8(f->filename()));
                break;
            }
            if (auto f = find_first_with_ext(fdir, ".safetensors", 1024ULL)) {
                installed = true;
                source = "local";
                is_valid_regular_file(*f, 1ULL, &bytes);
                path_val = util::path_to_utf8(*f);
                files.push_back(util::path_to_utf8(f->filename()));
                break;
            }
        }
        inpainting_models.push_back({
            {"id", "flux-2-klein"},
            {"name", "FLUX.2 Klein (Generative • 8GB+ VRAM)"},
            {"description", "Generative diffusion text-stroke ROI inpainter via local ComfyUI/A1111/sd-server adapter (with LaMa fallback)"},
            {"installed", installed},
            {"source", source},
            {"ready_for_inference", true},
            {"needs_conversion", false},
            {"size_mb", round_2(static_cast<double>(bytes) / (1024.0 * 1024.0))},
            {"path", std::move(path_val)},
            {"files", std::move(files)},
            {"hardware_target", "Generative • 8GB+ VRAM"},
        });
    }

    // 3d. rorem-mixed (./models/RORem-mixed-GGUF/ + local diffusion HTTP/CLI adapter)
    {
        bool installed = false;
        std::string source = "diffusion-api";
        std::uintmax_t bytes = 0;
        json path_val = nullptr;
        json files = json::array();
        for (const auto& base_dir : {std::optional<fs::path>(models_dir_), fallback_models_dir_}) {
            if (!base_dir.has_value()) continue;
            const fs::path rdir = *base_dir / L"RORem-mixed-GGUF";
            if (auto f = find_first_with_ext(rdir, ".gguf", 1024ULL)) {
                installed = true;
                source = "local";
                is_valid_regular_file(*f, 1ULL, &bytes);
                path_val = util::path_to_utf8(*f);
                files.push_back(util::path_to_utf8(f->filename()));
                break;
            }
        }
        inpainting_models.push_back({
            {"id", "rorem-mixed"},
            {"name", "RORem Mixed (Generative • Manga GGUF)"},
            {"description", "Manga object/text removal diffusion adapter via local diffusion worker (with LaMa/AOT fallback)"},
            {"installed", installed},
            {"source", source},
            {"ready_for_inference", true},
            {"needs_conversion", false},
            {"size_mb", round_2(static_cast<double>(bytes) / (1024.0 * 1024.0))},
            {"path", std::move(path_val)},
            {"files", std::move(files)},
            {"hardware_target", "Generative • Manga GGUF"},
        });
    }

    // 3e. lama-fp32
    {
        bool installed = false;
        std::string source = "missing";
        std::uintmax_t bytes = 0;
        json path_val = nullptr;
        json files = json::array();

        if (auto found = find_valid_file(kLamaFallbackFile, kMinLamaBytes)) {
            installed = true;
            source = "local";
            is_valid_regular_file(*found, 1ULL, &bytes);
            path_val = util::path_to_utf8(*found);
            files.push_back(util::path_to_utf8(found->filename()));
        }

        inpainting_models.push_back({
            {"id", "lama-fp32"},
            {"name", "LaMa · High-Precision Universal"},
            {"description", "General-purpose Big-LaMa FP32 ONNX inpainter (512×512 tiles)"},
            {"installed", installed},
            {"source", source},
            {"ready_for_inference", installed},
            {"needs_conversion", false},
            {"size_mb", round_2(static_cast<double>(bytes) / (1024.0 * 1024.0))},
            {"path", std::move(path_val)},
            {"files", std::move(files)},
            {"hardware_target", "GPU • FP32"},
        });
    }

    const json active = {
        {"detection_model", active_settings.value("detection_model", std::string("vellum-det-yolo11m-seg-1024-fp16"))},
        {"ocr_model", active_settings.value("ocr_model", std::string("auto"))},
        {"inpainting_model", active_settings.value("inpainting_model", std::string("lama-manga"))},
        {"inpaint_api_url", active_settings.value("inpaint_api_url", std::string("http://127.0.0.1:8188"))},
        {"text_threshold", active_settings.contains("text_threshold") ? active_settings["text_threshold"] : json(nullptr)},
        {"bubble_threshold", active_settings.contains("bubble_threshold") ? active_settings["bubble_threshold"] : json(nullptr)},
        {"panel_threshold", active_settings.contains("panel_threshold") ? active_settings["panel_threshold"] : json(nullptr)},
    };

    return {
        {"models_dir", util::path_to_utf8(models_dir_)},
        {"execution_provider", get_primary_provider()},
        {"available_providers", get_available_providers()},
        {"active", active},
        {"detection", std::move(detection_models)},
        {"ocr", std::move(ocr_models)},
        {"inpainting", std::move(inpainting_models)},
    };
}

// -----------------------------------------------------------------------------
// Model Resolution Helpers
// -----------------------------------------------------------------------------

std::optional<fs::path> ModelManager::find_valid_file(
    const std::string& rel_path,
    std::uintmax_t min_valid_bytes,
    std::string* source_out) const {
    const fs::path rel_w = util::utf8_to_wide(rel_path);
    const fs::path primary = models_dir_ / rel_w;
    if (is_valid_regular_file(primary, min_valid_bytes)) {
        if (source_out) *source_out = "persistent_models_dir";
        return primary;
    }
    if (fallback_models_dir_.has_value()) {
        const fs::path secondary = *fallback_models_dir_ / rel_w;
        if (is_valid_regular_file(secondary, min_valid_bytes)) {
            if (source_out) *source_out = "workspace_fallback";
            return secondary;
        }
    }
    return std::nullopt;
}

std::optional<ResolvedDetectorModel> ModelManager::resolve_detector_model(
    const std::string& preferred_backend,
    bool allow_fallback) const {
    std::string pref = util::to_lower_ascii(util::trim_ascii(preferred_backend));
    if (pref.empty() || pref == "auto") {
        if (auto env_pref = util::get_env_utf8(L"MANGA_DETECTOR_BACKEND"); env_pref.has_value()) {
            const std::string cleaned = util::to_lower_ascii(util::trim_ascii(*env_pref));
            if (!cleaned.empty()) pref = cleaned;
        }
    }

    const bool is_auto = (pref.empty() || pref == "auto" || pref == "default");
    const bool wants_ctd = is_auto ||
                           pref == "ctd" ||
                           pref == "comic_text_detector" ||
                           pref == "comic-text-detector" ||
                           pref == "comictextdetector" ||
                           allow_fallback;

    if (wants_ctd) {
        std::string src;
        if (auto found = find_valid_file(kCtdFile, kMinCtdBytes, &src)) {
            ResolvedDetectorModel res;
            res.backend = DetectorBackend::ComicTextDetector;
            res.model_path = *found;
            res.source = src;
            return res;
        }
    }

    return std::nullopt;
}

std::optional<ResolvedLamaModel> ModelManager::resolve_lama_model(
    const std::string& preferred_model,
    bool allow_fallback) const {
    const std::string pref = util::to_lower_ascii(util::trim_ascii(preferred_model));
    const bool is_auto = (pref.empty() || pref == "auto" || pref == "default");
    const bool wants_manga = is_auto ||
                             pref == "lama-manga" ||
                             pref == "lama_manga" ||
                             pref == "koharu-lama-manga" ||
                             pref == "koharu_lama_manga";
    const bool wants_fp32 = is_auto ||
                            pref == "lama-fp32" ||
                            pref == "lama_fp32" ||
                            pref == "lama" ||
                            pref == "fp32";

    auto try_manga = [&]() -> std::optional<ResolvedLamaModel> {
        std::string src;
        if (auto found = find_valid_file(kLamaMangaFile, kMinLamaBytes, &src)) {
            return ResolvedLamaModel{
                *found,
                (src == "persistent_models_dir") ? "local_models_dir" : src,
                true};
        }
        return std::nullopt;
    };

    auto try_fp32 = [&]() -> std::optional<ResolvedLamaModel> {
        std::string src;
        if (auto found = find_valid_file(kLamaFallbackFile, kMinLamaBytes, &src)) {
            return ResolvedLamaModel{*found, src, false};
        }
        return std::nullopt;
    };

    if (wants_manga) {
        if (auto m = try_manga(); m.has_value()) {
            return m;
        }
        if (is_auto || allow_fallback) {
            if (auto f = try_fp32(); f.has_value()) {
                return f;
            }
        }
        return std::nullopt;
    }

    if (wants_fp32) {
        if (auto f = try_fp32(); f.has_value()) {
            return f;
        }
        if (allow_fallback) {
            if (auto m = try_manga(); m.has_value()) {
                return m;
            }
        }
        return std::nullopt;
    }

    return std::nullopt;
}

std::optional<ResolvedAotModel> ModelManager::resolve_aot_model() const {
    std::string src;
    for (const char* fname : {"aot-inpainting.onnx", "aot_inpainting.onnx"}) {
        if (auto found = find_valid_file(fname, 5'000'000ULL, &src)) {
            return ResolvedAotModel{
                *found,
                (src == "persistent_models_dir") ? "local_models_dir" : src};
        }
    }
    return std::nullopt;
}

std::optional<ResolvedBaberuOcrModel> ModelManager::resolve_baberu_ocr_models() const {
    std::string src;
    auto vis = find_valid_file("baberu-ocr/vision_int4.onnx", 10'000'000ULL, &src);
    if (!vis.has_value()) {
        vis = find_valid_file("baberu-ocr/vision_fp16.onnx", 10'000'000ULL, &src);
    }
    auto pre = find_valid_file("baberu-ocr/decoder_prefill_int8.onnx", 5'000'000ULL);
    auto stp = find_valid_file("baberu-ocr/decoder_step_int8.onnx", 5'000'000ULL);
    auto voc = find_valid_file("baberu-ocr/vocab.json", 1'000ULL);
    if (vis.has_value() && pre.has_value() && stp.has_value() && voc.has_value()) {
        ResolvedBaberuOcrModel res;
        res.vision_onnx = *vis;
        res.decoder_prefill_onnx = *pre;
        res.decoder_step_onnx = *stp;
        res.vocab_json = *voc;
        res.source = (src == "persistent_models_dir") ? "local_models_dir" : src;
        return res;
    }
    return std::nullopt;
}

std::optional<ResolvedHayaiOcrModel> ModelManager::resolve_hayai_ocr_models() const {
    std::string src;
    auto enc = find_valid_file("hayai-ocr/encoder.onnx", 10'000'000ULL, &src);
    auto dec = find_valid_file("hayai-ocr/decoder.onnx", 10'000'000ULL);
    auto voc = find_valid_file("hayai-ocr/vocab.json", 1'000ULL);
    if (!voc.has_value()) {
        voc = find_valid_file("hayai-ocr/tokenizer.json", 1'000ULL);
    }
    if (enc.has_value() && dec.has_value() && voc.has_value()) {
        ResolvedHayaiOcrModel res;
        res.encoder_onnx = *enc;
        res.decoder_onnx = *dec;
        res.vocab_json = *voc;
        res.source = (src == "persistent_models_dir") ? "local_models_dir" : src;
        return res;
    }
    return std::nullopt;
}

std::optional<ResolvedMangaOcrModel> ModelManager::resolve_manga_ocr_models() const {
    std::string src;
    auto enc = find_valid_file(kMangaOcrEncoderFile, kMinMangaOcrEncBytes, &src);
    auto dec = find_valid_file(kMangaOcrDecoderFile, kMinMangaOcrDecBytes);
    auto vocab = find_valid_file(kMangaOcrVocabFile, kMinVocabBytes);
    if (enc.has_value() && dec.has_value() && vocab.has_value()) {
        ResolvedMangaOcrModel res;
        res.encoder_onnx = *enc;
        res.decoder_onnx = *dec;
        res.vocab_txt = *vocab;
        if (auto cfg = find_valid_file(kMangaOcrConfigFile, 100ULL)) {
            res.config_json = *cfg;
        }
        res.source = (src == "persistent_models_dir") ? "local_models_dir" : src;
        return res;
    }

    return std::nullopt;
}

std::optional<ResolvedPaddleOcrVlGguf> ModelManager::resolve_paddleocr_vl_gguf() const {
    // Standalone `./models/paddleocr-vl-gguf/` (or workspace fallback)
    for (const auto& entry : {
             std::pair{std::optional<fs::path>(models_dir_), "local_models_dir"},
             std::pair{fallback_models_dir_, "workspace_fallback"},
         }) {
        if (!entry.first.has_value()) continue;
        const fs::path target_dir = *entry.first / L"paddleocr-vl-gguf";
        auto model_f = find_first_with_ext(target_dir, ".gguf", kMinGgufBytes, "", "mmproj");
        auto mmproj_f = find_first_with_ext(target_dir, ".gguf", kMinGgufBytes, "mmproj");
        if (model_f.has_value()) {
            ResolvedPaddleOcrVlGguf res;
            res.model_gguf = *model_f;
            if (mmproj_f.has_value()) res.mmproj_gguf = *mmproj_f;
            res.source = entry.second;
            return res;
        }
    }

    return std::nullopt;
}

std::optional<RapidOcrPaths> ModelManager::resolve_rapidocr_paths(const std::string& language) const {
    const std::string lang = util::to_lower_ascii(util::trim_ascii(language));
    const char* det_rel = (lang == "ko")
        ? "rapidocr/Multilingual_PP-OCRv3_det_infer.onnx"
        : "rapidocr/ch_PP-OCRv4_det_infer.onnx";
    const char* rec_rel = (lang == "ko")
        ? "rapidocr/korean_PP-OCRv4_rec_infer.onnx"
        : "rapidocr/ch_PP-OCRv4_rec_infer.onnx";
    const char* cls_rel = "rapidocr/ch_ppocr_mobile_v2.0_cls_infer.onnx";

    auto det = find_valid_file(det_rel, 1'000'000ULL);
    auto rec = find_valid_file(rec_rel, 5'000'000ULL);
    auto cls = find_valid_file(cls_rel, 100'000ULL);
    if (!det.has_value() || !rec.has_value() || !cls.has_value()) {
        return std::nullopt;
    }
    return RapidOcrPaths{*det, *rec, *cls};
}

// -----------------------------------------------------------------------------
// Offline Status Verification (`GET /api/models/status`, SPEC.md Section 5.5)
// -----------------------------------------------------------------------------

json ModelManager::get_status_json() const {
    json models = json::object();
    bool all_ready = true;

    // 1. Detector (`comic_text_detector` key preserved for DownloadModal.jsx compatibility;
    //    satisfied by either Koharu RF-DETR ONNX or Comic Text Detector ONNX)
    {
        const auto det = resolve_detector_model("auto");
        const bool installed = det.has_value();
        if (!installed) all_ready = false;

        std::uintmax_t actual_bytes = 0;
        json files = json::array();
        json missing = json::array();
        if (installed) {
            is_valid_regular_file(det->model_path, 1ULL, &actual_bytes);
            files.push_back(util::path_to_utf8(det->model_path.filename()));
        } else {
            files.push_back(kCtdFile);
            missing.push_back(kCtdFile);
        }

        models["comic_text_detector"] = {
            {"installed", installed},
            {"size_mb", 90},
            {"downloaded_mb", round_2(static_cast<double>(actual_bytes) / (1024.0 * 1024.0))},
            {"files", std::move(files)},
            {"missing_files", std::move(missing)},
            {"active_backend", installed ? json(detector_backend_name(det->backend)) : json(nullptr)},
            {"active_filename", installed ? json(util::path_to_utf8(det->model_path.filename())) : json(nullptr)},
            {"active_path", installed ? json(util::path_to_utf8(det->model_path)) : json(nullptr)},
            {"source", installed ? json(det->source) : json(nullptr)},
        };
    }

    // 1b. Vellum Detector v2 (YOLO11m-Seg FP16 / INT8 in ./models/)
    for (const auto& [key, fname] : {
             std::pair<const char*, const char*>{"vellum_det_yolo11m_fp16", "vellum-det-yolo11m-seg-1024-fp16.onnx"},
             std::pair<const char*, const char*>{"vellum_det_yolo11m_int8", "vellum-det-yolo11m-seg-1024-int8.onnx"}}) {
        std::uintmax_t sz = 0;
        std::string src;
        auto found = find_valid_file(fname, 1024ULL * 1024ULL, &src);
        const bool ok = found.has_value();
        if (ok) {
            is_valid_regular_file(*found, 1ULL, &sz);
        }
        models[key] = {
            {"installed", ok},
            {"downloaded_mb", round_2(static_cast<double>(sz) / (1024.0 * 1024.0))},
            {"files", json::array({fname})},
            {"missing_files", ok ? json::array() : json::array({fname})},
            {"active_filename", ok ? json(fname) : json(nullptr)},
            {"active_path", ok ? json(util::path_to_utf8(*found)) : json(nullptr)},
            {"source", ok ? json(src) : json(nullptr)},
        };
    }

    // 2. Japanese Manga OCR (`manga_ocr`; satisfied by Koharu ONNX export OR legacy manga-ocr-base)
    {
        const auto onnx_mocr = resolve_manga_ocr_models();
        json files = json::array();
        json missing = json::array();
        std::uintmax_t actual_bytes = 0;
        bool installed = false;
        std::string active_format = "missing";
        std::string source;

        if (onnx_mocr.has_value()) {
            installed = true;
            active_format = "onnx";
            source = onnx_mocr->source;
            std::uintmax_t sz = 0;
            if (is_valid_regular_file(onnx_mocr->encoder_onnx, 1ULL, &sz)) actual_bytes += sz;
            if (is_valid_regular_file(onnx_mocr->decoder_onnx, 1ULL, &sz)) actual_bytes += sz;
            if (is_valid_regular_file(onnx_mocr->vocab_txt, 1ULL, &sz)) actual_bytes += sz;
            files = {kMangaOcrEncoderFile, kMangaOcrDecoderFile, kMangaOcrVocabFile};
        } else {
            // Check legacy Hugging Face manga-ocr-base files
            bool legacy_ok = true;
            for (const auto& spec : standard_download_specs()) {
                if (std::string(spec.model_key) != "manga_ocr") continue;
                files.push_back(spec.local_rel_path);
                std::string fsrc;
                if (auto found = find_valid_file(spec.local_rel_path, spec.min_valid_bytes, &fsrc)) {
                    std::uintmax_t sz = 0;
                    if (is_valid_regular_file(*found, 1ULL, &sz)) actual_bytes += sz;
                    if (source.empty()) source = fsrc;
                } else {
                    legacy_ok = false;
                    missing.push_back(spec.local_rel_path);
                }
            }
            if (legacy_ok) {
                installed = true;
                active_format = "pytorch_legacy";
            }
        }

        if (!installed) all_ready = false;
        json active_mocr_files = json::array();
        if (onnx_mocr.has_value()) {
            active_mocr_files = {
                util::path_to_utf8(onnx_mocr->encoder_onnx.filename()),
                util::path_to_utf8(onnx_mocr->decoder_onnx.filename()),
                util::path_to_utf8(onnx_mocr->vocab_txt.filename()),
            };
        }
        models["manga_ocr"] = {
            {"installed", installed},
            {"size_mb", 424},
            {"downloaded_mb", round_2(static_cast<double>(actual_bytes) / (1024.0 * 1024.0))},
            {"files", files},
            {"missing_files", std::move(missing)},
            {"format", active_format},
            {"active_filename", onnx_mocr.has_value()
                                    ? json(util::path_to_utf8(onnx_mocr->encoder_onnx.filename()) + " + " +
                                           util::path_to_utf8(onnx_mocr->decoder_onnx.filename()))
                                    : (installed ? json("pytorch_model.bin") : json(nullptr))},
            {"active_files", std::move(active_mocr_files)},
            {"active_path", onnx_mocr.has_value()
                                ? json(util::path_to_utf8(onnx_mocr->encoder_onnx.parent_path()))
                                : json(nullptr)},
            {"source", installed ? json(source) : json(nullptr)},
        };
    }

    // 3. RapidOCR PP-OCRv4 (Korean & Chinese ONNX)
    {
        json files = json::array();
        json missing = json::array();
        std::uintmax_t actual_bytes = 0;
        std::string source;
        for (const auto& spec : standard_download_specs()) {
            if (std::string(spec.model_key) != "rapid_ocr_ko_zh") continue;
            files.push_back(spec.local_rel_path);
            std::string fsrc;
            if (auto found = find_valid_file(spec.local_rel_path, spec.min_valid_bytes, &fsrc)) {
                std::uintmax_t sz = 0;
                if (is_valid_regular_file(*found, 1ULL, &sz)) actual_bytes += sz;
                if (source.empty()) source = fsrc;
            } else {
                missing.push_back(spec.local_rel_path);
            }
        }
        const bool installed = missing.empty();
        if (!installed) all_ready = false;
        models["rapid_ocr_ko_zh"] = {
            {"installed", installed},
            {"size_mb", 41},
            {"downloaded_mb", round_2(static_cast<double>(actual_bytes) / (1024.0 * 1024.0))},
            {"files", std::move(files)},
            {"missing_files", std::move(missing)},
            {"source", installed ? json(source) : json(nullptr)},
        };
    }

    // 4. LaMa Inpainter (`lama_inpaint`; satisfied by Koharu `lama_manga.onnx` OR `lama_fp32.onnx`)
    {
        const auto lama = resolve_lama_model();
        const bool installed = lama.has_value();
        if (!installed) all_ready = false;

        std::uintmax_t actual_bytes = 0;
        json files = json::array();
        json missing = json::array();
        if (installed) {
            is_valid_regular_file(lama->model_path, 1ULL, &actual_bytes);
            files.push_back(util::path_to_utf8(lama->model_path.filename()));
        } else {
            files.push_back(kLamaFallbackFile);
            missing.push_back(kLamaFallbackFile);
        }

        models["lama_inpaint"] = {
            {"installed", installed},
            {"size_mb", 198},
            {"downloaded_mb", round_2(static_cast<double>(actual_bytes) / (1024.0 * 1024.0))},
            {"files", std::move(files)},
            {"missing_files", std::move(missing)},
            {"is_manga_lama", installed ? lama->is_manga_lama : false},
            {"active_filename", installed ? json(util::path_to_utf8(lama->model_path.filename())) : json(nullptr)},
            {"active_path", installed ? json(util::path_to_utf8(lama->model_path)) : json(nullptr)},
            {"source", installed ? json(lama->source) : json(nullptr)},
            {"hardware_target", "GPU (RTX/DirectML)"},
        };
    }

    // 4b. Baberu OCR (`baberu_ocr`), Hayai OCR (`hayai_ocr`), AOT Inpainting (`aot_inpainting`)
    {
        const auto baberu = resolve_baberu_ocr_models();
        std::uintmax_t b_sz = 0, tmp = 0;
        if (baberu.has_value()) {
            for (const auto& p : {baberu->vision_onnx, baberu->decoder_prefill_onnx, baberu->decoder_step_onnx, baberu->vocab_json}) {
                if (is_valid_regular_file(p, 1ULL, &tmp)) b_sz += tmp;
            }
        }
        models["baberu_ocr"] = {
            {"installed", baberu.has_value()},
            {"downloaded_mb", round_2(static_cast<double>(b_sz) / (1024.0 * 1024.0))},
            {"hardware_target", "CPU (Fast AVX2)"},
            {"source", baberu.has_value() ? json(baberu->source) : json(nullptr)},
        };

        const auto hayai = resolve_hayai_ocr_models();
        std::uintmax_t h_sz = 0;
        if (hayai.has_value()) {
            for (const auto& p : {hayai->encoder_onnx, hayai->decoder_onnx, hayai->vocab_json}) {
                if (is_valid_regular_file(p, 1ULL, &tmp)) h_sz += tmp;
            }
        }
        models["hayai_ocr"] = {
            {"installed", hayai.has_value()},
            {"downloaded_mb", round_2(static_cast<double>(h_sz) / (1024.0 * 1024.0))},
            {"hardware_target", "GPU / CPU"},
            {"source", hayai.has_value() ? json(hayai->source) : json(nullptr)},
        };

        const auto aot = resolve_aot_model();
        std::uintmax_t a_sz = 0;
        if (aot.has_value()) {
            is_valid_regular_file(aot->model_path, 1ULL, &a_sz);
        }
        models["aot_inpainting"] = {
            {"installed", aot.has_value()},
            {"downloaded_mb", round_2(static_cast<double>(a_sz) / (1024.0 * 1024.0))},
            {"hardware_target", "GPU (RTX/DirectML)"},
            {"active_filename", aot.has_value() ? json(util::path_to_utf8(aot->model_path.filename())) : json(nullptr)},
            {"source", aot.has_value() ? json(aot->source) : json(nullptr)},
        };
    }

    // Optional Vision-OCR GGUF summary for LM Studio (`localhost:1234`)
    json paddle_vl_json = {{"installed", false}, {"model_gguf", nullptr}, {"mmproj_gguf", nullptr}};
    if (auto gguf = resolve_paddleocr_vl_gguf(); gguf.has_value()) {
        std::uintmax_t sz_total = 0;
        std::uintmax_t sz_part = 0;
        if (is_valid_regular_file(gguf->model_gguf, 1ULL, &sz_part)) sz_total += sz_part;
        if (!gguf->mmproj_gguf.empty() && is_valid_regular_file(gguf->mmproj_gguf, 1ULL, &sz_part)) {
            sz_total += sz_part;
        }
        paddle_vl_json = {
            {"installed", true},
            {"source", gguf->source},
            {"model_filename", util::path_to_utf8(gguf->model_gguf.filename())},
            {"mmproj_filename", gguf->mmproj_gguf.empty() ? json(nullptr) : json(util::path_to_utf8(gguf->mmproj_gguf.filename()))},
            {"model_gguf", util::path_to_utf8(gguf->model_gguf)},
            {"mmproj_gguf", gguf->mmproj_gguf.empty() ? json(nullptr) : json(util::path_to_utf8(gguf->mmproj_gguf))},
            {"size_mb", round_2(static_cast<double>(sz_total) / (1024.0 * 1024.0))},
            {"lm_studio_hint", "Load in LM Studio (http://localhost:1234/v1) for Vision-OCR mode"},
        };
    }

    return {
        {"all_ready", all_ready},
        {"execution_provider", get_primary_provider()},
        {"available_providers", get_available_providers()},
        {"models_dir", util::path_to_utf8(models_dir_)},
        {"fallback_models_dir", fallback_models_dir_.has_value()
                                    ? json(util::path_to_utf8(*fallback_models_dir_))
                                    : json(nullptr)},
        {"models", std::move(models)},
        {"paddleocr_vl_gguf", std::move(paddle_vl_json)},
    };
}

bool ModelManager::is_all_ready() const {
    return get_status_json().value("all_ready", false);
}

// -----------------------------------------------------------------------------
// Automatic HTTPS Download Fallback (WinHTTP streaming + SSE progress events)
// -----------------------------------------------------------------------------

std::vector<ModelManager::DownloadFileSpec> ModelManager::get_missing_download_specs() const {
    const bool det_ready = resolve_detector_model("auto").has_value();
    const bool lama_ready = resolve_lama_model().has_value();
    const bool mocr_onnx_ready = resolve_manga_ocr_models().has_value();

    std::vector<DownloadFileSpec> missing;
    for (const auto& s : standard_download_specs()) {
        const std::string key = s.model_key;
        if (key == "comic_text_detector" && det_ready) continue;
        if (key == "lama_inpaint" && lama_ready) continue;
        if (key == "manga_ocr" && mocr_onnx_ready) continue;

        if (!find_valid_file(s.local_rel_path, s.min_valid_bytes).has_value()) {
            missing.push_back({
                key,
                s.repo_id,
                s.remote_filename,
                s.local_rel_path,
                s.expected_bytes,
                s.min_valid_bytes,
            });
        }
    }
    return missing;
}

bool ModelManager::download_single_file_winhttp(
    const DownloadFileSpec& spec,
    const fs::path& target_path,
    size_t file_index,
    size_t file_count,
    std::uintmax_t& cumulative_downloaded,
    std::uintmax_t& total_expected_bytes,
    const ProgressCallback& on_event,
    std::string& error_out) const {
    const std::string url = build_hf_resolve_url(spec.repo_id, spec.remote_filename);
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
        error_out = "Invalid download URL: " + url;
        return false;
    }

    WinHttpHandle session(WinHttpOpen(
        L"Vellum/2.0",
        WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
        WINHTTP_NO_PROXY_NAME,
        WINHTTP_NO_PROXY_BYPASS,
        0));
    if (!session) {
        error_out = "WinHttpOpen failed (Win32 error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    WinHttpSetTimeouts(session.h, 15000, 15000, 30000, 60000);

    const std::wstring host(uc.lpszHostName, uc.dwHostNameLength);
    WinHttpHandle connect(WinHttpConnect(session.h, host.c_str(), uc.nPort, 0));
    if (!connect) {
        error_out = "WinHttpConnect failed for " + util::wide_to_utf8(host) +
                    " (Win32 error " + std::to_string(GetLastError()) + ")";
        return false;
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
        error_out = "WinHttpOpenRequest failed (Win32 error " + std::to_string(GetLastError()) + ")";
        return false;
    }

    DWORD redirect_policy = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
    WinHttpSetOption(request.h, WINHTTP_OPTION_REDIRECT_POLICY, &redirect_policy, sizeof(redirect_policy));

    if (!WinHttpSendRequest(request.h, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
        !WinHttpReceiveResponse(request.h, nullptr)) {
        error_out = "HTTPS request to Hugging Face failed for " + spec.local_rel_path +
                    " (Win32 error " + std::to_string(GetLastError()) + ")";
        return false;
    }

    DWORD status_code = 0;
    DWORD status_sz = sizeof(status_code);
    WinHttpQueryHeaders(
        request.h,
        WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
        WINHTTP_HEADER_NAME_BY_INDEX,
        &status_code,
        &status_sz,
        WINHTTP_NO_HEADER_INDEX);
    if (status_code < 200 || status_code >= 300) {
        error_out = "Hugging Face returned HTTP " + std::to_string(status_code) +
                    " for " + spec.repo_id + "/" + spec.remote_filename;
        return false;
    }

    std::uintmax_t file_total_bytes = spec.expected_bytes;
    wchar_t cl_buf[64]{};
    DWORD cl_sz = sizeof(cl_buf);
    if (WinHttpQueryHeaders(
            request.h,
            WINHTTP_QUERY_CONTENT_LENGTH,
            WINHTTP_HEADER_NAME_BY_INDEX,
            cl_buf,
            &cl_sz,
            WINHTTP_NO_HEADER_INDEX)) {
        try {
            const std::uintmax_t parsed_len = std::stoull(util::wide_to_utf8(cl_buf));
            if (parsed_len > 0) {
                file_total_bytes = parsed_len;
                if (file_total_bytes >= spec.expected_bytes) {
                    total_expected_bytes += (file_total_bytes - spec.expected_bytes);
                } else {
                    const auto diff = spec.expected_bytes - file_total_bytes;
                    total_expected_bytes = (total_expected_bytes > diff) ? (total_expected_bytes - diff) : file_total_bytes;
                }
            }
        } catch (...) {
        }
    }

    std::error_code ec;
    if (target_path.has_parent_path()) {
        fs::create_directories(target_path.parent_path(), ec);
    }
    const fs::path temp_path = fs::path(target_path.wstring() + L".part");

    std::ofstream ofs(temp_path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        error_out = "Failed to open temporary file for writing: " + util::path_to_utf8(temp_path);
        return false;
    }

    constexpr size_t kChunkBytes = 256ULL * 1024ULL;
    std::vector<char> buffer(kChunkBytes);
    std::uintmax_t file_downloaded = 0;
    double last_reported_pct = -1.0;

    while (true) {
        DWORD avail = 0;
        if (!WinHttpQueryDataAvailable(request.h, &avail)) {
            ofs.close();
            fs::remove(temp_path, ec);
            error_out = "WinHttpQueryDataAvailable failed (Win32 error " + std::to_string(GetLastError()) + ")";
            return false;
        }
        if (avail == 0) {
            break;
        }

        const DWORD to_read = static_cast<DWORD>(std::min<size_t>(avail, buffer.size()));
        DWORD bytes_read = 0;
        if (!WinHttpReadData(request.h, buffer.data(), to_read, &bytes_read)) {
            ofs.close();
            fs::remove(temp_path, ec);
            error_out = "WinHttpReadData failed (Win32 error " + std::to_string(GetLastError()) + ")";
            return false;
        }
        if (bytes_read == 0) {
            break;
        }

        ofs.write(buffer.data(), static_cast<std::streamsize>(bytes_read));
        if (!ofs.good()) {
            ofs.close();
            fs::remove(temp_path, ec);
            error_out = "Disk write failed while saving " + spec.local_rel_path;
            return false;
        }

        file_downloaded += bytes_read;
        cumulative_downloaded += bytes_read;

        const double file_pct = (file_total_bytes > 0)
            ? std::min(100.0, (static_cast<double>(file_downloaded) / static_cast<double>(file_total_bytes)) * 100.0)
            : 100.0;
        const double overall_pct = (total_expected_bytes > 0)
            ? std::min(100.0, (static_cast<double>(cumulative_downloaded) / static_cast<double>(total_expected_bytes)) * 100.0)
            : 100.0;

        if ((file_pct - last_reported_pct >= 1.0) || file_downloaded >= file_total_bytes) {
            last_reported_pct = file_pct;
            const json evt = {
                {"status", "downloading"},
                {"model_key", spec.model_key},
                {"repo_id", spec.repo_id},
                {"file", spec.local_rel_path},
                {"file_index", file_index},
                {"file_count", file_count},
                {"file_downloaded_bytes", file_downloaded},
                {"file_total_bytes", file_total_bytes},
                {"file_percent", round_1(file_pct)},
                {"overall_percent", round_1(overall_pct)},
                {"downloaded_bytes", cumulative_downloaded},
                {"total_bytes", total_expected_bytes},
                {"offline", false},
                {"message",
                 "[" + std::to_string(file_index) + "/" + std::to_string(file_count) + "] Downloading " +
                     spec.local_rel_path + ": " + std::to_string(static_cast<int>(std::round(file_pct))) +
                     "% (" + format_mb(file_downloaded, 1) + "/" + format_mb(file_total_bytes, 1) + " MB)"},
            };
            if (!on_event(evt)) {
                ofs.close();
                fs::remove(temp_path, ec);
                error_out = "Client disconnected during model download.";
                return false;
            }
        }
    }

    ofs.flush();
    ofs.close();

    std::uintmax_t actual_size = 0;
    if (!is_valid_regular_file(temp_path, spec.min_valid_bytes, &actual_size)) {
        fs::remove(temp_path, ec);
        error_out = "Downloaded file " + spec.local_rel_path + " is incomplete (" +
                    std::to_string(file_downloaded) + " bytes < " +
                    std::to_string(spec.min_valid_bytes) + " min bytes).";
        return false;
    }

    if (!MoveFileExW(
            temp_path.c_str(),
            target_path.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const DWORD win_err = GetLastError();
        fs::remove(temp_path, ec);
        error_out = "Failed to finalize " + spec.local_rel_path +
                    " (Win32 error " + std::to_string(win_err) + ")";
        return false;
    }

    const double overall_pct = (total_expected_bytes > 0)
        ? std::min(100.0, (static_cast<double>(cumulative_downloaded) / static_cast<double>(total_expected_bytes)) * 100.0)
        : 100.0;

    return on_event({
        {"status", "file_completed"},
        {"model_key", spec.model_key},
        {"repo_id", spec.repo_id},
        {"file", spec.local_rel_path},
        {"file_index", file_index},
        {"file_count", file_count},
        {"file_percent", 100.0},
        {"overall_percent", round_1(overall_pct)},
        {"downloaded_bytes", cumulative_downloaded},
        {"total_bytes", total_expected_bytes},
        {"offline", false},
        {"message", "Saved " + spec.local_rel_path + " (" + format_mb(actual_size, 2) + " MB)"},
    });
}

void ModelManager::download_missing_models(const ProgressCallback& on_event) const {
    const auto missing = get_missing_download_specs();
    if (missing.empty()) {
        on_event({
            {"status", "skipped"},
            {"model_key", "all"},
            {"file", nullptr},
            {"file_percent", 100.0},
            {"overall_percent", 100.0},
            {"downloaded_bytes", 0},
            {"total_bytes", 0},
            {"offline", true},
            {"message", "All required models are already present locally. Running 100% offline."},
        });
        return;
    }

    std::uintmax_t total_expected_bytes = 0;
    for (const auto& s : missing) {
        total_expected_bytes += s.expected_bytes;
    }
    std::uintmax_t cumulative_downloaded = 0;

    if (!on_event({
            {"status", "starting"},
            {"model_key", "all"},
            {"file", nullptr},
            {"file_percent", 0.0},
            {"overall_percent", 0.0},
            {"downloaded_bytes", 0},
            {"total_bytes", total_expected_bytes},
            {"offline", false},
            {"message",
             "Downloading " + std::to_string(missing.size()) + " missing model file(s) (~" +
                 format_mb(total_expected_bytes, 1) + " MB)..."},
        })) {
        return;
    }

    for (size_t i = 0; i < missing.size(); ++i) {
        const auto& spec = missing[i];
        const fs::path target_path = models_dir_ / util::utf8_to_wide(spec.local_rel_path);
        std::string err;
        if (!download_single_file_winhttp(
                spec,
                target_path,
                i + 1,
                missing.size(),
                cumulative_downloaded,
                total_expected_bytes,
                on_event,
                err)) {
            on_event({
                {"status", "error"},
                {"model_key", spec.model_key},
                {"file", spec.local_rel_path},
                {"file_percent", 0.0},
                {"overall_percent", 0.0},
                {"downloaded_bytes", cumulative_downloaded},
                {"total_bytes", total_expected_bytes},
                {"offline", false},
                {"message", "Model download error: " + err},
            });
            return;
        }
    }

    on_event({
        {"status", "completed"},
        {"model_key", "all"},
        {"file", nullptr},
        {"file_percent", 100.0},
        {"overall_percent", 100.0},
        {"downloaded_bytes", cumulative_downloaded},
        {"total_bytes", total_expected_bytes},
        {"offline", false},
        {"message", "All required models downloaded and verified in " + util::path_to_utf8(models_dir_) + "."},
    });
}

}  // namespace mangaflow::pipeline
