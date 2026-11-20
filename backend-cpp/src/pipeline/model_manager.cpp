#include "model_manager.hpp"
#include "../utils.hpp"

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

constexpr const wchar_t* kDefaultKoharuCachePath =
    L"C:\\FILE SOFTWARE\\Koharu\\store\\hugging-face\\models";

constexpr const char* kKoharuRfDetrFolder = "mayocream--koharu-layout-rfdetr-seg-2xl-1152";
constexpr const char* kKoharuLamaFolder = "mayocream--lama-manga";
constexpr const char* kKoharuMangaOcrFolder = "mayocream--manga-ocr";
constexpr const char* kKoharuPaddleVlFolder = "PaddlePaddle--PaddleOCR-VL-1.6-GGUF";

constexpr const char* kRfDetrPrimaryFile = "koharu-layout-rfdetr-seg-2xl-1152.onnx";
constexpr const char* kRfDetrAltFile = "rfdetr-seg-2xlarge.onnx";
constexpr const char* kRfDetrConfigFile = "koharu-layout-rfdetr-seg-2xl-1152.config.json";
constexpr const char* kCtdFile = "comictextdetector.pt.onnx";

constexpr const char* kLamaMangaFile = "lama_manga.onnx";
constexpr const char* kLamaFallbackFile = "lama_fp32.onnx";

constexpr const char* kMangaOcrEncoderFile = "manga-ocr/encoder_model.onnx";
constexpr const char* kMangaOcrDecoderFile = "manga-ocr/decoder_model.onnx";
constexpr const char* kMangaOcrVocabFile = "manga-ocr/vocab.txt";
constexpr const char* kMangaOcrConfigFile = "manga-ocr/config.json";

constexpr const char* kPaddleVlModelFile = "paddleocr-vl-gguf/PaddleOCR-VL-1.6-GGUF.gguf";
constexpr const char* kPaddleVlMmprojFile = "paddleocr-vl-gguf/PaddleOCR-VL-1.6-GGUF-mmproj.gguf";

constexpr std::uintmax_t kMinRfDetrBytes = 100'000'000ULL;
constexpr std::uintmax_t kMinCtdBytes = 50'000'000ULL;
constexpr std::uintmax_t kMinLamaBytes = 100'000'000ULL;
constexpr std::uintmax_t kMinMangaOcrEncBytes = 100'000'000ULL;
constexpr std::uintmax_t kMinMangaOcrDecBytes = 30'000'000ULL;
constexpr std::uintmax_t kMinVocabBytes = 1'000ULL;
constexpr std::uintmax_t kMinGgufBytes = 100'000'000ULL;

// Minimal ABI-compatible slice of OrtDmlApi from dml_provider_factory.h
// (avoids requiring DirectML.h from the DirectML NuGet SDK at compile time).
struct MinimalOrtDmlApi {
    OrtStatus*(ORT_API_CALL* SessionOptionsAppendExecutionProvider_DML)(
        OrtSessionOptions* options,
        int device_id);
};

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
    if (auto env_py = util::get_env_wide(L"MANGA_PYTHON_EXE"); env_py.has_value() && !env_py->empty()) {
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

fs::path ModelManager::default_koharu_cache_dir() {
    for (const wchar_t* env_var : {L"KOHARU_HF_CACHE", L"KOHARU_MODELS_DIR"}) {
        if (auto env_dir = util::get_env_wide(env_var); env_dir.has_value() && !env_dir->empty()) {
            return fs::path(*env_dir);
        }
    }
    return fs::path(kDefaultKoharuCachePath);
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
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MangaFlowAI");
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
    try {
        auto providers = Ort::GetAvailableProviders();
        if (!providers.empty()) {
            return providers;
        }
    } catch (...) {
    }
    return {"CPUExecutionProvider"};
}

std::vector<std::string> ModelManager::get_preferred_providers(bool force_cpu) {
    if (force_cpu || util::env_flag_enabled(L"MANGA_FORCE_CPU")) {
        return {"CPUExecutionProvider"};
    }
    const auto available = get_available_providers();
    std::vector<std::string> ordered;
    for (const char* gpu_ep : {"CUDAExecutionProvider", "DmlExecutionProvider"}) {
        if (std::find(available.begin(), available.end(), gpu_ep) != available.end()) {
            ordered.emplace_back(gpu_ep);
        }
    }
    ordered.emplace_back("CPUExecutionProvider");
    return ordered;
}

std::string ModelManager::get_primary_provider(bool force_cpu) {
    const auto preferred = get_preferred_providers(force_cpu);
    return preferred.empty() ? "CPUExecutionProvider" : preferred.front();
}

OrtSessionHandle ModelManager::create_ort_session(const fs::path& model_path, bool force_cpu) {
    std::error_code ec;
    if (!fs::is_regular_file(model_path, ec) || ec) {
        throw std::runtime_error("ONNX model file not found: " + util::path_to_utf8(model_path));
    }

    Ort::Env& env = shared_ort_env();
    const auto preferred = get_preferred_providers(force_cpu);

    const unsigned int hw_threads = std::thread::hardware_concurrency();
    const int intra_threads = static_cast<int>(std::max(1u, std::min(8u, hw_threads > 1 ? hw_threads / 2 : 1u)));

    auto build_cpu_session = [&]() -> OrtSessionHandle {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        opts.SetIntraOpNumThreads(intra_threads);
        opts.SetInterOpNumThreads(1);
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
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            opts.SetIntraOpNumThreads(intra_threads);
            opts.SetInterOpNumThreads(1);

            if (provider == "CUDAExecutionProvider") {
                OrtCUDAProviderOptions cuda_opts{};
                cuda_opts.device_id = 0;
                opts.AppendExecutionProvider_CUDA(cuda_opts);
            } else if (provider == "DmlExecutionProvider") {
                // DirectML-safe SessionOptions per SPEC.md Section 6
                opts.DisableMemPattern();
                opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

                const void* raw_dml_api = nullptr;
                OrtStatus* api_status = Ort::GetApi().GetExecutionProviderApi(
                    "DML", ORT_API_VERSION, &raw_dml_api);
                if (api_status != nullptr || raw_dml_api == nullptr) {
                    if (api_status != nullptr) {
                        Ort::GetApi().ReleaseStatus(api_status);
                    }
                    continue;
                }
                const auto* dml_api = static_cast<const MinimalOrtDmlApi*>(raw_dml_api);
                OrtStatus* append_status =
                    dml_api->SessionOptionsAppendExecutionProvider_DML(opts, 0);
                if (append_status != nullptr) {
                    Ort::GetApi().ReleaseStatus(append_status);
                    continue;
                }
            } else {
                continue;
            }

            OrtSessionHandle handle;
            handle.session = std::make_unique<Ort::Session>(env, model_path.c_str(), opts);
            handle.active_provider = provider;
            return handle;
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
    std::optional<fs::path> fallback_workspace_models_dir,
    std::optional<fs::path> koharu_cache_dir) {
    models_dir_ = models_dir.has_value() ? *models_dir : default_models_dir();
    koharu_cache_dir_ = koharu_cache_dir.has_value() ? *koharu_cache_dir : default_koharu_cache_dir();

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
// Koharu 4-Model Cache Auto-Discovery (`<model_dir>/snapshots/<hash>/`)
// -----------------------------------------------------------------------------

std::optional<fs::path> ModelManager::resolve_koharu_snapshot(
    const fs::path& cache_root,
    const std::string& folder_name) {
    const fs::path snapshots_root = cache_root / util::utf8_to_wide(folder_name) / L"snapshots";
    std::error_code ec;
    if (!fs::is_directory(snapshots_root, ec) || ec) {
        return std::nullopt;
    }

    struct Candidate {
        fs::file_time_type newest_time{};
        fs::path dir;
    };
    std::vector<Candidate> candidates;

    for (const auto& entry : fs::directory_iterator(snapshots_root, ec)) {
        if (ec) break;
        std::error_code d_ec;
        if (!entry.is_directory(d_ec) || d_ec) continue;
        const std::string dname = util::path_to_utf8(entry.path().filename());
        if (dname.empty() || dname.front() == '.') continue;

        bool has_model_file = false;
        fs::file_time_type newest{};
        for (const auto& file_entry : fs::directory_iterator(entry.path(), d_ec)) {
            if (d_ec) break;
            std::error_code f_ec;
            if (!file_entry.is_regular_file(f_ec) || f_ec) continue;
            const std::string fname = util::path_to_utf8(file_entry.path().filename());
            if (fname.empty() || fname.front() == '.') continue;
            const std::string ext = util::to_lower_ascii(util::path_to_utf8(file_entry.path().extension()));
            if (ext != ".onnx" && ext != ".safetensors" && ext != ".gguf") continue;
            const auto fsize = fs::file_size(file_entry.path(), f_ec);
            if (f_ec || fsize < 1024ULL) continue;

            const auto mtime = fs::last_write_time(file_entry.path(), f_ec);
            if (!has_model_file || (!f_ec && mtime > newest)) {
                newest = mtime;
            }
            has_model_file = true;
        }
        if (has_model_file) {
            candidates.push_back({newest, entry.path()});
        }
    }

    if (candidates.empty()) {
        return std::nullopt;
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.newest_time > b.newest_time;
    });
    return candidates.front().dir;
}

std::vector<KoharuSnapshotStatus> ModelManager::inspect_koharu_cache() const {
    struct EntryDef {
        const char* key;
        const char* folder;
        const char* role;
    };
    static const std::array<EntryDef, 4> kDefs = {{
        {"koharu_layout_rfdetr", kKoharuRfDetrFolder, "Layout & text segmentation detector (RF-DETR-Seg-2XL)"},
        {"koharu_lama_manga", kKoharuLamaFolder, "Manga screentone LaMa inpainter"},
        {"koharu_manga_ocr", kKoharuMangaOcrFolder, "Japanese manga OCR (ViT + BERT)"},
        {"paddleocr_vl_gguf", kKoharuPaddleVlFolder, "Vision-Language OCR GGUF (for LM Studio localhost:1235)"},
    }};

    std::vector<KoharuSnapshotStatus> statuses;
    statuses.reserve(kDefs.size());

    for (const auto& def : kDefs) {
        KoharuSnapshotStatus st;
        st.key = def.key;
        st.folder_name = def.folder;
        st.role = def.role;

        if (auto snap = resolve_koharu_snapshot(koharu_cache_dir_, def.folder)) {
            st.cache_found = true;
            st.snapshot_dir = *snap;

            bool has_onnx = false;
            bool has_safetensors = false;
            bool has_gguf = false;
            std::error_code ec;
            for (const auto& fe : fs::directory_iterator(*snap, ec)) {
                if (ec) break;
                std::error_code f_ec;
                if (!fe.is_regular_file(f_ec) || f_ec) continue;
                const std::string fname = util::path_to_utf8(fe.path().filename());
                if (fname.empty() || fname.front() == '.') continue;
                const std::string ext = util::to_lower_ascii(util::path_to_utf8(fe.path().extension()));
                if (ext == ".lock") continue;
                const auto fsz = fs::file_size(fe.path(), f_ec);
                if (!f_ec) {
                    st.snapshot_bytes += fsz;
                }
                st.snapshot_files.push_back(fname);
                if (ext == ".onnx") has_onnx = true;
                if (ext == ".safetensors") has_safetensors = true;
                if (ext == ".gguf") has_gguf = true;
            }
            std::sort(st.snapshot_files.begin(), st.snapshot_files.end());

            if (has_onnx) {
                st.detected_format = "onnx";
            } else if (has_gguf) {
                st.detected_format = "gguf";
            } else if (has_safetensors) {
                st.detected_format = "safetensors";
            }
        }

        // Check whether converted/bridged inference targets are ready in persistent storage
        if (st.key == "koharu_layout_rfdetr") {
            if (auto det = resolve_detector_model("koharu-layout-rfdetr-seg-2xl", /*allow_fallback=*/false); det.has_value()) {
                st.ready_for_inference = true;
                st.target_files.push_back(util::path_to_utf8(det->model_path));
            }
            st.needs_conversion = st.cache_found && st.detected_format == "safetensors" && !st.ready_for_inference;
        } else if (st.key == "koharu_lama_manga") {
            if (auto lama = resolve_lama_model("lama-manga", /*allow_fallback=*/false); lama.has_value() && lama->is_koharu_manga) {
                st.ready_for_inference = true;
                st.target_files.push_back(util::path_to_utf8(lama->model_path));
            }
            st.needs_conversion = st.cache_found && st.detected_format == "safetensors" && !st.ready_for_inference;
        } else if (st.key == "koharu_manga_ocr") {
            if (auto mocr = resolve_manga_ocr_models(); mocr.has_value()) {
                st.ready_for_inference = true;
                st.target_files.push_back(util::path_to_utf8(mocr->encoder_onnx));
                st.target_files.push_back(util::path_to_utf8(mocr->decoder_onnx));
                st.target_files.push_back(util::path_to_utf8(mocr->vocab_txt));
            }
            st.needs_conversion = st.cache_found && st.detected_format == "safetensors" && !st.ready_for_inference;
        } else if (st.key == "paddleocr_vl_gguf") {
            if (auto gguf = resolve_paddleocr_vl_gguf(); gguf.has_value()) {
                st.ready_for_inference = true;
                st.target_files.push_back(util::path_to_utf8(gguf->model_gguf));
                if (!gguf->mmproj_gguf.empty()) {
                    st.target_files.push_back(util::path_to_utf8(gguf->mmproj_gguf));
                }
            }
            const bool copied_in_persistent =
                is_valid_regular_file(models_dir_ / util::utf8_to_wide(kPaddleVlModelFile), kMinGgufBytes);
            st.needs_conversion = st.cache_found && !copied_in_persistent;
        }

        statuses.push_back(std::move(st));
    }

    return statuses;
}

json ModelManager::get_koharu_status_json() const {
    const auto entries = inspect_koharu_cache();
    std::error_code ec;
    const bool cache_dir_exists = fs::is_directory(koharu_cache_dir_, ec) && !ec;

    json models_obj = json::object();
    int found_count = 0;
    int ready_count = 0;
    bool any_needs_conversion = false;

    for (const auto& item : entries) {
        if (item.cache_found) ++found_count;
        if (item.ready_for_inference) ++ready_count;
        if (item.needs_conversion) any_needs_conversion = true;

        models_obj[item.key] = {
            {"folder", item.folder_name},
            {"role", item.role},
            {"cache_found", item.cache_found},
            {"snapshot_dir", item.cache_found ? json(util::path_to_utf8(item.snapshot_dir)) : json(nullptr)},
            {"detected_format", item.detected_format},
            {"snapshot_files", item.snapshot_files},
            {"snapshot_mb", round_2(static_cast<double>(item.snapshot_bytes) / (1024.0 * 1024.0))},
            {"ready_for_inference", item.ready_for_inference},
            {"needs_conversion", item.needs_conversion},
            {"target_files", item.target_files},
        };
    }

    return {
        {"cache_dir", util::path_to_utf8(koharu_cache_dir_)},
        {"cache_dir_exists", cache_dir_exists},
        {"discovered_count", found_count},
        {"ready_count", ready_count},
        {"total_supported", static_cast<int>(entries.size())},
        {"any_needs_conversion", any_needs_conversion},
        {"converter_script", "packaging/convert_koharu_to_onnx.py"},
        {"models", std::move(models_obj)},
    };
}

// -----------------------------------------------------------------------------
// Dynamic Pipeline Model Catalog (`GET /api/pipeline/models`)
// -----------------------------------------------------------------------------

json ModelManager::get_pipeline_models_json(const json& active_settings) const {
    std::error_code ec;
    const bool koharu_cache_detected = fs::is_directory(koharu_cache_dir_, ec) && !ec;

    // -------------------------------------------------------------------------
    // 1. Detection Models
    // -------------------------------------------------------------------------
    json detection_models = json::array();

    // 1a. koharu-layout-rfdetr-seg-2xl
    {
        bool installed = false;
        bool ready = false;
        bool needs_conversion = false;
        std::string source = "missing";
        std::uintmax_t bytes = 0;
        json path_val = nullptr;
        json files = json::array();

        const auto snap = resolve_koharu_snapshot(koharu_cache_dir_, kKoharuRfDetrFolder);
        const bool koharu_cached = snap.has_value();

        std::optional<fs::path> cfg_path;
        if (const fs::path local_cfg = models_dir_ / util::utf8_to_wide(kRfDetrConfigFile);
            is_valid_regular_file(local_cfg, 10ULL)) {
            cfg_path = local_cfg;
        } else if (snap.has_value()) {
            const fs::path snap_cfg = *snap / L"inference_config.json";
            if (is_valid_regular_file(snap_cfg, 10ULL)) {
                cfg_path = snap_cfg;
            }
        }

        std::optional<fs::path> local_onnx;
        for (const char* fname : {kRfDetrPrimaryFile, kRfDetrAltFile}) {
            if (auto found = find_valid_file(fname, kMinRfDetrBytes)) {
                local_onnx = found;
                break;
            }
        }

        if (local_onnx.has_value()) {
            installed = true;
            ready = true;
            source = "local";
            is_valid_regular_file(*local_onnx, 1ULL, &bytes);
            path_val = util::path_to_utf8(*local_onnx);
            files.push_back(util::path_to_utf8(local_onnx->filename()));
            if (cfg_path.has_value()) {
                files.push_back(util::path_to_utf8(cfg_path->filename()));
            }
        } else if (snap.has_value()) {
            if (auto snap_onnx = find_first_with_ext(*snap, ".onnx", kMinRfDetrBytes)) {
                installed = true;
                ready = true;
                source = "koharu-cache";
                is_valid_regular_file(*snap_onnx, 1ULL, &bytes);
                path_val = util::path_to_utf8(*snap_onnx);
                files.push_back(util::path_to_utf8(snap_onnx->filename()));
            } else if (auto snap_st = find_first_with_ext(*snap, ".safetensors", 1024ULL)) {
                installed = false;
                ready = false;
                needs_conversion = true;
                source = "koharu-cache";
                is_valid_regular_file(*snap_st, 1ULL, &bytes);
                path_val = util::path_to_utf8(*snap_st);
                files.push_back(util::path_to_utf8(snap_st->filename()));
            }
        }

        detection_models.push_back({
            {"id", "koharu-layout-rfdetr-seg-2xl"},
            {"name", "Koharu Layout RF-DETR-Seg-2XL"},
            {"description", "High-precision 1152px manga layout & text segmentation (text, SFX, bubble, panel)"},
            {"installed", installed},
            {"source", source},
            {"ready_for_inference", ready},
            {"needs_conversion", needs_conversion},
            {"koharu_cached", koharu_cached},
            {"size_mb", round_2(static_cast<double>(bytes) / (1024.0 * 1024.0))},
            {"path", std::move(path_val)},
            {"files", std::move(files)},
            {"supports_thresholds", json::array({"text_threshold", "bubble_threshold", "panel_threshold"})},
            {"default_thresholds", load_rfdetr_default_thresholds_json(cfg_path)},
        });
    }

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
            {"name", "Comic Text Detector (ONNX)"},
            {"description", "Fast 1024px manga/manhwa/manhua text box & segmentation mask detector"},
            {"installed", installed},
            {"source", source},
            {"ready_for_inference", installed},
            {"needs_conversion", false},
            {"koharu_cached", false},
            {"size_mb", round_2(static_cast<double>(bytes) / (1024.0 * 1024.0))},
            {"path", std::move(path_val)},
            {"files", std::move(files)},
            {"supports_thresholds", json::array({"text_threshold", "conf_threshold", "nms_threshold", "mask_threshold"})},
            {"default_thresholds", {{"text", 0.35}, {"bubble", nullptr}, {"panel", nullptr}}},
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
    bool manga_ocr_koharu_cached = false;
    {
        const auto snap = resolve_koharu_snapshot(koharu_cache_dir_, kKoharuMangaOcrFolder);
        manga_ocr_koharu_cached = snap.has_value();

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
        } else if (snap.has_value()) {
            auto snap_enc = find_first_with_ext(*snap, ".onnx", kMinMangaOcrEncBytes, "encoder");
            auto snap_dec = find_first_with_ext(*snap, ".onnx", kMinMangaOcrDecBytes, "decoder");
            const fs::path snap_vocab = *snap / L"vocab.txt";
            if (snap_enc.has_value() && snap_dec.has_value() && is_valid_regular_file(snap_vocab, kMinVocabBytes)) {
                manga_ocr_installed = true;
                manga_ocr_source = "koharu-cache";
                std::uintmax_t sz = 0;
                if (is_valid_regular_file(*snap_enc, 1ULL, &sz)) manga_ocr_bytes += sz;
                if (is_valid_regular_file(*snap_dec, 1ULL, &sz)) manga_ocr_bytes += sz;
                if (is_valid_regular_file(snap_vocab, 1ULL, &sz)) manga_ocr_bytes += sz;
                manga_ocr_path = util::path_to_utf8(*snap);
                manga_ocr_files = {
                    util::path_to_utf8(snap_enc->filename()),
                    util::path_to_utf8(snap_dec->filename()),
                    "vocab.txt",
                };
            } else if (auto snap_st = find_first_with_ext(*snap, ".safetensors", 1024ULL)) {
                manga_ocr_installed = false;
                manga_ocr_needs_conv = true;
                manga_ocr_source = "koharu-cache";
                is_valid_regular_file(*snap_st, 1ULL, &manga_ocr_bytes);
                manga_ocr_path = util::path_to_utf8(*snap_st);
                manga_ocr_files.push_back(util::path_to_utf8(snap_st->filename()));
            }
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
    bool paddle_vl_koharu_cached = false;
    {
        const auto snap = resolve_koharu_snapshot(koharu_cache_dir_, kKoharuPaddleVlFolder);
        paddle_vl_koharu_cached = snap.has_value();

        const fs::path local_dir = models_dir_ / L"paddleocr-vl-gguf";
        auto local_model = find_first_with_ext(local_dir, ".gguf", kMinGgufBytes, "", "mmproj");
        auto local_mmproj = find_first_with_ext(local_dir, ".gguf", kMinGgufBytes, "mmproj");
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
        } else if (snap.has_value()) {
            auto snap_model = find_first_with_ext(*snap, ".gguf", kMinGgufBytes, "", "mmproj");
            auto snap_mmproj = find_first_with_ext(*snap, ".gguf", kMinGgufBytes, "mmproj");
            if (snap_model.has_value()) {
                paddle_vl_installed = true;
                paddle_vl_source = "koharu-cache";
                std::uintmax_t sz = 0;
                if (is_valid_regular_file(*snap_model, 1ULL, &sz)) paddle_vl_bytes += sz;
                paddle_vl_files.push_back(util::path_to_utf8(snap_model->filename()));
                if (snap_mmproj.has_value() && is_valid_regular_file(*snap_mmproj, 1ULL, &sz)) {
                    paddle_vl_bytes += sz;
                    paddle_vl_files.push_back(util::path_to_utf8(snap_mmproj->filename()));
                }
                paddle_vl_path = util::path_to_utf8(*snap_model);
            }
        }
    }

    // Push OCR entries in canonical option order: auto, manga-ocr, paddleocr-vl-gguf, rapidocr-ppocrv4
    {
        const bool auto_installed = manga_ocr_installed || rapidocr_installed;
        std::string auto_source = "missing";
        if (manga_ocr_installed && rapidocr_installed) {
            auto_source = (manga_ocr_source == "local" && rapidocr_source == "local") ? "local" : "koharu-cache";
        } else if (manga_ocr_installed) {
            auto_source = manga_ocr_source;
        } else if (rapidocr_installed) {
            auto_source = rapidocr_source;
        } else if (manga_ocr_source == "koharu-cache") {
            auto_source = "koharu-cache";
        }

        ocr_models.push_back({
            {"id", "auto"},
            {"name", "Auto (Language Router)"},
            {"description", "Automatically routes Japanese → manga-ocr and Korean/Chinese → RapidOCR PP-OCRv4"},
            {"installed", auto_installed},
            {"source", auto_source},
            {"ready_for_inference", auto_installed},
            {"needs_conversion", !manga_ocr_installed && manga_ocr_needs_conv},
            {"koharu_cached", manga_ocr_koharu_cached},
            {"size_mb", round_2(static_cast<double>(manga_ocr_bytes + rapidocr_bytes) / (1024.0 * 1024.0))},
            {"path", nullptr},
            {"files", json::array()},
            {"routing", {{"ja", "manga-ocr"}, {"ko", "rapidocr-ppocrv4"}, {"zh", "rapidocr-ppocrv4"}}},
        });
    }

    ocr_models.push_back({
        {"id", "manga-ocr"},
        {"name", "Manga-OCR (Japanese ONNX)"},
        {"description", "ViT + BERT autoregressive Japanese manga OCR (vertical tategaki & furigana)"},
        {"installed", manga_ocr_installed},
        {"source", manga_ocr_source},
        {"ready_for_inference", manga_ocr_installed},
        {"needs_conversion", manga_ocr_needs_conv},
        {"koharu_cached", manga_ocr_koharu_cached},
        {"size_mb", round_2(static_cast<double>(manga_ocr_bytes) / (1024.0 * 1024.0))},
        {"path", std::move(manga_ocr_path)},
        {"files", std::move(manga_ocr_files)},
        {"languages", json::array({"ja"})},
    });

    ocr_models.push_back({
        {"id", "paddleocr-vl-gguf"},
        {"name", "PaddleOCR-VL-1.6 (GGUF Vision-OCR)"},
        {"description", "Multilingual Vision-Language OCR via LM Studio (http://localhost:1235/v1)"},
        {"installed", paddle_vl_installed},
        {"source", paddle_vl_source},
        {"ready_for_inference", paddle_vl_installed},
        {"needs_conversion", false},
        {"koharu_cached", paddle_vl_koharu_cached},
        {"size_mb", round_2(static_cast<double>(paddle_vl_bytes) / (1024.0 * 1024.0))},
        {"path", std::move(paddle_vl_path)},
        {"files", std::move(paddle_vl_files)},
        {"runtime", "lmstudio-vision"},
        {"lm_studio_hint", "Load PaddleOCR-VL-1.6-GGUF in LM Studio (http://localhost:1235/v1) for Vision-OCR mode"},
        {"languages", json::array({"ja", "ko", "zh", "en"})},
    });

    ocr_models.push_back({
        {"id", "rapidocr-ppocrv4"},
        {"name", "RapidOCR PP-OCRv4 (ONNX)"},
        {"description", "Fast CJK/Latin PP-OCRv4 ONNX recognition (Korean, Chinese & horizontal Japanese)"},
        {"installed", rapidocr_installed},
        {"source", rapidocr_source},
        {"ready_for_inference", rapidocr_installed},
        {"needs_conversion", false},
        {"koharu_cached", false},
        {"size_mb", round_2(static_cast<double>(rapidocr_bytes) / (1024.0 * 1024.0))},
        {"path", std::move(rapidocr_path)},
        {"files", std::move(rapidocr_files)},
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
        bool needs_conversion = false;
        std::string source = "missing";
        std::uintmax_t bytes = 0;
        json path_val = nullptr;
        json files = json::array();

        const auto snap = resolve_koharu_snapshot(koharu_cache_dir_, kKoharuLamaFolder);
        const bool koharu_cached = snap.has_value();

        if (auto found = find_valid_file(kLamaMangaFile, kMinLamaBytes)) {
            installed = true;
            ready = true;
            source = "local";
            is_valid_regular_file(*found, 1ULL, &bytes);
            path_val = util::path_to_utf8(*found);
            files.push_back(util::path_to_utf8(found->filename()));
        } else if (snap.has_value()) {
            if (auto snap_onnx = find_first_with_ext(*snap, ".onnx", kMinLamaBytes)) {
                installed = true;
                ready = true;
                source = "koharu-cache";
                is_valid_regular_file(*snap_onnx, 1ULL, &bytes);
                path_val = util::path_to_utf8(*snap_onnx);
                files.push_back(util::path_to_utf8(snap_onnx->filename()));
            } else if (auto snap_st = find_first_with_ext(*snap, ".safetensors", 1024ULL)) {
                installed = false;
                ready = false;
                needs_conversion = true;
                source = "koharu-cache";
                is_valid_regular_file(*snap_st, 1ULL, &bytes);
                path_val = util::path_to_utf8(*snap_st);
                files.push_back(util::path_to_utf8(snap_st->filename()));
            }
        }

        inpainting_models.push_back({
            {"id", "lama-manga"},
            {"name", "LaMa-Manga (Koharu Screentone)"},
            {"description", "Manga-tuned Big-LaMa ONNX inpainter optimized for screentones & line art"},
            {"installed", installed},
            {"source", source},
            {"ready_for_inference", ready},
            {"needs_conversion", needs_conversion},
            {"koharu_cached", koharu_cached},
            {"size_mb", round_2(static_cast<double>(bytes) / (1024.0 * 1024.0))},
            {"path", std::move(path_val)},
            {"files", std::move(files)},
        });
    }

    // 3b. lama-fp32
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
            {"name", "LaMa FP32 (Standard)"},
            {"description", "General-purpose Big-LaMa FP32 ONNX inpainter (512×512 tiles)"},
            {"installed", installed},
            {"source", source},
            {"ready_for_inference", installed},
            {"needs_conversion", false},
            {"koharu_cached", false},
            {"size_mb", round_2(static_cast<double>(bytes) / (1024.0 * 1024.0))},
            {"path", std::move(path_val)},
            {"files", std::move(files)},
        });
    }

    const json active = {
        {"detection_model", active_settings.value("detection_model", std::string("koharu-layout-rfdetr-seg-2xl"))},
        {"ocr_model", active_settings.value("ocr_model", std::string("auto"))},
        {"inpainting_model", active_settings.value("inpainting_model", std::string("lama-manga"))},
        {"text_threshold", active_settings.contains("text_threshold") ? active_settings["text_threshold"] : json(nullptr)},
        {"bubble_threshold", active_settings.contains("bubble_threshold") ? active_settings["bubble_threshold"] : json(nullptr)},
        {"panel_threshold", active_settings.contains("panel_threshold") ? active_settings["panel_threshold"] : json(nullptr)},
    };

    return {
        {"models_dir", util::path_to_utf8(models_dir_)},
        {"koharu_cache_dir", util::path_to_utf8(koharu_cache_dir_)},
        {"koharu_cache_detected", koharu_cache_detected},
        {"execution_provider", get_primary_provider()},
        {"available_providers", get_available_providers()},
        {"active", active},
        {"detection", std::move(detection_models)},
        {"ocr", std::move(ocr_models)},
        {"inpainting", std::move(inpainting_models)},
    };
}

// -----------------------------------------------------------------------------
// Koharu Local Cache Importer (`POST /api/models/import-koharu`)
// -----------------------------------------------------------------------------

json ModelManager::import_koharu_models(const KoharuImportOptions& options) const {
    const fs::path cache_root = options.koharu_cache_dir.has_value()
        ? *options.koharu_cache_dir
        : koharu_cache_dir_;

    std::error_code ec;
    fs::create_directories(models_dir_, ec);

    const bool cache_exists = fs::is_directory(cache_root, ec) && !ec;
    if (!cache_exists) {
        return {
            {"ok", false},
            {"status", "cache_not_found"},
            {"detail", "Koharu Hugging Face cache directory not found: " + util::path_to_utf8(cache_root)},
            {"koharu_cache_dir", util::path_to_utf8(cache_root)},
            {"models_dir", util::path_to_utf8(models_dir_)},
            {"conversion_invoked", false},
            {"results", json::object()},
        };
    }

    // Resolve requested subset (empty = all 4 Koharu models)
    std::unordered_set<std::string> selected;
    if (options.models.empty()) {
        selected = {"rfdetr", "lama", "manga_ocr", "paddleocr_vl"};
    } else {
        for (const auto& raw : options.models) {
            const std::string k = util::to_lower_ascii(util::trim_ascii(raw));
            if (k.empty() || k == "all") {
                selected = {"rfdetr", "lama", "manga_ocr", "paddleocr_vl"};
                break;
            }
            if (k == "rfdetr" || k == "koharu_layout_rfdetr" || k == "koharu-layout-rfdetr-seg-2xl") {
                selected.insert("rfdetr");
            } else if (k == "lama" || k == "lama-manga" || k == "lama_manga" || k == "koharu_lama_manga") {
                selected.insert("lama");
            } else if (k == "manga_ocr" || k == "manga-ocr" || k == "koharu_manga_ocr") {
                selected.insert("manga_ocr");
            } else if (k == "paddleocr_vl" || k == "paddleocr-vl" || k == "paddleocr-vl-gguf" || k == "paddleocr_vl_gguf") {
                selected.insert("paddleocr_vl");
            }
        }
    }

    json results = json::object();
    std::vector<std::string> need_python_conversion;
    bool had_error = false;

    // 1. koharu-layout-rfdetr-seg-2xl
    if (selected.count("rfdetr")) {
        const std::string model_id = "koharu-layout-rfdetr-seg-2xl";
        const auto snap = resolve_koharu_snapshot(cache_root, kKoharuRfDetrFolder);
        const fs::path dst_onnx = models_dir_ / util::utf8_to_wide(kRfDetrPrimaryFile);
        const fs::path dst_cfg = models_dir_ / util::utf8_to_wide(kRfDetrConfigFile);

        if (!snap.has_value()) {
            results[model_id] = {
                {"status", "missing"},
                {"installed", is_valid_regular_file(dst_onnx, kMinRfDetrBytes)},
                {"source", is_valid_regular_file(dst_onnx, kMinRfDetrBytes) ? "local" : "missing"},
                {"message", "Snapshot not found in Koharu cache."},
            };
        } else {
            // Sync inference_config.json if present
            const fs::path src_cfg = *snap / L"inference_config.json";
            if (is_valid_regular_file(src_cfg, 10ULL) &&
                (options.force || !is_valid_regular_file(dst_cfg, 10ULL))) {
                std::string act, err;
                copy_or_link_file_atomic(src_cfg, dst_cfg, false, act, err);
            }

            if (!options.force && is_valid_regular_file(dst_onnx, kMinRfDetrBytes)) {
                results[model_id] = {
                    {"status", "already_installed"},
                    {"action", "skipped"},
                    {"installed", true},
                    {"source", "local"},
                    {"snapshot_dir", util::path_to_utf8(*snap)},
                    {"output", util::path_to_utf8(dst_onnx)},
                };
            } else if (auto snap_onnx = find_first_with_ext(*snap, ".onnx", kMinRfDetrBytes)) {
                std::string act, err;
                if (copy_or_link_file_atomic(*snap_onnx, dst_onnx, options.prefer_hardlink, act, err)) {
                    results[model_id] = {
                        {"status", "imported"},
                        {"action", act},
                        {"installed", true},
                        {"source", "local"},
                        {"snapshot_dir", util::path_to_utf8(*snap)},
                        {"output", util::path_to_utf8(dst_onnx)},
                    };
                } else {
                    had_error = true;
                    results[model_id] = {{"status", "error"}, {"installed", false}, {"error", err}};
                }
            } else if (snapshot_has_safetensors(*snap)) {
                need_python_conversion.push_back("rfdetr");
                results[model_id] = {
                    {"status", "pending_conversion"},
                    {"installed", false},
                    {"source", "koharu-cache"},
                    {"snapshot_dir", util::path_to_utf8(*snap)},
                };
            } else {
                results[model_id] = {
                    {"status", "missing"},
                    {"installed", false},
                    {"source", "missing"},
                    {"message", "No .onnx or .safetensors weights found in Koharu snapshot."},
                };
            }
        }
    }

    // 2. lama-manga
    if (selected.count("lama")) {
        const std::string model_id = "lama-manga";
        const auto snap = resolve_koharu_snapshot(cache_root, kKoharuLamaFolder);
        const fs::path dst_onnx = models_dir_ / util::utf8_to_wide(kLamaMangaFile);

        if (!snap.has_value()) {
            results[model_id] = {
                {"status", "missing"},
                {"installed", is_valid_regular_file(dst_onnx, kMinLamaBytes)},
                {"source", is_valid_regular_file(dst_onnx, kMinLamaBytes) ? "local" : "missing"},
                {"message", "Snapshot not found in Koharu cache."},
            };
        } else if (!options.force && is_valid_regular_file(dst_onnx, kMinLamaBytes)) {
            results[model_id] = {
                {"status", "already_installed"},
                {"action", "skipped"},
                {"installed", true},
                {"source", "local"},
                {"snapshot_dir", util::path_to_utf8(*snap)},
                {"output", util::path_to_utf8(dst_onnx)},
            };
        } else if (auto snap_onnx = find_first_with_ext(*snap, ".onnx", kMinLamaBytes)) {
            std::string act, err;
            if (copy_or_link_file_atomic(*snap_onnx, dst_onnx, options.prefer_hardlink, act, err)) {
                results[model_id] = {
                    {"status", "imported"},
                    {"action", act},
                    {"installed", true},
                    {"source", "local"},
                    {"snapshot_dir", util::path_to_utf8(*snap)},
                    {"output", util::path_to_utf8(dst_onnx)},
                };
            } else {
                had_error = true;
                results[model_id] = {{"status", "error"}, {"installed", false}, {"error", err}};
            }
        } else if (snapshot_has_safetensors(*snap)) {
            need_python_conversion.push_back("lama");
            results[model_id] = {
                {"status", "pending_conversion"},
                {"installed", false},
                {"source", "koharu-cache"},
                {"snapshot_dir", util::path_to_utf8(*snap)},
            };
        } else {
            results[model_id] = {
                {"status", "missing"},
                {"installed", false},
                {"source", "missing"},
                {"message", "No .onnx or .safetensors weights found in Koharu snapshot."},
            };
        }
    }

    // 3. manga-ocr
    if (selected.count("manga_ocr")) {
        const std::string model_id = "manga-ocr";
        const auto snap = resolve_koharu_snapshot(cache_root, kKoharuMangaOcrFolder);
        const fs::path target_dir = models_dir_ / L"manga-ocr";
        const fs::path dst_enc = models_dir_ / util::utf8_to_wide(kMangaOcrEncoderFile);
        const fs::path dst_dec = models_dir_ / util::utf8_to_wide(kMangaOcrDecoderFile);
        const fs::path dst_vocab = models_dir_ / util::utf8_to_wide(kMangaOcrVocabFile);
        const bool already_local =
            is_valid_regular_file(dst_enc, kMinMangaOcrEncBytes) &&
            is_valid_regular_file(dst_dec, kMinMangaOcrDecBytes) &&
            is_valid_regular_file(dst_vocab, kMinVocabBytes);

        if (!snap.has_value()) {
            results[model_id] = {
                {"status", "missing"},
                {"installed", already_local},
                {"source", already_local ? "local" : "missing"},
                {"message", "Snapshot not found in Koharu cache."},
            };
        } else {
            // Sync tokenizer/config metadata files from snapshot
            for (const wchar_t* meta : {L"vocab.txt", L"config.json", L"preprocessor_config.json"}) {
                const fs::path src_m = *snap / meta;
                const fs::path dst_m = target_dir / meta;
                if (is_valid_regular_file(src_m, 10ULL) &&
                    (options.force || !is_valid_regular_file(dst_m, 10ULL))) {
                    std::string act, err;
                    copy_or_link_file_atomic(src_m, dst_m, false, act, err);
                }
            }

            if (!options.force && already_local) {
                results[model_id] = {
                    {"status", "already_installed"},
                    {"action", "skipped"},
                    {"installed", true},
                    {"source", "local"},
                    {"snapshot_dir", util::path_to_utf8(*snap)},
                    {"output_dir", util::path_to_utf8(target_dir)},
                };
            } else {
                auto snap_enc = find_first_with_ext(*snap, ".onnx", kMinMangaOcrEncBytes, "encoder");
                auto snap_dec = find_first_with_ext(*snap, ".onnx", kMinMangaOcrDecBytes, "decoder");
                if (snap_enc.has_value() && snap_dec.has_value()) {
                    std::string act1, act2, err1, err2;
                    const bool ok1 = copy_or_link_file_atomic(*snap_enc, dst_enc, options.prefer_hardlink, act1, err1);
                    const bool ok2 = ok1 && copy_or_link_file_atomic(*snap_dec, dst_dec, options.prefer_hardlink, act2, err2);
                    if (ok2) {
                        results[model_id] = {
                            {"status", "imported"},
                            {"action", act1},
                            {"installed", true},
                            {"source", "local"},
                            {"snapshot_dir", util::path_to_utf8(*snap)},
                            {"output_dir", util::path_to_utf8(target_dir)},
                        };
                    } else {
                        had_error = true;
                        results[model_id] = {{"status", "error"}, {"installed", false}, {"error", err1.empty() ? err2 : err1}};
                    }
                } else if (snapshot_has_safetensors(*snap)) {
                    need_python_conversion.push_back("manga_ocr");
                    results[model_id] = {
                        {"status", "pending_conversion"},
                        {"installed", false},
                        {"source", "koharu-cache"},
                        {"snapshot_dir", util::path_to_utf8(*snap)},
                    };
                } else {
                    results[model_id] = {
                        {"status", "missing"},
                        {"installed", false},
                        {"source", "missing"},
                        {"message", "No .onnx or .safetensors weights found in Koharu snapshot."},
                    };
                }
            }
        }
    }

    // 4. paddleocr-vl-gguf (direct copy / hardlink of .gguf + -mmproj.gguf, zero conversion needed)
    if (selected.count("paddleocr_vl")) {
        const std::string model_id = "paddleocr-vl-gguf";
        const auto snap = resolve_koharu_snapshot(cache_root, kKoharuPaddleVlFolder);
        const fs::path target_dir = models_dir_ / L"paddleocr-vl-gguf";

        if (!snap.has_value()) {
            const bool has_local = find_first_with_ext(target_dir, ".gguf", kMinGgufBytes, "", "mmproj").has_value();
            results[model_id] = {
                {"status", "missing"},
                {"installed", has_local},
                {"source", has_local ? "local" : "missing"},
                {"message", "Snapshot not found in Koharu cache."},
            };
        } else {
            std::vector<fs::path> gguf_files;
            std::error_code dir_ec;
            for (const auto& entry : fs::directory_iterator(*snap, dir_ec)) {
                if (dir_ec) break;
                std::error_code f_ec;
                if (!entry.is_regular_file(f_ec) || f_ec) continue;
                const std::string fname = util::path_to_utf8(entry.path().filename());
                if (fname.empty() || fname.front() == '.') continue;
                if (util::to_lower_ascii(util::path_to_utf8(entry.path().extension())) == ".gguf") {
                    gguf_files.push_back(entry.path());
                }
            }
            std::sort(gguf_files.begin(), gguf_files.end());

            if (gguf_files.empty()) {
                results[model_id] = {
                    {"status", "missing"},
                    {"installed", false},
                    {"source", "missing"},
                    {"message", "No .gguf files found in Koharu snapshot."},
                };
            } else {
                json file_reports = json::array();
                bool any_copied = false;
                bool gguf_err = false;
                std::string last_err;

                for (const auto& src : gguf_files) {
                    const fs::path dst = target_dir / src.filename();
                    std::uintmax_t src_sz = 0;
                    std::uintmax_t dst_sz = 0;
                    is_valid_regular_file(src, 1ULL, &src_sz);
                    const bool same_size = is_valid_regular_file(dst, 1ULL, &dst_sz) && (dst_sz == src_sz);

                    if (!options.force && same_size) {
                        file_reports.push_back({
                            {"file", util::path_to_utf8(dst)},
                            {"size_bytes", dst_sz},
                            {"action", "skipped"},
                        });
                        continue;
                    }

                    std::string act, err;
                    if (copy_or_link_file_atomic(src, dst, options.prefer_hardlink, act, err)) {
                        any_copied = true;
                        file_reports.push_back({
                            {"file", util::path_to_utf8(dst)},
                            {"size_bytes", src_sz},
                            {"action", act},
                        });
                    } else {
                        gguf_err = true;
                        had_error = true;
                        last_err = err;
                        break;
                    }
                }

                if (!gguf_err) {
                    const json hint = {
                        {"purpose", "Load PaddleOCR-VL-1.6-GGUF in LM Studio (http://localhost:1235/v1) for Vision-OCR mode"},
                        {"files", file_reports},
                        {"source_snapshot", util::path_to_utf8(*snap)},
                    };
                    const std::string hint_str = hint.dump(2) + "\n";
                    util::atomic_write_bytes(target_dir / L"lmstudio_vision_ocr.json", hint_str.data(), hint_str.size());

                    results[model_id] = {
                        {"status", any_copied ? "imported" : "already_installed"},
                        {"action", any_copied ? (options.prefer_hardlink ? "linked_or_copied" : "copied") : "skipped"},
                        {"installed", true},
                        {"source", "local"},
                        {"snapshot_dir", util::path_to_utf8(*snap)},
                        {"output_dir", util::path_to_utf8(target_dir)},
                        {"files", std::move(file_reports)},
                    };
                } else {
                    results[model_id] = {
                        {"status", "error"},
                        {"installed", false},
                        {"error", last_err},
                    };
                }
            }
        }
    }

    // 5. Invoke `packaging/convert_koharu_to_onnx.py` if any `.safetensors` models require conversion
    bool conversion_invoked = false;
    json conversion_exit_code = nullptr;
    std::string conversion_output;
    std::string conversion_error;

    if (!need_python_conversion.empty()) {
        if (!options.convert_safetensors) {
            for (const auto& conv_key : need_python_conversion) {
                const std::string mid = (conv_key == "rfdetr")
                    ? "koharu-layout-rfdetr-seg-2xl"
                    : (conv_key == "lama" ? "lama-manga" : "manga-ocr");
                results[mid]["status"] = "needs_conversion";
                results[mid]["message"] = "Safetensors conversion skipped (convert=false). Run packaging/convert_koharu_to_onnx.py.";
            }
        } else {
            const auto script_opt = find_koharu_converter_script(options.search_root_dir);
            const auto py_opt = script_opt.has_value()
                ? find_python_executable(script_opt->parent_path().parent_path())
                : std::nullopt;

            if (!script_opt.has_value() || !py_opt.has_value()) {
                had_error = true;
                conversion_error = !script_opt.has_value()
                    ? "packaging/convert_koharu_to_onnx.py was not found relative to the executable."
                    : "Python executable (backend/venv/Scripts/python.exe or python.exe on PATH) was not found for safetensors conversion.";
                for (const auto& conv_key : need_python_conversion) {
                    const std::string mid = (conv_key == "rfdetr")
                        ? "koharu-layout-rfdetr-seg-2xl"
                        : (conv_key == "lama" ? "lama-manga" : "manga-ocr");
                    results[mid]["status"] = "needs_conversion";
                    results[mid]["error"] = conversion_error;
                }
            } else {
                std::string models_csv;
                for (size_t i = 0; i < need_python_conversion.size(); ++i) {
                    if (i > 0) models_csv.push_back(',');
                    models_csv += need_python_conversion[i];
                }

                // Stage outside %LOCALAPPDATA% to avoid Microsoft Store Python MSIX virtualization
                const fs::path stage_dir =
                    script_opt->parent_path() / util::utf8_to_wide("_koharu_stage_" + util::generate_uuid_hex().substr(0, 8));
                std::error_code st_ec;
                fs::create_directories(stage_dir, st_ec);

                std::vector<std::wstring> argv = {
                    py_opt->wstring(),
                    script_opt->wstring(),
                    L"--koharu-cache",
                    cache_root.wstring(),
                    L"--output-dir",
                    stage_dir.wstring(),
                    L"--models",
                    util::utf8_to_wide(models_csv),
                };
                if (options.force) {
                    argv.push_back(L"--force");
                }

                int exit_code = -1;
                std::string run_err;
                conversion_invoked = true;
                const bool launched = run_subprocess_capture(
                    argv,
                    script_opt->parent_path().parent_path(),
                    exit_code,
                    conversion_output,
                    run_err);
                conversion_exit_code = exit_code;

                if (!launched) {
                    had_error = true;
                    conversion_error = run_err;
                } else if (exit_code != 0) {
                    had_error = true;
                    conversion_error = "convert_koharu_to_onnx.py exited with code " + std::to_string(exit_code);
                } else {
                    std::string move_err;
                    if (!move_stage_dir_contents(stage_dir, models_dir_, move_err)) {
                        had_error = true;
                        conversion_error = move_err;
                    }
                }

                fs::remove_all(stage_dir, st_ec);

                // Refresh status for every model handed to Python conversion
                for (const auto& conv_key : need_python_conversion) {
                    if (conv_key == "rfdetr") {
                        const fs::path dst_onnx = models_dir_ / util::utf8_to_wide(kRfDetrPrimaryFile);
                        const bool ok = is_valid_regular_file(dst_onnx, kMinRfDetrBytes);
                        results["koharu-layout-rfdetr-seg-2xl"]["installed"] = ok;
                        results["koharu-layout-rfdetr-seg-2xl"]["source"] = ok ? "local" : "koharu-cache";
                        results["koharu-layout-rfdetr-seg-2xl"]["status"] = ok ? "converted" : "error";
                        results["koharu-layout-rfdetr-seg-2xl"]["action"] = ok ? "converted_onnx" : "failed";
                        if (ok) results["koharu-layout-rfdetr-seg-2xl"]["output"] = util::path_to_utf8(dst_onnx);
                        if (!ok && !conversion_error.empty()) {
                            results["koharu-layout-rfdetr-seg-2xl"]["error"] = conversion_error;
                        }
                    } else if (conv_key == "lama") {
                        const fs::path dst_onnx = models_dir_ / util::utf8_to_wide(kLamaMangaFile);
                        const bool ok = is_valid_regular_file(dst_onnx, kMinLamaBytes);
                        results["lama-manga"]["installed"] = ok;
                        results["lama-manga"]["source"] = ok ? "local" : "koharu-cache";
                        results["lama-manga"]["status"] = ok ? "converted" : "error";
                        results["lama-manga"]["action"] = ok ? "converted_onnx" : "failed";
                        if (ok) results["lama-manga"]["output"] = util::path_to_utf8(dst_onnx);
                        if (!ok && !conversion_error.empty()) {
                            results["lama-manga"]["error"] = conversion_error;
                        }
                    } else if (conv_key == "manga_ocr") {
                        const bool ok =
                            is_valid_regular_file(models_dir_ / util::utf8_to_wide(kMangaOcrEncoderFile), kMinMangaOcrEncBytes) &&
                            is_valid_regular_file(models_dir_ / util::utf8_to_wide(kMangaOcrDecoderFile), kMinMangaOcrDecBytes) &&
                            is_valid_regular_file(models_dir_ / util::utf8_to_wide(kMangaOcrVocabFile), kMinVocabBytes);
                        results["manga-ocr"]["installed"] = ok;
                        results["manga-ocr"]["source"] = ok ? "local" : "koharu-cache";
                        results["manga-ocr"]["status"] = ok ? "converted" : "error";
                        results["manga-ocr"]["action"] = ok ? "converted_onnx" : "failed";
                        if (ok) results["manga-ocr"]["output_dir"] = util::path_to_utf8(models_dir_ / L"manga-ocr");
                        if (!ok && !conversion_error.empty()) {
                            results["manga-ocr"]["error"] = conversion_error;
                        }
                    }
                }
            }
        }
    }

    return {
        {"ok", !had_error},
        {"status", had_error ? "partial_or_error" : "ok"},
        {"koharu_cache_dir", util::path_to_utf8(cache_root)},
        {"models_dir", util::path_to_utf8(models_dir_)},
        {"conversion_invoked", conversion_invoked},
        {"conversion_exit_code", conversion_exit_code},
        {"conversion_output", conversion_output},
        {"conversion_error", conversion_error.empty() ? json(nullptr) : json(conversion_error)},
        {"results", std::move(results)},
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
    const bool wants_rfdetr = is_auto ||
                              pref == "rfdetr" ||
                              pref == "rf-detr" ||
                              pref == "koharu_rfdetr" ||
                              pref == "koharu-rfdetr" ||
                              pref == "koharu-layout-rfdetr-seg-2xl" ||
                              pref == "koharu-layout-rfdetr-seg-2xl-1152" ||
                              pref == "koharu_layout_rfdetr_seg_2xl" ||
                              pref == "rfdetr-seg-2xl" ||
                              pref == "rfdetr-seg-2xlarge";
    const bool wants_ctd = is_auto ||
                           pref == "ctd" ||
                           pref == "comic_text_detector" ||
                           pref == "comic-text-detector" ||
                           pref == "comictextdetector";

    auto try_rfdetr = [&]() -> std::optional<ResolvedDetectorModel> {
        // 1. Converted Koharu RF-DETR ONNX in persistent `%LOCALAPPDATA%\ComicTranslator\models` (or workspace)
        for (const char* fname : {kRfDetrPrimaryFile, kRfDetrAltFile}) {
            std::string src;
            if (auto found = find_valid_file(fname, kMinRfDetrBytes, &src)) {
                ResolvedDetectorModel res;
                res.backend = DetectorBackend::KoharuRfDetr;
                res.model_path = *found;
                res.source = (src == "persistent_models_dir") ? "koharu_converted_onnx" : src;
                const fs::path cfg = models_dir_ / util::utf8_to_wide(kRfDetrConfigFile);
                if (is_valid_regular_file(cfg, 10ULL)) {
                    res.config_path = cfg;
                } else if (auto snap = resolve_koharu_snapshot(koharu_cache_dir_, kKoharuRfDetrFolder)) {
                    const fs::path snap_cfg = *snap / L"inference_config.json";
                    if (is_valid_regular_file(snap_cfg, 10ULL)) {
                        res.config_path = snap_cfg;
                    }
                }
                return res;
            }
        }
        // 2. Direct `.onnx` inside Koharu HF snapshot (if Koharu ever ships ONNX directly)
        if (auto snap = resolve_koharu_snapshot(koharu_cache_dir_, kKoharuRfDetrFolder)) {
            if (auto snap_onnx = find_first_with_ext(*snap, ".onnx", kMinRfDetrBytes)) {
                ResolvedDetectorModel res;
                res.backend = DetectorBackend::KoharuRfDetr;
                res.model_path = *snap_onnx;
                res.source = "koharu_snapshot_onnx";
                const fs::path cfg = *snap / L"inference_config.json";
                if (is_valid_regular_file(cfg, 10ULL)) {
                    res.config_path = cfg;
                }
                return res;
            }
        }
        return std::nullopt;
    };

    auto try_ctd = [&]() -> std::optional<ResolvedDetectorModel> {
        std::string src;
        if (auto found = find_valid_file(kCtdFile, kMinCtdBytes, &src)) {
            ResolvedDetectorModel res;
            res.backend = DetectorBackend::ComicTextDetector;
            res.model_path = *found;
            res.source = src;
            return res;
        }
        return std::nullopt;
    };

    if (wants_rfdetr) {
        if (auto r = try_rfdetr(); r.has_value()) {
            return r;
        }
        if (is_auto || allow_fallback) {
            if (auto c = try_ctd(); c.has_value()) {
                return c;
            }
        }
        return std::nullopt;
    }

    if (wants_ctd) {
        if (auto c = try_ctd(); c.has_value()) {
            return c;
        }
        if (allow_fallback) {
            if (auto r = try_rfdetr(); r.has_value()) {
                return r;
            }
        }
        return std::nullopt;
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
            return ResolvedLamaModel{*found, "koharu_converted_onnx", true};
        }
        if (auto snap = resolve_koharu_snapshot(koharu_cache_dir_, kKoharuLamaFolder)) {
            if (auto snap_onnx = find_first_with_ext(*snap, ".onnx", kMinLamaBytes)) {
                return ResolvedLamaModel{*snap_onnx, "koharu_snapshot_onnx", true};
            }
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
        res.source = (src == "persistent_models_dir") ? "koharu_converted_onnx" : src;
        return res;
    }

    if (auto snap = resolve_koharu_snapshot(koharu_cache_dir_, kKoharuMangaOcrFolder)) {
        auto snap_enc = find_first_with_ext(*snap, ".onnx", kMinMangaOcrEncBytes, "encoder");
        auto snap_dec = find_first_with_ext(*snap, ".onnx", kMinMangaOcrDecBytes, "decoder");
        const fs::path snap_vocab = *snap / L"vocab.txt";
        if (snap_enc.has_value() && snap_dec.has_value() && is_valid_regular_file(snap_vocab, kMinVocabBytes)) {
            ResolvedMangaOcrModel res;
            res.encoder_onnx = *snap_enc;
            res.decoder_onnx = *snap_dec;
            res.vocab_txt = snap_vocab;
            const fs::path snap_cfg = *snap / L"config.json";
            if (is_valid_regular_file(snap_cfg, 100ULL)) {
                res.config_json = snap_cfg;
            }
            res.source = "koharu_snapshot_onnx";
            return res;
        }
    }

    return std::nullopt;
}

std::optional<ResolvedPaddleOcrVlGguf> ModelManager::resolve_paddleocr_vl_gguf() const {
    // 1. Persistent `%LOCALAPPDATA%\ComicTranslator\models\paddleocr-vl-gguf\`
    const fs::path target_dir = models_dir_ / L"paddleocr-vl-gguf";
    auto model_f = find_first_with_ext(target_dir, ".gguf", kMinGgufBytes, "", "mmproj");
    auto mmproj_f = find_first_with_ext(target_dir, ".gguf", kMinGgufBytes, "mmproj");
    if (model_f.has_value()) {
        ResolvedPaddleOcrVlGguf res;
        res.model_gguf = *model_f;
        if (mmproj_f.has_value()) res.mmproj_gguf = *mmproj_f;
        res.source = "persistent_models_dir";
        return res;
    }

    // 2. Direct Koharu Hugging Face cache snapshot
    if (auto snap = resolve_koharu_snapshot(koharu_cache_dir_, kKoharuPaddleVlFolder)) {
        auto snap_model = find_first_with_ext(*snap, ".gguf", kMinGgufBytes, "", "mmproj");
        auto snap_mmproj = find_first_with_ext(*snap, ".gguf", kMinGgufBytes, "mmproj");
        if (snap_model.has_value()) {
            ResolvedPaddleOcrVlGguf res;
            res.model_gguf = *snap_model;
            if (snap_mmproj.has_value()) res.mmproj_gguf = *snap_mmproj;
            res.source = "koharu_snapshot_gguf";
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
            {"size_mb", installed && det->backend == DetectorBackend::KoharuRfDetr ? 154 : 90},
            {"downloaded_mb", round_2(static_cast<double>(actual_bytes) / (1024.0 * 1024.0))},
            {"files", std::move(files)},
            {"missing_files", std::move(missing)},
            {"active_backend", installed ? json(detector_backend_name(det->backend)) : json(nullptr)},
            {"active_path", installed ? json(util::path_to_utf8(det->model_path)) : json(nullptr)},
            {"source", installed ? json(det->source) : json(nullptr)},
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
        models["manga_ocr"] = {
            {"installed", installed},
            {"size_mb", 424},
            {"downloaded_mb", round_2(static_cast<double>(actual_bytes) / (1024.0 * 1024.0))},
            {"files", std::move(files)},
            {"missing_files", std::move(missing)},
            {"format", active_format},
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
            {"is_koharu_manga", installed ? lama->is_koharu_manga : false},
            {"active_path", installed ? json(util::path_to_utf8(lama->model_path)) : json(nullptr)},
            {"source", installed ? json(lama->source) : json(nullptr)},
        };
    }

    // Optional Vision-OCR GGUF summary for LM Studio (`localhost:1235`)
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
            {"model_gguf", util::path_to_utf8(gguf->model_gguf)},
            {"mmproj_gguf", gguf->mmproj_gguf.empty() ? json(nullptr) : json(util::path_to_utf8(gguf->mmproj_gguf))},
            {"size_mb", round_2(static_cast<double>(sz_total) / (1024.0 * 1024.0))},
            {"lm_studio_hint", "Load in LM Studio (http://localhost:1235/v1) for Vision-OCR mode"},
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
        {"koharu_cache", get_koharu_status_json()},
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
        L"MangaFlowAI/2.0",
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
