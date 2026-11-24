#include "ort_provider.hpp"

#include "../utils.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi.h>
#include <dxgi1_4.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <mutex>
#include <thread>

namespace mangaflow::pipeline {

namespace {

std::mutex& preferred_backend_mutex() {
    static std::mutex s_mtx;
    return s_mtx;
}

std::string& preferred_backend_storage() {
    static std::string s_pref = "auto";
    return s_pref;
}

bool probe_dll_on_search_path(const wchar_t* dll_name) noexcept {
    HMODULE mod = LoadLibraryExW(
        dll_name,
        nullptr,
        LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (mod != nullptr) {
        FreeLibrary(mod);
        return true;
    }
    // Fallback to standard SearchPathW in case CUDA bin is on PATH
    wchar_t found[MAX_PATH] = {};
    return SearchPathW(nullptr, dll_name, nullptr, MAX_PATH, found, nullptr) > 0;
}

// Minimal ABI-compatible slice of OrtDmlApi from dml_provider_factory.h
// (avoids requiring DirectML.h from the DirectML NuGet SDK at compile time).
struct MinimalOrtDmlApi {
    OrtStatus*(ORT_API_CALL* SessionOptionsAppendExecutionProvider_DML)(
        OrtSessionOptions* options,
        int device_id);
};

std::atomic<uint64_t>& cuda_mem_limit_mb_storage() {
    static std::atomic<uint64_t> s_limit_mb{[]() -> uint64_t {
        if (auto env = util::get_env_utf8(L"VELLUM_CUDA_MEM_LIMIT_MB");
            env.has_value() && !env->empty()) {
            try {
                const uint64_t parsed = std::stoull(util::trim_ascii(*env));
                if (parsed >= 512ULL && parsed <= 65536ULL) {
                    return parsed;
                }
            } catch (...) {
            }
        }
        return kDefaultCudaMemLimitMb;
    }()};
    return s_limit_mb;
}

std::pair<uint64_t, std::string> query_primary_dxgi_gpu() {
    static const auto kStaticGpu = []() -> std::pair<uint64_t, std::string> {
        uint64_t best_vram_mb = 0;
        std::string best_name;
        IDXGIFactory1* factory = nullptr;
        if (SUCCEEDED(CreateDXGIFactory1(
                __uuidof(IDXGIFactory1),
                reinterpret_cast<void**>(&factory))) &&
            factory != nullptr) {
            IDXGIAdapter1* adapter = nullptr;
            for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
                DXGI_ADAPTER_DESC1 desc{};
                if (SUCCEEDED(adapter->GetDesc1(&desc))) {
                    if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
                        const uint64_t vram_mb =
                            static_cast<uint64_t>(desc.DedicatedVideoMemory) / (1024ULL * 1024ULL);
                        if (vram_mb > best_vram_mb || best_name.empty()) {
                            best_vram_mb = vram_mb;
                            best_name = util::wide_to_utf8(desc.Description);
                        }
                    }
                }
                adapter->Release();
            }
            factory->Release();
        }
        return {best_vram_mb, best_name};
    }();
    return kStaticGpu;
}

}  // namespace

bool probe_nvcuda_dll() noexcept {
    static const bool kHasNvCuda = []() noexcept -> bool {
        HMODULE mod = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (mod != nullptr) {
            FreeLibrary(mod);
            return true;
        }
        return false;
    }();
    return kHasNvCuda;
}

bool probe_cuda_backend_available(std::string* reason_out) noexcept {
    static const std::pair<bool, std::string> kCudaProbe = []() noexcept -> std::pair<bool, std::string> {
        std::vector<std::string> raw;
        try {
            raw = Ort::GetAvailableProviders();
        } catch (...) {
            return {false, "ONNX Runtime provider query failed"};
        }
        const bool ort_has_cuda =
            std::find(raw.begin(), raw.end(), "CUDAExecutionProvider") != raw.end();
        if (!ort_has_cuda) {
            return {false, "CUDA Runtime Not Installed (Universal DirectML build)"};
        }
        if (!probe_nvcuda_dll()) {
            return {false, "NVIDIA CUDA driver (nvcuda.dll) not found in System32"};
        }
        const bool has_provider_dll = probe_dll_on_search_path(L"onnxruntime_providers_cuda.dll");
        if (!has_provider_dll) {
            return {false, "onnxruntime_providers_cuda.dll not found"};
        }
        const bool has_cublas = probe_dll_on_search_path(L"cublas64_12.dll");
        const bool has_cudnn = probe_dll_on_search_path(L"cudnn64_9.dll");
        if (!has_cublas || !has_cudnn) {
            return {false, "CUDA 12 / cuDNN 9 runtime DLLs not installed"};
        }
        return {true, ""};
    }();
    if (reason_out != nullptr) {
        *reason_out = kCudaProbe.second;
    }
    return kCudaProbe.first;
}

std::string normalize_preferred_backend(const std::string& raw) {
    const std::string lower = util::to_lower_ascii(util::trim_ascii(raw));
    if (lower == "cuda" || lower == "nvidia") {
        return "cuda";
    }
    if (lower == "directml" || lower == "dml") {
        return "directml";
    }
    if (lower == "cpu" || lower == "cpu_only") {
        return "cpu";
    }
    return "auto";
}

std::string get_preferred_backend() {
    std::lock_guard<std::mutex> lock(preferred_backend_mutex());
    return preferred_backend_storage();
}

void set_preferred_backend(const std::string& backend) {
    const std::string normalized = normalize_preferred_backend(backend);
    std::lock_guard<std::mutex> lock(preferred_backend_mutex());
    preferred_backend_storage() = normalized;
}

uint64_t get_cuda_mem_limit_mb() noexcept {
    return cuda_mem_limit_mb_storage().load(std::memory_order_relaxed);
}

size_t get_cuda_mem_limit_bytes() noexcept {
    const uint64_t mb = get_cuda_mem_limit_mb();
    if (mb == 0ULL) {
        // Unlimited dynamic VRAM allocation (allows WDDM Shared GPU Memory to absorb peaks cleanly)
        return std::numeric_limits<size_t>::max();
    }
    return static_cast<size_t>(mb) * static_cast<size_t>(1024ULL * 1024ULL);
}

void set_cuda_mem_limit_mb(uint64_t limit_mb) noexcept {
    const uint64_t clamped = (limit_mb == 0ULL) ? 0ULL : std::clamp<uint64_t>(limit_mb, 512ULL, 65536ULL);
    cuda_mem_limit_mb_storage().store(clamped, std::memory_order_relaxed);
}

uint64_t query_free_vram_mb() noexcept {
    uint64_t best_total_mb = 0;
    uint64_t best_free_mb = 0;
    IDXGIFactory4* factory = nullptr;
    if (SUCCEEDED(CreateDXGIFactory1(
            __uuidof(IDXGIFactory4),
            reinterpret_cast<void**>(&factory))) &&
        factory != nullptr) {
        IDXGIAdapter1* adapter1 = nullptr;
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter1) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc{};
            if (SUCCEEDED(adapter1->GetDesc1(&desc)) &&
                (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0) {
                const uint64_t total_mb =
                    static_cast<uint64_t>(desc.DedicatedVideoMemory) / (1024ULL * 1024ULL);
                if (total_mb >= best_total_mb) {
                    IDXGIAdapter3* adapter3 = nullptr;
                    if (SUCCEEDED(adapter1->QueryInterface(
                            __uuidof(IDXGIAdapter3),
                            reinterpret_cast<void**>(&adapter3))) &&
                        adapter3 != nullptr) {
                        DXGI_QUERY_VIDEO_MEMORY_INFO mem_info{};
                        if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(
                                0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mem_info))) {
                            best_total_mb = total_mb;
                            const uint64_t avail_bytes =
                                (mem_info.Budget > mem_info.CurrentUsage)
                                    ? (mem_info.Budget - mem_info.CurrentUsage)
                                    : 0ULL;
                            best_free_mb = avail_bytes / (1024ULL * 1024ULL);
                        }
                        adapter3->Release();
                    }
                }
            }
            adapter1->Release();
        }
        factory->Release();
    }
    return best_free_mb;
}

bool is_vram_tight(uint64_t threshold_mb) noexcept {
    const auto gpu = query_primary_dxgi_gpu();
    if (gpu.first == 0ULL) {
        return false;  // No dedicated GPU
    }
    return query_free_vram_mb() < threshold_mb;
}

std::vector<std::string> get_safe_available_providers() {
    std::vector<std::string> raw;
    try {
        raw = Ort::GetAvailableProviders();
    } catch (...) {
        return {"CPUExecutionProvider"};
    }
    if (raw.empty()) {
        return {"CPUExecutionProvider"};
    }
    std::vector<std::string> filtered;
    filtered.reserve(raw.size());
    const bool cuda_ready = probe_cuda_backend_available(nullptr);
    for (const auto& ep : raw) {
        if (ep == "CUDAExecutionProvider" && !cuda_ready) {
            continue;
        }
        filtered.push_back(ep);
    }
    if (std::find(filtered.begin(), filtered.end(), "CPUExecutionProvider") == filtered.end()) {
        filtered.emplace_back("CPUExecutionProvider");
    }
    return filtered;
}

std::vector<std::string> get_ordered_execution_providers(bool force_cpu) {
    if (force_cpu || util::env_flag_enabled(L"MANGA_FORCE_CPU")) {
        return {"CPUExecutionProvider"};
    }
    const std::string pref = get_preferred_backend();
    if (pref == "cpu") {
        return {"CPUExecutionProvider"};
    }
    const auto available = get_safe_available_providers();
    const bool has_cuda =
        std::find(available.begin(), available.end(), "CUDAExecutionProvider") != available.end();
    const bool has_dml =
        std::find(available.begin(), available.end(), "DmlExecutionProvider") != available.end();

    std::vector<std::string> ordered;
    if (pref == "directml") {
        if (has_dml) {
            ordered.emplace_back("DmlExecutionProvider");
        }
        ordered.emplace_back("CPUExecutionProvider");
        return ordered;
    }
    // "auto" or "cuda" (if CUDA is unavailable, gracefully falls back to DirectML then CPU)
    if (has_cuda) {
        ordered.emplace_back("CUDAExecutionProvider");
    }
    if (has_dml) {
        ordered.emplace_back("DmlExecutionProvider");
    }
    ordered.emplace_back("CPUExecutionProvider");
    return ordered;
}

std::string get_primary_execution_provider(bool force_cpu) {
    const auto ordered = get_ordered_execution_providers(force_cpu);
    return ordered.empty() ? "CPUExecutionProvider" : ordered.front();
}

std::string backend_slug_from_provider(const std::string& execution_provider) {
    if (execution_provider == "CUDAExecutionProvider") {
        return "cuda";
    }
    if (execution_provider == "DmlExecutionProvider") {
        return "directml";
    }
    return "cpu";
}

bool append_gpu_execution_provider(
    Ort::SessionOptions& opts,
    const std::string& provider,
    int device_id,
    std::optional<size_t> cuda_mem_limit_bytes) noexcept {
    try {
        if (provider == "CUDAExecutionProvider") {
            if (!probe_cuda_backend_available(nullptr)) {
                return false;
            }
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = device_id;
            // HEURISTIC avoids long exhaustive cuDNN benchmarking & workspace OOM spikes
            cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
            // kSameAsRequested (1) prevents power-of-2 VRAM doubling so CUDA coexists with local LLMs
            cuda_opts.arena_extend_strategy = 1;
            cuda_opts.gpu_mem_limit = cuda_mem_limit_bytes.value_or(get_cuda_mem_limit_bytes());
            cuda_opts.do_copy_in_default_stream = 1;
            opts.AppendExecutionProvider_CUDA(cuda_opts);
            return true;
        }

        if (provider == "DmlExecutionProvider") {
            opts.DisableMemPattern();
            opts.DisableCpuMemArena();
            opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

            const void* raw_dml_api = nullptr;
            OrtStatus* api_status = Ort::GetApi().GetExecutionProviderApi(
                "DML", ORT_API_VERSION, &raw_dml_api);
            if (api_status != nullptr || raw_dml_api == nullptr) {
                if (api_status != nullptr) {
                    Ort::GetApi().ReleaseStatus(api_status);
                }
                return false;
            }
            const auto* dml_api = static_cast<const MinimalOrtDmlApi*>(raw_dml_api);
            OrtStatus* append_status =
                dml_api->SessionOptionsAppendExecutionProvider_DML(opts, device_id);
            if (append_status != nullptr) {
                Ort::GetApi().ReleaseStatus(append_status);
                return false;
            }
            return true;
        }
    } catch (...) {
        return false;
    }
    return false;
}

HardwareCapabilities detect_hardware_capabilities(bool force_cpu) {
    HardwareCapabilities caps;
    caps.cuda_driver_present = probe_nvcuda_dll();
    caps.cuda_available = probe_cuda_backend_available(&caps.cuda_unavailable_reason);
    caps.cuda_mem_limit_mb = get_cuda_mem_limit_mb();
    caps.available_providers = get_safe_available_providers();
    caps.directml_available =
        std::find(caps.available_providers.begin(), caps.available_providers.end(), "DmlExecutionProvider") !=
        caps.available_providers.end();

    // Dual-core runtime awareness: detect sibling CUDA core and DirectML core under `Vellum_Runtime/`
    // so both `CUDA` and `DirectML` buttons remain selectable when both cores are deployed.
    {
        std::error_code ec;
        const auto rt_dir = util::get_portable_root_dir() / L"Vellum_Runtime";
        const auto dml_dir = rt_dir / L"dml";
        if (!caps.directml_available) {
            const bool has_dml_core =
                std::filesystem::is_regular_file(dml_dir / L"Vellum_Core_DML.exe", ec) &&
                std::filesystem::is_regular_file(dml_dir / L"onnxruntime.dll", ec);
            if (has_dml_core) {
                caps.directml_available = true;
            }
        }
        if (!caps.cuda_available && caps.cuda_driver_present) {
            const bool has_cuda_core =
                (std::filesystem::is_regular_file(rt_dir / L"Vellum_Core.exe", ec) ||
                 std::filesystem::is_regular_file(rt_dir / L"Vellum.exe", ec)) &&
                std::filesystem::is_regular_file(rt_dir / L"onnxruntime_providers_cuda.dll", ec) &&
                std::filesystem::is_regular_file(rt_dir / L"cudnn64_9.dll", ec);
            if (has_cuda_core) {
                caps.cuda_available = true;
                caps.cuda_unavailable_reason.clear();
            }
        }
    }

    caps.cpu_available = true;
    caps.preferred_backend = get_preferred_backend();
    caps.execution_provider = get_primary_execution_provider(force_cpu);
    caps.backend = backend_slug_from_provider(caps.execution_provider);
    caps.active_backend = caps.backend;

    const auto gpu = query_primary_dxgi_gpu();
    caps.vram_mb = gpu.first;
    caps.free_vram_mb = query_free_vram_mb();
    caps.gpu_name = gpu.second;
    caps.logical_cpus = std::max(1u, std::thread::hardware_concurrency());

    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        caps.total_ram_mb = static_cast<uint64_t>(ms.ullTotalPhys) / (1024ULL * 1024ULL);
        caps.available_ram_mb = static_cast<uint64_t>(ms.ullAvailPhys) / (1024ULL * 1024ULL);
        caps.memory_load_percent = static_cast<uint32_t>(ms.dwMemoryLoad);
    }

    const bool low_ram = (caps.total_ram_mb > 0 && caps.total_ram_mb < 10240ULL);
    const bool low_vram = (caps.vram_mb > 0 && caps.vram_mb < 5120ULL);
    caps.low_memory_mode = low_ram || low_vram;

    // Hardware-aware defaults:
    // - CUDA / DirectML GPU: YOLO11m-Seg FP16 (8.5x faster, full v2.5 CV stack) + 3-Tier Inpaint ON
    // - CPU: YOLO11m-Seg INT8 + 3-Tier Inpaint ON
    if (caps.backend == "cuda" || caps.backend == "directml") {
        caps.recommended_detection_model = "vellum-det-yolo11m-seg-1024-fp16";
        caps.recommended_3tier_inpaint = true;
        caps.recommended_execution_mode = "hybrid";
    } else {
        caps.recommended_detection_model = "vellum-det-yolo11m-seg-1024-int8";
        caps.recommended_3tier_inpaint = true;
        caps.recommended_execution_mode = "cpu_only";
    }

    return caps;
}

nlohmann::json hardware_capabilities_to_json(const HardwareCapabilities& caps) {
    return {
        {"backend", caps.backend},
        {"active_backend", caps.active_backend},
        {"preferred_backend", caps.preferred_backend},
        {"execution_provider", caps.execution_provider},
        {"available_backends", {
            {"cuda", caps.cuda_available},
            {"directml", caps.directml_available},
            {"cpu", caps.cpu_available},
        }},
        {"cuda_unavailable_reason", caps.cuda_unavailable_reason},
        {"gpu_name", caps.gpu_name},
        {"vram_mb", caps.vram_mb},
        {"free_vram_mb", caps.free_vram_mb},
        {"dedicated_vram_mb", caps.vram_mb},
        {"total_ram_mb", caps.total_ram_mb},
        {"available_ram_mb", caps.available_ram_mb},
        {"memory_load_percent", caps.memory_load_percent},
        {"logical_cpus", caps.logical_cpus},
        {"cuda_driver_present", caps.cuda_driver_present},
        {"cuda_mem_limit_mb", caps.cuda_mem_limit_mb},
        {"low_memory_mode", caps.low_memory_mode},
        {"available_providers", caps.available_providers},
        {"recommended_defaults", {
            {"detection_model", caps.recommended_detection_model},
            {"enable_3tier_inpaint", caps.recommended_3tier_inpaint},
            {"execution_mode", caps.recommended_execution_mode},
        }},
    };
}

}  // namespace mangaflow::pipeline
