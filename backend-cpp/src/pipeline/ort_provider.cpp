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

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <thread>

namespace mangaflow::pipeline {

namespace {

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

uint64_t get_cuda_mem_limit_mb() noexcept {
    return cuda_mem_limit_mb_storage().load(std::memory_order_relaxed);
}

size_t get_cuda_mem_limit_bytes() noexcept {
    const uint64_t mb = get_cuda_mem_limit_mb();
    return static_cast<size_t>(mb) * static_cast<size_t>(1024ULL * 1024ULL);
}

void set_cuda_mem_limit_mb(uint64_t limit_mb) noexcept {
    const uint64_t clamped = std::clamp<uint64_t>(limit_mb, 512ULL, 65536ULL);
    cuda_mem_limit_mb_storage().store(clamped, std::memory_order_relaxed);
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
    const bool has_cuda_driver = probe_nvcuda_dll();
    for (const auto& ep : raw) {
        if (ep == "CUDAExecutionProvider" && !has_cuda_driver) {
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
    const auto available = get_safe_available_providers();
    std::vector<std::string> ordered;
    for (const char* gpu_ep : {"CUDAExecutionProvider", "DmlExecutionProvider"}) {
        if (std::find(available.begin(), available.end(), gpu_ep) != available.end()) {
            ordered.emplace_back(gpu_ep);
        }
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
            if (!probe_nvcuda_dll()) {
                return false;
            }
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = device_id;
            // kSameAsRequested (1) prevents power-of-2 VRAM doubling so CUDA coexists with local LLMs
            cuda_opts.arena_extend_strategy = 1;
            cuda_opts.gpu_mem_limit = cuda_mem_limit_bytes.value_or(get_cuda_mem_limit_bytes());
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
    caps.cuda_mem_limit_mb = get_cuda_mem_limit_mb();
    caps.available_providers = get_safe_available_providers();
    caps.execution_provider = get_primary_execution_provider(force_cpu);
    caps.backend = backend_slug_from_provider(caps.execution_provider);

    const auto gpu = query_primary_dxgi_gpu();
    caps.vram_mb = gpu.first;
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
    // - CUDA >= 6 GB: RF-DETR / YOLO11m-Seg FP16
    // - DirectML (or CUDA < 6 GB): YOLO11m-Seg FP16
    // - CPU: YOLO11m-Seg INT8 + 3-Tier Inpaint ON
    if (caps.backend == "cuda" && caps.vram_mb >= 6144ULL) {
        caps.recommended_detection_model = "koharu-layout-rfdetr-seg-2xl";
        caps.recommended_3tier_inpaint = true;
        caps.recommended_execution_mode = "hybrid";
    } else if (caps.backend == "directml" || caps.backend == "cuda") {
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
        {"execution_provider", caps.execution_provider},
        {"gpu_name", caps.gpu_name},
        {"vram_mb", caps.vram_mb},
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
