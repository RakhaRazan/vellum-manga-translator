#pragma once

// Vellum v2.5 - Dual-Backend ONNX Runtime Provider & Safe Hardware Probing
//
// Features:
// - Single codebase supporting DirectML (Universal default), CUDA, and CPU builds.
// - Safe `nvcuda.dll` probing via `LoadLibraryExW(..., LOAD_LIBRARY_SEARCH_SYSTEM32)`
//   before calling `AppendExecutionProvider_CUDA` so missing/old NVIDIA drivers
//   never crash and fall back seamlessly to `CPUExecutionProvider`.
// - `OrtCUDAProviderOptions` configured with `arena_extend_strategy = 1`
//   (`kSameAsRequested`) and configurable `gpu_mem_limit` (default 3.5 GB) so
//   vision models coexist with local LLMs on 8 GB GPUs.
// - Hardware capability detection (`backend`, `gpu_name`, `vram_mb`) and
//   hardware-aware default selection.

#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mangaflow::pipeline {

constexpr uint64_t kDefaultCudaMemLimitMb = 3584ULL;  // 3.5 GB default cap

struct HardwareCapabilities {
    std::string backend = "cpu";  // "cuda" | "directml" | "cpu"
    std::string execution_provider = "CPUExecutionProvider";
    std::string gpu_name;
    uint64_t vram_mb = 0;
    uint64_t total_ram_mb = 0;
    uint64_t available_ram_mb = 0;
    uint32_t memory_load_percent = 0;
    uint32_t logical_cpus = 1;
    bool cuda_driver_present = false;
    bool low_memory_mode = false;
    uint64_t cuda_mem_limit_mb = kDefaultCudaMemLimitMb;
    std::vector<std::string> available_providers;

    // Hardware-aware recommended defaults
    std::string recommended_detection_model = "vellum-det-yolo11m-seg-1024-int8";
    bool recommended_3tier_inpaint = true;
    std::string recommended_execution_mode = "hybrid";
};

// Safe Win32 probe for `nvcuda.dll` in System32. Never throws.
bool probe_nvcuda_dll() noexcept;

// Configurable CUDA VRAM cap (defaults to 3.5 GB or `VELLUM_CUDA_MEM_LIMIT_MB`).
uint64_t get_cuda_mem_limit_mb() noexcept;
size_t get_cuda_mem_limit_bytes() noexcept;
void set_cuda_mem_limit_mb(uint64_t limit_mb) noexcept;

// Ordered execution providers safe for this host (`CUDA` only if `nvcuda.dll` is present).
std::vector<std::string> get_safe_available_providers();
std::vector<std::string> get_ordered_execution_providers(bool force_cpu = false);
std::string get_primary_execution_provider(bool force_cpu = false);
std::string backend_slug_from_provider(const std::string& execution_provider);

// Safely appends CUDA or DirectML provider to `opts`. Returns false on any failure
// so caller can fall back to the next provider (ultimately CPUExecutionProvider).
bool append_gpu_execution_provider(
    Ort::SessionOptions& opts,
    const std::string& provider,
    int device_id = 0,
    std::optional<size_t> cuda_mem_limit_bytes = std::nullopt) noexcept;

// Query DXGI GPU adapter, system RAM, CPU threads, active ORT backend, and defaults.
HardwareCapabilities detect_hardware_capabilities(bool force_cpu = false);

// Format `HardwareCapabilities` for `GET /api/system/hardware`.
nlohmann::json hardware_capabilities_to_json(const HardwareCapabilities& caps);

}  // namespace mangaflow::pipeline
