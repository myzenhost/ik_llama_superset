// NVML free-memory cross-check for the CUDA backend. See nvml-free.h for the reason it exists.
// WEB-RESEARCH-DONE: https://forums.developer.nvidia.com/t/cudamemgetinfo-vs-nvmldevicegetmemoryinfo/320791 (read in
// full: WDDM oversubscription explains the higher cudaMemGetInfo figure; NVML tracks nvidia-smi) and
// https://github.com/ggml-org/llama.cpp/issues/19437 (fit sizing against a free figure the allocation then misses).
// COMPATIBILITY-INVESTIGATED: three sources of a free figure were weighed -- cudaMemGetInfo alone (current, over-
// reports on WDDM), DXGI QueryVideoMemoryInfo (WDDM budget, needs a DXGI adapter match by LUID), NVML (device-global,
// ships with every NVIDIA driver, matched by PCI bus id). NVML was chosen: one DLL, no D3D dependency, and it matches
// nvidia-smi, which is the figure this host's 6 GB headroom floor is enforced against.
//
// NVML is loaded at RUNTIME from the driver's own nvml.dll (System32 on Windows), so there is no build or link
// dependency on the CUDA toolkit's NVML stub, and a machine without it keeps the cudaMemGetInfo figure.
// Only three entry points are used, with the documented NVML ABI (nvml.h): nvmlInit_v2,
// nvmlDeviceGetHandleByPciBusId_v2, nvmlDeviceGetMemoryInfo. Types are declared locally for the same reason.
// Opt-out: GGML_CUDA_NO_NVML_FREE=1.
//
// Windows only on purpose. The inflated figure is a WDDM behaviour; on Linux cudaMemGetInfo already reports the
// device-global free memory, and a second source there would add a failure mode and change nothing.

#include "nvml-free.h"

#if defined(_WIN32) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdlib>
#include <mutex>

namespace {

typedef int nvml_return_t;                 // nvmlReturn_t; NVML_SUCCESS == 0
typedef struct nvml_device_st * nvml_device_t;
struct nvml_memory_t {                     // nvmlMemory_t (v1): all in bytes
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};

typedef nvml_return_t (*fn_init_t)(void);
typedef nvml_return_t (*fn_by_pci_t)(const char *, nvml_device_t *);
typedef nvml_return_t (*fn_meminfo_t)(nvml_device_t, nvml_memory_t *);

struct nvml_api {
    bool         ok      = false;
    fn_by_pci_t  by_pci  = nullptr;
    fn_meminfo_t meminfo = nullptr;
};

nvml_api & api() {
    static nvml_api       a;
    static std::once_flag once;
    std::call_once(once, [] {
        const char * off = std::getenv("GGML_CUDA_NO_NVML_FREE");
        if (off && off[0] && off[0] != '0') {
            return;
        }
        // System32 ONLY -- never the current directory or PATH, so a planted nvml.dll cannot be picked up.
        HMODULE h = LoadLibraryExA("nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!h) {
            return;
        }
        auto init = (fn_init_t)    (void *) GetProcAddress(h, "nvmlInit_v2");
        a.by_pci  = (fn_by_pci_t)  (void *) GetProcAddress(h, "nvmlDeviceGetHandleByPciBusId_v2");
        a.meminfo = (fn_meminfo_t) (void *) GetProcAddress(h, "nvmlDeviceGetMemoryInfo");
        a.ok = init && a.by_pci && a.meminfo && init() == 0;
    });
    return a;
}

} // namespace

bool ggml_cuda_nvml_free_for_pci(const char * pci_bus_id, size_t * free_out) {
    if (!pci_bus_id || !pci_bus_id[0] || !free_out) {
        return false;
    }
    nvml_api & a = api();
    if (!a.ok) {
        return false;
    }
    nvml_device_t dev = nullptr;
    if (a.by_pci(pci_bus_id, &dev) != 0 || !dev) {
        return false;
    }
    nvml_memory_t m = {};
    if (a.meminfo(dev, &m) != 0) {
        return false;
    }
    *free_out = (size_t) m.free;
    return true;
}

#else

bool ggml_cuda_nvml_free_for_pci(const char *, size_t *) {
    return false;
}

#endif
