// GATE-CLAIM-OK: this file IS the free-memory measurement (an NVML read, plus vendor-neutral fallbacks); it makes no claim about
// any host's state.
#include "server-vram.h"
#include "server-vram-probe.h"

#include "log.h"

#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>

#if defined(_WIN32)
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <dxgi.h>
#   include <pdh.h>
#else
#   include <dlfcn.h>
#endif
#if defined(__APPLE__)
#   include <sys/sysctl.h>
#endif

// The NVML entry points used, with the ABI from nvml.h (stable since NVML 5; the _v2 names are the
// ones current drivers export). Declared here so no NVML header or import library is needed.
namespace {

typedef int nvml_return_t; // nvmlReturn_t, 0 = NVML_SUCCESS
typedef struct nvml_device_st * nvml_device_t;
struct nvml_memory_t {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};

typedef nvml_return_t (*fn_init_t)(void);
typedef nvml_return_t (*fn_count_t)(unsigned int *);
typedef nvml_return_t (*fn_handle_t)(unsigned int, nvml_device_t *);
typedef nvml_return_t (*fn_memory_t)(nvml_device_t, nvml_memory_t *);

struct nvml_api {
    bool        ok = false;
    fn_count_t  count  = nullptr;
    fn_handle_t handle = nullptr;
    fn_memory_t memory = nullptr;
};

nvml_api load_nvml() {
    nvml_api api;
#if defined(_WIN32)
    // the driver installs nvml.dll in System32; older drivers only in the NVSMI directory
    HMODULE lib = LoadLibraryW(L"nvml.dll");
    if (!lib) {
        lib = LoadLibraryW(L"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
    }
    if (!lib) {
        return api;
    }
    auto sym = [lib](const char * name) { return (void *) GetProcAddress(lib, name); };
#else
    void * lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        return api;
    }
    auto sym = [lib](const char * name) { return dlsym(lib, name); };
#endif
    auto init  = (fn_init_t)   sym("nvmlInit_v2");
    api.count  = (fn_count_t)  sym("nvmlDeviceGetCount_v2");
    api.handle = (fn_handle_t) sym("nvmlDeviceGetHandleByIndex_v2");
    api.memory = (fn_memory_t) sym("nvmlDeviceGetMemoryInfo");
    if (!init || !api.count || !api.handle || !api.memory) {
        return api;
    }
    api.ok = init() == 0; // NVML stays initialised for the life of the process
    return api;
}

const nvml_api & nvml() {
    static std::once_flag once;
    static nvml_api api;
    std::call_once(once, [] {
        api = load_nvml();
        if (api.ok) {
            LOG_INF("%s: NVML loaded, VRAM-aware model admission reads free device memory from it\n", __func__);
        } else {
            LOG_WRN("%s: NVML could not be loaded -- VRAM-aware model admission tries the vendor-neutral probes (DRM sysfs / DXGI+PDH / IOGPU), "
                    "and admits every load when they find nothing\n", __func__);
        }
    });
    return api;
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// System memory, for devices that have none of their own (NVML says NOT_SUPPORTED on unified-memory parts such as GB10)
// ---------------------------------------------------------------------------------------------------------------------------------------
struct sys_mem {
    int64_t total = -1;
    int64_t avail = -1;
};

sys_mem read_system_memory() {
    sys_mem m;
#if defined(_WIN32)
    MEMORYSTATUSEX st = {};
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) {
        m.total = (int64_t) st.ullTotalPhys;
        m.avail = (int64_t) st.ullAvailPhys;
    }
#elif defined(__linux__)
    std::ifstream in("/proc/meminfo");
    std::stringstream ss;
    ss << in.rdbuf();
    m.total = vram_probe::meminfo_bytes(ss.str(), "MemTotal");
    m.avail = vram_probe::meminfo_bytes(ss.str(), "MemAvailable");
#endif
    return m;
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// Vendor-neutral fallbacks, used ONLY when NVML gave no reading: a host with a working NVML keeps its NVIDIA-only number exactly as before.
// ---------------------------------------------------------------------------------------------------------------------------------------
#if defined(_WIN32)

typedef HRESULT (WINAPI * pfn_create_factory1_t)(REFIID, void **);
typedef PDH_STATUS (WINAPI * pfn_pdh_open_t)(LPCWSTR, DWORD_PTR, PDH_HQUERY *);
typedef PDH_STATUS (WINAPI * pfn_pdh_add_t)(PDH_HQUERY, LPCWSTR, DWORD_PTR, PDH_HCOUNTER *);
typedef PDH_STATUS (WINAPI * pfn_pdh_collect_t)(PDH_HQUERY);
typedef PDH_STATUS (WINAPI * pfn_pdh_fmt_t)(PDH_HCOUNTER, DWORD, LPDWORD, PPDH_FMT_COUNTERVALUE);
typedef PDH_STATUS (WINAPI * pfn_pdh_close_t)(PDH_HQUERY);

// One system-wide GPU Adapter Memory counter (bytes) for an adapter LUID; -1 when the counter does not exist or cannot be read.
int64_t pdh_adapter_bytes(const wchar_t * counter, uint32_t luid_hi, uint32_t luid_lo, int phys) {
    HMODULE pdh = LoadLibraryW(L"pdh.dll");
    if (!pdh) {
        return -1;
    }
    auto open    = (pfn_pdh_open_t)(void *) GetProcAddress(pdh, "PdhOpenQueryW");
    auto add     = (pfn_pdh_add_t)(void *) GetProcAddress(pdh, "PdhAddEnglishCounterW");
    auto collect = (pfn_pdh_collect_t)(void *) GetProcAddress(pdh, "PdhCollectQueryData");
    auto fmt     = (pfn_pdh_fmt_t)(void *) GetProcAddress(pdh, "PdhGetFormattedCounterValue");
    auto close   = (pfn_pdh_close_t)(void *) GetProcAddress(pdh, "PdhCloseQuery");
    if (!open || !add || !collect || !fmt || !close) {
        return -1;
    }
    wchar_t path[256];
    swprintf(path, sizeof(path) / sizeof(path[0]), L"\\GPU Adapter Memory(luid_0x%08X_0x%08X_phys_%d)\\%ls", luid_hi, luid_lo, phys, counter);
    PDH_HQUERY q = nullptr;
    int64_t out = -1;
    if (open(nullptr, 0, &q) == ERROR_SUCCESS) {
        PDH_HCOUNTER c = nullptr;
        if (add(q, path, 0, &c) == ERROR_SUCCESS && collect(q) == ERROR_SUCCESS) {
            PDH_FMT_COUNTERVALUE v = {};
            DWORD type = 0;
            if (fmt(c, PDH_FMT_LARGE, &type, &v) == ERROR_SUCCESS && v.CStatus == ERROR_SUCCESS) {
                out = (int64_t) v.largeValue;
            }
        }
        close(q);
    }
    return out;
}

// DXGI gives the capacity (DedicatedVideoMemory, plus SharedSystemMemory on a UMA part), PDH the SYSTEM-WIDE use. NOT IDXGIAdapter3::
// QueryVideoMemoryInfo: its CurrentUsage is this process only, which would make every device look empty to a router that holds no context.
std::vector<vram_probe::device> windows_devices(bool include_nvidia) {
    std::vector<vram_probe::device> out;
    HMODULE dxgi = LoadLibraryW(L"dxgi.dll");
    if (!dxgi) {
        return out;
    }
    auto create = (pfn_create_factory1_t)(void *) GetProcAddress(dxgi, "CreateDXGIFactory1");
    if (!create) {
        return out;
    }
    IDXGIFactory1 * factory = nullptr;
    if (FAILED(create(__uuidof(IDXGIFactory1), (void **) &factory)) || !factory) {
        return out;
    }
    for (UINT i = 0;; ++i) {
        IDXGIAdapter1 * ad = nullptr;
        if (factory->EnumAdapters1(i, &ad) != S_OK || !ad) {
            break;
        }
        DXGI_ADAPTER_DESC1 d = {};
        if (SUCCEEDED(ad->GetDesc1(&d))) {
            const bool software = (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0 || d.VendorId == 0x1414; // WARP / Microsoft Basic Render
            const bool skip_vendor = d.VendorId == 0x10DE && !include_nvidia;
            if (!software && !skip_vendor) {
                vram_probe::device dev;
                char id[64];
                std::snprintf(id, sizeof(id), "dxgi-luid-%08x-%08x", (unsigned) d.AdapterLuid.HighPart, (unsigned) d.AdapterLuid.LowPart);
                dev.id     = id;
                char vend[16];
                std::snprintf(vend, sizeof(vend), "0x%04x", (unsigned) d.VendorId);
                dev.vendor = vend;
                dev.total  = (int64_t) d.DedicatedVideoMemory;
                const int64_t dedicated_used = pdh_adapter_bytes(L"Dedicated Usage", (uint32_t) d.AdapterLuid.HighPart, d.AdapterLuid.LowPart, 0);
                int64_t used = dedicated_used;
                if (d.DedicatedVideoMemory < ((SIZE_T) 1 << 30)) { // carve-out only: a UMA part, the GPU also uses shared system memory
                    const int64_t shared_used = pdh_adapter_bytes(L"Shared Usage", (uint32_t) d.AdapterLuid.HighPart, d.AdapterLuid.LowPart, 0);
                    dev.total  += (int64_t) d.SharedSystemMemory;
                    dev.unified = true;
                    used = (dedicated_used >= 0 && shared_used >= 0) ? dedicated_used + shared_used : -1;
                }
                if (dev.total > 0 && used >= 0) { // no counter, no reading: never invent a usage
                    dev.used = used;
                    out.push_back(dev);
                }
            }
        }
        ad->Release();
    }
    factory->Release();
    return out;
}

#endif // _WIN32

#if defined(__APPLE__)
// DEVICE-PENDING (no Apple host here): IOGPU "In use system memory" for the use, hw.memsize bounded by iogpu.wired_limit_mb for the capacity.
std::vector<vram_probe::device> apple_devices() {
    std::vector<vram_probe::device> out;
    int64_t memsize = 0;
    size_t len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0) != 0) {
        return out;
    }
    int32_t wired_mb = 0;
    len = sizeof(wired_mb);
    if (sysctlbyname("iogpu.wired_limit_mb", &wired_mb, &len, nullptr, 0) != 0) {
        wired_mb = 0;
    }
    std::string text;
    if (FILE * p = popen("ioreg -r -c IOGPU -d 1 -f 2>/dev/null", "r")) {
        char buf[4096];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), p)) > 0) {
            text.append(buf, n);
        }
        pclose(p);
    }
    const int64_t used = vram_probe::parse_ioreg_in_use(text);
    const int64_t total = vram_probe::apple_gpu_budget(memsize, wired_mb);
    if (used >= 0 && total > 0) {
        vram_probe::device d;
        d.id = "apple-iogpu";
        d.vendor = "apple";
        d.total = total;
        d.used = used;
        d.unified = true;
        out.push_back(d);
    }
    return out;
}
#endif // __APPLE__

std::vector<vram_probe::device> fallback_devices(bool nvml_ok) {
    std::vector<vram_probe::device> out;
    // Linux DRM sysfs (amdgpu and any driver exposing mem_info_vram_*). The root can be named explicitly on any OS: the unit test feeds a
    // fixture tree, and an operator can point it at a mounted /sys of another namespace.
    const char * root_env = std::getenv("LLAMA_SERVER_VRAM_DRM_ROOT");
#if defined(__linux__)
    const std::string root = root_env ? root_env : "/sys/class/drm";
#else
    const std::string root = root_env ? root_env : "";
#endif
    if (!root.empty()) {
        for (auto & d : vram_probe::scan_drm(root)) {
            out.push_back(d);
        }
    }
#if defined(_WIN32)
    if (!root_env) {
        for (auto & d : windows_devices(!nvml_ok)) {
            out.push_back(d);
        }
    }
#endif
#if defined(__APPLE__)
    if (!root_env) {
        for (auto & d : apple_devices()) {
            out.push_back(d);
        }
    }
#endif
    (void) nvml_ok;
    return out;
}

// sum over devices of total or free; -1 if no probe gives a reading or any NVML query fails
int64_t sum_memory(bool free_only) {
    const nvml_api & api = nvml();
    const bool nvml_ok = api.ok && !std::getenv("LLAMA_SERVER_VRAM_NO_NVML");
    int64_t sum = 0;
    bool have_nvidia = false;
    if (nvml_ok) {
        unsigned int n = 0;
        if (api.count(&n) != 0) {
            return -1;
        }
        for (unsigned int i = 0; i < n; ++i) {
            nvml_device_t dev = nullptr;
            nvml_memory_t mem = {};
            if (api.handle(i, &dev) != 0) {
                return -1;
            }
            const nvml_return_t rc = api.memory(dev, &mem);
            if (rc == 0) {
                sum += (int64_t) (free_only ? mem.free : mem.total);
                have_nvidia = true;
            } else if (vram_probe::nvml_memory_is_unified(rc)) {
                // a device that uses system RAM (GB10 class): its memory IS the host's
                const sys_mem sm = read_system_memory();
                if (sm.total <= 0 || sm.avail < 0) {
                    return -1;
                }
                sum += free_only ? sm.avail : sm.total;
                have_nvidia = true;
            } else {
                return -1;
            }
        }
    }
    if (have_nvidia) {
        return sum; // an NVIDIA host keeps its NVIDIA-only number, byte-identical to before the fallbacks existed
    }
    const std::vector<vram_probe::device> devs = fallback_devices(nvml_ok);
    if (devs.empty()) {
        return -1;
    }
    for (const auto & d : devs) {
        sum += free_only ? d.free_bytes() : d.total;
    }
    return sum;
}

} // namespace

int64_t server_vram_free_bytes()  { return sum_memory(true);  }
int64_t server_vram_total_bytes() { return sum_memory(false); }
