#include "cuda_dl.h"

#include <cstdlib>
#include <mutex>
#include <sstream>

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
using ggml_cuda_dl_handle_t = HMODULE;
#else
#   include <dlfcn.h>
using ggml_cuda_dl_handle_t = void *;
#endif

namespace ggml_cuda_dl {

namespace {

ggml_cuda_dl_handle_t dl_open(const std::string & path) {
#ifdef _WIN32
    // LOAD_LIBRARY_SEARCH_SYSTEM32 | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: search the system driver
    // store / default DLL search dirs, not the CWD -- matches how nvcuda.dll / nvrtc64_*.dll are
    // actually found on a real install (driver in system32, NVRTC beside the toolkit's bin on
    // PATH). No error dialog: LoadLibraryExW does not raise one for a missing DLL by default.
    return LoadLibraryExW(std::wstring(path.begin(), path.end()).c_str(), nullptr,
                           LOAD_LIBRARY_SEARCH_SYSTEM32 | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

#ifdef _WIN32
// Fully qualified path: also search the DLL's own directory for its dependencies
// (nvrtc64_130_0.dll sits next to nvrtc-builtins64_<ver>.dll). LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR is
// only valid with an absolute path, which is why dl_open() above cannot pass it for bare names.
ggml_cuda_dl_handle_t dl_open_full(const std::string & path) {
    return LoadLibraryExW(std::wstring(path.begin(), path.end()).c_str(), nullptr,
                          LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32 |
                              LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
}
#endif

void * dl_sym(ggml_cuda_dl_handle_t h, const char * name) {
#ifdef _WIN32
    return h ? reinterpret_cast<void *>(GetProcAddress(h, name)) : nullptr;
#else
    return h ? dlsym(h, name) : nullptr;
#endif
}

// cudarc lib.rs:186-225 get_lib_name_candidates() shape, ported: platform prefix/suffix, then a
// descending list of versioned names, narrowed to the majors this design targets (13/12/11 --
// the task's stated probe set) rather than cudarc's full historical list (9/10 included).
std::vector<std::string> driver_candidates() {
#ifdef _WIN32
    return { "nvcuda.dll" };
#else
    return { "libcuda.so.1", "libcuda.so" };
#endif
}

std::vector<std::string> nvrtc_candidates() {
    std::vector<std::string> out;
#ifdef _WIN32
    // nvrtc64_<major><minor>_0.dll (e.g. nvrtc64_120_0.dll for 12.0) and the major0 variant
    // (nvrtc64_130_0.dll for 13.x, cudarc issue #246) across majors 13/12/11.
    static const int majors[] = { 13, 12, 11 };
    static const int minors[] = { 0, 1, 2, 3, 4, 5, 6, 8 };
    for (int major : majors) {
        out.push_back("nvrtc64_" + std::to_string(major) + "0_0.dll");
        for (int minor : minors) {
            out.push_back("nvrtc64_" + std::to_string(major) + std::to_string(minor) + "_0.dll");
        }
    }
#else
    static const int majors[] = { 13, 12, 11 };
    for (int major : majors) {
        out.push_back("libnvrtc.so." + std::to_string(major));
    }
    out.push_back("libnvrtc.so");
#endif
    return out;
}

template <typename Fn>
void resolve(ggml_cuda_dl_handle_t h, const char * name, Fn * out) {
    *out = reinterpret_cast<Fn>(dl_sym(h, name));
}

struct LoadedState {
    LoadResult result;
    DriverApi  driver;
    NvrtcApi   nvrtc;
};

// Fills the driver function table. Missing individual symbols are left null (not treated as a
// hard load failure by themselves) -- ensure_loaded() only declares the driver "loaded" once the
// full set we depend on resolved, so a partial/odd build still reports a clean not-available.
bool fill_driver_table(ggml_cuda_dl_handle_t h, DriverApi & api) {
    resolve(h, "cuInit", &api.cuInit);
    resolve(h, "cuDeviceGet", &api.cuDeviceGet);
    resolve(h, "cuDeviceGetAttribute", &api.cuDeviceGetAttribute);
    resolve(h, "cuDevicePrimaryCtxRetain", &api.cuDevicePrimaryCtxRetain);
    resolve(h, "cuDevicePrimaryCtxRelease_v2", &api.cuDevicePrimaryCtxRelease);
    resolve(h, "cuCtxSetCurrent", &api.cuCtxSetCurrent);
    resolve(h, "cuModuleLoadData", &api.cuModuleLoadData);
    resolve(h, "cuModuleGetFunction", &api.cuModuleGetFunction);
    resolve(h, "cuLaunchKernel", &api.cuLaunchKernel);
    resolve(h, "cuMemAlloc_v2", &api.cuMemAlloc);
    resolve(h, "cuMemFree_v2", &api.cuMemFree);
    resolve(h, "cuMemcpyHtoD_v2", &api.cuMemcpyHtoD);
    resolve(h, "cuMemcpyDtoH_v2", &api.cuMemcpyDtoH);
    resolve(h, "cuStreamSynchronize", &api.cuStreamSynchronize);
    resolve(h, "cuGetErrorString", &api.cuGetErrorString);
    resolve(h, "cuDriverGetVersion", &api.cuDriverGetVersion);
    return api.cuInit && api.cuDeviceGet && api.cuDeviceGetAttribute && api.cuModuleLoadData &&
           api.cuModuleGetFunction && api.cuLaunchKernel && api.cuMemAlloc && api.cuMemFree &&
           api.cuMemcpyHtoD && api.cuMemcpyDtoH && api.cuStreamSynchronize && api.cuGetErrorString &&
           api.cuDevicePrimaryCtxRetain && api.cuDevicePrimaryCtxRelease && api.cuCtxSetCurrent &&
           api.cuDriverGetVersion;
}

bool fill_nvrtc_table(ggml_cuda_dl_handle_t h, NvrtcApi & api) {
    resolve(h, "nvrtcCreateProgram", &api.nvrtcCreateProgram);
    resolve(h, "nvrtcCompileProgram", &api.nvrtcCompileProgram);
    resolve(h, "nvrtcGetProgramLogSize", &api.nvrtcGetProgramLogSize);
    resolve(h, "nvrtcGetProgramLog", &api.nvrtcGetProgramLog);
    resolve(h, "nvrtcGetCUBINSize", &api.nvrtcGetCUBINSize);
    resolve(h, "nvrtcGetCUBIN", &api.nvrtcGetCUBIN);
    resolve(h, "nvrtcGetPTXSize", &api.nvrtcGetPTXSize);
    resolve(h, "nvrtcGetPTX", &api.nvrtcGetPTX);
    resolve(h, "nvrtcDestroyProgram", &api.nvrtcDestroyProgram);
    resolve(h, "nvrtcVersion", &api.nvrtcVersion);
    resolve(h, "nvrtcAddNameExpression", &api.nvrtcAddNameExpression);
    resolve(h, "nvrtcGetLoweredName", &api.nvrtcGetLoweredName);
    resolve(h, "nvrtcGetErrorString", &api.nvrtcGetErrorString);
    return api.nvrtcCreateProgram && api.nvrtcCompileProgram && api.nvrtcGetProgramLogSize &&
           api.nvrtcGetProgramLog && api.nvrtcGetCUBINSize && api.nvrtcGetCUBIN &&
           api.nvrtcGetPTXSize && api.nvrtcGetPTX && api.nvrtcDestroyProgram && api.nvrtcVersion &&
           api.nvrtcAddNameExpression && api.nvrtcGetLoweredName && api.nvrtcGetErrorString;
}

LoadedState do_load() {
    LoadedState st;

    // --- driver --- (GGML_CUDA_DL_DRIVER overrides the candidate list with exactly one path --
    // same mechanism as the NVRTC override below; used by the "bogus driver name" anti-bias test)
    std::vector<std::string> drv_names;
    if (const char * override_path = std::getenv("GGML_CUDA_DL_DRIVER")) {
        drv_names.push_back(override_path);
    } else {
        drv_names = driver_candidates();
    }
    ggml_cuda_dl_handle_t drv_handle = nullptr;
    for (const auto & name : drv_names) {
        st.result.driver_candidates_tried.push_back(name);
        ggml_cuda_dl_handle_t h = dl_open(name);
        if (h) {
            drv_handle = h;
            st.result.driver_path_loaded = name;
            break;
        }
    }
    bool driver_ok = drv_handle && fill_driver_table(drv_handle, st.driver);

    // --- nvrtc --- (env override takes exactly one path, still recorded as "tried")
    std::vector<std::string> nvrtc_names;
    if (const char * override_path = std::getenv("GGML_CUDA_DL_NVRTC")) {
        nvrtc_names.push_back(override_path);
    } else {
        nvrtc_names = nvrtc_candidates();
    }
    ggml_cuda_dl_handle_t nvrtc_handle = nullptr;
    for (const auto & name : nvrtc_names) {
        st.result.nvrtc_candidates_tried.push_back(name);
        ggml_cuda_dl_handle_t h = dl_open(name);
        if (h) {
            nvrtc_handle = h;
            st.result.nvrtc_path_loaded = name;
            break;
        }
    }
#ifdef _WIN32
    // dl_open's LOAD_LIBRARY_SEARCH_DEFAULT_DIRS does not search PATH, so a bare nvrtc64_*.dll name
    // only resolves when the DLL sits beside the exe. Measured 2026-10-15 (CUDA 13.2, RTX 5090): every
    // bare candidate failed while nvrtc64_130_0.dll was present in %CUDA_PATH%\bin\x64 -- CUDA 13 moved
    // the toolkit DLLs from bin\ to bin\x64\. Probe the installer's CUDA_PATH in both layouts. Skipped
    // when GGML_CUDA_DL_NVRTC is set: that override names exactly one path and must stay exact.
    const char * cuda_path = std::getenv("CUDA_PATH");
    if (!nvrtc_handle && cuda_path && *cuda_path && !std::getenv("GGML_CUDA_DL_NVRTC")) {
        for (const char * sub : { "\\bin\\x64\\", "\\bin\\" }) {
            for (const auto & name : nvrtc_candidates()) {
                const std::string full = std::string(cuda_path) + sub + name;
                if (GetFileAttributesA(full.c_str()) == INVALID_FILE_ATTRIBUTES) {
                    continue; // only real files are recorded as tried, so the summary stays readable
                }
                st.result.nvrtc_candidates_tried.push_back(full);
                ggml_cuda_dl_handle_t h = dl_open_full(full);
                if (h) {
                    nvrtc_handle = h;
                    st.result.nvrtc_path_loaded = full;
                    break;
                }
            }
            if (nvrtc_handle) {
                break;
            }
        }
    }
#endif
    bool nvrtc_ok = nvrtc_handle && fill_nvrtc_table(nvrtc_handle, st.nvrtc);

    st.result.available = driver_ok && nvrtc_ok;

    std::ostringstream summary;
    if (st.result.available) {
        summary << "ggml-cuda-dl: loaded driver=" << st.result.driver_path_loaded
                << " nvrtc=" << st.result.nvrtc_path_loaded;
    } else {
        summary << "ggml-cuda-dl: not available (driver "
                << (driver_ok ? "ok" : "FAILED") << ", nvrtc " << (nvrtc_ok ? "ok" : "FAILED")
                << "). driver candidates tried: [";
        for (size_t i = 0; i < st.result.driver_candidates_tried.size(); ++i) {
            summary << (i ? ", " : "") << st.result.driver_candidates_tried[i];
        }
        summary << "]; nvrtc candidates tried: [";
        for (size_t i = 0; i < st.result.nvrtc_candidates_tried.size(); ++i) {
            summary << (i ? ", " : "") << st.result.nvrtc_candidates_tried[i];
        }
        summary << "]";
    }
    st.result.summary = summary.str();
    return st;
}

LoadedState & state() {
    static LoadedState st = [] {
        // std::call_once-equivalent: function-local static init is itself thread-safe/once in
        // C++11+ (magic statics), which is what cudarc's OnceLock buys it too.
        return do_load();
    }();
    return st;
}

std::once_flag g_once;

} // namespace

const LoadResult & ensure_loaded() {
    std::call_once(g_once, [] { (void) state(); });
    return state().result;
}

const DriverApi * driver() {
    const LoadResult & r = ensure_loaded();
    return r.available ? &state().driver : nullptr;
}

const NvrtcApi * nvrtc() {
    const LoadResult & r = ensure_loaded();
    return r.available ? &state().nvrtc : nullptr;
}

std::string cu_error_string(CUresult r) {
    const DriverApi * d = driver();
    if (d && d->cuGetErrorString) {
        const char * s = nullptr;
        if (d->cuGetErrorString(r, &s) == CUDA_SUCCESS_DL && s) {
            return s;
        }
    }
    return "CUresult(" + std::to_string(r) + ")";
}

std::string nvrtc_error_string(nvrtcResult r) {
    const NvrtcApi * n = nvrtc();
    if (n && n->nvrtcGetErrorString) {
        const char * s = n->nvrtcGetErrorString(r);
        if (s) {
            return s;
        }
    }
    return "nvrtcResult(" + std::to_string(r) + ")";
}

} // namespace ggml_cuda_dl
