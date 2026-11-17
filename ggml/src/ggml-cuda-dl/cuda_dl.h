// ggml-cuda-dl: Ferrox/cudarc-style dynamic loading of the CUDA driver API + NVRTC.
//
// DESIGN: temp/research/findings/S69_FERROX_CUDARC_DYNAMIC_CUDA_DESIGN_2026-09-28.md
// SOURCE-READ (rule 1.87), cudarc 0.19.4 (MIT OR Apache-2.0), local cargo cache
// (upstream/cudarc-0.19.4/src),
// no code copied, design ported to C++:
//   - lib.rs:186-225 get_lib_name_candidates() -- versioned shared-library name list per lib
//     (DLL_PREFIX/DLL_SUFFIX, pointer width, major/minor/major0 variants).
//   - driver/sys/mod.rs:28287-28302 culib() -- std::sync::OnceLock<Lib>, tries "cuda"/"nvcuda"
//     candidates in order, first that loads wins; never aborts the process.
//   - nvrtc/sys/mod.rs:378-503 Lib struct -- one function pointer per NVRTC entry point, filled
//     by nvrtc/sys/mod.rs:505-520 Lib::new() = libloading::Library::new() + per-symbol get().
// Header/symbol names cross-checked against the installed toolkit (read, not included):
//   C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/include/cuda.h, nvrtc.h.
//   cuda.h:93-151 macro-redirects cuMemAlloc/cuMemFree/cuMemcpyHtoD/cuMemcpyDtoH/
//   cuDevicePrimaryCtxRelease to their _v2 symbols; cuInit, cuDeviceGet, cuDeviceGetAttribute,
//   cuDevicePrimaryCtxRetain, cuCtxSetCurrent, cuModuleLoadData (cuda.h:7555), cuModuleGetFunction
//   (cuda.h:7722), cuLaunchKernel, cuStreamSynchronize, cuGetErrorString are unsuffixed.
//   cuDriverGetVersion (cuda.h:5425, `CUresult CUDAAPI cuDriverGetVersion(int *driverVersion);`) is
//   unsuffixed too -- no macro redirect for it anywhere in cuda.h.
//
// Own typedefs only -- this header requires NO CUDA toolkit headers and NO CUDA toolkit installed
// to compile. Everything is resolved at RUN time via LoadLibraryExW (Windows) / dlopen (else).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml_cuda_dl {

// ---------------------------------------------------------------------------------------------
// Driver API: own opaque handle types (ABI-compatible with the real CUDA driver's handles, which
// are themselves opaque pointers / ints -- see cuda.h's typedef struct CUctx_st* CUcontext; etc.)
// ---------------------------------------------------------------------------------------------
using CUresult    = int;
using CUdevice    = int;
using CUdeviceptr = unsigned long long; // driver ABI is a 64-bit value on every platform we build for

struct CUctx_st;    using CUcontext  = CUctx_st *;
struct CUmod_st;    using CUmodule   = CUmod_st *;
struct CUfunc_st;   using CUfunction = CUfunc_st *;
struct CUstream_st; using CUstream   = CUstream_st *;

// CUdevice_attribute values we use (cuda.h enum CUdevice_attribute_enum), named _DL to make clear
// these are our own copies, not the toolkit's enum.
enum {
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR_DL = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR_DL = 76,
};

enum { CUDA_SUCCESS_DL = 0 };

// ---------------------------------------------------------------------------------------------
// NVRTC: own opaque handle types
// ---------------------------------------------------------------------------------------------
using nvrtcResult = int;
struct _nvrtcProgram; using nvrtcProgram = _nvrtcProgram *;
enum { NVRTC_SUCCESS_DL = 0 };

// ---------------------------------------------------------------------------------------------
// Function-pointer tables. One pointer per entry point we call; filled once by ensure_loaded().
// A null pointer means "not resolved" (either the library didn't load, or that particular symbol
// was missing in it) -- callers must check driver()/nvrtc() != nullptr before use, and individual
// members are non-null as a group once ensure_loaded().available is true.
// ---------------------------------------------------------------------------------------------
struct DriverApi {
    CUresult (*cuInit)(unsigned int Flags) = nullptr;
    CUresult (*cuDeviceGet)(CUdevice * device, int ordinal) = nullptr;
    CUresult (*cuDeviceGetAttribute)(int * pi, int attrib, CUdevice dev) = nullptr;
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext * pctx, CUdevice dev) = nullptr;
    CUresult (*cuDevicePrimaryCtxRelease)(CUdevice dev) = nullptr; // -> cuDevicePrimaryCtxRelease_v2
    CUresult (*cuCtxSetCurrent)(CUcontext ctx) = nullptr;
    CUresult (*cuModuleLoadData)(CUmodule * module, const void * image) = nullptr;
    CUresult (*cuModuleGetFunction)(CUfunction * hfunc, CUmodule hmod, const char * name) = nullptr;
    CUresult (*cuLaunchKernel)(CUfunction f,
                                unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                unsigned int sharedMemBytes, CUstream hStream,
                                void ** kernelParams, void ** extra) = nullptr;
    CUresult (*cuMemAlloc)(CUdeviceptr * dptr, size_t bytesize) = nullptr;         // -> _v2
    CUresult (*cuMemFree)(CUdeviceptr dptr) = nullptr;                             // -> _v2
    CUresult (*cuMemcpyHtoD)(CUdeviceptr dstDevice, const void * srcHost, size_t byteCount) = nullptr; // -> _v2
    CUresult (*cuMemcpyDtoH)(void * dstHost, CUdeviceptr srcDevice, size_t byteCount) = nullptr;       // -> _v2
    CUresult (*cuStreamSynchronize)(CUstream hStream) = nullptr;
    CUresult (*cuGetErrorString)(CUresult error, const char ** pStr) = nullptr;
    // Encoded as 1000*major + 10*minor (e.g. 13020 = 13.2), per the cuda.h:5403-5424 doc comment,
    // which lists no cuInit precondition. test-cuda-dl-nvrtc calls it BEFORE cuInit and asserts
    // success, so a driver that needed cuInit first would fail that test rather than pass silently.
    CUresult (*cuDriverGetVersion)(int * driverVersion) = nullptr;
};

struct NvrtcApi {
    nvrtcResult (*nvrtcCreateProgram)(nvrtcProgram * prog, const char * src, const char * name,
                                       int numHeaders, const char * const * headers,
                                       const char * const * includeNames) = nullptr;
    nvrtcResult (*nvrtcCompileProgram)(nvrtcProgram prog, int numOptions, const char * const * options) = nullptr;
    nvrtcResult (*nvrtcGetProgramLogSize)(nvrtcProgram prog, size_t * logSizeRet) = nullptr;
    nvrtcResult (*nvrtcGetProgramLog)(nvrtcProgram prog, char * log) = nullptr;
    nvrtcResult (*nvrtcGetCUBINSize)(nvrtcProgram prog, size_t * cubinSizeRet) = nullptr;
    nvrtcResult (*nvrtcGetCUBIN)(nvrtcProgram prog, char * cubin) = nullptr;
    nvrtcResult (*nvrtcGetPTXSize)(nvrtcProgram prog, size_t * ptxSizeRet) = nullptr;
    nvrtcResult (*nvrtcGetPTX)(nvrtcProgram prog, char * ptx) = nullptr;
    nvrtcResult (*nvrtcDestroyProgram)(nvrtcProgram * prog) = nullptr;
    nvrtcResult (*nvrtcVersion)(int * major, int * minor) = nullptr;
    nvrtcResult (*nvrtcAddNameExpression)(nvrtcProgram prog, const char * name_expression) = nullptr;
    nvrtcResult (*nvrtcGetLoweredName)(nvrtcProgram prog, const char * name_expression,
                                        const char ** lowered_name) = nullptr;
    const char * (*nvrtcGetErrorString)(nvrtcResult result) = nullptr;
};

struct LoadResult {
    bool available = false;
    std::vector<std::string> driver_candidates_tried;
    std::vector<std::string> nvrtc_candidates_tried;
    std::string driver_path_loaded;  // empty if driver did not load
    std::string nvrtc_path_loaded;   // empty if nvrtc did not load
    std::string summary;             // human-readable: what loaded / what was tried and failed
};

// Loads nvcuda + NVRTC exactly once (std::call_once). Candidate names follow cudarc's
// get_lib_name_candidates() shape (see file header). GGML_CUDA_DL_NVRTC / GGML_CUDA_DL_DRIVER env
// vars, if set, override the NVRTC / driver candidate list with that single path each (used by
// tests and as an operator escape hatch). Never throws, never calls abort()/exit() -- a missing
// or unloadable library yields available=false with every name tried recorded.
const LoadResult & ensure_loaded();

// nullptr unless ensure_loaded().available is true.
const DriverApi * driver();
const NvrtcApi  * nvrtc();

// Human-readable CUresult/nvrtcResult -> string, via the loaded library when available, else a
// generic "<code>" fallback so callers never need to null-check before logging an error.
std::string cu_error_string(CUresult r);
std::string nvrtc_error_string(nvrtcResult r);

} // namespace ggml_cuda_dl
