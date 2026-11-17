// Stage 2: NVRTC compile (source -> cubin) with a disk cache, keyed by a hash of everything that
// can change the output. Design: S69_FERROX_CUDARC_DYNAMIC_CUDA_DESIGN_2026-09-28.md stage 2.
#pragma once

#include "cuda_dl.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ggml_cuda_dl {

struct CompileOptions {
    int arch_major = 0;
    int arch_minor = 0;
    std::vector<std::string> extra_options; // passed to nvrtcCompileProgram verbatim, after -arch=sm_XY
};

struct CompileResult {
    bool ok = false;
    bool from_cache = false;
    std::vector<char> cubin;
    std::string log;             // NVRTC compile log (present even on success if non-empty)
    std::string error;           // set when ok == false: why (nvrtc unavailable, compile error, ...)
    std::vector<std::string> lowered_names; // parallel to the name_expressions passed in, "" if unresolved
};

// The loaded CUDA driver's version via cuDriverGetVersion (1000*major + 10*minor), or -1 when the
// driver is not loaded or the call fails. Every call is counted in cache_stats_for_test().
int driver_version();

// Cache-key inputs: source text, program name, name expressions, arch, extra options, the loaded
// NVRTC's own version (nvrtcVersion), and the CUDA driver version. The driver version is in the
// key because NVRTC (CUDA 12.9+) caches NVVM-IR -> PTX/SASS translation IN the driver library
// (NVRTC docs sec. 11 "Caching": "The cache is implemented in the CUDA driver library
// (libcuda.so.1/nvcuda.dll)"), so a driver-only upgrade can change the cubin for identical source.
std::string cache_key(const std::string & source, const std::string & program_name,
                       const std::vector<std::string> & name_expressions, const CompileOptions & opts,
                       int drv_version);

// Same key, with the driver version taken from driver_version() -- this is the key compile() uses.
std::string cache_key(const std::string & source, const std::string & program_name,
                       const std::vector<std::string> & name_expressions, const CompileOptions & opts);

// Per-user cache directory: %LOCALAPPDATA%/ggml/nvrtc-cache on Windows,
// $XDG_CACHE_HOME/ggml/nvrtc-cache (or ~/.cache/ggml/nvrtc-cache) elsewhere. Created on first use.
std::string cache_dir();

// Compiles `source` (program name `program_name`, for logging/errors) via NVRTC, resolving each
// of `name_expressions` (nvrtcAddNameExpression / nvrtcGetLoweredName -- for templates) before
// compiling. Checks the disk cache first (cache_key()); a hit that fails integrity verification
// (truncated/corrupted file, header mismatch) is treated as a miss and recompiled, never trusted.
// On a fresh compile that succeeds, writes the cache entry via write-temp-then-rename so a reader
// never observes a partial file.
CompileResult compile(const std::string & source, const std::string & program_name,
                       const std::vector<std::string> & name_expressions, const CompileOptions & opts);

// Test/inspection hooks.
struct CacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t corrupt_recompiles = 0;
    uint64_t driver_version_queries = 0; // calls to driver_version()
    int last_driver_version = -1;        // what the last driver_version() call returned
};
CacheStats & cache_stats_for_test();

} // namespace ggml_cuda_dl
