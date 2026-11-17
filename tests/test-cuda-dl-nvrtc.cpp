// Stage 2 test: real NVRTC compile of a tiny templated kernel -> cubin, WITHOUT launching (no GPU
// memory needed -- cuDeviceGetAttribute only, no context/alloc). Then exercises the disk cache:
// cold compile (miss), warm hit, and a corrupted entry being detected and recompiled rather than
// trusted. Separate process from test-cuda-dl.cpp / test-cuda-dl-bogus-names.cpp for its own
// call_once latch (see those files).
//
// This process overrides LOCALAPPDATA (Windows) / XDG_CACHE_HOME (else) to a PID-suffixed temp
// dir before the first cache_dir() call, so cache_dir()'s own %LOCALAPPDATA%/ggml/nvrtc-cache
// (see nvrtc_cache.cpp) lands somewhere hermetic instead of the real per-user cache.
//
// Also checks the CUDA driver version is part of the cache key: a pure cache_key() check (runs
// even without NVRTC) that two driver versions give different keys, and a live check that
// compile() keys on the value cuDriverGetVersion actually returns.
//
// If NVRTC is not available on this host (no toolkit / no driver), the live checks below are
// reported SKIPPED, not FAILED -- absence of the toolkit is not a bug in the loader.

#include "ggml-cuda-dl/cuda_dl.h"
#include "ggml-cuda-dl/nvrtc_cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#else
#   include <unistd.h>
#endif

static const char * kSource = R"(
template<typename T>
__global__ void axpy(int n, T a, const T* x, T* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { y[i] = a * x[i] + y[i]; }
}
)";

int main() {
    // Hermetic cache dir for THIS run: PID-suffixed so a second invocation of this binary (e.g.
    // ctest re-running it) starts cold again instead of hitting the previous run's leftover
    // cache entry -- measured 2026-09-28: a fixed (non-PID) temp dir made the SECOND run of this
    // binary see the first run's cache as a false "hit" on what should be a cold miss.
#ifdef _WIN32
    char tmp_dir[MAX_PATH];
    GetTempPathA(MAX_PATH, tmp_dir);
    std::string cache_override = std::string(tmp_dir) + "ggml_cuda_dl_test_cache_" +
                                  std::to_string(GetCurrentProcessId());
    _putenv_s("LOCALAPPDATA", cache_override.c_str()); // cache_dir() appends /ggml/nvrtc-cache
#else
    std::string cache_override = "/tmp/ggml_cuda_dl_test_cache_" + std::to_string(getpid());
    setenv("XDG_CACHE_HOME", cache_override.c_str(), 1);
#endif

#if defined(GGML_CUDA_DL_TEST_NVRTC_PATH) && defined(_WIN32)
    // Toolkit's NVRTC dir is not on LOAD_LIBRARY_SEARCH_DEFAULT_DIRS unless the caller adds it
    // (matches cudarc's own reliance on LD_LIBRARY_PATH/PATH) -- point at the path CMake found on
    // THIS host at configure time so the compile sub-tests below have something real to exercise,
    // unless the invoker already set GGML_CUDA_DL_NVRTC (env wins).
    if (!std::getenv("GGML_CUDA_DL_NVRTC")) {
        _putenv_s("GGML_CUDA_DL_NVRTC", GGML_CUDA_DL_TEST_NVRTC_PATH);
    }
#endif
    const ggml_cuda_dl::LoadResult & load = ggml_cuda_dl::ensure_loaded();
    std::printf("test-cuda-dl-nvrtc: loader summary: %s\n", load.summary.c_str());

    int failures = 0;

    // --- cache key must change with the driver version (no library needed: pure key function) ---
    // NVRTC docs sec. 11 "Caching": the NVVM IR -> PTX/SASS cache lives in the driver library, so
    // a driver-only upgrade can change the cubin; the key has to see it. Control: the same version
    // twice must give the SAME key, or the inequality below would prove nothing.
    {
        ggml_cuda_dl::CompileOptions ko;
        ko.arch_major = 12;
        ko.arch_minor = 0;
        const std::string ka = ggml_cuda_dl::cache_key(kSource, "axpy_test", { "axpy<float>" }, ko, 13020);
        const std::string kb = ggml_cuda_dl::cache_key(kSource, "axpy_test", { "axpy<float>" }, ko, 13030);
        const std::string ka2 = ggml_cuda_dl::cache_key(kSource, "axpy_test", { "axpy<float>" }, ko, 13020);
        if (ka == kb) {
            std::fprintf(stderr, "FAIL: cache_key identical for driver 13020 and 13030 -- driver version not in the key\n");
            failures++;
        }
        if (ka != ka2) {
            std::fprintf(stderr, "FAIL: cache_key not deterministic for the same driver version (control)\n");
            failures++;
        }
        std::printf("test-cuda-dl-nvrtc: driver-version key check: 13020 vs 13030 %s, 13020 vs 13020 %s\n",
                     ka == kb ? "SAME" : "differ", ka == ka2 ? "same" : "DIFFER");
    }

    if (!load.available) {
        std::printf("test-cuda-dl-nvrtc: NVRTC not available on this host -- live checks SKIPPED (not a failure)\n");
        std::printf("test-cuda-dl-nvrtc: %s\n", failures == 0 ? "ALL PASSED" : "FAILURES");
        return failures == 0 ? 0 : 1;
    }

    const ggml_cuda_dl::DriverApi * drv = ggml_cuda_dl::driver();

    // The real driver version, straight from the driver -- compile() must key on this same value.
    int real_driver_version = -1;
    if (drv->cuDriverGetVersion(&real_driver_version) != ggml_cuda_dl::CUDA_SUCCESS_DL || real_driver_version <= 0) {
        std::fprintf(stderr, "FAIL: cuDriverGetVersion did not return a positive version (got %d)\n",
                     real_driver_version);
        failures++;
    }
    std::printf("test-cuda-dl-nvrtc: cuDriverGetVersion = %d\n", real_driver_version);

    // cuInit + cuDeviceGet + cuDeviceGetAttribute: no context, no memory. This is the "WITHOUT
    // launching (no GPU memory needed)" query path.
    ggml_cuda_dl::CUresult rc = drv->cuInit(0);
    if (rc != ggml_cuda_dl::CUDA_SUCCESS_DL) {
        std::fprintf(stderr, "test-cuda-dl-nvrtc: cuInit failed: %s -- SKIPPED\n",
                     ggml_cuda_dl::cu_error_string(rc).c_str());
        return 0;
    }
    ggml_cuda_dl::CUdevice dev = 0;
    rc = drv->cuDeviceGet(&dev, 0);
    if (rc != ggml_cuda_dl::CUDA_SUCCESS_DL) {
        std::fprintf(stderr, "test-cuda-dl-nvrtc: cuDeviceGet failed: %s -- SKIPPED\n",
                     ggml_cuda_dl::cu_error_string(rc).c_str());
        return 0;
    }
    int major = 0, minor = 0;
    drv->cuDeviceGetAttribute(&major, ggml_cuda_dl::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR_DL, dev);
    drv->cuDeviceGetAttribute(&minor, ggml_cuda_dl::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR_DL, dev);
    std::printf("test-cuda-dl-nvrtc: device 0 compute capability sm_%d%d\n", major, minor);
    if (major == 0) {
        std::fprintf(stderr, "FAIL: compute capability major == 0 (attribute query broken)\n");
        failures++;
    }

    ggml_cuda_dl::CompileOptions opts;
    opts.arch_major = major;
    opts.arch_minor = minor;

    // --- cold compile: cache miss ---
    auto & stats = ggml_cuda_dl::cache_stats_for_test();
    uint64_t misses_before = stats.misses, hits_before = stats.hits;
    uint64_t dv_queries_before = stats.driver_version_queries;
    stats.last_driver_version = -12345; // sentinel: must be overwritten by compile()'s own query
    ggml_cuda_dl::CompileResult r1 = ggml_cuda_dl::compile(kSource, "axpy_test", { "axpy<float>" }, opts);
    if (stats.driver_version_queries <= dv_queries_before) {
        std::fprintf(stderr, "FAIL: compile() did not query the driver version\n");
        failures++;
    }
    if (stats.last_driver_version != real_driver_version) {
        std::fprintf(stderr, "FAIL: compile() keyed on driver version %d, driver reports %d\n",
                     stats.last_driver_version, real_driver_version);
        failures++;
    }
    // The key compile() used must be the one for the REAL version, and not the one for any other.
    if (ggml_cuda_dl::cache_key(kSource, "axpy_test", { "axpy<float>" }, opts) !=
            ggml_cuda_dl::cache_key(kSource, "axpy_test", { "axpy<float>" }, opts, real_driver_version) ||
        ggml_cuda_dl::cache_key(kSource, "axpy_test", { "axpy<float>" }, opts) ==
            ggml_cuda_dl::cache_key(kSource, "axpy_test", { "axpy<float>" }, opts, real_driver_version + 10)) {
        std::fprintf(stderr, "FAIL: default cache_key does not follow the real driver version\n");
        failures++;
    }
    if (!r1.ok) {
        std::fprintf(stderr, "FAIL: cold compile failed: %s\nlog:\n%s\n", r1.error.c_str(), r1.log.c_str());
        failures++;
    } else {
        if (r1.from_cache) {
            std::fprintf(stderr, "FAIL: cold compile reported from_cache==true\n");
            failures++;
        }
        if (r1.cubin.empty()) {
            std::fprintf(stderr, "FAIL: cold compile produced empty cubin\n");
            failures++;
        }
        if (r1.lowered_names.size() != 1 || r1.lowered_names[0].empty()) {
            std::fprintf(stderr, "FAIL: lowered name for axpy<float> not resolved\n");
            failures++;
        } else {
            std::printf("test-cuda-dl-nvrtc: lowered name = %s, cubin = %zu bytes\n",
                         r1.lowered_names[0].c_str(), r1.cubin.size());
        }
        if (stats.misses != misses_before + 1) {
            std::fprintf(stderr, "FAIL: miss counter did not advance on cold compile\n");
            failures++;
        }
    }

    // --- warm hit: identical inputs -> cache hit, no recompile ---
    ggml_cuda_dl::CompileResult r2 = ggml_cuda_dl::compile(kSource, "axpy_test", { "axpy<float>" }, opts);
    if (!r2.ok || !r2.from_cache) {
        std::fprintf(stderr, "FAIL: warm hit expected ok+from_cache, got ok=%d from_cache=%d\n",
                     r2.ok, r2.from_cache);
        failures++;
    }
    if (r1.ok && r2.ok && r1.cubin != r2.cubin) {
        std::fprintf(stderr, "FAIL: warm-hit cubin differs from cold-compile cubin\n");
        failures++;
    }
    // A hit must return the lowered names too (row S69-TEST-CUDA-DL-LAUNCH-ACCESS-VIOLATION-NO-OUTPUT-2026-11-11-01:
    // cache format 1 returned EMPTY lowered_names on every hit, and test-cuda-dl-launch crashed indexing [0]).
    if (r1.ok && r2.ok && r1.lowered_names != r2.lowered_names) {
        std::fprintf(stderr, "FAIL: warm-hit lowered_names (%zu) differ from cold-compile lowered_names (%zu)\n",
                     r2.lowered_names.size(), r1.lowered_names.size());
        failures++;
    }
    if (stats.hits != hits_before + 1) {
        std::fprintf(stderr, "FAIL: hit counter did not advance on warm hit\n");
        failures++;
    }
    std::printf("test-cuda-dl-nvrtc: cache hits=%llu misses=%llu corrupt_recompiles=%llu\n",
                (unsigned long long) stats.hits, (unsigned long long) stats.misses,
                (unsigned long long) stats.corrupt_recompiles);

    // --- corrupted entry: flip a byte in the cached file, recompile must not trust it ---
    std::string key = ggml_cuda_dl::cache_key(kSource, "axpy_test", { "axpy<float>" }, opts);
    std::string entry_path = ggml_cuda_dl::cache_dir() + "/" + key + ".cubin";
    {
        std::fstream f(entry_path, std::ios::binary | std::ios::in | std::ios::out);
        if (!f) {
            std::fprintf(stderr, "FAIL: could not reopen cache entry %s to corrupt it\n", entry_path.c_str());
            failures++;
        } else {
            f.seekp(20, std::ios::beg); // inside the stored digest -- guaranteed to change the hash
            char corrupt = '\xFF';
            f.write(&corrupt, 1);
        }
    }
    uint64_t corrupt_before = stats.corrupt_recompiles;
    ggml_cuda_dl::CompileResult r3 = ggml_cuda_dl::compile(kSource, "axpy_test", { "axpy<float>" }, opts);
    if (!r3.ok || r3.from_cache) {
        std::fprintf(stderr, "FAIL: corrupted entry expected ok+NOT from_cache (recompiled), got ok=%d from_cache=%d\n",
                     r3.ok, r3.from_cache);
        failures++;
    }
    if (stats.corrupt_recompiles != corrupt_before + 1) {
        std::fprintf(stderr, "FAIL: corrupt_recompiles counter did not advance\n");
        failures++;
    }

    std::printf("test-cuda-dl-nvrtc: %s\n", failures == 0 ? "ALL PASSED" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
