// Tests for ggml-cuda-dl (Ferrox/cudarc-style dynamic CUDA loading), stages 1-2.
// Design: temp/research/findings/S69_FERROX_CUDARC_DYNAMIC_CUDA_DESIGN_2026-09-28.md
//
// Deliberately links ONLY ggml-cuda-dl (see tests/CMakeLists.txt) -- not `llama`/`llama-common`,
// which do not compile on this tree's MinGW toolchain (see the note above
// test-ggml-autodiff-probe in tests/CMakeLists.txt). A bare add_executable + this file is the
// whole point, same pattern as test-ggml-autodiff-probe / test-iqk-golden.
//
// No GPU is touched by this file itself -- the ONE launch-correctness sub-test lives in
// test-cuda-dl-launch.cpp (separate binary, run only under a claimed GPU slot; see that file's
// header and the agent ledger for the claim/release transcript).

#include "ggml-cuda-dl/cuda_dl.h"
#include "ggml-cuda-dl/nvrtc_cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CHECK(cond)                                                               \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
            g_failures++;                                                        \
        }                                                                         \
    } while (0)

static int g_failures = 0;

// ---------------------------------------------------------------------------------------------
// Test 1: loader with a bogus override -> clean not-available, never a crash, list printed.
// (This process's env already has GGML_CUDA_DL_NVRTC set to a nonexistent path by the harness
// script before exec, OR we set it here via _putenv/setenv before the first ensure_loaded() call
// -- ensure_loaded() is call_once, so the override must be set before any earlier test in this
// binary touches it. This file therefore does the override sub-test FIRST.)
// ---------------------------------------------------------------------------------------------
static void test_loader_not_available_override() {
#ifdef _WIN32
    _putenv_s("GGML_CUDA_DL_NVRTC", "Z:\\definitely\\not\\a\\real\\path\\nvrtc_bogus_9999.dll");
#else
    setenv("GGML_CUDA_DL_NVRTC", "/definitely/not/a/real/path/libnvrtc_bogus.so.9999", 1);
#endif
    const ggml_cuda_dl::LoadResult & r = ggml_cuda_dl::ensure_loaded();
    // The override path must have been recorded as tried, and NVRTC must have failed to load from
    // it -- available may still be true only if a real system NVRTC ALSO happens to be reachable
    // through some other candidate, which cannot happen here because the override REPLACES the
    // candidate list (see cuda_dl.cpp do_load()), so with a bogus override nvrtc_ok is false and
    // therefore available is false regardless of the driver.
    CHECK(r.nvrtc_candidates_tried.size() == 1);
    CHECK(r.nvrtc_path_loaded.empty());
    CHECK(r.available == false);
    CHECK(!r.summary.empty());
    std::printf("test_loader_not_available_override: %s\n", r.summary.c_str());
}

int main() {
    test_loader_not_available_override();

    // ---------------------------------------------------------------------------------------
    // NOTE ON TEST ORDERING: ensure_loaded() is call_once-latched at process scope (cuda_dl.cpp),
    // so once test 1 above has forced a bogus NVRTC override, EVERY later call in THIS process
    // sees the same "not available" result -- there is no way to re-probe the real system NVRTC
    // in the same binary after that. That is why the bogus-name-candidates sub-test (second half
    // of the ANTI-BIAS instruction: "a second test with the driver name candidates replaced by
    // bogus names") and the real cold/warm NVRTC-compile-and-cache sub-tests run as SEPARATE
    // PROCESSES (test-cuda-dl-bogus-names.cpp, test-cuda-dl-nvrtc.cpp) rather than as more
    // functions in this main() -- each needs its own fresh call_once latch. Registered as
    // separate CTest entries in tests/CMakeLists.txt.
    // ---------------------------------------------------------------------------------------

    if (g_failures == 0) {
        std::printf("test-cuda-dl: ALL PASSED\n");
    } else {
        std::fprintf(stderr, "test-cuda-dl: %d FAILURE(S)\n", g_failures);
    }
    return g_failures == 0 ? 0 : 1;
}
