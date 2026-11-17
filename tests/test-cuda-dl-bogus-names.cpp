// ANTI-BIAS test (per task): replace BOTH the driver and NVRTC candidate names with bogus paths
// and prove ensure_loaded() still returns a clean not-available result -- no crash, no abort(),
// no exception escaping main. Separate process from test-cuda-dl.cpp: ensure_loaded() is
// call_once-latched per process, so this needs its own fresh process to set its own override
// before the first call. See cuda_dl.h / cuda_dl.cpp for GGML_CUDA_DL_DRIVER / GGML_CUDA_DL_NVRTC.

#include "ggml-cuda-dl/cuda_dl.h"

#include <cstdio>
#include <cstdlib>

int main() {
#ifdef _WIN32
    _putenv_s("GGML_CUDA_DL_DRIVER", "Q:\\nope\\not_a_real_driver_9999.dll");
    _putenv_s("GGML_CUDA_DL_NVRTC",  "Q:\\nope\\not_a_real_nvrtc_9999.dll");
#else
    setenv("GGML_CUDA_DL_DRIVER", "/nope/not_a_real_driver.so.9999", 1);
    setenv("GGML_CUDA_DL_NVRTC",  "/nope/not_a_real_nvrtc.so.9999", 1);
#endif

    const ggml_cuda_dl::LoadResult & r = ggml_cuda_dl::ensure_loaded();

    int failures = 0;
    if (r.available) {
        std::fprintf(stderr, "FAIL: available==true with bogus driver+nvrtc names\n");
        failures++;
    }
    if (r.driver_candidates_tried.size() != 1 || r.nvrtc_candidates_tried.size() != 1) {
        std::fprintf(stderr, "FAIL: expected exactly 1 driver + 1 nvrtc candidate tried, got %zu + %zu\n",
                     r.driver_candidates_tried.size(), r.nvrtc_candidates_tried.size());
        failures++;
    }
    if (!r.driver_path_loaded.empty() || !r.nvrtc_path_loaded.empty()) {
        std::fprintf(stderr, "FAIL: a bogus path reported as loaded\n");
        failures++;
    }

    // Reaching this line at all (no SEH exception, no abort) is itself part of what this test
    // proves -- the process exiting cleanly with a normal return code is the crash-safety check.
    std::printf("test-cuda-dl-bogus-names: %s\n", r.summary.c_str());
    std::printf("test-cuda-dl-bogus-names: %s\n", failures == 0 ? "ALL PASSED" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
