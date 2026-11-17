#pragma once
// Failed-CUDA-init diagnostic: name the driver version, the toolkit (runtime) version and a PTX-JIT hint.
// Row S69-ENGREV-JAN-CUDA-DRIVER-TOOLKIT-DIAG-2026-10-27-01. Design from Jan (Apache-2.0) driver.rs / error.rs:132-144, no code copied.
// Header only and CUDA-free (versions come in as ints: 12040 = 12.4) so a CPU test can drive it.
#include <cstdio>

// Returns the number of characters written (snprintf semantics). driver_ver == 0 means "could not be read".
static inline int ggml_cuda_driver_diag(char * buf, size_t n, int err_code, const char * err_str, int driver_ver, int runtime_ver) {
    const int cudaErrorInsufficientDriver_ = 35;
    const int cudaErrorNoDevice_           = 100;
    const int cudaErrorStubLibrary_        = 34;
    const char * what = err_str ? err_str : "unknown error";
    if (driver_ver > 0 && runtime_ver > 0 && driver_ver < runtime_ver) {
        return snprintf(buf, n,
            "NVIDIA driver supports CUDA %d.%d but this build uses toolkit CUDA %d.%d (%s): update the NVIDIA driver, or run a build made with an "
            "older toolkit; embedded PTX can only be JIT-compiled by a driver at least as new as the toolkit that produced it",
            driver_ver / 1000, (driver_ver % 1000) / 10, runtime_ver / 1000, (runtime_ver % 1000) / 10, what);
    }
    if (err_code == cudaErrorInsufficientDriver_ || err_code == cudaErrorStubLibrary_) {
        return snprintf(buf, n, "NVIDIA driver is missing or too old for toolkit CUDA %d.%d (driver reports %s): install or update the NVIDIA driver (%s)",
            runtime_ver / 1000, (runtime_ver % 1000) / 10,
            driver_ver > 0 ? "a version below the toolkit" : "no CUDA version", what);
    }
    if (err_code == cudaErrorNoDevice_) {
        return snprintf(buf, n, "no CUDA device found (driver CUDA %d.%d, toolkit CUDA %d.%d): %s",
            driver_ver / 1000, (driver_ver % 1000) / 10, runtime_ver / 1000, (runtime_ver % 1000) / 10, what);
    }
    return snprintf(buf, n, "driver CUDA %d.%d, toolkit CUDA %d.%d: %s",
        driver_ver / 1000, (driver_ver % 1000) / 10, runtime_ver / 1000, (runtime_ver % 1000) / 10, what);
}
