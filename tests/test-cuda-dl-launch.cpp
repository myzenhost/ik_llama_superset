// ONE GPU-launch correctness check (the only sub-test in this row that touches the GPU). NOT
// registered as a CTest add_test (see tests/CMakeLists.txt) -- it allocates real device memory
// and must only run while the GPU is claimed via agent_dispatch.py --claim-gpu, per the resource
// discipline this row was assigned under. Run it by hand:
//   agent_dispatch.py --claim-gpu <slot> --pid <this pid> --note "cuda-dl axpy check"
//   test-cuda-dl-launch.exe
//   agent_dispatch.py --release-gpu <slot>
//
// NVRTC-compiles axpy<float> (same kernel as test-cuda-dl-nvrtc.cpp), launches it on 1024
// elements via the driver API (cuModuleLoadData + cuModuleGetFunction + cuLaunchKernel), and
// compares the device result to a CPU reference. Tiny allocation (2 * 1024 * 4 bytes = 8 KiB).

#include "ggml-cuda-dl/cuda_dl.h"
#include "ggml-cuda-dl/nvrtc_cache.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const char * kSource = R"(
template<typename T>
__global__ void axpy(int n, T a, const T* x, T* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { y[i] = a * x[i] + y[i]; }
}
)";

#define CU_CHECK(expr)                                                                         \
    do {                                                                                        \
        ggml_cuda_dl::CUresult _rc = (expr);                                                    \
        if (_rc != ggml_cuda_dl::CUDA_SUCCESS_DL) {                                              \
            std::fprintf(stderr, "FAIL %s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,           \
                         ggml_cuda_dl::cu_error_string(_rc).c_str());                            \
            return 1;                                                                            \
        }                                                                                        \
    } while (0)

int main() {
    // Unbuffered: a piped stdout is fully buffered, so a crash used to take every line printed before it
    // with it (2026-11-11: the 0xC0000005 below the cache hit printed NOTHING, hiding how far it got).
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const ggml_cuda_dl::LoadResult & load = ggml_cuda_dl::ensure_loaded();
    std::printf("test-cuda-dl-launch: %s\n", load.summary.c_str());
    if (!load.available) {
        // 77 = SKIP (the automake/CTest convention the gate's tools already read): no CUDA here is not a
        // failure of the code under test.
        std::fprintf(stderr, "test-cuda-dl-launch: SKIPPED -- CUDA not available, cannot run launch check\n");
        return 77;
    }
    const ggml_cuda_dl::DriverApi * drv = ggml_cuda_dl::driver();

    CU_CHECK(drv->cuInit(0));
    ggml_cuda_dl::CUdevice dev = 0;
    CU_CHECK(drv->cuDeviceGet(&dev, 0));
    int major = 0, minor = 0;
    drv->cuDeviceGetAttribute(&major, ggml_cuda_dl::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR_DL, dev);
    drv->cuDeviceGetAttribute(&minor, ggml_cuda_dl::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR_DL, dev);
    std::printf("test-cuda-dl-launch: device 0, sm_%d%d\n", major, minor);

    ggml_cuda_dl::CUcontext ctx = nullptr;
    CU_CHECK(drv->cuDevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK(drv->cuCtxSetCurrent(ctx));

    ggml_cuda_dl::CompileOptions opts;
    opts.arch_major = major;
    opts.arch_minor = minor;
    ggml_cuda_dl::CompileResult cr = ggml_cuda_dl::compile(kSource, "axpy_launch", { "axpy<float>" }, opts);
    if (!cr.ok) {
        std::fprintf(stderr, "FAIL: nvrtc compile: %s\n%s\n", cr.error.c_str(), cr.log.c_str());
        drv->cuDevicePrimaryCtxRelease(dev);
        return 1;
    }

    // compile() returns one lowered name per name expression; an empty vector or name used to be indexed blind.
    if (cr.lowered_names.size() != 1 || cr.lowered_names[0].empty()) {
        std::fprintf(stderr, "FAIL: compile() ok (from_cache=%d) but lowered name for axpy<float> missing (%zu names)\n",
                     (int) cr.from_cache, cr.lowered_names.size());
        drv->cuDevicePrimaryCtxRelease(dev);
        return 1;
    }
    std::printf("test-cuda-dl-launch: compiled (from_cache=%d), kernel %s\n", (int) cr.from_cache,
                cr.lowered_names[0].c_str());

    ggml_cuda_dl::CUmodule mod = nullptr;
    CU_CHECK(drv->cuModuleLoadData(&mod, cr.cubin.data()));
    ggml_cuda_dl::CUfunction fn = nullptr;
    CU_CHECK(drv->cuModuleGetFunction(&fn, mod, cr.lowered_names[0].c_str()));

    int n = 1024;
    float a = 2.5f;
    std::vector<float> hx(n), hy(n), hy_ref(n);
    for (int i = 0; i < n; ++i) {
        hx[i] = static_cast<float>(i) * 0.5f;
        hy[i] = static_cast<float>(n - i);
        hy_ref[i] = a * hx[i] + hy[i]; // CPU reference
    }

    ggml_cuda_dl::CUdeviceptr dx = 0, dy = 0;
    CU_CHECK(drv->cuMemAlloc(&dx, n * sizeof(float)));
    CU_CHECK(drv->cuMemAlloc(&dy, n * sizeof(float)));
    CU_CHECK(drv->cuMemcpyHtoD(dx, hx.data(), n * sizeof(float)));
    CU_CHECK(drv->cuMemcpyHtoD(dy, hy.data(), n * sizeof(float)));

    void * args[] = { (void *) &n, (void *) &a, (void *) &dx, (void *) &dy };
    const unsigned int block = 256;
    const unsigned int grid  = (n + block - 1) / block;
    CU_CHECK(drv->cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, nullptr, args, nullptr));
    CU_CHECK(drv->cuStreamSynchronize(nullptr));

    std::vector<float> result(n);
    CU_CHECK(drv->cuMemcpyDtoH(result.data(), dy, n * sizeof(float)));

    drv->cuMemFree(dx);
    drv->cuMemFree(dy);

    int failures = 0;
    double max_abs_err = 0.0;
    for (int i = 0; i < n; ++i) {
        double err = std::fabs(static_cast<double>(result[i]) - static_cast<double>(hy_ref[i]));
        max_abs_err = std::max(max_abs_err, err);
        if (err > 1e-4) {
            failures++;
        }
    }
    std::printf("test-cuda-dl-launch: max_abs_err=%g mismatches=%d/%d\n", max_abs_err, failures, n);

    drv->cuDevicePrimaryCtxRelease(dev);

    std::printf("test-cuda-dl-launch: %s\n", failures == 0 ? "ALL PASSED" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
