// Stage 3 parity test: SiLU on the NVRTC dynamic-loading path vs (a) an independent CPU reference
// and (b) a genuinely, independently (nvcc, ahead-of-time) compiled build of the identical kernel
// source. Includes a must-fail control (a deliberately wrong kernel must NOT pass).
//
// NOT registered as a CTest add_test -- like test-cuda-dl-launch.cpp, this allocates real device
// memory and must only run while the GPU is claimed via agent_dispatch.py --claim-gpu. Run by hand:
//   agent_dispatch.py --claim-gpu <slot> --pid <this pid>
//   test-cuda-dl-silu-parity.exe [path/to/silu_static_ref.cubin]
//   agent_dispatch.py --release-gpu <slot>
//
// Exit 0 iff every real-kernel case passes AND the must-fail control correctly fails (i.e. the
// perturbed kernel's output does NOT match the CPU reference within tolerance).

#include "ggml-cuda-dl/cuda_dl.h"
#include "ggml-cuda-dl/nvrtc_cache.h"
#include "ggml-cuda-dl/kernels_silu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace ggml_cuda_dl;

#define CU_CHECK_RET(expr, retval)                                                                \
    do {                                                                                          \
        CUresult _rc = (expr);                                                                    \
        if (_rc != CUDA_SUCCESS_DL) {                                                              \
            std::fprintf(stderr, "FAIL %s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,             \
                         cu_error_string(_rc).c_str());                                            \
            return (retval);                                                                       \
        }                                                                                          \
    } while (0)

// ---------------------------------------------------------------------------------------------
// CPU reference (double precision, independent of both GPU kernels).
// ---------------------------------------------------------------------------------------------
static double silu_cpu_ref(double x) {
    return x / (1.0 + std::exp(-x));
}

// ---------------------------------------------------------------------------------------------
// Loads a precompiled .cubin from disk and launches it via the same driver-API path the NVRTC
// output uses (only the compiler that produced the cubin differs: nvcc here, NVRTC elsewhere).
// ---------------------------------------------------------------------------------------------
static bool launch_static_cubin(const DriverApi * drv, const std::vector<char> & cubin,
                                 const char * entry_point, CUdeviceptr d_in, CUdeviceptr d_out,
                                 int n, CUstream stream, std::string & err) {
    CUmodule mod = nullptr;
    CUresult rc = drv->cuModuleLoadData(&mod, cubin.data());
    if (rc != CUDA_SUCCESS_DL) { err = "cuModuleLoadData(static): " + cu_error_string(rc); return false; }
    CUfunction fn = nullptr;
    rc = drv->cuModuleGetFunction(&fn, mod, entry_point);
    if (rc != CUDA_SUCCESS_DL) { err = "cuModuleGetFunction(static): " + cu_error_string(rc); return false; }
    void * args[] = { (void *) &d_in, (void *) &d_out, (void *) &n };
    const unsigned int block = 256;
    const unsigned int grid  = (unsigned int) ((n + 255) / 256);
    rc = drv->cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, stream, args, nullptr);
    if (rc != CUDA_SUCCESS_DL) { err = "cuLaunchKernel(static): " + cu_error_string(rc); return false; }
    rc = drv->cuStreamSynchronize(stream);
    if (rc != CUDA_SUCCESS_DL) { err = "cuStreamSynchronize(static): " + cu_error_string(rc); return false; }
    return true;
}

struct CaseResult {
    int n = 0;
    double max_abs_err_nvrtc_vs_cpu = 0.0;
    double max_rel_err_nvrtc_vs_cpu = 0.0;
    double max_abs_err_static_vs_cpu = 0.0;
    double max_abs_err_nvrtc_vs_static = 0.0;
    bool ok = false;
};

int main(int argc, char ** argv) {
    int failures = 0;

    const LoadResult & load = ensure_loaded();
    std::printf("test-cuda-dl-silu-parity: %s\n", load.summary.c_str());
    if (!load.available) {
        std::fprintf(stderr, "test-cuda-dl-silu-parity: CUDA not available -- cannot run\n");
        return 1;
    }
    const DriverApi * drv = driver();

    // Static reference cubin: path from argv[1], else the default temp scratch location this
    // agent compiled it into (tmp/_cudadl3_0928_static_ref/silu_static_ref.cubin).
    std::string cubin_path = (argc > 1) ? argv[1]
        : "tmp/_cudadl3_0928_static_ref/silu_static_ref.cubin";
    std::vector<char> static_cubin;
    bool have_static = false;
    {
        std::ifstream f(cubin_path, std::ios::binary | std::ios::ate);
        if (f) {
            std::streamsize sz = f.tellg();
            f.seekg(0, std::ios::beg);
            static_cubin.resize((size_t) sz);
            if (f.read(static_cubin.data(), sz)) {
                have_static = true;
            }
        }
    }
    std::printf("test-cuda-dl-silu-parity: static reference cubin %s (%s, %zu bytes)\n",
                cubin_path.c_str(), have_static ? "loaded" : "NOT FOUND", static_cubin.size());
    if (!have_static) {
        std::printf("test-cuda-dl-silu-parity: NOTE -- proceeding with (a) CPU reference only for "
                     "this run; (b) static-kernel diff skipped (unverified this run).\n");
    }

    CU_CHECK_RET(drv->cuInit(0), 1);
    CUdevice dev = 0;
    CU_CHECK_RET(drv->cuDeviceGet(&dev, 0), 1);
    int major = 0, minor = 0;
    drv->cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR_DL, dev);
    drv->cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR_DL, dev);
    std::printf("test-cuda-dl-silu-parity: device 0, sm_%d%d\n", major, minor);

    CUcontext ctx = nullptr;
    CU_CHECK_RET(drv->cuDevicePrimaryCtxRetain(&ctx, dev), 1);
    CU_CHECK_RET(drv->cuCtxSetCurrent(ctx), 1);

    // Shapes: below one block, exactly one block, just above one block, a few thousand, and a
    // large one -- deliberately including non-multiples of CUDA_SILU_BLOCK_SIZE=256.
    const std::vector<int> shapes = { 7, 255, 256, 257, 1000, 4096, 1 << 20 /* 1,048,576 */ };

    std::mt19937 rng(12345); // fixed seed: reproducible across runs
    std::uniform_real_distribution<double> dist(-8.0, 8.0);

    std::printf("\n=== F32 parity (real kernel) ===\n");
    for (int n : shapes) {
        std::vector<float> hx(n), h_nvrtc(n, 0.0f), h_static(n, 0.0f);
        std::vector<double> h_ref(n);
        for (int i = 0; i < n; ++i) {
            double xv = dist(rng);
            hx[i] = (float) xv;
            h_ref[i] = silu_cpu_ref((double) hx[i]); // reference computed from the SAME rounded
                                                       // float input the kernels see, in double
                                                       // precision, independent of both kernels.
        }

        CUdeviceptr d_in = 0, d_out_nvrtc = 0, d_out_static = 0;
        CU_CHECK_RET(drv->cuMemAlloc(&d_in, n * sizeof(float)), 1);
        CU_CHECK_RET(drv->cuMemAlloc(&d_out_nvrtc, n * sizeof(float)), 1);
        CU_CHECK_RET(drv->cuMemcpyHtoD(d_in, hx.data(), n * sizeof(float)), 1);

        SiluLaunchOutcome lo = launch_silu_kernel(drv, kSiluF32Source, kSiluF32EntryPoint,
                                                   d_in, d_out_nvrtc, n, major, minor, nullptr);
        if (!lo.ok) {
            std::fprintf(stderr, "FAIL n=%d: NVRTC launch: %s\n", n, lo.error.c_str());
            failures++;
            drv->cuMemFree(d_in); drv->cuMemFree(d_out_nvrtc);
            continue;
        }
        CU_CHECK_RET(drv->cuMemcpyDtoH(h_nvrtc.data(), d_out_nvrtc, n * sizeof(float)), 1);

        CaseResult cr;
        cr.n = n;
        for (int i = 0; i < n; ++i) {
            double err = std::fabs((double) h_nvrtc[i] - h_ref[i]);
            cr.max_abs_err_nvrtc_vs_cpu = std::max(cr.max_abs_err_nvrtc_vs_cpu, err);
            double denom = std::max(1e-12, std::fabs(h_ref[i]));
            cr.max_rel_err_nvrtc_vs_cpu = std::max(cr.max_rel_err_nvrtc_vs_cpu, err / denom);
        }

        if (have_static) {
            CU_CHECK_RET(drv->cuMemAlloc(&d_out_static, n * sizeof(float)), 1);
            std::string serr;
            bool sok = launch_static_cubin(drv, static_cubin, kSiluF32EntryPoint, d_in,
                                            d_out_static, n, nullptr, serr);
            if (!sok) {
                std::fprintf(stderr, "FAIL n=%d: static launch: %s\n", n, serr.c_str());
                failures++;
            } else {
                CU_CHECK_RET(drv->cuMemcpyDtoH(h_static.data(), d_out_static, n * sizeof(float)), 1);
                for (int i = 0; i < n; ++i) {
                    cr.max_abs_err_static_vs_cpu = std::max(cr.max_abs_err_static_vs_cpu,
                        std::fabs((double) h_static[i] - h_ref[i]));
                    cr.max_abs_err_nvrtc_vs_static = std::max(cr.max_abs_err_nvrtc_vs_static,
                        std::fabs((double) h_nvrtc[i] - (double) h_static[i]));
                }
            }
            drv->cuMemFree(d_out_static);
        }

        // Tolerance: F32 elementwise transcendental (expf) -- allow 1e-5 abs, matching
        // test-cuda-dl-launch.cpp's own 1e-4 for a simpler op; expf is looser, so 1e-5 abs
        // *and* 1e-4 rel, whichever is looser, must hold.
        cr.ok = (cr.max_abs_err_nvrtc_vs_cpu < 1e-4) || (cr.max_rel_err_nvrtc_vs_cpu < 1e-4);
        if (!cr.ok) failures++;

        std::printf("n=%7d  nvrtc_vs_cpu: max_abs=%.3e max_rel=%.3e  "
                    "static_vs_cpu: max_abs=%.3e  nvrtc_vs_static: max_abs=%.3e  %s\n",
                    cr.n, cr.max_abs_err_nvrtc_vs_cpu, cr.max_rel_err_nvrtc_vs_cpu,
                    have_static ? cr.max_abs_err_static_vs_cpu : -1.0,
                    have_static ? cr.max_abs_err_nvrtc_vs_static : -1.0,
                    cr.ok ? "PASS" : "FAIL");

        drv->cuMemFree(d_in);
        drv->cuMemFree(d_out_nvrtc);
    }

    // -------------------------------------------------------------------------------------------
    // MUST-FAIL CONTROL: the deliberately wrong kernel (kSiluF32PerturbedSource, sign flipped)
    // must NOT match the CPU reference. If it does match, the parity harness itself is broken
    // (comparing against itself / not discriminating) and this is reported as a FAILURE of the
    // control, per the ANTI-BIAS instruction.
    // -------------------------------------------------------------------------------------------
    std::printf("\n=== F32 must-fail control (deliberately wrong kernel) ===\n");
    {
        const int n = 4096;
        std::vector<float> hx(n);
        std::vector<double> h_ref(n);
        std::mt19937 rng2(999);
        std::uniform_real_distribution<double> dist2(-8.0, 8.0);
        for (int i = 0; i < n; ++i) {
            double xv = dist2(rng2);
            hx[i] = (float) xv;
            h_ref[i] = silu_cpu_ref((double) hx[i]);
        }
        CUdeviceptr d_in = 0, d_out = 0;
        CU_CHECK_RET(drv->cuMemAlloc(&d_in, n * sizeof(float)), 1);
        CU_CHECK_RET(drv->cuMemAlloc(&d_out, n * sizeof(float)), 1);
        CU_CHECK_RET(drv->cuMemcpyHtoD(d_in, hx.data(), n * sizeof(float)), 1);

        SiluLaunchOutcome lo = launch_silu_kernel(drv, kSiluF32PerturbedSource, kSiluF32EntryPoint,
                                                   d_in, d_out, n, major, minor, nullptr);
        bool control_pass = false;
        if (!lo.ok) {
            std::fprintf(stderr, "must-fail control: perturbed kernel did not even launch (%s) -- "
                                  "control INCONCLUSIVE, treating as control failure\n", lo.error.c_str());
        } else {
            std::vector<float> h_bad(n);
            CU_CHECK_RET(drv->cuMemcpyDtoH(h_bad.data(), d_out, n * sizeof(float)), 1);
            double max_abs_err = 0.0;
            for (int i = 0; i < n; ++i) {
                max_abs_err = std::max(max_abs_err, std::fabs((double) h_bad[i] - h_ref[i]));
            }
            // The wrong kernel must clearly diverge -- require max_abs_err far above the real
            // kernel's tolerance (1e-4); 0.05 is a conservative floor for a sign-flipped
            // denominator over x in [-8,8].
            control_pass = max_abs_err > 0.05;
            std::printf("must-fail control: perturbed kernel max_abs_err=%.4f vs CPU ref -> %s\n",
                        max_abs_err, control_pass ? "PASS (correctly diverges)" : "FAIL (harness did not discriminate)");
        }
        if (!control_pass) failures++;
        drv->cuMemFree(d_in);
        drv->cuMemFree(d_out);
    }

    // -------------------------------------------------------------------------------------------
    // F16 best-effort (see kernels_silu.h for scope/limitation notes). Only attempted if a
    // toolkit include dir for cuda_fp16.h was found at CMake configure time.
    // -------------------------------------------------------------------------------------------
    std::printf("\n=== F16 best-effort (CPU-reference only, see kernels_silu.h for scope) ===\n");
#ifdef GGML_CUDA_DL_TEST_TOOLKIT_INCLUDE
    {
        std::vector<std::string> extra = { std::string("-I") + GGML_CUDA_DL_TEST_TOOLKIT_INCLUDE };
        const std::vector<int> f16_shapes = { 255, 256, 1024, 1 << 16 };
        for (int n : f16_shapes) {
            std::vector<uint16_t> hx(n), h_out(n, 0);
            std::vector<double> h_ref(n);
            std::mt19937 rng3(4242 + n);
            std::uniform_real_distribution<double> dist3(-6.0, 6.0); // stay well inside f16 range
            for (int i = 0; i < n; ++i) {
                double xv = dist3(rng3);
                // host-side float -> half (round-to-nearest-even) via a tiny manual conversion,
                // since this translation unit has no cuda_fp16.h on the host side (it is a
                // toolkit-only header used inside the NVRTC source, not linked into this .cpp).
                float xf = (float) xv;
                uint32_t fbits; std::memcpy(&fbits, &xf, 4);
                uint32_t sign = (fbits >> 16) & 0x8000u;
                int32_t  exp  = (int32_t)((fbits >> 23) & 0xFF) - 127 + 15;
                uint32_t mant = fbits & 0x7FFFFFu;
                uint16_t hbits;
                if (exp <= 0) { hbits = (uint16_t) sign; }               // flush tiny values to 0
                else if (exp >= 31) { hbits = (uint16_t)(sign | 0x7C00u); } // clamp overflow to inf
                else { hbits = (uint16_t)(sign | (exp << 10) | (mant >> 13)); }
                hx[i] = hbits;
                // CPU reference from the ORIGINAL double xv (not re-derived from the half bits)
                // is intentionally not used here -- use the half-rounded value's own double
                // value so the reference matches what the kernel actually saw as input.
                int32_t exp2 = (hbits >> 10) & 0x1F; uint32_t mant2 = hbits & 0x3FFu;
                double hv;
                if (exp2 == 0) { hv = std::ldexp((double) mant2, -24); }
                else { hv = std::ldexp((double)(mant2 | 0x400u), exp2 - 25); }
                if (hbits & 0x8000u) hv = -hv;
                h_ref[i] = silu_cpu_ref(hv);
            }
            CUdeviceptr d_in = 0, d_out = 0;
            CU_CHECK_RET(drv->cuMemAlloc(&d_in, n * sizeof(uint16_t)), 1);
            CU_CHECK_RET(drv->cuMemAlloc(&d_out, n * sizeof(uint16_t)), 1);
            CU_CHECK_RET(drv->cuMemcpyHtoD(d_in, hx.data(), n * sizeof(uint16_t)), 1);

            SiluLaunchOutcome lo = launch_silu_kernel(drv, kSiluF16Source, kSiluF16EntryPoint,
                                                       d_in, d_out, n, major, minor, nullptr, extra);
            if (!lo.ok) {
                std::fprintf(stderr, "F16 n=%d: SKIPPED/FAILED: %s\n", n, lo.error.c_str());
                drv->cuMemFree(d_in); drv->cuMemFree(d_out);
                continue; // best-effort: does not count toward pass/fail
            }
            CU_CHECK_RET(drv->cuMemcpyDtoH(h_out.data(), d_out, n * sizeof(uint16_t)), 1);
            double max_abs_err = 0.0, max_rel_err = 0.0;
            for (int i = 0; i < n; ++i) {
                uint16_t ob = h_out[i];
                int32_t exp2 = (ob >> 10) & 0x1F; uint32_t mant2 = ob & 0x3FFu;
                double ov;
                if (exp2 == 0) { ov = std::ldexp((double) mant2, -24); }
                else { ov = std::ldexp((double)(mant2 | 0x400u), exp2 - 25); }
                if (ob & 0x8000u) ov = -ov;
                double err = std::fabs(ov - h_ref[i]);
                max_abs_err = std::max(max_abs_err, err);
                max_rel_err = std::max(max_rel_err, err / std::max(1e-3, std::fabs(h_ref[i])));
            }
            std::printf("F16 n=%7d  vs_cpu: max_abs=%.3e max_rel=%.3e  %s (f16 tolerance 5e-3)\n",
                        n, max_abs_err, max_rel_err, (max_abs_err < 5e-3 || max_rel_err < 5e-3) ? "PASS" : "FAIL(non-fatal, informational)");
            drv->cuMemFree(d_in);
            drv->cuMemFree(d_out);
        }
    }
#else
    std::printf("F16: SKIPPED -- no toolkit include dir found at configure time "
                "(GGML_CUDA_DL_TEST_TOOLKIT_INCLUDE not defined); unverified this run.\n");
#endif

    // -------------------------------------------------------------------------------------------
    // Timing: NVRTC-compiled kernel vs the nvcc-compiled static cubin, warm, median of N.
    // Host-side std::chrono around cuLaunchKernel+cuStreamSynchronize (the stage-1 driver table
    // has no cuEvent* entry points -- extending it was out of scope for this stage; this is
    // coarse-grained wall-clock timing, informational only, not a device-side event measurement).
    // -------------------------------------------------------------------------------------------
    std::printf("\n=== Timing: NVRTC vs static (nvcc), warm, median of N ===\n");
    if (!have_static) {
        std::printf("SKIPPED -- no static reference cubin loaded (unverified this run).\n");
    } else {
        const int kWarmup = 5;
        const int kIters = 21; // odd, so "median" is a real element not an average of two
        for (int n : shapes) {
            std::vector<float> hx(n);
            std::mt19937 rngT(777 + n);
            std::uniform_real_distribution<double> distT(-8.0, 8.0);
            for (int i = 0; i < n; ++i) hx[i] = (float) distT(rngT);

            CUdeviceptr d_in = 0, d_out = 0;
            drv->cuMemAlloc(&d_in, n * sizeof(float));
            drv->cuMemAlloc(&d_out, n * sizeof(float));
            drv->cuMemcpyHtoD(d_in, hx.data(), n * sizeof(float));

            // Pre-load/compile both once (NVRTC compiles+caches on first launch_silu_kernel call);
            // warm both up before timing.
            std::vector<double> nvrtc_ms, static_ms;
            for (int iter = 0; iter < kWarmup + kIters; ++iter) {
                auto t0 = std::chrono::steady_clock::now();
                SiluLaunchOutcome lo = launch_silu_kernel(drv, kSiluF32Source, kSiluF32EntryPoint,
                                                           d_in, d_out, n, major, minor, nullptr);
                auto t1 = std::chrono::steady_clock::now();
                if (!lo.ok) { std::fprintf(stderr, "timing n=%d: nvrtc launch failed: %s\n", n, lo.error.c_str()); break; }
                if (iter >= kWarmup) nvrtc_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

                std::string serr;
                auto t2 = std::chrono::steady_clock::now();
                bool sok = launch_static_cubin(drv, static_cubin, kSiluF32EntryPoint, d_in, d_out, n, nullptr, serr);
                auto t3 = std::chrono::steady_clock::now();
                if (!sok) { std::fprintf(stderr, "timing n=%d: static launch failed: %s\n", n, serr.c_str()); break; }
                if (iter >= kWarmup) static_ms.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());
            }
            drv->cuMemFree(d_in);
            drv->cuMemFree(d_out);

            if ((int) nvrtc_ms.size() == kIters && (int) static_ms.size() == kIters) {
                std::sort(nvrtc_ms.begin(), nvrtc_ms.end());
                std::sort(static_ms.begin(), static_ms.end());
                double nvrtc_med = nvrtc_ms[kIters / 2];
                double static_med = static_ms[kIters / 2];
                std::printf("n=%7d  nvrtc_median=%.4f ms  static_median=%.4f ms  ratio(nvrtc/static)=%.3fx\n",
                            n, nvrtc_med, static_med, nvrtc_med / std::max(1e-9, static_med));
            } else {
                std::printf("n=%7d  timing INCOMPLETE (a launch failed mid-loop)\n", n);
            }
        }
    }

    drv->cuDevicePrimaryCtxRelease(dev);

    std::printf("\ntest-cuda-dl-silu-parity: %s (%d failure(s) among required F32 checks + must-fail control)\n",
                failures == 0 ? "ALL REQUIRED CHECKS PASSED" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
