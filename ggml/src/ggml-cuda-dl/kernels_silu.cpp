#include "kernels_silu.h"

namespace ggml_cuda_dl {

// RESEARCH-DONE: ggml/src/ggml-cuda/unary.cu:36-43 read in full --
//   static __global__ void silu_f32(const float * x, float * dst, const int k) {
//       const int i = blockDim.x*blockIdx.x + threadIdx.x;
//       if (i >= k) { return; }
//       dst[i] = x[i] / (1.0f + expf(-x[i]));
//   }
// reproduced below verbatim (own project, MIT) as an NVRTC source string, `extern "C"` added
// (no effect on the math, only on symbol naming) and renamed ggml_cuda_dl_silu_f32 to make clear
// in any log/profiler this is the NVRTC-path copy, not the nvcc-compiled original.
const char * kSiluF32Source = R"(
extern "C" __global__ void ggml_cuda_dl_silu_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) { return; }
    dst[i] = x[i] / (1.0f + expf(-x[i]));
}
)";
const char * kSiluF32EntryPoint = "ggml_cuda_dl_silu_f32";

// Deliberately wrong (must-fail control only -- see kernels_silu.h).
const char * kSiluF32PerturbedSource = R"(
extern "C" __global__ void ggml_cuda_dl_silu_f32(const float * x, float * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) { return; }
    dst[i] = x[i] / (1.0f - expf(-x[i]));
}
)";

// F16: computes in F32, stores F16. See header for why this is not asserted equal to the
// upstream templated F16 path.
const char * kSiluF16Source = R"(
#include <cuda_fp16.h>
extern "C" __global__ void ggml_cuda_dl_silu_f16(const __half * x, __half * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) { return; }
    float xi = __half2float(x[i]);
    float yi = xi / (1.0f + expf(-xi));
    dst[i] = __float2half_rn(yi);
}
)";
const char * kSiluF16EntryPoint = "ggml_cuda_dl_silu_f16";

static const int kSiluBlockSize = 256; // CUDA_SILU_BLOCK_SIZE, ggml/src/ggml-cuda/unary.cuh:13

SiluLaunchOutcome launch_silu_kernel(const DriverApi * drv, const char * source,
                                      const char * entry_point, CUdeviceptr d_in, CUdeviceptr d_out,
                                      int n, int arch_major, int arch_minor, CUstream stream,
                                      const std::vector<std::string> & extra_nvrtc_options) {
    SiluLaunchOutcome out;
    if (!drv) {
        out.error = "driver() is null (ensure_loaded().available must be true first)";
        return out;
    }

    CompileOptions opts;
    opts.arch_major = arch_major;
    opts.arch_minor = arch_minor;
    opts.extra_options = extra_nvrtc_options;

    CompileResult cr = compile(source, entry_point, /*name_expressions=*/{}, opts);
    if (!cr.ok) {
        out.error = std::string("nvrtc compile failed: ") + cr.error + "\n" + cr.log;
        return out;
    }
    out.from_cache = cr.from_cache;

    CUmodule mod = nullptr;
    CUresult rc = drv->cuModuleLoadData(&mod, cr.cubin.data());
    if (rc != CUDA_SUCCESS_DL) {
        out.error = "cuModuleLoadData: " + cu_error_string(rc);
        return out;
    }
    CUfunction fn = nullptr;
    // extern "C" entry point -- the raw name, no nvrtcGetLoweredName lookup needed (that dance is
    // only for C++-mangled / templated names; see kernels_silu.h).
    rc = drv->cuModuleGetFunction(&fn, mod, entry_point);
    if (rc != CUDA_SUCCESS_DL) {
        out.error = "cuModuleGetFunction: " + cu_error_string(rc);
        return out;
    }

    void * args[] = { (void *) &d_in, (void *) &d_out, (void *) &n };
    const unsigned int block = (unsigned int) kSiluBlockSize;
    const unsigned int grid  = (unsigned int) ((n + kSiluBlockSize - 1) / kSiluBlockSize);
    rc = drv->cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, stream, args, nullptr);
    if (rc != CUDA_SUCCESS_DL) {
        out.error = "cuLaunchKernel: " + cu_error_string(rc);
        return out;
    }
    rc = drv->cuStreamSynchronize(stream);
    if (rc != CUDA_SUCCESS_DL) {
        out.error = "cuStreamSynchronize: " + cu_error_string(rc);
        return out;
    }

    out.ok = true;
    return out;
}

} // namespace ggml_cuda_dl
