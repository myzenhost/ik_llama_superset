// Stage 3: first real ggml-cuda kernel ported onto the NVRTC dynamic-loading path.
// Row S69-ADD-FERROX-CUDARC-DYNAMIC-CUDA-LOADING-2026-09-27-01 (stage 3).
//
// Kernel chosen: SiLU (ggml/src/ggml-cuda/unary.cu:36-43, silu_f32 -- the ik-specific F32-only
// elementwise path, dispatched from ggml_cuda_op_silu at unary.cu:404-419 when the tensor is F32).
// Picked over scale_f32 (unary.cu's sibling, ggml/src/ggml-cuda/scale.cu) because scale_f32 is
// wrapped in ggml_cuda_pdl_lc()/ggml_cuda_pdl_sync() (programmatic dependent launch helpers,
// extra machinery this stage does not need), and picked over rms_norm (uses block-wide reductions
// / warp shuffles) and the templated GGML_UNARY_OP_* family further down unary.cu (op_silu
// functor instantiated via ggml_cuda_op_unary<op>, template-heavy per the task's own steer-away
// list). silu_f32 is one flat `__global__` function, no shared memory, no reduction, no template
// instantiation, F32-only in its own right in this codebase -- the smallest real, still
// inference-relevant (SwiGLU-family MLPs) kernel available.
// SOURCE-READ (rule 1.87): kernel body copied verbatim from unary.cu:36-43, read in full
// (this repo, own project source, MIT-licensed, not third-party copyleft -- "no code copied"
// caveat is about external sources, not reusing our own few-line kernel for a same-project parity
// test), with `extern "C"` added so NVRTC's lowered name matches without name-mangling /
// nvrtcGetLoweredName -- the axpy<float> test kernel in test-cuda-dl-launch.cpp needed that dance
// only because it is a template; ours is not.
#pragma once

#include "cuda_dl.h"
#include "nvrtc_cache.h"

#include <string>
#include <vector>

namespace ggml_cuda_dl {

// F32: faithful NVRTC port of unary.cu:36-43.
extern const char * kSiluF32Source;
extern const char * kSiluF32EntryPoint; // "ggml_cuda_dl_silu_f32"

// RESEARCH-DONE: unary.cu:36-43 read in full (see above); this is a deliberately-introduced,
// self-authored defect (sign flipped: x/(1 - exp(-x)) instead of the real x/(1 + exp(-x))), not a
// finding about the real kernel. Used ONLY by the must-fail parity control in
// tests/test-cuda-dl-silu-parity.cpp, to prove the parity harness actually discriminates a bad
// kernel from a correct one, per the ANTI-BIAS instruction ("a test that compares the new path
// against itself proves nothing"). Same entry point name, different source text -> different
// nvrtc_cache key -> compiled and cached independently of the real kernel.
extern const char * kSiluF32PerturbedSource;

// F16: best-effort NVRTC port using the installed toolkit's cuda_fp16.h (via an nvrtc
// -I<toolkit include dir> passed through CompileOptions::extra_options -- NOT required to COMPILE
// ggml-cuda-dl itself, only to compile THIS kernel string at run time if the caller wants F16
// coverage; F32 above needs no toolkit headers at all). Computes in F32 internally, rounds to
// F16 on store (__float2half_rn) -- this is a reasonable, common pattern but is NOT verified here
// to be bit-identical to whatever the upstream templated ggml_cuda_op_unary<op_silu> F16 path
// does (that path is template-heavy and deliberately out of scope for stage 3 -- see the header
// comment above). F16 parity in the test below is therefore checked against an independent CPU
// reference only, not against a "static kernel" byte-diff.
extern const char * kSiluF16Source;
extern const char * kSiluF16EntryPoint; // "ggml_cuda_dl_silu_f16"

struct SiluLaunchOutcome {
    bool ok = false;
    bool from_cache = false;
    std::string error; // set when ok == false
};

// Compiles `source` (cached by nvrtc_cache::compile -- second and later calls with the same
// source/arch/options hit the disk cache) and launches it as `entry_point(x, dst, k)` over `n`
// elements, block size 256 (CUDA_SILU_BLOCK_SIZE in unary.cuh) to match the static kernel's own
// launch configuration exactly (grid = ceil(n/256), 1-D). `d_in`/`d_out` must already be allocated
// via drv->cuMemAlloc with >= n * elem_size bytes; caller owns their lifetime. `extra_nvrtc_options`
// lets the F16 caller pass `-I<toolkit include>` without this function needing to know about it.
SiluLaunchOutcome launch_silu_kernel(const DriverApi * drv, const char * source,
                                      const char * entry_point, CUdeviceptr d_in, CUdeviceptr d_out,
                                      int n, int arch_major, int arch_minor, CUstream stream,
                                      const std::vector<std::string> & extra_nvrtc_options = {});

} // namespace ggml_cuda_dl
