#pragma once

#include "common.cuh"

// S69 unsloth port -- stage-1 dense device sampler for DiffusionGemma.
//
// Reads per-position canvas logits straight from a device tensor [n_vocab, n_tokens] (row-major per
// position) and returns the small per-position arrays, removing the per-step full-canvas D2H logits
// download plus the host full-vocab reductions. Its ONLY caller is llama_diffusion_device_sample()
// in src/llama.cpp, which runs it over the model's device-resident SC buffer (dg_sc_dev) -- an
// adopted kernel with no caller is not an adoption.
//
//   logits   : device tensor, F32, contiguous, ne[0]=n_vocab, nrows>=n_tokens
//   u_host   : host [n_tokens] pre-drawn uniforms (kept on the host RNG stream for reproducibility)
//   *_host   : host outputs [n_tokens] (argmax, entropy, sampled)
// Returns false on a non-CUDA / unsupported tensor, so the caller falls back to the host path.
bool ggml_cuda_diffusion_sample(
        struct ggml_tensor * logits,
        const float        * u_host,
        int                * argmax_host,
        float              * entropy_host,
        int                * sampled_host,
        int                  n_tokens,
        float                inv_temp);
