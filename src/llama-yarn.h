#pragma once

// YaRN attention-factor helpers, header-only and dependency-free so a unit test can include it directly
// (tests/test-rope-yarn-truncate.cpp).
//
// Spec (read, no code copied): transformers modeling_rope_utils._compute_yarn_parameters
//   get_mscale(scale, mscale = 1) = 1.0 if scale <= 1 else 0.1 * mscale * ln(scale) + 1.0
//   attention_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)   if mscale AND mscale_all_dim
//                    = get_mscale(factor)                                                otherwise
// (an explicit rope_parameters["attention_factor"] overrides both; this fork carries that as
//  <arch>.rope.scaling.yarn_attn_factor / attn_factor.)

#include <cmath>

static inline float llama_yarn_get_mscale(float scale, float mscale) {
    return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
}

// HF's falsy test: a missing key reads as 0 here, and HF's `if mscale and mscale_all_dim` is false for 0 too.
static inline float llama_yarn_attn_factor_ratio(float factor, float mscale, float mscale_all_dim) {
    if (mscale != 0.0f && mscale_all_dim != 0.0f) {
        return llama_yarn_get_mscale(factor, mscale) / llama_yarn_get_mscale(factor, mscale_all_dim);
    }
    return llama_yarn_get_mscale(factor, 1.0f);
}
