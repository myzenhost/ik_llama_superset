#pragma once
// Runtime hook for the AVX2 scale search of the IQ3_K / IQ3_KS / IQ5_K / IQ6_K / IQ5_KS quantizers.
//
// WHY. Those quantizers live in iqk_quantize_base.cpp, which is part of ggml-base and is compiled with NO
// arch flags, so the `#ifdef __AVX2__` blocks ik wrote there (calls to find_best_scale_avx2) were
// preprocessed out and every build ran the scalar search. find_best_scale_avx2 itself is in the ggml-cpu
// TU iqk_quantize.cpp, which IS built with -mavx2. ggml-cpu registers its instantiations here from
// ggml_cpu_init() when GGML_IQK_AVX2_SCALE_SEARCH=1; ggml-base calls them when registered and keeps the
// scalar search otherwise (the default, no CPU backend initialised, or a CPU variant built without AVX2).
// Opt-in because of a measured quality cost; the numbers are at the registration site in ggml-cpu.c.
// Merge-owed item 9 of row S69-ENGINE-MERGE-IK-LLAMA-MAIN-TRIAL-2026-09-29-01.
#include "ggml.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct iqk_scale_search_result {
    float d;
    float best;
    bool  is_shifted;
};

// Arguments are those of find_best_scale_avx2<block_size, nmap>(nval, values, f_values, k_simple_index, xb,
// imatrix, sigma2, ntry, use_sigma2, step, final_step) in iqk_quantize.cpp.
typedef struct iqk_scale_search_result (*iqk_scale_search_fn)(int nval, const int8_t * values, const float * f_values,
        const int8_t * k_simple_index, const float * xb, const float * imatrix, float sigma2, int ntry,
        bool use_sigma2, float step, bool final_step);

// one slot per <block_size, nmap> instantiation the quantizers use
enum iqk_scale_search_kind {
    IQK_SCALE_SEARCH_16_127 = 0, // IQ3_K
    IQK_SCALE_SEARCH_32_127,     // IQ3_KS
    IQK_SCALE_SEARCH_16_240,     // IQ5_K
    IQK_SCALE_SEARCH_32_240,     // IQ5_KS
    IQK_SCALE_SEARCH_16_248,     // IQ6_K
    IQK_SCALE_SEARCH_COUNT,
};

// ggml-base side. Passing NULL restores the scalar search for that kind.
GGML_API void                iqk_set_scale_search(enum iqk_scale_search_kind kind, iqk_scale_search_fn fn);
GGML_API iqk_scale_search_fn iqk_get_scale_search(enum iqk_scale_search_kind kind);

// ggml-cpu side (iqk_quantize.cpp): registers the AVX2 instantiations when that TU is built with AVX2,
// does nothing otherwise. Returns the number of kinds registered.
int iqk_register_scale_search_cpu(void);

#ifdef __cplusplus
}
#endif
