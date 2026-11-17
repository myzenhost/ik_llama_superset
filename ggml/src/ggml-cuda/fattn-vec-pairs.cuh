#pragma once

// RESEARCH-DONE: upstream ggml-org/llama.cpp master 80d2c0c2c56e ggml/cmake/common.cmake ggml_cuda_fattn_vec_instances and
// ggml-cuda/CMakeLists.txt:110-130 read in full (fetched copy temp/faq_src_1107); this is the fork-typed, drift-proof form of it.
//
// Selection of the non-turbo quantised KV (Dk, Dv, K, V) flash-attention vec cases that are compiled in, driven by CMake's
// GGML_CUDA_FA_QUANTS (upstream spelling: "all" or a list of <K>-<V>; f16-f16 is always built; GGML_CUDA_FA_ALL_QUANTS is the
// deprecated alias of "all").
//
//   * fattn-vec-pairs.def is the single list of cases. CMake parses it (ggml/cmake/common.cmake), and this header expands it into
//     the dispatch and the capability predicate, so a case cannot be dispatched without being offered, or offered without
//     being dispatched.
//   * CMake defines GGML_CUDA_FA_<K>_<V> = 1 or 0 for every pair named in the .def, plus GGML_CUDA_FA_PAIRS_FROM_CMAKE. A pair
//     whose flag is 0 compiles no dispatch line (so no reference to a missing instance, hence no link error and no null kernel)
//     and its predicate answers false, which makes ggml_backend_supports_op say no and the scheduler fall back cleanly.

#include "common.cuh"

#ifndef GGML_CUDA_FA_PAIRS_FROM_CMAKE
#error "GGML_CUDA_FA_<K>_<V> pair flags come from CMake (ggml/cmake/common.cmake ggml_cuda_fattn_vec_fork_instances); build through CMake"
#endif

#define GGML_FA_CAT_(a, b) a##b
#define GGML_FA_CAT(a, b)  GGML_FA_CAT_(a, b)
// the 0/1 flag of a pair, by its type-name suffixes: GGML_FA_FLAG(Q8_0, Q4_0) -> GGML_CUDA_FA_Q8_0_Q4_0 -> 1 or 0
#define GGML_FA_FLAG(K, V) GGML_FA_CAT(GGML_FA_CAT(GGML_CUDA_FA_, K), GGML_FA_CAT(_, V))

// ---- dispatch -------------------------------------------------------------------------------------------------------------
// Usage inside ggml_cuda_flash_attn_ext_vec_f16/f32, where CASE is FATTN_VEC_F16_CASE_DKDV / FATTN_VEC_F32_CASE_DKDV:
//     #define FA_VEC_PAIR(Dk, Dv, K, V)      GGML_FA_DISPATCH(FATTN_VEC_F16_CASE_DKDV, Dk, Dv, K, V)
//     #define FA_VEC_PAIR_F32(Dk, Dv, K, V)  (empty for the f16 kernel)
//     #include "fattn-vec-pairs.def"
#define GGML_FA_DISPATCH_1(CASE, Dk, Dv, K, V) CASE(Dk, Dv, GGML_TYPE_##K, GGML_TYPE_##V)
#define GGML_FA_DISPATCH_0(CASE, Dk, Dv, K, V)
#define GGML_FA_DISPATCH(CASE, Dk, Dv, K, V) GGML_FA_CAT(GGML_FA_DISPATCH_, GGML_FA_FLAG(K, V))(CASE, Dk, Dv, K, V)

// ---- capability -----------------------------------------------------------------------------------------------------------
// true iff (Dk, Dv, type_K, type_V) is one of the .def cases whose pair is compiled in. f32_kernel selects the kernel whose
// extra (FA_VEC_PAIR_F32) cases count too. The answer and the dispatch above come from the same file and the same flags.
#define GGML_FA_PRED_1(Dk, Dv, K, V) if (dk == (Dk) && dv == (Dv) && tk == GGML_TYPE_##K && tv == GGML_TYPE_##V) { return true; }
#define GGML_FA_PRED_0(Dk, Dv, K, V)
#define GGML_FA_PRED(Dk, Dv, K, V) GGML_FA_CAT(GGML_FA_PRED_, GGML_FA_FLAG(K, V))(Dk, Dv, K, V)

static inline bool ggml_cuda_fattn_vec_pair_compiled(const bool f32_kernel, const int64_t dk, const int64_t dv, const ggml_type tk, const ggml_type tv) {
#define FA_VEC_PAIR(Dk, Dv, K, V) GGML_FA_PRED(Dk, Dv, K, V)
#define FA_VEC_PAIR_F32(Dk, Dv, K, V) if (f32_kernel) { GGML_FA_PRED(Dk, Dv, K, V) }
#include "fattn-vec-pairs.def"
#undef FA_VEC_PAIR
#undef FA_VEC_PAIR_F32
    return false;
}
