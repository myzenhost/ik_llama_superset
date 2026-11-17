#include "fattn-vec-pairs.cuh"
#include "fattn-vec-f32.cuh"
#include "fattn-vec-f32-interface.cuh"

#define FATTN_VEC_F32_CASE(D, type_K, type_V)                               \
    if (Q->ne[0] == (D) && K->type == (type_K) && V->type == (type_V)) {    \
        ggml_cuda_flash_attn_ext_vec_f32_case<D, D, type_K, type_V>(ctx, dst); \
        return;                                                             \
    }                                                                       \

#define FATTN_VEC_F32_CASE_DKDV(Dk, Dv, type_K, type_V)                               \
    if (Q->ne[0] == (Dk) && V->ne[0] == Dv && K->type == (type_K) && V->type == (type_V)) {    \
        ggml_cuda_flash_attn_ext_vec_f32_case<Dk, Dv, type_K, type_V>(ctx, dst); \
        return;                                                             \
    }

void ggml_cuda_flash_attn_ext_vec_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

    // TurboQuant KV. Listed OUTSIDE the GGML_CUDA_FA_ALL_QUANTS split because the turbo instance
    // files are compiled in BOTH configurations -- turbo KV compression is the whole reason those
    // files exist, so gating them on a flag meant for exhaustive quant coverage would leave the
    // default build silently unable to use the feature. Head sizes 128 and 256 only: the rotation
    // group is 128 wide, so a 64-wide head is not a whole number of groups.
    FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0,     GGML_TYPE_TURBO2_0)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_Q8_0,     GGML_TYPE_TURBO2_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_TURBO4_0, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_TURBO4_0, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_TURBO2_0)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_F16, GGML_TYPE_TURBO2_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_TURBO3_0)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_F16, GGML_TYPE_TURBO3_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_TURBO4_0)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_F16, GGML_TYPE_TURBO4_0)

    // Non-turbo quantised KV: one shared case list (fattn-vec-pairs.def); each pair is compiled/dispatched only when selected by
    // GGML_CUDA_FA_QUANTS (f16-f16 always). The capability answer below expands the same list with the same flags.
#define FA_VEC_PAIR(Dk, Dv, K, V) GGML_FA_DISPATCH(FATTN_VEC_F32_CASE_DKDV, Dk, Dv, K, V)
#define FA_VEC_PAIR_F32(Dk, Dv, K, V) GGML_FA_DISPATCH(FATTN_VEC_F32_CASE_DKDV, Dk, Dv, K, V)
#include "fattn-vec-pairs.def"
#undef FA_VEC_PAIR
#undef FA_VEC_PAIR_F32

    on_no_fattn_vec_case(Q->ne[0], V->ne[0]);
}

// Exactly the (Dk, type_K, type_V) set the FATTN_VEC_*_CASE turbo block above dispatches, and the
// set the instance files instantiate. Kept as one predicate so the dispatch table and the capability
// answer cannot drift -- a supports_op that says yes to a combination the dispatch does not carry is
// the capability-proxy failure documented at length in fattn.cu.
static inline bool fattn_vec_turbo_combo_is_instantiated(const ggml_tensor * K, const ggml_tensor * V) {
    if (K->ne[0] != V->ne[0]) return false;
    if (K->ne[0] != 128 && K->ne[0] != 256) return false;
    auto side_ok = [](ggml_type t) {
        // turbo2/3/4 on either side, the other side turbo, q8_0 or f16 (the caller has already established that at least one
        // side is turbo). 21 (K,V) pairs x head sizes 128 and 256, the same set the instance files instantiate.
        return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0 || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_F16;
    };
    return side_ok(K->type) && side_ok(V->type);
}

bool ggml_cuda_fattn_vec_f32_is_supported([[maybe_unused]] int device, const ggml_tensor * dst) {
    auto K = dst->src[1];
    auto V = dst->src[2];
    if (turbo_fattn_type_is_turbo(K->type) || turbo_fattn_type_is_turbo(V->type)) {
        return fattn_vec_turbo_combo_is_instantiated(K, V);
    }
    // Same case list and same pair flags as the dispatch above (fattn-vec-pairs.def): a pair that is not compiled in answers false,
    // so supports_op says no and the scheduler falls back instead of reaching a missing instance.
    return ggml_cuda_fattn_vec_pair_compiled(true, K->ne[0], V->ne[0], K->type, V->type);
}
