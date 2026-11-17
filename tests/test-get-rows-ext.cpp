// test-get-rows-ext.cpp: CPU backend vs an independent byte-level reference for the two ggml_get_rows_ext modes the CPU kernel did not implement.
//
//   dim0 = true : out[j,i1,i2,i3] = a[idx[j,i1,i2,i3], i1,i2,i3]   (gather along dim 0; op_params[0] = 1; CUDA: k_get_rows_dim0)
//   dim0 = false, same_type = true : out[:,j,i2,i3] = a[:, idx[j,i2,i3], i2,i3]   (row copy WITHOUT the f32 conversion; dst->type == a->type)
//
// MEASURED 2026-10-07 before this kernel existed: DeepSeek-V4 single-token decode on the CPU backend (build_deepseek4.cpp, n_tokens == 1) aborted at
// ops.cpp get_rows_f16 GGML_ASSERT(i01 < ne01) because the dim0 flag was ignored and the mask was read as a row table; the same-type CSA/HCA K gather
// wrote f32 rows into an f16 destination. Row S69-FORK-TEST-LLAMA-ARCHS-DEEPSEEK4-41-TENSOR-BEARING-FIXTURE-2026-10-04-01.
// Results are compared as raw bytes: a gather moves bits, it must not change them. A CONTROL flips one expected byte and requires the comparison to notice.

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint64_t g_rng = 0x9e3779b97f4a7c15ULL;
static uint32_t rnd(void) {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t) (g_rng >> 33);
}

// a: type t, [ne0, ne1, ne2, ne3]; idx: I32 values below `range`, shape [k, ne1|ne2 ..] given per mode
static bool dim0_case(ggml_type t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t k, const char * label) {
    struct ggml_init_params ip = { (size_t) 16 << 20, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * a = ggml_new_tensor_4d(ctx, t, ne0, ne1, ne2, 1);
    struct ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, k, ne1, ne2, 1);
    for (size_t i = 0; i < ggml_nbytes(a); ++i) ((uint8_t *) a->data)[i] = (uint8_t) rnd();
    for (int64_t i = 0; i < ggml_nelements(b); ++i) ((int32_t *) b->data)[i] = (int32_t) (rnd() % (uint32_t) ne0);
    struct ggml_tensor * out = ggml_get_rows_ext(ctx, a, b, true, true);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    const size_t ts = ggml_type_size(t);
    std::vector<uint8_t> want(ggml_nbytes(out));
    for (int64_t i2 = 0; i2 < ne2; ++i2)
        for (int64_t i1 = 0; i1 < ne1; ++i1)
            for (int64_t j = 0; j < k; ++j) {
                const int32_t id = ((const int32_t *) b->data)[j + k*(i1 + ne1*i2)];
                memcpy(want.data() + ts*(j + k*(i1 + ne1*i2)), (const uint8_t *) a->data + ts*(id + ne0*(i1 + ne1*i2)), ts);
            }
    const bool ok = out->type == t && out->ne[0] == k && memcmp(want.data(), out->data, want.size()) == 0;
    printf("%s dim0 %-5s ne=[%lld,%lld,%lld] k=%lld\n", ok ? "ok  " : "FAIL", ggml_type_name(t), (long long) ne0, (long long) ne1, (long long) ne2, (long long) k);
    (void) label;
    ggml_free(ctx);
    return ok;
}

static bool rows_case(ggml_type t, int64_t ne0, int64_t nrows, int64_t ne2, int64_t k, bool control) {
    struct ggml_init_params ip = { (size_t) 16 << 20, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * a = ggml_new_tensor_4d(ctx, t, ne0, nrows, ne2, 1);
    struct ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, k, ne2, 1, 1);
    for (size_t i = 0; i < ggml_nbytes(a); ++i) ((uint8_t *) a->data)[i] = (uint8_t) rnd();
    for (int64_t i = 0; i < ggml_nelements(b); ++i) ((int32_t *) b->data)[i] = (int32_t) (rnd() % (uint32_t) nrows);
    struct ggml_tensor * out = ggml_get_rows_ext(ctx, a, b, true, false);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    const size_t rs = ggml_row_size(t, ne0);
    std::vector<uint8_t> want(ggml_nbytes(out));
    for (int64_t i2 = 0; i2 < ne2; ++i2)
        for (int64_t j = 0; j < k; ++j) {
            const int32_t id = ((const int32_t *) b->data)[j + k*i2];
            memcpy(want.data() + rs*(j + k*i2), (const uint8_t *) a->data + rs*(id + nrows*i2), rs);
        }
    if (control) {
        want[want.size()/2] ^= 0x5a;
        const bool noticed = memcmp(want.data(), out->data, want.size()) != 0;
        printf("%s control: a flipped expected byte is detected\n", noticed ? "ok  " : "FAIL");
        ggml_free(ctx);
        return noticed;
    }
    const bool ok = out->type == t && memcmp(want.data(), out->data, want.size()) == 0;
    printf("%s rows %-5s ne0=%lld rows=%lld ne2=%lld k=%lld\n", ok ? "ok  " : "FAIL", ggml_type_name(t), (long long) ne0, (long long) nrows, (long long) ne2, (long long) k);
    ggml_free(ctx);
    return ok;
}

int main(void) {
    int fails = 0;
    // dim0: 2-byte (F16/BF16, the DSV4 masks) and 4-byte (F32) elements, single-row and batched shapes
    fails += !dim0_case(GGML_TYPE_F16,  256, 1, 1,  8, "mask 1 token, top-k 8");
    fails += !dim0_case(GGML_TYPE_F16,   32, 3, 2,  5, "batched");
    fails += !dim0_case(GGML_TYPE_BF16,  17, 2, 2,  4, "odd ne0");
    fails += !dim0_case(GGML_TYPE_F32,   40, 4, 3,  7, "f32");
    fails += !dim0_case(GGML_TYPE_F16,    9, 2, 1,  9, "k == ne0 (every column gathered)");
    // same-type row gather: must be a byte copy, not an f32 conversion
    fails += !rows_case(GGML_TYPE_F16,    64, 12, 1, 4, false);
    fails += !rows_case(GGML_TYPE_F16,    64, 12, 3, 5, false);
    fails += !rows_case(GGML_TYPE_BF16,   48,  9, 2, 3, false);
    fails += !rows_case(GGML_TYPE_Q8_0,   96,  7, 2, 4, false);
    fails += !rows_case(GGML_TYPE_F32,    24,  6, 2, 4, false);
    fails += !rows_case(GGML_TYPE_F16,    64, 12, 1, 4, true);
    printf(fails == 0 ? "all get_rows_ext checks passed\n" : "%d get_rows_ext checks FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
