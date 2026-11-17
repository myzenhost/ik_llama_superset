// CPU-ONLY check of the TurboQuant KV code the Metal backend runs (ggml/src/ggml-metal/kernels/turbo.h).
//
// turbo.h is written so that it is valid Metal Shading Language AND valid C++ once the address-space keywords are neutralised. This test does
// exactly that: it defines device / thread / constant as nothing, supplies a float4x4 and the fp16 norm load/store, includes the real header and
// runs the Metal writer (quantize_turbo{2,3,4}_0) and the Metal 16-element reader (dequantize_turbo{2,3,4}_0 for il = 0..7) on the host against
// the project's reference (ggml-turbo-quant.c through the public type-traits table), through the same verdict as test-turbo-write-host.cpp.
// The kernels in quantize.metal / fa.metal call these functions unchanged, so the algorithm, the tables and the bit packing of the Metal path are
// executed here before any Metal device is involved.
//
// What this does NOT cover (DEVICE-PENDING, needs a Mac): the Metal compiler itself, `#pragma METAL fp math_mode(safe)` semantics, the kernels'
// indexing (kernel_set_rows_q / kernel_cpy_f32_q / kernel_get_rows_q / kernel_flash_attn_ext_kv_f16), kernel_turbo_wht, the host dispatch in
// ggml-metal-ops.cpp and supports_op. The device check is `test-backend-ops -o SET_ROWS,CPY,GET_ROWS,TURBO_WHT,FLASH_ATTN_EXT -b Metal`.
//
// Controls (a check that cannot fail proves nothing): (1) one flipped entry of the Metal WHT sign table must break the writer's agreement with the
// reference; (2) one nudged 3-bit centroid must break the reader's agreement; (3) a reader asked for the wrong 16-element chunk must disagree.

#include "turbo-write-check.h"

#include <cstdlib>

// ---- the shim: Metal address spaces and types as plain C++ --------------------------------------------------------------------------------------
using std::sqrt;
#define TURBO_HOSTTEST
#define TURBO_NORM_LOAD(x)     ggml_fp16_to_fp32(x)
#define TURBO_NORM_STORE(d, v) ((d) = ggml_fp32_to_fp16(v))
#define device
#define thread
#define constant

struct float4x4 {
    struct col {
        float v[4];
        float &       operator[](int k)       { return v[k]; }
        const float & operator[](int k) const { return v[k]; }
    } c[4];
    col &       operator[](int i)       { return c[i]; }
    const col & operator[](int i) const { return c[i]; }
};

#include "../ggml/src/ggml-metal/kernels/turbo.h"

#undef device
#undef thread
#undef constant

static int g_fail = 0;

static void expect(bool ok, const char * what) {
    printf("  %-4s %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++g_fail;
}

static std::vector<uint8_t> metal_quantize(ggml_type t, const std::vector<float> & x) {
    const size_t tsz = tw::traits_of(t)->type_size;
    const size_t nb  = x.size() / tw::kGroup;
    std::vector<uint8_t> out(nb * tsz, 0);
    for (size_t b = 0; b < nb; ++b) {
        const float * src = &x[b * tw::kGroup];
        uint8_t *     blk = &out[b * tsz];
        if (t == GGML_TYPE_TURBO3_0) {
            quantize_turbo3_0(src, *(block_turbo3_0 *) blk);
        } else if (t == GGML_TYPE_TURBO4_0) {
            quantize_turbo4_0(src, *(block_turbo4_0 *) blk);
        } else {
            quantize_turbo2_0(src, *(block_turbo2_0 *) blk);
        }
    }
    return out;
}

// kernel_flash_attn_ext_kv_f16 / kernel_cpy_q_f32 / kernel_get_rows_q walk a block as 8 chunks of 16 elements, il = 0..7; `il_shift` is the control.
static std::vector<float> metal_dequant(ggml_type t, const std::vector<uint8_t> & q, size_t n, int il_shift) {
    const size_t tsz = tw::traits_of(t)->type_size;
    const size_t nb  = n / tw::kGroup;
    std::vector<float> y(n, 0.0f);
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t * blk = &q[b * tsz];
        for (int il = 0; il < 8; ++il) {
            float4x4 r;
            const short ilx = (short) ((il + il_shift) & 7);
            if (t == GGML_TYPE_TURBO3_0) {
                dequantize_turbo3_0<float4x4>((const block_turbo3_0 *) blk, ilx, r);
            } else if (t == GGML_TYPE_TURBO4_0) {
                dequantize_turbo4_0<float4x4>((const block_turbo4_0 *) blk, ilx, r);
            } else {
                dequantize_turbo2_0<float4x4>((const block_turbo2_0 *) blk, ilx, r);
            }
            for (int i = 0; i < 4; ++i) {
                for (int k = 0; k < 4; ++k) {
                    y[b * tw::kGroup + il * 16 + 4 * i + k] = r[i][k];
                }
            }
        }
    }
    return y;
}

static bool same_floats(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

static void run(ggml_type t, const char * name) {
    printf("%s\n", name);
    const std::vector<float>   x   = tw::gen_data(96, 20261005);
    const std::vector<uint8_t> ref = tw::quantize_ref(t, x);

    // 1. the Metal writer against the reference
    const std::vector<uint8_t> got = metal_quantize(t, x);
    const tw::Verdict v = tw::judge(t, x, got, ref);
    tw::print_verdict("Metal writer", v);
    expect(v.pass, "Metal writer matches quantize_row_turbo*_0_ref within the fixed thresholds");
    expect(got == ref, "Metal writer is BYTE-IDENTICAL to the reference on this data (host arithmetic)");

    // 2. the Metal reader against the reference to_float, on the reference's own bytes
    expect(same_floats(metal_dequant(t, ref, x.size(), 0), tw::dequant(t, ref, x.size())),
           "Metal reader (il = 0..7) reproduces dequantize_row_turbo*_0 bit for bit (rotated basis)");

    // 3. controls
    {   // (a) a wrong chunk index must be visible
        expect(!same_floats(metal_dequant(t, ref, x.size(), 1), tw::dequant(t, ref, x.size())),
               "CONTROL: reading chunk il+1 instead of il differs from the reference");
    }
    {   // (b) one flipped entry of the Metal sign table breaks the writer
        TURBO_WHT_SIGNS1[5] = -TURBO_WHT_SIGNS1[5];
        const std::vector<uint8_t> bad = metal_quantize(t, x);
        TURBO_WHT_SIGNS1[5] = -TURBO_WHT_SIGNS1[5];
        const tw::Verdict vb = tw::judge(t, x, bad, ref);
        tw::print_verdict("CONTROL one flipped Metal signs1 entry", vb);
        expect(!vb.pass && bad != ref, "CONTROL: a flipped entry of the Metal sign table FAILS the verdict");
        expect(metal_quantize(t, x) == ref, "table restored: the writer agrees with the reference again");
    }
    {   // (c) one nudged centroid breaks the reader (3-bit table; the 2- and 4-bit tables are covered by the same mechanism)
        if (t == GGML_TYPE_TURBO3_0) {
            const float keep = TURBO_CENTROIDS_3BIT[3];
            TURBO_CENTROIDS_3BIT[3] = keep + 0.001f;
            const bool differs = !same_floats(metal_dequant(t, ref, x.size(), 0), tw::dequant(t, ref, x.size()));
            TURBO_CENTROIDS_3BIT[3] = keep;
            expect(differs, "CONTROL: a nudged Metal 3-bit centroid FAILS the reader comparison");
        }
    }
}

int main() {
    tw::init_tables();
    run(GGML_TYPE_TURBO3_0, "turbo3");
    run(GGML_TYPE_TURBO2_0, "turbo2");
    run(GGML_TYPE_TURBO4_0, "turbo4");
    printf("%s: %d check(s) failed\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
