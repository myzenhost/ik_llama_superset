// CPU turbo round trip (turbo3, turbo2, turbo4): quantise unit-Gaussian 128-vectors, dequantise, un-rotate, and bound the relative squared error and the cosine.
//
// WHY THESE BOUNDS. After the orthogonal WHT rotation each coordinate of a unit-norm 128-vector is ~N(0, 1/128); an n-bit Lloyd-Max quantiser has a relative
// distortion that is a textbook constant of the Gaussian, independent of any engine: 0.03454 (3 bits), 0.1175 (2 bits), 0.009497 (4 bits). So the mean relative
// squared error must sit near that number. Measured on this tree (see MEASURED below); each bound leaves ~20% headroom for the libm / rounding differences
// between toolchains and is tight enough that a wrong centroid table or a broken rotation trips it.
//
// CONTROL (the test must be able to fail): for every type the same metric is recomputed after corrupting the packed indices of half (turbo3, turbo4) or all
// (turbo2) of the coordinates of every block. That must push the error ABOVE the bound; if it does not, the metric is not measuring the quantiser and the
// test fails.
//   turbo3: flip the high index bit (signs bytes) of 64 coordinates   -> indices move by 4 of 8 levels
//   turbo2: invert every 2-bit index (qs ^ 0xFF)                       -> idx -> 3 - idx for all 128
//   turbo4: flip bit 3 of both nibbles of the first 32 qs bytes        -> indices move by 8 of 16 levels for 64 coordinates
//
// turbo4 is the 4-bit PolarQuant layout (TURBO4_USE_4BIT=1, 66-byte block: norm + 64 nibble-packed indices). Before 2026-10-05 this fork's TURBO4_USE_4BIT was
// never defined, so the CPU turbo4 path was the legacy 3-bit+QJL layout inside an 84-byte block and its round trip returned relative error 1.40 (task #278:
// worse than returning zeros). The turbo4 case below is that defect's fail-before / pass-after partner.
//
// GATE-CLAIM-OK: a unit test; the bounds are measured values, not resource claims.
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
void turbo_cpu_fwht_inverse(float * x, int group_size);
extern int turbo3_cpu_wht_group_size;
}

static uint64_t rs = 88172645463325252ULL;
static double u01() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return ((rs >> 11) + 0.5) / 9007199254740992.0; }
static double gauss() { return std::sqrt(-2.0 * std::log(u01())) * std::cos(6.283185307179586 * u01()); }

struct stats { double rel; double cosv; };

// Block layouts (ggml-common.h): [norm fp16 2B][...]. The 'sizeof == 14 bytes' comment next to block_turbo3_0 is stale: its static_assert gives 2 + 32 + 16 = 50.
static void corrupt_block(ggml_type t, unsigned char * blk) {
    switch (t) {
        case GGML_TYPE_TURBO3_0: {
            constexpr int SIGNS_OFF = 2 + 32;
            for (int b = 0; b < 8; b++) blk[SIGNS_OFF + b] ^= 0xFF;
        } break;
        case GGML_TYPE_TURBO2_0:
            for (int b = 0; b < 32; b++) blk[2 + b] ^= 0xFF;
            break;
        case GGML_TYPE_TURBO4_0:
            for (int b = 0; b < 32; b++) blk[2 + b] ^= 0x88;
            break;
        default: break;
    }
}

static stats run(ggml_type t, int n_vec, bool corrupt) {
    constexpr int D = 128;
    const ggml_type_traits * tr = ggml_get_type_traits(t);
    double sum_rel = 0.0, sum_cos = 0.0;
    alignas(16) unsigned char blk[256];  // one block is 34 / 50 / 66 bytes for 128 elements; the rest is slack
    for (int n = 0; n < n_vec; n++) {
        float x[D], y[D];
        for (int i = 0; i < D; i++) x[i] = (float) gauss();
        std::memset(blk, 0, sizeof(blk));
        tr->from_float_ref(x, blk, D);
        if (corrupt) corrupt_block(t, blk);
        tr->to_float(blk, y, D);
        turbo_cpu_fwht_inverse(y, D);
        double e = 0, nx = 0, ny = 0, dot = 0;
        for (int i = 0; i < D; i++) { double d = (double) x[i] - y[i]; e += d * d; nx += (double) x[i] * x[i]; ny += (double) y[i] * y[i]; dot += (double) x[i] * y[i]; }
        sum_rel += e / nx;
        sum_cos += dot / std::sqrt(nx * ny);
    }
    return { sum_rel / n_vec, sum_cos / n_vec };
}

// BLOCK-PURITY (chunk boundary) invariant. A chunked vec_dot stages its dequant through a fixed-size chunk, which is only correct if every row dequant is a pure per-block
// loop: dequantising [0,k) in one call must be BIT-identical to dequantising it in block-aligned pieces with the source advanced by (piece/blck)*type_size. Adapted in idea
// from TheTom/llama-cpp-turboquant tests/test-turbo-quant.c Test 4 (MIT); rewritten here with a stride-error control so the check can fail. This fork has no chunked turbo
// vec_dot today (TQ_DOT_CHUNK: 0 hits under ggml/src), so this pins the property a future one, or a GPU row decoder, relies on.
static int chunk_case(ggml_type t, const char * name, int64_t k, int64_t chunk_elems, bool wrong_stride) {
    const ggml_type_traits * tr = ggml_get_type_traits(t);
    const int64_t blck = tr->blck_size;
    std::vector<float> src(k), whole(k), piece(k, -12345.0f);
    for (int64_t i = 0; i < k; i++) src[i] = std::sin((float) i * 0.037f) * 3.0f;
    std::vector<unsigned char> q(ggml_row_size(t, k));
    ggml_quantize_chunk(t, src.data(), q.data(), 0, 1, k, nullptr, nullptr);
    tr->to_float(q.data(), whole.data(), k);
    const int64_t chunk = (chunk_elems / blck) * blck;
    const unsigned char * p = q.data();
    for (int64_t i = 0; i < k; i += chunk) {
        const int64_t nc = chunk < k - i ? chunk : k - i;
        tr->to_float(p, piece.data() + i, nc);
        const int64_t adv = (nc / blck) * (int64_t) tr->type_size;
        p += wrong_stride ? adv - 1 : adv;   // control: a one-byte stride error
    }
    int64_t bad = -1;
    for (int64_t i = 0; i < k; i++) if (std::memcmp(&whole[i], &piece[i], sizeof(float)) != 0) { bad = i; break; }
    if (!wrong_stride) std::printf("  chunk %-6s k=%-5lld chunk=%-4lld %s\n", name, (long long) k, (long long) chunk, bad < 0 ? "ok (bit-identical)" : "MISMATCH");
    return bad < 0 ? 0 : 1;
}

static int chunk_suite() {
    const struct { ggml_type t; const char * n; } types[] = { { GGML_TYPE_TURBO3_0, "turbo3" }, { GGML_TYPE_TURBO2_0, "turbo2" }, { GGML_TYPE_TURBO4_0, "turbo4" } };
    const int64_t lens[] = { 128, 256, 384, 512, 4096 };
    int fails = 0;
    for (const auto & ty : types) {
        for (int64_t k : lens) {
            if (chunk_case(ty.t, ty.n, k, 256, false)) { std::printf("FAIL: %s k=%lld chunked dequant differs from whole-row dequant\n", ty.n, (long long) k); fails++; }
        }
        // control: with a stride error the comparison MUST report a mismatch (k=512 gives two chunks), else the comparison is blind
        if (!chunk_case(ty.t, ty.n, 512, 256, true)) { std::printf("FAIL: %s stride-error control was not detected\n", ty.n); fails++; }
    }
    return fails;
}

struct case_t { ggml_type t; const char * name; double theory; double max_rel; double min_cos; };

int main() {
    // ggml_init fills the FP16->FP32 conversion table that GGML_FP16_TO_FP32 reads inside ggml-turbo-quant.c. Without it every block norm decodes to 0 and the
    // round trip returns all zeros (relative error exactly 1.0, cosine NaN) -- the first run of this test failed that way on 2026-10-05.
    struct ggml_init_params ip = { /*.mem_size =*/ 1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { std::printf("FAIL: ggml_init\n"); return 1; }
    turbo3_cpu_wht_group_size = 128;
    const case_t cases[] = {
        { GGML_TYPE_TURBO3_0, "turbo3", 0.0345, 0.040,  0.975  },
        { GGML_TYPE_TURBO2_0, "turbo2", 0.1175, 0.140,  0.930  },
        { GGML_TYPE_TURBO4_0, "turbo4", 0.0095, 0.0115, 0.9935 },
    };
    int fails = 0;
    for (const case_t & c : cases) {
        const stats good = run(c.t, 2000, false);
        std::printf("%s round trip: mean rel sq err %.6f (bound %.4f, theory ~%.4f), mean cosine %.6f (bound %.4f)\n", c.name, good.rel, c.max_rel, c.theory, good.cosv, c.min_cos);
        if (!(good.rel < c.max_rel)) { std::printf("FAIL: %s relative squared error above the bound\n", c.name); fails++; }
        if (!(good.cosv > c.min_cos)) { std::printf("FAIL: %s cosine below the bound\n", c.name); fails++; }
        const stats bad = run(c.t, 2000, true);
        std::printf("  control (%s indices corrupted): mean rel sq err %.6f, cosine %.6f\n", c.name, bad.rel, bad.cosv);
        if (!(bad.rel > c.max_rel)) { std::printf("FAIL: %s control did not push the error above the bound; the metric cannot detect a broken quantiser\n", c.name); fails++; }
    }
    fails += chunk_suite();
    ggml_free(ctx);
    std::printf("%s\n", fails ? "FAILED" : "PASS");
    return fails ? 1 : 0;
}
