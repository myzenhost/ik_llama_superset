// Gyro trellis quant types (ik df93e4cc9 #2812, ik 4955387a2 #2821): TQ2_T, TQK6, TQK7, TQP5, TQP6, TQP7.
//
// These types are DECODE-ONLY (from_float is NULL: real files come from an offline Viterbi search), so the generic
// quantize round-trip tests (test-quantize-fns, test-backend-ops) cannot construct them. But every byte pattern is a
// valid trellis encoding, so random qs bytes plus a random fp16 scale ARE a valid tensor, and the format definition
// in ggml-common.h is precise enough to decode independently. This test checks, per type:
//
//   1. DECODE: dequantize_row_<t> (to_float) equals a bit-by-bit reference decoder written from the format comments
//      in ggml-common.h (step state = 16-bit circular window of the bit stream, hash x = (s & 0x7fff) * 0x9e3779b1,
//      codebook point x >> 21 and, for TQ2_T/TQK, (x >> 10) & 2047). Exact float equality: both sides compute
//      fp16(d) * fp16(lut) in float.
//   2. MATMUL: ggml_mul_mat(W, X) on the CPU backend (vec_dot fallback for TQK/TQP; iqk kernels for TQ2_T with
//      Q8_2_X4 activations, and the Q8_0_R8 repack path at >= 16 activation columns) against a double reference
//      built from decode 1. Relative-to-max error must stay below the activation + int8-codebook quantization noise.
//   3. LOAD: a GGUF written the way Agention writes it (TQ types under on-disk ids 144-149, which are IQ4_KS / IQ2_KS /
//      IQ4_KSS / Q8_K16 / Q8_K32 / Q8_KR8 here, plus agention.trellis.state_bits = 15) loads as the TQ types; the same
//      file without the key keeps the on-disk ids; state_bits != 15 is refused.

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "../ggml/src/ggml-common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

struct tq_type_info { ggml_type type; const char * name; int kind; int K; size_t block_bytes; };  // kind 0 TQ2_T, 1 TQK, 2 TQP
static const tq_type_info k_types[] = {
    { GGML_TYPE_TQ2_T, "tq2_t", 0, 8, 2 + 32 },
    { GGML_TYPE_TQK6,  "tqk6",  1, 6, 2 + 4*6 },
    { GGML_TYPE_TQK7,  "tqk7",  1, 7, 2 + 4*7 },
    { GGML_TYPE_TQP5,  "tqp5",  2, 5, 2 + 8*5 },
    { GGML_TYPE_TQP6,  "tqp6",  2, 6, 2 + 8*6 },
    { GGML_TYPE_TQP7,  "tqp7",  2, 7, 2 + 8*7 },
};

// 16 stream bits starting at bit `off`, LSB-first, circular modulo nbits: bit by bit, deliberately unlike the
// three-byte window the kernels use.
static uint32_t ref_window(const uint8_t * qs, uint32_t nbits, uint32_t off) {
    uint32_t s = 0;
    for (uint32_t b = 0; b < 16; ++b) {
        const uint32_t p = (off + b) % nbits;
        s |= (uint32_t) ((qs[p >> 3] >> (p & 7)) & 1) << b;
    }
    return s;
}

static void ref_point(uint32_t idx, float d, float * out) {
    out[0] = d * ggml_fp16_to_fp32(tq2t_lut_f16[2*idx + 0]);
    out[1] = d * ggml_fp16_to_fp32(tq2t_lut_f16[2*idx + 1]);
}

static void ref_decode_block(const tq_type_info & ti, const uint8_t * blk, float * y) {
    uint16_t dh; std::memcpy(&dh, blk, 2);
    const float d = ggml_fp16_to_fp32(dh);
    const uint8_t * qs = blk + 2;
    if (ti.kind == 2) {
        for (int t = 0; t < 64; ++t) {
            const uint32_t s = ref_window(qs, 64u*ti.K, (uint32_t) (63 - t) * ti.K);
            const uint32_t x = (s & 0x7fffu) * 0x9e3779b1u;
            ref_point(x >> 21, d, y + 2*t);
        }
        return;
    }
    for (int t = 0; t < 32; ++t) {
        uint32_t s;
        if (ti.kind == 0) {
            s = ((uint32_t) qs[(t + 31) % 32] << 8) | qs[t];
        } else {
            s = ref_window(qs, 32u*ti.K, (uint32_t) (31 - t) * ti.K);
        }
        const uint32_t x = (s & 0x7fffu) * 0x9e3779b1u;
        ref_point(x >> 21, d, y + 4*t + 0);
        ref_point((x >> 10) & 2047u, d, y + 4*t + 2);
    }
}

static void fill_row(const tq_type_info & ti, uint8_t * row, int64_t K, std::mt19937 & rng) {
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002f, 0.02f);
    for (int64_t b = 0; b < K/128; ++b) {
        uint8_t * blk = row + b*ti.block_bytes;
        const uint16_t dh = ggml_fp32_to_fp16(scale(rng));
        std::memcpy(blk, &dh, 2);
        for (size_t i = 2; i < ti.block_bytes; ++i) blk[i] = (uint8_t) byte(rng);
    }
}

static double rel_to_max(const std::vector<double> & ref, const float * got, size_t n) {
    double mx = 0.0, err = 0.0;
    for (size_t i = 0; i < n; ++i) { mx = std::max(mx, std::fabs(ref[i])); err = std::max(err, std::fabs(ref[i] - (double) got[i])); }
    return mx > 0 ? err / mx : 1e30;
}

static void test_decode(const tq_type_info & ti, std::mt19937 & rng) {
    const ggml_type_traits * tr = ggml_get_type_traits(ti.type);
    CHECK(tr && tr->type_name && std::string(tr->type_name) == ti.name, "%s: type name '%s'", ti.name, tr && tr->type_name ? tr->type_name : "(null)");
    if (!tr || !tr->to_float) { CHECK(false, "%s: no to_float", ti.name); return; }
    CHECK(ggml_blck_size(ti.type) == 128 && ggml_type_size(ti.type) == ti.block_bytes,
          "%s: block %lld x %zu B, want 128 x %zu B", ti.name, (long long) ggml_blck_size(ti.type), ggml_type_size(ti.type), ti.block_bytes);
    const int64_t K = 128*64;
    std::vector<uint8_t> row(ggml_row_size(ti.type, K));
    fill_row(ti, row.data(), K, rng);
    std::vector<float> got(K), want(K);
    tr->to_float(row.data(), got.data(), K);
    for (int64_t b = 0; b < K/128; ++b) ref_decode_block(ti, row.data() + b*ti.block_bytes, want.data() + b*128);
    int64_t bad = 0, first = -1;
    for (int64_t i = 0; i < K; ++i) if (std::memcmp(&got[i], &want[i], 4) != 0) { if (first < 0) first = i; ++bad; }
    CHECK(bad == 0, "%s: decode differs from the reference at %lld of %lld weights (first %lld: %g vs %g)", ti.name,
          (long long) bad, (long long) K, (long long) first, first >= 0 ? got[first] : 0.f, first >= 0 ? want[first] : 0.f);
    printf("  %-6s decode: %lld weights, %lld differ from the bit-level reference\n", ti.name, (long long) K, (long long) bad);
}

static void test_matmul(const tq_type_info & ti, std::mt19937 & rng) {
    const ggml_type_traits * tr = ggml_get_type_traits(ti.type);
    const int64_t shapes[][3] = { {1024, 32, 1}, {1024, 32, 3}, {512, 16, 8}, {1024, 32, 16}, {1024, 64, 33}, {256, 8, 64} };  // K, N rows, M cols
    double worst = 0.0;
    for (auto & sh : shapes) {
        for (int nth : { 1, 4 }) {
            const int64_t K = sh[0], N = sh[1], M = sh[2];
            std::vector<uint8_t> buf(64u * 1024 * 1024);
            ggml_init_params ip = { buf.size(), buf.data(), false };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * W = ggml_new_tensor_2d(ctx, ti.type, K, N);
            ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
            const size_t rowA = ggml_row_size(ti.type, K);
            for (int64_t r = 0; r < N; r++) fill_row(ti, (uint8_t *) W->data + (size_t) r * rowA, K, rng);
            std::uniform_real_distribution<float> ad(-1.0f, 1.0f);
            for (int64_t i = 0; i < K * M; i++) ((float *) X->data)[i] = ad(rng);
            ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, Y);
            ggml_graph_compute_with_ctx(ctx, gf, nth);

            std::vector<float> Wd((size_t) N * K);
            for (int64_t r = 0; r < N; r++) tr->to_float((uint8_t *) W->data + (size_t) r * rowA, Wd.data() + (size_t) r * K, K);
            std::vector<double> ref((size_t) M * N);
            for (int64_t c = 0; c < M; c++)
                for (int64_t r = 0; r < N; r++) {
                    double s = 0.0;
                    for (int64_t e = 0; e < K; e++) s += (double) Wd[(size_t) r * K + e] * (double) ((float *) X->data)[(size_t) c * K + e];
                    ref[(size_t) c * N + r] = s;
                }
            const double e = rel_to_max(ref, (const float *) Y->data, (size_t) M * N);
            // q8 activations (~0.4% per value) and the TQ2_T int8 codebook (0.85% of its std) average out over K terms;
            // a decode or layout error gives O(1).
            CHECK(e <= 2e-2, "%s: mul_mat K=%lld N=%lld M=%lld nth=%d rel err %.3e > 2e-2", ti.name, (long long) K, (long long) N, (long long) M, nth, e);
            worst = std::max(worst, e);
            ggml_free(ctx);
        }
    }
    printf("  %-6s mul_mat: %zu shapes x 2 thread counts, worst relative-to-max error %.3e\n", ti.name, sizeof(shapes)/sizeof(shapes[0]), worst);
}

// ---- GGUF load remap ----------------------------------------------------------------------------------------------
static void put(std::string & s, const void * p, size_t n) { s.append((const char *) p, n); }
static void put_u32(std::string & s, uint32_t v) { put(s, &v, 4); }
static void put_u64(std::string & s, uint64_t v) { put(s, &v, 8); }
static void put_str(std::string & s, const std::string & v) { put_u64(s, v.size()); s += v; }

// A minimal GGUF v3 with one 2-D tensor of `on_disk_id` holding `nbytes` of data, optionally with the state_bits key.
static std::string make_gguf(uint32_t on_disk_id, int64_t ne0, int64_t ne1, size_t nbytes, int state_bits) {
    std::string s = "GGUF";
    put_u32(s, 3);
    put_u64(s, 1);                          // n_tensors
    put_u64(s, state_bits >= 0 ? 1 : 0);    // n_kv
    if (state_bits >= 0) {
        put_str(s, "agention.trellis.state_bits");
        put_u32(s, 4);                      // GGUF_TYPE_UINT32
        put_u32(s, (uint32_t) state_bits);
    }
    put_str(s, "blk.0.ffn_down.weight");
    put_u32(s, 2);
    put_u64(s, (uint64_t) ne0);
    put_u64(s, (uint64_t) ne1);
    put_u32(s, on_disk_id);
    put_u64(s, 0);                          // offset
    while (s.size() % 32) s.push_back(0);
    s.append(nbytes, '\x5a');
    return s;
}

static int load_type(const std::string & bytes, const char * path, bool * loaded) {
    FILE * f = fopen(path, "wb");
    if (!f) { *loaded = false; return -1; }
    fwrite(bytes.data(), 1, bytes.size(), f);
    fclose(f);
    gguf_init_params gp = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(path, gp);
    *loaded = g != nullptr;
    int t = -1;
    if (g) {
        t = (int) gguf_get_tensor_type(g, 0);
        gguf_free(g);
    }
    std::remove(path);
    return t;
}

static void test_gguf_remap() {
    struct { uint32_t disk; ggml_type want; } cases[] = {
        { 144, GGML_TYPE_TQ2_T }, { 145, GGML_TYPE_TQK6 }, { 146, GGML_TYPE_TQK7 },
        { 147, GGML_TYPE_TQP5 },  { 148, GGML_TYPE_TQP6 }, { 149, GGML_TYPE_TQP7 },
    };
    const char * path = "test-tq-trellis-remap.gguf";
    for (auto & c : cases) {
        const int64_t ne0 = 1024, ne1 = 4;
        const size_t nbytes = ggml_row_size(c.want, ne0) * ne1;
        bool ok = false;
        const int t = load_type(make_gguf(c.disk, ne0, ne1, nbytes, 15), path, &ok);
        CHECK(ok && t == (int) c.want, "on-disk id %u with state_bits 15: loaded=%d type %d, want %d (%s)",
              c.disk, (int) ok, t, (int) c.want, ggml_type_name(c.want));
        printf("  remap: on-disk %u + state_bits 15 -> %s (%s)\n", c.disk, ok && t >= 0 ? ggml_type_name((ggml_type) t) : "load failed",
               ok && t == (int) c.want ? "ok" : "WRONG");
    }
    {
        // without the key, 145 is this tree's IQ2_KS and must stay so (bytes sized for IQ2_KS)
        const size_t nbytes = ggml_row_size(GGML_TYPE_IQ2_KS, 1024) * 4;
        bool ok = false;
        const int t = load_type(make_gguf(145, 1024, 4, nbytes, -1), path, &ok);
        CHECK(ok && t == (int) GGML_TYPE_IQ2_KS, "on-disk 145 without the key: loaded=%d type %d, want IQ2_KS", (int) ok, t);
        printf("  no key: on-disk 145 -> %s\n", ok && t >= 0 ? ggml_type_name((ggml_type) t) : "load failed");
    }
    {
        const size_t nbytes = ggml_row_size(GGML_TYPE_TQK6, 1024) * 4;
        bool ok = true;
        load_type(make_gguf(145, 1024, 4, nbytes, 14), path, &ok);
        CHECK(!ok, "state_bits 14 must be refused (the decoders assume 15)");
        printf("  state_bits 14: %s\n", ok ? "LOADED (wrong)" : "refused");
    }
}

int main() {
    ggml_cpu_init();
    std::mt19937 rng(20261102);
    printf("TQ trellis types: decode, CPU mul_mat, GGUF on-disk id remap\n");
    for (auto & ti : k_types) {
        test_decode(ti, rng);
        test_matmul(ti, rng);
    }
    test_gguf_remap();
    printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
