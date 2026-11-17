// test-ffd-quant.cpp: validate GGML_OP_FFD_QUANT_K (contract in ggml.h) on the CPU backend.
// BENCH-PENDING: correctness only; no speed is claimed ("faster" below is only the upstream repo name).
//
// Checks, each able to FAIL:
//   1. every touched block is BIT-EXACT against a scalar reference written here from the ggml.h contract (not from
//      ops.cpp): scale = max(absmax/1.5, 1e-8) in float, code = clamp(round-half-even(k/scale + 1.5), 0, 3),
//      4 codes per byte low bits first -- across K types F32/F16/BF16, head dims, GQA shapes, streams, a short last
//      block, and write sets that are consecutive (prompt), single (decode), scattered and duplicated.
//   2. untouched blocks keep their sentinel bytes / scales exactly (the op must write ONLY what k_idxs touched).
//   3. round trip: |(code - 1.5)*scale - k| <= scale/2 (+ float slack) for every quantised element.
//   4. duplicate indices change nothing (idempotence), and the thread count changes nothing.
//   5. NEGATIVE control: the same comparison against a reference that rounds half AWAY from zero must find
//      mismatches on a planted tie, so check 1 really distinguishes the rounding rule.
// Quantiser source: qluoluo/faster-flash-decoding @ ca09458 (Apache-2.0) quantized_cache.py:111-192, reimplemented.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint64_t g_rng = 0x2545f4914f6cdd1dULL;
static float frand(void) {  // [-1, 1]
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((g_rng >> 33) & 0xffffff) / (float)0x800000 - 1.0f;
}

static int g_fail = 0;
static void check(bool ok, const char * what) {
    printf("  %-4s %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++g_fail;
}

struct geom {
    int D, H, kv_size, ns, block;
    ggml_type kt;
};

// K as the op sees it: row r = s*kv_size + c holds H*D values. `kf` keeps the exact float each stored element
// decodes to, so the reference and the op read the same numbers whatever the storage type.
struct kbuf {
    std::vector<uint8_t> raw;   // storage in g.kt
    std::vector<float>   kf;    // [row][h][d] as float
};

static void set_k(const geom & g, kbuf & k, size_t i, float v) {
    const size_t esz = ggml_type_size(g.kt);
    if (g.kt == GGML_TYPE_F32) {
        memcpy(&k.raw[i*esz], &v, 4);
        k.kf[i] = v;
    } else if (g.kt == GGML_TYPE_F16) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(v);
        memcpy(&k.raw[i*esz], &h, 2);
        k.kf[i] = ggml_fp16_to_fp32(h);
    } else {
        const ggml_bf16_t b = ggml_fp32_to_bf16(v);
        memcpy(&k.raw[i*esz], &b, 2);
        k.kf[i] = ggml_bf16_to_fp32(b);
    }
}

static kbuf make_k(const geom & g, float amp) {
    kbuf k;
    const size_t n = (size_t) g.kv_size*g.ns*g.H*g.D;
    k.raw.resize(n*ggml_type_size(g.kt));
    k.kf.resize(n);
    for (size_t i = 0; i < n; ++i) {
        set_k(g, k, i, amp*frand());
    }
    return k;
}

static int n_blk(const geom & g) { return (g.kv_size + g.block - 1)/g.block; }
static size_t k2_at(const geom & g, int s, int h, int c, int j) { return (((size_t) s*g.H + h)*g.kv_size + c)*(g.D/4) + j; }
static size_t ks_at(const geom & g, int s, int h, int b, int d) { return (((size_t) s*g.H + h)*n_blk(g) + b)*g.D + d; }

// Reference, from the contract. half_even=false rounds half away from zero (the NEGATIVE control).
static void reference(const geom & g, const kbuf & k, const std::vector<int64_t> & idx,
                      std::vector<uint8_t> & k2, std::vector<float> & ks, bool half_even) {
    std::vector<int> touched((size_t) g.ns*n_blk(g), 0);
    for (int64_t r : idx) {
        if (r >= 0 && r < (int64_t) g.kv_size*g.ns) {
            touched[(size_t) (r/g.kv_size)*n_blk(g) + (r % g.kv_size)/g.block] = 1;
        }
    }
    for (int s = 0; s < g.ns; ++s) for (int b = 0; b < n_blk(g); ++b) {
        if (!touched[(size_t) s*n_blk(g) + b]) continue;
        const int c0 = b*g.block, c1 = std::min(c0 + g.block, g.kv_size);
        for (int h = 0; h < g.H; ++h) {
            std::vector<float> sc(g.D, 0.0f);
            for (int c = c0; c < c1; ++c) for (int d = 0; d < g.D; ++d) {
                sc[d] = std::max(sc[d], std::fabs(k.kf[(((size_t) s*g.kv_size + c)*g.H + h)*g.D + d]));
            }
            for (int d = 0; d < g.D; ++d) {
                sc[d] = std::max(sc[d]/1.5f, 1e-8f);
                ks[ks_at(g, s, h, b, d)] = sc[d];
            }
            for (int c = c0; c < c1; ++c) for (int j = 0; j < g.D/4; ++j) {
                uint8_t byte = 0;
                for (int e = 0; e < 4; ++e) {
                    const int   d = 4*j + e;
                    const float x = k.kf[(((size_t) s*g.kv_size + c)*g.H + h)*g.D + d]/sc[d] + 1.5f;
                    const float v = half_even ? std::nearbyint(x) : std::round(x);
                    const int   q = (int) std::min(std::max(v, 0.0f), 3.0f);
                    byte |= (uint8_t) (q << (2*e));
                }
                k2[k2_at(g, s, h, c, j)] = byte;
            }
        }
    }
}

// Runs the op on CPU; k2/ks go in with the given contents and come back updated in place.
static void run_op(const geom & g, const kbuf & k, const std::vector<int64_t> & idx,
                   std::vector<uint8_t> & k2, std::vector<float> & ks, int n_threads) {
    ggml_init_params ip = { 256*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tk   = ggml_new_tensor_2d(ctx, g.kt, (int64_t) g.D*g.H, (int64_t) g.kv_size*g.ns);
    ggml_tensor * tidx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, (int64_t) idx.size());
    ggml_tensor * tk2  = ggml_new_tensor_4d(ctx, GGML_TYPE_I8,  g.D/4, g.kv_size, g.H, g.ns);
    ggml_tensor * tks  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, g.D, n_blk(g), g.H, g.ns);
    memcpy(tk->data,   k.raw.data(), k.raw.size());
    memcpy(tidx->data, idx.data(),   idx.size()*sizeof(int64_t));
    memcpy(tk2->data,  k2.data(),    k2.size());
    memcpy(tks->data,  ks.data(),    ks.size()*sizeof(float));
    ggml_tensor * out = ggml_ffd_quant_k(ctx, tk, tidx, tk2, tks, g.D, g.block);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    memcpy(k2.data(), tk2->data, k2.size());
    memcpy(ks.data(), tks->data, ks.size()*sizeof(float));
    ggml_free(ctx);
}

static std::vector<uint8_t> sentinel_k2(const geom & g) { return std::vector<uint8_t>((size_t) g.ns*g.H*g.kv_size*(g.D/4), 0xA5); }
static std::vector<float>   sentinel_ks(const geom & g) { return std::vector<float>((size_t) g.ns*g.H*n_blk(g)*g.D, -7.25f); }

static bool bit_equal(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

static void case_run(const char * name, const geom & g, const std::vector<int64_t> & idx, float amp) {
    kbuf k = make_k(g, amp);
    std::vector<uint8_t> k2o = sentinel_k2(g), k2r = sentinel_k2(g);
    std::vector<float>   kso = sentinel_ks(g), ksr = sentinel_ks(g);
    run_op(g, k, idx, k2o, kso, 4);
    reference(g, k, idx, k2r, ksr, true);
    char buf[256];
    snprintf(buf, sizeof buf, "%s: codes bit-exact vs reference (and untouched bytes still 0xA5)", name);
    check(k2o == k2r, buf);
    snprintf(buf, sizeof buf, "%s: scales bit-exact vs reference (and untouched scales still -7.25)", name);
    check(bit_equal(kso, ksr), buf);
    // on a mismatch, name the FIRST differing byte and scale so a failure says where, not just that
    for (size_t i = 0; i < k2o.size(); ++i) {
        if (k2o[i] == k2r[i]) continue;
        const size_t j = i % (g.D/4), c = (i/(g.D/4)) % g.kv_size, h = (i/(g.D/4)/g.kv_size) % g.H, s = i/(g.D/4)/g.kv_size/g.H;
        printf("       first code mismatch s=%zu h=%zu c=%zu j=%zu: op 0x%02X ref 0x%02X; k =", s, h, c, j, k2o[i], k2r[i]);
        for (int e = 0; e < 4; ++e) printf(" %g", k.kf[(((size_t) s*g.kv_size + c)*g.H + h)*g.D + 4*j + e]);
        printf("; scale op/ref =");
        for (int e = 0; e < 4; ++e) printf(" %g/%g", kso[ks_at(g, (int) s, (int) h, (int) c/g.block, 4*(int) j + e)], ksr[ks_at(g, (int) s, (int) h, (int) c/g.block, 4*(int) j + e)]);
        printf("\n");
        break;
    }
    for (size_t i = 0; i < kso.size(); ++i) {
        if (memcmp(&kso[i], &ksr[i], sizeof(float)) == 0) continue;
        printf("       first scale mismatch at flat %zu: op %.9g ref %.9g\n", i, kso[i], ksr[i]);
        break;
    }

    // round trip over every element of every touched block
    double worst = 0.0;
    bool   rt_ok = true;
    size_t n_q   = 0;
    for (int s = 0; s < g.ns; ++s) for (int h = 0; h < g.H; ++h) for (int c = 0; c < g.kv_size; ++c) {
        const int b = c/g.block;
        if (kso[ks_at(g, s, h, b, 0)] == -7.25f) continue;           // untouched block
        for (int d = 0; d < g.D; ++d) {
            const float  sc   = kso[ks_at(g, s, h, b, d)];
            const int    code = (k2o[k2_at(g, s, h, c, d/4)] >> (2*(d%4))) & 3;
            const float  kv   = k.kf[(((size_t) s*g.kv_size + c)*g.H + h)*g.D + d];
            const double err  = std::fabs(((double) code - 1.5)*sc - kv);
            worst = std::max(worst, err/sc);
            if (err > 0.5*sc*(1.0 + 1e-5) + 1e-12) rt_ok = false;
            ++n_q;
        }
    }
    snprintf(buf, sizeof buf, "%s: round trip |deq - k| <= scale/2 over %zu elements (worst %.4f scale)", name, n_q, worst);
    check(rt_ok && n_q > 0, buf);

    // idempotence: every index twice, in reverse order, and a different thread count
    std::vector<int64_t> dup(idx.rbegin(), idx.rend());
    dup.insert(dup.end(), idx.begin(), idx.end());
    std::vector<uint8_t> k2d = sentinel_k2(g);
    std::vector<float>   ksd = sentinel_ks(g);
    run_op(g, k, dup, k2d, ksd, 3);
    snprintf(buf, sizeof buf, "%s: duplicated + reversed indices, 3 threads -> identical", name);
    check(k2d == k2o && bit_equal(ksd, kso), buf);
}

int main(void) {
    printf("test-ffd-quant: GGML_OP_FFD_QUANT_K vs the ggml.h contract (CPU)\n");
    // FIRST, before anything calls ggml_init: the public F16 conversion must not depend on it. It did until 2026-09-28
    // (it read a table only ggml_init fills) and returned 0 here, which zeroed this file's reference data for case 1.
    check(ggml_fp16_to_fp32(ggml_fp32_to_fp16(1.5f)) == 1.5f && ggml_fp16_to_fp32(ggml_fp32_to_fp16(-2.25f)) == -2.25f,
          "ggml_fp16_to_fp32 is exact before any ggml_init (public API does not read an uninitialised table)");

    // prompt-like: consecutive cells 0..299 of stream 0 (blocks 0,1,2 touched; kv_size 400 -> 4 blocks, last short)
    {
        geom g = { 128, 4, 400, 1, 128, GGML_TYPE_F16 };
        std::vector<int64_t> idx;
        for (int i = 0; i < 300; ++i) idx.push_back(i);
        case_run("F16 D128 H4 prompt 0..299", g, idx, 3.0f);
    }
    // isolating variants of the case above: same write set as F32 (type), and as F16 with one head (GQA shape)
    {
        geom g = { 128, 4, 400, 1, 128, GGML_TYPE_F32 };
        std::vector<int64_t> idx;
        for (int i = 0; i < 300; ++i) idx.push_back(i);
        case_run("F32 D128 H4 prompt 0..299", g, idx, 3.0f);
    }
    {
        geom g = { 128, 1, 400, 1, 128, GGML_TYPE_F16 };
        std::vector<int64_t> idx;
        for (int i = 0; i < 300; ++i) idx.push_back(i);
        case_run("F16 D128 H1 prompt 0..299", g, idx, 3.0f);
    }
    // decode-like: one cell in the SHORT last block (cells 384..399), two streams, write lands in stream 1
    {
        geom g = { 64, 2, 400, 2, 128, GGML_TYPE_F32 };
        std::vector<int64_t> idx = { 400 + 390 };
        case_run("F32 D64 H2 ns2 decode, short last block", g, idx, 0.5f);
    }
    // scattered + BF16 + D=256
    {
        geom g = { 256, 3, 512, 1, 128, GGML_TYPE_BF16 };
        std::vector<int64_t> idx = { 5, 300, 301, 511, 130 };
        case_run("BF16 D256 H3 scattered", g, idx, 10.0f);
    }
    // block size 64 not dividing kv_size, MLA-sized head width
    {
        geom g = { 576, 1, 250, 1, 64, GGML_TYPE_F16 };
        std::vector<int64_t> idx = { 0, 63, 64, 249 };
        case_run("F16 D576 H1 block64", g, idx, 1.0f);
    }
    // an all-zero block: scale clamps to 1e-8 and every code is round(1.5) = 2 under half-even
    {
        geom g = { 32, 1, 128, 1, 128, GGML_TYPE_F32 };
        kbuf k = make_k(g, 1.0f);
        for (size_t i = 0; i < k.kf.size(); ++i) set_k(g, k, i, 0.0f);
        std::vector<uint8_t> k2 = sentinel_k2(g);
        std::vector<float>   ks = sentinel_ks(g);
        run_op(g, k, { 7 }, k2, ks, 2);
        bool all2 = true;
        for (uint8_t v : k2) all2 &= (v == 0xAA);              // 2,2,2,2 -> 0b10101010
        check(all2, "all-zero block: every code is 2 (round-half-even of 1.5) -> bytes 0xAA");
        check(ks[0] == 1e-8f && ks[31] == 1e-8f, "all-zero block: scale clamps to 1e-8");
    }
    // NEGATIVE control: a planted tie (k == scale, so k/scale + 1.5 == 2.5) must separate half-even from half-away
    {
        geom g = { 4, 1, 4, 1, 4, GGML_TYPE_F32 };
        kbuf k = make_k(g, 1.0f);
        // channel 0: absmax 1.5 -> scale 1.0; the cell holding 1.0 gives exactly 2.5. Channels 1-3 hold 0.5, which
        // gives 0.5/(0.5/1.5) + 1.5 = 2.9999998 in float -> 3 under both rules, so channel 0 is the only tie.
        const float col0[4] = { 1.5f, 1.0f, -1.5f, 0.25f };
        for (int c = 0; c < 4; ++c) for (int d = 0; d < 4; ++d) set_k(g, k, (size_t) c*4 + d, d == 0 ? col0[c] : 0.5f);
        std::vector<int64_t> idx = { 0 };
        std::vector<uint8_t> k2o = sentinel_k2(g), k2e = sentinel_k2(g), k2a = sentinel_k2(g);
        std::vector<float>   kso = sentinel_ks(g), kse = sentinel_ks(g), ksa = sentinel_ks(g);
        run_op(g, k, idx, k2o, kso, 1);
        reference(g, k, idx, k2e, kse, true);
        reference(g, k, idx, k2a, ksa, false);
        check(k2o == k2e, "planted tie: op == half-even reference");
        check(k2o != k2a, "NEGATIVE: op != half-away-from-zero reference (the comparison sees the rounding rule)");
        check(((k2o[1] >> 0) & 3) == 2, "planted tie: 2.5 rounds to 2 (even), not 3");
    }

    printf("%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
