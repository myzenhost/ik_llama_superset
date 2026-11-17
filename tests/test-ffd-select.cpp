// test-ffd-select.cpp: validate GGML_OP_FFD_SELECT (contract in ggml.h) on the CPU backend.
//
// Checks, each with a case that must FAIL if the property does not hold:
//   1. agreement with an independent double-precision reference written here from the ggml.h
//      contract (not from ops.cpp), across GQA ratios, block sizes and ragged tails. Decisions
//      within 1e-3 of the threshold are counted as near-ties and excluded, since float and double
//      may round them differently; every other decision must match.
//   2. delta = +INFINITY keeps every block that is not fully masked (dense-equivalence).
//   3. a fully masked block is pruned even when it would otherwise win.
//   4. planted structure: a block whose keys are anti-aligned with q is pruned at small delta,
//      a block aligned with q in the MIDDLE of the context is kept, and in a GQA group a block
//      that only ONE head likes is still kept (keep-if-any-head).
//   5. the quantiser round trip used to build the 2-bit input is itself checked, so a wrong
//      input cannot make the selector look right.
// Design source: qluoluo/FFD @ ca09458 (Apache-2.0), reimplemented, not copied.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint64_t g_rng = 0x9e3779b97f4a7c15ULL;
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
    int D, n_tokens, n_head, n_head_kv, n_kv, block;
};

// 2-bit symmetric per-block per-channel quantiser (the FFD scheme, quantized_cache.py:151-157):
// scale = absmax/1.5 over the block, code = clamp(round(k/scale + 1.5), 0, 3).
static void quantize_k2(const geom & g, const std::vector<float> & k, // [hkv][j][c]
                        std::vector<uint8_t> & k2, std::vector<float> & ks) {
    const int nb = (g.n_kv + g.block - 1)/g.block;
    k2.assign((size_t) g.n_head_kv*g.n_kv*(g.D/4), 0);
    ks.assign((size_t) g.n_head_kv*nb*g.D, 0.0f);
    for (int h = 0; h < g.n_head_kv; ++h) {
        for (int b = 0; b < nb; ++b) {
            for (int c = 0; c < g.D; ++c) {
                float amax = 0.0f;
                for (int j = b*g.block; j < std::min(g.n_kv, (b + 1)*g.block); ++j) {
                    amax = std::max(amax, std::fabs(k[((size_t) h*g.n_kv + j)*g.D + c]));
                }
                const float sc = std::max(amax/1.5f, 1e-8f);
                ks[((size_t) h*nb + b)*g.D + c] = sc;
                for (int j = b*g.block; j < std::min(g.n_kv, (b + 1)*g.block); ++j) {
                    int code = (int) std::lround(k[((size_t) h*g.n_kv + j)*g.D + c]/sc + 1.5f);
                    code = std::min(3, std::max(0, code));
                    k2[((size_t) h*g.n_kv + j)*(g.D/4) + c/4] |= (uint8_t) (code << (2*(c%4)));
                }
            }
        }
    }
}

// Independent reference: returns keep[hkv][t][b] and the margin |v - th| of each decision.
static void reference(const geom & g, const std::vector<float> & q, // [h][t][c]
                      const std::vector<uint8_t> & k2, const std::vector<float> & ks,
                      const std::vector<float> & mask, // [t][j], may be empty
                      double scale, double delta,
                      std::vector<int> & keep, std::vector<double> & margin) {
    const int nb = (g.n_kv + g.block - 1)/g.block;
    const int G  = g.n_head/g.n_head_kv;
    keep.assign((size_t) g.n_head_kv*g.n_tokens*nb, 0);
    margin.assign(keep.size(), INFINITY);
    for (int hkv = 0; hkv < g.n_head_kv; ++hkv) {
        for (int t = 0; t < g.n_tokens; ++t) {
            std::vector<double> bm((size_t) G*nb, -INFINITY);
            for (int gi = 0; gi < G; ++gi) {
                const int h = hkv*G + gi;
                for (int j = 0; j < g.n_kv; ++j) {
                    const double mv = mask.empty() ? 0.0 : mask[(size_t) t*g.n_kv + j];
                    if (mv == -INFINITY) continue;
                    const int b = j/g.block;
                    double dot = 0.0;
                    for (int c = 0; c < g.D; ++c) {
                        const int code = (k2[((size_t) hkv*g.n_kv + j)*(g.D/4) + c/4] >> (2*(c%4))) & 3;
                        dot += (double) q[((size_t) h*g.n_tokens + t)*g.D + c]*((code - 1.5)*ks[((size_t) hkv*nb + b)*g.D + c]);
                    }
                    bm[(size_t) gi*nb + b] = std::max(bm[(size_t) gi*nb + b], scale*dot + mv);
                }
            }
            for (int b = 0; b < nb; ++b) {
                int k = 0; double mg = INFINITY;
                for (int gi = 0; gi < G; ++gi) {
                    const double v = bm[(size_t) gi*nb + b];
                    if (v == -INFINITY) continue;
                    const double th = std::max(bm[(size_t) gi*nb], bm[(size_t) gi*nb + nb - 1]) - delta;
                    if (b == nb - 1 || v >= th) k = 1;
                    if (std::isfinite(th) && b != nb - 1) mg = std::min(mg, std::fabs(v - th));
                }
                keep[((size_t) hkv*g.n_tokens + t)*nb + b] = k;
                margin[((size_t) hkv*g.n_tokens + t)*nb + b] = mg;
            }
        }
    }
}

// Runs the op; returns keep[hkv][t][b].
static std::vector<int> run_op(const geom & g, const std::vector<float> & q, const std::vector<uint8_t> & k2,
                               const std::vector<float> & ks, const std::vector<float> & mask,
                               float scale, float delta, int n_threads) {
    const int nb = (g.n_kv + g.block - 1)/g.block;
    ggml_init_params ip = { 64*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tq  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, g.D, g.n_tokens, g.n_head, 1);
    ggml_tensor * tk2 = ggml_new_tensor_4d(ctx, GGML_TYPE_I8,  g.D/4, g.n_kv, g.n_head_kv, 1);
    ggml_tensor * tks = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, g.D, nb, g.n_head_kv, 1);
    ggml_tensor * tm  = nullptr;
    memcpy(tq->data,  q.data(),  q.size()*sizeof(float));   // q is [h][t][c] == ggml [c, t, h]
    memcpy(tk2->data, k2.data(), k2.size());
    memcpy(tks->data, ks.data(), ks.size()*sizeof(float));
    if (!mask.empty()) {
        tm = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, g.n_kv, g.n_tokens);
        for (size_t i = 0; i < mask.size(); ++i) ((ggml_fp16_t *) tm->data)[i] = ggml_fp32_to_fp16(mask[i]);
    }
    ggml_tensor * out = ggml_ffd_select(ctx, tq, tk2, tks, tm, g.block, scale, delta);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    std::vector<int> res((size_t) g.n_head_kv*g.n_tokens*nb);
    // dst [b, t, hkv] contiguous == res[hkv][t][b]
    memcpy(res.data(), out->data, res.size()*sizeof(int32_t));
    ggml_free(ctx);
    return res;
}

static void rand_fill(std::vector<float> & v, float amp) { for (auto & x : v) x = amp*frand(); }

int main(void) {
    ggml_cpu_init();

    // 5. the input builder: dequantised 2-bit K must correlate with K (rel. error well under 1).
    {
        const geom g = { 64, 1, 1, 1, 300, 128 };
        std::vector<float> k((size_t) g.n_kv*g.D); rand_fill(k, 1.0f);
        std::vector<uint8_t> k2; std::vector<float> ks;
        quantize_k2(g, k, k2, ks);
        double num = 0, den = 0;
        for (int j = 0; j < g.n_kv; ++j) for (int c = 0; c < g.D; ++c) {
            const int code = (k2[(size_t) j*(g.D/4) + c/4] >> (2*(c%4))) & 3;
            const double d = (code - 1.5)*ks[(size_t) (j/g.block)*g.D + c];
            num += (d - k[(size_t) j*g.D + c])*(d - k[(size_t) j*g.D + c]); den += k[(size_t) j*g.D + c]*k[(size_t) j*g.D + c];
        }
        const double rel = std::sqrt(num/den);
        printf("[quantiser] rel RMS of 2-bit round trip = %.4f\n", rel);
        check(rel > 0.05 && rel < 0.6, "2-bit round trip is lossy but correlated (0.05 < rel < 0.6)");
    }

    // 1 + 2. random agreement with the reference, then dense-equivalence at delta = inf.
    const geom geoms[] = {
        {  64, 1,  4, 4,  512, 128 },  // MHA
        { 128, 1,  8, 2, 1000, 128 },  // GQA 4, ragged tail (1000 = 7*128 + 104)
        {  64, 3,  6, 2,  257,  64 },  // several query tokens, tail of 1
        {  32, 2,  2, 1,   96, 128 },  // a single (partial) block
    };
    for (const geom & g : geoms) {
        printf("[random] D=%d tok=%d heads=%d/%d n_kv=%d block=%d\n", g.D, g.n_tokens, g.n_head, g.n_head_kv, g.n_kv, g.block);
        std::vector<float> q((size_t) g.n_head*g.n_tokens*g.D), k((size_t) g.n_head_kv*g.n_kv*g.D);
        rand_fill(q, 1.0f); rand_fill(k, 1.0f);
        std::vector<uint8_t> k2; std::vector<float> ks; quantize_k2(g, k, k2, ks);
        std::vector<float> mask((size_t) g.n_tokens*g.n_kv, 0.0f);
        for (int t = 0; t < g.n_tokens; ++t)                  // causal-style: last t tokens hidden
            for (int j = g.n_kv - (g.n_tokens - 1 - t); j < g.n_kv; ++j) mask[(size_t) t*g.n_kv + j] = -INFINITY;
        const float scale = 4.0f/std::sqrt((float) g.D);  // sharpen so a finite delta prunes something
        for (float delta : { 0.5f, 2.0f, 3.466f }) {
            std::vector<int> ref; std::vector<double> mg;
            reference(g, q, k2, ks, mask, scale, delta, ref, mg);
            const std::vector<int> got = run_op(g, q, k2, ks, mask, scale, delta, 4);
            int mism = 0, ties = 0, kept = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                kept += got[i];
                if (got[i] == ref[i]) continue;
                if (mg[i] < 1e-3) ++ties; else ++mism;
            }
            char buf[160];
            snprintf(buf, sizeof(buf), "delta=%.3f: %d/%zu kept, %d mismatches, %d near-ties", delta, kept, ref.size(), mism, ties);
            check(mism == 0, buf);
        }
        {
            // Expected at delta=+inf: keep == 1 exactly for blocks NOT fully masked for that token, computed
            // here straight from the mask (independent of both the op and the reference). A block can be
            // fully masked -- e.g. n_kv=257/block=64 leaves a 1-position last block the causal mask hides.
            const std::vector<int> got = run_op(g, q, k2, ks, mask, scale, INFINITY, 3);
            const int nb = (g.n_kv + g.block - 1)/g.block;
            int bad = 0, masked_blocks = 0;
            for (int hkv = 0; hkv < g.n_head_kv; ++hkv) for (int t = 0; t < g.n_tokens; ++t) for (int b = 0; b < nb; ++b) {
                bool any_open = false;
                for (int j = b*g.block; j < std::min(g.n_kv, (b + 1)*g.block); ++j) any_open |= mask[(size_t) t*g.n_kv + j] != -INFINITY;
                masked_blocks += !any_open;
                bad += got[((size_t) hkv*g.n_tokens + t)*nb + b] != (any_open ? 1 : 0);
            }
            char buf[160];
            snprintf(buf, sizeof(buf), "delta=+inf keeps exactly the not-fully-masked blocks (%d fully masked, %d wrong)", masked_blocks, bad);
            check(bad == 0, buf);
        }
    }

    // 3 + 4. planted structure, GQA 2.
    {
        const geom g = { 64, 1, 2, 1, 8*128, 128 };  // 8 blocks
        const int nb = 8;
        std::vector<float> q((size_t) g.n_head*g.D), k((size_t) g.n_kv*g.D);
        // Two independent random query heads (q0 . q1 ~ 0). The threshold of each head comes from
        // its FIRST and LAST blocks only, so the last block is made hot for BOTH heads -- otherwise
        // a head whose first and last blocks are cold gets a low threshold and keeps cold blocks,
        // which is the contract working as specified, not a selector fault.
        rand_fill(q, 1.0f);
        const float * q0 = q.data();
        const float * q1 = q.data() + g.D;
        rand_fill(k, 0.05f);                                            // background: near zero
        auto plant = [&](int j0, int j1, float a0, float a1) {
            for (int j = j0; j < j1; ++j) for (int c = 0; c < g.D; ++c) k[(size_t) j*g.D + c] += 0.5f*(a0*q0[c] + a1*q1[c]);
        };
        plant(0,     128,     1, 0);  // block 0: the sink, hot for head 0
        plant(3*128, 3*128+4, 1, 0);  // block 3: head 0 likes it, MIDDLE of the context
        plant(5*128, 5*128+4, 0, 1);  // block 5: ONLY head 1 likes it
        plant(6*128, 6*128+4, 1, 0);  // block 6: head 0 likes it, but it is fully masked
        plant(7*128, 8*128,   1, 1);  // block 7 (last): hot for both heads
        std::vector<uint8_t> k2; std::vector<float> ks; quantize_k2(g, k, k2, ks);
        std::vector<float> mask((size_t) g.n_kv, 0.0f);
        for (int j = 6*128; j < 7*128; ++j) mask[j] = -INFINITY;
        const float scale = 1.0f;
        std::vector<int> ref; std::vector<double> mg;
        reference(g, q, k2, ks, mask, scale, 2.0f, ref, mg);
        const std::vector<int> got = run_op(g, q, k2, ks, mask, scale, 2.0f, 2);
        printf("[planted] keep =");
        for (int b = 0; b < nb; ++b) printf(" %d", got[b]);
        printf("  (ref");
        for (int b = 0; b < nb; ++b) printf(" %d", ref[b]);
        printf(")\n");
        check(got == ref, "planted case matches the reference");
        check(got[0] == 1, "block 0 (the hot sink) kept");
        check(got[3] == 1, "block 3 (hot, in the MIDDLE of the context) kept");
        check(got[5] == 1, "block 5 kept although only head 1 of the group likes it (keep-if-any-head)");
        check(got[6] == 0, "block 6 pruned: fully masked, even though head 0 would keep it");
        check(got[7] == 1, "last block kept");
        const int cold = got[1] + got[2] + got[4];
        check(cold == 0, "cold blocks 1, 2, 4 pruned at delta=2");
        // mutation: with the mask removed block 6 must come back -- proves check 3 is about the mask
        std::vector<float> nomask;
        const std::vector<int> got2 = run_op(g, q, k2, ks, nomask, scale, 2.0f, 2);
        check(got2[6] == 1, "mutation: unmask block 6 -> kept (the prune above was the mask)");
    }

    printf("\n%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
