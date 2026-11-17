// test-ffd-kv.cpp: --ffd end to end through the real decode path (legacy llm_build_context + llama_kv_cache_fork).
// Usage: test-ffd-kv <model.gguf>   (any small model; the check is structural, not about output quality)
// BENCH-PENDING: correctness only; "FFD" names the upstream repo, no speed is claimed.
//
// Decodes a 300-token prompt and 5 single tokens with ffd on, then reads the cache itself -- K, the 2-bit shadow
// codes and the per-block scales -- and checks, for every layer that has a shadow:
//   1. every block the decode touched is BIT-EXACT against the ggml.h contract recomputed here from the real K
//      (so the input fill, the op's ordering after the K store and the layout all line up);
//   2. blocks the decode never touched still hold their initial bytes (the op wrote only what it was given);
//   3. NEGATIVE control: the same comparison against a reference built from ANOTHER layer's K must fail;
//   4. with ffd off, no shadow is allocated at all.

#include "llama.h"
#include "llama-context.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_fail = 0;
static void check(bool ok, const char * what) {
    printf("  %-4s %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++g_fail;
}

static llama_context * make_ctx(llama_model * model, bool ffd, int block) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = 512;
    cp.n_batch   = 512;
    cp.n_ubatch  = 512;
    cp.ffd       = ffd;
    cp.ffd_block = block;
    return llama_init_from_model(model, cp);
}

static bool decode_run(llama_context * ctx, int n_vocab, int n_prompt, int n_gen) {
    llama_batch b = llama_batch_init(512, 0, 1);
    b.n_tokens = n_prompt;
    for (int i = 0; i < n_prompt; ++i) {
        b.token[i] = 5 + (i*7919) % (n_vocab - 10);
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = i == n_prompt - 1;
    }
    bool ok = llama_decode(ctx, b) == 0;
    for (int g = 0; ok && g < n_gen; ++g) {
        b.n_tokens = 1;
        b.token[0] = 5 + (g*104729) % (n_vocab - 10);
        b.pos[0] = n_prompt + g;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 0;
        b.logits[0] = 1;
        ok = llama_decode(ctx, b) == 0;
    }
    llama_batch_free(b);
    return ok;
}

static std::vector<float> read_k(const ggml_tensor * t) {
    std::vector<float> out(ggml_nelements(t));
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    for (size_t i = 0; i < out.size(); ++i) {
        if (t->type == GGML_TYPE_F32) {
            out[i] = ((const float *) raw.data())[i];
        } else if (t->type == GGML_TYPE_F16) {
            out[i] = ggml_fp16_to_fp32(((const ggml_fp16_t *) raw.data())[i]);
        } else {
            out[i] = ggml_bf16_to_fp32(((const ggml_bf16_t *) raw.data())[i]);
        }
    }
    return out;
}

// mismatches of (codes, scales) in blocks [0, n_touched) against the contract applied to kf (layout of k_l:
// element (d, h, c) at d + D*(h + H*c)); also counts blocks >= n_touched whose scales are not all equal to `init_s`
struct cmp_result { int64_t code_bad = 0, scale_bad = 0, untouched_changed = 0; };

static cmp_result compare(const std::vector<float> & kf, const std::vector<uint8_t> & k2, const std::vector<float> & ks,
                          int D, int H, int kv, int block, int n_touched, float init_s) {
    cmp_result r;
    const int nblk = (kv + block - 1)/block;
    std::vector<float> sc(D);
    for (int h = 0; h < H; ++h) {
        for (int b = 0; b < nblk; ++b) {
            const float * sb = &ks[(size_t) D*(b + (size_t) nblk*h)];
            if (b >= n_touched) {
                for (int d = 0; d < D; ++d) r.untouched_changed += sb[d] != init_s;
                continue;
            }
            const int c0 = b*block, c1 = std::min(c0 + block, kv);
            for (int d = 0; d < D; ++d) {
                float am = 0.0f;
                for (int c = c0; c < c1; ++c) am = std::max(am, std::fabs(kf[(size_t) d + (size_t) D*(h + (size_t) H*c)]));
                sc[d] = std::max(am/1.5f, 1e-8f);
                r.scale_bad += memcmp(&sc[d], &sb[d], sizeof(float)) != 0;
            }
            for (int c = c0; c < c1; ++c) {
                for (int j = 0; j < D/4; ++j) {
                    uint8_t byte = 0;
                    for (int e = 0; e < 4; ++e) {
                        const int d = 4*j + e;
                        const float v = std::nearbyint(kf[(size_t) d + (size_t) D*(h + (size_t) H*c)]/sc[d] + 1.5f);
                        byte |= (uint8_t) ((int) std::min(std::max(v, 0.0f), 3.0f) << (2*e));
                    }
                    r.code_bad += k2[(size_t) j + (size_t) (D/4)*(c + (size_t) kv*h)] != byte;
                }
            }
        }
    }
    return r;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [n_gpu_layers (default 0 = CPU)]\n", argv[0]);
        return 2;
    }
    const int ngl = argc > 2 ? atoi(argv[2]) : 0;
    printf("test-ffd-kv: --ffd shadow maintenance through the real decode path (n_gpu_layers = %d)\n", ngl);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 2;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int block = 64, n_prompt = 300, n_gen = 5;

    // 4. control: ffd off -> no shadow
    {
        llama_context * ctx = make_ctx(model, false, block);
        check(ctx && ctx->kv_self.ffd_k2.empty(), "ffd off: no shadow tensors are allocated");
        llama_free(ctx);
    }

    llama_context * ctx = make_ctx(model, true, block);
    check(ctx != nullptr, "context with --ffd created");
    if (!ctx) return 1;
    const auto & kv = ctx->kv_self;
    int n_layers_shadowed = 0;
    for (auto * t : kv.ffd_k2) n_layers_shadowed += t != nullptr;
    printf("  layers with a shadow: %d of %zu\n", n_layers_shadowed, kv.k_l.size());
    check(n_layers_shadowed > 0, "at least one layer has a 2-bit K shadow");

    check(decode_run(ctx, n_vocab, n_prompt, n_gen), "decode 300-token prompt + 5 single tokens");
    const int used = n_prompt + n_gen;
    const int n_touched = (used - 1)/block + 1;

    int64_t code_bad = 0, scale_bad = 0, untouched = 0, n_checked = 0;
    int first_il = -1, second_il = -1;
    for (int il = 0; il < (int) kv.ffd_k2.size(); ++il) {
        const ggml_tensor * k2t = kv.ffd_k2[il];
        if (!k2t) continue;
        if (first_il < 0) first_il = il; else if (second_il < 0) second_il = il;
        const ggml_tensor * kt = kv.k_l[il];
        const int D = (int) k2t->ne[0]*4, kvn = (int) k2t->ne[1], H = (int) k2t->ne[2];
        std::vector<uint8_t> k2(ggml_nbytes(k2t));
        ggml_backend_tensor_get(k2t, k2.data(), 0, k2.size());
        std::vector<float> ks(ggml_nelements(kv.ffd_ks[il]));
        ggml_backend_tensor_get(kv.ffd_ks[il], ks.data(), 0, ks.size()*sizeof(float));
        const cmp_result r = compare(read_k(kt), k2, ks, D, H, kvn, block, n_touched, 0.0f);
        code_bad += r.code_bad; scale_bad += r.scale_bad; untouched += r.untouched_changed;
        ++n_checked;
    }
    char buf[256];
    snprintf(buf, sizeof buf, "touched blocks (%d per head) bit-exact vs the contract over %lld layers: codes off %lld, scales off %lld",
             n_touched, (long long) n_checked, (long long) code_bad, (long long) scale_bad);
    check(n_checked > 0 && code_bad == 0 && scale_bad == 0, buf);
    snprintf(buf, sizeof buf, "untouched blocks keep their initial (cleared) scales: %lld values changed", (long long) untouched);
    check(untouched == 0, buf);

    // 3. NEGATIVE control: layer A's shadow against layer B's K must disagree
    if (second_il >= 0) {
        const ggml_tensor * k2t = kv.ffd_k2[first_il];
        const int D = (int) k2t->ne[0]*4, kvn = (int) k2t->ne[1], H = (int) k2t->ne[2];
        std::vector<uint8_t> k2(ggml_nbytes(k2t));
        ggml_backend_tensor_get(k2t, k2.data(), 0, k2.size());
        std::vector<float> ks(ggml_nelements(kv.ffd_ks[first_il]));
        ggml_backend_tensor_get(kv.ffd_ks[first_il], ks.data(), 0, ks.size()*sizeof(float));
        const cmp_result r = compare(read_k(kv.k_l[second_il]), k2, ks, D, H, kvn, block, n_touched, 0.0f);
        check(r.code_bad > 0 || r.scale_bad > 0, "NEGATIVE: another layer's K does not match this shadow (the comparison discriminates)");
    } else {
        printf("  SKIP negative control: only one shadowed layer\n");
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    printf("%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
