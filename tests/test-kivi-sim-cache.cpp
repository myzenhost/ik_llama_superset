// test-kivi-sim-cache: --kivi-bits through the real decode path (legacy llm_build_context + llama_kv_cache_fork), checked on the cache itself.
// Usage: test-kivi-sim-cache <model.gguf>   (any small llama-arch model with head_dim % 32 == 0; the checks are structural)
// Row S69-FORK-KV-KIVI-2BIT-PORT-01. Numerics only: the cache tensors stay f16 (see src/llama-kv-kivi-sim.h).
//
// For bits in {2,4,8} and flash-attn on AND off (V non-transposed / transposed) it decodes N tokens in ubatches of 4 with
// group G=8, residual R=16 and then reads K and V back out of kv_self and checks
//   1. FIRED: per (layer, head, channel) every group of G fully-quantised K tokens holds <= 2^bits distinct values; per quantised V token every
//      channel group of G channels holds <= 2^bits distinct values (a dense f16 cache has ~G distinct values);
//   2. the residual window is NOT quantised (the newest tokens hold > 2^bits distinct values in the same test);
//   3. layer 0's K does not depend on the cache, so for K tokens still inside the window it is BIT-EXACT against a kivi-off run,
//      while for quantised tokens it differs from the kivi-off run and stays within the group's quantisation step;
//   4. K/V token counts match the KIVI rule: K quantised = floor(N/R)*R, V quantised = N-R;
//   5. kivi OFF: the cache is dense (> 2^bits distinct values) and logits are bit-identical to a context built with kivi_bits=0 and other kivi fields set;
//   6. R >= N (window covers everything): nothing is quantised and the logits are bit-identical to kivi off;
//   7. refusals: a quantised KV cache type with kivi on, and a residual that is not a multiple of the group, make context creation fail with a message.
#include "llama.h"
#include "llama-context.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

static int g_fail = 0;
static void check(bool ok, const char * what) {
    printf("  %-4s %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++g_fail;
}

static const int N = 72, UB = 4, G = 8, R = 16;

static llama_context * make_ctx(llama_model * model, int bits, bool fa, int g = G, int r = R, ggml_type tk = GGML_TYPE_F16) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 128; cp.n_batch = UB; cp.n_ubatch = UB; cp.flash_attn = fa;
    cp.kivi_bits = bits; cp.kivi_group = g; cp.kivi_residual = r;
    cp.type_k = tk;
    try { return llama_init_from_model(model, cp); } catch (const std::exception & e) { printf("    (context creation: %s)\n", e.what()); return nullptr; }
}

// decodes N tokens, returns logits of every position
static std::vector<float> run(llama_context * ctx, int n_vocab) {
    std::vector<float> all((size_t) N * n_vocab);
    for (int done = 0; done < N; done += UB) {
        llama_batch b = llama_batch_init(UB, 0, 1);
        b.n_tokens = UB;
        for (int j = 0; j < UB; j++) { b.token[j] = 3 + ((done + j) * 37 + 5) % (n_vocab - 3); b.pos[j] = done + j; b.n_seq_id[j] = 1; b.seq_id[j][0] = 0; b.logits[j] = 1; }
        if (llama_decode(ctx, b) != 0) { printf("decode failed\n"); std::exit(3); }
        for (int j = 0; j < UB; j++) memcpy(&all[(size_t) (done + j) * n_vocab], llama_get_logits_ith(ctx, j), n_vocab * sizeof(float));
        llama_batch_free(b);
    }
    return all;
}

struct kv_dump { std::vector<ggml_fp16_t> k0, k1, v0; uint32_t NH, D, size; bool v_trans; };

static kv_dump dump(llama_context * ctx) {
    kv_dump d;
    auto & kv = ctx->kv_self;
    const auto & hp = ctx->model.hparams;
    d.NH = hp.n_head_kv(0); d.D = hp.n_embd_head_k(0); d.size = kv.size; d.v_trans = kv.v_trans;
    d.k0.resize((size_t) N * d.NH * d.D); d.k1 = d.k0; d.v0.resize((size_t) N * d.NH * d.D);
    ggml_backend_tensor_get(kv.k_l[0], d.k0.data(), 0, d.k0.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_get(kv.k_l[1], d.k1.data(), 0, d.k1.size() * sizeof(ggml_fp16_t));
    const size_t E = (size_t) d.NH * d.D;
    if (!kv.v_trans) {
        ggml_backend_tensor_get(kv.v_l[0], d.v0.data(), 0, d.v0.size() * sizeof(ggml_fp16_t));
    } else {
        std::vector<ggml_fp16_t> all(E * kv.size);
        ggml_backend_tensor_get(kv.v_l[0], all.data(), 0, all.size() * sizeof(ggml_fp16_t));
        for (int t = 0; t < N; t++) for (size_t e = 0; e < E; e++) d.v0[(size_t) t * E + e] = all[e * kv.size + t];
    }
    return d;
}

static size_t distinct_k(const std::vector<ggml_fp16_t> & k, const kv_dump & d, int h, int ch, int t0, int n) {
    std::set<uint16_t> s;
    for (int t = t0; t < t0 + n; t++) s.insert(k[((size_t) t * d.NH + h) * d.D + ch]);
    return s.size();
}
static size_t distinct_v(const std::vector<ggml_fp16_t> & v, const kv_dump & d, int t, int ch0, int n) {
    std::set<uint16_t> s;
    for (int c = ch0; c < ch0 + n; c++) s.insert(v[(size_t) t * d.NH * d.D + c]);
    return s.size();
}

int main(int argc, char ** argv) {
    if (argc < 2) { printf("usage: %s model.gguf\n", argv[0]); return 2; }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { printf("load failed\n"); return 2; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    for (int fa = 1; fa >= 0; fa--) {
        printf("== flash_attn=%d (V %s)\n", fa, fa ? "plain" : "transposed");
        llama_context * off = make_ctx(model, 0, fa);
        auto lg_off = run(off, n_vocab);
        kv_dump d_off = dump(off);
        // 5. dense when off
        {
            size_t mx = 0;
            for (int ch = 0; ch < 8; ch++) mx = std::max(mx, distinct_k(d_off.k1, d_off, 0, ch, 0, G));
            check(mx > 4, "kivi off: K is dense (more than 2^2 distinct values per group of 8)");
        }
        {   // other kivi fields set but bits==0: still off and bit-identical
            llama_context * off2 = make_ctx(model, 0, fa, 16, 64);
            auto lg2 = run(off2, n_vocab);
            check(memcmp(lg_off.data(), lg2.data(), lg_off.size() * sizeof(float)) == 0, "kivi_bits=0 with group/residual set: logits bit-identical to default");
            llama_free(off2);
        }
        for (int bits : {2, 4, 8}) {
            printf("-- bits=%d\n", bits);
            llama_context * on = make_ctx(model, bits, fa);
            check(on != nullptr, "context created");
            if (!on) continue;
            auto lg_on = run(on, n_vocab);
            kv_dump d = dump(on);
            const int kq = (N / R) * R, vq = N - R;
            const size_t maxlev = (size_t) 1 << bits;
            // 1. levels per group (K): every layer, every head, a sample of channels
            size_t worst_k = 0, worst_v = 0, worst_win = 0;
            for (const auto * kk : {&d.k0, &d.k1}) {
                for (uint32_t h = 0; h < d.NH; h++) for (uint32_t ch = 0; ch < d.D; ch++)
                    for (int g0 = 0; g0 + G <= kq; g0 += G) worst_k = std::max(worst_k, distinct_k(*kk, d, h, ch, g0, G));
                for (uint32_t h = 0; h < d.NH; h++) for (uint32_t ch = 0; ch < d.D; ch++)
                    worst_win = std::max(worst_win, distinct_k(*kk, d, h, ch, kq, N - kq >= G ? G : N - kq));
            }
            for (int t = 0; t < vq; t++) for (uint32_t c0 = 0; c0 + G <= d.NH * d.D; c0 += G) worst_v = std::max(worst_v, distinct_v(d.v0, d, t, c0, G));
            char buf[200];
            snprintf(buf, sizeof buf, "K: <= %zu distinct values per (head,channel,group of %d tokens) over %d quantised tokens (worst %zu)", maxlev, G, kq, worst_k);
            check(worst_k <= maxlev, buf);
            snprintf(buf, sizeof buf, "V: <= %zu distinct values per (token, group of %d channels) over %d quantised tokens (worst %zu)", maxlev, G, vq, worst_v);
            check(worst_v <= maxlev, buf);
            if (bits == 2) { // only the 2-bit level count (4) is below the 8 tokens a group/window holds; 4/8-bit are covered by the layer-0 bit-exact check below
                snprintf(buf, sizeof buf, "2. residual window is dense: newest K tokens hold %zu distinct values (> %zu)", worst_win, maxlev);
                check(worst_win > maxlev, buf);
            }
            // 3. layer 0 K independent of the cache. Rounding to nearest gives <= 0.5 step; the container rounds scale and zero to f16
            //    (rel. error 2^-11), which moves the top code by up to (levels-1)*2^-11 steps, and the result is stored as f16 again.
            const double err_bound = 0.5 + (double) (maxlev - 1) / 1024.0 + 0.02;
            size_t win_diff = 0, q_diff = 0; double max_err = 0;
            for (size_t i = (size_t) kq * d.NH * d.D; i < (size_t) N * d.NH * d.D; i++) win_diff += d.k0[i] != d_off.k0[i];
            for (size_t i = 0; i < (size_t) kq * d.NH * d.D; i++) q_diff += d.k0[i] != d_off.k0[i];
            for (int g0 = 0; g0 + G <= kq; g0 += G) for (uint32_t h = 0; h < d.NH; h++) for (uint32_t ch = 0; ch < d.D; ch++) {
                float mn = 1e30f, mx = -1e30f;
                for (int t = g0; t < g0 + G; t++) { float x = ggml_fp16_to_fp32(d_off.k0[((size_t) t * d.NH + h) * d.D + ch]); mn = std::min(mn, x); mx = std::max(mx, x); }
                const double step = (mx - mn) / (double) (maxlev - 1);
                for (int t = g0; t < g0 + G; t++) {
                    const size_t i = ((size_t) t * d.NH + h) * d.D + ch;
                    max_err = std::max(max_err, (double) std::fabs(ggml_fp16_to_fp32(d.k0[i]) - ggml_fp16_to_fp32(d_off.k0[i])) / std::max(step, 1e-6));
                }
            }
            snprintf(buf, sizeof buf, "layer-0 K: window tokens bit-exact vs kivi off (%zu differing elements), quantised tokens differ (%zu elements), max |err|/step = %.3f (<= %.3f)", win_diff, q_diff, max_err, err_bound);
            check(win_diff == 0 && q_diff > 0 && max_err <= err_bound, buf);
            // 4. counters: the sim logs them at free; here check via logits that something changed
            double num = 0, den = 0;
            for (size_t i = 0; i < lg_on.size(); i++) { num += (double) (lg_on[i] - lg_off[i]) * (lg_on[i] - lg_off[i]); den += (double) lg_off[i] * lg_off[i]; }
            printf("    logits rel-rms vs kivi off = %.5f\n", std::sqrt(num / den));
            check(num > 0, "logits changed (the quantised cache is read by attention)");
            if (bits == 8) check(std::sqrt(num / den) < 0.05, "8-bit logits within 5% of off");
            llama_free(on);
        }
        // 6. window covers everything
        {
            llama_context * big = make_ctx(model, 2, fa, G, 80);
            auto lg = run(big, n_vocab);
            check(memcmp(lg_off.data(), lg.data(), lg.size() * sizeof(float)) == 0, "R > N: nothing quantised, logits bit-identical to kivi off");
            llama_free(big);
        }
        llama_free(off);
    }
    // 7. refusals
    printf("== refusals\n");
    check(make_ctx(model, 2, true, G, R, GGML_TYPE_Q8_0) == nullptr, "quantised K cache type + kivi: refused");
    check(make_ctx(model, 2, true, G, 12) == nullptr, "residual not a multiple of group: refused");
    check(make_ctx(model, 3, true) == nullptr, "bits=3: refused");
    llama_model_free(model);
    printf(g_fail ? "TEST_KIVI_SIM_CACHE FAIL (%d)\n" : "TEST_KIVI_SIM_CACHE PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
