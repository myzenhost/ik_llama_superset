// END-TO-END wiring test for the turbo KV cache types on the CPU, through the public llama API (the path llama-cli /
// llama-server take: llama_context_params.type_k/type_v + flash_attn, then llama_decode).
//   argv[1] = random-weight GGUF (tests/make_tiny_wire_gguf.py, head size 128).
// For each cache type: decode a prompt plus a continuation, keep the logits, compare with the f16 cache.
//   * f16 twice -> bit identical (the OFF path is deterministic, so a difference elsewhere is the cache type)
//   * q8_0 control -> small error (shows the model is not so sensitive that any quantised cache fails)
//   * turbo4 / turbo3 / turbo2 -> relative L2 error of the logits under the limits below, nonzero (the quantised path RAN),
//     and the info line "TurboQuant KV cache enabled: K=<type>" is seen in the log (the log line naming the type).
// Limits are stated BEFORE the first run: q8_0 0.02, turbo4 0.15, turbo3 0.30, turbo2 0.60 (relative L2 over all logits).
// CONTROLS: turbo with flash attention off must be refused (nullptr context); the top-1 token agreement is printed, not asserted.
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

static std::string g_log;
static void log_cb(ggml_log_level, const char * text, void *) { g_log += text; }

struct run_result { bool ok = false; std::vector<float> logits; };

static run_result run(llama_model * model, ggml_type tk, ggml_type tv, bool fa) {
    run_result r;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 64; cp.n_ubatch = 64;
    cp.type_k = tk; cp.type_v = tv;
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.flash_attn = fa;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { return r; }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> prompt;
    for (int i = 0; i < 40; i++) prompt.push_back(4 + (i * 7) % (nv - 4));
    llama_batch pb = llama_batch_get_one(prompt.data(), (int32_t) prompt.size());
    if (llama_decode(ctx, pb) != 0) { llama_free(ctx); return r; }
    const float * l = llama_get_logits_ith(ctx, -1);
    r.logits.insert(r.logits.end(), l, l + nv);
    for (int i = 0; i < 8; i++) {
        llama_token t = 4 + (i * 11 + 3) % (nv - 4);
        llama_batch b = llama_batch_get_one(&t, 1);
        if (llama_decode(ctx, b) != 0) { llama_free(ctx); return r; }
        l = llama_get_logits_ith(ctx, -1);
        r.logits.insert(r.logits.end(), l, l + nv);
    }
    r.ok = true;
    llama_free(ctx);
    return r;
}

static double rel_err(const std::vector<float> & a, const std::vector<float> & ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); i++) { const double d = (double) a[i] - ref[i]; num += d * d; den += (double) ref[i] * ref[i]; }
    return std::sqrt(num / den);
}

static double top1_agree(const std::vector<float> & a, const std::vector<float> & ref, int nv) {
    int ok = 0, n = (int) (ref.size() / nv);
    for (int s = 0; s < n; s++) {
        int ia = 0, ir = 0;
        for (int i = 1; i < nv; i++) { if (a[s*nv+i] > a[s*nv+ia]) ia = i; if (ref[s*nv+i] > ref[s*nv+ir]) ir = i; }
        ok += ia == ir;
    }
    return (double) ok / n;
}

int main(int argc, char ** argv) {
    if (argc < 2) { printf("usage: %s model.gguf\n", argv[0]); return 2; }
    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { printf("cannot load %s\n", argv[1]); return 2; }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));

    const auto f16a = run(model, GGML_TYPE_F16, GGML_TYPE_F16, true);
    const auto f16b = run(model, GGML_TYPE_F16, GGML_TYPE_F16, true);
    CHECK(f16a.ok && f16b.ok);
    CHECK(f16a.logits.size() == f16b.logits.size() && std::memcmp(f16a.logits.data(), f16b.logits.data(), f16a.logits.size() * 4) == 0);

    struct row { const char * name; ggml_type t; double lim; bool turbo; };
    const row rows[] = {
        { "q8_0",   GGML_TYPE_Q8_0,    0.02, false },
        { "turbo4", GGML_TYPE_TURBO4_0, 0.15, true  },
        { "turbo3", GGML_TYPE_TURBO3_0, 0.30, true  },
        { "turbo2", GGML_TYPE_TURBO2_0, 0.60, true  },
    };
    for (const auto & rw : rows) {
        g_log.clear();
        const auto r = run(model, rw.t, rw.t, true);
        CHECK(r.ok);
        if (!r.ok) { printf("%-7s run FAILED\n  log tail: %s\n", rw.name, g_log.substr(g_log.size() > 600 ? g_log.size() - 600 : 0).c_str()); continue; }
        const double e = rel_err(r.logits, f16a.logits);
        printf("%-7s rel-L2 err vs f16 cache = %.5f (limit %.2f), top-1 agreement = %.3f\n", rw.name, e, rw.lim, top1_agree(r.logits, f16a.logits, nv));
        CHECK(e < rw.lim);
        CHECK(e > 0.0); // the quantised cache really was used
        const std::string want = std::string("TurboQuant KV cache enabled: K=") + rw.name;
        CHECK((g_log.find(want) != std::string::npos) == rw.turbo);
    }

    // control: turbo K/V needs flash attention; without it the context must be refused
    g_log.clear();
    const auto nofa = run(model, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, false);
    CHECK(!nofa.ok);

    llama_model_free(model);
    printf("test-wire-runtime-e2e: %s (%d failed)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
