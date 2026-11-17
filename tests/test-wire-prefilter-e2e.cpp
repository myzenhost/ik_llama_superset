// END-TO-END wiring test for --sampler-topk-prefilter through the real sampler (common_sampler_sample on a real llama_context).
//   argv[1], argv[2] = random-weight GGUFs with vocab 64 and vocab 1024 (tests/make_tiny_wire_gguf.py).
// For many seeds x top-k x temperature x (vocab) a 24-token stream is generated twice from the same decode inputs, flag OFF
// then flag ON (params.sampling.topk_prefilter, what --sampler-topk-prefilter sets). The token streams must be IDENTICAL.
// The ON run must have taken the prefilter path (common_sampler::n_prefiltered > 0) for every eligible configuration,
// the OFF run must have n_prefiltered == 0, and a configuration that is NOT eligible (repeat penalty) must report 0 with the flag ON
// and still match. CONTROL: a deliberately WRONG stream compare (seed shifted by one) must differ, so "identical" can fail.
#include "common.h"
#include "sampling.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

struct cfg { int top_k; float temp; float penalty; };

static std::vector<llama_token> gen(llama_model * model, llama_context * ctx, const cfg & c, uint32_t seed, bool prefilter, uint64_t * n_pre) {
    common_params_sampling sp;
    sp.seed = seed;
    sp.top_k = c.top_k;
    sp.temp = c.temp;
    sp.penalty_repeat = c.penalty;
    sp.topk_prefilter = prefilter;
    common_sampler * gs = common_sampler_init(model, sp);
    llama_kv_cache_clear(ctx);
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> prompt, out;
    for (int i = 0; i < 6; i++) prompt.push_back(4 + (int) ((seed * 13 + i * 7) % (nv - 4)));
    llama_batch pb = llama_batch_get_one(prompt.data(), (int32_t) prompt.size());
    if (llama_decode(ctx, pb) != 0) { common_sampler_free(gs); return out; }
    for (int step = 0; step < 24; step++) {
        llama_token t = common_sampler_sample(gs, ctx, -1);
        common_sampler_accept(gs, ctx, t, true);
        out.push_back(t);
        llama_batch b = llama_batch_get_one(&t, 1);
        if (llama_decode(ctx, b) != 0) break;
    }
    *n_pre = gs->n_prefiltered;
    common_sampler_free(gs);
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 3) { printf("usage: %s v64.gguf v1024.gguf\n", argv[0]); return 2; }
    llama_backend_init();
    int n_streams = 0, n_tokens = 0;
    uint64_t on_total = 0;
    int n_ineligible_ok = 0;
    for (int m = 0; m < 2; m++) {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        llama_model * model = llama_model_load_from_file(argv[1 + m], mp);
        CHECK(model != nullptr);
        if (!model) { continue; }
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 128; cp.n_batch = 32; cp.n_ubatch = 32;
        llama_context * ctx = llama_init_from_model(model, cp);
        CHECK(ctx != nullptr);
        if (!ctx) { llama_model_free(model); continue; }
        const cfg cfgs[] = { {1, 0.8f, 1.0f}, {5, 0.8f, 1.0f}, {20, 1.0f, 1.0f}, {40, 1.5f, 1.0f}, {40, 0.0f, 1.0f}, {20, 0.8f, 1.1f} };
        for (const auto & c : cfgs) {
            const bool eligible = c.penalty == 1.0f;
            for (uint32_t seed = 1; seed <= 12; seed++) {
                uint64_t n_off = 99, n_on = 0;
                const auto off = gen(model, ctx, c, seed, false, &n_off);
                const auto on  = gen(model, ctx, c, seed, true,  &n_on);
                CHECK(off.size() == 24 && on.size() == 24);
                CHECK(off == on);
                CHECK(n_off == 0);
                if (eligible) { CHECK(n_on == 24); } else { CHECK(n_on == 0); n_ineligible_ok++; }
                if (off != on) { printf("  stream differs: vocab-model %d top_k %d temp %.2f seed %u\n", m, c.top_k, c.temp, seed); }
                n_streams++; n_tokens += (int) on.size(); on_total += n_on;
            }
        }
        // control: the comparison can fail -- different seed gives a different stream (temp > 0, top_k 20)
        uint64_t d1 = 0, d2 = 0;
        const auto a = gen(model, ctx, {20, 1.0f, 1.0f}, 1, true, &d1);
        const auto b = gen(model, ctx, {20, 1.0f, 1.0f}, 2, true, &d2);
        CHECK(a != b);
        llama_free(ctx);
        llama_model_free(model);
    }
    printf("streams compared: %d (%d tokens each side), ON-path positions: %llu, ineligible streams checked: %d\n", n_streams, n_tokens,
           (unsigned long long) on_total, n_ineligible_ok);
    printf("test-wire-prefilter-e2e: %s (%d failed)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
