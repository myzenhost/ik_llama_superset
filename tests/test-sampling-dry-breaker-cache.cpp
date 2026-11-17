// DRY breaker table cache. Row S69-ENGREV-EXLLAMAV2-DRY-BREAKER-PRECOMPUTE-2026-10-27-01.
// Building the breaker table scans the whole vocab per breaker string. Two samplers with the same vocab and the same breakers must
// build it once (second is a hit); different breakers must build again; and a cached table must give the same penalties as a fresh one.
// Needs models/ggml-vocab-qwen2.gguf (WORKING_DIRECTORY = source root).
#include "../src/llama-sampling.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fails; std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while (0)

static llama_sampler_dry * mk(const llama_vocab * v, const char ** br, size_t n) {
    return llama_sampler_dry_init(v, 0.8f, 1.75f, 2, 64, br, n);
}

// feed a repeating pattern, return the penalty-applied logit of the next pattern token
static float run(llama_sampler_dry * s, const std::vector<llama_token> & pat, int n_vocab) {
    for (int r = 0; r < 3; ++r) {
        for (auto t : pat) { llama_sampler_dry_accept(s, t); }
    }
    llama_sampler_dry_accept(s, pat[0]);
    std::vector<llama_token_data> d(n_vocab);
    for (int i = 0; i < n_vocab; ++i) { d[i] = { i, 1.0f, 0.0f }; }
    llama_token_data_array arr = { d.data(), d.size(), -1, false };
    llama_sampler_dry_apply(s, &arr);
    return d[pat[1]].logit;
}

int main() {
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * m = llama_model_load_from_file("models/ggml-vocab-qwen2.gguf", mp);
    if (!m) { std::printf("FAIL: cannot load vocab\n"); return 1; }
    const llama_vocab * v = llama_model_get_vocab(m);
    const int nv = llama_vocab_n_tokens(v);

    const char * b1[] = { "\n", ":", "\"", "*" };
    const char * b2[] = { "\n", ":", "\"" };
    size_t h0, m0, h1, m1, h2, m2, h3, m3;
    llama_dry_breaker_cache_stats(&h0, &m0);
    llama_sampler_dry * a = mk(v, b1, 4);
    llama_dry_breaker_cache_stats(&h1, &m1);
    llama_sampler_dry * b = mk(v, b1, 4);
    llama_dry_breaker_cache_stats(&h2, &m2);
    llama_sampler_dry * c = mk(v, b2, 3);
    llama_dry_breaker_cache_stats(&h3, &m3);

    CHECK(m1 - m0 == 1 && h1 - h0 == 0, "first build: want 1 miss 0 hit, got %zu miss %zu hit", m1 - m0, h1 - h0);
    CHECK(h2 - h1 == 1 && m2 - m1 == 0, "same breakers again: want 1 hit 0 miss, got %zu hit %zu miss", h2 - h1, m2 - m1);
    CHECK(m3 - m2 == 1, "different breakers: want a new build, got %zu miss", m3 - m2);

    // same behaviour from the cached table as from the fresh one
    std::vector<llama_token> pat = { 1000, 1001, 1002, 1003 };
    const float la = run(a, pat, nv), lb = run(b, pat, nv);
    CHECK(la == lb, "cached sampler penalty %f != fresh %f", lb, la);
    CHECK(la < 1.0f, "penalty not applied (logit %f)", la);

    // multiplier 0 builds no table at all
    size_t hh, mm;
    llama_dry_breaker_cache_stats(&hh, &mm);
    llama_sampler_dry * d = llama_sampler_dry_init(v, 0.0f, 1.75f, 2, 64, b1, 4);
    size_t hh2, mm2;
    llama_dry_breaker_cache_stats(&hh2, &mm2);
    CHECK(hh2 == hh && mm2 == mm, "zero-multiplier DRY touched the cache");

    llama_sampler_dry_free(a); llama_sampler_dry_free(b); llama_sampler_dry_free(c); llama_sampler_dry_free(d);
    llama_model_free(m);
    std::printf(g_fails ? "FAILED %d\n" : "OK\n", g_fails);
    return g_fails ? 1 : 0;
}
