// Publishing a finished slot's GENERATED tokens into the --radix-cache tree (tools/server/server-radix-cache.h,
// server_radix_gen_publish_len(), used by the post-decode publish loop in server-context.cpp).
// Row S69-MISTRALRS-PICK-PAGEDATTN-BLOCKPOOL-2026-09-28-01 (absorbs S69-ENGREV-EXLLAMAV2-RADIX-INSERT-GENERATED-TOKENS-2026-10-27-01).
// Part 1 needs no model. Part 2 (the real tree) runs when a tiny GGUF (tests/make_tiny_llama_gguf.py) is given as argv[1] or env
// RADIX_GEN_TEST_MODEL; without it that part prints SKIP, a path that fails to load is a failure.

#include "server-radix-cache.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>

static int g_fail = 0;

#define CHECK(cond, ...) do {                                   \
    if (!(cond)) {                                              \
        printf("  FAIL: %s -- ", #cond);                        \
        printf(__VA_ARGS__);                                    \
        printf("\n");                                           \
        g_fail++;                                               \
    } else {                                                    \
        printf("  ok:   %s\n", #cond);                          \
    }                                                           \
} while (0)

static void part1_len() {
    printf("part 1: server_radix_gen_publish_len\n");
    CHECK(server_radix_gen_publish_len(20, 12, 18, false, false) == 19, "KV holds 19 of 20 tokens (last sampled never decoded)");
    CHECK(server_radix_gen_publish_len(20, 12, 25, false, false) == 20, "KV ahead of the token list: never more than the tokens");
    CHECK(server_radix_gen_publish_len(20, 12, 18, true,  false) == 0,  "a context shift forbids publishing");
    CHECK(server_radix_gen_publish_len(20, 12, 18, false, true)  == 0,  "media chunks forbid publishing");
    CHECK(server_radix_gen_publish_len(20, 12, -1, false, false) == 0,  "an empty KV publishes nothing");
    CHECK(server_radix_gen_publish_len(12, 12, 11, false, false) == 0,  "nothing beyond what the prompt insert published");
    CHECK(server_radix_gen_publish_len(13, 12, 11, false, false) == 0,  "an undecoded generation adds nothing over the prompt");
}

static llama_tokens range_tokens(llama_token from, int n) {
    llama_tokens v;
    for (int i = 0; i < n; i++) {
        v.push_back(from + i);
    }
    return v;
}

static void part2_tree(const char * path) {
    printf("part 2: the tree after a generation publish (kv_unified context)\n");
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) {
        printf("  FAIL: could not open model %s\n", path);
        g_fail++;
        return;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 64; cp.n_ubatch = 64; cp.n_seq_max = 8; cp.kv_unified = true;
    cp.n_threads = 2; cp.n_threads_batch = 2;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        printf("  FAIL: could not create context\n");
        g_fail++;
        llama_model_free(model);
        return;
    }

    const llama_tokens prompt = range_tokens(5, 8);
    llama_tokens full = prompt;
    for (const llama_token x : range_tokens(50, 6)) {
        full.push_back(x);
    }
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, -1, -1, -1);

    llama_batch b = llama_batch_init((int32_t) full.size(), 0, 1);
    for (size_t i = 0; i < full.size(); i++) {
        common_batch_add(b, full[i], (llama_pos) i, { 0 }, i + 1 == full.size());
    }
    const bool ok = llama_decode(ctx, b) == 0;
    llama_batch_free(b);
    CHECK(ok, "decoded prompt + generation into seq 0");

    server_radix_cache radix(ctx, /*seq_id_base*/ 1, /*count*/ 4);
    radix.insert(prompt, 0);
    CHECK(radix.peek_prefix(full, 0) == prompt.size(), "before the publish only the prompt is cached");
    const size_t n_pub = server_radix_gen_publish_len(full.size(), prompt.size(), llama_memory_seq_pos_max(mem, 0), false, false);
    CHECK(n_pub == full.size(), "all %zu tokens are in KV (got %zu)", full.size(), n_pub);
    radix.insert(llama_tokens(full.begin(), full.begin() + n_pub), 0);
    CHECK(radix.peek_prefix(full, 0) == full.size(), "after the publish a replay of prompt + answer matches every token");
    CHECK(radix.peek_prefix(prompt, 0) == prompt.size(), "the prompt itself still matches");

    llama_free(ctx);
    llama_model_free(model);
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : std::getenv("RADIX_GEN_TEST_MODEL");
    printf("test-server-radix-gen-publish\n");
    part1_len();
    if (!path || !*path) {
        printf("part 2: SKIP (no model path: pass argv[1] or set RADIX_GEN_TEST_MODEL)\n");
    } else {
        part2_tree(path);
    }
    printf("%s -- %d failure(s)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
