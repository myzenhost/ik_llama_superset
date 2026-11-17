// LoRA-configuration key for the server's two prefix caches (tools/server/server-task.{h,cpp},
// server-radix-cache.{h,cpp}, server-common.{h,cpp}).
//
// Row S69-LORAX-PICK-LORA-CONFIG-IN-PROMPT-CACHE-AND-RADIX-KEY-2026-09-29-01. Idea from lorax
// (predibase/lorax router/src/radix.rs keeps one prefix tree root per adapter, Apache-2.0); written here against
// our own data structures, no code shared.
//
// Why: KV computed for tokens P under one adapter configuration is not the KV of P under another, but both caches
// used to match on tokens alone.
//
// Parts:
//   1. fingerprint helper properties            -- no model
//   2. server_prompt_cache::find_best / alloc   -- no model (think-token exclusion off, so no context is read)
//   3. server_prompt_cache::load() end to end   -- needs a tiny GGUF (argv[1] or env LORAKEY_TEST_MODEL)
//   4. server_radix_cache                       -- same model, kv_unified context
// Parts 3 and 4 print SKIP when no model path is given; a path that fails to load counts as a failure.
//
// Building WITHOUT -DLORAKEY_NEW_API compiles the same scenarios against the pre-change API (no fingerprint
// parameter anywhere) so the old behaviour can be measured: there the miss expectations cannot be written, and the
// test asserts what the old API does (restore / hit under any adapter configuration).

#include "server-task.h"
#include "server-common.h"
#include "server-radix-cache.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

#ifdef LORAKEY_NEW_API
static common_adapter_lora_info make_lora(uintptr_t fake_ptr, float scale) {
    common_adapter_lora_info a;
    a.ptr   = reinterpret_cast<llama_adapter_lora *>(fake_ptr);
    a.scale = scale;
    return a;
}

static void part1_fingerprint() {
    printf("part 1: fingerprint helper\n");
    using L = std::vector<common_adapter_lora_info>;
    const L none;
    const L zero_only = { make_lora(0x1000, 0.0f) };
    const L a1        = { make_lora(0x1000, 1.0f) };
    const L a1_b0     = { make_lora(0x1000, 1.0f), make_lora(0x2000, 0.0f) };
    const L a05       = { make_lora(0x1000, 0.5f) };
    const L b1        = { make_lora(0x2000, 1.0f) };
    const L a1_b1     = { make_lora(0x1000, 1.0f), make_lora(0x2000, 1.0f) };

    CHECK(lora_config_fingerprint(none) == 0, "no adapters must be fingerprint 0 (the pre-change key)");
    CHECK(lora_config_fingerprint(zero_only) == 0, "an adapter at scale 0 is not active, so it is fingerprint 0");
    CHECK(lora_config_fingerprint(a1) != 0, "an active adapter must not collide with 'no adapter'");
    CHECK(lora_config_fingerprint(a1) == lora_config_fingerprint(a1_b0), "an inactive extra adapter must not change the key");
    CHECK(lora_config_fingerprint(a1) != lora_config_fingerprint(a05), "different scale, different key");
    CHECK(lora_config_fingerprint(a1) != lora_config_fingerprint(b1), "different adapter, different key");
    CHECK(lora_config_fingerprint(a1) != lora_config_fingerprint(a1_b1), "a second active adapter changes the key");
    CHECK(lora_config_fingerprint(a1_b1) != lora_config_fingerprint(b1), "a1+b1 differs from b1");
    CHECK(lora_config_fingerprint(a1) == lora_config_fingerprint(a1), "deterministic");
}

// Row S69-VLLM-PICK-CONTROL-VECTOR-CACHE-INVALIDATION-2026-09-29-01: the control vector a context runs under is part of
// the key, for the same reason an adapter is (KV computed under another steering vector is not reusable).
static void part1b_control_vector_key() {
    printf("part 1b: control-vector fingerprint folded into the cache key\n");
    const float v1[] = { 0.5f, -1.0f, 2.0f, 0.25f };
    float       v2[] = { 0.5f, -1.0f, 2.0f, 0.25f };
    v2[2] = 2.5f;
    const uint64_t a = control_vector_fingerprint(v1, 4, 2, 1, 3);

    CHECK(control_vector_fingerprint(nullptr, 0, 0, 0, 0) == 0, "no vector is fingerprint 0 (the pre-change key)");
    CHECK(control_vector_fingerprint(v1, 0, 2, 1, 3) == 0, "an empty vector is fingerprint 0");
    CHECK(a != 0, "an active vector must not collide with 'no vector'");
    CHECK(a == control_vector_fingerprint(v1, 4, 2, 1, 3), "deterministic");
    CHECK(a != control_vector_fingerprint(v2, 4, 2, 1, 3), "one changed coefficient changes the key");
    CHECK(a != control_vector_fingerprint(v1, 4, 2, 1, 4), "a different layer range changes the key");
    CHECK(a != control_vector_fingerprint(v1, 4, 2, 0, 3), "a different layer start changes the key");

    const uint64_t lora = lora_config_fingerprint({ make_lora(0x1000, 1.0f) });
    CHECK(kv_config_fingerprint(0, 0) == 0, "no adapter and no vector: key 0, exactly as before");
    CHECK(kv_config_fingerprint(lora, 0) == lora, "no vector leaves the adapter key untouched");
    CHECK(kv_config_fingerprint(0, a) != 0, "a vector alone gives a nonzero key");
    CHECK(kv_config_fingerprint(0, a) != kv_config_fingerprint(0, control_vector_fingerprint(v2, 4, 2, 1, 3)),
          "two different vectors give two different keys");
    CHECK(kv_config_fingerprint(lora, a) != lora, "an adapter plus a vector differs from the adapter alone");
    CHECK(kv_config_fingerprint(lora, a) != kv_config_fingerprint(0, a), "an adapter plus a vector differs from the vector alone");
}

static server_prompt make_prompt(const llama_tokens & toks, uint64_t fp) {
    server_prompt p { server_tokens(toks, false), {} };
    p.lora_fp = fp;
    return p;
}

static void part2_cache_select() {
    printf("part 2: prompt cache selection by fingerprint (no model)\n");
    const uint64_t X = 0x1111, Y = 0x2222;
    const llama_tokens P = { 1, 2, 3, 4, 5, 6, 7, 8 };
    thinking_tokens think; think.exclude = false;

    server_prompt_cache cache(64, 0);
    server_prompt empty_slot { server_tokens(llama_tokens{}, false), {} };

    auto * st = cache.alloc(make_prompt(P, X), 1024, 0);
    CHECK(st != nullptr, "alloc under X stored");
    CHECK(st && st->prompt.lora_fp == X, "the entry records its fingerprint");

    const server_tokens req(P, false);
    CHECK(cache.find_best(empty_slot, req, nullptr, think, X) != cache.states.end(), "same fingerprint hits");
    CHECK(cache.find_best(empty_slot, req, nullptr, think, Y) == cache.states.end(), "other fingerprint misses");
    CHECK(cache.find_best(empty_slot, req, nullptr, think, 0)  == cache.states.end(), "no-adapter request misses an adapter entry");

    // the same tokens under a second fingerprint must be stored as a second entry, not skipped as 'already cached'
    auto * st2 = cache.alloc(make_prompt(P, Y), 1024, 0);
    CHECK(st2 != nullptr, "alloc of the same tokens under Y is not deduplicated against X");
    CHECK(cache.states.size() == 2, "two entries (one per fingerprint), got %zu", cache.states.size());
    // an extension under X replaces only the X entry
    llama_tokens P2 = P; P2.push_back(9);
    auto * st3 = cache.alloc(make_prompt(P2, X), 1024, 0);
    CHECK(st3 != nullptr, "extension under X stored");
    CHECK(cache.states.size() == 2, "extension under X removed the X prefix entry only (Y entry kept), got %zu", cache.states.size());
    bool y_kept = false;
    for (const auto & s : cache.states) { if (s.prompt.lora_fp == Y && s.prompt.tokens.size() == P.size()) y_kept = true; }
    CHECK(y_kept, "the Y entry survived an X extension");

    // empty fingerprint = old behaviour
    server_prompt_cache plain(64, 0);
    CHECK(plain.alloc(make_prompt(P, 0), 1024, 0) != nullptr, "alloc, no adapters");
    CHECK(plain.alloc(make_prompt(P, 0), 1024, 0) == nullptr, "identical no-adapter prompt is deduplicated as before");
    CHECK(plain.find_best(empty_slot, req, nullptr, think, 0) != plain.states.end(), "no-adapter request hits no-adapter entry");

    // the slot's own prompt (different fingerprint) must not act as the baseline an entry has to beat
    server_prompt_cache c2(64, 0);
    c2.alloc(make_prompt(llama_tokens{1, 2, 3, 4}, X), 1024, 0);
    server_prompt own = make_prompt(P, Y);   // slot holds the full prompt, but under Y
    CHECK(c2.find_best(own, req, nullptr, think, X) != c2.states.end(),
          "an entry under the requested fingerprint beats a slot prompt held under another one");
    server_prompt own_same = make_prompt(P, X);
    CHECK(c2.find_best(own_same, req, nullptr, think, X) == c2.states.end(),
          "a slot prompt under the requested fingerprint that is already better is kept");
}
#endif // LORAKEY_NEW_API

struct tiny {
    llama_model *   model = nullptr;
    llama_context * ctx   = nullptr;
};

static bool tiny_open(tiny & t, const char * path) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    t.model = llama_model_load_from_file(path, mp);
    if (!t.model) {
        return false;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = 256;
    cp.n_batch    = 64;
    cp.n_ubatch   = 64;
    cp.n_seq_max  = 4;
    cp.kv_unified = true;
    cp.n_threads       = 2;
    cp.n_threads_batch = 2;
    t.ctx = llama_init_from_model(t.model, cp);
    return t.ctx != nullptr;
}

static bool decode_seq(llama_context * ctx, const llama_tokens & toks, llama_seq_id seq) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); i++) {
        common_batch_add(b, toks[i], (llama_pos) i, { seq }, i + 1 == toks.size());
    }
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc == 0;
}

static llama_pos seq_max(llama_context * ctx, llama_seq_id seq) {
    return llama_memory_seq_pos_max(llama_get_memory(ctx), seq);
}

static void part3_load(tiny & t) {
    printf("part 3: server_prompt_cache::load() with a real context\n");
    const llama_tokens P = { 5, 6, 7, 8, 9, 10, 11, 12 };
    thinking_tokens think; think.exclude = false;
    CHECK(decode_seq(t.ctx, P, 0), "decode P into seq 0");

    const size_t sz = llama_state_seq_get_size_ext(t.ctx, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
    CHECK(sz > 0, "state size %zu", sz);

    const uint64_t X = 0x1111, Y = 0x2222;
    const server_tokens req(P, false);

    auto save = [&](server_prompt_cache & cache, uint64_t fp) {
        server_prompt p { server_tokens(P, false), {} };
#ifdef LORAKEY_NEW_API
        p.lora_fp = fp;
#else
        (void) fp;
#endif
        auto * st = cache.alloc(p, sz, 0);
        if (st) {
            llama_state_seq_get_data_ext(t.ctx, st->data.main.data(), sz, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
        }
        return st != nullptr;
    };
    auto try_load = [&](server_prompt_cache & cache, llama_seq_id dst, uint64_t fp, server_prompt & out) {
        llama_memory_seq_rm(llama_get_memory(t.ctx), dst, -1, -1);
        out = server_prompt { server_tokens(llama_tokens{}, false), {} };
#ifdef LORAKEY_NEW_API
        const bool ok = cache.load(out, req, t.ctx, nullptr, dst, think, fp);
#else
        (void) fp;
        const bool ok = cache.load(out, req, t.ctx, nullptr, dst, think);
#endif
        return ok;
    };

    server_prompt_cache cache(64, 0);
    CHECK(save(cache, X), "saved P computed under fingerprint X");

    server_prompt out { server_tokens(llama_tokens{}, false), {} };
#ifdef LORAKEY_NEW_API
    bool ok = try_load(cache, 1, Y, out);
    CHECK(ok, "load under Y returns true (a miss is not an error)");
    CHECK(out.tokens.size() == 0, "load under Y != X left the destination prompt untouched (miss), got %zu tokens", out.tokens.size());
    // the legacy-path memory answers pos_max of an empty sequence as 0 (not -1), so compare against the restored extent
    CHECK(seq_max(t.ctx, 1) < (llama_pos) P.size() - 1, "no KV was restored into the destination sequence on a miss, pos_max %d", (int) seq_max(t.ctx, 1));

    ok = try_load(cache, 1, X, out);
    CHECK(ok, "load under X succeeds");
    CHECK(out.tokens.size() == P.size(), "load under X restored %zu tokens", out.tokens.size());
    CHECK(seq_max(t.ctx, 1) == (llama_pos) P.size() - 1, "restored KV covers the prompt, pos_max %d", (int) seq_max(t.ctx, 1));
    CHECK(out.lora_fp == X, "restored prompt carries the fingerprint X");

    ok = try_load(cache, 1, 0, out);
    CHECK(ok && out.tokens.size() == 0, "a no-adapter request does not restore an adapter entry");

    // no-adapter entries behave as before
    server_prompt_cache plain(64, 0);
    CHECK(save(plain, 0), "saved P with no adapters");
    ok = try_load(plain, 1, 0, out);
    CHECK(ok && out.tokens.size() == P.size(), "no-adapter entry restored for a no-adapter request");
    ok = try_load(plain, 1, X, out);
    CHECK(ok && out.tokens.size() == 0, "no-adapter entry is not restored for an adapter request");
    (void) Y;
#else
    (void) X; (void) Y;
    bool ok = try_load(cache, 1, Y, out);
    printf("  OLD API: load() has no fingerprint parameter; entry was computed under 'X' but the request is under 'Y'\n");
    CHECK(ok && out.tokens.size() == P.size(),
          "OLD BEHAVIOUR: entry restored regardless of adapter configuration (%zu tokens restored)", out.tokens.size());
#endif
}

static void part4_radix(tiny & t) {
    printf("part 4: server_radix_cache with a real kv_unified context\n");
    const llama_tokens P = { 5, 6, 7, 8, 9, 10, 11, 12 };
    llama_memory_seq_rm(llama_get_memory(t.ctx), -1, -1, -1);
    CHECK(decode_seq(t.ctx, P, 0), "decode P into seq 0");

    server_radix_cache radix(t.ctx, /*seq_id_base*/ 2, /*count*/ 2);
#ifdef LORAKEY_NEW_API
    const uint64_t X = 0x1111, Y = 0x2222;
    radix.insert(P, 0, X);
    CHECK(radix.match_prefix(P, Y) == 0, "prefix inserted under X is not matched under Y");
    CHECK(radix.match_prefix(P, 0) == 0, "prefix inserted under X is not matched by a no-adapter request");
    const size_t n = radix.match_prefix(P, X);
    CHECK(n == P.size(), "prefix inserted under X is matched under X (%zu)", n);
    radix.commit(1, n);
    CHECK(seq_max(t.ctx, 1) == (llama_pos) P.size() - 1, "committed KV covers the prefix, pos_max %d", (int) seq_max(t.ctx, 1));

    // both configurations can hold the same tokens side by side (one tree root per fingerprint)
    radix.insert(P, 0, Y);
    CHECK(radix.match_prefix(P, Y) == P.size(), "the same tokens inserted under Y match under Y");
    radix.release();
    CHECK(radix.match_prefix(P, X) == P.size(), "and still match under X");
    radix.release();
    CHECK(radix.n_nodes() == 2, "one node per fingerprint, got %zu", radix.n_nodes());

    // no-adapter behaviour as before
    radix.clear();
    radix.insert(P, 0);
    CHECK(radix.match_prefix(P) == P.size(), "default fingerprint: insert/match without an argument works as before");
    radix.release();
#else
    radix.insert(P, 0);
    printf("  OLD API: insert()/match_prefix() have no fingerprint parameter\n");
    const size_t n = radix.match_prefix(P);
    CHECK(n == P.size(), "OLD BEHAVIOUR: prefix matched for any adapter configuration (%zu tokens)", n);
    radix.release();
#endif
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : std::getenv("LORAKEY_TEST_MODEL");
#ifdef LORAKEY_NEW_API
    printf("test-server-lora-cache-key: NEW API\n");
    part1_fingerprint();
    part1b_control_vector_key();
    part2_cache_select();
#else
    printf("test-server-lora-cache-key: OLD API (pre-change shim)\n");
#endif
    if (!path || !*path) {
        printf("parts 3-4: SKIP (no model path: pass argv[1] or set LORAKEY_TEST_MODEL)\n");
    } else {
        tiny t;
        if (!tiny_open(t, path)) {
            printf("  FAIL: could not open model %s\n", path);
            g_fail++;
        } else {
            part3_load(t);
            part4_radix(t);
            llama_free(t.ctx);
            llama_model_free(t.model);
        }
    }
    printf("%s -- %d failure(s)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
