// kv-dedup-check: the memory module of a legacy-graph model must hold NO K/V data (metadata only), kv_self holds the
// one real allocation, and decoding is unchanged.
//
// Row S69-KV-DEDUP-METADATA-ONLY-MEMORY-MODULE-2026-09-29-01. Before this change a legacy-graph model built the upstream
// shaped `memory` module (llama_kv_cache) AND the legacy kv_self, both with full K/V buffers; every graph reads only
// kv_self. The memory module is now created with llama_memory_params::metadata_only (zero-size dummy buffers, all cell
// bookkeeping live).
//
// The same process builds the context twice on one loaded model, toggling LLAMA_KV_MEMORY_METADATA_ONLY (0 = the old
// behaviour, unset = metadata only), and checks
//   A. bytes the memory module reports: OFF run > 0 (control: the check can see a second allocation),
//                                        ON  run == 0
//   B. bytes held by kv_self: > 0 and equal across the two runs (kv_self is the one real allocation)
//   C. the load log: the `llama_kv_cache:` KV buffer line reads 0.00 MiB in the ON run (nonzero in the OFF run)
//   D. decoding: a scripted multi-slot run gives IDENTICAL tokens ON vs OFF. The script covers what reaches the memory
//      module's bookkeeping: two sequences in one batch, prompt-prefix reuse by llama_memory_seq_cp, truncation by
//      llama_memory_seq_rm and re-decode, and a context shift (llama_memory_seq_rm + llama_memory_seq_add).
//      Also: within a run, the copied-prefix sequence and the truncate/re-decode sequence match a fresh decode.
//
// Exit code: 0 pass, 1 fail, 2 setup error.
// usage: llama-kv-dedup-check -m model.gguf [-c n_ctx]

#include "llama.h"
#include "../../src/llama-ext.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#  define SETENV(k, v) _putenv_s(k, v)
#  define UNSETENV(k)  _putenv_s(k, "")
#else
#  define SETENV(k, v) setenv(k, v, 1)
#  define UNSETENV(k)  unsetenv(k)
#endif

static std::vector<std::string> g_kv_lines;

static void log_cb(ggml_log_level level, const char * text, void * user) {
    (void) level; (void) user;
    if (strstr(text, "KV buffer size") != nullptr) {
        std::string s(text);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        g_kv_lines.push_back(s);
    }
}

static llama_token argmax_ith(llama_context * ctx, int i, int n_vocab) {
    const float * l = llama_get_logits_ith(ctx, i);
    llama_token t = 0;
    for (int k = 1; k < n_vocab; ++k) if (l[k] > l[t]) t = k;
    return t;
}

struct entry { llama_token tok; llama_pos pos; llama_seq_id seq; bool logits; };

static int decode_entries(llama_context * ctx, const std::vector<entry> & es) {
    llama_batch b = llama_batch_init((int32_t) es.size(), 0, 1);
    for (size_t i = 0; i < es.size(); ++i) {
        b.token[i]     = es[i].tok;
        b.pos[i]       = es[i].pos;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = es[i].seq;
        b.logits[i]    = es[i].logits;
    }
    b.n_tokens = (int32_t) es.size();
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc;
}

static std::vector<llama_token> tokenize(llama_model * model, const std::string & s, bool add_special) {
    std::vector<llama_token> t(s.size() + 16);
    const int n = llama_tokenize(model, s.c_str(), (int32_t) s.size(), t.data(), (int32_t) t.size(), add_special, false);
    if (n < 0) { t.clear(); return t; }
    t.resize(n);
    return t;
}

// decode a run of tokens on one sequence starting at pos0; logits on the last one. returns argmax of the last position
static bool decode_run(llama_context * ctx, llama_seq_id seq, const std::vector<llama_token> & toks, int pos0, int n_vocab, llama_token & next) {
    std::vector<entry> es;
    for (size_t i = 0; i < toks.size(); ++i) es.push_back({ toks[i], pos0 + (llama_pos) i, seq, i + 1 == toks.size() });
    if (decode_entries(ctx, es) != 0) return false;
    next = argmax_ith(ctx, -1, n_vocab);
    return true;
}

struct run_result {
    bool ok = false;
    size_t module_bytes = 0;
    size_t kv_self_bytes = 0;
    std::vector<std::string> kv_lines;
    std::vector<llama_token> stream; // every generated token, phases concatenated with -1 separators
    std::vector<llama_token> fresh_ref, via_cp, via_rm; // the three continuations that must agree
    std::vector<llama_token> shift_tokens;
};

static run_result run_once(llama_model * model, int n_ctx, bool metadata_only, int n_vocab) {
    run_result r;
    if (metadata_only) UNSETENV("LLAMA_KV_MEMORY_METADATA_ONLY"); else SETENV("LLAMA_KV_MEMORY_METADATA_ONLY", "0");

    g_kv_lines.clear();
    llama_log_set(log_cb, nullptr);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = n_ctx;
    cp.n_batch   = 256;
    cp.n_ubatch  = 256;
    cp.n_seq_max = 3;
    // unified: llama_kv_cache::seq_cp of a PARTIAL range asserts in per-sequence-stream mode (the memory module
    // enqueues a stream copy; MEASURED 2026-09-29, llama-kv-cache.cpp:574, on the base commit's code path too).
    cp.kv_unified = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    llama_log_set(nullptr, nullptr);
    if (!ctx) { fprintf(stderr, "SETUP: context creation failed\n"); return r; }
    r.kv_lines = g_kv_lines;

    r.module_bytes  = llama_get_memory_module_bytes(ctx);
    r.kv_self_bytes = llama_get_kv_self_bytes(ctx);

    llama_memory_t mem = llama_get_memory(ctx);

    const std::string text = "The committee decided that the budget deadline had passed because the treasurer failed to file the report within two weeks of the meeting, and the extension rule did not apply";
    const std::vector<llama_token> P = tokenize(model, text, true);
    if (P.size() < 12) { fprintf(stderr, "SETUP: prompt too short (%zu)\n", P.size()); llama_free(ctx); return r; }
    const int n0 = (int) P.size();
    const int n_reuse = n0 - 4;                       // prefix shared through seq_cp
    const std::vector<llama_token> alt = tokenize(model, " but the review was postponed", false);
    std::vector<llama_token> P1(P.begin(), P.begin() + n_reuse);
    P1.insert(P1.end(), alt.begin(), alt.end());      // seq 1 = shared prefix + a different suffix
    const int n1 = (int) P1.size();
    const int N_GEN = 20;

    // ---- phase 1: fresh decode of seq 0 alone -> reference continuation
    llama_token next0 = 0;
    if (!decode_run(ctx, 0, P, 0, n_vocab, next0)) { fprintf(stderr, "decode failed (phase 1)\n"); llama_free(ctx); return r; }
    {
        llama_token t = next0;
        for (int i = 0; i < N_GEN; ++i) {
            r.fresh_ref.push_back(t);
            if (!decode_run(ctx, 0, { t }, n0 + i, n_vocab, t)) { fprintf(stderr, "decode failed (phase 1 gen)\n"); llama_free(ctx); return r; }
        }
    }
    r.stream.insert(r.stream.end(), r.fresh_ref.begin(), r.fresh_ref.end());
    r.stream.push_back(-1);

    // ---- phase 2: truncate seq 0 back to the prompt (seq_rm) and re-decode the last prompt token: same continuation
    llama_memory_seq_rm(mem, 0, n0 - 1, -1);
    {
        llama_token t = 0;
        if (!decode_run(ctx, 0, { P[n0 - 1] }, n0 - 1, n_vocab, t)) { fprintf(stderr, "decode failed (phase 2)\n"); llama_free(ctx); return r; }
        for (int i = 0; i < N_GEN; ++i) {
            r.via_rm.push_back(t);
            if (!decode_run(ctx, 0, { t }, n0 + i, n_vocab, t)) { fprintf(stderr, "decode failed (phase 2 gen)\n"); llama_free(ctx); return r; }
        }
    }
    r.stream.insert(r.stream.end(), r.via_rm.begin(), r.via_rm.end());
    r.stream.push_back(-1);

    // ---- phase 3: two slots in ONE batch. seq 0 continues; seq 1 reuses seq 0's prompt prefix by seq_cp and adds its own suffix
    llama_memory_seq_rm(mem, 0, n0, -1);
    llama_memory_seq_cp(mem, 0, 1, 0, n_reuse);
    llama_token next1 = 0;
    {
        std::vector<llama_token> suffix(P1.begin() + n_reuse, P1.end());
        if (!decode_run(ctx, 1, suffix, n_reuse, n_vocab, next1)) { fprintf(stderr, "decode failed (phase 3 suffix)\n"); llama_free(ctx); return r; }
        // seq 0 logits again (its last prompt position is still cached; re-derive next0 by decoding nothing new is not possible, so re-decode it)
        llama_memory_seq_rm(mem, 0, n0 - 1, -1);
        if (!decode_run(ctx, 0, { P[n0 - 1] }, n0 - 1, n_vocab, next0)) { fprintf(stderr, "decode failed (phase 3 re-decode)\n"); llama_free(ctx); return r; }
    }
    {
        llama_token t0 = next0, t1 = next1;
        std::vector<llama_token> g0, g1;
        for (int i = 0; i < N_GEN; ++i) {
            g0.push_back(t0); g1.push_back(t1);
            std::vector<entry> es = { { t0, n0 + i, 0, true }, { t1, n1 + i, 1, true } };
            if (decode_entries(ctx, es) != 0) { fprintf(stderr, "decode failed (phase 3 batch)\n"); llama_free(ctx); return r; }
            t0 = argmax_ith(ctx, 0, n_vocab);
            t1 = argmax_ith(ctx, 1, n_vocab);
        }
        r.via_cp = g0;   // seq 0 alongside seq 1 in a shared batch must still match the reference
        r.stream.insert(r.stream.end(), g0.begin(), g0.end()); r.stream.push_back(-1);
        r.stream.insert(r.stream.end(), g1.begin(), g1.end()); r.stream.push_back(-1);
    }
    llama_memory_seq_rm(mem, 0, -1, -1);
    llama_memory_seq_rm(mem, 1, -1, -1);

    // ---- phase 4: context shift on a sequence that shares no cells (seq 2)
    if (llama_memory_can_shift(mem)) {
        const int n_keep = 4, n_gen_pre = 40, n_gen_post = 30;
        llama_token t = 0;
        if (!decode_run(ctx, 2, P, 0, n_vocab, t)) { fprintf(stderr, "decode failed (phase 4)\n"); llama_free(ctx); return r; }
        int pos = n0;
        for (int i = 0; i < n_gen_pre; ++i) {
            r.shift_tokens.push_back(t);
            if (!decode_run(ctx, 2, { t }, pos++, n_vocab, t)) { fprintf(stderr, "decode failed (phase 4 gen)\n"); llama_free(ctx); return r; }
        }
        const int n_discard = (pos - n_keep) / 2;
        llama_memory_seq_rm (mem, 2, n_keep, n_keep + n_discard);
        llama_memory_seq_add(mem, 2, n_keep + n_discard, -1, -n_discard);
        pos -= n_discard;
        // the token in `t` was sampled before the shift; feed it at the shifted position
        for (int i = 0; i < n_gen_post; ++i) {
            r.shift_tokens.push_back(t);
            if (!decode_run(ctx, 2, { t }, pos++, n_vocab, t)) { fprintf(stderr, "decode failed (phase 4 post-shift)\n"); llama_free(ctx); return r; }
        }
        r.stream.insert(r.stream.end(), r.shift_tokens.begin(), r.shift_tokens.end());
        r.stream.push_back(-1);
    } else {
        fprintf(stderr, "NOTE: this model cannot context-shift; phase 4 skipped\n");
    }

    llama_free(ctx);
    r.ok = true;
    return r;
}

static bool line_has_zero_mib(const std::vector<std::string> & lines, const char * fn_prefix) {
    bool found = false;
    for (const auto & l : lines) {
        if (l.find(fn_prefix) == std::string::npos) continue;
        found = true;
        if (l.find(" 0.00 MiB") == std::string::npos) return false;
    }
    return found;
}

static bool line_has_nonzero_mib(const std::vector<std::string> & lines, const char * fn_prefix) {
    for (const auto & l : lines) {
        if (l.find(fn_prefix) == std::string::npos) continue;
        if (l.find(" 0.00 MiB") == std::string::npos) return true;
    }
    return false;
}

int main(int argc, char ** argv) {
    std::string model_path;
    int n_ctx = 512;
    bool strict_self = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-s")) strict_self = true;
        else if (i + 1 < argc && !strcmp(argv[i], "-m")) model_path = argv[++i];
        else if (i + 1 < argc && !strcmp(argv[i], "-c")) n_ctx = atoi(argv[++i]);
    }
    if (model_path.empty()) { fprintf(stderr, "usage: %s -m model.gguf [-c n_ctx] [-s]\n", argv[0]); return 2; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0; // CPU only, always
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "SETUP: model load failed\n"); return 2; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    const run_result on  = run_once(model, n_ctx, true,  n_vocab);
    const run_result off = run_once(model, n_ctx, false, n_vocab);
    if (!on.ok || !off.ok) { fprintf(stderr, "SETUP: a run failed (on.ok=%d off.ok=%d)\n", on.ok, off.ok); return 2; }

    printf("--- KV lines, metadata_only ON ---\n");
    for (const auto & l : on.kv_lines) printf("%s\n", l.c_str());
    printf("--- KV lines, metadata_only OFF (old behaviour) ---\n");
    for (const auto & l : off.kv_lines) printf("%s\n", l.c_str());
    printf("memory module bytes: ON=%zu OFF=%zu | kv_self bytes: ON=%zu OFF=%zu\n", on.module_bytes, off.module_bytes, on.kv_self_bytes, off.kv_self_bytes);

    const bool a_on_zero     = on.module_bytes == 0;
    const bool a_off_nonzero = off.module_bytes > 0;                       // control
    const bool b_kvself      = on.kv_self_bytes > 0 && on.kv_self_bytes == off.kv_self_bytes;
    const bool b_full        = off.module_bytes == off.kv_self_bytes;      // the old duplicate was exactly the same size
    const bool c_log_on      = line_has_zero_mib(on.kv_lines, "llama_kv_cache:") && line_has_nonzero_mib(on.kv_lines, "llama_kv_cache_init:");
    const bool c_log_off     = line_has_nonzero_mib(off.kv_lines, "llama_kv_cache:");
    const bool d_same        = on.stream == off.stream && !on.stream.empty();
    const bool d_selfcons    = on.fresh_ref == on.via_rm && on.fresh_ref == on.via_cp && off.fresh_ref == off.via_rm && off.fresh_ref == off.via_cp;

    printf("A  module bytes ON == 0                       : %s\n", a_on_zero ? "yes" : "NO");
    printf("A' module bytes OFF > 0 (control)             : %s\n", a_off_nonzero ? "yes" : "NO");
    printf("B  kv_self bytes > 0 and identical ON/OFF     : %s\n", b_kvself ? "yes" : "NO");
    printf("B' OFF module bytes == kv_self bytes (the old duplicate)  : %s\n", b_full ? "yes" : "NO");
    printf("C  ON log: llama_kv_cache 0.00 MiB, llama_kv_cache_init nonzero: %s\n", c_log_on ? "yes" : "NO");
    printf("C' OFF log: llama_kv_cache nonzero (control)  : %s\n", c_log_off ? "yes" : "NO");
    printf("D  token streams identical ON vs OFF (%zu tokens) : %s\n", on.stream.size(), d_same ? "yes" : "NO");
    printf("D' fresh == seq_rm re-decode == seq_cp two-slot, both runs  : %s\n", d_selfcons ? "yes" : "NO");
    printf("   (ON: rm=%d cp=%d | OFF: rm=%d cp=%d)\n", on.fresh_ref == on.via_rm, on.fresh_ref == on.via_cp, off.fresh_ref == off.via_rm, off.fresh_ref == off.via_cp);
    if (!d_same) {
        for (size_t i = 0; i < std::min(on.stream.size(), off.stream.size()); ++i) {
            if (on.stream[i] != off.stream[i]) { printf("first divergence at stream index %zu: on=%d off=%d\n", i, on.stream[i], off.stream[i]); break; }
        }
    }

    // D' is INFORMATIONAL: a recurrent model refuses a partial seq_rm (its state cannot roll back) and a sliding-window
    // model does not re-derive identical logits over a truncated prefix, so the three continuations legitimately differ
    // there. It is identical ON and OFF in every case (D), which is the requirement. It is only enforced with -s.
    printf("(D' is %s)\n", strict_self ? "ENFORCED (-s)" : "informational");
    const bool pass = a_on_zero && a_off_nonzero && b_kvself && b_full && c_log_on && c_log_off && d_same && (!strict_self || d_selfcons);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");

    llama_model_free(model);
    llama_backend_free();
    return pass ? 0 : 1;
}
