// SCOUT-NA: unit test in the fork tree, no new tool.  GATE-CLAIM-OK: unit test, no resource claims.
// Row S69-ASYM-SPEC-ASYNC-DRAFT-VERIFY-OVERLAP-PEARL-AMUSD-2026-11-06-01 (--spec-async: draft || verify overlap, draft-model drafter).
//
// Drives the REAL pooled draft-model speculator (common_speculative_init -> common_speculative_impl_draft) the way llama-server does, one
// sequence, greedy, on a tiny model:   draft -> trim draft ctx -> [async_start] -> target decode of [id_last, draft] -> process() on the
// draft ctx -> accept -> roll BOTH contexts back to the accepted prefix (llama_memory_seq_rm).
//
//   A  LOSSLESS   the token stream with --spec-async equals the serial stream equals plain one-token-at-a-time greedy decoding of the
//                 target (the target decides every token), in both draft configurations below.
//   B  HIT PATH   target == draft model (every draft accepted, the target's token equals the lookahead's c1): drafts are served from the
//                 lookahead (n_hit > 0) and the worker's [id_last, draft] evaluation replaced process()'s (n_step0 == n_rounds).
//   C  ROLLBACK   draft = a Q4_0 requantisation of the target (partial acceptance): n_rollback > 0, and after every round the draft
//                 context holds exactly the positions below the next position (llama_memory_seq_pos_max == pos_next - 1), i.e. the
//                 lookahead KV the worker wrote past the accepted prefix is gone. Stream still equals plain greedy (part A).
//   D  OFF        async off: no worker, stats unavailable (common_speculative_async_get_stats == false).
//
// argv[1] = target model (f32 or any ggml-supported file). Exit 77 (skip) without it.
#include "common.h"
#include "llama.h"
#include "speculative.h"

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int n_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s  ", __FILE__, __LINE__, #c); fprintf(stderr, "" __VA_ARGS__); fprintf(stderr, "\n"); n_fail++; } } while (0)

static llama_token argmax(const float * l, int n) {
    int b = 0;
    for (int i = 1; i < n; ++i) { if (l[i] > l[b]) { b = i; } }
    return b;
}

struct ctxs {
    llama_model *   m_tgt = nullptr;
    llama_model *   m_dft = nullptr;
    llama_context * c_tgt = nullptr;
    llama_context * c_dft = nullptr;
    ~ctxs() {
        if (c_dft) { llama_free(c_dft); }
        if (c_tgt) { llama_free(c_tgt); }
        if (m_dft) { llama_model_free(m_dft); }
        if (m_tgt) { llama_model_free(m_tgt); }
    }
};

static llama_context * mk_ctx(llama_model * m) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 128; cp.n_ubatch = 128; cp.n_seq_max = 1; cp.n_threads = 2; cp.n_threads_batch = 2;
    return llama_init_from_model(m, cp);
}

struct result {
    std::vector<llama_token> stream;
    common_speculative_async_stats st;
    bool has_stats = false;
    int  n_rounds = 0;
    int  n_pos_mismatch = 0; // rounds where the draft ctx did not hold exactly pos_next positions after the rollback
    int  n_partial = 0;      // rounds with a partial acceptance
};

static const int N_GEN = 48;

// plain greedy decoding of the target, one token per decode
static std::vector<llama_token> plain(llama_model * m, const std::vector<llama_token> & prompt) {
    llama_context * c = mk_ctx(m);
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(m));
    std::vector<llama_token> out;
    llama_batch b = llama_batch_init(128, 0, 1);
    for (size_t i = 0; i < prompt.size(); ++i) { common_batch_add(b, prompt[i], (llama_pos) i, { 0 }, i + 1 == prompt.size()); }
    llama_decode(c, b);
    llama_token t = argmax(llama_get_logits_ith(c, b.n_tokens - 1), nv);
    llama_pos pos = (llama_pos) prompt.size();
    for (int i = 0; i < N_GEN; ++i) {
        out.push_back(t);
        common_batch_clear(b);
        common_batch_add(b, t, pos++, { 0 }, true);
        llama_decode(c, b);
        t = argmax(llama_get_logits_ith(c, 0), nv);
    }
    llama_batch_free(b);
    llama_free(c);
    return out;
}

// wait_ms = --spec-async-wait-ms. 0 (the default) stops the lookahead at the next token boundary, so whether a lookahead is complete
// when process() joins the worker is a scheduling race: MEASURED 2026-11-06 on gate zf's CUDA build, self-draft n_hit was 0 in 6/8
// runs with the GPU visible and 2/8 with it hidden, while every lossless check passed in all 17 runs. Hit assertions therefore run
// with a budget that lets the lookahead finish; the default-wait run keeps every lossless check.
static result run(const std::string & p_tgt, const std::string & p_dft, const std::vector<llama_token> & prompt, bool async, int n_max,
                  int wait_ms = 0, bool informed = false, bool greedy = true) {
    result r;
    ctxs x;
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    x.m_tgt = llama_model_load_from_file(p_tgt.c_str(), mp);
    x.m_dft = p_dft == p_tgt ? x.m_tgt : llama_model_load_from_file(p_dft.c_str(), mp);
    if (!x.m_tgt || !x.m_dft) { fprintf(stderr, "model load failed\n"); n_fail++; if (p_dft == p_tgt) { x.m_dft = nullptr; } return r; }
    x.c_tgt = mk_ctx(x.m_tgt);
    x.c_dft = mk_ctx(x.m_dft);
    if (p_dft == p_tgt) { x.m_dft = nullptr; } // one model, freed once through m_tgt

    common_params_speculative sp;
    sp.type  = COMMON_SPECULATIVE_TYPE_DRAFT;
    sp.n_max = n_max;
    sp.n_min = 0;
    sp.ctx_tgt = x.c_tgt;
    sp.ctx_dft = x.c_dft;
    sp.async = async;
    sp.async_wait_ms = wait_ms;
    sp.async_informed = informed;
    common_speculative * spec = common_speculative_init(sp, 1);
    CHECK(spec != nullptr);
    if (!spec) { return r; }

    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(x.c_tgt)));
    llama_batch bt = llama_batch_init(128, 0, 1);

    // prefill both contexts with the prompt except nothing: the prompt goes through the target, then process() feeds the draft ctx
    for (size_t i = 0; i < prompt.size(); ++i) { common_batch_add(bt, prompt[i], (llama_pos) i, { 0 }, i + 1 == prompt.size()); }
    llama_decode(x.c_tgt, bt);
    CHECK(common_speculative_process(spec, bt));
    llama_token sampled = argmax(llama_get_logits_ith(x.c_tgt, bt.n_tokens - 1), nv);

    std::vector<llama_token> P = prompt; // tokens in the contexts (text)
    while ((int) r.stream.size() < N_GEN) {
        const llama_pos pos0 = (llama_pos) P.size();
        llama_tokens D;
        auto & dp   = common_speculative_get_draft_params(spec, 0);
        dp.drafting = true;
        dp.n_max    = -1;
        dp.pos0     = pos0;
        dp.id_last  = sampled;
        dp.prompt   = &P;
        dp.result   = &D;
        common_speculative_draft(spec);

        // the server trims the draft context to the target's position after the draft call
        CHECK(llama_memory_seq_rm(llama_get_memory(x.c_dft), 0, pos0, -1));

        if (async && !D.empty()) {
            std::vector<common_speculative_async_req> reqs = { { 0, pos0, sampled, &D, greedy } };
            common_speculative_async_start(spec, reqs);
        }

        // verify batch [sampled, D...] on the target
        common_batch_clear(bt);
        common_batch_add(bt, sampled, pos0, { 0 }, true);
        for (size_t i = 0; i < D.size(); ++i) { common_batch_add(bt, D[i], pos0 + 1 + (llama_pos) i, { 0 }, true); }
        CHECK(llama_decode(x.c_tgt, bt) == 0);
        CHECK(common_speculative_process(spec, bt));

        size_t n_acc = 0;
        llama_token t = argmax(llama_get_logits_ith(x.c_tgt, 0), nv);
        while (n_acc < D.size() && D[n_acc] == t) {
            ++n_acc;
            t = argmax(llama_get_logits_ith(x.c_tgt, (int) n_acc), nv);
        }
        // accepted tokens: sampled, D[:n_acc] enter the text; t is the new sampled token
        r.stream.push_back(sampled);
        P.push_back(sampled);
        for (size_t i = 0; i < n_acc; ++i) { r.stream.push_back(D[i]); P.push_back(D[i]); }
        sampled = t;
        if (n_acc < D.size()) { r.n_partial++; }

        // roll both contexts back to the accepted prefix (what slot.mem.seq_rm does in the server)
        const llama_pos pos_next = (llama_pos) P.size();
        CHECK(llama_memory_seq_rm(llama_get_memory(x.c_tgt), 0, pos_next, -1));
        CHECK(llama_memory_seq_rm(llama_get_memory(x.c_dft), 0, pos_next, -1));
        if (llama_memory_seq_pos_max(llama_get_memory(x.c_dft), 0) != pos_next - 1) { r.n_pos_mismatch++; }
        r.n_rounds++;
    }
    r.stream.resize(N_GEN);
    r.has_stats = common_speculative_async_get_stats(spec, r.st);
    llama_batch_free(bt);
    common_speculative_free(spec);
    return r;
}

// S69-SPECASYNC-DIAG-COUNTERS-2026-11-06-01: the counters add up, whatever the run did
static void check_counters(const char * tag, const result & r) {
    if (!r.has_stats) {
        return;
    }
    const auto & s = r.st;
    const uint64_t classes = s.n_hit + s.n_o_nola + s.n_o_laempty + s.n_o_notfull + s.n_o_c1miss + s.n_o_trunc + s.n_o_guard + s.n_o_nodraft +
                             s.n_o_fallback + s.n_o_cancel + s.n_o_dropped + s.n_open;
    uint64_t hist = 0;
    for (int i = 0; i < 11; ++i) { hist += s.la_hist[i]; }
    uint64_t rank = 0;
    for (int i = 0; i < 6; ++i) { rank += s.c1_rank[i]; }
    printf("diag %-12s: seq_rounds=%llu hit=%llu nola=%llu laempty=%llu notfull=%llu c1miss=%llu trunc=%llu guard=%llu nodraft=%llu open=%llu "
           "full=%llu c1_match=%llu rollback=%llu hist_sum=%llu rank_sum=%llu la_hist[0..3]=%llu,%llu,%llu,%llu s0_us=%llu v_us=%llu d_us=%llu "
           "inf(rounds=%llu wait=%llu stop=%llu legacy=%llu wait_hit=%llu wait_miss=%llu)\n",
           tag, (unsigned long long) s.n_seq_rounds, (unsigned long long) s.n_hit, (unsigned long long) s.n_o_nola, (unsigned long long) s.n_o_laempty,
           (unsigned long long) s.n_o_notfull, (unsigned long long) s.n_o_c1miss, (unsigned long long) s.n_o_trunc, (unsigned long long) s.n_o_guard,
           (unsigned long long) s.n_o_nodraft, (unsigned long long) s.n_open, (unsigned long long) s.n_full, (unsigned long long) s.n_c1_match,
           (unsigned long long) s.n_rollback, (unsigned long long) hist, (unsigned long long) rank, (unsigned long long) s.la_hist[0],
           (unsigned long long) s.la_hist[1], (unsigned long long) s.la_hist[2], (unsigned long long) s.la_hist[3], (unsigned long long) s.ema_s0_us,
           (unsigned long long) s.ema_v_us, (unsigned long long) s.ema_d_us, (unsigned long long) s.n_inf_rounds, (unsigned long long) s.n_inf_wait,
           (unsigned long long) s.n_inf_stop, (unsigned long long) s.n_inf_legacy, (unsigned long long) s.n_inf_wait_hit, (unsigned long long) s.n_inf_wait_miss);
    CHECK(s.n_seq_rounds >= s.n_rounds && s.n_seq_rounds > 0, "%s: seq_rounds %llu rounds %llu", tag, (unsigned long long) s.n_seq_rounds, (unsigned long long) s.n_rounds);
    CHECK(classes == s.n_seq_rounds, "%s: outcome classes sum to %llu, seq_rounds %llu", tag, (unsigned long long) classes, (unsigned long long) s.n_seq_rounds);
    CHECK(hist + s.n_o_fallback + s.n_o_dropped + s.n_o_cancel + s.n_open == s.n_seq_rounds, "%s: lookahead histogram does not cover the rounds that reached draft()", tag);
    CHECK(s.n_rollback == s.n_o_notfull + s.n_o_c1miss, "%s: n_rollback must be real mispredictions only (notfull + c1miss)", tag);
    CHECK(s.n_hit <= s.n_c1_match && s.n_c1_match <= s.n_full, "%s: hit <= c1_match <= full", tag);
    CHECK(s.ema_s0_us > 0 && s.ema_v_us > 0, "%s: step 0 / verify EMAs were never measured", tag);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: test-spec-async <target.gguf>   (skipping)\n");
        return 77;
    }
    ggml_time_init();
    llama_backend_init();
    const std::string tgt = argv[1];

    std::vector<llama_token> prompt;
    {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        llama_model * m = llama_model_load_from_file(tgt.c_str(), mp);
        if (!m) { fprintf(stderr, "cannot load %s\n", tgt.c_str()); return 1; }
        const llama_vocab * v = llama_model_get_vocab(m);
        std::string text = "Once upon a time there was a little girl named Lily. She loved to play in the park with her friends.";
        prompt.resize(text.size() + 8);
        int n = llama_tokenize(v, text.c_str(), (int) text.size(), prompt.data(), (int) prompt.size(), true, false);
        prompt.resize(std::max(n, 0));
        llama_model_free(m);
    }
    if (prompt.size() < 4) { fprintf(stderr, "tokenization failed\n"); return 1; }

    // the oracle
    llama_model_params mp0 = llama_model_default_params();
    mp0.n_gpu_layers = 0;
    llama_model * mo = llama_model_load_from_file(tgt.c_str(), mp0);
    const std::vector<llama_token> ref = plain(mo, prompt);
    llama_model_free(mo);

    // ---- B: target == draft -------------------------------------------------------------------------------------------------
    const result s1 = run(tgt, tgt, prompt, false, 4);
    const result a1 = run(tgt, tgt, prompt, true, 4);
    CHECK(s1.stream == ref, "serial stream differs from plain greedy (self-draft)");
    CHECK(a1.stream == ref, "async stream differs from plain greedy (self-draft)");
    CHECK(a1.stream == s1.stream);
    CHECK(!s1.has_stats, "D: async off must report no stats");      // D
    CHECK(a1.has_stats, "async on must report stats");
    if (a1.has_stats) {
        printf("self-draft  : rounds=%llu step0=%llu hit=%llu rollback=%llu fallback=%llu la_tokens=%llu sync_drafts=%llu\n",
               (unsigned long long) a1.st.n_rounds, (unsigned long long) a1.st.n_step0, (unsigned long long) a1.st.n_hit,
               (unsigned long long) a1.st.n_rollback, (unsigned long long) a1.st.n_fallback, (unsigned long long) a1.st.n_la_tokens,
               (unsigned long long) a1.st.n_sync_drafts);
        CHECK(a1.st.n_rounds > 0);
        CHECK(a1.st.n_step0 == a1.st.n_rounds, "every round's [id_last,draft] evaluation ran on the worker (process() skipped it)");
        CHECK(a1.st.n_fallback == 0);
    }
    CHECK(a1.n_pos_mismatch == 0, "draft ctx positions after rollback (self-draft)");
    check_counters("self-w0", a1);
    // C9: a lookahead the join cut short is NOT a rollback. Self-draft accepts every draft, so rollback can only come from a c1 mismatch
    // (never from truncation); before 2026-11-06 every truncated round was booked as a rollback.
    if (a1.has_stats && a1.n_partial == 0) {
        CHECK(a1.st.n_o_notfull == 0, "self-draft w0: no partial acceptance, yet notfull=%llu", (unsigned long long) a1.st.n_o_notfull);
        CHECK(a1.st.n_rollback == a1.st.n_o_c1miss, "self-draft w0: rollback %llu must equal c1miss %llu (truncation is its own counter, trunc=%llu)",
              (unsigned long long) a1.st.n_rollback, (unsigned long long) a1.st.n_o_c1miss, (unsigned long long) a1.st.n_o_trunc);
    }

    // B2: same self-draft with a wait budget, so the lookahead completes before process() joins it: a hit is now deterministic
    const result a2 = run(tgt, tgt, prompt, true, 4, 2000);
    CHECK(a2.stream == ref, "async stream differs from plain greedy (self-draft, wait 2000 ms)");
    CHECK(a2.has_stats, "async on must report stats (wait 2000 ms)");
    if (a2.has_stats) {
        printf("self-draft-w: rounds=%llu step0=%llu hit=%llu rollback=%llu fallback=%llu la_tokens=%llu sync_drafts=%llu\n",
               (unsigned long long) a2.st.n_rounds, (unsigned long long) a2.st.n_step0, (unsigned long long) a2.st.n_hit,
               (unsigned long long) a2.st.n_rollback, (unsigned long long) a2.st.n_fallback, (unsigned long long) a2.st.n_la_tokens,
               (unsigned long long) a2.st.n_sync_drafts);
        CHECK(a2.st.n_hit > 0, "B: no draft was served from the lookahead (wait 2000 ms)");
        CHECK(a2.st.n_fallback == 0);
    }
    CHECK(a2.n_pos_mismatch == 0, "draft ctx positions after rollback (self-draft, wait 2000 ms)");
    check_counters("self-w2000", a2);

    // B3: --spec-async-informed-join with wait_ms = 0 (the legacy cut): the verify rows say "whole draft accepted, c1 == the target's token",
    // so the join lets the lookahead finish and a hit is deterministic. Greedy request.
    const result a3 = run(tgt, tgt, prompt, true, 4, 0, true, true);
    CHECK(a3.stream == ref, "async stream differs from plain greedy (self-draft, informed join)");
    CHECK(a3.has_stats, "async on must report stats (informed join)");
    check_counters("self-inf", a3);
    if (a3.has_stats) {
        CHECK(a3.st.n_hit > 0, "O1: informed join with wait_ms=0 produced no hit");
        CHECK(a3.st.n_inf_rounds > 0 && a3.st.n_inf_wait > 0, "O1: the join never decided from the verify rows (rounds=%llu wait=%llu legacy=%llu)",
              (unsigned long long) a3.st.n_inf_rounds, (unsigned long long) a3.st.n_inf_wait, (unsigned long long) a3.st.n_inf_legacy);
        CHECK(a3.st.n_inf_legacy == 0, "O1: rows should be readable in this harness, legacy joins=%llu", (unsigned long long) a3.st.n_inf_legacy);
        CHECK(a3.st.n_inf_wait_hit > 0, "O1: no waited lookahead was used");
        CHECK(a3.st.c1_rank[0] > 0, "O5 counter: the target's token should be the draft's top-1 in a self-draft");
        CHECK(a3.st.n_fallback == 0);
    }
    CHECK(a3.n_pos_mismatch == 0, "draft ctx positions after rollback (self-draft, informed join)");

    // B4: a non-greedy request keeps the wait_ms rule (n_inf_legacy), still lossless
    const result a4 = run(tgt, tgt, prompt, true, 4, 0, true, false);
    CHECK(a4.stream == ref, "async stream differs from plain greedy (self-draft, informed join, non-greedy hint)");
    check_counters("self-inf-ng", a4);
    if (a4.has_stats) {
        CHECK(a4.st.n_inf_rounds == 0 && a4.st.n_inf_legacy > 0, "non-greedy hint must take the legacy join (rounds=%llu legacy=%llu)",
              (unsigned long long) a4.st.n_inf_rounds, (unsigned long long) a4.st.n_inf_legacy);
    }

    // ---- C: requantised draft => rounds that accept only a prefix of the draft ------------------------------------------------------
    const std::string q4 = tgt + ".specasync-q4_0.tmp.gguf";
    llama_model_quantize_params qp = llama_model_quantize_default_params();
    {
        std::string lower = tgt;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return (char) std::tolower(c); });
        // a different quantisation than the target's own, so the draft differs from the target now and then
        qp.ftype = lower.find("q4_0") != std::string::npos ? LLAMA_FTYPE_MOSTLY_IQ4_NL : LLAMA_FTYPE_MOSTLY_Q4_0;
    }
    qp.allow_requantize = true;
    const uint32_t qr = llama_model_quantize(tgt.c_str(), q4.c_str(), &qp);
    if (qr != 0) {
        fprintf(stderr, "SKIP C: could not requantise the target (rc=%u)\n", qr);
    } else {
        const result s2 = run(tgt, q4, prompt, false, 4);
        const result a2 = run(tgt, q4, prompt, true, 4);
        CHECK(s2.stream == ref, "serial stream differs from plain greedy (q4 draft)");
        CHECK(a2.stream == ref, "async stream differs from plain greedy (q4 draft)");
        CHECK(a2.stream == s2.stream);
        CHECK(a2.n_pos_mismatch == 0, "C: draft ctx kept lookahead KV past the accepted prefix");
        CHECK(a2.n_partial > 0, "C: the q4 draft never produced a partial acceptance, rollback not exercised");
        if (a2.has_stats) {
            printf("q4-draft    : rounds=%llu step0=%llu hit=%llu rollback=%llu fallback=%llu partial_rounds=%d\n",
                   (unsigned long long) a2.st.n_rounds, (unsigned long long) a2.st.n_step0, (unsigned long long) a2.st.n_hit,
                   (unsigned long long) a2.st.n_rollback, (unsigned long long) a2.st.n_fallback, a2.n_partial);
            CHECK(a2.st.n_rollback > 0, "C: no lookahead was rolled back");
            CHECK(a2.st.n_fallback == 0);
        }
        check_counters("q4", a2);
        if (a2.has_stats) {
            CHECK(a2.st.n_o_notfull > 0, "C: partial acceptances must show up as notfull");
        }

        // C2: the same rollback case with the informed join: partial accepts => the join stops the worker at once, lossless, KV rolled back
        const result a5 = run(tgt, q4, prompt, true, 4, 0, true, true);
        CHECK(a5.stream == ref, "async stream differs from plain greedy (q4 draft, informed join)");
        CHECK(a5.n_pos_mismatch == 0, "C: draft ctx kept lookahead KV past the accepted prefix (informed join)");
        CHECK(a5.n_partial > 0, "C: informed run never produced a partial acceptance");
        check_counters("q4-inf", a5);
        if (a5.has_stats) {
            CHECK(a5.st.n_inf_stop > 0, "O1: partial accepts must stop the worker at once (stop=%llu)", (unsigned long long) a5.st.n_inf_stop);
            CHECK(a5.st.n_rollback > 0 && a5.st.n_fallback == 0);
        }
        std::remove(q4.c_str());
    }

    if (n_fail == 0) { printf("test-spec-async: OK\n"); }
    llama_backend_free();
    return n_fail == 0 ? 0 : 1;
}
