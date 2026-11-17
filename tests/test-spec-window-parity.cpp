// A T-token verify window must leave the model where T single-token decodes leave it, and a rollback to n accepted tokens must leave it where n single-token
// decodes leave it. Speculative decoding rests on both: it decodes a draft window in one batch, keeps the accepted prefix, and carries on from the kept state.
// A recurrent / hybrid layer that gets the window or the rollback wrong passes every per-token test and corrupts output only after a partial accept.
//
//   leg A (window)   for T in 1..8: prefix of 8 tokens, then one batch of T tokens (logits at every row) vs T batches of 1 token. Reports the max |logit| diff
//                    and argmax per row, then decodes ONE MORE token on both and compares again: that comparison sees state the window rows do not show.
//   leg B (rollback) window of 8 decoded in one batch, seq_rm(P+n, -1), then one token at P+n, vs n single decodes + the same token. Skipped (stated) when the
//                    context keeps no recurrent snapshots (n_rs_seq 0) or the model refuses the partial seq_rm.
//   state            after every comparison the serialised sequence state (llama_state_seq_get_data: KV rows + recurrent cells) must be BITWISE equal. Needed
//                    because a random-weight model's logits barely depend on a recurrent state: a rollback that restores nothing moved them ~1e-9 (measured).
//   control          the same comparison against a DIFFERENT next token must report a difference: a comparator that cannot tell them apart proves nothing.
//
// Not bitwise: fork kernels are not window-size invariant (a batched matmul sums in a different order), so the gate is an argmax match plus a stated tolerance.
// Pattern read from Strata src/kernels/verify_parity.cpp (MIT), whose kernels ARE window-invariant and so can demand bitwise; no code copied.
// Row S69-STRATA-S2-SPEC-WINDOW-PARITY-TEST-FORK-2026-10-07-01.
// GATE-CLAIM-OK: unit test, no resource claims.
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static constexpr uint32_t P     = 8; // prefix length
static constexpr uint32_t T_MAX = 8; // largest window

static llama_context * make_ctx(const common_params & params, llama_model * model) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = 1;
    // WP_RS_EXTRA widens the snapshot ring beyond the largest window (diagnostic: separates "rollback by exactly n_rs_seq" from "rollback in range")
    cparams.n_rs_seq   = T_MAX + (getenv("WP_RS_EXTRA") ? (uint32_t) atoi(getenv("WP_RS_EXTRA")) : 0u);
    cparams.n_ctx      = 256;
    cparams.n_batch    = 64;
    cparams.n_ubatch   = 64;
    cparams.kv_unified = false;
    return llama_init_from_model(model, cparams);
}

static bool decode_range(llama_context * ctx, const std::vector<llama_token> & tok, uint32_t from, uint32_t to, bool all_logits) {
    llama_batch batch = llama_batch_init(to - from, 0, 1);
    for (uint32_t pos = from; pos < to; ++pos) {
        common_batch_add(batch, tok[pos], pos, { 0 }, all_logits || pos + 1 == to);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

struct cmp_result { double max_diff; double scale; bool argmax_ok; };

static cmp_result compare(const float * a, const float * b, int n_vocab) {
    cmp_result r = { 0.0, 1.0, true };
    int ia = 0, ib = 0;
    for (int i = 0; i < n_vocab; ++i) {
        r.max_diff = std::max(r.max_diff, (double) std::fabs(a[i] - b[i]));
        r.scale    = std::max(r.scale, (double) std::fabs(b[i]));
        if (a[i] > a[ia]) ia = i;
        if (b[i] > b[ib]) ib = i;
    }
    if (ia != ib) {
        // a near-tie may flip under a different summation order: accept it only when the reference top-2 gap is within the measured diff
        float second = -INFINITY;
        for (int i = 0; i < n_vocab; ++i) if (i != ib) second = std::max(second, b[i]);
        r.argmax_ok = (double) (b[ib] - second) <= 2.0 * r.max_diff;
    }
    return r;
}

// Serialised state of sequence 0 (KV rows + recurrent cells). Logits of a random-weight model barely depend on a recurrent state (a rollback that restores
// NOTHING moves them by ~1e-9, measured 2026-10-07), so the state itself is the comparison that can see a rollback bug.
// first_diff / last_diff: byte offsets of the first and last differing byte, so a failure points into the state layout (which layer's rows)
struct state_cmp { bool same_size; size_t size; size_t n_diff; double max_rel; size_t first_diff = SIZE_MAX; size_t last_diff = 0; };

static state_cmp compare_state(llama_context * a, llama_context * b) {
    state_cmp r = { false, 0, 0, 0.0 };
    const size_t sa = llama_state_seq_get_size(a, 0, 0);
    const size_t sb = llama_state_seq_get_size(b, 0, 0);
    r.size = sa;
    r.same_size = sa == sb && sa > 0;
    if (!r.same_size) {
        return r;
    }
    std::vector<uint8_t> da(sa), db(sb);
    if (llama_state_seq_get_data(a, da.data(), da.size(), 0, 0) != sa || llama_state_seq_get_data(b, db.data(), db.size(), 0, 0) != sb) {
        r.same_size = false;
        return r;
    }
    for (size_t i = 0; i < sa; ++i) {
        if (da[i] != db[i]) {
            r.n_diff++;
            r.first_diff = std::min(r.first_diff, i);
            r.last_diff  = i;
        }
    }
    return r;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    // deterministic tokens, wrapped into the vocab; P prefix + T_MAX window + 2 spare
    std::vector<llama_token> tok(P + T_MAX + 2);
    for (uint32_t i = 0; i < tok.size(); ++i) {
        tok[i] = (llama_token) ((13*i + 5) % (uint32_t) n_vocab);
    }
    const llama_token alt = (llama_token) ((tok[P] + n_vocab/2 + 1) % n_vocab);

    // MEASURED 2026-10-07 (MinGW CPU, random-weight llama / qwen35 / mamba2): window and singles agree to 0 here, so 1e-4 leaves room for a different
    // summation order on another backend while the control below must still move the logits by more than 10x this
    constexpr double tol_rel = 1e-4; // vs the largest |logit| of the reference
    int fails = 0;
    double worst_window = 0, worst_next = 0, worst_rb = 0, ctl_min = INFINITY;
    int    rb_done = 0, rb_skipped = 0;

    const auto check = [&](const char * what, uint32_t t, const cmp_result & r) {
        const bool ok = r.max_diff <= tol_rel * r.scale && r.argmax_ok;
        if (!ok) {
            fprintf(stderr, "FAIL %s T=%u: max|dlogit| %.3g (tol %.3g) argmax %s\n", what, t, r.max_diff, tol_rel * r.scale, r.argmax_ok ? "ok" : "DIFFERS");
            fails++;
        }
    };

    // bitwise: MEASURED 2026-10-07 on nemotron_h (CPU, MinGW) a correct rollback leaves 0 of 291100 state bytes different and a broken one 206559
    size_t worst_state_bytes = 0;
    const auto check_state = [&](const char * what, uint32_t t, const state_cmp & s) {
        worst_state_bytes = std::max(worst_state_bytes, s.n_diff);
        if (!s.same_size || s.n_diff != 0) {
            fprintf(stderr, "FAIL %s %u: sequence state %s (%zu of %zu bytes differ, bytes [%zu, %zu])\n", what, t, s.same_size ? "differs" : "has a different size / unreadable", s.n_diff, s.size,
                    s.first_diff == SIZE_MAX ? (size_t) 0 : s.first_diff, s.last_diff);
            fails++;
        }
    };

    // ---- leg A: window vs singles ----
    for (uint32_t T = 1; T <= T_MAX; ++T) {
        llama_context * cw = make_ctx(params, model);
        llama_context * cs = make_ctx(params, model);
        if (cw == nullptr || cs == nullptr) {
            fprintf(stderr, "%s : failed to init contexts\n", __func__);
            return 1;
        }
        bool ok = decode_range(cw, tok, 0, P, false) && decode_range(cs, tok, 0, P, false) && decode_range(cw, tok, P, P + T, true);
        std::vector<std::vector<float>> single_rows;
        for (uint32_t i = 0; ok && i < T; ++i) {
            ok = decode_range(cs, tok, P + i, P + i + 1, true);
            if (ok) single_rows.emplace_back(llama_get_logits_ith(cs, 0), llama_get_logits_ith(cs, 0) + n_vocab);
        }
        if (!ok) {
            fprintf(stderr, "%s : decode failed at T=%u\n", __func__, T);
            return 1;
        }
        for (uint32_t i = 0; i < T; ++i) {
            const cmp_result r = compare(llama_get_logits_ith(cw, i), single_rows[i].data(), n_vocab);
            worst_window = std::max(worst_window, r.max_diff / r.scale);
            check("window row", T, r);
        }
        // state: one more token on both
        ok = decode_range(cw, tok, P + T, P + T + 1, true) && decode_range(cs, tok, P + T, P + T + 1, true);
        if (!ok) {
            fprintf(stderr, "%s : next-token decode failed at T=%u\n", __func__, T);
            return 1;
        }
        const cmp_result rn = compare(llama_get_logits_ith(cw, 0), llama_get_logits_ith(cs, 0), n_vocab);
        worst_next = std::max(worst_next, rn.max_diff / rn.scale);
        check("token after window (state)", T, rn);
        check_state("window then one token", T, compare_state(cw, cs));
        llama_free(cw);
        llama_free(cs);
    }

    // ---- leg B: rollback to n accepted, then continue ----
    for (uint32_t n : { 0u, 1u, 3u, 7u }) {
        llama_context * cw = make_ctx(params, model);
        llama_context * cs = make_ctx(params, model);
        llama_context * cd = make_ctx(params, model); // control: same history, DIFFERENT next token
        if (cw == nullptr || cs == nullptr || cd == nullptr) {
            fprintf(stderr, "%s : failed to init contexts\n", __func__);
            return 1;
        }
        if (llama_n_rs_seq(cw) == 0 && (llama_model_is_recurrent(model) || llama_model_is_hybrid(model))) {
            // the context keeps no snapshots, so a partial rollback is not offered: that is a stated limitation, not a parity result
            fprintf(stderr, "rollback n=%u skipped: n_rs_seq is 0 on a recurrent/hybrid model\n", n);
            rb_skipped++;
            llama_free(cw); llama_free(cs); llama_free(cd);
            continue;
        }
        bool ok = decode_range(cw, tok, 0, P, false);
        if (getenv("WP_SPLIT")) { // diagnostic: window in two batches, so a rollback of all T_MAX tokens crosses a batch boundary
            ok = ok && decode_range(cw, tok, P, P + T_MAX/2, true) && decode_range(cw, tok, P + T_MAX/2, P + T_MAX, true);
        } else {
            ok = ok && decode_range(cw, tok, P, P + T_MAX, true);
        }
        if (ok && !llama_memory_seq_rm(llama_get_memory(cw), 0, (llama_pos) (P + n), -1)) {
            fprintf(stderr, "rollback n=%u skipped: partial seq_rm(%u, -1) refused (n_rs_seq %u): no saved state that far back\n", n, P + n, (unsigned) llama_n_rs_seq(cw));
            rb_skipped++;
            llama_free(cw); llama_free(cs); llama_free(cd);
            continue;
        }
        ok = ok && decode_range(cs, tok, 0, P, false) && decode_range(cd, tok, 0, P, false);
        for (uint32_t i = 0; ok && i < n; ++i) {
            ok = decode_range(cs, tok, P + i, P + i + 1, false) && decode_range(cd, tok, P + i, P + i + 1, false);
        }
        std::vector<llama_token> alt_tok = tok;
        alt_tok[P + n] = alt;
        ok = ok && decode_range(cw, tok, P + n, P + n + 1, true) && decode_range(cs, tok, P + n, P + n + 1, true) && decode_range(cd, alt_tok, P + n, P + n + 1, true);
        if (!ok) {
            fprintf(stderr, "%s : rollback leg decode failed at n=%u\n", __func__, n);
            return 1;
        }
        const cmp_result r = compare(llama_get_logits_ith(cw, 0), llama_get_logits_ith(cs, 0), n_vocab);
        worst_rb = std::max(worst_rb, r.max_diff / r.scale);
        check("rollback then continue", n, r);
        // a rollback the memory accepts must leave the singles-path state bitwise; one it cannot back with a saved state is refused (skipped above).
        // Before row S69-FORK-RECURRENT-ROLLBACK-WHOLE-BATCH-AND-CROSS-BATCH-ACCEPTED-WRONG-STATE-2026-10-07-01 landed, n=0 (k == batch size) and a
        // rollback crossing a batch (WP_SPLIT=1) were accepted with 95% of the state bytes wrong (nemotron_h, CPU).
        check_state("rollback then one token", n, compare_state(cw, cs));
        const cmp_result c = compare(llama_get_logits_ith(cw, 0), llama_get_logits_ith(cd, 0), n_vocab);
        ctl_min = std::min(ctl_min, c.max_diff / c.scale);
        rb_done++;
        llama_free(cw); llama_free(cs); llama_free(cd);
    }

    printf("window parity: worst window-row rel diff %.3g, worst next-token rel diff %.3g (tol %.3g)\n", worst_window, worst_next, tol_rel);
    printf("sequence state: worst differing bytes %zu (must be 0)\n", worst_state_bytes);
    printf("rollback: %d done, %d skipped (no snapshots or partial seq_rm refused), worst rel diff %.3g\n", rb_done, rb_skipped, rb_done ? worst_rb : 0.0);
    if (rb_done) {
        printf("control (different next token): min rel diff %.3g\n", ctl_min);
        if (ctl_min <= 10.0 * tol_rel) {
            fprintf(stderr, "FAIL control: a different next token moved the logits by only %.3g (<= 10x tol): the comparator cannot see a wrong state\n", ctl_min);
            fails++;
        }
    }
    if (fails) {
        fprintf(stderr, "%d check(s) FAILED\n", fails);
        return 1;
    }
    printf("all window-parity checks passed\n");
    return 0;
}
