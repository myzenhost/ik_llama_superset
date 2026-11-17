// per-sequence speculative pool carries the suffix types.   GATE-CLAIM-OK: documentation only.
//
// FAIL-BEFORE: common_speculative_init(params, n_seq) (the speculator llama-server builds) enumerated ngram_map_k / k4v / draft only, so a
// suffix_automaton or suffix stage got a NULL speculator: check A fails at the parent commit ("init returned null").
//   A  a suffix_automaton stage builds a speculator for 2 sequences
//   B  sequence 0 sees a repeating history and drafts the continuation; the draft is the exact repeat (hand-derived: history 1 2 3 4 1 2 3 4 1 2 3, id_last 4
//      -> longest suffix match continues 1 2 3 4 ...)
//   C  sequence 1 (never given that history) drafts nothing, so per-sequence state does not leak across sequences
//   D  a stage override of suffix_min_match_len larger than any repeat in the history suppresses the draft (the knob reaches the pooled state)
//   E  the SUFFIX tree type also builds (no corpus) and drafts the repeat; with a corpus path it is NOT pooled (init returns null for that stage alone)
//   F  an unbuildable stage (dflash) ahead of suffix_automaton does not take the chain down; eagle3 / mtp alone without a companion
//      context give no speculator (each is named in a warning by the constructor)
// 2026-10-18, row S69-FORK-SERVER-POOL-NGRAM-MOD-SIMPLE-CACHE-DSPARK-IMPLS-2026-10-17-01 (FAIL-BEFORE at d279355c6: G/H/I init returned null,
// the pool warned "not built" for ngram_simple / ngram_mod / ngram_cache):
//   G  ngram_simple builds for 2 sequences and drafts the repeat (pattern 2 3 4 found at position 5, continuation starts 1)
//   H  ngram_mod: ONE table for both sequences -- seq 1 drafts from text only seq 0 fed (hand-derived 1 2 3 4 1 2 3 4 from n=3); after seq 1's
//      low-acceptance streak resets the shared table, seq 0 re-feeds its own history and still drafts (upstream keeps seq 0's feed mark and drafts
//      nothing there)
//   I  ngram_cache: a slot's second, SHORTER request with a different text drafts that text's continuation (its context cache is rebuilt at
//      begin; with the previous fill mark kept, none of the new text would be added); a sequence never given that text drafts nothing from it
#include "common.h"
#include "speculative.h"

#include "ggml.h"

#include <cstdio>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static common_params_speculative make(common_speculative_type type, int n_max, int min_match, bool override_min_match) {
    common_params_speculative p;
    common_speculative_stage_params st;
    st.type  = type;
    p.stages = { st };
    p.n_max  = n_max;
    p.n_min  = 0;
    p.suffix_min_match_len = min_match;
    (void) override_min_match;
    return p;
}

static llama_tokens draft_for(common_speculative * spec, llama_seq_id seq, const llama_tokens & hist, llama_token id_last) {
    llama_tokens out;
    auto & dp   = common_speculative_get_draft_params(spec, seq);
    dp.drafting = true;
    dp.n_max    = -1;
    dp.pos0     = (llama_pos) hist.size(); // upstream renamed n_past -> pos0 (the next position)
    dp.id_last  = id_last;
    dp.prompt   = &hist;
    dp.result   = &out;
    common_speculative_draft(spec);
    return out;
}

int main() {
    ggml_time_init();

    const llama_tokens hist = { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3 };
    const llama_tokens empty_hist = { 7, 8, 9 };

    // A / B / C
    {
        common_params_speculative p = make(COMMON_SPECULATIVE_TYPE_SUFFIX_AUTOMATON, 8, 2, false);
        common_speculative * spec = common_speculative_init(p, 2);
        CHECK(spec != nullptr);
        if (spec == nullptr) {
            fprintf(stderr, "init returned null\n");
            return 1;
        }
        common_speculative_begin(spec, 0, hist);
        common_speculative_begin(spec, 1, empty_hist);

        const llama_tokens d0 = draft_for(spec, 0, hist, 4);
        printf("B  seq0 draft:");
        for (auto t : d0) { printf(" %d", (int) t); }
        printf("\n");
        CHECK(d0.size() >= 4);
        const llama_tokens want = { 1, 2, 3, 4 };
        for (size_t i = 0; i < want.size() && i < d0.size(); ++i) {
            CHECK(d0[i] == want[i]);
        }

        const llama_tokens d1 = draft_for(spec, 1, empty_hist, 5);
        printf("C  seq1 draft size: %zu\n", d1.size());
        CHECK(d1.empty());

        common_speculative_free(spec);
    }

    // D: a match length longer than any repeat in the history gives no draft
    {
        common_params_speculative p = make(COMMON_SPECULATIVE_TYPE_SUFFIX_AUTOMATON, 8, 40, true);
        common_speculative * spec = common_speculative_init(p, 1);
        CHECK(spec != nullptr);
        if (spec != nullptr) {
            common_speculative_begin(spec, 0, hist);
            const llama_tokens d = draft_for(spec, 0, hist, 4);
            printf("D  min_match_len=40 draft size: %zu\n", d.size());
            CHECK(d.empty());
            common_speculative_free(spec);
        }
    }

    // E: suffix tree type (no corpus) is pooled; with a corpus it is not
    {
        common_params_speculative p = make(COMMON_SPECULATIVE_TYPE_SUFFIX, 8, 2, false);
        common_speculative * spec = common_speculative_init(p, 1);
        CHECK(spec != nullptr);
        if (spec != nullptr) {
            common_speculative_begin(spec, 0, hist);
            const llama_tokens d = draft_for(spec, 0, hist, 4);
            printf("E  suffix tree draft size: %zu\n", d.size());
            CHECK(!d.empty());
            common_speculative_free(spec);
        }
        common_params_speculative pc = make(COMMON_SPECULATIVE_TYPE_SUFFIX, 8, 2, false);
        pc.suffix_corpus = "nonexistent-corpus.txt";
        common_speculative * spec_c = common_speculative_init(pc, 1);
        CHECK(spec_c == nullptr);
        if (spec_c != nullptr) {
            common_speculative_free(spec_c);
        }
    }

    // F (2026-10-17, the pool constructor now builds stages in chain order and warns per unbuilt stage): a stage the pool
    // cannot build is skipped without taking the rest of the chain with it, and a chain of only unbuildable stages
    // (eagle3 without a draft context; mtp without the MTP context) yields no speculator at all.
    {
        common_params_speculative p = make(COMMON_SPECULATIVE_TYPE_SUFFIX_AUTOMATON, 8, 2, false);
        common_speculative_stage_params st_unbuilt;
        st_unbuilt.type = COMMON_SPECULATIVE_TYPE_DFLASH; // ngram_simple until 2026-10-18, when the pool learned to build it
        p.stages.insert(p.stages.begin(), st_unbuilt);
        common_speculative * spec = common_speculative_init(p, 1);
        CHECK(spec != nullptr);
        if (spec != nullptr) {
            common_speculative_begin(spec, 0, hist);
            const llama_tokens d = draft_for(spec, 0, hist, 4);
            printf("F  dflash(unbuilt),suffix_automaton draft size: %zu\n", d.size());
            CHECK(d.size() >= 4);
            common_speculative_free(spec);
        }
        for (const auto t : { COMMON_SPECULATIVE_TYPE_EAGLE3, COMMON_SPECULATIVE_TYPE_MTP }) {
            common_params_speculative pn = make(t, 8, 2, false);
            common_speculative * spec_n = common_speculative_init(pn, 2);
            printf("F  %s alone without a companion context: %s\n", common_speculative_type_to_str(t).c_str(), spec_n ? "BUILT" : "null");
            CHECK(spec_n == nullptr);
            if (spec_n != nullptr) {
                common_speculative_free(spec_n);
            }
        }
    }

    const llama_tokens other_hist = { 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7 };

    // G: ngram_simple, n = 3, m = 4
    {
        common_params_speculative p = make(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, 8, 2, false);
        p.ngram_size_n = 3;
        p.ngram_size_m = 4;
        common_speculative * spec = common_speculative_init(p, 2);
        CHECK(spec != nullptr);
        if (spec != nullptr) {
            common_speculative_begin(spec, 0, hist);
            common_speculative_begin(spec, 1, empty_hist);
            const llama_tokens d0 = draft_for(spec, 0, hist, 4);
            const llama_tokens d1 = draft_for(spec, 1, empty_hist, 5);
            printf("G  ngram_simple seq0 draft size %zu (first %d), seq1 draft size %zu\n", d0.size(), d0.empty() ? -1 : (int) d0[0], d1.size());
            CHECK(!d0.empty() && d0[0] == 1);
            CHECK(d1.empty());
            common_speculative_free(spec);
        } else {
            fprintf(stderr, "G  init returned null\n");
        }
    }

    // H: ngram_mod, n = 3, one shared table
    {
        common_params_speculative p = make(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, 8, 2, false);
        p.ngram_size_n = 3;
        common_speculative * spec = common_speculative_init(p, 2);
        CHECK(spec != nullptr);
        if (spec != nullptr) {
            common_speculative_begin(spec, 0, hist);
            common_speculative_begin(spec, 1, other_hist);

            const llama_tokens want = { 1, 2, 3, 4, 1, 2, 3, 4 };
            const llama_tokens d0 = draft_for(spec, 0, hist, 4);
            printf("H  ngram_mod seq0 draft:");
            for (auto t : d0) { printf(" %d", (int) t); }
            printf("\n");
            CHECK(d0 == want);

            // seq 1 asks with seq 0's tail: the shared table answers from seq 0's text
            const llama_tokens tail0 = { 9, 9, 9, 2, 3 };
            const llama_tokens d1 = draft_for(spec, 1, tail0, 4);
            printf("H  ngram_mod seq1 draft from the shared table: size %zu\n", d1.size());
            CHECK(!d1.empty() && d1[0] == 1);

            // seq 1: five rounds with nothing accepted -> the shared table is reset
            for (int r = 0; r < 5; ++r) {
                const llama_tokens dr = draft_for(spec, 1, other_hist, 8);
                CHECK(!dr.empty());
                common_speculative_accept(spec, 1, 0);
            }
            // nobody fed the table after the reset: seq 0 must re-feed its own history and draft again
            const llama_tokens d0b = draft_for(spec, 0, hist, 4);
            printf("H  ngram_mod seq0 draft after seq1's reset: size %zu\n", d0b.size());
            CHECK(d0b == want);
            common_speculative_free(spec);
        } else {
            fprintf(stderr, "H  init returned null\n");
        }
    }

    // I: ngram_cache, a second shorter request on the same slot
    {
        common_params_speculative p = make(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, 8, 2, false);
        common_speculative * spec = common_speculative_init(p, 2);
        CHECK(spec != nullptr);
        if (spec != nullptr) {
            llama_tokens long_hist;
            for (int r = 0; r < 6; ++r) {
                for (llama_token t : { 11, 12, 13, 14, 15, 16 }) {
                    long_hist.push_back(t);
                }
            }
            common_speculative_begin(spec, 0, long_hist);
            const llama_tokens da = draft_for(spec, 0, long_hist, 11);
            printf("I  ngram_cache first request draft size %zu\n", da.size());
            CHECK(!da.empty() && da[0] == 12);

            // new request on slot 0: shorter, different text
            common_speculative_begin(spec, 0, other_hist);
            const llama_tokens db = draft_for(spec, 0, other_hist, 8);
            printf("I  ngram_cache second (shorter) request draft:");
            for (auto t : db) { printf(" %d", (int) t); }
            printf("\n");
            CHECK(!db.empty() && db[0] == 5);

            common_speculative_begin(spec, 1, empty_hist);
            const llama_tokens dc = draft_for(spec, 1, empty_hist, 8);
            printf("I  ngram_cache seq1 (never given that text) draft size %zu\n", dc.size());
            CHECK(dc.empty() || dc[0] != 5);
            common_speculative_free(spec);
        } else {
            fprintf(stderr, "I  init returned null\n");
        }
    }

    if (n_fail != 0) {
        fprintf(stderr, "test-spec-pool-suffix: %d failure(s)\n", n_fail);
        return 1;
    }
    printf("test-spec-pool-suffix: all checks passed\n");
    return 0;
}
