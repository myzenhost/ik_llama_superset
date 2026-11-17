// Model-free test of examples/audio/asr-fallback: quality-gated temperature-fallback policy.
#include "asr-fallback.h"
#include "asr-compression.h"

#include <cmath>
#include <cstdio>
#include <string>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static std::string rep(const std::string & s, int n) { std::string o; for (int i = 0; i < n; ++i) o += s; return o; }
static const char * PROSE = "So I went to the store yesterday and bought some apples, then I walked home along the river.";

static asr_decode_attempt att(const std::string & text, double lp, double nsp = 0.0, int ntok = 20, double ent = 1e9) {
    asr_decode_attempt a; a.text = text; a.avg_logprob = lp; a.no_speech_prob = nsp; a.n_tokens = ntok; a.tail_entropy = ent; return a;
}

int main() {
    asr_fallback_params p;

    // ---- ladder
    auto lad = asr_temperature_ladder(0.0f, 0.2f);
    printf("ladder:"); for (float t : lad) printf(" %.2f", t); printf("\n");
    CHECK(lad.size() == 6);
    CHECK(std::fabs(lad[5] - 1.0f) < 1e-6f);
    CHECK(asr_temperature_ladder(0.0f, 0.0f).size() == 1);
    CHECK(asr_temperature_ladder(0.4f, 0.3f).size() == 3);   // 0.4 0.7 1.0
    CHECK(asr_temperature_ladder(1.5f, 0.2f).size() == 1);

    // ---- entropy helper
    CHECK(std::fabs(asr_tail_entropy({7, 7, 7, 7}) - 0.0) < 1e-12);
    CHECK(std::fabs(asr_tail_entropy({1, 2, 3, 4}) - std::log(4.0)) < 1e-12);
    // window: only the last 2 tokens count
    CHECK(std::fabs(asr_tail_entropy({1, 2, 3, 9, 9}, 2) - 0.0) < 1e-12);

    // ---- single-attempt verdicts
    { auto v = asr_fallback_check(p, att(PROSE, -0.3));                 CHECK(!v.needs_fallback && v.reasons == 0); }     // control: good
    { auto v = asr_fallback_check(p, att(rep("the ", 100), -0.3));      CHECK(v.needs_fallback && (v.reasons & ASR_FAIL_COMPRESSION)); }
    { auto v = asr_fallback_check(p, att(PROSE, -1.5));                 CHECK(v.needs_fallback && (v.reasons & ASR_FAIL_LOGPROB)); }
    { auto v = asr_fallback_check(p, att(PROSE, -1.0));                 CHECK(!v.needs_fallback); }                       // strict <
    { auto v = asr_fallback_check(p, att(PROSE, -0.3, 0, 40, 1.0));     CHECK(v.needs_fallback && (v.reasons & ASR_FAIL_ENTROPY)); }
    { auto v = asr_fallback_check(p, att(PROSE, -0.3, 0, 32, 1.0));     CHECK(!v.needs_fallback); }                       // needs > 32 tokens
    // silence override: low logprob but high no-speech => accept
    { auto v = asr_fallback_check(p, att("", -1.5, 0.9));               CHECK(!v.needs_fallback && v.is_silence); }
    { auto v = asr_fallback_check(p, att("", -1.5, 0.6));               CHECK(v.needs_fallback && !v.is_silence); }       // strict >
    // high no-speech alone with good logprob is not silence
    { auto v = asr_fallback_check(p, att(PROSE, -0.3, 0.9));            CHECK(!v.needs_fallback && !v.is_silence); }
    // gates can be disabled
    { auto q = p; q.check_logprob = false; auto v = asr_fallback_check(q, att(PROSE, -5.0)); CHECK(!v.needs_fallback); }
    { auto q = p; q.check_compression = false; auto v = asr_fallback_check(q, att(rep("the ", 100), -0.3)); CHECK(!v.needs_fallback); }

    // ---- driver: loops at T=0 and T=0.2, clean at T=0.4
    {
        int calls = 0;
        auto r = asr_run_fallback(p, [&](float t) {
            calls++;
            if (t < 0.3f) return att(rep("the ", 100), -0.3);
            return att(PROSE, -0.4);
        });
        printf("driver A: calls=%d chosen=%d T=%.2f exhausted=%d\n", calls, r.chosen, r.attempts[r.chosen].temperature, (int) r.exhausted);
        CHECK(calls == 3 && r.chosen == 2 && !r.exhausted);
        CHECK(std::fabs(r.attempts[2].temperature - 0.4f) < 1e-6f);
    }
    // ---- driver: first attempt good => exactly one decode
    {
        int calls = 0;
        auto r = asr_run_fallback(p, [&](float) { calls++; return att(PROSE, -0.2); });
        CHECK(calls == 1 && r.chosen == 0);
    }
    // ---- driver: everything fails; faster-whisper rule = best logprob among under-compression
    {
        int calls = 0;
        auto r = asr_run_fallback(p, [&](float t) {
            calls++;
            if (t < 0.5f) return att(rep("the ", 100), -0.1);  // looping, best logprob but over the ratio
            return att(PROSE, -1.0 - t);                       // low logprob; best of these is the smallest t
        });
        printf("driver B: calls=%d chosen=%d T=%.2f exhausted=%d\n", calls, r.chosen, r.attempts[r.chosen].temperature, (int) r.exhausted);
        CHECK(calls == 6 && r.exhausted);
        CHECK(std::fabs(r.attempts[r.chosen].temperature - 0.6f) < 1e-6f);
    }
    // ---- same, whisper.cpp rule: last attempt
    {
        auto q = p; q.on_exhausted = asr_fallback_params::EXHAUSTED_LAST;
        auto r = asr_run_fallback(q, [&](float) { return att(rep("the ", 100), -2.0); });
        CHECK(r.exhausted && r.chosen == 5);
    }
    // ---- all-looping: no attempt under the ratio => falls back to best over all
    {
        auto r = asr_run_fallback(p, [&](float t) { return att(rep("the ", 100), -1.0 - t); });
        CHECK(r.exhausted && r.chosen == 0);
    }
    // ---- silence accepted at the first attempt
    {
        int calls = 0;
        auto r = asr_run_fallback(p, [&](float) { calls++; return att("", -2.0, 0.95); });
        CHECK(calls == 1 && r.verdicts[0].is_silence);
    }

    printf(n_fail ? "test-asr-fallback: %d FAILED\n" : "test-asr-fallback: OK\n", n_fail);
    return n_fail ? 1 : 0;
}
