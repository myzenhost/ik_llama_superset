// Model-free test of examples/audio/asr-halluc: hallucination silence-threshold skip.
#include "asr-halluc.h"

#include <cmath>
#include <cstdio>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

static asr_word W(const char * s, double a, double b, double p = 0.9) { asr_word w; w.word = s; w.start = a; w.end = b; w.probability = p; return w; }
static asr_word_segment S(std::vector<asr_word> ws) { asr_word_segment s; s.words = ws; s.start = ws.front().start; s.end = ws.back().end; return s; }

// normal speech: 5 words of ~0.3 s, confident
static asr_word_segment good(double t0) {
    return S({W(" so", t0, t0 + .3), W(" I", t0 + .3, t0 + .5), W(" went", t0 + .5, t0 + .9), W(" to", t0 + .9, t0 + 1.1), W(" town", t0 + 1.1, t0 + 1.6)});
}
// hallucination: all words are 20 ms and improbable
static asr_word_segment bad(double t0) {
    return S({W(" Thank", t0, t0 + .02, .05), W(" you", t0 + .02, t0 + .04, .05), W(" for", t0 + .04, t0 + .06, .05), W(" watching", t0 + .06, t0 + .08, .05)});
}

int main() {
    // ---- word / segment scoring
    CHECK(std::fabs(asr_word_anomaly_score(W("a", 0, .3, .9))) < 1e-12);                         // control
    CHECK(std::fabs(asr_word_anomaly_score(W("a", 0, .3, .1)) - 1.0) < 1e-12);                   // low prob
    CHECK(std::fabs(asr_word_anomaly_score(W("a", 0, .033, .9)) - (0.1 * 15)) < 1e-9);           // short
    CHECK(std::fabs(asr_word_anomaly_score(W("a", 0, 3.5, .9)) - 1.5) < 1e-9);                   // long
    CHECK(!asr_segment_is_anomaly(good(1.0)));
    CHECK(asr_segment_is_anomaly(bad(1.0)));
    CHECK(!asr_segment_is_anomaly(asr_word_segment{}));
    { // punctuation words are ignored: a lone "." must not make a clean segment anomalous
        auto s = good(1.0); s.words.push_back(W(".", 1.6, 1.6, 0.01)); CHECK(!asr_segment_is_anomaly(s)); }
    { // only the first 8 words count: 8 clean words then garbage => clean
        asr_word_segment s; for (int i = 0; i < 8; ++i) s.words.push_back(W(" w", i * .3, i * .3 + .3)); for (int i = 0; i < 6; ++i) s.words.push_back(W(" z", 3 + i * .02, 3.02 + i * .02, .01));
        CHECK(!asr_segment_is_anomaly(s)); }

    asr_halluc_params p; p.threshold = 2.0; p.time_offset = 0; p.segment_duration = 30; p.window_end_time = 30; p.content_duration = 60; p.last_speech_timestamp = 0;

    // ---- control: clean speech => KEEP
    { auto d = asr_halluc_check(p, {good(0.5), good(3.0), good(6.0)}); CHECK(d.action == ASR_HALLUC_KEEP); }

    // ---- leading hallucination after 5 s of silence => skip leading, resume at its start
    { auto d = asr_halluc_check(p, {bad(5.0), good(12.0)});
      printf("leading: action=%d seek=%.2f\n", d.action, d.seek_time);
      CHECK(d.action == ASR_HALLUC_SKIP_LEADING && std::fabs(d.seek_time - 5.0) < 1e-9); }
    // same but gap below threshold (1.5 s) => not skipped as leading, and not silence-bounded either
    { auto d = asr_halluc_check(p, {bad(1.5), good(3.0)}); CHECK(d.action != ASR_HALLUC_SKIP_LEADING); }

    // ---- hallucination in the middle surrounded by silence => truncate from it
    { auto d = asr_halluc_check(p, {good(10.0), bad(20.0)});   // 20 -> next start = window end (30), 10 s of silence after
      printf("middle: action=%d keep=%zu seek=%.2f\n", d.action, d.n_keep, d.seek_time);
      CHECK(d.action == ASR_HALLUC_TRUNCATE && d.n_keep == 1 && std::fabs(d.seek_time - 20.0) < 1e-9 && !d.seek_to_end); }
    // hallucination ABUTTING real speech (no silence either side) => KEEP
    { auto d = asr_halluc_check(p, {good(10.0), bad(12.0), good(12.3)}); CHECK(d.action == ASR_HALLUC_KEEP); }
    // hallucination followed by another anomaly counts as "silence after"
    { auto d = asr_halluc_check(p, {good(10.0), bad(14.0), bad(14.3)}); CHECK(d.action == ASR_HALLUC_TRUNCATE && d.n_keep == 1); }
    // near the end of the content => jump to the end
    { auto q = p; q.content_duration = 21.0; q.window_end_time = 30;
      auto d = asr_halluc_check(q, {good(10.0), bad(20.0)}); CHECK(d.action == ASR_HALLUC_TRUNCATE && d.seek_to_end && d.seek_time == 21.0); }
    // seek never rewinds before time_offset + 1
    { auto q = p; q.time_offset = 10.0; q.segment_duration = 30; q.window_end_time = 40; q.last_speech_timestamp = 0; q.content_duration = 100;
      auto d = asr_halluc_check(q, {bad(10.4)});
      printf("rewind: action=%d seek=%.2f\n", d.action, d.seek_time);
      CHECK(d.action == ASR_HALLUC_KEEP || d.seek_time >= 11.0); }

    printf(n_fail ? "test-asr-halluc: %d FAILED\n" : "test-asr-halluc: OK\n", n_fail);
    return n_fail ? 1 : 0;
}
