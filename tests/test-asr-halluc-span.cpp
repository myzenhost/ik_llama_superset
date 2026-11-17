// test-asr-halluc-span.cpp -- the silence-threshold skip driven over a decoder that has no timestamp tokens
// (examples/audio/asr-halluc.h: asr_words_to_segments, asr_halluc_decode_span). A fake decoder with a known
// answer stands in for the model; the real-model check is examples/audio/tests/test-asr-halluc-cli.py.
#include "asr-halluc.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static asr_word W(const char * s, double a, double b, double p = 0.9) { asr_word w; w.word = s; w.start = a; w.end = b; w.probability = p; return w; }

// a confident sentence of 4 words, 0.3 s each, starting at t
static void speech(std::vector<asr_word> & v, double t, const char * last = "town.") {
    v.push_back(W("so", t, t + .3)); v.push_back(W("I", t + .3, t + .6)); v.push_back(W("went", t + .6, t + .9)); v.push_back(W(last, t + .9, t + 1.2));
}
// the classic fabricated caption: 20 ms words, improbable
static void halluc(std::vector<asr_word> & v, double t) {
    v.push_back(W("Thank", t, t + .02, .05)); v.push_back(W("you", t + .02, t + .04, .05));
    v.push_back(W("for", t + .04, t + .06, .05)); v.push_back(W("watching.", t + .06, t + .08, .05));
}
static std::string text_of(const std::vector<asr_word_segment> & segs) {
    std::string s;
    for (const auto & g : segs) { for (const auto & w : g.words) { if (!s.empty()) { s += ' '; } s += w.word; } }
    return s;
}

int main() {
    // ---- segmentation ----
    {
        std::vector<asr_word> v;
        speech(v, 0.0);                                  // ends "town."
        v.push_back(W("Then", 1.3, 1.5)); v.push_back(W("home", 1.5, 1.8));
        v.push_back(W("later", 5.0, 5.3));               // 3.2 s gap -> new segment
        v.push_back(W("\"Go.\"", 5.3, 5.5)); v.push_back(W("ok", 5.6, 5.8));
        const auto s = asr_words_to_segments(v, 1.0);
        CHECK(s.size() == 4);
        if (s.size() == 4) {
            CHECK(s[0].words.size() == 4 && s[0].start == 0.0 && std::fabs(s[0].end - 1.2) < 1e-12);
            CHECK(s[1].words.size() == 2);
            CHECK(s[2].words.size() == 2 && s[2].start == 5.0);   // "later \"Go.\"" -- the quoted stop ends it
            CHECK(s[3].words.size() == 1);
        }
        CHECK(asr_words_to_segments({}, 1.0).empty());
        CHECK(asr_words_to_segments(v, 0.0).size() == 3);         // gap splitting off
    }

    // ---- TRUNCATE: speech, 5 s silence, fabricated caption, silence, speech -> the caption is dropped ----
    // the decoder's answer depends on where it starts: from <= 3 s it sees everything; from later it sees
    // only what lies after its start (the hallucination appears only when the silence precedes it)
    {
        int calls = 0;
        std::vector<double> starts;
        auto dec = [&](double a, double b, std::vector<asr_word> & w) {
            ++calls; starts.push_back(a); (void) b;
            if (a <= 3.0) { speech(w, 0.5); halluc(w, 8.0); }
            speech(w, 15.0, "again.");
            return true;
        };
        asr_halluc_span_result r;
        CHECK(asr_halluc_decode_span(0.0, 20.0, 2.0, 20.0, 0.0, 1.0, 8, dec, r));
        std::printf("truncate: '%s' decodes=%d trunc=%d skip=%d\n", text_of(r.kept).c_str(), r.n_decodes, r.n_truncate, r.n_skip_leading);
        for (const auto & t : r.trace) { std::printf("  %s\n", t.c_str()); }
        CHECK(text_of(r.kept) == "so I went town. so I went again.");
        CHECK(r.n_decodes == 2 && r.n_truncate == 1 && r.n_skip_leading == 0);
        CHECK(starts.size() == 2 && std::fabs(starts[1] - 8.0) < 1e-9);   // resumes AT the dropped segment
        CHECK(std::fabs(r.last_speech - 16.2) < 1e-9);

        // negative control: a threshold no silence reaches keeps the caption -> the check, not the
        // segmentation, is what removed it above
        asr_halluc_span_result k;
        calls = 0; starts.clear();
        CHECK(asr_halluc_decode_span(0.0, 20.0, 100.0, 20.0, 0.0, 1.0, 8, dec, k));
        CHECK(text_of(k.kept) == "so I went town. Thank you for watching. so I went again.");
        CHECK(k.n_decodes == 1 && k.n_truncate == 0);
    }

    // ---- SKIP_LEADING: 6 s of leading silence decoded as a caption -> re-decode from where it started ----
    {
        std::vector<double> starts;
        auto dec = [&](double a, double, std::vector<asr_word> & w) {
            starts.push_back(a);
            if (a < 6.0) { halluc(w, 6.0); }
            speech(w, 7.0);
            return true;
        };
        asr_halluc_span_result r;
        CHECK(asr_halluc_decode_span(0.0, 30.0, 2.0, 30.0, 0.0, 1.0, 8, dec, r));
        for (const auto & t : r.trace) { std::printf("  %s\n", t.c_str()); }
        CHECK(text_of(r.kept) == "so I went town.");
        CHECK(r.n_skip_leading == 1 && r.n_decodes == 2);
        CHECK(starts.size() == 2 && std::fabs(starts[1] - 6.0) < 1e-9);
    }

    // ---- a window offset: times are FILE seconds, the span starts at 60 s ----
    {
        auto dec = [&](double a, double, std::vector<asr_word> & w) { if (a < 66.0) { halluc(w, 66.0); } speech(w, 67.0); return true; };
        asr_halluc_span_result r;
        CHECK(asr_halluc_decode_span(60.0, 90.0, 2.0, 120.0, 55.0, 1.0, 8, dec, r));
        CHECK(r.n_skip_leading == 1 && text_of(r.kept) == "so I went town.");
    }

    // ---- bounded: a decoder that hallucinates after every start ends at max_decodes ----
    {
        int calls = 0;
        auto dec = [&](double a, double, std::vector<asr_word> & w) { ++calls; halluc(w, a + 2.5); return true; };
        asr_halluc_span_result r;
        CHECK(asr_halluc_decode_span(0.0, 1000.0, 2.0, 1000.0, 0.0, 1.0, 5, dec, r));
        CHECK(calls == 5 && r.n_decodes == 5 && r.kept.empty());
        CHECK(!r.trace.empty() && r.trace.back().find("decode limit") != std::string::npos);
    }

    // ---- clean speech is one decode, kept whole; a decoder failure is returned ----
    {
        auto dec = [&](double, double, std::vector<asr_word> & w) { speech(w, 0.2); speech(w, 1.6, "home."); return true; };
        asr_halluc_span_result r;
        CHECK(asr_halluc_decode_span(0.0, 3.0, 2.0, 3.0, 0.0, 1.0, 8, dec, r));
        CHECK(r.n_decodes == 1 && r.kept.size() == 2);
        auto bad = [&](double, double, std::vector<asr_word> &) { return false; };
        CHECK(!asr_halluc_decode_span(0.0, 3.0, 2.0, 3.0, 0.0, 1.0, 8, bad, r));
    }

    // ---- word times on a coarse grid (time_resolution) ----
    // MEASURED 2026-10-19, Voxtral-Mini-3B Q4_K_M, llama-audio DTW (0.1604 s per audio position): clip A then 12 s
    // of silence. Two real words share one position (zero duration), each scoring (0.133-0)*15 = 2.0, so without
    // the floor the real sentence is anomalous and was dropped (probe_sw_hs2: "keep 0").
    const double RES = 0.1604;
    auto real_a = [](std::vector<asr_word> & v, double t) {
        v.push_back(W("Concord", t + 0.000, t + 1.123, .682)); v.push_back(W("returned", t + 1.123, t + 1.765, .849));
        v.push_back(W("to", t + 1.765, t + 2.727, .835));      v.push_back(W("its", t + 2.727, t + 2.727, .948));
        v.push_back(W("place", t + 2.727, t + 2.727, .993));   v.push_back(W("amidst", t + 2.727, t + 3.048, .884));
        v.push_back(W("the", t + 3.048, t + 3.209, .953));     v.push_back(W("tents.", t + 3.209, t + 3.369, .942));
    };
    {
        std::vector<asr_word> v; real_a(v, 0.0);
        const auto s = asr_words_to_segments(v, 1.0);
        CHECK(s.size() == 1);
        CHECK(asr_segment_is_anomaly(s[0]));            // the failure, reproduced: exact-time scoring
        CHECK(!asr_segment_is_anomaly(s[0], RES));      // one step cannot be measured below
        CHECK(std::fabs(asr_word_anomaly_score(v[3], RES)) < 1e-12);
        CHECK(std::fabs(asr_word_anomaly_score(v[3]) - 0.133 * 15) < 1e-9);
    }
    {
        // a finer grid than the limit keeps the short-word term: 0.05 s step, zero-duration word -> (0.133-0.05)*15
        CHECK(std::fabs(asr_word_anomaly_score(W("a", 1, 1, .9), 0.05) - 0.083 * 15) < 1e-9);
        // the probability and long-word terms do not change with the grid
        CHECK(std::fabs(asr_word_anomaly_score(W("a", 0, 3.5, .1), RES) - 2.5) < 1e-9);
    }
    {
        // MEASURED 2026-10-19, same model: 20 s of -50 dBFS noise -> "Yeah, I think it's a great idea." repeated,
        // the first word spanning 0-19.893 and the rest piled at 19.893. Still anomalous on the coarse grid.
        std::vector<asr_word> v;
        v.push_back(W("\"Yeah,", 0.0, 19.893, .281)); v.push_back(W("I", 19.893, 19.893, .153));
        v.push_back(W("think", 19.893, 19.893, .131)); v.push_back(W("it's", 19.893, 19.893, .408));
        v.push_back(W("a", 19.893, 19.893, .135));     v.push_back(W("great", 19.893, 19.893, .106));
        v.push_back(W("idea.", 19.893, 19.893, .324));
        const auto s = asr_words_to_segments(v, 1.0);
        CHECK(!s.empty() && asr_segment_is_anomaly(s[0], RES));
        // and the fabricated caption above (20 ms improbable words) too
        std::vector<asr_word> h; halluc(h, 5.0);
        CHECK(asr_segment_is_anomaly(asr_words_to_segments(h, 1.0)[0], RES));
    }
    {
        // the decode loop over A + 12 s silence: kept whole with the grid, dropped without it
        auto dec = [&](double a, double, std::vector<asr_word> & w) { if (a < 1e-6) { real_a(w, 0.0); } return true; };
        asr_halluc_span_result r0, r1;
        CHECK(asr_halluc_decode_span(0.0, 15.505, 2.0, 15.505, 0.0, 2.0, 16, dec, r0));
        CHECK(r0.kept.empty());                                       // the failure
        CHECK(asr_halluc_decode_span(0.0, 15.505, 2.0, 15.505, 0.0, 2.0, 16, dec, r1, RES));
        CHECK(r1.n_decodes == 1 && text_of(r1.kept) == "Concord returned to its place amidst the tents.");
        // the noise hallucination is still dropped on the grid
        auto dn = [&](double a, double, std::vector<asr_word> & w) {
            if (a < 1e-6) {
                w.push_back(W("\"Yeah,", 0.0, 19.893, .281));
                for (int k = 0; k < 6; ++k) { w.push_back(W(k == 5 ? "idea." : "I", 19.893, 19.893, .13)); }
            }
            return true;
        };
        asr_halluc_span_result rn;
        CHECK(asr_halluc_decode_span(0.0, 20.0, 2.0, 20.0, 0.0, 2.0, 16, dn, rn, RES));
        CHECK(rn.kept.empty() && rn.n_truncate >= 1);
    }

    std::printf("%s (%d failure(s))\n", g_fail ? "FAILED" : "OK", g_fail);
    return g_fail ? 1 : 0;
}
