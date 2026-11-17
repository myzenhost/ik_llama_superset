// Model-free test of examples/audio/asr-vad-post: Silero-style VAD post-processing state machine.
// Probabilities are synthetic; expected spans are derived by hand from the documented rules
// (window 512 samples @16 kHz = 32 ms/prob, pad 30 ms = 480 samples).
#include "asr-vad-post.h"

#include <cstdio>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

// build a probability array from (value, n_frames) runs
static std::vector<float> P(std::initializer_list<std::pair<float, int>> runs) {
    std::vector<float> v;
    for (auto & r : runs) v.insert(v.end(), (size_t) r.second, r.first);
    return v;
}
static void dump(const char * n, const std::vector<asr_vad_span> & s) {
    printf("%s:", n); for (auto & x : s) printf(" [%lld,%lld)", (long long) x.start, (long long) x.end); printf("\n");
}

int main() {
    asr_vad_post_params p;

    // 1) 10 silent, 30 speech, 20 silent frames: speech [5120, 20480) padded by 480
    {
        auto pr = P({{0.1f, 10}, {0.9f, 30}, {0.1f, 20}});
        auto s = asr_vad_postprocess(pr.data(), pr.size(), p); dump("basic", s);
        CHECK(s.size() == 1 && s[0].start == 4640 && s[0].end == 20960);
    }
    // 2) all silence => nothing; empty input => nothing
    {
        auto pr = P({{0.05f, 50}});
        CHECK(asr_vad_postprocess(pr.data(), pr.size(), p).empty());
        CHECK(asr_vad_postprocess(nullptr, 0, p).empty());
    }
    // 3) a 3-frame (96 ms) blip is shorter than min_speech (250 ms) => dropped
    {
        auto pr = P({{0.1f, 10}, {0.9f, 3}, {0.1f, 20}});
        CHECK(asr_vad_postprocess(pr.data(), pr.size(), p).empty());
    }
    // 4) hysteresis: a dip to 0.4 (>= 0.35 neg threshold, < 0.5) does NOT end the segment
    {
        auto pr = P({{0.1f, 5}, {0.9f, 15}, {0.4f, 10}, {0.9f, 15}, {0.1f, 20}});
        auto s = asr_vad_postprocess(pr.data(), pr.size(), p); dump("hyst", s);
        CHECK(s.size() == 1);
        // control: same dip at 0.2 (< 0.35) for 10 frames (320 ms >= min_silence) DOES split
        auto pr2 = P({{0.1f, 5}, {0.9f, 15}, {0.2f, 10}, {0.9f, 15}, {0.1f, 20}});
        auto s2 = asr_vad_postprocess(pr2.data(), pr2.size(), p); dump("split", s2);
        CHECK(s2.size() == 2);
    }
    // 5) merge gap: 5 silent frames (160 ms < 200 ms) merge; 10 frames (320 ms) do not
    {
        auto a = P({{0.9f, 20}, {0.1f, 5}, {0.9f, 20}, {0.1f, 20}});
        auto b = P({{0.9f, 20}, {0.1f, 10}, {0.9f, 20}, {0.1f, 20}});
        CHECK(asr_vad_postprocess(a.data(), a.size(), p).size() == 1);
        CHECK(asr_vad_postprocess(b.data(), b.size(), p).size() == 2);
        auto q = p; q.merge_gap_ms = 0;
        CHECK(asr_vad_postprocess(a.data(), a.size(), q).size() == 2);   // merge disabled
    }
    // 6) min_silence: 2-frame silence (64 ms < 100 ms) never closes a segment
    {
        auto pr = P({{0.9f, 20}, {0.1f, 2}, {0.9f, 20}, {0.1f, 20}});
        auto q = p; q.merge_gap_ms = 0;
        CHECK(asr_vad_postprocess(pr.data(), pr.size(), q).size() == 1);
    }
    // 7) padding at the file edges clamps to [0, audio_len]
    {
        auto pr = P({{0.9f, 30}});
        auto s = asr_vad_postprocess(pr.data(), pr.size(), p); dump("edges", s);
        CHECK(s.size() == 1 && s[0].start == 0 && s[0].end == 30 * 512);
    }
    // 8) max-speech splitting: 100 frames of speech with an 8-frame dip at 40..47, max 2 s.
    //    limit = 16000*2 - 512 - 960 = 30528 samples; split at the dip: [0,20480) and [24576,51200),
    //    padded: [0,20960) and [24096,51200)
    {
        auto pr = P({{0.9f, 40}, {0.1f, 8}, {0.9f, 52}});
        auto q = p; q.max_speech_s = 2.0f; q.min_silence_ms = 500;
        auto s = asr_vad_postprocess(pr.data(), pr.size(), q); dump("maxsplit", s);
        CHECK(s.size() == 2);
        CHECK(s.size() == 2 && s[0].start == 0 && s[0].end == 20960 && s[1].start == 24096 && s[1].end == 51200);
        // control: no limit => single segment spanning everything
        auto s1 = asr_vad_postprocess(pr.data(), pr.size(), [&] { auto z = p; z.min_silence_ms = 500; return z; }());
        CHECK(s1.size() == 1);
    }
    // 9) over-long speech with NO silence inside is hard-cut at the limit
    {
        auto pr = P({{0.9f, 100}});
        auto q = p; q.max_speech_s = 2.0f;
        auto s = asr_vad_postprocess(pr.data(), pr.size(), q); dump("hardcut(default merge)", s);
        // upstream quirk, reproduced on purpose: the 1-frame gap left by the hard cut is < 200 ms,
        // so the merge step glues the pieces back together
        CHECK(s.size() == 1);
        q.merge_gap_ms = 0;
        auto s0 = asr_vad_postprocess(pr.data(), pr.size(), q); dump("hardcut(merge off)", s0);
        CHECK(s0.size() >= 2);
        CHECK(s0.size() >= 2 && s0[0].start == 0 && s0[0].end - s0[0].start <= 16000 * 2 + 2 * 480);
    }

    printf(n_fail ? "test-asr-vad-post: %d FAILED\n" : "test-asr-vad-post: OK\n", n_fail);
    return n_fail ? 1 : 0;
}
