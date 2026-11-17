// test-audio-stream.cpp -- examples/audio/audio-stream: the capture ring and whisper.cpp's live sliding window.
// Expected values are worked out by hand from examples/stream/stream.cpp:129-139, 278-291, 408-412 (whisper.cpp
// 6e4ab854), not computed with the code under test.

#include "audio-stream.h"
#include "asr-stream-vad.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static std::vector<float> ramp(float from, size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) { v[i] = from + (float) i; }
    return v;
}

int main() {
    // 1. ring: in-order take, wrap-around, overflow keeps the newest and reports the loss
    {
        audio_ring r(8);
        auto a = ramp(0, 5);
        r.push(a.data(), a.size());
        std::vector<float> out;
        size_t had = r.take(out);
        CHECK(had == 5 && out == a, "take after 5: had %zu, got %zu", had, out.size());
        CHECK(r.pending() == 0, "pending after take");
        auto b = ramp(5, 6);                     // positions 5,6,7,0,1,2: wraps
        r.push(b.data(), b.size());
        had = r.take(out);
        CHECK(had == 6 && out == b, "wrap: had %zu", had);
        auto c = ramp(100, 12);                  // more than the capacity in one go
        r.push(c.data(), c.size());
        had = r.take(out);
        CHECK(had == 12 && out == ramp(104, 8), "overflow: had %zu, first %.0f", had, out.empty() ? -1.0 : out[0]);
        auto d = ramp(200, 5), e = ramp(205, 5); // two pushes, 10 > 8 pending: the newest 8 survive
        r.push(d.data(), d.size());
        r.push(e.data(), e.size());
        had = r.take(out);
        CHECK(had == 10 && out == ramp(202, 8), "two pushes over capacity: had %zu, first %.0f", had, out.empty() ? -1.0 : out[0]);
        CHECK(r.total() == 5 + 6 + 12 + 10, "total %llu", (unsigned long long) r.total());
    }
    // 2. window arithmetic: step 3 s, length 10 s, keep 0.2 s at 10 Hz -> 30 / 100 / 2 samples, a line every
    //    max(1, 10/3 - 1) = 2 steps. Hand trace of stream.cpp:
    //    step 0: take min(0, 2+100-30) = 0 old      -> [b0]          30, no line
    //    step 1: take min(30, 72) = 30 old           -> [b0 b1]       60, LINE; old = last 2 of b1
    //    step 2: take min(2, 72) = 2                 -> [b1[28:] b2]  32, no line, starts at 5.8 s
    //    step 3: take min(32, 72) = 32               -> [.. b2 b3]    62, LINE
    {
        audio_stream_params p;
        p.step_ms = 3000; p.length_ms = 10000; p.keep_ms = 200; p.sample_rate = 10;
        audio_stream_window w(p);
        CHECK(w.n_step() == 30 && w.n_len() == 100 && w.n_keep() == 2 && w.n_new_line() == 2, "derived sizes");
        const size_t exp_size[4] = {30, 60, 32, 62};
        const bool   exp_line[4] = {false, true, false, true};
        const double exp_t0[4]   = {0.0, 0.0, 5.8, 5.8};
        const double exp_t1[4]   = {3.0, 6.0, 9.0, 12.0};
        for (int s = 0; s < 4; ++s) {
            auto fresh = ramp(1000.0f * s, 30);
            const auto & win = w.next(fresh);
            CHECK(win.size() == exp_size[s], "step %d: window %zu, expected %zu", s, win.size(), exp_size[s]);
            CHECK(std::fabs(w.t0() - exp_t0[s]) < 1e-9 && std::fabs(w.t1() - exp_t1[s]) < 1e-9,
                  "step %d: [%.2f, %.2f) expected [%.2f, %.2f)", s, w.t0(), w.t1(), exp_t0[s], exp_t1[s]);
            CHECK(win.back() == 1000.0f * s + 29, "step %d: window must end with the fresh samples", s);
            if (s == 2) {
                CHECK(win[0] == 1000.0f + 28 && win[1] == 1000.0f + 29 && win[2] == 2000.0f, "step 2: keep carries b1[28:30]");
            }
            if (s == 1) {
                CHECK(win[0] == 0.0f && win[30] == 1000.0f, "step 1: [b0 b1]");
            }
            const bool line = w.done_step();
            CHECK(line == exp_line[s], "step %d: line %d expected %d", s, line, exp_line[s]);
        }
    }
    // 3. the length cap: with a long line (length 10 s, step 1 s -> a line every 9 steps) the window never exceeds
    //    keep + length, and the take shrinks once the old part would push it past that
    {
        audio_stream_params p;
        p.step_ms = 1000; p.length_ms = 10000; p.keep_ms = 200; p.sample_rate = 10;
        audio_stream_window w(p);
        CHECK(w.n_new_line() == 9, "n_new_line %d", w.n_new_line());
        size_t biggest = 0;
        for (int s = 0; s < 30; ++s) {
            auto fresh = ramp(0, 10);
            biggest = std::max(biggest, w.next(fresh).size());
            w.done_step();
        }
        CHECK(biggest <= 102, "window grew to %zu > keep + length = 102", biggest);
    }
    // 4. clamping as stream.cpp:129-130: keep <= step, length >= step
    {
        audio_stream_params p;
        p.step_ms = 2000; p.length_ms = 500; p.keep_ms = 5000; p.sample_rate = 100;
        audio_stream_window w(p);
        CHECK(w.n_keep() == 200 && w.n_len() == 200 && w.n_new_line() == 1, "clamp: keep %d len %d nl %d", w.n_keep(), w.n_len(), w.n_new_line());
    }
    // 5. a producer thread (the capture callback) against a consumer: nothing lost, nothing reordered
    {
        audio_ring r(4096);
        const size_t N = 200000;
        std::thread prod([&]() {
            std::vector<float> blk(7);
            for (size_t i = 0; i < N; i += blk.size()) {
                const size_t n = std::min(blk.size(), N - i);
                for (size_t k = 0; k < n; ++k) { blk[k] = (float) (i + k); }
                r.push(blk.data(), n);
                if ((i / 7) % 64 == 0) { std::this_thread::yield(); }
            }
        });
        // exact accounting: a take that reports `had` pushed samples and returns part.size() of them must hold
        // exactly the values next + had - part.size() .. next + had - 1 (the newest ones, in order)
        std::vector<float> part;
        size_t next = 0, taken = 0, lost = 0, bad = 0;
        while (next < N) {
            const size_t had = r.take(part);
            const size_t first = next + had - part.size();
            for (size_t k = 0; k < part.size(); ++k) { bad += part[k] != (float) (first + k); }
            next  += had;
            taken += part.size();
            lost  += had - part.size();
        }
        prod.join();
        CHECK(bad == 0, "threaded: %zu sample(s) not where the accounting puts them", bad);
        CHECK(next == N && taken + lost == N, "threaded: %zu taken + %zu lost != %zu", taken, lost, N);
        printf("threaded: %zu taken, %zu reported lost, %zu misplaced (ring 4096)\n", taken, lost, bad);
    }

    // stream VAD decisions (row S69-ENGREV-WHISPERCPP-STREAMVAD-2026-10-27-01): one probability = 32 ms; a 1 s step
    // is 31 probabilities (31 * 32 ms = 992 ms).
    {
        const size_t N = 31;
        auto step = [&](asr_stream_vad_state & s, const asr_stream_vad_params & p, const std::vector<float> & pr) {
            return asr_stream_vad_step(s, p, pr.data(), pr.size());
        };
        const std::vector<float> speech(N, 0.9f), silent(N, 0.01f);
        asr_stream_vad_params p;   // threshold 0.5, endpoint after 800 ms of silence

        // leading silence: nothing to transcribe, not decoded, no endpoint
        asr_stream_vad_state s;
        auto d = step(s, p, silent);
        CHECK(!d.decode && !d.endpoint && !d.step_speech, "leading silence must be skipped (decode %d endpoint %d)", d.decode, d.endpoint);
        d = step(s, p, silent);
        CHECK(!d.decode && !d.endpoint, "second silent step must be skipped too");
        // speech: decoded, the line is open
        d = step(s, p, speech);
        CHECK(d.decode && d.step_speech && !d.endpoint, "a speech step must be decoded, not ended");
        // 0.6 s pause inside a sentence, then speech again in the same step: no endpoint (the row's test)
        std::vector<float> pause_mid(N, 0.9f);
        for (size_t i = 5; i < 5 + 19; i++) pause_mid[i] = 0.01f;            // 19 * 32 = 608 ms of silence
        d = step(s, p, pause_mid);
        CHECK(d.decode && !d.endpoint, "a 0.6 s mid-sentence pause must NOT end the line");
        // trailing silence: one silent step is 992 ms >= 800 ms -> endpoint, still decoded (final text of the line)
        d = step(s, p, silent);
        CHECK(d.decode && d.endpoint, "0.99 s of trailing silence must end the line (decode %d endpoint %d)", d.decode, d.endpoint);
        // after the endpoint the line is closed: further silence is skipped again
        d = step(s, p, silent);
        CHECK(!d.decode && !d.endpoint, "silence after an endpoint must be skipped");
        CHECK(s.n_decode == 3 && s.n_skip == 3 && s.n_endpoint == 1, "counters: decode %lld skip %lld endpoint %lld",
              (long long) s.n_decode, (long long) s.n_skip, (long long) s.n_endpoint);

        // silence shorter than the limit at a step's end does not endpoint; the run continues into the next step
        asr_stream_vad_state s2;
        std::vector<float> tail_short(N, 0.9f);
        for (size_t i = N - 10; i < N; i++) tail_short[i] = 0.01f;           // 320 ms trailing
        d = step(s2, p, tail_short);
        CHECK(d.decode && !d.endpoint, "320 ms trailing silence must not end the line");
        std::vector<float> head_sil(N, 0.9f);
        for (size_t i = 0; i < 15; i++) head_sil[i] = 0.01f;                  // +480 ms = 800 ms, but speech follows
        d = step(s2, p, head_sil);
        CHECK(d.decode && !d.endpoint, "a silence run that speech interrupts must not end the line");
        d = step(s2, p, silent);
        CHECK(d.endpoint, "then a silent step ends it");

        // endpoint_silence_ms = 0 turns endpointing off; silent steps of an open line are still decoded
        asr_stream_vad_params p0; p0.endpoint_silence_ms = 0;
        asr_stream_vad_state s3;
        step(s3, p0, speech);
        d = step(s3, p0, silent);
        CHECK(d.decode && !d.endpoint, "endpoint off: no endpoint, line still open and decoded");
        // control: threshold decides -- the same probabilities under a higher threshold are silence
        asr_stream_vad_params ph; ph.threshold = 0.95f;
        asr_stream_vad_state s4;
        d = step(s4, ph, speech);
        CHECK(!d.step_speech && !d.decode, "0.9 under threshold 0.95 is not speech");
        printf("stream vad: decisions ok\n");
    }
    // end_line(): only `keep` samples carry over, and the fixed line count starts again
    {
        audio_stream_params sp;
        sp.step_ms = 1000; sp.length_ms = 4000; sp.keep_ms = 200; sp.sample_rate = 1000;   // n_new_line = 3
        audio_stream_window w(sp);
        w.next(ramp(0.0f, 1000));
        CHECK(!w.done_step(), "step 1 of 3 must not end the line");
        const auto & w2 = w.next(ramp(1000.0f, 1000));
        CHECK(w2.size() == 2000, "before end_line the window grows (%zu)", w2.size());
        w.end_line();
        const auto & w3 = w.next(ramp(2000.0f, 1000));
        CHECK(w3.size() == 1200 && w3.front() == 1800.0f, "after end_line only keep (200) + step carry: size %zu first %.0f",
              w3.size(), w3.front());
        CHECK(!w.done_step(), "the line count restarted: step 1 of 3 again");
        CHECK(!w.done_step(), "step 2 of 3");
        CHECK(w.done_step(), "step 3 of 3 ends the line");
        printf("end_line: ok\n");
    }

    // 8. token-driven end of utterance: frames of 80 ms, 10 blank frames (800 ms) after the last piece end the line
    {
        struct tk { int frame; int duration; };
        asr_stream_eou_params p; p.eou_blank_frames = 10;
        const std::vector<tk> none;
        CHECK(asr_stream_trailing_blank_frames(none.data(), 0, 40) == -1, "no piece: -1");
        CHECK(!asr_stream_eou(p, none.data(), 0, 100), "silence only (no piece) never endpoints");
        const std::vector<tk> sp = { {2, 1}, {5, 2}, {9, 1}, {20, 1} };   // last piece ends at frame 21
        // 0.6 s mid-sentence pause = 7 whole frames after frame 21: window of 28 frames
        CHECK(asr_stream_trailing_blank_frames(sp.data(), sp.size(), 28) == 7, "pause run is 7 frames");
        CHECK(!asr_stream_eou(p, sp.data(), sp.size(), 28), "0.6 s pause must not endpoint");
        CHECK(!asr_stream_eou(p, sp.data(), sp.size(), 30), "9 blank frames must not endpoint");
        CHECK(asr_stream_eou(p, sp.data(), sp.size(), 31), "exactly 10 blank frames endpoints");
        CHECK(asr_stream_eou(p, sp.data(), sp.size(), 35), "0.8 s+ trailing silence endpoints");
        // speech resumes after the pause: more pieces move the end forward, the run restarts
        std::vector<tk> sp2 = sp; sp2.push_back({30, 1});
        CHECK(!asr_stream_eou(p, sp2.data(), sp2.size(), 35), "resumed speech: only 4 blank frames");
        CHECK(asr_stream_eou(p, sp2.data(), sp2.size(), 41), "and it endpoints 10 frames after the new end");
        const std::vector<tk> z = { {4, 0} };   // duration 0 counts as one frame: end 5
        CHECK(asr_stream_trailing_blank_frames(z.data(), z.size(), 15) == 10, "duration 0 = 1 frame");
        CHECK(!asr_stream_trailing_blank_frames(z.data(), z.size(), 3) , "a window cut inside the piece: run 0");
        asr_stream_eou_params off;   // default off
        CHECK(!asr_stream_eou(off, sp.data(), sp.size(), 1000), "eou_blank_frames 0 is off");
        printf("token eou: ok\n");
    }

    printf(g_fail ? "FAILED (%d failure(s))\n" : "OK (%d failure(s))\n", g_fail);
    return g_fail ? 1 : 0;
}
