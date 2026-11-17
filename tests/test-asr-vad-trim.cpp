// test-asr-vad-trim.cpp -- examples/audio/asr-vad-trim.h: silence trimming from VAD spans and the time map back
// to the original clip. Std-only, no model: the spans are given.
//
// Invariants (from the row S69-ENGREV-WHISPERCPP-VADTRIM-2026-10-27-01; whisper.cpp tests/test-vad-full.cpp and
// faster-whisper patch a52b026ac as the spec of what must hold, MIT, SOURCE-READ, no code copied):
//   - every input span lies in exactly one kept piece; pieces are ordered and non-overlapping in both signals;
//   - the map is monotone and never returns a time past the original length;
//   - the end of a piece maps to the end of THAT piece in the original (whisper.cpp #3711), a start on a join
//     maps to the start of the NEXT piece;
//   - a word is placed in the piece that holds its midpoint, both edges inside that piece;
//   - an empty map changes nothing (a request that was not trimmed equals a run without the feature).
#include "asr-vad-trim.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;

#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);             \
            g_fail++;                                                               \
        }                                                                           \
    } while (0)

static bool close_to(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

static asr_vad_span span16(double s0, double s1) {
    asr_vad_span s;
    s.start = (int64_t) std::llround(s0 * 16000.0);
    s.end   = (int64_t) std::llround(s1 * 16000.0);
    return s;
}

static uint32_t rd_u32(const std::vector<uint8_t> & b, size_t at) {
    return (uint32_t) b[at] | ((uint32_t) b[at + 1] << 8) | ((uint32_t) b[at + 2] << 16) | ((uint32_t) b[at + 3] << 24);
}

static uint16_t rd_u16(const std::vector<uint8_t> & b, size_t at) {
    return (uint16_t) ((uint16_t) b[at] | ((uint16_t) b[at + 1] << 8));
}

// the checks every map must pass, whatever the spans
static void check_map_invariants(const asr_vad_trim_map & m, const std::vector<float> & out) {
    CHECK(m.n_out == (int64_t) out.size());
    CHECK(m.n_out <= m.n_in);
    int64_t prev_out = 0, prev_in = 0;
    for (size_t i = 0; i < m.pieces.size(); ++i) {
        const auto & pc = m.pieces[i];
        CHECK(pc.out_end > pc.out_start);
        CHECK(pc.out_start >= prev_out);
        CHECK(pc.in_start >= prev_in);
        CHECK(pc.in_end() <= m.n_in);
        CHECK(pc.out_end <= m.n_out);
        prev_out = pc.out_end;
        prev_in  = pc.in_end();
    }
    if (m.empty() || m.sample_rate <= 0) {
        return;
    }
    const double sr     = (double) m.sample_rate;
    const double len_in = (double) m.n_in / sr;
    double last_s = -1.0, last_e = -1.0;
    for (double t = 0.0; t <= (double) m.n_out / sr + 0.5; t += 0.01) {
        const double s = m.start_to_original(t);
        const double e = m.end_to_original(t);
        CHECK(s >= last_s);                 // monotone
        CHECK(e >= last_e);
        CHECK(s <= len_in + 1e-9);          // never past the original clip
        CHECK(e <= len_in + 1e-9);
        CHECK(e <= s + 1e-9);               // at one t, "ends here" is never later than "starts here"
        last_s = s;
        last_e = e;
        double a = t, b = t + 0.25;
        m.segment_to_original(a, b);
        CHECK(b >= a);
        double w0 = t, w1 = t + 0.05;
        m.word_to_original(w0, w1);
        CHECK(w1 >= w0);
        CHECK(w1 <= len_in + 1e-9);
    }
}

int main() {
    const int32_t SR = 16000;
    const size_t  N  = 20 * (size_t) SR;        // 20 s; x[i] = i + 1, exact in float (< 2^24), never 0
    std::vector<float> x(N);
    for (size_t i = 0; i < N; ++i) {
        x[i] = (float) (i + 1);
    }

    // ---- 1. three spans, 100 ms pad; the 50 ms gap between the last two is kept, not spliced ----
    {
        const std::vector<asr_vad_span> spans = { span16(1.0, 3.0), span16(10.0, 12.5), span16(12.55, 14.0) };
        asr_vad_trim_params p;
        p.pad_ms = 100;
        asr_vad_trim_map m;
        const std::vector<float> y = asr_vad_trim(x.data(), N, SR, spans, p, m);
        check_map_invariants(m, y);
        CHECK(m.pieces.size() == 2);
        CHECK(m.n_in == (int64_t) N);
        CHECK(m.n_out == 2 * SR + SR / 10 + 4 * SR);        // 2.0 s + 0.1 s pad + 4.0 s (10.0 .. 14.0)
        if (m.pieces.size() == 2) {
            CHECK(m.pieces[0].out_start == 0 && m.pieces[0].in_start == 1 * SR && m.pieces[0].out_end == 2 * SR);
            CHECK(m.pieces[1].out_start == 2 * SR + SR / 10 && m.pieces[1].in_start == 10 * SR);
            // the samples are the original ones, the pad is digital silence
            bool copy_ok = true, pad_ok = true;
            for (const auto & pc : m.pieces) {
                for (int64_t k = pc.out_start; k < pc.out_end; ++k) {
                    copy_ok = copy_ok && y[(size_t) k] == x[(size_t) (pc.in_start + (k - pc.out_start))];
                }
            }
            for (int64_t k = m.pieces[0].out_end; k < m.pieces[1].out_start; ++k) {
                pad_ok = pad_ok && y[(size_t) k] == 0.0f;
            }
            CHECK(copy_ok);
            CHECK(pad_ok);
        }
        // every input span lies in exactly one piece
        for (const auto & s : spans) {
            int holders = 0;
            for (const auto & pc : m.pieces) {
                if (s.start >= pc.in_start && s.end <= pc.in_end()) { holders++; }
            }
            CHECK(holders == 1);
        }
        // round trip inside a piece: exact to a sample
        CHECK(close_to(m.start_to_original(0.5), 1.5, 1e-9));
        CHECK(close_to(m.end_to_original(0.5), 1.5, 1e-9));
        CHECK(close_to(m.start_to_original(2.1 + 1.25), 11.25, 1e-9));
        // leading silence: the first kept sample is at 1.0 s of the upload
        CHECK(close_to(m.start_to_original(0.0), 1.0, 1e-9));
        // #3711: what ENDS at the end of piece 0 ends at 3.0 s, not after the removed silence
        CHECK(close_to(m.end_to_original(2.0), 3.0, 1e-9));
        // ... and in the pad after it
        CHECK(close_to(m.end_to_original(2.05), 3.0, 1e-9));
        // what STARTS at the join or in the pad starts with the next piece
        CHECK(close_to(m.start_to_original(2.0), 10.0, 1e-9));
        CHECK(close_to(m.start_to_original(2.05), 10.0, 1e-9));
        CHECK(close_to(m.start_to_original(2.1), 10.0, 1e-9));
        // past the end of the trimmed signal: the end of the last piece
        CHECK(close_to(m.start_to_original(99.0), 14.0, 1e-9));
        CHECK(close_to(m.end_to_original(99.0), 14.0, 1e-9));
        // a segment that spans the join keeps the silence between its two ends
        double a = 1.5, b = 2.6;
        m.segment_to_original(a, b);
        CHECK(close_to(a, 2.5, 1e-9) && close_to(b, 10.5, 1e-9));
        // two adjacent segments that meet in the pad: the first ends before the silence, the next starts after it
        double a1 = 1.0, b1 = 2.05, a2 = 2.05, b2 = 3.0;
        m.segment_to_original(a1, b1);
        m.segment_to_original(a2, b2);
        CHECK(close_to(b1, 3.0, 1e-9) && close_to(a2, 10.0, 1e-9) && close_to(b2, 10.9, 1e-9));
        // words: midpoint rule, edges inside the piece
        double w0 = 1.9, w1 = 2.04;      // midpoint 1.97, in piece 0: the end is clamped to 3.0
        m.word_to_original(w0, w1);
        CHECK(close_to(w0, 2.9, 1e-9) && close_to(w1, 3.0, 1e-9));
        w0 = 2.08; w1 = 2.3;             // midpoint 2.19, in piece 1: the start is clamped to 10.0
        m.word_to_original(w0, w1);
        CHECK(close_to(w0, 10.0, 1e-9) && close_to(w1, 10.2, 1e-9));
        w0 = 2.0; w1 = 2.06;             // midpoint 2.03 in the pad, nearer piece 0
        m.word_to_original(w0, w1);
        CHECK(close_to(w0, 3.0, 1e-9) && close_to(w1, 3.0, 1e-9));
        w0 = 2.06; w1 = 2.12;            // midpoint 2.09 in the pad, nearer piece 1
        m.word_to_original(w0, w1);
        CHECK(close_to(w0, 10.0, 1e-9) && close_to(w1, 10.02, 1e-9));
    }

    // ---- 2. pad 0 (contiguous, faster-whisper style): the join is one instant with two meanings ----
    {
        const std::vector<asr_vad_span> spans = { span16(1.0, 3.0), span16(10.0, 12.0) };
        asr_vad_trim_params p;
        p.pad_ms = 0;
        asr_vad_trim_map m;
        const std::vector<float> y = asr_vad_trim(x.data(), N, SR, spans, p, m);
        check_map_invariants(m, y);
        CHECK(m.pieces.size() == 2);
        CHECK(m.n_out == 4 * SR);
        CHECK(close_to(m.end_to_original(2.0), 3.0, 1e-9));
        CHECK(close_to(m.start_to_original(2.0), 10.0, 1e-9));
        CHECK(y.size() == 4 * (size_t) SR && y[2 * (size_t) SR - 1] == x[3 * (size_t) SR - 1] && y[2 * (size_t) SR] == x[10 * (size_t) SR]);
    }

    // ---- 3. the row's shape: speech A + 600 s of silence + speech B; B's times come back within a sample ----
    {
        const int32_t sr = 16000;
        const size_t  n  = 610 * (size_t) sr;
        std::vector<float> z(n, 0.25f);
        const std::vector<asr_vad_span> spans = { span16(0.5, 4.5), span16(604.5, 609.0) };
        asr_vad_trim_params p;
        asr_vad_trim_map m;
        const std::vector<float> y = asr_vad_trim(z.data(), n, sr, spans, p, m);
        check_map_invariants(m, y);
        CHECK(close_to((double) y.size() / sr, 4.0 + 0.1 + 4.5, 1e-9));     // 8.6 s go to the model, not 610
        double a = 4.1, b = 8.6;                                         // all of B on the trimmed signal
        m.segment_to_original(a, b);
        CHECK(close_to(a, 604.5, 1e-9) && close_to(b, 609.0, 1e-9));
        double c = 5.1, d = 6.1;                                         // 1 s into B
        m.segment_to_original(c, d);
        CHECK(close_to(c, 605.5, 1e-9) && close_to(d, 606.5, 1e-9));
    }

    // ---- 4. another clip rate: 16 kHz span indices are converted ----
    {
        const int32_t sr = 24000;
        std::vector<float> z(10 * (size_t) sr);
        for (size_t i = 0; i < z.size(); ++i) { z[i] = (float) (i + 1); }
        asr_vad_trim_params p;
        asr_vad_trim_map m;
        const std::vector<float> y = asr_vad_trim(z.data(), z.size(), sr, { span16(1.0, 2.0), span16(6.0, 7.0) }, p, m);
        check_map_invariants(m, y);
        CHECK(m.pieces.size() == 2);
        if (m.pieces.size() == 2) {
            CHECK(m.pieces[0].in_start == 24000 && m.pieces[0].in_end() == 48000);
            CHECK(m.pieces[1].in_start == 144000 && m.pieces[1].in_end() == 168000);
            CHECK(m.pieces[1].out_start == 24000 + 2400);
        }
        CHECK(close_to(m.start_to_original(1.1), 6.0, 1e-9));
    }

    // ---- 5. unordered, overlapping and out-of-range spans: clamped, ordered, nothing kept twice ----
    {
        std::vector<asr_vad_span> spans = { span16(8.0, 9.0), span16(2.0, 4.0), span16(3.0, 5.0), span16(19.5, 25.0) };
        asr_vad_span bad;
        bad.start = 100;
        bad.end   = 50;                  // end before start: dropped
        spans.push_back(bad);
        asr_vad_trim_params p;
        asr_vad_trim_map m;
        const std::vector<float> y = asr_vad_trim(x.data(), N, SR, spans, p, m);
        check_map_invariants(m, y);
        CHECK(m.pieces.size() == 3);
        if (m.pieces.size() == 3) {
            CHECK(m.pieces[0].in_start == 2 * SR && m.pieces[0].in_end() == 5 * SR);      // 2..4 and 3..5 merged
            CHECK(m.pieces[1].in_start == 8 * SR && m.pieces[1].in_end() == 9 * SR);
            CHECK(m.pieces[2].in_start == 19 * SR + SR / 2 && m.pieces[2].in_end() == 20 * SR);  // clamped to the clip
        }
    }

    // ---- 6. one span over the whole clip: the signal is unchanged and the map is the identity ----
    {
        asr_vad_trim_params p;
        asr_vad_trim_map m;
        const std::vector<float> y = asr_vad_trim(x.data(), N, SR, { span16(0.0, 20.0) }, p, m);
        check_map_invariants(m, y);
        CHECK(y == x);
        CHECK(m.pieces.size() == 1 && m.n_out == m.n_in);
        CHECK(close_to(m.start_to_original(7.25), 7.25, 1e-9) && close_to(m.end_to_original(7.25), 7.25, 1e-9));
    }

    // ---- 7. no span / no input: empty signal, empty map, and an empty map changes no time ----
    {
        asr_vad_trim_params p;
        asr_vad_trim_map m;
        std::vector<float> y = asr_vad_trim(x.data(), N, SR, {}, p, m);
        CHECK(y.empty() && m.empty() && m.n_out == 0 && m.n_in == (int64_t) N);
        y = asr_vad_trim(nullptr, 0, SR, { span16(0.0, 1.0) }, p, m);
        CHECK(y.empty() && m.empty());
        const asr_vad_trim_map id;
        double a = 3.25, b = 4.5;
        id.segment_to_original(a, b);
        CHECK(a == 3.25 && b == 4.5);
        id.word_to_original(a, b);
        CHECK(a == 3.25 && b == 4.5);
        CHECK(id.start_to_original(1.5) == 1.5 && id.end_to_original(1.5) == 1.5);
    }

    // ---- 8. a negative pad is treated as 0 ----
    {
        asr_vad_trim_params p;
        p.pad_ms = -50;
        asr_vad_trim_map m;
        const std::vector<float> y = asr_vad_trim(x.data(), N, SR, { span16(1.0, 2.0), span16(5.0, 6.0) }, p, m);
        check_map_invariants(m, y);
        CHECK(m.n_out == 2 * SR);
    }

    // ---- 9. the WAV the trimmed signal travels in: float32 mono, sizes consistent, samples bit-exact ----
    {
        const std::vector<float> s = { 0.0f, 0.5f, -0.25f, 1.0f, -1.0f, 1.0e-7f };
        const std::vector<uint8_t> w = asr_wav_f32_encode(s.data(), s.size(), 16000);
        CHECK(w.size() == 58 + s.size() * 4);
        CHECK(std::memcmp(w.data(), "RIFF", 4) == 0 && std::memcmp(w.data() + 8, "WAVE", 4) == 0);
        CHECK(rd_u32(w, 4) == (uint32_t) (w.size() - 8));
        CHECK(std::memcmp(w.data() + 12, "fmt ", 4) == 0 && rd_u32(w, 16) == 18);
        CHECK(rd_u16(w, 20) == 3);                       // IEEE float
        CHECK(rd_u16(w, 22) == 1);                       // mono
        CHECK(rd_u32(w, 24) == 16000);
        CHECK(rd_u32(w, 28) == 16000 * 4);
        CHECK(rd_u16(w, 32) == 4 && rd_u16(w, 34) == 32);
        CHECK(std::memcmp(w.data() + 38, "fact", 4) == 0 && rd_u32(w, 46) == (uint32_t) s.size());
        CHECK(std::memcmp(w.data() + 50, "data", 4) == 0 && rd_u32(w, 54) == (uint32_t) (s.size() * 4));
        std::vector<float> back(s.size());
        std::memcpy(back.data(), w.data() + 58, s.size() * 4);
        CHECK(back == s);
        const std::vector<uint8_t> e = asr_wav_f32_encode(nullptr, 0, 16000);
        CHECK(e.size() == 58 && rd_u32(e, 54) == 0);
    }

    std::printf("%s (%d failure(s))\n", g_fail == 0 ? "OK" : "FAILED", g_fail);
    return g_fail == 0 ? 0 : 1;
}
