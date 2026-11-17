// Native Silero VAD (examples/audio/asr-silero.cpp) vs whisper.cpp's own probabilities.
//
// Golden: examples/audio/tests/fixtures/silero/jfk-whispercpp-probs.txt, printed by whisper.cpp
// (ggml-org/whisper.cpp 7e26d3b, libwhisper built on this box) via whisper_vad_probs() and
// whisper_vad_segments_from_probs() on samples/jfk.wav with the same for-tests silero v6.2.0 file.
// The fixture directory is compiled in (ASR_SILERO_FIXTURES); argv[1] overrides it.
#include "asr-silero.h"
#include "asr-vad-post.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

static bool read_wav16(const std::string & p, std::vector<float> & out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (b.size() < 44 || std::memcmp(b.data(), "RIFF", 4) != 0) return false;
    size_t pos = 12;
    while (pos + 8 <= b.size()) {
        uint32_t sz; std::memcpy(&sz, &b[pos + 4], 4);
        if (std::memcmp(&b[pos], "data", 4) == 0) {
            const size_t n = std::min<size_t>(sz, b.size() - pos - 8) / 2;
            out.resize(n);
            for (size_t i = 0; i < n; i++) { int16_t v; std::memcpy(&v, &b[pos + 8 + 2 * i], 2); out[i] = v / 32768.0f; }
            return true;
        }
        pos += 8 + sz + (sz & 1);
    }
    return false;
}

int main(int argc, char ** argv) {
    const std::string dir = argc > 1 ? argv[1] : ASR_SILERO_FIXTURES;
    asr_silero_model m;
    std::string err;
    if (!asr_silero_load(dir + "/silero-v6.2.0-for-tests.ggml", m, &err)) { printf("FAIL load: %s\n", err.c_str()); return 1; }
    printf("loaded silero type=%s version=%s window=%d\n", m.type.c_str(), m.version.c_str(), m.n_window);

    // 1. parity with whisper.cpp on jfk.wav
    std::vector<float> wav;
    CHECK(read_wav16(dir + "/jfk.wav", wav));
    std::vector<float> ref; std::vector<std::pair<int, int>> ref_seg;
    {
        std::ifstream f(dir + "/jfk-whispercpp-probs.txt"); std::string tag;
        while (f >> tag) {
            if (tag == "P") { float v; f >> v; ref.push_back(v); }
            else if (tag == "SEG") { double a, b; f >> a >> b; ref_seg.push_back({(int) a, (int) b}); }
        }
    }
    const auto probs = asr_silero_probs(m, wav.data(), wav.size());
    printf("probs n=%zu (ref %zu)\n", probs.size(), ref.size());
    CHECK(probs.size() == ref.size());
    double maxd = 0;
    for (size_t i = 0; i < std::min(probs.size(), ref.size()); i++) maxd = std::max(maxd, (double) std::fabs(probs[i] - ref[i]));
    printf("max |p - ref| = %.3e\n", maxd);
    CHECK(maxd < 3e-3);

    // segments via the fork's post-processing, vs whisper.cpp's (centiseconds; whisper rounds to cs)
    asr_vad_post_params pp;
    const auto sp = asr_vad_postprocess(probs.data(), probs.size(), pp);
    printf("segments: %zu (ref %zu)\n", sp.size(), ref_seg.size());
    CHECK(sp.size() == ref_seg.size());
    for (size_t i = 0; i < std::min(sp.size(), ref_seg.size()); i++) {
        const double t0 = sp[i].start * 100.0 / 16000.0, t1 = sp[i].end * 100.0 / 16000.0;
        printf("  seg %zu: %.2f %.2f cs (ref %d %d)\n", i, t0, t1, ref_seg[i].first, ref_seg[i].second);
        CHECK(std::fabs(t0 - ref_seg[i].first) <= 1.0 && std::fabs(t1 - ref_seg[i].second) <= 1.0);
    }

    // 2. controls: digital silence and low-level noise are not speech
    {
        std::vector<float> sil(16000 * 3, 0.0f);
        const auto ps = asr_silero_probs(m, sil.data(), sil.size());
        float mx = 0; for (float v : ps) mx = std::max(mx, v);
        printf("silence: n=%zu max p=%.4f\n", ps.size(), mx);
        CHECK(mx < 0.5f);
        CHECK(asr_vad_postprocess(ps.data(), ps.size(), pp).empty());
        std::mt19937 rng(7); std::normal_distribution<float> nd(0.0f, 0.003f);
        std::vector<float> nz(16000 * 3); for (auto & v : nz) v = nd(rng);
        const auto pn = asr_silero_probs(m, nz.data(), nz.size());
        mx = 0; for (float v : pn) mx = std::max(mx, v);
        printf("noise: max p=%.4f\n", mx);
        CHECK(mx < 0.5f);
    }
    // 3. speech surrounded by silence is found where it is (jfk shifted by 2 s of zeros)
    {
        std::vector<float> sh(32000, 0.0f); sh.insert(sh.end(), wav.begin(), wav.end()); sh.resize(sh.size() + 32000, 0.0f);
        const auto ps = asr_silero_probs(m, sh.data(), sh.size());
        const auto s2 = asr_vad_postprocess(ps.data(), ps.size(), pp);
        CHECK(!s2.empty());
        if (!s2.empty()) {
            printf("shifted: first start %.3f s (ref %.3f + 2.0), last end %.3f s\n", s2.front().start / 16000.0, ref_seg[0].first / 100.0, s2.back().end / 16000.0);
            CHECK(std::fabs(s2.front().start / 16000.0 - (ref_seg[0].first / 100.0 + 2.0)) < 0.1);
        }
    }
    // 4. malformed files are refused, not crashed on
    {
        asr_silero_model bad; std::string e;
        CHECK(!asr_silero_load(dir + "/jfk.wav", bad, &e));      // wrong magic
        CHECK(!asr_silero_load(dir + "/nonexistent", bad, &e));
        std::ifstream f(dir + "/silero-v6.2.0-for-tests.ggml", std::ios::binary);
        std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        b.resize(b.size() / 2);
        const std::string tp = "asr-silero-trunc.tmp"; { std::ofstream o(tp, std::ios::binary); o.write(b.data(), b.size()); }
        CHECK(!asr_silero_load(tp, bad, &e));
        printf("truncated: refused (%s)\n", e.c_str());
        std::remove(tp.c_str());
    }
    // 5. streaming (row S69-ENGREV-WHISPERCPP-STREAMVAD-2026-10-27-01): audio fed in pieces of ANY size, state carried,
    //    gives the whole-clip probabilities; the old way (a fresh whole-clip call per piece) does not.
    {
        auto maxdiff = [](const std::vector<float> & a, const std::vector<float> & b) {
            double d = a.size() == b.size() ? 0.0 : 1e9;
            for (size_t i = 0; i < std::min(a.size(), b.size()); i++) d = std::max(d, (double) std::fabs(a[i] - b[i]));
            return d;
        };
        const size_t cuts[] = {1000, 37, 512, 5000, 1, 511, 16000, 3333};   // not multiples of the window
        for (int ctx = 0; ctx < 2; ctx++) {
            asr_silero_stream whole_s; asr_silero_stream_reset(whole_s); whole_s.left_context = ctx != 0;
            std::vector<float> whole;
            asr_silero_stream_feed(m, whole_s, wav.data(), wav.size(), whole);
            asr_silero_stream_flush(m, whole_s, whole);

            asr_silero_stream s; asr_silero_stream_reset(s); s.left_context = ctx != 0;
            std::vector<float> chunked;
            size_t pos = 0, k = 0;
            while (pos < wav.size()) {
                const size_t n = std::min(cuts[k++ % (sizeof(cuts) / sizeof(cuts[0]))], wav.size() - pos);
                asr_silero_stream_feed(m, s, wav.data() + pos, n, chunked);
                pos += n;
            }
            asr_silero_stream_flush(m, s, chunked);
            const double d = maxdiff(whole, chunked);
            printf("stream ctx=%d: n=%zu, max |chunked - whole| = %.3e\n", ctx, chunked.size(), d);
            CHECK(chunked.size() == probs.size());
            CHECK(d <= 1e-6);
            if (ctx == 0) CHECK(maxdiff(whole, probs) == 0.0);   // asr_silero_probs IS feed + flush

            // reset: after other audio, reset + the clip again is the whole-clip result
            std::vector<float> again;
            asr_silero_stream_reset(s);
            asr_silero_stream_feed(m, s, wav.data(), wav.size(), again);
            asr_silero_stream_flush(m, s, again);
            CHECK(maxdiff(whole, again) == 0.0);

            // control: WITHOUT the reset the second pass starts from a used state and must differ
            std::vector<float> dirty;
            asr_silero_stream_feed(m, s, wav.data(), wav.size(), dirty);
            asr_silero_stream_flush(m, s, dirty);
            printf("stream ctx=%d: no-reset second pass differs by %.3e\n", ctx, maxdiff(whole, dirty));
            CHECK(maxdiff(whole, dirty) > 1e-3);
        }
        // control (the defect the row names): one fresh whole-clip call per 1 s step loses the LSTM state at every step
        std::vector<float> per_step;
        for (size_t pos = 0; pos < wav.size(); pos += 16000) {                // 16000 is not a multiple of 512 either
            const auto p = asr_silero_probs(m, wav.data() + pos, std::min<size_t>(16000, wav.size() - pos));
            per_step.insert(per_step.end(), p.begin(), p.end());
        }
        std::vector<float> per_step512;
        for (size_t pos = 0; pos < wav.size(); pos += 512 * 32) {              // window-aligned steps: same count as whole
            const auto p = asr_silero_probs(m, wav.data() + pos, std::min<size_t>(512 * 32, wav.size() - pos));
            per_step512.insert(per_step512.end(), p.begin(), p.end());
        }
        printf("per-step fresh calls: aligned steps differ from whole by %.3e (n %zu vs %zu; unaligned n %zu)\n",
               maxdiff(probs, per_step512), per_step512.size(), probs.size(), per_step.size());
        CHECK(per_step512.size() == probs.size());
        CHECK(maxdiff(probs, per_step512) > 1e-3);
        // left context is a different input, so it must change some probability (else the option is dead)
        asr_silero_stream sc; asr_silero_stream_reset(sc); sc.left_context = true;
        std::vector<float> with_ctx;
        asr_silero_stream_feed(m, sc, wav.data(), wav.size(), with_ctx);
        asr_silero_stream_flush(m, sc, with_ctx);
        printf("left context on vs off: max diff %.3e\n", maxdiff(probs, with_ctx));
        CHECK(maxdiff(probs, with_ctx) > 1e-4);
    }
    printf(g_fail ? "FAILED %d\n" : "OK\n", g_fail);
    return g_fail ? 1 : 0;
}
