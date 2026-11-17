// mtmd audio: whisper / qwen3a mel floor must be 1e-10, as OpenAI whisper
// (log10(clamp(x, min=1e-10))) and whisper.cpp (src/whisper.cpp:3218, 3224).
// mtmd-audio.cpp is compiled straight into this test with a stubbed clip_get_hparams, so no
// clip context / model is needed. The expected values are computed here, independently of the
// code under test (naive double-precision DFT through the same filterbank).
#include "mtmd-audio.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846   // MSVC <cmath> defines it only under _USE_MATH_DEFINES
#endif

static void test_log_cb(enum ggml_log_level, const char * t, void *) { fputs(t, stderr); }
struct clip_logger_state g_logger_state = {test_log_cb, nullptr};   // clip.cpp:30 defines this; clip.cpp is not linked here

static clip_hparams g_hp;
const clip_hparams * clip_get_hparams(const struct clip_ctx *) { return &g_hp; }

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

int main() {
    g_hp.n_mel_bins        = 80;
    g_hp.audio_sample_rate = 16000;
    g_hp.audio_n_fft       = 400;
    g_hp.audio_window_len  = 400;
    g_hp.audio_hop_len     = 160;
    g_hp.audio_chunk_len   = 30;

    // 1. digital silence: every mel bin is exactly log10(1e-10) = -10 before the (x+4)/4 normalisation, i.e. -1.5 after it.
    {
        mtmd_audio_preprocessor_whisper pre(nullptr);
        pre.initialize();
        std::vector<float> sil(16000, 0.0f);
        std::vector<mtmd_audio_mel> out;
        CHECK(pre.preprocess(sil.data(), sil.size(), out));
        CHECK(!out.empty());
        double lo = 1e9, hi = -1e9;
        for (auto & m : out) for (float v : m.data) { lo = std::min(lo, (double) v); hi = std::max(hi, (double) v); }
        printf("whisper silence: min=%.6f max=%.6f (expected -1.5)\n", lo, hi);
        CHECK(std::fabs(lo + 1.5) < 1e-5 && std::fabs(hi + 1.5) < 1e-5);
    }
    {
        mtmd_audio_preprocessor_qwen3a pre(nullptr);
        pre.initialize();
        std::vector<float> sil(16000, 0.0f);
        std::vector<mtmd_audio_mel> out;
        CHECK(pre.preprocess(sil.data(), sil.size(), out));
        CHECK(!out.empty());
        // qwen3a normalises (clamp(max-8) + 4) / 4: silence everywhere => (-10 + 4) / 4 = -1.5
        double lo = 1e9, hi = -1e9;
        for (auto & m : out) for (int64_t r = 0; r < m.n_mel; r++) for (int64_t t = 0; t < 100; t++) { const double v = m.data[(size_t) (r * m.n_len + t)]; lo = std::min(lo, v); hi = std::max(hi, v); } // first 100 frames = the 1 s of input; the tail is zero padding
        printf("qwen3a silence: min=%.6f max=%.6f (expected -1.5)\n", lo, hi);
        CHECK(std::fabs(lo + 1.5) < 1e-5 && std::fabs(hi + 1.5) < 1e-5);
    }

    // 2. a 1 kHz tone: its peak mel bin must be where the Slaney mel scale puts 1 kHz
    //    (mel 15 of 45.24 at 8 kHz => filter centre (j+1) * 45.24 / 81 = 15 => j ~ 25.9), computed
    //    here from the scale definition, and the tone must sit far above the floor.
    {
        mtmd_audio_preprocessor_whisper pre(nullptr);
        pre.initialize();
        std::vector<float> tone(16000 * 2);
        for (size_t i = 0; i < tone.size(); i++) tone[i] = 0.5f * sinf(2.0f * (float) M_PI * 1000.0f * (float) i / 16000.0f);
        std::vector<mtmd_audio_mel> out;
        CHECK(pre.preprocess(tone.data(), tone.size(), out));
        const auto & m = out[0];
        const double mel_hi = 15.0 + 27.0 * std::log(8.0) / std::log(6.4);
        const double exp_j  = 15.0 * 81.0 / mel_hi - 1.0;
        int best = 0; float bv = -1e9f;
        for (int j = 0; j < m.n_mel; j++) { float v = m.data[(size_t) m.n_len * j + 100]; if (v > bv) { bv = v; best = j; } }
        printf("tone: peak bin %d (expected %.2f) value %.3f\n", best, exp_j, bv);
        CHECK(std::fabs(best - exp_j) <= 1.0);
        CHECK(bv > -3.0f);
    }

    // 3. long audio (40 s of tone with one 400 ms pause at 24.0 s): the whisper preprocessor must end its first
    //    30 s window in the pause, not at 30.000 s in the middle of the tone. Window 0 is then digital silence from
    //    the cut onward (padded with the mel of silence), and every tone frame of the input appears exactly once.
    //    The old fixed split put frames 24.4-30 s of tone into window 0, so the check at 29 s failed there.
    {
        mtmd_audio_preprocessor_whisper pre(nullptr);
        pre.initialize();
        std::vector<float> x(16000 * 40);
        for (size_t i = 0; i < x.size(); i++) {
            const double t = (double) i / 16000.0;
            x[i] = (t >= 24.0 && t < 24.4) ? 0.0f : 0.5f * sinf(2.0f * (float) M_PI * 1000.0f * (float) t);
        }
        std::vector<mtmd_audio_mel> out;
        CHECK(pre.preprocess(x.data(), x.size(), out));
        CHECK(out.size() == 2);
        if (out.size() == 2) {
            const int     j      = 26;                                   // the 1 kHz bin (case 2)
            const auto &  w0     = out[0];
            const auto &  w1     = out[1];
            CHECK(w0.n_len == 3000 && w1.n_len == 3000);
            const float   floor_ = w1.data[(size_t) w1.n_len * j + 2999]; // 54+ s: past the input, digital silence
            auto at = [&](const mtmd_audio_mel & m, int f) { return m.data[(size_t) m.n_len * j + f]; };
            printf("long: w0[23 s]=%.3f w0[29 s]=%.3f w1[0]=%.3f floor=%.3f\n", at(w0, 2300), at(w0, 2900), at(w1, 0), floor_);
            CHECK(at(w0, 2300) > floor_ + 1.0f);                  // speech before the pause stays in window 0
            CHECK(std::fabs(at(w0, 2900) - floor_) < 1e-4f);      // 29 s is NOT in window 0 any more
            int tone_frames = 0;
            for (const auto * m : {&w0, &w1}) for (int f = 0; f < m->n_len; f++) tone_frames += at(*m, f) > floor_ + 1.0f;
            printf("long: tone frames %d (input has %d)\n", tone_frames, (int) ((40.0 - 0.4) * 100));
            CHECK(std::abs(tone_frames - (int) ((40.0 - 0.4) * 100)) <= 10);  // nothing lost or duplicated (edge frames +-)
        }
    }
    printf(g_fail ? "FAILED %d\n" : "OK\n", g_fail);
    return g_fail ? 1 : 0;
}
