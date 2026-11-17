// mtmd audio: parakeet preprocess on clips shorter than 2 hops (320 samples) must not emit NaN.
// valid_frames = n / hop; std = sqrt(sq / (valid_frames - 1)) is 0/0 or x/0 for valid_frames 0 or 1,
// and the result is applied over all frames. Spec: whisper.cpp tests/test-whisper-zero-samples.cpp:24-26
// (no crash, zero segments) and src/whisper.cpp:7021-7029 (short-input guard); SOURCE-READ, no code copied (MIT).
// Same stub setup as test-mtmd-audio-mel.cpp: mtmd-audio.cpp compiled in, clip_get_hparams stubbed.
#include "mtmd-audio.h"

#include <cmath>
#include <cstdio>
#include <vector>

static void test_log_cb(enum ggml_log_level, const char * t, void *) { fputs(t, stderr); }
struct clip_logger_state g_logger_state = {test_log_cb, nullptr};

static clip_hparams g_hp;
const clip_hparams * clip_get_hparams(const struct clip_ctx *) { return &g_hp; }

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

int main() {
    g_hp.n_mel_bins        = 8;
    g_hp.audio_sample_rate = 16000;
    g_hp.audio_n_fft       = 512;
    g_hp.audio_window_len  = 400;
    g_hp.audio_hop_len     = 160;
    g_hp.audio_chunk_len   = 30;
    g_hp.mel_filters.assign((size_t) g_hp.n_mel_bins * (g_hp.audio_n_fft / 2 + 1), 0.01f);
    g_hp.window.assign((size_t) g_hp.audio_window_len, 1.0f);

    mtmd_audio_preprocessor_parakeet pre(nullptr);
    pre.initialize();

    const size_t sizes[] = {1, 159, 160, 319, 320, 480, 16000};
    for (size_t n : sizes) {
        std::vector<float> x(n);
        for (size_t i = 0; i < n; i++) x[i] = 0.3f * std::sin(0.05f * (float) i);
        std::vector<mtmd_audio_mel> out;
        const bool ok = pre.preprocess(x.data(), x.size(), out);
        size_t nan = 0;
        for (auto & m : out) for (float v : m.data) if (!std::isfinite(v)) nan++;
        printf("n=%zu ok=%d nan=%zu\n", n, (int) ok, nan);
        CHECK(nan == 0);                 // never NaN/inf, whether refused or normalised
        if (n < 320) CHECK(!ok);         // under 2 hops: clean refusal
        else         CHECK(ok);
    }
    printf(g_fail ? "FAILED %d\n" : "OK\n", g_fail);
    return g_fail ? 1 : 0;
}
