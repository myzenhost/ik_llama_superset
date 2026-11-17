// test-asr-vad-gate.cpp -- the whole-clip Silero gate behind llama-server --asr-vad (examples/audio/asr-vad-gate.h).
//
// Uses the real for-tests Silero v6.2.0 file (the same fixture test-asr-silero.cpp reads):
//   1. 3 s digital silence           -> no speech span (the server returns "" without running the model)
//   2. 3 s Gaussian noise at -50 dBFS -> no speech span
//   3. jfk.wav at 16 kHz             -> speech spans
//   4. jfk.wav resampled to 24 kHz and handed over at 24 kHz -> speech spans whose first start agrees with (3) to 0.1 s,
//      so a model whose audio rate is not 16 kHz still gets a working gate
//   5. resampler: identity at equal rates, length scales with the rate ratio, empty input -> no speech
#include "asr-vad-gate.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#ifndef ASR_SILERO_FIXTURES
#define ASR_SILERO_FIXTURES "."
#endif

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

static bool read_wav16(const std::string & p, std::vector<float> & out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { return false; }
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t i = 12;
    while (i + 8 <= b.size()) {
        const uint32_t sz = (uint8_t) b[i + 4] | ((uint8_t) b[i + 5] << 8) | ((uint8_t) b[i + 6] << 16) | ((uint32_t) (uint8_t) b[i + 7] << 24);
        if (std::string(&b[i], 4) == "data") {
            const size_t n = std::min<size_t>(sz, b.size() - i - 8) / 2;
            out.resize(n);
            for (size_t k = 0; k < n; k++) {
                const int16_t s = (int16_t) ((uint8_t) b[i + 8 + 2 * k] | ((uint8_t) b[i + 9 + 2 * k] << 8));
                out[k] = s / 32768.0f;
            }
            return true;
        }
        i += 8 + sz + (sz & 1);
    }
    return false;
}

int main(int argc, char ** argv) {
    const std::string dir = argc > 1 ? argv[1] : ASR_SILERO_FIXTURES;
    asr_silero_model m;
    std::string err;
    if (!asr_silero_load(dir + "/silero-v6.2.0-for-tests.ggml", m, &err)) { printf("FAIL load: %s\n", err.c_str()); return 1; }
    const asr_vad_post_params pp;   // defaults: threshold 0.5 (what --asr-vad-threshold defaults to)

    {
        std::vector<float> sil(16000 * 3, 0.0f);
        const auto r = asr_vad_gate(m, sil.data(), sil.size(), 16000, pp);
        printf("silence: n_probs=%zu max_p=%.4f spans=%zu\n", r.n_probs, r.max_prob, r.spans.size());
        CHECK(r.n_probs > 0);
        CHECK(!r.has_speech());
    }
    {
        // -50 dBFS RMS: 10^(-50/20) = 0.00316 of full scale
        std::mt19937 rng(11); std::normal_distribution<float> nd(0.0f, 0.0031623f);
        std::vector<float> nz(16000 * 3); for (auto & v : nz) { v = nd(rng); }
        const auto r = asr_vad_gate(m, nz.data(), nz.size(), 16000, pp);
        printf("noise -50 dBFS: max_p=%.4f spans=%zu\n", r.max_prob, r.spans.size());
        CHECK(!r.has_speech());
    }
    std::vector<float> wav;
    CHECK(read_wav16(dir + "/jfk.wav", wav));
    double start16 = -1.0;
    {
        const auto r = asr_vad_gate(m, wav.data(), wav.size(), 16000, pp);
        printf("jfk 16 kHz: max_p=%.4f spans=%zu\n", r.max_prob, r.spans.size());
        CHECK(r.has_speech());
        if (r.has_speech()) {
            start16 = r.spans.front().start / 16000.0;
            CHECK(r.spans.back().end <= (int64_t) r.n_samples_16k);
        }
    }
    {
        const std::vector<float> w24 = asr_resample_linear(wav.data(), wav.size(), 16000, 24000);
        CHECK(std::llabs((long long) w24.size() - (long long) (wav.size() * 3 / 2)) <= 1);
        const auto r = asr_vad_gate(m, w24.data(), w24.size(), 24000, pp);
        printf("jfk 24 kHz in: max_p=%.4f spans=%zu n16=%zu (orig %zu)\n", r.max_prob, r.spans.size(), r.n_samples_16k, wav.size());
        CHECK(r.has_speech());
        CHECK(std::llabs((long long) r.n_samples_16k - (long long) wav.size()) <= 1);
        if (r.has_speech() && start16 >= 0) {
            const double s = r.spans.front().start / 16000.0;
            printf("  first start %.3f s vs %.3f s at 16 kHz\n", s, start16);
            CHECK(std::fabs(s - start16) < 0.1);
        }
    }
    {
        const std::vector<float> same = asr_resample_linear(wav.data(), 100, 16000, 16000);
        CHECK(same.size() == 100 && same[50] == wav[50]);
        CHECK(asr_resample_linear(nullptr, 0, 16000, 16000).empty());
        const auto r = asr_vad_gate(m, nullptr, 0, 16000, pp);
        CHECK(!r.has_speech() && r.n_probs == 0);
    }
    printf(g_fail ? "FAILED %d\n" : "OK\n", g_fail);
    return g_fail ? 1 : 0;
}
