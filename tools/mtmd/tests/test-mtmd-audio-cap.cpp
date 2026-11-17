// Audio decode length cap: a small upload must not be able to ask the decoder for gigabytes.
//
// The decoder (miniaudio) reports a clip's length in frames AT THE TARGET RATE = the file's declared duration times that
// rate. The declared rate is the file's own claim, so a 40 KB WAV that says "1 Hz" is 14 500 s long once read at 16 kHz and
// the helper allocated 232 M floats (0.93 GB) for it before reading a sample. Ollama bounds the declared duration before it
// allocates (mlxrunner/model/audio/audio.go:149-154); mtmd_helper_audio_decode_mono / _stereo now do the same
// (LLAMA_MTMD_AUDIO_MAX_SECONDS, default 14400 s, 0 = off).
//
// Fail-before = the 14 500 s pathological clip is ACCEPTED (mono returns 1 with 232 M frames, stereo reports 232 M frames).
// Pass-after = both refuse it; the positive controls (a normal clip; the same clip with the cap off) still decode.
// GATE-CLAIM-OK: unit test, no resource claims (worst fail-before allocation ~0.93 GB, only on the unfixed code).
#include "mtmd-helper.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::fprintf(stderr, "FAIL line %d: ", __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); fails++; } else { std::printf("ok   " #cond "\n"); } } while (0)

static void put16(std::vector<unsigned char> & v, uint16_t x) { v.push_back(x & 255); v.push_back(x >> 8); }
static void put32(std::vector<unsigned char> & v, uint32_t x) { put16(v, x & 65535); put16(v, x >> 16); }
static void putstr(std::vector<unsigned char> & v, const char * s) { v.insert(v.end(), s, s + 4); }

// PCM16 WAV whose header claims `rate` Hz, `channels` channels, `frames` frames; samples are a slow sine
static std::vector<unsigned char> make_wav(uint32_t rate, uint16_t channels, uint32_t frames) {
    std::vector<unsigned char> w;
    const uint32_t data_bytes = frames * channels * 2;
    putstr(w, "RIFF"); put32(w, 36 + data_bytes); putstr(w, "WAVE");
    putstr(w, "fmt "); put32(w, 16); put16(w, 1); put16(w, channels); put32(w, rate); put32(w, rate * channels * 2);
    put16(w, channels * 2); put16(w, 16);
    putstr(w, "data"); put32(w, data_bytes);
    w.reserve(w.size() + data_bytes);
    for (uint32_t i = 0; i < frames; ++i) {
        const int16_t s = (int16_t) (8000.0 * std::sin(6.2831853 * (double) i / 50.0));
        for (uint16_t c = 0; c < channels; ++c) { put16(w, (uint16_t) s); }
    }
    return w;
}

// --guard-off: every cap is set to 0 (off), which is what the decode did before the cap existed; the refusal checks below must
// then FAIL (the fail-before run of this test)
static bool g_guard_off = false;

static void set_cap(const char * v) {
    if (g_guard_off) { v = "0"; }
#ifdef _WIN32
    _putenv_s("LLAMA_MTMD_AUDIO_MAX_SECONDS", v ? v : "");
#else
    if (v) { setenv("LLAMA_MTMD_AUDIO_MAX_SECONDS", v, 1); } else { unsetenv("LLAMA_MTMD_AUDIO_MAX_SECONDS"); }
#endif
}

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--guard-off") { g_guard_off = true; }
    }
    const int sr = 16000;

    // positive control: an ordinary 2 s clip at the model rate decodes under the default cap
    {
        set_cap(nullptr);
        const auto w = make_wav(16000, 1, 32000);
        size_t n = 0;
        const int32_t ok = mtmd_helper_audio_decode_mono(w.data(), w.size(), sr, nullptr, 0, &n);
        CHECK(ok == 1, "a 2 s clip was refused (ok=%d)", ok);
        CHECK(n == 32000, "a 2 s clip decoded to %zu frames, expected 32000", n);
    }

    // the pathological clip, default cap (14400 s): 14 500 frames declared at 1 Hz = 14 500 s = 232 M frames at 16 kHz
    {
        set_cap(nullptr);
        const auto w = make_wav(1, 1, 14500);
        size_t n = 0;
        const int32_t ok = mtmd_helper_audio_decode_mono(w.data(), w.size(), sr, nullptr, 0, &n);
        std::printf("[mono 1 Hz x 14500, default cap] ok=%d frames=%zu (%.0f MB of f32)\n", ok, n, (double) n * 4.0 / 1048576.0);
        CHECK(ok == 0, "the 14 500 s clip (%zu bytes) was accepted and decoded to %zu frames", w.size(), n);
        CHECK(n == 0, "frames reported for a refused clip: %zu", n);
    }

    // the same through the stereo (diarize) path, which sizes the caller's buffer from the count query
    {
        set_cap(nullptr);
        const auto w = make_wav(1, 2, 14500);
        size_t n = 0;
        const int32_t ch = mtmd_helper_audio_decode_stereo(w.data(), w.size(), sr, nullptr, 0, &n);
        std::printf("[stereo 1 Hz x 14500, default cap] channels=%d frames=%zu\n", ch, n);
        CHECK(ch == 0 && n == 0, "stereo count query accepted the 14 500 s clip: channels=%d frames=%zu", ch, n);
    }

    // a lower cap by env: 100 Hz x 100 000 frames = 1000 s; refused at 600 s, accepted at 1200 s and with the cap off
    {
        const auto w = make_wav(100, 1, 100000);
        size_t n = 0;
        set_cap("600");
        int32_t ok = mtmd_helper_audio_decode_mono(w.data(), w.size(), sr, nullptr, 0, &n);
        CHECK(ok == 0, "1000 s clip accepted under a 600 s cap (frames=%zu)", n);
        set_cap("1200");
        n = 0;
        ok = mtmd_helper_audio_decode_mono(w.data(), w.size(), sr, nullptr, 0, &n);
        CHECK(ok == 1 && n > 15000000 && n < 17000000, "1000 s clip under a 1200 s cap: ok=%d frames=%zu (want ~16 M)", ok, n);
        set_cap("0");
        n = 0;
        ok = mtmd_helper_audio_decode_mono(w.data(), w.size(), sr, nullptr, 0, &n);
        CHECK(ok == 1 && n > 15000000, "cap off (0) refused the 1000 s clip: ok=%d frames=%zu", ok, n);
        set_cap(nullptr);
    }

    std::printf("\n[RESULT] %s (%d failed check%s)\n", fails == 0 ? "PASS" : "FAIL", fails, fails == 1 ? "" : "s");
    return fails == 0 ? 0 : 1;
}
