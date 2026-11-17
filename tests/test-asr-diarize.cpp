// Stereo-energy diarization (examples/audio/asr-diarize.h) on synthetic channel-separated audio.
#include "asr-diarize.h"

#include <cmath>
#include <cstdio>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846   // MSVC <cmath> defines it only under _USE_MATH_DEFINES
#endif

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

int main() {
    const int sr = 16000, n = sr * 6;
    std::vector<float> L(n, 0.0f), R(n, 0.0f);
    // speaker A (left channel only) 0-2 s, speaker B (right only) 2-4 s, both equally 4-6 s
    for (int i = 0; i < n; i++) {
        const float s = 0.3f * std::sin(2.0f * (float) M_PI * 440.0f * i / sr);
        if (i < 2 * sr)      { L[i] = s; }
        else if (i < 4 * sr) { R[i] = s; }
        else                 { L[i] = s; R[i] = s; }
    }
    const std::string a = asr_stereo_speaker(L.data(), R.data(), 0, 2 * sr);
    const std::string b = asr_stereo_speaker(L.data(), R.data(), 2 * sr, 4 * sr);
    const std::string c = asr_stereo_speaker(L.data(), R.data(), 4 * sr, 6 * sr);
    printf("A=%s B=%s both=%s\n", a.c_str(), b.c_str(), c.c_str());
    CHECK(a == "0");
    CHECK(b == "1");
    CHECK(c == "?");
    // the 10% margin: right 9% louder is still "?", 11% louder wins
    std::vector<float> l2(1000, 0.1f), r2(1000, 0.109f), r3(1000, 0.111f);
    CHECK(asr_stereo_speaker(l2.data(), r2.data(), 0, 1000) == "?");
    CHECK(asr_stereo_speaker(l2.data(), r3.data(), 0, 1000) == "1");
    CHECK(asr_stereo_speaker(L.data(), R.data(), 5, 5) == "?");    // empty range
    printf(g_fail ? "FAILED %d\n" : "OK\n", g_fail);
    return g_fail ? 1 : 0;
}
