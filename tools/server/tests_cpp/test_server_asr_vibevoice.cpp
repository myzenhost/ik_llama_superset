// Standalone test for server-asr-vibevoice.h. Build: g++ -std=c++17 -I.. test_server_asr_vibevoice.cpp -o t && ./t
// Expected strings are written out by hand from the VibeVoice processor spec (vibevoice_asr_processor.py:388-411), not produced by the code under test.
#include "server-asr-vibevoice.h"
#include <cstdio>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)
#define EQ(a, b) do { const std::string _a = (a), _b = (b); if (_a != _b) { std::printf("FAIL line %d:\n--- got\n%s\n--- want\n%s\n", __LINE__, _a.c_str(), _b.c_str()); ++fails; } } while (0)

static size_t count(const std::string & s, const std::string & sub) {
    size_t n = 0;
    for (size_t p = s.find(sub); p != std::string::npos; p = s.find(sub, p + sub.size())) { n++; }
    return n;
}

int main() {
    using namespace asr_vibevoice;
    static const std::string SYS = "<|im_start|>system\nYou are a helpful assistant that transcribes audio input into text output in JSON format.<|im_end|>\n<|im_start|>user\n";

    // 7.3 s of 24 kHz mono = 175200 samples = 54.75 pad frames -> ceil = 55
    CHECK(n_pads(175200) == 55);
    CHECK(n_pads(3200) == 1);
    CHECK(n_pads(3201) == 2);
    CHECK(n_pads(0) == 0);
    const std::string blk = speech_block(175200);
    CHECK(count(blk, "<|speech_pad|>") == 55);
    CHECK(blk.rfind("<|speech_start|>", 0) == 0);
    CHECK(blk.size() > 14 && blk.substr(blk.size() - 14) == "<|speech_end|>");

    // no context: lower-case "please", no generation header after the user turn
    const std::string p = render(175200, SAMPLE_RATE, "", "<AUDIO>");
    EQ(p, SYS + "<AUDIO>\nThis is a 7.30 seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content<|im_end|>\n");
    CHECK(p.find("<|im_start|>assistant") == std::string::npos);
    CHECK(p.substr(p.size() - 11) == "<|im_end|>\n");

    // hotword context: "with extra info: X\n\nPlease transcribe ..."
    EQ(render(48000, SAMPLE_RATE, "Acme, Zyrtec", "<AUDIO>"),
       SYS + "<AUDIO>\nThis is a 2.00 seconds audio, with extra info: Acme, Zyrtec\n\nPlease transcribe it with these keys: Start time, End time, Speaker ID, Content<|im_end|>\n");

    // RMS: a -40 dBFS tone comes out at -25 +-0.1 dBFS
    std::vector<float> tone(24000);
    for (size_t i = 0; i < tone.size(); i++) {
        tone[i] = (float) (0.01 * 1.41421356 * std::sin(2.0 * 3.14159265358979 * 440.0 * (double) i / 24000.0));
    }
    CHECK(std::fabs(rms_dbfs(tone) - (-40.0)) < 0.1);
    rms_normalize(tone);
    CHECK(std::fabs(rms_dbfs(tone) - (-25.0)) < 0.1);

    // a loud full-scale square wave is scaled down to -25 dBFS RMS, then never exceeds 1.0 (and stays a square: no clipping flat-top change)
    std::vector<float> loud(24000);
    for (size_t i = 0; i < loud.size(); i++) { loud[i] = (i / 50) % 2 ? 1.0f : -1.0f; }
    rms_normalize(loud);
    double peak = 0.0;
    for (float v : loud) { peak = std::fmax(peak, std::fabs((double) v)); }
    CHECK(peak <= 1.0);
    CHECK(std::fabs(rms_dbfs(loud) - (-25.0)) < 0.1);

    // a crest-factor-heavy signal: one -0.2 dBFS spike on a quiet floor. Scaled to -25 dBFS RMS the spike would exceed 1.0; the guard divides by the peak
    std::vector<float> spiky(24000, 0.0005f);
    spiky[100] = 0.97f;
    rms_normalize(spiky);
    peak = 0.0;
    for (float v : spiky) { peak = std::fmax(peak, std::fabs((double) v)); }
    CHECK(peak <= 1.0 + 1e-6);
    CHECK(peak > 0.99);                // normalised to the peak, not hard-clipped lower

    // silence stays silence (the 1e-6 epsilon keeps the scalar finite)
    std::vector<float> zero(100, 0.0f);
    rms_normalize(zero);
    for (float v : zero) { CHECK(v == 0.0f); }
    std::vector<float> empty;
    CHECK(rms_normalize(empty) == 1.0);

    // WAV container: 44-byte header, float format tag 3, mono, data size
    const std::vector<float> s(10, 0.5f);
    const auto w = wav_f32_mono(s, 24000);
    CHECK(w.size() == 44 + 40);
    CHECK(std::string((const char *) &w[0], 4) == "RIFF" && std::string((const char *) &w[8], 8) == "WAVEfmt ");
    CHECK(w[20] == 3 && w[22] == 1);
    CHECK(w[24] == 0xC0 && w[25] == 0x5D);   // 24000 = 0x5DC0 little endian

    if (fails == 0) {
        std::printf("test_server_asr_vibevoice: PASS\n");
    }
    return fails ? 1 : 0;
}
