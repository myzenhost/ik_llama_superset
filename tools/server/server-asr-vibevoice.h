#pragma once
// VibeVoice-ASR front end for /v1/audio/transcriptions (--asr-native-prompt vibevoice). std-only, header-only, so a CPU
// test can include it (tests_cpp/test_server_asr_vibevoice.cpp).
//
// Spec: SOURCE-READ microsoft/VibeVoice (MIT), vibevoice/processor/vibevoice_asr_processor.py:388-411 (offline prompt),
// :185 / :215 (target_dB_FS -25), vibevoice/processor/audio_utils.py:157-203 (RMS normalisation, eps 1e-6, clipping
// guard), asr-hf/config.json:38-40 (token ids 151646 / 151647 / 151648). Written independently, no code copied.
//
//   system  "You are a helpful assistant that transcribes audio input into text output in JSON format."
//   user    <|speech_start|> <|speech_pad|> x ceil(samples / 3200) <|speech_end|> "\n"
//           "This is a {dur:.2f} seconds audio, " ["with extra info: {ctx}\n\nPlease " | "please "]
//           "transcribe it with these keys: Start time, End time, Speaker ID, Content"
//   The prompt ends right after the user <|im_end|>\n. NO assistant generation header follows: the model continues from
//   there. A chat template that appends one changes the output with no error, so this prompt bypasses the template.
//
// 3200 = 24000 Hz / 7.5 Hz: one speech_pad token per 3200 samples of the 24 kHz mono signal.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace asr_vibevoice {

static constexpr int32_t TOKEN_SPEECH_START = 151646;
static constexpr int32_t TOKEN_SPEECH_END   = 151647;
static constexpr int32_t TOKEN_SPEECH_PAD   = 151648;
static constexpr int32_t SAMPLE_RATE        = 24000;
static constexpr int32_t SAMPLES_PER_PAD    = 3200;
static constexpr double  TARGET_DBFS        = -25.0;

static inline size_t n_pads(size_t n_samples) {
    return (n_samples + SAMPLES_PER_PAD - 1) / SAMPLES_PER_PAD;
}

// The audio block as literal special tokens: used when the caller tokenizes the whole prompt itself (and by the unit test).
static inline std::string speech_block(size_t n_samples) {
    std::string s = "<|speech_start|>";
    for (size_t i = 0, n = n_pads(n_samples); i < n; i++) {
        s += "<|speech_pad|>";
    }
    return s + "<|speech_end|>";
}

// audio_slot: what stands for the audio in the user turn. speech_block(n) for a self-tokenized prompt, or the mtmd media
// marker when mtmd writes the audio wrapper tokens at the marker. n_samples / sample_rate give the duration string.
static inline std::string render(size_t n_samples, int32_t sample_rate, const std::string & ctx, const std::string & audio_slot) {
    char dur[64];
    std::snprintf(dur, sizeof(dur), "%.2f", sample_rate > 0 ? (double) n_samples / (double) sample_rate : 0.0);
    std::string u = audio_slot + "\nThis is a " + dur + " seconds audio, ";
    if (!ctx.empty()) {
        u += "with extra info: " + ctx + "\n\nPlease ";
    } else {
        u += "please ";
    }
    u += "transcribe it with these keys: Start time, End time, Speaker ID, Content";
    return "<|im_start|>system\nYou are a helpful assistant that transcribes audio input into text output in JSON format."
           "<|im_end|>\n<|im_start|>user\n" + u + "<|im_end|>\n";
}

// RMS normalisation of the whole file to target_dbfs (audio_utils.py:157-203): scalar = 10^(target/20) / (rms + 1e-6),
// then a clipping guard: when the scaled peak exceeds 1.0 the signal is divided by that peak (no hard clip, so no
// distortion). Returns the scalar that was applied (before the guard).
static inline double rms_normalize(std::vector<float> & x, double target_dbfs = TARGET_DBFS) {
    if (x.empty()) {
        return 1.0;
    }
    double ss = 0.0;
    for (float v : x) {
        ss += (double) v * (double) v;
    }
    const double rms    = std::sqrt(ss / (double) x.size());
    const double scalar = std::pow(10.0, target_dbfs / 20.0) / (rms + 1e-6);
    double peak = 0.0;
    for (float & v : x) {
        v = (float) ((double) v * scalar);
        peak = std::fmax(peak, std::fabs((double) v));
    }
    if (peak > 1.0) {
        const float g = (float) (1.0 / peak);
        for (float & v : x) {
            v *= g;
        }
    }
    return scalar;
}

static inline double rms_dbfs(const std::vector<float> & x) {
    if (x.empty()) {
        return -120.0;
    }
    double ss = 0.0;
    for (float v : x) {
        ss += (double) v * (double) v;
    }
    return 20.0 * std::log10(std::sqrt(ss / (double) x.size()) + 1e-12);
}

// Mono IEEE-float32 WAV (format tag 3), so the normalised samples reach the model path through the same decode as any upload.
static inline std::vector<uint8_t> wav_f32_mono(const std::vector<float> & x, int32_t sample_rate) {
    const uint32_t data_bytes = (uint32_t) (x.size() * sizeof(float));
    std::vector<uint8_t> w(44 + data_bytes);
    auto put32 = [&](size_t o, uint32_t v) { std::memcpy(&w[o], &v, 4); };
    auto put16 = [&](size_t o, uint16_t v) { std::memcpy(&w[o], &v, 2); };
    std::memcpy(&w[0], "RIFF", 4);
    put32(4, 36 + data_bytes);
    std::memcpy(&w[8], "WAVEfmt ", 8);
    put32(16, 16);
    put16(20, 3);                       // IEEE float
    put16(22, 1);                       // mono
    put32(24, (uint32_t) sample_rate);
    put32(28, (uint32_t) sample_rate * 4);
    put16(32, 4);
    put16(34, 32);
    std::memcpy(&w[36], "data", 4);
    put32(40, data_bytes);
    if (data_bytes) {
        std::memcpy(&w[44], x.data(), data_bytes);
    }
    return w;
}

} // namespace asr_vibevoice
