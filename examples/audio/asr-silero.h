// asr-silero.h -- native Silero VAD (v5/v6 16 kHz) in plain C++, no ggml dependency.
//
// Reads whisper.cpp's ggml-format Silero file (models/convert-silero-vad-to-ggml.py, magic "ggml",
// e.g. ggml-silero-v6.2.0.bin / models/for-tests-silero-v6.2.0-ggml.bin) and produces one speech
// probability per 512-sample window, the input of asr_vad_postprocess() (asr-vad-post.h).
//
// SOURCE-READ (MIT): ggml-org/whisper.cpp src/whisper.cpp:4678-4812 (graph: reflect pad 64+64,
// STFT conv k256 s128 -> |re,im|, 4 conv+ReLU encoder, single LSTM step, ReLU, 1x1 conv, sigmoid),
// :4933-5230 (file layout), :5264-5348 (chunking: zero-padded last window, LSTM state carried
// across windows, state reset at the start of every detect call). Written independently.
//
// Like whisper.cpp each window is processed on its own (no 64-sample left context from the
// previous window; the left context is the reflect pad), so probabilities are comparable with
// whisper.cpp's. ggml feeds the convolutions through F16 im2col, this code stays in F32, so the
// two agree to ~1e-3, not bit for bit (measured: tests/test-asr-silero.cpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct asr_silero_model {
    int32_t n_window  = 512;
    int32_t n_context = 64;
    std::vector<float> stft;                 // [258][256]
    std::vector<float> enc_w[4];             // [oc][ic][k]
    std::vector<float> enc_b[4];
    int32_t enc_ic[4] = {0, 0, 0, 0};
    int32_t enc_oc[4] = {0, 0, 0, 0};
    int32_t enc_k [4] = {0, 0, 0, 0};
    std::vector<float> lstm_wih, lstm_whh, lstm_bih, lstm_bhh; // [512][128] x2, [512] x2
    std::vector<float> out_w;                // [128]
    float              out_b = 0.0f;
    std::string        type, version;
};

// false + *err on any malformed / unexpected file (wrong magic, shapes, missing tensors, truncation).
bool asr_silero_load(const std::string & path, asr_silero_model & m, std::string * err);

// One probability per window (last window zero-padded). LSTM state starts at zero.
std::vector<float> asr_silero_probs(const asr_silero_model & m, const float * samples, size_t n_samples);

// Streaming entry (row S69-ENGREV-WHISPERCPP-STREAMVAD-2026-10-27-01). The whole-clip call above resets the LSTM state
// on every call, so a stream fed step by step lost the state at every step boundary. whisper.cpp splits the same way:
// detect_speech_no_reset keeps the state, detect_speech resets it (ggml-org/whisper.cpp src/whisper.cpp:5256-5340, issue
// #3677; MIT, SOURCE-READ, no code copied).
//   feed()  takes ANY number of samples: completed 512-sample windows append one probability each, the rest is kept
//           for the next call. So feed(a) then feed(b) gives the same probabilities as one feed(a ++ b).
//   flush() zero-pads and processes the unfinished window, if any (end of the audio).
//   reset() back to the start-of-audio state.
// asr_silero_probs() is feed(all) + flush() on a fresh stream: one code path for both.
//
// left_context (default off = whisper.cpp semantic, each window reflect-padded on both sides): when on, the 64 samples
// left of a window are the LAST 64 samples of the previous window (zeros before the first), the semantic of the
// reference Silero runtime (the SYSTRAN Whisper port's vad.py:329-385 SileroVADModel.__call__, MIT, SOURCE-READ, no
// code copied). They are carried in the stream object, or chunked audio would differ from the whole clip at every chunk
// boundary. BENCH-PENDING: this comment makes no speed claim; nothing here was timed.
struct asr_silero_stream {
    std::vector<float> h, c;      // LSTM state, 128 each
    std::vector<float> pending;   // samples of the unfinished window (fewer than n_window)
    std::vector<float> ctx;       // last 64 samples of the previous window (left_context mode)
    bool left_context = false;
};
void asr_silero_stream_reset(asr_silero_stream & s);
void asr_silero_stream_feed(const asr_silero_model & m, asr_silero_stream & s, const float * samples, size_t n_samples,
                            std::vector<float> & probs);
void asr_silero_stream_flush(const asr_silero_model & m, asr_silero_stream & s, std::vector<float> & probs);
