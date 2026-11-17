// mtmd-audio-chunks.h -- where to cut audio that is longer than an encoder window
//
// The audio preprocessors used to cut long input at blind multiples of the window (30 s for the whisper-style
// encoders and gemma4a): a word that straddles 30.000 s is split between two independently encoded windows and
// neither half is a word. whisper.cpp avoids that with its VAD: a speech segment longer than max_speech_duration_s
// is split at the last silence of at least 98 ms seen before the limit, and only hard-cut when there is none
// (SOURCE-READ ggml-org/whisper.cpp @ 6e4ab854 src/whisper.cpp:5396-5465, MIT; rewritten, no code copied).
//
// mtmd has no VAD model, so this plans the cuts from signal energy: each window ends at the quietest 100 ms stretch
// (the window length whisper.cpp uses as its minimum split silence, rounded to 10 ms frames) inside the last
// `search_samples` before the window limit, preferring the LATEST of the stretches within 3 dB of that minimum so
// that windows stay as long as possible. Input that fits one window gets exactly one chunk [0, n), so short audio
// is unchanged.

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

struct mtmd_audio_chunk_params {
    size_t max_samples    = 0;   // window limit (e.g. 30 s * sample rate); 0 = never split
    size_t search_samples = 0;   // how far back from the limit a cut may move (e.g. 10 s * sample rate)
    size_t frame_samples  = 160; // energy frame (10 ms at 16 kHz)
    int    smooth_frames  = 10;  // the quiet stretch looked for, in frames (100 ms)
    float  tie_db         = 3.0f; // stretches within this of the quietest one count as equally quiet; the latest wins
};

// [start, end) sample ranges that tile [0, n_samples) in order, each at most max_samples long
std::vector<std::pair<size_t, size_t>> mtmd_audio_plan_chunks(const float * samples, size_t n_samples,
                                                              const mtmd_audio_chunk_params & p);
