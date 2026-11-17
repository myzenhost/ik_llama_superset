// asr-vad-post.h
//
// Post-processing state machine of a Silero-style neural VAD, as a PURE function over the
// per-window speech-probability array. The neural network itself (weights) is NOT part of this;
// feed it any probability sequence (one value per `window_samples` samples).
//
// Semantics from ggml-org/whisper.cpp (MIT, src/whisper.cpp @ d1be6fde,
// whisper_vad_segments_from_probs ~L5371), itself derived from silero-vad's
// get_speech_timestamps. Written independently; three behaviours worth knowing:
//   * hysteresis: speech starts at p >= threshold, ends only below threshold - 0.15 (floor 0.01)
//   * over-long speech is split at the last silence >= 98 ms seen inside it (Silero constant)
//   * whisper.cpp then MERGES segments whose gap is < 200 ms (merge_gap_ms), which also re-joins
//     pieces that max-speech splitting just separated when the split silence is short; set
//     merge_gap_ms = 0 to disable that merge.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct asr_vad_post_params {
    float   threshold         = 0.5f;
    int32_t min_speech_ms     = 250;
    int32_t min_silence_ms    = 100;
    float   max_speech_s      = 3.4e38f; // FLT_MAX = unlimited
    int32_t speech_pad_ms     = 30;
    int32_t merge_gap_ms      = 200;     // whisper.cpp hard-codes 200; 0 disables
    int32_t window_samples    = 512;     // samples per probability (Silero @16 kHz: 512)
    int32_t sample_rate       = 16000;
};

struct asr_vad_span {
    int64_t start = 0; // sample index, inclusive
    int64_t end   = 0; // sample index, exclusive
    double start_s(int32_t sr) const { return (double) start / sr; }
    double end_s  (int32_t sr) const { return (double) end   / sr; }
};

std::vector<asr_vad_span> asr_vad_postprocess(const float * probs, size_t n_probs, const asr_vad_post_params & p);
