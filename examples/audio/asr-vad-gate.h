// asr-vad-gate.h -- "is there any speech in this clip?" for a whole upload, from the native Silero VAD.
//
// Composes asr_silero_probs() (asr-silero.h) and asr_vad_postprocess() (asr-vad-post.h) over mono PCM at any sample
// rate: Silero runs at 16 kHz only, so other rates are brought to 16 kHz first (linear interpolation -- the gate only
// needs the speech / no-speech decision and the span edges to ~32 ms, one Silero window). Used by llama-server's
// opt-in --asr-vad gate on /v1/audio/transcriptions: no span -> the model is not run and the transcript is empty.
//
// Behaviour follows whisper.cpp (MIT, src/whisper.cpp:7945-7953, whisper_full: with params.vad on and no speech
// sample left, result_all is cleared and the decoder is never run -> an empty transcript). Written independently.
#pragma once

#include "asr-silero.h"
#include "asr-vad-post.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct asr_vad_gate_result {
    std::vector<asr_vad_span> spans;     // speech spans, sample indices at 16 kHz
    size_t                    n_probs        = 0;    // Silero windows evaluated
    float                     max_prob       = 0.0f; // highest per-window speech probability
    size_t                    n_samples_16k  = 0;    // length of the 16 kHz signal the spans index
    bool has_speech() const { return !spans.empty(); }
};

// Linear-interpolation resampler (sr_in, sr_out > 0). Returns the input unchanged when the rates match.
std::vector<float> asr_resample_linear(const float * x, size_t n, int32_t sr_in, int32_t sr_out);

// pp.window_samples and pp.sample_rate are overwritten with the model's window and 16000.
asr_vad_gate_result asr_vad_gate(const asr_silero_model & m, const float * pcm, size_t n_samples, int32_t sample_rate,
                                 asr_vad_post_params pp);
