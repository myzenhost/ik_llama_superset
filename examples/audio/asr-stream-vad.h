// asr-stream-vad.h -- per-step decisions for a live transcription stream from Silero speech probabilities.
// std-only and model-free (it takes probabilities, not audio), so the rules are unit-tested without any model.
//
// Row S69-ENGREV-WHISPERCPP-STREAMVAD-2026-10-27-01. The stream loop decoded EVERY step, silent or not, and closed a
// line only on a fixed step count. With the probabilities of asr_silero_stream (asr-silero.h) a step is:
//   decode    false when the step has no speech AND the open line holds none: nothing to transcribe, skip the
//             encoder and decoder (whisper.cpp's stream example has the same idea with an energy VAD: wait for
//             speech; ggml-org/whisper.cpp examples/stream/stream.cpp use_vad branch, MIT, SOURCE-READ, no code copied).
//   endpoint  true when the open line holds speech and the audio has now been silent for endpoint_silence_ms: the
//             line ends here instead of at the fixed step count. A pause shorter than that inside a sentence does
//             not end the line (the silence run restarts at the next speech probability).
// One probability covers n_window samples (512 at 16 kHz = 32 ms).
#pragma once

#include <cstddef>
#include <cstdint>

struct asr_stream_vad_params {
    float threshold           = 0.5f;   // speech when probability >= threshold
    int   endpoint_silence_ms = 800;    // 0 = never endpoint on silence (decode skipping still applies)
    int   prob_ms             = 32;     // audio covered by one probability
};

struct asr_stream_vad_state {
    bool    line_has_speech = false;    // speech since the last line end
    int64_t silence_ms      = 0;        // length of the current silence run
    int64_t n_decode = 0, n_skip = 0, n_endpoint = 0;   // counters for the caller's report
};

struct asr_stream_vad_decision {
    bool step_speech = false;   // some probability of this step reached the threshold
    bool decode      = true;
    bool endpoint    = false;
};

// Feed the probabilities of ONE step. After an endpoint the line is closed: line_has_speech is false again.
inline asr_stream_vad_decision asr_stream_vad_step(asr_stream_vad_state & s, const asr_stream_vad_params & p,
                                                   const float * probs, size_t n_probs) {
    asr_stream_vad_decision d;
    for (size_t i = 0; i < n_probs; i++) {
        if (probs[i] >= p.threshold) {
            d.step_speech     = true;
            s.line_has_speech = true;
            s.silence_ms      = 0;
        } else {
            s.silence_ms += p.prob_ms;
        }
    }
    d.decode = d.step_speech || s.line_has_speech;
    if (s.line_has_speech && p.endpoint_silence_ms > 0 && s.silence_ms >= p.endpoint_silence_ms) {
        d.endpoint        = true;       // this step is still decoded: it gives the line its final text
        s.line_has_speech = false;
        s.n_endpoint++;
    }
    if (d.decode) s.n_decode++; else s.n_skip++;
    return d;
}

// Token-driven end of utterance (same row, needs no VAD model). A TDT decoder emits nothing on a blank frame, so the frames
// after the last emitted piece's end are the blank run the model itself saw. When the window holds at least one piece (the
// decoder confirmed speech) and that run reaches eou_blank_frames, the line ends. A pause inside a sentence is shorter than
// the run, or is followed by more pieces, which moves the last end forward.
// Tok is any type with integer .frame and .duration (asr_tdt_token); a duration of 0 counts as one frame.
struct asr_stream_eou_params {
    int eou_blank_frames = 0;   // 0 = off; 10 frames @ 80 ms = 800 ms of model-confirmed silence
};

// frames after the end of the last piece in a window of n_frames encoder frames; -1 when the window holds no piece
template <typename Tok>
inline int asr_stream_trailing_blank_frames(const Tok * v, size_t n, int n_frames) {
    if (n == 0) {
        return -1;
    }
    int end = 0;
    for (size_t i = 0; i < n; i++) {
        const int d = v[i].duration > 0 ? (int) v[i].duration : 1;
        const int e = (int) v[i].frame + d;
        if (e > end) {
            end = e;
        }
    }
    return n_frames > end ? n_frames - end : 0;
}

template <typename Tok>
inline bool asr_stream_eou(const asr_stream_eou_params & p, const Tok * v, size_t n, int n_frames) {
    if (p.eou_blank_frames <= 0) {
        return false;
    }
    const int t = asr_stream_trailing_blank_frames(v, n, n_frames);
    return t >= 0 && t >= p.eou_blank_frames;
}
