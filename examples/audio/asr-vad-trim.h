// asr-vad-trim.h -- cut the silence out of a clip from its VAD speech spans, and map times on the trimmed
// signal back to the original clip.
//
// Used by llama-server's opt-in --asr-vad-trim on /v1/audio/transcriptions: the model is given the speech only
// (silence draws refusals or invented text from an LLM audio model and costs encoder windows), and the segment
// times of the reply are mapped back so they index the upload the client sent.
//
// Std-only, model-free: the spans come from asr_vad_gate() (asr-vad-gate.h) or any other source.
//
// Design, from two readings (both MIT, SOURCE-READ, no code copied):
//   * whisper.cpp src/whisper.cpp (VAD path of whisper_full): speech spans joined with 100 ms of digital silence
//     so the model does not meet a hard splice; a (processed, original) point table maps times back. Its issue
//     #3711: the end of a span must map to the end of THAT span in the original, never past the removed silence.
//   * faster-whisper vad.py SpeechTimestampsMap / transcribe.py restore_speech_timestamps: no gap, and a word is
//     assigned to its chunk by its MIDPOINT, which removes the drift at a join.
// Here: the pad is a parameter (0 = contiguous); a boundary time is resolved by ROLE instead of by interpolation
// -- a start that falls on a join or in a pad belongs to the NEXT piece, an end to the PREVIOUS one -- and words
// take the midpoint rule. A gap no longer than the pad is kept as it is (the original audio is better than a
// splice and nothing would be saved).
#pragma once

#include "asr-vad-post.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct asr_vad_trim_params {
    int32_t pad_ms = 100;   // digital silence written between two kept pieces; 0 = pieces are contiguous
};

// One run of original samples that was kept. Sample indices at the clip's own rate.
struct asr_vad_trim_piece {
    int64_t out_start = 0;  // first sample of the piece in the trimmed signal
    int64_t out_end   = 0;  // one past its last sample there
    int64_t in_start  = 0;  // the original sample out_start was copied from
    int64_t in_end() const { return in_start + (out_end - out_start); }
};

struct asr_vad_trim_map {
    std::vector<asr_vad_trim_piece> pieces;     // ordered, non-overlapping in both signals
    int64_t n_in        = 0;                    // samples of the original clip
    int64_t n_out       = 0;                    // samples of the trimmed signal (pieces + pads)
    int32_t sample_rate = 0;

    bool empty() const { return pieces.empty(); }

    // Seconds on the trimmed signal -> seconds on the original clip. Both are monotone non-decreasing in t and
    // never exceed the original length. An empty map returns t unchanged.
    double start_to_original(double t) const;   // t is where something BEGINS
    double end_to_original(double t) const;     // t is where something ENDS

    // A segment may span several pieces: its start and its end are each mapped by role; t1 >= t0 afterwards.
    void segment_to_original(double & t0, double & t1) const;
    // A word lies in ONE piece, the one that holds its midpoint (the nearer piece when the midpoint is in a pad,
    // the earlier one on a tie); both edges are clamped into that piece.
    void word_to_original(double & t0, double & t1) const;
};

// Keeps the samples of `spans_16k` (speech spans as sample indices at 16 kHz, as asr_vad_gate() returns them) and
// drops the rest. Spans are converted to the clip rate, clamped to the clip, and made ordered and non-overlapping;
// two spans whose gap is no longer than the pad become one piece that includes the gap. Returns the trimmed
// signal and fills `map`. No span -> an empty signal and an empty map.
std::vector<float> asr_vad_trim(const float * pcm, size_t n_samples, int32_t sample_rate,
                                const std::vector<asr_vad_span> & spans_16k, const asr_vad_trim_params & p,
                                asr_vad_trim_map & map);

// Mono 32-bit float WAV (format tag 3) of `pcm`: the trimmed signal goes back to the audio path as an ordinary
// upload, bit-exact (no int16 rounding).
std::vector<uint8_t> asr_wav_f32_encode(const float * pcm, size_t n_samples, int32_t sample_rate);
