// asr-fallback.h
//
// Temperature-fallback, quality-gated retry policy for ASR decoding. Model-free: it decides
// from per-attempt statistics whether a decode must be retried at the next temperature.
//
// Semantics from ggml-org/whisper.cpp (MIT, src/whisper.cpp @ d1be6fde, whisper_full_with_state
// ~L7042 temperature ladder, ~L7722 entropy gate, ~L7747 logprob/no-speech gate) and
// SYSTRAN/faster-whisper (MIT, transcribe.py @ 9fa78645, generate_with_fallback). Written
// independently against those semantics.
//
// A retry is needed when ANY of
//   - compression ratio of the text   > compression_ratio_thold      (looping text)
//   - average token logprob           < logprob_thold                 (low confidence)
//   - last-32-token entropy           < entropy_thold, n_tokens > 32  (whisper.cpp repetition gate)
// UNLESS the segment looks like silence: no_speech_prob > no_speech_thold AND
// avg_logprob < logprob_thold -- then the (empty-ish) result is accepted (faster-whisper rule;
// whisper.cpp reaches the same outcome by only retrying when no_speech_prob < thold).
// Boundary note: at no_speech_prob == thold exactly, faster-whisper retries, whisper.cpp does not;
// this implementation follows faster-whisper (strict '>').
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

enum asr_fail_reason : uint32_t {
    ASR_FAIL_NONE        = 0,
    ASR_FAIL_COMPRESSION = 1u << 0, // text repeats
    ASR_FAIL_LOGPROB     = 1u << 1, // avg logprob too low
    ASR_FAIL_ENTROPY     = 1u << 2, // tail tokens too repetitive
};

struct asr_decode_attempt {
    float       temperature   = 0.0f;
    std::string text;
    double      avg_logprob   = 0.0;  // mean log-probability of the emitted tokens (<= 0)
    double      no_speech_prob = 0.0; // P(no speech) from the first decoded step
    int         n_tokens      = 0;
    double      tail_entropy  = 1e9;  // see asr_tail_entropy(); huge = "not computed / fine"
};

struct asr_fallback_params {
    float temperature      = 0.0f;
    float temperature_inc  = 0.2f;   // <= 0 : single attempt at `temperature`
    // thresholds (a *_enabled = false skips that gate)
    bool   check_compression  = true;
    double compression_ratio_thold = 2.4;
    bool   check_logprob      = true;
    double logprob_thold      = -1.0;
    bool   check_no_speech    = true;
    double no_speech_thold    = 0.6;
    bool   check_entropy      = true;
    double entropy_thold      = 2.4;
    int    entropy_min_tokens = 32;  // gate applies only when n_tokens > this
    // What to return when every temperature failed.
    enum exhausted_policy {
        EXHAUSTED_BEST_LOGPROB, // faster-whisper: best avg_logprob among attempts under the
                                // compression threshold (all attempts if none is)
        EXHAUSTED_LAST,         // whisper.cpp: the last attempt is accepted as-is
    } on_exhausted = EXHAUSTED_BEST_LOGPROB;
};

// Temperature ladder [t0, t0+inc, ...] while t < 1.0 + 1e-6 (whisper.cpp). Index-multiplied
// (not accumulated) so there is no float drift; capped at 64 entries.
std::vector<float> asr_temperature_ladder(float t0, float inc);

// Shannon entropy (nats) of the token-id histogram of the last `window` tokens (whisper.cpp: 32).
double asr_tail_entropy(const std::vector<int32_t> & tokens, int window = 32);

struct asr_fallback_verdict {
    bool     needs_fallback = false;
    bool     is_silence     = false; // accepted because it looks like silence
    uint32_t reasons        = ASR_FAIL_NONE;
    double   compression_ratio = 0.0;
};

// Decision for ONE attempt. Computes the compression ratio from att.text.
asr_fallback_verdict asr_fallback_check(const asr_fallback_params & p, const asr_decode_attempt & att);

struct asr_fallback_result {
    std::vector<asr_decode_attempt>   attempts;
    std::vector<asr_fallback_verdict> verdicts;
    int  chosen        = -1;   // index into attempts
    bool exhausted     = false; // every temperature failed
};

// Drives the ladder with a caller-supplied decoder. decode(temperature) must return one attempt.
asr_fallback_result asr_run_fallback(const asr_fallback_params & p,
                                     const std::function<asr_decode_attempt(float)> & decode);
