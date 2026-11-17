// asr-halluc.h
//
// Hallucination silence-threshold skip: given word-timestamped segments decoded for one
// window, decide whether the decoder hallucinated into silence and where to resume.
//
// Semantics from SYSTRAN/faster-whisper (MIT, faster_whisper/transcribe.py @ 9fa78645,
// word_anomaly_score / is_segment_anomaly / the `hallucination_silence_threshold` block
// ~L1247-1345). Written independently. Times are in seconds; the caller converts the returned
// seek time to frames.
#pragma once

#include <functional>
#include <string>
#include <vector>

struct asr_word {
    std::string word;
    double start = 0.0;
    double end   = 0.0;
    double probability = 0.0;
};

struct asr_word_segment {
    double start = 0.0;
    double end   = 0.0;
    std::vector<asr_word> words;
};

// 1.0 if probability < 0.15; + (0.133-dur)*15 if dur < 0.133; + (dur-2.0) if dur > 2.0.
//
// time_resolution: the step of the word times, in seconds (0 = exact). A word time cannot say a word is
// shorter than one step, so the short-word term uses max(dur, time_resolution). Whisper's 20 ms grid makes
// that a no-op; an audio LLM's DTW grid is one audio position (0.160 s on Voxtral), coarser than the 0.133 s
// limit itself, and real fast words share one position (zero duration). MEASURED 2026-10-19: without the
// floor two such words in "to its place amidst the tents." score 4.0 >= 3 and real speech was dropped.
// The long-word term and the probability term are unchanged.
double asr_word_anomaly_score(const asr_word & w, double time_resolution = 0.0);

// First 8 non-punctuation words: anomalous when score >= 3 or score + 0.01 >= n_words.
// A segment without words is never anomalous.
bool asr_segment_is_anomaly(const asr_word_segment & seg, double time_resolution = 0.0);

struct asr_halluc_params {
    double threshold = 2.0;              // hallucination_silence_threshold, seconds
    double time_offset = 0.0;            // start time of the decoded window
    double segment_duration = 30.0;      // length of the decoded window
    double window_end_time = 30.0;       // end time of the window (time_offset + window size)
    double last_speech_timestamp = 0.0;  // end of the last confirmed word before this window
    double content_duration = 30.0;      // total audio duration
    double time_resolution = 0.0;        // step of the word times (asr_word_anomaly_score)
};

enum asr_halluc_action {
    ASR_HALLUC_KEEP,          // nothing suspicious
    ASR_HALLUC_SKIP_LEADING,  // drop ALL segments, re-decode from seek_time
    ASR_HALLUC_TRUNCATE,      // keep segments [0, n_keep), resume at seek_time
};

struct asr_halluc_decision {
    asr_halluc_action action = ASR_HALLUC_KEEP;
    double seek_time = 0.0;      // where decoding resumes (seconds); only meaningful if action != KEEP
    size_t n_keep    = 0;        // TRUNCATE: number of leading segments kept
    bool   seek_to_end = false;  // TRUNCATE: remaining audio < threshold => jump to content end
};

asr_halluc_decision asr_halluc_check(const asr_halluc_params & p, const std::vector<asr_word_segment> & segs);


// ---- driving the check from a decoder WITHOUT timestamp tokens (an audio LLM: llama-audio) ----------------
// (BENCH-PENDING: n/a -- "faster-whisper" below is the project's name, no speed is claimed)
//
// faster-whisper gets its segments from Whisper's timestamp tokens. An audio LLM emits none, so segments are
// cut from the word stream: a segment ends after a word whose text ends in sentence punctuation (. ! ? and
// the full-width forms), or where the next word starts more than split_gap_s after this one ends.
std::vector<asr_word_segment> asr_words_to_segments(const std::vector<asr_word> & words, double split_gap_s);

// Decodes [t0, t1) of a file with the silence-threshold skip, the way faster-whisper's generate_segments loop
// moves its seek pointer (transcribe.py: the hallucination_silence_threshold block):
//   KEEP           -> every segment is kept, done
//   SKIP_LEADING   -> nothing is kept, decode again from seek_time (the leading silence is skipped)
//   TRUNCATE       -> segments [0, n_keep) are kept, decode again from seek_time (or stop: seek_to_end)
// decode(a, b, words) decodes audio [a, b) and returns its words with times in FILE seconds; false = a hard
// failure (returned as false). max_decodes bounds the loop (every re-decode starts >= 1 s later anyway).
struct asr_halluc_span_result {
    std::vector<asr_word_segment> kept;
    int    n_decodes      = 0;
    int    n_skip_leading = 0;
    int    n_truncate     = 0;
    double last_speech    = 0.0;   // end of the last kept word (carried into the next span)
    std::vector<std::string> trace; // one line per decision, for the log
};

using asr_decode_span_fn = std::function<bool(double t0, double t1, std::vector<asr_word> & words)>;

bool asr_halluc_decode_span(double t0, double t1, double threshold, double content_duration,
                            double last_speech_timestamp, double split_gap_s, int max_decodes,
                            const asr_decode_span_fn & decode, asr_halluc_span_result & out,
                            double time_resolution = 0.0);
