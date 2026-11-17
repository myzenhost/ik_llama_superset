// speak-stream.h -- building blocks for `llama-audio --speak-stream` (speak while the reply is still being written).
//
// Today (--speak) the reply is generated in full, then synthesized in full, and the WAV appears only at the end, so the time
// until the first audible sample is "whole LLM time + whole TTS time". --speak-stream cuts the reply into sentences AS THE
// TOKENS ARRIVE, synthesizes each sentence the moment it is complete, and appends it to a WAV that is a valid file after every
// chunk. First audio then costs "LLM time of sentence 1 + TTS time of sentence 1".
//
// Two pieces, both free of llama/ggml so they are unit-tested without a model (tests/test-speak-stream.cpp):
//   speak_sentence_end   the cutter: where does the next speakable sentence of a growing text end?
//   speak_wav_sink       a mono 16-bit WAV that stays a valid file after every append (sizes re-patched each time)
//
// SOURCE-READ (the scheduling idea only; Apache-2.0, no code copied): alibaba/MNN @ 67450001627fc91232779068d85933c22845fae5
//   transformers/llm/engine/src/omni.cpp:2297-2500 (Omni::responseInterleaved: the talker starts before the thinker has
//   finished, the two advance in lock step) and :2519-2523 / :2988-3007 / :3249-3283 (audio leaves in chunks through a
//   waveform callback while generation continues; their release notes: "TTFA reduced by 2.7x", MNN's own figure on its own
//   models, not a prediction for this fork). What does not transfer: their talker consumes the thinker's hidden states inside
//   one model; here the speaker is a separate text-in backbone (OuteTTS), so only the sentence-level scheduling applies.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct speak_cut_params {
    size_t min_chars = 16;    // a sentence shorter than this (trimmed) is merged into the next one: "Yes." is not worth a TTS call
    size_t max_chars = 240;   // text with no sentence end for this long is cut at a clause or word boundary instead
};

// The end (exclusive byte index) of the first speakable sentence of text[from..], or npos when more text is needed.
//   - a sentence ends at . ! ? (and the CJK full stops), after any closing quote / bracket, or at a newline;
//   - '.' only ends a sentence when whitespace follows (so "3.5", "a.b" and "example.com" stay whole) and the word before it
//     is not an abbreviation ("Dr.", "e.g.", ...) or a lone initial ("J.");
//   - a terminator at the very end of the buffer is NOT yet an end unless `final_flush`: the next token may continue it ("3." -> "3.5");
//   - a candidate shorter than min_chars (trimmed) is skipped, the scan continues into the next sentence;
//   - `final_flush` (the text is complete) returns the whole remainder when it holds anything.
size_t speak_sentence_end(const std::string & text, size_t from, bool final_flush, const speak_cut_params & p = speak_cut_params());

// true when the chunk holds at least one letter or digit (a chunk of "..." or "***" is not worth a TTS call and would fail)
bool speak_has_speech(const std::string & chunk);

// Digits spelled out as English words, everything else unchanged, so a letters-only TTS front end does not drop them
// ("I owe 42 dollars." -> "I owe forty-two dollars."; "2024." -> "two thousand twenty-four.").
//   - "1,234,567" (comma + exactly three digits) is one number; "3.14" -> "three point one four";
//   - a run with a leading zero ("007") or longer than 12 digits is read digit by digit;
//   - a space is inserted where the words would otherwise touch a letter ("42nd" -> "forty-two nd").
// Called first by speak_tts_text: speak_has_speech counts digits as speech, so the speaker must keep them.
std::string speak_numbers_to_words(const std::string & text);

// The OuteTTS v1 prompt text for `text`: digits spelled out, lower-case a-z words joined by "<|text_sep|>" (one after
// the last word too), accented Latin letters folded to ASCII ("café" -> "cafe", "Straße" -> "strasse", "Łódź" -> "lodz").
// Every other byte separates words. EMPTY when nothing is left -- e.g. a CJK-only chunk, which speak_has_speech still
// accepts: the caller must skip it rather than send the speaker an empty prompt.
std::string speak_tts_text(const std::string & text);

// Upper bound on the acoustic codes worth generating for the OuteTTS prompt text `said` (the speak_tts_text output):
// max(100, 12 * letters + 50), never above n_predict. OuteTTS v1 emits 75 codes/s and speech runs ~6-15 letters/s, so this
// is >= 2x the slowest natural pace: a model that never emits an end token is stopped in proportion to the text, not at
// the fixed n_predict. Pure function, no model needed.
int speak_code_budget(const std::string & said, int n_predict);

// A mono s16 WAV written incrementally. The file is a valid WAV after every append() (RIFF and data sizes are patched
// each time), which is what lets a player or a test open it while the reply is still being spoken.
class speak_wav_sink {
public:
    speak_wav_sink() = default;
    ~speak_wav_sink();
    speak_wav_sink(const speak_wav_sink &) = delete;
    speak_wav_sink & operator=(const speak_wav_sink &) = delete;

    bool   open(const std::string & path, int32_t sample_rate, std::string & err);
    bool   append(const std::vector<float> & mono, float gain, std::string & err);   // samples * gain, clipped to s16
    void   close();
    bool   is_open() const { return m_f != nullptr; }
    size_t n_samples() const { return m_n; }

private:
    std::FILE * m_f = nullptr;
    size_t      m_n = 0;
};
