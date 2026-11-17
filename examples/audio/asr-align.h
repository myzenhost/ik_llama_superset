// asr-align.h -- word timestamps for an LLM speech decoder, from its attention over the audio positions.
//
// whisper.cpp aligns tokens to audio with DTW over the cross-attention of chosen "alignment heads"
// (ggml-org/whisper.cpp, MIT, src/whisper.cpp: whisper_exp_compute_token_level_timestamps_dtw and
// median_filter; openai/whisper timing.py is the reference it follows). An audio LLM (Voxtral, Qwen3-ASR,
// ultravox ...) has no cross-attention: the audio embeddings sit in the SAME KV cache as the text, so the
// equivalent signal is the self-attention row of the query that produced each text token, restricted to the
// KV cells that hold audio embeddings. This file turns those rows into token and word times with the
// existing DTW pipeline (asr-dtw.h). Written independently; no code copied.
//
// Rows are captured by the caller (llama-audio: eval callback on the kq_soft_max tensors, flash attention
// off). Row r is the attention of the forward pass whose logits produced token r, so the last row may be the
// pass that produced the end-of-generation token: it then gives the END of the last real token.
#pragma once

#include "asr-halluc.h" // asr_word

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct asr_align_token {
    int32_t     id = -1;
    std::string piece;          // detokenised text of the token (leading space = word start)
    double      prob = 0.0;     // softmax probability of the sampled token (temperature-free)
};

struct asr_align_input {
    std::vector<asr_align_token> tokens;  // generated tokens, end-of-generation excluded
    std::vector<float>           attn;    // [n_rows][n_heads][n_frames], n_rows = tokens + (has_eog_row ? 1 : 0)
    int    n_heads       = 0;
    int    n_frames      = 0;             // audio positions that carry REAL audio (padding already cut)
    bool   has_eog_row   = false;
    double sec_per_frame = 0.0;           // seconds of audio per audio position
    double t_offset      = 0.0;           // start of the decoded window in the file
    double t_end         = 0.0;           // end of the decoded window in the file (clamp)
    int    medfilt_width = 7;             // odd; whisper's default
    // optional, n_frames values in [0, 1] (asr_align_speech_weights): weights the DTW cost per audio position and
    // trims every word to its first..last position with weight > 0.5. Empty = off (plain whisper.cpp DTW).
    std::vector<float> speech_w;
};

struct asr_align_result {
    std::vector<double>   tok_t0, tok_t1; // per token, seconds in the file
    std::vector<asr_word> words;
};

// [first token, one past last token) of every word. A token whose piece starts with whitespace starts a new
// word; a token with no letter or digit (punctuation, a lone space) is attached to the word before it.
std::vector<std::pair<int, int>> asr_align_group_words(const std::vector<std::string> & pieces);

// true when the piece has no letter/digit (ASCII test; any byte >= 0x80 counts as a letter so UTF-8 text
// is never treated as punctuation).
bool asr_align_is_punct(const std::string & piece);

// false + *err on invalid input (sizes, frames < 2, no tokens). Applies asr_align_truncate_long_words to the result.
bool asr_align_words(const asr_align_input & in, asr_align_result & out, std::string * err);

// The sentence-boundary hack of faster-whisper / openai-whisper add_word_timestamps: DTW hands a pause to the word
// AFTER it (the path has to stay on some row while the audio is silent), so a word right after a sentence end that
// is longer than 2 x median word duration (median capped at 0.7 s) is cut to start 2 x median before its end.
// MEASURED 2026-10-19 on Voxtral, clip + 3 s silence + clip: without it the first word of the second clip started at
// the end of the first clip (3.53 s, the true start 6.51 s). A pause inside a sentence is not handled (nor upstream).
void asr_align_truncate_long_words(std::vector<asr_word> & words);

// Seconds per audio position, from token counts the projector produced for two probe lengths.
//   n_a = tokens for a_s seconds, n_b = tokens for b_s seconds (b_s > a_s).
//   n_b > n_a  -> proportional projector: (b_s - a_s) / (n_b - n_a)
//   n_b == n_a -> fixed-chunk projector (whisper style, every chunk padded to chunk_s): chunk_s / n_a
// Returns <= 0 when the counts do not fit either model.
double asr_align_seconds_per_position(double a_s, int n_a, double b_s, int n_b, double chunk_s);

// Audio positions that carry real audio: min(n_positions, ceil(duration / sec_per_frame)).
int asr_align_valid_frames(int n_positions, double duration_s, double sec_per_frame);

// Per audio position speech weight in [0, 1] from the samples' energy (position f = samples
// [f*spf*sr, (f+1)*spf*sr)). ref = 95th percentile of the positions' dB, floor = 10th percentile; when the
// two are more than 20 dB apart (the window has silence or a noise bed) the threshold is a third of the way
// from floor to ref, else ref - 40 dB (a window of continuous speech: nothing is weighted down). The weight
// rises linearly from 0 at the threshold to 1 halfway to ref.
// WHY: whisper's cross-attention is trained to look at the audio; an audio LLM's self-attention over a long
// silence is noise, and the per-frame normalisation of the DTW amplifies it. MEASURED 2026-10-19 on
// Voxtral-Mini-3B (temp/arex1014/engq_b_1019/dtw_study3.log, 6 layouts of two LibriSpeech clips with 3-12 s of
// silence or -50 dBFS noise): a mid-sentence word swallowed 8-14 s of silence and the rest of the sentence
// collapsed onto one position (error score 264.8); with this weight + edge trimming 1.29. No head or layer
// subset came close (best single layer 4.4 on these data, all others >= 28).
std::vector<float> asr_align_speech_weights(const float * pcm, size_t n_samples, int sample_rate, double sec_per_frame,
                                            int n_frames);
