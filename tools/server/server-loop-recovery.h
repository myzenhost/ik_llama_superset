#pragma once

// Reasoning-loop recovery: the two model-free halves of the request option reasoning_loop_recovery.
//
//  1. reasoning_coverage(): a word-level score of how much of the recent reasoning text is made of 12-word phrases the
//     model has already written at least three times. It sees loops whose copies are far apart (a verification pass
//     repeated with long code drafts between the copies), which the token-span test in server-repetition.h cannot see
//     once the copies are more than its 4096-token window apart. The measurement that decided to add it is in
//     tests_cpp/test_server_loop_recovery.cpp (scenario A).
//
//  2. effort_splice(): the prompt for the once-only "recover" retry, built on TOKEN IDS. The high-effort sentence is
//     replaced by the low-effort one inside the first turn of the prompt (never a later turn, so text a user wrote
//     later in the conversation is never touched), without decoding and re-encoding the prompt. If the sentence is not
//     there as whole tokens the splice fails and the caller falls back to stopping.
//
// Contract read from Strata serve/server.py @82f46a8c8 (MIT): reasoning_repeat_coverage and focused_recovery_prompt
// (lines 130-209) and the decode loop (2960-3085). SOURCE-READ for the spec only; no code copied.
// Header-only and free of llama.h so it can be unit-tested with a plain compiler.

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace server_loop_recovery {

// The Qwen3.8 chat template's effort sentences (tokenizer.chat_template of Qwen3.8-27B-UD-Q4_K_XL.gguf, read
// 2026-10-17). The template puts one of them at the start of the system turn when thinking is on: "xhigh" is the
// default (and what "high" maps to), "low" is what the retry switches to.
static constexpr const char * QWEN38_EFFORT_HIGH =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider "
    "plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
static constexpr const char * QWEN38_EFFORT_LOW =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without "
    "unnecessary elaboration.";

// Retry sampler floor (Strata's values): temperature >= 1.0, presence penalty >= 1.5. Nothing else changes.
static constexpr float RETRY_MIN_TEMP     = 1.0f;
static constexpr float RETRY_MIN_PRESENCE = 1.5f;

enum mode : int32_t {
    MODE_OFF     = 0, // nothing new: repetition_detection behaves exactly as before
    MODE_STOP    = 1, // a loop found inside the reasoning ends the reply (stop_type "repetition")
    MODE_RECOVER = 2, // the first loop found inside the reasoning resumes once with the low-effort prompt
};

struct coverage_params {
    int32_t ngram     = 12;     // phrase length in words
    int32_t min_count = 3;      // a phrase counts as repeated once it has been written this many times
    int32_t window    = 2000;   // words scored at the end of the text
    int32_t history   = 30000;  // words kept for counting
    float   threshold = 0.25f;  // fraction of the window that must be covered; <= 0 = off
    int32_t every     = 512;    // output tokens between two checks
};

// Lowercased words and single punctuation marks, like the regex \w+|[^\w\s]. Bytes >= 0x80 (UTF-8) count as word
// characters, so non-ASCII words stay whole.
inline std::vector<uint64_t> words_hashed(const std::string & text, size_t keep_last) {
    std::vector<uint64_t> out;
    auto is_word = [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c >= 0x80;
    };
    auto is_space = [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    };
    const size_t n = text.size();
    size_t i = 0;
    while (i < n) {
        const unsigned char c = (unsigned char) text[i];
        if (is_space(c)) {
            ++i;
            continue;
        }
        uint64_t h = 1469598103934665603ULL; // FNV-1a
        if (is_word(c)) {
            while (i < n && is_word((unsigned char) text[i])) {
                unsigned char d = (unsigned char) text[i++];
                if (d >= 'A' && d <= 'Z') {
                    d = (unsigned char) (d - 'A' + 'a');
                }
                h = (h ^ d) * 1099511628211ULL;
            }
        } else {
            h = ((h ^ c) * 1099511628211ULL) ^ 0x9e3779b97f4a7c15ULL; // punctuation never equals a word
            ++i;
        }
        out.push_back(h);
    }
    if (keep_last > 0 && out.size() > keep_last) {
        out.erase(out.begin(), out.end() - (std::ptrdiff_t) keep_last);
    }
    return out;
}

// Coverage over a word stream: fraction of the last `window` words covered by an ngram that occurs >= min_count
// times in the whole stream. 0 when the stream is shorter than the window.
inline float coverage_of_words(const std::vector<uint64_t> & w, const coverage_params & p) {
    const size_t n   = w.size();
    const size_t win = (size_t) (p.window > 0 ? p.window : 0);
    const size_t g   = (size_t) (p.ngram > 0 ? p.ngram : 1);
    if (win == 0 || n < win || n < g) {
        return 0.0f;
    }
    // gram hash: a polynomial over the word hashes; a collision can only over-count, never hide a loop
    const size_t n_grams = n - g + 1;
    std::vector<uint64_t> gh(n_grams);
    for (size_t s = 0; s < n_grams; ++s) {
        uint64_t h = 0;
        for (size_t k = 0; k < g; ++k) {
            h = h * 0x100000001b3ULL + w[s + k];
        }
        gh[s] = h;
    }
    std::unordered_map<uint64_t, int32_t> count;
    count.reserve(n_grams * 2);
    for (const uint64_t h : gh) {
        ++count[h];
    }
    const size_t w0 = n - win; // first word of the scored window
    std::vector<uint8_t> covered(win, 0);
    // only grams that overlap the window can cover it
    const size_t s0 = w0 + 1 >= g ? w0 + 1 - g : 0;
    for (size_t s = s0; s < n_grams; ++s) {
        if (count[gh[s]] < p.min_count) {
            continue;
        }
        for (size_t k = 0; k < g; ++k) {
            const size_t pos = s + k;
            if (pos >= w0) {
                covered[pos - w0] = 1;
            }
        }
    }
    size_t c = 0;
    for (const uint8_t b : covered) {
        c += b;
    }
    return (float) c / (float) win;
}

inline float reasoning_coverage(const std::string & text, const coverage_params & p) {
    return coverage_of_words(words_hashed(text, (size_t) (p.history > 0 ? p.history : 0)), p);
}

// First index at which `needle` occurs in hay[from, to), or -1.
template <typename Tok>
inline int64_t find_tokens(const std::vector<Tok> & hay, const std::vector<Tok> & needle, size_t from, size_t to) {
    if (needle.empty() || to > hay.size() || to < needle.size()) {
        return -1;
    }
    for (size_t i = from; i + needle.size() <= to; ++i) {
        bool same = true;
        for (size_t k = 0; same && k < needle.size(); ++k) {
            same = hay[i + k] == needle[k];
        }
        if (same) {
            return (int64_t) i;
        }
    }
    return -1;
}

// One way of writing the sentence as tokens: the high and low forms must be swapped as a pair (the same trailing text
// is tokenized onto both, so a tokenizer that glues punctuation to the newlines after it still matches).
template <typename Tok>
struct splice_variant {
    std::vector<Tok> high;
    std::vector<Tok> low;
};

template <typename Tok>
struct splice_result {
    bool             ok    = false;
    size_t           pos   = 0;   // index of the first replaced token
    size_t           n_old = 0;   // tokens removed
    size_t           n_new = 0;   // tokens inserted
    std::vector<Tok> tokens;      // the new token sequence (empty unless ok)
};

// Replace the first whole-token occurrence of a variant's `high` inside tokens[0, first_turn_end) with its `low`.
// Variants are tried in order; the first that is found wins. Everything after the sentence (the rest of the prompt and
// every generated token) is kept as it is, token for token.
template <typename Tok>
inline splice_result<Tok> effort_splice(const std::vector<Tok> & tokens, size_t first_turn_end,
                                        const std::vector<splice_variant<Tok>> & variants) {
    splice_result<Tok> r;
    const size_t end = first_turn_end < tokens.size() ? first_turn_end : tokens.size();
    for (const auto & v : variants) {
        if (v.high.empty() || v.high == v.low) {
            continue;
        }
        const int64_t at = find_tokens(tokens, v.high, 0, end);
        if (at < 0) {
            continue;
        }
        r.ok    = true;
        r.pos   = (size_t) at;
        r.n_old = v.high.size();
        r.n_new = v.low.size();
        r.tokens.reserve(tokens.size() - r.n_old + r.n_new);
        r.tokens.insert(r.tokens.end(), tokens.begin(), tokens.begin() + at);
        r.tokens.insert(r.tokens.end(), v.low.begin(), v.low.end());
        r.tokens.insert(r.tokens.end(), tokens.begin() + at + (std::ptrdiff_t) r.n_old, tokens.end());
        return r;
    }
    return r;
}

// End (exclusive) of the first turn: the index just after the first end-of-turn token, or the whole prompt when there
// is none. `is_turn_end` is the vocabulary's end-of-generation test (for ChatML templates, <|im_end|>).
template <typename Tok, typename Pred>
inline size_t first_turn_end(const std::vector<Tok> & prompt, Pred is_turn_end) {
    for (size_t i = 0; i < prompt.size(); ++i) {
        if (is_turn_end(prompt[i])) {
            return i + 1;
        }
    }
    return prompt.size();
}

// Parse the request / flag value. Accepts false, true (= recover), "off", "stop", "recover". Returns false on anything else.
inline bool parse_mode(const std::string & s, int32_t & out) {
    if (s == "off" || s == "false" || s == "none" || s.empty()) {
        out = MODE_OFF;
    } else if (s == "stop") {
        out = MODE_STOP;
    } else if (s == "recover" || s == "true") {
        out = MODE_RECOVER;
    } else {
        return false;
    }
    return true;
}

inline const char * mode_name(int32_t m) {
    return m == MODE_STOP ? "stop" : m == MODE_RECOVER ? "recover" : "off";
}

} // namespace server_loop_recovery
