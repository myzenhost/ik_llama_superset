#pragma once

// Non-contiguous repeated-span detection for repetition_detection (span_min_len / span_min_count).
//
// The contiguous check in server-context.cpp (repetition_detected) only sees a pattern repeated BACK TO BACK. A model
// that loops on a paragraph with something different between the copies is invisible to it. This test asks: does the
// span that ends the output appear at least `min_count` times (non-overlapping) anywhere in the output window?
//
// Same contract as rtp-llm OnlineRepetitionTracker considerNonContiguousSpans (a rolling polynomial hash over
// power-of-two span lengths); SOURCE-READ rtp-llm OnlineRepetitionTracker.cc:85-136 (Apache-2.0), no code copied.
// Header-only and free of llama.h so it can be unit-tested with a plain compiler.
// SCOUT-DONE: BeeLlama server-loop-guard (engines/beellama.cpp) and rtp-llm OnlineRepetitionTracker scouted locally; both read for the contract.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace server_repetition {

// Longest window of recent tokens that is searched. Bounds the per-token cost at about
// window * (number of span lengths) operations.
static constexpr size_t NONCONTIG_WINDOW = 4096;

// true when the last L tokens, for some L in {min_len, 2*min_len, 4*min_len, ...} with L * min_count <= window size,
// occur at least min_count times without overlapping in the last NONCONTIG_WINDOW tokens (the tail copy included).
// A hash hit is confirmed by comparing the tokens, so a collision cannot produce a false positive.
template <typename Tok>
inline bool noncontiguous_span_repeated(const std::vector<Tok> & toks, int32_t min_len, int32_t min_count) {
    if (min_len < 1 || min_count < 2) {
        return false;
    }
    const size_t total = toks.size();
    const size_t n     = total < NONCONTIG_WINDOW ? total : NONCONTIG_WINDOW;
    const size_t base  = total - n;
    constexpr uint64_t B = 1000003ULL;

    // prefix hashes over the window: P[i] covers the first i tokens
    std::vector<uint64_t> P(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        P[i + 1] = P[i] * B + (uint64_t) ((int64_t) toks[base + i] + 1);
    }

    for (size_t L = (size_t) min_len; L * (size_t) min_count <= n; L *= 2) {
        uint64_t BL = 1;
        for (size_t i = 0; i < L; ++i) {
            BL *= B;
        }
        auto span_hash = [&](size_t s) { return P[s + L] - P[s] * BL; };
        const size_t tail_start = n - L;
        const uint64_t tail_h   = span_hash(tail_start);

        size_t found = 1;          // the tail copy
        size_t limit = tail_start; // earlier copies must END at or before this
        for (size_t end = limit; end >= L; ) {
            const size_t s = end - L;
            bool same = span_hash(s) == tail_h;
            for (size_t i = 0; same && i < L; ++i) {
                same = toks[base + s + i] == toks[base + tail_start + i];
            }
            if (same) {
                if (++found >= (size_t) min_count) {
                    return true;
                }
                end = s; // next copy must end at or before this copy's start
            } else {
                --end;
            }
        }
    }
    return false;
}

// Contiguous check moved out of server_slot::repetition_detected so it can be tested without llama.h: true when the tail of
// `toks` is one pattern of min_pat..max_pat tokens repeated min_count times back to back (shortest matching length wins).
// min_dup_tokens > 0 adds the rtp-llm min_duplicate_tokens floor: the back-to-back run must also span at least that many
// tokens (pattern length x copies, counting every copy the run really has), so "=====" or "-----" never ends a reply. A
// pattern that fails the floor is skipped and longer patterns are still tried. min_dup_tokens 0 = the old behaviour.
// SOURCE-READ rtp-llm OnlineRepetitionTracker.h:15 and .cc:113,146 (Apache-2.0), no code copied.
template <typename Tok>
inline bool contiguous_pattern_repeated(const std::vector<Tok> & toks, int32_t min_pat, int32_t max_pat, int32_t min_count,
                                        int32_t min_dup_tokens) {
    if (max_pat <= 0 || min_count < 2) {
        return false;
    }
    const size_t n = toks.size();
    for (int32_t len = min_pat < 1 ? 1 : min_pat; len <= max_pat; ++len) {
        if ((size_t) len * (size_t) min_count > n) {
            return false; // longer patterns need even more tokens
        }
        bool same = true;
        for (int32_t i = 1; same && i <= len; ++i) {
            const Tok ref = toks[n - i];
            for (int32_t m = 1; m < min_count; ++m) {
                if (toks[n - (size_t) len * m - i] != ref) {
                    same = false;
                    break;
                }
            }
        }
        if (!same) {
            continue;
        }
        if (min_dup_tokens <= 0) {
            return true;
        }
        // extend the run: copies beyond min_count, newest to oldest
        size_t copies = (size_t) min_count;
        while ((size_t) len * (copies + 1) <= n) {
            bool more = true;
            for (int32_t i = 1; more && i <= len; ++i) {
                more = toks[n - (size_t) len * copies - i] == toks[n - i];
            }
            if (!more) {
                break;
            }
            ++copies;
        }
        if ((size_t) len * copies >= (size_t) min_dup_tokens) {
            return true;
        }
    }
    return false;
}

// ---- token loop guard ------------------------------------------------------------------------------------------------
// Three model-free detectors over a sliding window of recent token ids, run every check_interval tokens:
//   periodic_tail    the whole window is one period of p <= max_period tokens, allowing a few mismatches
//                    (0.5% for p <= 8, 1% for p <= 64, 2% above) - a loop that stutters a token still counts;
//   ngram_dominance  reasoning region only: 8/16/32-grams that occur more than once cover >= half of the window;
//   low_entropy      the last 1024 tokens use <= 32 distinct ids and the 8 commonest ids make >= 95% of them.
// Reasoning tokens and visible tokens have separate windows; the reasoning region is not judged before
// min_reasoning_tokens. State is a plain value: copy it before speculative tokens are verified and assign it back when the
// checkpoint is restored (token_loop_guard<T> saved = g; ... g = saved;).
// Contract of BeeLlama server-loop-guard.cpp/.h (MIT), SOURCE-READ 1-253 and 1-62; no code copied (windowed hash maps and the
// mismatch budget are written here from the contract). SOURCE-NUANCE: BeeLlama forces the reasoning block closed on a hit; this
// header only reports - the slot decides to stop or to hand the hit to reasoning_loop_recovery.
struct loop_guard_params {
    int32_t min_reasoning_tokens  = 512;
    int32_t window_tokens         = 1024;
    int32_t max_period            = 128;
    int32_t min_repeated_coverage = 256;
    int32_t check_interval        = 64;
};

struct loop_guard_result {
    bool        triggered = false;
    std::string kind;             //"periodic_tail" | "ngram_dominance" | "low_entropy"
    int32_t     period    = 0;
    int32_t     coverage  = 0;
    float       score     = 0.0f;
};

template <typename Tok>
class token_loop_guard {
public:
    explicit token_loop_guard(loop_guard_params p = {}) : params(p) {}

    void reset() { reasoning.clear(); visible.clear(); n_reasoning = 0; n_visible = 0; }

    void accept(Tok t, bool in_reasoning) {
        auto & w = in_reasoning ? reasoning : visible;
        w.push_back(t);
        if ((int32_t) w.size() > params.window_tokens) {
            w.erase(w.begin(), w.begin() + ((int32_t) w.size() - params.window_tokens));
        }
        (in_reasoning ? n_reasoning : n_visible)++;
    }

    int32_t seen(bool in_reasoning) const { return in_reasoning ? n_reasoning : n_visible; }

    bool should_check(bool in_reasoning, bool token_is_eog) const {
        if (token_is_eog || params.check_interval < 1) {
            return false;
        }
        const int32_t s = seen(in_reasoning);
        if (in_reasoning && s < params.min_reasoning_tokens) {
            return false;
        }
        return s > 0 && s % params.check_interval == 0;
    }

    loop_guard_result check(bool in_reasoning) const {
        if (in_reasoning && n_reasoning < params.min_reasoning_tokens) {
            return {};
        }
        const auto & w = in_reasoning ? reasoning : visible;
        loop_guard_result r = periodic_tail(w);
        if (r.triggered) {
            return r;
        }
        if (in_reasoning) {
            r = ngram_dominance(w);
            if (r.triggered) {
                return r;
            }
        }
        return low_entropy(w);
    }

private:
    loop_guard_params params;
    std::vector<Tok>  reasoning, visible;
    int32_t           n_reasoning = 0, n_visible = 0;

    loop_guard_result periodic_tail(const std::vector<Tok> & w) const {
        const int32_t n = (int32_t) w.size();
        const int32_t max_p = params.max_period < n / 3 ? params.max_period : n / 3;
        for (int32_t p = 1; p <= max_p; ++p) {
            const int32_t cov = n - p;
            if (cov < params.min_repeated_coverage) {
                continue;
            }
            const float   th  = p <= 8 ? 0.995f : (p <= 64 ? 0.990f : 0.980f);
            const int32_t max_miss = (int32_t) ((1.0f - th) * (float) cov);
            int32_t miss = 0;
            for (int32_t i = p; i < n && miss <= max_miss; ++i) {
                miss += w[i] != w[i - p];
            }
            if (miss <= max_miss) {
                return {true, "periodic_tail", p, cov, (float) (cov - miss) / (float) cov};
            }
        }
        return {};
    }

    static uint64_t gram_hash(const std::vector<Tok> & w, int32_t s, int32_t g) {
        uint64_t h = 1469598103934665603ULL;
        for (int32_t i = 0; i < g; ++i) {
            h = (h ^ ((uint64_t) (uint32_t) w[s + i] + 0x9e3779b97f4a7c15ULL)) * 1099511628211ULL;
        }
        return h;
    }

    loop_guard_result ngram_dominance(const std::vector<Tok> & w) const {
        const int32_t n = (int32_t) w.size();
        if (n < params.min_repeated_coverage) {
            return {};
        }
        for (int32_t g : {8, 16, 32}) {
            if (n < g * 2) {
                continue;
            }
            std::unordered_map<uint64_t, std::vector<int32_t>> by_hash;
            std::vector<uint8_t> covered((size_t) n, 0);
            for (int32_t s = 0; s + g <= n; ++s) {
                auto & starts = by_hash[gram_hash(w, s, g)];
                for (int32_t prev : starts) {
                    bool eq = true;
                    for (int32_t i = 0; eq && i < g; ++i) {
                        eq = w[prev + i] == w[s + i];
                    }
                    if (eq) { // both occurrences count as covered
                        for (int32_t i = 0; i < g; ++i) {
                            covered[(size_t) prev + i] = covered[(size_t) s + i] = 1;
                        }
                        break;
                    }
                }
                starts.push_back(s);
            }
            int32_t cov = 0;
            for (uint8_t c : covered) {
                cov += c;
            }
            if (cov >= params.min_repeated_coverage && cov * 2 >= n) {
                return {true, "ngram_dominance", g, cov, (float) cov / (float) n};
            }
        }
        return {};
    }

    loop_guard_result low_entropy(const std::vector<Tok> & w) const {
        int32_t n_tail = (int32_t) w.size();
        if (n_tail > params.window_tokens) n_tail = params.window_tokens;
        if (n_tail > 1024) n_tail = 1024;
        if (n_tail < 1024 || n_tail < params.min_repeated_coverage) {
            return {};
        }
        std::unordered_map<Tok, int32_t> counts;
        for (size_t i = w.size() - (size_t) n_tail; i < w.size(); ++i) {
            counts[w[i]]++;
        }
        if (counts.size() > 32) {
            return {};
        }
        std::vector<int32_t> c;
        for (const auto & kv : counts) {
            c.push_back(kv.second);
        }
        std::sort(c.begin(), c.end(), [](int32_t a, int32_t b) { return a > b; });
        int32_t top8 = 0;
        for (size_t i = 0; i < c.size() && i < 8; ++i) {
            top8 += c[i];
        }
        const float cov = (float) top8 / (float) n_tail;
        if (cov >= 0.95f) {
            return {true, "low_entropy", 0, top8, cov};
        }
        return {};
    }
};

} // namespace server_repetition
