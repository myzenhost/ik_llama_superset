// asr-align.cpp -- see asr-align.h
#include "asr-align.h"
#include "asr-dtw.h"

#include <algorithm>
#include <cmath>
#include <cstring>

bool asr_align_is_punct(const std::string & piece) {
    for (unsigned char c : piece) {
        if (c >= 0x80 || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            return false;
        }
    }
    return true;
}

static bool starts_with_space(const std::string & s) {
    return !s.empty() && (s[0] == ' ' || s[0] == '\n' || s[0] == '\t');
}

std::vector<std::pair<int, int>> asr_align_group_words(const std::vector<std::string> & pieces) {
    std::vector<std::pair<int, int>> words;
    for (int i = 0; i < (int) pieces.size(); ++i) {
        const bool punct = asr_align_is_punct(pieces[i]);
        if (words.empty()) {
            // leading punctuation / whitespace before the first real token belongs to nothing yet:
            // it opens the first word so no token is ever dropped
            words.emplace_back(i, i + 1);
            continue;
        }
        if (punct || !starts_with_space(pieces[i])) {
            // continuation of the current word (sub-word piece or attached punctuation)
            words.back().second = i + 1;
            continue;
        }
        // a word made only of leading punctuation so far takes this token as its real start
        bool open_is_punct = true;
        for (int k = words.back().first; k < words.back().second; ++k) {
            if (!asr_align_is_punct(pieces[k])) { open_is_punct = false; break; }
        }
        if (open_is_punct) {
            words.back().second = i + 1;
        } else {
            words.emplace_back(i, i + 1);
        }
    }
    return words;
}

double asr_align_seconds_per_position(double a_s, int n_a, double b_s, int n_b, double chunk_s) {
    if (!(b_s > a_s) || n_a <= 0 || n_b <= 0 || n_b < n_a) {
        return 0.0;
    }
    if (n_b > n_a) {
        return (b_s - a_s) / (double) (n_b - n_a);
    }
    return chunk_s > 0.0 ? chunk_s / (double) n_a : 0.0;
}

int asr_align_valid_frames(int n_positions, double duration_s, double sec_per_frame) {
    if (n_positions <= 0 || !(sec_per_frame > 0.0) || !(duration_s > 0.0)) {
        return 0;
    }
    const int need = (int) std::ceil(duration_s / sec_per_frame - 1e-9);
    return std::max(1, std::min(n_positions, need));
}

bool asr_align_words(const asr_align_input & in, asr_align_result & out, std::string * err) {
    auto fail = [&](const char * m) { if (err) { *err = m; } return false; };
    out = {};
    const int n_tok  = (int) in.tokens.size();
    const int n_rows = n_tok + (in.has_eog_row ? 1 : 0);
    if (n_tok <= 0)                      { return fail("no tokens"); }
    if (in.n_heads <= 0)                 { return fail("no heads"); }
    if (in.n_frames < 2)                 { return fail("fewer than 2 audio frames"); }
    if (!(in.sec_per_frame > 0.0))       { return fail("seconds per frame <= 0"); }
    if (in.attn.size() != (size_t) n_rows * in.n_heads * in.n_frames) { return fail("attention size mismatch"); }
    if (!in.speech_w.empty() && in.speech_w.size() != (size_t) in.n_frames) { return fail("speech weight size mismatch"); }
    const float * sw = in.speech_w.empty() ? nullptr : in.speech_w.data();

    // [rows][heads][frames] -> [heads][rows][frames] (the layout asr_dtw_token_frames takes)
    const size_t F = (size_t) in.n_frames;
    std::vector<float> hw((size_t) in.n_heads * n_rows * F);
    for (int r = 0; r < n_rows; ++r) {
        for (int h = 0; h < in.n_heads; ++h) {
            const float * src = in.attn.data() + ((size_t) r * in.n_heads + h) * F;
            std::copy(src, src + F, hw.data() + ((size_t) h * n_rows + r) * F);
        }
    }
    int width = in.medfilt_width;
    if (width >= in.n_frames) { width = (in.n_frames - 1) | 1; if (width >= in.n_frames) { width -= 2; } }
    if (width < 1) { width = 1; }
    const std::vector<int> start = asr_dtw_token_frames(hw.data(), in.n_heads, n_rows, in.n_frames, width, sw);
    if ((int) start.size() != n_rows) { return fail("dtw failed"); }

    auto at = [&](double frame) {
        const double t = in.t_offset + frame * in.sec_per_frame;
        return std::min(std::max(t, in.t_offset), in.t_end > in.t_offset ? in.t_end : t);
    };
    out.tok_t0.resize(n_tok);
    out.tok_t1.resize(n_tok);
    for (int i = 0; i < n_tok; ++i) {
        out.tok_t0[i] = at(start[i]);
        out.tok_t1[i] = (i + 1 < n_rows) ? at(start[i + 1]) : (in.t_end > in.t_offset ? in.t_end : at(in.n_frames));
        if (out.tok_t1[i] < out.tok_t0[i]) { out.tok_t1[i] = out.tok_t0[i]; }
    }

    std::vector<std::string> pieces(n_tok);
    for (int i = 0; i < n_tok; ++i) { pieces[i] = in.tokens[i].piece; }
    for (const auto & wr : asr_align_group_words(pieces)) {
        asr_word w;
        double psum = 0.0;
        int    last_real = -1;
        for (int k = wr.first; k < wr.second; ++k) {
            w.word += pieces[k];
            psum   += in.tokens[k].prob;
            if (!asr_align_is_punct(pieces[k])) { last_real = k; }
        }
        // trailing punctuation does not stretch a word: it ends with its last real token (whisper.cpp does
        // the same for TDT: punctuation collapses onto the previous token's end)
        const int first_real = [&] { for (int k = wr.first; k < wr.second; ++k) { if (!asr_align_is_punct(pieces[k])) { return k; } } return wr.first; }();
        w.start = out.tok_t0[first_real];
        w.end   = out.tok_t1[last_real >= 0 ? last_real : wr.second - 1];
        if (sw) {
            // a word ends where its speech ends: positions weighted <= 0.5 at its edges (silence the DTW path had
            // to give to SOME row) are cut off. A word with no speech position at all keeps its DTW span.
            const int f0 = start[first_real];
            const int lr = last_real >= 0 ? last_real : wr.second - 1;
            const int f1 = (lr + 1 < n_rows) ? start[lr + 1] : in.n_frames;
            int a = -1, b = -1;
            for (int f = std::max(0, f0); f < std::min(f1, in.n_frames); ++f) {
                if (sw[f] > 0.5f) { if (a < 0) { a = f; } b = f + 1; }
            }
            if (a >= 0) {
                w.start = at(a);
                w.end   = std::max(w.start, at(b));
            }
        }
        w.probability = psum / (double) (wr.second - wr.first);
        size_t b = 0;
        while (b < w.word.size() && (w.word[b] == ' ' || w.word[b] == '\n' || w.word[b] == '\t')) { ++b; }
        w.word = w.word.substr(b);
        if (w.word.empty()) { continue; }
        out.words.push_back(std::move(w));
    }
    asr_align_truncate_long_words(out.words);
    return true;
}

static bool ends_sentence_mark(const std::string & w) {
    static const char * marks[] = { ".", "!", "?", "\xE3\x80\x82", "\xEF\xBC\x81", "\xEF\xBC\x9F" };
    for (const char * m : marks) {
        const size_t n = strlen(m);
        if (w.size() >= n && w.compare(w.size() - n, n, m) == 0) { return true; }
    }
    return false;
}

void asr_align_truncate_long_words(std::vector<asr_word> & words) {
    // SOURCE-READ: SYSTRAN/faster-whisper ed9a06cd, faster_whisper/transcribe.py:1593-1616 (add_word_timestamps; the
    // same hack is in openai/whisper timing.py). MIT. Written independently, no code copied.
    // (BENCH-PENDING: n/a -- "faster-whisper" is the project's name; no speed is claimed here.)
    // Median of the non-zero word durations, capped at 0.7 s; a word longer than twice that, right after a sentence
    // end, starts max_duration before its end.
    std::vector<double> d;
    for (const auto & w : words) {
        const double x = w.end - w.start;
        if (x != 0.0) { d.push_back(x); }
    }
    if (d.empty()) {
        return;
    }
    std::sort(d.begin(), d.end());
    const size_t n = d.size();
    double median = (n % 2) ? d[n / 2] : 0.5 * (d[n / 2 - 1] + d[n / 2]);
    median = std::min(0.7, median);
    const double max_duration = 2.0 * median;
    for (size_t i = 1; i < words.size(); ++i) {
        if (words[i].end - words[i].start > max_duration && ends_sentence_mark(words[i - 1].word)) {
            // in the source a lone punctuation "word" is first truncated from its START; here punctuation is
            // already merged into the word before it and carries no time of its own (asr_align_words), so only the
            // word that follows the sentence end can hold the pause
            words[i].start = words[i].end - max_duration;
        }
    }
}

static double percentile_sorted(const std::vector<double> & s, double p) {
    // linear interpolation between closest ranks (numpy's default)
    if (s.empty()) { return 0.0; }
    const double pos = p / 100.0 * (double) (s.size() - 1);
    const size_t lo  = (size_t) std::floor(pos);
    const size_t hi  = std::min(s.size() - 1, lo + 1);
    return s[lo] + (s[hi] - s[lo]) * (pos - (double) lo);
}

std::vector<float> asr_align_speech_weights(const float * pcm, size_t n_samples, int sample_rate, double sec_per_frame,
                                            int n_frames) {
    std::vector<float> w;
    if (!pcm || n_samples == 0 || sample_rate <= 0 || !(sec_per_frame > 0.0) || n_frames <= 0) {
        return w;
    }
    std::vector<double> db((size_t) n_frames, -120.0);
    for (int f = 0; f < n_frames; ++f) {
        const size_t i0 = (size_t) std::llround((double) f * sec_per_frame * sample_rate);
        const size_t i1 = std::min(n_samples, (size_t) std::llround((double) (f + 1) * sec_per_frame * sample_rate));
        if (i0 >= i1) { continue; }
        double e = 0.0;
        for (size_t i = i0; i < i1; ++i) { e += (double) pcm[i] * pcm[i]; }
        db[f] = 10.0 * std::log10(e / (double) (i1 - i0) + 1e-12);
    }
    std::vector<double> s = db;
    std::sort(s.begin(), s.end());
    const double ref   = percentile_sorted(s, 95.0);
    const double floor = percentile_sorted(s, 10.0);
    const double thr   = (ref - floor > 20.0) ? floor + (ref - floor) * 0.33 : ref - 40.0;
    const double half  = std::max(1e-6, (ref - thr) * 0.5);
    w.resize((size_t) n_frames);
    for (int f = 0; f < n_frames; ++f) {
        w[f] = (float) std::min(1.0, std::max(0.0, (db[f] - thr) / half));
    }
    return w;
}
