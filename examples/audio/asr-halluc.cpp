// asr-halluc.cpp -- see asr-halluc.h
#include "asr-halluc.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace {

// faster-whisper: punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"  and the test is Python `word not in punctuation`
// (substring test on the whole string, so "" and multi-char runs that appear in it count too).
const std::string PUNCT = "\"'\xE2\x80\x9C\xC2\xBF([{-\"'.\xE3\x80\x82,\xEF\xBC\x8C!\xEF\xBC\x81?\xEF\xBC\x9F:\xEF\xBC\x9A\xE2\x80\x9D)]}\xE3\x80\x81";

bool is_punct(const std::string & w) { return PUNCT.find(w) != std::string::npos; }

const asr_word_segment * next_words_segment(const std::vector<asr_word_segment> & segs, size_t from) {
    for (size_t i = from; i < segs.size(); ++i) {
        if (!segs[i].words.empty()) {
            return &segs[i];
        }
    }
    return nullptr;
}

} // namespace

double asr_word_anomaly_score(const asr_word & w, double time_resolution) {
    const double dur = w.end - w.start;
    const double dur_short = std::max(dur, time_resolution);   // cannot measure below one step (asr-halluc.h)
    double score = 0.0;
    if (w.probability < 0.15) { score += 1.0; }
    if (dur_short < 0.133)    { score += (0.133 - dur_short) * 15.0; }
    if (dur > 2.0)            { score += dur - 2.0; }
    return score;
}

bool asr_segment_is_anomaly(const asr_word_segment & seg, double time_resolution) {
    if (seg.words.empty()) {
        return false;
    }
    double score = 0.0;
    size_t n = 0;
    for (const auto & w : seg.words) {
        if (is_punct(w.word)) { continue; }
        if (n == 8) { break; }
        score += asr_word_anomaly_score(w, time_resolution);
        n++;
    }
    return score >= 3.0 || score + 0.01 >= (double) n;
}

asr_halluc_decision asr_halluc_check(const asr_halluc_params & p, const std::vector<asr_word_segment> & segs) {
    asr_halluc_decision d;

    // 1) a hallucinated FIRST segment preceded by a long gap: skip the leading silence
    const asr_word_segment * first = next_words_segment(segs, 0);
    if (first && asr_segment_is_anomaly(*first, p.time_resolution)) {
        const double gap = first->start - p.time_offset;
        if (gap > p.threshold) {
            d.action = ASR_HALLUC_SKIP_LEADING;
            d.seek_time = p.time_offset + gap;
            return d;
        }
    }

    // 2) an anomalous segment surrounded by silence (or more anomalies): cut from there
    double hal_last_end = p.last_speech_timestamp;
    for (size_t si = 0; si < segs.size(); ++si) {
        const asr_word_segment & seg = segs[si];
        if (seg.words.empty()) {
            continue;
        }
        if (asr_segment_is_anomaly(seg, p.time_resolution)) {
            const asr_word_segment * next = next_words_segment(segs, si + 1);
            const double hal_next_start = next ? next->words[0].start : p.time_offset + p.segment_duration;

            const bool silence_before =
                seg.start - hal_last_end > p.threshold ||
                seg.start < p.threshold ||
                seg.start - p.time_offset < 2.0;
            const bool silence_after =
                hal_next_start - seg.end > p.threshold ||
                (next && asr_segment_is_anomaly(*next, p.time_resolution)) ||
                p.window_end_time - seg.end < 2.0;

            if (silence_before && silence_after) {
                d.action    = ASR_HALLUC_TRUNCATE;
                d.n_keep    = si;
                d.seek_time = std::max(p.time_offset + 1.0, seg.start);
                if (p.content_duration - seg.end < p.threshold) {
                    d.seek_to_end = true;
                    d.seek_time   = p.content_duration;
                }
                return d;
            }
        }
        hal_last_end = seg.end;
    }
    return d;
}


// ---- segments from words + the decode loop (see asr-halluc.h) ------------------------------------------------

static bool ends_sentence(const std::string & w) {
    static const char * enders[] = { ".", "!", "?", "\xE3\x80\x82", "\xEF\xBC\x81", "\xEF\xBC\x9F" };
    std::string s = w;
    // closing quotes/brackets after the stop still end the sentence: "Go." / (yes.)
    while (!s.empty() && (s.back() == '"' || s.back() == '\'' || s.back() == ')' || s.back() == ']')) { s.pop_back(); }
    for (const char * e : enders) {
        const size_t n = strlen(e);
        if (s.size() >= n && s.compare(s.size() - n, n, e) == 0) {
            return true;
        }
    }
    return false;
}

std::vector<asr_word_segment> asr_words_to_segments(const std::vector<asr_word> & words, double split_gap_s) {
    std::vector<asr_word_segment> segs;
    bool open = false;
    for (size_t i = 0; i < words.size(); ++i) {
        const asr_word & w = words[i];
        if (open && split_gap_s > 0.0 && w.start - segs.back().end > split_gap_s) {
            open = false;
        }
        if (!open) {
            segs.emplace_back();
            segs.back().start = w.start;
            open = true;
        }
        segs.back().words.push_back(w);
        segs.back().end = std::max(segs.back().end, w.end);
        if (ends_sentence(w.word)) {
            open = false;
        }
    }
    return segs;
}

bool asr_halluc_decode_span(double t0, double t1, double threshold, double content_duration,
                            double last_speech_timestamp, double split_gap_s, int max_decodes,
                            const asr_decode_span_fn & decode, asr_halluc_span_result & out,
                            double time_resolution) {
    out = {};
    out.last_speech = last_speech_timestamp;
    double pos = t0;
    char buf[256];
    while (pos < t1 - 1e-3) {
        if (out.n_decodes >= max_decodes) {
            snprintf(buf, sizeof(buf), "%.3f-%.3f: decode limit %d reached, rest of the span not decoded", pos, t1, max_decodes);
            out.trace.push_back(buf);
            break;
        }
        std::vector<asr_word> words;
        if (!decode(pos, t1, words)) {
            return false;
        }
        out.n_decodes++;
        const std::vector<asr_word_segment> segs = asr_words_to_segments(words, split_gap_s);

        asr_halluc_params p;
        p.threshold             = threshold;
        p.time_offset           = pos;
        p.segment_duration      = t1 - pos;
        p.window_end_time       = t1;
        p.last_speech_timestamp = out.last_speech;
        p.content_duration      = content_duration;
        p.time_resolution       = time_resolution;
        const asr_halluc_decision d = asr_halluc_check(p, segs);

        if (d.action == ASR_HALLUC_KEEP) {
            snprintf(buf, sizeof(buf), "%.3f-%.3f: %zu word(s) in %zu segment(s) -> keep", pos, t1, words.size(), segs.size());
            out.trace.push_back(buf);
            for (const auto & s : segs) { out.kept.push_back(s); }
            if (!out.kept.empty()) { out.last_speech = out.kept.back().end; }
            break;
        }
        if (d.action == ASR_HALLUC_SKIP_LEADING) {
            out.n_skip_leading++;
            snprintf(buf, sizeof(buf), "%.3f-%.3f: first segment anomalous after %.3f s of silence -> skip to %.3f", pos, t1,
                     d.seek_time - pos, d.seek_time);
            out.trace.push_back(buf);
            if (!(d.seek_time > pos + 1e-3)) { break; }   // cannot happen (gap > threshold > 0); never loop in place
            pos = d.seek_time;
            continue;
        }
        // TRUNCATE
        out.n_truncate++;
        snprintf(buf, sizeof(buf), "%.3f-%.3f: segment %zu of %zu anomalous and surrounded by silence -> keep %zu, %s %.3f", pos, t1,
                 d.n_keep, segs.size(), d.n_keep, d.seek_to_end ? "stop at" : "resume at", d.seek_time);
        out.trace.push_back(buf);
        for (size_t k = 0; k < d.n_keep && k < segs.size(); ++k) { out.kept.push_back(segs[k]); }
        if (!out.kept.empty()) { out.last_speech = std::max(out.last_speech, out.kept.back().end); }
        if (d.seek_to_end || d.seek_time >= t1 - 1e-3) { break; }
        pos = std::max(d.seek_time, pos + 1.0);
    }
    return true;
}
