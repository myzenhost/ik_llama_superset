// asr-vad-post.cpp -- see asr-vad-post.h
#include "asr-vad-post.h"

#include <algorithm>

std::vector<asr_vad_span> asr_vad_postprocess(const float * probs, size_t n_probs, const asr_vad_post_params & p) {
    std::vector<asr_vad_span> out;
    if (!probs || n_probs == 0 || p.window_samples <= 0 || p.sample_rate <= 0) {
        return out;
    }

    const int64_t sr   = p.sample_rate;
    const int64_t win  = p.window_samples;
    const int64_t pad  = sr * p.speech_pad_ms / 1000;
    const int64_t min_speech  = sr * p.min_speech_ms / 1000;
    const int64_t min_silence = sr * p.min_silence_ms / 1000;
    const int64_t audio_len   = (int64_t) n_probs * win;
    const int64_t HUGE_SAMPLES = INT64_C(1) << 40;

    int64_t max_speech = HUGE_SAMPLES;
    if (p.max_speech_s <= 100000.0f) {
        const int64_t t = sr * (int64_t) p.max_speech_s - win - 2 * pad;  // whole seconds, as upstream
        max_speech = t < 0 ? HUGE_SAMPLES : t;
    }
    const int64_t min_silence_at_max = sr * 98 / 1000; // Silero constant
    const float   neg_threshold = std::max(p.threshold - 0.15f, 0.01f);

    std::vector<asr_vad_span> sp;

    bool    in_speech  = false;
    bool    has_speech = false;
    int64_t temp_end = 0, prev_end = 0, next_start = 0, cur_start = 0;

    for (size_t i = 0; i < n_probs; ++i) {
        const float   pr = probs[i];
        const int64_t cs = (int64_t) i * win;

        if (pr >= p.threshold && temp_end) {
            temp_end = 0;
            if (next_start < prev_end) {
                next_start = cs;
            }
        }
        if (pr >= p.threshold && !in_speech) {
            in_speech  = true;
            cur_start  = cs;
            has_speech = true;
            continue;
        }
        if (in_speech && cs - cur_start > max_speech) {
            if (prev_end) {
                sp.push_back({cur_start, prev_end});
                has_speech = true;
                if (next_start < prev_end) {
                    in_speech  = false;
                    has_speech = false;
                } else {
                    cur_start = next_start;
                }
                prev_end = next_start = temp_end = 0;
            } else {
                sp.push_back({cur_start, cs});
                prev_end = next_start = temp_end = 0;
                in_speech  = false;
                has_speech = false;
                continue;
            }
        }
        if (pr < neg_threshold && in_speech) {
            if (!temp_end) {
                temp_end = cs;
            }
            if (cs - temp_end > min_silence_at_max) {
                prev_end = temp_end;
            }
            if (cs - temp_end < min_silence) {
                continue;
            }
            if (temp_end - cur_start > min_speech) {
                sp.push_back({cur_start, temp_end});
            }
            prev_end = next_start = temp_end = 0;
            in_speech  = false;
            has_speech = false;
            continue;
        }
    }
    if (has_speech && audio_len - cur_start > min_speech) {
        sp.push_back({cur_start, audio_len});
    }

    // merge neighbours closer than merge_gap_ms
    if (p.merge_gap_ms > 0) {
        const int64_t gap = sr * p.merge_gap_ms / 1000;
        for (size_t i = 0; i + 1 < sp.size();) {
            if (sp[i + 1].start - sp[i].end < gap) {
                sp[i].end = sp[i + 1].end;
                sp.erase(sp.begin() + (long) i + 1);
            } else {
                ++i;
            }
        }
    }

    // final minimum-duration filter
    sp.erase(std::remove_if(sp.begin(), sp.end(), [&](const asr_vad_span & s) { return s.end - s.start < min_speech; }), sp.end());

    // padding: first start / last end get full pad; between neighbours split the gap if it is
    // smaller than 2*pad, else full pad on both sides
    for (size_t i = 0; i < sp.size(); ++i) {
        if (i == 0) {
            sp[i].start = sp[i].start > pad ? sp[i].start - pad : 0;
        }
        if (i + 1 < sp.size()) {
            const int64_t sil = sp[i + 1].start - sp[i].end;
            if (sil < 2 * pad) {
                sp[i].end += sil / 2;
                sp[i + 1].start = sp[i + 1].start > sil / 2 ? sp[i + 1].start - sil / 2 : 0;
            } else {
                sp[i].end = sp[i].end + pad < audio_len ? sp[i].end + pad : audio_len;
                sp[i + 1].start = sp[i + 1].start > pad ? sp[i + 1].start - pad : 0;
            }
        } else {
            sp[i].end = sp[i].end + pad < audio_len ? sp[i].end + pad : audio_len;
        }
    }
    return sp;
}
