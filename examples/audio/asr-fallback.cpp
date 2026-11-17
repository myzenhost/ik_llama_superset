// asr-fallback.cpp -- see asr-fallback.h
#include "asr-fallback.h"
#include "asr-compression.h"

#include <cmath>
#include <map>

std::vector<float> asr_temperature_ladder(float t0, float inc) {
    std::vector<float> out;
    if (!(inc > 0.0f)) {
        out.push_back(t0);
        return out;
    }
    for (int i = 0; i < 64; ++i) {
        const double t = (double) t0 + (double) i * (double) inc;
        if (!(t < 1.0 + 1e-6)) {
            break;
        }
        out.push_back((float) t);
    }
    if (out.empty()) { // t0 already > 1
        out.push_back(t0);
    }
    return out;
}

double asr_tail_entropy(const std::vector<int32_t> & tokens, int window) {
    const int n = (int) tokens.size();
    const int from = std::max(0, n - std::max(1, window));
    std::map<int32_t, int> cnt;
    for (int i = from; i < n; ++i) {
        cnt[tokens[i]]++;
    }
    const double total = (double) (n - from);
    double h = 0.0;
    for (const auto & kv : cnt) {
        const double pr = kv.second / total;
        h -= pr * std::log(pr);
    }
    return h;
}

asr_fallback_verdict asr_fallback_check(const asr_fallback_params & p, const asr_decode_attempt & att) {
    asr_fallback_verdict v;
    v.compression_ratio = asr_compression_ratio(att.text);

    if (p.check_compression && v.compression_ratio > p.compression_ratio_thold) {
        v.reasons |= ASR_FAIL_COMPRESSION;
    }
    const bool low_lp = p.check_logprob && att.avg_logprob < p.logprob_thold;
    if (low_lp) {
        v.reasons |= ASR_FAIL_LOGPROB;
    }
    if (p.check_entropy && att.n_tokens > p.entropy_min_tokens && att.tail_entropy < p.entropy_thold) {
        v.reasons |= ASR_FAIL_ENTROPY;
    }
    v.needs_fallback = v.reasons != ASR_FAIL_NONE;

    if (p.check_no_speech && p.check_logprob && att.no_speech_prob > p.no_speech_thold && low_lp) {
        v.is_silence = true;
        v.needs_fallback = false;
    }
    return v;
}

asr_fallback_result asr_run_fallback(const asr_fallback_params & p,
                                     const std::function<asr_decode_attempt(float)> & decode) {
    asr_fallback_result r;
    for (float t : asr_temperature_ladder(p.temperature, p.temperature_inc)) {
        asr_decode_attempt a = decode(t);
        a.temperature = t;
        const asr_fallback_verdict v = asr_fallback_check(p, a);
        r.attempts.push_back(a);
        r.verdicts.push_back(v);
        if (!v.needs_fallback) {
            r.chosen = (int) r.attempts.size() - 1;
            return r;
        }
    }

    r.exhausted = true;
    if (p.on_exhausted == asr_fallback_params::EXHAUSTED_LAST) {
        r.chosen = (int) r.attempts.size() - 1;
        return r;
    }
    // best avg_logprob among attempts under the compression threshold; all if none is
    int best = -1;
    for (int pass = 0; pass < 2 && best < 0; ++pass) {
        for (int i = 0; i < (int) r.attempts.size(); ++i) {
            const bool under = !(r.verdicts[i].reasons & ASR_FAIL_COMPRESSION);
            if (pass == 0 && !under) { continue; }
            if (best < 0 || r.attempts[i].avg_logprob > r.attempts[best].avg_logprob) {
                best = i;
            }
        }
    }
    r.chosen = best;
    return r;
}
