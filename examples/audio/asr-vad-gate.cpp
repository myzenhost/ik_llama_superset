#include "asr-vad-gate.h"

#include <algorithm>
#include <cmath>

std::vector<float> asr_resample_linear(const float * x, size_t n, int32_t sr_in, int32_t sr_out) {
    if (x == nullptr || n == 0 || sr_in <= 0 || sr_out <= 0) {
        return {};
    }
    if (sr_in == sr_out) {
        return std::vector<float>(x, x + n);
    }
    const size_t n_out = (size_t) std::llround((double) n * (double) sr_out / (double) sr_in);
    std::vector<float> y(n_out);
    const double step = (double) sr_in / (double) sr_out;
    for (size_t i = 0; i < n_out; ++i) {
        const double pos = (double) i * step;
        const size_t i0  = std::min((size_t) pos, n - 1);
        const size_t i1  = std::min(i0 + 1, n - 1);
        const double fr  = pos - (double) i0;
        y[i] = (float) ((1.0 - fr) * x[i0] + fr * x[i1]);
    }
    return y;
}

asr_vad_gate_result asr_vad_gate(const asr_silero_model & m, const float * pcm, size_t n_samples, int32_t sample_rate,
                                 asr_vad_post_params pp) {
    asr_vad_gate_result r;
    const std::vector<float> x = asr_resample_linear(pcm, n_samples, sample_rate, 16000);
    r.n_samples_16k = x.size();
    if (x.empty()) {
        return r;   // nothing decoded: no speech
    }
    pp.window_samples = m.n_window;
    pp.sample_rate    = 16000;
    const std::vector<float> probs = asr_silero_probs(m, x.data(), x.size());
    r.n_probs = probs.size();
    for (float p : probs) {
        r.max_prob = std::max(r.max_prob, p);
    }
    r.spans = asr_vad_postprocess(probs.data(), probs.size(), pp);
    for (auto & s : r.spans) {
        s.end = std::min<int64_t>(s.end, (int64_t) x.size());
    }
    r.spans.erase(std::remove_if(r.spans.begin(), r.spans.end(), [](const asr_vad_span & s) { return s.end <= s.start; }), r.spans.end());
    return r;
}
