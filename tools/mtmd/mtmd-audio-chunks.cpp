// mtmd-audio-chunks.cpp -- see mtmd-audio-chunks.h

#include "mtmd-audio-chunks.h"

#include <algorithm>
#include <cmath>

std::vector<std::pair<size_t, size_t>> mtmd_audio_plan_chunks(const float * samples, size_t n_samples,
                                                              const mtmd_audio_chunk_params & p) {
    std::vector<std::pair<size_t, size_t>> out;
    if (n_samples == 0) {
        return out;
    }
    if (p.max_samples == 0 || n_samples <= p.max_samples) {
        out.emplace_back(0, n_samples);
        return out;
    }
    const size_t fs     = std::max<size_t>(1, p.frame_samples);
    const int    sm     = std::max(1, p.smooth_frames);
    const size_t search = std::min(p.search_samples, p.max_samples > fs ? p.max_samples - fs : 0);

    // per-frame power, then the mean power of every run of `sm` frames starting at frame f
    const size_t n_frames = n_samples / fs;
    std::vector<double> pw(n_frames, 0.0);
    for (size_t f = 0; f < n_frames; ++f) {
        double s = 0.0;
        const float * x = samples ? samples + f * fs : nullptr;
        for (size_t i = 0; x && i < fs; ++i) { s += (double) x[i] * x[i]; }
        pw[f] = s / (double) fs;
    }
    std::vector<double> run(n_frames, 0.0); // run[f] = mean of pw[f .. f+sm)
    {
        double acc = 0.0;
        for (size_t f = 0; f < n_frames; ++f) {
            acc += pw[f];
            if (f >= (size_t) sm) { acc -= pw[f - sm]; }
            if (f + 1 >= (size_t) sm) { run[f + 1 - sm] = acc / sm; }
        }
    }
    auto db = [](double x) { return 10.0 * std::log10(x + 1e-12); };

    size_t start = 0;
    while (n_samples - start > p.max_samples) {
        const size_t limit = start + p.max_samples;          // the cut may not pass this
        size_t cut = limit;                                  // hard cut when nothing better is found (whisper.cpp does the same)
        if (search > 0 && samples != nullptr) {
            // candidate runs lie wholly inside [limit - search, limit); the cut goes to the middle of the chosen run
            const size_t lo_s = limit - search;
            size_t f_lo = (lo_s + fs - 1) / fs;
            size_t f_hi = limit / fs;                        // exclusive end frame of the candidate region
            if (f_hi >= (size_t) sm && f_hi <= n_frames && f_hi - (size_t) sm >= f_lo) {
                const size_t f_last = f_hi - sm;             // last run start whose run ends by f_hi
                double best = run[f_lo];
                for (size_t f = f_lo; f <= f_last; ++f) { best = std::min(best, run[f]); }
                const double lim_db = db(best) + p.tie_db;
                size_t pick = f_lo;
                for (size_t f = f_last + 1; f-- > f_lo;) {   // latest run within tie_db of the quietest
                    if (db(run[f]) <= lim_db) { pick = f; break; }
                }
                cut = (pick + (size_t) sm / 2) * fs;
                if (cut <= start || cut > limit) { cut = limit; }
            }
        }
        out.emplace_back(start, cut);
        start = cut;
    }
    out.emplace_back(start, n_samples);
    return out;
}
