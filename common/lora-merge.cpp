#include "lora-merge.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

static std::string lower(std::string s) {
    for (auto & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

bool lora_merge_strategy_from_name(const std::string & name, lora_merge_strategy & out) {
    const std::string n = lower(name);
    if (n == "linear")      { out = LORA_MERGE_LINEAR;      return true; }
    if (n == "ties")        { out = LORA_MERGE_TIES;        return true; }
    if (n == "dare_linear") { out = LORA_MERGE_DARE_LINEAR; return true; }
    if (n == "dare_ties")   { out = LORA_MERGE_DARE_TIES;   return true; }
    return false;
}

bool lora_sign_method_from_name(const std::string & name, lora_sign_method & out) {
    const std::string n = lower(name);
    if (n == "total")     { out = LORA_SIGN_TOTAL;     return true; }
    if (n == "frequency") { out = LORA_SIGN_FREQUENCY; return true; }
    return false;
}

const char * lora_merge_strategy_name(lora_merge_strategy s) {
    switch (s) {
        case LORA_MERGE_LINEAR:      return "linear";
        case LORA_MERGE_TIES:        return "ties";
        case LORA_MERGE_DARE_LINEAR: return "dare_linear";
        case LORA_MERGE_DARE_TIES:   return "dare_ties";
    }
    return "?";
}

// keep the floor(density * numel) largest-|x| entries; the others become 0. Equal magnitudes at the cut: lowest index first.
static void prune_by_magnitude(const float * in, size_t numel, float density, float * out) {
    const size_t k = (size_t) std::floor((double) density * (double) numel);
    if (k >= numel) {
        std::copy(in, in + numel, out);
        return;
    }
    if (k == 0) {
        std::fill(out, out + numel, 0.0f);
        return;
    }
    std::vector<float> mag(numel);
    for (size_t i = 0; i < numel; ++i) {
        mag[i] = std::fabs(in[i]);
    }
    std::vector<float> sorted(mag);
    std::nth_element(sorted.begin(), sorted.begin() + (k - 1), sorted.end(), std::greater<float>());
    const float thr = sorted[k - 1];                    // the k-th largest magnitude
    size_t n_above = 0;
    for (size_t i = 0; i < numel; ++i) {
        n_above += mag[i] > thr;
    }
    size_t n_equal_keep = k - n_above;                  // how many of the entries equal to thr still fit (>= 1)
    for (size_t i = 0; i < numel; ++i) {
        if (mag[i] > thr) {
            out[i] = in[i];
        } else if (mag[i] == thr && n_equal_keep > 0) {
            out[i] = in[i];
            --n_equal_keep;
        } else {
            out[i] = 0.0f;
        }
    }
}

// keep each entry with probability density and rescale the survivors by 1/density (expectation preserved)
static void prune_random_rescaled(const float * in, size_t numel, float density, std::mt19937_64 & rng, float * out) {
    if (density >= 1.0f) {
        std::copy(in, in + numel, out);
        return;
    }
    std::bernoulli_distribution keep((double) density);
    const float inv = density > 0.0f ? 1.0f / density : 0.0f;
    for (size_t i = 0; i < numel; ++i) {
        out[i] = keep(rng) ? in[i] * inv : 0.0f;
    }
}

bool lora_merge_deltas(const lora_merge_params & p, const float * const * deltas, const float * weights, int n, size_t numel,
                       float * out, std::string * err) {
    auto fail = [&](const std::string & msg) {
        if (err) {
            *err = msg;
        }
        return false;
    };
    if (n < 1 || deltas == nullptr || out == nullptr) {
        return fail("lora merge: need at least one delta and an output buffer");
    }
    for (int i = 0; i < n; ++i) {
        if (deltas[i] == nullptr) {
            return fail("lora merge: null delta " + std::to_string(i));
        }
    }
    if (!(p.density >= 0.0f && p.density <= 1.0f)) {
        return fail("lora merge: density must be in [0, 1], got " + std::to_string(p.density));
    }
    auto w = [&](int i) { return weights ? weights[i] : 1.0f; };

    if (p.strategy == LORA_MERGE_LINEAR) {
        for (size_t j = 0; j < numel; ++j) {
            float s = 0.0f;
            for (int i = 0; i < n; ++i) {
                s += w(i) * deltas[i][j];
            }
            out[j] = s;
        }
        return true;
    }

    // sparsify every delta into its own buffer
    std::vector<std::vector<float>> pr(n, std::vector<float>(numel));
    std::mt19937_64 rng(p.seed);
    for (int i = 0; i < n; ++i) {
        if (p.strategy == LORA_MERGE_TIES) {
            prune_by_magnitude(deltas[i], numel, p.density, pr[i].data());
        } else {
            prune_random_rescaled(deltas[i], numel, p.density, rng, pr[i].data());
        }
    }

    if (p.strategy == LORA_MERGE_DARE_LINEAR) {
        for (size_t j = 0; j < numel; ++j) {
            float s = 0.0f;
            for (int i = 0; i < n; ++i) {
                s += w(i) * pr[i][j];
            }
            out[j] = s;
        }
        return true;
    }

    // TIES and DARE-TIES: elect the sign on the sparsified, unweighted values, then merge the agreeing ones
    for (size_t j = 0; j < numel; ++j) {
        double mag = 0.0;
        for (int i = 0; i < n; ++i) {
            const float v = pr[i][j];
            mag += p.sign_method == LORA_SIGN_TOTAL ? (double) v : (double) ((v > 0.0f) - (v < 0.0f));
        }
        const int majority = mag >= 0.0 ? 1 : -1;
        float num = 0.0f;
        int   cnt = 0;
        for (int i = 0; i < n; ++i) {
            const float v = pr[i][j];
            const int   s = (v > 0.0f) - (v < 0.0f);
            if (s == majority) {
                num += w(i) * v;
                ++cnt;
            }
        }
        out[j] = num / (float) std::max(cnt, 1);
    }
    return true;
}

bool lora_delta_dense(const float * A, const float * B, int rank, int n_in, int n_out, float scale, float * out) {
    if (!A || !B || !out || rank < 1 || n_in < 1 || n_out < 1) {
        return false;
    }
    for (int o = 0; o < n_out; ++o) {
        for (int i = 0; i < n_in; ++i) {
            float s = 0.0f;
            for (int k = 0; k < rank; ++k) {
                s += B[(size_t) o * rank + k] * A[(size_t) k * n_in + i];
            }
            out[(size_t) o * n_in + i] = scale * s;
        }
    }
    return true;
}
