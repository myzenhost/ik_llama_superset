// Gumbel-max text watermark -- see watermark.h for the scheme and the source it was reimplemented from.

#include "watermark.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <utility>

namespace common_watermark {

// Philox4x32-10, Salmon et al. 2011 (https://doi.org/10.1145/2063384.2063405): the published round
// multipliers and Weyl key increments.
static constexpr uint32_t PHILOX_M0 = 0xD2511F53u;
static constexpr uint32_t PHILOX_M1 = 0xCD9E8D57u;
static constexpr uint32_t PHILOX_W0 = 0x9E3779B9u;
static constexpr uint32_t PHILOX_W1 = 0xBB67AE85u;
// Domain separators: ASCII "WMCX" seeds the context absorption, "WMTK" tags the per-token call.
static constexpr uint32_t CONTEXT_DOMAIN = 0x574D4358u;
static constexpr uint32_t TOKEN_DOMAIN   = 0x574D544Bu;

static inline void mulhilo(uint32_t a, uint32_t b, uint32_t & hi, uint32_t & lo) {
    const uint64_t p = (uint64_t) a * (uint64_t) b;
    hi = (uint32_t) (p >> 32);
    lo = (uint32_t) p;
}

static void philox4x32_10(uint32_t c[4], uint32_t k0, uint32_t k1) {
    for (int r = 0; r < 10; ++r) {
        uint32_t hi0, lo0, hi1, lo1;
        mulhilo(PHILOX_M0, c[0], hi0, lo0);
        mulhilo(PHILOX_M1, c[2], hi1, lo1);
        const uint32_t n0 = hi1 ^ c[1] ^ k0;
        const uint32_t n2 = hi0 ^ c[3] ^ k1;
        c[0] = n0;
        c[1] = lo1;
        c[2] = n2;
        c[3] = lo0;
        if (r != 9) {
            k0 += PHILOX_W0;
            k1 += PHILOX_W1;
        }
    }
}

context_state absorb(uint64_t key, const int32_t * context, int width) {
    context_state st;
    st.key0 = (uint32_t) (key & 0xFFFFFFFFu);
    st.key1 = (uint32_t) (key >> 32);

    uint32_t s[4] = { CONTEXT_DOMAIN, (uint32_t) width, 0u, 0u };
    for (int offset = 0; offset < width; offset += 4) {
        uint32_t c[4];
        for (int i = 0; i < 4; ++i) {
            const int pos = offset + i;
            // A token id is absorbed as its low 32 bits, so the -1 padding becomes 0xFFFFFFFF.
            // Lanes past the end of the context get a distinct filler per lane.
            const uint32_t w = pos < width ? (uint32_t) context[pos] : 0xFFFFFFFFu - (uint32_t) i;
            c[i] = s[i] ^ w;
        }
        philox4x32_10(c, st.key0 ^ (uint32_t) offset, st.key1);
        for (int i = 0; i < 4; ++i) {
            s[i] = c[i];
        }
    }
    for (int i = 0; i < 4; ++i) {
        st.s[i] = s[i];
    }
    return st;
}

float uniform(const context_state & st, int32_t token) {
    const uint32_t tw = (uint32_t) token;
    // One Philox call yields 4 words; token ids sharing tw >> 2 share the call and pick word tw & 3.
    uint32_t c[4] = { tw >> 2, st.s[0], st.s[1], st.s[2] };
    philox4x32_10(c, st.key0 ^ st.s[3], st.key1 ^ TOKEN_DOMAIN);
    const uint32_t word = c[tw & 3u];
    // Top 24 bits -> (0,1), open at both ends; computed in double then rounded to float, which is the
    // exact value the reference produces.
    const double m = (double) (word >> 8);
    return (float) ((m + 1.0) / (16777216.0 + 1.0));
}

float uniform(uint64_t key, const int32_t * context, int width, int32_t token) {
    return uniform(absorb(key, context, width), token);
}

void next_context(const std::vector<int32_t> & generated, int width, int32_t * out) {
    const int64_t n = (int64_t) generated.size();
    for (int j = 0; j < width; ++j) {
        const int64_t p = n - width + j;
        out[j] = p >= 0 ? generated[(size_t) p] : -1;
    }
}

bool next_context_repeated(const std::vector<int32_t> & generated, int width, int32_t max_history) {
    const int64_t n = (int64_t) generated.size();
    if (n == 0) {
        return false;
    }
    const int64_t start = max_history > 0 ? std::max<int64_t>(0, n - max_history) : 0;
    auto tok_at = [&](int64_t p) -> int32_t { return p >= 0 ? generated[(size_t) p] : -1; };
    // The window before position p is tokens [p - width, p); the current one is the window before n.
    for (int64_t p = start; p < n; ++p) {
        bool same = true;
        for (int j = 0; j < width && same; ++j) {
            same = tok_at(p - width + j) == tok_at(n - width + j);
        }
        if (same) {
            return true;
        }
    }
    return false;
}

int64_t gumbel_select(const context_state & st, const int32_t * ids, const float * logits, size_t n) {
    int64_t best     = -1;
    float   best_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < n; ++i) {
        if (std::isinf(logits[i]) && logits[i] < 0) {
            continue;
        }
        const float u = std::max(uniform(st, ids[i]), FLT_MIN);
        const float g = -std::log(-std::log(u));
        const float v = logits[i] + g;
        if (best < 0 || v > best_val) {
            best     = (int64_t) i;
            best_val = v;
        }
    }
    return best;
}

double gamma_survival_int(double x, int64_t shape) {
    if (shape <= 0 || x <= 0.0) {
        return 1.0;
    }
    // P(X > x) = e^-x * sum_{i<shape} x^i / i!, summed in log space so large shapes do not overflow.
    const double lx = std::log(x);
    std::vector<double> terms((size_t) shape);
    double t   = 0.0;
    double mx  = 0.0;
    terms[0]   = 0.0;
    for (int64_t i = 1; i < shape; ++i) {
        t += lx - std::log((double) i);
        terms[(size_t) i] = t;
        mx = std::max(mx, t);
    }
    double acc = 0.0;
    for (double v : terms) {
        acc += std::exp(v - mx);
    }
    const double log_sum = mx + std::log(acc);
    return std::min(1.0, std::exp(std::min(0.0, -x + log_sum)));
}

detection detect(uint64_t key, const std::vector<int32_t> & tokens, int width, bool dedup, double p_threshold) {
    detection d;
    std::vector<int32_t> prefix((size_t) width, -1);
    std::set<std::vector<int32_t>> seen;
    std::vector<int32_t> ctx((size_t) width);
    for (int32_t tok : tokens) {
        std::copy(prefix.end() - width, prefix.end(), ctx.begin());
        bool repeated = false;
        if (dedup) {
            repeated = !seen.insert(ctx).second;
        }
        if (!repeated) {
            const double u = (double) uniform(key, ctx.data(), width, tok);
            d.score += -std::log1p(-u);
            d.n_scored++;
        }
        prefix.push_back(tok);
    }
    if (d.n_scored == 0) {
        return d;
    }
    d.p_value        = gamma_survival_int(d.score, d.n_scored);
    d.is_watermarked = d.p_value <= p_threshold;
    return d;
}

// ---- KGW green list and SynthID tournament (see watermark.h for sources and departures) -------------

const char * scheme_name(int32_t s) {
    switch (s) {
        case SCHEME_GUMBEL:  return "gumbel";
        case SCHEME_KGW:     return "kgw";
        case SCHEME_SYNTHID: return "synthid";
        default:             return "unknown";
    }
}

int32_t scheme_from_name(const char * name) {
    const std::string s = name ? name : "";
    if (s == "gumbel")                     return SCHEME_GUMBEL;
    if (s == "kgw" || s == "greenlist")    return SCHEME_KGW;
    if (s == "synthid")                    return SCHEME_SYNTHID;
    return -1;
}

uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Domain separators so the three schemes never share a PRF stream under one key: ASCII "KGW\0", "SYID".
static constexpr uint64_t KGW_DOMAIN     = 0x4B4757000000000Full;
static constexpr uint64_t SYNTHID_DOMAIN = 0x5359494400000000ull;

uint64_t kgw_prev_state(uint64_t key, int32_t prev_token) {
    return mix64(mix64(key ^ KGW_DOMAIN) ^ (uint64_t) (uint32_t) prev_token);
}

bool kgw_green(uint64_t prev_state, int32_t token, float gamma) {
    const uint64_t h = mix64(prev_state ^ ((uint64_t) (uint32_t) token * 0x9E3779B97F4A7C15ull));
    const double   u = (double) (h >> 11) * (1.0 / 9007199254740992.0);  // [0,1), 53 bits
    return u < (double) gamma;
}

bool kgw_is_green(uint64_t key, int32_t prev_token, int32_t token, float gamma) {
    return kgw_green(kgw_prev_state(key, prev_token), token, gamma);
}

double binomial_survival(int64_t k, int64_t n, double p) {
    if (k <= 0) {
        return 1.0;
    }
    if (k > n || n <= 0) {
        return 0.0;
    }
    if (!(p > 0.0 && p < 1.0)) {
        return p >= 1.0 ? 1.0 : 0.0;
    }
    const double lp  = std::log(p);
    const double lq  = std::log1p(-p);
    const double lnf = std::lgamma((double) n + 1.0);
    double mx = -std::numeric_limits<double>::infinity();
    std::vector<double> terms;
    terms.reserve((size_t) (n - k + 1));
    for (int64_t i = k; i <= n; ++i) {
        const double t = lnf - std::lgamma((double) i + 1.0) - std::lgamma((double) (n - i) + 1.0) +
                         (double) i * lp + (double) (n - i) * lq;
        terms.push_back(t);
        mx = std::max(mx, t);
    }
    double acc = 0.0;
    for (double t : terms) {
        acc += std::exp(t - mx);
    }
    return std::min(1.0, std::exp(mx + std::log(acc)));
}

detection detect_kgw(uint64_t key, const std::vector<int32_t> & tokens, float gamma, bool dedup, double p_threshold) {
    detection d;
    std::set<std::pair<int32_t, int32_t>> seen;
    int64_t green = 0;
    int32_t prev  = -1;
    for (int32_t tok : tokens) {
        const bool repeated = dedup && !seen.insert({ prev, tok }).second;
        if (!repeated) {
            green += kgw_is_green(key, prev, tok, gamma) ? 1 : 0;
            d.n_scored++;
        }
        prev = tok;
    }
    if (d.n_scored == 0) {
        return d;
    }
    d.score          = (double) green / (double) d.n_scored;
    d.p_value        = binomial_survival(green, d.n_scored, (double) gamma);
    d.is_watermarked = d.p_value <= p_threshold;
    return d;
}

// transformers logits_process.py:2768-2803: h = (h + x) * 6364136223846793005 + 1, int64 wrap-around.
static inline uint64_t lcg_step(uint64_t h, uint64_t x) {
    return (h + x) * 6364136223846793005ull + 1ull;
}

uint64_t synthid_context_hash(const int32_t * context, int width) {
    uint64_t h = 1;  // transformers starts every hash at 1 (:2854)
    for (int i = 0; i < width; ++i) {
        h = lcg_step(h, (uint64_t) (int64_t) context[i]);
    }
    return h;
}

static inline uint64_t synthid_layer_key(uint64_t key, int layer) {
    return mix64(key ^ SYNTHID_DOMAIN ^ (uint64_t) (uint32_t) layer);
}

int synthid_g(uint64_t key, uint64_t ctx_hash, int32_t token, int layer) {
    const uint64_t hv = lcg_step(ctx_hash, (uint64_t) (int64_t) token);
    return (int) (mix64(lcg_step(hv, synthid_layer_key(key, layer))) >> 63);
}

void synthid_reweight(uint64_t key, uint64_t ctx_hash, const int32_t * ids, float * probs, size_t n, int depth) {
    std::vector<uint64_t> hv(n);
    for (size_t i = 0; i < n; ++i) {
        hv[i] = lcg_step(ctx_hash, (uint64_t) (int64_t) ids[i]);
    }
    std::vector<uint8_t> g(n);
    for (int layer = 0; layer < depth; ++layer) {
        const uint64_t lk = synthid_layer_key(key, layer);
        double mass = 0.0;
        for (size_t i = 0; i < n; ++i) {
            g[i] = (uint8_t) (mix64(lcg_step(hv[i], lk)) >> 63);
            mass += g[i] ? (double) probs[i] : 0.0;
        }
        // one tournament round (:2696-2699): winners of the coin keep their mass plus the losers' share
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            probs[i] = (float) ((double) probs[i] * (1.0 + (double) g[i] - mass));
            sum += probs[i];
        }
        // the update preserves the sum exactly in real arithmetic; renormalise float drift over 30 layers
        if (sum > 0.0) {
            for (size_t i = 0; i < n; ++i) {
                probs[i] = (float) ((double) probs[i] / sum);
            }
        }
    }
}

detection detect_synthid(uint64_t key, const std::vector<int32_t> & tokens, int width, int depth, bool dedup, double p_threshold) {
    detection d;
    if (width < 1 || depth < 1) {
        return d;
    }
    std::vector<int32_t> hist((size_t) width, -1);
    std::set<uint64_t>   seen;
    int64_t ones = 0;
    for (int32_t tok : tokens) {
        const uint64_t ch = synthid_context_hash(hist.data() + (hist.size() - (size_t) width), width);
        const bool repeated = dedup && !seen.insert(ch).second;
        if (!repeated) {
            for (int layer = 0; layer < depth; ++layer) {
                ones += synthid_g(key, ch, tok, layer);
            }
            d.n_scored++;
        }
        hist.push_back(tok);
    }
    if (d.n_scored == 0) {
        return d;
    }
    const int64_t n  = (int64_t) d.n_scored * depth;
    d.score          = (double) ones / (double) n;
    d.p_value        = binomial_survival(ones, n, 0.5);
    d.is_watermarked = d.p_value <= p_threshold;
    return d;
}

} // namespace common_watermark
