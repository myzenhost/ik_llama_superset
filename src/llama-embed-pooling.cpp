// llama-embed-pooling.cpp — see llama-embed-pooling.h. Pure-host pooling math (standalone-compiles +
// self-tests with no ggml/CUDA). Mirrors cortex/embedding/harness_pooling.py.
#include "llama-embed-pooling.h"
#include <algorithm>
#include <cmath>

namespace justician_embed {

static inline bool valid(const std::vector<int8_t>& mask, int i) {
    return mask.empty() || (i < (int) mask.size() && mask[i] != 0);
}

static int last_valid(int n_tok, const std::vector<int8_t>& mask) {
    for (int i = n_tok - 1; i >= 0; --i) if (valid(mask, i)) return i;
    return n_tok - 1;
}

std::vector<float> pool(const std::vector<float>& h, int n_tok, int dim,
                        PoolMode mode, const std::vector<int8_t>& mask) {
    std::vector<float> out(dim, 0.0f);
    if (n_tok <= 0 || dim <= 0) return out;
    auto row = [&](int t) -> const float* { return h.data() + (size_t) t * dim; };

    if (mode == PoolMode::Cls) {
        const float* r = row(0);
        out.assign(r, r + dim);
    } else if (mode == PoolMode::Last) {
        const float* r = row(last_valid(n_tok, mask));
        out.assign(r, r + dim);
    } else if (mode == PoolMode::Mean) {
        int cnt = 0;
        for (int t = 0; t < n_tok; ++t) if (valid(mask, t)) {
            const float* r = row(t); for (int d = 0; d < dim; ++d) out[d] += r[d];
            ++cnt;
        }
        if (cnt) for (int d = 0; d < dim; ++d) out[d] /= cnt;
    } else if (mode == PoolMode::Max) {
        bool init = false;
        for (int t = 0; t < n_tok; ++t) if (valid(mask, t)) {
            const float* r = row(t);
            if (!init) { for (int d = 0; d < dim; ++d) out[d] = r[d]; init = true; }
            else for (int d = 0; d < dim; ++d) out[d] = std::max(out[d], r[d]);
        }
    } else { // Attention (param-free latent-attn): query = masked mean; w = softmax(token . query)
        std::vector<float> q(dim, 0.0f);
        int cnt = 0;
        for (int t = 0; t < n_tok; ++t) if (valid(mask, t)) {
            const float* r = row(t); for (int d = 0; d < dim; ++d) q[d] += r[d];
            ++cnt;
        }
        if (cnt) for (int d = 0; d < dim; ++d) q[d] /= cnt;
        std::vector<float> score(n_tok, -1e30f);
        float mx = -1e30f;
        for (int t = 0; t < n_tok; ++t) if (valid(mask, t)) {
            const float* r = row(t); float s = 0.0f;
            for (int d = 0; d < dim; ++d) s += r[d] * q[d];
            score[t] = s; mx = std::max(mx, s);
        }
        float z = 0.0f;
        for (int t = 0; t < n_tok; ++t) if (valid(mask, t)) { score[t] = std::exp(score[t] - mx); z += score[t]; }
        if (z <= 0.0f) z = 1.0f;
        for (int t = 0; t < n_tok; ++t) if (valid(mask, t)) {
            const float* r = row(t); float w = score[t] / z;
            for (int d = 0; d < dim; ++d) out[d] += w * r[d];
        }
    }
    return out;
}

std::vector<float> mrl_truncate(const std::vector<float>& v, int in_dim, int keep_dim) {
    if (keep_dim <= 0 || keep_dim >= in_dim || in_dim <= 0) return v;
    int rows = (int) (v.size() / in_dim);
    std::vector<float> out((size_t) rows * keep_dim, 0.0f);
    for (int r = 0; r < rows; ++r) {
        const float* src = v.data() + (size_t) r * in_dim;
        float* dst = out.data() + (size_t) r * keep_dim;
        double n = 0.0;
        for (int d = 0; d < keep_dim; ++d) { dst[d] = src[d]; n += (double) src[d] * src[d]; }
        float nn = (float) (n > 0 ? std::sqrt(n) : 1.0);
        for (int d = 0; d < keep_dim; ++d) dst[d] /= nn;
    }
    return out;
}

int _selftest() {
    // 3 tokens (last is pad), dim 4
    std::vector<float> h = {1,0,0,0,  0,2,0,0,  9,9,9,9};
    std::vector<int8_t> mask = {1, 1, 0};
    auto mean = pool(h, 3, 4, PoolMode::Mean, mask);
    if (!(mean[0] == 0.5f && mean[1] == 1.0f && mean[2] == 0.0f)) return 1;   // pad excluded
    auto last = pool(h, 3, 4, PoolMode::Last, mask);
    if (!(last[1] == 2.0f && last[0] == 0.0f)) return 2;                       // last NON-pad = token 1
    auto cls = pool(h, 3, 4, PoolMode::Cls, mask);
    if (cls[0] != 1.0f) return 3;
    auto mx = pool(h, 3, 4, PoolMode::Max, mask);
    if (!(mx[0] == 1.0f && mx[1] == 2.0f)) return 4;                           // pad excluded from max
    auto att = pool(h, 3, 4, PoolMode::Attention, mask);
    // attention pool = convex combo of tokens 0,1 -> each dim within [min,max] of the two valid rows
    if (!(att[0] >= 0.0f && att[0] <= 1.0f && att[1] >= 0.0f && att[1] <= 2.0f && att[2] == 0.0f)) return 5;
    // MRL truncate: 1 row dim4 -> keep 2 + renorm to unit
    auto t = mrl_truncate(std::vector<float>{3,4,99,99}, 4, 2);
    double nn = std::sqrt((double) t[0]*t[0] + (double) t[1]*t[1]);
    if (t.size() != 2 || std::abs(nn - 1.0) > 1e-5) return 6;                  // (3,4)->(0.6,0.8)
    if (std::abs(t[0] - 0.6f) > 1e-4 || std::abs(t[1] - 0.8f) > 1e-4) return 7;
    return 0;
}

} // namespace justician_embed
