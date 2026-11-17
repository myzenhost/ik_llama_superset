// asr-dtw.cpp -- see asr-dtw.h
#include "asr-dtw.h"

#include <algorithm>
#include <cmath>
#include <limits>

std::vector<std::pair<int, int>> asr_dtw_path(const float * x, int N, int M) {
    std::vector<std::pair<int, int>> path;
    if (!x || N <= 0 || M <= 0) {
        return path;
    }
    const float INF = std::numeric_limits<float>::infinity();
    const size_t W = (size_t) M + 1;
    std::vector<float>  cost((size_t) (N + 1) * W, INF);
    std::vector<int8_t> trace((size_t) (N + 1) * W, -1);
    cost[0] = 0.0f;

    // i = token (row), j = frame (col); cost/trace have a leading border row and column
    for (int i = 1; i <= N; ++i) {
        for (int j = 1; j <= M; ++j) {
            const float c0 = cost[(size_t) (i - 1) * W + (j - 1)]; // diagonal
            const float c1 = cost[(size_t) (i - 1) * W + j];       // token advances
            const float c2 = cost[(size_t) i * W + (j - 1)];       // frame advances
            float  c;
            int8_t t;
            if (c0 < c1 && c0 < c2)      { c = c0; t = 0; }
            else if (c1 < c0 && c1 < c2) { c = c1; t = 1; }
            else                         { c = c2; t = 2; }
            cost[(size_t) i * W + j]  = x[(size_t) (i - 1) * M + (j - 1)] + c;
            trace[(size_t) i * W + j] = t;
        }
    }
    // borders: first row can only come from the left, first column only from above
    for (int j = 0; j <= M; ++j) { trace[j] = 2; }
    for (int i = 0; i <= N; ++i) { trace[(size_t) i * W] = 1; }

    int i = N, j = M;
    while (i > 0 || j > 0) {
        path.emplace_back(i - 1, j - 1);
        const int8_t t = trace[(size_t) i * W + j];
        if (t == 0)      { --i; --j; }
        else if (t == 1) { --i; }
        else             { --j; }
    }
    std::reverse(path.begin(), path.end());
    return path;
}

bool asr_dtw_median_filter(float * data, int n_rows, int n_cols, int width) {
    if (!data || n_rows <= 0 || n_cols <= 0 || width < 1 || width % 2 == 0 || width >= n_cols) {
        return width == 1 && data; // width 1 is the identity
    }
    const int h = width / 2;
    std::vector<float> src(n_cols), win(width);
    for (int r = 0; r < n_rows; ++r) {
        float * row = data + (size_t) r * n_cols;
        std::copy(row, row + n_cols, src.begin());
        for (int k = 0; k < n_cols; ++k) {
            for (int o = -h; o <= h; ++o) {
                int idx = k + o;
                if (idx < 0)            { idx = -idx; }
                else if (idx >= n_cols) { idx = 2 * (n_cols - 1) - idx; }
                win[o + h] = src[idx];
            }
            std::nth_element(win.begin(), win.begin() + h, win.end());
            row[k] = win[h];
        }
    }
    return true;
}

std::vector<int> asr_dtw_token_start_frames(const std::vector<std::pair<int, int>> & path, int n_tokens) {
    std::vector<int> start(n_tokens > 0 ? n_tokens : 0, -1);
    for (const auto & pr : path) {
        if (pr.first >= 0 && pr.first < n_tokens && start[pr.first] < 0) {
            start[pr.first] = pr.second;
        }
    }
    return start;
}

std::vector<int> asr_dtw_token_frames(const float * attn, int n_heads, int n_tokens, int n_frames, int medfilt_width,
                                      const float * frame_w) {
    if (!attn || n_heads <= 0 || n_tokens <= 0 || n_frames <= 0) {
        return {};
    }
    std::vector<float> w(attn, attn + (size_t) n_heads * n_tokens * n_frames);

    // normalise over the token axis, per head and frame
    for (int h = 0; h < n_heads; ++h) {
        float * hd = w.data() + (size_t) h * n_tokens * n_frames;
        for (int f = 0; f < n_frames; ++f) {
            double mean = 0.0;
            for (int t = 0; t < n_tokens; ++t) { mean += hd[(size_t) t * n_frames + f]; }
            mean /= n_tokens;
            double var = 0.0;
            for (int t = 0; t < n_tokens; ++t) { const double d = hd[(size_t) t * n_frames + f] - mean; var += d * d; }
            var /= n_tokens;
            const double inv = 1.0 / std::sqrt(var + 1e-9);
            for (int t = 0; t < n_tokens; ++t) { hd[(size_t) t * n_frames + f] = (float) ((hd[(size_t) t * n_frames + f] - mean) * inv); }
        }
        asr_dtw_median_filter(hd, n_tokens, n_frames, medfilt_width);
    }

    // mean over heads, negated -> cost
    std::vector<float> cost((size_t) n_tokens * n_frames, 0.0f);
    for (int h = 0; h < n_heads; ++h) {
        const float * hd = w.data() + (size_t) h * n_tokens * n_frames;
        for (size_t k = 0; k < cost.size(); ++k) { cost[k] += hd[k]; }
    }
    for (float & c : cost) { c = -c / (float) n_heads; }
    if (frame_w) {
        for (int t = 0; t < n_tokens; ++t) {
            for (int f = 0; f < n_frames; ++f) { cost[(size_t) t * n_frames + f] *= frame_w[f]; }
        }
    }

    return asr_dtw_token_start_frames(asr_dtw_path(cost.data(), n_tokens, n_frames), n_tokens);
}
