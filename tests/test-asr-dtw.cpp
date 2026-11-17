// Model-free test of examples/audio/asr-dtw: DTW token timestamps from synthetic attention.
#include "asr-dtw.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int n_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); n_fail++; } } while (0)

int main() {
    // ---- DTW path on a hand-made cost matrix (3 tokens x 5 frames), low cost on the staircase
    {
        const float INF = 9.0f;
        const float c[3 * 5] = {
            0, 0, INF, INF, INF,
            INF, INF, 0, INF, INF,
            INF, INF, INF, 0, 0,
        };
        auto path = asr_dtw_path(c, 3, 5);
        printf("path:"); for (auto & p : path) printf(" (%d,%d)", p.first, p.second); printf("\n");
        CHECK(path.front() == std::make_pair(0, 0));
        CHECK(path.back()  == std::make_pair(2, 4));
        // monotone, steps of at most 1 in each axis
        for (size_t i = 1; i < path.size(); ++i) {
            const int dt = path[i].first - path[i - 1].first, df = path[i].second - path[i - 1].second;
            CHECK(dt >= 0 && dt <= 1 && df >= 0 && df <= 1 && dt + df >= 1);
        }
        auto st = asr_dtw_token_start_frames(path, 3);
        CHECK(st.size() == 3 && st[0] == 0 && st[1] == 2 && st[2] == 3);
    }
    // ---- tie-break: uniform cost 1. Hand-derived: (1,1) is a strict diagonal win (borders are inf);
    //      at the last cell c0 == c2 < c1 so the tie goes to "frame advances" => path (0,0)(1,1)(1,2)
    {
        std::vector<float> c(2 * 3, 1.0f);
        auto path = asr_dtw_path(c.data(), 2, 3);
        printf("ties:"); for (auto & p : path) printf(" (%d,%d)", p.first, p.second); printf("\n");
        CHECK(path.size() == 3 && path[1] == std::make_pair(1, 1) && path.back() == std::make_pair(1, 2));
        auto st = asr_dtw_token_start_frames(path, 2);
        CHECK(st[0] == 0 && st[1] == 1);
    }
    // ---- degenerate input
    CHECK(asr_dtw_path(nullptr, 3, 3).empty());
    { float z = 1; CHECK(asr_dtw_path(&z, 0, 1).empty()); CHECK(asr_dtw_path(&z, 1, 1).size() == 1); }

    // ---- median filter: spike removed, reflect padding at the edges
    {
        std::vector<float> d = {0, 0, 9, 0, 0, 0, 5};
        CHECK(asr_dtw_median_filter(d.data(), 1, 7, 3));
        printf("medfilt:"); for (float v : d) printf(" %g", v); printf("\n");
        CHECK(d[2] == 0);                    // spike gone
        CHECK(d[6] == 0);                    // edge: window (0,5,0) via reflect => median 0
        std::vector<float> e = {1, 2, 3};
        CHECK(!asr_dtw_median_filter(e.data(), 1, 3, 4));   // even width rejected
        CHECK(e[0] == 1 && e[1] == 2 && e[2] == 3);
        CHECK(!asr_dtw_median_filter(e.data(), 1, 3, 3));   // width >= n_cols rejected
    }

    // ---- full pipeline: 2 heads, 6 tokens, 60 frames; token k attends to frames [10k, 10k+10)
    {
        const int H = 2, T = 6, F = 60;
        std::vector<float> a((size_t) H * T * F, 0.01f);
        for (int h = 0; h < H; ++h)
            for (int t = 0; t < T; ++t)
                for (int f = 10 * t; f < 10 * t + 10; ++f)
                    a[((size_t) h * T + t) * F + f] = 0.8f + 0.01f * h;
        auto st = asr_dtw_token_frames(a.data(), H, T, F, 5);
        printf("starts:"); for (int s : st) printf(" %d", s); printf("\n");
        CHECK((int) st.size() == T);
        for (int t = 0; t < T && (int) st.size() == T; ++t) CHECK(std::abs(st[t] - 10 * t) <= 1);
        CHECK(!st.empty() && st[0] == 0);
        // exact on this clean input
        for (int t = 0; t < T && (int) st.size() == T; ++t) CHECK(st[t] == 10 * t);
    }
    // ---- invalid args
    CHECK(asr_dtw_token_frames(nullptr, 1, 1, 1).empty());

    printf(n_fail ? "test-asr-dtw: %d FAILED\n" : "test-asr-dtw: OK\n", n_fail);
    return n_fail ? 1 : 0;
}
