// test-mtmd-audio-chunks.cpp -- tools/mtmd/mtmd-audio-chunks: long audio is cut in silence, not at blind 30 s marks

#include "mtmd-audio-chunks.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846   // MSVC <cmath> defines it only under _USE_MATH_DEFINES
#endif

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static const int SR = 16000;

// a 220 Hz tone (stand-in for continuous speech) with digital-silence gaps [a, b) seconds
static std::vector<float> tone_with_gaps(double seconds, const std::vector<std::pair<double, double>> & gaps) {
    std::vector<float> x((size_t) (seconds * SR));
    for (size_t i = 0; i < x.size(); ++i) {
        const double t = (double) i / SR;
        bool gap = false;
        for (const auto & g : gaps) { gap = gap || (t >= g.first && t < g.second); }
        x[i] = gap ? 0.0f : (float) (0.1 * std::sin(2.0 * M_PI * 220.0 * t));
    }
    return x;
}

static mtmd_audio_chunk_params params30(double search_s) {
    mtmd_audio_chunk_params p;
    p.max_samples    = 30 * SR;
    p.search_samples = (size_t) (search_s * SR);
    p.frame_samples  = SR / 100;
    return p;
}

static bool tiles(const std::vector<std::pair<size_t, size_t>> & c, size_t n, size_t max) {
    if (c.empty() || c.front().first != 0 || c.back().second != n) { return false; }
    for (size_t i = 0; i < c.size(); ++i) {
        if (c[i].second <= c[i].first || c[i].second - c[i].first > max) { return false; }
        if (i > 0 && c[i].first != c[i - 1].second) { return false; }
    }
    return true;
}

static bool in_gap(size_t s, const std::vector<std::pair<double, double>> & gaps) {
    const double t = (double) s / SR;
    for (const auto & g : gaps) { if (t >= g.first && t <= g.second) { return true; } }
    return false;
}

int main() {
    // 1. input that fits one window is one chunk, unchanged
    {
        auto x = tone_with_gaps(29.99, {});
        auto c = mtmd_audio_plan_chunks(x.data(), x.size(), params30(10));
        CHECK(c.size() == 1 && c[0].first == 0 && c[0].second == x.size(), "short input: %zu chunk(s)", c.size());
        auto e = mtmd_audio_plan_chunks(x.data(), 0, params30(10));
        CHECK(e.empty(), "empty input gives no chunk");
    }
    // 2. 70 s of "speech" with two 300 ms pauses: each cut lands in a pause, not at 30 s / 60 s
    const std::vector<std::pair<double, double>> gaps = {{24.0, 24.3}, {52.0, 52.3}};
    {
        auto x = tone_with_gaps(70.0, gaps);
        auto c = mtmd_audio_plan_chunks(x.data(), x.size(), params30(10));
        CHECK(tiles(c, x.size(), 30 * SR), "70 s: chunks do not tile [0,n) within 30 s");
        CHECK(c.size() == 3, "70 s: %zu chunks, expected 3", c.size());
        for (size_t i = 0; i + 1 < c.size(); ++i) {
            CHECK(in_gap(c[i].second, gaps), "cut %zu at %.3f s is not in a pause", i, (double) c[i].second / SR);
            printf("cut %zu at %.3f s\n", i, (double) c[i].second / SR);
        }
        // negative control: the blind planner (no search) reproduces the old fixed 30 s cuts, which sit in the tone
        auto b = mtmd_audio_plan_chunks(x.data(), x.size(), params30(0));
        CHECK(tiles(b, x.size(), 30 * SR) && b.size() == 3 && b[0].second == (size_t) 30 * SR && b[1].second == (size_t) 60 * SR,
              "blind planner is not the old fixed split");
        CHECK(!in_gap(b[0].second, gaps), "control: the blind 30 s cut should be mid-tone");
    }
    // 3. two equally silent pauses in the search window: the LATER one wins (longer windows)
    {
        const std::vector<std::pair<double, double>> g2 = {{22.0, 22.3}, {27.0, 27.3}};
        auto x = tone_with_gaps(40.0, g2);
        auto c = mtmd_audio_plan_chunks(x.data(), x.size(), params30(10));
        CHECK(c.size() == 2 && (double) c[0].second / SR >= 27.0 && (double) c[0].second / SR <= 27.3,
              "two pauses: cut at %.3f s, expected inside the later pause 27.0-27.3", (double) c[0].second / SR);
    }
    // 4. a pause deeper by more than tie_db beats a later shallow dip
    {
        auto x = tone_with_gaps(40.0, {{21.0, 21.3}});
        for (size_t i = (size_t) (27.0 * SR); i < (size_t) (27.3 * SR); ++i) { x[i] *= 0.5f; } // only -6 dB
        auto c = mtmd_audio_plan_chunks(x.data(), x.size(), params30(10));
        CHECK(c.size() == 2 && in_gap(c[0].second, {{21.0, 21.3}}), "deep pause: cut at %.3f s, expected 21.0-21.3",
              (double) c[0].second / SR);
    }
    // 5. no pause at all (steady noise): still a valid tiling, every chunk within the limit
    {
        std::mt19937 rng(1234);
        std::normal_distribution<float> nd(0.0f, 0.05f);
        std::vector<float> x((size_t) 95 * SR);
        for (auto & v : x) { v = nd(rng); }
        auto c = mtmd_audio_plan_chunks(x.data(), x.size(), params30(10));
        CHECK(tiles(c, x.size(), 30 * SR), "noise: chunks do not tile");
        for (size_t i = 0; i + 1 < c.size(); ++i) {
            CHECK(c[i].second - c[i].first >= (size_t) 20 * SR, "noise: chunk %zu shorter than limit - search", i);
        }
    }
    // 6. tiling invariant over odd lengths, including a pause right at the search edge and at the very end
    for (double secs : {30.01, 31.0, 59.99, 60.0, 61.234, 120.5}) {
        auto x = tone_with_gaps(secs, {{20.0, 20.1}, {secs - 0.2, secs}});
        auto c = mtmd_audio_plan_chunks(x.data(), x.size(), params30(10));
        CHECK(tiles(c, x.size(), 30 * SR), "%.3f s: chunks do not tile", secs);
    }

    printf(g_fail ? "FAILED (%d failure(s))\n" : "OK (%d failure(s))\n", g_fail);
    return g_fail ? 1 : 0;
}
