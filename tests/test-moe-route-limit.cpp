// test-moe-route-limit.cpp -- unit tests for src/llama-moe-route-limit.{h,cpp} (measured CPU/GPU expert split planner).
// Pure host code, no model, no GPU. Row S69-FASTLLM-PICK-MEASURED-EXPERT-ROUTE-LIMIT-2026-10-05-01.
//
// FAIL-BEFORE: before this change none of these symbols exist (link error); the static rules below
// (all-CPU, all-GPU) are what the code base could express, and test_row_example shows the planner beating both.
// ANTI-BIAS: every property is checked against an independent brute force that sweeps EVERY integer limit
// instead of the candidate set, so a planner that silently returned the default would read differently.

#include "../src/llama-moe-route-limit.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define CHECK_NEAR(a, b, tol) do { if (std::fabs((double)(a) - (double)(b)) > (tol)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s=%g vs %s=%g\n", __FILE__, __LINE__, #a, (double)(a), #b, (double)(b)); ++g_fail; } } while (0)

static double brute_min_makespan(const std::vector<llama_moe_expert_cost> & ex, int n_gpu) {
    int max_r = 0;
    for (auto & e : ex) max_r = std::max(max_r, e.routes);
    double best = 1e300;
    for (int limit = 0; limit <= max_r + 2; ++limit) { // every integer, not just the candidate set
        best = std::min(best, llama_moe_route_makespan(ex, limit, n_gpu));
    }
    return best;
}

// the row's own example: experts with routes {1, 2, 200}, cpu linear (10 us/route), gpu flat 150 us
static void test_row_example() {
    std::vector<llama_moe_expert_cost> ex = {{1, 10, 150}, {2, 20, 150}, {200, 2000, 150}};
    const double all_cpu = llama_moe_route_makespan(ex, 1000, 1);
    const double all_gpu = llama_moe_route_makespan(ex, 1, 1);
    CHECK_NEAR(all_cpu, 2030, 1e-9);
    CHECK_NEAR(all_gpu, 450, 1e-9);
    const int32_t t = llama_moe_select_route_limit(ex, 32, 1);
    CHECK(t == 3);                                   // experts with 1 and 2 routes on CPU, the 200-route one on GPU
    const double planned = llama_moe_route_makespan(ex, t, 1);
    CHECK_NEAR(planned, 150, 1e-9);
    CHECK(planned < all_cpu && planned < all_gpu);   // strictly better than either static rule
}

static void test_edge_cases() {
    CHECK(llama_moe_select_route_limit({}, 32, 1) == 32);                      // nothing active -> default
    CHECK(llama_moe_select_route_limit({{0, 5, 5}}, 17, 2) == 17);             // routes<=0 skipped
    CHECK(llama_moe_select_route_limit({{-3, 5, 5}}, 17, 2) == 17);
    // the default is a fallback, not a bound: optimum above and below it
    CHECK(llama_moe_select_route_limit({{1, 1, 100}}, 5000, 1) == 2);          // cpu wins -> T above the only expert
    CHECK(llama_moe_select_route_limit({{50, 1000, 10}}, 5000, 1) == 1);       // gpu wins -> T=1 even though default is huge
    // tie keeps the SMALLER limit (more experts on the GPU)
    CHECK(llama_moe_select_route_limit({{4, 7, 7}}, 99, 1) == 1);
    // n_gpu <= 0 behaves as one queue
    CHECK(llama_moe_route_makespan({{1, 1, 10}, {1, 1, 10}}, 1, 0) == 20);
    // a second GPU queue halves two equal GPU jobs
    CHECK(llama_moe_route_makespan({{1, 1, 10}, {1, 1, 10}}, 1, 2) == 10);
}

// equal-shape golden: one expert, cpu = 10n, gpu = 100 + 2n -> break-even at n = 12.5
static void test_break_even_golden() {
    for (int r = 1; r <= 40; ++r) {
        llama_moe_expert_cost e{r, 10.0 * r, 100.0 + 2.0 * r};
        const int32_t t = llama_moe_select_route_limit({e}, 8, 1);
        const bool cpu_cheaper = e.cpu_us < e.gpu_us;
        if (cpu_cheaper) CHECK(t == r + 1);          // above the expert's route count -> CPU
        else             CHECK(t == 1);              // GPU (tie -> smaller limit = GPU too)
        CHECK_NEAR(llama_moe_route_makespan({e}, t, 1), std::min(e.cpu_us, e.gpu_us), 1e-9);
    }
    CHECK(llama_moe_select_route_limit({{12, 120, 124}}, 8, 1) == 13);   // 12 < 12.5 -> CPU
    CHECK(llama_moe_select_route_limit({{13, 130, 126}}, 8, 1) == 1);    // 13 > 12.5 -> GPU
}

static void test_random_vs_brute_force() {
    std::mt19937 rng(11);
    for (int iter = 0; iter < 3000; ++iter) {
        const int n_exp = 1 + rng() % 12;
        const int n_gpu = 1 + rng() % 3;
        std::vector<llama_moe_expert_cost> ex;
        for (int i = 0; i < n_exp; ++i) {
            const int r = rng() % 40;                                    // includes 0
            ex.push_back({r, r * (1.0 + (rng() % 50) / 10.0), 20.0 + (rng() % 300) / 3.0 + r * (rng() % 10) / 10.0});
        }
        bool any = false;
        for (auto & e : ex) any |= e.routes > 0;
        const int32_t t = llama_moe_select_route_limit(ex, 7, n_gpu);
        if (!any) { CHECK(t == 7); continue; }
        const double planned = llama_moe_route_makespan(ex, t, n_gpu);
        CHECK_NEAR(planned, brute_min_makespan(ex, n_gpu), 1e-9);        // exact optimum
        CHECK(planned <= llama_moe_route_makespan(ex, 7, n_gpu) + 1e-9); // never worse than the default
        CHECK(planned <= llama_moe_route_makespan(ex, 1, n_gpu) + 1e-9); // never worse than all-GPU
        CHECK(planned <= llama_moe_route_makespan(ex, 1 << 20, n_gpu) + 1e-9); // never worse than all-CPU
        // no smaller limit achieves the same makespan (tie rule)
        // (a limit between two candidates is the same partition as the next candidate up, so only candidates are compared)
        for (auto & c : ex) if (c.routes > 0 && c.routes + 1 < t) CHECK(llama_moe_route_makespan(ex, c.routes + 1, n_gpu) >= planned - 1e-12);
        if (t > 1) CHECK(llama_moe_route_makespan(ex, 1, n_gpu) >= planned - 1e-12);
    }
}

static void test_curve() {
    llama_moe_cost_curve c;
    CHECK(c.at(5) == 0.0);                           // empty curve
    c.pts = {{0, 0.0}, {4, 40.0}, {8, 120.0}};
    CHECK(c.at(0) == 0.0 && c.at(-3) == 0.0);
    CHECK(c.at(4) == 40.0);                          // exact sample
    CHECK_NEAR(c.at(2), 20.0, 1e-12);                // interpolate
    CHECK_NEAR(c.at(6), 80.0, 1e-12);
    CHECK_NEAR(c.at(10), 120.0 + 20.0 * 2, 1e-12);   // extrapolate with the last segment slope (20 per token)
    llama_moe_cost_curve d;
    d.pts = {{0, 0.0}, {4, 40.0}, {8, 10.0}};        // decreasing tail -> negative extrapolation clamps at 0
    CHECK(d.at(100) == 0.0);
    llama_moe_cost_curve one;
    one.pts = {{4, 9.0}};                            // single point: flat
    CHECK(one.at(2) == 9.0 && one.at(40) == 9.0);
}

static void test_sample_points() {
    CHECK(llama_moe_route_sample_points(0, false).empty());
    auto g = llama_moe_route_sample_points(512, false);
    CHECK(g.front() == 1 && g.back() == 512);
    CHECK(std::is_sorted(g.begin(), g.end()) && std::adjacent_find(g.begin(), g.end()) == g.end());
    for (int i = 1; i <= 16; ++i) CHECK(std::find(g.begin(), g.end(), i) != g.end());     // step 1 up to 16
    CHECK(std::find(g.begin(), g.end(), 17) != g.end() && std::find(g.begin(), g.end(), 18) == g.end() && std::find(g.begin(), g.end(), 21) != g.end()); // 16 -> 17 (step 1), then step 4 -> 21
    auto c = llama_moe_route_sample_points(512, true);
    CHECK(c.size() < g.size());                      // cpu curve is sparser
    CHECK(c.front() == 1 && c.back() == 512);
    auto one = llama_moe_route_sample_points(1, true);
    CHECK(one.size() == 1 && one[0] == 1);
    auto odd = llama_moe_route_sample_points(37, false);
    CHECK(odd.back() == 37);
}

static void test_profile_build_and_plan() {
    int cpu_calls = 0, gpu_calls = 0;
    auto cpu = [&](int32_t n) { ++cpu_calls; return 10.0 * n; };
    auto gpu = [&](int32_t) { ++gpu_calls; return 150.0; };
    llama_moe_route_profile p;
    CHECK(!llama_moe_route_profile_build(p, 0, cpu, gpu));
    CHECK(!llama_moe_route_profile_build(p, 64, nullptr, gpu));
    CHECK(!p.initialized);                           // failed builds leave it untouched
    CHECK(llama_moe_route_profile_build(p, 64, cpu, gpu));
    CHECK(p.initialized && p.max_n == 64);
    const int n_cpu_pts = (int) llama_moe_route_sample_points(64, true).size();
    const int n_gpu_pts = (int) llama_moe_route_sample_points(64, false).size();
    CHECK(cpu_calls == n_cpu_pts * (LLAMA_MOE_ROUTE_WARMUP_ROUNDS + LLAMA_MOE_ROUTE_MEASURE_ROUNDS));
    CHECK(gpu_calls == n_gpu_pts * (LLAMA_MOE_ROUTE_WARMUP_ROUNDS + LLAMA_MOE_ROUTE_MEASURE_ROUNDS));
    CHECK(p.cpu.pts.count(0) && p.gpu.pts.count(0) && p.cpu.pts.at(0) == 0.0);
    CHECK_NEAR(p.cpu.at(10), 100.0, 1e-9);           // mean of identical measurements
    CHECK_NEAR(p.gpu.at(33), 150.0, 1e-9);
    // plan from a routing histogram: expert 0:1 route, 1:2 routes, 2:200 routes -> the row example
    std::vector<int32_t> routes = {1, 2, 200, 0};
    CHECK(llama_moe_plan_route_limit(p, routes, {}, 32, 1) == 3); // 200 is past the profile: extrapolated with slope 10 / 0
    // a missing-weights expert is skipped
    std::vector<char> has = {1, 1, 0, 1};
    const int32_t t_skip = llama_moe_plan_route_limit(p, routes, has, 32, 1);
    CHECK(t_skip == 3); // only the 1- and 2-route experts remain: all on the CPU (30 us) beats all on the GPU (300 us)
    // unusable profile -> default
    llama_moe_route_profile empty;
    CHECK(llama_moe_plan_route_limit(empty, routes, {}, 21, 1) == 21);
    // no GPU callback -> gpu curve is a copy of the cpu curve; the CPU and the GPU queue still run in PARALLEL, so splitting wins
    llama_moe_route_profile nog;
    CHECK(llama_moe_route_profile_build(nog, 32, cpu, nullptr));
    CHECK(nog.gpu.pts == nog.cpu.pts);
    CHECK(llama_moe_plan_route_limit(nog, {3, 9}, {}, 5, 1) == 4); // 3-route expert on the CPU (30), 9-route on the GPU queue (90): 90 < 120
}

static void test_plan_matches_row_example_through_profile() {
    // profile whose curves reproduce the row's example exactly at n = 1, 2, 200 (cpu 10n, gpu 150 flat)
    llama_moe_route_profile p;
    p.initialized = true; p.max_n = 512;
    for (int n : {0, 1, 2, 200, 512}) { p.cpu.pts[n] = 10.0 * n; p.gpu.pts[n] = n == 0 ? 0.0 : 150.0; }
    CHECK(llama_moe_plan_route_limit(p, {1, 2, 200}, {}, 32, 1) == 3);
}

static void test_cache() {
    llama_moe_route_profile_cache cache;
    llama_moe_route_shape_key k1; k1.n_embd = 4096; k1.n_ff_exp = 1536; k1.n_embd_out = 4096; k1.n_expert_used = 8;
    llama_moe_route_shape_key k2 = k1; k2.gpu_id = 1;
    int builds = 0;
    auto cpu = [&](int32_t n) { return 1.0 * n; };
    auto gpu = [&](int32_t n) { if (n == 1) ++builds; return 50.0; };   // n == 1 is sampled exactly once per build x rounds
    llama_moe_route_profile out;
    CHECK(cache.get(k1, 64, cpu, gpu, out) && out.max_n == 64);
    const int after_first = builds;
    CHECK(after_first == LLAMA_MOE_ROUTE_WARMUP_ROUNDS + LLAMA_MOE_ROUTE_MEASURE_ROUNDS);
    CHECK(cache.get(k1, 32, cpu, gpu, out) && out.max_n == 64);         // smaller request: cache hit, no rebuild
    CHECK(builds == after_first);
    CHECK(cache.get(k1, 128, cpu, gpu, out) && out.max_n == 128);       // larger request: rebuilt
    CHECK(builds == 2 * after_first);
    CHECK(cache.get(k1, 100000, cpu, gpu, out) && out.max_n == LLAMA_MOE_ROUTE_BENCH_MAX_N); // capped at 512
    CHECK(cache.get(k2, 16, cpu, gpu, out) && cache.size() == 2);       // another device = another profile
    CHECK(!cache.get(k2, 0, cpu, gpu, out));
    // concurrent readers do not corrupt it
    std::vector<std::thread> th;
    for (int i = 0; i < 4; ++i) th.emplace_back([&] { llama_moe_route_profile o; for (int j = 0; j < 50; ++j) cache.get(k1, 64, cpu, gpu, o); });
    for (auto & t : th) t.join();
    CHECK(cache.size() == 2);
}

static void test_override_parse() {
    CHECK(llama_moe_route_limit_parse_override(nullptr) == -1);
    CHECK(llama_moe_route_limit_parse_override("") == -1);
    CHECK(llama_moe_route_limit_parse_override("abc") == -1);
    CHECK(llama_moe_route_limit_parse_override("12x") == -1);
    CHECK(llama_moe_route_limit_parse_override("-4") == -1);
    CHECK(llama_moe_route_limit_parse_override(" 4") == -1);
    CHECK(llama_moe_route_limit_parse_override("0") == -1);
    CHECK(llama_moe_route_limit_parse_override("1") == 1);
    CHECK(llama_moe_route_limit_parse_override("48") == 48);
    CHECK(llama_moe_route_limit_parse_override("1048576") == 1048576);
    CHECK(llama_moe_route_limit_parse_override("1048577") == -1);
    CHECK(llama_moe_route_limit_parse_override("99999999999999999999") == -1);
}

int main() {
    test_row_example();
    test_edge_cases();
    test_break_even_golden();
    test_random_vs_brute_force();
    test_curve();
    test_sample_points();
    test_profile_build_and_plan();
    test_plan_matches_row_example_through_profile();
    test_cache();
    test_override_parse();
    if (g_fail) { std::fprintf(stderr, "test-moe-route-limit: %d FAILURES\n", g_fail); return 1; }
    std::printf("test-moe-route-limit: PASS\n");
    return 0;
}
