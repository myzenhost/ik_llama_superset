// test-cuda-autotune.cpp -- CPU tests of the kernel-variant autotuner (ggml/src/ggml-cuda/autotune.h).
//
// Row S69-EXLLAMAV3-PICK-COOPAUTOTUNE-2026-09-28-01.  The selection logic is host-only C++, so it is exercised here against
// SYNTHETIC cost functions (noise, spikes, first-launch penalty); the CUDA timing callback is the parent's GPU run (row text).
//
// Fail-before partner: a naive one-sample minimum must be measurably worse than select_best on the same noisy cost model --
// otherwise the machinery would not be earning its keep.
// Controls: a corrupted cache record, a foreign file, a stale candidate must all be refused.

#include "../ggml/src/ggml-cuda/autotune.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <thread>

namespace at = ggml_cuda_at;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

// ---- budgets / order / scoring -----------------------------------------------------------------------------------

static void test_budget() {
    CHECK(at::budget_for(1).repeats == 8 && at::budget_for(1).max_rounds == 32 && at::budget_for(1).max_cands == 7);
    CHECK(at::budget_for(1e6).repeats == 5 && at::budget_for(1e6).max_cands == 5);          // boundary belongs to the next tier
    CHECK(at::budget_for(9.99e6).max_rounds == 12);
    CHECK(at::budget_for(1e7).max_cands == 4);
    CHECK(at::budget_for(1e8).repeats == 3 && at::budget_for(1e8).max_rounds == 5);
    CHECK(at::budget_for(2e8).repeats == 2 && at::budget_for(2e8).max_rounds == 2 && at::budget_for(1e12).max_cands == 2);
    double prev_r = 1e9, prev_c = 1e9;
    for (double w : { 1e3, 1e6, 1e7, 1e8, 2e8, 1e10 }) {                                      // never grows with size
        CHECK(at::budget_for(w).repeats <= prev_r && at::budget_for(w).max_cands <= prev_c);
        prev_r = at::budget_for(w).repeats; prev_c = at::budget_for(w).max_cands;
    }
}

static void test_visit_order() {
    for (int n = 1; n <= 40; n++) {
        for (int round = 0; round < 60; round++) {
            const std::vector<int> o = at::visit_order(n, round);
            CHECK(int(o.size()) == n);
            std::set<int> s(o.begin(), o.end());
            CHECK(int(s.size()) == n);                         // a permutation: every slot exactly once
            CHECK(*s.begin() == 0 && *s.rbegin() == n - 1);
        }
    }
    CHECK(at::visit_order(0, 3).empty());
    // the first slot visited rotates, so no candidate is always timed first (clock ramp / thermal drift)
    for (int n : { 3, 4, 5, 6, 7 }) {
        std::set<int> firsts;
        for (int round = 0; round < 4 * n; round++) { firsts.insert(at::visit_order(n, round)[0]); }
        CHECK(int(firsts.size()) >= std::max(2, n / 2));
    }
    bool differs = false;
    for (int round = 0; round < 8; round++) { differs |= at::visit_order(6, round) != at::visit_order(6, 0); }
    CHECK(differs);
}

static void test_trimmed_mean() {
    CHECK(at::trimmed_mean({}) == 0.0);
    CHECK(at::trimmed_mean({ 5.0 }) == 5.0);
    CHECK(at::trimmed_mean({ 1.0, 3.0 }) == 2.0);                   // n/4 == 0: nothing dropped
    CHECK(std::fabs(at::trimmed_mean({ 1, 1, 1, 1, 1, 1, 1, 1000 }) - 1.0) < 1e-12);   // drop = 2: the stall is gone
    CHECK(std::fabs(at::trimmed_mean({ 0.001, 5, 5, 5, 5, 5, 5, 5 }) - 5.0) < 1e-12);  // and a lucky outlier does not win either
    CHECK(std::fabs(at::trimmed_mean({ 9, 1, 5, 3, 7, 2, 8, 4 }) - 4.75) < 1e-12);      // sorted 1,2,3,4,5,7,8,9; drop two each side -> 3,4,5,7
}

static void test_expand_sm() {
    auto v = at::expand_sm_candidates(100, 170, 4);
    CHECK(!v.empty() && v.front().num_sms == 100 && v.back().num_sms == 170);
    for (auto & c : v) { CHECK(c.num_sms >= 1 && c.num_sms <= 170 && c.concurrency >= 1 && c.concurrency <= 4); }
    CHECK(v[0].concurrency == 1);                                    // 170/100 == 1
    CHECK(v.size() == (144 - 100) / 2 + 1 + 1);                      // 100,102..144 (85% of 170) plus the full device
    for (auto & c : v) { CHECK(c.num_sms <= 144 || c.num_sms == 170); }
    auto w = at::expand_sm_candidates(10, 170, 4);                   // few tiles: small grids, real concurrency
    CHECK(w.front().num_sms == 10 && w.front().concurrency == 4);
    std::set<std::pair<int, int>> uniq;
    for (auto & c : w) { uniq.insert({ c.num_sms, c.concurrency }); }
    CHECK(uniq.size() == w.size());                                  // no duplicates
    CHECK(at::expand_sm_candidates(500, 100, 2).size() == 1);        // tiles > SMs: just the full device
    CHECK(at::expand_sm_candidates(0, 100, 2).empty() && at::expand_sm_candidates(5, 0, 2).empty());
}

// ---- selection against a synthetic noisy cost model --------------------------------------------------------------

struct model {
    std::vector<double> truth;      // true ms per launch
    double noise = 0.05;            // multiplicative gaussian sigma
    double spike_p = 0.03;          // chance of a 3x stall
    double lucky_p = 0.03;          // chance of a 0.6x too-good sample (clock boost / cache luck)
    std::vector<double> first_pen;  // per-candidate extra fraction on the FIRST launch if not warmed (lazy module load, attribute set)
    std::mt19937_64 rng;
    std::vector<int> launches;
    std::vector<bool> warmed;
    int calls = 0, flushes = 0, warmups = 0;

    model(std::vector<double> t, uint64_t seed) : truth(std::move(t)), rng(seed), launches(truth.size(), 0), warmed(truth.size(), false) {}

    double sample(int i) {
        std::normal_distribution<double> nd(0.0, noise);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double v = truth[i] * (1.0 + nd(rng));
        if (u(rng) < spike_p) { v *= 3.0; }
        if (u(rng) < lucky_p) { v *= 0.6; }
        if (launches[i]++ == 0 && !warmed[i] && i < int(first_pen.size())) { v *= 1.0 + first_pen[i]; }
        return std::max(v, 1e-9);
    }
    at::hooks make_hooks(bool with_warm = true, bool with_flush = true) {
        at::hooks h;
        h.time_ms = [this](int i, int rep) { calls++; double s = 0; for (int r = 0; r < rep; r++) { s += sample(i); } return s / rep; };
        if (with_warm)  { h.warmup = [this](int i) { warmups++; sample(i); warmed[i] = true; }; }
        if (with_flush) { h.flush = [this]() { flushes++; }; }
        return h;
    }
};

static void test_select() {
    // clear winner (idx 3, 12% faster than the next) among 6 candidates, noisy + spikes + lucky outliers
    const std::vector<double> truth = { 1.00, 1.05, 0.98, 0.86, 1.10, 1.02 };
    int hits = 0, naive_hits = 0, trials = 400;
    for (int t = 0; t < trials; t++) {
        model m(truth, 1000 + t);
        const at::budget b = at::budget_for(5e6);            // repeats 5, rounds 12; raise max_cands so all 6 compete
        at::budget bb { b.repeats, b.max_rounds, 6 };
        at::result r = at::select_best(6, bb, m.make_hooks());
        hits += (r.best == 3);
        // NAIVE partner: one sample each, take the minimum
        model m2(truth, 5000 + t);
        int best = 0; double bv = 1e30;
        for (int i = 0; i < 6; i++) { double s = m2.sample(i); if (s < bv) { bv = s; best = i; } }
        naive_hits += (best == 3);
    }
    std::fprintf(stderr, "select_best hit rate %.3f, naive one-sample %.3f\n", double(hits) / trials, double(naive_hits) / trials);
    CHECK(hits >= int(0.97 * trials));
    CHECK(naive_hits < int(0.85 * trials));                  // the partner must fail where the machinery passes
    CHECK(hits > naive_hits + int(0.10 * trials));
}

// samples a 3-stage halving run takes: stage s times every active candidate min(rounds[s], max_rounds) times, then keeps keep[s]
static int expected_samples(int n_cands, int max_rounds) {
    static const int keep[3] = { 6, 3, 1 }, rounds[3] = { 2, 8, 14 };
    int active = n_cands, total = 0;
    for (int s = 0; s < 3 && active > 1; s++) { total += active * std::min(rounds[s], max_rounds); active = std::min(keep[s], active); }
    return total;
}

static void test_select_contracts() {
    // n == 1: no timing at all
    { model m({ 2.0 }, 1); at::result r = at::select_best(1, at::budget_for(1), m.make_hooks()); CHECK(r.best == 0 && r.samples == 0 && m.calls == 0); }
    // n == 0 / no callback
    { model m({}, 1); CHECK(at::select_best(0, at::budget_for(1), m.make_hooks()).best == -1); }
    { at::hooks h; CHECK(at::select_best(3, at::budget_for(1), h).best == -1); }
    // max_cands caps what is even considered: slots >= cap are never launched
    {
        model m({ 5, 4, 3, 2, 1, 0.5, 0.1 }, 7);
        at::budget b { 2, 12, 3 };
        at::result r = at::select_best(7, b, m.make_hooks());
        CHECK(r.cand_count == 3 && r.best >= 0 && r.best < 3);
        for (int i = 3; i < 7; i++) { CHECK(m.launches[i] == 0); }
    }
    // one warm-up per candidate, a flush before EVERY timed sample, sample count within the 3-stage bound
    {
        model m({ 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6 }, 11);
        at::budget b { 3, 32, 7 };
        at::result r = at::select_best(7, b, m.make_hooks());
        CHECK(m.warmups == 7);
        CHECK(m.flushes == r.samples && r.samples == m.calls);
        CHECK(r.samples == expected_samples(7, 32));
        CHECK(r.best == 0);
    }
    // max_rounds clamps the stage round counts
    {
        model m({ 1.0, 1.2, 1.4, 1.6 }, 12);
        at::budget b { 1, 2, 4 };
        at::result r = at::select_best(4, b, m.make_hooks());
        CHECK(r.samples == expected_samples(4, 2) && r.samples == 22);
    }
    // first-launch penalty: without warm-up the candidate that happens to be first can be mis-ranked; with it the truth wins
    int ok_warm = 0, ok_cold = 0, trials = 300;
    for (int t = 0; t < trials; t++) {
        std::vector<double> truth = { 0.90, 1.00 };      // tiny budget (1 round/stage): the quartile trim cannot hide the penalty sample
        { model m(truth, 90000 + t); m.noise = 0.01; m.spike_p = 0; m.lucky_p = 0; m.first_pen = { 1.0, 0.0 };
          ok_warm += at::select_best(2, { 1, 1, 2 }, m.make_hooks(true, true)).best == 0; }
        { model m(truth, 90000 + t); m.noise = 0.01; m.spike_p = 0; m.lucky_p = 0; m.first_pen = { 1.0, 0.0 };
          ok_cold += at::select_best(2, { 1, 1, 2 }, m.make_hooks(false, true)).best == 0; }
    }
    std::fprintf(stderr, "first-launch penalty: warm %d/%d cold %d/%d\n", ok_warm, trials, ok_cold, trials);
    CHECK(ok_warm == trials);
    CHECK(ok_cold == 0);                                     // without the warm-up the first (penalised) sample sinks the true winner
}

// ---- keys and cache ----------------------------------------------------------------------------------------------

static std::string tmp_path(const char * name) {
    const char * d = std::getenv("TEMP");
    if (!d) d = std::getenv("TMPDIR");
    if (!d) d = ".";
    return std::string(d) + "/" + name;
}

static void put(std::vector<unsigned char> & img, const void * p, size_t n) {
    const unsigned char * c = static_cast<const unsigned char *>(p);
    img.insert(img.end(), c, c + n);
}
static at::record mk(uint64_t k, int32_t c) { return { k, c, at::record_check(k, c) }; }

static void test_keys() {
    const uint64_t a = at::key_builder(1).add(4096).add(11008).add(32).done();
    CHECK(a == at::key_builder(1).add(4096).add(11008).add(32).done());          // deterministic
    CHECK(a != at::key_builder(2).add(4096).add(11008).add(32).done());          // salted: a kernel change invalidates the cache
    CHECK(a != at::key_builder(1).add(11008).add(4096).add(32).done());          // field order matters
    CHECK(a != at::key_builder(1).add(4096).add(11008).add(33).done());
    CHECK(at::key_builder(1).add(std::string("q4_0")).done() != at::key_builder(1).add(std::string("q4_1")).done());
    CHECK(at::key_builder(1).add(std::string("ab")).add(std::string("c")).done() != at::key_builder(1).add(std::string("a")).add(std::string("bc")).done());
    CHECK(at::fnv1a64("", 0) == 14695981039346656037ull);
}

static void test_cache_parse() {
    std::map<uint64_t, int32_t> m;
    std::vector<unsigned char> img;
    const at::file_header h = at::make_header();
    put(img, &h, sizeof(h));
    { auto r = mk(1, 64); put(img, &r, sizeof(r)); }
    { auto r = mk(2, 32); put(img, &r, sizeof(r)); }
    int sk = -1, bad = -1;
    CHECK(at::parse(img, m, &sk, &bad) && m.size() == 2 && m[1] == 64 && m[2] == 32 && sk == 0 && bad == 0);

    // second header in the middle (two first writers): tolerated, both sides' records kept
    put(img, &h, sizeof(h));
    { auto r = mk(3, 16); put(img, &r, sizeof(r)); }
    m.clear();
    CHECK(at::parse(img, m, &sk, &bad) && m.size() == 3 && m[3] == 16 && sk == 1);

    // later record for the same key wins
    { auto r = mk(1, 96); put(img, &r, sizeof(r)); }
    m.clear();
    CHECK(at::parse(img, m) && m[1] == 96);

    // torn record (check mismatch) is dropped, the rest survive
    { at::record r = mk(9, 8); r.check ^= 0x5a5a; put(img, &r, sizeof(r)); }
    m.clear();
    CHECK(at::parse(img, m, &sk, &bad) && m.find(9) == m.end() && bad == 1 && m.size() == 3);

    // truncated tail (partial record) ignored
    { unsigned char junk[7] = { 1, 2, 3, 4, 5, 6, 7 }; put(img, junk, sizeof(junk)); }
    m.clear();
    CHECK(at::parse(img, m) && m.size() == 3);

    // foreign magic / wrong format / wrong record size / empty / too short => not our file
    { std::vector<unsigned char> f(64, 0); m.clear(); CHECK(!at::parse(f, m) && m.empty()); }
    { std::vector<unsigned char> f; m.clear(); CHECK(!at::parse(f, m)); }
    { at::file_header g = h; g.format = 99; std::vector<unsigned char> f; put(f, &g, sizeof(g)); auto r = mk(1, 1); put(f, &r, sizeof(r)); m.clear(); CHECK(!at::parse(f, m) && m.empty()); }
    { at::file_header g = h; g.record_size = 24; std::vector<unsigned char> f; put(f, &g, sizeof(g)); auto r = mk(1, 1); put(f, &r, sizeof(r)); m.clear(); CHECK(!at::parse(f, m) && m.empty()); }
    { std::vector<unsigned char> f; put(f, &h, 5); m.clear(); CHECK(!at::parse(f, m)); }
}

static void test_cache_file() {
    const std::string p = tmp_path("ggml_autotune_test.bin");
    std::remove(p.c_str());
    {
        at::cache c(p);
        CHECK(c.size() == 0);
        c.store(11, 64); c.store(12, 128); c.store(11, 72);
    }
    {
        at::cache c(p);
        int v = 0;
        CHECK(c.size() == 2);
        CHECK(c.lookup(11, { 64, 72, 128 }, v) && v == 72);                 // latest write for the key
        CHECK(c.lookup(12, { 64, 72, 128 }, v) && v == 128);
        CHECK(!c.lookup(12, { 64, 72 }, v));                                // candidate no longer valid -> not trusted (N8)
        CHECK(!c.lookup(13, { 64 }, v));
    }
    // file size is header + 3 records
    CHECK(at::read_all(p).size() == 16 + 3 * 16);
    // a corrupted byte inside a record is dropped on load, others remain
    {
        std::vector<unsigned char> img = at::read_all(p);
        img[16 + 16 + 9] ^= 0xff;                                           // record #2 payload
        FILE * f = std::fopen(p.c_str(), "wb"); std::fwrite(img.data(), 1, img.size(), f); std::fclose(f);
        at::cache c(p);
        CHECK(c.records_dropped == 1 && c.size() == 1);
    }
    // concurrent first writers: every record from every thread must be recoverable, however many headers got written
    std::remove(p.c_str());
    {
        std::vector<std::thread> th;
        for (int t = 0; t < 4; t++) {
            th.emplace_back([&, t]() { for (int i = 0; i < 100; i++) { at::append_record(p, uint64_t(t * 1000 + i + 1), int32_t(8 * (i % 16 + 1))); } });
        }
        for (auto & x : th) { x.join(); }
        at::cache c(p);
        CHECK(c.size() == 400);
        CHECK(c.records_dropped == 0);
        std::fprintf(stderr, "concurrent writers: %zu records, %d embedded header(s) tolerated\n", c.size(), c.headers_skipped);
    }
    std::remove(p.c_str());
    // an empty path is a pure in-memory cache
    { at::cache c(""); c.store(5, 8); int v; CHECK(c.lookup(5, { 8 }, v) && v == 8); }
}

// ---- roofline early stop (row S69-ENGREV-BURN-ROOFLINE-STOP-2026-10-27-01) -----------------------------------------

static void test_roofline_math() {
    const at::device_peaks p { 1e13, 5e11, 0.005 };
    // bandwidth bound: 1 GB at 500 GB/s = 2 ms (+0.005 launch); compute trivially small
    CHECK(std::fabs(at::roofline_min_ms({ 1e6, 1e9 }, p) - 2.005) < 1e-9);
    // compute bound: 1e12 flop at 1e13 = 100 ms (+0.005)
    CHECK(std::fabs(at::roofline_min_ms({ 1e12, 1e3 }, p) - 100.005) < 1e-9);
    // unusable peaks / negative work => 0 (never a bound)
    CHECK(at::roofline_min_ms({ 1e6, 1e6 }, { 0.0, 5e11, 0.0 }) == 0.0);
    CHECK(at::roofline_min_ms({ 1e6, 1e6 }, { 1e13, 0.0, 0.0 }) == 0.0);
    CHECK(at::roofline_min_ms({ -1.0, 1e6 }, p) == 0.0);
    // 14 Gbps-class card: 7001000 kHz mem clock, 256-bit bus => 2*7.001e9*32 = 448 GB/s
    CHECK(std::fabs(at::peak_bytes_per_s(7001000, 256) - 448.064e9) < 1e6);
    CHECK(at::peak_bytes_per_s(0, 256) == 0.0 && at::peak_bytes_per_s(7001000, 0) == 0.0);
    CHECK(std::fabs(at::peak_flops_per_s(10, 1000000, 128.0) - 1.28e12) < 1.0);
    CHECK(at::peak_flops_per_s(0, 1000000, 128.0) == 0.0);
    // levels: full registers nothing; unknown spelling must not switch the search off
    CHECK(at::parse_level(nullptr) == at::level::full && at::parse_level("bogus") == at::level::full);
    CHECK(at::parse_level("balanced") == at::level::balanced && at::parse_level("fast") == at::level::fast);
    CHECK(at::accept_fraction(at::level::full) == 0.0);
    CHECK(at::accept_fraction(at::level::fast) < at::accept_fraction(at::level::balanced));
    CHECK(!at::make_bound({ 1e6, 1e9 }, p, at::level::full).active());
    CHECK(at::make_bound({ 1e6, 1e9 }, p, at::level::balanced).active());
    CHECK(!at::make_bound({ 1e6, 1e9 }, { 0.0, 0.0, 0.0 }, at::level::balanced).active());
}

// quiet model: no spikes / lucky samples, low noise -> deterministic enough to assert on
static model quiet_model(std::vector<double> t, uint64_t seed) {
    model m(std::move(t), seed);
    m.noise = 0.01; m.spike_p = 0.0; m.lucky_p = 0.0;
    return m;
}

static void test_roofline_stop() {
    const at::device_peaks p { 1e13, 5e11, 0.005 };
    const at::budget b { 5, 12, 6 };
    const int full_samples = expected_samples(6, 12);

    // a bound that is NOT passed, and a `full`-level bound, are identical to the old behaviour
    {
        model m1 = quiet_model({ 1.3, 1.5, 1.2, 1.9, 1.4, 1.6 }, 7);
        at::result r1 = at::select_best(6, b, m1.make_hooks());
        model m2 = quiet_model({ 1.3, 1.5, 1.2, 1.9, 1.4, 1.6 }, 7);
        const at::bound off = at::make_bound({ 1e6, 1e9 }, p, at::level::full);
        at::result r2 = at::select_best(6, b, m2.make_hooks(), &off);
        CHECK(r1.best == r2.best && r1.samples == r2.samples && r1.samples == full_samples);
        CHECK(!r1.early_stop && !r2.early_stop && r1.stages == 3 && r2.stages == 3);
    }
    // best candidate (idx 2) already at ~96% of the bound: stop after stage 0 with far fewer samples
    {
        const double tmin = at::roofline_min_ms({ 1e6, 1e9 }, p);
        const at::bound bd = at::make_bound({ 1e6, 1e9 }, p, at::level::balanced);
        CHECK(std::fabs(bd.t_min_ms - tmin) < 1e-12);
        model m = quiet_model({ tmin * 1.6, tmin * 1.5, tmin / 0.96, tmin * 1.9, tmin * 1.4, tmin * 1.7 }, 11);
        at::result r = at::select_best(6, b, m.make_hooks(), &bd);
        CHECK(r.early_stop && r.stages == 1 && r.best == 2);
        CHECK(r.samples < full_samples && r.samples == 6 * 2);
        CHECK(r.frac > 0.90 && r.frac <= 1.0);
    }
    // best candidate far from the bound (60%): the full search runs, nothing is cut
    {
        const double tmin = at::roofline_min_ms({ 1e6, 1e9 }, p);
        const at::bound bd = at::make_bound({ 1e6, 1e9 }, p, at::level::balanced);
        model m = quiet_model({ tmin * 2.6, tmin * 2.5, tmin / 0.60, tmin * 2.9, tmin * 2.4, tmin * 2.7 }, 13);
        at::result r = at::select_best(6, b, m.make_hooks(), &bd);
        CHECK(!r.early_stop && r.stages == 3 && r.best == 2 && r.samples == full_samples);
        CHECK(r.frac > 0.55 && r.frac < 0.70);
    }
    // a reading 4x BELOW the bound means the peaks are wrong: not trusted as a reason to stop
    {
        const double tmin = at::roofline_min_ms({ 1e6, 1e9 }, p);
        const at::bound bd = at::make_bound({ 1e6, 1e9 }, p, at::level::balanced);
        model m = quiet_model({ tmin / 4.0, tmin / 3.5, tmin / 3.0, tmin / 3.2 }, 17);
        at::result r = at::select_best(4, b, m.make_hooks(), &bd);
        CHECK(!r.early_stop && r.stages == 3 && r.best == 0);
    }
}

// >=3 shapes (decode 1xKxN, prefill 512xKxN, MoE expert): rounds drop on >=1 shape, chosen within the threshold of the exhaustive best on all
static void test_roofline_shapes() {
    const at::device_peaks p { 4e13, 6e11, 0.004 };
    struct shape { const char * name; at::roofline_cost c; double best_frac; };
    const double K = 4096, N = 4096, wb = 0.5625;                       // q4-class weights
    const shape shapes[3] = {
        { "decode 1xKxN",    { 2 * K * N,       K * N * wb + (K + N) * 4 },            0.93 },   // bandwidth bound, static pick near peak
        { "prefill 512xKxN", { 2 * 512 * K * N, K * N * wb + 512 * (K + N) * 4.0 },    0.62 },   // compute bound, real headroom
        { "moe expert",      { 2 * 64 * K * 1536.0, K * 1536.0 * wb + 64 * (K + 1536) * 4.0 }, 0.90 },
    };
    const at::level lv = at::level::balanced;
    const double thr = at::accept_fraction(lv);
    int shapes_with_fewer = 0;
    for (const shape & s : shapes) {
        const at::bound bd = at::make_bound(s.c, p, lv);
        const double tmin = bd.t_min_ms;
        const std::vector<double> truth = { tmin / s.best_frac * 1.35, tmin / s.best_frac * 1.20, tmin / s.best_frac,
                                            tmin / s.best_frac * 1.45, tmin / s.best_frac * 1.10, tmin / s.best_frac * 1.60 };
        model mf = quiet_model(truth, 21);
        at::result rf = at::select_best(6, { 5, 12, 6 }, mf.make_hooks());           // exhaustive: no bound
        model mb = quiet_model(truth, 21);
        at::result rb = at::select_best(6, { 5, 12, 6 }, mb.make_hooks(), &bd);      // bounded
        std::fprintf(stderr, "%-16s exhaustive rounds %3d best %.5f ms | bounded rounds %3d best %.5f ms frac %.2f stop=%d\n",
                     s.name, rf.samples, truth[rf.best], rb.samples, truth[rb.best], rb.frac, int(rb.early_stop));
        CHECK(rb.samples <= rf.samples);
        shapes_with_fewer += rb.samples < rf.samples;
        CHECK(truth[rb.best] <= truth[rf.best] / thr + 1e-12);                      // within the threshold of the exhaustive best
    }
    CHECK(shapes_with_fewer >= 1);
}

int main() {
    test_roofline_math();
    test_roofline_stop();
    test_roofline_shapes();
    test_budget();
    test_visit_order();
    test_trimmed_mean();
    test_expand_sm();
    test_select();
    test_select_contracts();
    test_keys();
    test_cache_parse();
    test_cache_file();
    if (g_fail) {
        std::fprintf(stderr, "test-cuda-autotune: %d FAILED\n", g_fail);
        return 1;
    }
    std::printf("test-cuda-autotune: PASS\n");
    return 0;
}
