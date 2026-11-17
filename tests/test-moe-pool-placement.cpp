// test-moe-pool-placement.cpp -- S69-KT-EXPERT-POOL-2026-09-28
// Unit tests for src/llama-moe-pool.{h,cpp}: recorder, placement strategies, VRAM rail, and the
// host-side LRU pool bookkeeping. No ggml/model dependency -- these are pure host-side controls.
// ANTI-BIAS discipline: each check below is chosen so it would read out DIFFERENTLY if the piece
// under test were a stub (e.g. a strategy silently aliasing uniform, or a pool that always "hits").

#include "../src/llama-moe-pool.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <random>
#include <set>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

static int g_fail = 0;
#define CHECK(cond) do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            g_fail++; \
        } \
    } while (0)

// ───────────────────────── seeded-ids print + not-equal-to-uniform control ─────────────────────────
// "A seeded pool whose contents equal uniform's must be caught (print the seeded expert ids per
// layer for 2 layers per strategy)." -- task brief, TESTS.
static void test_print_and_distinguish_from_uniform() {
    const int32_t n_layers = 8, n_expert = 32, n_slots = 6;
    std::vector<uint64_t> freq(n_expert, 0);
    // Skewed synthetic activation frequency: experts 3,7,11,19 dominate.
    for (int32_t e = 0; e < n_expert; ++e) freq[e] = 1;
    freq[3] = 500; freq[7] = 400; freq[11] = 300; freq[19] = 200;

    const char * names[4] = {"uniform", "frequency", "front-loading", "random"};
    llama_moe_placement_strategy strategies[4] = {
        LLAMA_MOE_PLACEMENT_UNIFORM, LLAMA_MOE_PLACEMENT_FREQUENCY,
        LLAMA_MOE_PLACEMENT_FRONT_LOADING, LLAMA_MOE_PLACEMENT_RANDOM,
    };

    std::vector<int32_t> layers_to_print = {0, 1};
    std::vector<std::vector<int32_t>> uniform_ids(n_layers);
    for (int32_t il = 0; il < n_layers; ++il) {
        uniform_ids[il] = llama_moe_seed_layer(LLAMA_MOE_PLACEMENT_UNIFORM, il, n_layers, n_expert, n_slots, {}, 42);
    }

    for (int s = 0; s < 4; ++s) {
        for (int32_t il : layers_to_print) {
            auto ids = llama_moe_seed_layer(strategies[s], il, n_layers, n_expert, n_slots, freq, 42);
            fprintf(stdout, "[moe-pool-test] strategy=%-13s layer=%d slots=%d ids={", names[s], il, n_slots);
            for (size_t i = 0; i < ids.size(); ++i) fprintf(stdout, "%s%d", i ? "," : "", ids[i]);
            fprintf(stdout, "}\n");
            CHECK((int32_t) ids.size() == n_slots);
            for (int32_t id : ids) CHECK(id >= 0 && id < n_expert);
            std::set<int32_t> dedup(ids.begin(), ids.end());
            CHECK(dedup.size() == ids.size()); // no duplicate slots
        }
    }

    // The actual catch: frequency and front-loading must NOT silently equal uniform once there is
    // real skew / a nonzero layer index to distinguish them.
    auto freq_ids0  = llama_moe_seed_layer(LLAMA_MOE_PLACEMENT_FREQUENCY, 0, n_layers, n_expert, n_slots, freq, 42);
    CHECK(freq_ids0 != uniform_ids[0]);
    auto fl_ids1 = llama_moe_seed_layer(LLAMA_MOE_PLACEMENT_FRONT_LOADING, 1, n_layers, n_expert, n_slots, {}, 42);
    CHECK(fl_ids1 != uniform_ids[1]);
    auto rand_ids0 = llama_moe_seed_layer(LLAMA_MOE_PLACEMENT_RANDOM, 0, n_layers, n_expert, n_slots, {}, 42);
    // random CAN coincide with uniform by chance for a small slot count; assert only that random
    // differs from uniform on AT LEAST ONE of several layers (the real anti-bias claim: it is not
    // a hardcoded alias of uniform's code path).
    bool any_diff = false;
    for (int32_t il = 0; il < n_layers; ++il) {
        auto r = llama_moe_seed_layer(LLAMA_MOE_PLACEMENT_RANDOM, il, n_layers, n_expert, n_slots, {}, 42);
        if (r != uniform_ids[il]) { any_diff = true; break; }
    }
    CHECK(any_diff);
    (void) rand_ids0;

    // Frequency top-4 experts (3,7,11,19) must all be selected given they dominate and n_slots>=4.
    std::set<int32_t> freq_set(freq_ids0.begin(), freq_ids0.end());
    CHECK(freq_set.count(3) && freq_set.count(7) && freq_set.count(11) && freq_set.count(19));

    // Frequency with an all-zero / empty freq_row must fall back to uniform (a documented,
    // intentional fallback -- and this IS the one case where equality to uniform is correct).
    auto freq_fallback = llama_moe_seed_layer(LLAMA_MOE_PLACEMENT_FREQUENCY, 2, n_layers, n_expert, n_slots, {}, 42);
    CHECK(freq_fallback == uniform_ids[2]);
}

// ───────────────────────── recorder round-trip ─────────────────────────
static void test_recorder_json_roundtrip() {
    llama_moe_recorder rec(LLAMA_MOE_RECORDER_STAT);
    int32_t ids_l0[] = {3, 3, 7, 19};
    int32_t ids_l1[] = {1, 1, 1, 2};
    rec.record(0, ids_l0, 4);
    rec.record(1, ids_l1, 4);
    rec.record(0, ids_l0, 4);

    const char * path = "test_moe_pool_recorder.json";
    CHECK(rec.write_json(path));

    std::unordered_map<int32_t, std::unordered_map<int32_t, uint64_t>> loaded;
    CHECK(llama_moe_recorder::read_json(path, loaded));
    CHECK(loaded[0][3] == 4);   // two records of {3,3,7,19}: expert 3 seen twice per call -> 4
    CHECK(loaded[0][7] == 2);
    CHECK(loaded[0][19] == 2);
    CHECK(loaded[1][1] == 3);
    CHECK(loaded[1][2] == 1);
    remove(path);
}

// ───────────────────────── recorder write is atomic and reports failure ─────────────────────────
// S69-FORK-MOE-POOL-RECORDER-NONATOMIC-JSON-WRITE-2026-11-03-01. write_json used to write the
// target in place and return true regardless of fclose(). These checks fail on that version:
//  - overwrite: a second write must REPLACE an existing file (a plain rename() fails on Windows
//    when the target exists; the in-place version passes this one, the cases below it do not);
//  - no temp file may survive a successful write;
//  - unwritable location (missing parent dir) -> false, nothing created;
//  - target is a directory (temp writes fine, the replace step fails) -> false, temp removed.
static unsigned long test_pid() {
#ifdef _WIN32
    return (unsigned long) GetCurrentProcessId();
#else
    return (unsigned long) getpid();
#endif
}

static size_t count_tmp_leftovers(const std::filesystem::path & dir) {
    size_t n = 0;
    std::error_code ec;
    for (auto & e : std::filesystem::directory_iterator(dir, ec)) {
        if (e.path().filename().string().find(".tmp.") != std::string::npos) n++;
    }
    return n;
}

static void test_recorder_json_atomic_write() {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path(ec) / "test_moe_pool_atomic_dir";
    fs::remove_all(dir, ec);
    CHECK(fs::create_directories(dir, ec));
    const std::string path = (dir / "profile.json").string();

    // 1) overwrite an existing file with different content; reader must see ONLY the new content
    llama_moe_recorder a(LLAMA_MOE_RECORDER_STAT);
    int32_t ids_a[] = {5, 5, 5};
    a.record(0, ids_a, 3);
    CHECK(a.write_json(path));
    llama_moe_recorder b(LLAMA_MOE_RECORDER_STAT);
    int32_t ids_b[] = {9, 2};
    b.record(4, ids_b, 2);
    CHECK(b.write_json(path));
    std::unordered_map<int32_t, std::unordered_map<int32_t, uint64_t>> got;
    CHECK(llama_moe_recorder::read_json(path, got));
    CHECK(got.size() == 1 && got.count(4) == 1);
    CHECK(got[4][9] == 1 && got[4][2] == 1);
    CHECK(got.count(0) == 0); // old content gone, not merged/appended

    // 2) success leaves no temp behind
    CHECK(count_tmp_leftovers(dir) == 0);

    // 3) missing parent directory -> false, and nothing created there
    const fs::path missing = dir / "no_such_subdir" / "profile.json";
    CHECK(!b.write_json(missing.string()));
    CHECK(!fs::exists(missing, ec));
    CHECK(!fs::exists(dir / "no_such_subdir", ec));

    // 4) target is an existing directory: temp write succeeds, replace fails -> false + cleanup
    const fs::path as_dir = dir / "is_a_dir.json";
    CHECK(fs::create_directory(as_dir, ec));
    CHECK(!b.write_json(as_dir.string()));
    CHECK(fs::is_directory(as_dir, ec));     // target untouched
    CHECK(count_tmp_leftovers(dir) == 0);    // temp removed on the failure path

    // 5) DISCRIMINATING: a failed write must leave the previous good file intact. Block the temp
    //    name with a directory so the write cannot complete. The in-place writer opens the target
    //    directly, overwrites it and returns true (both checks below fail on it); the atomic
    //    writer returns false and the reader still sees the earlier content.
    const fs::path keep = dir / "keep.json";
    CHECK(a.write_json(keep.string()));
    const fs::path blocker = fs::path(keep.string() + ".tmp." + std::to_string(test_pid()));
    CHECK(fs::create_directory(blocker, ec));
    CHECK(!b.write_json(keep.string()));
    std::unordered_map<int32_t, std::unordered_map<int32_t, uint64_t>> kept;
    CHECK(llama_moe_recorder::read_json(keep.string(), kept));
    CHECK(kept.count(0) == 1 && kept[0][5] == 3 && kept.count(4) == 0);
    fs::remove(blocker, ec);

    fs::remove_all(dir, ec);
}

// ───────────────────────── frequency vs random, held-out (leakage check) ─────────────────────────
// "frequency-seeded must beat random on a recorded-then-held-out text (calibrate on one text,
// measure on another - leakage check)." No real routing trace was available offline this
// session, so this uses two independent synthetic draws from the SAME skewed distribution (a
// few hot experts dominate, matching the routing-skew property PR #27861's own measurement
// section reports: "Strong temporal locality... LRU-64 would hit ~67%") with different RNG
// streams standing in for "calibration text" vs "held-out text" -- labeled synthetic below, not
// presented as a real-workload result.
static std::vector<int32_t> draw_skewed_trace(uint32_t seed, int32_t n_expert, int32_t n_events, int32_t n_hot) {
    std::mt19937 rng(seed);
    // hot experts get 80% of mass, the rest share 20% -- synthetic stand-in for real routing skew.
    std::vector<int32_t> hot(n_hot);
    std::mt19937 hot_rng(1234); // same hot set for both traces (same underlying distribution)
    std::vector<int32_t> pool(n_expert);
    for (int32_t e = 0; e < n_expert; ++e) pool[e] = e;
    std::shuffle(pool.begin(), pool.end(), hot_rng);
    for (int32_t i = 0; i < n_hot; ++i) hot[i] = pool[i];

    std::uniform_real_distribution<double> coin(0.0, 1.0);
    std::uniform_int_distribution<int32_t> pick_hot(0, n_hot - 1);
    std::uniform_int_distribution<int32_t> pick_any(0, n_expert - 1);
    std::vector<int32_t> trace;
    trace.reserve(n_events);
    for (int32_t i = 0; i < n_events; ++i) {
        trace.push_back(coin(rng) < 0.8 ? hot[pick_hot(rng)] : pick_any(rng));
    }
    return trace;
}

static void test_frequency_beats_random_held_out() {
    const int32_t n_layers = 1, n_expert = 64, n_slots = 8, n_hot = 6;
    auto calib_trace   = draw_skewed_trace(/*seed*/ 111, n_expert, 4000, n_hot);
    auto heldout_trace = draw_skewed_trace(/*seed*/ 222, n_expert, 4000, n_hot); // different draw, same distribution

    llama_moe_recorder rec(LLAMA_MOE_RECORDER_STAT);
    for (int32_t id : calib_trace) {
        rec.record(0, &id, 1);
    }
    auto dense = rec.to_dense(n_layers, n_expert);

    auto freq_ids   = llama_moe_seed_layer(LLAMA_MOE_PLACEMENT_FREQUENCY, 0, n_layers, n_expert, n_slots, dense[0], 42);
    auto random_ids = llama_moe_seed_layer(LLAMA_MOE_PLACEMENT_RANDOM,    0, n_layers, n_expert, n_slots, {},       42);

    std::set<int32_t> freq_set(freq_ids.begin(), freq_ids.end());
    std::set<int32_t> random_set(random_ids.begin(), random_ids.end());

    int64_t freq_hits = 0, random_hits = 0;
    for (int32_t id : heldout_trace) {
        if (freq_set.count(id))   freq_hits++;
        if (random_set.count(id)) random_hits++;
    }
    double freq_rate   = (double) freq_hits   / (double) heldout_trace.size();
    double random_rate = (double) random_hits / (double) heldout_trace.size();
    fprintf(stdout, "[moe-pool-test] held-out hit rate: frequency=%.4f random=%.4f (n_slots=%d of %d experts)\n",
            freq_rate, random_rate, n_slots, n_expert);
    CHECK(freq_rate > random_rate);
    // Not a razor-thin win: the skew is strong (80% mass on 6/64 experts) and the calibration set
    // is large, so a real advantage should be substantial, not noise-level.
    CHECK(freq_rate > random_rate + 0.15);
}

// ───────────────────────── VRAM rail ─────────────────────────
static void test_vram_rail() {
    const int64_t GiB = int64_t(1) << 30;
    // 10 GiB free, 6 GiB reserve -> 4 GiB usable; 100 MiB/slot -> ~40 slots affordable.
    int64_t slots = llama_moe_pool_vram_budget_slots(10 * GiB, LLAMA_MOE_POOL_DEFAULT_RESERVE_BYTES, 100 * (int64_t(1) << 20), 1000);
    CHECK(slots > 0 && slots < 1000); // rail actually LOWERED the ceiling
    CHECK(slots == (4 * GiB) / (100 * (int64_t(1) << 20)));

    // Free memory below the reserve -> 0 slots, never negative.
    CHECK(llama_moe_pool_vram_budget_slots(4 * GiB, LLAMA_MOE_POOL_DEFAULT_RESERVE_BYTES, 100 * (int64_t(1) << 20), 1000) == 0);

    // Plenty of VRAM -> the requested ceiling wins, rail never INCREASES past what was asked.
    CHECK(llama_moe_pool_vram_budget_slots(200 * GiB, LLAMA_MOE_POOL_DEFAULT_RESERVE_BYTES, 1 * (int64_t(1) << 20), 32) == 32);
}

// ───────────────────────── LRU pool bookkeeping ─────────────────────────
static void test_lru_pool_step() {
    llama_moe_pool_layer layer;
    layer.init(/*n_expert*/ 16, /*n_slots*/ 4, /*admit_threshold*/ 1); // unconditional admission
    layer.seed({0, 1, 2, 3});
    CHECK(layer.is_resident(0) && layer.is_resident(3) && !layer.is_resident(4));

    // Step with ids all resident -> all hits, no admits/evicts.
    int32_t ids_hit[] = {0, 1, 2, 3};
    auto st1 = layer.step(ids_hit, 4, /*max_inserts*/ 2);
    CHECK(st1.hits == 4 && st1.misses == 0 && st1.admitted == 0 && st1.evicted == 0);

    // Step with a brand-new id -> a miss, then (threshold=1) admitted, evicting the LRU victim.
    int32_t ids_new[] = {9};
    auto st2 = layer.step(ids_new, 1, /*max_inserts*/ 2);
    CHECK(st2.misses == 1 && st2.admitted == 1 && st2.evicted == 1);
    CHECK(layer.is_resident(9));
    CHECK(!layer.is_resident(0)); // id 0 was the LRU victim (touched first, so lowest recency, in st1)

    // Admission gate: threshold=3 means the first two misses of a new id must NOT be admitted.
    llama_moe_pool_layer gated;
    gated.init(16, 4, /*admit_threshold*/ 3);
    gated.seed({0, 1, 2, 3});
    int32_t id5[] = {5};
    auto g1 = gated.step(id5, 1, 2); CHECK(g1.misses == 1 && g1.admitted == 0);
    auto g2 = gated.step(id5, 1, 2); CHECK(g2.misses == 1 && g2.admitted == 0);
    auto g3 = gated.step(id5, 1, 2); CHECK(g3.misses == 1 && g3.admitted == 1);
    CHECK(gated.is_resident(5));

    // Insert throttle: max_inserts caps admissions in a single step even with several new ids.
    llama_moe_pool_layer throttled;
    throttled.init(16, 8, /*admit_threshold*/ 1);
    throttled.seed({0, 1, 2, 3, 4, 5, 6, 7});
    int32_t many_new[] = {20, 21, 22, 23};
    auto st3 = throttled.step(many_new, 4, /*max_inserts*/ 2);
    CHECK(st3.misses == 4 && st3.admitted == 2); // throttled to 2 even though 4 were eligible
}

// ───────────────────────── cost-gated, aged, single-admission policy ─────────────────────────
// S69-FASTLLM-PICK-COST-GATED-EXPERT-ADMISSION-2026-09-29-01. Vectors are hand-derived from the policy text in llama-moe-pool.h.
// GATE-CLAIM-OK: cpu_us/gpu_us are injected timing numbers of a unit test; no resource state is claimed or probed here.
static void test_gated_policy() {
    // T1 cost gate: admit only when (cpu - gpu) > refill * 1.1.  refill 750 us -> threshold 825 us.
    {
        llama_moe_pool_layer a;
        a.init(16, 2, 1);
        a.seed({0, 1});
        llama_moe_pool_cost c; c.valid = true; c.refill_us = 750.0; c.cpu_us = 1000.0; c.gpu_us = 500.0;   // gap 500 < 825
        int32_t id[] = {7};
        auto s = a.step_gated(id, 1, 2, c);
        CHECK(s.misses == 1 && s.admitted == 0 && s.gated == 1 && !a.is_resident(7));
        c.cpu_us = 2000.0;                                                                                    // gap 1500 > 825
        s = a.step_gated(id, 1, 2, c);
        CHECK(s.admitted == 1 && s.gated == 0 && a.is_resident(7));
        // control: the old step() has no timing input and admits on the first miss in BOTH cases (this is the fail-before)
        llama_moe_pool_layer o;
        o.init(16, 2, 1);
        o.seed({0, 1});
        auto os = o.step(id, 1, 2);
        CHECK(os.admitted == 1 && o.is_resident(7));
    }
    // T2 aging: threshold 3, window 4 layers * 64 = 256 calls.  Expert 5 seen twice, 10000 calls that only hit the resident expert 9, then 5 once more.
    {
        llama_moe_pool_layer a;
        a.init(16, 1, 3);
        a.age_window = 256;
        a.seed({9});
        llama_moe_pool_cost none;
        int32_t five[] = {5}, nine[] = {9};
        a.step_gated(five, 1, 1, none);
        a.step_gated(five, 1, 1, none);
        for (int i = 0; i < 10000; ++i) a.step_gated(nine, 1, 1, none);
        auto s = a.step_gated(five, 1, 1, none);
        CHECK(s.admitted == 0 && !a.is_resident(5) && a.is_resident(9));          // heat restarted at 1 < 3
        // control: the same three sightings inside the window ARE admitted
        llama_moe_pool_layer b;
        b.init(16, 1, 3);
        b.age_window = 256;
        b.seed({9});
        b.step_gated(five, 1, 1, none);
        b.step_gated(five, 1, 1, none);
        s = b.step_gated(five, 1, 1, none);
        CHECK(s.admitted == 1 && b.is_resident(5));
        // and with aging off (window 0) the old forever-counter admits after the long gap
        llama_moe_pool_layer c;
        c.init(16, 1, 3);
        c.seed({9});
        c.step_gated(five, 1, 1, none);
        c.step_gated(five, 1, 1, none);
        for (int i = 0; i < 10000; ++i) c.step_gated(nine, 1, 1, none);
        s = c.step_gated(five, 1, 1, none);
        CHECK(s.admitted == 1);
    }
    // T3 single admission per call: three cold experts above the threshold in one call -> exactly one, the hottest, ties lowest id.
    {
        llama_moe_pool_layer a;
        a.init(32, 4, 1);
        a.seed({0, 1, 2, 3});
        llama_moe_pool_cost none;
        int32_t ids[] = {10, 11, 11, 12, 12, 12, 9, 9, 9};                              // heat: 10->1, 11->2, 12->3, 9->3
        auto s = a.step_gated(ids, 9, 8, none);
        CHECK(s.admitted == 1 && s.misses == 9);
        CHECK(a.is_resident(9) && !a.is_resident(12) && !a.is_resident(11) && !a.is_resident(10));   // 9 and 12 tie at 3 -> lowest id
        // control: step() admits all three it can reach (max_inserts 8)
        llama_moe_pool_layer o;
        o.init(32, 4, 1);
        o.seed({0, 1, 2, 3});
        CHECK(o.step(ids, 9, 8).admitted >= 3);
    }
    // T4 a one-shot 2048-token prefill that touches all 64 experts does not displace an expert with 3 prior decode hits
    {
        llama_moe_pool_layer a;
        a.init(64, 4, 3);
        a.seed({0, 1, 2, 3});
        llama_moe_pool_cost none;
        int32_t one[] = {0};
        for (int i = 0; i < 3; ++i) a.step_gated(one, 1, 4, none);
        std::vector<int32_t> prefill;
        for (int e = 4; e < 64; ++e) for (int r = 0; r < 12; ++r) prefill.push_back(e);   // 12 routes each, 2048-token ubatch
        llama_moe_pool_cost pf; pf.n_tokens = 2048;
        auto s = a.step_gated(prefill.data(), (int64_t) prefill.size(), 4, pf);
        CHECK(s.admitted == 0 && a.is_resident(0) && a.is_resident(1) && a.is_resident(2) && a.is_resident(3));
        // control: the same ids through the old step() (each occurrence counts) admit and evict
        llama_moe_pool_layer o;
        o.init(64, 4, 3);
        o.seed({0, 1, 2, 3});
        CHECK(o.step(prefill.data(), (int64_t) prefill.size(), 4).admitted >= 1);
    }
    // T5 visiting stride coprime with the layer count
    CHECK(llama_moe_pool_admit_interval(4, 8) == 5);
    CHECK(llama_moe_pool_admit_interval(6, 6) == 7);
    CHECK(llama_moe_pool_admit_interval(7, 7) == 8);
    CHECK(llama_moe_pool_admit_interval(3, 8) == 3);
    CHECK(llama_moe_pool_admit_interval(0, 5) == 1);
}

int main() {
    test_gated_policy();
    test_print_and_distinguish_from_uniform();
    test_recorder_json_roundtrip();
    test_recorder_json_atomic_write();
    test_frequency_beats_random_held_out();
    test_vram_rail();
    test_lru_pool_step();

    if (g_fail) {
        fprintf(stderr, "test-moe-pool-placement: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    fprintf(stdout, "test-moe-pool-placement: all checks passed\n");
    return 0;
}
