// Row S69-MLC-LLM-PICK-TOPK-TOPP-WITHOUT-FULL-SORT-2026-10-18-01 (engine row S69-ENGINE-MERGE-MLC-LLM-2026-09-09-01).
//
// The legacy host sampler (src/llama-sampling.cpp, used by common/sampling.cpp sampler_queue for every host-sampled
// token) sorted the WHOLE vocabulary for top_k <= 0 (a cut that removes nothing) and again inside top_p's softmax.
// MLC-LLM cpp/serve/sampler/cpu_sampler.cc:72-82 skips the sort when nothing is cut, and :84-162 sorts only the
// tokens with p >= top_p / 1024 (falling back to the full set when they do not reach top_p).
//
// This test pins (1) the result: top_p over an unsorted vocabulary gives the same ids, in the same order, with the
// same probabilities (to float rounding) as the full-sort reference written below, across peaked, flat, tied and small
// vocabularies; (2) its cost on a 152064-token vocabulary, median of 31 interleaved pair ratios, against sort-first: under half when the
// nucleus is a head of tens of tokens, and a ratio above 0.9 on a flat distribution whose nucleus is ~36k tokens
// (MEASURED by this test, numbers in its output); (3) the sampler queue (top_k that cuts nothing, now deferred) leaves
// on unsorted input exactly what it leaves on sorted input, over a parameter grid and with every order-reading sampler
// switched on; (4) the queue's cost, same two bounds. Before the change (2) fails (nosort_before_v1.txt, flat case:
// ratio 1.20 against a float full-sort reference, the unchanged top_p sorting everything) and (3)/(4) do not build
// (no common_sampler_queue_apply).

#include "llama.h"
#include "sampling.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

static std::vector<llama_token_data> make_vocab(size_t n, uint32_t seed, float scale, int n_tied) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<llama_token_data> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = { (llama_token) i, nd(rng) * scale, 0.0f };
    }
    for (int t = 0; t < n_tied && (size_t) t < n; ++t) {
        v[(t * 7919) % n].logit = 3.0f * scale; // exact ties at the top
    }
    return v;
}

// full-sort reference: sort by logit desc, ties by lower id; softmax with the normalising sum in long double (the
// exact value to float precision); smallest prefix with cumulative probability >= p and at least min_keep tokens.
// The pre-change code summed in float, largest first: run 4 of this test measured that sum moving the nucleus boundary
// (n=152064 seed=152163 scale=3 p=0.5: 165 tokens with the float sum, 166 here), so both library paths now sum in double.
static std::vector<llama_token_data> reference_top_p(std::vector<llama_token_data> v, float p, size_t min_keep) {
    std::sort(v.begin(), v.end(), [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
    });
    const float mx = v[0].logit;
    long double sum = 0.0L;
    for (auto & t : v) {
        t.p = expf(t.logit - mx);
        sum += t.p;
    }
    for (auto & t : v) t.p = (float) ((long double) t.p / sum);
    float cum = 0.0f;
    size_t last = v.size();
    for (size_t i = 0; i < v.size(); ++i) {
        cum += v[i].p;
        if (cum >= p && i + 1 >= min_keep) {
            last = i + 1;
            break;
        }
    }
    v.resize(last);
    return v;
}

static double g_worst_vs_exact = 0.0, g_worst_vs_ref = 0.0;

static void check_top_p(size_t n, uint32_t seed, float scale, int n_tied, float p, size_t min_keep) {
    auto v = make_vocab(n, seed, scale, n_tied);
    const auto want = reference_top_p(v, p, min_keep);
    // exact probabilities: the same exp terms summed in long double, over the WHOLE input (taken before top_p rewrites v).
    // The reference's float sum over up to 152064 terms, largest first, carries its own rounding (first run: off by more
    // than 1e-5 relative); the new path sums in double, so it is checked TIGHTLY against the exact value and only loosely
    // against the float reference.
    const float mx = want[0].logit;
    long double exact_sum = 0.0L;
    for (const auto & t : v) {
        exact_sum += (long double) expf(t.logit - mx);
    }
    llama_token_data_array a = { v.data(), v.size(), -1, false };
    llama_sample_top_p(nullptr, &a, p, min_keep);
    if (a.size != want.size()) {
        std::printf("top_p size mismatch n=%zu seed=%u scale=%g tied=%d p=%g min_keep=%zu: got %zu want %zu\n", n, seed,
                    scale, n_tied, p, min_keep, a.size, want.size());
        assert(false);
    }
    assert(a.sorted);
    double worst_vs_exact = 0.0, worst_vs_ref = 0.0, worst_ref_vs_exact = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        assert(a.data[i].id == want[i].id);
        assert(a.data[i].logit == want[i].logit);
        const double exact = (double) ((long double) expf(want[i].logit - mx) / exact_sum);
        worst_vs_exact = std::max(worst_vs_exact, std::fabs(a.data[i].p - exact) / exact);
        worst_vs_ref   = std::max(worst_vs_ref, std::fabs((double) a.data[i].p - want[i].p) / want[i].p);
        worst_ref_vs_exact = std::max(worst_ref_vs_exact, std::fabs(want[i].p - exact) / exact);
    }
    // the prefilter path (double sum) must be within 1e-6 of exact; when it falls back to the full softmax it IS the
    // float reference, so it may be as far from exact as the reference is
    if (worst_vs_exact > std::max(1e-6, worst_ref_vs_exact * 1.01) || worst_vs_ref > 1e-3) {
        std::printf("top_p p mismatch n=%zu seed=%u scale=%g p=%g: worst rel err vs exact %.3g, vs float reference %.3g\n", n,
                    seed, scale, p, worst_vs_exact, worst_vs_ref);
        assert(false);
    }
    g_worst_vs_exact = std::max(g_worst_vs_exact, worst_vs_exact);
    g_worst_vs_ref   = std::max(g_worst_vs_ref, worst_vs_ref);
}

static void test_top_p_matches_reference() {
    const size_t sizes[] = { 3, 64, 1000, 32000, 152064 };
    const float scales[] = { 0.3f, 1.0f, 3.0f, 8.0f }; // flat (falls back) .. peaked
    const float ps[]     = { 0.05f, 0.5f, 0.9f, 0.99f };
    int n = 0;
    for (size_t sz : sizes) {
        for (float sc : scales) {
            for (float p : ps) {
                check_top_p(sz, 1234u + (uint32_t) sz, sc, 0, p, 1);
                check_top_p(sz, 99u + (uint32_t) sz, sc, 5, p, 1);
                n += 2;
            }
        }
    }
    check_top_p(32000, 7, 3.0f, 0, 0.05f, 40); // min_keep beats the nucleus
    check_top_p(32000, 8, 0.1f, 0, 0.99f, 1);  // nearly uniform: the prefilter cannot reach p, fallback path
    std::printf("top_p == full-sort reference (ids, order, logits exact): %d cases OK; worst p rel err vs exact %.3g, "
                "vs the float reference %.3g\n", n + 2, g_worst_vs_exact, g_worst_vs_ref);
}

// Times the sort-first arm and the as-shipped arm in PAIRS, alternating which of the two runs first, and returns the
// median of the per-pair ratios (and each arm's median, for the printout). Two medians taken one block after the other
// let host load that changes between the blocks decide the ratio: with two compiles beside it the queue flat case
// aborted in 6 of 10 runs and printed 0.90-1.10 in the other 4 (row
// S69-FORK-TEST-SAMPLING-NOSORT-QUEUE-FLAT-FLOOR-FLAKY-2026-10-19-01). Inside a pair both arms see the same load.
static const int k_pairs = 31;

struct paired_cost { double ref_ms, lib_ms, ratio; };

template <typename FR, typename FL> static paired_cost paired_median(FR && ref, FL && lib, int runs) {
    auto time_ms = [](auto && f, int r) {
        const auto t0 = std::chrono::steady_clock::now();
        f(r);
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
    std::vector<double> tr, tl, q;
    for (int r = 0; r < runs; ++r) {
        double a, b;
        if (r % 2 == 0) { a = time_ms(ref, r); b = time_ms(lib, r); }
        else            { b = time_ms(lib, r); a = time_ms(ref, r); }
        tr.push_back(a);
        tl.push_back(b);
        q.push_back(a / b);
    }
    auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    return { med(tr), med(tl), med(q) };
}

// A next-token distribution with a head: N(0, 2) logits plus 20 tokens 13..15 above the bulk, so the 0.9 nucleus is
// the head (tens of tokens), the usual shape for a trained model. make_vocab(.., 2.0f, 0) alone is the opposite case:
// flat, the 0.9 nucleus is ~36k tokens (P(z > 0.72) of 152064), which the first cutoff cannot reach.
static std::vector<llama_token_data> make_peaked(size_t n, uint32_t seed) {
    auto v = make_vocab(n, seed, 2.0f, 0);
    std::mt19937 rng(seed ^ 0x9e3779b9u);
    std::uniform_real_distribution<float> ud(13.0f, 15.0f);
    for (int h = 0; h < 20; ++h) {
        v[(size_t) (h * 6151 + 17) % n].logit = ud(rng);
    }
    return v;
}

// returns sort-first / as-shipped (median of the per-pair ratios)
static double cost_ratio(const char * what, bool peaked) {
    const size_t n = 152064;
    std::vector<std::vector<llama_token_data>> inputs;
    for (int r = 0; r < 15; ++r) inputs.push_back(peaked ? make_peaked(n, 500u + r) : make_vocab(n, 500u + r, 2.0f, 0));
    std::vector<llama_token_data> work;
    // sort-first: what the library did before on this input (full sort, then the sorted top_p path)
    const paired_cost c = paired_median([&](int r) {
        work = inputs[r % inputs.size()];
        std::sort(work.begin(), work.end(), [](const llama_token_data & a, const llama_token_data & b) {
            return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
        });
        llama_token_data_array a = { work.data(), work.size(), -1, true };
        llama_sample_top_p(nullptr, &a, 0.9f, 1);
        assert(a.size > 0);
    }, [&](int r) {
        work = inputs[r % inputs.size()];
        llama_token_data_array a = { work.data(), work.size(), -1, false };
        llama_sample_top_p(nullptr, &a, 0.9f, 1);
        assert(a.size > 0);
    }, k_pairs);
    // the copy into `work` is inside the timed region on purpose: it is the same memory traffic the reference pays
    std::printf("top_p(0.9) cost, %s, %zu tokens, %d interleaved pairs: sort-first %.3f ms, unsorted input %.3f ms, "
                "median pair ratio %.2f\n", what, n, k_pairs, c.ref_ms, c.lib_ms, c.ratio);
    return c.ratio;
}

static void test_cost() {
    const double peaked = cost_ratio("peaked (head nucleus)", true);
    const double flat   = cost_ratio("flat (36k-token nucleus)", false);
    assert(peaked > 2.0); // the case the change is for
    assert(flat > 0.9);   // the worst case must not turn into a regression (run 5, one cutoff + full re-sort: 0.82)
}

// The queue (common/sampling.cpp sampler_queue, via common_sampler_queue_apply) on an UNSORTED vocabulary must leave
// exactly what it leaves when the vocabulary arrives already sorted -- which is what the old top_k(k <= 0) produced at
// its step: same ids in the same order, same logits, probabilities to float rounding. The order matters because the
// draw after the queue walks the list in order, so a seeded draw is unchanged too.
static void check_queue(const char * what, common_params_sampling params, size_t n, uint32_t seed, float scale) {
    auto in = make_vocab(n, seed, scale, 3);
    auto ref = in;
    std::sort(ref.begin(), ref.end(), [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
    });
    llama_token_data_array ar = { ref.data(), ref.size(), -1, true };
    common_sampler_queue_apply(params, ar, 1);
    llama_token_data_array au = { in.data(), in.size(), -1, false };
    common_sampler_queue_apply(params, au, 1);
    if (au.size != ar.size) {
        std::printf("queue %s n=%zu seed=%u: size %zu want %zu\n", what, n, seed, au.size, ar.size);
        assert(false);
    }
    for (size_t i = 0; i < ar.size; ++i) {
        if (au.data[i].id != ar.data[i].id || au.data[i].logit != ar.data[i].logit) {
            std::printf("queue %s n=%zu seed=%u: position %zu holds id %d want %d\n", what, n, seed, i, au.data[i].id,
                        ar.data[i].id);
            assert(false);
        }
        // p is recomputed from the logits by a later softmax in most sequences (then it is equal); where top_p's p
        // survives, the sorted-input side carries the float full-vocabulary sum's rounding (see check_top_p)
        assert(std::fabs(au.data[i].p - ar.data[i].p) <= 1e-3f * std::fabs(ar.data[i].p) + 1e-9f);
    }
}

static void test_queue_matches_sorted_input() {
    int n_cases = 0;
    for (int top_k : { 0, -1, 152064 }) {
        for (float top_p : { 1.0f, 0.9f }) {
            for (float min_p : { 0.0f, 0.05f }) {
                for (float smoothing : { 0.0f, 0.3f }) {
                    common_params_sampling sp;
                    sp.top_k = top_k;
                    sp.top_p = top_p;
                    sp.min_p = min_p;
                    sp.temp = 0.8f;
                    sp.smoothing_factor = smoothing;
                    check_queue("grid", sp, 152064, 11u + (uint32_t) n_cases, 2.0f);
                    ++n_cases;
                }
            }
        }
    }
    // samplers that read the order or sort for themselves, switched ON after the deferred top_k
    common_params_sampling sp;
    sp.top_k = 0;
    sp.temp = 0.8f;
    { auto q = sp; q.top_a = 0.1f;                      check_queue("top_a", q, 32000, 1, 2.0f); }
    { auto q = sp; q.tfs_z = 0.95f;                     check_queue("tfs_z", q, 32000, 2, 2.0f); }
    { auto q = sp; q.typical_p = 0.9f;                  check_queue("typical", q, 32000, 3, 2.0f); }
    { auto q = sp; q.epsilon_cutoff = 3e-4f;            check_queue("epsilon", q, 32000, 4, 2.0f); }
    { auto q = sp; q.eta_cutoff = 1e-3f;                check_queue("eta", q, 32000, 5, 2.0f); }
    { auto q = sp; q.top_n_sigma = 1.0f;                check_queue("top_n_sigma", q, 32000, 6, 2.0f); }
    { auto q = sp; q.dynatemp_range = 0.5f;             check_queue("dynatemp", q, 32000, 7, 2.0f); }
    { auto q = sp; q.top_p = 0.9f; q.min_p = 0.02f; q.top_n_sigma = 2.0f; check_queue("mixed", q, 32000, 8, 3.0f); }
    { auto q = sp; q.samplers_sequence = { llama_sampler_type::TOP_K, llama_sampler_type::TEMPERATURE, llama_sampler_type::TOP_P };
      q.top_p = 0.9f;                                   check_queue("temp-before-top_p", q, 32000, 9, 2.0f); }
    { auto q = sp; q.samplers_sequence = { llama_sampler_type::TOP_K, llama_sampler_type::MIN_P };
      q.min_p = 0.05f;                                  check_queue("no-temp-step", q, 32000, 10, 2.0f); }
    { auto q = sp; q.top_k = 40; q.top_p = 0.9f;        check_queue("top_k-cuts", q, 32000, 11, 2.0f); }
    std::printf("queue on unsorted input == queue on sorted input: %d grid + 11 sampler cases OK\n", n_cases);
}

static double queue_cost_ratio(const char * what, bool peaked) {
    const size_t n = 152064;
    common_params_sampling sp;
    sp.top_k = 0;
    sp.top_p = 0.9f;
    sp.min_p = 0.0f;
    sp.temp = 0.8f;
    std::vector<std::vector<llama_token_data>> inputs;
    for (int r = 0; r < 15; ++r) inputs.push_back(peaked ? make_peaked(n, 900u + r) : make_vocab(n, 900u + r, 2.0f, 0));
    std::vector<llama_token_data> work;
    // the old queue: top_k(0) sorted the whole vocabulary at its step; a full sort and the same queue on sorted input
    const paired_cost c = paired_median([&](int r) {
        work = inputs[r % inputs.size()];
        std::sort(work.begin(), work.end(), [](const llama_token_data & a, const llama_token_data & b) {
            return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
        });
        llama_token_data_array a = { work.data(), work.size(), -1, true };
        common_sampler_queue_apply(sp, a, 1);
        assert(a.size > 0);
    }, [&](int r) {
        work = inputs[r % inputs.size()];
        llama_token_data_array a = { work.data(), work.size(), -1, false };
        common_sampler_queue_apply(sp, a, 1);
        assert(a.size > 0);
    }, k_pairs);
    std::printf("queue top_k=0 top_p=0.9 temp=0.8, %s, %zu tokens, %d interleaved pairs: sort-first %.3f ms, "
                "as shipped %.3f ms, median pair ratio %.2f\n", what, n, k_pairs, c.ref_ms, c.lib_ms, c.ratio);
    return c.ratio;
}

static void test_queue_cost() {
    const double peaked = queue_cost_ratio("peaked (head nucleus)", true);
    const double flat   = queue_cost_ratio("flat (36k-token nucleus)", false);
    assert(peaked > 2.0);
    assert(flat > 0.9);
}

int main() {
    // unbuffered: an assert abort does not flush stdio, so a failing run redirected to a file lost every ratio it had
    // printed and showed only the assert line (b1019a _b1019a_cuda_unit_test-sampling-nosort.log)
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    // llama_sample_top_k/top_p read ggml_time_us(), which on Windows divides by a frequency only ggml_time_init() sets
    // (same note as tests/test-topk-prefilter.cpp:107)
    ggml_time_init();
    test_top_p_matches_reference();
    test_cost();
    test_queue_matches_sorted_input();
    test_queue_cost();
    std::printf("test-sampling-nosort: all passed\n");
    return 0;
}
