#include "ggml.h"
#include "llama.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

extern struct llama_sampler * llama_sampler_init_dry_testing(float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const std::vector<std::vector<llama_token>>& seq_breakers);

static void dump(const llama_token_data_array * cur_p) {
    for (size_t i = 0; i < cur_p->size; i++) {
        printf("%d: %f (%f)\n", cur_p->data[i].id, cur_p->data[i].p, cur_p->data[i].logit);
    }
}

#define DUMP(__cur_p) do { printf("%s:%d (%s)\n", __FILE__, __LINE__, __func__); dump((__cur_p)); printf("-\n"); } while(0)

struct sampler_tester {
    sampler_tester(size_t n_vocab) {
        cur.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
            const float logit = logf(token_id);
            cur.emplace_back(llama_token_data{token_id, logit, 0.0f});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    sampler_tester(const std::vector<float> & probs, const std::vector<float> & probs_expected) : probs_expected(probs_expected) {
        cur.reserve(probs.size());
        for (llama_token token_id = 0; token_id < (llama_token)probs.size(); token_id++) {
            const float logit = logf(probs[token_id]);
            cur.emplace_back(llama_token_data{token_id, logit, probs[token_id]});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    void apply(llama_sampler * sampler) {
        llama_sampler_apply(sampler, &cur_p);
        llama_sampler_free(sampler);
    }

    void check() {
        GGML_ASSERT(cur_p.size == probs_expected.size());
        for (size_t i = 0; i < cur_p.size; i++) {
            GGML_ASSERT(fabs(cur_p.data[i].p - probs_expected[i]) < 1e-5);
        }
    }

    llama_token_data_array cur_p;

private:
    const std::vector<float> probs_expected;

    std::vector<llama_token_data> cur;
};

static llama_token sample_dist(llama_sampler * sampler, const std::vector<float> & logits) {
    std::vector<llama_token_data> cur;
    for (llama_token token_id = 0; token_id < (llama_token) logits.size(); ++token_id) {
        cur.push_back({ token_id, logits[token_id], 0.0f });
    }

    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(sampler, &cur_p);
    GGML_ASSERT(cur_p.selected >= 0);
    GGML_ASSERT((size_t) cur_p.selected < cur_p.size);
    return cur_p.data[cur_p.selected].id;
}

static void test_dist_singleton_rng() {
    llama_sampler * singleton = llama_sampler_init_dist(4242);
    llama_sampler * control   = llama_sampler_init_dist(4242);

    sample_dist(singleton, { 0.0f });
    sample_dist(control,   { 0.0f, 0.0f });

    const std::vector<float> logits(256, 0.0f);
    for (int i = 0; i < 4; ++i) {
        GGML_ASSERT(sample_dist(singleton, logits) == sample_dist(control, logits));
    }

    llama_sampler_free(singleton);
    llama_sampler_free(control);
}

static void test_temp(const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_temp(temp));
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_temp_ext(const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp, float delta, float exponent) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_temp_ext(temp, delta, exponent));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_top_k(const std::vector<float> & probs, const std::vector<float> & probs_expected, int k) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_k(k));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

// Exact logit ties at a top-k cut must keep the LOWEST token ids, whatever the input order and whichever standard
// library sorts them. n_tied tokens share the top logit; the rest are strictly lower. Each trial feeds a different
// permutation (descending ids first, then fixed-seed shuffles); the surviving id set must be {0..k-1} every time.
// Both entry points are covered: the llama_sampler chain (llama-sampler.cpp, <=128 partial_sort and >128 bucket
// paths) and the older llama_sample_top_k (llama-sampling.cpp llama_sort, <=1024 and >1024 paths).
static void test_top_k_tie_break(size_t n_vocab, int n_tied, int k, bool legacy_api) {
    uint64_t rng = 0x243f6a8885a308d3ULL ^ (n_vocab*31 + (uint64_t) k);
    for (int trial = 0; trial < 6; ++trial) {
        std::vector<llama_token_data> cur;
        cur.reserve(n_vocab);
        for (int i = (int) n_vocab - 1; i >= 0; --i) {  // descending ids: the adversarial order for a tie
            const float logit = i < n_tied ? 5.0f : -1.0f - 0.001f*(float) i;
            cur.push_back({ (llama_token) i, logit, 0.0f });
        }
        for (size_t i = cur.size() - 1; trial > 0 && i > 0; --i) {  // trial 0 keeps the adversarial order
            rng = rng*6364136223846793005ULL + 1442695040888963407ULL;
            std::swap(cur[i], cur[(size_t) ((rng >> 33) % (i + 1))]);
        }
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
        if (legacy_api) {
            llama_sample_top_k(nullptr, &cur_p, k, 1);
        } else {
            llama_sampler * s = llama_sampler_init_top_k(k);
            llama_sampler_apply(s, &cur_p);
            llama_sampler_free(s);
        }
        GGML_ASSERT(cur_p.size == (size_t) k);
        std::vector<llama_token> kept;
        for (size_t i = 0; i < cur_p.size; ++i) kept.push_back(cur_p.data[i].id);
        std::sort(kept.begin(), kept.end());
        for (int i = 0; i < k; ++i) {
            if (kept[i] != i) {
                printf("top-k tie-break FAILED: n_vocab=%zu n_tied=%d k=%d legacy=%d trial=%d: kept[%d]=%d, want %d\n",
                       n_vocab, n_tied, k, (int) legacy_api, trial, i, kept[i], i);
                GGML_ASSERT(false);
            }
        }
    }
    printf("top-k tie-break OK: n_vocab=%zu n_tied=%d k=%d legacy=%d (6 input orders)\n", n_vocab, n_tied, k, (int) legacy_api);
}

static void test_top_p(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_p(p, 0));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_min_p(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_min_p(p, 0));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_xtc(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p, float t) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_xtc(p, t, 0, 0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_typical(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_typical(p, 0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_penalties(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & probs_expected, float repeat_penalty, float alpha_frequency, float alpha_presence
) {
    GGML_ASSERT(probs.size() == probs_expected.size());

    sampler_tester tester(probs, probs_expected);

    auto * sampler = llama_sampler_init_penalties((int32_t) probs.size(), (int32_t) last_tokens.size(), repeat_penalty, alpha_frequency, alpha_presence);

    for (size_t i = 0; i < last_tokens.size(); i++) {
        llama_sampler_accept(sampler, last_tokens[i]);
    }

    DUMP(&tester.cur_p);
    tester.apply(sampler);
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_dry(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & expected_probs, float dry_multiplier, float dry_base,
    int dry_allowed_length, int dry_penalty_last_n,
    const std::vector<std::vector<llama_token>> & seq_breakers
) {
    GGML_ASSERT(probs.size() == expected_probs.size());

    sampler_tester tester(probs, expected_probs);

    auto * sampler = llama_sampler_init_dry_testing(dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, seq_breakers);

    for (size_t i = 0; i < last_tokens.size(); i++) {
        llama_sampler_accept(sampler, last_tokens[i]);
    }

    DUMP(&tester.cur_p);
    tester.apply(sampler);
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);
    tester.check();
}

static void test_top_n_sigma(const std::vector<float> & probs, const std::vector<float> & probs_expected, int n) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_n_sigma(n));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_sampler_queue(const size_t n_vocab, const std::string & samplers_sequence, const int top_k, const float top_p, const float min_p
) {
    sampler_tester tester(n_vocab);

          llama_token min_token_id = 0;
    const llama_token max_token_id = n_vocab - 1;

    for (auto s : samplers_sequence) {
        switch (s) {
            case 'k': tester.apply(llama_sampler_init_top_k(top_k)); break;
            // 'f' is fork-only. Upstream removed the tail-free sampler entirely, but this fork
            // still declares llama_sample_tail_free in include/llama.h, so the case is kept as
            // the explicit placeholder rather than letting 'f' fall through to "Unknown sampler".
            // A real test needs a llama_sampler_init_tail_free chain constructor for tester to
            // apply; that is filed separately. HEAD did not test it either.
            case 'f': GGML_ABORT("tail_free test not implemented");
            case 'y': GGML_ABORT("typical test not implemented");
            case 'p': tester.apply(llama_sampler_init_top_p(top_p, 1)); break;
            case 'm': tester.apply(llama_sampler_init_min_p(min_p, 1)); break;
            case 't': GGML_ABORT("temperature test not implemented");
            default : GGML_ABORT("Unknown sampler");
        }

        tester.apply(llama_sampler_init_dist(0));

        auto & cur_p = tester.cur_p;

        const int size = cur_p.size;

        if (s == 'k') {
            const int expected_size = std::min(size, top_k);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - top_k));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(cur_p.data[0].id == max_token_id);
            GGML_ASSERT(cur_p.data[expected_size-1].id == min_token_id);
        } else if (s == 'p') {
            const int softmax_divisor = n_vocab * (n_vocab-1) / 2 - min_token_id * (min_token_id-1) / 2;
            const int softmax_numerator_target = ceilf(top_p * softmax_divisor);

                min_token_id  = n_vocab;
            int expected_size = 0;
            int cumsum        = 0;
            do { // do-while because always at least one token is sampled
                min_token_id--;
                expected_size++;

                cumsum += min_token_id;
            } while (cumsum < softmax_numerator_target);

            // token 0 has p == 0, need special consideration for cumsum because top_p immediately returns
            if (min_token_id == 1) {
                min_token_id--;
                expected_size += 1;
            }

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[0].id == max_token_id);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
        } else if (s == 'm') {
            int expected_size = ceilf((1.0f - min_p) * n_vocab);
            expected_size = std::max(expected_size, 1);
            expected_size = std::min(expected_size, size);

            min_token_id = floorf(min_p * n_vocab);
            min_token_id = std::max(min_token_id, 1);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - size));
            min_token_id = std::min(min_token_id, (llama_token)(n_vocab - 1));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[0].id == max_token_id);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
        } else {
            GGML_ABORT("fatal error");
        }
    }

    printf("Sampler queue %3s OK with n_vocab=%05zu top_k=%5d top_p=%f min_p=%f\n",
           samplers_sequence.c_str(), n_vocab, top_k, top_p, min_p);
}

static void bench(llama_sampler * cnstr, const char * cnstr_name, const std::vector<llama_token_data> & data, int n_iter) {
    std::vector<llama_token_data> cur(data.size());
    std::copy(data.begin(), data.end(), cur.begin());
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(cnstr, &cur_p);
    llama_sampler_reset(cnstr);
    const int64_t t_start = ggml_time_us();
    for (int i = 0; i < n_iter; i++) {
        std::copy(data.begin(), data.end(), cur.begin());
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(cnstr, &cur_p);
        llama_sampler_reset(cnstr);
    }
    const int64_t t_end = ggml_time_us();
    llama_sampler_free(cnstr);
    printf("%-43s: %8.3f us/iter\n", cnstr_name, (t_end - t_start) / (float)n_iter);
}

#define BENCH(__cnstr, __data, __n_iter) bench((__cnstr), #__cnstr, (__data), (__n_iter))

static void test_perf() {
    const int n_vocab = 1 << 17;

    std::vector<llama_token_data> data;

    data.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        const float logit = 2.0f*((double)(rand())/RAND_MAX - 0.5);
        data.emplace_back(llama_token_data{i, logit, 0.0f});
    }

    BENCH(llama_sampler_init_top_k  (40),                     data, 32);
    BENCH(llama_sampler_init_top_p  (0.8f, 1),                data, 32);
    BENCH(llama_sampler_init_min_p  (0.2f, 1),                data, 32);
    BENCH(llama_sampler_init_typical(0.5f, 1),                data, 32);
    BENCH(llama_sampler_init_xtc    (1.0f, 0.1f, 1, 1),       data, 32);
}

static void test_allowed_tokens(
        const std::vector<float>       & probs,
        const std::vector<llama_token> & allowed,
        int32_t                          n_vocab,
        const std::vector<bool>        & expect_masked) {
    // Built directly rather than through sampler_tester: this sampler is checked on LOGITS, and
    // sampler_tester::check compares normalised probabilities, which would hide the difference
    // between "masked to -INFINITY" and "given a very small probability".
    std::vector<llama_token_data> cur;
    for (llama_token id = 0; id < (llama_token) probs.size(); ++id) {
        cur.emplace_back(llama_token_data{id, logf(probs[id]), probs[id]});
    }
    const std::vector<llama_token_data> before = cur;

    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };

    llama_sampler * smpl = llama_sampler_init_allowed_tokens(
        n_vocab, (int32_t) allowed.size(), allowed.empty() ? nullptr : allowed.data());
    llama_sampler_apply(smpl, &cur_p);
    llama_sampler_free(smpl);

    GGML_ASSERT(cur_p.size == before.size());
    for (size_t i = 0; i < cur_p.size; ++i) {
        if (expect_masked[i]) {
            GGML_ASSERT(cur_p.data[i].logit == -INFINITY);
        } else {
            // POSITIVE CONTROL: an allowed token must come through untouched. Without this half,
            // a sampler that masked EVERYTHING would pass the test.
            GGML_ASSERT(cur_p.data[i].logit == before[i].logit);
        }
    }
}

static void test_allowed_tokens_clone(void) {
    // The clone must be independent of the original. Freeing the original FIRST and applying the
    // clone afterwards is the point of the ordering here.
    const std::vector<llama_token> allowed = {2};

    llama_sampler * orig  = llama_sampler_init_allowed_tokens(4, (int32_t) allowed.size(), allowed.data());
    llama_sampler * clone = llama_sampler_clone(orig);
    llama_sampler_free(orig);

    std::vector<llama_token_data> cur;
    for (llama_token id = 0; id < 4; ++id) {
        cur.emplace_back(llama_token_data{id, 1.0f, 0.25f});
    }
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };

    llama_sampler_apply(clone, &cur_p);
    llama_sampler_free(clone);

    for (size_t i = 0; i < cur_p.size; ++i) {
        GGML_ASSERT(cur_p.data[i].logit == (cur_p.data[i].id == 2 ? 1.0f : -INFINITY));
    }
}

// Dynamic temperature normalises the entropy by log(number of FINITE candidates), not log(all candidates): a logit
// masked to -inf (grammar, logit bias) carries no probability and must not widen the denominator, or the temperature
// range squeezes toward its minimum. Spec: text-generation-webui sampler_hijack.py (finite-logit count; AGPL, spec only).
// 40 candidates, 3 finite with p .90/.07/.03, temp 1.0 +/- 0.5, exponent 1 -> T = 0.5 + H/ln(3) = 0.8514. With the
// old all-candidate count T would be 0.5 + H/ln(40) = 0.6047. Expected probabilities are computed here in double.
static void test_temp_ext_finite_denominator() {
    const std::vector<double> pf = { 0.90, 0.07, 0.03 };
    double h = 0.0;
    for (double p : pf) { h -= p*std::log(p); }
    const double T = 0.5 + 1.0 * (h / std::log(3.0));
    GGML_ASSERT(std::fabs(T - 0.8514) < 1e-3);
    std::vector<double> want(pf.size());
    double z = 0.0;
    for (size_t i = 0; i < pf.size(); ++i) { want[i] = std::pow(pf[i], 1.0 / T); z += want[i]; }
    for (double & w : want) { w /= z; }

    for (int api = 0; api < 2; ++api) {  // 0: llama_sampler chain element, 1: legacy llama_sample_entropy
        std::vector<llama_token_data> cur;
        for (llama_token id = 0; id < 40; ++id) {
            const float logit = id < 3 ? (float) std::log(pf[id]) : -INFINITY;
            cur.push_back({ id, logit, 0.0f });
        }
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
        if (api == 0) {
            llama_sampler * s = llama_sampler_init_temp_ext(1.0f, 0.5f, 1.0f);
            llama_sampler_apply(s, &cur_p);
            llama_sampler_free(s);
        } else {
            llama_sample_entropy(nullptr, &cur_p, 0.5f, 1.5f, 1.0f);
        }
        for (size_t i = 0; i < cur_p.size; ++i) {
            const llama_token id = cur_p.data[i].id;
            if (id < 3) {
                GGML_ASSERT(std::fabs(cur_p.data[i].p - want[id]) < 1e-4);
            } else {
                GGML_ASSERT(cur_p.data[i].p == 0.0f);
            }
        }
    }
}

int main(void) {
    ggml_time_init();

    test_dist_singleton_rng();

    test_temp_ext_finite_denominator();

    test_temp({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);
    test_temp({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f);

    test_temp_ext({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f, 0.0f, 1.0f);
    test_temp_ext({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f, 0.0f, 1.0f);

    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 1);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 3);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f, 0.1f}, 4);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0);

    // exact-tie determinism at the top-k boundary (both APIs, small-k and bucket paths)
    test_top_k_tie_break(   64,   40,   10, false);  // llama_sampler, partial_sort path (k <= 128)
    test_top_k_tie_break( 8192, 1000,  300, false);  // llama_sampler, bucket path       (k > 128)
    test_top_k_tie_break(   64,   40,   10, true);   // llama_sort,    partial_sort path (k <= 1024)
    test_top_k_tie_break( 8192, 3000, 1500, true);   // llama_sort,    bucket path       (k > 1024)

    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 0);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.571429f, 0.428571f}, 0.7f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 0.8f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);

    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.00f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.24f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.26f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.49f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.51f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.74f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  0.76f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.00f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.05f);

    printf("XTC should:\n");
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.1f},                                0.99f, 0.09f);
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.2f, 0.1f},                          0.99f, 0.19f);
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.3f, 0.2f, 0.1f},                    0.99f, 0.29f);

    printf("XTC should not:\n");
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.4f, 0.3f, 0.2f, 0.1f},              0.99f, 0.39f);

    test_typical({0.97f, 0.01f, 0.01f, 0.01f}, {0.97f},            0.5f);
    test_typical({0.4f, 0.2f, 0.2f, 0.2f},     {0.2f, 0.2f, 0.2f}, 0.5f);

    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0}, {0, 0.25f, 0.25f, 0.25f, 0.25f},   50.0f, 0.0f, 0.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2}, {0, 0, 0, 0.5f, 0.5f},       50.0f, 0.0f, 0.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0, 0, 0, 0.5f, 0.5f}, 50.0f, 0.0f, 0.0f);

    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0},             {0.000011f, 0.249997f, 0.249997f, 0.249997f, 0.249997f}, 1.0f, 5.0f, 5.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2},       {0.000023f, 0.000023f, 0.000023f, 0.499966f, 0.499966f}, 1.0f, 5.0f, 5.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0.000000f, 0.000023f, 0.000023f, 0.499977f, 0.499977f}, 1.0f, 5.0f, 5.0f);


    test_dry({0.25f, 0.25f, 0.25f, 0.25f}, {0, 1}, {0.25f, 0.25f, 0.25f, 0.25f}, 1.0f, 1.1f, 2, 4, {});
    test_dry({0.25f, 0.25f, 0.25f, 0.25f}, {0, 1, 2, 0, 1}, {0.296923f, 0.296923f, 0.109232f, 0.296923f}, 1.0f, 1.1f, 2, 5, {});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 2, 6, {{3}});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 1}, {0.241818f, 0.241818f, 0.032727f, 0.241818f, 0.241818f}, 2.0f, 1.1f, 2, 5, {});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 4, 7, {});

    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.428571f, 0.571429f}, 1.00f);
    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0.00f); // top_n_sigma == 0 now represents a no-op rather than greedy decoding as of PR#13345
    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 3.00f);

    test_sampler_queue(10000, "k", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "k",     1, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1e-12);

    test_sampler_queue(10000, "k",   100, 1.0000f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0003f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.8000f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0000f, 9997.9f/9999.0f);
    test_sampler_queue(10000, "m", 10000, 1.0000f, 0.1f);

    test_sampler_queue(10000, "kp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "km", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mp", 100, 0.8f, 9997.9f/9999.0f);
    test_sampler_queue(10000, "mp", 100, 0.8f, 0.1f);

    test_sampler_queue(10000, "kpm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "kmp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pkm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pmk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mkp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mpk", 100, 0.8f, 0.1f);

    // allowed-tokens: hard allow-list constraint (added 2026-09-14, closing a measured vLLM gap)
    // 1. a plain allow-list masks exactly the unlisted tokens and leaves the listed ones alone
    test_allowed_tokens({0.25f, 0.25f, 0.25f, 0.25f}, {1, 3}, 4, {true, false, true, false});
    // 2. EMPTY list is a NO-OP, not "forbid everything" -- the dangerous alternative reading
    test_allowed_tokens({0.25f, 0.25f, 0.25f, 0.25f}, {},     4, {false, false, false, false});
    // 3. an out-of-range id is ignored WITHOUT discarding the rest of the list
    test_allowed_tokens({0.25f, 0.25f, 0.25f, 0.25f}, {1, 99}, 4, {true, false, true, true});
    // 4. and a negative id likewise
    test_allowed_tokens({0.25f, 0.25f, 0.25f, 0.25f}, {-5, 0}, 4, {false, true, true, true});
    test_allowed_tokens_clone();

    printf("OK\n");

    test_perf();

    return 0;
}
