// SCOUT-NA: unit test in the fork tree, no new tool.
// Row S69-ENGREV-UPSTREAM-MERGE-DRAFT-SAMPLE-EQUIVALENCE-TEST-2026-11-04-01 (concerns C2, C3, C4).
//
//   C4/C3  The upstream spelling (--spec-draft-sampling greedy|probabilistic) and the fork spelling (--spec-draft-sample /
//          --no-spec-draft-sample) are ONE state, params.speculative.draft_sample: either spelling, in either order, gives the same
//          value, and nothing else in params.speculative moves. Then the acceptance statistics (chi-square against the target p, the
//          acceptance rate against its analytic value) are computed for the value each spelling produced and must be identical:
//          same state, same seed, same inputs = bit-identical outcome.
//   C2     Grammar-mask renormalisation. A drafter that draws under a grammar records q = the MASKED, RENORMALISED distribution
//          (common_sampler_sampling_dist after common_sampler_sample(..., grammar_first = true)), and the rejection verifier fed that q
//          (and the masked p) emits exactly the masked p. Model-free part: the rule over a masked p/q (chi-square, acceptance =
//          sum min(pG, qG) > sum pG*qG). Model part (-m): the real sampler with a GBNF grammar on a tiny model.
//
// Negative controls: a q that is NOT renormalised after the mask (mass leaking to forbidden tokens) must fail the chi-square.
// GATE-CLAIM-OK: unit test, no resource claims.
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "spec-accept.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

static int g_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { std::printf("FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } \
                              else { std::printf("ok:   "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// ---------------------------------------------------------------------------------------------------------------------------------
// A. one state, two spellings
// ---------------------------------------------------------------------------------------------------------------------------------
struct parsed_spec {
    bool     ok = false;
    bool     draft_sample = false;
    bool     draft_grammar = false;
    int32_t  n_max = 0;
    int32_t  n_min = 0;
    float    p_min = 0.0f;
};

// Field-wise comparison. NOT memcmp: parsed_spec has a padding byte after the three bools (offset 3) that the default member
// initialisers do not zero, so memcmp compared an indeterminate byte -- MEASURED 2026-11-06: both "same speculative settings"
// checks passed on the agent's CPU MSVC build and FAILED on gate b1019ze's CUDA build while every field-level check passed.
// The differing fields are printed so a REAL difference is still caught and named.
static bool same_spec(const parsed_spec & a, const parsed_spec & b, std::string & diff) {
    diff.clear();
    if (a.ok != b.ok) { diff += " ok"; }
    if (a.draft_sample != b.draft_sample) { diff += " draft_sample"; }
    if (a.draft_grammar != b.draft_grammar) { diff += " draft_grammar"; }
    if (a.n_max != b.n_max) { diff += " n_max(" + std::to_string(a.n_max) + "/" + std::to_string(b.n_max) + ")"; }
    if (a.n_min != b.n_min) { diff += " n_min(" + std::to_string(a.n_min) + "/" + std::to_string(b.n_min) + ")"; }
    if (a.p_min != b.p_min) { diff += " p_min(" + std::to_string(a.p_min) + "/" + std::to_string(b.p_min) + ")"; }
    return diff.empty();
}

static parsed_spec parse_spec(const std::vector<std::string> & args) {
    // the speculative example insists on a model path; the parser only stores it (nothing is opened)
    std::vector<std::string> store = { "test-spec-draft-sample-equiv", "-m", "not-opened.gguf" };
    store.insert(store.end(), args.begin(), args.end());
    std::vector<char *> argv;
    for (auto & s : store) {
        argv.push_back(s.data());
    }
    common_params params;
    parsed_spec   r;
    r.ok = common_params_parse((int) argv.size(), argv.data(), params, LLAMA_EXAMPLE_SPECULATIVE);
    r.draft_sample  = params.speculative.draft_sample;
    r.draft_grammar = params.speculative.draft_grammar;
    r.n_max         = params.speculative.n_max;
    r.n_min         = params.speculative.n_min;
    r.p_min         = params.speculative.p_min;
    return r;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// B. the rule over a grammar-masked p / q (model-free)
// ---------------------------------------------------------------------------------------------------------------------------------
static const int    K         = 8;
static const int    N         = 200000;
static const double CHI2_CRIT = 29.88; // chi-square 0.9999 quantile, df = K - 1

static common_spec_dist make(const std::vector<double> & w) {
    common_spec_dist d;
    for (int i = 0; i < (int) w.size(); ++i) {
        if (w[i] > 0.0) {
            d.ids.push_back(i);
            d.probs.push_back((float) w[i]);
        }
    }
    return d;
}

static llama_token draw_idx(const std::vector<double> & w, std::mt19937 & rng) {
    std::discrete_distribution<int> dd(w.begin(), w.end());
    return dd(rng);
}

static double chi2(const std::vector<long> & counts, const std::vector<double> & p, long n) {
    double s = 0.0;
    for (int i = 0; i < K; ++i) {
        const double e = p[i] * n;
        if (e > 0.0) {
            s += (counts[i] - e) * (counts[i] - e) / e;
        } else if (counts[i] > 0) {
            s += 1e9; // a token with zero target mass was emitted
        }
    }
    return s;
}

static double sigma(double a) { return std::sqrt(a * (1.0 - a) / N); }

// zero the forbidden tokens and (renorm) rescale to total mass 1 -- the grammar mask
static std::vector<double> mask_renorm(const std::vector<double> & w, const std::vector<bool> & allowed, bool renorm) {
    std::vector<double> o(w.size(), 0.0);
    double sum = 0.0;
    for (size_t i = 0; i < w.size(); ++i) {
        if (allowed[i]) { o[i] = w[i]; sum += w[i]; }
    }
    if (renorm) {
        for (auto & v : o) { v /= sum; }
    }
    return o;
}

struct run_out { double chi2; double acc; };

// x ~ draft_w; the rule is told q and the target p. one_hot_p: the server's default under a grammar (p is the one-hot at the target's
// own draw `picked` ~ pw), which reduces the rule to sample-and-match.
static run_out run_rule(const std::vector<double> & pw, const std::vector<double> & draft_w, const common_spec_dist & q, bool one_hot_p,
                        uint32_t seed) {
    const common_spec_dist p = make(pw);
    std::mt19937 rng(seed);
    std::vector<long> counts(K, 0);
    long acc = 0;
    for (int t = 0; t < N; ++t) {
        const llama_token x = draw_idx(draft_w, rng);
        common_spec_accept_result r{};
        if (one_hot_p) {
            const llama_token picked = draw_idx(pw, rng);
            common_spec_dist oh;
            oh.ids.assign(1, picked);
            oh.probs.assign(1, 1.0f);
            r = common_spec_accept_rejection(oh, q, x, rng);
        } else {
            r = common_spec_accept_rejection(p, q, x, rng);
        }
        counts[r.token]++;
        acc += r.accepted;
    }
    return { chi2(counts, pw, N), (double) acc / N };
}

static void test_masked_rule() {
    // tokens 3, 4, 7 are forbidden by the grammar
    const std::vector<double> pw0 = { .30, .20, .15, .10, .10, .08, .05, .02 };
    const std::vector<double> qw0 = { .05, .10, .30, .20, .05, .15, .10, .05 };
    const std::vector<bool>   allowed = { true, true, true, false, false, true, true, false };
    const auto pw = mask_renorm(pw0, allowed, true);   // the masked, renormalised target
    const auto qw = mask_renorm(qw0, allowed, true);   // the masked, renormalised draft: what a grammar-first draw really samples
    const common_spec_dist p = make(pw), q = make(qw);

    double mass_forbidden = 0.0;
    for (int i = 0; i < K; ++i) { mass_forbidden += allowed[i] ? 0.0 : qw[i]; }
    CHECK(mass_forbidden == 0.0, "masked q has no mass on a forbidden token");

    const double a_rej = common_spec_accept_rate_rejection(p, q), a_sm = common_spec_accept_rate_sample_match(p, q);
    CHECK(a_rej > a_sm + 0.05, "under the mask rejection accepts more than sample-and-match (%.3f vs %.3f)", a_rej, a_sm);

    // 1. exact p AND exact q (grammar-first target + grammar-first draft): emits the masked p, accepts at sum min(pG, qG)
    run_out r = run_rule(pw, qw, q, /* one_hot_p = */ false, 11);
    CHECK(r.chi2 < CHI2_CRIT, "masked p + masked q: emits the masked p, chi2 = %.2f < %.2f", r.chi2, CHI2_CRIT);
    CHECK(std::fabs(r.acc - a_rej) < 5 * sigma(a_rej), "masked p + masked q: acceptance %.4f vs analytic sum min = %.4f", r.acc, a_rej);

    // 2. the server's default under a grammar: p is the one-hot at the target's own draw. Still exact, acceptance = sum pG*qG
    r = run_rule(pw, qw, q, /* one_hot_p = */ true, 12);
    CHECK(r.chi2 < CHI2_CRIT, "one-hot p (sample-first target) + masked q: still emits the masked p, chi2 = %.2f", r.chi2);
    CHECK(std::fabs(r.acc - a_sm) < 5 * sigma(a_sm), "one-hot p + masked q: acceptance %.4f vs analytic sum p*q = %.4f", r.acc, a_sm);

    // 3. the old behaviour: the draft ignores the grammar, so it proposes forbidden tokens; the rule is still exact (they are rejected)
    //    but acceptance is lower than with the masked draft
    {
        const common_spec_dist q_unmasked = make(qw0);
        r = run_rule(pw, qw0, q_unmasked, false, 13);
        const double a_un = common_spec_accept_rate_rejection(p, q_unmasked);
        CHECK(r.chi2 < CHI2_CRIT, "unconstrained draft: still exact, chi2 = %.2f", r.chi2);
        CHECK(a_rej > a_un + 0.05 && std::fabs(r.acc - a_un) < 5 * sigma(a_un),
              "unconstrained draft accepts less (%.4f measured, %.4f analytic) than the masked draft (%.4f)", r.acc, a_un, a_rej);
    }

    // NEGATIVE CONTROL: the draft draws from the masked qG but the verifier is told the UNMASKED q (a recorded q that is not the
    // distribution the token was drawn from) -> the emitted stream is no longer the target p, and the chi-square must say so
    {
        const common_spec_dist told = make(qw0);
        r = run_rule(pw, qw, told, false, 14);
        CHECK(r.chi2 > 10 * CHI2_CRIT, "NEGATIVE: verifying against a q that is not the drawing distribution is rejected: chi2 = %.1f", r.chi2);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// C. the real sampler with a GBNF grammar (needs a model)
// ---------------------------------------------------------------------------------------------------------------------------------
using smpl_ptr = std::unique_ptr<common_sampler, decltype(&common_sampler_free)>;

static llama_context * decode_window(llama_model * model, const common_params & params, const std::vector<llama_token> & window) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_ctx     = 128;
    cparams.n_batch   = 64;
    cparams.n_ubatch  = 64;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        return nullptr;
    }
    llama_batch batch = llama_batch_init((int) window.size(), 0, 1);
    for (size_t i = 0; i < window.size(); ++i) {
        common_batch_add(batch, window[i], (llama_pos) i, { 0 }, true);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    if (!ok) {
        llama_free(ctx);
        return nullptr;
    }
    return ctx;
}

static smpl_ptr make_sampler(llama_model * model, const common_params & params, float temp, uint32_t seed) {
    common_params_sampling sp = params.sampling;
    sp.seed = seed;
    sp.temp = temp;
    sp.logit_bias.clear();
    sp.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, "root ::= [a-z ]+");
    return smpl_ptr { common_sampler_init(model, sp), common_sampler_free };
}

static double total_variation(const std::map<llama_token, double> & a, const std::map<llama_token, double> & b) {
    std::map<llama_token, double> all = a;
    for (const auto & kv : b) {
        all.emplace(kv.first, 0.0);
    }
    double tv = 0.0;
    for (const auto & kv : all) {
        const auto ia = a.find(kv.first);
        const auto ib = b.find(kv.first);
        tv += std::fabs((ia == a.end() ? 0.0 : ia->second) - (ib == b.end() ? 0.0 : ib->second));
    }
    return 0.5 * tv;
}

static std::map<llama_token, double> as_map(const common_spec_dist & d) {
    std::map<llama_token, double> m;
    double sum = 0.0;
    for (float v : d.probs) { sum += v; }
    for (size_t i = 0; i < d.ids.size(); ++i) { m[d.ids[i]] += d.probs[i] / sum; }
    return m;
}

static void test_sampler_with_grammar(llama_model * model, const common_params & params) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> prompt;
    for (int i = 0; i < 6; ++i) {
        prompt.push_back((llama_token) ((17*i + 11) % n_vocab));
    }
    const int n_prompt = (int) prompt.size();
    const int NT = 4000;

    llama_context * ctx1 = decode_window(model, params, prompt);
    if (ctx1 == nullptr) {
        std::printf("FAIL: decode failed\n");
        ++g_fail;
        return;
    }

    // oracle: empirical draws of the grammar-FIRST target sampler, and of the sample-first one (the server's default)
    std::map<llama_token, double> emp_first, emp_after;
    common_spec_dist p_dist;
    for (int s = 0; s < NT; ++s) {
        smpl_ptr a = make_sampler(model, params, 1.0f, 1000 + (uint32_t) s);
        const llama_token ta = common_sampler_sample(a.get(), ctx1, n_prompt - 1, /* grammar_first = */ true);
        emp_first[ta] += 1.0 / NT;
        if (s == 0) {
            p_dist = common_sampler_sampling_dist(a.get(), ta);
        }
        smpl_ptr b = make_sampler(model, params, 1.0f, 1000 + (uint32_t) s);
        const llama_token tb = common_sampler_sample(b.get(), ctx1, n_prompt - 1, /* grammar_first = */ false);
        emp_after[tb] += 1.0 / NT;
        if (s == 0) {
            const common_spec_dist d = common_sampler_sampling_dist(b.get(), tb);
            CHECK(d.ids.size() == 1 && d.ids[0] == tb, "sample-first draw under a grammar keeps the exact one-hot fallback");
        }
    }
    CHECK(p_dist.ids.size() > 3, "grammar-first draw: sampling_dist is a real distribution over %zu tokens, not a one-hot", p_dist.ids.size());
    const double tv_p = total_variation(as_map(p_dist), emp_first);
    CHECK(tv_p <= 0.06, "grammar-first sampling_dist matches the sampler's own empirical draws: TV = %.3f", tv_p);

    // the draft: a hotter sampler under the same grammar, drawn grammar-first; q = its sampling_dist
    std::vector<llama_token> xs;
    common_spec_dist q;
    std::map<llama_token, double> emp_draft;
    for (int s = 0; s < 400; ++s) {
        smpl_ptr d = make_sampler(model, params, 1.5f, 5000 + (uint32_t) s);
        const llama_token x = common_sampler_sample(d.get(), ctx1, n_prompt - 1, true);
        xs.push_back(x);
        emp_draft[x] += 1.0 / 400;
        if (s == 0) { q = common_sampler_sampling_dist(d.get(), x); }
    }
    CHECK(q.ids.size() > 3, "draft q is a distribution over %zu tokens", q.ids.size());
    CHECK(total_variation(as_map(q), emp_draft) <= 0.12, "draft q matches the draft's own draws: TV = %.3f", total_variation(as_map(q), emp_draft));
    llama_free(ctx1);

    // verify with the rejection rule: target grammar-first (p = masked, renormalised) and target sample-first (p = one-hot)
    const double a_min = common_spec_accept_rate_rejection(p_dist, q);
    for (bool gf : { true, false }) {
        std::map<llama_token, double> first;
        int n_acc = 0, n_tr = 0;
        for (llama_token x : std::set<llama_token>(xs.begin(), xs.end())) {
            std::vector<llama_token> window = prompt;
            window.push_back(x);
            llama_context * ctx2 = decode_window(model, params, window);
            if (ctx2 == nullptr) { ++g_fail; std::printf("FAIL: decode failed\n"); return; }
            const int reps = (int) std::count(xs.begin(), xs.end(), x) * (NT / 400);
            for (int s = 0; s < reps; ++s) {
                smpl_ptr t = make_sampler(model, params, 1.0f, 9000 + (uint32_t) (n_tr));
                std::mt19937 rng(777 + n_tr);
                std::vector<common_spec_dist> dists(1, q);
                const auto got = common_sampler_sample_and_accept_n_rejection(t.get(), ctx2, { n_prompt - 1, n_prompt }, { x }, dists, rng, gf);
                ++n_tr;
                if (got.empty()) { continue; }
                n_acc += (got.size() == 2 && got[0] == x) ? 1 : 0;
                first[got[0]] += 1.0;
            }
            llama_free(ctx2);
        }
        for (auto & kv : first) { kv.second /= n_tr; }
        const double tv = total_variation(first, gf ? emp_first : emp_after);
        const double acc = (double) n_acc / n_tr;
        double expect = 0.0;
        if (gf) {
            expect = a_min;
        } else {
            for (size_t i = 0; i < q.ids.size(); ++i) {
                const auto it = emp_after.find(q.ids[i]);
                expect += (it == emp_after.end() ? 0.0 : it->second) * common_spec_dist_prob(q, q.ids[i]);
            }
        }
        CHECK(tv <= 0.08, "verify with grammar_first=%d: the first emitted token follows the target's own draw, TV = %.3f", (int) gf, tv);
        CHECK(std::fabs(acc - expect) <= 0.04, "verify with grammar_first=%d: acceptance %.3f vs analytic %.3f (%s)", (int) gf, acc, expect,
              gf ? "sum min(pG, qG)" : "sum P*q, one-hot p");
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // A. spellings
    {
        const parsed_spec def  = parse_spec({});
        const parsed_spec up_p = parse_spec({ "--spec-draft-sampling", "probabilistic" });
        const parsed_spec fk_p = parse_spec({ "--spec-draft-sample" });
        const parsed_spec up_g = parse_spec({ "--spec-draft-sampling", "greedy" });
        const parsed_spec fk_g = parse_spec({ "--no-spec-draft-sample" });
        CHECK(def.ok && up_p.ok && fk_p.ok && up_g.ok && fk_g.ok, "all spellings parse");
        CHECK(!def.draft_sample && !up_g.draft_sample && !fk_g.draft_sample, "default and both greedy spellings: draft_sample = false");
        CHECK(up_p.draft_sample && fk_p.draft_sample, "both probabilistic spellings: draft_sample = true");
        std::string d_p, d_g;
        CHECK(same_spec(up_p, fk_p, d_p), "probabilistic: the two spellings leave the same speculative settings (differs:%s)", d_p.c_str());
        CHECK(same_spec(up_g, fk_g, d_g), "greedy: the two spellings leave the same speculative settings (differs:%s)", d_g.c_str());
        const parsed_spec a = parse_spec({ "--spec-draft-sample", "--spec-draft-sampling", "greedy" });
        const parsed_spec b = parse_spec({ "--spec-draft-sampling", "greedy", "--spec-draft-sample" });
        const parsed_spec c = parse_spec({ "--spec-draft-sampling", "probabilistic", "--no-spec-draft-sample" });
        CHECK(!a.draft_sample && b.draft_sample && !c.draft_sample, "mixed spellings: ONE state, the last flag wins (%d %d %d)", (int) a.draft_sample,
              (int) b.draft_sample, (int) c.draft_sample);
        CHECK(!parse_spec({ "--spec-draft-sampling", "bogus" }).ok, "an invalid --spec-draft-sampling value is refused");

        // same state -> same acceptance statistics: run the rule the way each spelling configured it
        const std::vector<double> pw = { .30, .20, .15, .10, .10, .08, .05, .02 };
        const std::vector<double> qw = { .05, .10, .30, .20, .05, .15, .10, .05 };
        const common_spec_dist q = make(qw);
        const run_out r1 = up_p.draft_sample ? run_rule(pw, qw, q, false, 4321) : run_out{ 0, 0 };
        const run_out r2 = fk_p.draft_sample ? run_rule(pw, qw, q, false, 4321) : run_out{ 0, 0 };
        CHECK(r1.chi2 == r2.chi2 && r1.acc == r2.acc && r1.chi2 < CHI2_CRIT,
              "same seed through both spellings: identical statistics (chi2 %.3f / %.3f, acceptance %.5f / %.5f)", r1.chi2, r2.chi2, r1.acc, r2.acc);
    }

    // A2. coverage (C1(b)): only the draft-model and MTP stages draw from q; every other type must be reported as ignoring the flag
    {
        CHECK(common_speculative_type_draws_from_q(COMMON_SPECULATIVE_TYPE_DRAFT), "draft stage draws from q");
        CHECK(common_speculative_type_draws_from_q(COMMON_SPECULATIVE_TYPE_MTP), "mtp stage draws from q");
        for (auto t : { COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
                        COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, COMMON_SPECULATIVE_TYPE_SUFFIX, COMMON_SPECULATIVE_TYPE_SUFFIX_AUTOMATON,
                        COMMON_SPECULATIVE_TYPE_EAGLE3, COMMON_SPECULATIVE_TYPE_DFLASH, COMMON_SPECULATIVE_TYPE_DSPARK }) {
            CHECK(!common_speculative_type_draws_from_q(t), "%s stage does not draw from q (the flag is ignored, so it must warn)",
                  common_speculative_type_to_str(t).c_str());
        }
    }

    // B. the rule under the mask
    test_masked_rule();

    // C. the real sampler (only when a model is given)
    // No arguments at all = the model-free run (the CMake llama_build_and_test entry). Checked BEFORE the parser: this fork's
    // common_params_parse refuses an empty command line ("error: --model is required") and the run returned 1 with every
    // check passing -- MEASURED 2026-11-06 on gate b1019ze and again after the memcmp fix (RUN_RC=1, 0 FAIL lines).
    if (argc < 2) {
        std::printf("no -m: the sampler arm is skipped\n%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
        return g_fail ? 1 : 0;
    }
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    if (params.model.path.empty()) {
        std::printf("no -m: the sampler arm is skipped\n%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
        return g_fail ? 1 : 0;
    }
    if (FILE * f = fopen(params.model.path.c_str(), "rb")) {
        fclose(f);
    } else {
        std::printf("model %s not present, sampler arm skipped\n", params.model.path.c_str());
        return g_fail ? 1 : 77;
    }
    ggml_backend_load_all();
    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        std::printf("FAIL: failed to init model\n");
        return 1;
    }
    test_sampler_with_grammar(model, params);

    std::printf("\n%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
