// The proposal-dists verifier (common_sampler_sample_and_accept_n(..., common_speculative_token_dist), the DFlash2 / stochastic-draft path) must
// accept a draft token with probability min(1, p/q) where p is the distribution the target sampler really draws from.
// Row S69-VLLM-W2-DFLASH2-ACCEPT-USES-CUR-P-2026-10-27-01 (found by the vLLM engine-merge: docs/agent-reports/engine-vllm.md:39 "analytic
// only, not executed"; this test EXECUTES it).
//
// The suspect line read cur_p.data[j].p. After sampler_queue the .p field is not the softmax of the final logits: temperature rescales the
// logits without recomputing it, DIST leaves it as exp(logit - max), a greedy draw never writes it. So the accept test and the residual
// compared the wrong p with q.
//
// Method (no formula of the verifier is reused as the oracle): the oracle is the empirical distribution of the TARGET sampler itself over N
// seeds, f(y) = #{common_sampler_sample == y} / N, drawn from the same logits row. A correct verifier with a one-hot draft q = delta_x has
//   P(accept) = p(x)             and the first emitted token is distributed as p.
// Arms, for x = the most frequent target token and x = the second most frequent:
//   proposal   common_sampler_sample_and_accept_n(..., proposal dists)  (the code under test)
//   rejection  common_sampler_sample_and_accept_n_rejection             (CONTROL: already uses the post-chain distribution)
// Both are compared with f. The control must pass on the unfixed tree (otherwise the harness, not the verifier, is wrong).
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
#include <map>
#include <memory>
#include <random>
#include <vector>

using smpl_ptr = std::unique_ptr<common_sampler, decltype(&common_sampler_free)>;

enum arm_kind { ARM_PROPOSAL, ARM_REJECTION };

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
    return smpl_ptr { common_sampler_init(model, sp), common_sampler_free };
}

struct arm_result {
    double                        accept_rate = 0.0;
    std::map<llama_token, double> first;       // first emitted token histogram (fraction)
};

static arm_result run_arm(llama_model * model, const common_params & params, llama_context * ctx2, int n_prompt, float temp,
                          llama_token x, arm_kind kind, int n_trials) {
    arm_result out;
    int n_acc = 0;
    const std::vector<int> idxs = { n_prompt - 1, n_prompt };
    const std::vector<llama_token> draft = { x };
    for (int s = 0; s < n_trials; ++s) {
        smpl_ptr smpl = make_sampler(model, params, temp, 1000 + (uint32_t) s);
        std::vector<llama_token> got;
        if (kind == ARM_PROPOSAL) {
            std::vector<common_speculative_token_dist> dists(1);
            dists[0].ids.assign(1, x);
            dists[0].probs.assign(1, 1.0f);
            got = common_sampler_sample_and_accept_n(smpl.get(), ctx2, idxs, draft, dists);
        } else {
            std::vector<common_spec_dist> dists(1);
            dists[0].ids.assign(1, x);
            dists[0].probs.assign(1, 1.0f);
            std::mt19937 rng(777 + s);
            got = common_sampler_sample_and_accept_n_rejection(smpl.get(), ctx2, idxs, draft, dists, rng);
        }
        if (got.empty()) {
            continue;
        }
        n_acc += (got.size() == 2 && got[0] == x) ? 1 : 0;
        out.first[got[0]] += 1.0;
    }
    out.accept_rate = (double) n_acc / n_trials;
    for (auto & kv : out.first) {
        kv.second /= n_trials;
    }
    return out;
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

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    ggml_backend_load_all();

    if (FILE * f = fopen(params.model.path.c_str(), "rb")) {
        fclose(f);
    } else {
        fprintf(stderr, "%s : model %s not present, skipping\n", __func__, params.model.path.c_str());
        return 77;
    }

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<llama_token> prompt;
    for (int i = 0; i < 6; ++i) {
        prompt.push_back((llama_token) ((17*i + 11) % n_vocab));
    }
    const int n_prompt = (int) prompt.size();

    const int N = 4000;
    int n_fail = 0;
    const float temps[] = { 1.0f, 0.0f };
    for (float temp : temps) {
        // oracle: the target sampler's own empirical distribution at the last prompt row
        llama_context * ctx1 = decode_window(model, params, prompt);
        if (ctx1 == nullptr) {
            fprintf(stderr, "decode failed\n");
            return 1;
        }
        std::map<llama_token, double> target;
        for (int s = 0; s < N; ++s) {
            smpl_ptr smpl = make_sampler(model, params, temp, 1000 + (uint32_t) s);
            target[common_sampler_sample(smpl.get(), ctx1, n_prompt - 1)] += 1.0;
        }
        llama_free(ctx1);
        std::vector<std::pair<double, llama_token>> ranked;
        for (auto & kv : target) {
            kv.second /= N;
            ranked.emplace_back(kv.second, kv.first);
        }
        std::sort(ranked.rbegin(), ranked.rend());
        std::vector<llama_token> xs = { ranked[0].second };
        xs.push_back(ranked.size() > 1 ? ranked[1].second : (llama_token) ((ranked[0].second + 1) % n_vocab));

        for (llama_token x : xs) {
            std::vector<llama_token> window = prompt;
            window.push_back(x);
            llama_context * ctx2 = decode_window(model, params, window);
            if (ctx2 == nullptr) {
                fprintf(stderr, "decode failed\n");
                return 1;
            }
            const auto it = target.find(x);
            const double p_x = it == target.end() ? 0.0 : it->second;
            for (arm_kind kind : { ARM_REJECTION, ARM_PROPOSAL }) {
                const arm_result r = run_arm(model, params, ctx2, n_prompt, temp, x, kind, N);
                const double tv   = total_variation(r.first, target);
                // 4000 trials: sd(accept) <= 0.008, so 0.05 is > 6 sd for the accept rate; TV between two 4000-sample
                // histograms over <= 40 support points is about 0.05, so 0.15 separates a real distortion from noise
                const bool ok = std::fabs(r.accept_rate - p_x) <= 0.05 && tv <= 0.15;
                printf("temp %.1f  x=%-6d p(x)=%.3f  %-10s accept %.3f  TV(first token, target) %.3f  %s\n", temp, (int) x, p_x,
                       kind == ARM_PROPOSAL ? "proposal" : "rejection", r.accept_rate, tv, ok ? "PASS" : "FAIL");
                n_fail += ok ? 0 : 1;
            }
            llama_free(ctx2);
        }
    }
    printf("%s (%d failing)\n", n_fail == 0 ? "OK" : "FAILED", n_fail);
    return n_fail == 0 ? 0 : 1;
}
