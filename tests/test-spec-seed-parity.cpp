// Same seed, same temperature: speculative decoding with sample-and-match must emit the SAME tokens as plain decoding.
// Sample-and-match draws once per verified position from the sampler's own RNG and accepts a draft token only if it equals the draw, so the draws line up
// with plain decoding one per emitted token. A rejected draft position still consumed exactly the draw plain decoding would have used at that position.
//
//   reference   G tokens, one decode + one sample per token, temperature T, seed S
//   spec        same seed/prompt: windows of 1 + D tokens (last sampled + D drafts), common_sampler_sample_and_accept_n, seq_rm back to the accepted prefix.
//               Drafts are the reference continuation (an oracle) corrupted on a fixed schedule, so the run sees full accepts, partial accepts and a reject at
//               draft position 0.
//   control     a different seed must change the reference output, else equality proves nothing
//
// Row S69-STRATA-S2-DIST-SAMPLER-POSITION-KEYED-RNG-2026-10-07-01. Strata (MIT) keys a Philox counter by (seed, position); read from the spec, no code copied.
// GATE-CLAIM-OK: unit test, no resource claims.
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

static constexpr int      G       = 40; // tokens to generate
static constexpr int      D       = 4;  // draft length
static constexpr uint32_t N_PROMPT = 8;

static llama_context * make_ctx(const common_params & params, llama_model * model) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = 1;
    cparams.n_rs_seq   = D + 4; // recurrent / hybrid models need snapshots to roll back a partial accept
    cparams.n_ctx      = 256;
    cparams.n_batch    = 64;
    cparams.n_ubatch   = 64;
    cparams.kv_unified = false;
    return llama_init_from_model(model, cparams);
}

static bool decode_tokens(llama_context * ctx, const llama_token * tok, int n, llama_pos pos0) {
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        common_batch_add(batch, tok[i], pos0 + i, { 0 }, true);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

using smpl_ptr = std::unique_ptr<common_sampler, decltype(&common_sampler_free)>;

static bool run_plain(llama_model * model, llama_context * ctx, const common_params_sampling & sp, const std::vector<llama_token> & prompt, std::vector<llama_token> & out) {
    smpl_ptr smpl { common_sampler_init(model, sp), common_sampler_free };
    if (!decode_tokens(ctx, prompt.data(), (int) prompt.size(), 0)) {
        return false;
    }
    int idx = (int) prompt.size() - 1;
    llama_pos pos = (llama_pos) prompt.size();
    for (int g = 0; g < G; ++g) {
        const llama_token id = common_sampler_sample(smpl.get(), ctx, idx);
        common_sampler_accept(smpl.get(), ctx, id, true);
        out.push_back(id);
        if (!decode_tokens(ctx, &id, 1, pos++)) {
            return false;
        }
        idx = 0;
    }
    return true;
}

// returns false on a decode failure; *refused is set when the memory refuses a partial rollback (the test then does not apply)
static bool run_spec(llama_model * model, llama_context * ctx, const common_params_sampling & sp, const std::vector<llama_token> & prompt,
                     const std::vector<llama_token> & ref, int n_vocab, std::vector<llama_token> & out, bool * refused, int * n_full, int * n_partial, int * n_zero,
                     bool extra_draw = false) {
    smpl_ptr smpl { common_sampler_init(model, sp), common_sampler_free };
    if (!decode_tokens(ctx, prompt.data(), (int) prompt.size(), 0)) {
        return false;
    }
    // first token comes from the prompt's last row, exactly as in plain decoding
    llama_token last = common_sampler_sample(smpl.get(), ctx, (int) prompt.size() - 1);
    common_sampler_accept(smpl.get(), ctx, last, true);
    out.push_back(last);
    llama_pos n_past = (llama_pos) prompt.size(); // positions decoded into the context
    int round = 0;
    while ((int) out.size() < G) {
        // draft = the reference continuation after the tokens emitted so far, corrupted on a schedule
        std::vector<llama_token> draft;
        for (int i = 0; i < D; ++i) {
            const size_t at = out.size() + i;
            llama_token t = at < ref.size() ? ref[at] : (llama_token) ((out.back() + 1) % n_vocab);
            if (round % 3 == 1 && i == 2) {
                t = (llama_token) ((t + 1) % n_vocab); // reject at draft position 2
            }
            if (round % 5 == 4 && i == 0) {
                t = (llama_token) ((t + 1) % n_vocab); // reject at draft position 0
            }
            draft.push_back(t);
        }
        std::vector<llama_token> window;
        window.push_back(last);
        window.insert(window.end(), draft.begin(), draft.end());
        if (!decode_tokens(ctx, window.data(), (int) window.size(), n_past)) {
            return false;
        }
        if (extra_draw && round == 2) {
            common_sampler_sample(smpl.get(), ctx, 0); // control: one stray draw from the sampler's RNG, which a stream-aligned comparison must notice
        }
        std::vector<int> idxs(window.size());
        for (size_t i = 0; i < idxs.size(); ++i) {
            idxs[i] = (int) i;
        }
        const std::vector<llama_token> got = common_sampler_sample_and_accept_n(smpl.get(), ctx, idxs, draft);
        const int k = (int) got.size(); // window rows 0..k-1 are now part of the sequence; got[k-1] is sampled but not decoded
        n_past += k;
        if (k == D + 1) {
            ++*n_full;
        } else if (k == 1) {
            ++*n_zero;
        } else {
            ++*n_partial;
        }
        if (k < D + 1 && !llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past, -1)) {
            fprintf(stderr, "partial seq_rm(%d, -1) refused at round %d (window %d tokens, kept %d, n_rs_seq %u)\n", (int) n_past, round, D + 1, k, (unsigned) llama_n_rs_seq(ctx));
            *refused = true;
            return true;
        }
        for (int i = 0; i < k && (int) out.size() < G; ++i) {
            out.push_back(got[i]);
        }
        last = got[k - 1];
        ++round;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 4242;
    params.sampling.temp = 0.8f;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<llama_token> prompt(N_PROMPT);
    for (uint32_t i = 0; i < N_PROMPT; ++i) {
        prompt[i] = (llama_token) ((13*i + 5) % (uint32_t) n_vocab);
    }

    common_params_sampling sp = params.sampling;
    sp.temp = 0.8f;
    sp.seed = 4242;

    std::vector<llama_token> ref, ref_other, spec;
    {
        llama_context * ctx = make_ctx(params, model);
        if (ctx == nullptr || !run_plain(model, ctx, sp, prompt, ref)) {
            fprintf(stderr, "%s : plain decode failed\n", __func__);
            return 1;
        }
        llama_free(ctx);
    }
    {
        common_params_sampling sp2 = sp;
        sp2.seed = 4243;
        llama_context * ctx = make_ctx(params, model);
        if (ctx == nullptr || !run_plain(model, ctx, sp2, prompt, ref_other)) {
            fprintf(stderr, "%s : plain decode (control seed) failed\n", __func__);
            return 1;
        }
        llama_free(ctx);
    }
    int n_ctl = 0;
    for (int i = 0; i < G; ++i) {
        n_ctl += ref[i] != ref_other[i];
    }

    bool refused = false;
    int n_full = 0, n_partial = 0, n_zero = 0;
    {
        llama_context * ctx = make_ctx(params, model);
        if (ctx == nullptr || !run_spec(model, ctx, sp, prompt, ref, n_vocab, spec, &refused, &n_full, &n_partial, &n_zero)) {
            fprintf(stderr, "%s : speculative decode failed\n", __func__);
            return 1;
        }
        llama_free(ctx);
    }

    std::vector<llama_token> spec_ctl;
    bool refused_ctl = false;
    int c1 = 0, c2 = 0, c3 = 0;
    {
        llama_context * ctx = make_ctx(params, model);
        if (ctx == nullptr || !run_spec(model, ctx, sp, prompt, ref, n_vocab, spec_ctl, &refused_ctl, &c1, &c2, &c3, true)) {
            fprintf(stderr, "%s : speculative decode (stray-draw control) failed\n", __func__);
            return 1;
        }
        llama_free(ctx);
    }
    int n_stray = 0;
    for (int i = 0; i < G && i < (int) spec_ctl.size(); ++i) {
        n_stray += spec_ctl[i] != ref[i];
    }
    printf("control (seed 4243 vs 4242): %d of %d tokens differ\n", n_ctl, G);
    if (!refused_ctl) {
        printf("control (one stray RNG draw in the spec run): %d of %d tokens differ\n", n_stray, G);
    }
    if (refused) {
        printf("spec leg skipped: the context offers no partial rollback here (n_rs_seq 0 on legacy-graph recurrent models: the server uses a state checkpoint instead)\n");
        return n_ctl == 0 ? 1 : 0;
    }
    int n_diff = 0, first = -1;
    for (int i = 0; i < G && i < (int) spec.size(); ++i) {
        if (spec[i] != ref[i]) {
            n_diff++;
            if (first < 0) first = i;
        }
    }
    printf("windows: %d full accept, %d partial accept, %d reject at draft 0\n", n_full, n_partial, n_zero);
    printf("spec vs plain at one seed: %d of %d tokens differ%s\n", n_diff, G, first >= 0 ? " (first at index shown below)" : "");
    if (first >= 0) {
        printf("first difference at token %d: plain %d, spec %d\n", first, ref[first], spec[first]);
    }
    if (n_ctl == 0) {
        fprintf(stderr, "FAIL: control seed gave the same tokens, equality above would prove nothing\n");
        return 1;
    }
    if (n_stray == 0) {
        fprintf(stderr, "FAIL: a stray RNG draw changed nothing, so this comparison cannot see a misaligned stream\n");
        return 1;
    }
    if (n_full == 0 || n_partial == 0 || n_zero == 0) {
        fprintf(stderr, "FAIL: window mix did not cover full, partial and zero accepts\n");
        return 1;
    }
    if (n_diff != 0) {
        fprintf(stderr, "FAIL: speculative output differs from plain at the same seed\n");
        return 1;
    }
    printf("all spec-seed-parity checks passed\n");
    return 0;
}
