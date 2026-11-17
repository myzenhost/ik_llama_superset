// Draft tokens after an end-of-generation token must not be accepted (row S69-LLAMACPP-PICK-DRAFT-STOP-AT-EOG-2026-10-18-01).
// A draft accepted past an EOG is never output (generation stops at the EOG) but the accepted cells stay in the context.
// Behaviour from upstream llama.cpp d280808f (#29638, MIT), read whole; no code copied.
//
// The target is forced with a +100 logit bias, so every verified position samples the biased token. Each of the three
// verify functions is checked:
//   common_sampler_sample_and_accept_n            (sample-and-match)
//   common_sampler_sample_and_accept_n_rejection  (--spec-draft-sample rejection sampling)
//   common_sampler_sample_and_accept_n with proposal dists
// Cases:
//   EOG mid draft      draft [EOG, EOG, EOG], target EOG everywhere -> only the first EOG is returned (1 token)
//   trailing EOG       draft [EOG] (a replayed draft may end on the target's EOG) -> the EOG is accepted, 2 tokens
//   control            a non-EOG token forced and drafted 3 times -> all 3 accepted + 1 sampled = 4 tokens
// GATE-CLAIM-OK: unit test, no resource claims.
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "spec-accept.h"

#include <clocale>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

using smpl_ptr = std::unique_ptr<common_sampler, decltype(&common_sampler_free)>;

enum verify_kind { VERIFY_MATCH, VERIFY_REJECTION, VERIFY_PROPOSAL };

static const char * kind_name(verify_kind k) {
    switch (k) {
        case VERIFY_MATCH:     return "sample_and_accept_n";
        case VERIFY_REJECTION: return "sample_and_accept_n_rejection";
        case VERIFY_PROPOSAL:  return "sample_and_accept_n(proposal_dists)";
    }
    return "?";
}

// decode a fixed prompt, then the window [last, draft...], all with logits, and verify the draft
static int verify(llama_model * model, const common_params & params, verify_kind kind, llama_token forced,
                  const std::vector<llama_token> & draft, std::vector<llama_token> & got) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_ctx     = 128;
    cparams.n_batch   = 64;
    cparams.n_ubatch  = 64;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> window;
    for (int i = 0; i < 6; ++i) {
        window.push_back((llama_token) ((17*i + 11) % n_vocab));
    }
    const int n_prompt = (int) window.size();
    window.insert(window.end(), draft.begin(), draft.end());

    llama_batch batch = llama_batch_init((int) window.size(), 0, 1);
    for (size_t i = 0; i < window.size(); ++i) {
        common_batch_add(batch, window[i], (llama_pos) i, { 0 }, true);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    if (!ok) {
        llama_free(ctx);
        return 1;
    }

    common_params_sampling sp = params.sampling;
    sp.seed = 1234;
    sp.temp = kind == VERIFY_MATCH ? 0.0f : 0.8f;
    sp.logit_bias.clear();
    sp.logit_bias.push_back({ forced, 100.0f });
    smpl_ptr smpl { common_sampler_init(model, sp), common_sampler_free };

    // rows: the last prompt token verifies draft[0], draft[i] verifies draft[i+1], the last row samples the bonus token
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = n_prompt - 1 + (int) i;
    }

    switch (kind) {
        case VERIFY_MATCH:
            got = common_sampler_sample_and_accept_n(smpl.get(), ctx, idxs, draft);
            break;
        case VERIFY_REJECTION: {
            std::vector<common_spec_dist> dists(draft.size());
            for (size_t i = 0; i < draft.size(); ++i) {
                dists[i].ids.assign(1, draft[i]);
                dists[i].probs.assign(1, 1.0f);
            }
            std::mt19937 rng(42);
            got = common_sampler_sample_and_accept_n_rejection(smpl.get(), ctx, idxs, draft, dists, rng);
            break;
        }
        case VERIFY_PROPOSAL: {
            std::vector<common_speculative_token_dist> dists(draft.size());
            for (size_t i = 0; i < draft.size(); ++i) {
                dists[i].ids.assign(1, draft[i]);
                dists[i].probs.assign(1, 1.0f);
            }
            got = common_sampler_sample_and_accept_n(smpl.get(), ctx, idxs, draft, dists);
            break;
        }
    }
    llama_free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    ggml_backend_load_all();

    // the model is a downloaded test file (see tests/CMakeLists.txt); absent = skipped (77), not failed
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
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token eog = llama_vocab_eos(vocab);
    if (eog == LLAMA_TOKEN_NULL || !llama_vocab_is_eog(vocab, eog)) {
        eog = llama_vocab_eot(vocab);
    }
    if (eog == LLAMA_TOKEN_NULL || !llama_vocab_is_eog(vocab, eog)) {
        // the random-weight models from test-llama-archs have no EOS/EOT; this test needs a real vocab
        fprintf(stderr, "model has no EOS/EOT token, skipping\n");
        return 77;
    }
    // a plain token for the control: the first id from 100 up that is not an end-of-generation token
    llama_token plain = 100;
    while (llama_vocab_is_eog(vocab, plain) || llama_vocab_is_control(vocab, plain)) {
        ++plain;
    }

    struct tcase {
        const char *             name;
        llama_token              forced;
        std::vector<llama_token> draft;
        size_t                   want;
    };
    const std::vector<tcase> cases = {
        { "EOG mid draft", eog,   { eog, eog, eog },       1 },
        { "trailing EOG",  eog,   { eog },                 2 },
        { "control plain", plain, { plain, plain, plain }, 4 },
    };

    int n_fail = 0;
    for (verify_kind kind : { VERIFY_MATCH, VERIFY_REJECTION, VERIFY_PROPOSAL }) {
        for (const auto & c : cases) {
            std::vector<llama_token> got;
            if (verify(model, params, kind, c.forced, c.draft, got) != 0) {
                fprintf(stderr, "%s / %s: decode failed\n", kind_name(kind), c.name);
                return 1;
            }
            bool all_forced = true;
            for (llama_token t : got) {
                all_forced = all_forced && t == c.forced;
            }
            const bool pass = got.size() == c.want && all_forced;
            printf("%-38s %-14s draft %zu tokens -> returned %zu (want %zu)%s %s\n", kind_name(kind), c.name, c.draft.size(),
                   got.size(), c.want, all_forced ? "" : " [a returned token is not the forced one]", pass ? "PASS" : "FAIL");
            n_fail += pass ? 0 : 1;
        }
    }
    printf("%s (%d failing)\n", n_fail == 0 ? "OK" : "FAILED", n_fail);
    return n_fail == 0 ? 0 : 1;
}
