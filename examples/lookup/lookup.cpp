#include "arg.h"
#include "ggml.h"
#include "common.h"
#include "ngram-cache.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include "log-upstream.h" // upstream LOG() semantics -- see common/log-upstream.h

int main(int argc, char ** argv){
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LOOKUP)) {
        return 1;
    }

    // max. number of additional tokens to draft if match is found
    // TOOK HEAD: header's common_params_speculative has a flat n_max, no nested .draft
    // sub-struct (S69-ARGCPP-SPEC-SHAPE-MISMATCH-2026-09-08-01, same shape decision as arg.cpp).
    const int n_draft = params.speculative.n_max;

    const auto output_limits = common_speculative_get_output_limits(params.n_batch, params.n_parallel, n_draft);
    params.n_outputs_max = output_limits.total;
    params.n_outputs_max_per_seq = output_limits.per_seq;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model
    // TOOK unsloth/master: HEAD's llama_init_from_gpt_params/llama_init_result/gpt_params are
    // NOT declared anywhere in the staged common.h (confirmed: common.h has adopted
    // common_init_from_params/common_init_result throughout, gpt_params-shaped API dropped) --
    // HEAD's side does not compile. `vocab` (used below at llama_vocab_is_eog) is only
    // declared on this side too.
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx, params.prompt, true, true);

    common_ngram_cache ngram_cache_context;
    common_ngram_cache ngram_cache_dynamic;
    common_ngram_cache ngram_cache_static;
    int64_t t_draft_flat_us = 0;
    int64_t t_draft_us = 0;

    {
        // Fill up context ngram cache with tokens from user input:
        const int64_t t_start_draft_us = ggml_time_us();
        common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, inp, inp.size(), false);

        // TOOK HEAD (both hunks below): flat lookup_cache_static/lookup_cache_dynamic exist
        // directly on common_params_speculative in the staged header; master's nested
        // .ngram_cache sub-struct does not (same finding as arg.cpp's .ngram_cache.* --
        // S69-ARGCPP-SPEC-SHAPE-MISMATCH-2026-09-08-01).
        if (!params.speculative.lookup_cache_static.empty()) {
            try {
                ngram_cache_static = common_ngram_cache_load(params.speculative.lookup_cache_static);
            } catch (std::ifstream::failure const &) {
                LOG_ERR("failed to open static lookup cache: %s", params.speculative.lookup_cache_static.c_str());
                exit(1);
            }
        }

        if (!params.speculative.lookup_cache_dynamic.empty()) {
            try {
                ngram_cache_dynamic = common_ngram_cache_load(params.speculative.lookup_cache_dynamic);
            } catch (std::ifstream::failure const &) {} // if the file does not exist it will simply be created at the end of the program
        }

        t_draft_flat_us += ggml_time_us() - t_start_draft_us;
    }

    const int max_context_size     = llama_n_ctx(ctx);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        // TOOK unsloth/master: LOG() matches the rest of this file's unconflicted logging
        // convention (LOG_DBG/LOG_INF/LOG throughout); functionally equivalent to HEAD's
        // fprintf(stderr, ...), no compile-shape difference either way.
        LOG("%s", common_token_to_piece(ctx, id).c_str());
    }

    fflush(stderr);

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    {
        common_batch batch = common_batch_get_one(ctx, inp.data(), n_input - 1);
        llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());

        batch = common_batch_get_one(ctx, &inp.back(), 1);
        llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
    }

    const auto t_enc_end = ggml_time_us();

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    int n_past = inp.size();

    bool has_eos = false;

    // TOOK unsloth/master's field name: header renamed params.sparams -> params.sampling
    // (common.h's own comment on the field confirms this; sampling.h/.cpp already staged
    // against .sampling). Variable name `smpl` also matches this file's own unconflicted tail
    // (common_perf_print(ctx, smpl) / common_sampler_free(smpl), both already present below).
    struct common_sampler * smpl = common_sampler_init(model, params.sampling);

    std::vector<llama_token> draft;

    common_batch batch_tgt(ctx);

    const auto t_dec_start = ggml_time_us();

    while (true) {
        // print current draft sequence
        LOG_DBG("drafted %s\n", string_from(ctx, draft).c_str());

        int i_dft = 0;
        while (true) {
            // sample from the target model
            // TOOK: `smpl` variable name (unsloth/master, since ctx_sampling/params.sparams no
            // longer exist), but common_sampler_accept's REAL staged signature (sampling.h:415)
            // is (common_sampler*, llama_context* ctx_main, llama_token id, bool is_generated)
            // -- 4 args, matching HEAD's call shape, not master's 3-arg
            // common_sampler_accept(smpl, id, true) (which would not compile against the
            // staged declaration). Hybrid: master's name, HEAD's arity.
            llama_token id = common_sampler_sample(smpl, ctx, i_dft);

            common_sampler_accept(smpl, ctx, id, true);

            const std::string token_str = common_token_to_piece(ctx, id);

            if (!params.use_color) {
                LOG("%s", token_str.c_str());
            }

            if (llama_vocab_is_eog(vocab, id)) {
                has_eos = true;
            }

            ++n_predict;

            // check if the target token matches the draft
            if (i_dft < (int) draft.size() && id == draft[i_dft]) {
                LOG_DBG("the sampled target token matches the %dth drafted token (%d, '%s') - accepted\n", i_dft, id, token_str.c_str());
                ++n_accept;
                ++n_past;
                ++i_dft;
                inp.push_back(id);
                {
                    // Update context ngram cache with the newly accepted token:
                    const int64_t t_start_draft_us = ggml_time_us();
                    common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, inp, 1, false);
                    t_draft_us += ggml_time_us() - t_start_draft_us;
                }

                if (params.use_color) {
                    // color accepted draft token
                    LOG("\033[34m%s\033[0m", token_str.c_str());
                    fflush(stdout);
                }
                continue;
            }

            if (params.use_color) {
                LOG("%s", token_str.c_str());
            }
            fflush(stdout);


            LOG_DBG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", id, token_str.c_str());

            draft.clear();
            draft.push_back(id);
            inp.push_back(id);
            {
                // Update context ngram cache with the newly accepted token:
                const int64_t t_start_draft_us = ggml_time_us();
                common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, inp, 1, false);
                t_draft_us += ggml_time_us() - t_start_draft_us;
            }
            break;
        }

        if ((params.n_predict > 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }

        // KV cache management
        // clean the cache of draft tokens that weren't accepted
        llama_memory_seq_rm(llama_get_memory(ctx), 0, n_past, -1);

        batch_tgt.clear();
        batch_tgt.add(draft[0], n_past, 0, true);

        // Draft already contains a single token sampled from the model:
        GGML_ASSERT(draft.size() == 1);
        GGML_ASSERT(draft[0] == inp.back());
        const int64_t t_start_draft_us = ggml_time_us();

        common_ngram_cache_draft(inp, draft, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, ngram_cache_context, ngram_cache_dynamic, ngram_cache_static);

        for (size_t i = 1; i < draft.size(); ++i) {
            batch_tgt.add(draft[i], n_past + i, 0, true);
        }

        t_draft_us += ggml_time_us() - t_start_draft_us;
        n_drafted += draft.size() - 1;

        llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get());
        ++n_past;

        draft.erase(draft.begin());
    }

    auto t_dec_end = ggml_time_us();

    // Update dynamic ngram cache with context ngram cache and save it to disk:
    common_ngram_cache_merge(ngram_cache_dynamic, ngram_cache_context);
    // TOOK HEAD: flat field, same reasoning as the two lookup_cache_static/dynamic hunks above.
    common_ngram_cache_save(ngram_cache_dynamic, params.speculative.lookup_cache_dynamic);

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft      = %d\n", n_draft);
    LOG_INF("n_predict    = %d\n", n_predict);
    LOG_INF("n_drafted    = %d\n", n_drafted);
    LOG_INF("t_draft_flat = %.2f ms\n", t_draft_flat_us*1e-3);
    LOG_INF("t_draft      = %.2f ms, %.2f us per token, %.2f tokens per second\n",
            t_draft_us*1e-3, 1.0f*t_draft_us/n_drafted, n_drafted/(1e-6*t_draft_us));
    LOG_INF("n_accept     = %d\n", n_accept);
    LOG_INF("accept       = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\ntarget:\n\n");
    common_perf_print(ctx, smpl);

    // STRUCTURAL FIX beyond conflict markers: this file also had a stray
    // `common_sampler_free(ctx_sampling);` right after common_sampler_free(smpl) above --
    // `ctx_sampling` was HEAD's dropped variable name (see the sampler-init hunk); git's 3-way
    // merge left both cleanup calls with no conflict marker since they don't textually overlap.
    // Removed the dangling one (would not compile, ctx_sampling is undeclared here).
    common_sampler_free(smpl);


    llama_backend_free();

    LOG("\n\n");

    return 0;
}
