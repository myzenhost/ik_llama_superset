// Test for state restore with fragmented KV cache
// This tests the fix for: https://github.com/ggml-org/llama.cpp/issues/17527
// The issue was that state restore required contiguous KV cache slots,
// which fails when the cache is fragmented.
//
// The fix changes find_slot(ubatch, true) to find_slot(ubatch, false)
// in state_read_meta(), allowing non-contiguous slot allocation.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cstring>

int main(int argc, char ** argv) {
    common_params params;

    params.sampling.seed = 1234;
    params.kv_unified = true;
    params.n_parallel = 3;
    params.n_ctx = 256;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // init

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    GGML_UNUSED(model);

    // tokenize prompt
    std::vector<llama_token> tokens(70, 1);

    // interleave the 3 sequences:
    // 01201230123...
    llama_batch batch = llama_batch_init(params.n_parallel*tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        for (int s = 0; s < params.n_parallel; ++s) {
            common_batch_add(batch, tokens[i], i, {s}, false);
        }
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode seq 0\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s : processed prompt on seq 0, 1, 2 (%zu tokens each)\n", __func__, tokens.size());

    // Save state of seq 1
    std::vector<uint8_t> seq_state(llama_state_seq_get_size(ctx, 1, 0));
    const size_t ncopy = llama_state_seq_get_data(ctx, seq_state.data(), seq_state.size(), 1, 0);
    if (ncopy != seq_state.size()) {
        fprintf(stderr, "%s : failed to save seq 1 state\n", __func__);
        return 1;
    }
    fprintf(stderr, "%s : saved seq 1 state, %zu bytes\n", __func__, ncopy);

    // Reference for the restored ROWS (not just "the restore returned success"): decode one more token on the intact seq 1
    // and keep its logits; the same token decoded after the fragmented restore must give the same logits. Rows written to the
    // wrong cells (or the wrong run of a scattered restore) change what that token attends to.
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const llama_token probe_tok = 5;
    std::vector<float> logits_ref(n_vocab);
    {
        common_batch_clear(batch);
        common_batch_add(batch, probe_tok, (int) tokens.size(), {1}, true);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to decode the reference token on the intact seq 1\n", __func__);
            return 1;
        }
        memcpy(logits_ref.data(), llama_get_logits_ith(ctx, -1), n_vocab * sizeof(float));
        // take the probe token back out of seq 1: the saved state (taken above) does not contain it
        llama_memory_seq_rm(llama_get_memory(ctx), 1, (int) tokens.size(), -1);
    }

    // clear seq 1 to create a "hole" in the KV cache (fragmentation)
    // 0.20.20.20.2....
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 1, -1, -1);
    fprintf(stderr, "%s : cleared seq 1 to create fragmentation\n", __func__);

    // Now the cache has holes where seq 1 was
    // This creates fragmentation - there's no contiguous block large enough
    // for the seq 1 state if we only look for contiguous slots

    // Restore seq 1 state into seq 1 (should work with non-contiguous allocation)
    // We use seq 1 since it's a valid sequence ID (0 to n_parallel-1)
    // Before the fix, this would fail with "failed to find available cells in kv cache"
    const size_t nset = llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 1, 0);
    if (nset != seq_state.size()) {
        fprintf(stderr, "%s : FAILED to restore seq state into fragmented cache (got %zu, expected %zu)\n",
                __func__, nset, seq_state.size());
        fprintf(stderr, "%s : This is the bug - state restore fails with fragmented KV cache\n", __func__);
        llama_batch_free(batch);
        return 1;
    }
    fprintf(stderr, "%s : restored state into seq 1, %zu bytes\n", __func__, nset);

    // The restored rows must reproduce the reference logits. Cell order differs from the original (the rows now sit in the
    // holes), so attention sums in a different order: allow float noise, not a different value.
    {
        common_batch_clear(batch);
        common_batch_add(batch, probe_tok, (int) tokens.size(), {1}, true);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to decode the probe token after the restore\n", __func__);
            return 1;
        }
        const float * got = llama_get_logits_ith(ctx, -1);
        float max_diff = 0.0f;
        float max_abs  = 0.0f;
        int   argmax_ref = 0, argmax_got = 0;
        for (int i = 0; i < n_vocab; ++i) {
            max_diff = std::max(max_diff, std::fabs(got[i] - logits_ref[i]));
            max_abs  = std::max(max_abs, std::fabs(logits_ref[i]));
            if (logits_ref[i] > logits_ref[argmax_ref]) { argmax_ref = i; }
            if (got[i]        > got[argmax_got])        { argmax_got = i; }
        }
        fprintf(stderr, "%s : probe logits after the fragmented restore: max |diff| %g (max |logit| %g), argmax %d vs %d\n",
                __func__, (double) max_diff, (double) max_abs, argmax_got, argmax_ref);
        if (argmax_got != argmax_ref || max_diff > 1e-3f * std::max(1.0f, max_abs)) {
            fprintf(stderr, "%s : FAILED - the restored rows do not reproduce the original sequence\n", __func__);
            llama_batch_free(batch);
            return 1;
        }
        llama_memory_seq_rm(llama_get_memory(ctx), 1, (int) tokens.size(), -1);   // leave seq 1 as the section below expects it
    }

    // Verify we can decode with the restored state
    // Generate one token to verify the restored state is usable
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.sampling.seed));

    auto next_token = llama_sampler_sample(smpl, ctx, -1);
    auto next_token_str = common_token_to_piece(ctx, next_token);

    common_batch_clear(batch);
    common_batch_add(batch, next_token, (int)tokens.size(), {1}, true);

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode with restored state\n", __func__);
        llama_sampler_free(smpl);
        llama_batch_free(batch);
        return 1;
    }

    fprintf(stderr, "%s : successfully decoded with restored state, generated: '%s'\n", __func__, next_token_str.c_str());
    fprintf(stderr, "%s : SUCCESS - state restore works with fragmented KV cache\n", __func__);

    llama_sampler_free(smpl);
    llama_batch_free(batch);

    return 0;
}
