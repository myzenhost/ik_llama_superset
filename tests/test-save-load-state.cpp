#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"

#include <algorithm>
#include <cfloat>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct llama_batch_ptr {
    llama_batch batch;

    llama_batch_ptr(int32_t n_tokens, int32_t embd, int32_t n_seq_max)
        : batch{llama_batch_init(n_tokens, embd, n_seq_max)} {}

    ~llama_batch_ptr() { llama_batch_free(batch); }

    llama_batch_ptr(const llama_batch_ptr &) = delete;
    llama_batch_ptr & operator=(const llama_batch_ptr &) = delete;
    llama_batch_ptr(llama_batch_ptr &&) = default;
    llama_batch_ptr & operator=(llama_batch_ptr &&) = default;

    llama_batch & get() { return batch; }
    const llama_batch & get() const { return batch; }
};

static llama_tokens generate_tokens(llama_context * ctx, llama_sampler * smpl, int & n_past, int32_t n_predict, llama_seq_id seq_id) {
    llama_tokens result;
    llama_batch_ptr batch(1, 0, 1);

    for (int i = 0; i < n_predict; i++) {
        auto next_token = llama_sampler_sample(smpl, ctx, -1);

        LOG("%d ", next_token);
        result.push_back(next_token);

        common_batch_clear(batch.get());
        common_batch_add(batch.get(), next_token, n_past, {seq_id}, true);

        if (llama_decode(ctx, batch.get())) {
            LOG_ERR("\n%s: failed to evaluate\n", __func__);
            return {};
        }
        n_past++;
    }

    return result;
}

// Test 1: baseline
// - decode all but the last token
// - save state to disk
// - decode the last token
// - generate n_predict tokens
static llama_tokens test_baseline(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens) {
    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    auto n_past = 0;
    if (!common_prompt_batch_decode(ctx.get(), tokens, (int)tokens.size(), n_past, params.n_batch, params.out_file, true)) {
        LOG_ERR("%s: failed to decode prompt\n", __func__);
        return {};
    }

    LOG("\n=== Test 1: baseline ===\n");

    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 0);
    if (result.empty()) {
        return {};
    }

    LOG("\n");

    return result;
}


// Test 2: sequence removal isolation
// - decode the same prefix into two sequences
// - remove sequence 0
// - verify that sequence 1 remains unchanged
static bool test_seq_rm_isolated(
        struct llama_model         * model,
        const struct common_params & params,
        const llama_tokens         & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx      = 256;
    params_ctx.n_seq_max  = 2;
    params_ctx.kv_unified = true;

    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
    if (!ctx) {
        LOG_ERR("%s: failed to create context\n", __func__);
        return false;
    }

    LOG("\n=== Test 2: sequence removal isolation ===\n");

    const size_t n_tokens = tokens.size() < 128 ? tokens.size() : 128;
    for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
        llama_batch_ptr batch(n_tokens, 0, 1);
        for (size_t i = 0; i < n_tokens; ++i) {
            common_batch_add(batch.get(), tokens[i], i, { seq_id }, false);
        }

        if (llama_decode(ctx.get(), batch.get())) {
            LOG_ERR("%s: failed to decode prompt for sequence %d\n", __func__, seq_id);
            return false;
        }
    }

    const auto get_seq_state = [&](llama_seq_id seq_id, std::vector<uint8_t> & state) {
        const size_t state_size = llama_state_seq_get_size(ctx.get(), seq_id, 0);
        if (state_size == 0) {
            LOG_ERR("%s: sequence state is empty\n", __func__);
            return false;
        }

        state.resize(state_size);
        const size_t ncopy = llama_state_seq_get_data(ctx.get(), state.data(), state.size(), seq_id, 0);
        if (ncopy != state.size()) {
            LOG_ERR("%s: sequence state length %zu does not match expected length %zu\n",
                    __func__, ncopy, state.size());
            return false;
        }

        return true;
    };

    std::vector<uint8_t> state_before;
    if (!get_seq_state(1, state_before)) {
        return false;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, -1, -1)) {
        LOG_ERR("%s: failed to remove sequence 0\n", __func__);
        return false;
    }

    std::vector<uint8_t> state_after;
    if (!get_seq_state(1, state_after)) {
        return false;
    }

    if (state_before != state_after) {
        LOG_ERR("%s: removing sequence 0 changed sequence 1\n", __func__);
        return false;
    }

    LOG("PASS\n");
    return true;
}


// Test 3: state load
// - create a new context
// - load state from file
// - replay the last prompt token
// - generate n_predict tokens and compare against expected result
static bool test_state_load(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 3: state load ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Generate tokens
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 0);
    if (result.empty()) {
        return false;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


// Test 4: seq copy (host)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the CPU path
// - generate n_predict tokens on seq 1 and compare against expected result
static bool test_seq_cp_host(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 4: seq copy (host) ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (CPU path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size(ctx.get(), 0, 0));
        const size_t ncopy = llama_state_seq_get_data(ctx.get(), seq_store.data(), seq_store.size(), 0, 0);
        if (ncopy != seq_store.size()) {
            LOG_ERR("\n%s: seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data(ctx.get(), seq_store.data(), seq_store.size(), 1, 0);
        if (nset != seq_store.size()) {
            LOG_ERR("\n%s: seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens on seq 1
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 1);
    if (result.empty()) {
        return false;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


// Test 5: seq copy (device)
// - create a multi-seq context
// - load state from file
// - replay the last prompt token
// - migrate KV cache from seq 0 to seq 1 via the on-device path
// - generate n_predict tokens on seq 1 and compare against expected result
static bool test_seq_cp_device(struct llama_model * model, const struct common_params & params, const llama_tokens & tokens, const llama_tokens & expected_result) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};

    auto sparams = llama_sampler_chain_default_params();
    auto smpl = llama_sampler_ptr{llama_sampler_chain_init(sparams)};
    llama_sampler_chain_add(smpl.get(), llama_sampler_init_dist(params.sampling.seed));

    LOG("\n=== Test 5: seq copy (device) ===\n");

    // Load state from file
    llama_tokens unused_sts(tokens.size());
    size_t n_token_count_out = 0;

    if (!llama_state_load_file(ctx.get(), params.out_file.data(), unused_sts.data(), unused_sts.size(), &n_token_count_out)) {
        LOG_ERR("\n%s: failed to load state\n", __func__);
        return false;
    }

    LOG_TRC("%s: loaded state with %zu tokens\n", __func__, n_token_count_out);

    // Replay last token
    int n_past = (int) n_token_count_out - 1;
    if (!common_replay_last_token(ctx.get(), tokens.back(), n_past)) {
        return false;
    }
    n_past++;

    // Migrate KV cache from seq 0 to seq 1 (on-device path)
    {
        std::vector<uint8_t> seq_store(llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE));
        const size_t ncopy = llama_state_seq_get_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 0, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (ncopy != seq_store.size()) {
            LOG_ERR("\n%s: seq copy data length %zd does not match expected length %zd\n", __func__, ncopy, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 0 copied, %zd bytes\n", __func__, ncopy);

        llama_memory_clear(llama_get_memory(ctx.get()), true);
        LOG_TRC("%s: kv cache cleared\n", __func__);

        const size_t nset = llama_state_seq_set_data_ext(ctx.get(), seq_store.data(), seq_store.size(), 1, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (nset != seq_store.size()) {
            LOG_ERR("\n%s: seq set data length %zd does not match expected length %zd\n", __func__, nset, seq_store.size());
            return false;
        }
        LOG_TRC("%s: seq 1 restored, %zd bytes\n", __func__, nset);
    }

    // Generate tokens on seq 1
    auto result = generate_tokens(ctx.get(), smpl.get(), n_past, params.n_predict, 1);
    if (result.empty()) {
        return false;
    }

    if (result != expected_result) {
        LOG_ERR("\n%s: error: generation differs from expected\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}


// PRIOR-ART-NONE: every other test here decodes ONE sequence per llama_batch; none put two sequences in one batch.
//
// Test 6: two sequences in ONE batch must give the logits two separate decodes give.
// - reference: 32 tokens into sequence 0, then 32 different tokens into sequence 1, one batch each
// - mixed:     the same 64 tokens in a single batch (sequence 0 run, then sequence 1 run), logits on each run's last token
// The legacy mamba graph scans one sequence per ubatch, so llama_decode must cut the ubatch at the sequence boundary;
// before 2026-10-07 it aborted on GGML_ASSERT(n_kv == 1) at the first such batch (and in the constructor reserve for
// any n_seq_max > 1).
// Set when Test 6's leak control could not see a deliberate leak; main then exits 77 (ctest SKIP_RETURN_CODE).
static bool g_t6_inconclusive = false;

// Test 6 layer trace (diagnostic, runs only with LLAMA_T6_TRACE=1). Test 6 compares logits only, so on a backend where
// the mixed batch differs it cannot say WHICH op first differs. This captures, through cb_eval, the row of sequence 0's
// last token in every 1-2D F32 per-layer tensor of the first graph computed, once for the reference decode (sequence 0
// alone) and once for the mixed decode, then prints the first tensor whose rows differ and every layer's |diff|.
// cb_eval forces a graph split at every captured node, which can itself change kernel choice (fusion, graph capture),
// so the final logits are compared UNDER the trace too: if the logit diff disappears there, the trace changed the
// numerics and its layer list is not evidence.
struct t6_trace_rec {
    std::string name;
    int64_t     ne1;
    std::vector<float> row;
};
struct t6_trace_state {
    size_t n_run = 0;
    int    n_graph = 0;   // l_out-0 occurrences: graphs (ubatches) computed
    std::vector<t6_trace_rec> recs;
};
static bool t6_trace_wanted(const char * name) {
    static const char * pfx[] = { "l_out-", "attn_output-", "kqv_out-", "ffn_out-", "conv_output_silu-",
                                  "attn_norm-", "ffn_inp-", "result_norm", "result_output" };
    for (const char * p : pfx) {
        if (strncmp(name, p, strlen(p)) == 0) {
            return true;
        }
    }
    return false;
}
static int t6_trace_cb(struct ggml_tensor * t, bool ask, void * ud) {  // this fork's ggml_backend_sched_eval_callback returns int
    auto * st = (t6_trace_state *) ud;
    if (!t6_trace_wanted(t->name)) {
        return false;
    }
    if (ask) {
        return true;
    }
    if (strcmp(t->name, "l_out-0") == 0) {
        st->n_graph++;
    }
    if (st->n_graph > 1 || t->type != GGML_TYPE_F32 || ggml_n_dims(t) > 2 || t->nb[0] != sizeof(float)) {
        return true;
    }
    const int64_t row = t->ne[1] >= (int64_t) st->n_run ? (int64_t) st->n_run - 1 : 0;
    t6_trace_rec r;
    r.name = t->name;
    r.ne1  = t->ne[1];
    r.row.resize(t->ne[0]);
    ggml_backend_tensor_get(t, r.row.data(), (size_t) row * t->nb[1], t->ne[0] * sizeof(float));
    st->recs.push_back(std::move(r));
    return true;
}

static void test_mixed_seq_layer_trace(
        struct llama_model         * model,
        const struct common_params & params,
        const llama_tokens         & tokens) {
    const char * e = getenv("LLAMA_T6_TRACE");
    if (!e || strcmp(e, "1") != 0) {
        return;
    }
    const size_t n_run = tokens.size() < 64 ? tokens.size() / 2 : 32;
    if (n_run < 2) {
        return;
    }
    const auto tok = [&](llama_seq_id s, size_t i) { return tokens[(i + (size_t) s) % tokens.size()]; };
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    LOG("\n=== Test 6 layer trace (LLAMA_T6_TRACE=1) ===\n");

    t6_trace_state st_ref, st_mix;
    st_ref.n_run = st_mix.n_run = n_run;
    std::vector<float> lg_ref, lg_mix;
    for (int arm = 0; arm < 2; ++arm) {
        auto & st = arm == 0 ? st_ref : st_mix;
        auto params_ctx = common_context_params_to_llama(params);
        params_ctx.n_ctx             = 256;
        params_ctx.n_seq_max         = 2;
        params_ctx.kv_unified        = true;
        params_ctx.cb_eval           = t6_trace_cb;
        params_ctx.cb_eval_user_data = &st;
        auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
        if (!ctx) {
            LOG_ERR("%s: failed to create context\n", __func__);
            return;
        }
        const int n_seq = arm == 0 ? 1 : 2;
        llama_batch_ptr batch((int32_t) (n_seq * n_run), 0, 1);
        for (llama_seq_id s = 0; s < n_seq; ++s) {
            for (size_t i = 0; i < n_run; ++i) {
                common_batch_add(batch.get(), tok(s, i), (llama_pos) i, { s }, i == n_run - 1);
            }
        }
        if (llama_decode(ctx.get(), batch.get())) {
            LOG_ERR("%s: decode failed (arm %d)\n", __func__, arm);
            return;
        }
        const float * lg = llama_get_logits_ith(ctx.get(), (int32_t) n_run - 1);
        if (!lg) {
            LOG_ERR("%s: no logits (arm %d)\n", __func__, arm);
            return;
        }
        (arm == 0 ? lg_ref : lg_mix).assign(lg, lg + n_vocab);
    }
    float lg_diff = 0.0f;
    for (int i = 0; i < n_vocab; ++i) {
        lg_diff = std::max(lg_diff, std::fabs(lg_ref[i] - lg_mix[i]));
    }
    LOG("graphs computed: reference %d, mixed %d; captured tensors: reference %zu, mixed %zu\n",
        st_ref.n_graph, st_mix.n_graph, st_ref.recs.size(), st_mix.recs.size());
    LOG("sequence 0 final logits under the trace: max |diff| %g%s\n", lg_diff,
        lg_diff < 1e-5f ? "  (the trace REMOVED the difference: the layer list below is not evidence)" : "");
    if (st_ref.n_graph != st_mix.n_graph) {
        // MEASURED 2026-10-18 (Qwen3.5-0.8B, CUDA and CPU): reference 1 graph, mixed 21 -- llama_decode cut the two-sequence
        // batch into one-token-per-sequence ubatches, so the mixed rows are other positions under another batch shape.
        LOG("  the mixed decode was cut into %d ubatches (reference %d): per-tensor rows are not comparable, and the logit "
            "difference includes the one-token vs prefill batch shape (Test 6 measures that as its one-token floor)\n",
            st_mix.n_graph, st_ref.n_graph);
        return;
    }
    const size_t n = std::min(st_ref.recs.size(), st_mix.recs.size());
    bool first_found = false;
    for (size_t i = 0; i < n; ++i) {
        const auto & a = st_ref.recs[i];
        const auto & b = st_mix.recs[i];
        if (a.name != b.name || a.row.size() != b.row.size()) {
            LOG("  capture %zu: graph order differs (reference %s, mixed %s); stopping the comparison here\n",
                i, a.name.c_str(), b.name.c_str());
            break;
        }
        float d = 0.0f, mx = 0.0f;
        for (size_t j = 0; j < a.row.size(); ++j) {
            d  = std::max(d, std::fabs(a.row[j] - b.row[j]));
            mx = std::max(mx, std::fabs(a.row[j]));
        }
        const bool differs = d > 1e-5f * std::max(1.0f, mx);
        if (differs && !first_found) {
            first_found = true;
            LOG("  FIRST DIFFERING TENSOR: %s\n", a.name.c_str());
        }
        LOG("  %-22s rows ref %4lld mix %4lld  max |diff| %-12g max |x| %g%s\n", a.name.c_str(), (long long) a.ne1,
            (long long) b.ne1, d, mx, differs ? "  <-" : "");
    }
    if (!first_found) {
        LOG("  no captured tensor differs\n");
    }
}

static bool test_mixed_seq_batch(
        struct llama_model         * model,
        const struct common_params & params,
        const llama_tokens         & tokens) {
    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx      = 256;
    params_ctx.n_seq_max  = 2;
    params_ctx.kv_unified = true;

    const size_t n_run = tokens.size() < 64 ? tokens.size() / 2 : 32;
    if (n_run < 2) {
        LOG("\n=== Test 6: skipped (prompt too short) ===\n");
        return true;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // Sequence 1 starts half the prompt later, so its window is (mostly) different text. It used to be the prompt rotated
    // by ONE token: the leak control then fed sequence 0 its own next tokens one position early, nearly the content it
    // already attends to, and the deliberate leak moved its logits by 1.14 against a 1.42 tolerance (stories15M-q4_0,
    // GPU-backend arm of b1018c, 2026-10-18), so Test 6 reported itself blind. Row
    // S69-FORK-SLS-TEST6-LEAK-CONTROL-WEAK-ROTATE-BY-ONE-2026-10-18-01.
    const size_t seq1_shift = std::max<size_t>(1, tokens.size() / 2);
    const auto tok = [&](llama_seq_id s, size_t i) { return tokens[(i + (size_t) s * seq1_shift) % tokens.size()]; };

    LOG("\n=== Test 6: two sequences in one batch ===\n");

    std::vector<float> ref[2];
    {
        auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
        if (!ctx) {
            LOG_ERR("%s: failed to create context\n", __func__);
            return false;
        }
        for (llama_seq_id s = 0; s < 2; ++s) {
            llama_batch_ptr batch((int32_t) n_run, 0, 1);
            for (size_t i = 0; i < n_run; ++i) {
                common_batch_add(batch.get(), tok(s, i), (llama_pos) i, { s }, i == n_run - 1);
            }
            if (llama_decode(ctx.get(), batch.get())) {
                LOG_ERR("%s: reference decode failed for sequence %d\n", __func__, s);
                return false;
            }
            const float * lg = llama_get_logits_ith(ctx.get(), (int32_t) n_run - 1);
            if (!lg) {
                LOG_ERR("%s: no reference logits for sequence %d\n", __func__, s);
                return false;
            }
            ref[s].assign(lg, lg + n_vocab);
        }
    }

    std::vector<float> mix[2];
    {
        auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
        if (!ctx) {
            LOG_ERR("%s: failed to create context\n", __func__);
            return false;
        }
        llama_batch_ptr batch((int32_t) (2 * n_run), 0, 1);
        for (llama_seq_id s = 0; s < 2; ++s) {
            for (size_t i = 0; i < n_run; ++i) {
                common_batch_add(batch.get(), tok(s, i), (llama_pos) i, { s }, i == n_run - 1);
            }
        }
        if (llama_decode(ctx.get(), batch.get())) {
            LOG_ERR("%s: mixed-sequence decode failed\n", __func__);
            return false;
        }
        for (llama_seq_id s = 0; s < 2; ++s) {
            const float * lg = llama_get_logits_ith(ctx.get(), (int32_t) (s * n_run + n_run - 1));
            if (!lg) {
                LOG_ERR("%s: no mixed logits for sequence %d\n", __func__, s);
                return false;
            }
            mix[s].assign(lg, lg + n_vocab);
        }
    }

    // Noise floor. The mixed batch has 2*n_run rows, the reference n_run, and CPU kernels pick their path and reduction
    // order by row count (a quantized matmul re-quantizes activations per batch shape), so an ABSOLUTE tolerance mixes
    // two effects: on stories15M-q4_0 the mixed decode differed by 0.28 with argmax always equal, F16 by 0.011
    // (2026-10-17, row S69-FORK-TEST-SAVE-LOAD-STATE-T6-MIXED-SEQ-TOLERANCE-QUANT-2026-10-17-01). The floor batch has the
    // mixed batch's shape but ONE sequence: sequence s's n_run tokens, then n_run more tokens continuing sequence s at
    // positions n_run..2*n_run-1. Row n_run-1 cannot see later positions (causal mask), so its logits differ from the
    // reference by batch shape only -- never by another sequence.
    std::vector<float> flr[2];
    for (llama_seq_id s = 0; s < 2; ++s) {
        auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
        if (!ctx) {
            LOG_ERR("%s: failed to create context\n", __func__);
            return false;
        }
        llama_batch_ptr batch((int32_t) (2 * n_run), 0, 1);
        for (size_t i = 0; i < 2 * n_run; ++i) {
            common_batch_add(batch.get(), tok(s, i), (llama_pos) i, { s }, i == n_run - 1);
        }
        if (llama_decode(ctx.get(), batch.get())) {
            LOG_ERR("%s: noise-floor decode failed for sequence %d\n", __func__, s);
            return false;
        }
        const float * lg = llama_get_logits_ith(ctx.get(), (int32_t) n_run - 1);
        if (!lg) {
            LOG_ERR("%s: no noise-floor logits for sequence %d\n", __func__, s);
            return false;
        }
        flr[s].assign(lg, lg + n_vocab);
    }

    // One-token floor. The floor above keeps the batch whole, but llama_decode is free to cut the mixed batch smaller:
    // on Qwen3.5-0.8B it cut it into one-token-per-sequence ubatches (Test 6 layer trace, 2026-10-18: 21 graphs for the
    // mixed decode, 1 for the reference), so the mixed logits came from one-token steps while the reference came from a
    // prefill. On CUDA those two shapes differ by 0.30 with no second sequence anywhere (the prefill floor read 0), and
    // Test 6 failed as a leak (row S69-FORK-T6-CUDA-MIXED-SEQ-BATCH-LOGIT-DIFF-QWEN35-2026-10-18-01). This arm decodes
    // sequence s alone, one token per llama_decode, so the floor covers the smallest shape the mixed decode can take.
    std::vector<float> flr1[2];
    for (llama_seq_id s = 0; s < 2; ++s) {
        auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
        if (!ctx) {
            LOG_ERR("%s: failed to create context\n", __func__);
            return false;
        }
        for (size_t i = 0; i < n_run; ++i) {
            llama_batch_ptr batch(1, 0, 1);
            common_batch_add(batch.get(), tok(s, i), (llama_pos) i, { s }, i == n_run - 1);
            if (llama_decode(ctx.get(), batch.get())) {
                LOG_ERR("%s: one-token floor decode failed for sequence %d at %zu\n", __func__, s, i);
                return false;
            }
        }
        const float * lg = llama_get_logits_ith(ctx.get(), 0);
        if (!lg) {
            LOG_ERR("%s: no one-token floor logits for sequence %d\n", __func__, s);
            return false;
        }
        flr1[s].assign(lg, lg + n_vocab);
    }

    // Positive control: the same mixed batch with sequence 1's tokens ALSO tagged sequence 0, i.e. a deliberate leak.
    // If the comparison below cannot tell this from the clean mixed decode, it cannot catch a real leak either.
    // Recurrent and hybrid models do not accept a token in two sequences, so the control is skipped there (reported).
    std::vector<float> leak;
    const bool leak_arm = !llama_model_is_recurrent(model) && !llama_model_is_hybrid(model);
    if (leak_arm) {
        auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
        if (!ctx) {
            LOG_ERR("%s: failed to create context\n", __func__);
            return false;
        }
        // Interleaved, not run-after-run: sequence 0 then sees positions 0,0,1,1,..., which the batch validator accepts
        // (src/llama-batch.cpp:378 rejects only DECREASING positions). Run-after-run restarted sequence 0 at position 0
        // and every model on the memory-module path refused the batch (31 of 111 in the 2026-10-17 matrix).
        llama_batch_ptr batch((int32_t) (2 * n_run), 0, 2);
        for (size_t i = 0; i < n_run; ++i) {
            common_batch_add(batch.get(), tok(1, i), (llama_pos) i, { 0, 1 }, false);
            common_batch_add(batch.get(), tok(0, i), (llama_pos) i, { 0 }, i == n_run - 1);
        }
        if (llama_decode(ctx.get(), batch.get())) {
            LOG_ERR("%s: leak-control decode failed\n", __func__);
            return false;
        }
        const float * lg = llama_get_logits_ith(ctx.get(), (int32_t) (2 * n_run - 1));
        if (!lg) {
            LOG_ERR("%s: no leak-control logits\n", __func__);
            return false;
        }
        leak.assign(lg, lg + n_vocab);
    }

    const auto max_abs_diff = [&](const std::vector<float> & a, const std::vector<float> & b) {
        float d = 0.0f;
        for (int v = 0; v < n_vocab; ++v) {
            d = std::max(d, std::fabs(a[v] - b[v]));
        }
        return d;
    };
    const auto argmax = [&](const std::vector<float> & a) {
        int m = 0;
        for (int v = 1; v < n_vocab; ++v) {
            if (a[v] > a[m]) m = v;
        }
        return m;
    };

    // Tolerance: what batch shape alone measurably moves (k_floor times the floor), plus a few float32 ulps of the
    // largest logit. It used to be max(1e-2, k_floor * floor). On the random-weight F32 test models the floor is exactly
    // 0 and a deliberate leak moves the logits by only 4e-8..7e-6 (matrix of 111 models, 2026-10-17), so the absolute
    // 1e-2 hid every leak there: 62 of 111 failed as "check is blind" with nothing wrong in the engine.
    const float k_floor = 4.0f;
    const float k_ulps  = 4.0f;
    const auto top2_margin = [&](const std::vector<float> & a) {
        float t1 = -INFINITY, t2 = -INFINITY;
        for (int v = 0; v < n_vocab; ++v) {
            if (a[v] > t1) { t2 = t1; t1 = a[v]; } else if (a[v] > t2) { t2 = a[v]; }
        }
        return t1 - t2;
    };
    float tol[2];
    for (int s = 0; s < 2; ++s) {
        const float d_mix    = max_abs_diff(ref[s], mix[s]);
        const float d_floor0 = max_abs_diff(ref[s], flr[s]);
        const float d_floor1 = max_abs_diff(ref[s], flr1[s]);
        // Only recurrent and hybrid memory cuts a mixed batch per sequence; an attention-only model keeps the 2*n_run batch
        // whole (n_ubatch >= 64), so widening its floor would only blunt the leak control that runs on exactly those models.
        const bool  cut_small = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
        const float d_floor  = cut_small ? std::max(d_floor0, d_floor1) : d_floor0;
        // shape-matched view: the mixed decode against the one-token decode of the same sequence alone
        LOG("sequence %d: floors prefill %g, one-token %g; mixed vs one-token %g\n",
            s, (double) d_floor0, (double) d_floor1, (double) max_abs_diff(flr1[s], mix[s]));
        float scale = 0.0f;
        for (int v = 0; v < n_vocab; ++v) {
            scale = std::max(scale, std::fabs(ref[s][v]));
        }
        tol[s] = std::max(k_floor * d_floor, k_ulps * FLT_EPSILON * scale);
        const int arg_ref = argmax(ref[s]), arg_mix = argmax(mix[s]);
        // An argmax flip is only evidence when the reference's top two logits are further apart than the tolerance;
        // inside it the flip is a near-tie moved by batch shape (stories15M-q4_0 on CUDA: floor 0.48, a random-weight
        // model on CPU: floor 3.6e-6, both flipped with the mixed diff EQUAL to the floor).
        const float margin = top2_margin(ref[s]);
        const bool  arg_counts = margin > tol[s];
        LOG("sequence %d: max |logit diff| mixed %g, batch-shape floor %g, max |logit| %g, tolerance %g, "
            "argmax ref %d mixed %d, top-2 margin %g (%s)\n",
            s, (double) d_mix, (double) d_floor, (double) scale, (double) tol[s], arg_ref, arg_mix, (double) margin,
            arg_counts ? "argmax checked" : "near-tie, argmax not checked");
        if ((arg_counts && arg_ref != arg_mix) || d_mix > tol[s]) {
            LOG_ERR("%s: error: sequence %d logits differ between separate and mixed decode (max |diff| %g > tolerance %g, "
                    "floor %g, argmax ref %d mixed %d)\n",
                    __func__, s, (double) d_mix, (double) tol[s], (double) d_floor, arg_ref, arg_mix);
            return false;
        }
    }

    if (leak_arm) {
        const float d_leak = max_abs_diff(ref[0], leak);
        LOG("leak control: sequence 0 with sequence 1 leaked in, max |logit diff| %g (must exceed tolerance %g)\n",
            (double) d_leak, (double) tol[0]);
        if (d_leak <= tol[0] && argmax(leak) == argmax(ref[0])) {
            // Not a failure of the engine: this model's logits do not move enough under a leak for the comparison to see
            // one, so Test 6 has verified NOTHING on it. Reported as a skip (exit 77), never as a pass.
            LOG_ERR("%s: INCONCLUSIVE: a deliberate cross-sequence leak stays inside the tolerance -- the check is blind "
                    "on this model\n", __func__);
            g_t6_inconclusive = true;
            return true;
        }
    } else {
        LOG("leak control: skipped (recurrent/hybrid model does not accept a token in two sequences)\n");
    }

    LOG("\nPASS\n");
    return true;
}

// Test 7: the server's speculative rollback must leave the state a plain decode leaves.
// llama-server on a context that cannot remove a partial sequence (recurrent / hybrid, n_rs_seq == 0) does, per draft
// (tools/server/server-context.cpp:6206 and :7487):
//   save the PARTIAL_ONLY state of the sequence -> decode the verify batch -> on partial acceptance restore that state,
//   seq_rm past the checkpoint, and replay the accepted tokens.
// - replay arm: prompt, save, verify batch of k tokens, restore, seq_rm, replay the first m tokens, then one more token
// - plain arm:  prompt, the same m tokens as one batch, then the same one more token (fresh context, no rollback)
// Both arms run the same batches through the same graph shapes, so a correct restore gives the SAME logits; the only
// allowed difference is what two runs of the plain arm already differ by. Speculative decoding on CUDA x hybrid
// Qwen3.5 changed the greedy text with top-2 probability gaps up to 0.91 (row
// S69-FORK-SERVER-SPEC-DECODE-LOSSY-AT-TEMP0-HYBRID-QWEN35-2026-10-18-01); this isolates the rollback from the drafter.
// Runs before Test 6, so a Test 6 failure cannot hide it.
static bool test_spec_rollback_replay(
        struct llama_model         * model,
        const struct common_params & params,
        const llama_tokens         & tokens) {
    const size_t k = 8; // verify batch
    const size_t m = 3; // accepted, replayed
    if (tokens.size() < 16 + k + 1) {
        LOG("\n=== Test 7: skipped (prompt too short) ===\n");
        return true;
    }
    const size_t n_p = std::min<size_t>(48, tokens.size() - k - 1);

    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx     = 256;
    params_ctx.n_seq_max = 1;

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const llama_seq_id s0 = 0;

    LOG("\n=== Test 7: speculative rollback (save, verify, restore, replay) ===\n");

    // decode tokens[i0, i1) at their own positions, one batch; logits for every row when all_logits
    const auto decode_run = [&](llama_context * ctx, size_t i0, size_t i1, bool all_logits) {
        llama_batch_ptr batch((int32_t) (i1 - i0), 0, 1);
        for (size_t i = i0; i < i1; ++i) {
            common_batch_add(batch.get(), tokens[i], (llama_pos) i, { s0 }, all_logits || i == i1 - 1);
        }
        return llama_decode(ctx, batch.get()) == 0;
    };
    const auto rows = [&](llama_context * ctx, int n) {
        std::vector<float> out;
        for (int i = 0; i < n; ++i) {
            const float * lg = llama_get_logits_ith(ctx, i);
            if (!lg) {
                return std::vector<float>{};
            }
            out.insert(out.end(), lg, lg + n_vocab);
        }
        return out;
    };

    // plain arm: prompt, m tokens, then one token
    const auto plain = [&](std::vector<float> & out) {
        auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
        if (!ctx || !decode_run(ctx.get(), 0, n_p, false) || !decode_run(ctx.get(), n_p, n_p + m, true)) {
            return false;
        }
        out = rows(ctx.get(), (int) m);
        if (!decode_run(ctx.get(), n_p + m, n_p + m + 1, true)) {
            return false;
        }
        const auto last = rows(ctx.get(), 1);
        out.insert(out.end(), last.begin(), last.end());
        return out.size() == (m + 1) * (size_t) n_vocab;
    };

    std::vector<float> ref, ref2;
    if (!plain(ref) || !plain(ref2)) {
        LOG_ERR("%s: plain-arm decode failed\n", __func__);
        return false;
    }

    // REAL reference: what temperature 0 must reproduce is the one-token-at-a-time decode, not another batch of the
    // same shape. Prompt, then m+1 single-token decodes. Compared by argmax beyond a near-tie margin (the batch shape
    // differs, so rounding differs).
    std::vector<float> seq_ref;
    {
        auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
        if (!ctx || !decode_run(ctx.get(), 0, n_p, false)) {
            LOG_ERR("%s: sequential reference prompt failed\n", __func__);
            return false;
        }
        for (size_t i = n_p; i < n_p + m + 1; ++i) {
            if (!decode_run(ctx.get(), i, i + 1, true)) {
                LOG_ERR("%s: sequential reference decode failed\n", __func__);
                return false;
            }
            const auto r1 = rows(ctx.get(), 1);
            seq_ref.insert(seq_ref.end(), r1.begin(), r1.end());
        }
    }

    // replay arm. mode 0: restore right after save (mechanics only); mode 1: verify batch, restore, seq_rm, replay
    // (the server's rollback); mode 2 = POSITIVE CONTROL: verify batch, NO restore, seq_rm, replay -- the "soft
    // rollback" defect of ggml-org/llama.cpp#20075, where only positions rewind and the recurrent state stays past the
    // draft. On a model with recurrent state this arm MUST be flagged; if it is not, the comparison is blind.
    bool soft_refused = false;
    const auto replay = [&](int mode, std::vector<float> & out) {
        auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
        if (!ctx || !decode_run(ctx.get(), 0, n_p, false)) {
            return false;
        }
        const llama_state_seq_flags fl = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        // upstream's contract (merged 2026-11-05, llama_kv_cache::state_write; tests/llama-context case "ps"): the PARTIAL_ONLY state
        // of a cache with no sliding window and nothing recurrent is EMPTY -- seq_rm alone rolls it back. An empty checkpoint is a
        // valid one: there is nothing to save or restore, and the seq_rm + replay below is the whole rollback.
        std::vector<uint8_t> ckpt(llama_state_seq_get_size_ext(ctx.get(), s0, fl));
        if (!ckpt.empty() && llama_state_seq_get_data_ext(ctx.get(), ckpt.data(), ckpt.size(), s0, fl) != ckpt.size()) {
            LOG_ERR("%s: checkpoint save failed\n", __func__);
            return false;
        }
        if (mode >= 1 && !decode_run(ctx.get(), n_p, n_p + k, true)) {
            return false;
        }
        if (mode != 2 && !ckpt.empty() && llama_state_seq_set_data_ext(ctx.get(), ckpt.data(), ckpt.size(), s0, fl) != ckpt.size()) {
            LOG_ERR("%s: checkpoint restore failed\n", __func__);
            return false;
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx.get()), s0, (llama_pos) n_p, -1)) {
            if (mode == 2) {
                soft_refused = true; // the memory refuses a partial removal it cannot honour: that IS the guard
                return true;
            }
            LOG_ERR("%s: seq_rm past the checkpoint failed\n", __func__);
            return false;
        }
        if (!decode_run(ctx.get(), n_p, n_p + m, true)) {
            return false;
        }
        out = rows(ctx.get(), (int) m);
        if (!decode_run(ctx.get(), n_p + m, n_p + m + 1, true)) {
            return false;
        }
        const auto last = rows(ctx.get(), 1);
        out.insert(out.end(), last.begin(), last.end());
        return out.size() == (m + 1) * (size_t) n_vocab;
    };

    std::vector<float> rep_mech, rep_full, rep_soft;
    if (!replay(0, rep_mech) || !replay(1, rep_full) || !replay(2, rep_soft)) {
        LOG_ERR("%s: replay-arm decode failed\n", __func__);
        return false;
    }

    // per row: max |diff| against the plain arm, argmax, the plain arm's top-2 margin
    const auto row_stats = [&](const std::vector<float> & a, const std::vector<float> & b, size_t r,
                               float & d, int & am, int & bm, float & margin, float & scale) {
        const float * x = a.data() + r * n_vocab;
        const float * y = b.data() + r * n_vocab;
        d = 0.0f; am = 0; bm = 0; scale = 0.0f;
        float t1 = -INFINITY, t2 = -INFINITY;
        for (int v = 0; v < n_vocab; ++v) {
            d = std::max(d, std::fabs(x[v] - y[v]));
            scale = std::max(scale, std::fabs(x[v]));
            if (x[v] > x[am]) am = v;
            if (y[v] > y[bm]) bm = v;
            if (x[v] > t1) { t2 = t1; t1 = x[v]; } else if (x[v] > t2) { t2 = x[v]; }
        }
        margin = t1 - t2;
    };

    bool ok = true;
    for (size_t r = 0; r <= m; ++r) {
        float d0, dm, df, mg, sc;
        int a0, b0, am, bm, af, bf;
        row_stats(ref, ref2,     r, d0, a0, b0, mg, sc);
        row_stats(ref, rep_mech, r, dm, am, bm, mg, sc);
        row_stats(ref, rep_full, r, df, af, bf, mg, sc);
        // allowed: what two plain runs differ by, times 4, plus a few float32 ulps of the largest logit
        const float tol = std::max(4.0f * d0, 4.0f * FLT_EPSILON * sc);
        LOG("row %zu (%s): plain-vs-plain %g | restore-only %g (argmax %d vs %d) | verify+restore %g (argmax %d vs %d) | "
            "tolerance %g, top-2 margin %g\n",
            r, r < m ? "replayed" : "after replay", (double) d0, (double) dm, bm, am, (double) df, bf, af,
            (double) tol, (double) mg);
        if (dm > tol || df > tol || (mg > tol && (am != bm || af != bf))) {
            ok = false;
        }
        // against the one-token reference: argmax only, and only where that reference is not a near-tie
        float ds, smg, ssc;
        int sa, sb;
        row_stats(seq_ref, rep_full, r, ds, sa, sb, smg, ssc);
        const bool seq_bad = sa != sb && smg > 1.0f;
        LOG("        vs one-token decode: max|diff| %g, argmax %d vs %d, margin %g%s\n", (double) ds, sb, sa,
            (double) smg, seq_bad ? "  <-- ARGMAX CHANGED" : "");
        if (seq_bad) {
            ok = false;
        }
    }

    // positive control: the soft rollback must be visible to the same comparison on a model with recurrent state
    if (llama_model_has_recurrent(model)) {
        if (soft_refused) {
            LOG("positive control: seq_rm refused a partial removal without a restore (the memory guards it)\n");
        } else {
            float worst_soft = 0.0f;
            bool caught = false;
            for (size_t r = 0; r <= m; ++r) {
                float d0, dsf, mg, sc;
                int a0, b0, as, bs;
                row_stats(ref, ref2,     r, d0, a0, b0, mg, sc);
                row_stats(ref, rep_soft, r, dsf, as, bs, mg, sc);
                const float tol = std::max(4.0f * d0, 4.0f * FLT_EPSILON * sc);
                worst_soft = std::max(worst_soft, dsf);
                caught = caught || dsf > tol;
            }
            LOG("positive control (soft rollback, no restore): worst max|diff| %g -> %s\n", (double) worst_soft,
                caught ? "DETECTED" : "NOT DETECTED");
            if (!caught) {
                LOG_ERR("%s: error: BLIND -- a rollback that skips the state restore was not detected\n", __func__);
                return false;
            }
        }
    } else {
        LOG("positive control: skipped (no recurrent state; seq_rm alone is a correct rollback here)\n");
    }

    if (!ok) {
        LOG_ERR("%s: error: logits after a speculative rollback differ from a plain decode of the same tokens\n", __func__);
        return false;
    }

    LOG("\nPASS\n");
    return true;
}

// Test 8: a verify batch must score each row as a one-token-at-a-time decode does. llama-server spec decoding on CUDA
// x hybrid Qwen3.5 accepted far fewer drafts than the CPU path given the SAME drafts (prose round 1: 10/68 vs 178/231),
// and the reference token it lost had a top-2 probability gap of 0.91 -- far beyond batch-shape rounding. Test 7 cannot
// see this: its replay and plain arms both decode the m tokens as ONE batch. Here the k-token batch after the prompt is
// compared row by row against k single-token decodes. Shape-rounding differences are allowed; an argmax change is a
// failure only where the sequential top-2 logit margin is above MARGIN (not a near-tie).
static bool test_verify_vs_sequential(
        struct llama_model         * model,
        const struct common_params & params,
        const llama_tokens         & tokens) {
    const size_t k = 16; // llama-server's default --draft-max
    if (tokens.size() < 16 + k) {
        LOG("\n=== Test 8: skipped (prompt too short) ===\n");
        return true;
    }
    const size_t n_p = tokens.size() - k;
    const float MARGIN = 1.0f; // logits: below this the two top tokens are a near-tie

    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx     = 4096; // as llama-server -c 4096
    params_ctx.n_seq_max = 1;

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    LOG("\n=== Test 8: verify batch (%zu tokens after a %zu-token prompt) vs one-token decodes ===\n", k, n_p);

    const auto decode_run = [&](llama_context * ctx, size_t i0, size_t i1, bool all_logits) {
        llama_batch_ptr batch((int32_t) (i1 - i0), 0, 1);
        for (size_t i = i0; i < i1; ++i) {
            common_batch_add(batch.get(), tokens[i], (llama_pos) i, { 0 }, all_logits || i == i1 - 1);
        }
        return llama_decode(ctx, batch.get()) == 0;
    };

    std::vector<float> bat((size_t) k * n_vocab), seq((size_t) k * n_vocab);
    {
        auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
        if (!ctx || !decode_run(ctx.get(), 0, n_p, false) || !decode_run(ctx.get(), n_p, n_p + k, true)) {
            LOG_ERR("%s: batch arm decode failed\n", __func__);
            return false;
        }
        for (size_t r = 0; r < k; ++r) {
            const float * lg = llama_get_logits_ith(ctx.get(), (int32_t) r);
            if (!lg) { LOG_ERR("%s: no logits for batch row %zu\n", __func__, r); return false; }
            std::copy(lg, lg + n_vocab, bat.begin() + r * n_vocab);
        }
    }
    {
        auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
        if (!ctx || !decode_run(ctx.get(), 0, n_p, false)) {
            LOG_ERR("%s: sequential arm prompt failed\n", __func__);
            return false;
        }
        for (size_t r = 0; r < k; ++r) {
            if (!decode_run(ctx.get(), n_p + r, n_p + r + 1, true)) {
                LOG_ERR("%s: sequential decode %zu failed\n", __func__, r);
                return false;
            }
            const float * lg = llama_get_logits_ith(ctx.get(), 0);
            std::copy(lg, lg + n_vocab, seq.begin() + r * n_vocab);
        }
    }

    bool ok = true;
    float worst = 0.0f;
    int decisive = 0;
    for (size_t r = 0; r < k; ++r) {
        const float * x = seq.data() + r * n_vocab;
        const float * y = bat.data() + r * n_vocab;
        float d = 0.0f, t1 = -INFINITY, t2 = -INFINITY;
        int ax = 0, ay = 0;
        for (int v = 0; v < n_vocab; ++v) {
            d = std::max(d, std::fabs(x[v] - y[v]));
            if (x[v] > x[ax]) ax = v;
            if (y[v] > y[ay]) ay = v;
            if (x[v] > t1) { t2 = t1; t1 = x[v]; } else if (x[v] > t2) { t2 = x[v]; }
        }
        worst = std::max(worst, d);
        decisive += (t1 - t2) > MARGIN ? 1 : 0;
        const bool bad = ax != ay && (t1 - t2) > MARGIN;
        LOG("row %2zu: max|diff| %-10g argmax seq %6d batch %6d  seq top-2 margin %g%s\n", r, (double) d, ax, ay,
            (double) (t1 - t2), bad ? "  <-- ARGMAX CHANGED" : "");
        ok = ok && !bad;
    }
    LOG("worst row max|diff| %g\n", (double) worst);

    // detector power: a changed argmax counts only on a row whose top-2 margin is above MARGIN, so a row at or below
    // it can never fail. On 2026-10-18 stories15M fed 100 random ids had ONE such row of 16 on the host (margin 1.018)
    // and NONE on the device backend (0.955), so the test could not have failed whatever the batch did. Too few
    // decided rows is a void run, reported as such rather than passed.
    const int MIN_DECISIVE = 4;
    LOG("detector power: %d of %zu rows have a top-2 margin above %g (need %d)\n", decisive, k, (double) MARGIN,
        MIN_DECISIVE);
    if (decisive < MIN_DECISIVE) {
        LOG_ERR("%s: error: VACUOUS -- only %d of %zu rows have a top-2 margin above %g; a changed token is invisible "
                "on every other row (use a natural-text prompt)\n", __func__, decisive, k, (double) MARGIN);
        return false;
    }

    // positive control: the same detector, fed rows shifted by one (batch row r vs one-token row r+1 = a known
    // off-by-one row mapping), must flag at least one row; a detector that cannot see that is blind
    int shifted_flags = 0;
    for (size_t r = 0; r + 1 < k; ++r) {
        const float * x = seq.data() + (r + 1) * n_vocab;
        const float * y = bat.data() + r * n_vocab;
        float t1 = -INFINITY, t2 = -INFINITY;
        int ax = 0, ay = 0;
        for (int v = 0; v < n_vocab; ++v) {
            if (x[v] > x[ax]) ax = v;
            if (y[v] > y[ay]) ay = v;
            if (x[v] > t1) { t2 = t1; t1 = x[v]; } else if (x[v] > t2) { t2 = x[v]; }
        }
        shifted_flags += (ax != ay && (t1 - t2) > MARGIN) ? 1 : 0;
    }
    LOG("positive control (rows shifted by one): %d of %zu rows flagged -> %s\n", shifted_flags, k - 1,
        shifted_flags > 0 ? "DETECTED" : "NOT DETECTED");
    if (shifted_flags == 0) {
        LOG_ERR("%s: error: BLIND -- an off-by-one row mapping was not detected\n", __func__);
        return false;
    }
    if (!ok) {
        LOG_ERR("%s: error: a verify-batch row picks a different token than a one-token decode (not a near-tie)\n", __func__);
        return false;
    }
    LOG("\nPASS\n");
    return true;
}

// Test 9: replay llama-server's own first verify step, token for token. Test 8 passed on CUDA x hybrid Qwen3.5 while
// the server, given a draft equal to its own temperature-0 reference, accepted 7 of 16 and then emitted a different
// token where the reference top-2 gap was 0.91. Test 8 could not reproduce the server: it tokenizes its prompt from
// text without special-token parsing, so the "<think>" turn the server generated became plain text, and it verified
// 16 rows where the server verifies 17 (the sampled token plus 16 drafts). This test takes the server's prompt text
// (tokenized as the server does, special tokens parsed) and the server's generated ids, then runs the verify batch
// under each of the server's other conditions: the prompt split where the server places its context checkpoint, a
// partial-state read (the speculative checkpoint) right before the verify, and 16 vs 17 rows. Each arm is compared
// row by row with one-token decodes. The arm that fails names the condition.
//   LLAMA_T9_PROMPT_FILE  prompt text exactly as POSTed to /completion
//   LLAMA_T9_GEN_IDS      whitespace-separated token ids the server generated without speculation
static bool test_server_verify_replay(
        struct llama_model         * model,
        const struct common_params & params) {
    const char * pf = std::getenv("LLAMA_T9_PROMPT_FILE");
    const char * gf = std::getenv("LLAMA_T9_GEN_IDS");
    if (pf == nullptr || gf == nullptr) {
        LOG("\n=== Test 9: skipped (LLAMA_T9_PROMPT_FILE / LLAMA_T9_GEN_IDS not set) ===\n");
        return true;
    }
    std::string text;
    {
        FILE * f = std::fopen(pf, "rb");
        if (!f) { LOG_ERR("%s: cannot open %s\n", __func__, pf); return false; }
        char buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
        std::fclose(f);
    }
    llama_tokens gen;
    {
        FILE * f = std::fopen(gf, "rb");
        if (!f) { LOG_ERR("%s: cannot open %s\n", __func__, gf); return false; }
        int id;
        while (std::fscanf(f, "%d", &id) == 1) gen.push_back(id);
        std::fclose(f);
    }
    const auto * vocab = llama_model_get_vocab(model);
    const llama_tokens prompt = common_tokenize(vocab, text, true, true); // as the server: add + parse special
    const size_t K = 17;  // rows the server verifies: the sampled token + --draft-max 16
    if (prompt.size() < 8 || gen.size() < K + 1) {
        LOG_ERR("%s: need a prompt of 8+ tokens and %zu+ generated ids (got %zu, %zu)\n", __func__, K + 1,
                prompt.size(), gen.size());
        return false;
    }
    const size_t n_p   = prompt.size();
    const size_t split = n_p - 4; // the server creates its context checkpoint 4 tokens before the prompt end
    const float MARGIN = 1.0f;
    const int n_vocab  = llama_vocab_n_tokens(vocab);

    auto params_ctx = common_context_params_to_llama(params);
    params_ctx.n_ctx     = 4096; // llama-server -c 4096 -np 1, default batch sizes
    params_ctx.n_seq_max = 1;
    params_ctx.n_batch   = 2048;
    params_ctx.n_ubatch  = 512;

    LOG("\n=== Test 9: server verify replay (%zu-token prompt, %zu generated ids, split at %zu) ===\n", n_p, gen.size(),
        split);

    const auto decode = [&](llama_context * ctx, const llama_tokens & t, size_t i0, size_t i1, llama_pos p0, bool all) {
        llama_batch_ptr batch((int32_t) (i1 - i0), 0, 1);
        for (size_t i = i0; i < i1; ++i) {
            common_batch_add(batch.get(), t[i], p0 + (llama_pos) (i - i0), { 0 }, all || i == i1 - 1);
        }
        return llama_decode(ctx, batch.get()) == 0;
    };
    const auto state_read = [&](llama_context * ctx) {
        const size_t need = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        std::vector<uint8_t> buf(need);
        return llama_state_seq_get_data_ext(ctx, buf.data(), need, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) > 0;
    };
    const auto prompt_in = [&](llama_context * ctx, bool split_prompt, bool read_state) {
        if (!split_prompt) return decode(ctx, prompt, 0, n_p, 0, false);
        if (!decode(ctx, prompt, 0, split, 0, false)) return false;
        if (read_state && !state_read(ctx)) return false;
        return decode(ctx, prompt, split, n_p, (llama_pos) split, false);
    };

    // reference: one-token decodes of gen[0..K-1]; row r predicts gen[r+1]
    std::vector<float> seq(K * (size_t) n_vocab);
    {
        auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
        if (!ctx || !prompt_in(ctx.get(), true, true)) { LOG_ERR("%s: reference prompt failed\n", __func__); return false; }
        for (size_t r = 0; r < K; ++r) {
            if (!decode(ctx.get(), gen, r, r + 1, (llama_pos) (n_p + r), true)) {
                LOG_ERR("%s: reference decode %zu failed\n", __func__, r);
                return false;
            }
            const float * lg = llama_get_logits_ith(ctx.get(), 0);
            std::copy(lg, lg + n_vocab, seq.begin() + r * n_vocab);
        }
    }
    const auto top2 = [&](const float * x, int & a, float & m) {
        float t1 = -INFINITY, t2 = -INFINITY;
        a = 0;
        for (int v = 0; v < n_vocab; ++v) {
            if (x[v] > x[a]) a = v;
            if (x[v] > t1) { t2 = t1; t1 = x[v]; } else if (x[v] > t2) { t2 = x[v]; }
        }
        m = t1 - t2;
    };
    // the reference must reproduce the server's own ids, or this test is not replaying the server
    int ref_miss = 0;
    for (size_t r = 0; r < K; ++r) {
        int a; float m;
        top2(seq.data() + r * n_vocab, a, m);
        if (a != gen[r + 1] && m > MARGIN) {
            LOG("reference row %2zu: argmax %d, server generated %d (margin %g)\n", r, a, gen[r + 1], (double) m);
            ref_miss++;
        }
    }
    if (ref_miss > 0) {
        LOG_ERR("%s: error: INVALID -- %d one-token rows differ from the server's ids; the prompt or ids are not the server's\n",
                __func__, ref_miss);
        return false;
    }
    LOG("reference reproduces the server's %zu ids\n", K);

    bool ok = true;
    for (int v = 0; v < 8; ++v) {
        const bool split_prompt = v & 1, read_state = v & 2;
        const size_t rows = (v & 4) ? K : K - 1;
        auto ctx = llama_context_ptr{llama_init_from_model(model, params_ctx)};
        if (!ctx || !prompt_in(ctx.get(), split_prompt, read_state) || (read_state && !state_read(ctx.get())) ||
            !decode(ctx.get(), gen, 0, rows, (llama_pos) n_p, true)) {
            LOG_ERR("%s: arm %d decode failed\n", __func__, v);
            return false;
        }
        int flagged = 0, first = -1, shifted = 0;
        float worst = 0.0f;
        for (size_t r = 0; r < rows; ++r) {
            const float * x = seq.data() + r * n_vocab;
            const float * y = llama_get_logits_ith(ctx.get(), (int32_t) r);
            if (!y) { LOG_ERR("%s: arm %d no logits for row %zu\n", __func__, v, r); return false; }
            int ax, ay; float mx, my;
            top2(x, ax, mx);
            top2(y, ay, my);
            float d = 0.0f;
            for (int t = 0; t < n_vocab; ++t) d = std::max(d, std::fabs(x[t] - y[t]));
            worst = std::max(worst, d);
            if (ax != ay && mx > MARGIN) { flagged++; if (first < 0) first = (int) r; }
            // positive control: batch row r against one-token row r+1 must look wrong somewhere
            if (r + 1 < rows) {
                int as; float ms;
                top2(seq.data() + (r + 1) * n_vocab, as, ms);
                shifted += (as != ay && ms > MARGIN) ? 1 : 0;
            }
        }
        LOG("arm %d (prompt %s, state read %s, %2zu rows): worst |diff| %-10g argmax changed in %d row(s)%s%s; "
            "shifted control %d -> %s\n", v, split_prompt ? "split" : "whole", read_state ? "yes" : "no ", rows,
            (double) worst, flagged, first >= 0 ? ", first at row " : "", first >= 0 ? std::to_string(first).c_str() : "",
            shifted, shifted > 0 ? "DETECTED" : "BLIND");
        if (shifted == 0) {
            LOG_ERR("%s: error: BLIND -- arm %d did not detect an off-by-one row mapping\n", __func__, v);
            return false;
        }
        ok = ok && flagged == 0;
    }
    if (!ok) {
        LOG_ERR("%s: error: a server-shaped verify batch picks a different token than a one-token decode\n", __func__);
        return false;
    }
    LOG("\nPASS\n");
    return true;
}


// PRIOR-ART-NONE: tests/ was searched for "does not match expectation" and for a llama_log_set wrapper; no test
// counted or failed on the ~llama_context compute-buffer warning.
//
// ~llama_context logs a WARN when the live compute buffer is larger than the size the constructor reserved for it
// (src/llama-context.cpp, the backend_buf_exp_size check). That means the worst-case reserve was not a superset of
// a real decode: nemotron_h with n_seq_max=2 and the legacy qwen35 / qwen3next graphs did this until 2026-10-07.
// A log line nobody reads is not a guard, so count it and fail the run.
static ggml_log_callback g_prev_log_cb = nullptr;
static void *            g_prev_log_ud = nullptr;
static int               g_n_compute_buf_warn = 0;

static void count_compute_buf_warn_cb(ggml_log_level level, const char * text, void * user_data) {
    if (text != nullptr && std::strstr(text, "does not match expectation") != nullptr) {
        g_n_compute_buf_warn++;
    }
    if (g_prev_log_cb != nullptr) {
        g_prev_log_cb(level, text, g_prev_log_ud);
    }
    (void) user_data;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt = "";
    params.n_batch = 100;
    params.out_file = "dump_state.bin";
    params.sampling.seed = 1234;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (params.n_parallel == 1) {
        LOG_TRC("%s: n_parallel == 1, enabling unified kv cache\n", __func__);
        params.kv_unified = true;
    }

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    ggml_backend_load_all();

    auto llama_init = common_init_from_params(params, true);
    auto * model = llama_init->model();

    if (model == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    GGML_ASSERT(llama_init->context() == nullptr);

    // Tokenize prompt or generate random tokens
    llama_tokens tokens;
    if (params.prompt.empty()) {
        const int n_prompt = params.n_batch;

        // this path is useful for model files that do not have a tokenizer
        LOG_INF("%s: no prompt provided, generating %d (n_batch) random tokens\n", __func__, n_prompt);

        const auto * vocab = llama_model_get_vocab(model);
        const auto n_vocab = llama_vocab_n_tokens(vocab);

        std::mt19937 rng(params.sampling.seed);
        std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);
        for (int i = 0; i < n_prompt; i++) {
            tokens.push_back(dist(rng));
        }
    } else {
        LOG_INF("%s: tokenizing prompt '%s'\n", __func__, params.prompt.c_str());

        auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
        tokens = common_tokenize(ctx.get(), params.prompt, true);
    }

    LOG_INF("%s: the input prompt is %d tokens\n", __func__, (int)tokens.size());

    // Installed HERE, after common_params_parse() and common_init_from_params(): either may re-install the llama
    // log callback, and a callback set earlier would be replaced before any context exists.
    llama_log_get(&g_prev_log_cb, &g_prev_log_ud);
    llama_log_set(count_compute_buf_warn_cb, nullptr);

    // Test 1: baseline (saves state to disk)
    auto result_baseline = test_baseline(model, params, tokens);
    if (result_baseline.empty()) {
        return 1;
    }

    // Test 2: sequence removal isolation
    if (!test_seq_rm_isolated(model, params, tokens)) {
        return 1;
    }

    // Test 3: state load
    if (!test_state_load(model, params, tokens, result_baseline)) {
        return 1;
    }

    // Test 4: seq copy (host)
    if (!test_seq_cp_host(model, params, tokens, result_baseline)) {
        return 1;
    }

    // Test 5: seq copy (device)
    if (!test_seq_cp_device(model, params, tokens, result_baseline)) {
        return 1;
    }

    // Test 7: speculative rollback (before Test 6 so a Test 6 failure cannot hide it)
    if (!test_spec_rollback_replay(model, params, tokens)) {
        return 1;
    }

    // Test 8: verify batch vs one-token decodes (also before Test 6). Its detector needs rows where the model is
    // decided; random ids leave a small model near-flat (see its detector-power check), so when no prompt was given
    // and the model has a tokenizer, Test 8 reads a short natural-text passage instead.
    llama_tokens tokens_t8 = tokens;
    if (params.prompt.empty() && llama_vocab_type(llama_model_get_vocab(model)) != LLAMA_VOCAB_TYPE_NONE) {
        static const char * T8_TEXT =
            "Once upon a time, there was a little girl named Lily. She loved to play outside in the park with her "
            "friends. One day, she saw a big red ball under a tree. She ran to the ball and kicked it high into the "
            "sky. Her friend Tom laughed and clapped his hands. They played with the ball all day long, and when the "
            "sun went down, they went home happy.";
        tokens_t8 = common_tokenize(llama_model_get_vocab(model), T8_TEXT, true, false);
        LOG_INF("%s: Test 8 reads a %zu-token natural-text passage (no prompt given)\n", __func__, tokens_t8.size());
    }
    if (!test_verify_vs_sequential(model, params, tokens_t8)) {
        return 1;
    }

    // Test 9: llama-server's first verify step, replayed with its own ids (runs only when its env inputs are set)
    if (!test_server_verify_replay(model, params)) {
        return 1;
    }

    // Test 6 layer trace: diagnostic only (LLAMA_T6_TRACE=1), prints which op first differs, never fails the run
    test_mixed_seq_layer_trace(model, params, tokens);

    // Test 6: two sequences in one batch
    if (!test_mixed_seq_batch(model, params, tokens)) {
        return 1;
    }

    // Every context above has been destroyed by now, so every ~llama_context warning has been logged.
    if (g_n_compute_buf_warn != 0) {
        LOG_ERR("%s: FAIL: %d compute buffer size warning(s) at context teardown (reserve was not a worst case)\n",
                __func__, g_n_compute_buf_warn);
        return 1;
    }

    if (g_t6_inconclusive) {
        LOG("\nTests 1-5 passed; Test 6 INCONCLUSIVE on this model (leak control blind). Exit 77 = skipped.\n");
        return 77;
    }

    LOG("\nAll tests passed.\n");

    return 0;
}
