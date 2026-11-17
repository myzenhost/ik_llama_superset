// End-to-end control for the MoE expert-activation recorder (S69-KT-EXPERT-POOL-2026-09-28 unit
// 1): does the real decode graph's cb_eval wrapper (src/llama-context.cpp
// llama_moe_record_eval_callback) actually feed src/llama-moe-pool.h's llama_moe_recorder, and
// does the resulting per-layer histogram sum to exactly n_tokens * n_expert_used, as the task
// brief's control requires?
//
// tests/test-moe-pool-placement.cpp already exercises llama_moe_recorder / llama_moe_seed_layer /
// llama_moe_pool_layer as PURE HOST-SIDE STRUCTS with synthetic ids -- it proves the bookkeeping
// is correct in isolation, not that anything in the real graph calls it. This file is the other
// half: a real model, a real llama_decode(), and only the write_json() output as evidence -- if
// the cb_eval wrapper were never installed, or filtered on the wrong tensor name, or read the
// wrong buffer, this test fails where test-moe-pool-placement.cpp could not have caught it.
//
// SELF-CALIBRATING, NOT A HARDCODED n_expert_used. This file does not assume any particular
// architecture's expert-per-token count. It measures it: decode exactly ONE token in a throwaway
// context and read back the recorder's own per-layer sum for that single token -- that sum IS
// n_expert_used for that layer, by construction (llm_build_moe_ffn's ggml_top_k selects exactly
// n_expert_used distinct ids per token, so one token produces exactly n_expert_used activations).
// A second, independent context then decodes N>1 tokens and the test asserts every layer's sum
// equals N * (that layer's own measured k) -- catching a recorder that fires per-call instead of
// per-token, or double-counts, or drops layers, none of which a hardcoded expected constant would
// distinguish from "the model's true n_expert_used differs from what I assumed".
//
// Model: this task's own ledger (tmp/_agent_ledger_s69-kt-expert-pool-2026-09-28.md,
// entry 3) identifies ggml-org/stories15M_MOE-F16.gguf as the small on-disk MoE model for this
// task's tests. This file takes the path as argv[1] rather than hardcoding it, so it is not tied
// to one machine's cache layout.
//
// GATE-CLAIM-OK: this file declares no resource unavailable; "no model path given" is argv
// arity, checked below, not an assumption about disk state.
//
// Usage: test-moe-pool-recorder-e2e <model.gguf>
// Skips with exit 0 and a printed reason when no model path is given (same convention as
// test-logits-sanity.cpp), so it stays green in the CTest suite on a machine with no GGUF; a
// caller with the model (this task's harness) runs the built exe directly with the path.

#include "llama.h"
#include "common.h"
#include "llama-moe-pool.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
static std::string make_tmp_path(const char * tag) {
    char dir[MAX_PATH];
    GetTempPathA(MAX_PATH, dir);
    return std::string(dir) + "_kt2_0928_moe_e2e_" + tag + ".json";
}
#else
static std::string make_tmp_path(const char * tag) {
    return std::string("/tmp/_kt2_0928_moe_e2e_") + tag + ".json";
}
#endif

static int g_failures = 0;

using layer_counts = std::unordered_map<int32_t, std::unordered_map<int32_t, uint64_t>>;

// INDEPENDENT READER of the same routing decisions, chained behind the recorder via cparams.cb_eval
// (the recorder forwards every node to the previously installed callback). It reads a DIFFERENT
// buffer than the recorder: the topk VIEW's SOURCE, the contiguous [n_expert, n_tokens] argsort,
// in one piece, then keeps the first ne0 entries of each row. The recorder reads the view itself,
// strided, row by row. Both must agree id for id. A sum-only check cannot see a reader that returns
// the right NUMBER of ids but the wrong ones -- measured 2026-09-28: a flat read of the view
// returned whole argsort rows, same count, unselected experts included.
struct direct_reader {
    layer_counts counts;
};

// This fork's ggml_backend_sched_eval_callback returns an int need level (0 = not needed,
// 1 = synchronous materialization), like llama_moe_record_eval_callback, not upstream's bool.
static int direct_reader_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    static const char prefix[] = "ffn_moe_topk-";
    if (std::strncmp(t->name, prefix, sizeof(prefix) - 1) != 0) {
        return 0;
    }
    if (ask) {
        return 1;
    }
    auto * dr = static_cast<direct_reader *>(user_data);
    const ggml_tensor * src = t->view_src;
    // A warmup pass routes to all experts, so its view is as wide as its source; skip it, as the
    // recorder does, without needing the model's n_expert_used.
    if (src == nullptr || t->ne[0] >= src->ne[0] || t->view_offs != 0 || src->type != GGML_TYPE_I32) {
        return 1;
    }
    std::vector<int32_t> all((size_t) ggml_nelements(src));
    ggml_backend_tensor_get(src, all.data(), 0, all.size() * sizeof(int32_t));
    const int32_t il = (int32_t) std::strtol(t->name + sizeof(prefix) - 1, nullptr, 10);
    auto & row = dr->counts[il];
    for (int64_t r = 0; r < t->ne[1]; ++r) {
        for (int64_t j = 0; j < t->ne[0]; ++j) {
            row[all[(size_t) (r * src->ne[0] + j)]] += 1;
        }
    }
    return 1;
}

// Sum every expert's count for one layer from the recorder's own (re-parsed) JSON output --
// deliberately going through llama_moe_recorder::read_json rather than re-implementing a parser,
// so this test also stands as a round-trip check of write_json/read_json against a REAL file the
// destructor wrote, not just the in-memory round trip test-moe-pool-placement.cpp already covers.
static uint64_t layer_sum(const std::unordered_map<int32_t, std::unordered_map<int32_t, uint64_t>> & counts,
                           int32_t layer) {
    auto it = counts.find(layer);
    if (it == counts.end()) return 0;
    uint64_t total = 0;
    for (auto & kv : it->second) total += kv.second;
    return total;
}

// Runs one decode of n_tokens fresh tokens (positions 0..n_tokens-1, no prior KV) on a NEW
// context over the given model, with the recorder writing to json_path, then frees the context
// (which is where llama_context::~llama_context flushes write_json -- see llama-context.cpp).
static bool run_and_record(llama_model * model, const std::vector<llama_token> & tokens,
                            const std::string & json_path, direct_reader * dr = nullptr) {
    llama_context_params cparams = llama_context_default_params();
    if (dr) {
        cparams.cb_eval           = direct_reader_cb;
        cparams.cb_eval_user_data = dr;
    }
    cparams.n_ctx        = 512;
    cparams.n_batch       = 512;
    cparams.no_perf       = true;
    cparams.moe_stats_path = json_path.c_str();
    cparams.moe_stats_mode = "stat";

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        printf("FAIL -- could not create context for %s\n", json_path.c_str());
        return false;
    }

    // Logits requested for EVERY token: the last layer is pruned to the output rows, so with
    // llama_batch_get_one (logits for the last token only) the last MoE layer sees 1 token, not N --
    // measured 2026-09-28: layer 5 recorded 1*k while layers 0-4 recorded N*k. All-outputs keeps
    // every layer at N tokens, so one expectation (N*k) holds for all of them.
    const int32_t n = (int32_t) tokens.size();
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int32_t i = 0; i < n; ++i) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
    }
    batch.n_tokens = n;
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    llama_free(ctx); // <-- flush point: llama_moe_recorder::write_json runs here

    if (rc != 0) {
        printf("FAIL -- llama_decode returned %d for a %zu-token batch\n", rc, tokens.size());
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("SKIP -- no model path given. Usage: %s <model.gguf>\n", argv[0]);
        printf("(no exercise of the real recorder wiring on this machine; "
               "tests/test-moe-pool-placement.cpp still covers the host-side structs.)\n");
        return 0;
    }

    const std::string model_path = argv[1];
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU-only per this unit's scope (S69-KT-EXPERT-POOL stage-2 brief)

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        printf("FAIL -- could not load %s\n", model_path.c_str());
        llama_backend_free();
        return 1;
    }

    llama_context * probe_ctx = llama_init_from_model(model, llama_context_default_params());
    if (!probe_ctx) {
        printf("FAIL -- could not create a probe context for tokenization\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    std::vector<llama_token> prompt_tokens = common_tokenize(probe_ctx, "The capital of France is", true, true);
    llama_free(probe_ctx);
    if (prompt_tokens.size() < 4) {
        printf("FAIL -- prompt tokenized to only %zu tokens, need >= 4 for the multi-token run\n",
               prompt_tokens.size());
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // ---- Warmup run: a fresh context's first eval of a lone BOS is a WARMUP pass -------------
    // (llama-build-context.cpp: n_eval == 0 && one token == BOS), which routes to ALL experts on
    // purpose. The recorder must record NOTHING for it, or every load-time warmup credits every
    // expert once per layer. Measured 2026-09-28: before the recorder skipped warmup, this run
    // recorded 4 ids per layer on stories15M_MOE (4 experts, 2 used).
    const std::string json0 = make_tmp_path("warmup");
    const llama_token bos = llama_vocab_bos(llama_model_get_vocab(model));
    if (prompt_tokens[0] != bos) {
        printf("FAIL -- prompt_tokens[0] is not BOS (%d vs %d); the warmup check needs a BOS-first prompt\n",
               prompt_tokens[0], bos);
        g_failures++;
    }
    if (!run_and_record(model, { bos }, json0)) {
        g_failures++;
    }
    layer_counts counts0;
    if (!llama_moe_recorder::read_json(json0, counts0)) {
        printf("FAIL -- could not read back %s\n", json0.c_str());
        g_failures++;
    }
    uint64_t warm_total = 0;
    for (auto & kv : counts0) warm_total += layer_sum(counts0, kv.first);
    printf("  %-4s warmup (lone BOS) recorded %llu ids (must be 0)\n", warm_total == 0 ? "ok" : "FAIL",
           (unsigned long long) warm_total);
    if (warm_total != 0) g_failures++;

    // ---- Calibration run: exactly 1 NON-BOS token -> layer sum IS n_expert_used ----------------
    // Deliberately NOT prompt_tokens[0]: that is BOS, and a lone BOS first eval is the warmup case
    // above (all experts), which is how this calibration once measured k=4 for a k=2 model.
    const std::string json1 = make_tmp_path("calib1");
    std::vector<llama_token> one_token = { prompt_tokens[1] };
    if (!run_and_record(model, one_token, json1)) {
        g_failures++;
    }

    std::unordered_map<int32_t, std::unordered_map<int32_t, uint64_t>> counts1;
    if (!llama_moe_recorder::read_json(json1, counts1)) {
        printf("FAIL -- could not read back %s (recorder never wrote it, or wrote nothing "
               "parseable -- this is exactly the failure mode this test exists to catch: the "
               "wiring silently not firing)\n", json1.c_str());
        g_failures++;
    }
    if (counts1.empty()) {
        printf("FAIL -- calibration recorder JSON has ZERO layers: either this gguf's own "
               "metadata (loaded above by llama_model_load_from_file) carries no MoE tensors for "
               "the recorder to observe, or ffn_moe_topk-<il> never reached the callback. Pass a "
               "known MoE gguf (this task's own ledger entry 3 names ggml-org/stories15M_MOE) as "
               "argv[1] to exercise this test meaningfully.\n");
        g_failures++;
    }

    // ---- Measurement run: N > 1 tokens, same model, fresh context, fresh JSON -----------------
    const int n_tokens = (int) prompt_tokens.size(); // >= 4, checked above
    const std::string json2 = make_tmp_path("measureN");
    direct_reader dr;
    if (!run_and_record(model, prompt_tokens, json2, &dr)) {
        g_failures++;
    }

    std::unordered_map<int32_t, std::unordered_map<int32_t, uint64_t>> countsN;
    if (!llama_moe_recorder::read_json(json2, countsN)) {
        printf("FAIL -- could not read back %s\n", json2.c_str());
        g_failures++;
    }

    printf("model        : %s\n", model_path.c_str());
    printf("calib layers : %zu   measure(N=%d) layers : %zu\n",
           counts1.size(), n_tokens, countsN.size());

    if (counts1.size() != countsN.size()) {
        printf("FAIL -- layer count differs between the 1-token and %d-token runs (%zu vs %zu) -- "
               "the recorder is not seeing the same set of MoE layers on every call.\n",
               n_tokens, counts1.size(), countsN.size());
        g_failures++;
    }

    int n_layers_checked = 0;
    for (auto & kv : counts1) {
        const int32_t il = kv.first;
        const uint64_t k = layer_sum(counts1, il);           // measured n_expert_used, layer il
        const uint64_t sum_n = layer_sum(countsN, il);        // observed sum after N tokens
        const uint64_t expected = k * (uint64_t) n_tokens;

        const bool ok = k > 0 && sum_n == expected;
        printf("  %-4s layer %-3d k(measured n_expert_used)=%-4llu  N*k=%-6llu  observed=%-6llu\n",
               ok ? "ok" : "FAIL", il, (unsigned long long) k,
               (unsigned long long) expected, (unsigned long long) sum_n);
        if (!ok) g_failures++;
        n_layers_checked++;
    }

    if (n_layers_checked == 0) {
        printf("FAIL -- no layers were checked at all (calibration run produced no layers).\n");
        g_failures++;
    }

    // ---- IDs, not just sums: recorder vs the independent reader, expert by expert ---------------
    const bool ids_match = !dr.counts.empty() && dr.counts == countsN;
    printf("  %-4s recorder ids == independent reader ids (%zu vs %zu layers)\n", ids_match ? "ok" : "FAIL",
           countsN.size(), dr.counts.size());
    if (!ids_match) {
        g_failures++;
        for (auto & kv : dr.counts) {
            for (auto & e : kv.second) {
                const uint64_t got = countsN.count(kv.first) && countsN.at(kv.first).count(e.first)
                                   ? countsN.at(kv.first).at(e.first) : 0;
                if (got != e.second) {
                    printf("       layer %d expert %d: recorder %llu, direct %llu\n", kv.first, e.first,
                           (unsigned long long) got, (unsigned long long) e.second);
                }
            }
        }
    }

    printf("\n%s -- %d failure(s), %d layer(s) checked\n",
           g_failures == 0 ? "PASS" : "FAIL", g_failures, n_layers_checked);
    printf("SCOPE: this proves the cb_eval wrapper installed in the llama_context constructor "
           "(llama-context.cpp, S69-KT-EXPERT-POOL unit 1) reaches the real ffn_moe_topk tensor "
           "and llama_moe_recorder::record()/write_json() round-trip correctly for a real decode. "
           "It does NOT touch the resident-pool graph splice (unit 2, not wired by this file).\n");

    // Print both raw recorder files: their "steps" field counts record() CALLS, which separates
    // "the callback fired twice" from "one call carried twice the ids" when a count is off.
    for (const std::string * p : { &json0, &json1, &json2 }) {
        if (FILE * f = fopen(p->c_str(), "rb")) {
            char buf[4096];
            const size_t n = fread(buf, 1, sizeof(buf) - 1, f);
            buf[n] = 0;
            fclose(f);
            printf("raw %s: %s", p->c_str(), buf);
        }
    }

    std::remove(json0.c_str());
    std::remove(json1.c_str());
    std::remove(json2.c_str());

    llama_model_free(model);
    llama_backend_free();
    return g_failures == 0 ? 0 : 1;
}
