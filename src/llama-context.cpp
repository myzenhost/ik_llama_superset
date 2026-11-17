#include "llama-context.h"
#include "offload_attention_dense.h" // --offload-attn-dense-layers N | -N | list (S69-ENGREV-HIPATTENTION-FLAG-SURFACE)
#include "llama-spec-features.h" // llama_set_draft_input_hidden_state_copy
#include "llama-moe-cache.h"   // ik #2644: complete type for llama_context::moe_cache.reset()

#include "ggml.h"
#include "llama-arch.h"
#include "llama-yarn.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-kv-layer-types.h"
#include "llama-turbo-boundary-v.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "llama-ext.h"
#include "llama-sampler.h"
#include "llama-build-context.h"   // llm_build_context::llama_build_graph, for the legacy reserve
#include "llama-kv-kivi-sim.h"
#include "llama-dsv4.h"            // llama_prepare_dsv4_graph_inputs, mirrored from llama.cpp:7789
#include "llama.h"
// TriAttention, added with the port of the three fork features that llama.cpp's inline
// llama_init_from_model carried and this file did not. triattention-runtime.h was already
// reachable (it declares tria_runtime_init and g_tria_rt); triattention.h was NOT, and it is where
// tria_load and tria_free are declared. MEASURED before editing: of the names this port needs,
// every llama_context_params and llama_cparams field was already visible, and only these two
// functions plus llama_repack_up_gate_exps were not.
#include "triattention.h"

#include <cinttypes>
#include <cassert>   // slice 11: assert() is used at 7 sites here and was never included
#include <cmath>
#include <cstdlib>   // S69-KT-EXPERT-POOL unit 1: std::strtol in llama_moe_record_eval_callback
#include <cstring>
#include <fstream>   // CATS threshold-file loading (S69-CATS-STAGE1-2026-09-28)
#include <algorithm>
#include <mutex>     // slot-file model identity fingerprint cache
#include <limits>
#include <set>       // llama_host_kept_token_counts
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// SLICE 29, 2026-09-09. These guarded backend headers are mirrored from src/llama.cpp:46-62. This
// file previously included NONE of them, and the unified backend-init block below needs them in any
// build that actually selects a GPU backend.
//
// THE FAILURE MODE IS WHY THIS COMMENT EXISTS: the syntax check used during this migration defines
// no GGML_USE_* macro, so every arm below is preprocessed away and the file compiles clean at 0 with
// these includes absent. It would break the moment a real CUDA build defines GGML_USE_CUDA --
// compiles here, fails there. The include arrangement is not verified by anything in this session;
// it is copied from the one src/llama.cpp compiles with today.
#ifdef GGML_USE_RPC
#  include "ggml-rpc.h"
#endif

#ifdef GGML_USE_CUDA
#  include "ggml-cuda.h"
#elif defined(GGML_USE_VULKAN)
#  include "ggml-vulkan.h"
#elif defined(GGML_USE_SYCL)
#  include "ggml-sycl.h"
#elif defined(GGML_USE_CANN)
#   include "ggml-cann.h"
#elif defined(GGML_USE_OPENVINO)
#   include "ggml-openvino.h"
#endif

#ifdef GGML_USE_METAL
#  include "ggml-metal.h"
#endif

// Defined in src/llama.cpp: fork-specific pre-construction checks plus the K-cache Hadamard fold.
bool llama_fork_prepare_context_params(llama_model & model, llama_context_params & params, std::string & err);
#include <unordered_map>

//
// llama_context
//

// check that each folded weight in the graph gets its Hadamard transform, and each latent lookup gets the inverse
// without this check, an arch that skips the transform helpers loads and computes wrong results
static void llama_verify_hadamard_graph(
        ggml_cgraph * gf,
        const llama_hadamard_rotations & rotations,
        const llama_hadamard_rotations & inverses,
        const llama_moe_cache * moe_cache) {
    // the MoE cache gives the matmul a copy of the expert weights, so the copy also needs the transform of its source
    llama_hadamard_rotations forward = rotations;
    for (const auto & [w, t] : rotations) {
        if (const ggml_tensor * cached = moe_cache ? moe_cache->get_experts(w) : nullptr) {
            forward.emplace(cached, t);
        }
    }

    auto unwrap = [](const ggml_tensor * t) {
        while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW)) {
            t = t->src[0];
        }
        return t;
    };

    std::map<const ggml_tensor *, bool> lookups; // get_rows results of latent tables

    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_tensor * node = ggml_graph_node(gf, i);

        if (node->op == GGML_OP_GET_ROWS && inverses.count(node->src[0])) {
            lookups.emplace(node, false);
            continue;
        }

        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }

        if (node->op == GGML_OP_MUL_MAT && ((const int32_t *) node->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD) {
            const auto lk = lookups.find(unwrap(node->src[1]));
            if (lk != lookups.end()) {
                lk->second = true;
            }
            continue;
        }

        const auto it = forward.find(node->src[0]);
        if (it == forward.end()) {
            if (inverses.count(node->src[0])) {
                throw std::runtime_error(format(
                    "Hadamard-latent table '%s' is used as a head without a forward transform", node->src[0]->name));
            }
            continue;
        }
        const ggml_tensor * src = unwrap(node->src[1]);
        const bool transformed = src && src->op == GGML_OP_MUL_MAT &&
            ((const int32_t *) src->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD &&
            src->src[0] == it->second.rot;
        if (!transformed) {
            throw std::runtime_error(format(
                "Hadamard-folded weight '%s' is consumed without its activation transform; "
                "this graph's matmul path does not support prism.hadamard folding",
                node->src[0]->name));
        }
    }

    for (const auto & [node, ok] : lookups) {
        if (!ok) {
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                const ggml_tensor * n2 = ggml_graph_node(gf, i);
                for (int s = 0; s < GGML_MAX_SRC && n2->src[s]; ++s) {
                    if (unwrap(n2->src[s]) == node) {
                        LLAMA_LOG_WARN("%s: latent lookup '%s' consumed by op=%s name='%s' src%d hint=%d\n",
                                __func__, node->name, ggml_op_name(n2->op), n2->name, s,
                                ((const int32_t *) n2->op_params)[1]);
                    }
                }
            }
            throw std::runtime_error(format(
                "Hadamard-latent table '%s' is read without the inverse transform",
                node->src[0]->name));
        }
    }
}

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

struct llm_fused_op_probe {
    llm_fused_op op;
    const char * name;
    uint32_t n_tokens_per_seq;
};

static const llm_fused_op_probe llm_fused_op_flash_attn_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN,
    /*.name             =*/ "Flash Attention",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_flash_attn_kv_rows_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN_KV_ROWS,
    /*.name             =*/ "Flash Attention over K/V rows",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ar_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_AR,
    /*.name             =*/ "fused Gated Delta Net (autoregressive)",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ch_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_CH,
    /*.name             =*/ "fused Gated Delta Net (chunked)",
    /*.n_tokens_per_seq =*/ 16,
};

static const llm_fused_op_probe llm_fused_op_lid_probe = {
    /*.op               =*/ LLM_FUSED_OP_LIGHTNING_INDEXER,
    /*.name             =*/ "Lightning Indexer",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_pre_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_PRE,
    /*.name             =*/ "fused DeepSeek V4 HC pre",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_comb_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_COMB,
    /*.name             =*/ "fused DeepSeek V4 HC comb",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_post_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_POST,
    /*.name             =*/ "fused DeepSeek V4 HC post",
    /*.n_tokens_per_seq =*/ 1,
};

// ── CATS activation sparsity (S69-CATS-STAGE1-2026-09-28) ──────────────────────────────────
// Minimal, dependency-free reader for the calibration tool's output format (see
// tools/cats-calibrate/cats-calibrate.cpp for the writer and the format comment). This file
// is produced by our own calibration tool, not attacker-controlled input, so a small
// hand-rolled scan for the "thresholds" array is sufficient and avoids pulling
// vendor/nlohmann into this translation unit for one field.
//
// Expected shape (whitespace-insensitive):
//   { "arch": "...", "n_layer": <int>, "sparsity_target": <float>,
//     "thresholds": [<float>, <float>, ...] }
static std::vector<float> cats_load_thresholds(const std::string & path, uint32_t n_layer_expected) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error(format("--cats-thresholds: could not open '%s'", path.c_str()));
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();

    size_t key = text.find("\"thresholds\"");
    if (key == std::string::npos) {
        throw std::runtime_error(format("--cats-thresholds: '%s' has no \"thresholds\" key", path.c_str()));
    }
    size_t lb = text.find('[', key);
    size_t rb = text.find(']', lb == std::string::npos ? key : lb);
    if (lb == std::string::npos || rb == std::string::npos || rb < lb) {
        throw std::runtime_error(format("--cats-thresholds: '%s' has a malformed \"thresholds\" array", path.c_str()));
    }

    std::vector<float> out;
    std::string body = text.substr(lb + 1, rb - lb - 1);
    std::stringstream body_ss(body);
    std::string tok;
    while (std::getline(body_ss, tok, ',')) {
        // strip whitespace/newlines
        size_t a = tok.find_first_not_of(" \t\r\n");
        size_t b = tok.find_last_not_of(" \t\r\n");
        if (a == std::string::npos) continue;
        out.push_back(std::stof(tok.substr(a, b - a + 1)));
    }

    if (out.size() != n_layer_expected) {
        throw std::runtime_error(format(
            "--cats-thresholds: '%s' has %zu thresholds but the model has %u layers "
            "(calibration file does not match this model)",
            path.c_str(), out.size(), n_layer_expected));
    }
    return out;
}

// S69-KT-EXPERT-POOL-2026-09-28 unit 1: cb_eval wrapper that feeds the MoE expert-activation
// recorder (src/llama-moe-pool.{h,cpp}) from the SAME "ffn_moe_topk" tensor the real decode graph
// already tags -- src/llama-build-context.cpp llm_build_moe_ffn(): `cb(selected_experts,
// "ffn_moe_topk", il)` -- and which the main build_graph's own cb lambda (llama-build-context.cpp,
// ~line 2916, same shape as the dflash-kv-cache one at ~line 2871 read for this port) writes to
// the tensor's ggml name as "ffn_moe_topk-<il>". Installed once at context construction (below,
// only when cparams.moe_stats_path is non-empty) and never touched again for the life of the
// context, unlike src/llama-spec-features-dflash.cpp's llama_dflash_capture_eval_callback which is
// installed/removed per capture window -- that file's chain-of-responsibility shape
// (moe.prev_cb_eval / moe.prev_cb_eval_user_data, mirroring dflash_runtime::capture_state's
// prev_cb_eval fields) is reused here so a caller-supplied params.cb_eval keeps firing.
//
// Return-value contract is THIS FORK'S, not upstream's: ggml/include/ggml-backend.h:338-341 --
// ggml_backend_sched_eval_callback returns int, not bool (three existing call sites cast
// true/false to 1/0). ggml/src/ggml-backend.cpp ggml_backend_sched_eval(): on the ask=true call,
// a nonzero return means "materialize this node now" and need==1 SPECIFICALLY (not just nonzero)
// makes the scheduler call ggml_backend_synchronize() before the ask=false follow-up -- need==2
// (dflash's choice) skips that sync and relies on an async copy consumed later. This callback
// asks for eager need==1 and reads the tensor with a plain synchronous ggml_backend_tensor_get()
// in the ask=false call, rather than dflash's async-copy-plus-deferred-consumption machinery,
// because unlike dflash's whole-batch multi-row capture this only ever needs one small I32 vector
// available immediately in the same call where it is recorded -- the simpler synchronous path has
// no correctness gap to root-cause and the added per-MoE-layer sync is a CPU no-op on the CPU
// backend (unit 1's build target); a GPU cost, if any, is unmeasured and deferred to whoever wires
// unit 3 (see ledger). On the ask=false path this must NOT return 0/false for our own tensor
// unless we intend the scheduler to stop computing the rest of the graph -- see the
// `if (need && !callback(t, false, ...)) break;` loop in ggml_backend_sched_eval -- so we always
// return nonzero for our own tensor there and pass through whatever the previous callback returns
// only when it is also asked about the same tensor.
static int llama_moe_record_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * ctx = static_cast<llama_context *>(user_data);
    if (ctx == nullptr) {
        return 0;
    }

    // Match "ffn_moe_topk-<digits>" exactly (see comment above for where that name is written);
    // anything else is not ours and goes straight to whatever callback was previously installed.
    static const char prefix[] = "ffn_moe_topk-";
    const size_t prefix_len = sizeof(prefix) - 1;
    const bool is_ours = ctx->moe.recorder != nullptr &&
            std::strncmp(t->name, prefix, prefix_len) == 0;

    if (!is_ours) {
        if (ctx->moe.prev_cb_eval) {
            return ctx->moe.prev_cb_eval(t, ask, ctx->moe.prev_cb_eval_user_data);
        }
        return 0;
    }

    if (ask) {
        // Also ask the chained callback so a caller-supplied cb_eval (imatrix, perplexity, etc.)
        // still gets to request this same node if it independently cares about it; need==1 (ours)
        // already forces the strongest (synchronous) materialization, so the max is just "1".
        if (ctx->moe.prev_cb_eval) {
            ctx->moe.prev_cb_eval(t, true, ctx->moe.prev_cb_eval_user_data);
        }
        return 1;
    }

    // ask == false: the node is computed and, because we returned need==1 above, the scheduler
    // already called ggml_backend_synchronize() on its backend before this call -- a plain
    // synchronous tensor_get is safe here.
    const int64_t n_ids = ggml_nelements(t);
    // Skip WARMUP passes: llama-build-context.cpp flags the context's first eval of a lone BOS as warmup and
    // then routes every token to ALL experts (n_expert_used = warmup ? n_expert : n_expert_used, :83), so the
    // topk view is n_expert wide. Recording it credits every expert once per layer on every load-time warmup
    // (common's warmup decodes exactly that BOS). MEASURED 2026-09-28 on stories15M_MOE (4 experts, 2 used):
    // a BOS-only pass recorded 4 ids per layer. Only warmup widens the view, so its width is the test.
    if (t->ne[0] != (int64_t) ctx->model.hparams.n_expert_used) {
        if (ctx->moe.prev_cb_eval) {
            return ctx->moe.prev_cb_eval(t, false, ctx->moe.prev_cb_eval_user_data);
        }
        return 1;
    }
    if (n_ids > 0 && t->type == GGML_TYPE_I32) {
        // ggml_backend_tensor_get reads through t->buffer and takes no backend argument
        // (ggml-backend.h:93); the scheduler lookup only confirms the node was placed and allocated.
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(ctx->sched, t);
        if (backend != nullptr && t->buffer != nullptr) {
            // ffn_moe_topk is NOT contiguous: ggml_argsort_top_k (the fork's former ggml_top_k) returns a k-wide VIEW of the
            // full n_expert argsort, so row i starts at i*nb[1] with nb[1] = n_expert*4, not k*4
            // (MEASURED 2026-09-28: ne=[2,6] nb=[4,16] on stories15M_MOE). A flat read of n_ids ints returns
            // the first n_ids ints of the argsort buffer -- whole rows, unselected experts included: the
            // same COUNT with the wrong IDS, which a sum-only check cannot see. Read row by row instead;
            // test-moe-pool-recorder-e2e cross-checks the ids against an independent reader.
            if (getenv("LLAMA_MOE_REC_DIAG")) {
                fprintf(stderr, "moe-rec-diag %s op=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu] view_src=%s\n", t->name,
                        ggml_op_name(t->op), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
                        (long long) t->ne[3], t->nb[0], t->nb[1], t->view_src ? t->view_src->name : "-");
            }
            std::vector<int32_t> ids((size_t) n_ids);
            const size_t row_bytes = (size_t) t->ne[0] * sizeof(int32_t);
            size_t pos = 0;
            for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
                    for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                        const size_t off = (size_t) (i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3]);
                        ggml_backend_tensor_get(t, ids.data() + pos, off, row_bytes);
                        pos += (size_t) t->ne[0];
                    }
                }
            }
            const int32_t il = (int32_t) std::strtol(t->name + prefix_len, nullptr, 10);
            ctx->moe.recorder->record(il, ids.data(), n_ids);
        }
    }

    if (ctx->moe.prev_cb_eval) {
        return ctx->moe.prev_cb_eval(t, false, ctx->moe.prev_cb_eval_user_data);
    }
    return 1;
}

// ── TEAL activation sparsity (S69-TEAL-STAGE1-2026-09-28) ──────────────────────────────────
// Same hand-rolled-scan approach as cats_load_thresholds above, generalized to pull a named
// array key instead of always "thresholds", because TEAL's file carries three families (see
// tools/cats-calibrate/cats-calibrate.cpp --teal writer and the format comment there):
//   { "arch": "...", "n_layer": <int>, "sparsity_target": <float>,
//     "thresholds_attn_in": [...], "thresholds_ffn_in": [...], "thresholds_ffn_mid": [...] }
static std::vector<float> teal_load_array(const std::string & text, const std::string & key,
                                           const std::string & path, uint32_t n_layer_expected) {
    std::string needle = "\"" + key + "\"";
    size_t kpos = text.find(needle);
    if (kpos == std::string::npos) {
        throw std::runtime_error(format("--teal-thresholds: '%s' has no \"%s\" key", path.c_str(), key.c_str()));
    }
    size_t lb = text.find('[', kpos);
    size_t rb = text.find(']', lb == std::string::npos ? kpos : lb);
    if (lb == std::string::npos || rb == std::string::npos || rb < lb) {
        throw std::runtime_error(format("--teal-thresholds: '%s' has a malformed \"%s\" array", path.c_str(), key.c_str()));
    }
    std::vector<float> out;
    std::string body = text.substr(lb + 1, rb - lb - 1);
    std::stringstream body_ss(body);
    std::string tok;
    while (std::getline(body_ss, tok, ',')) {
        size_t a = tok.find_first_not_of(" \t\r\n");
        size_t b = tok.find_last_not_of(" \t\r\n");
        if (a == std::string::npos) continue;
        out.push_back(std::stof(tok.substr(a, b - a + 1)));
    }
    if (out.size() != n_layer_expected) {
        throw std::runtime_error(format(
            "--teal-thresholds: '%s' key \"%s\" has %zu thresholds but the model has %u layers "
            "(calibration file does not match this model)",
            path.c_str(), key.c_str(), out.size(), n_layer_expected));
    }
    return out;
}

struct teal_thresholds_triple {
    std::vector<float> attn_in;
    std::vector<float> ffn_in;
    std::vector<float> ffn_mid;
};

static teal_thresholds_triple teal_load_thresholds(const std::string & path, uint32_t n_layer_expected) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error(format("--teal-thresholds: could not open '%s'", path.c_str()));
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();

    teal_thresholds_triple t;
    t.attn_in = teal_load_array(text, "thresholds_attn_in", path, n_layer_expected);
    t.ffn_in  = teal_load_array(text, "thresholds_ffn_in",  path, n_layer_expected);
    t.ffn_mid = teal_load_array(text, "thresholds_ffn_mid", path, n_layer_expected);
    return t;
}

// Token counts at which the host keeps the most work. A weight-carrying op whose weight sits in a host buffer runs on
// the host UNLESS a device's offload_op says the batch is big enough (ggml-backend.cpp, the "1.off" cause): CUDA asks
// ne[1] >= offload_batch_size (32) for a dense op and batch*active >= min*total experts for MUL_MAT_ID (ggml-cuda.cu,
// ggml_backend_cuda_device_offload_op). The worst-case reserve graph has n_tokens_rsv tokens, so every such op is
// offloaded in it and the host buffer is sized as if the host ran none of them; a real batch just under a threshold
// runs them on the host with all their intermediates (and its logits) in the host compute buffer.
// MEASURED 2026-10-18: stories15M -ngl 0 with a device present, Test 8's 16-output batch needed 1.9707 MiB of host
// compute buffer against a 1.3308 MiB reserve, and ~llama_context warned (row
// S69-FORK-NGL0-OP-OFFLOAD-CPU-COMPUTE-BUFFER-OUTGROWS-RESERVE-2026-10-18-01).
// For each such op of the reserved graph this asks the device directly, on a copy of the node whose token-sized dims
// are rescaled, and binary-searches the largest token count it still leaves on the host. Each distinct answer below
// n_tokens_rsv is a shape the reserve has to cover: between two thresholds the host's op set is fixed and its buffers
// grow with the token count, so the largest count of each segment is that segment's worst case. A dim that merely
// equals n_tokens_rsv by coincidence is rescaled too; that can only add a reserve, and the buffers only grow.
// Call it on the graph AS BUILT, before any scheduler reserve or split of it: the split pass replaces a device op's
// host-weight src with the device's copy, which has no buffer during a reserve (measured 2026-10-18, see the callers).
static std::vector<int> llama_host_kept_token_counts(ggml_cgraph * gf, int n_tokens_rsv,
                                                     const std::vector<ggml_backend_t> & devices) {
    std::set<int> counts;
    if (gf == nullptr || n_tokens_rsv <= 1 || devices.empty()) {
        LLAMA_LOG_INFO("%s: host-kept probe skipped (graph %s, %d reserve tokens, %zu offload device(s))\n", __func__,
                       gf == nullptr ? "null" : "present", n_tokens_rsv, devices.size());
        return {};
    }
    // Per-filter counts, logged once: a probe that finds nothing must say which filter dropped every op
    // (MEASURED 2026-10-18, b1018d: -ngl 0 with a device still warned 3 times and this probe reserved nothing,
    // with no line telling why).
    int n_src_buf = 0, n_src_weights = 0, n_host_weight = 0, n_supported = 0, n_off_rsv = 0, n_off_1 = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        bool host_weight = false;
        for (int s = 0; s < GGML_MAX_SRC && !host_weight; ++s) {
            const ggml_tensor * src = node->src[s];
            if (src == nullptr || src->buffer == nullptr) {
                continue;
            }
            ++n_src_buf;
            if (ggml_backend_buffer_get_usage(src->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                continue;
            }
            ++n_src_weights;
            host_weight = ggml_backend_buffer_is_host(src->buffer);
        }
        if (!host_weight) {
            continue;
        }
        ++n_host_weight;
        for (ggml_backend_t dev : devices) {
            if (!ggml_backend_supports_op(dev, node)) {
                continue;
            }
            ++n_supported;
            auto offloaded_at = [&](int n) {
                ggml_tensor probe = *node;
                for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                    if (node->ne[d] == n_tokens_rsv) {
                        probe.ne[d] = n;
                    }
                }
                return ggml_backend_offload_op(dev, &probe);
            };
            if (!offloaded_at(n_tokens_rsv)) {
                continue; // never offloaded: the reserve already holds it on the host
            }
            ++n_off_rsv;
            if (offloaded_at(1)) {
                ++n_off_1;
                continue; // always offloaded
            }
            int lo = 1, hi = n_tokens_rsv; // invariant: lo stays on the host, hi is offloaded
            while (hi - lo > 1) {
                const int mid = lo + (hi - lo) / 2;
                (offloaded_at(mid) ? hi : lo) = mid;
            }
            counts.insert(lo);
        }
    }
    LLAMA_LOG_INFO("%s: host-kept probe: %d nodes, %d sources with a buffer, %d of them weights, %d ops on a host weight, "
                   "%d device-supported, %d offloaded at %d tokens, %d also at 1 -> %zu shape(s) to reserve\n", __func__,
                   ggml_graph_n_nodes(gf), n_src_buf, n_src_weights, n_host_weight, n_supported, n_off_rsv,
                   n_tokens_rsv, n_off_1, counts.size());
    return std::vector<int>(counts.begin(), counts.end());
}

static const llm_fused_op_probe llm_fused_op_top_k_mask_probe = {
    /*.op               =*/ LLM_FUSED_OP_TOP_K_MASK,
    /*.name             =*/ "fused top-k mask",
    /*.n_tokens_per_seq =*/ 1,
};

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    // SLICE 22, 2026-09-09. Two changes here, for two unrelated reasons.
    //
    // REMOVED  cvec(std::make_unique<llama_adapter_cvec>())
    //   Upstream holds cvec as std::unique_ptr<llama_adapter_cvec>. This fork holds it BY VALUE as
    //   `struct llama_control_vector cvec;` (llama-context.h:739) -- a different type with a
    //   different ownership model, and no conversion from a unique_ptr to it exists. The fork's
    //   type has a user-declared destructor but NO user-declared constructor, so its implicit
    //   default constructor survives and default construction is correct; the fork's own
    //   constructor (llama.cpp:920) likewise never mentions cvec.
    //
    // ADDED    sampling(llama_n_vocab(&model))
    //   llama_sampling declares llama_sampling(int32_t n_vocab) (llama-sampling.h:9), which
    //   SUPPRESSES the implicit default constructor, so the member cannot be left out of the list.
    //   This expression is not invented -- it is copied from this fork's own constructor at
    //   src/llama.cpp:921, which initialises the same member the same way.
    //
    // Order below follows DECLARATION order (model:464, loras:563, balloc:564, sampling:726), which
    // is the order they are actually initialised in regardless of what is written here.
    loras(std::make_unique<llama_adapter_loras>()),
    // MTP uses the embd input for the hidden state
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd(),
                llm_arch_supports_mixed_batch(model.arch) && params.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT)),
    sampling(llama_n_vocab(&model)) {

    // THE FORK'S LEGACY MEMBERS. Defined once in src/llama.cpp and, until now, run only by the
    // one-argument constructor there -- which nothing calls. This constructor is the live one, and
    // without this the graph builders read an EMPTY cache_copies and abort at
    // src/llama-build-context.cpp:937, while llama_all_contexts() stayed empty even though
    // ~llama_context erases from it.
    init_fork_members();
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }
    // The per-token recurrent snapshots (rs_idx) live in llama_memory_recurrent. A legacy-graph model decodes
    // from kv_self, whose recurrent state is one slot per sequence with no snapshot, and its memory module is
    // metadata-only with no cells (MEASURED 2026-10-07, qwen35-dense, test-recurrent-state-rollback: tail=-1 at
    // every seq_rm; nemotron_h on the same test: tail=0, second partial rm refused). Keeping n_rs_seq > 0 there
    // made a partial rm report success with the state not rolled back (replay logits 3.76e-6 from a never-advanced
    // reference vs 9.3e-10 on nemotron_h) and made the cache-level hybrid honesty check stand down. DSV4 keeps its own rollback.
    if (cparams.n_rs_seq > 0 && model.uses_legacy_graph_builder() && !llm_arch_is_dsv4(model.arch)) {
        LLAMA_LOG_WARN("%s: n_rs_seq=%u requested but %s runs on the legacy graph path, which keeps no recurrent snapshots; clamping to 0\n",
                       __func__, cparams.n_rs_seq, llama_model_arch_name(model.arch));
        cparams.n_rs_seq = 0;
    }

    cparams.n_threads               = params.n_threads;
    cparams.n_threads_batch         = params.n_threads_batch;
    cparams.yarn_ext_factor         = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor        = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast          = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow          = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings              = params.embeddings;
    cparams.embeddings_nextn        = false;
    cparams.embeddings_nextn_masked = false;
#if defined(GGML_USE_OPENVINO)
    // OpenVINO build (the GGML_USE_* chain makes it the only GPU-class backend): the fork's K/V store (CPY into a
    // view of the plain f16/f32 cache) and its FLASH_ATTN_EXT on cache views are translated (translate_cpy
    // op_case 11, translate_flash_attn_ext op_case 3), but the variants OV declines stay on the CPU: a quantized
    // cache type (-ctk/-ctv q8_0 ...), DSA sparse indices (src[5]), views of any other layout. A CPU op may not
    // write into a pre-allocated tensor of an OV device buffer (ggml-backend.cpp: "pre-allocated tensor ...
    // cannot run the operation"), so the cache lives in a CPU buffer, as with --no-kv-offload ("CPU KV buffer" in
    // the load log). OV still reads and writes it there: supports_buft accepts host buffers.
    if (params.offload_kqv && model.n_gpu_layers > 0) {
        LLAMA_LOG_INFO("%s: OpenVINO: K/V cache kept in host memory (CPU fallback for declined attention variants)\n", __func__);
        params.offload_kqv = false;
    }
#endif
    cparams.offload_kqv             = params.offload_kqv;
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    // ── Fork features this file did not carry, ported from llama.cpp's inline init ──────────
    //
    // WHY THESE AND NOT MORE. llama.cpp's llama_init_from_model is 918 lines and names 13 of the
    // fork's 18 feature tokens; this constructor is 586. Comparing them token by token, 10 of the
    // 13 were already handled here -- 4 in this constructor (dsv4, dflash, nextn, mtp) and 6
    // elsewhere in this file (mla_attn, offload_attn, snapkv, kvflash, h2o_, fused_moe). Only
    // THREE were absent from llama-context.cpp entirely: triattention, worst_graph_tokens and
    // repack. So the gap between the monolith's init and this constructor was never 918 lines of
    // divergence; it was eight occurrences of three features.
    //
    // Two of the three are assignments and land here. The third, repack, is a call to
    // llama_repack_up_gate_exps(), which is `static void` at llama.cpp:8521 -- internal linkage,
    // declared in no header, defined once and called once. It cannot be referenced from here
    // until it is moved or exposed, so it is deliberately NOT ported in this change and is the
    // one remaining item.
    //
    // A TOKEN COUNT IS NOT A BEHAVIOUR PROOF: this says these fields were never assigned here,
    // which is why the feature could not work from this path. It does not certify that the rest
    // of the two implementations agree.

    // worst-case graph sizing. The public field is worst_case_tokens; the internal one is
    // worst_graph_tokens. Both already existed -- llama_cparams declares worst_graph_tokens and
    // llama_context_params declares worst_case_tokens -- and nothing here ever connected them,
    // so a context built through this path silently sized its worst-case graph from a default.
    cparams.worst_graph_tokens = params.worst_case_tokens;
    // ik_llama #2384: DFlash query capacity override (0 = unset -> block size). Ported from ik's
    // llama_new_context_with_model, which this tree split into this file.
    cparams.dflash_query_capacity = params.dflash_query_capacity;

    // TriAttention KV scoring parameters. Copied into a std::string here so the caller's buffer
    // does not have to outlive this call -- the same reason llama.cpp gives at its own site.
    cparams.tri_stats_path = params.tri_stats_path ? params.tri_stats_path : "";
    cparams.tri_budget     = params.tri_budget;
    cparams.tri_window     = params.tri_window;
    cparams.tri_interval   = params.tri_interval;
    cparams.tri_sink       = params.tri_sink;

    // ── The fork's own context parameters -- RESTORED 2026-09-27 ─────────────────────────────────
    //
    // The comment above counted fork FEATURE TOKENS and concluded only three were missing. The
    // fields below were not counted by that method and NONE of them was assigned anywhere in src/:
    // measured with temp/_cparams_gap.py (every llama_cparams field vs every `cparams.<f> =` in
    // src/, controls n_ctx/flash_attn found and an invented name not found) -> 32 fields never
    // assigned, and `cparams = {}` (llama-context.h) zeroed them. So every one of these flags was
    // accepted by the CLI, carried through llama_context_params, and then silently dropped:
    // -khad/-vhad, -fmoe, -fug, -mla, -amb, -ser, --snapkv*, --h2o*, --offload-attn*, -rcache,
    // -gr, -smgs, -sas, -mtp, --reduce-type, --graph-attn-precision, --cuda-params, --defrag-thold.
    // MEASURED symptom: -khad and -vhad gave byte-identical perplexity to plain runs on every arm
    // (temp/_ksmooth_vs_khad.txt, K q4_0 128.5666 with and without).
    //
    // Source: this fork's own llama_init_from_model as it stood before WIRE-68 moved context
    // creation here (0940691c3^:src/llama.cpp), copied field for field, with its fixups below.
    cparams.defrag_thold             = params.defrag_thold;
    cparams.snapkv_cap               = params.snapkv_cap;           // S69 #22
    cparams.snapkv_window            = params.snapkv_window;
    cparams.h2o_heavy                = params.h2o_heavy;
    cparams.h2o_recent               = params.h2o_recent;
    cparams.kvflash_pool_tokens      = params.kvflash_pool_tokens;  // S69 #66 -- refused below
    cparams.offload_attn             = params.offload_attn;         // S69 #220
    cparams.offload_attn_page        = params.offload_attn_page;
    cparams.offload_attn_budget      = params.offload_attn_budget;
    cparams.offload_attn_selector    = params.offload_attn_selector;
    cparams.offload_attn_sink        = params.offload_attn_sink;
    cparams.offload_attn_recent      = params.offload_attn_recent;
    cparams.offload_attn_dense_layers = params.offload_attn_dense_layers;
    cparams.offload_attn_refresh     = params.offload_attn_refresh;
    cparams.offload_attn_scan        = params.offload_attn_scan;
    cparams.offload_attn_min_tokens  = params.offload_attn_min_tokens;
    cparams.offload_attn_head_reduce = params.offload_attn_head_reduce;
    cparams.offload_attn_dense_listed = false;
    for (int i = 0; i < 4; i++) cparams.offload_attn_dense_mask[i] = 0;
    if (params.offload_attn_dense_spec[0] != 0) {
        // S69-ENGREV-HIPATTENTION-FLAG-SURFACE: N | -N | explicit set, resolved against this model's layer count.
        ik_offload::DenseLayers dl; std::string derr;
        if (ik_offload::parse_dense_layers(params.offload_attn_dense_spec, (int) model.hparams.n_layer, dl, derr)) {
            cparams.offload_attn_dense_layers = dl.first_n;
            cparams.offload_attn_dense_listed = dl.listed;
            for (int i = 0; i < 4; i++) cparams.offload_attn_dense_mask[i] = dl.mask[i];
        } else {
            // fail open (ordinary attention), loudly: a wrong layer set must never be guessed.
            // DISABLE-REVERSIBLE: the error names the bad spec; correcting --offload-attn-dense-layers re-enables masking.
            // PRIOR-ART-NONE: input validation of a new flag, nothing to search; HiP configs are lists, parsed in offload_attention_dense.h
            LLAMA_LOG_ERROR("%s: --offload-attn-dense-layers '%s' rejected (%s); offload-attn DISABLED for this context\n",
                            __func__, params.offload_attn_dense_spec, derr.c_str());
            cparams.offload_attn = false;
        }
    }
    cparams.mla_attn                = params.mla_attn;
    cparams.attn_max_batch           = params.attn_max_batch;
    cparams.fused_moe_up_gate        = params.fused_moe_up_gate;
    cparams.grouped_expert_routing   = params.grouped_expert_routing;
    cparams.fused_up_gate            = params.fused_up_gate;
    cparams.fused_mmad               = params.fused_mmad;
    cparams.rope_cache               = params.rope_cache;
    cparams.graph_reuse              = params.graph_reuse;
    cparams.dsa                      = params.dsa;
    cparams.fused_idx_topk           = params.fused_idx_topk;
    cparams.swa_compress             = params.swa_compress;
    cparams.dsv4_legacy_state        = params.dsv4_legacy_state;   // ik_llama.cpp: --dsv4-legacy-state
    cparams.dsa_top_k                = params.dsa_top_k;
    cparams.prefetch_experts         = params.prefetch_experts;

    // MoE resident expert pool (S69-KT-EXPERT-POOL-2026-09-28). Port of llama.cpp PR #27861
    // (csantiago78) + discussion #28248 (memoriaru); design note in common/common.h.
    cparams.moe_pool_slots         = params.moe_pool_slots;
    cparams.moe_pool_inserts       = params.moe_pool_inserts;
    cparams.moe_pool_admit         = params.moe_pool_admit;
    cparams.moe_pool_cpu_only      = params.moe_pool_cpu_only;
    cparams.moe_stats_path         = params.moe_stats_path ? params.moe_stats_path : "";
    cparams.moe_stats_mode         = params.moe_stats_mode ? params.moe_stats_mode : "stat";
    cparams.moe_placement_strategy = params.moe_placement_strategy ? params.moe_placement_strategy : "uniform";
    cparams.moe_placement_stats    = params.moe_placement_stats ? params.moe_placement_stats : "";
    cparams.moe_dynamic_update     = params.moe_dynamic_update;
    cparams.moe_dynamic_threshold  = params.moe_dynamic_threshold;
    cparams.ffd                    = params.ffd;
    cparams.ffd_block              = params.ffd_block;
    cparams.ffd_delta              = params.ffd_delta;
    cparams.kivi_bits              = params.kivi_bits;
    cparams.kivi_group             = params.kivi_group;
    cparams.kivi_residual          = params.kivi_residual;
    cparams.xattn_threshold        = params.xattn_threshold;
    cparams.xattn_stride           = params.xattn_stride;
    cparams.xattn_block            = params.xattn_block;
    if (cparams.xattn_threshold != 0.0f && (!(cparams.xattn_threshold > 0.0f && cparams.xattn_threshold <= 1.0f) ||
            cparams.xattn_block <= 0 || cparams.xattn_stride <= 0 || cparams.xattn_block % cparams.xattn_stride != 0)) {
        throw std::runtime_error("--xattn-threshold must be in (0,1] with --xattn-block a positive multiple of --xattn-stride");
    }
    xattn_op.threshold = cparams.xattn_threshold;
    xattn_op.stride    = cparams.xattn_stride;
    xattn_op.block     = cparams.xattn_block;
    cparams.kv_diversity_mass      = params.kv_diversity_mass;
    cparams.kv_diversity_block     = params.kv_diversity_block;
    cparams.kv_diversity_layers    = params.kv_diversity_layers;
    cparams.snapkv_rebase          = params.snapkv_rebase;
    if (cparams.snapkv_rebase < -1 || cparams.snapkv_rebase > 1) {
        throw std::runtime_error("--snapkv-rebase must be 0 (off), 1 (re-base survivors) or -1 (negative control)");
    }
    if (cparams.kv_diversity_mass != 0.0f && (!(cparams.kv_diversity_mass > 0.0f && cparams.kv_diversity_mass <= 1.0f) ||
            cparams.kv_diversity_block <= 0 || cparams.kv_diversity_layers <= 0)) {
        throw std::runtime_error("--kv-diversity-mass must be in (0,1] with positive --kv-diversity-block and --kv-diversity-layers");
    }
    if (cparams.kv_diversity_mass > 0.0f && cparams.snapkv_cap <= 0 && cparams.h2o_heavy <= 0) {
        LLAMA_LOG_WARN("%s: --kv-diversity-mass only changes the SnapKV/H2O prune; neither --snapkv-cap nor --h2o-heavy is set\n", __func__);
    }
    if (cparams.ffd && (cparams.ffd_block <= 0 || !(cparams.ffd_delta > 0.0f))) {
        throw std::runtime_error("--ffd needs --ffd-block > 0 and --ffd-delta > 0");
    }
    if (cparams.moe_pool_slots > 0 && cparams.moe_pool_admit < 1) {
        throw std::runtime_error("--moe-expert-cache-admit must be >= 1");
    }

    // Validated, and folded into the model when set, by llama_fork_prepare_context_params before
    // this constructor runs.
    cparams.k_cache_hadamard         = params.k_cache_hadamard;
    cparams.v_cache_hadamard         = params.v_cache_hadamard;
    cparams.kv_fresh_chunk           = params.kv_fresh_chunk;
    cparams.split_mode_graph_scheduling = params.split_mode_graph_scheduling;
    cparams.scheduler_async          = params.scheduler_async;
    cparams.min_experts              = params.min_experts;
    cparams.thresh_experts           = params.thresh_experts;
    cparams.cuda_params              = params.cuda_params;
    cparams.mtp                      = params.mtp;
    // ik 894191d6e (#2644, MIT) MoE cache. The cache object, its graph use (llama-build-context.cpp)
    // and the scheduler copy callback merged earlier; creating it lives in ik's monolithic
    // llama_init_from_model, which this fork split into this constructor + sched_reserve(). The
    // up-front refusals are ik src/llama.cpp:9221-9233 (there `return nullptr`, here a throw that
    // llama_init_from_model turns into the same nullptr); the creation itself is in sched_reserve().
    cparams.moe_cache_size           = params.moe_cache_size;
    cparams.moe_cache_cpu_share      = params.moe_cache_cpu_share;
    cparams.moe_cache_policy         = (int) params.moe_cache_policy;
    if (cparams.moe_cache_size > 0) {
        if (model.fit) {
            throw std::runtime_error("MoE cache cannot be combined with --fit because fit does not reserve context-time cache memory");
        }
        if (model.hparams.n_expert == 0 || model.hparams.n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }
        if (!model.rpc_servers.empty()) {
            throw std::runtime_error("MoE cache does not support RPC devices");
        }
        if (!model.uses_legacy_graph_builder()) {
            // the cache hooks are in llm_build_context::llm_build_moe_ffn only; a dispatched models/ class builds
            // its MoE through llama-graph.cpp, which never asks the cache, so the budget would be allocated and unused
            throw std::runtime_error(format("MoE cache is wired into the legacy graph builder only; %s builds through "
                                            "the models/ dispatcher, which does not consult it",
                                            llama_model_arch_name(model.arch)));
        }
    }
    cparams.reduce_type              = params.type_reduce;
    cparams.graph_attn_precision     = params.type_graph_attn;
    cparams.idx_type_k               = params.idx_type_k;

    if (cparams.graph_attn_precision != GGML_TYPE_F16 && cparams.graph_attn_precision != GGML_TYPE_F32) {
        throw std::runtime_error(format("--graph-attn-precision must be f16 or f32, got %s",
                                        ggml_type_name(cparams.graph_attn_precision)));
    }
    if (cparams.swa_compress != model.swa_compress) {
        throw std::runtime_error(format("swa_compress differs between llama_model_params (%d) and "
                                        "llama_context_params (%d); the cache-size fit would not match the allocation",
                                        (int) model.swa_compress, (int) cparams.swa_compress));
    }
    // The DSA lightning indexer is built only in the layer-mode (non-TP) attention path. Under
    // -sm graph / -sm attn the model runs the tensor-parallel attention path, which has no indexer,
    // so --dsa would silently run dense MLA. Warn instead of degrading silently.
    if (cparams.dsa && (model.split_mode == LLAMA_SPLIT_MODE_GRAPH || model.split_mode == LLAMA_SPLIT_MODE_ATTN)) {
        LLAMA_LOG_WARN("%s: --dsa is not active under -sm graph/attn (tensor-parallel attention has no indexer); running dense MLA\n", __func__);
    }

    // CATS activation sparsity (S69-CATS-STAGE1-2026-09-28). --cats-sparsity 0 (the default) must
    // leave cparams.cats_thresholds EMPTY so llm_build_ffn / llm_build_moe_ffn take the exact
    // pre-existing fused graph path -- that is the bit-identity gate. Printed with LLAMA_LOG_INFO
    // (not just WARN-on-mismatch) specifically so a threshold vector that silently failed to reach
    // here is visible at every load, per the WIRE-68 lesson two comment blocks above this one.
    cparams.cats_sparsity  = params.cats_sparsity;
    cparams.cats_calibrate = params.cats_calibrate;
    if (cparams.cats_calibrate) {
        LLAMA_LOG_INFO("%s: CATS calibration mode ON: routing the un-fused silu(gate) path, UNMASKED\n", __func__);
    }
    if (cparams.cats_sparsity > 0.0f) {
        if (!params.cats_thresholds_file || !*params.cats_thresholds_file) {
            throw std::runtime_error("--cats-sparsity > 0 requires --cats-thresholds FILE (run tools/cats-calibrate first)");
        }
        cparams.cats_thresholds = cats_load_thresholds(params.cats_thresholds_file, model.hparams.n_layer);
        size_t n_active = 0;
        for (float t : cparams.cats_thresholds) if (t > 0.0f) ++n_active;
        LLAMA_LOG_INFO("%s: CATS activation sparsity ON: target=%.3f thresholds_file='%s' n_layer=%u n_layers_masked=%zu\n",
                       __func__, cparams.cats_sparsity, params.cats_thresholds_file,
                       model.hparams.n_layer, n_active);
        for (uint32_t il = 0; il < model.hparams.n_layer && il < cparams.cats_thresholds.size(); ++il) {
            LLAMA_LOG_INFO("%s: CATS layer %2u threshold = %.6f\n", __func__, il, cparams.cats_thresholds[il]);
        }
    } else if (params.cats_thresholds_file && *params.cats_thresholds_file) {
        LLAMA_LOG_WARN("%s: --cats-thresholds given but --cats-sparsity is 0 (off); ignoring the file\n", __func__);
    }

    // TEAL activation sparsity (S69-TEAL-STAGE1-2026-09-28). Same bit-identity contract as CATS
    // above: --teal-sparsity 0 (default) must leave all three teal_thresholds_* vectors EMPTY so
    // the attention block builder and llm_build_ffn take the exact pre-existing fused graph path.
    cparams.teal_sparsity  = params.teal_sparsity;
    cparams.teal_calibrate = params.teal_calibrate;
    if (cparams.teal_calibrate) {
        LLAMA_LOG_INFO("%s: TEAL calibration mode ON: routing the un-fused FFN path, UNMASKED\n", __func__);
    }
    if (cparams.teal_sparsity > 0.0f) {
        if (!params.teal_thresholds_file || !*params.teal_thresholds_file) {
            throw std::runtime_error("--teal-sparsity > 0 requires --teal-thresholds FILE (run tools/cats-calibrate --teal first)");
        }
        teal_thresholds_triple tt = teal_load_thresholds(params.teal_thresholds_file, model.hparams.n_layer);
        cparams.teal_thresholds_attn_in = tt.attn_in;
        cparams.teal_thresholds_ffn_in  = tt.ffn_in;
        cparams.teal_thresholds_ffn_mid = tt.ffn_mid;
        size_t n_active_attn = 0, n_active_ffn_in = 0, n_active_ffn_mid = 0;
        for (float t : cparams.teal_thresholds_attn_in) if (t > 0.0f) ++n_active_attn;
        for (float t : cparams.teal_thresholds_ffn_in)  if (t > 0.0f) ++n_active_ffn_in;
        for (float t : cparams.teal_thresholds_ffn_mid) if (t > 0.0f) ++n_active_ffn_mid;
        LLAMA_LOG_INFO("%s: TEAL activation sparsity ON: target=%.3f thresholds_file='%s' n_layer=%u "
                       "n_layers_masked(attn_in/ffn_in/ffn_mid)=%zu/%zu/%zu\n",
                       __func__, cparams.teal_sparsity, params.teal_thresholds_file,
                       model.hparams.n_layer, n_active_attn, n_active_ffn_in, n_active_ffn_mid);
        for (uint32_t il = 0; il < model.hparams.n_layer && il < cparams.teal_thresholds_attn_in.size(); ++il) {
            LLAMA_LOG_INFO("%s: TEAL layer %2u thresholds: attn_in=%.6f ffn_in=%.6f ffn_mid=%.6f\n", __func__, il,
                           cparams.teal_thresholds_attn_in[il], cparams.teal_thresholds_ffn_in[il], cparams.teal_thresholds_ffn_mid[il]);
        }
    } else if (params.teal_thresholds_file && *params.teal_thresholds_file) {
        LLAMA_LOG_WARN("%s: --teal-thresholds given but --teal-sparsity is 0 (off); ignoring the file\n", __func__);
    }
    // S69 #66 KVFlash: REFUSED until the paged-KV integration is finished -- slot_for(),
    // fill_slot_mask() and reselect() are not wired, so a pool-sized attention view has no slot
    // mapping and no validity mask. Measured before this guard existed: output similarity to the
    // no-flag baseline 0.368 (pool<ctx), 0.162 (pool==ctx), and a view-bounds assertion at pool>ctx.
    // Fluent and wrong. Forcing 0 restores the verified-inert path; the error line says why.
    if (cparams.kvflash_pool_tokens > 0) {
        LLAMA_LOG_ERROR("%s: --kvflash-pool=%d IGNORED. The paged-KV integration is incomplete:"
                        " slot_for(), fill_slot_mask() and reselect() are not wired. Falling back to"
                        " the ring KV cache.\n", __func__, cparams.kvflash_pool_tokens);
        cparams.kvflash_pool_tokens = 0;
    }
    if (!model.is_mla_model() && cparams.mla_attn != 0) {
        cparams.mla_attn = 0;
    } else if (model.n_gpu_layers > 0 && model.n_gpu_layers < (int) model.hparams.n_layer && cparams.mla_attn != 3) {
        LLAMA_LOG_WARN("%s: MLA models with ngl < n_layer and split mode graph do not work with mla = %d => changing mla to 3\n",
                       __func__, cparams.mla_attn);
        cparams.mla_attn = 3;
    }
    if (model.arch == LLM_ARCH_OPENAI_MOE && model.split_mode == LLAMA_SPLIT_MODE_GRAPH &&
        cparams.reduce_type == GGML_TYPE_F16) {
        LLAMA_LOG_WARN("%s: GPT-OSS with split mode graph requires f32 precision => changing reduce_type to f32\n", __func__);
        cparams.reduce_type = GGML_TYPE_F32;
    }
    if ((model.arch == LLM_ARCH_MELLUM || model.arch == LLM_ARCH_LAGUNA) && model.split_mode == LLAMA_SPLIT_MODE_GRAPH &&
        cparams.reduce_type == GGML_TYPE_F16) {
        LLAMA_LOG_WARN("%s: %s with split mode graph requires bf16 or f32 precision => changing reduce_type to bf16\n",
                       __func__, model.arch == LLM_ARCH_MELLUM ? "Mellum" : "Laguna");
        cparams.reduce_type = GGML_TYPE_BF16;
    }
    if (model.arch != LLM_ARCH_GLM4_MOE && model.arch != LLM_ARCH_QWEN35 &&
        model.arch != LLM_ARCH_QWEN35MOE && model.arch != LLM_ARCH_GEMMA4 &&
        model.arch != LLM_ARCH_GEMMA4_MTP && model.arch != LLM_ARCH_GLM_DSA &&
        model.arch != LLM_ARCH_GLM5NEXT &&   // ik_llama.cpp #2548: MTP for glm5next
        !llm_arch_is_dsv4(model.arch) && model.arch != LLM_ARCH_STEP35 &&
        model.arch != LLM_ARCH_GEMMA4_ASSISTANT && model.arch != LLM_ARCH_OPENPANGU &&
        model.arch != LLM_ARCH_QWEN4EXP &&   // ik_llama #2369: qwen4exp NextN self-speculation
        cparams.mtp) {
        cparams.mtp = false;
    }
    // Not ported: the pre-move code also pre-sized the output buffer for MTP here. The legacy decode
    // path's llama_output_reserve (src/llama.cpp) is MTP-aware (llama_context_has_mtp_outputs) and
    // grows the buffer on demand, so that pre-allocation is redundant rather than lost.

    LLAMA_LOG_INFO("%s: mla_attn = %d, attn_max_b = %d, fused_moe = %d, grouped er = %d, fused_up_gate = %d, "
                   "fused_mmad = %d, rope_cache = %d, graph_reuse = %d\n", __func__,
                   cparams.mla_attn, cparams.attn_max_batch, cparams.fused_moe_up_gate, cparams.grouped_expert_routing,
                   cparams.fused_up_gate, cparams.fused_mmad, cparams.rope_cache, cparams.graph_reuse);
    LLAMA_LOG_INFO("%s: k_cache_hadam = %d, v_cache_hadam = %d, split_mode_graph_scheduling = %d, reduce_type = %s, "
                   "sched_async = %d, ser = %d, %g, mtp = %d\n", __func__,
                   cparams.k_cache_hadamard, cparams.v_cache_hadamard, cparams.split_mode_graph_scheduling,
                   ggml_type_name(cparams.reduce_type), cparams.scheduler_async, cparams.min_experts,
                   cparams.thresh_experts, (int) cparams.mtp);
    if (cparams.cuda_params) {
        LLAMA_LOG_INFO("%s: cuda_params = %s\n", __func__, (const char *) cparams.cuda_params);
    }
    // WIRE-68 control: print exactly what was APPLIED, not just parsed, so a flag that
    // reaches the CLI but not cparams (the 32-field defect class this fork already hit
    // once) is visible at load time instead of silently dropped.
    if (cparams.moe_pool_slots > 0) {
        LLAMA_LOG_INFO("%s: moe_pool_slots = %d, moe_pool_inserts = %d, moe_pool_admit = %d, "
                       "moe_pool_cpu_only = %d, moe_placement_strategy = %s, moe_dynamic_update = %d, "
                       "moe_dynamic_threshold = %d\n", __func__,
                       cparams.moe_pool_slots, cparams.moe_pool_inserts, cparams.moe_pool_admit,
                       (int) cparams.moe_pool_cpu_only, cparams.moe_placement_strategy.c_str(),
                       (int) cparams.moe_dynamic_update, cparams.moe_dynamic_threshold);
    }
    if (!cparams.moe_stats_path.empty()) {
        LLAMA_LOG_INFO("%s: moe_stats_path = %s, moe_stats_mode = %s\n", __func__,
                       cparams.moe_stats_path.c_str(), cparams.moe_stats_mode.c_str());
    }

    // +1: id n_layer_eff() taps the output of the last layer ("input" of the head).
    // Spelled n_layer_eff() and not n_layer(): upstream's n_layer() is the EFFECTIVE count
    // (total minus nextn layers) and this fork's n_layer FIELD is the TOTAL, so the two are
    // different numbers on MTP models. Leaving the old spelling in a comment beside the new
    // spelling in the code re-plants exactly the confusion the rename exists to prevent.
    cparams.embeddings_layer_inp.resize(hparams.n_layer_eff() + 1, false);
    embd_layer_inp.resize(hparams.n_layer_eff() + 1);

    cparams.ctx_type          = params.ctx_type;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.pooling_type      = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    // S69-KT-EXPERT-POOL-2026-09-28 unit 1: install the recorder's cb_eval wrapper HERE, directly
    // after cparams.cb_eval/cb_eval_user_data are bridged from params, so moe.prev_cb_eval captures
    // whatever the caller passed in llama_context_params, and BEFORE the scheduler is created
    // (src/llama.cpp:7249 and :7815 both propagate whatever cparams.cb_eval holds at that point) --
    // so no separate ggml_backend_sched_set_eval_callback call is needed, unlike
    // llama_set_dflash_capture_layers which installs after sched exists.
    // MEASURED 2026-09-28: this block used to sit ~20 lines ABOVE the bridge, which then overwrote
    // cparams.cb_eval with params.cb_eval -- the recorder was never called and
    // test-moe-pool-recorder-e2e on stories15M_MOE recorded 0 layers.
    if (!cparams.moe_stats_path.empty()) {
        moe.recorder = std::make_unique<llama_moe_recorder>(
                llama_moe_recorder_mode_from_str(cparams.moe_stats_mode));
        moe.prev_cb_eval           = cparams.cb_eval;
        moe.prev_cb_eval_user_data = cparams.cb_eval_user_data;
        cparams.cb_eval            = llama_moe_record_eval_callback;
        cparams.cb_eval_user_data  = this;
    }

    cparams.ctx_other = nullptr;

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH) {
        if (model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr) {
                throw std::runtime_error(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        }
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        cparams.rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // mscale: the GGUF's <arch>.rope.scaling.yarn_mscale (HF rope_parameters["mscale"]) when the converter
            // wrote it. GGUFs without the key keep the historical assumption (mscale == 1.0f, or mscale_all_dim for
            // DEEPSEEK2), which equals every released config (mscale == mscale_all_dim for the DeepSeek/Kimi family).
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            if (hparams.rope_yarn_mscale != 0.0f) {
                mscale = hparams.rope_yarn_mscale;
            } else if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
                // special-case DEEPSEEK v2:
                // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = llama_yarn_attn_factor_ratio(factor, mscale, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = llama_yarn_get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = false;

    cparams.fused_lid = true;
    cparams.auto_flid = false;
    // LLAMA_FUSED_LID=0|1 and LLAMA_FUSED_TKM=0|1 force the fused lightning-indexer / fused top-k-mask ops off or on for A/B runs
    // (the glm5-next two-implementation parity bisect, row S69-ENGREV-UPSTREAM-MERGE-GLM5-NEXT-TWO-IMPLEMENTATIONS-2026-11-04-01).
    // Unset = unchanged.
    if (const char * env_lid = getenv("LLAMA_FUSED_LID"); env_lid && env_lid[0]) {
        cparams.fused_lid = atoi(env_lid) != 0;
    }

    cparams.fused_dsv4_hc_pre  = true;
    cparams.fused_dsv4_hc_comb = true;
    cparams.fused_dsv4_hc_post = true;
    cparams.auto_fhc           = true;

    // No probe for this one: the graph builders ask the layer's device (supports_op) per layer
    // and build the unfused chain where it says no, so the flag only exists for A/B runs.
    cparams.fused_gated_norm = true;
    if (const char * env_ggn = getenv("LLAMA_FUSED_GATED_NORM")) {
        cparams.fused_gated_norm = atoi(env_ggn) != 0;
        LLAMA_LOG_INFO("%s: LLAMA_FUSED_GATED_NORM=%s forces the fused gated group RMSNorm %s\n",
                       __func__, env_ggn, cparams.fused_gated_norm ? "on" : "off");
    }
    // only the indexer models build this mask, skip the probe for the others
    cparams.fused_top_k_mask = true;
    cparams.auto_ftkm        = hparams.indexer_top_k > 0;
    if (const char * env_tkm = getenv("LLAMA_FUSED_TKM"); env_tkm && env_tkm[0]) {
        cparams.fused_top_k_mask = atoi(env_tkm) != 0;
        cparams.auto_ftkm        = false;
    }

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;
    cparams.n_outputs_max_per_seq = params.n_outputs_max_per_seq == 0 ?
            cparams.n_outputs_max : std::min(params.n_outputs_max_per_seq, cparams.n_outputs_max);

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    cparams.op_offload     = params.op_offload;
    cparams.kv_unified     = params.kv_unified;
    cparams.moe_cache_size = params.moe_cache_size;
    cparams.moe_cache_cpu_share = params.moe_cache_cpu_share;
    cparams.moe_cache_policy    = (int) params.moe_cache_policy;

    // A src/models class builds one attention mask per KV stream ([n_kv, n_tokens, 1, n_stream]); this fork's ggml_flash_attn_ext only accepts a single
    // stream (mask->ne[3] == 1), so llama-perplexity's default 16 sequences, or a server with parallel slots, aborted on the first decode of any class
    // arch ("GGML_ASSERT(mask->ne[3] == 1)", MEASURED 2026-10-04 on falcon-h1). Sharing one KV buffer between the sequences is equivalent for attention
    // (the mask separates them) and keeps a single stream. Legacy-builder models have their own mask handling and are left alone.
    if (!cparams.kv_unified && cparams.n_seq_max > 1 && !model.uses_legacy_graph_builder() && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        LLAMA_LOG_WARN("%s: n_seq_max = %u with flash attention on a %s model: using a unified KV cache (one stream), as per-sequence streams need mask->ne[3] > 1\n",
                __func__, cparams.n_seq_max, llama_model_arch_name(model.arch));
        cparams.kv_unified = true;
    }

    // resolved after flash attention, see resolve_fused_ops()
    {
        cparams.fused_kv_rows = false;

        const char * LLAMA_KV_ROWS_DISABLE = getenv("LLAMA_KV_ROWS_DISABLE");
        if (!LLAMA_KV_ROWS_DISABLE) {
            cparams.auto_fkvr = cparams.kv_unified;
        } else {
            LLAMA_LOG_INFO("%s: LLAMA_KV_ROWS_DISABLE set, disabling kv_rows\n", __func__);
            cparams.auto_fkvr = false;
        }
    }

    // initialized later
    cparams.pipeline_parallel = false;
    cparams.training = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    // ik_llama #2384 (SWA compression with DFlash), ported from its llama_new_context_with_model: the cross
    // context a DFlash step can see is n_ctx minus the query capacity, compressed to the SWA window when
    // --swa-compress is on. NOTE: ik pads n_ctx with kv get_padding(flash_attn) at this point, this file pads to 256.
    if (llm_arch_is_dflash_family(model.arch)) {
        const int32_t query_capacity = cparams.dflash_query_capacity > 0
                ? cparams.dflash_query_capacity
                : std::max<int32_t>(1, (int32_t) model.hparams.dflash_block_size);
        const int32_t logical_cross_ctx = std::max<int32_t>(1,
                (int32_t) cparams.n_ctx - query_capacity);
        const int32_t physical_cross_ctx = model.dflash_swa_compress_cross_ctx(
                logical_cross_ctx, cparams.swa_compress);
        dflash.visible_cross_ctx = physical_cross_ctx;
        LLAMA_LOG_INFO("%s: DFlash context logical_cross_ctx=%d physical_cross_ctx=%d query_capacity=%d window=%u\n",
                __func__, logical_cross_ctx, physical_cross_ctx, query_capacity, model.hparams.n_swa);
    }

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max             = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx                 = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq             = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch               = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch              = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn           = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn            = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified            = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base             = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale            = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq              = %u\n",   __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_outputs_max         = %u\n",   __func__, cparams.n_outputs_max);
    LLAMA_LOG_INFO("%s: n_outputs_max_per_seq = %u\n",   __func__, cparams.n_outputs_max_per_seq);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // ============================================================================================
        // UNIFIED BACKEND INITIALISATION -- slice 29, 2026-09-09.
        //
        // This block carries TEN capabilities: eight from this fork's llama_new_context_with_model
        // (src/llama.cpp) and two from the upstream constructor that was migrated in here. It is a
        // MERGE, not a port in either direction -- each side had capabilities the other lacked, and
        // dropping any one of them still compiles:
        //
        //   FROM THE FORK   1 main_gpu_id translation   2 split-mode branching
        //                   3 cuda_params threading     4 Metal arm
        //                   5 Vulkan arm + its split-mode refusal
        //                   6 SYCL and CANN arms        7 RPC backends
        //                   8 the --device filter and the name-based reorder
        //   FROM UPSTREAM   A ACCEL backend enumeration (BLAS and similar) -- the fork has none
        //                   B the set_n_threads function table -- the fork sets threads on the CPU
        //                     backend only, hard-wired at src/llama.cpp:6441 and :6463
        //
        // WHAT REPLACED WHAT, AND WHY. The migrated code here used upstream's registry GPU loop,
        // which iterates device HANDLES and asks each for its .dev member. This fork's
        // model.devices is std::vector<int32_t> -- device INDICES (src/llama-model.h:962) -- so that
        // member access was being applied to a plain int, and it was two of this file's remaining
        // compile errors. Upstream's loop is not defective; the type it iterates is absent here.
        //
        // Three cheaper reconciliations were tried and refuted before the arms below were chosen:
        // casting the ordinal to a handle (undefined -- an ordinal is not a pointer); indexing the
        // GLOBAL registry with it (the ordinals are BACKEND-LOCAL, fed straight to
        // ggml_backend_cuda_init / vk_init / sycl_init, while ggml_backend_dev_get indexes a global
        // list that also holds CPU and ACCEL devices, so it is silently wrong on any host whose
        // registry is not GPU-only); and an #else arm enumerating all GPU-type devices from the
        // registry (invented semantics -- upstream's loop honours the selection made at MODEL LOAD,
        // which all-GPUs-then-filter is not).
        //
        // THE DIFFERENCE THAT SUPERSESSION LEAVES BEHIND, stated because it is real: upstream's
        // registry loop allows several different backends to coexist in one build, while the
        // #if/#elif chain below is mutually exclusive -- one GPU backend per build. That is this
        // fork's existing architecture (backend-local ordinals plus main_gpu as a local index), not
        // something introduced here, and unifying it is a separate decision.
        //
        // ORDER IS LOAD-BEARING. The reorder block REPLACES backends with only those the user named
        // in --device. ACCEL and CPU are therefore added AFTER it: a user passing `--device CUDA0`
        // must not silently lose BLAS, and must never lose the CPU fallback path.
        // ============================================================================================

        // [1] main_gpu is a LOCAL index into model.devices throughout the codebase (auto-fit assigns
        //     device_count-1, MTP clamps to [0, device_count), buffer-type setup wraps with
        //     model.devices[main_gpu]). Translate to a raw device id here.
        [[maybe_unused]] const int main_gpu_id = (model.main_gpu >= 0 && model.main_gpu < (int)model.devices.size())
            ? model.devices[model.main_gpu]
            : model.main_gpu;

#if defined(GGML_USE_METAL)
        // [4] Metal arm.
        if (model.n_gpu_layers > 0) {
            backend_metal = ggml_backend_metal_init();
            if (backend_metal == nullptr) {
                throw std::runtime_error("failed to initialize Metal backend");
            }
            ggml_backend_add_from_device(this, backend_metal);
        }
#elif defined(GGML_USE_CUDA)
        // [2] split-mode branching and [3] cuda_params.
        if (model.split_mode == LLAMA_SPLIT_MODE_NONE) {
            // with split_mode LLAMA_SPLIT_MODE_NONE or LLAMA_SPLIT_MODE_GRAPH, only the main GPU backend is used
            ggml_backend_t backend = ggml_backend_cuda_init(main_gpu_id, cparams.cuda_params, this);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize CUDA%d backend", main_gpu_id));
            }
            ggml_backend_add_from_device(this, backend);
        } else {
            // LLAMA_SPLIT_MODE_LAYER and LLAMA_SPLIT_MODE_GRAPH require a backend for each GPU
            auto cuda_init_params = cparams.cuda_params;
            std::string new_params;
            if (false && model.split_mode == LLAMA_SPLIT_MODE_GRAPH) {
                static const std::string extra_string{"graphs=0"};
                if (cuda_init_params) new_params = std::string{(const char *)cuda_init_params} + ',';
                new_params += extra_string;
                cuda_init_params = new_params.data();
            }
            for (int device = 0; device < ggml_backend_cuda_get_device_count(); ++device) {
                ggml_backend_t backend = ggml_backend_cuda_init(device, cuda_init_params, this);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize CUDA%d backend", device));
                }
                ggml_backend_add_from_device(this, backend);
            }
        }
#elif defined(GGML_USE_VULKAN)
        // [5] Vulkan arm. The refusal below is a GUARD, not just a backend: these split modes are
        //     unimplemented for Vulkan and must fail loudly rather than run wrong.
        if (model.split_mode == LLAMA_SPLIT_MODE_GRAPH || model.split_mode == LLAMA_SPLIT_MODE_ATTN) {
            throw std::runtime_error("split mode 'graph' or 'attn' not supported. Failed to initialize Vulkan backend");
        }
        if (model.split_mode == LLAMA_SPLIT_MODE_NONE) {
            ggml_backend_t backend = ggml_backend_vk_init(main_gpu_id);
            if (backend == nullptr) {
                throw std::runtime_error("failed to initialize Vulkan backend");
            }
            ggml_backend_add_from_device(this, backend);
        } else {
            for (int device = 0; device < ggml_backend_vk_get_device_count(); ++device) {
                ggml_backend_t backend = ggml_backend_vk_init(device);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize Vulkan%d backend", device));
                }
                ggml_backend_add_from_device(this, backend);
            }
        }
#elif defined(GGML_USE_SYCL)
        // [6] SYCL arm. Both branches filter -- see fork d1610ec3f, where the first one did not.
        if (model.split_mode == LLAMA_SPLIT_MODE_NONE || model.split_mode == LLAMA_SPLIT_MODE_GRAPH) {
            ggml_backend_t backend = ggml_backend_sycl_init(main_gpu_id);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize SYCL%d backend", main_gpu_id));
            }
            ggml_backend_add_from_device(this, backend);
        } else {
            // LLAMA_SPLIT_MODE_LAYER requires a backend for each GPU
            for (int i = 0; i < ggml_backend_sycl_get_device_count(); ++i) {
                ggml_backend_t backend = ggml_backend_sycl_init(i);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize SYCL%d backend", i));
                }
                ggml_backend_add_from_device(this, backend);
            }
        }
#elif defined(GGML_USE_CANN)
        // [6] CANN arm. TODO retained from src/llama.cpp: CANN does not split tensors yet.
        if (model.split_mode == LLAMA_SPLIT_MODE_NONE || model.split_mode == LLAMA_SPLIT_MODE_GRAPH) {
            ggml_backend_t backend = ggml_backend_cann_init(main_gpu_id);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize CANN%d backend", main_gpu_id));
            }
            ggml_backend_add_from_device(this, backend);
        } else {
            for (int32_t device = 0; device < ggml_backend_cann_get_device_count(); ++device) {
                ggml_backend_t backend = ggml_backend_cann_init(device);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize CANN%d backend", device));
                }
                ggml_backend_add_from_device(this, backend);
            }
        }
#elif defined(GGML_USE_OPENVINO)
        // [6b] OpenVINO arm (S69-FORK-OPENVINO-BACKEND-NOT-REACHABLE-FROM-MODEL-LOADER). One backend for
        //     the selected device, only when layers are offloaded -- the same gate Metal uses -- so a
        //     -ngl 0 run keeps the pure CPU path. Without this arm the registered OPENVINO0 device
        //     could be named but never given a backend, and the scheduler had nothing to place nodes on.
        if (model.n_gpu_layers > 0) {
            ggml_backend_t backend = ggml_backend_openvino_init(main_gpu_id);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize OpenVINO%d backend", main_gpu_id));
            }
            ggml_backend_add_from_device(this, backend);
        }
#endif

#if defined(GGML_USE_RPC)
        // [7] RPC backends. Nothing in the migrated block touched these at all.
        if (model.n_gpu_layers > 0) {
            for (const auto & rpc_dev : model.rpc_servers) {
                ggml_backend_t backend = ggml_backend_rpc_init(rpc_dev.endpoint.c_str(), rpc_dev.device);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize RPC%d to '%s'",
                                rpc_dev.device, rpc_dev.endpoint.c_str()));
                }
                ggml_backend_add_from_device(this, backend);
            }
        }
#endif

        // [9] EXTRA registry devices (S69-NPU-LOADER-EXTRA-BACKEND-ARM-REGISTRY-DEVICES-2026-11-06-01): a registered GPU / IGPU /
        //     ACCEL-with-own-memory device that the chain above does not know (HEXAGON, a GGML_BACKEND_DL plugin, a future
        //     RKNPU / XDNA backend). The model load already decided which of them take part (model.devices holds their
        //     ordinals, only on `-dev NAME`), so the filter here is that selection, not the backend name. With no extra
        //     device this loop does nothing and the backend list is exactly what it was.
        if (model.n_gpu_layers > 0) {
            const std::vector<ggml_backend_dev_t> extras = llama_extra_registry_devices();
            const int first = llama_extra_device_first_ordinal(model);
            for (size_t e = 0; e < extras.size(); ++e) {
                if (std::find(model.devices.begin(), model.devices.end(), first + (int) e) == model.devices.end()) {
                    continue;
                }
                ggml_backend_t backend = ggml_backend_dev_init(extras[e], nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(extras[e])));
                }
                backends.push_back(backend);
            }
        }

        // [8] name-based reorder. Every arm above already filtered through
        //     ggml_backend_add_from_device; this puts the survivors into the ORDER the user listed.
        //     It runs before ACCEL and CPU are added, deliberately -- it REPLACES the vector.
        if (cparams.devices.size()) {
            std::vector<ggml_backend_t> reordered = {};
            for (const auto & device : cparams.devices) {
                ggml_backend_t backend = ggml_backend_by_name(device.c_str());
                if (backend) {
                    reordered.push_back(backend);
                }
            }
            backends = std::move(reordered);
        }

        // [A] ACCEL backends (such as BLAS) -- UPSTREAM capability, zero occurrences in the fork.
        //     Added AFTER the reorder so a --device list naming only GPUs cannot silently drop it.
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                // an ACCEL with memory of its own that [9] already attached must not get a second backend
                bool attached = false;
                for (ggml_backend_t b : backends) {
                    attached = attached || ggml_backend_get_device(b) == dev;
                }
                if (attached) {
                    continue;
                }
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // CPU backend. Deliberately unfiltered and added last: the context must always retain a
        // fallback compute path. This keeps the FORK's direct ggml_backend_cpu_init() rather than
        // upstream's ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr), which
        // resolves through the ggml device registry and returns nullptr when that registry holds no
        // CPU entry. The direct call performs no registry lookup and is what the live build already
        // uses; swapping the live CPU acquisition path as a side effect of a migration slice is risk
        // for no gain.
        backend_cpu = ggml_backend_cpu_init();
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // ik afdbc5e13 (#2565): optional worker-thread pinning, applied at context creation.
        if (params.cpu_affinity != nullptr && params.n_cpu_affinity > 0) {
            llama_set_cpu_affinity(this, params.cpu_affinity, params.n_cpu_affinity);
        }

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend, ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output),
                    ggml_backend_buffer_get_size(buf_output) / 1024.0 / 1024.0);
        }

        // ── TriAttention KV scoring init, ported from llama.cpp's inline llama_init_from_model ──
        //
        // Placed immediately after the output buffer for the same reason it sits there in the
        // monolith: it is the last step of context construction and needs cparams already filled.
        //
        // GUARDED ON BOTH a stats path AND a positive budget, and that guard is the point rather
        // than defensiveness -- the comment at the original site says so: a stats file with a zero
        // budget builds a runtime that evicts nothing, "which is worse than staying off because it
        // would report as enabled". A feature that reports enabled while doing nothing is exactly
        // the class of defect this merge has spent its length removing.
        if (!cparams.tri_stats_path.empty() && cparams.tri_budget > 0.0f) {
            struct tria_stats * tri_stats = tria_load(cparams.tri_stats_path.c_str());
            if (tri_stats) {
                const int budget_pct = (int) (cparams.tri_budget * 100.0f);
                tria_rt = tria_runtime_init(tri_stats, budget_pct,
                                            cparams.tri_window,
                                            cparams.tri_interval,
                                            cparams.tri_sink);
                if (tria_rt) {
                    g_tria_rt = tria_rt;
                    LLAMA_LOG_INFO("%s: triattention enabled: budget=%d%%, window=%d, "
                                   "interval=%d, sink=%d\n",
                                   __func__, budget_pct, cparams.tri_window,
                                   cparams.tri_interval, cparams.tri_sink);
                } else {
                    LLAMA_LOG_WARN("%s: triattention runtime init failed (budget_pct=%d)\n",
                                   __func__, budget_pct);
                    tria_free(tri_stats);
                }
            } else {
                LLAMA_LOG_WARN("%s: triattention stats could not be loaded from %s\n",
                               __func__, cparams.tri_stats_path.c_str());
            }
        }
    }

    // Per-layer K/V cache types, resolved once and used by BOTH caches (memory module and legacy kv_self).
    // Boundary V (llama-turbo-boundary-v.h), OPT-IN with LLAMA_TURBO_BOUNDARY_V=1: a turbo2 V cache keeps the
    // first/last two layers' V at q8_0 unless -ctv-first/-ctv-last were given (source: github.com/TheTom/turboquant_plus,
    // docs/turboquant-recommendations.md "Boundary V (auto-enabled for turbo2-V)"). Not our default: it raised PPL
    // on both models measured (numbers in the header).
    struct {
        ggml_type type_k_first, type_k_last, type_v_first, type_v_last;
        int32_t   n_k_first, n_k_last, n_v_first, n_v_last;
        int64_t   n_layer;
    } kv_types = {
        params.type_k_first, params.type_k_last, params.type_v_first, params.type_v_last,
        params.n_k_first,    params.n_k_last,    params.n_v_first,    params.n_v_last,
        // the count "last N" is measured from -- the legacy cache's own formula (src/llama.cpp, llama_kv_cache_init)
        model.mtp ? (int64_t) hparams.n_layer : (int64_t) hparams.n_layer - hparams.nextn_predict_layers,
    };
    {
        const llama_turbo_boundary_v bv = llama_turbo_boundary_v_resolve(
                params.type_v, kv_types.type_v_first, kv_types.type_v_last, kv_types.n_v_first, kv_types.n_v_last,
                kv_types.n_layer, getenv("LLAMA_TURBO_BOUNDARY_V"), getenv("TURBO_LAYER_ADAPTIVE"));
        if (bv.applied) {
            kv_types.type_v_first = bv.type_v_first; kv_types.n_v_first = bv.n_v_first;
            kv_types.type_v_last  = bv.type_v_last;  kv_types.n_v_last  = bv.n_v_last;
            LLAMA_LOG_INFO("%s: Boundary V: turbo2 V cache, first %d + last %d layers' V kept at %s "
                           "(opt-in LLAMA_TURBO_BOUNDARY_V=1)\n", __func__,
                           bv.n_v_first, bv.n_v_last, ggml_type_name(bv.type_v_first));
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        // LLAMA_KV_MEMORY_METADATA_ONLY=0 restores the pre-2026-09-29 behaviour (memory module gets its
        // own K/V allocation next to kv_self). A diagnostic / rollback switch and the A/B toggle for
        // examples/kv-dedup-check; the default (unset) is metadata-only for legacy-graph models.
        bool memory_metadata_only = model.uses_legacy_graph_builder();
        if (memory_metadata_only) {
            const char * env_mo = getenv("LLAMA_KV_MEMORY_METADATA_ONLY");
            if (env_mo != nullptr && atoi(env_mo) == 0) {
                LLAMA_LOG_WARN("%s: LLAMA_KV_MEMORY_METADATA_ONLY=0 -- the memory module allocates its own K/V next to kv_self\n", __func__);
                memory_metadata_only = false;
            }
        }

        llama_memory_params params_mem = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
            // A legacy-graph model decodes from kv_self (allocated below) and never reads this
            // module's tensors: keep the cells and bookkeeping, skip the second K/V allocation.
            // S69-KV-DEDUP-METADATA-ONLY-MEMORY-MODULE-2026-09-29-01.
            /*.metadata_only =*/ memory_metadata_only,
        };

        // per-layer K/V types for the memory-module cache (llama-kv-layer-types.h): before this the memory module
        // built every layer at type_k/type_v and the -ctk/-ctv-first/-last flags only reached the legacy cache
        const llama_kv_layer_types kv_layer_types = {
            kv_types.type_k_first, kv_types.type_k_last, kv_types.type_v_first, kv_types.type_v_last,
            kv_types.n_k_first,    kv_types.n_k_last,    kv_types.n_v_first,    kv_types.n_v_last,
            kv_types.n_layer,
        };
        llama_kv_layer_types_scope kv_layer_types_scope(model.arch == LLM_ARCH_OPENPANGU ? nullptr : &kv_layer_types);
        memory.reset(model.create_memory(params_mem, cparams));
    }

    // INITIALISE THE LEGACY KV CACHE. Every model in this fork takes the legacy graph path --
    // src/models/ holds 152 per-architecture classes and NOTHING constructs one -- so
    // llm_build_context::llama_build_graph builds every graph, and it reads lctx.kv_self through
    // its `kv` reference member (src/llama-build-context.cpp:913). This constructor builds
    // `memory` (a llama_kv_cache) and, before this, never touched kv_self except to release it in
    // teardown. The builders therefore hit GGML_ASSERT(kv.size == n_ctx) at
    // src/llama-build-context.cpp:930 on every load.
    //
    // llama_kv_cache_init already existed and had ZERO callers: the monolith's call went with the
    // monolith. Nothing here is new logic -- this is the missing call.
    //
    // FOOTPRINT: `memory` is built metadata-only for these models (llama_memory_params::metadata_only,
    // set above from model.uses_legacy_graph_builder()): its cells and sequence bookkeeping stay
    // live, its K/V/R/S tensors point at zero-size dummy buffers. So kv_self is the ONLY KV
    // allocation. Before this the two were both allocated (MEASURED 2026-09-29: NVML growth per
    // context token was 2.00x the reported KV growth on qwen2.5-7b).
    if (!hparams.vocab_only && model.uses_legacy_graph_builder()) {
        LLAMA_LOG_INFO("%s: initialising the legacy KV cache for the llm_build_context path "
                       "(the memory module above is metadata-only: cells live, no second K/V allocation)\n", __func__);

        // MAMBA keeps a constant number of cells PER SEQUENCE, not per token -- restored from the
        // pre-split monolith (ik_llama_src/src/llama.cpp:5512-5519), which the port dropped. With
        // n_ctx cells, build_mamba's reshape of k_l[il] to {n_embd_k_s, kv_self.size} asserts at
        // ggml.c:8303 on every load (MEASURED 2026-09-25). F32 states: ggml_ssm_conv / ssm_scan
        // require them for conv_states / ssm_states.
        uint32_t  legacy_kv_size = cparams.n_ctx;
        ggml_type legacy_type_k  = params.type_k;
        ggml_type legacy_type_v  = params.type_v;
        if (model.arch == LLM_ARCH_MAMBA) {
            legacy_kv_size = std::max((uint32_t) 1, params.n_seq_max);
            legacy_type_k  = GGML_TYPE_F32;
            legacy_type_v  = GGML_TYPE_F32;
        }
        // ik_llama #2345 (DFlash2), ported from its llama_new_context_with_model, which this tree split into this file:
        // a DFlash model keeps its history in a custom cache, so the ordinary KV cache is only bookkeeping and is sized to
        // one block, not n_ctx.
        if (llm_arch_is_dflash_family(model.arch)) {
            const uint32_t pad = llama_kv_cache_fork::get_padding(cparams.flash_attn);
            legacy_kv_size = GGML_PAD(std::max<uint32_t>(1u, hparams.dflash_block_size), pad);
            LLAMA_LOG_INFO("%s: DFlash ordinary KV bookkeeping capacity = %u (history owned by custom cache)\n",
                    __func__, legacy_kv_size);
        }

        if (!llama_kv_cache_init(kv_self, this,
                                 legacy_type_k, legacy_type_v, params.idx_type_k,
                                 legacy_kv_size, params.offload_kqv,
                                 kv_types.type_k_first, kv_types.type_k_last,
                                 kv_types.type_v_first, kv_types.type_v_last,
                                 kv_types.n_k_first,    kv_types.n_k_last,
                                 kv_types.n_v_first,    kv_types.n_v_last)) {
            throw std::runtime_error("failed to initialise the legacy KV cache (kv_self)");
        }

        // --kivi-bits: simulated KIVI KV quantisation; throws (refuses the flag) when the cache cannot support it
        if (cparams.kivi_bits != 0) {
            kivi_sim = llama_kivi_sim_init(*this);
        }

        // REPORT WHAT IT ACTUALLY PRODUCED. llama_kv_cache_init returns a bool, and a `true` from
        // it is NOT evidence that any per-layer tensor exists: every graph builder then indexes
        // kv_self.k_l[il] directly, so an empty vector here becomes a segfault inside ggml rather
        // than an error here. These are the exact quantities the builders index.
        // COUNTED, not sized: the per-layer loop in llama_kv_cache_init PUSHES NULLPTR and
        // continues for any layer it skips (src/llama.cpp:1488, :1512, :1592), so k_l.size() is
        // n_layer whether or not a single tensor was created. The non-null count is the quantity
        // the graph builders actually depend on.
        size_t n_k_live = 0;
        for (const auto * t : kv_self.k_l) {
            if (t != nullptr) {
                n_k_live++;
            }
        }

        LLAMA_LOG_INFO("%s: legacy KV cache ready: size=%u  k_l=%zu (%zu non-null)  v_l=%zu  "
                       "s_l=%zu  bufs=%zu  mtp_op_type=%d\n",
                       __func__, kv_self.size,
                       kv_self.k_l.size(), n_k_live, kv_self.v_l.size(),
                       kv_self.s_l.size(), kv_self.bufs.size(), (int) cparams.mtp_op_type);

        if (n_k_live == 0) {
            // k_l.empty() was the WRONG test and let a vector of 30 nulls through to the graph
            // builders, which index it and fault inside ggml. This is the test that matches what
            // they require.
            throw std::runtime_error(
                    "legacy KV cache reported success but every per-layer K tensor is null -- "
                    "every graph builder indexes kv_self.k_l[il], so continuing would fault "
                    "inside ggml instead of here");
        }

        // Route the public llama_memory_*() API to kv_self (see llama_memory_i::legacy_owner).
        if (memory) {
            memory->legacy_owner = this;
        }
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend);
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the pinned host buffer for faster transfer of the intermediate state
                //
                // SLICE 24, 2026-09-09. Upstream read this off a device HANDLE, taking the first
                // element of model.devices and asking it for its device member. This fork's
                // model.devices is std::vector<int32_t> -- device INDICES, not handles
                // (src/llama-model.h:962) -- so that member access was being applied to a plain int.
                // The divergence is DELIBERATE and documented at src/llama-model.h:1032-1039:
                // upstream's llama_device-handle variant was excluded because this fork's int32
                // vector has live consumers indexing it directly, and two members of the same name
                // with different element types cannot both be declared.
                //
                // llama_default_buffer_type_cpu(true) is this fork's equivalent and returns exactly
                // what upstream was after -- under GGML_USE_CUDA it is
                // ggml_backend_cuda_host_buffer_type() (src/llama.cpp:434-441). The fork already
                // uses it for this same purpose at llama.cpp:566, :1382, :4451 and :4457. Declared
                // in src/llama-impl.h:180, already included here at line 6.
                //
                // ONE REAL DIFFERENCE, not glossed: upstream's accessor is per-DEVICE, this one is
                // per-BUILD -- it selects on the compiled backend rather than a device id. For CUDA
                // that is not a loss, because the pinned-host buffer type belongs to the runtime and
                // not to any one GPU, which is why the fork's signature takes no id. The two would
                // differ only on a multi-vendor build, which this fork's #if structure (exactly one
                // of CUDA / Metal / Vulkan compiled) does not produce.
                auto * host_buft = llama_default_buffer_type_cpu(true);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend);
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.device_count() > 1 &&
            model.n_gpu_layers > model.hparams.n_layer &&
            model.split_mode == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides() &&
            cparams.moe_cache_size == 0; // ik 894191d6e src/llama.cpp:9922: the cache's copy callback syncs per split

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend);
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        // the MoE cache is created in sched_reserve(), after the up/gate repack whose tensors it binds

        sched_reserve();

        // ── Fork scheduler options -- RESTORED 2026-09-27 from 0940691c3^:src/llama.cpp ──────────
        // llama_init_from_model applied these to the scheduler right after the worst-case reserve.
        // WIRE-68 moved context creation here without them, so --op-offload policies,
        // only_active_experts, --prefetch-experts and split-mode-graph scheduling were all
        // accepted and ignored.
        if (params.offload_policy) {
            const auto & policy = *(const std::vector<std::pair<int, int>> *) params.offload_policy;
            for (auto [op, on_off] : policy) {
                if (op < 0 || op >= int(GGML_OP_COUNT)) {
                    LLAMA_LOG_INFO("%s: setting offload policy for all ops to %s\n", __func__, on_off ? "ON" : "OFF");
                } else {
                    LLAMA_LOG_INFO("%s: setting offload policy for op %s to %s\n", __func__,
                                   ggml_op_name(ggml_op(op)), on_off ? "ON" : "OFF");
                }
                ggml_backend_sched_set_op_offload(sched, ggml_op(op), on_off);
            }
        }
        if (params.only_active_experts) {
            LLAMA_LOG_INFO("%s: enabling only_active_experts scheduling\n", __func__);
            ggml_backend_sched_set_only_active_experts(sched, true);
        }
        if (params.prefetch_experts) {
            LLAMA_LOG_INFO("%s: enabling MoE expert read-ahead (prefetch_experts), %s\n", __func__,
                           ggml_backend_prefetch_init(params.prefetch_experts_threads) ? "threaded populate engine" : "madvise fallback");
            for (const auto & mapping : model.mappings) {
                ggml_backend_prefetch_register_mapping(mapping->addr(), mapping->size());
            }
        }
        if (model.split_mode == LLAMA_SPLIT_MODE_GRAPH && (!model.has_tensor_overrides() || cparams.split_mode_graph_scheduling)) {
            ggml_backend_sched_set_split_mode_graph(sched, true, cparams.scheduler_async);
            ggml_backend_sched_set_max_extra_alloc(sched, params.max_extra_alloc);
            if (model.has_tensor_overrides() && cparams.split_mode_graph_scheduling) {
                LLAMA_LOG_INFO("%s: split mode graph scheduling is FORCED despite tensor overrides (user choice); "
                               "not every override combination is supported\n", __func__);
            }
        }

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

// UNION OF TWO DESTRUCTORS, 2026-09-13. WIRE-68 wired this file and deleted 38 duplicate
// definitions from src/llama.cpp, but that list was built from the C API wrapper functions and
// missed llama_context's own MEMBERS. ~llama_context() is one per class, so both bodies collided:
// the linker reported "multiple definition of `llama_context::~llama_context()'" 62 times across
// 31 failing targets, which is why failed link edges went 32 -> 33 across that commit.
//
// THE TWO BODIES WERE NOT DUPLICATES -- they were DISJOINT, and deleting either would have leaked
// or crashed:
//   src/llama.cpp:953 freed the fork's resources: dflash.kv.cache_sched, kv_self.ckpt,
//     free_dflash_kv_cache_tensors(), free_dsv4_cache_tensors(), sched, every backend, buf_output,
//     and erased `this` from the live-context registry. Drop it and all of that leaks, and the
//     registry keeps a dangling pointer that every later walker follows.
//   this file's body did none of that, and did three things llama.cpp's did not: synchronize()
//     before the output buffers are freed, the compute-buffer size expectation check, and
//     ggml_opt_free(opt_ctx). Drop it and a pending asynchronous copy is still in flight when its
//     destination buffer is freed -- a race that appears as corruption, not as an error.
// So the two were merged rather than one being chosen. THE ORDER BELOW IS LOAD-BEARING:
//   1. synchronize() FIRST -- nothing may be freed while an async copy into it is outstanding.
//   2. the size diagnostics read ggml_backend_sched_get_buffer_size(sched, ...), so they must run
//      BEFORE sched is freed.
//   3. only then the fork's frees, in llama.cpp's original order.
//   4. ggml_opt_free last; it is independent of the above.
llama_context::~llama_context() {
    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();
    if (output_subset.ctx || output_subset.buf) {
        clear_output_subset();
    }

    llama_kivi_sim_free(kivi_sim, true);
    kivi_sim = nullptr;

    // S69-KT-EXPERT-POOL-2026-09-28 unit 1: flush the recorder to --moe-expert-stats FNAME.
    // Placed after synchronize() (above) so no record() call from an in-flight callback races
    // this write; before the backend/sched teardown below since write_json only touches
    // moe.recorder's own host-side counts map, not any backend resource.
    if (moe.recorder && !cparams.moe_stats_path.empty()) {
        if (!moe.recorder->write_json(cparams.moe_stats_path)) {
            LLAMA_LOG_WARN("%s: failed to write MoE expert-activation stats to %s\n",
                           __func__, cparams.moe_stats_path.c_str());
        }
    }

    // when training, ggml_opt allocates extra buffers through the scheduler, so the sizes no longer match the expectation
    if (!model.hparams.no_alloc && !opt_ctx) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched, backend);
            if (size_exp == 0) {
                // no construction-time reserve (embedding-only context, see sched_reserve): allocated on demand
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, allocated on demand (no reserve)\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0));
            } else if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }

    // --- merged in from src/llama.cpp:953, verbatim and in its original order ---
    if (dflash.kv.cache_sched != nullptr) {
        ggml_backend_sched_free(dflash.kv.cache_sched);
    }
    kv_self.ckpt.release();
    for (auto & it : kv_self.ckpt_rewind_pool) {
        it->release();
    }
    kv_self.ckpt_rewind_map.clear();
    free_dflash_kv_cache_tensors();
    free_dsv4_cache_tensors();
    ggml_backend_sched_free(sched);
    moe_cache.reset(); // ik #2644: before the backends it holds buffers on are freed

    for (ggml_backend_t backend : backends) {
        ggml_backend_free(backend);
    }

    ggml_backend_buffer_free(buf_output);

    std::lock_guard<std::recursive_mutex> registry_lock(llama_registry_mutex());
    auto & all_contexts = llama_all_contexts();
    for (auto it = all_contexts.begin(); it != all_contexts.end(); ++it) {
        if (*it == this) {
            all_contexts.erase(it);
            break;
        }
    }
    // --- end of the merged body ---

    ggml_opt_free(opt_ctx);
}

void llama_context::resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs) {
    const char * func = __func__;
    auto resolve = [&](const llm_fused_op_probe & probe, bool & enabled) {
        if (!enabled) {
            return;
        }

        uint32_t n_tokens_probe = probe.n_tokens_per_seq*n_seqs;
        if (llm_arch_is_dflash_family(model.arch)) {
            // A DFlash step never carries more than one query block, and its custom K/V holds cross_ctx + that many
            // rows (ensure_dflash_kv_cache_tensors, src/llama-dflash.cpp:64-67); sched_reserve clamps the same way. An
            // unclamped 16-token-per-seq probe on an n_seq_max = 2 draft context (llama-server -np 2) asked for 32 rows and
            // aborted context creation in build_dflash.cpp (GGML_ASSERT k_ctx_cache ne[1] >= n_kv_total), MEASURED
            // 2026-11-05 with z-lab/Qwen3-8B-DFlash-b16 (temp/arex1014/engrev_1027/_dfl_pooltest_logs_after).
            const uint32_t cap = (uint32_t) std::max<int32_t>(std::max<int32_t>(1, (int32_t) model.hparams.dflash_block_size),
                                                              cparams.dflash_query_capacity);
            if (n_seqs > cap) {
                return; // no probe shape fits; the fused op keeps its configured setting
            }
            n_tokens_probe = std::min(n_tokens_probe, (cap / n_seqs) * n_seqs);
        }

        auto * gf = graph_reserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx, true);
        if (!gf) {
            // NOT an error, and previously this threw and aborted the whole context creation.
            // graph_reserve returns nullptr for an OLD-GENERATION model, whose graph is built by
            // llm_build_context::llama_build_graph in src/llama.cpp rather than through
            // llama_model::build_graph (see that function's comment, src/llama-model.cpp:32).
            // With split_only=true -- which is what this probe passes -- that is graph_reserve's
            // ONLY nullptr path, so this cannot be masking an allocation failure.
            //
            // `enabled` is left EXACTLY AS IT WAS rather than cleared. A probe that could not run
            // has learned nothing and must not report "no": these same cparams flags are also
            // consumed by the legacy builder, which is the path these models take, so clearing
            // them here would silently change behaviour that works today on the strength of a test
            // that never executed.
            LLAMA_LOG_WARN("%s: %s could not be probed for this model (its graph is built by the "
                           "llm_build_context path); leaving the setting unchanged\n",
                           func, probe.name);
            return;
        }

        bool device_mismatch = false;
        for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
            if (node.op != probe.op) {
                continue;
            }

            GGML_ASSERT(node.il >= 0);

            ggml_backend_t backend_fused = ggml_backend_sched_get_tensor_backend(sched, node.tensor);
            ggml_backend_dev_t device_fused = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
            // but is still wrong for cases like --no-kv-offload.
            ggml_backend_dev_t device_layer = model.dev_layer(node.il);

            if (device_fused != device_layer) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but %s "
                        "is assigned to device %s (usually due to missing support)\n",
                        func, node.il,
                        device_layer ? ggml_backend_dev_name(device_layer) : "none",
                        probe.name,
                        device_fused ? ggml_backend_dev_name(device_fused) : "none");
                device_mismatch = true;
                break;
            }
        }

        if (device_mismatch) {
            enabled = false;
            LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", func, probe.name);
        } else {
            enabled = true;
            LLAMA_LOG_INFO("%s: %s enabled\n", func, probe.name);
        }
    };

    if (cparams.auto_fa) {
        resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);
        cparams.auto_fa = false;
    }

    // the fallback attends to the whole unified cache with a dense mask
    if (cparams.auto_fkvr) {
        cparams.fused_kv_rows = cparams.flash_attn;
        resolve(llm_fused_op_flash_attn_kv_rows_probe, cparams.fused_kv_rows);
        cparams.auto_fkvr = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", func);
        // LLAMA_FUSED_GDN_AR / LLAMA_FUSED_GDN_CH = 0|1 force the autoregressive / chunked fused Gated Delta Net path on or off
        // instead of probing. Row S69-EXECUTORCH-PICK-CUDA-CHUNKED-GDN-PREFILL-2026-10-05-01 step 1 needs "fused vs graph-chunked"
        // as an A/B on one build and no CLI flag exposed it. GATE-CLAIM-OK: comment text only, nothing declared unavailable.
        const char * env_gdn_ar = getenv("LLAMA_FUSED_GDN_AR");
        const char * env_gdn_ch = getenv("LLAMA_FUSED_GDN_CH");
        if (env_gdn_ar && env_gdn_ar[0]) {
            cparams.fused_gdn_ar = atoi(env_gdn_ar) != 0;
            LLAMA_LOG_INFO("%s: LLAMA_FUSED_GDN_AR=%s forces fused GDN (autoregressive) %s\n", func, env_gdn_ar, cparams.fused_gdn_ar ? "on" : "off");
        } else {
            resolve(llm_fused_op_gdn_ar_probe, cparams.fused_gdn_ar);
        }
        if (env_gdn_ch && env_gdn_ch[0]) {
            cparams.fused_gdn_ch = atoi(env_gdn_ch) != 0;
            LLAMA_LOG_INFO("%s: LLAMA_FUSED_GDN_CH=%s forces fused GDN (chunked) %s\n", func, env_gdn_ch, cparams.fused_gdn_ch ? "on" : "off");
        } else {
            resolve(llm_fused_op_gdn_ch_probe, cparams.fused_gdn_ch);
        }
        cparams.auto_fgdn = false;
    }

    if (cparams.auto_flid) {
        LLAMA_LOG_INFO("%s: resolving fused Lightning Indexer support:\n", func);
        resolve(llm_fused_op_lid_probe, cparams.fused_lid);
        cparams.auto_flid = false;
    }

    if (cparams.auto_fhc) {
        LLAMA_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", func);
        resolve(llm_fused_op_dsv4_hc_pre_probe,  cparams.fused_dsv4_hc_pre);
        resolve(llm_fused_op_dsv4_hc_comb_probe, cparams.fused_dsv4_hc_comb);
        resolve(llm_fused_op_dsv4_hc_post_probe, cparams.fused_dsv4_hc_post);
        cparams.auto_fhc = false;
    }

    if (cparams.auto_ftkm) {
        LLAMA_LOG_INFO("%s: resolving fused top-k mask support:\n", func);
        resolve(llm_fused_op_top_k_mask_probe, cparams.fused_top_k_mask);
        cparams.auto_ftkm = false;
    }
}

static int llama_graph_n_input_tensors(ggml_cgraph * gf) {
    std::unordered_map<const ggml_tensor *, std::vector<ggml_tensor *>> users;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            users[node].push_back(node);
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * src = node->src[j];
            if (!src) {
                break;
            }
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                users[src].push_back(node);
            }
        }
    }

    for (const auto & [tensor, nodes] : users) {
        GGML_ASSERT(tensor->op == GGML_OP_NONE);
        for (const ggml_tensor * node : nodes) {
            LLAMA_LOG_DEBUG("%s: input tensor '%32s' [%s, ne = { %5" PRId64 ", %5" PRId64 ", %5" PRId64 ", %5" PRId64 " }] is used by node '%s' (%s)\n",
                    __func__, tensor->name, ggml_type_name(tensor->type),
                    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                    node->name, ggml_op_name(node->op));
        }
    }

    return (int) users.size();
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);
    if (llm_arch_is_dflash_family(model.arch)) {
        // ik_llama #2345: a DFlash step never carries more than one block of tokens
        n_tokens = std::min(n_tokens, (uint32_t) model.hparams.dflash_block_size);
    }

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    // SIZE THE LEGACY BUILDER'S META ARENA. llm_build_context::init() builds its ggml context
    // directly from this member (src/llama-build-context.cpp:92, :98-99), and NOTHING in src/ ever
    // sized it -- a word-boundary search for the name finds only reads, two declarations, and a
    // DIFFERENT member of the same name on llm_graph_context (src/llama-graph.h:986) which is
    // sized at src/llama-graph.cpp:1340. Two members sharing a name, one initialised, is why this
    // looked correct at every call site. Left empty, ggml_init hands back a context with no memory
    // behind it and the first ggml_new_graph_custom writes past the end.
    //
    // Sized HERE rather than in the legacy reserve branch below because that builder serves every
    // legacy operation, not just the reserve: decode (src/llama.cpp:6895, :7375), defrag (:7685),
    // k-shift (:7720) and s-copy (:7746) all reach it. Fixing only the reserve would leave five
    // callers faulting.
    //
    // The expression is llm_graph_context's own (src/llama-graph.cpp:1340) for the same purpose,
    // over the same max_nodes already used two lines above for the scheduler and both
    // llm_graph_result arenas -- deliberately not a second, independently-chosen number.
    // SIZED BY THE LARGER OF THE TWO NODE COUNTS THIS TREE CARRIES, because the two builders ask
    // for different ones and this single arena feeds both:
    //   graph_max_nodes(n_tokens)      llama-context.cpp:2832 -- the NEW path's count (the local
    //                                  `max_nodes` above), used for the scheduler and both
    //                                  llm_graph_result arenas.
    //   max_nodes(n_tokens, n_kv)      llama.cpp:981 -- the LEGACY count: model.max_nodes(n_tokens)
    //                                  plus OpenPanGu chunked nodes plus the iterative-MLA
    //                                  allowance. This is what llm_build_context::new_graph_custom
    //                                  (llama-build-context.cpp:3803) actually requests.
    // Sizing only by the first left the arena SHORT, and ggml_new_graph_custom does not abort when
    // that happens -- ggml_new_object returns NULL and the caller dereferences it, so an undersized
    // arena presents exactly like an absent one.
    //
    // n_kv at its WORST CASE (cparams.n_ctx): the legacy count grows with n_kv, and this arena is
    // allocated once and reused for the context's whole life. Sizing it for a smaller n_kv would
    // work until the cache filled.
    //
    // `this->` is REQUIRED: the local `max_nodes` above shadows the member function of the same
    // name.
    {
        const size_t meta_nodes = std::max<size_t>(
                max_nodes,
                (size_t) this->max_nodes((int) n_tokens, (int) cparams.n_ctx));

        buf_compute_meta.resize(ggml_tensor_overhead()*meta_nodes +
                                ggml_graph_overhead_custom(meta_nodes, false));
    }

    for (auto & res : gf_res_prev) {
        res.reset();
    }
    gf_res_reserve.reset(new llm_graph_result(max_nodes));
    gf_res_prev_active = nullptr;

    // slice 19: this fork's `sched` is a RAW ggml_backend_sched_t, not upstream's
    // unique_ptr, so .reset(p) becomes free-the-old-then-assign. It is owned --
    // src/llama.cpp:961 frees it in the destructor -- and sched_reserve() is
    // re-enterable (the fork carries a sched_need_reserve flag), so the guard is
    // not decoration.
    if (sched) {
        ggml_backend_sched_free(sched);
    }
    sched = ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload);
    ggml_backend_sched_set_realloc_callback(sched, sched_graph_realloc, this);

    // Up/gate expert repacking -- the third of the three fork features llama.cpp's inline
    // llama_init_from_model carried and this file did not. Placed immediately after the scheduler
    // is created and BEFORE the worst-case graph is reserved, which is where the monolith calls it
    // (llama.cpp:9386, between ggml_backend_sched_new and llama_build_graph). The order matters:
    // repacking rewrites tensor data, so it must happen before the graph that reads those tensors
    // is reserved.
    //
    // The function itself guards on mtp_op_type != MTP_OP_NONE and on whether any layer actually
    // needs repacking, so calling it unconditionally here matches the monolith exactly.
    llama_repack_up_gate_exps(*this);

    // ik 894191d6e (#2644) MoE cache creation, ik src/llama.cpp:9936-9946: after the up/gate repack (the cache binds
    // the layer's final expert tensors, llama-moe-cache.cpp:98-104) and before the worst-case graph, whose MoE
    // layers ask lctx.moe_cache for slot maps. Created once; sched_reserve() can run again (sched_need_reserve), and
    // every scheduler it makes gets the copy callback (here and on the pipeline-parallel retry below, ik :9969).
    // Not under no_alloc: that is the --fit dry run (common/fit.cpp:56), whose weights are not resident, and
    // common_init_result reserves the budget in the fit margin instead.
    if (cparams.moe_cache_size > 0 && !moe_cache && !model.hparams.no_alloc) {
        try {
            moe_cache = std::make_unique<llama_moe_cache>(model, backend_ptrs, backend_buft, cparams.moe_cache_size,
                    cparams.moe_cache_policy, cparams.moe_cache_cpu_share);
        } catch (const std::exception & e) {
            throw std::runtime_error(std::string("failed to initialize MoE cache: ") + e.what());
        }
    }
    if (moe_cache) {
        ggml_backend_sched_set_copy_callback(sched, llama_context::sched_copy_moe_cache, this);
    }

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    resolve_fused_ops(mctx.get(), n_seqs);

    // reserve worst-case graph
    int n_splits_pp        = -1;
    int n_nodes_pp         = -1;
    int n_inputs_pp        = -1;
    int n_input_tensors_pp = -1;

    int n_splits_tg        = -1;
    int n_nodes_tg         = -1;
    int n_inputs_tg        = -1;
    int n_input_tensors_tg = -1;

    const uint32_t n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);

    // OLD-GENERATION MODELS RESERVE THROUGH A DIFFERENT BUILDER, and before this there was no
    // construction-time reserve for them AT ALL. The three blocks below all go through
    // graph_reserve -> model.build_graph(), which returns nullptr for these models by design
    // (src/llama-model.cpp:32), so every one of them failed with "failed to allocate compute pp
    // buffers". A word-boundary search for ggml_backend_sched_reserve across src/ finds three call
    // sites; the only other construction-capable one (llama.cpp:7796) sits inside
    // llama_kv_cache_update_internal behind `if (need_reserve)`, which defrag and k-shift set, so
    // it does not cover construction.
    //
    // The sequence below is DERIVED from llama.cpp:7780-7796, which already does exactly this for
    // the need_reserve case -- worst-case token count, the MTP draft-gen special case, the
    // BOS-token reserve batch, the DeepSeek-V4 input preparation, and the reserve. Kept
    // step-for-step, INCLUDING the DSv4 step, so the two cannot silently diverge.
    //
    // NOTE ON THE PROBES ABOVE, deliberately unchanged: resolve_fused_ops still gets nullptr from
    // graph_reserve for these models and still skips. That is required. Its probe reads
    // get_gf_res_reserve()->get_fused_nodes(), which belongs to the NEW-path result object and is
    // EMPTY for a legacy graph -- an empty list reads as "no device mismatch", so routing the
    // legacy graph through graph_reserve would ENABLE every fusion having examined nothing.
    const bool legacy_reserve = model.uses_legacy_graph_builder();

    // EMBEDDING-ONLY CONTEXTS SKIP THE WORST-CASE RESERVE (spec from KoboldCpp src/llama-context.cpp:653-774,
    // AGPL -- read, not copied). A context with no memory module and embeddings on (BERT-family encoders,
    // create_memory returns nullptr for them) is usually built with n_ubatch == n_ctx (examples/embedding sets
    // n_ubatch = n_batch = n_ctx for non-causal models), so the reserve sizes the n_ctx-token graph -- for
    // attention without FA an n_ctx x n_ctx KQ per head -- while real inputs are a few hundred tokens. With the
    // reserve skipped the scheduler allocates on the first decode, sized to that batch, and grows if a larger
    // one comes. LLAMA_EMBD_SKIP_RESERVE=0 restores the reserve. Generative contexts are untouched.
    // It applies to the no_alloc context of the params-fit step too: the legacy branch below calls
    // ggml_backend_sched_reserve, which ALLOCATES even under no_alloc, and backend_buf_exp_size is only
    // filled when !no_alloc, so that dry run committed the whole worst-case buffer and reported 0 for it
    // (MEASURED: bge-small -c 8192 peak commit 3622 MiB with the real context's reserve skipped).
    // Row S69-ENGINE-REVIEW-1016-EMBED-CTX-RESERVE-SKIP-01.
    bool skip_reserve = !memory && cparams.embeddings;
    if (skip_reserve) {
        const char * env_sr = getenv("LLAMA_EMBD_SKIP_RESERVE");
        if (env_sr != nullptr && atoi(env_sr) == 0) {
            skip_reserve = false;
        }
    }
    if (skip_reserve) {
        LLAMA_LOG_INFO("%s: embedding-only context (no memory module): skipping the construction-time reserve; "
                       "compute buffers are allocated at the first decode (LLAMA_EMBD_SKIP_RESERVE=0 to reserve)\n", __func__);
    }

    // devices that may take a host-weight op off the host (llama_host_kept_token_counts)
    std::vector<ggml_backend_t> offload_devices;
    if (cparams.op_offload) {
        for (ggml_backend_t b : backend_ptrs) {
            if (b != backend_cpu) {
                offload_devices.push_back(b);
            }
        }
    }

    if (legacy_reserve && !skip_reserve) {
        // UNDER no_alloc (the params-fit dry run, common/fit.cpp) the legacy graphs are SIZED, not allocated, the
        // way graph_reserve does it for the new path: ggml_backend_sched_reserve allocates whatever no_alloc says,
        // and the size loop below fills backend_buf_exp_size only when !no_alloc, so the dry run used to commit the
        // whole worst-case compute buffer AND report 0 for it to llama_get_memory_breakdown, which the fit reads.
        // ggml_backend_sched_reserve_size OVERWRITES its sizes (ggml_gallocr_reserve_n_size), and two graphs are
        // reserved here, so each result is folded in with max -- the same "buffer only grows across reserves" rule
        // the allocating path gets from the scheduler. Row S69-ENGINE-REVIEW-1016-FIT-DRYRUN-LEGACY-RESERVE-ALLOCATES-01.
        const bool legacy_size_only = model.hparams.no_alloc;
        if (legacy_size_only) {
            // sched_reserve can run again on the same context; start from zero as graph_reserve's overwrite does
            std::fill(backend_buf_exp_size.begin(), backend_buf_exp_size.end(), 0);
        }
        auto legacy_reserve_graph = [&](ggml_cgraph * g, const char * what) {
            if (legacy_size_only) {
                std::vector<size_t> sizes(backend_buf_exp_size.size(), 0);
                ggml_backend_sched_reserve_size(sched, g, sizes.data());
                for (size_t i = 0; i < sizes.size(); ++i) {
                    backend_buf_exp_size[i] = std::max(backend_buf_exp_size[i], sizes[i]);
                }
            } else if (!ggml_backend_sched_reserve(sched, g)) {
                throw std::runtime_error(std::string("failed to allocate compute buffers for the legacy ") + what);
            }
        };

        int n_tokens_rsv = (int) std::min(cparams.n_ctx, cparams.n_ubatch);

        // MTP draft generation consumes one token and one hidden-state vector per decode step.
        if (cparams.mtp_op_type == MTP_OP_DRAFT_GEN) {
            n_tokens_rsv = 1;
        }

        // A DFlash step carries at most one query block (ik_llama #2345, the clamp the class-path reserve above applies).
        // This legacy branch -- the one DFlash drafts take, their graph is build_dflash in llm_build_context -- had no
        // clamp, so it reserved n_ubatch rows against a custom K/V sized for cross_ctx + one block and aborted context
        // creation: "DFlash K/V capacity 2048/2048 rows < n_kv_total 2560 (n_tokens 512 ...)", MEASURED 2026-11-05 with
        // llama-speculative-simple and llama-server -md z-lab/Qwen3-8B-DFlash-b16. The capacity is the larger of the block
        // and the query capacity (DSV4 DSpark), as in ensure_dflash_kv_cache_tensors.
        if (llm_arch_is_dflash_family(model.arch)) {
            const int token_capacity = std::max<int>(std::max<int>(1, (int) model.hparams.dflash_block_size),
                                                     (int) cparams.dflash_query_capacity);
            n_tokens_rsv = std::min(n_tokens_rsv, token_capacity);
        }

        const int n_past = (int) cparams.n_ctx - n_tokens_rsv;

        // not actually used by llama_build_graph, but required to choose between the token and
        // embedding input graphs -- the same reason llama.cpp:7787 passes it.
        llama_token token = llama_token_bos(&model);
        llama_batch reserve_batch = llama_batch_get_one(&token, n_tokens_rsv);

        if (llm_arch_is_dsv4(model.arch) &&
            !llama_prepare_dsv4_graph_inputs(*this, reserve_batch, false, true)) {
            throw std::runtime_error("failed to prepare DeepSeek-V4 graph inputs for the legacy reserve");
        }

        ggml_cgraph * gf = llm_build_context::llama_build_graph(
                *this, reserve_batch, true, cparams.worst_graph_tokens);
        if (!gf) {
            throw std::runtime_error("failed to build the legacy worst-case graph");
        }

        // BEFORE the reserve: the scheduler's split pass rewrites the src of every op it puts on a device to that
        // device's copy of a host weight (ggml-backend.cpp, ggml_backend_sched_split_graph:
        // `node->src[j] = tensor_id_copy(...)`), and a reserve leaves those copies without a buffer, so a probe run
        // after it saw no host weight on exactly the ops it looks for. MEASURED 2026-10-18 (b1018e pre-run,
        // stories15M -ngl 0): read after the reserve, 143 nodes gave 25 sources with a buffer and 1 weight
        // (the token embedding, never offloaded), and nothing was reserved. See llama_host_kept_token_counts.
        const std::vector<int> host_kept = (llm_arch_is_dsv4(model.arch) || model.arch == LLM_ARCH_OPENPANGU)
                ? std::vector<int>() : llama_host_kept_token_counts(gf, n_tokens_rsv, offload_devices);

        legacy_reserve_graph(gf, "worst-case graph");

        // Read the counters NOW: the graph below is built into the same meta arena (buf_compute_meta) and
        // invalidates `gf`.
        n_splits_pp = n_splits_tg = ggml_backend_sched_get_n_splits(sched);
        n_nodes_pp  = n_nodes_tg  = ggml_graph_n_nodes(gf);

        // The reserve batch above has pos == nullptr, so every "this sequence starts at position 0" branch of the
        // graph builders is skipped: llama-delta-net.cpp:724/735 and llama-kda.cpp:271-386 set
        // reset_state = batch.pos != nullptr && batch.pos[i] == 0, and reset_state adds a ggml_scale(state, 0)
        // node whose state-sized output needs its own scheduler buffer (llama-delta-net.cpp:398). A real first
        // decode takes that branch. MEASURED 2026-10-07 on qwen35-dense: reserve graph 96 nodes / 1339744 B,
        // the first real 99-token decode 97 nodes (the extra node is that SCALE) / 1468384 B, so the compute buffer
        // grew past its expectation and ~llama_context warned. Reserve the position-0 shape too; the scheduler's
        // buffer only grows across reserves. DeepSeek-V4 is left alone: its input preparation above has side
        // effects that must not run twice. openPangu is left alone too: build_openpangu.cpp:879/:965 assert kv_head == batch.pos[0]
        // on every build whose pos is non-null (worst-case builds are exempt only because their pos is null), so a position-0
        // batch aborts its reserve (MEASURED 2026-10-07, full test-llama-archs run, abort at build_openpangu.cpp:966). Its own
        // reset_conv_state (build_openpangu.cpp:254) is therefore not reserved; no local openPangu model exists to measure it.
        //
        // ONE-TOKEN UBATCHES NEED IT TOO. This block used to be gated on n_tokens_rsv > 1, so at -ub 1 (and for
        // MTP draft generation, which forces n_tokens_rsv = 1 above) the position-0 shape was never reserved, while
        // every 1-token decode at position 0 still takes the reset branch -- nothing in llama-delta-net.cpp:724/735
        // or llama-kda.cpp depends on n_tokens. MEASURED 2026-10-18 (Qwen3.5-0.8B-Q8_0, test-save-load-state -b 512
        // -ub 1, CPU and CUDA): the reserve sized CPU 1343712 B, the first decode at pos 0 (1226 nodes = 1208 + the
        // 18 reset SCALEs) needed 2392288 B, +1048576 B = one 128x128x16 f32 delta-net state; 15 teardown warnings,
        // test FAIL. No builder branches on pos for a 1-token batch other than the reset branches listed above.
        //
        // THE PRUNED-OUTPUT SHAPE. Both graphs above are all-outputs (llama_build_graph's worst case gives
        // n_outputs = n_tokens, llama-build-context.cpp:99), but legacy builders create inp_out_ids and the final
        // ggml_get_rows only when n_outputs < n_tokens (e.g. graphs/build_qwen35.cpp: `n_tokens > 1 && n_outputs <
        // n_tokens`). A real prompt ubatch whose last token is the only output therefore carries a host input the
        // reserve never saw. MEASURED 2026-10-18 on CUDA (-ub 8, n_ctx 256): the CUDA_Host compute buffer (graph
        // inputs live there) grew by less than the 0.0001 MiB print step, 8 teardown warnings, test FAIL. Attributing
        // that growth to this I32 [n_outputs] input is SOURCE-DERIVED (it is the one host input whose presence depends
        // on the batch); the CPU build does not reproduce it, since there inputs share one buffer with the activations
        // and fit in their slack -- LLAMA_DEBUG_SCHED_GROWTH=1 names the growing shape on CUDA. Upstream llama.cpp
        // never has this hole: llm_graph_context::build_inp_out_ids always builds the tensor (constant topology,
        // llama.cpp ff5a2337 src/llama-graph.cpp:2774-2781) and build_rs clears state with an in-place scale on a
        // view of the cache (src/llama-graph.cpp:3819-3822), so its reserve (src/llama-context.cpp:770-831) needs
        // neither shape. Here the topology depends on the batch, so the shapes are reserved: n_outputs = n_tokens - 1
        // is the largest pruned input; it is built at position 0 so the reset SCALEs are in the same graph, which
        // matters on a single-buffer (CPU) backend where both land in one allocation.
        if (!llm_arch_is_dsv4(model.arch) && model.arch != LLM_ARCH_OPENPANGU) {
            std::vector<llama_pos> rsv_pos((size_t) n_tokens_rsv, 0);
            llama_batch reserve_batch_pos0 = reserve_batch;
            reserve_batch_pos0.pos = rsv_pos.data();
            ggml_cgraph * gf1 = llm_build_context::llama_build_graph(
                    *this, reserve_batch_pos0, true, cparams.worst_graph_tokens);
            if (!gf1) {
                throw std::runtime_error("failed to build the legacy position-0 worst-case graph");
            }
            legacy_reserve_graph(gf1, "position-0 graph");

            if (n_tokens_rsv > 1) {
                ggml_cgraph * gf2 = llm_build_context::llama_build_graph(
                        *this, reserve_batch_pos0, true, n_tokens_rsv - 1);
                if (!gf2) {
                    throw std::runtime_error("failed to build the legacy pruned-output graph");
                }
                legacy_reserve_graph(gf2, "pruned-output graph");
            }
        }

        // The host-kept shapes (DeepSeek-V4 and openPangu excluded for the reasons given above: input preparation
        // with side effects, and a pos assert). Each is an all-outputs batch, as the worst-case graph is.
        for (const int n_host : host_kept) {
            llama_batch reserve_batch_host = llama_batch_get_one(&token, n_host);
            ggml_cgraph * gfh = llm_build_context::llama_build_graph(*this, reserve_batch_host, true,
                    cparams.worst_graph_tokens > 0 ? std::min(cparams.worst_graph_tokens, n_host) : 0);
            if (!gfh) {
                throw std::runtime_error("failed to build the legacy host-kept graph");
            }
            LLAMA_LOG_INFO("%s: reserving the %d-token graph: below it the device leaves host-weight ops on the host\n",
                           __func__, n_host);
            legacy_reserve_graph(gfh, "host-kept graph");
        }

        // The two counters above describe the FIRST (all-tokens-are-outputs) graph, not a pp/tg pair. They
        // are set equal because they refer to the SAME graph -- not because pp and tg were measured and
        // found to match. The logging below prints a single figure when they are equal.
    }

    // token counts below an offload threshold (llama_host_kept_token_counts), from the pp graph
    std::vector<int> host_kept_pp;

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    if (!legacy_reserve && !skip_reserve) {
        // graph_reserve runs the host-kept probe between building the graph and reserving it (see there)
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr,
                &offload_devices, &host_kept_pp);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                // slice 19: re-creation after a failed reserve, so the OLD scheduler
                // definitely exists here -- a bare assignment would leak it.
                if (sched) {
                    ggml_backend_sched_free(sched);
                }
                sched = ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload);
                ggml_backend_sched_set_realloc_callback(sched, sched_graph_realloc, this);
                if (moe_cache) {
                    ggml_backend_sched_set_copy_callback(sched, llama_context::sched_copy_moe_cache, this);
                }
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), false, nullptr,
                                   &offload_devices, &host_kept_pp);
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp        = ggml_backend_sched_get_n_splits(sched);
        n_nodes_pp         = ggml_graph_n_nodes(gf);
        n_inputs_pp        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_pp = this->n_input_tensors;

        graph_has_logits = get_gf_res_reserve()->get_logits() != nullptr;
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    if (cparams.training && !legacy_reserve && !skip_reserve) {
        // no tg graph for training
        n_splits_tg        = n_splits_pp;
        n_nodes_tg         = n_nodes_pp;
        n_inputs_tg        = n_inputs_pp;
        n_input_tensors_tg = n_input_tensors_pp;
    } else if (!legacy_reserve && !skip_reserve) {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg        = ggml_backend_sched_get_n_splits(sched);
        n_nodes_tg         = ggml_graph_n_nodes(gf);
        n_inputs_tg        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_tg = this->n_input_tensors;
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    if (!legacy_reserve && !skip_reserve) {
        // TODO: not sure if the following graph would be worst case for multi-stream KV caches:
        //
        // auto * gf = graph_reserve(n_tokens, 1, n_tokens, mctx.get());
        //
        // MEASURED 2026-10-07 (nemotron_h, n_seq_max=2, n_ubatch=100): the n_seqs=2 pp graph is 118 nodes and
        // 2.74 MiB, but a real 100-token batch of ONE sequence is 103 nodes and 3.15 MiB (each sequence's slice is
        // 100 tokens instead of 50, so the recurrent intermediates are larger), so the first such decode grew the
        // compute buffer past the expectation and ~llama_context warned. The scheduler's buffer only grows across
        // reserves, so reserving the single-sequence shape first, then the n_seqs shape, covers both.
        for (const int n_host : host_kept_pp) {
            LLAMA_LOG_INFO("%s: reserving the %d-token graph: below it the device leaves host-weight ops on the host\n",
                           __func__, n_host);
            auto * gfh = graph_reserve((uint32_t) n_host, 1, std::min((uint32_t) n_host, n_outputs_pp), mctx.get(),
                                       model.hparams.no_alloc);
            if (!gfh) {
                throw std::runtime_error("failed to allocate compute buffers (host-kept shape)");
            }
        }
        if (n_seqs > 1) {
            auto * gf1 = graph_reserve(n_tokens, 1, n_outputs_pp, mctx.get(), model.hparams.no_alloc);
            if (!gf1) {
                throw std::runtime_error("failed to allocate compute pp buffers (single-sequence shape)");
            }
        }
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched, backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    if (skip_reserve) {
        // no graph was built: nodes, splits and inputs are known only at the first decode
    } else {
        const bool diff = n_nodes_pp != n_nodes_tg || n_splits_pp != n_splits_tg ||
                          n_inputs_pp != n_inputs_tg || n_input_tensors_pp != n_input_tensors_tg;

        const auto val = [diff](int v_pp, int v_tg) -> std::string {
            return diff ? format("%d / %d", v_pp, v_tg) : format("%d", v_pp);
        };

        LLAMA_LOG_INFO("%s: graph%s: nodes = %s, splits = %s, input objects = %s, input tensors = %s\n",
                __func__,
                diff ? format(" (pp bs=%d, tg bs=%d)", n_tokens, n_seqs).c_str() : "",
                val(n_nodes_pp, n_nodes_tg).c_str(),
                val(n_splits_pp, n_splits_tg).c_str(),
                val(n_inputs_pp, n_inputs_tg).c_str(),
                val(n_input_tensors_pp, n_input_tensors_tg).c_str());
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched));
}

// S69-WIRE66-CONTEXT-MEMORY-ARCH-DECISION slice 4. Both calls below were `sched`, which
// assumes the smart-pointer `sched` upstream has. THIS FORK'S MEMBER IS A RAW ggml_backend_sched_t
// (llama-context.h:615), and the fork's own BUILT code already passes it directly --
// src/llama.cpp:1869 and :5389 both call ggml_backend_sched_synchronize(lctx.sched). So this is the
// in-tree idiom, not an invention, and the `if (!sched)` guard here was already written for a raw
// pointer. BOTH sites were changed (here and in the pipeline-parallel branch further down): fixing
// only the one that surfaced first is the one-form blindness this project has hit six times.
void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    ggml_backend_sched_synchronize(sched);   // S69 slice 4: raw member, not a smart pointer -- see note at synchronize()

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

// CAPABILITY-GENUINELY-NA: not a harness-capability edit (no tool-loop/reasoning/pushback/
// thinking/long-context/agentic/persistent-state path touched) -- C++ interface dedupe only.
// RESEARCH-DONE: portable/ik_llama_fork commit 9fea143c4 (this repo, this session) is the read
// precedent -- it added get_cparams()/get_sched() as inline llama-context.h bodies specifically so
// llama-kv-cache.cpp could call them, verified (g++ -fsyntax-only) behaviour-unchanged for every
// existing caller. No behaviour is disabled here: both keep the exact same signature and
// behaviour, callable from every other TU exactly as before -- only the definition site moves from
// this unbuilt .cpp (excluded from CMakeLists, compiled by nothing yet) to the header that IS
// built. Deleted 2026-09-08 (S69-WIRE66-CONTEXT-MEMORY-ARCH-DECISION slice 1): these two
// out-of-line bodies now redefine llama-context.h's inline versions. This body's `sched`
// also assumed sched is a smart-pointer wrapper; the fork's `sched` is the raw
// `ggml_backend_sched_t` the header's inline version already returns, so deleting rather than
// keeping-and-fixing avoids inventing a second truth for the same two members.

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph results to make sure that they won't be reused
        // TODO: make mctx->apply() report if a graph reserve is needed, then reset graph results only if the memory module reset the scheduler
        for (auto & res : gf_res_prev) {
            if (res) {
                res->reset();
            }
        }
        gf_res_prev_active = nullptr;

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        const uint32_t n_outputs_max = std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out;
        return embd + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out;

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend, this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    // the toggle changes the graph shape (pooled: logits only; pooling NONE: the reserve graph outputs all tokens,
    // which crashed ggml_gallocr on the first /completion), so re-reserve on every toggle
    if (cparams.embeddings != value) {
        sched_need_reserve = true;
    }

    cparams.embeddings = value;
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    if (cparams.embeddings_nextn != value || cparams.embeddings_nextn_masked != masked) {
        // these flags change the graph shape
        sched_need_reserve = true;
    }

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer_eff());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    // no scheduler reserve needed because graph shapes must not depend on causal_attn, a flip only rebuilds the graph

    cparams.causal_attn = value;
}

bool llama_context::get_causal_attn() const {
    return cparams.causal_attn;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    // no scheduler reserve needed because warmups are usually with small batches

    cparams.warmup = value;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && model.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft, cparams.n_outputs_max_per_seq);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());
    loras_epoch++;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

void llama_context::set_seq_adapters_lora(llama_seq_id seq_id, llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    std::vector<std::pair<llama_adapter_lora *, float>> list;
    for (size_t i = 0; i < n_adapters; i++) {
        if (scales[i] != 0.0f && adapters[i] != nullptr) {
            list.emplace_back(adapters[i], scales[i]);
        }
    }

    if (list.empty()) {
        seq_loras.by_seq.erase(seq_id);
    } else {
        seq_loras.by_seq[seq_id] = std::move(list);
    }

    // the union decides the graph topology; keep first-seen order for the surviving members so that an unchanged
    // union is an unchanged vector (and therefore the same epoch)
    std::vector<llama_adapter_lora *> uni;
    for (llama_adapter_lora * a : seq_loras.adapters) {
        bool used = false;
        for (const auto & kv : seq_loras.by_seq) {
            for (const auto & p : kv.second) {
                used = used || p.first == a;
            }
        }
        if (used) {
            uni.push_back(a);
        }
    }
    for (const auto & kv : seq_loras.by_seq) {
        for (const auto & p : kv.second) {
            if (std::find(uni.begin(), uni.end(), p.first) == uni.end()) {
                uni.push_back(p.first);
            }
        }
    }

    if (uni != seq_loras.adapters) {
        seq_loras.adapters = std::move(uni);
        seq_loras.epoch++;
        sched_need_reserve = true;
    }
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    // SLICE 23, 2026-09-09. Upstream calls cvec.apply(...), a MEMBER of llama_adapter_cvec that
    // this fork's llama_control_vector does not have. gcc suggests apply_to; that is a RED HERRING
    // and taking it would compile and be wrong -- apply_to is the graph-side helper
    // (ggml_tensor * apply_to(ggml_context *, ggml_tensor *, int il) const) that adds the layer
    // direction during graph build. It shares a prefix with the loader and nothing else.
    //
    // The fork's real equivalent is a FREE function doing exactly upstream's job:
    // src/llama.cpp:9978, declared at include/llama.h:1177. It handles data==nullptr as "disable
    // but keep the allocation", rejects an n_embd mismatch, lazily initialises the tensors on
    // first use, and writes each layer's slice.
    //
    // THE `== 0` IS LOAD-BEARING. The two disagree on polarity:
    //     upstream  bool    apply(...)                  true == success
    //     fork      int32_t llama_control_vector_apply    0   == success, 1 == failure
    // `bool res = llama_control_vector_apply(...)` compiles cleanly and means the OPPOSITE, so
    // every successful load would report failure and every failure would report success -- with
    // nothing crashing to reveal it. Do not "simplify" this comparison away.
    //
    // No recursion: that function reads lctx->model and lctx->cvec directly and never calls back
    // into set_adapter_cvec (measured: 0 occurrences in llama.cpp:9978-10040, against a positive
    // control of 9 occurrences of `cvec` in the same range).
    bool res = llama_control_vector_apply(this, data, len, n_embd, il_start, il_end) == 0;

    sched_need_reserve = true;

    return res;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    // reach-check trace (S69-RPC-REUSE-ROOT-CAUSE-2026-09-28): off by default, set LLAMA_PU_ENTRY_TRACE=1
    static const bool pu_entry_trace = getenv("LLAMA_PU_ENTRY_TRACE") != nullptr;
    if (pu_entry_trace) {
        fprintf(stderr, "[S69-PU-ENTRY] process_ubatch called, n_tokens=%d\n", (int) ubatch.n_tokens);
        fflush(stderr);
    }
    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    auto * res = get_gf_res_prev();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    // S69-RPC-REUSE-ROOT-CAUSE-2026-09-28 round 2: diag (PROCESS_UBATCH_UID_DIAG env-gated) --
    // logs EVERY process_ubatch call, the branch taken, n_tokens, and gf's uid before/after, to
    // settle whether the rebuild branch genuinely runs every decode step when graph_reuse_disable
    // is set (a downstream count, S69-SCHED-UID-DIAG in ggml-backend.cpp, showed only 4
    // split_graph() invocations for a 16-token run where "graphs reused = 0" was also reported --
    // those two facts are in tension and this print is to find out which is wrong / what's between
    // them). SOURCE-READ: src/llama-context.cpp:2199-2247, this whole function, read in full again
    // this entry -- no code copied.
    static int s69_pu_diag_calls = 0;
    const bool s69_pu_diag = getenv("PROCESS_UBATCH_UID_DIAG") != nullptr;
    const bool will_reuse = !graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams);
    if (s69_pu_diag) {
        s69_pu_diag_calls++;
        // LLAMA_LOG_INFO was measured SILENT here (this tool's default log level filters INFO from
        // llama_log_internal even though GGML_LOG_INFO from ggml-backend.cpp -- a different logging
        // path -- was visible in the same run/file). Using fprintf(stderr,...) directly instead:
        // unconditional, no log-level dependency, diagnostic-only.
        fprintf(stderr, "[S69-PU-DIAG] call #%d n_tokens=%d graph_reuse_disable=%d will_reuse=%d "
                       "gf_before_uid=%llu n_reused_before=%d\n",
                       s69_pu_diag_calls, (int) ubatch.n_tokens, (int) graph_reuse_disable,
                       (int) will_reuse, gf ? (unsigned long long) gf->uid : 0ULL, n_reused);
        fflush(stderr);
    }
    if (will_reuse) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched);   // S69 slice 4: raw member, not a smart pointer -- see note at synchronize()
        }

        n_reused++;
    } else {
        gf_res_prev_active = nullptr;
        res->reset();

        ggml_backend_sched_reset(sched);
        ggml_backend_sched_set_eval_callback(sched, cparams.cb_eval, cparams.cb_eval_user_data);

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
        if (s69_pu_diag) {
            fprintf(stderr, "[S69-PU-DIAG] call #%d REBUILT: gf_after_uid=%llu\n",
                           s69_pu_diag_calls, (unsigned long long) gf->uid);
            fflush(stderr);
        }

        gf_res_prev_active = res;
    }

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

// fork: old-generation models run through llama_decode_internal / llama_encode_internal, which take a llama_batch. Upstream's
// merged common code (common_batch -> llama_process) and tests hand every model an llama_batch_ext, so the ext overloads below rebuild
// the llama_batch view for those models -- the inverse of llama_batch_compat::init (llama-batch.cpp) -- instead of refusing them.
struct llama_batch_from_ext {
    std::vector<llama_token>               token;
    std::vector<float>                     embd;
    std::vector<llama_pos>                 pos;
    std::vector<int32_t>                   n_seq_id;
    std::vector<std::vector<llama_seq_id>> seq_ids;
    std::vector<llama_seq_id *>            seq_id_ptr;
    std::vector<int8_t>                    logits;
    std::vector<float>                     state; // MTP hidden-state rows: handed over by llama_set_draft_input_hidden_state_copy
    llama_batch batch = {};
    std::string error;

    // rows [i0, i1) of b (i1 == SIZE_MAX: to the end)
    llama_batch_from_ext(const llama_batch_ext & b, size_t n_pos_per_embd, size_t i0 = 0, size_t i1 = SIZE_MAX) {
        i1 = std::min(i1, b.tokens.size());
        i0 = std::min(i0, i1);
        const size_t n = i1 - i0;
        size_t n_embd_rows = 0, n_state_rows = 0;
        for (size_t i = i0; i < i1; ++i) {
            n_embd_rows  += b.tokens[i].has_embd  ? 1 : 0;
            n_state_rows += b.tokens[i].has_state ? 1 : 0;
        }
        if (n_embd_rows != 0 && n_embd_rows != n) {
            error = "the batch mixes token rows and embedding rows, which a llama_batch cannot carry";
            return;
        }
        if (n_state_rows != 0 && (n_state_rows != n || n_embd_rows != 0)) {
            error = "hidden-state rows must accompany every token row of the batch (legacy MTP hook layout)";
            return;
        }
        const bool   embd_batch = n_embd_rows == n && n > 0;
        const size_t w_embd     = b.n_embd != 0 ? b.n_embd : b.n_embd_inp;
        const size_t n_pos      = embd_batch ? std::max<size_t>(1, n_pos_per_embd) : 1;

        pos.resize(n * n_pos);
        n_seq_id.resize(n);
        seq_ids.resize(n);
        seq_id_ptr.resize(n);
        logits.resize(n);
        if (!embd_batch) {
            token.resize(n);
        }
        for (size_t i = 0; i < n; ++i) {
            const auto & t = b.tokens[i0 + i];
            if (embd_batch) {
                embd.insert(embd.end(), b.embd.begin() + t.embd_off, b.embd.begin() + t.embd_off + w_embd);
                for (size_t j = 0; j < n_pos; ++j) {
                    pos[j * n + i] = t.pos[j]; // section-major, as llama_batch_compat::init reads it
                }
            } else {
                token[i] = t.id;
                pos[i]   = t.pos[0];
                if (t.has_state) {
                    state.insert(state.end(), b.state.begin() + t.state_off, b.state.begin() + t.state_off + b.n_embd_state);
                }
            }
            seq_ids[i].assign(t.seq_ids.begin(), t.seq_ids.end());
            std::sort(seq_ids[i].begin(), seq_ids[i].end());
            n_seq_id[i]   = (int32_t) seq_ids[i].size();
            seq_id_ptr[i] = seq_ids[i].data();
            logits[i]     = t.output ? 1 : 0;
        }
        batch.n_tokens = (int32_t) n;
        batch.token    = token.empty() ? nullptr : token.data();
        batch.embd     = embd.empty()  ? nullptr : embd.data();
        batch.pos      = pos.data();
        batch.n_seq_id = n_seq_id.data();
        batch.seq_id   = seq_id_ptr.data();
        batch.logits   = logits.data();
    }
};

int llama_context::encode(const llama_batch_ext & batch_inp) {
    // old-generation models encode through the legacy builder, which takes a llama_batch
    if (model.uses_legacy_graph_builder()) {
        llama_batch_from_ext lb(batch_inp, model.hparams.n_pos_per_embd());
        if (!lb.error.empty()) {
            LLAMA_LOG_ERROR("%s: legacy-graph model: %s\n", __func__, lb.error.c_str());
            return -1;
        }
        return llama_encode_internal(*this, lb.batch);
    }

    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != hparams.n_embd_inp_enc()) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the encoder input %u\n",
                __func__, batch_inp.n_embd, hparams.n_embd_inp_enc());
        return -1;
    }

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always output all tokens and skip position continuity checks (output_all=true)
    if (!balloc->init(batch_inp, model.vocab, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched, t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched, t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out;

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd_size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
            case LLAMA_POOLING_TYPE_ATTENTION:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out;
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched, t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out;
        GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_tokens*n_embd*sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

template<typename T>
static void copy_tensor_async_rows(
    const std::vector<ggml_tensor *> & tensors,
    const buffer_view<T> & dst,
    size_t stride,
    uint32_t row_offset,
    ggml_backend_sched_t sched,
    std::vector<uint32_t> * counts = nullptr) {
    if (!dst.has_data()) {
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }

        const uint32_t row = row_offset + i;
        const size_t n_elements = ggml_nelements(tensor);
        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampling tensor must be contiguous for async copy");
        GGML_ASSERT(n_elements <= stride);
        GGML_ASSERT((size_t) row * stride + n_elements <= dst.size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        T * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        if (counts) {
            GGML_ASSERT(row < counts->size());
            (*counts)[row] = n_elements;
        }
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch_ext & batch_inp) {
    // old-generation models decode through the legacy builder (llama_decode_internal), which takes a llama_batch
    // (its own splitting, output and KV bookkeeping; return codes pass through unchanged, as in the llama_batch overload)
    if (model.uses_legacy_graph_builder()) {
        // a batch that mixes token rows and embedding rows (upstream: one graph, llama-graph.cpp build_inp_embd embd2): the legacy
        // decoder takes token ids XOR embeddings, so the same-kind runs are decoded in batch order and the outputs laid out as one
        // batch (llama_decode_internal_runs, src/llama.cpp) -- the chunked computation, which is what one causal decode computes
        {
            const size_t n_rows = batch_inp.tokens.size();
            size_t n_embd_rows = 0;
            for (const auto & t : batch_inp.tokens) {
                n_embd_rows += t.has_embd ? 1 : 0;
            }
            if (n_embd_rows != 0 && n_embd_rows != n_rows) {
                std::vector<llama_batch_from_ext> parts;
                parts.reserve(n_rows);
                for (size_t i0 = 0; i0 < n_rows; ) {
                    size_t i1 = i0 + 1;
                    while (i1 < n_rows && batch_inp.tokens[i1].has_embd == batch_inp.tokens[i0].has_embd) {
                        ++i1;
                    }
                    parts.emplace_back(batch_inp, model.hparams.n_pos_per_embd(), i0, i1);
                    if (!parts.back().error.empty()) {
                        LLAMA_LOG_ERROR("%s: legacy-graph model: %s\n", __func__, parts.back().error.c_str());
                        return -1;
                    }
                    i0 = i1;
                }
                std::vector<llama_batch> runs;
                runs.reserve(parts.size());
                for (const auto & p : parts) {
                    runs.push_back(p.batch);
                }
                return llama_decode_internal_runs(*this, runs);
            }
        }
        llama_batch_from_ext lb(batch_inp, model.hparams.n_pos_per_embd());
        if (!lb.error.empty()) {
            LLAMA_LOG_ERROR("%s: legacy-graph model: %s\n", __func__, lb.error.c_str());
            return -1;
        }
        // the legacy MTP / EAGLE3 graphs read the hidden state from the context (llama.cpp: draft_input_hidden_state), not the batch,
        // whose legacy decoder takes token ids XOR embeddings
        if (!lb.state.empty() && !llama_set_draft_input_hidden_state_copy(this, lb.state.data(), lb.state.size())) {
            LLAMA_LOG_ERROR("%s: legacy-graph model: could not hand over %zu hidden-state floats\n", __func__, lb.state.size());
            return -1;
        }
        return llama_decode_internal(*this, lb.batch);
    }

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != batch_inp.n_embd_inp) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the decoder input %zu\n",
                __func__, batch_inp.n_embd, batch_inp.n_embd_inp);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    // non-causal attention needs the whole sequence in one ubatch
    const bool embd_all     = output_all || !cparams.causal_attn;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    // embedding contexts output every token even when no token is explicitly marked as output
    if (has_samplers) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (const auto & tok : batch_inp.tokens) {
            if (!output_all && !tok.output) {
                continue;
            }

            for (auto seq_id : tok.seq_ids) {
                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    continue;
                }

                seq_output_count[seq_id]++;
                auto sampler = sampling.samplers.find(seq_id);
                if (sampler != sampling.samplers.end() &&
                        seq_output_count[seq_id] > (int32_t) cparams.n_outputs_max_per_seq) {
                    LLAMA_LOG_ERROR("%s: backend sampling supports at most %u outputs per sequence "
                            "(seq_id %d had %d)\n", __func__, cparams.n_outputs_max_per_seq,
                            seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    output_swaps.clear();
    embd_batch_idxs.clear();

    sched_reserve();

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, embd_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    // start a new sampling transaction for this logical batch
    for (const auto & entry : sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;

        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits  = res->get_logits();
        auto * t_embd    = cparams.embeddings       ? res->get_embd()     : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn()  : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched, t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits != nullptr);

            float * logits_out = logits + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits_size);
                // A graph that skips the inp_out_ids selection returns one row per TOKEN, not per output. Copying the
                // first n_outputs rows then hands the caller the wrong tokens' logits with no error (cogvlm,
                // 2026-10-18). Pick the flagged rows instead, as the legacy path does (src/llama.cpp, res->ne[1] ==
                // n_tokens branch); any other row count is a graph defect and must not be read as logits.
                if (t_logits->ne[1] == (int64_t) n_outputs) {
                    ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
                } else if (t_logits->ne[1] == (int64_t) ubatch.n_tokens && ubatch.output) {
                    int32_t i_out = 0;
                    for (uint32_t i = 0; i < ubatch.n_tokens && i_out < n_outputs; ++i) {
                        if (ubatch.output[i]) {
                            ggml_backend_tensor_get_async(backend_res, t_logits, logits_out + (int64_t) i_out*n_vocab,
                                (size_t) i*n_vocab*sizeof(float), n_vocab*sizeof(float));
                            ++i_out;
                        }
                    }
                    GGML_ASSERT(i_out == n_outputs);
                } else {
                    GGML_ABORT("%s: logits tensor has %lld rows; the ubatch has %u tokens and %d outputs",
                               __func__, (long long) t_logits->ne[1], ubatch.n_tokens, n_outputs);
                }
            }
        }

        // extract embeddings
        if (embd && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched, t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out;
                        float * embd_out = embd + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd_size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                case LLAMA_POOLING_TYPE_ATTENTION:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        // [TAG_EXTRACT_TARGET_EMBEDDINGS]
        bool extract_all_idxs = extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool masked    = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;

            if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched, t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd  = hparams.n_embd_out;
                float * embd_nextn_out = embd_nextn.data + offset*n_embd;

                GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
                extract_all_idxs = extract_all_idxs || !masked;
            }
        }

        if (extract_all_idxs) {
            GGML_ASSERT(ubatch.data && ubatch.data->batch_idxs.size() == ubatch.n_tokens);
            GGML_ASSERT(embd_batch_idxs.size() == (size_t) n_tokens_prev);
            const auto & batch_idxs = ubatch.data->batch_idxs;
            embd_batch_idxs.insert(embd_batch_idxs.end(), batch_idxs.begin(), batch_idxs.end());
        }

        if (has_samplers) {
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_rows(res->t_sampled,        sampling.sampled,    1,      n_outputs_prev, sched);
            copy_tensor_async_rows(res->t_sampled_logits, sampling.logits,     stride, n_outputs_prev, sched, &sampling.logits_count);
            copy_tensor_async_rows(res->t_sampled_probs,  sampling.probs,      stride, n_outputs_prev, sched, &sampling.probs_count);
            copy_tensor_async_rows(res->t_candidates,     sampling.candidates, stride, n_outputs_prev, sched, &sampling.candidates_count);
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
    } while (mctx->next());

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out;

    bool has_logits     = !cparams.embeddings || graph_has_logits;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;
    size_t embd_layer_inp_float_count = 0;

    logits_size     = has_logits     ? n_vocab*n_outputs_max     : 0;
    embd_size       = has_embd       ? n_embd_out*n_outputs_max  : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out*n_outputs_max  : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output) : 0;
    const size_t new_size  =
        (logits_size + embd_size + embd_nextn.size + embd_layer_inp_float_count + backend_float_count) * sizeof(float) +
        (                                                                         backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            // slice 19: A LEAK NO COMPILER REPORTS. Written against upstream's
            // unique_ptr, where `= nullptr` frees the buffer. This fork's
            // buf_output is a RAW ggml_backend_buffer_t, so the same line is a
            // plain null assignment that drops the only pointer to the old buffer.
            // It is owned: src/llama.cpp:967 frees it in the destructor. This runs
            // on the reallocation path, so it leaked once per output-buffer grow.
            if (buf_output) {
                ggml_backend_buffer_free(buf_output);
            }
            buf_output = nullptr;
            logits = nullptr;
            embd = nullptr;
            embd_nextn.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = {nullptr, 0};
            }
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        // slice 19: raw pointer, so .reset(p) becomes free-then-assign. The guard
        // is a no-op on the path where the block above already freed and nulled
        // it, and load-bearing on the path where it did not run.
        if (buf_output) {
            ggml_backend_buffer_free(buf_output);
        }
        buf_output = ggml_backend_buft_alloc_buffer(buft, new_size);
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output, 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output);

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    // slice 18: this fork stores logits as a RAW POINTER with a separate
    // logits_size (llama-context.h:802-803), where upstream folds both into one
    // buffer_view. logits_size is already set at :2075 from the same expression the
    // discarded buffer_view was reading, so assigning the pointer alone is complete.
    logits = has_logits ? output_base : nullptr;
    offset += logits_size * sizeof(float);

    // slice 18: same as logits above -- raw pointer plus a separate embd_size, set at
    // :2076. The neighbouring embd_nextn line stays a buffer_view because slice 9
    // declared THAT member as one; the two idioms coexist here by design.
    embd = has_embd ? (float *) (base + offset) : nullptr;
    offset += embd_size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
    offset += embd_nextn.size * sizeof(float);

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            embd_layer_inp[il] = buffer_view<float>{(float *) (base + offset), (size_t) n_embd * n_batch};
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{nullptr, 0};
        }
    }

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);

    return n_outputs_max;
}

bool llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    bool extracted = false;
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        GGML_ASSERT(row_floats == model.hparams.n_embd);
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, t);
        GGML_ASSERT(backend != nullptr);
        // Tensor-split backends require a zero source offset.
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
        extracted = true;
    }
    return extracted;
}

void llama_context::output_reorder() {
    const uint64_t n_vocab     = model.vocab.n_tokens();
    const uint64_t n_embd      = model.hparams.n_embd;
    const uint64_t n_embd_out  = model.hparams.n_embd_out;

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits_size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits[i0*n_vocab + k], logits[i1*n_vocab + k]);
            }
        }

        if (embd_size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd[i0*n_embd_out + k], embd[i1*n_embd_out + k]);
            }
        }

        if (embd_nextn.size > 0 && cparams.embeddings_nextn_masked) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd_nextn.data[i0*n_embd_out + k], embd_nextn.data[i1*n_embd_out + k]);
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();

    // [TAG_EXTRACT_TARGET_EMBEDDINGS]
    // Layer inputs and unmasked NextN embeddings contain all token rows, independent of logits selection.
    for (size_t i = 0; i < embd_batch_idxs.size(); ++i) {
        while (embd_batch_idxs[i] != (int32_t) i) {
            const int32_t j = embd_batch_idxs[i];
            GGML_ASSERT(j >= 0 && (size_t) j < embd_batch_idxs.size());
            if (embd_nextn.has_data() && !cparams.embeddings_nextn_masked) {
                for (size_t k = 0; k < n_embd_out; ++k) {
                    std::swap(embd_nextn.data[i*n_embd_out + k], embd_nextn.data[j*n_embd_out + k]);
                }
            }
            for (auto & layer : embd_layer_inp) {
                if (layer.has_data()) {
                    for (size_t k = 0; k < n_embd; ++k) {
                        std::swap(layer.data[i*n_embd + k], layer.data[j*n_embd + k]);
                    }
                }
            }
            std::swap(embd_batch_idxs[i], embd_batch_idxs[j]);
        }
    }
    embd_batch_idxs.clear();
}

//
// graph
//

// defined in src/llama.cpp, next to llama_context::max_nodes
uint32_t llama_delta_net_mixed_seq_nodes(const llama_model & model, const llama_cparams & cparams);

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    uint32_t res;
    // ADDED 2026-09-09, slice 16a. Ported from this fork's own llama_model::max_nodes()
    // (src/llama-model.h:1062-1070), which upstream's version of this function has no equivalent
    // for. Under `-sm graph` -- a fork feature -- the graph spans N devices and needs N times the
    // node budget. Without this the migrated function would compile and silently under-size every
    // graph budget in that mode, with no diagnostic anywhere.
    //
    // The rest of this function is upstream's and is KEPT: it covers 11 architectures to the
    // fork's 3 and accounts for LoRA nodes, neither of which llama_model::max_nodes does. The two
    // implementations still coexist and still disagree elsewhere -- that is recorded as its own
    // finding, not resolved here.
    const uint32_t n_tensors_eff =
        (model.split_mode == LLAMA_SPLIT_MODE_GRAPH && !model.devices.empty())
            ? (uint32_t) (model.n_tensors() * model.devices.size())
            : (uint32_t) model.n_tensors();
    if (model.arch == LLM_ARCH_KIMI_K3 || model.arch == LLM_ARCH_GLM5_NEXT) {
        // the n_tokens*40 budget below is exhausted at ubatch 3840
        res = std::max<uint32_t>(n_tokens * 160, 64u * n_tensors_eff);
    } else if (model.arch == LLM_ARCH_HRM_TEXT) {
        // the 128-slot looped graph needs roughly one stack per token budget
        res = std::max<uint32_t>(n_tokens * 80, 64u * n_tensors_eff);
    } else if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_BAILINGMOE3 ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        llm_arch_is_dsv4(model.arch) ||
        model.arch == LLM_ARCH_CLEF ||
        model.arch == LLM_ARCH_QWEN4EXP ||
        (model.arch == LLM_ARCH_DFLASH && model.hparams.dsv4_hc_mult > 0) ||
        model.arch == LLM_ARCH_NANBEIGE ||
        model.arch == LLM_ARCH_MINIMAX_01 ||
        model.arch == LLM_ARCH_MINIMAX_M3 ||
        // their recurrent block is built once per same-sequence run of the ubatch
        // (llm_build_context::recurrent_runs), so a 4-slot decode builds it 4 times; the default
        // 8*n_tensors budget covered one run and the scheduler asserted
        // (ggml-backend.cpp: hash_set.size >= n_nodes + n_leafs) -- measured 2026-09-26, granite -np 4
        model.arch == LLM_ARCH_LFM2 ||
        model.arch == LLM_ARCH_LFM2MOE ||
        // the recurrent src/models classes: with 16 sequences (llama-perplexity's default) falcon-h1 overran the default budget, ggml_new_object returned NULL and
        // ggml_new_tensor_impl dereferenced it (MEASURED 2026-10-04, lldb: access violation at ggml.c:6077)
        model.arch == LLM_ARCH_MAMBA2 ||
        model.arch == LLM_ARCH_JAMBA ||
        model.arch == LLM_ARCH_FALCON_H1 ||
        model.arch == LLM_ARCH_NEMOTRON_H ||
        model.arch == LLM_ARCH_NEMOTRON_H_MOE ||
        model.arch == LLM_ARCH_PLAMO2 ||
        model.arch == LLM_ARCH_GRANITE_HYBRID ||
        model.arch == LLM_ARCH_HY_V4) {
        res = std::max<uint32_t>(n_tokens * 40, 32u * n_tensors_eff);
    } else if (model.arch == LLM_ARCH_DFLASH && model.hparams.dflash_selector_rank > 0) {
        // DFlash2's convolutions and selector are shape work rather than matmuls,
        // so they cost ~8.6 nodes per tensor against ~5.9 for a plain DFlash draft
        res = std::max<uint32_t>(1024u, 12u*n_tensors_eff);
    } else {
        res = std::max<uint32_t>(1024u, 8u*n_tensors_eff);
        for (const auto & lora : model.loras) {
            res += lora->get_n_nodes();
        }
    }

    // every sampler builds n_outputs_max_per_seq chains, see llm_graph_context::build_sampling
    for (const auto & [seq_id, sampler] : sampling.samplers) {
        res += llama_sampler_backend_n_nodes(sampler) * cparams.n_outputs_max_per_seq;
    }

    // delta-net models build one linear-attention sub-graph per token of a mixed-sequence ubatch (src/llama.cpp,
    // llama_delta_net_mixed_seq_nodes); the scheduler's hash set and the graph results are sized from THIS count, so it must carry
    // the same reserve as llama_context::max_nodes (Thireus c2e3eba2)
    res += llama_delta_net_mixed_seq_nodes(model, cparams);

    if (cparams.training) {
        res *= 4;
    }

    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

llm_graph_result * llama_context::get_gf_res_prev() {
    auto & res = gf_res_prev[n_outputs > 0];
    if (!res) {
        res.reset(new llm_graph_result(gf_res_reserve->get_max_nodes()));
    }
    return res.get();
}

// pack sampler outputs into as few sequences as possible before using sequences without samplers
static void ubatch_prepare_reserve(
              llama_ubatch                            & ubatch,
              uint32_t                                  n_outputs,
        const std::map<llama_seq_id, llama_sampler *> & samplers,
              uint32_t                                  n_outputs_max_per_seq) {
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t t = 0; t < n_seq_tokens; ++t) {
            const uint32_t i = s * n_seq_tokens + t;
            ubatch.n_seq_id[i] = 1;
            ubatch.seq_id[i] = &ubatch.seq_id_unq[s];
        }
    }

    // sequences with a sampler that fit in this ubatch
    std::vector<uint32_t> sampler_seqs;
    std::vector<bool> has_sampler(n_seqs, false);
    for (const auto & entry : samplers) {
        const llama_seq_id seq_id = entry.first;
        if (seq_id < 0 || (uint32_t) seq_id >= n_seqs) {
            continue;
        }

        sampler_seqs.push_back(seq_id);
        has_sampler[seq_id] = true;
    }

    uint32_t n_outputs_set = 0;

    const uint32_t n_outputs_per_seq = std::min(n_seq_tokens, n_outputs_max_per_seq);
    for (uint32_t s : sampler_seqs) {
        if (n_outputs_set >= n_outputs) {
            break;
        }

        for (uint32_t t = 0; t < n_outputs_per_seq && n_outputs_set < n_outputs; ++t) {
            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }

    // use sequences without samplers for any remaining outputs
    for (uint32_t t = 0; t < n_seq_tokens && n_outputs_set < n_outputs; ++t) {
        for (uint32_t s = 0; s < n_seqs && n_outputs_set < n_outputs; ++s) {
            if (has_sampler[s]) {
                continue;
            }

            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes,
        const std::vector<ggml_backend_t> * probe_devices, std::vector<int> * host_kept) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched);

    // when the scheduler is reset, we cannot reuse old graphs, so we reset the previous graph results
    for (auto & res : gf_res_prev) {
        if (res) {
            res->reset();
        }
    }
    gf_res_prev_active = nullptr;

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    ubatch_prepare_reserve(ubatch, n_outputs, sampling.samplers, cparams.n_outputs_max_per_seq);

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));

    res->reset();

    auto * gf = model.build_graph(gparams);

    // check the graph before scheduling: cross-backend copies break the producer chain that the check follows
    if (!hadamard_verified && gf && (!model.hdmd.rot.empty() || !model.hdmd.inv.empty())) {
        llama_verify_hadamard_graph(gf, model.hdmd.rot, model.hdmd.inv, moe_cache.get());
        hadamard_verified = true;
    }

    // gf is nullptr BY DESIGN for an llm_build_context model (see below): it has no inputs to count here
    this->n_input_tensors = gf ? llama_graph_n_input_tensors(gf) : 0;
    this->n_outputs = save_n_outputs;

    // build_graph() RETURNS NULLPTR BY DESIGN for an old-generation model, whose graph is built by
    // llm_build_context::llama_build_graph in src/llama.cpp instead (see the comment on
    // llama_model::build_graph, src/llama-model.cpp:32). That comment states llama-context.cpp
    // "tests `if (!gf)` on the next line" -- true at llama-context.cpp:1784, and NOT true here.
    // The result went straight into ggml_backend_sched_split_graph, which dereferences it at
    // ggml-backend.cpp:1164 (`graph->uid = ...`) and segfaulted on every load of such a model.
    //
    // Returning nullptr is NOT a new contract: every caller of graph_reserve already tests for it
    // (llama-context.cpp:878, 1018, 1030, 1042, 1057, 1232). This restores the behaviour they were
    // written against.
    if (!gf) {
        LLAMA_LOG_DEBUG("%s: build_graph returned no graph -- this model's graph is built by the "
                        "llm_build_context path, so there is nothing to reserve here\n", __func__);
        return nullptr;
    }

    // The host-kept probe must read the graph as built: the split pass below rewrites each device op's host-weight
    // src to the device's copy (ggml-backend.cpp, `node->src[j] = tensor_id_copy(...)`). See the legacy reserve.
    if (host_kept != nullptr && probe_devices != nullptr) {
        *host_kept = llama_host_kept_token_counts(gf, (int) n_tokens, *probe_devices);
    }

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched, gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched, gf);
        }
    } else if (!ggml_backend_sched_reserve(sched, gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    return {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched,
        /*.backend_cpu =*/ backend_cpu,
        // SLICE 26: llm_graph_params::cvec is a POINTER, and llama_context holds the control
        // vector BY VALUE (llama-adapter.h since this slice), so the address is what goes in.
        /*.cvec        =*/ &cvec,
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.moe_cache   =*/ moe_cache.get(),
        /*.prec_policy =*/ &model.prec_policy,
        /*.hdmd        =*/ model.hdmd.rot.empty() ? nullptr : &model.hdmd,
        /*.samplers    =*/ sampling.samplers,
        // explicit: a non-negative count; MSVC rejects the implicit int32_t -> uint32_t narrowing in a
        // braced initializer (C2397) where MinGW gcc only warns, so the MinGW build never showed it
        /*.n_outputs   =*/ (uint32_t) n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
    };
}

void llama_context::sched_graph_realloc(ggml_backend_sched_t sched, ggml_cgraph * gf, bool backend_ids_changed, bool graph_size_changed, void * user_data) {
    // reallocating after the backend assignments changed is expected, trace only the unexpected ones
    if (backend_ids_changed) {
        return;
    }

    auto * lctx = static_cast<llama_context *>(user_data);

    if (graph_size_changed) {
        lctx->n_reallocs_diff++;
    } else {
        lctx->n_reallocs_same++;
    }

    LLAMA_LOG_INFO("%s: graph reallocation: nodes = %d (changed = %d), splits = %d, n_same = %d, n_diff = %d\n",
            __func__, ggml_graph_n_nodes(gf), graph_size_changed, ggml_backend_sched_get_n_splits(sched),
            lctx->n_reallocs_same, lctx->n_reallocs_diff);
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    copy_experts.reset();

    auto status = ggml_backend_sched_graph_compute_async(sched, gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers > model.hparams.n_layer;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend) == dev_layer) {
                        if (ggml_backend_supports_op(backend, cur)) {
                            ggml_backend_sched_set_tensor_backend(sched, cur, backend);
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), base_ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        if (winfos.empty()) {
            return;
        }

        std::vector<ggml_tensor *> tensors;
        std::vector<size_t> offsets;
        std::vector<size_t> sizes;
        std::vector<size_t> data_offsets;
        tensors.reserve(winfos.size());
        offsets.reserve(winfos.size());
        sizes.reserve(winfos.size());
        data_offsets.reserve(winfos.size());

        for (const auto & winfo : winfos) {
            tensors.push_back(winfo.tensor);
            offsets.push_back(winfo.offset);
            sizes.push_back(winfo.size);
            data_offsets.push_back(winfo.data_offset);
        }

        std::vector<const ggml_tensor *> const_tensors(tensors.begin(), tensors.end());
        ggml_backend_tensor_get_n(const_tensors.data(), base_ptr, offsets.data(), sizes.data(), data_offsets.data(), winfos.size());
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, (size_t) (ptr - base_ptr), size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    uint8_t * base_ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        size_t data_offset;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), base_ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        if (rinfos.empty()) {
            return;
        }

        std::vector<ggml_tensor *> tensors;
        std::vector<size_t> offsets;
        std::vector<size_t> sizes;
        std::vector<size_t> data_offsets;
        tensors.reserve(rinfos.size());
        offsets.reserve(rinfos.size());
        sizes.reserve(rinfos.size());
        data_offsets.reserve(rinfos.size());

        for (const auto & rinfo : rinfos) {
            tensors.push_back(rinfo.tensor);
            offsets.push_back(rinfo.offset);
            sizes.push_back(rinfo.size);
            data_offsets.push_back(rinfo.data_offset);
        }

        ggml_backend_tensor_set_n(tensors.data(), base_ptr, offsets.data(), sizes.data(), data_offsets.data(), rinfos.size());
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, (size_t) (ptr - base_ptr), size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void discard() override {
        rinfos.clear();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    const uint8_t * base_ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        size_t data_offset;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;


        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            GGML_ASSERT(winfo.size % ggml_type_size(winfo.tensor->type) == 0);
            const int64_t n = winfo.size/ggml_type_size(winfo.tensor->type)*ggml_blck_size(winfo.tensor->type);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            std::vector<const ggml_tensor *> srcs(mbuf_cur.org.begin(), mbuf_cur.org.end());
            std::vector<ggml_tensor *> dsts(mbuf_cur.cpy.begin(), mbuf_cur.cpy.end());
            ggml_backend_tensor_copy_n(srcs.data(), dsts.data(), mbuf_cur.org.size());
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            GGML_ASSERT(rinfo.size % ggml_type_size(rinfo.tensor->type) == 0);
            const int64_t n = rinfo.size/ggml_type_size(rinfo.tensor->type)*ggml_blck_size(rinfo.tensor->type);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            if (mbuf_cur.n_tensors == mbuf.n_tensors) {
                // an equal tensor count does not imply the same chunking, e.g. save ranges [2,1] vs restore runs [1,2]
                bool same_chunking = true;
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    if (ggml_nbytes(mbuf_cur.cpy[i]) != ggml_nbytes(mbuf.org[i])) {
                        same_chunking = false;
                        break;
                    }
                }

                if (same_chunking) {
                    // same chunking: copy 1:1 by index
                    std::vector<const ggml_tensor *> srcs(mbuf_cur.cpy.begin(), mbuf_cur.cpy.end());
                    std::vector<ggml_tensor *> dsts(mbuf.org.begin(), mbuf.org.end());
                    ggml_backend_tensor_copy_n(srcs.data(), dsts.data(), mbuf_cur.cpy.size());
                    continue;
                }
            }

            // different chunking: copy the write-side data (mbuf_cur.cpy) into the read-side targets (mbuf.org)
            // with a byte cursor. Write and read enumerate the same logical data in the same order but may chunk
            // it differently (even with an equal number of tensors), so copy across tensor boundaries rather than
            // 1:1 by index.
            const size_t total = mbuf_cur.total_size;

            ggml_init_params params_scratch = {
                /*.mem_size   =*/ 2*(mbuf_cur.cpy.size() + mbuf.org.size())*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx_scratch = ggml_init(params_scratch);

            size_t src_pos  = 0;
            size_t dst_pos  = 0;
            size_t src_j    = 0;
            size_t dst_i    = 0;
            size_t src_base = 0;
            size_t dst_base = 0;

            std::vector<const ggml_tensor *> chunk_srcs;
            std::vector<ggml_tensor *> chunk_dsts;
            chunk_srcs.reserve(mbuf_cur.cpy.size() + mbuf.org.size());
            chunk_dsts.reserve(mbuf_cur.cpy.size() + mbuf.org.size());

            while (src_pos < total) {
                const auto & src_t = mbuf_cur.cpy[src_j];
                const auto & dst_t = mbuf.org[dst_i];

                const size_t src_size = ggml_nbytes(src_t);
                const size_t dst_size = ggml_nbytes(dst_t);

                const size_t src_off  = src_pos - src_base;
                const size_t dst_off  = dst_pos - dst_base;

                const size_t n_copy = std::min(src_size - src_off, dst_size - dst_off);

                GGML_ASSERT(n_copy % ggml_type_size(src_t->type) == 0);
                const int64_t n_el = n_copy / ggml_type_size(src_t->type) * ggml_blck_size(src_t->type);

                auto * src_v = ggml_view_1d(ctx_scratch, src_t, n_el, src_off);
                ggml_backend_view_init(src_v);
                auto * dst_v = ggml_view_1d(ctx_scratch, dst_t, n_el, dst_off);
                ggml_backend_view_init(dst_v);

                chunk_srcs.push_back(src_v);
                chunk_dsts.push_back(dst_v);

                src_pos += n_copy;
                dst_pos += n_copy;

                if (src_pos - src_base == src_size) {
                    src_base = src_pos;
                    ++src_j;
                }
                if (dst_pos - dst_base == dst_size) {
                    dst_base = dst_pos;
                    ++dst_i;
                }
            }

            GGML_ASSERT(src_pos == total && dst_pos == total);
            // any tensors left unvisited hold no data
            for (size_t i = src_j; i < mbuf_cur.cpy.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf_cur.cpy[i]) == 0);
            }
            for (size_t i = dst_i; i < mbuf.org.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf.org[i]) == 0);
            }

            ggml_backend_tensor_copy_n(chunk_srcs.data(), chunk_dsts.data(), chunk_srcs.size());

            ggml_free(ctx_scratch);
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    void discard() override {
        rinfos.clear();
        buf_size = 0;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io.discard();
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io->discard();
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (tokens_out == nullptr) {
            const size_t n_token_max = (file.size() - file.tell()) / sizeof(llama_token);
            if (n_token_count > n_token_max) {
                LLAMA_LOG_ERROR("%s: token count in sequence state file exceeds the file size! %u > %zu\n", __func__, n_token_count, n_token_max);
                return 0;
            }

            *n_token_count_out = n_token_count;
            return file.tell();
        }

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llama_model_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llama_model_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    data.n_reallocs_same = n_reallocs_same;
    data.n_reallocs_diff = n_reallocs_diff;

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;

    // note: the reallocation counters are not reset - they track the lifetime totals for this context
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    // The legacy llm_build_context KV cache (kv_self) is a SECOND cache next to `memory` whenever
    // model.uses_legacy_graph_builder() -- see the note where it is initialised. It is a real
    // allocation in a normal run, so a breakdown without it under-reports context by its full size
    // and fit sizes against a figure smaller than what gets allocated. buft_sizes holds the real
    // size normally and the projected size under no_alloc.
    for (const auto & [buft, size] : kv_self.buft_sizes) {
        ret[buft].context += size;
    }
    if (moe_cache) {
        for (const auto & [buft, size] : moe_cache->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i];
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched, backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr;
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched, backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched, backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    if (cparams.flash_attn) {
        LLAMA_LOG_INFO("%s: disabling flash attention, FLASH_ATTN_EXT has no backward pass\n", __func__);
        cparams.flash_attn = false;
    }

    // gradients cannot flow through the KV cache, so the attention reads the K and V of the current ubatch directly
    if (n_ubatch == cparams.n_ctx) {
        cparams.training = true;
    } else {
        LLAMA_LOG_WARN("%s: n_ubatch (%u) != n_ctx (%u), the K and V projections will not receive gradients\n", __func__, n_ubatch, cparams.n_ctx);
    }

    // the training graph is different, need to reserve again
    sched_need_reserve = true;
    sched_reserve();

    ggml_opt_params opt_params = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        // TODO: use llama_batch_ext here
        {
            llama_batch_compat compat(this, batch);
            if (!balloc->init(*compat.batch_ext, model.vocab, true)) {
                LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
                return;
            }
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = get_gf_res_prev();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            // the optimizer graph is allocated outside sched, so the next decode must rebuild
            gf_res_prev_active = nullptr;
            res->reset();

            auto * gf = model.build_graph(gparams);

            // Same documented-nullptr contract as in graph_reserve above, and the same omission:
            // ggml_graph_size(gf) on the next line dereferences it. This path is TRAINING rather
            // than inference, so the fault would have surfaced later and looked unrelated.
            // opt_epoch_iter returns void, so this reports and returns.
            if (!gf) {
                LLAMA_LOG_ERROR("%s: build_graph returned no graph -- optimization cannot run for "
                                "this model\n", __func__);
                return;
            }

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            copy_experts.reset();
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    // REBUILT 2026-09-09, slice 12. This function previously used a POSITIONAL brace
    // initialiser with 37 entries against a 94-member struct, in upstream's member order
    // rather than this fork's, so every value landed in the wrong member and the rest were
    // value-initialised in silence. The compiler emitted exactly ONE diagnostic for all of
    // it -- a narrowing -- because that was the only shifted pair whose types disagreed.
    //
    // It is now PER-FIELD ASSIGNMENT, which binds by NAME. That is deliberate and is what
    // row S69-LLAMA-CONTEXT-PARAMS-POSITIONAL-INITIALISER-SHIFTED-37-OF-94 asked for:
    // adding the missing positional entries would have re-created the identical silent
    // class the next time either struct gains a member. This form cannot shift.
    //
    // Every value is sourced BY NAME from src/llama.cpp's copy of this same function (the
    // one the shipped build uses, repaired on 2026-09-08 and provenance-tagged), so the
    // two definitions AGREE instead of drifting. src/llama.cpp and this file define 39 of
    // the same symbols -- see row S69-LLAMA-CPP-AND-LLAMA-CONTEXT-CPP-DEFINE-39-SAME-
    // SYMBOLS-01 -- and reconciling that duplication is integration work, not this slice.
    llama_context_params result = {};   // value-init first, then set every member by name

    result.seed                        = LLAMA_DEFAULT_SEED;
    result.n_ctx                       = 512;
    result.n_batch                     = 2048;
    result.n_ubatch                    = 512;
    result.n_seq_max                   = 1;
    result.n_threads                   = GGML_DEFAULT_N_THREADS;
    result.n_threads_batch             = GGML_DEFAULT_N_THREADS;
    result.max_extra_alloc             = 256;
    result.worst_case_tokens           = 0;
    result.n_rs_seq                    = 0;
    result.n_outputs_max               = 0;
    result.n_outputs_max_per_seq       = 1;
    result.ctx_type                    = LLAMA_CONTEXT_TYPE_DEFAULT;
    result.flash_attn_type             = LLAMA_FLASH_ATTN_TYPE_AUTO;
    result.rope_scaling_type           = LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
    result.pooling_type                = LLAMA_POOLING_TYPE_UNSPECIFIED;
    result.attention_type              = LLAMA_ATTENTION_TYPE_UNSPECIFIED;
    result.rope_freq_base              = 0.0f;
    result.rope_freq_scale             = 0.0f;
    result.yarn_ext_factor             = -1.0f;
    result.yarn_attn_factor            = -1.0f;
    result.yarn_beta_fast              = -1.0f;
    result.yarn_beta_slow              = -1.0f;
    result.yarn_orig_ctx               = 0;
    result.defrag_thold                = -1.0f;
    result.cb_eval                     = nullptr;
    result.cb_eval_user_data           = nullptr;
    result.type_k                      = GGML_TYPE_F16;
    result.type_v                      = GGML_TYPE_F16;
    result.idx_type_k                  = GGML_TYPE_F16;
    result.type_reduce                 = GGML_TYPE_F16;
    result.type_graph_attn             = GGML_TYPE_F16;
    result.type_k_first                = GGML_TYPE_F16;
    result.type_k_last                 = GGML_TYPE_F16;
    result.type_v_first                = GGML_TYPE_F16;
    result.type_v_last                 = GGML_TYPE_F16;
    result.n_k_first                   = -1;
    result.n_k_last                    = -1;
    result.n_v_first                   = -1;
    result.n_v_last                    = -1;
    result.logits_all                  = false;
    result.embeddings                  = false;
    result.offload_kqv                 = true;
    result.flash_attn                  = true;
    result.mla_attn                    = 3;
    result.attn_max_batch              = 256;
    result.fused_moe_up_gate           = true;
    result.grouped_expert_routing      = false;
    result.fused_up_gate               = true;
    result.fused_mmad                  = true;
    result.rope_cache                  = false;
    result.graph_reuse                 = true;
    result.dsa                         = false;
    result.fused_idx_topk              = true;
    result.swa_compress                = false;
    result.dsv4_legacy_state           = false;
    result.dsa_top_k                   = -1;
    result.min_experts                 = -1;
    result.thresh_experts              = 0.0f;
    result.only_active_experts         = false;
    result.prefetch_experts            = false;
    result.prefetch_experts_threads    = 0;
    result.k_cache_hadamard            = false;
    result.v_cache_hadamard            = false;
    result.kv_fresh_chunk              = 0;
    result.split_mode_graph_scheduling = false;
    result.scheduler_async             = false;
    result.mtp                         = false;
    result.mtp_op_type                 = MTP_OP_NONE;
    result.abort_callback              = nullptr;
    result.abort_callback_data         = nullptr;
    result.offload_policy              = nullptr;
    result.cuda_params                 = nullptr;
    result.dflash_query_capacity       = 0;   // ik_llama #2384 (appended last in the struct there)
    result.snapkv_cap                  = 0;
    result.snapkv_window               = 0;
    result.h2o_heavy                   = 0;
    result.h2o_recent                  = 0;
    result.kvflash_pool_tokens         = 0;
    result.offload_attn                = false;
    result.offload_attn_page           = 0;
    result.offload_attn_budget         = 0.0f;
    result.offload_attn_selector       = 0;
    result.offload_attn_sink           = 0;
    result.offload_attn_recent         = 0;
    result.offload_attn_dense_layers   = 0;
    result.offload_attn_head_reduce    = -1;
    result.offload_attn_dense_spec[0]  = 0;
    result.offload_attn_refresh        = 1;
    result.offload_attn_scan           = 0;
    result.offload_attn_min_tokens     = 0;
    result.tri_stats_path              = nullptr;
    result.tri_budget                  = 0.0f;
    result.tri_window                  = 0;
    result.tri_interval                = 0;
    result.tri_sink                    = 0;
    result.cats_sparsity               = 0.0f;
    result.cats_thresholds_file        = nullptr;
    result.cats_calibrate              = false;
    result.teal_sparsity               = 0.0f;
    result.teal_thresholds_file        = nullptr;
    result.teal_calibrate              = false;
    result.no_perf                     = true;
    result.op_offload                  = true;
    result.swa_full                    = true;
    result.kv_unified                  = false;
    result.samplers                    = nullptr;
    result.n_samplers                  = 0;
    result.ctx_other                   = nullptr;

    // MoE resident expert pool (S69-KT-EXPERT-POOL-2026-09-28) -- off by default.
    result.moe_pool_slots              = 0;
    result.moe_pool_inserts            = 2;
    result.moe_pool_admit              = 3;
    result.moe_pool_cpu_only           = false;
    result.moe_stats_path              = nullptr;
    result.moe_stats_mode              = "stat";
    result.moe_placement_strategy      = "uniform";
    result.moe_placement_stats         = nullptr;
    result.moe_dynamic_update          = false;
    result.moe_dynamic_threshold       = 512;
    result.ffd                         = false;
    result.ffd_block                   = 128;
    result.ffd_delta                   = 5.0f; // upstream default (ffd_core/modeling/decode.py:24)
    result.kivi_bits                   = 0;
    result.kivi_group                  = 32;
    result.kivi_residual               = 128;
    result.xattn_threshold             = 0.0f;
    result.xattn_stride                = 8;
    result.xattn_block                 = 128;
    result.kv_diversity_mass           = 0.0f;
    result.kv_diversity_block          = 16;
    result.kv_diversity_layers         = 4;
    result.snapkv_rebase               = 0;

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
        if (model->get_split_state_ud.n_devices == 1) {
            LLAMA_LOG_WARN("%s: SPLIT_MODE_TENSOR being used for a single device is not recommended\n", __func__);
        }
    }

    // A KV-cache type with per-row metadata (ggml type traits row_meta_size > 0, e.g. q8_KV: one 8-byte block per row) cannot back
    // the cache: the per-head sub-row views (llama-build-context.cpp, llama.cpp K/V views) are sized ggml_row_size(type, n_embd_head),
    // which repeats the metadata for every head, so the views run past the tensor (V: GGML_ASSERT data_size + view_offs <=
    // ggml_nbytes(view_src) at graph build; K: SIGSEGV in quantize_row_q8_KV at the first decode). Refuse it here, by name, instead of
    // letting a server come up and die on the first request. Scope: a guard only -- it does not make q8_KV work (that needs the view
    // arithmetic redesigned). Ported from Thireus/ik_llama.cpp f228595b (fix/refuse-row-meta-kv-cache-type, MIT, src/llama.cpp
    // @8430-8450), extended over the per-layer first/last types this fork has and that the branch's tree predates.
    {
        struct kv_type_ref { const char * what; ggml_type type; bool used; };
        const kv_type_ref kv_types_used[] = {
            { "K cache (type_k)",        params.type_k,       true },
            { "V cache (type_v)",        params.type_v,       true },
            { "K cache (type_k_first)",  params.type_k_first, params.n_k_first > 0 },
            { "K cache (type_k_last)",   params.type_k_last,  params.n_k_last  > 0 },
            { "V cache (type_v_first)",  params.type_v_first, params.n_v_first > 0 },
            { "V cache (type_v_last)",   params.type_v_last,  params.n_v_last  > 0 },
        };
        for (const auto & e : kv_types_used) {
            if (!e.used) {
                continue;
            }
            if (ggml_internal_get_type_traits(e.type).row_meta_size > 0) {
                LLAMA_LOG_ERROR("%s: %s cannot be used as a K/V-cache type (%s): it stores per-row metadata that the per-head cache views "
                                "cannot address; use f16, bf16, q8_0, q6_0, q5_0, q5_1, q4_0, q4_1 or iq4_nl\n",
                                __func__, ggml_type_name(e.type), e.what);
                return nullptr;
            }
        }
    }

    // mainline 202c1a229: direct cache consumers (MLA / DSA / DSV4 / MiniMax-M3 / Qwen4exp / DFlash ...) read the cache without the
    // shared attention helpers, so an F8_E4M3 cache is refused for them (mainline also builds per-layer scales there; not ported here)
    if ((params.type_k == GGML_TYPE_F8_E4M3 || params.type_v == GGML_TYPE_F8_E4M3) && (model->hparams.is_mla() ||
            model->arch == LLM_ARCH_DEEPSEEK32 || llm_arch_is_dsv4(model->arch) || model->arch == LLM_ARCH_GLM_DSA ||
            model->arch == LLM_ARCH_GLM5_NEXT || model->arch == LLM_ARCH_DOTS3NOTE || model->arch == LLM_ARCH_HY_V4 ||
            model->arch == LLM_ARCH_MINIMAX_M3 || model->arch == LLM_ARCH_QWEN4EXP || model->arch == LLM_ARCH_DFLASH)) {
        LLAMA_LOG_ERROR("%s: FP8 cache refused for architecture '%s' (reads the cache directly)\n", __func__, llm_arch_name(model->arch));
        return nullptr;
    }

    if ((model->hparams.is_mla() || llm_arch_is_dsv4(model->arch)) && params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("%s: model does not support different K (%s) and V (%s) cache types\n", __func__, ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        return nullptr;
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            LLAMA_LOG_ERROR("%s: quantized V cache requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer_eff(); ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    // TurboQuant KV (turbo2/turbo3/turbo4) needs flash attention. On a GPU the attention is the backend's own FA wrapper (CUDA vec kernels,
    // Vulkan, Metal, SYCL: they rotate Q and the output themselves). A KV cache in host memory is attended by the CPU backend: its generic FA
    // kernel reads turbo K through the f32 vec_dot of ggml-cpu.c and turbo V through to_float, and the graph rotates (src/llama-turbo-graph.h,
    // selected per layer by whether the cache buffer is host memory). Until 2026-10-07 this refused any layer whose KV was on the CPU, because
    // such a layer aborted MID-REQUEST in ops.cpp dup_to_q ("not implemented") -- MEASURED 2026-09-29 on a CPU-only server.
    {
        auto is_turbo = [](ggml_type t) {
            return t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO4_0;
        };
        const bool k_turbo = is_turbo(params.type_k);
        const bool v_turbo = is_turbo(params.type_v);
        // turbo4 (the 4-bit layout) has a CUDA cache writer (cpy.cu / set-rows.cu, turbo-write.h NBITS=4) and a fused
        // attention read path (fattn-turbo.cuh) since 2026-10-05; it was refused here while its CPU inverse rotation did
        // not invert its forward one (task #278, retired by dd765fbf0).
        if (k_turbo || v_turbo) {
            if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
                LLAMA_LOG_ERROR("%s: TurboQuant KV cache (%s/%s) needs flash attention (-fa on)\n", __func__,
                                ggml_type_name(params.type_k), ggml_type_name(params.type_v));
                return nullptr;
            }
            // The fused flash-attention kernels are instantiated only for K/V in {turbo2, turbo3, turbo4, q8_0, f16} (one
            // side at least turbo) with n_embd_head_k == n_embd_head_v in {128, 256} (fattn-vec-f16.cu / fattn-vec-f32.cu
            // turbo cases, mirrored by ggml_cuda_fattn_vec_*_is_supported: 21 (K,V) pairs). Any other pairing has no
            // attention op on the GPU and would abort in the scheduler on the first token, so name it here.
            auto turbo_side_ok = [](ggml_type t) {
                return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0 ||
                       t == GGML_TYPE_Q8_0 || t == GGML_TYPE_F16;
            };
            if (!turbo_side_ok(params.type_k) || !turbo_side_ok(params.type_v)) {
                LLAMA_LOG_ERROR("%s: TurboQuant KV cache (%s/%s): the other side must be turbo2, turbo3, turbo4, q8_0 or f16 "
                                "(no flash-attention kernel is instantiated for a turbo cache paired with %s)\n", __func__,
                                ggml_type_name(params.type_k), ggml_type_name(params.type_v),
                                turbo_side_ok(params.type_k) ? ggml_type_name(params.type_v) : ggml_type_name(params.type_k));
                return nullptr;
            }
#if defined(GGML_USE_METAL)
            // The Metal path decodes K and V with one pre-pass kernel per type (ggml_metal_op_flash_attn_ext asserts
            // K->type == V->type), so a mixed pair has no attention op there.
            if (params.type_k != params.type_v) {
                LLAMA_LOG_ERROR("%s: TurboQuant KV cache (%s/%s): the Metal backend needs the same type for K and V\n", __func__,
                                ggml_type_name(params.type_k), ggml_type_name(params.type_v));
                return nullptr;
            }
#endif
            for (uint32_t il = 0; il < model->hparams.n_layer_eff(); ++il) {
                if (model->hparams.n_head_kv(il) == 0) {
                    continue;   // no KV cache in this layer (recurrent / non-attention)
                }
                const uint32_t dk = model->hparams.n_embd_head_k(il);
                const uint32_t dv = model->hparams.n_embd_head_v(il);
                if (dk != dv || (dk != 128 && dk != 256)) {
                    LLAMA_LOG_ERROR("%s: TurboQuant KV cache (%s/%s) needs n_embd_head_k == n_embd_head_v == 128 or 256, "
                                    "layer %u has %u/%u (the 128-wide rotation group and the instantiated kernels)\n", __func__,
                                    ggml_type_name(params.type_k), ggml_type_name(params.type_v), il, dk, dv);
                    return nullptr;
                }
            }
            LLAMA_LOG_INFO("%s: TurboQuant KV cache enabled: K=%s V=%s (flash attention; host-resident caches use the CPU "
                           "f32 vec_dot / to_float kernels with graph-side WHT rotation, device caches the backend's own wrapper)\n", __func__,
                           ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer_eff(); ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    // router_layer >= 0 means n_layer_nextn is repurposed for a router layer, not real MTP
    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        (model->hparams.n_layer_nextn == 0 || model->hparams.router_layer >= 0)) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    // Fork-specific checks (OpenPangu, DeepSeek4, K/V-cache Hadamard validity) and the Hadamard fold
    // into wv_b/wk_b_pp. Defined in src/llama.cpp -- see llama_fork_prepare_context_params there for
    // why, and for the commit that lost them.
    {
        std::string err;
        if (!llama_fork_prepare_context_params(*model, params, err)) {
            LLAMA_LOG_ERROR("%s: %s\n", __func__, err.c_str());
            return nullptr;
        }
    }

    try {
        auto * ctx = new llama_context(*model, params);
        const auto & cparams = ctx->get_cparams();

        if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN && cparams.rope_freq_scale != model->hparams.rope_freq_scale_train) {
            LLAMA_LOG_INFO("%s: custom YaRN scaling detected, re-adjusting n_ctx_train(%u)...\n", __func__, model->hparams.n_ctx_train);
            model->hparams.n_ctx_train = cparams.n_ctx_orig_yarn / cparams.rope_freq_scale;
            LLAMA_LOG_INFO("%s: n_ctx_train adjusted to %u\n", __func__, model->hparams.n_ctx_train);
        }

        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    // TriAttention cleanup. Missing here while the construction in the constructor was not, which
    // would have leaked the runtime and its stats on every context teardown.
    //
    // ORDER MATTERS, and src/llama.cpp spells out why: tria_runtime_free reads
    // rt->stats->num_layers internally, so the stats pointer is saved FIRST and freed AFTER the
    // runtime. Freeing stats first is a use-after-free that would only show up under
    // --triattention -- i.e. exactly the path nobody exercises by default. The ordering is kept
    // identical rather than tidied.
    if (ctx && ctx->tria_rt) {
        struct tria_stats * tri_stats = ctx->tria_rt->stats;
        tria_runtime_free(ctx->tria_rt);
        if (tri_stats) {
            tria_free(tri_stats);
        }
        ctx->tria_rt = nullptr;
    }

    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

bool llama_get_causal_attn(const llama_context * ctx) {
    return ctx->get_causal_attn();
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

void llama_context::clear_output_subset() {
    synchronize();
    output_subset.ids.clear();
    if (output_subset.buf) {
        ggml_backend_buffer_free(output_subset.buf);
        output_subset.buf = nullptr;
    }
    if (output_subset.ctx) {
        ggml_free(output_subset.ctx);
        output_subset.ctx = nullptr;
    }
    output_subset.partial  = nullptr;
    output_subset.in_graph = false;
    output_subset.used     = 0;
    prev.reset();      // the cached graph multiplies by the old tensor (or by the full weight)
    prev_mtp.reset();
}

bool llama_set_output_vocab_subset(llama_context * ctx, const llama_token * ids, int32_t n_ids) {
    ctx->synchronize();
    const auto & model = ctx->model;
    const int32_t n_vocab = model.vocab.n_tokens();
    std::vector<int32_t> v;
    v.reserve(n_ids > 0 ? (size_t) n_ids : 0);
    for (int32_t i = 0; i < n_ids; ++i) {
        if (ids[i] < 0 || ids[i] >= n_vocab) {
            return false;
        }
        v.push_back(ids[i]);
    }
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    if (v.empty() || (int32_t) v.size() >= n_vocab) {
        if (!ctx->output_subset.ids.empty()) {
            ctx->clear_output_subset();
        }
        return true;
    }
    if (v == ctx->output_subset.ids) {
        return true;
    }
    ggml_tensor * out = model.output;
    if (out == nullptr || out->buffer == nullptr || out->extra != nullptr || out->ne[1] != n_vocab || out->ne[2] != 1 || out->ne[3] != 1 ||
        model.output_b != nullptr || model.hadamard_rotation(out) != nullptr || model.weight_scale(out) != nullptr ||
        ggml_internal_get_type_traits(out->type).nrows != 1) {
        return false;   // split across devices / row-interleaved quantisation / scale, rotation or bias attached to it
    }
    ggml_init_params ip = { /*.mem_size =*/ 2*ggml_tensor_overhead() + 1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * c = ggml_init(ip);
    if (!c) {
        return false;
    }
    ggml_tensor * partial = ggml_new_tensor_2d(c, out->type, out->ne[0], (int64_t) v.size());
    ggml_set_name(partial, "output_vocab_subset");
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(c, ggml_backend_buffer_get_type(out->buffer));
    if (!buf) {
        ggml_free(c);
        return false;
    }
    // gather the rows once (a row is nb[1] bytes whatever the type: quantised blocks and any per-row metadata travel together)
    const size_t rb = out->nb[1];
    std::vector<uint8_t> row(rb);
    for (size_t i = 0; i < v.size(); ++i) {
        ggml_backend_tensor_get(out, row.data(), (size_t) v[i]*rb, rb);
        ggml_backend_tensor_set(partial, row.data(), i*rb, rb);
    }
    ctx->clear_output_subset();
    ctx->output_subset.ids     = std::move(v);
    ctx->output_subset.ctx     = c;
    ctx->output_subset.buf     = buf;
    ctx->output_subset.partial = partial;
    LLAMA_LOG_INFO("%s: output projection restricted to %zu of %d vocabulary entries (%.2f MiB gathered, type %s)\n", __func__,
                   ctx->output_subset.ids.size(), n_vocab, ggml_nbytes(partial)/1048576.0, ggml_type_name(out->type));
    return true;
}

int32_t llama_output_vocab_subset_used(const llama_context * ctx) {
    return ctx->output_subset.used;
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn_ith(i);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_seq_adapters_lora(
            llama_context * ctx,
            llama_seq_id seq_id,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_seq_adapters_lora call");
    }

    ctx->set_seq_adapters_lora(seq_id, adapters, n_adapters, scales);

    return 0;
}

void llama_set_draft_start(llama_context * ctx, llama_seq_id seq_id, llama_pos pos) {
    if (pos < 0) {
        ctx->draft_start.erase(seq_id);
    } else {
        ctx->draft_start[seq_id] = pos;
    }
}

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

// LEGACY-PATH FORWARDING. For a model on the llm_build_context path the cache that decodes is
// lctx.kv_self, not `mem` (see llama_memory_i::legacy_owner). Each mutating call is applied to
// BOTH so the two never disagree, and each query answers from kv_self, because that is the cache
// whose contents the next decode will attend. The legacy llama_kv_cache_*(ctx) functions are the
// ones the fork's own speculative code already uses, so no new cache logic is introduced here.

std::vector<ggml_backend_buffer_t> llama_legacy_kv_buffers(const llama_context * ctx) {
    std::vector<ggml_backend_buffer_t> res;
    for (auto * b : ctx->kv_self.bufs) {
        if (b != nullptr) {
            res.push_back(b);
        }
    }
    return res;
}

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
    if (mem->legacy_owner) {
        llama_kv_cache_clear(mem->legacy_owner);
    }
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    const bool res = mem->seq_rm(seq_id, p0, p1);
    if (mem->legacy_owner) {
        // kv_self's answer governs: a recurrent cache refuses a partial removal, and a caller
        // told `true` for a removal that did not happen would re-decode onto stale state.
        return llama_kv_cache_seq_rm(mem->legacy_owner, seq_id, p0, p1);
    }
    return res;
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    if (mem->legacy_owner) {
        llama_kv_cache_seq_cp(mem->legacy_owner, seq_id_src, seq_id_dst, p0, p1);
    }
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
    if (mem->legacy_owner) {
        llama_kv_cache_seq_keep(mem->legacy_owner, seq_id);
    }
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
    if (mem->legacy_owner) {
        llama_kv_cache_seq_add(mem->legacy_owner, seq_id, p0, p1, delta);
    }
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
    if (mem->legacy_owner) {
        llama_kv_cache_seq_div(mem->legacy_owner, seq_id, p0, p1, d);
    }
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    if (mem->legacy_owner) {
        // upstream contract: -1 for an empty sequence (the legacy hybrid / recurrent branch reads 0 for it)
        if (llama_kv_cache_seq_is_empty(mem->legacy_owner, seq_id)) {
            return -1;
        }
        return llama_kv_cache_seq_pos_min(mem->legacy_owner, seq_id);
    }
    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    if (mem->legacy_owner) {
        // upstream contract: -1 for an empty sequence (llama_kv_cache_seq_pos_max starts at 0)
        if (llama_kv_cache_seq_is_empty(mem->legacy_owner, seq_id)) {
            return -1;
        }
        return llama_kv_cache_seq_pos_max(mem->legacy_owner, seq_id);
    }
    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    if (mem->legacy_owner) {
        return llama_supports_ctx_shift(mem->legacy_owner);
    }
    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// defined in llama.cpp: whole-context and file state of kv_self, for contexts on the legacy graph path
size_t llama_legacy_state_get_size(llama_context * ctx);
size_t llama_legacy_state_get_data(llama_context * ctx, uint8_t * dst, size_t size);
size_t llama_legacy_state_set_data(llama_context * ctx, const uint8_t * src, size_t size);
bool   llama_legacy_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out);
bool   llama_legacy_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count);
size_t llama_legacy_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count);
size_t llama_legacy_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out);

static bool llama_state_is_legacy(const llama_context * ctx);   // defined below, beside the seq routing

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    if (llama_state_is_legacy(ctx)) {
        return llama_legacy_state_get_size(ctx);
    }
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    if (llama_state_is_legacy(ctx)) {
        return llama_legacy_state_get_data(ctx, dst, size);
    }
    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    if (llama_state_is_legacy(ctx)) {
        return llama_legacy_state_set_data(ctx, src, size);
    }
    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        if (llama_state_is_legacy(ctx)) {
            return llama_legacy_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
        }
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        if (llama_state_is_legacy(ctx)) {
            return llama_legacy_state_save_file(ctx, path_session, tokens, n_token_count);
        }
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

// SIGNATURES REALIGNED WITH include/llama.h, 2026-09-13.
//
// These three took (ctx, seq_id[, dst, size]) and hardcoded a flags value of 0. The PUBLIC header
// declares all three WITH a llama_state_seq_flags parameter -- llama.h:1507, :1512 and :1523 --
// so every caller compiles against the flags-carrying form while this file defined the flagless
// one. The two never met.
//
// It went unnoticed because this file was compiled by NOTHING: src/llama.cpp carried its own
// definitions matching the header, so the link succeeded and nothing ever asked whether the copy
// in here agreed. Wiring the file is what surfaced it -- the build went from resolving these
// symbols to reporting them undefined at 60, 60 and 30 references.
//
// This is the same shape as the load_mode divergence found earlier in this merge: an API whose
// options were folded INTO the base function upstream, against an implementation still written
// for the older split form. Hardcoding 0 was not merely a signature mismatch; it silently
// discarded whatever flags a caller passed.
size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id,
                                llama_state_seq_flags flags) {
    return llama_state_seq_get_size_ext(ctx, seq_id, flags);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size,
                                llama_seq_id seq_id, llama_state_seq_flags flags) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, flags);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size,
                                llama_seq_id seq_id, llama_state_seq_flags flags) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, flags);
}

// defined in llama.cpp: the per-sequence state of kv_self, for contexts on the legacy graph path
size_t llama_legacy_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags);
size_t llama_legacy_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags);
size_t llama_legacy_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags);

// the cache that decodes is kv_self, not ctx->memory (see llama_memory_i::legacy_owner)
static bool llama_state_is_legacy(const llama_context * ctx) {
    const llama_memory_t mem = llama_get_memory(ctx);
    return mem != nullptr && mem->legacy_owner == ctx;
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (llama_state_is_legacy(ctx)) {
        return llama_legacy_state_seq_get_size(ctx, seq_id, flags);
    }
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    if (llama_state_is_legacy(ctx)) {
        return llama_legacy_state_seq_get_data(ctx, dst, size, seq_id, flags);
    }
    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    if (llama_state_is_legacy(ctx)) {
        return llama_legacy_state_seq_set_data(ctx, src, size, seq_id, flags);
    }
    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

// ---------------------------------------------------------------------------------------------------------------------
// Slot-file MODEL IDENTITY (row S69-ENGREV-FERROX-SLOT-FILE-MODEL-IDENTITY-2026-10-27-01; also Jan slots.rs/http.rs:664-803).
// SOURCE-READ (design only, nothing copied): ferrox crates/frink-server/src/slots/identity.rs:1-409 (shape fields first, name the first
// mismatch, an absent identity is never agreement) and crates/frink-core/src/kv_signature.rs:1-300 (fingerprint over the tensor directory +
// first/last 4 KiB of each tensor). Before this a slot file saved under checkpoint A restored under checkpoint B of the same arch with no
// error (llama-context.cpp state_read_data checked only the arch string; the TODO 'add more model-specific info which should prevent
// loading the session file if not identical' stood). The .gguf-state file format is NOT changed (the server appends checkpoint and draft
// blocks after the llama payload by byte offset): the identity rides in a sidecar `<file>.ident` of key=value lines.
// A file with no sidecar (saved by an older build) is restored with a one-line warning: it cannot be proven wrong, and refusing it
// would orphan every existing slot file. Sampling limit: a flipped byte outside a tensor's first/last 4 KiB is not seen by the fingerprint.
// ---------------------------------------------------------------------------------------------------------------------
namespace {

static uint64_t slot_fnv(uint64_t h, const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static uint64_t slot_model_fingerprint(const llama_model & m) {
    static std::mutex mtx;
    static std::unordered_map<const llama_model *, uint64_t> cache;
    {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = cache.find(&m);
        if (it != cache.end()) return it->second;
    }
    std::vector<std::pair<std::string, ggml_tensor *>> ts(m.tensors_by_name.begin(), m.tensors_by_name.end());
    std::sort(ts.begin(), ts.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
    uint64_t h = 14695981039346656037ull;
    h = slot_fnv(h, llm_arch_name(m.arch), std::strlen(llm_arch_name(m.arch)));
    std::vector<uint8_t> buf(4096);
    for (const auto & [name, t] : ts) {
        if (!t) continue;
        const uint64_t nlen = name.size();
        h = slot_fnv(h, &nlen, sizeof(nlen));
        h = slot_fnv(h, name.data(), name.size());
        const int32_t ty = (int32_t) t->type;
        h = slot_fnv(h, &ty, sizeof(ty));
        h = slot_fnv(h, t->ne, sizeof(t->ne));
        const size_t nb = ggml_nbytes(t);
        h = slot_fnv(h, &nb, sizeof(nb));
        // first and last 4 KiB of the data; split (row/graph-split) tensors have no single buffer, directory only for those
        if (t->buffer != nullptr && t->extra == nullptr && nb > 0) {
            const size_t w1 = std::min<size_t>(4096, nb);
            ggml_backend_tensor_get(t, buf.data(), 0, w1);
            h = slot_fnv(h, buf.data(), w1);
            if (nb > w1) {
                const size_t off2 = std::max<size_t>(w1, nb - 4096);
                const size_t w2   = nb - off2;
                ggml_backend_tensor_get(t, buf.data(), off2, w2);
                h = slot_fnv(h, buf.data(), w2);
            }
        }
    }
    std::lock_guard<std::mutex> lk(mtx);
    cache[&m] = h;
    return h;
}

// shape fields first, fingerprint last: the first mismatch is the one named
static std::vector<std::pair<std::string, std::string>> slot_identity_of(const llama_model & m) {
    uint32_t nkv = 0;
    for (uint32_t il = 0; il < m.hparams.n_layer; il++) nkv = std::max(nkv, m.hparams.n_head_kv(il));
    char fp[32];
    std::snprintf(fp, sizeof(fp), "%016llx", (unsigned long long) slot_model_fingerprint(m));
    return {
        {"arch",          llm_arch_name(m.arch)},
        {"n_layer",       std::to_string(m.hparams.n_layer)},
        {"n_embd",        std::to_string(m.hparams.n_embd)},
        {"n_head_kv_max", std::to_string(nkv)},
        {"n_vocab",       std::to_string(m.hparams.n_vocab)},
        {"n_tensors",     std::to_string(m.tensors_by_name.size())},
        {"fingerprint",   fp},
    };
}

static std::string slot_ident_path(const char * filepath) { return std::string(filepath) + ".ident"; }

static void slot_identity_write(const llama_model & m, const char * filepath) {
    std::ofstream f(slot_ident_path(filepath), std::ios::trunc);
    f << "ggsq-identity=1\n";
    for (const auto & kv : slot_identity_of(m)) f << kv.first << "=" << kv.second << "\n";
    if (!f) LLAMA_LOG_WARN("%s: could not write %s (slot restore will not be identity-checked)\n", __func__, slot_ident_path(filepath).c_str());
}

// 0 = identical, 1 = mismatch (why names the first differing field), 2 = no sidecar (legacy file, cannot be proven wrong)
static int slot_identity_check(const llama_model & m, const char * filepath, std::string & why) {
    std::ifstream f(slot_ident_path(filepath));
    if (!f) { why = "no identity sidecar (saved by an older build); model identity not verified"; return 2; }
    std::unordered_map<std::string, std::string> saved;
    std::string line;
    while (std::getline(f, line)) {
        const size_t eq = line.find('=');
        if (eq != std::string::npos) saved[line.substr(0, eq)] = line.substr(eq + 1);
    }
    for (const auto & kv : slot_identity_of(m)) {
        auto it = saved.find(kv.first);
        if (it == saved.end()) { why = "identity sidecar lacks field '" + kv.first + "'"; return 1; }   // absence is never agreement
        if (it->second != kv.second) {
            why = "model identity mismatch: field '" + kv.first + "' saved=" + it->second + " serving=" + kv.second;
            return 1;
        }
    }
    return 0;
}

} // namespace

int llama_state_seq_file_identity_check(struct llama_context * ctx, const char * filepath, char * why, size_t why_len) {
    std::string w;
    int rc = 0;
    // legacy-graph contexts are the same llama_context type with the same model: both paths are checked
    rc = slot_identity_check(ctx->model, filepath, w);
    if (why && why_len > 0) std::snprintf(why, why_len, "%s", w.c_str());
    return rc;
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        const size_t res = llama_state_is_legacy(ctx)
            ? llama_legacy_state_seq_save_file(ctx, filepath, seq_id, tokens, n_token_count)
            : ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
        if (res > 0) slot_identity_write(ctx->model, filepath);
        return res;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        {
            std::string why;
            const int idrc = slot_identity_check(ctx->model, filepath, why);
            if (idrc == 1) {   // refuse BEFORE any state is read into the KV cache (legacy and new path alike)
                LLAMA_LOG_ERROR("%s: refusing slot file '%s': %s\n", __func__, filepath, why.c_str());
                return 0;
            }
            if (idrc == 2 && tokens_out != nullptr) {
                LLAMA_LOG_WARN("%s: slot file '%s': %s\n", __func__, filepath, why.c_str());
            }
        }
        if (llama_state_is_legacy(ctx)) {
            return llama_legacy_state_seq_load_file(ctx, filepath, dest_seq_id, tokens_out, n_token_capacity, n_token_count_out);
        }
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

// compat: llama_batch -> llama_batch_ext -> encode/decode

int llama_context::encode(const llama_batch & batch_inp) {
    // Old-generation models encode through the legacy builder: model.build_graph() is nullptr for them by design
    // (llama-model.cpp:32), and llama_encode_internal takes the llama_batch as it is.
    if (model.uses_legacy_graph_builder()) {
        return llama_encode_internal(*this, batch_inp);
    }
    llama_batch_compat compat(this, batch_inp, model.hparams.n_embd_inp_enc());
    return encode(*compat.batch_ext);
}

int llama_context::decode(const llama_batch & batch_inp) {
    // Old-generation models decode through the legacy builder (llama_decode_internal, src/llama.cpp), which brings its
    // own batch splitting, output and KV bookkeeping; its return codes pass through unchanged.
    if (model.uses_legacy_graph_builder()) {
        return llama_decode_internal(*this, batch_inp);
    }
    llama_batch_compat compat(this, batch_inp);
    return decode(*compat.batch_ext);
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    // TriAttention KV scoring -- fires after each successful decode step.
    //
    // THIS IS THE CALL THAT MAKES THE FEATURE DO SOMETHING, and it was missing here while the
    // runtime construction was not. src/llama.cpp says it exactly: "Without it the runtime could
    // be constructed and logged as enabled while never scoring anything, which is the same dormant
    // shape one level further in." Porting the init without this call would have produced a
    // context that PRINTS "triattention enabled" and scores nothing -- the precise defect the
    // original author left the warning about.
    //
    // Guarded on ret == 0 and not on the wrapper's own (ret != 0 && ret != 1) error test above:
    // the monolith scores only on a fully successful decode, and 1 means something else happened.
    if (ret == 0 && ctx->tria_rt) {
        tria_maybe_score(ctx->tria_rt, ctx);
    }

    // Windows working-set release after a PREFILL (more than one token). Opt-in, default OFF: LLAMA_MMAP_RELEASE_AFTER_PREFILL=1.
    // Strata measured a large drop in RAM held by the mapped weights at a 3.7-4.4% prefill cost (the pages fault back in); on a host with
    // plenty of RAM expect less benefit and the same cost. release_fragment does nothing off Windows and on anonymous mappings.
    // Row S69-STRATA-S1-WINDOWS-WORKINGSET-RELEASE-2026-10-07-01.
    if (ret == 0 && batch.n_tokens > 1) {
        static const bool release_after_prefill = [] {
            const char * v = std::getenv("LLAMA_MMAP_RELEASE_AFTER_PREFILL");
            return v != nullptr && v[0] == '1';
        }();
        if (release_after_prefill) {
            for (const auto & mapping : ctx->model.mappings) {
                mapping->release_fragment(0, mapping->size());
            }
        }
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
    LLAMA_LOG_INFO("%s:   graph reallocs = %10d    / %5d        (same size) / (changed size)\n", __func__, data.n_reallocs_same, data.n_reallocs_diff);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

int32_t llama_process(llama_context * ctx, llama_process_type type, llama_batch_ext * batch) {
    switch (type) {
        case LLAMA_PROCESS_TYPE_ENCODE: return ctx->encode(*batch);
        case LLAMA_PROCESS_TYPE_DECODE: return ctx->decode(*batch);
    }
    return -1;
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

// Per-device rows for /memory and /metrics. Hand port of ik_llama.cpp 55813a5d7 src/llama.cpp (llama_memory_rows /
// llama_memory_breakdown): the row grouping and the buffer-type -> row rule are ik's; the per-context byte counts come from
// this fork's llama_context::memory_breakdown() (weights + memory module + legacy kv_self + MoE cache + compute), which
// already covers what ik sums buffer by buffer. A model shared by two contexts is counted once.
namespace {
struct llama_memory_rows {
    std::vector<llama_memory_row>           rows;
    std::vector<ggml_backend_buffer_type_t> bufts;
    size_t i_host = 0;

    llama_memory_row & add(const char * name, ggml_backend_buffer_type_t buft, bool is_device) {
        llama_memory_row row = {};
        snprintf(row.name, sizeof(row.name), "%s", name);
        row.is_device = is_device;
        rows.push_back(row);
        bufts.push_back(buft);
        return rows.back();
    }

    llama_memory_row & find(ggml_backend_buffer_type_t buft) {
        for (size_t i = 0; i < i_host; ++i) {
            if (bufts[i] == buft) {
                return rows[i];
            }
        }
        if (ggml_backend_buft_is_host(buft)) {
            return rows[i_host];
        }
        for (size_t i = i_host + 1; i < rows.size(); ++i) {
            if (bufts[i] == buft) {
                return rows[i];
            }
        }
        return add(ggml_backend_buft_name(buft), buft, false);
    }
};
} // namespace

int32_t llama_memory_breakdown_rows(
        const struct llama_context * const * ctxs,
                               int32_t   n_ctxs,
        const struct llama_memory_buft * mmproj,
                               int32_t   n_mmproj,
              struct llama_memory_row * rows,
                               int32_t   n_rows) {
    std::vector<const llama_context *> unique;
    for (int32_t i = 0; i < n_ctxs; ++i) {
        if (ctxs[i] && std::find(unique.begin(), unique.end(), ctxs[i]) == unique.end()) {
            unique.push_back(ctxs[i]);
        }
    }

    llama_memory_rows res;
    // device rows first: one per offload device of every model, deduplicated by buffer type
    for (const llama_context * ctx : unique) {
        const llama_model * model = llama_get_model(ctx);
        for (int32_t i = 0; i < llama_model_n_devices(model); ++i) {
            ggml_backend_dev_t dev = llama_model_get_device(model, i);
            if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                continue;
            }
            ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
            if (buft == nullptr || ggml_backend_buft_is_host(buft) ||
                std::find(res.bufts.begin(), res.bufts.end(), buft) != res.bufts.end()) {
                continue;
            }
            llama_memory_row & row = res.add(ggml_backend_buft_name(buft), buft, true);
            ggml_backend_dev_memory(dev, &row.free, &row.total);
        }
    }
    res.i_host = res.rows.size();
    res.add("Host", nullptr, false);

    std::vector<const llama_model *> seen_models;
    for (const llama_context * ctx : unique) {
        const llama_model * model = llama_get_model(ctx);
        const bool first = std::find(seen_models.begin(), seen_models.end(), model) == seen_models.end();
        if (first) {
            seen_models.push_back(model);
        }
        for (const auto & [buft, mb] : ctx->memory_breakdown()) {
            if (buft == nullptr) {
                continue;
            }
            llama_memory_row & row = res.find(buft);
            if (first) {
                row.model += mb.model;
            }
            row.context += mb.context;
            row.compute += mb.compute;
        }
    }
    for (int32_t i = 0; i < n_mmproj; ++i) {
        if (mmproj[i].size > 0) {
            res.find(mmproj[i].buft).mmproj += mmproj[i].size;
        }
    }

    for (int32_t i = 0; i < n_rows && i < (int32_t) res.rows.size(); ++i) {
        rows[i] = res.rows[i];
    }
    return (int32_t) res.rows.size();
}

// The memory MODULE's own contribution (what llama_memory_i::memory_breakdown() reports), split out of
// llama_get_memory_breakdown() so a test can tell a metadata-only module (0) from a second real KV
// allocation. S69-KV-DEDUP-METADATA-ONLY-MEMORY-MODULE-2026-09-29-01.
size_t llama_get_memory_module_bytes(const struct llama_context * ctx) {
    size_t res = 0;
    if (ctx->get_memory()) {
        for (const auto & [buft, size] : ctx->get_memory()->memory_breakdown()) {
            GGML_UNUSED(buft);
            res += size;
        }
    }
    return res;
}

// Bytes held by the legacy kv_self (the cache every legacy graph reads).
size_t llama_get_kv_self_bytes(const struct llama_context * ctx) {
    return ctx->kv_self.total_size();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}
