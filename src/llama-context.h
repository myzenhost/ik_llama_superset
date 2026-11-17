#pragma once

#include "llama-impl.h"
#include "llama-cparams.h"
#include "llama-sampling.h"
// S69-WIRE66-CONTEXT-MEMORY-ARCH-DECISION slice 2: llama_memory_ptr for the `memory` member
// below. Upstream shape (PR #11213), added ALONGSIDE kv_self (not repointing it) -- see
// member comment.
#include "llama-memory.h"
// TriAttention KV scoring (unified ik_llama build). The accessors below and
// triattention-bridge.cpp were preserved through the consolidation; this include and
// the tria_rt member were not, which left the feature compiled but unreachable.
#include "triattention-runtime.h"
#include "llama-moe-cache.h"

#include "llama-spec-features.h"
// S69-KT-EXPERT-POOL-2026-09-28 unit 1: llama_moe_recorder for the moe_runtime member below.
// Header-only types (no llama_context dependency of its own), see llama-moe-pool.h for
// attribution (llama.cpp PR #27861 / discussion #28248, KTransformers doc, all cited there).
#include "llama-moe-pool.h"
// slice 14: llm_graph_result_ptr and llama_cross for the members below. This also
// supplies llama-batch.h (llama_batch_allocr) and llama-adapter.h
// (llama_adapter_loras_ptr) transitively. Checked for a cycle: nothing in that chain
// includes llama-context.h.
#include "llama-graph.h"
// slice 14: llama_memory_breakdown, the return type of memory_breakdown(). It lives
// here rather than in llama.h, and this header not including it is one of the two
// causes that made all four BUILT files fail in slice 13.
#include "llama-ext.h"
// slice 11: ggml_opt_context_t for the opt_ctx member below (upstream declares it at
// 279b4bfaf^2:src/llama-context.h:352). It was the only one of the eight added members
// whose type this header could not already name.
#include "ggml-opt.h"

#include <array>
#include <map>
#include <vector>

struct llama_model;
class llama_batch_allocr;
class llama_moe_cache;

#include <vector>
#include <map>
#include <tuple>
#include <set>
#include <memory>
#include <unordered_set>   // std::unordered_set in triattention_compact
#include <unordered_map>   // std::unordered_map, previously only transitive via llama-sampling.h
#include <algorithm>   // std::sort in triattention_compact
#include <cctype>      // slice 28: std::tolower, used by the shared striequals below
#include <string>      // slice 28: std::vector<std::string> in item_in_list below

// S69-WIRE66-CONTEXT-MEMORY-ARCH-DECISION slice 3: ggml_backend_buffer_ptr / ggml_context_ptr for
// llama_memory_buffer below. Upstream's llama-context.h includes this for the same two types.
#include "ggml-cpp.h"

// PORTED VERBATIM FROM UPSTREAM, src/llama-context.h:27-40 at merge parent 279b4bfaf^2. Not
// invented: llama-context.cpp's state-save/load code was already WRITTEN against this exact shape
// and had nothing to compile against, which is why ~11 of its errors name a type that exists
// nowhere in this fork. The bodies index it as mbufs_new[buft].n_tensors / .total_size, iterate it
// with a structured binding, move-assign the mapped value, and call .reset()/.get() on buf and ctx
// -- every one of those requirements is satisfied by upstream's own definition, so copying it is
// both the smallest change and the one that cannot drift from the code that uses it.
//
// A NEW TYPE WITH NO EXISTING CALLERS is why this slice is additive rather than a migration: adding
// it can break nothing that compiles today, unlike the sched/opt_ctx family, which would require
// changing a member other code already reads.

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_swa_window_view {
    int64_t w_view  = 0;
    int64_t win_off = 0;
    bool engaged    = false;
};

static inline llama_swa_window_view llama_swa_calc_window_view(
        int64_t n_kv, int64_t n_tokens, int64_t window, int64_t pad) {
    llama_swa_window_view result;
    if (window <= 0 || n_kv <= 0) {
        result.w_view = n_kv;
        return result;
    }

    const int64_t unpadded = window + pad + n_tokens;
    const int64_t overcovered = pad > 1 ? ((unpadded + pad - 1) / pad) * pad : unpadded;
    result.w_view  = overcovered < n_kv ? overcovered : n_kv;
    result.win_off = n_kv - result.w_view;
    result.engaged = result.w_view < n_kv;
    return result;
}

static inline llama_swa_window_view llama_swa_calc_window_view_compact(
        int64_t live, int64_t sink_rows, int64_t n_tokens, int64_t window, int64_t pad) {
    llama_swa_window_view result;
    const int64_t live_padded = pad > 1 ? ((live + pad - 1) / pad) * pad : live;

    const int64_t unpadded = window + pad + n_tokens;
    const int64_t overcovered = pad > 1 ? ((unpadded + pad - 1) / pad) * pad : unpadded;
    result.w_view  = overcovered < live_padded ? overcovered : live_padded;
    result.win_off = sink_rows + live_padded - result.w_view;
    result.engaged = true;
    return result;
}

struct llama_kv_cell {
    llama_pos pos   = -1;
    // the token stored in this cell; n-gram architectures (DeepSeek-V4.1 engram)
    // read a token's predecessors from here, which can sit in an earlier decode call
    llama_token tok = -1;
    llama_pos delta = 0;
    int32_t   src   = 0; // used by recurrent state models to copy states

    std::set<llama_seq_id> seq_id;

    bool has_seq_id(const llama_seq_id & id) const {
        return seq_id.find(id) != seq_id.end();
    }

    bool is_empty() const {
        return seq_id.empty();
    }

    bool is_same_seq(const llama_kv_cell & other) const {
        return seq_id == other.seq_id;
    }
};

// ring-buffer of cached KV data
struct llama_kv_cache_fork {
    // the FA kernels require padding to avoid extra runtime boundary checks
    static uint32_t get_padding(bool flash_attn) { return flash_attn ? 256u : 32u; }

    bool has_shift = false;
    // cell-index order no longer matches position order; index-based SWA attention
    // windowing must not be used
    bool cells_disordered = false;
    bool do_defrag = false;
    bool do_copy   = false;
    bool recurrent = false; // with recurrent state models, a cell can hold the state for more than one past token
    bool hybrid    = false;
    bool v_trans   = true;  // the value tensor is transposed

    // openPangu s_l holds position-strict MoME conv state, not per-sequence recurrent slots: qnext
    // seq ops and generic serialization skip it, the openPangu state layouts carry it instead.
    bool s_l_position_strict = false;

    // Note: The value of head isn't only used to optimize searching
    // for a free KV slot. llama_decode_internal also uses it, so it
    // cannot be freely changed after a slot has been allocated.
    uint32_t head = 0;
    uint32_t size = 0;
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    std::vector<uint32_t> row_count;

    bool any_compacted() const { return !row_count.empty(); }

    uint32_t rows(int il) const {
        return row_count.empty() ? size : row_count[il];
    }

    bool is_compacted(int il) const {
        return !row_count.empty() && row_count[il] < size;
    }

    // rows [sinks|window]; row(pos) = sink_rows + pos - pos_base_swa
    // sink_rows == hparams.param_sink_number, which only the openPangu loader sets (0 elsewhere)
    uint32_t  size_swa     = 0;
    uint32_t  sink_rows    = 0;
    uint32_t  window_swa   = 0;
    uint32_t  head_swa     = 0;
    llama_pos pos_base_swa = 0;

    uint32_t live_swa() const { return head_swa - sink_rows; }

    // computed before each graph build
    uint32_t n = 0;

    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;

    std::vector<llama_kv_cell> cells;

    std::vector<struct ggml_tensor *> k_l; // per layer
    std::vector<struct ggml_tensor *> v_l;

    // FFD (--ffd) 2-bit per-block K shadow, per layer; nullptr for a layer the shadow does not cover (see
    // llama_kv_cache_init). ffd_k2: I8 [head_dim/4, kv_size, n_head_kv, 1] codes; ffd_ks: F32 [head_dim, n_blocks,
    // n_head_kv, 1] per-channel scales. Kept current by GGML_OP_FFD_QUANT_K after every K store.
    std::vector<struct ggml_tensor *> ffd_k2;
    std::vector<struct ggml_tensor *> ffd_ks;
    int32_t ffd_block = 0;
    std::vector<struct ggml_tensor *> s_l; // per layer recurrent state storage (Qwen3Next)

    // In-Place TTT fast-weight state (row S69-FORK-INPLACE-TTT-2026-10-26-01), see src/llama-ttt.h: one F32 row per
    // sequence slot (slot = seq_id) on each TTT layer, nullptr elsewhere; empty and ttt_chunk = 0 unless the model
    // carries ttt.* keys. The state is position-strict: seq_rm may only drop a whole sequence or a tail that stays in
    // the current chunk, and seq_add / seq_div are refused.
    std::vector<struct ggml_tensor *> ttt_l;
    uint32_t ttt_chunk = 0;

    // Persistent DSA indexer-key cache. One per indexer layer, MQA single head:
    // [indexer_head_size, kv_size]. Stores architecture-specific indexer keys in their
    // scoring representation so a decoded token scores against all past indexer keys.
    // Empty unless the model has the DSA indexer.
    std::vector<struct ggml_tensor *> kr_l;

    // Pooled block keys for Qwen sparse attention: [indexer_head_size, kv_size/compress_ratio],
    // already normalized and rotated. Empty unless the model scores blocks.
    std::vector<struct ggml_tensor *> kp_l;

    // When true, the delta_net graph builder will enable per-step SSM state saves
    bool save_per_step_ssm = false;

    std::vector<llama_split_tensor> split_k_l;
    std::vector<llama_split_tensor> split_v_l;
    std::vector<llama_split_tensor> split_s_l;

    // Per-device replicas of the MLA compressed-latent KV cache (-sm graph for DEEPSEEK2/GLM_DSA/MISTRAL4).
    std::vector<llama_split_tensor> replicated_k_l;

    std::vector<struct ggml_context *> ctxs;
    std::vector<ggml_backend_buffer_t> bufs;

    // Bytes per buffer type this cache occupies -- the real buffer size in a normal run, the projected
    // size under hparams.no_alloc (where `bufs` holds zero-size dummies). Filled by llama_kv_cache_init
    // so llama_context::memory_breakdown can count this cache; before 2026-09-27 nothing counted it.
    std::map<ggml_backend_buffer_type_t, size_t> buft_sizes;

    size_t total_size() const {
        size_t size = 0;
        for (ggml_backend_buffer_t buf : bufs) {
            size += ggml_backend_buffer_get_size(buf);
        }
        return size;
    }

    // GPU-resident checkpoint for recurrent/hybrid speculative decoding
    struct gpu_checkpoint {
        std::vector<llama_kv_cell> cells_snapshot;
        uint32_t head_snapshot = 0;
        uint32_t used_snapshot = 0;

        std::vector<ggml_tensor *> s_l_shadow;

        std::vector<std::vector<ggml_tensor *>> split_s_l_shadow;

        // Per-step SSM state checkpoints for speculative decoding.
        std::vector<std::vector<ggml_tensor *>> per_step_ssm;

        // Per-step conv feature buffer: stores qkv_mixed features from the
        // verification forward pass so conv state can be reconstructed at any step.
        // One tensor per recurrent layer, each sized [conv_dim * max_tokens].
        //std::vector<std::vector<ggml_tensor *>> per_step_qkv;
        std::vector<std::vector<ggml_tensor *>> per_step_conv;
        // Qwen4Exp PLE checkpoint state
        std::vector<std::vector<ggml_tensor *>> per_step_ple;
        std::vector<int64_t> per_step_ple_dim;
        std::vector<int64_t> per_step_ple_offset;

        int32_t per_step_n_tokens = 0;
        int32_t per_step_max_allocated = 0;
        int64_t per_step_ssm_state_size = 0;
        int64_t per_step_conv_state_dim = 0;
        int64_t per_step_conv_dim = 0;
        int32_t per_step_d_conv = 0;

        // DSV4 per-step compressor-state base and per-row deltas.
        std::vector<ggml_tensor *> dsv4_per_step_state;
        std::vector<ggml_tensor *> dsv4_per_step_state_shadow;
        std::vector<ggml_tensor *> dsv4_per_step_delta;
        std::vector<ggml_context *> dsv4_per_step_shadow_ctxs;
        std::vector<ggml_backend_buffer_t> dsv4_per_step_shadow_bufs;
        std::vector<int32_t> dsv4_per_step_csa_dst;
        std::vector<int32_t> dsv4_per_step_hca_dst;
        std::vector<int32_t> dsv4_per_step_lid_dst;
        std::vector<int32_t> dsv4_per_step_csa_src;
        std::vector<int32_t> dsv4_per_step_hca_src;
        std::vector<int32_t> dsv4_per_step_lid_src;
        bool dsv4_per_step_allocated = false;
        bool dsv4_per_step_saved = false;
        int32_t dsv4_per_step_max_tokens = 0;
        size_t dsv4_per_step_base_bytes = 0;
        size_t dsv4_per_step_delta_bytes = 0;

        int selected_spec_mode = -1;
        int fixed_spec_mode = LLAMA_SPEC_CKPT_NONE;
        int32_t fixed_max_tokens = 0;

        // Serialized sequence state for CPU mode
        std::vector<uint8_t> cpu_state_data;

        // Private DSV4 state snapshot used by GPU/CPU fallback modes.
        std::vector<std::vector<uint8_t>> dsv4_state_data;
        std::vector<ggml_tensor *> dsv4_state_shadow;
        std::vector<struct ggml_context *> dsv4_shadow_ctxs;
        std::vector<ggml_backend_buffer_t> dsv4_shadow_bufs;
        bool dsv4_shadow_allocated = false;
        bool dsv4_shadow_saved = false;

        // Separate storage for per-step allocations
        std::vector<struct ggml_context *>   per_step_ctxs;
        std::vector<ggml_backend_buffer_t>   per_step_bufs;

        std::vector<struct ggml_context *>   shadow_ctxs;
        std::vector<ggml_backend_buffer_t>   shadow_bufs;

        bool allocated = false;
        bool shadow_conv_only = false;
        bool saved     = false;

        void release_dsv4_per_step();
        void release_dsv4_snapshot();

        void release_per_step() {
            for (struct ggml_context * ctx : per_step_ctxs) {
                ggml_free(ctx);
            }
            for (ggml_backend_buffer_t buf : per_step_bufs) {
                ggml_backend_buffer_free(buf);
            }
            per_step_ctxs.clear();
            per_step_bufs.clear();
            per_step_ssm.clear();
            per_step_conv.clear();
            per_step_ple.clear();
            per_step_ple_dim.clear();
            per_step_ple_offset.clear();
            per_step_n_tokens = 0;
            per_step_max_allocated = 0;
            per_step_ssm_state_size = 0;
            per_step_conv_state_dim = 0;
            per_step_conv_dim = 0;
            per_step_d_conv = 0;
        }

        void release() {
            release_dsv4_per_step();
            release_dsv4_snapshot();

            for (struct ggml_context * ctx : shadow_ctxs) {
                ggml_free(ctx);
            }
            shadow_ctxs.clear();
            for (ggml_backend_buffer_t buf : shadow_bufs) {
                ggml_backend_buffer_free(buf);
            }
            shadow_bufs.clear();
            s_l_shadow.clear();
            split_s_l_shadow.clear();
            allocated = false;
            saved = false;

            release_per_step();
        }

        ~gpu_checkpoint() {
            release();
        }
    };

    gpu_checkpoint ckpt;
    std::map<llama_pos, gpu_checkpoint *> ckpt_rewind_map;
    std::vector<std::unique_ptr<gpu_checkpoint>> ckpt_rewind_pool;

    bool checkpoint_alloc_shadows(llama_ckpt_data ckpt_type, bool conv_only_shadow = false);
    bool checkpoint_supported() const;
    bool checkpoint_save(ggml_backend_sched_t sched, llama_ckpt_data ckpt_type);
    bool checkpoint_restore(ggml_backend_sched_t sched, llama_ckpt_data ckpt_type);
    void checkpoint_delete(llama_ckpt_data ckpt_type);

    gpu_checkpoint & llama_get_ckpt(llama_ckpt_data ckpt_type);

    bool llama_ckpt_mode_equal(llama_ckpt_data ckpt_type, int mode) const;

    // Per-step checkpoint: allocate, restore step k's full state (SSM + conv) to cache
    bool per_step_alloc(const llama_model & model, int max_tokens);
    bool per_step_restore(ggml_backend_sched_t sched, llama_ckpt_data ckpt_type, int step, uint32_t slot);

    ~llama_kv_cache_fork() {
        for (struct ggml_context * ctx : ctxs) {
            ggml_free(ctx);
        }
        for (ggml_backend_buffer_t buf : bufs) {
            ggml_backend_buffer_free(buf);
        }
    }


    // ---- JusticianAI fork: KV accessors used by triattention-bridge.cpp and
    // offload_attention_bridge.cpp. These came from our source tree, not from the
    // branch this file was merged with, so the union that resolved this file could
    // not have preserved them.
    // Return raw key tensor for a given layer (used by TriAttention scorer).
    struct ggml_tensor * get_k_raw(int layer_idx) const {
        if (layer_idx < 0 || (size_t)layer_idx >= k_l.size()) return nullptr;
        return k_l[layer_idx];
    }

    // Return count of used (non-empty) KV cells.
    uint32_t get_used_n_kv() const {
        return used;
    }

    // Collect positions of all non-empty cells into positions vector.
    // Returns true on success, false if empty.
    bool get_cell_positions(std::vector<llama_pos> & positions) const {
        positions.clear();
        for (const auto & cell : cells) {
            if (!cell.is_empty()) {
                positions.push_back(cell.pos);
            }
        }
        return !positions.empty();
    }

    // Return the maximum position seen in any cell for the given seq_id.
    // Used to estimate n_kv for TriAttention scheduling.
    llama_pos seq_pos_max(llama_seq_id seq_id) const {
        llama_pos pmax = -1;
        for (const auto & cell : cells) {
            if (!cell.is_empty() && cell.has_seq_id(seq_id)) {
                if (cell.pos > pmax) pmax = cell.pos;
            }
        }
        return pmax;
    }

    // Compact the KV cache: remove all cells whose position is NOT in keep_positions.
    // keep_positions: set of logical indices (0..n_used-1) to KEEP from the
    // position-sorted cell list. Returns true if any cells were removed.
    // NOTE: This performs cell invalidation only; the caller must trigger a
    //       defrag via kv_self.do_defrag = true before the next decode.
    bool triattention_compact(const std::vector<uint32_t> & keep_positions) {
        // Build sorted list of used cell positions (cell_idx, pos) for indexing
        std::vector<std::pair<uint32_t, llama_pos>> used_cells;
        used_cells.reserve(used);
        for (uint32_t i = 0; i < (uint32_t)cells.size(); ++i) {
            if (!cells[i].is_empty()) {
                used_cells.push_back({i, cells[i].pos});
            }
        }
        // Sort by position (ascending) so logical index == position index
        std::sort(used_cells.begin(), used_cells.end(),
                  [](const auto & a, const auto & b){ return a.second < b.second; });

        // Build set of keep indices for O(1) lookup
        std::unordered_set<uint32_t> keep_set(keep_positions.begin(), keep_positions.end());

        bool changed = false;
        for (uint32_t logical_idx = 0; logical_idx < (uint32_t)used_cells.size(); ++logical_idx) {
            if (keep_set.find(logical_idx) == keep_set.end()) {
                // Evict this cell
                uint32_t cell_idx = used_cells[logical_idx].first;
                cells[cell_idx].seq_id.clear();
                cells[cell_idx].pos = -1;
                cells[cell_idx].delta = 0;
                used--;
                changed = true;
            }
        }

        if (changed) {
            do_defrag = true;  // Schedule defrag on next decode
        }
        return changed;
    }
};

// MOVED 2026-09-09, slice 26, to src/llama-adapter.h -- next to llama_adapter_cvec, the
// upstream-shaped sibling it coexists with. It had to move because src/llama-graph.h needs to name
// this type in llm_graph_params, and llama-graph.h cannot include llama-context.h: this header
// already includes llama-graph.h, so that would be a cycle. llama-adapter.h is included by
// llama-graph.h at line 6, and this header still sees the type transitively through it.
// MOVED, not copied -- a second definition would be an ODR violation across translation units that
// no compiler here would diagnose.

// S69 #66 KVFlash: forward-declared so the pager member is held without pulling the pager header (and its
// ggml-backend deps) into this widely-included file. The full type is included where ~llama_context is defined.
#include "kvflash_ref/kvflash_pager.h"   // S69 #66 full KvFlashPager (unique_ptr dtor needs complete type)
#include "llama-kv-xattn-op.h"           // XAttention prefill block-selection op config (llama_context::xattn_op)
namespace dflash { namespace common { class KvFlashPager; } }

struct llama_context {

    llama_context(const llama_model & model);

    ~llama_context();

    const struct llama_model & model;

    // VALUE-INITIALISED, 2026-09-14, and the `= {}` is load-bearing. llama_cparams has a
    // std::vector member and seven default member initialisers, so it is not an aggregate-POD:
    // a bare `cparams;` is DEFAULT-initialisation, which leaves the other 84 of its 91 fields
    // INDETERMINATE. The two-argument constructor assigns most of them from llama_context_params,
    // which is why this mostly worked -- but it does not assign mtp_op_type, and nothing else does
    // (the MTP_OP_NONE at llama-context.cpp:4255 sets llama_context_params, a different struct).
    //
    // MEASURED: that field printed as 546 in one context and -997342176 in another within a single
    // run. llama_kv_cache_init's per-layer loop opens with
    //     if (cparams.mtp_op_type != MTP_OP_NONE && i < n_mtp_first_layer) { push nullptr; continue; }
    // so garbage there made EVERY layer skip: k_l held 30 nulls, no tensor was created, no backend
    // buffer was allocated, and the graph builder faulted indexing it.
    //
    // `= {}` gives the seven fields that have default member initialisers THEIR values and
    // zero-initialises the rest; zero is MTP_OP_NONE. Nothing the constructor assigns is affected.
    // Adding per-field defaults to all 84 in llama-cparams.h would also help callers that build a
    // llama_cparams on the stack, but it means selecting a semantic default 84 times, and a
    // `false` where `true` was intended is a silent behaviour change. That is separate, deliberate
    // work -- not a side effect of this fix.
    struct llama_cparams        cparams = {};

    // ADDED 2026-09-08. Two accessors upstream's src/llama-kv-cache.cpp calls on llama_context.
    // Both are pure returns of members that already exist on this struct -- `cparams` immediately
    // above, and `sched` further down -- so nothing changes shape or behaviour for any existing
    // caller, all of which keep using the public fields directly.
    //
    // These are 2 of the 7 accessors that file wants. The other 5 are deliberately NOT added, and
    // the reason is recorded here because writing them is easy and would be WRONG:
    //   get_gf_res_reserve() / graph_compute() -- this struct owns no llm_graph_result and computes
    //     graphs through free functions in llama.cpp. Adding them is the llama_context migration
    //     (S69-WIRE66-CONTEXT-MEMORY-ARCH-DECISION, 196 errors in llama-context.cpp), not a
    //     one-liner, and doing it here would land that migration accidentally and out of order.
    //   llama_model::get_rope_freq_base / _freq_scale / _factors -- MEASURED: 4 architectures pick
    //     the per-layer rope frequency conditionally, and deepseek4's condition is not a function
    //     of (cparams, il) at all. src/graphs/build_deepseek4.cpp:996-998 selects on
    //     `use_compress_rope = ratio != 0`, where `ratio` is computed during graph construction;
    //     build_diffusion_gemma.cpp:323 selects on `is_sliding`; build_gemma3.cpp and
    //     build_gemma4.cpp likewise. The obvious body `return cparams.rope_freq_base;` would
    //     compile, would remove three errors, and would silently hand deepseek4 the wrong rope base
    //     whenever compression is on -- a behaviour change wearing a build fix's clothes.
    const struct llama_cparams & get_cparams() const { return cparams; }
    ggml_backend_sched_t         get_sched()   const { return sched;   }

    // ADDED 2026-09-08, slice 4. Slice 1 REFUSED this one, correctly at the time: its body called
    // `sched`, so declaring it would not have compiled. That was a statement about the BODY,
    // not about the method -- and the body's assumption was simply wrong for this fork, whose
    // `sched` is a raw ggml_backend_sched_t (line 615 below) and whose own BUILT code already
    // passes it directly (src/llama.cpp:1869, :5389). With both call sites corrected to the in-tree
    // idiom the declaration is honest, and it is the largest single cluster left: 24 of the 72
    // "has no member named" errors are C-API entry points calling ctx->synchronize().
    // Declaration only -- llama-context.cpp's out-of-line body satisfies it.
    // ADDED 2026-09-09, slice 9. The two members the get_*/set_* wrapper family has been
    // blocked on since slice 1, where they were refused as "genuinely absent" -- correctly, at the
    // time: this fork carries neither a raw pointer nor a size for either.
    //
    // SHAPES READ FROM THE CALL SITES, not chosen. embd_nextn is used as `.data` and `.size`
    // WITHOUT parentheses, which is buffer_view<T>'s field API. embd_layer_inp is used as
    // `.resize(hparams.n_layer() + 1)`, `.size()`, `[lid].data` and `[lid].has_data()`, which is a
    // CONTAINER of buffer_views. The parentheses are what distinguish a view from a vector.
    //
    // ADDITIVE, which is why this is not option A: retyping an EXISTING member (logits ->
    // buffer_view) breaks 23 sites across two BUILT files, measured. Adding a member that did not
    // exist breaks nothing -- src/llama.cpp references neither name.
    //
    // Declaring the family on top of these still costs +21 today, so it is deliberately NOT done
    // here: the remaining dependency is a `sampled` output struct (sampled / probs / candidates /
    // logits and their *_count), which is slice 10's work.
    buffer_view<float>              embd_nextn;
    std::vector<buffer_view<float>> embd_layer_inp;
    std::vector<int32_t> embd_batch_idxs; // upstream (merge 2026-11-04): extracted index -> original batch index

    // ADDED 2026-09-09, slice 11. The eight members the migrated bodies in llama-context.cpp
    // address by bare name, ported from 279b4bfaf^2:src/llama-context.h (line numbers in the
    // comments). Each was confirmed ABSENT from this header before being added.
    //
    // These are dependencies, not features: 28 method bodies in this file are still masked behind
    // "no declaration matches", and their contents reference exactly these names. Declaring those
    // methods before supplying these would price them the way the wrapper family was priced cold
    // (+40); supplying first is what turned that same family into -27.

    // up:337-342. The pair is a unit -- output_reorder() builds swaps then replays them.
    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };
    std::vector<swap_info> output_swaps;

    // up:346. Set when the graph must be re-reserved before the next compute.
    bool sched_need_reserve = true;

    // up:352 (training). ggml-opt.h is included above for this type; it was the only one of the
    // eight not already reachable from this header.
    ggml_opt_context_t opt_ctx = nullptr;

    // up:363-365. Pointers and buffer types for each backend's compute buffer, plus the expected
    // sizes -- upstream's own comment calls them "expected buffer sizes".
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size;

    // up:374. Per-sequence device memory copies. llama_memory_buffers already exists at line 56 of
    // this header (supplied by slice 3), so this is declarable today without a new type.
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    // up:393. mutable because perf accounting updates it from const paths.
    mutable int32_t n_reused = 0;

    // ADDED 2026-09-09, slice 14. Nine members the migrated bodies use by bare name,
    // ported from 279b4bfaf^2:src/llama-context.h. Every one was confirmed ABSENT from this
    // header first, and every type was confirmed PRESENT in this fork first.
    //
    // These were invisible until slice 13 declared the 28 masked methods: gcc stops
    // semantic-checking a member function after its first "no declaration matches", so nothing
    // inside those bodies appeared in any error count. That discarded state was the measurement.

    // up:285, 331. The adapter set and the batch allocator; the constructor's init list names
    // both, which is why slice 13 reported "class llama_context does not have any field named".
    llama_adapter_loras_ptr             loras;
    // per-sequence adapters + the epoch of the context-wide set. Both epochs go into the graph-reuse key by value
    // (see llm_graph_params), so a graph that holds tensors of an adapter that has since been freed is never reused.
    llama_seq_lora_state                seq_loras;
    uint64_t                            loras_epoch = 0;
    std::unique_ptr<llama_batch_allocr> balloc;

    // up:287. Upstream's own comment calls this temporary: "tmp for handling cross-attention --
    // need something better probably". Kept by VALUE as upstream has it, which is why
    // llama-graph.h must be included above for the complete type.
    llama_cross cross;

    // up:367-368. Previous and reserve graph results, used by the graph-reuse path.
    std::array<llm_graph_result_ptr, 2> gf_res_prev; // [has outputs], see get_gf_res_prev()
    llm_graph_result_ptr gf_res_reserve;
    llm_graph_result * gf_res_prev_active = nullptr; // the arena whose graph the scheduler holds now

    // up:360. Backends that can be told a thread count, paired with the setter for each.
    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // up:354-355.
    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    // up:379. env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    void synchronize();

    // ADDED 2026-09-09, slice 15. Declarations for the member functions whose bodies already
    // exist in src/llama-context.cpp. Until now they were MASKED: gcc stops semantic-checking
    // a member function after its first "no declaration matches", so nothing inside them
    // appeared in any error count, and no slice could see what they needed.
    //
    // Signatures are gcc's own, quoted from those diagnostics. Declaring this family cost +47
    // and broke all four BUILT files when it was tried cold in slice 13. It is a NET WIN now
    // only because slices 11 and 14 supplied what these bodies use -- the same price curve the
    // wrapper family showed: +40 declared cold, -27 declared once supported.
    //
    // TWO RETURN TYPES NEEDED REPAIR, and both are name-resolution problems rather than
    // missing functionality:
    //   pooling_type()     `enum` is required -- the bare name llama_pooling_type resolves to
    //                      the FUNCTION at include/llama.h:913, not the enum at :293.
    //   memory_breakdown() returns llama_memory_breakdown, an alias in src/llama-ext.h, which
    //                      this header now includes (added by slice 14).
    // Those two were also the only 2 of 28 whose harvested signature failed to satisfy its
    // definition. One cause, two symptoms.
    bool adapters_lora_are_same(llama_adapter_lora**, size_t, float*);
    bool memory_update(bool);
    bool set_adapter_cvec(const float*, size_t, int32_t, int32_t, int32_t);
    bool set_sampler(llama_seq_id, llama_sampler*);
    // slice 20: the two DEFAULT ARGUMENTS restored. Slice 15 declared this method by
    // harvesting gcc's signature text from a "no declaration matches" diagnostic, and that
    // text reports the DEFINITION's signature -- which never carries defaults, because in
    // C++ they live on the declaration. Six callers rely on them. Values are upstream's own,
    // verbatim from 279b4bfaf^2:src/llama-context.h:251-252.
    // probe_devices/host_kept: when both are set, llama_host_kept_token_counts reads the graph before it is reserved
    ggml_cgraph* graph_reserve(uint32_t, uint32_t, uint32_t, const llama_memory_context_i*, bool = false, size_t* = nullptr,
                               const std::vector<ggml_backend_t>* probe_devices = nullptr, std::vector<int>* host_kept = nullptr);
    ggml_status graph_compute(ggml_cgraph*, bool);
    int decode(const llama_batch&);
    int encode(const llama_batch&);
    // upstream (merge 2026-11-04, batch 28): the llama_batch_ext entry points, defined in llama-context.cpp; the llama_batch
    // overloads above keep the fork's legacy route and convert for the rest
    int decode(const llama_batch_ext &);
    int encode(const llama_batch_ext &);
    llama_context(const llama_model&, llama_context_params);
    llama_memory_breakdown memory_breakdown() const;
    enum llama_pooling_type pooling_type() const;
    llm_graph_cb graph_get_cb() const;
    llm_graph_params graph_params(llm_graph_result*, const llama_ubatch&, const llama_memory_context_i*, llm_graph_type) const;
    llm_graph_result* get_gf_res_reserve() const;
    llm_graph_result* process_ubatch(const llama_ubatch&, llm_graph_type, llama_memory_context_i*, ggml_status&);
    size_t state_seq_set_data(llama_seq_id, const uint8_t*, size_t, llama_state_seq_flags);
    uint32_t graph_max_nodes(uint32_t) const;
    uint32_t output_reserve(int32_t);
    void attach_threadpool(ggml_threadpool_t, ggml_threadpool_t);
    bool get_causal_attn() const;
    void detach_threadpool();
    bool extract_layer_inputs(const llm_graph_result*, size_t, size_t); // upstream: true when it extracted any
    void opt_epoch(ggml_opt_dataset_t, ggml_opt_result_t, ggml_opt_result_t, int64_t, ggml_opt_epoch_callback, ggml_opt_epoch_callback);
    void opt_epoch_iter(ggml_opt_dataset_t, ggml_opt_result_t, const std::vector<int>&, const std::vector<int>&, llama_batch&, ggml_opt_epoch_callback, bool, int64_t, int64_t, int64_t);
    void opt_init(llama_model*, llama_opt_params);
    void resolve_fused_ops(const llama_memory_context_i*, uint32_t);
    void sched_reserve();
    void set_abort_callback(bool (*)(void*), void*);
    void set_adapters_lora(llama_adapter_lora**, size_t, float*);
    void set_seq_adapters_lora(llama_seq_id seq_id, llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    // slice 10 scratch: family declarations
    float* get_embeddings();
    float* get_embeddings_ith(int32_t);
    float* get_embeddings_layer_inp(uint32_t);
    float* get_embeddings_nextn();
    float* get_embeddings_nextn_ith(int32_t);
    float* get_logits();
    float* get_logits_ith(int32_t);
    size_t get_sampled_candidates_count(int32_t);
    const llama_token* get_sampled_candidates_ith(int32_t);
    size_t get_sampled_logits_count(int32_t);
    float* get_sampled_logits_ith(int32_t);
    size_t get_sampled_probs_count(int32_t);
    float* get_sampled_probs_ith(int32_t);
    llama_token get_sampled_token_ith(int32_t);
    llama_token* get_sampled_tokens() const;
    void output_reorder();
    void set_embeddings(bool);
    void set_embeddings_layer_inp(uint32_t, bool);

    // ADDED 2026-09-08, slice 6. Re-derived against the 101-error baseline rather
    // than carried forward from slice 5: the candidate set fell 68 -> 49 and these 3
    // became viable only after slice 5's declarations stopped masking their bodies.
    // state_write_data and state_read_data are the exact names slice 5's category
    // arithmetic predicted would surface, so this slice tests that prediction.
    // Signatures are the compiler's own; declaration-only, bodies already exist.
    size_t state_seq_get_data(llama_seq_id, uint8_t*, size_t, llama_state_seq_flags);
    size_t state_write_data(llama_io_write_i&);
    size_t state_read_data(llama_io_read_i&);

    // ADDED 2026-09-08, slice 5. Each of these 19 was tested ALONE by
    // temp/_probe_slice5_candidates.py: declaring it reduced the error count AND removed its
    // own 'no declaration matches'. Eleven are the state_* family, which only became
    // declarable once slice 3 supplied llama_memory_buffers -- the payoff of that slice, and
    // the reason this set is re-derived after every landing rather than fixed in advance.
    // Signatures are the compiler's own, not retyped: a hand-typed const or reference that
    // differs produces a second unmatched overload instead of a fix.
    // Declaration-only; llama-context.cpp's existing out-of-line bodies satisfy them.
    float* get_embeddings_seq(llama_seq_id);
    void set_n_threads(int32_t, int32_t);
    void set_embeddings_nextn(bool, bool);
    void set_nextn_layer_offset(int32_t);
    void set_warmup(bool);
    int64_t output_resolve_row(int32_t) const;
    void set_causal_attn(bool);
    size_t state_get_size();
    size_t state_get_data(uint8_t*, size_t);
    size_t state_set_data(const uint8_t*, size_t);
    size_t state_seq_get_size(llama_seq_id, llama_state_seq_flags);
    bool state_load_file(const char*, llama_token*, size_t, size_t*);
    bool state_save_file(const char*, const llama_token*, size_t);
    size_t state_seq_load_file(llama_seq_id, const char*, llama_token*, size_t, size_t*);
    size_t state_seq_save_file(llama_seq_id, const char*, const llama_token*, size_t);
    size_t state_seq_write_data(llama_io_write_i&, llama_seq_id, llama_state_seq_flags);
    size_t state_seq_read_data(llama_io_read_i&, llama_seq_id, llama_state_seq_flags);
    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    // ADDED 2026-09-08, S69-WIRE66-CONTEXT-MEMORY-ARCH-DECISION slice 1. Eight more pure-return
    // accessors src/llama-context.cpp (upstream's, unbuilt) calls on llama_context. Declaration
    // only (no inline body) -- llama-context.cpp's own out-of-line bodies for these eight already
    // do exactly `return cparams.<field>;` / `return model;` against fields that already exist
    // (llama-cparams.h: n_ctx, n_ctx_seq, n_batch, n_ubatch, n_seq_max, n_threads, n_threads_batch;
    // `model` at this struct's own top). Declaring here lets those existing bodies satisfy the
    // declaration with zero duplication, unlike get_cparams()/get_sched() above which had to be
    // given inline bodies HERE because llama-kv-cache.cpp needed them before llama-context.cpp's
    // own out-of-line versions were reachable -- that is also why this commit deletes
    // llama-context.cpp's now-redundant out-of-line get_cparams()/get_sched(), which redefined
    // these two and would not otherwise stop erroring.
    const struct llama_model & get_model() const;
    uint32_t n_ctx()           const;
    uint32_t n_ctx_seq()       const;
    uint32_t n_batch()         const;
    uint32_t n_ubatch()        const;
    uint32_t n_seq_max()       const;
    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    // ADDED slice 2, S69-WIRE66-CONTEXT-MEMORY-ARCH-DECISION. Ninth pure-return accessor,
    // same shape/reasoning as the eight above: llama-context.cpp:785-787's out-of-line body
    // is exactly `return memory.get();` against the `memory` member this slice adds below.
    // Declaring it here is what makes the member measurably reachable (not a hollow add) --
    // it also resolves the two free-function call sites (llama_get_memory, decode()'s
    // internal ctx->get_memory() use) that were erroring on the missing METHOD, not the
    // missing member; the member alone did not move llama-context.cpp's error count (see
    // ledger entry 6 -- 175->175 on the member add alone, 175->172 with this declaration).
    llama_memory_t get_memory() const;
    struct llama_sampling       sampling;
    struct llama_kv_cache_fork       kv_self;
    struct llama_kivi_sim *          kivi_sim = nullptr; // --kivi-bits: simulated KIVI KV quantisation (null = off)
    // S69-WIRE66-CONTEXT-MEMORY-ARCH-DECISION slice 2: polymorphic memory member (upstream
    // PR #11213 shape: llama_memory_ptr = std::unique_ptr<llama_memory_i>). ADDED ALONGSIDE
    // kv_self, not repointing it -- kv_self stays the fork's live KV path for its 411 existing
    // member-access sites (out of scope, untouched). `memory` is a real polymorphic slot: no
    // model in this fork constructs it yet (no model.create_memory() override exists), so it
    // stays nullptr at runtime until a future slice wires a llama_memory_i-derived cache into
    // it. Declaring it here unblocks llama-context.cpp's own out-of-line bodies that already
    // read/write it (memory->init_full(), memory.reset(...), memory.get(), etc.) -- this slice
    // does not add behaviour, it lets already-written, previously-uncompilable code compile.
    llama_memory_ptr                 memory;
    struct llama_context      * mtp_target_ctx   = nullptr;
    struct llama_control_vector cvec;

    std::vector<float> scale_data;

    std::unordered_map<struct llama_lora_adapter *, float> lora_adapters;

    std::vector<ggml_backend_t> backends;
    std::unique_ptr<llama_moe_cache> moe_cache;
#ifdef GGML_USE_METAL
    ggml_backend_t backend_metal = nullptr;
#endif
    ggml_backend_t backend_cpu = nullptr;

    bool has_evaluated_once = false;

    // S69 #22 SnapKV/H2O: aggregate post-softmax kq per-KV-position attention mass captured during
    // prefill (eval callback), used to drop low-attention prompt positions when snapkv_cap>0. Opt-in;
    // empty/inactive => default path untouched. Requires the non-flash attention graph (kq materialized).
    std::vector<float> snapkv_obs;
    bool               snapkv_active = false;

    // S69 #31 EAGLE3 target hidden-state capture: when eagle3_capture is set (by the spec-decode loop on the
    // TARGET ctx), the decoder graph tags the residual stream at 3 layers (e3_lo/mid/hi) and an eval callback
    // copies the LAST token's hidden into eagle3_feats at the matching 1/3 offset -> [3*n_embd] concat fed to
    // the EAGLE3 draft (llama_get_eagle3_target_feats). Opt-in; default OFF => live serving path is identical.
    std::vector<float> eagle3_feats;     // sized 3*n_embd when capturing; layout [lo | mid | hi]
    bool               eagle3_capture = false;
    int                e3_lo  = -1;
    int                e3_mid = -1;
    int                e3_hi  = -1;
    // S69 #34 EAGLE-3 recurrence (arxiv 2503.01840 §3.1): step 1 uses g = FC(3 target feats); steps 2+ replace
    // g with the draft's OWN previous-step hidden `a` (BYPASS fc). eagle3_g_direct, when non-null, makes
    // build_eagle3 take g from a direct n_embd input instead of fc(feats). eagle3_capture_hidden makes the
    // draft decode copy its last-token l_out (= `a`) into eagle3_draft_hidden for the next step.
    const float *      eagle3_g_direct = nullptr;          // [n_embd] previous draft hidden, or null = feats path
    bool               eagle3_capture_hidden = false;      // capture the draft's l_out into eagle3_draft_hidden
    std::vector<float> eagle3_draft_hidden;                // [n_embd] last draft-step hidden `a`
    int                eagle3_n_embd_tgt = 0;              // target n_embd (set by llama_eagle3_set_target); feats = 3*this
    // S69 Stage-1 API 3/4: embeddings_nextn equivalent — capture the draft pre-norm hidden ("eagle3_a") for ALL
    // rows of a decode (recurrence source for steps 2+ and, post enc/dec split, the encoder g_embd output).
    bool               eagle3_nextn_capture = false;       // when set, the eval-callback copies "eagle3_a" rows here
    std::vector<float> eagle3_nextn;                       // [n_rows * n_embd] last decode's pre-norm hidden, row-major
    int                eagle3_nextn_rows = 0;              // n_rows captured
    // S69 Stage-2: PER-POSITION 3-layer feats buffer (was last-token only in eagle3_feats). Row i = [lo|mid|hi]
    // for token i in the last target ubatch. process() reads this to build draft KV via (token[k+1], feats[k]).
    std::vector<float> eagle3_feats_pp;                    // [n_rows * 3*n_embd_tgt] row-major, interleaved [lo;mid;hi]
    int                eagle3_feats_pp_rows = 0;
    struct ggml_tensor * inp_eagle3_g_direct = nullptr;    // the build_eagle3 direct-g input (steps 2+)

    // S69 #66 KVFlash: paged-KV pager, instantiated only when cparams.kvflash_pool_tokens > 0 (else nullptr =
    // default ring-buffer KV path, untouched). attach()/slot_for()/fill_slot_mask()/reselect() wired in
    // subsequent increments (the pager LOGIC is validated host-side, llama-kvflash.selftest PASS).
    std::unique_ptr<dflash::common::KvFlashPager> kvflash_pager;

    // XAttention prefill block selection (--xattn-threshold; row S69-openvino-PICK-XATTENTION-PREFILL-BLOCK-SELECT-2026-09-29-01).
    // Filled from cparams at context creation; its address is the userdata of the xattn custom op, so it lives as long as the graphs do.
    justician_kv::xattn_op_cfg xattn_op;

    int64_t t_start_us;
    int64_t t_load_us;
    int64_t t_p_eval_us = 0;
    int64_t t_eval_us   = 0;

    int64_t t_compute_start_us = 0;
    int64_t n_queued_tokens = 0;

    int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    int32_t n_eval   = 0; // number of eval calls

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_t buf_output = nullptr;

    // the previous graph result for this batch's output kind (with / without outputs), created on first use;
    // two arenas give the two kinds distinct CUDA graph cache keys (upstream llama.cpp, merged 2026-11-03)
    llm_graph_result * get_gf_res_prev();

    // ggml_backend_sched realloc callback, counts and traces the graph reallocations of this context
    static void sched_graph_realloc(ggml_backend_sched_t sched, ggml_cgraph * gf, bool backend_ids_changed, bool graph_size_changed, void * user_data);

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    size_t  logits_size = 0; // capacity (of floats) for logits
    float * logits      = nullptr;

    // Vocabulary-restricted output projection (llama_set_output_vocab_subset)
    struct output_vocab_subset_state {
        std::vector<int32_t>  ids;              // sorted, unique; empty = off
        ggml_context *        ctx     = nullptr;
        ggml_backend_buffer_t buf     = nullptr;
        ggml_tensor *         partial = nullptr; // same type as model.output, [n_embd, ids.size()]
        bool                  in_graph = false;  // the graph that was built last multiplies by `partial`
        bool                  allowed  = false;  // set while llama_build_graph builds a real (not worst-case) graph
        int32_t               used     = 0;      // ids.size() when the last decode used it, else 0
        std::vector<float>    scratch;
    } output_subset;
    void clear_output_subset();

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers
    size_t  output_size = 0; // capacity (of tokens positions) for the output buffers
    int32_t n_outputs   = 0; // number of actually-used outputs in the current ubatch or last logical batch
    int32_t n_outputs_embd = 0; // number of embedding rows produced for the current logical batch

    bool logits_all = false;

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    size_t  embd_size = 0; // capacity (of floats) for embeddings
    float * embd      = nullptr;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // whether we are computing encoder output or decoder output
    bool is_encoding = false;

    // output of the encoder part of the encoder-decoder models
    std::vector<float> embd_enc;
    std::vector<std::set<llama_seq_id>> seq_ids_enc;
    uint32_t n_input_tensors = 0; // number of tensors marked as input during the last graph reserve

    // memory buffers used to evaluate the model
    std::vector<uint8_t> buf_compute_meta;
    ggml_backend_sched_t sched = nullptr;

    static bool sched_copy_moe_cache(ggml_backend_t backend, const ggml_tensor * src,
            ggml_tensor * dst, ggml_cgraph * graph, ggml_backend_sched_copy_phase phase, void * user_data);

    // whether the reserve graph has a logits tensor, set in sched_reserve()
    // true until then, because the first output_reserve() runs before any graph is reserved
    bool graph_has_logits = true;

    // state of the large-batch expert copy in sched_copy_moe_cache, reset before each graph compute
    struct copy_experts_info {
        const ggml_tensor *  ids = nullptr;
        std::vector<int32_t> ids_data;
        std::vector<bool>    used;

        void reset() {
            ids = nullptr;
            ids_data.clear();
            used.clear();
        }
    };

    copy_experts_info copy_experts;

    // graph reallocations during inference where the backend assignments did not change, split by whether the
    // graph size is the same as the previously scheduled one (unexpected) or different
    int32_t n_reallocs_same = 0;
    int32_t n_reallocs_diff = 0;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    const float * draft_input_hidden_state = nullptr;
    size_t draft_input_hidden_state_n_floats = 0;
    std::vector<float> draft_input_hidden_state_owned;

    struct dflash_runtime {
        struct target_window_state {
            const float * features = nullptr;
            size_t features_n_floats = 0;
            int32_t features_n_rows = 0;
            const float * append_features = nullptr;
            size_t append_features_n_floats = 0;
            int32_t append_features_n_rows = 0;
            const llama_pos * positions = nullptr;
            size_t positions_n = 0;
            uint64_t version = 0;
            int32_t keep_rows = 0;
            int32_t append_rows = 0;
            bool replace = false;
            std::vector<float> features_owned;
            std::vector<float> append_features_owned;
            std::vector<llama_pos> positions_owned;
            std::vector<float> features_padded;
            std::vector<llama_pos> pos_ctx_data;
            std::vector<float> kq_mask_data;
            std::vector<float> kq_mask_swa_data;
        };

        struct kv_runtime_state {
            std::vector<struct ggml_tensor *> k_ctx_cache;
            std::vector<struct ggml_tensor *> v_ctx_cache;
            struct ggml_context * cache_ctx = nullptr;
            std::vector<ggml_backend_buffer_t> cache_bufs;
            std::vector<llama_pos> cache_pos;
            std::vector<uint8_t> cache_slot_valid;
            int32_t cache_write_pos = 0;
            int32_t cache_n_filled = 0;
            int32_t cache_update_rows = 0;
            int32_t cache_reserved_rows = 0;
            int32_t cache_view_write_pos = 0;
            int32_t cache_view_n_filled = 0;
            uint64_t cache_applied_window_version = 0;
            bool cache_valid = false;
            bool cache_view_valid = false;
            std::vector<uint8_t> cache_compute_meta;
            ggml_backend_sched_t cache_sched = nullptr;
            ggml_cgraph * cache_graph = nullptr;
            int32_t cache_graph_rows = 0;
            int32_t cache_graph_write_pos = 0;
            struct ggml_tensor * cache_input_target_features = nullptr;
            struct ggml_tensor * cache_input_pos_ctx = nullptr;
            struct ggml_tensor * cache_input_rows = nullptr;
            struct ggml_tensor * kq_mask_tensor = nullptr;
            struct ggml_tensor * kq_mask_swa_tensor = nullptr;
            struct ggml_tensor * draft_tail_rows_tensor = nullptr;
        };

        struct capture_state {
            std::vector<int32_t> layer_ids;
            std::vector<std::vector<float>> layer_rows;
            struct capture_chunk {
                int32_t row_offset = 0;
                int32_t row_count = 0;
                size_t byte_offset = 0;
                ggml_type type = GGML_TYPE_COUNT;
            };
            std::vector<std::vector<capture_chunk>> layer_chunks;
            std::vector<int32_t> layer_rows_written;
            int32_t row_count = 0;
            int32_t row_width = 0;
            int32_t expected_rows = 0;
            uint64_t capture_batch_id = 0;
            std::vector<uint64_t> layer_seen_batch_id;
            bool readback_pending = false;
            bool invalid = false;
            ggml_backend_sched_eval_callback prev_cb_eval = nullptr;
            void * prev_cb_eval_user_data = nullptr;
        };

        struct input_state {
            struct ggml_tensor * target_features = nullptr; // F32 [n_target_features, cross_ctx]
            struct ggml_tensor * pos_ctx = nullptr;         // I32 [cross_ctx]
            struct ggml_tensor * kq_mask = nullptr;         // F32 [cross_ctx + n_batch, GGML_PAD(n_batch)]
            struct ggml_tensor * kq_mask_swa = nullptr;     // F32 [cross_ctx + n_batch, GGML_PAD(n_batch)]
        };

        target_window_state target;
        kv_runtime_state kv;
        std::unique_ptr<capture_state> capture;
        std::vector<float> feature_view_buffer;
        input_state inputs;
        int32_t visible_cross_ctx = 0;
        bool dspark = false;
        bool want_dspark_logits = false;

        std::vector<llama_token> draft_tokens;
        struct ggml_tensor * draft_tokens_tensor = nullptr;
        struct ggml_tensor * draft_lattice_tensor = nullptr;
        struct ggml_tensor * draft_lattice_ids_tensor = nullptr;
        std::vector<float> draft_lattice;
        std::vector<int32_t> draft_lattice_ids;
        int32_t draft_lattice_top_k = 0;
    };
    dflash_runtime dflash;
    using dflash_capture_state = dflash_runtime::capture_state;

    // S69-KT-EXPERT-POOL-2026-09-28 unit 1: MoE expert-activation recorder, wired into the real
    // decode graph's cb_eval path (llama-context.cpp constructor + llama_moe_record_eval_callback).
    // Chain-of-responsibility fields mirror dflash_runtime::capture_state::prev_cb_eval above --
    // same pattern, so a caller-supplied params.cb_eval (imatrix, perplexity, server debug tools)
    // still fires. moe.recorder is non-null only when cparams.moe_stats_path is non-empty (the
    // --moe-expert-stats FNAME flag); nullptr means the wrapper callback is a pure pass-through.
    // pool_layers is reserved for unit 2's graph splice (llm_build_lora_mm_id); NOT populated or
    // consulted by anything in unit 1 -- see llama-moe-pool.h "GRAPH SPLICE STATUS".
    struct moe_runtime {
        std::unique_ptr<llama_moe_recorder> recorder;
        ggml_backend_sched_eval_callback prev_cb_eval = nullptr;
        void * prev_cb_eval_user_data = nullptr;
        std::vector<llama_moe_pool_layer> pool_layers; // unreferenced in unit 1
    };
    moe_runtime moe;
    // S69 #34: the EAGLE3 draft head has no token embeddings of its own — it reuses the TARGET model's tok_embd.
    // The spec loop sets this (llama_eagle3_set_target) so build_eagle3 can embed draft tokens. nullptr = none.
    const struct ggml_tensor * draft_external_tok_embd = nullptr;

    // true after the prism.hadamard coverage check passes on a reserved graph
    bool hadamard_verified = false;

    struct dsv4_runtime {
        static constexpr uint32_t CSA_RATIO = 4;
        static constexpr uint32_t HCA_RATIO = 128;

        struct slot_info {
            int32_t s0 = 0;
            int32_t s1 = 0;
            std::vector<llama_seq_id> strm;
            std::vector<std::vector<uint32_t>> idxs;

            void resize(size_t n) {
                strm.resize(n);
                idxs.resize(n);
            }

            size_t size() const {
                GGML_ASSERT(strm.size() == idxs.size());
                if (idxs.empty()) {
                    return 0;
                }

                return idxs[0].size();
            }

            size_t n_stream() const {
                GGML_ASSERT(strm.size() == idxs.size());
                return strm.size();
            }

            bool empty() const {
                return idxs.empty();
            }
        };

        struct raw_context {
            std::vector<int32_t> write_src_idxs;
            std::vector<int32_t> write_dst_idxs;
            std::vector<int32_t> read_dst_idxs;
            std::vector<int32_t> write_counts;
            std::vector<int32_t> read_counts;
            slot_info sinfo_write;
            slot_info sinfo_read;
            int64_t graph_n_stream = 1;
            int64_t n_kv = 0;
        };

        struct comp_context {
            slot_info sinfo;
            int64_t graph_n_stream = 1;
            int64_t n_kv = 0;
        };

        struct comp_plan {
            std::vector<int32_t> state_pos;
            std::vector<int32_t> state_delta_src_idxs;
            std::vector<int32_t> state_delta_dst_idxs;
            std::vector<int32_t> state_persist_src_idxs;
            std::vector<int32_t> state_persist_dst_idxs;
            std::vector<int32_t> state_read_idxs;
            std::vector<int64_t> state_write_idxs;
            std::vector<int64_t> state_write_idxs_lid;
            std::vector<int32_t> state_write_pos;
            std::vector<int32_t> n_visible;
            int64_t n_stream = 1;
            int64_t n_kv = 0;
        };

        struct comp_inputs {
            struct ggml_tensor * state_pos = nullptr;
            struct ggml_tensor * state_persist_src_idxs = nullptr;
            struct ggml_tensor * state_persist_dst_idxs = nullptr;
            struct ggml_tensor * state_read_idxs = nullptr;
            struct ggml_tensor * state_write_idxs = nullptr;
            struct ggml_tensor * state_write_idxs_lid = nullptr;
            struct ggml_tensor * cand_pin = nullptr;
            struct ggml_tensor * state_write_pos = nullptr;
            struct ggml_tensor * kq_mask = nullptr;
            struct ggml_tensor * n_visible = nullptr;
        };

        struct storage {
            std::vector<struct ggml_tensor *> csa_k;
            std::vector<struct ggml_tensor *> hca_k;
            std::vector<struct ggml_tensor *> lid_k;

            std::vector<struct ggml_tensor *> csa_state_kv;
            std::vector<struct ggml_tensor *> csa_state_score;
            std::vector<struct ggml_tensor *> hca_state_kv;
            std::vector<struct ggml_tensor *> hca_state_score;
            std::vector<struct ggml_tensor *> lid_state_kv;
            std::vector<struct ggml_tensor *> lid_state_score;

            struct ggml_tensor * cand_topk = nullptr;

            struct ggml_context * cache_ctx = nullptr;
            std::vector<ggml_backend_buffer_t> cache_bufs;
            uint32_t n_stream = 1;
        };

        struct input_state {
            struct ggml_tensor * raw_k_write_src_idxs = nullptr;
            struct ggml_tensor * raw_k_write_idxs = nullptr;
            struct ggml_tensor * raw_k_read_idxs = nullptr;
            comp_inputs csa;
            comp_inputs hca;
            comp_inputs lid;
        };

        storage cache;
        input_state inputs;
        raw_context raw;
        comp_context csa_ctx;
        comp_context hca_ctx;
        comp_context lid_ctx;
        comp_plan csa_plan;
        comp_plan hca_plan;
        comp_plan lid_plan;
        // compressor state ring -> per-token state rows of the current graph (per-step checkpoint capture)
        std::map<const ggml_tensor *, ggml_tensor *> capture_rows;
        // compressor state rings the current graph persists into
        std::set<const ggml_tensor *> persisted_rings;

        std::vector<float> csa_mask_data;
        std::vector<float> hca_mask_data;

        // the indexer top-k an index source picked, reused by its stream; one graph build only
        struct ggml_tensor * top_k_a = nullptr;
        struct ggml_tensor * top_k_b = nullptr;
    };
    dsv4_runtime dsv4;

    // input tensors
    struct ggml_tensor * inp_tokens;      // I32 [n_batch]
    std::vector<struct ggml_tensor *> inp_engram_rows; // I32 [n_cols*n_batch], one per engram layer
    std::vector<struct ggml_tensor *> inp_engram_gate_ids; // I32 [hc]: 0..hc-1, dequantizes the gate scales via get_rows
    struct ggml_tensor * inp_engram_gate_mask = nullptr; // F32 [n_batch]: 0 at image tokens (engram gate shut), 1 elsewhere
    struct ggml_tensor * inp_embd;        // F32 [n_embd, n_batch]
    struct ggml_tensor * inp_pos;         // I32 [n_batch]
    struct ggml_tensor * inp_ffd_idx = nullptr; // I64 [n_batch] KV cells written this ubatch (--ffd); kv_self.head + i
    struct ggml_tensor * inp_out_ids;     // I32 [n_outputs]
    struct ggml_tensor * inp_KQ_mask;     // F32 [kv_size, n_batch]
    struct ggml_tensor * inp_KQ_mask_swa; // F32 [kv_size, n_batch]
    struct ggml_tensor * inp_KQ_mask_swa_win = nullptr; // F32 [SWA W_view, n_batch]
    // DiffusionGemma host self-conditioning upload: F32 [n_vocab, C] previous-step canvas logits.
    // nullptr when SC is off or when the device-resident SC path is in use (no host round-trip).
    struct ggml_tensor * inp_dg_sc_logits = nullptr;
    struct ggml_tensor * inp_K_shift;     // I32 [kv_size]
    struct ggml_tensor * inp_mean;        // F32 [n_batch, n_batch]
    struct ggml_tensor * inp_cls;         // I32 [n_batch]
    struct ggml_tensor * inp_s_copy;      // I32 [kv_size]
    struct ggml_tensor * inp_s_mask;      // F32 [1, n_kv]
    struct ggml_tensor * inp_s_seq;       // I32 [n_kv, n_batch]
    struct ggml_tensor * inp_s_seq_qnext; // I32 [1, n_batch]
    struct ggml_tensor * inp_ple_rows = nullptr; // I32 [ple_n_heads * n_batch], qwen4exp n-gram rows
    // Inkling (S69 P1). Host-filled per ubatch in llama_set_inputs.
    struct ggml_tensor * inp_inkling_tau         = nullptr; // F32 [1, 1, n_tokens]: log-N query scale of the global layers
    struct ggml_tensor * inp_inkling_rel_idx     = nullptr; // I32 [n_kv, n_tokens]: row of the relative-bias table each (query, cell) reads, global layers
    struct ggml_tensor * inp_inkling_rel_idx_swa = nullptr; // I32 [n_kv, n_tokens]: the same for the sliding-window layers
    struct ggml_tensor * inp_inkling_vocab_mask  = nullptr; // F32 [n_vocab]: 0 for a real token id, -inf for the padded tail
    struct ggml_tensor * inp_inkling_shexp_idx   = nullptr; // I32 [n_shexp, n_tokens]: 0..n_shexp-1 per token, the shared-expert bank ids
    struct ggml_tensor * inp_pos_bucket;    // I32 [n_batch|n_kv, n_batch]
    struct ggml_tensor * inp_embd_enc;      // F32 [n_embd, n_outputs_enc]
    struct ggml_tensor * inp_KQ_mask_cross; // F32 [n_outputs_enc, n_batch]
    struct ggml_tensor * inp_scale = nullptr; // F32 [n_tokens]
    // per-sequence LoRA (llama_set_seq_adapters_lora): one F32 [1, n_tokens] mask per adapter of the union, created by the legacy mat-mul funnel
    // llm_build_context::llm_build_lora_mm and filled by llama_set_inputs(); cleared with the other inputs when a graph is (re)built
    std::vector<std::pair<struct llama_adapter_lora *, struct ggml_tensor *>> lora_seq_masks;
    int64_t lora_seq_n_tokens = 0; // n_tokens of the graph being built (the mask length)

    // position-gated LoRA (adapter.lora.gate): llama_set_draft_start per sequence, and one F32 [1, n_tokens] 0/1 input per gated adapter
    std::map<llama_seq_id, llama_pos> draft_start;
    std::vector<std::pair<struct llama_adapter_lora *, struct ggml_tensor *>> lora_gate_masks;
    // Lily (row S69-ADAPTERS-W2-LILY-PER-SEQUENCE-ROUTER-2026-10-26-01): F32 [n_tokens, n_tokens] same-sequence averaging matrix,
    // [k, j] = 1/n(seq of j) when token k is in j's sequence; created by the first Lily weight of the graph, filled by llama_set_inputs()
    struct ggml_tensor * inp_lily_seq_mean = nullptr;
    struct ggml_tensor * inp_mtp_states = nullptr;
    struct ggml_tensor * inp_mtp_carry = nullptr; // F32 [n_embd, nextn-1] per-head hidden at the last committed position
    struct ggml_tensor * inp_dsa_sink = nullptr; // F32 [n_kv, n_tokens] per-sequence attention-sink boost for DSA indexer top-k
    struct ggml_tensor * inp_kpool_cells     = nullptr; // I32 [kpool*n_pool] cell index of each pool member (pool b, member j at [b*kpool+j])
    struct ggml_tensor * inp_kpool_bias      = nullptr; // F32 [n_pool, n_tokens] 0 if pool complete & visible to query, else -inf
    struct ggml_tensor * inp_kpool_tail      = nullptr; // I32 [kpool-1, n_tokens] trailing incomplete pool cells (null when kpool==1)
    struct ggml_tensor * inp_kpool_ape_slots = nullptr; // I32 [kpool] identity [0..kpool-1], gathers the ape rows in order

    // Qwen sparse attention: everything that depends on cache layout is computed on the host,
    // so the graph only gathers, pools and scores. One entry per distinct compress ratio.
    struct qsa_input {
        int32_t ratio    = 0;
        struct ggml_tensor * cell_blk  = nullptr; // I32 [n_kv]           block each cell belongs to
        struct ggml_tensor * bias      = nullptr; // F32 [n_kv, n_tokens] causal mask plus the tail boost
        // only the blocks this ubatch writes into are pooled again; the rest are read from kp_l
        struct ggml_tensor * win_blocks = nullptr; // I32 [n_win]         which block each entry rebuilds
        struct ggml_tensor * win_cells  = nullptr; // I32 [ratio*n_win]   member cells of those blocks
        struct ggml_tensor * win_pos    = nullptr; // I32 [4*n_win]       mrope position of those blocks
        struct ggml_tensor * head_w     = nullptr; // F32 [n_idx_h, n_tokens] all ones; the head sum is unweighted
        struct ggml_tensor * cut_mask   = nullptr;
    };
    std::vector<qsa_input> inp_qsa;

    // the pooled block keys no longer match the raw indexer keys and the next built graph must
    // pool every block; set on state restore and defrag, cleared by that graph's host fill
    bool qsa_pooled_stale = false;

    // token at each position of a sequence, read by the PLE n-gram hash
    std::map<llama_seq_id, std::vector<llama_token>> ple_hist;

    // per-graph memo of Hadamard-rotated activations (prism.hadamard folded weights):
    // several folded weights consume the same input activation; rotate it once per graph.
    // Keyed by content (signs tensor + transform params), not by the per-weight transform
    // pointer, so weights sharing one rotation reuse a single rotated tensor.
    std::map<std::tuple<const struct ggml_tensor *, const struct ggml_tensor *, int64_t, int64_t, int64_t, int64_t>, struct ggml_tensor *> hadamard_memo;

    struct swa_window_view_state {
        bool active       = false;
        bool compacted    = false;
        int32_t n_kv      = 0;
        int32_t n_tokens  = 0;
        uint32_t window   = 0;
        uint32_t pad      = 0;
        int64_t w_view    = 0;
        int64_t win_off   = 0;
    } swa_window_view;

    // multi-head MTP chaining state: head k's output row at the last committed position,
    // written back after each warmup/update decode and fed into the next MTP graph through
    // inp_mtp_carry (zeroed when a prompt warmup restarts from position 0). The readback is
    // issued async after compute; mtp_carry_pending marks a copy that must be synchronized
    // before the host buffer is read or resized.
    std::vector<float> mtp_carry;
    bool mtp_carry_pending = false;

    std::vector<uint8_t> swa_compact_buf;
    struct ggml_tensor * inp_eagle3_feats = nullptr;  // S69 #33: the build_eagle3 "eagle3_target_feats" input, fed from draft_input_hidden_state

    ggml_backend_t ggml_backend_by_name(const char * name);

    // RELOCATED HERE 2026-09-09, slice 22, from src/llama.cpp:643 -- verbatim, all 14 fields,
    // in their original order and with their original types.
    //
    // It was a forward declaration and the definition sat in src/llama.cpp, which made the type
    // complete in exactly ONE translation unit. std::unique_ptr<Prev> below needs sizeof(Prev) for
    // its default deleter, so every other TU that constructs or destroys a llama_context was
    // relying on a type it could not see. Nothing caught that while llama_context's constructor
    // also lived in llama.cpp; migrating the constructor into src/llama-context.cpp exposed it.
    //
    // The definition was DELETED from llama.cpp in the same commit -- a copy would be an ODR
    // violation across translation units, which no compiler here would diagnose.
    //
    // KEEP IT AN AGGREGATE. src/llama.cpp:6939 constructs one with `llama_context::Prev{...}`
    // brace initialisation; adding a constructor, a base class or an access specifier would break
    // that call site.
    struct Prev {
        int seq_id_0;  // first token's sequence id (was the removed llama_batch::all_seq_id)
        int n_outputs;
        int n_kv;
        int n_tokens;
        int save_per_step_ssm;
        int per_step_max_allocated;
        llama_mtp_op_type mtp_op_type;
        int32_t mtp_step_idx;
        int32_t mtp_n_heads;
        int64_t swa_w_view;
        int64_t swa_win_off;
        uint64_t seq_fingerprint;
        uint64_t model_state_hash;
        ggml_cgraph * graph;
    };
    std::unique_ptr<Prev> prev;
    std::unique_ptr<Prev> prev_mtp;
    int32_t mtp_step_idx = 0;
    int32_t mtp_n_heads = 0;

    void reset_scheduler();
    bool can_reuse_graph(const llama_batch & u_batch, uint64_t seq_fingerprint, uint64_t model_state_hash);

    struct CacheCopy {
        ggml_tensor * cpy = nullptr;
        size_t        step = 0;
    };
    // Initialises every member below that the fork's own one-argument constructor used to set,
    // so BOTH constructors run one definition of the rule. Defined in src/llama.cpp beside the
    // sizing it performs. Added 2026-09-14: the two-argument constructor -- the only one anything
    // calls -- ran none of it.
    void init_fork_members();

    std::vector<CacheCopy> cache_copies;
    // GLM-DSA lightning indexer: the indexer-key cache (kr_l) write is a separate ggml_cpy that
    // the K/V cache_copies fixup does NOT cover. Under graph reuse (FA pads KV to 256, so n_kv
    // stays constant across consecutive decode ubatches and the graph IS reused) its view_offs
    // would stay baked at the first ubatch's kv_head, scattering this ubatch's indexer keys to a
    // stale slot. Later ubatches never populate their own recent index-key cells (those cells read
    // uninitialized -> wrong block-max-pool/top-k -> degraded/NaN sparse-FA decode). Register the
    // kr_l cpy per layer here and patch its offset in update_cache_copies(), exactly like K/V.
    std::vector<CacheCopy> dsa_cache_copies;
    std::vector<CacheCopy> openpangu_cache_copies;
    std::vector<CacheCopy> openpangu_cache_copies_mtp;

    bool update_cache_copies();

    bool ensure_dflash_kv_cache_tensors(int32_t cross_ctx);
    void free_dflash_kv_cache_tensors();
    bool ensure_dsv4_cache_tensors();
    void free_dsv4_cache_tensors();

    bool prepare_mtp_graph_inputs(
        struct llama_context & lctx);
    void set_mtp_op_type(llama_mtp_op_type value);
    void set_mtp_step_idx(int32_t value);
    void set_mtp_n_heads(int32_t value);

    int max_nodes(int n_tokens, int n_kv) const;

    // TriAttention KV scoring runtime (unified ik_llama build).
    // nullptr when --triattention is not specified, which is also what it stays at when the
    // stats file fails to load -- so every consumer must null-check rather than assume the
    // flag implies a live runtime.
    struct tria_runtime * tria_rt = nullptr;
};


//
// device selection helpers -- shared, so the rule has exactly ONE home
//
// MOVED HERE 2026-09-09, slice 28, from src/llama.cpp:3971-4008, where all three were `static` and
// therefore invisible outside that translation unit. src/llama-context.cpp needs
// ggml_backend_add_from_device for the unified backend-init block, and the alternative -- a second
// copy of the filter over there -- is exactly the shape that just cost this fork a real defect:
// slice 27 found the SYCL arm bypassing this rule in one of its two branches, failing permissively
// and unnoticed. A rule with two homes drifts. This one now has one.
//
// They are INLINE and defined after struct llama_context closes, because
// ggml_backend_add_from_device dereferences ctx->backends and ctx->cparams and needs the complete
// type. src/llama.cpp includes this header, so its existing call sites resolve here unchanged.

inline bool striequals(const char * a, const char * b) {
    for (; *a && *b; a++, b++) {
        if (std::tolower(*a) != std::tolower(*b)) {
            return false;
        }
    }
    return *a == *b;
}

inline bool item_in_list(const std::vector<std::string> & devices, const char * name) {
    for (auto & device : devices) {
        if (striequals(device.c_str(), name)) {
            return true;
        }
    }
    return false;
}

// Up/gate expert repacking. DEFINED IN src/llama.cpp, declared here so llama-context.cpp can call
// it. It was `static void` and therefore had internal linkage, declared in no header at all,
// defined once and called once -- which is why it was the one fork feature of the three missing
// from llama-context.cpp that could not be ported as an assignment: outside the monolith the name
// did not exist.
//
// EXPOSED RATHER THAN MOVED, on purpose. The body depends on interleaved_properties(), which lives
// in src/llama-quantize.cpp, and on llama_model's layer internals. Relocating it would drag those
// dependencies across a file boundary for no gain, while dropping `static` costs nothing: the
// function is defined exactly once, so there is no second symbol to collide with -- and the
// duplicate-symbol-sources and duplicate-archive-symbols gates in the fork's pre-commit hook check
// exactly that, on the source and on the emitted archives respectively.
//
// When src/llama.cpp's llama_init_from_model is eventually deleted in favour of this file's, this
// declaration is what keeps the repack path alive.
void llama_repack_up_gate_exps(llama_context & lctx);

// THE LEGACY DECODE, defined in src/llama.cpp. Declared here because llama_context::decode must
// route old-generation models to it: decode() builds graphs via model.build_graph(), which returns
// nullptr for those models, so without this route every decode of such a model fails with
// GGML_STATUS_FAILED. This function owns the llm_build_context::llama_build_graph call that does
// work for them, along with its own batch splitting and output bookkeeping.
int llama_decode_internal(llama_context & lctx, llama_batch batch_all);
// a mixed token/embd batch on a legacy-graph model: `runs` are its maximal same-kind row runs in batch order; each is decoded by
// llama_decode_internal, then the output buffer is laid out over the whole batch as one decode would leave it (-1: not equivalent here)
int llama_decode_internal_runs(llama_context & lctx, const std::vector<llama_batch> & runs);
// no cell of the legacy cache (kv_self) holds seq_id -- llama_memory_seq_pos_min / _max then return -1 (upstream contract)
bool llama_kv_cache_seq_is_empty(struct llama_context * ctx, llama_seq_id seq_id);
// the legacy encoder entry (t5 and the other encoder-decoder models); llama_context::encode sends old-generation models here
int llama_encode_internal(llama_context & lctx, llama_batch batch);

// THE LEGACY KV CACHE'S INITIALISER, defined in src/llama.cpp. Declared here because
// llama_context's constructor must call it: this fork's graph builders read lctx.kv_self through
// llm_build_context's `kv` reference member, and the new constructor builds `memory` instead, so
// without this call the cache the builders read is never sized and they abort at
// src/llama-build-context.cpp:930, GGML_ASSERT(kv.size == n_ctx).
bool llama_kv_cache_init(
             struct llama_kv_cache_fork & cache,
               const llama_context * ctx,
                         ggml_type   type_k,
                         ggml_type   type_v,
                         ggml_type   idx_type_k,
                          uint32_t   kv_size,
                              bool   offload,
                          ggml_type  type_k_first,
                          ggml_type  type_k_last,
                          ggml_type  type_v_first,
                          ggml_type  type_v_last,
                          int32_t    n_k_first,
                          int32_t    n_k_last,
                          int32_t    n_v_first,
                          int32_t    n_v_last);

// THE LIVE-CONTEXT REGISTRY, defined in src/llama.cpp. This fork's one-argument
// llama_context(const llama_model &) constructor pushes `this` onto it and the destructor must
// erase it again -- a context left in this vector after it is freed is a dangling pointer every
// later walker of the list will follow.
//
// EXPOSED 2026-09-13, and only because the destructor moved. It was `static` in src/llama.cpp,
// which is correct as long as both the constructor and the destructor live there. ~llama_context()
// now lives in llama-context.cpp (it has to: the union of the two bodies needs opt_ctx and
// backend_buf_exp_size, which src/llama.cpp neither includes nor mentions), so the erase needs a
// name it can reach. Same mechanism as llama_repack_up_gate_exps directly above -- drop the
// `static`, declare it here, one definition only.
std::vector<llama_context *> & llama_all_contexts();

// Registry devices the compile-time #if chain does not cover (HEXAGON, GGML_BACKEND_DL plugins, ...), in registry order;
// they take model.devices ordinals starting at llama_extra_device_first_ordinal(model). Defined in src/llama.cpp.
// Row S69-NPU-LOADER-EXTRA-BACKEND-ARM-REGISTRY-DEVICES-2026-11-06-01.
std::vector<ggml_backend_dev_t> llama_extra_registry_devices();
int llama_extra_device_first_ordinal(const llama_model & model);

// THE ENFORCEMENT POINT for the user's --device selection. Adding a backend with a raw
// ctx->backends.push_back instead of this call attaches it whether or not the user asked for that
// device; the only place that is correct is the CPU backend, which must always be present.
inline void ggml_backend_add_from_device(llama_context * ctx, ggml_backend_t backend) {
    const char * name = ggml_backend_name(backend);
    if (ctx->cparams.devices.size()) {
        if (item_in_list(ctx->cparams.devices, name)) {
            ctx->backends.push_back(backend);
        }
    } else {
        ctx->backends.push_back(backend);
    }
}
