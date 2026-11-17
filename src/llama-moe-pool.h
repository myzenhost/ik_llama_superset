// llama-moe-pool.h -- MoE resident expert pool: activation recorder + placement strategies
// + (host-side, ggml-free) LRU pool bookkeeping.
//
// S69-KT-EXPERT-POOL-2026-09-28. Port of the *design* (not the runtime) of two independent
// upstream sources, both MIT-licensed llama.cpp branches, cited again at each function this
// file's logic comes from:
//   - llama.cpp PR #27861 (csantiago78, "GPU-resident LRU cache for host-offloaded MoE expert
//     weights", opened 2026-08-28, still draft/open as of this port): --moe-expert-cache N
//     slots per host-resident expert layer, id->slot remap via ggml_get_rows, no new ggml ops.
//   - llama.cpp discussion #28248 (memoriaru, RFC "Persistent expert slot pool for MoE CPU
//     offload"): same id-remap idea with a safer prefill/working-set gate and the root-caused
//     fix for a map-table race (the table must be a real buffer-backed tensor, never a view).
// and of KTransformers' *design* (doc/en/kt-kernel/experts-sched-Tutorial.md, Apache-2.0, read
// in full) for the placement-strategy names and semantics; the actual strategy math
// (generate_gpu_experts_masks, top-K by flattened activation frequency) is read from
// kt-kernel/python/experts_base.py:162-213 and re-expressed here in C++ against a per-layer
// slot budget rather than KT's global-flattened budget, because this fork's
// --moe-expert-cache N is PER LAYER (matching PR #27861's own flag semantics, not KT's).
//
// SCOPE OF THIS FILE:
//   2026-09-28 stage 1 session: the recorder, the placement strategies and the VRAM rail sizing
//   were wired into the runtime eval-callback path (llama-context.cpp cb_eval, unit 1 of stage 2)
//   and are unit-tested (tests/test-moe-pool-placement.cpp, tests/test-moe-pool-recorder-e2e.cpp).
//   The llama_moe_pool struct below is deliberately ggml-free host-side bookkeeping (LRU +
//   admission + byte-accounting) so it is unit-testable without a GPU.
//   2026-09-28 stage 2 session (unit 2): the tensor-level graph splice -- allocating the
//   companion weight tensors, the get_rows id remap, routing llm_build_lora_mm_id through the
//   pool -- was RESEARCHED AND DESIGNED (this session read PR #27861's full diff/thread and
//   discussion #28248's full RFC/thread, not summaries) but NOT IMPLEMENTED. See "GRAPH SPLICE
//   STATUS" below for exactly what is and is not wired at HEAD of this branch, and the ledger
//   (tmp/_agent_ledger_s69-kt-pool-stage2-2026-09-28.md) for why implementation
//   was deferred rather than attempted blind: this fork's shared build-and-test queue measured
//   11 jobs deep with a position advancing roughly 1/7.5min during this session, so a new
//   tensor-allocation subsystem could not get even one verified build-test-fix cycle in the time
//   available, and both #27861 and #28248 independently shipped a silent-wrong-weight-read bug
//   class exactly because their first cut of this same mechanism was not caught before it looked
//   done -- landing unverified code in that same class was judged the wrong trade against
//   reporting a researched design honestly instead.

#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

// ───────────────────────── Expert activation recorder ─────────────────────────
// Mirrors KT's --record-kt-gpu-expert-distribution / --expert-distribution-recorder-mode
// (doc/en/kt-kernel/experts-sched-Tutorial.md table, "Key Parameters"). Counts are accumulated
// per (layer, expert) from the SAME "ffn_moe_topk" tensor name llm_build_moe_ffn already tags
// (src/llama-build-context.cpp: cb(selected_experts, "ffn_moe_topk", il)), read back through the
// existing ggml_backend_sched_eval_callback (cparams.cb_eval) -- no new upcall mechanism, unlike
// PR #27861's own caveat about needing "a cleaner layering" for its ggml_set_moe_obs_callback.
enum llama_moe_recorder_mode {
    LLAMA_MOE_RECORDER_STAT        = 0, // running totals only (default)
    LLAMA_MOE_RECORDER_STAT_APPROX = 1, // running totals, decayed (halved) periodically
    LLAMA_MOE_RECORDER_PER_PASS    = 2, // one snapshot per llama_decode() call
    LLAMA_MOE_RECORDER_PER_TOKEN   = 3, // one snapshot per generated token
};

llama_moe_recorder_mode llama_moe_recorder_mode_from_str(const std::string & s);

struct llama_moe_recorder {
    explicit llama_moe_recorder(llama_moe_recorder_mode mode) : mode(mode) {}

    llama_moe_recorder_mode mode;
    // counts[layer][expert_id] = activation count
    std::unordered_map<int32_t, std::unordered_map<int32_t, uint64_t>> counts;
    uint64_t steps_recorded = 0;

    // Record one ubatch's selected-expert ids for a layer (ids has n_ids entries; duplicates
    // within a token are possible when n_expert_used>1 and are each counted, matching KT's
    // activation-frequency semantics -- it counts activations, not distinct experts-per-token).
    void record(int32_t layer, const int32_t * ids, int64_t n_ids);

    // stat_approx decay: halve every `half_life` calls to record() for this recorder's whole
    // lifetime (matches the admission-gate idea from the PR #27861 thread: "counters halved
    // every 16 decode steps so the gate tracks recent locality rather than all-time frequency").
    void maybe_decay(int32_t half_life);

    // Flatten to a dense [n_layers][n_experts] table for the frequency placement strategy.
    std::vector<std::vector<uint64_t>> to_dense(int32_t n_layers, int32_t n_experts) const;

    // Hand-rolled JSON writer (src/ does not depend on common/json.h; the shape is simple enough
    // not to need it): {"mode":"stat","steps":N,"layers":{"0":{"3":12,"7":4},...}}
    bool write_json(const std::string & path) const;

    // Inverse of write_json, for the frequency strategy's --init-expert-location loader and for
    // the leakage-safe test (calibrate on one file, measure on another).
    static bool read_json(const std::string & path, std::unordered_map<int32_t, std::unordered_map<int32_t, uint64_t>> & out);
};

// ───────────────────────── Placement strategies ─────────────────────────
// Seeds which expert ids should be resident in a layer's pool at load time (uniform/random/
// front-loading) or from recorded stats (frequency), before any runtime LRU activity. Strategy
// names and semantics match KT's table (Tutorial.md "Expert Placement Strategies"); "frequency"
// selects the layer's own top-N by recorded count (KT's generate_gpu_experts_masks top-K,
// re-expressed per-layer instead of KT's flattened-global top-K because our slot budget is
// per-layer -- see file header).
enum llama_moe_placement_strategy {
    LLAMA_MOE_PLACEMENT_UNIFORM       = 0, // first N expert ids, in id order (no prior stats needed)
    LLAMA_MOE_PLACEMENT_FREQUENCY     = 1, // top-N by recorded activation count for this layer
    LLAMA_MOE_PLACEMENT_FRONT_LOADING = 2, // same as uniform per-layer; the "front-loading" bias
                                            // is expressed ACROSS layers by the caller (see note
                                            // in llama-moe-pool.cpp) since a single layer has no
                                            // "front" of its own -- documented, not silently equal.
    LLAMA_MOE_PLACEMENT_RANDOM        = 3, // KT's fixed seed 42 (Tutorial.md: "fixed seed (42)")
};

llama_moe_placement_strategy llama_moe_placement_strategy_from_str(const std::string & s);

// Returns up to n_slots expert ids (0 <= id < n_expert), sorted ascending, deterministic for a
// given (strategy, layer, seed, freq_row). freq_row may be empty (uniform/front-loading/random
// do not need it); for frequency with an empty/all-zero freq_row this falls back to uniform and
// the caller is expected to log that fallback (a seeded pool that silently equals uniform must
// be caught -- see the ANTI-BIAS test requirement in the task brief).
std::vector<int32_t> llama_moe_seed_layer(
    llama_moe_placement_strategy strategy,
    int32_t                      layer,
    int32_t                      n_layers,
    int32_t                      n_expert,
    int32_t                      n_slots,
    const std::vector<uint64_t> & freq_row, // freq_row[expert_id] = count, may be empty
    uint32_t                     seed = 42);

// ───────────────────────── VRAM rail ─────────────────────────
// Sizes the pool from free device memory, not from the requested slot count alone (a requested
// N is a CEILING, never a guarantee -- task brief, VRAM RAIL). Pure function: no NVML call here,
// so it is unit-testable with injected values; the real NVML free-memory read (the WDDM-correct
// one, per tmp/research/findings' WDDM free-VRAM draft, NOT a naive total-memory
// read) is a separate call at the registration site, not in this function.
int64_t llama_moe_pool_vram_budget_slots(
    int64_t free_bytes,
    int64_t reserve_bytes,   // headroom kept free (task brief default: 6 GiB)
    int64_t bytes_per_slot,  // one slot's worth of up+gate+down rows for this layer's dtype
    int64_t requested_slots);// the ceiling from --moe-expert-cache; the rail only ever LOWERS this

constexpr int64_t LLAMA_MOE_POOL_DEFAULT_RESERVE_BYTES = int64_t(6) * 1024 * 1024 * 1024;

// ───────────────────────── Host-side LRU pool bookkeeping (ggml-free) ─────────────────────────
// GRAPH SPLICE STATUS (2026-09-28, updated stage 2): this struct is the correctness-tested
// foundation for the resident pool's admission/eviction policy (tests/test-moe-pool-placement.cpp,
// synthetic ids). It is STILL NOT wired into the live decode graph. Stage 2 read both upstream
// sources in full (not abstracts) and narrowed the splice to a concrete, low-risk plan for a
// FUTURE session, rather than implementing it blind this session (see file-header SCOPE note and
// the ledger for why). The plan, with file:line citations:
//
//   SCOPE CUT that makes this tractable: only a STATIC pool (seeded once at context init from a
//   llama_moe_seed_layer() placement strategy, never re-placed during decode) is in scope for a
//   first splice. cparams.moe_dynamic_update (runtime re-placement) stays unwired -- a table that
//   never changes after context init cannot hit either upstream source's root-caused bug (both
//   were races between a RUNTIME table mutation and an in-flight graph read; see below). Dynamic
//   re-placement is a later, separate unit once the static splice is verified.
//
//   CORRECTNESS BAR (this fork's own task brief, CPU-build control): with moe_pool_slots == 0,
//   logits byte-identical to the unspliced path (true for free -- DISABLE-REVERSIBLE, nothing new
//   is on the call path). With moe_pool_slots > 0 on a CPU-only build sized so n_slots >= n_expert
//   ("pool degenerate" -- LLAMA_MOE_PLACEMENT_UNIFORM with n_slots==n_expert already produces the
//   IDENTITY assignment: llama_moe_seed_layer's uniform branch, this file's .cpp, returns
//   ids[i]=i -- slot i holds expert i, no expert is ever the evicted "miss" case), logits must
//   also be byte-identical, because: (a) the id remap get_rows(identity_table, selected_experts)
//   returns selected_experts unchanged, and (b) a companion weight tensor built as an exact copy
//   of up_exps/gate_exps/down_exps has bit-identical bytes to the original. mul_mat_id is a pure
//   function of (weight bytes, input, ids), so (a)+(b) together make the spliced and unspliced
//   paths compute the identical op on identical bytes -- not merely an empirically-hoped-for
//   match. A non-degenerate CPU-only config (n_slots < n_expert) is explicitly OUT of what this
//   plan claims to verify; it would need the CPU mul_mat_id skip-table (PR #27861's src[3]
//   mechanism) to avoid computing the missed experts twice, which is separate, harder, riskier
//   work not scoped here.
//
//   MECHANISM, ported from PR #27861 (csantiago78, MIT, read in full, still draft/open at the
//   time of this read) with the map-table fix from discussion #28248 (memoriaru, MIT RFC, root
//   cause read in full):
//     1. Per pooled layer, allocate a REAL BUFFER-BACKED (never a ggml_view/ggml_reshape) I32
//        tensor shaped [1, n_expert] -- discussion #28248's own root-caused bug (its "Design"
//        and "Validation" sections, read in full via the cached fetch this session) was exactly
//        that its first cut read this table through a RESHAPE view, which carries no buffer, so
//        the scheduler's split-boundary dependency check could not see it and a running split
//        could consume the PREVIOUS ubatch's table -- deterministic ~20-30% wrong weight reads on
//        a 512-expert model at 32 slots, found and fixed upstream, cited here so it is not
//        repeated. A STATIC table (this plan's scope cut, above) cannot hit this specific race --
//        there is no "previous ubatch's table", it is the only table for the context's lifetime --
//        but the tensor must still be real and buffer-backed rather than a view, both because
//        get_rows requires a data buffer to read from and to keep the door open for a later,
//        non-static extension without a second rewrite.
//     2. Build the table's VALUES from llama_moe_seed_layer()'s chosen strategy (this file, above):
//        table[e] = slot index of expert e if resident (0..n_slots-1), else n_slots (the dummy
//        slot, PR #27861's design: "SlotK is permanently zero (the dummy slot)... uncached ids
//        map to the zero slot and contribute exactly 0"). For the verified degenerate config
//        (n_slots>=n_expert, uniform) table[e]=e for every e and the dummy slot is provably never
//        selected.
//     3. Allocate the companion weight tensor per pooled layer, shaped [ne0, ne1, n_slots+1] (the
//        "+1" is the dummy slot, PR #27861's own shape), and fill slot n_slots with zero, slots
//        0..n_slots-1 by copying the resident experts' rows from the original up_exps/gate_exps/
//        down_exps tensor (a per-layer copy this session did not write -- see status below).
//        Lifecycle: model these two allocations (map table + companion tensor) on
//        llama_context::dflash_runtime::kv_runtime_state's cache_ctx/cache_bufs pattern
//        (src/llama-context.h ~868-895, filled by ensure_dflash_kv_cache_tensors declared at
//        llama-context.h:1181) -- an existing, working precedent in THIS codebase for "own
//        ggml_context + ggml_backend_buffer_t, allocated once outside the model's weight context,
//        referenced by pointer from every graph build" -- rather than inventing a new allocation
//        idiom. NOT READ IN FULL this session (named as the pattern to follow, not yet copied).
//     4. At each of the three llm_build_lora_mm_id call sites inside llm_build_moe_ffn
//        (src/llama-build-context.cpp:1745 up_exps, :1748 gate_exps, :1790 down_exps), when a
//        pool is registered for this layer AND n_tokens == 1 (decode only -- PR #27861's own
//        original gate; "prefill routed around the pool" per this task's brief, matching
//        upstream's own most-reviewed, least-extended design rather than the thread's later
//        n_tokens<=4 extension, which was NOT read closely enough here to port safely):
//          ggml_tensor * slot_ids = ggml_get_rows(ctx, table_il, selected_experts);
//        This reuses an idiom ALREADY PRESENT in this exact file (rule 1.87: read the source
//        before reimplementing) -- src/llama-build-context.cpp:1655-1656 already does
//        `ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens), selected_experts)`
//        to turn per-token expert ids into a per-token gathered value via get_rows over a
//        [1, n_expert, n_tokens]-shaped table -- the id-remap table here is the same shape
//        convention, [1, n_expert] repeated across n_tokens (PR-thread's Gemma4 extension shows
//        the general form: `ggml_repeat_4d(ctx0, mc_tbl, 1, mc_tbl->ne[1], n_tokens, 1)` then
//        `ggml_get_rows(...)`, reshaped to [n_expert_used, n_tokens] -- not needed for n_tokens==1
//        but the reshape-back step is the same either way). Then call
//        `llm_build_lora_mm_id(lctx, ctx, companion_tensor_il, cur, slot_ids)` in place of the
//        original tensor/ids pair.
//   NOT DONE: the allocation code (step 3), the call-site edit (step 4), and therefore no build or
//   test of any of this. The pieces above (recorder, placement strategies, VRAM rail, this LRU
//   struct) are real and tested; this design is real and cited; the splice itself is not written.
// Measured-cost inputs for llama_moe_pool_layer::step_gated (S69-FASTLLM-PICK-COST-GATED-EXPERT-ADMISSION-2026-09-29-01; behaviour of
// fastllm's CUDA expert cache, Apache-2.0, from its docs/cuda-expert-cache.md and the admission code described there; no code copied).
// All times are microseconds, observed by the caller (EMAs of cudaEvent/steady-clock timings); valid == false switches the gate off.
struct llama_moe_pool_cost {
    bool    valid     = false;
    double  cpu_us    = 0.0;   // CPU work this call performs for the experts that stay on the host
    double  gpu_us    = 0.0;   // device work this call performs for the experts that are resident
    double  refill_us = 0.0;   // measured time to copy ONE expert into a slot
    double  margin    = 1.1;   // the copy must hide behind the CPU work with this much slack
    int32_t n_tokens  = 0;     // tokens in the ubatch; > 1 normalises heat so a prefill scan cannot flood the counters
};

// A cold expert is only worth a slot when the copy overlaps work the CPU has to do anyway: (cpu_us - gpu_us) > refill_us * margin.
// The caller visits layers with a stride coprime with n_layers so every layer is reached; this returns the smallest interval >= base
// with gcd(interval, n_layers) == 1 (base < 1 is treated as 1).
int32_t llama_moe_pool_admit_interval(int32_t base, int32_t n_layers);

struct llama_moe_pool_layer {
    int32_t              n_expert  = 0;
    int32_t              n_slots   = 0;
    int32_t              admit_threshold = 3; // PR #27861-thread admission-gate idea
    // slot i holds expert id resident_id[i], or -1 if empty.
    std::vector<int32_t> resident_id;
    // recency[i] = a monotonically increasing counter, higher = more recently used.
    std::vector<uint64_t> recency;
    // per-expert recent-use counter (PR thread's admission gate: "per-layer uint8_t use counter
    // per expert, incremented on uncached observation, saturated"). Uses uint32_t here (no need
    // to match the PR's byte-sized field; this struct is not that binary layout).
    std::unordered_map<int32_t, uint32_t> use_count;
    uint64_t              clock = 0;

    struct step_stats {
        int64_t hits = 0, misses = 0, admitted = 0, evicted = 0;
        int64_t gated = 0;   // step_gated only: admissions withheld because the copy would not hide behind the CPU work
    };

    // step_gated state. age_window = 0 keeps the counters forever (the old behaviour); otherwise a counter whose expert was last seen more
    // than age_window calls ago is reset before it is incremented (fastllm: 64 x n_layers).
    int32_t  age_window = 0;
    uint64_t call_no    = 0;
    std::unordered_map<int32_t, uint64_t> last_seen;

    void init(int32_t n_expert_, int32_t n_slots_, int32_t admit_threshold_);

    // Seed initial residency from a placement strategy's output (empty slots stay -1).
    void seed(const std::vector<int32_t> & ids);

    // Process one step's observed ids (already-decoded expert ids for this layer, this ubatch).
    // Applies the admission gate + LRU eviction, respecting max_inserts (PR's
    // --moe-expert-cache-inserts throttle). Returns per-call stats; does NOT touch any tensor --
    // the caller is responsible for the corresponding host->device weight-row copy once the
    // tensor-level splice above exists.
    step_stats step(const int32_t * ids, int64_t n_ids, int32_t max_inserts);

    // Cost-gated, aged, single-admission policy for one call (one ubatch's routed ids for this layer). Differences from step():
    //   * heat of a cold expert: += its number of routes in this call for a decode ubatch; for a prefill ubatch (cost.n_tokens > 1)
    //     += clamp(ceil(8 * routes / n_tokens), 1, 8) so one big scan cannot make every expert hotter than a few decode hits;
    //   * a counter not touched for more than age_window calls restarts from zero;
    //   * AT MOST ONE expert is admitted per call: the hottest cold one at or above admit_threshold (ties: lowest id), and only when
    //     max_inserts >= 1 and, if cost.valid, (cpu_us - gpu_us) > refill_us * margin.
    step_stats step_gated(const int32_t * ids, int64_t n_ids, int32_t max_inserts, const llama_moe_pool_cost & cost);

    bool is_resident(int32_t expert_id) const;
};
