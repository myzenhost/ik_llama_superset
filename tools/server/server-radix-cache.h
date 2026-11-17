// On-device token-level radix prefix cache for tools/server (--radix-cache, kv-unified only).
//
// Row S69-ADD-TGI-ALTERNATIVES-LLGUIDANCE-PAGEDKV-RADIX-SYNTHID-2026-09-27-01, parent design
// S69-ENGINE-WAVE2-SGLANG-RADIXATTENTION-2026-09-26-01. Distinct from the already-landed
// server_prompt_cache (tools/server/server-task.h:694-732, publish hook server-context.cpp
// ~5085-5128): that mechanism serializes a slot's WHOLE prompt state to host RAM keyed by an
// EXACT-match hash and restores it for one sibling at a time (now non-consuming/N-way per
// S69-ENGINE-WAVE2-SGLANG-RADIXATTENTION-NCONSUMER-FOLLOWUP-2026-09-27-01). This class instead
// shares KV at TOKEN granularity, on-device, zero-copy: two DIFFERENT prompts that merely share
// a prefix (e.g. a common system/RAG preamble followed by different user turns) both reuse the
// shared portion, because llama_memory_seq_cp() between two sequences in the SAME kv_unified
// stream only tags existing cells with an extra seq id -- no data copy
// (src/llama-kv-cache.cpp:527-556, seq_cp same-stream branch).
//
// BENCH-PENDING: no throughput/latency numbers measured yet for this class; the fail-before/
// pass-after functional tests (tools/server/tests/unit/test_radix_prefix_cache.py) check
// correctness (prompt_n reduced to the suffix length, byte-identical output at temperature 0)
// and cache-hit accounting, not speed. It is a pure caching layer either way: a cache miss can
// only cost the seq-id bookkeeping overhead, never request correctness.
//
// SOURCE-READ (rule 1.87), design ported, no code copied (both Apache-2.0):
//   - sglang/python/sglang/srt/mem_cache/radix_cache.py @ b4584ab83884ff40a4e5a77f0ea4ef98c038d3b9
//     -- RadixCache.match_prefix() (longest-prefix walk with node splitting),
//        RadixCache.insert()/_insert_helper() (split-on-divergence trie insert),
//        RadixCache.evict() (LRU over leaves), inc_lock_ref()/dec_lock_ref() (usage refcounting).
//   - lightllm/lightllm/server/router/dynamic_prompt/radix_cache.py:111 -- same shape, cross-
//     checked for the split-node refcount-transfer detail.
//   - text-generation-inference/backends/v3/src/radix.rs -- partial eviction keeps the matched
//     prefix alive while trimming the unused tail; informed evict_idle_leaf() below (evict whole
//     leaves only, never truncate a live consumer's still-locked prefix).
//
// LoRA configuration (row S69-LORAX-PICK-LORA-CONFIG-IN-PROMPT-CACHE-AND-RADIX-KEY-2026-09-29-01): KV of the same
// tokens differs between adapter configurations, so the tree is keyed by a LoRA fingerprint (lora_config_fingerprint(),
// server-common.h) in addition to tokens -- one root per fingerprint, the way lorax (predibase/lorax, Apache-2.0,
// router/src/radix.rs) keeps one root per adapter. Fingerprint 0 = no adapter active = the historical single tree,
// and it is the default of every entry point, so a caller that never passes one behaves exactly as before.
//
// Mapping notes (why this is a valid design, not just SGLang's Python restated in C++):
//   - SGLang's tree stores actual KV tensors per node and moves *pointers* on eviction/match.
//     Here a node owns a CONTIGUOUS ABSOLUTE POSITION RANGE [pos_start, pos_end) in llama.cpp's
//     canonical per-sequence position numbering (position i = token i of the prefix, counted
//     from 0), realized as KV cells tagged with the node's own reserved llama_seq_id. Two
//     sequences can only share a cell if they agree on that cell's position -- true by
//     construction for a shared prefix (both start counting from 0), which is exactly SGLang's
//     own assumption (RadixAttention shares the KV of an identical token AT THE SAME position).
//   - A node's cells stay resident as long as ITS OWN seq id remains tagged to them, independent
//     of any slot that later seq_cp'd a copy of the tag onto its own seq id -- so evicting a tree
//     node never corrupts a slot already mid-request (that slot's copy is its own independent
//     tag on the same physical cells, per llama_kv_cells' seq bitset-per-cell design).
//
// DISABLE-REVERSIBLE: GATED OFF (this class is simply never constructed; --radix-cache stays
// parsed and reserves its seq ids regardless, but is a runtime no-op) for anything where "same
// absolute position" does not mean "same KV content/meaning" -- enforced by the caller
// (server-context.cpp) before construction, enumerated there: !kv_unified,
// llama_model_is_recurrent/is_hybrid/is_diffusion, llama_model_n_swa() > 0. Re-enable condition
// is simply not being one of those model/memory types; no fix is owed for them (see the row's
// CHALLENGE section -- shared position does not imply shared meaning for recurrent state, SWA
// windowing, or a diffusion model's non-causal mask). Scope note: the ORIGINAL PROMPT is
// inserted at SLOT_STATE_DONE_PROMPT (before this request's own generation can trigger the
// fork's slot-context-shift, server-context.cpp ~7859) -- so a shift's position renumbering can
// never invalidate an already-inserted node. A FINISHED slot's GENERATED tokens are published
// too, once, in the post-decode phase after release (server_radix_gen_publish_len() below says
// how many tokens): only when no context shift happened during the request (a shift renumbers
// positions, so the cells would no longer sit at their token index), and clipped to the KV the
// slot really holds (the last sampled token is never decoded). NOT covered
// (filed, not silently dropped): using a radix
// match to beat a slot's OWN already-longer leftover cache (server-context.cpp only tries a
// radix match when the slot is otherwise empty, n_past == 0 and slot.prompt.tokens.size() == 0 --
// see the call site comment there for why interleaving with cache_prompt/n_cache_reuse/
// alora/checkpoints was left alone instead).
#pragma once

#include "llama.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

using llama_tokens = std::vector<llama_token>;

// Which idle leaf the tree gives up first when its reserved seq-id pool is exhausted (--radix-evict-policy). LRU is what the cache
// always did and stays the default. The others are SGLang's selectable strategies, written against our own node fields.
// SOURCE-READ: sglang/python/sglang/srt/mem_cache/evict_policy.py:19-60 (LRU/LFU/FIFO/MRU/FILO/Priority strategies: each maps a node to a
// sort key, the smallest key is evicted first), :85-99 (SLRU: segment 0 = hit_count < threshold goes before segment 1),
// mem_cache/utils.py:48-84 (name -> strategy table), radix_cache.py:541-569 (evict loop: pop the smallest key, then re-offer the parent
// once it has become a childless leaf), :667-668 (a split keeps the child's hit_count and priority), :689-690,:725-739 (hit_count counts
// insert() passes through a node; a new node starts at 1), :706,:723-726 (priority is raised along the whole inserted path).
// Apache-2.0, behaviour only, no code copied.
enum class server_radix_evict_policy {
    LRU,       // oldest last use first
    LFU,       // fewest hits first, then oldest last use
    FIFO,      // oldest creation first
    MRU,       // newest last use first
    FILO,      // newest creation first
    SLRU,      // nodes with fewer than slru_hits hits (probation) before the rest, each group oldest last use first
    PRIORITY,  // least important request priority first (highest numeric value in OUR convention, see below), then oldest last use
};

// How many leading tokens of a just-finished slot's token list can be published into the tree as prompt + generation; 0 = publish
// nothing extra. Row S69-MISTRALRS-PICK-PAGEDATTN-BLOCKPOOL-2026-09-28-01 (publish generated tokens, ExLlamaV2 page hashing idea:
// a follow-up turn that replays the previous answer then hits the cache). Pure, so it is unit-tested without a model.
//   n_tokens      slot.prompt.tokens.size() at release (prompt + generated, the last sampled token included)
//   n_prompt_pub  tokens the prompt-time insert already published (a path is only worth re-inserting when it grew)
//   kv_pos_max    llama_memory_seq_pos_max() of the slot's seq (-1 when it holds nothing); position i is token i, so KV covers kv_pos_max + 1
//   ctx_shifted   the request hit the slot context shift: cells were renumbered, token index != position any more
//   has_mtmd      the prompt carries media chunks: a token id does not identify the KV there
inline size_t server_radix_gen_publish_len(size_t n_tokens, size_t n_prompt_pub, int64_t kv_pos_max, bool ctx_shifted, bool has_mtmd) {
    if (ctx_shifted || has_mtmd || kv_pos_max < 0) {
        return 0;
    }
    const size_t n_kv = (size_t) kv_pos_max + 1;
    const size_t n    = n_tokens < n_kv ? n_tokens : n_kv;
    return n > n_prompt_pub ? n : 0;
}

// "lru" ... "priority"; false when the name is unknown (out is left untouched).
bool        server_radix_evict_policy_parse(const std::string & name, server_radix_evict_policy & out);
const char * server_radix_evict_policy_name(server_radix_evict_policy p);

// The bookkeeping the policies read, kept apart from the tree so the ordering can be tested without a llama context.
struct server_radix_evict_info {
    uint64_t last_use = 0;   // clock value of the latest match/insert touching the node
    uint64_t created  = 0;   // clock value when the node was created
    uint64_t hits     = 0;   // see hit_count below
    int32_t  priority = 0;   // request priority of the most important request that inserted through the node (lower value = more important,
                             // the same convention as the `priority` request field; SGLang's is inverted, see PRIORITY above)
};

// True when `a` must be evicted before `b` under `p` (strict weak ordering; equal keys fall back to last_use, then creation).
bool server_radix_evict_before(server_radix_evict_policy p, uint32_t slru_hits, const server_radix_evict_info & a, const server_radix_evict_info & b);

struct server_radix_node {
    llama_tokens                                     tokens;          // this node's own edge label
    llama_pos                                        pos_start = 0;   // absolute position of tokens[0]
    llama_seq_id                                      seq_id    = -1;  // reserved seq id owning these cells
    int32_t                                          ref       = 0;   // live match_prefix() locks on this node
    uint64_t                                         last_use  = 0;   // LRU clock, touched on every match/insert
    uint64_t                                         created   = 0;   // clock value at creation (FIFO/FILO)
    uint64_t                                         hit_count = 1;   // insert() passes through this node (a new node counts its own insert)
    int32_t                                          priority  = 0;   // most important (lowest) request priority that inserted through it
    server_radix_node *                              parent    = nullptr;
    std::vector<std::unique_ptr<server_radix_node>>  children;

    llama_pos pos_end() const { return pos_start + (llama_pos) tokens.size(); }
    bool is_leaf() const { return children.empty(); }
};

class server_radix_cache {
public:
    // ctx must already be kv_unified (caller's job to gate). seq_id_base/seq_id_count define the
    // reserved, EXCLUSIVE seq id range [seq_id_base, seq_id_base + seq_id_count) this cache may
    // use for its own tree nodes -- must not overlap any slot's own id or a CFG guidance id.
    server_radix_cache(llama_context * ctx, llama_seq_id seq_id_base, int32_t seq_id_count);

    // Find the longest prefix of `tokens` resident in the tree WITHOUT copying anything yet.
    // Locks (ref++) every matched node along the path so it cannot be evicted before the caller
    // calls commit() or release(); exactly one of those must follow every match_prefix() call
    // that returns > 0 tokens locked (a return of 0 locks nothing and needs no release).
    // `lora_fp`: only nodes inserted under the same LoRA fingerprint can match (see the note at the top).
    size_t match_prefix(const llama_tokens & tokens, uint64_t lora_fp = 0);

    // Length of the longest resident prefix of `tokens` under `lora_fp`, read-only: no locks, no recency touch, no stats. Used by the
    // scheduler to rank waiting requests (--sched-policy lpm) without disturbing what the tree would evict.
    size_t peek_prefix(const llama_tokens & tokens, uint64_t lora_fp = 0) const;

    // Realize the match: llama_memory_seq_cp() each locked node's covered range (clipped to `n`,
    // which must be <= the value match_prefix() returned) from its own seq id into `dst_seq`
    // (same kv_unified stream => metadata-only, no device copy), then unlocks (ref--).
    void commit(llama_seq_id dst_seq, size_t n);

    // Discard the pending match without copying anything (unlocks every node match_prefix()
    // locked). Call when the match was not useful (e.g. shorter than what the caller already had).
    void release();

    // Insert `tokens` (already fully resident under src_seq at absolute positions
    // [0, tokens.size())) into the tree, splitting/creating nodes as needed. Each newly-created
    // node's range is seq_cp'd (same-stream, zero-copy) from src_seq into a freshly claimed
    // reserved seq id. Evicts idle (ref == 0) LRU leaves first if the reserved pool is exhausted;
    // if a range still can't get a seq id, that suffix is silently left uncached (a cache miss
    // costs only bookkeeping, never correctness -- see the BENCH-PENDING note above).
    // `lora_fp`: the LoRA fingerprint src_seq's KV was computed under; the range goes into that fingerprint's tree.
    // `priority`: the inserting request's priority (lower = more important); only read by the PRIORITY eviction policy.
    void insert(const llama_tokens & tokens, llama_seq_id src_seq, uint64_t lora_fp = 0, int32_t priority = 0);

    // Eviction policy (default LRU = the historical behaviour). slru_hits is the SLRU protection threshold (>= 1).
    void set_evict_policy(server_radix_evict_policy p, uint32_t slru_hits = 2);
    server_radix_evict_policy get_evict_policy() const { return evict_policy; }

    // Slot erase: drop the nodes that exist only to serve `tokens` (the erased prompt) under `lora_fp`. Walks the matched path
    // and removes, deepest first, every matched node that is an idle (ref == 0) leaf, stopping at the first node that still has
    // other children or is locked, because that content is shared with other prompts. Returns the number of nodes removed.
    // SOURCE-READ: vllm/v1/core/sched/scheduler.py:2836 (reset_prefix_cache). Apache-2.0, own implementation.
    size_t purge_path_of(const llama_tokens & tokens, uint64_t lora_fp = 0);

    // Instrumentation (--verbose), per the row's "log hits/misses/evictions at verbose").
    struct stats_t {
        uint64_t hits = 0;         // match_prefix() calls returning > 0
        uint64_t misses = 0;       // match_prefix() calls returning 0
        uint64_t hit_tokens = 0;   // sum of tokens reused across hits
        uint64_t evictions = 0;    // idle leaves evicted to make room
        uint64_t evict_tokens = 0; // tokens held by those leaves
        uint64_t inserts = 0;      // insert() calls (successful, i.e. tree grew by >= 1 node)
    };
    const stats_t & get_stats() const { return stats; }

    size_t n_nodes() const;
    size_t n_free_seq_ids() const { return free_seq_ids.size(); }

    // Drop the whole tree (seq_rm every reserved id it currently owns) and reset the free list.
    // Used when disabling the cache after construction (e.g. a later-discovered incompatibility)
    // or in tests.
    void clear();

private:
    llama_context * ctx;
    llama_seq_id seq_id_base  = 0; // for clear()'s free-list rebuild
    int32_t      seq_id_count = 0;
    std::map<uint64_t, std::unique_ptr<server_radix_node>> roots; // one tree per LoRA fingerprint, created on first insert
    std::vector<server_radix_node *> locked; // nodes currently locked by an in-flight match_prefix()
    std::vector<llama_seq_id> free_seq_ids;  // unused reserved ids, LIFO
    uint64_t clock = 0;
    stats_t stats;
    server_radix_evict_policy evict_policy = server_radix_evict_policy::LRU;
    uint32_t                  slru_hits    = 2;

    server_radix_node * root_for(uint64_t lora_fp, bool create); // nullptr when absent and !create
    llama_seq_id claim_seq_id();                 // pops free_seq_ids, evicting one idle leaf first if empty
    bool evict_one_idle_leaf();                  // LRU across ref==0 leaves; false if none exist
    server_radix_node * split(server_radix_node * node, size_t common_len); // -> the new mid node
    void remove_node(server_radix_node * node);  // detach from parent, seq_rm its range, free its id
};
