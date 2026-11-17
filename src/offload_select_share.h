// Page-selection reuse across decode STEPS and across LAYERS for the ik_offload page selector.
// Pure std-only decision logic, so it is unit-testable without a model (tests/test-offload-select-share.cpp).
//
// SPEC read from KTransformers (Apache-2.0) kvcache_attn.cpp:737-848 (select_every_n_steps, layer_step / layer_offset); behaviour only,
// no code copied. The across-steps half for ONE layer already lives in offload_attention_mask_op.cpp (RefreshState, refresh_interval).
// This header adds the two missing pieces as pure functions:
//   * layer sharing: with share_layers = S > 1 only layers il with il % S == offset % S run the selector (leaders); every other
//     layer reuses the kept-page list of the nearest leader at or below it;
//   * step reuse with an append: between full selections the kept list is reused, and the NEWEST page is added when the
//     sequence grew onto a new page (the one key written since the scan must not be invisible).
// Defaults (share_layers = 1, every_n = 1) mean "select every layer, every step" = today's behaviour exactly.
//
// NOT YET WIRED into mask_from_kq_op: the op's userdata is one static OffloadConfig shared by all layers, so it cannot know its layer
// index. See the row S69-KT-PICK-ATTN-SELECTION-REUSE-LAYER-TOKEN-STEP-2026-10-05-01 note.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace ik_offload {

// true when layer `il` must run the selector itself.
inline bool select_share_is_leader(int il, int share_layers, int offset) {
    if (share_layers <= 1) return true;
    const int off = ((offset % share_layers) + share_layers) % share_layers;
    return il % share_layers == off;
}

// the layer whose kept-page list layer `il` uses: itself when it leads (or when no leader precedes it), else the nearest leader below.
inline int select_share_leader(int il, int share_layers, int offset) {
    if (select_share_is_leader(il, share_layers, offset)) return il;
    for (int l = il - 1; l >= 0; --l) {
        if (select_share_is_leader(l, share_layers, offset)) return l;
    }
    return il;   // before the first leader: scan on its own
}

struct SelectReuse {
    bool              valid  = false;
    std::vector<char> keep;       // per page, 1 = kept (the last full selection)
    int64_t           n_kv   = 0; // attendable keys at that selection
    int64_t           age    = 0; // reuse steps since the selection
    int64_t           seq    = -1;// sequence / edit generation the list belongs to
};

// Record a full selection.
inline void select_reuse_store(SelectReuse & st, const std::vector<char> & keep, int64_t n_kv, int64_t seq) {
    st.valid = true;
    st.keep  = keep;
    st.n_kv  = n_kv;
    st.age   = 0;
    st.seq   = seq;
}

// Decide for one decode step. Returns true and fills `out` when the stored list may be reused; false means "run the selector".
// Not reusable when: nothing stored, every_n <= 1, the list is every_n steps old, the KV shrank (cell removed / context shift / edit),
// the sequence changed, or it grew by more than one key (not a decode chain).
inline bool select_reuse_step(SelectReuse & st, int64_t every_n, int64_t n_kv, int64_t page_size, int64_t seq, std::vector<char> & out) {
    if (!st.valid || every_n <= 1 || page_size <= 0) return false;
    if (st.seq != seq || n_kv < st.n_kv || n_kv > st.n_kv + 1 || st.age + 1 >= every_n) { st.valid = false; return false; }
    const int64_t nP = (n_kv + page_size - 1) / page_size;
    out.assign((size_t) nP, 0);
    for (size_t p = 0; p < st.keep.size() && p < (size_t) nP; ++p) out[p] = st.keep[p];
    if (n_kv > 0) out[(size_t) ((n_kv - 1) / page_size)] = 1;   // append the newest page
    st.keep  = out;
    st.n_kv  = n_kv;
    st.age  += 1;
    return true;
}

} // namespace ik_offload
