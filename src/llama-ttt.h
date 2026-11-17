#pragma once

// In-Place TTT (arXiv 2604.06169) -- inference support for the down-projection fast weights.
// Row S69-FORK-INPLACE-TTT-2026-10-26-01.
//
// Semantics follow the OFFICIAL code (Apache-2.0), ByteDance-Seed/In-Place-TTT commit
// be2324829b0e91c8fd10a74d4b43714fde6676e1, inference_model/hf_qwen3/modeling_qwen3.py:
//   :117-152  per chunk of C tokens: y = W h (W = down_proj weight plus every earlier update), THEN
//             W += lr * contract("c h, c d, d e -> e h", H, conv(T), P); the last partial chunk never updates
//   :345-377  the decoder layer keeps the incomplete chunk's inputs (h, t) and W across calls
//   :102      conv = depthwise Conv1d, kernel 5, symmetric padding 2, applied to each chunk on its own
// No code is copied; this is a re-expression in ggml.
//
// Per-sequence state lives in llama_kv_cache_fork::ttt_l[il], one F32 row per sequence slot (slot = seq_id):
//   [ S       : n_ff * n_embd ]  the accumulated update, ggml layout of ffn_down (ne {n_ff, n_embd});
//                                 only read once the sequence is past its first chunk (pos >= C), so it is
//                                 never cleared -- position strictness makes every read follow its write
//   [ X tail  : n_embd * C    ]  FFN inputs (post ffn_norm) of the current, incomplete chunk; column j = pos % C
//   [ T tail  : n_embd * C    ]  target rows for that chunk -- only when target = input_embed (for hidden_states
//                                 the target IS the FFN input, so the X tail serves both)
// A ubatch must hold ONE sequence, consecutive positions continuing it, inside one chunk; the decode loop splits
// batches that way and refuses anything else (llama_ttt_check_ubatch in src/llama.cpp).

#include <cstdint>

struct llama_hparams;

struct llama_ttt_layout {
    int64_t n_s = 0;  // floats of S
    int64_t n_x = 0;  // floats of the X tail
    int64_t n_t = 0;  // floats of the T tail (0 unless target = input_embed)

    int64_t row() const { return n_s + n_x + n_t; }
};

llama_ttt_layout llama_ttt_row_layout(const llama_hparams & hparams, uint32_t il);

// Speculative decoding: the most draft tokens that may follow the last accepted token (at position pos_last)
// so that no DRAFT token completes a chunk. A chunk update runs when the token at position p with
// p % C == C - 1 is decoded; if that token were a rejected draft, the update could not be undone.
// The verify batch is [id_last @ pos_last, draft_1 @ pos_last + 1, ...]; id_last itself may complete a chunk.
// (the public llama_ttt_draft_cap(model, pos_last) in include/llama.h calls this)
//
// DEVIATION from the row brief, which named the cap `C - pos % C`: that lets the last draft land on residue C - 1
// (e.g. C = 8, pos_last = 3: cap 5 puts draft 4 at position 7, which completes the chunk). The cap here is
// r = pos_last % C; r == C - 1 ? C - 1 : C - 2 - r -- the drafts then end at residue <= C - 2, every rejected
// suffix is a tail of the current chunk, and seq_rm of it is always allowed.
int32_t llama_ttt_chunk_draft_cap(uint32_t chunk, int32_t pos_last);
