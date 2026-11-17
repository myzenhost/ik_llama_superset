// KIVI-style 2/4/8-bit KV quantisation, SIMULATED on the legacy f16 KV cache (kv_self). Row S69-FORK-KV-KIVI-2BIT-PORT-01.
//
// WHAT THIS IS: after every ubatch the tokens that have just left KIVI's full-precision residual window are fed through
// llama_kivi::cache (src/llama-kv-kivi.h, the unit-tested container) and the DEQUANTISED values are written back into the
// f16 K/V tensors in place. Attention (the ordinary CPU/CUDA flash-attn or non-FA path) then reads values that have been
// through the 2-bit round trip. So the NUMERICS of KIVI are real and reachable from llama-cli/llama-server/llama-perplexity;
// the MEMORY saving is NOT (the f16 tensors keep their size, the packed codes live in a host-side copy that is only used
// for accounting). A packed-read attention kernel is not implemented (see docs/agent-reports/wire-types-kv-gate.md).
//
// Preconditions (checked in llama_kivi_sim_init, which throws with a message so the flag is refused, not ignored):
//   f16 K and V cache, plain per-head layout (no MLA/latent/recurrent/compacted/split cache), head_dim % group == 0.
// Runtime assumption (checked per ubatch; on violation the simulation stops with a warning): one sequence, cell index == position.
#pragma once

#include <cstdint>

struct llama_context;
struct llama_batch;

struct llama_kivi_sim;

// throws std::runtime_error when the configuration cannot be honoured
llama_kivi_sim * llama_kivi_sim_init(llama_context & lctx);
void             llama_kivi_sim_free(llama_kivi_sim * sim, bool log_summary);

// call after the ubatch's graph has been computed; first_pos/n_tokens describe the ubatch (single sequence)
void llama_kivi_sim_after_ubatch(llama_context & lctx, int64_t n_tokens);

// observable counters (tests / logs)
uint64_t llama_kivi_sim_k_tokens_quantised(const llama_kivi_sim * sim);
uint64_t llama_kivi_sim_v_tokens_quantised(const llama_kivi_sim * sim);
