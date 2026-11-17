// llama-embed-pooling.h — engine-side embedding pooling math for the JusticianAI ik_llama fork.
// S69-IKLLAMA-POOLING-ATTENTION-01. Mirrors cortex/embedding/harness_pooling.py so harness + engine agree
// (DEC-S69-HARNESS-EMBED-POOLING). Grounded by NV-Embed modeling_nvembed.py::LatentAttentionModel (learned
// latents = query, hidden states = K/V cross-attn + FF + mean) — the param-free variant here uses the
// masked mean as the query (training-free), which is the harness-equivalent the governing principle calls for.
//
// Pure host C++ (no ggml dep) so it standalone-compiles + self-tests, exactly like llama-kv-snapkv. The graph
// path in llm_build_context::build_pooling() calls these on the final per-token hidden states for
// LLAMA_POOLING_TYPE_ATTENTION; the learned NV-Embed weld (cross-attn weights from gguf) is a later step.
#pragma once
#include <cstdint>
#include <vector>

namespace justician_embed {

// pooling modes — match LLAMA_POOLING_TYPE_* (mean/cls/last existing; attention=new) + max.
enum class PoolMode { Mean, Cls, Last, Attention, Max };

// h = row-major [n_tok * dim] hidden states; mask[n_tok] (1=valid,0=pad), or empty=all valid.
// out = [dim] pooled embedding. Attention = param-free latent-attn (query=masked-mean, softmax token·query).
std::vector<float> pool(const std::vector<float>& h, int n_tok, int dim,
                        PoolMode mode, const std::vector<int8_t>& mask = {});

// Matryoshka truncation: keep first `dim` of each [in_dim] row + L2-renormalize. in-place-safe copy.
std::vector<float> mrl_truncate(const std::vector<float>& v, int in_dim, int keep_dim);

int _selftest();   // 0 = pass; pure host, no ggml/CUDA

} // namespace justician_embed
