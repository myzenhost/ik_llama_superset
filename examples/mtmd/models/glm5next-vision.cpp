#include "models.h"

// The GLM-OCR ViT plus a clamp on the SwiGLU gate and up projections, which the per-block MLP and
// the merger both pick up from hparams.ffn_op (FFN_SILU_CLAMP).
// ref: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/modeling_glm5_next.py
//
// Adopted from unslothai/llama.cpp 2026-09-02, row S69-PORT-UNSLOTH-GLM5NEXT-ARCH-AND-KPOOL-01.
// This delegates rather than duplicating the graph: the ONLY difference from glm4v is the clamp,
// and that lives in hparams, not in the graph structure. Copying glm4v's body here would create a
// second graph to keep in sync with no behavioural difference to justify it.
//
// The three asserts are the port's real content. They are not defensive padding: each names a
// shape that WOULD build a graph and produce plausible-looking output while being wrong.
//   ffn_op != FFN_SILU_CLAMP      -> an unclamped MLP; the model runs and the activations are wrong
//   norm_embd_w present           -> glm4v's build() does not apply it, so it would be ignored
//   position_embeddings present   -> likewise ignored rather than used
ggml_cgraph * clip_graph_glm5next::build() {
    GGML_ASSERT(hparams.ffn_op == FFN_SILU_CLAMP);
    GGML_ASSERT(model.norm_embd_w == nullptr);
    GGML_ASSERT(model.position_embeddings == nullptr);

    return clip_graph_glm4v::build();
}
