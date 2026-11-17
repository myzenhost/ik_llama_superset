# Reference logits for the tiny random-weight Inkling fixture, from the Apache-2.0 transformers implementation
# (transformers/models/inkling/modeling_inkling.py, InklingForCausalLM), run in eager fp32. test-inkling-parity compares the
# fork's CPU logits against these.
# PROGRESS-OK: a 4-layer 256-wide model over 40 tokens, seconds.  REWRITE-SAFE: writes only <fixture-dir>/ref_*.
#
# Usage: python ref_inkling_logits.py <fixture-dir> [n_tokens]
#   <fixture-dir> is the output of make_tiny_inkling.py (it holds ckpt_hf/config.json and weights.npz). Random weights only.
# Writes <fixture-dir>/ref_tokens.txt (one id per line) and <fixture-dir>/ref_logits.bin (float32, [n_tokens, n_unpadded]).
#
# Why the weights are mapped here rather than loaded through from_pretrained: the full conditional-generation class also builds
# the vision and audio towers at their production sizes, which a text-only fixture neither has nor needs. The mapping below is the
# one transformers' own conversion table applies to the original checkpoint names (conversion_mapping.py, "inkling_mm_model"):
# gate/up rows are stored interleaved (gate = even rows, up = odd rows), the expert bank likewise along dim 1, and the sconv
# kernels, norms and projections only change name. load_state_dict(strict=True) proves every tensor was placed.
#
# The token ids are fixed by a seeded generator and drawn from the UNPADDED vocab. 40 positions are enough to cross every
# boundary the graph has: the 24-token sliding window and rel extent of the local layers, the 32-token log-N floor (tau > 1
# from position 31 on) of the global layer, and the 4-wide short-conv history.
import json
import sys
from pathlib import Path

import numpy as np
import torch

fx = Path(sys.argv[1]).resolve()
n_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 40

from transformers.models.inkling import modeling_inkling as mi  # noqa: E402
from transformers.models.inkling.configuration_inkling import InklingTextConfig  # noqa: E402

tc = dict(json.loads((fx / "ckpt_hf" / "config.json").read_text(encoding="utf-8"))["text_config"])
tc.pop("model_type", None)
cfg = InklingTextConfig(**tc)
cfg._attn_implementation = "eager"
model = mi.InklingForCausalLM(cfg).float().eval()

W = {k.replace("__", "."): torch.from_numpy(v).float() for k, v in np.load(fx / "weights.npz").items()}
sd = {
    "model.embed_tokens.weight": W["model.llm.embed.weight"],
    "model.embed_norm.weight": W["model.llm.embed_norm.weight"],
    "model.norm.weight": W["model.llm.norm.weight"],
    "lm_head.weight": W["model.llm.unembed.weight"],
}
for il in range(cfg.num_hidden_layers):
    s = f"model.layers.{il}."
    o = f"model.llm.layers.{il}."
    ren = {
        "attn_norm.weight": "input_layernorm.weight", "mlp_norm.weight": "post_attention_layernorm.weight",
        "attn.wq_du.weight": "self_attn.q_proj.weight", "attn.wk_dv.weight": "self_attn.k_proj.weight",
        "attn.wv_dv.weight": "self_attn.v_proj.weight", "attn.wr_du.weight": "self_attn.r_proj.weight",
        "attn.wo_ud.weight": "self_attn.o_proj.weight", "attn.q_norm.weight": "self_attn.q_norm.weight",
        "attn.k_norm.weight": "self_attn.k_norm.weight", "attn.rel_logits_proj.proj": "self_attn.rel_logits_proj.proj",
        "attn.k_sconv.weight": "self_attn.k_sconv.conv1d.weight", "attn.v_sconv.weight": "self_attn.v_sconv.conv1d.weight",
        "attn_sconv.weight": "attn_sconv.conv1d.weight", "mlp_sconv.weight": "mlp_sconv.conv1d.weight",
    }
    for a, b in ren.items():
        sd[s + b] = W[o + a]
    if cfg.mlp_layer_types[il] == "dense":
        w13 = W[o + "mlp.w13_dn.weight"]
        sd[s + "mlp.gate_proj.weight"] = w13[0::2]
        sd[s + "mlp.up_proj.weight"] = w13[1::2]
        sd[s + "mlp.down_proj.weight"] = W[o + "mlp.w2_md.weight"]
        sd[s + "mlp.global_scale"] = W[o + "mlp.global_scale"]
    else:
        sd[s + "mlp.gate.weight"] = W[o + "mlp.gate.weight"]
        sd[s + "mlp.gate.e_score_correction_bias"] = W[o + "mlp.gate.bias"]
        sd[s + "mlp.gate.global_scale"] = W[o + "mlp.gate.global_scale"]
        e13 = W[o + "mlp.experts.w13_weight"]
        sd[s + "mlp.experts.gate_up_proj"] = torch.cat([e13[:, 0::2], e13[:, 1::2]], dim=1)
        sd[s + "mlp.experts.down_proj"] = W[o + "mlp.experts.w2_weight"]
        s13 = W[o + "mlp.shared_experts.shared_w13_weight"]
        sd[s + "mlp.shared_experts.gate_proj"] = s13[:, 0::2]
        sd[s + "mlp.shared_experts.up_proj"] = s13[:, 1::2]
        sd[s + "mlp.shared_experts.down_proj"] = W[o + "mlp.shared_experts.shared_w2_weight"]
model.load_state_dict(sd, strict=True)

g = torch.Generator().manual_seed(7)
ids = torch.randint(0, cfg.unpadded_vocab_size, (1, n_tokens), generator=g)
with torch.no_grad():
    out = model(input_ids=ids, use_cache=False)
logits = out.logits[0].float().numpy()
print("logits", logits.shape, "finite:", bool(np.isfinite(logits).all()), "absmax %.4f" % float(np.abs(logits).max()))

(fx / "ref_tokens.txt").write_text("\n".join(str(int(t)) for t in ids[0]) + "\n", encoding="utf-8")
logits[:, : cfg.unpadded_vocab_size].astype(np.float32).tofile(str(fx / "ref_logits.bin"))
print("wrote", fx / "ref_logits.bin", logits[:, : cfg.unpadded_vocab_size].shape)
