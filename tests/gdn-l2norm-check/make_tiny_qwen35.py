# Tiny random-weight Qwen3.5 (gated-delta-net hybrid) checkpoint, converted with this tree's own converter, plus the transformers
# reference logits for a fixed token sequence. Used by the GDN q/k l2-norm A/B (row S69-UNSLOTH-TRAITS-ADOPTED-GDN-L2NORM-AND-PENALTY-INDEX-01):
# test-inkling-parity is arch-agnostic and compares the fork's logits with these.
# PROGRESS-OK: a 4-layer 256-wide model, seconds.  REWRITE-SAFE: writes only OUT.  RANDOM WEIGHTS ONLY.
#
# Usage: python make_tiny_qwen35.py <worktree-root> <out-dir> [qk_scale]
#   qk_scale (default 1.0) multiplies the q and k rows of every linear-attention in_proj_qkv. With a small value (5e-5) the q/k rows reaching the
#   l2 norm have sum(x^2) of the order of its epsilon, which is the only regime in which flash-linear-attention's ADDITIVE epsilon and ggml_l2_norm's
#   CLAMPED divisor disagree; with 1.0 the rows are ordinary and the two arms must agree.
# Writes <out-dir>/model_f32.gguf, ref_tokens.txt, ref_logits.bin (float32 [n_tokens, vocab]) and the HF checkpoint in <out-dir>/hf/.
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
QK_SCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
N_TOKENS = 40
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gguf-py"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers  # noqa: E402

import gguf  # noqa: E402
from conversion import load_all_models  # noqa: E402
from conversion.base import ModelBase, ModelType  # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq  # noqa: E402
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402

load_all_models()

H, NH, NKV, HD = 256, 4, 2, 64
LKH, LVH, LKD, LVD, K = 2, 4, 64, 64, 4        # linear attention: key heads, value heads, key dim, value dim, conv kernel
N_LAYER, FF = 4, 512
V = 320                                         # padded vocab; the tokenizer below has 262 real entries
EOS = 261

# GATE-CLAIM-OK: code comment about library behaviour, no device state asserted.
# The reference must be transformers' pure-torch path: when flash-linear-attention is importable the layers bind its triton kernels, which
# cannot take host-memory tensors; the pure-torch gated delta rule is also the readable statement of the l2 norm under test (additive epsilon).
mq.chunk_gated_delta_rule = None
mq.fused_recurrent_gated_delta_rule = None
mq.causal_conv1d_fn = None
mq.causal_conv1d_update = None
mq.FusedRMSNormGated = None

torch.manual_seed(20261005)
cfg = Qwen3_5TextConfig(
    vocab_size=V, hidden_size=H, intermediate_size=FF, num_hidden_layers=N_LAYER, num_attention_heads=NH, num_key_value_heads=NKV,
    head_dim=HD, linear_conv_kernel_dim=K, linear_key_head_dim=LKD, linear_value_head_dim=LVD, linear_num_key_heads=LKH,
    linear_num_value_heads=LVH, layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    max_position_embeddings=256, rms_norm_eps=1e-6, tie_word_embeddings=False, pad_token_id=0, bos_token_id=EOS, eos_token_id=EOS,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10], "mrope_interleaved": True},
)
cfg._attn_implementation = "eager"
model = mq.Qwen3_5ForCausalLM(cfg).float().eval()
with torch.no_grad():
    for n, p in model.named_parameters():
        if p.ndim >= 2:
            p.normal_(0.0, 0.08)
        elif n.endswith("norm.weight") or n.endswith("layernorm.weight"):
            p.copy_(1.0 + 0.1 * torch.randn_like(p))
    for il, lt in enumerate(cfg.layer_types):
        if lt != "linear_attention":
            continue
        qkv = model.model.layers[il].linear_attn.in_proj_qkv.weight
        qkv[: 2 * LKH * LKD] *= QK_SCALE                    # q rows then k rows
(OUT / "hf").mkdir(exist_ok=True)


def make_tokenizer(d: Path):
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    vocab["Ġt"] = 256
    vocab["th"] = 257
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[("Ġ", "t"), ("t", "h")]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    specials = ["<|bos_unused|>", "<|image_bos|>", "<|end_message|>", "<|content_model_end_sampling|>"]
    tok.add_special_tokens([AddedToken(s, special=True, normalized=False) for s in specials])
    assert tok.get_vocab_size() == 262
    tok.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 256,
        "added_tokens_decoder": {str(tok.token_to_id(s)): {"content": s, "special": True, "normalized": False, "lstrip": False, "rstrip": False, "single_word": False} for s in specials},
    }), encoding="utf-8")


model.save_pretrained(str(OUT / "hf"), safe_serialization=True)
make_tokenizer(OUT / "hf")
conf = json.loads((OUT / "hf" / "config.json").read_text(encoding="utf-8"))
conf["architectures"] = ["Qwen3_5ForCausalLM"]
(OUT / "hf" / "config.json").write_text(json.dumps(conf), encoding="utf-8")

# the tiny tokenizer matches no known pre-tokenizer hash; name the one the architecture uses so the converter does not refuse
from conversion.base import TextModel  # noqa: E402
TextModel.get_vocab_base_pre = lambda self, tokenizer: "qwen35"
cls = ModelBase.from_model_architecture("Qwen3_5ForCausalLM", model_type=ModelType.TEXT)
cls.no_mtp = True            # the converter's CLI sets these class flags; this checkpoint has no MTP head
cls.mtp_only = False
cls.mtp_shared_embd = False
m = cls(OUT / "hf", gguf.LlamaFileType.ALL_F32, OUT / "model_f32.gguf", is_big_endian=False, use_temp_file=False, eager=True)
m.write()

g = torch.Generator().manual_seed(11)
ids = torch.randint(0, 262, (1, N_TOKENS), generator=g)
with torch.no_grad():
    logits = model(input_ids=ids, use_cache=False).logits[0].float().numpy()
(OUT / "ref_tokens.txt").write_text("\n".join(str(int(t)) for t in ids[0]) + "\n", encoding="utf-8")
logits.astype(np.float32).tofile(str(OUT / "ref_logits.bin"))
print("wrote", OUT, "logits", logits.shape, "absmax %.4f" % float(np.abs(logits).max()), "finite", bool(np.isfinite(logits).all()), flush=True)
