# End-to-end conversion test for the DFlash-2 draft converter (conversion/dflash2.py). PROGRESS-OK. REWRITE-SAFE: writes only a temp dir.
# Usage: python test_dflash2_conversion.py <worktree-root>    (the dir holding convert_hf_to_gguf.py, conversion/ and gguf-py/)
# Builds a TINY random DFlash-2 checkpoint (2 layers), converts it IN-PROCESS with the registered converter (tokenizer step stubbed:
# tensors and hyper-parameters are the subject), reads the GGUF back and requires:
#   (a) the architecture string is "dflash" and the five dflash.* hyper-parameter keys, target_layers (one-based) and the mask token id are written;
#   (b) fc.weight and enc.output_norm.weight (the names src/llama-model.cpp loads) equal the original fc / hidden_norm tensors;
#   (c) blk.N.attn_conv_base / ffn_conv_base carry NO suffix and ..._conv_proj carry ".weight" (src/llama-load-tensors.cpp:3049-3055);
#   (d) selector_predecessor/successor/hidden carry ".weight" (the HF codebook names have none: normalize_tensor_name adds it).
# CONTROLS: a checkpoint without dflash_config.selector_rank must be REFUSED; an unregistered architecture name must not resolve.
import json
import sys
import tempfile
from pathlib import Path

# run as a script: the worktree root is argv[1]; under pytest it is the checkout this file is in
ROOT = Path(sys.argv[1]).resolve() if __name__ == "__main__" and len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gguf-py"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import gguf  # noqa: E402
from conversion import load_all_models  # noqa: E402
from conversion.base import ModelBase, ModelType  # noqa: E402

load_all_models()

H, NH, NKV, HD, FF, V, L, NT, K, RANK, TOPK = 32, 4, 2, 8, 48, 64, 2, 3, 4, 8, 2
g = torch.Generator().manual_seed(77)
r = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32)

BASE_CFG = {
    "architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "hidden_size": H, "num_hidden_layers": L,
    "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD, "intermediate_size": FF, "vocab_size": V,
    "max_position_embeddings": 128, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "has_embed_tokens": False,
    "tie_word_embeddings": False, "block_size": 4,
    "dflash_config": {"mask_token_id": 3, "target_layer_ids": [1, 5, 9], "conv_kernel_size": K, "conv_group_size": 2,
                      "selector_rank": RANK, "selector_top_k": TOPK},
}

FC, HIDDEN_NORM = r(H, NT * H), r(H)
TENSORS = {"model.fc.weight": FC, "model.hidden_norm.weight": HIDDEN_NORM, "model.norm.weight": r(H),
           "model.candidate_selector.predecessor_codebook": r(RANK, V), "model.candidate_selector.successor_codebook": r(RANK, V),
           "model.candidate_selector.hidden_projection.weight": r(H, RANK)}
for i in range(L):
    p = f"model.layers.{i}"
    TENSORS.update({
        f"{p}.input_layernorm.weight": r(H), f"{p}.post_attention_layernorm.weight": r(H),
        f"{p}.self_attn.q_proj.weight": r(NH * HD, H), f"{p}.self_attn.k_proj.weight": r(NKV * HD, H),
        f"{p}.self_attn.v_proj.weight": r(NKV * HD, H), f"{p}.self_attn.o_proj.weight": r(H, NH * HD),
        f"{p}.self_attn.q_norm.weight": r(HD), f"{p}.self_attn.k_norm.weight": r(HD),
        f"{p}.mlp.gate_proj.weight": r(FF, H), f"{p}.mlp.up_proj.weight": r(FF, H), f"{p}.mlp.down_proj.weight": r(H, FF),
        f"{p}.attention_conv.base_kernel": r(H, K, 2), f"{p}.attention_conv.kernel_projection.weight": r(H, 6),
        f"{p}.mlp_conv.base_kernel": r(H, K, 2), f"{p}.mlp_conv.kernel_projection.weight": r(H, 6),
    })


def write_ckpt(d: Path, cfg: dict):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    save_file({k: v.contiguous() for k, v in TENSORS.items()}, str(d / "model.safetensors"))


def convert(d: Path, out: Path):
    cls = ModelBase.from_model_architecture("DFlash2DraftModel", model_type=ModelType.TEXT)
    m = cls(d, gguf.LlamaFileType.ALL_F32, out, is_big_endian=False, use_temp_file=False, eager=True, target_model_dir=d)
    m.set_vocab = lambda: None   # tokenizer is out of scope; the subject is tensors and hyper-parameters
    m.write()


def run() -> bool:
    ok = True
    with tempfile.TemporaryDirectory(prefix="dflash2_conv_", ignore_cleanup_errors=True) as td:
        td = Path(td)
        write_ckpt(td / "good", BASE_CFG)
        try:
            convert(td / "good", td / "good.gguf")
        except Exception as e:  # noqa: BLE001 -- the failure IS the result; printed with its type
            print("CONVERSION FAILED:", type(e).__name__, e)
            print("RESULT: FAIL")
            return False
        rd = gguf.GGUFReader(str(td / "good.gguf"))
        tens = {t.name: np.array(t.data, copy=True) for t in rd.tensors}
        kv = {}
        for f in rd.fields.values():
            try:
                kv[f.name] = f.contents()
            except Exception:  # noqa: BLE001 -- unrenderable field is listed as such, not dropped
                kv[f.name] = "<unreadable>"
        print("converted:", len(tens), "tensors"); print("  names:", sorted(n for n in tens if "conv" in n))

        want_keys = {"general.architecture": "dflash", "dflash.block_size": 4, "dflash.conv_kernel_size": K, "dflash.conv_group_size": 2,
                     "dflash.selector_rank": RANK, "dflash.selector_top_k": TOPK, "tokenizer.ggml.mask_token_id": 3}
        for k, v in want_keys.items():
            got = kv.get(k, "ABSENT")
            same = got == v
            print("  key %-32s %-10s %s" % (k, got, "OK" if same else "WANT %r" % (v,)))
            ok &= bool(same)
        tl = kv.get("dflash.target_layers", "ABSENT")
        tl_ok = list(tl) == [2, 6, 10] if tl != "ABSENT" else False
        print("  key dflash.target_layers (one-based)   %s %s" % (tl, "OK" if tl_ok else "WANT [2, 6, 10]"))
        ok &= tl_ok

        def eq(name, ref):
            got = tens.get(name)
            same = got is not None and got.shape == tuple(ref.shape) and np.array_equal(got.astype(np.float32), ref.numpy())
            print("  tensor %-36s %s" % (name, "MATCH" if same else "MISMATCH/ABSENT (%s)" % (None if got is None else got.shape,)))
            return bool(same)

        ok &= eq("fc.weight", FC)
        ok &= eq("enc.output_norm.weight", HIDDEN_NORM)
        for i in range(L):
            for nm, hf in ((f"blk.{i}.attn_conv_base", "attention_conv.base_kernel"), (f"blk.{i}.ffn_conv_base", "mlp_conv.base_kernel"),
                           (f"blk.{i}.attn_conv_proj.weight", "attention_conv.kernel_projection.weight"), (f"blk.{i}.ffn_conv_proj.weight", "mlp_conv.kernel_projection.weight")):
                ok &= eq(nm, TENSORS[f"model.layers.{i}.{hf}"])
        for nm, hf in (("selector_predecessor.weight", "predecessor_codebook"), ("selector_successor.weight", "successor_codebook"),
                       ("selector_hidden.weight", "hidden_projection.weight")):
            ok &= eq(nm, TENSORS[f"model.candidate_selector.{hf}"])

        # CONTROL 1: a checkpoint missing selector_rank must be refused, not converted with a guess
        bad = json.loads(json.dumps(BASE_CFG))
        del bad["dflash_config"]["selector_rank"]
        write_ckpt(td / "bad", bad)
        try:
            convert(td / "bad", td / "bad.gguf")
            refused = False
        except ValueError as e:
            refused = "selector_rank" in str(e)
        print("CONTROL missing selector_rank is refused:", refused, "(must be True)")
        ok &= refused

        # CONTROL 2: an unregistered architecture must not resolve through the same lookup
        try:
            ModelBase.from_model_architecture("DFlash2NoSuchModel", model_type=ModelType.TEXT)
            unresolved = False
        except NotImplementedError:
            unresolved = True
        print("CONTROL unregistered architecture is rejected:", unresolved, "(must be True)")
        ok &= unresolved

    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


def test_dflash2_conversion():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
