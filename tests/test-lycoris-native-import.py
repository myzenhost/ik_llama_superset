# PROGRESS-OK: pure-python unit test, sub-second. SWALLOW-OK: none.
"""Unit test for convert_lycoris_to_peft.py (row S69-ENGINE-MERGE-ADAPTER-FORMAT-IMPORTERS-LYCORIS-PYREFT-2026-11-06-01).
Run: python tests/test-lycoris-native-import.py   (needs numpy only)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("cl2p", ROOT / "convert_lycoris_to_peft.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

MODS = {
    "model_layers_0_self_attn_q_proj": "model.layers.0.self_attn.q_proj",
    "model_layers_0_mlp_down_proj": "model.layers.0.mlp.down_proj",
}
z = lambda *s: np.zeros(s, dtype=np.float32)  # noqa: E731
al = lambda v: np.array(float(v), dtype=np.float32)  # noqa: E731


def loha(name, r, alpha):
    return {f"lora_unet_{name}.hada_w1_a": z(8, r), f"lora_unet_{name}.hada_w1_b": z(r, 6),
            f"lora_unet_{name}.hada_w2_a": z(8, r), f"lora_unet_{name}.hada_w2_b": z(r, 6),
            f"lora_unet_{name}.alpha": al(alpha)}


def refused(state, **kw):
    try:
        m.convert_state(state, MODS, **kw)
    except m.LycorisRefusal as e:
        return str(e)
    return None


def test_loha_positive():
    st = {**loha("model_layers_0_self_attn_q_proj", 4, 2), **loha("model_layers_0_mlp_down_proj", 2, 2)}
    kind, out, ranks, alphas = m.convert_state(st, MODS)
    assert kind == "LOHA"
    assert ranks == {"model.layers.0.self_attn.q_proj": 4, "model.layers.0.mlp.down_proj": 2}
    assert alphas["model.layers.0.self_attn.q_proj"] == 2.0
    assert "base_model.model.model.layers.0.self_attn.q_proj.hada_w1_a" in out
    assert not any(k.endswith(".alpha") for k in out)
    cfg = m.adapter_config(kind, ranks, alphas, None)
    assert cfg["peft_type"] == "LOHA" and cfg["r"] == 4


def test_lokr_decomposed_and_full():
    n = "model_layers_0_self_attn_q_proj"
    st = {f"lora_unet_{n}.lokr_w1": z(2, 2), f"lora_unet_{n}.lokr_w2_a": z(4, 3),
          f"lora_unet_{n}.lokr_w2_b": z(3, 3), f"lora_unet_{n}.alpha": al(3)}
    kind, _out, ranks, _a = m.convert_state(st, MODS)
    assert kind == "LOKR" and ranks["model.layers.0.self_attn.q_proj"] == 3
    full = {f"lora_unet_{n}.lokr_w1": z(2, 2), f"lora_unet_{n}.lokr_w2": z(4, 3), f"lora_unet_{n}.alpha": al(5)}
    assert m.convert_state(full, MODS)[2]["model.layers.0.self_attn.q_proj"] == 5  # rank = forced alpha


def test_negative_controls():
    n = "model_layers_0_self_attn_q_proj"
    base = loha(n, 4, 2)
    assert "dora_scale" in refused({**base, f"lora_unet_{n}.dora_scale": z(8, 1)})
    assert "tucker" in refused({**base, f"lora_unet_{n}.hada_t1": z(4, 4, 3, 3)})
    mixed = {**base, **{k.replace(n, "model_layers_0_mlp_down_proj").replace("hada_w", "lokr_w"): v for k, v in base.items()}}
    assert refused(mixed)
    assert "no base module" in refused(loha("not_a_module", 4, 2))
    assert "prefix" in refused({"foo_bar.alpha": al(1)})
    assert refused({**base, f"lora_unet_{n}.lora_down.weight": z(4, 6)})  # other algorithm
    ambiguous = {"a_b": ""}
    try:
        m.convert_state(loha("a_b", 2, 2), ambiguous)
        raise AssertionError("ambiguous underscore name must be refused")
    except m.LycorisRefusal:
        pass


if __name__ == "__main__":
    for k, f in sorted(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
    print("PASS")
