"""Row S69-ENGREV-RKNN-QUANT-RESIDUAL-LORA-2026-10-27-01: residual-LoRA generator for a quantised GGUF.
Run: python gguf-py/tests/test_quant_residual_lora.py   (or pytest). Fails before the port: the script does not exist."""
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "gguf" / "scripts"))
import gguf  # noqa: E402
import gguf_quant_residual_lora as G  # noqa: E402

F32, Q4 = gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.Q4_0


def _write(path, tensors, arch="llama"):
    w = gguf.GGUFWriter(str(path), arch)
    for n, (arr, qt) in tensors.items():
        w.add_tensor(n, arr if qt == F32 else gguf.quants.quantize(arr, qt), raw_dtype=qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _fixture(d):
    rng = np.random.default_rng(7)
    W = rng.standard_normal((64, 96)).astype(np.float32)
    V = rng.standard_normal((64, 64)).astype(np.float32)
    ref, q = Path(d) / "ref.gguf", Path(d) / "q.gguf"
    _write(ref, {"blk.0.attn_q.weight": (W, F32), "blk.0.attn_v.weight": (V, F32), "blk.0.attn_norm.weight": (W[0], F32)})
    _write(q, {"blk.0.attn_q.weight": (W, Q4), "blk.0.attn_v.weight": (V, Q4), "blk.0.attn_norm.weight": (W[0], F32)})
    return ref, q, W, V


def test_residual_adapter_closes_the_gap_and_negative_control():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:  # Windows: memmap readers keep files open
        ref, q, W, V = _fixture(d)
        out = Path(d) / "a.gguf"
        st = G.build(ref, q, out, rank=8)
        assert st["done"] == 2 and "blk.0.attn_norm.weight" in st["skipped"]
        assert st["e_after"] < st["e_before"] * 0.95, st
        r = gguf.GGUFReader(str(out))
        names = {t.name: t for t in r.tensors}
        a, b = names["blk.0.attn_q.weight.lora_a"], names["blk.0.attn_q.weight.lora_b"]
        assert tuple(int(x) for x in a.shape) == (96, 8) and tuple(int(x) for x in b.shape) == (8, 64)  # GGUF ne: [in, r], [r, out]
        assert float(r.get_field("adapter.lora.alpha").parts[-1][0]) == 0.0
        wq = G._as_f32({t.name: t for t in gguf.GGUFReader(str(q)).tensors}["blk.0.attn_q.weight"])
        A_, B_ = np.asarray(a.data).reshape(8, 96), np.asarray(b.data).reshape(64, 8)
        assert np.linalg.norm(W - (wq + B_ @ A_)) < np.linalg.norm(W - wq)
        # negative control: rank >= min(dims) recovers W (up to fp32)
        st2 = G.build(ref, q, Path(d) / "full.gguf", rank=64)
        assert st2["e_after"] < 1e-3 * st2["e_before"], st2
        # rank 0 is refused, not silently an empty adapter
        try:
            G.build(ref, q, Path(d) / "z.gguf", rank=0)
            raise AssertionError("rank 0 accepted")
        except ValueError:
            pass


if __name__ == "__main__":
    test_residual_adapter_closes_the_gap_and_negative_control()
    print("PASS")
