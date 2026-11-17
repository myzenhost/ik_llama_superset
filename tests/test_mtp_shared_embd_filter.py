#!/usr/bin/env python3
"""--mtp-shared-embd: with --mtp, the exported draft must LEAVE OUT the shared token embeddings, output norm and
lm head, and must still carry the NextN block; without the flag nothing changes.

Drives the real filter_tensors of every converter class that was patched (glm x3, qwen mixin, command_r, dots3,
bailingmoe3). Control: the same call with mtp_shared_embd False keeps the three tensors, so a pass means the flag
is what removed them (and not the filter being broken). No model files; milliseconds.

   python tests/test_mtp_shared_embd_filter.py <tree-root>

Row: S69-UNSLOTH-ADOPTABLE-BRANCHES-NOT-YET-PORTED-01 (unslothai/llama.cpp f36eda84 converter hunks, MIT; tested here).
"""
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "gguf-py"))

import conversion.base as base  # noqa: E402
import conversion.glm as glm  # noqa: E402
import conversion.qwen as qwen  # noqa: E402
import conversion.command_r as command_r  # noqa: E402
import conversion.dots3 as dots3  # noqa: E402
import conversion.bailingmoe3 as bailingmoe3  # noqa: E402

HF = ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")
BM3 = ("model.word_embeddings.weight", "model.norm.weight", "lm_head.weight")
N_MAIN = 4
MTP_TENSOR = "model.layers.%d.eh_proj.weight" % N_MAIN        # a NextN-block tensor: index >= main layer count
MAIN_TENSOR = "model.layers.1.mlp.up_proj.weight"              # an ordinary main-block tensor

cases = [
    ("Glm4MoeModel", glm.Glm4MoeModel, HF, "_n_main_layers"),
    ("Glm4MoeLiteModel", glm.Glm4MoeLiteModel, HF, "_n_main_layers"),
    ("GlmMoeDsaModel", glm.GlmMoeDsaModel, HF, "_n_main_layers"),
    ("Cohere2MoeModel", command_r.Cohere2MoeModel, HF, "_n_main_layers"),
    ("Dots3NoteModel", dots3.Dots3NoteModel, HF, "_n_main_layers"),
    ("BailingMoeV3Model", bailingmoe3.BailingMoeV3Model, BM3, "_main_layers"),
    ("Qwen3NextModel (mixin)", qwen.Qwen3NextModel, HF, "_original_block_count"),
]

fails = 0
n_ok = 0


def check(cond, msg):
    global fails, n_ok
    if cond:
        n_ok += 1
        print("ok    " + msg)
    else:
        fails += 1
        print("FAIL  " + msg)


def run(cls, name):
    return cls.filter_tensors((name, lambda: None))


for label, cls, shared_names, attr in cases:
    saved = {a: getattr(cls, a, None) for a in ("mtp_only", "no_mtp", "mtp_shared_embd", attr)}
    try:
        setattr(cls, attr, N_MAIN)
        cls.no_mtp = False
        cls.mtp_only = True
        for shared in (False, True):
            cls.mtp_shared_embd = shared
            for n in shared_names:
                r = run(cls, n)
                if shared:
                    check(r is None, "%s --mtp --mtp-shared-embd drops %s" % (label, n))
                else:
                    check(r is not None, "%s --mtp (control, flag off) keeps %s" % (label, n))
            r = run(cls, "mtp.layers.0.mlp.up_proj.weight" if cls is qwen.Qwen3NextModel else MTP_TENSOR)
            check(r is not None, "%s keeps the NextN tensor (shared=%s)" % (label, shared))
        # without --mtp at all the flag must change nothing: a main-block tensor survives
        cls.mtp_only = False
        cls.mtp_shared_embd = True
        r = run(cls, MAIN_TENSOR)
        check(r is not None, "%s: flag without --mtp changes nothing (main tensor kept)" % label)
        r = run(cls, shared_names[0])
        check(r is not None, "%s: flag without --mtp keeps the embedding" % label)
    finally:
        for a, v in saved.items():
            if v is not None:
                setattr(cls, a, v)

check(getattr(base.ModelBase, "mtp_shared_embd", None) is False, "ModelBase.mtp_shared_embd defaults to False")
print("RESULT: %s (%d ok, %d failed)" % ("PASS" if fails == 0 else "FAIL", n_ok, fails))
sys.exit(1 if fails else 0)
