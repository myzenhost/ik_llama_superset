# Tiny random LFM2 GGUF WITH a toy gpt2-BPE vocab (EOS id 2, newline piece at id 3, one merge), for tests/probe-xtc-eog-exempt.cpp.
# Derived at run time from make_tiny_gguf.py (same weights, same converter); only set_vocab differs, and only the dense model is written.
# A vocab with an EMPTY merges list is rejected by the loader ("cannot find tokenizer merges"), hence the single merge.
# Usage: python make_tiny_vocab_gguf.py <worktree-root> <out-dir>      -> <out-dir>/dense.gguf
# Row: S69-EXLLAMAV2-PICK-XTC-EXEMPT-EOG-NEWLINE-2026-09-29-01
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "make_tiny_gguf.py").read_text(encoding="utf-8")
needle = "    m.set_vocab = m._set_vocab_none"
assert needle in src, "make_tiny_gguf.py changed: set_vocab override line not found"
custom = '''    def _toyvocab(m=m):
        toks = ["<pad>", "<s>", "</s>", "\u010a"] + ["t%d" % i for i in range(4, V - 1)] + ["t4t5"]
        types = [gguf.TokenType.CONTROL] * 3 + [gguf.TokenType.NORMAL] * (V - 3)
        w = m.gguf_writer
        w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre("default")
        w.add_token_list(toks); w.add_token_types(types); w.add_token_merges(["t4 t5"])
        w.add_bos_token_id(1); w.add_eos_token_id(2); w.add_pad_token_id(0)
    m.set_vocab = _toyvocab'''
lines = src.split("\n")
i = next(k for k, l in enumerate(lines) if l.startswith(needle))
lines[i] = custom
src = "\n".join(lines).replace("for kind, cfg in CONFIGS.items():", 'for kind, cfg in [("dense", CONFIGS["dense"])]:')
exec(compile(src, str(HERE / "make_tiny_gguf.py"), "exec"))
