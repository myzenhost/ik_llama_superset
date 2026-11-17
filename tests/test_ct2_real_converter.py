#!/usr/bin/env python3
"""The REAL CTranslate2 converter's output (ctranslate2.converters.TransformersConverter, every CPU quantisation) through convert_hf_to_gguf.py
and llama-server, against the real CTranslate2 engine and against the original Hugging Face model.

   python tests/test_ct2_real_converter.py <tree-root> <llama-server>

tests/test_ct2_gguf_e2e.py uses an independent WRITER of model.bin; this test removes the last doubt about that writer: the model.bin files here
are produced by CTranslate2's own converter from a random-weight Hugging Face Llama (GQA; with and without attention biases), for float32, float16,
bfloat16, int8, int8_float16, int8_bfloat16 and int16. For each:
  (1) the log-probability of every token of a prompt from the real CTranslate2 engine (compute_type float32, which dequantises the stored weights as
      the engine does at load time) must equal llama-server's prompt_logprobs for the GGUF converted from the same model.bin (max error < 1e-4);
  (2) for float32 the same numbers must equal the original Hugging Face model's (torch) to 1e-4: the whole chain keeps the model;
  (3) for float32 the GGUF converted straight from the HF directory (the fork's pre-existing path) gives the same numbers too.
Controls: the lossy formats differ from float32 under the real engine (so a converter that ignored the quantisation scales could not pass), and
the bias model differs from the no-bias one.
"""
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
server_bin = sys.argv[2]
sys.path.insert(0, str(root / "tests"))
sys.argv = [sys.argv[0], str(root), server_bin]

import test_ct2_gguf_e2e as E  # noqa: E402

T = E.T
check = T.check
np = E.np

try:
    import torch  # noqa: E402
    from ctranslate2.converters import TransformersConverter  # noqa: E402
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM  # noqa: E402
    from transformers import MistralConfig, MistralForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM  # noqa: E402
except Exception as e:
    print("SKIP: %s" % e)
    sys.exit(0)

QUANT = ("float32", "float16", "bfloat16", "int8", "int8_float16", "int8_bfloat16", "int16")


def make_hf(d, attention_bias, seed):
    vocab = E.make_tokenizer(d)
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128, rms_norm_eps=1e-5, rope_theta=500000.0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False,
                      attention_bias=attention_bias)
    torch.manual_seed(seed)
    m = LlamaForCausalLM(cfg)
    for p in m.parameters():
        torch.nn.init.normal_(p, 0, 0.25)
    m.save_pretrained(d)
    return m.eval(), vocab


FAMILIES = {   # name -> (config class, model class, extra config) ; all share the Llama-family CT2 layout (RMSNorm, SwiGLU, HF-layout rotary)
    "mistral": (MistralConfig, MistralForCausalLM, dict(sliding_window=None)),
    "qwen2": (Qwen2Config, Qwen2ForCausalLM, dict()),
    "qwen3": (Qwen3Config, Qwen3ForCausalLM, dict(head_dim=16)),
}


def make_family(name, d, seed):
    cfgc, modc, extra = FAMILIES[name]
    vocab = E.make_tokenizer(d)
    cfg = cfgc(vocab_size=len(vocab) + (1 if name == 'qwen2' else 0),   # CT2's Qwen2 loader appends one pad token to the vocabulary
               hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
               max_position_embeddings=128, rms_norm_eps=1e-5, rope_theta=500000.0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False, **extra)
    torch.manual_seed(seed)
    m = modc(cfg)
    for p in m.parameters():
        torch.nn.init.normal_(p, 0, 0.25)
    m.save_pretrained(d)
    return m.eval(), vocab


def hf_logprobs(model, ids):
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([ids])).logits[0].float(), -1)
    return [float(lp[t, ids[t + 1]]) for t in range(len(ids) - 1)]


def convert_gguf(src, tok, out):
    r = E.subprocess.run([sys.executable, str(root / "convert_hf_to_gguf.py"), str(src), "--outfile", str(out), "--outtype", "f32"]
                         + (["--ct2-tokenizer-dir", str(tok)] if tok else []), capture_output=True, text=True, cwd=str(root))
    return r.returncode == 0 and Path(out).exists(), (r.stderr or r.stdout)[-600:]


def fork_logprobs(gguf, ids):
    srv = E.Server(gguf)
    try:
        res = srv.post("/completion", {"prompt": ids, "n_predict": 1, "temperature": 0.0, "prompt_logprobs": True, "cache_prompt": False})
    finally:
        srv.close()
    return res["prompt_logprobs"]["logprobs"][1:]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(5)
        results = {}
        for tag, bias, seed in (("llama", False, 0), ("llama-bias", True, 1)):
            hf = Path(tmp) / ("hf-" + tag)
            model, vocab = make_hf(hf, bias, seed)
            ids = [1] + [int(x) for x in rng.integers(3, len(vocab), size=14)]
            with torch.no_grad():
                lp = torch.log_softmax(model(torch.tensor([ids])).logits[0].float(), -1)
            hf_ref = [float(lp[t, ids[t + 1]]) for t in range(len(ids) - 1)]
            ok, err = convert_gguf(hf, None, Path(tmp) / (tag + "-direct.gguf"))
            check(f"[{tag}] convert_hf_to_gguf.py on the HF directory (existing path)", ok, err)
            direct = fork_logprobs(Path(tmp) / (tag + "-direct.gguf"), ids) if ok else None
            if direct:
                e = max(abs(a - b) for a, b in zip(direct, hf_ref))
                check(f"[{tag}] fork from the HF directory == torch (max err {e:.1e})", e < 1e-4)
            for q in QUANT:
                name = f"{tag}/{q}"
                ct2 = Path(tmp) / f"ct2-{tag}-{q}"
                TransformersConverter(str(hf)).convert(str(ct2), quantization=q, force=True)
                check(f"[{name}] the real converter wrote model.bin", (ct2 / "model.bin").is_file())
                gen = E.ctranslate2.Generator(str(ct2), device="cpu", compute_type="float32")
                vocab_ct2 = json.loads((ct2 / "vocabulary.json").read_text())
                ref = gen.score_batch([[vocab_ct2[i] for i in ids]])[0].log_probs
                ok, err = convert_gguf(ct2, hf, Path(tmp) / f"{tag}-{q}.gguf")
                check(f"[{name}] convert_hf_to_gguf.py converts the CT2 directory", ok, err)
                if not ok:
                    continue
                got = fork_logprobs(Path(tmp) / f"{tag}-{q}.gguf", ids)
                e = max(abs(a - b) for a, b in zip(got, ref))
                check(f"[{name}] fork == real CTranslate2 engine (max err {e:.1e}, {len(got)} tokens)", e < 1e-4 and len(got) == len(ref) == len(ids) - 1,
                      f"got={got[:3]} ref={ref[:3]}")
                results[(tag, q)] = ref
                if q == "float32":
                    e2 = max(abs(a - b) for a, b in zip(ref, hf_ref))
                    check(f"[{name}] real CTranslate2 == torch (max err {e2:.1e})", e2 < 1e-4)
                    e3 = max(abs(a - b) for a, b in zip(got, hf_ref))
                    check(f"[{name}] fork from model.bin == torch (max err {e3:.1e})", e3 < 1e-4)
        # other families that share the layout: HF level (the converted directory loaded by transformers) against the real engine and torch
        import convert_ct2_to_hf as C
        for fam in FAMILIES:
            hf = Path(tmp) / ("hf-" + fam)
            model, vocab = make_family(fam, hf, 3)
            ids = [1] + [int(x) for x in rng.integers(3, len(vocab), size=14)]
            hf_ref = hf_logprobs(model, ids)
            for q in ("float32", "int8", "bfloat16"):
                name = f"{fam}/{q}"
                ct2 = Path(tmp) / f"ct2-{fam}-{q}"
                try:
                    TransformersConverter(str(hf)).convert(str(ct2), quantization=q, force=True)
                except Exception as e:   # the real converter may not accept this tiny config: reported, not hidden
                    check(f"[{name}] the real converter wrote model.bin", False, f"{type(e).__name__}: {e}")
                    continue
                gen = E.ctranslate2.Generator(str(ct2), device="cpu", compute_type="float32")
                vocab_ct2 = json.loads((ct2 / "vocabulary.json").read_text())
                ref = gen.score_batch([[vocab_ct2[i] for i in ids]])[0].log_probs
                out = Path(tmp) / f"conv-{fam}-{q}"
                try:
                    info = C.convert(ct2, out, hf)
                except C.CT2Error as e:
                    check(f"[{name}] convert_ct2_to_hf.py converts the CT2 directory", False, str(e))
                    continue
                loaded = AutoModelForCausalLM.from_pretrained(str(out), torch_dtype=torch.float32).eval()
                got = hf_logprobs(loaded, ids)
                e1 = max(abs(a - b) for a, b in zip(got, ref))
                check(f"[{name}] converted HF directory (transformers) == real CTranslate2 engine (max err {e1:.1e}, arch {info['arch']})", e1 < 1e-4)
                if q == "float32":
                    e2 = max(abs(a - b) for a, b in zip(ref, hf_ref))
                    check(f"[{name}] real CTranslate2 == original torch model (max err {e2:.1e})", e2 < 1e-4)
        # controls
        for q in ("int8", "int16", "float16"):
            d = max(abs(a - b) for a, b in zip(results[("llama", q)], results[("llama", "float32")]))
            check(f"control: {q} differs from float32 under the real engine ({d:.2e})", d > 1e-5)
        d = max(abs(a - b) for a, b in zip(results[("llama", "float32")], results[("llama-bias", "float32")]))
        check(f"control: the bias model differs from the no-bias model ({d:.2e})", d > 1e-3)


if __name__ == "__main__":
    main()
    print()
    print("FAILED: " + ", ".join(T.failures) if T.failures else "all checks passed")
    sys.exit(1 if T.failures else 0)
