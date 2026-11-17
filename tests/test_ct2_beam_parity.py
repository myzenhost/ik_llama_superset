#!/usr/bin/env python3
"""llama-server beam search (beam_mode "ct2") against the REAL CTranslate2 engine's beam search on the SAME model.

   python tests/test_ct2_beam_parity.py <tree-root> <llama-server>

One random-weight Llama-family model is written as a CTranslate2 model.bin (the independent writer of tests/test_ct2_to_hf.py), run by the pip
`ctranslate2` Generator (generate_batch with beam_size / patience / length_penalty / min_length / max_length / num_hypotheses, return_scores),
converted by convert_hf_to_gguf.py (--ct2-tokenizer-dir) and run by llama-server with use_beam_search + beam_mode "ct2". For every case the
hypotheses must have the SAME token ids in the same order and scores within 1e-3 (float32 on both sides; the server runs beams batched, so
the last digits differ). The model's end-of-sequence row is amplified so hypotheses really end at different steps (control (a)).
This replaces "equals a reference written from the algorithm" (tests/unit/test_beam_ct2.py) by "equals the engine the algorithm comes from".
"""
import json
import math
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
server_bin = sys.argv[2]
sys.path.insert(0, str(root / "tests"))
sys.argv = [sys.argv[0], str(root), server_bin]

import numpy as np  # noqa: E402

import test_ct2_gguf_e2e as E  # noqa: E402  (server launcher, tokenizer maker, independent writer)

T = E.T
check = T.check

EOS = 2
CASES = [
    # (beam width, length_penalty, patience, min_length, num_hypotheses, max_length)
    (1, 1.0, 1.0, 0, 1, 20),
    (3, 0.0, 1.0, 0, 3, 20),
    (3, 1.0, 1.0, 0, 3, 20),
    (4, 1.0, 2.0, 0, 4, 20),
    (4, 0.6, 1.5, 0, 4, 20),
    (4, 1.0, 1.0, 0, 2, 20),
    (3, 1.0, 1.0, 12, 3, 20),
    (5, 2.0, 1.0, 0, 5, 20),
]


def build(tmp, tokdir, vocab, eos_boost, seed, name):
    T.RNG = np.random.default_rng(seed)
    wr, orig, cfg, _v, dims = T.make_model("float32", n_heads=4, n_kv=2, hidden=32, ffn=64, vocab=len(vocab), layers=2, rotary_interleave=0, w_std=0.5)
    # amplify the EOS row of the output projection so end-of-sequence is reachable (a random model almost never ends)
    for i, (name, dtype, shape, raw) in enumerate(wr.vars):
        if name == "decoder/projection/weight":
            w = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
            w[EOS] = w[EOS] * eos_boost + w[EOS].mean() * 0
            wr.vars[i] = (name, dtype, shape, w.astype("<f4").tobytes())
    cfg.update({"bos_token": vocab[1], "eos_token": vocab[2], "unk_token": vocab[0], "layer_norm_epsilon": 1e-5})
    ct2 = Path(tmp) / ("ct2-" + name)
    wr.write(ct2, config=cfg, vocab=vocab)
    return ct2


def run_model(tmp, tokdir, vocab, seed, boost, name):
    ct2dir = build(tmp, tokdir, vocab, eos_boost=boost, seed=seed, name=name)
    gguf = Path(tmp) / (name + ".gguf")
    r = E.subprocess.run([sys.executable, str(root / "convert_hf_to_gguf.py"), str(ct2dir), "--ct2-tokenizer-dir", str(tokdir), "--outfile", str(gguf),
                          "--outtype", "f32"], capture_output=True, text=True, cwd=str(root))
    check(f"[{name}] convert_hf_to_gguf.py converts the CT2 directory", r.returncode == 0 and gguf.exists(), (r.stderr or r.stdout)[-600:])
    gen = E.ctranslate2.Generator(str(ct2dir), device="cpu", compute_type="float32")
    prompt = [1, 20, 31, 7]
    srv = E.Server(gguf, slots=8)
    n_ended = n_hyp = 0
    lengths = set()
    try:
        for (W, lp, pat, minlen, nh, maxlen) in CASES:
            tag = f"{name} width={W} lp={lp} patience={pat} min_length={minlen} hyps={nh}"
            res = gen.generate_batch([[vocab[i] for i in prompt]], beam_size=W, patience=pat, num_hypotheses=nh, length_penalty=lp, max_length=maxlen,
                                     min_length=minlen, include_prompt_in_result=False, return_scores=True)[0]
            ref = [(list(ids), sc) for ids, sc in zip(res.sequences_ids, res.scores)]
            req = {"prompt": prompt, "n_predict": maxlen, "temperature": 0.0, "use_beam_search": True, "beam_mode": "ct2", "n": W, "beam_num_hypotheses": nh,
                   "length_penalty": lp, "beam_patience": pat, "min_tokens": minlen, "cache_prompt": False}
            if W == 1:
                req.pop("beam_num_hypotheses")
            out = srv.post("/completion", req)
            got = [(b["tokens"], b["score"]) for b in out["beams"]]
            same_ids = [g[0] for g in got] == [x[0] for x in ref]
            ok_scores = len(got) == len(ref) and all(math.isclose(g[1], x[1], rel_tol=1e-3, abs_tol=1e-3) for g, x in zip(got, ref))
            check(f"[{tag}] same hypotheses, same order", same_ids, f"fork={[g[0] for g in got]} ct2={[x[0] for x in ref]}")
            check(f"[{tag}] same scores (1e-3)", ok_scores, f"fork={[round(g[1], 4) for g in got]} ct2={[round(x[1], 4) for x in ref]}")
            n_hyp += len(ref)
            n_ended += sum(1 for ids, _ in ref if len(ids) < maxlen)
            lengths |= {len(ids) for ids, _ in ref}
            print("     ct2:", [(len(i), round(s, 3)) for i, s in ref])
    finally:
        srv.close()
    return n_ended, n_hyp, lengths


def main():
  with tempfile.TemporaryDirectory() as tmp:
    tokdir = Path(tmp) / "tok"
    vocab = E.make_tokenizer(tokdir)
    for seed, boost, name in ((7, 6.0, "peaked-eos"), (17, 2.5, "mid-eos"), (23, 2.5, "spread-eos"), (23, 4.5, "early-eos")):
        n_ended, n_hyp, lengths = run_model(tmp, tokdir, vocab, seed, boost, name)
        print(f"     {name}: {n_ended}/{n_hyp} hypotheses ended on EOS before max_length, {len(lengths)} distinct lengths")
        check(f"control (a) [{name}]: hypotheses have several lengths (EOS reachable, search is not trivial)", len(lengths) >= 3, str(sorted(lengths)))



if __name__ == "__main__":
    main()
    print()
    print("FAILED: " + ", ".join(T.failures) if T.failures else "all checks passed")
    sys.exit(1 if T.failures else 0)
