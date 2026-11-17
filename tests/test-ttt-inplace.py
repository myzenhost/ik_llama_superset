#!/usr/bin/env python3
"""In-Place TTT parity test (row S69-FORK-INPLACE-TTT-2026-10-26-01).

Oracle: the OFFICIAL In-Place TTT inference code (https://github.com/ByteDance-Seed/In-Place-TTT, Apache-2.0, commit
be2324829b0e91c8fd10a74d4b43714fde6676e1, inference_model/hf_qwen3/modeling_qwen3.py), vendored UNMODIFIED in tests/ttt-inplace-oracle/
and sha256-checked before use. No code of it is copied into the fork; it is imported and run as the reference (fp32, CPU, batch 1).

What it does, nothing downloaded:
  1. builds a random-weight Qwen3 with that code: hidden 64, ffn 128, 4 layers, 4 heads / 2 KV heads (head_dim 16), vocab 512,
     ttt_mode on for layers [1, 3], ttt_chunk 8, ttt_lr 3, ttt_proj on, ttt_target hidden_states; ttt_conv (which the official
     _init_weights zero-inits, modeling_qwen3.py:406-409) gets random weights so the update is not vacuous. A ttt-off twin has the same
     weights minus ttt_conv/ttt_proj. Both are converted with this repo's convert_hf_to_gguf.py --outtype f32.
  2. oracle logits: sequence A = 37 random ids + 12 greedy tokens (prefill then 12 single-token steps through TTTDynamicCache, checked
     against model.generate), sequence B = 29 other random ids.
  3. the fork (tests/test-ttt-inplace.cpp) decodes the same ids several ways; every comparison is max |logit_fork - logit_oracle|.
Cases (TOL = 1e-3, F32, CPU):
  POSITIVE CONTROLS (must hold at any HEAD)
    oracle_consistency    oracle prefill+greedy == token-by-token == one whole forward (<= 1e-4); generate() picks the same 12 tokens
    oracle_ttt_effect     oracle ttt vs ttt-off: rows 0..7 equal (<= 1e-5, fp32 noise of the chunked einsum: chunk 0 runs with the unadapted down_proj), EVERY row >= 8
                          differs by > 10*TOL (so a fork that ignores TTT cannot pass the TTT cases)
    off_parity            fork(ttt-off GGUF) vs oracle ttt-off, token-by-token (converter + qwen3 graph parity)
  NEGATIVE CONTROL (must miss by > 10*TOL at rows >= 8)
    neg_off_vs_ttt        fork(ttt-off GGUF) vs the TTT oracle
  TTT CASES (FAIL before the port, PASS after)
    convert_ttt           the converter accepts the TTT checkpoint
    ttt_load              GGUF ttt.* keys + ffn_ttt_conv (5, d) / ffn_ttt_proj (P^T) equal the oracle's weights (gguf-py read-back), only on
                          layers 1 and 3; the fork loads it and its loader logs the parsed TTT hparams
    ttt_gen               prefill 37 in one batch + 12 greedy steps; the fork's greedy tokens must equal the oracle's
    ttt_tbt               every sequence token by token
    ttt_ub5               one llama_decode per sequence with n_ubatch 5 (ubatches straddle the chunk grid)
    ttt_effect_fork       fork ttt vs fork ttt-off: rows 0..7 within TOL, every row >= 8 apart by > 10*TOL
    ttt_rewind            A to 12 token by token, seq_rm from 5 and of [2,4) refused, from 10 allowed, seq_cp 0 -> 1, 1 continues from 10
    ttt_multi             A and B interleaved in the same batches (3 ids of A + 4 of B per llama_decode) vs the oracle
    ttt_multi_bitwise     isolation: replacing the OTHER sequence's tokens in the interleaved run leaves each sequence's logits BITWISE
                          equal (control: the replaced one changes); the gap to each sequence alone is reported for the ttt and
                          ttt-off models (a KV-layout effect of attention, present without TTT)
    ttt_state             A: decode 20 ids, llama_state_seq_get_data, free the context, set_data into a new one, decode the rest
    ttt_spec_draft        greedy speculative decoding (K 4, the ttt-off twin as draft model, llama_ttt_draft_cap on): tokens == oracle
                          greedy (== draft-off ttt_gen) and logits within TOL
    ttt_spec_adv          the same with adversarial drafts (nearly every round discards its drafts via seq_rm)
    ttt_spec_self         the same with the TTT GGUF as its own (exact) drafter: drafts must be KEPT (kept >= 1)
    ttt_spec_mix          exact drafter with draft (round % (nd + 1)) corrupted: every keep length 0..nd occurs
  NEGATIVE CONTROL FOR THE CAP
    ttt_spec_nocap_refused  adversarial drafts WITHOUT the cap: a draft completes a chunk, and seq_rm of the discarded suffix is refused
usage: test-ttt-inplace.py --bin PATH/test-ttt-inplace[.exe] [--work DIR] [--cases a,b] [--json OUT.json]
Exit 0 iff every selected case met its expectation; 1 otherwise; 77 (CTest SKIPPED) when a Python dependency is missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORACLE_DIR = os.path.join(REPO, "tests", "ttt-inplace-oracle")
ORACLE_SHA256 = {
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "hf_qwen3/__init__.py": "a38d76873e560e9c034258408e1c67fdfc08f2be846da1937cea1434f6e10e80",
    "hf_qwen3/configuration_qwen3.py": "39ef69d7c79114ae9bd2710320c4260796fc5dbed9f725e18b5a0f5771114ec2",
    "hf_qwen3/modeling_qwen3.py": "98ec94bcf1780635d7817b993a0bf4fa1f1a7e3a35415dce8a46c345a52a2858",
}
TOL = 1e-3
VOCAB = 512
CHUNK = 8
N_PROMPT = 37
N_GEN = 12
N_B = 29
SEED = 1026
STATE_K = 20
MULTI_STEPS = (3, 4)
SPEC_K = 4  # drafts per round before the cap
REWIND = (12, 5, 10)  # decode to 12, try rm from 5 (refused: crosses the chunk-0 update) and from 10 (allowed: inside [8, 16))

try:
    import numpy as np
    import torch
    import transformers  # noqa: F401
    import einops  # noqa: F401
    import opt_einsum  # noqa: F401
    import sentencepiece as spm
except ImportError as e:  # FAILURES-SURFACED: printed and mapped to exit 77 (CTest SKIPPED)
    print(f"test-ttt-inplace: SKIPPED ({e})")
    sys.exit(77)


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    return p.returncode, p.stdout


def check_oracle() -> None:
    for rel, want in ORACLE_SHA256.items():
        with open(os.path.join(ORACLE_DIR, rel), "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got != want:
            raise SystemExit(f"oracle file {rel} sha256 {got} != pinned {want}: the oracle must run UNMODIFIED")
    sys.path.insert(0, ORACLE_DIR)


# ---------------------------------------------------------------- model + oracle


def make_tokenizer(out_dir: str) -> None:
    rng = np.random.default_rng(SEED)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    words = ["".join(rng.choice(letters, size=int(rng.integers(2, 7)))) for _ in range(300)]
    lines = [" ".join(rng.choice(words, size=12)) for _ in range(2000)]
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(lines), model_prefix=os.path.join(out_dir, "tokenizer"), vocab_size=400, model_type="unigram",
        hard_vocab_limit=False, unk_id=0, bos_id=1, eos_id=2, pad_id=-1, minloglevel=2,
    )
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"tokenizer_class": "LlamaTokenizer", "add_bos_token": False, "add_eos_token": False,
                   "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}, f)


def strip_ttt_config(path: str) -> None:
    # REWRITE-SAFE: config.json in a scratch model dir this script wrote moments earlier (save_pretrained); nothing else owns it
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    for k in [k for k in c if k.startswith("ttt_")]:
        del c[k]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(c, f, indent=2)


def make_models(work: str, conv_std: float, proj_std: float, init_std: float):
    import hf_qwen3  # noqa: F401  (registers the official classes with the Auto* factories)
    from hf_qwen3.configuration_qwen3 import Qwen3Config
    from hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM

    common = dict(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=512, rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=False,
        initializer_range=init_std, bos_token_id=1, eos_token_id=2, attn_implementation="eager",
    )
    cfg = Qwen3Config(**common, ttt_mode=True, ttt_layers=[1, 3], ttt_chunk=CHUNK, ttt_lr=3.0, ttt_proj=True, ttt_target="hidden_states")
    torch.manual_seed(SEED)
    model = Qwen3ForCausalLM(cfg).float().eval()
    g = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n:  # non-unit norms, so a norm slip in the converter or graph is visible
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            elif "ttt_conv" in n:  # the official init zeroes it, which would make every update zero
                p.copy_(conv_std * torch.randn(p.shape, generator=g))
            elif "ttt_proj" in n:
                p.copy_(proj_std * torch.randn(p.shape, generator=g))
    ttt_dir = os.path.join(work, "hf_ttt")
    off_dir = os.path.join(work, "hf_off")
    model.save_pretrained(ttt_dir, safe_serialization=True)
    make_tokenizer(ttt_dir)

    cfg_off = Qwen3Config(**common, ttt_mode=False)
    model_off = Qwen3ForCausalLM(cfg_off).float().eval()
    sd = {k: v for k, v in model.state_dict().items() if ".ttt_" not in k}
    res = model_off.load_state_dict(sd, strict=True)
    assert not res.missing_keys and not res.unexpected_keys, res
    model_off.save_pretrained(off_dir, safe_serialization=True)
    make_tokenizer(off_dir)
    strip_ttt_config(os.path.join(off_dir, "config.json"))
    n_ttt = sum(1 for k in model.state_dict() if ".ttt_" in k)
    log(f"models: ttt dir {ttt_dir} ({n_ttt} ttt tensors), off dir {off_dir}")
    return model, model_off


@torch.no_grad()
def oracle_gen(model, prompt: list[int], n_gen: int) -> tuple[list[int], np.ndarray]:
    ids = list(prompt)
    out = model(input_ids=torch.tensor([ids]), use_cache=True)
    rows = [out.logits[0]]
    cache = out.past_key_values
    for _ in range(n_gen):
        nxt = int(torch.argmax(rows[-1][-1]))
        ids.append(nxt)
        out = model(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        rows.append(out.logits[0])
    return ids, torch.cat(rows, 0).float().numpy()


@torch.no_grad()
def oracle_tbt(model, ids: list[int]) -> np.ndarray:
    cache = None
    rows = []
    for t in ids:
        out = model(input_ids=torch.tensor([[t]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        rows.append(out.logits[0])
    return torch.cat(rows, 0).float().numpy()


@torch.no_grad()
def oracle_whole(model, ids: list[int]) -> np.ndarray:
    return model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float().numpy()


# ---------------------------------------------------------------- fork side


def read_out(path: str) -> list[tuple[list[int], np.ndarray]]:
    with open(path, "rb") as f:
        data = f.read()
    ns, nv = struct.unpack_from("<ii", data, 0)
    off = 8
    res = []
    for _ in range(ns):
        (nt,) = struct.unpack_from("<i", data, off)
        off += 4
        tok = list(struct.unpack_from(f"<{nt}i", data, off))
        off += 4 * nt
        lg = np.frombuffer(data, dtype=np.float32, count=nt * nv, offset=off).reshape(nt, nv).copy()
        off += 4 * nt * nv
        res.append((tok, lg))
    return res


class Fork:
    def __init__(self, binary: str, work: str):
        self.bin = binary
        self.work = work
        self.n = 0

    def __call__(self, gguf: str, seqs: list[list[int]], mode: list[str]):
        self.n += 1
        tok = os.path.join(self.work, f"tok_{self.n}.txt")
        out = os.path.join(self.work, f"out_{self.n}.bin")
        with open(tok, "w", encoding="utf-8") as f:
            for s in seqs:
                f.write(" ".join(str(t) for t in s) + "\n")
        rc, text = run([self.bin, gguf, tok, out] + mode)
        self.last_text = text
        with open(os.path.join(self.work, f"out_{self.n}.log"), "w", encoding="utf-8") as f:
            f.write(" ".join(mode) + "\n" + text)
        if rc != 0:
            tail = " | ".join(text.splitlines()[-4:])
            return None, f"fork rc={rc} mode={' '.join(mode)}: {tail}"
        return read_out(out), None


def check_ttt_gguf(path: str, model) -> list[str]:
    """Problems with the TTT GGUF's keys and tensors, read back with this repo's gguf-py and compared with the oracle's weights."""
    sys.path.insert(0, os.path.join(REPO, "gguf-py"))
    import gguf  # noqa: E402

    r = gguf.GGUFReader(path)
    probs = []
    want = {"qwen3.ttt.layers": [1, 3], "qwen3.ttt.chunk": CHUNK, "qwen3.ttt.lr": 3.0, "qwen3.ttt.target": 0,
            "qwen3.ttt.conv_padding": 0, "qwen3.ttt.clip_tau": 0.0}
    for k, v in want.items():
        f = r.fields.get(k)
        got = None if f is None else f.contents()
        if got is None or (list(got) if isinstance(v, list) else got) != v:
            probs.append(f"{k}={got!r} want {v!r}")
    tensors = {t.name: t for t in r.tensors}
    sd = model.state_dict()
    for il in range(4):
        for kind in ("conv", "proj"):
            name = f"blk.{il}.ffn_ttt_{kind}.weight"
            if il not in (1, 3):
                if name in tensors:
                    probs.append(f"{name} present on a non-TTT layer")
                continue
            t = tensors.get(name)
            if t is None:
                probs.append(f"{name} missing")
                continue
            ref = sd[f"model.layers.{il}.mlp.ttt_{kind}.weight"]
            ref = (ref.squeeze(1).T if kind == "conv" else ref.T).contiguous().numpy()  # conv (5, d); proj stored as P^T
            got = np.asarray(t.data)
            if t.tensor_type != gguf.GGMLQuantizationType.F32 or got.shape != ref.shape or not np.array_equal(got, ref):
                probs.append(f"{name}: type {t.tensor_type.name} shape {got.shape} vs ref {ref.shape} equal "
                             f"{got.shape == ref.shape and np.array_equal(got, ref)}")
    return probs


def rowmax(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(np.float64) - b.astype(np.float64)).max(axis=1)


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--cases", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--conv-std", type=float, default=0.05)
    ap.add_argument("--proj-std", type=float, default=0.1)
    ap.add_argument("--init-std", type=float, default=0.1)
    ap.add_argument("--oracle-only", action="store_true", help="build the models + oracle and run the oracle controls only")
    a = ap.parse_args()
    check_oracle()
    work = a.work or tempfile.mkdtemp(prefix="ttt_inplace_")
    os.makedirs(work, exist_ok=True)
    log(f"work dir: {work}")
    t0 = time.time()
    torch.set_num_threads(4)

    model, model_off = make_models(work, a.conv_std, a.proj_std, a.init_std)
    rng = np.random.default_rng(SEED + 7)
    prompt = [int(x) for x in rng.integers(3, VOCAB, size=N_PROMPT)]
    seq_b = [int(x) for x in rng.integers(3, VOCAB, size=N_B)]

    ids_a, o_gen = oracle_gen(model, prompt, N_GEN)
    o_tbt = oracle_tbt(model, ids_a)
    o_whole = oracle_whole(model, ids_a)
    gen_ids = model.generate(input_ids=torch.tensor([prompt]), max_new_tokens=N_GEN, do_sample=False, pad_token_id=0).tolist()[0]
    o_b = oracle_whole(model, seq_b)
    o_b_tbt = oracle_tbt(model, seq_b)
    o_off = oracle_tbt(model_off, ids_a)
    o_off_b = oracle_tbt(model_off, seq_b)
    top2 = np.sort(o_gen[N_PROMPT - 1:-1], axis=1)[:, -2:]
    log(f"oracle: A = {N_PROMPT} prompt + {N_GEN} greedy ids {ids_a[N_PROMPT:]}; min top-2 gap over the greedy steps "
        f"{float((top2[:, 1] - top2[:, 0]).min()):.4g}; |logit| max {float(np.abs(o_gen).max()):.4g}")

    cases: dict[str, dict] = {}

    def record(name: str, ok: bool, **kw) -> None:
        cases[name] = {"ok": bool(ok), **kw}
        log(f"  {'PASS' if ok else 'FAIL'}  {name}  " + " ".join(f"{k}={v}" for k, v in kw.items()))

    want = set(a.cases.split(",")) if a.cases else None

    def sel(name: str) -> bool:
        return want is None or name in want

    # ---- positive controls on the oracle itself
    d1 = float(rowmax(o_gen, o_tbt).max())
    d2 = float(rowmax(o_gen, o_whole).max())
    d3 = float(rowmax(o_b, o_b_tbt).max())
    record("oracle_consistency", max(d1, d2, d3) <= 1e-4 and gen_ids == ids_a,
           gen_vs_tbt=f"{d1:.3g}", gen_vs_whole=f"{d2:.3g}", b_whole_vs_tbt=f"{d3:.3g}", generate_tokens_equal=gen_ids == ids_a)
    eff = rowmax(o_gen, o_off)
    effb = rowmax(o_b, o_off_b)
    lo = float(max(eff[:CHUNK].max(), effb[:CHUNK].max()))
    hi_min = float(min(eff[CHUNK:].min(), effb[CHUNK:].min()))
    record("oracle_ttt_effect", lo <= 1e-5 and hi_min > 10 * TOL, rows_lt8_max=f"{lo:.3g}",
           rows_ge8_min=f"{hi_min:.4g}", rows_ge8_max=f"{float(max(eff[CHUNK:].max(), effb[CHUNK:].max())):.4g}")
    if a.oracle_only:
        n_fail = sum(1 for v in cases.values() if not v["ok"])
        log(f"RESULT (oracle only): {len(cases) - n_fail}/{len(cases)}")
        return 0 if n_fail == 0 else 1

    # ---- conversion
    py = sys.executable
    conv = os.path.join(REPO, "convert_hf_to_gguf.py")
    g_off = os.path.join(work, "off.gguf")
    g_ttt = os.path.join(work, "ttt.gguf")
    rc, text = run([py, conv, os.path.join(work, "hf_off"), "--outtype", "f32", "--outfile", g_off])
    if rc != 0:
        log(text[-3000:])
        raise SystemExit("converting the ttt-off twin failed: that is plain Qwen3 and must always convert")
    rc, text = run([py, conv, os.path.join(work, "hf_ttt"), "--outtype", "f32", "--outfile", g_ttt])
    with open(os.path.join(work, "convert_ttt.log"), "w", encoding="utf-8") as f:
        f.write(text)
    conv_ok = rc == 0
    record("convert_ttt", conv_ok, detail=("ok" if conv_ok else repr((text.strip().splitlines() or ["?"])[-1][:300])))

    fork = Fork(os.path.abspath(a.bin), work)

    # ---- fork controls on the ttt-off GGUF
    r_off, err = fork(g_off, [ids_a, seq_b], ["tbt"])
    if err:
        record("off_parity", False, error=err)
    else:
        d = max(float(rowmax(r_off[0][1], o_off).max()), float(rowmax(r_off[1][1], o_off_b).max()))
        record("off_parity", d <= TOL, max_dlogit=f"{d:.3g}")
        n = float(min(rowmax(r_off[0][1], o_gen)[CHUNK:].min(), rowmax(r_off[1][1], o_b)[CHUNK:].min()))
        record("neg_off_vs_ttt", n > 10 * TOL, rows_ge8_min_dlogit=f"{n:.4g}", note="fork-without-TTT-vs-TTT-oracle-must-MISS")

    # ---- TTT cases
    def ttt_case(name: str, seqs: list[list[int]], mode: list[str], refs: list[np.ndarray], tokens_ref: list[int] | None = None):
        if not sel(name):
            return None
        if not conv_ok:
            record(name, False, error="no TTT GGUF (convert_ttt failed)")
            return None
        r, e = fork(g_ttt, seqs, mode)
        if e:
            record(name, False, error=e)
            return None
        worst = max(float(rowmax(r[i][1], refs[i]).max()) for i in range(len(refs)))
        kw = {"max_dlogit": f"{worst:.3g}"}
        ok = worst <= TOL
        if not ok:
            rows = rowmax(r[0][1], refs[0])
            kw["first_bad_row_seq0"] = int(np.argmax(rows > TOL)) if (rows > TOL).any() else -1
        if tokens_ref is not None:
            kw["tokens_equal"] = r[0][0] == tokens_ref
            ok = ok and r[0][0] == tokens_ref
        record(name, ok, **kw)
        return r

    if sel("ttt_load"):
        if not conv_ok:
            record("ttt_load", False, error="no TTT GGUF (convert_ttt failed)")
        else:
            probs = check_ttt_gguf(g_ttt, model)
            _, e = fork(g_ttt, [ids_a], ["load"])
            want_line = "in-place TTT on 2 layer(s): chunk 8, lr 3, target hidden_states, conv symmetric, clip_tau 0"
            hp_ok = want_line in (fork.last_text or "")
            record("ttt_load", not probs and e is None and hp_ok, gguf_problems=probs[:4], fork_load=e or "ok", hparams_line=hp_ok)

    r_gen = ttt_case("ttt_gen", [ids_a], ["gen", str(N_PROMPT), str(N_GEN)], [o_gen], tokens_ref=ids_a)
    r_tbt = ttt_case("ttt_tbt", [ids_a, seq_b], ["tbt"], [o_gen, o_b])
    ttt_case("ttt_ub5", [ids_a, seq_b], ["ub", "5"], [o_gen, o_b])
    if sel("ttt_effect_fork"):
        src = r_tbt if r_tbt is not None else r_gen
        if src is None or r_off is None:
            record("ttt_effect_fork", False, error="no fork TTT run")
        else:
            e = rowmax(src[0][1], r_off[0][1])
            record("ttt_effect_fork", float(e[:CHUNK].max()) <= TOL and float(e[CHUNK:].min()) > 10 * TOL,
                   rows_lt8_max=f"{float(e[:CHUNK].max()):.3g}", rows_ge8_min=f"{float(e[CHUNK:].min()):.4g}")
    if sel("ttt_rewind"):
        # rewind 12 -> 10 inside chunk [8, 16) is exact; 12 -> 5 (across the chunk-0 update) and an interior drop are refused;
        # sequence 1 continues from a seq_cp of the rewound sequence 0, so a missing state copy also fails the parity
        r = ttt_case("ttt_rewind", [ids_a], ["rewind", str(REWIND[0]), str(REWIND[1]), str(REWIND[2])], [o_gen])
        want_line = f"rewind: rm({REWIND[1]})=0 rm(2,4)=0 rm({REWIND[2]})=1 can_shift=0"
        line_ok = want_line in (fork.last_text or "")
        if r is not None or "ttt_rewind" in cases:
            cases["ttt_rewind"]["ok"] = bool(cases["ttt_rewind"]["ok"] and line_ok)
            cases["ttt_rewind"]["rm_line"] = line_ok
            log(f"  {'PASS' if cases['ttt_rewind']['ok'] else 'FAIL'}  ttt_rewind (with rm line {want_line!r}: {line_ok})")
    r_multi = ttt_case("ttt_multi", [ids_a, seq_b], ["multi", str(MULTI_STEPS[0]), str(MULTI_STEPS[1])], [o_gen, o_b])
    if sel("ttt_multi_bitwise"):
        if r_multi is None:
            record("ttt_multi_bitwise", False, error="ttt_multi did not run")
        else:
            alone = []
            for i, st in enumerate(MULTI_STEPS):
                r, e2 = fork(g_ttt, [[ids_a, seq_b][i]], ["step", str(st)])
                alone.append(None if e2 else r[0][1])
            # Isolation, bitwise: the same interleaved run with the OTHER sequence's tokens replaced (same lengths, so the
            # same ubatch shapes and KV cell layout) must leave each sequence's logits bit-identical -- any leak of one
            # sequence's fast weights or tail into the other changes them. Against "alone" the KV layout differs (the
            # other sequence's cells sit between this one's), which moves attention sums in the last bits for the
            # ttt-off model too; that gap is measured on both models and reported, the bitwise verdict is the swap.
            rng2 = np.random.default_rng(SEED + 99)
            a_alt = [int(x) for x in rng2.integers(3, VOCAB, size=len(ids_a))]
            b_alt = [int(x) for x in rng2.integers(3, VOCAB, size=len(seq_b))]
            ms = ["multi", str(MULTI_STEPS[0]), str(MULTI_STEPS[1])]
            r_ab2, e_ab2 = fork(g_ttt, [ids_a, b_alt], ms)
            r_a2b, e_a2b = fork(g_ttt, [a_alt, seq_b], ms)
            off_m, e_om = fork(g_off, [ids_a, seq_b], ms)
            off_alone = []
            for i, st in enumerate(MULTI_STEPS):
                r, e2 = fork(g_off, [[ids_a, seq_b][i]], ["step", str(st)])
                off_alone.append(None if e2 else r[0][1])
            if any(x is None for x in alone) or e_ab2 or e_a2b or e_om or any(x is None for x in off_alone):
                record("ttt_multi_bitwise", False, error="a run failed", errs=[e_ab2, e_a2b, e_om])
            else:
                eq = [bool(np.array_equal(r_multi[0][1], r_ab2[0][1])), bool(np.array_equal(r_multi[1][1], r_a2b[1][1]))]
                neg = bool(not np.array_equal(r_multi[1][1], r_ab2[1][1]))   # control: the swapped sequence DID change
                dd = [f"{float(rowmax(r_multi[i][1], alone[i]).max()):.3g}" for i in range(2)]
                do = [f"{float(rowmax(off_m[i][1], off_alone[i]).max()):.3g}" for i in range(2)]
                ae = [bool(np.array_equal(r_multi[i][1], alone[i])) for i in range(2)]
                record("ttt_multi_bitwise", all(eq) and neg, swap_bitwise=eq, swapped_seq_changed=neg,
                       ttt_vs_alone_max_abs=dd, ttt_vs_alone_bitwise=ae, off_model_vs_alone_max_abs=do)
    ttt_case("ttt_state", [ids_a], ["state", str(STATE_K)], [o_gen])

    # ---- speculative decoding with the draft cap (unit 6): greedy text with drafts == greedy text without (== the oracle)
    spec_base = ["spec", str(N_PROMPT), str(N_GEN), str(SPEC_K)]
    # min_kept: an exact drafter (the TTT GGUF itself) must get drafts KEPT, or the keep path went unexercised
    for name, extra, min_kept in (("ttt_spec_draft", ["1", g_off], 0), ("ttt_spec_adv", ["1", "adv"], 0),
                                  ("ttt_spec_self", ["1", g_ttt], 1), ("ttt_spec_mix", ["1", g_ttt, "mix"], 1)):
        r = ttt_case(name, [ids_a], spec_base + extra, [o_gen], tokens_ref=ids_a)
        if r is not None:
            line = [x for x in (fork.last_text or "").splitlines() if x.startswith("spec:")]
            m = re.search(r"(\d+) rounds, (\d+) drafted, (\d+) kept", line[-1] if line else "")
            kept = int(m.group(3)) if m else -1
            cases[name]["stats"] = line[-1] if line else None
            if kept < min_kept:
                cases[name]["ok"] = False
                log(f"  FAIL  {name}  kept {kept} < {min_kept}: the keep path was not exercised")
    if sel("ttt_spec_nocap_refused"):
        # NEGATIVE CONTROL: without the cap an adversarial draft completes a chunk inside the verify batch; the update
        # cannot be undone, so seq_rm of the non-matching suffix must be refused (exit 6) rather than corrupt the state
        if not conv_ok:
            record("ttt_spec_nocap_refused", False, error="no TTT GGUF")
        else:
            _, e = fork(g_ttt, [ids_a], spec_base + ["0", "adv"])
            refused = e is not None and "seq_rm refused" in e
            record("ttt_spec_nocap_refused", refused, detail=(e or "ran to completion")[-200:])

    always = ("oracle_consistency", "oracle_ttt_effect", "convert_ttt", "off_parity", "neg_off_vs_ttt")
    selected = {k: v for k, v in cases.items() if sel(k) or k in always}
    n_fail = sum(1 for v in selected.values() if not v["ok"])
    log(f"RESULT: {len(selected) - n_fail}/{len(selected)} cases met their expectation ({time.time() - t0:.1f} s); work dir {work}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"tol": TOL, "chunk": CHUNK, "ids_a": ids_a, "seq_b": seq_b, "cases": selected,
                       "oracle_commit": "be2324829b0e91c8fd10a74d4b43714fde6676e1"}, f, indent=1)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
