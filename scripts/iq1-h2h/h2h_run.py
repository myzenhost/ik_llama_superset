#!/usr/bin/env python3
# REWRITE-SAFE: results.jsonl is only ever opened in append mode; every other write is a fresh per-candidate path.
"""Head-to-head of sub-2-bit quant types on a real model: same weights, same imatrix, same text, KLD + PPL.

HOW A CANDIDATE IS MEASURED (and why it is "fake-quant"):
  1. llama-quantize --imatrix IM --allow-requantize --pure --custom-q <rules> SRC cand.gguf <TYPE>
         every block matrix gets TYPE (regime A) or only the FFN matrices do while attention stays IQ4_XS (regime B);
         token_embd is Q6_K in every candidate, so the ONLY thing that varies between candidates is TYPE.
  2. llama-quantize --allow-requantize cand.gguf cand_f32.gguf F32
         dequantises every tensor through the type's own to_float. The F32 file holds EXACTLY the weights the
         quantised type represents, so step 3 measures the quantisation error of the TYPE, not the speed or the
         activation-quantisation noise of whichever vec_dot kernel the type happens to have. (A scalar kernel for
         a new type would otherwise make the evaluation a day long.) One real-kernel check closes the loop below.
  3. llama-perplexity -m cand_f32.gguf -f wiki.test.raw --kl-divergence --kl-divergence-base base.kld
         against the logits of SRC itself (the reference), so KLD is paired and far less noisy than PPL.

USAGE: h2h_run.py <stage> [more stages]   stages: imatrix | base | run | real | all     (run accepts REGIME:type or type filters)
       env H2H_WORK, H2H_BIN, H2H_SRC, H2H_CHUNKS_KLD, H2H_CHUNKS_IM override the defaults below.
Every artefact lands in H2H_WORK; results are appended to results.jsonl (one JSON object per candidate) and a finished
candidate is skipped on re-run, so a killed run resumes where it stopped.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

W = os.environ.get("H2H_WORK", "build/work/h2h")
BIN = os.environ.get("H2H_BIN", "build/bin")
SRC = os.environ.get("H2H_SRC", "models/gguf/llama3.2-1b/Llama-3.2-1B-Instruct-Q8_0.gguf")
TRAIN = os.environ.get("H2H_TRAIN", "tmp/research/experiments/altnumerics/wikitext-2-raw/wiki.train.raw")
TEST = os.environ.get("H2H_TEST", "tmp/research/experiments/altnumerics/wikitext-2-raw/wiki.test.raw")
CTX = 512
THREADS = int(os.environ.get("H2H_THREADS", "8"))
CH_IM = int(os.environ.get("H2H_CHUNKS_IM", "60"))
CH_KLD = int(os.environ.get("H2H_CHUNKS_KLD", "32"))
NOWIN = 0x08000000   # CREATE_NO_WINDOW (rule 1.45)

# (name, bits-per-weight of the type itself). Order = the order they are run (most informative first).
TYPES = [
    ("iq1_xxxs", 1.1875), ("iq1_xxs", 1.3125), ("iq1_xs", 1.4375), ("iq1_s", 1.5625),
    ("iq1_m", 1.75), ("iq2_xxs", 2.0625),
    ("iq1_kt", 1.75), ("iq2_kt", 2.125), ("iq2_ks", 2.1875), ("iq2_xs", 2.3125),
    ("q1_0_g128", 1.125), ("iq1_s_r4", 1.5), ("tq1_0", 1.6875),
]
REGIMES = {
    # B: the routed-expert regime (FFN carries ~2/3 of the weights, attention stays at a sane precision)
    "B": lambda t: [r"^token_embd\.weight$=q6_k", r"\.attn_(q|k|v|output)\.weight$=iq4_xs", r"\.ffn_(gate|up|down)\.weight$=" + t],
    # A: every block matrix at the type under test
    "A": lambda t: [r"^token_embd\.weight$=q6_k", r"\.(attn_(q|k|v|output)|ffn_(gate|up|down))\.weight$=" + t],
}


# Equal-bpw competitors built ONLY from types this fork already had: a fraction f of the FFN layers at Q1_0_G128 (1.125 bpw)
# and the rest at IQ1_S (1.5625 bpw) averages to a narrow type's bpw. f = (1.5625 - target) / (1.5625 - 1.125).
#   target 1.4375 (IQ1_XS)  -> f = 0.286 -> 4.6 of 16 layers   (5 layers: 1.426 bpw, i.e. slightly SMALLER than IQ1_XS)
#   target 1.3125 (IQ1_XXS) -> f = 0.571 -> 9.1 of 16 layers   (9 layers: 1.317 bpw, essentially equal)
#   target 1.1875 (IQ1_XXXS)-> f = 0.857 -> 13.7 of 16 layers  (14 layers: 1.1875 -> 1.1875 + ... = 1.1875 bpw)
# name -> (layers at the 1.125-bpw type, base type for the other layers). Layers are the middle block (the early/late
# layers are the fragile ones), and a second placement is run for the first mix to expose placement sensitivity.
MIXES = {
    "mix5_mid":  (list(range(5, 10)), "iq1_s"),
    "mix5_edge": ([0, 1, 2, 14, 15], "iq1_s"),
    "mix9_mid":  (list(range(3, 12)), "iq1_s"),
    "mix14_mid": (list(range(1, 15)), "iq1_s"),
}


# Equal-FILE-SIZE competitors that move bits somewhere other than the FFN (relevant for a small dense model whose tied
# embedding and attention are a big share of the file; for a MoE whose routed experts are ~95% of the bytes these knobs are
# worth almost nothing, which is exactly why regime B is the one that generalises). name -> (rules, default type, note)
ALLOCS = {
    "S_attn_iq3xxs":  ([r"^token_embd\.weight$=q6_k", r"\.attn_(q|k|v|output)\.weight$=iq3_xxs", r"\.ffn_(gate|up|down)\.weight$=iq1_s"], "IQ1_S",
                       "IQ1_S FFN, attention IQ3_XXS (file ~ B-iq1_xxs)"),
    "S_attn_iq2xxs":  ([r"^token_embd\.weight$=q6_k", r"\.attn_(q|k|v|output)\.weight$=iq2_xxs", r"\.ffn_(gate|up|down)\.weight$=iq1_s"], "IQ1_S",
                       "IQ1_S FFN, attention IQ2_XXS (file ~ B-iq1_xxxs)"),
    "S_embd_q4k":     ([r"^token_embd\.weight$=q4_k", r"\.attn_(q|k|v|output)\.weight$=iq4_xs", r"\.ffn_(gate|up|down)\.weight$=iq1_s"], "IQ1_S",
                       "IQ1_S FFN, attention IQ4_XS, token_embd Q4_K (file smaller than B-iq1_xxxs)"),
}


def mix_rules(layers: list[int], base: str) -> list[str]:
    alt = "|".join(str(i) for i in layers)
    return [r"^token_embd\.weight$=q6_k", r"\.attn_(q|k|v|output)\.weight$=iq4_xs",
            r"^blk\.(" + alt + r")\.ffn_(gate|up|down)\.weight$=q1_0_g128", r"\.ffn_(gate|up|down)\.weight$=" + base]


def log(msg: str) -> None:
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(W, "h2h.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(argv: list[str], out: str) -> int:
    t0 = time.time()
    with open(out, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, creationflags=NOWIN)
    log(f"rc={p.returncode} {time.time()-t0:7.1f}s  {os.path.basename(argv[0])} -> {os.path.basename(out)}")
    return p.returncode


def parse_ppl(text: str) -> dict:
    r: dict = {}
    pats = {
        "ppl_final": r"Final estimate: PPL = ([0-9.]+)",
        "ppl_q": r"Mean PPL\(Q\)\s*:\s*([0-9.]+)",
        "ppl_base": r"Mean PPL\(base\)\s*:\s*([0-9.]+)",
        "kld_mean": r"Mean\s+KLD:\s+([0-9.eE+-]+)",
        "kld_median": r"Median\s+KLD:\s+([0-9.eE+-]+)",
        "kld_p99": r"99\.0%\s+KLD:\s+([0-9.eE+-]+)",
        "kld_max": r"Maximum KLD:\s+([0-9.eE+-]+)",
        "same_top_p": r"Same top p:\s+([0-9.]+)",
    }
    for k, pat in pats.items():
        m = re.search(pat, text)
        if m:
            r[k] = float(m.group(1))
    m = re.search(r"Mean\s+KLD:\s+[0-9.eE+-]+\s+\+/-\s+([0-9.eE+-]+)", text)
    if m:
        r["kld_mean_se"] = float(m.group(1))
    return r


def read(p: str) -> str:
    return open(p, encoding="utf-8", errors="replace").read()


def stage_imatrix() -> None:
    im = os.path.join(W, "imatrix.dat")
    if os.path.exists(im):
        log("imatrix exists, skipping")
        return
    rc = run([f"{BIN}/llama-imatrix.exe", "-m", SRC, "-f", TRAIN, "-c", str(CTX), "--chunks", str(CH_IM), "-t", str(THREADS), "-o", im],
             os.path.join(W, "imatrix.out"))
    if rc != 0 or not os.path.exists(im):
        raise SystemExit("imatrix failed -- see imatrix.out")


def stage_base() -> None:
    kld = os.path.join(W, "base.kld")
    if os.path.exists(kld) and os.path.exists(os.path.join(W, "base.out")):
        log("base logits exist, skipping")
        return
    rc = run([f"{BIN}/llama-perplexity.exe", "-m", SRC, "-f", TEST, "-c", str(CTX), "--chunks", str(CH_KLD), "-t", str(THREADS),
              "--kl-divergence-base", kld], os.path.join(W, "base.out"))
    if rc != 0:
        raise SystemExit("base run failed -- see base.out")
    log("base: " + json.dumps(parse_ppl(read(os.path.join(W, "base.out")))))


def stage_run(only: list[str] | None = None) -> None:
    res_path = os.path.join(W, "results.jsonl")
    done = set()
    if os.path.exists(res_path):
        for ln in open(res_path, encoding="utf-8"):
            if ln.strip():
                d = json.loads(ln)
                done.add((d["regime"], d["type"]))
    cand = os.path.join(W, "cand.gguf")
    cand_f32 = os.path.join(W, "cand_f32.gguf")
    for regime in ("B", "A"):
        for t, bpw in TYPES:
            if only and f"{regime}:{t}" not in only and t not in only:
                continue
            if (regime, t) in done:
                continue
            tag = f"{regime}-{t}"
            log(f"=== {tag} ({bpw} bpw)")
            q0 = time.time()
            rules = ",".join(REGIMES[regime](t))
            rc = run([f"{BIN}/llama-quantize.exe", "--imatrix", os.path.join(W, "imatrix.dat"), "--allow-requantize", "--pure",
                      "--custom-q", rules, SRC, cand, t.upper(), str(THREADS)], os.path.join(W, f"q_{tag}.out"))
            if rc != 0:
                log(f"QUANTIZE FAILED for {tag}; recording the failure and moving on")
                with open(res_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"regime": regime, "type": t, "bpw_type": bpw, "failed": "quantize", "rc": rc}) + "\n")
                continue
            qsec = time.time() - q0
            size = os.path.getsize(cand)
            rc = run([f"{BIN}/llama-quantize.exe", "--allow-requantize", cand, cand_f32, "F32", str(THREADS)], os.path.join(W, f"d_{tag}.out"))
            if rc != 0:
                log(f"DEQUANT-REQUANTIZE FAILED for {tag}")
                with open(res_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"regime": regime, "type": t, "bpw_type": bpw, "failed": "dequant", "rc": rc}) + "\n")
                continue
            e0 = time.time()
            out = os.path.join(W, f"p_{tag}.out")
            rc = run([f"{BIN}/llama-perplexity.exe", "-m", cand_f32, "-f", TEST, "-c", str(CTX), "--chunks", str(CH_KLD), "-t", str(THREADS),
                      "--kl-divergence-base", os.path.join(W, "base.kld"), "--kl-divergence"], out)
            r = parse_ppl(read(out))
            r.update({"regime": regime, "type": t, "bpw_type": bpw, "file_bytes": size, "quantize_s": round(qsec, 1),
                      "eval_s": round(time.time() - e0, 1), "eval_rc": rc, "chunks": CH_KLD, "ctx": CTX})
            with open(res_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
            log("result " + json.dumps(r))


def stage_mix(only: list[str] | None = None) -> None:
    """Equal-bpw competitors made of pre-existing types only (see MIXES)."""
    res_path = os.path.join(W, "results.jsonl")
    done = set()
    if os.path.exists(res_path):
        for ln in open(res_path, encoding="utf-8"):
            if ln.strip():
                d = json.loads(ln)
                done.add((d["regime"], d["type"]))
    # own file names: this stage may run beside stage_run (two light jobs), which owns cand.gguf / cand_f32.gguf
    cand = os.path.join(W, "cand_mix.gguf")
    cand_f32 = os.path.join(W, "cand_mix_f32.gguf")
    for name, (layers, base) in MIXES.items():
        if only and name not in only:
            continue
        if ("M", name) in done:
            continue
        frac = len(layers) / 16.0
        bpw = round(dict(TYPES)[base] - frac*(dict(TYPES)[base] - 1.125), 4)
        log(f"=== M-{name} ({bpw} bpw FFN: layers {layers} at q1_0_g128, rest {base})")
        rc = run([f"{BIN}/llama-quantize.exe", "--imatrix", os.path.join(W, "imatrix.dat"), "--allow-requantize", "--pure",
                  "--custom-q", ",".join(mix_rules(layers, base)), SRC, cand, base.upper(), str(THREADS)], os.path.join(W, f"q_M-{name}.out"))
        if rc != 0:
            continue
        size = os.path.getsize(cand)
        if run([f"{BIN}/llama-quantize.exe", "--allow-requantize", cand, cand_f32, "F32", str(THREADS)], os.path.join(W, f"d_M-{name}.out")) != 0:
            continue
        e0 = time.time()
        out = os.path.join(W, f"p_M-{name}.out")
        rc = run([f"{BIN}/llama-perplexity.exe", "-m", cand_f32, "-f", TEST, "-c", str(CTX), "--chunks", str(CH_KLD), "-t", str(THREADS),
                  "--kl-divergence-base", os.path.join(W, "base.kld"), "--kl-divergence"], out)
        r = parse_ppl(read(out))
        r.update({"regime": "M", "type": name, "bpw_type": bpw, "file_bytes": size, "eval_s": round(time.time() - e0, 1), "eval_rc": rc,
                  "chunks": CH_KLD, "ctx": CTX})
        with open(res_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
        log("result " + json.dumps(r))


def stage_alloc() -> None:
    """Equal-file-size competitors that spend the saved bits outside the FFN (see ALLOCS)."""
    res_path = os.path.join(W, "results.jsonl")
    done = set()
    if os.path.exists(res_path):
        for ln in open(res_path, encoding="utf-8"):
            if ln.strip():
                d = json.loads(ln)
                done.add((d["regime"], d["type"]))
    cand = os.path.join(W, "cand_alloc.gguf")
    cand_f32 = os.path.join(W, "cand_alloc_f32.gguf")
    for name, (rules, deft, note) in ALLOCS.items():
        if ("C", name) in done:
            continue
        log(f"=== C-{name}: {note}")
        if run([f"{BIN}/llama-quantize.exe", "--imatrix", os.path.join(W, "imatrix.dat"), "--allow-requantize", "--pure",
                "--custom-q", ",".join(rules), SRC, cand, deft, str(THREADS)], os.path.join(W, f"q_C-{name}.out")) != 0:
            continue
        size = os.path.getsize(cand)
        if run([f"{BIN}/llama-quantize.exe", "--allow-requantize", cand, cand_f32, "F32", str(THREADS)], os.path.join(W, f"d_C-{name}.out")) != 0:
            continue
        e0 = time.time()
        out = os.path.join(W, f"p_C-{name}.out")
        rc = run([f"{BIN}/llama-perplexity.exe", "-m", cand_f32, "-f", TEST, "-c", str(CTX), "--chunks", str(CH_KLD), "-t", str(THREADS),
                  "--kl-divergence-base", os.path.join(W, "base.kld"), "--kl-divergence"], out)
        r = parse_ppl(read(out))
        r.update({"regime": "C", "type": name, "bpw_type": 1.5625, "note": note, "file_bytes": size, "eval_s": round(time.time() - e0, 1),
                  "eval_rc": rc, "chunks": CH_KLD, "ctx": CTX})
        with open(res_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
        log("result " + json.dumps(r))


def stage_real() -> None:
    """Real-kernel check: the quantised file itself (vec_dot + q8_K activations) vs its dequantised twin, few chunks."""
    for t in ("iq1_xs", "iq1_s"):
        tag = f"real-{t}"
        cand = os.path.join(W, f"{tag}.gguf")
        rules = ",".join(REGIMES["B"](t))
        run([f"{BIN}/llama-quantize.exe", "--imatrix", os.path.join(W, "imatrix.dat"), "--allow-requantize", "--pure", "--custom-q", rules, SRC, cand, t.upper(), str(THREADS)],
            os.path.join(W, f"q_{tag}.out"))
        twin = os.path.join(W, f"{tag}_f32.gguf")
        run([f"{BIN}/llama-quantize.exe", "--allow-requantize", cand, twin, "F32", str(THREADS)], os.path.join(W, f"d_{tag}.out"))
        for label, model in (("kernel", cand), ("dequant-twin", twin)):
            out = os.path.join(W, f"p_{tag}_{label}.out")
            run([f"{BIN}/llama-perplexity.exe", "-m", model, "-f", TEST, "-c", str(CTX), "--chunks", "2", "-t", str(THREADS)], out)
            log(f"real-check {t} {label}: " + json.dumps(parse_ppl(read(out))))


def main(argv: list[str]) -> int:
    os.makedirs(W, exist_ok=True)
    stages = argv or ["all"]
    if "all" in stages:
        stages = ["imatrix", "base", "run", "real"]
    filt = [a for a in argv if ":" in a or a in dict(TYPES) or a in MIXES] or None
    for s in stages:
        if s == "mix":
            stage_mix(filt)
            continue
        if s == "alloc":
            stage_alloc()
            continue
        if s == "imatrix":
            stage_imatrix()
        elif s == "base":
            stage_base()
        elif s == "run":
            stage_run(filt)
        elif s == "real":
            stage_real()
        elif s in (filt or []):
            continue
        else:
            raise SystemExit(f"unknown stage {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
