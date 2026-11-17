#!/usr/bin/env python
# REWRITE-SAFE: this script never reads-then-filters an existing store. Every write goes
# to a NEW path keyed by --tag (baseline_<tag>.json / compare_<a>_vs_<b>.json). The only
# repeated write is the incremental save of the record THIS run is building, so nothing
# authored by an earlier run or another agent can be dropped by a wrong predicate.
"""KV-cache / quant regression harness for the ik_llama fork.

WHY THIS EXISTS
---------------
Any change to how K/V are stored or read can look like a successful port and still
destroy the IQK quant paths this fork exists for (memory:
project-ik-llama-is-divergent-not-behind).  "It builds and generates text" is exactly
the false positive we keep hitting.  So: capture a deterministic BASELINE before the
change, re-run it after, and compare token-level output, not prose.

WHAT IT MEASURES
----------------
For each arm (model x KV cache dtype x flash-attn x offload) it runs llama-cli at
greedy decode (--temp 0, fixed --seed, fixed prompt) and hashes the generated text.
Greedy decode makes the text a faithful proxy for the token-id stream.

CONTROLS -- without these the run measures the harness, not the engine:
  * SELF control (determinism): every arm is run TWICE.  An arm that does not equal
    itself makes the whole comparison UNRELIABLE and is reported as such.
  * DISCRIMINATOR control (sensitivity): arms that MUST differ from each other are
    declared up front (f16 weights vs iq2_bn weights; f16 KV vs q4_0 KV).  If a pair
    that must differ comes back identical, the harness is not reaching the code path
    it claims to test, and its "no regression" verdict is worthless.

USAGE
  pythonw kv_regression_harness.py capture  --tag before
  ... make the change ...
  pythonw kv_regression_harness.py capture  --tag after
  pythonw kv_regression_harness.py compare  --before before --after after
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000   # rule 1.45: windows stay hidden

FORK = Path(".")
BIN = FORK / "build-canonical" / "bin"
CLI = BIN / "llama-cli.exe"
TBO = BIN / "test-backend-ops.exe"
MODELS = Path("models/gguf")
OUT = Path("tmp/_agF_kvregress")

SEED = 42
NPREDICT = 128
THREADS = 12  # deliberately not -j 64: no stacked heavy jobs (memory: power-loss note)

# A KV-CACHE HARNESS MUST FILL THE KV CACHE.
# The first version of this file used a six-token prompt with -n 48.  Every arm was
# deterministic and every discriminator passed, so it looked healthy -- but on GPU the
# KV-f16 and KV-q8_0 arms produced BYTE-IDENTICAL output, because six tokens of context
# never put enough pressure on the cache for a quantisation difference to flip an argmax.
# An arm that cannot tell f16 from q8_0 cannot detect a KV-cache regression either, which
# is the entire point of the file.  So: a long prompt, more generated tokens, and (below)
# a small -c so the cache actually wraps.
LONG_PROMPT = (
    "You are reading a technical report. Answer strictly from the text.\n\n"
    "REPORT. The archive system stores records in three tiers. Tier one holds active "
    "records on solid-state media and is rebuilt nightly. Tier two holds records older "
    "than ninety days on spinning disks and is rebuilt weekly. Tier three is offline "
    "tape, written once and verified twice, and is never rebuilt. Each record carries a "
    "checksum computed at ingest and recomputed on every migration between tiers. A "
    "mismatch quarantines the record and raises an alert to the duty engineer. In the "
    "last audit, tier one held 412,000 records, tier two held 3,180,000 records, and "
    "tier three held 41,900,000 records. Seventeen records were quarantined, of which "
    "fourteen were traced to a single failed disk controller in tier two and three "
    "remained unexplained. The retention policy requires tier one records to migrate to "
    "tier two after ninety days, tier two records to migrate to tier three after two "
    "years, and tier three records to be retained for twenty-five years before "
    "destruction. Destruction requires two authorised signatures and is logged to an "
    "append-only ledger that is itself replicated to two geographically separate sites. "
    "The duty engineer rotates weekly and is responsible for clearing the quarantine "
    "queue within one business day. The audit recommended three changes: recompute "
    "checksums on read as well as on migration, replace the failed controller and its "
    "unfailed twin, and investigate the three unexplained quarantines before the next "
    "audit cycle.\n\n"
    "QUESTION. Summarise the three tiers, state how many records were quarantined and "
    "how many of those were explained, and list the three recommended changes.\n\n"
    "ANSWER:"
)
PROMPT = LONG_PROMPT

# --- models -------------------------------------------------------------------
# One IQK-path narrow quant (IQ2_BN is ik_llama-local), one K-quant, one F16 control.
#
# MODEL CHOICE IS A CONTROL DECISION, NOT A CONVENIENCE.
# The first version of this harness used TriLM 190M/830M as the IQ2_BN arms because they
# are small and fast. MEASURED 2026-09-02: they are DEGENERATE -- TriLM base models
# collapse into a repeating attractor ("Table 1 / Table 2 / Table 3 ...") whose argmax is
# locked, so their output is byte-identical across KV f16, q8_0 and q4_0. An arm that
# cannot tell f16 from q4_0 cannot detect a KV-cache regression either, which is the whole
# purpose of the file. (memory:project-ik-llama-fork-consolidated-location already records
# TriLM's repetition defeating an A/B control once before.)
# BitNet-2B4T is an INSTRUCT model on the same ik_llama-local IQ2_BN quant, it stays
# coherent, and it was verified KV-sensitive (f16 vs q4_0 DIFFER) before being adopted here.
# The TriLM arms are kept -- they still catch gross breakage -- but they are NOT
# discriminators.
#
MODELS_UNDER_TEST = {
    # IQK narrow quant, non-degenerate, KV-sensitive: the arms that certify the IQK path
    "bitnet2b4t_iq2bn": MODELS / "bitnet-2b4t-iq2bn/bitnet1582b4t-iq2_bn.gguf",
    # the R4 REPACKED variant -- exercises ik_llama's quant-repacking path specifically
    "bitnet2b4t_iq2bn_r4": MODELS / "bitnet-2b4t-iq2bn/bitnet1582b4t-iq2_bn_r4.gguf",
    # K-quant, verified KV-sensitive
    "outetts500m_q4km": MODELS / "audio/OuteTTS-0.3-500M-Q4_K_M.gguf",
    # kept for gross-breakage coverage only -- see the note above, NOT discriminators
    "trilm190m_iq2bn": MODELS / "trilm/TriLM_190M/TriLM_190M-iq2_bn.gguf",
    "trilm190m_f16": MODELS / "trilm/TriLM_190M/TriLM_190M-f16.gguf",
    "trilm830m_iq2bn": MODELS / "trilm/TriLM_830M/TriLM_830M-iq2_bn.gguf",
}

# --- arms ---------------------------------------------------------------------
# (arm_name, model_key, cache_type_k, cache_type_v, flash_attn, n_gpu_layers)
#
# MEASURED PRECONDITION (2026-09-02, this fork, build 4926 ea0fd2d8): a QUANTISED V
# cache requires flash-attn.  `-ctv q8_0 -fa off` does not print an error -- it
# fast-fails with 0xC0000409 (STATUS_STACK_BUFFER_OVERRUN) right after "main: seed".
# `-ctk q8_0 -ctv f16 -fa off` is fine, so it is the V path specifically.  Arms below
# respect that; a quantised-V arm always carries -fa on.  Do not "fix" this by removing
# the FA flag -- you will be measuring a crash, not the KV cache.
#
# (arm, model_key, ctk, ctv, fa, ngl, n_ctx)
ARMS = [
    # --- PRIMARY: IQ2_BN on a coherent instruct model. These arms certify the IQK path.
    ("bitnet.kv_f16.fa_off.cpu", "bitnet2b4t_iq2bn", "f16", "f16", "off", 0, 2048),
    ("bitnet.kv_f16.fa_on.cpu", "bitnet2b4t_iq2bn", "f16", "f16", "on", 0, 2048),
    ("bitnet.kv_q8_0.fa_on.cpu", "bitnet2b4t_iq2bn", "q8_0", "q8_0", "on", 0, 2048),
    ("bitnet.kv_q4_0.fa_on.cpu", "bitnet2b4t_iq2bn", "q4_0", "q4_0", "on", 0, 2048),
    ("bitnet.kq8_vf16.fa_off.cpu", "bitnet2b4t_iq2bn", "q8_0", "f16", "off", 0, 2048),
    ("bitnet.kv_f16.fa_off.cpu.ctx512", "bitnet2b4t_iq2bn", "f16", "f16", "off", 0, 512),
    ("bitnet.kv_f16.fa_on.gpu", "bitnet2b4t_iq2bn", "f16", "f16", "on", 99, 2048),
    ("bitnet.kv_q4_0.fa_on.gpu", "bitnet2b4t_iq2bn", "q4_0", "q4_0", "on", 99, 2048),
    # the R4 repacked quant -- ik_llama's repacking path
    ("bitnetr4.kv_f16.fa_off.cpu", "bitnet2b4t_iq2bn_r4", "f16", "f16", "off", 0, 2048),
    ("bitnetr4.kv_q8_0.fa_on.cpu", "bitnet2b4t_iq2bn_r4", "q8_0", "q8_0", "on", 0, 2048),
    # --- SECONDARY: TriLM. Degenerate (see MODELS_UNDER_TEST note); gross-breakage only.
    ("iq2bn.kv_f16.fa_off.cpu", "trilm190m_iq2bn", "f16", "f16", "off", 0, 2048),
    ("iq2bn.kv_f16.fa_on.cpu", "trilm190m_iq2bn", "f16", "f16", "on", 0, 2048),
    ("iq2bn.kv_q8_0.fa_on.cpu", "trilm190m_iq2bn", "q8_0", "q8_0", "on", 0, 2048),
    ("iq2bn.kv_q4_0.fa_on.cpu", "trilm190m_iq2bn", "q4_0", "q4_0", "on", 0, 2048),
    # quantised K with f16 V and NO flash-attn: exercises the quantised-K read path
    # through the ordinary (non-FA) attention kernels.  This is the combination a
    # KV-cache rewrite is most likely to break silently.
    ("iq2bn.kq8_vf16.fa_off.cpu", "trilm190m_iq2bn", "q8_0", "f16", "off", 0, 2048),
    # SMALL CONTEXT: prompt + generation exceed n_ctx, so the cache must wrap/shift.
    # Cache eviction and the K-shift are where KV-cache defects actually live; an arm
    # that never fills the cache cannot see them.
    ("iq2bn.kv_f16.fa_off.cpu.ctx512", "trilm190m_iq2bn", "f16", "f16", "off", 0, 512),
    ("iq2bn.kv_q8_0.fa_on.cpu.ctx512", "trilm190m_iq2bn", "q8_0", "q8_0", "on", 0, 512),
    # K-quant arms
    ("q4km.kv_f16.fa_off.cpu", "outetts500m_q4km", "f16", "f16", "off", 0, 2048),
    ("q4km.kv_q8_0.fa_on.cpu", "outetts500m_q4km", "q8_0", "q8_0", "on", 0, 2048),
    ("q4km.kq8_vf16.fa_off.cpu", "outetts500m_q4km", "q8_0", "f16", "off", 0, 2048),
    # F16 weights -- discriminator partner for the iq2bn arm
    ("f16w.kv_f16.fa_off.cpu", "trilm190m_f16", "f16", "f16", "off", 0, 2048),
    # larger IQK model, guards against a defect that only shows with more layers
    ("iq2bn830.kv_f16.fa_off.cpu", "trilm830m_iq2bn", "f16", "f16", "off", 0, 2048),
    ("iq2bn830.kv_q8_0.fa_on.cpu", "trilm830m_iq2bn", "q8_0", "q8_0", "on", 0, 2048),
    ("iq2bn830.kv_f16.fa_off.cpu.ctx512", "trilm830m_iq2bn", "f16", "f16", "off", 0, 512),
    # GPU arms -- the custom CUDA FA tree is part of what the fork exists for
    ("iq2bn.kv_f16.fa_on.gpu", "trilm190m_iq2bn", "f16", "f16", "on", 99, 2048),
    ("iq2bn.kv_q8_0.fa_on.gpu", "trilm190m_iq2bn", "q8_0", "q8_0", "on", 99, 2048),
    ("iq2bn.kv_q4_0.fa_on.gpu", "trilm190m_iq2bn", "q4_0", "q4_0", "on", 99, 2048),
    ("iq2bn.kv_f16.fa_on.gpu.ctx512", "trilm190m_iq2bn", "f16", "f16", "on", 99, 512),
    ("iq2bn830.kv_q8_0.fa_on.gpu", "trilm830m_iq2bn", "q8_0", "q8_0", "on", 99, 2048),
    ("q4km.kv_f16.fa_on.gpu", "outetts500m_q4km", "f16", "f16", "on", 99, 2048),
]

# Pairs that MUST differ.  If they match, the harness is blind and its verdict is void.
# Note the GPU pair: it is here BECAUSE the short-prompt version of this harness had
# GPU KV-f16 and KV-q8_0 coming back identical. Declaring it makes that blindness a
# harness FAILURE next time instead of a silent gap.
#
# A DISCRIMINATOR MUST ASSERT SOMETHING THAT *MUST* BE TRUE.
# MEASURED 2026-09-02: an earlier version of this list required KV f16 vs KV **q8_0** to
# differ on BitNet, and it came back identical -- twice. That was not the engine failing
# and not the model being degenerate: q8_0 is a near-lossless KV quant, so on 128 greedy
# tokens it very often perturbs nothing enough to flip an argmax. Identical output there
# is the EXPECTED result. Asserting it must differ would have made the harness cry wolf on
# every future run, and a control that fails when nothing is wrong gets ignored, which
# costs more than having no control at all.
# So the required pairs use q4_0 (coarse enough that a difference is genuinely obligatory)
# and cross-model pairs. The q8_0 pairs are kept as INFORMATIONAL: reported, never fatal.
#
DISCRIMINATORS = [
    # IQK path sensitivity -- these are the ones that make a "quants still fine" verdict
    # mean anything. All on BitNet, the model verified to respond to KV dtype.
    ("bitnet.kv_f16.fa_on.cpu", "bitnet.kv_q4_0.fa_on.cpu", "IQ2_BN: KV f16 vs KV q4_0"),
    ("bitnet.kv_f16.fa_on.gpu", "bitnet.kv_q4_0.fa_on.gpu", "IQ2_BN on GPU: KV f16 vs q4_0"),
    # K-quant path sensitivity
    ("q4km.kv_f16.fa_off.cpu", "q4km.kv_q8_0.fa_on.cpu", "Q4_K_M: KV f16 vs KV q8_0"),
    ("q4km.kv_f16.fa_off.cpu", "q4km.kq8_vf16.fa_off.cpu", "Q4_K_M: f16 K vs quantised K"),
    # weights actually reach the output at all
    ("bitnet.kv_f16.fa_off.cpu", "q4km.kv_f16.fa_off.cpu", "different model + quant family"),
    ("iq2bn.kv_f16.fa_off.cpu", "f16w.kv_f16.fa_off.cpu", "iq2_bn weights vs f16 weights"),
]

# Reported, never fatal: near-lossless KV quants legitimately produce identical greedy
# output. Tracked because a CHANGE in whether they match is itself interesting after a port.
INFORMATIONAL_PAIRS = [
    ("bitnet.kv_f16.fa_on.cpu", "bitnet.kv_q8_0.fa_on.cpu", "IQ2_BN: KV f16 vs KV q8_0"),
    ("bitnet.kv_f16.fa_off.cpu", "bitnet.kq8_vf16.fa_off.cpu",
     "IQ2_BN no-FA: f16 K vs quantised K"),
    ("bitnetr4.kv_f16.fa_off.cpu", "bitnetr4.kv_q8_0.fa_on.cpu",
     "IQ2_BN_R4 repacked: KV f16 vs KV q8_0"),
]

# ops the KV cache reads/writes/reshapes
KV_OPS = [
    "CPY", "SET_ROWS", "GET_ROWS", "ROPE", "SOFT_MAX",
    "MUL_MAT", "FLASH_ATTN_EXT", "ADD", "CONT", "PERMUTE", "VIEW", "RESHAPE",
]

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def normalise(raw: str) -> str:
    """Normalise the GENERATED COMPLETION only.

    MEASURED 2026-09-02: an earlier version of this harness hashed stdout+stderr
    together and a line-prefix filter was supposed to strip the log.  It did not.
    `CUDA0: using device CUDA0 - 30994 MiB free` survived the filter and its VRAM
    number moves between runs, so FOUR arms were reported NON-DETERMINISTIC while the
    engine's actual token output was identical every time.  That is the
    instrument-reliability failure this project keeps hitting: the harness measured
    itself.  The fix is not a longer filter list -- it is to stop reading the log at
    all.  llama-cli writes the prompt and the completion to STDOUT and every log line
    to STDERR (verified: `2>/dev/null` leaves exactly the completion), so only stdout
    is hashed.  stderr is kept for diagnosis and never enters the hash.
    """
    return ANSI.sub("", raw).replace("\r", "").strip()


def run_arm(arm, model_path, ctk, ctv, fa, ngl, n_ctx=2048, timeout=900):
    cmd = [
        str(CLI), "-m", str(model_path),
        "-p", PROMPT,
        "-n", str(NPREDICT),
        "-c", str(n_ctx),
        "--temp", "0",
        "--seed", str(SEED),
        "--top-k", "1",
        "-t", str(THREADS),
        "-ctk", ctk, "-ctv", ctv,
        "-fa", fa,
        "-ngl", str(ngl),
        "--no-warmup",
        # NOTE: no -no-cnv here. That flag does not exist in this fork (verified against
        # --help with --seed as a positive control); passing it makes llama-cli print
        # usage and exit 0, which would have looked like a clean run with empty output.
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              errors="replace", creationflags=CREATE_NO_WINDOW)
        rc = proc.returncode
        out, err = proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired:
        rc, out, err = -9, "", "<TIMEOUT>"
    # llama-cli ECHOES the prompt to stdout before the completion. Strip it, so the hash
    # and the stored sample both describe what the MODEL produced. Without this the
    # 1689-char prompt fills the whole 1200-char diagnostic sample and a changed arm
    # cannot be read at all -- the hash still discriminates, but nobody can see WHAT
    # moved, which is half of what a regression harness is for.
    body = normalise(out)
    pfx = normalise(PROMPT)
    if body.startswith(pfx):
        body = body[len(pfx):].strip()
        prompt_echoed = True
    else:
        prompt_echoed = False
    return {
        "arm": arm, "rc": rc,
        "cmd": " ".join(cmd),
        "elapsed_s": round(time.time() - t0, 2),
        "sha256": hashlib.sha256(body.encode("utf-8", "replace")).hexdigest(),
        "chars": len(body),
        # an arm that produced no completion is a FAILED arm even if rc==0
        "empty_completion": len(body) == 0,
        "prompt_echoed": prompt_echoed,
        "text": body[:1200],
        "stderr_tail": err[-800:],
    }


def run_backend_ops():
    res = {}
    for op in KV_OPS:
        try:
            p = subprocess.run([str(TBO), "test", "-o", op], capture_output=True,
                               text=True, timeout=1800, errors="replace")
            out = (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired:
            res[op] = {"rc": -9, "ok": None, "fail": None, "note": "timeout"}
            continue
        ok = fail = None
        m = re.search(r"(\d+)/(\d+) tests passed", out)
        if m:
            ok, fail = int(m.group(1)), int(m.group(2)) - int(m.group(1))
        res[op] = {"rc": p.returncode, "ok": ok, "fail": fail,
                   "tail": out.strip()[-600:]}
    return res


def cmd_capture(args):
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"baseline_{args.tag}.json"
    rec = {"tag": args.tag, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "seed": SEED, "n_predict": NPREDICT, "prompt": PROMPT,
           "git_head": subprocess.run(["git", "-C", str(FORK), "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
           "arms": {}, "self_control": {}, "discriminators": {}, "backend_ops": {}}

    missing = [k for k, v in MODELS_UNDER_TEST.items() if not v.exists()]
    rec["missing_models"] = missing
    if missing:
        print(f"[warn] missing models, arms skipped: {missing}", flush=True)

    for arm, mk, ctk, ctv, fa, ngl, n_ctx in ARMS:
        if mk in missing:
            continue
        mp = MODELS_UNDER_TEST[mk]
        print(f"[run ] {arm} (pass 1)", flush=True)
        r1 = run_arm(arm, mp, ctk, ctv, fa, ngl, n_ctx)
        print(f"[run ] {arm} (pass 2, determinism control)", flush=True)
        r2 = run_arm(arm, mp, ctk, ctv, fa, ngl, n_ctx)
        rec["arms"][arm] = r1
        rec["self_control"][arm] = {
            "deterministic": r1["sha256"] == r2["sha256"],
            "sha_pass1": r1["sha256"], "sha_pass2": r2["sha256"],
            "rc1": r1["rc"], "rc2": r2["rc"],
        }
        print(f"[ {'OK ' if r1['sha256']==r2['sha256'] else 'NON-DET'} ] {arm} "
              f"rc={r1['rc']} {r1['sha256'][:12]} {r1['elapsed_s']}s", flush=True)
        with open(dest, "w", encoding="utf-8") as fh:   # incremental save
            json.dump(rec, fh, indent=1)

    for a, b, why in DISCRIMINATORS:
        if a in rec["arms"] and b in rec["arms"]:
            rec["discriminators"][f"{a} != {b}"] = {
                "why": why,
                "differ": rec["arms"][a]["sha256"] != rec["arms"][b]["sha256"],
            }
    rec["informational_pairs"] = {}
    for a, b, why in INFORMATIONAL_PAIRS:
        if a in rec["arms"] and b in rec["arms"]:
            rec["informational_pairs"][f"{a} vs {b}"] = {
                "why": why,
                "differ": rec["arms"][a]["sha256"] != rec["arms"][b]["sha256"],
            }

    if not args.no_backend_ops:
        print("[run ] test-backend-ops for KV-touching ops", flush=True)
        rec["backend_ops"] = run_backend_ops()

    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(f"[done] {dest}", flush=True)
    print(json.dumps(summarise(rec), indent=1), flush=True)
    return 0


def summarise(rec):
    nondet = [a for a, v in rec["self_control"].items() if not v["deterministic"]]
    baddisc = [k for k, v in rec["discriminators"].items() if not v["differ"]]
    badrc = [a for a, v in rec["arms"].items() if v["rc"] != 0]
    # rc==0 with no completion is the failure mode that made an unknown flag look like a
    # clean run: llama-cli printed usage and exited 0. Treat it as a failed arm.
    empty = [a for a, v in rec["arms"].items() if v.get("empty_completion")]
    return {
        "arms_run": len(rec["arms"]),
        "nonzero_rc": badrc,
        "empty_completion": empty,
        "non_deterministic": nondet,
        "discriminators_that_failed_to_differ": baddisc,
        "harness_reliable": (not nondet) and (not baddisc) and (not badrc) and (not empty),
        "backend_ops": {k: (v.get("ok"), v.get("fail")) for k, v in rec.get("backend_ops", {}).items()},
    }


def cmd_compare(args):
    b = json.load(open(OUT / f"baseline_{args.before}.json", encoding="utf-8"))
    a = json.load(open(OUT / f"baseline_{args.after}.json", encoding="utf-8"))
    rows, changed, missing = [], [], []
    for arm in sorted(set(b["arms"]) | set(a["arms"])):
        if arm not in b["arms"] or arm not in a["arms"]:
            missing.append(arm)
            continue
        same = b["arms"][arm]["sha256"] == a["arms"][arm]["sha256"]
        if not same:
            changed.append(arm)
        rows.append({"arm": arm, "identical": same,
                     "before": b["arms"][arm]["sha256"][:16],
                     "after": a["arms"][arm]["sha256"][:16],
                     "rc_before": b["arms"][arm]["rc"], "rc_after": a["arms"][arm]["rc"],
                     "before_text": b["arms"][arm]["text"][:200] if not same else "",
                     "after_text": a["arms"][arm]["text"][:200] if not same else ""})
    ops = {}
    for op in set(b.get("backend_ops", {})) | set(a.get("backend_ops", {})):
        ops[op] = {"before": b.get("backend_ops", {}).get(op, {}).get("ok"),
                   "after": a.get("backend_ops", {}).get(op, {}).get("ok")}
    verdict = {
        "before_reliable": summarise(b)["harness_reliable"],
        "after_reliable": summarise(a)["harness_reliable"],
        "arms_compared": len(rows),
        "arms_changed": changed,
        "arms_missing_one_side": missing,
        "backend_ops": ops,
        "PASS": (not changed) and (not missing)
                and summarise(b)["harness_reliable"] and summarise(a)["harness_reliable"],
    }
    dest = OUT / f"compare_{args.before}_vs_{args.after}.json"
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump({"verdict": verdict, "rows": rows}, fh, indent=1)
    print(json.dumps(verdict, indent=1))
    print(f"[done] {dest}")
    return 0 if verdict["PASS"] else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--tag", required=True)
    c.add_argument("--no-backend-ops", action="store_true")
    c.set_defaults(fn=cmd_capture)
    d = sub.add_parser("compare")
    d.add_argument("--before", required=True)
    d.add_argument("--after", required=True)
    d.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
