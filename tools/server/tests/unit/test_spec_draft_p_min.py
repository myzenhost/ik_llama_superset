#!/usr/bin/env python
# --spec-draft-p-min must stop a draft at the first low-confidence position. Row
# S69-SPEC-DRAFT-P-MIN-GATE-NEVER-FIRES-UNNORMALISED-DIST-2026-09-28-01: the per-sequence draft impl gated on
# cur_p->data[0].p, which this fork's DIST leaves UNNORMALISED (exp(logit - max): the top always reads 1.0), so the gate
# never fired and every p_min drafted exactly as much as p_min 0. Fails before the fix (equal draft counts at every
# p_min), passes after (counts fall as p_min rises).
#
# Same model drafts for itself (stories260K; see test_spec_draft_sample.py for why not stories15M_MOE), temperature 1,
# fixed seeds, so the three runs differ ONLY in p_min.
import glob
import os
import pytest

from pathlib import Path
import sys
path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(path))

from utils import ServerPreset, ServerProcess  # noqa: E402

server: ServerProcess | None = None


def _draft_path() -> str:
    roots = [os.environ.get("LLAMA_CACHE", ""), os.environ.get("HUGGINGFACE_HUB_CACHE", "")]
    for r in roots:
        if r:
            hits = glob.glob(os.path.join(r, "models--ggml-org--test-model-stories260K", "snapshots", "*", "stories260K-f32.gguf"))
            if hits:
                return hits[0]
    pytest.skip("stories260K-f32.gguf not in the HF cache (LLAMA_CACHE / HUGGINGFACE_HUB_CACHE)")


def _drafts(p_min: float) -> tuple[int, int]:
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 2048
    server.server_port = 18433             # NOT 8081: that port is the live serving instance on this host
    server.fa = "off"
    server.model_draft = _draft_path()
    server.spec_type = "draft-simple"
    server.spec_draft_n_max = 8
    server.extra_args = ["--spec-draft-p-min", str(p_min)]
    server.start()
    n_draft = n_acc = 0
    try:
        for seed in (11, 12, 13, 14, 15, 16):
            res = server.make_request("POST", "/completion", data={
                "prompt": "Once upon a time, there was a little girl", "temperature": 1.0, "top_k": 10, "top_p": 1.0,
                "min_p": 0.0, "seed": seed, "n_predict": 64})
            assert res.status_code == 200, res.body
            t = res.body["timings"]
            n_draft += t.get("draft_n", 0)
            n_acc += t.get("draft_n_accepted", 0)
    finally:
        server.stop()
    return n_draft, n_acc


def test_p_min_gate_stops_low_confidence_drafts():
    d0, a0 = _drafts(0.0)
    d5, a5 = _drafts(0.5)
    d75, a75 = _drafts(0.75)  # the server default (common.h), recorded so the change to DEFAULT drafting is measured
    d9, a9 = _drafts(0.9)
    print(f"p_min 0.0: {a0}/{d0} accepted; 0.5: {a5}/{d5}; 0.75 (default): {a75}/{d75}; 0.9: {a9}/{d9}")
    assert d0 > 50, f"too few drafted tokens at p_min 0 to compare: {d0}"
    assert d5 < d0, f"p_min 0.5 drafted as much as p_min 0 ({d5} vs {d0}): the gate never fired"
    assert d9 <= d75 <= d5, f"drafting is not monotone in p_min: 0.5 {d5}, 0.75 {d75}, 0.9 {d9}"
    assert d9 < d5, f"p_min 0.9 did not draft less than 0.5 ({d9} vs {d5})"
    # the gate keeps the confident drafts, so what survives is accepted at least as often
    if d9 > 0:
        assert a9 / d9 >= a0 / d0 - 0.05, f"gated drafts accepted less often: {a9}/{d9} vs {a0}/{d0}"
