#!/usr/bin/env python
# Per-request speculative.n_max / n_min / p_min must reach the draft. They were compiled out (server-schema.cpp `#if 0`,
# naming an upstream sub-struct this fork does not have), so a request carrying them was silently served with the
# server-wide values. Row S69-SERVER-PER-REQUEST-SPECULATIVE-FIELDS-PARSED-NEVER-READ-2026-09-28-01.
#
# One server, one set of seeds; the requests differ ONLY in their speculative.* fields. Same model drafts for itself
# (stories260K; see test_spec_draft_p_min.py), temperature 1.
import glob
import os
import pytest

from pathlib import Path
import sys
path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(path))

from utils import ServerPreset, ServerProcess  # noqa: E402

server: ServerProcess | None = None
SEEDS = (11, 12, 13, 14, 15, 16)


def _draft_path() -> str:
    roots = [os.environ.get("LLAMA_CACHE", ""), os.environ.get("HUGGINGFACE_HUB_CACHE", "")]
    for r in roots:
        if r:
            hits = glob.glob(os.path.join(r, "models--ggml-org--test-model-stories260K", "snapshots", "*", "stories260K-f32.gguf"))
            if hits:
                return hits[0]
    pytest.skip("stories260K-f32.gguf not in the HF cache (LLAMA_CACHE / HUGGINGFACE_HUB_CACHE)")


@pytest.fixture(autouse=True)          # per test: conftest stops every server after each test
def spec_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 2048
    server.server_port = 18436             # NOT 8081: that port is the live serving instance on this host
    server.fa = "off"
    server.model_draft = _draft_path()
    server.spec_type = "draft-simple"
    server.spec_draft_n_max = 8             # server-wide; no --spec-draft-p-min, so the gate is off by default
    server.start()
    yield
    server.stop()


def _drafts(extra: dict) -> tuple[int, int, int]:
    n_draft = n_acc = n_steps = 0
    for seed in SEEDS:
        body = {"prompt": "Once upon a time, there was a little girl", "temperature": 1.0, "top_k": 10, "top_p": 1.0,
                "min_p": 0.0, "seed": seed, "n_predict": 64}
        body.update(extra)
        res = server.make_request("POST", "/completion", data=body)
        assert res.status_code == 200, res.body
        t = res.body["timings"]
        n_draft += t.get("draft_n", 0)
        n_acc += t.get("draft_n_accepted", 0)
        n_steps += t.get("predicted_n", 0)
    return n_draft, n_acc, n_steps


def test_per_request_speculative_fields_reach_the_draft():
    base = _drafts({})
    same = _drafts({})                                   # determinism control: identical requests, identical counts
    nmax1 = _drafts({"speculative.n_max": 1})
    pmin9 = _drafts({"speculative.p_min": 0.9})
    nmin = _drafts({"speculative.n_min": 1000})          # longer than any draft can be -> every draft discarded
    print(f"base {base}  same {same}  n_max=1 {nmax1}  p_min=0.9 {pmin9}  n_min=1000 {nmin}")
    assert base == same, f"not deterministic, comparison is void: {base} vs {same}"
    assert base[0] > 50, f"too few drafted tokens to compare: {base}"
    # n_max = 1: at most one drafted token per draft step, far fewer than with the server's 8
    assert nmax1[0] < base[0], f"speculative.n_max ignored: {nmax1} vs {base}"
    assert nmax1[0] <= nmax1[2], f"n_max=1 drafted more than one per generated token: {nmax1}"
    # p_min = 0.9 on a server whose gate is off: fewer drafted tokens
    assert pmin9[0] < base[0], f"speculative.p_min ignored: {pmin9} vs {base}"
    # n_min larger than any possible draft: nothing is proposed
    assert nmin[0] == 0, f"speculative.n_min ignored: {nmin}"


def test_bad_p_min_is_refused():
    res = server.make_request("POST", "/completion", data={"prompt": "hi", "n_predict": 1, "speculative.p_min": 1.5})
    assert res.status_code == 400, res.body
