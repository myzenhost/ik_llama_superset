#!/usr/bin/env python
# --spec-draft-sample: the draft model samples x ~ q and the server verifies by rejection sampling (accept min(1, p/q),
# else draw from the residual). Row S69-SPEC-DECODE-REJECTION-SAMPLING-ACCEPTANCE-PICK-2026-09-28-01.
#
# MODEL CHOICE, MEASURED 2026-09-28 (temp/_samp_0928_diag.log): stories15M_MOE is degenerate here (" girl girl ...",
# top-token prob >= 0.999 at every position) so p is one-hot and BOTH rules accept 100% -- it cannot discriminate.
# stories260K has real entropy (mean top-token prob 0.62-0.68, varied text across seeds) and is used instead.
#
# Discriminating setup: the SAME model drafts for itself, and the request samples with temperature 1 / top_k 10 -- the
# same chain the draft sampler runs (TOP_K 10 + DIST) -- so p and q are (nearly) the same distribution. Then
#   rejection sampling accepts with sum_x min(p, q)  ~= 1
#   the argmax draft (flag off) accepts with p(argmax q) = the top token's probability, well below 1 at T=1.
# So the flag must raise the acceptance rate by a wide margin. The exactness control: at temperature 0 the target is
# one-hot, and the output with the flag must equal the output with no draft model at all.
import glob
import os
import pytest

from pathlib import Path
import sys
path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(path))

from utils import ServerPreset, ServerProcess  # noqa: E402

server: ServerProcess | None = None   # set by create_server()


def _draft_path() -> str:
    roots = [os.environ.get("LLAMA_CACHE", ""), os.environ.get("HUGGINGFACE_HUB_CACHE", "")]
    for r in roots:
        if r:
            hits = glob.glob(os.path.join(r, "models--ggml-org--test-model-stories260K", "snapshots", "*", "stories260K-f32.gguf"))
            if hits:
                return hits[0]
    pytest.skip("stories260K-f32.gguf not in the HF cache (LLAMA_CACHE / HUGGINGFACE_HUB_CACHE)")


def create_server(draft_sample: bool, with_draft: bool = True):
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 2048
    server.server_port = 18432             # NOT 8081: that port is the live serving instance on this host
    server.fa = "off"
    if with_draft:
        server.model_draft = _draft_path()
        server.spec_type = "draft-simple"
        server.spec_draft_n_max = 8
        # p_min 0: draft every position, so both modes are measured over the same positions instead of the
        # greedy gate stopping the draft early
        server.extra_args = ["--spec-draft-p-min", "0"] + (["--spec-draft-sample"] if draft_sample else [])


def _acceptance(draft_sample: bool) -> tuple[int, int]:
    create_server(draft_sample)
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


def test_rejection_sampling_accepts_more_when_p_equals_q():
    d_off, a_off = _acceptance(False)
    d_on, a_on = _acceptance(True)
    assert d_off > 50 and d_on > 50, f"too few drafted tokens to compare: off={d_off} on={d_on}"
    r_off, r_on = a_off / d_off, a_on / d_on
    print(f"argmax draft acceptance {a_off}/{d_off} = {r_off:.3f}; --spec-draft-sample {a_on}/{d_on} = {r_on:.3f}")
    assert r_on >= 0.9, f"p ~= q, so rejection sampling should accept almost every token; got {r_on:.3f}"
    assert r_on >= r_off + 0.1, f"the flag should beat the argmax draft by a wide margin: {r_on:.3f} vs {r_off:.3f}"


def test_greedy_output_unchanged_by_the_flag():
    req = {"prompt": "I believe the meaning of life is", "temperature": 0.0, "seed": 42, "n_predict": 32,
           "return_tokens": True}
    create_server(False, with_draft=False)
    server.start()
    try:
        res = server.make_request("POST", "/completion", data=req)
        assert res.status_code == 200, res.body
        tokens_plain = res.body["tokens"]
    finally:
        server.stop()
    create_server(True)
    server.start()
    try:
        res = server.make_request("POST", "/completion", data=req)
        assert res.status_code == 200, res.body
        assert res.body["timings"]["draft_n"] > 0, "the draft model must actually have drafted"
        assert res.body["tokens"] == tokens_plain, "a one-hot target must make rejection sampling exact"
    finally:
        server.stop()
