"""Skew sampling (ExLlamaV2 / Aphrodite), row S69-APHRODITE-PICK-SKEW-SAMPLING-2026-10-18-01.

The draw over the candidates the other samplers kept, sorted by probability with cumulative F_j, picks
token j with probability F_j^a - F_{j-1}^a, a = exp(skew) (common/sampling-skew.h). post_sampling_probs
reports the list the draw used, so the skewed numbers can be predicted from the unskewed ones.

A server that ignores skew returns the unskewed numbers and no "skew" in generation_settings; both are
refused below.
"""
import math

import pytest
from utils import ServerPreset

server = ServerPreset.tinyllama2()

PROMPT = "Once upon a time, there was a little girl who"
TOP_K = 40


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def _probs(**extra) -> list:
    data = {
        "prompt": PROMPT, "n_predict": 1, "n_probs": TOP_K, "post_sampling_probs": True,
        "temperature": 1.0, "top_k": TOP_K, "top_p": 1.0, "min_p": 0.0, "seed": 42, "cache_prompt": False,
    }
    data.update(extra)
    res = server.make_request("POST", "/completion", data=data)
    assert res.status_code == 200, res.body
    return [(t["id"], t["prob"]) for t in res.body["completion_probabilities"][0]["top_probs"]]


def _predict(base: list, skew: float) -> dict:
    a = math.exp(skew)
    out, cum, prev = {}, 0.0, 0.0
    for tid, p in sorted(base, key=lambda x: (-x[1], x[0])):
        cum = min(cum + p, 1.0)
        fa = cum ** a
        out[tid] = fa - prev
        prev = fa
    return out


def test_skew_zero_is_identity():
    server.start()
    assert _probs(skew=0.0) == _probs()


@pytest.mark.parametrize("skew", [-1.0, -0.4, 0.5, 1.5])
def test_skew_matches_the_closed_form_over_the_survivors(skew):
    server.start()
    base = _probs()
    assert len(base) == TOP_K, "premise: top_k survivors all reported"
    assert sum(p for _, p in base) == pytest.approx(1.0, abs=1e-3)
    got = dict(_probs(skew=skew))
    want = _predict(base, skew)
    assert set(got) == set(want)  # skew reshapes; it does not add or drop candidates
    for tid, w in want.items():
        if w > 1e-4:
            assert got[tid] == pytest.approx(w, rel=2e-3, abs=2e-5), (tid, w, got[tid])
    top = max(base, key=lambda x: x[1])[0]
    if skew < 0:
        assert got[top] > dict(base)[top]   # toward the head
    else:
        assert got[top] < dict(base)[top]   # toward the tail


def test_greedy_is_unaffected():
    server.start()
    data = {"prompt": PROMPT, "n_predict": 8, "temperature": 0.0, "cache_prompt": False}
    a = server.make_request("POST", "/completion", data=data)
    b = server.make_request("POST", "/completion", data=dict(data, skew=-2.0))
    assert a.status_code == 200 and b.status_code == 200
    assert a.body["content"] == b.body["content"]


def test_echo_and_refusals():
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": "x", "n_predict": 1, "skew": 0.75})
    assert res.status_code == 200, res.body
    assert res.body["generation_settings"]["skew"] == pytest.approx(0.75)
    for bad in ({"skew": 11.0}, {"skew": -10.5},
                {"skew": 0.5, "mirostat": 2},
                {"skew": 0.5, "adaptive_target": 0.5, "top_k": 40},
                {"skew": 0.5, "fsd_alpha": 0.5}):
        r = server.make_request("POST", "/completion", data=dict({"prompt": "x", "n_predict": 1}, **bad))
        assert r.status_code == 400, (bad, r.status_code, r.body)
        assert "skew" in str(r.body), (bad, r.body)  # refused for skew, not for some other setting
