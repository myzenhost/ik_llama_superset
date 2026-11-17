"""top-a and quadratic ("smooth") sampling, added 2026-09-26 while reviewing koboldcpp's sampler set.

Both act on the candidate list, so they are observed through post_sampling_probs: the probabilities
the server reports after the whole chain has run. Every other truncation is switched off so the
sampler under test is the only thing that can change that list. A server that ignores the field
reports the same list as the control, so each assertion fails on a build without the sampler.
"""
import pytest
from utils import *

server = ServerPreset.tinyllama2()

N_PROBS = 64


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def _top_probs(**extra) -> list[dict]:
    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time", "n_predict": 1, "n_probs": N_PROBS, "post_sampling_probs": True,
        "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "seed": 42, "cache_prompt": False,
        **extra,
    })
    assert res.status_code == 200, res.body
    return res.body["completion_probabilities"][0]["top_probs"]


def test_top_a_drops_tokens_below_its_floor():
    server.start()
    base = _top_probs()
    assert len(base) == N_PROBS  # control: nothing truncates without top-a
    p_max = base[0]["prob"]
    a = 0.5
    got = _top_probs(top_a=a)
    assert 0 < len(got) < len(base)
    assert got[0]["id"] == base[0]["id"]
    # everything the base list held at or above the floor survives, everything below it does not
    expect = [t["id"] for t in base if t["prob"] >= a * p_max * p_max]
    assert [t["id"] for t in got] == expect


def test_top_a_zero_is_a_no_op():
    server.start()
    assert _top_probs(top_a=0.0) == _top_probs()


def test_smoothing_sharpens_the_top_and_keeps_its_order():
    server.start()
    base = _top_probs()
    got = _top_probs(smoothing_factor=2.0)
    assert got[0]["id"] == base[0]["id"]
    assert got[0]["prob"] > base[0]["prob"]
    assert got[1]["prob"] / got[0]["prob"] < base[1]["prob"] / base[0]["prob"]
    # curve 1.0 is purely quadratic in the distance from the top: monotonic, so the ranking holds
    # over whatever the smoothed list still carries (its tail underflows to 0 and is not reported)
    n = min(10, len(got))
    assert [t["id"] for t in got[:n]] == [t["id"] for t in base[:n]]


def test_post_sampling_probs_are_probabilities():
    # 2026-09-26: the fork reported exp(logit - max) here, so the top token always read 1.0 and the
    # list summed past 1. With every truncation off, the post-sampling distribution at temperature 1
    # IS the model's distribution, so it must match the pre-sampling probabilities.
    server.start()
    post = _top_probs()
    assert sum(t["prob"] for t in post) <= 1.0 + 1e-4
    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time", "n_predict": 1, "n_probs": 4, "post_sampling_probs": False,
        "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "seed": 42, "cache_prompt": False,
    })
    pre = res.body["completion_probabilities"][0]["top_logprobs"]
    import math
    for a, b in zip(pre, post):
        assert a["id"] == b["id"]
        assert math.exp(a["logprob"]) == pytest.approx(b["prob"], rel=1e-3)


@pytest.mark.parametrize("curve", [1.0, 2.0])
def test_smoothing_runs_after_temperature_like_koboldcpp(curve):
    # koboldcpp (AGPL-3.0; behaviour read from sample_temperature, gpttype_adapter.cpp:2112, no code
    # copied) divides by the temperature first and smooths the scaled logits. The polynomial does
    # not commute with scaling, so at T != 1 the two orders give different ratios. Reference built
    # from the raw distribution: d = logit - max, scaled d/T, then logit' = -k f d'^2 + s f d'^3.
    import math
    server.start()
    T, f = 2.0, 0.5
    k, s = (3 - curve) / 2, (curve - 1) / 2
    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time", "n_predict": 1, "n_probs": 6, "post_sampling_probs": False,
        "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "seed": 42, "cache_prompt": False,
    })
    raw = res.body["completion_probabilities"][0]["top_logprobs"]
    got = _top_probs(temperature=T, smoothing_factor=f, smoothing_curve=curve)
    got_by_id = {t["id"]: t["prob"] for t in got}
    top = raw[0]["logprob"]
    for t in raw[1:]:
        d = (t["logprob"] - top) / T
        expect_ratio = math.exp(-k * f * d * d + s * f * d * d * d)
        assert got_by_id[t["id"]] / got_by_id[raw[0]["id"]] == pytest.approx(expect_ratio, rel=1e-3), t["id"]


def test_smoothing_is_skipped_on_the_greedy_path():
    server.start()
    greedy = dict(temperature=0.0, n_probs=4)
    assert _top_probs(smoothing_factor=2.0, smoothing_curve=2.5, **greedy)[0]["id"] == _top_probs(**greedy)[0]["id"]


def test_smoothing_zero_is_a_no_op():
    server.start()
    assert _top_probs(smoothing_factor=0.0) == _top_probs()


def test_settings_are_echoed_and_limits_enforced():
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time", "n_predict": 1, "top_a": 0.25, "smoothing_factor": 0.5, "smoothing_curve": 1.5,
    })
    assert res.status_code == 200
    gs = res.body["generation_settings"]
    assert gs["top_a"] == pytest.approx(0.25)
    assert gs["smoothing_factor"] == pytest.approx(0.5)
    assert gs["smoothing_curve"] == pytest.approx(1.5)
    assert "top_a" in gs["samplers"]
    bad = server.make_request("POST", "/completion", data={"prompt": "x", "n_predict": 1, "top_a": 1.5})
    assert bad.status_code == 400
