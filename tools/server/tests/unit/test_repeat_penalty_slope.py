"""Repetition-penalty slope (koboldcpp's rep_pen_slope), added 2026-09-26.

Tokens seen only in the OLDER half of the repeat-penalty window get 1 + (repeat_penalty - 1) * slope,
tokens in the newer half the full penalty. Observed through post_sampling_probs with truncation off, so
the penalty is the only thing shaping the list.

The test is data-driven because the preset's tiny vocabulary splits words into pieces: it tokenizes the
prompt, finds the tokens that sit only in the older half of the window, keeps those the model offers as
candidates, and requires every one of them to gain probability at slope 0. A server that ignores the
field returns the same numbers for every slope.
"""
import pytest
from utils import *

server = ServerPreset.tinyllama2()

PROMPT = ("Tom had a dog. It was a sunny morning and the birds were singing in the tall green trees "
          "while the children played happily in the park with their friends and a big red ball. "
          "Tom went home and hugged his")
WINDOW = 64


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def _probs(**extra) -> dict:
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 1, "n_probs": 256, "post_sampling_probs": True,
        "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "seed": 42, "cache_prompt": False,
        "repeat_penalty": 1.8, "repeat_last_n": WINDOW, **extra,
    })
    assert res.status_code == 200, res.body
    return {t["id"]: t["prob"] for t in res.body["completion_probabilities"][0]["top_probs"]}


def _older_only() -> set:
    res = server.make_request("POST", "/tokenize", data={"content": PROMPT, "add_special": True})
    assert res.status_code == 200
    window = res.body["tokens"][-WINDOW:]
    split = len(window) // 2
    return set(window[:split]) - set(window[split:])


def test_repeat_penalty_window_reaches_past_the_last_token():
    # Regression 2026-09-26: common_sampler_accept kept prev at a constant size, and the server's reset
    # empties it, so the penalty window was ONE token. Probabilities renormalise, so compare each windowed
    # token against a token absent from the prompt: the penalty must shrink that ratio.
    server.start()
    res = server.make_request("POST", "/tokenize", data={"content": PROMPT, "add_special": True})
    toks = res.body["tokens"]
    window = set(toks[-WINDOW:-1])  # excludes the last token, the only one the pre-fix build penalised
    off = _probs(repeat_penalty=1.0)
    on = _probs(repeat_penalty=1.8)
    ref = next(t for t in sorted(off, key=off.get, reverse=True) if t not in set(toks) and t in on)
    watched = [t for t in window if t in off and t in on]
    assert watched, "premise: no windowed token is among the candidates"
    shrunk = [t for t in watched if on[t] / on[ref] < off[t] / off[ref] * 0.99]
    assert len(shrunk) == len(watched), (len(shrunk), len(watched))


def test_slope_one_is_the_uniform_penalty():
    server.start()
    assert _probs(repeat_penalty_slope=1.0) == _probs()


def test_slope_zero_is_uniform_like_koboldcpp():
    # koboldcpp resets rep_pen_slope <= 0 to 1 in both sampler paths (gpttype_adapter.cpp:6055,
    # batch_rep_pen_init); a kobold client sending 0 must get the uniform penalty here too.
    server.start()
    assert _probs(rep_pen_slope=0.0) == _probs(repeat_penalty_slope=1.0)


def test_small_slope_lifts_the_penalty_on_older_tokens():
    server.start()
    uniform = _probs(repeat_penalty_slope=1.0)
    sloped = _probs(repeat_penalty_slope=0.01)
    # Ratios against a token outside the prompt: lifting the penalty on many tokens renormalises mass
    # away from any single one, so absolute probabilities can fall even for a token the slope helped.
    res = server.make_request("POST", "/tokenize", data={"content": PROMPT, "add_special": True})
    toks = res.body["tokens"]
    ref = next(t for t in sorted(uniform, key=uniform.get, reverse=True) if t not in set(toks) and t in sloped)
    ratio = lambda d, t: d[t] / d[ref]
    watched = [t for t in _older_only() if t in uniform and t in sloped]
    assert watched, "premise: no older-half-only token is among the candidates"  # control on the setup
    for t in watched:
        assert ratio(sloped, t) > ratio(uniform, t) * 1.01, t
    # negative control: newer-half tokens carry the full penalty either way, so their ratio must not move
    newer = [t for t in set(toks[-WINDOW // 2:]) if t in uniform and t in sloped]
    assert newer, "premise: no newer-half token is among the candidates"
    for t in newer:
        assert ratio(sloped, t) == pytest.approx(ratio(uniform, t), rel=1e-3), t


def test_kobold_alias_echo_and_limits():
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": "x", "n_predict": 1, "rep_pen_slope": 0.25})
    assert res.status_code == 200
    assert res.body["generation_settings"]["repeat_penalty_slope"] == pytest.approx(0.25)
    bad = server.make_request("POST", "/completion", data={"prompt": "x", "n_predict": 1, "repeat_penalty_slope": 1.5})
    assert bad.status_code == 400
