"""Penalty fade (ExLlamaV3 decay range), row S69-EXLLAMAV3-PICK-PENALTY-FADE-2026-10-18-01.

Past the repeat_last_n window the penalty falls linearly to 0 over repeat_decay more tokens
(exllamav3 rep_pen.cu:47-84). Observed through post_sampling_probs with truncation off, so the
penalty is the only thing shaping the list, and through ratios against a token absent from the
prompt (probabilities renormalise, so absolute values move for every token).

A server that ignores repeat_decay returns the window-only numbers, which the strict-inequality
test below refuses.
"""
import pytest
from utils import ServerPreset

server = ServerPreset.tinyllama2()

PROMPT = ("Tom had a dog. It was a sunny morning and the birds were singing in the tall green trees "
          "while the children played happily in the park with their friends and a big red ball. "
          "Tom went home and hugged his")
SUSTAIN = 8
DECAY = 32


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def _probs(**extra) -> dict:
    data = {
        "prompt": PROMPT, "n_predict": 1, "n_probs": 256, "post_sampling_probs": True,
        "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "seed": 42, "cache_prompt": False,
        "repeat_penalty": 1.8, "repeat_last_n": SUSTAIN,
    }
    data.update(extra)
    res = server.make_request("POST", "/completion", data=data)
    assert res.status_code == 200, res.body
    return {t["id"]: t["prob"] for t in res.body["completion_probabilities"][0]["top_probs"]}


def _tokens() -> list:
    res = server.make_request("POST", "/tokenize", data={"content": PROMPT, "add_special": True})
    assert res.status_code == 200
    return res.body["tokens"]


def _nearest_dist(toks: list) -> dict:
    d = {}
    for i, t in enumerate(toks):
        d[t] = len(toks) - i  # later occurrences overwrite: the smallest distance wins
    return d


def test_decay_zero_is_the_hard_window():
    server.start()
    assert _probs(repeat_decay=0) == _probs()


def test_fade_lies_between_the_short_and_the_long_window():
    server.start()
    toks = _tokens()
    near = _nearest_dist(toks)
    short = _probs()                                       # window = SUSTAIN only
    long_ = _probs(repeat_last_n=SUSTAIN + DECAY)          # window = SUSTAIN + DECAY at full strength
    fade = _probs(repeat_decay=DECAY)                      # SUSTAIN full, then fading over DECAY
    ref = next(t for t in sorted(short, key=short.get, reverse=True)
               if t not in set(toks) and t in long_ and t in fade)
    r = lambda d, t: d[t] / d[ref]
    # tokens whose nearest occurrence sits in the fade zone with a factor >= 0.25 (dist SUSTAIN+1 .. SUSTAIN+24)
    zone = [t for t, dist in near.items() if SUSTAIN < dist <= SUSTAIN + 24 and t in short and t in long_ and t in fade]
    assert zone, "premise: no fade-zone token among the candidates"
    for t in zone:
        assert r(fade, t) < r(short, t) * 0.999, (t, near[t])   # penalised, unlike the short window
        assert r(fade, t) > r(long_, t) * 1.001, (t, near[t])   # but less than the full long window
    # negative control: tokens inside the sustain range carry the full penalty in all three
    inner = [t for t, dist in near.items() if dist <= SUSTAIN and t in short and t in fade]
    assert inner, "premise: no sustain-range token among the candidates"
    for t in inner:
        assert r(fade, t) == pytest.approx(r(long_, t), rel=1e-3), t


def test_alias_echo_and_refusals():
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": "x", "n_predict": 1, "repetition_decay": 16})
    assert res.status_code == 200, res.body
    assert res.body["generation_settings"]["repeat_decay"] == 16
    bad = server.make_request("POST", "/completion", data={"prompt": "x", "n_predict": 1, "repeat_decay": -1})
    assert bad.status_code == 400
    both = server.make_request("POST", "/completion", data={"prompt": "x", "n_predict": 1,
                                                            "repeat_decay": 8, "repeat_penalty_slope": 0.5})
    assert both.status_code == 400
