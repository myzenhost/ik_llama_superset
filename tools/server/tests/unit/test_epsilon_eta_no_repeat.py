"""epsilon / eta truncation and no-repeat n-gram, added 2026-09-28 from the Aphrodite sampler review.

Epsilon and eta (Hewitt et al. 2022) act on the candidate list, so they are observed through
post_sampling_probs with every other truncation off. The test model's vocabulary is 512 tokens, so
n_probs=512 returns the WHOLE distribution and the entropy eta needs can be computed exactly here.
No-repeat n-gram is observed on greedy output: the control run must repeat an n-gram, or the
assertion on the treated run would prove nothing.
"""
import math

import pytest
from utils import ServerPreset

server = ServerPreset.tinyllama2()

N_VOCAB = 512


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def _probs(n_probs=N_VOCAB, prompt="Once upon a time", **extra) -> list[dict]:
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt, "n_predict": 1, "n_probs": n_probs, "post_sampling_probs": True,
        "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "seed": 42, "cache_prompt": False,
        **extra,
    })
    assert res.status_code == 200, res.body
    return res.body["completion_probabilities"][0]["top_probs"]


def _kept_ids(base, thr, rel=1e-3):
    """ids the reference keeps (p >= thr), plus the ids too close to thr for float noise to decide"""
    keep = {t["id"] for t in base if t["prob"] >= thr * (1 + rel)}
    edge = {t["id"] for t in base if abs(t["prob"] - thr) <= thr * rel}
    return keep, edge


def test_epsilon_cutoff_drops_tokens_below_the_fixed_floor():
    server.start()
    base = _probs(n_probs=64)
    assert len(base) == 64  # control: nothing truncates without the sampler
    eps = (base[4]["prob"] + base[5]["prob"]) / 2
    got = _probs(n_probs=64, epsilon_cutoff=eps)
    assert [t["id"] for t in got] == [t["id"] for t in base[:5]]


# Candidate prompts for a FLAT next-token distribution. MEASURED 2026-09-28: after "Once upon a time" the entropy is
# only 0.157 nats, too peaked for the entropy branch of eta to decide anything but the top token.
FLAT_PROMPTS = ["Once upon a time, there was a little girl", "One day, a", "The", "She saw a", "Once upon a time"]


def _flattest():
    best = None
    for pr in FLAT_PROMPTS:
        base = _probs(prompt=pr)
        assert sum(t["prob"] for t in base) == pytest.approx(1.0, abs=1e-3), "n_probs did not return the whole distribution"
        H = -sum(t["prob"] * math.log(t["prob"]) for t in base if t["prob"] > 0)
        if best is None or H > best[2]:
            best = (pr, base, H)
    return best


def test_eta_cutoff_uses_the_entropy_of_the_whole_distribution():
    server.start()
    prompt, base, H = _flattest()
    assert H > 0.7, f"no candidate prompt is flat enough (best {prompt!r}, H={H:.3f})"
    ps = [t["prob"] for t in base]  # descending
    # Pick the threshold t the ENTROPY branch should produce, midway between two listed probabilities, and solve
    # sqrt(eta) * exp(-H) = t for eta. t > exp(-2H) makes eta > t, so min() takes the entropy branch; the pick also
    # requires a fixed floor at eta (epsilon) to keep a DIFFERENT list, so the assertion can tell the branches apart.
    chosen = None
    for k in range(len(ps) - 1):
        t = (ps[k] + ps[k + 1]) / 2
        eta = t * t * math.exp(2 * H)
        if t > math.exp(-2 * H) * 1.05 and eta < 0.99 and any(t <= p < eta for p in ps):
            chosen = (t, eta)
            break
    assert chosen, f"no usable threshold for prompt {prompt!r} (H={H:.3f}, top probs {ps[:6]})"
    t, eta = chosen
    got = {x["id"] for x in _probs(prompt=prompt, eta_cutoff=eta)}
    keep, edge = _kept_ids(base, t)
    assert keep <= got, (prompt, H, eta, t)
    assert got - keep <= edge, (prompt, H, eta, t)
    fixed = {x["id"] for x in _probs(prompt=prompt, epsilon_cutoff=eta)}
    assert got != fixed, "eta kept the same list as a fixed floor at eta: the entropy branch did not act"
    # small eta: min() takes eta itself when sqrt(eta)*exp(-H) >= eta, i.e. eta <= exp(-2H); check that branch too
    eta2 = math.exp(-2 * H) * 0.5
    thr2 = min(eta2, math.sqrt(eta2) * math.exp(-H))
    got2 = {x["id"] for x in _probs(prompt=prompt, eta_cutoff=eta2)}
    keep2, edge2 = _kept_ids(base, thr2)
    assert keep2 <= got2 and got2 - keep2 <= edge2, (prompt, H, eta2, thr2)


def test_cutoffs_at_zero_are_no_ops():
    server.start()
    base = _probs(n_probs=64)
    assert _probs(n_probs=64, epsilon_cutoff=0.0) == base
    assert _probs(n_probs=64, eta_cutoff=0.0) == base


def test_cutoff_keeps_the_top_token_when_every_token_is_below_it():
    server.start()
    base = _probs(n_probs=4)
    got = _probs(n_probs=4, epsilon_cutoff=0.999)
    assert [t["id"] for t in got] == [base[0]["id"]]


def _greedy(n_predict=200, **extra):  # MEASURED: 64 greedy tokens never repeat a 3-gram here, so the control needs 200
    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time", "n_predict": n_predict, "temperature": 0.0, "seed": 42,
        "cache_prompt": False, "return_tokens": True, **extra,
    })
    assert res.status_code == 200, res.body
    return res.body["tokens"]


def _prompt_tokens():
    res = server.make_request("POST", "/tokenize", data={"content": "Once upon a time"})
    assert res.status_code == 200
    return res.body["tokens"]


def _repeated_ngram_ends(hist, n, first_generated):
    """positions >= first_generated whose n-gram (ending there) already occurred earlier in hist"""
    bad = []
    for j in range(max(first_generated, n - 1), len(hist)):
        gram = tuple(hist[j - n + 1: j + 1])
        for s in range(0, j - n + 1):
            if tuple(hist[s: s + n]) == gram:
                bad.append(j)
                break
    return bad


@pytest.mark.parametrize("n", [2, 3])
def test_no_repeat_ngram_bans_every_repeat_the_control_makes(n):
    server.start()
    prompt = _prompt_tokens()
    control = prompt + _greedy()
    assert _repeated_ngram_ends(control, n, len(prompt)), "control never repeats -- test has no teeth"
    treated = prompt + _greedy(no_repeat_ngram_size=n)
    assert len(treated) > len(prompt) + 8
    assert _repeated_ngram_ends(treated, n, len(prompt)) == []


def test_no_repeat_ngram_size_one_never_reuses_a_token():
    server.start()
    prompt = _prompt_tokens()
    out = _greedy(n_predict=24, no_repeat_ngram_size=1)
    seen = set(prompt)
    for t in out:
        assert t not in seen
        seen.add(t)


def test_no_repeat_ngram_window_limits_the_search():
    # n = 2 over the last 6 tokens: at every generated position j, the bigram (hist[j-1], hist[j]) must not occur
    # INSIDE hist[j-6 : j] -- and the run must differ from the unwindowed one, or the window changed nothing.
    server.start()
    n, last_n = 2, 6
    prompt = _prompt_tokens()
    hist = prompt + _greedy(no_repeat_ngram_size=n, no_repeat_ngram_last_n=last_n)
    for j in range(len(prompt), len(hist)):
        win = hist[max(0, j - last_n): j]
        grams = {tuple(win[s: s + n]) for s in range(len(win) - n + 1)}
        assert (hist[j - 1], hist[j]) not in grams, j
    assert hist != prompt + _greedy(no_repeat_ngram_size=n), "the window made no difference"


def test_no_repeat_ngram_zero_is_a_no_op():
    server.start()
    assert _greedy(no_repeat_ngram_size=0) == _greedy()


def test_settings_are_echoed_and_limits_enforced():
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time", "n_predict": 1, "epsilon_cutoff": 0.001, "eta_cutoff": 0.002,
        "no_repeat_ngram_size": 3, "no_repeat_ngram_last_n": 128,
    })
    assert res.status_code == 200
    gs = res.body["generation_settings"]
    assert gs["epsilon_cutoff"] == pytest.approx(0.001)
    assert gs["eta_cutoff"] == pytest.approx(0.002)
    assert gs["no_repeat_ngram_size"] == 3
    assert gs["no_repeat_ngram_last_n"] == 128
    assert "epsilon_cutoff" in gs["samplers"] and "eta_cutoff" in gs["samplers"]
    for bad in ({"epsilon_cutoff": 3.0}, {"eta_cutoff": -0.1}, {"no_repeat_ngram_size": -1}):
        r = server.make_request("POST", "/completion", data={"prompt": "x", "n_predict": 1, **bad})
        assert r.status_code == 400, bad
