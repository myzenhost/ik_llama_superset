import math

import pytest
from utils import *
from utils import use_local_stories260k  # explicit, so static checkers see where it comes from

# Beam search, CTranslate2's loop (beam_mode "ct2"; OpenNMT/CTranslate2 @ 998bb99f, MIT, src/decoding.cc:411-713).
# The server's answer must equal a REFERENCE written here from the algorithm, run over the server's own full raw
# next-token distribution (n_probs = whole vocabulary, post_sampling_probs=false). A loop that ranks, refills, stops,
# masks or biases differently fails the equality of tokens and scores, not a plausibility check.

PROMPT = "Once upon a time"
server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 4
    use_local_stories260k(server)
    # stories260K never emits its real EOS (p ~ e^-23), so no EOS rule would ever run. Make "." (token 426) the end-of-generation
    # token: it ends a sentence often enough that hypotheses finish at different steps.
    server.extra_args = ["--override-kv", "tokenizer.ggml.eos_token_id=int:%d" % DOT]


DOT = 426


class Model:
    """Raw next-token log-probabilities of the running server, cached per token sequence."""

    def __init__(self, srv):
        self.srv = srv
        props = srv.make_request("GET", "/props").body
        self.eos = srv.make_request("POST", "/tokenize", data={
            "content": props["eos_token"], "add_special": False, "parse_special": True}).body["tokens"][-1]
        assert self.eos == DOT, (props["eos_token"], self.eos)
        self.prompt = srv.make_request("POST", "/tokenize", data={"content": PROMPT, "add_special": True}).body["tokens"]
        self.cache = {}

    def tok(self, text):
        return self.srv.make_request("POST", "/tokenize", data={"content": text, "add_special": False}).body["tokens"]

    def dist(self, gen):
        key = tuple(gen)
        if key not in self.cache:
            res = self.srv.make_request("POST", "/completion", data={
                "prompt": self.prompt + list(gen), "n_predict": 1, "n_probs": 4096, "temperature": 0.0,
                "post_sampling_probs": False})
            assert res.status_code == 200, res.body
            self.cache[key] = {t["id"]: t["logprob"] for t in res.body["completion_probabilities"][0]["top_logprobs"]}
        return self.cache[key]


def _masked(d, eos):
    """EOS logit = -inf before the softmax: renormalise the rest (src/decoding.cc:384-408, :494-505)."""
    keep = 1.0 - math.exp(d.get(eos, -math.inf))
    return {t: lp - math.log(keep) for t, lp in d.items() if t != eos}


def ref_ct2(m, width, max_len, length_penalty=0.0, patience=1.0, min_len=0, num_hyps=None, prefix=None, beta=0.0):
    """CTranslate2's BeamSearch::search for one sentence, written from the algorithm."""
    num_hyps = num_hyps or width
    prefix = prefix or []
    hard = bool(prefix) and beta == 0
    soft = bool(prefix) and beta > 0
    beams = [([], 0.0, False)]          # tokens, cum, diverged
    start = 0
    if hard:
        beams = [(list(prefix), 0.0, False)]
        start = len(prefix)
    hyps, top_done = [], False
    for step in range(start, max_len):
        last = step + 1 == max_len
        flat = []
        for bi, (toks, cum, div) in enumerate(beams):
            d = dict(m.dist(toks))
            if step < min_len:
                d = _masked(d, m.eos)
            if soft and step < len(prefix) and not div:
                p = prefix[step]
                pp = math.exp(d.get(p, -math.inf))
                d = {t: lp + math.log1p(-beta) if beta < 1 else -math.inf for t, lp in d.items()}
                d[p] = math.log((1 - beta) * pp + beta)
            flat += [(cum + lp, t, bi) for t, lp in d.items() if lp > -math.inf]
        flat.sort(key=lambda c: -c[0])
        cands = flat[:2 * width]
        if hard and step == start:
            cands = [c for k, c in enumerate(cands) if k == 0 or c[1] != m.eos]
        nxt, sec = [], width
        for k in range(min(width, len(cands))):
            pick = k
            if cands[k][1] == m.eos or last:
                if k == 0:
                    top_done = True
                toks, _, _ = beams[cands[k][2]]
                hyps.append((toks + [cands[k][1]], cands[k][0], cands[k][1] == m.eos))
                for j in range(sec, len(cands)):
                    if cands[j][1] != m.eos:
                        pick, sec = j, j + 1
                        break
            sc, t, bi = cands[pick]
            toks, _, div = beams[bi]
            nd = div or step >= len(prefix) or t != prefix[step] if soft else False
            nxt.append((toks + [t], sc, nd))
        if last:
            break
        if length_penalty == 0:
            if top_done and len(hyps) >= num_hyps:
                break
        elif len(hyps) >= round(width * patience):
            break
        beams = nxt
        if not beams:
            break
    scored = sorted(((c / (len(h) ** length_penalty), h, c, e) for h, c, e in hyps), key=lambda x: -x[0])
    return scored[:num_hyps]


def _post(srv, **kw):
    data = {"prompt": PROMPT, "use_beam_search": True, "beam_mode": "ct2"}
    data.update(kw)
    return srv.make_request("POST", "/completion", data=data)


def _same(got, ref, tol=5e-3):  # beams run batched on the server, one at a time in the reference: float noise accumulates
    assert len(got) == len(ref)
    for b, (score, h, cum, eos) in zip(got, ref):
        assert math.isclose(b["cumulative_logprob"], cum, rel_tol=5e-4, abs_tol=tol), (b, cum)
        assert math.isclose(b["score"], score, rel_tol=5e-4, abs_tol=tol), (b, score)
        assert b["tokens"] == (h[:-1] if eos else h)
        assert b["finish_reason"] == ("stop" if eos else "length")


@pytest.mark.parametrize("width,lp,patience", [(3, 0.0, 1.0), (3, 1.0, 1.0), (3, 1.0, 2.0), (4, 0.6, 1.5)])
def test_ct2_matches_reference(width, lp, patience):
    global server
    server.start()
    m = Model(server)
    res = _post(server, n_predict=14, n=width, length_penalty=lp, beam_patience=patience)
    assert res.status_code == 200, res.body
    _same(res.body["beams"], ref_ct2(m, width, 14, lp, patience))
    assert res.body["beam_search"]["mode"] == "ct2"


def test_patience_changes_the_search_and_early_stop_fires():
    global server
    server.start()
    a = _post(server, n_predict=24, n=3, length_penalty=1.0, beam_patience=1.0).body["beam_search"]
    b = _post(server, n_predict=24, n=3, length_penalty=1.0, beam_patience=3.0).body["beam_search"]
    assert a["early_stop"] and a["steps"] < 24
    assert a["registered"] == 3 and b["registered"] > a["registered"]   # patience 3 keeps going past the first width hypotheses
    assert a["steps"] < b["steps"]
    c = _post(server, n_predict=24, n=3, length_penalty=0.0).body["beam_search"]
    assert c["early_stop"] and c["steps"] < 24


def test_min_tokens_ct2_matches_reference_and_fires():
    global server
    server.start()
    m = Model(server)
    plain = _post(server, n_predict=16, n=3, length_penalty=1.0)
    masked = _post(server, n_predict=16, n=3, length_penalty=1.0, min_tokens=13)
    assert masked.status_code == 200, masked.body
    assert masked.body["beam_search"]["min_masked"] == 13
    # CONTROL: without min_tokens a hypothesis ends before step 9, so the mask really changes the search
    assert any(b["finish_reason"] == "stop" and len(b["tokens"]) < 13 for b in plain.body["beams"]), plain.body["beams"]
    assert all(len(b["tokens"]) >= 13 for b in masked.body["beams"])
    assert plain.body["beam_search"]["min_masked"] == 0
    _same(masked.body["beams"], ref_ct2(m, 3, 16, 1.0, 1.0, min_len=13), tol=8e-3)  # renormalising by 1 - p(EOS) amplifies float noise
    for b in masked.body["beams"]:
        assert len(b["tokens"]) >= 13 or b["finish_reason"] == "length"


def ref_vllm(m, width, max_len, length_penalty, min_len):
    """vLLM's loop (every EOS candidate of the 2*width best finishes; run to max_len) with min_tokens applied as CT2 does."""
    live, done = [([], 0.0)], []
    for step in range(max_len):
        cands = []
        for gen, cum in live:
            d = m.dist(gen)
            if step < min_len:
                d = _masked(d, m.eos)
            for t, lp in sorted(d.items(), key=lambda kv: -kv[1])[:2 * width]:
                if t == m.eos:
                    done.append((gen + [t], cum + lp, True))
                else:
                    cands.append((cum + lp, t, gen))
        cands.sort(key=lambda c: -c[0])
        live = [(gen + [t], lp) for lp, t, gen in cands[:width]]
        if not live:
            break
    pool = done + [(g, c, False) for g, c in live]
    score = lambda h: h[1] / (max(len(h[0]) - (1 if h[2] else 0), 1) ** length_penalty)
    pool.sort(key=lambda h: -score(h))
    return [(score(h), h[0], h[1], h[2]) for h in pool[:width]]


def test_min_tokens_vllm_mode_is_honoured_not_an_error():
    """Before: min_tokens > 1 made every 1-token step fail ('min_tokens must be <= n_predict'), min_tokens == 1 was ignored."""
    global server
    server.start()
    m = Model(server)
    plain = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 16, "use_beam_search": True, "n": 2, "min_tokens": 0})
    assert plain.status_code == 200
    assert "beam_search" not in plain.body   # the vllm reply keeps its old shape
    assert any(b["finish_reason"] == "stop" and len(b["tokens"]) < 13 for b in plain.body["beams"])   # CONTROL
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 16, "use_beam_search": True, "n": 2, "min_tokens": 13})
    assert res.status_code == 200, res.body
    assert all(len(b["tokens"]) >= 13 for b in res.body["beams"])
    ref = ref_vllm(m, 2, 16, 1.0, 13)
    for b, (sc, h, cum, eos) in zip(res.body["beams"], ref):
        assert b["tokens"] == (h[:-1] if eos else h)
        assert math.isclose(b["cumulative_logprob"], cum, rel_tol=5e-4, abs_tol=8e-3)


def test_soft_prefix_bias_matches_reference_and_fires():
    global server
    server.start()
    m = Model(server)
    prefix = m.tok(" there was a big")
    for beta in (0.3, 0.9):
        res = _post(server, n_predict=12, n=3, length_penalty=1.0, beam_prefix=" there was a big", beam_prefix_bias_beta=beta)
        assert res.status_code == 200, res.body
        _same(res.body["beams"], ref_ct2(m, 3, 12, 1.0, 1.0, prefix=prefix, beta=beta))
        assert res.body["beam_search"]["bias_steps"] >= 1
    nobias = _post(server, n_predict=12, n=3, length_penalty=1.0)
    assert nobias.body["beam_search"]["bias_steps"] == 0
    assert nobias.body["beams"][0]["tokens"][:len(prefix)] != prefix   # CONTROL: the prefix is not what the model does unprompted
    hi = _post(server, n_predict=12, n=1, length_penalty=1.0, beam_prefix=" there was a big", beam_prefix_bias_beta=1.0)
    assert hi.body["beams"][0]["tokens"][:len(prefix)] == prefix   # beta 1 follows the prefix exactly


def test_hard_prefix_matches_reference_and_fires():
    global server
    server.start()
    m = Model(server)
    prefix = m.tok(" there was a big")
    res = _post(server, n_predict=12, n=3, length_penalty=1.0, beam_prefix=" there was a big")
    assert res.status_code == 200, res.body
    _same(res.body["beams"], ref_ct2(m, 3, 12, 1.0, 1.0, prefix=prefix, beta=0.0))
    assert res.body["beam_search"]["prefix_forced"] == len(prefix)
    assert all(b["tokens"][:len(prefix)] == prefix for b in res.body["beams"])


def test_cli_flag_beam_mode_reaches_the_loop():
    global server
    server.start()
    default = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 8, "use_beam_search": True, "n": 2})
    assert "beam_search" not in default.body
    server.stop()
    server = ServerPreset.tinyllama2()
    server.n_slots = 4
    use_local_stories260k(server)
    server.extra_args = ["--override-kv", "tokenizer.ggml.eos_token_id=int:%d" % DOT, "--beam-mode", "ct2"]
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 8, "use_beam_search": True, "n": 2})
    assert res.status_code == 200, res.body
    assert res.body["beam_search"]["mode"] == "ct2"
    back = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 8, "use_beam_search": True, "n": 2, "beam_mode": "vllm"})
    assert "beam_search" not in back.body


def test_ct2_refusals():
    global server
    server.start()
    for kw in ({"beam_patience": 0}, {"beam_prefix_bias_beta": 1.5}, {"beam_num_hypotheses": 9}, {"beam_mode": "x"}):
        res = _post(server, n_predict=4, n=2, **kw)
        assert res.status_code == 400, (kw, res.body)
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 4, "use_beam_search": True, "n": 2, "beam_patience": 2.0})
    assert res.status_code == 400   # ct2-only field in vllm mode
