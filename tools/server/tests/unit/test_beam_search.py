import math

import pytest
from utils import *
from utils import use_local_stories260k  # explicit, so static checkers see where it comes from

# In-band beam search (row S69-VLLM-BEAM-SEARCH-SERVER-PORT-2026-09-27-01). The server's result must equal a
# REFERENCE computed in this file from the same server's plain 1-token requests -- vLLM's loop
# (vllm/entrypoints/generate/beam_search/online.py:28-229 + utils.py:137-154, Apache-2.0, reimplemented from reading).
# A port that ranks, prunes or scores differently fails the equality, not just a plausibility check.

server = ServerPreset.tinyllama2()
PROMPT = "Once upon a time"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 4  # beam width is bounded by n_parallel (n_cmpl's hard limit)
    use_local_stories260k(server)


def _eos_id(srv):
    props = srv.make_request("GET", "/props").body
    ids = srv.make_request("POST", "/tokenize", data={"content": props["eos_token"], "add_special": False,
                                                        "parse_special": True}).body["tokens"]
    assert len(ids) == 1
    return ids[0]


def _reference(srv, width, n_predict, length_penalty=1.0):
    """vLLM's algorithm over the server's RAW next-token distribution (n_probs, post_sampling_probs=false)."""
    eos = _eos_id(srv)
    prompt = srv.make_request("POST", "/tokenize", data={"content": PROMPT, "add_special": True}).body["tokens"]
    live, done = [([], 0.0)], []
    for _ in range(n_predict):
        cands = []
        for gen, cum in live:
            res = srv.make_request("POST", "/completion", data={
                "prompt": prompt + gen, "n_predict": 1, "n_probs": 2 * width, "temperature": 0.0,
                "post_sampling_probs": False})
            assert res.status_code == 200
            for t in res.body["completion_probabilities"][0]["top_logprobs"]:
                lp = cum + t["logprob"]
                if t["id"] == eos:
                    done.append((gen + [t["id"]], lp, True))
                else:
                    cands.append((lp, t["id"], gen))
        cands.sort(key=lambda c: -c[0])
        live = [(gen + [tid], lp) for lp, tid, gen in cands[:width]]
        if not live:
            break
    pool = done + [(g, c, False) for g, c in live]

    def score(h):
        n = len(h[0]) - (1 if h[2] else 0)
        return h[1] / (max(n, 1) ** length_penalty)
    pool.sort(key=lambda h: -score(h))
    return [(h[0][:-1] if h[2] else h[0]) for h in pool[:width]], [h[1] for h in pool[:width]]


def test_beam_width_one_equals_greedy():
    global server
    server.start()
    greedy = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 16, "temperature": 0.0, "return_tokens": True})
    beam = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 16, "use_beam_search": True, "n": 1})
    assert beam.status_code == 200, beam.body
    assert beam.body["beams"][0]["tokens"] == greedy.body["tokens"]
    assert beam.body["content"] == greedy.body["content"]


def test_beam_matches_reference_width3():
    global server
    server.start()
    ref_tokens, ref_cum = _reference(server, 3, 8)
    beam = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 8, "use_beam_search": True, "n": 3})
    assert beam.status_code == 200, beam.body
    got = beam.body["beams"]
    assert [b["tokens"] for b in got] == ref_tokens
    for b, c in zip(got, ref_cum):
        assert math.isclose(b["cumulative_logprob"], c, rel_tol=1e-5, abs_tol=1e-5)
    assert len({tuple(b["tokens"]) for b in got}) == len(got)  # hypotheses are distinct
    scores = [b["score"] for b in got]
    assert scores == sorted(scores, reverse=True)


def test_beam_oai_completions_returns_n_choices():
    global server
    server.start()
    res = server.make_request("POST", "/v1/completions", data={
        "prompt": PROMPT, "max_tokens": 6, "use_beam_search": True, "n": 2})
    assert res.status_code == 200, res.body
    assert res.body["object"] == "text_completion"
    assert len(res.body["choices"]) == 2
    assert all(isinstance(c["text"], str) and c["text"] for c in res.body["choices"])


def test_beam_refuses_stream():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 4, "use_beam_search": True, "n": 2, "stream": True})
    assert res.status_code == 400
    assert "beam search" in res.body["error"]["message"]


def test_beam_refuses_grammar():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 4, "use_beam_search": True, "n": 2, "grammar": 'root ::= "A"'})
    assert res.status_code == 400
    assert "grammar" in res.body["error"]["message"]


def test_no_beam_flag_is_untouched():
    """Negative control: without use_beam_search the normal path answers exactly as before (no 'beams' key)."""
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={"prompt": PROMPT, "n_predict": 4, "temperature": 0.0})
    assert res.status_code == 200
    assert "beams" not in res.body
