# prompt_logprobs request field (row S69-NEXA-PICK-SERVER-PROMPT-LOGPROB-SCORING-2026-09-29-01).
#
# SOURCE-READ: engines/vllm/vllm/sampling_params.py:289 (the field) and :553 (such a request never reads the prefix cache, so this
# server evaluates every prompt token for it). RESEARCH-DONE: that file, read at those lines. Apache-2.0; reimplemented against this
# server's batch/slot code, no code copied. The response carries, for each prompt token, the log-probability the model gave it
# given the tokens before it, and whether it was the model's top choice. Entry 0 has no context and is null.
#
# Oracle: an independent path. For each k the prefix prompt[:k] is run as its own request with n_probs over the whole vocabulary
# (n_vocab 512), and the logprob of prompt[k] is read from the first generated position's top_logprobs. That path uses the
# ordinary sampling-side probabilities and shares no code with the new field.
import math
import os
import pytest

from utils import ServerPreset, use_local_stories260k  # noqa: E402

server = ServerPreset.tinyllama2()

N_VOCAB = 512

# The field is computed in ONE batched evaluation of the whole prompt; the oracle runs each prefix as its own request. Both come from
# the same server on the same device, but a GPU backend picks different kernels / reduction orders for different batch shapes, so the
# two differ by float noise there. Measured 2026-10-01 on the RTX 5090 CUDA build: got -3.5470667 vs oracle -3.5482233 (diff 1.16e-3
# on a value of -3.55; the CPU build of the same source passes at 1e-3). The failure this test guards (a wrong or stale sub-batch row)
# is an error of order 1, so the GPU tolerance can be 5x looser and still be ~1000x tighter than that signature. Default stays the
# CPU-calibrated 1e-3; set LLAMA_TEST_LOGPROB_ATOL=5e-3 when LLAMA_SERVER_BIN_PATH is a GPU build.
ATOL = float(os.environ.get("LLAMA_TEST_LOGPROB_ATOL", "1e-3"))


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_slots = 2
    server.n_batch = 512


def _tokens(srv, text):
    res = srv.make_request("POST", "/tokenize", data={"content": text, "add_special": True})
    assert res.status_code == 200
    return res.body["tokens"]


def _score(srv, toks, **extra):
    data = {"prompt": toks, "n_predict": 1, "temperature": 0.0, "prompt_logprobs": True}
    data.update(extra)
    return srv.make_request("POST", "/completion", data=data)


def _oracle(srv, toks, k):
    """logprob of toks[k] given toks[:k], from the sampling-side distribution of a one-token generation."""
    res = srv.make_request("POST", "/completion", data={
        "prompt": toks[:k], "n_predict": 1, "temperature": 0.0, "n_probs": N_VOCAB,
        "cache_prompt": False, "post_sampling_probs": False,
    })
    assert res.status_code == 200
    top = res.body["completion_probabilities"][0]["top_logprobs"]
    by_id = {e["id"]: e["logprob"] for e in top}
    assert toks[k] in by_id, "control: full-vocabulary n_probs must list every token"
    best = max(top, key=lambda e: e["logprob"])["id"]
    return by_id[toks[k]], best == toks[k]


def test_prompt_logprobs_match_independent_oracle():
    global server
    server.start()
    toks = _tokens(server, "Once upon a time there was a little girl named Lily who loved to play")
    assert len(toks) >= 12

    res = _score(server, toks)
    assert res.status_code == 200
    pl = res.body["prompt_logprobs"]
    assert len(pl["logprobs"]) == len(toks), "one entry per prompt token"
    assert len(pl["is_greedy"]) == len(toks)
    assert pl["logprobs"][0] is None, "the first token has no context"

    n_greedy_seen = 0
    for k in range(1, len(toks)):
        want, want_greedy = _oracle(server, toks, k)
        got = pl["logprobs"][k]
        assert got is not None and math.isfinite(got)
        assert got <= 0.0
        assert abs(got - want) < ATOL, f"position {k}: got {got}, oracle {want} (atol {ATOL})"
        assert bool(pl["is_greedy"][k]) == want_greedy, f"position {k}: greedy flag"
        n_greedy_seen += int(want_greedy)
    # control: the flag is not trivially all-true or all-false, so it is compared against something that varies
    assert 0 < n_greedy_seen < len(toks) - 1


def test_prompt_logprobs_with_a_warm_prompt_cache():
    """The request must score every token itself; a slot/radix prefix from an earlier request may not leave gaps or stale values."""
    global server
    server.start()
    toks = _tokens(server, "Once upon a time there was a little girl named Lily who loved to play")

    cold = _score(server, toks)
    assert cold.status_code == 200

    warm_up = server.make_request("POST", "/completion", data={
        "prompt": toks, "n_predict": 4, "temperature": 0.0, "cache_prompt": True})
    assert warm_up.status_code == 200
    warm = _score(server, toks, cache_prompt=True)
    assert warm.status_code == 200

    assert warm.body["prompt_logprobs"]["logprobs"] == cold.body["prompt_logprobs"]["logprobs"]
    assert warm.body["prompt_logprobs"]["logprobs"][1:] != [None] * (len(toks) - 1)


def test_prompt_logprobs_longer_than_one_batch():
    """A prompt split across several decode batches still scores every position (rows are read per sub-batch)."""
    global server
    server.n_batch = 32
    server.n_ubatch = 32
    server.start()
    toks = _tokens(server, "Once upon a time there was a little girl named Lily who loved to play. " * 6)
    assert len(toks) > 64, "control: the prompt must need more than one batch"
    res = _score(server, toks)
    assert res.status_code == 200
    lp = res.body["prompt_logprobs"]["logprobs"]
    assert len(lp) == len(toks)
    assert lp[0] is None
    assert all(v is not None and math.isfinite(v) for v in lp[1:])
    # spot-check both sides of a batch boundary against the oracle
    for k in (31, 32, 33, 63, 64, 65):
        want, _ = _oracle(server, toks, k)
        assert abs(lp[k] - want) < ATOL, f"position {k} (atol {ATOL})"


def test_prompt_logprobs_absent_unless_asked():
    global server
    server.start()
    toks = _tokens(server, "Once upon a time")
    res = server.make_request("POST", "/completion", data={"prompt": toks, "n_predict": 1, "temperature": 0.0})
    assert res.status_code == 200
    assert "prompt_logprobs" not in res.body


def test_prompt_logprobs_rejects_n_cmpl_above_one():
    global server
    server.start()
    toks = _tokens(server, "Once upon a time")
    res = _score(server, toks, n_cmpl=2)
    assert res.status_code == 400
    assert "prompt_logprobs" in str(res.body)
