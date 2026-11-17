# POST /v1/score: shared-prefix multi-candidate scoring (row S69-ENGREV-LOCALAI-SHARED-PREFIX-CANDIDATE-SCORING-2026-10-27-01).
#
# Oracle: an independent path. For each candidate one /completion request carries prompt + candidate with prompt_logprobs, and the
# candidate tokens' logprobs are read from that field. The new route decodes the prompt once and each candidate from the prompt
# boundary; the answer must agree, and the work counter must be prompt + sum(candidate tokens), not N * (prompt + candidate).
# Random-weight model, token ids as input (no network).
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tiny_random_model import PROMPT, tiny_server  # noqa: E402

ATOL = float(os.environ.get("LLAMA_TEST_LOGPROB_ATOL", "1e-3"))

CANDS = [[10, 11, 12], [20], [30, 31, 32, 33, 34], [7, 9]]


def _reference(srv, prompt, cand):
    res = srv.make_request("POST", "/completion", data={
        "prompt": prompt + cand, "n_predict": 1, "temperature": 0.0, "prompt_logprobs": True, "cache_prompt": False,
    })
    assert res.status_code == 200, res.body
    return res.body["prompt_logprobs"]["logprobs"][len(prompt):]


def test_score_matches_per_candidate_reference_and_shares_prefix(tmp_path):
    server = tiny_server(tmp_path)
    server.start()
    try:
        res = server.make_request("POST", "/v1/score", data={"prompt": PROMPT, "candidates": CANDS})
        assert res.status_code == 200, res.body
        body = res.body
        assert body["object"] == "score"
        assert len(body["results"]) == len(CANDS)
        for i, c in enumerate(CANDS):
            want = _reference(server, PROMPT, c)
            got = body["results"][i]
            assert got["index"] == i and got["n_tokens"] == len(c)
            assert len(got["token_logprobs"]) == len(c)
            for a, b in zip(got["token_logprobs"], want):
                assert abs(a - b) < ATOL, f"candidate {i}: {a} vs reference {b}"
            assert abs(got["logprob_sum"] - sum(want)) < ATOL * len(c)
        # the prompt is decoded once: prompt + sum(candidate tokens), not N * prompt
        assert body["usage"]["n_prompt_tokens_processed"] == len(PROMPT) + sum(len(c) for c in CANDS)
    finally:
        server.stop()


def test_score_rejects_bad_request(tmp_path):
    server = tiny_server(tmp_path)
    server.start()
    try:
        for bad in [{"prompt": [1, 2]}, {"prompt": [1, 2], "candidates": []}, {"candidates": [[3]]},
                    {"prompt": [1, 2], "candidates": "x"}]:
            res = server.make_request("POST", "/v1/score", data=bad)
            assert res.status_code == 400, (bad, res.status_code, res.body)
    finally:
        server.stop()
