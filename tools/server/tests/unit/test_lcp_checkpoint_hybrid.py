"""Branch-point (LCP) checkpoint for hybrid / recurrent models.

A recurrent state cannot be rolled back to an earlier token, so when a new prompt shares only a PREFIX with the slot's cached
prompt the server restores the nearest checkpoint at or before the shared length, and without one it reprocesses the whole
prompt. The fork made checkpoints only at user-message starts and 4 / 4+n_ubatch tokens before a prompt's end, so a shared
prefix that ends anywhere else (a plain-text prefix, the middle of an assistant turn whose reasoning the next request strips)
was never a restore point.

MEASURED 2026-10-27, LFM2.5-350M-Q8_0 (lfm2, hybrid), CPU, prefix 2400 tokens + divergent 4800-token tails:
  without the branch-point checkpoint (LLAMA_SERVER_LCP_CHECKPOINT=0): request C (same prefix, third tail)
      prompt_n = 7201, cache_n = 0      (the whole prompt again; lcp_fail_before_disabled.txt)
  with it: C prompt_n = 4800, cache_n = 2401 (ideal prompt_n 4800, cache_n 2400)   (lcp_pass_after_numbers.txt)

Ollama reaches the ideal by snapshotting at the divergence offset DURING the request that discovers it
(mlxrunner/prefix_cache.go:131-135 "Schedule a snapshot at the branch point during prefill so future requests diverging here
can restore"), so the request that finds the branch still pays the full prefill and the NEXT one does not. This test pins
exactly that: A (cold), B (shares the prefix, diverges: pays full), C (shares the prefix, diverges: must restore at the branch).

Runs against a LOCAL hybrid GGUF named by LLAMA_TEST_HYBRID_MODEL (os.pathsep separated); skipped without it.
"""
import os

import pytest
from utils import ServerProcess

MODELS = [p for p in os.environ.get("LLAMA_TEST_HYBRID_MODEL", "").split(os.pathsep) if p]
pytestmark = pytest.mark.skipif(not MODELS, reason="set LLAMA_TEST_HYBRID_MODEL to a local hybrid GGUF")

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima"]
PREFIX = " ".join(f"{WORDS[i % 12]}{i}" for i in range(900))                    # plain text, no message structure
TAIL_A = " " + " ".join(f"quick{i} brown{i} fox{i}" for i in range(600))
TAIL_B = " " + " ".join(f"pack{i} my{i} box{i}" for i in range(600))
TAIL_C = " " + " ".join(f"sphinx{i} of{i} quartz{i}" for i in range(600))


def _server(model: str) -> ServerProcess:
    s = ServerProcess()
    s.offline = True
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = model
    s.n_ctx = 8192
    s.n_slots = 1
    s.temperature = 0.0
    s.seed = 1
    return s


def _ntok(s: ServerProcess, text: str) -> int:
    res = s.make_request("POST", "/tokenize", data={"content": text})
    assert res.status_code == 200, res.body
    return len(res.body["tokens"])


def _run(s: ServerProcess, prompt: str):
    res = s.make_request("POST", "/completion", data={
        "prompt": prompt, "n_predict": 8, "temperature": 0, "cache_prompt": True,
    })
    assert res.status_code == 200, res.body
    return res.body


@pytest.mark.parametrize("model", MODELS)
def test_branch_point_checkpoint_restores_shared_prefix(model: str):
    s = _server(model)
    s.start()
    try:
        n_prefix = _ntok(s, PREFIX)
        n_tail_c = _ntok(s, PREFIX + TAIL_C) - n_prefix
        assert n_prefix > 2000 and n_tail_c > 1500

        n_a = _ntok(s, PREFIX + TAIL_A)
        _run(s, PREFIX + TAIL_A)                                                  # cold
        again = _run(s, PREFIX + TAIL_A)                                          # control: exact repeat reuses the end checkpoint
        assert again["timings"]["cache_n"] > 0.9 * n_a
        b = _run(s, PREFIX + TAIL_B)                                              # discovers the branch at n_prefix (pays full)
        c = _run(s, PREFIX + TAIL_C)                                              # must restore the snapshot taken while B prefilled
        ct = c["timings"]
        print(f"[lcp] n_prefix={n_prefix} n_tail_c={n_tail_c} B prompt_n={b['timings']['prompt_n']} cache_n={b['timings']['cache_n']} C prompt_n={ct['prompt_n']} cache_n={ct['cache_n']}")
        assert ct["cache_n"] >= n_prefix - 8, f"branch point not restored: cache_n={ct['cache_n']} of shared prefix {n_prefix}"
        assert ct["prompt_n"] <= n_tail_c + 64, f"reprocessed the prefix: prompt_n={ct['prompt_n']} ideal~{n_tail_c}"
        c_text = c["content"]
    finally:
        s.stop()

    # correctness: the restored state continues exactly like a fresh server that processed C from scratch
    f = _server(model)
    f.start()
    try:
        assert _run(f, PREFIX + TAIL_C)["content"] == c_text
    finally:
        f.stop()
