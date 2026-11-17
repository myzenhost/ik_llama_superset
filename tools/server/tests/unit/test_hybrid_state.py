"""Recurrent-state isolation for hybrid models (conv / SSM layers + attention).

Regression for 2026-09-26: the legacy builders for lfm2 and granitehybrid read and wrote ONE
recurrent-state row for every sequence and never zeroed it, so under llama-server the second
request on a slot continued from the first request's state ("Water boils at a temperature of" ->
"a"), and with --parallel > 1 all slots shared it. llama-cli / llama-perplexity were unaffected
(one run, or memory_clear with data), which is why only the server showed it.

The ggml-org test presets carry no hybrid model, so this module runs against a LOCAL GGUF named by
LLAMA_TEST_HYBRID_MODEL (one or more paths separated by os.pathsep) and is skipped without it.
Measured with LFM2.5-350M-Q8_0 and granite-4.0-h-350m-Q8_0.
"""
import os
# DETERMINISM-SETUP: temperature 0, seed 1, cache_prompt false on every request, one slot per concurrent prompt; references come from fresh single-prompt servers.
# NULL-RUN-CONTROL: test_hybrid_state_does_not_leak_between_requests (same references, sequential, no concurrency) must PASS before the parallel test is trusted.
import threading

import pytest
from utils import *

MODELS = [p for p in os.environ.get("LLAMA_TEST_HYBRID_MODEL", "").split(os.pathsep) if p]
pytestmark = pytest.mark.skipif(not MODELS, reason="set LLAMA_TEST_HYBRID_MODEL to a local hybrid GGUF")

PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The largest planet in the solar system is",
    "Photosynthesis converts sunlight into",
]
N_PREDICT = 24

server: ServerProcess


def _server(model: str, n_slots: int) -> ServerProcess:
    s = ServerProcess()
    s.offline = True
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = model
    s.n_ctx = 1024 * n_slots
    s.n_slots = n_slots
    s.temperature = 0.0
    s.seed = 1
    return s


def _complete(s: ServerProcess, prompt: str) -> str:
    res = s.make_request("POST", "/completion", data={
        "prompt": prompt, "n_predict": N_PREDICT, "temperature": 0, "cache_prompt": False,
    })
    assert res.status_code == 200, res.body
    return res.body["content"]


def _fresh_answers(model: str) -> list[str]:
    """each prompt on its own freshly started server: no state can carry over"""
    answers = []
    for p in PROMPTS:
        s = _server(model, 1)
        s.start()
        try:
            answers.append(_complete(s, p))
        finally:
            s.stop()
    return answers


@pytest.mark.parametrize("model", MODELS)
def test_hybrid_state_does_not_leak_between_requests(model: str):
    ref = _fresh_answers(model)
    s = _server(model, 1)
    s.start()
    try:
        got = [_complete(s, p) for p in PROMPTS]  # same slot, one after another
    finally:
        s.stop()
    assert got == ref


@pytest.mark.parametrize("model", MODELS)
def test_hybrid_state_is_per_slot_under_parallel_requests(model: str):
    ref = _fresh_answers(model)
    s = _server(model, len(PROMPTS))
    s.start()
    try:
        got: list[str | None] = [None] * len(PROMPTS)

        def work(i: int):
            got[i] = _complete(s, PROMPTS[i])

        threads = [threading.Thread(target=work, args=(i,)) for i in range(len(PROMPTS))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        s.stop()
    assert got == ref
