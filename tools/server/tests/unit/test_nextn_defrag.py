"""KV defrag on a model that carries a nextn (MTP) head it is not using.

Regression for 2026-09-26: without an MTP context the KV cache holds n_layer - nextn_predict_layers
layers, but llm_build_context::build_defrag looped over hparams.n_layer and read k_l[k_l.size()]
past the end of the vector. Whatever sat there was dereferenced as a tensor: qwen35
(Qwen3.8-27B, n_layer 65, k_l 64) died with 0xC0000005 "reading address FFFFFFFFFFFFFFFF" at
build_defrag on the first defrag after a --parallel 4 batch -- 4 of 5 runs, and 2 of 2 with
--ctx-checkpoints 0. The read is undefined behaviour, so a single pass proves little; the test
repeats the batch-then-sequential cycle, which is what fragments the cache and queues a defrag.

No ggml-org preset carries a nextn model, so this runs against a LOCAL GGUF named by
LLAMA_TEST_NEXTN_MODEL and is skipped without it. N_GPU_LAYERS is honoured (utils.py).
"""
import os
import threading

import pytest
from utils import *

MODEL = os.environ.get("LLAMA_TEST_NEXTN_MODEL", "")
pytestmark = pytest.mark.skipif(not MODEL, reason="set LLAMA_TEST_NEXTN_MODEL to a local GGUF with nextn layers")

PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The largest planet in the solar system is",
    "Photosynthesis converts sunlight into",
]
N_PREDICT = 32
CYCLES = 3


def _complete(s: ServerProcess, prompt: str) -> str:
    res = s.make_request("POST", "/completion", data={
        "prompt": prompt, "n_predict": N_PREDICT, "temperature": 0, "cache_prompt": False,
    }, timeout=900)
    assert res.status_code == 200, res.body
    return res.body["content"]


def test_defrag_after_parallel_batch_does_not_read_past_the_cache():
    s = ServerProcess()
    s.offline = True
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = MODEL
    s.n_slots = len(PROMPTS)
    s.n_ctx = 2048 * len(PROMPTS)
    s.temperature = 0.0
    s.seed = 1
    s.start(timeout_seconds=900)
    try:
        for _ in range(CYCLES):
            got: list[str | None] = [None] * len(PROMPTS)

            def work(i: int):
                got[i] = _complete(s, PROMPTS[i])

            threads = [threading.Thread(target=work, args=(i,)) for i in range(len(PROMPTS))]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert all(g is not None for g in got)
            # one at a time on the same server: the slot relaunch after the batch is what
            # ran the out-of-range defrag
            for p in PROMPTS:
                _complete(s, p)
            assert s.process is not None and s.process.poll() is None, "server died"
    finally:
        s.stop()
