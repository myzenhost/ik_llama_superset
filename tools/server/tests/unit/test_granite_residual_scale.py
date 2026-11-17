# Dense granite scales each BRANCH by f_residual_scale and then adds the residual. build_llama fused the residual
# add first, so the scale multiplied (branch + residual): granite4.1:8b answered "The capital of France is" with
# " true true true ..." where a reference engine on the same GGUF bytes says
# " Paris. The capital of Germany is Berlin. The capital of Italy is Rome." (MEASURED 2026-09-28).
# Reads a local GGUF (LLAMA_TEST_GRANITE, default: the Ollama-store blob of granite4.1:8b); skips when absent.
import os
import pytest
from utils import ServerProcess

GRANITE = os.environ.get("LLAMA_TEST_GRANITE",
    "models/ollama/blobs/sha256-ed902ac9eb6adce5a90c6a08c8ea201b50e23fdc5976d1cd0362006afac5309e")
REFERENCE = " Paris. The capital of Germany is Berlin. The capital of Italy is Rome."


@pytest.mark.skipif(not os.path.exists(GRANITE), reason="granite GGUF not present (set LLAMA_TEST_GRANITE)")
@pytest.mark.parametrize("fa", ["off", "on"])
def test_granite_greedy_matches_reference(fa):
    s = ServerProcess()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = GRANITE
    s.n_ctx = 2048
    s.n_slots = 1
    s.fa = fa
    s.start(timeout_seconds=600)
    try:
        res = s.make_request("POST", "/completion", data={
            "prompt": "The capital of France is", "temperature": 0.0, "top_k": 1, "n_predict": 16, "n_probs": 1,
        })
        assert res.status_code == 200
        assert res.body["completion_probabilities"][0]["top_logprobs"][0]["token"] == " Paris", res.body
        assert res.body["content"].startswith(REFERENCE[:40]), res.body["content"]
    finally:
        s.stop()
