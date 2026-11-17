# qwen35 GGUFs written by Ollama's converter must load and match the Ollama engine on the same bytes.
# Three differences from llama.cpp-converted files, each handled in the loader (none changes a llama.cpp-named file):
#   * dt bias is `blk.N.ssm_dt`, not `blk.N.ssm_dt.bias`                      -> name alias in get_weight()
#   * attention.head_count_kv is a PER-LAYER array, 0 on linear-attention layers -> per-layer kv width at tensor creation
#   * the MTP head is `mtp.*`, not `blk.N.nextn.*`                              -> left unused, counted and logged
# SOURCE-READ: upstream/ollama-v0.24.0-read/convert_convert_qwen3next.go:316-318,443-478,921-930 and
#              upstream/llama.cpp/conversion/qwen.py:455-476,569-616 (same grouped->tiled V reorder).
# Reference: the Ollama engine (v0.24.0), raw prompt, greedy, same blob (MEASURED 2026-09-28).
# Control: an upstream-converted qwen35 file must still load (its logits were BIT-IDENTICAL before/after this change).
# Local GGUFs only (LLAMA_TEST_QWEN36_OLLAMA / LLAMA_TEST_QWEN35_UPSTREAM); each case skips when its file is absent.
import os
import pytest
from utils import ServerProcess

OLLAMA_QWEN36 = os.environ.get("LLAMA_TEST_QWEN36_OLLAMA",
    "models/ollama/blobs/sha256-83c54730a5fea8a0958598c01617c1419c431e93b33bacf980b49a420c798926")
UPSTREAM_QWEN35 = os.environ.get("LLAMA_TEST_QWEN35_UPSTREAM",
    "models/hf/ggml-org_Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf")


def _greedy(path, n_predict=16):
    s = ServerProcess()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = path
    s.n_ctx = 2048
    s.n_slots = 1
    s.fa = "on"
    s.start(timeout_seconds=1800)
    try:
        res = s.make_request("POST", "/completion", data={
            "prompt": "The capital of France is", "temperature": 0.0, "top_k": 1, "n_predict": n_predict, "n_probs": 1,
        })
        assert res.status_code == 200, res.body
        return res.body
    finally:
        s.stop()


@pytest.mark.skipif(not os.path.exists(OLLAMA_QWEN36), reason="Ollama qwen3.6 GGUF not present (set LLAMA_TEST_QWEN36_OLLAMA)")
def test_ollama_qwen36_loads_and_matches_ollama():
    body = _greedy(OLLAMA_QWEN36)
    assert body["content"].startswith(" Paris.\n\n<think>\nHere's a thinking process:"), body["content"]


@pytest.mark.skipif(not os.path.exists(UPSTREAM_QWEN35), reason="upstream qwen35 GGUF not present (set LLAMA_TEST_QWEN35_UPSTREAM)")
def test_upstream_qwen35_still_loads():
    body = _greedy(UPSTREAM_QWEN35, n_predict=8)
    assert body["completion_probabilities"][0]["top_logprobs"][0]["token"] == " Paris", body
