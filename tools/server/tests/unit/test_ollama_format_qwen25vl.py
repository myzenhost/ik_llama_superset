# Ollama writes Qwen2.5-VL as arch "qwen25vl": the qwen2vl text model plus the vision tower in ONE file, every key under
# "qwen25vl.*", and the M-RoPE sections as rope.mrope_section. The fork now resolves the name (llm_arch_from_ollama_name),
# reads keys under the FILE's prefix (LLM_KV::prefix), and falls back to rope.mrope_section.
# SOURCE-READ: upstream/ollama-v0.24.0-read/convert_convert_qwen25vl.go:32-40 and
#              upstream/ollama-v0.24.0-read/model_models_qwen25vl_model_text.go:20-22,34-58,69-89
# Reference: the Ollama engine (v0.24.0), raw prompt, same blob, next-token top-5 (MEASURED 2026-09-29).
# Local GGUF only (LLAMA_TEST_QWEN25VL_OLLAMA); skips when absent.
import os
import pytest
from utils import ServerProcess

QWEN25VL = os.environ.get("LLAMA_TEST_QWEN25VL_OLLAMA",
    "models/ollama/blobs/sha256-a99b7f834d754b88f122d865f32758ba9f0994a83f8363df2c1e71c17605a025")
OLLAMA_NEXT = {" Paris": -0.5603, " ______": -2.8791, " __": -3.0325, " ____": -3.4143, ":\n": -3.7143}


@pytest.mark.skipif(not os.path.exists(QWEN25VL), reason="Ollama qwen2.5vl GGUF not present (set LLAMA_TEST_QWEN25VL_OLLAMA)")
def test_ollama_qwen25vl_loads_and_matches_ollama_logprobs():
    s = ServerProcess()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = QWEN25VL
    s.n_ctx = 2048
    s.n_slots = 1
    s.fa = "on"
    s.start(timeout_seconds=900)
    try:
        res = s.make_request("POST", "/completion", data={
            "prompt": "The capital of France is", "temperature": 0.0, "top_k": 1, "n_predict": 1, "n_probs": 5,
        })
        assert res.status_code == 200, res.body
        got = {t["token"]: t["logprob"] for t in res.body["completion_probabilities"][0]["top_logprobs"]}
        shared = [k for k in OLLAMA_NEXT if k in got]
        assert len(shared) >= 4, (got, OLLAMA_NEXT)
        worst = max(shared, key=lambda k: abs(got[k] - OLLAMA_NEXT[k]))
        assert abs(got[worst] - OLLAMA_NEXT[worst]) < 0.5, f"{worst}: ours {got[worst]:.4f} vs Ollama {OLLAMA_NEXT[worst]:.4f}"
    finally:
        s.stop()
