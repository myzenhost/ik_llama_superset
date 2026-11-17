# Flash attention on and off must produce the same greedy text.
#
# The two modes store V differently: with FA the V cache is written row-major through a 1-D view; without FA
# it is written TRANSPOSED into a [n_tokens, n_embd_v_gqa] view. llm_build_kv_store transposed v_cur as it
# arrived, and builders that RMS-norm V per head (gemma3n/gemma4, ...) hand it over 3-D
# [n_embd_head_v, n_head_kv, n_tokens] -- ggml_transpose swaps only dims 0 and 1, so every element landed in
# the wrong cell. gemma-4-E2B answered "The capital of France is" with " Paris." under -fa on and with
# "of\n\text{ in ..." under -fa off, on CPU and CUDA alike. Comparing the two modes catches any such layout
# defect without needing a reference engine.
import os
import pytest
from utils import ServerPreset, ServerProcess

PROMPT = "The capital of France is"
GEMMA4_E2B = os.environ.get("LLAMA_TEST_GEMMA4_E2B",
                            "models/hf/gemma-4-E2B-it-Q4_0-GGUF/gemma-4-E2B-it-Q4_0.gguf")


def _greedy(server: ServerProcess, fa: str, n_predict: int = 16):
    server.fa = fa
    server.start()
    try:
        res = server.make_request("POST", "/completion", data={
            "prompt": PROMPT, "temperature": 0.0, "top_k": 1, "n_predict": n_predict, "n_probs": 1,
        })
        assert res.status_code == 200
        first = res.body["completion_probabilities"][0]["top_logprobs"][0]
        return res.body["content"], first["token"], first["logprob"]
    finally:
        server.stop()


@pytest.mark.parametrize("preset", ["tinygemma3", "stories15m_moe"])
def test_fa_on_off_same_greedy_text(preset):
    on = _greedy(getattr(ServerPreset, preset)(), "on")
    off = _greedy(getattr(ServerPreset, preset)(), "off")
    assert on[1] == off[1], (on, off)
    assert abs(on[2] - off[2]) < 0.05, (on, off)
    assert on[0] == off[0], (on, off)


@pytest.mark.skipif(not os.path.exists(GEMMA4_E2B), reason="gemma-4-E2B GGUF not present (set LLAMA_TEST_GEMMA4_E2B)")
def test_gemma4_e2b_answers_paris_with_and_without_fa():
    def make():
        s = ServerProcess()
        s.model_hf_repo = None
        s.model_hf_file = None
        s.model_file = GEMMA4_E2B
        s.n_ctx = 2048
        s.n_slots = 1
        return s
    on = _greedy(make(), "on", n_predict=4)
    off = _greedy(make(), "off", n_predict=4)
    assert on[1] == " Paris", on
    assert off[1] == " Paris", off
    assert abs(on[2] - off[2]) < 0.05, (on, off)
