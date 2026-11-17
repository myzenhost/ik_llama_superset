# deepseek-v2 (MLA) on the host backend, both attention paths.
#   -fa on : the MLA prompt path multiplies the quantized wkv_b by an F16 KV-cache view; the split host backend
#            refused that MUL_MAT and the scheduler aborted at context creation (fixed in ggml-cpu, c72ce7272).
#   -fa off: the params-fit probe loads with no_alloc, which returned before llm_prepare_mla, left wk_b NULL, and the
#            absorb branch of build_deepseek2 read wk_b->ne[1] -- access violation before the server came up.
#   both   : build_deepseek2 fed the context's already-YaRN-cancelled attn_factor into ik's kq_scale formula, which
#            expects the raw value: kq_scale x0.53, logits up to 1.3 nat from upstream (gap gone with
#            --rope-scaling none). Fixed by un-cancelling first, as upstream llama.cpp does.
#            SOURCE-READ: upstream/llama.cpp/src/models/deepseek2.cpp:437-443
# Reference: upstream llama.cpp on the same GGUF bytes, same attention path (MEASURED 2026-09-28; the two paths differ
# from each other at a near-tie, in upstream as in ours, so each path is held to upstream's SAME path).
# Reads a local GGUF (LLAMA_TEST_DEEPSEEK2, default: this host's model-store blob of deepseek-v2:16b); skips when absent.
import os
import pytest
from utils import ServerProcess

DEEPSEEK2 = os.environ.get("LLAMA_TEST_DEEPSEEK2",
    "models/ollama/blobs/sha256-d8d69f2a1bfa02efc9c757c5dba444886cf33f54651c9cfe78c94046ddc048bc")
REFERENCE = {"on":  " Paris.\nThe official language of France is French.",
             "off": " Paris.\nThe currency of France is the Euro."}


@pytest.mark.skipif(not os.path.exists(DEEPSEEK2), reason="deepseek-v2 GGUF not present (set LLAMA_TEST_DEEPSEEK2)")
@pytest.mark.parametrize("fa", ["on", "off"])
def test_deepseek2_mla_greedy_matches_reference(fa):
    s = ServerProcess()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = DEEPSEEK2
    s.n_ctx = 2048
    s.n_slots = 1
    s.fa = fa
    s.start(timeout_seconds=900)
    try:
        res = s.make_request("POST", "/completion", data={
            "prompt": "The capital of France is", "temperature": 0.0, "top_k": 1, "n_predict": 16, "n_probs": 1,
        })
        assert res.status_code == 200
        assert res.body["completion_probabilities"][0]["top_logprobs"][0]["token"] == " Paris", res.body
        assert res.body["content"].startswith(REFERENCE[fa]), res.body["content"]
    finally:
        s.stop()


# Upstream next-token logprobs at "...Paris.\nThe official language", same GGUF, same path (MEASURED 2026-09-28).
# The greedy text above cannot see the YaRN kq_scale defect (the argmax survived it); these can: before the fix
# ' spoken' was -1.569 against upstream's -2.852, after it -2.679. Tolerance 0.5 nat: the tinyllama cross-engine
# control differed by <= 0.11, this model's fixed build by <= 0.23.
UPSTREAM_NEXT = {"on":  {" of": -0.5627, " is": -1.2009, " in": -2.7355, " spoken": -2.852},
                 "off": {" of": -0.6516, " is": -1.0538, " in": -2.6563, " spoken": -2.945}}


@pytest.mark.skipif(not os.path.exists(DEEPSEEK2), reason="deepseek-v2 GGUF not present (set LLAMA_TEST_DEEPSEEK2)")
@pytest.mark.parametrize("fa", ["on", "off"])
def test_deepseek2_next_token_logprobs_match_upstream(fa):
    s = ServerProcess()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = DEEPSEEK2
    s.n_ctx = 2048
    s.n_slots = 1
    s.fa = fa
    s.start(timeout_seconds=900)
    try:
        res = s.make_request("POST", "/completion", data={
            "prompt": "The capital of France is Paris.\nThe official language", "temperature": 0.0, "top_k": 1,
            "n_predict": 1, "n_probs": 5,
        })
        assert res.status_code == 200
        got = {t["token"]: t["logprob"] for t in res.body["completion_probabilities"][0]["top_logprobs"]}
        ref = UPSTREAM_NEXT[fa]
        shared = [k for k in ref if k in got]
        assert len(shared) >= 3, (got, ref)
        worst = max(shared, key=lambda k: abs(got[k] - ref[k]))
        assert abs(got[worst] - ref[worst]) < 0.5, f"{worst}: ours {got[worst]:.4f} vs upstream {ref[worst]:.4f}; {got}"
    finally:
        s.stop()
