# Regression test: llama-arch MoE layers must keep the FFN residual stream.
#
# ROOT-CAUSE: src/graphs/build_llama.cpp passed add_input = true to llm_build_moe_ffn with the NORMED ffn input as
# `cur`, and llm_build_moe_ffn adds its own `cur` argument (`auto input = cur;`) -- so every MoE layer returned
# moe(norm(x)) + norm(x) instead of moe(norm(x)) + x. Inherited from upstream ik_llama 0d7eb3418 (#1080).
# CONFIGS-TRIED: fused MoE on and -no-fmoe, CPU and CUDA -- all gave ' is is is ...' with the first token ' is' at
# logprob -0.0. ISOLATED: the dense stories15M model on the same build matched upstream, so only the MoE path.
#
# Reference: upstream llama.cpp llama-completion (CPU, --temp 0 --top-k 1, 2026-09-28) continues the prompt with
# ' to be kind." I," I said. I am a very wise owl, and I know that I can help you find the answer ...'.
import pytest
from utils import ServerPreset, ServerProcess

server: ServerProcess

PROMPT = "I believe the meaning of life is"
REFERENCE_PREFIX = " to be kind."
REFERENCE_WORDS = ("wise", "owl", "help you find")


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    server.fa = "off"


@pytest.mark.parametrize("extra", [[], ["-no-fmoe"]])
def test_moe_greedy_matches_upstream(extra):
    global server
    server.extra_args = extra
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "temperature": 0.0, "top_k": 1, "n_predict": 32, "n_probs": 3,
    })
    assert res.status_code == 200
    content = res.body["content"]
    assert content.startswith(REFERENCE_PREFIX), content[:80]
    for w in REFERENCE_WORDS:
        assert w in content, (w, content)
    # the residual-less graph put all the mass on one token; a residual-carrying model does not
    top = res.body["completion_probabilities"][0]["top_logprobs"]
    assert top[0]["logprob"] < -0.5, top


def test_moe_output_is_not_degenerate_repetition():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "temperature": 0.0, "top_k": 1, "n_predict": 48,
    })
    assert res.status_code == 200
    words = res.body["content"].split()
    assert len(words) >= 10
    # ' is is is ...' has one distinct word; real text has many
    assert len(set(words)) >= len(words) // 3, words
