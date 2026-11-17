import pytest
from utils import ServerPreset

# Regression guard for the class of ik_llama.cpp issue #2398
# (https://github.com/ikawrakow/ik_llama.cpp/issues/2398): /completion parsed, stored and echoed the penalty family
# through /slots, yet generation was byte-identical at every value, while /v1/chat/completions applied it. Nothing
# failed; the only symptom was an output that did not move. MEASURED 2026-09-28 on this tree (temp/_pen_0928_probe.log,
# qwen2.5-coder-0.5b, greedy): presence/frequency/repeat all change the output on BOTH endpoints. This test keeps it so.
#
# Each case compares greedy output with the penalty OFF vs strongly ON. A positive control on the same endpoint proves
# a sampler parameter can move this output at all, so "identical" means the penalty was dropped, not that the tiny
# model is too peaked to move.

server = ServerPreset.tinyllama2()

PROMPT = "Once upon a time there was a little girl. The little girl had a little dog. The little dog"
MESSAGES = [{"role": "user", "content": PROMPT}]


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def _generate(endpoint: str, extra: dict) -> str:
    body = {"temperature": 0.0, "seed": 7, "cache_prompt": False}
    body.update(extra)
    if endpoint == "/completion":
        body["prompt"] = PROMPT
        body["n_predict"] = 48
        res = server.make_request("POST", "/completion", data=body)
        assert res.status_code == 200
        return res.body["content"]
    body["messages"] = MESSAGES
    body["max_tokens"] = 48
    res = server.make_request("POST", "/v1/chat/completions", data=body)
    assert res.status_code == 200
    return res.body["choices"][0]["message"]["content"]


@pytest.mark.parametrize("endpoint", ["/completion", "/v1/chat/completions"])
@pytest.mark.parametrize("key,off,on", [
    ("presence_penalty", 0.0, 50.0),
    ("frequency_penalty", 0.0, 5.0),
    ("repeat_penalty", 1.0, 3.0),
])
def test_penalty_changes_greedy_output(endpoint: str, key: str, off: float, on: float):
    global server
    server.start()
    # positive control: a sampler parameter CAN move this endpoint's output
    hot_a = _generate(endpoint, {"temperature": 2.0, "seed": 1})
    hot_b = _generate(endpoint, {"temperature": 2.0, "seed": 2})
    assert hot_a != hot_b, "control failed: temperature/seed did not change output, so a no-change result below proves nothing"

    out_off = _generate(endpoint, {key: off})
    out_on = _generate(endpoint, {key: on})
    assert out_off != out_on, f"{key}={on} left {endpoint} greedy output byte-identical to {key}={off} (issue #2398 class)"

# --- penalty_scope (row S69-mistralrs-PICK-PENALTY-SCOPE-OPENAI-SEMANTICS-2026-09-28-01) ---
# "generated" = OpenAI/vLLM semantics: frequency/presence count only tokens generated for this request
# (vLLM model_executor/layers/utils.py:71-79; mistral.rs sampler.rs:1983-2029). Default "all" is upstream
# llama.cpp: the prompt is inside the window too. So on the FIRST generated token, "generated" has nothing
# to penalise and must equal the penalty-off token, while "all" penalises the prompt's tokens.


def _complete(extra: dict, n_predict: int):
    body = {"prompt": PROMPT, "n_predict": n_predict, "temperature": 0.0, "seed": 7, "cache_prompt": False}
    body.update(extra)
    res = server.make_request("POST", "/completion", data=body)
    assert res.status_code == 200, res.body
    return res.body


def test_penalty_scope_generated_ignores_the_prompt_on_the_first_token():
    global server
    server.start()
    off = _complete({"presence_penalty": 0.0}, 1)["content"]
    all_scope = _complete({"presence_penalty": 50.0}, 1)["content"]
    gen_scope = _complete({"presence_penalty": 50.0, "penalty_scope": "generated"}, 1)["content"]
    # control: in the default scope the prompt's tokens ARE penalised, and that moves the first token here
    assert all_scope != off, "control failed: presence over the prompt did not move the first token"
    assert gen_scope == off, "penalty_scope=generated penalised a prompt token on the first generated token"


def test_penalty_scope_generated_still_penalises_generated_tokens():
    global server
    server.start()
    off = _complete({"presence_penalty": 0.0}, 48)["content"]
    gen_scope = _complete({"presence_penalty": 50.0, "penalty_scope": "generated"}, 48)["content"]
    assert gen_scope != off, "penalty_scope=generated left 48-token greedy output identical to penalty off"


# --- penalize_prompt (row S69-ENGREV-UNSLOTH-PENALIZE-PROMPT-2026-10-27-01, unsloth U283) ---
# SOURCE-READ: tmp/unsloth_pr_1107/U283.diff (unslothai/llama.cpp PR 283, MIT; test written here, not copied)
# SOURCE-NUANCE: upstream's switch also covers the repeat penalty and DRY and makes repeat_last_n count generated
#   tokens only (README hunk), unlike this fork's penalty_scope, which still applies repeat_penalty over the prompt.
# So repeat_penalty is the discriminating case below.


@pytest.mark.parametrize("key,on", [("presence_penalty", 50.0), ("repeat_penalty", 3.0), ("frequency_penalty", 5.0)])
def test_penalize_prompt_false_ignores_the_prompt_on_the_first_token(key: str, on: float):
    global server
    server.start()
    off = _complete({}, 1)["content"]
    pen = _complete({key: on}, 1)["content"]
    nopen = _complete({key: on, "penalize_prompt": False}, 1)["content"]
    assert pen != off, f"control failed: {key} over the prompt did not move the first token"
    assert nopen == off, f"penalize_prompt=false still penalised a prompt token ({key})"


def test_penalize_prompt_false_still_penalises_generated_tokens():
    global server
    server.start()
    off = _complete({"repeat_penalty": 1.0}, 48)["content"]
    on = _complete({"repeat_penalty": 3.0, "penalize_prompt": False, "repeat_last_n": 4096}, 48)["content"]
    assert on != off, "penalize_prompt=false left 48-token greedy output identical to penalty off"


def test_penalize_prompt_is_echoed():
    global server
    server.start()
    assert _complete({}, 1)["generation_settings"]["penalize_prompt"] is True
    assert _complete({"penalize_prompt": False}, 1)["generation_settings"]["penalize_prompt"] is False


def test_penalty_scope_is_echoed_and_validated():
    global server
    server.start()
    body = _complete({"penalty_scope": "generated"}, 1)
    assert body["generation_settings"]["penalty_scope"] == "generated"
    assert _complete({}, 1)["generation_settings"]["penalty_scope"] == "all"
    res = server.make_request("POST", "/completion", data={"prompt": PROMPT, "n_predict": 1,
                                                           "penalty_scope": "prompt-only"})
    assert res.status_code != 200
