# OpenAI /v1/completions echo + logprobs (row S69-PROMPT-LOGPROBS-OPENAI-ECHO-AND-SCORE-SURFACE-2026-09-30-01, item 1).
#
# SOURCE-READ: portable/envs/lm_eval/Lib/site-packages/lm_eval/models/openai_completions.py:61-122 (lm-eval 0.4.12, MIT, contract only).
# A loglikelihood client sends {prompt: token ids, max_tokens: 1, logprobs: 1, echo: true, temperature: 0} and reads
# choices[].logprobs.token_logprobs / top_logprobs: sum(token_logprobs[ctxlen:-1]) and "greedy" when every token's log-probability equals
# max(top_logprobs value) at its position. The numbers must agree with the native prompt_logprobs field, which test_prompt_logprobs.py
# checks against an independent oracle (one-token requests with n_probs over the whole vocabulary).
import math
import pytest

from utils import ServerPreset, use_local_stories260k  # noqa: E402

server = ServerPreset.tinyllama2()

TEXT = "Once upon a time there was a little girl named Lily who loved to play"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_slots = 2
    server.n_batch = 512


def _tokens():
    res = server.make_request("POST", "/tokenize", data={"content": TEXT, "add_special": False})
    assert res.status_code == 200
    return res.body["tokens"]


def _detok(toks):
    """The prompt text the server itself reports for this token list (the `prompt` field of a native completion).
    /detokenize is not the reference: it puts back the leading space of the first piece, the request's own prompt text does not."""
    res = server.make_request("POST", "/completion", data={
        "prompt": toks, "n_predict": 1, "temperature": 0.0, "cache_prompt": False})
    assert res.status_code == 200
    return res.body["prompt"]


def _echo(prompt, **extra):
    data = {"prompt": prompt, "max_tokens": 1, "logprobs": 1, "echo": True, "temperature": 0, "seed": 1}
    data.update(extra)
    return server.make_request("POST", "/v1/completions", data=data)


def _lm_eval_parse(choice, ctxlen):
    """lm-eval 0.4.12 LocalCompletionsAPI.parse_logprobs for one choice (openai_completions.py:113-121)."""
    lp = choice["logprobs"]
    logprobs = sum(lp["token_logprobs"][ctxlen:-1])
    tokens_logprobs = lp["token_logprobs"][ctxlen:-1]
    top_logprobs = lp["top_logprobs"][ctxlen:-1]
    is_greedy = True
    for tok, top in zip(tokens_logprobs, top_logprobs):
        if tok != max(top.values()):
            is_greedy = False
            break
    return logprobs, is_greedy


def test_echo_returns_the_legacy_arrays_for_every_prompt_token():
    global server
    server.start()
    toks = _tokens()
    assert len(toks) >= 12
    res = _echo(toks)
    assert res.status_code == 200
    choice = res.body["choices"][0]
    lp = choice["logprobs"]
    n = len(toks) + 1                       # the prompt tokens, then the one generated token
    assert len(lp["tokens"]) == len(lp["token_logprobs"]) == len(lp["top_logprobs"]) == len(lp["text_offset"]) == n
    assert lp["token_logprobs"][0] is None and lp["top_logprobs"][0] is None, "the first prompt token has no context"
    assert all(isinstance(v, float) and math.isfinite(v) and v <= 0.0 for v in lp["token_logprobs"][1:])
    assert lp["text_offset"][0] == 0 and lp["text_offset"] == sorted(lp["text_offset"])
    assert choice["text"].startswith(_detok(toks)), "echo: the text starts with the prompt"
    assert len(choice["text"]) > len(_detok(toks)), "control: the generated token follows the prompt"


def test_values_equal_the_native_prompt_logprobs_and_the_lm_eval_rule_holds():
    global server
    server.start()
    toks = _tokens()
    native = server.make_request("POST", "/completion", data={
        "prompt": toks, "n_predict": 1, "temperature": 0.0, "prompt_logprobs": True, "cache_prompt": False})
    assert native.status_code == 200
    npl = native.body["prompt_logprobs"]

    res = _echo(toks)
    assert res.status_code == 200
    choice = res.body["choices"][0]
    got = choice["logprobs"]["token_logprobs"]
    for k in range(1, len(toks)):
        assert abs(got[k] - npl["logprobs"][k]) < 1e-6, f"position {k}: echo {got[k]}, native {npl['logprobs'][k]}"

    n_greedy_true = n_greedy_false = 0
    for ctxlen in range(1, len(toks)):
        logprob, greedy = _lm_eval_parse(choice, ctxlen)
        want = sum(npl["logprobs"][ctxlen:])
        assert abs(logprob - want) < 1e-5, f"ctxlen {ctxlen}: {logprob} vs {want}"
        want_greedy = all(g == 1 for g in npl["is_greedy"][ctxlen:])
        assert greedy == want_greedy, f"ctxlen {ctxlen}: lm-eval rule says greedy={greedy}, native flags say {want_greedy}"
        n_greedy_true += int(greedy)
        n_greedy_false += int(not greedy)
    assert n_greedy_true > 0 and n_greedy_false > 0, "control: the greedy verdict is not constant across context lengths"


def test_echo_without_logprobs_only_prefixes_the_prompt():
    global server
    server.start()
    toks = _tokens()
    res = server.make_request("POST", "/v1/completions", data={
        "prompt": toks, "max_tokens": 3, "echo": True, "temperature": 0, "seed": 1})
    assert res.status_code == 200
    choice = res.body["choices"][0]
    plain = server.make_request("POST", "/v1/completions", data={
        "prompt": toks, "max_tokens": 3, "temperature": 0, "seed": 1}).body["choices"][0]
    assert choice["text"] == _detok(toks) + plain["text"]
    assert choice["logprobs"] is None


def test_without_echo_the_response_has_no_legacy_arrays():
    global server
    server.start()
    res = server.make_request("POST", "/v1/completions", data={
        "prompt": _tokens(), "max_tokens": 1, "logprobs": 1, "temperature": 0, "seed": 1})
    assert res.status_code == 200
    lp = res.body["choices"][0]["logprobs"]
    assert "tokens" not in lp and "token_logprobs" not in lp


def test_a_batch_of_prompts_gets_one_choice_each():
    global server
    server.start()
    a = _tokens()
    b = a[:8]
    res = _echo([a, b])
    assert res.status_code == 200
    choices = sorted(res.body["choices"], key=lambda c: c["index"])
    assert len(choices) == 2
    assert len(choices[0]["logprobs"]["token_logprobs"]) == len(a) + 1
    assert len(choices[1]["logprobs"]["token_logprobs"]) == len(b) + 1


def test_echo_with_stream_answers_400():
    global server
    server.start()
    res = server.make_request("POST", "/v1/completions", data={
        "prompt": _tokens(), "max_tokens": 1, "echo": True, "stream": True})
    assert res.status_code == 400
    assert "echo" in str(res.body)
