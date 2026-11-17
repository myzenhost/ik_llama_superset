# repetition_detection request field (row S69-rtp-llm-PICK-ONLINE-REPETITION-LOOP-GUARD-2026-09-29-01).
#
# SOURCE-READ: engines/vllm/vllm/v1/core/sched/utils.py:10-62 (tail test: the last L tokens repeated min_count times back to back,
# L from min_pattern_size to max_pattern_size), :126-138 (runs after the limits and after min_tokens; stop reason
# "repetition_detected"), engines/vllm/vllm/sampling_params.py:165-191 (parameter rules). Apache-2.0, no code copied. Prior art with the
# same contract: rtp-llm OnlineRepetitionTracker, exllamav3/TabbyAPI loop_detect_window, Aphrodite repetition_detection.
#
# The tiny model does not loop by itself, so a loop is forced: logit_bias +50 on one token makes every greedy step pick it. A run of
# identical tokens matches a pattern of any length, which is what lets the parameters be told apart by WHERE the stop lands.
import pytest

from utils import ServerPreset, use_local_stories260k  # noqa: E402

server = ServerPreset.tinyllama2()

LOOP_TOKEN = 5
BIAS = [[LOOP_TOKEN, 50.0]]


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_slots = 1


def _gen(n_predict=64, **extra):
    data = {"prompt": "Once upon a time", "n_predict": n_predict, "temperature": 0.0, "seed": 42, "cache_prompt": False,
            "return_tokens": True, "logit_bias": BIAS}
    data.update(extra)
    return server.make_request("POST", "/completion", data=data)


def test_a_loop_is_stopped_at_the_count():
    global server
    server.start()
    plain = _gen()
    assert plain.status_code == 200
    assert len(plain.body["tokens"]) == 64 and plain.body["stop_type"] == "limit", "control: the forced loop runs to n_predict"
    assert set(plain.body["tokens"]) == {LOOP_TOKEN}, "control: every token is the boosted one"

    res = _gen(repetition_detection={"max_pattern_size": 2, "min_count": 4})
    assert res.status_code == 200
    assert res.body["stop_type"] == "repetition"
    assert res.body["tokens"] == [LOOP_TOKEN] * 4, "length-1 pattern repeated 4 times ends the run at the 4th token"


def test_min_pattern_size_moves_the_stop():
    global server
    server.start()
    short = _gen(repetition_detection={"min_pattern_size": 1, "max_pattern_size": 2, "min_count": 3})
    assert len(short.body["tokens"]) == 3, "length-1 pattern x3"
    longer = _gen(repetition_detection={"min_pattern_size": 2, "max_pattern_size": 2, "min_count": 3})
    assert len(longer.body["tokens"]) == 6, "min_pattern_size 2 leaves length 1 out: length-2 pattern x3 needs 6 tokens"
    assert longer.body["stop_type"] == "repetition"


def test_a_count_that_is_never_reached_changes_nothing():
    global server
    server.start()
    res = _gen(repetition_detection={"max_pattern_size": 4, "min_count": 100})
    assert res.status_code == 200
    assert len(res.body["tokens"]) == 64 and res.body["stop_type"] == "limit"


def test_min_tokens_holds_the_check_back():
    global server
    server.start()
    res = _gen(repetition_detection={"max_pattern_size": 1, "min_count": 2}, min_tokens=10)
    assert res.status_code == 200
    assert res.body["stop_type"] == "repetition"
    assert len(res.body["tokens"]) == 10, "the loop is there from token 2, but the check waits for min_tokens"


def test_finish_reason_is_stop_on_the_openai_route():
    global server
    server.start()
    res = server.make_request("POST", "/v1/completions", data={
        "prompt": "Once upon a time", "max_tokens": 64, "temperature": 0.0, "seed": 42, "cache_prompt": False,
        "logit_bias": {str(LOOP_TOKEN): 50.0}, "repetition_detection": {"max_pattern_size": 2, "min_count": 4}})
    assert res.status_code == 200
    assert res.body["choices"][0]["finish_reason"] == "stop"
    assert res.body["usage"]["completion_tokens"] == 4


def test_null_or_zero_size_leaves_it_off():
    global server
    server.start()
    for off in (None, {"max_pattern_size": 0}):
        res = _gen(repetition_detection=off)
        assert res.status_code == 200
        assert len(res.body["tokens"]) == 64 and res.body["stop_type"] == "limit"


@pytest.mark.parametrize("bad", [
    {"max_pattern_size": 2},                                    # min_count missing
    {"max_pattern_size": 2, "min_count": 1},                    # min_count < 2
    {"max_pattern_size": 2, "min_pattern_size": 3, "min_count": 2},  # min > max
    {"max_pattern_size": -1, "min_count": 2},
    {"max_pattern_size": 2, "min_count": "x"},
    {"max_pattern_size": 100000, "min_count": 2},
    "yes",
])
def test_invalid_parameters_answer_400(bad):
    global server
    server.start()
    res = _gen(repetition_detection=bad)
    assert res.status_code == 400
    assert "repetition_detection" in str(res.body)
