# stop_regex request field (row S69-STOP-REGEX-INCLUDE-STOP-AND-GRAMMAR-RESTORE-REMAINDER-2026-09-30-01, item 1).
#
# Idea and limits from SGLang SamplingParams (engines/sglang/python/sglang/srt/sampling/sampling_params.py:339-351,
# MAX_STOP_REGEX_LEN = 256, Apache-2.0); reimplemented, no code copied.
#
#   stop_regex  generation ends when a regular expression matches the decoded text; the match is cut like a "stop" string
#
# Model: stories260K. Each expectation is computed from an uncut baseline run of the same request, and the field is what
# changes the result (the baseline without it is the control).
import re
import pytest

from utils import ServerPreset, use_local_stories260k  # noqa: E402

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_slots = 1


def _req(**extra):
    data = {
        "prompt": "Once upon a time",
        "n_predict": 120,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": False,
    }
    data.update(extra)
    return data


def _text(**extra):
    res = server.make_request("POST", "/completion", data=_req(**extra))
    assert res.status_code == 200
    return res.body


def _stream(**extra):
    data = _req(stream=True, **extra)
    return "".join(chunk.get("content", "") for chunk in server.make_stream_request("POST", "/completion", data=data))


# two words, a space and a third word: present in any story text, not a fixed literal
PATTERN = r"[a-z]+ [a-z]+ [a-z]+ [a-z]+"


def test_stop_regex_cuts_at_the_first_match():
    global server
    server.start()
    base = _text()
    m = re.search(PATTERN, base["content"])
    assert m and m.start() > 0, "control: the uncut text contains a match that does not start at 0"
    body = _text(stop_regex=[PATTERN])
    assert body["content"] == base["content"][:m.start()]
    assert body["stop_type"] == "word"
    assert len(base["content"]) > len(body["content"]), "control: the field shortened the text"


def test_include_stop_str_keeps_the_matched_text():
    global server
    server.start()
    base = _text()
    m = re.search(PATTERN, base["content"])
    assert m
    body = _text(stop_regex=PATTERN, include_stop_str_in_output=True)
    assert body["content"] == base["content"][:m.end()]


def test_the_earliest_of_regex_and_stop_string_wins():
    global server
    server.start()
    base = _text()["content"]
    m = re.search(PATTERN, base)
    assert m
    # a literal stop string that starts before the regex match
    early = base[max(0, m.start() // 2): max(0, m.start() // 2) + 3]
    assert early and base.find(early) < m.start()
    assert _text(stop_regex=[PATTERN], stop=[early])["content"] == base[:base.find(early)]
    # a literal stop string that starts after the match: the regex wins
    late_start = m.end() + 3
    late = base[late_start:late_start + 3]
    assert late and base.find(late) > m.start()
    assert _text(stop_regex=[PATTERN], stop=[late])["content"] == base[:m.start()]


def test_streaming_holds_back_then_flushes_so_nothing_is_lost():
    global server
    server.start()
    full = _text(n_predict=400)["content"]
    assert len(full) > 600, "control: the run is longer than the 512-byte hold-back window"
    # a pattern that never matches: the whole text must still arrive, the held-back tail included, on a length stop
    never = "QQQQ[0-9]{9}"
    assert _text(n_predict=400, stop_regex=never)["content"] == full
    assert _stream(n_predict=400, stop_regex=never) == full
    # a short run (shorter than the window: everything is held until the final flush)
    short = _text(n_predict=30)["content"]
    assert _stream(n_predict=30, stop_regex=never) == short
    # streaming with a match equals non-streaming with a match
    m = re.search(PATTERN, full)
    assert m
    assert _stream(n_predict=400, stop_regex=PATTERN) == full[:m.start()]


@pytest.mark.parametrize("bad", [
    "(",                       # invalid pattern
    "a*",                      # matches the empty string
    "",                        # empty
    "x" * 257,                 # over the length cap
    [1, 2],                    # not strings
    7,                         # not a string or an array
    ["a"] * 17,                # too many
])
def test_stop_regex_invalid_is_refused(bad):
    global server
    server.start()
    res = server.make_request("POST", "/completion", data=_req(n_predict=4, stop_regex=bad))
    assert res.status_code == 400


def test_a_pattern_of_exactly_the_cap_is_accepted():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data=_req(n_predict=4, stop_regex="Q" * 256))
    assert res.status_code == 200
