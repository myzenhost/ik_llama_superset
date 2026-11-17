import pytest
from utils import ServerPreset, use_local_stories260k

# Watermark schemes (common/watermark.h): gumbel (default, vLLM-compatible), kgw (TGI's green list), synthid
# (SynthID-Text tournament). Row S69-ADD-TGI-ALTERNATIVES-LLGUIDANCE-PAGEDKV-RADIX-SYNTHID-2026-09-27-01.
# Every scheme uses the server's --watermark-key; a request only chooses the scheme.

server = ServerPreset.tinyllama2()
PROMPT = "Once upon a time"
KEY = "0x5eed1234abcd0001"
SCHEMES = ["gumbel", "kgw", "synthid"]


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.extra_args = ["--watermark-key", KEY]


def complete(watermark, scheme=None, seed=1, n_predict=200):
    data = {"prompt": PROMPT, "n_predict": n_predict, "temperature": 1.0, "seed": seed,
            "return_tokens": True, "watermark": watermark}
    if scheme is not None:
        data["watermark_scheme"] = scheme
    return server.make_request("POST", "/completion", data=data)


def detect(tokens, scheme):
    return server.make_request("POST", "/watermark/detect", data={"tokens": tokens, "scheme": scheme})


@pytest.mark.parametrize("scheme", SCHEMES)
def test_scheme_is_echoed_and_detected_both_ways(scheme):
    global server
    server.start()
    marked = complete(True, scheme)
    assert marked.status_code == 200, marked.body
    assert marked.body["generation_settings"]["watermark_scheme"] == scheme
    plain = complete(False, scheme)
    assert plain.status_code == 200, plain.body
    dm = detect(marked.body["tokens"], scheme)
    dp = detect(plain.body["tokens"], scheme)
    assert dm.status_code == 200 and dp.status_code == 200, (dm.body, dp.body)
    assert dm.body["scheme"] == scheme
    print(scheme, "marked", dm.body, "plain", dp.body)
    assert dm.body["is_watermarked"], dm.body
    assert not dp.body["is_watermarked"], dp.body


def test_default_scheme_is_gumbel():
    global server
    server.start()
    res = complete(True, None)
    assert res.status_code == 200, res.body
    assert res.body["generation_settings"]["watermark_scheme"] == "gumbel"


def test_scheme_mismatch_is_not_detected():
    # text marked with one scheme, checked with another scheme's detector under the SAME key
    global server
    server.start()
    marked = complete(True, "synthid")
    assert marked.status_code == 200, marked.body
    d = detect(marked.body["tokens"], "kgw")
    assert d.status_code == 200, d.body
    assert not d.body["is_watermarked"], d.body


def test_unknown_scheme_is_refused_on_both_routes():
    global server
    server.start()
    res = complete(True, "bogus")
    assert res.status_code == 400, res.body
    res = detect([1, 2, 3], "bogus")
    assert res.status_code == 400, res.body


def test_watermark_with_mirostat_is_refused():
    # mirostat draws in a branch no watermark scheme reaches: the text would come back unmarked
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "n_predict": 8, "temperature": 1.0, "mirostat": 2, "watermark": True})
    assert res.status_code == 400, res.body
