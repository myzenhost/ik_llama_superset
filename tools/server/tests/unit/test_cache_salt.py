# vLLM `cache_salt` (vllm/v1/core/kv_cache_utils.py:630-640): KV reuse is restricted to requests with the same salt.
# Random-weight model, prompt sent as token ids. timings.cache_n is the reuse signal.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tiny_random_model import PROMPT, tiny_server  # noqa: E402


def _run(server, extra):
    body = {"prompt": PROMPT, "n_predict": 2, "temperature": 0, "cache_prompt": True}
    body.update(extra)
    res = server.make_request("POST", "/completion", data=body)
    assert res.status_code == 200, res.body
    return res.body


def test_cache_salt_isolates_reuse(tmp_path, capfd):
    server = tiny_server(tmp_path)
    server.extra_args = ["--cache-ram", "0"]
    server.start()
    try:
        _run(server, {})
        assert _run(server, {})["timings"]["cache_n"] > 40          # control: unsalted reuse unchanged
        _run(server, {"cache_salt": "tenant-a"})
        assert _run(server, {"cache_salt": "tenant-a"})["timings"]["cache_n"] > 40   # same salt reuses
        assert _run(server, {"cache_salt": "tenant-b"})["timings"]["cache_n"] == 0   # other salt must not
    finally:
        server.stop()
    assert "cache namespace active" in capfd.readouterr().out       # FIRED
