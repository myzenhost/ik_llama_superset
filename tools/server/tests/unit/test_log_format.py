import json
import os
import pytest
from utils import *

# item 4: --log-format, ported from the pre-merge fork's examples/server (row
# S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, item 4).
# SOURCE-READ: examples/server/server.cpp:62,470 @ 35f0cac75 (server_log_json),
#              examples/server/server-common.cpp:26-65 @ 35f0cac75 (server_log()).
#
# Before this port --log-format was not registered in common/arg.cpp at all, so passing it made
# llama-server refuse to start (RuntimeError from the harness's own process-death check). That is the
# fail-before control for the flag-acceptance tests.

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.temperature = 0.0
    os.makedirs("tmp", exist_ok=True)


def test_log_format_json_flag_accepted():
    global server
    server.extra_args = ["--log-format", "json"]
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "cache_prompt": True,
    })
    assert res.status_code == 200


def test_log_format_text_flag_accepted():
    global server
    server.extra_args = ["--log-format", "text"]
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "cache_prompt": True,
    })
    assert res.status_code == 200


def test_log_format_invalid_value_rejected():
    global server
    server.extra_args = ["--log-format", "xml"]
    with pytest.raises(RuntimeError):
        server.start()


def test_log_format_json_emits_json_lines():
    # NOTE (measured): --log-format only reaches messages that flow through the shared common_log
    # (SRV_*/SLT_*/LOG_* macros -- CMN/SRV-prefixed lines below). Some startup diagnostics
    # (llama_model_loader's own dump, "CPU: using device ...") are raw fprintf/printf calls in the
    # model loader, never routed through common_log, so they stay plain text under --log-format json
    # too. This is not a regression: the pre-merge fork's own server_log_json had the identical scope
    # (server-common.cpp's server_log(), used only by that file's own LOG_INFO/LOG_ERROR macros, never
    # by llama_model_loader's prints either). This test asserts the lines this flag DOES control are
    # valid JSON, not that 100% of process stdout is JSON.
    global server
    server.extra_args = ["--log-format", "json", "-lv", "3"]
    server.log_path = "tmp/log_format_json.log"
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    server.stop()

    with open(server.log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [l for l in f.read().splitlines() if l.strip()]

    assert len(lines) > 3, f"expected several log lines, got {len(lines)}"

    # lines from common_log always start with '{' (a JSON object) in json mode; any line that starts
    # differently is a raw fprintf from outside common_log's reach (see NOTE above), not this flag's
    # failure to apply.
    common_log_lines = [l for l in lines if l.lstrip().startswith("{")]
    assert len(common_log_lines) > 3, f"expected several common_log-routed lines, got {len(common_log_lines)}"

    parsed_objs = []
    for line in common_log_lines:
        obj = json.loads(line)  # raises if a '{'-prefixed line is not valid, self-contained JSON
        assert set(["timestamp", "level", "msg"]).issubset(obj.keys())
        assert isinstance(obj["timestamp"], int)
        assert isinstance(obj["level"], str)
        assert isinstance(obj["msg"], str)
        parsed_objs.append(obj)

    # the startup line added by this port's common_params_print_info hook must say so
    assert any("log format = json" in o["msg"] for o in parsed_objs)

    # the startup line added by this port's common_params_print_info hook must say so
    assert any("log format = json" in l["msg"] for l in (json.loads(x) for x in lines))


def test_log_format_text_is_not_json_lines():
    global server
    server.extra_args = ["--log-format", "text", "-lv", "3"]
    server.log_path = "tmp/log_format_text.log"
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    server.stop()

    with open(server.log_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    lines = [l for l in content.splitlines() if l.strip()]
    assert len(lines) > 3

    json_parseable = 0
    for line in lines:
        try:
            json.loads(line)
            json_parseable += 1
        except (json.JSONDecodeError, ValueError):
            pass
    assert json_parseable == 0, "text format must not emit any line that parses as standalone JSON"
