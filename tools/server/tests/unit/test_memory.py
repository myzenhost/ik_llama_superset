import re

import pytest
from utils import *

# GET /memory and the memory_*_bytes gauges: port of ik_llama.cpp 55813a5d7 examples/server/tests/features/memory.feature
# (the behave scenarios that apply to a CPU run: endpoint 501 without --memory, Host row shape, model memory against the
# file size, gauges equal the rows, gauges without --memory).

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_metrics = True


def test_memory_endpoint_off_by_default():
    global server
    server.start()
    res = server.make_request("GET", "/memory")
    assert res.status_code == 501


def test_memory_report_shape():
    global server
    server.server_memory = True
    server.start()
    res = server.make_request("GET", "/memory")
    assert res.status_code == 200
    body = res.body
    assert isinstance(body["n_layer"], int) and body["n_layer"] > 0
    rows = body["data"]
    names = [r["name"] for r in rows]
    assert len(names) == len(set(names)), names
    host = next(r for r in rows if r["name"] == "Host")
    assert "total" not in host and "free" not in host
    for r in rows:
        for f in ("model", "context", "compute"):
            assert isinstance(r[f], int) and r[f] >= 0, r
        assert ("total" in r) == ("free" in r), r
        if "total" in r:
            assert 0 < r["free"] <= r["total"], r
    assert sum(r["model"] for r in rows) > 0
    assert sum(r["context"] for r in rows) > 0
    assert sum(r["compute"] for r in rows) > 0


def test_memory_gauges_match_rows():
    global server
    server.server_memory = True
    server.start()
    rows = server.make_request("GET", "/memory").body["data"]
    text = server.make_request("GET", "/metrics").body
    for r in rows:
        for f in ("model", "context", "compute", "mmproj", "total", "free"):
            if f in r:
                line = 'llamacpp:memory_%s_bytes{device="%s"} ' % (f, r["name"])
                assert line in text, line
    for f in ("model", "context", "compute"):
        assert text.count("# HELP llamacpp:memory_%s_bytes " % f) == 1


def test_memory_gauges_without_memory_flag():
    global server
    server.start()
    text = server.make_request("GET", "/metrics").body
    m = re.search(r'llamacpp:memory_model_bytes\{device="Host"\} (\d+)', text)
    assert m and int(m.group(1)) > 0, text
