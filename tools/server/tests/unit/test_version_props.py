# Fail-before/pass-after test for group G5 (info) of row
# S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01: GET /version
# (old handle_version) and GET /v1/props (old handle_props_simple) were routes on the fork's
# pre-merge server (examples/server/server.cpp, unbuilt since merge 279b4bfaf swapped in upstream's
# split tools/server) and had no equivalent in tools/server at all -- these two requests 404 on an
# unpatched build. See temp/research/findings/S69_ENGINE_SERVER_ROUTES_2026-09-27.md.
import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_version():
    global server
    server.start()
    res = server.make_request("GET", "/version")
    assert res.status_code == 200
    assert res.body["version"] == 4
    assert res.body["features"]["sql"] is False
    assert res.body["features"]["zstd_compression"] is False


def test_v1_props():
    global server
    server.start()
    res = server.make_request("GET", "/v1/props")
    assert res.status_code == 200
    assert isinstance(res.body["model_name"], str)
    assert isinstance(res.body["model_path"], str)
    assert "vision" in res.body["modalities"]
    assert "audio" in res.body["modalities"]
    assert res.body["n_ctx"] > 0
