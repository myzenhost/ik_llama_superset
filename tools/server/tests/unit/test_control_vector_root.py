import os

import pytest
from utils import ServerPreset, use_local_stories260k

# --control-vector-root: POST /control-vectors/load takes a file path from the request. Without the flag any readable path is accepted
# (unchanged). With the flag only files under that directory are, a relative path is taken relative to it, and a path that escapes it
# (absolute elsewhere, or ..) is refused with 403. The check runs before the file is touched, so none of these tests needs a real
# control vector: a path inside the root reaches the loader and fails THERE ("Failed to load", 400), a path outside never gets that far.

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_ctx = 256
    server.n_slots = 1


def _load(path):
    return server.make_request("POST", "/control-vectors/load", data={"path": path})


def test_path_outside_the_root_is_refused(tmp_path):
    global server
    root = tmp_path / "vectors"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    server.extra_args = ["--control-vector-root", str(root)]
    server.start()
    res = _load(str(elsewhere / "v.gguf"))
    assert res.status_code == 403, (res.status_code, res.body)
    assert "control-vector-root" in res.body["error"]["message"], res.body


def test_dotdot_escape_is_refused(tmp_path):
    global server
    root = tmp_path / "vectors"
    root.mkdir()
    server.extra_args = ["--control-vector-root", str(root)]
    server.start()
    res = _load("../escape.gguf")
    assert res.status_code == 403, (res.status_code, res.body)
    res = _load(str(root / ".." / "escape.gguf"))
    assert res.status_code == 403, (res.status_code, res.body)


def test_path_inside_the_root_reaches_the_loader(tmp_path):
    global server
    root = tmp_path / "vectors"
    root.mkdir()
    server.extra_args = ["--control-vector-root", str(root)]
    server.start()
    for p in (str(root / "v.gguf"), "v.gguf", "sub/v.gguf"):
        res = _load(p)
        assert res.status_code == 400, (p, res.status_code, res.body)
        assert "Failed to load" in str(res.body), (p, res.body)


def test_control_without_the_flag_accepts_any_path(tmp_path):
    global server
    server.start()
    res = _load(str(tmp_path / "anywhere" / "v.gguf"))
    assert res.status_code == 400, (res.status_code, res.body)
    assert "Failed to load" in str(res.body), res.body
