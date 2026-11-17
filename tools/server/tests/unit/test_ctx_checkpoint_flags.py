import pytest
from utils import *

# item 3: -ctx-ckpt-t/--ctx-checkpoints-tolerance and -ctx-ckpt-e/--ctx-checkpoints-eviction, ported
# from the pre-merge fork's examples/server (row
# S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, item 3).
# SOURCE-READ: examples/server/server-context.cpp:3791-3834,3853-3854,4205,4860 @ 35f0cac75.
#
# Before this port neither flag was registered in common/arg.cpp, so passing either one made
# llama-server refuse to start (unrecognized argument) -- ServerProcess.start() observes that as the
# child process exiting immediately and raises RuntimeError. That is the fail-before control: these
# tests would raise RuntimeError on the pre-patch binary and pass (server starts, completion works)
# on the patched one.

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.temperature = 0.0


def test_ctx_checkpoints_tolerance_flag_accepted():
    global server
    server.extra_args = ["-ctx-ckpt-t", "64"]
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0


def test_ctx_checkpoints_tolerance_long_flag_accepted():
    global server
    server.extra_args = ["--ctx-checkpoints-tolerance", "0"]
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "cache_prompt": True,
    })
    assert res.status_code == 200


@pytest.mark.parametrize("policy", ["variance", "fifo", "auto"])
def test_ctx_checkpoints_eviction_flag_accepted(policy: str):
    global server
    # small checkpoint budget + no minimum spacing, so a longer generation actually exercises
    # eviction rather than just accepting the flag and never filling the checkpoint list.
    server.extra_args = ["-ctx-ckpt", "2", "-ctx-ckpt-i", "0", "-ctx-ckpt-e", policy]
    server.n_predict = 96
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time, in a small village, there lived a curious cat named Whiskers who loved to explore the forest every morning.",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0


def test_ctx_checkpoints_spill_flags_accepted_and_dir_prepared(tmp_path):
    """--ctx-ckpt-spill-dir / --ctx-ckpt-live-n (row S69-FLAG-WIRING-CTX-CKPT-SPILL-NOT-PORTED-2026-10-01-01).

    Before the port both flags were unknown arguments, so the server refused to start (RuntimeError from start()).
    Now it must start, create the directory, remove stale ckpt-s* files left by a crashed earlier run, and leave
    anything else in the directory alone (the control that the wipe is scoped, not a delete-everything).
    """
    global server
    spill = tmp_path / "spill"
    spill.mkdir()
    stale = spill / "ckpt-s0-1.bin"
    stale.write_bytes(b"left by a crashed run")
    keep = spill / "keep.txt"
    keep.write_text("not a checkpoint file")

    server.extra_args = ["--ctx-ckpt-spill-dir", str(spill), "--ctx-ckpt-live-n", "1",
                         "-ctx-ckpt", "2", "-ctx-ckpt-i", "0"]
    server.n_predict = 64
    server.start()

    assert spill.is_dir()
    assert not stale.exists(), "a stale ckpt-s* file must be removed at startup"
    assert keep.exists(), "files that are not checkpoint spill files must be left alone"

    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time, in a small village, there lived a curious cat named Whiskers.",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0


def test_ctx_checkpoints_spill_long_flags_create_missing_dir(tmp_path):
    global server
    spill = tmp_path / "does" / "not" / "exist" / "yet"
    server.extra_args = ["--ctx-checkpoints-spill-dir", str(spill), "--ctx-checkpoints-live-n", "2"]
    server.start()
    assert spill.is_dir(), "the spill directory (with parents) must be created at startup"


def test_ctx_checkpoints_eviction_long_flag_accepted():
    global server
    server.extra_args = ["--ctx-checkpoints", "2", "--ctx-checkpoints-interval", "0",
                          "--ctx-checkpoints-eviction", "variance"]
    server.n_predict = 96
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time, in a small village, there lived a curious cat named Whiskers.",
        "cache_prompt": True,
    })
    assert res.status_code == 200
