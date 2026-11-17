import os
import pytest
from utils import *

# G2: saved slot prompts, ported from the pre-merge fork's examples/server/server.cpp
# (row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01).
# SOURCE-READ: examples/server/server.cpp:1786-1855, 1858-1965, 2214-2220 @ 35f0cac75.

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.temperature = 0.0
    # tools/server/tests/.gitignore excludes tmp/, so a fresh checkout has none; --slot-save-path
    # refuses to start against a missing directory ("not a directory: ./tmp"), which is correct
    # server behaviour -- the fix belongs here, not in the server. test_slot_save.py (upstream's own
    # test file) makes the same assumption and is left alone; this file creates its own directory.
    os.makedirs("tmp", exist_ok=True)


def test_slots_list_always_available():
    # GET /slots/list must work even without --slot-save-path and without --slots.
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("GET", "/slots/list")
    assert res.status_code == 200
    assert isinstance(res.body, list)
    assert len(res.body) >= 1

    slot1 = next((s for s in res.body if s["slot_id"] == 1), None)
    assert slot1 is not None
    assert slot1["token_count"] > 0
    assert isinstance(slot1["prompt"], str)
    assert len(slot1["prompt"]) > 0


def test_slots_list_reflects_idle_slot():
    # a never-used slot must still appear, with token_count == 0.
    global server
    server.start()

    res = server.make_request("GET", "/slots/list")
    assert res.status_code == 200
    for slot in res.body:
        assert slot["token_count"] == 0
        assert slot["prompt"] == ""


def test_saved_prompts_gated_without_slot_save_path():
    # /list, /delete_prompt, /rename_prompt all require --slot-save-path.
    global server
    server.start()

    res = server.make_request("GET", "/list")
    assert res.status_code == 501  # ERROR_TYPE_NOT_SUPPORTED

    res = server.make_request("POST", "/delete_prompt", data={"filename": "x.bin"})
    assert res.status_code == 501

    res = server.make_request("POST", "/rename_prompt", data={"old_filename": "x.bin", "new_filename": "y.bin"})
    assert res.status_code == 501


def test_saved_prompts_list_delete_rename():
    global server
    server.slot_save_path = "./tmp"
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={"filename": "prompt_a.bin"})
    assert res.status_code == 200
    n_saved = res.body["n_saved"]

    # GET /list must show the saved file with metadata + decoded prompt text.
    res = server.make_request("GET", "/list")
    assert res.status_code == 200
    entry = next((e for e in res.body if e["filename"] == "prompt_a.bin"), None)
    assert entry is not None
    assert entry["token_count"] == n_saved
    assert entry["filesize"] > 0
    assert isinstance(entry["prompt"], str)
    assert len(entry["prompt"]) > 0

    # rename
    res = server.make_request("POST", "/rename_prompt", data={
        "old_filename": "prompt_a.bin",
        "new_filename": "prompt_b.bin",
    })
    assert res.status_code == 200
    assert res.body["status"] == "renamed"

    res = server.make_request("GET", "/list")
    names = [e["filename"] for e in res.body]
    assert "prompt_b.bin" in names
    assert "prompt_a.bin" not in names

    # renaming onto an existing file must fail (409-style invalid-request, not silently overwrite)
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 2,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    res = server.make_request("POST", "/slots/2?action=save", data={"filename": "prompt_c.bin"})
    assert res.status_code == 200
    res = server.make_request("POST", "/rename_prompt", data={
        "old_filename": "prompt_c.bin",
        "new_filename": "prompt_b.bin",
    })
    assert res.status_code == 400

    # delete
    res = server.make_request("POST", "/delete_prompt", data={"filename": "prompt_b.bin"})
    assert res.status_code == 200
    assert res.body["status"] == "deleted"

    res = server.make_request("GET", "/list")
    names = [e["filename"] for e in res.body]
    assert "prompt_b.bin" not in names


def test_delete_prompt_rejects_path_traversal():
    global server
    server.slot_save_path = "./tmp"
    server.start()

    res = server.make_request("POST", "/delete_prompt", data={"filename": "../evil.bin"})
    assert res.status_code == 400

    res = server.make_request("POST", "/rename_prompt", data={
        "old_filename": "a.bin",
        "new_filename": "../../evil.bin",
    })
    assert res.status_code == 400


def test_delete_prompt_missing_file():
    global server
    server.slot_save_path = "./tmp"
    server.start()

    res = server.make_request("POST", "/delete_prompt", data={"filename": "does_not_exist.bin"})
    assert res.status_code == 404
