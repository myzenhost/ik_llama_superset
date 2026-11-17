# Slot erase must also drop the slot's --radix-cache nodes (they used to hand the erased prompt back: the old comment in
# server-context.cpp SLOT_ERASE said "NOT covered"). vLLM counterpart: scheduler.py:2836 reset_prefix_cache.
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tiny_random_model import PROMPT, tiny_server  # noqa: E402




def _run(tmp_path, erase):
    server = tiny_server(tmp_path)
    server.n_slots = 3
    server.slot_save_path = str(tmp_path)
    server.extra_args = ["--radix-cache", "--kv-unified", "--cache-ram", "0"]
    server.start()
    try:
        body = {"prompt": PROMPT, "n_predict": 2, "temperature": 0, "cache_prompt": True}
        r = server.make_request("POST", "/completion", data=dict(body, id_slot=0))
        assert r.status_code == 200
        if erase:
            r = server.make_request("POST", "/slots/0?action=erase")
            assert r.status_code == 200
        r = server.make_request("POST", "/completion", data=dict(body, id_slot=2))
        assert r.status_code == 200
        return r.body["timings"]["cache_n"]
    finally:
        server.stop()


def test_radix_reuse_without_erase_control(tmp_path):
    assert _run(tmp_path, erase=False) > 40


def test_erase_purges_radix_nodes(tmp_path, capfd):
    assert _run(tmp_path, erase=True) == 0
    assert "radix node(s) purged" in capfd.readouterr().out         # FIRED
