# REWRITE-SAFE: tests write only under pytest's tmp_path.
"""Shard-name containment of tools/safetensors_autoload.py (row S69-ENGREV-INFERFLOW-SHARD-NAME-GUARD-2026-10-27-01).

An index.json weight_map value is a file name taken from a downloaded model dir. "../outside.safetensors" or an absolute path
used to be joined to the model dir unchecked (pathlib: an absolute right-hand side REPLACES the left), so the autoload could
hardlink/copy a file from outside the dir. Fail-before: _shards_from_index returned the escaping path, no SystemExit.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("st_autoload_guard_under_test", ROOT / "tools" / "safetensors_autoload.py")
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)


def _model(tmp_path, values):
    d = tmp_path / "model"
    d.mkdir()
    (tmp_path / "outside.safetensors").write_bytes(b"x")
    for v in values:
        if "/" not in v and "\\" not in v and ".." not in v and ":" not in v and v != "." and "\0" not in v:
            (d / v).write_bytes(b"x")
    idx = d / "model.safetensors.index.json"
    idx.write_text(json.dumps({"weight_map": {"t%d" % i: v for i, v in enumerate(values)}}), encoding="utf-8")
    return d, idx


@pytest.mark.parametrize("bad", ["../outside.safetensors", "sub/x.safetensors", "sub\\x.safetensors", "..", "C:/Windows/x.safetensors", "/etc/x",
                                 # cases of the second port of this row (88e5ac94f6): Windows-style parent, '.', a NUL byte
                                 "..\\outside.safetensors", ".", "a\0b.safetensors"])
def test_shards_from_index_refuses_escaping_name(tmp_path, bad):
    _, idx = _model(tmp_path, [bad])
    with pytest.raises(SystemExit) as e:
        st._shards_from_index(idx)
    assert "[safetensors_autoload]" in str(e.value)
    assert bad in str(e.value)


def test_shards_from_index_normal_sharded_dir_unchanged(tmp_path):
    names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    d, idx = _model(tmp_path, names)
    shards, got_idx = st._shards_from_index(idx)
    assert [p.name for p in shards] == names and got_idx == idx


def test_converter_would_find_skips_escaping_entry(tmp_path):
    d, idx = _model(tmp_path, ["shard-a.safetensors"])
    idx.write_text(json.dumps({"weight_map": {"a": "../outside.safetensors", "b": "shard-a.safetensors"}}), encoding="utf-8")
    # converter_would_find lists model*.safetensors first; shard-a does not start with "model", so the index path is exercised
    found = st.converter_would_find(d)
    assert [p.name for p in found] == ["shard-a.safetensors"]
    assert all(p.parent == d for p in found)
