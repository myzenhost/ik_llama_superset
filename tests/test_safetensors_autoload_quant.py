# REWRITE-SAFE: tests write only under pytest's tmp_path.
"""Quantize-at-autoload of tools/safetensors_autoload.py (row S69-MISTRALRS-PICK-ISQAUTOLOAD-2026-09-28-01).

These cases fake the converter and llama-quantize (no model, no CPU load) and pin the cache logic: key contents, refusals, the .part rule, the
early return when the base was dropped. The real run (TriLM_99M safetensors -> F16 GGUF -> Q4_K_M, tensor types read back, greedy ids compared with
an offline convert + quantize) is in the commit message of the change that added this file.

Fail-before: the module had no --quant option and no ensure_quantized(), so every case below failed with an AttributeError.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("st_autoload_under_test", ROOT / "tools" / "safetensors_autoload.py")
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """A cache dir, a fake base GGUF, a fake quantizer binary, and a Popen stand-in that 'quantizes' by writing the .part file."""
    cache = tmp_path / "cache"
    cache.mkdir()
    base = cache / "model.f16.0123456789abcdef.gguf"
    base.write_bytes(b"BASE" * 100)
    binroot = tmp_path / "bin"
    binroot.mkdir()
    qbin = binroot / ("llama-quantize.exe" if st.os.name == "nt" else "llama-quantize")
    qbin.write_bytes(b"x")
    calls = []
    state = {"rc": 0, "write": True}

    class FakePopen:
        def __init__(self, cmd, stdout=None, stderr=None, creationflags=0):
            calls.append(list(cmd))
            self.cmd = cmd
            if state["write"]:
                pathlib.Path(cmd[-3] if cmd[-1].isdigit() else cmd[-2]).write_bytes(b"QUANTIZED")

        def wait(self):
            return state["rc"]

    monkeypatch.setattr(st.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(st, "_tensor_type_names", lambda p: {"F32", "Q4_K", "Q6_K"})
    return types.SimpleNamespace(cache=cache, base=base, qbin=qbin, bindir=str(binroot), calls=calls, state=state)


def test_quantize_writes_a_cache_hit_and_second_call_does_not_rerun(env):
    out = st.ensure_quantized(env.base, "Q4_K_M", env.bindir)
    assert out.name.startswith("model.f16.0123456789abcdef.Q4_K_M.") and out.suffix == ".gguf" and out.read_bytes() == b"QUANTIZED"
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["quant"] == "Q4_K_M" and meta["tensor_types"] == ["F32", "Q4_K", "Q6_K"]
    assert len(env.calls) == 1 and env.calls[0][-2:] == [str(out) + ".part", "Q4_K_M"]
    again = st.ensure_quantized(env.base, "Q4_K_M", env.bindir)
    assert again == out and len(env.calls) == 1                    # served from the cache


def test_cache_key_contains_type_binary_and_imatrix(env, tmp_path):
    a = st.ensure_quantized(env.base, "Q4_K_M", env.bindir)
    b = st.ensure_quantized(env.base, "Q6_K", env.bindir)
    assert a != b                                                  # another type is another file, base reused
    imx = tmp_path / "x.imatrix"
    imx.write_bytes(b"1")
    c = st.ensure_quantized(env.base, "Q4_K_M", env.bindir, imatrix=imx)
    assert c not in (a, b)
    assert "--imatrix" in env.calls[-1] and env.calls[-1][env.calls[-1].index("--imatrix") + 1] == str(imx)
    env.qbin.write_bytes(b"a different build of the quantizer")      # size and mtime change -> new key
    d = st.ensure_quantized(env.base, "Q4_K_M", env.bindir)
    assert d != a and len(env.calls) == 4


def test_requantizing_an_already_quantized_conversion_is_refused_up_front(env):
    with pytest.raises(SystemExit) as e:
        st.ensure_quantized(env.base, "Q4_K_M", env.bindir, outtype="q8_0")
    assert "full-precision base" in str(e.value) and not env.calls


def test_bad_type_name_and_missing_binary_are_errors(env, tmp_path):
    with pytest.raises(SystemExit):
        st.ensure_quantized(env.base, "Q4_K_M; rm -rf", env.bindir)
    with pytest.raises(SystemExit) as e:
        st.ensure_quantized(env.base, "Q4_K_M", str(tmp_path / "empty"))
    assert "llama-quantize" in str(e.value)


def test_failed_quantize_leaves_no_part_and_no_hit(env):
    env.state["rc"] = 1
    with pytest.raises(SystemExit) as e:
        st.ensure_quantized(env.base, "Q4_K_M", env.bindir)
    assert "FAILED rc=1" in str(e.value)
    assert not list(env.cache.glob("*.part")) and not list(env.cache.glob("*.Q4_K_M.*.gguf"))
    env.state.update(rc=0, write=False)                            # exit 0 but no file written
    with pytest.raises(SystemExit):
        st.ensure_quantized(env.base, "Q4_K_M", env.bindir)


def test_a_file_without_the_requested_family_is_not_cached(env, monkeypatch):
    monkeypatch.setattr(st, "_tensor_type_names", lambda p: {"F16", "F32"})   # quantizer exited 0 but nothing was quantized
    with pytest.raises(SystemExit) as e:
        st.ensure_quantized(env.base, "Q4_K_M", env.bindir)
    assert "none of the family Q4_K" in str(e.value)
    assert not list(env.cache.glob("*.Q4_K_M.*.gguf")) and not list(env.cache.glob("*.part"))


def test_drop_base_removes_the_base_and_a_later_run_is_still_a_hit(env):
    out = st.ensure_quantized(env.base, "Q4_K_M", env.bindir, drop_base=True)
    assert not env.base.exists() and out.exists()
    qbin = st.quantize_binary(env.bindir)
    hit = st.quant_hit(env.base, "Q4_K_M", qbin, None)              # base is GONE, the key is by name
    assert hit == out


def test_autoload_resolve_serves_the_quantized_hit_without_converting(env, monkeypatch, tmp_path):
    converted = []

    def fake_ensure_gguf(model_path, cache_dir, outtype="auto", force=False, model_dir_override=None, allow_quantized=False, python=None,
                         verbose=False, verify_tok="warn", bin_dir=None, early_return=None):
        if early_return is not None and not force:
            hit = early_return(env.base)
            if hit is not None:
                return hit
        converted.append(1)
        env.base.write_bytes(b"BASE" * 100)
        return env.base

    monkeypatch.setattr(st, "ensure_gguf", fake_ensure_gguf)
    a = types.SimpleNamespace(quant="Q4_K_M", outtype="auto", cache_dir=str(env.cache), force=False, model_dir=None, allow_quantized=False,
                              python=None, verbose=False, verify_tokenizer="warn", quantize_bin=None, imatrix=None, quant_threads=0, drop_base=True)
    first = st._autoload_resolve(a, "m.safetensors", env.bindir)
    assert converted == [1] and first.read_bytes() == b"QUANTIZED" and not env.base.exists()
    second = st._autoload_resolve(a, "m.safetensors", env.bindir)
    assert second == first and converted == [1]                    # nothing reconverted although the base was dropped
    a.quant = None                                                 # without the option the behaviour is the old one
    assert st._autoload_resolve(a, "m.safetensors", env.bindir) == env.base and converted == [1, 1]
    a.quant, a.outtype = "Q4_K_M", "q8_0"
    with pytest.raises(SystemExit):
        st._autoload_resolve(a, "m.safetensors", env.bindir)
