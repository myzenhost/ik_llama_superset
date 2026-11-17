# Fail-before/pass-after test for group G1 (runtime control vectors) of row
# S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01: GET /control-vectors,
# POST /control-vectors/{load,unload,apply} were routes on the fork's pre-merge server
# (examples/server/server.cpp, unbuilt since merge 279b4bfaf swapped in upstream's split
# tools/server) and had no equivalent in tools/server at all (grep: 0 control_vector handlers) --
# these requests 404 on an unpatched build. See
# temp/research/findings/S69_ENGINE_SERVER_ROUTES_2026-09-27.md.
#
# Correctness controls: a ZEROS control vector applied at scale 1.0 must leave greedy output
# byte-identical to no control vector at all (the direction has no effect on any layer). A
# LARGE-SCALE control vector must change greedy output (the direction dominates the residual
# stream). Both are measured against the SAME model (tinyllama2 / stories260K, n_embd=64,
# n_layer=5, read directly from the gguf via gguf-py rather than hardcoded) so a false pass from
# picking numbers the model happens to ignore is not possible.
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "gguf-py"))
import gguf  # noqa: E402

from utils import *  # noqa: E402,F401,F403
from utils import local_model_path, use_local_stories260k  # noqa: E402

server = ServerPreset.tinyllama2()

# Resolved from the model cache (utils.local_model_path). It used to be a hardcoded tests/.test_models/ path,
# which is gitignored (*.gguf) and absent from every fresh checkout, so this file failed everywhere but one tree.
_MODEL_GGUF = local_model_path("ggml-org/test-model-stories260K", "stories260K-f32.gguf")
# Control-vector files this test writes go in the suite's own tmp/, never next to the model (which may now be
# inside a shared HF cache).
_CVEC_DIR = os.path.join(os.path.dirname(__file__), "..", "tmp")


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)  # same model bytes as _MODEL_GGUF; skips with the reason if not cached
    os.makedirs(_CVEC_DIR, exist_ok=True)


def _read_model_dims(path: str):
    reader = gguf.GGUFReader(path)
    n_embd = int(reader.fields["llama.embedding_length"].parts[-1][0])
    n_layer = int(reader.fields["llama.block_count"].parts[-1][0])
    return n_embd, n_layer


def _make_cvec_gguf(path: str, n_embd: int, n_layer: int, magnitude: float):
    """Writes a control-vector GGUF: one 1-D F32 tensor 'direction.<layer>' per layer 1..n_layer,
    matching common_control_vector_load_one's expected format (common/common.cpp:2833-2913)."""
    writer = gguf.GGUFWriter(path, "control_vector")
    rng = np.random.default_rng(1234)
    for layer in range(1, n_layer + 1):
        if magnitude == 0.0:
            data = np.zeros(n_embd, dtype=np.float32)
        else:
            data = (rng.standard_normal(n_embd).astype(np.float32)) * magnitude
        writer.add_tensor(f"direction.{layer}", data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _greedy_completion(srv, prompt="Once upon a time"):
    res = srv.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": 32,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": False,
    })
    assert res.status_code == 200
    return res.body["content"]


def test_control_vectors_list_empty():
    global server
    server.start()
    res = server.make_request("GET", "/control-vectors")
    assert res.status_code == 200
    assert res.body == []


def test_control_vectors_load_unload_roundtrip():
    global server
    server.start()
    n_embd, n_layer = _read_model_dims(_MODEL_GGUF)
    cvec_path = os.path.join(_CVEC_DIR, "cvec_zeros.gguf")
    _make_cvec_gguf(cvec_path, n_embd, n_layer, 0.0)

    res = server.make_request("POST", "/control-vectors/load", data={"path": cvec_path, "scale": 1.0})
    assert res.status_code == 200
    assert res.body["success"] is True
    cv_id = res.body["id"]

    res = server.make_request("GET", "/control-vectors")
    assert res.status_code == 200
    assert len(res.body) == 1
    assert res.body[0]["id"] == cv_id
    assert res.body[0]["applied"] is True

    res = server.make_request("POST", "/control-vectors/unload", data={"id": cv_id})
    assert res.status_code == 200
    assert res.body["success"] is True

    res = server.make_request("GET", "/control-vectors")
    assert res.status_code == 200
    assert res.body == []


def test_control_vectors_unload_invalid_id():
    global server
    server.start()
    res = server.make_request("POST", "/control-vectors/unload", data={"id": 99})
    assert res.status_code == 400
    assert res.body["success"] is False


def test_control_vector_zero_scale_is_byte_identical():
    """Negative control: a zeros direction, scale 1.0, must not change greedy output at all."""
    global server
    server.start()
    n_embd, n_layer = _read_model_dims(_MODEL_GGUF)
    baseline = _greedy_completion(server)

    cvec_path = os.path.join(_CVEC_DIR, "cvec_zeros2.gguf")
    _make_cvec_gguf(cvec_path, n_embd, n_layer, 0.0)
    res = server.make_request("POST", "/control-vectors/load", data={"path": cvec_path, "scale": 1.0})
    assert res.status_code == 200

    after = _greedy_completion(server)
    assert after == baseline, "a zero-magnitude control vector must not change greedy output"


def test_control_vector_large_scale_changes_output():
    """Positive control: a large-magnitude direction must change greedy output."""
    global server
    server.start()
    n_embd, n_layer = _read_model_dims(_MODEL_GGUF)
    baseline = _greedy_completion(server)

    cvec_path = os.path.join(_CVEC_DIR, "cvec_large.gguf")
    _make_cvec_gguf(cvec_path, n_embd, n_layer, 50.0)
    res = server.make_request("POST", "/control-vectors/load", data={"path": cvec_path, "scale": 1.0})
    assert res.status_code == 200

    after = _greedy_completion(server)
    assert after != baseline, "a large-magnitude control vector must change greedy output"


def test_control_vectors_apply_scale_toggle():
    """apply (POST /control-vectors/apply) rewrites scales without reloading; scale=0 must
    reproduce the baseline (matches the zero-scale control above, via the apply path instead)."""
    global server
    server.start()
    n_embd, n_layer = _read_model_dims(_MODEL_GGUF)
    baseline = _greedy_completion(server)

    cvec_path = os.path.join(_CVEC_DIR, "cvec_large2.gguf")
    _make_cvec_gguf(cvec_path, n_embd, n_layer, 50.0)
    res = server.make_request("POST", "/control-vectors/load", data={"path": cvec_path, "scale": 1.0})
    assert res.status_code == 200
    cv_id = res.body["id"]

    changed = _greedy_completion(server)
    assert changed != baseline

    res = server.make_request("POST", "/control-vectors/apply", data=[{"id": cv_id, "scale": 0.0}])
    assert res.status_code == 200
    assert res.body["success"] is True

    restored = _greedy_completion(server)
    assert restored == baseline, "apply scale=0.0 must restore the unmodified baseline"


def _cached_completion(srv, prompt, cache_prompt=True):
    res = srv.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": 16,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": cache_prompt,
    })
    assert res.status_code == 200
    return res.body["content"], int(res.body["timings"]["cache_n"])


def test_control_vector_change_does_not_serve_stale_prompt_cache():
    """KV computed under one steering vector must not be reused under another (row
    S69-VLLM-PICK-CONTROL-VECTOR-CACHE-INVALIDATION-2026-09-29-01). Before the fix the slot prompt
    survived a vector change, so request 2 reused the KV computed with no vector."""
    global server
    server.n_slots = 1
    server.start()
    n_embd, n_layer = _read_model_dims(_MODEL_GGUF)
    prompt = "Once upon a time there was a little girl named Lily who liked to play in the garden " * 4

    _, cache_n0 = _cached_completion(server, prompt)            # warms the slot, no vector
    assert cache_n0 == 0
    _, cache_n_same = _cached_completion(server, prompt)        # control: nothing changed, the cache IS used
    assert cache_n_same > 0, "the prompt cache must still work when the configuration is unchanged"

    cvec_path = os.path.join(_CVEC_DIR, "cvec_large_cache.gguf")
    _make_cvec_gguf(cvec_path, n_embd, n_layer, 50.0)
    res = server.make_request("POST", "/control-vectors/load", data={"path": cvec_path, "scale": 1.0})
    assert res.status_code == 200
    cv_id = res.body["id"]

    out_cached, cache_n_after = _cached_completion(server, prompt)
    out_true, _ = _cached_completion(server, prompt, cache_prompt=False)
    assert cache_n_after == 0, "KV computed before the vector was loaded must not be reused"
    assert out_cached == out_true, "output with cache_prompt must equal the from-scratch output under the vector"

    # re-applying the identical configuration keeps the cache usable (the key is stable, not a counter)
    _cached_completion(server, prompt)
    res = server.make_request("POST", "/control-vectors/apply", data=[{"id": cv_id, "scale": 1.0}])
    assert res.status_code == 200
    _, cache_n_reapply = _cached_completion(server, prompt)
    assert cache_n_reapply > 0, "an identical re-apply must not invalidate the cache"

    # and removing the vector invalidates again
    res = server.make_request("POST", "/control-vectors/apply", data=[{"id": cv_id, "scale": 0.0}])
    assert res.status_code == 200
    _, cache_n_removed = _cached_completion(server, prompt)
    assert cache_n_removed == 0
