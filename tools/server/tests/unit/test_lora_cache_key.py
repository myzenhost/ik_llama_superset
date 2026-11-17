"""Served end-to-end check that the host prompt cache and the radix cache are keyed by the LoRA configuration.

Row S69-LORAX-PICK-LORA-CONFIG-IN-PROMPT-CACHE-AND-RADIX-KEY-2026-09-29-01 (fix commits e8d4a1707, 7cd7b18c3, 611b8cb8b) was
verified by a unit test on the cache API (tests/test-server-lora-cache-key.cpp) and by a one-off script. Nothing ran the
SERVED path under a test: a request reaching prompt_load()/the radix match in get_available_slot() with a different adapter
configuration than the entry it matches. This file is that test (residual carved out in
S69-FORK-STANDALONE-PROBES-AND-SERVED-E2E-SCRIPTS-NOT-RUN-BY-ANY-GATE-2026-10-01-01).

WHAT IS OBSERVED, AND WHAT IS NOT. The signal is timings.cache_n (prompt tokens served from a cache rather than prefilled).
It is NOT the generated text: on a model built through the llm_build_context path the server's LoRA is ignored (row
S69-SERVER-LORA-IGNORED-ON-LLM-BUILD-CONTEXT-PATH-MODELS-2026-09-29-01, measured: stories260K logprobs identical for scale 0..16),
so on the only tiny model this host has, a wrongly restored KV cannot change the text. The cache accounting is still exact:
before the fix a request under adapter configuration B was served KV that another slot had computed under configuration A
(cache_n = 84 of 84 prompt tokens); after it, that request prefills the prompt itself (cache_n ~ 0). The text-equality test
below is kept for the day the adapter applies on this model; today it cannot fail and says so.

THE ADAPTER IS SYNTHETIC. No small trained adapter for a tiny model exists on this host (models/lora_adapters/legal/Qwen2.5-0.5B is a
140 MB rank-64 adapter whose base model, qwen2.5-0.5b, is not on disk; the PEFT dir of ggml-org/stories15M_MOE ships configs without
weights). The adapter is built here from random A/B matrices with a fixed seed ("random-weight approach"); it only has to exist, be
loadable, and give the server two distinguishable configurations (scale 0 vs scale 1).

FAIL-BEFORE / PASS-AFTER: run against a llama-server built before e8d4a1707 (for example build-cpu/bin/
llama-server-BASE.exe, base 6d3bbeb79) the *_not_shared_* tests fail with cache_n 84; against a build with the fix they pass. The
*_still_hits_* tests are the controls: they pass on both, so a cache that never hits cannot pass the other tests.
"""
import sys
import threading
from pathlib import Path

import pytest
from utils import ServerPreset, ServerProcess, use_local_stories260k

# stories260K shapes (llama arch): n_embd 64, n_head_kv * head_dim 32, n_ff 172, 5 blocks
_N_EMBD, _N_KV, _N_FF, _N_BLOCKS = 64, 32, 172, 5
_RANK, _ALPHA = 4, 16.0

P = ('Once upon a time there was a little girl named Lily. She loved to play in the garden with her red ball. '
     'One sunny day she saw a big brown dog near the old tree and she said hello. The dog wagged its tail and they became friends.')
Q = 'Tom had a small blue car and he drove it to the park every morning before school started and the sun came out.'

OFF = [{"id": 0, "scale": 0.0}]
ON = [{"id": 0, "scale": 1.0}]

# a cache hit on this prompt is ~84 tokens; a miss leaves only an incidental 0-1 token coincidence (BOS)
HIT_MIN = 50
MISS_MAX = 5


def _make_synthetic_lora(path: Path) -> None:
    """Deterministic random LoRA adapter GGUF for stories260K (not trained)."""
    gguf_py = Path(__file__).resolve().parents[4] / "gguf-py"
    if str(gguf_py) not in sys.path:
        sys.path.insert(0, str(gguf_py))
    try:
        import numpy as np
        import gguf
    except Exception as e:  # visible skip, not a silent pass
        pytest.skip(f"cannot build the synthetic LoRA adapter (needs numpy + gguf-py): {e}")
    rng = np.random.default_rng(20260929)
    targets = {
        "attn_q": (_N_EMBD, _N_EMBD), "attn_k": (_N_EMBD, _N_KV), "attn_v": (_N_EMBD, _N_KV), "attn_output": (_N_EMBD, _N_EMBD),
        "ffn_gate": (_N_EMBD, _N_FF), "ffn_up": (_N_EMBD, _N_FF), "ffn_down": (_N_FF, _N_EMBD),
    }
    w = gguf.GGUFWriter(str(path), arch="llama")
    w.add_type(gguf.GGUFType.ADAPTER)
    w.add_string(gguf.Keys.Adapter.TYPE, "lora")
    w.add_float32(gguf.Keys.Adapter.LORA_ALPHA, _ALPHA)
    for blk in range(_N_BLOCKS):
        for name, (n_in, n_out) in targets.items():
            # ggml ne [n_in, r] is numpy shape (r, n_in); ne [r, n_out] is numpy shape (n_out, r)
            w.add_tensor(f"blk.{blk}.{name}.weight.lora_a", (rng.standard_normal((_RANK, n_in)) * 0.5).astype(np.float32))
            w.add_tensor(f"blk.{blk}.{name}.weight.lora_b", (rng.standard_normal((n_out, _RANK)) * 0.5).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.fixture(scope="module")
def lora_path(tmp_path_factory) -> str:
    p = tmp_path_factory.mktemp("lora_cache_key") / "synth_lora_stories260k.gguf"
    _make_synthetic_lora(p)
    assert p.stat().st_size > 10_000
    return str(p)


def _server(lora_path: str, **kw) -> ServerProcess:
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.lora_files = [lora_path]
    server.n_slots = 2
    server.n_ctx = kw.pop("n_ctx", 2048)
    server.n_batch = 512
    server.n_predict = 24
    server.cache_ram = kw.pop("cache_ram", 256)
    for k, v in kw.items():
        setattr(server, k, v)
    return server


def _comp(server: ServerProcess, prompt: str, n_predict: int, slot: int, lora=None) -> dict:
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "seed": 1, "cache_prompt": True, "id_slot": slot}
    if lora is not None:
        body["lora"] = lora
    res = server.make_request("POST", "/completion", data=body)
    assert res.status_code == 200, res.body
    assert res.body.get("id_slot") == slot, f"ran on slot {res.body.get('id_slot')}, expected {slot} (the slot choreography did not hold)"
    return res.body


def _choreo(server: ServerProcess, lora_a, lora_c) -> dict:
    """W warms slot 1 with an unrelated prompt UNDER THE SAME CONFIGURATION AS C (so slot 1 is non-empty, the LRU slot, and
    launch_slot_with_task sees no configuration change for C -- it would otherwise drop slot 1's prompt itself and hide the
    stale host-cache restore; measured: with W under a different configuration the pre-fix binary passes the forward test).
    A runs P on slot 0 under lora_a;
    its prompt is published to the host prompt cache at the end of prefill. C then runs the SAME prompt P pinned to slot 1
    under lora_c and must be served from that entry only if the adapter configuration is the same."""
    _comp(server, Q, 1, 1, lora_c)
    _comp(server, P, 8, 0, lora_a)
    return _comp(server, P, 24, 1, lora_c)


def test_host_cache_not_shared_across_lora_configs(lora_path):
    """Entry computed with the adapter ON (server-wide, scale 1) must not be restored into a request that runs it OFF."""
    server = _server(lora_path)
    server.start()
    c = _choreo(server, None, OFF)
    assert c["timings"]["cache_n"] < MISS_MAX, (
        f"request under adapter scale 0 reused {c['timings']['cache_n']} tokens of KV computed under scale 1 "
        f"(prompt_n={c['timings']['prompt_n']}): the prompt cache is not keyed by the LoRA configuration")


def test_host_cache_not_shared_across_lora_configs_reverse(lora_path):
    """The other direction: entry computed with the adapter OFF (per-request scale 0) must not serve a request that runs ON."""
    server = _server(lora_path)
    server.start()
    c = _choreo(server, OFF, None)
    assert c["timings"]["cache_n"] < MISS_MAX, (
        f"request under the server-wide adapter reused {c['timings']['cache_n']} tokens of KV computed under scale 0")


def test_host_cache_still_hits_for_the_same_lora_config(lora_path):
    """CONTROL (must pass before AND after the fix): same configuration on both sides still shares the entry."""
    server = _server(lora_path)
    server.start()
    c = _choreo(server, None, None)
    assert c["timings"]["cache_n"] > HIT_MIN, f"same-config request got cache_n={c['timings']['cache_n']}: the cache hit is gone"


def test_host_cache_hit_when_explicit_scale_equals_server_wide(lora_path):
    """CONTROL: the configuration is compared by value. Per-request [{id 0, scale 1.0}] is the same configuration as the
    server-wide adapter (scale 1.0), so it must hit the entry A published."""
    server = _server(lora_path)
    server.start()
    c = _choreo(server, None, ON)
    assert c["timings"]["cache_n"] > HIT_MIN, (
        f"explicit scale 1.0 == server-wide scale 1.0 but cache_n={c['timings']['cache_n']}: configuration compared by identity, not value")


def test_scale_zero_request_text_matches_a_fresh_server(lora_path):
    """The text a scale-0 request produces after the adapter-on entry exists equals what a fresh server produces for it.
    HONEST LIMIT: on stories260K the adapter does not change the text at all (see module docstring), so this cannot fail
    on this model, before or after the fix; test_host_cache_not_shared_across_lora_configs carries the verdict. It becomes a real
    wrong-text check once the adapter applies on the model."""
    server = _server(lora_path)
    server.start()
    after_on = _choreo(server, None, OFF)["content"]
    server.stop()
    fresh = _server(lora_path)
    fresh.start()
    fresh_text = _comp(fresh, P, 24, 1, OFF)["content"]
    assert after_on == fresh_text


# ---- radix cache (--radix-cache --kv-unified, host prompt cache disabled so only the radix tree can serve the reuse) ----

SHARED_PREFIX = "The quick brown fox jumps over the lazy dog. " * 40


def _radix_run(lora_path: str, lora_a, lora_b) -> dict:
    """A (SHARED_PREFIX + cat) streams on slot 0 under lora_a; once its first token arrives (its prompt is in the radix tree)
    B (SHARED_PREFIX + dog) runs on the never-used slot 1 under lora_b. Same shape as test_radix_prefix_cache.py."""
    server = _server(lora_path, n_ctx=4096, cache_ram=0, extra_args=["--radix-cache", "--kv-unified"])
    server.start()
    first = threading.Event()
    result = {}

    def run_a():
        chunks = 0
        body = {"prompt": SHARED_PREFIX + " Tell me a story about a cat.", "seed": 42, "temperature": 0.0, "n_predict": 32,
                "id_slot": 0, "stream": True}
        if lora_a is not None:
            body["lora"] = lora_a
        try:
            for _ in server.make_stream_request("POST", "/completion", data=body):
                chunks += 1
                first.set()
        finally:
            result["chunks"] = chunks
            first.set()

    t = threading.Thread(target=run_a)
    t.start()
    assert first.wait(timeout=60), "A produced no streamed token within 60 s"
    body_b = {"prompt": SHARED_PREFIX + " Tell me a story about a dog.", "seed": 42, "temperature": 0.0, "n_predict": 8, "id_slot": 1}
    if lora_b is not None:
        body_b["lora"] = lora_b
    res_b = server.make_request("POST", "/completion", data=body_b)
    t.join(timeout=60)
    server.stop()
    assert result.get("chunks", 0) > 0
    assert res_b.status_code == 200 and res_b.body.get("id_slot") == 1
    return res_b.body


def test_radix_cache_not_shared_across_lora_configs(lora_path):
    b = _radix_run(lora_path, None, OFF)
    assert b["timings"]["cache_n"] < MISS_MAX, (
        f"radix tree served {b['timings']['cache_n']} tokens computed under the server-wide adapter to a scale-0 request")


def test_radix_cache_still_hits_for_the_same_lora_config(lora_path):
    """CONTROL: without it a radix cache that never hits at all would pass the test above."""
    b = _radix_run(lora_path, None, None)
    assert b["timings"]["cache_n"] > 100, f"same-config radix request got cache_n={b['timings']['cache_n']}: the radix hit is gone"
