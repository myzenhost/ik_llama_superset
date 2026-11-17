# TurboQuant KV (turbo2/turbo3) had fused CUDA flash-attention kernels (ggml/src/ggml-cuda/fattn-turbo.cuh) but NO
# parser accepted the names -- neither common/arg.cpp kv_cache_types nor common/common.cpp kv_cache_type_from_str --
# so `-ctk turbo3` answered "Invalid cache type" and the kernels were unreachable from any CLI.
# Turbo KV is GPU-only by design (no CPU traits), so on a CPU-only run the right answer is a clean refusal at context
# creation naming what is needed -- before the fix of llama-context.cpp it aborted mid-request in ops.cpp dup_to_q.
# The GPU serve case runs only with LLAMA_TEST_GPU=1 (one GPU tenant at a time on this host).
import os
import subprocess
import pytest
from utils import ServerProcess

# Needs n_embd_head_k divisible by the turbo block (128): qwen2.5:1.5b has head_dim 128. tinyllama (head_dim 64) is
# refused at context creation ("K cache type turbo3 with block size 128 does not divide n_embd_head_k=64"), correctly.
MODEL = os.environ.get("LLAMA_TEST_HD128_GGUF",
    "models/ollama/blobs/sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4")
SERVER = os.environ.get("LLAMA_SERVER_BIN_PATH", "")


def _run_cpu_only(ctk, ctv, port):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    p = subprocess.run([SERVER, "-m", MODEL, "--port", str(port), "-c", "1024", "-np", "1", "-ngl", "0", "-fa", "on",
                        "--cache-type-k", ctk, "--cache-type-v", ctv],
                       env=env, capture_output=True, text=True, timeout=300, creationflags=0x08000000 if os.name == "nt" else 0)
    return p.returncode, p.stdout + p.stderr


@pytest.mark.skipif(not os.path.exists(MODEL) or not os.path.exists(SERVER), reason="model or LLAMA_SERVER_BIN_PATH missing")
@pytest.mark.parametrize("ctk,ctv,port", [("turbo3", "turbo3", 18471), ("q8_0", "turbo3", 18472), ("turbo2", "turbo2", 18473)])
def test_turbo_kv_names_parse_and_cpu_refuses_cleanly(ctk, ctv, port):
    rc, log = _run_cpu_only(ctk, ctv, port)
    assert "Invalid cache type" not in log, log[-2000:]          # the parsers accept the names
    assert "TurboQuant KV cache" in log, log[-2000:]             # refused at context creation, with the reason
    assert "not implemented" not in log, log[-2000:]             # and never reaches the CPU dup_to_q abort
    assert rc != 0


@pytest.mark.skipif(not os.path.exists(MODEL) or not os.path.exists(SERVER), reason="model or LLAMA_SERVER_BIN_PATH missing")
def test_turbo4_kv_is_refused_by_the_parser():
    rc, log = _run_cpu_only("turbo4", "turbo4", 18474)
    assert "Invalid cache type" in log, log[-2000:]
    assert rc != 0


@pytest.mark.skipif(os.environ.get("LLAMA_TEST_GPU") != "1" or not os.path.exists(MODEL), reason="GPU serve case: set LLAMA_TEST_GPU=1")
@pytest.mark.parametrize("ctk,ctv", [("turbo3", "turbo3"), ("q8_0", "turbo3"), ("turbo2", "turbo2")])
def test_turbo_kv_serves_on_gpu(ctk, ctv):
    s = ServerProcess()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = MODEL
    s.n_ctx = 1024
    s.n_slots = 1
    s.fa = "on"
    s.extra_args = ["-ngl", "99", "--cache-type-k", ctk, "--cache-type-v", ctv]
    s.start(timeout_seconds=300)
    try:
        res = s.make_request("POST", "/completion", data={"prompt": "The capital of France is", "temperature": 0.0,
                                                           "n_predict": 8})
        assert res.status_code == 200, res.body
        assert "Paris" in res.body["content"], res.body["content"]
    finally:
        s.stop()
