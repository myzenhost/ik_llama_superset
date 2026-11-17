import random
import pytest
from utils import ServerPreset, ServerProcess, use_local_stories260k

# NVMe tier under the host prompt cache: --cache-disk-dir / --cache-disk-size / --cache-disk-min-tokens /
# --cache-disk-write-budget (row S69-TIERED-KV-DISK-TIER-SLICE1-2026-10-05-01; unit tests without a model:
# tests/probe-disk-tier.cpp and tests/test-server-prompt-cache-disk.cpp).
#
# Fail-before: none of the four flags was registered, so passing one made llama-server refuse to start and
# ServerProcess.start() raised RuntimeError. The end-to-end test also has its own control (the same traffic without the
# flag restores nothing), so it shows the tier is what produced the restore.

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.temperature = 0.0
    server.n_slots = 1          # every new, different prompt moves the slot's old prompt into the cache
    server.server_metrics = True
    server.n_predict = 2


def metrics(server: ServerProcess) -> dict:
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    out = {}
    for line in res.body.splitlines():
        if line.startswith("llamacpp:") and "{" not in line:
            name, value = line.split(" ", 1)
            out[name] = float(value)
    return out


def make_prompts(n: int, n_words: int = 40) -> list:
    rng = random.Random(7)
    words = ["lily", "tom", "park", "ball", "tree", "sun", "dog", "cat", "red", "big", "happy", "ran", "jump", "play"]
    return [f"story {i}: " + " ".join(rng.choice(words) for _ in range(n_words)) for i in range(n)]


def complete(server: ServerProcess, prompt: str) -> dict:
    res = server.make_request("POST", "/completion", data={"prompt": prompt, "cache_prompt": True, "n_predict": 2})
    assert res.status_code == 200
    return res.body["timings"]


def test_cache_disk_flags_accepted_and_dir_prepared(tmp_path):
    """Before the port the flags were unknown arguments (start() raised). Now the server starts, creates the directory,
    removes stale pcache-* files from a crashed earlier run, leaves anything else alone, and exports the tier's metrics."""
    global server
    d = tmp_path / "spill" / "nested"
    d.mkdir(parents=True)
    stale = d / "pcache-7.bin"
    stale.write_bytes(b"left by a crashed run")
    keep = d / "keep.txt"
    keep.write_text("not ours")

    server.extra_args = ["--cache-ram", "1", "--cache-disk-dir", str(d), "--cache-disk-size", "64",
                         "--cache-disk-min-tokens", "1", "--cache-disk-write-budget", "0"]
    server.start()
    assert not stale.exists(), "a stale pcache-* file must be removed at startup"
    assert keep.exists(), "files that are not ours must be left alone"
    complete(server, "once upon a time")
    m = metrics(server)
    for name in ["spill_total", "written_bytes_total", "restore_total", "restore_failed_total", "checksum_failed_total",
                 "refused_cap_total", "refused_budget_total", "refused_space_total", "used_bytes", "limit_bytes", "entries"]:
        assert f"llamacpp:prompt_cache_disk_{name}" in m, name
    # /metrics prints doubles with 6 significant digits, so 67108864 arrives as 6.71089e+07
    assert abs(m["llamacpp:prompt_cache_disk_limit_bytes"] - 64 * 1024 * 1024) < 1000


def test_no_flag_no_disk_metrics():
    """Control: without --cache-disk-dir the metrics are exactly what they were."""
    global server
    server.extra_args = ["--cache-ram", "1"]
    server.start()
    complete(server, "once upon a time")
    assert not any(k.startswith("llamacpp:prompt_cache_disk") for k in metrics(server))


def test_evicted_prompt_is_restored_from_disk(tmp_path):
    global server
    d = tmp_path / "tier"
    prompts = make_prompts(24)

    # 1 MiB of cache cannot hold 24 prompts, so the oldest are evicted; with the tier they go to disk
    server.extra_args = ["--cache-ram", "1", "--cache-disk-dir", str(d), "--cache-disk-min-tokens", "1"]
    server.start()
    for p in prompts:
        complete(server, p)
    m = metrics(server)
    assert m["llamacpp:prompt_cache_disk_spill_total"] > 0, "nothing was spilled although the RAM cache overflowed"
    assert m["llamacpp:prompt_cache_disk_written_bytes_total"] > 0
    assert m["llamacpp:prompt_cache_disk_used_bytes"] > 0
    assert len(list(d.glob("pcache-*.bin"))) == m["llamacpp:prompt_cache_disk_entries"]

    t = complete(server, prompts[0])      # the oldest prompt, now only on disk
    n_total = t["prompt_n"] + t["cache_n"]
    print(f"RESTORE RESULT tier on: cache_n={t['cache_n']} prompt_n={t['prompt_n']} of {n_total}; "
          f"spilled={m['llamacpp:prompt_cache_disk_spill_total']:.0f} written={m['llamacpp:prompt_cache_disk_written_bytes_total']:.0f} B")
    assert t["cache_n"] >= n_total // 2, f"the spilled prompt was not reused: cache_n={t['cache_n']} of {n_total}"
    m2 = metrics(server)
    assert m2["llamacpp:prompt_cache_disk_restore_total"] >= 1
    assert m2["llamacpp:prompt_cache_disk_read_bytes_total"] > 0
    assert m2["llamacpp:prompt_cache_disk_checksum_failed_total"] == 0
    assert m2["llamacpp:prompt_cache_disk_restore_failed_total"] == 0


def test_control_same_traffic_without_the_tier_restores_nothing():
    global server
    prompts = make_prompts(24)
    server.extra_args = ["--cache-ram", "1"]
    server.start()
    for p in prompts:
        complete(server, p)
    t = complete(server, prompts[0])
    n_total = t["prompt_n"] + t["cache_n"]
    print(f"RESTORE RESULT tier off (control): cache_n={t['cache_n']} prompt_n={t['prompt_n']} of {n_total}")
    # a few tokens (BOS and the common "story " start) match any earlier prompt; a restore would be about n_total - 1
    assert t["cache_n"] < n_total // 4, f"without the tier the evicted prompt must be recomputed, cache_n={t['cache_n']} of {n_total}"


def test_corrupted_spill_file_falls_back_to_a_recompute(tmp_path):
    """A flipped byte on disk must never reach the model: the request is answered from a plain prefill, the failure is counted."""
    global server
    d = tmp_path / "tier"
    prompts = make_prompts(24)
    server.extra_args = ["--cache-ram", "1", "--cache-disk-dir", str(d), "--cache-disk-min-tokens", "1"]
    server.start()
    for p in prompts:
        complete(server, p)
    files = sorted(d.glob("pcache-*.bin"))
    assert files, "nothing was spilled"
    for f in files:                       # damage every spilled blob, so whichever one is chosen is bad
        raw = bytearray(f.read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        f.write_bytes(bytes(raw))

    t = complete(server, prompts[0])
    n_total = t["prompt_n"] + t["cache_n"]
    print(f"RESTORE RESULT corrupted blobs: cache_n={t['cache_n']} prompt_n={t['prompt_n']} of {n_total}")
    assert t["cache_n"] < n_total // 4, f"a corrupted blob was used: cache_n={t['cache_n']} of {n_total}"
    m = metrics(server)
    assert m["llamacpp:prompt_cache_disk_checksum_failed_total"] >= 1
    assert m["llamacpp:prompt_cache_disk_restore_failed_total"] >= 1


def test_disk_cap_is_honoured(tmp_path):
    global server
    d = tmp_path / "tier"
    server.extra_args = ["--cache-ram", "1", "--cache-disk-dir", str(d), "--cache-disk-size", "1",
                         "--cache-disk-min-tokens", "1"]
    server.start()
    for p in make_prompts(40):
        complete(server, p)
    m = metrics(server)
    assert m["llamacpp:prompt_cache_disk_used_bytes"] <= 1024 * 1024
    assert sum(f.stat().st_size for f in d.glob("pcache-*.bin")) <= 1024 * 1024
