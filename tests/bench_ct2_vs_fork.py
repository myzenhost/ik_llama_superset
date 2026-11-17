#!/usr/bin/env python3
"""CTranslate2 (pip wheel 4.x, CPU) against this fork's llama-server on the SAME random-weight Llama, same threads, same machine.

   python tests/bench_ct2_vs_fork.py <tree-root> <llama-server> <llama-quantize> [--hidden 1024] [--layers 12] [--threads 4] [--reps 5]

The model is one Hugging Face Llama with random weights. CTranslate2 side: its own converter (float32 and int8), Generator(compute_type=...).
Fork side: convert_hf_to_gguf.py f32, and llama-quantize Q8_0 from it. Measured, each (rep + 1 warm-up dropped) reps times, mean and sd:
  A  decode, one sequence:   ms per generated token (CT2: wall time of generate_batch / tokens; fork: the server's timings.predicted_ms / predicted_n)
  B  prefill, 256 tokens:     ms per prompt token (CT2: generate_batch with max_length 1; fork: timings.prompt_ms / prompt_n)
  C  four sequences at once:  aggregate generated tokens per second, as the three scheduling models do it
       CT2 batched      inter_threads=1, intra_threads=T, one generate_batch of 4 prompts
       CT2 replicas     inter_threads=4, intra_threads=T/4, four asynchronous generate_batch calls (ReplicaPool, include/ctranslate2/replica_pool.h)
       fork slots       llama-server -np 4, four concurrent requests (continuous batching in one graph)
Accuracy of the two int8 schemes on the same weights is in docs/agent-reports/engine-ctranslate2-logs/int8_scheme_error.txt.
Nothing here is a claim about another CPU: the numbers are for this machine, printed with its flags.
"""
import concurrent.futures
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

root = Path(sys.argv[1]).resolve()
server_bin, quantize_bin = sys.argv[2], sys.argv[3]
args = sys.argv[4:]


def opt(name, default):
    return type(default)(args[args.index(name) + 1]) if name in args else default


HIDDEN, LAYERS, THREADS, REPS = opt("--hidden", 1024), opt("--layers", 12), opt("--threads", 4), opt("--reps", 5)
sys.path.insert(0, str(root / "tests"))
sys.argv = [sys.argv[0], str(root), server_bin]

import test_ct2_gguf_e2e as E  # noqa: E402

import torch  # noqa: E402
from ctranslate2.converters import TransformersConverter  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

np = E.np


def stats(xs):
    return f"{statistics.mean(xs):9.3f} sd {statistics.pstdev(xs):7.3f} (n={len(xs)})"


def main():
    flags = ""
    for line in open("/proc/cpuinfo"):
        if line.startswith("flags"):
            flags = " ".join(f for f in line.split() if f in ("avx2", "fma", "avx512f", "avx512_vnni", "avx_vnni", "f16c"))
            break
    print(f"machine: {[l.split(':')[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')][0]}; flags: {flags}; threads {THREADS}")
    with tempfile.TemporaryDirectory() as tmp:
        hf = Path(tmp) / "hf"
        vocab = E.make_tokenizer(hf)
        cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=HIDDEN, intermediate_size=int(HIDDEN * 2.75) // 64 * 64, num_hidden_layers=LAYERS, num_attention_heads=HIDDEN // 64,
                          num_key_value_heads=max(1, HIDDEN // 256), max_position_embeddings=1024, rms_norm_eps=1e-5, rope_theta=10000.0, bos_token_id=1, eos_token_id=2,
                          tie_word_embeddings=False)
        torch.manual_seed(0)
        m = LlamaForCausalLM(cfg)
        for p in m.parameters():
            torch.nn.init.normal_(p, 0, 0.02)
        m.save_pretrained(hf)
        print(f"model: hidden {HIDDEN} layers {LAYERS} ffn {cfg.intermediate_size} heads {cfg.num_attention_heads}/{cfg.num_key_value_heads}, {sum(p.numel() for p in m.parameters()) / 1e6:.0f}M parameters")
        ct2 = {}
        for q in ("float32", "int8"):
            ct2[q] = Path(tmp) / ("ct2-" + q)
            TransformersConverter(str(hf)).convert(str(ct2[q]), quantization=q, force=True)
        gg = {"f32": Path(tmp) / "m-f32.gguf", "q8_0": Path(tmp) / "m-q8_0.gguf"}
        r = subprocess.run([sys.executable, str(root / "convert_hf_to_gguf.py"), str(hf), "--outfile", str(gg["f32"]), "--outtype", "f32"], capture_output=True, text=True, cwd=str(root))
        assert r.returncode == 0, r.stderr[-500:]
        r = subprocess.run([quantize_bin, str(gg["f32"]), str(gg["q8_0"]), "Q8_0"], capture_output=True, text=True)
        assert gg["q8_0"].exists(), r.stderr[-500:]

        rng = np.random.default_rng(1)
        short = [1] + [int(x) for x in rng.integers(3, len(vocab), size=7)]
        long_ = [1] + [int(x) for x in rng.integers(3, len(vocab), size=255)]
        N = 64
        out = {}

        def ct2_run(q, inter, intra, fn):
            gen = E.ctranslate2.Generator(str(ct2[q]), device="cpu", compute_type=q, inter_threads=inter, intra_threads=intra)
            try:
                return fn(gen)
            finally:
                del gen

        for q in ("float32", "int8"):
            # A decode
            def dec(gen):
                xs = []
                for i in range(REPS + 1):
                    t0 = time.perf_counter()
                    res = gen.generate_batch([[vocab[t] for t in short]], beam_size=1, max_length=N, min_length=N, include_prompt_in_result=False)
                    dt = time.perf_counter() - t0
                    assert len(res[0].sequences_ids[0]) == N
                    if i:
                        xs.append(1000 * dt / N)
                return xs

            def pre(gen):
                xs = []
                for i in range(REPS + 1):
                    t0 = time.perf_counter()
                    gen.generate_batch([[vocab[t] for t in long_]], beam_size=1, max_length=1, include_prompt_in_result=False)
                    dt = time.perf_counter() - t0
                    if i:
                        xs.append(1000 * dt / len(long_))
                return xs

            def batched(gen):
                xs = []
                for i in range(REPS + 1):
                    t0 = time.perf_counter()
                    gen.generate_batch([[vocab[t] for t in short]] * 4, beam_size=1, max_length=N, min_length=N, include_prompt_in_result=False)
                    dt = time.perf_counter() - t0
                    if i:
                        xs.append(4 * N / dt)
                return xs

            def replicas(gen):
                xs = []
                for i in range(REPS + 1):
                    t0 = time.perf_counter()
                    futs = [gen.generate_batch([[vocab[t] for t in short]], beam_size=1, max_length=N, min_length=N, include_prompt_in_result=False, asynchronous=True)[0] for _ in range(4)]
                    for f in futs:
                        f.result()
                    dt = time.perf_counter() - t0
                    if i:
                        xs.append(4 * N / dt)
                return xs

            out[f"CT2 {q}"] = {"A": ct2_run(q, 1, THREADS, dec), "B": ct2_run(q, 1, THREADS, pre), "C-batched": ct2_run(q, 1, THREADS, batched),
                               "C-replicas": ct2_run(q, 4, max(1, THREADS // 4), replicas)}
        for name, path in (("fork f32", gg["f32"]), ("fork Q8_0", gg["q8_0"])):
            res = {"A": [], "B": [], "C-slots": []}
            srv = E.Server(path, "-t", str(THREADS), "-ctk", "f16", "-ctv", "f16", slots=4, ctx_per_slot=512)
            try:
                for i in range(REPS + 1):
                    o = srv.post("/completion", {"prompt": short, "n_predict": N, "temperature": 0.0, "ignore_eos": True, "cache_prompt": False}, timeout=900)
                    if i:
                        res["A"].append(o["timings"]["predicted_ms"] / o["timings"]["predicted_n"])
                for i in range(REPS + 1):
                    o = srv.post("/completion", {"prompt": long_, "n_predict": 1, "temperature": 0.0, "ignore_eos": True, "cache_prompt": False}, timeout=900)
                    if i:
                        res["B"].append(o["timings"]["prompt_ms"] / o["timings"]["prompt_n"])
                for i in range(REPS + 1):
                    t0 = time.perf_counter()
                    with concurrent.futures.ThreadPoolExecutor(4) as ex:
                        futs = [ex.submit(srv.post, "/completion", {"prompt": short, "n_predict": N, "temperature": 0.0, "ignore_eos": True, "cache_prompt": False}, 900) for _ in range(4)]
                        for f in futs:
                            f.result()
                    dt = time.perf_counter() - t0
                    if i:
                        res["C-slots"].append(4 * N / dt)
            finally:
                srv.close()
            out[name] = res
        print()
        print(f"A decode, one sequence (ms per generated token, lower is better)")
        for k, v in out.items():
            if "A" in v:
                print(f"   {k:12s} {stats(v['A'])}")
        print(f"B prefill, 256 prompt tokens (ms per prompt token, lower is better)")
        for k, v in out.items():
            if "B" in v:
                print(f"   {k:12s} {stats(v['B'])}")
        print(f"C four sequences at once (aggregate generated tokens/s, higher is better)")
        for k, v in out.items():
            for sub in ("C-batched", "C-replicas", "C-slots"):
                if sub in v:
                    print(f"   {k:12s} {sub:11s} {stats(v[sub])}")
        print(json.dumps(out))


main()
