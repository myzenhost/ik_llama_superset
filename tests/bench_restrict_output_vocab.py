#!/usr/bin/env python3
"""Decode speed of llama-server with and without --restrict-output-vocab, on a random-weight Llama with a large vocabulary.

   python tests/bench_restrict_output_vocab.py <tree-root> <llama-server> [--vocab 128000] [--hidden 512] [--layers 2] [--allowed 32] [--rounds 3] [--reps 5]

Not a correctness test (tests/test-output-vocab-subset.cpp and tools/server/tests/unit/test_restrict_output_vocab.py are). Prints, for each variant, the
per-token decode time reported by the server over rounds x reps requests, in an interleaved order (OFF, ON, OFF, ON, ...) so drift of a shared machine hits
both. A speed claim from this output needs the run count and the spread, both printed. The model is synthetic: the saving is the output projection's share of
a decode step, which for this shape is printed too (output weight elements / all weight elements).
"""
import json
import statistics
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
server_bin = sys.argv[2]
args = sys.argv[3:]


def opt(name, default):
    return type(default)(args[args.index(name) + 1]) if name in args else default


vocab, hidden, layers, n_allowed, rounds, reps = opt("--vocab", 128000), opt("--hidden", 512), opt("--layers", 2), opt("--allowed", 32), opt("--rounds", 3), opt("--reps", 5)
sys.path.insert(0, str(root / "tests"))
sys.argv = [sys.argv[0], str(root), server_bin]

import test_ct2_gguf_e2e as E  # noqa: E402  (Server launcher)
import make_synth_llama_gguf as G  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        gguf = Path(tmp) / "synth.gguf"
        G.make(gguf, vocab=vocab, hidden=hidden, layers=layers, heads=8, ffn=hidden * 2, seed=1)
        ffn = hidden * 2
        per_layer = 4 * hidden * hidden + 3 * hidden * ffn
        total = layers * per_layer + 2 * vocab * hidden
        print(f"model: vocab {vocab} hidden {hidden} layers {layers}; output weight = {vocab * hidden / 1e6:.1f}M of {total / 1e6:.1f}M weight elements "
              f"({100 * vocab * hidden / (layers * per_layer + vocab * hidden):.1f}% of the matrices a decode step multiplies)")
        allowed = list(range(1000, 1000 + n_allowed))
        prompt = [1] + list(range(100, 107))
        results = {False: [], True: []}
        order = []
        for r in range(rounds):
            order += [False, True] if r % 2 == 0 else [True, False]
        tokens = {}
        for flag in order:
            srv = E.Server(gguf, *(["--restrict-output-vocab"] if flag else []), "-t", "4", slots=1)
            try:
                for rep in range(reps + 1):   # the first request of a server process is a warm-up and is dropped
                    out = srv.post("/completion", {"prompt": prompt, "n_predict": 48, "temperature": 0.0, "allowed_token_ids": allowed, "cache_prompt": False,
                                                   "ignore_eos": True, "return_tokens": True}, timeout=600)
                    if rep:
                        results[flag].append(out["timings"]["predicted_ms"] / out["timings"]["predicted_n"])
                    tokens[flag] = out["tokens"]
            finally:
                srv.close()
        for flag in (False, True):
            xs = results[flag]
            print(f"--restrict-output-vocab {'ON ' if flag else 'OFF'}: {statistics.mean(xs):7.3f} ms/token  sd {statistics.pstdev(xs):.3f}  min {min(xs):.3f}  max {max(xs):.3f}  "
                  f"(n = {len(xs)} requests of 48 tokens, {rounds} server runs per variant)")
        print(f"ratio OFF/ON of the means: {statistics.mean(results[False]) / statistics.mean(results[True]):.3f}")
        print("same tokens on both:", tokens[False] == tokens[True])
        print(json.dumps({"off": results[False], "on": results[True]}))


main()
