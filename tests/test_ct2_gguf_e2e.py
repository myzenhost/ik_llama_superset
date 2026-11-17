#!/usr/bin/env python3
"""A CTranslate2 model directory, converted by convert_hf_to_gguf.py (--ct2-tokenizer-dir), run by llama-server, against the REAL
CTranslate2 engine (pip `ctranslate2`) running the SAME model directory.

   python tests/test_ct2_gguf_e2e.py <tree-root> <llama-server>

For each CT2 weight format (float32, float16, bfloat16, int8, int8_float16, int16) one random-weight Llama-family model (GQA: 4 q heads, 2 kv
heads) is written as model.bin by the independent writer of tests/test_ct2_to_hf.py. CTranslate2 loads it with compute_type=float32, which
dequantises the stored weights exactly as the engine does at load time (src/models/model.cc:280-345); its score_batch gives the log-probability
of every token given the ones before (src/scoring.cc:6-70). The fork's side: convert_hf_to_gguf.py <ct2 dir> --ct2-tokenizer-dir ... --outtype f32,
llama-server prompt_logprobs for the same token ids. The two must agree to 1e-4 on every token: a wrong QKV split, a wrong dequantisation, a
head-count mix-up, a rotary layout or a norm epsilon would each show as an error of order 1.
Controls: (a) the log-probabilities are not all equal (a constant model cannot pass), (b) the same-seed int8 model differs from the float32 one under the real
engine (the formats are not interchangeable), (c) both engines score the same token count. Mutations of the converter (k/v swap, rope base,
epsilon) each fail 7 of 8 checks, see docs/agent-reports/engine-ctranslate2-logs/ct2e2e-mutation-*.txt.
"""
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

root = Path(sys.argv[1]).resolve()
server_bin = sys.argv[2]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "tests"))
sys.argv = [sys.argv[0], str(root)]

import numpy as np  # noqa: E402

import test_ct2_to_hf as T  # noqa: E402  (the independent model.bin writer)

try:
    import ctranslate2  # noqa: E402
    import sentencepiece as spm  # noqa: E402
except Exception as e:
    print("SKIP: %s" % e)
    sys.exit(0)

check = T.check


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, gguf, *extra, slots=1, ctx_per_slot=256):
        self.port = free_port()
        self.log = tempfile.NamedTemporaryFile("w+", suffix=".srv.log")
        self.p = subprocess.Popen([server_bin, "-m", str(gguf), "--host", "127.0.0.1", "--port", str(self.port), "-c", str(ctx_per_slot * slots), "-np", str(slots),
                                   "-ngl", "0", "--no-webui", "-ctk", "f32", "-ctv", "f32", *extra], stdout=self.log, stderr=subprocess.STDOUT, text=True)
        t0 = time.time()
        while time.time() - t0 < 60:
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/health" % self.port, timeout=1).read()
                return
            except Exception:
                if self.p.poll() is not None:
                    raise RuntimeError("llama-server exited: " + self.tail())
                time.sleep(0.2)
        raise RuntimeError("llama-server did not start")

    def tail(self, n=1500):
        self.log.flush()
        return Path(self.log.name).read_text()[-n:]

    def post(self, path, data, timeout=120):
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path), json.dumps(data).encode(), {"Content-Type": "application/json"})
        try:
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except TimeoutError:
            raise RuntimeError("timeout for %s; server log tail:\n%s" % (json.dumps(data)[:300], self.tail()))
        except urllib.error.HTTPError as e:
            raise RuntimeError("HTTP %d for %s: %s" % (e.code, json.dumps(data)[:300], e.read().decode()[:400]))

    def close(self):
        self.p.terminate()
        try:
            self.p.wait(10)
        except Exception:
            self.p.kill()


def make_tokenizer(d, V=60):
    d.mkdir()
    corpus = d.parent / "corpus.txt"
    corpus.write_text("\n".join("the quick brown fox %d jumps over the lazy dog %d" % (i, i * 7) for i in range(200)))
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(d / "tokenizer"), vocab_size=V, model_type="bpe",
                                   character_coverage=1.0, bos_id=1, eos_id=2, unk_id=0, pad_id=-1, minloglevel=2)
    (d / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "LlamaTokenizer"}))
    sp = spm.SentencePieceProcessor(model_file=str(d / "tokenizer.model"))
    return [sp.id_to_piece(i) for i in range(sp.vocab_size())]


def run_scheme(scheme, tmp, tokdir, vocab, ids):
    T.RNG = np.random.default_rng(99)
    wr, orig, cfg, _v, dims = T.make_model(scheme, n_heads=4, n_kv=2, hidden=32, ffn=64, vocab=len(vocab), layers=2, rotary_interleave=0, w_std=0.25)
    cfg.update({"bos_token": vocab[1], "eos_token": vocab[2], "unk_token": vocab[0], "layer_norm_epsilon": 1e-5})
    ct2 = Path(tmp) / ("ct2-" + scheme + "")
    wr.write(ct2, config=cfg, vocab=vocab)
    gguf = Path(tmp) / ("m-" + scheme + "" + ".gguf")
    env = dict(os.environ)
    r = subprocess.run([sys.executable, str(root / "convert_hf_to_gguf.py"), str(ct2), "--ct2-tokenizer-dir", str(tokdir), "--outfile", str(gguf),
                        "--outtype", "f32"], capture_output=True, text=True, cwd=str(root), env=env)
    check("[%s] convert_hf_to_gguf.py converts the CT2 directory" % scheme, r.returncode == 0 and gguf.exists(), (r.stderr or r.stdout)[-700:])
    if not gguf.exists():
        return None
    gen = ctranslate2.Generator(str(ct2), device="cpu", compute_type="float32")
    ref = gen.score_batch([[vocab[i] for i in ids]])[0].log_probs   # log P(ids[k] | ids[:k]) for k = 1..n-1
    srv = Server(gguf)
    try:
        res = srv.post("/completion", {"prompt": ids, "n_predict": 1, "temperature": 0.0, "prompt_logprobs": True, "cache_prompt": False})
    finally:
        srv.close()
    got = res["prompt_logprobs"]["logprobs"][1:]
    check("[%s] same number of scored tokens" % scheme, len(got) == len(ref) == len(ids) - 1, "%d vs %d" % (len(got), len(ref)))
    err = max(abs(a - b) for a, b in zip(got, ref))
    check("[%s] fork log-probs == real CTranslate2 (max err %.2e)" % (scheme, err), err < 1e-4, "got=%s ref=%s" % (got[:4], ref[:4]))
    return ref, got


def main():
  with tempfile.TemporaryDirectory() as tmp:
    tokdir = Path(tmp) / "tok"
    vocab = make_tokenizer(tokdir)
    rng = np.random.default_rng(5)
    ids = [1] + [int(x) for x in rng.integers(3, len(vocab), size=14)]
    results = {}
    for scheme in ("float32", "float16", "bfloat16", "int8", "int8_float16", "int16"):
        results[scheme] = run_scheme(scheme, tmp, tokdir, vocab, ids)
    ref = results["float32"][0] if results["float32"] else None
    if ref:
        check("control (a): the log-probabilities are not constant", max(ref) - min(ref) > 0.05, str(ref[:5]))
        # the 6 formats hold DIFFERENT weights for the same seed (lossy), so their log-probs differ from float32's: the test is not vacuous
        d = max(abs(a - b) for a, b in zip(results["int8"][0], ref))
        check("control: the int8 model differs from the float32 one under the REAL engine (%.3g)" % d, d > 1e-4)



if __name__ == "__main__":
    main()
    print()
    print("FAILED: " + ", ".join(T.failures) if T.failures else "all checks passed")
    sys.exit(1 if T.failures else 0)
