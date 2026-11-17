#!/usr/bin/env python3
# SINGLE-SHOT-OK: one repo's tokenizer files, one vocab-only conversion and one encode pass over a 46-entry corpus: under two minutes, re-run is idempotent.
"""Make one HF-derived tokenizer golden for tests/test-tokenizer-0: a vocab-only GGUF + the shared .inp corpus + the HF ids as .out.

    python tests/make-hf-vocab-golden.py <name> <hf-repo> [--revision SHA] [--tok-dir DIR] [--no-download]

What it does, in order (every step prints what it did):
  1. downloads ONLY the tokenizer files of <hf-repo> (tokenizer.json, tokenizer.model, tokenizer_config.json, vocab/merges, special-token maps,
     config.json, generation_config.json, chat template files; never weights) into <tok-dir>/<name>, and writes the repo's commit sha and the
     sha256 of every downloaded file into <tok-dir>/<name>/SOURCE.json (the standing rule for downloads: vet + record the sha);
  2. converts them with this tree's convert_hf_to_gguf.py --vocab-only to models/ggml-vocab-<name>.gguf;
  3. copies the corpus (models/ggml-vocab-llama-bpe.gguf.inp: the multilingual corpus every golden shares, so one defect shows on every vocab)
     to models/ggml-vocab-<name>.gguf.inp and writes models/ggml-vocab-<name>.gguf.out with the ids of the HF AutoTokenizer
     (add_special_tokens=False), one corpus entry per line -- the same format convert_hf_to_gguf_update.py writes.
The HF `tokenizers` library is the oracle on purpose: it is independent of llama.cpp's own tables (see the row text).

Row: S69-A-PICK-TOKENIZER-HF-GOLDEN-COVERAGE-FOR-SERVED-PRETOKENIZER-PRESETS-2026-10-05-01.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json", "vocab.txt", "merges.txt", "special_tokens_map.json",
         "added_tokens.json", "config.json", "generation_config.json", "chat_template.json", "chat_template.jinja", "tekken.json",
         "spiece.model", "tiktoken.model", "tokenization_*.py", "*.tiktoken", "sentencepiece.bpe.model"]
SEP = "\n__ggml_vocab_test__\n"
SPECIAL_TEXT = "Hi there"  # tests/test-tokenizer-0.cpp encodes the same text for the .special check


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("repo")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--tok-dir", default=os.path.join(ROOT, "models", "tokenizers"))
    ap.add_argument("--no-download", action="store_true", help="the tokenizer files are already in <tok-dir>/<name>")
    ap.add_argument("--trust-remote-code", action="store_true", help="the HF tokenizer needs its repo's tokenization_*.py (reads that code: only for repos you have read)")
    ap.add_argument("--config-repo", default=None, help="tokenizer-only repo (e.g. alvarobartt/grok-2-tokenizer): fetch config.json alone from this model repo "
                                                       "so the converter can detect the architecture; its sha and sha256 go into SOURCE.json")
    ap.add_argument("--special-only", action="store_true", help="only (re)write models/ggml-vocab-<name>.gguf.special from <tok-dir>/<name>: "
                                                                 "no download, no conversion, .inp/.out untouched")
    a = ap.parse_args()
    d = os.path.join(a.tok_dir, a.name)
    gguf = os.path.join(ROOT, "models", "ggml-vocab-%s.gguf" % a.name)
    if a.special_only:
        return write_special(gguf, d, a.trust_remote_code)
    os.makedirs(d, exist_ok=True)
    if not a.no_download:
        from huggingface_hub import HfApi, snapshot_download
        info = HfApi().model_info(a.repo, revision=a.revision)
        snapshot_download(a.repo, revision=info.sha, allow_patterns=FILES, local_dir=d)
        print("downloaded %s @ %s -> %s" % (a.repo, info.sha, d), flush=True)
        src = {"repo": a.repo, "revision": info.sha, "files": {}}
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            if os.path.isfile(p) and fn != "SOURCE.json":
                src["files"][fn] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
        with open(os.path.join(d, "SOURCE.json"), "w") as f:
            json.dump(src, f, indent=1)
    if a.config_repo:
        from huggingface_hub import HfApi, hf_hub_download
        csha = HfApi().model_info(a.config_repo).sha
        cfg = hf_hub_download(a.config_repo, "config.json", revision=csha, local_dir=d)
        src_path = os.path.join(d, "SOURCE.json")
        src = json.load(open(src_path)) if os.path.exists(src_path) else {"files": {}}
        src["config_repo"] = {"repo": a.config_repo, "revision": csha, "file": "config.json", "sha256": sha256(cfg), "bytes": os.path.getsize(cfg)}
        with open(src_path, "w") as f:
            json.dump(src, f, indent=1)
        print("config.json from %s @ %s -> %s" % (a.config_repo, csha, cfg), flush=True)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "convert_hf_to_gguf.py"), d, "--outfile", gguf, "--vocab-only"]
                       + (["--trust-remote-code"] if a.trust_remote_code else []),  # PROGRESS-OK: one vocab-only conversion
                       cwd=ROOT, timeout=900)
    if r.returncode != 0 or not os.path.exists(gguf):
        print("FAIL: convert_hf_to_gguf.py --vocab-only rc=%d (no golden written)" % r.returncode)
        return 3
    inp = os.path.join(ROOT, "models", "ggml-vocab-llama-bpe.gguf.inp")
    # The corpus blob is LF, but a core.autocrlf=true checkout writes it with CRLF; test-tokenizer-0 reads it in text
    # mode (CRLF -> LF on Windows), so split the LF form too. Read raw, CRLF would leave the whole file as one entry.
    text = open(inp, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    tests = text.split(SEP)
    if tests and tests[-1] == "":
        tests = tests[:-1]
    if len(tests) < 2:
        print("FAIL: the corpus %s split into %d entries (separator not found); no golden written" % (inp, len(tests)))
        return 4
    # a golden may carry entries of its own after the shared corpus (afmoe's digit-stage strings, dd6681d1e): keep them,
    # or a regeneration silently drops that coverage
    if os.path.exists(gguf + ".inp"):
        own = open(gguf + ".inp", encoding="utf-8", newline="").read().replace("\r\n", "\n").split(SEP)
        extra = [t for t in own if t != "" and t not in tests]
        if extra:
            print("keeping %d entr%s of %s.inp that the shared corpus lacks" % (len(extra), "y" if len(extra) == 1 else "ies", gguf))
        tests += extra
    tok = load_tokenizer(d, a.trust_remote_code)
    with open(gguf + ".inp", "w", encoding="utf-8", newline="") as f:
        for t in tests:
            f.write(t + SEP)
    with open(gguf + ".out", "w", newline="") as f:
        for t in tests:
            ids = tok.encode(t, add_special_tokens=False)
            f.write("".join(" %d" % i for i in ids) + "\n")
    print("OK %s: %d corpus entries, golden %s" % (a.name, len(tests), gguf))
    return write_special(gguf, d, a.trust_remote_code, tok)


def load_tokenizer(d, trust_remote_code):
    # the converter's loader: AutoTokenizer, plus its fallbacks (a transformers-v5 tokenizer_config on transformers 4.x, and
    # the byte-level Llama override of transformers 5), so the oracle loads every tokenizer the conversion could load
    sys.path.insert(0, ROOT)
    import conversion.base
    from conversion.base import load_hf_tokenizer
    conversion.base.ALLOW_REMOTE_CODE = trust_remote_code  # load_hf_tokenizer honours trust_remote_code only with this set
    return load_hf_tokenizer(d, trust_remote_code=trust_remote_code)


def write_special(gguf, d, trust_remote_code, tok=None):
    # SINGLE-SHOT-OK: two encodes of one short text, one 2-line file
    # .special: the ids HF puts around SPECIAL_TEXT when add_special_tokens is on (its BOS/EOS), as "prefix <ids>" and
    # "suffix <ids>". The .out corpus is encoded with add_special_tokens off, so without this a wrong add_bos/add_eos in the
    # GGUF passes every golden (falcon-h1 prepended a BOS HF does not, with all goldens passing).
    # Without tokenizer_config.json HF falls back to its class defaults (a LlamaTokenizerFast then prepends "<s>", which in
    # deepseek-llm is an ordinary added token, not its BOS), so the oracle would be the class, not the model: write nothing.
    if not os.path.exists(os.path.join(d, "tokenizer_config.json")):
        print("SKIP %s.special: %s has no tokenizer_config.json, so HF's special tokens are class defaults, not the model's" % (gguf, d))
        return 0
    tok = tok or load_tokenizer(d, trust_remote_code)
    plain = tok.encode(SPECIAL_TEXT, add_special_tokens=False)
    full = tok.encode(SPECIAL_TEXT)
    at = next((i for i in range(len(full) - len(plain) + 1) if full[i:i + len(plain)] == plain), None)
    if not plain or at is None:
        print("FAIL: %r with special tokens %s does not contain the plain encoding %s; no .special written" % (SPECIAL_TEXT, full, plain))
        return 5
    prefix, suffix = full[:at], full[at + len(plain):]
    with open(gguf + ".special", "w", newline="") as f:
        f.write("prefix" + "".join(" %d" % i for i in prefix) + "\n")
        f.write("suffix" + "".join(" %d" % i for i in suffix) + "\n")
    print("OK %s.special: prefix %s suffix %s" % (gguf, prefix, suffix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
