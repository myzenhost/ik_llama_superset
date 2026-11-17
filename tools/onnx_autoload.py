"""Run any ik_llama fork binary directly against a .onnx model.

MODEL-CHOICE-OK: DEC-S68-QWEN3-BEATS-BGEM3-REVERSAL stands; nothing here selects an embedder.
This is a launcher and is model-agnostic.

WHAT THIS IS FOR. The user's directive was "load ONNX models into our fork". The proven converter
(cortex/scripts/onnx_gguf_convert.py) does the hard part, but it leaves the user holding a manual
step: convert first, then remember which GGUF came from which ONNX. This closes that gap without
any C++ risk -- you point a fork binary at a .onnx and it runs:

    onnx_autoload.py --bin build-canonical/bin/llama-server.exe -- -m model.onnx --embedding

It finds `-m <something>.onnx`, converts it ONCE into a cached GGUF, rewrites the argument, and
execs the real binary. Conversion is a one-time cost per model: the cache is keyed on the source's
size and mtime AND on the sidecar's, so a re-export invalidates it and an unchanged model never
reconverts.

WHY THE SIDECAR IS PART OF THE CACHE KEY. A large ONNX keeps its weights in model.onnx_data and
the .onnx holds only pointers. Re-exporting can rewrite the 2 GB sidecar while the small .onnx
keeps its size and even its mtime, so keying on the .onnx alone would happily serve a GGUF built
from the PREVIOUS weights -- silently, and with no error anywhere. Both files are stat'd.

THIS IS NOT the in-process reader. It is the cheap path to the same user-visible behaviour, and it
is deliberately honest about that: the fork still loads a GGUF, this just makes one on demand. The
real reader is scoped on S69-ONNX-STAGE1-IN-PROCESS-READER-NOT-BUILT-SCOPED-01.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]        # <root>/portable/ik_llama_fork/tools/this.py
CONVERTER = ROOT / "cortex" / "scripts" / "onnx_gguf_convert.py"
DEFAULT_CACHE = ROOT / "temp" / "onnx_gguf_cache"
CREATE_NO_WINDOW = 0x08000000


def _stat_key(path):
    st = os.stat(path)
    return "%d:%d" % (st.st_size, int(st.st_mtime))


def source_fingerprint(onnx_path):
    """Fingerprint the .onnx AND every external sidecar it points at."""
    import onnx as onnx_mod

    parts = [_stat_key(onnx_path)]
    model = onnx_mod.load(str(onnx_path), load_external_data=False)
    locs = set()
    for t in model.graph.initializer:
        if t.data_location == onnx_mod.TensorProto.EXTERNAL:
            for e in t.external_data:
                if e.key == "location":
                    locs.add(e.value)
    base = os.path.dirname(os.path.abspath(onnx_path))
    for loc in sorted(locs):
        p = os.path.join(base, loc)
        if not os.path.exists(p):
            raise IOError("external sidecar %s is MISSING; the .onnx alone carries no weights"
                          % loc)
        parts.append("%s=%s" % (loc, _stat_key(p)))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16], sorted(locs)


def find_hf_dir(onnx_path, explicit=None):
    """Locate the config.json + tokenizer that ONNX does not carry.

    ONNX holds the graph and the weights but NOT llama.cpp's hparams and NOT the tokenizer, so
    they have to come from beside the model. Looked for in the .onnx's own directory first (many
    exports drop a config.json in onnx/), then the parent, which is the usual HF layout.
    """
    if explicit:
        return Path(explicit)
    d = Path(onnx_path).resolve().parent
    for cand in (d, d.parent):
        if (cand / "config.json").is_file():
            return cand
    raise SystemExit("no config.json found beside %s or in its parent; pass --hf-dir. ONNX does "
                     "not carry hparams or the tokenizer, so they must come from somewhere."
                     % onnx_path)


def ensure_gguf(onnx_path, hf_dir, cache_dir, outtype="f32", force=False):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp, sidecars = source_fingerprint(onnx_path)
    stem = Path(onnx_path).stem
    gguf = cache_dir / ("%s.%s.%s.gguf" % (stem, outtype, fp))
    meta = cache_dir / ("%s.%s.%s.json" % (stem, outtype, fp))

    if gguf.is_file() and meta.is_file() and not force:
        print("[onnx_autoload] cache HIT  %s" % gguf, flush=True)
        return gguf

    print("[onnx_autoload] cache MISS -> converting %s" % onnx_path, flush=True)
    if sidecars:
        print("[onnx_autoload]   external sidecar(s) in the key: %s" % sidecars, flush=True)
    log = cache_dir / ("%s.%s.convert.log" % (stem, fp))
    args = [sys.executable, str(CONVERTER), "--onnx", str(onnx_path),
            "--hf-dir", str(hf_dir), "--outfile", str(gguf), "--outtype", outtype]
    with open(log, "w", encoding="utf-8", errors="replace") as lf:
        lf.write("$ %s\n" % " ".join(args))
        lf.flush()
        rc = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT,
                              creationflags=CREATE_NO_WINDOW).wait()
    if rc != 0 or not gguf.is_file():
        tail = open(log, encoding="utf-8", errors="replace").read()[-3000:]
        raise SystemExit("[onnx_autoload] conversion FAILED rc=%s; see %s\n%s" % (rc, log, tail))
    meta.write_text(json.dumps({"onnx": str(onnx_path), "hf_dir": str(hf_dir),
                                "outtype": outtype, "fingerprint": fp,
                                "sidecars": sidecars}, indent=2), encoding="utf-8")
    print("[onnx_autoload] converted -> %s (%.1f MB)"
          % (gguf, gguf.stat().st_size / 1e6), flush=True)
    return gguf


def rewrite_args(argv, gguf):
    out = list(argv)
    for i, a in enumerate(out):
        if a in ("-m", "--model") and i + 1 < len(out):
            out[i + 1] = str(gguf)
            return out
    raise SystemExit("no -m/--model argument to rewrite")


def find_onnx(argv):
    for i, a in enumerate(argv):
        if a in ("-m", "--model") and i + 1 < len(argv):
            if str(argv[i + 1]).lower().endswith(".onnx"):
                return argv[i + 1]
            return None
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Run a fork binary against a .onnx by converting to a cached GGUF first.")
    ap.add_argument("--bin", required=True, help="the fork binary to exec")
    ap.add_argument("--hf-dir", default=None,
                    help="dir holding config.json + tokenizer; auto-detected beside the .onnx")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--outtype", default="f32", choices=["f32", "f16"])
    ap.add_argument("--force", action="store_true", help="reconvert even on a cache hit")
    ap.add_argument("--print-only", action="store_true",
                    help="resolve and convert, print the command, do not exec")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="arguments for the binary, after --")
    a = ap.parse_args()

    rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
    onnx_path = find_onnx(rest)
    if not onnx_path:
        print("[onnx_autoload] no .onnx in -m; passing through unchanged", flush=True)
        cmd = [a.bin] + rest
    else:
        hf_dir = find_hf_dir(onnx_path, a.hf_dir)
        print("[onnx_autoload] hf-dir: %s" % hf_dir, flush=True)
        gguf = ensure_gguf(onnx_path, hf_dir, a.cache_dir, a.outtype, a.force)
        cmd = [a.bin] + rewrite_args(rest, gguf)

    print("[onnx_autoload] exec: %s" % " ".join(cmd), flush=True)
    if a.print_only:
        return 0
    # CREATE_NO_WINDOW keeps rule 1.45 (windows stay hidden): stdout/stderr are INHERITED
    # handles, so the binary's output still reaches this process's console -- the flag only
    # suppresses allocating a NEW console window for the child.
    return subprocess.Popen(cmd, creationflags=CREATE_NO_WINDOW).wait()


if __name__ == "__main__":
    sys.exit(main())
