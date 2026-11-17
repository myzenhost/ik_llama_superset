"""Run any ik_llama fork binary directly against a .safetensors model.

MODEL-CHOICE-OK: nothing here selects a model or an embedder. This is a launcher and is
model-agnostic -- it runs whatever the caller pointed -m at.

WHAT THIS IS FOR. The fork's loader is GGUF-ONLY. Measured 2026-09-02 on this tree: zero
occurrences of 'safetensors' anywhere in src/, ggml/, common/ or examples/ C/C++ (positive
control gguf_init_from_file matched 5 files on the same scope, so the search reached the
corpus), and `llama-cli -m model.safetensors` dies with

    gguf_init_from_file: invalid magic characters ...
    llama_model_load: error loading model: llama_model_loader: failed to load model from ...

This closes that gap the same way tools/onnx_autoload.py closed the ONNX one, and with the same
idiom, so the fork gains ONE non-GGUF-autoload pattern rather than two rival ones:

    safetensors_autoload.py --bin build-canonical/bin/llama-cli.exe -- -m model.safetensors -p hi

It finds `-m <something>.safetensors`, converts it ONCE into a cached GGUF, rewrites the
argument, and execs the real binary. Conversion is a one-time cost per model.

WHY CONVERT INSTEAD OF READING SAFETENSORS IN-PROCESS. safetensors is a pure tensor container:
an 8-byte header length, a JSON header of {name: {dtype, shape, data_offsets}}, then raw bytes.
It carries NO architecture, NO rope/norm hyper-parameters, NO tokenizer and NO llama.cpp tensor
naming. Every single thing the fork's loader actually needs lives in config.json and
tokenizer.json beside it, and the code that turns those into GGUF metadata -- per-architecture
tensor remapping, vocab construction, MoE/MLA/tied-embedding special cases -- is the 6753 lines
of convert_hf_to_gguf.py already in this tree. A C++ reader would have to reimplement all of it
inside the loader of a fork whose value is its divergent SOTA quant path. So the weights get
converted by the code that already knows how, and the user-visible behaviour (`-m foo.safetensors`
just works) is identical.

WHY THE FULL SHARD SET IS PART OF THE CACHE KEY. A sharded model is model-0000K-of-0000N.safetensors
plus model.safetensors.index.json. Keying on the file named in -m alone would happily serve a GGUF
built from a PREVIOUS export whenever any OTHER shard was rewritten -- silently, with no error.
Every resolved shard is stat'd, and so are config.json and the tokenizer files, because the GGUF's
metadata and vocab are derived from those and not from the weights.

WHY THE SHARD SET IS RESOLVED HERE AND NOT LEFT TO THE CONVERTER. convert_hf_to_gguf.py discovers
parts by `filename.startswith("model") and filename.endswith(".safetensors")` and only falls back
to the index weight_map when that finds nothing. That is right for a clean HF checkout and wrong
in three real cases on this disk: a model whose single file is not named model*.safetensors (e.g.
flux-2-klein-9b.safetensors) discovers ZERO parts and silently falls through to the pytorch_model
.bin path; a directory holding both model.safetensors and a model-0000K-of-0000N set sweeps up
all of them; and an index.json naming shards that are not on disk reads SHORT. This tool resolves
the exact shard set from the index (or from the -of-0000N filename pattern, verifying the count),
refuses a short read, and stages a canonical directory of hardlinks only when the converter's own
discovery would disagree with that set.
"""
import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[3]        # <root>/portable/ik_llama_fork/tools/this.py
FORK = Path(__file__).resolve().parents[1]        # <root>/portable/ik_llama_fork
CONVERTER = FORK / "convert_hf_to_gguf.py"
LORA_CONVERTER = FORK / "convert_lora_to_gguf.py"
DEFAULT_CACHE = ROOT / "temp" / "safetensors_gguf_cache"
CREATE_NO_WINDOW = 0x08000000

SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.safetensors$")

# Files that carry hparams / vocab / chat template. The GGUF is derived from these as much as
# from the weights, so they belong in the cache key and in a staged directory.
SIDE_FILES = (
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "tokenizer.model", "vocab.json", "merges.txt", "special_tokens_map.json",
    "added_tokens.json", "chat_template.json", "chat_template.jinja",
    "preprocessor_config.json", "processor_config.json", "spiece.model",
)


# --------------------------------------------------------------------------------------------
# safetensors header (no dependency on the safetensors package -- this must work before any
# conversion env is involved, and it is how the dtype coverage below is reported honestly)
# --------------------------------------------------------------------------------------------
def _log(msg, flush=True):
    """Banner/progress output. stderr, so a caller piping stdout gets the model's output only.

    `flush` is accepted (and always honoured) so call sites read the same as the print() they
    replaced; every log line here is flushed regardless.
    """
    print(msg, file=sys.stderr, flush=True)


def read_st_header(path):
    """Return the JSON header dict of a .safetensors file. Pure stdlib."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise IOError("%s is too short to be a safetensors file" % path)
        (n,) = struct.unpack("<Q", raw)
        size = os.path.getsize(path)
        if n <= 0 or n + 8 > size:
            raise IOError("%s: header length %d is impossible for a %d byte file "
                          "(not a safetensors file?)" % (path, n, size))
        head = f.read(n)
    try:
        return json.loads(head.decode("utf-8"))
    except Exception as e:
        raise IOError("%s: safetensors header is not valid JSON (%s)" % (path, e))


def dtype_histogram(shards):
    """{dtype: (count, n_elements)} across every shard, from the headers alone."""
    hist = {}
    for s in shards:
        for name, spec in read_st_header(s).items():
            if name == "__metadata__" or not isinstance(spec, dict):
                continue
            dt = spec.get("dtype", "?")
            n = 1
            for d in spec.get("shape", []) or [1]:
                n *= d
            c, e = hist.get(dt, (0, 0))
            hist[dt] = (c + 1, e + n)
    return hist


def _stat_key(path):
    st = os.stat(path)
    return "%d:%d" % (st.st_size, int(st.st_mtime))


# --------------------------------------------------------------------------------------------
# shard resolution
# --------------------------------------------------------------------------------------------
def resolve_shards(model_path):
    """Resolve the COMPLETE shard set for the safetensors file (or dir) named in -m.

    Returns (model_dir, [shard Paths in load order], index_json_path or None).
    Raises on a short read rather than converting part of a model.
    """
    p = Path(model_path).resolve()

    if p.is_dir():
        # -m <dir>: prefer an index, else the single/greatest model*.safetensors set.
        idx = _find_index(p)
        if idx is not None:
            return (p,) + _shards_from_index(idx)
        cands = sorted(x for x in p.glob("*.safetensors"))
        if not cands:
            raise SystemExit("[safetensors_autoload] no .safetensors in directory %s" % p)
        grouped = _group_by_shard_stem(cands)
        if len(grouped) != 1:
            raise SystemExit("[safetensors_autoload] %s holds %d distinct safetensors models "
                             "(%s); point -m at one file, not the directory"
                             % (p, len(grouped), ", ".join(sorted(grouped))))
        return (p, list(grouped.values())[0], None)

    if not p.is_file():
        raise SystemExit("[safetensors_autoload] %s does not exist" % p)

    d = p.parent
    m = SHARD_RE.match(p.name)

    idx = _find_index(d, prefer_stem=(m.group("stem") if m else p.stem))
    if idx is not None:
        _, shards, _ = (d,) + _shards_from_index(idx)
        if p.resolve() in [s.resolve() for s in shards]:
            return d, shards, idx
        # An index exists but does not list the file we were pointed at -- that index describes
        # a DIFFERENT model in the same directory. Fall through and resolve from the name.

    if m:
        total = int(m.group("total"))
        stem = m.group("stem")
        shards = sorted(d.glob("%s-?????-of-%s.safetensors" % (stem, m.group("total"))))
        if len(shards) != total:
            raise SystemExit(
                "[safetensors_autoload] SHORT READ REFUSED: %s says there are %d shards but only "
                "%d are on disk in %s (%s). Converting this would silently produce a model with "
                "missing tensors."
                % (p.name, total, len(shards), d, ", ".join(s.name for s in shards)))
        return d, shards, None

    return d, [p], None


def _find_index(d, prefer_stem=None):
    cands = sorted(d.glob("*.safetensors.index.json"))
    if not cands:
        return None
    if prefer_stem:
        for c in cands:
            if c.name == "%s.safetensors.index.json" % prefer_stem:
                return c
    for c in cands:
        if c.name == "model.safetensors.index.json":
            return c
    return cands[0]


def _safe_shard_name(n):
    """True when an index.json weight_map value is a bare file name (no directory part, not absolute, not '..')."""
    # also from the second port of this row (work/fork-triage-unclear-03-shardguard 88e5ac94f6): a NUL byte, and a name that
    # is absolute under Windows rules even when the host's own rules say it is not (PureWindowsPath)
    return bool(n) and n not in (".", "..") and Path(n).name == n and "/" not in n and "\\" not in n \
        and not os.path.isabs(n) and ":" not in n and "\0" not in n and not PureWindowsPath(n).is_absolute()


def _shards_from_index(idx):
    data = json.loads(Path(idx).read_text(encoding="utf-8"))
    wm = data.get("weight_map")
    if not isinstance(wm, dict) or not wm:
        raise SystemExit("[safetensors_autoload] %s has no usable weight_map" % idx)
    names = sorted(set(str(v) for v in wm.values()))
    d = Path(idx).parent
    # one check (two ports of the row met in batch b1114f; the second's refusal was the same test, so it is folded into
    # _safe_shard_name above)
    bad = [n for n in names if not _safe_shard_name(n)]
    if bad:
        raise SystemExit(
            "[safetensors_autoload] %s names a shard outside its model dir (path traversal refused): %s"
            % (idx, ", ".join(bad)))
    missing = [n for n in names if not (d / n).is_file()]
    if missing:
        raise SystemExit(
            "[safetensors_autoload] SHORT READ REFUSED: %s lists %d shards, %d are MISSING from "
            "%s (%s). Converting this would silently produce a model with missing tensors."
            % (idx, len(names), len(missing), d, ", ".join(missing)))
    return [d / n for n in names], Path(idx)


def _group_by_shard_stem(paths):
    groups = {}
    for p in paths:
        m = SHARD_RE.match(p.name)
        key = m.group("stem") if m else p.stem
        groups.setdefault(key, []).append(p)
    return {k: sorted(v) for k, v in groups.items()}


# --------------------------------------------------------------------------------------------
# what the converter would find on its own -- staging is only needed when it disagrees
# --------------------------------------------------------------------------------------------
def converter_would_find(model_dir):
    """Reproduce convert_hf_to_gguf.py Model.__init__ part discovery, exactly."""
    names = [f for f in sorted(os.listdir(model_dir))
             if f.startswith("model") and f.endswith(".safetensors")]
    if names:
        return [Path(model_dir) / n for n in names]
    idx = Path(model_dir) / "model.safetensors.index.json"
    if idx.is_file():
        data = json.loads(idx.read_text(encoding="utf-8"))
        wm = data.get("weight_map") or {}
        names = sorted(set(str(v) for v in wm.values()))
        return [Path(model_dir) / n for n in names if _safe_shard_name(n) and (Path(model_dir) / n).is_file()]
    return []


def _link_or_copy(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)                     # NTFS hardlink: instant, no extra bytes
        return
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return
    except OSError:
        pass
    import shutil
    shutil.copy2(src, dst)


def stage_dir(model_dir, shards, cache_dir, fp):
    """Build a canonical HF directory the converter cannot mis-discover."""
    stage = Path(cache_dir) / "staging" / fp
    stage.mkdir(parents=True, exist_ok=True)
    if len(shards) == 1:
        _link_or_copy(shards[0], stage / "model.safetensors")
    else:
        n = len(shards)
        mapping = {}
        for i, s in enumerate(shards, 1):
            name = "model-%05d-of-%05d.safetensors" % (i, n)
            _link_or_copy(s, stage / name)
            mapping[s.name] = name
        # Rewrite the index so weight_map points at the canonical shard names. Without this the
        # converter opens the index, sees the OLD names, and reads nothing.
        src_idx = _find_index(Path(model_dir))
        if src_idx is not None:
            data = json.loads(Path(src_idx).read_text(encoding="utf-8"))
            wm = {k: mapping.get(str(v), str(v)) for k, v in (data.get("weight_map") or {}).items()}
            data["weight_map"] = wm
            (stage / "model.safetensors.index.json").write_text(
                json.dumps(data, indent=2), encoding="utf-8")
        else:
            # No index on disk: build one from the shard headers so the converter's multi-part
            # path (which requires the index) works.
            wm = {}
            for s, name in ((s, mapping[s.name]) for s in shards):
                for tname in read_st_header(s):
                    if tname != "__metadata__":
                        wm[tname] = name
            (stage / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"generated_by": "safetensors_autoload"},
                            "weight_map": wm}, indent=2), encoding="utf-8")
    for f in side_files(model_dir):
        _link_or_copy(f, stage / f.name)
    # Some HF models need their own modelling code for config parsing.
    for f in sorted(Path(model_dir).glob("*.py")):
        _link_or_copy(f, stage / f.name)
    return stage


def side_files(model_dir):
    d = Path(model_dir)
    out = []
    for name in SIDE_FILES:
        p = d / name
        if p.is_file():
            out.append(p)
    return out


# --------------------------------------------------------------------------------------------
# cache key + conversion
# --------------------------------------------------------------------------------------------
def source_fingerprint(model_dir, shards, index, outtype):
    parts = ["outtype=%s" % outtype]
    for s in shards:
        parts.append("%s=%s" % (s.name, _stat_key(s)))
    if index is not None:
        parts.append("%s=%s" % (index.name, _stat_key(index)))
    for f in side_files(model_dir):
        parts.append("%s=%s" % (f.name, _stat_key(f)))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def check_loadable(model_dir, allow_quantized):
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.is_file():
        raise SystemExit(
            "[safetensors_autoload] no config.json in %s. safetensors carries only tensors -- the "
            "architecture, hparams and tokenizer must be beside it. Pass --model-dir if they live "
            "elsewhere." % model_dir)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    if (Path(model_dir) / "adapter_config.json").is_file():
        raise SystemExit(
            "[safetensors_autoload] %s is a LoRA ADAPTER, not a model. -m cannot take one. Convert "
            "it with:\n    %s --base <base-model-dir> --outfile adapter.gguf %s\nthen pass the "
            "result to the binary with --lora." % (model_dir, LORA_CONVERTER, model_dir))

    # The tokenizer has to travel with the weights: safetensors holds no vocab, and a GGUF with
    # no vocab cannot be generated from -- convert_hf_to_gguf.py dies deep inside set_vocab()
    # with a message that says nothing about the missing file.
    d = Path(model_dir)
    have_tok = (
        (d / "tokenizer.json").is_file()
        or (d / "tokenizer.model").is_file()
        or (d / "spiece.model").is_file()
        or ((d / "vocab.json").is_file() and (d / "merges.txt").is_file())
    )
    if not have_tok:
        raise SystemExit(
            "[safetensors_autoload] no tokenizer beside the weights in %s (looked for "
            "tokenizer.json, tokenizer.model, spiece.model, vocab.json+merges.txt). safetensors "
            "carries tensors only; the GGUF's vocab comes from these files. This checkout is "
            "INCOMPLETE -- fetch the tokenizer, or pass --model-dir at a complete copy." % d)

    qc = cfg.get("quantization_config")
    if qc and not allow_quantized:
        raise SystemExit(
            "[safetensors_autoload] %s carries pre-quantized weights (quantization_config: %s). "
            "convert_hf_to_gguf.py reads FLOAT tensors (BF16/F16/F32) and cannot dequantize this "
            "on the way in -- the conversion would fail or, worse, misread the packed blocks. "
            "Use the unquantized checkpoint, or pass --allow-quantized to try anyway."
            % (model_dir, qc.get("quant_method", qc)))
    return cfg


PROBE_STRINGS = (
    "The capital of France is",
    "Hello world",
    "  leading and trailing  ",
    "def f(x):\n    return x + 1\n",
)


def verify_tokenizer(gguf, model_dir, bin_dir):
    """Tokenize probe strings with BOTH the HF tokenizer and the produced GGUF, compare IDS.

    This is the control that can fail. A converted model whose tensors are perfect still answers
    a DIFFERENT question if the prompt tokenizes differently, and nothing about the weights, the
    shapes or the file size can see that -- MEASURED here on a Llama-SPM model whose
    tokenizer_config.json carries "add_prefix_space": null: convert_hf_to_gguf.py writes that
    JSON null through as GGUF bool FALSE, so "The" tokenized as 1576 in the fork and 450 in
    transformers, and the greedy continuation diverged from the first token.

    Returns (status, detail). status is "ok", "mismatch", or "skipped:<why>".
    """
    tokbin = Path(bin_dir) / ("llama-tokenize.exe" if os.name == "nt" else "llama-tokenize")
    if not tokbin.is_file():
        return "skipped", "llama-tokenize not built at %s" % tokbin
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        return "skipped", "transformers unavailable (%s)" % type(e).__name__
    try:
        hf = AutoTokenizer.from_pretrained(str(model_dir))
    except Exception as e:
        return "skipped", "HF tokenizer would not load (%s: %s)" % (type(e).__name__, e)

    bad = []
    for probe in PROBE_STRINGS:
        r = subprocess.run([str(tokbin), "--model", str(gguf), "--prompt", probe],
                           capture_output=True, text=True, creationflags=CREATE_NO_WINDOW)
        if r.returncode != 0:
            return "skipped", "llama-tokenize rc=%s" % r.returncode
        got = []
        for line in r.stdout.splitlines():
            head = line.split("->")[0].strip()
            if head.lstrip("-").isdigit():
                got.append(int(head))
        want = hf(probe).input_ids
        if got != want:
            bad.append({"probe": probe, "gguf_ids": got, "hf_ids": want})
    if bad:
        return "mismatch", bad
    return "ok", "%d probe string(s) tokenize identically" % len(PROBE_STRINGS)


def _report_tokenizer(gguf, model_dir, bin_dir, verify_tok, meta_path=None):
    """Run the parity check and act on it: warn loudly, or refuse, per --verify-tokenizer."""
    if verify_tok == "off" or not bin_dir:
        return
    status, detail = verify_tokenizer(gguf, model_dir, bin_dir)
    if status == "ok":
        _log("[safetensors_autoload] tokenizer parity OK -- %s" % detail)
    elif status == "skipped":
        _log("[safetensors_autoload] tokenizer parity NOT CHECKED -- %s" % detail)
    else:
        msg = ("[safetensors_autoload] TOKENIZER MISMATCH: the GGUF tokenizes differently from "
               "the HF tokenizer that shipped with these weights. The model will answer a "
               "DIFFERENT prompt than you typed and no shape or size check can see it.\n" +
               json.dumps(detail, indent=2, ensure_ascii=False))
        if meta_path is not None:
            try:
                m = json.loads(Path(meta_path).read_text(encoding="utf-8"))
                m["tokenizer_parity"] = "MISMATCH"
                m["tokenizer_parity_detail"] = detail
                Path(meta_path).write_text(json.dumps(m, indent=2), encoding="utf-8")
            except Exception:
                _log("[safetensors_autoload] (could not record the mismatch in %s)" % meta_path,
                      flush=True)
        if verify_tok == "error":
            raise SystemExit(msg)
        _log(msg)


def ensure_gguf(model_path, cache_dir, outtype="auto", force=False, model_dir_override=None,
                allow_quantized=False, python=None, verbose=False, verify_tok="warn",
                bin_dir=None, early_return=None):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_dir, shards, index = resolve_shards(model_path)
    if model_dir_override:
        model_dir = Path(model_dir_override).resolve()

    check_loadable(model_dir, allow_quantized)

    hist = dtype_histogram(shards)
    _log("[safetensors_autoload] %d shard(s), dtypes: %s"
          % (len(shards), ", ".join("%s x%d" % (k, v[0]) for k, v in sorted(hist.items()))),
          flush=True)

    fp = source_fingerprint(model_dir, shards, index, outtype)
    stem = Path(model_dir).name or "model"
    gguf = cache_dir / ("%s.%s.%s.gguf" % (stem, outtype, fp))
    meta = cache_dir / ("%s.%s.%s.json" % (stem, outtype, fp))

    if early_return is not None and not force:
        # --quant: if the quantized file for this exact base is already cached, the full-precision base is not needed at all (it may have been
        # dropped with --drop-base)
        hit = early_return(gguf)
        if hit is not None:
            _log("[safetensors_autoload] quantized cache HIT  %s (base conversion skipped)" % hit)
            return hit

    if gguf.is_file() and meta.is_file() and not force:
        _log("[safetensors_autoload] cache HIT  %s" % gguf)
        _report_tokenizer(gguf, model_dir, bin_dir, verify_tok, meta)
        return gguf

    want = sorted(s.resolve() for s in shards)
    found = sorted(p.resolve() for p in converter_would_find(model_dir))
    if want == found:
        src_dir = model_dir
        staged = False
    else:
        _log("[safetensors_autoload] converter's own discovery would use %d file(s), we resolved "
              "%d -- staging canonical names" % (len(found), len(want)), flush=True)
        src_dir = stage_dir(model_dir, shards, cache_dir, fp)
        staged = True

    _log("[safetensors_autoload] cache MISS -> converting %s" % src_dir)
    log = cache_dir / ("%s.%s.convert.log" % (stem, fp))
    args = [python or sys.executable, str(CONVERTER), str(src_dir),
            "--outfile", str(gguf), "--outtype", outtype]
    if verbose:
        args.append("--verbose")
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    with open(log, "w", encoding="utf-8", errors="replace") as lf:
        lf.write("$ %s\n" % " ".join(args))
        lf.flush()
        rc = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, env=env,
                              creationflags=CREATE_NO_WINDOW).wait()
    if rc != 0 or not gguf.is_file():
        tail = open(log, encoding="utf-8", errors="replace").read()[-4000:]
        raise SystemExit("[safetensors_autoload] conversion FAILED rc=%s; see %s\n%s"
                         % (rc, log, tail))
    meta.write_text(json.dumps({
        "model_arg": str(model_path),
        "model_dir": str(model_dir),
        "staged_dir": str(src_dir) if staged else None,
        "shards": [s.name for s in shards],
        "index": index.name if index is not None else None,
        "side_files": [f.name for f in side_files(model_dir)],
        "dtypes": {k: {"tensors": v[0], "elements": v[1]} for k, v in hist.items()},
        "outtype": outtype,
        "fingerprint": fp,
    }, indent=2), encoding="utf-8")
    _log("[safetensors_autoload] converted -> %s (%.1f MB)"
          % (gguf, gguf.stat().st_size / 1e6), flush=True)
    _report_tokenizer(gguf, model_dir, bin_dir, verify_tok, meta)
    return gguf


# --------------------------------------------------------------------------------------------
# quantize-at-autoload (row S69-MISTRALRS-PICK-ISQAUTOLOAD-2026-09-28-01)
#
# mistral.rs loads a HF safetensors model straight into a chosen quant type ("ISQ", in-situ quantization). The idea
# transfers without any new quantizer: after the full-precision GGUF is cached, run the fork's own llama-quantize on it
# and cache THAT, so `-m model.safetensors --quant Q4_K_M` costs one conversion plus one quantization per model, and a
# second request for another type only repeats the cheap second step. Design source: mistralrs-quant/src/lib.rs:938-1026
# (MIT), idea only; no code copied.
#
# Nuances the row did not say:
#   1. The quantized file's cache key must include the llama-quantize BINARY (size+mtime) and the imatrix file, not only
#      the source: a rebuilt quantizer or a different importance matrix gives different bytes under the same type name.
#   2. llama-quantize needs the base GGUF to be full precision. A converter --outtype that already quantized (q8_0 ...)
#      cannot be requantized without --allow-requantize, so --quant with such an outtype is refused up front.
#   3. The output is written to <name>.part and renamed only after llama-quantize exits 0 AND the tensor types read back
#      from the file include the requested family: a killed run must not leave a half file that a later run takes as a hit.
#   4. The full-precision base stays cached by default (re-quantizing to another type is then cheap); E: runs near full, so
#      --drop-base removes it once the quantized file is in place.
#
# SAFE-DELETE-OK: the only files this section ever removes are its own derived cache artifacts: the <name>.part temp it
# just wrote, and with --drop-base the cached full-precision GGUF that the converter regenerates from the safetensors.
# REWRITE-SAFE: every write here goes to a NEW cache path (<base>.<QUANT>.<fp>.gguf / .json / .part); no store is read and rewritten.
# --------------------------------------------------------------------------------------------
_FULL_PRECISION_OUTTYPES = ("auto", "f32", "f16", "bf16")
_QUANT_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def quantize_binary(bin_dir, override=None):
    if override:
        p = Path(override)
        if p.is_file():
            return p
        raise SystemExit("[safetensors_autoload] --quantize-bin %s does not exist" % override)
    name = "llama-quantize.exe" if os.name == "nt" else "llama-quantize"
    for d in (bin_dir, FORK / "build-canonical" / "bin"):
        if d and (Path(d) / name).is_file():
            return Path(d) / name
    raise SystemExit("[safetensors_autoload] --quant needs llama-quantize; none found in %s or the canonical build. "
                     "Build it (target llama-quantize) or pass --quantize-bin." % bin_dir)


def quant_fingerprint(base_gguf, quant, qbin, imatrix):
    # the base is identified by NAME: it already carries the source fingerprint and the outtype, and keying on its name (not its mtime) lets a
    # quantized hit be served after --drop-base removed the base
    parts = ["quant=%s" % quant, "base=%s" % Path(base_gguf).name, "qbin=%s" % _stat_key(qbin)]
    if imatrix:
        parts.append("imatrix=%s" % _stat_key(imatrix))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def quant_paths(base_gguf, quant, qbin, imatrix):
    base_gguf = Path(base_gguf)
    out = base_gguf.with_name("%s.%s.%s.gguf" % (base_gguf.stem, quant, quant_fingerprint(base_gguf, quant, qbin, imatrix)))
    return out, out.with_suffix(".json")


def quant_hit(base_gguf, quant, qbin, imatrix):
    """The cached quantized file for this base/type/binary/imatrix, or None."""
    out, meta = quant_paths(base_gguf, quant, qbin, imatrix)
    return out if out.is_file() and meta.is_file() else None


def _tensor_type_names(gguf_path):
    """Set of ggml tensor type names in a GGUF, or None when gguf-py cannot read the file (unknown fork-only type)."""
    try:
        sys.path.insert(0, str(FORK / "gguf-py"))
        import gguf as _gguf
        rd = _gguf.GGUFReader(str(gguf_path), "r")
        return {t.tensor_type.name for t in rd.tensors}
    except Exception:
        return None


def ensure_quantized(base_gguf, quant, bin_dir, outtype="auto", qbin_override=None, imatrix=None, threads=0,
                     force=False, drop_base=False):
    if outtype not in _FULL_PRECISION_OUTTYPES:
        raise SystemExit("[safetensors_autoload] --quant %s needs a full-precision base, but --outtype %s already quantized "
                         "the conversion; use --outtype auto/f16/bf16/f32 with --quant." % (quant, outtype))
    if not _QUANT_NAME_RE.match(quant):
        raise SystemExit("[safetensors_autoload] --quant %r is not a quantization type name" % quant)
    base_gguf = Path(base_gguf)
    qbin = quantize_binary(bin_dir, qbin_override)
    if imatrix and not Path(imatrix).is_file():
        raise SystemExit("[safetensors_autoload] --imatrix %s does not exist" % imatrix)
    fp = quant_fingerprint(base_gguf, quant, qbin, imatrix)
    out, meta = quant_paths(base_gguf, quant, qbin, imatrix)
    if out.is_file() and meta.is_file() and not force:
        _log("[safetensors_autoload] quantized cache HIT  %s" % out)
    else:
        part = out.with_name(out.name + ".part")
        if part.exists():
            os.remove(part)
        cmd = [str(qbin)]
        if imatrix:
            cmd += ["--imatrix", str(imatrix)]
        cmd += [str(base_gguf), str(part), quant]
        if threads:
            cmd.append(str(threads))
        _log("[safetensors_autoload] quantizing -> %s" % out)
        log = out.with_suffix(".quantize.log")
        with open(log, "w", encoding="utf-8", errors="replace") as lf:
            lf.write("$ %s\n" % " ".join(cmd))
            lf.flush()
            rc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, creationflags=CREATE_NO_WINDOW).wait()
        if rc != 0 or not part.is_file():
            if part.exists():
                os.remove(part)
            tail = open(log, encoding="utf-8", errors="replace").read()[-3000:]
            raise SystemExit("[safetensors_autoload] llama-quantize FAILED rc=%s for type %s; see %s\n%s" % (rc, quant, log, tail))
        names = _tensor_type_names(part)
        family = "_".join(quant.upper().split("_")[:2])
        if names is not None and not any(n.startswith(family) for n in names):
            os.remove(part)
            raise SystemExit("[safetensors_autoload] llama-quantize exited 0 but the file holds tensor types %s and none of the family %s; "
                             "refusing to cache it as %s" % (sorted(names), family, quant))
        os.replace(part, out)
        meta.write_text(json.dumps({"base": base_gguf.name, "quant": quant, "fingerprint": fp, "quantize_bin": str(qbin),
                                    "imatrix": str(imatrix) if imatrix else None, "tensor_types": sorted(names) if names else None,
                                    "bytes": out.stat().st_size}, indent=2), encoding="utf-8")
        _log("[safetensors_autoload] quantized %s -> %s (%.1f MB from %.1f MB)" % (quant, out.name, out.stat().st_size / 1e6, base_gguf.stat().st_size / 1e6))
    if drop_base and base_gguf.is_file():
        os.remove(base_gguf)
        _log("[safetensors_autoload] --drop-base: removed %s" % base_gguf)
    return out


# --------------------------------------------------------------------------------------------
# arg plumbing (same shape as tools/onnx_autoload.py)
# --------------------------------------------------------------------------------------------
def relay_run(cmd):
    """Run the binary, relaying its output, with stdout and stderr KEPT SEPARATE.

    Two things force this. First, under pythonw.exe (rule 1.45: windows stay hidden) there is no
    console for a child to inherit and Python only fills STARTF_USESTDHANDLES when the streams are
    passed explicitly -- MEASURED: llama-cli ran to completion, exit 0, and printed nothing at all.
    Second, the streams must not be merged: llama.cpp writes its logs to STDERR and the generated
    text to STDOUT, and anything capturing this wrapper's stdout (a harness, a pipe, a caller
    reading the completion) must get the model's output alone -- MEASURED: merging them fed 4 KB of
    CUDA banner into a token-id comparison and scored a correct model 0/16.
    """
    import threading

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=CREATE_NO_WINDOW)

    def pump(src, dst):
        out = getattr(dst, "buffer", dst)
        while True:
            chunk = src.read(1)
            if not chunk:
                break
            out.write(chunk)
            if chunk in (b"\n", b"\r"):
                out.flush()
        out.flush()

    threads = [threading.Thread(target=pump, args=(proc.stdout, sys.stdout), daemon=True),
               threading.Thread(target=pump, args=(proc.stderr, sys.stderr), daemon=True)]
    for t in threads:
        t.start()
    rc = proc.wait()
    for t in threads:
        t.join(timeout=10)
    return rc


def find_safetensors(argv):
    for i, a in enumerate(argv):
        if a in ("-m", "--model") and i + 1 < len(argv):
            v = str(argv[i + 1])
            if v.lower().endswith(".safetensors"):
                return v
            p = Path(v)
            if p.is_dir() and any(p.glob("*.safetensors")):
                return v
            return None
    return None


def rewrite_args(argv, gguf):
    out = list(argv)
    for i, a in enumerate(out):
        if a in ("-m", "--model") and i + 1 < len(out):
            out[i + 1] = str(gguf)
            return out
    raise SystemExit("no -m/--model argument to rewrite")


def _autoload_resolve(a, st, bin_dir):
    """The GGUF to run: the converted base, or with the quant option the quantized file derived from it (both cached)."""
    if not a.quant:
        return ensure_gguf(st, a.cache_dir, a.outtype, a.force, a.model_dir, a.allow_quantized, a.python, a.verbose, a.verify_tokenizer, bin_dir)
    if a.outtype not in _FULL_PRECISION_OUTTYPES:
        raise SystemExit("[safetensors_autoload] --quant %s needs a full-precision base, but --outtype %s already quantized the conversion; "
                         "use --outtype auto/f16/bf16/f32 with --quant." % (a.quant, a.outtype))
    qbin = quantize_binary(bin_dir, a.quantize_bin)                  # fail before the (long) conversion, not after it
    served = {}

    def early(base):
        hit = quant_hit(base, a.quant, qbin, a.imatrix)
        if hit is not None:
            served["quantized"] = hit
        return hit

    got = ensure_gguf(st, a.cache_dir, a.outtype, a.force, a.model_dir, a.allow_quantized, a.python, a.verbose, a.verify_tokenizer, bin_dir,
                      early_return=early)
    if "quantized" in served:
        return got
    return ensure_quantized(got, a.quant, bin_dir, a.outtype, a.quantize_bin, a.imatrix, a.quant_threads, a.force, a.drop_base)


def main():
    ap = argparse.ArgumentParser(
        description="Run a fork binary against a .safetensors by converting to a cached GGUF "
                    "first.")
    ap.add_argument("--bin", required=False,
                    help="the fork binary to exec (omit with --convert-only)")
    ap.add_argument("--model-dir", default=None,
                    help="dir holding config.json + tokenizer; defaults to the shards' own dir")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--outtype", default="auto",
                    choices=["auto", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0",
                             "q5_1", "q6_0"],
                    help="auto = highest-fidelity 16-bit float matching the source dtype")
    ap.add_argument("--quant", default=None, metavar="TYPE",
                    help="after conversion, quantize the cached full-precision GGUF to TYPE (Q4_K_M, Q6_K, IQ4_XS, any llama-quantize name) "
                         "and run on that; the result is cached too (needs llama-quantize beside --bin or in the canonical build)")
    ap.add_argument("--quantize-bin", default=None, help="llama-quantize to use for --quant (default: next to --bin / --bin-dir)")
    ap.add_argument("--imatrix", default=None, help="importance matrix file passed to llama-quantize for --quant")
    ap.add_argument("--quant-threads", type=int, default=0, help="threads for llama-quantize (0 = its default)")
    ap.add_argument("--drop-base", action="store_true",
                    help="with --quant: delete the cached full-precision GGUF once the quantized one is in place (saves disk, costs a "
                         "reconversion if another type is wanted later)")
    ap.add_argument("--force", action="store_true", help="reconvert even on a cache hit")
    ap.add_argument("--allow-quantized", action="store_true",
                    help="attempt conversion even when config.json declares quantization_config")
    ap.add_argument("--python", default=None,
                    help="interpreter used for convert_hf_to_gguf.py (default: this one)")
    ap.add_argument("--convert-only", action="store_true",
                    help="convert and print the GGUF path, do not exec anything")
    ap.add_argument("--print-only", action="store_true",
                    help="resolve and convert, print the command, do not exec")
    ap.add_argument("--inspect", default=None,
                    help="resolve shards + dtypes for this path and exit (no conversion)")
    ap.add_argument("--verify-tokenizer", default="warn", choices=["off", "warn", "error"],
                    help="after conversion, tokenize probe strings with BOTH the HF tokenizer "
                         "and the produced GGUF and compare ids; 'error' refuses to run on a "
                         "mismatch")
    ap.add_argument("--bin-dir", default=None,
                    help="where llama-tokenize lives, for --verify-tokenizer; defaults to the "
                         "directory of --bin, else the canonical build")
    ap.add_argument("--no-relay", action="store_true",
                    help="let the child inherit stdout/stderr instead of relaying it; only "
                         "correct when this script itself has a real console (python.exe, not "
                         "pythonw.exe), and required for an INTERACTIVE llama-cli session")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="arguments for the binary, after --")
    a = ap.parse_args()

    if a.inspect:
        model_dir, shards, index = resolve_shards(a.inspect)
        hist = dtype_histogram(shards)
        print(json.dumps({
            "model_dir": str(model_dir),
            "shards": [s.name for s in shards],
            "index": index.name if index is not None else None,
            "side_files": [f.name for f in side_files(model_dir)],
            "converter_would_find": [p.name for p in converter_would_find(model_dir)],
            "dtypes": {k: {"tensors": v[0], "elements": v[1]} for k, v in hist.items()},
        }, indent=2))
        return 0

    rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest

    bin_dir = a.bin_dir or (str(Path(a.bin).resolve().parent) if a.bin else None) \
        or str(FORK / "build-canonical" / "bin")

    if a.convert_only:
        st = find_safetensors(rest)
        if not st:
            raise SystemExit("--convert-only needs -m <path>.safetensors in the trailing args")
        gguf = _autoload_resolve(a, st, bin_dir)
        print(str(gguf))
        return 0

    if not a.bin:
        raise SystemExit("--bin is required (or use --convert-only / --inspect)")

    # CreateProcess resolves a relative path against the CWD but not against forward-slash
    # separators reliably; resolve it here so a working relative --bin never dies with
    # "[WinError 2] The system cannot find the file specified" for a file that plainly exists.
    binp = Path(a.bin)
    if not binp.is_file():
        for cand in (Path.cwd() / a.bin, FORK / a.bin, FORK / "build-canonical" / "bin" / a.bin):
            if cand.is_file():
                binp = cand
                break
        else:
            raise SystemExit("[safetensors_autoload] --bin %s not found" % a.bin)
    a.bin = str(binp.resolve())

    st = find_safetensors(rest)
    if not st:
        _log("[safetensors_autoload] no .safetensors in -m; passing through unchanged",
              flush=True)
        cmd = [a.bin] + rest
    else:
        gguf = _autoload_resolve(a, st, str(Path(a.bin).parent))
        cmd = [a.bin] + rewrite_args(rest, gguf)

    _log("[safetensors_autoload] exec: %s" % " ".join(cmd))
    if a.print_only:
        return 0
    # CREATE_NO_WINDOW keeps rule 1.45 (windows stay hidden). Under a CONSOLE host the child can
    # simply inherit stdout/stderr (--no-relay), but under pythonw.exe -- which is how this is
    # launched here, because a console host would flash a window -- there is no console to
    # inherit and the binary's output goes NOWHERE: measured, llama-cli ran to completion and
    # printed nothing. So by default the child's output is piped and relayed byte-for-byte.
    if a.no_relay:
        return subprocess.Popen(cmd, creationflags=CREATE_NO_WINDOW).wait()
    return relay_run(cmd)


if __name__ == "__main__":
    sys.exit(main())
