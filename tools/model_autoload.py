"""ONE entry point for every non-GGUF model format the fork can be pointed at.

    model_autoload.py --bin build-canonical/bin/llama-server.exe -- -m model.safetensors --port 8137
    model_autoload.py --bin build-canonical/bin/llama-server.exe -- -m model.onnx --embedding
    model_autoload.py --bin build-canonical/bin/llama-cli.exe    -- -m model.gguf -p hi

The fork's loader is GGUF-only. Two converters already exist beside it -- convert_hf_to_gguf.py
for HuggingFace safetensors and cortex/scripts/onnx_gguf_convert.py for ONNX -- and each has a
launcher here that converts once into a cached GGUF and rewrites `-m`. This dispatches to the
right one by extension so callers do not have to know which, and so the fork ends up with ONE
non-GGUF-autoload idiom instead of two rival ones. A .gguf (or anything unrecognised) is passed
straight through untouched: the GGUF path must stay exactly as it was.

Every flag after `--` belongs to the binary. Flags for the underlying launcher (--outtype,
--cache-dir, --force, --verify-tokenizer, ...) are forwarded with --pass, e.g.

    model_autoload.py --bin ... --pass=--outtype=f32 --pass=--force -- -m model.safetensors -p hi

Use the `--pass=--flag=value` form: argparse cannot tell a bare `--pass --flag` from a --pass
with its value missing, and rejects it.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORK = HERE.parent
CREATE_NO_WINDOW = 0x08000000

HANDLERS = {
    ".safetensors": HERE / "safetensors_autoload.py",
    ".onnx": HERE / "onnx_autoload.py",
}


def model_arg(argv):
    for i, a in enumerate(argv):
        if a in ("-m", "--model") and i + 1 < len(argv):
            return argv[i + 1]
    return None


def pick_handler(value):
    if value is None:
        return None, "no -m/--model argument"
    p = Path(value)
    suffix = p.suffix.lower()
    if suffix in HANDLERS:
        return HANDLERS[suffix], suffix
    if p.is_dir():
        # A bare HF checkout directory: safetensors first, then ONNX, so `-m <dir>` works.
        for suf in (".safetensors", ".onnx"):
            if any(p.glob("*%s" % suf)):
                return HANDLERS[suf], "%s (directory)" % suf
        return None, "directory holds neither .safetensors nor .onnx"
    return None, "%s is already a native format" % (suffix or "extensionless")


def spawn(cmd):
    """Start a child with EXPLICIT std handles.

    Plain inheritance is not enough here, and neither is passing fd 1/2. This runs under
    pythonw.exe (rule 1.45: windows stay hidden), a GUI-subsystem host with no console, and a
    console child spawned from it writes its output NOWHERE -- MEASURED twice: llama-cli ran to
    completion, exit 0, and printed nothing at all. So the child is piped and relayed.
    """
    sys.path.insert(0, str(HERE))
    from safetensors_autoload import relay_run     # one relay, shared, streams kept separate
    return relay_run(cmd)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--pass", dest="passthru", action="append", default=[],
                    help="a flag to forward to the format launcher; repeatable. Use the "
                         "--pass=--flag=value form")
    ap.add_argument("--python", default=None, help="interpreter for the launcher")
    ap.add_argument("--print-only", action="store_true")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest

    # CreateProcess will not find a relative forward-slash path spelled the way a shell accepts
    # it -- MEASURED: "[WinError 2] The system cannot find the file specified" for a binary that
    # plainly exists. Resolve --bin before anything is spawned.
    binp = Path(a.bin)
    if not binp.is_file():
        for cand in (Path.cwd() / a.bin, FORK / a.bin, FORK / "build-canonical" / "bin" / a.bin):
            if cand.is_file():
                binp = cand
                break
        else:
            raise SystemExit("[model_autoload] --bin %s not found" % a.bin)
    a.bin = str(binp.resolve())

    handler, why = pick_handler(model_arg(rest))

    if handler is None:
        print("[model_autoload] %s -- running the binary directly" % why,
              file=sys.stderr, flush=True)
        cmd = [a.bin] + rest
        print("[model_autoload] exec: %s" % " ".join(cmd), file=sys.stderr, flush=True)
        if a.print_only:
            return 0
        return spawn(cmd)

    if not handler.is_file():
        raise SystemExit("[model_autoload] handler for %s is missing: %s" % (why, handler))

    forwarded = []
    for f in a.passthru:
        forwarded.extend(f.split("=", 1) if f.startswith("--") and "=" in f else [f])
    cmd = [a.python or sys.executable, str(handler), "--bin", a.bin] + forwarded
    if a.print_only:
        cmd.append("--print-only")
    cmd += ["--"] + rest
    print("[model_autoload] %s -> %s" % (why, handler.name), file=sys.stderr, flush=True)
    # The launcher below already relays the binary's output; this only has to hand it real
    # std handles (see spawn()).
    return spawn(cmd)


if __name__ == "__main__":
    sys.exit(main())
