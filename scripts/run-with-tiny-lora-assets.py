#!/usr/bin/env python3
"""Run a test executable with random-weight assets (scripts/gen-tiny-lora-test-assets.py) appended to its arguments:
    run-with-tiny-lora-assets.py [--work-dir DIR] EXE [EXE-ARGS...]   ->   EXE EXE-ARGS... tiny-llama.gguf lora-a.gguf lora-b.gguf
Exits 77 (CTest SKIPPED) when numpy is missing; the assets need numpy + gguf-py."""
import os
import subprocess
import sys
import tempfile

here = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
work = None
if args[:1] == ["--work-dir"]:
    work, args = args[1], args[2:]
try:
    import numpy  # noqa: F401
except ImportError:
    print("numpy missing: SKIPPED")
    sys.exit(77)
work = work or tempfile.mkdtemp(prefix="tiny-lora-assets-")
sys.path.insert(0, here)
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("gen_assets", os.path.join(here, "gen-tiny-lora-test-assets.py"))
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
sys.argv = ["gen", work]
gen.main()
cmd = args + [os.path.join(work, n) for n in ("tiny-llama.gguf", "lora-a.gguf", "lora-b.gguf")]
sys.exit(subprocess.call(cmd))
