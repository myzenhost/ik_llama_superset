#!/usr/bin/env python3
"""Red-then-green test for scripts/check-ggml-type-ids.py."""
# REWRITE-SAFE: read-only test, writes nothing
import importlib.util
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("chk", root / "scripts" / "check-ggml-type-ids.py")
chk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)

hdr = (root / "ggml/include/ggml.h").read_text(encoding="utf-8", errors="replace")
golden = json.loads(chk.GOLDEN.read_text())
cur = chk.parse_enum(hdr)
assert chk.check(cur, golden) == [], "green: real header must pass"
assert cur["GGML_TYPE_I2_S"] == 36 and len(cur) > 100

# red: renumber a member
bad = hdr.replace("GGML_TYPE_I2_S      = 36,", "GGML_TYPE_I2_S      = 37,")
assert bad != hdr
errs = chk.check(chk.parse_enum(bad), golden)
assert any("I2_S renumbered 36 -> 37" in e for e in errs), errs

# red: duplicate id
dup = hdr.replace("GGML_TYPE_Q6_0    = 133,", "GGML_TYPE_Q6_0    = 134,")
assert dup != hdr and any("used by both" in e for e in chk.check(chk.parse_enum(dup), golden))

# red: removed member
gone = hdr.replace("        GGML_TYPE_TQ1_0     = 34,", "        // GGML_TYPE_TQ1_0     = 34,")
assert gone != hdr and any("TQ1_0" in e and "removed" in e for e in chk.check(chk.parse_enum(gone), golden))
print("PASS")
