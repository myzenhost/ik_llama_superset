#!/usr/bin/env python3
"""Fail when a ggml_type enum value changes (row S69-GGML-TYPE-ID-IS-ONDISK-ABI-DO-NOT-RENUMBER-01).

ggml_type ids are an on-disk ABI: a GGUF stores the id in each tensor-info entry, so renumbering
a member makes every stored file decode as the wrong quant.  This pins name -> id in
scripts/ggml_type_ids.golden.json.

  check-ggml-type-ids.py            verify (exit 1 on a changed/removed id or a duplicate id)
  check-ggml-type-ids.py --update   add NEW members to the golden (never edits an existing one)

New members are allowed but must be recorded with --update so the next change is guarded too.
"""
# REWRITE-SAFE: --update only ADDS keys to the golden after check() proves no existing id changed; git tracks the file
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "ggml" / "include" / "ggml.h"
GOLDEN = Path(__file__).resolve().parent / "ggml_type_ids.golden.json"


def parse_enum(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    m = re.search(r"enum\s+ggml_type\s*\{(.*?)\n\s*\};", text, flags=re.S)
    if not m:
        raise SystemExit("enum ggml_type not found")
    ids, nxt = {}, 0
    for item in m.group(1).split(","):
        item = item.strip()
        if not item:
            continue
        mm = re.fullmatch(r"(GGML_TYPE_\w+)\s*(?:=\s*(\d+))?", item)
        if not mm:
            raise SystemExit("unparsed enum item: %r" % item)
        val = int(mm.group(2)) if mm.group(2) is not None else nxt
        ids[mm.group(1)] = val
        nxt = val + 1
    ids.pop("GGML_TYPE_COUNT", None)
    return ids


def check(current, golden):
    errs = []
    for name, val in golden.items():
        if name not in current:
            errs.append("%s (id %d) removed or renamed" % (name, val))
        elif current[name] != val:
            errs.append("%s renumbered %d -> %d" % (name, val, current[name]))
    seen = {}
    for name, val in current.items():
        if val in seen:
            errs.append("id %d used by both %s and %s" % (val, seen[val], name))
        seen[val] = name
    return errs


def main(argv):
    current = parse_enum(HEADER.read_text(encoding="utf-8", errors="replace"))
    golden = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
    if "--update" in argv:
        errs = check(current, golden)
        if errs:
            print("refusing to update; fix first:\n  " + "\n  ".join(errs))
            return 1
        added = {k: v for k, v in current.items() if k not in golden}
        golden.update(added)
        GOLDEN.write_text(json.dumps(dict(sorted(golden.items(), key=lambda kv: kv[1])), indent=1) + "\n")
        print("golden has %d ids (%d added)" % (len(golden), len(added)))
        return 0
    errs = check(current, golden)
    new = [k for k in current if k not in golden]
    if new:
        print("note: %d unrecorded new member(s); run with --update: %s" % (len(new), ", ".join(new)))
    if errs:
        print("ggml_type id ABI broken:\n  " + "\n  ".join(errs))
        return 1
    print("ok: %d ggml_type ids match the golden" % len(golden))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
