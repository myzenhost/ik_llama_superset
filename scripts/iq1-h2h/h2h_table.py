#!/usr/bin/env python3
# PROGRESS-OK: prints a table from a few dozen rows of an existing results file; finishes in milliseconds.
"""Markdown table from results.jsonl written by h2h_run.py. Read-only: prints to stdout."""
import json
import os
import re

W = os.environ.get("H2H_WORK", "build/work/h2h")
rows = [json.loads(l) for l in open(os.path.join(W, "results.jsonl"), encoding="utf-8") if l.strip()]
base = None
bo = os.path.join(W, "base.out")
if os.path.exists(bo):
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", open(bo, encoding="utf-8", errors="replace").read())
    base = float(m.group(1)) if m else None
print(f"reference (Q8_0 source) PPL = {base}")
for regime in ("B", "A", "M", "C"):
    rs = sorted([r for r in rows if r["regime"] == regime and "failed" not in r], key=lambda r: r["bpw_type"])
    failed = [r for r in rows if r["regime"] == regime and "failed" in r]
    if not rs and not failed:
        continue
    print(f"\n### regime {regime}\n")
    print("| type | bpw (FFN) | PPL | mean KLD | median KLD | 99% KLD | same top-p % | file MB |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rs:
        print(f"| {r['type']} | {r['bpw_type']} | {r.get('ppl_q', float('nan')):.1f} | {r.get('kld_mean', float('nan')):.3f} | {r.get('kld_median', float('nan')):.3f} | "
              f"{r.get('kld_p99', float('nan')):.2f} | {r.get('same_top_p', float('nan')):.1f} | {r.get('file_bytes', 0)/1e6:.0f} |")
    for r in failed:
        print(f"| {r['type']} | {r['bpw_type']} | FAILED at {r['failed']} (rc {r.get('rc')}) | | | | | |")
