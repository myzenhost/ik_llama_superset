#!/usr/bin/env python3
"""MoE-aware per-layer quantization recipe -> llama-quantize rules.

WHAT THIS IS. The "dynamic per-layer MoE-aware quant" idea (Unsloth's Dynamic 2.0 family) is a quantize-time
POLICY, not a format: decide, tensor by tensor, how many bits each part of a mixture-of-experts model deserves.
llama-quantize in this fork already has the mechanism (src/llama-quantize.cpp llama_custom_quant_type: the first
--custom-q / --tensor-type regex that matches a tensor name wins, and it overrides every built-in heuristic, in
--pure mode too). What was missing is the policy, written down, parameterised and testable. This script is that.

WHAT IT IS NOT. It is not a reproduction of Unsloth's recipe: their per-layer choices come from a private
KL-divergence-prioritised calibration and are not published (research note on AGENT-20260709105849: docs give
results and philosophy, no code or config). This is OUR policy in the open mechanism; its quality is whatever
`llama-perplexity --kl-divergence` says on the model it is applied to, never a claim inherited from theirs.

POLICY (first match wins, so the order below is the order of the rules):
  1. router / gating   ffn_gate_inp*, exp_probs_b      keep full precision (tiny, and a mis-route is catastrophic)
  2. embeddings/output token_embd, output              embed tier / output tier
  3. boundary layers   first/last N layers, any weight  boundary tier (the layers every published sensitivity study
                                                        finds most fragile)
  4. dense FFN         leading dense layers (blk.0..k)  dense tier
  5. shared experts    ffn_*_shexp                      shared tier
  6. attention         attn_* (incl. MLA a/b projs)     attn tier
  7. routed experts    ffn_gate_exps / ffn_up_exps      LOW tier (this is where the bytes are)
                       ffn_down_exps                    LOW tier + 1 step (down_proj is the usual outlier)
  8. anything else 2-D left to the default type.

Tiers are plain ggml type names, overridable per flag. With --low iq1_xs/iq1_xxs/iq1_xxxs the narrow-grid types are
used for routed experts, which is the one place they would pay off.

USAGE
  moe_dynamic_recipe.py --n-layer 61 --first-dense 3 --tier iq1 --custom-q        # one-line string for --custom-q
  moe_dynamic_recipe.py --n-layer 61 --first-dense 3 --tier iq1 --tensor-type-file out.txt
  moe_dynamic_recipe.py --n-layer 61 --first-dense 3 --tier iq1 --resolve blk.5.ffn_up_exps.weight
"""
from __future__ import annotations

import argparse
import re
import sys

# name -> (routed gate/up, routed down, attention, shared, dense, boundary, embed, output)
TIERS = {
    "iq1": dict(low="iq1_s",   down="iq2_xxs", attn="iq4_xs", shared="iq4_xs", dense="q4_k", boundary="q4_k",   embed="q4_k", output="q6_k"),
    "iq2": dict(low="iq2_xxs", down="iq2_s",   attn="q4_k",   shared="q5_k",   dense="q5_k", boundary="q5_k",   embed="q5_k", output="q6_k"),
    "iq3": dict(low="iq3_xxs", down="iq3_s",   attn="q5_k",   shared="q6_k",   dense="q6_k", boundary="q6_k",   embed="q6_k", output="q6_k"),
}

# tensors that must never be quantized below their stored precision
KEEP_FULL = r"ffn_gate_inp|exp_probs_b|ffn_gate_inp_shexp"


def layer_alt(layers: list[int]) -> str:
    return "|".join(str(i) for i in sorted(set(layers)))


def build_rules(n_layer: int, first_dense: int, boundary_head: int, boundary_tail: int, t: dict) -> list[tuple[str, str]]:
    """Ordered (regex, ggml_type) list for llama-quantize --custom-q (std::regex_search semantics)."""
    rules: list[tuple[str, str]] = []
    rules.append((KEEP_FULL, "f32"))
    rules.append((r"^token_embd\.weight$", t["embed"]))
    rules.append((r"^output\.weight$", t["output"]))
    boundary = list(range(first_dense, first_dense + boundary_head)) + list(range(n_layer - boundary_tail, n_layer))
    boundary = [i for i in boundary if 0 <= i < n_layer]
    if boundary:
        # the boundary tier applies to the MoE weights of those layers only; attention follows its own tier below
        rules.append((r"^blk\.(" + layer_alt(boundary) + r")\.ffn_(gate|up|down)_exps\.weight$", t["boundary"]))
    if first_dense > 0:
        rules.append((r"^blk\.(" + layer_alt(list(range(first_dense))) + r")\.ffn_(gate|up|down)\.weight$", t["dense"]))
    rules.append((r"ffn_(gate|up|down)_shexp\.weight$", t["shared"]))
    rules.append((r"\.attn_[a-z_0-9]+\.weight$", t["attn"]))
    rules.append((r"ffn_down_exps\.weight$", t["down"]))
    rules.append((r"ffn_(gate|up)_exps\.weight$", t["low"]))
    return rules


def resolve(rules: list[tuple[str, str]], name: str) -> str | None:
    for rx, ty in rules:
        if re.search(rx, name):
            return ty
    return None


def to_custom_q(rules: list[tuple[str, str]]) -> str:
    # --custom-q splits on ',' and then on the FIRST '=' of each item, so a regex must contain neither ',' nor '='
    for rx, _ in rules:
        assert "," not in rx and "=" not in rx, rx
    return ",".join(f"{rx}={ty}" for rx, ty in rules)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-layer", type=int, required=True, help="block_count of the model")
    ap.add_argument("--first-dense", type=int, default=0, help="leading dense (non-MoE) layers, e.g. 3 for DeepSeek-V3, 1 for GLM-4.5")
    ap.add_argument("--boundary-head", type=int, default=2, help="MoE layers right after the dense ones that get the boundary tier")
    ap.add_argument("--boundary-tail", type=int, default=2, help="last layers that get the boundary tier")
    ap.add_argument("--tier", choices=sorted(TIERS), default="iq1")
    for k in ("low", "down", "attn", "shared", "dense", "boundary", "embed", "output"):
        ap.add_argument("--" + k, help=f"override the {k} type (default from --tier)")
    ap.add_argument("--custom-q", action="store_true", help="print the --custom-q argument")
    ap.add_argument("--tensor-type-file", help="write rules as a --tensor-type-file (lower-cased patterns, one per line)")
    ap.add_argument("--resolve", action="append", default=[], help="print the type a tensor name resolves to (repeatable)")
    a = ap.parse_args(argv)

    t = dict(TIERS[a.tier])
    for k in t:
        v = getattr(a, k)
        if v:
            t[k] = v.lower()
    rules = build_rules(a.n_layer, a.first_dense, a.boundary_head, a.boundary_tail, t)

    if a.custom_q:
        print(to_custom_q(rules))
    if a.tensor_type_file:
        with open(a.tensor_type_file, "w", encoding="utf-8") as f:
            for rx, ty in rules:
                # parse_tensor_type lower-cases the pattern, so refuse a pattern whose meaning depends on case
                assert rx == rx.lower() or re.sub(r"\\[A-Za-z]", "", rx) == re.sub(r"\\[A-Za-z]", "", rx).lower(), rx
                f.write(f"{rx.lower()}={ty}\n")
    for n in a.resolve:
        print(f"{n} -> {resolve(rules, n)}")
    if not (a.custom_q or a.tensor_type_file or a.resolve):
        for rx, ty in rules:
            print(f"{ty:9s} {rx}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
