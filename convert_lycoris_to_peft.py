#!/usr/bin/env python3
# SCOUT-NA: row S69-ENGINE-MERGE-ADAPTER-FORMAT-IMPORTERS-LYCORIS-PYREFT-2026-11-06-01 names this importer; PRIOR-ART-NONE: git log --all -S lora_unet_ is empty.
# SWALLOW-OK: every failure is a raised LycorisRefusal that main() prints and turns into exit 3.
# REWRITE-SAFE: reads the input adapter, writes only to a NEW --outdir (never back to the input path).
"""Convert a LyCORIS-native adapter (kohya key scheme) into a PEFT adapter directory that convert_lora_to_gguf.py already serves.

LyCORIS (KohakuBlueleaf, Apache-2.0) saves one safetensors file whose keys are "<lora_name>.<tensor>" with
lora_name = "<prefix>_<module path with dots replaced by underscores>" (prefix lora_unet / lora_te / lora_te1.. / lycoris) and the
tensors hada_w1_a/b, hada_w2_a/b (LoHa) or lokr_w1 | lokr_w1_a/b, lokr_w2 | lokr_w2_a/b (LoKr) plus one scalar "alpha".
The converter's PEFT LOHA / LOKR handlers (convert_lora_to_gguf.py, lowrank_tuner_factors) hold the exact delta math, so this
script only renames: it recovers the real module path from the base checkpoint's tensor names (the underscore form is not
invertible on its own), writes PEFT keys "base_model.model.<path>.<tensor>" and an adapter_config.json whose rank_pattern /
alpha_pattern carry each module's rank and saved alpha.

SOURCE-READ (Apache-2.0, specification only, no code copied): lycoris/kohya.py:184-209 (lora_name), :878-899 (save_weights);
lycoris/modules/loha.py:20-30,134-140,249-261 (weight_list, scale = alpha / r_factor, saved alpha = scale * lora_dim,
custom_state_dict: hada_w1_a saved already multiplied by the scalar); lycoris/modules/lokr.py:158-175,207-220,400-418 (the same
alpha rule, "both factors full" forces alpha = lora_dim so scale = 1); functional/loha.py:20-32 (dW = (w1u w1d) * (w2u w2d) * scale);
functional/lokr.py:16-25 (dW = kron(w1, w2) * scale). Both modules compute scale = saved_alpha / lora_dim, the rule used here.

Refused loudly (the row of each is in the message): conv / tucker modules (hada_t1/t2, lokr_t2), dora_scale (LyCORIS weight
decomposition, not PEFT DoRA), a file that mixes LoHa and LoKr, other algorithms (locon, glora, ia3, full, diag-oft, boft, norms).
pyreft / LoReFT checkpoints are NOT handled here: they need the HIDDEN-STATE kind, row S69-ADAPTERS-RESIDUAL-HOOKS-REFT-ABLATE-2026-10-26-01.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path
from typing import Any

ROW_THIS = "S69-ENGINE-MERGE-ADAPTER-FORMAT-IMPORTERS-LYCORIS-PYREFT-2026-11-06-01"
ROW_DISPATCHER = "S69-ADAPTERS-CONVERTER-DISPATCHER-2026-10-26-01"
LOHA_SUFFIXES = ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b")
LOKR_SUFFIXES = ("lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2", "lokr_w2_a", "lokr_w2_b")
CONV_SUFFIXES = ("hada_t1", "hada_t2", "lokr_t2")
TENSOR_NAMES = LOHA_SUFFIXES + LOKR_SUFFIXES + CONV_SUFFIXES + ("alpha", "dora_scale")


class LycorisRefusal(Exception):
    pass


def split_key(key: str) -> tuple[str, str]:
    """'<lora_name>.<tensor>' -> (lora_name, tensor). lora_name has no dot (dots became underscores)."""
    if "." not in key:
        raise LycorisRefusal(f"'{key}': not a LyCORIS key (expected '<lora_name>.<tensor>')")
    name, rest = key.split(".", 1)
    return name, rest


def strip_prefix(lora_name: str, prefix: str | None) -> tuple[str, str]:
    """(prefix, normalised module name with underscores) for one lora_name."""
    if prefix:
        if lora_name.startswith(prefix + "_"):
            return prefix, lora_name[len(prefix) + 1:]
        raise LycorisRefusal(f"lora_name '{lora_name}' does not start with --prefix '{prefix}_'")
    m = re.match(r"^(lora_unet|lora_te\d*|lycoris)_(.+)$", lora_name)
    if not m:
        raise LycorisRefusal(f"lora_name '{lora_name}': prefix is not one of lora_unet / lora_te[N] / lycoris; pass --prefix")
    return m.group(1), m.group(2)


def read_safetensors_header(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def base_module_names(dir_base: Path) -> dict[str, str]:
    """normalised ('.'->'_') module path -> real module path, from the 2-D '*.weight' tensor names in the base safetensors headers.
    Ambiguous normalised names (two real modules collapse to one) map to '' and are refused when an adapter module needs them."""
    out: dict[str, str] = {}
    files = sorted(Path(dir_base).glob("*.safetensors"))
    if not files:
        raise LycorisRefusal(f"no *.safetensors in --base {dir_base}: the module paths cannot be recovered from the underscore names")
    for f in files:
        for name, info in read_safetensors_header(f).items():
            if name == "__metadata__" or not name.endswith(".weight") or len(info.get("shape", ())) != 2:
                continue
            mod = name[: -len(".weight")]
            norm = mod.replace(".", "_")
            out[norm] = "" if (norm in out and out[norm] != mod) else mod
    return out


def convert_state(state: dict[str, Any], module_names: dict[str, str], prefix: str | None = None
                  ) -> tuple[str, dict[str, Any], dict[str, int], dict[str, float]]:
    """LyCORIS state dict -> (peft_type, PEFT state dict, rank_pattern, alpha_pattern). Tensors pass through untouched
    (only .shape and the alpha scalar are read)."""
    per: dict[str, dict[str, Any]] = {}
    for key, t in state.items():
        lora_name, tensor = split_key(key)
        per.setdefault(lora_name, {})[tensor] = t
    kinds: set[str] = set()
    out: dict[str, Any] = {}
    ranks: dict[str, int] = {}
    alphas: dict[str, float] = {}
    for lora_name, parts in sorted(per.items()):
        _pfx, norm = strip_prefix(lora_name, prefix)
        real = module_names.get(norm)
        if real is None:
            raise LycorisRefusal(f"'{lora_name}': no base module named '{norm}' (underscore form) in --base; wrong base model or --prefix")
        if real == "":
            raise LycorisRefusal(f"'{lora_name}': the underscore name '{norm}' matches more than one base module; row {ROW_THIS} needs a module map")
        names = set(parts)
        if names & set(CONV_SUFFIXES):
            raise LycorisRefusal(f"'{lora_name}': tucker / conv tensors {sorted(names & set(CONV_SUFFIXES))} are a conv kernel, not a Linear; row {ROW_DISPATCHER}")
        if "dora_scale" in names:
            raise LycorisRefusal(f"'{lora_name}': dora_scale is LyCORIS weight decomposition (wd_on_out), not PEFT DoRA; row {ROW_THIS}")
        loha, lokr = names & set(LOHA_SUFFIXES), names & set(LOKR_SUFFIXES)
        extra = names - set(TENSOR_NAMES)
        if extra or (loha and lokr) or not (loha or lokr):
            raise LycorisRefusal(f"'{lora_name}': tensors {sorted(names)} are not a single LoHa or LoKr module (other LyCORIS algorithms "
                                 f"locon/glora/ia3/full/diag-oft/boft/norms are separate rows); row {ROW_THIS}")
        if "alpha" not in names:
            raise LycorisRefusal(f"'{lora_name}': no 'alpha' tensor; LyCORIS always saves it")
        kind = "LOHA" if loha else "LOKR"
        kinds.add(kind)
        if loha:
            need = set(LOHA_SUFFIXES)
            if names - {"alpha"} != need:
                raise LycorisRefusal(f"LoHa '{lora_name}': have {sorted(names)}, need {sorted(need | {'alpha'})}")
            rank = int(parts["hada_w1_a"].shape[1])
        else:
            for side in ("w1", "w2"):
                full = f"lokr_{side}" in names
                pair = {f"lokr_{side}_a", f"lokr_{side}_b"} <= names
                if full == pair:
                    raise LycorisRefusal(f"LoKr '{lora_name}': {side} must be exactly one of lokr_{side} or the lokr_{side}_a/_b pair; have {sorted(names)}")
            rank = None
            for k in ("lokr_w1_a", "lokr_w2_a"):
                if k in parts:
                    rank = int(parts[k].shape[1])
                    break
        a = float(parts["alpha"].reshape(-1)[0]) if hasattr(parts["alpha"], "reshape") else float(parts["alpha"])
        if rank is None:
            # both factors full: LyCORIS forces alpha = lora_dim (scale 1) and saves exactly that, so the rank is the alpha
            rank = max(1, int(round(a)))
        ranks[real] = rank
        alphas[real] = a
        for tn, t in parts.items():
            if tn != "alpha":
                out[f"base_model.model.{real}.{tn}"] = t
    if len(kinds) != 1:
        raise LycorisRefusal(f"the file mixes {sorted(kinds) or 'nothing'}: convert one algorithm at a time; row {ROW_THIS}")
    return kinds.pop(), out, ranks, alphas


def adapter_config(peft_type: str, ranks: dict[str, int], alphas: dict[str, float], base: str | None) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "peft_type": peft_type,
        "r": max(ranks.values()),
        "alpha": max(alphas.values()),
        "rank_pattern": {re.escape(m): r for m, r in ranks.items()},
        "alpha_pattern": {re.escape(m): alphas[m] for m in ranks},
        "target_modules": sorted(ranks),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    if base:
        cfg["base_model_name_or_path"] = base
    return cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("lycoris_file", type=Path, help="LyCORIS-native .safetensors adapter")
    ap.add_argument("--base", type=Path, required=True, help="directory of the base model's safetensors (module paths are read from its headers)")
    ap.add_argument("--outdir", type=Path, required=True, help="PEFT adapter directory to write (adapter_config.json + adapter_model.safetensors)")
    ap.add_argument("--prefix", default=None, help="lora_name prefix (lora_unet, lora_te, lycoris, ...); default: autodetect")
    a = ap.parse_args(argv)
    try:
        from safetensors.torch import load_file, save_file
        state = load_file(str(a.lycoris_file), device="cpu")
        peft_type, out, ranks, alphas = convert_state(state, base_module_names(a.base), a.prefix)
    except LycorisRefusal as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    a.outdir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, str(a.outdir / "adapter_model.safetensors"))
    (a.outdir / "adapter_config.json").write_text(json.dumps(adapter_config(peft_type, ranks, alphas, str(a.base)), indent=2), encoding="utf-8")
    print(f"wrote {a.outdir}: {peft_type}, {len(ranks)} modules; next: convert_lora_to_gguf.py {a.outdir} --base {a.base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
