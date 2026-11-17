"""numpy cross-check for tests/test-rope-yarn-truncate.cpp --dump <file>.

Reimplements transformers' modeling_rope_utils._compute_yarn_parameters (truncate, attention_factor) in numpy float64 and
compares it to the CPU ggml output stored in the dump (truncate=false and truncate=true cases).

    test-rope-yarn-truncate --dump yarn.txt && python rope_yarn_truncate_ref.py yarn.txt

Spec source: huggingface/transformers src/transformers/modeling_rope_utils.py (Apache-2.0), data only; no code copied.
"""
import sys

import numpy as np


def corr_dim(n_rot, dim, base, max_pos):
    return (dim * np.log(max_pos / (n_rot * 2 * np.pi))) / (2 * np.log(base))


def rope_ref(mode, dim, base, factor, orig, bf, bs, attn, truncate, pos, x):
    low, high = corr_dim(bf, dim, base, orig), corr_dim(bs, dim, base, orig)
    if truncate:
        low, high = np.floor(low), np.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    half = dim // 2
    pos_freqs = base ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    extrap = 1.0 / pos_freqs
    interp = 1.0 / (factor * pos_freqs)
    ramp = np.clip((np.arange(half, dtype=np.float64) - low) / (high - low), 0, 1)
    extrap_factor = 1 - ramp
    inv_freq = interp * (1 - extrap_factor) + extrap * extrap_factor
    ang = pos * inv_freq
    cs, sn = np.cos(ang) * attn, np.sin(ang) * attn
    out = np.empty_like(x)
    if mode == 2:  # GGML_ROPE_TYPE_NEOX
        x0, x1 = x[:half], x[half:]
        out[:half], out[half:] = x0 * cs - x1 * sn, x0 * sn + x1 * cs
    else:          # GGML_ROPE_TYPE_NORMAL, interleaved
        x0, x1 = x[0::2], x[1::2]
        out[0::2], out[1::2] = x0 * cs - x1 * sn, x0 * sn + x1 * cs
    return out


def main(path):
    lines = open(path).read().splitlines()
    bad = 0
    for i in range(0, len(lines), 5):
        h = lines[i].split()
        name, mode, dim = h[1], int(h[2]), int(h[3])
        base, factor, orig, bf, bs, attn = float(h[4]), float(h[5]), int(h[6]), float(h[7]), float(h[8]), float(h[9])
        n_head, n_tok = int(h[10]), int(h[11])
        pos = [int(v) for v in lines[i + 1].split()[1:]]
        x = np.array(lines[i + 2].split()[1:], dtype=np.float64).reshape(n_tok, n_head, dim)
        g_false = np.array(lines[i + 3].split()[1:], dtype=np.float64).reshape(n_tok, n_head, dim)
        g_true = np.array(lines[i + 4].split()[1:], dtype=np.float64).reshape(n_tok, n_head, dim)
        for trunc, g in ((False, g_false), (True, g_true)):
            ref = np.empty_like(x)
            for t in range(n_tok):
                for hd in range(n_head):
                    ref[t, hd] = rope_ref(mode, dim, base, factor, orig, bf, bs, attn, trunc, pos[t], x[t, hd])
            err = np.abs(ref - g).max()
            ok = err < 3e-3
            bad += not ok
            print(f"{name:12s} truncate={trunc!s:5s} max|ggml-numpy|={err:.3e} {'OK' if ok else 'MISMATCH'}")
        # discrimination: truncate=true ggml output vs truncate=false numpy reference == pre-change behaviour
        ref_n = np.empty_like(x)
        for t in range(n_tok):
            for hd in range(n_head):
                ref_n[t, hd] = rope_ref(mode, dim, base, factor, orig, bf, bs, attn, False, pos[t], x[t, hd])
        before = np.abs(ref_n - g_true).max()
        print(f"{name:12s} BEFORE (floor/ceil ggml vs numpy truncate=false) max err={before:.3e} {'MISMATCH (expected)' if before > 0.06 else 'NOT DISTINGUISHABLE'}")
        bad += before <= 0.06
    print("numpy cross-check:", "OK" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
