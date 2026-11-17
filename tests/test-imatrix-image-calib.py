#!/usr/bin/env python3
"""llama-imatrix --image-calib: calibrate on image-token activations (S69-ENGREV-RKNN-IMATRIX-MM-EMBD-CALIB-2026-10-27-01).

Usage: test-imatrix-image-calib.py <llama-imatrix> <vl-model.gguf> <mmproj.gguf>
Exit 0 = pass, 1 = fail, 77 = a needed file is missing (skip).
Builds a directory of two small PNGs, runs the tool with no -f text, and asserts the tool decoded both
images and wrote a non-trivial imatrix. Negative controls: an empty directory and a missing --mmproj must fail.
"""
import os
import struct
import subprocess
import sys
import tempfile
import zlib


def png(path, w, h, seed):
    raw = b"".join(b"\x00" + bytes(((x * 7 + y * 3 + seed * 41) % 256) for x in range(w) for _c in range(3)) for y in range(h))
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) \
        + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(data)


def run(args):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    p = subprocess.run(args, capture_output=True, text=True, creationflags=flags, timeout=1800)
    return p.returncode, p.stdout + p.stderr


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 1
    exe, model, mmproj = sys.argv[1:4]
    for f in (exe, model, mmproj):
        if not os.path.isfile(f):
            print("SKIP: missing", f)
            return 77
    fails = 0
    with tempfile.TemporaryDirectory() as td:
        imgs = os.path.join(td, "imgs")
        os.mkdir(imgs)
        png(os.path.join(imgs, "a.png"), 64, 64, 1)
        png(os.path.join(imgs, "b.png"), 64, 48, 2)
        empty = os.path.join(td, "empty")
        os.mkdir(empty)
        out = os.path.join(td, "img.imatrix")
        base = [exe, "-m", model, "--image-calib", imgs, "-o", out, "-c", "2048", "-b", "2048", "-ub", "2048", "-ngl", "0", "-t", "8"]

        rc, log = run(base + ["-mm", mmproj])
        ok = rc == 0 and "calibrated on 2 image(s)" in log and os.path.isfile(out) and os.path.getsize(out) > 4096
        print("image-calib run:", "PASS" if ok else "FAIL", "rc=%d size=%d" % (rc, os.path.getsize(out) if os.path.isfile(out) else -1))
        if not ok:
            print(log[-3000:])
            fails += 1

        rc, log = run([exe, "-m", model, "--image-calib", empty, "-mm", mmproj, "-o", out + ".e", "-c", "2048", "-ngl", "0"])
        ok = rc != 0 and "no images" in log
        print("negative: empty dir refused:", "PASS" if ok else "FAIL")
        fails += 0 if ok else 1

        rc, log = run(base)  # same args, no --mmproj
        ok = rc != 0 and "needs --mmproj" in log
        print("negative: no mmproj refused:", "PASS" if ok else "FAIL")
        fails += 0 if ok else 1
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
