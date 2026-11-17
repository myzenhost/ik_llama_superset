#!/usr/bin/env python3
"""Adreno constant-register workaround check for the OpenCL kernels.

Adreno's instruction validator rejects an op that pairs a constant-class register
(get_local_size(0)) with a uniform-class one ("# of constant registers must be < 2").
The workaround routes the value through a volatile local so it lands in a general
register. It is semantically neutral on every other device, so the only thing a host
without an Adreno can verify is that the kernels still compile and that the volatile
round trip survives -O1 (a volatile load is never folded away).

Checks, using clang's OpenCL front end:
  add.cl / cpy.cl                    : compile; the IR has a `load volatile` of the local size
  gemv_noshuffle_q4_k_f32.cl         : WITH -DADRENO_643_Q4K_CONSTREG_WORKAROUND the IR has a
                                       `load volatile`; WITHOUT it there is none (the define is
                                       device-model specific and must stay off elsewhere)
Positive control: the IR must contain the kernel's own symbol, otherwise the run measured the
extraction and not the kernel.

usage: check-opencl-adreno-constreg.py --clang <clang> [--kernel-dir <dir>]
exit 0 = all pass, 1 = a check failed, 77 = no clang (ctest SKIP).
"""
import argparse
import os
import subprocess
import sys


def emit_ir(clang, src, defines):
    cmd = [clang, "-x", "cl", "-cl-std=CL2.0", "-S", "-emit-llvm", "-O1",
           "-Xclang", "-finclude-default-header"] + ["-D" + d for d in defines] + [src, "-o", "-"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("clang failed for %s:\n%s" % (src, p.stderr))
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clang", required=True)
    ap.add_argument("--kernel-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "ggml", "src", "ggml-opencl", "kernels"))
    a = ap.parse_args()
    if not os.path.exists(a.clang):
        print("SKIP: clang not found at", a.clang)
        return 77

    fails = []

    def check(name, ok, detail=""):
        print("%-4s %s %s" % ("PASS" if ok else "FAIL", name, detail))
        if not ok:
            fails.append(name)

    cases = [
        ("add.cl", [], "kernel_add", True),
        ("cpy.cl", [], "kernel_cpy_f32_f32_pack", True),
        ("gemv_noshuffle_q4_k_f32.cl", ["ADRENO_643_Q4K_CONSTREG_WORKAROUND=1"], "kernel_gemv_noshuffle_q4_k_f32", True),
        ("gemv_noshuffle_q4_k_f32.cl", [], "kernel_gemv_noshuffle_q4_k_f32", False),
    ]
    for fname, defines, sym, want_volatile in cases:
        tag = fname + (" +" + ",".join(defines) if defines else "")
        ir = emit_ir(a.clang, os.path.join(a.kernel_dir, fname), defines)
        check(tag + " positive-control(%s in IR)" % sym, sym in ir)
        check(tag + " volatile-load " + ("present" if want_volatile else "absent"),
              ("load volatile" in ir) == want_volatile)
    print("RESULT:", "FAIL" if fails else "PASS", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
