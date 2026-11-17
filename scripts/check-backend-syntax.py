#!/usr/bin/env python3
# PROGRESS-OK: each compile is one source file (seconds); every file's rc is printed flushed as it finishes.
"""check-backend-syntax.py -- compile-only check of the backend source groups that no configured build on
the maintainer's host compiles (the groups listed in .carried-not-compiled-baseline.txt under "CLASSES ...").

Why this exists
  A baseline line that says "needs X, not installed here" is a belief about the host, not a measurement.
  This script replaces the belief with a command: for each group it runs the compiler that group needs, in
  compile-only mode (no link, nothing is run), and records the exit code of every file.  A group whose rc is 0
  for every file is COMPILE-CHECKABLE-NOW; it still may be DEVICE-PENDING (cannot be RUN without the device).

Toolchains (all optional; a group whose toolchain is missing is reported SKIPPED with the missing path, never
as passed):
  --zig      zig >= 0.15 (a multi-target clang driver; bundles libc headers for linux/musl/wasi/macos)
  --gxx      a host g++ (used for host-ISA groups: opencl, openvino, webgpu)
  --deps     directory holding the small vendor header sets the checks need (kleidiai tarball, zdnn.h,
             libdrm/venus_hw.h, OpenVINO include dir, OpenCL headers, Dawn headers); see build_groups()

Usage
  python scripts/check-backend-syntax.py --root <fork root> --zig <zig.exe> --gxx <g++.exe> --deps <dir> \
        [--group arm,riscv,...] [--json out.json]
Exit code: 0 if every executed group compiled clean (SKIPPED groups do not fail the run, they are listed);
           1 if any file failed to compile.

Design notes
  * -c -o <scratch.o> for zig (its -fsyntax-only trips a cache-rename bug on Windows); -S where the group's inline
    assembly uses a vendor mnemonic the LLVM assembler does not know (spacemit).
  * One arch dir per ggml-cpu build is the reason these arch groups are not in a host build; this script is the
    cross-compile that proves they still compile.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
SRC_ROOT = [""]


def run(cmd, timeout=900, env=None):
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=CREATE_NO_WINDOW, env=env)
        return r.returncode, (r.stderr or b"").decode("utf-8", "replace")
    except FileNotFoundError as e:
        return 127, "toolchain missing: %s" % e
    except subprocess.TimeoutExpired:
        return 124, "timeout after %ds" % timeout


def build_groups(root, deps):
    """Return {name: dict(kind, target, flags, defs, incs, globs, needs)}."""
    g = {}
    cpu_inc = [root + "/ggml/include", root + "/ggml/src", root + "/ggml/src/ggml-cpu", root + "/include"]
    base_inc = [root + "/ggml/include", root + "/ggml/src", root + "/include"]
    cpu = root + "/ggml/src/ggml-cpu"

    def arch(name, target, flags, d):
        g[name] = dict(kind="zig", target=target, flags=flags, defs=[], incs=cpu_inc,
                       globs=[cpu + "/arch/%s/*.c" % d, cpu + "/arch/%s/*.cpp" % d], needs=[])

    arch("arm", "aarch64-linux-gnu", ["-mcpu=baseline+dotprod+i8mm+fullfp16"], "arm")
    arch("riscv", "riscv64-linux-gnu", ["-mcpu=baseline_rv64+v"], "riscv")
    arch("s390", "s390x-linux-gnu", ["-mcpu=z15"], "s390")
    arch("powerpc", "powerpc64le-linux-gnu", ["-mcpu=pwr9"], "powerpc")
    arch("loongarch-lasx", "loongarch64-linux-gnu", ["-mlasx"], "loongarch")
    arch("loongarch-lsx", "loongarch64-linux-gnu", ["-mlsx"], "loongarch")
    arch("wasm", "wasm32-wasi", ["-msimd128"], "wasm")

    kai = deps + "/kleidiai/kleidiai-v1.24.0"
    g["kleidiai"] = dict(kind="zig", target="aarch64-linux-gnu", flags=["-mcpu=baseline+dotprod+i8mm+fullfp16"],
                         defs=["GGML_USE_CPU_KLEIDIAI"], incs=cpu_inc + [kai], globs=[cpu + "/kleidiai/*.cpp"],
                         needs=[kai + "/kai/kai_common.h"])
    sp = cpu + "/spacemit/*.cpp"
    for tag, defs in (("spacemit", ["RISCV64_SPACEMIT_IME1", "RISCV64_SPACEMIT_IME2"]),):
        g[tag] = dict(kind="zig", target="riscv64-linux-gnu", flags=["-mcpu=baseline_rv64+v+zfh+zvfh+zba+zicbop+zihintpause", "-S"],
                      defs=["GGML_USE_CPU_RISCV64_SPACEMIT"] + defs, incs=cpu_inc, globs=[sp], needs=[])
    mt = root + "/ggml/src/ggml-metal"
    g["metal-cpp"] = dict(kind="zig", target="aarch64-macos", flags=[], defs=[], incs=base_inc + [mt],
                          globs=[mt + "/*.cpp"], needs=[])
    zd = root + "/ggml/src/ggml-zdnn"
    g["zdnn"] = dict(kind="zig", target="s390x-linux-gnu", flags=["-mcpu=z16"], defs=["GGML_USE_ZDNN"],
                     incs=base_inc + [zd, deps + "/zdnn/include"], globs=[zd + "/*.cpp"], needs=[deps + "/zdnn/include/zdnn.h"])
    vg = root + "/ggml/src/ggml-virtgpu"
    vinc = base_inc + [vg, vg + "/include", vg + "/backend", vg + "/backend/shared", deps + "/virtgpu/include", deps + "/virtgpu/libdrm", deps + "/virtgpu/libdrm/drm"]
    g["virtgpu-frontend"] = dict(kind="zig", target="x86_64-linux-gnu", flags=[], defs=[], incs=vinc,
                                 globs=[vg + "/*.cpp"], needs=[deps + "/virtgpu/libdrm/xf86drm.h"])
    g["virtgpu-backend"] = dict(kind="zig", target="x86_64-linux-gnu", flags=[], defs=[], incs=vinc,
                                globs=[vg + "/backend/*.cpp"], needs=[deps + "/virtgpu/libdrm/xf86drm.h"])
    ocl = (sorted(glob.glob(deps + "/khronos_opencl_extract/*")) or [deps + "/khronos_opencl_extract/MISSING"])[0].replace("\\", "/")
    ov = deps + "/openvino_include"
    clhpp = (sorted(glob.glob(deps + "/khronos_clhpp_extract/*/include")) or [deps + "/khronos_clhpp_extract/MISSING"])[0].replace("\\", "/")
    g["openvino"] = dict(kind="gxx", flags=[], defs=[], incs=base_inc + [root + "/ggml/src/ggml-openvino", ov, ocl, clhpp],
                         globs=[root + "/ggml/src/ggml-openvino/**/*.cpp"], needs=[ov + "/openvino/openvino.hpp", ocl + "/CL/cl.h"])
    oc = root + "/ggml/src/ggml-opencl"
    g["opencl"] = dict(kind="gxx", flags=[], defs=["GGML_OPENCL_SOA_Q", "GGML_OPENCL_TARGET_VERSION=300"],
                       incs=base_inc + [oc, ocl], globs=[oc + "/*.cpp"], needs=[ocl + "/CL/cl.h"])
    wg = root + "/ggml/src/ggml-webgpu"
    dawn = (sorted(glob.glob(deps + "/dawn_extract/Dawn-*/include")) or [deps + "/dawn_extract/MISSING/include"])[0].replace("\\", "/")
    # Dawn webgpu_cpp.h needs C++20 (std::span, std::type_identity); dawn::webgpu_dawn carries it as a usage requirement
    g["webgpu"] = dict(kind="gxx", flags=[], defs=[], std="gnu++20", incs=base_inc + [wg, dawn, deps + "/webgpu_generated"],
                       globs=[wg + "/*.cpp"], needs=[dawn + "/webgpu/webgpu_cpp.h", deps + "/webgpu_generated/ggml-wgsl-shaders.hpp"])
    # SYCL: intel/llvm nightly clang++ -fsycl (needs the MSVC STL/SDK environment, see --msvc) + oneMKL SYCL headers.
    sy = deps + "/sycl"
    mkl = (sorted(glob.glob(sy + "/whl_onemkl-sycl-include/*.data/data/Library/include")) or [sy + "/MISSING"])[0].replace("\\", "/")
    mkli = (sorted(glob.glob(sy + "/whl_mkl-include/*.data/data/Library/include")) or [sy + "/MISSING"])[0].replace("\\", "/")
    sd = root + "/ggml/src/ggml-sycl"
    g["sycl"] = dict(kind="sycl", tool=sy + "/llvm/bin/clang++.exe", flags=["-fsycl"], defs=["GGML_SYCL_DNNL=0", "GGML_SYCL_WARP_SIZE=16", "GGML_SYCL_F16"],
                     incs=base_inc + [sd, mkl, mkli], globs=[sd + "/*.cpp", root + "/ggml/src/ggml-sycl.cpp"],
                     needs=[mkl + "/oneapi/mkl.hpp", sy + "/llvm/bin/clang++.exe"])
    return g


def msvc_env(msvc):
    """Environment of portable MSVC (same values as portable/msvc/activate_x64.bat)."""
    sdk = (sorted(os.listdir(msvc + "/Windows Kits/10/Include")) or [""])[-1]
    tools = os.path.join(msvc, "VC", "Tools", "MSVC")
    ver = sorted(os.listdir(tools))[-1]
    env = dict(os.environ)
    inc = [os.path.join(tools, ver, "include")] + [os.path.join(msvc, "Windows Kits", "10", "Include", sdk, d) for d in ("ucrt", "shared", "um", "winrt", "cppwinrt")]
    env["INCLUDE"] = os.pathsep.join(inc)
    env["LIB"] = os.pathsep.join([os.path.join(tools, ver, "lib", "x64"), os.path.join(msvc, "Windows Kits", "10", "Lib", sdk, "ucrt", "x64"), os.path.join(msvc, "Windows Kits", "10", "Lib", sdk, "um", "x64")])
    return env


def expand(globs):
    out = []
    for p in globs:
        out += sorted(glob.glob(p, recursive=True))
    return out


def check_group(name, spec, zig, gxx, scratch, env=None):
    missing = [n for n in spec["needs"] if not os.path.exists(n)]
    tool = zig if spec["kind"] == "zig" else spec.get("tool", gxx) if spec["kind"] == "sycl" else gxx
    if spec["kind"] == "sycl" and env is None:
        return dict(group=name, status="SKIPPED", reason="needs --msvc <portable msvc root> for the STL/SDK headers", files=[])
    if not tool or not os.path.exists(tool):
        return dict(group=name, status="SKIPPED", reason="toolchain missing: %s" % tool, files=[])
    if missing:
        return dict(group=name, status="SKIPPED", reason="dependency missing: %s" % missing, files=[])
    files = expand(spec["globs"])
    if not files:
        return dict(group=name, status="SKIPPED", reason="no source matched %s" % spec["globs"], files=[])
    res = []
    for f in files:
        cpp = f.endswith((".cpp", ".cc"))
        std = ["-std=" + spec.get("std", "gnu++17")] if cpp else ["-std=gnu11"]
        incs = ["-I" + i for i in spec["incs"]]
        defs = ["-D" + d for d in spec["defs"]]
        flags = [x for x in spec["flags"] if x != "-S"]
        asm = "-S" in spec["flags"]
        if spec["kind"] == "zig":
            out = os.path.join(scratch, "o.s" if asm else "o.o")
            cmd = [tool, "c++" if cpp else "cc", "-target", spec["target"], "-S" if asm else "-c", "-o", out] + std + flags + defs + incs + [f]
        else:
            cmd = [tool, "-fsyntax-only"] + std + flags + defs + incs + [f]
        rc, err = run(cmd, env=env if spec["kind"] == "sycl" else None)
        errs = [x for x in err.splitlines() if "error" in x]
        res.append(dict(file=os.path.relpath(f, SRC_ROOT[0]).replace("\\", "/"), rc=rc, errors=len(errs),
                        first=(errs[0][-260:] if errs else (err[-200:] if rc else ""))))
        print("  %-18s %-60s rc=%d" % (name, res[-1]["file"][-60:], rc), flush=True)
    bad = [r for r in res if r["rc"] != 0]
    return dict(group=name, status="PASS" if not bad else "FAIL", files=res, n=len(res), failed=len(bad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--zig", default="")
    ap.add_argument("--gxx", default="")
    ap.add_argument("--deps", default="")
    ap.add_argument("--group", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--msvc", default="", help="portable MSVC root (STL + Windows SDK) for the sycl group")
    a = ap.parse_args()
    root = os.path.abspath(a.root).replace("\\", "/")
    SRC_ROOT[0] = root
    groups = build_groups(root, a.deps.replace("\\", "/"))
    want = [x for x in a.group.split(",") if x] or list(groups)
    scratch = tempfile.mkdtemp(prefix="bsyn_")
    results = []
    for name in want:
        if name not in groups:
            print("unknown group", name, "known:", list(groups))
            return 2
        r = check_group(name, groups[name], a.zig, a.gxx, scratch, msvc_env(a.msvc.replace("\\", "/")) if a.msvc else None)
        print("GROUP %-18s %s %s" % (name, r["status"], r.get("reason", "") or "%d file(s), %d failed" % (r["n"], r["failed"])), flush=True)
        results.append(r)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
    return 1 if any(r["status"] == "FAIL" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
