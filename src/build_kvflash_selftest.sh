#!/usr/bin/env bash
# S69 #66 — build + run the KVFlash pager host-side selftest (CPU, no GPU). Proves GATE C/D/E + invariants.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
GXX="${GXX:-g++}"
INC="$(mktemp -d)"
cp "$HERE/_kvflash_test_stub_ggml.h"         "$INC/ggml.h"
cp "$HERE/_kvflash_test_stub_ggml_backend.h" "$INC/ggml-backend.h"
"$GXX" -std=c++17 -O2 -I "$INC" -I "$HERE/kvflash_ref" -I "$HERE" "$HERE/llama-kvflash.selftest.cpp" -o "$INC/kvf.exe"
"$INC/kvf.exe"
