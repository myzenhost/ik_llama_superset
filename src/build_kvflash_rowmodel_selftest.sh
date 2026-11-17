#!/usr/bin/env bash
# S69 #66 Solution-1 validation: KVFlash pager on ik_llama's TOKEN-ROW layout.
# Proves the UNCHANGED pager works when the 2D K-cache [n_embd_head_k, n_head_kv*pool] is presented as a
# reshape [n_embd_k_gqa, pool, 1] (heads folded into the row, n_head_kv=1) — ik_llama K is token-major.
# CPU/g++, no GPU, no ggml. Gates C/D/E. (Companion to build_kvflash_selftest.sh which uses the 3D model.)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
GXX="${GXX:-g++}"
INC="$(mktemp -d)"
cp "$HERE/_kvflash_test_stub_ggml.h"         "$INC/ggml.h"
cp "$HERE/_kvflash_test_stub_ggml_backend.h" "$INC/ggml-backend.h"
"$GXX" -std=c++17 -O2 -I "$INC" -I "$HERE/kvflash_ref" -I "$HERE" "$HERE/llama-kvflash-rowmodel.selftest.cpp" -o "$INC/kvf_row.exe"
"$INC/kvf_row.exe"
