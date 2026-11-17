#!/usr/bin/env bash
# S69 #22-followon (M4) — build + run the RocketKV stage-2 HSA host selftest (CPU, no GPU/ggml).
# Proves GATEs A(bound soundness)/B(recall)/C(budget)/D(headdim-reduction)/E(noop).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
GXX="${GXX:-g++}"
OUT="$(mktemp -d)"
"$GXX" -std=c++17 -O2 -I "$HERE" \
    "$HERE/llama-kv-rocketkv.cpp" "$HERE/llama-kv-rocketkv.selftest.cpp" -o "$OUT/rocketkv.exe"
"$OUT/rocketkv.exe"
