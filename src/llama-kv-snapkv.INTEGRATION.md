# SnapKV/H2O integration — remaining wire edits before full ik_llama compile

Branch `s69-engine-kernels`. Core `llama-kv-snapkv.{h,cpp}` written + **standalone-compiled + selftest PASS**
(g++ -std=c++17, 8 host cases) + registered in `src/CMakeLists.txt`. TRAINING-FREE (no weights, no gguf change).
Grounded: `temp/research/findings/S69_ENGINE_KERNEL_REF_2026-06-16.md` §2 (SnapKV 2404.14469 + H2O 2306.14048, VERIFIED-FROM-REF).

The host core (scoring → keep-plan) is done + proven. Three bounded edits remain to fire it in the real build
(each touches a VERIFIED line; do at the next compile window, then rule-1.12 A/B vs current q8_0 KV):

| # | Edit | File:line (VERIFIED) | What |
|---|---|---|---|
| 1 | Flags + params | `common/common.cpp:717` (after `--grp-attn-n`) + `gpt_params`/`llama_cparams` | add `--snapkv-cap`/`--snapkv-window`/`--h2o-heavy`/`--h2o-recent`; store on params → cparams. Env mirror `LLAMA_ARG_*` @498. |
| 2 | Score capture | `src/llama-build-context.cpp:1646` (`cb(kq,"kq",il)`) | register graph callback copying post-softmax `kq` per-position sums into a host buffer per layer (obs-window for SnapKV, running col-sum for H2O). |
| 3 | Evict apply | `src/llama-context.{h,cpp}` `struct llama_kv_cache` (@35) | once after prefill: `snapkv_keep_plan()` → compact `kv.k_l/v_l` (ggml_get_rows by keep-idx) + rebuild `cells`/`head`. Per-decode when `used>cache_size`: `H2OAccumulator::evict_plan()` → same compaction OR `llama_kv_cache_seq_rm` (include/llama.h:805) per dropped pos. |

Risk (findings §2): LOW–MED. No model change → compiles against current binaries. Risk = plumbing post-softmax
scores out of the graph + correct cache compaction (cells/head bookkeeping). q8_0 KV rows compact via raw
byte gather (`ggml_row_size(type_k, n_embd_k_gqa)`).

NOT compiled into the full build yet (per user: all code up-to-compile; GPU busy w/ roster + power-fragility).
Next-after-this: KVFlash pager port (Apache-2.0, findings §3, cleanest tensor interface) then NSA (needs adapt-train).
