# kvflash_ref — upstream port-reference, PARTIALLY WIRED and DELIBERATELY REFUSED at runtime

Verbatim source from **Luce-Org/lucebox-hub** `main` (`server/src/common/`, `server/src/qwen3/`),
license **Apache-2.0** (DEC-S69-KVFLASH-LICENSE-CLEARED; verified via GitHub API). Fetched
2026-06-16 as the porting reference for the KVFlash KV-pager (rank-2 ik_llama C++ gap, findings
`temp/research/findings/S69_ENGINE_KERNEL_REF_2026-06-16.md` §3).

## Status, measured 2026-08-30 — read this before porting anything

This file previously said *"NOT yet wired … reference only — not in the build, not adapted"*.
**That was wrong**, and a doc that UNDER-claims is expensive: a reader assessing whether to port
PagedAttention would read "not in the build" and port something already present. Stale docs that
OVER-claim get caught when the feature fails; ones that UNDER-claim cause duplicated work that
looks like progress.

But the opposite summary — "it was wired" — is also wrong, and would be worse: it invites someone
to pass the flag. The accurate state is a third thing.

**Structurally wired.** 16 files under `src/` reference kvflash (control: 128 files there contain
"llama", so that count is a fact about the tree, not about the search):

| where | what |
|---|---|
| `llama-cparams.h:16` | `int32_t kvflash_pool_tokens` (0 = off) |
| `llama-context.h:407,473` | includes `kvflash_ref/kvflash_pager.h`; holds `std::unique_ptr<dflash::common::KvFlashPager> kvflash_pager` |
| `llama.cpp:8258, 8693-8715+` | plumbs the flag; constructs the pager; reshapes k_l/v_l to per-token rows for `attach()` |
| `llama-build-context.cpp:2104, 2150, 2247` | pool-sized read view `n_kv_eff`, used at 3 sites |
| `llama-kvflash.selftest.cpp`, `llama-kvflash-rowmodel.selftest.cpp` | both **PASS** |

**And actively REFUSED.** `llama.cpp:8668-8691` forces `kvflash_pool_tokens = 0` and logs
`--kvflash-pool=N IGNORED. The paged-KV integration is incomplete`. That guard is correct and
should not be removed casually — the reason is measured, not cautious:

> `attach()` and the pool-sized read views are connected, but `slot_for()`, `fill_slot_mask()` and
> `reselect()` are **not**. With `pool > 0` the K/V views shrink to `pool_tokens` while nothing maps
> logical positions to pool slots and nothing marks slot validity, so attention reads the first
> `pool_tokens` rows as if they were the context.
>
> Measured on this binary at temperature 0 with a fixed seed — output similarity to the no-flag
> baseline: **0.368** (pool < ctx), **0.162** (pool == ctx), and a ggml view-bounds assertion at
> pool > ctx. Forcing the pool to 0 restores the verified-inert path (**similarity 1.000**).

The failure mode is the dangerous kind: the text stays fluent and is wrong. The guard fails *safe*
rather than *stop* — aborting would turn a bad flag into an outage, and staying silent would serve
different text.

**So: the algorithm is not the gap. The wiring is.** The pager itself is sound; both selftests pass
host-side.

## What remains

- wire `slot_for(pos)` to replace the ring `kv.head` advance
- upload `fill_slot_mask()` **before every compute** — this is the specific precondition named in
  the guard's own error text for removing it
- `reselect()` every τ=64 tokens in the decode loop; bind `score_hook` to a drafter scorer wrapping
  **Qwen3.5-0.8B** (draft-compatible)
- flags `--kvflash auto|<tok>` `--kvflash-tau` `--kvflash-policy lru` `--prefill-drafter`
- **rule-1.12 A/B vs current q8_0 KV before any default.** NOT RUN — the A/B this file has
  prescribed since June still has not happened, and cannot until the wiring above lands.

TRAINING-FREE (pager + LRU; the drafter needs the 0.8B but no training).
Risk MED: q8_0 KV row byte-copy must match `ggml_row_size(type_k,...)`; don't break seq_cp/seq_rm.

Tracked: `S69-KVFLASH-README-STALE-UNDERCLAIMS-01`.
