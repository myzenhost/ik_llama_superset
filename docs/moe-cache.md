<!-- RESEARCH-DONE: https://github.com/ggml-org/llama.cpp/pull/30966 (upstream docs/moe-cache.md, read in full; this file is the fork's version) -->
# MoE cache

When the experts of MoE layers stay in host memory (`-cmoe`, `-ncmoe`, `-ot`), the MoE cache keeps the most used ones in device memory for small batches (decode, speculative decoding). Large batches still upload the experts they use.

| Argument | |
|---|---|
| `--moe-cache-mib N\|auto` | cache size in MiB, split among the GPUs like the layers (default: 0, no cache; `auto`: `--fit` picks it from the free device memory) |
| `--moe-cache-cpu-share F\|auto` | share of the cache misses computed on the CPU from the host experts instead of uploaded, once the cache is full (default: 0; `auto` measures the upload and CPU speeds at load) |
| `--moe-cache-policy slru\|classlru` | replacement policy (default: `slru`, the fork's segmented LRU; `classlru` is upstream's LRFU/MQ-like policy, #31233) |

The cache hits run on the device. With `--moe-cache-cpu-share F` the CPU computes a share `F` of the misses while the device computes the hits and uploads the other misses into the cache. This is faster when the CPU reads the host experts faster than the device can upload them, for example with fast RAM, a PCIe 4 link or experts that are not in pinned memory. The host experts are pinned with `-lm mlock` and `-lm dio`, but not when they are mmap'd (the default). The device uploads pinned experts at the full speed of the link, so the CPU share often does not help then. Until the cache is full every miss is uploaded, so the cache fills as fast as without the CPU share. The best share depends on the machine, the model and the cache size.

A CPU share above 0 (or `auto`) uses the `classlru` policy, which keeps the combined recency and frequency of every expert that the CPU part picks its misses by. With `--moe-cache-cpu-share 0` and the default policy the cache behaves as before.

What the CPU part does not cover: weights applied before the FFN, LoRA adapters, Hadamard-rotated experts and, on the graph path of the legacy architectures (`llama-build-context.cpp`), MoE layers with expert biases. Those layers keep uploading their misses.

## How it works

The graph builds the expert FFN twice: on the device from the cache slots, where the misses left to the CPU read zero slots (`n_zero` slots after the cached ones in each bank), and on the CPU from the host experts, where the experts computed on the device have id -1. `MUL_MAT_ID` on the CPU skips negative ids and writes zero rows. The two results are added before the down bias. The scheduler copies the inputs of the CPU split from the device before the device split computes, so the two run at the same time. `LLAMA_MOE_CACHE_TRACE=<file>` writes the selected experts of each lookup; `scripts/moe-cache-tune.py` replays them.

## Picking the arguments

Put the script in front of your server command:

```bash
python3 scripts/moe-cache-tune.py -- build/bin/llama-server -m model.gguf -cmoe -t 8 -c 32768
```

It starts the server 6 times, at two cache sizes × shares 0, 0.5 and 0.95, and times the generation of a few thousand tokens. That takes a few minutes, mostly model loading.

From these runs it fits the upload and CPU speeds of the machine, then replays the routing of all the runs through a copy of the cache logic for other sizes and shares. It prints the measured speeds, the predicted t/s and the recommended arguments: the lowest share within 5% of the fastest.

- **Server command:** keep your own arguments (`-t`, `-c`, expert placement, load mode); the speeds and the free device memory depend on them.
- **`--max-mib N`:** tunes a smaller cache than what fits on the device. The runs in `--out` are reused while the command stays the same, so trying another value is instant.
- **`--markdown`:** prints Markdown tables to paste in a PR.

The script is upstream's (llama.cpp e95426f16, #30966) with this fork's log lines and the `classlru` policy; on Windows it stops the server through the hot-swap control file so the cache logs its statistics.

## Measured on this fork

See `temp/research/findings/S69_MOE_CACHE_CPU_SHARE_AB_2026-11-10.md` for the A/B (RTX 5090, 3990X, DDR4-2133) and the reason for the default of `--moe-cache-cpu-share`.
