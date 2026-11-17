# LFM2 / LFM2-MoE forward-pass check (CPU, model-free)

Builds tiny random LFM2 (dense) and LFM2-MoE checkpoints, converts them with this tree's own converter
(`conversion/lfm2.py`), runs them through the engine on the CPU backend and compares **every position's
logits** with an independent float64 numpy implementation of the architecture
(`reference_forward.py`, written from the HF architecture, not from the engine's graph).

| file | role |
|---|---|
| `make_tiny_gguf.py <tree-root> <out-dir>` | random weights (seeded) for `dense`, `moe`, `moe2` (norm_topk_prob off, scale 2.5); writes `<kind>.gguf` (F32, no vocab), `weights_<kind>.npz`, `config_<kind>.json` |
| `run_logits.cpp` | `prog model.gguf out.bin T chunk`: decodes `T` fixed tokens in chunks of `chunk` (chunk < T carries the conv state across ubatches) and writes `[T, 64]` float32 logits |
| `reference_forward.py weights.npz config.json T out.bin` | the reference; tokens are `(7*i+3) % vocab`, same as the C++ side |
| `compare.py <out-dir>` | per model/chunk max diff, diff relative to logit range, argmax agreement; exit 0 on PASS |

Steps (mingw CPU build shown; any build that provides `libllama` + ggml works):

```
python make_tiny_gguf.py <tree> <out>
python reference_forward.py <out>/weights_dense.npz <out>/config_dense.json 12 <out>/ref_dense.bin      # and moe, moe2
python reference_forward.py <out>/weights_moe2.npz  <out>/config_moe.json   12 <out>/ref_moe2_wrongcfg.bin
g++ -O2 -std=c++17 -I <tree>/include -I <tree>/ggml/include run_logits.cpp -Wl,--start-group libllama.a ggml.a ggml-cpu.a ggml-base.a -Wl,--end-group -o run_logits
run_logits <out>/dense.gguf <out>/out_dense_c12.bin 12 12 ; ... 12 5   # and moe, moe2
python compare.py <out>
```

Controls inside `compare.py`, so a pass means the check can see: the dense output against the MoE reference (and the
reverse) must disagree widely; the MoE2 output against a reference built with the default-looking routing config
(norm on, scale 1.5) on the same weights must disagree, which shows the written `expert_weights_norm` /
`expert_weights_scale` keys reach the graph; chunk 5 must equal chunk 12 exactly.

Limits: the reference was written from memory of the HF `modeling_lfm2_moe.py`; `transformers` 4.57.6 on this host does
not know `lfm2_moe`, so it could not be run. The dense model is the control for the shared pieces (conv, attention,
rope, norms), and it agrees. Logits are compared at about 5e-4 of the logit range, not bit for bit; the engine's
K/V cache defaults to F16, the likely source of that gap (not isolated by a run with an F32 cache). Vocabulary handling, real tokenizers and real checkpoints are not covered.
