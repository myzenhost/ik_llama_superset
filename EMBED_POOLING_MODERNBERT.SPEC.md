# ik_llama embed extensions — latent-attention pooling + ModernBERT arch (impl spec, ready-to-port)

Branch `s69-engine-kernels`. User directive 2026-06-16: add `--pooling`+MRL head, NV-Embed latent-attention
pooling, and ModernBERT arch to the FORK so the 3-4 unsupported embedders (gte-modernbert, modernbert-embed,
nv-embed) run on ik_llama — and add the same to harness+bench+pipeline (Harness-as-Model) so unsupported
models get them either way.

**Harness side = DONE + verified** (this is the "even models that don't have them get access" half):
`cortex/embedding/harness_pooling.py` (selftest PASS) provides mean/cls/last/latent-attention/max + MRL over
ANY model's token outputs; wired into `legal_qa_benchmark.py --pooling` (py_compile OK). So the capability
exists NOW regardless of the engine. The engine-native port below is the SPEED/unification half.

These ports are written here as a GROUNDED SPEC (exact insertion points read from the fork source) rather than
blind C++, because a half-written arch does not compile — port at a GPU/compile window, incrementally, where
each step is verifiable. NOT yet implemented in source.

---

## 1. Latent-attention pooling (`--pooling attention`) — BOUNDED, smallest first

Existing pooling enum (`include/llama.h:260-264`): UNSPECIFIED/-1, NONE/0, MEAN/1, CLS/2, LAST/3.
Flag parse (`common/common.cpp:776-781`): none/mean/cls/(last).
Pooling inputs built at `src/llama-build-context.cpp:398` (`build_inp_mean`), `:405` (`build_inp_cls`).

**Edits:**
1. `include/llama.h:264` — add `LLAMA_POOLING_TYPE_ATTENTION = 4,` after LAST.
2. `common/common.cpp` after :781 — `else if (value == "attention") { params.pooling_type = LLAMA_POOLING_TYPE_ATTENTION; }`.
3. `src/llama-build-context.cpp` build_pooling section (where MEAN/CLS/LAST are applied to the final
   hidden states) — add an ATTENTION branch. Two impls:
   - **Param-free (training-free, default)**: query = mean(H); `w = softmax(H · qᵀ)`; out = `Σ w_i H_i`.
     Mirrors `build_inp_mean` (it already builds the mean-reduction matrix) → reuse it for the query, then
     a `ggml_mul_mat(H, q)` → `ggml_soft_max` → weighted sum. All existing ops.
   - **Learned (NV-Embed)**: load the model's trained latent array `q_latent [n_latent, dim]` as a tensor
     (register in `llama-arch.cpp` tensor names for arch `nvembed`); same softmax-weighted sum over the
     multi-query max. Requires the gguf to carry the latent tensor (conversion change).
4. Conversion: `convert_hf_to_gguf.py` — for `NVEmbedModel`, export the `latent_attention_model.latents`
   weight + set `pooling_type=attention`.

Risk: LOW for param-free (existing ops, one branch). MED for learned (gguf tensor + conversion).

## 2. MRL head — NO engine change needed

MRL = truncate the output embedding to `dim` + L2-renormalize. This is post-pooling, client/harness side
(`harness_pooling.mrl_truncate`, done). Optional engine convenience: a `--embd-dim N` that slices the output
in `llama_get_embeddings` — trivial, not required. Keep MRL in the harness (already measured,
DEC-S69-MRL-TRUNC-MEASURED).

## 3. ModernBERT arch — LARGE port (full architecture)

ModernBERT (`ModernBertModel`, `model_type=modernbert`) differs from BERT: RoPE (not abs pos-embeds),
ALTERNATING local (sliding-window) / global attention every N layers, GeGLU MLP, no bias, pre-norm.
Reference impl in-tree: `build_bert()` at `src/llama-build-context.cpp:3250` (+ jina-bert-v2 variant @3293).

**Edits (port, do at compile window, mirror build_bert step by step):**
1. `src/llama-arch.h` + `src/llama-arch.cpp` — `LLM_ARCH_MODERNBERT` enum + `{ LLM_ARCH_MODERNBERT, "modernbert" }`
   name + tensor-name table (attn_qkv/out, mlp Wi/Wo for GeGLU, 2× layernorm, no biases).
2. `src/llama-hparams.*` — n_swa (local-attn window, ModernBERT=128), global-attn-every-N (=3),
   rope_theta_local vs rope_theta_global (ModernBERT uses two thetas).
3. `src/llama-build-context.cpp` — `build_modernbert()` mirroring `build_bert()` @3250 with:
   RoPE on Q/K (reuse the existing rope path), per-layer choice of `n_swa` window vs full attn
   (the fork already threads `n_swa` into FA @1607 — reuse), GeGLU MLP (`ggml_gelu` on the gate half).
4. Arch dispatch: `src/llama-model.cpp` / build switch (`:10371` area, `case LLM_ARCH_BERT:`) — add
   `case LLM_ARCH_MODERNBERT: result = build_modernbert(); break;`.
5. Model load: `src/llama-model.cpp` n_layer/n_head/hparams load for the new arch.
6. Conversion: `convert_hf_to_gguf.py` — `ModernBertModel` → write arch=modernbert + the two rope thetas +
   global-attn-interval + GeGLU tensor split. (Upstream llama.cpp has a ModernBERT converter to port from.)

Risk: HIGH (full arch). But bounded by the build_bert reference + the fork already supporting RoPE,
n_swa sliding-window, and GeGLU-style MLPs elsewhere. Port incrementally: arch+load → build graph →
convert one model → compare embeddings vs HF (rule 1.12).

---

## Sequence (one compile window, after the embedding roster frees the GPU)
1. `--pooling attention` (param-free) — smallest, validates the pooling-extension path.
2. NV-Embed learned latent (gguf tensor + conversion).
3. ModernBERT arch port (largest).
4. Per-model: convert gguf → embed via ik_llama → compare vs the harness PyTorch numbers (rule 1.12:
   quality must hold under quant; pick fp16/Q8). Then the unified ik_llama embed A/B answers the speed Q.

All of §1-3 are flag/arch-gated: absent the flags / the arch, the rebuilt binary is unchanged for existing
models. Pairs with the batched rebuild (SnapKV/H2O + KVFlash + qwen35-GGUF + Self-Extend + multi-LoRA).

### build_modernbert() precise deltas vs build_bert() (read from src/llama-build-context.cpp:3256, 2026-06-16)
build_bert is the template. ModernBERT-specific changes (config ground-truth modernbert-embed-base):
- PRE-NORM not post-norm (structural): build_bert adds-then-norms. ModernBERT = x = x + attn(norm(x)); x = x + mlp(norm(x)). Norm BEFORE each sub-layer.
- First-layer attn_norm = Identity (L0 attention_norm is nn.Identity; embeddings already normed by tok_norm).
- RoPE Q/K with TWO thetas: global layers (every global_attn_every_n_layers=3) theta=160000; others theta=10000. Per-layer freq_base into ggml_rope_ext (build_bert non-BERT branch already ropes Q/K).
- Alternating attention: non-global layers use sliding-window n_swa=local_attention=128; global=full. Fork threads n_swa already.
- GeGLU MLP: Wi: n_embd->2*intermediate(2*1152); split, gelu(gate)*up, Wo back. Mirror jina-bert-v2 GELU+PAR with Wi single gate+up.
- NO biases (attention_bias/mlp_bias/norm_bias false): NULL every *_b tensor.
- hparams: n_layer=22 n_embd=768 n_head=12 head=64 intermediate=1152 n_ctx_train=8192 vocab=50368 norm_eps=1e-5 pooling=mean; +2 rope thetas +global-interval (NEW hparams).
- Tensors: tok_embd, tok_norm(embeddings.norm), per-layer attn_norm(Identity L0), Wqkv, attn_out(Wo), mlp_norm, Wi, Wo, final_norm.
- Conversion: register ModernBertModel in convert_hf_to_gguf.py (upstream HAS it -> port the converter class).
IMPLEMENT IN PHASE 2 with COMPILE + numerical iteration (embeddings vs HF cosine >=0.99), NOT blind — wrong norm-order/theta = plausible-but-wrong vectors only the numerical test catches.

---

## TAXONOMY + METHODOLOGY + PER-CYCLE BUILD PLAN (S69 #MODERNBERT, appended 2026-06-20)

### TAXONOMY (where ModernBERT sits)
Family: bidirectional encoder embedder (BERT-lineage), NOT a decoder LM.
Distinctives vs vanilla BERT (build_bert @ llama-build-context.cpp:3250 is the mirror base):
- PRE-norm (RMSNorm/LayerNorm BEFORE each sub-layer; BERT is post-norm).
- ALTERNATING attention: global full-attention every 3rd layer; the rest local sliding-window (n_swa, e.g. 128).
- TWO RoPE thetas: global layers theta_global (~160000), local layers theta_local (~10000).
- GeGLU MLP (Wi -> split -> gelu(gate)*up -> Wo), no biases.
Arch dispatch family: LLM_ARCH_* (same registry as BERT/NOMIC-BERT). Pooling: works with the #1
latent-attention/mean/cls pooling already in this spec. Unlocks gte-modernbert, modernbert-embed (Floor/embed tier).

### METHODOLOGY (arxiv 2412.13663 ModernBERT, VERIFY-FROM-REF before each edit)
Encoder forward: tok_embd -> per layer { pre-norm; attn (rope theta by layer%3==0?global:local, swa mask for
local); residual; pre-norm; GeGLU MLP; residual } -> final norm -> pooling head -> (optional MRL slice) ->
L2-normalize. No causal mask (bidirectional). No KV cache (single forward). Numeric-faithful port REQUIRED:
each tensor op must match HF ModernBert; the acceptance test is cosine >= 0.99 vs HF sentence-transformers on a
sample (rule 1.12) — a near-miss (wrong theta/swa/norm-placement) silently degrades retrieval, so iterate.

### WIRING (exact, from §3 of this spec)
- llama-arch.{h,cpp}: register LLM_ARCH_MODERNBERT + tensor-name table (pre-norm names, Wi/Wo GeGLU, no bias).
- llama-hparams: n_swa, rope_theta_global/local, global-every-N.
- llama-build-context.cpp: build_modernbert() mirroring build_bert():3250 with the 4 distinctives above.
- llama-model.cpp arch dispatch (~10371): case LLM_ARCH_MODERNBERT -> build_modernbert().
- convert_hf_to_gguf.py: ModernBertModel class (tensor remap + hparams: theta pair, swa, global-every-3).

### PER-CYCLE BUILD PLAN (cron: ONE step, build_sm120.bat, verify, then next; default path untouched — new arch only fires for ModernBERT GGUFs)
1. Arch enum + tensor table + hparams (compiles, no behavior change) -> build.
2. build_modernbert() graph (pre-norm + GeGLU + 2-theta rope + alt-swa mask) -> build (compiles).
3. arch dispatch case -> build.
4. convert_hf_to_gguf.py ModernBertModel -> convert gte-modernbert-base -> GGUF.
5. ACCEPTANCE (rule 1.12): serve the GGUF on a temp port, embed a sample, cosine vs HF >= 0.99; iterate steps
   1-2 (theta/swa/norm) until it passes. THEN deploy + register in models.json.
SAFETY: ModernBERT is a NEW arch — it cannot affect existing models (dispatch only hits it for modernbert GGUFs).
Lower live-serving risk than EAGLE3; still build+verify each cycle.

---
## GROUND-TRUTH (web full-read 2026-06-21, rule 1.23) — answerdotai/ModernBERT-base config + HF modeling forward
CONFIG (exact): hidden_size=768 num_hidden_layers=22 num_attention_heads=12 head_dim=64 intermediate_size=1152
max_position_embeddings=8192 vocab_size=50368 global_attn_every_n_layers=3 global_rope_theta=160000.0
local_rope_theta=10000.0 local_attention=128 (sliding window TOTAL width = ±64) norm_eps=1e-5
attention_bias=false mlp_bias=false norm_bias=false hidden_activation=gelu classifier_pooling=mean
position_embedding_type=absolute(but RoPE used in attn) tie_word_embeddings=true pad=50283 cls/bos=50281 sep/eos=50282.
FORWARD (HF modeling_modernbert.py, verbatim ops):
- Embeddings: hidden = drop(norm(tok_embeddings(ids)))   [LayerNorm, no bias]
- EncoderLayer (PRE-NORM): attn_out = attn(attn_norm(x)); x = x + attn_out; x = x + mlp(mlp_norm(x))
  -> layer 0 attn_norm = nn.Identity() (embeddings already normed). mlp_norm always LayerNorm.
- Attention: qkv = Wqkv(x); view[*,3,n_head,head_dim]; unbind -> q,k,v; rotary(q,k) with theta = (layer%3==0? global:local);
  BIDIRECTIONAL (no causal). global layers = full attn; local layers = bidirectional sliding-window width 128 (±64).
- MLP GeGLU: input,gate = Wi(x).chunk(2,-1); out = Wo(gelu(input) * gate).   [Wi: n_embd->2*intermediate]
- final: x = final_norm(x); pooling = mean over tokens; (optional MRL slice); L2-normalize.
- ALL LayerNorm, NO biases anywhere (norm/attn/mlp bias all false). LayerNorm not RMSNorm.
TENSORS (gguf): token_embd.weight, token_embd_norm.weight(=embeddings.norm), blk.N.attn_norm.weight(absent L0),
blk.N.attn_qkv.weight(Wqkv, n_embd->3*n_embd), blk.N.attn_output.weight(Wo), blk.N.ffn_norm.weight(mlp_norm),
blk.N.ffn_up.weight(Wi, n_embd->2*n_ff), blk.N.ffn_down.weight(Wo, n_ff->n_embd), output_norm.weight(final_norm).
Fork already has: n_swa/n_swa_pattern + rope_freq_base_per_layer[] (2-theta), LayerNorm path in build_bert. Mirror create_bert_tensors @874 + build_bert @3250.
