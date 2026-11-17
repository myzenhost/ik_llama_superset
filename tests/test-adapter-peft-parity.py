#!/usr/bin/env python3
"""PEFT bit-exact parity harness for the fork's adapter path (row S69-ADAPTERS-PEFT-PARITY-HARNESS-2026-10-26-01).

What it does, end to end, with nothing downloaded:
  1. builds a random-weight HF LlamaForCausalLM (hidden 64, 2 layers, 4 heads, 2 KV heads so GQA is exercised, vocab 512, fp32, fixed seed)
     plus a tiny sentencepiece tokenizer, and converts it with this repo's convert_hf_to_gguf.py --outtype f32;
  2. per case, wraps the base in PEFT (get_peft_model + LoraConfig), fills EVERY adapter parameter with N(0, std) under a fixed seed
     (PEFT's zero-init B would make the comparison vacuous), saves it, and converts it with this repo's convert_lora_to_gguf.py;
  3. computes the PEFT fp32 logits (eager attention) on fixed token ids of length 1, 7 and 33, and runs the fork on the same ids through
     tests/test-adapter-peft-parity.cpp (every position's logits);
  4. PASS for a case  <=>  max |logit_fork - logit_peft| < TOL (1e-4, F32 model + F32 adapter) AND the adapter's own effect
     max |logit_peft - logit_peft_base| > EFFECT_MIN (1e-1), so a case cannot pass because the adapter does nothing.
     fp64 ARBITER: PEFT fp32 is itself only fp32-accurate. When |fork - peft32| >= TOL the same PEFT model is rerun in float64 (the
     F32 GGUF holds exactly the fp32 weights, so fp64 is the true value). The case still PASSES iff |fork - peft64| < ARB_K*TOL
     AND |fork - peft64| <= ARB_K * |peft32 - peft64|, i.e. the fork's own fp32 error is of the same order as the reference's.
     MEASURED on pattern_rslora (effective scale 8 on q/k/gate/up/down), |peft32-peft64| 9.561e-5 in every run:
       2026-10-26 gcc CPU build   |fork-peft64| 5.185e-5 (0.54x the reference's error)
       2026-10-27 MSVC CUDA build |fork-peft64| 1.001e-4 (1.05x), same harness, peft, seed, 4 threads, op_offload off
     The two fork builds differ only in compiler rounding (fp contraction / FMA), so ARB_K = 1 ("at least as close as PEFT
     fp32") made the verdict depend on the compiler: two equally accurate fp32 computations each land on either side of the
     other about half the time. ARB_K = 2 asks for the same ORDER of error, which both builds meet. Negative controls never
     use the arbiter (they must miss PEFT fp32 by > 100*TOL = 1e-2; measured misses are >= 2.2, so ARB_K does not reach them).

Controls built in:
  * base                 no adapter on either side: converter + graph parity of the base model itself (must PASS)
  * lora_r8_a16          POSITIVE control, plain LoRA r=8 alpha=16 on q/k/v/o/gate/up/down (must PASS at any HEAD)
  * neg_wrong_scale      NEGATIVE control, the same adapter GGUF run at scale 0.5 instead of 1.0 (must FAIL: diff > 100*TOL)
  * neg_adapter_removed  NEGATIVE control, the fork runs the base model against the PEFT adapter reference (must FAIL)
Cases that exercise a defect (expected to PASS once the defect is fixed, FAIL before):
  * rslora_r16_a32       use_rslora=True (PEFT scaling alpha/sqrt(r), peft tuners/lora/layer.py:221-224 at 0.17.1)
  * alpha_pattern        two distinct per-module alphas (peft tuners/lora/model.py:201-204, utils/other.py:1244-1252)
  * pattern_rslora       rank_pattern + alpha_pattern + use_rslora together
  * threshold_converted  the config shape PEFT's convert_to_lora emits for a float rank: r=1, lora_alpha=1, rank/alpha_pattern carry the real values
  * perseq_two_adapters  one --batch decode, sequence 0 under adapter A, sequence 1 under adapter B (per-sequence LoRA path)
  * only_<proj>          r=8 alpha=16 on ONE of the seven projections: localizes a positive-control failure to the graph site
                         (found the fused up/gate node dropping LoRA on ffn_up/ffn_gate, row S69-ADAPTERS-FUSED-UP-GATE-DROPS-LORA-2026-10-26-01)
MLA family (random DeepSeek-V2: q_lora_rank 32, kv_lora_rank 512, rope 64 + nope 16, all layers dense -> arch deepseek2;
  kv_lora_rank 512 + rope 64 are REQUIRED: src/llama-hparams.cpp aborts on a mainline-format MLA GGUF whose K/V MQA head
  sizes are not 576/512 or whose n_rot is not 64):
  * mla_base                 base parity of the MLA graph (src/graphs/build_deepseek2.cpp)
  * mla_lora_qa_kva_qb       LoRA on q_a_proj + kv_a_proj_with_mqa + q_b_proj (raw ggml_mul_mat before the fix: no effect in the fork)
  * mla_neg_adapter_removed  NEGATIVE control for the MLA family (must FAIL)
BitNet family (random BitNet b1.58, relu2, projections k*(+-1) so the converter's ternarization is exact):
  * bitnet_base / bitnet_lora_all (LoRA r=8 alpha=16 on all seven projections) / bitnet_neg_adapter_removed
DoRA family (row S69-ADAPTERS-DORA-RUNTIME-KIND-2026-10-26-01; PEFT use_dora=True). The magnitude m is left at PEFT's own init
  ||W||_row (peft 0.17.1 tuners/lora/dora.py:37-64, B = 0 at init) times (1 + 0.1 N(0,1)) per row, so d = m / ||W + s B A||_row != 1:
  * dora_r8_qv                DoRA r=8 alpha=16 on q/v (q exercises the converter's rope row permutation of m)
  * dora_all                  DoRA on all seven projections (incl. the fused up/gate node)
  * dora_rslora_pattern       DoRA + use_rslora + alpha_pattern (the folded scale must enter ||W + s B A|| too)
  * dora_neg_m_zeroed         NEGATIVE control: the same adapter with m = 0 in the converted file (must FAIL)
  * dora_neg_as_lora          NEGATIVE control: magnitude dropped, run as plain LoRA -- what TensorRT-LLM does (must FAIL)
  * dora_neg_adapter_removed  NEGATIVE control: base model against the DoRA reference (must FAIL)
  * dora_q8_base              REPORT only: DoRA over a Q8_0 base (d from the dequantised base) vs PEFT fp32 -- quantisation error, not gated
  * perseq_two_dora           two DIFFERENT DoRA adapters, one per sequence, in ONE --batch decode, each vs PEFT alone
  * perseq_dora_and_lora      sequence 0 DoRA, sequence 1 plain LoRA, one batch (the (d-1) term must be masked off sequence 1)
Lily family (row S69-ADAPTERS-W2-LILY-PER-SEQUENCE-ROUTER-2026-10-26-01; needs a PEFT with tuners/lily, e.g. --peft-src
  upstream/peft/src at a7e8ea4a1227). PEFT's router mean spans every token of its forward batch
  (tuners/lily/layer.py:150-179), so the reference is PEFT at BATCH 1 per sequence; the fork's mean is per sequence:
  * lily_alone                Lily r=8 stride_A=2 num_B=3 on all seven projections, each sequence decoded alone
  * lily_qv_scaling2          r=4 stride_A=1 num_B=2 scaling=2 on q/v/down (scaling folded into lily_b)
  * lily_batch_two_seqs       two 33-token sequences in ONE --batch decode, context-wide Lily, each vs PEFT alone
  * lily_neg_vs_peft_batched  NEGATIVE control: the same fork batch vs PEFT run on both sequences in one batch (must FAIL:
                              PEFT's mean mixes the sequences, the fork's does not)
  * lily_neg_router_zeroed    NEGATIVE control: router zeroed in the converted file (uniform mix; must FAIL)
  * lily_neg_adapter_removed  NEGATIVE control (must FAIL)
  * lily_perseq_two / lily_perseq_lily_and_lora   per-sequence adapters in one batch (Lily + Lily, Lily + LoRA)
Converter dispatcher (row S69-ADAPTERS-CONVERTER-DISPATCHER-2026-10-26-01): one case per dispatched peft_type / LoRA variant, each an
  exact closed form in convert_lora_to_gguf.py (no SVD, no PEFT convert_to_lora), plus loud-refusal cases:
  * qalora_g16                QA-LoRA (use_qalora, group 16). PEFT dispatches QALoraLinearVariant only for GPTQ layers (lora/gptq.py:
                              59-70), so the reference forces that same variant onto the fp32 nn.Linear; lora_A is [r, in/16]
  * qalora_plain_dispatch     use_qalora=True on a plain nn.Linear WITHOUT the forced variant: PEFT trains a plain LoRA (A [r, in]);
                              the converter must not expand it
  * adalora_r4 / vera_r8 / loha_r4 / lokr_r2_decomposed / lokr_full_factors / fourierft_n24   one case per non-LORA low-rank
                              peft_type (FourierFT: rank <= 2 per distinct column frequency, exact)
  * ia3_kv_down / ia3_all / ia3_neg_wrong_scale / ia3_neg_adapter_removed   IA3 vector-scale kind (input scale on feedforward
                              modules, output scale on the rest); refuse_ia3 became a served case
  * refuse_c3a / refuse_modules_to_save / refuse_lora_bias / refuse_pissa_unconverted   the converter must exit
                              non-zero and name the peft_type (or flag) and the row that would serve it
Remaining low-rank tuners (row S69-ADAPTERS-W3-CONVERTER-REMAINING-LOWRANK-TUNERS-2026-10-27-01; PEFT a7e8ea4a1227 via --peft-src),
  each folded to exact float64 factors in convert_lora_to_gguf.py (dense B=delta, A=I fallback when the natural rank >= min(out,in)):
  * vblora_r4 / vblora_topk_only        VB-LoRA top-k mix of the shared vector bank (topk-only reload rebuilds the logits)
  * unilora_indices_saved / unilora_indices_regenerated   Uni-LoRA theta_d gathered by saved or proj_seed-regenerated indices
  * tinylora_u8 / tinylora_tied_proj_regenerated          TinyLoRA U S (sum v P) V^T; tied groups, P regenerated from projection_seed
  * delora_r4 / gralora_k2 / gralora_hybrid / psoft_r4 / psoft_neumann / psoft_lowrank_pissa / paca_r4 / miss_r4 / miss_mini /
    randlora_r4 / pvera_r4 / waveft_db1 / waveft_db2_pattern / waveft_no_idwt   one case per closed form (PSOFT recomputes its frozen
                              A/B from the base weight's SVD; WaveFT folds the 2-D inverse DWT into an exact matrix)
  * bdlora_nb2                LoRA use_bdlora: block-diagonal A or B per module family expanded to dense factors
  * psoft_neg_adapter_removed / bdlora_neg_wrong_scale   NEGATIVE controls (must FAIL)
  * refuse_miss_bat / refuse_pvera_sampled / refuse_randlora_noproj / refuse_vera_noproj   refusals that name the variant and the
                              row (residual variants: S69-ADAPTERS-W4-CONVERTER-RESIDUAL-VARIANTS-2026-10-27-01)
GPT-OSS family (random GptOss: 4 experts top-2, fused gate_up_proj [E, hidden, 2I] / down_proj [E, I, hidden] nn.Parameters, attention
  sinks, swiglu-oai) -> arch gpt-oss, row S69-ADAPTERS-TARGET-PARAMETERS-MOE-EXPERT-LORA-2026-10-26-01:
  * gptoss_base                     base parity of the gpt-oss graph
  * gptoss_tp_experts               LoRA via target_parameters on BOTH expert tensors (+ q/v target_modules)
  * gptoss_tp_experts_only          target_parameters only (no target_modules)
  * gptoss_tp_neg_adapter_removed   NEGATIVE control (must FAIL)

usage: test-adapter-peft-parity.py --bin PATH/test-adapter-peft-parity[.exe] [--work DIR] [--cases a,b,...] [--json OUT.json] [--cuda-arm]
--cuda-arm (CUDA build only): the fork side keeps the library-default device op offload (PARITY_OP_OFFLOAD=1) and an expect=pass case is
judged |fork-peft| <= CUDA_DRIFT_K x (the device's own no-adapter drift on the same base) + TOL; negative controls are unchanged.
Exit 0 iff every selected case met its expectation; 1 otherwise; 77 (CTest SKIPPED) when torch/transformers/peft/sentencepiece are missing.
Without --work the assets go to a fresh temp dir, which is left in place (its path is printed) so a failure can be inspected.
Licence note: PEFT (https://github.com/huggingface/peft) is Apache-2.0; no PEFT code is copied here, PEFT is imported and run as the reference.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-4
EFFECT_MIN = 1e-1
NEG_MARGIN = 100.0  # a negative control must miss by at least NEG_MARGIN * TOL
ARB_K = 2.0  # fp64 arbiter: the fork's fp32 error may be up to ARB_K x the reference's own (module docstring, item 4)
CUDA_DRIFT_K = 2.0  # --cuda-arm: an adapter case may miss PEFT by up to CUDA_DRIFT_K x the device's own no-adapter drift + TOL.
# PROVISIONAL constant: the 2026-10-27 b1019g CUDA measurement (base 2.3e-3..3.9e-3, adapters 2.5e-3..1.478e-1) fits K=2 for every
# case but pattern_rslora, which this rule flags on purpose (30x the base drift, row S69-ADAPTERS-W2-PEFT-PARITY-CUDA-ARM-TOLERANCE-2026-10-27-01).
# BENCH-PENDING: the CUDA run that confirms or moves K.
LORA_STD = 0.2
DORA_M_STD = 0.1  # DoRA magnitude: m = ||W||_row * (1 + DORA_M_STD * N(0,1)), see make_adapter
SEQ_LENS = (1, 7, 33)
VOCAB = 512
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def _peft_src_from_argv() -> str | None:
    # --peft-src must act before `import peft`, i.e. before argparse runs
    for i, arg in enumerate(sys.argv):
        if arg == "--peft-src" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--peft-src="):
            return arg.split("=", 1)[1]
    return None


if _peft_src_from_argv():
    sys.path.insert(0, _peft_src_from_argv())

LILY_ROUTER_STD = 1.0  # router weights: larger than LORA_STD so the per-token router probabilities (and so their mean) differ

try:
    import numpy as np
    import torch
    import transformers
    import peft
    import sentencepiece as spm
except ImportError as e:  # FAILURES-SURFACED: printed and mapped to exit 77 (CTest SKIPPED)
    print(f"test-adapter-peft-parity: SKIPPED ({e})")
    sys.exit(77)


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str], cwd: str | None = None) -> str:
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        tail = "\n".join(p.stdout.splitlines()[-40:])
        raise RuntimeError(f"command failed rc={p.returncode}: {' '.join(cmd)}\n{tail}")
    return p.stdout


# ---------------------------------------------------------------- base model


def make_tokenizer(out_dir: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    words = ["".join(rng.choice(letters, size=int(rng.integers(2, 7)))) for _ in range(300)]
    lines = [" ".join(rng.choice(words, size=12)) for _ in range(2000)]
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(lines), model_prefix=os.path.join(out_dir, "tokenizer"), vocab_size=400, model_type="unigram",
        hard_vocab_limit=False, unk_id=0, bos_id=1, eos_id=2, pad_id=-1, minloglevel=2,
    )
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"tokenizer_class": "LlamaTokenizer", "add_bos_token": True, "add_eos_token": False,
                   "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}, f)


def make_base(base_dir: str, seed: int, family: str = "llama"):
    if family == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM
        cfg = LlamaConfig(
            vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=512, rms_norm_eps=1e-5, rope_theta=10000.0, tie_word_embeddings=False, initializer_range=0.1,
            bos_token_id=1, eos_token_id=2, attn_implementation="eager",
        )
        cls = LlamaForCausalLM
    elif family == "mla":
        # DeepSeek-V2 MLA (q_lora_rank set, so q goes q_a -> norm -> q_b; kv through kv_a_proj_with_mqa), every layer dense
        # (first_k_dense_replace = n_layers) so the graph under test is the attention path. Converts to arch deepseek2.
        from transformers import DeepseekV2Config, DeepseekV2ForCausalLM
        cfg = DeepseekV2Config(
            vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            q_lora_rank=32, kv_lora_rank=512, qk_nope_head_dim=16, qk_rope_head_dim=64, v_head_dim=16,
            first_k_dense_replace=2, n_routed_experts=4, n_shared_experts=1, moe_intermediate_size=32, num_experts_per_tok=2,
            max_position_embeddings=512, rms_norm_eps=1e-5, rope_theta=10000.0, tie_word_embeddings=False, initializer_range=0.1,
            bos_token_id=1, eos_token_id=2, attn_implementation="eager",
        )
        cls = DeepseekV2ForCausalLM
    elif family == "moe":
        # Mixtral (4 experts, top-2, per-expert w1/w2/w3 nn.Linear) -> arch llama with stacked ffn_{gate,down,up}_exps, i.e. the
        # llm_build_moe_ffn / llm_build_lora_mm_id path (row S69-ADAPTERS-W2-MULTI-DORA-BATCH-AND-MOE-2026-10-26-01)
        from transformers import MixtralConfig, MixtralForCausalLM
        cfg = MixtralConfig(
            vocab_size=VOCAB, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            num_local_experts=4, num_experts_per_tok=2, router_jitter_noise=0.0, sliding_window=None,
            max_position_embeddings=512, rms_norm_eps=1e-5, rope_theta=10000.0, tie_word_embeddings=False, initializer_range=0.1,
            bos_token_id=1, eos_token_id=2, attn_implementation="eager",
        )
        cls = MixtralForCausalLM
    elif family == "bitnet":
        # BitNet b1.58 (relu2 MLP, attn/ffn sub-norms) -> arch bitnet. HF runs these projections as plain nn.Linear in full
        # precision; conversion/bitnet.py weight_quant() ternarizes them (scale = mean|w|). Weights of the form k * (+-1) with no
        # zeros are a fixed point of that (mean|w| = k), so the F32 GGUF holds exactly what PEFT computes with.
        from transformers import BitNetConfig, BitNetForCausalLM
        cfg = BitNetConfig(
            vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            # tied: gguf's BITNET arch has no output tensor (the real 1bitLLM / 2B-4T checkpoints tie it), so an lm_head cannot map
            hidden_act="relu2", max_position_embeddings=512, rms_norm_eps=1e-5, rope_theta=10000.0, tie_word_embeddings=True,
            initializer_range=0.1, bos_token_id=1, eos_token_id=2, attn_implementation="eager",
        )
        cls = BitNetForCausalLM
    elif family == "gptoss":
        # GPT-OSS: experts are FUSED 3-D nn.Parameters (gate_up_proj [E, hidden, 2I] interleaved gate/up, down_proj [E, I, hidden]),
        # so PEFT can only adapt them through target_parameters (lora.ParamWrapper). intermediate 48 != hidden 64 keeps every
        # expert shape distinct, so an orientation slip fails a shape check instead of hiding. rope_scaling None (plain rope; the
        # real checkpoints use yarn, which is not under test here); sliding_window 128 > every test length.
        from transformers import GptOssConfig, GptOssForCausalLM
        cfg = GptOssConfig(
            vocab_size=VOCAB, hidden_size=64, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, num_local_experts=4, num_experts_per_tok=2, sliding_window=128, rope_scaling=None,
            max_position_embeddings=512, rms_norm_eps=1e-5, rope_theta=10000.0, tie_word_embeddings=False, initializer_range=0.1,
            bos_token_id=1, eos_token_id=2, attn_implementation="eager",
        )
        cls = GptOssForCausalLM
    elif family == "gemma3":
        # Gemma3 text model: tied lm_head (the shipped shape) and an embedding scaled by sqrt(hidden) inside the model, so
        # an embedding LoRA delta must land BEFORE that scale (PEFT scales the delta too) and the tied output must not pick the
        # embedding adapter up by name (row S69-ENGREV-PEFT-EMBED-LORA-NOT-APPLIED-LEGACY-2026-10-27-01 + its TIED-OUTPUT twin)
        from transformers import Gemma3TextConfig, Gemma3ForCausalLM
        cfg = Gemma3TextConfig(
            vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, query_pre_attn_scalar=16, sliding_window=128, max_position_embeddings=512, rms_norm_eps=1e-6,
            tie_word_embeddings=True, initializer_range=0.1, bos_token_id=1, eos_token_id=2, pad_token_id=0,
            attn_implementation="eager",
        )
        cls = Gemma3ForCausalLM
    else:
        raise ValueError(family)
    torch.manual_seed(seed)
    model = cls(cfg).float().eval()
    if family == "bitnet":
        gt = torch.Generator().manual_seed(seed + 2)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if any(f".{t}.weight" in n for t in TARGETS):
                    k = 0.05 + 0.1 * float(torch.rand(1, generator=gt))
                    p.copy_(k * torch.where(torch.randn(p.shape, generator=gt) >= 0, 1.0, -1.0))
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n:  # non-unit norms, so a norm-handling slip in the converter or graph is visible
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
    os.makedirs(base_dir, exist_ok=True)
    model.save_pretrained(base_dir, safe_serialization=True)
    make_tokenizer(base_dir, seed)
    return model


def load_base(base_dir: str):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32, attn_implementation="eager").eval()


def convert_base(py: str, base_dir: str, out: str, family: str) -> None:
    if family in ("llama", "bitnet", "moe", "gemma3"):
        run([py, os.path.join(REPO, "convert_hf_to_gguf.py"), base_dir, "--outtype", "f32", "--outfile", out])
    else:
        # DeepseekV2Model.set_vocab (conversion/deepseek.py) and GptOssModel.set_vocab (conversion/gpt_oss.py) want a
        # GPT-2/tiktoken tokenizer; the vocab is irrelevant to a logits comparison on raw token ids, so this re-entry runs the
        # same converter with the sentencepiece vocab writer instead.
        run([py, os.path.abspath(__file__), "--convert-spm-vocab", base_dir, "--outtype", "f32", "--outfile", out])


def _convert_spm_vocab_main(argv: list[str]) -> int:
    import importlib.util
    spec = importlib.util.spec_from_file_location("convert_hf_to_gguf", os.path.join(REPO, "convert_hf_to_gguf.py"))
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    import conversion.deepseek as _ds
    _ds.DeepseekV2Model.set_vocab = lambda self: self._set_vocab_sentencepiece()
    import conversion.gpt_oss as _go  # GptOssModel.set_vocab wants the o200k GPT-2-style tokenizer; same reasoning as above
    _go.GptOssModel.set_vocab = lambda self: self._set_vocab_sentencepiece()
    sys.argv = [spec.origin] + argv
    conv.main()
    return 0


# ---------------------------------------------------------------- adapters


def _peft_skip_torchao() -> None:
    # MEASURED 2026-10-26: peft 0.17.1 + torchao 0.18.0 -> get_peft_model raises "cannot import name 'LinearActivationQuantizedTensor' from
    # 'torchao.quantization'" (peft/tuners/lora/torchao.py:147-153 imports it whenever torchao is installed). The base here is plain fp32
    # nn.Linear, so the torchao dispatcher can never match; telling that module torchao is absent changes nothing for these layers.
    try:
        import peft.tuners.lora.torchao as _pt
    except ImportError:
        return
    _pt.is_torchao_available = lambda: False


# PEFT config class per harness peft_type (each tuner's own config; dropout only where the class has lora_dropout)
_PEFT_CONFIGS = {"lily": "LilyConfig", "adalora": "AdaLoraConfig", "vera": "VeraConfig", "loha": "LoHaConfig", "lokr": "LoKrConfig",
                 "ia3": "IA3Config", "fourierft": "FourierFTConfig", "c3a": "C3AConfig",
                 "vblora": "VBLoRAConfig", "unilora": "UniLoraConfig", "tinylora": "TinyLoraConfig", "delora": "DeloraConfig",
                 "gralora": "GraloraConfig", "psoft": "PsoftConfig", "paca": "PacaConfig", "miss": "MissConfig",
                 "randlora": "RandLoraConfig", "pvera": "PveraConfig", "waveft": "WaveFTConfig"}
# parameter-name markers of trainable adapter tensors filled with N(0, std) (PEFT's zero inits would make a case vacuous). Buffers
# (TinyLoRA's SVD A/B and P, Uni-LoRA indices, DeLoRA w_norm, VeRA/RandLoRA/PVeRA projections) are not parameters and stay as PEFT made them.
_ADAPTER_PARAM_MARKERS = ("lora_", "lily_A", "lily_B", "vera_lambda", "hada_", "lokr_", "ia3_l", "fourierft_spectrum", "c3a_kernel",
                          "vblora_", "unilora_theta_d", "tinylora_v", "delora_", "gralora_", "psoft_R", "paca_delta", "miss_block",
                          "randlora_lambda", "randlora_gamma", "pvera_lambda", "waveft_spectrum")


def _get_peft_model_forced_qalora(model, cfg):
    """get_peft_model with QALoraLinearVariant dispatched onto plain nn.Linear. PEFT resolves that variant only in its GPTQ layer
    (peft tuners/lora/gptq.py:59-70); the variant's forward (variants.py:545-582: group-mean pool over qalora_group_size, times
    in/group, then B A) does not depend on the base being quantised, so this is the QA-LoRA math on an fp32 base."""
    from peft import get_peft_model
    from peft.tuners.lora import layer as _ll
    from peft.tuners.lora.variants import QALoraLinearVariant
    orig = _ll.Linear.resolve_lora_variant

    def forced(self, *, config, **kwargs):
        if getattr(config, "use_qalora", False) and not config.use_dora:
            return QALoraLinearVariant()
        return orig(self, config=config, **kwargs)

    _ll.Linear.resolve_lora_variant = forced
    try:
        return get_peft_model(model, cfg)
    finally:
        _ll.Linear.resolve_lora_variant = orig


def make_adapter(base_dir: str, out_dir: str, lora_kwargs: dict, seed: int):
    from peft import LoraConfig, get_peft_model
    _peft_skip_torchao()
    model = load_base(base_dir)
    kw = dict(lora_kwargs)
    ptype = kw.pop("peft_type", "lora")
    std = float(kw.pop("_std", LORA_STD))
    plain_qalora = bool(kw.pop("_qalora_plain_dispatch", False))
    convert_init = bool(kw.pop("_convert_init", False))  # save the trained PiSSA/OLoRA adapter converted to a plain LoRA (r x2, alpha x2)
    if isinstance(kw.get("use_bdlora"), dict):
        from peft.tuners.lora.config import BdLoraConfig
        kw["use_bdlora"] = BdLoraConfig(**kw["use_bdlora"])
    if ptype in _PEFT_CONFIGS:
        try:
            cfg_cls = getattr(peft, _PEFT_CONFIGS[ptype])
        except AttributeError as e:
            raise RuntimeError(f"peft {peft.__version__} has no {_PEFT_CONFIGS[ptype]} ({e}); pass --peft-src <peft clone>/src") from e
        cfg = cfg_cls(task_type="CAUSAL_LM", **kw)
    elif ptype == "lora":
        cfg = LoraConfig(task_type="CAUSAL_LM", lora_dropout=0.0, **kw)
    else:
        raise ValueError(ptype)
    if ptype == "lora" and kw.get("use_qalora") and not plain_qalora:
        pm = _get_peft_model_forced_qalora(model, cfg)
    else:
        pm = get_peft_model(model, cfg)
    init_dir = None
    if convert_init:
        # PEFT keeps the PiSSA/OLoRA initial adapter to turn the trained one into a plain LoRA on the ORIGINAL base:
        # save_pretrained(path_initial_model_for_weight_conversion=) -> delta = B A - B0 A0, rank 2r, alpha 2 alpha (peft 0.17.1 peft_model.py)
        init_dir = out_dir + "_init"
        pm.save_pretrained(init_dir)
    g = torch.Generator().manual_seed(seed)
    n_filled = 0
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_magnitude_vector" in n:
                # DoRA magnitude: PEFT initialised it to ||W||_row while B was 0; keep that scale (a N(0, std) magnitude would
                # be ~1/30 of ||W|| and mostly negative) and perturb it per row so m differs from ||W + s B A||.
                p.mul_(1.0 + DORA_M_STD * torch.randn(p.shape, generator=g))
                n_filled += 1
            elif "ia3_l" in n:
                # IA3 scale vectors start at 1.0 (identity); perturb around 1 so the adapter is neither vacuous nor a zeroing of the
                # activation it scales (an N(0, std) vector would be ~0)
                p.copy_(1.0 + std * torch.randn(p.shape, generator=g))
                n_filled += 1
            elif "psoft_R" in n and n.rsplit(".", 1)[-1] in ("vector_a", "vector_b"):
                # PSOFT magnitudes scale R (diag(b) R diag(a), PEFT init 1.0): kept near 1, else R ~ 0 and delta ~ -B0 A0 hides R
                p.copy_(1.0 + std * torch.randn(p.shape, generator=g))
                n_filled += 1
            elif any(m in n for m in _ADAPTER_PARAM_MARKERS):
                # named_parameters() lists a Lily A/B shared across layers once (model.py:102-166), so sharing is kept
                p.copy_(std * torch.randn(p.shape, generator=g))
                n_filled += 1
            elif "lily_router" in n:
                p.copy_(LILY_ROUTER_STD * torch.randn(p.shape, generator=g))
                n_filled += 1
    if n_filled == 0:
        raise RuntimeError(f"no LoRA parameters were created for {lora_kwargs}")
    pm.eval()
    if init_dir is not None:
        pm.save_pretrained(out_dir, path_initial_model_for_weight_conversion=init_dir)
    else:
        pm.save_pretrained(out_dir)
    # Scaled embeddings (Gemma3TextScaledWordEmbedding, embed_scale = sqrt(n_embd)): newer PEFT multiplies the embedding-LoRA delta by
    # embed_scale (tuners_utils.py:1976-1990, lora/layer.py:1548-1550 at a7e8ea4); peft 0.17.1 does not (no 'embed_scale' in the package),
    # so its delta is ~embed_scale times smaller than the engine's (base+delta)*scale. Reference only (the saved adapter stays as PEFT
    # wrote it): scale the in-memory lora_embedding_B by embed_scale when the installed PEFT lacks the handling.
    if any(isinstance(getattr(m, "embed_scale", None), (int, float, torch.Tensor)) or hasattr(m, "embed_scale") for m in pm.modules()):
        import inspect
        src = inspect.getsource(sys.modules["peft.tuners.tuners_utils"]) + inspect.getsource(sys.modules["peft.tuners.lora.layer"])
        if "embed_scale" not in src:
            with torch.no_grad():
                for m in pm.modules():
                    if hasattr(m, "lora_embedding_B") and hasattr(m, "base_layer") and hasattr(m.base_layer, "embed_scale"):
                        for k in m.lora_embedding_B:
                            m.lora_embedding_B[k].mul_(float(m.base_layer.embed_scale))
    return pm


def adapter_variant(src_dir: str, dst_dir: str, mode: str) -> None:
    """A copy of a saved PEFT DoRA adapter for a NEGATIVE control: 'zero_m' sets every lora_magnitude_vector to 0 (d = 0);
    'drop_m' removes the magnitudes and sets use_dora false, i.e. the file a DoRA-ignorant engine effectively serves."""
    from safetensors.torch import load_file, save_file
    os.makedirs(dst_dir, exist_ok=True)
    sd = load_file(os.path.join(src_dir, "adapter_model.safetensors"))
    if mode == "zero_router":
        # Lily NEGATIVE control: every router zeroed -> uniform 1/num_B mix of the B's for every token
        if not any("lily_router" in k for k in sd):
            raise RuntimeError(f"{src_dir}: no lily_router tensors (not a Lily adapter)")
        sd = {k: (torch.zeros_like(v) if "lily_router" in k else v) for k, v in sd.items()}
        save_file(sd, os.path.join(dst_dir, "adapter_model.safetensors"))
        with open(os.path.join(src_dir, "adapter_config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        with open(os.path.join(dst_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=1)
        return
    n_m = sum(1 for k in sd if "lora_magnitude_vector" in k)
    if n_m == 0:
        raise RuntimeError(f"{src_dir}: no lora_magnitude_vector tensors (not a DoRA adapter)")
    if mode == "zero_m":
        sd = {k: (torch.zeros_like(v) if "lora_magnitude_vector" in k else v) for k, v in sd.items()}
    elif mode == "drop_m":
        sd = {k: v for k, v in sd.items() if "lora_magnitude_vector" not in k}
    else:
        raise ValueError(mode)
    save_file(sd, os.path.join(dst_dir, "adapter_model.safetensors"))
    with open(os.path.join(src_dir, "adapter_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    if mode == "drop_m":
        cfg["use_dora"] = False
    with open(os.path.join(dst_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)


@torch.no_grad()
def hf_logits(model, seqs: list[list[int]]) -> list[np.ndarray]:
    return [model(torch.tensor([s], dtype=torch.long)).logits[0].float().numpy() for s in seqs]


@torch.no_grad()
def hf_logits64(model, seqs: list[list[int]]) -> list[np.ndarray]:
    """The same model's logits in float64 (a deep copy; the fp32 model is left as is). Used only by the fp64 arbiter."""
    import copy
    m = copy.deepcopy(model).double()
    return [m(torch.tensor([s], dtype=torch.long)).logits[0].double().numpy() for s in seqs]


@torch.no_grad()
def hf_logits_gated(model, seqs: list[list[int]], gate: str, starts=None, gate_token=None) -> list[np.ndarray]:
    """PEFT reference for a position-gated adapter (row S69-ADAPTERS-POSITION-GATED-LORA-UNO-SPEC-2026-10-26-01): every LoRA Linear
    returns base(x) + mask * (adapter(x) - base(x)) where mask is 1 on exactly the gated rows (draft_rows: position >= starts[i];
    mask_token: token == gate_token) and 0 elsewhere, so the adapter changes those rows' K/V/FFN and nothing else.
    Written from the math (the adapter is applied only on the gated rows), no code taken from any implementation."""
    from peft.tuners.lora import LoraLayer
    holder = {"mask": None}
    patched = []
    for m in model.modules():
        if isinstance(m, LoraLayer) and hasattr(m, "base_layer"):
            orig = m.forward

            def fwd(x, *a, _orig=orig, _m=m, **kw):
                full = _orig(x, *a, **kw)
                base = _m.base_layer(x)
                return base + holder["mask"].to(full.dtype) * (full - base)
            m.forward = fwd
            patched.append((m, orig))
    try:
        out = []
        for i, s in enumerate(seqs):
            if gate == "draft_rows":
                mk = [1.0 if k >= starts[i] else 0.0 for k in range(len(s))]
            else:
                mk = [1.0 if t == gate_token else 0.0 for t in s]
            holder["mask"] = torch.tensor(mk, dtype=torch.float32).reshape(1, -1, 1)
            out.append(model(torch.tensor([s], dtype=torch.long)).logits[0].float().numpy())
        return out
    finally:
        for m, orig in patched:
            m.forward = orig


# ---------------------------------------------------------------- fork side


def fork_logits(bin_path: str, model_gguf: str, seqs: list[list[int]], work: str, tag: str, loras=(), seq_loras=(), batch=False, draft_starts=(),
                vocab_stride=0) -> list[np.ndarray]:
    tok = os.path.join(work, f"{tag}.tokens.txt")
    out = os.path.join(work, f"{tag}.logits.bin")
    with open(tok, "w", encoding="utf-8") as f:
        for s in seqs:
            f.write(" ".join(str(t) for t in s) + "\n")
    cmd = [bin_path, model_gguf, tok, out]
    for path, scale in loras:
        cmd += ["--lora", path, repr(float(scale))]
    for sq, path, scale in seq_loras:
        cmd += ["--seq-lora", str(sq), path, repr(float(scale))]
    for sq, pos in draft_starts:
        cmd += ["--draft-start", str(sq), str(pos)]
    if batch:
        cmd.append("--batch")
    if vocab_stride:
        cmd += ["--vocab-subset", str(vocab_stride)]  # the dumper writes only the ids t with t % stride == 0 (and checks the fast path ran)
    run(cmd)
    res = []
    with open(out, "rb") as f:
        n_seq, n_vocab = struct.unpack("<ii", f.read(8))
        for _ in range(n_seq):
            (n_tok,) = struct.unpack("<i", f.read(4))
            res.append(np.frombuffer(f.read(4 * n_tok * n_vocab), dtype=np.float32).reshape(n_tok, n_vocab))
    if len(res) != len(seqs):
        raise RuntimeError(f"fork wrote {len(res)} sequences, expected {len(seqs)}")
    return res


def max_abs(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    m = 0.0
    for x, y in zip(a, b):
        if x.shape[0] != y.shape[0]:
            return float("inf")
        n = min(x.shape[1], y.shape[1])  # the GGUF vocab equals the HF vocab here; min() guards a padded vocab
        m = max(m, float(np.max(np.abs(x[:, :n].astype(np.float64) - y[:, :n].astype(np.float64)))))
    return m


# ---------------------------------------------------------------- cases

BASE_LORA = dict(r=8, lora_alpha=16, target_modules=TARGETS)
LM_HEAD_LORA = dict(r=8, lora_alpha=16, target_modules=["lm_head"])
MLA_LORA = dict(r=8, lora_alpha=16, target_modules=["q_a_proj", "kv_a_proj_with_mqa", "q_b_proj"])
DORA_QV = dict(r=8, lora_alpha=16, use_dora=True, target_modules=["q_proj", "v_proj"])
DORA_ALL = dict(r=8, lora_alpha=16, use_dora=True, target_modules=TARGETS)
MOE_TARGETS = ["q_proj", "v_proj", "w1", "w2", "w3"]  # w1 = gate, w2 = down, w3 = up of every expert
MOE_LORA = dict(r=8, lora_alpha=16, target_modules=MOE_TARGETS)
MOE_DORA = dict(r=8, lora_alpha=16, use_dora=True, target_modules=MOE_TARGETS)
LILY_ALL = dict(peft_type="lily", r=8, stride_A=2, num_B=3, scaling=1.0, init_weights=False, target_modules=TARGETS)
LILY_QV = dict(peft_type="lily", r=4, stride_A=1, num_B=2, scaling=2.0, init_weights=False, target_modules=["q_proj", "v_proj", "down_proj"])
# per-sequence pairs: name -> (PEFT kwargs, seed offset)
PERSEQ_ADAPTERS = {
    "lora8": (BASE_LORA, 7),
    "lora4": (dict(r=4, lora_alpha=8, target_modules=TARGETS), 11),
    "dora_qv": (DORA_QV, 7),
    "dora_all4": (dict(r=4, lora_alpha=8, use_dora=True, target_modules=TARGETS), 13),
    "moe_dora8": (MOE_DORA, 7),
    "moe_dora4": (dict(r=4, lora_alpha=8, use_dora=True, target_modules=MOE_TARGETS), 13),
    "moe_lora4": (dict(r=4, lora_alpha=8, target_modules=MOE_TARGETS), 11),
    "lily_all": (LILY_ALL, 17),
    "lily_qv": (LILY_QV, 19),
}
# Converter dispatcher (row S69-ADAPTERS-CONVERTER-DISPATCHER-2026-10-26-01). `_std` overrides LORA_STD where the tuner's product
# structure (Hadamard / Kronecker / lambda scaling) would otherwise leave an effect below EFFECT_MIN.
QALORA_G16 = dict(r=8, lora_alpha=16, use_qalora=True, qalora_group_size=16, target_modules=TARGETS)
ADALORA_R4 = dict(peft_type="adalora", init_r=4, target_r=2, lora_alpha=8, total_step=10, target_modules=TARGETS, _std=0.3)
VERA_R8 = dict(peft_type="vera", r=8, target_modules=TARGETS, _std=1.0)
LOHA_R4 = dict(peft_type="loha", r=4, alpha=8, rank_pattern={"v_proj": 2}, alpha_pattern={"o_proj": 2}, target_modules=TARGETS, _std=0.4)
LOKR_R2 = dict(peft_type="lokr", r=2, alpha=4, decompose_both=True, target_modules=TARGETS, _std=0.4)
LOKR_FULL = dict(peft_type="lokr", r=8, alpha=4, target_modules=TARGETS, _std=0.3)
# FourierFT: 24 random frequencies per module (n_frequency_pattern gives v_proj 8); delta = Re(ifft2(sparse spectrum)) * scaling has
# rank <= 2 * #distinct column frequencies, written exactly as B @ A by the converter (the indices are regenerated from
# random_loc_seed exactly as PEFT does at load, fourierft/layer.py:72-78)
FOURIERFT_N24 = dict(peft_type="fourierft", n_frequency=24, n_frequency_pattern={"v_proj": 8}, scaling=150.0, random_loc_seed=777,
                     target_modules=TARGETS, _std=2.0)
# Remaining low-rank tuners (row S69-ADAPTERS-W3-CONVERTER-REMAINING-LOWRANK-TUNERS-2026-10-27-01): each is rewritten by the
# converter to an exact B @ A (or the exact dense pair when its natural rank reaches min(out, in)).
ROW_REMAINING = "S69-ADAPTERS-W3-CONVERTER-REMAINING-LOWRANK-TUNERS-2026-10-27-01"
# variants still refused once that row closed (projection regeneration, UniLoRA all-linear, TinyLoRA non-Linear, PVeRA sampling)
ROW_RESIDUAL = "S69-ADAPTERS-W4-CONVERTER-RESIDUAL-VARIANTS-2026-10-27-01"
VBLORA_R4 = dict(peft_type="vblora", r=4, num_vectors=16, vector_length=16, topk=2, target_modules=TARGETS, _std=0.2)
VBLORA_TOPK = dict(VBLORA_R4, save_only_topk_weights=True)
UNILORA_IDX = dict(peft_type="unilora", r=4, theta_d_length=256, save_indices=True, target_modules=TARGETS, _std=1.0)
UNILORA_NOIDX = dict(peft_type="unilora", r=4, theta_d_length=256, target_modules=TARGETS, _std=1.0)  # indices regenerated
TINYLORA_U8 = dict(peft_type="tinylora", r=2, u=8, target_modules=TARGETS, _std=0.2)
TINYLORA_TIED_NOPROJ = dict(peft_type="tinylora", r=2, u=8, weight_tying=0.5, save_projection=False, target_modules=TARGETS, _std=0.2)
DELORA_R4 = dict(peft_type="delora", r=4, delora_lambda=15, rank_pattern={"v_proj": 2}, target_modules=TARGETS, _std=2.0)
GRALORA_K2 = dict(peft_type="gralora", r=8, gralora_k=2, alpha=16, target_modules=TARGETS, _std=0.1)
GRALORA_HYBRID = dict(GRALORA_K2, hybrid_r=2)
PSOFT_R4 = dict(peft_type="psoft", r=4, psoft_alpha=4, target_modules=TARGETS, _std=0.4)
PSOFT_NEUMANN = dict(peft_type="psoft", r=4, psoft_alpha=8, use_cayley_neumann=True, target_modules=TARGETS, _std=0.4)
PSOFT_LOWRANK_PISSA = dict(peft_type="psoft", r=4, psoft_alpha=4, psoft_svd="lowrank", ab_svd_init="pissa_init", psoft_orth=False,
                           target_modules=TARGETS, _std=0.4)
PACA_R4 = dict(peft_type="paca", r=4, paca_alpha=8, target_modules=TARGETS, _std=0.1)
MISS_R4 = dict(peft_type="miss", r=4, target_modules=TARGETS, _std=0.1)
MISS_MINI = dict(peft_type="miss", r=4, mini_r=4, init_weights="mini", target_modules=TARGETS, _std=0.1)
RANDLORA_R4 = dict(peft_type="randlora", r=4, randlora_alpha=8, save_projection=True, target_modules=TARGETS, _std=0.1)
PVERA_R4 = dict(peft_type="pvera", r=4, save_projection=True, target_modules=TARGETS, _std=0.4)
# shared projections not saved (save_projection=False): the converter regenerates them from projection_prng_key as PEFT does at load
# (row S69-ADAPTERS-W4-CONVERTER-RESIDUAL-VARIANTS-2026-10-27-01 items 1, 2)
VERA_NOPROJ = dict(VERA_R8, save_projection=False, projection_prng_key=7)
RANDLORA_NOPROJ = dict(RANDLORA_R4, save_projection=False, projection_prng_key=7)
PVERA_NOPROJ = dict(PVERA_R4, save_projection=False, projection_prng_key=7)
# UniLoRA indices regenerated over a module set the old explicit-list derivation could not rebuild
UNILORA_ALLLINEAR = dict(UNILORA_NOIDX, target_modules="all-linear")
UNILORA_LAYERS = dict(UNILORA_NOIDX, layers_to_transform=[0], layers_pattern="layers")
WAVEFT_DB1 = dict(peft_type="waveft", n_frequency=24, target_modules=TARGETS, _std=0.01)
WAVEFT_DB2 = dict(WAVEFT_DB1, wavelet_family="db2", n_frequency_pattern={"v_proj": 8})
WAVEFT_NOIDWT = dict(WAVEFT_DB1, use_idwt=False)
# IA3 (row S69-ADAPTERS-LINEAR-WRAPPER-IA3-RED-VEC-SCALE-2026-10-26-01): feedforward_modules scale the INPUT (base(x * l)), the rest
# scale the OUTPUT (base(x) * l). IA3_KVD = k/v output scales + down_proj input scale; IA3_ALL = every projection, o/down input scale.
IA3_KVD = dict(peft_type="ia3", target_modules=["k_proj", "v_proj", "down_proj"], feedforward_modules=["down_proj"], _std=0.3)
IA3_ALL = dict(peft_type="ia3", target_modules=TARGETS, feedforward_modules=["o_proj", "down_proj"], _std=0.3)
BDLORA_NB2 = dict(r=8, lora_alpha=16, target_modules=TARGETS, use_bdlora=dict(
    target_modules_bd_a=["o_proj", "down_proj"], target_modules_bd_b=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"], nblocks=2))
# refusals: (PEFT kwargs, strings the converter's output must contain)
REFUSE = {
    "refuse_miss_bat": (dict(peft_type="miss", r=4, init_weights="bat", target_modules=["q_proj"]),
                        ["MISS", "bat", "S69-ADAPTERS-DENSE-DELTA-KIND-2026-10-26-01"]),
    "refuse_pvera_sampled": (dict(peft_type="pvera", r=4, save_projection=True, sample_at_inference=True, target_modules=["q_proj"]),
                             ["PVERA", "sample_at_inference", ROW_RESIDUAL]),
    "refuse_c3a": (dict(peft_type="c3a", block_size=16, target_modules=["q_proj"]),
                   ["C3A", "S69-ADAPTERS-DENSE-DELTA-KIND-2026-10-26-01"]),
    "refuse_modules_to_save": (dict(r=4, lora_alpha=8, target_modules=["q_proj"], modules_to_save=["lm_head"]),
                               ["modules_to_save", "S69-ADAPTERS-DENSE-DELTA-KIND-2026-10-26-01"]),
    "refuse_lora_bias": (dict(r=4, lora_alpha=8, target_modules=["q_proj"], lora_bias=True),
                         ["lora_bias", "S69-ADAPTERS-LORA-BIAS-BIAS-DELTA-KIND-2026-10-26-01"]),
    "refuse_pissa_unconverted": (dict(r=4, lora_alpha=8, target_modules=["q_proj"], init_lora_weights="pissa"),
                                 ["pissa", "S69-ADAPTERS-CONVERTER-DISPATCHER-2026-10-26-01"]),
}
# GPT-OSS target_parameters (row S69-ADAPTERS-TARGET-PARAMETERS-MOE-EXPERT-LORA-2026-10-26-01)
GPTOSS_EXPERT_PARAMS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
GPTOSS_TP = dict(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], target_parameters=GPTOSS_EXPERT_PARAMS)
GPTOSS_TP_ONLY = dict(r=4, lora_alpha=16, use_rslora=True, target_modules=[], target_parameters=GPTOSS_EXPERT_PARAMS)

CASES = [
    # name, kind, expectation, PEFT kwargs, fork runtime scale
    ("base", "base", "pass", None, None),
    ("lora_r8_a16", "lora", "pass", BASE_LORA, 1.0),
    ("neg_wrong_scale", "lora", "fail", BASE_LORA, 0.5),
    ("neg_adapter_removed", "lora", "fail", BASE_LORA, None),
    ("rslora_r16_a32", "lora", "pass", dict(r=16, lora_alpha=32, use_rslora=True, target_modules=TARGETS), 1.0),
    ("alpha_pattern", "lora", "pass", dict(r=8, lora_alpha=16, alpha_pattern={"q_proj": 4, "down_proj": 64}, target_modules=TARGETS), 1.0),
    ("pattern_rslora", "lora", "pass",
     dict(r=8, lora_alpha=16, use_rslora=True, rank_pattern={"v_proj": 4}, alpha_pattern={"o_proj": 32}, target_modules=TARGETS), 1.0),
    ("threshold_converted", "lora", "pass",
     dict(r=1, lora_alpha=1, rank_pattern={"q_proj": 8, "v_proj": 8}, alpha_pattern={"q_proj": 8, "v_proj": 8}, target_modules=["q_proj", "v_proj"]), 1.0),
    ("perseq_two_adapters", "perseq", "pass", None, None),
    # one projection at a time: localizes a positive-control failure to the graph site that drops (or mis-applies) the branch
    *[(f"only_{t}", "lora", "pass", dict(r=8, lora_alpha=16, target_modules=[t]), 1.0) for t in TARGETS],
    # Embedding (token_embd) LoRA: loaded and converted but never applied on the live legacy graph (row
    # S69-ENGREV-PEFT-EMBED-LORA-NOT-APPLIED-LEGACY-2026-10-27-01). embed_tokens = untied llama; embed_tokens_gemma3 = tied
    # lm_head + sqrt(n_embd)-scaled embedding (delta before the scale, tied output must not use the embedding adapter).
    ("embed_tokens", "lora", "pass", dict(r=8, lora_alpha=16, target_modules=["embed_tokens"]), 1.0),
    ("embed_tokens_neg_adapter_removed", "lora", "fail", dict(r=8, lora_alpha=16, target_modules=["embed_tokens"]), None),
    ("gemma3_base", "gemma3_base", "pass", None, None),
    ("embed_tokens_gemma3", "gemma3_lora", "pass", dict(r=8, lora_alpha=16, target_modules=["embed_tokens"]), 1.0),
    ("embed_tokens_gemma3_neg_adapter_removed", "gemma3_lora", "fail", dict(r=8, lora_alpha=16, target_modules=["embed_tokens"]), None),
    # MLA (DeepSeek-V2 arch, graph src/graphs/build_deepseek2.cpp). The adapted projections were raw ggml_mul_mat at 737e1a8a2
    # (row S69-ADAPTERS-RAW-MATMUL-LORA-BYPASS-MLA-BITNET-2026-10-26-01): the adapter loaded and changed nothing there.
    ("mla_base", "mla_base", "pass", None, None),
    ("mla_lora_qa_kva_qb", "mla_lora", "pass", MLA_LORA, 1.0),
    ("mla_neg_adapter_removed", "mla_lora", "fail", MLA_LORA, None),
    # BitNet (src/graphs/build_bitnet.cpp): all seven projections were raw ggml_mul_mat at 737e1a8a2 (same row).
    ("bitnet_base", "bitnet_base", "pass", None, None),
    ("bitnet_lora_all", "bitnet_lora", "pass", BASE_LORA, 1.0),
    ("bitnet_neg_adapter_removed", "bitnet_lora", "fail", BASE_LORA, None),
    # DoRA (row S69-ADAPTERS-DORA-RUNTIME-KIND-2026-10-26-01). Before the DoRA kind the converter exits on lora_magnitude_vector.
    ("dora_r8_qv", "lora", "pass", DORA_QV, 1.0),
    ("dora_all", "lora", "pass", DORA_ALL, 1.0),
    ("dora_rslora_pattern", "lora", "pass", dict(DORA_ALL, use_rslora=True, alpha_pattern={"o_proj": 32}), 1.0),
    ("dora_neg_m_zeroed", "dora_variant:zero_m", "fail", DORA_QV, 1.0),
    ("dora_neg_as_lora", "dora_variant:drop_m", "fail", DORA_QV, 1.0),
    ("dora_neg_adapter_removed", "lora", "fail", DORA_QV, None),
    ("dora_q8_base", "q8base", "report", DORA_ALL, 1.0),
    ("perseq_two_dora", "perseq:dora_qv+dora_all4", "pass", None, None),
    ("perseq_dora_and_lora", "perseq:dora_qv+lora4", "pass", None, None),
    # MoE (Mixtral -> stacked experts, llm_build_lora_mm_id; row S69-ADAPTERS-W2-MULTI-DORA-BATCH-AND-MOE-2026-10-26-01).
    # moe_lora_experts is the plain-LoRA positive control of the expert path: at fdfaa679e the fused ggml_moe_up_gate took the
    # expert stacks only, so an adapter on w1/w3 (gate/up) was dropped whenever fused_moe_up_gate was on (the default).
    ("moe_base", "moe_base", "pass", None, None),
    ("moe_lora_experts", "moe_lora", "pass", MOE_LORA, 1.0),
    ("moe_dora_experts", "moe_lora", "pass", MOE_DORA, 1.0),
    ("moe_dora_neg_m_zeroed", "moe_dora_variant:zero_m", "fail", MOE_DORA, 1.0),
    ("moe_dora_neg_adapter_removed", "moe_lora", "fail", MOE_DORA, None),
    ("moe_perseq_two_dora", "moe_perseq:moe_dora8+moe_dora4", "pass", None, None),
    ("moe_perseq_dora_and_lora", "moe_perseq:moe_dora8+moe_lora4", "pass", None, None),
    # Lily (row S69-ADAPTERS-W2-LILY-PER-SEQUENCE-ROUTER-2026-10-26-01). Needs --peft-src (a PEFT with tuners/lily); before the Lily
    # kind the converter exits on the lily_* tensor names.
    ("lily_alone", "lora", "pass", LILY_ALL, 1.0),
    ("lily_qv_scaling2", "lora", "pass", LILY_QV, 1.0),
    ("lily_batch_two_seqs", "lily_batch", "pass", LILY_ALL, 1.0),
    ("lily_neg_vs_peft_batched", "lily_batch:peft_batched", "fail", LILY_ALL, 1.0),
    ("lily_neg_router_zeroed", "lily_variant:zero_router", "fail", LILY_ALL, 1.0),
    ("lily_neg_adapter_removed", "lora", "fail", LILY_ALL, None),
    ("lily_perseq_two", "perseq:lily_all+lily_qv", "pass", None, None),
    ("lily_perseq_lily_and_lora", "perseq:lily_all+lora4", "pass", None, None),
    # converter dispatcher (row S69-ADAPTERS-CONVERTER-DISPATCHER-2026-10-26-01): one case per dispatched peft_type / variant
    ("qalora_g16", "lora", "pass", QALORA_G16, 1.0),
    ("qalora_plain_dispatch", "lora", "pass", dict(QALORA_G16, _qalora_plain_dispatch=True), 1.0),
    ("adalora_r4", "lora", "pass", ADALORA_R4, 1.0),
    ("vera_r8", "lora", "pass", VERA_R8, 1.0),
    ("loha_r4", "lora", "pass", LOHA_R4, 1.0),
    ("lokr_r2_decomposed", "lora", "pass", LOKR_R2, 1.0),
    ("lokr_full_factors", "lora", "pass", LOKR_FULL, 1.0),
    ("fourierft_n24", "lora", "pass", FOURIERFT_N24, 1.0),
    # remaining low-rank tuners + LoRA use_bdlora (row S69-ADAPTERS-W3-CONVERTER-REMAINING-LOWRANK-TUNERS-2026-10-27-01)
    ("vblora_r4", "lora", "pass", VBLORA_R4, 1.0),
    ("vblora_topk_only", "lora", "pass", VBLORA_TOPK, 1.0),
    ("unilora_indices_saved", "lora", "pass", UNILORA_IDX, 1.0),
    ("unilora_indices_regenerated", "lora", "pass", UNILORA_NOIDX, 1.0),
    ("unilora_indices_all_linear", "lora", "pass", UNILORA_ALLLINEAR, 1.0),
    ("unilora_indices_layers_to_transform", "lora", "pass", UNILORA_LAYERS, 1.0),
    ("vera_projection_regenerated", "lora", "pass", VERA_NOPROJ, 1.0),
    ("randlora_projection_regenerated", "lora", "pass", RANDLORA_NOPROJ, 1.0),
    ("pvera_projection_regenerated", "lora", "pass", PVERA_NOPROJ, 1.0),
    ("tinylora_u8", "lora", "pass", TINYLORA_U8, 1.0),
    ("tinylora_tied_proj_regenerated", "lora", "pass", TINYLORA_TIED_NOPROJ, 1.0),
    ("delora_r4", "lora", "pass", DELORA_R4, 1.0),
    ("gralora_k2", "lora", "pass", GRALORA_K2, 1.0),
    ("gralora_hybrid", "lora", "pass", GRALORA_HYBRID, 1.0),
    ("psoft_r4", "lora", "pass", PSOFT_R4, 1.0),
    ("psoft_neumann", "lora", "pass", PSOFT_NEUMANN, 1.0),
    ("psoft_lowrank_pissa", "lora", "pass", PSOFT_LOWRANK_PISSA, 1.0),
    # saved-converted PiSSA / OLoRA (row S69-ENGREV-PEFT-PISSA-CONVERTED-HARNESS-CASE-2026-10-27-01): the converter must take the plain
    # rank-2r / alpha-2x adapter PEFT writes on conversion and match the in-memory PiSSA model (residual base + trained adapter)
    ("pissa_converted_r4", "lora", "pass", dict(r=4, lora_alpha=8, target_modules=TARGETS, init_lora_weights="pissa", _convert_init=True), 1.0),
    ("pissa_converted_rslora", "lora", "pass", dict(r=4, lora_alpha=8, use_rslora=True, target_modules=TARGETS, init_lora_weights="pissa", _convert_init=True), 1.0),
    ("olora_converted_r4", "lora", "pass", dict(r=4, lora_alpha=8, target_modules=TARGETS, init_lora_weights="olora", _convert_init=True), 1.0),
    # output.weight (lm_head) adapter on the untied llama model: plain, then with the output-vocab subset active (fast path), plus its negatives
    ("lm_head_lora", "lora", "pass", LM_HEAD_LORA, 1.0),
    ("lm_head_subset_lora", "subset_lora", "pass", LM_HEAD_LORA, 1.0),
    ("lm_head_subset_neg_adapter_removed", "subset_lora", "fail", LM_HEAD_LORA, None),
    ("lm_head_subset_neg_wrong_scale", "subset_lora", "fail", LM_HEAD_LORA, 0.5),
    ("paca_r4", "lora", "pass", PACA_R4, 1.0),
    ("miss_r4", "lora", "pass", MISS_R4, 1.0),
    ("miss_mini", "lora", "pass", MISS_MINI, 1.0),
    ("randlora_r4", "lora", "pass", RANDLORA_R4, 1.0),
    ("pvera_r4", "lora", "pass", PVERA_R4, 1.0),
    ("waveft_db1", "lora", "pass", WAVEFT_DB1, 1.0),
    ("waveft_db2_pattern", "lora", "pass", WAVEFT_DB2, 1.0),
    ("waveft_no_idwt", "lora", "pass", WAVEFT_NOIDWT, 1.0),
    ("bdlora_nb2", "lora", "pass", BDLORA_NB2, 1.0),
    ("ia3_kv_down", "lora", "pass", IA3_KVD, 1.0),
    ("ia3_all", "lora", "pass", IA3_ALL, 1.0),
    ("ia3_neg_wrong_scale", "lora", "fail", IA3_KVD, 0.5),
    ("ia3_neg_adapter_removed", "lora", "fail", IA3_KVD, None),
    ("psoft_neg_adapter_removed", "lora", "fail", PSOFT_R4, None),
    ("bdlora_neg_wrong_scale", "lora", "fail", BDLORA_NB2, 0.5),
    *[(n, "refuse", "pass", REFUSE[n][0], None) for n in REFUSE],
    # GPT-OSS fused experts via target_parameters (row S69-ADAPTERS-TARGET-PARAMETERS-MOE-EXPERT-LORA-2026-10-26-01)
    ("gptoss_base", "gptoss_base", "pass", None, None),
    ("gptoss_tp_experts", "gptoss_lora", "pass", GPTOSS_TP, 1.0),
    ("gptoss_tp_experts_only", "gptoss_lora", "pass", GPTOSS_TP_ONLY, 1.0),
    ("gptoss_tp_neg_adapter_removed", "gptoss_lora", "fail", GPTOSS_TP, None),
    # position-gated LoRA (row S69-ADAPTERS-POSITION-GATED-LORA-UNO-SPEC-2026-10-26-01): kind "gated:<gate>[:ungated]"; PEFT reference =
    # the adapter applied on exactly the gated rows. The negative controls: adapter removed, and the SAME adapter converted
    # WITHOUT the gate (adapter on every row) against the gated reference
    ("gated_draft_rows", "gated:draft_rows", "pass", BASE_LORA, 1.0),
    ("gated_draft_rows_neg_adapter_removed", "gated:draft_rows", "fail", BASE_LORA, None),
    ("gated_draft_rows_neg_ungated_adapter", "gated:draft_rows:ungated", "fail", BASE_LORA, 1.0),
    ("gated_mask_token", "gated:mask_token", "pass", BASE_LORA, 1.0),
    ("gated_mask_token_neg_adapter_removed", "gated:mask_token", "fail", BASE_LORA, None),
]


def run_sharded(a) -> int:
    """--jobs N: split the selected cases into N contiguous chunks (contiguous keeps cases that share a base model or an
    adapter in one child, so little is rebuilt), run one child process of this script per chunk with --cases/--work/--json,
    stream each child's log after it ends, and merge the per-case results. The merged run FAILS unless every selected case
    appears exactly once across the children and every child exited 0 or 1 (1 = some case unmet, still reported):
    a crashed or truncated child cannot read as a pass. Measured cost before this mode: 90 cases serial in 1120.7 s
    (gate b1109h, 2026-11-09); BENCH-PENDING for the sharded wall time."""
    wanted = set(a.cases.split(",")) if a.cases else None
    names = [c[0] for c in CASES if wanted is None or c[0] in wanted]
    if wanted and len(names) != len(wanted):
        log(f"unknown case name(s): {sorted(wanted - set(names))}")
        return 2
    work = a.work or tempfile.mkdtemp(prefix="adapter-peft-parity-")
    os.makedirs(work, exist_ok=True)
    n = max(1, min(a.jobs, len(names)))
    size = -(-len(names) // n)
    chunks = [names[i:i + size] for i in range(0, len(names), size)]
    t0 = time.time()
    procs = []
    for k, chunk in enumerate(chunks):
        sw = os.path.join(work, f"shard_{k}")
        os.makedirs(sw, exist_ok=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--bin", a.bin, "--cases", ",".join(chunk), "--work", sw,
               "--json", os.path.join(sw, "result.json"), "--seed", str(a.seed), "--lora-converter", a.lora_converter]
        if a.peft_src:
            cmd += ["--peft-src", a.peft_src]
        if a.cuda_arm:
            cmd.append("--cuda-arm")
        logf = open(os.path.join(sw, "shard.log"), "w", encoding="utf-8")
        # each child sizes OpenBLAS/OpenMP to ALL host threads by default; N children x 128-thread BLAS buffers exhausted the
        # commit limit (2026-11-09 A/B: "OpenBLAS error: Memory allocation still failed" in convert_hf_to_gguf). 4 threads per
        # child = the torch.set_num_threads(4) this script already uses; a caller's explicit setting wins.
        env = dict(os.environ)
        for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            env.setdefault(var, "4")
        procs.append((k, chunk, sw, logf, subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)))
    log(f"sharded run: {len(names)} case(s) in {len(chunks)} child process(es); work {work}")
    results, problems = [], []
    for k, chunk, sw, logf, p in procs:
        rc = p.wait()
        logf.close()
        with open(os.path.join(sw, "shard.log"), encoding="utf-8", errors="replace") as fh:
            for line in fh:
                print(f"[shard {k}] {line}", end="", flush=True)
        if rc not in (0, 1):
            problems.append(f"shard {k} exited {rc}")
        try:
            with open(os.path.join(sw, "result.json"), encoding="utf-8") as fh:
                results += json.load(fh)["results"]
        except (OSError, ValueError, KeyError) as e:
            problems.append(f"shard {k} wrote no readable result.json ({type(e).__name__}: {e})")
    seen = [r["case"] for r in results]
    missing = sorted(set(names) - set(seen))
    dup = sorted({c for c in seen if seen.count(c) > 1})
    if missing:
        problems.append(f"{len(missing)} case(s) produced no result: {missing}")
    if dup:
        problems.append(f"case(s) reported more than once: {dup}")
    n_bad = sum(1 for r in results if not r["met_expectation"]) + len(missing)
    summary = {"tol": TOL, "effect_min": EFFECT_MIN, "neg_margin": NEG_MARGIN, "lora_std": LORA_STD, "seq_lens": SEQ_LENS,
               "peft": peft.__version__, "transformers": transformers.__version__, "torch": torch.__version__,
               "jobs": len(chunks), "n_cases": len(names), "n_unmet": n_bad, "problems": problems, "results": results}
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
    for pr in problems:
        log(f"SHARD PROBLEM: {pr}")
    log(f"test-adapter-peft-parity: {len(names) - n_bad}/{len(names)} cases met their expectation "
        f"({time.time() - t0:.1f}s, {len(chunks)} shards)")
    return 0 if (n_bad == 0 and not problems) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="test-adapter-peft-parity executable")
    ap.add_argument("--work", help="work directory (default: a fresh temp dir, left in place)")
    ap.add_argument("--cases", help="comma-separated subset of case names")
    ap.add_argument("--json", help="write the per-case results here")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--peft-src", help="a PEFT source tree (its src/ dir) imported instead of the installed peft; the Lily cases "
                                       "need one with tuners/lily (consumed before argparse, see _peft_src_from_argv)")
    ap.add_argument("--lora-converter", default=os.path.join(REPO, "convert_lora_to_gguf.py"),
                    help="convert_lora_to_gguf.py to test (default: this repo's). A copy outside the repo, e.g. an older "
                         "revision for a fail-before run, imports conversion/ and gguf-py/ from this repo via PYTHONPATH")
    ap.add_argument("--jobs", type=int, default=1,
                    help="run the cases in N child processes of this script (contiguous chunks of the case list, each with "
                         "its own work dir), then merge their JSON results; every case runs exactly as with --jobs 1")
    ap.add_argument("--cuda-arm", action="store_true",
                    help="run the fork side with the library-default device op offload (PARITY_OP_OFFLOAD=1, needs a CUDA build) and judge "
                         "each expect=pass case against the device's own no-adapter drift: |fork-peft| <= CUDA_DRIFT_K x |fork_base-peft_base| + TOL "
                         "instead of TOL; negative controls keep NEG_MARGIN. Default (off) is the fp32 CPU arm.")
    a = ap.parse_args()
    if a.cuda_arm:
        os.environ["PARITY_OP_OFFLOAD"] = "1"  # inherited by the fork binary and by --jobs children
    if a.jobs > 1:
        return run_sharded(a)
    if os.path.dirname(os.path.abspath(a.lora_converter)) != os.path.abspath(REPO):
        os.environ["PYTHONPATH"] = os.pathsep.join([REPO, os.path.join(REPO, "gguf-py")] + [p for p in [os.environ.get("PYTHONPATH")] if p])
        os.environ["NO_LOCAL_GGUF"] = "1"
    if a.peft_src:
        # the converter subprocess imports the same PEFT as this process (WaveFT's wavelet filter taps are read from peft)
        os.environ["PYTHONPATH"] = os.pathsep.join([p for p in [os.environ.get("PYTHONPATH")] if p] + [os.path.abspath(a.peft_src)])

    torch.set_num_threads(4)
    work = a.work or tempfile.mkdtemp(prefix="adapter-peft-parity-")
    os.makedirs(work, exist_ok=True)
    py = sys.executable
    wanted = set(a.cases.split(",")) if a.cases else None
    cases = [c for c in CASES if wanted is None or c[0] in wanted]
    if wanted and len(cases) != len(wanted):
        log(f"unknown case name(s): {sorted(wanted - {c[0] for c in cases})}")
        return 2
    log(f"peft {peft.__version__} transformers {transformers.__version__} torch {torch.__version__}; work {work}; "
        f"lora converter {a.lora_converter}")

    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    seqs = [[int(t) for t in rng.integers(3, VOCAB, size=n)] for n in SEQ_LENS]
    bases: dict[str, tuple[str, str, list]] = {}  # family -> (HF dir, GGUF path, PEFT-free base logits)
    ref_models: dict[int, object] = {}  # id(reference logits list) -> the HF/PEFT model that produced it (fp64 arbiter)
    ref_cache64: dict[tuple[int, int], np.ndarray] = {}

    def ref_model(ref: list, model) -> list:
        ref_models[id(ref)] = model
        return ref

    def ref64(ref: list, i: int) -> np.ndarray:
        """fp64 logits of the model behind `ref` on seqs[i] (computed once, only when the arbiter needs it)."""
        k = (id(ref), i)
        if k not in ref_cache64:
            ref_cache64[k] = hf_logits64(ref_models[id(ref)], [seqs[i]])[0]
        return ref_cache64[k]

    def base_for(family: str) -> tuple[str, str, list]:
        if family not in bases:
            t = time.time()
            suffix = "" if family == "llama" else f"_{family}"
            base_dir = os.path.join(work, f"base_hf{suffix}")
            base_gguf = os.path.join(work, f"base{suffix}-f32.gguf")
            base = make_base(base_dir, a.seed, family)
            convert_base(py, base_dir, base_gguf, family)
            bases[family] = (base_dir, base_gguf, ref_model(hf_logits(base, seqs), base))
            log(f"{family} base model built + converted in {time.time() - t:.1f}s")
        return bases[family]

    adapters: dict[str, tuple[str, list]] = {}  # key -> (gguf path, PEFT logits)
    adapter_dirs: dict[str, str] = {}  # key -> saved PEFT adapter dir

    def adapter_key(kw: dict, seed: int, family: str) -> str:
        return json.dumps(kw, sort_keys=True) + f"#{seed}#{family}"

    def adapter_for(kw: dict, seed: int, family: str = "llama") -> tuple[str, list]:
        base_dir = base_for(family)[0]
        key = adapter_key(kw, seed, family)
        if key not in adapters:
            d = os.path.join(work, f"lora_{len(adapters)}")
            pm = make_adapter(base_dir, d, kw, seed)
            adapter_dirs[key] = d
            g = os.path.join(work, f"lora_{len(adapters)}-f32.gguf")
            run([py, a.lora_converter, d, "--base", base_dir, "--outtype", "f32", "--outfile", g])
            adapters[key] = (g, ref_model(hf_logits(pm, seqs), pm))
        return adapters[key]

    def variant_gguf(kw: dict, seed: int, mode: str, family: str = "llama") -> tuple[str, list]:
        """The converted GGUF of adapter_variant(<the adapter of kw>, mode), plus the TRUE (unmodified) PEFT reference."""
        _, ref = adapter_for(kw, seed, family)
        src = adapter_dirs[adapter_key(kw, seed, family)]
        d = src + f"_{mode}"
        g = d + "-f32.gguf"
        if not os.path.exists(g):
            adapter_variant(src, d, mode)
            run([py, a.lora_converter, d, "--base", base_for(family)[0], "--outtype", "f32", "--outfile", g])
        return g, ref

    def base_q8() -> str:
        g = os.path.join(work, "base-q8_0.gguf")
        if not os.path.exists(g):
            run([py, os.path.join(REPO, "convert_hf_to_gguf.py"), base_for("llama")[0], "--outtype", "q8_0", "--outfile", g])
        return g

    dev_drift: dict[str, float] = {}  # --cuda-arm: family -> max|fork_device_base - peft_base| (no adapter)
    results = []
    for name, kind, expect, kw, scale in cases:
        t1 = time.time()
        rec = {"case": name, "expect": expect}
        try:
            family = kind.split("_")[0] if kind.split("_")[0] in ("mla", "bitnet", "moe", "gptoss", "gemma3") else "llama"
            fkind = kind[len(family) + 1:] if family != "llama" else kind  # the kind with its family prefix removed
            _, base_gguf, ref_base = base_for(family)
            if fkind == "refuse":
                # the converter must exit non-zero AND name what it refused (peft_type / flag) and the row that would serve it
                d = os.path.join(work, f"refuse_{name}")
                make_adapter(base_for(family)[0], d, kw, a.seed + 7)
                p = subprocess.run([py, a.lora_converter, d, "--base", base_for(family)[0], "--outtype", "f32",
                                    "--outfile", d + "-f32.gguf"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace")
                must = REFUSE[name][1]
                missing = [s for s in must if s not in p.stdout]
                rec.update(converter_rc=p.returncode, must_name=must, missing=missing,
                           converter_tail="\n".join(p.stdout.splitlines()[-6:])[-1500:])
                outcome = "PASS" if (p.returncode != 0 and not missing) else "FAIL"
                rec["outcome"] = outcome
                rec["met_expectation"] = (outcome == "PASS") == (expect == "pass")
                rec["seconds"] = round(time.time() - t1, 2)
                results.append(rec)
                log(f"{name:22s} expect={expect:4s} outcome={outcome:5s} converter rc={p.returncode} missing={missing} "
                    f"met={rec['met_expectation']}  tail: {rec['converter_tail'].splitlines()[-1] if rec['converter_tail'] else ''}")
                continue
            if fkind == "base":
                fk = fork_logits(a.bin, base_gguf, seqs, work, name)
                diff, effect = max_abs(fk, ref_base), None
                pairs = [(fk[i], ref_base, i) for i in range(len(seqs))]
            elif fkind.startswith("subset_lora"):
                # output-vocab-subset fast path (build_output_mm) + an output.weight (lm_head) adapter on an UNTIED model (row
                # S69-ENGREV-PEFT-LORA-FASTPATH-GUARD-LEGACY-MAP-ONLY-2026-10-27-01): the subset columns must equal the PEFT logits of the
                # same ids; before the llm_lora_targets guard the adapter was skipped and the columns equalled the base model
                stride = 3
                g, ref = adapter_for(kw, a.seed + 7, family)
                loras = [] if scale is None else [(g, scale)]
                fk = fork_logits(a.bin, base_gguf, seqs, work, name, loras=loras, vocab_stride=stride)
                ids = np.arange(0, VOCAB, stride)
                ref = ref_sub = [r[:, ids] for r in ref]
                diff = max_abs(fk, ref_sub)
                effect = max_abs(ref_sub, [r[:, ids] for r in ref_base])
                pairs = []  # a near miss on the subset columns has no fp64 reference here: it is an ERROR, never a pass
            elif fkind == "lora":
                g, ref = adapter_for(kw, a.seed + 7, family)
                loras = [] if scale is None else [(g, scale)]
                fk = fork_logits(a.bin, base_gguf, seqs, work, name, loras=loras)
                diff, effect = max_abs(fk, ref), max_abs(ref, ref_base)
                pairs = [(fk[i], ref, i) for i in range(len(seqs))]
            elif fkind == "lily_batch" or fkind.startswith("lily_batch:"):
                # two equal-length sequences (no padding, so PEFT's batched forward is well defined) in ONE fork --batch decode
                g, ref = adapter_for(kw, a.seed + 7, family)
                pm = ref_models[id(ref)]
                x_seq, y_seq = seqs[2], list(reversed(seqs[2]))
                fk = fork_logits(a.bin, base_gguf, [x_seq, y_seq], work, name, loras=[(g, scale)], batch=True)
                if fkind == "lily_batch":
                    ref_xy = [ref[2], hf_logits(pm, [y_seq])[0]]  # PEFT at batch 1, each sequence alone
                else:
                    with torch.no_grad():  # PEFT with both sequences in one batch: its router mean spans both
                        lb = pm(torch.tensor([x_seq, y_seq], dtype=torch.long)).logits.float().numpy()
                    ref_xy = [lb[0], lb[1]]
                    rec["peft_batch_dependence"] = max_abs(ref_xy, [ref[2], hf_logits(pm, [y_seq])[0]])
                diff, effect = max_abs(fk, ref_xy), max_abs([ref[2]], [ref_base[2]])
                pairs = []  # the fp64 arbiter needs a seqs[] index; y_seq is not one, so a near miss here is an ERROR, never a pass
            elif fkind.startswith(("dora_variant:", "lily_variant:")):
                g, ref = variant_gguf(kw, a.seed + 7, fkind.split(":", 1)[1], family)
                fk = fork_logits(a.bin, base_gguf, seqs, work, name, loras=[(g, scale)])
                diff, effect = max_abs(fk, ref), max_abs(ref, ref_base)
                pairs = [(fk[i], ref, i) for i in range(len(seqs))]
            elif fkind.startswith("gated:"):
                parts = fkind.split(":")
                gate, ungated = parts[1], len(parts) > 2 and parts[2] == "ungated"
                _, ref_plain = adapter_for(kw, a.seed + 7, family)
                pm = ref_models[id(ref_plain)]
                starts = [len(s) // 2 for s in seqs]
                gtok = seqs[2][3]
                ref_g = hf_logits_gated(pm, seqs, gate, starts, gtok)
                gg = os.path.join(work, f"gated_{gate}-f32.gguf")
                if not ungated and not os.path.exists(gg):
                    cargs = ["--gate", gate] + (["--gate-token", str(gtok)] if gate == "mask_token" else [])
                    run([py, a.lora_converter, adapter_dirs[adapter_key(kw, a.seed + 7, family)], "--base", base_for(family)[0],
                         "--outtype", "f32", "--outfile", gg] + cargs)
                g = adapter_for(kw, a.seed + 7, family)[0] if ungated else gg
                loras = [] if scale is None else [(g, scale)]
                ds = [(i, starts[i]) for i in range(len(seqs))] if gate == "draft_rows" else []
                fk = fork_logits(a.bin, base_gguf, seqs, work, name, loras=loras, batch=True, draft_starts=ds)
                diff, effect = max_abs(fk, ref_g), max_abs(ref_g, ref_base)
                pairs = []
            elif fkind == "q8base":
                g, ref = adapter_for(kw, a.seed + 7)
                bq = base_q8()
                fk = fork_logits(a.bin, bq, seqs, work, name, loras=[(g, scale)])
                fkb = fork_logits(a.bin, bq, seqs, work, name + "_nolora")
                diff, effect = max_abs(fk, ref), max_abs(ref, ref_base)
                rec["q8_base_alone_vs_peft_base"] = max_abs(fkb, ref_base)  # the quantisation error with no adapter at all
                pairs = []
            elif fkind == "perseq" or fkind.startswith("perseq:"):
                # two sequences, each with its own adapter, in ONE batch; each must equal its adapter run alone in PEFT
                na, nb = ("lora8", "lora4") if fkind == "perseq" else fkind.split(":", 1)[1].split("+")
                ga, ra = adapter_for(PERSEQ_ADAPTERS[na][0], a.seed + PERSEQ_ADAPTERS[na][1], family)
                gb, rb = adapter_for(PERSEQ_ADAPTERS[nb][0], a.seed + PERSEQ_ADAPTERS[nb][1], family)
                two = [seqs[2], seqs[1]]
                fk = fork_logits(a.bin, base_gguf, two, work, name, seq_loras=[(0, ga, 1.0), (1, gb, 1.0)], batch=True)
                ref = [ra[2], rb[1]]
                diff = max_abs(fk, ref)
                effect = min(max_abs([ra[2]], [ref_base[2]]), max_abs([rb[1]], [ref_base[1]]))
                pairs = [(fk[0], ra, 2), (fk[1], rb, 1)]
            else:
                raise ValueError(kind)
            rec.update(max_abs_diff=diff, adapter_effect=effect)
            if expect == "pass":
                ok_effect = effect is None or effect > EFFECT_MIN
                ok_diff = diff < TOL
                if a.cuda_arm:
                    # device arm: the device matmul is not fp32-exact, so the bar is the device's own drift on the NO-adapter base
                    if family not in dev_drift:
                        dev_drift[family] = max_abs(fork_logits(a.bin, base_gguf, seqs, work, f"devbase_{family}"), ref_base)
                    bound = CUDA_DRIFT_K * dev_drift[family] + TOL
                    ok_diff = diff <= bound
                    rec.update(cuda_arm=True, device_base_drift=dev_drift[family], cuda_bound=bound,
                               diff_over_base_drift=(diff / dev_drift[family]) if dev_drift[family] > 0 else None)
                    # the arm only widens the CPU bar (bound >= TOL), and the fp64 arbiter below still applies to a near miss:
                    # measured on the CPU binary, pattern_rslora is 1.475e-04 vs PEFT fp32 and passes only through it
                if not ok_diff and diff < 10 * TOL:
                    # fp64 arbiter (see the module docstring): only for a near miss, never for a negative control
                    d64 = max(max_abs([f], [ref64(r, i)]) for f, r, i in pairs)
                    floor = max(max_abs([r[i]], [ref64(r, i)]) for f, r, i in pairs)
                    ok_diff = d64 < ARB_K * TOL and d64 <= ARB_K * floor
                    rec.update(max_abs_diff_fp64=d64, peft_fp32_floor=floor, arbiter="fp64 PASS" if ok_diff else "fp64 FAIL")
                outcome = "PASS" if (ok_diff and ok_effect) else "FAIL"
                if diff < TOL and not ok_effect:
                    rec["note"] = f"adapter effect {effect:.3e} <= {EFFECT_MIN}: comparison is vacuous"
            elif expect == "report":
                outcome = "REPORT"  # measured and recorded, never gated (quantised-base arm, harness design 5b)
            else:
                outcome = "FAIL" if diff > NEG_MARGIN * TOL else "PASS"
            rec["outcome"] = outcome
            rec["met_expectation"] = True if expect == "report" else (outcome == "PASS") == (expect == "pass")
        except Exception as e:  # FAILURES-SURFACED: recorded per case as outcome ERROR, counted in the summary and the exit code
            rec.update(outcome="ERROR", error=str(e)[-2000:], met_expectation=False)
        rec["seconds"] = round(time.time() - t1, 2)
        results.append(rec)
        eff = rec.get("adapter_effect")
        log(f"{name:22s} expect={expect:4s} outcome={rec['outcome']:5s} max|fork-peft|={rec.get('max_abs_diff', float('nan')):.3e} "
            f"effect={'n/a' if eff is None else f'{eff:.3e}'} met={rec['met_expectation']}"
            + (f"  [{rec['arbiter']}: |fork-peft64| {rec['max_abs_diff_fp64']:.3e} <= {ARB_K:g} x |peft32-peft64| "
               f"{ARB_K * rec['peft_fp32_floor']:.3e} (= {ARB_K:g} x {rec['peft_fp32_floor']:.3e})?]"
               if "arbiter" in rec else "")
            + (f"  [cuda-arm: bound {rec['cuda_bound']:.3e} = {CUDA_DRIFT_K:g} x device base drift {rec['device_base_drift']:.3e} + TOL]"
               if "cuda_bound" in rec else "")
            # the tail, not the head: the head is the (long) command line, the real error is last (the MLA cases lost it to [:300])
            + (f"  ERROR: {rec['error'][-600:]}" if "error" in rec else ""))

    n_bad = sum(1 for r in results if not r["met_expectation"])
    summary = {"tol": TOL, "effect_min": EFFECT_MIN, "neg_margin": NEG_MARGIN, "lora_std": LORA_STD, "seq_lens": SEQ_LENS,
               "peft": peft.__version__, "transformers": transformers.__version__, "torch": torch.__version__,
               "n_cases": len(results), "n_unmet": n_bad, "results": results}
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
    log(f"test-adapter-peft-parity: {len(results) - n_bad}/{len(results)} cases met their expectation ({time.time() - t0:.1f}s)")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--convert-spm-vocab":
        sys.exit(_convert_spm_vocab_main(sys.argv[2:]))
    sys.exit(main())
