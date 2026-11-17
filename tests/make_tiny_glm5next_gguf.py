#!/usr/bin/env python3
# Tiny random-weight `glm5-next` (GLM-5.3-Flash layout) GGUF for the two-implementation parity test
# (tests/test-glm5next-two-impl-parity.py).
#
#   python make_tiny_glm5next_gguf.py <tree-root> <out.gguf> [seed]
#
# The SAME file is loaded by two code paths (llama-model-loader.h get_arch(): fork GLM5NEXT, or upstream GLM5_NEXT when
# LLAMA_GLM5NEXT_IMPL=upstream), so it must carry every key either loader reads and every tensor either one creates.
# Shapes (ggml ne, innermost first; numpy shape = reversed):
#   n_embd 64, 2 heads, vocab 64, 5 blocks = 4 trunk + 1 NextN (nextn_predict_layers 1)
#   layer 0 MLA + dense FFN, layer 1 KDA + MoE, layer 2 MLA + MoE, layer 3 KDA + MoE, layer 4 = NextN block (MLA + MoE + eh_proj/enorm/hnorm)
#   head_count_kv per layer = [1,0,1,0,1]: 0 marks a KDA (linear attention) layer, 1 the single latent KV head of an MLA layer
#   MLA is nope-only (rope.dimension_count 0): q head 24 wide, v head 16, kv_lora_rank 32, q_lora_rank 32; the cache row (key_length/value_length) is the
#   32-wide latent. KDA head dim 16 (d_inner 32), conv kernel 4, bounded gate (lower bound -5).
#   mHC: 4 streams, 2 Sinkhorn iterations. k-pool DSA indexer: 4 heads x 16, kpool 4, top_k 8 (2 of n_pool pools picked once the prompt holds > 8 tokens).
#   MoE: 4 experts, 2 used, 1 shared, sigmoid gating, expert ff 32.
# Weights are random but seeded, so the file is deterministic.
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
out = Path(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 2468
sys.path.insert(0, str(root / "gguf-py"))
import gguf  # noqa: E402

ARCH = "glm5-next"
V, E = 64, 64
NH = 2                      # attention.head_count: KDA heads AND MLA heads
KV_PER_LAYER = [1, 0, 1, 0, 1]
NL = len(KV_PER_LAYER)      # block_count, NextN block included
NEXTN = 1
N_TRUNK = NL - NEXTN
KDA_D = 16
D_INNER = KDA_D * NH
D_CONV = 4
QR, KVR = 32, 32
K_MLA, V_MLA = 24, 16       # nope-only: the whole q head is the NoPE part
HC = 4
HC_MIX = (2 + HC) * HC
IDX_NH, IDX_D, KPOOL = 4, 16, 4
TOP_K = int(sys.argv[4]) if len(sys.argv) > 4 else 8   # optional 4th arg: indexer top_k (multiple of kpool); 64 = no selection (dense) for a prompt < 64 tokens
NEXP, NUSED, NSHARED, FFE = 4, 2, 1, 32
FF = 64

rng = np.random.default_rng(seed)


def rnd(*ne, scale=0.1):
    # ne is the ggml shape (innermost first); numpy wants the reverse
    return (rng.standard_normal(tuple(reversed(ne))) * scale).astype(np.float32)


def ones(*ne):
    return (1.0 + rnd(*ne, scale=0.05)).astype(np.float32)


w = gguf.GGUFWriter(str(out), ARCH)
w.add_name("tiny-random-glm5next")
w.add_uint32(f"{ARCH}.context_length", 256)
w.add_uint32(f"{ARCH}.embedding_length", E)
w.add_uint32(f"{ARCH}.block_count", NL)
w.add_uint32(f"{ARCH}.feed_forward_length", FF)
w.add_uint32(f"{ARCH}.leading_dense_block_count", 1)
w.add_uint32(f"{ARCH}.attention.head_count", NH)
w.add_array(f"{ARCH}.attention.head_count_kv", KV_PER_LAYER)
w.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", 1e-5)
w.add_uint32(f"{ARCH}.attention.key_length", KVR)             # cache row = latent width (converter: kv_lora_rank + qk_rope_head_dim)
w.add_uint32(f"{ARCH}.attention.value_length", KVR)
w.add_uint32(f"{ARCH}.attention.key_length_mla", K_MLA)
w.add_uint32(f"{ARCH}.attention.value_length_mla", V_MLA)
w.add_uint32(f"{ARCH}.attention.q_lora_rank", QR)
w.add_uint32(f"{ARCH}.attention.kv_lora_rank", KVR)
w.add_uint32(f"{ARCH}.rope.dimension_count", 0)
w.add_uint32(f"{ARCH}.expert_count", NEXP)
w.add_uint32(f"{ARCH}.expert_used_count", NUSED)
w.add_uint32(f"{ARCH}.expert_shared_count", NSHARED)
w.add_uint32(f"{ARCH}.expert_feed_forward_length", FFE)
w.add_float32(f"{ARCH}.expert_weights_scale", 1.0)
w.add_bool(f"{ARCH}.expert_weights_norm", True)
w.add_uint32(f"{ARCH}.expert_gating_func", 2)                  # sigmoid
w.add_uint32(f"{ARCH}.nextn_predict_layers", NEXTN)
w.add_uint32(f"{ARCH}.attention.indexer.head_count", IDX_NH)
w.add_uint32(f"{ARCH}.attention.indexer.key_length", IDX_D)
w.add_uint32(f"{ARCH}.attention.indexer.top_k", TOP_K)
w.add_uint32(f"{ARCH}.attention.indexer.kpool", KPOOL)
w.add_bool(f"{ARCH}.attention.indexer.kpool_select_tail", True)
w.add_uint32(f"{ARCH}.hyper_connection.count", HC)
w.add_uint32(f"{ARCH}.hyper_connection.sinkhorn_iterations", 2)
w.add_float32(f"{ARCH}.hyper_connection.epsilon", 1e-6)
w.add_uint32(f"{ARCH}.ssm.conv_kernel", D_CONV)
w.add_uint32(f"{ARCH}.kda.head_dim", KDA_D)
w.add_float32(f"{ARCH}.kda.gate_lower_bound", -5.0)

toks = ["<pad>", "<s>", "</s>", "Ċ"] + ["t%d" % i for i in range(4, V - 1)] + ["t4t5"]
types = [gguf.TokenType.CONTROL] * 3 + [gguf.TokenType.NORMAL] * (V - 3)
w.add_tokenizer_model("gpt2")
w.add_tokenizer_pre("default")
w.add_token_list(toks)
w.add_token_types(types)
w.add_token_merges(["t4 t5"])
w.add_bos_token_id(1)
w.add_eos_token_id(2)
w.add_pad_token_id(0)

w.add_tensor("token_embd.weight", rnd(E, V))
w.add_tensor("output_norm.weight", ones(E))
w.add_tensor("output.weight", rnd(E, V))

for i in range(NL):
    p = f"blk.{i}."
    nextn = i >= N_TRUNK
    w.add_tensor(p + "attn_norm.weight", ones(E))
    w.add_tensor(p + "ffn_norm.weight", ones(E))
    if not nextn:
        # mHC wraps every trunk block; the NextN block has none
        for blk in ("attn", "ffn"):
            w.add_tensor(p + f"hc_{blk}_fn.weight", rnd(HC * E, HC_MIX, scale=0.05))
            w.add_tensor(p + f"hc_{blk}_base.weight", rnd(HC_MIX, scale=0.1))
            w.add_tensor(p + f"hc_{blk}_scale.weight", (0.5 + np.abs(rnd(3, scale=0.1))).astype(np.float32))
    if KV_PER_LAYER[i] == 0:      # KDA
        for nm in ("attn_q", "attn_k", "attn_v"):
            w.add_tensor(p + nm + ".weight", rnd(E, D_INNER))
        for nm in ("q", "k", "v"):
            w.add_tensor(p + f"ssm_conv1d_{nm}.weight", rnd(D_CONV, 1, D_INNER, scale=0.3))
        w.add_tensor(p + "ssm_f_a.weight", rnd(E, KDA_D))
        w.add_tensor(p + "ssm_f_b.weight", rnd(KDA_D, D_INNER))
        w.add_tensor(p + "ssm_g_a.weight", rnd(E, KDA_D))
        w.add_tensor(p + "ssm_g_b.weight", rnd(KDA_D, D_INNER))
        w.add_tensor(p + "ssm_beta.weight", rnd(E, NH))
        w.add_tensor(p + "ssm_a", (-(0.5 + np.abs(rnd(NH, scale=1.0)))).astype(np.float32))   # -exp(A_log)
        w.add_tensor(p + "ssm_dt.bias", rnd(D_INNER, scale=0.5))
        w.add_tensor(p + "ssm_norm.weight", ones(KDA_D))
        w.add_tensor(p + "attn_output.weight", rnd(D_INNER, E))
    else:                          # absorbed nope-only MLA + k-pool DSA indexer
        w.add_tensor(p + "attn_q_a.weight", rnd(E, QR))
        w.add_tensor(p + "attn_q_a_norm.weight", ones(QR))
        w.add_tensor(p + "attn_q_b.weight", rnd(QR, NH * K_MLA))
        w.add_tensor(p + "attn_kv_a_mqa.weight", rnd(E, KVR))
        w.add_tensor(p + "attn_kv_a_norm.weight", ones(KVR))
        w.add_tensor(p + "attn_k_b.weight", rnd(K_MLA, KVR, NH))
        w.add_tensor(p + "attn_v_b.weight", rnd(KVR, V_MLA, NH))
        w.add_tensor(p + "attn_output.weight", rnd(NH * V_MLA, E))
        w.add_tensor(p + "indexer.k_norm.weight", ones(IDX_D))
        w.add_tensor(p + "indexer.k_norm.bias", rnd(IDX_D, scale=0.05))
        w.add_tensor(p + "indexer.proj.weight", rnd(E, IDX_NH, scale=0.5))
        w.add_tensor(p + "indexer.attn_k.weight", rnd(E, IDX_D, scale=0.5))
        w.add_tensor(p + "indexer.attn_q_b.weight", rnd(QR, IDX_NH * IDX_D, scale=0.5))
        w.add_tensor(p + "indexer_compressor_gate.weight", rnd(E, IDX_D))
        w.add_tensor(p + "indexer_compressor_ape.weight", rnd(IDX_D, KPOOL))
    if i < 1:                      # leading dense block
        w.add_tensor(p + "ffn_gate.weight", rnd(E, FF))
        w.add_tensor(p + "ffn_down.weight", rnd(FF, E))
        w.add_tensor(p + "ffn_up.weight", rnd(E, FF))
    else:
        w.add_tensor(p + "ffn_gate_inp.weight", rnd(E, NEXP, scale=0.5))
        w.add_tensor(p + "exp_probs_b.bias", rnd(NEXP, scale=0.1))
        w.add_tensor(p + "ffn_gate_exps.weight", rnd(E, FFE, NEXP))
        w.add_tensor(p + "ffn_down_exps.weight", rnd(FFE, E, NEXP))
        w.add_tensor(p + "ffn_up_exps.weight", rnd(E, FFE, NEXP))
        w.add_tensor(p + "ffn_gate_shexp.weight", rnd(E, FFE * NSHARED))
        w.add_tensor(p + "ffn_down_shexp.weight", rnd(FFE * NSHARED, E))
        w.add_tensor(p + "ffn_up_shexp.weight", rnd(E, FFE * NSHARED))
    if nextn:
        w.add_tensor(p + "nextn.eh_proj.weight", rnd(2 * E, E))
        w.add_tensor(p + "nextn.enorm.weight", ones(E))
        w.add_tensor(p + "nextn.hnorm.weight", ones(E))
        w.add_tensor(p + "nextn.shared_head_norm.weight", ones(E))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out, out.stat().st_size, "bytes")
