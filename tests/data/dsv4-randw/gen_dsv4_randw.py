# Generate a tiny RANDOM-WEIGHT deepseek4 GGUF that carries real tensor entries. test-llama-archs cannot drive deepseek4/deepseek41: their loader infers every shape from the
# FILE's tensor metadata, and a weightless model has none. This file has 3 layers: 0 = raw sliding window, 1 = CSA (compress ratio 4, overlapped pooling, with indexer),
# 2 = HCA (ratio 128), hyper-connection width 2, grouped output projection, 4 routed experts + 1 shared. Needs gguf-py (pip install gguf) and numpy.
# Shapes follow src/llama-load-tensors.cpp create_deepseek4_tensors and src/graphs/build_deepseek4.cpp; run_differential.bat runs it CPU-only vs CUDA (set GGML_CUDA_NO_TF32=1,
# otherwise TF32 noise is amplified by the large random weights). argv: <out.gguf> [seed] [weight scale, default 0.05; 0.4 gives non-uniform logits].
# GATE-CLAIM-OK: test-asset generator, no model claims.
import sys
import numpy as np
from gguf import GGUFWriter, GGUFValueType

out = sys.argv[1]
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)

N_EMBD, N_HEAD, HEAD, ROPE = 128, 4, 64, 32
N_LAYER, HC = 3, 2
Q_LORA, KV_LORA = 64, 64
N_GROUPS, O_LORA = 4, 32
N_EXP, N_USED, N_SHARED, FF_EXP = 4, 2, 1, 64
VOCAB = 259  # 3 specials + 256 byte pieces (SPM byte fallback), so llama-perplexity can tokenize text

V41 = len(sys.argv) > 4 and sys.argv[4] == "41"  # 4th argument "41": DeepSeek-V4.1 layout (shared compressed stream + engram) instead of V4
ARCH = "deepseek41" if V41 else "deepseek4"
w = GGUFWriter(out, ARCH)
A = ARCH + "."

w.add_uint32(A + "context_length", 512)
w.add_uint32(A + "embedding_length", N_EMBD)
w.add_uint32(A + "block_count", N_LAYER)
w.add_uint32(A + "feed_forward_length", 128)
w.add_uint32(A + "attention.head_count", N_HEAD)
w.add_uint32(A + "attention.head_count_kv", 1)
w.add_uint32(A + "attention.key_length", HEAD)
w.add_uint32(A + "attention.value_length", HEAD)
w.add_uint32(A + "rope.dimension_count", ROPE)
w.add_float32(A + "rope.freq_base", 10000.0)
w.add_float32(A + "attention.layer_norm_rms_epsilon", 1e-6)
w.add_uint32(A + "attention.q_lora_rank", Q_LORA)
w.add_uint32(A + "attention.kv_lora_rank", KV_LORA)
w.add_uint32(A + "attention.sliding_window", 32)
w.add_array(A + "attention.sliding_window_pattern", [True] * N_LAYER)
w.add_uint32(A + "attention.indexer.head_count", 4)
w.add_uint32(A + "attention.indexer.key_length", 64)
w.add_uint32(A + "attention.indexer.top_k", 8)
w.add_uint32(A + "attention.output_group_count", N_GROUPS)
w.add_uint32(A + "attention.output_lora_rank", O_LORA)
RATIOS = [0, 4, 128]  # layer 0 raw sliding window, layer 1 CSA (indexed, overlapped pooling), layer 2 HCA (disjoint pooling)
if V41:
    RATIOS = [0, 4, 128]  # two distinct ratios (the loader needs a CSA and an HCA segment); each of layers 1 and 2 is its own stream source: compressor + index key + index query
w.add_array(A + "attention.compress_ratios", RATIOS)
if V41:
    # engram: n-gram hashed embedding rows added into every hyper-connection copy of the layers in layer_ids (llama.cpp llama_set_engram_rows computes
    # row = (xor of token_map[t_s]*mult[s]) % prime[b] + offset[b]; the table needs max(offset + prime) rows)
    ENG_HEADS, ENG_KEY, ENG_NGRAM, ENG_LAYERS = 2, 16, 3, [1]
    ENG_COLS = (ENG_NGRAM - 1) * ENG_HEADS
    ENG_PRIMES, ENG_OFFSETS = [11, 13, 17, 19], [0, 11, 24, 41]
    ENG_ROWS = 60
    assert len(ENG_PRIMES) == ENG_COLS * len(ENG_LAYERS) and max(o + p for o, p in zip(ENG_OFFSETS, ENG_PRIMES)) <= ENG_ROWS
    w.add_uint32(A + "engram.head_count", ENG_HEADS)
    w.add_uint32(A + "engram.key_length", ENG_KEY)
    w.add_uint32(A + "engram.max_ngram_size", ENG_NGRAM)
    w.add_uint32(A + "engram.pad_id", 0)
    w.add_array(A + "engram.layer_ids", ENG_LAYERS)
    # multipliers / primes / offsets are read as uint64 (model.engram_*), token_map as uint32 (llama-model.h:1039-1042)
    for key, vals in (("multipliers", [1, 3, 5]), ("primes", ENG_PRIMES), ("offsets", ENG_OFFSETS)):
        w.add_key_value(A + "engram." + key, vals, GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT64)
    w.add_key_value(A + "engram.token_map", list(range(259)), GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT32)
w.add_float32(A + "attention.compress_rope_freq_base", 160000.0)
w.add_uint32(A + "hyper_connection.count", HC)
w.add_uint32(A + "hyper_connection.sinkhorn_iterations", 2)
w.add_float32(A + "hyper_connection.epsilon", 1e-6)
w.add_uint32(A + "hash_layer_count", 0)
w.add_float32(A + "swiglu_clamp_exp", 10.0)
w.add_bool(A + "expert_weights_norm", True)
w.add_uint32(A + "expert_feed_forward_length", FF_EXP)
w.add_uint32(A + "expert_count", N_EXP)
w.add_uint32(A + "expert_used_count", N_USED)
w.add_uint32(A + "expert_shared_count", N_SHARED)
w.add_float32(A + "expert_weights_scale", 1.0)
w.add_uint32(A + "expert_gating_func", 4)
w.add_uint32(A + "leading_dense_block_count", 0)

# byte-level SPM tokenizer
toks, scores, types = ["<unk>", "<s>", "</s>"], [0.0, 0.0, 0.0], [2, 3, 3]
for b in range(256):
    toks.append("<0x%02X>" % b)
    scores.append(0.0)
    types.append(6)
w.add_tokenizer_model("llama")
w.add_token_list(toks)
w.add_token_scores(scores)
w.add_token_types(types)
w.add_bos_token_id(1)
w.add_eos_token_id(2)
w.add_unk_token_id(0)
w.add_add_bos_token(True)
assert len(toks) == VOCAB

n = 0


SCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05


def t(name, *ne, scale=None, kind="rand"):
    """ne is ggml order (ne[0] fastest); numpy shape is the reverse."""
    global n
    shape = tuple(reversed(ne))
    if kind == "ones":
        arr = np.ones(shape, dtype=np.float32)
    elif kind == "zeros":
        arr = np.zeros(shape, dtype=np.float32)
    else:
        arr = (rng.standard_normal(shape) * (SCALE if scale is None else scale)).astype(np.float32)
    w.add_tensor(name, arr)
    n += 1


t("token_embd.weight", N_EMBD, VOCAB)
t("output_norm.weight", N_EMBD, kind="ones")
t("output.weight", N_EMBD, VOCAB)
t("hc_head_fn.weight", N_EMBD * HC, HC)
t("hc_head_scale.weight", 1, kind="ones")
t("hc_head_base.weight", HC, kind="zeros")

NTOT = HC * HC + 2 * HC
for i in range(N_LAYER):
    p = "blk.%d." % i
    t(p + "attn_norm.weight", N_EMBD, kind="ones")
    t(p + "attn_sinks.weight", N_HEAD, kind="zeros")
    t(p + "attn_q_a.weight", N_EMBD, Q_LORA)
    t(p + "attn_q_a_norm.weight", Q_LORA, kind="ones")
    t(p + "attn_q_b.weight", Q_LORA, N_HEAD * HEAD)
    t(p + "attn_kv.weight", N_EMBD, KV_LORA)
    t(p + "attn_kv_a_norm.weight", KV_LORA, kind="ones")
    t(p + "attn_output_a.weight", N_HEAD * HEAD // N_GROUPS, N_GROUPS * O_LORA)
    t(p + "attn_output_b.weight", N_GROUPS * O_LORA, N_EMBD)
    r = RATIOS[i]
    if V41:
        # V4.1 names (attn_compressor_*), no ape, no overlap: every stream pools `ratio` tokens into one row of width HEAD. Layer 1 owns the compressor, the index key
        # (indexer.attn_k + k_norm, fed from the pooled latent) and the index query (indexer.attn_q_b + proj). Both compressed layers own all three.
        IDX, NIH = 64, 4
        if r:
            t(p + "attn_compressor_kv.weight", N_EMBD, HEAD)
            t(p + "attn_compressor_gate.weight", N_EMBD, HEAD)
            t(p + "attn_compressor_norm.weight", HEAD, kind="ones")
            t(p + "indexer.attn_k.weight", HEAD, IDX)
            t(p + "indexer.k_norm.weight", IDX, kind="ones")
            t(p + "indexer.attn_q_b.weight", Q_LORA, NIH * IDX)
            t(p + "indexer.proj.weight", N_EMBD, NIH)
        if i == 1:
            t(p + "engram_embd.weight", ENG_KEY, ENG_ROWS)
            # 5th argument "noengram": the engram path still runs but adds nothing (ablation)
            t(p + "engram_wkv.weight", ENG_COLS * ENG_KEY, N_EMBD * (HC + 1), kind=("zeros" if "noengram" in sys.argv else "rand"))
            t(p + "engram_k.weight", N_EMBD * HC, kind="ones")
            t(p + "engram_q.weight", N_EMBD * HC, kind="ones")
        r = 0  # the V4 branch below does not apply
    if r:
        co = 2 if r == 4 else 1  # CSA pools two overlapping groups (state width 2*head), HCA one disjoint group per row
        t(p + "attn_compress_kv.weight", N_EMBD, co * HEAD)
        t(p + "attn_compress_gate.weight", N_EMBD, co * HEAD)
        t(p + "attn_compress_ape.weight", co * HEAD, r)
        t(p + "attn_compress_norm.weight", HEAD, kind="ones")
    if r == 4:
        IDX, NIH = 64, 4
        t(p + "indexer.attn_q_b.weight", Q_LORA, NIH * IDX)
        t(p + "indexer.proj.weight", N_EMBD, NIH)
        t(p + "indexer.compress_kv.weight", N_EMBD, 2 * IDX)
        t(p + "indexer.compress_gate.weight", N_EMBD, 2 * IDX)
        t(p + "indexer.compress_ape.weight", 2 * IDX, r)
        t(p + "indexer.compress_norm.weight", IDX, kind="ones")
    for s in ("attn", "ffn"):
        t(p + "hc_%s_fn.weight" % s, N_EMBD * HC, NTOT)
        t(p + "hc_%s_scale.weight" % s, 3, kind="ones")
        t(p + "hc_%s_base.weight" % s, NTOT, kind="zeros")
    t(p + "ffn_norm.weight", N_EMBD, kind="ones")
    t(p + "ffn_gate_inp.weight", N_EMBD, N_EXP)
    t(p + "exp_probs_b.bias", N_EXP, kind="zeros")
    t(p + "ffn_gate_exps.weight", N_EMBD, FF_EXP, N_EXP)
    t(p + "ffn_up_exps.weight", N_EMBD, FF_EXP, N_EXP)
    t(p + "ffn_down_exps.weight", FF_EXP, N_EMBD, N_EXP)
    t(p + "ffn_gate_shexp.weight", N_EMBD, FF_EXP * N_SHARED)
    t(p + "ffn_up_shexp.weight", N_EMBD, FF_EXP * N_SHARED)
    t(p + "ffn_down_shexp.weight", FF_EXP * N_SHARED, N_EMBD)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out, "tensors", n)
