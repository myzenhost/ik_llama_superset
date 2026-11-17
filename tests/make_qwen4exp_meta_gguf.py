#!/usr/bin/env python3
# Metadata-only qwen4exp GGUFs for the load-validation checks in src/llama-hparams.cpp (row
# S69-QWEN4EXP-VALIDATION-PARITY-2026-10-28-01). No tensors: a GOOD file gets past hparams and fails later
# (tensors), each BAD file must fail IN hparams with its own message. Driven by
# tests/test_qwen4exp_meta_validation.py.
#
#   python make_qwen4exp_meta_gguf.py <tree-root> <out.gguf> <case>
#   case: good | ssm_conv0 | idx_top_k0 | hc1 | ple2 | ple_ngram1
import sys
from pathlib import Path

root, out, case = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(root / "gguf-py"))
import gguf  # noqa: E402

A = "qwen4exp"
w = gguf.GGUFWriter(str(out), A)
k = lambda s: s % A  # noqa: E731

# hparams read n_vocab from the token list before the arch block, so even a metadata-only file needs one
w.add_tokenizer_model("gpt2")
w.add_token_list(["<eos>"] + ["t%d" % i for i in range(63)])
w.add_uint32(k("%s.block_count"), 4)
w.add_uint32(k("%s.context_length"), 256)
w.add_uint32(k("%s.embedding_length"), 32)
w.add_uint32(k("%s.feed_forward_length"), 64)
w.add_uint32(k("%s.attention.head_count"), 2)
w.add_uint32(k("%s.attention.head_count_kv"), 1)
w.add_float32(k("%s.attention.layer_norm_rms_epsilon"), 1e-6)
w.add_array(k("%s.rope.dimension_sections"), [4, 4, 0, 0])
w.add_uint32(k("%s.expert_count"), 4)
w.add_uint32(k("%s.expert_used_count"), 2)
w.add_uint32(k("%s.ssm.conv_kernel"), 0 if case == "ssm_conv0" else 4)
w.add_uint32(k("%s.ssm.inner_size"), 32)
w.add_uint32(k("%s.ssm.state_size"), 16)
w.add_uint32(k("%s.ssm.time_step_rank"), 2)
w.add_uint32(k("%s.ssm.group_count"), 2)
w.add_uint32(k("%s.attention.indexer.head_count"), 2)
w.add_uint32(k("%s.attention.indexer.key_length"), 16)
w.add_uint32(k("%s.attention.indexer.top_k"), 0 if case == "idx_top_k0" else 64)
w.add_uint32(k("%s.hyper_connection.count"), 1 if case == "hc1" else 4)
w.add_uint32(k("%s.hyper_connection.low_rank"), 8)
w.add_array(k("%s.attention.compress_ratios"), [4, 4, 4, 4])
# one PLE layer (layer 0 is linear: (0+1) % 4 != 0)
w.add_array(k("%s.ple.layers"), [0, 1] if case == "ple2" else [0])
w.add_uint32(k("%s.ple.ngram_size"), 1 if case == "ple_ngram1" else 3)
w.add_uint32(k("%s.ple.heads_per_ngram"), 2)
w.add_uint32(k("%s.ple.conv_kernel"), 4)
w.add_uint32(k("%s.embedding_length_per_layer_input"), 8)
w.add_uint32(k("%s.ple.eos_token_id"), 1)
w.add_array(k("%s.ple.layer_multipliers"), [3, 5, 7])
w.add_array(k("%s.ple.head_offsets"), [0, 16, 32, 48])
w.add_array(k("%s.ple.head_vocab_sizes"), [16, 16, 16, 16])

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out, case)
