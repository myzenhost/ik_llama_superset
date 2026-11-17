# TODO: should not use this
if (WIN32)
    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()
llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

set(LLAMA_CORE_SOURCES
    llama.cpp
    llama-adapter.cpp
    llama-arch.cpp
    llama-batch.cpp
    llama-chat.cpp
    llama-context.cpp
    llama-cparams.cpp
    llama-grammar.cpp
    llama-graph.cpp
    llama-hparams.cpp
    llama-impl.cpp
    llama-io.cpp
    llama-kv-cache.cpp
    llama-kv-cache-iswa.cpp
    llama-kv-cache-dsa.cpp
    llama-kv-cache-dsa-iswa.cpp
    llama-kv-cache-msa.cpp
    llama-kv-cache-dsv4.cpp
    llama-memory.cpp
    llama-moe-cache.cpp
    llama-memory-hybrid.cpp
    llama-memory-hybrid-iswa.cpp
    llama-memory-hybrid-idx.cpp
    llama-memory-recurrent.cpp
    llama-mmap.cpp
    llama-model-loader.cpp
    llama-model-saver.cpp
    llama-model.cpp
    llama-quant.cpp
    llama-sampler.cpp
    llama-vocab.cpp
    unicode-data.cpp
    unicode.cpp
)

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-spec-features.cpp
            llama-spec-features-dflash.cpp
            llama-dflash.cpp
            llama-dsv4.cpp
            llama-vocab.cpp
            llama-grammar.cpp
            llama-sampling.cpp
            # ADDED 2026-09-08. llama-sampler.cpp is upstream's complete sampler-chain
            # implementation. It has been sitting on disk since the merge, named by NOTHING -- the
            # same build-graph invisibility that stranded examples/mtmd: file present, symbols
            # present, no marker, nothing deleted, and never compiled.
            # MEASURED (temp/_sampler_overlap.py, both controls passing): it defines 38 non-static
            # symbols the build does not have AT ALL -- llama_sampler_init, _chain_init/_add/_get/
            # _n/_remove, _accept/_apply/_sample/_clone/_free/_name, and the whole init_* family
            # (top_k, top_p, min_p, typical, temp, temp_ext, xtc, mirostat, mirostat_v2, dist,
            # greedy, infill, penalties, dry, logit_bias, top_n_sigma, adaptive_p, empty) plus the
            # llama_perf_sampler helpers. include/llama.h DECLARES every one of them, so each was
            # an unresolved symbol waiting at link.
            # RESOLVED 2026-09-27 (S69-CHAT-TEMPLATES-MONOLITH-LACKS-14-UPSTREAM-2026-09-27-01): the
            # line below this comment used to name llama-chat.cpp among the files "deliberately NOT
            # in this group; they need the collision decided first". The collision is now decided
            # the same way WIRE-68 decided llama-context.cpp's: src/llama.cpp's own duplicate
            # `enum llm_chat_template` / `LLM_CHAT_TEMPLATES` map / llama_chat_detect_template /
            # llama_chat_apply_template_internal / llama_chat_builtin_templates were deleted in the
            # same commit that adds llama-chat.cpp below, so exactly one definition of each exists.
            # Measured before/after with tests/test-chat-template-parity.cpp (union of every named
            # template on either side, applied through the public llama_chat_apply_template):
            # 40 names shared with the monolith's old copy stayed byte-identical except 5 judged
            # dispositions (gemma/orion adopt upstream's more-correct multi-system-message `+=`;
            # rwkv-world/kimi-k2 adopt upstream as bug fixes -- the monolith inverted kimi-k2's
            # user/assistant tags and ignored add_ass entirely for rwkv-world; chatglm4/granite-3.x/
            # hunyuan-moe ALSO follow upstream -- a first pass kept the monolith's output, and each was
            # reverted after reading the model's own template: the monolith dropped chatglm4's trailing
            # \n, added one to granite's generation prompt, and prefixed hunyuan's assistant turn with
            # <|startoftext|>, none of which the models' templates do), the fork's 3 extras (bitnet, deepseek4,
            # falcon_e) were ported into llama-chat.cpp, and the 14 upstream-only names (phi4,
            # granite-4.0/4.1, mistral-v7-tekken, etc.) now work instead of erroring or, for
            # mistral-v7-tekken, silently mis-detecting as plain llama2.
            # CORRECTED 2026-09-08: this line used to read "Only 2 names collide with
            # llama-sampling.cpp, and both are resolved there." The measured number
            # (temp/_sampler_file_overlap.py, both controls passing) is 13 names defined in BOTH
            # files, of which 4 genuinely collide:
            #   llama_sampler_init_grammar / _lazy / _lazy_patterns  same params, different return
            #   llama_sampler_reset                                  identical signature, 2 bodies
            # The other 9 are `static` in at least one file, so they have internal linkage and are
            # harmless -- which is very likely where "2" came from: counting what the COMPILER
            # reported rather than comparing the files. llama_sampler_reset produced no diagnostic
            # at all, two identical non-static definitions in one library with one silently winning
            # at link, and it was the more dangerous of the two shapes.
            # All four are now resolved: the duplicated block was removed from llama-sampling.cpp
            # (its 26 fork-only llama_sample_*_impl symbols, adaptive_p included, are untouched)
            # and llama-sampler.cpp owns them, matching include/llama.h:2362 and the call sites in
            # common/sampling.cpp:88,91.
            llama-sampler.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-load-tensors.cpp
            llama-build-context.h
            llama-build-context.cpp
            llama-kv-snapkv.h
            llama-kv-snapkv.cpp
            # S69-KT-EXPERT-POOL-2026-09-28: added here on creation, not after-the-fact --
            # WIRE-67 above shows what "present on disk, absent from this list" costs.
            llama-moe-pool.h
            llama-moe-pool.cpp
            # S69-FASTLLM-PICK-MEASURED-EXPERT-ROUTE-LIMIT-2026-10-05-01: planner (decision function) for the CPU/GPU expert split; named here on creation
            llama-moe-route-limit.h
            llama-moe-route-limit.cpp
            llama-cram-probe.h
            llama-cram-probe.cpp
            llama-cram-soft.h
            llama-cram-soft.cpp
            # WIRE-67 (2026-09-13): the memory / KV-cache family. Present on disk since the merge,
            # named by NOTHING, so never compiled -- the same build-graph invisibility as WIRE-66
            # and llama-sampler.cpp above. The linker was the instrument that found it: the
            # full-tree build produced 32 ld failures whose undefined symbols are dominated by
            # these files' CONSTRUCTORS, at 120 references each for llama_kv_cache and
            # llama_kv_cache_iswa, 60 for llama_kv_cache_dsa, 30 for llama_memory_recurrent and
            # llama_memory_hybrid_iswa (temp/_fullbuild3.log, grouped in temp/_unbuilt_src_out.txt).
            #
            # MEASURED before adding, because the risk here is a duplicate definition rather than a
            # missing one -- src/llama.cpp is this fork's monolith and a duplicate that AGREES on
            # its signature produces NO diagnostic, one of them silently wins (the llama_sampler_reset
            # shape). temp/_unbuilt_src_files.py compared each stranded file's definitions against
            # everything already compiled: this group reported ZERO collisions. llama-context.cpp
            # (38 collisions) and llama-batch.cpp / llama-chat.cpp / llama-kv-cache-dsv4.cpp are
            # deliberately NOT in this group; they need the collision decided first and are tracked
            # separately.
            #
            # The .selftest.cpp files are also left out: they are test mains, not library code.
            #
            # WIRE-68 (2026-09-13): llama-context.cpp, the file this comment block names as
            # "deliberately NOT in this group; needs the collision decided first". The collision is
            # now decided, and src/llama.cpp's 38 duplicate definitions are deleted in the same
            # commit that adds this line -- upstream PR #10902 split llama_context out and never
            # let both bodies exist, and neither do we. Adding the file without deleting them would
            # simply fail to link, which is why the two halves are one change.
            #
            # THE 38 WERE NOT ASSUMED EQUIVALENT. 31 are 3-5 line forwarders. The other 7 were each
            # compared body-against-ported-implementation by identifier set:
            #   llama_synchronize             nothing in the monolith's body goes unmentioned
            #   llama_decode, llama_free      REAL gaps -- TriAttention scoring and cleanup were
            #                                 missing from the ported side. Ported in ba0196f20 and
            #                                 confirmed by a second method (whole-file counts at
            #                                 three revisions: 0 before, 1 after, control passing)
            #   llama_get_logits_ith          NOT a gap: the monolith's inline bounds checks were
            #   llama_get_embeddings_ith      factored into llama_context::output_resolve_row()
            #                                 (llama-context.cpp:1203) with identical messages, and
            #                                 the ported side adds a get_sampled_logits_ith path
            #                                 the monolith lacks
            #   llama_context_default_params  its 41-identifier gap is comment prose and two shas
            #
            # AND THE THREE FORK FEATURES the monolith's inline llama_init_from_model carried and
            # this file did not -- triattention, worst_graph_tokens, repack -- are ported
            # (85eafd618, fb6f527c1, ba0196f20). Ten of the thirteen were already here: four in the
            # constructor, six elsewhere in the file. repack needed llama_repack_up_gate_exps to
            # lose its `static`, since internal linkage was the only thing keeping it unreachable.
            llama-context.cpp
            llama-memory.cpp
            llama-memory-recurrent.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
            # llama-memory-hybrid-idx.cpp: S69-CLOSE-LLAMA-MEMORY-HYBRID-IDX-KV-API-GAP-2026-09-13.
            # Previously DELIBERATELY excluded here: it was written against a newer kv-cache API
            # this tree did not carry. All three gaps the prior comment named are now closed, HONEST
            # implementations not stubs, in llama-kv-cache.h/.cpp and llama-kv-cells.h:
            #   * llama_kv_cache ctor gained a 17th parameter, `const char * name_prefix = ""`,
            #     threaded into the cache_k_l%d/cache_v_l%d ggml_format_name calls so a sibling
            #     cache over the same layer range (the "idx_" indexer) does not mint tensor names
            #     identical to the attention cache's;
            #   * llama_kv_cache::state_read_sinfo(io, seq_id, flags, out, in) is a new public
            #     method: state_read(...) is now defined as state_read_sinfo(..., nullptr, nullptr).
            #     `out` collects the slot_info actually used per non-empty stream; `in`, when given,
            #     is consumed positionally instead of computing a slot_info, via a new private
            #     state_read_meta_adopt() that force-writes cells at the given indices (dropping any
            #     stale content there first) rather than running find_slot(). No upstream mainline
            #     construct exists for this -- the fork's own restore path was extended in place;
            #   * llama_kv_cells::seq_get_all(i) returns the cell's full sequence bitset (the
            #     bitset's type, seq_set_t, was made public to expose it) for comparing whether two
            #     cells share the same sequence set.
            llama-memory-hybrid-idx.cpp
            llama-kv-cache.cpp
            llama-kv-kivi-sim.cpp
            llama-kv-cache-iswa.cpp
            llama-kv-cache-dsa.cpp
            llama-kv-cache-dsa-iswa.cpp
            llama-kv-cache-msa.cpp
            llama-kv-rocketkv.cpp
            llama-io.cpp
            llama-model-saver.cpp
            # Added in the same group after the first link measurement: llama_batch_allocr is the
            # largest remaining undefined cluster (300 references to split_reset/get_n_used/
            # get_n_tokens, plus split_equal/split_simple/split_seq/ubatch_reserve/ctor). Its three
            # colliding C-linkage definitions were cut at the source -- see the comment at the
            # bottom of llama-batch.cpp for which three and why src/llama.cpp keeps them.
            llama-batch.cpp
            # WIRE-67b (2026-09-13): llama-graph.cpp, the last stranded file with a collision
            # surface small enough to decide in one step. It resolves llm_graph_result::
            # {set_outputs,set_inputs,reset,add_input} and llm_graph_context::{build_sampling,
            # build_pooling,build_dense_out}, 30 references each. Its ONE collision,
            # llama_relative_position_bucket, is resolved by cutting src/llama.cpp's copy --
            # see the comment at the cut site there, and the row it closes.
            #
            # src/CMakeLists.txt:103 asserts "built src/llama-graph.cpp already CALLS ...".
            # That was FALSE until this line: the file appeared in this CMakeLists only inside
            # that comment, and its .obj was absent from the build directory.
            llama-graph.cpp
            # WIRE-67c (2026-09-13). Held out of WIRE-67 on a measured "7 collisions"
            # with src/llama-dsv4.cpp. That measurement was WRONG: the extractor that made
            # it did not exclude `static`, and all seven names are static in BOTH files, so
            # they have internal linkage and cannot collide. Re-measured with the corrected
            # extractor (temp/_verify_def_regex.py, 13 controls both directions): the two
            # files share ZERO non-static symbols, while still yielding 14 and 77 of their
            # own -- so the clean result is not an empty parse.
            #
            # Needed now because wiring llama-graph.cpp above resolved 7 symbols and exposed
            # 16 new ones, every one a llama_kv_cache_dsv4_*_context member (get_csa/get_hca/
            # get_lid/get_raw, the *_plan accessors, build/set_input_k_rot, k_idxs, kq_mask,
            # get_n_kv, get_n_write). They are all defined here.
            llama-kv-cache-dsv4.cpp
            # TriAttention KV scoring (S55-IKLLAMAUNIFIED-002)
            triattention.c
            triattention.h
            triattention-runtime.c
            triattention-runtime.h
            triattention-bridge.cpp
            # offload-attention (#220): select + gather + attend. The header has declared
            # this contract since 2026-07-06; the CPU reference is
            # cortex/cortex/inference/offload_attention.py (cosine 1.000000 vs dense).
            offload_attention.h
            offload_attention.cpp
            offload_attention_bridge.cpp
            offload_attention_op.cpp
            offload_attention_mask_op.cpp
            llama-embed-pooling.h
            llama-embed-pooling.cpp
            # WIRE-66 (2026-09-08) group 1: proof-of-route, zero measured collision with
            # src/llama.cpp (temp/_adapter_rename_surface.py). Provides llama_max_parallel_sequences
            # (declared include/llama.h:898, defined nowhere built before this).
            llama-cparams.h
            llama-cparams.cpp
            # WIRE-66 group 1: provides llama_log_get (declared-nowhere-built before this).
            # llama_log_set/_internal/_callback_default/gguf_kv_to_str were cut from the .cpp --
            # see the comment block at the top of src/llama-impl.cpp -- because src/llama.cpp
            # already defines all four against its own g_state; wiring them verbatim would be a
            # duplicate-symbol link error for a symbol that already resolves correctly.
            llama-impl.h
            llama-impl.cpp
            llama-moe-cache.h
            llama-moe-cache.cpp
            llama-model.h
            llama-model.cpp
            # factory for the architectures that exist only as src/models classes (no legacy loader); GENERATED by temp/_tca_gen_factory_1004.py
            llama-model-create.cpp
            # WIRE-66 (2026-09-08) group 2. Measured (temp/_wire66_adapter_overlap.py, 3 controls
            # passing): ZERO symbol-name collision with src/llama.cpp (0 of 12 non-static defs
            # shared). The fork keeps a SEPARATE, distinct old-name family alive in src/llama.cpp
            # (llama_lora_adapter_init/_set/_remove/_clear/_free operating on struct
            # llama_lora_adapter) alongside include/llama.h's new-name block ("--- adopted from
            # upstream in this merge ---", :1111) declaring llama_adapter_lora_* over a DIFFERENT
            # struct (llama_adapter_lora, src/llama-adapter.h) -- different member layout
            # (ab_map/ctxs/bufs/alora_invocation_tokens vs whatever llama.cpp's version carries).
            # Different types => a thin forwarding wrapper is not a cast (per task instructions),
            # so this wires the real implementation rather than wrapping the old one.
            # Already load-bearing, not optional: src/llama-graph.h (built, included transitively
            # by src/llama-model.h) stores `const llama_adapter_cvec *` / `const llama_adapter_loras
            # *` in its live per-request param structs, and built src/llama-graph.cpp already CALLS
            # cvec->apply_to(...) and lora.first->get_weight(w) (:1508,1522,1558,1717,2337) -- both
            # member functions defined ONLY in llama-adapter.cpp. So the current build already has
            # undefined-reference link failures for these before this file is wired; this closes
            # them as well as the 8 declared-API free functions. common/common.cpp:1806 already
            # calls the new name (llama_adapter_lora_init) into this exact type.
            # src/llama.cpp's old-name family is left untouched (not my file per task scope) --
            # dead-but-harmless, zero collision, someone else's cleanup.
            llama-adapter.h
            llama-adapter.cpp
            llama-reload-info.h
            llama-reload.cpp
            llama-quantize.cpp
            # WIRE-66 (2026-09-08) group 3. Provides the 6 llama-ext.h-declared testing/tooling
            # accessors (llama_ftype_get_default_type, llama_quant_init/_free/
            # _model_from_metadata/_tensor_allows_quantization/_compute_types) called by built
            # tests/test-quant-type-selection.cpp. TWO duplicate-symbol collisions measured against
            # already-built files (llama_model_quantize_default_params vs src/llama.cpp:8150,
            # llama_model_quantize vs src/llama-quantize.cpp:1870 -- the task brief named only the
            # second) -- both cut from src/llama-quant.cpp itself; see the comment block left at
            # that exact spot in the file.
            llama-quant.h
            llama-quant.cpp
            llama-arch.h
            llama-arch.cpp
            llama-hparams.h
            llama-hparams.cpp
            llama-delta-net.h
            llama-delta-net.cpp
            llama-kda.h
            llama-kda.cpp
            # In-Place TTT graph route + state layout (row S69-FORK-INPLACE-TTT-2026-10-26-01)
            llama-ttt.h
            llama-ttt.cpp
            # S69-CHAT-TEMPLATES-MONOLITH-LACKS-14-UPSTREAM-2026-09-27-01: upstream's chat-template
            # implementation (MIT), now the ONLY one -- src/llama.cpp's duplicate copy was deleted
            # in this same commit. See the resolution comment above the `add_library(llama` block.
            llama-chat.h
            llama-chat.cpp
            unicode.h
            unicode.cpp
            unicode-data.cpp
            unicode-nfkc-data.h
            unicode-nfkc-data.cpp
            unicode-script-data.cpp
            # graphs
            graphs/build_eagle3.cpp
            graphs/build_kimi_k3.cpp
            graphs/build_granite_hybrid.cpp
            graphs/build_lfm2.cpp
            graphs/build_inkling.cpp
            graphs/build_hyperclovax.cpp
            graphs/build_llama.cpp
            graphs/build_muse_glimmer.cpp
            graphs/build_mistral3.cpp
            graphs/build_deci.cpp
            graphs/build_baichuan.cpp
            graphs/build_xverse.cpp
            graphs/build_falcon.cpp
            graphs/build_grok.cpp
            graphs/build_dbrx.cpp
            graphs/build_starcoder.cpp
            graphs/build_starcoder2.cpp
            graphs/build_refact.cpp
            graphs/build_bert.cpp
            graphs/build_bloom.cpp
            graphs/build_mpt.cpp
            graphs/build_stablelm.cpp
            graphs/build_seedoss.cpp
            graphs/build_laguna.cpp
            graphs/build_modernbert.cpp
            graphs/build_k2horizon.cpp
            graphs/build_step35.cpp
            graphs/build_qwen.cpp
            graphs/build_qwen2.cpp
            graphs/build_qwen3.cpp
            graphs/build_mellum.cpp
            graphs/build_qwen3next.cpp
            graphs/build_qwen4exp.cpp
            graphs/build_qwen35.cpp
            graphs/build_phi2.cpp
            graphs/build_phi3.cpp
            graphs/build_plamo.cpp
            graphs/build_gpt2.cpp
            graphs/build_codeshell.cpp
            graphs/build_orion.cpp
            graphs/build_internlm2.cpp
            graphs/build_minicpm.cpp
            graphs/build_gemma.cpp
            graphs/build_gemma2.cpp
            graphs/build_gemma3.cpp
            graphs/build_gemma_embedding.cpp
            graphs/build_gemma4.cpp
            graphs/build_diffusion_gemma.cpp
            graphs/build_dflash.cpp
            graphs/build_mamba.cpp
            graphs/build_command_r.cpp
            graphs/build_olmo.cpp
            graphs/build_openelm.cpp
            graphs/build_gptneox.cpp
            graphs/build_arctic.cpp
            graphs/build_deepseek2.cpp
            graphs/build_deepseek4.cpp
            graphs/build_openpangu.cpp
            graphs/build_glm4.cpp
            graphs/build_bitnet.cpp
            graphs/build_cohere2.cpp
            graphs/build_cohere2_moe.cpp
            graphs/build_t5.cpp
            graphs/build_jais.cpp
            graphs/build_chatglm.cpp
            graphs/build_dots1.cpp
            graphs/build_ernie45.cpp
            graphs/build_hunyuan.cpp
            graphs/build_mimo2.cpp
            graphs/build_openai.cpp
            graphs/build_bailingmoe2.cpp
            graphs/build_bailingmoe3.cpp
            graphs/build_glm5next.cpp
            graphs/build_minimaxm2.cpp
            graphs/build_minimaxm3.cpp
            graphs/build_smollm3.cpp
            
            models/afmoe.cpp
            models/apertus.cpp
            models/arcee.cpp
            models/arctic.cpp
            models/arwkv7.cpp
            models/baichuan.cpp
            models/bailingmoe.cpp
            models/bailingmoe2.cpp
            models/bailingmoe3.cpp
            models/beam.cpp
            models/bert.cpp
            models/bitnet.cpp
            models/bloom.cpp
            models/chameleon.cpp
            models/chatglm.cpp
            models/clef.cpp
            models/clip.cpp
            models/codeshell.cpp
            models/cogvlm.cpp
            models/cohere2.cpp
            models/cohere2moe.cpp
            models/command-r.cpp
            models/dbrx.cpp
            models/deci.cpp
            models/deepseek.cpp
            models/deepseek2.cpp
            models/deepseek2ocr.cpp
            models/deepseek32.cpp
            models/deepseek4.cpp
            models/deepseek41.cpp
            models/delta-net-base.cpp
            models/dflash.cpp
            models/dots1.cpp
            models/dots3note.cpp
            models/dream.cpp
            models/eagle3.cpp
            models/ernie4-5-moe.cpp
            models/ernie4-5.cpp
            models/eurobert.cpp
            models/exaone-moe.cpp
            models/exaone.cpp
            models/exaone4.cpp
            models/falcon-h1.cpp
            models/falcon.cpp
            models/flint.cpp
            models/gemma-embedding.cpp
            models/gemma-embedding2.cpp
            models/gemma.cpp
            models/gemma2.cpp
            models/gemma3.cpp
            models/gemma3n.cpp
            models/gemma4-assistant.cpp
            models/gemma4.cpp
            models/glm-dsa.cpp
            models/glm4-moe.cpp
            models/glm4.cpp
            models/glm5-next.cpp
            models/gpt2.cpp
            models/gptj.cpp
            models/gptneox.cpp
            models/granite-hybrid.cpp
            models/granite-moe.cpp
            models/granite-swa.cpp
            models/granite-switch.cpp
            models/granite.cpp
            models/grok.cpp
            models/grovemoe.cpp
            models/hrm-text.cpp
            models/hunyuan-dense.cpp
            models/hunyuan-moe.cpp
            models/hunyuan-vl.cpp
            models/hy-v3.cpp
            models/hy-v4.cpp
            models/internlm2.cpp
            models/jais.cpp
            models/jais2.cpp
            models/jamba.cpp
            models/jina-bert-v2.cpp
            models/jina-bert-v3.cpp
            models/k2-horizon.cpp
            models/kimi-k3.cpp
            models/kimi-linear.cpp
            models/kolibri1.cpp
            models/mmfreelm.cpp
            models/laguna.cpp
            models/lfm-base.cpp
            models/lfm2.cpp
            models/lfm2moe.cpp
            models/llada-moe.cpp
            models/llada.cpp
            models/llama-embed.cpp
            models/llama.cpp
            models/llama4.cpp
            models/maincoder.cpp
            models/maion-coder.cpp
            models/mamba-base.cpp
            models/mamba.cpp
            models/mamba2.cpp
            models/maple.cpp
            models/mellum.cpp
            models/mimo2.cpp
            models/minicpm.cpp
            models/minicpm3.cpp
            models/minimax-01.cpp
            models/minimax-m2.cpp
            models/minimax-m3.cpp
            models/mistral3.cpp
            models/mistral4.cpp
            models/modern-bert.cpp
            models/mpt.cpp
            models/muse-glimmer.cpp
            models/muse-spark.cpp
            models/nanbeige.cpp
            models/nemotron-h-moe.cpp
            models/nemotron-h.cpp
            models/nemotron.cpp
            models/neo-bert.cpp
            models/nomic-bert-moe.cpp
            models/nomic-bert.cpp
            models/olmo.cpp
            models/olmo2.cpp
            models/olmoe.cpp
            models/openai-moe.cpp
            models/openelm.cpp
            models/orion.cpp
            models/paddleocr.cpp
            models/pangu-embed.cpp
            models/phi2.cpp
            models/phi3.cpp
            models/phimoe.cpp
            models/plamo.cpp
            models/plamo2.cpp
            models/plamo3.cpp
            models/plm.cpp
            models/pockettts.cpp
            models/qwen.cpp
            models/qwen2.cpp
            models/qwen2moe.cpp
            models/qwen2vl.cpp
            models/qwen3.cpp
            models/qwen35.cpp
            models/qwen35moe.cpp
            models/qwen3moe.cpp
            models/qwen3next.cpp
            models/qwen3tts.cpp
            models/qwen3vl.cpp
            models/qwen3vlmoe.cpp
            models/qwen4exp.cpp
            models/refact.cpp
            models/rnd1.cpp
            models/rwkv6-base.cpp
            models/rwkv6.cpp
            models/rwkv6qwen2.cpp
            models/rwkv7-base.cpp
            models/rwkv7.cpp
            models/seed-oss.cpp
            models/sensenova-u1.cpp
            models/smallthinker.cpp
            models/smollm3.cpp
            models/spark2-5.cpp
            models/stablelm.cpp
            models/starcoder.cpp
            models/starcoder2.cpp
            models/step35.cpp
            models/t5.cpp
            models/t5encoder.cpp
            models/talkie.cpp
            models/vllm-decision2.cpp
            models/wavtokenizer-dec.cpp
            models/xverse.cpp)

# ADDED 2026-11-04 (upstream merge batch 41): 16 upstream-new models/*.cpp (beam, clef, deepseek41, flint,
# gemma-embedding2, glm5-next, hrm-text, k2-horizon, kolibri1, lfm-base, maion-coder, maple, muse-spark, qwen4exp,
# sensenova-u1, vllm-decision2). Upstream compiles them through its models/*.cpp glob; this list is explicit.

# ADDED 2026-09-27 (engine/unsloth-models-2026-09-27, S69-ENGINE-MERGE-UNSLOTH-LLAMACPP-2026-09-09-01
# / S69-MODELS-GLOB-GATED-ON-N-LAYER-ACCESSOR-132-OF-150-2026-09-09-01): the 18 models/*.cpp files
# above marked new this pass (apertus, deepseek32, deepseek4, delta-net-base, dots3note,
# gemma4-assistant, glm-dsa, granite-switch, hy-v4, kimi-k3, mamba-base, nanbeige, plamo2,
# rwkv6-base, rwkv7-base, spark2-5, step35, wavtokenizer-dec) compile clean under this target's own
# flags (measured via a throwaway EXCLUDE_FROM_ALL probe target built from the pre-existing
# LLAMA_MODELS_SOURCES glob at the top of this file, then reverted). The 132-of-150 accessor defect
# named in the row above (hparams.n_layer() vs this fork's n_layer field) and the 7 one-off
# ggml-level gaps it also named were BOTH already fixed by prior commits (7b620ae9 accessor
# mapping, 670dd0efb ggml op constructors, per findings/S69_ENGINE_UNSLOTH_MODELS_2026-09-27.md) --
# re-verified by this pass's fresh compile, not assumed from the stale row.
#
# 3 of the 18 (deepseek4, kimi-k3, step35) have a same-named twin in src/graphs/build_*.cpp; checked
# each pair defines DIFFERENT symbols (llama_model_X::load_arch_hparams/tensors/build_arch_graph vs
# llm_build_context::build_X()), the same shape already proven safe for the other 56 such pairs.
#
# THIS IS TU WIRING ONLY, NOT A DISPATCHER. Per memory/project-fork-models-wired-but-no-dispatcher
# and row S69-MODELS-CLASSES-COMPILE-AND-LINK-BUT-NOTHING-CALLS-THEM-2026-09-09-01: nothing calls
# load_arch_hparams/load_arch_tensors/build_arch_graph on ANY of these 152 classes (0 call sites
# outside the pure-virtual declarations and each class's own pure-virtual overrides), so adding
# these 18 changes only what the linker sees, not runtime behaviour -- verified by a before/after
# llama-server build + test-backend-ops run in this task's findings file, both green.

target_include_directories(llama PUBLIC . ../include)
# upstream: llama-version.h is generated (src/llama.cpp includes it)
configure_file(llama-version.h.in ${CMAKE_CURRENT_BINARY_DIR}/llama-version.h @ONLY)
target_include_directories(llama PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(llama PRIVATE ../ggml/src)
target_compile_features   (llama PUBLIC cxx_std_17)

# ADDED 2026-09-08. src/llama.cpp:198 is upstream's `const char * llama_version(void) { return
# LLAMA_VERSION; }`, which arrived with the merge while upstream's BUILD WIRING for it did not:
# 279b4bfaf^2:src/CMakeLists.txt:54-57 passes LLAMA_VERSION (and LLAMA_COMMIT) as compile
# definitions, and this file had neither. So the code compiled everywhere except at the one line
# that needs the macro -- `error C2065: 'LLAMA_VERSION': undeclared identifier`.
# Both CMake variables already exist at the top level (CMakeLists.txt:17/20 and :162-163), so this
# only connects them; nothing new is invented and no version string is hard-coded here.
# 2026-11-03 upstream merge: LLAMA_VERSION / LLAMA_COMMIT now come from the generated llama-version.h
# (configure_file above; src/llama.cpp includes it). Defining them here as well would be a second definition.

find_package(Threads REQUIRED)
target_link_libraries(llama PUBLIC ggml PRIVATE Threads::Threads)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()

if (GGML_MAX_CONTEXTS)
    add_compile_definitions(GGML_MAX_CONTEXTS=${GGML_MAX_CONTEXTS})
endif()
if (GGML_MAX_SRC)
  add_compile_definitions(GGML_MAX_SRC=${GGML_MAX_SRC})
endif()

# Same MinGW _WIN32_WINNT gap for the llama target: llama-mmap.cpp's model-load
# PrefetchVirtualMemory call sits behind `#if _WIN32_WINNT >= 0x602` and was compiled out.
# SOURCE-READ: Thireus/ik_llama.cpp 6a1a8f57b (ik_llama #2563, MIT) src/CMakeLists.txt; scoped
# to MINGW here so MSVC/CUDA build flags are unchanged. Test: tests/test-win-version-gates.cpp.
if (MINGW AND DEFINED GGML_WIN_VER)
    target_compile_definitions(llama PRIVATE _WIN32_WINNT=${GGML_WIN_VER})
endif()
