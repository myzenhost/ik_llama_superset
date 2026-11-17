llama_add_compile_flags()

# Optional args:
# - LINK: libraries to link against (defaults to llama-common, which also brings llama)
function(llama_build source)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs LINK)
    cmake_parse_arguments(LLAMA_BUILD "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_BUILD_UNPARSED_ARGUMENTS})

    if (NOT LLAMA_BUILD_LINK)
        set(LLAMA_BUILD_LINK llama-common)
    endif()

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    target_link_libraries(${TEST_TARGET} PRIVATE ${LLAMA_BUILD_LINK})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
# - LINK: libraries to link against (defaults to llama-common, which also brings llama)
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS LINK)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS})

    if (NOT LLAMA_TEST_LINK)
        set(LLAMA_TEST_LINK llama-common)
    endif()

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE ${LLAMA_TEST_LINK})

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

llama_build(test-sampling-history.cpp)
llama_test(test-sampling-history)

# RPC client supports_op queries the remote backend (RPC_CMD_SUPPORTS_OP) and caches it; in-process loopback rpc server on CPU with a
# test-only op refusal (GGML_RPC_TEST_UNSUPPORTED_OP). Needs no NPU. Row S69-NPU-RPC-CLIENT-SUPPORTS-OP-QUERY-REMOTE-2026-11-06-01.
if (GGML_RPC AND NOT BUILD_SHARED_LIBS)
    llama_build_and_test(test-rpc-supports-op.cpp LINK ggml)
    if (WIN32)
        target_link_libraries(test-rpc-supports-op PRIVATE ws2_32)
    endif()
endif()

# RPC correctness guards from upstream #29283 (alloc-size floor), #24292 (free_buffer drops stored graphs), #26500 (graph tensors owned by
# another server are sent host-side): two in-process CPU rpc servers, no model. Row S69-ENGREV-UPSTREAM-MERGE-RPC-V8-FEATURE-PORT-2026-11-03-01.
if (GGML_RPC AND NOT BUILD_SHARED_LIBS)
    llama_build_and_test(test-rpc-guards.cpp LINK ggml)
    if (WIN32)
        target_link_libraries(test-rpc-guards PRIVATE ws2_32)
    endif()
endif()

# RPC TCP transport (ggml/src/ggml-rpc/transport.cpp, internal symbols -> static build): listen queue, per-socket byte counters + recv wait,
# ack-frequency / QUICKACK, GGML_RPC_TCP_SPIN. Loopback, no model. Rows S69-ENGREV-DLLAMA-RPC-{SERVER-LISTEN-BACKLOG,SYNC-TIME-AND-BYTES-COUNTERS,TCP-TURBO-QUICKACK}-2026-11-06-01.
if (GGML_RPC AND NOT BUILD_SHARED_LIBS)
    llama_build_and_test(test-rpc-transport.cpp LINK ggml)
    target_include_directories(test-rpc-transport PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
    if (WIN32)
        target_link_libraries(test-rpc-transport PRIVATE ws2_32)
    endif()
endif()

# Loader arm for registry devices the #if chain does not cover (HEXAGON, DL plugins, future RKNPU/XDNA): a CPU-backed STUB0 device
# registered as type GPU must be nameable with -dev, hold layers and give CPU-equal logits. Needs LLAMA_TEST_TINY_GGUF (exit 77 = SKIPPED).
# Uses internal ggml-backend-impl.h, so not on a Windows shared build. Row S69-NPU-LOADER-EXTRA-BACKEND-ARM-REGISTRY-DEVICES-2026-11-06-01.
if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    llama_build_and_test(test-loader-extra-backend.cpp LINK llama)
    target_include_directories(test-loader-extra-backend PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
    set_tests_properties(test-loader-extra-backend PROPERTIES SKIP_RETURN_CODE 77)
endif()

# Per-type / per-device table of the weight-type placement gate (mirrors create_tensors_helper::buft_runs_weight over the public ggml
# backend API). Builds CPU-only; exit 77 = SKIPPED (no non-CPU device). Row S69-ENGREV-UPSTREAM-MERGE-TENSOR-INFOS-TABLE-DECIDE-WITH-LOADER-2026-11-04-01.
llama_build_and_test(test-weight-type-gate-cuda.cpp LINK ggml)
set_tests_properties(test-weight-type-gate-cuda PROPERTIES SKIP_RETURN_CODE 77)

# penalty fade (ExLlamaV3 decay), row S69-EXLLAMAV3-PICK-PENALTY-FADE-2026-10-18-01
llama_build(test-sampling-penalty-fade.cpp)
llama_test(test-sampling-penalty-fade)

# strict JSON-schema check (SGLang json_schema_validation.py), row S69-SGLANG-PICK-STRICT-JSON-SCHEMA-CHECK-2026-10-18-01
llama_build(test-json-schema-strict.cpp)
llama_test(test-json-schema-strict)

# top-k / top-p without a full-vocabulary sort (MLC cpu_sampler.cc), row S69-MLC-LLM-PICK-TOPK-TOPP-WITHOUT-FULL-SORT-2026-10-18-01
llama_build(test-sampling-nosort.cpp)
llama_test(test-sampling-nosort)
llama_build(test-sampling-skew.cpp)
llama_test(test-sampling-skew)

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
# HF-derived golden (tests/make-hf-vocab-golden.py: the .out ids come from the HF tokenizers library, not from this tree)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-v3 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-v3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-r1-qwen LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-r1-qwen.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-exaone LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-exaone.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-glm4 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-glm4.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-granite-3 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-granite-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-olmo2 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-olmo2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-4 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-4.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-seed-coder LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-seed-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-smollm LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-smollm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-exaone4 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-exaone4.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon-h1 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon-h1.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-hunyuan-dense LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-hunyuan-dense.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-minimax-m2 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-minimax-m2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mellum LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mellum.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-megrez LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-megrez.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-trillion LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-trillion.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-viking LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-viking.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-poro-chat LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-poro-chat.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-bloom LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bloom.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-stablelm2 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-stablelm2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-2 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-olmo LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-olmo.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-bailingmoe2 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bailingmoe2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-kanana2 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-kanana2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-youtu LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-youtu.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-minicpm5 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-minicpm5.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-superbpe LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-superbpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-smaug-bpe LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-smaug-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-lfm2 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-lfm2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-a.x-4.0 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-a.x-4.0.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-grok-2 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-grok-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon3 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-afmoe LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-afmoe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-oss LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-oss.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-granite-4.2 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-granite-4.2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-jina-v1-en LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-jina-v1-en.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-kimi-k2 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-kimi-k2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-tekken LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-tekken.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-bitnet-b1.58-2b-4t LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bitnet-b1.58-2b-4t.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama4 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama4.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf --check-decode)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen35 LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen35.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        LABEL behavior
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp  LABEL "main behavior"  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

llama_build(test-recurrent-state-rollback.cpp)
# llama_context state tests: KV and recurrent state save, load, copy, rollback (upstream; the rollback test above is this fork's)
set(LLAMA_CONTEXT_TEST_SRC
    test-llama-context.cpp
    llama-context/cases-kv.cpp
    llama-context/cases-rs.cpp
    llama-context/test-case.cpp
    llama-context/test-log.cpp
    llama-context/test-runner.cpp
    llama-context/test-util.cpp
)
llama_build(${LLAMA_CONTEXT_TEST_SRC})
# common_params_parse Windows argv repair must keep a caller-rewritten argv; driven by test-arg-argv-repair.py
llama_build(test-arg-argv-repair.cpp)
llama_build(test-spec-window-parity.cpp)
# DFlash target capture must not change target logits, one sequence or several per batch (row
# S69-SPEC-POOL-DFLASH-CONCURRENT-VERIFY-CORRUPTS-TARGET-2026-11-05-01); needs a real target model, so built, not registered
llama_build(test-dflash-capture-batch-invariance.cpp)
llama_build(test-spec-seed-parity.cpp)
# draft tokens after an EOG are not accepted (row S69-LLAMACPP-PICK-DRAFT-STOP-AT-EOG-2026-10-18-01, upstream d280808f)
llama_build(test-spec-draft-eog.cpp)
# KV-cache type with per-row metadata (q8_KV) is refused at context creation (row S69-ENGREV-THIREUS-ROWMETA-KV-TYPE-REFUSE-2026-10-27-01,
# upstream Thireus f228595b); needs a small GGUF as argv[1] (tests/make_tiny_llama_gguf.py), so built, not registered
llama_build(test-kv-row-meta-refuse.cpp)
# Slot file saved under checkpoint A is refused under checkpoint B of the same arch, naming the field (row S69-ENGREV-FERROX-SLOT-FILE-MODEL-IDENTITY-2026-10-27-01);
# needs a small GGUF as argv[1] (tests/make_tiny_llama_gguf.py), so built, not registered
llama_build(test-slot-identity.cpp)
# Hybrid (delta-net) multi-slot decode: node budget for the per-token sub-graphs + sequence-run split of mixed batches (row
# S69-ENGREV-THIREUS-HYBRID-MIXEDSEQ-MAXNODES-2026-10-27-01, upstream Thireus c2e3eba2); needs a qwen35/qwen3next GGUF as argv[1], so built, not registered
llama_build(test-hybrid-mixedseq-nodes.cpp)
# DFlash2 / proposal-dist acceptance must match the target's post-chain distribution (row S69-VLLM-W2-DFLASH2-ACCEPT-USES-CUR-P-2026-10-27-01)
llama_build(test-spec-proposal-accept.cpp)
llama_build_and_test(test-sampling-adaptive-order.cpp)
# upstream's common_params_sampling::samplers spelling + INFILL on the samplers_sequence chain (row S69-ENGREV-UPSTREAM-MERGE-SAMPLERS-SPELLING-INFILL-2026-11-05-01)
llama_build_and_test(test-sampling-samplers-spelling.cpp)
# one draft_sample state behind both spellings + the rule under a grammar mask; the -m run below drives the real sampler with a GBNF grammar (row S69-ENGREV-UPSTREAM-MERGE-DRAFT-SAMPLE-EQUIVALENCE-TEST-2026-11-04-01)
llama_build_and_test(test-spec-draft-sample-equiv.cpp)
# Parallel constrained decisions (tools/parallel-decision, POST /v1/decision). Part A needs no model; part B runs when a tiny GGUF
# (tests/make_tiny_llama_gguf.py <tree> out.gguf 2048 bytes) is given as argv[1] / env DECISION_TEST_MODEL.
llama_build_and_test(test-decision-engine.cpp)
target_link_libraries(test-decision-engine PRIVATE llama-decision)
# llama_mmap::release_fragment (Windows working-set release): internal class, not exported from a shared library
if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    llama_build_and_test(test-mmap-release.cpp)
    # Windows expert-warm arm of ggml_moe_prefetch_kernel_hook (internal ggml symbols)
    llama_build_and_test(test-moe-prefetch-win.cpp)
endif()

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-unicode.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-sampling.cpp  LABEL behavior  LINK llama)
    # Speculative-decoding acceptance rules (common/spec-accept.h): both emit p exactly (chi-square), each accepts at
    # its analytic rate, and two wrong rules must fail the same test. Added 2026-09-28.
    llama_build_and_test(test-spec-accept.cpp)
    # EAGLE3 d2t map convention (common/eagle3-d2t.h): absolute and delta maps both resolve to the right target ids,
    # and the old delta-only reading is shown wrong on an absolute map. Model-free. Added 2026-10-17.
    llama_build_and_test(test-eagle3-d2t.cpp)
    # On-demand LoRA adapter pool + round-robin scheduler (common/lora-adapter-pool.h), fake loader, model-free.
    llama_build_and_test(test-lora-adapter-pool.cpp)
    # B*A*scale delta builder + TIES/DARE merge checked against a naive reference written in the test (common/lora-merge.h). Model-free.
    llama_build_and_test(test-lora-merge-ref.cpp)
    # SGMV-style heterogeneous batched LoRA reference (common/lora-sgmv.h), grouped vs per-token, plus ggml MUL_MAT_ID timing. Model-free.
    llama_build_and_test(test-lora-sgmv-grouped.cpp)
    # Per-sequence LoRA (llama_set_seq_adapters_lora): one batch whose sequences use different adapters equals each sequence alone; a random-weight
    # model + adapters come from scripts/gen-tiny-lora-test-assets.py (numpy + gguf-py). Exits 77 (SKIPPED) without numpy.
    # ggml_gallocr_alloc_graph with a graph whose leaf ORDER differs from the previous one's (same counts): needs the leaf check in ggml_gallocr_needs_realloc
    llama_build_and_test(test-gallocr-leaf-order.cpp)
    llama_build(test-lora-per-seq.cpp)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_Interpreter_FOUND)
        add_test(NAME test-lora-per-seq
                 COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/run-with-tiny-lora-assets.py $<TARGET_FILE:test-lora-per-seq>)
        set_tests_properties(test-lora-per-seq PROPERTIES SKIP_RETURN_CODE 77 LABELS main)
    endif()
    # Prevention scan (row S69-ENGREV-PEFT-LORA-FASTPATH-GUARD-LEGACY-MAP-ONLY-2026-10-27-01): no graph-side fast path may be guarded by
    # lora_adapters.empty() (the legacy map only); llm_lora_targets() checks legacy + context-wide + per-sequence adapters. Source scan, no model.
    if (Python3_Interpreter_FOUND)
        add_test(NAME test-lora-fastpath-guard-scan
                 COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/test-lora-fastpath-guard-scan.py ${PROJECT_SOURCE_DIR})
        set_tests_properties(test-lora-fastpath-guard-scan PROPERTIES LABELS adapters)
    endif()
    # Chat-handler overlap (row S69-ENGREV-TABBYAPI-CHAT-HANDLER-LOADLOG-2026-10-27-01 part 2): reads the detect ladder from
    # common/chat.cpp and reports, per models/templates/*.jinja, the winning handler and every shadowed one. Source scan, no model.
    if (Python3_Interpreter_FOUND)
        add_test(NAME test-chat-handler-overlap
                 COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/test-chat-handler-overlap.py ${PROJECT_SOURCE_DIR})
        set_tests_properties(test-chat-handler-overlap PROPERTIES LABELS chat)
    endif()
    # PEFT parity harness (row S69-ADAPTERS-PEFT-PARITY-HARNESS-2026-10-26-01): random-weight HF model + PEFT adapters, converted with
    # convert_hf_to_gguf.py / convert_lora_to_gguf.py, fork logits vs PEFT fp32 logits on the same tokens. The driver exits 77 (SKIPPED)
    # without torch + transformers + peft + sentencepiece.
    llama_build(test-adapter-peft-parity.cpp)
    if (Python3_Interpreter_FOUND)
        add_test(NAME test-adapter-peft-parity
                 COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/test-adapter-peft-parity.py --bin $<TARGET_FILE:test-adapter-peft-parity>)
        set_tests_properties(test-adapter-peft-parity PROPERTIES SKIP_RETURN_CODE 77 LABELS adapters)
    endif()
    # In-Place TTT (arXiv 2604.06169) parity: a tiny random-weight Qwen3 with TTT on layers 1 and 3 (chunk 8, lr 3), run through the OFFICIAL
    # ByteDance-Seed/In-Place-TTT modeling_qwen3.py UNMODIFIED (vendored, sha256-checked, tests/ttt-inplace-oracle/) as the oracle, converted with
    # this repo's converter and decoded by test-ttt-inplace three ways (prefill+greedy, token-by-token, n_ubatch 5) plus multi-sequence and
    # state save/load. max|dlogit| <= 1e-3 (F32, CPU). Exits 77 (SKIPPED) without torch/transformers/einops/opt_einsum/sentencepiece.
    # Row S69-FORK-INPLACE-TTT-2026-10-26-01.
    llama_build(test-ttt-inplace.cpp)
    # An arch without a TTT graph route refuses a GGUF that carries "<arch>.ttt.chunk" instead of ignoring it.
    # Row S69-FORK-INPLACE-TTT-OTHER-ARCHS-2026-10-26-01.
    llama_build_and_test(test-ttt-arch-guard.cpp)
    # ggml-cuda.h decls must use GGML_BACKEND_API (own DLL); row S69-GGML-CUDA-H-13-OF-14-DECLS-CARRY-THE-WRONG-EXPORT-MACRO-2026-09-15-01
    if (Python3_Interpreter_FOUND)
        add_test(NAME test-cuda-h-backend-api
                 COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/test-cuda-h-backend-api.py ${PROJECT_SOURCE_DIR})
    endif()
    if (Python3_Interpreter_FOUND)
        add_test(NAME test-ttt-inplace
                 COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/test-ttt-inplace.py --bin $<TARGET_FILE:test-ttt-inplace>)
        set_tests_properties(test-ttt-inplace PROPERTIES SKIP_RETURN_CODE 77 LABELS main)
    endif()
    # Host kernels for this fork FUSED_* ops, checked against the same maths written with
    # ordinary ggml ops. Added 2026-09-14 with those kernels: a model that stopped crashing
    # is not a model that computes correctly, and nothing else in this suite covers them.
    llama_build_and_test(test-fused-cpu-ops.cpp)
    # TEAL activation sparsity (S69-TEAL-STAGE1, arXiv:2408.14690): mask op vs scalar reference incl. the
    # |x| == tp boundary, the calibrator's exact-count contract, the --teal-* flag pair and the OFF defaults.
    # Model-free. The model-level gates (bit-identity flag off, every attn_in/ffn_in/ffn_mid family actually
    # masks, PPL vs dense) are tests/test-teal-sparsity-e2e.py.
    llama_build_and_test(test-teal-sparsity.cpp)
    # Logits probe. With no argument it runs its model-free CONTROLS and SKIPs (exit 0), so it is
    # safe in a suite on a machine with no GGUF; given a model path it answers the two questions
    # that "every generated token is identical" cannot distinguish between.
    llama_build_and_test(test-logits-sanity.cpp)
    # llama_set_output_vocab_subset (CTranslate2's vocabulary map): the restricted output projection equals the full one on the subset, -inf elsewhere.
    # Takes a model path as argv[1]; without one it SKIPs with exit 0 (same convention as test-logits-sanity).
    llama_build_and_test(test-output-vocab-subset.cpp)
    # MUL_MAT single-column vs wide-batch agreement. Model-free reproducer for the first
    # non-finite node found by the scan above.
    llama_build_and_test(test-mulmat-gemv.cpp)
    # CPU SUM over transposed/permuted views (nb[0] != element size): the kernels read ne00 consecutive
    # elements per row and summed the wrong memory. Model-free; carries contiguous and row-contiguous controls.
    llama_build_and_test(test-cpu-sum-strided.cpp)
    # SOFTPLUS: ggml_softplus_inplace builds the plain op (was SQRT_SOFTPLUS) and the shared scalar helper keeps
    # relative precision for x << 0 (was logf(1 + expf(x)), 0 below x ~ -17). Model-free; control = CPU softplus op.
    llama_build_and_test(test-softplus-numerics.cpp)
    # Out-of-bounds-write guards: ggml_acc/ggml_set offset >= 2^31 (narrowed to int32 op params) and get_rows_back
    # row outside dst. Each case runs in a child process; carries an abort-detection control and legal-input controls.
    llama_build_and_test(test-ggml-offset-row-guards.cpp)
    # Same binary, --isolation-check: runs the suite twice, the second time with an unrelated type inserted at the
    # front of the class sweep, and FAILS if any verdict present in both runs moved (per-test generators; row
    # S69-GEMV-Q6K-R4-RATIO-1080-AFTER-SWEEP-RNG-SHIFT-2026-10-17-01). Carries its own comparison control.
    llama_test(test-mulmat-gemv NAME test-mulmat-gemv-isolation ARGS --isolation-check)
    # Do a quant type's TWO decoders -- dequantize_row_<t> and the GEMM behind vec_dot -- agree
    # about the same bytes? test-mulmat-gemv can only report THAT they disagree; this says which
    # weights and by how much, by recovering the kernel's decoded weight exactly with a one-hot
    # activation (lossless in q8_K). Model-free.
    llama_build_and_test(test-iqk-decode-agreement.cpp)
    # Prism PQ2_0: is the KLD gap the kernel decode or the activation quantiser? Double-precision reference, no model. Added 2026-10-04.
    llama_build_and_test(test-pq2-activation-noise.cpp)
    # Prism ternary dot throughput against Q8_0 (prints a ratio, asserts none). Added 2026-10-04 for the scalar-dot row.
    llama_build_and_test(test-prism-dot-throughput.cpp)
    # MUL_MAT, quantized weight x F16/BF16 src1, host backend: claimed by supports_op AND numerically
    # right. The MLA prompt path builds exactly this node (deepseek-v2 was unloadable on CPU). Added 2026-09-28.
    llama_build_and_test(test-mul-mat-src1-convert.cpp)
    # ggml_conv_2d_direct (im2col + mul_mat emulation of upstream CONV_2D) must accumulate in F32: its MUL_MAT carries
    # GGML_PREC_F32, and convs whose outputs exceed the F16 range stay finite on every backend present. gemma3n's
    # MobileNetV5 vision tower returned inf on CUDA without it. Added 2026-10-20.
    llama_build_and_test(test-conv-2d-direct-prec.cpp)
    llama_build_and_test(test-split-points.cpp)
    # CPU ADD1 (a + scalar b) with an F16/BF16 scalar: the F32 kernel read the scalar as a float whatever its type.
    # Model-free; contiguous and padded-1d strided a, with F32-scalar and F16-a controls.
    llama_build_and_test(test-cpu-add1-scalar.cpp)
    llama_build_and_test(test-ollama-gemma4-tokenizer.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf ${CMAKE_CURRENT_BINARY_DIR})
    llama_build_and_test(test-ollama-util.cpp)
    target_include_directories(test-ollama-util PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    llama_build_and_test(test-ollama-store.cpp)
    target_include_directories(test-ollama-store PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    llama_build_and_test(test-ollama-gotmpl.cpp ARGS ${PROJECT_SOURCE_DIR}/tests/ollama-engine/gotmpl_oracle.json)
    target_include_directories(test-ollama-gotmpl PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    llama_build_and_test(test-ollama-jinja-corpus.cpp ARGS ${PROJECT_SOURCE_DIR}/tests/ollama-engine/jinja_corpus_oracle.json)
    # Inkling (S69 P4): a converted random-weight GGUF loads -- hparams, every per-layer tensor shape/type, vocab. With no
    # argument it SKIPs (exit 0); the fixture is made by tests/inkling-arch-check/make_tiny_inkling.py.
    llama_build_and_test(test-inkling-arch-load.cpp)
    # Inkling (S69 P1): the graph's logits match the reference implementation's on the same random weights, in one batch, one token
    # at a time, in chunks, with two sequences, and after a sequence is removed. SKIPs (exit 0) with no argument; the fixture is
    # made by tests/inkling-arch-check/make_tiny_inkling.py and ref_inkling_logits.py (see the header of the test).
    llama_build_and_test(test-inkling-parity.cpp)
    # Kimi-K3 refuses a GGUF without expert_latent_length (or with it / attn_res.block_size at 0), naming the key. SKIPs (exit 0)
    # with no argument; the metadata-only fixtures are made by tests/make_tiny_kimi_k3_meta.py.
    llama_build_and_test(test-kimi-k3-required-keys.cpp)
    # Repetition/frequency/presence penalties: the index-by-token-id fast path equals the per-candidate lookup on every layout (and stays in range).
    llama_build_and_test(test-penalty-index.cpp)
    # Same one-hot extraction against a BACKEND's matmul (default CUDA0), MMVQ (N=1) and wide
    # (N=K) paths separately, so an NMSE~1 test-backend-ops failure can be read per weight.
    llama_build(test-backend-onehot-decode.cpp)
    # GGML_OP_TURBO_WHT on a GPU backend, by properties (round trip, orthonormality, basis -> +-1/sqrt(g),
    # InnerQ scale). CPU declines the op as GPU-only, so test-backend-ops never covers it. SKIPs (exit 0)
    # when no GPU backend can be initialised, so it is safe in a CPU-only suite.
    llama_build_and_test(test-turbo-wht.cpp)
    # --offload-attn FA mask op: query head -> kv head must be the GROUPED mapping the attention kernel uses (was interleaved; wrong for every GQA model).
    # CPU only. The op source is compiled into the test so it needs no export from libllama. Fail-before / pass-after + must-fail partner: see the file banner.
    # Also covers the NON-FA op (mask_op, offload_attention_op.cpp) for the recent window, so both op sources are compiled in.
    # CUDA init failure diagnostic text (driver vs toolkit versions + PTX hint); header-only, CPU only
    # (row S69-ENGREV-JAN-CUDA-DRIVER-TOOLKIT-DIAG-2026-10-27-01).
    llama_build_and_test(test-cuda-driver-diag.cpp)
    llama_build_and_test(test-offload-mask-gqa.cpp ${PROJECT_SOURCE_DIR}/src/offload_attention_mask_op.cpp ${PROJECT_SOURCE_DIR}/src/offload_attention_op.cpp)
    # --offload-attn-dense-layers N | -N | list grammar (src/offload_attention_dense.h, header-only, model-free); row S69-ENGREV-HIPATTENTION-FLAG-SURFACE-2026-10-27-01
    llama_build_and_test(test-offload-dense-layers.cpp)
    # page-selection reuse across steps and layers (src/offload_select_share.h, header-only, model-free); row S69-KT-PICK-ATTN-SELECTION-REUSE-LAYER-TOKEN-STEP-2026-10-05-01
    llama_build_and_test(test-offload-select-share.cpp)
    # NVFP4 block-scale code search + two-level (per-tensor) scale: reference-vs-searched error, never-worse, weighted, scale
    # equivariance with a one-level must-fail partner (S69-A-PICK-NVFP4-TWO-LEVEL-SCALE, 2026-10-05). CPU only, no model file.
    add_executable(test-nvfp4-two-level test-nvfp4-two-level.cpp)
    target_link_libraries(test-nvfp4-two-level PRIVATE ggml)
    add_test(NAME test-nvfp4-two-level COMMAND test-nvfp4-two-level)
    # Recurrent ops (GLA, GATED_DELTA_NET, RWKV WKV6/WKV7) on the host backend vs scalar double references of the mathematical recurrence,
    # n_seqs 1/>1, n_tokens 1/>1, head size 16/64, state carried across two calls == one call. Rows S69-GGML-OP-1/2/3-*. CPU only, no model file.
    foreach(_rec_test test-gated-linear-attn test-gated-delta-net test-rwkv-wkv6 test-rwkv-wkv7)
        add_executable(${_rec_test} ${_rec_test}.cpp)
        target_link_libraries(${_rec_test} PRIVATE ggml)
        add_test(NAME ${_rec_test} COMMAND ${_rec_test})
    endforeach()
    # GATED_DELTA_NET on the host backend against a double-precision reference of the recurrence (scalar + KDA gate, K=1/3, SIMD tails),
    # with a must-fail partner; S69-EXECUTORCH-PICK-CPU-GDN-TWO-PASS-SWEEP-2026-10-05-01. GATE-CLAIM-OK: test-backend-ops compares a
    # backend with the host backend, so a build with only the host backend has nothing to compare against; this test carries its own reference.
    add_executable(test-gdn-cpu-ref test-gdn-cpu-ref.cpp)
    target_link_libraries(test-gdn-cpu-ref PRIVATE ggml)
    add_test(NAME test-gdn-cpu-ref COMMAND test-gdn-cpu-ref)
    # CPU flash attention exact trailing-mask KV clamp (mask_effective_nk1): every mask shape the sampled skip got wrong (#1910) plus
    # finite-negative tails, against a double-precision softmax reference. S69-EXECUTORCH-PICK-CPU-FA-EXACT-CAUSAL-KV-CLAMP-2026-10-05-01.
    add_executable(test-fa-causal-clamp test-fa-causal-clamp.cpp)
    target_link_libraries(test-fa-causal-clamp PRIVATE ggml)
    add_test(NAME test-fa-causal-clamp COMMAND test-fa-causal-clamp)
    # optional L2-derived CPU FA K tile (pure selector, header-only). S69-NCNN-PICK-L2-CACHE-DERIVED-FA-TILE-2026-10-05-01.
    add_executable(test-fa-kstep-l2 test-fa-kstep-l2.cpp)
    target_include_directories(test-fa-kstep-l2 PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
    add_test(NAME test-fa-kstep-l2 COMMAND test-fa-kstep-l2)
    # CPU flash attention skips interior all -inf mask tiles (XAttention kernel half): NaN V in a deselected block must not reach the output,
    # NaN V in a kept block must (control). S69-openvino-PICK-XATTENTION-PREFILL-BLOCK-SELECT-2026-09-29-01.
    add_executable(test-fa-xattn-skip test-fa-xattn-skip.cpp)
    target_link_libraries(test-fa-xattn-skip PRIVATE ggml)
    add_test(NAME test-fa-xattn-skip COMMAND test-fa-xattn-skip)
    # CPU flash attention with kv_rows (ggml_flash_attn_ext_rows) vs an explicit gather + plain flash_attn_ext and vs a double softmax, with a
    # wrong-row-index negative control. test-backend-ops runs 0 cases on CUDA0 (cuda declines kv_rows), so this is the executed check.
    # Task fork-farows-kvrows-cpu-selfcheck-test.
    add_executable(test-flash-attn-rows test-flash-attn-rows.cpp)
    target_link_libraries(test-flash-attn-rows PRIVATE ggml)
    add_test(NAME test-flash-attn-rows COMMAND test-flash-attn-rows)
    # CPU flash attention vs OpenVINO's CPU plugin ScaledDotProductAttention (opset13), NMSE on random Q/K/V, decode + causal prefill,
    # MHA + GQA, with a scale x1.25 control. OpenVINO's C API is loaded at runtime from $OV_C_LIB (exit 77 = skip when unset), so only
    # its headers are needed here. S69-A-PICK-INTEL-NPU-XPU-OPENVINO-PLUGIN-FEATURES-IPEXLLM-OPENVINO-CORE-2026-10-05-01.
    set(LLAMA_TEST_OPENVINO_INCLUDE "" CACHE PATH "directory holding openvino/c/*.h, for test-fa-ov-cross-check")
    if (LLAMA_TEST_OPENVINO_INCLUDE AND EXISTS "${LLAMA_TEST_OPENVINO_INCLUDE}/openvino/c/ov_core.h")
        add_executable(test-fa-ov-cross-check test-fa-ov-cross-check.cpp)
        target_include_directories(test-fa-ov-cross-check PRIVATE ${LLAMA_TEST_OPENVINO_INCLUDE})
        target_link_libraries(test-fa-ov-cross-check PRIVATE ggml ${CMAKE_DL_LIBS})
        add_test(NAME test-fa-ov-cross-check COMMAND test-fa-ov-cross-check)
        set_tests_properties(test-fa-ov-cross-check PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    # Hexagon HTP power-mode name mapping: header-only, SDK-free, so it runs on any host. The DSP-side vote
    # needs a Snapdragon (DEVICE-PENDING; see the test banner). Adreno OpenCL constant-register workaround: a clang
    # -emit-llvm check (skips with 77 when the portable clang is absent).
    add_executable(test-hexagon-power-mode test-hexagon-power-mode.cpp)
    add_test(NAME test-hexagon-power-mode COMMAND test-hexagon-power-mode)
    find_program(CLANG_OPENCL_CHECK_CLANG clang HINTS "${PROJECT_SOURCE_DIR}/../../portable/mingw64/bin")
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (CLANG_OPENCL_CHECK_CLANG AND Python3_Interpreter_FOUND)
        add_test(NAME check-opencl-adreno-constreg
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check-opencl-adreno-constreg.py
                         --clang ${CLANG_OPENCL_CHECK_CLANG} --kernel-dir ${PROJECT_SOURCE_DIR}/ggml/src/ggml-opencl/kernels)
        set_tests_properties(check-opencl-adreno-constreg PROPERTIES SKIP_RETURN_CODE 77)
    endif()
if (WIN32)
    add_executable(test-cpu-affinity-groups test-cpu-affinity-groups.cpp)
    target_link_libraries(test-cpu-affinity-groups PRIVATE ggml)
    add_test(NAME test-cpu-affinity-groups COMMAND test-cpu-affinity-groups)
endif()
    # TurboQuant KV WRITER (2026-09-29). test-turbo-write-host runs the SAME whole-group algorithm the CUDA kernels
    # run (ggml/src/ggml-cuda/turbo-write.h, compiled as plain C++) against quantize_row_turbo{2,3}_0_ref, with a
    # wrong-rotation negative control; CPU only, no backend is initialised. test-turbo-cpy is the CUDA half
    # (GGML_OP_CPY / GGML_OP_SET_ROWS F32 -> turbo2/turbo3 vs the same host reference): BUILD ONLY, deliberately not a
    # ctest entry, because it needs the GPU and is run by hand, alone.
    llama_build_and_test(test-turbo-write-host.cpp)
    # CPU turbo3 round trip with a must-fail control; bounds anchored on the textbook 3-bit Lloyd-Max distortion (0.0345), see the file banner.
    llama_build_and_test(test-turbo3-roundtrip.cpp)
    llama_build(test-turbo-cpy.cpp)
    # TurboQuant flash attention. `host` mode (the ctest entry) loads NO backend: it proves the rotation identity (rotate Q, attend in the rotated basis, un-rotate the
    # output == attention over the un-rotated K/V) on real quantised data and that the two ways to get it wrong are visible (controls). The device run
    # (`test-turbo-fa Vulkan0` / `CUDA0`) needs the GPU and is run by hand, alone.
    llama_build_and_test(test-turbo-fa.cpp ARGS host)
    # The same test on the CPU backend (`CPU` mode): the graph carries GGML_OP_TURBO_WHT around FLASH_ATTN_EXT, compared with f32 attention over the un-rotated K/V,
    # including head sizes that are not a multiple of 128 (K zero padded). test-turbo-graph-rot counts the rotations the graph builders add per attention call on a
    # host cache and on a (fake) device cache, with no-rotation / double-rotation controls (src/llama-turbo-graph.h).
    llama_test(test-turbo-fa NAME test-turbo-fa-cpu ARGS CPU)
    llama_build_and_test(test-turbo-graph-rot.cpp)
    # Metal TurboQuant KV code (ggml-metal/kernels/turbo.h) compiled as plain C++ behind a shim for the address-space keywords and compared with the
    # reference writer / reader, with must-fail controls; the Metal compiler and the kernels themselves are DEVICE-PENDING (see the file banner).
    llama_build_and_test(test-turbo-metal-host.cpp)
    # IQ2_BN AVX2 (no-VNNI) lookup-table prototype behind GGML_IK_IQ2BN_LUT (default OFF, so ctest
    # exercises the existing mul_mat_iq2bn_q8_K64 kernel here; the LUT path's own numeric/bench A/B
    # is run manually with the env var set -- see the file banner and
    # temp/research/findings/S69_ENGINE_THIREUS_2026-09-26.md). Model-free.
    llama_build_and_test(test-iq2bn-lut.cpp)
    llama_build_and_test(test-quant-roundtrip-distribution.cpp)
    llama_build_and_test(test-reasoning-budget.cpp  LABEL behavior)
    llama_build_and_test(test-grammar-parser.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-grammar-integration.cpp  LABEL behavior)
    llama_build_and_test(test-llama-grammar.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-batch-alloc.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-moe-pool-placement.cpp)
    # measured CPU/GPU expert-split planner (llama-moe-route-limit), pure host (2026-10-05)
    llama_build_and_test(test-moe-route-limit.cpp)
    # S69-KT-EXPERT-POOL-2026-09-28 unit 1: proves the cb_eval wrapper in llama-context.cpp
    # actually reaches the real ffn_moe_topk tensor and feeds llama_moe_recorder, which
    # test-moe-pool-placement.cpp (synthetic ids only) cannot. SKIPs exit-0 with no model arg,
    # same convention as test-logits-sanity.cpp -- registered here without a model path so CTest
    # stays green; run the built exe directly with a small MoE gguf to exercise it for real.
    llama_build_and_test(test-moe-pool-recorder-e2e.cpp)
    # ik #2644 MoE cache reachability (row S69-FORK-MOE-CACHE-WIRING-IK-2644-NOT-WIRED-2026-10-27-01): model-free part =
    # --moe-cache-mib parsing into llama_context_params. With -DLLAMA_TEST_MOE_GGUF=<moe.gguf> (e.g. ggml-org
    # stories15M_MOE-F16.gguf) a second test runs the context arm: CPU build = cache construction is reached and refuses
    # for want of a GPU; GPU build = cache-on logits match cache-off and the cache's hit/miss stats move.
    llama_build_and_test(test-moe-cache-wiring.cpp)
    if (LLAMA_TEST_MOE_GGUF)
        llama_test(test-moe-cache-wiring NAME test-moe-cache-wiring-model ARGS ${LLAMA_TEST_MOE_GGUF})
    endif()
    # BLOCK-GATED 2026-09-14 on LLAMA_BUILD_SERVER. test-chat CALLS INTO the server, it does
    # not merely link it -- measured: server_chat_convert_responses_to_chatcmpl and
    # oaicompat_chat_params_parse are undefined without tools/server.
    # Enabling LLAMA_BUILD_SERVER does NOT currently help: tools/server fails to compile
    # against this fork's common/ API (server-schema.cpp:112 'common_params_sampling' has no
    # 'typ_p'; :506 no 'samplers'; server-task.cpp:81 'common_params_speculative' has no
    # 'types'; server-context.cpp:378 llama_token -> llama_context* conversion). 3 TUs fail.
    # The server is UNMERGED, and that is tracked as its own port rather than papered over
    # here. Gating the test on the same option as the code it exercises means it returns
    # AUTOMATICALLY when that port lands.
    # CONSEQUENCE: with the option OFF this test is not built and its coverage is absent from
    # such a configuration. Not a regression -- it has never linked here.
    if (LLAMA_BUILD_SERVER)
        llama_build_and_test(test-chat.cpp  LABEL behavior  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        # ADDED 2026-09-13: tools/server/server-common.h:7 does `#include "mtmd.h"`, reached from
        # server-chat.h, which test-chat.cpp:10 includes -- so this test needs mtmd's directory even
        # though it is not an mtmd test. It died on `fatal error: mtmd.h: No such file or directory`.
        #
        # It is stated HERE rather than relied on from server-context, because in THIS configuration
        # server-context IS NOT A TARGET: `ninja -t targets all` lists 0 occurrences of it while
        # listing 33 for llama-bench and 2 for libmtmd. tools/server is not configured in this build,
        # so the target_link_libraries below degrades to a raw `-lserver-context` that carries no
        # usage requirements at all -- the same silent degradation that left tools/parser without
        # common/ on its include path. (server-context's own ../mtmd was also PRIVATE and relative,
        # both fixed in tools/server/CMakeLists.txt, which is correct but inert while that directory
        # is not built.)
        target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
        # GUARDED 2026-09-14: server-context exists only when LLAMA_BUILD_SERVER is ON, and this
        # build has it OFF while LLAMA_BUILD_TESTS is ON. Unguarded, CMake treats the unknown target
        # name as a SYSTEM LIBRARY and emits a bare `-lserver-context`, which fails at link with
        # "cannot find -lserver-context" -- the same silent degradation the comment above describes for
        # include paths. The includes test-chat needs are stated explicitly above for that reason, so
        # they survive whether or not the target exists.
        # LLAMA_BUILD_SERVER, not `if (TARGET server-context)`. MEASURED 2026-09-14: the root
        # CMakeLists adds tests (line 278) BEFORE tools (line 287), so at this point the
        # server-context target has not been created yet and a TARGET test never fires -- even
        # with the option ON, test-chat's LINK_LIBRARIES came out without it and no archive was
        # built. The option is known before any subdirectory is processed. Naming a target that
        # is created later is fine here: target_link_libraries resolves at GENERATE time, once
        # every directory has been read. Same class as the `if (TARGET ggml-cpu)` placement bug
        # fixed in ggml/src/CMakeLists.txt -- a conditional evaluated somewhere other than where
        # it was assumed to be.
        if (LLAMA_BUILD_SERVER)
            target_link_libraries(test-chat PRIVATE server-context)
        endif()
        # upstream: the server component test (needs server-context, so it shares test-chat's gate)
        llama_build_and_test(test-server-component.cpp  LABEL behavior)
        target_include_directories(test-server-component PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        target_include_directories(test-server-component PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
        target_link_libraries(test-server-component PRIVATE server-context)
    endif()
    # server_prompt_cache randomized invariant test (row S69-tgi-PICK-PROMPTCACHE-RANDTEST-2026-09-28-01). Same
    # LLAMA_BUILD_SERVER gate and include dirs as test-chat above, for the same reasons (server-common.h includes
    # mtmd.h; server-context is only a target when the server is configured).
    if (LLAMA_BUILD_SERVER)
        llama_build_and_test(test-server-prompt-cache.cpp)
        target_include_directories(test-server-prompt-cache PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        target_include_directories(test-server-prompt-cache PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
        target_link_libraries(test-server-prompt-cache PRIVATE server-context)
        # terminate() before start_loop() must not be lost (row S69-ENGREV-JAN-QUEUE-TERMINATE-BEFORE-LOOP-LOST-2026-10-27-01); needs no model.
        add_executable(test-server-queue-terminate ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_queue_terminate.cpp)
        target_include_directories(test-server-queue-terminate PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR}/tools/mtmd)
        target_link_libraries(test-server-queue-terminate PRIVATE server-context llama-common)
        add_test(NAME test-server-queue-terminate COMMAND $<TARGET_FILE:test-server-queue-terminate>)
        set_property(TEST test-server-queue-terminate PROPERTY LABELS main)
        # LoRA-configuration key of the prompt cache and radix cache (row S69-LORAX-PICK-LORA-CONFIG-IN-PROMPT-CACHE-AND-RADIX-KEY-2026-09-29-01).
        # Parts 1-2 need no model; parts 3-4 run when a tiny GGUF is given as argv[1] / env LORAKEY_TEST_MODEL.
        llama_build_and_test(test-server-lora-cache-key.cpp)
        target_include_directories(test-server-lora-cache-key PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        target_include_directories(test-server-lora-cache-key PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
        target_link_libraries(test-server-lora-cache-key PRIVATE server-context)
        target_compile_definitions(test-server-lora-cache-key PRIVATE LORAKEY_NEW_API)
        # Segmented-LRU eviction of the host prompt cache (row S69-VLLM-PICK-PROMPT-CACHE-SLRU-EVICTION-2026-09-29-01); needs no model.
        llama_build_and_test(test-server-prompt-cache-slru.cpp)
        target_include_directories(test-server-prompt-cache-slru PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        target_include_directories(test-server-prompt-cache-slru PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
        target_link_libraries(test-server-prompt-cache-slru PRIVATE server-context)
        # SGLang picks: request cache namespace, radix eviction policies, --sched-policy (row S69-ENGINE-MERGE-PROGRAMME-ALL-ENGINES-2026-09-09-01).
        # Needs no model except part 2b, which runs when a tiny GGUF (tests/make_tiny_llama_gguf.py) is given as argv[1] / env SGLANG_TEST_MODEL.
        llama_build_and_test(test-server-sglang-picks.cpp)
        target_include_directories(test-server-sglang-picks PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        target_include_directories(test-server-sglang-picks PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
        target_link_libraries(test-server-sglang-picks PRIVATE server-context)
        # Publishing a finished slot's generated tokens into the radix tree (row S69-MISTRALRS-PICK-PAGEDATTN-BLOCKPOOL-2026-09-28-01); part 2 takes a tiny GGUF as argv[1].
        llama_build_and_test(test-server-radix-gen-publish.cpp)
        target_include_directories(test-server-radix-gen-publish PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        target_link_libraries(test-server-radix-gen-publish PRIVATE server-context)
        # what a compiled-grammar cache would save per request: times json_schema_to_grammar + llama_sampler_init_grammar (build only, not a test).
        llama_build(probe-grammar-parse-cost.cpp)

        # Pinned (never evicted) prefixes of the host prompt cache (--warm-prefix-file, row S69-rtp-llm-PICK-STARTUP-WARMED-PINNED-PREFIX-CACHE-2026-09-29-01); needs no model.
        llama_build_and_test(test-server-prompt-cache-pin.cpp)
        target_include_directories(test-server-prompt-cache-pin PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        target_include_directories(test-server-prompt-cache-pin PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
        target_link_libraries(test-server-prompt-cache-pin PRIVATE server-context)
        # NVMe tier of the host prompt cache (--cache-disk-dir, row S69-TIERED-KV-DISK-TIER-SLICE1-2026-10-05-01); needs no model.
        llama_build_and_test(test-server-prompt-cache-disk.cpp)
        target_include_directories(test-server-prompt-cache-disk PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        target_include_directories(test-server-prompt-cache-disk PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
        target_link_libraries(test-server-prompt-cache-disk PRIVATE server-context)
        # /v1/systemone typed decision models (ik c069e89e1): request parsing + answer formatting; needs no model.
        # The served path runs in tests/test-server-systemone.py (tiny random-weight model, needs llama-server).
        llama_build_and_test(test-server-systemone.cpp)
        target_include_directories(test-server-systemone PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        target_include_directories(test-server-systemone PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
        target_link_libraries(test-server-systemone PRIVATE server-context)
        # built-in server tools: output budget (read_file/exec/get_info, glob/grep result counts) from the serving context or --tools-max-output; needs no model
        llama_build_and_test(test-server-tools-output-budget.cpp LABEL behavior)
        target_include_directories(test-server-tools-output-budget PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        target_include_directories(test-server-tools-output-budget PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
        target_link_libraries(test-server-tools-output-budget PRIVATE server-context)
        llama_build_and_test(test-server-tools-result-budget.cpp LABEL behavior)
        target_include_directories(test-server-tools-result-budget PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        target_include_directories(test-server-tools-result-budget PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
        target_link_libraries(test-server-tools-result-budget PRIVATE server-context)
        # router + model-swap budget (provider per call, reload at another context); needs no model
        llama_build_and_test(test-server-tools-budget-swap.cpp LABEL behavior)
        target_include_directories(test-server-tools-budget-swap PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        target_include_directories(test-server-tools-budget-swap PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
        target_link_libraries(test-server-tools-budget-swap PRIVATE server-context)
        # router loaded-set snapshot helpers (--models-snapshot, row S69-TORCHSERVE-PICK-ROUTER-LOADED-SET-SNAPSHOT-2026-10-18-01); needs no model.
        llama_build_and_test(test-server-models-snapshot.cpp)
        target_include_directories(test-server-models-snapshot PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        # the JSON layer survives a deeply nested request body: parse, copy, dump, main thread and worker thread
        # (row S69-ENGREV-LOCALAI2-REQUEST-BODY-JSON-DEPTH-CAP-PARSER-COPY-2026-10-28-01); needs no model.
        llama_build_and_test(test-json-deep-body.cpp)
        # router reload empty-preset guard decision (--reload-allow-empty, row S69-ENGREV-TFSERVING-RELOAD-EMPTY-GUARD-2026-10-27-01); needs no model.
        llama_build_and_test(test-server-models-reload-guard.cpp)
        target_include_directories(test-server-models-reload-guard PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        # --chat-template-kwargs-force merge (row S69-ENGREV-TABBYAPI-TEMPLATE-KWARGS-FORCE-2026-10-27-01); needs no model.
        llama_build_and_test(test-server-template-kwargs-force.cpp)
        target_include_directories(test-server-template-kwargs-force PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        # /health decode-failure latch (row S69-ENGREV-TGI-HEALTH-DECODE-FAIL-LATCH-2026-10-27-01); needs no model.
        llama_build_and_test(test-server-decode-fail-latch.cpp)
        target_include_directories(test-server-decode-fail-latch PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        # control-token literals neutralised inside message text: no silent truncation, placeholder block chosen by capacity
        # (row S69-NO-STATIC-LIMITS-CPP-SERVER-SWEEP-2026-11-07-01, D7); needs models/ggml-vocab-qwen2.gguf, hence the source root as cwd.
        llama_build_and_test(test-plain-literals-capacity.cpp  LABEL behavior  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        target_include_directories(test-plain-literals-capacity PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        target_include_directories(test-plain-literals-capacity PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
        target_link_libraries(test-plain-literals-capacity PRIVATE server-context)
        # SPM vocab: control-token literals neutralised, SPM dummy-prefix not duplicated. Row S69-ENGREV-BEELLAMA-INPUT-PROVENANCE-SPM-GAP-2026-10-27-01.
        llama_build_and_test(test-plain-literals-spm.cpp  LABEL behavior  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        target_include_directories(test-plain-literals-spm PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
        target_include_directories(test-plain-literals-spm PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
        target_link_libraries(test-plain-literals-spm PRIVATE server-context)
    endif()
    # TODO: disabled on loongarch64 because the ggml-ci node lacks Python 3.8
    if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
        llama_build_and_test(test-json-schema-to-grammar.cpp  LABEL behavior  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        target_include_directories(test-json-schema-to-grammar PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    endif()

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp LINK llama)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW

    llama_build_and_test(test-llama-archs.cpp)
    # OCP FP8 E4M3 (mainline 68d20b884/202c1a229): generate a random-weight llama GGUF, then test-fp8 requantizes it to F8_E4M3
    if (GGML_CPU)
        llama_build(test-fp8.cpp)
        set(FP8_MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models-fp8")
        file(MAKE_DIRECTORY "${FP8_MODEL_DIR}")
        llama_test(test-llama-archs NAME test-generate-fp8-models LABEL "main behavior backend"
            ARGS -a llama -s 1 -o "${FP8_MODEL_DIR}")
        set_tests_properties(test-generate-fp8-models PROPERTIES FIXTURES_SETUP fp8-models)
        llama_test(test-fp8 NAME test-fp8-requantize LABEL "main behavior backend"
            ARGS --models "${FP8_MODEL_DIR}")
        set_tests_properties(test-fp8-requantize PROPERTIES FIXTURES_REQUIRED fp8-models)
        # DISABLE-TEMPORARY: S69-DRIFT-1110-FP8-TYPE-AND-FP8-KV-CACHE-2026-11-10-01 (deferred-on-prereq, route back = port generic
        # scale-tensor loading). test-fp8 loads *.scale / *.input_scale sidecar tensors, which this fork's loader does not create
        # (no generic wq_s/output_s loading), so fixtures fail with "done_getting_tensors: wrong number of tensors"
        # (MEASURED 2026-11-10, builds/fp8-1110-test1.log). Enable when scaled-weight loading lands.
        set_tests_properties(test-fp8-requantize PROPERTIES DISABLED TRUE)
    endif()
    # batch vs one-token steps vs chunks vs state checkpoint, CPU only, every architecture (S69-A-PICK-PREFILL-VS-STEPWISE-DECODE-PARITY, 2026-10-05)
    llama_test(test-llama-archs NAME test-llama-archs-parity LABEL parity ARGS --parity)
    # granite deepstack_mapping_arr must default to -1 (upstream 64086f2b2); fails with n_embd_inp 512 != 256 without the fill
    llama_test(test-llama-archs NAME test-llama-archs-deepstack-mapping-default LABEL main ARGS --deepstack-mapping-default)
    # draft-tree verify in one batch (seq-id sets + depth positions) == each path alone, FA off/on, with a single-sequence must-fail control
    # (S69-F1-PICK-TREE-MASK-VERIFY-BATCH-API-2026-10-05-01)
    llama_test(test-llama-archs NAME test-llama-archs-tree-batch LABEL main ARGS --tree-batch --seed 42)
    # BitNet norm_in tensors: per-projection input RMS norms on qwen3 (S69-ENGREV-BITNET-NORMIN-2026-10-27-01)
    llama_test(test-llama-archs NAME test-llama-archs-norm-in LABEL main ARGS --norm-in --seed 42)

    set(MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models/")
    file(MAKE_DIRECTORY "${MODEL_DIR}")

    llama_test(
        test-llama-archs
        NAME test-generate-models
        LABEL "main behavior backend"
        ARGS -o "${MODEL_DIR}"
    )
    set_tests_properties(test-generate-models PROPERTIES
        FIXTURES_SETUP generate-models
    )

    # Target model for test-speculative-rejection-sampling
    set(SPEC_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-speculative-models/target")
    file(MAKE_DIRECTORY "${SPEC_TARGET_DIR}")
    llama_test(
        test-llama-archs
        NAME test-generate-speculative-target
        LABEL main
        ARGS -a llama -s 999 -o "${SPEC_TARGET_DIR}"
    )
    set_tests_properties(
        test-generate-speculative-target
        PROPERTIES FIXTURES_SETUP speculative-model
    )

    # To avoid recreating the model use the fixture exclude any option:
    # -FA "^speculative-model$"
    llama_build_and_test(test-speculative-rejection-sampling.cpp ARGS
        -m "${SPEC_TARGET_DIR}/llama-dense.gguf")
    set_tests_properties(test-speculative-rejection-sampling PROPERTIES FIXTURES_REQUIRED speculative-model)

    # Test context state handling across all architectures, using the generated dummy models
    llama_test(
        test-llama-context
        LABEL "main behavior backend"
        ARGS --models "${MODEL_DIR}"
    )
    # abort on graph reallocations that the initial reserve should have covered
    set_tests_properties(test-llama-context PROPERTIES
        FIXTURES_REQUIRED generate-models
        ENVIRONMENT "GGML_SCHED_DEBUG_REALLOC=1"
    )

    llama_test(
        test-recurrent-state-rollback
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen35-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback PROPERTIES
        FIXTURES_REQUIRED generate-models
    )

    # The n_rs_seq clamp for legacy-graph hybrids (src/llama-context.cpp, "runs on the legacy graph path") is only exercised by a
    # model that is on llm_arch_supports_rs_rollback() AND on the legacy graph. MEASURED 2026-10-07 (both ways, MinGW, models from
    # test-llama-archs): clamp present -> rc 0 on qwen35 / qwen35moe / lfm2 / lfm2moe; clamp branch never taken -> rc 1 on all four
    # ("second pending partial seq_rm must be refused FAILED"). Only qwen35-dense was registered, so one arch carried the guard.
    foreach(rr_arch IN ITEMS qwen35moe-moe lfm2-dense lfm2moe-moe)
        llama_test(
            test-recurrent-state-rollback
            NAME test-recurrent-state-rollback-${rr_arch}
            LABEL main
            ARGS -m "${MODEL_DIR}/${rr_arch}.gguf"
        )
        set_tests_properties(test-recurrent-state-rollback-${rr_arch} PROPERTIES
            FIXTURES_REQUIRED generate-models
        )
    endforeach()

    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-nemotron-h
        LABEL main
        ARGS -m "${MODEL_DIR}/nemotron_h-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-nemotron-h PROPERTIES
        FIXTURES_REQUIRED generate-models
    )

    # A T-token verify window and a rollback to n accepted tokens must leave the same state as T / n single-token decodes (Strata S2 pattern, 2026-10-07).
    foreach(wp_arch IN ITEMS llama-dense qwen35-dense qwen3next-moe lfm2-dense nemotron_h-dense mamba2-dense jamba-dense)
        llama_test(
            test-spec-window-parity
            NAME test-spec-window-parity-${wp_arch}
            LABEL main
            ARGS -m "${MODEL_DIR}/${wp_arch}.gguf"
        )
        set_tests_properties(test-spec-window-parity-${wp_arch} PROPERTIES
            FIXTURES_REQUIRED generate-models
        )
    endforeach()
    # Same seed + temperature: sample-and-match speculative decoding must emit the plain-decoding tokens (row S69-STRATA-S2-DIST-SAMPLER-POSITION-KEYED-RNG-2026-10-07-01).
    # mamba2 / qwen3next keep n_rs_seq 0 (legacy graph), refuse a partial rollback, and are not covered
    foreach(sp_arch IN ITEMS llama-dense nemotron_h-dense)
        llama_test(
            test-spec-seed-parity
            NAME test-spec-seed-parity-${sp_arch}
            LABEL main
            ARGS -m "${MODEL_DIR}/${sp_arch}.gguf"
        )
        set_tests_properties(test-spec-seed-parity-${sp_arch} PROPERTIES
            FIXTURES_REQUIRED generate-models
        )
    endforeach()
    # the three verify functions stop at an EOG inside the draft (row S69-LLAMACPP-PICK-DRAFT-STOP-AT-EOG-2026-10-18-01).
    # It needs a vocab with an EOS/EOT token, which the random-weight test-llama-archs models lack (MEASURED 2026-10-18:
    # "model has no EOS/EOT token" on llama-dense), so it runs on stories15M q4_0, the file the server tests download
    # into tools/server/tests/tmp. Absent file = skipped (77).
    llama_test(
        test-spec-draft-eog
        NAME test-spec-draft-eog-stories15m
        LABEL main
        ARGS -m "${PROJECT_SOURCE_DIR}/tools/server/tests/tmp/stories15M-q4_0.gguf" -t 8
    )
    set_tests_properties(test-spec-draft-eog-stories15m PROPERTIES
        SKIP_RETURN_CODE 77
    )
    llama_test(
        test-spec-proposal-accept
        NAME test-spec-proposal-accept-stories15m
        LABEL main
        ARGS -m "${PROJECT_SOURCE_DIR}/tools/server/tests/tmp/stories15M-q4_0.gguf" -t 8
    )
    set_tests_properties(test-spec-proposal-accept-stories15m PROPERTIES
        SKIP_RETURN_CODE 77
    )
    llama_test(
        test-spec-draft-sample-equiv
        NAME test-spec-draft-sample-equiv-stories15m
        LABEL main
        ARGS -m "${PROJECT_SOURCE_DIR}/tools/server/tests/tmp/stories15M-q4_0.gguf" -t 8
    )
    set_tests_properties(test-spec-draft-sample-equiv-stories15m PROPERTIES
        SKIP_RETURN_CODE 77
    )

    # --spec-async (draft || verify overlap): lossless vs serial + plain greedy, lookahead hit path, rollback of the draft context
    # (row S69-ASYM-SPEC-ASYNC-DRAFT-VERIFY-OVERLAP-PEARL-AMUSD-2026-11-06-01). Skips (77) when the model file is absent; best run on an f32
    # tiny model (stories260K-f32.gguf) so the Q4_0 requantised draft differs from the target.
    llama_build(test-spec-async.cpp)
    llama_test(
        test-spec-async
        NAME test-spec-async-stories15m
        LABEL main
        ARGS "${PROJECT_SOURCE_DIR}/tools/server/tests/tmp/stories15M-q4_0.gguf"
    )
    set_tests_properties(test-spec-async-stories15m PROPERTIES
        SKIP_RETURN_CODE 77
    )
    # self-speculative decoding with a draft_rows gated LoRA: the draft pass rows carry the adapter, the verify rows are gate 0
    # (row S69-ADAPTERS-POSITION-GATED-LORA-UNO-SPEC-2026-10-26-01). Synthetic adapter written by the test. Skips (77) without the model.
    llama_build(test-spec-gated-lora.cpp)
    llama_test(
        test-spec-gated-lora
        NAME test-spec-gated-lora-stories15m
        LABEL main
        ARGS "${PROJECT_SOURCE_DIR}/tools/server/tests/tmp/stories15M-q4_0.gguf"
    )
    set_tests_properties(test-spec-gated-lora-stories15m PROPERTIES
        SKIP_RETURN_CODE 77
    )

    llama_build(test-fusion.cpp)
endif()

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp  LABEL behavior)
llama_build_and_test(test-jinja.cpp  LABEL behavior)
llama_build_and_test(test-chat-handler-loadlog.cpp LABEL behavior)
llama_build_and_test(test-model-checksum.cpp LABEL behavior)
llama_build_and_test(test-xgrammar-header.cpp LABEL behavior)
llama_test(test-jinja  NAME test-jinja-py  ARGS -py  LABEL "python behavior")
# mainline 68d20b884: FP8 conversion unit tests (needs torch + numpy; skipped when no interpreter is found)
find_program(LLAMA_FP8_PYTHON NAMES python3 python)
if (LLAMA_FP8_PYTHON)
    llama_test_cmd(${LLAMA_FP8_PYTHON}
        NAME test-conversion-fp8
        LABEL "python behavior"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        ARGS -m unittest discover -s tests -p test_conversion_fp8.py -v
    )
endif()
# test-jinja-py renders through real Jinja2 in a child Python (test-jinja.cpp, LLAMA_TEST_PYTHON, else python.exe /
# python3 off PATH). On a host with several Pythons the PATH one need not have jinja2, and every -py case then fails
# with ModuleNotFoundError (rows S69-CTEST-ENV-CA-BUNDLE-AND-JINJA2-FOR-MINGW-BUILDS-2026-10-05-01,
# S69-FORK-CTEST-HOST-CONFIG-TLS-PYTHON-2026-10-17-01). Pick the interpreter at configure time: $ENV{LLAMA_TEST_PYTHON}
# if set, else the Python cmake found, and probe `import jinja2`. A passing probe pins that interpreter on the test;
# a failing one registers the test DISABLED, so ctest reports it Not Run (Disabled) instead of a red failure that
# is about the host, not the code (DISABLED: skipped, status Not Run, not counted in the total --
# https://cmake.org/cmake/help/latest/prop_test/DISABLED.html).
find_package(Python3 COMPONENTS Interpreter QUIET)
set(_llama_jinja_py "")
if (DEFINED ENV{LLAMA_TEST_PYTHON} AND NOT "$ENV{LLAMA_TEST_PYTHON}" STREQUAL "")
    set(_llama_jinja_py "$ENV{LLAMA_TEST_PYTHON}")
elseif (Python3_Interpreter_FOUND)
    set(_llama_jinja_py "${Python3_EXECUTABLE}")
endif()
set(_llama_jinja_ok FALSE)
if (NOT _llama_jinja_py STREQUAL "")
    execute_process(COMMAND ${_llama_jinja_py} -c "import jinja2"
                    RESULT_VARIABLE _llama_jinja_rc OUTPUT_QUIET ERROR_QUIET)
    if (_llama_jinja_rc EQUAL 0)
        set(_llama_jinja_ok TRUE)
    endif()
endif()
if (_llama_jinja_ok)
    set_tests_properties(test-jinja-py PROPERTIES ENVIRONMENT "LLAMA_TEST_PYTHON=${_llama_jinja_py}")
    message(STATUS "test-jinja-py: Jinja2 interpreter ${_llama_jinja_py}")
else()
    set_tests_properties(test-jinja-py PROPERTIES DISABLED TRUE)
    message(STATUS "test-jinja-py: DISABLED, no interpreter with jinja2 (tried '${_llama_jinja_py}'; set LLAMA_TEST_PYTHON)")
endif()
llama_build_and_test(test-json-schema.cpp  LABEL behavior)
llama_build_and_test(test-chat-auto-parser.cpp  LABEL behavior  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp  LABEL behavior)
# Gemma 4 template render parity against real Jinja2 (tests/gemma4-golden/, regenerate with gen_gemma4_golden.py); row S69-OLLAMA-PICK-GEMMA4-RENDERER-GOLDEN-VECTORS-2026-09-29-01
llama_build_and_test(test-chat-gemma4-golden.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# ---------------------------------------------------------------------------------------------
# ik_llama fork tests. Upstream's list is the base here -- it registers 44 tests to this fork's
# 12 -- so only what upstream has no equivalent for is added back. Each of the four below was
# verified present on disk and named zero times in upstream's CMakeLists before being written.
# ---------------------------------------------------------------------------------------------

llama_build_and_test(test-json-partial.cpp)
llama_build_and_test(test-regex-partial.cpp)
# ik afdbc5e13 (#2565) + c75ad6b6b (#2588): CPU-affinity parsers; merged in the ik-main trial (w11-1017)
llama_build_and_test(test-cpu-affinity.cpp)

# ik_llama #2332: IQ4_KS/IQ4_KT decode (Vulkan port validation); merged 2026-09-29 (ik-main slice 1)
llama_build_and_test(test-iq4-ks-kt-decode.cpp)
target_include_directories(test-iq4-ks-kt-decode PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
# Unsloth IQ1_XS/XXS/XXXS (enum ids 64-66): layout, encode, independent decode (+ negative control), vec_dot agreement,
# CPU MUL_MAT / MUL_MAT_ID / FUSED_UP_GATE / MOE_UP_GATE. Model-free.
llama_build_and_test(test-iq1-narrow-quants.cpp)
target_include_directories(test-iq1-narrow-quants PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
# OCP MX block-scaled MXFP6 (E2M3/E3M2) and MXFP8 (E4M3/E5M2), enum ids 70-73: exhaustive code tables, ties-to-even,
# E8M0 scale extremes, round trip, vec_dot vs dequant, block boundaries, CPU MUL_MAT. Model-free.
llama_build_and_test(test-mx-quant.cpp)
target_include_directories(test-mx-quant PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
# TQ3_1S / TQ4_1S (TheTom turboquant WHT-rotated Lloyd-Max weights, ids 74/75): byte-exact vs their reference quantiser,
# dequant bit-exact, NMSE, chunk, vec_dot vs dequant, CPU MUL_MAT / GET_ROWS. Model-free; looks the types up by name.
llama_build_and_test(test-tq1s-quant.cpp)
# Deterministic byte dump of the narrow types for a diff against the Unsloth reference build (build-only, not a ctest).
llama_build(probe-iq1-narrow-golden-dump.cpp)
target_compile_definitions(probe-iq1-narrow-golden-dump PRIVATE IQ1_DUMP_FORK)
# Weight-level quantisation-type head-to-head on real GGUF tensors with one imatrix (build-only probe).
llama_build(probe-quant-h2h-weights.cpp)
# MUL_MAT throughput by quantisation type through the real CPU backend (build-only probe).
llama_build(probe-quant-mulmat-speed.cpp)
# Gumbel watermark: 288 PRF known answers from vLLM's own reference + detection both ways (common/watermark.cpp).
llama_build_and_test(test-watermark.cpp)

# Deterministic FLASH_ATTN_EXT probe. It builds one fixed q/k/v/mask graph, computes it on a
# chosen backend and prints an FNV-1a-64 hash, so two backends can be compared bit-for-bit.
# Deliberately not a ctest: it takes a backend name and an output path, and the CUDA arm needs a
# device. Invoke it explicitly:  test-flash-attn-generic-hash cpu  /  ... cuda out.bin
# Links ggml only -- it touches no llama API.
add_executable(test-flash-attn-generic-hash test-flash-attn-generic-hash.cpp)
target_link_libraries(test-flash-attn-generic-hash PRIVATE ggml)

# Counts the CPU-fallback ops in a reconstruction of the parakeet local-attention pad chain.
# Needs only ggml. Also kept out of the default ctest run: it requires a CUDA device.
if (GGML_CUDA)
    add_executable(test-pad-sched-parakeet test-pad-sched-parakeet.cpp)
    target_link_libraries(test-pad-sched-parakeet PRIVATE ggml)

    # CUDA graph reuse when two graph shapes alternate in one meta buffer (the speculative decode
    # pattern). Needs a GPU; exit 77 = skip. GGML_CUDA_GRAPH_LEGACY=1 is its fail-before arm.
    add_executable(test-cuda-graph-alternating test-cuda-graph-alternating.cpp)
    target_link_libraries(test-cuda-graph-alternating PRIVATE ggml)

    # 1D TMA (cp.async.bulk + mbarrier) loader vs cp.async reference, byte-exact, with a shifted-source
    # negative control. Needs a GPU (exit 77 = skip); exit 3 = the barrier never completed. Built from
    # the ggml-cuda headers directly, so it carries the same CMake-only defines as ggml-cuda.
    # enable_language is directory-scoped: ggml-cuda's call does not reach tests/ (measured: without
    # it, generate fails with "Missing variable CMAKE_CUDA_COMPILE_OBJECT").
    enable_language(CUDA)
    add_executable(test-cuda-tma-bulk test-cuda-tma-bulk.cu)
    target_include_directories(test-cuda-tma-bulk PRIVATE ../ggml/src ../ggml/src/ggml-cuda ../ggml/include)
    target_compile_definitions(test-cuda-tma-bulk PRIVATE
        GGML_CUDA_MIN_BATCH_OFFLOAD=${GGML_CUDA_MIN_BATCH_OFFLOAD} GGML_CUDA_FUSION=1)
    target_link_libraries(test-cuda-tma-bulk PRIVATE ggml)

    # Native FP4 prefill GEMM for NVFP4 weights (GGML_CUDA_NVFP4_PREFILL=1) vs a host reference. Needs a Blackwell GPU;
    # exit 77 = skip. GGML_CUDA_NVFP4_PREFILL_MIN_M=100000000 is its fail-before arm.
    add_executable(test-cuda-nvfp4-prefill test-cuda-nvfp4-prefill.cpp)
    target_link_libraries(test-cuda-nvfp4-prefill PRIVATE ggml)

    # cuBLASLt must stay a DELAY-loaded import (it is 455 MB; an eager import cost +215 MiB startup commit, row
    # S69-B1019F-STARTUP-COMMIT-TRANSIENT-AND-FIT-CHECK4-REACH-2026-10-26-01). No GPU needed: the loader decides before main.
    # exit 0 pass, 1 eager import, 2 control failed, 77 skip (not Windows).
    if (WIN32)
        add_executable(test-cuda-cublaslt-lazy test-cuda-cublaslt-lazy.cpp)
        target_link_libraries(test-cuda-cublaslt-lazy PRIVATE ggml)
        add_test(NAME test-cuda-cublaslt-lazy COMMAND $<TARGET_FILE:test-cuda-cublaslt-lazy>)
        set_property(TEST test-cuda-cublaslt-lazy PROPERTY SKIP_RETURN_CODE 77)
    endif()
endif()

# S69-CHAT-TEMPLATES-MONOLITH-LACKS-14-UPSTREAM-2026-09-27-01: before/after parity probe for the
# named-template C API (llama_chat_apply_template). Links ONLY `llama` -- it is not a jinja test,
# it exercises the compiled llm_chat_* / LLM_CHAT_TEMPLATES path directly. Not registered as a
# ctest (no expected-output baked in yet); run manually and diff the two dumps.
add_executable(test-chat-template-parity test-chat-template-parity.cpp)
target_link_libraries(test-chat-template-parity PRIVATE llama)

# debug tool for chat template differential analysis (not registered as a test, run it manually)
llama_build(test-chat-analysis.cpp)
llama_build_and_test(test-log.cpp  LABEL behavior)
llama_build_and_test(test-log-drain-at-exit.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
    LABEL behavior
)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

# CA bundle for the fixture download. A portable cmake (mingw) has no CA store: file(DOWNLOAD) fails with
# status 60 and test-thread-safety, test-save-load-state, test-state-restore-fragmented and every other
# test-download-model consumer is Not Run (rows S69-CTEST-ENV-CA-BUNDLE-AND-JINJA2-FOR-MINGW-BUILDS-2026-10-05-01,
# S69-FORK-CTEST-HOST-CONFIG-TLS-PYTHON-2026-10-17-01; measured: cmake honours SSL_CERT_FILE / TLS_CAINFO, not
# CURL_CA_BUNDLE). Discovery order: the LLAMA_TEST_CA_BUNDLE cache entry, then $ENV{SSL_CERT_FILE} at configure
# time, then the certifi bundle of the Python cmake found. None found = cmake's own default, as before.
set(LLAMA_TEST_CA_BUNDLE "" CACHE FILEPATH "CA bundle for test fixture downloads (empty = discover)")
set(_llama_test_ca "${LLAMA_TEST_CA_BUNDLE}")
if (_llama_test_ca STREQUAL "" AND DEFINED ENV{SSL_CERT_FILE} AND EXISTS "$ENV{SSL_CERT_FILE}")
    set(_llama_test_ca "$ENV{SSL_CERT_FILE}")
endif()
if (_llama_test_ca STREQUAL "")
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_Interpreter_FOUND)
        # certifi, else the copy pip vendors (pip._vendor.certifi ships with every pip, so a Python without
        # certifi installed still has a bundle).
        foreach(_llama_certifi_mod certifi pip._vendor.certifi)
            if (_llama_test_ca STREQUAL "")
                execute_process(COMMAND ${Python3_EXECUTABLE} -c "import ${_llama_certifi_mod} as c; print(c.where())"
                                OUTPUT_VARIABLE _llama_certifi OUTPUT_STRIP_TRAILING_WHITESPACE
                                RESULT_VARIABLE _llama_certifi_rc ERROR_QUIET)
                if (_llama_certifi_rc EQUAL 0 AND EXISTS "${_llama_certifi}")
                    set(_llama_test_ca "${_llama_certifi}")
                endif()
            endif()
        endforeach()
    endif()
endif()
if (_llama_test_ca STREQUAL "")
    message(STATUS "test-download-model: no CA bundle found; the download uses cmake's default CA store")
else()
    message(STATUS "test-download-model: CA bundle ${_llama_test_ca}")
endif()

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -DCA_BUNDLE=${_llama_test_ca}
    -P ${PROJECT_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

# the context state cases against a real model, the sweep above uses the generated dummy ones
llama_test(test-llama-context  NAME test-llama-context-model  LABEL "model behavior backend"  ARGS -m "${MODEL_DEST}")
set_tests_properties(test-llama-context-model PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-thread-safety.cpp  LABEL "main backend"  ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp  LABEL behavior)
# --moe-cache-mib auto: the cache-size rule of the fit, without a GPU
llama_build_and_test(test-fit-moe-cache-auto.cpp  LABEL behavior)
# llguidance byte table of a vocab, sized by the vocab (row S69-NO-STATIC-LIMITS-CPP-SERVER-SWEEP-2026-11-07-01, D2); needs models/ggml-vocab-qwen2.gguf
llama_build_and_test(test-vocab-token-table.cpp  LABEL behavior  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
# GLM <think>/<tool_call>/<arg_key>/<arg_value> added tokens survive tokenize -> detokenize (row S69-ENGREV-TRNRT-CHECK-GLM-ADDED-TOKENS-DETOKENIZE-2026-11-13-01); needs models/ggml-vocab-glm4.gguf
llama_build_and_test(test-glm-added-tokens-detok.cpp  LABEL behavior  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
# Mistral [PREFIX]/[SUFFIX]/[MIDDLE] are the vocab's FIM tokens (row S69-ENGINE-MERGE-MISTRAL-COMMON-2026-11-06-01); needs models/ggml-vocab-qwen2.gguf
llama_build_and_test(test-vocab-mistral-fim.cpp  LABEL behavior  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
# --cpu-mask length: digits past GGML_MAX_N_THREADS/4 (row S69-NO-STATIC-LIMITS-CPP-SERVER-SWEEP-2026-11-07-01, D3)
llama_build_and_test(test-cpu-mask-length.cpp  LABEL behavior)
# fit: tensor-name pattern per layer has no layer-count cap (row S69-NO-STATIC-LIMITS-CPP-SERVER-SWEEP-2026-11-07-01, D1)
llama_build_and_test(test-fit-overflow-pattern.cpp  LABEL behavior)
# fit: -ngl 0 on 2+ devices divided by zero in the context reduction (row S69-ENGREV-ACCELERATE-FIT-NGL0-DIV-BY-ZERO-MULTIGPU-2026-10-27-01)
llama_build_and_test(test-fit-ngl0.cpp  LABEL behavior)
# fit: the post-fit receipt block + host-resident WARN (row S69-ENGREV-ACCELERATE-FIT-RESULT-RECEIPT-2026-10-27-01)
llama_build_and_test(test-fit-receipt.cpp  LABEL behavior)
# grammar samplers fail closed on a matcher error (row S69-ENGREV-APHRODITE-LLG-FAILCLOSED-2026-10-27-01)
llama_build_and_test(test-grammar-failclosed.cpp  LABEL behavior)
# grammar mask computed on a worker thread overlapping the next decode (row S69-EXLLAMAV2-PICK-GRAMMAR-MASK-OVERLAP-MEASURE-2026-10-20-01)
llama_build_and_test(test-grammar-mask-overlap.cpp  LABEL behavior)
target_include_directories(test-grammar-mask-overlap PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
# NVFP4 encoder scale search with the amax/4 candidate (row S69-ENGREV-APHRODITE-NVFP4-AMAX4-CANDIDATE-2026-10-27-01)
llama_build_and_test(test-nvfp4-amax4.cpp  LABEL behavior)
# --cpu-placement physical: CPU-list builder (synthetic + live topology) and the -C precedence / draft wiring.
llama_build_and_test(test-cpu-placement.cpp)
llama_build_and_test(test-model-resolution.cpp  LABEL behavior)
# the test serves its repos from an httplib server, and the library links it privately
target_link_libraries(test-model-resolution PRIVATE cpp-httplib)
# --hf-revision pin (branch / tag / commit) on a loopback HF API; no real download runs
llama_build_and_test(test-hf-revision.cpp  LABEL behavior)
target_link_libraries(test-hf-revision PRIVATE cpp-httplib)
# a finished download must match the announced length / SHA-256 before it is installed (loopback httplib server)
llama_build_and_test(test-lora-fit-reserve.cpp)
llama_build_and_test(test-hf-tree-pagination.cpp)
target_link_libraries(test-hf-tree-pagination PRIVATE cpp-httplib)
llama_build_and_test(test-hf-token-file.cpp)
llama_build_and_test(test-hf-finalize-hardlink.cpp)
llama_build_and_test(test-download-integrity.cpp)
target_link_libraries(test-download-integrity PRIVATE cpp-httplib vendor::hash)
# /tokenize with_offsets: header-only code-point spans of tokens (no model)
llama_build_and_test(test-server-token-offsets.cpp)
target_include_directories(test-server-token-offsets PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
# parallel range download + resume bitmap (planner, bitmap, loopback server)
llama_build_and_test(test-download-chunks.cpp)
target_link_libraries(test-download-chunks PRIVATE cpp-httplib)
# remote media fetch SSRF policy: private / loopback / link-local hosts are refused (literal checks only, no network)
llama_build_and_test(test-http-private-host.cpp  LABEL behavior)
target_link_libraries(test-http-private-host PRIVATE cpp-httplib)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp  LABEL "main behavior backend"  LINK ggml)
endif()
if (GGML_METAL AND NOT GGML_BACKEND_DL)
    # calls the Metal backend's graph-rewrite pass and its fusion table directly
    llama_build_and_test(test-metal-graph-optimize.cpp  LABEL "main behavior backend"  LINK ggml ggml-metal)
    target_include_directories(test-metal-graph-optimize PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-metal)
endif()
# ---------------------------------------------------------------------------------------------
# A ggml-ONLY probe, deliberately not built through llama_build_and_test.
#
# WHY IT EXISTS. test-opt was the only instrument the autodiff work had, and it links `common`,
# which does not currently compile on the mingw toolchain in this tree (measured 2026-09-10:
# 434 error lines, roughly ten distinct, in common/reasoning-budget.h, common/sampling.h and
# common/common.h -- llama_tokens undeclared, lr_opt incomplete, cpu_get_num_math against
# common_cpu_get_num_math, and more). So `ninja test-opt` cannot produce a binary here at all,
# and the autodiff port had no runnable test. That breakage has its own row; this target exists
# so a broken `common` cannot also block measuring ggml itself.
#
# It links ONLY ggml. Hand-linking the archives instead requires
# -Wl,--allow-multiple-definition, because ggml-base.a and ggml-cpu.a define 44 symbols in
# common, and a hand-linked binary returned 0 from ggml_sum(ggml_add(4,3)) -- an artifact of the
# link, not a fact about ggml. Letting CMake link it removes that entire class of doubt, which is
# the point: a probe is only worth running when its own build is above suspicion.
#
# NOW REGISTERED WITH CTEST, and the comment that stood here said the opposite: "it prints numbers
# for a human and always exits 0, so as a ctest it would pass unconditionally, which is worse than
# having no test." That was true, and it was the problem. This tree spent days in a state where
# every graph on every backend computed nothing and returned GGML_STATUS_SUCCESS, because
# ggml_visit_parents no longer set GGML_TENSOR_FLAG_COMPUTE while a dozen backends still tested it.
# A test that always exits 0 would have stayed green through all of it. The probe now asserts
# ggml_add(4,3) == 7 and that ggml_build_forward_expand sets that flag, and exits non-zero if
# either fails.
# Not llama_build() either: that links `llama llama-common` (tests/CMakeLists.txt:13), so it would
# drag in the same broken `common`. A bare add_executable against ggml is the whole point.
add_executable(test-ggml-autodiff-probe test-ggml-autodiff-probe.c)
target_link_libraries(test-ggml-autodiff-probe PRIVATE ggml)
add_test(NAME test-ggml-autodiff-probe
         WORKING_DIRECTORY .
         COMMAND $<TARGET_FILE:test-ggml-autodiff-probe>)
set_property(TEST test-ggml-autodiff-probe PROPERTY LABELS main)

# ---------------------------------------------------------------------------------------------
# IQ*_K BYTE-EXACT GOLDEN VECTORS -- a ggml-ONLY target, deliberately not llama_build_and_test.
#
# llama_build (tests/CMakeLists.txt:13) links `llama llama-common`, and `common` currently fails to
# compile on the mingw toolchain in this tree (434 error lines, ~10 distinct, measured 2026-09-10,
# tracked in its own row). Chaining this harness to that would leave the oracle the IQ*_K port
# depends on producing no binary. test-iqk-golden.cpp includes ONLY ggml.h and ggml-cpu.h and
# references no llama or common symbol, so linking `ggml` is not a workaround -- it is what the file
# needs. `ggml` carries ggml-base (ggml/src/CMakeLists.txt:456) and the CPU backend (:529), both
# PUBLIC, so the type-traits tables this test reads come with it.
#
# Same pattern and same reason as test-ggml-autodiff-probe above.
add_executable(test-iqk-golden test-iqk-golden.cpp)
target_link_libraries(test-iqk-golden PRIVATE ggml)
# WORKING_DIRECTORY is the SOURCE tests/ dir, not the default `.`: the golden path below is relative
# to it, so a test invoked from elsewhere would otherwise fail on a missing file rather than on a
# real mismatch. (test-ggml-autodiff-probe above can use `.` because it takes no file argument.)
# Registered only after the goldens existed AND the harness was shown to FAIL on a one-nibble
# change -- a verifier that cannot fail is not an oracle.
add_test(NAME test-iqk-golden
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
         COMMAND $<TARGET_FILE:test-iqk-golden> --verify iqk-goldens.txt)
set_property(TEST test-iqk-golden PROPERTY LABELS main)

# IQ5_K / IQ5_KS bulk quantizer: AVX2 fast path (ik_llama #2608) vs the scalar reference -- dispatch counter, the
# GGML_IQK_QUANT_REFERENCE=1 escape equals from_float_ref byte for byte, weighted RMSE(fast) <= 1.01 * RMSE(reference).
# Calls the extern "C" iqk counter directly, so it links `ggml` alone for the same reason test-iqk-golden does.
add_executable(test-iqk-quant-fast test-iqk-quant-fast.cpp)
target_link_libraries(test-iqk-quant-fast PRIVATE ggml)
add_test(NAME test-iqk-quant-fast COMMAND $<TARGET_FILE:test-iqk-quant-fast>)
set_property(TEST test-iqk-quant-fast PROPERTY LABELS main)

# Q2_0_G128 x Q8_2_X4 AVX2 iqk kernel vs a double-precision reference (random weight bytes, Ny 1..17).
# Calls the extern "C" iqk entries directly, so it links `ggml` alone for the same reason test-iqk-golden does.
add_executable(test-iqk-q2-0-g128 test-iqk-q2-0-g128.cpp)
target_link_libraries(test-iqk-q2-0-g128 PRIVATE ggml)
add_test(NAME test-iqk-q2-0-g128 COMMAND $<TARGET_FILE:test-iqk-q2-0-g128>)
set_property(TEST test-iqk-q2-0-g128 PROPERTY LABELS main)

# Mainline / ik Q2_0 (GGML_TYPE_Q2_0 = 401) x Q8_2_X4 AVX2 iqk kernel (ik 422c259a2 port), incl. odd 64-block tails.
add_executable(test-iqk-q2-0 test-iqk-q2-0.cpp)
target_link_libraries(test-iqk-q2-0 PRIVATE ggml)
add_test(NAME test-iqk-q2-0 COMMAND $<TARGET_FILE:test-iqk-q2-0>)
set_property(TEST test-iqk-q2-0 PROPERTY LABELS main)

# Gyro trellis types TQ2_T / TQK6 / TQK7 / TQP5-7 (ik df93e4cc9 #2812): bit-level decode vs to_float, CPU ggml_mul_mat
# (vec_dot fallback + iqk TQ2_T kernels incl. the Q8_0_R8 repack at >= 16 columns), Agention on-disk id remap in gguf.cpp.
add_executable(test-tq-trellis test-tq-trellis.cpp)
target_link_libraries(test-tq-trellis PRIVATE ggml)
add_test(NAME test-tq-trellis COMMAND $<TARGET_FILE:test-tq-trellis>)
set_property(TEST test-tq-trellis PROPERTY LABELS main)

# IQ5_KS_R16 (ik 1b18524bd #2703): the multi-row harness test-quantize-fns lacks -- R16 repack == IQ5_KS bit-exact,
# from_float_ref (16 rows), CPU ggml_mul_mat vs a double reference.
add_executable(test-iqk-iq5-ks-r16 test-iqk-iq5-ks-r16.cpp)
target_link_libraries(test-iqk-iq5-ks-r16 PRIVATE ggml)
add_test(NAME test-iqk-iq5-ks-r16 COMMAND $<TARGET_FILE:test-iqk-iq5-ks-r16>)
set_property(TEST test-iqk-iq5-ks-r16 PROPERTY LABELS main)

# NVFP4 x Q8_0 AVX2 iqk kernel (S69-FASTLLM-PICK-NVFP4-CPU-IQK-2026-10-27-01) vs the type's scalar vec_dot: direct iqk_mul_mat,
# ggml_mul_mat, ggml_mul_mat_id and ggml_moe_up_gate. First check (iqk_gemm_type_b names Q8_0) FAILS before the kernel; x86 only (exit 77 elsewhere).
add_executable(test-iqk-nvfp4 test-iqk-nvfp4.cpp)
target_link_libraries(test-iqk-nvfp4 PRIVATE ggml)
add_test(NAME test-iqk-nvfp4 COMMAND $<TARGET_FILE:test-iqk-nvfp4>)
set_property(TEST test-iqk-nvfp4 PROPERTY LABELS main)
set_property(TEST test-iqk-nvfp4 PROPERTY SKIP_RETURN_CODE 77)
add_test(NAME test-iqk-nvfp4-noiqk COMMAND $<TARGET_FILE:test-iqk-nvfp4>)
set_property(TEST test-iqk-nvfp4-noiqk PROPERTY LABELS main)
set_property(TEST test-iqk-nvfp4-noiqk PROPERTY SKIP_RETURN_CODE 77)
set_property(TEST test-iqk-nvfp4-noiqk PROPERTY ENVIRONMENT "GGML_NO_IQK_GEMM=1")

# ggml_iqk_supports_type(type, site): pins each iqk call site's weight-type set (S69-FORK-CARRY-IQK-SUPPORTS-TYPE-PREDICATE-01).
add_executable(test-iqk-site-types test-iqk-site-types.cpp)
target_link_libraries(test-iqk-site-types PRIVATE ggml)
add_test(NAME test-iqk-site-types COMMAND $<TARGET_FILE:test-iqk-site-types>)
set_property(TEST test-iqk-site-types PROPERTY LABELS main)

# IQ3_K / IQ3_KS / IQ5_K / IQ5_KS / IQ6_K: ggml-cpu registers ik's AVX2 scale search for the ggml-base quantizers
# (iqk_scale_search.h) when GGML_IQK_AVX2_SCALE_SEARCH=1. Default run: nothing registered (scalar, host-independent
# from_float_ref). Opt-in run: registered, deterministic, error vs the scalar search, broken-search control.
add_executable(test-iqk-scale-search test-iqk-scale-search.cpp)
target_link_libraries(test-iqk-scale-search PRIVATE ggml)
add_test(NAME test-iqk-scale-search COMMAND $<TARGET_FILE:test-iqk-scale-search>)
set_property(TEST test-iqk-scale-search PROPERTY LABELS main)
set_property(TEST test-iqk-scale-search PROPERTY ENVIRONMENT "GGML_IQK_AVX2_SCALE_SEARCH=")
add_test(NAME test-iqk-scale-search-optin COMMAND $<TARGET_FILE:test-iqk-scale-search> --opt-in)
set_property(TEST test-iqk-scale-search-optin PROPERTY LABELS main)
set_property(TEST test-iqk-scale-search-optin PROPERTY ENVIRONMENT "GGML_IQK_AVX2_SCALE_SEARCH=1")

# CPU backend supports_op must not claim an op it has no type_traits_cpu row to compute (TURBO*, Q8_KR8, Q8_1_X4, the q8_* activation
# layouts, plain int/double types ... all answered TRUE and then segfaulted/aborted when executed; measured 2026-09-30). Model-free,
# CPU only, links `ggml` alone for the same reason test-iqk-golden does (no `common`).
add_executable(test-cpu-supports-op-rows test-cpu-supports-op-rows.cpp)
target_link_libraries(test-cpu-supports-op-rows PRIVATE ggml)
add_test(NAME test-cpu-supports-op-rows COMMAND $<TARGET_FILE:test-cpu-supports-op-rows>)
set_property(TEST test-cpu-supports-op-rows PROPERTY LABELS main)

# GGML_OP_FMA (ggml_fma, ik_llama #2794): CPU kernel numbers against a scalar host reference + supports_op answers. CPU only, links `ggml` alone.
# ISA guard decision logic: header-only, no ggml link (the header is included directly)
add_executable(test-cpu-isa-guard test-cpu-isa-guard.cpp)
add_test(NAME test-cpu-isa-guard COMMAND $<TARGET_FILE:test-cpu-isa-guard>)
set_property(TEST test-cpu-isa-guard PROPERTY LABELS main)

add_executable(test-fma-cpu test-fma-cpu.cpp)
target_link_libraries(test-fma-cpu PRIVATE ggml)
add_test(NAME test-fma-cpu COMMAND $<TARGET_FILE:test-fma-cpu>)
set_property(TEST test-fma-cpu PROPERTY LABELS main)

add_executable(test-cpu-reduce test-cpu-reduce.cpp)
target_link_libraries(test-cpu-reduce PRIVATE ggml)
add_test(NAME test-cpu-reduce COMMAND $<TARGET_FILE:test-cpu-reduce>)
set_property(TEST test-cpu-reduce PROPERTY LABELS main)

# Audio container routing for the mtmd helper (header-only, model-free).
add_executable(test-mtmd-audio-route ${PROJECT_SOURCE_DIR}/tests/test-mtmd-audio-route.cpp)
target_include_directories(test-mtmd-audio-route PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
add_test(NAME test-mtmd-audio-route COMMAND $<TARGET_FILE:test-mtmd-audio-route>)
set_property(TEST test-mtmd-audio-route PROPERTY LABELS main)

# DeepSeek-V4 dense-mask removal (ik #2779/#2801): visible_mask, mask_to_index_topk, indexer_mask_to_index against host
# references and the dense routes they replace, and maskless FLASH_ATTN_EXT with a src[5] row list against the dense-mask
# FA (iqk indexed and generic kernels). CPU only, links `ggml` alone.
add_executable(test-dsv4-mask-drop test-dsv4-mask-drop.cpp)
target_link_libraries(test-dsv4-mask-drop PRIVATE ggml)
add_test(NAME test-dsv4-mask-drop COMMAND $<TARGET_FILE:test-dsv4-mask-drop>)
set_property(TEST test-dsv4-mask-drop PROPERTY LABELS main)

# Run-time repack (-rtr) and iqk_repacked_type() must leave token_embd / per_layer_token_embd and the DeepSeek-V4 engram tensors
# (engram_embd / engram_k / engram_q) in their plain layout: get_rows / as_f32 have no row for the interleaved one. Model-free,
# CPU only, links `ggml` alone for the same reason test-iqk-golden does. ik_llama 11b54e80f (#2612).
add_executable(test-iqk-repack-forbidden test-iqk-repack-forbidden.cpp)
target_link_libraries(test-iqk-repack-forbidden PRIVATE ggml)
add_test(NAME test-iqk-repack-forbidden COMMAND $<TARGET_FILE:test-iqk-repack-forbidden>)
set_property(TEST test-iqk-repack-forbidden PROPERTY LABELS main)

# ggml_backend_alloc_ctx_tensors_from_buft must view_init every view, also when max_size splits the context and its tail is views
# only (upstream llama.cpp PR #25584, carried by llamafile). Model-free, CPU only, links `ggml`; needs ggml-backend-impl.h to
# build a CPU buffer type with a small max_size.
add_executable(test-alloc-tail-views test-alloc-tail-views.cpp)
target_link_libraries(test-alloc-tail-views PRIVATE ggml)
target_include_directories(test-alloc-tail-views PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
add_test(NAME test-alloc-tail-views COMMAND $<TARGET_FILE:test-alloc-tail-views>)
set_property(TEST test-alloc-tail-views PROPERTY LABELS main)

# REGISTERED != WIRED: every REGISTERED flag in .fork-cli-flags-baseline.txt must reach a behaviour (a reader that is not a cparams
# copy or a log line), REFUSE with a hint, or say "no effect" at parse time. Static, model-free, no compile needed -- it reads the
# sources. The same script is the fork pre-commit gate "cli-flags-wired" (cortex/scripts/check_fork_cli_flags_wired.py runs it).
# scripts/check-fork-modules-wired.py is registered right below it (same Python3 guard); the pre-commit hook lives outside this repo (cortex/), so its wrapper is NOT UPDATED here.
# Skipped (not failed) when no Python 3 interpreter is on PATH at configure time.
# Row S69-FLAG-WIRING-AUDIT-2026-10-01-01; first live finding: -ser (min_experts/thresh_experts written, copied, printed, never read).
find_package(Python3 COMPONENTS Interpreter QUIET)
if (Python3_Interpreter_FOUND)
    add_test(NAME test-cli-flags-wired
             COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/check-fork-cli-flags-wired.py --root ${PROJECT_SOURCE_DIR})
    set_property(TEST test-cli-flags-wired PROPERTY LABELS main)
    add_test(NAME test-cli-flags-wired-selftest
             COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/check-fork-cli-flags-wired.py --selftest)
    set_property(TEST test-cli-flags-wired-selftest PROPERTY LABELS main)
    # BUILT != REACHABLE (modules and type ids): every .cpp under src/ and common/ needs a non-test caller of one of its symbols, every GGML_TYPE_* id needs a
    # reference in the application layer (src/ common/ tools/ examples/ include/), unless scripts/modules-wired-allowlist.txt lists it with a task-row id and a reason.
    # --selftest carries the positive control (a wired module/type passes) and the negative control (a planted unwired module/type fails). Static, model-free.
    # Row S69-MXFP6-MXFP8-INTEGRATION-FTYPE-GGUFPY-SIMD-MSVC-BUILD-2026-10-08-01 (first findings: MX type ids with no ftype; KIVI/koboldcpp/cram modules linked into nothing).
    add_test(NAME test-modules-wired
             COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/check-fork-modules-wired.py --root ${PROJECT_SOURCE_DIR})
    set_property(TEST test-modules-wired PROPERTY LABELS main)
    add_test(NAME test-modules-wired-selftest
             COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/check-fork-modules-wired.py --selftest)
    set_property(TEST test-modules-wired-selftest PROPERTY LABELS main)
    # Turbo KV constants: the WHT sign arrays and the 3/4-bit centroid tables must agree across ggml-turbo-quant.c, ggml-cuda/turbo-quant.cuh and
    # tests/turbo-write-check.h, and be the Lloyd-Max optimum for N(0,1/128). --selftest carries the must-fail controls. Model-free, no compile needed.
    add_test(NAME test-turbo-tables-consistent
             COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/check-turbo-tables-consistent.py --root ${PROJECT_SOURCE_DIR} --selftest)
    set_property(TEST test-turbo-tables-consistent PROPERTY LABELS main)
    # The Vulkan shaders switch on the ggml_type the host passes as a spec constant; ggml_type_ids.glsl must carry ggml.h's values (this fork
    # renumbered Q1_0 / Q2_0, so upstream's 41 / 42 there would silently skip their decode). --selftest carries the must-fail control.
    add_test(NAME test-vk-type-ids-consistent
             COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/check-vk-type-ids-consistent.py --root ${PROJECT_SOURCE_DIR} --selftest)
    set_property(TEST test-vk-type-ids-consistent PROPERTY LABELS main)
    # Every torch.load() in the tree's Python must pass weights_only=True (unpickling an untrusted checkpoint runs code on torch < 2.6). --selftest carries the must-fail cases.
    add_test(NAME test-torch-load-safe
             COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/check-torch-load-safe.py --root ${PROJECT_SOURCE_DIR} --selftest)
    set_property(TEST test-torch-load-safe PROPERTY LABELS main)
    # docs/parameters.md: every flag row complete (GFM drops cells past the header count; a raw '|' in a code span splits a cell) and no flag row twice in one table.
    add_test(NAME test-docs-parameters-table
             COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/check-docs-parameters-table.py --root ${PROJECT_SOURCE_DIR} --selftest)
    set_property(TEST test-docs-parameters-table PROPERTY LABELS main)
    # docs <-> flags two-way: every flag declared in common/arg.cpp is named in a doc, every flag in docs/parameters.md is accepted by a source (ratchet baselines in scripts/docs_flags_known_*.txt).
    add_test(NAME test-docs-flags
             COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/check-docs-flags.py --root ${PROJECT_SOURCE_DIR} --selftest)
    set_property(TEST test-docs-flags PROPERTY LABELS main)
    # ggml_type ids are an on-disk ABI (a GGUF stores the id per tensor): ggml.h must match scripts/ggml_type_ids.golden.json, and the test carries the must-fail cases.
    add_test(NAME test-ggml-type-ids
             COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/check-ggml-type-ids.py)
    set_property(TEST test-ggml-type-ids PROPERTY LABELS main)
    add_test(NAME test-ggml-type-ids-selftest
             COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/test-ggml-type-ids.py)
    set_property(TEST test-ggml-type-ids-selftest PROPERTY LABELS main)
else()
    message(STATUS "Python3 not found: test-cli-flags-wired(-selftest) not registered")
endif()

# test-dsv4-randw-cpu-smoke: deepseek4 and deepseek41 cannot be driven by test-llama-archs (their loader infers every shape from the file's tensor
# metadata, so a weightless model has none). This generates a tiny tensor-bearing random-weight GGUF for each and requires llama-perplexity to load and
# decode it on the CPU with a finite PPL, plus a must-fail control. The CUDA-vs-CPU comparison needs the GPU and stays in tests/data/dsv4-randw/run_differential.bat.
# Exits 77 (reported as SKIPPED, not passed) when gguf/numpy or llama-perplexity is absent. Row S69-FORK-TEST-LLAMA-ARCHS-DEEPSEEK4-41-TENSOR-BEARING-FIXTURE-2026-10-04-01.
if (Python3_Interpreter_FOUND)
    add_test(NAME test-dsv4-randw-cpu-smoke
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/data/dsv4-randw/run_smoke.py --bin-dir ${CMAKE_BINARY_DIR}/bin)
    set_tests_properties(test-dsv4-randw-cpu-smoke PROPERTIES SKIP_RETURN_CODE 77 LABELS main)

    # deepseek4 cannot be written by test-llama-archs (see above), so a generic "-m ${MODEL_DIR}/deepseek4-moe.gguf" registration never had a file.
    # This generates the random-weight fixture and runs test-recurrent-state-rollback on it. MEASURED 2026-10-07 (MinGW, CPU): the single-sequence
    # check passes once the CPU backend implements ggml_get_rows_ext (test-get-rows-ext), but the multi-sequence check FAILS: the second pending
    # partial seq_rm is not refused and the replayed logits differ from a never-advanced reference by 0.998 (eps 1e-7). Left RED on purpose; row
    # S69-FORK-RECURRENT-SECOND-PENDING-SEQ-RM-NOT-REFUSED-LEGACY-OWNER-GOVERNS-2026-10-07-01.
    add_test(NAME test-recurrent-state-rollback-dsv4
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/data/dsv4-randw/run_rollback.py --bin-dir ${CMAKE_BINARY_DIR}/bin
                     --work-dir ${CMAKE_CURRENT_BINARY_DIR}/test-models)
    set_tests_properties(test-recurrent-state-rollback-dsv4 PROPERTIES SKIP_RETURN_CODE 77 LABELS main)

    # test-swa-compress-gemma3: --swa-compress on gemma3 must engage, shrink the KV cache and keep greedy output byte-identical over a
    # prompt that wraps the compacted ring (2544 tokens vs 768 rows); a dense model is the no-change control. Needs real gemma3 GGUFs
    # (LLAMA_TEST_GEMMA3_MODELS, path-separated) and llama-completion; exits 77 (SKIPPED) without them. MEASURED 2026-10-17 MinGW CPU:
    # before 4 FAIL (flag refused, 72.0 -> 72.0 MiB on gemma-3-270m), after PASS (72.0 -> 23.25 MiB; 1b 104.0 -> 32.5 MiB).
    # Row S69-DECL-OVERTURN-GEMMA3-SWA-COMPRESS-2026-10-17-01.
    add_test(NAME test-swa-compress-gemma3
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/kv_regression/test_swa_compress_gemma3.py --bin ${CMAKE_BINARY_DIR}/bin)
    set_tests_properties(test-swa-compress-gemma3 PROPERTIES SKIP_RETURN_CODE 77 LABELS model)

    # test-ollama-qwen35-mtp: the ollama qwen3.6:27b blob run with --spec-type mtp must translate its mtp.* head and get drafts
    # accepted; LLAMA_OLLAMA_COMPAT=0 is the no-translation control; the speculative text must equal plain greedy. Runs a 27B model
    # on CPU (~18 GB RAM, minutes). Exits 77 without the blob. Row S69-DECL-OVERTURN-OLLAMA-QWEN35-MTP-MAP-2026-10-17-01.
    add_test(NAME test-ollama-qwen35-mtp
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-ollama-qwen35-mtp.py)
    set_tests_properties(test-ollama-qwen35-mtp PROPERTIES SKIP_RETURN_CODE 77 LABELS model
                         ENVIRONMENT "LLAMA_BIN_DIR=${CMAKE_BINARY_DIR}/bin")

    # test-server-mtp-serving: llama-server's per-sequence speculator must draft with the MTP head (--spec-type mtp, -np 1 and
    # -np 2 concurrent), name every stage it cannot build in a warning, and log "draft policy on" for suffix,mtp
    # --spec-draft-policy; every output must match plain greedy (exactly, or a near-tie at the first difference, re-scored with
    # speculation off). Needs a small GGUF with a NextN head (LLAMA_TEST_MTP_GGUF, default Qwen3.5-0.8B-Q8_0); exits 77 without
    # it. Row S69-FORK-SERVER-PER-SEQ-SPECULATOR-NO-MTP-NO-DRAFT-POLICY-2026-10-17-01.
    add_test(NAME test-server-mtp-serving
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-server-mtp-serving.py ${CMAKE_BINARY_DIR}/bin)
    set_tests_properties(test-server-mtp-serving PROPERTIES SKIP_RETURN_CODE 77 LABELS model)

    # test-server-probs-utf8: completion_probabilities (and streamed ids) must carry one entry per generated token when a
    # character spans several tokens. Until 2026-10-18 a token ending inside a multi-byte character was left out, which
    # shifted every later entry and made a text-offset losslessness check read the wrong token's top-2 gap. A grammar
    # forces split characters; exits 77 without the model (LLAMA_TEST_PROBS_GGUF, default Qwen3.5-0.8B-Q8_0) or when no
    # character splits. Row S69-FORK-SERVER-SPEC-DECODE-LOSSY-AT-TEMP0-HYBRID-QWEN35-2026-10-18-01.
    add_test(NAME test-server-probs-utf8
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-server-probs-utf8.py ${CMAKE_BINARY_DIR}/bin)
    set_tests_properties(test-server-probs-utf8 PROPERTIES SKIP_RETURN_CODE 77 LABELS model)

    # test-server-spec-pool-eagle3: --spec-type eagle3 must draft in the per-sequence pool that llama-server (-np 2, concurrent
    # requests, a reused slot) and llama-speculative-simple build; outputs must match plain greedy (the MTP test's check, with its
    # controls). Fixtures LLAMA_TEST_E3_TARGET / LLAMA_TEST_E3_DRAFT (default Qwen3-8B Q4_K_M + its converted EAGLE3 draft); exits
    # 77 without them. Ports 18940-18941. Row S69-FORK-SERVER-POOL-EAGLE3-DFLASH-IMPLS-2026-10-17-01.
    add_test(NAME test-server-spec-pool-eagle3
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-server-spec-pool-eagle3.py ${CMAKE_BINARY_DIR}/bin)
    set_tests_properties(test-server-spec-pool-eagle3 PROPERTIES SKIP_RETURN_CODE 77 LABELS model)

    # test-server-spec-pool-stages: the per-sequence pool must build ngram_simple / ngram_mod / ngram_cache (each was warned "not
    # built" and dropped), share an ngram_mod table across slots, rebuild an ngram_cache context for a new request, and honour
    # LLAMA_ARG_SPEC_DRAFT_POLICY (with a no-env control). Fixture LLAMA_TEST_MTP_GGUF (default Qwen3.5-0.8B-Q8_0); exits 77
    # without it. Ports 18930-18937. Rows S69-FORK-ARG-SPEC-DRAFT-POLICY-ENV-2026-10-17-01,
    # S69-FORK-SERVER-POOL-NGRAM-MOD-SIMPLE-CACHE-DSPARK-IMPLS-2026-10-17-01.
    add_test(NAME test-server-spec-pool-stages
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-server-spec-pool-stages.py ${CMAKE_BINARY_DIR}/bin)
    set_tests_properties(test-server-spec-pool-stages PROPERTIES SKIP_RETURN_CODE 77 LABELS model)

    # test-server-bdd-steps: static check that every step line of tools/server/tests/features resolves to a step definition and every
    # `steps.<name>` the step modules use exists (what `behave --dry-run` would report; behave is not installed here). Positive and
    # negative controls inside. No model, no server.
    add_test(NAME test-server-bdd-steps
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-server-bdd-steps.py ${PROJECT_SOURCE_DIR})
    set_tests_properties(test-server-bdd-steps PROPERTIES LABELS main)
endif()

llama_build_and_test(test-backend-ops.cpp)

# A CPU op reading a quantized weight held in an OpenVINO weight buffer must see the GGUF bytes, not the
# extracted layout; S69-FORK-OPENVINO-QUANT-WEIGHT-READBACK-BY-CPU-OP-GARBAGE-2026-10-27-01. SKIPs (exit 0)
# when no OPENVINO0 device is registered.
if (GGML_OPENVINO)
    llama_build_and_test(test-openvino-weight-readback.cpp)
endif()

llama_build_and_test(test-backend-tensor.cpp LINK ggml)

# OpenVINO zero-scale constant sub-block re-encoding: header-only arithmetic, runs on any host (no OpenVINO needed);
# S69-ENGREV-EXECUTORCH-OV-QK-ZERO-SCALE-CONSTANT-SUBBLOCK-2026-10-27-01.
llama_build_and_test(test-ov-constant-group.cpp)
target_include_directories(test-ov-constant-group PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-openvino)

llama_build_and_test(test-tiled-mulmat.cpp  LABEL "main behavior backend"  LINK ggml)
llama_build_and_test(test-model-load-cancel.cpp LABEL "model behavior")
llama_build_and_test(test-autorelease.cpp       LABEL "model behavior")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model behavior backend")

# Test for state restore with fragmented KV cache
# Requires a model, uses same args pattern as test-thread-safety
llama_build_and_test(test-state-restore-fragmented.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-state-restore-fragmented PROPERTIES FIXTURES_REQUIRED test-download-model)

# Test state save/load functionality
llama_build_and_test(test-save-load-state.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-save-load-state PROPERTIES FIXTURES_REQUIRED test-download-model SKIP_RETURN_CODE 77)
# KV state basis tag (k/v_cache_hadamard): cross-basis seq-state restore must be refused, same-basis must reproduce the reference
# logits. Needs a model with a Hadamard-capable K head (gemma-3-270m default, or LLAMA_TEST_STATE_BASIS_MODEL); exits 77 without one.
# Row S69-DECL-OVERTURN-KV-STATE-BASIS-TAG-2026-10-17-01.
llama_build_and_test(test-state-kv-basis.cpp LABEL "model")
set_tests_properties(test-state-kv-basis PROPERTIES SKIP_RETURN_CODE 77)

# Ollama qwen35 blob, MTP requested: the mtp.* head must map onto the NextN layer (65 layers, blk.64.nextn.*); mtp off and the
# LLAMA_OLLAMA_COMPAT=0 opt-out must leave the old 64-layer load. Metadata-only (no_alloc) loads of the qwen3.6:27b blob, so no
# tensor data is read. Exits 77 without the blob. Row S69-DECL-OVERTURN-OLLAMA-QWEN35-MTP-MAP-2026-10-17-01.
llama_build_and_test(test-ollama-qwen35-mtp-map.cpp LABEL "model")
set_tests_properties(test-ollama-qwen35-mtp-map PROPERTIES SKIP_RETURN_CODE 77)
# Same Ollama layout on a tiny self-written qwen35 GGUF (no blob, < 1 MiB): with MTP the three +1.0 norms get owned storage and
# every other tensor stays in the file mapping; --no-mmap, LLAMA_OLLAMA_COMPAT=0 and mtp off are the controls. Before the fix the
# compat turned mmap off for the whole load (16.5 GB anonymous buffer on the 27B blob, refused under a commit cap).
# Row S69-FORK-OLLAMA-QWEN35-MTP-COMPAT-NO-MMAP-16GB-2026-10-18-01.
llama_build_and_test(test-ollama-qwen35-mtp-mmap.cpp)

if (APPLE)
    llama_build(test-rset-release.cpp)
endif()

if (NOT GGML_BACKEND_DL AND GGML_CPU)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp  LABEL "main behavior"  LINK ggml)
    llama_build_and_test(test-cpu-flush-denormals.cpp  LABEL "main behavior"  LINK ggml)
    llama_build_and_test(test-quantize-fns.cpp  LABEL "main behavior"  LINK ggml)
    # llama_model_quantize on a tiny synthetic GGUF: 4D tensors quantize in ALL rows, MXFP4_MOE picks experts by NAME, a 4D imatrix
    # (ne0*ne2*ne3) is accepted, a slab/max-buffer capped run is byte-identical (upstream #30820 5c965fd8c, #27795 732707dff)
    llama_build_and_test(test-quantize-4d.cpp  LABEL "main behavior"  LINK llama)
    # Differential fixture for the base/cpu split of .to_float / .from_float_ref (row
    # S69-GGML-BASE-DLL-CANNOT-LINK-CIRCULAR-ACROSS-A-DLL-BOUNDARY-2026-09-15-01 half 2): quantize + dequantize a
    # fixed-seed row per type and print an FNV-64 of both buffers. Links ggml only (no llama), so it is cheap to rebuild.
    add_executable(test-iqk-base-dequant test-iqk-base-dequant.cpp)
    target_link_libraries(test-iqk-base-dequant PRIVATE ggml)
    add_test(NAME test-iqk-base-dequant COMMAND $<TARGET_FILE:test-iqk-base-dequant>)
    set_property(TEST test-iqk-base-dequant PROPERTY LABELS main)
    # Q1_0 / Q2_0 traits + layout, with a negative control. Separate from test-quantize-fns
    # because that loop aborts at q8_KV (iqk nrc_x%4 assert) long before it reaches types 400/401.
    llama_build_and_test(probe-q1-q2-traits.cpp)
    # validate_row_data must accept every row its own quantizer produces (NVFP4 fell to `default`, 2026-09-27)
    llama_build_and_test(test-validate-row-data.cpp)
    # LLM_KV suffix must be appended (named chat templates: rerank, tool_use)
    llama_build_and_test(test-llm-kv-suffix.cpp)
    # CPU add / add1 / out_prod with a quantised src0, per fork type (the merge dropped their dispatch)
    llama_build_and_test(test-quant-binary-ops.cpp)
    # upstream's settable quantize fudge factors: defaults byte-identical, knob wired
    llama_build_and_test(test-quantize-fudge-factor.cpp)
    # I2_S -> IQ2_BN requantisation of an exact ternary tensor is lossless (BitNet-embedding conversion path)
    llama_build_and_test(test-i2s-iq2bn-requant.cpp  LABEL "main behavior"  LINK ggml)
    llama_build_and_test(test-sparse-downproj-axpy.cpp  LABEL "main behavior"  LINK ggml)
    # IQ3_KT / IQ4_KT rows that are a multiple of 32 but not of QK_K (upstream "tail" format)
    llama_build_and_test(test-kt-tail.cpp)
    llama_build_and_test(test-quantize-perf.cpp  LABEL "main behavior backend"  LINK ggml)
    # row-interleaved (_r4/_r8/_r16) quantize + dequantize stay inside their buffers and round-trip
    # (dequantize_row_mxfp4_r8 overran its output 8x -- the test-quantize-perf heap corruption, 2026-09-28)
    llama_build_and_test(test-rowint-dequant-bounds.cpp)
    llama_build_and_test(test-r8-mulmat-vs-dequant.cpp)
    # TQ2_0 MUL_MAT / MUL_MAT_ID: the iqk GEMM against the type's own vec_dot (same Q8_K bytes) and a double
    # dequant reference, plus the dispatch check that failed before the GEMM existed (row
    # S69-TQ2-0-CPU-PATH-5X-SLOWER-THAN-IQ2-BN-2026-10-16-01). The dispatch probe calls iqk_gemm_type_b directly,
    # so it is compiled in only where that symbol is linked statically.
    llama_build_and_test(test-tq2_0-mulmat.cpp)
    if (GGML_IQK_MUL_MAT AND NOT BUILD_SHARED_LIBS)
        target_compile_definitions(test-tq2_0-mulmat PRIVATE TEST_HAVE_IQK)
    endif()
    # TQ1_0 twin of the test above (S69-TQ1-0-CPU-SCALAR-VEC-DOT-AND-NO-IQK-GEMM-2026-10-17-01): bit-exact AVX2
    # vec_dot + iqk GEMM vs the scalar reference, incl. raw random bytes (all 256 byte values, incl. 243..255 that the quantizer never emits).
    llama_build_and_test(test-tq1_0-mulmat.cpp)
    if (GGML_IQK_MUL_MAT AND NOT BUILD_SHARED_LIBS)
        target_compile_definitions(test-tq1_0-mulmat PRIVATE TEST_HAVE_IQK)
    endif()
    # quantized weight x F16 activation on the CPU backend for row counts that are not a multiple of the Q8_1 interleave
    # (iqk_quantize.cpp:354 assert killed test-backend-ops MUL_MAT on the CUDA run, 2026-10-01)
    llama_build_and_test(test-mul-mat-f16-src1.cpp)
    # NaN/+inf raw-logit counter behind the server's refusal to sample a corrupted row (2026-09-28)
    llama_build_and_test(test-sampling-nonfinite.cpp)
    # adaptive-p never draws a candidate an earlier stage masked (ik #2337, arrived with the ik main merge)
    llama_build_and_test(test-sampling-adaptive-p.cpp)
    # --adaptive-p-untruncated: which requests count as untruncated, the refusal text, the mode names (2026-10-01)
    llama_build_and_test(test-sampling-adaptive-untruncated.cpp)
    # vLLM engine merge (2026-10-08): CPU sampler edge cases found by comparing with vllm/v1/sample
    llama_build_and_test(test-engine-vllm-sampling.cpp)
    # an all -inf candidate list (unsatisfiable grammar/choice) returns LLAMA_TOKEN_NULL instead of aborting the server (2026-10-11)
    llama_build_and_test(test-sampling-all-masked.cpp)
    # DRY breaker table built once per (vocab, breakers), not once per sampler; needs models/ggml-vocab-qwen2.gguf (row S69-ENGREV-EXLLAMAV2-DRY-BREAKER-PRECOMPUTE-2026-10-27-01)
    llama_build_and_test(test-sampling-dry-breaker-cache.cpp  LABEL behavior  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    # DRY apply() per-token host cost at 8k/32k/128k history (CPU, no model; decides whether max_ngram/occurrence caps are worth it)
    llama_build_and_test(test-sampling-dry-cost.cpp  LABEL behavior  LINK llama)
    # FSD (Frustratingly Simple Decoding) sampler: hand-derived vectors, state, clone/reset, randomized variant (2026-10-05)
    llama_build_and_test(test-sampling-fsd.cpp)
    # exact top-k prefilter over raw logits + its eligibility rule (2026-10-05)
    llama_build_and_test(test-sampling-topk-prefilter.cpp)
    # top-k threshold scan + greedy argmax vs the real old path (llama_sample_top_k / llama_sample_token_greedy), thousands of cases (2026-10-07)
    llama_build_and_test(test-topk-prefilter.cpp)
    # suffix automaton for self-speculative drafting: brute-force longest-repeated-suffix oracle, overlapping copy, memory (2026-10-05)
    llama_build_and_test(test-suffix-automaton.cpp)
    # named external n-gram corpora (registry + budget split + versioned incremental matcher + automaton cursor), S69-ENGREV-SGLANG-NGRAM-EXTERNAL-CORPUS
    llama_build_and_test(test-ngram-corpus-registry.cpp)
    # lookup-vs-model-drafter pick and chain depth from learned acceptance and measured round cost (Strata S2 spec, 2026-10-07)
    llama_build_and_test(test-draft-policy.cpp)
    # qwen4exp MTP head: hnorm is one RMS per hc stream, not one over the flat hc*n_embd vector (2026-10-07)
    llama_build_and_test(test-qwen4exp-hnorm.cpp)
    llama_build_and_test(test-spec-pool-suffix.cpp)
    # -md without --spec-type gives a draft-model stage (2026-10-27)
    llama_build_and_test(test-spec-md-implicit-draft.cpp)
    llama_build_and_test(test-spec-tuner-suffix-automaton.cpp)
    llama_build_and_test(test-kv-diversity.cpp)
    llama_build_and_test(test-kv-diversity-prune.cpp)   # KVCrush/AdaptiveR-KV keep plan vs the mass-only prune plan
    llama_build_and_test(test-kv-xattn.cpp)
    # XAttention step 2: the mask custom op on graph-layout K/Q views vs an expectation from the step-1 estimator (S69-openvino-PICK-XATTENTION-...)
    llama_build_and_test(test-kv-xattn-op.cpp)
    llama_build_and_test(test-spec-tree.cpp)
    llama_build_and_test(test-spec-mtp-state.cpp)
    llama_build_and_test(test-spec-mtp-embd.cpp)
    llama_build_and_test(test-lora-pool.cpp)
    llama_build_and_test(test-lora-sgmv-equiv.cpp)
    llama_build_and_test(test-lora-merge.cpp)
    llama_build_and_test(test-rope.cpp  LABEL "main behavior backend"  LINK ggml)
    # YaRN truncate=false correction range + mscale/mscale_all_dim ratio vs an independent HF-formula reference (CPU)
    # (row S69-EXO-YARN-TRUNCATE-FALSE-AND-MSCALE-RATIO-2026-11-06-01); numpy twin: tests/rope_yarn_truncate_ref.py
    llama_build_and_test(test-rope-yarn-truncate.cpp  LABEL "main behavior backend"  LINK ggml)
    # Loader: <arch>.rope.scaling.yarn_truncate / yarn_mscale reach llama_hparams (and an old GGUF without them keeps the
    # defaults). The two metadata-only fixtures are written at configure time by the real gguf-py writer; with no Python
    # (or no numpy) the test is registered without arguments and SKIPs (exit 0).
    set(YARN_FIX_DIR ${CMAKE_CURRENT_BINARY_DIR}/yarn-truncate-check)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    set(YARN_FIX_ARGS)
    if (Python3_Interpreter_FOUND)
        execute_process(COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/yarn-truncate-check/make_tiny_yarn_gguf.py ${YARN_FIX_DIR}
                        RESULT_VARIABLE YARN_FIX_RC OUTPUT_QUIET ERROR_QUIET)
        if (YARN_FIX_RC EQUAL 0)
            set(YARN_FIX_ARGS ARGS ${YARN_FIX_DIR}/yarn_keys.gguf ${YARN_FIX_DIR}/yarn_nokeys.gguf)
        endif()
    endif()
    llama_build_and_test(test-yarn-hparams-load.cpp ${YARN_FIX_ARGS})
    llama_build_and_test(test-col2im-1d.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-conv-1d.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-gelu-quick.cpp  LABEL "main behavior backend"  LINK ggml)
    # ggml_get_rows_ext dim0 gather and same-type row copy on the CPU backend (DeepSeek-V4 single-token decode, 2026-10-07)
    llama_build_and_test(test-get-rows-ext.cpp)
    llama_build_and_test(test-fs-validate-filename.cpp)
    # GGUF read: upstream Q2_0 (id 42, 64 elem / 18 B) vs our Q2_0_G128 (id 42, 128 elem / 34 B), decided by block geometry (Strata S2, 2026-10-07)
    llama_build_and_test(test-gguf-upstream-q2_0-remap.cpp)
    # GGUF read: TheTom turboquant TQ3_1S/TQ4_1S on disk as 45/46 (our TURBO4_0/TURBO2_0) -> 74/75, decided by block geometry (2026-10-17)
    llama_build_and_test(test-gguf-tq1s-remap.cpp)
    # GGUF v1 read (32-bit counts / lengths / dims), written back as the current version (2026-10-17)
    llama_build_and_test(test-gguf-v1-read.cpp)
    # general.file_type read from a GGUF head prefix + ranged probe (nexa-sdk gguf.rs idea, BSD-3, no code copied;
    # row S69-ENGREV-NEXA-GGUF-REMOTE-HEADER-PROBE-2026-11-15-01)
    llama_build_and_test(test-gguf-header-probe.cpp)
    # llama-imatrix --hessian-block: block-diagonal X^T X vs a naive double reference, sidecar round trip, CLI bounds,
    # negative controls; `test-imatrix-hessian --baseline-diag-only` is the fail-before (row S69-F1-PICK-IMATRIX-BLOCK-HESSIAN-OUTPUT-2026-10-06-01)
    llama_build_and_test(test-imatrix-hessian.cpp)
    # llama-quantize --gptq-hessian: GPTQ group update vs references, byte-identity controls, objective vs ordinary rounding,
    # sign-flip negative control (row S69-ncnn-PICK-GPTQ-ERROR-COMPENSATED-ROUNDING-2026-09-29-01)
    llama_build_and_test(test-gptq-rounding.cpp)
    # GGML_OP_FFD_SELECT: FFD decode block selector vs an independent double reference (2026-09-28)
    llama_build_and_test(test-ffd-select.cpp)
    # GGML_OP_FFD_QUANT_K: FFD 2-bit K shadow maintenance, bit-exact vs a contract reference (2026-09-28)
    llama_build_and_test(test-ffd-quant.cpp)
    # --ffd shadow maintenance through the real decode path; needs a model: test-ffd-kv <model.gguf> (2026-09-28)
    llama_build(test-ffd-kv.cpp)
    # real-model attention statistics for the SageAttention and FFD rows (CPU, needs a model): attn-stats <model.gguf> <text> (2026-10-05)
    llama_build(attn-stats.cpp)
    # model-level cross-backend verify (greedy tokens + per-layer K/V divergence + MoE routing TV, MATCH/DRIFT/TIEFLIP/WRONG with a measured
    # noise band): the --selftest is model-free; verify-backends.py runs it end to end on a random-weight GGUF (needs numpy).
    # Row S69-ENGREV-FERROX-BACKEND-VERIFY-LAYER-DIVERGENCE-2026-10-27-01.
    llama_build(verify-backends.cpp)
    llama_test(verify-backends NAME test-verify-backends-selftest ARGS --selftest LABEL behavior)
    # per-tensor quantisation sensitivity (one tensor through a Q4_0/Q8_0 round trip, KL + top-1 flip + per-block relmse, family rollup),
    # CPU only; --selftest is model-free, quant-sensitivity.py runs it on a random-weight GGUF. Row S69-ENGREV-FERROX-QUANT-SENSITIVITY-TOOL-2026-10-27-01.
    llama_build(quant-sensitivity.cpp)
    llama_test(quant-sensitivity NAME test-quant-sensitivity-selftest ARGS --selftest LABEL behavior)
    # CUDA VMM pool free-order bookkeeping (ggml-cuda/pool-extents.h), CPU only (2026-10-05)
    llama_build_and_test(test-cuda-pool-extents.cpp)
    # CUDA-graph cache key (ggml-cuda/graph-key.h): n_nodes == 0 returns the fixed key without reading nodes[-1]; CPU only (row S69-ENGREV-THECODACUS-CUDA-GRAPH-KEY-EMPTY-GUARD-2026-10-27-01)
    llama_build_and_test(test-cuda-graph-key-empty.cpp)
    # persistent Vulkan pipeline-cache file format: identity, damage and path checks, no Vulkan (2026-10-05)
    llama_build_and_test(test-vk-pipeline-cache-file.cpp)
    # Vulkan heap free bytes clamp at 0 when heapUsage > heapBudget (ggml-vulkan/vk-heap-free.h), no Vulkan (2026-10-17)
    llama_build_and_test(test-vk-heap-free.cpp)
    # per-layer K/V cache types for the memory-module cache + the Boundary V turbo2 default, pure policy (2026-10-17)
    llama_build_and_test(test-kv-layer-types.cpp)
    # DSA indexer top-k radix select: the kernel body run on the CPU, one thread per CUDA thread (2026-10-05)
    llama_build_and_test(test-topk-select.cpp)
    # kernel-variant autotuner (ggml-cuda/autotune.h): budgets, successive halving, salted disk cache; CPU only (2026-10-05)
    llama_build_and_test(test-cuda-autotune.cpp)
    # GGML_CUDA_FATTN_FORCE parse + legality (ggml-cuda/fattn-force.h); CPU only (2026-11-17)
    llama_build_and_test(test-fattn-force.cpp)
    # flash-attention invariants (poisoned unused KV rows, fully masked row, sink, padded mask rows) on any backend: test-fa-invariants [backend] (2026-10-06)
    # args: backend, path (default = iqk when it accepts the shape | generic = GGML_NO_IQK_FA=1), group (all | finite | nan)
    llama_build_and_test(test-fa-invariants.cpp ARGS CPU default finite)
    llama_test(test-fa-invariants NAME test-fa-invariants-generic ARGS CPU generic all)
    # iqk adds the mask instead of skipping masked keys, so NaN in masked KV rows leaks (K=NaN: finite wrong 0.95, V=NaN: all NaN).
    # Expected failure until fixed: row S69-F2-IQK-FA-MASKED-KEYS-NOT-SKIPPED-NAN-KV-2026-10-11-01. When iqk is fixed this goes RED
    # (WILL_FAIL inverts the pass), which is the signal to delete the property.
    llama_test(test-fa-invariants NAME test-fa-invariants-iqk-nan ARGS CPU default nan)
    set_tests_properties(test-fa-invariants-iqk-nan PROPERTIES WILL_FAIL TRUE)
endif()

# libmtmd
llama_build_and_test(test-mtmd-c-api.c  LABEL behavior  LINK mtmd)

llama_build_and_test(test-mtmd-impl.cpp  LABEL behavior  LINK mtmd)

# mtmd audio-encoder provider hook (2026-10-05): the contract, the loader, the refusal paths and the comparison, run
# against a test plugin module loaded by path.
add_library(mtmd-encoder-provider-test-plugin MODULE test-mtmd-encoder-provider-plugin.cpp)
set_target_properties(mtmd-encoder-provider-test-plugin PROPERTIES PREFIX "")
target_include_directories(mtmd-encoder-provider-test-plugin PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
target_compile_features(mtmd-encoder-provider-test-plugin PRIVATE cxx_std_17)
llama_build_and_test(test-mtmd-encoder-provider.cpp ARGS $<TARGET_FILE:mtmd-encoder-provider-test-plugin>)
target_link_libraries(test-mtmd-encoder-provider PRIVATE mtmd)
add_dependencies(test-mtmd-encoder-provider mtmd-encoder-provider-test-plugin)

# real-audio provider-vs-ggml comparison (needs an mmproj, an exported model and wav files, so not an add_test):
#   test-mtmd-encoder-compare <mmproj> <provider module> <provider model> <device> <tol> <threads> <wav>...
llama_build(test-mtmd-encoder-compare.cpp)
target_link_libraries(test-mtmd-encoder-compare PRIVATE mtmd)

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        add_library(gguf-model-data STATIC gguf-model-data.cpp)
        target_link_libraries(gguf-model-data PRIVATE llama-common cpp-httplib)
        target_include_directories(gguf-model-data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

        add_executable(test-gguf-model-data test-gguf-model-data.cpp)
        target_link_libraries(test-gguf-model-data PRIVATE gguf-model-data llama-common)
        llama_test(test-gguf-model-data  LABEL "model behavior")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp  LABEL "model behavior"  LINK llama)
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp  LABEL behavior  LINK ggml)
    # Needs non-public ggml-impl.h (gguf_write_to_buf). GGUF write-back of a context READ from a file: byte-exact full
    # write, clean refusal without tensor data (2026-10-18)
    llama_build_and_test(test-gguf-write-back.cpp)

    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp  LABEL "behavior backend"  LINK ggml)
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()

# ---------------------------------------------------------------------------------------------
# ggml-cuda-dl (S69-ADD-FERROX-CUDARC-DYNAMIC-CUDA-LOADING-2026-09-27-01, stages 1-2).
# Bare add_executable + link ggml-cuda-dl ONLY -- NOT llama_build_and_test, which links
# llama-common, and `common` does not compile on this tree's MinGW toolchain (see the note above
# test-ggml-autodiff-probe, this file, ~line 420). Same pattern as that target / test-iqk-golden.
# ---------------------------------------------------------------------------------------------

if (TARGET ggml-cuda-dl)  # skipped when ggml/src/CMakeLists.txt finds no vendor/hash
add_executable(test-cuda-dl test-cuda-dl.cpp)
target_link_libraries(test-cuda-dl PRIVATE ggml-cuda-dl)
add_test(NAME test-cuda-dl WORKING_DIRECTORY . COMMAND $<TARGET_FILE:test-cuda-dl>)
set_property(TEST test-cuda-dl PROPERTY LABELS main)

add_executable(test-cuda-dl-bogus-names test-cuda-dl-bogus-names.cpp)
target_link_libraries(test-cuda-dl-bogus-names PRIVATE ggml-cuda-dl)
add_test(NAME test-cuda-dl-bogus-names WORKING_DIRECTORY . COMMAND $<TARGET_FILE:test-cuda-dl-bogus-names>)
set_property(TEST test-cuda-dl-bogus-names PROPERTY LABELS main)

add_executable(test-cuda-dl-nvrtc test-cuda-dl-nvrtc.cpp)
target_link_libraries(test-cuda-dl-nvrtc PRIVATE ggml-cuda-dl)
# Point the test at this host's real NVRTC so the cold/warm/corrupt cache checks exercise the real
# path instead of just reporting SKIPPED -- LOAD_LIBRARY_SEARCH_DEFAULT_DIRS does not include the
# toolkit's own bin dir unless the caller adds it (matches cudarc's reliance on LD_LIBRARY_PATH).
# $ENV{CUDA_PATH} first (standard NVIDIA installer var), else the path measured on this host
# 2026-09-28; either way this is compile-time convenience only -- GGML_CUDA_DL_NVRTC set by the
# invoker always wins (see test-cuda-dl-nvrtc.cpp).
if (WIN32)
    set(_cuda_dl_nvrtc_candidates)
    if (DEFINED ENV{CUDA_PATH})
        file(GLOB _cuda_dl_nvrtc_candidates "$ENV{CUDA_PATH}/bin/x64/nvrtc64_*.dll" "$ENV{CUDA_PATH}/bin/nvrtc64_*.dll")
    endif()
    if (NOT _cuda_dl_nvrtc_candidates)
        file(GLOB _cuda_dl_nvrtc_candidates
             "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*/bin/x64/nvrtc64_*.dll")
    endif()
    if (_cuda_dl_nvrtc_candidates)
        list(GET _cuda_dl_nvrtc_candidates 0 _cuda_dl_nvrtc_path)
        file(TO_CMAKE_PATH "${_cuda_dl_nvrtc_path}" _cuda_dl_nvrtc_path)
        target_compile_definitions(test-cuda-dl-nvrtc PRIVATE
            GGML_CUDA_DL_TEST_NVRTC_PATH="${_cuda_dl_nvrtc_path}")
        message(STATUS "ggml-cuda-dl: test-cuda-dl-nvrtc will default GGML_CUDA_DL_NVRTC to ${_cuda_dl_nvrtc_path}")
    else()
        message(STATUS "ggml-cuda-dl: no nvrtc64_*.dll found at configure time; test-cuda-dl-nvrtc will report SKIPPED unless GGML_CUDA_DL_NVRTC is set at run time")
    endif()
endif()
add_test(NAME test-cuda-dl-nvrtc WORKING_DIRECTORY . COMMAND $<TARGET_FILE:test-cuda-dl-nvrtc>)
set_property(TEST test-cuda-dl-nvrtc PROPERTY LABELS main)

# test-cuda-dl-launch: built but DELIBERATELY NOT add_test()'d -- it allocates real GPU memory and
# must only be invoked by hand under an agent_dispatch.py --claim-gpu slot (see the file header).
# `ctest` running the full main label must never touch the GPU on its own.
add_executable(test-cuda-dl-launch test-cuda-dl-launch.cpp)
target_link_libraries(test-cuda-dl-launch PRIVATE ggml-cuda-dl)

# test-cuda-dl-silu-parity (stage 3, S69-ADD-FERROX-CUDARC-DYNAMIC-CUDA-LOADING-2026-09-27-01):
# first real ggml-cuda kernel (SiLU) ported onto the NVRTC path, parity vs CPU ref + a static
# (nvcc) reference cubin, must-fail control, timing. Same GPU-memory rule as test-cuda-dl-launch:
# built but NOT add_test()'d -- run by hand under a claimed GPU slot.
add_executable(test-cuda-dl-silu-parity test-cuda-dl-silu-parity.cpp)
target_link_libraries(test-cuda-dl-silu-parity PRIVATE ggml-cuda-dl)
# F16 coverage is best-effort and needs cuda_fp16.h from the installed toolkit at NVRTC compile
# time (not at C++ compile time here -- this only tells the NVRTC call at run time where to -I).
# Same $ENV{CUDA_PATH}-first, then glob-fallback pattern as the test-cuda-dl-nvrtc block above.
if (WIN32)
    set(_cuda_dl_toolkit_include "")
    if (DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}/include/cuda_fp16.h")
        set(_cuda_dl_toolkit_include "$ENV{CUDA_PATH}/include")
    else()
        file(GLOB _cuda_dl_toolkit_include_candidates
             "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*/include/cuda_fp16.h")
        if (_cuda_dl_toolkit_include_candidates)
            list(GET _cuda_dl_toolkit_include_candidates 0 _cuda_dl_fp16_hdr)
            get_filename_component(_cuda_dl_toolkit_include "${_cuda_dl_fp16_hdr}" DIRECTORY)
        endif()
    endif()
    if (_cuda_dl_toolkit_include)
        file(TO_CMAKE_PATH "${_cuda_dl_toolkit_include}" _cuda_dl_toolkit_include)
        target_compile_definitions(test-cuda-dl-silu-parity PRIVATE
            GGML_CUDA_DL_TEST_TOOLKIT_INCLUDE="${_cuda_dl_toolkit_include}")
        message(STATUS "ggml-cuda-dl: test-cuda-dl-silu-parity F16 will use toolkit include ${_cuda_dl_toolkit_include}")
    else()
        message(STATUS "ggml-cuda-dl: no cuda_fp16.h found at configure time; test-cuda-dl-silu-parity F16 will report SKIPPED")
    endif()
endif()
endif()

# ---------------------------------------------------------------------------------------------
# STANDALONE PROBES UNDER CTEST (row S69-FORK-STANDALONE-PROBES-AND-SERVED-E2E-SCRIPTS-NOT-RUN-BY-ANY-GATE-2026-10-01-01).
# These probes used to link the static archives by hand (build line in each header) and were run only
# when someone remembered; no build, hook or schedule executed them. They are CMake targets now, so
# `ctest -R probe-` runs them. Each one prints "N checks, M failed" and exits non-zero on a failure;
# each carries its own negative control.
# This block is the END of the file on purpose (append-only touch point shared with other branches).
# ---------------------------------------------------------------------------------------------
if (NOT GGML_BACKEND_DL)
    # needs only libllama (llama_chat_apply_template, named template "deepseek4")
    llama_build_and_test(probe-chat-builtin-deepseek4.cpp)

    # common_sampler_needs_host(): an active XTC request must fall back to the host sampler (links llama-common)
    llama_build_and_test(probe-xtc-needs-host.cpp)

    # common_prompt_checkpoint::spill()/reload() (--ctx-ckpt-spill-dir): fake-byte checkpoints, no model (links llama-common)
    llama_build_and_test(probe-ckpt-spill.cpp)
    # common_ckpt_dominated() (--ctx-ckpt-dominance): pure list logic, no model (row TRNRT-ADAPT-CKPT-DOMINATED-FREE)
    llama_build_and_test(test-ckpt-dominance.cpp)

    # common_disk_tier (--cache-disk-dir): checksummed blobs, size cap, write budget; payloads are patterned bytes, no model
    llama_build_and_test(probe-disk-tier.cpp)

    # --cache-disk-io-chunk (row S69-LINK-STORAGE-OFFLOAD-1014): pipelined read/checksum overlap in common_disk_tier; links only disk-tier.cpp
    add_executable(probe-disk-tier-pipeline probe-disk-tier-pipeline.cpp ${PROJECT_SOURCE_DIR}/common/disk-tier.cpp)
    target_include_directories(probe-disk-tier-pipeline PRIVATE ${PROJECT_SOURCE_DIR}/common)
    target_compile_features(probe-disk-tier-pipeline PRIVATE cxx_std_17)
    find_package(Threads REQUIRED)
    target_link_libraries(probe-disk-tier-pipeline PRIVATE Threads::Threads)
    add_test(NAME probe-disk-tier-pipeline COMMAND $<TARGET_FILE:probe-disk-tier-pipeline>)
    # free-space guard of common_disk_tier (row S69-MLX-PICK-DISK-TIER-FREE-SPACE-GUARD-2026-10-05-01): injected free-space query, no disk filled; links only disk-tier.cpp
    add_executable(test-disk-tier-space test-disk-tier-space.cpp ${PROJECT_SOURCE_DIR}/common/disk-tier.cpp)
    target_include_directories(test-disk-tier-space PRIVATE ${PROJECT_SOURCE_DIR}/common)
    target_compile_features(test-disk-tier-space PRIVATE cxx_std_17)
    target_link_libraries(test-disk-tier-space PRIVATE Threads::Threads)
    add_test(NAME test-disk-tier-space COMMAND $<TARGET_FILE:test-disk-tier-space>)
    # async restore of common_disk_tier blobs (row S69-ENGREV-RTPLLM-ASYNC-DISK-RESTORE-2026-10-27-01); links only disk-tier.cpp
    add_executable(test-disk-tier-async test-disk-tier-async.cpp ${PROJECT_SOURCE_DIR}/common/disk-tier.cpp)
    target_include_directories(test-disk-tier-async PRIVATE ${PROJECT_SOURCE_DIR}/common)
    target_compile_features(test-disk-tier-async PRIVATE cxx_std_17)
    target_link_libraries(test-disk-tier-async PRIVATE Threads::Threads)
    add_test(NAME test-disk-tier-async COMMAND $<TARGET_FILE:test-disk-tier-async>)
    # the same flag through the real parser (links llama-common)
    llama_build_and_test(probe-disk-tier-flag.cpp)

    # restore-from-tier versus recompute on a real model, CPU only (row S69-TIERED-RESTORE-VS-RECOMPUTE-POLICY-2026-10-05-01); needs a model, so not a ctest
    llama_build(bench-tier-restore.cpp)
    # buffered versus unbuffered / overlapped / I/O-ring file reads and writes for the tier's blobs (row S69-TIERED-WINDOWS-STORAGE-IO-PATH-2026-10-05-01); needs a big file, so not a ctest
    llama_build(bench-tier-io.cpp)

    # ggml-only numeric probes (bare add_executable against `ggml`, same reason as test-iqk-golden above)
    add_executable(probe-cpu-op-reference probe-cpu-op-reference.cpp)
    target_link_libraries(probe-cpu-op-reference PRIVATE ggml)
    add_test(NAME probe-cpu-op-reference COMMAND $<TARGET_FILE:probe-cpu-op-reference>)
    set_property(TEST probe-cpu-op-reference PROPERTY LABELS main)

    add_executable(probe-interleaved-vs-base-mulmat probe-interleaved-vs-base-mulmat.cpp)
    target_link_libraries(probe-interleaved-vs-base-mulmat PRIVATE ggml)
    add_test(NAME probe-interleaved-vs-base-mulmat COMMAND $<TARGET_FILE:probe-interleaved-vs-base-mulmat>)
    set_property(TEST probe-interleaved-vs-base-mulmat PROPERTY LABELS main)

    add_executable(probe-kt-dequant-consistency probe-kt-dequant-consistency.cpp)
    target_link_libraries(probe-kt-dequant-consistency PRIVATE ggml)
    add_test(NAME probe-kt-dequant-consistency COMMAND $<TARGET_FILE:probe-kt-dequant-consistency>)
    set_property(TEST probe-kt-dequant-consistency PROPERTY LABELS main)

    # probe-xtc-eog-exempt needs a model WITH a vocabulary (EOS id 2, a newline-piece token). No such tiny model is
    # checked in, so a fixture test generates one (tests/lfm2-forward-check/make_tiny_vocab_gguf.py: random LFM2 weights
    # converted by the fork's own converter). That script needs numpy + torch + safetensors + gguf. Point
    # -DLLAMA_PROBE_PYTHON=<python> at an interpreter that has them. Without one the probe is registered DISABLED, which
    # ctest lists as "Not Run (Disabled)" in its summary -- a visible gap rather than a silently missing test.
    set(LLAMA_PROBE_PYTHON "" CACHE FILEPATH "python with numpy+torch+safetensors+gguf, used to make the tiny vocab model for probe-xtc-eog-exempt")
    if (NOT LLAMA_PROBE_PYTHON)
        find_package(Python3 QUIET COMPONENTS Interpreter)
        if (Python3_Interpreter_FOUND)
            set(_probe_py "${Python3_EXECUTABLE}")
        endif()
    else()
        set(_probe_py "${LLAMA_PROBE_PYTHON}")
    endif()
    set(_probe_py_ok FALSE)
    if (_probe_py)
        execute_process(COMMAND "${_probe_py}" -c "import numpy, torch, safetensors, gguf"
                        RESULT_VARIABLE _probe_py_rc OUTPUT_QUIET ERROR_QUIET)
        if (_probe_py_rc EQUAL 0)
            set(_probe_py_ok TRUE)
        endif()
    endif()
    set(PROBE_MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/probe-models")
    file(MAKE_DIRECTORY "${PROBE_MODEL_DIR}")
    llama_build(probe-xtc-eog-exempt.cpp)
    llama_test(probe-xtc-eog-exempt ARGS "${PROBE_MODEL_DIR}/dense.gguf")
    if (_probe_py_ok)
        add_test(NAME probe-make-tiny-vocab-model
                 COMMAND "${_probe_py}" "${CMAKE_CURRENT_SOURCE_DIR}/lfm2-forward-check/make_tiny_vocab_gguf.py"
                         "${PROJECT_SOURCE_DIR}" "${PROBE_MODEL_DIR}")
        set_tests_properties(probe-make-tiny-vocab-model PROPERTIES FIXTURES_SETUP probe-tiny-vocab-model LABELS main)
        set_tests_properties(probe-xtc-eog-exempt PROPERTIES FIXTURES_REQUIRED probe-tiny-vocab-model)
    else()
        message(STATUS "probe-xtc-eog-exempt: DISABLED (no python with numpy+torch+safetensors+gguf; pass -DLLAMA_PROBE_PYTHON=<python>)")
        set_tests_properties(probe-xtc-eog-exempt PROPERTIES DISABLED TRUE)
    endif()
endif()

# Header-only server guards: standalone unit tests with no server process and no model (they include tools/server/server-repetition.h
# and server-toolloop.h only). Each carries opposite-answer controls inside. They were compiled by hand (their headers say so) and run by nobody.
add_executable(test-server-repetition ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_repetition.cpp)
target_include_directories(test-server-repetition PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-repetition COMMAND $<TARGET_FILE:test-server-repetition>)
set_property(TEST test-server-repetition PROPERTY LABELS main)
add_executable(test-server-loop-guard ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_loop_guard.cpp)
target_include_directories(test-server-loop-guard PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-loop-guard COMMAND $<TARGET_FILE:test-server-loop-guard>)
set_property(TEST test-server-loop-guard PROPERTY LABELS main)

add_executable(test-server-toolloop ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_toolloop.cpp)
target_include_directories(test-server-toolloop PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-toolloop COMMAND $<TARGET_FILE:test-server-toolloop>)
set_property(TEST test-server-toolloop PROPERTY LABELS main)

# queue-aware LRU slot eviction, --slot-evict-lookahead (row S69-NANOVLLM-PICK-QUEUE-AWARE-SLOT-EVICTION-2026-10-18-01), model-free
add_executable(test-server-slot-evict ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_slot_evict.cpp)
target_include_directories(test-server-slot-evict PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-slot-evict COMMAND $<TARGET_FILE:test-server-slot-evict>)
set_property(TEST test-server-slot-evict PROPERTY LABELS main)

# child log line -> out-of-memory verdict (tools/server/server-oom-line.h; row S69-ENGREV-JAN-OOM-LINE-BARE-FAILED-TO-ALLOCATE-2026-10-27-01), model-free
add_executable(test-server-oom-line ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_oom_line.cpp)
target_include_directories(test-server-oom-line PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-oom-line COMMAND $<TARGET_FILE:test-server-oom-line>)
set_property(TEST test-server-oom-line PROPERTY LABELS main)

# router model lifecycle: stop reason + bounded event ring (tools/server/server-model-events.h; row S69-ENGREV-TFSERVING-LIFECYCLE-EVENT-RING-2026-10-27-01), model-free
add_executable(test-server-model-events ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_model_events.cpp)
target_include_directories(test-server-model-events PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-model-events COMMAND $<TARGET_FILE:test-server-model-events>)
set_property(TEST test-server-model-events PROPERTY LABELS main)

# --kv-unified idle-slot purge order: least recently used first (row S69-VLLM-W2-IDLE-SLOT-LRU-PURGE-2026-10-27-01), model-free
add_executable(test-server-slot-purge ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_slot_purge.cpp)
target_include_directories(test-server-slot-purge PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-slot-purge COMMAND $<TARGET_FILE:test-server-slot-purge>)
set_property(TEST test-server-slot-purge PROPERTY LABELS main)

# --kv-unified preemption victim policies (row S69-VLLM-W2-PREEMPT-VICTIM-LEAST-PROGRESS-AND-E2E-2026-10-27-01), model-free
add_executable(test-server-slot-victim ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_slot_victim.cpp)
target_include_directories(test-server-slot-victim PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-slot-victim COMMAND $<TARGET_FILE:test-server-slot-victim>)
set_property(TEST test-server-slot-victim PROPERTY LABELS main)

# the SWA state-reuse threshold (row S69-LLAMACPP-PICK-SWA-REUSE-THRESHOLD-OFF-BY-TWO-2026-10-18-01), model-free
add_executable(test-server-swa-reuse ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_swa_reuse.cpp)
target_include_directories(test-server-swa-reuse PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR}/include
                           ${PROJECT_SOURCE_DIR}/ggml/include)
add_test(NAME test-server-swa-reuse COMMAND $<TARGET_FILE:test-server-swa-reuse>)
set_property(TEST test-server-swa-reuse PROPERTY LABELS main)

add_executable(test-server-asr-format ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_asr_format.cpp)
target_include_directories(test-server-asr-format PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-asr-format COMMAND $<TARGET_FILE:test-server-asr-format>)
set_property(TEST test-server-asr-format PROPERTY LABELS main)

add_executable(test-server-asr-vibevoice ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_asr_vibevoice.cpp)
target_include_directories(test-server-asr-vibevoice PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-asr-vibevoice COMMAND $<TARGET_FILE:test-server-asr-vibevoice>)
set_property(TEST test-server-asr-vibevoice PROPERTY LABELS main)

add_executable(test-server-hostguard ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_hostguard.cpp)
target_include_directories(test-server-hostguard PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-hostguard COMMAND $<TARGET_FILE:test-server-hostguard>)
set_property(TEST test-server-hostguard PROPERTY LABELS main)
# /metrics histograms, step composition and finish reasons (tools/server/server-latency.h): the arithmetic, model-free.
add_executable(test-server-latency ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_latency.cpp)
target_include_directories(test-server-latency PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-latency COMMAND $<TARGET_FILE:test-server-latency>)
set_property(TEST test-server-latency PROPERTY LABELS main)
# Forward-progress watchdog state machine (tools/server/server-watchdog.h), injected clock, model-free.
add_executable(test-server-watchdog ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_watchdog.cpp)
target_include_directories(test-server-watchdog PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-watchdog COMMAND $<TARGET_FILE:test-server-watchdog>)
set_property(TEST test-server-watchdog PROPERTY LABELS main)
# KV audit: canary chain, pool leak check, non-finite logits (tools/server/server-kv-audit.h), model-free.
add_executable(test-server-kv-audit ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_kv_audit.cpp)
target_include_directories(test-server-kv-audit PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-kv-audit COMMAND $<TARGET_FILE:test-server-kv-audit>)
set_property(TEST test-server-kv-audit PROPERTY LABELS main)
# per-request JSONL log, in-flight journal, traceparent / X-Request-Id ingestion (tools/server/server-request-log.h): model-free.
add_executable(test-server-request-log ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_request_log.cpp)
target_include_directories(test-server-request-log PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-request-log COMMAND $<TARGET_FILE:test-server-request-log>)
set_property(TEST test-server-request-log PROPERTY LABELS main)

# budget groups: one token + wall-time pool shared by a recursion tree of requests (tools/server/server-budget-group.h), model-free.
add_executable(test-server-budget-group ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_budget_group.cpp)
target_include_directories(test-server-budget-group PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-budget-group COMMAND $<TARGET_FILE:test-server-budget-group>)
set_property(TEST test-server-budget-group PROPERTY LABELS main)

# Retry-After arithmetic + the model-load failure cooldown (tools/server/server-retry.h), injected clock, model-free.
add_executable(test-server-retry ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_retry.cpp)
target_include_directories(test-server-retry PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-retry COMMAND $<TARGET_FILE:test-server-retry>)
set_property(TEST test-server-retry PROPERTY LABELS main)

# Draining state set by the stop signal (tools/server/server-drain.h), model-free.
add_executable(test-server-drain ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_drain.cpp)
target_include_directories(test-server-drain PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-drain COMMAND $<TARGET_FILE:test-server-drain>)
set_property(TEST test-server-drain PROPERTY LABELS main)
add_executable(test-server-truncate ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_truncate.cpp)
target_include_directories(test-server-truncate PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-truncate COMMAND $<TARGET_FILE:test-server-truncate>)
set_property(TEST test-server-truncate PROPERTY LABELS main)

# Per-client rate limit --rate-limit (tools/server/server-ratelimit.h): fixed window, keys, parse; injected clock, model-free.
add_executable(test-server-ratelimit ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_ratelimit.cpp)
target_include_directories(test-server-ratelimit PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-ratelimit COMMAND $<TARGET_FILE:test-server-ratelimit>)
set_property(TEST test-server-ratelimit PROPERTY LABELS main)

# Two-tier API keys + hot-reloadable key files (tools/server/server-auth.h), model-free. Also checks that the admin route table
# equals the routes tools/server/server.cpp wraps with mgmt(...), so the two cannot drift apart.
add_executable(test-server-auth ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_auth.cpp)
target_include_directories(test-server-auth PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-auth COMMAND $<TARGET_FILE:test-server-auth> ${PROJECT_SOURCE_DIR}/tools/server/server.cpp)
set_property(TEST test-server-auth PROPERTY LABELS main)

# Binary / int8 embedding output (tools/server/server-embd-quant.h), model-free. The test file existed since the 10-15 pick but
# had no target, so nothing ever ran it; registered with the per-token rows pick (row S69-VESPA-PICK-PER-TOKEN-BINARY-EMBEDDING-2026-10-18-01).
add_executable(test-server-embd-quant ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_embd_quant.cpp)
target_include_directories(test-server-embd-quant PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-embd-quant COMMAND $<TARGET_FILE:test-server-embd-quant>)
set_property(TEST test-server-embd-quant PROPERTY LABELS main)

# long_text_mode chunk plan + mean pool (tools/server/server-embd-chunk.h), header-only and model-free
# (row S69-ENGINE-MERGE-GPT4ALL-2026-09-27-01).
add_executable(test-server-embd-chunk ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_embd_chunk.cpp)
target_include_directories(test-server-embd-chunk PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-embd-chunk COMMAND $<TARGET_FILE:test-server-embd-chunk>)
set_property(TEST test-server-embd-chunk PROPERTY LABELS main)

# Operator sampler-override layers: default / force / additive (tools/server/server-overrides.h), model-free; the JSON class
# comes from llama-common.
add_executable(test-server-overrides ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_overrides.cpp)
target_include_directories(test-server-overrides PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
target_link_libraries(test-server-overrides PRIVATE llama-common)
add_test(NAME test-server-overrides COMMAND $<TARGET_FILE:test-server-overrides>)
set_property(TEST test-server-overrides PROPERTY LABELS main)

# Responses compaction (POST /v1/responses/compact + compaction_trigger): tools/server/server-responses-compact.h, model-free.
add_executable(test-server-responses-compact ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_responses_compact.cpp)
target_include_directories(test-server-responses-compact PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
target_link_libraries(test-server-responses-compact PRIVATE llama-common)
add_test(NAME test-server-responses-compact COMMAND $<TARGET_FILE:test-server-responses-compact>)
set_property(TEST test-server-responses-compact PROPERTY LABELS main)

add_executable(test-server-refused-fields ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_refused_fields.cpp)
target_include_directories(test-server-refused-fields PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
target_link_libraries(test-server-refused-fields PRIVATE llama-common)
add_test(NAME test-server-refused-fields COMMAND $<TARGET_FILE:test-server-refused-fields>)
set_property(TEST test-server-refused-fields PROPERTY LABELS main)

# One warning per unknown request key (tools/server/server-unknown-keys.h): unknown-key finder, once-per-key registry with its cap, log
# sanitiser; model-free, the JSON class comes from llama-common. Row S69-ENGREV-TABBYAPI-SAMPLER-ALIASES-AND-UNKNOWN-KEY-WARN-2026-10-27-01.
add_executable(test-server-unknown-keys ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_unknown_keys.cpp)
target_include_directories(test-server-unknown-keys PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
target_link_libraries(test-server-unknown-keys PRIVATE llama-common)
add_test(NAME test-server-unknown-keys COMMAND $<TARGET_FILE:test-server-unknown-keys>)
set_property(TEST test-server-unknown-keys PROPERTY LABELS main)

# Draft length by the number of generating slots (tools/server/server-draft-schedule.h): the schedule parser and lookup, model-free.
add_executable(test-server-draft-schedule ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_draft_schedule.cpp)
target_include_directories(test-server-draft-schedule PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-draft-schedule COMMAND $<TARGET_FILE:test-server-draft-schedule>)
set_property(TEST test-server-draft-schedule PROPERTY LABELS main)

add_executable(test-server-lora-fair ${PROJECT_SOURCE_DIR}/tests/test-server-lora-fair.cpp)
target_include_directories(test-server-lora-fair PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR}/common)
add_test(NAME test-server-lora-fair COMMAND $<TARGET_FILE:test-server-lora-fair>)
set_property(TEST test-server-lora-fair PROPERTY LABELS main)

add_executable(test-server-load-fail ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_load_fail.cpp)
target_include_directories(test-server-load-fail PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-load-fail COMMAND $<TARGET_FILE:test-server-load-fail>)
set_property(TEST test-server-load-fail PROPERTY LABELS main)

# Model-family sampling fallback (common/sampling-family-defaults.h): table sanity, name normaliser and matcher rules, model-free.
add_executable(test-sampling-family-defaults ${PROJECT_SOURCE_DIR}/tests/test-sampling-family-defaults.cpp)
target_include_directories(test-sampling-family-defaults PRIVATE ${PROJECT_SOURCE_DIR}/common)
add_test(NAME test-sampling-family-defaults COMMAND $<TARGET_FILE:test-sampling-family-defaults>)
set_property(TEST test-sampling-family-defaults PROPERTY LABELS main)

# Vision-encoder output cache (tools/mtmd/mtmd-embd-cache.h): the byte-budgeted LRU with request-scope protection, model-free.
add_executable(test-mtmd-embd-cache ${PROJECT_SOURCE_DIR}/tests/test-mtmd-embd-cache.cpp)
target_include_directories(test-mtmd-embd-cache PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
add_test(NAME test-mtmd-embd-cache COMMAND $<TARGET_FILE:test-mtmd-embd-cache>)
set_property(TEST test-mtmd-embd-cache PROPERTY LABELS main)

# Video sampling planner (tools/mtmd/mtmd-video-plan.h): frame cap and vision-token budget, model-free.
add_executable(test-mtmd-video-plan ${PROJECT_SOURCE_DIR}/tests/test-mtmd-video-plan.cpp)
target_include_directories(test-mtmd-video-plan PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
add_test(NAME test-mtmd-video-plan COMMAND $<TARGET_FILE:test-mtmd-video-plan>)
set_property(TEST test-mtmd-video-plan PROPERTY LABELS main)

# In-process webp decode (tools/mtmd/mtmd-helper.cpp, vendor/simplewebp): lossy / lossless / lossy+alpha fixtures decode to
# the Pillow reference with ffmpeg unreachable (row S69-ENGREV-OLLAMA-SHIM-WEBP-IN-PROCESS-DECODE-2026-11-10-01).
add_executable(test-mtmd-webp-decode ${PROJECT_SOURCE_DIR}/tests/test-mtmd-webp-decode.cpp)
target_include_directories(test-mtmd-webp-decode PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
target_link_libraries(test-mtmd-webp-decode PRIVATE mtmd)
target_compile_features(test-mtmd-webp-decode PRIVATE cxx_std_17)
add_test(NAME test-mtmd-webp-decode COMMAND $<TARGET_FILE:test-mtmd-webp-decode> ${PROJECT_SOURCE_DIR}/tests/webp-fixtures)
set_property(TEST test-mtmd-webp-decode PROPERTY LABELS main)

# still images whose header declares more pixels than LLAMA_MTMD_IMAGE_MAX_DECODE_PIXELS are refused before decode
# (row S69-ENGREV-MLXENGINE-IMAGE-PIXEL-CAP-2026-10-27-01); needs no model, reuses the webp fixtures.
add_executable(test-mtmd-image-pixel-cap ${PROJECT_SOURCE_DIR}/tests/test-mtmd-image-pixel-cap.cpp)
target_include_directories(test-mtmd-image-pixel-cap PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
target_link_libraries(test-mtmd-image-pixel-cap PRIVATE mtmd)
target_compile_features(test-mtmd-image-pixel-cap PRIVATE cxx_std_17)
add_test(NAME test-mtmd-image-pixel-cap COMMAND $<TARGET_FILE:test-mtmd-image-pixel-cap> ${PROJECT_SOURCE_DIR}/tests/webp-fixtures)
set_property(TEST test-mtmd-image-pixel-cap PROPERTY LABELS main)

# reasoning_effort -> thinking budget table (tools/server/server-effort.h), model-free.
add_executable(test-server-effort ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_effort.cpp)
target_include_directories(test-server-effort PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-effort COMMAND $<TARGET_FILE:test-server-effort>)
set_property(TEST test-server-effort PROPERTY LABELS main)

# /classify label distribution (tools/server/server-classify.h): softmax / sigmoid and the label fallback, model-free.
add_executable(test-server-classify ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_classify.cpp)
target_include_directories(test-server-classify PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
add_test(NAME test-server-classify COMMAND $<TARGET_FILE:test-server-classify>)
set_property(TEST test-server-classify PROPERTY LABELS main)
# unsl-branches (S69-UNSLOTH-ADOPTABLE-BRANCHES-NOT-YET-PORTED-01): a draft head that borrows the target's
# token_embd / output_norm / output (llama_model_params::model_shared). Built but not add_test()'d: it needs
# the GGUF pair that tests/make_tiny_shared_pair_gguf.py writes. Run:
#   python tests/make_tiny_shared_pair_gguf.py <tree> <dir> && test-model-shared-tensors <dir>
llama_build(test-model-shared-tensors.cpp)

# llama_model_size / llama_model_n_params count a tied head's duplicated token_embd once (S69-FORK-REPORTS-EXTRA-PARAMS-...).
# Same fixtures: python tests/make_tiny_shared_pair_gguf.py <tree> <dir> && test-model-count-tied <dir>
llama_build(test-model-count-tied.cpp)

# unsl-branches: batched row readahead for a gather over an mmap'd table (prefetch/lazy-tensor-row-readahead).
# Part C needs the GGUF from tests/make_tiny_gemma4_ple_gguf.py:  test-mmap-row-prefetch <gemma4_ple.gguf>
llama_build(test-mmap-row-prefetch.cpp)

# Both use internal symbols (llama_mmap, llama_model_ftype_name) that a Windows shared llama.dll does not export.
if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # MinGW defaults _WIN32_WINNT to 0x0601, which compiled out the model-load PrefetchVirtualMemory and the
    # CPU workers' power-throttling opt-out (S69-DECL-OVERTURN-THIREUS-DEFER-PLE-WINVER-2026-10-17-01).
    llama_build_and_test(test-win-version-gates.cpp)
    # every llama_ftype in include/llama.h has a llama_model_ftype_name (Thireus 77eff8798 remainder)
    llama_build_and_test(test-ftype-names.cpp)
    target_compile_definitions(test-ftype-names PRIVATE LLAMA_H_PATH="${PROJECT_SOURCE_DIR}/include/llama.h")
endif()

# unsl-branches: per-run mmap buffers (mmap/per-run-buffers): get_mapping_ranges + load_all_data on a real mapping.
llama_build(test-loader-mapping-ranges.cpp)

# ik_llama 30903876b (row-scaled 4-bit float type 168 + opt-in load of mainline 4-bit float GGUFs): ggml CPU kernels vs a double
# reference, the scale-fold identity, and the loader's type-40 -> type-168 import (default OFF) on a synthetic GGUF.
llama_build_and_test(test-nvfp4-ks.cpp)

# unsl-branches: split GGUF whose first shard is metadata only (S69-GGUF-SHARD1-IS-METADATA-ONLY-NTENSORS-ZERO-01).
# Needs the shards from tests/make_tiny_split_gguf.py:  test-split-metadata-only-shard <dir>
llama_build(test-split-metadata-only-shard.cpp)

# ---------------------------------------------------------------------------------------------
# CRAM software tier (cloud/cram-software-tier). Standalone targets: they compile the two
# self-contained src/llama-cram-*.cpp files directly and do NOT link llama/llama-common, so they
# build without the full library. (The same sources are also listed in src/CMakeLists.txt.)
add_executable(test-cram-probe test-cram-probe.cpp ${PROJECT_SOURCE_DIR}/src/llama-cram-probe.cpp)
target_include_directories(test-cram-probe PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/ggml/include)
add_test(NAME test-cram-probe COMMAND $<TARGET_FILE:test-cram-probe>)
set_property(TEST test-cram-probe PROPERTY LABELS main)

add_executable(test-cram-soft test-cram-soft.cpp ${PROJECT_SOURCE_DIR}/src/llama-cram-soft.cpp)
target_include_directories(test-cram-soft PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/ggml/include)
find_package(Threads REQUIRED)
target_link_libraries(test-cram-soft PRIVATE Threads::Threads)
add_test(NAME test-cram-soft COMMAND $<TARGET_FILE:test-cram-soft>)
set_property(TEST test-cram-soft PROPERTY LABELS main)

add_executable(test-cram-fault test-cram-fault.cpp ${PROJECT_SOURCE_DIR}/src/llama-cram-soft.cpp)
target_include_directories(test-cram-fault PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/ggml/include)
target_link_libraries(test-cram-fault PRIVATE Threads::Threads)
add_test(NAME test-cram-fault COMMAND $<TARGET_FILE:test-cram-fault>)
set_property(TEST test-cram-fault PROPERTY LABELS main)

# S69-KVQUANT-PICK-STATIC-PERCHANNEL-K-POSTROPE-2026-09-29-01: static calibrated per-channel K quantisation container
# (src/llama-kv-kvquant.h). Header-only, CPU, model-free; links ggml only to use the real q4_0 as the comparison baseline.
add_executable(test-kvquant-perchannel-k test-kvquant-perchannel-k.cpp)
target_include_directories(test-kvquant-perchannel-k PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test-kvquant-perchannel-k PRIVATE ggml)
add_test(NAME test-kvquant-perchannel-k COMMAND $<TARGET_FILE:test-kvquant-perchannel-k>)
set_property(TEST test-kvquant-perchannel-k PROPERTY LABELS main)

# S69-FORK-KV-KIVI-2BIT-PORT-01: KIVI-style K-per-channel / V-per-token low-bit container with a full-precision residual
# window (src/llama-kv-kivi.h, from-scratch re-implementation of jy-yuan/KIVI, MIT). Header-only, CPU, model-free.
add_executable(test-kivi-2bit test-kivi-2bit.cpp)
target_include_directories(test-kivi-2bit PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/tests)
target_link_libraries(test-kivi-2bit PRIVATE ggml)
add_test(NAME test-kivi-2bit COMMAND $<TARGET_FILE:test-kivi-2bit>)
set_property(TEST test-kivi-2bit PROPERTY LABELS main)

# ---- ASR decode-quality pure functions (examples/audio/asr-*.cpp) -------------------------------
# Model-free: the sources are compiled straight into each test (tests/ is configured before
# examples/, so the audio-common target does not exist yet here).
find_package(ZLIB QUIET)
function(asr_quality_test name)
    add_executable(${name} ${name}.cpp ${ARGN})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../examples/audio)
    target_compile_features(${name} PRIVATE cxx_std_17)
    if (ZLIB_FOUND)
        target_compile_definitions(${name} PRIVATE ASR_HAVE_ZLIB=1)
        target_link_libraries(${name} PRIVATE ZLIB::ZLIB)
    endif()
    add_test(NAME ${name} COMMAND $<TARGET_FILE:${name}>)
    set_property(TEST ${name} PROPERTY LABELS main)
endfunction()
set(ASR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../examples/audio)
asr_quality_test(test-asr-compression ${ASR_DIR}/asr-compression.cpp)
asr_quality_test(test-asr-fallback ${ASR_DIR}/asr-fallback.cpp ${ASR_DIR}/asr-compression.cpp)
asr_quality_test(test-asr-halluc ${ASR_DIR}/asr-halluc.cpp)
asr_quality_test(test-asr-halluc-span ${ASR_DIR}/asr-halluc.cpp)
asr_quality_test(test-asr-tdt ${ASR_DIR}/asr-tdt.cpp)
target_link_libraries(test-asr-tdt PRIVATE ggml)
asr_quality_test(test-audio-stream ${ASR_DIR}/audio-stream.cpp)
find_package(Threads REQUIRED)
target_link_libraries(test-audio-stream PRIVATE Threads::Threads)
asr_quality_test(test-speak-stream ${ASR_DIR}/speak-stream.cpp ${ASR_DIR}/audio-common.cpp)
asr_quality_test(test-asr-vad-post ${ASR_DIR}/asr-vad-post.cpp)
asr_quality_test(test-asr-dtw ${ASR_DIR}/asr-dtw.cpp)
asr_quality_test(test-asr-align ${ASR_DIR}/asr-align.cpp ${ASR_DIR}/asr-dtw.cpp ${ASR_DIR}/asr-halluc.cpp)
asr_quality_test(test-asr-diarize)
asr_quality_test(test-asr-align-capture ${ASR_DIR}/asr-align-capture.cpp)
target_link_libraries(test-asr-align-capture PRIVATE ggml)
asr_quality_test(test-asr-silero ${ASR_DIR}/asr-silero.cpp ${ASR_DIR}/asr-vad-post.cpp)
target_compile_definitions(test-asr-silero PRIVATE ASR_SILERO_FIXTURES="${ASR_DIR}/tests/fixtures/silero")
set_property(TEST test-asr-silero PROPERTY WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
# llama-server --asr-vad's whole-clip gate, linked from the same asr-vad library the server and llama-audio link
# (examples/audio/asr-vad.cmake; tests/ is configured first, so the include here may be the one that defines it).
include(${ASR_DIR}/asr-vad.cmake)
add_executable(test-asr-vad-gate test-asr-vad-gate.cpp)
target_link_libraries(test-asr-vad-gate PRIVATE asr-vad)
target_compile_features(test-asr-vad-gate PRIVATE cxx_std_17)
target_compile_definitions(test-asr-vad-gate PRIVATE ASR_SILERO_FIXTURES="${ASR_DIR}/tests/fixtures/silero")
add_test(NAME test-asr-vad-gate COMMAND $<TARGET_FILE:test-asr-vad-gate>)
set_property(TEST test-asr-vad-gate PROPERTY LABELS main)
# llama-server --asr-vad-trim: silence trimming from the VAD spans and the time map back to the upload
# (examples/audio/asr-vad-trim.h, in the same asr-vad library). Std-only, no model: the spans are given.
add_executable(test-asr-vad-trim test-asr-vad-trim.cpp)
target_link_libraries(test-asr-vad-trim PRIVATE asr-vad)
target_compile_features(test-asr-vad-trim PRIVATE cxx_std_17)
add_test(NAME test-asr-vad-trim COMMAND $<TARGET_FILE:test-asr-vad-trim>)
set_property(TEST test-asr-vad-trim PROPERTY LABELS main)

# test-asr-silero-cli and test-asr-fallback-cli run the real llama-audio binary. They are registered from
# llama_audio_define_target (examples/audio/CMakeLists.txt), the deferred call that creates llama-audio, because
# that target exists only with LLAMA_BUILD_EXAMPLES=ON and mtmd available; naming it here made every
# LLAMA_BUILD_TESTS=ON + LLAMA_BUILD_EXAMPLES=OFF configure fail at generate time ("No target llama-audio").
add_test(NAME test-asr-tiny-model COMMAND bash -c
    "for py in $PYTHON python3 python python3.13 python3.12 python3.11; do if $py -c 'import numpy' 2>/dev/null; then exec $py ${ASR_DIR}/tests/make_tiny_asr_gguf.py ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/tiny-asr.gguf; fi; done; echo 'no python with numpy'; exit 1")
set_tests_properties(test-asr-tiny-model PROPERTIES FIXTURES_SETUP tiny_asr_gguf LABELS main)

# mtmd audio mel floor (mtmd-audio.cpp compiled in with a stubbed clip_get_hparams; no model)
add_executable(test-mtmd-audio-mel test-mtmd-audio-mel.cpp ${PROJECT_SOURCE_DIR}/tools/mtmd/mtmd-audio.cpp
                                   ${PROJECT_SOURCE_DIR}/tools/mtmd/mtmd-audio-chunks.cpp)
target_include_directories(test-mtmd-audio-mel PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(test-mtmd-audio-mel PRIVATE ggml Threads::Threads)
target_compile_features(test-mtmd-audio-mel PRIVATE cxx_std_17)
add_test(NAME test-mtmd-audio-mel COMMAND $<TARGET_FILE:test-mtmd-audio-mel>)
set_property(TEST test-mtmd-audio-mel PROPERTY LABELS main)

# parakeet preprocess on clips under 320 samples: refuse, never NaN
add_executable(test-mtmd-audio-shortnan test-mtmd-audio-shortnan.cpp ${PROJECT_SOURCE_DIR}/tools/mtmd/mtmd-audio.cpp
                                        ${PROJECT_SOURCE_DIR}/tools/mtmd/mtmd-audio-chunks.cpp)
target_include_directories(test-mtmd-audio-shortnan PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/common)
target_link_libraries(test-mtmd-audio-shortnan PRIVATE ggml Threads::Threads)
target_compile_features(test-mtmd-audio-shortnan PRIVATE cxx_std_17)
add_test(NAME test-mtmd-audio-shortnan COMMAND $<TARGET_FILE:test-mtmd-audio-shortnan>)
set_property(TEST test-mtmd-audio-shortnan PROPERTY LABELS main)

# ASR client prompt is a spelling hint appended to the instruction, never a replacement (std-only)
add_executable(test-asr-prompt-hint test-asr-prompt-hint.cpp)
target_compile_features(test-asr-prompt-hint PRIVATE cxx_std_17)
add_test(NAME test-asr-prompt-hint COMMAND $<TARGET_FILE:test-asr-prompt-hint>)
set_property(TEST test-asr-prompt-hint PROPERTY LABELS main)

# long-audio window planner: cuts in silence instead of at blind 30 s marks (std-only, no model)
add_executable(test-mtmd-audio-chunks test-mtmd-audio-chunks.cpp ${PROJECT_SOURCE_DIR}/tools/mtmd/mtmd-audio-chunks.cpp)
target_include_directories(test-mtmd-audio-chunks PRIVATE ${PROJECT_SOURCE_DIR}/tools/mtmd)
target_compile_features(test-mtmd-audio-chunks PRIVATE cxx_std_17)
add_test(NAME test-mtmd-audio-chunks COMMAND $<TARGET_FILE:test-mtmd-audio-chunks>)
set_property(TEST test-mtmd-audio-chunks PROPERTY LABELS main)

# koboldcpp-port features (std-only; compiled without llama so they build fast)
foreach(kt token-ban prompt-reuse)
    add_executable(test-${kt} test-${kt}.cpp ${PROJECT_SOURCE_DIR}/common/${kt}.cpp)
    target_include_directories(test-${kt} PRIVATE ${PROJECT_SOURCE_DIR}/common)
    target_compile_features(test-${kt} PRIVATE cxx_std_17)
    add_test(NAME test-${kt} COMMAND $<TARGET_FILE:test-${kt}>)
    set_property(TEST test-${kt} PROPERTY LABELS main)
endforeach()

# ---------------------------------------------------------------------------------------------------------------
# cloud/wire-runtime-paths: tests that drive the opt-in runtime paths (flags) on a real llama_context / the real CLI,
# with a random-weight GGUF made by tests/make_tiny_wire_gguf.py (needs python3 + numpy; skipped without them).
set(WIRE_PYTHON "")
# $ENV{PYTHON} first: on Windows "python3" is often the Microsoft Store alias, which runs nothing
foreach(cand IN ITEMS $ENV{PYTHON} python3 /usr/bin/python3 python)
    if (NOT WIRE_PYTHON)
        execute_process(COMMAND ${cand} -c "import numpy" RESULT_VARIABLE _wp_rc OUTPUT_QUIET ERROR_QUIET)
        if (_wp_rc EQUAL 0)
            set(WIRE_PYTHON ${cand})
        endif()
    endif()
endforeach()
if (NOT WIRE_PYTHON)
    message(WARNING "wire tests (test-wire-*) NOT registered: no python with numpy found (set PYTHON)")
endif()
if (WIRE_PYTHON)
    set(WIRE_TINY_DIR ${CMAKE_CURRENT_BINARY_DIR}/wire-tiny)
    add_test(NAME wire-make-tiny-model
        COMMAND ${CMAKE_COMMAND} -E make_directory ${WIRE_TINY_DIR})
    add_test(NAME wire-make-tiny-model-gen
        COMMAND ${WIRE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/make_tiny_wire_gguf.py ${PROJECT_SOURCE_DIR} ${WIRE_TINY_DIR}/tiny.gguf 1 64 0.05)
    set_tests_properties(wire-make-tiny-model PROPERTIES FIXTURES_SETUP wire_dir)
    set_tests_properties(wire-make-tiny-model-gen PROPERTIES FIXTURES_REQUIRED wire_dir FIXTURES_SETUP wire_model)

    llama_build(test-wire-ckpt-cram.cpp)
    add_test(NAME test-wire-ckpt-cram COMMAND $<TARGET_FILE:test-wire-ckpt-cram> ${WIRE_TINY_DIR}/tiny.gguf)
    set_tests_properties(test-wire-ckpt-cram PROPERTIES FIXTURES_REQUIRED wire_model LABELS main)
endif()
if (WIRE_PYTHON)
    llama_build(test-wire-runtime-e2e.cpp)
    add_test(NAME test-wire-runtime-e2e COMMAND $<TARGET_FILE:test-wire-runtime-e2e> ${WIRE_TINY_DIR}/tiny.gguf)
    set_tests_properties(test-wire-runtime-e2e PROPERTIES FIXTURES_REQUIRED wire_model LABELS main)
endif()
if (WIRE_PYTHON)
    add_test(NAME wire-make-tiny-model-v1024
        COMMAND ${WIRE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/make_tiny_wire_gguf.py ${PROJECT_SOURCE_DIR} ${WIRE_TINY_DIR}/tiny_v1024.gguf 2 1024 0.05)
    set_tests_properties(wire-make-tiny-model-v1024 PROPERTIES FIXTURES_REQUIRED wire_dir FIXTURES_SETUP wire_model)
    llama_build(test-wire-prefilter-e2e.cpp)
    add_test(NAME test-wire-prefilter-e2e COMMAND $<TARGET_FILE:test-wire-prefilter-e2e> ${WIRE_TINY_DIR}/tiny.gguf ${WIRE_TINY_DIR}/tiny_v1024.gguf)
    set_tests_properties(test-wire-prefilter-e2e PROPERTIES FIXTURES_REQUIRED wire_model LABELS main)
endif()
# Heap-corruption regression (turbo vec_dot scratch sized 2*n, needs 4*n): --old-size must FAIL.
llama_build_and_test(test-turbo-vecdot-scratch.cpp)
llama_test(test-turbo-vecdot-scratch NAME test-turbo-vecdot-scratch-oldsize ARGS --old-size)
set_tests_properties(test-turbo-vecdot-scratch-oldsize PROPERTIES WILL_FAIL TRUE)
if (WIRE_PYTHON AND LLAMA_BUILD_TOOLS AND LLAMA_BUILD_SERVER AND NOT EMSCRIPTEN)
    add_test(NAME test-wire-cli COMMAND ${WIRE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/test-wire-cli.py $<TARGET_FILE:llama-cli> ${WIRE_TINY_DIR}/tiny.gguf)
    set_tests_properties(test-wire-cli PROPERTIES FIXTURES_REQUIRED wire_model LABELS main TIMEOUT 600)
endif()

# tiny-logits: loads a GGUF through the real llama library, decodes a fixed token sequence in ubatch chunks and dumps the logits. Driven by
# test-wiring-tiny.py (MX quant types end to end, KIVI KV flag on/off). Row S69-MXFP6-MXFP8-INTEGRATION-FTYPE-GGUFPY-SIMD-MSVC-BUILD-2026-10-08-01.
llama_build(tiny-logits.cpp)

# --kivi-bits (simulated KIVI KV quantisation) through the real decode path, checked on the cache contents. Needs a model: registered in
# test-wiring-tiny.py's flow (tests/make_tiny_mx_gguf.py writes one). Row S69-FORK-KV-KIVI-2BIT-PORT-01.
llama_build(test-kivi-sim-cache.cpp)

# qwen4exp PLE conv-history state bytes: layout map (fork-packed kv_self row vs the memory module's separate p_l row), state_seq_get_size == the
# live rows, save -> restore and seq_cp logits. Needs a qwen4exp GGUF: test-llama-archs -a qwen4exp -o <dir> writes a random-weight one
# (wiring only). Row S69-ENGREV-UPSTREAM-MERGE-PLE-STATE-TWO-LAYOUTS-2026-11-04-01.
llama_build(test-ple-state-bytes.cpp)

# Prism Hadamard, one source of truth: llama_hdmd_fill makes model.hdmd (class-path graphs) from the data the fork loader uses for
# hadamard_map (legacy builder): matrices equal the legacy ggml_hadamard element-wise, rot*inv = I, signs equal. `--no-fill` runs the
# same checks on the pre-change state and must FAIL. Row S69-ENGREV-UPSTREAM-MERGE-PRISM-HADAMARD-TWO-IMPLEMENTATIONS-2026-11-04-01.
llama_build_and_test(test-prism-hdmd-fill.cpp)

# End-to-end wiring on a random-weight GGUF: MX ftypes through llama-quantize + a real forward pass, and --kivi-bits on/off. Needs numpy for gguf-py
# (uses WIRE_PYTHON from the wire block above: $PYTHON first, then python3/python; ${Python3_EXECUTABLE} may lack numpy).
if (WIRE_PYTHON)
    add_test(NAME test-wiring-tiny
             COMMAND ${WIRE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/test-wiring-tiny.py ${PROJECT_SOURCE_DIR} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set_property(TEST test-wiring-tiny PROPERTY LABELS main)
    # glm5-next two implementations (fork GLM5NEXT graph vs merged-upstream GLM5_NEXT builder, same random-weight GGUF, LLAMA_GLM5NEXT_IMPL=upstream, CPU):
    # logits within float noise, same argmax and greedy tokens, plus two must-fail controls. Row S69-ENGREV-UPSTREAM-MERGE-GLM5-NEXT-TWO-IMPLEMENTATIONS-2026-11-04-01.
    add_test(NAME test-glm5next-two-impl-parity
             COMMAND ${WIRE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/test-glm5next-two-impl-parity.py ${PROJECT_SOURCE_DIR} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set_property(TEST test-glm5next-two-impl-parity PROPERTY LABELS main)
    set_tests_properties(test-glm5next-two-impl-parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
    # Recurrent ops through the real rwkv6 graph (GGML_OP_RWKV_WKV6): batched vs token-by-token state carry + kernel digest. Row S69-GGML-OP-3-RWKV-WKV6-WKV7.
    add_test(NAME test-wiring-rwkv6
             COMMAND ${WIRE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/test-wiring-rwkv6.py ${PROJECT_SOURCE_DIR} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set_property(TEST test-wiring-rwkv6 PROPERTY LABELS main)
    # end to end on random-weight GGUFs with must-fail controls (needs the verify-backends / quant-sensitivity binaries built)
    add_test(NAME test-verify-backends-e2e
             COMMAND ${WIRE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/verify-backends.py ${PROJECT_SOURCE_DIR} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set_property(TEST test-verify-backends-e2e PROPERTY LABELS behavior)
    set_tests_properties(test-verify-backends-e2e PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
    add_test(NAME test-quant-sensitivity-e2e COMMAND ${WIRE_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/quant-sensitivity.py ${PROJECT_SOURCE_DIR} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set_property(TEST test-quant-sensitivity-e2e PROPERTY LABELS behavior)
    set_tests_properties(test-quant-sensitivity-e2e PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
else()
    message(WARNING "test-wiring-tiny / test-wiring-rwkv6 NOT registered: no python with numpy found (set PYTHON)")
endif()

# bench-mx-vec-dot: scalar vs CPU-backend (AVX2) MXFP6/MXFP8 dot, repeated-run mean +- sd. Not a test.
llama_build(bench-mx-vec-dot.cpp)
target_include_directories(bench-mx-vec-dot PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)

# ---------------------------------------------------------------------------------------------
# VRAM spill policy (cloud/link-vram-spill-1014). Standalone: compiles the self-contained policy
# source directly (the same file is also in ggml-base), links no library.
add_executable(test-vram-spill test-vram-spill.cpp ${PROJECT_SOURCE_DIR}/ggml/src/ggml-vram-spill.cpp)
target_include_directories(test-vram-spill PRIVATE ${PROJECT_SOURCE_DIR}/ggml/include)
add_test(NAME test-vram-spill COMMAND $<TARGET_FILE:test-vram-spill>)
set_property(TEST test-vram-spill PROPERTY LABELS main)
llama_build_and_test(test-vram-spill-cli.cpp)

# Vendor-neutral free-GPU-memory probes (tools/server/server-vram-probe.h) and the dispatch in server-vram.cpp, fixture-driven, model-free.
# server-vram.cpp is compiled in directly (it only needs the logging of llama-common); the probes are dlopen()/LoadLibrary()d, so no GPU
# library is linked.
add_executable(test-server-vram-probe ${PROJECT_SOURCE_DIR}/tools/server/tests_cpp/test_server_vram_probe.cpp ${PROJECT_SOURCE_DIR}/tools/server/server-vram.cpp)
target_include_directories(test-server-vram-probe PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
target_link_libraries(test-server-vram-probe PRIVATE llama-common ${CMAKE_DL_LIBS})
add_test(NAME test-server-vram-probe COMMAND $<TARGET_FILE:test-server-vram-probe>)
set_property(TEST test-server-vram-probe PROPERTY LABELS main)

add_executable(test-moe-cache-shares test-moe-cache-shares.cpp)
target_include_directories(test-moe-cache-shares PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test-moe-cache-shares COMMAND $<TARGET_FILE:test-moe-cache-shares>)

# ClassLRU policy, the CPU share's plan() and MUL_MAT_ID's negative ids on the CPU backend, no model and no GPU (upstream #31233, #30966)
llama_build_and_test(test-moe-cache-classlru.cpp  LABEL behavior)
target_include_directories(test-moe-cache-classlru PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Bonsai (Q1_0_G128 / Q2_0_G128) scalar vec_dot vs the activation layout the CPU traits hand it, plus graph-level mul_mat in the default
# and the GGML_NO_IQK_GEMM=1 (forced scalar) modes. Links `ggml` alone (it calls the traits and ggml_graph_compute only).
add_executable(test-bonsai-scalar-vecdot test-bonsai-scalar-vecdot.cpp)
target_link_libraries(test-bonsai-scalar-vecdot PRIVATE ggml)
add_test(NAME test-bonsai-scalar-vecdot COMMAND $<TARGET_FILE:test-bonsai-scalar-vecdot>)
set_property(TEST test-bonsai-scalar-vecdot PROPERTY LABELS main)
add_test(NAME test-bonsai-scalar-vecdot-noiqk COMMAND $<TARGET_FILE:test-bonsai-scalar-vecdot>)
set_property(TEST test-bonsai-scalar-vecdot-noiqk PROPERTY LABELS main)
set_property(TEST test-bonsai-scalar-vecdot-noiqk PROPERTY ENVIRONMENT "GGML_NO_IQK_GEMM=1")

# S69-CONTRIB-PICK-SM-GRAPH-LOAD-HOST-RAM-BUDGET-2026-10-18-01: host-RAM budget for parallel tensor loads
# (src/llama-load-budget.h, header-only; the loader's -sm graph path uses it). Model-free, no CUDA, no llama link.
add_executable(test-load-budget test-load-budget.cpp)
target_include_directories(test-load-budget PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/ggml/src)
target_compile_features(test-load-budget PRIVATE cxx_std_17)
target_link_libraries(test-load-budget PRIVATE Threads::Threads)
add_test(NAME test-load-budget COMMAND $<TARGET_FILE:test-load-budget>)
set_property(TEST test-load-budget PROPERTY LABELS main)

# S69-OV-PICK-AMX-BF16-ATTENTION-DECODE-GEMV-AND-PREFILL-2026-10-06-01: AMX-BF16 attention kernel
# (ggml/src/ggml-cpu/amx/attn-bf16.h, header-only) vs a double reference. Built with -mamx-tile -mamx-bf16 whatever the
# host; on a CPU without AMX it exits 77 (SKIP) after CPUID. Real run: `sde -spr -- test-amx-attn-bf16` (Intel SDE).
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mamx-tile -mamx-bf16" LLAMA_TEST_CXX_HAS_AMX_BF16)
if (LLAMA_TEST_CXX_HAS_AMX_BF16)
    add_executable(test-amx-attn-bf16 test-amx-attn-bf16.cpp)
    target_include_directories(test-amx-attn-bf16 PRIVATE ${PROJECT_SOURCE_DIR}/ggml/include)
    target_compile_features(test-amx-attn-bf16 PRIVATE cxx_std_17)
    target_compile_options(test-amx-attn-bf16 PRIVATE -mamx-tile -mamx-bf16)
    add_test(NAME test-amx-attn-bf16 COMMAND $<TARGET_FILE:test-amx-attn-bf16>)
    set_property(TEST test-amx-attn-bf16 PROPERTY LABELS main)
    set_property(TEST test-amx-attn-bf16 PROPERTY SKIP_RETURN_CODE 77)
endif()

# Same row, wiring half: FLASH_ATTN_EXT with bf16 K/V through the CPU backend vs a double reference. Passes on every
# build; in a GGML_AMX_TILE + GGML_AMX_BF16 build with GGML_AMX_ATTN=1 the AMX path takes it (stderr says FIRED).
add_executable(test-amx-attn-ggml test-amx-attn-ggml.cpp)
target_compile_features(test-amx-attn-ggml PRIVATE cxx_std_17)
target_link_libraries(test-amx-attn-ggml PRIVATE ggml)
add_test(NAME test-amx-attn-ggml COMMAND $<TARGET_FILE:test-amx-attn-ggml>)
set_property(TEST test-amx-attn-ggml PROPERTY LABELS main)

# The CPU device's total/free through the public API (ggml-cpu.cpp ggml_backend_cpu_device_get_memory). With
# GGML_TEST_MEM_LIMIT=<bytes> it also checks total/free against a container memory limit (Linux cgroup v2 path).
add_executable(test-cpu-device-memory test-cpu-device-memory.cpp)
target_link_libraries(test-cpu-device-memory PRIVATE ggml)
add_test(NAME test-cpu-device-memory COMMAND $<TARGET_FILE:test-cpu-device-memory>)
set_property(TEST test-cpu-device-memory PROPERTY LABELS main)

# Stack-depth probe for the JSON-schema depth cap (row S69-ENGREV-UPSTREAM-MERGE-SCHEMA-DEPTH-CAP-MEASURE-2026-11-04-01): compiles the two
# schema sources itself with the cap raised so the real stack use past 128 levels can be measured; driven by tests/probe-schema-depth.py.
llama_build(probe-schema-depth.cpp ${PROJECT_SOURCE_DIR}/common/json-schema.cpp ${PROJECT_SOURCE_DIR}/common/json-schema-to-grammar.cpp ${PROJECT_SOURCE_DIR}/common/jinja/value.cpp)
target_compile_definitions(probe-schema-depth PRIVATE COMMON_CHAT_SCHEMA_MAX_DEPTH_VALUE=100000 COMMON_JINJA_FROM_JSON_MAX_DEPTH=100000)

# GGML_OP_COL2IM_3D (row S69-FORK-COL2IM-3D-SCATTER-ADD-OP-2026-11-15-01): CPU scatter-add adjoint of IM2COL_3D, checked against a
# naive reference and by the adjoint identity <im2col(x), c> == <x, col2im(c)>.
add_executable(test-col2im-3d test-col2im-3d.cpp)
target_compile_features(test-col2im-3d PRIVATE cxx_std_17)
target_link_libraries(test-col2im-3d PRIVATE ggml)
add_test(NAME test-col2im-3d COMMAND $<TARGET_FILE:test-col2im-3d>)
set_property(TEST test-col2im-3d PROPERTY LABELS "main behavior backend")
