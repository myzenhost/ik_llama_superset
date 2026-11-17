# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# examples

if (EMSCRIPTEN)
else()
    add_subdirectory(batched)
    add_subdirectory(cparams-wiring-check)
    add_subdirectory(snapkv-rebase-ab)   # SnapKV/H2O survivor re-base A/B (S69-F1-PICK-SNAPKV-CONTIGUOUS-REBASE-AB-2026-10-06-01)
    add_subdirectory(debug)
    add_subdirectory(embedding)
    add_subdirectory(eval-callback)

    add_subdirectory(gguf-hash)
    add_subdirectory(gguf)
    add_subdirectory(idle)
    add_subdirectory(lookahead)
    add_subdirectory(lookup)
    add_subdirectory(parallel)
    add_subdirectory(passkey)
    add_subdirectory(retrieval)
    add_subdirectory(simple)

    # ik_llama fork examples. Upstream moved 13 of this fork's examples/ entries into
    # tools/ (and renamed main -> cli), so most of what a symbol diff calls "fork-only"
    # here is simply relocated and must NOT be re-added. These seven are the ones that
    # are genuinely this fork's and still have a CMakeLists.txt on disk -- verified before
    # being written, because add_subdirectory on a directory without one fails the
    # configure rather than warning.
    #
    # Deliberately NOT entered, though a directory exists for each: baby-llama, benchmark,
    # gritlm, infill, main, save-load-state (upstream deleted their CMakeLists) and
    # gbnf-validator (gone entirely).
    add_subdirectory(audio)
    add_subdirectory(legacy-state-check)   # state save/restore must restore what DECODES (kv_self on legacy models)
    add_subdirectory(kv-dedup-check)       # the memory module of a legacy model holds no K/V (metadata only); decode unchanged
    add_subdirectory(diffusion-gemma-eval)
    add_subdirectory(diffusion-gemma-server)
    add_subdirectory(offload-attn-test)
    add_subdirectory(offload-attn-recall)   # attention-mass recall of the offload-attn page selector vs oracle/random, on real q/K (--selftest = no model)
    add_subdirectory(quantize-stats)
    add_subdirectory(spec-bench)
    add_subdirectory(sweep-bench)

    add_subdirectory(simple-chat)
    add_subdirectory(speculative)
    add_subdirectory(speculative-simple)
    add_subdirectory(gen-docs)
    add_subdirectory(training)
    add_subdirectory(diffusion)
    if (NOT GGML_BACKEND_DL)
        add_subdirectory(convert-llama2c-to-ggml)
        # these examples use the backends directly and cannot be built with dynamic loading
        if (GGML_SYCL)
            add_subdirectory(sycl)
        endif()
    endif()
endif()
