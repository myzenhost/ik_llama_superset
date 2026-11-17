cmake_minimum_required(VERSION 3.14...3.28) # for add_link_options and implicit target directories.

project("ggml" C CXX ASM)

### GGML Version
set(GGML_VERSION_MAJOR 0)
set(GGML_VERSION_MINOR 27)
set(GGML_VERSION_PATCH 0)
set(GGML_VERSION_BASE "${GGML_VERSION_MAJOR}.${GGML_VERSION_MINOR}.${GGML_VERSION_PATCH}")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

find_program(GIT_EXE NAMES git git.exe NO_CMAKE_FIND_ROOT_PATH)
if(GIT_EXE)
    # Get current git commit hash
    execute_process(COMMAND ${GIT_EXE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE GGML_BUILD_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    # Check if the working directory is dirty (i.e., has uncommitted changes)
    execute_process(COMMAND ${GIT_EXE} diff-index --quiet HEAD -- .
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE GGML_GIT_DIRTY
        ERROR_QUIET
    )
endif()

set(GGML_VERSION "${GGML_VERSION_BASE}")

if(NOT GGML_BUILD_COMMIT)
    set(GGML_BUILD_COMMIT "unknown")
endif()

# Build the commit string with optional dirty flag
if(DEFINED GGML_GIT_DIRTY AND GGML_GIT_DIRTY EQUAL 1)
    set(GGML_BUILD_COMMIT "${GGML_BUILD_COMMIT}-dirty")
endif()

include(CheckIncludeFileCXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(GGML_STANDALONE ON)

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    # configure project version
    # TODO
else()
    set(GGML_STANDALONE OFF)

    if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    endif()
endif()

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    option(GGML_WASM_SINGLE_FILE "ggml: embed WASM inside the generated ggml.js" ON)
    # WIRED 2026-09-08, same defect and same fix as LLAMA_WASM_SINGLE_FILE in ../CMakeLists.txt:
    # declared at f3f65429c and never read in any tree, so the description was unenforced.
    # -sSINGLE_FILE=1 base64-embeds the .wasm into the emitted .js.
    # NOTE ON THE ASYMMETRY, kept deliberately: this defaults ON while the llama-side option
    # defaults OFF. Those are upstream's own defaults and they govern different subtrees
    # (add_link_options is directory-scoped, and ggml is also built standalone), so they are left
    # as upstream wrote them rather than coupled into one. Unreachable outside EMSCRIPTEN.
    if (GGML_WASM_SINGLE_FILE)
        add_link_options("-sSINGLE_FILE=1")
    endif()
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()

# remove the lib prefix on win32 mingw
if (WIN32)
    set(CMAKE_STATIC_LIBRARY_PREFIX "")
    set(CMAKE_SHARED_LIBRARY_PREFIX "")
    set(CMAKE_SHARED_MODULE_PREFIX  "")
endif()

option(BUILD_SHARED_LIBS           "ggml: build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})
option(GGML_BACKEND_DL             "ggml: build backends as dynamic libraries (requires BUILD_SHARED_LIBS)" OFF)
set(GGML_BACKEND_DIR "" CACHE PATH "ggml: directory to load dynamic backends from (requires GGML_BACKEND_DL")

#
# option list
#

# TODO: mark all options as advanced when not GGML_STANDALONE

if (APPLE)
    set(GGML_METAL_DEFAULT ON)
    set(GGML_BLAS_DEFAULT ON)
    set(GGML_BLAS_VENDOR_DEFAULT "Apple")
else()
    set(GGML_METAL_DEFAULT OFF)
    set(GGML_BLAS_DEFAULT OFF)
    set(GGML_BLAS_VENDOR_DEFAULT "Generic")
endif()

if (CMAKE_CROSSCOMPILING OR DEFINED ENV{SOURCE_DATE_EPOCH})
    message(STATUS "Setting GGML_NATIVE_DEFAULT to OFF")
    set(GGML_NATIVE_DEFAULT OFF)
else()
    set(GGML_NATIVE_DEFAULT ON)
endif()

# defaults
if (NOT GGML_LLAMAFILE_DEFAULT)
    set(GGML_LLAMAFILE_DEFAULT OFF)
endif()

if (NOT GGML_CUDA_GRAPHS_DEFAULT)
    set(GGML_CUDA_GRAPHS_DEFAULT OFF)
endif()

# general
option(GGML_STATIC "ggml: static link libraries"                     OFF)
option(GGML_NATIVE "ggml: optimize the build for the current system" ${GGML_NATIVE_DEFAULT})
option(GGML_LTO    "ggml: enable link time optimization"             OFF)
option(GGML_CCACHE "ggml: use ccache if available"                   ON)

# debug
option(GGML_ALL_WARNINGS           "ggml: enable all compiler warnings"                   ON)
option(GGML_ALL_WARNINGS_3RD_PARTY "ggml: enable all compiler warnings in 3rd party libs" OFF)
option(GGML_GPROF                  "ggml: enable gprof"                                   OFF)

# GGML_ALL_WARNINGS_3RD_PARTY: KEPT, AND DELIBERATELY *NOT* FORWARDED. MEASURED 2026-09-17.
# The R1 lint flags it "declared but no reader". That is true here -- and equally true
# UPSTREAM, which is what decides the fix: it is not a fork decoy, and there is no live twin
# to forward it to.
#     portable/llama-cpp-turboquant  (current mainline llama.cpp layout)
#         ggml/CMakeLists.txt:126      option(GGML_ALL_WARNINGS_3RD_PARTY ...)  <- declared
#         ggml/src/CMakeLists.txt:37                                            <- reads
#         ggml/src/ggml-cuda/CMakeLists.txt:206                                    GGML_ALL_
#                                                                                  WARNINGS
#                                                                                  ONLY
#         CMakeLists.txt:93            option(LLAMA_ALL_WARNINGS_3RD_PARTY ...) <- also unread
#     portable/ik_llama_src/ggml/CMakeLists.txt:66-67   same pair, same asymmetry
# Forwarding it to GGML_ALL_WARNINGS -- the twin a name-similarity sweep proposes -- would be
# wrong twice over. The two options name OPPOSITE scopes: warnings in VENDORED 3rd-party code
# versus warnings in ggml's own sources. And GGML_ALL_WARNINGS already defaults ON, so the
# forward would be a semantic error that ALSO does nothing. Implementing it properly means
# suppressing/enabling warnings across vendor/ and the fetched KleidiAI tree, which is a
# change to files outside this one.
# THIS GUARD IS THE READER THE LINT ASKS FOR. IT IS NOT THE FEATURE. A user who sets the
# option still does not get warnings in 3rd-party libs -- they get told so, which is strictly
# better than today's silence, and strictly worse than the option working.
if (GGML_ALL_WARNINGS_3RD_PARTY)
    message(WARNING
        "GGML_ALL_WARNINGS_3RD_PARTY is declared for upstream parity but is NOT IMPLEMENTED "
        "in ggml; upstream llama.cpp does not implement it either. Setting it does not change "
        "how any 3rd-party source is compiled. GGML_ALL_WARNINGS (default ON) governs ggml's "
        "own sources.")
endif()

# build
option(GGML_FATAL_WARNINGS    "ggml: enable -Werror flag"    OFF)

# sanitizers
option(GGML_SANITIZE_THREAD    "ggml: enable thread sanitizer"    OFF)
option(GGML_SANITIZE_ADDRESS   "ggml: enable address sanitizer"   OFF)
option(GGML_SANITIZE_UNDEFINED "ggml: enable undefined sanitizer" OFF)

# instruction set specific
if (GGML_NATIVE OR NOT GGML_NATIVE_DEFAULT)
    set(INS_ENB OFF)
else()
    set(INS_ENB ON)
endif()

message(DEBUG "GGML_NATIVE         : ${GGML_NATIVE}")
message(DEBUG "GGML_NATIVE_DEFAULT : ${GGML_NATIVE_DEFAULT}")
message(DEBUG "INS_ENB             : ${INS_ENB}")

option(GGML_CPU_HBM          "ggml: use memkind for CPU HBM" OFF)
option(GGML_CPU_REPACK       "ggml: use runtime weight conversion of Q4_0 to Q4_X_X" ON)
option(GGML_CPU_BASE_QUANTS_ISA "ggml: compile ggml-base's ggml-quants.c with the CPU backend's ISA flags (x86, single CPU backend only)" ON)
option(GGML_CPU_KLEIDIAI     "ggml: use KleidiAI optimized kernels if applicable" OFF)
option(GGML_SSE42            "ggml: enable SSE 4.2"          ${INS_ENB})
option(GGML_AVX              "ggml: enable AVX"              ${INS_ENB})
option(GGML_AVX_VNNI         "ggml: enable AVX-VNNI"         OFF)
option(GGML_AVX2             "ggml: enable AVX2"             ${INS_ENB})
option(GGML_BMI2             "ggml: enable BMI2"             ${INS_ENB})
option(GGML_AVX512           "ggml: enable AVX512F"          OFF)
option(GGML_AVX512_VBMI      "ggml: enable AVX512-VBMI"      OFF)
option(GGML_AVX512_VNNI      "ggml: enable AVX512-VNNI"      OFF)
option(GGML_AVX512_BF16      "ggml: enable AVX512-BF16"      OFF)
if (NOT MSVC)
    # in MSVC F16C and FMA is implied with AVX2/AVX512
    option(GGML_FMA          "ggml: enable FMA"              ${INS_ENB})
    option(GGML_F16C         "ggml: enable F16C"             ${INS_ENB})
    # MSVC does not seem to support AMX
    option(GGML_AMX_TILE     "ggml: enable AMX-TILE"         OFF)
    option(GGML_AMX_INT8     "ggml: enable AMX-INT8"         OFF)
    option(GGML_AMX_BF16     "ggml: enable AMX-BF16"         OFF)
endif()
option(GGML_LASX             "ggml: enable lasx"              ON)
option(GGML_LSX              "ggml: enable lsx"               ON)
option(GGML_RVV              "ggml: enable rvv"               ON)
option(GGML_RV_ZFH           "ggml: enable riscv zfh"         ON)
option(GGML_RV_ZVFH          "ggml: enable riscv zvfh"        ON)
option(GGML_RV_ZICBOP        "ggml: enable riscv zicbop"      ON)
option(GGML_RV_ZIHINTPAUSE   "ggml: enable riscv zihintpause" ON)
option(GGML_RV_ZVFBFWMA      "ggml: enable riscv zvfbfwma"    OFF)
option(GGML_XTHEADVECTOR     "ggml: enable xtheadvector"      OFF)
option(GGML_VXE              "ggml: enable vxe"               ${GGML_NATIVE})

option(GGML_CPU_ALL_VARIANTS "ggml: build all variants of the CPU backend (requires GGML_BACKEND_DL)" OFF)
set(GGML_CPU_ARM_ARCH        "" CACHE STRING "ggml: CPU architecture for ARM")
set(GGML_CPU_POWERPC_CPUTYPE "" CACHE STRING "ggml: CPU type for PowerPC")

# ggml core
set(GGML_SCHED_MAX_COPIES  "4" CACHE STRING "ggml: max input copies for pipeline parallelism")
option(GGML_CPU                             "ggml: enable CPU backend"                        ON)
option(GGML_SCHED_NO_REALLOC                "ggml: disallow reallocations in ggml-alloc (for debugging)" OFF)

# 3rd party libs / backends
option(GGML_ACCELERATE                      "ggml: enable Accelerate framework"               ON)
option(GGML_BLAS                            "ggml: use BLAS"                                  ${GGML_BLAS_DEFAULT})
set(GGML_BLAS_VENDOR ${GGML_BLAS_VENDOR_DEFAULT} CACHE STRING
                                            "ggml: BLAS library vendor")
option(GGML_LLAMAFILE                       "ggml: use LLAMAFILE"                             ${GGML_LLAMAFILE_DEFAULT})

option(GGML_CUDA                            "ggml: use CUDA"                                  OFF)
set   (GGML_CUDA_CCCL_VERSION "" CACHE STRING "ggml: CCCL git tag to fetch, empty to use the version bundled with the installed CUDA Toolkit")
option(GGML_MUSA                            "ggml: use MUSA"                                  OFF)
option(GGML_CUDA_FORCE_MMQ                  "ggml: use mmq kernels instead of cuBLAS"         OFF)
option(GGML_CUDA_FORCE_CUBLAS               "ggml: always use cuBLAS instead of mmq kernels"  OFF)
option(GGML_CUDA_NO_PEER_COPY               "ggml: do not use peer to peer copies"            OFF)
option(GGML_CUDA_NO_VMM                     "ggml: do not try to use CUDA VMM"                OFF)
option(GGML_CUDA_FA                         "ggml: compile ggml FlashAttention CUDA kernels"  ON)
option(GGML_CUDA_FA_ALL_QUANTS              "ggml: compile all quants for FlashAttention"     OFF)
# Default = exactly what the old default glob built (the fork's quantised KV set). Upstream's default also lists bf16-bf16; the vec kernels
# here have no bf16, so a bf16 pair is skipped with a warning (ggml/cmake/common.cmake) and is left out of the default.
set   (GGML_CUDA_FA_QUANTS_FORK_DEFAULT "q4_0-q4_0;q8_0-q8_0;f16-f16;q8_0-iq4_nl;iq4_nl-iq4_nl;q6_0-q5_0;q8_0-q6_0")
set   (GGML_CUDA_FA_QUANTS "${GGML_CUDA_FA_QUANTS_FORK_DEFAULT}" CACHE STRING
                                            "ggml: FlashAttention K-V type combinations to compile, \"all\" or a list such as \"q8_0-q8_0;q8_0-q4_0\" (f16-f16 is always built)")
option(GGML_CUDA_GRAPHS                     "ggml: use CUDA graphs (llama.cpp only)"          ${GGML_CUDA_GRAPHS_DEFAULT})
option(GGML_CUDA_NCCL                       "ggml: use NVIDIA Collective Comm. Library"       ON)

# --- kept from this fork ------------------------------------------------------------------
# HISTORICAL HEADING, PRESERVED AND MARKED REFUTED 2026-09-17. This block used to open:
#     "every name below is READ by the LIVE monolith -- ggml/src/CMakeLists.txt is a
#      single-library build that never enters ggml/src/ggml-*/, so these are the spellings
#      that actually take effect."
# BOTH HALVES OF THAT ARE NOW FALSE, and the premise was the entire justification for keeping
# the fork spellings below as bare option() lines.
#   * REFUTED BY ggml/src/CMakeLists.txt:576 -- `add_subdirectory(${backend_target})`, inside
#     `function(ggml_add_backend backend)` (:572-583), invoked 18 times at :629 and :840-856.
#     That file is upstream's DISPATCHER, not a monolith, and entering ggml/src/ggml-*/ is
#     precisely what it exists to do. The monolith it describes survives only as the
#     non-building backups ggml/src/CMakeLists.txt.pre-zt (80,082 B) and .pre-small (77,943 B).
#   * The "LIVE n" counts below were measured against THAT monolith. They are provenance, not
#     current state, and are kept verbatim for that reason -- do not re-use them as evidence.
# A SECOND, SEPARATE TRAP, worth stating because it is what makes these names hard to audit:
# the dispatcher reads backend options through a CONSTRUCTED name --
# `string(TOUPPER "GGML_${backend}" backend_id)` then `if (${backend_id})` (:573-574), plus
# `set(GGML_${feat} ...)` (:594/:598) and `set(GGML_INTERNAL_${feat} ON)` (:602). No
# word-bounded text search can see those, so "no reader found" from such a sweep is a claim
# about the SWEEP unless the constructed spellings have been enumerated too. They have been,
# for this block: the 18 ggml_add_backend() arguments and the x86/ARM feature lists produce
# GGML_AVX_VNNI and GGML_INTERNAL_SVE -- never GGML_AVXVNNI, never GGML_SVE.
option(GGML_AVXVNNI         "ggml: enable AVX-VNNI (fork spelling; LIVE 2)"        OFF)
option(GGML_SVE             "ggml: enable SVE (LIVE 1)"                            OFF)
option(GGML_NCCL            "ggml: enable NCCL (fork spelling; LIVE 1)"            ON)
option(GGML_EXPERT_CHUNKING "ggml: enable expert chunking (LIVE 2, CODE 2)"        ON)
option(GGML_IQK_MUL_MAT     "ggml: use optimized iqk matrix multiplications (LIVE 1)" ON)

# GGML_AVXVNNI -> GGML_AVX_VNNI. FORWARDED. MEASURED 2026-09-17: no behaviour claim here
# beyond flag equivalence, which is read off the two files named below.
# What the monolith did with it, and therefore what this must reproduce:
#     ggml/src/CMakeLists.txt.pre-zt:1240  (MSVC/AVX2 branch) -mavxvnni for clang-cl, else
#                                          add_compile_definitions(__AVXVNNI__)
#     ggml/src/CMakeLists.txt.pre-zt:1270  (non-MSVC branch)  -mavxvnni
# The dispatcher's equivalent is GGML_AVX_VNNI, read at
#     ggml/src/ggml-cpu/CMakeLists.txt:372-373  __AVXVNNI__ + GGML_AVX_VNNI (MSVC)
#     ggml/src/ggml-cpu/CMakeLists.txt:407-409  -mavxvnni  + GGML_AVX_VNNI (non-MSVC)
# and consumed at ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp:293. Same flag, same macro, so the
# forward genuinely restores what the old name did -- this one is not merely a lint-silencer.
# A PLAIN set(), not CACHE ... FORCE: ggml/src and ggml/src/ggml-cpu inherit this directory's
# scope, so the normal variable reaches every reader above, and the ^GGML_ capture at the
# bottom of this file (get_cmake_property(... VARIABLES), which runs later) still exports it.
# CACHE FORCE would permanently stomp a user's GGML_AVX_VNNI=OFF and keep stomping it on every
# later reconfigure, long after they stopped passing the old name.
# RESIDUAL, stated rather than hidden: under GGML_CPU_ALL_VARIANTS the per-variant helper at
# ggml/src/CMakeLists.txt:589-599 resets GGML_${feat} OFF and re-enables only the features
# named for that variant, so this value is overridden per variant BY DESIGN. The forward is
# effective on the single-variant path (the default), not against GGML_CPU_ALL_VARIANTS.
if (GGML_AVXVNNI)
    message(WARNING
        "GGML_AVXVNNI is this fork's pre-merge spelling and has no reader left; forwarding to "
        "GGML_AVX_VNNI. Set -DGGML_AVX_VNNI=ON directly. (Ignored per-variant under "
        "GGML_CPU_ALL_VARIANTS, which selects features itself.)")
    set(GGML_AVX_VNNI ON)
endif()

# GGML_SVE -> GGML_CPU_ARM_ARCH. FORWARDED, BUT ONLY PARTLY, AND ON PURPOSE.
# MEASURED 2026-09-17. This is the one where a uniform `set(TWIN ON)` would be WRONG: GGML_SVE
# is a BOOL and GGML_CPU_ARM_ARCH (declared above as a CACHE STRING) is an -march STRING.
# set(GGML_CPU_ARM_ARCH ON) would emit `-march=ON`.
# What the monolith actually did -- ggml/src/CMakeLists.txt.pre-zt:1199 --
#     if (GGML_SVE)
#         list(APPEND ARCH_FLAGS -march=armv8.6-a+sve)
# i.e. it never selected any SVE CODE; it only changed the -march string. That matches the
# maintainer's own statement in github-data/issues/539: "no usage is made of SVE anywhere in
# ik_llama.cpp. The only ARM implementation that exists is NEON."
# The dispatcher has THREE live routes to SVE, none of them spelled GGML_SVE:
#   1. GGML_NATIVE=ON (the default): ggml/src/ggml-cpu/CMakeLists.txt:239 auto-probes SVE with
#      check_arm_feature(sve SVE ...) and adds +sve when the machine really has it. Strictly
#      better than the old option, and needs nothing from the user.
#   2. GGML_NATIVE=OFF: ggml/src/ggml-cpu/CMakeLists.txt:244 appends -march=${GGML_CPU_ARM_ARCH}.
#   3. GGML_CPU_ALL_VARIANTS: GGML_INTERNAL_SVE, set per variant at ggml/src/CMakeLists.txt:602.
# The forward below reproduces route 2 -- the monolith's exact flag -- AND NOTHING ELSE.
# WHAT A USER STILL DOES NOT GET, SAID PLAINLY: route 2 sits in the `else()` of
# `if (GGML_NATIVE)`, and GGML_NATIVE DEFAULTS ON. With the default this forward is inert and
# the auto-probe (route 1) decides. The warning text says so. Deliberately NOT flipping
# GGML_NATIVE here: that is a global switch governing x86 -march=native and INS_ENB (line ~151)
# as well, and turning it off silently to satisfy an ARM-only legacy flag would trade one quiet
# behaviour change for a far larger one.
# Harmless off-ARM: GGML_CPU_ARM_ARCH is only read inside
# `if (GGML_SYSTEM_ARCH STREQUAL "ARM")`. Not set when the user already set it, and
# ggml/src/CMakeLists.txt:634 still raises its own FATAL_ERROR if it collides with
# GGML_CPU_ALL_VARIANTS.
if (GGML_SVE)
    if (GGML_CPU_ARM_ARCH)
        message(WARNING
            "GGML_SVE has no reader and is being IGNORED because GGML_CPU_ARM_ARCH is already "
            "set to '${GGML_CPU_ARM_ARCH}'. Add +sve there if you want it.")
    else()
        message(WARNING
            "GGML_SVE is this fork's pre-merge spelling and has no reader left. It never "
            "selected SVE code -- it only appended -march=armv8.6-a+sve. Forwarding to "
            "GGML_CPU_ARM_ARCH=armv8.6-a+sve. NOTE: that path is taken only with "
            "-DGGML_NATIVE=OFF, which is NOT the default; with GGML_NATIVE=ON (default) ggml "
            "auto-detects SVE and this setting does nothing.")
        set(GGML_CPU_ARM_ARCH "armv8.6-a+sve")
    endif()
endif()

if (WIN32)
    # Default to Windows 10 (0x0A00). Windows 8 / 8.1 reached end of support in January 2023,
    # and the vendored cpp-httplib used by the server target rejects anything below 0x0A00.
    # Override with -DGGML_WIN_VER=0x602 to target Windows 8. (LIVE 1)
    set(GGML_WIN_VER "0x0A00" CACHE STRING "ggml: Windows Version")
endif()

# ggml core, fork values. NOTE GGML_SCHED_MAX_COPIES: this fork sets 1 where upstream sets 4.
# Upstream's value is set above; the fork's is NOT re-applied here, because that is a
# pipeline-parallelism memory tradeoff rather than a merge question and deciding it silently
# inside a merge would bury it. Tracked on the build-architecture row.
set(GGML_MAX_CONTEXTS "" CACHE STRING
    "ggml: max model contexts (override only; defaults to 64 in the code) (LIVE 3, CODE 4)")

# CUDA options this fork's monolith reads. Upstream renamed GGML_CUDA_USE_GRAPHS to
# GGML_CUDA_GRAPHS and bridges it in ggml/src/ggml-cuda/CMakeLists.txt:134-135 -- a file this
# build never enters -- so the fork spelling is the one that reaches the .cu sources here.
# Both are declared: the upstream name above costs nothing, this one does the work.
option(GGML_CUDA_FORCE_DMMV      "ggml: use dmmv instead of mmvq CUDA kernels (LIVE 4)"   OFF)
option(GGML_CUDA_IQK_FORCE_BF16  "ggml: use bf16 cuBLAS when no MMQ kernel exists (LIVE 2)" OFF)
set   (GGML_CUDA_DMMV_X "32" CACHE STRING "ggml: x stride for dmmv CUDA kernels (LIVE 4)")
set   (GGML_CUDA_MMV_Y   "1" CACHE STRING "ggml: y block size for mmv CUDA kernels (LIVE 5)")
option(GGML_CUDA_F16             "ggml: use 16 bit floats for some calculations (LIVE 3)" OFF)
set   (GGML_CUDA_KQUANTS_ITER "2" CACHE STRING
                                 "ggml: iters./thread per block for Q2_K/Q6_K (LIVE 2)")
set   (GGML_CUDA_MIN_BATCH_OFFLOAD "32" CACHE STRING
                                 "ggml: min batch size for GPU offload (LIVE 2)")
option(GGML_CUDA_USE_GRAPHS      "ggml: use CUDA graphs (fork spelling; LIVE 2, CODE 10)" ON)
set   (GGML_CUDA_FUSION "1" CACHE STRING "ggml: enable/disable fusion (LIVE 2)")
# -----------------------------------------------------------------------------------------
set   (GGML_CUDA_COMPRESSION_MODE "size" CACHE STRING
                                            "ggml: cuda link binary compression mode; requires cuda 12.8+")
set_property(CACHE GGML_CUDA_COMPRESSION_MODE PROPERTY STRINGS "none;speed;balance;size")

option(GGML_HIP                             "ggml: use HIP"                                   OFF)
option(GGML_HIP_GRAPHS                      "ggml: use HIP graph"                              ON)
option(GGML_HIP_RCCL                        "ggml: use ROCm Collective Comm. Library"         OFF)
option(GGML_HIP_NO_VMM                      "ggml: do not try to use HIP VMM"                 ON)
option(GGML_HIP_MMQ_MFMA                    "ggml: enable MFMA MMA for CDNA in MMQ"           ON)
option(GGML_HIP_EXPORT_METRICS              "ggml: enable kernel perf metrics output"         OFF)
option(GGML_MUSA_GRAPHS                     "ggml: use MUSA graph, experimental, unstable"    OFF)
option(GGML_MUSA_MUDNN_COPY                 "ggml: enable muDNN for accelerated copy"         OFF)

# --- kept from this fork ------------------------------------------------------------------
# The IQK FlashAttention CPU kernels are fork-only and heavily read: LIVE 2 each, with 12 and
# 8 source readers respectively. Upstream has no equivalent option.
option(GGML_IQK_FLASH_ATTENTION "ggml: enable the IQK FlashAttention CPU kernels" ON)
option(GGML_IQK_FA_ALL_QUANTS   "ggml: compile all quants for IQK FlashAttention" ON)

# GGML_CURL: REMOVED 2026-09-17. History kept here, per the note it replaces.
# The previous comment read: "GGML_CURL has no reader under ggml/src. It is kept anyway
# because that sweep was SCOPED to ggml/, so 'no reader' here is not a statement about common/
# or tools/. One dead option line is cheaper than a silently-unset one; drop it only after a
# tree-wide sweep with a control." That condition has now been met, so the option goes.
# THE TREE-WIDE SWEEP: across the whole fork, GGML_CURL occurred ONLY on its own option() line
# and in this comment -- no reader AND no setter anywhere, in any file type. Reach of that run
# is not in question: the same search returned 30+ LLAMA_CURL hits spread over .bat,
# .Dockerfile, .Containerfile, docs/build.md, ci/ and the root CMakeLists.txt, so a sibling
# spelling served as the positive control inside the run that reported the zero.
# FORWARDING TO LLAMA_CURL WAS CONSIDERED AND REJECTED -- and not on layering taste, but
# because it does not work. LLAMA_CURL is declared in the ROOT CMakeLists.txt and consumed by
# common/ and tools/, which are SIBLING directories added from the root. A `set(LLAMA_CURL ON)`
# written here is a ggml-directory-scoped normal variable those siblings never observe. The
# only escape is `set(... CACHE BOOL "" FORCE)`, which would then also fire when ggml is built
# STANDALONE (GGML_STANDALONE, line ~51) -- a configuration in which LLAMA_CURL means nothing
# and there is no llama target to enable.
# Every in-tree caller already uses the right name (-DLLAMA_CURL=..., 30+ call sites above).
# Anyone still passing -DGGML_CURL now gets CMake's own "Manually-specified variables were not
# used by the project" notice, which is louder than the silence this line used to provide.

# The fork's older HIP spelling. The claim it used to carry -- "still what the monolith
# reads" -- is REFUTED: the monolith is gone (see the REFUTED heading above), and its reader,
# ggml/src/CMakeLists.txt.pre-zt:434 `if (GGML_HIPBLAS)`, now survives only in a backup file
# that nothing builds. Upstream's GGML_HIP is declared above and is what the dispatcher reads.
option(GGML_HIPBLAS "ggml: use hipBLAS (fork spelling; LIVE 1)"                    OFF)
option(GGML_HIP_UMA "ggml: use HIP unified memory architecture (LIVE 2, CODE 1)"   OFF)

# GGML_HIPBLAS -> GGML_HIP. FORWARDED. MEASURED 2026-09-17.
# This is the strongest of the five: the old name is still live DOCUMENTATION inside this
# repository, so leaving it inert is an active trap rather than dead weight --
#     .devops/full-rocm.Dockerfile:39, .devops/llama-cli-rocm.Dockerfile:39 and
#     .devops/llama-server-rocm.Dockerfile:39 all carry `ENV GGML_HIPBLAS=1`
#     github-data/discussions/562 documents `-DGGML_HIPBLAS=ON` as the way to build for ROCm
# (Honesty about that evidence: those three Dockerfiles then invoke `make -j$(nproc)`, and the
# root Makefile is a 266-byte hard `$(error Build system changed ... replaced by CMake)` stub,
# so that particular path already fails for its own unrelated reason. They are stale setters,
# not proof of a live CMake reader. They do prove the spelling is still published to users.)
# Where GGML_HIP is actually read, and why a plain set() suffices:
#     ggml/src/CMakeLists.txt:844  ggml_add_backend(HIP), which at :573-574 builds the name
#                                  `GGML_HIP` with string(TOUPPER "GGML_${backend}") and tests
#                                  `if (${backend_id})` -- a CONSTRUCTED name, invisible to any
#                                  word-bounded text search, which is exactly why this option
#                                  looked twinless. :580 then derives the GGML_USE_HIP macro.
#     ggml/cmake/ggml-config.cmake.in:88  `if (GGML_HIP)` in the exported package config,
#                                  populated by the ^GGML_ variable capture further down this
#                                  file -- which runs AFTER this line, so the forward is
#                                  exported too.
# ggml/src inherits this directory's scope, so a normal variable reaches both. NOT
# CACHE ... FORCE: that would permanently overwrite a user's cached GGML_HIP=OFF and keep
# overwriting it on every later reconfigure once they stop passing the old name.
# A user setting GGML_HIPBLAS now genuinely gets the HIP backend built, not just a quieter lint.
if (GGML_HIPBLAS)
    message(WARNING
        "GGML_HIPBLAS is this fork's pre-merge spelling and has no reader left; forwarding to "
        "GGML_HIP. Set -DGGML_HIP=ON directly.")
    set(GGML_HIP ON)
endif()
# -----------------------------------------------------------------------------------------
option(GGML_VULKAN                          "ggml: use Vulkan"                                OFF)
option(GGML_VULKAN_CHECK_RESULTS            "ggml: run Vulkan op checks"                      OFF)
option(GGML_VULKAN_DEBUG                    "ggml: enable Vulkan debug output"                OFF)
option(GGML_VULKAN_MEMORY_DEBUG             "ggml: enable Vulkan memory debug output"         OFF)
option(GGML_VULKAN_SHADER_DEBUG_INFO        "ggml: enable Vulkan shader debug info"           OFF)
option(GGML_VULKAN_VALIDATE                 "ggml: enable Vulkan validation"                  OFF)
option(GGML_VULKAN_RUN_TESTS                "ggml: run Vulkan tests"                          OFF)
option(GGML_WEBGPU                          "ggml: use WebGPU"                                OFF)
option(GGML_WEBGPU_DEBUG                    "ggml: enable WebGPU debug output"                OFF)
option(GGML_WEBGPU_CPU_PROFILE              "ggml: enable WebGPU profiling (CPU)"             OFF)
option(GGML_WEBGPU_GPU_PROFILE              "ggml: enable WebGPU profiling (GPU)"             OFF)
option(GGML_WEBGPU_JSPI                     "ggml: use JSPI for WebGPU"                       ON)
option(GGML_ZDNN                            "ggml: use zDNN"                                  OFF)
option(GGML_VIRTGPU                         "ggml: use the VirtGPU/Virglrenderer API Remoting frontend"     OFF)
option(GGML_VIRTGPU_BACKEND                 "ggml: build the VirtGPU/Virglrenderer API Remoting backend"    OFF)

# --- kept from this fork: the Vulkan opt-outs ---------------------------------------------
# These matter on this hardware specifically: an RTX 5090 takes the coopmat branch, where the
# Vulkan MMQ path is known-dead, so GGML_VULKAN_NO_COOPMAT is the lever for that. Upstream has
# no equivalent switches.
#
# CORRECTED 2026-09-08. The comment here previously claimed GGML_VULKAN_NO_COOPMAT was "a working
# lever and not dead weight", and the annotation "each LIVE 1 in the monolith" described a build
# that no longer exists: the merge replaced this fork's monolithic ggml/src/CMakeLists.txt with
# upstream's split-backend layout, so whatever read these in the monolith is gone. MEASURED after
# the merge: a whole-repository word-bounded search for each of the four returned ONLY its own
# option() line. They were NO-OPS -- setting any of them changed nothing at all -- while the
# comment asserted the opposite, which is worse than no comment. They are wired for real below,
# in ggml/src/ggml-vulkan/CMakeLists.txt, where the glslc feature tests they must override live.
option(GGML_VULKAN_NO_COOPMAT  "ggml: don't use Vulkan coopmat even if supported"      OFF)
option(GGML_VULKAN_NO_COOPMAT2 "ggml: don't use Vulkan coopmat2 even if supported"     OFF)
option(GGML_VULKAN_NO_BF16     "ggml: don't use Vulkan bf16 even if supported"         OFF)
option(GGML_VULKAN_NO_INT_DOT  "ggml: don't use Vulkan integer dot even if supported"  OFF)
# -----------------------------------------------------------------------------------------
option(GGML_METAL                           "ggml: use Metal"                                 ${GGML_METAL_DEFAULT})
option(GGML_METAL_NDEBUG                    "ggml: disable Metal debugging"                   OFF)
option(GGML_METAL_SHADER_DEBUG              "ggml: compile Metal with -fno-fast-math"         OFF)
option(GGML_METAL_EMBED_LIBRARY             "ggml: embed Metal library"                       ${GGML_METAL})
set   (GGML_METAL_MACOSX_VERSION_MIN "" CACHE STRING
                                            "ggml: metal minimum macOS version")
set   (GGML_METAL_STD "" CACHE STRING       "ggml: metal standard version (-std flag)")
# MinGW + GCC defaults GGML_OPENMP to OFF. Its OpenMP runtime is libgomp's mingw-w64 port, whose barrier has
# NO spin phase: bar.o calls pthread_mutex_* and sem_*, never gomp_spin_count_var, so every ggml_barrier
# (#pragma omp barrier, once per graph node) is a Windows kernel semaphore sleep/wake per thread, and no
# environment variable changes it (GOMP_SPINCOUNT / OMP_WAIT_POLICY are compiled out of that path; KMP_BLOCKTIME
# is LLVM-only). Upstream ggml-org/llama.cpp#26200 documents it (+40% decode with OFF, 4 cores; closed stale,
# not fixed). MEASURED here 2026-09-26, Threadripper 3990X, SmolLM2-135M Q8_0 decode, same source, only this
# option differing: 1 thread 72.7 vs 72.7 t/s, 4 threads 32.2 -> 118.7, 16 threads 8.4 -> 123.5 (14.6x); gdb
# showed 15 of 16 workers blocked in gomp_team_barrier_wait -> pthread_mutex_lock -> WaitForSingleObject.
# Greedy output identical; granite-h-350m PPL identical (8.5043) and batch 23% faster (0.67 -> 0.52 s/pass).
# OFF uses ggml's own threadpool and spin barrier (ggml-cpu.c non-OpenMP path). An existing CMakeCache keeps
# whatever it already holds -- reconfigure with -DGGML_OPENMP=OFF.
if (MINGW AND CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(GGML_OPENMP_DEFAULT OFF)
else()
    set(GGML_OPENMP_DEFAULT ON)
endif()
option(GGML_OPENMP                          "ggml: use OpenMP"                                ${GGML_OPENMP_DEFAULT})
set   (GGML_METAL_TARGET_OS "macos" CACHE STRING
                                            "ggml: metal -mtargetos OS name (macos, ios, xros, tvos)")
option(GGML_OPENMP_FETCH                    "ggml: fetch LLVM OpenMP"                         OFF)
if (GGML_OPENMP AND NOT GGML_OPENMP_FETCH AND MINGW AND CMAKE_C_COMPILER_ID STREQUAL "GNU")
    message(WARNING "GGML_OPENMP=ON with MinGW GCC uses libgomp's mingw-w64 barrier, which cannot spin: CPU "
                    "decode measured 14.6x slower at 16 threads on this project's host. Use -DGGML_OPENMP=OFF "
                    "(see ggml/CMakeLists.txt and upstream ggml-org/llama.cpp#26200).")
endif()
option(GGML_RPC                             "ggml: use RPC"                                   OFF)
option(GGML_SYCL                            "ggml: use SYCL"                                  OFF)
option(GGML_SYCL_F16                        "ggml: use 16 bit floats for sycl calculations"   OFF)
option(GGML_SYCL_GRAPH                      "ggml: enable graphs in the SYCL backend"         ON)
option(GGML_SYCL_HOST_MEM_FALLBACK          "ggml: allow host memory fallback in SYCL reorder (requires kernel 6.8+)" ON)
option(GGML_SYCL_SUPPORT_LEVEL_ZERO_API     "ggml: use Level Zero API in SYCL backend"  ON)
option(GGML_SYCL_DNN                        "ggml: enable oneDNN in the SYCL backend"         ON)
set   (GGML_SYCL_TARGET "INTEL" CACHE STRING
                                            "ggml: sycl target device")
set   (GGML_SYCL_DEVICE_ARCH "" CACHE STRING
                                            "ggml: sycl device architecture")

option(GGML_OPENVINO                        "ggml: use OPENVINO"                              OFF)
option(GGML_ET                              "ggml: use ET backend"                            OFF)
option(GGML_ET_SYSEMU                       "ggml: use ET backend via sysemu"                 OFF)

option(GGML_OPENCL                          "ggml: use OpenCL"                                OFF)
option(GGML_OPENCL_PROFILING                "ggml: use OpenCL profiling (increases overhead)" OFF)
option(GGML_OPENCL_EMBED_KERNELS            "ggml: embed kernels"                             ON)
option(GGML_OPENCL_USE_ADRENO_KERNELS       "ggml: use optimized kernels for Adreno"          ON)
set   (GGML_OPENCL_TARGET_VERSION "300" CACHE STRING
                                            "ggml: OpenCL API version to target")

option(GGML_HEXAGON                         "ggml: enable Hexagon backend"                    OFF)

# toolchain for vulkan-shaders-gen
set   (GGML_VULKAN_SHADERS_GEN_TOOLCHAIN "" CACHE FILEPATH "ggml: toolchain file for vulkan-shaders-gen")

option(GGML_ZENDNN                          "ggml: use ZenDNN"                                OFF)
option(ZENDNN_ROOT                          "ggml: path to ZenDNN installation"               "")

# extra artifacts
option(GGML_BUILD_TESTS    "ggml: build tests"    ${GGML_STANDALONE})
option(GGML_BUILD_EXAMPLES "ggml: build examples" ${GGML_STANDALONE})

#
# dependencies
#

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)

include(GNUInstallDirs)

#
# build the library
#

add_subdirectory(src)

#
# tests and examples
#

if (GGML_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

if (GGML_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

#
# install
#

include(CMakePackageConfigHelpers)

# all public headers
set(GGML_PUBLIC_HEADERS
    include/ggml.h
    include/ggml-cpu.h
    include/ggml-alloc.h
    include/ggml-backend.h
    include/ggml-blas.h
    include/ggml-cann.h
    include/ggml-cpp.h
    include/ggml-cuda.h
    include/ggml-opt.h
    include/ggml-metal.h
    include/ggml-rpc.h
    include/ggml-virtgpu.h
    include/ggml-sycl.h
    include/ggml-vulkan.h
    include/ggml-webgpu.h
    include/ggml-zendnn.h
    include/ggml-openvino.h
    include/gguf.h)

set_target_properties(ggml PROPERTIES PUBLIC_HEADER "${GGML_PUBLIC_HEADERS}")
install(TARGETS ggml LIBRARY PUBLIC_HEADER)
# RESTORED 2026-09-08. This was dropped while ggml/src/CMakeLists.txt was still this fork's
# MONOLITH, in which ggml-base was never created (measured: created=False, mentioned 0 times) and
# an unguarded reference would have failed the configure outright. That was a WORKAROUND for a
# missing target, not a decision about installing. ggml/src/CMakeLists.txt is now upstream's
# dispatcher and creates ggml-base, so the workaround is obsolete and the real statement is back.
install(TARGETS ggml-base LIBRARY)

# Kept from this fork. ggml/src/ggml-metal.metal is present here, and upstream only dropped
# this install because it embeds the metal library instead.
if (GGML_METAL)
    install(
        FILES src/ggml-metal.metal
        PERMISSIONS
            OWNER_READ
            OWNER_WRITE
            GROUP_READ
            WORLD_READ
        DESTINATION ${CMAKE_INSTALL_BINDIR})

    if (NOT GGML_METAL_EMBED_LIBRARY)
        install(
            FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
            DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()
endif()

if (GGML_STANDALONE)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ggml.pc.in
        ${CMAKE_CURRENT_BINARY_DIR}/ggml.pc
        @ONLY)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ggml.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()

#
# Create CMake package
#



# Capture variables prefixed with GGML_.

set(variable_set_statements
"
####### Expanded from @GGML_VARIABLES_EXPANED@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run        #######

")

set(GGML_SHARED_LIB ${BUILD_SHARED_LIBS})

get_cmake_property(all_variables VARIABLES)
foreach(variable_name IN LISTS all_variables)
    if(variable_name MATCHES "^GGML_")
        string(REPLACE ";" "\\;"
               variable_value "${${variable_name}}")

        set(variable_set_statements
            "${variable_set_statements}set(${variable_name} \"${variable_value}\")\n")
    endif()
endforeach()

set(GGML_VARIABLES_EXPANDED ${variable_set_statements})

# Create the CMake package and set install location.

set(GGML_INSTALL_VERSION ${GGML_VERSION})
set(GGML_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(GGML_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(GGML_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ggml-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/ggml-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ggml
    PATH_VARS GGML_INCLUDE_INSTALL_DIR
              GGML_LIB_INSTALL_DIR
              GGML_BIN_INSTALL_DIR)

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/ggml-config-version.cmake
    VERSION ${GGML_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion)

message(STATUS "ggml version: ${GGML_INSTALL_VERSION}")
message(STATUS "ggml commit:  ${GGML_BUILD_COMMIT}")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ggml-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/ggml-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ggml)

if (MSVC)
    set(MSVC_WARNING_FLAGS
        /wd4005  # Macro redefinition
        /wd4244  # Conversion from one type to another type, possible loss of data
        /wd4267  # Conversion from 'size_t' to a smaller type, possible loss of data
        /wd4305  # Conversion from 'type1' to 'type2', possible loss of data
        /wd4566  # Conversion from 'char' to 'wchar_t', possible loss of data
        /wd4996  # Disable POSIX deprecation warnings
        /wd4702  # Unreachable code warnings
    )
    set(MSVC_COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:C>:/utf-8>"
        "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
    )
    function(configure_msvc_target target_name)
        if(TARGET ${target_name})
            target_compile_options(${target_name} PRIVATE ${MSVC_WARNING_FLAGS})
            target_compile_options(${target_name} PRIVATE ${MSVC_COMPILE_OPTIONS})
        endif()
    endfunction()

    configure_msvc_target(ggml-base)
    configure_msvc_target(ggml)
    configure_msvc_target(ggml-cpu)
    configure_msvc_target(ggml-cpu-x64)
    configure_msvc_target(ggml-cpu-sse42)
    configure_msvc_target(ggml-cpu-sandybridge)
    # __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512
    # skipping            ggml-cpu-ivybridge
    # skipping            ggml-cpu-piledriver
    configure_msvc_target(ggml-cpu-haswell)
    configure_msvc_target(ggml-cpu-skylakex)
    configure_msvc_target(ggml-cpu-cannonlake)
    configure_msvc_target(ggml-cpu-cascadelake)
    configure_msvc_target(ggml-cpu-icelake)
    # MSVC 2022 doesn't support BF16 intrinsics without `/arch:AVX10.1` ?!
    # https://learn.microsoft.com/en-us/cpp/intrinsics/x64-amd64-intrinsics-list?view=msvc-170
    # https://learn.microsoft.com/en-us/cpp/build/reference/arch-x64?view=msvc-170
    # skipping            ggml-cpu-cooperlake
    # skipping            ggml-cpu-zen4
    configure_msvc_target(ggml-cpu-alderlake)
    # MSVC doesn't support AMX
    # skipping            ggml-cpu-sapphirerapids

    if (GGML_BUILD_EXAMPLES)
        configure_msvc_target(common-ggml)
        configure_msvc_target(common)

        configure_msvc_target(mnist-common)
        configure_msvc_target(mnist-eval)
        configure_msvc_target(mnist-train)

        configure_msvc_target(gpt-2-ctx)
        configure_msvc_target(gpt-2-alloc)
        configure_msvc_target(gpt-2-backend)
        configure_msvc_target(gpt-2-sched)
        configure_msvc_target(gpt-2-quantize)
        configure_msvc_target(gpt-2-batched)

        configure_msvc_target(gpt-j)
        configure_msvc_target(gpt-j-quantize)

        configure_msvc_target(magika)
        configure_msvc_target(yolov3-tiny)
        configure_msvc_target(sam)

        configure_msvc_target(simple-ctx)
        configure_msvc_target(simple-backend)
    endif()

    if (GGML_BUILD_TESTS)
        configure_msvc_target(test-mul-mat)
        configure_msvc_target(test-arange)
        configure_msvc_target(test-backend-ops)
        configure_msvc_target(test-cont)
        configure_msvc_target(test-conv-transpose)
        configure_msvc_target(test-conv-transpose-1d)
        configure_msvc_target(test-conv1d)
        configure_msvc_target(test-conv2d)
        configure_msvc_target(test-conv2d-dw)
        configure_msvc_target(test-customop)
        configure_msvc_target(test-dup)
        configure_msvc_target(test-opt)
        configure_msvc_target(test-pool)
    endif ()
endif()
